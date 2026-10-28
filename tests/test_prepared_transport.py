"""Prepared property selection, provenance and wall-film equivalence."""

import math

import pytest

from chemical_properties import ChemicalDatabase
from perry_properties import PerryPropertyLibrary
from property_resolver import PropertyResolver
from property_resolution.common import PropertyResolutionResult
from thermodynamics import create_thermodynamics
from thermodynamics_models.common import ThermodynamicsError


def test_prepared_conductivity_selects_ranges_and_fallback_at_each_temperature(monkeypatch):
    resolver = PropertyResolver()
    library = PerryPropertyLibrary.__new__(PerryPropertyLibrary)
    library._load_thermal_conductivity = lambda: None
    library.get = lambda *args, **kwargs: {'cas': 'fixture'}
    library._source = lambda table: 'conductivity range fixture'
    library.saturated_liquid_thermal_conductivity_W_per_m_K = lambda *args: None
    def row(low, high, value):
        return {'equation_id': 100, 'coefficients': [value, 0., 0., 0., 0.],
                'T_min_K': low, 'T_max_K': high, 'source_table': 'fixture'}
    library.thermal_conductivity_chemicals = {
        'fixture': {'liquid_thermal_conductivity': [row(250., 500., .6), row(320., 340., .7)]}}
    monkeypatch.setattr(resolver, '_get_perry_library', lambda: library)
    monkeypatch.setattr(resolver, '_identifier_candidates', lambda *args: ['fixture'])
    monkeypatch.setattr(resolver, '_estimated_liquid_thermal_conductivity',
        lambda *args, **kwargs: PropertyResolutionResult(.2, 'estimated', 'outside_range', .7))
    kernel = resolver.resolve_thermal_conductivity_kernel('fixture', 'liquid', {}, allow_online=False)
    assert kernel.evaluate(300.).value == .6
    assert kernel.evaluate(330.).value == .7
    assert kernel.evaluate(450.).value == .6
    fallback = kernel.evaluate(550.)
    assert fallback.value == .2
    assert fallback.method == 'outside_range'
    assert fallback.quality == .7
    assert kernel.evaluate(330.).method == 'perry_liquid_thermal_conductivity_eq100'


def test_prepared_provided_conductivity_is_a_snapshot_and_has_priority(monkeypatch):
    resolver = PropertyResolver()
    props = {'property_correlations': {'kl': {
        'equation': 'dippr_eq100', 'coefficients': {'A': .9, 'B': -.001},
        'Tmin_K': 250., 'Tmax_K': 400., '_pfd_override': True, 'quality': .99}}}
    monkeypatch.setattr(resolver, '_get_perry_library',
                        lambda: pytest.fail('Provided conductivity must not consult Perry'))
    kernel = resolver.resolve_thermal_conductivity_kernel('fixture', 'liquid', props, allow_online=False)
    props['property_correlations']['kl']['coefficients']['A'] = 100.
    result = kernel.evaluate(300.)
    assert result.value == pytest.approx(.6)
    assert result.method == 'provided_liquid_thermal_conductivity_dippr_eq100'


@pytest.mark.parametrize('method,composition,phase,pressure,temperatures', [
    ('IDEAL', {'water': 1.}, 'liquid', 3., [350., 300., 330., 350.]),
    ('IDEAL', {'water': .8, 'ethanol': .2}, 'liquid', 3., [300., 330.]),
    ('NRTL', {'water': .8, 'ethanol': .2}, 'liquid', 3., [300., 330.]),
    ('STEAM', {'water': 1.}, 'liquid', 3., [360., 300., 330., 360.]),
    ('STEAM', {'water': 1.}, 'vapor', 1., [500., 450., 550., 500.]),
    ('PR', {'CH4': .5, 'C2H6': .5}, 'vapor', 10., [400., 450.]),
])
def test_prepared_wall_prandtl_matches_full_property_bundle(method, composition, phase, pressure, temperatures):
    thermo = create_thermodynamics(list(composition), method, ChemicalDatabase(enable_online=False))
    evaluate = thermo.prepare_wall_prandtl(composition, pressure, phase)
    vf = float(phase == 'vapor')
    for T in temperatures:
        state = thermo.calculate_state(T, pressure, 1., composition, phase=phase, flash=False, include=('rho', 'Cp'))
        conductivity = thermo.transport_mixture_thermal_conductivity(composition, T, pressure, vf)
        k = conductivity.vapor if vf else conductivity.liquid
        expected = state.Cp*1000/thermo.mixture_MW(composition)*thermo.mixture_viscosity(composition, T, pressure, vf)/k
        assert evaluate(T) == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ThermodynamicsError, match='temperature must be positive and finite'):
        evaluate(math.nan)


def test_prepared_steam_honors_custom_viscosity(monkeypatch):
    thermo = create_thermodynamics(['water'], 'STEAM', ChemicalDatabase(enable_online=False))
    monkeypatch.setattr(thermo, 'mixture_viscosity', lambda *args: .00123)
    evaluate = thermo.prepare_wall_prandtl({'water': 1.}, 3., 'liquid')
    state = thermo.calculate_state(330., 3., 1., {'water': 1.}, phase='liquid', include=('Cp',))
    k = thermo.transport_mixture_thermal_conductivity({'water': 1.}, 330., 3., 0.).liquid
    assert evaluate(330.) == pytest.approx(state.Cp*1000/thermo.mixture_MW({'water': 1.})*.00123/k)


def test_joint_steam_queries_preserve_a_custom_scalar_backend(monkeypatch):
    thermo = create_thermodynamics(['water'], 'STEAM', ChemicalDatabase(enable_online=False))
    original = thermo._props_si
    def scalar(output, *args):
        assert isinstance(output, str)
        value = original(output, *args)
        return value*1.02 if output in ('C', 'Cpmass') else value
    monkeypatch.setattr(thermo, '_props_si', scalar)
    evaluate = thermo.prepare_wall_prandtl({'water': 1.}, 3., 'liquid')
    for T in (360., 300., 330.):
        state = thermo.calculate_state(T, 3., 1., {'water': 1.}, phase='liquid', include=('Cp',))
        mu = thermo.mixture_viscosity({'water': 1.}, T, 3., 0.)
        k = thermo.transport_mixture_thermal_conductivity({'water': 1.}, T, 3., 0.).liquid
        assert evaluate(T) == pytest.approx(state.Cp*1000/thermo.mixture_MW({'water': 1.})*mu/k)


@pytest.mark.parametrize('phase,quality', [('liquid', 0.), ('vapor', 1.)])
def test_prepared_steam_wall_properties_match_saturated_phase(phase, quality):
    thermo = create_thermodynamics(['water'], 'STEAM', ChemicalDatabase(enable_online=False))
    composition, pressure = {'water': 1.}, 3.
    state = thermo.calculate_state_PQ(pressure, quality, 1., composition, include=('Cp',))
    evaluate = thermo.prepare_wall_prandtl(composition, pressure, phase)
    mu = thermo.mixture_viscosity(composition, state.T, pressure, quality)
    conductivities = thermo.transport_mixture_thermal_conductivity(composition, state.T, pressure, quality)
    k = conductivities.vapor if quality else conductivities.liquid
    assert evaluate(state.T) == pytest.approx(state.Cp*1000/thermo.mixture_MW(composition)*mu/k, rel=1e-12)


@pytest.mark.parametrize('phase,temperature', [('liquid', 450.), ('vapor', 350.)])
def test_prepared_steam_preserves_forced_phase_continuation(phase, temperature):
    thermo = create_thermodynamics(['water'], 'STEAM', ChemicalDatabase(enable_online=False))
    composition, pressure = {'water': 1.}, 3.
    quality = float(phase == 'vapor')
    evaluate = thermo.prepare_wall_prandtl(composition, pressure, phase)
    state = thermo.calculate_state(temperature, pressure, 1., composition,
                                  phase=phase, flash=False, include=('Cp',))
    mu = thermo.mixture_viscosity(composition, temperature, pressure, quality)
    conductivities = thermo.transport_mixture_thermal_conductivity(composition, temperature, pressure, quality)
    k = conductivities.vapor if quality else conductivities.liquid
    assert evaluate(temperature) == pytest.approx(state.Cp*1000/thermo.mixture_MW(composition)*mu/k, rel=1e-12)


def test_prepared_conductivity_records_temperature_and_context_provenance():
    thermo = create_thermodynamics(['water'], 'IDEAL', ChemicalDatabase(enable_online=False))
    for unit, temperatures in [('hot', [310., 350.]), ('cold', [310., 350.])]:
        with thermo.quality_context(kind='unit', unit_id=unit, affects_result=True):
            evaluate = thermo.prepare_wall_prandtl({'water': 1.}, 3., 'liquid')
            for T in temperatures:
                assert evaluate(T) > 0
    rows = [r for r in thermo.lazy_property_quality_sources() if r['property'] == 'liquid_thermal_conductivity(T)']
    assert rows
    contexts = {c['unit_id']: c for r in rows for c in r['contexts'] if c.get('unit_id') in ('hot', 'cold')}
    assert contexts['hot']['T_min'] == 310.
    assert contexts['hot']['T_max'] == 350.
    assert contexts['cold']['T_min'] == 310.
    assert contexts['cold']['T_max'] == 350.
    viscosity = [r for r in thermo.lazy_property_quality_sources() if r['property'] == 'liquid_viscosity(T)']
    assert {c.get('unit_id') for r in viscosity for c in r['contexts']} >= {'hot', 'cold'}


def test_cached_liquid_cp_keeps_degraded_range_provenance_in_new_context():
    thermo = create_thermodynamics(['water'], 'IDEAL', ChemicalDatabase(enable_online=False))
    kernel = thermo._liquid_cp_kernel('water')
    temperature = kernel.Tmin-1.
    for unit in ('first', 'second'):
        with thermo.quality_context(kind='unit', unit_id=unit, affects_result=True):
            assert thermo.Cp_liquid('water', temperature) > 0
    rows = [r for r in thermo.lazy_property_quality_sources() if r['property'] == 'Cp_liquid(T)']
    assert {c.get('unit_id') for r in rows for c in r['contexts']} >= {'first', 'second'}
