"""Finite-rate layer balances with controlled, explicitly supplied properties."""

from types import SimpleNamespace
import math

import pytest

from chemical_properties import ChemicalDatabase
from crystallizer_specs import CrystallizerSpecificationError, validate_crystallizer_specification
from dof_analyzer import SpecificationStatus, analyze_dof
from layer_crystallization import _wilke_chang, solve_layer_growth
from pfd_parser import parse_pfd, validate_pfd
from simulator import Simulator
from thermodynamics_models.base import IdealThermodynamics
from thermodynamics_models.common import ThermodynamicsError
from unit_operations_base import UnitOperationError
from unit_operations_solids import Crystallizer


@pytest.fixture
def thermo(monkeypatch):
    backend = IdealThermodynamics(['water', 'ethanol'], ChemicalDatabase(enable_online=False))
    backend.configure_permanent_solids(
        ['water', 'ethanol'], [], conventional_solid_components=['water']
    )
    monkeypatch.setattr(backend, 'pure_thermal_conductivity',
                        lambda c, T, phase: 2.2 if phase == 'solid' else 0.4)
    return backend


def configuration():
    return dict(component='water', amounts_kmol={'water': 9, 'ethanol': 1},
                bulk_temperature_K=270, pressure_bar=1, wall_temperature_K=250,
                area_m2=10, film_thickness_m=1e-3, growth_time_h=0.1,
                binary_diffusivity_m2_s=1e-9, profile_points=5)


def test_growth_material_stefan_and_sle_balances(thermo):
    result = solve_layer_growth(thermo, **configuration())
    assert 0 < result.solid_amount_kmol < 9
    assert result.thickness_m == pytest.approx(
        result.solid_amount_kmol * thermo._solid_molar_volume('water', 250) / 10
    )
    assert result.wall_energy_kJ < 0
    for point in result.profile:
        assert abs(point['sle_residual']) < 1e-8
        assert abs(point['stefan_residual_W_m2']) < 0.1
        assert 250 <= point['interface_temperature_K'] < 270
        assert point['interface_mole_fraction'] < point['bulk_mole_fraction']
    refined = solve_layer_growth(thermo, **{**configuration(), 'relative_tolerance': 1e-8})
    assert result.solid_amount_kmol == pytest.approx(refined.solid_amount_kmol, rel=1e-5)
    assert result.wall_energy_kJ == pytest.approx(refined.wall_energy_kJ, rel=1e-5)


def test_transport_changes_growth_rate(thermo):
    slow = solve_layer_growth(thermo, **configuration())
    fast = solve_layer_growth(thermo, **{**configuration(), 'binary_diffusivity_m2_s': 2e-9})
    assert fast.solid_amount_kmol > slow.solid_amount_kmol


def test_nonlinear_solid_conductivity_matches_integrated_fourier_law(thermo, monkeypatch):
    monkeypatch.setattr(thermo, 'pure_thermal_conductivity', lambda c, T, phase:
                        2 + 0.2 * (T - 250)**2 if phase == 'solid' else 0.4)
    result = solve_layer_growth(thermo, **configuration())
    for point in result.profile[1:]:
        delta = point['interface_temperature_K'] - 250
        expected = (2 * delta + 0.2 * delta**3 / 3) / point['thickness_m']
        assert point['wall_heat_flux_W_m2'] == pytest.approx(expected, rel=1e-6)
        # Independently check Stefan heat release for this ideal solution.
        latent = 1e6 * (thermo.enthalpy_liquid('water', 250 + delta)
                        - thermo.process_solid_enthalpy('water', 250 + delta))
        assert expected == pytest.approx(
            0.4 / 1e-3 * (270 - 250 - delta) + point['flux'] * latent, rel=1e-6
        )


def test_thermal_film_can_change_without_changing_initial_mass_flux(thermo):
    baseline = solve_layer_growth(thermo, **configuration())
    insulated = solve_layer_growth(thermo, **configuration(), thermal_film_thickness_m=2e-3)
    assert insulated.profile[0]['flux'] == pytest.approx(baseline.profile[0]['flux'])
    assert insulated.solid_amount_kmol > baseline.solid_amount_kmol
    assert insulated.profile[0]['wall_heat_flux_W_m2'] < baseline.profile[0]['wall_heat_flux_W_m2']


def test_undersaturated_liquid_does_not_form_layer(thermo):
    result = solve_layer_growth(thermo, **{
        **configuration(), 'amounts_kmol': {'water': 1, 'ethanol': 9},
    })
    assert result.solid_amount_kmol == 0
    assert result.thickness_m == 0
    assert result.wall_energy_kJ < 0


@pytest.mark.parametrize('purity', [0.9, 0.999])
def test_ideal_film_flux_matches_analytic_stefan_diffusion(monkeypatch, thermo, purity):
    monkeypatch.setattr(thermo, 'mixture_liquid_density', lambda *args: 50.0)
    result = solve_layer_growth(thermo, **{
        **configuration(), 'amounts_kmol': {'water': purity * 10, 'ethanol': (1 - purity) * 10},
    })
    first = result.profile[0]
    expected = 50 * 1e-9 / 1e-3 * math.log(
        (1 - first['interface_mole_fraction']) / (1 - first['bulk_mole_fraction'])
    )
    assert first['flux'] == pytest.approx(expected, rel=1e-6)


def test_specified_diffusivity_and_nonideal_activity_path(monkeypatch, thermo):
    monkeypatch.setattr(thermo, '_pure_viscosity', lambda *args: 1e-3)
    monkeypatch.setattr(thermo, 'mixture_liquid_molar_volume', lambda *args: 0.03)
    # Symmetric Margules with temperature-independent A: zero excess enthalpy.
    monkeypatch.setattr(thermo, 'activity_coefficients', lambda T, x: {
        'water': math.exp(0.4 * x['ethanol']**2),
        'ethanol': math.exp(0.4 * x['water']**2),
    }, raising=False)
    result = solve_layer_growth(thermo, **configuration())
    assert result.solid_amount_kmol > 0
    assert max(abs(p['sle_residual']) for p in result.profile) < 1e-8


def test_wilke_chang_units_and_override_bypasses_estimation(monkeypatch, thermo):
    monkeypatch.setattr(thermo, '_pure_viscosity', lambda *args: 1e-3)
    monkeypatch.setattr(thermo, 'mixture_liquid_molar_volume', lambda *args: 0.06)
    expected = 7.4e-12 * (2.6 * thermo.props['water'].MW)**0.5 * 300 / 60**0.6
    assert _wilke_chang(thermo, 'ethanol', 'water', 300, 1) == pytest.approx(expected)
    estimated = solve_layer_growth(thermo, **{**configuration(), 'binary_diffusivity_m2_s': None},
                                    thermal_mode='cooling')
    assert estimated.profile[0]['bulk_maxwell_stefan_diffusivity_m2_s'] > 0
    assert estimated.profile[-1]['bulk_maxwell_stefan_diffusivity_m2_s'] != pytest.approx(
        estimated.profile[0]['bulk_maxwell_stefan_diffusivity_m2_s'], rel=1e-3, abs=0
    )
    def forbidden(*args):
        raise AssertionError('Override should bypass diffusivity estimation')
    monkeypatch.setattr('layer_crystallization._wilke_chang', forbidden)
    supplied = solve_layer_growth(thermo, **configuration())
    assert supplied.profile[0]['bulk_maxwell_stefan_diffusivity_m2_s'] == pytest.approx(1e-9)


def test_cooling_integrates_bulk_enthalpy_and_conserves_energy(thermo):
    result = solve_layer_growth(thermo, **configuration(), thermal_mode='cooling')
    assert 250 < result.bulk_temperature_K < 270
    assert result.solid_amount_kmol > 0
    for point in result.profile:
        liquid = point['remaining_component_amounts_kmol']
        total = sum(liquid.values())
        actual = total * thermo.mixture_enthalpy(
            {c: n/total for c, n in liquid.items()}, point['bulk_temperature_K'], 0, P=1
        )
        assert actual == pytest.approx(point['bulk_energy_ledger_kJ'], abs=1e-4)
        assert abs(point['energy_residual_kJ']) < 1e-5


def test_cooling_without_crystals_matches_liquid_energy_change(thermo):
    config = {**configuration(), 'amounts_kmol': {'water': 1, 'ethanol': 9}}
    result = solve_layer_growth(thermo, **config, thermal_mode='cooling')
    assert result.solid_amount_kmol == 0
    assert 250 < result.bulk_temperature_K < 270
    x = {'water': 0.1, 'ethanol': 0.9}
    delta = 10 * (thermo.mixture_enthalpy(x, result.bulk_temperature_K, 0, P=1)
                  - thermo.mixture_enthalpy(x, 270, 0, P=1))
    assert result.wall_energy_kJ == pytest.approx(delta, abs=1e-4)


def test_inclusions_preserve_history_and_components(thermo):
    result = solve_layer_growth(thermo, **configuration(), thermal_mode='cooling',
                                inclusion_max_fraction=0.3)
    assert result.trapped_component_amounts_kmol['ethanol'] > 0
    assert result.thickness_m > result.solid_amount_kmol * thermo._solid_molar_volume('water', 250) / 10
    assert abs(result.energy_residual_kJ) < 1e-5
    for c, initial in configuration()['amounts_kmol'].items():
        assert result.liquid_component_amounts_kmol[c] + result.trapped_component_amounts_kmol[c] + (
            result.solid_amount_kmol if c == 'water' else 0
        ) == pytest.approx(initial)
    trapped = result.trapped_component_amounts_kmol
    final_x = result.profile[-1]['bulk_mole_fraction']
    assert trapped['water'] / sum(trapped.values()) > final_x
    assert all(0 <= p['inclusion_fraction'] < 0.3 for p in result.profile)


def test_flat_plate_couples_transfer_to_velocity(thermo, monkeypatch):
    monkeypatch.setattr(thermo, 'mixture_viscosity', lambda *args: 1e-3)
    config = {**configuration(), 'film_thickness_m': None, 'film_model': 'flat_plate',
              'plate_length_m': 0.1, 'liquid_velocity_m_s': 0.1}
    slow = solve_layer_growth(thermo, **config)
    fast = solve_layer_growth(thermo, **{**config, 'liquid_velocity_m_s': 0.4})
    for key in ('film_thickness_m', 'thermal_film_thickness_m'):
        assert fast.profile[0][key] == pytest.approx(slow.profile[0][key] / 2)
    assert fast.profile[0]['flux'] == pytest.approx(slow.profile[0]['flux'] * 2)


@pytest.mark.parametrize('estimated', [False, True])
def test_multicomponent_cooling_films_and_inclusions(monkeypatch, estimated):
    backend = IdealThermodynamics(['ethanol', 'water', 'acetone'], ChemicalDatabase(enable_online=False))
    backend.configure_permanent_solids(
        ['ethanol', 'water', 'acetone'], [], conventional_solid_components=['water']
    )
    monkeypatch.setattr(backend, 'pure_thermal_conductivity', lambda c, T, phase: 2.2 if phase == 'solid' else 0.4)
    monkeypatch.setattr(backend, 'mixture_viscosity', lambda *args: 1e-3)
    monkeypatch.setattr(backend, '_pure_viscosity', lambda *args: 1e-3)
    amounts = {'ethanol': 0.6, 'water': 9, 'acetone': 0.4}
    result = solve_layer_growth(backend, **{
        **configuration(), 'amounts_kmol': amounts, 'thermal_mode': 'cooling',
        'film_model': 'flat_plate', 'film_thickness_m': None,
        'plate_length_m': 0.1, 'liquid_velocity_m_s': 0.1,
        'inclusion_max_fraction': 0.3,
        'binary_diffusivity_m2_s': None if estimated else 1e-9,
    })
    assert result.bulk_temperature_K < 270
    assert result.solid_amount_kmol > 0
    assert abs(result.energy_residual_kJ) < 1e-5
    for c, n in amounts.items():
        assert result.trapped_component_amounts_kmol[c] > 0
        assert result.trapped_component_amounts_kmol[c] + result.liquid_component_amounts_kmol[c] + (
            result.solid_amount_kmol if c == 'water' else 0
        ) == pytest.approx(n)
    assert result.liquid_component_amounts_kmol['ethanol'] / result.liquid_component_amounts_kmol['acetone'] == pytest.approx(1.5)


def test_rejects_multiple_crystallizing_components(thermo):
    thermo.configure_permanent_solids(
        ['water', 'ethanol'], [], conventional_solid_components=['water', 'ethanol']
    )
    with pytest.raises(ThermodynamicsError, match='only one crystallizing'):
        solve_layer_growth(thermo, **configuration())


def unit_params():
    return {'model': 'layer_growth', 'T': 270, 'T_wall': 250,
            'cooled_area': 10, 'film_thickness': 1e-3,
            'growth_time': 0.1, 'cycle_time': 1, 'binary_diffusivity': 1e-9,
            'layer_profile_points': 3}


def test_unit_cycle_conversion_and_total_energy(thermo):
    feed = thermo.calculate_state(280, 1, 10, {'water': 0.9, 'ethanol': 0.1}, phase='liquid')
    result = Crystallizer('L', thermo, unit_params()).solve({'in': feed})
    layer, mother = result.outlet_streams['cake'], result.outlet_streams['mother_liquor']
    assert not layer.solid_particle_size_distributions
    assert layer.F == pytest.approx(result.performance['solid_amount_kmol_per_batch'])
    for c, amount in feed.component_flows().items():
        assert layer.component_flows().get(c, 0) + mother.component_flows().get(c, 0) == pytest.approx(amount)
    assert result.heat_duty == pytest.approx(layer.F * layer.H + mother.F * mother.H - feed.F * feed.H)
    assert result.performance['wall_duty_kW'] + result.performance['conditioning_duty_kW'] == pytest.approx(result.heat_duty / 3600)


@pytest.mark.parametrize('retention', [0, 0.2, 1])
def test_inclusion_outlet_split_separates_trapped_and_drainage_liquid(thermo, retention):
    feed = thermo.calculate_state(270, 1, 10, {'water': 0.9, 'ethanol': 0.1}, phase='liquid')
    params = {**unit_params(), 'thermal_mode': 'cooling', 'inclusion_max_fraction': 0.3,
              'mother_liquor_retention': retention}
    params.pop('T')
    result = Crystallizer('L', thermo, params).solve({'in': feed})
    cake, liquor = result.outlet_streams['cake'], result.outlet_streams['mother_liquor']
    details = result.performance
    trapped = details['trapped_component_amounts_kmol_per_batch']
    free = details['profile'][-1]['remaining_component_amounts_kmol']
    assert cake.T < feed.T
    for c in feed.composition:
        assert cake.phase_component_flows()['liquid1'].get(c, 0) == pytest.approx(
            trapped[c] + retention * free[c], abs=1e-10
        )
        assert liquor.component_flows().get(c, 0) == pytest.approx((1-retention) * free[c], abs=1e-10)
        assert cake.component_flows().get(c, 0) + liquor.component_flows().get(c, 0) == pytest.approx(feed.component_flows()[c])
    assert result.heat_duty == pytest.approx(cake.F * cake.H + liquor.F * liquor.H - feed.F * feed.H)


def test_missing_solid_conductivity_is_not_substituted(monkeypatch):
    thermo = IdealThermodynamics(['water', 'ethanol'], ChemicalDatabase(enable_online=False))
    import property_resolver
    calls = []
    def resolve(identifier, T, *, phase, props):
        calls.append(phase)
        raise ValueError('solid conductivity unavailable')
    monkeypatch.setattr(property_resolver, 'get_property_resolver',
                        lambda: SimpleNamespace(resolve_thermal_conductivity=resolve))
    with pytest.raises(ThermodynamicsError, match='solid conductivity unavailable'):
        solve_layer_growth(thermo, **configuration())
    assert calls == ['solid']


@pytest.mark.parametrize('change, message', [
    ({'cycle_time': 0.01}, 'cycle_time'),
    ({'T_wall': 280}, 'wall temperature'),
    ({'film_thickness': 0}, 'film_thickness'),
    ({'thermal_film_thickness': 0}, 'thermal_film_thickness'),
])
def test_invalid_design_inputs(thermo, change, message):
    feed = thermo.calculate_state(280, 1, 10, {'water': 0.9, 'ethanol': 0.1}, phase='liquid')
    with pytest.raises(UnitOperationError, match=message):
        Crystallizer('L', thermo, {**unit_params(), **change}).solve({'in': feed})


def test_missing_design_variables_and_wrong_mode_are_rejected():
    with pytest.raises(CrystallizerSpecificationError, match='layer_growth requires'):
        validate_crystallizer_specification({'model': 'layer_growth', 'T': 270})
    with pytest.raises(CrystallizerSpecificationError, match='requires crystallization_mode=layer'):
        validate_crystallizer_specification({**unit_params(), 'crystallization_mode': 'suspension'})


@pytest.mark.parametrize('spec, message', [
    ({'thermal_mode': 'cooling'}, 'omit T/T_out'),
    ({'film_model': 'flat_plate'}, 'conflicts'),
    ({'plate_length': 1}, 'conflicts'),
    ({'film_model': 'unknown'}, 'film_model'),
    ({'thermal_mode': 'unknown'}, 'thermal_mode'),
])
def test_incompatible_mode_inputs_are_rejected(spec, message):
    with pytest.raises(CrystallizerSpecificationError, match=message):
        validate_crystallizer_specification({**unit_params(), **spec})


@pytest.mark.parametrize('cooling', [False, True])
def test_pfd_growth_units(monkeypatch, cooling):
    monkeypatch.setattr(IdealThermodynamics, 'pure_thermal_conductivity',
                        lambda self, c, T, phase: 2.2 if phase == 'solid' else 0.4)
    source = '''
PROCESS: Layer growth
ONLINE_LOOKUP: false
COMPONENTS:
    water | Water | type=conventional_with_solid
    ethanol | Ethanol
STREAM Feed : FEED -> C.in
    T = 280 [K]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.9, ethanol:0.1
STREAM Layer : C.layer -> PRODUCT
STREAM Mother : C.mother_liquor -> PRODUCT
UNIT C : Crystallizer
    model = layer_growth
    T = -3.15 [C]
    T_wall = -23.15 [C]
    cooled_area = 10 [m2]
    film_thickness = 1 [mm]
    thermal_film_thickness = 2 [mm]
    growth_time = 6 [min]
    cycle_time = 60 [min]
    binary_diffusivity = 0.00001 [cm2/s]
    layer_profile_points = 3
'''
    if cooling:
        source = source.replace('    T = -3.15 [C]', '    thermal_mode = cooling')
        source = source.replace('    film_thickness = 1 [mm]\n    thermal_film_thickness = 2 [mm]',
                                '    film_model = flat_plate\n    plate_length = 100 [mm]\n    liquid_velocity = 10 [cm/s]')
        monkeypatch.setattr(IdealThermodynamics, 'mixture_viscosity', lambda *args: 1e-3)
        source = source.replace('    ethanol | Ethanol', '    ethanol | Ethanol\n    acetone | Acetone')
        source = source.replace('water:0.9, ethanol:0.1', 'water:0.9, ethanol:0.06, acetone:0.04')
        source = source.replace('    layer_profile_points = 3', '    layer_profile_points = 3\n    inclusion_max_fraction = 0.3')
    pfd = parse_pfd(source)
    assert not validate_pfd(pfd)[0]
    assert analyze_dof(pfd).unit_results[0].status == SpecificationStatus.OK
    result = Simulator(pfd).run()
    assert result.converged, result.errors
    details = result.streams['Layer'].phase_details['layer_crystallization']
    assert details['T_wall_K'] == pytest.approx(250)
    if cooling:
        assert details['T_bulk_K'] < 280
        assert details['plate_length_m'] == pytest.approx(0.1)
        assert details['liquid_velocity_m_s'] == pytest.approx(0.1)
        assert result.streams['Layer'].component_flows()['acetone'] > 0
        assert set(details['trapped_component_amounts_kmol_per_batch']) == {'water', 'ethanol', 'acetone'}
    else:
        assert details['T_bulk_K'] == pytest.approx(270)
        assert details['thermal_film_thickness_m'] == pytest.approx(0.002)
    assert details['growth_time_h'] == pytest.approx(0.1)
