"""Constrained VLE pressure/quality states conserve every component."""

import pytest

from chemical_properties import ChemicalDatabase
from thermodynamics import ThermodynamicsError, create_thermodynamics


@pytest.fixture(params=['IDEAL', 'NRTL', 'RK', 'PR', 'SRK', 'PSRK', 'RKSMHV2', 'LKP'])
def mixture(request):
    components = ['water', 'ethanol'] if request.param in ('IDEAL', 'NRTL') else ['CH4', 'C2H6']
    model = create_thermodynamics(components, request.param, ChemicalDatabase(enable_online=False))
    return model, dict.fromkeys(components, .5), 1. if 'water' in components else 10.


@pytest.mark.parametrize('quality', [0., 1e-8, .0001, .001, .5, .999, .9999, 1-1e-8, 1.])
def test_requested_quality_closes_component_and_equilibrium_balances(mixture, quality):
    model, z, P = mixture
    state = model.calculate_state_PQ(P, quality, 2., z, include=())
    liquid, vapor = state.x or z, state.y or z
    assert state.vapor_fraction == pytest.approx(quality, abs=1e-12)
    for component in z:
        assert (1-quality)*liquid[component]+quality*vapor[component] == pytest.approx(
            z[component], abs=1e-9)
    if 0. < quality < 1.:
        assert abs(sum(liquid.values())-1.) < 1e-12
        assert abs(sum(vapor.values())-1.) < 1e-12
        # Ordinary activity-model TP flashes use a 1e-6 substitution tolerance.
        tolerance = 1e-6 if callable(getattr(model, 'activity_coefficients', None)) else 1e-7
        model._check_diagram_vle(liquid, vapor, state.T, P, fugacity_tolerance=tolerance)


@pytest.mark.parametrize('method', ['IDEAL', 'PR', 'PSRK', 'LKP'])
@pytest.mark.parametrize('quality', [.0001, .5, .9999])
def test_pure_fluid_quality_uses_coexistence_temperature(method, quality):
    model = create_thermodynamics(['CH4'], method, ChemicalDatabase(enable_online=False))
    z = {'CH4': 1.}
    state = model.calculate_state_PQ(10., quality, 1., z, include=())
    assert state.T == pytest.approx(model.bubble_point_T(z, 10.), abs=1e-6)
    assert state.x == state.y == z
    assert state.vapor_fraction == quality


def test_azeotropic_quality_has_a_single_temperature_and_identical_phases():
    model = create_thermodynamics(['water', 'H2O'], 'IDEAL', ChemicalDatabase(enable_online=False))
    z = {'water': .3, 'H2O': .7}
    state = model.calculate_state_PQ(1., .25, 1., z, include=())
    assert state.x == pytest.approx(z, abs=1e-12)
    assert state.y == pytest.approx(z, abs=1e-12)
    assert state.vapor_fraction == .25
    model._check_diagram_vle(state.x, state.y, state.T, state.P)


def test_unattainable_quality_cannot_return_an_unconverged_state(monkeypatch):
    model = create_thermodynamics(['CH4', 'C2H6'], 'PR', ChemicalDatabase(enable_online=False))
    z = {'CH4': .5, 'C2H6': .5}
    # A discontinuous TP result can bracket a numerical zero while never
    # supplying the requested split. The final physical audit must reject it.
    monkeypatch.setattr(model, 'flash_TP',
                        lambda composition, T, P, **options: (float(T > 190.), dict(z), dict(z)))
    with pytest.raises(ThermodynamicsError, match='quality|converge'):
        model.calculate_state_PQ(10., .25, 1., z, include=())


def test_a_matching_quality_cannot_hide_inconsistent_phase_inventories(monkeypatch):
    model = create_thermodynamics(['CH4', 'C2H6'], 'PR', ChemicalDatabase(enable_online=False))
    monkeypatch.setattr(model, 'flash_TP', lambda *args, **options: (
        .25, {'CH4': .8, 'C2H6': .2}, {'CH4': .9, 'C2H6': .1}))
    with pytest.raises(ThermodynamicsError, match='material residual'):
        model.calculate_state_PQ(10., .25, 1., {'CH4': .5, 'C2H6': .5}, include=())
