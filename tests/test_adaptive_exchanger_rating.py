"""Full-grid acceptance and physical regressions for calculated-U rating."""

import math
from types import SimpleNamespace

import pytest

from chemical_properties import ChemicalDatabase
from thermodynamics import ThermodynamicsError, create_thermodynamics
from unit_operations_basic import HeatExchanger
from unit_operations_base import UnitOperationError
from tests.test_heat_exchanger_transport import ConstantLiquid, parameters
from tests.test_heat_transfer_extensions import shell_parameters


class RatingCurve(HeatExchanger):
    """Independent area equation: A(Q)=Q on the requested grid, target A=1.

    Coarse estimates can be biased or invalid. Every returned duty must solve
    the full equation, independently of how an initial estimate was obtained.
    """

    def __init__(self, fault='bias', full=16):
        self.unit_id, self.fault, self.full = 'RATING', fault, full
        self.calls = []

    def _thermo_for_stream(self, state):
        return None

    def _curve_metrics(self, hot, cold, P_hot, P_cold, duty, segments, *args, **kwargs):
        self.calls.append((segments, duty))
        if self.fault == 'programming':
            raise KeyError('unrelated programming error')
        if segments < self.full:
            if self.fault == 'coarse-domain':
                raise ThermodynamicsError('coarse property domain')
            if self.fault == 'coarse-budget':
                raise RuntimeError('coarse numerical iteration budget')
            factor = 2. if self.fault == 'no-local-bracket' else 1.001
            if self.fault == 'exact':
                factor = 1.
            if self.fault == 'local-hole':
                factor = 1.002 if segments == 4 else 1.
            area = duty*factor
        else:
            area = duty
            if self.fault == 'domain':
                raise ThermodynamicsError('full property domain')
            if self.fault == 'nan':
                area = math.nan
            if self.fault == 'inf':
                area = math.inf
            if self.fault == 'jump':
                area = 0. if duty < 1. else 2.
            if self.fault == 'local-hole' and .9989 < duty < .9991:
                raise ThermodynamicsError('invalid local correction endpoint')
        return {'calculated_U_area_required_m2': area, 'UA_required_W_per_K': area,
                'temperature_cross': self.fault == 'cross' and segments == self.full}

    def rate(self, calculated=True):
        return self._rate_exchanger_by_curves(
            SimpleNamespace(T=2.), SimpleNamespace(T=1.), 1., 1.,
            None if calculated else 1., 1., None, False, self.full, 'countercurrent',
            transport=SimpleNamespace(inlet_capacity_W_K=lambda *args: 1.) if calculated else None)


@pytest.mark.parametrize('full', [4, 7, 9, 16, 40])
def test_coarse_duty_is_never_accepted_without_solving_requested_area(full):
    unit = RatingCurve(full=full)
    duty = unit.rate()
    assert duty == pytest.approx(1., abs=1e-8)
    # The requested resolution, including non-power-of-two grids, determines
    # the answer. A coarse root is deliberately wrong by about 0.1 percent.
    assert any(grid == full and abs(Q-1.) < 1e-8 for grid, Q in unit.calls)


def test_exact_coarse_solution_needs_only_one_expensive_verification():
    unit = RatingCurve('exact', full=40)
    assert unit.rate() == pytest.approx(1., abs=1e-8)
    full_calls = [Q for grid, Q in unit.calls if grid == 40]
    assert len(full_calls) == 1
    assert full_calls[0] == pytest.approx(1., abs=1e-8)


@pytest.mark.parametrize('fault', ['nan', 'inf', 'cross', 'domain', 'jump'])
def test_invalid_or_discontinuous_full_equation_cannot_accept_a_coarse_root(fault):
    unit = RatingCurve(fault)
    with pytest.raises((ThermodynamicsError, UnitOperationError, ValueError)):
        unit.rate()
    assert any(grid == unit.full for grid, _ in unit.calls)


@pytest.mark.parametrize('fault', ['coarse-domain', 'coarse-budget', 'no-local-bracket'])
def test_failed_acceleration_returns_authoritative_full_grid_solution(fault):
    unit = RatingCurve(fault)
    assert unit.rate() == pytest.approx(1., abs=1e-8)
    assert any(grid == unit.full and Q > 1. for grid, Q in unit.calls)


def test_invalid_local_endpoint_abandons_the_bracket_and_refines():
    unit = RatingCurve('local-hole')
    assert unit.rate() == pytest.approx(1., abs=1e-8)
    invalid = next(i for i, (grid, Q) in enumerate(unit.calls) if grid == unit.full and .9989 < Q < .9991)
    # No attempt is made to extend the local bracket across that invalid
    # endpoint; the next work belongs to the finer trial grid.
    assert unit.calls[invalid+1][0] == 8


def test_unrelated_programming_error_is_not_hidden_by_refinement():
    with pytest.raises(KeyError, match='programming error'):
        RatingCurve('programming').rate()


def test_prescribed_u_keeps_the_requested_grid_without_calculated_u_trials():
    unit = RatingCurve(full=16)
    assert unit.rate(calculated=False) == pytest.approx(1., abs=1e-8)
    assert {grid for grid, _ in unit.calls} == {16}


class PropertyRidge(ConstantLiquid):
    def __init__(self, center, width, invalid=False):
        self.center, self.width, self.invalid = center, width, invalid

    def mixture_viscosity(self, composition, T, P, V):
        if P != 4.:
            return .001
        if self.invalid and abs(T-self.center) < self.width:
            raise ThermodynamicsError('unsupported narrow transport domain')
        return .001*(1+.8*math.exp(-((T-self.center)/self.width)**2))


def check_rating(unit, inlets, area, design_duty, *, minimum_nodes=48, temperature_tolerance=1e-5):
    result = unit.solve(inlets)
    perf = result.performance
    assert perf['duty_kW'] == pytest.approx(design_duty, rel=1e-6)
    assert perf['area_required_m2'] == pytest.approx(area, rel=1e-6)
    assert len(perf['calculated_U_nodes']) >= minimum_nodes
    assert perf['curve_segments'] == minimum_nodes//3
    assert not perf['temperature_cross']
    assert max(n['wall_temperature_residual_K'] for n in perf['calculated_U_nodes']) < temperature_tolerance
    for port, inlet in inlets.items():
        outlet = result.outlet_streams[port.replace('_in', '_out')]
        assert outlet.composition == inlet.composition
        assert outlet.F == inlet.F
        assert abs(inlet.F*(outlet.H-inlet.H)/3600) == pytest.approx(perf['duty_kW'], rel=1e-7, abs=2e-7)
    return result


@pytest.mark.parametrize('center', [301., 302.1, 303.])
@pytest.mark.parametrize('width', [.03, .08, .2, .6])
def test_narrow_property_features_require_requested_grid_area(center, width):
    thermo = PropertyRidge(center, width)
    inlets = {'tube_in': thermo.calculate_state(360., 3., 200., {'water': 1.}),
              'shell_in': thermo.calculate_state(300., 4., 300., {'water': 1.})}
    params = parameters(curve_segments=40)
    area = HeatExchanger('DESIGN', thermo, {**params, 'Q': 25.}).solve(inlets).performance['area_required_m2']
    check_rating(HeatExchanger('RATING', thermo, {**params, 'A': area}), inlets, area, 25., minimum_nodes=120)


def test_unsupported_transport_region_still_fails_instead_of_being_skipped():
    thermo = PropertyRidge(302.1, .15, invalid=True)
    inlets = {'tube_in': thermo.calculate_state(360., 3., 200., {'water': 1.}),
              'shell_in': thermo.calculate_state(300., 4., 300., {'water': 1.})}
    with pytest.raises((ThermodynamicsError, UnitOperationError), match='domain'):
        HeatExchanger('RATING', thermo, parameters(curve_segments=16, A=.14)).solve(inlets)


@pytest.mark.parametrize('pattern', ['countercurrent', 'cocurrent'])
@pytest.mark.parametrize('hot_F,duty', [(1., .1), (10., 1.), (200., 25.), (200., 240.)])
def test_laminar_transition_turbulent_and_pinch_rating(pattern, hot_F, duty):
    thermo = ConstantLiquid()
    inlets = {'tube_in': thermo.calculate_state(360., 3., hot_F, {'water': 1.}),
              'shell_in': thermo.calculate_state(300., 3., 300., {'water': 1.})}
    params = parameters(curve_segments=16, flow_pattern=pattern)
    # A cocurrent water/water curve crosses before the countercurrent 240 kW
    # pinch case. Its physically valid near-pinch specification is 145 kW.
    if pattern == 'cocurrent' and duty == 240.:
        duty = 145.
    area = HeatExchanger('DESIGN', thermo, {**params, 'Q': duty}).solve(inlets).performance['area_required_m2']
    check_rating(HeatExchanger('RATING', thermo, {**params, 'A': area}), inlets, area, duty)


@pytest.mark.parametrize('method,composition,hot_T,cold_T', [
    ('STEAM', {'water': 1.}, 360., 300.),
    ('IDEAL', {'water': 1.}, 360., 300.),
    ('NRTL', {'water': .9, 'ethanol': .1}, 350., 305.),
])
def test_real_liquid_and_mixture_rating_closes_energy_and_area(method, composition, hot_T, cold_T):
    thermo = create_thermodynamics(list(composition), method, ChemicalDatabase(enable_online=False))
    inlets = {'tube_in': thermo.calculate_state(hot_T, 5., 200., composition),
              'shell_in': thermo.calculate_state(cold_T, 4., 300., composition)}
    params = parameters(curve_segments=16, P_drop_tube=.5, P_drop_shell=.2)
    area = HeatExchanger('DESIGN', thermo, {**params, 'Q': 20.}).solve(inlets).performance['area_required_m2']
    result = check_rating(HeatExchanger('RATING', thermo, {**params, 'A': area}), inlets, area, 20.)
    assert result.outlet_streams['tube_out'].P == pytest.approx(4.5)
    assert result.outlet_streams['shell_out'].P == pytest.approx(3.8)


@pytest.mark.parametrize('phase_case', ['condense_tube', 'condense_shell', 'boil_and_condense', 'complete_transition'])
def test_phase_changes_are_audited_on_the_requested_grid(phase_case):
    thermo = create_thermodynamics(['water'], 'STEAM', ChemicalDatabase(enable_online=False))
    z, segments, duty = {'water': 1.}, 16, 20.
    params = parameters(curve_segments=segments)
    if phase_case == 'condense_tube':
        tube = thermo.calculate_state_PQ(5., .8, 20., z)
        shell = thermo.calculate_state(350., 3., 200., z)
    elif phase_case == 'condense_shell':
        tube = thermo.calculate_state(320., 3., 400., z)
        shell = thermo.calculate_state_PQ(5., .95, 15., z)
        params = shell_parameters(curve_segments=segments, baffle_count=1)
        duty = 10.
    elif phase_case == 'boil_and_condense':
        tube = thermo.calculate_state_PQ(3., .05, 100., z)
        shell = thermo.calculate_state_PQ(5., .8, 20., z)
    else:
        tube = thermo.calculate_state(450., 5., 1., z)
        shell = thermo.calculate_state(310., 3., 100., z)
    inlets = {'tube_in': tube, 'shell_in': shell}
    spec = {'T_tube_out': 380.} if phase_case == 'complete_transition' else {'Q': duty}
    design = HeatExchanger('DESIGN', thermo, {**params, **spec}).solve(inlets)
    area = design.performance['area_required_m2']
    result = check_rating(HeatExchanger('RATING', thermo, {**params, 'A': area}), inlets, area,
                          design.performance['duty_kW'])
    regimes = {n[side]['flow_regime'] for n in result.performance['calculated_U_nodes']
               for side in ('tube', 'shell', 'annulus') if side in n}
    assert 'film_condensation' in regimes
    if phase_case == 'boil_and_condense':
        assert 'saturated_boiling' in regimes
    if phase_case == 'complete_transition':
        assert regimes & {'laminar', 'transition', 'turbulent'}
        assert result.outlet_streams['tube_out'].vapor_fraction == 0.
