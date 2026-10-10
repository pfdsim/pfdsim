"""Geometry-based exchanger coefficients, independent energy/NTU checks."""

import math
from pathlib import Path

import pytest

from chemical_properties import ChemicalDatabase
from dof_analyzer import DOFAnalyzer, SpecificationStatus
from pfd_parser import parse_pfd
from thermodynamics import StreamState, create_thermodynamics, TransportPhaseValues
from transport_correlations import TransportCorrelationError, gnielinski_liquid_transfer
from unit_operations_basic import HeatExchanger
from unit_operations_base import UnitOperationError
from unit_settings import unit_setting_schema


class ConstantLiquid:
    """Analytical single-liquid reference, independent of production properties."""

    components = ['water']

    def calculate_state(self, T, P, F, composition, phase=None, flash=True, include=None):
        return StreamState(T, P, F, composition, vapor_fraction=0.0,
                           H=75 * (T - 298.15), Cp=75, MW=18,
                           rho=1000 / 18, x=dict(composition))

    def calculate_state_PH(self, P, H, F, composition, include=None, phase=None):
        return self.calculate_state(298.15 + H / 75, P, F, composition)

    def mixture_MW(self, composition):
        return 18.0

    def mixture_Cp(self, composition, T, vapor_fraction, P=None):
        return 75.0

    def mixture_viscosity(self, composition, T, P, V):
        return 1e-3

    def mixture_liquid_thermal_conductivity(self, composition, T):
        return 0.6

    def transport_mixture_thermal_conductivity(self, composition, T, P, vapor_fraction=1, x=None, y=None):
        return TransportPhaseValues(liquid=.6 if vapor_fraction < 1 else None,
                                    vapor=.025 if vapor_fraction > 0 else None)


@pytest.fixture
def constant_case():
    thermo = ConstantLiquid()
    hot = thermo.calculate_state(360, 3, 200, {'water': 1})
    cold = thermo.calculate_state(300, 3, 300, {'water': 1})
    return thermo, {'tube_in': hot, 'shell_in': cold}


def parameters(**updates):
    return {
        'U_model': 'double_pipe_gnielinski',
        'tube_inner_diameter': 0.02, 'tube_outer_diameter': 0.024,
        'shell_inner_diameter': 0.04, 'wall_conductivity': 16,
        'curve_segments': 8, **updates,
    }


def test_correlation_reference_values_and_annulus_geometry():
    # Independent numerical evaluation of the published smooth-tube formula
    # at Pr=Pr_wall=1; the denominator and property correction are exactly 1.
    tube = gnielinski_liquid_transfer(100000, 1, 1)
    assert tube['nusselt'] == pytest.approx(222.651, rel=1e-5)
    annulus = gnielinski_liquid_transfer(100000, 1, 1, annulus_diameter_ratio=0.6)
    assert annulus['nusselt'] > 0
    assert annulus['nusselt'] != pytest.approx(tube['nusselt'])
    corrected = gnielinski_liquid_transfer(100000, 1, 2)
    assert corrected['nusselt'] / tube['nusselt'] == pytest.approx(2**-0.11)


@pytest.mark.parametrize('re,pr,wall', [(0, 1, 1), (3999, 1, 1), (1e6 + 1, 1, 1),
                                      (1e5, .49, 1), (1e5, 1, 1001), (float('nan'), 1, 1)])
def test_correlation_does_not_extrapolate(re, pr, wall):
    with pytest.raises(TransportCorrelationError):
        gnielinski_liquid_transfer(re, pr, wall)


@pytest.mark.parametrize('pattern', ['countercurrent', 'cocurrent'])
@pytest.mark.parametrize('tube_is_hot', [True, False])
def test_rating_matches_independent_epsilon_ntu_and_conserves_energy(constant_case, pattern, tube_is_hot):
    thermo, inlets = constant_case
    if not tube_is_hot:
        inlets = {'tube_in': inlets['shell_in'], 'shell_in': inlets['tube_in']}
    length = 8
    result = HeatExchanger('HX', thermo, parameters(length=length, flow_pattern=pattern)).solve(inlets)
    perf = result.performance
    U = perf['calculated_U_nodes'][0]['U_W_m2_K']
    capacity_hot, capacity_cold = 200 * 75 / 3.6, 300 * 75 / 3.6  # W/K
    cmin, cmax = min(capacity_hot, capacity_cold), max(capacity_hot, capacity_cold)
    cr = cmin / cmax
    ntu = U * math.pi * 0.024 * length / cmin
    if pattern == 'countercurrent':
        exponential = math.exp(-ntu * (1 - cr))
        effectiveness = (1 - exponential) / (1 - cr * exponential)
    else:
        effectiveness = (1 - math.exp(-ntu * (1 + cr))) / (1 + cr)
    expected_kW = effectiveness * cmin * 60 / 1000
    assert perf['duty_kW'] == pytest.approx(expected_kW, rel=1e-7)
    assert perf['area_utilization'] == pytest.approx(1, rel=1e-6)
    assert perf['U_area_basis'] == 'tube_outer_surface'
    assert perf['tube_side'] == ('hot' if tube_is_hot else 'cold')
    assert len(perf['calculated_U_nodes']) == 24
    for node in perf['calculated_U_nodes']:
        resistance = sum(node[key] for key in (
            'tube_film_resistance_m2_K_W', 'annulus_film_resistance_m2_K_W',
            'wall_resistance_m2_K_W', 'fouling_resistance_m2_K_W'))
        assert node['U_W_m2_K'] * resistance == pytest.approx(1)
        assert node['wall_temperature_residual_K'] < 1e-6
    for side in ('hot', 'cold'):
        inlet = inlets[perf[f'{side}_in_port']]
        outlet = result.outlet_streams[perf[f'{side}_out_port']]
        expected = perf['duty_kW'] * 3600 * (-1 if side == 'hot' else 1)
        assert inlet.F * (outlet.H - inlet.H) == pytest.approx(expected, rel=1e-8)
        assert inlet.F == outlet.F
    assert result.heat_duty == 0


def test_design_rating_round_trip_fouling_and_area_units(constant_case):
    thermo, inlets = constant_case
    design = HeatExchanger('D', thermo, parameters(Q=25)).solve(inlets)
    required = design.performance['area_required_m2']
    rating = HeatExchanger('R', thermo, parameters(A=required)).solve(inlets)
    dirty = HeatExchanger('F', thermo, parameters(A=required, fouling_tube=.0002, fouling_shell=.0003)).solve(inlets)
    imperial = HeatExchanger('I', thermo, parameters(A=required / .09290304, __unit__A='ft2')).solve(inlets)
    assert rating.performance['duty_kW'] == pytest.approx(25, rel=1e-7)
    assert imperial.performance['duty_kW'] == pytest.approx(25, rel=1e-7)
    assert dirty.performance['duty_kW'] < 25
    assert dirty.performance['U_W_m2_K'] < design.performance['U_W_m2_K']
    assert design.performance['length_required_m'] == pytest.approx(required / (math.pi * .024))


@pytest.mark.parametrize('updates,message', [
    ({'tube_outer_diameter': .01}, 'geometry'),
    ({'tube_inner_diameter': float('nan')}, 'finite'),
    ({'fouling_tube': -.01}, 'nonnegative'),
    ({'wall_conductivity': 0}, 'positive'),
    ({'__unit__tube_inner_diameter': 'furlong'}, 'unsupported unit'),
    ({'U': 500}, 'cannot be combined'),
    ({'U': 'auto'}, 'cannot be combined'),
    ({'estimate_U': True}, 'cannot be combined'),
    ({'UA': 500}, 'cannot be combined'),
    ({'tube_side': 'cold'}, 'conflicts'),
    ({'tube_passes': 2}, 'one tube pass'),
    ({'A': 1, 'length': 1}, 'must equal'),
    ({'U_model': 'invented'}, 'unknown U_model'),
    ({'type': 'SHELL_TUBE'}, 'concentric double-pipe'),
    ({'flow_pattern': 'plate'}, 'concentric double-pipe'),
])
def test_invalid_or_conflicting_specs_fail(constant_case, updates, message):
    thermo, inlets = constant_case
    with pytest.raises(UnitOperationError, match=message):
        HeatExchanger('HX', thermo, parameters(Q=25, **updates)).solve(inlets)


def test_generic_ports_need_explicit_geometry_assignment(constant_case):
    thermo, inlets = constant_case
    generic = {'hot_in': inlets['tube_in'], 'cold_in': inlets['shell_in']}
    with pytest.raises(UnitOperationError, match='explicit tube_side'):
        HeatExchanger('HX', thermo, parameters(Q=25)).solve(generic)
    result = HeatExchanger('HX', thermo, parameters(Q=25, tube_side='hot')).solve(generic)
    assert result.performance['tube_side'] == 'hot'


def test_laminar_inlets_are_supported(constant_case):
    thermo, inlets = constant_case
    inlets['tube_in'].F = 1
    result = HeatExchanger('HX', thermo, parameters(A=.01)).solve(inlets)
    assert result.performance['calculated_U_nodes'][0]['tube']['flow_regime'] == 'laminar'
    assert result.performance['calculated_U_nodes'][0]['tube']['nusselt'] == pytest.approx(48/11)


def test_liquid_wall_phase_is_checked_even_for_forced_liquid(constant_case):
    class BoilingLiquid(ConstantLiquid):
        def calculate_state(self, T, P, F, composition, phase=None, flash=True, include=None):
            state = super().calculate_state(T, P, F, composition, phase, flash, include)
            if P == 1 and T > 305 and phase is None:
                state.vapor_fraction = 1
            return state
    thermo = BoilingLiquid()
    hot = thermo.calculate_state(360, 3, 200, {'water': 1})
    cold = thermo.calculate_state(300, 1, 300, {'water': 1})
    with pytest.raises(UnitOperationError, match='annulus wall'):
        HeatExchanger('HX', thermo, parameters(Q=1, shell_phase='liquid')).solve({'tube_in': hot, 'shell_in': cold})


def test_out_of_domain_rating_is_not_accepted_as_a_discontinuous_root(constant_case):
    class VariableViscosity(ConstantLiquid):
        def mixture_viscosity(self, composition, T, P, V):
            return 1e-3 * math.exp((350 - T) / 5) if P == 3 else 1e-3
    thermo = VariableViscosity()
    inlets = {'tube_in': thermo.calculate_state(360, 3, 200, {'water': 1}),
              'shell_in': thermo.calculate_state(300, 4, 300, {'water': 1})}
    with pytest.raises(UnitOperationError, match='cannot rate.*domain'):
        HeatExchanger('HX', thermo, parameters(length=100)).solve(inlets)


def test_calculated_coefficients_are_not_deferred_in_recycle_context(constant_case):
    thermo, inlets = constant_case
    unit = HeatExchanger('HX', thermo, parameters(Q=25))
    unit.solve_context = {'recycle_evaluation': 2, 'expensive_diagnostics': False}
    result = unit.solve(inlets)
    assert not result.performance['curve_metrics_delayed']
    assert result.performance['calculated_U_nodes']


def test_real_water_properties_and_variable_u():
    thermo = create_thermodynamics(['water'], 'IDEAL', db=ChemicalDatabase(enable_online=False))
    inlets = {'tube_in': thermo.calculate_state(360, 3, 200, {'water': 1}),
              'shell_in': thermo.calculate_state(300, 3, 300, {'water': 1})}
    result = HeatExchanger('HX', thermo, parameters(Q=25)).solve(inlets)
    nodes = result.performance['calculated_U_nodes']
    assert nodes[0]['U_W_m2_K'] != pytest.approx(nodes[-1]['U_W_m2_K'])
    assert 500 < result.performance['U_W_m2_K'] < 10000
    for node in nodes:
        assert 1e4 < node['tube']['reynolds'] < 1e6
        assert 1e4 < node['annulus']['reynolds'] < 1e6
        assert 300 < node['tube']['wall_T_K'] < 360
        assert 300 < node['annulus']['wall_T_K'] < 360


@pytest.mark.parametrize('method,composition', [('STEAM', {'water': 1}),
                                               ('NRTL', {'water': .9, 'ethanol': .1})])
def test_real_steam_and_liquid_mixture_backends(method, composition):
    thermo = create_thermodynamics(list(composition), method, db=ChemicalDatabase(enable_online=False))
    inlets = {'tube_in': thermo.calculate_state(350, 3, 300, composition),
              'shell_in': thermo.calculate_state(305, 3, 400, composition)}
    coarse = HeatExchanger('C', thermo, parameters(Q=20, curve_segments=8)).solve(inlets)
    fine = HeatExchanger('F', thermo, parameters(Q=20, curve_segments=16)).solve(inlets)
    assert coarse.performance['area_required_m2'] == pytest.approx(fine.performance['area_required_m2'], rel=2e-4)
    rating = HeatExchanger('R', thermo, parameters(A=fine.performance['area_required_m2'], curve_segments=16)).solve(inlets)
    assert rating.performance['duty_kW'] == pytest.approx(20, rel=2e-6)


@pytest.mark.parametrize('unsupported', ['vapor', 'two_liquids', 'solid'])
def test_unsupported_phases_are_rejected(constant_case, unsupported):
    thermo, inlets = constant_case
    state = inlets['tube_in']
    if unsupported == 'vapor':
        state.vapor_fraction = 1
    elif unsupported == 'two_liquids':
        state.liquid2_fraction = .1
    else:
        state.solid_fraction = .1
    with pytest.raises(UnitOperationError, match='phase|fluid-only'):
        HeatExchanger('HX', thermo, parameters(Q=25)).solve(inlets)


def test_missing_transport_properties_fail_descriptively(constant_case):
    class MissingCp(ConstantLiquid):
        def calculate_state(self, *args, **kwargs):
            state = super().calculate_state(*args, **kwargs)
            state.Cp = None
            return state
    _, inlets = constant_case
    with pytest.raises(UnitOperationError, match='transport properties must be finite and positive'):
        HeatExchanger('HX', MissingCp(), parameters(Q=25)).solve(inlets)


def test_calculated_u_rejects_temperature_cross_even_when_override_is_requested(constant_case):
    thermo, inlets = constant_case
    with pytest.raises(UnitOperationError, match='Temperature cross'):
        HeatExchanger('HX', thermo, parameters(Q=1000, allow_temperature_cross=True)).solve(inlets)


def test_pfd_dof_and_guided_controls_support_calculated_rating():
    for specification in ('A = 1 [m2]', 'length = 8 [m]'):
        pfd = parse_pfd(f'UNIT HX : HeatExchanger\n    U_model = double_pipe_gnielinski\n    {specification}\n')
        result = DOFAnalyzer(pfd)._analyze_unit(pfd.units[0])
        assert result.status is SpecificationStatus.OK
    controls = {item['name'].lower(): item for item in unit_setting_schema('HeatExchanger')}
    for name in ('tube_inner_diameter', 'tube_outer_diameter', 'shell_inner_diameter', 'wall_conductivity', 'length', 'fouling_tube', 'fouling_shell'):
        assert controls[name]['unit']
    assert 'double_pipe_gnielinski' in controls['u_model']['values']


def test_maintained_double_pipe_example_runs_through_the_simulator():
    from simulator import Simulator
    path = Path(__file__).resolve().parents[1] / 'examples' / 'double_pipe_calculated_u.pfd'
    sim = Simulator.from_file(str(path))
    result = sim.run()
    assert result.converged
    assert sim.get_results_dict()['units']['HX-100']['performance']['U_model'] == 'double_pipe_gnielinski'
