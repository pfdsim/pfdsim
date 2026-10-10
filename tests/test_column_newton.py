"""Column globalization: numerical contracts and reproducible recovery cases."""

from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from equilibrium_stage_column import _dogleg_step
from pfd_parser import parse_pfd
from dof_analyzer import DOFAnalyzer, SpecificationStatus
from thermodynamics import create_thermodynamics
from thermodynamics_models.common import ThermodynamicsError
from unit_operations_base import UnitOperationError
from unit_operations_distillation import RigorousDistillation, CMODistillation
from unit_operations_separation import RigorousAbsorber, RigorousStripper, RigorousLiquidLiquidExtractor
from unit_settings import unit_setting_schema


CLASSES = (RigorousDistillation, CMODistillation, RigorousAbsorber,
           RigorousStripper, RigorousLiquidLiquidExtractor)


def options(**updates):
    result = dict(mesh_tolerance=1e-8, max_iterations=80, max_jacobian_evaluations=80,
                  line_search_steps=16, finite_difference_rel_step=1e-6)
    result.update(updates)
    return result


def test_dogleg_respects_radius_and_decreases_quadratic():
    rng = np.random.default_rng(20261010)
    for _ in range(30):
        J = rng.normal(size=(5, 5)) + 4*np.eye(5)
        f = rng.normal(size=5)
        g = J.T @ f
        newton = np.linalg.solve(J, -f)
        radius = 10**rng.uniform(-2, 1)
        p = _dogleg_step(newton, g, csr_matrix(J), radius)
        assert np.linalg.norm(p) <= radius*(1+1e-12)
        assert -g @ p - .5*np.linalg.norm(J @ p)**2 > 0
        if np.linalg.norm(newton) <= radius:
            np.testing.assert_allclose(p, newton)
    # A shifted linear solve can provide an endpoint that increases the model.
    np.testing.assert_allclose(_dogleg_step(np.array([-3.]), np.array([1.]),
                                          csr_matrix([[1.]]), 10.), [-1.])
    assert _dogleg_step(None, np.zeros(1), csr_matrix([[0.]]), 1.) is None


@pytest.mark.parametrize('cls', CLASSES)
def test_invalid_strategy_fails_before_initialization(cls):
    unit = cls('BAD', None, {'newton_globalization': 'bogus'})
    with pytest.raises(UnitOperationError, match='newton_globalization must be line_search or dogleg'):
        unit.solve({})


@pytest.mark.parametrize('cls', (RigorousDistillation, RigorousLiquidLiquidExtractor))
def test_default_matches_explicit_line_search_and_extractor_step_limit(cls):
    def residual(z):
        return np.array([np.exp(z[0])-2.])
    def jac(z, f, h):
        return csr_matrix([[np.exp(z[0])]]), 0
    default = cls('DEFAULT', None, {})
    explicit = cls('EXPLICIT', None, {'newton_globalization': 'line_search'})
    a = default._sparse_newton_solve(residual, csr_matrix([[1.]]), np.array([0.]), options(), jacobian=jac)
    b = explicit._sparse_newton_solve(residual, csr_matrix([[1.]]), np.array([0.]), options(), jacobian=jac)
    np.testing.assert_array_equal(a['x'], b['x'])
    for key in ('residual_norm', 'iterations', 'function_evaluations', 'jacobian_evaluations'):
        assert a[key] == b[key]
    assert a['success'] and a['newton_globalization'] == 'line_search'
    if cls is RigorousLiquidLiquidExtractor:
        limited = cls('LIMITED', None, {'newton_step_limit': .01})
        result = limited._sparse_newton_solve(residual, csr_matrix([[1.]]), np.array([0.]),
                                              options(max_iterations=1), jacobian=jac)
        assert result['x'][0] <= .01*(1+1e-12)


def test_rejected_domain_trials_reuse_jacobian_and_ignore_line_search_limit():
    unit = RigorousDistillation('DOMAIN', None, {'newton_globalization': 'dogleg'})
    function_calls = jacobian_calls = 0
    def residual(z):
        nonlocal function_calls
        function_calls += 1
        if z[0] > .8:
            raise ThermodynamicsError('trial outside property domain')
        return np.array([np.exp(z[0])-2.])
    def jac(z, f, h):
        nonlocal jacobian_calls
        jacobian_calls += 1
        return csr_matrix([[np.exp(z[0])]]), 0
    answer = unit._sparse_newton_solve(residual, csr_matrix([[1.]]), np.array([0.]),
                                     options(line_search_steps=0), jacobian=jac)
    assert answer['success'] and answer['residual_norm'] < 1e-8
    assert answer['rejected_steps'] > 0
    assert answer['function_evaluations'] == function_calls
    assert answer['jacobian_evaluations'] == jacobian_calls
    assert function_calls > jacobian_calls+1


def test_initial_domain_error_propagates_and_jacobian_budget_is_enforced():
    unit = RigorousDistillation('BUDGET', None, {'newton_globalization': 'dogleg'})
    def invalid(z):
        raise ThermodynamicsError('invalid initial state')
    with pytest.raises(ThermodynamicsError, match='initial state'):
        unit._sparse_newton_solve(invalid, csr_matrix([[1.]]), np.zeros(1), options())
    result = unit._sparse_newton_solve(lambda z: np.exp(z)-2., csr_matrix([[1.]]),
                                      np.zeros(1), options(max_jacobian_evaluations=0))
    assert not result['success']
    assert result['jacobian_evaluations'] == 0 and result['function_evaluations'] == 1
    accepted = unit._sparse_newton_solve(lambda z: z+.1, csr_matrix([[1.]]), np.zeros(1),
        options(max_jacobian_evaluations=0, acceptable_mesh_residual=.2))
    assert accepted['success'] and accepted['residual_norm'] > 1e-8


def test_topology_event_receives_untruncated_direction_and_progress():
    unit = RigorousDistillation('EVENT', None, {'newton_globalization': 'dogleg'})
    class Change(RuntimeError):
        def add_solver_progress(self, **values):
            self.progress = values
    def event(x, f, dx):
        np.testing.assert_allclose(dx, [100.])
        raise Change('topology changed')
    with pytest.raises(Change) as error:
        unit._sparse_newton_solve(lambda z: z-100., csr_matrix([[1.]]), np.zeros(1),
                                 options(), jacobian=lambda x, f, h: (csr_matrix([[1.]]), 0),
                                 step_event=event)
    assert error.value.progress == dict(iterations=1, function_evaluations=1, jacobian_evaluations=1)


def test_cauchy_fallback_when_linear_newton_solve_fails():
    unit = RigorousDistillation('CAUCHY', None, {'newton_globalization': 'dogleg'})
    with patch('scipy.sparse.linalg.spsolve', side_effect=RuntimeError('singular solve')):
        answer = unit._sparse_newton_solve(lambda z: z-1., csr_matrix([[1.]]), np.zeros(1),
            options(), jacobian=lambda z, f, h: (csr_matrix([[1.]]), 0))
    assert answer['success'] and answer['residual_norm'] < 1e-8
    stationary = unit._sparse_newton_solve(lambda z: np.ones(1), csr_matrix([[1.]]), np.zeros(1),
        options(), jacobian=lambda z, f, h: (csr_matrix([[0.]]), 0))
    assert not stationary['success'] and stationary['residual_norm'] == 1.


def test_line_search_counts_nonfinite_rejected_trials():
    calls = 0
    unit = RigorousDistillation('FINITE', None, {})
    def residual(z):
        nonlocal calls
        calls += 1
        return np.array([np.nan if z[0] > .8 else np.exp(z[0])-2.])
    answer = unit._sparse_newton_solve(residual, csr_matrix([[1.]]), np.zeros(1), options(),
        jacobian=lambda z, f, h: (csr_matrix([[np.exp(z[0])]]), 0))
    assert answer['success']
    assert answer['rejected_steps'] >= 1 and answer['function_evaluations'] == calls


@pytest.mark.parametrize('unit_type', ('RigorousDistillation', 'CMODistillation', 'RigorousExtractor',
                                     'RigorousAbsorber', 'RigorousStripper'))
def test_pfd_and_editor_expose_strategy_as_a_numerical_setting(unit_type):
    parsed = parse_pfd(f'UNIT COL : {unit_type}\n    newton_globalization = dogleg\n')
    assert parsed.units[0].params[0].value == 'dogleg'
    dof = DOFAnalyzer(parsed)._analyze_unit(parsed.units[0])
    assert dof.status is SpecificationStatus.OK and dof.dof == 0
    setting = next(s for s in unit_setting_schema(unit_type) if s['name'] == 'newton_globalization')
    assert setting['values'] == ['line_search', 'dogleg']
    assert setting['default'] == 'line_search'
    assert setting['section'] == 'solver' and setting['label'] == 'Newton step strategy'


def probed_case(kind):
    if kind in ('absorber', 'stripper'):
        thermo = create_thermodynamics(['water', 'acetaldehyde', 'nitrogen', 'oxygen'], 'UNIQUAC')
        if kind == 'absorber':
            gas = thermo.calculate_state(298.15, 1.01325, 100.,
                {'acetaldehyde': .1, 'nitrogen': .711, 'oxygen': .189}, phase='vapor', flash=False)
            liquid = thermo.calculate_state(298.15, 1.01325, 400., {'water': 1.}, phase='liquid', flash=False)
            cls, inlets = RigorousAbsorber, {'gas': gas, 'water': liquid}
        else:
            gas = thermo.calculate_state(353.15, 1.01325, 100., {'nitrogen': .79, 'oxygen': .21}, phase='vapor', flash=False)
            liquid = thermo.calculate_state(323.15, 1.01325, 200., {'water': .95, 'acetaldehyde': .05}, phase='liquid', flash=False)
            cls, inlets = RigorousStripper, {'liquid': liquid, 'air': gas}
        return cls('PROBED', thermo, {'N_stages': 5, 'P_drop_per_stage': 0.}), inlets
    if kind == 'cmo':
        thermo = create_thermodynamics(['methanol', 'water'], 'NRTL')
        z = {'methanol': .3, 'water': .7}
        feed = thermo.calculate_state(thermo.bubble_point_T(z, 1.), 1., 100., z, phase='liquid', flash=False)
        return CMODistillation('PROBED', thermo, {'N_stages': 20, 'feed_stage': 15, 'reflux_ratio': 1.,
            'D_to_F': .3, 'P_condenser': 1., 'P_drop_per_stage': 0., 'mesh_tolerance': 1e-7,
            'max_iterations': 160, 'max_jacobian_evaluations': 160}), {'feed': feed}
    thermo = create_thermodynamics(['diethyl ether', 'n-hexane', 'acrylic acid', 'water'], 'UNIFAC')
    def mass_stream(masses):
        flows = {c: mass*1000/thermo.props[c].MW for c, mass in masses.items()}
        total = sum(flows.values())
        return thermo.calculate_state(298.15, 1., total, {c: f/total for c, f in flows.items()}, phase='liquid', flash=False)
    return RigorousLiquidLiquidExtractor('PROBED', thermo, {'N_stages': 3, 'T': 25., 'max_iterations': 80,
        'solver_algorithm': 'equation_oriented', 'semi_analytic_local_thermo_jacobian': False}), {
        'feed': mass_stream({'acrylic acid': 200., 'water': 1500.}),
        'solvent': mass_stream({'diethyl ether': 250., 'n-hexane': 250.})}


class ProblemCaptured(BaseException):
    pass


def capture_problem(unit, inlets):
    captured = {}
    def capture(residual, sparsity, x0, opts, jacobian=None, step_event=None):
        captured.update(residual=residual, sparsity=sparsity, x0=x0.copy(), options=dict(opts), jacobian=jacobian)
        raise ProblemCaptured
    with patch.object(unit, '_sparse_newton_solve', capture), pytest.raises(ProblemCaptured):
        unit.solve(inlets)
    return captured


@pytest.mark.parametrize('kind,noise,seed', (
    ('absorber', .5, 11), ('absorber', .5, 47),
    ('cmo', .5, 29), ('extractor', 1.5, 11),
))
def test_dogleg_recovers_reproducible_probed_starts(kind, noise, seed):
    answers = {}
    initial = None
    for method in ('line_search', 'dogleg'):
        # Isolate thermodynamic/initializer caches between the paired solves,
        # while checking that both receive the same transformed starting point.
        unit, inlets = probed_case(kind)
        problem = capture_problem(unit, inlets)
        if initial is None:
            initial = problem['x0']
        np.testing.assert_array_equal(problem['x0'], initial)
        x0 = initial + noise*np.random.default_rng(seed).standard_normal(initial.size)
        opts = dict(problem['options'], acceptable_mesh_residual=problem['options']['mesh_tolerance'])
        unit.params['newton_globalization'] = method
        answer = unit._sparse_newton_solve(problem['residual'], problem['sparsity'], x0, opts,
                                           jacobian=problem['jacobian'])
        answers[method] = float(np.linalg.norm(problem['residual'](answer['x']), ord=np.inf))
        if method == 'dogleg':
            assert answer['success'], (answer['residual_norm'], answer['message'], answer['jacobian_evaluations'])
    baseline, improved = answers['line_search'], answers['dogleg']
    assert improved < opts['mesh_tolerance']
    # Future improvements to line search are welcome; while it misses tolerance,
    # verify the observed dogleg recovery improves the residual substantially.
    assert baseline < opts['mesh_tolerance'] or improved < baseline*1e-3
    print(f'{kind}, noise={noise}, seed={seed}: line={baseline:.6e}, dogleg={improved:.6e}')


def test_nitrile_overdraw_converges_with_dogleg_from_cheap_start():
    groups = {'acrylonitrile': {68: 1}, 'acetonitrile': {40: 1}, 'water': {16: 1}}
    thermo = create_thermodynamics(['acrylonitrile', 'acetonitrile', 'water'], 'UNIFNIST', None, groups)
    z = {'acrylonitrile': .6, 'acetonitrile': .1, 'water': .3}
    feed = thermo.calculate_state(thermo.bubble_point_T(z, 1.01325), 1.01325, 100., z, phase='liquid', flash=False)
    params = {'N_stages': 80, 'feed_stage': 70, 'reflux_ratio': 10., 'P_condenser': 1.01325,
        'P_drop_per_stage': 0., 'condenser_type': 'total', 'initializer': 'cheap_estimate',
        'mesh_tolerance': 1e-6, 'acceptable_mesh_residual': 1e-6, 'max_iterations': 100,
        'max_jacobian_evaluations': 100, 'finite_difference_rel_step': 1e-6, 'colored_jacobian_fallback': False}
    finder = RigorousDistillation('FINDER', thermo, params)
    candidate = min(finder._vle_azeotrope_candidates(list(z), 1.01325), key=lambda c: c['T'])
    capacity = min(z[c]/x for c, x in candidate['composition'].items() if x > 1e-8)
    params['D_to_F'] = 1.005*capacity
    norms = {}
    for method in ('line_search', 'dogleg'):
        unit = RigorousDistillation('OVERDRAW', thermo, dict(params, newton_globalization=method))
        original = unit._sparse_newton_solve
        def observed(*args, **kwargs):
            answer = original(*args, **kwargs)
            norms[method] = answer['residual_norm']
            return answer
        with patch.object(unit, '_sparse_newton_solve', observed):
            if method == 'line_search':
                try:
                    unit.solve({'feed': feed})
                except UnitOperationError as error:
                    assert 'MESH solve failed' in str(error)
            else:
                result = unit.solve({'feed': feed})
                assert result.performance['mesh_residual'] < 1e-6
                assert result.performance['component_balance_error'] < 1e-6
                assert result.performance['newton_globalization'] == 'dogleg'
                for c in z:
                    output = sum(s.F*s.composition.get(c, 0.) for s in result.outlet_streams.values())
                    assert abs(output-feed.F*z[c])/feed.F < 1e-6
    assert norms['line_search'] < 1e-6 or norms['dogleg'] < norms['line_search']*1e-3
    print(f"nitrile +0.5%: line={norms['line_search']:.6e}, dogleg={norms['dogleg']:.6e}")


@pytest.mark.parametrize('kind', ('absorber', 'stripper', 'extractor', 'cmo'))
def test_public_column_solve_reports_dogleg_and_preserves_balances(kind):
    unit, inlets = probed_case(kind)
    unit.params['newton_globalization'] = 'dogleg'
    result = unit.solve(inlets)
    p = result.performance
    tolerance = unit.get_param('mesh_tolerance', 1e-5 if kind == 'extractor' else 2e-6)
    assert p.get('mesh_residual', p.get('mes_residual')) < tolerance
    assert p['newton_globalization'] == 'dogleg'
    total = sum(s.F for s in inlets.values())
    for c in unit.thermo.components:
        incoming = sum(s.F*s.composition.get(c, 0.) for s in inlets.values())
        outgoing = sum(s.F*s.composition.get(c, 0.) for s in result.outlet_streams.values())
        assert abs(outgoing-incoming)/total < 1e-6


@pytest.mark.parametrize('model_jacobian', (True, False))
def test_partial_mass_cut_and_efficiencies_with_dogleg(model_jacobian):
    thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
    feed = thermo.calculate_state(298.15, 1., 100., {'methanol': .4, 'water': .6}, phase='liquid', flash=False)
    params = {'N_stages': 12, 'feed_stage': 7, 'reflux_ratio': 2., 'D_mass_to_F_mass': .35,
        'P_condenser': 1., 'P_drop_per_stage': 0., 'condenser_type': 'partial',
        'mesh_tolerance': 1e-6, 'acceptable_mesh_residual': 1e-6, 'max_iterations': 100,
        'initializer': 'cheap_estimate', 'stage_efficiency': .7,
        'semi_analytic_flow_jacobian': model_jacobian, 'newton_globalization': 'dogleg'}
    result = RigorousDistillation('PARTIAL', thermo, params).solve({'feed': feed})
    assert result.performance['mesh_residual'] < 1e-6
    assert result.performance['max_murphree_residual'] < 1e-6
    assert result.performance['component_balance_error'] < 1e-6
    output_mass = result.outlet_streams['distillate'].mass_flow()
    assert output_mass/feed.mass_flow() == pytest.approx(.35, abs=1e-6)
    if not model_jacobian:
        assert result.performance['jacobian_method'] == 'colored_finite_difference'


@pytest.mark.parametrize('initializer', ('cmo', 'coarse_rigorous'))
def test_nested_initializer_newton_solves_inherit_strategy(initializer):
    from equilibrium_stage_column import EquilibriumStageColumnMixin
    thermo = create_thermodynamics(['methanol', 'water'], 'NRTL')
    feed = thermo.calculate_state(298.15, 1., 100., {'methanol': .4, 'water': .6}, phase='liquid', flash=False)
    calls = []
    original = EquilibriumStageColumnMixin._sparse_newton_solve
    def observed(self, *args, **kwargs):
        calls.append((self.unit_id, self._newton_globalization()))
        return original(self, *args, **kwargs)
    with patch.object(EquilibriumStageColumnMixin, '_sparse_newton_solve', observed):
        result = RigorousDistillation('NESTED', thermo, {'N_stages': 20, 'feed_stage': 10,
            'reflux_ratio': 2., 'D_to_F': .35, 'P_condenser': 1., 'P_drop_per_stage': 0.,
            'initializer': initializer, 'newton_globalization': 'dogleg', 'mesh_tolerance': 1e-6,
            'acceptable_mesh_residual': 1e-6, 'max_iterations': 100, 'max_jacobian_evaluations': 100}).solve({'feed': feed})
    assert result.performance['mesh_residual'] < 1e-6
    assert calls and all(method == 'dogleg' for _, method in calls)
    assert any(unit_id != 'NESTED' for unit_id, _ in calls)


@pytest.mark.parametrize('case', ('binary_invariant', 'phase_disappearance'))
def test_vlle_dogleg_preserves_active_set_events_and_fugacity_closure(case):
    if case == 'binary_invariant':
        thermo = create_thermodynamics(['water', 'chloroform'], 'UNIFNIST')
        z, pressure, stages = {'water': .5, 'chloroform': .5}, 1.01325, 6
    else:
        thermo = create_thermodynamics(['ethanol', 'water', 'benzene'], 'UNIFAC')
        z, pressure, stages = {'ethanol': .75, 'water': .10, 'benzene': .15}, 1., 12
    feed = thermo.calculate_state(298.15, pressure, 20., z, phase='liquid', flash=False)
    result = RigorousDistillation('VLLE', thermo, {'N_stages': stages, 'feed_stage': stages//2,
        'reflux_ratio': 2., 'D_to_F': .5 if case == 'binary_invariant' else .4,
        'P_condenser': pressure, 'P_drop_per_stage': 0., 'condenser_type': 'total',
        'stage_phase_model': 'VLLE', 'vlle_seed': 'cheap', 'vlle_initial_topology': 'all_vlle',
        'mesh_tolerance': 1e-5, 'acceptable_mesh_residual': 1e-5,
        'max_iterations': 120, 'max_jacobian_evaluations': 120,
        'newton_globalization': 'dogleg'}).solve({'feed': feed})
    p = result.performance
    assert p['newton_globalization'] == 'dogleg'
    assert p['mesh_residual'] < 1e-5 and p['component_balance_error'] < 1e-6
    assert p['vlle_max_log_fugacity_residual'] < 1e-5
    if case == 'binary_invariant':
        assert p['stage_phase_counts'] == [3]*stages
    else:
        assert len(p['vlle_topology_history']) > 1
        assert p['stage_phase_counts'] == [2]*stages


@pytest.mark.parametrize('algorithm_param,algorithm', (
    ('solver_algorithm', 'split_sweep'), ('algorithm', 'nested_lle'),
))
@pytest.mark.parametrize('has_lle', (True, False))
def test_extractor_split_sweep_cannot_silently_ignore_dogleg(algorithm_param, algorithm, has_lle):
    unit, inlets = probed_case('extractor')
    unit.params.pop('solver_algorithm')
    unit.params.update({algorithm_param: algorithm, 'newton_globalization': 'dogleg'})
    if has_lle:
        with pytest.raises(UnitOperationError, match='dogleg requires solver_algorithm=equation_oriented'):
            unit.solve(inlets)
    else:
        with patch.object(unit.thermo, 'liquid_liquid_equilibrium', return_value=(False, {}, {}, 0.)):
            with pytest.raises(UnitOperationError, match='dogleg requires solver_algorithm=equation_oriented'):
                unit.solve(inlets)
