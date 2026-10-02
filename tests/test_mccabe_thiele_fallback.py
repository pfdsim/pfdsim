"""Finite-domain regression coverage for binary McCabe-Thiele fallback searches."""

import math
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import pytest

from unit_operations_distillation import McCabeThieleDistillation, UnitOperationError


@pytest.mark.parametrize('auto_feed', [False, True])
def test_binary_fallback_keeps_a_finite_split_when_optimizer_trials_are_infeasible(auto_feed):
    unit = McCabeThieleDistillation('MT', None, {})
    inlet = SimpleNamespace(F=100., composition={'A':.5, 'B':.5})

    def lines(*args):
        return {'x_intersect':.45, 'y_intersect':.65, 'xB_lk':args[7],
                'rectifying_y':lambda x:x, 'stripping_y':lambda x:x}

    # At xD >= 0.7 the stepping pinches and cannot reach the target.
    # The finite grid points below that boundary still exceed two stages,
    # so both solvers must exercise the scalar minimization fallback.
    table = {'x_of_y':lambda y:y if y >= .7 else max(y-.03, 0.)}
    with patch.object(unit, '_composition_bounds', return_value=(.55, .99)), \
         patch.object(unit, '_distillate_flow_for_composition', return_value=40.), \
         patch.object(unit, '_binary_equilibrium_table', return_value=table), \
         patch.object(unit, '_binary_operating_lines', side_effect=lines), \
         warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        if auto_feed:
            result = unit._solve_binary_column_auto_feed(inlet, 'A', 'B', 2, 1., 1., [1., 1.])
        else:
            result = unit._solve_binary_column(inlet, 'A', 'B', 2, 1, 1., 1., [1., 1.])

    assert .55 <= result['x_D']['A'] < .7
    assert math.isfinite(result['stage_error'])
    assert result['stage_error'] > 0
    assert result['D']+result['B'] == pytest.approx(inlet.F)
    assert result['D']*result['x_D']['A']+result['B']*result['x_B']['A'] == pytest.approx(50.)


def test_fallback_improves_the_best_finite_grid_sample():
    def excess(x):
        return 1+(x-.2)**2 if x < .6 else math.inf

    samples = [(x, excess(x)) for x in (.1, .4)]
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        root = McCabeThieleDistillation._minimum_binary_stage_excess(excess, samples, (0., 1.))
    assert excess(root) < min(value for _, value in samples)
    assert excess(root) == pytest.approx(1., abs=1e-12)


@pytest.mark.parametrize('invalid', [math.inf, -math.inf, math.nan])
def test_fallback_retains_grid_sample_when_search_stays_in_an_invalid_region(invalid):
    def excess(x):
        return 1+(x-.03)**2 if x < .1 else invalid

    samples = [(x, excess(x)) for x in (.02, .04)]
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        root = McCabeThieleDistillation._minimum_binary_stage_excess(excess, samples, (0., 1.))
    assert math.isfinite(excess(root))
    assert excess(root) <= min(value for _, value in samples)


def test_fallback_retains_improved_trial_even_if_optimizer_returns_an_invalid_candidate():
    def excess(x):
        return 1+(x-.2)**2 if x < .6 else math.inf

    def optimizer(objective, **kwargs):
        assert objective(.2) == 1.
        assert math.isfinite(objective(.9))
        return SimpleNamespace(x=.9, success=False)

    samples = [(x, excess(x)) for x in (.1, .4)]
    with patch('scipy.optimize.minimize_scalar', side_effect=optimizer):
        root = McCabeThieleDistillation._minimum_binary_stage_excess(excess, samples, (0., 1.))
    assert excess(root) == 1.


@pytest.mark.parametrize('auto_feed', [False, True])
def test_binary_search_still_rejects_a_grid_with_no_finite_split(auto_feed):
    unit = McCabeThieleDistillation('MT', None, {})
    inlet = SimpleNamespace(F=100., composition={'A':.5, 'B':.5})
    with patch.object(unit, '_composition_bounds', return_value=(.55, .99)), \
         patch.object(unit, '_distillate_flow_for_composition', return_value=0.), \
         patch.object(unit, '_binary_equilibrium_table', return_value={'x_of_y':lambda y:y}), \
         pytest.raises(UnitOperationError, match='could not bracket a binary split'):
        if auto_feed:
            unit._solve_binary_column_auto_feed(inlet, 'A', 'B', 2, 1., 1., [1., 1.])
        else:
            unit._solve_binary_column(inlet, 'A', 'B', 2, 1, 1., 1., [1., 1.])
