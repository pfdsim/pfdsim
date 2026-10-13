"""Checks that rankings use identical observations and compound weights."""

import numpy as np
import pytest

from compare_methods import compare, disjoint_test_rows


def test_common_population_and_compound_weights():
    rows = [
        {'id': 'a1', 'cas': 'A', 'reference': 10, 'chemical_classes': [],
         'predictions': {'left': {'value': 11}, 'right': {'value': 10.5}}},
        {'id': 'a2', 'cas': 'A', 'reference': 10, 'chemical_classes': [],
         'predictions': {'left': {'value': 13}, 'right': {'value': 10.5}}},
        {'id': 'b', 'cas': 'B', 'reference': 10, 'chemical_classes': [],
         'predictions': {'left': {'value': 10}, 'right': {'value': 12}}},
        {'id': 'c', 'cas': 'C', 'reference': 10, 'chemical_classes': [],
         'predictions': {'left': {'value': 100}, 'right': {'value': None}}},
    ]
    result = compare(rows, ['left', 'right'], 'reference', np.random.default_rng(1729), 100)
    assert result['observations'] == 3
    assert result['compounds'] == 2
    assert result['methods']['left']['compound_equal_mape_percent'] == 10
    assert result['methods']['right']['compound_equal_mape_percent'] == 12.5
    difference = result['paired_differences']['left vs right']
    assert difference['right_minus_left_MAPE_percentage_points'] == 2.5
    assert difference['left_lower_error_compounds'] == 1
    assert difference['right_lower_error_compounds'] == 1


def test_bootstrap_is_reproducible_and_methods_can_be_tied():
    rows = [{'id': str(i), 'reference': 10, 'chemical_classes': [],
             'predictions': {'left': {'value': 10 + i}, 'right': {'value': 10 + i}}}
            for i in range(3)]
    results = [compare(rows, ['left', 'right'], 'reference', np.random.default_rng(1729), 100)
               for _ in range(2)]
    assert results[0] == results[1]
    comparison = results[0]['paired_differences']['left vs right']
    assert comparison['tied_compounds'] == 3
    assert comparison['bootstrap_95_percent_interval'] == pytest.approx([0, 0])


def test_disjoint_test_uses_saved_training_audit_including_rejected_rows():
    # A numerically rejected training observation is absent from predictions,
    # but its identity still disqualifies the same compound from a disjoint test.
    rows = [{'cas': cas, 'fit_role': 'test', 'paper_excluded': False,
             'has_solid_solid_transition': False} for cas in ('overlap', 'new')]
    saved = {'audit': {'identity_resolved_training_CAS': ['overlap']},
             'rows': {'paper': rows}}
    assert disjoint_test_rows(saved) == [rows[1]]
