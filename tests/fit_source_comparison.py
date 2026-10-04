"""Compare regenerated fit sources without pinning numerical root roundoff."""

import math


def _observable_tolerance(path):
    if not path or path[0] in {'raw_data', 'metadata'}:
        return None
    if 'temperature_error_K' in path or (
        'T_K' in path and ('score' in path or 'fit_evidence' in path)
    ):
        return 1e-10
    if 'y1_error' in path:
        return 1e-12
    if path[0] == 'analysis' and any(
        isinstance(part, str) and 'azeotropes' in part for part in path
    ):
        if path[-1] == 'T_K':
            return 1e-10
        if path[-1] in {'x1', 'y1'}:
            return 1e-12
    return None


def assert_fit_source_equal(case, actual, expected, path=()):
    """Keep structure, raw data and coefficients exact; bound observable noise."""
    label = '$' + ''.join(f'[{part}]' if isinstance(part, int) else f'.{part}'
                          for part in path)
    if isinstance(expected, dict):
        case.assertIsInstance(actual, dict, label)
        case.assertEqual(actual.keys(), expected.keys(), label)
        for key in expected:
            assert_fit_source_equal(case, actual[key], expected[key], path+(key,))
    elif isinstance(expected, list):
        case.assertIsInstance(actual, list, label)
        case.assertEqual(len(actual), len(expected), label)
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_fit_source_equal(case, left, right, path+(index,))
    else:
        tolerance = _observable_tolerance(path)
        if tolerance is not None and isinstance(expected, float):
            case.assertIsInstance(actual, float, label)
            case.assertTrue(math.isfinite(actual) and math.isfinite(expected), label)
            case.assertAlmostEqual(actual, expected, delta=tolerance, msg=label)
        else:
            case.assertEqual(actual, expected, label)
