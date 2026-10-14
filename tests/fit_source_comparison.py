"""Separate approved source inputs from calculated, fixed-parameter reports."""

import math
import re
from copy import deepcopy


def assert_source_inputs_equal(case, actual, expected, *, refitted_antoine=False):
    """Refitting may change coefficients/reports, never data or model policy."""
    case.assertEqual(actual.keys(), expected.keys())
    metadata = deepcopy(actual['metadata'])
    reference = deepcopy(expected['metadata'])
    if refitted_antoine:
        metadata['pure_component_pressure_basis'].pop('coefficients')
        reference['pure_component_pressure_basis'].pop('coefficients')
    case.assertEqual(metadata, reference, 'metadata')
    for key in ('raw_data', 'source_property_basis'):
        if key in expected:
            case.assertEqual(actual[key], expected[key], key)
    if 'analyses' in expected:
        case.assertEqual(actual['analyses'].keys(), expected['analyses'].keys(), 'analyses')
        for name, approved in expected['analyses'].items():
            for field in ('source_index', 'source_table', 'components', 'cas', 'uniquac_rq',
                          'mixed_composition_data', 'fit_exclusions'):
                case.assertEqual(actual['analyses'][name][field], approved[field],
                                 f'analyses.{name}.{field}')
    case.assertEqual(len(actual['interactions']), len(expected['interactions']))
    for generated, approved in zip(actual['interactions'], expected['interactions']):
        def fixed_fields(record):
            return {key: value for key, value in record.items()
                    if not re.fullmatch(r'tau(?:12|21)_[a-g]', key)
                    and key not in {'fit_evidence', 'evidence', 'regularization'}}
        case.assertEqual(fixed_fields(generated), fixed_fields(approved), 'interaction policy')


def _observable_tolerance(path):
    if not path or path[0] in {'raw_data', 'metadata'}:
        return None
    if 'published_nrtl_reproduction' in path:
        if path[-1] in {'pressure_AARD_percent', 'pressure_RMSE_percent',
                        'pressure_max_abs_percent', 'pressure_bias_percent'}:
            return 1e-6
        if path[-1] == 'AICc':
            # AICc contains n*log(RSS/n), magnifying source-Antoine roundoff.
            return 1e-4
    if 'temperature_error_K' in path or (
        'T_K' in path and ('score' in path or 'fit_evidence' in path)
    ):
        # Bubble-T solves terminate at about 1e-7 K. Reports must not demand
        # sub-solver accuracy, even when coefficients are held fixed.
        return 1e-6
    if 'y1_error' in path or 'y1' in path:
        return 1e-8
    if any('HE_J_mol' in str(part) for part in path):
        return 1e-3
    if 'Moreau_pressure_kPa' in path:
        return 1e-7
    if 'P_percent' in path:
        return 1e-6
    if any('gamma' in str(part) and 'infinite_dilution' in str(part) for part in path):
        return 1e-9
    if any(str(part).startswith('interaction_values_') for part in path):
        return 1e-10
    if path[0] == 'analysis' and any(
        isinstance(part, str) and 'azeotropes' in part for part in path
    ):
        if path[-1] == 'T_K':
            return 1e-6
        if path[-1] in {'x1', 'y1'}:
            return 1e-8
    return None


def assert_fit_source_equal(case, actual, expected, path=()):
    """Compare reports evaluated at the SAME approved parameter values.

    This is deliberately unsuitable for comparing two fresh regressions:
    coefficients, input observations and metadata remain exact.
    """
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
        if isinstance(actual, float):
            case.assertTrue(math.isfinite(actual), label)
        tolerance = _observable_tolerance(path)
        if tolerance is not None and isinstance(expected, float):
            case.assertIsInstance(actual, float, label)
            case.assertTrue(math.isfinite(actual) and math.isfinite(expected), label)
            case.assertAlmostEqual(actual, expected, delta=tolerance, msg=label)
        else:
            case.assertEqual(actual, expected, label)
