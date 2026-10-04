from copy import deepcopy
import unittest

from tests.fit_source_comparison import assert_fit_source_equal


class FitSourceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            'metadata': {'model': 'NRTL'},
            'raw_data': {'points': [{'T_K': 300.0, 'x1': 0.25}]},
            'interactions': [{
                'tau12_d': 552.0,
                'fit_evidence': {'T_K': {'ME': 0.005}},
            }],
            'analysis': {
                'evaluation': {'temperature_error_K': {'MAE': 0.5},
                               'y1_error': {'MAE': 0.001}},
                'zero_liquid_azeotropes': [{'T_K': 350.0, 'x1': 0.6}],
            },
        }

    def test_calculated_temperature_and_composition_roundoff_is_accepted(self):
        regenerated = deepcopy(self.source)
        regenerated['interactions'][0]['fit_evidence']['T_K']['ME'] += 1e-12
        regenerated['analysis']['evaluation']['temperature_error_K']['MAE'] += 7e-12
        regenerated['analysis']['evaluation']['y1_error']['MAE'] += 1e-15
        regenerated['analysis']['zero_liquid_azeotropes'][0]['T_K'] += 7e-12
        regenerated['analysis']['zero_liquid_azeotropes'][0]['x1'] += 1e-15
        assert_fit_source_equal(self, regenerated, self.source)

    def test_raw_temperatures_remain_exact(self):
        regenerated = deepcopy(self.source)
        regenerated['raw_data']['points'][0]['T_K'] += 1e-11
        with self.assertRaisesRegex(AssertionError, 'raw_data'):
            assert_fit_source_equal(self, regenerated, self.source)

    def test_raw_compositions_remain_exact(self):
        regenerated = deepcopy(self.source)
        regenerated['raw_data']['points'][0]['x1'] += 1e-15
        with self.assertRaisesRegex(AssertionError, 'raw_data'):
            assert_fit_source_equal(self, regenerated, self.source)

    def test_fit_parameters_remain_exact(self):
        regenerated = deepcopy(self.source)
        regenerated['interactions'][0]['tau12_d'] += 1e-11
        with self.assertRaisesRegex(AssertionError, 'tau12_d'):
            assert_fit_source_equal(self, regenerated, self.source)

    def test_material_temperature_changes_fail(self):
        regenerated = deepcopy(self.source)
        regenerated['analysis']['evaluation']['temperature_error_K']['MAE'] += 1e-8
        with self.assertRaisesRegex(AssertionError, 'temperature_error_K'):
            assert_fit_source_equal(self, regenerated, self.source)

    def test_material_composition_changes_fail(self):
        regenerated = deepcopy(self.source)
        regenerated['analysis']['evaluation']['y1_error']['MAE'] += 1e-10
        with self.assertRaisesRegex(AssertionError, 'y1_error'):
            assert_fit_source_equal(self, regenerated, self.source)

    def test_missing_fields_and_nonfinite_results_fail(self):
        missing = deepcopy(self.source)
        del missing['metadata']['model']
        with self.assertRaises(AssertionError):
            assert_fit_source_equal(self, missing, self.source)
        nonfinite = deepcopy(self.source)
        nonfinite['analysis']['evaluation']['temperature_error_K']['MAE'] = float('nan')
        with self.assertRaises(AssertionError):
            assert_fit_source_equal(self, nonfinite, self.source)
