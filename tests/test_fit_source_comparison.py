from copy import deepcopy
import unittest

from tests.fit_source_comparison import assert_fit_source_equal, assert_source_inputs_equal


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
        regenerated['analysis']['evaluation']['temperature_error_K']['MAE'] += 1e-3
        with self.assertRaisesRegex(AssertionError, 'temperature_error_K'):
            assert_fit_source_equal(self, regenerated, self.source)

    def test_material_composition_changes_fail(self):
        regenerated = deepcopy(self.source)
        regenerated['analysis']['evaluation']['y1_error']['MAE'] += 1e-5
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

    def test_fixed_parameter_report_accepts_only_solver_sized_temperature_noise(self):
        regenerated = deepcopy(self.source)
        regenerated['analysis']['evaluation']['temperature_error_K']['MAE'] += 1e-7
        assert_fit_source_equal(self, regenerated, self.source)

    def test_calculated_caloric_noise_does_not_relax_raw_enthalpy_data(self):
        source = {'raw_data': {'HE_J_mol': 100.},
                  'score': {'Moreau_HE_J_mol': {'298.15': {'MAE': 20.}}}}
        changed = deepcopy(source)
        changed['score']['Moreau_HE_J_mol']['298.15']['MAE'] += 1e-4
        assert_fit_source_equal(self, changed, source)
        changed['raw_data']['HE_J_mol'] += 1e-4
        with self.assertRaisesRegex(AssertionError, 'raw_data'):
            assert_fit_source_equal(self, changed, source)
        changed = deepcopy(source)
        changed['score']['Moreau_HE_J_mol']['298.15']['MAE'] += .01
        with self.assertRaisesRegex(AssertionError, 'Moreau_HE_J_mol'):
            assert_fit_source_equal(self, changed, source)

    def test_refitting_contract_keeps_raw_data_and_temperature_policy_exact(self):
        source = deepcopy(self.source)
        source['interactions'][0].update(model='NRTL', Tmin_K=298.15, tau_tref=313.15)
        regenerated = deepcopy(source)
        regenerated['interactions'][0]['tau12_d'] += 1.
        assert_source_inputs_equal(self, regenerated, source)
        for key in ('Tmin_K', 'tau_tref'):
            changed = deepcopy(regenerated)
            changed['interactions'][0][key] += 1.
            with self.assertRaisesRegex(AssertionError, 'interaction policy'):
                assert_source_inputs_equal(self, changed, source)
        regenerated['raw_data']['points'][0]['T_K'] += .1
        with self.assertRaisesRegex(AssertionError, 'raw_data'):
            assert_source_inputs_equal(self, regenerated, source)

    def test_refitting_contract_keeps_fendu_observations_and_policy_exact(self):
        source = deepcopy(self.source)
        source['analyses'] = {'binary': {
            'source_index': 19, 'source_table': 3,
            'components': ['A', 'B'], 'cas': ['a', 'b'],
            'uniquac_rq': {'A': {'r': 1., 'q': 1.}},
            'mixed_composition_data': {'0.5': [{'T_K': 300., 'p_kPa': 5.823}]},
            'fit_exclusions': [],
        }}
        regenerated = deepcopy(source)
        regenerated['interactions'][0]['tau12_d'] += 1.
        assert_source_inputs_equal(self, regenerated, source)
        for field, value in {
            'source_index': 20, 'source_table': 4, 'components': ['B', 'A'],
            'cas': ['b', 'a'], 'uniquac_rq': {'A': {'r': 2., 'q': 1.}},
            'mixed_composition_data': {'0.5': [{'T_K': 300., 'p_kPa': 6.823}]},
            'fit_exclusions': [{'x1': .5, 'T_K': 300., 'p_kPa': 5.823}],
        }.items():
            with self.subTest(field=field):
                changed = deepcopy(regenerated)
                changed['analyses']['binary'][field] = value
                with self.assertRaisesRegex(AssertionError, field):
                    assert_source_inputs_equal(self, changed, source)
        del regenerated['analyses']['binary']
        with self.assertRaisesRegex(AssertionError, 'analyses'):
            assert_source_inputs_equal(self, regenerated, source)

    def test_published_fendu_report_tolerates_roundoff_but_rejects_corruption(self):
        source = {'published_nrtl_reproduction': {
            'point_count': 65, 'pressure_AARD_percent': 1.,
            'pressure_RMSE_percent': 1.56, 'pressure_max_abs_percent': 2.,
            'pressure_bias_percent': .2, 'AICc': -500.,
        }}
        rounded = deepcopy(source)
        for field in source['published_nrtl_reproduction']:
            if field != 'point_count':
                rounded['published_nrtl_reproduction'][field] += (
                    2e-6 if field == 'AICc' else 1e-7)
        assert_fit_source_equal(self, rounded, source)
        for field in source['published_nrtl_reproduction']:
            with self.subTest(field=field):
                changed = deepcopy(source)
                changed['published_nrtl_reproduction'][field] += 1
                with self.assertRaisesRegex(AssertionError, field):
                    assert_fit_source_equal(self, changed, source)
