import json
import math
import os
import sys
import unittest
import warnings

from scipy.optimize import brentq, least_squares


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pfd_parser import parse_pfd
from thermodynamics import create_thermodynamics
from chemical_properties import ChemicalDatabase, ChemicalProperties
from cubic_eos import CubicEOS
from unifac import UNIFACModel, get_unifac_groups
from interaction_parameters import (
    _select_eos_kij,
    cas_for_component,
    eos_binary_interaction,
    nrtl_binary_interaction,
    uniquac_binary_interaction,
    uniquac_rq_for_component,
)
from scripts.build_cas_interaction_parameters import (
    build_interaction_payload,
    resolve_component_ids,
    supplemental_acetic_acid_vle_records,
    supplemental_assorted_alcohol_ether_records,
    supplemental_ester_alcohol_fit_records,
    supplemental_isopropanol_water_records,
    supplemental_literature_vle_activity_records,
    supplemental_water_organic_binary_fit_records,
)
from scripts.build_uniquac_rq_parameters import (
    assorted_alcohol_ether_rq_records,
    build_uniquac_rq_payload,
)


class InteractionParameterTests(unittest.TestCase):
    def test_uniquac_matches_chemsep_ethanol_water_reference(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIQUAC')
        gamma = thermo.activity_coefficients(343.15, {'ethanol': 0.252, 'water': 0.748})

        self.assertAlmostEqual(gamma['ethanol'], 1.977454, places=5)
        self.assertAlmostEqual(gamma['water'], 1.1397696, places=5)
        self.assertEqual(
            parse_pfd(
                'PROCESS: UNIQUAC Alias\n'
                'THERMO_METHOD: UNIQUAC_RK\n'
                'COMPONENTS:\n'
                '    ethanol | Ethanol | MW=46.07\n'
                '    water | Water | MW=18.015\n'
            ).metadata.thermo_method,
            'UNIQUAC-RK',
        )
        self.assertEqual(
            parse_pfd(
                'PROCESS: UNIQUAC Alias\n'
                'THERMO_METHOD: UNIQUAC_PENG_ROBINSON\n'
                'COMPONENTS:\n'
                '    ethanol | Ethanol | MW=46.07\n'
                '    water | Water | MW=18.015\n'
            ).metadata.thermo_method,
            'UNIQUAC-PR',
        )

    def test_uniquac_prefers_first_chemsep_duplicate_and_uses_direct_rq_when_available(self):
        interaction = uniquac_binary_interaction('67-56-1', '7732-18-5')
        self.assertIsNotNone(interaction)
        self.assertAlmostEqual(interaction['a12_cal_per_mol'], -337.1298, places=4)
        self.assertAlmostEqual(interaction['a21_cal_per_mol'], 549.2958, places=4)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            thermo = create_thermodynamics(['1-propanol', 'water'], 'UNIQUAC')

        self.assertAlmostEqual(thermo.r['1-propanol'], 2.7800, places=8)
        self.assertAlmostEqual(thermo.q['1-propanol'], 2.5130, places=8)
        self.assertFalse(any('estimated from UNIFAC' in warning for warning in thermo.warnings))
        self.assertFalse(
            any('estimated from UNIFAC' in str(warning.message) for warning in caught)
        )

    def test_fitted_acetic_acid_nrtl_and_uniquac_pairs_are_available(self):
        built_nrtl, nrtl_new_pairs = supplemental_acetic_acid_vle_records(
            [], 'NRTL'
        )
        built_uniquac, uniquac_new_pairs = supplemental_acetic_acid_vle_records(
            [], 'UNIQUAC'
        )
        self.assertEqual(nrtl_new_pairs, 2)
        self.assertEqual(uniquac_new_pairs, 1)
        self.assertEqual(
            {(record['cas1'], record['cas2']) for record in built_nrtl},
            {('64-19-7', '7732-18-5'), ('64-19-7', '141-78-6')},
        )
        self.assertEqual(
            {(record['cas1'], record['cas2']) for record in built_uniquac},
            {('64-19-7', '141-78-6')},
        )

        acid_water = nrtl_binary_interaction('64-19-7', '7732-18-5')
        water_acid = nrtl_binary_interaction('7732-18-5', '64-19-7')
        self.assertAlmostEqual(
            acid_water['a12_cal_per_mol'], -6.1320111905598615, places=12
        )
        self.assertAlmostEqual(
            acid_water['a21_cal_per_mol'], 590.4171250260812, places=12
        )
        self.assertAlmostEqual(acid_water['alpha12'], 0.3, places=12)
        self.assertAlmostEqual(
            water_acid['a12_cal_per_mol'], acid_water['a21_cal_per_mol'], places=12
        )
        self.assertAlmostEqual(
            water_acid['a21_cal_per_mol'], acid_water['a12_cal_per_mol'], places=12
        )
        self.assertIn('1952, 44(8), 1864-1872', acid_water['comment'])

        acid_ester_nrtl = nrtl_binary_interaction('64-19-7', '141-78-6')
        ester_acid_nrtl = nrtl_binary_interaction('141-78-6', '64-19-7')
        self.assertAlmostEqual(
            acid_ester_nrtl['a12_cal_per_mol'], -223.13760955083086, places=12
        )
        self.assertAlmostEqual(
            acid_ester_nrtl['a21_cal_per_mol'], 525.2700041929397, places=12
        )
        self.assertAlmostEqual(acid_ester_nrtl['alpha12'], 0.3, places=12)
        self.assertAlmostEqual(
            ester_acid_nrtl['a12_cal_per_mol'],
            acid_ester_nrtl['a21_cal_per_mol'],
            places=12,
        )
        self.assertIn('1937, 29(6), 709-710', acid_ester_nrtl['comment'])

        acid_ester_uniquac = uniquac_binary_interaction('64-19-7', '141-78-6')
        ester_acid_uniquac = uniquac_binary_interaction('141-78-6', '64-19-7')
        self.assertAlmostEqual(
            acid_ester_uniquac['a12_cal_per_mol'],
            -257.42432034225703,
            places=12,
        )
        self.assertAlmostEqual(
            acid_ester_uniquac['a21_cal_per_mol'],
            465.43722267062594,
            places=12,
        )
        self.assertAlmostEqual(
            ester_acid_uniquac['a12_cal_per_mol'],
            acid_ester_uniquac['a21_cal_per_mol'],
            places=12,
        )
        self.assertIn('1937, 29(6), 709-710', acid_ester_uniquac['comment'])

        legacy_water_uniquac = uniquac_binary_interaction('64-19-7', '7732-18-5')
        self.assertAlmostEqual(
            legacy_water_uniquac['a12_cal_per_mol'], 407.0073, places=4
        )
        self.assertAlmostEqual(
            legacy_water_uniquac['a21_cal_per_mol'], -251.6868, places=4
        )

        nrtl_water = create_thermodynamics(['acetic acid', 'water'], 'NRTL')
        nrtl_ester = create_thermodynamics(
            ['acetic acid', 'ethyl acetate'], 'NRTL'
        )
        self.assertFalse(any('missing' in warning for warning in nrtl_water.warnings))
        self.assertFalse(any('missing' in warning for warning in nrtl_ester.warnings))

    def test_uniquac_estimates_missing_rq_from_unifac(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            thermo = create_thermodynamics(['formaldehyde', 'water'], 'UNIQUAC')

        r_expected, q_expected = UNIFACModel().calculate_r_q(
            get_unifac_groups('formaldehyde')
        )
        self.assertAlmostEqual(thermo.r['formaldehyde'], r_expected, places=8)
        self.assertAlmostEqual(thermo.q['formaldehyde'], q_expected, places=8)
        self.assertTrue(any('estimated from UNIFAC' in warning for warning in thermo.warnings))
        self.assertFalse(
            any('estimated from UNIFAC' in str(warning.message) for warning in caught)
        )

    def test_uniquac_rq_fallback_uses_native_numeric_smiles_groups(self):
        db = ChemicalDatabase(enable_online=False)
        db.chemicals['NATIVE_ETHER'] = ChemicalProperties(
            symbol='NATIVE_ETHER',
            name='Native ether probe',
            formula=None,
            MW=102.18,
            Tb=340.0,
            smiles='CC(C)OC(C)C',
        )

        thermo = create_thermodynamics(
            ['NATIVE_ETHER', 'water'],
            'UNIQUAC',
            db,
        )
        expected_r, expected_q = UNIFACModel().calculate_r_q(
            {1: 4, 3: 1, 26: 1}
        )

        self.assertAlmostEqual(thermo.r['NATIVE_ETHER'], expected_r)
        self.assertAlmostEqual(thermo.q['NATIVE_ETHER'], expected_q)
        self.assertTrue(any(
            "UNIQUAC r/q parameters missing for 'NATIVE_ETHER'" in warning
            for warning in thermo.warnings
        ))

    def test_uniquac_supports_temperature_dependent_direct_tau_parameters(self):
        forward = uniquac_binary_interaction('7697-37-2', '7732-18-5')
        reverse = uniquac_binary_interaction('7732-18-5', '7697-37-2')

        self.assertAlmostEqual(forward['tau12_a'], 0.23936, places=6)
        self.assertAlmostEqual(forward['tau12_b'], -85.163, places=6)
        self.assertAlmostEqual(reverse['tau12_a'], -0.59832, places=6)
        self.assertAlmostEqual(reverse['tau12_b'], 626.111, places=6)

        thermo = create_thermodynamics(['HNO3', 'H2O'], 'UNIQUAC')
        tau = thermo._uniquac_tau_matrix(298.15)

        self.assertAlmostEqual(tau[0][1], math.exp(0.23936 - 85.163 / 298.15))
        self.assertAlmostEqual(tau[1][0], math.exp(-0.59832 + 626.111 / 298.15))

        rk_thermo = create_thermodynamics(['HNO3', 'H2O'], 'UNIQUAC-RK')
        self.assertEqual(rk_thermo.r['HNO3'], thermo.r['HNO3'])
        self.assertEqual(rk_thermo.q['HNO3'], thermo.q['HNO3'])

        pr_thermo = create_thermodynamics(['HNO3', 'H2O'], 'UNIQUAC-PR')
        self.assertEqual(pr_thermo.r['HNO3'], thermo.r['HNO3'])
        self.assertEqual(pr_thermo.q['HNO3'], thermo.q['HNO3'])

    def test_nrtl_supports_anchored_log_and_quadratic_tau_terms(self):
        T = 360.0
        T_ref = 310.0
        override = {
            'model': 'NRTL',
            'component1': 'ethanol',
            'component2': 'water',
            'alpha12': 0.27,
            'tau12_c': 0.2,
            'tau12_d': 45.0,
            'tau12_e': 0.7,
            'tau12_f': 1.0e-3,
            'tau12_g': 2.0e-6,
            'tau21_c': -0.3,
            'tau21_d': 80.0,
            'tau21_e': -0.4,
            'tau21_f': -5.0e-4,
            'tau21_g': -1.0e-6,
            'tau_tref': T_ref,
        }
        thermo = create_thermodynamics(
            ['ethanol', 'water'], 'NRTL', interaction_overrides=[override]
        )
        tau, _ = thermo._nrtl_matrices(T)
        anchored = (T_ref - T) / T + math.log(T / T_ref)
        self.assertAlmostEqual(
            tau[0][1], 0.2 + 45.0 / T + 0.7 * anchored + 1.0e-3 * T + 2.0e-6 * T * T
        )
        reverse = thermo._nrtl_interaction_for_components('water', 'ethanol')
        self.assertEqual(reverse['tau12_g'], -1.0e-6)
        backend = thermo._compiled_activity_backend(T)
        compiled = backend.activity_coefficients([0.4, 0.6], T)
        thermo._compiled_activity_cache['all_temperatures'] = None
        thermo._activity_cache.clear()
        readable = thermo.activity_coefficients(T, {'ethanol': 0.4, 'water': 0.6})
        self.assertAlmostEqual(compiled[0], readable['ethanol'], places=12)
        self.assertAlmostEqual(compiled[1], readable['water'], places=12)

        from compiled_lle import CompiledNRTLLLEBackend
        from compiled_vlle import CompiledActivityVLLEBackend
        lle_backend = CompiledNRTLLLEBackend.from_activity_backend(backend)
        vlle_backend = CompiledActivityVLLEBackend.from_thermo(thermo)
        self.assertIsNotNone(lle_backend)
        self.assertIsNotNone(vlle_backend)
        lle_backend.compile_kernels()
        vlle_backend.compile_kernels()
        V, x, y = vlle_backend.flash_VLE_TP(
            {'ethanol': 0.4, 'water': 0.6}, T, 1.01325
        )
        self.assertTrue(math.isfinite(V))
        self.assertTrue(all(math.isfinite(value) for value in x))
        self.assertTrue(all(math.isfinite(value) for value in y))

        minimal = create_thermodynamics(
            ['ethanol', 'water'],
            'NRTL',
            interaction_overrides=[{
                'model': 'NRTL',
                'component1': 'ethanol',
                'component2': 'water',
                'alpha12': 0.3,
                'tau12_c': 0.1,
                'tau12_d': 20.0,
                'tau21_c': -0.2,
                'tau21_d': 30.0,
            }],
        )
        minimal_interaction = minimal._nrtl_interaction_for_components(
            'ethanol', 'water'
        )
        for field in ('tau12_e', 'tau12_f', 'tau12_g', 'tau21_e', 'tau21_f', 'tau21_g'):
            self.assertEqual(minimal_interaction.get(field, 0.0), 0.0)

    def test_uniquac_supports_anchored_log_linear_and_quadratic_tau_terms(self):
        T = 360.0
        T_ref = 310.0
        override = {
            'model': 'UNIQUAC',
            'component1': 'ethanol',
            'component2': 'water',
            'tau12_a': 0.2,
            'tau12_b': 45.0,
            'tau12_c': 0.7,
            'tau12_d': 1.0e-3,
            'tau12_e': 2.0e-6,
            'tau21_a': -0.3,
            'tau21_b': 80.0,
            'tau21_c': -0.4,
            'tau21_d': -5.0e-4,
            'tau21_e': -1.0e-6,
            'tau_tref': T_ref,
            'use_q_prime': False,
        }
        thermo = create_thermodynamics(
            ['ethanol', 'water'], 'UNIQUAC', interaction_overrides=[override]
        )
        tau = thermo._uniquac_tau_matrix(T)
        anchored = (T_ref - T) / T + math.log(T / T_ref)
        exponent = 0.2 + 45.0 / T + 0.7 * anchored + 1.0e-3 * T + 2.0e-6 * T * T
        self.assertAlmostEqual(tau[0][1], math.exp(exponent))
        reverse = thermo._uniquac_interaction_for_components('water', 'ethanol')
        self.assertEqual(reverse['tau12_e'], -1.0e-6)
        backend = thermo._compiled_activity_backend(T)
        compiled = backend.activity_coefficients([0.4, 0.6], T)
        thermo._compiled_activity_cache['all_temperatures'] = None
        thermo._activity_cache.clear()
        readable = thermo.activity_coefficients(T, {'ethanol': 0.4, 'water': 0.6})
        self.assertAlmostEqual(compiled[0], readable['ethanol'], places=12)
        self.assertAlmostEqual(compiled[1], readable['water'], places=12)

        from compiled_lle import CompiledUNIQUACLLEBackend
        from compiled_vlle import CompiledActivityVLLEBackend
        lle_backend = CompiledUNIQUACLLEBackend.from_activity_backend(backend)
        vlle_backend = CompiledActivityVLLEBackend.from_thermo(thermo)
        self.assertIsNotNone(lle_backend)
        self.assertIsNotNone(vlle_backend)
        lle_backend.compile_kernels()
        vlle_backend.compile_kernels()
        V, x, y = vlle_backend.flash_VLE_TP(
            {'ethanol': 0.4, 'water': 0.6}, T, 1.01325
        )
        self.assertTrue(math.isfinite(V))
        self.assertTrue(all(math.isfinite(value) for value in x))
        self.assertTrue(all(math.isfinite(value) for value in y))

        minimal = create_thermodynamics(
            ['ethanol', 'water'],
            'UNIQUAC',
            interaction_overrides=[{
                'model': 'UNIQUAC',
                'component1': 'ethanol',
                'component2': 'water',
                'tau12_a': 0.1,
                'tau12_b': 20.0,
                'tau21_a': -0.2,
                'tau21_b': 30.0,
            }],
        )
        minimal_interaction = minimal._uniquac_interaction_for_components(
            'ethanol', 'water'
        )
        for field in ('tau12_c', 'tau12_d', 'tau12_e', 'tau21_c', 'tau21_d', 'tau21_e'):
            self.assertEqual(minimal_interaction.get(field, 0.0), 0.0)

    def test_phenolic_temperature_interactions_use_paper_cij_form(self):
        T = 333.15
        tref = 273.15

        nrtl = nrtl_binary_interaction('108-88-3', '108-95-2')
        self.assertIn('phenolic temperature-dependent NRTL', nrtl['comment'])
        self.assertAlmostEqual(nrtl['alpha12'], 0.2)
        self.assertAlmostEqual(nrtl['tau12_c'], -4.3775, places=6)
        self.assertAlmostEqual(
            nrtl['tau12_d'],
            857.14 - tref * -4.3775,
            places=6,
        )
        self.assertAlmostEqual(nrtl['tau21_c'], 2.8430, places=6)
        self.assertAlmostEqual(
            nrtl['tau21_d'],
            -308.41 - tref * 2.8430,
            places=6,
        )

        nrtl_thermo = create_thermodynamics(['toluene', 'phenol'], 'NRTL')
        tau, alpha = nrtl_thermo._nrtl_matrices(T)
        self.assertAlmostEqual(
            tau[0][1],
            (857.14 + -4.3775 * (T - tref)) / T,
            places=8,
        )
        self.assertAlmostEqual(
            tau[1][0],
            (-308.41 + 2.8430 * (T - tref)) / T,
            places=8,
        )
        self.assertAlmostEqual(alpha[0][1], 0.2, places=8)
        self.assertAlmostEqual(alpha[1][0], 0.2, places=8)

        uniquac = uniquac_binary_interaction('108-88-3', '108-95-2')
        self.assertIn('phenolic temperature-dependent UNIQUAC', uniquac['comment'])
        self.assertAlmostEqual(uniquac['tau12_a'], 1.6642, places=6)
        self.assertAlmostEqual(
            uniquac['tau12_b'],
            -369.57 + tref * -1.6642,
            places=6,
        )
        self.assertAlmostEqual(uniquac['tau21_a'], -0.9573, places=6)
        self.assertAlmostEqual(
            uniquac['tau21_b'],
            146.32 + tref * 0.9573,
            places=6,
        )

        uniquac_thermo = create_thermodynamics(['toluene', 'phenol'], 'UNIQUAC')
        tau = uniquac_thermo._uniquac_tau_matrix(T)
        self.assertAlmostEqual(
            tau[0][1],
            math.exp(-(369.57 + -1.6642 * (T - tref)) / T),
            places=8,
        )
        self.assertAlmostEqual(
            tau[1][0],
            math.exp(-(-146.32 + 0.9573 * (T - tref)) / T),
            places=8,
        )

    def test_cesari_phenolic_nrtl_interactions_use_energy_over_rt_form(self):
        R = 8.31446261815324
        T = 323.15

        water_guaiacol = nrtl_binary_interaction('7732-18-5', '90-05-1')
        self.assertIn('Water/Guaiacol Cesari phenolic NRTL', water_guaiacol['comment'])
        self.assertAlmostEqual(water_guaiacol['alpha12'], 0.3)
        self.assertAlmostEqual(water_guaiacol['tau12_c'], 50.56 / R, places=8)
        self.assertAlmostEqual(water_guaiacol['tau12_d'], -2904.47 / R, places=8)
        self.assertAlmostEqual(water_guaiacol['tau21_c'], -41.82 / R, places=8)
        self.assertAlmostEqual(water_guaiacol['tau21_d'], 14631.14 / R, places=8)

        thermo = create_thermodynamics(['ethanol', 'phenol'], 'NRTL')
        tau, alpha = thermo._nrtl_matrices(T)
        self.assertAlmostEqual(
            tau[0][1],
            (-0.04 + 0.01 * T) / (R * T),
            places=8,
        )
        self.assertAlmostEqual(
            tau[1][0],
            (-10.99 + 5.47 * T) / (R * T),
            places=8,
        )
        self.assertAlmostEqual(alpha[0][1], 0.3, places=8)
        self.assertAlmostEqual(alpha[1][0], 0.3, places=8)

        ethanol_phenol = nrtl_binary_interaction('64-17-5', '108-95-2')
        self.assertIn('Ethanol/Phenol Cesari phenolic NRTL', ethanol_phenol['comment'])
        self.assertAlmostEqual(ethanol_phenol['tau12_c'], 0.01 / R, places=8)
        self.assertAlmostEqual(ethanol_phenol['tau21_c'], 5.47 / R, places=8)

        water_cresol = nrtl_binary_interaction('7732-18-5', '95-48-7')
        self.assertIn('Water/2-Cresol phenolic temperature-dependent NRTL', water_cresol['comment'])
        self.assertNotIn('Cesari', water_cresol['comment'])

    def test_diethyl_ether_water_broad_fit_preserves_35c_vlle_anchor(self):
        T = 308.15
        psat_ether = 103.264
        psat_water = 5.633
        x_ether = 0.011948

        nrtl = nrtl_binary_interaction('60-29-7', '7732-18-5')
        self.assertIn('curated water/organic NRTL regression', nrtl['comment'])
        self.assertAlmostEqual(nrtl['alpha12'], 0.3)
        self.assertAlmostEqual(nrtl['tau12_c'], 0.099330046, places=8)
        self.assertAlmostEqual(nrtl['tau12_d'], 635.321936037, places=8)
        self.assertAlmostEqual(nrtl['tau21_c'], 11.754188386, places=8)
        self.assertAlmostEqual(nrtl['tau21_d'], -2538.603239639, places=8)

        thermo = create_thermodynamics(['diethyl ether', 'water'], 'NRTL')
        gamma = thermo.activity_coefficients(
            T,
            {'diethyl ether': x_ether, 'water': 1.0 - x_ether},
        )
        pressure = (
            x_ether * gamma['diethyl ether'] * psat_ether
            + (1.0 - x_ether) * gamma['water'] * psat_water
        )
        y_ether = x_ether * gamma['diethyl ether'] * psat_ether / pressure
        self.assertAlmostEqual(pressure, 104.57, delta=0.1)
        self.assertAlmostEqual(y_ether, 0.9467, delta=0.001)

        uniquac = uniquac_binary_interaction('60-29-7', '7732-18-5')
        self.assertIn('curated water/organic UNIQUAC regression', uniquac['comment'])
        self.assertAlmostEqual(uniquac['tau12_a'], 11.059254211, places=8)
        self.assertAlmostEqual(uniquac['tau12_b'], -2688.791776972, places=8)
        self.assertAlmostEqual(uniquac['tau21_a'], -12.439879602, places=8)
        self.assertAlmostEqual(uniquac['tau21_b'], 2315.818669335, places=8)

    def test_1_butanol_water_temperature_interactions_override_legacy_records(self):
        nrtl = nrtl_binary_interaction('71-36-3', '7732-18-5')
        self.assertIn('temperature-dependent NRTL', nrtl['comment'])
        self.assertAlmostEqual(nrtl['tau12_c'], 3.07626601, places=8)
        self.assertAlmostEqual(nrtl['tau12_d'], -489.80683594, places=8)
        self.assertAlmostEqual(nrtl['tau12_e'], -60.05225941, places=8)
        self.assertAlmostEqual(nrtl['tau21_c'], 4.36302560, places=8)
        self.assertAlmostEqual(nrtl['tau21_d'], -241.22842207, places=8)
        self.assertAlmostEqual(nrtl['tau21_e'], -8.71391481, places=8)
        self.assertAlmostEqual(nrtl['tau_tref'], 298.15, places=8)
        self.assertAlmostEqual(nrtl['alpha12'], 0.45131325, places=8)

        uniquac = uniquac_binary_interaction('71-36-3', '7732-18-5')
        self.assertIn('temperature-dependent UNIQUAC', uniquac['comment'])
        self.assertAlmostEqual(uniquac['tau12_a'], -12.81485434, places=8)
        self.assertAlmostEqual(uniquac['tau12_b'], 1905.74507135, places=8)
        self.assertAlmostEqual(uniquac['tau12_d'], 0.02169653866, places=10)
        self.assertAlmostEqual(uniquac['tau21_a'], 8.68179405, places=8)
        self.assertAlmostEqual(uniquac['tau21_b'], -1368.13065752, places=8)
        self.assertAlmostEqual(uniquac['tau21_d'], -0.01678360690, places=10)

    def test_extended_uniquac_interactions_are_retained_but_disabled(self):
        with open(
            os.path.join(ROOT, 'data', 'uniquac_binary_interactions_cas.json'),
            encoding='utf-8',
        ) as handle:
            payload = json.load(handle)
        self.assertEqual(payload['metadata']['disabled_records'], 39)
        self.assertEqual(payload['metadata']['activity_curated_disabled_records'], 10)
        records = payload['interactions']
        extended = [
            record for record in records
            if record.get('comment', '').startswith('Ethanol/Benzene extended UNIQUAC')
        ]
        self.assertEqual(len(extended), 1)
        self.assertTrue(extended[0]['disabled'])
        self.assertEqual(extended[0]['model_variant'], 'extended_uniquac')
        self.assertTrue(extended[0]['use_q_prime'])
        self.assertAlmostEqual(extended[0]['tau12_a'], -2.5229, places=6)
        self.assertAlmostEqual(extended[0]['tau12_b'], 216.07, places=6)
        self.assertAlmostEqual(extended[0]['tau12_d'], -0.0041, places=6)

        interaction = uniquac_binary_interaction('64-17-5', '71-43-2')
        self.assertIsNotNone(interaction)
        self.assertEqual(interaction['model_variant'], 'standard_uniquac')
        self.assertFalse(interaction['use_q_prime'])
        self.assertNotIn('tau12_a', interaction)
        self.assertAlmostEqual(interaction['a12_cal_per_mol'], -127.9893, places=4)
        self.assertAlmostEqual(interaction['a21_cal_per_mol'], 744.8826, places=4)

        methanol_water = uniquac_binary_interaction('67-56-1', '7732-18-5')
        self.assertEqual(methanol_water['comment'], 'Methanol/Water (Kojima+Kato)')
        self.assertAlmostEqual(methanol_water['a12_cal_per_mol'], -337.1298, places=4)

        methanol_carbon_tet = uniquac_binary_interaction('67-56-1', '56-23-5')
        self.assertEqual(methanol_carbon_tet['comment'], 'Methanol/Tetrachloromethane p18 1/2c')
        self.assertAlmostEqual(methanol_carbon_tet['a12_cal_per_mol'], -95.2921, places=4)

        selected_uniquac = [
            ('7732-18-5', '78-93-3', '2-Butanone/Water p279 1/1a'),
            ('64-17-5', '78-93-3', 'Ethanol/2-Butanone p342 1/2a'),
            ('110-82-7', '67-56-1', 'Methanol/Cyclohexane p211 1/2c'),
            ('142-82-5', '67-56-1', 'Methanol/n-Heptane'),
            ('110-82-7', '64-17-5', 'Ethanol/CycloHexane p441 1/2a'),
            ('64-17-5', '67-64-1', 'Ethanol/Acetone p323 1/2a'),
            ('67-56-1', '64-17-5', 'Methanol/Ethanol p60 1/2c'),
        ]
        for cas1, cas2, comment in selected_uniquac:
            with self.subTest(model='UNIQUAC', pair=(cas1, cas2)):
                self.assertEqual(uniquac_binary_interaction(cas1, cas2)['comment'], comment)

        thermo = create_thermodynamics(['ethanol', 'benzene'], 'UNIQUAC')
        params = thermo._uniquac_parameter_matrices()
        self.assertAlmostEqual(params['q_residual'][0], thermo.q['ethanol'], places=8)
        self.assertAlmostEqual(params['q_residual'][1], thermo.q['benzene'], places=8)

        def y_minus_x(x_ethanol):
            composition = {'ethanol': x_ethanol, 'benzene': 1.0 - x_ethanol}
            T = thermo.bubble_point_T(composition, 1.01325)
            K = thermo.K_values(T, 1.01325, composition)
            y_ethanol = x_ethanol * K['ethanol']
            y_benzene = (1.0 - x_ethanol) * K['benzene']
            return y_ethanol / (y_ethanol + y_benzene) - x_ethanol

        x_azeotrope = brentq(y_minus_x, 0.3, 0.6)
        composition = {'ethanol': x_azeotrope, 'benzene': 1.0 - x_azeotrope}
        T_azeotrope = thermo.bubble_point_T(composition, 1.01325)
        mw_ethanol = thermo.props['ethanol'].MW
        mw_benzene = thermo.props['benzene'].MW
        w_ethanol = (
            x_azeotrope * mw_ethanol
            / (x_azeotrope * mw_ethanol + (1.0 - x_azeotrope) * mw_benzene)
        )

        self.assertAlmostEqual(w_ethanol, 0.324, delta=0.01)
        self.assertAlmostEqual(T_azeotrope - 273.15, 68.2, delta=0.5)

    def test_nrtl_prefers_temperature_dependent_tau_duplicate(self):
        forward = nrtl_binary_interaction('67-64-1', '67-56-1')
        reverse = nrtl_binary_interaction('67-56-1', '67-64-1')

        self.assertNotIn('a12_cal_per_mol', forward)
        self.assertAlmostEqual(forward['tau12_c'], 0.0, places=6)
        self.assertAlmostEqual(forward['tau12_d'], 101.9, places=6)
        self.assertAlmostEqual(forward['tau21_c'], 0.0, places=6)
        self.assertAlmostEqual(forward['tau21_d'], 114.1, places=6)
        self.assertAlmostEqual(reverse['tau12_d'], 114.1, places=6)
        self.assertAlmostEqual(reverse['tau21_d'], 101.9, places=6)

    def test_nrtl_disabled_records_do_not_override_better_legacy_data(self):
        with open(
            os.path.join(ROOT, 'data', 'nrtl_binary_interactions_cas.json'),
            encoding='utf-8',
        ) as handle:
            payload = json.load(handle)
        self.assertEqual(payload['metadata']['disabled_records'], 15)
        self.assertEqual(payload['metadata']['activity_curated_disabled_records'], 9)

        disabled = [
            record for record in payload['interactions']
            if record.get('disabled')
        ]
        self.assertEqual(
            {record['comment'] for record in disabled},
            {
                '1 Chloroform/Benzene p72 1/7',
                'Acetone/Ethanol p312 1/2c',
                'Benzene/Chloroform supplemental NRTL matrix; tau_ij = a_ij + b_ij/T',
                'Benzene/Water supplemental NRTL matrix; tau_ij = a_ij + b_ij/T',
                'Ethanol/2-Butanone p327 1/2c',
                'Ethanol/Cyclohexane p419 1/2c',
                'Methanol/1-Heptane p241 1/2c',
                'Methanol/CycloHexane p243 1/2a',
                'Methanol/Ethanol p55 1/2a',
                'Methanol/Heptane p243 1/2c',
                'p-Xylene/Chloroform supplemental NRTL matrix; tau_ij = a_ij + b_ij/T',
                'p-Xylene/Water supplemental NRTL matrix; tau_ij = a_ij + b_ij/T',
                'Tetrachloromethane/Methanol p279 1/2c',
                'Water/2-Butanone p277 1/1a',
                'Toluene/Water supplemental NRTL matrix; tau_ij = a_ij + b_ij/T',
            },
        )

        interaction = nrtl_binary_interaction('71-43-2', '67-66-3')
        self.assertEqual(interaction['comment'], 'Benzene/Chloroform 1/7')
        self.assertNotIn('tau12_c', interaction)
        self.assertAlmostEqual(interaction['a12_cal_per_mol'], -227.3671, places=4)
        self.assertAlmostEqual(interaction['a21_cal_per_mol'], -86.1025, places=4)
        self.assertIsNone(nrtl_binary_interaction('74-87-3', '71-43-2'))
        methanol_carbon_tet = nrtl_binary_interaction('67-56-1', '56-23-5')
        self.assertEqual(methanol_carbon_tet['comment'], 'Methanol/Tetrachloromethane p18 1/2c')
        self.assertAlmostEqual(methanol_carbon_tet['a12_cal_per_mol'], 378.8254, places=4)

        selected_nrtl = [
            ('7732-18-5', '78-93-3', '2-Butanone/Water p279 1/1a'),
            ('64-17-5', '78-93-3', 'Ethanol/2-Butanone p342 1/2a'),
            ('110-82-7', '67-56-1', 'Methanol/Cyclohexane p211 1/2c'),
            ('142-82-5', '67-56-1', 'Methanol/n-Heptane'),
            ('110-82-7', '64-17-5', 'Ethanol/CycloHexane p441 1/2a'),
        ]
        for cas1, cas2, comment in selected_nrtl:
            with self.subTest(model='NRTL', pair=(cas1, cas2)):
                self.assertEqual(nrtl_binary_interaction(cas1, cas2)['comment'], comment)

    def test_nrtl_supports_temperature_dependent_tau_parameters(self):
        forward = nrtl_binary_interaction('7697-37-2', '7732-18-5')
        reverse = nrtl_binary_interaction('7732-18-5', '7697-37-2')

        self.assertAlmostEqual(forward['tau12_c'], 1.533, places=6)
        self.assertAlmostEqual(forward['tau12_d'], 84.6, places=6)
        self.assertAlmostEqual(reverse['tau12_c'], 0.178, places=6)
        self.assertAlmostEqual(reverse['tau12_d'], -865.8, places=6)

        thermo = create_thermodynamics(['HNO3', 'H2O'], 'NRTL')
        tau, alpha = thermo._nrtl_matrices(298.15)

        self.assertAlmostEqual(tau[0][1], 1.817, places=3)
        self.assertAlmostEqual(tau[1][0], -2.726, places=3)
        self.assertAlmostEqual(alpha[0][1], 0.3, places=6)
        self.assertAlmostEqual(alpha[1][0], 0.3, places=6)

    def test_interaction_parameter_tables_are_cas_keyed(self):
        def load_data(filename):
            with open(os.path.join(ROOT, 'data', filename), encoding='utf-8') as handle:
                return json.load(handle)

        for filename in (
            'eos_binary_interactions_cas.json',
            'nrtl_binary_interactions_cas.json',
            'uniquac_binary_interactions_cas.json',
        ):
            with self.subTest(filename=filename):
                payload = load_data(filename)
                self.assertEqual(payload['metadata']['key_basis'], 'CAS')
                for record in payload['interactions']:
                    self.assertIn('cas1', record)
                    self.assertIn('cas2', record)
                    self.assertNotIn('id1', record)
                    self.assertNotIn('id2', record)

        eos_payload = load_data('eos_binary_interactions_cas.json')
        self.assertEqual(
            {record['comment'] for record in eos_payload['metadata']['skipped']},
            {'Water/HC'},
        )
        self.assertEqual(eos_payload['metadata']['converted_records'], 621)
        self.assertEqual(eos_payload['metadata']['base_converted_records'], 353)
        self.assertEqual(eos_payload['metadata']['supplemental_records'], 2)
        self.assertEqual(eos_payload['metadata']['supplemental_new_pairs'], 1)
        self.assertEqual(eos_payload['metadata']['eos_collapsed_duplicate_groups'], 2)
        self.assertEqual(eos_payload['metadata']['eos_collapsed_duplicate_records'], 2)
        self.assertEqual(eos_payload['metadata']['ipd_hydration_records'], 266)
        self.assertEqual(eos_payload['metadata']['ipd_hydration_new_pairs'], 141)
        self.assertEqual(
            eos_payload['metadata']['ipd_hydration_by_source'],
            {'pr.ipd': 6, 'srk.ipd': 260},
        )
        eos_groups = {}
        for record in eos_payload['interactions']:
            key = (
                record['model'],
                tuple(sorted((record['cas1'], record['cas2']))),
                (record.get('Tmin_K'), record.get('Tmax_K')),
            )
            eos_groups[key] = eos_groups.get(key, 0) + 1
        self.assertTrue(all(count == 1 for count in eos_groups.values()))
        self.assertAlmostEqual(
            eos_binary_interaction('PR', '7727-37-9', '124-38-9', 273.15),
            -0.01445,
            places=6,
        )
        self.assertAlmostEqual(
            eos_binary_interaction('PR', '7783-06-4', '74-98-6'),
            0.02055,
            places=6,
        )
        self.assertAlmostEqual(
            eos_binary_interaction('SRK', '1333-74-0', '7727-37-9', 100.0),
            0.0563,
            places=6,
        )
        self.assertAlmostEqual(
            eos_binary_interaction('SRK', '1333-74-0', '7727-37-9', 90.0),
            0.1141,
            places=6,
        )
        nrtl_payload = load_data('nrtl_binary_interactions_cas.json')
        self.assertEqual(nrtl_payload['metadata']['skipped_records'], 0)
        self.assertEqual(nrtl_payload['metadata']['converted_records'], 464)
        self.assertEqual(nrtl_payload['metadata']['base_converted_records'], 346)
        self.assertEqual(nrtl_payload['metadata']['supplemental_records'], 113)
        self.assertEqual(nrtl_payload['metadata']['supplemental_new_pairs'], 78)
        self.assertEqual(nrtl_payload['metadata']['water_organic_overlay_records'], 7)
        self.assertEqual(nrtl_payload['metadata']['water_organic_overlay_replaced_records'], 4)
        self.assertEqual(nrtl_payload['metadata']['water_organic_overlay_replaced_pairs'], 3)
        self.assertEqual(nrtl_payload['metadata']['literature_vle_overlay_records'], 9)
        self.assertEqual(nrtl_payload['metadata']['literature_vle_overlay_replaced_records'], 0)
        self.assertEqual(nrtl_payload['metadata']['literature_vle_overlay_replaced_pairs'], 0)
        self.assertEqual(nrtl_payload['metadata']['assorted_overlay_records'], 28)
        self.assertEqual(nrtl_payload['metadata']['assorted_overlay_zero_records'], 3)
        self.assertEqual(nrtl_payload['metadata']['assorted_overlay_replaced_records'], 2)
        self.assertEqual(nrtl_payload['metadata']['assorted_overlay_replaced_pairs'], 2)
        self.assertEqual(nrtl_payload['metadata']['isopropanol_water_overlay_records'], 1)
        self.assertEqual(nrtl_payload['metadata']['isopropanol_water_overlay_replaced_records'], 0)
        self.assertEqual(nrtl_payload['metadata']['isopropanol_water_overlay_replaced_pairs'], 0)
        self.assertEqual(nrtl_payload['metadata']['ipd_hydration_records'], 5)
        self.assertEqual(
            nrtl_payload['metadata']['activity_collapsed_equivalent_duplicate_groups'],
            3,
        )
        self.assertEqual(
            nrtl_payload['metadata']['activity_collapsed_equivalent_duplicate_records'],
            3,
        )
        nrtl_pxylene = [
            record for record in nrtl_payload['interactions']
            if record.get('comment') == 'Benzene/pXylene p310 1/7 | 6 Benzene/P-Xylene p310 1/7'
        ]
        self.assertEqual(len(nrtl_pxylene), 1)
        self.assertAlmostEqual(nrtl_pxylene[0]['a12_cal_per_mol'], -50.2635)
        self.assertEqual(len(nrtl_pxylene[0]['duplicate_records']), 2)
        self.assertIsNotNone(nrtl_binary_interaction('106-99-0', '67-56-1'))
        uniquac_payload = load_data('uniquac_binary_interactions_cas.json')
        self.assertEqual(uniquac_payload['metadata']['skipped_records'], 0)
        self.assertEqual(uniquac_payload['metadata']['converted_records'], 443)
        self.assertEqual(uniquac_payload['metadata']['base_converted_records'], 324)
        self.assertEqual(uniquac_payload['metadata']['supplemental_records'], 101)
        self.assertEqual(uniquac_payload['metadata']['supplemental_new_pairs'], 70)
        self.assertEqual(uniquac_payload['metadata']['water_organic_overlay_records'], 7)
        self.assertEqual(uniquac_payload['metadata']['water_organic_overlay_replaced_records'], 3)
        self.assertEqual(uniquac_payload['metadata']['water_organic_overlay_replaced_pairs'], 2)
        self.assertEqual(uniquac_payload['metadata']['literature_vle_overlay_records'], 9)
        self.assertEqual(uniquac_payload['metadata']['literature_vle_overlay_replaced_records'], 3)
        self.assertEqual(uniquac_payload['metadata']['literature_vle_overlay_replaced_pairs'], 3)
        self.assertEqual(uniquac_payload['metadata']['assorted_overlay_records'], 22)
        self.assertEqual(uniquac_payload['metadata']['assorted_overlay_zero_records'], 3)
        self.assertEqual(uniquac_payload['metadata']['assorted_overlay_replaced_records'], 2)
        self.assertEqual(uniquac_payload['metadata']['assorted_overlay_replaced_pairs'], 2)
        self.assertEqual(uniquac_payload['metadata']['isopropanol_water_overlay_records'], 1)
        self.assertEqual(uniquac_payload['metadata']['isopropanol_water_overlay_replaced_records'], 1)
        self.assertEqual(uniquac_payload['metadata']['isopropanol_water_overlay_replaced_pairs'], 1)
        self.assertEqual(uniquac_payload['metadata']['ipd_hydration_records'], 18)
        self.assertEqual(
            uniquac_payload['metadata']['activity_collapsed_equivalent_duplicate_groups'],
            45,
        )
        self.assertEqual(
            uniquac_payload['metadata']['activity_collapsed_equivalent_duplicate_records'],
            45,
        )
        uniquac_methane_water = [
            record for record in uniquac_payload['interactions']
            if record.get('comment') == 'Methane/Water | Water/Methane'
        ]
        self.assertEqual(len(uniquac_methane_water), 1)
        self.assertEqual(len(uniquac_methane_water[0]['duplicate_records']), 2)
        self.assertIsNotNone(uniquac_binary_interaction('616-38-6', '107-21-1'))
        self.assertIsNotNone(nrtl_binary_interaction('67-56-1', '75-25-2'))
        self.assertEqual(cas_for_component('3-Methylpyridien'), '108-99-6')
        self.assertEqual(cas_for_component('p-Xylene'), '106-42-3')

    def test_binary_interaction_builders_match_runtime_json(self):
        resolved, unresolved = resolve_component_ids()
        cases = (
            (
                'eos_binary_interactions.json',
                'eos_binary_interactions_cas.json',
            ),
            (
                'nrtl_binary_interactions.json',
                'nrtl_binary_interactions_cas.json',
            ),
            (
                'uniquac_binary_interactions.json',
                'uniquac_binary_interactions_cas.json',
            ),
        )
        for source_name, runtime_name in cases:
            with self.subTest(runtime_name=runtime_name):
                built = build_interaction_payload(
                    source_name,
                    resolved,
                    unresolved,
                )
                with open(
                    os.path.join(ROOT, 'data', runtime_name),
                    encoding='utf-8',
                ) as handle:
                    runtime_text = handle.read()
                built_text = json.dumps(built, indent=2, sort_keys=True) + '\n'
                self.assertEqual(built_text, runtime_text)

    def test_uniquac_rq_builder_matches_runtime_json(self):
        built = build_uniquac_rq_payload()
        with open(
            os.path.join(ROOT, 'data', 'uniquac_rq_cas.json'),
            encoding='utf-8',
        ) as handle:
            runtime_text = handle.read()
        built_text = json.dumps(built, indent=2, sort_keys=True) + '\n'
        self.assertEqual(built_text, runtime_text)
        self.assertEqual(built['metadata']['component_count'], 73)
        expected = {
            '2-butanol': (3.45, 3.04),
            'n-propyl acetate': (4.153, 3.656),
            '1-pentanol': (4.129, 3.592),
            '2-pentanol': (4.283, 3.556),
            '3-methyl-1-butanol': (4.273, 3.478),
            'butyric acid': (3.5514, 3.15242),
            '2-octanol': (6.15128, 5.20828),
            'ethyl propanoate': (4.1535, 3.6559),
            'ethyl butanoate': (4.8279, 4.19632),
            'diisopropyl ether': (4.7421, 4.088),
            'anisole': (4.1668, 3.208),
            'isopropyl acetate': (4.1523, 3.652),
            '2-methoxyethanol': (3.4938, 3.368),
            'tetrahydrofuran': (2.9415, 2.72),
            'dipropyl ether': (4.7437, 4.096),
            'dibutyl ether': (6.0925, 5.176),
            'methyl isobutyl ketone': (4.596, 3.952),
        }
        for component, (expected_r, expected_q) in expected.items():
            with self.subTest(component=component):
                record = uniquac_rq_for_component(component)
                self.assertIsNotNone(record)
                self.assertAlmostEqual(record['r'], expected_r, places=8)
                self.assertAlmostEqual(record['q'], expected_q, places=8)
                self.assertNotIn('estimated', record['source'].lower())

        # The assorted source fills gaps only; established records retain
        # their pre-existing database values even when the fit source lists
        # a slightly different structural basis.
        self.assertEqual(
            assorted_alcohol_ether_rq_records(
                set(built['components']) - {
                    '100-66-3', '108-21-4', '109-86-4',
                    '109-99-9', '111-43-3', '142-96-1',
                }
            ),
            [
                record for record in assorted_alcohol_ether_rq_records(set())
                if record['cas'] in {
                    '100-66-3', '108-21-4', '109-86-4',
                    '109-99-9', '111-43-3', '142-96-1',
                }
            ],
        )
        self.assertAlmostEqual(uniquac_rq_for_component('methanol')['r'], 1.43)
        self.assertAlmostEqual(uniquac_rq_for_component('1-butanol')['q'], 3.048)

    def test_water_organic_binary_fits_replace_existing_runtime_pairs(self):
        source_path = os.path.join(
            ROOT,
            'data',
            'source',
            'water_organic_binary_fits.json',
        )
        with open(source_path, encoding='utf-8') as handle:
            source = json.load(handle)

        resolved, unresolved = resolve_component_ids()
        for model in ('NRTL', 'UNIQUAC'):
            records, new_pairs, covered_pairs = (
                supplemental_water_organic_binary_fit_records([], model)
            )
            self.assertEqual(len(records), 7)
            self.assertEqual(new_pairs, 7)
            self.assertEqual(len(covered_pairs), 7)
            built = build_interaction_payload(
                f'{model.lower()}_binary_interactions.json',
                resolved,
                unresolved,
            )
            self.assertEqual(built['metadata']['water_organic_overlay_records'], 7)
            by_pair = {
                tuple(sorted((record['cas1'], record['cas2']))): record
                for record in records
            }
            for fit in source['pairs']:
                with self.subTest(model=model, pair=fit['pair_id']):
                    pair = tuple(sorted((fit['cas1'], fit['cas2'])))
                    matches = [
                        record for record in built['interactions']
                        if tuple(sorted((record['cas1'], record['cas2']))) == pair
                    ]
                    self.assertEqual(len(matches), 1)
                    record = matches[0]
                    expected_record = dict(by_pair[pair])
                    if model == 'NRTL':
                        optional_fields = (
                            'tau12_e', 'tau12_f', 'tau12_g',
                            'tau21_e', 'tau21_f', 'tau21_g',
                        )
                        record = {
                            **record,
                            **{field: record.get(field, 0.0) for field in optional_fields},
                            'tau_tref': record.get('tau_tref', 298.15),
                        }
                        expected_record = {
                            **expected_record,
                            **{
                                field: expected_record.get(field, 0.0)
                                for field in optional_fields
                            },
                            'tau_tref': expected_record.get('tau_tref', 298.15),
                        }
                    else:
                        optional_fields = (
                            'tau12_c', 'tau12_d', 'tau12_e',
                            'tau21_c', 'tau21_d', 'tau21_e',
                        )
                        record = {
                            **record,
                            **{field: record.get(field, 0.0) for field in optional_fields},
                            'tau_tref': record.get('tau_tref', 298.15),
                        }
                        expected_record = {
                            **expected_record,
                            **{
                                field: expected_record.get(field, 0.0)
                                for field in optional_fields
                            },
                            'tau_tref': expected_record.get('tau_tref', 298.15),
                        }
                    self.assertEqual(record, expected_record)
                    self.assertEqual(record['Tmin_K'], fit['temperature_range_K']['Tmin'])
                    self.assertEqual(record['Tmax_K'], fit['temperature_range_K']['Tmax'])
                    if model == 'NRTL':
                        runtime = nrtl_binary_interaction(fit['cas1'], fit['cas2'])
                        reverse = nrtl_binary_interaction(fit['cas2'], fit['cas1'])
                        self.assertAlmostEqual(
                            runtime['tau12_c'],
                            fit['nrtl']['parameters']['tau12_c'],
                        )
                        self.assertAlmostEqual(
                            reverse['tau12_c'],
                            fit['nrtl']['parameters']['tau21_c'],
                        )
                        self.assertAlmostEqual(
                            runtime['tau_tref'], fit['nrtl']['tau_tref_K']
                        )
                    else:
                        runtime = uniquac_binary_interaction(fit['cas1'], fit['cas2'])
                        reverse = uniquac_binary_interaction(fit['cas2'], fit['cas1'])
                        self.assertAlmostEqual(
                            runtime['tau12_a'],
                            fit['uniquac']['parameters']['tau12_a'],
                        )
                        self.assertAlmostEqual(
                            reverse['tau12_a'],
                            fit['uniquac']['parameters']['tau21_a'],
                        )

        dipe_rq = uniquac_rq_for_component('diisopropyl ether')
        self.assertEqual(dipe_rq['source'].split(';', 1)[0], 'water_organic_binary_fits.json')
        self.assertAlmostEqual(dipe_rq['r'], 4.7421)
        self.assertAlmostEqual(dipe_rq['q'], 4.088)
        dipe_thermo = create_thermodynamics(
            ['water', 'diisopropyl ether'],
            'UNIQUAC',
        )
        self.assertFalse(any('r/q parameters missing' in warning for warning in dipe_thermo.warnings))

    def test_literature_vle_activity_overlay_retains_both_models(self):
        expected_pairs = {
            ('75-07-0', '7732-18-5'),
            ('64-19-7', '79-09-4'),
            ('64-19-7', '79-10-7'),
            ('79-09-4', '79-10-7'),
            ('79-10-7', '7732-18-5'),
            ('79-09-4', '7732-18-5'),
            ('108-10-1', '7732-18-5'),
            ('79-10-7', '108-10-1'),
            ('79-09-4', '108-10-1'),
        }
        zero_pair = tuple(sorted(('79-09-4', '79-10-7')))

        for model in ('NRTL', 'UNIQUAC'):
            with self.subTest(model=model):
                records, new_pairs, covered_pairs = (
                    supplemental_literature_vle_activity_records([], model)
                )
                self.assertEqual(len(records), 9)
                self.assertEqual(new_pairs, 9)
                self.assertEqual(covered_pairs, {
                    tuple(sorted(pair)) for pair in expected_pairs
                })
                by_pair = {
                    tuple(sorted((record['cas1'], record['cas2']))): record
                    for record in records
                }
                zero = by_pair[zero_pair]
                self.assertEqual(
                    zero['fit_status'],
                    'recommended_defensible_zero_interaction',
                )
                self.assertIn(
                    'well below the experimental error margin',
                    zero['comment'],
                )
                if model == 'NRTL':
                    self.assertEqual(zero['tau12_c'], 0.0)
                    self.assertEqual(zero['tau12_d'], 0.0)
                    self.assertEqual(zero['tau21_c'], 0.0)
                    self.assertEqual(zero['tau21_d'], 0.0)
                    acetaldehyde = by_pair[
                        tuple(sorted(('75-07-0', '7732-18-5')))
                    ]
                    self.assertAlmostEqual(acetaldehyde['tau12_c'], 5.25487)
                    self.assertAlmostEqual(acetaldehyde['tau12_d'], -1310.86)
                    water_mibk = by_pair[
                        tuple(sorted(('7732-18-5', '108-10-1')))
                    ]
                    self.assertAlmostEqual(water_mibk['tau12_c'], 28.0912309)
                    self.assertAlmostEqual(water_mibk['tau12_f'], -0.0358249005)
                else:
                    self.assertEqual(zero['tau12_a'], 0.0)
                    self.assertEqual(zero['tau12_b'], 0.0)
                    self.assertEqual(zero['tau21_a'], 0.0)
                    self.assertEqual(zero['tau21_b'], 0.0)
                    acetaldehyde = by_pair[
                        tuple(sorted(('75-07-0', '7732-18-5')))
                    ]
                    self.assertAlmostEqual(acetaldehyde['tau12_a'], 3.71393)
                    self.assertAlmostEqual(acetaldehyde['tau12_b'], -1316.43)
                    water_mibk = by_pair[
                        tuple(sorted(('7732-18-5', '108-10-1')))
                    ]
                    self.assertEqual(water_mibk['tau12_c'], 0.0)
                    self.assertAlmostEqual(water_mibk['tau12_d'], 0.01411794871)

                acrylic_mibk = by_pair[
                    tuple(sorted(('79-10-7', '108-10-1')))
                ]
                propionic_mibk = by_pair[
                    tuple(sorted(('79-09-4', '108-10-1')))
                ]
                self.assertEqual(acrylic_mibk['Tmin_K'], 352.5)
                self.assertEqual(acrylic_mibk['Tmax_K'], 373.75)
                self.assertNotIn('Tmin_K', propionic_mibk)
                self.assertEqual(
                    acrylic_mibk['fit_vapor_treatment']['associated_component'],
                    'Acrylic acid',
                )
                self.assertEqual(
                    propionic_mibk['fit_vapor_treatment']['delta_H_J_per_mol'],
                    -63490.0,
                )
                self.assertEqual(
                    acrylic_mibk['fit_status'],
                    'validated_secondary_vdm_literature_interaction'
                    if model == 'NRTL'
                    else 'recommended_vdm_literature_interaction',
                )

                resolved, unresolved = resolve_component_ids()
                built = build_interaction_payload(
                    f'{model.lower()}_binary_interactions.json',
                    resolved,
                    unresolved,
                )
                self.assertEqual(
                    built['metadata']['literature_vle_overlay_records'], 9
                )
                for pair in covered_pairs:
                    matches = [
                        record for record in built['interactions']
                        if tuple(sorted((record['cas1'], record['cas2']))) == pair
                    ]
                    self.assertEqual(len(matches), 1)

                if model == 'NRTL':
                    runtime = nrtl_binary_interaction('7732-18-5', '108-10-1')
                    reverse = nrtl_binary_interaction('108-10-1', '7732-18-5')
                    acid_runtime = nrtl_binary_interaction('79-09-4', '108-10-1')
                    self.assertAlmostEqual(runtime['tau12_f'], -0.0358249005)
                    self.assertAlmostEqual(reverse['tau12_f'], -0.0231481413)
                    self.assertAlmostEqual(acid_runtime['tau12_d'], 75.00358)
                else:
                    runtime = uniquac_binary_interaction('7732-18-5', '108-10-1')
                    reverse = uniquac_binary_interaction('108-10-1', '7732-18-5')
                    acid_runtime = uniquac_binary_interaction('79-09-4', '108-10-1')
                    self.assertAlmostEqual(runtime['tau12_d'], 0.01411794871)
                    self.assertAlmostEqual(reverse['tau12_d'], -0.005833157694)
                    self.assertAlmostEqual(acid_runtime['tau12_b'], 76.83818)

                thermo = create_thermodynamics(
                    ['water', 'methyl isobutyl ketone'],
                    model,
                )
                gamma = thermo.activity_coefficients(
                    360.0,
                    {'water': 0.5, 'methyl isobutyl ketone': 0.5},
                )
                self.assertTrue(all(
                    math.isfinite(value) and value > 0.0
                    for value in gamma.values()
                ))

    def test_assorted_alcohol_ether_overlay_uses_only_recommended_models(self):
        excluded_pair = tuple(sorted(('67-63-0', '7732-18-5')))
        zero_pairs = {
            tuple(sorted(pair)) for pair in (
                ('110-54-3', '142-96-1'),
                ('111-65-9', '142-96-1'),
                ('60-29-7', '142-96-1'),
            )
        }
        expected_counts = {'NRTL': 28, 'UNIQUAC': 22}
        for model in ('NRTL', 'UNIQUAC'):
            with self.subTest(model=model):
                records, new_pairs, covered_pairs = (
                    supplemental_assorted_alcohol_ether_records([], model)
                )
                self.assertEqual(len(records), expected_counts[model])
                self.assertEqual(new_pairs, expected_counts[model])
                self.assertEqual(len(covered_pairs), expected_counts[model])
                self.assertNotIn(excluded_pair, covered_pairs)
                zero_records = [
                    record for record in records
                    if record['fit_status']
                    == 'recommended_defensible_zero_interaction'
                ]
                self.assertEqual(len(zero_records), 3)
                self.assertEqual({
                    tuple(sorted((record['cas1'], record['cas2'])))
                    for record in zero_records
                }, zero_pairs)
                for record in zero_records:
                    self.assertIn(';', record['source'])
                    self.assertIn(
                        'near-ideal', record['comment'].lower()
                    )
                for record in records:
                    vapor = record['fit_vapor_treatment']
                    self.assertEqual(
                        vapor['type'],
                        'likely_HOC_or_equivalent_association_correction',
                    )
                    self.assertEqual(
                        vapor['certainty'],
                        'inferred_not_confirmed_for_every_source',
                    )
                    self.assertIn('likely', vapor['note'])
                    self.assertIn('not confirmed uniformly', vapor['note'])
                if model == 'UNIQUAC':
                    self.assertNotIn(
                        tuple(sorted(('67-56-1', '142-96-1'))),
                        covered_pairs,
                    )
                    self.assertTrue(all(
                        not record['use_q_prime'] for record in records
                    ))

                resolved, unresolved = resolve_component_ids()
                built = build_interaction_payload(
                    f'{model.lower()}_binary_interactions.json',
                    resolved,
                    unresolved,
                )
                self.assertEqual(
                    built['metadata']['assorted_overlay_records'],
                    expected_counts[model],
                )
                self.assertEqual(
                    built['metadata']['assorted_overlay_zero_records'], 3
                )
                self.assertEqual(
                    built['metadata']['assorted_overlay_replaced_pairs'], 2
                )
                for pair in covered_pairs:
                    matches = [
                        record for record in built['interactions']
                        if tuple(sorted((record['cas1'], record['cas2']))) == pair
                    ]
                    self.assertEqual(len(matches), 1)
                if model == 'NRTL':
                    runtime = nrtl_binary_interaction('60-29-7', '67-56-1')
                    self.assertAlmostEqual(runtime['tau12_c'], -0.557447)
                    self.assertAlmostEqual(runtime['tau12_d'], 710.993)
                    zero_runtime = nrtl_binary_interaction(
                        '110-54-3', '142-96-1'
                    )
                    self.assertEqual(zero_runtime['tau12_c'], 0.0)
                    self.assertEqual(zero_runtime['tau21_c'], 0.0)
                else:
                    runtime = uniquac_binary_interaction('60-29-7', '67-56-1')
                    self.assertAlmostEqual(runtime['tau12_a'], 3.34369)
                    self.assertAlmostEqual(runtime['tau12_b'], -1594.29)
                    zero_runtime = uniquac_binary_interaction(
                        '110-54-3', '142-96-1'
                    )
                    self.assertEqual(zero_runtime['tau12_a'], 0.0)
                    self.assertEqual(zero_runtime['tau21_a'], 0.0)

    def test_isopropanol_water_broad_range_overlay(self):
        pair = ('67-63-0', '7732-18-5')
        expected_replacements = {'NRTL': 0, 'UNIQUAC': 1}
        for model in ('NRTL', 'UNIQUAC'):
            with self.subTest(model=model):
                records, new_pairs, covered_pairs = (
                    supplemental_isopropanol_water_records([], model)
                )
                self.assertEqual(len(records), 1)
                self.assertEqual(new_pairs, 1)
                self.assertEqual(covered_pairs, {pair})
                record = records[0]
                self.assertEqual(record['Tmin_K'], 308.15)
                self.assertEqual(record['Tmax_K'], 423.15)
                for source_name in (
                    'Sada and Morisue (1975)',
                    'Wu, Hagewiesche and Sandler (1988)',
                    'Moioli et al. (2021)',
                    'Barbieri et al. (2024)',
                    'Wilson and Simons (1952)',
                    'Barr-David and Dodge (1959)',
                ):
                    self.assertIn(source_name, record['source'])
                self.assertIn('B12=-220 cm^3/mol', record['comment'])
                self.assertEqual(
                    record['fit_vapor_treatment']['cross_second_virial']
                    ['B12_cm3_per_mol'],
                    -220.0,
                )
                self.assertEqual(
                    record['source_file'],
                    'data/source/isopropanol_water_interactions.json',
                )

                resolved, unresolved = resolve_component_ids()
                built = build_interaction_payload(
                    f'{model.lower()}_binary_interactions.json',
                    resolved,
                    unresolved,
                )
                self.assertEqual(
                    built['metadata']['isopropanol_water_overlay_records'], 1
                )
                self.assertEqual(
                    built['metadata']
                    ['isopropanol_water_overlay_replaced_records'],
                    expected_replacements[model],
                )
                matches = [
                    item for item in built['interactions']
                    if tuple(sorted((item['cas1'], item['cas2']))) == pair
                ]
                self.assertEqual(len(matches), 1)
                self.assertTrue(
                    matches[0]['source'].startswith(
                        'isopropanol_water_interactions.json; '
                    )
                )
                if model == 'NRTL':
                    self.assertAlmostEqual(record['tau12_c'], -0.83315)
                    self.assertAlmostEqual(record['tau21_f'], -0.0203443)
                    runtime = nrtl_binary_interaction(*pair)
                    reverse = nrtl_binary_interaction(*reversed(pair))
                    self.assertAlmostEqual(runtime['tau12_c'], -0.83315)
                    self.assertAlmostEqual(runtime['tau21_f'], -0.0203443)
                    self.assertAlmostEqual(reverse['tau12_f'], -0.0203443)
                else:
                    self.assertFalse(record['use_q_prime'])
                    self.assertAlmostEqual(record['tau12_a'], 0.31849185)
                    self.assertAlmostEqual(record['tau21_d'], 0.001803034)
                    runtime = uniquac_binary_interaction(*pair)
                    reverse = uniquac_binary_interaction(*reversed(pair))
                    self.assertAlmostEqual(runtime['tau12_a'], 0.31849185)
                    self.assertAlmostEqual(runtime['tau21_d'], 0.001803034)
                    self.assertAlmostEqual(reverse['tau12_d'], 0.001803034)

    def test_runtime_interaction_records_keep_provenance_compact(self):
        forbidden = {
            'source_ids',
            'source_pair_id',
            'fit_provenance',
            'fit_evidence',
            'fit_metrics',
            'model_selection',
            'fit_validity',
            'fit_temperature_range',
            'fit_protocol',
            'fit_and_validation',
            'fit_assessment',
            'uniquac_r_q_use',
            'zero_interaction_basis',
            'parameter_treatment',
            'recommended',
        }
        resolved, unresolved = resolve_component_ids()
        for model in ('NRTL', 'UNIQUAC'):
            with self.subTest(model=model):
                payload = build_interaction_payload(
                    f'{model.lower()}_binary_interactions.json',
                    resolved,
                    unresolved,
                )
                for record in payload['interactions']:
                    self.assertFalse(
                        forbidden & set(record),
                        msg=f"{model} {record.get('comment', '')}",
                    )

    def test_all_recommended_ester_interactions_are_built_and_used(self):
        source_path = os.path.join(
            ROOT,
            'data',
            'source',
            'ester_alcohol_nrtl_uniquac_recommended_fits.json',
        )
        with open(source_path, encoding='utf-8') as handle:
            source = json.load(handle)
        for model in ('NRTL', 'UNIQUAC'):
            built, new_pairs = supplemental_ester_alcohol_fit_records([], model)
            self.assertEqual(len(built), 17)
            self.assertEqual(new_pairs, 17)
            by_pair = {record['comment'].split(' recommended ', 1)[0]: record for record in built}
            fits = [
                fit for fit in source['recommended_fits']
                if fit['model'] == model
            ]
            self.assertEqual(len(fits), 17)
            for fit in fits:
                with self.subTest(model=model, pair=fit['pair']):
                    record = by_pair[fit['pair']]
                    self.assertEqual(record['Tmin_K'], fit['Tmin_K'])
                    self.assertEqual(record['Tmax_K'], fit['Tmax_K'])
                    self.assertEqual(
                        record['fit_vapor_treatment'], fit['vapor_treatment']
                    )
                    self.assertEqual(record['fit_points'], fit['n_points'])
                    self.assertEqual(record['fit_sources'], fit['n_sources'])
                    if model == 'NRTL':
                        self.assertAlmostEqual(record['tau12_c'], fit['A12'])
                        self.assertAlmostEqual(record['tau12_d'], fit['B12_K'])
                        self.assertAlmostEqual(record['tau21_c'], fit['A21'])
                        self.assertAlmostEqual(record['tau21_d'], fit['B21_K'])
                        runtime = nrtl_binary_interaction(
                            record['cas1'], record['cas2']
                        )
                        self.assertAlmostEqual(runtime['tau12_c'], fit['A12'])
                    else:
                        self.assertAlmostEqual(record['tau12_a'], -fit['A12'])
                        self.assertAlmostEqual(record['tau12_b'], -fit['B12_K'])
                        self.assertAlmostEqual(record['tau21_a'], -fit['A21'])
                        self.assertAlmostEqual(record['tau21_b'], -fit['B21_K'])
                        runtime = uniquac_binary_interaction(
                            record['cas1'], record['cas2']
                        )
                        self.assertAlmostEqual(runtime['tau12_a'], -fit['A12'])

    def test_ethyl_acetate_water_25c_nrtl_solubilities(self):
        thermo = create_thermodynamics(['ethyl acetate', 'water'], 'NRTL')
        overall = {
            'ethyl acetate': 0.728342018634107,
            'water': 0.271657981365893,
        }
        split, ester_rich, aqueous, ester_rich_fraction = (
            thermo.liquid_liquid_equilibrium(
                overall,
                298.15,
                max_iter=500,
                tol=1.0e-10,
            )
        )

        self.assertTrue(split)
        self.assertAlmostEqual(
            ester_rich['water'], 0.13887931039065995, places=8
        )
        self.assertAlmostEqual(
            aqueous['ethyl acetate'], 0.016026170013681607, places=8
        )
        self.assertAlmostEqual(
            ester_rich['ethyl acetate'], 0.8611206896093401, places=8
        )
        self.assertAlmostEqual(
            aqueous['water'], 0.9839738299863184, places=8
        )
        self.assertAlmostEqual(ester_rich_fraction, 0.15711694715374797, places=8)

    def test_ethyl_formate_methanol_nrtl_azeotrope_matches_literature(self):
        thermo = create_thermodynamics(['ethyl formate', 'methanol'], 'NRTL')

        def decode(values):
            ethyl_formate = 1.0 / (1.0 + math.exp(-values[0]))
            return (
                {
                    'ethyl formate': ethyl_formate,
                    'methanol': 1.0 - ethyl_formate,
                },
                values[1],
            )

        def residual(values):
            composition, temperature = decode(values)
            k_values = thermo.K_values(temperature, 1.0, composition)
            return (
                math.log(k_values['ethyl formate']),
                math.log(k_values['methanol']),
            )

        solutions = []
        for initial_fraction in (0.2, 0.5, 0.8):
            solution = least_squares(
                residual,
                (
                    math.log(initial_fraction / (1.0 - initial_fraction)),
                    324.0,
                ),
                bounds=((-12.0, 280.0), (12.0, 380.0)),
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
            )
            composition, temperature = decode(solution.x)
            solutions.append((composition, temperature))

        reference_composition, reference_temperature = solutions[0]
        for composition, temperature in solutions[1:]:
            self.assertAlmostEqual(temperature, reference_temperature, places=8)
            self.assertAlmostEqual(
                composition['ethyl formate'],
                reference_composition['ethyl formate'],
                places=8,
            )

        x_ethyl_formate = reference_composition['ethyl formate']
        mw_ethyl_formate = thermo.props['ethyl formate'].MW
        mw_methanol = thermo.props['methanol'].MW
        mass_fraction = (
            x_ethyl_formate * mw_ethyl_formate
            / (
                x_ethyl_formate * mw_ethyl_formate
                + (1.0 - x_ethyl_formate) * mw_methanol
            )
        )

        self.assertEqual(round(reference_temperature - 273.15), 51)
        self.assertEqual(round(100.0 * mass_fraction), 84)

    def test_water_ethylene_oxide_fitted_interactions_are_built_and_used(self):
        water_cas = '7732-18-5'
        eo_cas = '75-21-8'

        with open(
            os.path.join(ROOT, 'data', 'source', 'water_ethylene_oxide_interactions.json'),
            encoding='utf-8',
        ) as handle:
            source = json.load(handle)
        self.assertEqual(source['metadata']['cas'], [water_cas, eo_cas])
        self.assertEqual(source['metadata']['activity_temperature_range_K'], [280.0, 420.0])
        self.assertEqual(source['metadata']['eos_temperature_range_K'], [330.0, 400.0])
        self.assertEqual(source['fitted_parameters']['UNIQUAC']['component2_r'], 1.59)
        self.assertEqual(source['fitted_parameters']['UNIQUAC']['component2_q'], 1.64)

        nrtl = nrtl_binary_interaction(water_cas, eo_cas)
        self.assertAlmostEqual(nrtl['tau12_c'], 5.85370)
        self.assertAlmostEqual(nrtl['tau12_d'], -1547.70)
        self.assertAlmostEqual(nrtl['tau21_c'], -4.70938)
        self.assertAlmostEqual(nrtl['tau21_d'], 1926.66)
        self.assertAlmostEqual(nrtl['alpha12'], 0.20)
        reversed_nrtl = nrtl_binary_interaction(eo_cas, water_cas)
        self.assertAlmostEqual(reversed_nrtl['tau12_c'], nrtl['tau21_c'])
        self.assertAlmostEqual(reversed_nrtl['tau12_d'], nrtl['tau21_d'])
        self.assertAlmostEqual(reversed_nrtl['tau21_c'], nrtl['tau12_c'])
        self.assertAlmostEqual(reversed_nrtl['tau21_d'], nrtl['tau12_d'])

        uniquac = uniquac_binary_interaction(water_cas, eo_cas)
        self.assertAlmostEqual(uniquac['tau12_a'], -2.31209)
        self.assertAlmostEqual(uniquac['tau12_b'], 673.029)
        self.assertAlmostEqual(uniquac['tau21_a'], 3.13674)
        self.assertAlmostEqual(uniquac['tau21_b'], -1397.36)
        reversed_uniquac = uniquac_binary_interaction(eo_cas, water_cas)
        self.assertAlmostEqual(reversed_uniquac['tau12_a'], uniquac['tau21_a'])
        self.assertAlmostEqual(reversed_uniquac['tau12_b'], uniquac['tau21_b'])

        self.assertAlmostEqual(
            eos_binary_interaction('PR', water_cas, eo_cas, 360.0),
            -0.10,
        )
        self.assertAlmostEqual(
            eos_binary_interaction('SRK', water_cas, eo_cas, 360.0),
            -0.13,
        )

        with open(
            os.path.join(ROOT, 'data', 'eos_binary_interactions_cas.json'),
            encoding='utf-8',
        ) as handle:
            eos_payload = json.load(handle)
        eos_records = [
            record for record in eos_payload['interactions']
            if {record['cas1'], record['cas2']} == {water_cas, eo_cas}
        ]
        self.assertEqual({record['model'] for record in eos_records}, {'PR', 'SRK'})
        for record in eos_records:
            self.assertEqual(record['Tmin_K'], 330.0)
            self.assertEqual(record['Tmax_K'], 400.0)

        nrtl_pr = create_thermodynamics(['H2O', 'C2H4O'], 'NRTL-PR')
        gamma = nrtl_pr.activity_coefficients(
            298.15,
            {'H2O': 1.0 - 1.0e-10, 'C2H4O': 1.0e-10},
        )
        self.assertAlmostEqual(gamma['C2H4O'], 6.666631833463465, places=8)
        self.assertFalse(any(
            'interaction parameters missing' in warning
            for warning in nrtl_pr.warnings
        ))

        bottoms = {
            'H2O': 0.9923309487336059,
            'C2H4O': 0.007669051266394127,
        }
        nrtl_temperature = nrtl_pr.bubble_point_T(bottoms, 2.24, 380.0)
        self.assertAlmostEqual(nrtl_temperature, 382.21945717059197, places=6)

        uniquac_pr = create_thermodynamics(['H2O', 'C2H4O'], 'UNIQUAC-PR')
        self.assertAlmostEqual(uniquac_pr.r['C2H4O'], 1.59)
        self.assertAlmostEqual(uniquac_pr.q['C2H4O'], 1.64)
        uniquac_gamma = uniquac_pr.activity_coefficients(
            298.15,
            {'H2O': 1.0 - 1.0e-10, 'C2H4O': 1.0e-10},
        )
        self.assertAlmostEqual(
            uniquac_gamma['C2H4O'],
            6.749482481162802,
            places=8,
        )
        self.assertFalse(any(
            "UNIQUAC r/q parameters missing for 'C2H4O'" in warning
            for warning in uniquac_pr.warnings
        ))
        uniquac_temperature = uniquac_pr.bubble_point_T(bottoms, 2.24, 380.0)
        self.assertAlmostEqual(uniquac_temperature, 382.4763417742024, places=6)

    def test_interaction_conversion_corrects_row_level_identity_mismatches(self):
        def records(filename, comment):
            with open(os.path.join(ROOT, 'data', filename), encoding='utf-8') as handle:
                payload = json.load(handle)
            return [
                record for record in payload['interactions']
                if record.get('comment') == comment
            ]

        for filename in (
            'nrtl_binary_interactions_cas.json',
            'uniquac_binary_interactions_cas.json',
        ):
            with self.subTest(filename=filename, comment='Ethanol/2-Butanol'):
                record = records(filename, 'Ethanol/2-Butanol p346 1/2c')[0]
                self.assertEqual(record['cas2'], '78-92-2')
                self.assertEqual(record['component2'], '2-butanol')

            with self.subTest(filename=filename, comment='Ethanol/2.6-Dimethylpyridine'):
                record = records(filename, 'Ethanol/2.6-Dimethylpyridine p447 1/2c')[0]
                self.assertEqual(record['cas2'], '108-48-5')
                self.assertEqual(record['component2'], '2,6-dimethylpyridine')

            with self.subTest(filename=filename, comment='Propylamine/1-Propanol'):
                record = records(filename, 'Propylamine/1-Propanol p492 1/2c')[0]
                self.assertEqual(record['cas1'], '107-10-8')
                self.assertEqual(record['cas2'], '71-23-8')

            with self.subTest(filename=filename, comment='Methanol/Formamide'):
                record = records(filename, 'Methanol/Formamide p30 1/2c')[0]
                self.assertEqual(record['cas1'], '67-56-1')
                self.assertEqual(record['cas2'], '75-12-7')

    def test_eos_temperature_specific_kij_selection_blends_static_fallback(self):
        self.assertAlmostEqual(
            eos_binary_interaction('PR', '124-38-9', '7783-06-4', 300.0),
            0.0967,
            places=6,
        )
        self.assertAlmostEqual(
            eos_binary_interaction('PR', '124-38-9', '7783-06-4', 400.0),
            0.09835,
            places=6,
        )

        eos = CubicEOS(['CO2', 'H2S'], 'PR')
        self.assertAlmostEqual(eos._kij('CO2', 'H2S', 300.0), 0.0967, places=6)
        self.assertAlmostEqual(eos._kij('CO2', 'H2S', 400.0), 0.09835, places=6)

    def test_eos_point_temperature_records_get_ten_kelvin_window(self):
        records = [
            {'kij': 0.10},
            {'kij': 0.20, 'temperature_range': (190.0, 220.0)},
            {'kij': 0.30, 'temperature_range': (202.0, 202.0)},
        ]

        self.assertAlmostEqual(_select_eos_kij(records, 200.0), 0.30)
        self.assertAlmostEqual(_select_eos_kij(records, 202.0), 0.30)
        self.assertAlmostEqual(_select_eos_kij(records, 215.0), 0.20)
        self.assertAlmostEqual(_select_eos_kij(records, 230.0), 0.15)

        eos = object.__new__(CubicEOS)
        eos._temperature_kij = {
            ('A', 'B'): [
                (0.20, 190.0, 220.0),
                (0.30, 202.0, 202.0),
            ],
        }
        eos._static_kij = {('A', 'B'): 0.10}
        eos._kij_cache = {}

        self.assertAlmostEqual(eos._kij('A', 'B', 200.0), 0.30)
        self.assertAlmostEqual(eos._kij('A', 'B', 215.0), 0.20)
        self.assertAlmostEqual(eos._kij('A', 'B', 230.0), 0.15)

    def test_temperature_dependent_pfd_kij_contributes_to_da_mix_dT(self):
        eos = CubicEOS(
            ['CO2', 'H2S'],
            'PR',
            ChemicalDatabase(enable_online=False),
            interaction_overrides=[{
                'model': 'PR',
                'component1': 'CO2',
                'component2': 'H2S',
                'kij_a': 0.1,
                'kij_b': 30.0,
                'kij_c': 0.001,
            }],
        )
        composition = {'CO2': 0.4, 'H2S': 0.6}
        T = 325.0
        h = 1e-3
        finite_difference = (
            eos.mixture_params(T + h, composition)[0]
            - eos.mixture_params(T - h, composition)[0]
        ) / (2.0 * h)

        self.assertAlmostEqual(eos.mixture_da_dT(T, composition), finite_difference, delta=abs(finite_difference) * 1e-5)

    def test_nrtl_supplemental_tau_matrix_predicts_ethanol_water_azeotrope(self):
        interaction = nrtl_binary_interaction('64-17-5', '7732-18-5')
        self.assertIsNotNone(interaction)
        self.assertAlmostEqual(interaction['tau12_c'], -0.801)
        self.assertAlmostEqual(interaction['tau12_d'], 246.2)
        self.assertAlmostEqual(interaction['tau21_c'], 3.458)
        self.assertAlmostEqual(interaction['tau21_d'], -586.1)
        self.assertAlmostEqual(interaction['alpha12'], 0.3)

        thermo = create_thermodynamics(['ethanol', 'water'], 'NRTL')
        azeotrope = thermo.generate_Txy_data(
            'ethanol',
            'water',
            1.01325,
            n_points=101,
        )['azeotrope']

        self.assertIsNotNone(azeotrope)
        self.assertAlmostEqual(azeotrope['x'], 0.898, delta=0.01)
        self.assertAlmostEqual(azeotrope['T'], 78.1, delta=0.3)

    def test_water_aromatic_regressions_roughly_reproduce_solubility_data(self):
        cases = [
            ('benzene', 343.15, 104.0, 11.3e-3, 6.10e-4),
            ('toluene', 348.15, 71.0, 12.2e-3, 1.89e-4),
            ('p-Xylene', 348.15, 52.0, 12.4e-3, 6.38e-5),
        ]

        for method in ('NRTL', 'UNIQUAC'):
            for organic, T, P_kPa, xw_organic_rich, xo_water_rich in cases:
                with self.subTest(method=method, organic=organic):
                    thermo = create_thermodynamics(['water', organic], method)
                    has_lle, phase1, phase2, _beta = thermo.liquid_liquid_equilibrium(
                        {'water': 0.5, organic: 0.5},
                        T,
                    )
                    self.assertTrue(has_lle)
                    organic_rich, water_rich = sorted(
                        (phase1, phase2),
                        key=lambda phase: phase[organic],
                        reverse=True,
                    )
                    self.assertLess(
                        abs(math.log(organic_rich['water'] / xw_organic_rich)),
                        0.12,
                    )
                    self.assertLess(
                        abs(math.log(water_rich[organic] / xo_water_rich)),
                        0.12,
                    )
                    pressure_bar = P_kPa / 100.0
                    average_bubble_pressure = 0.5 * (
                        thermo.bubble_point_P(organic_rich, T)
                        + thermo.bubble_point_P(water_rich, T)
                    )
                    self.assertLess(
                        abs(average_bubble_pressure / pressure_bar - 1.0),
                        0.08,
                    )
