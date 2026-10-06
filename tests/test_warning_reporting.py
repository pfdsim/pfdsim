import os
import json
import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chemical_properties import ChemicalDatabase, ChemicalProperties
from interaction_parameters import uniquac_rq_for_component
from property_resolution import (
    ConstantLiquidCpKernel,
    PropertyResolutionResult,
    ShomateCpKernel,
)
from simulator import Simulator
from thermodynamics import create_thermodynamics


class WarningReportingTests(unittest.TestCase):
    @staticmethod
    def _quality_test_thermo():
        db = ChemicalDatabase(enable_online=False)
        db.chemicals['X'] = ChemicalProperties(
            symbol='X',
            name='Quality test component',
            formula='X',
            MW=50.0,
            Tb=350.0,
            Tc=500.0,
            Pc=40.0,
            Vc=150.0,
            omega=0.2,
            Hvap=30.0,
            Cp_coeffs=[33.0, 0.0, 0.0, 0.0],
            phase_at_STP='liquid',
        )
        return create_thermodynamics(['X'], 'IDEAL', db)

    @staticmethod
    def _fake_lazy_resolver():
        class FakeResolver:
            def resolve_liquid_cp_kernel(self, symbol, props=None, allow_online=True, allow_estimation=True):
                return ConstantLiquidCpKernel(
                    Tmin=250.0,
                    Tmax=500.0,
                    quality=0.78,
                    source='online',
                    method='nist_constant_liquid_cp_kernel',
                    notes='constant average liquid Cp from sparse data',
                    value=180.0,
                )

            def resolve_ideal_gas_cp_kernel(self, symbol, props=None, allow_online=True, allow_estimation=True):
                return ShomateCpKernel(
                    Tmin=250.0,
                    Tmax=500.0,
                    quality=0.88,
                    source='online',
                    method='nist_shomate_gas_cp_kernel',
                    notes='Shomate-style gas Cp fit',
                    coefficients=(150.0, 0.0, 0.0, 0.0, 0.0),
                )

            def resolve_heat_capacity(self, symbol, T, phase='liquid', props=None, allow_online=True):
                if phase == 'liquid':
                    return PropertyResolutionResult(
                        value=180.0,
                        source='online',
                        method='nist_constant_liquid_cp',
                        quality=0.78,
                        notes='constant average liquid Cp from sparse data',
                    )
                return PropertyResolutionResult(
                    value=150.0,
                    source='online',
                    method='nist_shomate_gas_cp_fit',
                    quality=0.88,
                    notes='Shomate-style gas Cp fit',
                )

            def resolve_liquid_molar_volume(self, symbol, T, props=None):
                return PropertyResolutionResult(
                    value=0.1,
                    source='online',
                    method='pubchem_rackett_fitted_zra_liquid_volume',
                    quality=0.91,
                    notes='Rackett fit from PubChem density',
                )

        return FakeResolver()

    def test_lazy_temperature_dependent_property_sources_are_aggregated(self):
        thermo = self._quality_test_thermo()
        fake_resolver = self._fake_lazy_resolver()

        with patch('property_resolver.get_property_resolver', return_value=fake_resolver):
            for T in (329.54, 340.0):
                thermo.Cp_liquid('X', T)
                thermo.Cp_ideal_gas('X', T)
                thermo.mixture_liquid_molar_volume({'X': 1.0}, T)

        rows = {
            (row['property'], row['source'], row['method']): row
            for row in thermo.lazy_property_quality_sources()
        }

        self.assertEqual(rows[('Cp_liquid(T)', 'online', 'nist_constant_liquid_cp_kernel')]['count'], 1)
        self.assertAlmostEqual(rows[('Cp_liquid(T)', 'online', 'nist_constant_liquid_cp_kernel')]['T_min'], 298.15)
        self.assertAlmostEqual(rows[('Cp_liquid(T)', 'online', 'nist_constant_liquid_cp_kernel')]['T_max'], 298.15)
        self.assertEqual(
            rows[
                (
                    'liquid_molar_volume(T)',
                    'online',
                    'pubchem_rackett_fitted_zra_liquid_volume',
                )
            ]['count'],
            2,
        )

    def test_lazy_temperature_dependent_property_sources_are_written_to_quality_report(self):
        thermo = self._quality_test_thermo()
        fake_resolver = self._fake_lazy_resolver()

        with patch('property_resolver.get_property_resolver', return_value=fake_resolver):
            for T in (329.54, 340.0):
                thermo.Cp_liquid('X', T)
                thermo.Cp_ideal_gas('X', T)
                thermo.mixture_liquid_molar_volume({'X': 1.0}, T)

        sim = Simulator.__new__(Simulator)
        sim.thermo = thermo
        report = sim._property_quality_report()
        keyed = {
            (row['property'], row['source'], row['method']): row
            for row in report
            if row['component'] == 'X'
        }

        self.assertIn(('Cp_liquid(T)', 'online', 'nist_constant_liquid_cp_kernel'), keyed)
        self.assertIn(('Cp_ideal_gas(T)', 'online', 'nist_shomate_gas_cp_kernel'), keyed)
        self.assertIn(
            ('liquid_molar_volume(T)', 'online', 'pubchem_rackett_fitted_zra_liquid_volume'),
            keyed,
        )

        cp_liq = keyed[('Cp_liquid(T)', 'online', 'nist_constant_liquid_cp_kernel')]
        self.assertEqual(cp_liq['severity'], 'low')
        self.assertEqual(cp_liq['count'], 1)
        self.assertAlmostEqual(cp_liq['T_min'], 298.15)
        self.assertAlmostEqual(cp_liq['T_max'], 298.15)
        self.assertIn('constant average liquid Cp', cp_liq['notes'])

    def test_nannoolal_static_source_notes_are_written_to_quality_report(self):
        thermo = self._quality_test_thermo()
        thermo.props['X'].property_sources['Tb'] = {
            'source': 'estimated',
            'method': 'nannoolal_tb',
            'quality': 0.80,
            'notes': (
                'Estimated by Nannoolal group contribution from SMILES; '
                'warnings: group 219 local extension'
            ),
        }
        thermo.mark_property_source_context(
            'X',
            'Tb',
            kind='thermo_model',
            phase='test_parameters',
            affects_result=True,
        )

        sim = Simulator.__new__(Simulator)
        sim.thermo = thermo
        report = sim._property_quality_report()

        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]['method'], 'nannoolal_tb')
        self.assertIn('group 219 local extension', report[0]['notes'])

    def test_nannoolal_lookup_warnings_are_method_specific(self):
        db = ChemicalDatabase(enable_online=False)
        props = ChemicalProperties(symbol='X', name='X', formula='X')
        props.property_sources['Tb'] = {
            'source': 'estimated',
            'method': 'nannoolal_tb',
            'quality': 0.80,
            'notes': 'Nannoolal warning details',
        }
        props.property_sources['Tc'] = {
            'source': 'estimated',
            'method': 'nannoolal_tc',
            'quality': 0.75,
            'notes': 'Nannoolal warning details',
        }

        db._refresh_estimation_warnings(props)

        joined = '\n'.join(props.lookup_warnings)
        self.assertIn("Boiling point for 'X' was estimated from Nannoolal group contribution.", joined)
        self.assertIn("Critical temperature for 'X' was estimated from Nannoolal group contribution.", joined)
        self.assertNotIn('estimated from MW', joined)
        self.assertNotIn('estimated from Tb.', joined)

    def test_lazy_temperature_dependent_property_sources_are_written_to_pfr_text(self):
        thermo = self._quality_test_thermo()
        fake_resolver = self._fake_lazy_resolver()

        with patch('property_resolver.get_property_resolver', return_value=fake_resolver):
            for T in (329.54, 340.0):
                thermo.Cp_liquid('X', T)

        sim = Simulator.__new__(Simulator)
        sim.thermo = thermo
        sim.thermo_method = 'IDEAL'
        sim.pfd = SimpleNamespace(
            metadata=SimpleNamespace(process_name='Lazy Quality Text', version='1.0'),
            streams=[],
            units=[],
        )
        sim.result = SimpleNamespace(
            recycle_info={},
            converged=True,
            iterations=0,
            mass_balance_error=0.0,
            energy_balance_error=0.0,
            streams={},
            units={},
            errors=[],
            warnings=[],
        )

        pfr = sim._generate_pfr()

        self.assertIn('low: X.Cp_liquid(T) quality=0.780', pfr)
        self.assertIn('resolver_calls = 1, T = 298.15 K', pfr)

    def test_thermo_warnings_are_written_to_pfr_global_warning_section(self):
        pfd = """PROCESS: Warning Reporting
VERSION: 1.0
THERMO_METHOD: UNIQUAC

COMPONENTS:
    HCHO | Formaldehyde | MW=30.026, formula=CH2O
    H2O | Water | MW=18.015

STREAM Feed : FEED -> FLASH-1.in
    T = 80 [C]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = HCHO:0.5, H2O:0.5

STREAM Vapor : FLASH-1.vap -> PRODUCT
STREAM Liquid : FLASH-1.liq -> PRODUCT

UNIT FLASH-1
    TYPE: Flash
    PORTS:
        in : inlet
        vap : outlet
        liq : outlet
    PARAMS:
        T = 80 [C]
        P = 1 [bar]
"""
        sim = Simulator.from_string(pfd)
        result = sim.run()
        pfr = sim._generate_pfr()

        self.assertTrue(result.converged)
        self.assertTrue(any("UNIQUAC r/q parameters missing" in w for w in result.warnings))
        self.assertIn("# WARNINGS", pfr)
        self.assertIn("UNIQUAC r/q parameters missing", pfr)

    def test_unused_static_property_quality_is_retained_but_not_written_to_pfr(self):
        pfd = """PROCESS: Quality Report
VERSION: 1.0

COMPONENTS:
    X | Test component | MW=50

STREAM Feed : FEED -> HEAT-1.in
    T = 25 [C]
    P = 1 [bar]
    F = 1 [kmol/h]
    x = X:1

STREAM Product : HEAT-1.out -> PRODUCT

UNIT HEAT-1
    TYPE: Heater
    PORTS:
        in : inlet
        out : outlet
    PARAMS:
        T_out = 50 [C]
"""
        sim = Simulator.from_string(pfd)
        result = sim.run()
        self.assertTrue(result.converged)

        sim.thermo.props['X'].property_sources['Tc'] = {
            'source': 'estimated',
            'method': 'guldberg_rule',
            'quality': 0.55,
            'notes': 'test low-quality critical temperature',
        }
        retained = sim._property_quality_report(include_suppressed=True)
        retained_keyed = {
            (item['component'], item['property']): item
            for item in retained
        }
        pfr = sim._generate_pfr()

        self.assertIn("# QUALITY REPORT", pfr)
        self.assertNotIn("estimated: X.Tc quality=0.550", pfr)
        self.assertIn("X.Cp_liquid(T)", pfr)
        self.assertIn(('X', 'Tc'), retained_keyed)
        self.assertEqual(retained_keyed[('X', 'Tc')]['suppressed_reason'], 'no_result_context')

    def test_uniquac_rq_estimation_is_structured_warning_not_runtime_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            thermo = create_thermodynamics(['formaldehyde', 'water'], 'UNIQUAC')

        self.assertTrue(any("UNIQUAC r/q parameters missing" in w for w in thermo.warnings))
        self.assertFalse(
            any("UNIQUAC r/q parameters missing" in str(warning.message) for warning in caught)
        )

    def test_uniquac_rq_lookup_uses_specific_identity_not_formula_isomers(self):
        db = ChemicalDatabase(enable_online=False)

        ethyl_acetate = uniquac_rq_for_component(
            'ethyl acetate',
            db.get('ethyl acetate', fetch_online=False),
        )
        self.assertIsNotNone(ethyl_acetate)
        self.assertAlmostEqual(ethyl_acetate['r'], 3.4789)
        self.assertAlmostEqual(ethyl_acetate['q'], 3.1159)
        self.assertEqual(ethyl_acetate['cas'], '141-78-6')
        self.assertIn('extended_uniquac', ethyl_acetate)
        self.assertAlmostEqual(
            ethyl_acetate['extended_uniquac']['q_prime'],
            3.1160 ** 0.1,
        )

        ethanol = uniquac_rq_for_component(
            'ethanol',
            db.get('ethanol', fetch_online=False),
        )
        self.assertIsNotNone(ethanol)
        self.assertAlmostEqual(ethanol['r'], 2.1055)
        self.assertAlmostEqual(ethanol['q'], 1.9720)
        self.assertAlmostEqual(ethanol['extended_uniquac']['q_prime'], 0.92)

        chloroform = uniquac_rq_for_component(
            'chloroform',
            db.get('chloroform', fetch_online=False),
        )
        self.assertIsNotNone(chloroform)
        self.assertAlmostEqual(chloroform['r'], 2.87)
        self.assertAlmostEqual(chloroform['q'], 2.41)
        self.assertEqual(chloroform['cas'], '67-66-3')
        self.assertIn('Abrams and Prausnitz', chloroform['source'])

        expected_entries = {
            'acetaldehyde': (1.90, 1.80, '75-07-0', 'Abrams and Prausnitz'),
            'dimethylamine': (2.33, 2.09, '124-40-3', 'Abrams and Prausnitz'),
            'furfural': (2.80, 2.58, '98-01-1', 'Abrams and Prausnitz'),
            'aniline': (3.72, 2.83, '62-53-3', 'Abrams and Prausnitz'),
            'triethylamine': (5.01, 4.26, '121-44-8', 'Abrams and Prausnitz'),
            'n-hexadecane': (11.2438, 9.256, '544-76-3', 'Thermochimica Acta 268 (1995) 45-68'),
        }
        for name, (expected_r, expected_q, expected_cas, expected_source) in expected_entries.items():
            with self.subTest(name=name):
                entry = uniquac_rq_for_component(
                    name,
                    db.get(name, fetch_online=False),
                )
                self.assertIsNotNone(entry)
                self.assertAlmostEqual(entry['r'], expected_r)
                self.assertAlmostEqual(entry['q'], expected_q)
                self.assertEqual(entry['cas'], expected_cas)
                self.assertIn(expected_source, entry['source'])

        propylene_oxide = uniquac_rq_for_component(
            'propylene oxide',
            db.get('propylene oxide', fetch_online=False),
        )
        self.assertIsNone(propylene_oxide)

        methylpentane = uniquac_rq_for_component(
            '3-methylpentane',
            db.get('3-methylpentane', fetch_online=False),
        )
        self.assertIsNone(methylpentane)

        self.assertIsNone(uniquac_rq_for_component('C6H14'))

        with open(os.path.join(ROOT, 'data', 'uniquac_rq_cas.json'), encoding='utf-8') as handle:
            payload = json.load(handle)
        self.assertEqual(payload['metadata']['key_basis'], 'CAS')
        self.assertGreaterEqual(payload['metadata']['component_count'], 73)
        self.assertEqual(payload['metadata']['component_count'], len(payload['components']))
        self.assertIn('data/source/activity_fitting/dwsim_uniquac_combinatorial_parameters.csv', payload['metadata']['source_files'])
        self.assertIn('data/source/activity_fitting/nagata_gmehling_extended_uniquac_rq.json', payload['metadata']['source_files'])
        self.assertIn(
            'data/source/activity_fitting/ester_uniquac_combinatorial_parameters.json',
            payload['metadata']['source_files'],
        )

    def test_missing_binary_interaction_warnings_are_reported_for_local_models(self):
        cases = [
            ('NRTL', ['ethanol', 'nitrogen'], 'NRTL binary interaction parameters missing'),
            ('UNIQUAC', ['ethanol', 'nitrogen'], 'UNIQUAC binary interaction parameters missing'),
            ('PR', ['water', 'nitrogen'], 'PR EOS binary interaction parameters missing'),
        ]

        for method, components, expected in cases:
            with self.subTest(method=method):
                thermo = create_thermodynamics(components, method)
                self.assertTrue(any(expected in warning for warning in thermo.warnings))

    def test_unifac_warns_when_any_group_interaction_pair_is_missing(self):
        db = ChemicalDatabase(enable_online=False)
        db.chemicals['MISSINGPAIR'] = ChemicalProperties(
            symbol='MISSINGPAIR',
            name='Missing Pair',
            formula='X',
            MW=100.0,
            Tb=350.0,
            Tc=500.0,
            Pc=40.0,
            omega=0.2,
        )

        thermo = create_thermodynamics(
            ['C2H4', 'MISSINGPAIR'],
            'UNIFAC',
            db=db,
            unifac_groups={
                'C2H4': {'CH2=CH': 1},
                'MISSINGPAIR': {'ACNO2': 1},
            },
        )

        self.assertTrue(
            any('UNIFAC group interaction parameters missing' in w for w in thermo.warnings)
        )


if __name__ == '__main__':
    unittest.main()
