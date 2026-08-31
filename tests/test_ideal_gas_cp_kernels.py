import math
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chemical_properties import ChemicalDatabase, ChemicalProperties
from pfd_parser import parse_and_validate
from property_resolver import PropertyResolver
from property_resolution.ideal_gas_cp import (
    ATOM_INCREMENT_MODEL_PATH,
    AtomIncrementCpModel,
    ChebyshevCpKernel,
    PolynomialCpKernel,
    ShomateCpKernel,
    kernel_from_payload,
    load_bundled_kernel,
    clear_bundled_kernel_cache,
    load_atom_increment_model,
)
from thermodynamics import IdealThermodynamics


class IdealGasCpKernelTests(unittest.TestCase):
    def assertClose(self, actual, expected, *, rel=1e-10, abs_tol=1e-10):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol),
            f'{actual!r} != {expected!r}',
        )

    def test_bundled_kernel_round_trip_and_primitives(self):
        kernel = load_bundled_kernel('64-17-5')
        self.assertIsInstance(kernel, ChebyshevCpKernel)
        restored = kernel_from_payload(kernel.to_payload())
        for T in (kernel.Tmin, 298.15, 500.0, kernel.Tmax):
            self.assertClose(restored.cp(T), kernel.cp(T), rel=1e-13)
        self.assertClose(
            restored.delta_h(298.15, 900.0),
            kernel.delta_h(298.15, 900.0),
            rel=1e-13,
        )
        self.assertClose(
            restored.delta_s(298.15, 900.0),
            kernel.delta_s(298.15, 900.0),
            rel=1e-13,
        )

    def test_bundled_cache_is_isolated_by_database_path(self):
        clear_bundled_kernel_cache()
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory, 'empty.sqlite')
            with sqlite3.connect(empty) as connection:
                connection.execute(
                    'CREATE TABLE canonical_ideal_gas_cp (cas TEXT PRIMARY KEY)'
                )
            self.assertIsNone(load_bundled_kernel('64-17-5', path=empty))
            self.assertIsNotNone(load_bundled_kernel('64-17-5'))

    def test_range_conditioning_is_continuous_and_penalized(self):
        kernel = PolynomialCpKernel(
            Tmin=300.0,
            Tmax=500.0,
            quality=0.98,
            source='test',
            method='test',
            coefficients=(20.0, 0.1),
        )
        self.assertClose(kernel.evaluate(505.0).quality, 0.96)
        self.assertClose(kernel.evaluate(510.0).quality, 0.96)
        self.assertClose(kernel.evaluate(515.0).quality, 0.94)
        self.assertClose(kernel.cp(510.0), 71.0)
        self.assertClose(kernel.cp(511.0), 71.0)
        self.assertClose(kernel.evaluate(1000.0).quality, 0.58)

        dT = 1.0e-4
        for T in (295.0, 505.0, 510.0, 520.0):
            derivative = (
                kernel.delta_h(298.15, T + dT)
                - kernel.delta_h(298.15, T - dT)
            ) / (2.0 * dT)
            self.assertClose(derivative, kernel.cp(T), rel=1e-7)

    def test_exact_shomate_and_polynomial_integrals(self):
        polynomial = PolynomialCpKernel(
            273.15, 1500.0, 1.0, 'test', 'polynomial',
            coefficients=(30.0, 0.01, 2.0e-5),
        )
        shomate = ShomateCpKernel(
            273.15, 1500.0, 1.0, 'test', 'shomate',
            coefficients=(30.0, 1.0, 2.0, 3.0, 4.0),
        )
        for kernel in (polynomial, shomate):
            T = 600.0
            dT = 1.0e-4
            dH = (
                kernel.delta_h(298.15, T + dT)
                - kernel.delta_h(298.15, T - dT)
            ) / (2.0 * dT)
            dS = (
                kernel.delta_s(298.15, T + dT)
                - kernel.delta_s(298.15, T - dT)
            ) / (2.0 * dT)
            self.assertClose(dH, kernel.cp(T), rel=2e-9)
            self.assertClose(dS, kernel.cp(T) / T, rel=2e-9)

    def test_nonpositive_curve_is_rejected_before_runtime(self):
        with self.assertRaises(ValueError):
            PolynomialCpKernel(
                273.15, 1500.0, 1.0, 'test', 'invalid',
                coefficients=(1.0, -1.0),
            )

    def test_atom_increment_model_aggregates_shomate_coefficients(self):
        model = load_atom_increment_model()
        self.assertIsNotNone(model)
        payload = json.loads(ATOM_INCREMENT_MODEL_PATH.read_text())
        counts = {'C': 2, 'H': 6, 'O': 1}
        kernel = model.kernel(counts, identity_note='ethanol formula')
        self.assertIsInstance(kernel, ShomateCpKernel)
        expected = tuple(
            2.0 * payload['parameters']['C']['coefficients'][name]
            + 6.0 * payload['parameters']['H']['coefficients'][name]
            + payload['parameters']['O']['coefficients'][name]
            for name in 'ABCDE'
        )
        self.assertEqual(kernel.coefficients, expected)
        self.assertEqual((kernel.Tmin, kernel.Tmax), (273.15, 1500.0))
        self.assertClose(kernel.quality, 0.80)
        self.assertIn('conventional-organic validation tier', kernel.notes)
        self.assertEqual(
            kernel.source_fingerprint,
            model.fingerprint,
        )
        self.assertIn('formula-grouped 5-fold CV', kernel.notes)

    def test_atom_increment_model_has_separate_selenium_and_noble_gas(self):
        model = load_atom_increment_model()
        parameters = dict(model.parameters)
        routes = dict(model.element_routes)
        self.assertIn('Se', parameters)
        self.assertIn('noble_gas', parameters)
        self.assertEqual(routes['Se'], 'Se')
        for element in ('He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'):
            self.assertEqual(routes[element], 'noble_gas')
        argon = model.kernel({'Ar': 1})
        self.assertClose(
            argon.cp(500.0),
            2.5 * 8.31446261815324,
            rel=1.0e-4,
        )
        self.assertClose(argon.quality, 0.75)
        self.assertIn('general-species validation tier', argon.notes)


class IdealGasCpResolverTests(unittest.TestCase):
    def assertClose(self, actual, expected, *, rel=1e-10, abs_tol=1e-10):
        self.assertTrue(math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol))

    def test_provided_override_precedes_bundled_database(self):
        resolver = PropertyResolver()
        props = {
            'CAS': '64-17-5',
            'property_correlations': {
                'Cpg': {
                    'equation': 'shomate',
                    'coefficients': {'A': 123.0},
                    'Tmin_K': 250.0,
                    'Tmax_K': 500.0,
                    'quality': 1.0,
                    '_pfd_override': True,
                }
            },
        }
        kernel = resolver.resolve_ideal_gas_cp_kernel(
            'ethanol', props, allow_online=False
        )
        self.assertIsInstance(kernel, ShomateCpKernel)
        self.assertEqual(kernel.cp(300.0), 123.0)

    def test_non_pfd_cpg_defaults_to_local_correlation_quality(self):
        resolver = PropertyResolver()
        correlation = {
            'equation': 'shomate',
            'coefficients': {'A': 123.0},
            'Tmin_K': 250.0,
            'Tmax_K': 500.0,
        }
        kernel = resolver.resolve_ideal_gas_cp_kernel(
            'local-cpg',
            {'property_correlations': {'Cpg': correlation}},
            allow_online=False,
            allow_estimation=False,
        )
        self.assertEqual(kernel.quality, 0.96)

        correlation['quality'] = 0.91
        explicit = resolver.resolve_ideal_gas_cp_kernel(
            'explicit-local-cpg',
            {'property_correlations': {'Cpg': correlation}},
            allow_online=False,
            allow_estimation=False,
        )
        self.assertEqual(explicit.quality, 0.91)

        correlation.pop('quality')
        correlation['_pfd_override'] = True
        pfd = resolver.resolve_ideal_gas_cp_kernel(
            'pfd-cpg',
            {'property_correlations': {'Cpg': correlation}},
            allow_online=False,
            allow_estimation=False,
        )
        self.assertEqual(pfd.quality, 1.0)

    def test_bundled_database_precedes_online(self):
        database = ChemicalDatabase(enable_online=False)
        props = database.get('ethanol', fetch_online=False).to_dict()
        resolver = PropertyResolver()
        with patch.object(
            resolver,
            '_fetch_nist_cp_source',
            side_effect=AssertionError('online source consulted'),
        ):
            kernel = resolver.resolve_ideal_gas_cp_kernel(
                'ethanol', props, allow_online=True
            )
        self.assertIn('canonical_perry_9e', kernel.method)

    def test_cached_online_kernel_is_available_when_network_is_disabled(self):
        source = {
            'gas': [[300.0, 30.0], [400.0, 40.0], [500.0, 50.0], [600.0, 60.0]],
            '_source': 'synthetic NIST',
        }
        with tempfile.TemporaryDirectory() as directory:
            first = PropertyResolver()
            first.CACHE_DIR = Path(directory)
            kernel = first._kernel_from_nist_source(source)
            first._set_derived_cp_kernel('online', 'madeupium', kernel)

            second = PropertyResolver()
            second.CACHE_DIR = Path(directory)
            with patch(
                'urllib.request.urlopen',
                side_effect=AssertionError('network consulted'),
            ):
                restored = second.resolve_ideal_gas_cp_kernel(
                    'madeupium', {}, allow_online=False, allow_estimation=False
                )
            self.assertIsInstance(restored, ShomateCpKernel)
            self.assertClose(restored.cp(450.0), 45.0)

    def test_cached_raw_nist_source_rebuilds_kernel_without_network(self):
        source = {
            'gas': [[300.0, 30.0], [400.0, 40.0], [500.0, 50.0]],
            '_source': 'synthetic NIST',
        }
        with tempfile.TemporaryDirectory() as directory:
            first = PropertyResolver()
            first.CACHE_DIR = Path(directory)
            first._set_cache(first._nist_cp_cache_key('madeupium'), source)

            second = PropertyResolver()
            second.CACHE_DIR = Path(directory)
            with patch(
                'urllib.request.urlopen',
                side_effect=AssertionError('network consulted'),
            ):
                kernel = second.resolve_ideal_gas_cp_kernel(
                    'madeupium', {}, allow_online=False, allow_estimation=False
                )
            self.assertIsInstance(kernel, ShomateCpKernel)
            self.assertClose(kernel.cp(450.0), 45.0)
            self.assertIsNotNone(second._get_derived_cp_kernel('online', 'madeupium'))

    def test_fresh_cas_negative_skips_weaker_online_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.CACHE_DIR = Path(directory)
            props = {'CAS': '9999-99-9', 'name': 'madeupium'}
            resolver._set_missing_cache(resolver._nist_cp_cache_key('9999-99-9'))
            with patch(
                'urllib.request.urlopen',
                side_effect=AssertionError('network consulted'),
            ):
                self.assertIsNone(resolver._fetch_nist_cp_source(
                    'madeupium', props, allow_network=True
                ))

    def test_disabling_estimation_does_not_use_atom_increment_kernel(self):
        resolver = PropertyResolver()
        with patch.object(resolver, '_fetch_nist_cp_source', return_value=None):
            kernel = resolver.resolve_ideal_gas_cp_kernel(
                'madeupium', {'formula': 'C2H6O'},
                allow_online=False,
                allow_estimation=False,
            )
            self.assertIsNone(kernel)

    def test_mw_only_props_no_longer_produce_an_emergency_kernel(self):
        resolver = PropertyResolver()
        with patch.object(resolver, '_fetch_nist_cp_source', return_value=None):
            kernel = resolver.resolve_ideal_gas_cp_kernel(
                'madeupium', {'MW': 100.0},
                allow_online=False,
                allow_estimation=True,
            )
        self.assertIsNone(kernel)

    def test_atom_increment_estimator_uses_formula_and_implicit_smiles_hydrogens(self):
        resolver = PropertyResolver()
        formula_counts = resolver._ideal_gas_cp_formula_counts('C2H5OH')
        smiles_counts = resolver._ideal_gas_cp_smiles_counts('CCO')
        self.assertEqual(formula_counts, {'C': 2, 'H': 6, 'O': 1})
        self.assertEqual(smiles_counts, formula_counts)
        self.assertEqual(
            resolver._ideal_gas_cp_formula_counts('C6D6'),
            {'C': 6, 'D': 6},
        )
        self.assertEqual(
            resolver._ideal_gas_cp_formula_counts('H4N+'),
            {'H': 4, 'N': 1},
        )

        with tempfile.TemporaryDirectory() as directory:
            resolver.CACHE_DIR = Path(directory)
            with patch.object(resolver, '_fetch_nist_cp_source', return_value=None):
                kernel = resolver.resolve_ideal_gas_cp_kernel(
                    'madeupium', {'formula': 'C2H5OH'},
                    allow_online=False,
                )
        self.assertIsInstance(kernel, ShomateCpKernel)
        self.assertEqual(
            kernel.method,
            'atom_increment_shomate_ideal_gas_cp_kernel',
        )
        self.assertEqual(kernel.source, 'estimated')
        self.assertClose(kernel.quality, 0.80)

    def test_atom_increment_kernel_round_trips_through_derived_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            first = PropertyResolver()
            first.CACHE_DIR = Path(directory)
            props = {'formula': 'C2H6O'}
            with patch.object(first, '_fetch_nist_cp_source', return_value=None):
                kernel = first.resolve_ideal_gas_cp_kernel(
                    'madeupium', props, allow_online=False
                )

            model = load_atom_increment_model()
            identity = model.cache_identity({'C': 2, 'H': 6, 'O': 1})
            key = first._derived_cp_cache_key('atom_increment', identity)
            payload = first._ideal_gas_cp_derived_cache().get(key)
            self.assertEqual(payload['origin'], 'atom_increment')
            self.assertEqual(payload['kind'], 'shomate')

            second = PropertyResolver()
            second.CACHE_DIR = Path(directory)
            with (
                patch.object(second, '_fetch_nist_cp_source', return_value=None),
                patch.object(
                    AtomIncrementCpModel,
                    'kernel',
                    side_effect=AssertionError('atom kernel rebuilt'),
                ),
            ):
                restored = second.resolve_ideal_gas_cp_kernel(
                    'madeupium', props, allow_online=False
                )
            self.assertIsInstance(restored, ShomateCpKernel)
            self.assertEqual(restored.coefficients, kernel.coefficients)

    def test_online_kernel_precedes_atom_increment_estimation(self):
        resolver = PropertyResolver()
        source = {
            'gas': [[300.0, 30.0], [400.0, 40.0], [500.0, 50.0]],
        }
        with tempfile.TemporaryDirectory() as directory:
            resolver.CACHE_DIR = Path(directory)
            with patch.object(resolver, '_fetch_nist_cp_source', return_value=source):
                kernel = resolver.resolve_ideal_gas_cp_kernel(
                    'madeupium', {'formula': 'C2H6O'}, allow_online=True
                )
        self.assertEqual(kernel.method, 'nist_linear_ideal_gas_cp_kernel')

    def test_live_webbook_shomate_layout_is_parsed(self):
        html = '''
        <table class="data" aria-label="Gas Phase Heat Capacity (Shomate Equation)">
          <tr><th>Temperature (K)</th><th>500. to 1700.</th><th>1700. to 6000.</th></tr>
          <tr><td>A</td><td>30.09200</td><td>41.96426</td></tr>
          <tr><td>B</td><td>6.832514</td><td>8.622053</td></tr>
          <tr><td>C</td><td>6.793435</td><td>-1.499780</td></tr>
          <tr><td>D</td><td>-2.534480</td><td>0.098119</td></tr>
          <tr><td>E</td><td>0.082139</td><td>-11.15764</td></tr>
          <tr><td>F</td><td>-250.8810</td><td>-272.1797</td></tr>
        </table>
        '''
        resolver = PropertyResolver()
        parsed = resolver._parse_nist_cp_tables(html)
        self.assertEqual(len(parsed['gas_shomate']), 2)
        self.assertEqual(parsed['gas_shomate'][0]['Tmin_K'], 500.0)
        self.assertEqual(parsed['gas_shomate'][1]['Tmax_K'], 6000.0)
        kernel = resolver._kernel_from_nist_source(parsed)
        self.assertIsInstance(kernel, ChebyshevCpKernel)
        self.assertLess(kernel.fit_max_error_percent, 0.1)

    def test_pfd_kernel_forms_and_default_range(self):
        resolver = PropertyResolver()
        cases = {
            'shomate': ShomateCpKernel,
            'poly_x': PolynomialCpKernel,
            'exp_poly_x': ChebyshevCpKernel,
        }
        for equation, expected_type in cases.items():
            with self.subTest(equation=equation):
                coefficients = {'A': 30.0}
                if equation == 'exp_poly_x':
                    coefficients = {'A': math.log(30.0), 'B': 0.01}
                kernel = resolver.resolve_ideal_gas_cp_kernel(
                    equation,
                    {'property_correlations': {'Cpg': {
                        'equation': equation,
                        'coefficients': coefficients,
                        '_pfd_override': True,
                    }}},
                    allow_online=False,
                )
                self.assertIsInstance(kernel, expected_type)
                self.assertEqual(kernel.Tmin, 273.15)
                self.assertEqual(kernel.Tmax, 1500.0)

        pfd = '''
PROCESS: Cp warning
VERSION: 1.0
COMPONENTS:
    X | test | MW=40
PROPERTY_CORRELATIONS:
    X.Cpg | equation=shomate, A=30
'''
        _, errors, warnings = parse_and_validate(pfd)
        self.assertFalse(errors)
        self.assertTrue(any('defaulting to 273.15-1500 K' in item for item in warnings))

    def test_online_table_quality_tiers_and_single_point_range(self):
        resolver = PropertyResolver()
        single = resolver._kernel_from_nist_source({'gas': [[300.0, 40.0]]})
        sparse = resolver._kernel_from_nist_source({
            'gas': [[300.0, 30.0], [400.0, 40.0], [500.0, 50.0]],
        })
        dense_rows = []
        for T in range(300, 1301, 100):
            t = T / 1000.0
            dense_rows.append([float(T), 30.0 + 10.0 * t + t * t])
        dense = resolver._kernel_from_nist_source({'gas': dense_rows})

        self.assertEqual((single.Tmin, single.Tmax), (295.0, 305.0))
        self.assertClose(single.quality, 0.84)
        self.assertClose(single.evaluate(315.0).quality, 0.82)
        self.assertClose(sparse.quality, 0.86)
        self.assertClose(dense.quality, 0.92)

        noisy = resolver._kernel_from_nist_source({
            'gas': [
                [300.0, 30.0], [350.0, 38.0], [400.0, 40.0],
                [450.0, 42.0], [500.0, 50.0],
            ],
        })
        expected_penalty = (
            0.01 * max(0.0, noisy.fit_max_error_percent - 2.0)
            + 0.06 * max(0.0, noisy.fit_mape_percent - 0.5)
        )
        self.assertClose(noisy.quality, 0.86 - expected_penalty)

        pfd_coefficients = resolver.resolve_ideal_gas_cp_kernel(
            'X',
            {
                'Cp_coeffs': [30.0, 0.0, 0.0, 0.0],
                'property_sources': {
                    'Cp_coeffs': {'method': 'pfd_component_override', 'quality': 0.7},
                },
            },
            allow_online=False,
        )
        self.assertClose(pfd_coefficients.quality, 1.0)

    def test_nist_point_count_selection_boundaries_are_preserved(self):
        resolver = PropertyResolver()
        for count, expected_method in (
            (2, 'nist_linear_ideal_gas_cp_kernel'),
            (9, 'nist_linear_ideal_gas_cp_kernel'),
            (10, 'nist_tabulated_shomate_ideal_gas_cp_kernel'),
        ):
            rows = []
            for index in range(count):
                T = 300.0 + 50.0 * index
                t = T / 1000.0
                rows.append([T, 30.0 + 10.0 * t + t * t])
            with self.subTest(count=count):
                kernel = resolver._kernel_from_nist_source({'gas': rows})
                self.assertEqual(kernel.method, expected_method)


class IdealGasCpThermoTests(unittest.TestCase):
    def assertClose(self, actual, expected, *, rel=1e-10, abs_tol=1e-10):
        self.assertTrue(math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol))

    def test_thermo_prebinds_one_kernel_for_cp_h_and_s(self):
        database = ChemicalDatabase(enable_online=False)
        thermo = IdealThermodynamics(['C2H5OH'], database)
        thermo._resolver_known_props['C2H5OH']['_allow_online_lookup'] = False
        kernel = thermo._ideal_gas_cp_kernel('C2H5OH')
        self.assertIs(kernel, thermo._ideal_gas_cp_kernel('C2H5OH'))

        T = 500.0
        dT = 1.0e-3
        self.assertClose(thermo.Cp_ideal_gas('C2H5OH', T), kernel.cp(T))
        dH = (
            thermo.enthalpy_ideal_gas('C2H5OH', T + dT)
            - thermo.enthalpy_ideal_gas('C2H5OH', T - dT)
        ) * 1000.0 / (2.0 * dT)
        dS = (
            thermo.entropy_ideal_gas('C2H5OH', T + dT)
            - thermo.entropy_ideal_gas('C2H5OH', T - dT)
        ) / (2.0 * dT)
        self.assertClose(dH, kernel.cp(T), rel=1e-7)
        self.assertClose(dS, kernel.cp(T) / T, rel=1e-7)

        # Once bound, the hot Cp/H/S paths do not return to source resolution.
        with patch(
            'property_resolver.get_property_resolver',
            side_effect=AssertionError('resolver re-entered after kernel binding'),
        ):
            self.assertGreater(thermo.Cp_ideal_gas('C2H5OH', T + 1.0), 0.0)
            self.assertTrue(math.isfinite(thermo.enthalpy_ideal_gas('C2H5OH', T + 2.0)))
            self.assertTrue(math.isfinite(thermo.entropy_ideal_gas('C2H5OH', T + 3.0)))

    def test_chemical_properties_uses_kernel_for_h_and_s(self):
        props = ChemicalProperties(
            symbol='X', name='X', formula='X', MW=50.0,
            Cp_coeffs=[30.0, 0.01, 0.0, 0.0],
        )
        self.assertClose(props.Cp(400.0), 34.0)
        self.assertClose(props.delta_H(300.0, 400.0), 3350.0)
        self.assertClose(
            props.delta_S(300.0, 400.0),
            30.0 * math.log(4.0 / 3.0) + 1.0,
        )


if __name__ == '__main__':
    unittest.main()
