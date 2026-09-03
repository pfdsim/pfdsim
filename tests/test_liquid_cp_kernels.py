import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chemical_properties import ChemicalDatabase, ChemicalProperties
from pfd_parser import parse_and_validate
from property_resolver import PropertyResolver
from property_resolution.liquid_cp import (
    ConstantLiquidCpKernel,
    LinearChebyshevLiquidCpKernel,
    NativeZabranskyLiquidCpKernel,
    PolynomialLiquidCpKernel,
    ScaledIdealGasLiquidCpKernel,
    ShomateLiquidCpKernel,
    clear_bundled_liquid_kernel_cache,
    liquid_kernel_from_payload,
    load_bundled_liquid_kernel,
)
from property_resolution.common import PropertyResolutionResult
from property_resolution.ideal_gas_cp import ShomateCpKernel
from thermodynamics import IdealThermodynamics


class LiquidCpKernelTests(unittest.TestCase):
    def assertClose(self, actual, expected, *, rel=1e-10, abs_tol=1e-10):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol),
            f'{actual!r} != {expected!r}',
        )

    def test_bundled_linear_and_native_models_round_trip(self):
        for cas, expected_type in (
            ('64-17-5', LinearChebyshevLiquidCpKernel),
            ('117-81-7', NativeZabranskyLiquidCpKernel),
        ):
            with self.subTest(cas=cas):
                kernel = load_bundled_liquid_kernel(cas)
                self.assertIsInstance(kernel, expected_type)
                restored = liquid_kernel_from_payload(kernel.to_payload())
                for T in (kernel.Tmin, 298.15, kernel.Tmax):
                    self.assertClose(restored.cp(T), kernel.cp(T), rel=2e-13)
                self.assertClose(
                    restored.delta_h(298.15, min(350.0, kernel.Tmax)),
                    kernel.delta_h(298.15, min(350.0, kernel.Tmax)),
                    rel=2e-12,
                )
                self.assertClose(
                    restored.delta_s(298.15, min(350.0, kernel.Tmax)),
                    kernel.delta_s(298.15, min(350.0, kernel.Tmax)),
                    rel=2e-12,
                )

    def test_bundled_cache_is_isolated_by_database_path(self):
        clear_bundled_liquid_kernel_cache()
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory, 'empty.sqlite')
            with sqlite3.connect(empty) as connection:
                connection.execute(
                    'CREATE TABLE canonical_liquid_cp (cas TEXT PRIMARY KEY)'
                )
            self.assertIsNone(load_bundled_liquid_kernel('64-17-5', path=empty))
            self.assertIsNotNone(load_bundled_liquid_kernel('64-17-5'))

    def test_liquid_range_penalties_match_the_gas_values(self):
        kernel = PolynomialLiquidCpKernel(
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

    def test_native_zabransky_primitives_differentiate_to_cp(self):
        kernel = load_bundled_liquid_kernel('117-81-7')
        for T in (300.0, 400.0, min(460.0, kernel.Tmax - 1.0)):
            dT = 1.0e-4
            dH = (
                kernel.delta_h(298.15, T + dT)
                - kernel.delta_h(298.15, T - dT)
            ) / (2.0 * dT)
            dS = (
                kernel.delta_s(298.15, T + dT)
                - kernel.delta_s(298.15, T - dT)
            ) / (2.0 * dT)
            self.assertClose(dH, kernel.cp(T), rel=2e-7)
            self.assertClose(dS, kernel.cp(T) / T, rel=2e-7)


class LiquidCpResolverTests(unittest.TestCase):
    def assertClose(self, actual, expected, *, rel=1e-10, abs_tol=1e-10):
        self.assertTrue(math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol))

    @staticmethod
    def estimator_props(
        formula,
        smiles=None,
        *,
        gas_quality=1.0,
        tc_quality=1.0,
        omega_quality=1.0,
        identity_quality=1.0,
    ):
        props = {
            'formula': formula,
            'Tc': 500.0,
            'omega': 0.2,
            'property_correlations': {
                'Cpg': {
                    'equation': 'shomate',
                    'coefficients': {'A': 80.0},
                    'Tmin_K': 200.0,
                    'Tmax_K': 1000.0,
                    'quality': gas_quality,
                },
            },
            'property_sources': {
                'formula': {'method': 'test_formula', 'quality': identity_quality},
                'Tc': {'method': 'test_tc', 'quality': tc_quality},
                'omega': {'method': 'test_omega', 'quality': omega_quality},
            },
        }
        if smiles is not None:
            props['smiles'] = smiles
            props['property_sources']['smiles'] = {
                'method': 'test_smiles',
                'quality': identity_quality,
            }
        return props

    def resolve_estimator(self, name, props):
        resolver = PropertyResolver()
        with (
            patch('property_resolution.heat_capacity.load_bundled_liquid_kernel', return_value=None),
            patch('property_resolution.heat_capacity.lookup_bundled_liquid_cas', return_value=None),
            patch.object(resolver, '_get_derived_liquid_cp_kernel', return_value=None),
            patch.object(resolver, '_fetch_nist_cp_source', return_value=None),
        ):
            return resolver.resolve_liquid_cp_kernel(
                name,
                props,
                allow_online=False,
            )

    def test_pfd_correlation_precedes_bundle_and_remains_authoritative(self):
        resolver = PropertyResolver()
        props = {
            'CAS': '64-17-5',
            'property_correlations': {
                'Cpl': {
                    'equation': 'poly_x',
                    'coefficients': {'A': 123.0},
                    'Tmin_K': 290.0,
                    'Tmax_K': 310.0,
                    'quality': 1.0,
                    '_pfd_override': True,
                }
            },
        }
        kernel = resolver.resolve_liquid_cp_kernel(
            'ethanol', props, allow_online=False
        )
        self.assertIsInstance(kernel, PolynomialLiquidCpKernel)
        self.assertEqual(kernel.cp(300.0), 123.0)
        self.assertEqual(kernel.cp(500.0), 123.0)
        self.assertIn('clamped above', kernel.evaluate(500.0).range_note)

    def test_non_pfd_cpl_defaults_to_local_correlation_quality(self):
        resolver = PropertyResolver()
        correlation = {
            'equation': 'poly_x',
            'coefficients': {'A': 123.0},
            'Tmin_K': 290.0,
            'Tmax_K': 310.0,
        }
        kernel = resolver.resolve_liquid_cp_kernel(
            'local-cpl',
            {'property_correlations': {'Cpl': correlation}},
            allow_online=False,
            allow_estimation=False,
        )
        self.assertEqual(kernel.quality, 0.96)

        correlation['quality'] = 0.91
        explicit = resolver.resolve_liquid_cp_kernel(
            'explicit-local-cpl',
            {'property_correlations': {'Cpl': correlation}},
            allow_online=False,
            allow_estimation=False,
        )
        self.assertEqual(explicit.quality, 0.91)

        correlation.pop('quality')
        correlation['_pfd_override'] = True
        pfd = resolver.resolve_liquid_cp_kernel(
            'pfd-cpl',
            {'property_correlations': {'Cpl': correlation}},
            allow_online=False,
            allow_estimation=False,
        )
        self.assertEqual(pfd.quality, 1.0)

    def test_explicit_pfd_constant_is_unbounded(self):
        resolver = PropertyResolver()
        props = {
            'CAS': '64-17-5',
            'Cp_liquid': 222.0,
            'property_sources': {
                'Cp_liquid': {
                    'source': 'provided',
                    'method': 'pfd_component_override',
                    'quality': 1.0,
                }
            },
        }
        kernel = resolver.resolve_liquid_cp_kernel('ethanol', props, allow_online=False)
        self.assertIsInstance(kernel, ConstantLiquidCpKernel)
        self.assertTrue(kernel.unbounded)
        for T in (100.0, 298.15, 1000.0):
            self.assertEqual(kernel.cp(T), 222.0)
            self.assertEqual(kernel.quality_at(T), 1.0)

    def test_bundle_precedes_non_pfd_stp_scalar_and_live_perry(self):
        database = ChemicalDatabase(enable_online=False)
        props = database.get('ethanol', fetch_online=False).to_dict()
        props['Cp_liquid'] = 999.0
        props.setdefault('property_sources', {})['Cp_liquid'] = {
            'source': 'chemicals.json',
            'quality': 0.95,
        }
        resolver = PropertyResolver()
        with (
            patch.object(
                resolver,
                '_get_perry_evaluation',
                side_effect=AssertionError('live Perry consulted'),
            ),
            patch.object(
                resolver,
                '_fetch_nist_cp_source',
                side_effect=AssertionError('online source consulted'),
            ),
        ):
            kernel = resolver.resolve_liquid_cp_kernel(
                'ethanol', props, allow_online=True
            )
        self.assertIn('canonical_', kernel.method)
        self.assertNotEqual(kernel.cp(298.15), 999.0)

    def test_exact_bundled_identity_reaches_database_without_cas_or_perry(self):
        resolver = PropertyResolver()
        props = {
            'symbol': 'Biphenyl',
            'name': 'Biphenyl',
            'CAS': '',
            'formula': '',
        }
        with (
            patch.object(
                resolver,
                '_get_perry_evaluation',
                side_effect=AssertionError('live Perry property rung consulted'),
            ),
            patch.object(
                resolver,
                '_get_perry_identity',
                side_effect=AssertionError('Perry identity hydration consulted'),
            ),
        ):
            kernel = resolver.resolve_liquid_cp_kernel(
                'Biphenyl', props, allow_online=False, allow_estimation=False
            )
        self.assertIsInstance(kernel, LinearChebyshevLiquidCpKernel)
        self.assertEqual(kernel.method, 'canonical_perry_9e_liquid_cp')

    def test_non_pfd_scalar_is_one_stp_point(self):
        resolver = PropertyResolver()
        props = {
            'Cp_liquid': 88.0,
            'property_sources': {
                'Cp_liquid': {'source': 'test data', 'quality': 0.95},
            },
        }
        kernel = resolver.resolve_liquid_cp_kernel(
            'madeupium', props, allow_online=False, allow_estimation=False
        )
        self.assertEqual((kernel.Tmin, kernel.Tmax), (293.15, 303.15))
        self.assertEqual(kernel.quality_at(298.15), 0.95)
        self.assertAlmostEqual(kernel.quality_at(313.15), 0.93)
        self.assertLess(kernel.quality_at(400.0), 0.95)

    def test_pfd_kernel_forms_and_default_range(self):
        resolver = PropertyResolver()
        cases = {
            'shomate': ShomateLiquidCpKernel,
            'poly_x': PolynomialLiquidCpKernel,
            'exp_poly_x': LinearChebyshevLiquidCpKernel,
        }
        for equation, expected_type in cases.items():
            with self.subTest(equation=equation):
                coefficients = {'A': 80.0}
                if equation == 'exp_poly_x':
                    coefficients = {'A': math.log(80.0), 'B': 0.01}
                kernel = resolver.resolve_liquid_cp_kernel(
                    equation,
                    {'property_correlations': {'Cpl': {
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
PROCESS: liquid Cp warning
VERSION: 1.0
COMPONENTS:
    X | test | MW=40
PROPERTY_CORRELATIONS:
    X.Cpl | equation=shomate, A=80
'''
        _, errors, warnings = parse_and_validate(pfd)
        self.assertFalse(errors)
        self.assertTrue(any('X.Cpl' in item and '273.15-1500 K' in item for item in warnings))

    def test_cached_online_liquid_kernel_is_available_offline(self):
        source = {
            'liquid': [[300.0, 80.0], [350.0, 85.0], [400.0, 90.0]],
            '_source': 'synthetic NIST',
        }
        with tempfile.TemporaryDirectory() as directory:
            first = PropertyResolver()
            first.CACHE_DIR = Path(directory)
            kernel = first._kernel_from_nist_liquid_source(source)
            identity = 'cached-liquid-madeupium'
            first._set_derived_liquid_cp_kernel('online', identity, kernel)
            self.assertIsNotNone(first._get_derived_liquid_cp_kernel('online', identity))

            second = PropertyResolver()
            second.CACHE_DIR = Path(directory)
            restored = second._get_derived_liquid_cp_kernel('online', identity)
            self.assertIsInstance(restored, PolynomialLiquidCpKernel)
            self.assertClose(restored.cp(375.0), 87.5)

    def test_predictive_routing_covers_bondi_hbd_gc_and_terminal_fallback(self):
        cases = (
            ('hydrocarbon', self.estimator_props('C6H14', 'CCCCCC'),
             'rowlinson_bondi_liquid_cp_kernel', 0.89),
            ('thiol', self.estimator_props('CH4S', 'CS'),
             'rowlinson_bondi_liquid_cp_kernel', 0.89),
            ('alcohol', self.estimator_props('C2H6O', 'CCO'),
             'hbd_ratio_gc_liquid_cp_kernel', 0.82),
            ('polyol', self.estimator_props('C3H8O3', 'OCC(O)CO'),
             'hbd_ratio_gc_liquid_cp_kernel', 0.82),
            ('mixed', self.estimator_props('C4H11NO2', 'OCCNCCO'),
             'mixed_donor_rowlinson_bondi_liquid_cp_kernel', 0.70),
            ('inorganic-hcn', self.estimator_props('CHN', 'C#N'),
             'scaled_ideal_gas_liquid_cp_kernel', 0.60),
        )
        for name, props, method, quality in cases:
            with self.subTest(name=name):
                kernel = self.resolve_estimator(name, props)
                self.assertEqual(kernel.method, method)
                self.assertClose(kernel.quality, quality, rel=1e-12)
                self.assertGreater(kernel.cp(300.0), 0.0)

    def test_formula_only_organic_routing_is_conservative_for_heteroatoms(self):
        resolver = PropertyResolver()
        cases = (
            ('formula-hydrocarbon', self.estimator_props('C6H14'),
             'rowlinson_bondi_liquid_cp_kernel', 0.89),
            ('formula-ketone', self.estimator_props('C3H6O'),
             'mixed_donor_rowlinson_bondi_liquid_cp_kernel', 0.70),
        )
        with (
            patch('property_resolution.heat_capacity.load_bundled_liquid_kernel', return_value=None),
            patch('property_resolution.heat_capacity.lookup_bundled_liquid_cas', return_value=None),
            patch.object(resolver, '_get_derived_liquid_cp_kernel', return_value=None),
            patch.object(resolver, '_fetch_nist_cp_source', return_value=None),
            patch.object(resolver, '_resolve_smiles_result', return_value=None),
        ):
            for name, props, method, quality in cases:
                with self.subTest(name=name):
                    kernel = resolver.resolve_liquid_cp_kernel(
                        name,
                        props,
                        allow_online=False,
                    )
                    self.assertEqual(kernel.method, method)
                    self.assertClose(kernel.quality, quality, rel=1e-12)

    def test_predictive_quality_factors_multiply_required_input_quality(self):
        common = dict(
            gas_quality=0.80,
            tc_quality=0.90,
            omega_quality=0.70,
            identity_quality=0.95,
        )
        gc = self.resolve_estimator(
            'quality-gc', self.estimator_props('C2H6O', 'CCO', **common)
        )
        mixed = self.resolve_estimator(
            'quality-mixed', self.estimator_props('C4H11NO2', 'OCCNCCO', **common)
        )
        fallback = self.resolve_estimator(
            'quality-inorganic', self.estimator_props('CHN', 'C#N', **common)
        )
        self.assertClose(gc.quality, 0.82 * 0.80, rel=1e-12)
        self.assertClose(mixed.quality, 0.70 * 0.70, rel=1e-12)
        self.assertClose(fallback.quality, 0.60 * 0.80, rel=1e-12)

        ideal_gas = ShomateCpKernel(
            Tmin=200.0,
            Tmax=1000.0,
            quality=0.80,
            source='test',
            method='test',
            coefficients=(80.0, 0.0, 0.0, 0.0, 0.0),
        )
        inputs = tuple(
            PropertyResolutionResult(
                value=1.0,
                source='test',
                method='test',
                quality=quality,
            )
            for quality in (0.90, 0.70, 0.95)
        )
        self.assertClose(
            PropertyResolver._liquid_cp_estimator_quality(
                0.89,
                ideal_gas,
                *inputs,
            ),
            0.89 * 0.70,
            rel=1e-12,
        )

    def test_low_quality_required_criticals_reject_predictive_rung(self):
        def result(value, quality, method):
            return PropertyResolutionResult(
                value=value,
                source='test',
                method=method,
                quality=quality,
            )

        cases = (
            (
                'gc-low-tc',
                self.estimator_props('C2H6O', 'CCO'),
                {'Tc': result(500.0, 0.69, 'low_tc'),
                 'omega': result(0.2, 1.0, 'good_omega')},
            ),
            (
                'bondi-low-tc',
                self.estimator_props('C6H14', 'CCCCCC'),
                {'Tc': result(500.0, 0.69, 'low_tc'),
                 'omega': result(0.2, 1.0, 'good_omega')},
            ),
            (
                'bondi-low-omega',
                self.estimator_props('C6H14', 'CCCCCC'),
                {'Tc': result(500.0, 1.0, 'good_tc'),
                 'omega': result(0.2, 0.69, 'low_omega')},
            ),
        )
        for name, props, critical in cases:
            with self.subTest(name=name):
                resolver = PropertyResolver()
                with (
                    patch('property_resolution.heat_capacity.load_bundled_liquid_kernel', return_value=None),
                    patch('property_resolution.heat_capacity.lookup_bundled_liquid_cas', return_value=None),
                    patch.object(resolver, '_get_derived_liquid_cp_kernel', return_value=None),
                    patch.object(resolver, '_fetch_nist_cp_source', return_value=None),
                    patch.object(resolver, 'resolve_critical_properties', return_value=critical),
                ):
                    kernel = resolver.resolve_liquid_cp_kernel(
                        name,
                        props,
                        allow_online=False,
                    )
                self.assertIsInstance(kernel, ScaledIdealGasLiquidCpKernel)
                self.assertEqual(kernel.method, 'scaled_ideal_gas_liquid_cp_kernel')
                self.assertClose(kernel.quality, 0.60, rel=1e-12)

    def test_predictive_kernel_primitives_differentiate_to_cp(self):
        for name, props in (
            ('bondi-primitives', self.estimator_props('C6H14', 'CCCCCC')),
            ('gc-primitives', self.estimator_props('C2H6O', 'CCO')),
        ):
            kernel = self.resolve_estimator(name, props)
            for T in (250.0, 350.0, 450.0):
                with self.subTest(name=name, T=T):
                    dT = 1.0e-3
                    dH = (
                        kernel.delta_h(T, T + dT)
                        - kernel.delta_h(T, T - dT)
                    ) / (2.0 * dT)
                    dS = (
                        kernel.delta_s(T, T + dT)
                        - kernel.delta_s(T, T - dT)
                    ) / (2.0 * dT)
                    self.assertClose(dH, kernel.cp(T), rel=2e-6)
                    self.assertClose(dS, kernel.cp(T) / T, rel=2e-6)

    def test_predictive_kernels_reproduce_bondi_and_ratio_gc_equations(self):
        T = 300.0
        Tc = 500.0
        Tr = T / Tc
        omega = 0.2
        R = 8.31446261815324
        cp_ideal = 80.0

        bondi = self.resolve_estimator(
            'bondi-equation', self.estimator_props('C6H14', 'CCCCCC')
        )
        one_minus_Tr = 1.0 - Tr
        expected_bondi = cp_ideal + R * (
            1.45
            + 0.45 / one_minus_Tr
            + omega * (
                4.2775
                + 6.3 * one_minus_Tr ** (1.0 / 3.0) / Tr
                + 0.4355 / one_minus_Tr
            )
        )
        self.assertClose(bondi.cp(T), expected_bondi, rel=1e-3)

        alcohol = self.resolve_estimator(
            'gc-equation', self.estimator_props('C2H6O', 'CCO')
        )
        size = math.log(cp_ideal / R)
        y = (
            -0.696566247027313
            + 0.132964405534084 * size
            - 0.422120326414122 * Tr
        )
        expected_gc = cp_ideal / min(1.25, max(0.15, math.exp(y)))
        self.assertClose(alcohol.cp(T), expected_gc, rel=1e-6)

    def test_scalar_resolver_is_a_kernel_view(self):
        resolver = PropertyResolver()
        props = {'CAS': '64-17-5'}
        kernel = resolver.resolve_liquid_cp_kernel('ethanol', props, allow_online=False)
        result = resolver.resolve_heat_capacity(
            'ethanol', 350.0, phase='liquid', props=props, allow_online=False
        )
        self.assertClose(result.value, kernel.cp(350.0))
        self.assertClose(result.quality, kernel.quality_at(350.0))
        self.assertEqual(result.method, kernel.method)


class LiquidCpThermoTests(unittest.TestCase):
    def assertClose(self, actual, expected, *, rel=1e-10, abs_tol=1e-10):
        self.assertTrue(math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol))

    def test_thermo_reuses_one_kernel_for_cp_h_and_s(self):
        database = ChemicalDatabase(enable_online=False)
        thermo = IdealThermodynamics(['C2H5OH'], database)
        thermo._resolver_known_props['C2H5OH']['_allow_online_lookup'] = False
        kernel = thermo._liquid_cp_kernel('C2H5OH')
        self.assertIs(kernel, thermo._liquid_cp_kernel('C2H5OH'))

        T = 350.0
        dT = 1.0e-3
        self.assertClose(thermo.Cp_liquid('C2H5OH', T), kernel.cp(T))
        derivative_h = (
            thermo.enthalpy_liquid('C2H5OH', T + dT)
            - thermo.enthalpy_liquid('C2H5OH', T - dT)
        ) / (2.0 * dT) * 1000.0
        self.assertClose(derivative_h, kernel.cp(T), rel=2e-7)
        self.assertClose(
            thermo._integrate_cp_over_T('C2H5OH', 300.0, T, 'liquid'),
            kernel.delta_s(300.0, T),
            rel=2e-12,
        )

    def test_pfd_constant_controls_cp_and_enthalpy(self):
        database = ChemicalDatabase(enable_online=False)
        props = ChemicalProperties(
            symbol='X',
            name='constant liquid',
            formula='X',
            MW=50.0,
            Cp_liquid=123.0,
            Cp_coeffs=[30.0, 0.0, 0.0, 0.0],
            Hvap=40.0,
            property_sources={
                'Cp_liquid': {
                    'source': 'provided',
                    'method': 'pfd_component_override',
                    'quality': 1.0,
                }
            },
        )
        database.chemicals = {'X': props}
        database._build_aliases()
        thermo = IdealThermodynamics(['X'], database)
        kernel = thermo._liquid_cp_kernel('X')
        self.assertTrue(kernel.unbounded)
        self.assertEqual(thermo.Cp_liquid('X', 600.0), 123.0)
        self.assertClose(
            thermo._integrate_cp_analytic('X', 300.0, 400.0, 'liquid'),
            12.3,
        )


if __name__ == '__main__':
    unittest.main()
