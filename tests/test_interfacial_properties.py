import math
import unittest

import numpy as np

from interfacial_properties import (
    InterfacialPropertyError,
    InterfacialPropertyWarning,
    MixtureSurfaceTensionCalculator,
    SurfaceComponent,
    estimate_mixture_surface_tension,
    goldsack_white_area,
    normalize_mole_fractions,
)


class CountingIdealActivity:
    def __init__(self):
        self.calls = 0

    def activity_coefficients(self, T, composition):
        self.calls += 1
        return {component: 1.0 for component in composition}


class FailingActivity:
    def activity_coefficients(self, T, composition):
        raise RuntimeError('activity model unavailable')


def ordinary_components():
    return [
        SurfaceComponent(
            'water',
            55.95e-6,
            lambda T: 55_000.0,
            lambda T: 0.072,
            CAS='7732-18-5',
        ),
        SurfaceComponent(
            'ethanol',
            168.0e-6,
            lambda T: 17_000.0,
            lambda T: 0.022,
            CAS='64-17-5',
        ),
    ]


class InterfacialPropertiesTests(unittest.TestCase):
    def assertClose(self, actual, expected, *, rel=1e-9, abs_tol=1e-12):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol),
            f'{actual!r} != {expected!r}',
        )

    def test_normalization_and_goldsack_white_area(self):
        normalized = normalize_mole_fractions([2.0, 3.0])
        np.testing.assert_allclose(normalized, [0.4, 0.6])

        Vc = np.array([55.95e-6, 168.0e-6])
        Vb = np.array([1.0 / 55_000.0, 1.0 / 17_000.0])
        area = goldsack_white_area(Vc, Vb)
        self.assertTrue(np.all(np.isfinite(area)))
        self.assertTrue(np.all(area > 0.0))

        for invalid in ([], [0.0, 0.0], [math.nan, 1.0], [-0.1, 1.1]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(InterfacialPropertyError):
                    normalize_mole_fractions(invalid)

    def test_pure_component_limit_does_not_resolve_inactive_properties(self):
        def unavailable(_T):
            raise AssertionError('inactive component property was evaluated')

        components = [
            ordinary_components()[0],
            SurfaceComponent('inactive', 100e-6, unavailable, unavailable),
        ]
        result = MixtureSurfaceTensionCalculator(components).calculate(
            298.15,
            {'water': 1.0, 'inactive': 0.0},
        )

        self.assertClose(result.sigma, 0.072)
        np.testing.assert_allclose(result.surface_x, [1.0, 0.0])
        self.assertTrue(math.isnan(result.A[1]))
        self.assertEqual(result.method, 'pure-component surface tension')

    def test_wsd_matches_closed_form_and_emits_limitation_warning(self):
        components = ordinary_components()
        x = np.array([0.25, 0.75])
        Vb = np.array([1.0 / 55_000.0, 1.0 / 17_000.0])
        sigma = np.array([0.072, 0.022])
        weights = x * Vb
        expected = np.sum(
            np.outer(weights, weights) * np.sqrt(np.outer(sigma, sigma))
        ) / np.sum(weights) ** 2

        with self.assertWarns(InterfacialPropertyWarning):
            result = MixtureSurfaceTensionCalculator(components).calculate(
                298.15,
                x,
                method='wsd',
            )

        self.assertClose(result.sigma, expected)
        self.assertEqual(result.method, 'Winterfeld-Scriven-Davis')
        self.assertIn('does not model surface-phase segregation', result.warnings[0])

    def test_butler_prefers_superlinear_newton_and_enriches_low_sigma_component(self):
        model = CountingIdealActivity()
        calculator = MixtureSurfaceTensionCalculator(
            ordinary_components(),
            activity_model=model,
        )

        result = calculator.calculate(298.15, {'water': 0.9, 'ethanol': 0.1})

        self.assertTrue(result.converged)
        self.assertEqual(result.solver, 'damped-newton')
        self.assertLessEqual(result.residual_max_abs, 1e-8)
        self.assertGreater(result.surface_x[1], result.bulk_x[1])
        self.assertGreater(result.sigma, 0.022)
        self.assertLess(result.sigma, 0.072)
        self.assertGreater(result.activity_evaluations, 0)
        self.assertEqual(model.calls, calculator.cache_info()['activity_evaluations'])

    def test_activity_cache_and_automatic_warm_start_are_reused(self):
        model = CountingIdealActivity()
        calculator = MixtureSurfaceTensionCalculator(
            ordinary_components(),
            activity_model=model,
            activity_cache_size=128,
        )
        first = calculator.calculate(298.15, [0.9, 0.1])
        calls_after_first = model.calls
        second = calculator.calculate(298.15, [0.9, 0.1])

        self.assertTrue(first.converged and second.converged)
        self.assertEqual(second.warm_start_used, 'cache')
        self.assertEqual(second.activity_evaluations, 0)
        self.assertGreater(second.activity_cache_hits, 0)
        self.assertEqual(model.calls, calls_after_first)
        self.assertClose(second.sigma, first.sigma)

    def test_explicit_warm_start_can_be_selected(self):
        calculator = MixtureSurfaceTensionCalculator(
            ordinary_components(),
            activity_model=CountingIdealActivity(),
        )
        first = calculator.calculate(298.15, [0.8, 0.2])
        second = calculator.calculate(
            300.0,
            [0.79, 0.21],
            warm_start=first,
            use_cached_warm_start=False,
        )

        self.assertTrue(second.converged)
        self.assertEqual(second.warm_start_used, 'explicit')

    def test_activity_cache_is_bounded(self):
        calculator = MixtureSurfaceTensionCalculator(
            ordinary_components(),
            activity_model=CountingIdealActivity(),
            activity_cache_size=3,
        )
        calculator.calculate(298.15, [0.9, 0.1])
        self.assertLessEqual(calculator.cache_info()['activity_entries'], 3)

    def test_auto_falls_back_to_wsd_with_warning_when_activity_model_fails(self):
        calculator = MixtureSurfaceTensionCalculator(
            ordinary_components(),
            activity_model=FailingActivity(),
        )
        with self.assertWarns(InterfacialPropertyWarning):
            result = calculator.calculate(298.15, [0.5, 0.5])

        self.assertEqual(result.method, 'Winterfeld-Scriven-Davis')
        self.assertIn('Butler failed', result.warnings[0])

        with self.assertRaises(InterfacialPropertyError):
            calculator.calculate(298.15, [0.5, 0.5], method='butler')

    def test_one_shot_wrapper_uses_activity_coefficients_contract(self):
        result = estimate_mixture_surface_tension(
            298.15,
            [0.5, 0.5],
            ordinary_components(),
            activity_model=CountingIdealActivity(),
        )
        self.assertTrue(result.converged)
        self.assertTrue(result.method.startswith('Butler-'))

    def test_existing_compiled_unifac_backend_is_used_transparently(self):
        from thermodynamics import UNIFNISTThermodynamics

        thermo = UNIFNISTThermodynamics(['water', 'ethanol'])
        calculator = MixtureSurfaceTensionCalculator(
            ordinary_components(),
            activity_model=thermo,
        )
        compiled_before = thermo._compiled_unifac
        result = calculator.calculate(
            298.15,
            {'water': 0.99, 'ethanol': 0.01},
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.method, 'Butler-UNIFNIST')
        self.assertIs(thermo._compiled_unifac, compiled_before)
        self.assertGreater(result.surface_x[1], result.bulk_x[1])

    def test_standard_unifac_is_rejected_only_for_interfacial_calculator(self):
        from thermodynamics import UNIFACThermodynamics

        thermo = UNIFACThermodynamics(['water', 'ethanol'])
        with self.assertRaises(InterfacialPropertyError):
            MixtureSurfaceTensionCalculator(
                ordinary_components(),
                activity_model=thermo,
            )

    def test_tabulated_areas_are_used_when_every_component_is_covered(self):
        calculator = MixtureSurfaceTensionCalculator(
            ordinary_components(),
            activity_model=CountingIdealActivity(),
        )
        result = calculator.calculate(298.15, [0.98, 0.02])

        self.assertEqual(result.area_method, 'tabulated-cas')
        np.testing.assert_allclose(result.A, [7225.0, 80520.0])

    def test_one_missing_area_forces_goldsack_white_for_every_component(self):
        components = [
            ordinary_components()[0],
            SurfaceComponent(
                'custom',
                120.0e-6,
                lambda T: 20_000.0,
                lambda T: 0.03,
            ),
        ]
        calculator = MixtureSurfaceTensionCalculator(
            components,
            activity_model=CountingIdealActivity(),
        )
        result = calculator.calculate(298.15, [0.5, 0.5])
        expected = goldsack_white_area(
            np.array([55.95e-6, 120.0e-6]),
            np.array([1.0 / 55_000.0, 1.0 / 20_000.0]),
        )

        self.assertEqual(result.area_method, 'goldsack-white')
        np.testing.assert_allclose(result.A, expected)
        self.assertNotEqual(result.A[0], 7225.0)

    def test_fitted_isopropanol_area_is_pinned(self):
        component = SurfaceComponent(
            'isopropanol',
            None,
            lambda T: 13_000.0,
            lambda T: 0.021,
            CAS='67-63-0',
        )
        result = MixtureSurfaceTensionCalculator([component]).calculate(
            298.15,
            [1.0],
        )

        self.assertEqual(result.area_method, 'tabulated-cas')
        self.assertClose(result.A[0], 106948.68498366099)

    def test_ternary_butler_system_converges_with_compiled_unifnist(self):
        from thermodynamics import UNIFNISTThermodynamics

        components = [
            ordinary_components()[0],
            SurfaceComponent(
                'methanol',
                118.0e-6,
                lambda T: 24_600.0,
                lambda T: 0.02215,
                CAS='67-56-1',
            ),
            ordinary_components()[1],
        ]
        thermo = UNIFNISTThermodynamics(['water', 'methanol', 'ethanol'])
        result = MixtureSurfaceTensionCalculator(
            components,
            activity_model=thermo,
        ).calculate(
            298.15,
            {'water': 0.90, 'methanol': 0.05, 'ethanol': 0.05},
        )

        self.assertTrue(result.converged)
        self.assertEqual(result.method, 'Butler-UNIFNIST')
        self.assertEqual(result.solver, 'damped-newton')
        self.assertLessEqual(result.residual_max_abs, 1e-8)
        self.assertClose(float(np.sum(result.surface_x)), 1.0)
        self.assertEqual(result.area_method, 'tabulated-cas')
        np.testing.assert_allclose(result.A, [7225.0, 39870.0, 80520.0])

    def test_invalid_activity_coefficients_are_rejected_in_explicit_butler_mode(self):
        class InvalidActivity:
            def activity_coefficients(self, T, composition):
                return {component: -1.0 for component in composition}

        calculator = MixtureSurfaceTensionCalculator(
            ordinary_components(),
            activity_model=InvalidActivity(),
        )
        with self.assertRaises(InterfacialPropertyError):
            calculator.calculate(298.15, [0.5, 0.5], method='butler')

    def test_unknown_mapping_component_is_rejected(self):
        calculator = MixtureSurfaceTensionCalculator(ordinary_components())
        with self.assertRaises(InterfacialPropertyError):
            calculator.calculate(298.15, {'water': 1.0, 'mystery': 1.0})


if __name__ == '__main__':
    unittest.main()
