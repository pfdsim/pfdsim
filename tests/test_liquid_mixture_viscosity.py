import math
import unittest

from liquid_mixture_viscosity import (
    JouybanAcreeParameters,
    LiquidMixtureViscosityError,
    estimate_liquid_mixture_viscosity,
    grunberg_nissan_viscosity,
    jouyban_acree_viscosity,
    unifac_visco_viscosity,
    _component_unifac_visco_groups,
    _unifac_visco_combinatorial_excess_g_over_rt,
    _unifac_visco_excess_g_over_rt,
)


class LiquidMixtureViscosityTests(unittest.TestCase):
    def test_grunberg_nissan_reduces_to_log_mixing_without_interactions(self):
        viscosity = grunberg_nissan_viscosity(
            {'a': 0.25, 'b': 0.75},
            {'a': 1.0e-3, 'b': 4.0e-3},
        )

        expected = math.exp(0.25 * math.log(1.0e-3) + 0.75 * math.log(4.0e-3))
        self.assertAlmostEqual(viscosity, expected, places=18)

    def test_grunberg_nissan_adds_binary_interaction_once(self):
        viscosity = grunberg_nissan_viscosity(
            {'a': 0.4, 'b': 0.6},
            {'a': 1.0e-3, 'b': 2.0e-3},
            {('b', 'a'): 0.5},
        )

        expected = math.exp(
            0.4 * math.log(1.0e-3)
            + 0.6 * math.log(2.0e-3)
            + 0.4 * 0.6 * 0.5
        )
        self.assertAlmostEqual(viscosity, expected, places=18)

    def test_unifac_visco_uses_simplified_excess_g_term(self):
        viscosity = unifac_visco_viscosity(
            {'a': 0.5, 'b': 0.5},
            {'a': 1.0e-3, 'b': 4.0e-3},
            excess_g_over_rt=0.2,
        )

        expected = math.exp(
            0.5 * math.log(1.0e-3)
            + 0.5 * math.log(4.0e-3)
            + 0.2
        )
        self.assertAlmostEqual(viscosity, expected, places=18)

    def test_jouyban_acree_reduces_to_log_mixing_without_pair_parameters(self):
        viscosity = jouyban_acree_viscosity(
            {'a': 0.25, 'b': 0.75},
            {'a': 1.0e-3, 'b': 4.0e-3},
            298.15,
            {},
        )

        expected = math.exp(0.25 * math.log(1.0e-3) + 0.75 * math.log(4.0e-3))
        self.assertAlmostEqual(viscosity, expected, places=18)

    def test_jouyban_acree_sums_multicomponent_pair_terms(self):
        viscosity = jouyban_acree_viscosity(
            {'a': 0.2, 'b': 0.3, 'c': 0.5},
            {'a': 1.0e-3, 'b': 2.0e-3, 'c': 4.0e-3},
            300.0,
            {
                ('a', 'b'): JouybanAcreeParameters((90.0, 30.0)),
                ('a', 'c'): JouybanAcreeParameters((-60.0,)),
            },
        )

        log_mixing = (
            0.2 * math.log(1.0e-3)
            + 0.3 * math.log(2.0e-3)
            + 0.5 * math.log(4.0e-3)
        )
        ab_term = 0.2 * 0.3 / 300.0 * (90.0 + 30.0 * (0.2 - 0.3))
        ac_term = 0.2 * 0.5 / 300.0 * -60.0
        expected = math.exp(log_mixing + ab_term + ac_term)
        self.assertAlmostEqual(viscosity, expected, places=18)

    def test_jouyban_acree_reversed_pair_uses_stored_orientation(self):
        forward = jouyban_acree_viscosity(
            {'a': 0.2, 'b': 0.8},
            {'a': 1.0e-3, 'b': 2.0e-3},
            300.0,
            {('a', 'b'): JouybanAcreeParameters((0.0, 100.0))},
        )
        reversed_key = jouyban_acree_viscosity(
            {'a': 0.2, 'b': 0.8},
            {'a': 1.0e-3, 'b': 2.0e-3},
            300.0,
            {('b', 'a'): JouybanAcreeParameters((0.0, -100.0))},
        )

        self.assertAlmostEqual(forward, reversed_key, places=18)

    def test_estimator_uses_grunberg_nissan_override(self):
        result = estimate_liquid_mixture_viscosity(
            {'a': 0.4, 'b': 0.6},
            {'a': 1.0e-3, 'b': 2.0e-3},
            300.0,
            interaction_overrides=[{
                'model': 'LIQUID_VISCOSITY',
                'component1': 'a',
                'component2': 'b',
                'viscosity_form': 'grunberg_nissan',
                'G': 0.5,
            }],
        )

        expected = math.exp(
            0.4 * math.log(1.0e-3)
            + 0.6 * math.log(2.0e-3)
            + 0.4 * 0.6 * 0.5
        )
        self.assertAlmostEqual(result.value, expected, places=18)
        self.assertEqual(result.pair_methods[('a', 'b')], 'override_grunberg_nissan')

    def test_estimator_uses_jouyban_acree_override_orientation(self):
        result = estimate_liquid_mixture_viscosity(
            {'a': 0.2, 'b': 0.8},
            {'a': 1.0e-3, 'b': 2.0e-3},
            300.0,
            interaction_overrides=[{
                'model': 'LIQUID_VISCOSITY',
                'component1': 'b',
                'component2': 'a',
                'viscosity_form': 'jouyban_acree',
                'A0_K': 0.0,
                'A1_K': -100.0,
                'A2_K': 0.0,
            }],
        )

        log_mixing = 0.2 * math.log(1.0e-3) + 0.8 * math.log(2.0e-3)
        expected = math.exp(log_mixing + 0.8 * 0.2 / 300.0 * (-100.0 * (0.8 - 0.2)))
        self.assertAlmostEqual(result.value, expected, places=18)
        self.assertEqual(result.pair_methods[('a', 'b')], 'override_jouyban_acree')

    def test_estimator_uses_temperature_ranged_excess_poly_override(self):
        overrides = [
            {
                'model': 'LIQUID_VISCOSITY',
                'component1': 'a',
                'component2': 'b',
                'viscosity_form': 'excess_poly',
                'Tmin_K': 280.0,
                'Tmax_K': 310.0,
                'A': 0.4,
                'B': 0.1,
                'C': 0.2,
                'D': 0.3,
                'E': 0.4,
                'F': 0.5,
                'T_ref_K': 300.0,
            },
            {
                'model': 'LIQUID_VISCOSITY',
                'component1': 'a',
                'component2': 'b',
                'viscosity_form': 'grunberg_nissan',
                'Tmin_K': 330.0,
                'Tmax_K': 360.0,
                'G': 2.0,
            },
        ]
        result = estimate_liquid_mixture_viscosity(
            {'a': 0.4, 'b': 0.6},
            {'a': 1.0e-3, 'b': 2.0e-3},
            305.0,
            interaction_overrides=overrides,
        )

        delta = 0.4 - 0.6
        theta = (305.0 - 300.0) / 100.0
        excess = 0.4 * 0.6 * (
            0.4 + 0.1 * delta + 0.2 * delta * delta
            + 0.3 * theta + 0.4 * theta * delta + 0.5 * theta * theta
        )
        expected = math.exp(0.4 * math.log(1.0e-3) + 0.6 * math.log(2.0e-3) + excess)
        self.assertAlmostEqual(result.value, expected, places=18)
        self.assertEqual(result.pair_methods[('a', 'b')], 'override_excess_poly')

    def test_rejects_nonpositive_pure_viscosity(self):
        with self.assertRaises(LiquidMixtureViscosityError):
            grunberg_nissan_viscosity(
                {'a': 1.0},
                {'a': 0.0},
            )

    def test_jouyban_acree_rejects_nonpositive_temperature(self):
        with self.assertRaises(LiquidMixtureViscosityError):
            jouyban_acree_viscosity(
                {'a': 0.5, 'b': 0.5},
                {'a': 1.0e-3, 'b': 2.0e-3},
                0.0,
                {('a', 'b'): JouybanAcreeParameters((100.0,))},
            )

    def test_estimator_prefers_jouyban_acree_pair_data(self):
        result = estimate_liquid_mixture_viscosity(
            {'water': 0.5, 'acetonitrile': 0.5},
            {'water': 0.00089, 'acetonitrile': 0.00034},
            298.15,
        )

        self.assertEqual(result.method, 'jouyban_acree')
        self.assertEqual(
            result.pair_methods[('water', 'acetonitrile')],
            'jouyban_acree',
        )
        self.assertEqual(result.warnings, ())
        self.assertGreater(result.value, 0.0)

    def test_estimator_warns_for_far_jouyban_acree_temperature(self):
        result = estimate_liquid_mixture_viscosity(
            {'water': 0.5, 'acetonitrile': 0.5},
            {'water': 0.00089, 'acetonitrile': 0.00034},
            360.0,
        )

        self.assertEqual(
            result.pair_methods[('water', 'acetonitrile')],
            'jouyban_acree',
        )
        self.assertTrue(any('used' in warning for warning in result.warnings))

    def test_estimator_uses_unifac_visco_for_missing_jouyban_pair(self):
        result = estimate_liquid_mixture_viscosity(
            {'acetonitrile': 0.5, 'benzene': 0.5},
            {'acetonitrile': 0.00034, 'benzene': 0.0006},
            298.15,
        )

        self.assertEqual(result.method, 'unifac_visco')
        self.assertEqual(
            result.pair_methods[('acetonitrile', 'benzene')],
            'unifac_visco',
        )
        self.assertEqual(result.warnings, ())
        self.assertNotAlmostEqual(
            result.value,
            math.exp(0.5 * math.log(0.00034) + 0.5 * math.log(0.0006)),
        )

    def test_estimator_warns_and_uses_zero_for_uncovered_pair(self):
        result = estimate_liquid_mixture_viscosity(
            {'water': 0.5, 'benzene': 0.5},
            {'water': 0.00089, 'benzene': 0.0006},
            298.15,
        )

        self.assertEqual(
            result.pair_methods[('water', 'benzene')],
            'zero',
        )
        self.assertTrue(result.warnings)
        self.assertAlmostEqual(
            result.value,
            math.exp(0.5 * math.log(0.00089) + 0.5 * math.log(0.0006)),
            places=18,
        )

    def test_unifac_visco_splits_ester_and_ketone_compound_groups(self):
        ethyl_acetate = _component_unifac_visco_groups('ethyl acetate')
        acetone = _component_unifac_visco_groups('acetone')
        acetic_acid = _component_unifac_visco_groups('acetic acid')
        numeric_groups = _component_unifac_visco_groups(
            'synthetic',
            component_groups={'synthetic': {21: 1, 19: 1, 43: 1}},
        )

        self.assertEqual(ethyl_acetate['std:1'], 2.0)
        self.assertEqual(ethyl_acetate['std:2'], 1.0)
        self.assertEqual(ethyl_acetate['std:77'], 1.0)
        self.assertNotIn('CH3COO', ethyl_acetate)
        self.assertEqual(acetone['std:1'], 2.0)
        self.assertEqual(acetone['std_main:9'], 1.0)
        self.assertEqual(acetic_acid['std:1'], 1.0)
        self.assertEqual(acetic_acid['std:42'], 1.0)
        self.assertNotIn('std:77', acetic_acid)
        self.assertEqual(numeric_groups['std:1'], 1.0)
        self.assertEqual(numeric_groups['std:2'], 1.0)
        self.assertEqual(numeric_groups['std:77'], 1.0)
        self.assertEqual(numeric_groups['std_main:9'], 1.0)
        self.assertEqual(numeric_groups['std:42'], 1.0)

    def test_unifac_visco_preserves_cycloalkane_groups(self):
        cyclohexane = _component_unifac_visco_groups('cyclohexane')
        cyclopentanone = _component_unifac_visco_groups('cyclopentanone')
        cyclohexanone = _component_unifac_visco_groups('cyclohexanone')

        self.assertEqual(cyclohexane, {'do:78': 6.0})
        self.assertEqual(cyclopentanone['do:78'], 3.0)
        self.assertEqual(cyclopentanone['std:2'], 1.0)
        self.assertEqual(cyclopentanone['std_main:9'], 1.0)
        self.assertEqual(cyclohexanone['do:78'], 4.0)
        self.assertEqual(cyclohexanone['std:2'], 1.0)
        self.assertEqual(cyclohexanone['std_main:9'], 1.0)

    def test_unifac_visco_accepts_native_numeric_smiles_fragmentation(self):
        cyclohexane = _component_unifac_visco_groups(
            'native_cycle_probe',
            component_smiles={'native_cycle_probe': 'C1CCCCC1'},
        )
        ester = _component_unifac_visco_groups(
            'native_ester_probe',
            component_smiles={'native_ester_probe': 'CC(=O)OCC'},
        )

        self.assertEqual(cyclohexane, {'do:78': 6.0})
        self.assertEqual(
            ester,
            {'std:1': 2.0, 'std:2': 1.0, 'std:77': 1.0},
        )

    def test_unifac_visco_does_not_confuse_cross_variant_numeric_ids(self):
        pyridine = _component_unifac_visco_groups(
            'native_pyridine_probe',
            component_smiles={'native_pyridine_probe': 'n1ccccc1'},
        )
        self.assertEqual(pyridine, {'std:37': 1.0})
        self.assertNotIn('std:9', pyridine)
        supplied_dortmund_pyridine = _component_unifac_visco_groups(
            'native_pyridine_probe',
            component_groups={'native_pyridine_probe': {9: 3, 37: 1}},
            component_group_variant='UNIFDMD',
            component_smiles={'native_pyridine_probe': 'n1ccccc1'},
        )
        self.assertEqual(supplied_dortmund_pyridine, {'std:37': 1.0})

        with self.assertRaisesRegex(
            LiquidMixtureViscosityError,
            'mapping unavailable',
        ):
            _component_unifac_visco_groups(
                'native_isocyanate_probe',
                component_smiles={'native_isocyanate_probe': 'CCN=C=O'},
            )

        with self.assertRaises(LiquidMixtureViscosityError):
            _component_unifac_visco_groups(
                'ambiguous_numeric_probe',
                component_groups={'ambiguous_numeric_probe': {118: 1}},
            )

        with self.assertRaisesRegex(
            LiquidMixtureViscosityError,
            'mapping unavailable',
        ):
            _component_unifac_visco_groups(
                'native_isocyanate_probe',
                component_groups={'native_isocyanate_probe': {1: 1, 118: 1}},
                component_group_variant='UNIFNIST',
                component_smiles={'native_isocyanate_probe': 'CCN=C=O'},
            )

    def test_unifac_visco_includes_combinatorial_term(self):
        ethanol = _component_unifac_visco_groups('ethanol')
        benzene = _component_unifac_visco_groups('benzene')
        x = [0.5113, 0.4887]

        combinatorial = _unifac_visco_combinatorial_excess_g_over_rt(
            [ethanol, benzene],
            x,
        )
        total = _unifac_visco_excess_g_over_rt(
            [ethanol, benzene],
            x,
            298.15,
        )

        self.assertNotAlmostEqual(combinatorial, 0.0)
        self.assertNotAlmostEqual(total, total - combinatorial)

    def test_estimator_uses_corrected_unifac_visco_ethanol_benzene_anchor(self):
        result = estimate_liquid_mixture_viscosity(
            {'ethanol': 0.5113, 'benzene': 0.4887},
            {
                'ethanol': 0.0010774308462863267,
                'benzene': 0.0005997344042534111,
            },
            298.15,
        )

        self.assertEqual(result.method, 'unifac_visco')
        self.assertAlmostEqual(result.excess_g_over_rt, -0.23891264910536866)
        self.assertAlmostEqual(result.value, 0.000637222061033296)


if __name__ == '__main__':
    unittest.main()
