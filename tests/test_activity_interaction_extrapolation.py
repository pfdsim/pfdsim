import math
import unittest

import numpy as np

import compiled_vlle
from interaction_parameters import (
    nrtl_binary_interaction,
    uniquac_binary_interaction,
)
from scripts.build_cas_interaction_parameters import (
    supplemental_literature_vle_activity_records,
)
from thermodynamics import create_thermodynamics
from thermodynamics_models.common import ThermodynamicsError


class ActivityInteractionExtrapolationTests(unittest.TestCase):
    @staticmethod
    def _record(model, component1, component2, *, clamp=None):
        record = {
            'component1': component1,
            'component2': component2,
            'model': model,
            'Tmin_K': 300.0,
            'Tmax_K': 350.0,
        }
        if clamp is not None:
            record['do_not_extrapolate'] = clamp
        if model == 'NRTL':
            record.update({
                'alpha12': 0.3,
                'tau12_c': 0.1,
                'tau12_d': 70.0,
                'tau21_c': -0.2,
                'tau21_d': 35.0,
            })
        else:
            record.update({
                'tau12_a': 0.1,
                'tau12_b': 70.0,
                'tau21_a': -0.2,
                'tau21_b': 35.0,
            })
        return record

    def test_omitted_or_false_policy_extrapolates_past_declared_range(self):
        for model in ('NRTL', 'UNIQUAC'):
            for clamp in (None, False):
                with self.subTest(model=model, clamp=clamp):
                    record = self._record(
                        model, 'water', 'ethanol', clamp=clamp
                    )
                    thermo = create_thermodynamics(
                        ['water', 'ethanol'],
                        model,
                        interaction_overrides=[record],
                    )
                    if model == 'NRTL':
                        actual = thermo._nrtl_matrices(400.0)[0][0][1]
                        expected = 0.1 + 70.0 / 400.0
                        clamped = 0.1 + 70.0 / 350.0
                    else:
                        actual = thermo._uniquac_tau_matrix(400.0)[0][1]
                        expected = math.exp(0.1 + 70.0 / 400.0)
                        clamped = math.exp(0.1 + 70.0 / 350.0)
                    self.assertAlmostEqual(actual, expected)
                    self.assertNotAlmostEqual(actual, clamped)
                    self.assertFalse(any(
                        'do_not_extrapolate' in warning
                        for warning in thermo.warnings
                    ))

    def test_true_policy_requires_a_valid_two_sided_range(self):
        for model in ('NRTL', 'UNIQUAC'):
            base = self._record(model, 'water', 'ethanol', clamp=True)
            cases = (
                ({key: value for key, value in base.items() if key != 'Tmin_K'},
                 'requires Tmin_K and Tmax_K'),
                (base | {'Tmin_K': 350.0, 'Tmax_K': 300.0},
                 '0 < Tmin_K < Tmax_K'),
                (base | {'do_not_extrapolate': 'true'},
                 'must be boolean'),
            )
            for record, message in cases:
                with self.subTest(model=model, message=message):
                    with self.assertRaisesRegex(ThermodynamicsError, message):
                        create_thermodynamics(
                            ['water', 'ethanol'],
                            model,
                            interaction_overrides=[record],
                        )

    def test_clamping_is_pair_specific_and_matches_compiled_backend(self):
        composition = {'water': 0.4, 'ethanol': 0.35, 'methanol': 0.25}
        for model in ('NRTL', 'UNIQUAC'):
            overrides = [
                self._record(model, 'water', 'ethanol', clamp=True),
                self._record(model, 'water', 'methanol'),
            ]
            with self.subTest(model=model):
                thermo = create_thermodynamics(
                    list(composition), model, interaction_overrides=overrides
                )
                if model == 'NRTL':
                    hot = thermo._nrtl_matrices(400.0)[0]
                    self.assertAlmostEqual(hot[0][1], 0.1 + 70.0 / 350.0)
                    self.assertAlmostEqual(hot[0][2], 0.1 + 70.0 / 400.0)
                else:
                    hot = thermo._uniquac_tau_matrix(400.0)
                    self.assertAlmostEqual(hot[0][1], math.exp(0.1 + 70.0 / 350.0))
                    self.assertAlmostEqual(hot[0][2], math.exp(0.1 + 70.0 / 400.0))

                backend = thermo._compiled_activity_backend(400.0)
                if backend is None:
                    self.skipTest(f'Compiled {model} backend is unavailable')
                x = [composition[component] for component in thermo.components]
                compiled = backend.activity_coefficients(x, 400.0)

                reference = create_thermodynamics(
                    list(composition), model, interaction_overrides=overrides
                )
                reference._compiled_activity_backend = lambda _T: None
                generic = reference.activity_coefficients(400.0, composition)
                for index, component in enumerate(thermo.components):
                    self.assertAlmostEqual(
                        compiled[index], generic[component], places=12
                    )

                thermo.activity_coefficients(400.0, composition)
                thermo.activity_coefficients(425.0, composition)
                warnings = [
                    warning for warning in thermo.warnings
                    if 'do_not_extrapolate=true' in warning
                ]
                self.assertEqual(len(warnings), 1)
                self.assertIn('350 K instead of 400 K', warnings[0])
                thermo.activity_coefficients(275.0, composition)
                warnings = [
                    warning for warning in thermo.warnings
                    if 'do_not_extrapolate=true' in warning
                ]
                self.assertEqual(len(warnings), 2)
                self.assertIn('300 K instead of 275 K', warnings[1])

                lle = thermo._compiled_lle_backend(400.0)
                self.assertIsNotNone(lle)
                self.assertEqual(lle.interaction_tmax[0, 1], 350.0)
                self.assertTrue(math.isinf(lle.interaction_tmax[0, 2]))

                vlle = thermo.compiled_vlle_backend()
                self.assertIsNotNone(vlle)
                if model == 'NRTL':
                    self.assertEqual(vlle.parameter_9[0, 1], 350.0)
                    self.assertTrue(math.isinf(vlle.parameter_9[0, 2]))
                else:
                    self.assertEqual(vlle.parameter_9[0, 1], 300.0)
                    self.assertEqual(vlle.parameter_9[1, 0], 350.0)
                    self.assertTrue(math.isinf(vlle.parameter_9[2, 0]))
                activity_gamma = getattr(compiled_vlle, '_activity_gamma', None)
                if activity_gamma is None:
                    self.skipTest('Compiled activity VLLE kernel is unavailable')
                vlle_gamma = activity_gamma(
                    vlle.model_id,
                    np.asarray(x, dtype=np.float64),
                    400.0,
                    vlle.integer_parameters,
                    vlle.parameter_0,
                    vlle.parameter_1,
                    vlle.parameter_2,
                    vlle.parameter_3,
                    vlle.parameter_4,
                    vlle.parameter_5,
                    vlle.parameter_6,
                    vlle.parameter_7,
                    vlle.parameter_8,
                    vlle.parameter_9,
                )
                for index in range(len(compiled)):
                    self.assertAlmostEqual(
                        vlle_gamma[index], compiled[index], places=12
                    )

    def test_water_propionic_uniquac_policy_is_built_and_runtime_visible(self):
        records, _, _ = supplemental_literature_vle_activity_records(
            [], 'UNIQUAC'
        )
        built = next(
            record for record in records
            if {record['cas1'], record['cas2']} == {'7732-18-5', '79-09-4'}
        )
        self.assertTrue(built['do_not_extrapolate'])
        self.assertEqual((built['Tmin_K'], built['Tmax_K']), (313.15, 373.15))

        uniquac = uniquac_binary_interaction('7732-18-5', '79-09-4')
        self.assertTrue(uniquac['do_not_extrapolate'])
        self.assertEqual(
            (uniquac['Tmin_K'], uniquac['Tmax_K']),
            (313.15, 373.15),
        )
        nrtl = nrtl_binary_interaction('7732-18-5', '79-09-4')
        self.assertNotIn('do_not_extrapolate', nrtl)

        thermo = create_thermodynamics(
            ['water', 'propionic acid'], 'UNIQUAC'
        )
        hot = thermo._uniquac_tau_matrix(626.45)
        at_limit = thermo._uniquac_tau_matrix(373.15)
        self.assertEqual(hot, at_limit)
        K = thermo.K_values(
            626.45,
            1.01,
            {'water': 0.999, 'propionic acid': 0.001},
        )
        self.assertGreater(min(K.values()), 1.0)

    def test_estimated_interactions_inherit_declared_clamping_policy(self):
        thermo = create_thermodynamics(
            ['water', '2,3-pentanedione'],
            'UNIQUAC',
            interaction_estimation=[{
                'model': 'UNIQUAC',
                'source': 'UNIFDMD',
                'policy': 'missing_only',
                'parameter_order': 'source',
                'do_not_extrapolate': True,
                'Tmin_K': 293.15,
                'Tmax_K': 423.15,
            }],
        )
        interaction = thermo._uniquac_interaction_for_components(
            'water', '2,3-pentanedione'
        )
        self.assertTrue(interaction['do_not_extrapolate'])
        self.assertEqual(interaction['Tmin_K'], 293.15)
        self.assertEqual(interaction['Tmax_K'], 423.15)
        self.assertEqual(
            thermo._uniquac_tau_matrix(600.0),
            thermo._uniquac_tau_matrix(423.15),
        )


if __name__ == '__main__':
    unittest.main()
