import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from thermodynamics import create_thermodynamics
from thermodynamics_models.base import FluidPhaseEquilibrium
from thermodynamics_models.interaction_estimation import _fit_pair
from pfd_parser import PFDParser
from simulator import Simulator


class InteractionEstimationTests(unittest.TestCase):
    def _rule(self, model, source='UNIFDMD', **extra):
        return [{
            'model': model,
            'source': source,
            'policy': 'missing_only',
            'parameter_order': 'source',
            'Tmin_K': 293.15,
            'Tmax_K': 423.15,
            **extra,
        }]

    def test_modified_unifac_fits_and_freezes_uniquac_once(self):
        import thermodynamics_models.interaction_estimation as estimation

        with patch.object(estimation, '_fit_pair', wraps=_fit_pair) as fit_pair:
            thermo = create_thermodynamics(
                ['water', '2,3-pentanedione'],
                'UNIQUAC',
                interaction_estimation=self._rule('UNIQUAC'),
            )
            self.assertEqual(fit_pair.call_count, 1)
            interaction = thermo._uniquac_interaction_for_components(
                'water', '2,3-pentanedione'
            )
            self.assertIn('tau12_a', interaction)
            self.assertIn('tau12_b', interaction)
            self.assertEqual(interaction['tau12_c'], 0.0)
            self.assertNotEqual(interaction['tau12_d'], 0.0)
            self.assertEqual(interaction['tau12_e'], 0.0)
            details = thermo.estimated_interaction_metadata[
                ('2,3-pentanedione', 'water')
            ]
            self.assertLess(details['rmse_ln_gamma'], 0.05)
            self.assertTrue(details['fitted_once'])

            for T in (313.15, 373.15):
                thermo.activity_coefficients(
                    T, {'water': 0.9, '2,3-pentanedione': 0.1}
                )
            self.assertEqual(fit_pair.call_count, 1)

    def test_original_unifac_uses_two_directional_energy_parameters(self):
        thermo = create_thermodynamics(
            ['water', '2,3-pentanedione'],
            'UNIQUAC',
            interaction_estimation=self._rule('UNIQUAC', source='UNIFAC'),
        )
        interaction = thermo._uniquac_interaction_for_components(
            'water', '2,3-pentanedione'
        )
        self.assertIn('a12_cal_per_mol', interaction)
        self.assertIn('a21_cal_per_mol', interaction)
        self.assertNotIn('tau12_a', interaction)
        self.assertNotIn('tau21_a', interaction)
        details = thermo.estimated_interaction_metadata[
            ('2,3-pentanedione', 'water')
        ]
        self.assertEqual(details['source'], 'UNIFAC')

    def test_modified_unifac_fits_nrtl_with_fixed_alpha_and_linear_T(self):
        source = '''ONLINE_LOOKUP: false
THERMO_METHOD: NRTL
COMPONENTS:
    W | Water
    PD | 2,3-Pentanedione
INTERACTION_ESTIMATION:
    NRTL | source=UNIFDMD, policy=missing_only, parameter_order=source, alpha=0.25, Tmin=20 [C], Tmax=150 [C]
    W/PD | model=NRTL, alpha=0.35, Tmin=30 [C], Tmax=120 [C]
'''
        thermo = Simulator(PFDParser().parse(source)).initialize().thermo
        interaction = thermo._nrtl_interaction_for_components(
            'W', 'PD'
        )
        self.assertAlmostEqual(interaction['alpha12'], 0.35)
        self.assertEqual(interaction['tau12_e'], 0.0)
        self.assertEqual(interaction['tau21_e'], 0.0)
        self.assertNotEqual(interaction['tau12_f'], 0.0)
        self.assertNotEqual(interaction['tau21_f'], 0.0)
        details = thermo.estimated_interaction_metadata[
            ('PD', 'W')
        ]
        self.assertEqual(details['fit_Tmin_K'], 303.15)
        self.assertEqual(details['fit_Tmax_K'], 393.15)
        self.assertLess(details['rmse_ln_gamma'], 0.05)

    def test_explicit_and_database_interactions_precede_estimation(self):
        explicit = [{
            'component1': 'water',
            'component2': '2,3-pentanedione',
            'model': 'UNIQUAC',
            'a12_cal_per_mol': 12.0,
            'a21_cal_per_mol': 34.0,
        }]
        thermo = create_thermodynamics(
            ['water', '2,3-pentanedione'],
            'UNIQUAC',
            interaction_overrides=explicit,
            interaction_estimation=self._rule('UNIQUAC'),
        )
        self.assertEqual(thermo.estimated_interaction_metadata, {})
        interaction = thermo._uniquac_interaction_for_components(
            'water', '2,3-pentanedione'
        )
        self.assertEqual(interaction['a12_cal_per_mol'], 12.0)
        self.assertEqual(interaction['a21_cal_per_mol'], 34.0)

        database = create_thermodynamics(
            ['water', 'acrylic acid'],
            'UNIQUAC',
            interaction_estimation=self._rule('UNIQUAC'),
        )
        self.assertEqual(database.estimated_interaction_metadata, {})

    def test_always_policy_replaces_database_but_not_explicit_pfd_parameters(self):
        always = self._rule('UNIQUAC')
        always[0]['policy'] = 'always'
        thermo = create_thermodynamics(
            ['water', 'acrylic acid'],
            'UNIQUAC',
            interaction_estimation=always,
        )
        details = thermo.estimated_interaction_metadata[('acrylic acid', 'water')]
        self.assertEqual(details['policy'], 'always')
        interaction = thermo._uniquac_interaction_for_components(
            'water', 'acrylic acid'
        )
        self.assertIn('Estimated from UNIFDMD', interaction['comment'])

        explicit = [{
            'component1': 'water',
            'component2': 'acrylic acid',
            'model': 'UNIQUAC',
            'a12_cal_per_mol': 12.0,
            'a21_cal_per_mol': 34.0,
        }]
        overridden = create_thermodynamics(
            ['water', 'acrylic acid'],
            'UNIQUAC',
            interaction_overrides=explicit,
            interaction_estimation=always,
        )
        self.assertEqual(overridden.estimated_interaction_metadata, {})
        self.assertEqual(
            overridden._uniquac_interaction_for_components(
                'water', 'acrylic acid'
            )['a12_cal_per_mol'],
            12.0,
        )

    def test_contextual_henry_exclusion_is_not_fitted_as_a_liquid_interaction(self):
        thermo = create_thermodynamics(
            ['water', 'carbon monoxide'],
            'NRTL',
            interaction_estimation=self._rule('NRTL', source='UNIFAC'),
        )
        self.assertEqual(thermo.estimated_interaction_metadata, {})
        self.assertIsNone(
            thermo._nrtl_interaction_for_components('water', 'carbon monoxide')
        )
        self.assertTrue(any(
            'no molecular liquid interaction can be regressed' in warning
            for warning in thermo.warnings
        ))

    def test_contextual_henry_components_are_resolved_once_and_prefiltered(self):
        import thermodynamics_models.interaction_estimation as estimation
        import thermodynamics_models.unifac_models as unifac_models

        real_source = estimation._source_classes()['UNIFAC']

        class SourceMustNotBeConstructed(real_source):
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    'excluded source components must be filtered before model construction'
                )

        with (
            patch.dict(
                estimation._source_classes(),
                {'UNIFAC': SourceMustNotBeConstructed},
            ),
            patch.object(
                unifac_models,
                'resolve_component_unifac_groups',
                wraps=unifac_models.resolve_component_unifac_groups,
            ) as resolve_groups,
        ):
            thermo = create_thermodynamics(
                ['water', 'carbon monoxide', 'carbon dioxide'],
                'NRTL',
                interaction_estimation=self._rule('NRTL', source='UNIFAC'),
            )

        resolved_components = [
            call.args[0] for call in resolve_groups.call_args_list
        ]
        self.assertEqual(resolved_components.count('carbon monoxide'), 1)
        self.assertEqual(resolved_components.count('carbon dioxide'), 1)
        self.assertEqual(thermo.estimated_interaction_metadata, {})
        exclusions = [
            warning for warning in thermo.warnings
            if 'no molecular liquid interaction can be regressed' in warning
        ]
        self.assertEqual(len(exclusions), 2)

    def test_unifdmd_keeps_group_resolvable_volatile_condensable(self):
        source = create_thermodynamics(
            ['acetaldehyde', 'propionic acid'],
            'UNIFDMD',
        )

        self.assertIn('acetaldehyde', source.component_groups)
        self.assertNotIn('acetaldehyde', source.unifac_excluded_components)
        gamma = source.activity_coefficients(
            313.15,
            {'acetaldehyde': 0.5, 'propionic acid': 0.5},
        )
        self.assertTrue(math.isfinite(gamma['acetaldehyde']))
        self.assertTrue(math.isfinite(gamma['propionic acid']))

    def test_unifac_group_resolution_never_uses_pfd_symbols(self):
        import unifac

        source = '''ONLINE_LOOKUP: false
THERMO_METHOD: UNIFDMD
COMPONENTS:
    AcH_ALIAS | Acetaldehyde
    PA_ALIAS | Propionic Acid
'''
        with patch.object(
            unifac,
            'get_unifac_groups',
            wraps=unifac.get_unifac_groups,
        ) as groups:
            thermo = Simulator(PFDParser().parse(source)).initialize().thermo

        attempted = [call.args[0] for call in groups.call_args_list]
        self.assertNotIn('AcH_ALIAS', attempted)
        self.assertNotIn('PA_ALIAS', attempted)
        self.assertIn('AcH_ALIAS', thermo.component_groups)
        self.assertIn('PA_ALIAS', thermo.component_groups)

    def test_modified_unifac_estimates_acetaldehyde_liquid_pair(self):
        thermo = create_thermodynamics(
            ['acetaldehyde', 'propionic acid'],
            'UNIQUAC',
            interaction_estimation=self._rule('UNIQUAC'),
        )

        pair = ('acetaldehyde', 'propionic acid')
        self.assertIn(pair, thermo.estimated_interaction_metadata)
        self.assertIsNotNone(
            thermo._uniquac_interaction_for_components(*pair)
        )
        self.assertFalse(any(
            'no molecular liquid interaction can be regressed' in warning
            for warning in thermo.warnings
        ))

    def test_completed_fit_is_reused_from_runtime_cache(self):
        import thermodynamics_models.interaction_estimation as estimation

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / 'interaction_fits.sqlite'
            with (
                patch.object(estimation, '_FIT_CACHE_PATH', cache_path),
                patch.object(estimation, '_FIT_CACHE', None),
                patch.object(
                    estimation,
                    'least_squares',
                    wraps=estimation.least_squares,
                ) as optimizer,
            ):
                first = create_thermodynamics(
                    ['acetaldehyde', 'propionic acid'],
                    'UNIQUAC',
                    interaction_estimation=self._rule('UNIQUAC'),
                )
                first_calls = optimizer.call_count
                self.assertGreater(first_calls, 0)
                first_details = first.estimated_interaction_metadata[
                    ('acetaldehyde', 'propionic acid')
                ]
                self.assertFalse(first_details['fit_cache_hit'])

                estimation._FIT_CACHE = None
                real_source = estimation._source_classes()['UNIFDMD']

                class CacheHitMustNotConstructSource(real_source):
                    def __init__(self, *args, **kwargs):
                        raise AssertionError(
                            'persistent cache hit constructed its UNIFDMD source model'
                        )

                with patch.dict(
                    estimation._source_classes(),
                    {'UNIFDMD': CacheHitMustNotConstructSource},
                ):
                    second = create_thermodynamics(
                        ['acetaldehyde', 'propionic acid'],
                        'UNIQUAC',
                        interaction_estimation=self._rule('UNIQUAC'),
                    )
                self.assertEqual(optimizer.call_count, first_calls)
                second_details = second.estimated_interaction_metadata[
                    ('acetaldehyde', 'propionic acid')
                ]
                self.assertTrue(second_details['fit_cache_hit'])
                self.assertEqual(
                    first._uniquac_interaction_for_components(
                        'acetaldehyde', 'propionic acid'
                    ),
                    second._uniquac_interaction_for_components(
                        'acetaldehyde', 'propionic acid'
                    ),
                )
            estimation._FIT_CACHE = None

    def test_failed_fit_is_reused_from_runtime_cache(self):
        import thermodynamics_models.interaction_estimation as estimation

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / 'interaction_fit_failures.sqlite'
            failed = SimpleNamespace(
                success=False,
                x=np.zeros(6, dtype=float),
                message='synthetic optimizer failure',
            )
            with (
                patch.object(estimation, '_FIT_CACHE_PATH', cache_path),
                patch.object(estimation, '_FIT_CACHE', None),
                patch.object(
                    estimation,
                    'least_squares',
                    return_value=failed,
                ) as optimizer,
            ):
                first = create_thermodynamics(
                    ['acetaldehyde', 'propionic acid'],
                    'UNIQUAC',
                    interaction_estimation=self._rule('UNIQUAC'),
                )
                self.assertEqual(optimizer.call_count, 1)
                self.assertEqual(first.estimated_interaction_metadata, {})

                estimation._FIT_CACHE = None
                second = create_thermodynamics(
                    ['acetaldehyde', 'propionic acid'],
                    'UNIQUAC',
                    interaction_estimation=self._rule('UNIQUAC'),
                )
                self.assertEqual(optimizer.call_count, 1)
                self.assertEqual(second.estimated_interaction_metadata, {})
                self.assertTrue(any(
                    'synthetic optimizer failure' in warning
                    for warning in second.warnings
                ))
            estimation._FIT_CACHE = None

    def test_activity_interaction_pressure_and_temperature_caps(self):
        thermo = create_thermodynamics(
            ['water', 'propionic acid'],
            'UNIQUAC',
            activity_interaction_max_psat_bar=10.0,
            activity_interaction_max_temperature_K=423.15,
        )
        actual_T = 626.45
        interaction_T = thermo.activity_interaction_temperature(
            'water',
            'propionic acid',
            actual_T,
        )

        self.assertLessEqual(interaction_T, 423.15)
        self.assertLessEqual(thermo.Psat('water', interaction_T), 10.0 + 1e-7)
        self.assertLessEqual(
            thermo.Psat('propionic acid', interaction_T),
            10.0 + 1e-7,
        )
        hot_tau = thermo._uniquac_tau_matrix(actual_T)
        capped_tau = thermo._uniquac_tau_matrix(interaction_T)
        for hot_row, capped_row in zip(hot_tau, capped_tau):
            for hot_value, capped_value in zip(hot_row, capped_row):
                self.assertAlmostEqual(hot_value, capped_value, places=12)
        K = thermo.K_values(
            actual_T,
            1.01,
            {'water': 0.999, 'propionic acid': 0.001},
        )
        self.assertGreater(min(K.values()), 1.0)

        nrtl = create_thermodynamics(
            ['water', 'ethanol'],
            'NRTL',
            interaction_overrides=[{
                'component1': 'water',
                'component2': 'ethanol',
                'model': 'NRTL',
                'alpha12': 0.3,
                'tau12_c': 0.1,
                'tau12_d': 20.0,
                'tau12_e': 0.0,
                'tau12_f': 0.002,
                'tau21_c': -0.2,
                'tau21_d': 15.0,
                'tau21_e': 0.0,
                'tau21_f': -0.001,
                'tau_tref': 298.15,
            }],
            activity_interaction_max_psat_bar=1.0e6,
            activity_interaction_max_temperature_K=350.0,
        )
        hot, _ = nrtl._nrtl_matrices(500.0)
        capped, _ = nrtl._nrtl_matrices(350.0)
        self.assertEqual(hot, capped)

    def test_nrtl_manual_five_term_temperature_form_and_orientation(self):
        source = '''ONLINE_LOOKUP: false
THERMO_METHOD: NRTL
COMPONENTS:
    W | Water
    E | Ethanol
INTERACTION_PARAMETERS:
    W/E | model=NRTL, alpha=0.3, tau12_c=1.0, tau12_d=20.0, tau12_e=0.4, tau12_f=0.002, tau12_g=0.000001, tau21_c=-0.5, tau21_d=10.0, tau21_e=-0.3, tau21_f=-0.001, tau21_g=-0.000002, tau_tref=298.15
'''
        thermo = Simulator(PFDParser().parse(source)).initialize().thermo
        T = 350.0
        tau, _, _ = thermo._nrtl_cached_matrices(T)
        anchored = (298.15 - T) / T + math.log(T / 298.15)
        self.assertAlmostEqual(
            tau[0][1],
            1.0 + 20.0 / T + 0.4 * anchored + 0.002 * T + 0.000001 * T * T,
        )
        self.assertAlmostEqual(
            tau[1][0],
            -0.5 + 10.0 / T - 0.3 * anchored - 0.001 * T - 0.000002 * T * T,
        )
        reverse = thermo._nrtl_interaction_for_components('E', 'W')
        self.assertEqual(reverse['tau12_f'], -0.001)
        self.assertEqual(reverse['tau21_f'], 0.002)
        self.assertEqual(reverse['tau12_g'], -0.000002)
        self.assertEqual(reverse['tau21_g'], 0.000001)
        backend = thermo._compiled_activity_backend(T)
        self.assertIsNotNone(backend)
        compiled = backend.activity_coefficients([0.4, 0.6], T)
        thermo._compiled_activity_cache['all_temperatures'] = None
        thermo._activity_cache.clear()
        readable = thermo.activity_coefficients(
            T, {'W': 0.4, 'E': 0.6}
        )
        self.assertAlmostEqual(compiled[0], readable['W'], places=12)
        self.assertAlmostEqual(compiled[1], readable['E'], places=12)

        from compiled_lle import CompiledNRTLLLEBackend
        from compiled_vlle import CompiledActivityVLLEBackend
        lle_backend = CompiledNRTLLLEBackend.from_activity_backend(backend)
        vlle_backend = CompiledActivityVLLEBackend.from_thermo(thermo)
        self.assertIsNotNone(lle_backend)
        self.assertIsNotNone(vlle_backend)
        lle_backend.compile_kernels()
        vlle_backend.compile_kernels()

    def test_extrapolation_warning_requires_accepted_relevant_liquid(self):
        thermo = create_thermodynamics(['water', 'ethanol'], 'NRTL')
        thermo.estimated_interaction_metadata = {
            ('ethanol', 'water'): {
                'component1': 'water',
                'component2': 'ethanol',
                'model': 'NRTL',
                'source': 'UNIFDMD',
                'fit_Tmin_K': 290.0,
                'fit_Tmax_K': 400.0,
            }
        }
        before = tuple(thermo.warnings)
        thermo.activity_coefficients(450.0, {'water': 0.5, 'ethanol': 0.5})
        self.assertEqual(tuple(thermo.warnings), before)

        vapor = FluidPhaseEquilibrium(
            vapor_fraction=1.0,
            liquid1_fraction=0.0,
            liquid2_fraction=0.0,
            y={'water': 0.5, 'ethanol': 0.5},
            x1={},
            x2={},
            status='single_vapor',
            stability='test',
            extra={},
        )
        with patch.object(thermo, '_fluid_phase_equilibrium_TP', return_value=vapor):
            thermo.calculate_state(
                450.0, 1.0, 1.0, {'water': 0.5, 'ethanol': 0.5}, include=()
            )
        self.assertEqual(tuple(thermo.warnings), before)

        liquid = FluidPhaseEquilibrium(
            vapor_fraction=0.0,
            liquid1_fraction=1.0,
            liquid2_fraction=0.0,
            y={},
            x1={'water': 0.5, 'ethanol': 0.5},
            x2={},
            status='single_liquid',
            stability='test',
            extra={},
        )
        with patch.object(thermo, '_fluid_phase_equilibrium_TP', return_value=liquid):
            thermo.calculate_state(
                450.0, 1.0, 1.0, {'water': 0.5, 'ethanol': 0.5}, include=()
            )
            thermo.calculate_state(
                460.0, 1.0, 1.0, {'water': 0.5, 'ethanol': 0.5}, include=()
            )
        extrapolation = [
            warning for warning in thermo.warnings
            if 'frozen interaction is being extrapolated' in warning
        ]
        self.assertEqual(len(extrapolation), 1)


if __name__ == '__main__':
    unittest.main()
