import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simulator import Simulator
from thermodynamics import create_thermodynamics
from unit_operations_base import UnitOperationError
from unit_operations_distillation import RigorousDistillation


class VLLEInitializerEdgeCaseTests(unittest.TestCase):
    def test_terminal_newton_failure_includes_colored_fallback_work(self):
        from equilibrium_stage_vlle import EquationOrientedVLLEColumn, VLLESolveFailure

        model = Mock(spec=EquationOrientedVLLEColumn)
        model.projection_enabled = False
        model.topology_policy = 'residual_gate'
        model.solver_options = {}
        model.unit = RigorousDistillation('FALLBACK-ACCOUNTING', None, {})
        model.unit._sparse_newton_solve = Mock(side_effect=[
            {'success': False, 'iterations': 2, 'function_evaluations': 7,
             'jacobian_evaluations': 2, 'residual_norm': 1.0, 'message': 'stalled'},
            {'success': False, 'iterations': 3, 'function_evaluations': 11,
             'jacobian_evaluations': 3, 'residual_norm': 0.5, 'message': 'stalled'},
        ])
        with self.assertRaises(VLLESolveFailure) as caught:
            EquationOrientedVLLEColumn.solve(model)
        self.assertEqual(caught.exception.work, {
            'solver_iterations': 5, 'function_evaluations': 18,
            'jacobian_evaluations': 5,
        })

    def test_failed_active_set_attempts_retain_work(self):
        from equilibrium_stage_vlle import (
            VLLEProfile, VLLESolveFailure, VLLETopologyCycle,
            _ActiveSetChange, solve_vlle_active_set,
        )

        for scenario in ('cycle', 'exhausted', 'newton'):
            with self.subTest(scenario=scenario):
                profile = VLLEProfile([350.0], [{'a': 0.5, 'b': 0.5}],
                                      [1.0], [1.0], 0.0, 0.0, [None])
                unit = RigorousDistillation('ACCOUNTING', None, {
                    'vlle_max_topology_updates': 1 if scenario == 'exhausted' else 3,
                })
                change = _ActiveSetChange(
                    profile, [True], reason='test', residual_norm=1.0
                )
                change.add_solver_progress(
                    iterations=2, function_evaluations=7, jacobian_evaluations=2
                )
                if scenario == 'cycle':
                    terminal = _ActiveSetChange(
                        profile, [False], reason='test', residual_norm=1.0
                    )
                    terminal.add_solver_progress(
                        iterations=3, function_evaluations=11, jacobian_evaluations=3
                    )
                else:
                    terminal = VLLESolveFailure('VLLE MESH failed', {
                        'solver_iterations': 3, 'function_evaluations': 11,
                        'jacobian_evaluations': 3,
                    })
                model = Mock()
                model.projection_direction_assessments = 2
                model.projection_checks = 1
                model.solve.side_effect = [change, terminal]
                with patch('equilibrium_stage_vlle.EquationOrientedVLLEColumn',
                           return_value=model):
                    with self.assertRaises(VLLESolveFailure) as caught:
                        solve_vlle_active_set(
                            unit, None, [], ['a', 'b'], [1.0], 1.0, 0.0,
                            {'kind': 'molar', 'value': 1.0}, 250.0, 450.0,
                            1.0, 1.0, {'a': 1.0, 'b': 1.0}, profile, {},
                            initial_active=[False],
                        )
                work = caught.exception.work
                self.assertEqual(work['jacobian_evaluations'],
                                 2 if scenario == 'exhausted' else 5)
                self.assertEqual(work['function_evaluations'],
                                 7 if scenario == 'exhausted' else 18)
                self.assertEqual(work['vlle_topology_solves'],
                                 1 if scenario == 'exhausted' else 2)
                self.assertEqual(work['vlle_projection_checks'],
                                 work['vlle_topology_solves'])
                if scenario == 'cycle':
                    self.assertIsInstance(caught.exception, VLLETopologyCycle)

    def test_two_stage_log_profile_retains_both_endpoints(self):
        thermo = create_thermodynamics(['butanol', 'water'], 'NRTL')
        feed = thermo.calculate_state(
            298.15, 1.0, 100.0, {'butanol': 0.4, 'water': 0.6},
            phase='liquid', flash=False,
        )
        unit = RigorousDistillation('TWO-STAGE-SEED', thermo, {
            'initializer': 'azeotropic',
        })
        candidate = {
            'name': 'provided', 'order': 2, 'T': 366.18958,
            'composition': {'butanol': 0.22459337, 'water': 0.77540663},
        }
        for feed_stage in (1, 2):
            with self.subTest(feed_stage=feed_stage):
                initial = unit._initial_guess(
                    feed, ['butanol', 'water'], feed.composition,
                    2, feed_stage, 1.2, 1.0, [1.0, 1.0], 'total', 0.0,
                    {'kind': 'molar', 'value': 80.0}, [], 280.0, 450.0,
                    azeotrope_candidates=[candidate],
                    azeotropic_profile='log_feed_anchor',
                )
                endpoints = initial['azeotropic_endpoints']
                for actual, expected in zip(
                    initial['x'], [endpoints['x_top'], endpoints['x_bottom']]
                ):
                    for comp in feed.composition:
                        self.assertAlmostEqual(actual[comp], expected[comp])

    def test_duplicate_candidate_names_preserve_component_inventory(self):
        components = ['a', 'b', 'c']
        thermo = SimpleNamespace(props={
            comp: SimpleNamespace(MW=mw)
            for comp, mw in zip(components, [20.0, 40.0, 60.0])
        })
        feed = SimpleNamespace(
            F=100.0, T=300.0, composition={'a': 0.5, 'b': 0.3, 'c': 0.2}
        )
        unit = RigorousDistillation('DUPLICATE-NAMES', thermo, {})
        candidates = [
            {'name': 'same', 'order': 2, 'T': 350.0,
             'composition': {'a': 0.5, 'b': 0.5, 'c': 0.0}},
            {'name': 'same', 'order': 2, 'T': 370.0,
             'composition': {'a': 0.5, 'b': 0.0, 'c': 0.5}},
        ]
        for kind, target in [('molar', 50.0), ('mass', 1700.0)]:
            with self.subTest(kind=kind):
                endpoints = unit._azeotropic_endpoint_initial_guess(
                    feed, components, {'kind': kind, 'value': target},
                    8, 2.0, 1.0, candidates=candidates,
                )
                self.assertIsNotNone(endpoints)
                for comp in components:
                    self.assertAlmostEqual(
                        endpoints['D'] * endpoints['x_top'][comp]
                        + endpoints['B'] * endpoints['x_bottom'][comp],
                        feed.F * feed.composition[comp],
                    )
                distillate_amount = endpoints['D']
                if kind == 'mass':
                    distillate_amount *= sum(
                        endpoints['x_top'][comp] * thermo.props[comp].MW
                        for comp in components
                    )
                self.assertAlmostEqual(distillate_amount, target)

    def test_invalid_supplied_composition_is_rejected(self):
        for value in (-0.1, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                unit = RigorousDistillation('INVALID-SEED', None, {
                    'vlle_azeotrope_temperature': 350.0,
                    'vlle_azeotrope_composition': {
                        'a': 0.4, 'b': 0.6, 'c': value,
                    },
                })
                with self.assertRaisesRegex(
                    UnitOperationError, 'finite and nonnegative'
                ):
                    unit._provided_vlle_azeotrope_candidates(['a', 'b', 'c'])

    def test_failed_thermodynamic_trials_do_not_abort_multistart_search(self):
        def unavailable_activity(*args):
            raise ValueError('outside thermodynamic model domain')

        thermo = SimpleNamespace(
            bubble_point_T=lambda *args: 350.0,
            activity_coefficients=unavailable_activity,
        )
        unit = RigorousDistillation('FAILED-TRIAL', thermo, {})
        from scipy.optimize import least_squares

        with (
            patch.object(unit, '_temperature_bounds', return_value=(250.0, 450.0)),
            patch('scipy.optimize.least_squares', wraps=least_squares) as solve,
        ):
            self.assertEqual(
                unit._binary_vlle_azeotropes(('a', 'b'), ['a', 'b'], 1.0), []
            )
            self.assertEqual(solve.call_count, 12)
            solve.reset_mock()
            self.assertEqual(unit._ternary_vlle_azeotropes(
                ('a', 'b', 'c'), ['a', 'b', 'c'], 1.0, []
            ), [])
            self.assertEqual(solve.call_count, 3)


class RigorousDistillationVLLETests(unittest.TestCase):
    def test_lactic_recovery_column_retains_gibbs_favorable_stage_13_lle(self):
        root = Path(__file__).resolve().parents[1]
        simulator = Simulator.from_file(
            root / 'examples' / 'lactic_acid_dehydration_pbr.pfd'
        ).initialize()
        thermo = simulator.thermo_packages['global']
        column = simulator.solver.units['C-401']
        feed = thermo.calculate_state(
            298.48253699169915,
            0.16,
            79.79197900206256,
            {
                'H2O': 0.4393767665649053,
                'AA': 0.25149758937416866,
                'AcH': 9.632033783440319e-8,
                'PA': 0.007609656944632908,
                'MIBK': 0.3015158904858782,
            },
            phase='liquid',
            flash=False,
        )
        feed.thermo_scope = 'global'
        column.solve_context = {
            'recycle_evaluation': 1,
            'recycle_final_pass': False,
            'expensive_diagnostics': True,
        }
        result = column.solve({'feed': feed})
        column.solve_context = {}
        performance = result.performance

        self.assertEqual(
            performance['vlle_active_stages'], list(range(1, 14))
        )
        self.assertEqual(
            performance['vlle_topology'], 'L' * 13 + '.' * 7
        )

        stage = 12
        temperature = performance['stage_temperatures_C'][stage] + 273.15
        overall = performance['stage_liquid_compositions'][stage]
        phase1 = performance['stage_liquid1_compositions'][stage]
        phase2 = performance['stage_liquid2_compositions'][stage]
        beta = performance['stage_liquid2_fractions'][stage]
        gamma1 = thermo.activity_coefficients(temperature, phase1)
        gamma2 = thermo.activity_coefficients(temperature, phase2)
        components = tuple(overall)

        equilibrium_residual = max(
            abs(
                math.log(max(phase1[component] * gamma1[component], 1e-300))
                - math.log(max(
                    phase2[component] * gamma2[component], 1e-300
                ))
            )
            for component in components
        )
        material_residual = max(
            abs(
                overall[component]
                - (1.0 - beta) * phase1[component]
                - beta * phase2[component]
            )
            for component in components
        )
        phase_distance = sum(
            abs(phase1[component] - phase2[component])
            for component in components
        ) / len(components)

        def dimensionless_gibbs(composition, gamma):
            return sum(
                composition[component]
                * math.log(max(
                    composition[component] * gamma[component], 1e-300
                ))
                for component in components
            )

        homogeneous_gibbs = dimensionless_gibbs(
            overall,
            thermo.activity_coefficients(temperature, overall),
        )
        split_gibbs = (
            (1.0 - beta) * dimensionless_gibbs(phase1, gamma1)
            + beta * dimensionless_gibbs(phase2, gamma2)
        )

        self.assertGreater(min(beta, 1.0 - beta), 0.4)
        self.assertGreater(phase_distance, 0.1)
        self.assertLess(equilibrium_residual, 1e-8)
        self.assertLess(material_residual, 1e-10)
        self.assertGreater(homogeneous_gibbs - split_gibbs, 0.002)

    @staticmethod
    def _nrtl_rk_case(stages=16, condenser_type='total'):
        thermo = create_thermodynamics(
            ['ethanol', 'water', 'benzene'],
            'NRTL-RK',
        )
        feed = thermo.calculate_state(
            298.15,
            5.0,
            20.0,
            {'ethanol': 0.35, 'water': 0.25, 'benzene': 0.40},
            phase='liquid',
            flash=False,
        )
        params = {
            'N_stages': stages,
            'feed_stage': max(1, stages // 2),
            'reflux_ratio': 2.0,
            'D_to_F': 0.4,
            'P_condenser': 5.0,
            'P_drop_per_stage': 0.0,
            'condenser_type': condenser_type,
            'stage_phase_model': 'VLLE',
            'mesh_tolerance': 1e-5,
            'max_iterations': 100,
            'max_jacobian_evaluations': 100,
        }
        return thermo, feed, params

    def test_nrtl_rk_five_bar_column_solves_all_vlle_stages(self):
        thermo, feed, params = self._nrtl_rk_case()
        result = RigorousDistillation('VLLE-5BAR', thermo, params).solve(
            {'feed': feed}
        )
        performance = result.performance

        self.assertEqual(performance['stage_phase_model'], 'VLLE')
        self.assertEqual(
            performance['liquid_phase_routing'],
            'co_routed_equilibrium',
        )
        self.assertEqual(performance['vlle_topology'], 'L' * 16)
        self.assertEqual(performance['stage_phase_counts'], [3] * 16)
        self.assertEqual(performance['vlle_active_stages'], list(range(1, 17)))
        self.assertLess(performance['mesh_residual'], 1e-5)
        self.assertLess(performance['component_balance_error'], 1e-7)
        self.assertEqual(performance['jacobian_method'], 'vlle_semi_analytic_local_thermo')
        self.assertEqual(
            performance['vlle_vapor_fugacity_closure'],
            'shared_gamma_phi',
        )
        self.assertLess(performance['vlle_max_log_fugacity_residual'], 1e-7)
        self.assertLess(
            performance['vlle_max_vapor_liquid_log_fugacity_residual'],
            1e-7,
        )
        self.assertIn(
            performance['vlle_max_fugacity_residual_stage'],
            range(1, 17),
        )
        self.assertLessEqual(performance['vlle_projection_checks'], 1)
        self.assertEqual(
            performance['vlle_final_topology_projection_checks'], 0
        )

        distillate = result.outlet_streams['distillate']
        bottoms = result.outlet_streams['bottoms']
        self.assertAlmostEqual(performance['T_top_C'], 111.4784, places=2)
        self.assertAlmostEqual(performance['T_bottom_C'], 111.5139, places=2)
        self.assertAlmostEqual(distillate.composition['ethanol'], 0.31081, places=4)
        self.assertAlmostEqual(distillate.composition['water'], 0.25796, places=4)
        self.assertAlmostEqual(bottoms.composition['benzene'], 0.37918, places=4)
        self.assertAlmostEqual(
            result.heat_duty / 3600.0,
            performance['condenser_duty_kW'] + performance['reboiler_duty_kW'],
            places=4,
        )

    def test_vlle_projection_advanced_settings_are_configurable(self):
        thermo, feed, params = self._nrtl_rk_case(stages=6)
        params.update({
            'vlle_seed': 'homogeneous',
            'vlle_homogeneous_initializer': 'cmo',
            'vlle_projection_gate_fraction': 0.2,
            'vlle_projection_contraction_ratio': 0.4,
            'vlle_projection_candidate_streak': 3,
        })
        result = RigorousDistillation(
            'VLLE-PROJECTION-SETTINGS', thermo, params
        ).solve({'feed': feed})
        performance = result.performance

        self.assertEqual(performance['initializer'], 'vlle_homogeneous')
        self.assertEqual(performance['vlle_seed_requested'], 'homogeneous')
        self.assertEqual(performance['vlle_homogeneous_initializer'], 'cmo')
        self.assertFalse(performance['vlle_seed_fallback'])
        self.assertEqual(performance['vlle_projection_gate_fraction'], 0.2)
        self.assertEqual(
            performance['vlle_projection_contraction_ratio'], 0.4
        )
        self.assertEqual(performance['vlle_projection_candidate_streak'], 3)
        self.assertTrue(performance['vlle_projection_enabled'])
        self.assertLess(performance['mesh_residual'], 1e-5)

        invalid_settings = (
            (
                'vlle_projection_gate_fraction',
                0.5,
                'vlle_projection_gate_fraction must be between 0 and 0.5',
            ),
            (
                'vlle_projection_contraction_ratio',
                1.0,
                'vlle_projection_contraction_ratio must be between 0 and 1',
            ),
        )
        for name, value, message in invalid_settings:
            with self.subTest(name=name):
                invalid = dict(params)
                invalid[name] = value
                with self.assertRaisesRegex(UnitOperationError, message):
                    RigorousDistillation(
                        f'VLLE-PROJECTION-INVALID-{name}', thermo, invalid
                    ).solve({'feed': feed})

        invalid_initializer = dict(params)
        invalid_initializer['vlle_homogeneous_initializer'] = 'not_real'
        with self.assertRaisesRegex(
            UnitOperationError,
            'initializer must be estimate, coarse_rigorous, cmo, cmo_hvap, or azeotropic',
        ):
            RigorousDistillation(
                'VLLE-HOMOGENEOUS-INVALID-INITIALIZER',
                thermo,
                invalid_initializer,
            ).solve({'feed': feed})

        provided = RigorousDistillation(
            'VLLE-PROVIDED-AZEOTROPES',
            thermo,
            {
                'vlle_azeotropes': [
                    {
                        'name': 'ternary-marker',
                        'T_K': 350.0,
                        'composition': {
                            'ethanol': 2.0,
                            'water': 1.0,
                            'benzene': 1.0,
                        },
                    }
                ],
            },
        )._provided_vlle_azeotrope_candidates(
            ['ethanol', 'water', 'benzene']
        )
        self.assertEqual(provided[0]['name'], 'ternary-marker')
        self.assertEqual(provided[0]['order'], 3)
        self.assertEqual(provided[0]['composition']['ethanol'], 0.5)

        with self.assertRaisesRegex(
            UnitOperationError,
            'contains unknown component',
        ):
            RigorousDistillation(
                'VLLE-PROVIDED-AZEOTROPE-INVALID',
                thermo,
                {
                    'vlle_azeotrope_temperature': 350.0,
                    'vlle_azeotrope_composition': {
                        'ethanol': 0.5,
                        'not-a-component': 0.5,
                    },
                },
            )._provided_vlle_azeotrope_candidates(
                ['ethanol', 'water', 'benzene']
            )

    def test_vlle_mode_retains_vle_stages_for_homogeneous_profile(self):
        thermo = create_thermodynamics(
            ['ethanol', 'water', 'benzene'],
            'UNIFAC',
        )
        feed = thermo.calculate_state(
            298.15,
            1.0,
            20.0,
            {'ethanol': 0.75, 'water': 0.10, 'benzene': 0.15},
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'VLLE-HOMOGENEOUS',
            thermo,
            {
                'N_stages': 12,
                'feed_stage': 6,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'stage_phase_model': 'VLLE',
                'mesh_tolerance': 1e-5,
            },
        ).solve({'feed': feed})

        self.assertEqual(result.performance['vlle_topology'], '.' * 12)
        self.assertEqual(result.performance['stage_phase_counts'], [2] * 12)
        self.assertTrue(all(
            value == 0.0
            for value in result.performance['stage_liquid2_fractions']
        ))
        self.assertAlmostEqual(
            result.outlet_streams['distillate'].composition['ethanol'],
            0.53666943,
            places=6,
        )
        self.assertLess(result.performance['component_balance_error'], 1e-7)

    def test_vlle_mode_supports_mixed_condenser_phase_outlets(self):
        thermo, feed, params = self._nrtl_rk_case(
            stages=12,
            condenser_type='mixed',
        )
        params['distillate_vapor_fraction'] = 0.25
        result = RigorousDistillation('VLLE-MIXED', thermo, params).solve(
            {'feed': feed}
        )

        distillate = result.outlet_streams['distillate']
        liquid = result.outlet_streams['distillate_liquid']
        vapor = result.outlet_streams['distillate_vapor']
        self.assertAlmostEqual(distillate.F, 8.0, places=7)
        self.assertAlmostEqual(liquid.F, 6.0, places=7)
        self.assertAlmostEqual(vapor.F, 2.0, places=7)
        self.assertAlmostEqual(liquid.F + vapor.F, distillate.F, places=10)
        self.assertLess(result.performance['component_balance_error'], 1e-7)
        self.assertAlmostEqual(
            result.heat_duty / 3600.0,
            result.performance['condenser_duty_kW']
            + result.performance['reboiler_duty_kW'],
            places=4,
        )

    def test_vlle_mode_rejects_unimplemented_phase_routing(self):
        thermo = create_thermodynamics(
            ['ethanol', 'water', 'benzene'],
            'UNIFAC',
        )
        feed = thermo.calculate_state(
            298.15,
            1.0,
            20.0,
            {'ethanol': 0.35, 'water': 0.25, 'benzene': 0.40},
            phase='liquid',
            flash=False,
        )
        common = {
            'N_stages': 8,
            'feed_stage': 4,
            'reflux_ratio': 2.0,
            'D_to_F': 0.4,
            'P_condenser': 1.0,
            'stage_phase_model': 'VLLE',
        }
        with self.assertRaisesRegex(UnitOperationError, 'removed.*VLLE'):
            RigorousDistillation(
                'VLLE-DECANTER',
                thermo,
                {**common, 'condenser_type': 'decanter'},
            ).solve({'feed': feed})
        with self.assertRaisesRegex(UnitOperationError, 'side draws'):
            RigorousDistillation(
                'VLLE-SIDE',
                thermo,
                {
                    **common,
                    'side_draws': [
                        {'stage': 4, 'phase': 'liquid', 'flow': 1.0}
                    ],
                },
            ).solve({'feed': feed})

    def test_vlle_mode_supports_mass_distillate_specification(self):
        thermo, feed, params = self._nrtl_rk_case()
        reference_composition = {
            'ethanol': 0.3108141416332551,
            'water': 0.2579577018625229,
            'benzene': 0.43122815650422197,
        }
        target_mass_flow = 8.0 * thermo.mixture_MW(reference_composition)
        params.pop('D_to_F')
        params['distillate_mass_flow'] = target_mass_flow

        result = RigorousDistillation('VLLE-MASS', thermo, params).solve(
            {'feed': feed}
        )

        self.assertAlmostEqual(
            result.outlet_streams['distillate'].mass_flow(),
            target_mass_flow,
            places=5,
        )
        self.assertAlmostEqual(
            result.outlet_streams['distillate'].F,
            8.0,
            places=5,
        )
        self.assertLess(result.performance['mesh_residual'], 1e-5)
        self.assertLess(result.performance['component_balance_error'], 1e-7)

    def test_vlle_active_set_adds_liquid_phases_during_newton(self):
        thermo = create_thermodynamics(
            ['ethanol', 'water', 'benzene'],
            'UNIFAC',
        )
        feed = thermo.calculate_state(
            298.15,
            1.0,
            20.0,
            {'ethanol': 0.55, 'water': 0.15, 'benzene': 0.30},
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'VLLE-APPEAR',
            thermo,
            {
                'N_stages': 12,
                'feed_stage': 6,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'stage_phase_model': 'VLLE',
                'vlle_seed': 'cheap',
                'vlle_initial_topology': 'all_vle',
                'mesh_tolerance': 1e-5,
                'max_iterations': 100,
                'max_jacobian_evaluations': 100,
            },
        ).solve({'feed': feed})

        history = result.performance['vlle_topology_history']
        self.assertEqual(history[0], '.' * 12)
        self.assertIn('L' * 6 + '.' * 6, history)
        self.assertEqual(history[-1], 'L' * 10 + '.' * 2)
        self.assertGreaterEqual(result.performance['vlle_topology_solves'], 2)
        self.assertLess(result.performance['mesh_residual'], 1e-5)
        self.assertLess(result.performance['component_balance_error'], 1e-7)

    def test_vlle_active_set_removes_absent_seed_phases(self):
        thermo = create_thermodynamics(
            ['ethanol', 'water', 'benzene'],
            'UNIFAC',
        )
        feed = thermo.calculate_state(
            298.15,
            1.0,
            20.0,
            {'ethanol': 0.75, 'water': 0.10, 'benzene': 0.15},
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'VLLE-DISAPPEAR',
            thermo,
            {
                'N_stages': 12,
                'feed_stage': 6,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'stage_phase_model': 'VLLE',
                'vlle_initial_topology': 'all_vlle',
                'mesh_tolerance': 1e-5,
            },
        ).solve({'feed': feed})

        history = result.performance['vlle_topology_history']
        self.assertEqual(history[0], 'L' * 12)
        self.assertEqual(history[1], '.' * 12)
        self.assertEqual(result.performance['vlle_topology'], '.' * 12)
        self.assertEqual(result.performance['stage_phase_counts'], [2] * 12)
        self.assertTrue(all(
            value == 0.0
            for value in result.performance['stage_liquid2_fractions']
        ))

    def test_vlle_recycle_warm_start_reuses_profile_and_topology(self):
        thermo, feed, params = self._nrtl_rk_case(stages=12)
        column = RigorousDistillation('VLLE-RECYCLE-WARM', thermo, params)
        column.solve_context = {'recycle_evaluation': 1}
        first = column.solve({'feed': feed})

        column.solve_context = {'recycle_evaluation': 2}
        second = column.solve({'feed': feed})
        column.solve_context = {}

        self.assertEqual(second.performance['initializer'], 'previous_recycle')
        self.assertEqual(
            second.performance['vlle_topology_history'][0],
            first.performance['vlle_topology'],
        )
        self.assertEqual(
            second.performance['vlle_topology'],
            first.performance['vlle_topology'],
        )
        self.assertLessEqual(second.performance['vlle_topology_solves'], 2)
        self.assertLess(
            second.performance['function_evaluations'],
            first.performance['function_evaluations'],
        )
        self.assertLess(second.performance['mesh_residual'], 1e-5)
        self.assertLess(second.performance['component_balance_error'], 1e-7)

    def test_vlle_mode_supports_partial_vapor_condenser(self):
        thermo, feed, params = self._nrtl_rk_case(
            stages=12,
            condenser_type='partial',
        )
        result = RigorousDistillation('VLLE-PARTIAL', thermo, params).solve(
            {'feed': feed}
        )

        distillate = result.outlet_streams['distillate']
        self.assertAlmostEqual(distillate.F, 8.0, places=7)
        self.assertAlmostEqual(distillate.vapor_fraction, 1.0, places=12)
        self.assertNotIn('distillate_liquid', result.outlet_streams)
        self.assertNotIn('distillate_vapor', result.outlet_streams)
        self.assertEqual(result.performance['vlle_topology'], 'L' * 12)
        self.assertLess(result.performance['component_balance_error'], 1e-7)

    def test_vlle_mode_supports_multiple_feed_stages(self):
        thermo = create_thermodynamics(
            ['ethanol', 'water', 'benzene'],
            'NRTL-RK',
        )
        feed = thermo.calculate_state(
            298.15,
            5.0,
            12.0,
            {'ethanol': 0.40, 'water': 0.20, 'benzene': 0.40},
            phase='liquid',
            flash=False,
        )
        wash = thermo.calculate_state(
            298.15,
            5.0,
            8.0,
            {'ethanol': 0.275, 'water': 0.325, 'benzene': 0.40},
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'VLLE-MULTIFEED',
            thermo,
            {
                'N_stages': 12,
                'feed_stage': 6,
                'feed_stages': {'feed': 7, 'wash': 3},
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 5.0,
                'P_drop_per_stage': 0.0,
                'stage_phase_model': 'VLLE',
                'mesh_tolerance': 1e-5,
                'max_iterations': 100,
                'max_jacobian_evaluations': 100,
            },
        ).solve({'feed': feed, 'wash': wash})

        feed_metadata = {
            item['port']: item['stage'] for item in result.performance['feeds']
        }
        self.assertEqual(feed_metadata, {'feed': 7, 'wash': 3})
        self.assertEqual(result.performance['vlle_topology'], 'L' * 12)
        self.assertAlmostEqual(result.outlet_streams['distillate'].F, 8.0, places=7)
        self.assertLess(result.performance['mesh_residual'], 1e-5)
        self.assertLess(result.performance['component_balance_error'], 1e-7)

    def test_binary_invariant_vlle_column_converges(self):
        thermo = create_thermodynamics(
            ['water', 'chloroform'],
            'UNIFNIST',
        )
        feed = thermo.calculate_state(
            298.15,
            1.01325,
            20.0,
            {'water': 0.5, 'chloroform': 0.5},
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'VLLE-BINARY',
            thermo,
            {
                'N_stages': 6,
                'feed_stage': 3,
                'reflux_ratio': 2.0,
                'D_to_F': 0.5,
                'P_condenser': 1.01325,
                'P_drop_per_stage': 0.0,
                'stage_phase_model': 'VLLE',
                'vlle_seed': 'cheap',
                'mesh_tolerance': 1e-5,
                'max_iterations': 120,
                'max_jacobian_evaluations': 120,
            },
        ).solve({'feed': feed})

        performance = result.performance
        self.assertEqual(performance['vlle_topology'], 'L' * 6)
        self.assertEqual(performance['stage_phase_counts'], [3] * 6)
        self.assertLess(performance['mesh_residual'], 1e-5)
        self.assertLess(performance['component_balance_error'], 1e-7)
        self.assertGreater(
            abs(
                performance['stage_liquid1_compositions'][0]['water']
                - performance['stage_liquid2_compositions'][0]['water']
            ),
            0.5,
        )
        self.assertAlmostEqual(
            result.outlet_streams['distillate'].composition['water'],
            1.0 - result.outlet_streams['bottoms'].composition['water'],
            places=6,
        )

    def test_butanol_water_projection_contracts_quickly_near_pure_bottoms(self):
        thermo = create_thermodynamics(['butanol', 'water'], 'NRTL')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'butanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'VLLE-BUTANOL-BOUNDARY',
            thermo,
            {
                'N_stages': 20,
                'feed_stage': 10,
                'reflux_ratio': 1.2,
                'D_to_F': 0.773,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'condenser_type': 'total',
                'stage_phase_model': 'VLLE',
                'vlle_seed': 'cheap',
                'max_iterations': 60,
                'max_jacobian_evaluations': 60,
            },
        ).solve({'feed': feed})
        performance = result.performance
        projected = [
            event
            for event in performance['vlle_topology_events']
            if event['reason'] == 'projected_newton_boundary'
        ]

        self.assertEqual(performance['vlle_topology'], 'L' * 14 + '.' * 6)
        self.assertLessEqual(len(projected), 3)
        self.assertFalse(performance['jacobian_fallback'])
        self.assertLessEqual(
            max((event['jacobian_evaluations'] for event in projected), default=0), 7
        )
        self.assertLessEqual(performance['jacobian_evaluations'], 25)
        self.assertLessEqual(performance['vlle_projection_checks'], 13)
        self.assertLess(performance['mesh_residual'], 1e-5)
        self.assertLess(performance['component_balance_error'], 1e-7)
        self.assertEqual(performance['vlle_projection_phase_fraction_min'], 1e-6)
        self.assertAlmostEqual(
            result.outlet_streams['bottoms'].composition['butanol'],
            0.997309789,
            places=7,
        )

        final_beta = performance['stage_liquid2_fractions'][13]
        self.assertLess(min(final_beta, 1.0 - final_beta), 0.15)
        self.assertGreater(min(final_beta, 1.0 - final_beta), 1e-3)
        self.assertEqual(
            performance['vlle_final_topology_projection_checks'], 0
        )

    def test_butanol_water_auto_seed_falls_back_but_homogeneous_stays_strict(self):
        thermo = create_thermodynamics(['butanol', 'water'], 'NRTL')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'butanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )
        params = {
            'N_stages': 20,
            'feed_stage': 10,
            'reflux_ratio': 1.2,
            'D_to_F': 0.77,
            'P_condenser': 1.0,
            'P_drop_per_stage': 0.0,
            'condenser_type': 'total',
            'stage_phase_model': 'VLLE',
            'max_iterations': 60,
            'max_jacobian_evaluations': 60,
        }
        original_solve = RigorousDistillation.solve

        def fail_homogeneous_seed(unit, inlets):
            if unit.unit_id.endswith('_vlle_seed'):
                raise UnitOperationError(
                    f"{unit.unit_id} MESH solve failed (forced regression case)"
                )
            return original_solve(unit, inlets)

        with patch.object(RigorousDistillation, 'solve', fail_homogeneous_seed):
            automatic = RigorousDistillation(
                'VLLE-BUTANOL-AUTO-SEED', thermo, params
            ).solve({'feed': feed})
        performance = automatic.performance

        self.assertEqual(performance['initializer'], 'vlle_cheap_fallback')
        self.assertEqual(performance['vlle_seed_requested'], 'auto')
        self.assertTrue(performance['vlle_seed_fallback'])
        self.assertIn('MESH solve failed', performance['vlle_seed_failure'])
        self.assertFalse(
            performance['vlle_auto_homogeneous_colored_fallback']
        )
        self.assertTrue(any(
            'continued with the cheap VLLE seed' in warning
            for warning in automatic.warnings
        ))
        self.assertAlmostEqual(
            automatic.outlet_streams['bottoms'].composition['butanol'],
            0.987231,
            places=5,
        )

        strict = dict(params)
        strict.update({
            'vlle_seed': 'homogeneous',
            'max_iterations': 6,
            'max_jacobian_evaluations': 6,
            'finite_difference_rel_step': 1e-6,
            'colored_jacobian_fallback': False,
        })
        with (
            patch.object(RigorousDistillation, 'solve', fail_homogeneous_seed),
            self.assertRaisesRegex(
                UnitOperationError,
                "VLLE-BUTANOL-HOMOGENEOUS-SEED_vlle_seed.*MESH solve failed",
            ),
        ):
            RigorousDistillation(
                'VLLE-BUTANOL-HOMOGENEOUS-SEED', thermo, strict
            ).solve({'feed': feed})

    def test_direct_azeotropic_seed_finds_post_azeotrope_branch_without_vle_solve(self):
        thermo = create_thermodynamics(['butanol', 'water'], 'NRTL')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'butanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )
        params = {
            'N_stages': 20,
            'feed_stage': 10,
            'reflux_ratio': 1.2,
            'D_to_F': 0.8,
            'P_condenser': 1.0,
            'P_drop_per_stage': 0.0,
            'condenser_type': 'total',
            'stage_phase_model': 'VLLE',
            'mesh_tolerance': 1e-5,
            'max_iterations': 60,
            'max_jacobian_evaluations': 60,
        }
        original_solve = RigorousDistillation.solve

        def reject_homogeneous_seed(unit, inlets):
            if unit.unit_id.endswith('_vlle_seed'):
                raise AssertionError('direct azeotropic mode solved a VLE seed')
            return original_solve(unit, inlets)

        with patch.object(
            RigorousDistillation,
            'solve',
            reject_homogeneous_seed,
        ):
            result = RigorousDistillation(
                'VLLE-BUTANOL-080-AZEOTROPIC',
                thermo,
                {**params, 'vlle_seed': 'azeotropic'},
            ).solve({'feed': feed})

        performance = result.performance
        candidate = performance['vlle_azeotropic_candidates'][0]
        self.assertEqual(performance['initializer'], 'vlle_azeotropic_linear')
        self.assertEqual(
            performance['vlle_azeotropic_candidate_source'],
            'simultaneous_binary_vlle',
        )
        self.assertEqual(performance['vlle_topology'], 'L' + '.' * 19)
        self.assertEqual(len(performance['vlle_initializer_attempts']), 1)
        self.assertTrue(performance['vlle_initializer_attempts'][0]['success'])
        self.assertLessEqual(candidate['function_evaluations'], 12)
        self.assertAlmostEqual(candidate['temperature_C'], 93.03958, places=3)
        self.assertAlmostEqual(
            candidate['composition']['butanol'], 0.22459337, places=6
        )
        self.assertAlmostEqual(
            result.outlet_streams['distillate'].composition['butanol'],
            0.25000107,
            places=6,
        )
        self.assertGreater(
            result.outlet_streams['bottoms'].composition['butanol'], 0.99999
        )
        self.assertLess(performance['mesh_residual'], 1e-5)

    def test_binary_vlle_azeotrope_solver_closes_full_fugacity_equations(self):
        cases = (
            (
                ['butanol', 'water'],
                'NRTL',
                1.0,
                ('butanol', 'water'),
                0.22459337,
                366.18958,
            ),
            (
                ['water', 'chloroform'],
                'UNIFNIST',
                1.01325,
                ('water', 'chloroform'),
                0.16270089,
                329.12646,
            ),
        )
        for components, method, pressure, pair, expected, temperature in cases:
            with self.subTest(components=components):
                thermo = create_thermodynamics(components, method)
                unit = RigorousDistillation('VLLE-AZEOTROPE-PROBE', thermo, {})
                candidates = unit._binary_vlle_azeotropes(
                    pair, components, pressure
                )
                candidate = candidates[0] if candidates else None

                self.assertIsNotNone(candidate)
                self.assertLess(candidate['fugacity_residual'], 1e-10)
                self.assertLessEqual(candidate['function_evaluations'], 12)
                self.assertAlmostEqual(
                    candidate['composition'][pair[0]], expected, places=6
                )
                self.assertAlmostEqual(candidate['T'], temperature, places=3)

    def test_binary_vlle_candidate_cache_respects_search_controls(self):
        thermo = create_thermodynamics(['butanol', 'water'], 'NRTL')
        unit = RigorousDistillation(
            'VLLE-AZEOTROPE-CACHE',
            thermo,
            {'vlle_azeotropic_max_binary_pairs': 0},
        )
        components = ['butanol', 'water']
        self.assertEqual(
            unit._vlle_azeotrope_candidates(components, 1.0), []
        )
        unit.params['vlle_azeotropic_max_binary_pairs'] = 1
        candidates = unit._vlle_azeotrope_candidates(components, 1.0)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(
            candidates[0]['composition']['butanol'],
            0.22459337,
            places=6,
        )

    def test_ternary_vlle_azeotrope_solver_is_simultaneous_and_fast(self):
        components = ['ethanol', 'water', 'benzene']
        thermo = create_thermodynamics(components, 'UNIFAC')
        unit = RigorousDistillation('VLLE-TERNARY-AZEOTROPE', thermo, {})
        candidates = unit._vlle_azeotrope_candidates(components, 1.0)
        candidate = next(item for item in candidates if item['order'] == 3)

        self.assertLess(candidate['fugacity_residual'], 1e-10)
        self.assertLess(candidate['fixed_point_residual'], 1e-10)
        self.assertLessEqual(candidate['function_evaluations'], 10)
        self.assertAlmostEqual(candidate['T'], 336.82960, places=3)
        self.assertAlmostEqual(
            candidate['composition']['ethanol'], 0.26933251, places=6
        )
        self.assertAlmostEqual(
            candidate['composition']['water'], 0.19416911, places=6
        )
        self.assertAlmostEqual(
            candidate['composition']['benzene'], 0.53649838, places=6
        )

    def test_direct_azeotropic_seed_requires_a_vlle_candidate(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'NRTL')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'ethanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )
        with self.assertRaisesRegex(
            UnitOperationError,
            'could not find a binary VLLE azeotrope',
        ):
            RigorousDistillation(
                'VLLE-NO-HETEROAZEOTROPE',
                thermo,
                {
                    'N_stages': 8,
                    'feed_stage': 4,
                    'reflux_ratio': 2.0,
                    'D_to_F': 0.4,
                    'P_condenser': 1.0,
                    'stage_phase_model': 'VLLE',
                    'vlle_seed': 'azeotropic',
                },
            ).solve({'feed': feed})

    def test_direct_azeotropic_seed_falls_back_to_log_feed_profile(self):
        root = Path(__file__).resolve().parents[1]
        simulator = Simulator.from_file(
            root / 'examples' / 'lactic_acid_dehydration_pbr.pfd'
        ).initialize()
        thermo = simulator.thermo_packages['global']
        original = simulator.solver.units['C-401']
        feed = thermo.calculate_state(
            298.48253699169915,
            0.16,
            79.79197900206256,
            {
                'H2O': 0.4393767665649053,
                'AA': 0.25149758937416866,
                'AcH': 9.632033783440319e-8,
                'PA': 0.007609656944632908,
                'MIBK': 0.3015158904858782,
            },
            phase='liquid',
            flash=False,
        )
        params = dict(original.params)
        params['vlle_seed'] = 'azeotropic'
        result = RigorousDistillation(
            'VLLE-LACTIC-AZEOTROPIC', thermo, params
        ).solve({'feed': feed})
        performance = result.performance
        attempts = performance['vlle_initializer_attempts']

        self.assertLessEqual(len(attempts), 2)
        self.assertEqual(attempts[0]['initializer'], 'vlle_azeotropic_linear')
        self.assertTrue(attempts[-1]['success'])
        self.assertEqual(performance['initializer'], attempts[-1]['initializer'])
        self.assertEqual(performance['vlle_topology'], 'L' * 13 + '.' * 7)
        self.assertLess(performance['mesh_residual'], 1e-5)
        self.assertLessEqual(performance['component_balance_error'], 2.632e-7)
        self._assert_attempt_accounting(performance)
        self.assertLessEqual(performance['jacobian_evaluations'], 41)
        self.assertLessEqual(performance['function_evaluations'], 4981)
        self.assertLessEqual(performance['solver_iterations'], 41)

    def test_direct_azeotropic_seed_recovers_full_stage_topology_cycle(self):
        thermo, feed, params = self._nrtl_rk_case(stages=12)
        params['vlle_seed'] = 'azeotropic'
        params['vlle_azeotropic_max_ternary_combinations'] = 0
        result = RigorousDistillation(
            'VLLE-NRTL-RK-AZEOTROPIC', thermo, params
        ).solve({'feed': feed})
        performance = result.performance
        attempts = performance['vlle_initializer_attempts']

        self.assertEqual(performance['vlle_topology'], 'L' * 12)
        self.assertLessEqual(len(attempts), 3)
        self.assertTrue(attempts[-1]['success'])
        self.assertLessEqual(attempts[-1]['work']['jacobian_evaluations'], 8)
        self.assertLessEqual(performance['jacobian_evaluations'], 19)
        self.assertLessEqual(performance['function_evaluations'], 1087)
        self.assertLessEqual(performance['solver_iterations'], 19)
        self.assertLess(performance['mesh_residual'], 1e-5)
        self.assertLessEqual(performance['component_balance_error'], 1.871e-7)
        self._assert_attempt_accounting(performance)

    def _assert_attempt_accounting(self, performance):
        attempts = performance['vlle_initializer_attempts']
        self.assertEqual(performance['solver_work_basis'], 'all_vlle_mesh_attempts')
        for name in attempts[-1]['work']:
            self.assertEqual(
                performance[name], sum(attempt['work'][name] for attempt in attempts)
            )
        for attempt in attempts:
            self.assertGreater(attempt['work']['function_evaluations'], 0)


class AcrylicAcidRecoveryVLLETests(unittest.TestCase):
    FEED_FLOW = 70.66427321297607
    FEED_Z_RAW = {
        'H2O': 0.37049321630854976,
        'MIBK': 0.33780317143923255,
        'AA': 0.28311082465910160,
        'PA': 0.008592684817684328,
    }
    AZEOTROPE = {
        'H2O': 0.6478905556344128,
        'MIBK': 0.3521094443655872,
    }

    @classmethod
    def setUpClass(cls):
        pfd = (
            Path(__file__).resolve().parents[1]
            / 'examples'
            / 'lactic_acid_dehydration_pbr.pfd'
        )
        cls.simulator = Simulator.from_file(str(pfd)).initialize()
        cls.thermo = cls.simulator.thermo_packages['global']

    def _base_feed(self):
        return self.thermo.calculate_state(
            298.3188792640525,
            1.01325,
            self.FEED_FLOW,
            self.FEED_Z_RAW,
            phase='liquid',
            flash=False,
        )

    def _assert_efficient_topology_path(self, performance, maximum_events):
        events = performance['vlle_topology_events']
        self.assertLessEqual(len(events), maximum_events)
        self.assertTrue(all(
            event['from'] != event['to']
            for event in events
        ))
        visited = [events[0]['from']] if events else []
        visited.extend(event['to'] for event in events)
        self.assertEqual(len(visited), len(set(visited)))

    @staticmethod
    def _params(cut):
        return {
            'N_stages': 20,
            'feed_stage': 10,
            'reflux_ratio': 1.2,
            'D_to_F': cut,
            'P_condenser': 1.01325,
            'P_drop_per_stage': 0.0,
            'condenser_type': 'total',
            'stage_phase_model': 'VLLE',
            'vlle_seed': 'cheap',
            'mesh_tolerance': 1e-5,
            'acceptable_mesh_residual': 1e-4,
            'max_iterations': 140,
            'max_jacobian_evaluations': 140,
            'vlle_colored_jacobian_fallback': False,
        }

    def test_adaptive_topology_recovers_moderate_azeotrope_cut(self):
        result = RigorousDistillation(
            'AA-RECOVERY-056',
            self.thermo,
            self._params(0.56),
        ).solve({'feed': self._base_feed()})

        performance = result.performance
        self.assertLess(performance['mesh_residual'], 1e-5)
        self.assertEqual(performance['vlle_topology_policy'], 'adaptive')
        self.assertEqual(performance['vlle_topology'], 'L' * 17 + '.' * 3)
        self.assertTrue(performance['vlle_topology_events'])
        self._assert_efficient_topology_path(performance, maximum_events=2)

    def test_direct_azeotropic_seed_uses_associated_vapor_fugacity(self):
        result = RigorousDistillation(
            'AA-RECOVERY-056-AZEOTROPIC',
            self.thermo,
            {**self._params(0.56), 'vlle_seed': 'azeotropic'},
        ).solve({'feed': self._base_feed()})
        performance = result.performance
        candidate = next(
            item
            for item in performance['vlle_azeotropic_candidates']
            if item['composition']['H2O'] > 0.6
            and item['composition']['MIBK'] > 0.3
        )

        self.assertEqual(performance['initializer'], 'vlle_azeotropic_linear')
        self.assertEqual(performance['vlle_topology'], 'L' * 17 + '.' * 3)
        self.assertLess(performance['mesh_residual'], 1e-5)
        self.assertLess(candidate['fugacity_residual'], 1e-10)
        self.assertLessEqual(candidate['function_evaluations'], 12)
        self.assertAlmostEqual(candidate['temperature_C'], 88.34898, places=3)
        self.assertAlmostEqual(
            candidate['composition']['H2O'], 0.64789031, places=6
        )

    def test_adaptive_topology_contracts_near_water_depletion(self):
        result = RigorousDistillation(
            'AA-RECOVERY-WATER-LIMIT',
            self.thermo,
            self._params(0.5718453727817714),
        ).solve({'feed': self._base_feed()})

        performance = result.performance
        self.assertLess(performance['mesh_residual'], 1e-5)
        self.assertEqual(performance['vlle_topology'], 'L' * 11 + '.' * 9)
        self.assertLess(
            result.outlet_streams['bottoms'].composition['H2O'],
            1e-5,
        )
        self.assertLessEqual(performance['vlle_topology_solves'], 12)
        self._assert_efficient_topology_path(performance, maximum_events=8)

    def test_adaptive_topology_recovers_9995_percent_mibk_with_added_water(self):
        feed = self._base_feed()
        component_flows = {
            comp: feed.F * feed.composition[comp]
            for comp in self.FEED_Z_RAW
        }
        water_add = (
            component_flows['MIBK']
            * self.AZEOTROPE['H2O']
            / self.AZEOTROPE['MIBK']
            - component_flows['H2O']
        )
        mixed_flow = feed.F + water_add
        mixed_z = {
            'H2O': (component_flows['H2O'] + water_add) / mixed_flow,
            'MIBK': component_flows['MIBK'] / mixed_flow,
            'AA': component_flows['AA'] / mixed_flow,
            'PA': component_flows['PA'] / mixed_flow,
        }
        watered_feed = self.thermo.calculate_state(
            feed.T,
            feed.P,
            mixed_flow,
            mixed_z,
            phase='liquid',
            flash=False,
        )
        target_distillate = (
            0.9995
            * component_flows['MIBK']
            / self.AZEOTROPE['MIBK']
        )
        params = self._params(target_distillate / mixed_flow)
        baseline_params = dict(params)
        baseline_params['vlle_projection_enabled'] = False
        baseline = RigorousDistillation(
            'AA-RECOVERY-WATERED-9995-BASELINE',
            self.thermo,
            baseline_params,
        ).solve({'feed': watered_feed})
        result = RigorousDistillation(
            'AA-RECOVERY-WATERED-9995',
            self.thermo,
            params,
        ).solve({'feed': watered_feed})

        distillate = result.outlet_streams['distillate']
        mibk_recovery = (
            distillate.F * distillate.composition['MIBK']
            / component_flows['MIBK']
        )
        performance = result.performance
        self.assertLess(performance['mesh_residual'], 1e-5)
        self.assertGreater(mibk_recovery, 0.9994)
        self.assertEqual(performance['vlle_topology'], 'L' * 13 + '.' * 7)
        self._assert_efficient_topology_path(performance, maximum_events=7)
        self.assertLessEqual(
            performance['jacobian_evaluations'],
            baseline.performance['jacobian_evaluations'],
        )
        self.assertLessEqual(baseline.performance['jacobian_evaluations'], 26)
        self.assertLessEqual(performance['jacobian_evaluations'], 22)
        self.assertLessEqual(performance['vlle_projection_checks'], 5)
        projected = [
            event
            for event in performance['vlle_topology_events']
            if event['reason'] == 'projected_newton_boundary'
        ]
        self.assertLessEqual(len(projected), 2)
        self.assertLessEqual(
            max((event['jacobian_evaluations'] for event in projected), default=0), 4
        )


if __name__ == '__main__':
    unittest.main()
