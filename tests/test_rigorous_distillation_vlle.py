import unittest

from thermodynamics import create_thermodynamics
from unit_operations_base import UnitOperationError
from unit_operations_distillation import RigorousDistillation


class RigorousDistillationVLLETests(unittest.TestCase):
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
        with self.assertRaisesRegex(UnitOperationError, 'decanter condenser'):
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


if __name__ == '__main__':
    unittest.main()
