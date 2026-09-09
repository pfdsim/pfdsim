import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dof_analyzer import SpecificationStatus, analyze_dof
from pfd_parser import parse_pfd
from simulator import Simulator
from thermodynamics import create_thermodynamics
from unit_operations import Pipe, UNIT_CLASSES, UnitOperationError
from unit_operations_transport import PIPE_MATERIAL_ROUGHNESS_M


class PipeUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.steam = create_thermodynamics(['H2O'], 'STEAM')
        cls.methane = create_thermodynamics(['CH4'], 'PR')
        cls.glycerol = create_thermodynamics(['C3H8O3'], 'IDEAL')

    @staticmethod
    def _mass_flow_state(thermo, component, T, P, mass_flow_kg_h, phase):
        composition = {component: 1.0}
        molar_flow = mass_flow_kg_h / thermo.mixture_MW(composition)
        return thermo.calculate_state(
            T,
            P,
            molar_flow,
            composition,
            phase=phase,
            flash=False,
        )

    def test_pipe_is_publicly_registered(self):
        self.assertIs(UNIT_CLASSES['Pipe'], Pipe)

    def test_pipe_diameter_profile_shapes(self):
        for profile in ('linear', 'smooth'):
            diameter_at = Pipe._diameter_function(0.02, 0.04, 10.0, profile)
            self.assertAlmostEqual(diameter_at(2.5), 0.025, places=12)
            self.assertAlmostEqual(diameter_at(7.5), 0.035, places=12)

        smoothstep_at = Pipe._diameter_function(0.02, 0.04, 10.0, 'smoothstep')
        self.assertAlmostEqual(smoothstep_at(2.5), 0.023125, places=12)
        self.assertAlmostEqual(smoothstep_at(7.5), 0.036875, places=12)

    def test_commercial_steel_water_probe(self):
        inlet = self._mass_flow_state(
            self.steam, 'H2O', 353.15, 10.0, 5000.0, 'liquid'
        )
        result = Pipe(
            'WATER-PIPE',
            self.steam,
            {
                'length': 100.0,
                'diameter': 0.04,
                'material': 'commercial_steel',
                'max_step': 10.0,
                'profile_points': 5,
            },
        ).solve({'in': inlet})
        performance = result.performance

        self.assertAlmostEqual(performance['pressure_drop_bar'], 0.3510947, delta=2e-6)
        self.assertAlmostEqual(performance['P_out_bar'], 9.6489053, delta=2e-6)
        self.assertAlmostEqual(performance['reynolds_in'], 124778.0, delta=5.0)
        self.assertAlmostEqual(performance['reynolds_out'], 124791.0, delta=5.0)
        self.assertAlmostEqual(performance['roughness_m'], 4.5e-5)
        self.assertLess(abs(performance['pressure_drop_closure_bar']), 1e-9)
        self.assertLess(performance['maximum_energy_residual_J_per_kg'], 1e-3)

    def test_commercial_steel_methane_probe(self):
        inlet = self._mass_flow_state(
            self.methane, 'CH4', 333.15, 20.0, 5000.0, 'vapor'
        )
        result = Pipe(
            'METHANE-PIPE',
            self.methane,
            {
                'length': 100.0,
                'diameter': 0.10,
                'material': 'steel',
                'max_step': 10.0,
                'profile_points': 5,
            },
        ).solve({'in': inlet})
        performance = result.performance

        self.assertAlmostEqual(
            performance['pressure_drop_bar'], 0.22158274728790772, delta=2e-6
        )
        self.assertAlmostEqual(
            performance['P_out_bar'], 19.778417252712092, delta=2e-6
        )
        self.assertAlmostEqual(performance['reynolds_in'], 1408811.0, delta=5.0)
        self.assertAlmostEqual(performance['reynolds_out'], 1409492.0, delta=5.0)
        self.assertGreater(performance['acceleration_pressure_drop_bar'], 0.0)
        self.assertLess(abs(performance['pressure_drop_closure_bar']), 1e-9)

    def test_commercial_steel_glycerol_45c_probe(self):
        inlet = self._mass_flow_state(
            self.glycerol, 'C3H8O3', 318.15, 10.0, 5000.0, 'liquid'
        )
        result = Pipe(
            'GLYCEROL-PIPE',
            self.glycerol,
            {
                'length': 100.0,
                'diameter': 0.04,
                'material': 'commercial_steel',
                'max_step': 10.0,
                'profile_points': 5,
            },
        ).solve({'in': inlet})
        performance = result.performance
        inlet_profile = performance['profile'][0]

        self.assertAlmostEqual(performance['pressure_drop_bar'], 3.70953236, delta=0.006)
        self.assertAlmostEqual(performance['P_out_bar'], 6.29046764, delta=0.006)
        self.assertAlmostEqual(inlet_profile['viscosity_Pa_s'], 0.209111039, delta=4e-4)
        self.assertAlmostEqual(performance['reynolds_in'], 211.4174, delta=0.4)
        self.assertAlmostEqual(performance['friction_factor_in'], 0.30271874, delta=5e-4)
        self.assertLess(performance['reynolds_in'], 2300.0)
        self.assertAlmostEqual(
            performance['friction_factor_in'],
            64.0 / performance['reynolds_in'],
            places=12,
        )
        self.assertLess(abs(performance['pressure_drop_closure_bar']), 1e-9)

    def test_pipe_reports_where_liquid_exhausts_available_pressure(self):
        inlet = self._mass_flow_state(
            self.glycerol, 'C3H8O3', 303.15, 10.0, 5000.0, 'liquid'
        )
        pipe = Pipe(
            'GLYCEROL-30C',
            self.glycerol,
            {
                'length': 100.0,
                'diameter': 0.04,
                'material': 'commercial_steel',
                'max_step': 10.0,
                'profile_points': 5,
            },
        )

        with self.assertRaises(UnitOperationError) as raised:
            pipe.solve({'in': inlet})

        message = str(raised.exception)
        self.assertIn('exhausted its available inlet pressure', message)
        distance = float(message.split(' after ', 1)[1].split(' m of ', 1)[0])
        self.assertAlmostEqual(distance, 91.99, delta=0.08)
        self.assertIn('m of 100 m', message)
        self.assertIn('liquid flow cannot reach the outlet', message)

    def test_vertical_pipe_reports_hydrostatic_component(self):
        inlet = self._mass_flow_state(
            self.steam, 'H2O', 353.15, 10.0, 1000.0, 'liquid'
        )
        result = Pipe(
            'RISER',
            self.steam,
            {
                'length': 5.0,
                'diameter': 0.08,
                'orientation': 'vertical_up',
                'material': 'commercial_steel',
                'max_step': 5.0,
                'profile_points': 2,
            },
        ).solve({'in': inlet})
        performance = result.performance

        self.assertAlmostEqual(performance['elevation_change_m'], 5.0)
        self.assertAlmostEqual(performance['inclination_angle_degrees'], 90.0)
        self.assertGreater(performance['gravity_pressure_drop_bar'], 0.47)
        self.assertLess(performance['gravity_pressure_drop_bar'], 0.49)
        self.assertGreater(performance['friction_pressure_drop_bar'], 0.0)

    def test_explicit_roughness_units_override_material(self):
        warnings = []
        pipe = Pipe(
            'ROUGH',
            self.steam,
            {
                'roughness': 0.1,
                '__unit__roughness': 'mm',
                'material': 'pvc',
            },
        )

        roughness, source, material = pipe._roughness(warnings)

        self.assertAlmostEqual(roughness, 1e-4)
        self.assertEqual(source, 'explicit')
        self.assertEqual(material, 'pvc')
        self.assertTrue(any('overrides material' in warning for warning in warnings))
        self.assertEqual(PIPE_MATERIAL_ROUGHNESS_M['steel'], 4.5e-5)
        self.assertEqual(PIPE_MATERIAL_ROUGHNESS_M['smooth'], 0.0)

    def test_two_phase_inlet_returns_zero_drop_warning(self):
        inlet = self.steam.calculate_state_PQ(
            1.0, 0.4, 10.0, {'H2O': 1.0}
        )
        result = Pipe(
            'WET',
            self.steam,
            {'length': 100.0, 'diameter': 0.1, 'phase_model': 'single'},
        ).solve({'in': inlet})

        self.assertEqual(result.outlet_streams['out'].P, inlet.P)
        self.assertEqual(result.performance['pressure_drop_bar'], 0.0)
        self.assertTrue(result.performance['two_phase_not_implemented'])
        self.assertTrue(any(
            'two-phase pressure drop is not implemented' in warning
            for warning in result.warnings
        ))

    def test_two_phase_inlet_uses_beggs_brill_by_default(self):
        inlet = self.steam.calculate_state_PQ(
            1.0, 0.4, 10.0, {'H2O': 1.0}
        )
        result = Pipe(
            'WET',
            self.steam,
            {'length': 100.0, 'diameter': 0.1, 'profile_points': 3},
        ).solve({'in': inlet})
        performance = result.performance
        inlet_profile = performance['profile'][0]

        self.assertEqual(performance['model_status'], 'two_phase_beggs_brill_integrated')
        self.assertTrue(performance['two_phase_encountered'])
        self.assertFalse(performance['two_phase_not_implemented'])
        self.assertAlmostEqual(performance['energy_iteration_tolerance_kJ_per_kmol'], 1e-6)
        self.assertEqual(performance['maximum_local_energy_iterations'], 1)
        self.assertAlmostEqual(performance['pressure_drop_bar'], 0.004453066, delta=2e-8)
        self.assertAlmostEqual(performance['P_out_bar'], 0.995546934, delta=2e-8)
        self.assertAlmostEqual(performance['friction_pressure_drop_bar'], 0.004451835, delta=2e-8)
        self.assertAlmostEqual(performance['acceleration_pressure_drop_bar'], 1.230572e-6, delta=2e-9)
        self.assertEqual(inlet_profile['phase'], 'two_phase')
        self.assertEqual(inlet_profile['flow_regime'], 'segregated')
        self.assertAlmostEqual(inlet_profile['surface_tension_N_m'], 0.058997251, delta=2e-9)
        self.assertAlmostEqual(inlet_profile['liquid_holdup'], 0.025670866, delta=2e-9)
        self.assertAlmostEqual(inlet_profile['mass_quality'], 0.4, delta=1e-12)
        self.assertAlmostEqual(inlet_profile['darcy_friction_factor'], 0.032261916, delta=2e-9)
        self.assertEqual(inlet_profile['energy_iterations'], 1)

    def test_inclined_two_phase_pipe_reports_holdup_gravity_component(self):
        inlet = self.steam.calculate_state_PQ(
            1.0, 0.4, 10.0, {'H2O': 1.0}
        )
        result = Pipe(
            'WET-RISER',
            self.steam,
            {
                'length': 5.0,
                'diameter': 0.1,
                'orientation': 'vertical_up',
                'profile_points': 3,
            },
        ).solve({'in': inlet})
        performance = result.performance
        inlet_profile = performance['profile'][0]

        self.assertEqual(performance['model_status'], 'two_phase_beggs_brill_integrated')
        self.assertGreater(performance['gravity_pressure_drop_bar'], 0.03)
        self.assertLess(performance['gravity_pressure_drop_bar'], 0.04)
        self.assertGreater(
            performance['gravity_pressure_drop_bar'],
            performance['friction_pressure_drop_bar'],
        )
        self.assertGreater(inlet_profile['liquid_holdup'], inlet_profile['no_slip_liquid_fraction'])
        self.assertEqual(inlet_profile['flow_regime'], 'segregated')

    def test_two_phase_energy_iteration_controls_can_be_overridden(self):
        inlet = self.steam.calculate_state_PQ(
            1.0, 0.4, 10.0, {'H2O': 1.0}
        )
        result = Pipe(
            'WET-ENERGY-ITERATIONS',
            self.steam,
            {
                'length': 100.0,
                'diameter': 0.1,
                'profile_points': 3,
                'energy_max_iterations': 3,
                'energy_tolerance': 1e-12,
            },
        ).solve({'in': inlet})
        performance = result.performance

        self.assertTrue(performance['local_energy_iterations_overridden'])
        self.assertEqual(performance['local_energy_max_iterations'], 3)
        self.assertAlmostEqual(
            performance['energy_iteration_tolerance_kJ_per_kmol'],
            1e-12,
        )
        self.assertEqual(performance['maximum_local_energy_iterations'], 3)
        self.assertLess(
            performance['maximum_local_energy_iteration_residual_kJ_per_kmol'],
            1e-10,
        )

    def test_two_phase_model_can_be_forced(self):
        inlet = self.steam.calculate_state_PQ(
            1.0, 0.4, 10.0, {'H2O': 1.0}
        )
        forced = Pipe(
            'WET-FORCED',
            self.steam,
            {
                'length': 100.0,
                'diameter': 0.1,
                'phase_model': 'two_phase',
            },
        ).solve({'in': inlet})

        self.assertEqual(
            forced.performance['model_status'],
            'two_phase_beggs_brill_integrated',
        )

        dry_inlet = self._mass_flow_state(
            self.steam, 'H2O', 353.15, 10.0, 5000.0, 'liquid'
        )
        with self.assertRaisesRegex(UnitOperationError, 'requires a two-phase inlet'):
            Pipe(
                'DRY-FORCED',
                self.steam,
                {
                    'length': 100.0,
                    'diameter': 0.1,
                    'phase_model': 'two_phase',
                },
            ).solve({'in': dry_inlet})

    def test_conflicting_sizing_specs_and_unknown_material_are_rejected(self):
        inlet = self._mass_flow_state(
            self.steam, 'H2O', 353.15, 10.0, 1000.0, 'liquid'
        )
        with self.assertRaisesRegex(UnitOperationError, 'diameter or velocity'):
            Pipe(
                'BAD-SIZE',
                self.steam,
                {'length': 10.0, 'diameter': 0.1, 'velocity': 1.0},
            ).solve({'in': inlet})
        with self.assertRaisesRegex(UnitOperationError, 'unknown material'):
            Pipe(
                'BAD-MATERIAL',
                self.steam,
                {'length': 10.0, 'diameter': 0.1, 'material': 'unobtainium'},
            ).solve({'in': inlet})
        with self.assertRaisesRegex(UnitOperationError, 'phase_model'):
            Pipe(
                'BAD-PHASE-MODEL',
                self.steam,
                {'length': 10.0, 'diameter': 0.1, 'phase_model': 'wizardry'},
            ).solve({'in': inlet})
        with self.assertRaisesRegex(UnitOperationError, 'surface_tension_method'):
            Pipe(
                'BAD-SIGMA',
                self.steam,
                {
                    'length': 10.0,
                    'diameter': 0.1,
                    'surface_tension_method': 'sparkles',
                },
            ).solve({'in': inlet})
        with self.assertRaisesRegex(UnitOperationError, 'energy_tolerance'):
            Pipe(
                'BAD-ENERGY-TOLERANCE',
                self.steam,
                {'length': 10.0, 'diameter': 0.1, 'energy_tolerance': 0.0},
            ).solve({'in': inlet})
        with self.assertRaisesRegex(UnitOperationError, 'energy_max_iterations'):
            Pipe(
                'BAD-ENERGY-ITERATIONS',
                self.steam,
                {'length': 10.0, 'diameter': 0.1, 'energy_max_iterations': 0},
            ).solve({'in': inlet})

    def test_pipe_dof_requires_one_sizing_method(self):
        template = (
            'PROCESS: Pipe DOF\n'
            'COMPONENTS:\n'
            '    H2O | Water | MW=18.015\n'
            'STREAM Feed : FEED -> P.in\n'
            '    T = 25 [C]\n'
            '    P = 5 [bar]\n'
            '    F = 10 [kmol/h]\n'
            '    x = H2O:1\n'
            'STREAM Product : P.out -> PRODUCT\n'
            'UNIT P\n'
            '    TYPE: Pipe\n'
            '    PORTS:\n'
            '        in : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '{params}'
        )
        missing = analyze_dof(parse_pfd(template.format(params='        length = 10 [m]\n')))
        conflicting = analyze_dof(parse_pfd(template.format(
            params=(
                '        length = 10 [m]\n'
                '        diameter = 0.1 [m]\n'
                '        velocity = 1 [m/s]\n'
            )
        )))

        self.assertEqual(missing.unit_results[0].status, SpecificationStatus.UNDER_SPECIFIED)
        self.assertEqual(conflicting.unit_results[0].status, SpecificationStatus.OVER_SPECIFIED)

    def test_unifac_mixture_example_auto_sizes_and_tapers(self):
        result = Simulator.from_file(
            os.path.join(ROOT, 'examples', 'ethanol_water_inclined_pipe_unifac.pfd')
        ).run()
        self.assertTrue(result.converged, result.errors)
        performance = result.units['PIPE-100'].performance
        profile = performance['profile']

        self.assertEqual(performance['model_status'], 'single_phase_integrated')
        self.assertAlmostEqual(performance['velocity_in_m_s'], 1.5, places=10)
        self.assertAlmostEqual(performance['diameter_in_m'], 0.0284444486, places=8)
        self.assertAlmostEqual(performance['diameter_out_m'], 0.04, places=12)
        self.assertAlmostEqual(performance['pressure_drop_bar'], 1.3188131157, delta=2e-8)
        self.assertAlmostEqual(performance['P_out_bar'], 3.6811868843, delta=2e-8)
        self.assertAlmostEqual(performance['friction_pressure_drop_bar'], 0.6700066217, delta=2e-8)
        self.assertAlmostEqual(performance['gravity_pressure_drop_bar'], 0.6559407920, delta=2e-8)
        self.assertAlmostEqual(performance['acceleration_pressure_drop_bar'], -0.0071342980, delta=2e-8)
        self.assertAlmostEqual(performance['reynolds_in'], 16782.2876, delta=0.01)
        self.assertAlmostEqual(performance['reynolds_out'], 11930.9616, delta=0.01)
        self.assertAlmostEqual(profile[0]['viscosity_Pa_s'], 0.002166136426, delta=2e-12)
        self.assertAlmostEqual(profile[-1]['viscosity_Pa_s'], 0.002166701302, delta=2e-12)
        for point in profile:
            axial_fraction = point['position_m'] / performance['length_m']
            expected_diameter = performance['diameter_in_m'] + axial_fraction * (
                performance['diameter_out_m'] - performance['diameter_in_m']
            )
            self.assertAlmostEqual(point['diameter_m'], expected_diameter, places=12)
        self.assertGreater(performance['friction_pressure_drop_bar'], 0.0)
        self.assertGreater(performance['gravity_pressure_drop_bar'], 0.0)
        self.assertLess(performance['acceleration_pressure_drop_bar'], 0.0)
        self.assertLess(result.energy_balance_error, 1e-4)


if __name__ == '__main__':
    unittest.main()
