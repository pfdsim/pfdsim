import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from simulator import Simulator
from dof_analyzer import SpecificationStatus, analyze_dof
from pfd_parser import PFDParser


class PFDIntegrationTests(unittest.TestCase):
    def test_vlle_azeotrope_seed_parameters_parse_into_runtime(self):
        pfd = (
            'PROCESS: Direct VLLE Azeotrope Parameter Parsing\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: NRTL\n'
            'ONLINE_LOOKUP: false\n'
            'COMPONENTS:\n'
            '    butanol | 1-Butanol | MW=74.12\n'
            '    water | Water | MW=18.02\n'
            'STREAM Feed : FEED -> COL-1.feed\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 100 [kmol/h]\n'
            '    x = butanol:0.4, water:0.6\n'
            'STREAM Distillate : COL-1.distillate -> PRODUCT\n'
            'STREAM Bottoms : COL-1.bottoms -> PRODUCT\n'
            'UNIT COL-1 : RigorousDistillation\n'
            '    N_stages = 20\n'
            '    feed_stage = 10\n'
            '    reflux_ratio = 1.2\n'
            '    D_to_F = 0.8\n'
            '    P_condenser = 1 [bar]\n'
            '    stage_phase_model = VLLE\n'
            '    vlle_seed = azeotropic\n'
            '    vlle_azeotrope_composition = {butanol:0.22459337, water:0.77540663}\n'
            '    vlle_azeotrope_temperature = 93.03958 [C]\n'
        )
        simulator = Simulator.from_string(pfd).initialize()
        column = simulator.solver.units['COL-1']

        self.assertEqual(
            column.params['vlle_azeotrope_composition'],
            {'butanol': 0.22459337, 'water': 0.77540663},
        )
        self.assertAlmostEqual(
            column.params['vlle_azeotrope_temperature'],
            366.18958,
            places=5,
        )
        result = simulator.run()
        self.assertTrue(result.converged, result.errors)
        performance = result.units['COL-1'].performance
        self.assertEqual(
            performance['vlle_azeotropic_candidate_source'], 'provided'
        )
        self.assertEqual(performance['vlle_topology'], 'L' + '.' * 19)

    def test_flash_dof_accepts_inherited_upstream_pressure(self):
        pfd = PFDParser().parse(
            'PROCESS: Inherited Pressure Flash\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            'COMPONENTS:\n'
            '    H2O | Water\n'
            'STREAM Feed : FEED -> F-1.in\n'
            '    T = 80 [C]\n'
            '    P = 2 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = H2O:1\n'
            'STREAM Vapor : F-1.vapor_out\n'
            'STREAM Liquid : F-1.liquid_out\n'
            'UNIT F-1 : Flash\n'
            '    Q = 0 [kW]\n'
        )

        unit = next(item for item in analyze_dof(pfd).unit_results if item.entity_id == 'F-1')
        self.assertEqual(unit.status, SpecificationStatus.OK)

    def test_initialize_is_idempotent_and_run_reuses_ready_runtime(self):
        pfd = (
            'PROCESS: Initialization Lifecycle\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            'ONLINE_LOOKUP: false\n'
            '\n'
            'COMPONENTS:\n'
            '    H2O | Water\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = H2O:1.0\n'
        )
        simulator = Simulator.from_string(pfd)

        self.assertFalse(simulator._initialized)
        self.assertIsNone(simulator.thermo)
        self.assertIsNone(simulator.solver)
        self.assertIsNone(simulator.result)

        returned = simulator.initialize()
        thermo = simulator.thermo
        solver = simulator.solver

        self.assertIs(returned, simulator)
        self.assertTrue(simulator._initialized)
        self.assertTrue(thermo._runtime_initialized)
        self.assertEqual(solver.streams, {})
        self.assertEqual(solver.unit_results, {})
        self.assertIsNone(simulator.result)

        self.assertIs(simulator.initialize(), simulator)
        self.assertIs(simulator.thermo, thermo)
        self.assertIs(simulator.solver, solver)

        result = simulator.run()
        self.assertTrue(result.converged, result.errors)
        self.assertIs(simulator.thermo, thermo)
        self.assertIs(simulator.solver, solver)

        simulator.initialize(thermo_method='PR')
        pr_thermo = simulator.thermo
        pr_solver = simulator.solver
        self.assertEqual(simulator.thermo_method, 'PR')
        self.assertIsNot(pr_thermo, thermo)
        self.assertIsNot(pr_solver, solver)
        self.assertTrue(
            pr_thermo.cubic._compiled_backend.compilation_complete
        )

        result = simulator.run()
        self.assertTrue(result.converged, result.errors)
        self.assertEqual(simulator.thermo_method, 'PR')
        self.assertIs(simulator.thermo, pr_thermo)
        self.assertIs(simulator.solver, pr_solver)

    def test_rigorous_absorber_pfd_registration_and_isothermal_mode(self):
        pfd = (
            'PROCESS: Rigorous Absorber Registration\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            '\n'
            'COMPONENTS:\n'
            '    H2O | Water | MW=18.015\n'
            '    C2H5OH | Ethanol | MW=46.069\n'
            '    N2 | Nitrogen | MW=28.014\n'
            '    O2 | Oxygen | MW=31.999\n'
            '\n'
            'STREAM GasFeed : FEED -> ABS-1.gas\n'
            '    T = 50 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 10 [kmol/h]\n'
            '    x = N2:0.79, O2:0.20, C2H5OH:0.01\n'
            '\n'
            'STREAM Solvent : FEED -> ABS-1.liquid\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 20 [kmol/h]\n'
            '    x = H2O:1.0\n'
            '\n'
            'STREAM CleanGas : ABS-1.gas_out -> PRODUCT\n'
            'STREAM RichLiquid : ABS-1.liquid_out -> PRODUCT\n'
            '\n'
            'UNIT ABS-1\n'
            '    TYPE: RigorousAbsorber\n'
            '    PORTS:\n'
            '        gas : inlet\n'
            '        liquid : inlet\n'
            '        gas_out : vapor_outlet\n'
            '        liquid_out : liquid_outlet\n'
            '    PARAMS:\n'
            '        N_stages = 1\n'
            '        mode = isothermal\n'
            '        T = 25 [C]\n'
            '        P_drop_per_stage = 0\n'
            '        mesh_tolerance = 1e-5\n'
        )

        result = Simulator.from_string(pfd).run()

        performance = result.units['ABS-1'].performance
        self.assertEqual(performance['mode'], 'isothermal')
        self.assertAlmostEqual(performance['stage_temperatures_C'][0], 25.0, places=4)
        self.assertIn('CleanGas', result.streams)
        self.assertIn('RichLiquid', result.streams)
        self.assertAlmostEqual(
            result.units['ABS-1'].heat_duty / 3600.0,
            performance['duty_kW'],
            places=10,
        )

    def test_initialized_recycle_simulation_is_repeatable(self):
        simulator = Simulator.from_file(
            os.path.join(ROOT, 'examples', 'methanol_synthesis_psrk.pfd')
        )
        simulator.initialize()

        first = simulator.run()
        first_snapshot = {
            name: (
                state.T,
                state.P,
                state.F,
                state.vapor_fraction,
                dict(state.composition),
            )
            for name, state in first.streams.items()
        }
        second = simulator.run()
        second_snapshot = {
            name: (
                state.T,
                state.P,
                state.F,
                state.vapor_fraction,
                dict(state.composition),
            )
            for name, state in second.streams.items()
        }

        self.assertTrue(first.converged, first.errors)
        self.assertTrue(second.converged, second.errors)
        self.assertEqual(first.iterations, second.iterations)
        self.assertEqual(first_snapshot, second_snapshot)

    def test_pressure_units_are_converted_for_feed_streams_and_unit_params(self):
        pfd = (
            'PROCESS: Benzene Toluene Pressure Unit Test\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            '\n'
            'COMPONENTS:\n'
            '    C6H6 | Benzene | MW=78.11\n'
            '    C7H8 | Toluene | MW=92.14\n'
            '\n'
            'STREAM Feed : FEED -> FLASH-1.in\n'
            '    T = 25 [C]\n'
            '    P = 1 [atm]\n'
            '    F = 10 [kmol/h]\n'
            '    x = C6H6:0.40, C7H8:0.60\n'
            '\n'
            'STREAM Vapor : FLASH-1.vap -> PRODUCT\n'
            'STREAM Liquid : FLASH-1.liq -> PRODUCT\n'
            '\n'
            'UNIT FLASH-1\n'
            '    TYPE: Flash\n'
            '    PORTS:\n'
            '        in  : inlet\n'
            '        vap : vapor_outlet\n'
            '        liq : liquid_outlet\n'
            '    PARAMS:\n'
            '        T = 96.326003 [C]\n'
            '        P = 1 [atm]\n'
        )

        result = Simulator.from_string(pfd).run()

        self.assertAlmostEqual(result.streams['Feed'].P, 1.01325, places=6)
        self.assertAlmostEqual(result.units['FLASH-1'].performance['P_bar'], 1.01325, places=6)
        toluene_recovery = (
            result.streams['Liquid'].F * result.streams['Liquid'].composition['C7H8']
        ) / (10.0 * 0.60)
        self.assertAlmostEqual(toluene_recovery, 0.8808235335472205, places=5)

    def test_pfd_ambiguous_formula_component_name_takes_precedence_for_unifac(self):
        pfd = (
            'PROCESS: Ambiguous Formula Name Priority\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: UNIFAC\n'
            '\n'
            'COMPONENTS:\n'
            '    C6H14 | 3-methylpentane | MW=86.18\n'
            '\n'
            'STREAM Feed : FEED -> FLASH-1.in\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 10 [kmol/h]\n'
            '    x = C6H14:1.0\n'
            '\n'
            'STREAM Vapor : FLASH-1.vap -> PRODUCT\n'
            'STREAM Liquid : FLASH-1.liq -> PRODUCT\n'
            '\n'
            'UNIT FLASH-1\n'
            '    TYPE: Flash\n'
            '    PORTS:\n'
            '        in  : inlet\n'
            '        vap : vapor_outlet\n'
            '        liq : liquid_outlet\n'
            '    PARAMS:\n'
            '        T = 25 [C]\n'
            '        P = 1 [bar]\n'
        )

        sim = Simulator.from_string(pfd)
        sim.run()

        self.assertEqual(sim.thermo.props['C6H14'].name, '3-methylpentane')
        self.assertEqual(
            sim.thermo.component_groups['C6H14'],
            {'CH3': 3, 'CH2': 2, 'CH': 1},
        )

    def test_flash_pfd_accepts_vapor_fraction_specification(self):
        pfd = (
            'PROCESS: Propane Butane Vapor Fraction Flash\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PR\n'
            '\n'
            'COMPONENTS:\n'
            '    C3H8 | Propane | MW=44.10\n'
            '    C4H10 | Butane | MW=58.12\n'
            '\n'
            'STREAM Feed : FEED -> FLASH-1.in\n'
            '    T = 50 [C]\n'
            '    P = 10 [bar]\n'
            '    F = 100 [kmol/h]\n'
            '    x = C3H8:0.50, C4H10:0.50\n'
            '\n'
            'STREAM Vapor : FLASH-1.vap -> PRODUCT\n'
            'STREAM Liquid : FLASH-1.liq -> PRODUCT\n'
            '\n'
            'UNIT FLASH-1\n'
            '    TYPE: Flash\n'
            '    PORTS:\n'
            '        in  : inlet\n'
            '        vap : vapor_outlet\n'
            '        liq : liquid_outlet\n'
            '    PARAMS:\n'
            '        P = 10 [bar]\n'
            '        vapor_fraction = 0.25\n'
        )

        result = Simulator.from_string(pfd).run()

        self.assertTrue(result.converged)
        self.assertAlmostEqual(
            result.units['FLASH-1'].performance['vapor_fraction'], 0.25, places=8
        )
        self.assertAlmostEqual(result.streams['Vapor'].F, 25.0, places=8)
        self.assertAlmostEqual(result.streams['Liquid'].F, 75.0, places=8)

    def test_flash3_pfd_routes_three_phase_outlets(self):
        pfd = (
            'PROCESS: Ternary VLLE Flash\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: NRTL\n'
            '\n'
            'COMPONENTS:\n'
            '    water | Water\n'
            '    methanol | Methanol\n'
            '    benzene | Benzene\n'
            '\n'
            'STREAM Feed : FEED -> FLASH-3.in\n'
            '    T = 59.85 [C]\n'
            '    P = 1.01325 [bar]\n'
            '    F = 100 [kmol/h]\n'
            '    x = water:0.20, methanol:0.30, benzene:0.50\n'
            '\n'
            'STREAM Vapor : FLASH-3.vapor_out -> PRODUCT\n'
            'STREAM Liquid1 : FLASH-3.liquid1_out -> PRODUCT\n'
            'STREAM Liquid2 : FLASH-3.liquid2_out -> PRODUCT\n'
            '\n'
            'UNIT FLASH-3\n'
            '    TYPE: Flash3\n'
            '    PORTS:\n'
            '        in : inlet\n'
            '        vapor_out : vapor_outlet\n'
            '        liquid1_out : liquid1_outlet\n'
            '        liquid2_out : liquid2_outlet\n'
            '    PARAMS:\n'
            '        T = 59.85 [C]\n'
            '        P = 1.01325 [bar]\n'
            '        max_iter = 200\n'
        )

        result = Simulator.from_string(pfd).run()

        self.assertTrue(result.converged)
        self.assertEqual(result.units['FLASH-3'].performance['phase_count'], 3)
        self.assertGreater(result.streams['Vapor'].F, 0.0)
        self.assertGreater(result.streams['Liquid1'].F, 0.0)
        self.assertGreater(result.streams['Liquid2'].F, 0.0)
        self.assertAlmostEqual(
            result.streams['Vapor'].F + result.streams['Liquid1'].F + result.streams['Liquid2'].F,
            100.0,
            places=6,
        )

    def test_feed_stream_accepts_vapor_fraction_with_direct_pq_thermo(self):
        pfd = (
            'PROCESS: Steam Feed Vapor Fraction\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: STEAM\n'
            '\n'
            'COMPONENTS:\n'
            '    H2O | Water | MW=18.015\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    P = 0.1 [bar]\n'
            '    F = 10 [kmol/h]\n'
            '    vapor_fraction = 0\n'
            '    x = H2O:1.0\n'
        )

        result = Simulator.from_string(pfd).run()
        feed = result.streams['Feed']

        self.assertTrue(result.converged)
        self.assertAlmostEqual(feed.P, 0.1, places=8)
        self.assertAlmostEqual(feed.T - 273.15, 45.807548, places=5)
        self.assertEqual(feed.vapor_fraction, 0.0)
        self.assertAlmostEqual(feed.H, -284264.75457, places=4)

    def test_decanter_pfr_reports_no_lle_diagnostics(self):
        pfd = (
            'PROCESS: Decanter No LLE Report\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: NRTL\n'
            '\n'
            'COMPONENTS:\n'
            '    butanol | 1-Butanol | MW=74.12\n'
            '    water | Water | MW=18.02\n'
            '\n'
            'STREAM Feed : FEED -> DECANT-1.in\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 100 [kmol/h]\n'
            '    x = butanol:0.5, water:0.5\n'
            '\n'
            'STREAM Light : DECANT-1.light -> PRODUCT\n'
            'STREAM Heavy : DECANT-1.heavy -> PRODUCT\n'
            '\n'
            'UNIT DECANT-1\n'
            '    TYPE: Decanter\n'
            '    PORTS:\n'
            '        in    : inlet\n'
            '        light : outlet\n'
            '        heavy : outlet\n'
            '    PARAMS:\n'
            '        P = 1 [bar]\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        report = sim._generate_pfr()

        self.assertTrue(result.converged)
        self.assertFalse(result.units['DECANT-1'].performance['two_phases'])
        self.assertIn('no_lle = True', report)
        self.assertIn('light_flow_kmol_hr', report)
        self.assertIn('heavy_flow_kmol_hr', report)
        self.assertIn('outlet_enthalpy_residual_kW', report)
        self.assertIn('No liquid-liquid split', report)

    def test_flash_pfd_accepts_heat_duty_specification(self):
        pfd = (
            'PROCESS: Propane Butane Duty Flash\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PR\n'
            '\n'
            'COMPONENTS:\n'
            '    C3H8 | Propane | MW=44.10\n'
            '    C4H10 | Butane | MW=58.12\n'
            '\n'
            'STREAM Feed : FEED -> FLASH-1.in\n'
            '    T = 50 [C]\n'
            '    P = 10 [bar]\n'
            '    F = 100 [kmol/h]\n'
            '    x = C3H8:0.50, C4H10:0.50\n'
            '\n'
            'STREAM Vapor : FLASH-1.vap -> PRODUCT\n'
            'STREAM Liquid : FLASH-1.liq -> PRODUCT\n'
            '\n'
            'UNIT FLASH-1\n'
            '    TYPE: Flash\n'
            '    PORTS:\n'
            '        in  : inlet\n'
            '        vap : vapor_outlet\n'
            '        liq : liquid_outlet\n'
            '    PARAMS:\n'
            '        P = 10 [bar]\n'
            '        heat_duty = 0 [kW]\n'
        )

        result = Simulator.from_string(pfd).run()

        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.units['FLASH-1'].heat_duty, 0.0, places=8)
        self.assertAlmostEqual(
            result.units['FLASH-1'].performance['vapor_fraction'],
            0.227121267,
            places=8,
        )



if __name__ == '__main__':
    unittest.main()
