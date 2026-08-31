import math
import unittest

from dof_analyzer import DOFAnalyzer, SpecificationStatus
from pfd_parser import PFDParser, ProcessFlowDiagram, validate_pfd
from simulator import Simulator
from thermodynamics import create_thermodynamics
from unit_operations_base import UnitOperationError
from unit_operations_reactors import KineticsBatch


def reaction_definition(**overrides):
    definition = {
        'equation': 'C2H4O -> CH3CHO',
        'A': 2.0,
        'Ea': 0.0,
        'Ea_unit': 'J/mol',
        'rate_basis': 'concentration',
        'concentration_unit': 'kmol/m3',
        'pressure_unit': 'bar',
        'rate_unit': 'kmol/m3/h',
    }
    definition.update(overrides)
    return definition


class BatchReactorUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thermo = create_thermodynamics(['C2H4O', 'CH3CHO'], 'IDEAL')
        cls.feed = cls.thermo.calculate_state(
            500.0, 10.0, 10.0, {'C2H4O': 1.0},
            phase='vapor', flash=False,
        )

    def solve(self, inlets=None, **extra):
        params = {
            'V_batch': 10.0,
            't_reaction': 0.5,
            'phase': 'vapor',
            'mode': 'isothermal',
            'T': 500.0,
            'profile_points': 9,
            'reactions': [reaction_definition()],
        }
        params.update(extra)
        return KineticsBatch('B-1', self.thermo, params).solve(
            inlets or {'charge': self.feed}
        )

    def solve_expanding_vapor(self, **extra):
        thermo = create_thermodynamics(['C2H5OH', 'C2H4', 'H2O'], 'IDEAL')
        feed = thermo.calculate_state(
            600.0, 5.0, 2.0, {'C2H5OH': 1.0},
            phase='vapor', flash=False,
        )
        params = {
            'V_batch': 20.0,
            't_rxn': 1.0,
            'phase': 'vapor',
            'mode': 'isothermal',
            'T': 600.0,
            'profile_points': 9,
            'reactions': [reaction_definition(
                equation='C2H5OH -> C2H4 + H2O', A=1.0,
            )],
        }
        params.update(extra)
        return KineticsBatch('B-expand', thermo, params).solve({'charge': feed})

    def test_isothermal_batch_maps_transient_conversion_to_average_outlet(self):
        result = self.solve()
        outlet = result.outlet_streams['out']
        performance = result.performance

        expected_conversion = 1.0 - math.exp(-2.0 * 0.5)
        self.assertAlmostEqual(
            performance['component_conversions']['C2H4O'],
            expected_conversion,
            places=6,
        )
        self.assertAlmostEqual(outlet.F, self.feed.F, places=8)
        self.assertAlmostEqual(outlet.T, 500.0, places=10)
        self.assertEqual(performance['profile_points'], 9)
        self.assertAlmostEqual(performance['profile'][0]['time_h'], 0.0)
        self.assertAlmostEqual(performance['profile'][-1]['time_h'], 0.5)
        self.assertEqual(performance['outlet_basis'], 'time_averaged')
        self.assertLess(
            performance['maximum_material_balance_residual_kmol_batch'], 1.0e-8
        )
        self.assertLess(abs(performance['energy_balance_residual_kJ_batch']), 1.0e-6)

    def test_two_of_three_schedule_resolves_each_missing_quantity(self):
        volume_time = self.solve()
        N = volume_time.performance['N_vessels']

        count_time = self.solve(V_batch=None, N=N)
        self.assertAlmostEqual(
            count_time.performance['fleet_utilization'], 1.0, places=12
        )

        volume_count = self.solve(t_reaction=None, N=N)
        self.assertGreater(volume_count.performance['reaction_time_h'], 0.0)
        self.assertAlmostEqual(
            volume_count.performance['fleet_utilization'], 1.0, places=12
        )

        with self.assertRaisesRegex(UnitOperationError, 'exactly two'):
            self.solve(N=N)

    def test_adiabatic_duty_and_jacketed_modes_close_energy_balance(self):
        adiabatic = self.solve(mode='adiabatic', T=None)
        heated = self.solve(
            mode='duty', T=None, Q_batch=100.0, __unit__Q_batch='kJ'
        )
        jacketed = self.solve(
            mode='jacketed', T=None, UA=50.0, __unit__UA='W/K',
            T_jacket=450.0,
        )

        self.assertAlmostEqual(adiabatic.performance['heat_per_batch_kJ'], 0.0)
        self.assertAlmostEqual(heated.performance['heat_per_batch_kJ'], 100.0, places=7)
        self.assertLess(jacketed.performance['heat_per_batch_kJ'], 0.0)
        for result in (adiabatic, heated, jacketed):
            self.assertLess(
                abs(result.performance['energy_balance_residual_kJ_batch']), 1.0e-5
            )

        alias_jacket = self.solve(
            mode='jacketed', T=None,
            heat_transfer_coefficient=25.0,
            __unit__heat_transfer_coefficient='W/m2/K',
            heat_transfer_area=2.0,
            __unit__heat_transfer_area='m2',
            T_coolant=450.0,
        )
        self.assertAlmostEqual(alias_jacket.performance['UA_W_per_K'], 50.0)
        self.assertAlmostEqual(
            alias_jacket.performance['heat_per_batch_kJ'],
            jacketed.performance['heat_per_batch_kJ'],
            places=7,
        )

        with self.assertRaisesRegex(UnitOperationError, 'constant_pressure'):
            self.solve(pressure_mode='rigid')

    def test_semi_batch_feed_is_apportioned_and_added_over_requested_window(self):
        initial = self.thermo.calculate_state(
            500.0, 10.0, 6.0, {'C2H4O': 1.0}, phase='vapor', flash=False
        )
        addition = self.thermo.calculate_state(
            500.0, 10.0, 4.0, {'C2H4O': 1.0}, phase='vapor', flash=False
        )
        result = self.solve(
            {'charge': initial, 'addition': addition},
            semi_batch_feeds='addition',
            feed_start_times='addition:0.1',
            feed_stop_times='addition:0.4',
        )

        profile_times = [row['time_h'] for row in result.performance['profile']]
        self.assertIn(0.1, profile_times)
        self.assertIn(0.4, profile_times)
        self.assertEqual(result.performance['semi_batch_feed_ports'], ['addition'])
        self.assertAlmostEqual(result.outlet_streams['out'].F, 10.0, places=7)
        self.assertGreater(
            result.performance['component_conversions']['C2H4O'], 0.0
        )
        self.assertLess(
            result.performance['component_conversions']['C2H4O'],
            1.0 - math.exp(-1.0),
        )

        with self.assertRaisesRegex(UnitOperationError, 'trajectory exceeds usable'):
            self.solve(
                {'charge': initial, 'addition': addition},
                semi_batch_feeds='addition',
                feed_start_times='addition:0.1',
                feed_stop_times='addition:0.4',
                V_vessel=9.0,
            )

    def test_recycle_context_reduces_profile_but_keeps_outlet_validation(self):
        unit = KineticsBatch('B-1', self.thermo, {
            'V_batch': 10.0,
            't_reaction': 0.5,
            'phase': 'vapor',
            'T': 500.0,
            'profile_points': 11,
            'reactions': [reaction_definition()],
        })
        unit.solve_context = {
            'recycle_evaluation': 3,
            'recycle_final_pass': False,
            'expensive_diagnostics': False,
        }
        reduced = unit.solve({'charge': self.feed})
        self.assertEqual(reduced.performance['profile_points'], 2)
        self.assertFalse(reduced.performance['detailed_diagnostics'])
        self.assertTrue(reduced.performance['phase_stability'])

    def test_custom_transfer_schedule_reports_buffering_and_capacity(self):
        result = self.solve(t_fill=0.01, t_drain=0.001, t_turnaround=0.02)
        self.assertTrue(result.performance['surge_buffer_required'])
        self.assertLess(result.performance['continuous_discharge_coverage'], 1.0)
        self.assertLessEqual(result.performance['fleet_utilization'], 1.0)

        with self.assertRaisesRegex(UnitOperationError, 'exceeds usable vessel volume'):
            self.solve(V_vessel=5.0, fill_fraction=1.0)

    def test_multiple_reactions_are_integrated_simultaneously(self):
        result = self.solve(reactions=[
            reaction_definition(A=2.0),
            reaction_definition(equation='CH3CHO -> C2H4O', A=0.5),
        ])
        reactions = result.performance['reactions']
        self.assertEqual(len(reactions), 2)
        self.assertGreater(reactions[0]['extent_kmol_batch'], 0.0)
        self.assertGreater(reactions[1]['extent_kmol_batch'], 0.0)
        self.assertLess(
            result.performance['maximum_material_balance_residual_kmol_batch'],
            1.0e-8,
        )

    def test_custom_net_batch_can_run_written_stoichiometry_in_reverse(self):
        product_feed = self.thermo.calculate_state(
            500.0,
            10.0,
            10.0,
            {'CH3CHO': 1.0},
            phase='vapor',
            flash=False,
        )
        result = self.solve(
            {'charge': product_feed},
            reactions=[reaction_definition(
                type='custom_net',
                A=0.5,
                expression="-k * C['CH3CHO']",
            )],
        )

        reaction = result.performance['reactions'][0]
        self.assertEqual(reaction['kinetic_type'], 'custom_net')
        self.assertLess(reaction['extent_kmol_batch'], 0.0)
        self.assertGreater(result.outlet_streams['out'].composition['C2H4O'], 0.0)
        self.assertGreater(
            result.performance['component_conversions']['CH3CHO'],
            0.0,
        )

    def test_depletion_before_requested_reaction_time_is_rejected(self):
        with self.assertRaises(UnitOperationError) as caught:
            self.solve(reactions=[reaction_definition(A=100.0, order_C2H4O=0.0)])
        message = str(caught.exception)
        self.assertIn('inventory depleted', message)
        self.assertNotIn('LSODA', message)

    def test_repeated_batch_solve_is_deterministic(self):
        first = self.solve(mode='adiabatic', T=None)
        second = self.solve(mode='adiabatic', T=None)
        self.assertEqual(first.outlet_streams['out'].to_dict(), second.outlet_streams['out'].to_dict())
        self.assertEqual(first.performance['profile'], second.performance['profile'])
        self.assertEqual(
            first.performance['solver_evaluations'],
            second.performance['solver_evaluations'],
        )

    def test_explicit_adaptive_solver_methods_agree(self):
        baseline = self.solve(profile_points=3, solver='RK45')
        for method in ('BDF', 'Radau', 'LSODA'):
            with self.subTest(method=method):
                result = self.solve(profile_points=3, solver=method)
                self.assertEqual(result.performance['solver_method'], method)
                self.assertAlmostEqual(
                    result.outlet_streams['out'].T,
                    baseline.outlet_streams['out'].T,
                    places=8,
                )
                for component, fraction in baseline.outlet_streams['out'].composition.items():
                    self.assertAlmostEqual(
                        result.outlet_streams['out'].composition.get(component, 0.0),
                        fraction,
                        places=6,
                    )

    def test_average_outlet_preserves_reaction_induced_total_mole_change(self):
        thermo = create_thermodynamics(['C2H5OH', 'C2H4', 'H2O'], 'IDEAL')
        feed = thermo.calculate_state(
            600.0, 5.0, 2.0, {'C2H5OH': 1.0}, phase='vapor', flash=False
        )
        result = KineticsBatch('B-D', thermo, {
            'V_batch': 20.0,
            't_reaction': 0.2,
            'phase': 'vapor',
            'mode': 'isothermal',
            'T': 600.0,
            'profile_points': 5,
            'reactions': [reaction_definition(
                equation='C2H5OH -> C2H4 + H2O', A=1.0,
            )],
        }).solve({'charge': feed})

        self.assertGreater(result.outlet_streams['out'].F, feed.F)
        self.assertAlmostEqual(
            result.outlet_streams['out'].F - feed.F,
            result.performance['reactions'][0]['average_extent_kmol_h'],
            places=7,
        )

    def test_reaction_generated_vapor_expansion_is_reported_and_can_be_sized(self):
        unconstrained = self.solve_expanding_vapor()
        sized = self.solve_expanding_vapor(V_vessel=35.0)
        expected_conversion = 1.0 - math.exp(-1.0)
        expected_volume = 20.0 * (1.0 + expected_conversion)

        for result in (unconstrained, sized):
            self.assertAlmostEqual(
                result.performance['profile'][0]['working_volume_m3'],
                20.0,
                places=9,
            )
            self.assertAlmostEqual(
                result.performance['maximum_working_volume_m3'],
                expected_volume,
                places=6,
            )
            self.assertAlmostEqual(
                result.outlet_streams['out'].F / 2.0,
                1.0 + expected_conversion,
                places=6,
            )
        self.assertIsNone(unconstrained.performance['V_vessel_m3'])
        self.assertEqual(sized.performance['V_vessel_m3'], 35.0)

    def test_reaction_generated_vapor_expansion_trips_capacity_once(self):
        with self.assertRaises(UnitOperationError) as caught:
            self.solve_expanding_vapor(V_vessel=30.0)

        message = str(caught.exception)
        self.assertIn('trajectory exceeds usable vessel volume', message)
        self.assertIn('t=0.693147 h', message)
        self.assertEqual(message.count('trajectory exceeds usable vessel volume'), 1)
        self.assertNotIn('RK45:', message)
        self.assertNotIn('LSODA:', message)

    def test_fill_fraction_limits_reaction_generated_vapor_expansion(self):
        with self.assertRaisesRegex(
            UnitOperationError,
            r'trajectory exceeds usable vessel volume.*t=0\.693147 h',
        ):
            self.solve_expanding_vapor(V_vessel=40.0, fill_fraction=0.75)

    def test_liquid_batch_rejects_reaction_generated_vapor_phase(self):
        thermo = create_thermodynamics(['C2H5OH', 'C2H4', 'H2O'], 'IDEAL')
        feed = thermo.calculate_state(
            350.0, 5.0, 2.0, {'C2H5OH': 1.0},
            phase='liquid', flash=False,
        )
        with self.assertRaisesRegex(UnitOperationError, 'phase-unstable'):
            KineticsBatch('B-phase', thermo, {
                'V_batch': 1.0,
                't_rxn': 1.0,
                'phase': 'liquid',
                'mode': 'isothermal',
                'T': 350.0,
                'profile_points': 5,
                'reactions': [reaction_definition(
                    equation='C2H5OH -> C2H4 + H2O', A=1.0,
                )],
            }).solve({'charge': feed})

    def test_liquid_activity_batch_retains_global_stability_checks(self):
        thermo = create_thermodynamics(['C2H4O', 'CH3CHO'], 'NRTL')
        feed = thermo.calculate_state(
            300.0, 10.0, 1.0, {'C2H4O': 0.5, 'CH3CHO': 0.5},
            phase='liquid', flash=False,
        )
        result = KineticsBatch('B-L', thermo, {
            'V_batch': 0.01,
            't_reaction': 0.2,
            'phase': 'liquid',
            'mode': 'isothermal',
            'T': 300.0,
            'profile_points': 5,
            'reactions': [reaction_definition(
                type='custom', A=0.2, expression="k*a['C2H4O']",
            )],
        }).solve({'charge': feed})

        checks = result.performance['global_phase_stability_profile_checks']
        self.assertEqual(sorted(checks), [0, 2, 4])
        self.assertEqual(
            result.performance['phase_stability']['liquid_spinodal'][
                'activity_backend'
            ],
            'CompiledNRTLBackend',
        )


class BatchReactorPFDTests(unittest.TestCase):
    PFD = """
PROCESS: Batch reactor PFD
THERMO_METHOD: IDEAL

COMPONENTS:
    C2H4O | Ethylene oxide | formula=C2H4O
    CH3CHO | Acetaldehyde | formula=C2H4O

STREAM Feed : FEED -> B-1.charge
    T = 226.85 [C]
    P = 10 [bar]
    F = 10 [kmol/h]
    x = C2H4O:1

STREAM Product : B-1.out -> PRODUCT

UNIT B-1 : BatchReactor
    V_batch = 10 [m3]
    t_rxn = 0.5 [h]
    phase = vapor
    mode = isothermal
    T = 226.85 [C]
    profile_points = 7
    REACTIONS:
        C2H4O -> CH3CHO | A=2, Ea=0, Ea_unit=J/mol, rate_basis=concentration, concentration_unit=kmol/m3, pressure_unit=bar, rate_unit=kmol/m3/h
"""

    def test_compact_batch_pfd_validates_round_trips_and_solves(self):
        pfd = PFDParser().parse(self.PFD)
        errors, _warnings = validate_pfd(pfd)
        self.assertEqual(errors, [])
        serialized = pfd.to_pfd()
        self.assertIn('TYPE: BatchReactor', serialized)
        restored = PFDParser().parse(serialized)
        self.assertEqual(validate_pfd(restored)[0], [])
        self.assertEqual(restored.units[0].unit_type, 'BatchReactor')
        restored_dict = ProcessFlowDiagram.from_dict(pfd.to_dict())
        self.assertEqual(validate_pfd(restored_dict)[0], [])
        self.assertEqual(restored_dict.units[0].unit_type, 'BatchReactor')
        dof = DOFAnalyzer(pfd).analyze()
        batch_dof = next(item for item in dof.unit_results if item.entity_id == 'B-1')
        self.assertEqual(batch_dof.status, SpecificationStatus.OK)

        result = Simulator.from_string(self.PFD).run()
        self.assertTrue(result.converged, result.errors)
        self.assertIn('B-1', result.units)
        self.assertAlmostEqual(result.streams['Product'].F, 10.0, places=7)

    def test_batch_dof_distinguishes_under_and_over_specification(self):
        under = PFDParser().parse(self.PFD.replace('    t_rxn = 0.5 [h]\n', ''))
        under_result = next(
            item for item in DOFAnalyzer(under).analyze().unit_results
            if item.entity_id == 'B-1'
        )
        self.assertEqual(under_result.status, SpecificationStatus.UNDER_SPECIFIED)

        over = PFDParser().parse(
            self.PFD.replace('    t_rxn = 0.5 [h]', '    t_rxn = 0.5 [h]\n    N = 4')
        )
        over_result = next(
            item for item in DOFAnalyzer(over).analyze().unit_results
            if item.entity_id == 'B-1'
        )
        self.assertEqual(over_result.status, SpecificationStatus.OVER_SPECIFIED)

    def test_kinetics_batch_alias_canonicalizes_to_batch_reactor(self):
        pfd = PFDParser().parse(
            self.PFD.replace('UNIT B-1 : BatchReactor', 'UNIT B-1 : KineticsBatch')
        )
        self.assertEqual(pfd.units[0].unit_type, 'BatchReactor')
        self.assertEqual(validate_pfd(pfd)[0], [])

    def test_compact_semi_batch_ports_preserve_schedule_names(self):
        text = self.PFD.replace(
            'F = 10 [kmol/h]',
            'F = 6 [kmol/h]',
        ).replace(
            'STREAM Product : B-1.out -> PRODUCT',
            '''STREAM Addition : FEED -> B-1.addition
    T = 226.85 [C]
    P = 10 [bar]
    F = 4 [kmol/h]
    x = C2H4O:1

STREAM Product : B-1.out -> PRODUCT''',
        ).replace(
            '    profile_points = 7',
            '''    profile_points = 7
    semi_batch_feeds = addition
    feed_start_times = addition:0.1
    feed_stop_times = addition:0.4''',
        )
        pfd = PFDParser().parse(text)
        self.assertEqual(
            [stream.destination.port_id for stream in pfd.streams[:2]],
            ['charge', 'addition'],
        )
        errors, _warnings = validate_pfd(pfd)
        self.assertEqual(errors, [])

        result = Simulator.from_string(text).run()
        self.assertTrue(result.converged, result.errors)
        self.assertEqual(
            result.units['B-1'].performance['semi_batch_feed_ports'],
            ['addition'],
        )
        self.assertAlmostEqual(result.streams['Product'].F, 10.0, places=7)

    def test_batch_reactor_converges_inside_continuous_recycle(self):
        text = """
PROCESS: Batch reactor recycle
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
RECYCLE_METHOD: BROYDEN
COMPONENTS:
    C2H4O | Ethylene oxide | formula=C2H4O
    CH3CHO | Acetaldehyde | formula=C2H4O
STREAM Fresh : FEED -> M.fresh
    T = 500 [K]
    P = 10 [bar]
    F = 10 [kmol/h]
    x = C2H4O:1
STREAM Recycle : S.recycle -> M.recycle
STREAM Mixed : M.out -> B.charge
STREAM Reacted : B.out -> S.in
STREAM Product : S.product -> PRODUCT
UNIT M : Mixer
UNIT B : BatchReactor
    V_batch = 10 [m3]
    t_reaction = 0.2 [h]
    phase = vapor
    mode = isothermal
    T = 500 [K]
    profile_points = 7
    REACTIONS:
        C2H4O -> CH3CHO | A=2, Ea=0, Ea_unit=J/mol, rate_basis=concentration, concentration_unit=kmol/m3, pressure_unit=bar, rate_unit=kmol/m3/h
UNIT S : Splitter
    outlets = product,recycle
    product_split_frac = 0.5
"""
        result = Simulator.from_string(text).run(max_iterations=50)
        self.assertTrue(result.converged, result.errors)
        self.assertEqual(result.recycle_info['tear_streams'], ['Recycle'])
        self.assertLess(result.mass_balance_error, 1.0e-4)
        self.assertLess(result.energy_balance_error, 1.0e-4)
        self.assertEqual(result.units['B'].performance['profile_points'], 7)


if __name__ == '__main__':
    unittest.main()
