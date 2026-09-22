import math
import unittest
from unittest.mock import patch

from thermodynamics import create_thermodynamics
from pfd_parser import PFDParser, validate_pfd
from simulator import Simulator
from unit_operations_base import UnitOperationError
from unit_operations_reactors import KineticsPFR


def reaction_definition(equation='C2H4O -> CH3CHO', **overrides):
    definition = {
        'equation': equation,
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


class AdaptivePFRUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.components = ['C2H4O', 'CH3CHO']
        cls.thermo = create_thermodynamics(cls.components, 'IDEAL')
        cls.feed = cls.thermo.calculate_state(
            500.0,
            10.0,
            10.0,
            {'C2H4O': 1.0},
            phase='vapor',
            flash=False,
        )

    def solve(self, **extra):
        params = {
            'volume': 0.1,
            'diameter': 0.1,
            'mode': 'isothermal',
            'T': 500.0,
            'phase': 'vapor',
            'profile_points': 11,
            'reactions': [reaction_definition()],
        }
        params.update(extra)
        return KineticsPFR('PFR-1', self.thermo, params).solve({'in': self.feed})

    def test_isothermal_pfr_closes_material_energy_and_reports_profiles(self):
        result = self.solve()
        outlet = result.outlet_streams['out']
        performance = result.performance

        self.assertEqual(performance['solver_method'], 'RK45')
        self.assertEqual(performance['profile_points'], 11)
        self.assertEqual(performance['profile'][0]['reactor_volume_m3'], 0.0)
        self.assertAlmostEqual(
            performance['profile'][-1]['reactor_volume_m3'],
            0.1,
            places=13,
        )
        self.assertAlmostEqual(outlet.T, 500.0, places=13)
        self.assertLess(
            performance['maximum_material_balance_residual_kmol_h'],
            1.0e-9,
        )
        self.assertAlmostEqual(
            outlet.F * outlet.H - self.feed.F * self.feed.H,
            result.heat_duty,
            places=7,
        )

    def test_adiabatic_and_specified_duty_integrate_exact_enthalpy_flow(self):
        adiabatic = self.solve(mode='adiabatic', T=None)
        heated = self.solve(
            mode='duty',
            T=None,
            Q=1.0,
            __unit__Q='kW',
        )

        self.assertEqual(adiabatic.heat_duty, 0.0)
        self.assertGreater(adiabatic.outlet_streams['out'].T, self.feed.T)
        self.assertAlmostEqual(heated.heat_duty, 3600.0, places=9)
        self.assertGreater(
            heated.outlet_streams['out'].T,
            adiabatic.outlet_streams['out'].T,
        )
        for result in (adiabatic, heated):
            self.assertLess(
                abs(result.performance['energy_balance_residual_kW']),
                1.0e-8,
            )
            self.assertEqual(result.performance['solver_method'], 'LSODA')

    def test_jacketed_mode_uses_resolved_perimeter_and_heat_direction(self):
        result = self.solve(
            mode='jacketed',
            T=None,
            U=100.0,
            __unit__U='W/m2/K',
            T_jacket=450.0,
        )
        outlet = result.outlet_streams['out']

        self.assertLess(outlet.T, self.feed.T)
        self.assertGreater(outlet.T, 450.0)
        self.assertLess(result.heat_duty, 0.0)
        self.assertLess(
            abs(result.performance['energy_balance_residual_kW']),
            1.0e-8,
        )

    def test_specified_pressure_drop_is_distributed_linearly(self):
        result = self.solve(P_drop=1.0)
        pressures = [row['pressure_bar'] for row in result.performance['profile']]

        self.assertAlmostEqual(result.outlet_streams['out'].P, 9.0, places=12)
        self.assertTrue(all(
            left >= right for left, right in zip(pressures, pressures[1:])
        ))
        for index, pressure in enumerate(pressures):
            self.assertAlmostEqual(pressure, 10.0 - index / 10.0, places=12)

    def test_darcy_and_ergun_use_local_transport_properties(self):
        darcy = self.solve(
            pressure_drop_model='darcy',
            roughness=1.0e-5,
            friction_model='colebrook',
        )
        ergun = self.solve(
            pressure_drop_model='ergun',
            void_fraction=0.4,
            particle_diameter=0.01,
        )

        for result in (darcy, ergun):
            performance = result.performance
            self.assertGreater(performance['pressure_drop_bar'], 0.0)
            self.assertGreater(performance['inlet_velocity_m_s'], 0.0)
            self.assertGreater(performance['inlet_reynolds_number'], 0.0)
            pressures = [row['pressure_bar'] for row in performance['profile']]
            self.assertTrue(all(
                left >= right for left, right in zip(pressures, pressures[1:])
            ))
        self.assertGreater(
            ergun.performance['pressure_drop_bar'],
            darcy.performance['pressure_drop_bar'],
        )
        inlet_volumetric_flow = self.feed.F / self.feed.rho
        self.assertLess(
            ergun.performance['residence_time_h'],
            0.1 / inlet_volumetric_flow,
        )

    def test_viscosity_is_only_requested_for_hydraulic_models(self):
        with patch.object(
            self.thermo,
            'mixture_viscosity',
            side_effect=AssertionError('viscosity should not be requested'),
        ):
            result = self.solve()
        self.assertIsNone(result.outlet_streams['out'].mu)

        with patch.object(
            self.thermo,
            'mixture_viscosity',
            wraps=self.thermo.mixture_viscosity,
        ) as viscosity:
            hydraulic = self.solve(
                pressure_drop_model='darcy',
                roughness=1.0e-5,
            )
        self.assertGreater(viscosity.call_count, 0)
        self.assertIsNotNone(hydraulic.outlet_streams['out'].mu)

    def test_recycle_iterations_use_minimal_profile_and_outlet_validation(self):
        unit = KineticsPFR('PFR-R', self.thermo, {
            'volume': 0.1,
            'diameter': 0.1,
            'mode': 'adiabatic',
            'phase': 'vapor',
            'profile_points': 11,
            'reactions': [reaction_definition()],
        })
        try:
            unit.solve_context = {
                'recycle_evaluation': 3,
                'recycle_final_pass': False,
                'expensive_diagnostics': False,
            }
            reduced = unit.solve({'in': self.feed})
            unit.solve_context = {
                'recycle_evaluation': 5,
                'recycle_final_pass': False,
                'expensive_diagnostics': True,
            }
            scheduled = unit.solve({'in': self.feed})
            unit.solve_context = {
                'recycle_evaluation': 3,
                'recycle_final_pass': True,
                'expensive_diagnostics': True,
            }
            detailed = unit.solve({'in': self.feed})
        finally:
            unit.solve_context = {}

        self.assertFalse(reduced.performance['detailed_diagnostics'])
        self.assertEqual(reduced.performance['requested_profile_points'], 11)
        self.assertEqual(reduced.performance['profile_points'], 2)
        self.assertEqual(len(reduced.performance['phase_stability_profile']), 1)
        self.assertTrue(scheduled.performance['detailed_diagnostics'])
        self.assertEqual(scheduled.performance['profile_points'], 11)
        self.assertEqual(len(scheduled.performance['phase_stability_profile']), 11)
        self.assertTrue(detailed.performance['detailed_diagnostics'])
        self.assertEqual(detailed.performance['profile_points'], 11)
        self.assertEqual(len(detailed.performance['phase_stability_profile']), 11)
        self.assertAlmostEqual(
            reduced.outlet_streams['out'].T,
            detailed.outlet_streams['out'].T,
            places=8,
        )
        self.assertAlmostEqual(
            reduced.outlet_streams['out'].F,
            detailed.outlet_streams['out'].F,
            places=8,
        )

    def test_compiled_cubic_pfr_uses_temperature_only_ph_in_rhs(self):
        thermo = create_thermodynamics(self.components, 'RKS-BM')
        feed = thermo.calculate_state(
            500.0,
            10.0,
            10.0,
            {'C2H4O': 1.0},
            phase='vapor',
            flash=False,
        )
        unit = KineticsPFR('PFR-fast-PH', thermo, {
            'volume': 0.1,
            'diameter': 0.1,
            'mode': 'adiabatic',
            'phase': 'vapor',
            'profile_points': 5,
            'reactions': [reaction_definition()],
        })
        with (
            patch.object(
                thermo,
                'temperature_at_PH',
                wraps=thermo.temperature_at_PH,
            ) as temperature_ph,
            patch.object(
                thermo,
                'calculate_state',
                wraps=thermo.calculate_state,
            ) as full_state,
        ):
            result = unit.solve({'in': feed})

        self.assertGreater(temperature_ph.call_count, 0)
        self.assertGreater(temperature_ph.call_count, full_state.call_count)
        self.assertEqual(result.performance['profile_points'], 5)
        self.assertEqual(len(result.performance['profile']), 5)
        self.assertLess(
            abs(result.performance['energy_balance_residual_kW']),
            1.0e-8,
        )
        self.assertLess(
            result.performance['maximum_material_balance_residual_kmol_h'],
            1.0e-9,
        )

    def test_geometry_is_inferred_or_rejected_when_inconsistent(self):
        by_length = self.solve(volume=None, length=10.0, diameter=0.1)
        expected_volume = math.pi * 0.1**2 / 4.0 * 10.0
        self.assertAlmostEqual(
            by_length.performance['volume_m3'],
            expected_volume,
            places=13,
        )

        with self.assertRaisesRegex(UnitOperationError, 'inconsistent'):
            self.solve(volume=0.1, length=1.0, diameter=0.1)
        with self.assertRaisesRegex(UnitOperationError, 'length and diameter'):
            self.solve(
                diameter=None,
                mode='jacketed',
                T=None,
                U=100.0,
                T_jacket=450.0,
            )

    def test_stiff_auto_selection_and_constant_rate_depletion_continuation(self):
        stiff = self.solve(
            reactions=[
                reaction_definition(A=1.0e5),
                reaction_definition(
                    'CH3CHO -> C2H4O',
                    A=1.0e2,
                ),
            ],
        )
        self.assertEqual(stiff.performance['solver_method'], 'LSODA')
        self.assertLess(
            stiff.performance['maximum_material_balance_residual_kmol_h'],
            1.0e-7,
        )

        depleted = self.solve(
            volume=100.0,
            mode='duty',
            T=None,
            Q=1.0,
            __unit__Q='kW',
            reactions=[reaction_definition(
                type='custom',
                A=1.0,
                expression='k',
            )],
        )
        performance = depleted.performance
        self.assertAlmostEqual(
            performance['component_conversions']['C2H4O'],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            performance['profile'][-1]['reactor_volume_m3'],
            100.0,
            places=12,
        )
        self.assertEqual(performance['profile'][-1]['reaction_rates'], [0.0])
        self.assertAlmostEqual(depleted.heat_duty, 3600.0, places=8)
        self.assertAlmostEqual(
            performance['component_depletion_events'][0][
                'reaction_measure_at_event'
            ],
            10.0,
            places=7,
        )
        self.assertAlmostEqual(
            performance['inactive_remaining_reactor_volume_m3'],
            90.0,
            places=7,
        )
        self.assertTrue(any(
            'component inventory depleted for C2H4O' in warning
            for warning in depleted.warnings
        ))

    def test_depletion_active_set_allows_producing_reaction_to_continue(self):
        thermo = create_thermodynamics(
            ['C2H4O', 'CH3CHO', 'C2H5OH', 'H2'],
            'IDEAL',
        )
        feed = thermo.calculate_state(
            500.0,
            10.0,
            10.0,
            {'C2H4O': 0.1, 'C2H5OH': 0.9},
            phase='vapor',
            flash=False,
        )
        result = KineticsPFR('PFR-active-set', thermo, {
            'volume': 1.0,
            'phase': 'vapor',
            'mode': 'isothermal',
            'T': 500.0,
            'profile_points': 11,
            'reactions': [
                reaction_definition(
                    type='custom',
                    A=10.0,
                    expression='k',
                ),
                reaction_definition(
                    'C2H4O + H2 -> C2H5OH',
                    type='custom_net',
                    A=1.0,
                    expression='-k',
                ),
            ],
        }).solve({'in': feed})
        performance = result.performance
        extents = [reaction['extent_kmol_h'] for reaction in performance['reactions']]

        self.assertAlmostEqual(extents[0], 2.0, places=7)
        self.assertAlmostEqual(extents[1], -1.0, places=7)
        self.assertAlmostEqual(
            performance['component_conversions']['C2H4O'],
            1.0,
            places=10,
        )
        self.assertGreater(
            result.outlet_streams['out'].component_flows()['C2H5OH'],
            0.0,
        )
        self.assertEqual(performance['profile'][-1]['reaction_rates'], [1.0, -1.0])
        self.assertFalse(
            performance['component_depletion_events'][0][
                'all_reactions_inactive_downstream'
            ]
        )
        self.assertNotIn('inactive_remaining_reactor_volume_m3', performance)

    def test_repeated_solve_is_deterministic(self):
        unit = KineticsPFR('PFR-1', self.thermo, {
            'volume': 0.1,
            'diameter': 0.1,
            'mode': 'adiabatic',
            'phase': 'vapor',
            'profile_points': 11,
            'reactions': [reaction_definition()],
        })
        first = unit.solve({'in': self.feed})
        second = unit.solve({'in': self.feed})
        self.assertEqual(first.outlet_streams['out'].to_dict(), second.outlet_streams['out'].to_dict())
        self.assertEqual(first.performance, second.performance)

    def test_explicit_adaptive_solver_methods_are_supported(self):
        for method in ('RK45', 'BDF', 'Radau', 'LSODA'):
            with self.subTest(method=method):
                result = self.solve(solver=method, profile_points=5)
                self.assertEqual(result.performance['solver_method'], method)
                self.assertLess(
                    result.performance['maximum_material_balance_residual_kmol_h'],
                    1.0e-8,
                )

    def test_adiabatic_ergun_couples_pressure_temperature_and_composition(self):
        result = self.solve(
            mode='adiabatic',
            T=None,
            pressure_drop_model='ergun',
            void_fraction=0.4,
            particle_diameter=0.02,
            profile_points=7,
        )
        outlet = result.outlet_streams['out']
        self.assertLess(outlet.P, self.feed.P)
        self.assertGreater(outlet.T, self.feed.T)
        self.assertGreater(
            result.performance['component_conversions']['C2H4O'],
            0.0,
        )
        self.assertLess(
            abs(result.performance['energy_balance_residual_kW']),
            1.0e-8,
        )

    def test_tolerance_refinement_converges_to_same_outlet(self):
        coarse = self.solve(
            relative_tolerance=1.0e-5,
            absolute_tolerance=1.0e-7,
            profile_points=5,
        )
        fine = self.solve(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-11,
            profile_points=17,
        )
        self.assertAlmostEqual(
            coarse.outlet_streams['out'].composition['C2H4O'],
            fine.outlet_streams['out'].composition['C2H4O'],
            places=8,
        )
        self.assertNotEqual(
            coarse.performance['profile_points'],
            fine.performance['profile_points'],
        )

    def test_phase_unstable_axial_path_is_rejected(self):
        thermo = create_thermodynamics(['CO', 'H2', 'CH3OH'], 'IDEAL')
        feed = thermo.calculate_state(
            350.0,
            80.0,
            3.0,
            {'CO': 1.0 / 3.0, 'H2': 2.0 / 3.0},
            phase='vapor',
            flash=False,
        )
        with self.assertRaisesRegex(UnitOperationError, 'phase-unstable'):
            KineticsPFR('PFR-U', thermo, {
                'volume': 0.01,
                'mode': 'isothermal',
                'T': 350.0,
                'phase': 'vapor',
                'profile_points': 3,
                'reactions': [reaction_definition(
                    'CO + 2 H2 <=> CH3OH',
                    A=1.0,
                    rate_basis='activity',
                )],
            }).solve({'in': feed})


class AdaptivePFRPFDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.components = ['C2H4O', 'CH3CHO']

    def test_compact_pfd_units_catalog_pressure_and_duty_round_trip(self):
        text = '''\
PROCESS: Compact adaptive PFR
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
COMPONENTS:
    C2H4O | Ethylene oxide | formula=C2H4O
    CH3CHO | Acetaldehyde | formula=C2H4O
REACTIONS:
    isomerization : C2H4O -> CH3CHO | A=2, Ea=0, Ea_unit=J/mol, rate_basis=concentration, concentration_unit=kmol/m3, pressure_unit=bar, rate_unit=kmol/m3/h
STREAM Feed : -> PFR-1.in
    T = 500 [K]
    P = 10 [bar]
    F = 10 [kmol/h]
    x = C2H4O:1
STREAM Product : PFR-1.out
UNIT PFR-1 : PFR
    volume = 100 [L]
    diameter = 10 [cm]
    mode = duty
    Q = 1 [kW]
    phase = vapor
    P_drop = 100 [kPa]
    solver = Radau
    maximum_step = 1 [m]
    profile_points = 5
    REACTIONS:
        @isomerization
'''
        pfd = PFDParser().parse(text)
        self.assertEqual(validate_pfd(pfd), ([], []))
        restored = PFDParser().parse(pfd.to_pfd())
        self.assertEqual(validate_pfd(restored), ([], []))

        result = Simulator(restored).run()
        self.assertTrue(result.converged, result.warnings)
        self.assertEqual(result.errors, [])
        performance = result.units['PFR-1'].performance
        self.assertAlmostEqual(performance['volume_m3'], 0.1, places=13)
        self.assertAlmostEqual(performance['diameter_m'], 0.1, places=13)
        self.assertAlmostEqual(performance['P_out_bar'], 9.0, places=12)
        self.assertAlmostEqual(performance['duty_kW'], 1.0, places=10)
        self.assertEqual(performance['solver_method'], 'Radau')
        self.assertEqual(performance['profile_points'], 5)

    def test_adaptive_pfr_catalog_recycle_converges_and_closes_balances(self):
        text = '''\
PROCESS: Adaptive PFR recycle
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
RECYCLE_METHOD: WEGSTEIN
COMPONENTS:
    C2H4O | Ethylene oxide | formula=C2H4O
    CH3CHO | Acetaldehyde | formula=C2H4O
REACTIONS:
    isomerization : C2H4O -> CH3CHO | A=2, Ea=0, Ea_unit=J/mol, rate_basis=concentration, concentration_unit=kmol/m3, pressure_unit=bar, rate_unit=kmol/m3/h
STREAM Feed : -> M.fresh
    T = 500 [K]
    P = 2 [bar]
    F = 10 [kmol/h]
    x = C2H4O:1
STREAM Recycle : SP.recycle -> M.recycle
STREAM Mixed : M.out -> PFR.in
STREAM Reacted : PFR.out -> SP.in
STREAM Product : SP.product
UNIT M : Mixer
    T_out = 500 [K]
    P = 2 [bar]
UNIT PFR : PFR
    volume = 0.1 [m3]
    mode = isothermal
    T = 500 [K]
    phase = vapor
    profile_points = 5
    REACTIONS:
        @isomerization
UNIT SP : Splitter
    outlets = product, recycle
    product_split_frac = 0.5
'''
        result = Simulator.from_string(text).run(max_iterations=100)

        self.assertTrue(result.converged, result.warnings)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.recycle_info['tear_streams'], ['Recycle'])
        self.assertLess(result.mass_balance_error, 1.0e-4)
        self.assertLess(result.energy_balance_error, 1.0e-4)
        self.assertLess(
            result.units['PFR'].performance[
                'maximum_material_balance_residual_kmol_h'
            ],
            1.0e-8,
        )

    def test_liquid_custom_activity_pfr_uses_compiled_backend_and_stability(self):
        thermo = create_thermodynamics(self.components, 'NRTL')
        feed = thermo.calculate_state(
            300.0,
            10.0,
            1.0,
            {'C2H4O': 0.5, 'CH3CHO': 0.5},
            phase='liquid',
            flash=False,
        )
        result = KineticsPFR('PFR-L', thermo, {
            'volume': 0.01,
            'mode': 'isothermal',
            'T': 300.0,
            'phase': 'liquid',
            'profile_points': 2,
            'reactions': [reaction_definition(
                type='custom',
                A=0.2,
                expression="k*a['C2H4O']",
            )],
        }).solve({'in': feed})

        self.assertGreater(
            result.performance['component_conversions']['C2H4O'],
            0.0,
        )
        self.assertEqual(
            result.performance['phase_stability'][
                'liquid_spinodal'
            ]['activity_backend'],
            'CompiledNRTLBackend',
        )
        self.assertFalse(
            result.performance['phase_stability'][
                'global_liquid_liquid_stability'
            ]['split']
        )


if __name__ == '__main__':
    unittest.main()
