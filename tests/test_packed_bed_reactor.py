import math
import unittest
from unittest.mock import patch

from dof_analyzer import SpecificationStatus, analyze_dof
from kinetic_models import HomogeneousRateState
from pfd_parser import PFDParser, validate_pfd
from pellet_models import (
    first_order_spherical_effectiveness,
    generalized_power_law_spherical_effectiveness,
    rigorous_power_law_network_spherical_effectiveness,
    rigorous_power_law_spherical_effectiveness,
)
from simulator import Simulator
from thermodynamics import create_thermodynamics
from unit_operations_base import UnitOperationError
from unit_operations_reactors import (
    KineticsBatch,
    KineticsCSTR,
    KineticsPFR,
    KineticsPackedBed,
)


def mass_rate_reaction(**overrides):
    definition = {
        'equation': 'C2H4O -> CH3CHO',
        'A': 0.02,
        'Ea': 0.0,
        'Ea_unit': 'J/mol',
        'rate_basis': 'concentration',
        'concentration_unit': 'kmol/m3',
        'pressure_unit': 'bar',
        'rate_unit': 'kmol/kg_cat/h',
    }
    definition.update(overrides)
    return definition


class PackedBedReactorUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thermo = create_thermodynamics(['C2H4O', 'CH3CHO'], 'IDEAL')
        cls.feed = cls.thermo.calculate_state(
            500.0,
            2.0,
            10.0,
            {'C2H4O': 1.0},
            phase='vapor',
            flash=False,
        )

    def solve(self, **extra):
        params = {
            'catalyst_mass': 100.0,
            'bulk_catalyst_density': 500.0,
            'bed_void_fraction': 0.4,
            'diameter': 0.5,
            'particle_diameter': 0.005,
            'phase': 'vapor',
            'mode': 'isothermal',
            'T': 500.0,
            'profile_points': 9,
            'reactions': [mass_rate_reaction()],
        }
        params.update(extra)
        return KineticsPackedBed('R-PBR', self.thermo, params).solve(
            {'in': self.feed}
        )

    def test_generalized_pellet_correlation_retains_exact_first_order_limit(self):
        for phi in (0.0, 1.0e-8, 0.1, 1.0, 10.0, 100.0):
            with self.subTest(phi=phi):
                exact = first_order_spherical_effectiveness(phi)
                generalized = generalized_power_law_spherical_effectiveness(
                    phi, 1.0
                )
                self.assertAlmostEqual(
                    generalized.effectiveness_factor, exact, places=14
                )
                self.assertAlmostEqual(
                    generalized.generalized_thiele_modulus, phi, places=14
                )
                self.assertEqual(
                    generalized.method, 'exact_first_order_sphere'
                )

    def test_rigorous_scalar_bvp_matches_exact_and_reference_solutions(self):
        first_order = rigorous_power_law_spherical_effectiveness(3.0, 1.0)
        self.assertAlmostEqual(
            first_order.effectiveness_factor,
            first_order_spherical_effectiveness(3.0),
            places=14,
        )
        second_order = rigorous_power_law_spherical_effectiveness(3.0, 2.0)
        fractional = rigorous_power_law_spherical_effectiveness(3.0, 0.5)
        self.assertAlmostEqual(
            second_order.effectiveness_factor, 0.57029448, places=6
        )
        self.assertAlmostEqual(
            fractional.effectiveness_factor, 0.76172919, places=6
        )
        self.assertGreater(second_order.radial_nodes, 2)
        self.assertGreater(second_order.center_concentration_ratio, 0.0)

    def test_rigorous_network_returns_reaction_specific_factors(self):
        result = rigorous_power_law_network_spherical_effectiveness(
            (4.0, 9.0),
            (1.0, 2.0),
        )
        self.assertEqual(len(result.effectiveness_factors), 2)
        self.assertGreater(
            result.effectiveness_factors[0],
            result.effectiveness_factors[1],
        )
        expected_overall = (
            4.0 * result.effectiveness_factors[0]
            + 9.0 * result.effectiveness_factors[1]
        ) / 13.0
        self.assertAlmostEqual(
            result.overall_limiting_species_effectiveness,
            expected_overall,
            places=14,
        )

    def test_apparent_mass_rate_matches_analytical_plug_flow_solution(self):
        result = self.solve()
        performance = result.performance
        density = self.thermo.mixture_molar_density(
            {'C2H4O': 1.0}, 500.0, 2.0, 1.0, y={'C2H4O': 1.0}
        )
        expected_conversion = 1.0 - math.exp(-0.02 * density * 100.0 / 10.0)

        self.assertAlmostEqual(
            performance['component_conversions']['C2H4O'],
            expected_conversion,
            places=10,
        )
        self.assertAlmostEqual(performance['catalyst_mass_kg'], 100.0)
        self.assertAlmostEqual(performance['volume_m3'], 0.2)
        self.assertAlmostEqual(
            performance['particle_density_kg_m3_particle'],
            500.0 / 0.6,
        )
        self.assertAlmostEqual(
            performance['profile'][-1]['catalyst_mass_kg'], 100.0
        )
        self.assertEqual(
            performance['profile'][-1]['reaction_measure_basis'],
            'catalyst_mass',
        )
        self.assertEqual(
            performance['reactions'][0]['rate_output_basis'],
            'catalyst_mass',
        )
        self.assertLess(
            performance['maximum_material_balance_residual_kmol_h'], 1.0e-12
        )
        self.assertAlmostEqual(result.outlet_streams['out'].F, self.feed.F)
        self.assertEqual(
            set(result.outlet_streams['out'].composition),
            {'C2H4O', 'CH3CHO'},
        )

    def test_geometry_can_derive_mass_and_rejects_inconsistent_redundancy(self):
        result = self.solve(catalyst_mass=None, length=1.0)
        expected_volume = math.pi * 0.5**2 / 4.0
        self.assertAlmostEqual(result.performance['volume_m3'], expected_volume)
        self.assertAlmostEqual(
            result.performance['catalyst_mass_kg'], 500.0 * expected_volume
        )

        with self.assertRaisesRegex(UnitOperationError, 'inconsistent'):
            self.solve(length=1.0)

        by_volume = self.solve(catalyst_mass=None, bed_volume=0.3)
        self.assertAlmostEqual(by_volume.performance['volume_m3'], 0.3)
        self.assertAlmostEqual(by_volume.performance['catalyst_mass_kg'], 150.0)
        parallel = self.solve(
            catalyst_mass=None,
            length=1.0,
            diameter=0.25,
            n_tubes=4,
        )
        self.assertAlmostEqual(parallel.performance['volume_m3'], expected_volume)

    def test_pfr_and_pbr_reject_each_others_rate_dimensions(self):
        with self.assertRaisesRegex(UnitOperationError, 'fluid-volume rate units'):
            KineticsPFR('R-PFR', self.thermo, {
                'volume': 1.0,
                'phase': 'vapor',
                'T': 500.0,
                'reactions': [mass_rate_reaction()],
            }).solve({'in': self.feed})

        with self.assertRaisesRegex(UnitOperationError, 'catalyst-mass rate units'):
            self.solve(reactions=[mass_rate_reaction(rate_unit='kmol/m3/h')])

        with self.assertRaisesRegex(UnitOperationError, 'fluid-volume rate units'):
            KineticsCSTR('R-CSTR', self.thermo, {
                'volume': 1.0,
                'phase': 'vapor',
                'T': 500.0,
                'reactions': [mass_rate_reaction()],
            }).solve({'in': self.feed})
        with self.assertRaisesRegex(UnitOperationError, 'fluid-volume rate units'):
            KineticsBatch('R-batch', self.thermo, {
                'V_batch': 1.0,
                't_rxn': 0.1,
                'phase': 'vapor',
                'T': 500.0,
                'reactions': [mass_rate_reaction()],
            }).solve({'charge': self.feed})

    def test_first_order_spherical_diffusion_reduces_intrinsic_rate(self):
        apparent = self.solve()
        diffusive = self.solve(
            kinetic_basis='intrinsic',
            diffusion_model='first_order_sphere',
            effective_diffusivity=1.0e-9,
        )
        eta = diffusive.performance['minimum_effectiveness_factor']
        phi = 0.0025 * math.sqrt(
            0.02 * (500.0 / 0.6) / (1.0e-9 * 3600.0)
        )
        expected_eta = 3.0 / phi * (
            1.0 / math.tanh(phi) - 1.0 / phi
        )

        self.assertGreater(eta, 0.0)
        self.assertLess(eta, 1.0)
        self.assertAlmostEqual(eta, expected_eta, places=12)
        self.assertAlmostEqual(
            eta,
            diffusive.performance['maximum_effectiveness_factor'],
            places=12,
        )
        self.assertLess(
            diffusive.performance['component_conversions']['C2H4O'],
            apparent.performance['component_conversions']['C2H4O'],
        )
        reaction = diffusive.performance['reactions'][0]
        self.assertAlmostEqual(reaction['outlet_effectiveness_factor'], eta)
        self.assertAlmostEqual(
            reaction['outlet_rate_kmol_kg_cat_h'],
            eta * reaction['outlet_intrinsic_rate_kmol_kg_cat_h'],
            places=12,
        )

        with self.assertRaisesRegex(UnitOperationError, 'first-order'):
            self.solve(
                kinetic_basis='intrinsic',
                diffusion_model='first_order_sphere',
                effective_diffusivity=1.0e-9,
                reactions=[mass_rate_reaction(order_C2H4O=2.0)],
            )

    def test_generalized_power_law_supports_arbitrary_positive_order(self):
        for order in (0.5, 2.0, 3.0):
            with self.subTest(order=order):
                result = self.solve(
                    kinetic_basis='intrinsic',
                    effective_diffusivity=1.0e-9,
                    reactions=[mass_rate_reaction(order_C2H4O=order)],
                )
                performance = result.performance
                eta = performance['minimum_effectiveness_factor']
                inlet_rate = (
                    performance['profile'][0][
                        'intrinsic_reaction_rates_kmol_kg_cat_h'
                    ][0]
                )
                concentration = 0.048108942017090414
                phi = 0.0025 * math.sqrt(
                    (500.0 / 0.6) * inlet_rate
                    / (1.0e-9 * 3600.0 * concentration)
                )
                generalized_phi = phi * math.sqrt((order + 1.0) / 2.0)
                expected_eta = 3.0 / generalized_phi * (
                    1.0 / math.tanh(generalized_phi)
                    - 1.0 / generalized_phi
                )

                self.assertEqual(
                    performance['diffusion_model'],
                    'generalized_power_law_sphere',
                )
                self.assertEqual(
                    performance['common_limiting_species_order'], order
                )
                self.assertAlmostEqual(eta, expected_eta, places=12)
                self.assertAlmostEqual(
                    performance['inlet_thiele_modulus'], phi, places=12
                )
                self.assertAlmostEqual(
                    performance['inlet_generalized_thiele_modulus'],
                    generalized_phi,
                    places=12,
                )
                self.assertAlmostEqual(
                    performance['inlet_apparent_order'], order, places=12
                )
                self.assertAlmostEqual(
                    eta, performance['maximum_effectiveness_factor'], places=12
                )

    def test_multiple_reactions_share_factor_only_for_common_order(self):
        compatible = self.solve(
            kinetic_basis='intrinsic',
            effective_diffusivity=1.0e-9,
            reactions=[
                mass_rate_reaction(A=0.01, order_C2H4O=2.0),
                mass_rate_reaction(A=0.02, order_C2H4O=2.0),
            ],
        )
        reactions = compatible.performance['reactions']
        self.assertEqual(len(reactions), 2)
        self.assertAlmostEqual(
            reactions[0]['outlet_effectiveness_factor'],
            reactions[1]['outlet_effectiveness_factor'],
            places=14,
        )
        self.assertAlmostEqual(
            reactions[1]['extent_kmol_h'] / reactions[0]['extent_kmol_h'],
            2.0,
            places=10,
        )

        with self.assertRaisesRegex(UnitOperationError, 'same order'):
            self.solve(
                kinetic_basis='intrinsic',
                effective_diffusivity=1.0e-9,
                reactions=[
                    mass_rate_reaction(A=0.01, order_C2H4O=1.0),
                    mass_rate_reaction(A=0.02, order_C2H4O=2.0),
                ],
            )

    def test_rigorous_mode_supports_different_orders_and_changes_selectivity(self):
        result = self.solve(
            catalyst_mass=20.0,
            kinetic_basis='intrinsic',
            diffusion_model='rigorous_power_law_sphere',
            effective_diffusivity=1.0e-9,
            reactions=[
                mass_rate_reaction(A=0.01, order_C2H4O=1.0),
                mass_rate_reaction(A=0.02, order_C2H4O=2.0),
            ],
        )
        reactions = result.performance['reactions']
        self.assertEqual(
            result.performance['diffusion_model'],
            'rigorous_power_law_sphere',
        )
        self.assertEqual(
            result.performance['effectiveness_factor_scope'],
            'rigorous_per_reaction_shared_limiting_species',
        )
        self.assertGreater(
            reactions[0]['outlet_effectiveness_factor'],
            reactions[1]['outlet_effectiveness_factor'],
        )
        self.assertGreater(result.performance['pellet_radial_nodes'], 2)
        self.assertEqual(
            result.performance['effectiveness_factor_resolves_actual'], 1
        )
        self.assertEqual(result.performance['pellet_maximum_nodes'], 2000)
        self.assertAlmostEqual(
            result.performance['pellet_relative_tolerance'], 1.0e-6
        )
        self.assertIsNone(result.performance['common_limiting_species_order'])
        self.assertLess(
            result.performance['inlet_center_concentration_ratio'], 1.0
        )

        with self.assertRaisesRegex(UnitOperationError, 'irreversible reactions'):
            self.solve(
                kinetic_basis='intrinsic',
                diffusion_model='rigorous_power_law_sphere',
                effective_diffusivity=1.0e-9,
                reactions=[mass_rate_reaction(equation='C2H4O <=> CH3CHO')],
            )

    def test_rigorous_sampled_interpolation_uses_requested_resolves(self):
        common = {
            'catalyst_mass': 50.0,
            'kinetic_basis': 'intrinsic',
            'diffusion_model': 'rigorous_power_law_sphere',
            'effective_diffusivity': 1.0e-9,
            'reactions': [mass_rate_reaction(A=0.2, order_C2H4O=2.0)],
        }
        inlet_only = self.solve(**common)
        sampled = self.solve(
            **common,
            effectiveness_factor_resolves=5,
        )
        performance = sampled.performance

        self.assertEqual(
            performance['effectiveness_factor_policy'],
            'sampled_interpolation',
        )
        self.assertEqual(
            performance['effectiveness_factor_resolves_requested'], 5
        )
        self.assertEqual(
            performance['effectiveness_factor_resolves_actual'], 5
        )
        self.assertEqual(
            len(performance['effectiveness_factor_support_positions_m']), 5
        )
        self.assertEqual(
            len(performance['effectiveness_factor_support_values']), 5
        )
        self.assertEqual(
            performance['effectiveness_factor_preliminary_axial_passes'], 1
        )
        support = [row[0] for row in performance['effectiveness_factor_support_values']]
        self.assertGreater(max(support), min(support))
        self.assertNotAlmostEqual(
            performance['component_conversions']['C2H4O'],
            inlet_only.performance['component_conversions']['C2H4O'],
            places=10,
        )

        for invalid in (0, 102):
            with self.subTest(effectiveness_factor_resolves=invalid):
                with self.assertRaisesRegex(UnitOperationError, 'between 1 and 101'):
                    self.solve(
                        **common,
                        effectiveness_factor_resolves=invalid,
                    )
        with self.assertRaisesRegex(UnitOperationError, 'only for rigorous'):
            self.solve(
                kinetic_basis='intrinsic',
                effective_diffusivity=1.0e-9,
                effectiveness_factor_resolves=2,
            )
        with self.assertRaisesRegex(UnitOperationError, 'calculated diffusion model'):
            self.solve(pellet_relative_tolerance=1.0e-7)
        with self.assertRaisesRegex(UnitOperationError, 'positive and finite'):
            self.solve(
                **common,
                pellet_relative_tolerance=0.0,
            )
        with self.assertRaisesRegex(UnitOperationError, 'at least 50'):
            self.solve(
                **common,
                pellet_maximum_nodes=49,
            )
        with self.assertRaisesRegex(UnitOperationError, 'constant_inlet'):
            self.solve(
                kinetic_basis='intrinsic',
                diffusion_model='rigorous_power_law_sphere',
                effective_diffusivity=1.0e-9,
                effectiveness_factor_policy='local',
            )

    def test_sampled_rigorous_network_interpolates_each_pathway_factor(self):
        result = self.solve(
            catalyst_mass=30.0,
            kinetic_basis='intrinsic',
            diffusion_model='rigorous_power_law_sphere',
            effective_diffusivity=1.0e-9,
            effectiveness_factor_resolves=3,
            reactions=[
                mass_rate_reaction(A=0.1, order_C2H4O=1.0),
                mass_rate_reaction(A=0.2, order_C2H4O=2.0),
            ],
        )
        supports = result.performance['effectiveness_factor_support_values']
        self.assertEqual(len(supports), 3)
        self.assertTrue(all(len(values) == 2 for values in supports))
        self.assertTrue(all(values[0] > values[1] for values in supports))
        self.assertGreater(
            max(values[0] for values in supports),
            min(values[0] for values in supports),
        )
        self.assertGreater(
            max(values[1] for values in supports),
            min(values[1] for values in supports),
        )
        self.assertEqual(
            result.performance['effectiveness_factor_resolves_actual'], 3
        )

    def test_rigorous_fractional_order_dead_core_is_rejected(self):
        with self.assertRaisesRegex(UnitOperationError, 'dead core'):
            self.solve(
                catalyst_mass=20.0,
                kinetic_basis='intrinsic',
                diffusion_model='rigorous_power_law_sphere',
                effective_diffusivity=1.0e-12,
                reactions=[mass_rate_reaction(order_C2H4O=0.5)],
            )

    def test_multiple_reactants_and_parallel_paths_can_share_one_limiter(self):
        thermo = create_thermodynamics(
            ['CO', 'H2', 'CH3OH', 'CH4', 'H2O'], 'IDEAL'
        )
        feed = thermo.calculate_state(
            600.0,
            20.0,
            10.0,
            {'CO': 0.2, 'H2': 0.8},
            phase='vapor',
            flash=False,
        )
        common = {
            'Ea': 0.0,
            'Ea_unit': 'J/mol',
            'rate_basis': 'concentration',
            'concentration_unit': 'kmol/m3',
            'pressure_unit': 'bar',
            'rate_unit': 'kmol/kg_cat/h',
            'order_CO': 1.0,
        }
        methanol = {
            **common,
            'equation': 'CO + 2 H2 -> CH3OH',
            'A': 0.01,
            'order_H2': 2.0,
        }
        methane = {
            **common,
            'equation': 'CO + 3 H2 -> CH4 + H2O',
            'A': 0.005,
            'order_H2': 3.0,
        }
        params = {
            'catalyst_mass': 10.0,
            'bulk_catalyst_density': 500.0,
            'bed_void_fraction': 0.4,
            'diameter': 0.5,
            'particle_diameter': 0.005,
            'phase': 'vapor',
            'mode': 'isothermal',
            'T': 600.0,
            'kinetic_basis': 'intrinsic',
            'effective_diffusivity': 1.0e-9,
            'reactions': [methanol, methane],
        }
        with self.assertRaisesRegex(UnitOperationError, 'cannot infer one'):
            KineticsPackedBed('R-ambiguous', thermo, params).solve({'in': feed})

        params['diffusion_limiting_component'] = 'CO'
        result = KineticsPackedBed('R-parallel', thermo, params).solve(
            {'in': feed}
        )
        reactions = result.performance['reactions']
        self.assertEqual(
            result.performance['diffusion_limiting_component'], 'CO'
        )
        self.assertGreater(result.outlet_streams['out'].composition['CH3OH'], 0.0)
        self.assertGreater(result.outlet_streams['out'].composition['CH4'], 0.0)
        self.assertAlmostEqual(
            reactions[0]['outlet_effectiveness_factor'],
            reactions[1]['outlet_effectiveness_factor'],
            places=14,
        )

    def test_constant_inlet_and_local_factor_policies_are_distinct(self):
        common = {
            'kinetic_basis': 'intrinsic',
            'effective_diffusivity': 1.0e-9,
            'reactions': [mass_rate_reaction(A=0.2, order_C2H4O=2.0)],
        }
        constant = self.solve(**common)
        local = self.solve(
            **common,
            effectiveness_factor_policy='local',
        )

        self.assertEqual(
            constant.performance['effectiveness_factor_policy'],
            'constant_inlet',
        )
        self.assertAlmostEqual(
            constant.performance['minimum_effectiveness_factor'],
            constant.performance['maximum_effectiveness_factor'],
            places=14,
        )
        self.assertGreater(
            local.performance['maximum_effectiveness_factor'],
            local.performance['minimum_effectiveness_factor'],
        )
        self.assertLess(
            local.performance['minimum_thiele_modulus'],
            local.performance['maximum_thiele_modulus'],
        )
        self.assertGreater(
            local.performance['component_conversions']['C2H4O'],
            constant.performance['component_conversions']['C2H4O'],
        )

    def test_single_reversible_reaction_uses_local_apparent_order(self):
        result = self.solve(
            catalyst_mass=10.0,
            kinetic_basis='intrinsic',
            effective_diffusivity=1.0e-9,
            reactions=[mass_rate_reaction(equation='C2H4O <=> CH3CHO')],
        )
        self.assertEqual(
            result.performance['effectiveness_factor_scope'],
            'single_reversible_reaction_local_apparent_order',
        )
        self.assertTrue(result.performance['reactions'][0]['reversible'])
        self.assertGreater(
            result.performance['reactions'][0]['extent_kmol_h'], 0.0
        )

        with self.assertRaisesRegex(UnitOperationError, 'multiple reactions'):
            self.solve(
                catalyst_mass=10.0,
                kinetic_basis='intrinsic',
                effective_diffusivity=1.0e-9,
                reactions=[
                    mass_rate_reaction(equation='C2H4O <=> CH3CHO'),
                    mass_rate_reaction(A=0.01),
                ],
            )

    def test_ergun_pressure_drop_and_void_residence_time_are_reported(self):
        result = self.solve(
            catalyst_mass=10.0,
            pressure_drop_model='ergun',
        )
        performance = result.performance
        self.assertGreater(performance['pressure_drop_bar'], 0.0)
        self.assertLess(performance['P_out_bar'], performance['P_in_bar'])
        self.assertGreater(performance['inlet_reynolds_number'], 0.0)
        self.assertAlmostEqual(
            performance['fluid_void_volume_m3'],
            performance['volume_m3'] * 0.4,
        )

    def test_adiabatic_bed_closes_full_enthalpy_balance(self):
        result = self.solve(mode='adiabatic', T=None)
        self.assertLess(
            abs(result.performance['energy_balance_residual_kW']), 1.0e-8
        )
        inlet_enthalpy = self.feed.F * self.feed.H
        outlet = result.outlet_streams['out']
        self.assertAlmostEqual(
            outlet.F * outlet.H - inlet_enthalpy,
            result.heat_duty,
            places=7,
        )

    def test_duty_and_jacketed_modes_share_exact_axial_energy_balance(self):
        duty = self.solve(mode='duty', T=None, Q=1.0, __unit__Q='kW')
        jacketed = self.solve(
            mode='jacketed',
            T=None,
            U=100.0,
            __unit__U='W/m2/K',
            T_jacket=450.0,
        )

        self.assertAlmostEqual(duty.heat_duty, 3600.0, places=8)
        self.assertLess(jacketed.heat_duty, 0.0)
        self.assertLess(jacketed.outlet_streams['out'].T, self.feed.T)
        self.assertGreater(jacketed.outlet_streams['out'].T, 450.0)
        for result in (duty, jacketed):
            self.assertLess(
                abs(result.performance['energy_balance_residual_kW']), 1.0e-8
            )
            self.assertEqual(result.performance['solver_method'], 'LSODA')

    def test_none_specified_and_ergun_pressure_models_are_distinct(self):
        no_drop = self.solve(catalyst_mass=10.0)
        specified = self.solve(catalyst_mass=10.0, P_drop=0.1)
        ergun = self.solve(
            catalyst_mass=10.0,
            pressure_drop_model='ergun',
        )

        self.assertAlmostEqual(no_drop.performance['pressure_drop_bar'], 0.0)
        self.assertAlmostEqual(specified.performance['pressure_drop_bar'], 0.1)
        self.assertGreater(ergun.performance['pressure_drop_bar'], 0.0)
        with self.assertRaisesRegex(UnitOperationError, 'does not support Darcy'):
            self.solve(
                catalyst_mass=10.0,
                pressure_drop_model='darcy',
                roughness=1.0e-5,
            )

    def test_viscosity_is_requested_only_for_ergun_hydraulics(self):
        with patch.object(
            self.thermo,
            'mixture_viscosity',
            side_effect=AssertionError('viscosity should not be requested'),
        ):
            self.solve(catalyst_mass=10.0)
        with patch.object(
            self.thermo,
            'mixture_viscosity',
            wraps=self.thermo.mixture_viscosity,
        ) as viscosity:
            self.solve(catalyst_mass=10.0, pressure_drop_model='ergun')
        self.assertGreater(viscosity.call_count, 0)

    def test_intrinsic_and_specified_effectiveness_factor_contracts(self):
        intrinsic = self.solve(kinetic_basis='intrinsic')
        specified = self.solve(
            kinetic_basis='intrinsic',
            effectiveness_factor=0.5,
        )
        apparent = self.solve()

        self.assertAlmostEqual(intrinsic.performance['minimum_effectiveness_factor'], 1.0)
        self.assertAlmostEqual(specified.performance['minimum_effectiveness_factor'], 0.5)
        self.assertAlmostEqual(
            intrinsic.performance['component_conversions']['C2H4O'],
            apparent.performance['component_conversions']['C2H4O'],
            places=11,
        )
        self.assertLess(
            specified.performance['component_conversions']['C2H4O'],
            intrinsic.performance['component_conversions']['C2H4O'],
        )
        with self.assertRaisesRegex(UnitOperationError, 'apparent kinetics'):
            self.solve(effectiveness_factor=0.5)
        for invalid in (0.0, 1.1):
            with self.subTest(effectiveness_factor=invalid):
                with self.assertRaisesRegex(UnitOperationError, 'must lie in'):
                    self.solve(
                        kinetic_basis='intrinsic',
                        effectiveness_factor=invalid,
                    )

    def test_calculated_diffusion_rejects_every_unsupported_contract(self):
        base = {
            'kinetic_basis': 'intrinsic',
            'effective_diffusivity': 1.0e-9,
        }
        cases = (
            ({'kinetic_basis': 'apparent'}, 'apparent kinetics'),
            ({'diffusion_model': 'unknown'}, 'diffusion_model must be'),
            ({'reactions': [mass_rate_reaction(type='custom', expression="k*C['C2H4O']")]}, 'requires power-law'),
            ({'reactions': [mass_rate_reaction(equation='C2H4O <=> CH3CHO'), mass_rate_reaction()]}, 'multiple reactions'),
            ({'diffusion_limiting_component': 'CH3CHO'}, 'must be consumed'),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                params = dict(base)
                params.update(changes)
                with self.assertRaisesRegex(UnitOperationError, message):
                    self.solve(**params)

        with self.assertRaisesRegex(UnitOperationError, 'cannot also specify'):
            self.solve(
                kinetic_basis='intrinsic',
                diffusion_model='specified',
                effectiveness_factor=0.5,
                effective_diffusivity=1.0e-9,
            )
        with self.assertRaisesRegex(UnitOperationError, 'cannot be combined'):
            self.solve(
                kinetic_basis='intrinsic',
                diffusion_model='generalized_power_law_sphere',
                effectiveness_factor=0.5,
                effective_diffusivity=1.0e-9,
            )
        with self.assertRaisesRegex(UnitOperationError, 'requires effectiveness_factor'):
            self.solve(
                kinetic_basis='intrinsic',
                diffusion_model='specified',
            )
        with self.assertRaisesRegex(UnitOperationError, 'requires effective_diffusivity'):
            self.solve(
                kinetic_basis='intrinsic',
                diffusion_model='first_order_sphere',
            )
        with self.assertRaisesRegex(UnitOperationError, 'require a calculated'):
            self.solve(diffusion_limiting_component='C2H4O')
        with self.assertRaisesRegex(UnitOperationError, 'constant throughout'):
            self.solve(
                kinetic_basis='intrinsic',
                effectiveness_factor=0.5,
                effectiveness_factor_policy='local',
            )

    def test_diffusivity_units_and_pellet_radius_alias_are_consistent(self):
        by_seconds = self.solve(
            kinetic_basis='intrinsic',
            effective_diffusivity=1.0e-9,
        )
        by_hours = self.solve(
            kinetic_basis='intrinsic',
            effective_diffusivity=3.6e-6,
            __unit__effective_diffusivity='m2/h',
        )
        by_radius = self.solve(
            kinetic_basis='intrinsic',
            effective_diffusivity=1.0e-9,
            particle_diameter=None,
            pellet_radius=0.0025,
        )
        for result in (by_hours, by_radius):
            self.assertAlmostEqual(
                result.performance['component_conversions']['C2H4O'],
                by_seconds.performance['component_conversions']['C2H4O'],
                places=11,
            )
        with self.assertRaisesRegex(UnitOperationError, 'inconsistent'):
            self.solve(
                kinetic_basis='intrinsic',
                effective_diffusivity=1.0e-9,
                pellet_radius=0.001,
            )

    def test_mass_rate_unit_conversion_custom_net_and_multiple_reactions(self):
        canonical = self.solve()
        converted = self.solve(reactions=[mass_rate_reaction(
            A=0.02 / 3.6,
            rate_unit='mol/kg_cat/s',
        )])
        self.assertAlmostEqual(
            converted.performance['component_conversions']['C2H4O'],
            canonical.performance['component_conversions']['C2H4O'],
            places=11,
        )
        custom = self.solve(reactions=[mass_rate_reaction(
            type='custom',
            expression="k*C['C2H4O']",
        )])
        self.assertAlmostEqual(
            custom.performance['component_conversions']['C2H4O'],
            canonical.performance['component_conversions']['C2H4O'],
            places=9,
        )

        multiple = self.solve(reactions=[
            mass_rate_reaction(A=0.01),
            mass_rate_reaction(A=0.02),
        ])
        extents = [row['extent_kmol_h'] for row in multiple.performance['reactions']]
        self.assertEqual(len(extents), 2)
        self.assertGreater(extents[0], 0.0)
        self.assertGreater(extents[1], extents[0])

        product_feed = self.thermo.calculate_state(
            500.0, 2.0, 10.0, {'CH3CHO': 1.0}, phase='vapor', flash=False
        )
        reverse = KineticsPackedBed('R-reverse', self.thermo, {
            'catalyst_mass': 100.0,
            'bulk_catalyst_density': 500.0,
            'bed_void_fraction': 0.4,
            'diameter': 0.5,
            'phase': 'vapor',
            'T': 500.0,
            'reactions': [mass_rate_reaction(
                type='custom_net',
                expression="-k*C['CH3CHO']",
            )],
        }).solve({'in': product_feed})
        self.assertLess(reverse.performance['reactions'][0]['extent_kmol_h'], 0.0)
        self.assertGreater(
            reverse.outlet_streams['out'].composition['C2H4O'], 0.0
        )

    def test_pressure_fugacity_and_activity_power_law_bases_are_supported(self):
        for basis in ('partial_pressure', 'fugacity', 'activity'):
            with self.subTest(rate_basis=basis):
                result = self.solve(
                    catalyst_mass=10.0,
                    reactions=[mass_rate_reaction(
                        A=0.001,
                        rate_basis=basis,
                    )],
                )
                self.assertGreater(
                    result.performance['component_conversions']['C2H4O'], 0.0
                )

    def test_reversible_rate_uses_thermodynamic_q_over_k_correction(self):
        reaction = mass_rate_reaction(equation='C2H4O <=> CH3CHO')
        reactant_rich = self.solve(
            catalyst_mass=10.0,
            reactions=[reaction],
        )
        mixed_feed = self.thermo.calculate_state(
            500.0,
            2.0,
            10.0,
            {'C2H4O': 0.2, 'CH3CHO': 0.8},
            phase='vapor',
            flash=False,
        )
        product_rich = KineticsPackedBed('R-reversible', self.thermo, {
            'catalyst_mass': 10.0,
            'bulk_catalyst_density': 500.0,
            'bed_void_fraction': 0.4,
            'diameter': 0.5,
            'phase': 'vapor',
            'T': 500.0,
            'reactions': [reaction],
        }).solve({'in': mixed_feed})

        self.assertTrue(reactant_rich.performance['reactions'][0]['reversible'])
        self.assertGreater(
            reactant_rich.performance['reactions'][0]['extent_kmol_h'],
            product_rich.performance['reactions'][0]['extent_kmol_h'],
        )

    def test_homogeneous_liquid_bed_uses_liquid_density_and_stability(self):
        feed = self.thermo.calculate_state(
            250.0,
            10.0,
            10.0,
            {'C2H4O': 1.0},
            phase='liquid',
            flash=False,
        )
        result = KineticsPackedBed('R-liquid', self.thermo, {
            'catalyst_mass': 10.0,
            'bulk_catalyst_density': 500.0,
            'bed_void_fraction': 0.4,
            'diameter': 0.5,
            'phase': 'liquid',
            'T': 250.0,
            'profile_points': 5,
            'reactions': [mass_rate_reaction()],
        }).solve({'in': feed})

        self.assertGreater(
            result.performance['component_conversions']['C2H4O'], 0.0
        )
        self.assertEqual(
            result.performance['phase_stability']['unconstrained_vapor_fraction'],
            0.0,
        )
        self.assertTrue(
            result.performance['phase_stability'][
                'global_liquid_liquid_stability_checked'
            ]
        )

        unstable_feed = self.thermo.calculate_state(
            250.0,
            10.0,
            10.0,
            {'C2H4O': 1.0},
            phase='vapor',
            flash=False,
        )
        with self.assertRaisesRegex(UnitOperationError, 'phase-unstable'):
            KineticsPackedBed('R-unstable', self.thermo, {
                'catalyst_mass': 1.0,
                'bulk_catalyst_density': 500.0,
                'bed_void_fraction': 0.4,
                'diameter': 0.5,
                'phase': 'vapor',
                'T': 250.0,
                'reactions': [mass_rate_reaction(A=1.0e-6)],
            }).solve({'in': unstable_feed})

    def test_vdm_vapor_density_preserves_nominal_kinetics_and_warns_on_viscosity_use(self):
        thermo = create_thermodynamics(
            ['acetic acid', 'acetone', 'CO2', 'water'], 'UNIQUAC-VDM'
        )
        composition = {
            'acetic acid': 0.5,
            'acetone': 0.0,
            'CO2': 0.0,
            'water': 0.5,
        }
        feed = thermo.calculate_state(
            400.0,
            1.0,
            10.0,
            composition,
            phase='vapor',
            flash=False,
        )
        _acid, model = thermo._active_vdm_model(composition)
        association = model.association_state(400.0, 1.0, composition, rk_model=None)
        physical_factor = thermo._vdm_physical_moles_per_nominal(association)
        ideal = create_thermodynamics(list(composition), 'IDEAL')
        ideal_density = ideal.mixture_molar_density(
            composition, 400.0, 1.0, 1.0, y=composition
        )
        self.assertAlmostEqual(
            thermo.mixture_molar_density(
                composition, 400.0, 1.0, 1.0, y=composition
            ),
            ideal_density / physical_factor,
            places=12,
        )

        rate_state = HomogeneousRateState.from_flows(
            thermo,
            {'acetic acid': 5.0, 'water': 5.0},
            400.0,
            1.0,
            'vapor',
        )
        self.assertAlmostEqual(rate_state.partial_pressures_bar['acetic acid'], 0.5)
        self.assertAlmostEqual(
            rate_state.fugacities_bar['acetic acid'],
            0.5 * association['phi_total']['acetic acid'],
            places=12,
        )

        params = {
            'catalyst_mass': 10.0,
            'bulk_catalyst_density': 500.0,
            'bed_void_fraction': 0.4,
            'diameter': 0.5,
            'particle_diameter': 0.005,
            'phase': 'vapor',
            'mode': 'isothermal',
            'T': 400.0,
            'reactions': [{
                'equation': '2 acetic acid -> acetone + CO2 + water',
                'A': 0.01,
                'Ea': 0.0,
                'Ea_unit': 'J/mol',
                'rate_basis': 'concentration',
                'concentration_unit': 'kmol/m3',
                'pressure_unit': 'bar',
                'rate_unit': 'kmol/kg_cat/h',
            }],
        }
        thermo.warnings.clear()
        no_hydraulics = KineticsPackedBed(
            'R-vdm-none', thermo, params
        ).solve({'in': feed})
        self.assertGreater(
            no_hydraulics.performance['component_conversions']['acetic acid'],
            0.0,
        )
        self.assertFalse(any('VDM vapor viscosity' in item for item in thermo.warnings))

        ergun = KineticsPackedBed(
            'R-vdm-ergun',
            thermo,
            {**params, 'pressure_drop_model': 'ergun'},
        ).solve({'in': feed})
        self.assertGreater(ergun.performance['pressure_drop_bar'], 0.0)
        self.assertEqual(
            sum('VDM vapor viscosity' in item for item in thermo.warnings),
            1,
        )

    def test_component_depletion_reports_inactive_remaining_catalyst(self):
        result = self.solve(reactions=[mass_rate_reaction(
            A=100.0,
            order_C2H4O=0.0,
        )], pressure_drop_model='ergun')
        performance = result.performance

        self.assertAlmostEqual(
            performance['component_conversions']['C2H4O'],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            performance['profile'][-1]['position_m'],
            performance['length_m'],
            places=12,
        )
        self.assertEqual(performance['profile'][-1]['reaction_rates'], [0.0])
        event_coordinate = performance['reaction_inactive_from_coordinate']
        event_row = min(
            performance['profile'],
            key=lambda row: abs(row['position_m'] - event_coordinate),
        )
        self.assertLess(performance['P_out_bar'], event_row['pressure_bar'])
        self.assertGreater(performance['pressure_drop_bar'], 0.0)
        self.assertAlmostEqual(
            performance['inactive_remaining_catalyst_mass_kg'],
            99.9,
            places=6,
        )
        self.assertTrue(any(
            'all reactions are inactive' in warning
            for warning in result.warnings
        ))

    def test_recycle_context_reduces_and_restores_profile_diagnostics(self):
        unit = KineticsPackedBed('R-profile', self.thermo, {
            'catalyst_mass': 100.0,
            'bulk_catalyst_density': 500.0,
            'bed_void_fraction': 0.4,
            'diameter': 0.5,
            'phase': 'vapor',
            'mode': 'adiabatic',
            'profile_points': 11,
            'reactions': [mass_rate_reaction()],
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
            final = unit.solve({'in': self.feed})
        finally:
            unit.solve_context = {}

        self.assertFalse(reduced.performance['detailed_diagnostics'])
        self.assertEqual(reduced.performance['profile_points'], 2)
        self.assertEqual(scheduled.performance['profile_points'], 11)
        self.assertEqual(final.performance['profile_points'], 11)
        self.assertAlmostEqual(
            reduced.outlet_streams['out'].T,
            final.outlet_streams['out'].T,
            places=8,
        )

    def test_repeated_solve_and_explicit_integrators_are_deterministic(self):
        unit = KineticsPackedBed('R-repeat', self.thermo, {
            'catalyst_mass': 100.0,
            'bulk_catalyst_density': 500.0,
            'bed_void_fraction': 0.4,
            'diameter': 0.5,
            'phase': 'vapor',
            'mode': 'adiabatic',
            'profile_points': 7,
            'reactions': [mass_rate_reaction()],
        })
        first = unit.solve({'in': self.feed})
        second = unit.solve({'in': self.feed})
        self.assertEqual(
            first.outlet_streams['out'].to_dict(),
            second.outlet_streams['out'].to_dict(),
        )
        self.assertEqual(first.performance, second.performance)

        outlets = []
        for method in ('RK45', 'BDF', 'Radau', 'LSODA'):
            with self.subTest(method=method):
                result = self.solve(solver=method, profile_points=5)
                self.assertEqual(result.performance['solver_method'], method)
                self.assertLess(
                    result.performance['maximum_material_balance_residual_kmol_h'],
                    1.0e-9,
                )
                outlets.append(
                    result.performance['component_conversions']['C2H4O']
                )
        self.assertLess(max(outlets) - min(outlets), 1.0e-8)

    def test_duplicate_aliases_and_missing_geometry_are_rejected(self):
        with self.assertRaisesRegex(UnitOperationError, 'duplicate aliases'):
            self.solve(void_fraction=0.4)
        with self.assertRaisesRegex(UnitOperationError, 'requires bed diameter'):
            self.solve(diameter=None)
        with self.assertRaisesRegex(UnitOperationError, 'requires length'):
            self.solve(catalyst_mass=None)
        with self.assertRaisesRegex(UnitOperationError, 'bulk_catalyst_density'):
            self.solve(bulk_catalyst_density=None)
        with self.assertRaisesRegex(UnitOperationError, 'bed_void_fraction'):
            self.solve(bed_void_fraction=None)
        with self.assertRaisesRegex(UnitOperationError, 'requires exactly one inlet'):
            KineticsPackedBed('R-empty', self.thermo, {
                'catalyst_mass': 1.0,
                'bulk_catalyst_density': 500.0,
                'bed_void_fraction': 0.4,
                'diameter': 0.5,
                'phase': 'vapor',
                'T': 500.0,
                'reactions': [mass_rate_reaction()],
            }).solve({})
        with self.assertRaisesRegex(UnitOperationError, 'at least one kinetic reaction'):
            self.solve(reactions=[])
        with self.assertRaisesRegex(UnitOperationError, 'does not recognize'):
            self.solve(
                kinetic_basis='intrinsic',
                effective_diffusivity=1.0,
                __unit__effective_diffusivity='furlong2/fortnight',
            )
        with self.assertRaisesRegex(UnitOperationError, 'requires pellet_radius'):
            self.solve(
                kinetic_basis='intrinsic',
                effective_diffusivity=1.0e-9,
                particle_diameter=None,
            )


class PackedBedReactorPFDTests(unittest.TestCase):
    PFD = """
PROCESS: packed bed example
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL

COMPONENTS:
    C2H4O  | Ethylene Oxide | formula=C2H4O, MW=44.053
    CH3CHO | Acetaldehyde   | formula=C2H4O, MW=44.053

REACTIONS:
    packed_isomerization : C2H4O -> CH3CHO | A=0.02, Ea=0, Ea_unit=J/mol, rate_basis=concentration, concentration_unit=kmol/m3, pressure_unit=bar, rate_unit=kmol/kg_cat/h

STREAM Feed : FEED -> R-101.in
    T = 500 [K]
    P = 2 [bar]
    F = 10 [kmol/h]
    x = C2H4O:1

STREAM Product : R-101.out -> PRODUCT

UNIT R-101 : PBR
    catalyst_mass = 100 [kg]
    bulk_catalyst_density = 500 [kg/m3]
    bed_void_fraction = 0.4
    diameter = 0.5 [m]
    particle_diameter = 5 [mm]
    phase = vapor
    mode = isothermal
    T = 500 [K]
    profile_points = 7
    REACTIONS:
        @packed_isomerization
"""

    def test_compact_pbr_validates_round_trips_and_solves(self):
        pfd = PFDParser().parse(self.PFD)
        self.assertEqual(pfd.units[0].unit_type, 'PackedBedReactor')
        self.assertEqual(validate_pfd(pfd), ([], []))
        restored = type(pfd).from_dict(pfd.to_dict())
        self.assertEqual(validate_pfd(restored), ([], []))
        dof = analyze_dof(restored)
        unit_dof = next(
            item for item in dof.unit_results if item.entity_id == 'R-101'
        )
        self.assertEqual(unit_dof.status, SpecificationStatus.OK)

        result = Simulator(restored).run()
        self.assertTrue(result.converged, result.errors)
        self.assertGreater(
            result.units['R-101'].performance['component_conversions']['C2H4O'],
            0.0,
        )

        for alias in ('PackedBedReactor', 'KineticsPackedBed'):
            with self.subTest(alias=alias):
                aliased = PFDParser().parse(
                    self.PFD.replace('UNIT R-101 : PBR', f'UNIT R-101 : {alias}')
                )
                self.assertEqual(
                    aliased.units[0].unit_type, 'PackedBedReactor'
                )
                self.assertEqual(validate_pfd(aliased), ([], []))

    def test_pbr_dof_reports_missing_bed_specs(self):
        pfd = PFDParser().parse(
            self.PFD.replace('    catalyst_mass = 100 [kg]\n', '')
        )
        result = analyze_dof(pfd)
        unit_dof = next(
            item for item in result.unit_results if item.entity_id == 'R-101'
        )
        self.assertEqual(unit_dof.status, SpecificationStatus.UNDER_SPECIFIED)
        self.assertIn('length, bed_volume, or catalyst_mass', unit_dof.message)

    def test_parser_rejects_wrong_rate_dimension_for_each_axial_reactor(self):
        wrong_pbr = PFDParser().parse(
            self.PFD.replace('rate_unit=kmol/kg_cat/h', 'rate_unit=kmol/m3/h')
        )
        errors, _warnings = validate_pfd(wrong_pbr)
        self.assertTrue(any(
            'PackedBedReactor requires catalyst-mass rate units' in error
            for error in errors
        ))

        wrong_pfr = PFDParser().parse(
            self.PFD.replace('UNIT R-101 : PBR', 'UNIT R-101 : PFR')
        )
        errors, _warnings = validate_pfd(wrong_pfr)
        self.assertTrue(any(
            'PFR requires fluid-volume rate units' in error
            for error in errors
        ))

    def test_packed_bed_converges_inside_continuous_recycle(self):
        text = """
PROCESS: packed bed recycle
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
RECYCLE_METHOD: BROYDEN
COMPONENTS:
    C2H4O | Ethylene oxide | formula=C2H4O
    CH3CHO | Acetaldehyde | formula=C2H4O
STREAM Fresh : FEED -> M.fresh
    T = 500 [K]
    P = 2 [bar]
    F = 10 [kmol/h]
    x = C2H4O:1
STREAM Recycle : S.recycle -> M.recycle
STREAM Mixed : M.out -> R.in
STREAM Reacted : R.out -> S.in
STREAM Product : S.product -> PRODUCT
UNIT M : Mixer
UNIT R : PackedBedReactor
    catalyst_mass = 50 [kg]
    bulk_catalyst_density = 500 [kg/m3]
    bed_void_fraction = 0.4
    diameter = 0.5 [m]
    phase = vapor
    mode = isothermal
    T = 500 [K]
    profile_points = 7
    REACTIONS:
        C2H4O -> CH3CHO | A=0.02, Ea=0, Ea_unit=J/mol, rate_basis=concentration, concentration_unit=kmol/m3, pressure_unit=bar, rate_unit=kmol/kg_cat/h
UNIT S : Splitter
    outlets = product,recycle
    product_split_frac = 0.5
"""
        result = Simulator.from_string(text).run(max_iterations=50)
        self.assertTrue(result.converged, result.errors)
        self.assertEqual(result.recycle_info['tear_streams'], ['Recycle'])
        self.assertLess(result.mass_balance_error, 1.0e-4)
        self.assertLess(result.energy_balance_error, 1.0e-4)
        self.assertEqual(result.units['R'].performance['profile_points'], 7)


if __name__ == '__main__':
    unittest.main()
