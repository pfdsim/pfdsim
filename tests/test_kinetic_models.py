import math
import unittest
from copy import deepcopy

from kinetic_models import (
    HomogeneousRateState,
    KineticsError,
    evaluate_forward_reaction_rate,
    evaluate_reaction_rate,
    kinetic_reaction_from_mapping,
    solve_isothermal_cstr,
)
from pfd_parser import PFDParser, validate_pfd
from reaction_models import ReactionDefinitionError
from thermodynamics import create_thermodynamics
from unit_operations_base import UnitOperationError
from unit_operations_reactors import KineticsCSTR, KineticsPFR


def kinetic_definition(equation, **overrides):
    definition = {
        'equation': equation,
        'A': 1.0,
        'Ea': 0.0,
        'Ea_unit': 'J/mol',
        'rate_basis': 'concentration',
        'concentration_unit': 'kmol/m3',
        'pressure_unit': 'bar',
        'rate_unit': 'kmol/m3/h',
    }
    definition.update(overrides)
    return definition


class KineticReactionDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.components = ['C2H4O', 'CH3CHO']
        self.thermo = create_thermodynamics(self.components, 'IDEAL')

    def test_bare_legacy_parameters_are_rejected(self):
        with self.assertRaisesRegex(ReactionDefinitionError, 'Ea_unit'):
            kinetic_reaction_from_mapping(
                {'equation': 'C2H4O -> CH3CHO', 'A': 1, 'Ea': 0},
                self.components,
                self.thermo.props,
            )

    def test_power_law_units_are_invariant_after_normalization(self):
        state = HomogeneousRateState.from_flows(
            self.thermo,
            {'C2H4O': 1.0, 'CH3CHO': 1.0},
            500.0,
            2.0,
            'vapor',
        )
        hourly = kinetic_reaction_from_mapping(
            kinetic_definition('C2H4O -> CH3CHO', A=2.0),
            self.components,
            self.thermo.props,
        )
        per_second = kinetic_reaction_from_mapping(
            kinetic_definition(
                'C2H4O -> CH3CHO',
                A=2.0 / 3600.0,
                concentration_unit='mol/m3',
                rate_unit='mol/m3/s',
            ),
            self.components,
            self.thermo.props,
        )

        self.assertAlmostEqual(
            evaluate_reaction_rate(hourly, state, self.thermo),
            evaluate_reaction_rate(per_second, state, self.thermo),
            places=13,
        )

    def test_power_law_accepts_all_thermodynamic_rate_bases(self):
        state = HomogeneousRateState.from_flows(
            self.thermo,
            {'C2H4O': 0.6, 'CH3CHO': 0.4},
            500.0,
            2.0,
            'vapor',
        )
        for basis in ('concentration', 'partial_pressure', 'fugacity', 'activity'):
            with self.subTest(basis=basis):
                reaction = kinetic_reaction_from_mapping(
                    kinetic_definition('C2H4O -> CH3CHO', rate_basis=basis),
                    self.components,
                    self.thermo.props,
                )
                self.assertGreater(
                    evaluate_reaction_rate(reaction, state, self.thermo),
                    0.0,
                )

    def test_rate_state_uses_eos_density_and_component_fugacity_wrappers(self):
        components = ['CO', 'H2', 'CH3OH']
        composition = {'CO': 0.3, 'H2': 0.6, 'CH3OH': 0.1}
        flows = {component: fraction * 10.0 for component, fraction in composition.items()}
        for method in ('PR', 'RKS-BM', 'PSRK'):
            with self.subTest(method=method):
                thermo = create_thermodynamics(components, method)
                state = HomogeneousRateState.from_flows(
                    thermo, flows, 500.0, 80.0, 'vapor'
                )
                phi = thermo.fugacity_coefficients(
                    500.0, 80.0, composition, 'vapor'
                )
                expected_density = 1.0 / thermo.vapor_molar_volume_for_density(
                    500.0, 80.0, composition
                )
                self.assertAlmostEqual(
                    state.molar_density_kmol_m3,
                    expected_density,
                    places=12,
                )
                for component, fraction in composition.items():
                    self.assertAlmostEqual(
                        state.fugacities_bar[component],
                        fraction * phi[component] * 80.0,
                        places=11,
                    )

    def test_liquid_rate_state_uses_activity_model_and_resolver_density(self):
        components = ['ethanol', 'water']
        thermo = create_thermodynamics(components, 'NRTL-PR')
        composition = {'ethanol': 0.4, 'water': 0.6}
        state = HomogeneousRateState.from_flows(
            thermo,
            {'ethanol': 4.0, 'water': 6.0},
            350.0,
            5.0,
            'liquid',
        )
        expected_density = thermo.mixture_molar_density(
            composition,
            350.0,
            5.0,
            0.0,
            x=composition,
        )
        expected_activities = thermo.component_activities(
            350.0,
            5.0,
            composition,
            'liquid',
        )
        self.assertAlmostEqual(state.molar_density_kmol_m3, expected_density, places=12)
        for component in components:
            self.assertAlmostEqual(
                state.activities[component],
                expected_activities[component],
                places=12,
            )

    def test_custom_expression_is_safe_and_can_mix_declared_variables(self):
        reaction = kinetic_reaction_from_mapping(
            kinetic_definition(
                'C2H4O -> CH3CHO',
                type='custom',
                expression="k * C['C2H4O']**0.5 * exp(-B/T) / (1 + log(L))",
                param_B=20.0,
                param_L=2.0,
            ),
            self.components,
            self.thermo.props,
        )
        state = HomogeneousRateState.from_flows(
            self.thermo,
            {'C2H4O': 1.0, 'CH3CHO': 1.0},
            500.0,
            2.0,
            'vapor',
        )
        self.assertGreater(evaluate_reaction_rate(reaction, state, self.thermo), 0.0)

        for expression in (
            "__import__('os').system('false')",
            "C.get('C2H4O')",
            "[value for value in C]",
            "abs(C['C2H4O'])",
        ):
            with self.subTest(expression=expression):
                with self.assertRaises(ReactionDefinitionError):
                    kinetic_reaction_from_mapping(
                        kinetic_definition(
                            'C2H4O -> CH3CHO',
                            type='custom',
                            expression=expression,
                        ),
                        self.components,
                        self.thermo.props,
                    )

    def test_custom_expression_receives_declared_concentration_and_pressure_units(self):
        reaction = kinetic_reaction_from_mapping(
            kinetic_definition(
                'C2H4O -> CH3CHO',
                type='custom',
                A=1.0,
                concentration_unit='mol/m3',
                pressure_unit='kPa',
                expression=(
                    "k * (C['C2H4O'] + p['C2H4O'] + "
                    "f['C2H4O'] + a['C2H4O'])"
                ),
            ),
            self.components,
            self.thermo.props,
        )
        state = HomogeneousRateState.from_flows(
            self.thermo,
            {'C2H4O': 1.0, 'CH3CHO': 1.0},
            500.0,
            2.0,
            'vapor',
        )
        expected = (
            state.concentrations_kmol_m3['C2H4O'] * 1000.0
            + state.partial_pressures_bar['C2H4O'] * 100.0
            + state.fugacities_bar['C2H4O'] * 100.0
            + state.activities['C2H4O']
        )
        self.assertAlmostEqual(
            evaluate_reaction_rate(reaction, state, self.thermo),
            expected,
            places=12,
        )

    def test_custom_net_accepts_signed_rate_but_custom_remains_nonnegative(self):
        state = HomogeneousRateState.from_flows(
            self.thermo,
            {'C2H4O': 0.25, 'CH3CHO': 0.75},
            500.0,
            2.0,
            'vapor',
        )
        expression = "k * (C['C2H4O'] - C['CH3CHO'])"
        signed = kinetic_reaction_from_mapping(
            kinetic_definition(
                'C2H4O -> CH3CHO',
                type='custom_net',
                expression=expression,
            ),
            self.components,
            self.thermo.props,
        )
        ordinary = kinetic_reaction_from_mapping(
            kinetic_definition(
                'C2H4O -> CH3CHO',
                type='custom',
                expression=expression,
            ),
            self.components,
            self.thermo.props,
        )

        self.assertTrue(signed.signed_net_rate)
        self.assertLess(evaluate_reaction_rate(signed, state, self.thermo), 0.0)
        with self.assertRaisesRegex(KineticsError, 'negative rate'):
            evaluate_reaction_rate(ordinary, state, self.thermo)

    def test_custom_net_rejects_reversible_equation(self):
        with self.assertRaisesRegex(
            ReactionDefinitionError,
            'custom_net kinetics requires an irreversible',
        ):
            kinetic_reaction_from_mapping(
                kinetic_definition(
                    'C2H4O <=> CH3CHO',
                    type='custom_net',
                    expression="k * (C['C2H4O'] - C['CH3CHO'])",
                ),
                self.components,
                self.thermo.props,
            )

    def test_custom_expression_uses_dynamic_raw_liquid_volume_percent(self):
        state = HomogeneousRateState.from_flows(
            self.thermo,
            {'C2H4O': 1.0, 'CH3CHO': 3.0},
            300.0,
            2.0,
            'liquid',
        )
        reaction = kinetic_reaction_from_mapping(
            kinetic_definition(
                'C2H4O -> CH3CHO',
                type='custom_net',
                expression="volpct['C2H4O']",
            ),
            self.components,
            self.thermo.props,
        )
        expected = state.raw_liquid_volume_percent(self.thermo)['C2H4O']

        self.assertAlmostEqual(
            evaluate_reaction_rate(reaction, state, self.thermo),
            expected,
            places=12,
        )
        with self.assertRaisesRegex(KineticsError, 'active thermodynamic model'):
            evaluate_forward_reaction_rate(reaction, state)

    def test_custom_volume_percent_rejects_vapor_rate_state(self):
        state = HomogeneousRateState.from_flows(
            self.thermo,
            {'C2H4O': 1.0, 'CH3CHO': 1.0},
            500.0,
            2.0,
            'vapor',
        )
        reaction = kinetic_reaction_from_mapping(
            kinetic_definition(
                'C2H4O -> CH3CHO',
                type='custom_net',
                expression="volpct['C2H4O']",
            ),
            self.components,
            self.thermo.props,
        )

        with self.assertRaisesRegex(KineticsError, 'homogeneous liquid phase'):
            evaluate_reaction_rate(reaction, state, self.thermo)

    def test_log_arrhenius_path_preserves_representable_reverse_rate(self):
        components = ['CO', 'H2', 'CH3OH']
        thermo = create_thermodynamics(components, 'IDEAL')
        reaction = kinetic_reaction_from_mapping(
            kinetic_definition(
                'CO + 2 H2 <=> CH3OH',
                A=1.0,
                Ea=3.26e6,
                rate_basis='activity',
            ),
            components,
            thermo.props,
        )
        state = HomogeneousRateState.from_flows(
            thermo,
            {'CH3OH': 1.0},
            523.15,
            80.0,
            'vapor',
        )
        self.assertEqual(reaction.rate_constant(state.T), 0.0)
        rate = evaluate_reaction_rate(reaction, state, thermo)
        self.assertLess(rate, 0.0)
        self.assertTrue(math.isfinite(rate))


class IsothermalCSTRSolverTests(unittest.TestCase):
    def setUp(self):
        self.components = ['C2H4O', 'CH3CHO']
        self.thermo = create_thermodynamics(self.components, 'IDEAL')

    def reaction(self, **overrides):
        return kinetic_reaction_from_mapping(
            kinetic_definition('C2H4O -> CH3CHO', **overrides),
            self.components,
            self.thermo.props,
        )

    def test_first_order_cstr_matches_analytic_constant_volume_solution(self):
        reaction = self.reaction(A=2.0)
        inlet = {'C2H4O': 10.0}
        temperature = 500.0
        pressure = 2.0
        volume = 5.0
        inlet_state = HomogeneousRateState.from_flows(
            self.thermo, inlet, temperature, pressure, 'vapor'
        )
        volumetric_flow = sum(inlet.values()) / inlet_state.molar_density_kmol_m3
        residence_time = volume / volumetric_flow
        expected_conversion = 2.0 * residence_time / (1.0 + 2.0 * residence_time)

        solution = solve_isothermal_cstr(
            inlet,
            [reaction],
            self.thermo,
            temperature,
            pressure,
            'vapor',
            volume,
        )

        self.assertAlmostEqual(
            solution.extents_kmol_h[0] / inlet['C2H4O'],
            expected_conversion,
            places=12,
        )
        self.assertLess(solution.maximum_residual_kmol_h, 1.0e-10)
        self.assertAlmostEqual(
            solution.outlet_component_flows['C2H4O']
            + solution.outlet_component_flows['CH3CHO'],
            10.0,
            places=12,
        )

    def test_competing_reactions_are_simultaneous_and_never_clip_inventory(self):
        reactions = [self.reaction(A=1.0), self.reaction(A=3.0)]
        solution = solve_isothermal_cstr(
            {'C2H4O': 10.0},
            reactions,
            self.thermo,
            500.0,
            2.0,
            'vapor',
            5.0,
        )

        self.assertAlmostEqual(
            solution.extents_kmol_h[1],
            3.0 * solution.extents_kmol_h[0],
            places=7,
        )
        self.assertGreaterEqual(solution.outlet_component_flows['C2H4O'], 0.0)
        self.assertLess(solution.maximum_residual_kmol_h, 1.0e-8)

    def test_coupled_network_can_consume_an_intermediate_absent_from_feed(self):
        forward = kinetic_reaction_from_mapping(
            kinetic_definition('C2H4O -> CH3CHO', A=2.0),
            self.components,
            self.thermo.props,
        )
        return_path = kinetic_reaction_from_mapping(
            kinetic_definition('CH3CHO -> C2H4O', A=0.5),
            self.components,
            self.thermo.props,
        )
        solution = solve_isothermal_cstr(
            {'C2H4O': 10.0},
            [forward, return_path],
            self.thermo,
            500.0,
            2.0,
            'vapor',
            5.0,
        )

        self.assertGreater(solution.extents_kmol_h[0], 0.0)
        self.assertGreater(solution.extents_kmol_h[1], 0.0)
        self.assertLess(solution.maximum_residual_kmol_h, 1.0e-7)
        self.assertAlmostEqual(
            sum(solution.outlet_component_flows.values()),
            10.0,
            places=10,
        )

    def test_custom_net_cstr_can_solve_negative_extent(self):
        reaction = self.reaction(
            type='custom_net',
            A=2.0,
            expression="-k * C['CH3CHO']",
        )
        solution = solve_isothermal_cstr(
            {'CH3CHO': 10.0},
            [reaction],
            self.thermo,
            500.0,
            2.0,
            'vapor',
            5.0,
        )

        self.assertLess(solution.extents_kmol_h[0], 0.0)
        self.assertGreater(solution.outlet_component_flows['C2H4O'], 0.0)
        self.assertLess(solution.outlet_component_flows['CH3CHO'], 10.0)
        self.assertLess(solution.maximum_residual_kmol_h, 1.0e-9)

    def test_custom_net_cstr_remains_two_sided_in_reaction_network(self):
        forward = self.reaction(A=0.5)
        signed_return = self.reaction(
            type='custom_net',
            A=1.0,
            expression="-k * C['CH3CHO']",
        )
        solution = solve_isothermal_cstr(
            {'CH3CHO': 10.0},
            [forward, signed_return],
            self.thermo,
            500.0,
            2.0,
            'vapor',
            5.0,
        )

        self.assertGreater(solution.extents_kmol_h[0], 0.0)
        self.assertLess(solution.extents_kmol_h[1], 0.0)
        self.assertGreater(solution.outlet_component_flows['C2H4O'], 0.0)
        self.assertLess(solution.maximum_residual_kmol_h, 1.0e-7)

    def test_fast_reversible_cstr_reaches_same_state_from_both_directions(self):
        components = ['CO', 'H2', 'CH3OH']
        thermo = create_thermodynamics(components, 'PR')
        reaction = kinetic_reaction_from_mapping(
            kinetic_definition(
                'CO + 2 H2 <=> CH3OH',
                A=1.0e4,
                rate_basis='activity',
            ),
            components,
            thermo.props,
        )
        forward = solve_isothermal_cstr(
            {'CO': 1.0, 'H2': 2.0},
            [reaction],
            thermo,
            523.15,
            80.0,
            'vapor',
            100.0,
        )
        reverse = solve_isothermal_cstr(
            {'CH3OH': 1.0},
            [reaction],
            thermo,
            523.15,
            80.0,
            'vapor',
            100.0,
        )

        self.assertEqual(forward.reaction_statuses, ('near_equilibrium_conditioned',))
        self.assertEqual(reverse.reaction_statuses, ('near_equilibrium_conditioned',))
        self.assertLess(abs(forward.driving_residuals[0]), 1.0e-10)
        self.assertLess(abs(reverse.driving_residuals[0]), 1.0e-10)
        for component in components:
            self.assertAlmostEqual(
                forward.outlet_component_flows[component],
                reverse.outlet_component_flows[component],
                places=8,
            )


class KineticsCSTRUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.components = ['C2H4O', 'CH3CHO']
        cls.thermo = create_thermodynamics(cls.components, 'IDEAL')

    def feed(self):
        return self.thermo.calculate_state(
            500.0,
            2.0,
            10.0,
            {'C2H4O': 1.0},
            phase='vapor',
            flash=False,
        )

    def params(self):
        return {
            'volume': 5.0,
            'mode': 'isothermal',
            'phase': 'vapor',
            'T': 500.0,
            'P_drop': 0.2,
            'reactions': [kinetic_definition('C2H4O -> CH3CHO', A=2.0)],
        }

    def test_unit_closes_material_energy_and_pressure_consistently(self):
        feed = self.feed()
        result = KineticsCSTR('CSTR-1', self.thermo, self.params()).solve({'in': feed})
        outlet = result.outlet_streams['out']

        self.assertAlmostEqual(outlet.P, 1.8, places=13)
        self.assertAlmostEqual(outlet.T, 500.0, places=13)
        self.assertLess(
            result.performance['maximum_material_rate_residual_kmol_h'],
            1.0e-9,
        )
        self.assertAlmostEqual(
            outlet.F * outlet.H - feed.F * feed.H,
            result.heat_duty,
            places=8,
        )
        self.assertEqual(
            result.performance['phase_stability']['requested_phase'],
            'vapor',
        )

    def test_repeated_solve_is_deterministic(self):
        unit = KineticsCSTR('CSTR-1', self.thermo, self.params())
        first = unit.solve({'in': self.feed()})
        second = unit.solve({'in': self.feed()})
        self.assertEqual(first.outlet_streams['out'].to_dict(), second.outlet_streams['out'].to_dict())
        self.assertEqual(first.performance, second.performance)

    def test_adiabatic_mode_rejects_isothermal_temperature_specification(self):
        params = self.params()
        params['mode'] = 'adiabatic'
        with self.assertRaisesRegex(UnitOperationError, 'cannot specify T'):
            KineticsCSTR('CSTR-1', self.thermo, params).solve({'in': self.feed()})

    def test_liquid_custom_activity_cstr_uses_compiled_activity_backend(self):
        thermo = create_thermodynamics(self.components, 'NRTL')
        feed = thermo.calculate_state(
            300.0,
            10.0,
            1.0,
            {'C2H4O': 0.5, 'CH3CHO': 0.5},
            phase='liquid',
            flash=False,
        )
        reaction = kinetic_definition(
            'C2H4O -> CH3CHO',
            type='custom',
            A=0.2,
            expression="k * a['C2H4O']",
        )
        result = KineticsCSTR('CSTR-L', thermo, {
            'volume': 0.1,
            'T': 300.0,
            'phase': 'liquid',
            'reactions': [reaction],
        }).solve({'in': feed})

        self.assertEqual(
            result.performance['phase_stability']['requested_phase'],
            'liquid',
        )
        self.assertEqual(
            result.performance['reactions'][0]['kinetic_type'],
            'custom',
        )
        self.assertGreater(
            result.performance['component_conversions']['C2H4O'],
            0.0,
        )

    def test_custom_net_cstr_reports_reverse_rate_and_closes_energy(self):
        feed = self.thermo.calculate_state(
            500.0,
            2.0,
            10.0,
            {'CH3CHO': 1.0},
            phase='vapor',
            flash=False,
        )
        result = KineticsCSTR('CSTR-net', self.thermo, {
            'volume': 5.0,
            'mode': 'isothermal',
            'phase': 'vapor',
            'T': 500.0,
            'reactions': [kinetic_definition(
                'C2H4O -> CH3CHO',
                type='custom_net',
                A=0.5,
                expression="-k * C['CH3CHO']",
            )],
        }).solve({'in': feed})

        reaction = result.performance['reactions'][0]
        outlet = result.outlet_streams['out']
        self.assertEqual(reaction['kinetic_type'], 'custom_net')
        self.assertLess(reaction['rate_kmol_m3_h'], 0.0)
        self.assertLess(reaction['extent_kmol_h'], 0.0)
        self.assertGreater(outlet.composition['C2H4O'], 0.0)
        self.assertGreater(
            result.performance['component_conversions']['CH3CHO'],
            0.0,
        )
        self.assertAlmostEqual(
            outlet.F * outlet.H - feed.F * feed.H,
            result.heat_duty,
            places=8,
        )

    def test_reversible_cstr_requires_complete_equilibrium_thermochemistry(self):
        thermo = create_thermodynamics(self.components, 'IDEAL')
        thermo.props['CH3CHO'] = deepcopy(thermo.props['CH3CHO'])
        thermo.props['CH3CHO'].Gf = None
        feed = thermo.calculate_state(
            500.0,
            2.0,
            10.0,
            {'C2H4O': 1.0},
            phase='vapor',
            flash=False,
        )
        params = {
            'volume': 5.0,
            'T': 500.0,
            'phase': 'vapor',
            'reactions': [kinetic_definition(
                'C2H4O <=> CH3CHO',
                rate_basis='activity',
            )],
        }
        with self.assertRaisesRegex(UnitOperationError, 'Gf: CH3CHO'):
            KineticsCSTR('CSTR-R', thermo, params).solve({'in': feed})


class AdaptivePFRKineticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.components = ['C2H4O', 'CH3CHO']
        cls.thermo = create_thermodynamics(cls.components, 'IDEAL')
        cls.feed = cls.thermo.calculate_state(
            500.0,
            2.0,
            10.0,
            {'C2H4O': 1.0},
            phase='vapor',
            flash=False,
        )

    def solve(self, reaction, **extra):
        params = {
            'volume': 1.0,
            'profile_points': 21,
            'mode': 'isothermal',
            'phase': 'vapor',
            'T': 500.0,
            'reactions': [reaction],
        }
        params.update(extra)
        return KineticsPFR('PFR-1', self.thermo, params).solve({'in': self.feed})

    def test_adaptive_pfr_normalizes_concentration_and_rate_units(self):
        hourly = self.solve(kinetic_definition('C2H4O -> CH3CHO', A=2.0))
        per_second = self.solve(kinetic_definition(
            'C2H4O -> CH3CHO',
            A=2.0 / 3600.0,
            concentration_unit='mol/m3',
            rate_unit='mol/m3/s',
        ))
        self.assertAlmostEqual(
            hourly.outlet_streams['out'].composition['C2H4O'],
            per_second.outlet_streams['out'].composition['C2H4O'],
            places=12,
        )

    def test_adaptive_pfr_supports_activity_custom_and_reversible_rates(self):
        cases = (
            kinetic_definition('C2H4O -> CH3CHO', rate_basis='activity'),
            kinetic_definition(
                'C2H4O -> CH3CHO',
                type='custom',
                expression="k * C['C2H4O']",
            ),
            kinetic_definition('C2H4O <=> CH3CHO'),
        )
        for reaction in cases:
            with self.subTest(reaction=reaction):
                result = self.solve(reaction)
                self.assertGreater(
                    result.performance['component_conversions']['C2H4O'],
                    0.0,
                )
                self.assertLess(
                    result.performance['maximum_material_balance_residual_kmol_h'],
                    1.0e-8,
                )

    def test_custom_net_pfr_can_run_written_stoichiometry_in_reverse(self):
        product_feed = self.thermo.calculate_state(
            500.0,
            2.0,
            10.0,
            {'CH3CHO': 1.0},
            phase='vapor',
            flash=False,
        )
        reaction = kinetic_definition(
            'C2H4O -> CH3CHO',
            type='custom_net',
            A=0.5,
            expression="-k * C['CH3CHO']",
        )
        result = KineticsPFR('PFR-net', self.thermo, {
            'volume': 1.0,
            'profile_points': 7,
            'mode': 'isothermal',
            'phase': 'vapor',
            'T': 500.0,
            'reactions': [reaction],
        }).solve({'in': product_feed})

        reaction_result = result.performance['reactions'][0]
        self.assertEqual(reaction_result['kinetic_type'], 'custom_net')
        self.assertLess(reaction_result['extent_kmol_h'], 0.0)
        self.assertGreater(result.outlet_streams['out'].composition['C2H4O'], 0.0)
        self.assertGreater(
            result.performance['component_conversions']['CH3CHO'],
            0.0,
        )

    def test_first_order_equal_mole_pfr_matches_analytic_solution(self):
        rate_constant = 2.0
        result = self.solve(kinetic_definition(
            'C2H4O -> CH3CHO', A=rate_constant
        ))
        inlet_density = self.thermo.mixture_molar_density(
            {'C2H4O': 1.0}, 500.0, 2.0, 1.0, y={'C2H4O': 1.0}
        )
        residence_time = 1.0 / (self.feed.F / inlet_density)
        expected = 1.0 - math.exp(-rate_constant * residence_time)
        self.assertAlmostEqual(
            result.performance['component_conversions']['C2H4O'],
            expected,
            places=10,
        )
        self.assertEqual(result.performance['solver_method'], 'RK45')


class KineticPFDValidationTests(unittest.TestCase):
    def pfd(self, reaction_line):
        return PFDParser().parse(f'''\
PROCESS: Compact kinetic validation
VERSION: 1.0
ONLINE_LOOKUP: false
COMPONENTS:
    C2H4O | Ethylene oxide | formula=C2H4O
    CH3CHO | Acetaldehyde | formula=C2H4O
STREAM Feed : -> CSTR-1.in
    T = 500 [K]
    P = 2 [bar]
    F = 10 [kmol/h]
    x = C2H4O:1
STREAM Product : CSTR-1.out
UNIT CSTR-1 : CSTR
    volume = 5 [m3]
    T = 500 [K]
    phase = vapor
    REACTIONS:
        {reaction_line}
''')

    def test_compact_pfd_accepts_explicit_kinetic_units(self):
        pfd = self.pfd(
            'C2H4O -> CH3CHO | A=2, Ea=0, Ea_unit=J/mol, '
            'rate_basis=activity, concentration_unit=kmol/m3, '
            'pressure_unit=bar, rate_unit=kmol/m3/h'
        )
        errors, warnings = validate_pfd(pfd)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_compact_pfd_accepts_and_round_trips_custom_net(self):
        pfd = self.pfd(
            "C2H4O -> CH3CHO | type=custom_net, A=1, Ea=0, "
            "Ea_unit=J/mol, concentration_unit=kmol/m3, pressure_unit=bar, "
            "rate_unit=kmol/m3/h, "
            "expression=k*(C['C2H4O']-C['CH3CHO'])"
        )
        self.assertEqual(validate_pfd(pfd), ([], []))
        restored = PFDParser().parse(pfd.to_pfd())
        self.assertEqual(validate_pfd(restored), ([], []))
        self.assertEqual(
            restored.units[0].reactions[0].parameters['type'],
            'custom_net',
        )

    def test_compact_pfd_rejects_unitless_legacy_kinetics(self):
        pfd = self.pfd('C2H4O -> CH3CHO | A=2, Ea=0')
        errors, _warnings = validate_pfd(pfd)
        self.assertTrue(any('Ea_unit' in error for error in errors), errors)


if __name__ == '__main__':
    unittest.main()
