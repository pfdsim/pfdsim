import unittest

from chemical_properties import ChemicalDatabase
from pfd_parser import PFDParser, validate_pfd
from reaction_models import (
    ReactionDefinitionError,
    equilibrium_reaction_from_mapping,
    solve_homogeneous_equilibrium,
)
from simulator import Simulator
from thermodynamics import create_thermodynamics
from thermodynamics_models.common import P_REF, R
from unit_operations_base import UnitOperationError
from unit_operations_reactors import EquilibriumReactor


class FormationProvenanceTests(unittest.TestCase):
    def test_legacy_chemicals_json_formation_values_default_to_quality_098(self):
        database = ChemicalDatabase(enable_online=False)

        for component, properties in (
            ('CO', ('Hf', 'Gf')),
            ('CH3OH', ('Hf_liquid', 'Gf_liquid')),
        ):
            with self.subTest(component=component):
                sources = database.chemicals[component].property_sources
                for property_name in properties:
                    source = sources[property_name]
                    self.assertEqual(source['source'], 'local')
                    self.assertEqual(source['method'], 'chemicals_json')
                    self.assertEqual(source['quality'], 0.98)

    def test_equilibrium_reactor_marks_formation_sources_as_result_affecting(self):
        simulator = Simulator.from_file(
            'examples/equilibrium_methanol_synthesis_recycle.pfd'
        )
        simulator.run(max_iterations=100)
        report = simulator._property_quality_report(include_suppressed=True)
        by_key = {
            (entry['component'], entry['property']): entry
            for entry in report
        }

        for component in ('CO', 'H2', 'CH3OH'):
            for property_name in ('Hf', 'Gf', 'S'):
                with self.subTest(
                    component=component,
                    property=property_name,
                ):
                    entry = by_key[(component, property_name)]
                    self.assertTrue(entry['affects_result'])
                    self.assertTrue(any(
                        context.get('unit_id') == 'EQ-100'
                        for context in entry['contexts']
                    ))
        self.assertEqual(by_key[('CO', 'Hf')]['quality'], 0.98)
        self.assertEqual(by_key[('CO', 'Gf')]['quality'], 0.98)


class ReactionThermodynamicInterfaceTests(unittest.TestCase):
    def test_standard_reaction_gibbs_matches_log_equilibrium_constant(self):
        thermo = create_thermodynamics(['CO', 'H2', 'CH3OH'], 'IDEAL')
        stoichiometry = {'CO': -1.0, 'H2': -2.0, 'CH3OH': 1.0}
        temperature = 523.15

        delta_g = thermo.reaction_standard_gibbs(stoichiometry, temperature)
        log_k = thermo.reaction_log_equilibrium_constant(
            stoichiometry,
            temperature,
        )

        self.assertAlmostEqual(log_k, -delta_g / (R * temperature), places=13)

    def test_cubic_vapor_activities_use_compiled_wrapper_fugacities(self):
        thermo = create_thermodynamics(['CO', 'H2', 'CH3OH'], 'PR')
        composition = {'CO': 0.3, 'H2': 0.6, 'CH3OH': 0.1}
        temperature = 500.0
        pressure = 80.0

        activities = thermo.component_activities(
            temperature,
            pressure,
            composition,
            'vapor',
        )
        phi = thermo.fugacity_coefficients(
            temperature,
            pressure,
            composition,
            'vapor',
        )

        for component, fraction in composition.items():
            self.assertAlmostEqual(
                activities[component],
                fraction * phi[component] * pressure / P_REF,
                places=13,
            )

    def test_compiled_nrtl_liquid_activities_use_gamma_phi_reference(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'NRTL-PR')
        composition = {'ethanol': 0.4, 'water': 0.6}
        temperature = 350.0
        pressure = 5.0

        activities = thermo.component_activities(
            temperature,
            pressure,
            composition,
            'liquid',
        )
        gamma = thermo.activity_coefficients(temperature, composition)
        reference = thermo._gamma_phi_reference_factors(temperature, pressure)

        for component, fraction in composition.items():
            self.assertAlmostEqual(
                activities[component],
                fraction * gamma[component] * reference[component] / P_REF,
                places=13,
            )


class HomogeneousEquilibriumSolverTests(unittest.TestCase):
    @staticmethod
    def methanol_reaction(thermo):
        return equilibrium_reaction_from_mapping(
            {'equation': 'CO + 2 H2 <=> CH3OH'},
            thermo.components,
            thermo.props,
        )

    def test_ideal_pr_and_psrk_vapor_equilibria_match_reference_extents(self):
        expected = {
            'IDEAL': 0.6792631683007853,
            'PR': 0.7383797296064345,
            'PSRK': 0.7482556863611427,
        }
        for method, expected_extent in expected.items():
            with self.subTest(method=method):
                thermo = create_thermodynamics(['CO', 'H2', 'CH3OH'], method)
                solution = solve_homogeneous_equilibrium(
                    {'CO': 1.0, 'H2': 2.0},
                    [self.methanol_reaction(thermo)],
                    thermo,
                    523.15,
                    80.0,
                    'vapor',
                )

                self.assertAlmostEqual(
                    solution.extents_kmol_h[0], expected_extent, places=9
                )
                self.assertLess(solution.max_interior_residual, 1.0e-12)
                self.assertEqual(
                    solution.reaction_statuses,
                    ('interior_equilibrium',),
                )

    def test_two_independent_reactions_close_simultaneously(self):
        components = ['CO', 'H2', 'H2O', 'CO2', 'CH3OH']
        thermo = create_thermodynamics(components, 'IDEAL')
        reactions = [
            equilibrium_reaction_from_mapping(
                {'equation': 'CO + H2O <=> CO2 + H2'},
                components,
                thermo.props,
            ),
            equilibrium_reaction_from_mapping(
                {'equation': 'CO + 2 H2 <=> CH3OH'},
                components,
                thermo.props,
            ),
        ]

        solution = solve_homogeneous_equilibrium(
            {'CO': 2.0, 'H2': 4.0, 'H2O': 1.0},
            reactions,
            thermo,
            600.0,
            20.0,
            'vapor',
        )

        self.assertEqual(len(solution.extents_kmol_h), 2)
        self.assertLess(solution.max_interior_residual, 1.0e-10)
        self.assertTrue(all(flow >= 0.0 for flow in solution.outlet_component_flows.values()))

    def test_ammonia_rks_bm_pressure_response_is_nonideal(self):
        components = ['N2', 'H2', 'NH3']
        extents = {}
        for method in ('IDEAL', 'RKS-BM'):
            thermo = create_thermodynamics(components, method)
            reaction = equilibrium_reaction_from_mapping(
                {'equation': 'N2 + 3 H2 <=> 2 NH3'},
                components,
                thermo.props,
            )
            extents[method] = []
            for pressure in (100.0, 200.0):
                solution = solve_homogeneous_equilibrium(
                    {'N2': 1.0, 'H2': 3.0},
                    [reaction],
                    thermo,
                    700.0,
                    pressure,
                    'vapor',
                )
                extents[method].append(solution.extents_kmol_h[0])

        ideal_low, ideal_high = extents['IDEAL']
        eos_low, eos_high = extents['RKS-BM']
        self.assertGreater(eos_high, eos_low)
        self.assertGreater(eos_low, ideal_low)
        self.assertGreater(eos_high, ideal_high)
        self.assertGreater(
            abs((eos_high - eos_low) - (ideal_high - ideal_low)),
            0.01,
        )

        thermo = create_thermodynamics(components, 'RKS-BM')
        reaction = equilibrium_reaction_from_mapping(
            {'equation': 'N2 + 3 H2 <=> 2 NH3'},
            components,
            thermo.props,
        )
        hotter = solve_homogeneous_equilibrium(
            {'N2': 1.0, 'H2': 3.0},
            [reaction],
            thermo,
            750.0,
            200.0,
            'vapor',
        )
        self.assertLess(hotter.extents_kmol_h[0], eos_high)

    def test_linearly_dependent_reactions_are_rejected(self):
        thermo = create_thermodynamics(['CO', 'H2', 'CH3OH'], 'IDEAL')
        reactions = [
            equilibrium_reaction_from_mapping(
                {'equation': 'CO + 2 H2 <=> CH3OH'},
                thermo.components,
                thermo.props,
            ),
            equilibrium_reaction_from_mapping(
                {'equation': '2 CO + 4 H2 <=> 2 CH3OH'},
                thermo.components,
                thermo.props,
            ),
        ]

        with self.assertRaisesRegex(ReactionDefinitionError, 'linearly dependent'):
            solve_homogeneous_equilibrium(
                {'CO': 1.0, 'H2': 2.0},
                reactions,
                thermo,
                523.15,
                80.0,
                'vapor',
            )

    def test_complete_conversion_is_reported_as_boundary_limited(self):
        thermo = create_thermodynamics(['C2H4O', 'CH3CHO'], 'NRTL')
        reaction = equilibrium_reaction_from_mapping(
            {'equation': 'C2H4O <=> CH3CHO'},
            thermo.components,
            thermo.props,
        )

        solution = solve_homogeneous_equilibrium(
            {'C2H4O': 0.5, 'CH3CHO': 0.5},
            [reaction],
            thermo,
            300.0,
            10.0,
            'liquid',
        )

        self.assertEqual(solution.outlet_component_flows, {'CH3CHO': 1.0})
        self.assertEqual(solution.reaction_statuses, ('boundary_limited',))
        self.assertEqual(solution.max_interior_residual, 0.0)
        self.assertLess(solution.residuals[0], 0.0)


class EquilibriumReactorUnitTests(unittest.TestCase):
    def test_isothermal_vapor_reactor_reports_equilibrium_and_closes_energy(self):
        thermo = create_thermodynamics(['CO', 'H2', 'CH3OH'], 'PR')
        feed = thermo.calculate_state(
            523.15,
            80.0,
            3.0,
            {'CO': 1.0 / 3.0, 'H2': 2.0 / 3.0},
            phase='vapor',
            flash=False,
        )
        result = EquilibriumReactor('EQ', thermo, {
            'phase': 'vapor',
            'T_out': 523.15,
            'P_drop': 2.0,
            'reactions': [{'equation': 'CO + 2 H2 <=> CH3OH'}],
        }).solve({'feed': feed})
        outlet = result.outlet_streams['out']

        self.assertAlmostEqual(outlet.P, 78.0)
        self.assertAlmostEqual(outlet.T, 523.15)
        self.assertLess(
            result.performance['maximum_interior_ln_equilibrium_residual'],
            1.0e-10,
        )
        self.assertAlmostEqual(
            outlet.F * outlet.H - feed.F * feed.H,
            result.heat_duty,
            delta=1.0e-5,
        )
        self.assertEqual(
            result.performance['reactions'][0]['equilibrium_status'],
            'interior_equilibrium',
        )

    def test_adiabatic_and_duty_reactors_close_nested_energy_balance(self):
        thermo = create_thermodynamics(['N2', 'H2', 'NH3'], 'PR')
        feed = thermo.calculate_state(
            700.0,
            200.0,
            4.0,
            {'N2': 0.25, 'H2': 0.75},
            phase='vapor',
            flash=False,
        )
        cases = (
            ({'mode': 'adiabatic'}, 0.0),
            ({'mode': 'duty', 'Q': 10.0, '__unit__Q': 'kW'}, 36000.0),
        )
        for extra, expected_duty in cases:
            with self.subTest(mode=extra['mode']):
                result = EquilibriumReactor('EQ', thermo, {
                    'phase': 'vapor',
                    **extra,
                    'reactions': [{'equation': 'N2 + 3 H2 <=> 2 NH3'}],
                }).solve({'feed': feed})
                outlet = result.outlet_streams['out']
                self.assertAlmostEqual(result.heat_duty, expected_duty, places=8)
                self.assertAlmostEqual(
                    outlet.F * outlet.H - feed.F * feed.H - expected_duty,
                    0.0,
                    delta=1.0e-4,
                )

    def test_liquid_activity_reactor_accepts_boundary_equilibrium(self):
        thermo = create_thermodynamics(['C2H4O', 'CH3CHO'], 'NRTL')
        feed = thermo.calculate_state(
            300.0,
            10.0,
            1.0,
            {'C2H4O': 0.5, 'CH3CHO': 0.5},
            phase='liquid',
            flash=False,
        )
        result = EquilibriumReactor('EQ', thermo, {
            'phase': 'liquid',
            'T': 300.0,
            'reactions': [{'equation': 'C2H4O <=> CH3CHO'}],
        }).solve({'feed': feed})

        self.assertEqual(result.outlet_streams['out'].composition, {'CH3CHO': 1.0})
        reaction = result.performance['reactions'][0]
        self.assertEqual(reaction['equilibrium_status'], 'boundary_limited')
        self.assertEqual(
            reaction['reaction_quotient_basis'],
            '1e-14_mole_fraction_floor',
        )
        self.assertFalse(any(
            'saturated vapor fugacity coefficient' in warning
            for warning in thermo.warnings
        ))

    def test_global_liquid_liquid_instability_is_rejected(self):
        thermo = create_thermodynamics(['water', 'butanol'], 'UNIFNIST')
        outlet = thermo.calculate_state(
            298.15,
            1.0,
            1.0,
            {'water': 0.5, 'butanol': 0.5},
            phase='liquid',
            flash=False,
        )
        unit = EquilibriumReactor('EQ', thermo, {})

        with self.assertRaisesRegex(UnitOperationError, 'liquid-liquid unstable'):
            unit._validate_equilibrium_phase_stability(outlet, 'liquid')

    def test_vapor_phase_instability_is_rejected(self):
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
            EquilibriumReactor('EQ', thermo, {
                'phase': 'vapor',
                'T': 350.0,
                'reactions': [{'equation': 'CO + 2 H2 <=> CH3OH'}],
            }).solve({'feed': feed})

    def test_repeated_solve_is_deterministic(self):
        thermo = create_thermodynamics(['CO', 'H2', 'CH3OH'], 'IDEAL')
        feed = thermo.calculate_state(
            523.15,
            80.0,
            3.0,
            {'CO': 1.0 / 3.0, 'H2': 2.0 / 3.0},
            phase='vapor',
            flash=False,
        )
        params = {
            'phase': 'vapor',
            'T': 523.15,
            'reactions': [{'equation': 'CO + 2 H2 <=> CH3OH'}],
        }
        unit = EquilibriumReactor('EQ', thermo, params)

        first = unit.solve({'feed': feed}).outlet_streams['out']
        second = unit.solve({'feed': feed}).outlet_streams['out']

        self.assertEqual(first.component_flows(), second.component_flows())
        self.assertEqual(params['reactions'], [
            {'equation': 'CO + 2 H2 <=> CH3OH'}
        ])

    def test_old_approach_and_irreversible_arrows_are_rejected(self):
        thermo = create_thermodynamics(['CO', 'H2', 'CH3OH'], 'IDEAL')
        feed = thermo.calculate_state(
            523.15,
            80.0,
            3.0,
            {'CO': 1.0 / 3.0, 'H2': 2.0 / 3.0},
            phase='vapor',
            flash=False,
        )
        for reaction, message in (
            ({'equation': 'CO + 2 H2 <=> CH3OH', 'approach': 0.9}, 'does not accept'),
            ({'equation': 'CO + 2 H2 -> CH3OH'}, 'reversible arrow'),
        ):
            with self.subTest(reaction=reaction):
                with self.assertRaisesRegex(UnitOperationError, message):
                    EquilibriumReactor('EQ', thermo, {
                        'phase': 'vapor',
                        'T': 523.15,
                        'reactions': [reaction],
                    }).solve({'feed': feed})


class EquilibriumReactorPFDTests(unittest.TestCase):
    def test_pfd_validation_requires_reversible_reaction_and_no_approach(self):
        template = """
COMPONENTS:
    CO | Carbon monoxide | formula=CO
    H2 | Hydrogen | formula=H2
    CH3OH | Methanol | formula=CH4O
UNIT EQ
    TYPE: EquilibriumReactor
    PORTS:
        in: inlet
        out: outlet
    PARAMS:
        phase = vapor
        T = 523.15 [K]
    REACTIONS:
        {reaction}
"""
        valid = PFDParser().parse(template.format(
            reaction='CO + 2 H2 <=> CH3OH',
        ))
        errors, _warnings = validate_pfd(valid)
        self.assertEqual(errors, [])

        for reaction in (
            'CO + 2 H2 -> CH3OH',
            'CO + 2 H2 <=> CH3OH | approach=0.9',
        ):
            with self.subTest(reaction=reaction):
                pfd = PFDParser().parse(template.format(reaction=reaction))
                errors, _warnings = validate_pfd(pfd)
                self.assertEqual(len(errors), 1)

    def test_equilibrium_recycle_converges_and_closes_reaction_residual(self):
        pfd = """
PROCESS: equilibrium methanol recycle
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
RECYCLE_METHOD: WEGSTEIN
COMPONENTS:
    CO | Carbon monoxide | formula=CO
    H2 | Hydrogen | formula=H2
    CH3OH | Methanol | formula=CH4O
STREAM Feed : FEED -> M.fresh
    T = 523.15 [K]
    P = 80 [bar]
    F = 3 [kmol/h]
    x = CO:0.3333333333333333, H2:0.6666666666666667
STREAM Recycle : SP.recycle -> M.recycle
STREAM Mixed : M.out -> EQ.in
STREAM Reacted : EQ.out -> SP.in
STREAM Product : SP.product -> PRODUCT
UNIT M
    TYPE: Mixer
    PORTS:
        fresh : inlet
        recycle : inlet
        out : outlet
    PARAMS:
        T_out = 523.15 [K]
        P = 80 [bar]
UNIT EQ
    TYPE: EquilibriumReactor
    PORTS:
        in : inlet
        out : outlet
    PARAMS:
        phase = vapor
        T_out = 523.15 [K]
    REACTIONS:
        CO + 2 H2 <=> CH3OH
UNIT SP
    TYPE: Splitter
    PORTS:
        in : inlet
        product : outlet
        recycle : outlet
    PARAMS:
        outlets = product, recycle
        product_split_frac = 0.5
"""
        result = Simulator.from_string(pfd).run(max_iterations=100)

        self.assertTrue(result.converged, result.warnings)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.recycle_info['tear_streams'], ['Recycle'])
        self.assertLess(
            result.units['EQ'].performance[
                'maximum_interior_ln_equilibrium_residual'
            ],
            1.0e-9,
        )
        self.assertLess(result.mass_balance_error, 1.0e-4)
        self.assertLess(result.energy_balance_error, 1.0e-4)


if __name__ == '__main__':
    unittest.main()
