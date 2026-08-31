import copy
import unittest

from pfd_parser import PFDParser, validate_pfd
from reaction_models import (
    ReactionDefinitionError,
    conversion_reaction_from_mapping,
    parse_reaction_equation,
    solve_conversion_extents,
    validate_reaction_balance,
)
from simulator import Simulator
from thermodynamics import create_thermodynamics
from unit_operations_base import UnitOperationError
from unit_operations_reactors import Reactor


class ReactionEquationTests(unittest.TestCase):
    def test_parser_combines_repeated_species_and_cancels_both_sides(self):
        reaction = parse_reaction_equation(
            'A + A + B -> 2 C + A',
            {'A', 'B', 'C'},
        )

        self.assertEqual(reaction.reactants, {'A': 2.0, 'B': 1.0})
        self.assertEqual(reaction.products, {'C': 2.0, 'A': 1.0})
        self.assertEqual(reaction.stoichiometry, {'A': -1.0, 'B': -1.0, 'C': 2.0})
        self.assertEqual(reaction.default_basis_component, 'A')

    def test_parser_supports_non_word_component_symbols(self):
        reaction = parse_reaction_equation(
            '(C2H5)2O + H+ -> protonated-ether',
            {'(C2H5)2O', 'H+', 'protonated-ether'},
        )

        self.assertEqual(reaction.stoichiometry, {
            '(C2H5)2O': -1.0,
            'H+': -1.0,
            'protonated-ether': 1.0,
        })

    def test_parser_supports_compact_coefficients_without_misreading_numeric_names(self):
        reaction = parse_reaction_equation(
            '2H2 + O2 -> 2H2O',
            {'H2', 'O2', 'H2O'},
        )
        named = parse_reaction_equation(
            '2-propanol -> acetone + H2',
            {'2-propanol', 'acetone', 'H2'},
        )

        self.assertEqual(reaction.stoichiometry, {
            'H2': -2.0,
            'O2': -1.0,
            'H2O': 2.0,
        })
        self.assertEqual(named.stoichiometry['2-propanol'], -1.0)

    def test_parser_rejects_unknown_species_and_multiple_arrows(self):
        with self.assertRaisesRegex(ReactionDefinitionError, 'undefined component'):
            parse_reaction_equation('A -> Missing', {'A', 'B'})
        with self.assertRaisesRegex(ReactionDefinitionError, 'exactly one reaction arrow'):
            parse_reaction_equation('A -> B -> C', {'A', 'B', 'C'})

    def test_elemental_balance_is_validated_when_available(self):
        balanced = parse_reaction_equation(
            '2 H2 + O2 -> 2 H2O',
            {'H2', 'O2', 'H2O'},
        )
        metadata = {
            'H2': {'formula': 'H2'},
            'O2': {'formula': 'O2'},
            'H2O': {'formula': 'H2O'},
        }
        self.assertEqual(validate_reaction_balance(balanced, metadata), [])

        unbalanced = parse_reaction_equation(
            'H2 + O2 -> H2O',
            {'H2', 'O2', 'H2O'},
        )
        with self.assertRaisesRegex(ReactionDefinitionError, 'elementally balanced'):
            validate_reaction_balance(unbalanced, metadata)

    def test_missing_formula_produces_an_explicit_validation_warning(self):
        reaction = parse_reaction_equation('A -> B', {'A', 'B'})
        warnings = validate_reaction_balance(reaction, {
            'A': {'formula': 'H2'},
            'B': {'formula': None},
        })

        self.assertEqual(len(warnings), 1)
        self.assertIn('elemental balance was not checked', warnings[0])
        self.assertIn('B', warnings[0])


class ConversionExtentTests(unittest.TestCase):
    COMPONENTS = ('C2H4', 'O2', 'C2H4O', 'CO2', 'H2O')
    METADATA = {
        'C2H4': {'formula': 'C2H4'},
        'O2': {'formula': 'O2'},
        'C2H4O': {'formula': 'C2H4O'},
        'CO2': {'formula': 'CO2'},
        'H2O': {'formula': 'H2O'},
    }

    def reaction(self, equation, conversion, **extra):
        return conversion_reaction_from_mapping(
            {'equation': equation, 'conversion': conversion, **extra},
            self.COMPONENTS,
            self.METADATA,
        )

    def test_competing_conversions_use_original_inlet_basis_and_are_order_independent(self):
        desired = self.reaction('C2H4 + 0.5 O2 -> C2H4O', 0.70)
        combustion = self.reaction('C2H4 + 3 O2 -> 2 CO2 + 2 H2O', 0.10)
        inlet = {'C2H4': 10.0, 'O2': 30.0}

        forward = solve_conversion_extents(inlet, [desired, combustion])
        reverse = solve_conversion_extents(inlet, [combustion, desired])

        expected = {
            'C2H4': 2.0,
            'O2': 23.5,
            'C2H4O': 7.0,
            'CO2': 2.0,
            'H2O': 2.0,
        }
        self.assertEqual(forward.outlet_component_flows, expected)
        self.assertEqual(reverse.outlet_component_flows, expected)

    def test_explicit_basis_controls_extent(self):
        reaction = self.reaction(
            'C2H4 + 0.5 O2 -> C2H4O',
            0.50,
            basis='O2',
        )
        solution = solve_conversion_extents(
            {'C2H4': 10.0, 'O2': 4.0},
            [reaction],
        )

        self.assertAlmostEqual(solution.reaction_extents[0].extent_kmol_h, 4.0)
        self.assertAlmostEqual(solution.outlet_component_flows['O2'], 2.0)
        self.assertAlmostEqual(solution.outlet_component_flows['C2H4'], 6.0)

    def test_jointly_infeasible_conversions_are_rejected_without_clipping(self):
        first = self.reaction('C2H4 + 0.5 O2 -> C2H4O', 0.70)
        second = self.reaction('C2H4 + 3 O2 -> 2 CO2 + 2 H2O', 0.40)

        with self.assertRaisesRegex(ReactionDefinitionError, 'jointly infeasible'):
            solve_conversion_extents(
                {'C2H4': 10.0, 'O2': 30.0},
                [first, second],
            )

    def test_selectivity_and_yield_are_result_only_fields(self):
        for field in ('selectivity', 'yield'):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ReactionDefinitionError, 'calculated reactor result'):
                    conversion_reaction_from_mapping(
                        {
                            'equation': 'C2H4 + 0.5 O2 -> C2H4O',
                            field: 0.9,
                            'conversion': 0.5,
                        },
                        self.COMPONENTS,
                        self.METADATA,
                    )


class ConversionReactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thermo = create_thermodynamics(
            ['C2H4', 'O2', 'C2H4O', 'CO2', 'H2O'],
            'IDEAL',
        )

    def feed(self):
        return self.thermo.calculate_state(
            600.0,
            10.0,
            40.0,
            {'C2H4': 0.25, 'O2': 0.75},
            phase='vapor',
        )

    @staticmethod
    def reactions():
        return [
            {
                'equation': 'C2H4 + 0.5 O2 -> C2H4O',
                'conversion': 0.70,
                'name': 'epoxidation',
            },
            {
                'equation': 'C2H4 + 3 O2 -> 2 CO2 + 2 H2O',
                'conversion': 0.10,
                'name': 'combustion',
            },
        ]

    def test_isothermal_reactor_closes_material_energy_and_reports_selectivity(self):
        feed = self.feed()
        reactor = Reactor('R', self.thermo, {
            'T_out': 600.0,
            'P_drop': 1.5,
            'desired_product': 'C2H4O',
            'reactions': self.reactions(),
        })

        result = reactor.solve({'feed': feed})
        outlet = result.outlet_streams['out']
        flows = outlet.component_flows()

        self.assertAlmostEqual(flows['C2H4'], 2.0, places=10)
        self.assertAlmostEqual(flows['O2'], 23.5, places=10)
        self.assertAlmostEqual(flows['C2H4O'], 7.0, places=10)
        self.assertAlmostEqual(flows['CO2'], 2.0, places=10)
        self.assertAlmostEqual(flows['H2O'], 2.0, places=10)
        self.assertAlmostEqual(outlet.P, 8.5, places=12)
        self.assertAlmostEqual(outlet.T, 600.0, places=12)
        self.assertAlmostEqual(
            outlet.F * outlet.H - feed.F * feed.H,
            result.heat_duty,
            delta=1.0e-6,
        )
        reaction_results = result.performance['reactions']
        self.assertAlmostEqual(reaction_results[0]['yield_on_inlet_basis'], 0.70)
        self.assertAlmostEqual(reaction_results[0]['selectivity_on_converted_basis'], 0.875)
        self.assertAlmostEqual(reaction_results[1]['yield_on_inlet_basis'], 0.10)
        self.assertAlmostEqual(reaction_results[1]['selectivity_on_converted_basis'], 0.125)
        desired = result.performance['desired_product']
        self.assertAlmostEqual(desired['yield_mol_per_mol_inlet_basis'], 0.70)
        self.assertAlmostEqual(desired['selectivity_mol_per_mol_basis_consumed'], 0.875)
        self.assertAlmostEqual(result.performance['component_conversions']['C2H4'], 0.80)

    def test_adiabatic_and_specified_duty_modes_close_full_enthalpy_balance(self):
        feed = self.feed()
        for params, expected_duty in (
            ({'mode': 'adiabatic'}, 0.0),
            ({'mode': 'duty', 'Q': 125.0, '__unit__Q': 'kW'}, 450000.0),
        ):
            with self.subTest(mode=params['mode']):
                result = Reactor('R', self.thermo, {
                    **params,
                    'reactions': [{
                        'equation': 'C2H4 + 0.5 O2 -> C2H4O',
                        'conversion': 0.10,
                    }],
                }).solve({'in': feed})
                outlet = result.outlet_streams['out']
                self.assertAlmostEqual(result.heat_duty, expected_duty, places=7)
                self.assertAlmostEqual(
                    outlet.F * outlet.H - feed.F * feed.H - expected_duty,
                    0.0,
                    delta=1.0e-4,
                )

    def test_repeated_solve_is_deterministic_and_does_not_mutate_reactions(self):
        params = {
            'T': 600.0,
            'reactions': self.reactions(),
        }
        original = copy.deepcopy(params)
        reactor = Reactor('R', self.thermo, params)

        first = reactor.solve({'in': self.feed()}).outlet_streams['out']
        second = reactor.solve({'in': self.feed()}).outlet_streams['out']

        self.assertEqual(params, original)
        self.assertEqual(first.component_flows(), second.component_flows())
        self.assertEqual(first.T, second.T)

    def test_missing_formation_enthalpy_is_rejected(self):
        original = self.thermo.props['C2H4O'].Hf
        self.thermo.props['C2H4O'].Hf = None
        try:
            with self.assertRaisesRegex(UnitOperationError, 'formation enthalpy Hf'):
                Reactor('R', self.thermo, {
                    'T': 600.0,
                    'reactions': [{
                        'equation': 'C2H4 + 0.5 O2 -> C2H4O',
                        'conversion': 0.10,
                    }],
                }).solve({'in': self.feed()})
        finally:
            self.thermo.props['C2H4O'].Hf = original

    def test_ambiguous_desired_product_basis_requires_report_basis(self):
        with self.assertRaisesRegex(UnitOperationError, 'requires report_basis'):
            Reactor('R', self.thermo, {
                'T': 600.0,
                'desired_product': 'C2H4O',
                'reactions': [
                    {
                        'equation': 'C2H4 + 0.5 O2 -> C2H4O',
                        'conversion': 0.10,
                        'basis': 'C2H4',
                    },
                    {
                        'equation': 'C2H4 + 0.5 O2 -> C2H4O',
                        'conversion': 0.10,
                        'basis': 'O2',
                    },
                ],
            }).solve({'in': self.feed()})


class ReactionPFDValidationTests(unittest.TestCase):
    def test_reaction_round_trip(self):
        text = """
COMPONENTS:
    A | A | formula=H, Hf=0
    B | B | formula=H, Hf=0

UNIT R
    TYPE: Reactor
    PORTS:
        in: inlet
        out: outlet
    PARAMS:
        mode = isothermal
        T = 300 [K]
    REACTIONS:
        A -> B | conversion=0.5, basis=A
"""
        pfd = PFDParser().parse(text)
        errors, _warnings = validate_pfd(pfd)

        self.assertEqual(errors, [])
        serialized = pfd.to_pfd()
        reparsed = PFDParser().parse(serialized)
        self.assertEqual(
            reparsed.get_unit('R').reactions[0].parameters,
            {'conversion': '0.5', 'basis': 'A'},
        )

    def test_pfd_validation_rejects_selectivity_and_unbalanced_reactions(self):
        text = """
COMPONENTS:
    H2 | Hydrogen | formula=H2
    O2 | Oxygen | formula=O2
    H2O | Water | formula=H2O

UNIT R
    TYPE: Reactor
    PORTS:
        in: inlet
        out: outlet
    REACTIONS:
        H2 + O2 -> H2O | conversion=0.5, selectivity=0.9
"""
        pfd = PFDParser().parse(text)
        errors, _warnings = validate_pfd(pfd)

        self.assertEqual(len(errors), 1)
        self.assertIn('selectivity is a calculated reactor result', errors[0])

    def test_reactor_recycle_converges_with_deterministic_inlet_basis_extents(self):
        pfd = """
PROCESS: conversion reactor recycle
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
RECYCLE_METHOD: WEGSTEIN
COMPONENTS:
    CO | Carbon monoxide | formula=CO
    H2 | Hydrogen | formula=H2
    CH3OH | Methanol | formula=CH3OH
STREAM Feed : FEED -> M.fresh
    T = 600 [K]
    P = 10 [bar]
    F = 100 [kmol/h]
    x = CO:0.3333333333333333, H2:0.6666666666666667
STREAM Recycle : SP.recycle -> M.recycle
STREAM Mixed : M.out -> R.in
STREAM Reacted : R.out -> SP.in
STREAM Product : SP.product -> PRODUCT
UNIT M
    TYPE: Mixer
    PORTS:
        fresh : inlet
        recycle : inlet
        out : outlet
    PARAMS:
        T_out = 600 [K]
        P = 10 [bar]
UNIT R
    TYPE: Reactor
    PORTS:
        in : inlet
        out : outlet
    PARAMS:
        T_out = 600 [K]
        desired_product = CH3OH
    REACTIONS:
        CO + 2 H2 -> CH3OH | conversion=0.20, basis=CO
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

        self.assertTrue(result.converged)
        self.assertEqual(result.recycle_info['tear_streams'], ['Recycle'])
        product = result.streams['Product'].component_flows()
        recycle = result.streams['Recycle'].component_flows()
        self.assertAlmostEqual(product['CO'], 22.2222222222, places=5)
        self.assertAlmostEqual(product['H2'], 44.4444444444, places=5)
        self.assertAlmostEqual(product['CH3OH'], 11.1111111111, places=5)
        self.assertAlmostEqual(recycle['CO'], product['CO'], places=9)
        self.assertAlmostEqual(recycle['H2'], product['H2'], places=9)
        self.assertAlmostEqual(recycle['CH3OH'], product['CH3OH'], places=9)


if __name__ == '__main__':
    unittest.main()
