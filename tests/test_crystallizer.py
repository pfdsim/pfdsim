import math
import unittest
from types import SimpleNamespace

from chemical_properties import ChemicalDatabase
from dof_analyzer import SpecificationStatus, analyze_dof
from pfd_parser import parse_pfd, validate_pfd
from simulator import Simulator
from thermodynamics_models.base import IdealThermodynamics
from thermodynamics_models.common import P_REF, R, ThermodynamicsError
from thermodynamics_models.factory import create_thermodynamics
from thermodynamics_models.sle import (
    pure_solid_log_saturation_activity,
    solve_pure_solid_sle,
)
from unit_operations_base import UnitOperationError
from unit_operations_solids import Crystallizer


class ConstantCpFusionThermo:
    def __init__(self, liquid_cp, solid_cp, Tm=300.0, Hfus=12.0):
        self.components = ['solute', 'solvent']
        self.props = {
            'solute': SimpleNamespace(Tm=Tm, Hfus=Hfus),
        }
        self.liquid_cp = liquid_cp
        self.solid_cp = solid_cp

    def mark_property_source_context_once(self, *_args, **_kwargs):
        pass

    def _integrate_liquid_cp(self, _component, T1, T2):
        return self.liquid_cp * (T2 - T1) / 1000.0

    def _integrate_solid_cp(self, _component, T1, T2):
        return self.solid_cp * (T2 - T1) / 1000.0

    def _integrate_cp_over_T(self, _component, T1, T2, phase):
        cp = self.liquid_cp if phase == 'liquid' else self.solid_cp
        return cp * math.log(T2 / T1)


class PureSolidSLETests(unittest.TestCase):
    def test_fusion_equation_includes_liquid_solid_cp_difference(self):
        thermo = ConstantCpFusionThermo(
            liquid_cp=90.0,
            solid_cp=50.0,
        )
        temperature = 270.0
        melting_temperature = 300.0
        heat_of_fusion = 12.0
        delta_cp = 40.0
        expected = (
            -1000.0 * heat_of_fusion / R
            * (1.0 / temperature - 1.0 / melting_temperature)
            + delta_cp / R
            * (
                melting_temperature / temperature
                - 1.0
                - math.log(melting_temperature / temperature)
            )
        )
        actual = pure_solid_log_saturation_activity(
            thermo, 'solute', temperature, P_REF
        )
        self.assertAlmostEqual(actual, expected, places=12)

    def test_ideal_binary_sle_matches_analytic_solubility(self):
        thermo = IdealThermodynamics(
            ['water', 'ethanol'],
            ChemicalDatabase(enable_online=False),
        )
        thermo.configure_permanent_solids(
            ['water', 'ethanol'],
            [],
            conventional_solid_components=['water'],
        )
        result = solve_pure_solid_sle(
            thermo,
            250.0,
            1.0,
            {'water': 9.0, 'ethanol': 1.0},
            ['water'],
        )
        expected_x = result.saturation_activities['water']
        expected_liquid_water = expected_x / (1.0 - expected_x)
        expected_solid_water = 9.0 - expected_liquid_water
        self.assertAlmostEqual(
            result.liquid_composition['water'], expected_x, places=10
        )
        self.assertAlmostEqual(
            result.solid_component_flows['water'],
            expected_solid_water,
            places=9,
        )
        self.assertLess(
            abs(result.saturation_residuals['water']), 1.0e-10
        )

    def test_missing_fusion_properties_fail_clearly(self):
        thermo = ConstantCpFusionThermo(90.0, 50.0, Hfus=None)
        with self.assertRaisesRegex(ThermodynamicsError, 'positive Hfus'):
            pure_solid_log_saturation_activity(
                thermo, 'solute', 270.0, P_REF
            )

    def test_activity_model_equilibrates_activity_not_mole_fraction(self):
        thermo = create_thermodynamics(['water', 'ethanol'], 'NRTL')
        thermo.configure_permanent_solids(
            ['water', 'ethanol'],
            [],
            conventional_solid_components=['water'],
        )
        result = solve_pure_solid_sle(
            thermo,
            250.0,
            1.0,
            {'water': 9.0, 'ethanol': 1.0},
            ['water'],
        )
        self.assertAlmostEqual(
            result.liquid_activities['water'],
            result.saturation_activities['water'],
            places=10,
        )
        self.assertNotAlmostEqual(
            result.liquid_composition['water'],
            result.saturation_activities['water'],
            places=3,
        )

    def test_multiple_pure_solids_precipitate_simultaneously(self):
        temperature = 270.0
        melting_temperature = 300.0
        target_solubility = 0.3
        heat_of_fusion = (
            -math.log(target_solubility)
            * R
            / (1.0 / temperature - 1.0 / melting_temperature)
            / 1000.0
        )
        thermo = ConstantCpFusionThermo(
            liquid_cp=60.0,
            solid_cp=60.0,
            Tm=melting_temperature,
            Hfus=heat_of_fusion,
        )
        thermo.components = ['a', 'b', 'solvent']
        thermo.props = {
            component: SimpleNamespace(
                Tm=melting_temperature,
                Hfus=heat_of_fusion,
            )
            for component in ('a', 'b')
        }
        result = solve_pure_solid_sle(
            thermo,
            temperature,
            P_REF,
            {'a': 4.5, 'b': 4.5, 'solvent': 1.0},
            ['a', 'b'],
        )
        self.assertAlmostEqual(
            result.liquid_composition['a'], target_solubility, places=9
        )
        self.assertAlmostEqual(
            result.liquid_composition['b'], target_solubility, places=9
        )
        self.assertGreater(result.solid_component_flows['a'], 0.0)
        self.assertGreater(result.solid_component_flows['b'], 0.0)

    def test_above_melting_candidate_stays_liquid_without_solid_properties(self):
        thermo = ConstantCpFusionThermo(
            liquid_cp=60.0,
            solid_cp=60.0,
            Tm=250.0,
            Hfus=None,
        )
        result = solve_pure_solid_sle(
            thermo,
            260.0,
            P_REF,
            {'solute': 9.0, 'solvent': 1.0},
            ['solute'],
            initial_solid_flows={'solute': 4.0},
        )
        self.assertEqual(result.solid_component_flows, {})
        self.assertEqual(
            result.details['above_melting_candidates'], ['solute']
        )
        self.assertAlmostEqual(result.liquid_composition['solute'], 0.9)

    def test_above_melting_candidate_does_not_block_subcooled_candidate(self):
        temperature = 270.0
        target_solubility = 0.3
        heat_of_fusion = (
            -math.log(target_solubility)
            * R
            / (1.0 / temperature - 1.0 / 300.0)
            / 1000.0
        )
        thermo = ConstantCpFusionThermo(
            liquid_cp=60.0,
            solid_cp=60.0,
        )
        thermo.components = ['cold', 'warm', 'solvent']
        thermo.props = {
            'cold': SimpleNamespace(Tm=300.0, Hfus=heat_of_fusion),
            'warm': SimpleNamespace(Tm=250.0, Hfus=None),
        }
        result = solve_pure_solid_sle(
            thermo,
            temperature,
            P_REF,
            {'cold': 4.5, 'warm': 1.0, 'solvent': 4.5},
            ['cold', 'warm'],
        )
        self.assertGreater(result.solid_component_flows['cold'], 0.0)
        self.assertNotIn('warm', result.solid_component_flows)
        self.assertEqual(result.details['above_melting_candidates'], ['warm'])


class CrystallizerUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thermo = IdealThermodynamics(
            ['water', 'ethanol'],
            ChemicalDatabase(enable_online=False),
        )
        cls.thermo.configure_permanent_solids(
            ['water', 'ethanol'],
            [],
            conventional_solid_components=['water'],
        )

    def make_feed(self):
        return self.thermo.calculate_state(
            280.0,
            1.0,
            10.0,
            {'water': 0.9, 'ethanol': 0.1},
            phase='liquid',
        )

    def test_single_outlet_retains_equilibrium_slurry(self):
        feed = self.make_feed()
        result = Crystallizer('C', self.thermo, {'T': 250.0}).solve({
            'in': feed,
        })
        self.assertEqual(set(result.outlet_streams), {'out'})
        slurry = result.outlet_streams['out']
        self.assertGreater(slurry.solid_component_flows['water'], 0.0)
        self.assertLess(slurry.solid_component_flows['water'], 9.0)
        self.assertAlmostEqual(slurry.F, feed.F)
        self.assertAlmostEqual(slurry.composition['water'], 0.9)
        self.assertAlmostEqual(
            slurry.x['water'],
            result.performance['saturation_activities']['water'],
            places=9,
        )
        self.assertLess(result.heat_duty, 0.0)
        self.assertAlmostEqual(
            result.heat_duty,
            slurry.F * slurry.H - feed.F * feed.H,
            places=7,
        )

    def test_conventional_solid_energy_reference_closes_at_melting_point(self):
        props = self.thermo.props['water']
        melting_temperature = float(props.Tm)
        heat_of_fusion = float(props.Hfus)
        self.assertAlmostEqual(
            self.thermo.enthalpy_liquid('water', melting_temperature)
            - self.thermo.process_solid_enthalpy(
                'water', melting_temperature
            ),
            heat_of_fusion,
            places=10,
        )
        self.assertAlmostEqual(
            (
                self.thermo.entropy_liquid('water', melting_temperature)
                - self.thermo.process_solid_entropy(
                    'water', melting_temperature
                )
            )
            * melting_temperature
            / 1000.0,
            heat_of_fusion,
            places=10,
        )

    def test_preexisting_crystals_redissolve_when_liquid_is_undersaturated(self):
        feed = self.make_feed()
        cold = Crystallizer('C1', self.thermo, {'T': 250.0}).solve({
            'in': feed,
        }).outlet_streams['out']
        warmed = Crystallizer('C2', self.thermo, {'T': 280.0}).solve({
            'in': cold,
        })
        self.assertEqual(
            warmed.outlet_streams['out'].solid_component_flows,
            {},
        )
        self.assertGreater(
            warmed.performance['dissolved_component_flows_kmol_per_h']['water'],
            0.0,
        )

    def test_optional_cake_split_retains_requested_mother_liquor(self):
        feed = self.make_feed()
        slurry_result = Crystallizer(
            'C', self.thermo, {'T': 250.0}
        ).solve({'in': feed})
        split_result = Crystallizer(
            'C',
            self.thermo,
            {'T': 250.0, 'mother_liquor_retention': 0.2},
        ).solve({'in': feed})
        self.assertEqual(
            set(split_result.outlet_streams),
            {'cake', 'mother_liquor'},
        )
        slurry = slurry_result.outlet_streams['out']
        cake = split_result.outlet_streams['cake']
        mother_liquor = split_result.outlet_streams['mother_liquor']
        self.assertEqual(
            cake.solid_component_flows,
            slurry.solid_component_flows,
        )
        slurry_liquid_flow = slurry.F * slurry.effective_liquid1_fraction
        cake_liquid_flow = cake.F - sum(cake.solid_component_flows.values())
        self.assertAlmostEqual(cake_liquid_flow, 0.2 * slurry_liquid_flow)
        self.assertAlmostEqual(
            mother_liquor.F, 0.8 * slurry_liquid_flow
        )
        self.assertAlmostEqual(cake.F + mother_liquor.F, feed.F)
        for component, inlet_flow in feed.component_flows().items():
            outlet_flow = sum(
                stream.component_flows().get(component, 0.0)
                for stream in split_result.outlet_streams.values()
            )
            self.assertAlmostEqual(outlet_flow, inlet_flow, places=10)
        self.assertAlmostEqual(
            split_result.heat_duty,
            slurry_result.heat_duty,
            places=7,
        )

    def test_mass_retention_rate_uses_liquor_to_dry_crystal_mass_ratio(self):
        feed = self.make_feed()
        result = Crystallizer(
            'C',
            self.thermo,
            {'T': 250.0, 'mother_liquor_retention_rate': 0.05},
        ).solve({'in': feed})
        cake = result.outlet_streams['cake']
        mother_liquor = result.outlet_streams['mother_liquor']
        crystal_mass = sum(
            flow * self.thermo.props[component].MW
            for component, flow in cake.solid_component_flows.items()
        )
        cake_mass = cake.mass_flow()
        retained_liquor_mass = cake_mass - crystal_mass
        self.assertAlmostEqual(
            retained_liquor_mass / crystal_mass, 0.05, places=10
        )
        self.assertAlmostEqual(
            result.performance[
                'mother_liquor_retention_rate_kg_per_kg_crystals'
            ],
            0.05,
            places=10,
        )
        self.assertAlmostEqual(cake.F + mother_liquor.F, feed.F)
        self.assertAlmostEqual(
            result.heat_duty,
            sum(stream.F * stream.H for stream in result.outlet_streams.values())
            - feed.F * feed.H,
            places=7,
        )

    def test_retention_bases_are_mutually_exclusive(self):
        with self.assertRaisesRegex(UnitOperationError, 'only one'):
            Crystallizer(
                'C',
                self.thermo,
                {
                    'T': 250.0,
                    'mother_liquor_retention_fraction': 0.1,
                    'mother_liquor_retention_rate': 0.05,
                },
            ).solve({'in': self.make_feed()})

    def test_mass_retention_rate_rejects_unavailable_liquor(self):
        with self.assertRaisesRegex(UnitOperationError, 'only .* is available'):
            Crystallizer(
                'C',
                self.thermo,
                {'T': 250.0, 'mother_liquor_retention_rate': 100.0},
            ).solve({'in': self.make_feed()})

    def test_positive_mass_retention_rate_requires_crystals(self):
        with self.assertRaisesRegex(UnitOperationError, 'no conventional crystals'):
            Crystallizer(
                'C',
                self.thermo,
                {'T': 280.0, 'mother_liquor_retention_rate': 0.05},
            ).solve({'in': self.make_feed()})

    def test_retention_parameter_is_bounded(self):
        with self.assertRaisesRegex(UnitOperationError, 'between 0 and 1'):
            Crystallizer(
                'C',
                self.thermo,
                {'T': 250.0, 'mother_liquor_retention': 1.1},
            ).solve({'in': self.make_feed()})

    def test_feed_requires_marked_crystallizable_component(self):
        thermo = IdealThermodynamics(
            ['water', 'ethanol'],
            ChemicalDatabase(enable_online=False),
        )
        feed = thermo.calculate_state(
            280.0,
            1.0,
            10.0,
            {'water': 0.9, 'ethanol': 0.1},
            phase='liquid',
        )
        with self.assertRaisesRegex(
            UnitOperationError, 'conventional_with_solid'
        ):
            Crystallizer('C', thermo, {'T': 250.0}).solve({'in': feed})


class CrystallizerPFDTests(unittest.TestCase):
    def test_compact_ports_and_dof(self):
        pfd = parse_pfd(
            """
UNIT C : Crystallizer
    T = 250 [K]
    mother_liquor_retention = 0.1
STREAM Feed : -> C.solution
STREAM Cake : C.crystals
STREAM Liquor : C.filtrate
"""
        )
        self.assertEqual(pfd.streams[0].destination.port_id, 'in')
        self.assertEqual(pfd.streams[1].source.port_id, 'cake')
        self.assertEqual(pfd.streams[2].source.port_id, 'mother_liquor')
        self.assertEqual(validate_pfd(pfd), ([], []))
        result = analyze_dof(pfd).unit_results[0]
        self.assertEqual(result.status, SpecificationStatus.OK)

    def test_cake_ports_require_retention_parameter(self):
        pfd = parse_pfd(
            """
UNIT C : Crystallizer
    T = 250 [K]
STREAM Feed : -> C.in
STREAM Cake : C.cake
STREAM Liquor : C.mother_liquor
"""
        )
        result = analyze_dof(pfd).unit_results[0]
        self.assertEqual(result.status, SpecificationStatus.UNDER_SPECIFIED)
        self.assertIn('mother_liquor_retention', result.message)

    def test_retention_fraction_and_mass_rate_are_over_specified(self):
        pfd = parse_pfd(
            """
UNIT C : Crystallizer
    T = 250 [K]
    mother_liquor_retention_fraction = 0.1
    mother_liquor_retention_rate = 0.05
STREAM Feed : -> C.in
STREAM Cake : C.cake
STREAM Liquor : C.mother_liquor
"""
        )
        result = analyze_dof(pfd).unit_results[0]
        self.assertEqual(result.status, SpecificationStatus.OVER_SPECIFIED)
        self.assertIn('only one', result.message)

    def test_complete_pfd_runs_with_cake_and_mother_liquor_outlets(self):
        simulator = Simulator.from_string(
            """
PROCESS: Ice crystallizer
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
COMPONENTS:
    water | Water | type=three_phase
    ethanol | Ethanol
STREAM Feed : FEED -> C.in
    T = 6.85 [C]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.9, ethanol:0.1
STREAM Cake : C.cake -> PRODUCT
STREAM MotherLiquor : C.mother_liquor -> PRODUCT
UNIT C
    TYPE: Crystallizer
    PORTS:
        in : inlet
        cake : solid_outlet
        mother_liquor : liquid_outlet
    PARAMS:
        temperature = -23.15 [C]
        mother_liquor_retention_rate = 0.1
"""
        )
        result = simulator.run()
        self.assertTrue(result.converged, result.errors)
        self.assertGreater(
            result.streams['Cake'].solid_component_flows['water'], 0.0
        )
        self.assertEqual(
            result.streams['MotherLiquor'].solid_component_flows,
            {},
        )
        self.assertAlmostEqual(
            result.streams['Cake'].F + result.streams['MotherLiquor'].F,
            result.streams['Feed'].F,
        )
        report = simulator._generate_pfr()
        self.assertIn('type = Crystallizer', report)
        self.assertEqual(
            result.units['C'].performance['mother_liquor_retention_basis'],
            'kg_liquor_per_kg_crystals',
        )
        self.assertAlmostEqual(
            result.units['C'].performance[
                'mother_liquor_retention_rate_kg_per_kg_crystals'
            ],
            0.1,
        )
        self.assertIn(
            'mother_liquor_retention_rate_kg_per_kg_crystals = 0.1000',
            report,
        )
        self.assertIn('SOLID_COMPONENT_FLOWS:', report)


if __name__ == '__main__':
    unittest.main()
