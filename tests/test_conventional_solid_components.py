import unittest

from chemical_properties import ChemicalDatabase
from pfd_parser import ProcessFlowDiagram, parse_pfd
from phase_behaviors import normalize_phase_behavior
from simulator import Simulator
from thermodynamics_models.base import IdealThermodynamics
from thermodynamics_models.common import ThermodynamicsError


class ConventionalSolidParserTests(unittest.TestCase):
    def test_names_and_aliases_normalize_to_conventional_with_solid(self):
        for value in (
            'conventional_with_solid',
            'conventional-with-solid',
            'conventional with solid',
            'three_phase',
            'three-phase',
            'three phase',
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    normalize_phase_behavior(value),
                    'conventional_with_solid',
                )

    def test_three_phase_round_trips_as_canonical_behavior(self):
        pfd = parse_pfd(
            """
PROCESS: parser
VERSION: 1.0
COMPONENTS:
    water | Water | type=three_phase
"""
        )
        self.assertEqual(
            pfd.components[0].phase_behavior,
            'conventional_with_solid',
        )
        reparsed = parse_pfd(pfd.to_pfd())
        self.assertEqual(
            reparsed.components[0].phase_behavior,
            'conventional_with_solid',
        )
        restored = ProcessFlowDiagram.from_dict(pfd.to_dict())
        self.assertEqual(
            restored.components[0].phase_behavior,
            'conventional_with_solid',
        )


class ConventionalSolidStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = ChemicalDatabase(enable_online=False)

    def make_thermo(self, behavior_components=('water',)):
        thermo = IdealThermodynamics(['water'], self.db)
        thermo.configure_permanent_solids(
            ['water'],
            [],
            conventional_solid_components=behavior_components,
        )
        return thermo

    @staticmethod
    def recovered_component_flows(state):
        recovered = {}
        for phase in state.phase_component_flows().values():
            for component, flow in phase.items():
                recovered[component] = recovered.get(component, 0.0) + flow
        return recovered

    def test_regular_state_calculation_does_not_freeze_marked_component(self):
        thermo = self.make_thermo()
        state = thermo.calculate_state(
            250.0,
            1.0,
            10.0,
            {'water': 1.0},
            phase='liquid',
        )
        self.assertEqual(state.solid_component_flows, {})
        self.assertEqual(state.solid_fraction, 0.0)
        self.assertEqual(state.phase_component_flows()['liquid1'], {'water': 10.0})

    def test_explicit_solid_flow_splits_one_component_between_fluid_and_solid(self):
        thermo = self.make_thermo()
        state = thermo.calculate_state_with_solid_flows(
            250.0,
            1.0,
            10.0,
            {'water': 1.0},
            {'water': 2.5},
            phase='liquid',
        )
        self.assertAlmostEqual(state.solid_fraction, 0.25)
        self.assertEqual(state.solid_composition, {'water': 1.0})
        self.assertEqual(state.solid_component_flows, {'water': 2.5})
        self.assertEqual(state.phase_component_flows()['liquid1'], {'water': 7.5})
        self.assertEqual(self.recovered_component_flows(state), {'water': 10.0})
        self.assertIn('water', thermo.components)

    def test_ordinary_conventional_component_rejects_solid_allocation(self):
        thermo = self.make_thermo(behavior_components=())
        with self.assertRaisesRegex(
            ThermodynamicsError,
            'phase_behavior=conventional_with_solid',
        ):
            thermo.calculate_state_with_solid_flows(
                250.0,
                1.0,
                10.0,
                {'water': 1.0},
                {'water': 2.5},
                phase='liquid',
            )

    def test_solid_flow_cannot_exceed_total_component_flow(self):
        thermo = self.make_thermo()
        with self.assertRaisesRegex(ThermodynamicsError, 'exceeds'):
            thermo.calculate_state_with_solid_flows(
                250.0,
                1.0,
                10.0,
                {'water': 1.0},
                {'water': 10.1},
                phase='liquid',
            )

    def test_simulator_keeps_three_phase_component_in_fluid_backend(self):
        simulator = Simulator.from_string(
            """
PROCESS: fluid backend
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
COMPONENTS:
    water | Water | type=three_phase
"""
        )
        simulator.initialize()
        self.assertEqual(simulator.thermo.components, ['water'])
        self.assertEqual(
            simulator.thermo.conventional_solid_components,
            ('water',),
        )
        self.assertEqual(simulator.thermo.permanent_solid_components, ())


if __name__ == '__main__':
    unittest.main()
