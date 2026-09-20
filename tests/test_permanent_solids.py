import unittest

from chemical_properties import ChemicalDatabase
from pfd_parser import PFDParser, PFDValidator, ProcessFlowDiagram, parse_pfd
from phase_behaviors import normalize_phase_behavior
from simulator import Simulator
from thermodynamics_models.base import IdealThermodynamics
from thermodynamics_models.common import ThermodynamicsError
from unit_operations_base import UnitOperationError
from unit_operations import UNIT_CLASSES, Filter
from unit_operations_basic import Cooler, Flash, HeatExchanger, Heater, Mixer, Pump, Splitter
from unit_operations_separation import Flash3
from unit_operations_solids import Crystallizer, LayerCrystallizer


WATER_SALT_COMPONENTS = """
COMPONENTS:
    water | Water | MW=18.0153
    salt | Sodium chloride | CAS=7647-14-5, MW=58.4428, type=permanent_solid, solid_material_form=crystalline, particle_diameter=0.0002, particle_sphericity=0.85
"""


class PermanentSolidParserTests(unittest.TestCase):
    def test_phase_behavior_aliases_and_round_trip(self):
        pfd = parse_pfd(
            """
PROCESS: parser
VERSION: 1.0
COMPONENTS:
    salt | Sodium chloride | type=permanent_solid, particle_diameter_m=0.0001, sphericity=0.9
"""
        )
        component = pfd.components[0]
        self.assertEqual(component.phase_behavior, 'permanent_solid')
        self.assertEqual(component.particle_diameter, 0.0001)
        self.assertEqual(component.particle_sphericity, 0.9)
        reparsed = parse_pfd(pfd.to_pfd()).components[0]
        self.assertEqual(reparsed.phase_behavior, 'permanent_solid')
        self.assertEqual(reparsed.particle_diameter, 0.0001)
        self.assertEqual(reparsed.particle_sphericity, 0.9)
        restored = ProcessFlowDiagram.from_dict(pfd.to_dict()).components[0]
        self.assertEqual(restored.phase_behavior, 'permanent_solid')
        self.assertEqual(restored.solid_material_form, None)

    def test_conventional_is_default_and_aliases_normalize(self):
        self.assertEqual(normalize_phase_behavior(None), 'conventional')
        self.assertEqual(normalize_phase_behavior('permanent-solid'), 'permanent_solid')
        pfd = parse_pfd(
            """
PROCESS: parser
VERSION: 1.0
COMPONENTS:
    water | Water | phase_at_STP=solid
"""
        )
        self.assertIsNone(pfd.components[0].phase_behavior)

    def test_invalid_particle_defaults_are_validation_errors(self):
        parser = PFDParser()
        pfd = parser.parse(
            """
PROCESS: parser
VERSION: 1.0
COMPONENTS:
    salt | Salt | type=permanent_solid, particle_diameter=-1, particle_sphericity=1.1
"""
        )
        errors, _warnings = PFDValidator(pfd).validate()
        self.assertTrue(any('particle_diameter' in error for error in errors))
        self.assertTrue(any('particle_sphericity' in error for error in errors))

    def test_particle_defaults_require_permanent_solid_behavior(self):
        pfd = parse_pfd(
            """
PROCESS: parser
VERSION: 1.0
COMPONENTS:
    water | Water | particle_diameter=0.001
"""
        )
        errors, _warnings = PFDValidator(pfd).validate()
        self.assertTrue(any(
            'require phase_behavior=permanent_solid' in error
            for error in errors
        ))


class PermanentSolidStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = ChemicalDatabase(enable_online=False)

    def make_thermo(self, fluid=('water',)):
        thermo = IdealThermodynamics(list(fluid), self.db)
        thermo.configure_permanent_solids(
            [*fluid, 'NaCl'],
            ['NaCl'],
            {'NaCl': {'diameter_m': 1.0e-4, 'sphericity': 0.9}},
        )
        return thermo

    def assert_component_conservation(self, state):
        recovered = {}
        for phase in state.phase_component_flows().values():
            for component, flow in phase.items():
                recovered[component] = recovered.get(component, 0.0) + flow
        for component, fraction in state.composition.items():
            self.assertAlmostEqual(
                recovered.get(component, 0.0),
                state.F * fraction,
                places=10,
            )

    def test_dry_solid_state_uses_no_fluid_flash(self):
        state = self.make_thermo().calculate_state(
            298.15, 1.0, 5.0, {'NaCl': 1.0}
        )
        self.assertEqual(state.phase_status, 'solid_only')
        self.assertEqual(state.phase_fractions(), {
            'vapor': 0.0, 'liquid1': 0.0, 'liquid2': 0.0, 'solid': 1.0,
        })
        self.assertIsNone(state.fluid_vapor_fraction)
        self.assertEqual(state.solid_component_flows, {'NaCl': 5.0})
        self.assertAlmostEqual(state.Cp, 50.509, places=6)
        self.assertGreater(state.rho, 0.0)
        self.assert_component_conservation(state)

    def test_slurry_fractions_and_properties_use_total_basis(self):
        thermo = self.make_thermo()
        state = thermo.calculate_state(
            298.15, 1.0, 10.0, {'water': 0.8, 'NaCl': 0.2}
        )
        self.assertAlmostEqual(state.effective_liquid1_fraction, 0.8)
        self.assertAlmostEqual(state.solid_fraction, 0.2)
        self.assertAlmostEqual(sum(state.phase_fractions().values()), 1.0)
        self.assertEqual(state.fluid_vapor_fraction, 0.0)
        expected_cp = (
            0.8 * thermo.mixture_Cp({'water': 1.0}, 298.15, 0.0, 1.0)
            + 0.2 * thermo.Cp_solid('NaCl', 298.15)
        )
        self.assertAlmostEqual(state.Cp, expected_cp, places=9)
        self.assert_component_conservation(state)
        properties = {
            source['property']
            for source in thermo.lazy_property_quality_sources()
        }
        self.assertIn('Cp_solid(T)', properties)
        self.assertIn('solid_molar_volume(T)', properties)

    def test_total_quality_converts_to_fluid_quality(self):
        state = self.make_thermo().calculate_state_PQ(
            1.0, 0.4, 10.0, {'water': 0.8, 'NaCl': 0.2}
        )
        self.assertAlmostEqual(state.vapor_fraction, 0.4)
        self.assertAlmostEqual(state.effective_liquid1_fraction, 0.4)
        self.assertAlmostEqual(state.solid_fraction, 0.2)
        self.assertAlmostEqual(state.fluid_vapor_fraction, 0.5)
        self.assert_component_conservation(state)

    def test_quality_above_fluid_fraction_is_rejected(self):
        with self.assertRaisesRegex(
            ThermodynamicsError, 'exceeds the available fluid fraction'
        ):
            self.make_thermo().calculate_state_PQ(
                1.0, 0.9, 10.0, {'water': 0.8, 'NaCl': 0.2}
            )

    def test_dry_solid_pressure_quality_spec_requires_temperature(self):
        with self.assertRaisesRegex(ThermodynamicsError, 'cannot determine'):
            self.make_thermo().calculate_state_PQ(
                1.0, 0.0, 5.0, {'NaCl': 1.0}
            )

    def test_particle_defaults_copy_and_report(self):
        state = self.make_thermo().calculate_state(
            298.15, 1.0, 5.0, {'NaCl': 1.0}
        )
        copied = state.copy()
        copied.solid_particle_properties['NaCl']['diameter_m'] = 2.0e-4
        self.assertEqual(
            state.solid_particle_properties['NaCl']['diameter_m'], 1.0e-4
        )
        payload = state.to_dict()
        self.assertNotIn('fluid_vapor_fraction', payload)
        self.assertEqual(
            payload['solid_particle_properties']['NaCl']['sphericity'], 0.9
        )

    def test_combined_enthalpy_and_entropy_derivatives_match_heat_capacity(self):
        thermo = self.make_thermo()
        composition = {'water': 0.8, 'NaCl': 0.2}
        temperature = 300.0
        step = 1.0e-3
        low = thermo.calculate_state(
            temperature - step, 1.0, 10.0, composition,
            phase='liquid', flash=False,
        )
        center = thermo.calculate_state(
            temperature, 1.0, 10.0, composition,
            phase='liquid', flash=False,
        )
        high = thermo.calculate_state(
            temperature + step, 1.0, 10.0, composition,
            phase='liquid', flash=False,
        )
        water_low = thermo.calculate_state(
            temperature - step, 1.0, 1.0, {'water': 1.0},
            phase='liquid', flash=False,
        )
        water_high = thermo.calculate_state(
            temperature + step, 1.0, 1.0, {'water': 1.0},
            phase='liquid', flash=False,
        )
        dH_dT = (high.H - low.H) / (2.0 * step)
        dS_dT = (high.S - low.S) / (2.0 * step)
        water_dS_dT = (water_high.S - water_low.S) / (2.0 * step)
        self.assertAlmostEqual(dH_dT, center.Cp, delta=2.0e-5)
        self.assertAlmostEqual(
            dS_dT - 0.8 * water_dS_dT,
            0.2 * thermo.Cp_solid('NaCl', temperature) / temperature,
            delta=2.0e-7,
        )

    def test_bulk_density_is_additive_phase_volume(self):
        thermo = self.make_thermo()
        state = thermo.calculate_state(
            298.15, 1.0, 10.0, {'water': 0.8, 'NaCl': 0.2},
            phase='liquid', flash=False,
        )
        liquid_density = thermo.mixture_molar_density(
            {'water': 1.0}, 298.15, 1.0, 0.0, x={'water': 1.0}
        )
        expected_volume = (
            0.8 / liquid_density
            + 0.2 * thermo._solid_molar_volume('NaCl', 298.15)
        )
        self.assertAlmostEqual(1.0 / state.rho, expected_volume, places=12)

    def test_multiple_pure_solids_have_no_artificial_mixing_entropy(self):
        thermo = IdealThermodynamics([], self.db)
        thermo.configure_permanent_solids(
            ['NaCl', 'NaOH'], ['NaCl', 'NaOH']
        )
        state = thermo.calculate_state(
            298.15, 1.0, 4.0, {'NaCl': 0.25, 'NaOH': 0.75}
        )
        expected = (
            0.25 * thermo.entropy_solid('NaCl', 298.15)
            + 0.75 * thermo.entropy_solid('NaOH', 298.15)
        )
        self.assertAlmostEqual(state.S, expected, places=12)
        self.assertEqual(state.solid_component_flows, {
            'NaCl': 1.0,
            'NaOH': 3.0,
        })


class PermanentSolidUnitTests(unittest.TestCase):
    @staticmethod
    def make_direct_states():
        db = ChemicalDatabase(enable_online=False)
        thermo = IdealThermodynamics(['water'], db)
        thermo.configure_permanent_solids(
            ['water', 'NaCl'],
            ['NaCl'],
            {'NaCl': {'diameter_m': 1.0e-4, 'sphericity': 0.9}},
        )
        water = thermo.calculate_state(298.15, 1.0, 8.0, {'water': 1.0})
        salt = thermo.calculate_state(298.15, 1.0, 2.0, {'NaCl': 1.0})
        slurry = thermo.calculate_state(
            298.15, 1.0, 10.0, {'water': 0.8, 'NaCl': 0.2}
        )
        return thermo, water, salt, slurry

    def test_heater_preserves_solid_flow_and_particle_defaults(self):
        pfd = f"""
PROCESS: heater
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
{WATER_SALT_COMPONENTS}
STREAM Feed : FEED -> H.in
    T = 25 [C]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.8, salt:0.2
STREAM Product : H.out -> PRODUCT
UNIT H
    TYPE: Heater
    PORTS:
        in : inlet
        out : outlet
    PARAMS:
        T_out = 50 [C]
"""
        simulator = Simulator.from_string(pfd)
        result = simulator.run()
        inlet = result.streams['Feed']
        outlet = result.streams['Product']
        self.assertEqual(outlet.solid_component_flows, {'salt': 2.0})
        self.assertEqual(outlet.solid_particle_properties, {
            'salt': {'diameter_m': 0.0002, 'sphericity': 0.85}
        })
        expected_duty = outlet.F * outlet.H - inlet.F * inlet.H
        self.assertAlmostEqual(result.units['H'].heat_duty, expected_duty, places=7)
        report = simulator._generate_pfr()
        self.assertIn('fluid_vapor_fraction = 0.0000', report)
        self.assertIn('SOLID_COMPONENT_FLOWS:', report)
        self.assertIn('salt: F=2.0000 [kmol/h]', report)
        self.assertIn('SOLID_PARTICLE_PROPERTIES:', report)
        self.assertIn('diameter=0.0002 [m], sphericity=0.85', report)

    def test_ordinary_flash_routes_all_solids_to_nonvapor_outlet(self):
        pfd = f"""
PROCESS: flash
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
{WATER_SALT_COMPONENTS}
STREAM Feed : FEED -> F.in
    T = 120 [C]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.8, salt:0.2
STREAM Vapor : F.vapor_out -> PRODUCT
STREAM Nonvapor : F.liquid_out -> PRODUCT
UNIT F
    TYPE: Flash
    PORTS:
        in : inlet
        vapor_out : vapor_outlet
        liquid_out : liquid_outlet
    PARAMS:
        T = 120 [C]
        P = 1 [bar]
"""
        result = Simulator.from_string(pfd).run()
        vapor = result.streams['Vapor']
        nonvapor = result.streams['Nonvapor']
        self.assertEqual(vapor.solid_component_flows, {})
        self.assertEqual(nonvapor.solid_component_flows, {'salt': 2.0})
        self.assertEqual(nonvapor.phase_fractions()['solid'], 1.0)
        self.assertAlmostEqual(sum(nonvapor.phase_fractions().values()), 1.0)
        self.assertAlmostEqual(vapor.F + nonvapor.F, 10.0)

    def test_mixer_combines_dry_solid_and_liquid_without_phase_guessing(self):
        thermo, water, salt, _slurry = self.make_direct_states()
        result = Mixer('M', thermo, {'T_out': 298.15, 'P': 1.0}).solve({
            'water': water,
            'salt': salt,
        })
        outlet = result.outlet_streams['out']
        self.assertAlmostEqual(outlet.F, 10.0)
        self.assertEqual(outlet.solid_component_flows, {'NaCl': 2.0})
        self.assertAlmostEqual(outlet.solid_fraction, 0.2)
        self.assertAlmostEqual(result.heat_duty, 0.0, places=6)

    def test_bulk_splitter_preserves_solid_fraction_and_particle_metadata(self):
        thermo, _water, _salt, slurry = self.make_direct_states()
        result = Splitter(
            'S', thermo,
            {'outlets': 'a,b', 'split_frac_a': 0.3},
        ).solve({'in': slurry})
        a = result.outlet_streams['a']
        b = result.outlet_streams['b']
        self.assertAlmostEqual(a.F, 3.0)
        self.assertAlmostEqual(b.F, 7.0)
        self.assertEqual(a.solid_component_flows, {'NaCl': 0.6})
        self.assertEqual(b.solid_component_flows, {'NaCl': 1.4})
        self.assertEqual(a.solid_particle_properties, slurry.solid_particle_properties)

    def test_component_splitter_can_route_permanent_solid_explicitly(self):
        thermo, _water, _salt, slurry = self.make_direct_states()
        splitter = Splitter('S', thermo, {
            'outlets': 'liquid,solids',
            'component_splits': {
                'water': {'liquid': 1.0},
                'NaCl': {'solids': 1.0},
            },
        })
        result = splitter.solve({'in': slurry})
        liquid = result.outlet_streams['liquid']
        solids = result.outlet_streams['solids']
        self.assertEqual(liquid.solid_component_flows, {})
        self.assertEqual(solids.solid_component_flows, {'NaCl': 2.0})
        self.assertEqual(solids.phase_status, 'solid_only')
        self.assertAlmostEqual(liquid.F, 8.0)
        self.assertAlmostEqual(solids.F, 2.0)

    def test_heat_exchanger_explicit_outlet_temperature_supports_slurry(self):
        thermo, _water, _salt, _slurry = self.make_direct_states()
        hot = thermo.calculate_state(
            350.0, 1.0, 10.0, {'water': 0.8, 'NaCl': 0.2}
        )
        cold = thermo.calculate_state(290.0, 1.0, 10.0, {'water': 1.0})
        exchanger = HeatExchanger(
            'HX', thermo, {'T_hot_out': 330.0, 'curve_segments': 4}
        )
        result = exchanger.solve({'hot_in': hot, 'cold_in': cold})
        hot_out = result.outlet_streams['hot_out']
        self.assertEqual(hot_out.solid_component_flows, {'NaCl': 2.0})
        self.assertAlmostEqual(hot_out.T, 330.0)
        self.assertGreater(result.performance['heat_transferred_kW'], 0.0)

    def test_unsupported_pressure_unit_rejects_solid_bearing_feed(self):
        pfd = f"""
PROCESS: pump rejection
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
{WATER_SALT_COMPONENTS}
STREAM Feed : FEED -> P.in
    T = 25 [C]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.8, salt:0.2
STREAM Product : P.out -> PRODUCT
UNIT P
    TYPE: Pump
    PORTS:
        in : inlet
        out : outlet
    PARAMS:
        P_out = 2 [bar]
"""
        result = Simulator.from_string(pfd).run()
        self.assertFalse(result.converged)
        self.assertTrue(any(
            'does not support permanent-solid-bearing' in error
            for error in result.errors
        ))

    def test_flash3_does_not_inherit_ordinary_flash_solid_capability(self):
        self.assertFalse(Flash3.supports_permanent_solids)

    def test_direct_unsupported_unit_api_runs_shared_solid_guard_first(self):
        thermo, _water, _salt, slurry = self.make_direct_states()
        pump = Pump('P', thermo, {'P_out': 2.0})
        with self.assertRaisesRegex(
            UnitOperationError, 'does not support permanent-solid-bearing'
        ):
            pump.solve({'in': slurry})
        flash3 = Flash3('F3', thermo, {'T': 298.15, 'P': 1.0})
        with self.assertRaisesRegex(
            UnitOperationError, 'does not support permanent-solid-bearing'
        ):
            flash3.solve({'in': slurry})

    def test_unit_registry_capability_boundary_is_explicit(self):
        supported = {
            Mixer, Splitter, Heater, Cooler, HeatExchanger, Flash,
            Crystallizer, LayerCrystallizer, Filter,
        }
        for unit_class in set(UNIT_CLASSES.values()):
            with self.subTest(unit_class=unit_class.__name__):
                self.assertEqual(
                    bool(unit_class.supports_permanent_solids),
                    unit_class in supported,
                )

    def test_heat_exchanger_rating_mode_rejects_solid_film_assumption(self):
        db = ChemicalDatabase(enable_online=False)
        thermo = IdealThermodynamics(['water'], db)
        thermo.configure_permanent_solids(['water', 'NaCl'], ['NaCl'])
        hot = thermo.calculate_state(
            350.0, 1.0, 10.0, {'water': 0.8, 'NaCl': 0.2}
        )
        cold = thermo.calculate_state(290.0, 1.0, 10.0, {'water': 1.0})
        exchanger = HeatExchanger('HX', thermo, {'U': 500.0, 'A': 10.0})
        with self.assertRaisesRegex(UnitOperationError, 'slurry/powder'):
            exchanger.solve({'hot_in': hot, 'cold_in': cold})

    def test_eos_backend_contains_only_conventional_components(self):
        pfd = """
PROCESS: eos partition
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: PR
COMPONENTS:
    methane | Methane | MW=16.043
    salt | Sodium chloride | CAS=7647-14-5, MW=58.4428, type=permanent_solid, particle_diameter=0.0001
STREAM Feed : FEED -> H.in
    T = 25 [C]
    P = 5 [bar]
    F = 10 [kmol/h]
    x = methane:0.9, salt:0.1
STREAM Product : H.out -> PRODUCT
UNIT H
    TYPE: Heater
    PORTS:
        in : inlet
        out : outlet
    PARAMS:
        T_out = 35 [C]
"""
        simulator = Simulator.from_string(pfd).initialize()
        self.assertEqual(simulator.thermo.process_components, ['methane', 'salt'])
        self.assertEqual(simulator.thermo.components, ['methane'])
        self.assertEqual(simulator.thermo.permanent_solid_components, ('salt',))
        self.assertEqual(simulator.thermo.cubic.components, ['methane'])

    def test_global_vlle_and_ordinary_flash_retain_two_liquids_plus_solid(self):
        pfd = """
PROCESS: VLLE solid flash
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: UNIFNIST
FLUID_PHASE_MODEL: VLLE
COMPONENTS:
    water | Water | MW=18.0153
    toluene | Toluene | MW=92.1405
    salt | Sodium chloride | CAS=7647-14-5, MW=58.4428, type=permanent_solid, particle_diameter=0.0001
STREAM Feed : FEED -> F.in
    T = 25 [C]
    P = 1 [bar]
    F = 100 [kmol/h]
    x = water:0.45, toluene:0.45, salt:0.10
STREAM Vapor : F.vapor_out -> PRODUCT
STREAM Condensed : F.liquid_out -> PRODUCT
UNIT F
    TYPE: Flash
    PORTS:
        in : inlet
        vapor_out : vapor_outlet
        liquid_out : liquid_outlet
    PARAMS:
        T = 25 [C]
        P = 1 [bar]
"""
        simulator = Simulator.from_string(pfd)
        result = simulator.run()
        condensed = result.streams['Condensed']
        self.assertEqual(simulator.thermo.components, ['water', 'toluene'])
        self.assertEqual(
            simulator.thermo.process_components,
            ['water', 'toluene', 'salt'],
        )
        self.assertGreater(condensed.effective_liquid1_fraction, 0.0)
        self.assertGreater(condensed.liquid2_fraction, 0.0)
        self.assertAlmostEqual(condensed.solid_fraction, 0.1, places=10)
        self.assertAlmostEqual(sum(condensed.phase_fractions().values()), 1.0)
        self.assertEqual(condensed.solid_component_flows, {'salt': 10.0})
        self.assertEqual(condensed.solid_particle_properties, {
            'salt': {'diameter_m': 0.0001, 'sphericity': 1.0}
        })
        self.assertEqual(result.streams['Vapor'].solid_component_flows, {})
        for component in ('water', 'toluene'):
            recovered = sum(
                phase.get(component, 0.0)
                for phase in condensed.phase_component_flows().values()
            )
            self.assertAlmostEqual(
                recovered,
                100.0 * condensed.composition[component],
                places=8,
            )

    def test_solid_only_process_does_not_initialize_selected_fluid_backend(self):
        pfd = """
PROCESS: custom dry solid
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: PR
FLUID_PHASE_MODEL: VLLE
COMPONENTS:
    powder | Custom powder | MW=100, type=permanent_solid, Cp_solid=80, Vm_solid=0.04
STREAM Feed : FEED -> H.in
    T = 25 [C]
    P = 1 [bar]
    F = 2 [kmol/h]
    x = powder:1
STREAM Product : H.out -> PRODUCT
UNIT H
    TYPE: Heater
    PORTS:
        in : inlet
        out : outlet
    PARAMS:
        T_out = 50 [C]
"""
        simulator = Simulator.from_string(pfd)
        result = simulator.run()
        self.assertTrue(result.converged)
        self.assertEqual(simulator.thermo.components, [])
        self.assertEqual(result.streams['Product'].phase_status, 'solid_only')
        self.assertAlmostEqual(result.streams['Product'].Cp, 80.0)
        self.assertAlmostEqual(result.streams['Product'].rho, 25.0)

    def test_steam_backend_solves_only_water_subtotal(self):
        pfd = f"""
PROCESS: steam slurry
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: STEAM
{WATER_SALT_COMPONENTS}
STREAM Feed : FEED -> H.in
    T = 25 [C]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.8, salt:0.2
STREAM Product : H.out -> PRODUCT
UNIT H
    TYPE: Heater
    PORTS:
        in : inlet
        out : outlet
    PARAMS:
        T_out = 80 [C]
"""
        simulator = Simulator.from_string(pfd)
        result = simulator.run()
        outlet = result.streams['Product']
        self.assertEqual(type(simulator.thermo).__name__, 'SteamThermodynamics')
        self.assertEqual(simulator.thermo.components, ['water'])
        self.assertEqual(outlet.solid_component_flows, {'salt': 2.0})
        self.assertAlmostEqual(outlet.solid_fraction, 0.2)
        self.assertIsNotNone(outlet.H)
        self.assertIsNotNone(outlet.Cp)
        self.assertIsNotNone(outlet.rho)

    def test_supported_mixer_splitter_recycle_conserves_solid_flow(self):
        pfd = """
PROCESS: permanent solid recycle
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
RECYCLE_METHOD: WEGSTEIN
COMPONENTS:
    water | Water | MW=18.0153
    salt | Sodium chloride | CAS=7647-14-5, MW=58.4428, type=permanent_solid
STREAM WaterFeed : FEED -> M.water
    T = 25 [C]
    P = 1 [bar]
    F = 8 [kmol/h]
    x = water:1
STREAM SaltFeed : FEED -> M.salt
    T = 25 [C]
    P = 1 [bar]
    F = 2 [kmol/h]
    x = salt:1
STREAM Recycle : SP.recycle -> M.recycle
STREAM Mixed : M.out -> SP.in
STREAM Product : SP.product -> PRODUCT
UNIT M
    TYPE: Mixer
    PORTS:
        water : inlet
        salt : inlet
        recycle : inlet
        out : outlet
    PARAMS:
        T_out = 25 [C]
        P = 1 [bar]
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
        expected = {
            'Recycle': (10.0, 2.0),
            'Mixed': (20.0, 4.0),
            'Product': (10.0, 2.0),
        }
        for stream_name, (flow, solid_flow) in expected.items():
            with self.subTest(stream=stream_name):
                state = result.streams[stream_name]
                self.assertAlmostEqual(state.F, flow, places=9)
                self.assertAlmostEqual(
                    state.solid_component_flows['salt'], solid_flow, places=9
                )


if __name__ == '__main__':
    unittest.main()
