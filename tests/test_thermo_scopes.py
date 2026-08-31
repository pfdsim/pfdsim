import unittest
from unittest.mock import patch

from flowsheet_solver import FlowsheetSolver
from pfd_parser import ParseError, PFDParser, ProcessFlowDiagram, validate_pfd
from simulator import SimulationError, Simulator
from thermodynamics_models.base import StreamState


class ThermodynamicScopeTests(unittest.TestCase):
    @staticmethod
    def _scope_header(extra_scopes=""):
        return f"""
PROCESS: thermodynamic scope test
THERMO_METHOD: NRTL
ONLINE_LOOKUP: false
THERMO_SCOPES:
    inherited | method=NRTL, inherit=global
    isolated | method=NRTL
    child | method=NRTL, inherit=inherited
{extra_scopes}COMPONENTS:
    H2O | Water | MW=18.015
    ETOH | Ethanol | MW=46.069
"""

    @staticmethod
    def _heater_chain_pfd():
        return """
PROCESS: scoped heater chain
THERMO_METHOD: IDEAL
ONLINE_LOOKUP: false
THERMO_SCOPES:
    extraction | method=NRTL
COMPONENTS:
    H2O | Water | MW=18.015
    ETOH | Ethanol | MW=46.069
INTERACTION_PARAMETERS:
    H2O/ETOH | model=NRTL, scope=extraction, a12=300, a21=-100, alpha=0.3
STREAM Feed : FEED -> H-1.in
    T = 25 [C]
    P = 1 [bar]
    F = 100 [kmol/h]
    x = H2O:0.6, ETOH:0.4
STREAM Scoped-Internal : H-1.out -> H-2.in
STREAM Product : H-2.out -> PRODUCT
UNIT H-1 : Heater
    T = 40 [C]
    thermo_scope = extraction
UNIT H-2 : Heater
    T = 50 [C]
    thermo_scope = extraction
"""

    def test_parser_round_trip_preserves_scopes_inheritance_and_namespaced_records(self):
        text = self._scope_header() + """
INTERACTION_PARAMETERS:
    H2O/ETOH | model=NRTL, a12=111, a21=222
    H2O/ETOH | model=NRTL, scope=inherited, a12=333, a21=444
    H2O/ETOH | model=NRTL, scope=isolated, a12=555, a21=666
INTERACTION_ESTIMATION:
    NRTL | source=UNIFDMD, Tmin=293.15, Tmax=373.15
    H2O/ETOH | model=NRTL, scope=child, policy=always
"""
        pfd = PFDParser().parse(text)
        errors, _warnings = validate_pfd(pfd)
        self.assertEqual(errors, [])
        self.assertEqual(
            [(scope.name, scope.method, scope.inherit) for scope in pfd.thermo_scopes],
            [
                ('inherited', 'NRTL', 'global'),
                ('isolated', 'NRTL', None),
                ('child', 'NRTL', 'inherited'),
            ],
        )
        self.assertEqual(
            [record.scope for record in pfd.interaction_parameters],
            [None, 'inherited', 'isolated'],
        )
        self.assertEqual(
            [record.scope for record in pfd.interaction_estimation],
            [None, 'child'],
        )

        reparsed = PFDParser().parse(pfd.to_pfd())
        self.assertEqual(reparsed.to_dict(), pfd.to_dict())
        rebuilt = ProcessFlowDiagram.from_dict(pfd.to_dict())
        self.assertEqual(rebuilt.to_dict(), pfd.to_dict())

    def test_scope_validation_rejects_bad_names_references_cycles_and_units(self):
        cases = {
            'reserved': (
                "    global | method=NRTL\n",
                "reserved scope 'global'",
            ),
            'duplicate': (
                "    duplicate | method=NRTL\n    duplicate | method=UNIQUAC\n",
                'Duplicate thermodynamic scope',
            ),
            'unknown parent': (
                "    orphan | method=NRTL, inherit=missing\n",
                'inherits unknown scope',
            ),
            'cycle': (
                "    first | method=NRTL, inherit=second\n"
                "    second | method=NRTL, inherit=first\n",
                'inheritance cycle',
            ),
            'unsupported method': (
                "    badmethod | method=NOT_A_METHOD\n",
                'Unsupported thermodynamics method',
            ),
        }
        for label, (rows, expected) in cases.items():
            with self.subTest(label=label):
                pfd = PFDParser().parse(
                    "PROCESS: bad scope\nTHERMO_SCOPES:\n"
                    + rows
                    + "COMPONENTS:\n    H2O | Water | MW=18.015\n"
                )
                errors, _warnings = validate_pfd(pfd)
                self.assertIn(expected, '\n'.join(errors))

        with self.assertRaisesRegex(ParseError, 'must be declared earlier'):
            PFDParser().parse("""
PROCESS: unknown scope references
COMPONENTS:
    H2O | Water | MW=18.015
    ETOH | Ethanol | MW=46.069
INTERACTION_PARAMETERS:
    H2O/ETOH | model=NRTL, scope=missing, a12=1, a21=2
STREAM Feed : FEED -> H-1.in
    T = 25 [C]
    P = 1 [bar]
    F = 1 [kmol/h]
    x = H2O:1
STREAM Product : H-1.out -> PRODUCT
UNIT H-1 : Heater
    T = 30 [C]
    thermo_scope = missing
""")

    def test_same_method_scopes_are_distinct_and_inheritance_is_pairwise(self):
        pfd = self._scope_header() + """
INTERACTION_PARAMETERS:
    H2O/ETOH | model=NRTL, a12=111, a21=222
    H2O/ETOH | model=NRTL, scope=child, a12=333, a21=444
"""
        simulator = Simulator.from_string(pfd).initialize()
        packages = simulator.thermo_packages
        self.assertEqual(set(packages), {'global', 'inherited', 'isolated', 'child'})
        self.assertEqual(len({id(package) for package in packages.values()}), 4)

        global_record = packages['global']._nrtl_interaction_for_components(
            'H2O', 'ETOH'
        )
        inherited_record = packages['inherited']._nrtl_interaction_for_components(
            'H2O', 'ETOH'
        )
        isolated_record = packages['isolated']._nrtl_interaction_for_components(
            'H2O', 'ETOH'
        )
        child_record = packages['child']._nrtl_interaction_for_components(
            'H2O', 'ETOH'
        )
        self.assertEqual(global_record['a12_cal_per_mol'], 111.0)
        self.assertEqual(inherited_record['a12_cal_per_mol'], 111.0)
        self.assertNotEqual(isolated_record.get('a12_cal_per_mol'), 111.0)
        self.assertEqual(child_record['a12_cal_per_mol'], 333.0)

    def test_method_specific_component_overrides_are_available_to_named_scope(self):
        simulator = Simulator.from_string("""
PROCESS: scoped UNIQUAC component parameters
THERMO_METHOD: IDEAL
ONLINE_LOOKUP: false
THERMO_SCOPES:
    liquid | method=UNIQUAC
COMPONENTS:
    H2O | Water | MW=18.015, uniquac_r=9.0, uniquac_q=8.0
    ETOH | Ethanol | MW=46.069
""").initialize()
        self.assertEqual(
            simulator.thermo_packages['liquid']._uniquac_rq('H2O'),
            (9.0, 8.0),
        )

    def test_estimation_rules_follow_scope_inheritance_without_leaking(self):
        pfd = self._scope_header() + """
INTERACTION_ESTIMATION:
    NRTL | source=UNIFDMD, Tmin=293.15, Tmax=373.15
    H2O/ETOH | model=NRTL, scope=child, policy=always
"""
        captured = []

        def capture_rules(
            destination_thermo,
            destination,
            rules,
            existing_interaction,
            unifac_groups=None,
            explicit_pair_keys=None,
        ):
            captured.append([dict(rule) for rule in rules or []])
            return [], {}

        with patch(
            'thermodynamics_models.interaction_estimation.estimate_missing_interactions',
            side_effect=capture_rules,
        ):
            Simulator.from_string(pfd).initialize()

        self.assertEqual(len(captured), 3)
        global_rules, inherited_rules, child_rules = captured
        self.assertEqual(len(global_rules), 1)
        self.assertEqual(len(inherited_rules), 1)
        self.assertEqual(len(child_rules), 2)
        self.assertTrue(any(
            rule.get('component1') == 'H2O'
            and rule.get('policy') == 'always'
            for rule in child_rules
        ))

    def test_duplicate_pair_is_rejected_only_within_the_same_scope(self):
        pfd = self._scope_header() + """
INTERACTION_PARAMETERS:
    H2O/ETOH | model=NRTL, scope=isolated, a12=1, a21=2
    H2O/ETOH | model=NRTL, scope=isolated, a12=3, a21=4
"""
        with self.assertRaisesRegex(
            SimulationError,
            'Duplicate NRTL INTERACTION_PARAMETERS override',
        ):
            Simulator.from_string(pfd).initialize()

    def test_scoped_liquid_viscosity_override_is_isolated(self):
        pfd = """
PROCESS: scoped viscosity
THERMO_METHOD: IDEAL
ONLINE_LOOKUP: false
THERMO_SCOPES:
    viscous | method=IDEAL
COMPONENTS:
    H2O | Water | MW=18.015
    ETOH | Ethanol | MW=46.069
INTERACTION_PARAMETERS:
    H2O/ETOH | model=LIQUID_VISCOSITY, scope=viscous, form=grunberg_nissan, G=2.0
"""
        simulator = Simulator.from_string(pfd).initialize()
        composition = {'H2O': 0.5, 'ETOH': 0.5}
        global_mu = simulator.thermo_packages['global'].mixture_viscosity(
            composition, 298.15, 1.0, vapor_fraction=0.0
        )
        scoped_mu = simulator.thermo_packages['viscous'].mixture_viscosity(
            composition, 298.15, 1.0, vapor_fraction=0.0
        )
        self.assertGreater(abs(scoped_mu / global_mu - 1.0), 0.05)

    def test_scoped_lle_unit_uses_its_method_instead_of_global_method(self):
        scoped = """
PROCESS: scoped decanter validation
THERMO_METHOD: IDEAL
ONLINE_LOOKUP: false
THERMO_SCOPES:
    extraction | method=UNIFAC
COMPONENTS:
    H2O | Water | MW=18.015
    BUOH | 1-Butanol | MW=74.12
STREAM Feed : FEED -> D-1.in
    T = 25 [C]
    P = 1 [bar]
    F = 100 [kmol/h]
    x = H2O:0.7, BUOH:0.3
STREAM Light : D-1.light -> PRODUCT
STREAM Heavy : D-1.heavy -> PRODUCT
UNIT D-1 : Decanter
    T = 25 [C]
    P = 1 [bar]
    thermo_scope = extraction
"""
        simulator = Simulator.from_string(scoped)
        self.assertEqual(simulator.pfd.get_unit('D-1').unit_type, 'Decanter')

        unscoped = scoped.replace(
            '    thermo_scope = extraction\n',
            '',
        )
        with self.assertRaisesRegex(SimulationError, 'requires LLE capability'):
            Simulator.from_string(unscoped)

    def test_optional_backends_are_prepared_only_for_units_in_the_scope(self):
        from thermodynamics_models.nrtl_uniquac import NRTLThermodynamics

        pfd = """
PROCESS: scoped backend preparation
THERMO_METHOD: NRTL
ONLINE_LOOKUP: false
THERMO_SCOPES:
    extraction | method=NRTL
COMPONENTS:
    H2O | Water | MW=18.015
    BUOH | 1-Butanol | MW=74.12
STREAM Feed : FEED -> D-1.in
    T = 25 [C]
    P = 1 [bar]
    F = 100 [kmol/h]
    x = H2O:0.7, BUOH:0.3
STREAM Light : D-1.light -> PRODUCT
STREAM Heavy : D-1.heavy -> PRODUCT
UNIT D-1 : Decanter
    T = 25 [C]
    P = 1 [bar]
    thermo_scope = extraction
"""
        calls = []

        def capture_backend_needs(self, *, need_lle=False, need_vlle=False):
            calls.append((self, need_lle, need_vlle))

        with patch.object(
            NRTLThermodynamics,
            'prepare_compiled_backends',
            new=capture_backend_needs,
        ):
            simulator = Simulator.from_string(pfd).initialize()

        by_package = {
            id(package): (need_lle, need_vlle)
            for package, need_lle, need_vlle in calls
        }
        self.assertEqual(by_package[id(simulator.thermo_packages['global'])], (False, False))
        self.assertEqual(by_package[id(simulator.thermo_packages['extraction'])], (True, False))

    def test_topology_converts_only_entry_and_exit_and_closes_energy(self):
        simulator = Simulator.from_string(self._heater_chain_pfd()).initialize()
        solver = simulator.solver
        self.assertEqual(
            solver.thermo_scope_boundaries,
            {
                'Feed': ('global', 'extraction'),
                'Product': ('extraction', 'global'),
            },
        )
        self.assertNotIn('Scoped-Internal', solver.thermo_scope_boundaries)
        self.assertIs(solver.units['H-1'].thermo, simulator.thermo_packages['extraction'])
        self.assertIs(solver.units['H-2'].thermo, simulator.thermo_packages['extraction'])

        with patch.object(
            solver,
            '_state_in_thermo_scope',
            wraps=solver._state_in_thermo_scope,
        ) as rehydrate:
            result = simulator.run()

        self.assertTrue(result.converged)
        # Two topology boundaries, with source- and destination-package
        # property views evaluated at each boundary.
        self.assertEqual(rehydrate.call_count, 4)
        self.assertEqual(
            [item['stream_id'] for item in result.thermo_scope_corrections],
            ['Feed', 'Product'],
        )
        self.assertLess(result.energy_balance_error, 1e-12)
        self.assertEqual(result.streams['Product'].thermo_scope, 'global')
        payload = result.to_dict()
        self.assertEqual(
            payload['thermo_scope_enthalpy_correction'],
            result.thermo_scope_enthalpy_correction,
        )
        results_payload = simulator.get_results_dict()
        self.assertEqual(
            results_payload['metadata']['thermo_scopes']['extraction'],
            'NRTL',
        )
        self.assertEqual(
            len(results_payload['thermo_scope_corrections']),
            2,
        )
        pfr = simulator._generate_pfr()
        self.assertIn('THERMO_SCOPE extraction: NRTL', pfr)
        self.assertIn('THERMO_SCOPE_CORRECTIONS:', pfr)

    def test_boundary_recomputes_density_and_viscosity_in_destination_scope(self):
        simulator = Simulator.from_string(self._heater_chain_pfd()).initialize()
        solver = simulator.solver
        global_thermo = simulator.thermo_packages['global']

        vapor = global_thermo.calculate_state(
            400.0,
            1.0,
            10.0,
            {'H2O': 0.2, 'ETOH': 0.8},
            phase='vapor',
            flash=False,
            include=('H', 'S', 'Cp', 'rho', 'mu'),
        )
        vapor.thermo_scope = 'global'
        vapor.rho = -1.0
        vapor.mu = 123.0
        scoped_vapor = solver._transition_stream_state(
            'vapor-probe', vapor, 'extraction'
        )
        self.assertGreater(scoped_vapor.rho, 0.0)
        self.assertNotEqual(scoped_vapor.rho, -1.0)
        self.assertNotEqual(scoped_vapor.mu, 123.0)

        liquid = global_thermo.calculate_state(
            298.15,
            1.0,
            10.0,
            {'H2O': 0.8, 'ETOH': 0.2},
            phase='liquid',
            flash=False,
            include=('H', 'S', 'Cp', 'rho', 'mu'),
        )
        liquid.thermo_scope = 'global'
        liquid.rho = -2.0
        liquid.mu = 456.0
        scoped_liquid = solver._transition_stream_state(
            'liquid-probe', liquid, 'extraction'
        )
        self.assertNotEqual(scoped_liquid.rho, -2.0)
        self.assertNotEqual(scoped_liquid.mu, 456.0)

        first = dict(solver._thermo_scope_corrections['liquid-probe'])
        liquid.F = 20.0
        solver._transition_stream_state('liquid-probe', liquid, 'extraction')
        self.assertEqual(len([
            key for key in solver._thermo_scope_corrections
            if key == 'liquid-probe'
        ]), 1)
        self.assertNotEqual(
            first['flow_kmol_per_h'],
            solver._thermo_scope_corrections['liquid-probe']['flow_kmol_per_h'],
        )

        scoped_liquid.thermo_scope = 'extraction'
        self.assertIs(
            solver._transition_stream_state(
                'liquid-probe', scoped_liquid, 'extraction'
            ),
            scoped_liquid,
        )
        self.assertNotIn('liquid-probe', solver._thermo_scope_corrections)

    def test_only_explicitly_constrained_phase_is_preserved_at_boundary(self):
        unconstrained = StreamState(
            T=298.15,
            P=1.0,
            F=1.0,
            composition={'H2O': 1.0},
            vapor_fraction=0.0,
            phase_status='single_liquid',
            phase_stability='stable_single_liquid',
        )
        forced = unconstrained.copy()
        forced.phase_status = 'forced_liquid'
        forced.phase_stability = 'explicit_phase_constraint'
        self.assertIsNone(FlowsheetSolver._scope_phase_constraint(unconstrained))
        self.assertEqual(
            FlowsheetSolver._scope_phase_constraint(forced),
            'liquid',
        )

    def test_no_declared_scopes_preserves_single_global_package(self):
        simulator = Simulator.from_string("""
PROCESS: global-only
THERMO_METHOD: IDEAL
ONLINE_LOOKUP: false
COMPONENTS:
    H2O | Water | MW=18.015
""").initialize()
        self.assertEqual(simulator.thermo_packages, {'global': simulator.thermo})
        self.assertEqual(simulator.solver.thermo_scope_boundaries, {})

    def test_recycle_inside_one_scope_has_only_external_boundary_corrections(self):
        pfd = """
PROCESS: scoped recycle
RECYCLE_METHOD: DIRECT
TEAR_STREAMS: Recycle
THERMO_SCOPES:
    loop | method=IDEAL
COMPONENTS:
    H2O | Water | MW=18.015
STREAM Fresh : FEED -> M-1.fresh
    T = 25 [C]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = H2O:1
STREAM Mixed : M-1.out -> S-1.in
STREAM Product : S-1.product -> PRODUCT
STREAM Recycle : S-1.recycle -> M-1.recycle
    T = 25 [C]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = H2O:1
UNIT M-1 : Mixer
    mode = adiabatic
    thermo_scope = loop
UNIT S-1 : Splitter
    split_fracs = product:0.5,recycle:0.5
    thermo_scope = loop
"""
        result = Simulator.from_string(pfd).run(
            max_iterations=10,
            tolerance=1e-7,
        )
        self.assertTrue(result.converged, result.warnings)
        self.assertEqual(
            [record['stream_id'] for record in result.thermo_scope_corrections],
            ['Fresh', 'Product'],
        )
        self.assertNotIn(
            'Recycle',
            [record['stream_id'] for record in result.thermo_scope_corrections],
        )
        self.assertLess(result.energy_balance_error, 1e-12)


if __name__ == '__main__':
    unittest.main()
