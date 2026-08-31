import unittest

from pfd_parser import PFDParser, validate_pfd
from simulator import Simulator
from thermodynamics import create_thermodynamics
from unit_operations_base import UnitOperationError
from unit_operations_reactors import KineticsCSTR


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


class ThermalCSTRUnitTests(unittest.TestCase):
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

    def solve(self, **extra):
        params = {
            'volume': 5.0,
            'phase': 'vapor',
            'mode': 'adiabatic',
            'T_min': 300.0,
            'T_max': 900.0,
            'thermal_scan_points': 61,
            'reactions': [reaction_definition()],
        }
        params.update(extra)
        return KineticsCSTR('CSTR-1', self.thermo, params).solve({'in': self.feed})

    def test_adiabatic_and_duty_modes_close_exact_energy_balance(self):
        adiabatic = self.solve()
        heated = self.solve(mode='duty', Q=1.0, __unit__Q='kW')

        self.assertEqual(adiabatic.heat_duty, 0.0)
        self.assertGreater(adiabatic.outlet_streams['out'].T, self.feed.T)
        self.assertAlmostEqual(heated.heat_duty, 3600.0, places=10)
        self.assertGreater(
            heated.outlet_streams['out'].T,
            adiabatic.outlet_streams['out'].T,
        )
        for result in (adiabatic, heated):
            self.assertEqual(len(result.performance['thermal_roots']), 1)
            self.assertTrue(
                result.performance['thermal_roots'][0]['thermally_stable']
            )
            self.assertLess(
                abs(result.performance['energy_balance_residual_kW']),
                1.0e-7,
            )

    def test_jacketed_UA_and_U_area_are_equivalent(self):
        explicit_ua = self.solve(
            mode='jacketed',
            UA=100.0,
            __unit__UA='W/K',
            T_jacket=450.0,
        )
        factored = self.solve(
            mode='jacketed',
            U=50.0,
            __unit__U='W/m2/K',
            A_heat=2.0,
            __unit__A_heat='m2',
            T_jacket=450.0,
        )

        self.assertEqual(explicit_ua.performance['UA_W_per_K'], 100.0)
        self.assertEqual(factored.performance['UA_W_per_K'], 100.0)
        self.assertAlmostEqual(
            explicit_ua.outlet_streams['out'].T,
            factored.outlet_streams['out'].T,
            places=10,
        )
        self.assertAlmostEqual(
            explicit_ua.heat_duty,
            factored.heat_duty,
            places=7,
        )
        self.assertLess(explicit_ua.heat_duty, 0.0)
        self.assertGreater(explicit_ua.outlet_streams['out'].T, 450.0)

    def multiplicity_params(self, branch=None):
        params = {
            'volume': 10.0,
            'phase': 'vapor',
            'mode': 'jacketed',
            'UA': 10.0,
            'T_jacket': 350.0,
            'T_min': 250.0,
            'T_max': 1500.0,
            'thermal_scan_points': 81,
            'reactions': [reaction_definition(A=1.0e5, Ea=60000.0)],
        }
        if branch is not None:
            params['thermal_branch'] = branch
        return params

    def test_multiple_steady_states_require_explicit_branch(self):
        feed = self.thermo.calculate_state(
            350.0,
            2.0,
            10.0,
            {'C2H4O': 1.0},
            phase='vapor',
            flash=False,
        )
        with self.assertRaisesRegex(UnitOperationError, 'multiple thermal steady states'):
            KineticsCSTR(
                'CSTR-M', self.thermo, self.multiplicity_params()
            ).solve({'in': feed})

        results = {
            branch: KineticsCSTR(
                f'CSTR-{branch}',
                self.thermo,
                self.multiplicity_params(branch),
            ).solve({'in': feed})
            for branch in ('lowest', 'nearest_inlet', 'highest')
        }
        roots = results['lowest'].performance['thermal_roots']
        self.assertEqual(len(roots), 3)
        self.assertEqual(
            [root['thermally_stable'] for root in roots],
            [True, False, True],
        )
        self.assertAlmostEqual(
            results['lowest'].outlet_streams['out'].T,
            roots[0]['temperature_K'],
            places=7,
        )
        self.assertAlmostEqual(
            results['nearest_inlet'].outlet_streams['out'].T,
            roots[0]['temperature_K'],
            places=7,
        )
        self.assertAlmostEqual(
            results['highest'].outlet_streams['out'].T,
            roots[-1]['temperature_K'],
            places=7,
        )

    def test_thermal_specification_conflicts_and_unbracketed_bounds_fail(self):
        cases = (
            ({'mode': 'adiabatic', 'T': 500.0}, 'cannot specify T'),
            ({'mode': 'duty'}, 'requires duty'),
            ({'mode': 'jacketed', 'UA': 10.0}, 'requires UA'),
            ({
                'mode': 'jacketed', 'UA': 10.0, 'U': 10.0,
                'A_heat': 1.0, 'T_jacket': 450.0,
            }, 'not both'),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(UnitOperationError, message):
                    self.solve(**overrides)
        with self.assertRaisesRegex(UnitOperationError, 'could not bracket'):
            self.solve(T_min=300.0, T_max=400.0)

    def test_thermal_repeated_solve_is_deterministic(self):
        params = {
            'volume': 5.0,
            'phase': 'vapor',
            'mode': 'duty',
            'Q': 1.0,
            '__unit__Q': 'kW',
            'T_min': 300.0,
            'T_max': 900.0,
            'thermal_scan_points': 61,
            'reactions': [reaction_definition()],
        }
        unit = KineticsCSTR('CSTR-D', self.thermo, params)
        first = unit.solve({'in': self.feed})
        second = unit.solve({'in': self.feed})
        self.assertEqual(first.outlet_streams['out'].to_dict(), second.outlet_streams['out'].to_dict())
        self.assertEqual(first.performance, second.performance)

    def test_reversible_PR_adiabatic_cstr_couples_K_rate_and_enthalpy(self):
        components = ['CO', 'H2', 'CH3OH']
        thermo = create_thermodynamics(components, 'PR')
        feed = thermo.calculate_state(
            523.15,
            80.0,
            3.0,
            {'CO': 1.0 / 3.0, 'H2': 2.0 / 3.0},
            phase='vapor',
            flash=False,
        )
        reaction = {
            'equation': 'CO + 2 H2 <=> CH3OH',
            'A': 1.0,
            'Ea': 0.0,
            'Ea_unit': 'J/mol',
            'rate_basis': 'activity',
            'concentration_unit': 'kmol/m3',
            'pressure_unit': 'bar',
            'rate_unit': 'kmol/m3/h',
        }
        result = KineticsCSTR('CSTR-PR', thermo, {
            'volume': 0.1,
            'mode': 'adiabatic',
            'phase': 'vapor',
            'T_min': 350.0,
            'T_max': 1500.0,
            'thermal_scan_points': 81,
            'reactions': [reaction],
        }).solve({'in': feed})

        self.assertGreater(result.outlet_streams['out'].T, feed.T)
        self.assertGreater(result.performance['component_conversions']['CO'], 0.0)
        self.assertTrue(result.performance['reactions'][0]['reversible'])
        self.assertLess(
            abs(result.performance['energy_balance_residual_kW']),
            1.0e-8,
        )

    def test_thermal_root_with_phase_unstable_outlet_is_rejected(self):
        components = ['CO', 'H2', 'CH3OH']
        thermo = create_thermodynamics(components, 'IDEAL')
        feed = thermo.calculate_state(
            350.0,
            80.0,
            3.0,
            {'CO': 1.0 / 3.0, 'H2': 2.0 / 3.0},
            phase='vapor',
            flash=False,
        )
        reaction = {
            'equation': 'CO + 2 H2 <=> CH3OH',
            'A': 1.0,
            'Ea': 0.0,
            'Ea_unit': 'J/mol',
            'rate_basis': 'activity',
            'concentration_unit': 'kmol/m3',
            'pressure_unit': 'bar',
            'rate_unit': 'kmol/m3/h',
        }
        with self.assertRaisesRegex(UnitOperationError, 'phase-unstable'):
            KineticsCSTR('CSTR-U', thermo, {
                'volume': 0.1,
                'mode': 'jacketed',
                'phase': 'vapor',
                'UA': 1000.0,
                'T_jacket': 350.0,
                'T_min': 250.0,
                'T_max': 1000.0,
                'thermal_scan_points': 61,
                'thermal_branch': 'lowest',
                'reactions': [reaction],
            }).solve({'in': feed})


class ThermalCSTRPFDTests(unittest.TestCase):
    def test_compact_catalog_duty_cstr_round_trip_and_recycle(self):
        text = '''\
PROCESS: Thermal CSTR recycle
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
STREAM Mixed : M.out -> CSTR.in
STREAM Reacted : CSTR.out -> SP.in
STREAM Product : SP.product
UNIT M : Mixer
    T_out = 500 [K]
    P = 2 [bar]
UNIT CSTR : CSTR
    volume = 5000 [L]
    mode = duty
    Q = 1 [kW]
    phase = vapor
    T_min = 300 [K]
    T_max = 900 [K]
    thermal_scan_points = 61
    REACTIONS:
        @isomerization
UNIT SP : Splitter
    outlets = product, recycle
    product_split_frac = 0.5
'''
        pfd = PFDParser().parse(text)
        self.assertEqual(validate_pfd(pfd), ([], []))
        restored = PFDParser().parse(pfd.to_pfd())
        self.assertEqual(validate_pfd(restored), ([], []))
        result = Simulator(restored).run(max_iterations=100)

        self.assertTrue(result.converged, result.warnings)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.recycle_info['tear_streams'], ['Recycle'])
        self.assertLess(result.mass_balance_error, 1.0e-4)
        self.assertLess(result.energy_balance_error, 1.0e-4)
        self.assertAlmostEqual(
            result.units['CSTR'].performance['duty_kW'],
            1.0,
            places=10,
        )


if __name__ == '__main__':
    unittest.main()
