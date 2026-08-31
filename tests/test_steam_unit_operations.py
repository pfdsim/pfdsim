import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from thermodynamics import create_thermodynamics
from unit_operations_basic import Expander, HeatExchanger, Valve


class SteamUnitOperationTests(unittest.TestCase):
    def test_expander_uses_direct_ps_ph_paths_for_steam(self):
        thermo = create_thermodynamics(['H2O'], 'STEAM')
        flow = 59.02 * 3600.0 / 18.015
        inlet = thermo.calculate_state(
            500.0 + 273.15,
            86.0,
            flow,
            {'H2O': 1.0},
            phase='vapor',
            flash=False,
        )
        calls = {'PH': 0, 'PS': 0}
        original_ph = thermo.calculate_state_PH
        original_ps = thermo.calculate_state_PS

        def counted_ph(*args, **kwargs):
            calls['PH'] += 1
            return original_ph(*args, **kwargs)

        def counted_ps(*args, **kwargs):
            calls['PS'] += 1
            return original_ps(*args, **kwargs)

        thermo.calculate_state_PH = counted_ph
        thermo.calculate_state_PS = counted_ps

        result = Expander(
            'T1',
            thermo,
            {'P_out': 0.1, 'eta': 0.75, 'eta_mech': 1.0},
        ).solve({'in': inlet})
        outlet = result.outlet_streams['out']

        self.assertEqual(calls['PS'], 1)
        self.assertEqual(calls['PH'], 1)
        self.assertAlmostEqual(outlet.vapor_fraction, 0.9381606, delta=5e-5)
        self.assertAlmostEqual(-result.work / 3600.0, 56435.58, delta=5.0)

    def test_valve_uses_direct_ph_path_for_steam(self):
        thermo = create_thermodynamics(['H2O'], 'STEAM')
        inlet = thermo.calculate_state(
            200.0 + 273.15,
            10.0,
            100.0,
            {'H2O': 1.0},
            phase='liquid',
            flash=False,
        )
        calls = {'PH': 0}
        original_ph = thermo.calculate_state_PH

        def counted_ph(*args, **kwargs):
            calls['PH'] += 1
            return original_ph(*args, **kwargs)

        thermo.calculate_state_PH = counted_ph

        result = Valve('V1', thermo, {'P_out': 1.01325}).solve({'in': inlet})

        self.assertEqual(calls['PH'], 1)
        self.assertAlmostEqual(
            result.outlet_streams['out'].H,
            inlet.H,
            delta=1e-6,
        )

    def test_heat_exchanger_uses_direct_ph_path_for_steam_duty(self):
        thermo = create_thermodynamics(['H2O'], 'STEAM')
        hot = thermo.calculate_state(
            500.0 + 273.15,
            86.0,
            1000.0,
            {'H2O': 1.0},
            phase='vapor',
            flash=False,
        )
        cold = thermo.calculate_state_PQ(
            1.0,
            0.0,
            1000.0,
            {'H2O': 1.0},
        )
        calls = {'PH': 0}
        original_ph = thermo.calculate_state_PH

        def counted_ph(*args, **kwargs):
            calls['PH'] += 1
            return original_ph(*args, **kwargs)

        thermo.calculate_state_PH = counted_ph

        result = HeatExchanger(
            'HX',
            thermo,
            {'Q': 5000.0},
        ).solve({
            'tube_in': hot,
            'shell_in': cold,
        })

        self.assertGreaterEqual(calls['PH'], 2)
        self.assertAlmostEqual(result.performance['duty_kW'], 5000.0, places=8)
        self.assertGreater(result.performance['UA_required_W_per_K'], 0.0)
        self.assertLess(result.outlet_streams['tube_out'].H, hot.H)
        self.assertGreater(result.outlet_streams['shell_out'].H, cold.H)

    def test_heat_exchanger_uses_direct_pq_for_steam_vapor_fraction_spec(self):
        thermo = create_thermodynamics(['H2O'], 'STEAM')
        hot = thermo.calculate_state(
            220.0 + 273.15,
            10.0,
            1000.0,
            {'H2O': 1.0},
            phase='vapor',
            flash=False,
        )
        cold = thermo.calculate_state(
            50.0 + 273.15,
            1.0,
            1000.0,
            {'H2O': 1.0},
            phase='liquid',
            flash=False,
        )
        calls = {'PQ': 0, 'PH': 0}
        original_pq = thermo.calculate_state_PQ
        original_ph = thermo.calculate_state_PH

        def counted_pq(*args, **kwargs):
            calls['PQ'] += 1
            return original_pq(*args, **kwargs)

        def counted_ph(*args, **kwargs):
            calls['PH'] += 1
            return original_ph(*args, **kwargs)

        thermo.calculate_state_PQ = counted_pq
        thermo.calculate_state_PH = counted_ph

        result = HeatExchanger(
            'HX',
            thermo,
            {'hot_vapor_fraction': 0.85},
        ).solve({
            'hot_in': hot,
            'cold_in': cold,
        })

        hot_out = result.outlet_streams['hot_out']
        cold_out = result.outlet_streams['cold_out']

        self.assertEqual(calls['PQ'], 1)
        self.assertGreaterEqual(calls['PH'], 1)
        self.assertAlmostEqual(hot_out.vapor_fraction, 0.85, places=8)
        self.assertAlmostEqual(
            hot.F * (hot.H - hot_out.H),
            cold.F * (cold_out.H - cold.H),
            delta=1e-4,
        )

    def test_heat_exchanger_auto_u_segments_steam_condensation(self):
        thermo = create_thermodynamics(['H2O'], 'STEAM')
        hot = thermo.calculate_state(
            220.0 + 273.15,
            10.0,
            100.0,
            {'H2O': 1.0},
            phase='vapor',
            flash=False,
        )
        cold = thermo.calculate_state(
            50.0 + 273.15,
            1.0,
            5000.0,
            {'H2O': 1.0},
            phase='liquid',
            flash=False,
        )

        result = HeatExchanger(
            'HX',
            thermo,
            {'hot_vapor_fraction': 0.8, 'U': 'auto', 'curve_segments': 12},
        ).solve({
            'hot_in': hot,
            'cold_in': cold,
        })

        services = result.performance['auto_U_service_counts']
        self.assertTrue(result.performance['auto_U_estimation'])
        self.assertIn('gas_liquid', services)
        self.assertIn('steam_feedwater', services)
        self.assertGreater(result.performance['auto_U_area_required_m2'], 0.0)
        self.assertTrue(any('auto-U estimates are preliminary' in warning for warning in result.warnings))


if __name__ == '__main__':
    unittest.main()
