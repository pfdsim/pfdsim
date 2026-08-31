import math
import os
import sys
import unittest

from scipy.optimize import brentq


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from thermodynamics import R, create_thermodynamics


class EntropySupportTests(unittest.TestCase):
    def test_ideal_gas_entropy_has_pressure_dependence(self):
        thermo = create_thermodynamics(['CH4'], 'IDEAL')
        composition = {'CH4': 1.0}

        low_pressure = thermo.calculate_state(
            350.0, 1.0, 1.0, composition, phase='vapor', flash=False
        )
        high_pressure = thermo.calculate_state(
            350.0, 10.0, 1.0, composition, phase='vapor', flash=False
        )

        self.assertIsNotNone(low_pressure.S)
        self.assertIsNotNone(high_pressure.S)
        self.assertAlmostEqual(
            low_pressure.S - high_pressure.S,
            R * math.log(10.0),
            places=6,
        )
        self.assertIn('S', low_pressure.to_dict())

    def test_calculate_state_include_limits_optional_properties(self):
        thermo = create_thermodynamics(['CH4'], 'IDEAL')

        probe = thermo.calculate_state(
            350.0,
            1.0,
            1.0,
            {'CH4': 1.0},
            phase='vapor',
            flash=False,
            include=('H',),
        )
        self.assertIsNotNone(probe.H)
        self.assertIsNone(probe.Cp)
        self.assertIsNone(probe.S)
        self.assertIsNone(probe.rho)
        self.assertIsNotNone(probe.MW)

        default = thermo.calculate_state(
            350.0,
            1.0,
            1.0,
            {'CH4': 1.0},
            phase='vapor',
            flash=False,
        )
        self.assertIsNotNone(default.H)
        self.assertIsNotNone(default.Cp)
        self.assertIsNotNone(default.S)
        self.assertIsNotNone(default.rho)

    def test_two_phase_density_uses_phase_volume_average(self):
        thermo = create_thermodynamics(['water'], 'IDEAL')
        composition = {'water': 1.0}
        vapor_fraction = 0.25
        T = 350.0
        P = 1.0

        thermo.flash_TP = lambda _composition, _T, _P: (
            vapor_fraction,
            dict(composition),
            dict(composition),
        )
        state = thermo.calculate_state(T, P, 1.0, composition, include=('rho',))

        liquid_volume = thermo.mixture_liquid_molar_volume(composition, T)
        vapor_volume = thermo.vapor_molar_volume_for_density(T, P, composition)
        expected_rho = 1.0 / (
            (1.0 - vapor_fraction) * liquid_volume
            + vapor_fraction * vapor_volume
        )
        self.assertAlmostEqual(state.rho, expected_rho, places=12)

    def test_rks_bm_entropy_support_reproduces_isentropic_compressor_state(self):
        components = ['N2', 'O2', 'NH3', 'CH4', 'H2O', 'AR', 'H2']
        flows = {
            'N2': 1347.537,
            'O2': 3.220,
            'NH3': 2.401,
            'CH4': 45.018,
            'H2O': 0.0,
            'AR': 40.996,
            'H2': 4184.463,
        }
        total_flow = 5623.635
        composition = {comp: flows[comp] / total_flow for comp in components}
        thermo = create_thermodynamics(components, 'RKS-BM')

        inlet = thermo.calculate_state(
            64.8 + 273.15,
            10.0,
            total_flow,
            composition,
            phase='vapor',
            flash=False,
        )

        def state_at(temperature, pressure):
            return thermo.calculate_state(
                temperature,
                pressure,
                total_flow,
                composition,
                phase='vapor',
                flash=False,
                include=('H', 'S'),
            )

        def solve_temperature(residual, points):
            values = [(point, residual(point)) for point in points]
            for (left, f_left), (right, f_right) in zip(values, values[1:]):
                if abs(f_left) < 1e-9:
                    return left
                if f_left * f_right < 0:
                    return brentq(residual, left, right, xtol=1e-8, rtol=1e-10)
            self.fail(f'Could not bracket entropy/enthalpy root: {values!r}')

        outlet_pressure = 100.0
        isentropic_temperature = solve_temperature(
            lambda temperature: state_at(temperature, outlet_pressure).S - inlet.S,
            [250.0, 300.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1200.0],
        )
        isentropic_outlet = state_at(isentropic_temperature, outlet_pressure)

        eta_isen = 0.8
        eta_mech = 0.95
        actual_enthalpy = inlet.H + (isentropic_outlet.H - inlet.H) / eta_isen
        actual_temperature = solve_temperature(
            lambda temperature: state_at(temperature, outlet_pressure).H - actual_enthalpy,
            [300.0, 400.0, 500.0, 650.0, 800.0, 1000.0, 1200.0, 1400.0],
        )
        fluid_power_kw = total_flow * (actual_enthalpy - inlet.H) / 3600.0
        shaft_power_kw = fluid_power_kw / eta_mech
        expected_fluid_power_kw = 18167.0
        expected_shaft_power_kw = expected_fluid_power_kw / eta_mech
        power_tolerance = 0.0002

        self.assertAlmostEqual(isentropic_temperature - 273.15, 374.87, delta=0.05)
        self.assertAlmostEqual(actual_temperature - 273.15, 452.5, delta=0.1)
        self.assertAlmostEqual(
            fluid_power_kw,
            expected_fluid_power_kw,
            delta=expected_fluid_power_kw * power_tolerance,
        )
        self.assertAlmostEqual(
            shaft_power_kw,
            expected_shaft_power_kw,
            delta=expected_shaft_power_kw * power_tolerance,
        )


if __name__ == '__main__':
    unittest.main()
