import os
import sys
import unittest
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from thermodynamics import P_REF, create_thermodynamics, ThermodynamicsError


class SteamThermodynamicsTests(unittest.TestCase):
    def test_steam_method_uses_pfdsim_reference_for_liquid_water(self):
        thermo = create_thermodynamics(['H2O'], 'STEAM')

        state = thermo.calculate_state(
            298.15,
            P_REF,
            1.0,
            {'H2O': 1.0},
            phase='liquid',
            flash=False,
        )

        self.assertAlmostEqual(state.H, -285830.0, places=6)
        self.assertAlmostEqual(state.S, 69.95, places=6)
        self.assertEqual(state.vapor_fraction, 0.0)
        self.assertGreater(state.rho, 50.0)
        self.assertIsNotNone(state.Cp)

    def test_steam_method_supports_saturation_and_wet_inverse_states(self):
        thermo = create_thermodynamics(['H2O'], 'IF97')
        P = 1.01325
        quality = 0.25

        saturated_liquid = thermo.calculate_state_PQ(P, 0.0, 1.0, {'H2O': 1.0})
        saturated_vapor = thermo.calculate_state_PQ(P, 1.0, 1.0, {'H2O': 1.0})
        wet_H = (
            (1.0 - quality) * saturated_liquid.H
            + quality * saturated_vapor.H
        )
        wet_S = (
            (1.0 - quality) * saturated_liquid.S
            + quality * saturated_vapor.S
        )

        from_H = thermo.calculate_state_PH(P, wet_H, 1.0, {'H2O': 1.0})
        from_S = thermo.calculate_state_PS(P, wet_S, 1.0, {'H2O': 1.0})

        self.assertAlmostEqual(from_H.T, saturated_liquid.T, places=6)
        self.assertAlmostEqual(from_H.vapor_fraction, quality, places=8)
        self.assertAlmostEqual(from_H.H, wet_H, places=6)
        self.assertIsNone(from_H.Cp)
        self.assertAlmostEqual(from_S.T, saturated_liquid.T, places=6)
        self.assertAlmostEqual(from_S.vapor_fraction, quality, places=8)
        self.assertAlmostEqual(from_S.S, wet_S, places=6)

    def test_steam_method_calculate_state_pq_directly_builds_wet_state(self):
        thermo = create_thermodynamics(['H2O'], 'STEAM')
        P = 0.1
        quality = 0.4

        liquid = thermo.calculate_state_PQ(P, 0.0, 1.0, {'H2O': 1.0})
        vapor = thermo.calculate_state_PQ(P, 1.0, 1.0, {'H2O': 1.0})
        wet = thermo.calculate_state_PQ(P, quality, 2.5, {'H2O': 1.0})

        self.assertAlmostEqual(wet.T, liquid.T, places=8)
        self.assertAlmostEqual(wet.P, P, places=8)
        self.assertAlmostEqual(wet.F, 2.5, places=12)
        self.assertAlmostEqual(wet.vapor_fraction, quality, places=12)
        self.assertAlmostEqual(
            wet.H,
            (1.0 - quality) * liquid.H + quality * vapor.H,
            places=6,
        )
        self.assertAlmostEqual(
            wet.S,
            (1.0 - quality) * liquid.S + quality * vapor.S,
            places=6,
        )
        self.assertIsNone(wet.Cp)
        self.assertGreater(wet.rho, 0.0)

    def test_steam_method_round_trips_superheated_ph_and_ps_states(self):
        thermo = create_thermodynamics(['water'], 'STEAM')
        state = thermo.calculate_state(
            500.0 + 273.15,
            86.0,
            2.0,
            {'water': 1.0},
        )

        from_H = thermo.calculate_state_PH(86.0, state.H, 2.0, {'water': 1.0})
        from_S = thermo.calculate_state_PS(86.0, state.S, 2.0, {'water': 1.0})

        self.assertEqual(state.vapor_fraction, 1.0)
        self.assertAlmostEqual(from_H.T, state.T, delta=0.005)
        self.assertAlmostEqual(from_H.H, state.H, delta=0.05)
        self.assertAlmostEqual(from_S.T, state.T, delta=0.005)
        self.assertAlmostEqual(from_S.S, state.S, delta=0.001)

    def test_steam_method_rejects_non_water_components(self):
        with self.assertRaises(ThermodynamicsError):
            create_thermodynamics(['H2O', 'CH4'], 'STEAM')

        thermo = create_thermodynamics(['H2O'], 'STEAM')
        with self.assertRaises(ThermodynamicsError):
            thermo.calculate_state(300.0, 1.0, 1.0, {'H2O': 0.9, 'CH4': 0.1})

    def test_steam_method_honors_calculate_state_include(self):
        thermo = create_thermodynamics(['H2O'], 'STEAM')

        state = thermo.calculate_state(
            400.0,
            1.0,
            1.0,
            {'H2O': 1.0},
            include=('H',),
        )

        self.assertIsNotNone(state.H)
        self.assertIsNone(state.S)
        self.assertIsNone(state.Cp)
        self.assertIsNone(state.rho)
        self.assertIsNotNone(state.MW)

    def test_steam_viscosity_uses_if97_without_property_resolver(self):
        thermo = create_thermodynamics(['H2O'], 'STEAM')
        T = 473.15
        P = 1.0
        expected = thermo._props('V', 'T', T, 'P', P * 1e5)

        with patch(
            'property_resolution.resolver.PropertyResolver.resolve_viscosity',
            side_effect=AssertionError('generic resolver should not be called'),
        ):
            state = thermo.calculate_state(
                T,
                P,
                1.0,
                {'H2O': 1.0},
                phase='vapor',
                flash=False,
                include=('mu',),
            )

        self.assertAlmostEqual(state.mu, expected, places=15)

    def test_steam_wet_transport_viscosity_uses_saturated_if97_phases(self):
        thermo = create_thermodynamics(['water'], 'IF97')
        P = 1.01325
        quality = 0.25
        composition = {'water': 1.0}
        state = thermo.calculate_state_PQ(
            P, quality, 1.0, composition, include=('mu',)
        )

        viscosities = thermo.transport_mixture_viscosity(
            composition,
            state.T,
            P,
            quality,
            state.x,
            state.y,
        )
        expected_liquid = thermo._props('V', 'P', P * 1e5, 'Q', 0.0)
        expected_vapor = thermo._props('V', 'P', P * 1e5, 'Q', 1.0)

        self.assertAlmostEqual(viscosities.liquid, expected_liquid, places=15)
        self.assertAlmostEqual(viscosities.vapor, expected_vapor, places=15)
        self.assertAlmostEqual(
            state.mu,
            (1.0 - quality) * expected_liquid + quality * expected_vapor,
            places=15,
        )


if __name__ == '__main__':
    unittest.main()
