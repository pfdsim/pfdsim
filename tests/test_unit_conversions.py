import unittest

from unit_conversions import (
    mass_flow_to_kg_per_hour,
    molar_flow_to_kmol_per_hour,
    pressure_to_bar,
    temperature_to_kelvin,
)


class UnitConversionTests(unittest.TestCase):
    def test_pressure_conversion(self):
        self.assertAlmostEqual(pressure_to_bar(1.0, "atm"), 1.01325)
        self.assertAlmostEqual(pressure_to_bar(1.0, "Pa"), 1.0e-5)
        self.assertAlmostEqual(pressure_to_bar(1.0, "kPa"), 0.01)
        with self.assertRaises(ValueError):
            pressure_to_bar(1.0, "invalid", strict=True)

    def test_temperature_conversion(self):
        self.assertAlmostEqual(
            temperature_to_kelvin(25.0, "C"),
            298.15,
        )
        self.assertAlmostEqual(
            temperature_to_kelvin(77.0, "F"),
            298.15,
        )
        self.assertAlmostEqual(
            temperature_to_kelvin(
                25.0,
                None,
                infer_unitless_celsius_below=200.0,
            ),
            298.15,
        )
        self.assertEqual(temperature_to_kelvin(300.0, "K"), 300.0)

    def test_mass_flow_conversion(self):
        self.assertAlmostEqual(
            mass_flow_to_kg_per_hour(1.0, "lb/h"),
            0.45359237,
        )
        self.assertAlmostEqual(
            mass_flow_to_kg_per_hour(1000.0, "g/h"),
            1.0,
        )
        self.assertAlmostEqual(
            mass_flow_to_kg_per_hour(1.0, "kg/s"),
            3600.0,
        )

    def test_molar_flow_conversion(self):
        self.assertAlmostEqual(
            molar_flow_to_kmol_per_hour(1000.0, "mol/h"),
            1.0,
        )
        self.assertAlmostEqual(
            molar_flow_to_kmol_per_hour(1.0, "mol/s"),
            3.6,
        )
        self.assertAlmostEqual(
            molar_flow_to_kmol_per_hour(1.0, "kmol/s"),
            3600.0,
        )


if __name__ == "__main__":
    unittest.main()
