import math
import unittest
from unittest.mock import patch

from chemical_properties import ChemicalDatabase
from cubic_eos import CubicEOS
from property_resolver import PropertyResolver
from simulator import Simulator
from thermodynamics import PSRKThermodynamics, create_thermodynamics
from thermodynamics_models.factory import SUPPORTED_METHODS


class PSRKIntegrationTests(unittest.TestCase):
    def test_factory_constructs_psrk_with_symbol_to_cas_mapping(self):
        thermo = create_thermodynamics(["ethanol", "water"], "PSRK")

        self.assertIsInstance(thermo, PSRKThermodynamics)
        self.assertIn("PSRK", SUPPORTED_METHODS)
        self.assertEqual(
            thermo.component_cas,
            {"ethanol": "64-17-5", "water": "7732-18-5"},
        )
        phi = thermo.fugacity_coefficients(
            350.0,
            1.0,
            {"ethanol": 0.5, "water": 0.5},
            "liquid",
        )
        self.assertAlmostEqual(phi["ethanol"], 1.1388618715648329, places=12)
        self.assertAlmostEqual(phi["water"], 0.6054606414838393, places=12)

    def test_psrk_uses_resolver_liquid_volume_and_eos_vapor_volume(self):
        thermo = create_thermodynamics(["ethanol", "water"], "PSRK")
        composition = {"ethanol": 0.5, "water": 0.5}

        liquid = thermo.molar_volume_liquid(350.0, 1.0, composition)
        resolved = thermo.mixture_liquid_molar_volume(composition, 350.0)
        vapor = thermo.vapor_molar_volume_for_density(350.0, 1.0, composition)

        self.assertAlmostEqual(liquid, resolved, places=14)
        self.assertGreater(vapor, 20.0)
        self.assertLess(vapor, 35.0)

    def test_integrated_psrk_tp_flash_closes_material_and_fugacity_balances(self):
        thermo = create_thermodynamics(["ethanol", "water"], "PSRK")
        overall = {"ethanol": 0.5, "water": 0.5}
        temperature = 355.0
        pressure = 1.0

        vapor_fraction, liquid, vapor = thermo.flash_TP(
            overall, temperature, pressure
        )
        phi_liquid = thermo.fugacity_coefficients(
            temperature, pressure, liquid, "liquid"
        )
        phi_vapor = thermo.fugacity_coefficients(
            temperature, pressure, vapor, "vapor"
        )

        self.assertGreater(vapor_fraction, 0.0)
        self.assertLess(vapor_fraction, 1.0)
        for component in overall:
            self.assertAlmostEqual(
                overall[component],
                (1.0 - vapor_fraction) * liquid[component]
                + vapor_fraction * vapor[component],
                places=9,
            )
            self.assertAlmostEqual(
                liquid[component] * phi_liquid[component],
                vapor[component] * phi_vapor[component],
                places=8,
            )

    def test_starred_psrk_critical_bundle_falls_back_to_resolved_soave(self):
        thermo = create_thermodynamics(["acetaldehyde"], "PSRK")
        component = thermo.psrk._component_data[0]

        self.assertFalse(component.uses_mathias_copeman)
        self.assertEqual(
            thermo.psrk.component_parameter_sources["75-07-0"],
            "normal property resolution",
        )
        self.assertTrue(
            any("generalized omega-based Soave alpha" in item for item in thermo.warnings)
        )
        phi = thermo.fugacity_coefficients(
            350.0, 10.0, {"acetaldehyde": 1.0}
        )
        self.assertTrue(math.isfinite(phi["acetaldehyde"]))
        self.assertGreater(phi["acetaldehyde"], 0.0)

    def test_pfd_parser_simulator_reporting_and_explicit_bundle_override(self):
        pfd = (
            "PROCESS: Integrated PSRK\n"
            "VERSION: 1.0\n"
            "ONLINE_LOOKUP: false\n"
            "THERMO_METHOD: PSRK\n"
            "\n"
            "COMPONENTS:\n"
            "    CH4 | Methane | Tc=200.0, Pc=50.0, omega=0.2, "
            "mc_c1=0.8, mc_c2=-0.1\n"
            "\n"
            "STREAM Feed : FEED -> PRODUCT\n"
            "    T = 180 [K]\n"
            "    P = 5 [bar]\n"
            "    F = 1 [kmol/h]\n"
            "    x = CH4:1.0\n"
        )

        simulator = Simulator.from_string(pfd)
        result = simulator.run()
        component = simulator.thermo.psrk._component_data[0]
        report = simulator._generate_pfr()

        self.assertTrue(result.converged)
        self.assertEqual(simulator.thermo_method, "PSRK")
        self.assertIsInstance(simulator.thermo, PSRKThermodynamics)
        self.assertEqual(component.Tc_K, 200.0)
        self.assertEqual(component.Pc_bar, 50.0)
        self.assertTrue(component.uses_mathias_copeman)
        self.assertEqual(component.c1, 0.8)
        self.assertEqual(component.c2, -0.1)
        self.assertEqual(component.c3, 0.0)
        self.assertIn("THERMO_METHOD: PSRK", report)

    def test_generic_resolver_offers_only_unstarred_psrk_tc_pc(self):
        resolver = PropertyResolver()
        water = resolver._get_psrk_critical_properties(
            "water", {"CAS": "7732-18-5"}
        )
        acetaldehyde = resolver._get_psrk_critical_properties(
            "acetaldehyde", {"CAS": "75-07-0"}
        )

        self.assertEqual(set(water), {"Tc", "Pc"})
        self.assertEqual(water["Tc"].method, "psrk2005_unstarred_critical")
        self.assertEqual(water["Tc"].quality, 0.95)
        self.assertEqual(set(acetaldehyde), {"Tc"})

        with patch.object(resolver, "_coolprop_critical_properties", return_value=None), \
             patch.object(resolver, "_get_acs_jced_critical_properties", return_value=None), \
             patch.object(resolver, "_get_perry_critical_properties", return_value=None), \
             patch.object(resolver, "_get_textbook_entry", return_value=None), \
             patch.object(resolver, "_get_effective_critical_properties", return_value=None):
            resolved = resolver._resolve_critical_properties_uncached(
                "water",
                {"CAS": "7732-18-5"},
                allow_online=False,
                allow_estimation=False,
            )

        self.assertEqual(resolved["Tc"].method, "psrk2005_unstarred_critical")
        self.assertEqual(resolved["Pc"].method, "psrk2005_unstarred_critical")
        self.assertNotEqual(resolved["omega"].method, "psrk2005_unstarred_critical")

    def test_effective_critical_at_095_precedes_psrk_but_lower_quality_does_not(self):
        def resolve_with_quality(quality):
            resolver = PropertyResolver()
            effective = {
                "CAS": "7732-18-5",
                "name": "Water effective",
                "Tc": 648.0,
                "Pc": 221.0,
                "Tc_quality": quality,
                "Pc_quality": quality,
            }
            with patch.object(resolver, "_coolprop_critical_properties", return_value=None), \
                 patch.object(resolver, "_get_acs_jced_critical_properties", return_value=None), \
                 patch.object(resolver, "_get_perry_critical_properties", return_value=None), \
                 patch.object(resolver, "_get_textbook_entry", return_value=None), \
                 patch.object(resolver, "_get_effective_critical_properties", return_value=effective):
                return resolver._resolve_critical_properties_uncached(
                    "water",
                    {"CAS": "7732-18-5"},
                    allow_online=False,
                    allow_estimation=False,
                )

        at_threshold = resolve_with_quality(0.95)
        below_threshold = resolve_with_quality(0.949)

        self.assertEqual(at_threshold["Tc"].method, "effective_critical")
        self.assertEqual(at_threshold["Pc"].method, "effective_critical")
        self.assertEqual(
            below_threshold["Tc"].method,
            "psrk2005_unstarred_critical",
        )
        self.assertEqual(
            below_threshold["Pc"].method,
            "psrk2005_unstarred_critical",
        )

    def test_srk_mc_prefers_compatible_psrk_coefficients_to_chemsep(self):
        database = ChemicalDatabase(enable_online=False)
        eos = CubicEOS(["CH4"], "SRK-MC", database)
        params = eos.params["CH4"]

        self.assertEqual(
            params.mc_source,
            "PSRK 2005 supplementary pure-component table",
        )
        self.assertEqual(params.mc_Tmin, 89.15)
        self.assertEqual(params.mc_Tmax, 191.03)
        eos._alpha(params, 80.0)
        self.assertTrue(
            any("outside its published temperature range" in item for item in eos.warnings)
        )

        incompatible_database = ChemicalDatabase(enable_online=False)
        methane = incompatible_database.get("CH4", fetch_online=False)
        methane.Tc = 250.0
        incompatible = CubicEOS(["CH4"], "SRK-MC", incompatible_database)
        self.assertEqual(
            incompatible.params["CH4"].mc_source,
            "ChemSep SRK-MC parameter table",
        )


if __name__ == "__main__":
    unittest.main()
