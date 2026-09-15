import json
import math
import unittest
from unittest.mock import patch

import govender_method
from pfd_parser import ParseError, parse_pfd
from perry_properties import PerryPropertyLibrary, THERMAL_CONDUCTIVITY_DATA_PATH
from pfdsim_modified_stiel_thodos import (
    GROUP_CORRECTIONS,
    evaluate_pfdsim_modified_stiel_thodos,
)
from physical_constants import R_J_MOL_K
from property_resolution.common import PropertyResolutionError, PropertyResolutionResult
from property_resolution.thermal_conductivity import GROUP_CORRECTED_MAPE_PERCENT
from property_resolver import PropertyResolver, resolve_thermal_conductivity
from simulator import Simulator


class PerryThermalConductivityDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from property_resolution.thermal_conductivity import (
            PERRY_CONDUCTIVITY_EQUATIONS,
        )

        cls.equations = PERRY_CONDUCTIVITY_EQUATIONS
        with THERMAL_CONDUCTIVITY_DATA_PATH.open() as stream:
            cls.payload = json.load(stream)

    def test_all_three_source_tables_are_preserved(self):
        metadata = self.payload["metadata"]
        tables = metadata["tables"]

        self.assertEqual(metadata["source_pdf_pages"], [324, 339])
        self.assertEqual(tables["vapor_thermal_conductivity"]["table"], "2-145")
        self.assertEqual(tables["vapor_thermal_conductivity"]["rows_extracted"], 355)
        self.assertEqual(
            tables["saturated_liquid_thermal_conductivity"]["table"], "2-146"
        )
        self.assertEqual(
            tables["saturated_liquid_thermal_conductivity"]["substances_extracted"],
            25,
        )
        self.assertEqual(tables["liquid_thermal_conductivity"]["table"], "2-147")
        self.assertEqual(tables["liquid_thermal_conductivity"]["rows_extracted"], 343)
        self.assertEqual(len(self.payload["chemicals"]), 345)

    def test_every_extracted_equation_has_one_shared_evaluator_form(self):
        equation_ids = {
            row["equation_id"]
            for entry in self.payload["chemicals"].values()
            for key in (
                "vapor_thermal_conductivity",
                "liquid_thermal_conductivity",
            )
            for row in entry.get(key, [])
        }

        self.assertEqual(equation_ids, {100, 102})
        self.assertEqual(set(self.equations), equation_ids)

    def test_extracted_correlations_reproduce_printed_endpoints(self):
        resolver = PropertyResolver()
        for cas, entry in self.payload["chemicals"].items():
            for phase in ("vapor", "liquid"):
                for row in entry.get(f"{phase}_thermal_conductivity", []):
                    correlation = resolver._normalized_perry_conductivity_correlation(row)
                    for suffix in ("min", "max"):
                        with self.subTest(cas=cas, phase=phase, endpoint=suffix):
                            evaluated = resolver._evaluate_correlation(
                                correlation, row[f"T_{suffix}_K"]
                            )
                            self.assertIsNotNone(evaluated)
                            expected = row[f"value_at_T_{suffix}"]
                            self.assertTrue(
                                math.isclose(evaluated[0], expected, rel_tol=0.002)
                            )
        errors = [
            row["maximum_endpoint_relative_error"]
            for entry in self.payload["chemicals"].values()
            for key in (
                "vapor_thermal_conductivity",
                "liquid_thermal_conductivity",
            )
            for row in entry.get(key, [])
        ]

        self.assertEqual(len(errors), 698)
        self.assertLess(max(errors), 0.002)
        self.assertAlmostEqual(
            self.payload["metadata"]["maximum_endpoint_relative_error"],
            max(errors),
        )

    def test_pdf_text_layer_corrections_are_auditable(self):
        chemicals = self.payload["chemicals"]
        acetic = chemicals["64-19-7"]["vapor_thermal_conductivity"][-1]
        acetone = chemicals["67-64-1"]["vapor_thermal_conductivity"][0]
        cyclohexanone = chemicals["108-94-1"]["vapor_thermal_conductivity"][0]

        self.assertEqual(acetic["coefficients"][-1], -14_086_000.0)
        self.assertEqual(acetic["printed_coefficients"][-1], 14_086_000.0)
        self.assertIn("sign restored", acetic["extraction_correction"])
        self.assertEqual(acetone["coefficients"][0], 26.8)
        self.assertEqual(cyclohexanone["coefficients"][2], 498_780.0)


class ThermalConductivityResolutionTests(unittest.TestCase):
    def setUp(self):
        self.resolver = PropertyResolver()

    def assertClose(self, actual, expected, *, rel=1e-10, abs_tol=1e-12):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol),
            f"{actual!r} != {expected!r}",
        )

    def test_quality_table_covers_every_fitted_group(self):
        self.assertEqual(set(GROUP_CORRECTED_MAPE_PERCENT), set(GROUP_CORRECTIONS))

    def test_shared_evaluator_supports_dippr_equations_100_and_102(self):
        polynomial = self.resolver._evaluate_correlation(
            {
                "equation": "dippr_eq100",
                "coefficients": {"A": 1.0, "B": 2.0, "C": 3.0},
            },
            10.0,
        )
        transport = self.resolver._evaluate_correlation(
            {
                "equation": "dippr_eq102",
                "coefficients": {"A": 2.0, "B": 1.5, "C": 4.0, "D": 5.0},
            },
            10.0,
        )

        self.assertEqual(polynomial[0], 321.0)
        self.assertClose(transport[0], 2.0 * 10.0**1.5 / (1.0 + 0.4 + 0.05))

    def test_perry_vapor_and_liquid_correlations_use_shared_equations(self):
        vapor = self.resolver.resolve_thermal_conductivity(
            "water", 300.0, "gas", {}, allow_online=False
        )
        liquid = self.resolver.resolve_thermal_conductivity(
            "7732-18-5", 300.0, "liquid", {}, allow_online=False
        )

        self.assertClose(vapor.value, 6.2041e-6 * 300.0**1.3973)
        self.assertEqual(vapor.method, "perry_vapor_thermal_conductivity_eq102")
        self.assertIn("Table 2-145", vapor.notes)
        self.assertClose(
            liquid.value,
            -0.432 + 0.0057255 * 300.0 - 0.000008078 * 300.0**2 + 1.861e-9 * 300.0**3,
        )
        self.assertEqual(liquid.method, "perry_liquid_thermal_conductivity_eq100")
        self.assertIn("Table 2-147", liquid.notes)

    def test_central_identity_candidates_honor_symbol_name_and_cas(self):
        props = {
            "symbol": "AQUA",
            "name": "water",
            "CAS": "7732-18-5",
            "formula": "H2O",
        }

        result = self.resolver.resolve_thermal_conductivity(
            "AQUA", 300.0, "vapour", props, allow_online=False
        )

        self.assertEqual(result.method, "perry_vapor_thermal_conductivity_eq102")

    def test_ambiguous_formula_does_not_collapse_isomers(self):
        with self.assertRaisesRegex(PropertyResolutionError, "Cannot determine"):
            self.resolver.resolve_thermal_conductivity(
                "C4H10O", 300.0, "liquid", {}, allow_online=False
            )

    def test_table_146_is_a_liquid_only_fallback(self):
        liquid = self.resolver.resolve_thermal_conductivity(
            "gasoline", 288.15, "l", {}, allow_online=False
        )

        self.assertClose(liquid.value, 0.117)
        self.assertEqual(
            liquid.method,
            "perry_saturated_liquid_thermal_conductivity_linear_interpolation",
        )
        self.assertEqual(liquid.quality, 0.93)
        with self.assertRaisesRegex(PropertyResolutionError, "Cannot determine"):
            self.resolver.resolve_thermal_conductivity(
                "gasoline", 288.15, "vapor", {}, allow_online=False
            )

    def test_table_146_does_not_bridge_blank_temperature_cells(self):
        library = PerryPropertyLibrary()
        self.assertIsNone(
            library.saturated_liquid_thermal_conductivity_W_per_m_K(
                "propanol", 273.15
            )
        )

    def _govender_fallback(
        self,
        smiles,
        *,
        temperature=300.0,
        tb=400.0,
        smiles_quality=1.0,
        tb_quality=1.0,
    ):
        smiles_result = PropertyResolutionResult(
            smiles, "local", "test_smiles", smiles_quality
        )
        tb_result = PropertyResolutionResult(
            tb, "local", "test_tb", tb_quality, "units K"
        )
        with (
            patch.object(self.resolver, "_get_perry_library", return_value=None),
            patch.object(
                self.resolver, "_resolve_smiles_result", return_value=smiles_result
            ),
            patch.object(
                self.resolver, "resolve_boiling_point", return_value=tb_result
            ),
        ):
            return self.resolver.resolve_thermal_conductivity(
                "test compound",
                temperature,
                "liquid",
                {"smiles": smiles, "Tb": tb},
                allow_online=False,
            )

    def test_liquid_fallback_uses_govender_with_base_quality(self):
        result = self._govender_fallback("CCCO", tb=370.0)
        expected = govender_method.estimate("CCCO", tb=370.0)
        self.assertClose(result.value, expected.conductivity_W_m_K(300.0))
        self.assertEqual(
            result.method, "govender_saturated_liquid_thermal_conductivity"
        )
        self.assertEqual(result.source, "calculated")
        self.assertEqual(result.quality, 0.80)
        self.assertIn("Govender local refits enabled", result.notes)
        self.assertIn("test_tb", result.notes)
        self.assertIn("test_smiles", result.notes)

    def test_liquid_govender_quality_penalties(self):
        small = self._govender_fallback("CCO")
        self.assertEqual(small.quality, 0.72)
        self.assertIn("fewer than 3 carbon atoms", small.notes)

        single_component = self._govender_fallback("COC(=O)OC")
        self.assertEqual(single_component.quality, 0.72)
        self.assertIn("single-component group(s) [61]", single_component.notes)

        caution = self._govender_fallback("CCSCC")
        self.assertEqual(caution.quality, 0.76)
        self.assertIn("other paper caution group(s) [100]", caution.notes)

        combined = self._govender_fallback("CCI")
        self.assertEqual(combined.quality, 0.68)
        self.assertIn("fewer than 3 carbon atoms", combined.notes)
        self.assertIn("other paper caution group(s) [47]", combined.notes)

    def test_liquid_govender_scales_quality_with_tb_quality(self):
        result = self._govender_fallback("CCCO", tb_quality=0.5)
        self.assertEqual(result.quality, 0.60)
        self.assertIn("factor 1-(1-0.5)^2=0.7500", result.notes)

    def test_liquid_govender_honors_unconditional_refusal(self):
        for smiles in ("C1CCSC1", "O=c1ccoc2ccccc12"):
            with (
                self.subTest(smiles=smiles),
                self.assertRaisesRegex(PropertyResolutionError, "Cannot determine"),
            ):
                self._govender_fallback(smiles)

    def test_table_146_fallback_honors_perry_cas_and_name_aliases(self):
        for identifier in ("carbon disulfide", "75-15-0"):
            with self.subTest(identifier=identifier):
                result = self.resolver.resolve_thermal_conductivity(
                    identifier, 343.15, "liquid", {}, allow_online=False
                )
                self.assertClose(result.value, 0.152)
                self.assertEqual(
                    result.method,
                    "perry_saturated_liquid_thermal_conductivity_tabulated",
                )

        library = PerryPropertyLibrary()
        # Perry calls this substance "Ethyl amine", while Table 2-146
        # uses "ethylamine". All names must reach the same tabulated run.
        for identifier in ("ethylamine", "ethyl amine", "75-04-7"):
            with self.subTest(identifier=identifier):
                result = library.saturated_liquid_thermal_conductivity_W_per_m_K(
                    identifier, 228.15
                )
                self.assertIsNotNone(result)
                self.assertClose(result.value, 0.2025)

        # This identity is absent from the main Perry correlation dataset.
        for identifier in ("ethyl iodide", "iodoethane", "75-03-6"):
            with self.subTest(identifier=identifier):
                result = self.resolver.resolve_thermal_conductivity(
                    identifier, 273.15, "liquid", {}, allow_online=False
                )
                self.assertClose(result.value, 0.092)

        # Identity expansion must preserve blanks and the single-knot range.
        self.assertIsNone(
            library.saturated_liquid_thermal_conductivity_W_per_m_K(
                "71-23-8", 273.15
            )
        )
        self.assertIsNone(
            library.saturated_liquid_thermal_conductivity_W_per_m_K(
                "7664-93-9", 274.15
            )
        )

    def test_pfd_overrides_are_phase_specific_and_highest_priority(self):
        props = {
            "CAS": "7732-18-5",
            "property_correlations": {
                "kg": {
                    "equation": "dippr_eq102",
                    "coefficients": {"A": 0.001, "B": 1.0},
                    "Tmin_K": 250.0,
                    "Tmax_K": 400.0,
                    "quality": 1.0,
                    "_pfd_override": True,
                },
                "kl": {
                    "equation": "dippr_eq100",
                    "coefficients": {"A": 0.9, "B": -0.001},
                    "Tmin_K": 250.0,
                    "Tmax_K": 400.0,
                    "quality": 1.0,
                    "_pfd_override": True,
                },
            },
        }

        vapor = self.resolver.resolve_thermal_conductivity(
            "water", 300.0, "vapor", props, allow_online=False
        )
        liquid = self.resolver.resolve_thermal_conductivity(
            "water", 300.0, "liquid", props, allow_online=False
        )

        self.assertClose(vapor.value, 0.3)
        self.assertClose(liquid.value, 0.6)
        self.assertEqual(
            vapor.method, "provided_vapor_thermal_conductivity_dippr_eq102"
        )
        self.assertEqual(
            liquid.method, "provided_liquid_thermal_conductivity_dippr_eq100"
        )
        self.assertEqual(vapor.quality, 1.0)
        self.assertEqual(liquid.quality, 1.0)

    def test_vapor_fallback_uses_resolved_inputs_and_group_quality(self):
        props = {
            "formula": "C2H6O",
            "smiles": "CCO",
            "MW": 46.07,
            "Tc": 514.0,
        }
        cp = PropertyResolutionResult(
            value=60.0 + R_J_MOL_K,
            source="local",
            method="test_ideal_gas_cp",
            quality=0.96,
        )
        viscosity = PropertyResolutionResult(
            value=1.0e-5,
            source="local",
            method="test_dilute_viscosity",
            quality=0.95,
        )
        with (
            patch.object(self.resolver, "_get_perry_library", return_value=None),
            patch.object(self.resolver, "resolve_heat_capacity", return_value=cp),
            patch.object(self.resolver, "resolve_viscosity", return_value=viscosity) as resolve_viscosity,
        ):
            result = self.resolver.resolve_thermal_conductivity(
                "ethanol", 300.0, "gas", props, allow_online=False
            )
        expected = evaluate_pfdsim_modified_stiel_thodos(
            300.0, 1.0e-5, 60.0, 46.07, 514.0, "nonlinear", "CCO", "C2H6O"
        )
        self.assertClose(result.value, expected.value_W_per_m_K)
        self.assertEqual(result.method, "pfdsim_modified_stiel_thodos_dilute_vapor")
        self.assertEqual(result.quality, 0.76)
        self.assertIn("quality group alcohol", result.notes)
        self.assertIn("test_dilute_viscosity", result.notes)
        resolve_viscosity.assert_called_once_with(
            "ethanol", 300.0, "vapor", props, allow_online=False
        )

    def test_vapor_fallback_uses_inorganic_group_without_smiles(self):
        props = {"formula": "N2", "MW": 28.0, "Tc": 126.0}
        cp = PropertyResolutionResult(2.5 * R_J_MOL_K, "local", "test_cp", 0.95)
        viscosity = PropertyResolutionResult(1.0e-5, "local", "test_mu", 0.95)
        with (
            patch.object(self.resolver, "_get_perry_library", return_value=None),
            patch.object(self.resolver, "_resolve_smiles_result") as resolve_smiles,
            patch.object(self.resolver, "resolve_heat_capacity", return_value=cp),
            patch.object(self.resolver, "resolve_viscosity", return_value=viscosity),
        ):
            result = self.resolver.resolve_thermal_conductivity(
                "nitrogen", 300.0, "vapor", props, allow_online=False
            )
        self.assertEqual(result.quality, 0.73)
        self.assertIn("quality group inorganic", result.notes)
        resolve_smiles.assert_not_called()

    def test_vapor_fallback_uses_inorganic_group_for_polyatomic_gas(self):
        props = {"formula": "CO2", "smiles": "O=C=O", "MW": 44.01, "Tc": 304.2}
        cp = PropertyResolutionResult(37.0, "local", "test_cp", 0.95)
        viscosity = PropertyResolutionResult(1.5e-5, "local", "test_mu", 0.95)
        with (
            patch.object(self.resolver, "_get_perry_library", return_value=None),
            patch.object(self.resolver, "resolve_heat_capacity", return_value=cp),
            patch.object(self.resolver, "resolve_viscosity", return_value=viscosity),
        ):
            result = self.resolver.resolve_thermal_conductivity(
                "carbon dioxide", 300.0, "vapor", props, allow_online=False
            )
        self.assertEqual(result.quality, 0.73)
        self.assertIn("quality group inorganic", result.notes)
        self.assertIn("geometry linear", result.notes)
        self.assertIn("correction factor 1", result.notes)

    def test_vapor_fallback_uses_worst_overlapping_group_and_input_quality(self):
        props = {"formula": "C8H8", "smiles": "C=Cc1ccccc1", "MW": 104.15, "Tc": 617.0}
        cp = PropertyResolutionResult(60.0 + R_J_MOL_K, "local", "test_cp", 0.99)
        viscosity = PropertyResolutionResult(2.0e-5, "estimated", "test_mu", 0.55)
        with (
            patch.object(self.resolver, "_get_perry_library", return_value=None),
            patch.object(self.resolver, "resolve_heat_capacity", return_value=cp),
            patch.object(self.resolver, "resolve_viscosity", return_value=viscosity),
        ):
            result = self.resolver.resolve_thermal_conductivity(
                "styrene", 500.0, "vapor", props, allow_online=False
            )
        self.assertEqual(result.quality, 0.55)
        self.assertIn("quality group alkene", result.notes)
        self.assertIn("'alkene', 'aromatic_ring', 'hydrocarbon_only'", result.notes)

    def test_vapor_fallback_caps_high_scoring_group_at_point_88(self):
        props = {"formula": "CH4S", "smiles": "CS", "MW": 48.1, "Tc": 470.0}
        cp = PropertyResolutionResult(40.0 + R_J_MOL_K, "local", "test_cp", 0.99)
        viscosity = PropertyResolutionResult(1.0e-5, "local", "test_mu", 0.99)
        with (
            patch.object(self.resolver, "_get_perry_library", return_value=None),
            patch.object(self.resolver, "resolve_heat_capacity", return_value=cp),
            patch.object(self.resolver, "resolve_viscosity", return_value=viscosity),
        ):
            result = self.resolver.resolve_thermal_conductivity(
                "methanethiol", 300.0, "vapor", props, allow_online=False
            )
        self.assertEqual(result.quality, 0.88)
        self.assertIn("quality group thiol", result.notes)

    def test_vapor_fallback_uses_overall_error_for_ungrouped_organic(self):
        props = {"formula": "C2H6O", "smiles": "COC", "MW": 46.07, "Tc": 400.0}
        cp = PropertyResolutionResult(40.0 + R_J_MOL_K, "local", "test_cp", 0.99)
        viscosity = PropertyResolutionResult(1.0e-5, "local", "test_mu", 0.99)
        with (
            patch.object(self.resolver, "_get_perry_library", return_value=None),
            patch.object(self.resolver, "resolve_heat_capacity", return_value=cp),
            patch.object(self.resolver, "resolve_viscosity", return_value=viscosity),
        ):
            result = self.resolver.resolve_thermal_conductivity(
                "dimethyl ether", 300.0, "vapor", props, allow_online=False
            )
        self.assertEqual(result.quality, 0.78)
        self.assertIn("quality group ungrouped organic", result.notes)

    def test_vapor_fallback_rounds_input_limit_down(self):
        props = {"formula": "C2H6O", "smiles": "CCO", "MW": 46.07, "Tc": 514.0}
        cp = PropertyResolutionResult(60.0 + R_J_MOL_K, "local", "test_cp", 0.99)
        viscosity = PropertyResolutionResult(1.0e-5, "estimated", "test_mu", 0.755)
        with (
            patch.object(self.resolver, "_get_perry_library", return_value=None),
            patch.object(self.resolver, "resolve_heat_capacity", return_value=cp),
            patch.object(self.resolver, "resolve_viscosity", return_value=viscosity),
        ):
            result = self.resolver.resolve_thermal_conductivity(
                "ethanol", 300.0, "vapor", props, allow_online=False
            )
        self.assertEqual(result.quality, 0.75)
        self.assertLessEqual(result.quality, viscosity.quality)

    def test_vapor_fallback_skips_when_required_tc_is_unavailable(self):
        props = {"formula": "C2H6O", "smiles": "CCO", "MW": 46.07}
        with (
            patch.object(self.resolver, "_get_perry_library", return_value=None),
            patch.object(self.resolver, "resolve_critical_properties", return_value={}),
            patch.object(self.resolver, "resolve_molecular_weight") as resolve_mw,
        ):
            with self.assertRaisesRegex(PropertyResolutionError, "Cannot determine"):
                self.resolver.resolve_thermal_conductivity(
                    "ethanol", 300.0, "vapor", props, allow_online=False
                )
        resolve_mw.assert_not_called()

    def test_liquid_never_calls_vapor_predictive_fallback(self):
        liquid_result = PropertyResolutionResult(
            0.123, "calculated", "test_liquid_fallback", 0.8
        )
        with (
            patch.object(self.resolver, "_get_perry_library", return_value=None),
            patch.object(
                self.resolver,
                "_estimated_liquid_thermal_conductivity",
                return_value=liquid_result,
            ),
            patch.object(
                self.resolver,
                "_estimated_vapor_thermal_conductivity",
                side_effect=AssertionError("gas fallback called for liquid"),
            ),
        ):
            result = self.resolver.resolve_thermal_conductivity(
                "ethanol", 300.0, "liquid", {}, allow_online=False
            )
        self.assertIs(result, liquid_result)

    def test_pfd_parser_accepts_and_round_trips_kg_and_kl(self):
        source = """PROCESS: Conductivity override
VERSION: 1.0
COMPONENTS:
    W | water
PROPERTY_CORRELATIONS:
    W.kg | equation=dippr_eq102, A=0.001, B=1, Tmin_K=250, Tmax_K=400
    W.kl | equation=dippr_eq100, A=0.9, B=-0.001, Tmin_K=250, Tmax_K=400
"""

        pfd = parse_pfd(source)
        component = pfd.get_component("W")

        self.assertEqual(
            component.property_correlations["kg"]["equation"], "dippr_eq102"
        )
        self.assertEqual(
            component.property_correlations["kl"]["equation"], "dippr_eq100"
        )
        serialized = pfd.to_pfd()
        reparsed = parse_pfd(serialized)
        self.assertEqual(
            reparsed.get_component("W").property_correlations,
            component.property_correlations,
        )

    def test_pfd_parser_rejects_vapor_only_equation_for_liquid_conductivity(self):
        source = """PROCESS: Invalid conductivity override
VERSION: 1.0
COMPONENTS:
    W | water
PROPERTY_CORRELATIONS:
    W.kl | equation=dippr_eq102, A=0.001, B=1
"""

        with self.assertRaisesRegex(ParseError, "Unknown kl correlation equation"):
            parse_pfd(source)

    def test_pfd_overrides_reach_resolver_but_are_not_calculated_for_streams(self):
        source = """PROCESS: Conductivity override
VERSION: 1.0
THERMO_METHOD: IDEAL
COMPONENTS:
    W | water
PROPERTY_CORRELATIONS:
    W.kg | equation=dippr_eq102, A=0.001, B=1, Tmin_K=250, Tmax_K=400
    W.kl | equation=dippr_eq100, A=0.9, B=-0.001, Tmin_K=250, Tmax_K=400
STREAM Feed : FEED -> PRODUCT
    T = 25 [C]
    P = 1 [bar]
    F = 1 [kmol/h]
    x = W:1
"""

        with patch.object(
            PropertyResolver,
            "resolve_thermal_conductivity",
            side_effect=AssertionError("streams must not calculate conductivity"),
        ):
            simulator = Simulator.from_string(source)
            result = simulator.run()

        self.assertTrue(result.converged)
        known = simulator.thermo._resolver_known_props["W"]
        self.assertTrue(known["property_correlations"]["kg"]["_pfd_override"])
        self.assertTrue(known["property_correlations"]["kl"]["_pfd_override"])
        conductivity = self.resolver.resolve_thermal_conductivity(
            "W", 300.0, "liquid", known, allow_online=False
        )
        self.assertClose(conductivity.value, 0.6)
        self.assertEqual(conductivity.quality, 1.0)

    def test_correlations_are_strictly_range_limited(self):
        with self.assertRaisesRegex(PropertyResolutionError, "Cannot determine"):
            self.resolver.resolve_thermal_conductivity(
                "water", 200.0, "vapor", {}, allow_online=False
            )

    def test_invalid_state_inputs_are_rejected(self):
        for temperature in (0.0, -1.0, math.inf, math.nan, "not a number"):
            with (
                self.subTest(temperature=temperature),
                self.assertRaisesRegex(
                    PropertyResolutionError,
                    "positive finite",
                ),
            ):
                self.resolver.resolve_thermal_conductivity(
                    "water", temperature, "liquid", {}, allow_online=False
                )

        with self.assertRaisesRegex(
            PropertyResolutionError, "expected liquid or vapor"
        ):
            self.resolver.resolve_thermal_conductivity(
                "water", 300.0, "solid", {}, allow_online=False
            )

    def test_compatibility_facade_uses_the_property_resolver(self):
        result = resolve_thermal_conductivity(
            "water", 300.0, "ideal-gas", {}, allow_online=False
        )

        self.assertEqual(result.method, "perry_vapor_thermal_conductivity_eq102")


if __name__ == "__main__":
    unittest.main()
