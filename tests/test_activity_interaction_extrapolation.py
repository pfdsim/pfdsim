import math
import unittest
from unittest.mock import patch

import numpy as np

import compiled_vlle
from interaction_parameters import (
    nrtl_binary_interaction,
    uniquac_binary_interaction,
)
from scripts.build_cas_interaction_parameters import (
    supplemental_literature_vle_activity_records,
)
from thermodynamics import create_thermodynamics
from thermodynamics_models.common import ThermodynamicsError


class ActivityInteractionExtrapolationTests(unittest.TestCase):
    @staticmethod
    def _record(
        model,
        component1,
        component2,
        *,
        extrapolation=None,
    ):
        record = {
            "component1": component1,
            "component2": component2,
            "model": model,
            "Tmin_K": 300.0,
            "Tmax_K": 350.0,
        }
        if extrapolation is not None:
            record["extrapolation"] = extrapolation
        if model == "NRTL":
            record.update(
                {
                    "alpha12": 0.3,
                    "tau12_c": 0.1,
                    "tau12_d": 70.0,
                    "tau21_c": -0.2,
                    "tau21_d": 35.0,
                }
            )
        else:
            record.update(
                {
                    "tau12_a": 0.1,
                    "tau12_b": 70.0,
                    "tau21_a": -0.2,
                    "tau21_b": 35.0,
                }
            )
        return record

    def test_omitted_or_unrestricted_policy_extrapolates_past_declared_range(self):
        for model in ("NRTL", "UNIQUAC"):
            for extrapolation in (None, "unrestricted"):
                with self.subTest(model=model, extrapolation=extrapolation):
                    record = self._record(
                        model, "water", "ethanol", extrapolation=extrapolation
                    )
                    thermo = create_thermodynamics(
                        ["water", "ethanol"],
                        model,
                        interaction_overrides=[record],
                    )
                    if model == "NRTL":
                        actual = thermo._nrtl_matrices(400.0)[0][0][1]
                        expected = 0.1 + 70.0 / 400.0
                        clamped = 0.1 + 70.0 / 350.0
                    else:
                        actual = thermo._uniquac_tau_matrix(400.0)[0][1]
                        expected = math.exp(0.1 + 70.0 / 400.0)
                        clamped = math.exp(0.1 + 70.0 / 350.0)
                    self.assertAlmostEqual(actual, expected)
                    self.assertNotAlmostEqual(actual, clamped)
                    self.assertFalse(
                        any("declared range" in warning for warning in thermo.warnings)
                    )

    def test_true_policy_requires_a_valid_two_sided_range(self):
        for model in ("NRTL", "UNIQUAC"):
            base = self._record(model, "water", "ethanol", extrapolation="clamp")
            cases = (
                (
                    {key: value for key, value in base.items() if key != "Tmin_K"},
                    "requires Tmin_K and Tmax_K",
                ),
                (base | {"Tmin_K": 350.0, "Tmax_K": 300.0}, "0 < Tmin_K <= Tmax_K"),
                (base | {"extrapolation": "unknown"}, "must be one of"),
            )
            for record, message in cases:
                with self.subTest(model=model, message=message):
                    with self.assertRaisesRegex(ThermodynamicsError, message):
                        create_thermodynamics(
                            ["water", "ethanol"],
                            model,
                            interaction_overrides=[record],
                        )

    def test_single_temperature_range_accepts_energy_and_tau_records(self):
        for model in ("NRTL", "UNIQUAC"):
            for mode in ("unrestricted", "clamp", "constant_inverse", "inverse_linear_quadratic", "inverse_square_cubic"):
                for energy in (False, True):
                    with self.subTest(model=model, mode=mode, energy=energy):
                        record = self._record(model, "water", "ethanol", extrapolation=mode)
                        record["Tmax_K"] = record["Tmin_K"]
                        if energy:
                            record = {key: value for key, value in record.items() if not key.startswith("tau")}
                            record.update(a12_cal_per_mol=70, a21_cal_per_mol=35)
                        thermo = create_thermodynamics(
                            ["water", "ethanol"], model, interaction_overrides=[record]
                        )
                        for temperature in (290, 300, 310):
                            gamma = thermo.activity_coefficients(temperature, {"water": 0.4, "ethanol": 0.6})
                            self.assertTrue(all(math.isfinite(value) and value > 0 for value in gamma.values()))
                            self.assertTrue(math.isfinite(thermo.excess_enthalpy({"water": 0.4, "ethanol": 0.6}, temperature)))

    def test_tangent_regularizations_are_c1_and_match_compiled_backends(self):
        for model in ("NRTL", "UNIQUAC"):
            for mode, encoded in (
                ("constant_inverse", 21),
                ("inverse_linear_quadratic", 31),
                ("inverse_square_cubic", 41),
            ):
                record = self._record(
                    model,
                    "water",
                    "ethanol",
                    extrapolation=mode,
                )
                if model == "NRTL":
                    record.update(
                        {
                            "tau12_e": 1.7,
                            "tau12_f": 2.0e-4,
                            "tau12_g": -3.0e-7,
                            "tau21_e": -0.8,
                            "tau_tref": 325.0,
                        }
                    )
                else:
                    record.update(
                        {
                            "tau12_c": 1.7,
                            "tau12_d": 2.0e-4,
                            "tau12_e": -3.0e-7,
                            "tau21_c": -0.8,
                            "tau_tref": 325.0,
                        }
                    )
                with self.subTest(model=model, mode=mode):
                    self._assert_tangent_regularization(model, mode, encoded, record)

    def _assert_tangent_regularization(self, model, mode, encoded, record):
        thermo = create_thermodynamics(
            ["water", "ethanol"],
            model,
            interaction_overrides=[record],
        )

        def parameter(temperature):
            if model == "NRTL":
                return thermo._nrtl_matrices(temperature)[0][0][1]
            return math.log(thermo._uniquac_tau_matrix(temperature)[0][1])

        boundary = 350.0
        step = 1.0e-4
        value = parameter(boundary)
        left_slope = (value - parameter(boundary - step)) / step
        right_slope = (parameter(boundary + step) - value) / step
        self.assertAlmostEqual(left_slope, right_slope, places=6)

        hot = 500.0
        if mode == "constant_inverse":
            outside_b = -boundary * boundary * left_slope
            outside_a = value + boundary * left_slope
            expected = outside_a + outside_b / hot
        elif mode == "inverse_linear_quadratic":
            outside_b = boundary * (2.0 * value + boundary * left_slope)
            outside_c = -(boundary**2) * (value + boundary * left_slope)
            expected = outside_b / hot + outside_c / hot**2
        else:
            outside_c = boundary**2 * (3.0 * value + boundary * left_slope)
            outside_d = -(boundary**3) * (2.0 * value + boundary * left_slope)
            expected = outside_c / hot**2 + outside_d / hot**3
        self.assertAlmostEqual(parameter(hot), expected, places=6)
        regularization_warnings = [
            warning for warning in thermo.warnings if mode in warning
        ]
        self.assertEqual(len(regularization_warnings), 1)
        unregularized = dict(record)
        unregularized["extrapolation"] = "unrestricted"
        raw = create_thermodynamics(
            ["water", "ethanol"],
            model,
            interaction_overrides=[unregularized],
        )
        raw_parameter = (
            raw._nrtl_matrices(hot)[0][0][1]
            if model == "NRTL"
            else math.log(raw._uniquac_tau_matrix(hot)[0][1])
        )
        self.assertNotAlmostEqual(parameter(hot), raw_parameter)

        lower_boundary = 300.0
        lower_value = parameter(lower_boundary)
        interior_slope = (parameter(lower_boundary + step) - lower_value) / step
        exterior_slope = (lower_value - parameter(lower_boundary - step)) / step
        self.assertAlmostEqual(interior_slope, exterior_slope, places=6)

        backend = thermo._compiled_activity_backend(hot)
        if backend is None:
            self.skipTest(f"Compiled {model} backend is unavailable")
        composition = {"water": 0.4, "ethanol": 0.6}
        compiled = backend.activity_coefficients([0.4, 0.6], hot)
        compiled_he = backend.excess_enthalpy([0.4, 0.6], hot)
        lle = thermo._compiled_lle_backend(hot)
        self.assertIsNotNone(lle)
        self.assertEqual(lle.tau_mode[0, 1], encoded)
        vlle = thermo.compiled_vlle_backend()
        self.assertIsNotNone(vlle)
        self.assertEqual(vlle.integer_parameters[0, 1], encoded)
        thermo._compiled_activity_backend = lambda _T: None
        interpreted = thermo.activity_coefficients(hot, composition)
        interpreted_he = thermo.excess_enthalpy(composition, hot)
        self.assertAlmostEqual(compiled[0], interpreted["water"], places=12)
        self.assertAlmostEqual(compiled[1], interpreted["ethanol"], places=12)
        self.assertAlmostEqual(compiled_he, interpreted_he, places=6)
        vlle_gamma = compiled_vlle._activity_gamma(
            vlle.model_id,
            np.asarray([0.4, 0.6], dtype=np.float64),
            hot,
            vlle.integer_parameters,
            vlle.parameter_0,
            vlle.parameter_1,
            vlle.parameter_2,
            vlle.parameter_3,
            vlle.parameter_4,
            vlle.parameter_5,
            vlle.parameter_6,
            vlle.parameter_7,
            vlle.parameter_8,
            vlle.parameter_9,
        )
        self.assertAlmostEqual(vlle_gamma[0], compiled[0], places=12)
        self.assertAlmostEqual(vlle_gamma[1], compiled[1], places=12)

    def test_regularization_policy_validation(self):
        for model in ("NRTL", "UNIQUAC"):
            base = self._record(
                model,
                "water",
                "ethanol",
                extrapolation="inverse_square_cubic",
            )
            cases = (
                (
                    {key: value for key, value in base.items() if key != "Tmin_K"},
                    "requires Tmin_K and Tmax_K",
                ),
                (base | {"extrapolation": True}, "must be a string"),
                (base | {"extrapolation": "perhaps"}, "must be one of"),
            )
            for record, message in cases:
                with self.subTest(model=model, message=message):
                    with self.assertRaisesRegex(ThermodynamicsError, message):
                        create_thermodynamics(
                            ["water", "ethanol"],
                            model,
                            interaction_overrides=[record],
                        )

    def test_energy_parameters_use_the_same_continuation_as_equivalent_tau_law(self):
        composition = {"water": 0.4, "ethanol": 0.6}
        for model in ("NRTL", "UNIQUAC"):
            for mode in (
                "constant_inverse",
                "inverse_linear_quadratic",
                "inverse_square_cubic",
            ):
                with self.subTest(model=model, mode=mode):
                    energy = {
                        "model": model,
                        "component1": "water",
                        "component2": "ethanol",
                        "alpha12": 0.3,
                        "a12_cal_per_mol": 500.0,
                        "a21_cal_per_mol": 100.0,
                        "extrapolation": mode,
                        "Tmin_K": 300.0,
                        "Tmax_K": 350.0,
                    }
                    thermo = create_thermodynamics(
                        list(composition), model, interaction_overrides=[energy]
                    )
                    equivalent = self._record(
                        model, "water", "ethanol", extrapolation=mode
                    )
                    if model == "NRTL":
                        equivalent.update(
                            tau12_c=0.0,
                            tau12_d=500.0 / thermo.R_CAL,
                            tau21_c=0.0,
                            tau21_d=100.0 / thermo.R_CAL,
                        )
                    else:
                        equivalent.update(
                            tau12_a=0.0,
                            tau12_b=-500.0 / thermo.R_CAL,
                            tau21_a=0.0,
                            tau21_b=-100.0 / thermo.R_CAL,
                        )
                    reference = create_thermodynamics(
                        list(composition), model, interaction_overrides=[equivalent]
                    )
                    backend = thermo._compiled_activity_backend()
                    with (
                        patch.object(
                            thermo, "_compiled_activity_backend", return_value=None
                        ),
                        patch.object(
                            reference, "_compiled_activity_backend", return_value=None
                        ),
                    ):
                        for temperature in (250.0, 325.0, 500.0):
                            expected = reference.activity_coefficients(
                                temperature, composition
                            )
                            actual = thermo.activity_coefficients(
                                temperature, composition
                            )
                            for component in composition:
                                self.assertAlmostEqual(
                                    actual[component], expected[component], places=12
                                )
                            if model == "UNIQUAC":
                                subset = thermo._activity_coefficients_for_components(
                                    temperature, composition, list(composition)
                                )
                                for component in composition:
                                    self.assertAlmostEqual(
                                        subset[component],
                                        expected[component],
                                        places=12,
                                    )
                            if backend is not None:
                                compiled = backend.activity_coefficients(
                                    [0.4, 0.6], temperature
                                )
                                for index, component in enumerate(composition):
                                    self.assertAlmostEqual(
                                        compiled[index], expected[component], places=12
                                    )
                                self.assertAlmostEqual(
                                    backend.excess_enthalpy([0.4, 0.6], temperature),
                                    reference.excess_enthalpy(composition, temperature),
                                    places=6,
                                )

    def test_compiled_evaluation_reports_every_continuation_without_scalar_warmup(self):
        for model in ("NRTL", "UNIQUAC"):
            for mode in (
                "constant_inverse",
                "inverse_linear_quadratic",
                "inverse_square_cubic",
            ):
                with self.subTest(model=model, mode=mode):
                    thermo = create_thermodynamics(
                        ["water", "ethanol"],
                        model,
                        interaction_overrides=[
                            self._record(model, "water", "ethanol", extrapolation=mode)
                        ],
                    )
                    if thermo._compiled_activity_backend() is None:
                        self.skipTest(f"Compiled {model} backend is unavailable")
                    for temperature in (500.0, 550.0, 250.0, 225.0):
                        thermo.activity_coefficients(
                            temperature, {"water": 0.4, "ethanol": 0.6}
                        )
                    messages = [
                        warning for warning in thermo.warnings if mode in warning
                    ]
                    self.assertEqual(len(messages), 2)
                    self.assertIn("500 K", messages[0])
                    self.assertIn("250 K", messages[1])

    def test_clamping_is_pair_specific_and_matches_compiled_backend(self):
        composition = {"water": 0.4, "ethanol": 0.35, "methanol": 0.25}
        for model in ("NRTL", "UNIQUAC"):
            overrides = [
                self._record(model, "water", "ethanol", extrapolation="clamp"),
                self._record(model, "water", "methanol"),
            ]
            with self.subTest(model=model):
                thermo = create_thermodynamics(
                    list(composition), model, interaction_overrides=overrides
                )
                if model == "NRTL":
                    hot = thermo._nrtl_matrices(400.0)[0]
                    self.assertAlmostEqual(hot[0][1], 0.1 + 70.0 / 350.0)
                    self.assertAlmostEqual(hot[0][2], 0.1 + 70.0 / 400.0)
                else:
                    hot = thermo._uniquac_tau_matrix(400.0)
                    self.assertAlmostEqual(hot[0][1], math.exp(0.1 + 70.0 / 350.0))
                    self.assertAlmostEqual(hot[0][2], math.exp(0.1 + 70.0 / 400.0))

                backend = thermo._compiled_activity_backend(400.0)
                if backend is None:
                    self.skipTest(f"Compiled {model} backend is unavailable")
                x = [composition[component] for component in thermo.components]
                compiled = backend.activity_coefficients(x, 400.0)

                reference = create_thermodynamics(
                    list(composition), model, interaction_overrides=overrides
                )
                reference._compiled_activity_backend = lambda _T: None
                generic = reference.activity_coefficients(400.0, composition)
                for index, component in enumerate(thermo.components):
                    self.assertAlmostEqual(
                        compiled[index], generic[component], places=12
                    )

                thermo.activity_coefficients(400.0, composition)
                thermo.activity_coefficients(425.0, composition)
                warnings = [
                    warning
                    for warning in thermo.warnings
                    if "extrapolation=clamp" in warning
                ]
                self.assertEqual(len(warnings), 1)
                self.assertIn("350 K instead of 400 K", warnings[0])
                thermo.activity_coefficients(275.0, composition)
                warnings = [
                    warning
                    for warning in thermo.warnings
                    if "extrapolation=clamp" in warning
                ]
                self.assertEqual(len(warnings), 2)
                self.assertIn("300 K instead of 275 K", warnings[1])

                lle = thermo._compiled_lle_backend(400.0)
                self.assertIsNotNone(lle)
                self.assertEqual(lle.interaction_tmax[0, 1], 350.0)
                self.assertTrue(math.isinf(lle.interaction_tmax[0, 2]))

                vlle = thermo.compiled_vlle_backend()
                self.assertIsNotNone(vlle)
                if model == "NRTL":
                    self.assertEqual(vlle.parameter_9[0, 1], 350.0)
                    self.assertTrue(math.isinf(vlle.parameter_9[0, 2]))
                else:
                    self.assertEqual(vlle.parameter_9[0, 1], 300.0)
                    self.assertEqual(vlle.parameter_9[1, 0], 350.0)
                    self.assertTrue(math.isinf(vlle.parameter_9[2, 0]))
                activity_gamma = getattr(compiled_vlle, "_activity_gamma", None)
                if activity_gamma is None:
                    self.skipTest("Compiled activity VLLE kernel is unavailable")
                vlle_gamma = activity_gamma(
                    vlle.model_id,
                    np.asarray(x, dtype=np.float64),
                    400.0,
                    vlle.integer_parameters,
                    vlle.parameter_0,
                    vlle.parameter_1,
                    vlle.parameter_2,
                    vlle.parameter_3,
                    vlle.parameter_4,
                    vlle.parameter_5,
                    vlle.parameter_6,
                    vlle.parameter_7,
                    vlle.parameter_8,
                    vlle.parameter_9,
                )
                for index in range(len(compiled)):
                    self.assertAlmostEqual(
                        vlle_gamma[index], compiled[index], places=12
                    )

    def test_runtime_warning_checks_use_prepared_bounds_without_resolving_pairs(self):
        for model, resolver_name in (
            ("NRTL", "_nrtl_interaction_for_components"),
            ("UNIQUAC", "_uniquac_interaction_for_components"),
        ):
            with self.subTest(model=model):
                thermo = create_thermodynamics(
                    ["water", "ethanol"],
                    model,
                    interaction_overrides=[
                        self._record(model, "water", "ethanol", extrapolation="clamp"),
                    ],
                )
                with patch.object(
                    thermo,
                    resolver_name,
                    side_effect=AssertionError("interaction pair was re-resolved"),
                ):
                    for temperature in (325.0, 400.0, 425.0, 275.0, 250.0):
                        thermo._warn_activity_interaction_extrapolation(temperature)

                warnings = [
                    warning
                    for warning in thermo.warnings
                    if "extrapolation=clamp" in warning
                ]
                self.assertEqual(len(warnings), 2)
                self.assertIn("350 K instead of 400 K", warnings[0])
                self.assertIn("300 K instead of 275 K", warnings[1])

    def test_water_propionic_uniquac_policy_is_built_and_runtime_visible(self):
        records, _, _ = supplemental_literature_vle_activity_records([], "UNIQUAC")
        built = next(
            record
            for record in records
            if {record["cas1"], record["cas2"]} == {"7732-18-5", "79-09-4"}
        )
        self.assertEqual(built["extrapolation"], "clamp")
        self.assertEqual((built["Tmin_K"], built["Tmax_K"]), (313.15, 373.15))

        uniquac = uniquac_binary_interaction("7732-18-5", "79-09-4")
        self.assertEqual(uniquac["extrapolation"], "clamp")
        self.assertEqual(
            (uniquac["Tmin_K"], uniquac["Tmax_K"]),
            (313.15, 373.15),
        )
        nrtl = nrtl_binary_interaction("7732-18-5", "79-09-4")
        self.assertEqual(nrtl.get("extrapolation"), "unrestricted")

        thermo = create_thermodynamics(["water", "propionic acid"], "UNIQUAC")
        hot = thermo._uniquac_tau_matrix(626.45)
        at_limit = thermo._uniquac_tau_matrix(373.15)
        self.assertEqual(hot, at_limit)
        K = thermo.K_values(
            626.45,
            1.01,
            {"water": 0.999, "propionic acid": 0.001},
        )
        self.assertGreater(min(K.values()), 1.0)

    def test_estimated_interactions_inherit_declared_clamping_policy(self):
        thermo = create_thermodynamics(
            ["water", "2,3-pentanedione"],
            "UNIQUAC",
            interaction_estimation=[
                {
                    "model": "UNIQUAC",
                    "source": "UNIFDMD",
                    "policy": "missing_only",
                    "parameter_order": "source",
                    "extrapolation": "clamp",
                    "Tmin_K": 293.15,
                    "Tmax_K": 423.15,
                }
            ],
        )
        interaction = thermo._uniquac_interaction_for_components(
            "water", "2,3-pentanedione"
        )
        self.assertEqual(interaction["extrapolation"], "clamp")
        self.assertEqual(interaction["Tmin_K"], 293.15)
        self.assertEqual(interaction["Tmax_K"], 423.15)
        self.assertEqual(
            thermo._uniquac_tau_matrix(600.0),
            thermo._uniquac_tau_matrix(423.15),
        )

        regularized = create_thermodynamics(
            ["water", "2,3-pentanedione"],
            "UNIQUAC",
            interaction_estimation=[
                {
                    "model": "UNIQUAC",
                    "source": "UNIFDMD",
                    "policy": "missing_only",
                    "parameter_order": "source",
                    "extrapolation": "inverse_square_cubic",
                    "Tmin_K": 293.15,
                    "Tmax_K": 423.15,
                }
            ],
        )
        interaction = regularized._uniquac_interaction_for_components(
            "water", "2,3-pentanedione"
        )
        self.assertEqual(interaction["extrapolation"], "inverse_square_cubic")
        self.assertNotEqual(
            regularized._uniquac_tau_matrix(600.0),
            regularized._uniquac_tau_matrix(423.15),
        )
        self.assertEqual(
            regularized._uniquac_parameter_matrices()["tau_mode"][0][1],
            41,
        )


if __name__ == "__main__":
    unittest.main()
