"""Physical and source-data checks for the ordinary-basis refitting script."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from unittest.mock import patch

from chemical_properties import ChemicalDatabase
from interaction_parameters import uniquac_binary_interaction
from scripts import build_cas_interaction_parameters as builder

from scripts.activity_fitting.refit_uniquac_common_basis import (
    PreparedCase,
    cases,
    fit,
    ln_gamma,
    minimum_stability,
    observations,
    physical_parameters,
    psat,
    stability_curvature,
    stability_grid,
)
from thermodynamics_models.interaction_estimation import _uniquac_ln_gamma
from thermodynamics_models.nrtl_uniquac import UNIQUACThermodynamics

ROOT = Path(__file__).resolve().parents[1]


def test_vectorized_gamma_matches_shared_uniquac_formula():
    x = np.array([0.0, 1e-8, 0.02, 0.4, 0.95, 1.0])
    r, q = [2.1055, 6.1519], [1.972, 5.212]
    first, second = np.linspace(-1.0, 0.8, len(x)), np.linspace(0.6, -0.4, len(x))
    actual = ln_gamma(x, r, q, q, first, second)
    expected = [
        _uniquac_ln_gamma(float(z), r, q, float(np.exp(a)), float(np.exp(b)))
        for z, a, b in zip(x, first, second)
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert abs(actual[0, 1]) < 1e-12
    assert abs(actual[-1, 0]) < 1e-12


def test_vectorized_gamma_obeys_gibbs_duhem_with_modified_residual_areas():
    x = np.linspace(0.05, 0.95, 13)
    step = 1e-6
    args = ([2.5755, 6.6219], [2.588, 5.828], [0.92, 5.5], -0.7, 0.3)
    slope = (ln_gamma(x + step, *args) - ln_gamma(x - step, *args)) / (2 * step)
    np.testing.assert_allclose(x * slope[:, 0] + (1 - x) * slope[:, 1], 0.0, atol=2e-8)


def test_common_basis_evaluator_matches_production_for_all_five_binaries():
    db = ChemicalDatabase(enable_online=False)
    for definition in cases().values():
        first, second = definition["components"]
        parameters = definition["source_parameters"]
        record = {
            "model": "UNIQUAC",
            "component1": first,
            "component2": second,
            "tau12_a": parameters[0],
            "tau12_b": parameters[1],
            "tau21_a": parameters[2],
            "tau21_b": parameters[3],
            "use_q_prime": False,
        }
        thermo = UNIQUACThermodynamics(
            [first, second], db, interaction_overrides=[record]
        )
        temperature = float(np.mean([row["T_K"] for row in definition["rows"]]))
        x = np.array([0.02, 0.4, 0.95])
        expected = ln_gamma(
            x,
            [thermo.r[first], thermo.r[second]],
            [thermo.q[first], thermo.q[second]],
            [thermo.q[first], thermo.q[second]],
            parameters[0] + parameters[1] / temperature,
            parameters[2] + parameters[3] / temperature,
        )
        with patch.object(thermo, "_compiled_activity_backend", return_value=None):
            actual = [
                thermo.activity_coefficients(
                    temperature, {first: float(z), second: 1 - float(z)}
                )
                for z in x
            ]
        np.testing.assert_allclose(
            [[np.log(row[first]), np.log(row[second])] for row in actual],
            expected,
            atol=1e-12,
        )


def test_source_counts_and_derived_columns_are_identified():
    definitions = cases()
    assert {name: len(observations(row)) for name, row in definitions.items()} == {
        "ethanol_hexanol": 13,
        "ethanol_octanol": 23,
        "ethanol_glycerol": 72,
        "butanol_glycerol": 27,
        "isobutanol_glycerol": 27,
    }
    assert len(definitions["ethanol_glycerol"]["rows"]) == 84
    assert all("y1" not in row for row in definitions["ethanol_hexanol"]["rows"])
    assert all("y1" not in row for row in definitions["ethanol_glycerol"]["rows"])


def test_optimizer_coordinates_preserve_the_temperature_law():
    values, tref = np.array([0.3, -0.5, -0.2, 0.7]), 350.0
    a, b, c, d = physical_parameters(values, "A_plus_B_over_T", tref)
    for temperature in (273.15, 350.0, 467.85):
        np.testing.assert_allclose(
            [a + b / temperature, c + d / temperature],
            [
                values[0] + values[1] * (tref / temperature - 1),
                values[2] + values[3] * (tref / temperature - 1),
            ],
        )


def test_pure_saturation_corrections_cancel_at_the_pure_endpoint():
    class Provider:
        def second_virial_matrix(self, temperature):
            return [[-0.001, -0.0006], [-0.0006, -0.002]]

    definition = cases()["ethanol_glycerol"] | {"vapor": "TSONOPOULOS"}
    case = PreparedCase(
        definition, np.array([2.1055, 4.7957]), np.array([1.972, 4.908]), {}, Provider()
    )
    for x, index in ((1.0, 0), (0.0, 1)):
        temperature = 333.15
        pressure = float(psat(definition["psat"][index], temperature))
        calculated, vapor = case.evaluate(
            [0.4, 80.0, -0.2, -50.0], [{"x1": x, "T_K": temperature, "P_kPa": pressure}]
        )
        np.testing.assert_allclose(calculated, [pressure], rtol=1e-12)
        np.testing.assert_allclose(vapor, [x], atol=1e-12)


def test_stability_audit_detects_an_accurate_but_spuriously_immiscible_fit():
    definition = cases()["isobutanol_glycerol"]
    case = PreparedCase(
        definition, np.array([3.4535, 4.7957]), np.array([3.048, 4.908]), {}, None
    )
    assert (
        np.min(
            stability_curvature(case, [0.0, -1.456530749348, 0.0, -176.154921750857])
        )
        < -0.1
    )
    assert np.min(stability_curvature(case, [0.0, 0.0, 0.0, 0.0])) > 0.0


def test_continuous_stability_audit_refines_between_sampled_compositions():
    definition = cases()["isobutanol_glycerol"]
    case = PreparedCase(
        definition, np.array([3.4535, 4.7957]), np.array([3.048, 4.908]), {}, None
    )
    parameters = [0.0, -1.456530749348, 0.0, -176.154921750857]
    points = stability_grid(case, 3, 3)
    scale = brentq(
        lambda factor: (
            np.min(
                stability_curvature(
                    case, np.asarray(parameters) * factor, points=points
                )
            )
            - 0.02
        ),
        0.0,
        1.0,
    )
    parameters = np.asarray(parameters) * scale
    assert np.min(stability_curvature(case, parameters, points=points)) > 0
    assert minimum_stability(case, parameters, points)[0] < 0


def test_stable_refit_retains_small_pressure_residuals():
    definition = cases()["isobutanol_glycerol"]
    case = PreparedCase(
        definition, np.array([3.4535, 4.7957]), np.array([3.048, 4.908]), {}, None
    )
    rows = observations(definition)
    fitted = fit(case, rows, "B_over_T", initial=[0.0, -0.55])
    assert fitted["stability_constraint_active"]
    assert fitted["minimum_total_G_over_RT_curvature"] >= 0
    pressure, _ = case.evaluate(fitted["parameters"], rows)
    observed = np.array([row["P_kPa"] for row in rows])
    assert np.sqrt(np.mean((pressure - observed) ** 2)) < 0.2


def test_active_refits_match_the_reviewed_coefficients_in_both_directions():
    payload = json.loads(
        (ROOT / "data/source/activity_fitting/uniquac_common_basis_refits.json").read_text()
    )
    runtime = json.loads(
        (ROOT / "data/uniquac_binary_interactions_cas.json").read_text()
    )
    for record in payload["interactions"]:
        forward = uniquac_binary_interaction(record["cas1"], record["cas2"])
        reverse = uniquac_binary_interaction(record["cas2"], record["cas1"])
        assert forward["use_q_prime"] is False
        for suffix in ("a", "b"):
            np.testing.assert_allclose(
                forward[f"tau12_{suffix}"], record[f"tau12_{suffix}"], rtol=1e-12
            )
            np.testing.assert_allclose(
                forward[f"tau21_{suffix}"], record[f"tau21_{suffix}"], rtol=1e-12
            )
            assert reverse[f"tau12_{suffix}"] == forward[f"tau21_{suffix}"]
            assert reverse[f"tau21_{suffix}"] == forward[f"tau12_{suffix}"]
        matching = [
            row
            for row in runtime["interactions"]
            if {row["cas1"], row["cas2"]} == {record["cas1"], record["cas2"]}
            and not row.get("disabled")
        ]
        assert len(matching) == 1
        assert (
            matching[0]["source_file"] == "data/source/activity_fitting/uniquac_common_basis_refits.json"
        )
        assert matching[0]["fit_status"] == "recommended_common_basis_refit"
        assert "fit_quality_flags" in matching[0]


def test_zero_concentration_components_do_not_change_the_ethanol_water_binary():
    db = ChemicalDatabase(enable_online=False)
    composition = {"ethanol": 0.4, "water": 0.6}
    binary = UNIQUACThermodynamics(list(composition), db)
    mixture = UNIQUACThermodynamics(
        list(composition)
        + ["1-hexanol", "1-octanol", "glycerol", "1-butanol", "isobutanol"],
        db,
    )
    for temperature in (300.0, 350.0, 400.0):
        expected = binary.activity_coefficients(temperature, composition)
        actual = mixture.activity_coefficients(temperature, composition)
        for component in composition:
            np.testing.assert_allclose(
                actual[component], expected[component], rtol=1e-12
            )
        with patch.object(mixture, "_compiled_activity_backend", return_value=None):
            mixture._activity_cache.clear()
            scalar = mixture.activity_coefficients(temperature, composition)
        for component in composition:
            np.testing.assert_allclose(
                scalar[component], expected[component], rtol=1e-12
            )


def test_builder_rejects_an_incomplete_common_basis_report():
    actual_load = builder.load_json

    def partial_load(path):
        payload = actual_load(path)
        if path.name == builder.COMMON_BASIS_UNIQUAC_FILE:
            payload["interactions"] = payload["interactions"][:-1]
        return payload

    with patch.object(builder, "load_json", side_effect=partial_load):
        with pytest.raises(ValueError, match="exactly the five reviewed pairs"):
            builder.supplemental_curated_water_nonwater_records([], "UNIQUAC")
