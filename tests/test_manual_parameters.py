"""Raw parameter conventions, runtime equivalence and read-only previews."""

from copy import deepcopy

import numpy as np
import pytest

from physical_constants import R_J_MOL_K
from thermodynamics_models.manual_parameters import (
    normalize_manual_parameters, prepare_manual_parameters, preview_manual_parameters,
)
from thermodynamics_models.interaction_fitting import validate_runtime_inclusion


def manual(model="NRTL", basis="energy", **overrides):
    return {"components": ["ethanol", "water"], "model": model, "basis": basis,
            "values": {"12": 900, "21": -200}, **overrides}


@pytest.mark.parametrize("model", ["NRTL", "UNIQUAC"])
@pytest.mark.parametrize("unit,factor", [("J/mol", 1), ("kJ/mol", .001), ("cal/mol", 1 / 4.184), ("kcal/mol", 1 / 4184)])
def test_energy_units_match_tau_law_and_runtime(model, unit, factor):
    energy, thermo = prepare_manual_parameters(manual(model, unit=unit, values={"12": 900 * factor, "21": -200 * factor}))
    fields = "cdefg" if model == "NRTL" else "abcde"
    coefficients = {f"{direction}.{field}": 0 for direction in ("12", "21") for field in fields}
    sign = 1 if model == "NRTL" else -1
    for direction, value in (("12", 900), ("21", -200)):
        coefficients[f"{direction}.{fields[1]}"] = sign * value / R_J_MOL_K
    _, law = prepare_manual_parameters(manual(model, "law", values=coefficients))
    for T in (290, 330, 380):
        composition = dict(zip(energy["components"], (.37, .63)))
        assert np.allclose(list(thermo.activity_coefficients(T, composition).values()), list(law.activity_coefficients(T, composition).values()), rtol=1e-12)
        assert thermo.excess_enthalpy(composition, T) == pytest.approx(law.excess_enthalpy(composition, T), abs=1e-7)
    assert validate_runtime_inclusion(energy)["origin"] == "manual_parameters"
    assert energy["request"]["observations"] == []


def test_positive_uniquac_tau_is_logged_and_source_statistics_survive():
    result, _ = prepare_manual_parameters(manual("UNIQUAC", "tau", values={"12": .6, "21": 1.2}, fit_method="Least squares", statistics="RMS 1.2%; 42 data points"))
    assert result["parameters"]["tau12_a"] == pytest.approx(np.log(.6))
    assert result["parameters"]["tau21_a"] == pytest.approx(np.log(1.2))
    assert result["reported_fit"] == {"method": "Least squares", "statistics": "RMS 1.2%; 42 data points"}
    assert result["objectives"] == {}
    validate_runtime_inclusion(result)
    changed = deepcopy(result)
    changed["parameters"]["tau12_a"] *= -1
    with pytest.raises(ValueError, match="disagree"):
        validate_runtime_inclusion(changed)
    changed = deepcopy(result)
    changed["rq"] = [{"r": 2, "q": 2}, {"r": 1, "q": 1}]
    with pytest.raises(ValueError, match="disagree"):
        validate_runtime_inclusion(changed)


@pytest.mark.parametrize("data", [
    manual(values={"12": float("nan"), "21": 1}),
    manual(values={"12": True, "21": 1}),
    manual(values={"12": 1}),
    manual("UNIQUAC", "tau", values={"12": 0, "21": 1}),
    manual(Tmin_K=300), manual(Tmin_K=400, Tmax_K=300),
    manual(alpha=-.1), manual(extrapolation="clamp"), manual(statistics=12),
])
def test_invalid_manual_parameters_are_rejected(data):
    with pytest.raises(ValueError):
        normalize_manual_parameters(data)


def test_aliases_for_the_same_chemical_are_rejected():
    with pytest.raises(ValueError, match="distinct resolved CAS"):
        prepare_manual_parameters(manual(components=["ethanol", "64-17-5"]))


def test_zero_nonrandomness_can_be_entered_without_fitting_bounds():
    result, thermo = prepare_manual_parameters(manual("NRTL", "tau", alpha=0, values={"12": 1, "21": 2}))
    assert result["parameters"]["alpha12"] == 0
    composition = dict(zip(result["components"], (.5, .5)))
    assert all(value > 0 for value in thermo.activity_coefficients(300, composition).values())
    validate_runtime_inclusion(result)


def test_overflowing_law_is_rejected_before_publication():
    result, _ = prepare_manual_parameters(manual("NRTL", "tau", values={"12": -1e300, "21": -1e300}))
    with pytest.raises(ValueError, match="parameter law"):
        validate_runtime_inclusion(result)


@pytest.mark.parametrize("kind", ["GAMMA", "VLE", "LLE"])
def test_previews_use_entered_parameters_and_return_finite_curves(kind):
    output = preview_manual_parameters({"parameters": manual(), "kind": kind, "T_K": 330, "n_points": 10}, lambda message: None)
    assert output["result"]["manual_input"]["values"] == {"12": 900, "21": -200}
    assert output["plots"][0]["kind"] == kind
    assert not output["plots"][0]["errors"]
    for series in output["plots"][0]["series"]:
        assert len(series["x"]) == len(series["y"]) == 11
        assert all(np.isfinite(v) for v in series["y"])
    if kind == "VLE":
        assert np.allclose(output["plots"][0]["series"][0]["y"], output["plots"][0]["series"][1]["y"], rtol=1e-5)


def test_lle_preview_shows_a_real_split():
    output = preview_manual_parameters({"parameters": manual("NRTL", "tau", values={"12": 4, "21": 4}), "kind": "LLE", "T_K": 300, "n_points": 10}, lambda message: None)
    a, b = output["plots"][0]["series"]
    assert not output["plots"][0]["errors"]
    assert a["y"][5] < .2 and b["y"][5] > .8
