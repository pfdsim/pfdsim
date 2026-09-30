"""Compare tangent continuations for temperature-dependent activity fits.

The case matrix intentionally spans NRTL and UNIQUAC, ordinary A+B/T fits,
anchored A+B/T+C*h(T) fits, mild nonwater mixtures, and strongly nonideal
water/organic mixtures.  The script never changes runtime data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermodynamics import create_thermodynamics
from thermodynamics_models.interaction_estimation import _nrtl_ln_gamma

R = 8.31446261815324
NRTL_PATH = ROOT / "data/nrtl_binary_interactions_cas.json"
UNIQUAC_PATH = ROOT / "data/uniquac_binary_interactions_cas.json"

TEST_CASES = {
    "NRTL": (
        ("71-41-0", "110-82-7", "ABC target pentanol/cyclohexane"),
        ("71-41-0", "108-88-3", "ABC target pentanol/toluene"),
        ("71-41-0", "110-54-3", "ABC target pentanol/hexane"),
        ("78-83-1", "71-36-3", "AB similar alcohols"),
        ("78-83-1", "104-76-7", "AB mild alcohol"),
        ("111-27-3", "142-92-7", "AB alcohol/ester"),
        ("64-17-5", "111-87-5", "AB asymmetric alcohol"),
        ("71-36-3", "142-96-1", "AB ether/alcohol"),
        ("7732-18-5", "71-41-0", "ABC water/pentanol"),
        ("7732-18-5", "111-87-5", "ABC water/octanol"),
        ("142-82-5", "7732-18-5", "ABC heptane/water"),
    ),
    "UNIQUAC": (
        ("71-41-0", "110-82-7", "ABC target pentanol/cyclohexane"),
        ("71-41-0", "108-88-3", "ABC target pentanol/toluene"),
        ("71-41-0", "110-54-3", "ABC target pentanol/hexane"),
        ("78-83-1", "71-36-3", "AB similar alcohols"),
        ("78-83-1", "104-76-7", "AB mild alcohol"),
        ("64-17-5", "111-87-5", "AB asymmetric alcohol"),
        ("104-76-7", "143-07-7", "AB alcohol/acid"),
        ("7732-18-5", "71-41-0", "ABC water/pentanol"),
        ("7732-18-5", "111-87-5", "ABC water/octanol"),
        ("142-82-5", "7732-18-5", "ABC heptane/water"),
    ),
}

FORM_LABELS = {
    "constant_inverse": "A+B/T",
    "inverse_linear_quadratic": "M/T+N/T^2",
    "inverse_square_cubic": "M/T^2+N/T^3",
}


def _constant_inverse(value: float, slope: float, boundary: float) -> tuple:
    inverse = -(boundary**2) * slope
    constant = value + boundary * slope
    return constant, inverse


def _inverse_linear_quadratic(
    value: float,
    slope: float,
    boundary: float,
) -> tuple:
    inverse = boundary * (2.0 * value + boundary * slope)
    inverse_squared = -(boundary**2) * (value + boundary * slope)
    return inverse, inverse_squared


def _inverse_square_cubic(
    value: float,
    slope: float,
    boundary: float,
) -> tuple:
    inverse_squared = boundary**2 * (3.0 * value + boundary * slope)
    inverse_cubed = -(boundary**3) * (2.0 * value + boundary * slope)
    return inverse_squared, inverse_cubed


def _continuation_value(form: str, coefficients: tuple, temperature: float) -> float:
    if form == "constant_inverse":
        constant, inverse = coefficients
        return constant + inverse / temperature
    if form == "inverse_linear_quadratic":
        inverse, inverse_squared = coefficients
        return inverse / temperature + inverse_squared / temperature**2
    inverse_squared, inverse_cubed = coefficients
    return inverse_squared / temperature**2 + inverse_cubed / temperature**3


def _continuation_slope(form: str, coefficients: tuple, temperature: float) -> float:
    if form == "constant_inverse":
        return -coefficients[1] / temperature**2
    if form == "inverse_linear_quadratic":
        inverse, inverse_squared = coefficients
        return -inverse / temperature**2 - 2.0 * inverse_squared / temperature**3
    inverse_squared, inverse_cubed = coefficients
    return (
        -2.0 * inverse_squared / temperature**3 - 3.0 * inverse_cubed / temperature**4
    )


FORM_BUILDERS = {
    "constant_inverse": _constant_inverse,
    "inverse_linear_quadratic": _inverse_linear_quadratic,
    "inverse_square_cubic": _inverse_square_cubic,
}


def _load_records(model: str) -> list[dict]:
    path = NRTL_PATH if model == "NRTL" else UNIQUAC_PATH
    return json.loads(path.read_text())["interactions"]


def _record_for_pair(model: str, cas1: str, cas2: str) -> dict:
    target = {cas1, cas2}
    for record in _load_records(model):
        if {record["cas1"], record["cas2"]} == target:
            return record
    raise KeyError(f"No {model} record for {cas1}/{cas2}")


def _directional_coefficients(
    model: str,
    record: dict,
    direction: int,
) -> tuple[float, ...]:
    prefix = "tau12_" if direction == 12 else "tau21_"
    if model == "NRTL":
        suffixes = ("c", "d", "e", "f", "g")
    else:
        suffixes = ("a", "b", "c", "d", "e")
    return tuple(float(record.get(prefix + suffix, 0.0)) for suffix in suffixes)


def _interior_value_slope(
    coefficients: tuple[float, ...],
    temperature: float,
    reference_temperature: float,
) -> tuple[float, float]:
    constant, inverse, anchored, linear, quadratic = coefficients
    h = (reference_temperature - temperature) / temperature + math.log(
        temperature / reference_temperature
    )
    value = (
        constant
        + inverse / temperature
        + anchored * h
        + linear * temperature
        + quadratic * temperature**2
    )
    slope = (
        -inverse / temperature**2
        + anchored * (temperature - reference_temperature) / temperature**2
        + linear
        + 2.0 * quadratic * temperature
    )
    return value, slope


def _parameter_value(
    coefficients: tuple[float, ...],
    temperature: float,
    reference_temperature: float,
    boundary: float,
    form: str,
) -> float:
    if temperature <= boundary:
        return _interior_value_slope(coefficients, temperature, reference_temperature)[
            0
        ]
    value, slope = _interior_value_slope(coefficients, boundary, reference_temperature)
    continuation = FORM_BUILDERS[form](value, slope, boundary)
    return _continuation_value(form, continuation, temperature)


def _validate_forms() -> None:
    probes = (
        (2.5, -0.01, 350.0),
        (-1.25, 0.004, 420.0),
        (0.0, 0.02, 300.0),
    )
    for form, builder in FORM_BUILDERS.items():
        for value, slope, boundary in probes:
            coefficients = builder(value, slope, boundary)
            actual_value = _continuation_value(form, coefficients, boundary)
            actual_slope = _continuation_slope(form, coefficients, boundary)
            if not math.isclose(actual_value, value, rel_tol=1.0e-12, abs_tol=1.0e-12):
                raise AssertionError(f"{form} does not preserve boundary value")
            if not math.isclose(actual_slope, slope, rel_tol=1.0e-12, abs_tol=1.0e-12):
                raise AssertionError(f"{form} does not preserve boundary slope")


def _uniquac_ln_gamma(
    x1: float,
    r: list[float],
    q: list[float],
    q_residual: list[float],
    tau12: float,
    tau21: float,
) -> tuple[float, float]:
    x = (x1, 1.0 - x1)
    rx = sum(r[index] * x[index] for index in range(2))
    qx = sum(q[index] * x[index] for index in range(2))
    qx_residual = sum(q_residual[index] * x[index] for index in range(2))
    ell = [5.0 * (r[index] - q[index]) - (r[index] - 1.0) for index in range(2)]
    xl = sum(x[index] * ell[index] for index in range(2))
    theta = [q_residual[index] * x[index] / qx_residual for index in range(2)]
    tau = ((1.0, tau12), (tau21, 1.0))
    columns = [sum(theta[j] * tau[j][i] for j in range(2)) for i in range(2)]
    result = []
    for i in range(2):
        combinatorial = (
            math.log(r[i] / rx)
            + 5.0 * q[i] * math.log(q[i] * rx / (r[i] * qx))
            + ell[i]
            - (r[i] / rx) * xl
        )
        residual_sum = sum(theta[j] * tau[i][j] / columns[j] for j in range(2))
        residual = q_residual[i] * (1.0 - math.log(columns[i]) - residual_sum)
        result.append(combinatorial + residual)
    return float(result[0]), float(result[1])


def _uniquac_structure(record: dict) -> tuple[list[float], ...]:
    thermo = create_thermodynamics(
        [record["component1"], record["component2"]],
        "UNIQUAC",
    )
    parameters = thermo._uniquac_parameter_matrices()
    return (
        parameters["r_combinatorial"],
        parameters["q_combinatorial"],
        parameters["q_residual"],
    )


def _ln_gamma(
    model: str,
    record: dict,
    structure,
    x1: float,
    temperature: float,
    form: str,
) -> tuple[float, float]:
    reference = float(record.get("tau_tref", 298.15))
    boundary = float(record["Tmax_K"])
    first = _parameter_value(
        _directional_coefficients(model, record, 12),
        temperature,
        reference,
        boundary,
        form,
    )
    second = _parameter_value(
        _directional_coefficients(model, record, 21),
        temperature,
        reference,
        boundary,
        form,
    )
    if model == "NRTL":
        return _nrtl_ln_gamma(x1, first, second, float(record["alpha12"]))
    r, q, q_residual = structure
    return _uniquac_ln_gamma(
        x1,
        r,
        q,
        q_residual,
        math.exp(first),
        math.exp(second),
    )


def _excess_enthalpy(
    model: str,
    record: dict,
    structure,
    temperature: float,
    form: str,
) -> float:
    step = max(1.0e-3, temperature * 1.0e-5)

    def reduced_ge(at_temperature: float) -> float:
        ln1, ln2 = _ln_gamma(model, record, structure, 0.5, at_temperature, form)
        return 0.5 * (ln1 + ln2)

    derivative = (reduced_ge(temperature + step) - reduced_ge(temperature - step)) / (
        2.0 * step
    )
    return -R * temperature**2 * derivative


def _crossings(temperatures: np.ndarray, values: np.ndarray) -> list[float]:
    result = []
    for index in range(len(temperatures) - 1):
        if values[index] * values[index + 1] < 0.0:
            fraction = -values[index] / (values[index + 1] - values[index])
            result.append(
                float(
                    temperatures[index]
                    + fraction * (temperatures[index + 1] - temperatures[index])
                )
            )
    return result


def build_comparison(max_temperature: float, points: int) -> dict:
    _validate_forms()
    payload = {
        "metadata": {
            "description": "Tangent activity-interaction continuation comparison",
            "script": "scripts/activity_fitting/compare_activity_extrapolation_regularizations.py",
            "maximum_temperature_K": max_temperature,
            "grid_points_per_case": points,
            "forms": {
                "constant_inverse": "A+B/T; finite parameter asymptote",
                "inverse_linear_quadratic": (
                    "M/T+N/T^2; zero residual-parameter limit and generally "
                    "finite nonzero HE limit"
                ),
                "inverse_square_cubic": (
                    "M/T^2+N/T^3; zero residual-parameter and HE limits"
                ),
            },
        },
        "cases": {},
    }
    for model, selections in TEST_CASES.items():
        for cas1, cas2, category in selections:
            record = _record_for_pair(model, cas1, cas2)
            structure = _uniquac_structure(record) if model == "UNIQUAC" else None
            boundary = float(record["Tmax_K"])
            temperatures = np.geomspace(boundary, max_temperature, points)
            key = f"{model}: {record['component1']} + {record['component2']}"
            case = {
                "model": model,
                "category": category,
                "cas1": record["cas1"],
                "cas2": record["cas2"],
                "component1": record["component1"],
                "component2": record["component2"],
                "Tmax_K": boundary,
                "forms": {},
            }
            for form in FORM_BUILDERS:
                ln_gamma1 = []
                ln_gamma2 = []
                enthalpy = []
                for temperature in temperatures:
                    first, _ = _ln_gamma(
                        model,
                        record,
                        structure,
                        1.0e-10,
                        float(temperature),
                        form,
                    )
                    _, second = _ln_gamma(
                        model,
                        record,
                        structure,
                        1.0 - 1.0e-10,
                        float(temperature),
                        form,
                    )
                    ln_gamma1.append(first)
                    ln_gamma2.append(second)
                    enthalpy.append(
                        _excess_enthalpy(
                            model, record, structure, float(temperature), form
                        )
                    )
                ln_gamma1 = np.asarray(ln_gamma1)
                ln_gamma2 = np.asarray(ln_gamma2)
                enthalpy = np.asarray(enthalpy)
                samples = {}
                for multiplier in (1.0, 1.5, 2.0, 5.0, 10.0):
                    temperature = boundary * multiplier
                    first, _ = _ln_gamma(
                        model, record, structure, 1.0e-10, temperature, form
                    )
                    _, second = _ln_gamma(
                        model, record, structure, 1.0 - 1.0e-10, temperature, form
                    )
                    samples[str(temperature)] = {
                        "gamma1_infinite_dilution": math.exp(first),
                        "gamma2_infinite_dilution": math.exp(second),
                        "equimolar_HE_J_mol": _excess_enthalpy(
                            model, record, structure, temperature, form
                        ),
                    }
                first, _ = _ln_gamma(
                    model, record, structure, 1.0e-10, max_temperature, form
                )
                _, second = _ln_gamma(
                    model,
                    record,
                    structure,
                    1.0 - 1.0e-10,
                    max_temperature,
                    form,
                )
                samples[str(max_temperature)] = {
                    "gamma1_infinite_dilution": math.exp(first),
                    "gamma2_infinite_dilution": math.exp(second),
                    "equimolar_HE_J_mol": _excess_enthalpy(
                        model, record, structure, max_temperature, form
                    ),
                }
                case["forms"][form] = {
                    "label": FORM_LABELS[form],
                    "gamma1_equals_one_K": _crossings(temperatures, ln_gamma1),
                    "gamma2_equals_one_K": _crossings(temperatures, ln_gamma2),
                    "HE_equals_zero_K": _crossings(temperatures, enthalpy),
                    "gamma1_minimum": float(math.exp(max(ln_gamma1.min(), -700.0))),
                    "gamma2_minimum": float(math.exp(max(ln_gamma2.min(), -700.0))),
                    "HE_minimum_J_mol": float(enthalpy.min()),
                    "HE_maximum_J_mol": float(enthalpy.max()),
                    "samples": samples,
                    "curve": {
                        "temperature_K": temperatures.tolist(),
                        "ln_gamma1_infinite_dilution": ln_gamma1.tolist(),
                        "ln_gamma2_infinite_dilution": ln_gamma2.tolist(),
                        "equimolar_HE_J_mol": enthalpy.tolist(),
                    },
                }
            payload["cases"][key] = case
    return payload


def write_plot(payload: dict, path: Path) -> None:
    figures, axes = plt.subplots(
        3, 2, figsize=(15, 12), sharex="col", constrained_layout=True
    )
    styles = {
        "constant_inverse": ":",
        "inverse_linear_quadratic": "--",
        "inverse_square_cubic": "-",
    }
    for column, model in enumerate(("NRTL", "UNIQUAC")):
        for key, case in payload["cases"].items():
            if case["model"] != model:
                continue
            for form, result in case["forms"].items():
                curve = result["curve"]
                temperature = np.asarray(curve["temperature_K"])
                label = f"{key.removeprefix(model + ': ')} — {FORM_LABELS[form]}"
                axes[0, column].plot(
                    temperature,
                    np.exp(np.clip(curve["ln_gamma1_infinite_dilution"], -700, 700)),
                    linestyle=styles[form],
                    label=label,
                )
                axes[1, column].plot(
                    temperature,
                    np.exp(np.clip(curve["ln_gamma2_infinite_dilution"], -700, 700)),
                    linestyle=styles[form],
                )
                axes[2, column].plot(
                    temperature,
                    curve["equimolar_HE_J_mol"],
                    linestyle=styles[form],
                )
        for row in range(3):
            axes[row, column].set_xscale("log")
            axes[row, column].grid(True, which="both", alpha=0.2)
        axes[0, column].set_yscale("log")
        axes[1, column].set_yscale("log")
        axes[0, column].set_title(model)
        axes[0, column].legend(fontsize=6, ncol=2)
    axes[0, 0].set_ylabel(r"$\gamma_1^\infty$")
    axes[1, 0].set_ylabel(r"$\gamma_2^\infty$")
    axes[2, 0].set_ylabel(r"$H^E(x_1=0.5)$ / J mol$^{-1}$")
    axes[2, 0].set_xlabel("Temperature / K")
    axes[2, 1].set_xlabel("Temperature / K")
    figures.suptitle("Activity-interaction tangent continuation comparison")
    figures.savefig(path, dpi=170)


def compact_payload(payload: dict) -> dict:
    result = json.loads(json.dumps(payload))
    for case in result["cases"].values():
        for form in case["forms"].values():
            form.pop("curve", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-temperature", type=float, default=1.0e7)
    parser.add_argument("--points", type=int, default=1600)
    parser.add_argument("--write-json", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    payload = build_comparison(args.max_temperature, args.points)
    if args.write_json is not None:
        args.write_json.write_text(
            json.dumps(compact_payload(payload), indent=2, sort_keys=True) + "\n"
        )
        print(f"Wrote {args.write_json}")
    if args.plot is not None:
        write_plot(payload, args.plot)
        print(f"Wrote {args.plot}")
    if args.write_json is None and args.plot is None:
        print(json.dumps(compact_payload(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
