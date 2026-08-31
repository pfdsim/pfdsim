import csv
import math
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import brentq

from benchmark_hybrid_curve_fit import generate_curve
from benchmark_lower_hybrid import is_banned
from benchmark_perry_aw import load_curves, percentile
from benchmark_switch_cutoff import build_model
from perry_properties import PerryPropertyLibrary
from pressure_standards import NORMAL_BOILING_PRESSURE_BAR


OUTPUT_PATH = Path("/tmp/hybrid_curve_fit_constrained_benchmark.csv")
MODELS = {
    "constrained_5term": None,
    "constrained_T5": 5,
    "constrained_T6": 6,
}


def design_matrix(temperatures, extra_power):
    temperatures = np.asarray(temperatures, dtype=float)
    columns = [
        np.ones_like(temperatures),
        1.0 / temperatures,
        np.log(temperatures),
        temperatures,
        temperatures**2,
    ]
    if extra_power is not None:
        columns.append(temperatures**extra_power)
    return np.column_stack(columns)


def constrained_fit(points, tb, tc, pc_bar, extra_power):
    temperatures = np.array([point[0] for point in points], dtype=float)
    ln_pressures = np.log([point[1] for point in points])
    matrix = design_matrix(temperatures, extra_power)
    constraints = design_matrix([tb, tc], extra_power)
    targets = np.array(
        [math.log(NORMAL_BOILING_PRESSURE_BAR), math.log(pc_bar)],
        dtype=float,
    )

    scales = np.linalg.norm(matrix, axis=0)
    scaled_matrix = matrix / scales
    scaled_constraints = constraints / scales
    gram = scaled_constraints @ scaled_constraints.T
    particular = scaled_constraints.T @ np.linalg.solve(gram, targets)
    basis = null_space(scaled_constraints)
    reduced_matrix = scaled_matrix @ basis
    reduced_target = ln_pressures - scaled_matrix @ particular
    reduced_coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        reduced_matrix,
        reduced_target,
        rcond=1.0e-13,
    )
    scaled_coefficients = particular + basis @ reduced_coefficients
    coefficients = scaled_coefficients / scales
    predicted = matrix @ coefficients
    constraint_residual = constraints @ coefficients - targets
    return (
        coefficients,
        predicted,
        np.linalg.cond(reduced_matrix),
        constraint_residual,
    )


def fitted_ln_pressure(coefficients, temperature, extra_power):
    return float(design_matrix([temperature], extra_power)[0] @ coefficients)


def fitted_derivative(coefficients, temperature, extra_power):
    _a, b, c, d, e, *rest = coefficients
    derivative = -b / temperature**2 + c / temperature + d + 2.0 * e * temperature
    if extra_power is not None:
        derivative += extra_power * rest[0] * temperature ** (extra_power - 1)
    return derivative


def fitted_temperature(coefficients, extra_power, pressure_bar, lower, upper):
    target = math.log(pressure_bar)
    function = lambda temperature: (
        fitted_ln_pressure(coefficients, temperature, extra_power) - target
    )
    if function(lower) > 0.0 or function(upper) < 0.0:
        return None
    return brentq(function, lower, upper, xtol=1.0e-10, rtol=1.0e-12)


def evaluate_model(curve, model, points, model_name, extra_power):
    coefficients, predicted, condition, constraint_residual = constrained_fit(
        points,
        model["tb"],
        curve.tc,
        curve.pc_bar,
        extra_power,
    )
    errors = []
    temperature_errors = []
    regional_errors = {
        "clapeyron": [],
        "linear_omega_aw": [],
        "plain_aw": [],
    }
    temperatures = [point[0] for point in points]
    lower = min(temperatures)
    upper = max(temperatures)
    for (temperature, pressure_bar, region), predicted_ln in zip(points, predicted):
        error = abs(math.exp(predicted_ln) / pressure_bar - 1.0)
        errors.append(error)
        regional_errors[region].append(error)
        fitted_t = fitted_temperature(
            coefficients,
            extra_power,
            pressure_bar,
            lower,
            upper,
        )
        if fitted_t is not None:
            temperature_errors.append(abs(fitted_t - temperature))

    derivative_temperatures = np.linspace(lower, upper, 1001)
    derivatives = [
        fitted_derivative(coefficients, temperature, extra_power)
        for temperature in derivative_temperatures
    ]
    padded = list(coefficients) + [None] * (6 - len(coefficients))
    return {
        "model": model_name,
        "A": padded[0],
        "B": padded[1],
        "C": padded[2],
        "D": padded[3],
        "E": padded[4],
        "F": padded[5],
        "extra_power": extra_power,
        "point_count": len(points),
        "minimum_pressure_bar": min(point[1] for point in points),
        "full_to_0p001bar": min(point[1] for point in points) <= 0.001 * (1.0 + 1.0e-12),
        "mard": mean(errors),
        "median_abs_relative_error": median(errors),
        "p95_abs_relative_error": percentile(errors, 0.95),
        "max_abs_relative_error": max(errors),
        "temperature_mae_K": mean(temperature_errors),
        "temperature_p95_K": percentile(temperature_errors, 0.95),
        "nonmonotone": min(derivatives) <= 0.0,
        "scaled_reduced_condition_number": condition,
        "tb_constraint_residual": constraint_residual[0],
        "critical_constraint_residual": constraint_residual[1],
        "clapeyron_mard": (
            None
            if not regional_errors["clapeyron"]
            else mean(regional_errors["clapeyron"])
        ),
        "linear_omega_mard": (
            None
            if not regional_errors["linear_omega_aw"]
            else mean(regional_errors["linear_omega_aw"])
        ),
        "plain_aw_mard": (
            None
            if not regional_errors["plain_aw"]
            else mean(regional_errors["plain_aw"])
        ),
    }


def summarize(rows, label):
    print(f"\n{label}: n={len(rows)}")
    for model_name in MODELS:
        subset = [row for row in rows if row["model"] == model_name]
        mards = [row["mard"] for row in subset]
        p95s = [row["p95_abs_relative_error"] for row in subset]
        maxima = [row["max_abs_relative_error"] for row in subset]
        temperature = [row["temperature_mae_K"] for row in subset]
        print(
            f"  {model_name}: MARD median={100*median(mards):.4f}% "
            f"mean={100*mean(mards):.4f}% p95={100*percentile(mards, .95):.4f}%; "
            f"curve-p95 median={100*median(p95s):.4f}%; "
            f"max median={100*median(maxima):.4f}%; "
            f"Tsat MAE median={median(temperature):.4f} K; "
            f"nonmonotone={sum(row['nonmonotone'] for row in subset)}"
        )
        print(
            f"    regional medians: Clapeyron="
            f"{100*median(row['clapeyron_mard'] for row in subset if row['clapeyron_mard'] is not None):.4f}%, "
            f"linear omega={100*median(row['linear_omega_mard'] for row in subset if row['linear_omega_mard'] is not None):.4f}%, "
            f"plain AW={100*median(row['plain_aw_mard'] for row in subset if row['plain_aw_mard'] is not None):.4f}%"
        )

    base = {row["cas"]: row for row in rows if row["model"] == "constrained_5term"}
    for model_name in ("constrained_T5", "constrained_T6"):
        subset = [row for row in rows if row["model"] == model_name]
        print(
            f"  {model_name} wins versus constrained_5term: "
            f"{sum(row['mard'] < base[row['cas']]['mard'] for row in subset)}/{len(subset)}"
        )


def main():
    library = PerryPropertyLibrary()
    library._load()
    rows = []
    unavailable = []
    for curve in load_curves():
        if is_banned(curve):
            continue
        tb_result = library.normal_boiling_point_K(curve.cas)
        if tb_result is None:
            continue
        model = build_model(library, curve, float(tb_result.value))
        if model is None:
            unavailable.append(curve.name)
            continue
        generated = generate_curve(library, curve, model)
        if generated is None:
            unavailable.append(curve.name)
            continue
        points, _transition_pressure = generated
        if len(points) < 20:
            unavailable.append(curve.name)
            continue
        for model_name, extra_power in MODELS.items():
            result = evaluate_model(
                curve,
                model,
                points,
                model_name,
                extra_power,
            )
            result.update(
                {
                    "cas": curve.cas,
                    "name": curve.name,
                    "Tb_K": model["tb"],
                    "Tc_K": curve.tc,
                    "Pc_bar": curve.pc_bar,
                    "omega": curve.omega,
                }
            )
            rows.append(result)

    print("CONSTRAINED FULL-HYBRID Psat REGRESSION")
    print(f"components={len(rows)//len(MODELS)}, unavailable={len(unavailable)}")
    summarize(rows, "all fitted curves")
    full_cas = {
        row["cas"]
        for row in rows
        if row["model"] == "constrained_5term" and row["full_to_0p001bar"]
    }
    summarize([row for row in rows if row["cas"] in full_cas], "full to 0.001 bar")

    maximum_constraint_residual = max(
        max(abs(row["tb_constraint_residual"]), abs(row["critical_constraint_residual"]))
        for row in rows
    )
    print(f"\nmaximum log-pressure constraint residual={maximum_constraint_residual:.3e}")

    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
