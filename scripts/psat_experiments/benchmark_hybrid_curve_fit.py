import csv
import math
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from benchmark_lower_clapeyron import correlation_hvap_kJ_mol
from benchmark_lower_hybrid import is_banned
from benchmark_lower_psat import invert_ln_pressure
from benchmark_perry_aw import aw_ln_p, load_curves, percentile
from benchmark_switch_cutoff import build_model, linear_omega_ln_pressure
from perry_properties import PerryPropertyLibrary
from property_resolution.common import R


START_PRESSURE_BAR = 1.01325
SWITCH_PRESSURE_BAR = 0.25
MINIMUM_PRESSURE_BAR = 0.001
PRESSURE_STEP_UP = 1.05
PRESSURE_STEP_DOWN = 0.95
OUTPUT_PATH = Path("/tmp/hybrid_curve_fit_benchmark.csv")


def pressure_grid(pc_bar, transition_pressure_bar):
    pressures = [START_PRESSURE_BAR]
    pressure = START_PRESSURE_BAR
    while pressure * PRESSURE_STEP_UP < pc_bar:
        pressure *= PRESSURE_STEP_UP
        pressures.append(pressure)
    pressures.append(pc_bar)

    pressure = START_PRESSURE_BAR
    while pressure * PRESSURE_STEP_DOWN > MINIMUM_PRESSURE_BAR:
        pressure *= PRESSURE_STEP_DOWN
        pressures.append(pressure)
    pressures.append(MINIMUM_PRESSURE_BAR)
    pressures.extend((SWITCH_PRESSURE_BAR, transition_pressure_bar))
    return sorted(set(pressures))


def upper_temperature(curve, model, pressure_bar, transition_pressure_bar):
    target = math.log(pressure_bar)
    if pressure_bar >= curve.pc_bar:
        return curve.tc
    if pressure_bar >= transition_pressure_bar:
        return invert_ln_pressure(
            lambda temperature: aw_ln_p(curve, temperature),
            target,
            0.7 * curve.tc,
            curve.tc,
        )
    return invert_ln_pressure(
        lambda temperature: linear_omega_ln_pressure(
            curve,
            model["tb"],
            model["tb_omega"],
            temperature,
        ),
        target,
        model["switch_temperature"],
        0.7 * curve.tc,
    )


def low_pressure_temperatures(library, curve, model, pressures):
    if not pressures:
        return {}
    row = model["row"]
    minimum_temperature = float(row["T_min_K"])
    switch_temperature = model["switch_temperature"]

    def derivative(_ln_pressure, values):
        temperature = values[0]
        hvap = correlation_hvap_kJ_mol(library, row, curve, temperature)
        if hvap is None or hvap <= 0.0:
            return [0.0]
        return [R * temperature**2 / (hvap * 1000.0)]

    def range_event(_ln_pressure, values):
        return values[0] - minimum_temperature

    range_event.terminal = True
    range_event.direction = -1
    targets = sorted((math.log(pressure) for pressure in pressures), reverse=True)
    solution = solve_ivp(
        derivative,
        (math.log(SWITCH_PRESSURE_BAR), targets[-1]),
        [switch_temperature],
        t_eval=targets,
        events=range_event,
        rtol=1.0e-9,
        atol=1.0e-10,
        max_step=0.1,
    )
    return {
        round(math.exp(log_pressure), 14): float(temperature)
        for log_pressure, temperature in zip(solution.t, solution.y[0])
        if temperature >= minimum_temperature - 1.0e-8
    }


def generate_curve(library, curve, model):
    model["switch_temperature"] = invert_ln_pressure(
        lambda temperature: linear_omega_ln_pressure(
            curve,
            model["tb"],
            model["tb_omega"],
            temperature,
        ),
        math.log(SWITCH_PRESSURE_BAR),
        max(1.0, 0.15 * curve.tc),
        model["tb"],
    )
    if model["switch_temperature"] is None:
        return None
    transition_pressure = math.exp(aw_ln_p(curve, 0.7 * curve.tc))
    pressures = pressure_grid(curve.pc_bar, transition_pressure)
    low_pressures = [pressure for pressure in pressures if pressure < SWITCH_PRESSURE_BAR]
    low_temperatures = low_pressure_temperatures(
        library,
        curve,
        model,
        low_pressures,
    )

    points = []
    for pressure in pressures:
        if pressure < SWITCH_PRESSURE_BAR:
            temperature = low_temperatures.get(round(pressure, 14))
            if temperature is None:
                continue
            region = "clapeyron"
        elif pressure < transition_pressure:
            temperature = upper_temperature(
                curve,
                model,
                pressure,
                transition_pressure,
            )
            region = "linear_omega_aw"
        else:
            temperature = upper_temperature(
                curve,
                model,
                pressure,
                transition_pressure,
            )
            region = "plain_aw"
        if temperature is not None:
            points.append((temperature, pressure, region))
    return points, transition_pressure


def fit_coefficients(points):
    temperatures = np.array([point[0] for point in points], dtype=float)
    ln_pressures = np.log([point[1] for point in points])
    matrix = np.column_stack(
        (
            np.ones_like(temperatures),
            1.0 / temperatures,
            np.log(temperatures),
            temperatures,
            temperatures**2,
        )
    )
    scales = np.linalg.norm(matrix, axis=0)
    scaled_matrix = matrix / scales
    scaled_coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        scaled_matrix,
        ln_pressures,
        rcond=1.0e-13,
    )
    coefficients = scaled_coefficients / scales
    predicted = matrix @ coefficients
    return coefficients, predicted, np.linalg.cond(scaled_matrix)


def fitted_ln_pressure(coefficients, temperature):
    a, b, c, d, e = coefficients
    return a + b / temperature + c * math.log(temperature) + d * temperature + e * temperature**2


def fitted_temperature(coefficients, pressure_bar, lower, upper):
    target = math.log(pressure_bar)
    function = lambda temperature: fitted_ln_pressure(coefficients, temperature) - target
    if function(lower) > 0.0 or function(upper) < 0.0:
        return None
    return brentq(function, lower, upper, xtol=1.0e-10, rtol=1.0e-12)


def evaluate_fit(curve, points, transition_pressure):
    coefficients, predicted, condition = fit_coefficients(points)
    errors = []
    temperature_errors = []
    regional_errors = {region: [] for region in ("clapeyron", "linear_omega_aw", "plain_aw")}
    temperatures = [point[0] for point in points]
    lower = min(temperatures)
    upper = max(temperatures)
    for (temperature, pressure, region), predicted_ln in zip(points, predicted):
        error = math.exp(predicted_ln) / pressure - 1.0
        errors.append(abs(error))
        regional_errors[region].append(abs(error))
        fitted_temperature_value = fitted_temperature(
            coefficients,
            pressure,
            lower,
            upper,
        )
        if fitted_temperature_value is not None:
            temperature_errors.append(abs(fitted_temperature_value - temperature))

    derivative_temperatures = np.linspace(lower, upper, 1001)
    _, b, c, d, e = coefficients
    derivatives = (
        -b / derivative_temperatures**2
        + c / derivative_temperatures
        + d
        + 2.0 * e * derivative_temperatures
    )
    critical_error = math.exp(fitted_ln_pressure(coefficients, curve.tc)) / curve.pc_bar - 1.0
    return {
        "A": coefficients[0],
        "B": coefficients[1],
        "C": coefficients[2],
        "D": coefficients[3],
        "E": coefficients[4],
        "point_count": len(points),
        "minimum_pressure_bar": min(point[1] for point in points),
        "full_to_0p001bar": min(point[1] for point in points) <= MINIMUM_PRESSURE_BAR * (1.0 + 1.0e-12),
        "transition_pressure_bar": transition_pressure,
        "mard": mean(errors),
        "median_abs_relative_error": median(errors),
        "p95_abs_relative_error": percentile(errors, 0.95),
        "max_abs_relative_error": max(errors),
        "temperature_mae_K": mean(temperature_errors),
        "temperature_p95_K": percentile(temperature_errors, 0.95),
        "critical_relative_error": critical_error,
        "nonmonotone": float(np.min(derivatives)) <= 0.0,
        "scaled_condition_number": condition,
        "clapeyron_mard": (
            None if not regional_errors["clapeyron"] else mean(regional_errors["clapeyron"])
        ),
        "linear_omega_mard": (
            None
            if not regional_errors["linear_omega_aw"]
            else mean(regional_errors["linear_omega_aw"])
        ),
        "plain_aw_mard": (
            None if not regional_errors["plain_aw"] else mean(regional_errors["plain_aw"])
        ),
    }


def summarize(rows, label):
    print(f"\n{label}: n={len(rows)}")
    for key, display, scale in (
        ("mard", "MARD", 100.0),
        ("p95_abs_relative_error", "per-curve p95", 100.0),
        ("max_abs_relative_error", "per-curve max", 100.0),
        ("temperature_mae_K", "Tsat MAE", 1.0),
        ("temperature_p95_K", "Tsat p95", 1.0),
    ):
        values = [row[key] * scale for row in rows]
        unit = "%" if scale == 100.0 else " K"
        print(
            f"  {display}: mean={mean(values):.4f}{unit} "
            f"median={median(values):.4f}{unit} "
            f"p95={percentile(values, .95):.4f}{unit}"
        )
    print(
        f"  nonmonotone={sum(row['nonmonotone'] for row in rows)}; "
        f"critical |error| median="
        f"{100*median(abs(row['critical_relative_error']) for row in rows):.4f}%"
    )
    for key, display in (
        ("clapeyron_mard", "Clapeyron region"),
        ("linear_omega_mard", "linear-omega region"),
        ("plain_aw_mard", "plain-AW region"),
    ):
        values = [row[key] for row in rows if row[key] is not None]
        print(
            f"  {display} MARD: median={100*median(values):.4f}% "
            f"p95={100*percentile(values, .95):.4f}%"
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
        points, transition_pressure = generated
        if len(points) < 20:
            unavailable.append(curve.name)
            continue
        result = evaluate_fit(curve, points, transition_pressure)
        result.update({"cas": curve.cas, "name": curve.name, "Tb_K": model["tb"], "Tc_K": curve.tc, "Pc_bar": curve.pc_bar, "omega": curve.omega})
        rows.append(result)

    print("FIVE-TERM ln(Psat) FIT TO FULL HYBRID CURVES")
    print(f"eligible fits={len(rows)}, unavailable={len(unavailable)}")
    summarize(rows, "all fitted curves")
    full = [row for row in rows if row["full_to_0p001bar"]]
    summarize(full, "curves reaching 0.001 bar within Hvap range")

    print("\nWORST 20 BY MARD")
    for row in sorted(rows, key=lambda item: item["mard"], reverse=True)[:20]:
        print(
            f"  {row['name']} [{row['cas']}]: Pmin={row['minimum_pressure_bar']:.6g} bar, "
            f"MARD={100*row['mard']:.3f}%, p95={100*row['p95_abs_relative_error']:.3f}%, "
            f"max={100*row['max_abs_relative_error']:.3f}%, "
            f"Tc error={100*row['critical_relative_error']:+.3f}%"
        )

    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
