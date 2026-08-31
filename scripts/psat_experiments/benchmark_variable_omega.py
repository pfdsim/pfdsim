import csv
import math
from pathlib import Path
from statistics import mean, median

from scipy.optimize import brentq

from benchmark_lower_clapeyron import (
    correlation_hvap_kJ_mol,
    fit_local_watson,
    hvap_row_at_tb,
    integrated_ln_pressure,
    watson_hvap_kJ_mol,
)
from benchmark_lower_hybrid import SWITCH_PRESSURE_BAR, is_banned
from benchmark_lower_psat import PRESSURE_LEVELS_BAR, invert_ln_pressure
from benchmark_perry_aw import aw_ln_p, load_curves, percentile, perry_ln_p
from benchmark_table210_nannoolal_aw import tb_valid_matches
from benchmark_tb_constrained_aw import aw_ln_p_with_omega, tb_constrained_omega
from benchmark_tb_nannoolal_aw import direct_tb_candidates, preferred_candidates
from perry_properties import PerryPropertyLibrary
from property_resolution.common import R


UPPER_OUTPUT_PATH = Path("/tmp/variable_omega_upper_benchmark.csv")
LOWER_OUTPUT_PATH = Path("/tmp/variable_omega_lower_benchmark.csv")
TRANSITION_METHODS = (
    "plain_aw",
    "constant_tb_omega",
    "linear_full",
    "smooth_linear_05",
    "smooth_linear_10",
    "smooth_linear_20",
    "smooth_full",
    "smooth_half",
    "smooth_quarter",
)
WATSON_BOUNDS = {
    "watson_unbounded": None,
    "watson_020_060": (0.20, 0.60),
    "watson_025_050": (0.25, 0.50),
    "watson_030_045": (0.30, 0.45),
}


def smootherstep(value):
    return value**3 * (value * (6.0 * value - 15.0) + 10.0)


def endpoint_smoothed_linear(value, endpoint_width):
    if value < endpoint_width:
        coordinate = value / endpoint_width
        return endpoint_width * (
            6.0 * coordinate**3 - 8.0 * coordinate**4 + 3.0 * coordinate**5
        )
    if value > 1.0 - endpoint_width:
        coordinate = (1.0 - value) / endpoint_width
        return 1.0 - endpoint_width * (
            6.0 * coordinate**3 - 8.0 * coordinate**4 + 3.0 * coordinate**5
        )
    return value


def effective_omega(curve, tb, tb_omega, temperature, method):
    if method == "plain_aw":
        return curve.omega
    if method == "constant_tb_omega" or temperature <= tb:
        return tb_omega
    transition_end = 0.7 * curve.tc
    if temperature >= transition_end:
        return curve.omega
    coordinate = (temperature - tb) / (transition_end - tb)
    if method == "linear_full":
        weight = coordinate
    elif method.startswith("smooth_linear_"):
        endpoint_width = float(method.rsplit("_", 1)[1]) / 100.0
        weight = endpoint_smoothed_linear(coordinate, endpoint_width)
    else:
        width = {
            "smooth_full": 1.0,
            "smooth_half": 0.5,
            "smooth_quarter": 0.25,
        }[method]
        weight = smootherstep(min(coordinate / width, 1.0))
    return tb_omega + (curve.omega - tb_omega) * weight


def variable_omega_ln_pressure(curve, tb, tb_omega, temperature, method):
    omega = effective_omega(curve, tb, tb_omega, temperature, method)
    return aw_ln_p_with_omega(curve, temperature, omega)


def evaluate_upper_curve(curve, tb, tb_source):
    tb_omega, roots = tb_constrained_omega(curve, tb, 1.01325)
    if tb_omega is None:
        return []
    temperatures = [
        tb + (0.7 * curve.tc - tb) * index / 400.0
        for index in range(401)
    ]
    reference = [perry_ln_p(curve, temperature) for temperature in temperatures]
    rows = []
    for method in TRANSITION_METHODS:
        predicted = [
            variable_omega_ln_pressure(
                curve,
                tb,
                tb_omega,
                temperature,
                method,
            )
            for temperature in temperatures
        ]
        errors = [
            abs(math.exp(value - truth) - 1.0)
            for value, truth in zip(predicted, reference)
        ]
        rows.append(
            {
                "cas": curve.cas,
                "name": curve.name,
                "tb_source": tb_source,
                "method": method,
                "tb_K": tb,
                "tb_Tr": tb / curve.tc,
                "source_omega": curve.omega,
                "tb_omega": tb_omega,
                "omega_change": tb_omega - curve.omega,
                "other_root": max(
                    roots,
                    key=lambda value: abs(value - tb_omega),
                ),
                "mard": mean(errors),
                "max_error": max(errors),
                "nonmonotone": any(
                    right <= left for left, right in zip(predicted, predicted[1:])
                ),
            }
        )
    return rows


def upper_rows(curves, library):
    rows = []
    excluded = []
    ineligible = []
    for curve in curves:
        if is_banned(curve):
            excluded.append(curve.name)
            continue
        tb_result = library.normal_boiling_point_K(curve.cas)
        if tb_result is None:
            continue
        tb = float(tb_result.value)
        if tb >= 0.7 * curve.tc:
            ineligible.append(curve.name)
            continue
        rows.extend(evaluate_upper_curve(curve, tb, "Perry curve crossing"))
    return rows, excluded, ineligible


def independent_upper_rows(curves):
    rows = []
    excluded = []
    ineligible = []
    candidates = preferred_candidates(direct_tb_candidates(curves))
    for curve, tb, source in candidates:
        if is_banned(curve):
            excluded.append(curve.name)
            continue
        if not curve.t_min <= tb < 0.7 * curve.tc:
            ineligible.append(curve.name)
            continue
        rows.extend(evaluate_upper_curve(curve, tb, source))
    return rows, excluded, ineligible


def parameterized_aw_temperature(curve, omega, pressure_bar):
    return invert_ln_pressure(
        lambda temperature: aw_ln_p_with_omega(curve, temperature, omega),
        math.log(pressure_bar),
        max(1.0, 0.15 * curve.tc),
        curve.tc,
    )


def parameterized_aw_slope(curve, omega, temperature):
    step = max(1.0e-5 * curve.tc, 1.0e-4)
    return (
        aw_ln_p_with_omega(curve, temperature + step, omega)
        - aw_ln_p_with_omega(curve, temperature - step, omega)
    ) / (2.0 * step)


def clipped(value, bounds):
    if bounds is None:
        return value
    return min(max(value, bounds[0]), bounds[1])


def lower_models(library, curve, tb):
    tb_omega, _roots = tb_constrained_omega(curve, tb, 1.01325)
    if tb_omega is None:
        return None
    row = hvap_row_at_tb(library, curve, tb)
    if row is None:
        return None
    watson_fit = fit_local_watson(library, row, curve, tb)
    if watson_fit is None:
        return None
    switch_temperature = parameterized_aw_temperature(
        curve,
        tb_omega,
        SWITCH_PRESSURE_BAR,
    )
    if switch_temperature is None:
        return None
    tb_hvap, fitted_exponent, _fit_min, _fit_max = watson_fit
    correlation_function = lambda temperature: correlation_hvap_kJ_mol(
        library,
        row,
        curve,
        temperature,
    )
    watson_functions = {}
    for method, bounds in WATSON_BOUNDS.items():
        exponent = clipped(fitted_exponent, bounds)
        watson_functions[method] = (
            exponent,
            lambda temperature, exponent=exponent: watson_hvap_kJ_mol(
                curve,
                tb,
                tb_hvap,
                exponent,
                temperature,
            ),
        )
    aw_slope = parameterized_aw_slope(curve, tb_omega, switch_temperature)
    slope_jumps = {
        "correlation_hybrid": (
            correlation_function(switch_temperature)
            * 1000.0
            / (R * switch_temperature**2)
            / aw_slope
            - 1.0
        )
    }
    for method, (_exponent, function) in watson_functions.items():
        slope_jumps[method] = (
            function(switch_temperature)
            * 1000.0
            / (R * switch_temperature**2)
            / aw_slope
            - 1.0
        )
    return {
        "tb": tb,
        "tb_omega": tb_omega,
        "row": row,
        "switch_temperature": switch_temperature,
        "correlation_function": correlation_function,
        "watson_functions": watson_functions,
        "fitted_exponent": fitted_exponent,
        "slope_jumps": slope_jumps,
    }


def hybrid_ln_pressure(curve, models, temperature, hvap_function):
    switch_temperature = models["switch_temperature"]
    if temperature >= switch_temperature:
        return aw_ln_p_with_omega(curve, temperature, models["tb_omega"])
    return integrated_ln_pressure(
        switch_temperature,
        SWITCH_PRESSURE_BAR,
        temperature,
        hvap_function,
    )


def lower_row(curve, models, temperature, pressure_bar, reference):
    predictions = {
        "plain_aw": aw_ln_p(curve, temperature),
        "tb_omega_aw": aw_ln_p_with_omega(
            curve,
            temperature,
            models["tb_omega"],
        ),
    }
    extrapolated_omega = models["tb_omega"] + (
        curve.omega - models["tb_omega"]
    ) * (
        (temperature - models["tb"])
        / (0.7 * curve.tc - models["tb"])
    )
    predictions["linear_extrapolated_aw"] = aw_ln_p_with_omega(
        curve,
        temperature,
        extrapolated_omega,
    )
    correlation_minimum = float(models["row"]["T_min_K"])
    predictions["correlation_hybrid"] = (
        None
        if temperature < correlation_minimum
        else hybrid_ln_pressure(
            curve,
            models,
            temperature,
            models["correlation_function"],
        )
    )
    for method, (_exponent, function) in models["watson_functions"].items():
        predictions[method] = hybrid_ln_pressure(
            curve,
            models,
            temperature,
            function,
        )
    result = {
        "reference": reference,
        "cas": curve.cas,
        "name": curve.name,
        "pressure_bar": pressure_bar,
        "pressure_mmHg": pressure_bar * 750.061683,
        "temperature_K": temperature,
        "reduced_temperature": temperature / curve.tc,
        "tb_omega": models["tb_omega"],
        "switch_temperature_K": models["switch_temperature"],
        "fitted_watson_exponent": models["fitted_exponent"],
    }
    for method, prediction in predictions.items():
        result[f"{method}_relative_error"] = (
            None
            if prediction is None
            else math.exp(prediction) / pressure_bar - 1.0
        )
    for method, jump in models["slope_jumps"].items():
        result[f"{method}_slope_jump"] = jump
    for method, (exponent, _function) in models["watson_functions"].items():
        result[f"{method}_exponent"] = exponent
    return result


def lower_table_rows(library):
    rows = []
    excluded = []
    ineligible = []
    unavailable = []
    for curve, pairs, trusted_tb in tb_valid_matches():
        if is_banned(curve):
            excluded.append(curve.name)
            continue
        if trusted_tb >= 0.7 * curve.tc:
            ineligible.append(curve.name)
            continue
        models = lower_models(library, curve, trusted_tb)
        if models is None:
            unavailable.append(curve.name)
            continue
        for temperature, pressure_bar in pairs:
            if pressure_bar > SWITCH_PRESSURE_BAR + 1.0e-6:
                continue
            rows.append(
                lower_row(
                    curve,
                    models,
                    temperature,
                    pressure_bar,
                    "Perry 2-10 table",
                )
            )
    return rows, excluded, ineligible, unavailable


def lower_correlation_rows(library):
    rows = []
    excluded = []
    ineligible = []
    unavailable = []
    for curve in load_curves():
        if is_banned(curve):
            excluded.append(curve.name)
            continue
        tb_result = library.normal_boiling_point_K(curve.cas)
        if tb_result is None:
            continue
        tb = float(tb_result.value)
        if tb >= 0.7 * curve.tc:
            ineligible.append(curve.name)
            continue
        models = lower_models(library, curve, tb)
        if models is None:
            unavailable.append(curve.name)
            continue
        minimum_pressure = math.exp(perry_ln_p(curve, curve.t_min))
        for pressure_bar in PRESSURE_LEVELS_BAR:
            if pressure_bar > SWITCH_PRESSURE_BAR or pressure_bar < minimum_pressure:
                continue
            temperature = invert_ln_pressure(
                lambda value: perry_ln_p(curve, value),
                math.log(pressure_bar),
                curve.t_min,
                tb,
            )
            if temperature is None:
                continue
            rows.append(
                lower_row(
                    curve,
                    models,
                    temperature,
                    pressure_bar,
                    "Perry 2-8 correlation",
                )
            )
    return rows, excluded, ineligible, unavailable


def report_upper(rows, label):
    print(f"\nTb TO Tr=0.7: {label}")
    for method in TRANSITION_METHODS:
        subset = [row for row in rows if row["method"] == method]
        mards = [row["mard"] for row in subset]
        maxima = [row["max_error"] for row in subset]
        print(
            f"  {method}: n={len(subset)} mean={100*mean(mards):.4f}% "
            f"median={100*median(mards):.4f}% "
            f"p95={100*percentile(mards, .95):.4f}% "
            f"max-error median={100*median(maxima):.4f}% "
            f"nonmonotone={sum(row['nonmonotone'] for row in subset)}"
        )


def report_lower(rows, label):
    methods = (
        "plain_aw",
        "tb_omega_aw",
        "linear_extrapolated_aw",
        "correlation_hybrid",
        *WATSON_BOUNDS,
    )
    print(f"\n{label}: curves={len({row['cas'] for row in rows})} points={len(rows)}")
    for method in methods:
        values = [
            abs(row[f"{method}_relative_error"])
            for row in rows
            if row[f"{method}_relative_error"] is not None
        ]
        print(
            f"  {method}: n={len(values)} MARD={100*mean(values):.2f}% "
            f"median={100*median(values):.2f}% "
            f"p95={100*percentile(values, .95):.2f}%"
        )
    print("  by pressure:")
    for pressure_bar in sorted({row["pressure_bar"] for row in rows}, reverse=True):
        subset = [row for row in rows if row["pressure_bar"] == pressure_bar]
        pieces = []
        for method in methods:
            values = [
                abs(row[f"{method}_relative_error"])
                for row in subset
                if row[f"{method}_relative_error"] is not None
            ]
            pieces.append(
                f"{method}={100*median(values):.2f}%"
            )
        print(
            f"    {pressure_bar * 750.061683:4.0f} mmHg: "
            + "; ".join(pieces)
        )
    unique = {row["cas"]: row for row in rows}.values()
    for method in ("correlation_hybrid", *WATSON_BOUNDS):
        jumps = [abs(row[f"{method}_slope_jump"]) for row in unique]
        print(
            f"  {method} handoff |slope mismatch|: "
            f"median={100*median(jumps):.2f}% "
            f"p95={100*percentile(jumps, .95):.2f}%"
        )


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    library = PerryPropertyLibrary()
    library._load()
    curves = load_curves()
    upper, upper_excluded, upper_ineligible = upper_rows(curves, library)
    independent_upper, independent_upper_excluded, independent_upper_ineligible = (
        independent_upper_rows(curves)
    )
    table, table_excluded, table_ineligible, table_unavailable = lower_table_rows(library)
    correlations, correlation_excluded, correlation_ineligible, correlation_unavailable = (
        lower_correlation_rows(library)
    )

    print("VARIABLE OMEGA FROM Tb TO Tr=0.7; CLAPEYRON BELOW 200 mmHg")
    report_upper(upper, "Perry-exact Tb diagnostic")
    report_upper(independent_upper, "independent direct Tb")
    report_lower(table, "Perry 2-10 independent table points")
    report_lower(correlations, "Perry 2-8 continuous correlations")
    print(
        f"\nUpper excluded={len(set(upper_excluded))} "
        f"Tb>=0.7Tc={len(set(upper_ineligible))}; "
        f"independent upper excluded={len(set(independent_upper_excluded))} "
        f"ineligible={len(set(independent_upper_ineligible))}; "
        f"table excluded={len(set(table_excluded))} "
        f"Tb>=0.7Tc={len(set(table_ineligible))} "
        f"unavailable={len(set(table_unavailable))}; "
        f"correlation excluded={len(set(correlation_excluded))} "
        f"Tb>=0.7Tc={len(set(correlation_ineligible))} "
        f"unavailable={len(set(correlation_unavailable))}"
    )

    write_rows(UPPER_OUTPUT_PATH, upper + independent_upper)
    write_rows(LOWER_OUTPUT_PATH, table + correlations)
    print(f"Wrote {UPPER_OUTPUT_PATH} and {LOWER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
