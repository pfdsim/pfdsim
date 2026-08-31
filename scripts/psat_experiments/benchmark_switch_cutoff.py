import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

from scipy.stats import pearsonr, spearmanr

from benchmark_lower_clapeyron import (
    correlation_hvap_kJ_mol,
    fit_local_watson,
    hvap_row_at_tb,
    integrated_ln_pressure,
    watson_hvap_kJ_mol,
)
from benchmark_lower_hybrid import is_banned
from benchmark_lower_psat import PRESSURE_LEVELS_BAR, invert_ln_pressure
from benchmark_perry_aw import load_curves, percentile, perry_ln_p
from benchmark_table210_nannoolal_aw import tb_valid_matches
from benchmark_tb_constrained_aw import aw_ln_p_with_omega, tb_constrained_omega
from perry_properties import PerryPropertyLibrary


MMHG_PER_BAR = 750.061683
SWITCH_CANDIDATES_MMHG = (760.0, 400.0, 200.0, 100.0, 60.0, 40.0, 20.0, 10.0, 5.0)
FIXED_SWITCHES_MMHG = (200.0, 100.0)
WATSON_BOUNDS = (0.20, 0.60)
OUTPUT_PATH = Path("/tmp/switch_cutoff_benchmark.csv")


def upper_eased_linear_weight(coordinate, endpoint_width=0.05):
    if coordinate <= 1.0 - endpoint_width:
        return coordinate
    if coordinate >= 1.0:
        return 1.0
    local = (1.0 - coordinate) / endpoint_width
    polynomial = 6.0 * local**3 - 8.0 * local**4 + 3.0 * local**5
    return 1.0 - endpoint_width * polynomial


def linear_omega(curve, tb, tb_omega, temperature):
    coordinate = (temperature - tb) / (0.7 * curve.tc - tb)
    weight = upper_eased_linear_weight(coordinate)
    return tb_omega + (curve.omega - tb_omega) * weight


def linear_omega_ln_pressure(curve, tb, tb_omega, temperature):
    return aw_ln_p_with_omega(
        curve,
        temperature,
        linear_omega(curve, tb, tb_omega, temperature),
    )


def switch_temperature(curve, tb, tb_omega, pressure_bar):
    return invert_ln_pressure(
        lambda temperature: linear_omega_ln_pressure(
            curve,
            tb,
            tb_omega,
            temperature,
        ),
        math.log(pressure_bar),
        max(1.0, 0.15 * curve.tc),
        min(tb, 0.7 * curve.tc),
    )


def clipped(value, bounds):
    return min(max(value, bounds[0]), bounds[1])


def build_model(library, curve, tb):
    if tb >= 0.7 * curve.tc:
        return None
    tb_omega, _roots = tb_constrained_omega(curve, tb, 1.01325)
    if tb_omega is None:
        return None
    row = hvap_row_at_tb(library, curve, tb)
    if row is None:
        return None
    fit = fit_local_watson(library, row, curve, tb)
    if fit is None:
        return None
    tb_hvap, fitted_exponent, _fit_min, _fit_max = fit
    bounded_exponent = clipped(fitted_exponent, WATSON_BOUNDS)
    return {
        "tb": tb,
        "tb_omega": tb_omega,
        "row": row,
        "fitted_exponent": fitted_exponent,
        "bounded_exponent": bounded_exponent,
        "correlation_function": lambda temperature: correlation_hvap_kJ_mol(
            library,
            row,
            curve,
            temperature,
        ),
        "watson_function": lambda temperature: watson_hvap_kJ_mol(
            curve,
            tb,
            tb_hvap,
            bounded_exponent,
            temperature,
        ),
    }


def hybrid_ln_pressure(
    curve,
    model,
    temperature,
    switch_pressure_bar,
    switch_temperature_K,
    hvap_function,
):
    if temperature >= switch_temperature_K:
        return linear_omega_ln_pressure(
            curve,
            model["tb"],
            model["tb_omega"],
            temperature,
        )
    return integrated_ln_pressure(
        switch_temperature_K,
        switch_pressure_bar,
        temperature,
        hvap_function,
    )


def reference_error(
    curve,
    model,
    temperature,
    pressure_bar,
    switch_mmHg,
    method,
):
    switch_pressure_bar = switch_mmHg / MMHG_PER_BAR
    switch_temperature_K = switch_temperature(
        curve,
        model["tb"],
        model["tb_omega"],
        switch_pressure_bar,
    )
    if switch_temperature_K is None:
        return None
    if method == "correlation":
        if temperature < switch_temperature_K and (
            temperature < float(model["row"]["T_min_K"])
            or switch_temperature_K < float(model["row"]["T_min_K"])
        ):
            return None
        function = model["correlation_function"]
    else:
        function = model["watson_function"]
    prediction = hybrid_ln_pressure(
        curve,
        model,
        temperature,
        switch_pressure_bar,
        switch_temperature_K,
        function,
    )
    return abs(math.exp(prediction) / pressure_bar - 1.0)


def correlation_reference_points(curve, tb):
    points = []
    minimum_pressure = math.exp(perry_ln_p(curve, curve.t_min))
    for pressure_bar in PRESSURE_LEVELS_BAR:
        if pressure_bar < minimum_pressure:
            continue
        temperature = invert_ln_pressure(
            lambda value: perry_ln_p(curve, value),
            math.log(pressure_bar),
            curve.t_min,
            tb,
        )
        if temperature is not None:
            points.append((temperature, pressure_bar))
    return points


def table_records(library):
    records = []
    for curve, pairs, trusted_tb in tb_valid_matches():
        if is_banned(curve):
            continue
        model = build_model(library, curve, trusted_tb)
        if model is None:
            continue
        points = [
            (temperature, pressure_bar)
            for temperature, pressure_bar in pairs
            if pressure_bar <= PRESSURE_LEVELS_BAR[0] + 1.0e-6
        ]
        records.append((curve, model, points, "Perry 2-10 table"))
    return records


def correlation_records(library):
    records = []
    for curve in load_curves():
        if is_banned(curve):
            continue
        tb_result = library.normal_boiling_point_K(curve.cas)
        if tb_result is None:
            continue
        tb = float(tb_result.value)
        model = build_model(library, curve, tb)
        if model is None:
            continue
        points = correlation_reference_points(curve, tb)
        records.append((curve, model, points, "Perry 2-8 correlation"))
    return records


def fixed_switch_errors(records, switch_mmHg, method):
    errors = []
    by_pressure = {}
    for curve, model, points, _reference in records:
        for temperature, pressure_bar in points:
            error = reference_error(
                curve,
                model,
                temperature,
                pressure_bar,
                switch_mmHg,
                method,
            )
            if error is None:
                continue
            errors.append(error)
            by_pressure.setdefault(pressure_bar, []).append(error)
    return errors, by_pressure


def report_fixed(records, label):
    print(f"\n{label}: curves={len(records)}")
    for switch_mmHg in FIXED_SWITCHES_MMHG:
        for method in ("correlation", "watson"):
            errors, by_pressure = fixed_switch_errors(records, switch_mmHg, method)
            print(
                f"  switch={switch_mmHg:.0f} mmHg {method}: n={len(errors)} "
                f"MARD={100*mean(errors):.2f}% median={100*median(errors):.2f}% "
                f"p95={100*percentile(errors, .95):.2f}%"
            )
            print(
                "    "
                + "; ".join(
                    f"{pressure_bar * MMHG_PER_BAR:.0f}mm={100*median(values):.2f}%"
                    for pressure_bar, values in sorted(by_pressure.items(), reverse=True)
                )
            )


def optimize_records(records, method):
    rows = []
    for curve, model, points, reference in records:
        scores = {}
        for switch_mmHg in SWITCH_CANDIDATES_MMHG:
            errors = [
                reference_error(
                    curve,
                    model,
                    temperature,
                    pressure_bar,
                    switch_mmHg,
                    method,
                )
                for temperature, pressure_bar in points
            ]
            errors = [error for error in errors if error is not None]
            if len(errors) >= 4:
                scores[switch_mmHg] = mean(errors)
        if not scores:
            continue
        best = min(scores, key=scores.get)
        ordered = sorted(scores.values())
        critical = curve.pc_bar
        rows.append(
            {
                "reference": reference,
                "method": method,
                "cas": curve.cas,
                "name": curve.name,
                "Tb_K": model["tb"],
                "Tc_K": curve.tc,
                "Pc_bar": critical,
                "omega": curve.omega,
                "Tb_over_Tc": model["tb"] / curve.tc,
                "fitted_watson_exponent": model["fitted_exponent"],
                "bounded_watson_exponent": model["bounded_exponent"],
                "best_switch_mmHg": best,
                "best_mard": scores[best],
                "second_best_gap": (
                    0.0 if len(ordered) < 2 else ordered[1] - ordered[0]
                ),
            }
        )
    return rows


def report_optimization(rows, label):
    print(f"\n{label}: optimized compounds={len(rows)}")
    counts = Counter(row["best_switch_mmHg"] for row in rows)
    print(
        "  cutoff distribution: "
        + ", ".join(
            f"{cutoff:g}={counts[cutoff]}"
            for cutoff in SWITCH_CANDIDATES_MMHG
            if counts[cutoff]
        )
    )
    log_cutoff = [math.log10(row["best_switch_mmHg"]) for row in rows]
    for key in ("Tb_K", "Tc_K", "Pc_bar", "omega", "Tb_over_Tc"):
        values = [row[key] for row in rows]
        pearson = pearsonr(values, log_cutoff).statistic
        spearman = spearmanr(values, log_cutoff).statistic
        print(f"  {key}: Pearson={pearson:+.3f}, Spearman={spearman:+.3f}")
    gaps = [row["second_best_gap"] for row in rows]
    print(
        f"  second-best MARD gap: median={100*median(gaps):.3f}% "
        f"p90={100*percentile(gaps, .90):.3f}%"
    )


def main():
    library = PerryPropertyLibrary()
    library._load()
    table = table_records(library)
    correlations = correlation_records(library)
    print("LINEAR-OMEGA AW TO VARIABLE PRESSURE, THEN CLAPEYRON")
    report_fixed(table, "Perry 2-10 independent table points")
    report_fixed(correlations, "Perry 2-8 continuous correlations")

    rows = []
    for records, reference_label in (
        (table, "Perry 2-10 table"),
        (correlations, "Perry 2-8 correlation"),
    ):
        for method in ("correlation", "watson"):
            optimized = optimize_records(records, method)
            rows.extend(optimized)
            report_optimization(optimized, f"{reference_label}, {method}")

    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
