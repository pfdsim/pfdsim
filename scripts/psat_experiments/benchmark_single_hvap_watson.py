import csv
import math
from pathlib import Path
from statistics import mean, median

from benchmark_lower_clapeyron import (
    correlation_hvap_kJ_mol,
    hvap_row_at_tb,
    integrated_ln_pressure,
    watson_hvap_kJ_mol,
)
from benchmark_lower_hybrid import is_banned
from benchmark_lower_psat import PRESSURE_LEVELS_BAR, invert_ln_pressure
from benchmark_perry_aw import aw_ln_p, load_curves, percentile, perry_ln_p
from benchmark_switch_cutoff import linear_omega_ln_pressure
from benchmark_table210_nannoolal_aw import tb_valid_matches
from benchmark_tb_constrained_aw import tb_constrained_omega
from perry_properties import PerryPropertyLibrary


SWITCH_PRESSURE_BAR = 0.25
WATSON_EXPONENT = 0.38
OUTPUT_PATH = Path("/tmp/single_hvap_watson_benchmark.csv")
METHODS = (
    "plain_aw",
    "linear_omega_aw",
    "plain_aw_fixed038_hybrid",
    "linear_omega_fixed038_hybrid",
    "linear_omega_direct_hvap_hybrid",
)


def switch_temperature(function, curve, tb):
    return invert_ln_pressure(
        function,
        math.log(SWITCH_PRESSURE_BAR),
        max(1.0, 0.15 * curve.tc),
        tb,
    )


def build_model(library, curve, tb):
    if tb >= 0.7 * curve.tc:
        return None
    tb_omega, _roots = tb_constrained_omega(curve, tb, 1.01325)
    if tb_omega is None:
        return None
    row = hvap_row_at_tb(library, curve, tb)
    if row is None:
        return None
    tb_hvap = correlation_hvap_kJ_mol(library, row, curve, tb)
    if tb_hvap is None or tb_hvap <= 0.0:
        return None

    plain_switch = switch_temperature(
        lambda temperature: aw_ln_p(curve, temperature),
        curve,
        tb,
    )
    linear_switch = switch_temperature(
        lambda temperature: linear_omega_ln_pressure(
            curve,
            tb,
            tb_omega,
            temperature,
        ),
        curve,
        tb,
    )
    if plain_switch is None or linear_switch is None:
        return None

    return {
        "tb": tb,
        "tb_omega": tb_omega,
        "row": row,
        "tb_hvap": tb_hvap,
        "plain_switch": plain_switch,
        "linear_switch": linear_switch,
        "fixed_watson": lambda temperature: watson_hvap_kJ_mol(
            curve,
            tb,
            tb_hvap,
            WATSON_EXPONENT,
            temperature,
        ),
        "direct_hvap": lambda temperature: correlation_hvap_kJ_mol(
            library,
            row,
            curve,
            temperature,
        ),
    }


def hybrid_ln_pressure(switch_temperature_K, temperature, hvap_function):
    return integrated_ln_pressure(
        switch_temperature_K,
        SWITCH_PRESSURE_BAR,
        temperature,
        hvap_function,
    )


def evaluate_point(curve, model, temperature, pressure_bar, reference):
    plain_prediction = aw_ln_p(curve, temperature)
    linear_prediction = linear_omega_ln_pressure(
        curve,
        model["tb"],
        model["tb_omega"],
        temperature,
    )
    plain_hybrid = hybrid_ln_pressure(
        model["plain_switch"],
        temperature,
        model["fixed_watson"],
    )
    linear_hybrid = hybrid_ln_pressure(
        model["linear_switch"],
        temperature,
        model["fixed_watson"],
    )

    direct_hybrid = None
    row_minimum = float(model["row"]["T_min_K"])
    if temperature >= row_minimum and model["linear_switch"] >= row_minimum:
        direct_hybrid = hybrid_ln_pressure(
            model["linear_switch"],
            temperature,
            model["direct_hvap"],
        )

    predictions = {
        "plain_aw": plain_prediction,
        "linear_omega_aw": linear_prediction,
        "plain_aw_fixed038_hybrid": plain_hybrid,
        "linear_omega_fixed038_hybrid": linear_hybrid,
        "linear_omega_direct_hvap_hybrid": direct_hybrid,
    }
    row = {
        "reference": reference,
        "cas": curve.cas,
        "name": curve.name,
        "pressure_bar": pressure_bar,
        "pressure_mmHg": pressure_bar * 750.061683,
        "temperature_K": temperature,
        "reduced_temperature": temperature / curve.tc,
        "Tb_K": model["tb"],
        "Hvap_Tb_kJ_mol": model["tb_hvap"],
        "plain_switch_temperature_K": model["plain_switch"],
        "linear_switch_temperature_K": model["linear_switch"],
    }
    for method, prediction in predictions.items():
        row[f"{method}_relative_error"] = (
            None
            if prediction is None
            else math.exp(prediction) / pressure_bar - 1.0
        )
    return row


def table_rows(library):
    rows = []
    unavailable = []
    for curve, pairs, trusted_tb in tb_valid_matches():
        if is_banned(curve):
            continue
        model = build_model(library, curve, trusted_tb)
        if model is None:
            unavailable.append(curve.name)
            continue
        for temperature, pressure_bar in pairs:
            if pressure_bar >= SWITCH_PRESSURE_BAR:
                continue
            rows.append(
                evaluate_point(
                    curve,
                    model,
                    temperature,
                    pressure_bar,
                    "Perry 2-10 table",
                )
            )
    return rows, unavailable


def correlation_rows(library):
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
        minimum_pressure = math.exp(perry_ln_p(curve, curve.t_min))
        for pressure_bar in PRESSURE_LEVELS_BAR:
            if pressure_bar >= SWITCH_PRESSURE_BAR or pressure_bar < minimum_pressure:
                continue
            temperature = invert_ln_pressure(
                lambda value: perry_ln_p(curve, value),
                math.log(pressure_bar),
                curve.t_min,
                model["tb"],
            )
            if temperature is None:
                continue
            rows.append(
                evaluate_point(
                    curve,
                    model,
                    temperature,
                    pressure_bar,
                    "Perry 2-8 correlation",
                )
            )
    return rows, unavailable


def errors(rows, method):
    return [
        abs(row[f"{method}_relative_error"])
        for row in rows
        if row[f"{method}_relative_error"] is not None
    ]


def per_curve_mards(rows, method):
    grouped = {}
    for row in rows:
        value = row[f"{method}_relative_error"]
        if value is not None:
            grouped.setdefault(row["cas"], []).append(abs(value))
    return {cas: mean(values) for cas, values in grouped.items()}


def report(rows, label, unavailable):
    print(
        f"\n{label}: curves={len({row['cas'] for row in rows})} "
        f"points={len(rows)} unavailable={len(set(unavailable))}"
    )
    for method in METHODS:
        values = errors(rows, method)
        curve_values = per_curve_mards(rows, method)
        print(
            f"  {method}: n={len(values)} MARD={100*mean(values):.2f}% "
            f"median={100*median(values):.2f}% "
            f"p95={100*percentile(values, .95):.2f}% "
            f"curve-median={100*median(curve_values.values()):.2f}%"
        )

    print("  by pressure:")
    for pressure_bar in sorted({row["pressure_bar"] for row in rows}, reverse=True):
        subset = [row for row in rows if row["pressure_bar"] == pressure_bar]
        pieces = []
        for method in METHODS:
            values = errors(subset, method)
            pieces.append(
                f"{method}={100*median(values):.2f}%"
                if values else
                f"{method}=n/a"
            )
        print(f"    {pressure_bar * 750.061683:4.0f} mmHg: " + "; ".join(pieces))

    linear = per_curve_mards(rows, "linear_omega_aw")
    fixed = per_curve_mards(rows, "linear_omega_fixed038_hybrid")
    paired = sorted(set(linear) & set(fixed))
    print(
        "  fixed-0.38 hybrid vs linear-omega AW curve wins: "
        f"hybrid={sum(fixed[cas] < linear[cas] for cas in paired)} "
        f"AW={sum(linear[cas] < fixed[cas] for cas in paired)}"
    )


def main():
    library = PerryPropertyLibrary()
    library._load()
    table, table_unavailable = table_rows(library)
    correlations, correlation_unavailable = correlation_rows(library)
    print("SINGLE Hvap(Tb), FIXED WATSON n=0.38, CLAPEYRON BELOW 0.25 BAR")
    report(table, "Perry 2-10 independent table points", table_unavailable)
    report(correlations, "Perry 2-8 continuous correlations", correlation_unavailable)

    rows = table + correlations
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
