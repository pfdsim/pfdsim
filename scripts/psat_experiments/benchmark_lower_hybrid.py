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
from benchmark_lower_psat import PRESSURE_LEVELS_BAR, invert_ln_pressure
from benchmark_perry_aw import (
    aw_dlnp_dt,
    aw_ln_p,
    load_curves,
    percentile,
    perry_ln_p,
)
from benchmark_table210_nannoolal_aw import tb_valid_matches
from perry_properties import PerryPropertyLibrary
from property_resolution.common import R


SWITCH_PRESSURE_BAR = 0.2666447
OUTPUT_PATH = Path("/tmp/lower_hybrid_benchmark.csv")
SMALL_POLAR_INORGANICS = {
    "ammonia",
    "hydrazine",
    "hydrogen bromide",
    "hydrogen chloride",
    "hydrogen cyanide",
    "hydrogen fluoride",
    "hydrogen iodide",
    "hydrogen sulfide",
    "sulfur dioxide",
    "sulfur trioxide",
    "water",
}


def is_banned(curve):
    name = curve.name.lower()
    return "acid" in name or name in SMALL_POLAR_INORGANICS


def aw_temperature(curve, pressure_bar):
    return invert_ln_pressure(
        lambda temperature: aw_ln_p(curve, temperature),
        math.log(pressure_bar),
        max(1.0, 0.15 * curve.tc),
        curve.tc,
    )


def hybrid_ln_pressure(
    curve,
    switch_temperature,
    temperature,
    hvap_function,
):
    if temperature >= switch_temperature:
        return aw_ln_p(curve, temperature)
    return integrated_ln_pressure(
        switch_temperature,
        SWITCH_PRESSURE_BAR,
        temperature,
        hvap_function,
    )


def hybrid_temperature(
    curve,
    switch_temperature,
    target_pressure_bar,
    lower_temperature,
    hvap_function,
):
    if target_pressure_bar >= SWITCH_PRESSURE_BAR:
        return aw_temperature(curve, target_pressure_bar)
    target = math.log(target_pressure_bar)
    function = lambda temperature: (
        hybrid_ln_pressure(curve, switch_temperature, temperature, hvap_function)
        - target
    )
    if function(lower_temperature) > 0.0:
        return None
    return brentq(
        function,
        lower_temperature,
        switch_temperature,
        xtol=1.0e-10,
        rtol=1.0e-12,
    )


def models_for_curve(library, curve, tb):
    row = hvap_row_at_tb(library, curve, tb)
    if row is None:
        return None
    watson_fit = fit_local_watson(library, row, curve, tb)
    if watson_fit is None:
        return None
    switch_temperature = aw_temperature(curve, SWITCH_PRESSURE_BAR)
    if switch_temperature is None:
        return None
    tb_hvap, exponent, _fit_min, _fit_max = watson_fit
    correlation_function = lambda temperature: correlation_hvap_kJ_mol(
        library,
        row,
        curve,
        temperature,
    )
    watson_function = lambda temperature: watson_hvap_kJ_mol(
        curve,
        tb,
        tb_hvap,
        exponent,
        temperature,
    )
    aw_slope = aw_dlnp_dt(curve, switch_temperature)
    return {
        "row": row,
        "watson_exponent": exponent,
        "switch_temperature": switch_temperature,
        "correlation_function": correlation_function,
        "watson_function": watson_function,
        "correlation_slope_jump": (
            correlation_function(switch_temperature)
            * 1000.0
            / (R * switch_temperature**2)
            / aw_slope
            - 1.0
        ),
        "watson_slope_jump": (
            watson_function(switch_temperature)
            * 1000.0
            / (R * switch_temperature**2)
            / aw_slope
            - 1.0
        ),
    }


def evaluate_point(curve, models, temperature, pressure_bar, reference):
    switch_temperature = models["switch_temperature"]
    aw_prediction = aw_ln_p(curve, temperature)
    watson_prediction = hybrid_ln_pressure(
        curve,
        switch_temperature,
        temperature,
        models["watson_function"],
    )
    watson_temperature = hybrid_temperature(
        curve,
        switch_temperature,
        pressure_bar,
        max(1.0, 0.15 * curve.tc),
        models["watson_function"],
    )

    correlation_prediction = None
    correlation_temperature = None
    correlation_minimum = float(models["row"]["T_min_K"])
    if temperature >= correlation_minimum:
        correlation_prediction = hybrid_ln_pressure(
            curve,
            switch_temperature,
            temperature,
            models["correlation_function"],
        )
        correlation_temperature = hybrid_temperature(
            curve,
            switch_temperature,
            pressure_bar,
            correlation_minimum,
            models["correlation_function"],
        )

    aw_predicted_temperature = aw_temperature(curve, pressure_bar)
    return {
        "reference": reference,
        "cas": curve.cas,
        "name": curve.name,
        "pressure_bar": pressure_bar,
        "pressure_mmHg": pressure_bar * 750.061683,
        "temperature_K": temperature,
        "reduced_temperature": temperature / curve.tc,
        "switch_temperature_K": switch_temperature,
        "switch_reduced_temperature": switch_temperature / curve.tc,
        "watson_exponent": models["watson_exponent"],
        "correlation_slope_jump": models["correlation_slope_jump"],
        "watson_slope_jump": models["watson_slope_jump"],
        "aw_relative_error": math.exp(aw_prediction) / pressure_bar - 1.0,
        "correlation_hybrid_relative_error": (
            None
            if correlation_prediction is None
            else math.exp(correlation_prediction) / pressure_bar - 1.0
        ),
        "watson_hybrid_relative_error": (
            math.exp(watson_prediction) / pressure_bar - 1.0
        ),
        "aw_temperature_error_K": (
            None
            if aw_predicted_temperature is None
            else aw_predicted_temperature - temperature
        ),
        "correlation_hybrid_temperature_error_K": (
            None
            if correlation_temperature is None
            else correlation_temperature - temperature
        ),
        "watson_hybrid_temperature_error_K": (
            None
            if watson_temperature is None
            else watson_temperature - temperature
        ),
    }


def table_rows(library):
    rows = []
    excluded = []
    unavailable = []
    for curve, pairs, trusted_tb in tb_valid_matches():
        if is_banned(curve):
            excluded.append(curve.name)
            continue
        models = models_for_curve(library, curve, trusted_tb)
        if models is None:
            unavailable.append(curve.name)
            continue
        for temperature, pressure_bar in pairs:
            if pressure_bar > SWITCH_PRESSURE_BAR + 1.0e-6:
                continue
            rows.append(
                evaluate_point(
                    curve,
                    models,
                    temperature,
                    pressure_bar,
                    "Perry 2-10 table",
                )
            )
    return rows, excluded, unavailable


def correlation_rows(library):
    rows = []
    excluded = []
    unavailable = []
    for curve in load_curves():
        if is_banned(curve):
            excluded.append(curve.name)
            continue
        tb_result = library.normal_boiling_point_K(curve.cas)
        if tb_result is None:
            continue
        tb = float(tb_result.value)
        models = models_for_curve(library, curve, tb)
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
                evaluate_point(
                    curve,
                    models,
                    temperature,
                    pressure_bar,
                    "Perry 2-8 correlation",
                )
            )
    return rows, excluded, unavailable


def method_summary(rows, method):
    pressure_errors = [
        abs(row[f"{method}_relative_error"])
        for row in rows
        if row[f"{method}_relative_error"] is not None
    ]
    temperature_errors = [
        abs(row[f"{method}_temperature_error_K"])
        for row in rows
        if row[f"{method}_temperature_error_K"] is not None
    ]
    return (
        f"n={len(pressure_errors)} MARD={100*mean(pressure_errors):.2f}% "
        f"median={100*median(pressure_errors):.2f}% "
        f"p95={100*percentile(pressure_errors, .95):.2f}% "
        f"|dT| median={median(temperature_errors):.2f} K "
        f"p95={percentile(temperature_errors, .95):.2f} K"
    )


def report(rows, label):
    print(f"\n{label}: curves={len({row['cas'] for row in rows})} points={len(rows)}")
    for method in ("aw", "correlation_hybrid", "watson_hybrid"):
        print(f"  {method}: {method_summary(rows, method)}")
    unique = {row["cas"]: row for row in rows}.values()
    for method in ("correlation", "watson"):
        jumps = [abs(row[f"{method}_slope_jump"]) for row in unique]
        print(
            f"  {method} handoff |slope mismatch|: "
            f"median={100*median(jumps):.2f}% "
            f"p95={100*percentile(jumps, .95):.2f}%"
        )

    print("  by pressure:")
    for pressure_bar in sorted({row["pressure_bar"] for row in rows}, reverse=True):
        subset = [row for row in rows if row["pressure_bar"] == pressure_bar]
        pieces = []
        for method in ("aw", "correlation_hybrid", "watson_hybrid"):
            values = [
                abs(row[f"{method}_relative_error"])
                for row in subset
                if row[f"{method}_relative_error"] is not None
            ]
            pieces.append(
                f"{method} med={100*median(values):.2f}% "
                f"p95={100*percentile(values, .95):.2f}%"
            )
        print(
            f"    {pressure_bar * 750.061683:4.0f} mmHg: "
            + "; ".join(pieces)
        )


def main():
    library = PerryPropertyLibrary()
    library._load()
    table, table_excluded, table_unavailable = table_rows(library)
    correlations, correlation_excluded, correlation_unavailable = correlation_rows(library)
    print("AW TO 200 mmHg, THEN CLAPEYRON; ACIDS/POLAR INORGANICS EXCLUDED")
    report(table, "Perry 2-10 independent table points")
    report(correlations, "Perry 2-8 continuous correlations")
    print(
        f"\nExcluded: table={len(set(table_excluded))}, "
        f"correlations={len(set(correlation_excluded))}; "
        f"unavailable: table={len(set(table_unavailable))}, "
        f"correlations={len(set(correlation_unavailable))}"
    )

    rows = table + correlations
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
