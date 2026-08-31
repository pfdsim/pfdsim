import csv
import math
from pathlib import Path
from statistics import mean, median

from scipy.integrate import quad
from scipy.optimize import brentq

from benchmark_lower_psat import PRESSURE_LEVELS_BAR, invert_ln_pressure
from benchmark_perry_aw import load_curves, percentile, perry_ln_p
from benchmark_table210_nannoolal_aw import tb_valid_matches
from perry_properties import PerryPropertyLibrary
from property_resolution.common import R


OUTPUT_PATH = Path("/tmp/lower_clapeyron_benchmark.csv")


def hvap_row_at_tb(library, curve, tb):
    entry = library.get(curve.cas)
    if not entry:
        return None
    for row in entry.get("heat_of_vaporization", []):
        if float(row["T_min_K"]) <= tb <= float(row["T_max_K"]):
            return row
    return None


def correlation_hvap_kJ_mol(library, row, curve, temperature):
    value = library._eval_heat_of_vaporization_J_per_kmol(
        row,
        temperature,
        curve.tc,
    )
    return None if value is None else value / 1.0e6


def fit_local_watson(library, row, curve, tb):
    tb_hvap = correlation_hvap_kJ_mol(library, row, curve, tb)
    if tb_hvap is None:
        return None
    half_width = min(
        0.025 * curve.tc,
        tb - float(row["T_min_K"]),
        float(row["T_max_K"]) - tb,
    )
    if half_width <= 0.0:
        return None
    temperatures = [tb + half_width * (index - 2) / 2.0 for index in range(5)]
    reduced_reference = 1.0 - tb / curve.tc
    coordinates = []
    responses = []
    for temperature in temperatures:
        hvap = correlation_hvap_kJ_mol(library, row, curve, temperature)
        if hvap is None:
            return None
        coordinates.append(
            math.log((1.0 - temperature / curve.tc) / reduced_reference)
        )
        responses.append(math.log(hvap / tb_hvap))
    denominator = sum(value**2 for value in coordinates)
    if denominator <= 0.0:
        return None
    exponent = sum(
        coordinate * response
        for coordinate, response in zip(coordinates, responses)
    ) / denominator
    return tb_hvap, exponent, temperatures[0], temperatures[-1]


def watson_hvap_kJ_mol(curve, tb, tb_hvap, exponent, temperature):
    return tb_hvap * (
        (1.0 - temperature / curve.tc) / (1.0 - tb / curve.tc)
    ) ** exponent


def integrated_ln_pressure(tb, pressure_bar, temperature, hvap_function):
    integral, _error = quad(
        lambda value: hvap_function(value) * 1000.0 / (R * value**2),
        tb,
        temperature,
        epsabs=1.0e-10,
        epsrel=1.0e-10,
        limit=100,
    )
    return math.log(pressure_bar) + integral


def integrated_temperature(
    tb,
    pressure_bar,
    target_pressure_bar,
    lower_temperature,
    hvap_function,
):
    target = math.log(target_pressure_bar)
    function = lambda temperature: (
        integrated_ln_pressure(tb, pressure_bar, temperature, hvap_function)
        - target
    )
    if function(lower_temperature) > 0.0 or function(tb) < 0.0:
        return None
    return brentq(function, lower_temperature, tb, xtol=1.0e-10, rtol=1.0e-12)


def evaluate_point(
    library,
    curve,
    row,
    watson_fit,
    anchor_temperature,
    anchor_pressure_bar,
    temperature,
    pressure_bar,
    reference,
):
    tb_hvap, exponent, fit_min, fit_max = watson_fit
    watson_function = lambda value: watson_hvap_kJ_mol(
        curve,
        anchor_temperature,
        tb_hvap,
        exponent,
        value,
    )
    watson_ln_pressure = integrated_ln_pressure(
        anchor_temperature,
        anchor_pressure_bar,
        temperature,
        watson_function,
    )
    watson_temperature = integrated_temperature(
        anchor_temperature,
        anchor_pressure_bar,
        pressure_bar,
        max(1.0, 0.15 * curve.tc),
        watson_function,
    )

    correlation_ln_pressure = None
    correlation_temperature = None
    correlation_minimum = float(row["T_min_K"])
    if temperature >= correlation_minimum:
        correlation_function = lambda value: correlation_hvap_kJ_mol(
            library,
            row,
            curve,
            value,
        )
        correlation_ln_pressure = integrated_ln_pressure(
            anchor_temperature,
            anchor_pressure_bar,
            temperature,
            correlation_function,
        )
        correlation_temperature = integrated_temperature(
            anchor_temperature,
            anchor_pressure_bar,
            pressure_bar,
            correlation_minimum,
            correlation_function,
        )

    return {
        "reference": reference,
        "cas": curve.cas,
        "name": curve.name,
        "pressure_bar": pressure_bar,
        "pressure_mmHg": pressure_bar * 750.061683,
        "temperature_K": temperature,
        "reduced_temperature": temperature / curve.tc,
        "anchor_temperature_K": anchor_temperature,
        "watson_exponent": exponent,
        "watson_fit_min_K": fit_min,
        "watson_fit_max_K": fit_max,
        "correlation_in_range": correlation_ln_pressure is not None,
        "correlation_relative_error": (
            None
            if correlation_ln_pressure is None
            else math.exp(correlation_ln_pressure) / pressure_bar - 1.0
        ),
        "watson_relative_error": math.exp(watson_ln_pressure) / pressure_bar - 1.0,
        "correlation_temperature_error_K": (
            None
            if correlation_temperature is None
            else correlation_temperature - temperature
        ),
        "watson_temperature_error_K": (
            None
            if watson_temperature is None
            else watson_temperature - temperature
        ),
    }


def table_rows(library):
    rows = []
    unavailable = []
    for curve, pairs, _trusted_tb in tb_valid_matches():
        anchor_temperature, anchor_pressure = pairs[-1]
        row = hvap_row_at_tb(library, curve, anchor_temperature)
        if row is None:
            unavailable.append((curve.name, "Hvap unavailable at table anchor"))
            continue
        watson_fit = fit_local_watson(library, row, curve, anchor_temperature)
        if watson_fit is None:
            unavailable.append((curve.name, "Watson fit unavailable"))
            continue
        for temperature, pressure_bar in pairs[:-1]:
            rows.append(
                evaluate_point(
                    library,
                    curve,
                    row,
                    watson_fit,
                    anchor_temperature,
                    anchor_pressure,
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
        tb_result = library.normal_boiling_point_K(curve.cas)
        if tb_result is None:
            continue
        tb = float(tb_result.value)
        row = hvap_row_at_tb(library, curve, tb)
        if row is None:
            unavailable.append((curve.name, "Hvap unavailable at Tb"))
            continue
        watson_fit = fit_local_watson(library, row, curve, tb)
        if watson_fit is None:
            unavailable.append((curve.name, "Watson fit unavailable"))
            continue
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
            if temperature is None:
                continue
            rows.append(
                evaluate_point(
                    library,
                    curve,
                    row,
                    watson_fit,
                    tb,
                    1.01325,
                    temperature,
                    pressure_bar,
                    "Perry 2-8 correlation",
                )
            )
    return rows, unavailable


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
        len(pressure_errors),
        f"MARD={100*mean(pressure_errors):.2f}% "
        f"median={100*median(pressure_errors):.2f}% "
        f"p95={100*percentile(pressure_errors, .95):.2f}% "
        f"|dT| median={median(temperature_errors):.2f} K "
        f"p95={percentile(temperature_errors, .95):.2f} K",
    )


def report(rows, label):
    print(f"\n{label}: curves={len({row['cas'] for row in rows})} points={len(rows)}")
    for method in ("correlation", "watson"):
        count, summary = method_summary(rows, method)
        print(f"  {method}: n={count} {summary}")
    exponents = [row["watson_exponent"] for row in rows]
    print(
        f"  Watson exponent: median={median(exponents):.4f} "
        f"p05={percentile(exponents, .05):.4f} "
        f"p95={percentile(exponents, .95):.4f}"
    )

    print("  by pressure:")
    for pressure_bar in sorted({row["pressure_bar"] for row in rows}, reverse=True):
        subset = [row for row in rows if row["pressure_bar"] == pressure_bar]
        pieces = []
        for method in ("correlation", "watson"):
            errors = [
                abs(row[f"{method}_relative_error"])
                for row in subset
                if row[f"{method}_relative_error"] is not None
            ]
            pieces.append(
                f"{method} n={len(errors)} median={100*median(errors):.2f}% "
                f"p95={100*percentile(errors, .95):.2f}%"
            )
        print(
            f"    {pressure_bar * 750.061683:4.0f} mmHg: "
            + "; ".join(pieces)
        )


def main():
    library = PerryPropertyLibrary()
    library._load()
    table, table_unavailable = table_rows(library)
    correlations, correlation_unavailable = correlation_rows(library)
    print("LOWER-RANGE CLAPEYRON INTEGRATION, deltaZ=1")
    report(table, "Perry 2-10 independent table points")
    report(correlations, "Perry 2-8 continuous correlations")
    print(
        f"\nUnavailable: table={len(table_unavailable)}, "
        f"correlations={len(correlation_unavailable)}"
    )

    rows = table + correlations
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
