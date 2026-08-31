import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

from benchmark_lower_clapeyron import (
    correlation_hvap_kJ_mol,
    integrated_ln_pressure,
    watson_hvap_kJ_mol,
)
from benchmark_lower_psat import PRESSURE_LEVELS_BAR, invert_ln_pressure
from benchmark_perry_aw import load_curves, percentile, perry_ln_p
from benchmark_switch_cutoff import (
    WATSON_BOUNDS,
    build_model,
    clipped,
    linear_omega_ln_pressure,
)
from benchmark_lower_hybrid import is_banned
from perry_properties import PerryPropertyLibrary


SAMPLE_TEMPERATURES_K = (273.15, 298.15, 313.15, 333.15, 353.15, 373.15)
SWITCH_PRESSURE_BAR = 0.25
OUTPUT_PATH = Path("/tmp/sparse_hvap_watson_benchmark.csv")


def realistic_watson_fit(library, curve, model):
    row = model["row"]
    tb = model["tb"]
    minimum = float(row["T_min_K"])
    maximum = min(float(row["T_max_K"]), curve.tc)
    tb_hvap = correlation_hvap_kJ_mol(library, row, curve, tb)
    if tb_hvap is None or tb_hvap <= 0.0:
        return None

    temperatures = [
        temperature
        for temperature in SAMPLE_TEMPERATURES_K
        if minimum <= temperature < maximum
    ]
    if all(abs(temperature - tb) > 1.0e-9 for temperature in temperatures):
        temperatures.append(tb)
    temperatures.sort()
    if len(temperatures) < 3:
        return None

    reduced_reference = 1.0 - tb / curve.tc
    coordinates = []
    responses = []
    samples = []
    for temperature in temperatures:
        hvap = correlation_hvap_kJ_mol(library, row, curve, temperature)
        if hvap is None or hvap <= 0.0:
            return None
        samples.append((temperature, hvap))
        coordinate = math.log(
            (1.0 - temperature / curve.tc) / reduced_reference
        )
        if abs(coordinate) <= 1.0e-14:
            continue
        coordinates.append(coordinate)
        responses.append(math.log(hvap / tb_hvap))

    denominator = sum(coordinate**2 for coordinate in coordinates)
    if denominator <= 0.0:
        return None
    exponent = sum(
        coordinate * response
        for coordinate, response in zip(coordinates, responses)
    ) / denominator
    return {
        "tb_hvap": tb_hvap,
        "exponent": exponent,
        "bounded_exponent": clipped(exponent, WATSON_BOUNDS),
        "samples": samples,
    }


def switch_temperature(curve, model):
    return invert_ln_pressure(
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


def prediction_error(
    curve,
    model,
    temperature,
    pressure_bar,
    switch_temperature_K,
    hvap_function,
):
    predicted = integrated_ln_pressure(
        switch_temperature_K,
        SWITCH_PRESSURE_BAR,
        temperature,
        hvap_function,
    )
    return math.exp(predicted) / pressure_bar - 1.0


def rows_for_curve(library, curve, model, fit):
    switch_temperature_K = switch_temperature(curve, model)
    if switch_temperature_K is None:
        return []
    row = model["row"]
    row_minimum = float(row["T_min_K"])
    local_watson = model["watson_function"]
    realistic_unbounded = lambda temperature: watson_hvap_kJ_mol(
        curve,
        model["tb"],
        fit["tb_hvap"],
        fit["exponent"],
        temperature,
    )
    realistic_bounded = lambda temperature: watson_hvap_kJ_mol(
        curve,
        model["tb"],
        fit["tb_hvap"],
        fit["bounded_exponent"],
        temperature,
    )
    direct = lambda temperature: correlation_hvap_kJ_mol(
        library,
        row,
        curve,
        temperature,
    )

    rows = []
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
        direct_error = None
        if temperature >= row_minimum and switch_temperature_K >= row_minimum:
            direct_error = prediction_error(
                curve,
                model,
                temperature,
                pressure_bar,
                switch_temperature_K,
                direct,
            )
        rows.append(
            {
                "cas": curve.cas,
                "name": curve.name,
                "pressure_bar": pressure_bar,
                "pressure_mmHg": pressure_bar * 750.061683,
                "temperature_K": temperature,
                "sample_count": len(fit["samples"]),
                "sample_temperatures_K": ";".join(
                    f"{sample_temperature:.2f}"
                    for sample_temperature, _hvap in fit["samples"]
                ),
                "local_exponent": model["fitted_exponent"],
                "local_bounded_exponent": model["bounded_exponent"],
                "realistic_exponent": fit["exponent"],
                "realistic_bounded_exponent": fit["bounded_exponent"],
                "direct_relative_error": direct_error,
                "local_watson_relative_error": prediction_error(
                    curve,
                    model,
                    temperature,
                    pressure_bar,
                    switch_temperature_K,
                    local_watson,
                ),
                "realistic_unbounded_relative_error": prediction_error(
                    curve,
                    model,
                    temperature,
                    pressure_bar,
                    switch_temperature_K,
                    realistic_unbounded,
                ),
                "realistic_bounded_relative_error": prediction_error(
                    curve,
                    model,
                    temperature,
                    pressure_bar,
                    switch_temperature_K,
                    realistic_bounded,
                ),
            }
        )
    return rows


def method_errors(rows, method):
    return [
        abs(row[f"{method}_relative_error"])
        for row in rows
        if row[f"{method}_relative_error"] is not None
    ]


def report(rows, eligible, unavailable):
    print("REALISTIC SPARSE-HVAP WATSON FIT, LINEAR OMEGA TO 0.25 BAR")
    print(
        f"eligible curves={eligible} fitted curves={len({row['cas'] for row in rows})} "
        f"unavailable={unavailable} points={len(rows)}"
    )
    counts = Counter()
    for row in {row["cas"]: row for row in rows}.values():
        counts[row["sample_count"]] += 1
    print(
        "sample counts: "
        + ", ".join(f"{count} points={curves}" for count, curves in sorted(counts.items()))
    )
    for method in (
        "direct",
        "local_watson",
        "realistic_unbounded",
        "realistic_bounded",
    ):
        errors = method_errors(rows, method)
        print(
            f"{method}: n={len(errors)} MARD={100*mean(errors):.2f}% "
            f"median={100*median(errors):.2f}% "
            f"p95={100*percentile(errors, .95):.2f}%"
        )

    print("by pressure:")
    for pressure_bar in sorted({row["pressure_bar"] for row in rows}, reverse=True):
        subset = [row for row in rows if row["pressure_bar"] == pressure_bar]
        pieces = []
        for method in (
            "direct",
            "local_watson",
            "realistic_unbounded",
            "realistic_bounded",
        ):
            errors = method_errors(subset, method)
            pieces.append(
                f"{method} med={100*median(errors):.2f}% p95={100*percentile(errors, .95):.2f}%"
            )
        print(f"  {pressure_bar * 750.061683:4.0f} mmHg: " + "; ".join(pieces))

    unique = {row["cas"]: row for row in rows}.values()
    local_exponents = [row["local_bounded_exponent"] for row in unique]
    realistic_exponents = [row["realistic_bounded_exponent"] for row in unique]
    print(
        f"bounded exponent median: local={median(local_exponents):.4f} "
        f"realistic={median(realistic_exponents):.4f}"
    )


def main():
    library = PerryPropertyLibrary()
    library._load()
    rows = []
    eligible = 0
    unavailable = 0
    for curve in load_curves():
        if is_banned(curve):
            continue
        tb_result = library.normal_boiling_point_K(curve.cas)
        if tb_result is None:
            continue
        model = build_model(library, curve, float(tb_result.value))
        if model is None:
            continue
        eligible += 1
        fit = realistic_watson_fit(library, curve, model)
        if fit is None:
            unavailable += 1
            continue
        rows.extend(rows_for_curve(library, curve, model, fit))

    report(rows, eligible, unavailable)
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
