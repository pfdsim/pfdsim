import csv
import math
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.interpolate import PchipInterpolator

from benchmark_perry_aw import (
    Curve,
    aw_dlnp_dt,
    aw_ln_p,
    completed_ln_p,
    normalize_vapor_pressure_coefficients,
    percentile,
    perry_dlnp_dt,
    perry_ln_p,
)
from perry_properties import PerryPropertyLibrary
from pressure_standards import NORMAL_BOILING_PRESSURE_BAR
from property_resolution.common import (
    ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K,
    ONLINE_ANTOINE_TB_REL_TOL,
)


def endpoint_bin(reduced_temperature):
    if reduced_temperature < 0.60:
        return "<0.60"
    if reduced_temperature < 0.65:
        return "0.60-0.65"
    if reduced_temperature < 0.70:
        return "0.65-0.70"
    if reduced_temperature < 0.75:
        return "0.70-0.75"
    return "0.75-0.80"


def load_matches():
    library = PerryPropertyLibrary()
    library._load()
    library._load_table_2_10()
    matches = []
    for cas, table_entry in library.table_2_10_chemicals.items():
        perry_entry = library.get(cas)
        if not perry_entry:
            continue
        vapor_rows = perry_entry.get("vapor_pressure") or []
        critical = perry_entry.get("critical_constants") or {}
        if len(vapor_rows) != 1 or any(critical.get(key) is None for key in ("Tc_K", "Pc_MPa", "omega")):
            continue
        pairs = sorted(
            (float(point["T_K"]), float(point["P_bar"]))
            for point in table_entry.get("vapor_pressure", [])
            if point.get("T_K") is not None and point.get("P_bar") is not None
        )
        if len(pairs) < 4:
            continue
        row = vapor_rows[0]
        tc = float(critical["Tc_K"])
        if not (
            float(row["T_min_K"]) <= pairs[-1][0] < tc
            and float(row["T_max_K"]) >= tc - 0.01
        ):
            continue
        curve = Curve(
            cas=cas,
            name=perry_entry.get("name") or table_entry.get("name") or cas,
            tc=tc,
            pc_bar=float(critical["Pc_MPa"]) * 10.0,
            omega=float(critical["omega"]),
            t_min=float(row["T_min_K"]),
            t_max=float(row["T_max_K"]),
            coefficients=normalize_vapor_pressure_coefficients(row),
        )
        matches.append((curve, pairs))
    return sorted(matches, key=lambda item: item[1][-1][0] / item[0].tc)


def tb_valid_matches(matches):
    library = PerryPropertyLibrary()
    validated = []
    for curve, pairs in matches:
        tb = library.normal_boiling_point_K(curve.cas)
        if tb is None:
            continue
        trusted_tb = float(tb.value)
        temperatures = [pair[0] for pair in pairs]
        ln_pressures = [math.log(pair[1]) for pair in pairs]
        if not (
            temperatures[0] - ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K
            <= trusted_tb
            <= temperatures[-1] + ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K
        ):
            continue
        pressure = math.exp(float(PchipInterpolator(
            temperatures,
            ln_pressures,
            extrapolate=True,
        )(trusted_tb)))
        if abs(pressure / NORMAL_BOILING_PRESSURE_BAR - 1.0) > ONLINE_ANTOINE_TB_REL_TOL:
            continue
        validated.append((curve, pairs))
    return validated


def pchip_slope(pairs):
    temperatures = np.array([pair[0] for pair in pairs])
    ln_pressures = np.log([pair[1] for pair in pairs])
    interpolator = PchipInterpolator(temperatures, ln_pressures)
    return float(interpolator.derivative()(temperatures[-1]))


def secant_slope(pairs):
    (t1, p1), (t2, p2) = pairs[-2:]
    return math.log(p2 / p1) / (t2 - t1)


def inverse_temperature_linear_slope(pairs, count):
    selected = pairs[-count:]
    temperatures = np.array([pair[0] for pair in selected])
    ln_pressures = np.log([pair[1] for pair in selected])
    inverse_temperature = 1.0 / temperatures
    slope, _ = np.polyfit(inverse_temperature, ln_pressures, 1)
    t0 = temperatures[-1]
    return float(-slope / t0**2)


def inverse_temperature_quadratic_slope(pairs, count):
    selected = pairs[-count:]
    temperatures = np.array([pair[0] for pair in selected])
    ln_pressures = np.log([pair[1] for pair in selected])
    t0 = temperatures[-1]
    coordinate = (1.0 / temperatures - 1.0 / t0) * t0
    coefficients = np.polynomial.polynomial.polyfit(coordinate, ln_pressures, 2)
    return float(-coefficients[1] / t0)


def temperature_quadratic_slope(pairs, count):
    selected = pairs[-count:]
    temperatures = np.array([pair[0] for pair in selected])
    ln_pressures = np.log([pair[1] for pair in selected])
    t0 = temperatures[-1]
    coordinate = (temperatures - t0) / t0
    coefficients = np.polynomial.polynomial.polyfit(coordinate, ln_pressures, 2)
    return float(coefficients[1] / t0)


SLOPE_METHODS = {
    "pchip_endpoint": pchip_slope,
    "last_two_secant": secant_slope,
    "clausius_linear_last3": lambda pairs: inverse_temperature_linear_slope(pairs, 3),
    "clausius_linear_last4": lambda pairs: inverse_temperature_linear_slope(pairs, 4),
    "clausius_linear_last5": lambda pairs: inverse_temperature_linear_slope(pairs, 5),
    "inverse_T_quadratic_last4": lambda pairs: inverse_temperature_quadratic_slope(pairs, 4),
    "inverse_T_quadratic_last5": lambda pairs: inverse_temperature_quadratic_slope(pairs, 5),
    "temperature_quadratic_last4": lambda pairs: temperature_quadratic_slope(pairs, 4),
}


def completed_table_ln_p(curve, endpoint_temperature, endpoint_pressure, endpoint_slope, temperature):
    span = curve.tc - endpoint_temperature
    x = (temperature - endpoint_temperature) / span
    endpoint_ln_pressure = math.log(endpoint_pressure)
    delta_value = endpoint_ln_pressure - aw_ln_p(curve, endpoint_temperature)
    delta_slope = endpoint_slope - aw_dlnp_dt(curve, endpoint_temperature)
    coefficient = span * delta_slope + 2.0 * delta_value
    return aw_ln_p(curve, temperature) + (1.0 - x) ** 2 * (delta_value + coefficient * x)


def evaluate(curve, pairs, method_name, slope_function):
    t0, p0 = pairs[-1]
    cutoff_tr = t0 / curve.tc
    slope = slope_function(pairs)
    reference_slope = perry_dlnp_dt(curve, t0)
    temperatures = [t0 + (curve.tc - t0) * index / 400.0 for index in range(401)]
    reference = [perry_ln_p(curve, temperature) for temperature in temperatures]
    completed = [
        completed_table_ln_p(curve, t0, p0, slope, temperature)
        for temperature in temperatures
    ]
    oracle = [completed_ln_p(curve, temperature, cutoff_tr, 2) for temperature in temperatures]
    errors = [math.exp(predicted - truth) - 1.0 for predicted, truth in zip(completed, reference)]
    oracle_errors = [math.exp(predicted - truth) - 1.0 for predicted, truth in zip(oracle, reference)]
    derivatives = [
        (completed[index + 1] - completed[index - 1]) / (temperatures[index + 1] - temperatures[index - 1])
        for index in range(1, len(temperatures) - 1)
    ]
    return {
        "method": method_name,
        "cas": curve.cas,
        "name": curve.name,
        "point_count": len(pairs),
        "endpoint_T_K": t0,
        "endpoint_Tr": cutoff_tr,
        "endpoint_bin": endpoint_bin(cutoff_tr),
        "endpoint_pressure_bar": p0,
        "perry_endpoint_relative_error": p0 / math.exp(perry_ln_p(curve, t0)) - 1.0,
        "estimated_slope": slope,
        "perry_slope": reference_slope,
        "slope_relative_error": slope / reference_slope - 1.0,
        "c1_mard": mean(abs(error) for error in errors),
        "c1_max_abs_relative_error": max(abs(error) for error in errors),
        "oracle_mard": mean(abs(error) for error in oracle_errors),
        "nonmonotone": min(derivatives) <= 0.0,
        "overshoots_pc": max(completed[:-1]) > math.log(curve.pc_bar) + 1.0e-10,
    }


def summarize(rows, label):
    slope_errors = [abs(row["slope_relative_error"]) for row in rows]
    pressure_errors = [abs(row["perry_endpoint_relative_error"]) for row in rows]
    mards = [row["c1_mard"] for row in rows]
    maxima = [row["c1_max_abs_relative_error"] for row in rows]
    print(f"{label}: n={len(rows)}")
    print(
        f"  endpoint |P mismatch| median={100*median(pressure_errors):.3f}% "
        f"p95={100*percentile(pressure_errors,.95):.3f}%"
    )
    print(
        f"  endpoint |slope error| median={100*median(slope_errors):.3f}% "
        f"p90={100*percentile(slope_errors,.90):.3f}% "
        f"p95={100*percentile(slope_errors,.95):.3f}%"
    )
    print(
        f"  completion MARD mean={100*mean(mards):.3f}% median={100*median(mards):.3f}% "
        f"p90={100*percentile(mards,.90):.3f}% p95={100*percentile(mards,.95):.3f}% "
        f"max={100*max(mards):.3f}%"
    )
    print(
        f"  max-error median={100*median(maxima):.3f}% p95={100*percentile(maxima,.95):.3f}% "
        f"nonmonotone={sum(row['nonmonotone'] for row in rows)} "
        f"overshoots={sum(row['overshoots_pc'] for row in rows)}"
    )


def main():
    raw_matches = load_matches()
    matches = tb_valid_matches(raw_matches)
    rows = []
    for curve, pairs in matches:
        for method_name, slope_function in SLOPE_METHODS.items():
            rows.append(evaluate(curve, pairs, method_name, slope_function))

    print(f"MATCHED TABLES: {len(raw_matches)}")
    print(f"TB-VALIDATED TABLES: {len(matches)}")
    print("\nDERIVATIVE ESTIMATOR COMPARISON")
    for method_name in SLOPE_METHODS:
        summarize([row for row in rows if row["method"] == method_name], method_name)

    method_medians = {
        method_name: median(
            row["c1_mard"] for row in rows if row["method"] == method_name
        )
        for method_name in SLOPE_METHODS
    }
    best_method = min(method_medians, key=method_medians.get)
    best_rows = [row for row in rows if row["method"] == best_method]
    print(f"\nBEST MEDIAN COMPLETION METHOD: {best_method}")
    print("\nBY ENDPOINT Tr")
    for label in ("<0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80"):
        subset = [row for row in best_rows if row["endpoint_bin"] == label]
        if subset:
            summarize(subset, label)

    print("\nWORST 20 WITH BEST METHOD")
    for row in sorted(best_rows, key=lambda item: item["c1_mard"], reverse=True)[:20]:
        print(
            f"  {row['name']} [{row['cas']}]: Tr0={row['endpoint_Tr']:.3f}, "
            f"P={100*row['perry_endpoint_relative_error']:+.2f}%, "
            f"slope={100*row['slope_relative_error']:+.2f}%, "
            f"MARD={100*row['c1_mard']:.2f}%, max={100*row['c1_max_abs_relative_error']:.2f}%"
        )

    oracle = [row["oracle_mard"] for row in best_rows]
    print(
        f"\nOracle Perry value+slope handoff at same endpoint: "
        f"mean={100*mean(oracle):.3f}% median={100*median(oracle):.3f}% "
        f"p95={100*percentile(oracle,.95):.3f}%"
    )

    output = Path("/tmp/perry_table210_aw_benchmark.csv")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
