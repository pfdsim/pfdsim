"""Infer fixed-enthalpy vapor-dimerization entropies from Perry curves.

For every Perry monocarboxylic acid with overlapping Table 2-8 Psat and
Table 2-69 Hvap correlations, use the differential Clapeyron relation and a
Peng-Robinson saturated-state delta Z to infer the dimer extent required by
the two curves.  Holding delta H at the production generic fallback value then
gives the pointwise delta S required by the dimer equilibrium relation.
"""

import csv
import math
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import numpy as np
from scipy.stats import t as student_t

from benchmark_direct_lower_completion import perry_cases
from benchmark_end_to_end_lower_completion import (
    MONOCARBOXYLIC_ACID_CAS,
    peng_robinson_delta_z,
)
from benchmark_perry_aw import percentile
from property_resolution.common import R
from vapor_dimerization import (
    GENERIC_MONOCARBOXYLIC_DELTA_H_J_MOL,
    GENERIC_MONOCARBOXYLIC_DELTA_S_J_MOL_K,
    get_dimerization_params,
)


FIXED_DELTA_H_J_MOL = GENERIC_MONOCARBOXYLIC_DELTA_H_J_MOL
GENERIC_DELTA_S_J_MOL_K = GENERIC_MONOCARBOXYLIC_DELTA_S_J_MOL_K
SAMPLE_COUNT = 201
REFERENCE_PRESSURES_BAR = (0.01, 0.1, 1.01325)
POINT_OUTPUT_PATH = Path("/tmp/perry_dimer_entropy_required_points.csv")
SUMMARY_OUTPUT_PATH = Path("/tmp/perry_dimer_entropy_required_summary.csv")
NORMAL_ACID_CARBON_COUNTS = {
    "64-18-6": 1,
    "64-19-7": 2,
    "79-09-4": 3,
    "107-92-6": 4,
    "109-52-4": 5,
    "142-62-1": 6,
    "111-14-8": 7,
    "124-07-2": 8,
    "112-05-0": 9,
    "334-48-5": 10,
}


def dimer_extent(
    temperature_K: float,
    pressure_bar: float,
    entropy_J_mol_K: float,
) -> float:
    exponent = (
        entropy_J_mol_K / R
        - FIXED_DELTA_H_J_MOL / (R * temperature_K)
        + math.log(pressure_bar)
    )
    equilibrium_pressure_product = math.exp(
        min(700.0, max(-700.0, exponent))
    )
    return 0.5 * (
        1.0
        - 1.0 / math.sqrt(1.0 + 4.0 * equilibrium_pressure_product)
    )


def required_entropy(
    temperature_K: float,
    pressure_bar: float,
    extent: float,
) -> float:
    pressure_function = (
        extent
        * (1.0 - extent)
        / (1.0 - 2.0 * extent) ** 2
    )
    equilibrium_constant = pressure_function / pressure_bar
    entropy = (
        R * math.log(equilibrium_constant)
        + FIXED_DELTA_H_J_MOL / temperature_K
    )
    reconstructed = dimer_extent(
        temperature_K,
        pressure_bar,
        entropy,
    )
    if abs(reconstructed - extent) > 2.0e-11:
        raise ArithmeticError("Dimer entropy inversion failed its round trip")
    return entropy


def evaluate_point(case, temperature_K: float) -> dict:
    row = {
        "cas": case.cas,
        "name": case.name,
        "temperature_K": float(temperature_K),
        "pressure_bar": None,
        "pr_delta_z": None,
        "perry_hvap_J_mol": None,
        "perry_dlnp_dT_per_K": None,
        "required_apparent_hvap_J_mol": None,
        "required_extent": None,
        "required_delta_S_J_mol_K": None,
        "generic_extent": None,
        "generic_slope_relative_error": None,
        "status": "unavailable",
    }
    try:
        ln_pressure = case.ln_pressure(temperature_K)
        pressure_bar = math.exp(ln_pressure)
        slope = case.dln_pressure_dT(temperature_K)
        hvap_J_mol = case.hvap_J_mol(temperature_K)
        delta_z = peng_robinson_delta_z(
            SimpleNamespace(case=case),
            temperature_K,
            pressure_bar,
            case.preferred_omega,
        )
        apparent_hvap = R * temperature_K**2 * delta_z * slope
        required_extent = 1.0 - hvap_J_mol / apparent_hvap
        generic_extent = dimer_extent(
            temperature_K,
            pressure_bar,
            GENERIC_DELTA_S_J_MOL_K,
        )
        generic_slope = (
            hvap_J_mol
            / (
                R
                * temperature_K**2
                * delta_z
                * (1.0 - generic_extent)
            )
        )
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return row

    row.update({
        "pressure_bar": pressure_bar,
        "pr_delta_z": delta_z,
        "perry_hvap_J_mol": hvap_J_mol,
        "perry_dlnp_dT_per_K": slope,
        "required_apparent_hvap_J_mol": apparent_hvap,
        "required_extent": required_extent,
        "generic_extent": generic_extent,
        "generic_slope_relative_error": generic_slope / slope - 1.0,
    })
    if required_extent <= 0.0:
        row["status"] = "no_positive_association_solution"
    elif required_extent >= 0.5:
        row["status"] = "extent_at_or_above_half"
    else:
        row["status"] = "inferred"
        row["required_delta_S_J_mol_K"] = required_entropy(
            temperature_K,
            pressure_bar,
            required_extent,
        )
    return row


def reference_pressure_entropy(case, pressure_bar: float):
    temperature = case.temperature_at_pressure(pressure_bar)
    if temperature is None:
        return None
    point = evaluate_point(case, temperature)
    if point["status"] != "inferred":
        return None
    return point["required_delta_S_J_mol_K"]


def summarize_case(case, points: list[dict]) -> dict:
    inferred = [
        row for row in points
        if row["status"] == "inferred"
    ]
    pr_valid = [
        row for row in points
        if row["status"] != "unavailable"
    ]
    entropies = [
        float(row["required_delta_S_J_mol_K"])
        for row in inferred
    ]
    slope_errors = [
        abs(float(row["generic_slope_relative_error"]))
        for row in pr_valid
    ]
    current = get_dimerization_params(case.cas) or {}
    summary = {
        "cas": case.cas,
        "name": case.name,
        "normal_acid_carbon_count": NORMAL_ACID_CARBON_COUNTS.get(case.cas),
        "current_source": current.get("source"),
        "current_delta_H_J_mol": current.get("delta_H_J_per_mol"),
        "current_delta_S_J_mol_K": current.get("delta_S_J_per_mol_K"),
        "sample_count": len(points),
        "pr_valid_count": len(pr_valid),
        "inferred_count": len(inferred),
        "no_positive_association_count": sum(
            row["status"] == "no_positive_association_solution"
            for row in points
        ),
        "extent_at_or_above_half_count": sum(
            row["status"] == "extent_at_or_above_half"
            for row in points
        ),
        "required_delta_S_median_J_mol_K": (
            median(entropies) if entropies else None
        ),
        "required_delta_S_p05_J_mol_K": (
            percentile(entropies, 0.05) if entropies else None
        ),
        "required_delta_S_p95_J_mol_K": (
            percentile(entropies, 0.95) if entropies else None
        ),
        "generic_abs_slope_error_median": (
            median(slope_errors) if slope_errors else None
        ),
        "generic_abs_slope_error_p95": (
            percentile(slope_errors, 0.95) if slope_errors else None
        ),
    }
    for pressure_bar in REFERENCE_PRESSURES_BAR:
        label = str(pressure_bar).replace(".", "p")
        summary[f"required_delta_S_at_{label}_bar_J_mol_K"] = (
            reference_pressure_entropy(case, pressure_bar)
        )
    return summary


def linear_size_fit(summary_rows: list[dict], field_name: str):
    points = [
        (
            int(row["normal_acid_carbon_count"]),
            float(row[field_name]),
        )
        for row in summary_rows
        if row["normal_acid_carbon_count"] is not None
        and int(row["normal_acid_carbon_count"]) >= 4
        and row[field_name] is not None
    ]
    if len(points) < 3:
        return None
    carbon_numbers = np.asarray([point[0] for point in points], dtype=float)
    entropies = np.asarray([point[1] for point in points], dtype=float)
    slope, intercept = np.polyfit(carbon_numbers, entropies, 1)
    predictions = intercept + slope * carbon_numbers
    residuals = entropies - predictions
    residual_sum = float(residuals @ residuals)
    total_sum = float((entropies - entropies.mean()) @ (entropies - entropies.mean()))
    degrees_of_freedom = len(points) - 2
    slope_standard_error = math.sqrt(
        (residual_sum / degrees_of_freedom)
        / float(
            (carbon_numbers - carbon_numbers.mean())
            @ (carbon_numbers - carbon_numbers.mean())
        )
    )
    critical_value = float(student_t.ppf(0.975, degrees_of_freedom))
    return {
        "count": len(points),
        "slope_J_mol_K_per_carbon": float(slope),
        "entropy_at_C4_J_mol_K": float(intercept + 4.0 * slope),
        "r_squared": (
            1.0 - residual_sum / total_sum
            if total_sum > 0.0 else 1.0
        ),
        "rmse_J_mol_K": math.sqrt(residual_sum / len(points)),
        "slope_ci95_low": float(slope - critical_value * slope_standard_error),
        "slope_ci95_high": float(slope + critical_value * slope_standard_error),
    }


def normal_series_error_statistics(
    point_rows: list[dict],
    entropy_J_mol_K: float,
) -> dict:
    by_acid = {}
    point_errors = []
    for row in point_rows:
        carbon_count = NORMAL_ACID_CARBON_COUNTS.get(row["cas"])
        if (
            carbon_count is None
            or carbon_count < 4
            or row["status"] == "unavailable"
        ):
            continue
        temperature = float(row["temperature_K"])
        pressure = float(row["pressure_bar"])
        extent = dimer_extent(
            temperature,
            pressure,
            entropy_J_mol_K,
        )
        predicted_slope = (
            float(row["perry_hvap_J_mol"])
            / (
                R
                * temperature**2
                * float(row["pr_delta_z"])
                * (1.0 - extent)
            )
        )
        relative_error = abs(
            predicted_slope
            / float(row["perry_dlnp_dT_per_K"])
            - 1.0
        )
        point_errors.append(relative_error)
        by_acid.setdefault(row["cas"], []).append(relative_error)
    curve_mards = [
        sum(errors) / len(errors)
        for errors in by_acid.values()
    ]
    return {
        "point_mean": sum(point_errors) / len(point_errors),
        "point_median": median(point_errors),
        "point_p95": percentile(point_errors, 0.95),
        "curve_mard_median": median(curve_mards),
        "curve_mard_p95": percentile(curve_mards, 0.95),
        "curve_mard_maximum": max(curve_mards),
    }


def evaluate():
    cases, unavailable = perry_cases()
    acid_cases = [
        case for case in cases
        if case.cas in MONOCARBOXYLIC_ACID_CAS
    ]
    point_rows = []
    summary_rows = []
    for case in acid_cases:
        points = [
            evaluate_point(case, float(temperature))
            for temperature in np.linspace(
                case.T_min,
                case.T_max,
                SAMPLE_COUNT,
            )
        ]
        point_rows.extend(points)
        summary_rows.append(summarize_case(case, points))
    return point_rows, summary_rows, unavailable


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(summary_rows: list[dict], point_rows: list[dict]) -> None:
    fallback_rows = [
        row for row in summary_rows
        if str(row["current_source"]).startswith("estimated")
    ]
    inferred_points = [
        row for row in point_rows
        if row["status"] == "inferred"
    ]
    pr_valid_points = [
        row for row in point_rows
        if row["status"] != "unavailable"
    ]
    medians = [
        float(row["required_delta_S_median_J_mol_K"])
        for row in fallback_rows
        if row["required_delta_S_median_J_mol_K"] is not None
    ]
    within_ten = sum(
        abs(value - GENERIC_DELTA_S_J_MOL_K) <= 10.0
        for value in medians
    )
    print("PERRY FIXED-DELTA-H DIMER ENTROPY INVERSION")
    print(
        f"delta_H={FIXED_DELTA_H_J_MOL / 1000.0:.3f} kJ/mol; "
        f"acids={len(summary_rows)}; points={len(point_rows)}; "
        f"PR-valid={len(pr_valid_points)}; inferred={len(inferred_points)}"
    )
    print(
        f"generic-fallback acids={len(fallback_rows)}; "
        f"per-acid median delta_S median/p05/p95="
        f"{median(medians):.2f}/"
        f"{percentile(medians, 0.05):.2f}/"
        f"{percentile(medians, 0.95):.2f} J/mol/K; "
        f"within 10 J/mol/K of {GENERIC_DELTA_S_J_MOL_K:g}="
        f"{within_ten}/{len(medians)}"
    )
    print("normal saturated fallback series C4-C10:")
    for field_name, label in (
        ("required_delta_S_median_J_mol_K", "whole-overlap median"),
        ("required_delta_S_at_0p01_bar_J_mol_K", "0.01 bar"),
        ("required_delta_S_at_0p1_bar_J_mol_K", "0.1 bar"),
        ("required_delta_S_at_1p01325_bar_J_mol_K", "1.01325 bar"),
    ):
        fit = linear_size_fit(summary_rows, field_name)
        if fit is None:
            continue
        print(
            f"  {label:20s} S(C4)={fit['entropy_at_C4_J_mol_K']:.2f}; "
            f"slope={fit['slope_J_mol_K_per_carbon']:+.3f} "
            f"J/mol/K/carbon; 95% CI "
            f"[{fit['slope_ci95_low']:+.3f},"
            f"{fit['slope_ci95_high']:+.3f}]; "
            f"R2={fit['r_squared']:.3f}; "
            f"RMSE={fit['rmse_J_mol_K']:.2f}"
        )
    print("normal saturated C4-C10 direct constant comparison:")
    for entropy in (-140.0, -143.0, -144.0, -145.0):
        statistics = normal_series_error_statistics(point_rows, entropy)
        print(
            f"  S={entropy:6.1f} point med/p95="
            f"{100.0 * statistics['point_median']:.2f}%/"
            f"{100.0 * statistics['point_p95']:.2f}%; "
            f"curve MARD med/p95/max="
            f"{100.0 * statistics['curve_mard_median']:.2f}%/"
            f"{100.0 * statistics['curve_mard_p95']:.2f}%/"
            f"{100.0 * statistics['curve_mard_maximum']:.2f}%"
        )
    print()
    print(
        "name".ljust(27)
        + " source      infer/no+/half/bad  "
        + "S med [p05,p95]       generic slope |err| med/p95"
    )
    for row in sorted(summary_rows, key=lambda item: str(item["name"])):
        source = (
            "database"
            if row["current_source"] == "database"
            else "fallback"
        )
        entropy = row["required_delta_S_median_J_mol_K"]
        p05 = row["required_delta_S_p05_J_mol_K"]
        p95 = row["required_delta_S_p95_J_mol_K"]
        print(
            str(row["name"])[:26].ljust(27)
            + source.ljust(12)
            + f"{row['inferred_count']:3d}/"
            + f"{row['no_positive_association_count']:3d}/"
            + f"{row['extent_at_or_above_half_count']:3d}/"
            + f"{row['sample_count'] - row['pr_valid_count']:3d}  "
            + f"{entropy:7.2f} [{p05:7.2f},{p95:7.2f}]  "
            + f"{100.0 * row['generic_abs_slope_error_median']:7.2f}%/"
            + f"{100.0 * row['generic_abs_slope_error_p95']:7.2f}%"
        )


def main():
    point_rows, summary_rows, unavailable = evaluate()
    report(summary_rows, point_rows)
    write_csv(POINT_OUTPUT_PATH, point_rows)
    write_csv(SUMMARY_OUTPUT_PATH, summary_rows)
    print(f"\nwrote {POINT_OUTPUT_PATH}")
    print(f"wrote {SUMMARY_OUTPUT_PATH}")
    print(f"non-acid Perry cases unavailable for unrelated reasons={len(unavailable)}")


if __name__ == "__main__":
    main()
