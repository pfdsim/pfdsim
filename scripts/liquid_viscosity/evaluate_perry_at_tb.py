#!/usr/bin/env python3
"""Evaluate every Perry liquid-viscosity correlation at normal boiling point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perry_properties import DATA_PATH, PerryPropertyLibrary  # noqa: E402
from compound_identity import parse_formula_counts  # noqa: E402


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    coordinate = (len(ordered) - 1) * quantile
    lower = int(math.floor(coordinate))
    upper = int(math.ceil(coordinate))
    if lower == upper:
        return ordered[lower]
    fraction = coordinate - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def value_summary(values: Sequence[float]) -> dict[str, float | int]:
    values = [float(value) for value in values]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "minimum": min(values),
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "mean": statistics.fmean(values),
        "geometric_mean": math.exp(statistics.fmean(math.log(value) for value in values)),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": max(values),
    }


def viscosity_band(value_mPa_s: float) -> str:
    if value_mPa_s < 0.10:
        return "<0.10"
    if value_mPa_s < 0.20:
        return "0.10-0.20"
    if value_mPa_s < 0.30:
        return "0.20-0.30"
    if value_mPa_s < 0.50:
        return "0.30-0.50"
    if value_mPa_s < 1.00:
        return "0.50-1.00"
    return ">=1.00"


def acentric_correlation(records: Sequence[dict]) -> dict[str, float | int]:
    selected = [
        record for record in records
        if record["viscosity_mPa_s"] >= 0.10
        and record.get("acentric_factor") is not None
    ]
    omega = [float(record["acentric_factor"]) for record in selected]
    viscosity = [float(record["viscosity_mPa_s"]) for record in selected]
    log_viscosity = [math.log(value) for value in viscosity]

    pearson = stats.pearsonr(omega, viscosity)
    spearman = stats.spearmanr(omega, viscosity)
    kendall = stats.kendalltau(omega, viscosity)
    linear = stats.linregress(omega, viscosity)
    log_linear = stats.linregress(omega, log_viscosity)
    linear_residuals = [
        value - (linear.intercept + linear.slope * factor)
        for factor, value in zip(omega, viscosity)
    ]
    log_residuals = [
        value - (log_linear.intercept + log_linear.slope * factor)
        for factor, value in zip(omega, log_viscosity)
    ]

    lower = percentile(viscosity, 0.05)
    upper = percentile(viscosity, 0.95)
    central = [
        (factor, value) for factor, value in zip(omega, viscosity)
        if lower <= value <= upper
    ]
    central_omega = [factor for factor, _ in central]
    central_viscosity = [value for _, value in central]
    central_pearson = stats.pearsonr(central_omega, central_viscosity)
    central_spearman = stats.spearmanr(central_omega, central_viscosity)

    return {
        "count": len(selected),
        "excluded_below_0_10_mPa_s": sum(
            record["viscosity_mPa_s"] < 0.10 for record in records
        ),
        "missing_acentric_factor": sum(
            record["viscosity_mPa_s"] >= 0.10
            and record.get("acentric_factor") is None
            for record in records
        ),
        "pearson_r": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "kendall_tau": float(kendall.statistic),
        "kendall_p_value": float(kendall.pvalue),
        "linear_fit_intercept_mPa_s": float(linear.intercept),
        "linear_fit_slope_mPa_s_per_omega": float(linear.slope),
        "linear_fit_r_squared": float(linear.rvalue**2),
        "linear_fit_rmse_mPa_s": math.sqrt(
            statistics.fmean(residual * residual for residual in linear_residuals)
        ),
        "log_linear_fit_intercept": float(log_linear.intercept),
        "log_linear_fit_slope_per_omega": float(log_linear.slope),
        "log_linear_fit_r_squared": float(log_linear.rvalue**2),
        "log_linear_fit_rmse": math.sqrt(
            statistics.fmean(residual * residual for residual in log_residuals)
        ),
        "central_90_percent_count": len(central),
        "central_90_percent_pearson_r": float(central_pearson.statistic),
        "central_90_percent_pearson_p_value": float(central_pearson.pvalue),
        "central_90_percent_spearman_rho": float(central_spearman.statistic),
        "central_90_percent_spearman_p_value": float(central_spearman.pvalue),
    }


def atom_count_analysis(records: Sequence[dict]) -> dict:
    selected = []
    missing_formula = []
    for record in records:
        if record["viscosity_mPa_s"] < 0.10:
            continue
        counts = parse_formula_counts(record.get("formula") or "")
        if not counts:
            missing_formula.append({
                "cas": record["cas"],
                "name": record["name"],
                "formula": record.get("formula"),
            })
            continue
        derived = dict(counts)
        derived["heavy_atoms"] = sum(
            count for element, count in counts.items() if element != "H"
        )
        derived["heteroatoms"] = sum(
            count for element, count in counts.items() if element not in {"C", "H"}
        )
        derived["halogens"] = sum(
            counts.get(element, 0) for element in ("F", "Cl", "Br", "I")
        )
        selected.append((record, derived))

    viscosity = np.array(
        [record["viscosity_mPa_s"] for record, _ in selected],
        dtype=float,
    )
    count_names = ("C", "H", "O", "N", "S", "halogens", "heavy_atoms", "heteroatoms")
    correlations = {}
    for name in count_names:
        values = np.array([counts.get(name, 0) for _, counts in selected], dtype=float)
        pearson = stats.pearsonr(values, viscosity)
        spearman = stats.spearmanr(values, viscosity)
        correlations[name] = {
            "present_count": int(np.count_nonzero(values)),
            "maximum_count": int(np.max(values)),
            "pearson_r": float(pearson.statistic),
            "pearson_p_value": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p_value": float(spearman.pvalue),
        }

    bucket_definitions = {
        "C": (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4),
              ("5-6", 5, 6), ("7-8", 7, 8), ("9-12", 9, 12), ("13+", 13, math.inf)),
        "O": (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3+", 3, math.inf)),
        "N": (("0", 0, 0), ("1", 1, 1), ("2+", 2, math.inf)),
        "S": (("0", 0, 0), ("1+", 1, math.inf)),
        "halogens": (("0", 0, 0), ("1", 1, 1), ("2", 2, 2),
                     ("3+", 3, math.inf)),
        "heavy_atoms": (("1-2", 1, 2), ("3-4", 3, 4), ("5-6", 5, 6),
                        ("7-8", 7, 8), ("9-12", 9, 12), ("13+", 13, math.inf)),
    }
    buckets = {}
    for name, definitions in bucket_definitions.items():
        bucket_rows = []
        for label, lower, upper in definitions:
            values = [
                record["viscosity_mPa_s"]
                for record, counts in selected
                if lower <= counts.get(name, 0) <= upper
            ]
            if values:
                bucket_rows.append({
                    "label": label,
                    "count": len(values),
                    "mean_mPa_s": statistics.fmean(values),
                    "p25_mPa_s": percentile(values, 0.25),
                    "median_mPa_s": percentile(values, 0.50),
                    "p75_mPa_s": percentile(values, 0.75),
                    "p90_mPa_s": percentile(values, 0.90),
                })
        buckets[name] = bucket_rows

    feature_names = ("C", "O", "N", "S", "halogens")
    design = np.array(
        [[1.0, *(counts.get(name, 0) for name in feature_names)] for _, counts in selected],
        dtype=float,
    )
    coefficients = np.linalg.lstsq(design, viscosity, rcond=None)[0]
    fitted = design @ coefficients
    residuals = viscosity - fitted
    total_sum_squares = float(np.sum((viscosity - np.mean(viscosity)) ** 2))
    hat_diagonal = np.diag(design @ np.linalg.pinv(design.T @ design) @ design.T)
    leave_one_out_residuals = residuals / (1.0 - hat_diagonal)
    model = {
        "features": list(feature_names),
        "coefficients_mPa_s_per_atom": {
            "intercept": float(coefficients[0]),
            **{
                name: float(value)
                for name, value in zip(feature_names, coefficients[1:])
            },
        },
        "training_r_squared": 1.0 - float(np.sum(residuals**2)) / total_sum_squares,
        "training_rmse_mPa_s": math.sqrt(float(np.mean(residuals**2))),
        "leave_one_out_r_squared": (
            1.0 - float(np.sum(leave_one_out_residuals**2)) / total_sum_squares
        ),
        "leave_one_out_rmse_mPa_s": math.sqrt(
            float(np.mean(leave_one_out_residuals**2))
        ),
    }

    return {
        "count": len(selected),
        "missing_or_nonmolecular_formula": missing_formula,
        "correlations": correlations,
        "buckets": buckets,
        "common_atom_linear_model": model,
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(args: argparse.Namespace) -> dict:
    library = PerryPropertyLibrary(path=args.database)
    library._load()
    records = []

    for cas, entry in sorted(library.chemicals.items()):
        for correlation_index, correlation in enumerate(
            entry.get("liquid_viscosity") or []
        ):
            critical = entry.get("critical_constants") or {}
            base = {
                "cas": cas,
                "name": entry.get("name"),
                "formula": entry.get("formula"),
                "correlation_index": correlation_index,
                "equation_id": correlation.get("equation_id"),
                "source_table": correlation.get("source_table"),
                "T_min_K": correlation.get("T_min_K"),
                "T_max_K": correlation.get("T_max_K"),
                "coefficients": correlation.get("coefficients"),
                "acentric_factor": critical.get("omega"),
            }
            boiling = library.normal_boiling_point_K(cas)
            if boiling is None:
                table_2_10 = library.table_2_10_normal_boiling_point_K(cas)
                records.append({
                    **base,
                    "status": "normal_boiling_point_unavailable",
                    "Tb_K": None,
                    "table_2_10_transition_K": (
                        float(table_2_10.value) if table_2_10 else None
                    ),
                })
                continue

            Tb_K = float(boiling.value)
            value = library._eval_liquid_viscosity_Pa_s(correlation, Tb_K)
            if value is None or value <= 0.0 or not math.isfinite(value):
                records.append({
                    **base,
                    "status": "invalid_correlation_evaluation",
                    "Tb_K": Tb_K,
                    "Tb_method": boiling.method,
                })
                continue

            T_min = float(correlation["T_min_K"])
            T_max = float(correlation["T_max_K"])
            if Tb_K < T_min:
                range_position = "below"
                range_offset_K = T_min - Tb_K
            elif Tb_K > T_max:
                range_position = "above"
                range_offset_K = Tb_K - T_max
            else:
                range_position = "in_range"
                range_offset_K = 0.0
            records.append({
                **base,
                "status": "evaluated",
                "Tb_K": Tb_K,
                "Tb_method": boiling.method,
                "Tb_source": boiling.source,
                "range_position": range_position,
                "range_offset_K": range_offset_K,
                "within_0_5_K_of_range": range_offset_K <= 0.5,
                "within_1_K_of_range": range_offset_K <= 1.0,
                "viscosity_Pa_s": float(value),
                "viscosity_mPa_s": float(value) * 1000.0,
            })

    evaluated = [record for record in records if record["status"] == "evaluated"]
    strict = [record for record in evaluated if record["range_position"] == "in_range"]
    within_half_K = [record for record in evaluated if record["within_0_5_K_of_range"]]
    within_one_K = [record for record in evaluated if record["within_1_K_of_range"]]
    missing = [
        record for record in records
        if record["status"] == "normal_boiling_point_unavailable"
    ]
    invalid = [
        record for record in records
        if record["status"] == "invalid_correlation_evaluation"
    ]

    def summarize(subset: Sequence[dict]) -> dict:
        return value_summary([record["viscosity_mPa_s"] for record in subset])

    bands = Counter(viscosity_band(record["viscosity_mPa_s"]) for record in evaluated)
    range_positions = Counter(record["range_position"] for record in evaluated)
    equation_ids = Counter(record["equation_id"] for record in records)
    meaningful_extrapolations = sorted(
        (
            record for record in evaluated
            if record["range_offset_K"] > 1.0
        ),
        key=lambda record: record["range_offset_K"],
        reverse=True,
    )
    by_viscosity = sorted(evaluated, key=lambda record: record["viscosity_mPa_s"])

    return {
        "benchmark": "perry_liquid_viscosity_evaluated_at_normal_boiling_point",
        "database": str(args.database),
        "database_sha256": sha256_path(args.database),
        "normal_boiling_pressure_bar": 1.01325,
        "units": {
            "reported_viscosity": "mPa*s (numerically equal to cP)",
            "stored_viscosity": "Pa*s",
        },
        "coverage": {
            "perry_chemical_records": len(library.chemicals),
            "liquid_viscosity_correlations": len(records),
            "evaluated_at_Tb": len(evaluated),
            "normal_boiling_point_unavailable": len(missing),
            "invalid_evaluations": len(invalid),
            "strictly_in_correlation_range": len(strict),
            "within_0_5_K_of_correlation_range": len(within_half_K),
            "within_1_K_of_correlation_range": len(within_one_K),
        },
        "equation_id_counts": dict(sorted(equation_ids.items())),
        "range_position_counts": dict(sorted(range_positions.items())),
        "viscosity_band_counts_mPa_s": dict(sorted(bands.items())),
        "statistics_mPa_s": {
            "all_evaluated": summarize(evaluated),
            "strictly_in_range": summarize(strict),
            "within_0_5_K_of_range": summarize(within_half_K),
            "within_1_K_of_range": summarize(within_one_K),
        },
        "acentric_factor_correlation_excluding_below_0_10_mPa_s": (
            acentric_correlation(evaluated)
        ),
        "atom_count_analysis_excluding_below_0_10_mPa_s": atom_count_analysis(evaluated),
        "lowest": by_viscosity[: args.top],
        "highest": list(reversed(by_viscosity[-args.top :])),
        "meaningful_extrapolations_over_1_K": meaningful_extrapolations,
        "normal_boiling_point_unavailable": missing,
        "invalid_evaluations": invalid,
        "records": records,
    }


def format_statistics(summary: dict) -> str:
    return (
        f"n={summary['count']:3d} min={summary['minimum']:.6g} "
        f"P05={summary['p05']:.6g} P25={summary['p25']:.6g} "
        f"median={summary['median']:.6g} mean={summary['mean']:.6g} "
        f"gmean={summary['geometric_mean']:.6g} P75={summary['p75']:.6g} "
        f"P95={summary['p95']:.6g} max={summary['maximum']:.6g}"
    )


def build_report(payload: dict) -> str:
    coverage = payload["coverage"]
    statistics_payload = payload["statistics_mPa_s"]
    correlation = payload[
        "acentric_factor_correlation_excluding_below_0_10_mPa_s"
    ]
    atom_analysis = payload["atom_count_analysis_excluding_below_0_10_mPa_s"]
    lines = [
        "Perry liquid-viscosity correlations evaluated at normal boiling point",
        "viscosity units: mPa*s (numerically equal to cP)",
        "normal boiling pressure: 1.01325 bar",
        "",
        "COVERAGE",
        f"  Perry chemical records: {coverage['perry_chemical_records']}",
        f"  liquid-viscosity correlations: {coverage['liquid_viscosity_correlations']}",
        f"  evaluated at Tb: {coverage['evaluated_at_Tb']}",
        f"  Tb unavailable: {coverage['normal_boiling_point_unavailable']}",
        f"  invalid evaluations: {coverage['invalid_evaluations']}",
        f"  strictly in correlation range: {coverage['strictly_in_correlation_range']}",
        (
            "  within 0.5 K of correlation range: "
            f"{coverage['within_0_5_K_of_correlation_range']}"
        ),
        f"  within 1 K of correlation range: {coverage['within_1_K_of_correlation_range']}",
        f"  equation IDs: {payload['equation_id_counts']}",
        f"  range positions: {payload['range_position_counts']}",
        "",
        "VISCOSITY DISTRIBUTIONS",
        f"  all evaluated:      {format_statistics(statistics_payload['all_evaluated'])}",
        f"  strict in-range:    {format_statistics(statistics_payload['strictly_in_range'])}",
        f"  within 0.5 K:       {format_statistics(statistics_payload['within_0_5_K_of_range'])}",
        f"  within 1 K:         {format_statistics(statistics_payload['within_1_K_of_range'])}",
        "",
        f"VISCOSITY BANDS: {payload['viscosity_band_counts_mPa_s']}",
        "",
        "ACENTRIC-FACTOR CORRELATION (mu >= 0.10 mPa*s)",
        (
            f"  n={correlation['count']}; Pearson r={correlation['pearson_r']:.6f} "
            f"(p={correlation['pearson_p_value']:.4g}); "
            f"Spearman rho={correlation['spearman_rho']:.6f} "
            f"(p={correlation['spearman_p_value']:.4g})"
        ),
        (
            "  linear: mu[mPa*s] = "
            f"{correlation['linear_fit_intercept_mPa_s']:.6f} + "
            f"{correlation['linear_fit_slope_mPa_s_per_omega']:.6f} omega; "
            f"R^2={correlation['linear_fit_r_squared']:.6f}; "
            f"RMSE={correlation['linear_fit_rmse_mPa_s']:.6f} mPa*s"
        ),
        (
            "  log-linear: ln(mu[mPa*s]) = "
            f"{correlation['log_linear_fit_intercept']:.6f} + "
            f"{correlation['log_linear_fit_slope_per_omega']:.6f} omega; "
            f"R^2={correlation['log_linear_fit_r_squared']:.6f}"
        ),
        (
            f"  central 90%: n={correlation['central_90_percent_count']}; "
            f"Pearson r={correlation['central_90_percent_pearson_r']:.6f}; "
            f"Spearman rho={correlation['central_90_percent_spearman_rho']:.6f}"
        ),
        "",
        "ATOM-COUNT CORRELATIONS (mu >= 0.10 mPa*s)",
    ]
    for name in ("C", "H", "O", "N", "S", "halogens", "heavy_atoms", "heteroatoms"):
        item = atom_analysis["correlations"][name]
        lines.append(
            f"  {name:<12} present={item['present_count']:3d} "
            f"Pearson r={item['pearson_r']:+.4f} "
            f"Spearman rho={item['spearman_rho']:+.4f}"
        )
    model = atom_analysis["common_atom_linear_model"]
    lines.extend([
        (
            "  common-atom OLS (C,O,N,S,total halogens): "
            f"training R^2={model['training_r_squared']:.4f}, "
            f"LOO R^2={model['leave_one_out_r_squared']:.4f}, "
            f"LOO RMSE={model['leave_one_out_rmse_mPa_s']:.5f} mPa*s"
        ),
        "",
        "ATOM-COUNT BUCKETS (n, median, mean in mPa*s)",
    ])
    for name in ("C", "O", "halogens", "heavy_atoms"):
        lines.append(f"  {name}")
        for bucket in atom_analysis["buckets"][name]:
            lines.append(
                f"    {bucket['label']:<5} n={bucket['count']:3d} "
                f"median={bucket['median_mPa_s']:.6f} "
                f"mean={bucket['mean_mPa_s']:.6f}"
            )
    lines.extend([
        "",
        "LOWEST VALUES",
    ])
    for record in payload["lowest"]:
        lines.append(
            f"  {record['name']:<30} {record['viscosity_mPa_s']:10.6g} "
            f"at {record['Tb_K']:.3f} K ({record['range_position']}, "
            f"offset {record['range_offset_K']:.3g} K)"
        )
    lines.extend(["", "HIGHEST VALUES"])
    for record in payload["highest"]:
        lines.append(
            f"  {record['name']:<30} {record['viscosity_mPa_s']:10.6g} "
            f"at {record['Tb_K']:.3f} K ({record['range_position']}, "
            f"offset {record['range_offset_K']:.3g} K)"
        )
    lines.extend(["", "EXTRAPOLATIONS OVER 1 K"])
    for record in payload["meaningful_extrapolations_over_1_K"]:
        lines.append(
            f"  {record['name']:<30} offset={record['range_offset_K']:.3f} K "
            f"mu={record['viscosity_mPa_s']:.6g} mPa*s"
        )
    lines.extend(["", "NORMAL BOILING POINT UNAVAILABLE"])
    for record in payload["normal_boiling_point_unavailable"]:
        transition = record.get("table_2_10_transition_K")
        transition_note = (
            f"; Table 2-10 1-atm transition={transition:.2f} K"
            if transition is not None else ""
        )
        lines.append(f"  {record['name']} ({record['cas']}){transition_note}")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, records: Sequence[dict]) -> None:
    fields = (
        "cas", "name", "formula", "acentric_factor", "equation_id", "status", "Tb_K",
        "T_min_K", "T_max_K", "range_position", "range_offset_K",
        "viscosity_Pa_s", "viscosity_mPa_s",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "outputs" / "perry_liquid_viscosity_at_tb.json",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=ROOT / "outputs" / "perry_liquid_viscosity_at_tb.txt",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=ROOT / "outputs" / "perry_liquid_viscosity_at_tb.csv",
    )
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    report = build_report(payload)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.text_output.write_text(report)
    write_csv(args.csv_output, payload["records"])
    print(report, end="")


if __name__ == "__main__":
    main()
