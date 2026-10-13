"""Paired binary-level bootstrap for the saved raw-VLE UNIFAC comparison.

Resample binary pairs, never individual points, preserving equal pair weights
and dependence between the old/new methods. These are exploratory intervals
conditional on the studied convenience sample, not proof of general chemical
space performance. No equilibrium or fitting calculations are repeated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SEED = 20261012
REPLICATES = 20000
METRICS = (
    ("Pressure AARD", "pressure_AARD_percent", False, "percentage points"),
    ("Temperature RMS", "T_MSE", True, "K"),
    ("Vapor composition RMS", "y_MSE", True, "mole fraction"),
    ("ln(gamma) RMS", "ln_gamma_MSE", True, "dimensionless"),
)


def statistic(values, square_root):
    value = np.mean(values, axis=-1)
    return np.sqrt(value) if square_root else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pair_path = args.results_dir / "pairs.csv"
    rows = list(csv.DictReader(pair_path.open()))
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["pair"]] = row
    rng = np.random.default_rng(SEED)
    comparisons = []
    for old, new in (("UNIFAC", "UNIFAC2"), ("UNIFDMD", "UNIFM2")):
        for label, field, square_root, unit in METRICS:
            pairs = sorted(p for p in by_method[old].keys() & by_method[new].keys()
                           if by_method[old][p].get(field) and by_method[new][p].get(field))
            before = np.array([float(by_method[old][p][field]) for p in pairs])
            after = np.array([float(by_method[new][p][field]) for p in pairs])
            old_score, new_score = float(statistic(before, square_root)), float(statistic(after, square_root))
            sampled_differences, sampled_improvements = [], []
            for count in range(0, REPLICATES, 2000):
                indices = rng.integers(0, len(pairs), size=(min(2000, REPLICATES-count), len(pairs)))
                old_sample = statistic(before[indices], square_root)
                new_sample = statistic(after[indices], square_root)
                sampled_differences.extend((new_sample-old_sample).tolist())
                sampled_improvements.extend((100*(1-new_sample/old_sample)).tolist())
            interval = np.quantile(sampled_differences, [.025, .975]).tolist()
            adjusted = np.quantile(sampled_differences, [.05/(2*8), 1-.05/(2*8)]).tolist()
            comparisons.append({
                "old_method": old, "new_method": new, "metric": label, "unit": unit,
                "pairs": len(pairs), "old_score": old_score, "new_score": new_score,
                "difference_new_minus_old": new_score-old_score,
                "paired_percentile_95_CI_difference": interval,
                "relative_error_reduction_percent": 100*(1-new_score/old_score),
                "paired_percentile_95_CI_relative_reduction_percent": np.quantile(sampled_improvements, [.025, .975]).tolist(),
                "CI_excludes_zero": interval[1] < 0 or interval[0] > 0,
                "Bonferroni_familywise_95_CI_difference_8_comparisons": adjusted,
                "adjusted_CI_excludes_zero": adjusted[1] < 0 or adjusted[0] > 0,
                "new_method_pair_wins": int(np.sum(after < before)),
                "pair_ids": pairs,
            })
    ethanol_water = []
    for method, records in by_method.items():
        row = records["64-17-5 / 7732-18-5"]
        ethanol_water.append({"method": method, "points": int(row["points"]),
                              "pressure_AARD_percent": float(row["pressure_AARD_percent"]),
                              "temperature_RMS_K": math.sqrt(float(row["T_MSE"])),
                              "vapor_composition_RMS": math.sqrt(float(row["y_MSE"])),
                              "ln_gamma_MAE": float(row["ln_gamma_MAE"])})
    summary = {"seed": SEED, "bootstrap_replicates": REPLICATES,
               "resampling_unit": "distinct binary CAS pair; paired old/new resampling; equal weight per pair",
               "comparisons": comparisons, "ethanol_water": ethanol_water,
               "source_pairs_sha256": hashlib.sha256(pair_path.read_bytes()).hexdigest(),
               "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "limitations": "Exploratory paired percentile intervals; both unadjusted 95% and conservative Bonferroni familywise 95% intervals across eight model/metric comparisons are reported. Pairs share components and are not a random chemical-space sample. Point uncertainty, vapor/Psat uncertainty, and ML training overlap are not propagated. Bootstrap independence across binary pairs is an approximation."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    lines = ["# Paired raw-VLE uncertainty check", "",
             "20,000 seeded bootstrap replicates resample binary pairs with equal pair weights. Difference = new error minus old error; negative favors 2.0.", "",
             "| Comparison | Metric | Pairs | Old | New | Difference | 95% CI of difference |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        lo, hi = row["paired_percentile_95_CI_difference"]
        lines.append(f"| {row['old_method']} → {row['new_method']} | {row['metric']} ({row['unit']}) | {row['pairs']} | {row['old_score']:.5f} | {row['new_score']:.5f} | {row['difference_new_minus_old']:.5f} | [{lo:.5f}, {hi:.5f}] |")
    lines += ["", "## Adjustment across eight comparisons", "", "| Comparison | Metric | Familywise 95% CI of difference |", "|---|---|---:|"]
    for row in comparisons:
        lo, hi = row["Bonferroni_familywise_95_CI_difference_8_comparisons"]
        lines.append(f"| {row['old_method']} → {row['new_method']} | {row['metric']} | [{lo:.5f}, {hi:.5f}] |")
    lines += ["", "## Ethanol/water", "", "| Method | Points | Pressure AARD % | T RMS K | y RMS | ln(gamma) MAE |", "|---|---:|---:|---:|---:|---:|"]
    for row in ethanol_water:
        lines.append(f"| {row['method']} | {row['points']} | {row['pressure_AARD_percent']:.4f} | {row['temperature_RMS_K']:.4f} | {row['vapor_composition_RMS']:.5f} | {row['ln_gamma_MAE']:.5f} |")
    lines += ["", summary["limitations"], ""]
    (output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
