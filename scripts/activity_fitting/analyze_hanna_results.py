"""Verify and bootstrap saved HANNA comparisons without rerunning predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 20261013
REPLICATES = 20000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.results_dir/"summary.json").read_text())
    with (args.results_dir/"points.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    keys = defaultdict(set)
    for r in rows:
        if r["status"] != "ok":
            continue
        grouped[r["cohort"], r["method"]].append(r)
        keys[r["cohort"], r["method"]].add((r["system"], r["point_id"]))
        predicted, inferred, errors = (json.loads(r[k] or "[]") for k in ("predicted_ln_gamma", "inferred_ln_gamma", "errors"))
        expected = [p-i for p, i in zip(predicted, inferred) if i is not None]
        np.testing.assert_allclose(errors, expected, atol=1e-13, rtol=1e-12)
    per_system = {}
    for cohort, methods in summary["scores"].items():
        common = set.intersection(*(keys[cohort, m] for m in methods))
        for method, scores in methods.items():
            systems = defaultdict(list)
            for r in grouped[cohort, method]:
                if (r["system"], r["point_id"]) in common:
                    systems[r["system"]].append(r)
            assert len(systems) == scores["systems"]
            assert sum(map(len, systems.values())) == scores["states"]
            values, squares, closures = {}, {}, {}
            for name, rr in systems.items():
                e = [x for r in rr for x in json.loads(r["errors"])]
                if e:
                    values[name], squares[name] = float(np.mean(np.abs(e))), float(np.mean(np.square(e)))
                c = [float(r["closure"]) for r in rr if r.get("closure")]
                if c:
                    closures[name] = float(np.mean(np.abs(c)))*100
            np.testing.assert_allclose(np.mean(list(values.values())), scores["MAE"], rtol=1e-12)
            np.testing.assert_allclose(np.sqrt(np.mean(list(squares.values()))), scores["RMS"], rtol=1e-12)
            if closures:
                np.testing.assert_allclose(np.mean(list(closures.values())), scores["mean_system_bubble_equation_AARD_percent"], rtol=1e-12)
            per_system[cohort, method, "ln_gamma_MAE"] = values
            per_system[cohort, method, "bubble_equation_AARD_percent"] = closures
    rng = np.random.default_rng(SEED)
    comparisons = []
    for cohort, metric in (("raw_binary", "ln_gamma_MAE"), ("raw_binary", "bubble_equation_AARD_percent"), ("close_fits", "ln_gamma_MAE")):
        for competitor in summary["scores"][cohort]:
            if competitor == "HANNA":
                continue
            a, b = per_system[cohort, "HANNA", metric], per_system[cohort, competitor, metric]
            names = sorted(a.keys() & b.keys())
            difference = np.array([a[n]-b[n] for n in names])
            indices = rng.integers(len(names), size=(REPLICATES, len(names)))
            distribution = difference[indices].mean(axis=1)
            comparisons.append({"cohort": cohort, "metric": metric, "competitor": competitor, "systems": len(names),
                                "HANNA_minus_competitor": float(difference.mean()),
                                "95_CI": np.quantile(distribution, [.025, .975]).tolist(),
                                "familywise_95_CI_15_comparisons": np.quantile(distribution, [.05/30, 1-.05/30]).tolist(),
                                "HANNA_system_wins": int(np.sum(difference < 0))})
    result = {"comparisons": comparisons, "seed": SEED, "replicates": REPLICATES,
              "verification": "All point errors and matched-cohort counts/MAE/RMS/closure statistics independently verified",
              "points_sha256": hashlib.sha256((args.results_dir/"points.csv").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limitations": "Exploratory pair-level resampling; shared components, unknown training overlap, and reference uncertainty are not propagated. Fifteen comparisons; both unadjusted and conservative Bonferroni intervals reported."}
    (args.output_dir/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(result["verification"])
    for r in comparisons:
        print(r["cohort"], r["metric"], r["competitor"], r["systems"], "delta", round(r["HANNA_minus_competitor"], 6), "CI", [round(x, 6) for x in r["95_CI"]], "wins", r["HANNA_system_wins"])


if __name__ == "__main__":
    main()
