#!/usr/bin/env python3
"""Analyze recorded column trust-region runs without rerunning any solves.

Example:
  python scripts/performance/analyze_column_trust_region.py \
      --stress /tmp/pfdsim-trust-stress-final-20261006 \
      --end-to-end /tmp/pfdsim-trust-end-final-20261006 --warmup-repeats 1

Prints Markdown. Redirect it to a new report file to retain the analysis.
Paired speed comparisons include only starts on which both algorithms reach
the requested residual. Failed solves never count as speed improvements.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

METHODS = ("line", "dogleg", "dogleg_scaled", "subspace", "trf", "trf_unscaled")


def read(directory):
    return [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]


def key(row):
    return row["case"], row["noise"], row["seed"], row["repeat"]


def stress_report(records):
    methods = tuple(m for m in METHODS if any(r["method"] == m for r in records))
    index = {(key(r), r["method"]): r for r in records}
    # Every paired algorithm must receive exactly the same transformed start.
    for identity in dict.fromkeys(key(r) for r in records):
        hashes = {r["attempts"][0]["x0_sha256"] for r in records if key(r) == identity}
        assert len(hashes) == 1, f"unmatched starts: {identity}"
    print("## Perturbed initial guesses\n")
    print("| Method | Strict | Acceptable | Moderate noise strict | Severe noise strict | Rescues vs line | Losses vs line | Median paired time / line |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        if not rows:
            continue
        gains = losses = 0
        ratios = []
        for r in rows:
            baseline = index[key(r), "line"]
            gains += r["strict_success"] and not baseline["strict_success"]
            losses += baseline["strict_success"] and not r["strict_success"]
            if baseline["strict_success"] and r["strict_success"]:
                ratios.append(r["solver_seconds"] / baseline["solver_seconds"])
        def rate(noise):
            subset = [r for r in rows if r["noise"] == noise]
            return f"{sum(r['strict_success'] for r in subset)}/{len(subset)}"
        print(f"| {method} | {sum(r['strict_success'] for r in rows)}/{len(rows)} | "
              f"{sum(r['success'] for r in rows)}/{len(rows)} | {rate(.5)} | {rate(1.5)} | "
              f"{gains} | {losses} | {median(ratios):.3f} |" if ratios else
              f"| {method} | {sum(r['strict_success'] for r in rows)}/{len(rows)} | "
              f"{sum(r['success'] for r in rows)}/{len(rows)} | {rate(.5)} | {rate(1.5)} | {gains} | {losses} | — |")
    print("\n| Case | " + " | ".join(methods) + " |")
    print("|---|" + "---:|" * len(methods))
    for case in dict.fromkeys(r["case"] for r in records):
        counts = []
        for method in methods:
            group = [r for r in records if r["case"] == case and r["method"] == method]
            counts.append(f"{sum(r['strict_success'] for r in group)}/{len(group)}")
        print(f"| {case} | " + " | ".join(counts) + " |")
    combined = sum(any(index.get((identity, m), {}).get("strict_success", False)
                       for m in ("line", "dogleg", "subspace"))
                   for identity in dict.fromkeys(key(r) for r in records))
    print(f"\nUnion of observed line/dogleg/subspace strict successes: {combined}.")
    print("This is a union of separate experiments, not a measured hybrid solver.")


def end_report(records):
    methods = tuple(m for m in METHODS if any(r["method"] == m for r in records))
    print("\n## Normal initializers: median full unit solve time, seconds\n")
    print("Failed runs are marked explicitly; their time is excluded from successful timing medians.\n")
    print("| Case | " + " | ".join(methods) + " |")
    print("|---|" + "---:|" * len(methods))
    for case in dict.fromkeys(r["case"] for r in records):
        cells = []
        for method in methods:
            rows = [r for r in records if r["case"] == case and r["method"] == method]
            successes = [r for r in rows if r["success"]]
            if not rows:
                cells.append("—")
            elif not successes:
                cells.append(f"FAIL {len(rows)}/{len(rows)}")
            else:
                cell = f"{median(r['solve_seconds'] for r in successes):.4f}"
                if len(successes) != len(rows):
                    cell += f" ({len(successes)}/{len(rows)} succeeded)"
                if any(not r.get("strict_success", False) for r in successes):
                    cell += "†"
                cells.append(cell)
        print(f"| {case} | " + " | ".join(cells) + " |")
    print("\n† Accepted by the unit, but at least one run missed the requested residual tolerance.")
    print("\n| Method | Accepted / runs | Strict / runs | Max accepted residual | Max external component error | Max external energy error |")
    print("|---|---:|---:|---:|---:|---:|")
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        successes = [r for r in rows if r["success"]]
        if not rows:
            continue
        residuals = [r["residual_norm"] for r in successes if r.get("residual_norm") is not None]
        balances = [r["component_balance_error"] for r in successes]
        energies = [r["external_energy_relative_error"] for r in successes]
        assert all(math.isfinite(x) for x in residuals + balances + energies)
        print(f"| {method} | {len(successes)}/{len(rows)} | {sum(r.get('strict_success', False) for r in successes)}/{len(rows)} | "
              f"{max(residuals, default=math.nan):.3e} | {max(balances, default=math.nan):.3e} | {max(energies, default=math.nan):.3e} |")
    print("\n## Paired numerical differences and speed ratios\n")
    index = {(key(r), r["method"]): r for r in records}
    print("| Method | Accepted pairs | Max composition delta | Max temperature delta K | Max flow delta / feed | Strict timing pairs | Median strict paired time / line |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for method in methods[1:]:
        differences = [0., 0., 0.]
        ratios = []
        accepted_pairs = 0
        for r in records:
            if r["method"] != method or not r["success"]:
                continue
            baseline = index.get((key(r), "line"))
            if baseline is None or not baseline["success"]:
                continue
            assert r["inputs"] == baseline["inputs"], "paired inlets differ"
            assert r["outputs"].keys() == baseline["outputs"].keys(), "paired outlet ports differ"
            accepted_pairs += 1
            if r.get("strict_success", False) and baseline.get("strict_success", False):
                ratios.append(r["solve_seconds"] / baseline["solve_seconds"])
            feed_flow = max(sum(s["F"] for s in r["inputs"].values()), 1.)
            for port, stream in r["outputs"].items():
                other = baseline["outputs"][port]
                differences[1] = max(differences[1], abs(stream["T"] - other["T"]))
                differences[2] = max(differences[2], abs(stream["F"] - other["F"]) / feed_flow)
                for component in stream["composition"].keys() | other["composition"].keys():
                    differences[0] = max(differences[0], abs(
                        stream["composition"].get(component, 0.) - other["composition"].get(component, 0.)))
        ratio_cell = f"{median(ratios):.3f}" if ratios else "—"
        print(f"| {method} | {accepted_pairs} | " + " | ".join(f"{x:.3e}" for x in differences) +
              f" | {len(ratios)} | {ratio_cell} |")
    print("\n## Sum of per-case medians, restricted to complete strict case coverage\n")
    cases = tuple(dict.fromkeys(r["case"] for r in records))
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        if all(any(r["case"] == case for r in rows) for case in cases) and all(
            r["success"] and r.get("strict_success", False) for r in rows
        ):
            total = sum(median(r["solve_seconds"] for r in rows if r["case"] == case) for case in cases)
            print(f"- {method}: {total:.6f} seconds for {len(cases)} cases.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress", type=Path, nargs="+")
    parser.add_argument("--end-to-end", type=Path, nargs="+")
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--method-repeats", nargs="+", default=[], metavar="METHOD=1,2",
                        help="Override which complete repetitions to use for a method")
    args = parser.parse_args()
    method_repeats = {}
    for selection in args.method_repeats:
        method, repeats = selection.split("=", 1)
        method_repeats[method] = {int(value) for value in repeats.split(",")}
    if args.stress:
        stress_report([r for directory in args.stress for r in read(directory)])
    if args.end_to_end:
        end_report([r for directory in args.end_to_end for r in read(directory)
                    if (r["repeat"] in method_repeats[r["method"]] if r["method"] in method_repeats
                        else r["repeat"] > args.warmup_repeats)])


if __name__ == "__main__":
    main()
