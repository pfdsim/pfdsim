#!/usr/bin/env python3
"""Measure butanol/water fitting cost and residual quality on a fixed data basis.

Run with stdout/stderr persisted, for example:
python scripts/activity_fitting/benchmark_butanol_lle_fitting.py > /tmp/butanol-fit.log 2>&1
Initialization and optional kernel compilation are outside the measured solve.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from thermodynamics_models.interaction_fitting import FORMS, prepare_fit


def observations():
    # IUPAC SDS 15, pp. 34–35: recommended nominal mass percentages.
    table = [
        (273.15, 10.4, 80.3),
        (283.15, 8.9, 80.4),
        (293.15, 7.8, 80.0),
        (298.15, 7.4, 79.7),
        (303.15, 7.1, 79.4),
        (308.15, None, 78.9),
        (313.15, 6.6, 78.6),
        (323.15, None, 77.6),
        (333.15, 6.5, 76.3),
        (343.15, 6.7, 74.8),
        (348.15, 6.9, 73.7),
        (353.15, 7.0, 72.5),
        (358.15, 7.3, 71.2),
        (363.15, 7.7, 69.7),
        (368.15, 8.3, 68.0),
        (373.15, 9.1, 66.2),
        (378.15, 10.0, 63.9),
        (383.15, 11.1, 61.4),
        (388.15, 13.0, 57.5),
        (393.15, None, 52.6),
    ]
    rows = []
    for T, a, b in table:
        row = {"kind": "LLE", "T_K": T}
        for field, percent in (("x1_alpha", a), ("x1_beta", b)):
            if percent is not None:
                moles = percent / 74.1216
                row[field] = moles / (moles + (100 - percent) / 18.01528)
        rows.append(row)
    # A soft working critical target, not an IUPAC recommended observation.
    rows.append({"kind": "UCST", "T_K": 398.0, "x1": 0.107})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forms", nargs="+", choices=list(FORMS), default=["constant_inverse", "full"]
    )
    parser.add_argument(
        "--models", nargs="+", choices=["NRTL", "UNIQUAC"], default=["NRTL"]
    )
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument("--starts", type=int, default=1)
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Compare the previous scalar objective with batched runtime evaluation; do not optimize.",
    )
    args = parser.parse_args()
    results = []
    for model in args.models:
        for form in args.forms:
            initial = (
                {"12.constant": 2.5, "21.constant": 2.5}
                if model == "NRTL"
                else {"12.constant": -1.0, "21.constant": -1.0}
            )
            problem = prepare_fit(
                {
                    "components": ["1-butanol", "water"],
                    "model": model,
                    "form": form,
                    "observations": observations(),
                    "starts": args.starts,
                    "max_nfev": args.max_nfev,
                    "initial": initial,
                    "seed": 1729,
                }
            )
            rows = problem.request["observations"]
            problem.residuals(problem.initial, rows)
            if args.evaluation_only:
                batched_lle, batched_critical = (
                    problem._lle_errors,
                    problem.critical_derivatives,
                )

                def scalar_lle(T, a, b, sigma):
                    mua, mub = (
                        problem.chemical_potentials(T, a),
                        problem.chemical_potentials(T, b),
                    )
                    average = (mua + mub) / 2
                    grid = np.r_[np.linspace(0.00001, 0.99999, 41), a, b]
                    gaps = np.array(
                        [
                            problem.gibbs(T, z)
                            - (z * average[0] + (1 - z) * average[1])
                            for z in grid
                        ]
                    )
                    raw = (mua - mub).tolist()
                    return (
                        np.r_[
                            np.array(raw) / sigma,
                            np.minimum(gaps, 0) / sigma / math.sqrt(len(grid)),
                        ],
                        raw,
                        {
                            "log_activity_residuals": raw,
                            "minimum_tangent_gap": float(min(gaps)),
                        },
                    )

                def scalar_critical(T, x):
                    h = min(0.001, x / 4, (1 - x) / 4)
                    m = [
                        float(-np.diff(problem.chemical_potentials(T, x + i * h))[0])
                        for i in (-2, -1, 0, 1, 2)
                    ]
                    return (
                        (m[0] - 8 * m[1] + 8 * m[3] - m[4]) / (12 * h),
                        (-m[0] + 16 * m[1] - 30 * m[2] + 16 * m[3] - m[4])
                        / (12 * h * h),
                        (-m[0] + 2 * m[1] - 2 * m[3] + m[4]) / (2 * h**3),
                    )

                evaluations = {}
                for label, lle, critical in (
                    ("scalar", scalar_lle, scalar_critical),
                    ("batched", batched_lle, batched_critical),
                ):
                    problem._lle_errors, problem.critical_derivatives = lle, critical
                    started = time.perf_counter()
                    for index in range(20):
                        values = problem.initial.copy()
                        values[0] += (index + 1) * 1e-5
                        result = problem.residuals(values, rows)
                    evaluations[label] = {
                        "seconds": time.perf_counter() - started,
                        "residuals": result,
                    }
                error = float(
                    np.max(
                        np.abs(
                            evaluations["scalar"]["residuals"]
                            - evaluations["batched"]["residuals"]
                        )
                    )
                )
                entry = {
                    "model": model,
                    "form": form,
                    "scalar_seconds": evaluations["scalar"]["seconds"],
                    "batched_seconds": evaluations["batched"]["seconds"],
                    "max_residual_difference": error,
                    "evaluations": 20,
                }
                results.append(entry)
                print(json.dumps(entry), flush=True)
                continue
            count = 0
            residuals = problem.residuals

            def counted(*arguments, **keywords):
                nonlocal count
                count += 1
                return residuals(*arguments, **keywords)

            problem.residuals = counted
            started = time.perf_counter()
            try:
                fitted = problem.solve(rows)
            except ValueError as error:
                entry = {
                    "model": model,
                    "form": form,
                    "seconds": time.perf_counter() - started,
                    "residual_calls": count,
                    "error": str(error),
                    "max_nfev": args.max_nfev,
                }
                results.append(entry)
                print(json.dumps(entry), flush=True)
                continue
            seconds = time.perf_counter() - started
            report = problem.report(fitted["values"], rows)
            entry = {
                "model": model,
                "form": form,
                "seconds": seconds,
                "residual_calls": fitted["nfev"],
                "row_evaluations": fitted.get("row_evaluations"),
                "objective": fitted["objective"],
                "success": fitted["success"],
                "nfev": fitted["nfev"],
                "max_nfev": args.max_nfev,
                "starts": args.starts,
                "parameter_count": len(problem.names),
                "rank": fitted["rank"],
                "temperature_basis_condition": fitted.get(
                    "temperature_basis_condition"
                ),
                "conditioned_temperature_basis_condition": fitted.get(
                    "conditioned_temperature_basis_condition"
                ),
                "within_bounds": bool(
                    np.all(fitted["values"] >= problem.lower - 1e-8)
                    and np.all(fitted["values"] <= problem.upper + 1e-8)
                ),
                "physical_points": int(
                    sum(point["physical"] for point in report["points"])
                ),
                "physical_metrics": report["physical_metrics"],
                "coefficients": dict(
                    zip(problem.names, (fitted["values"] * problem.scales).tolist())
                ),
            }
            results.append(entry)
            print(json.dumps(entry), flush=True)
    print(json.dumps({"results": results}), flush=True)


if __name__ == "__main__":
    main()
