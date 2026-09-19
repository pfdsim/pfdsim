#!/usr/bin/env python3
"""Benchmark direct VLLE azeotropic initialization and profile fallback."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe_azeotropic_initializer_matrix import prepare_case


DIFFICULT_CASES = (
    "example_ethanol_benzene_entrainer",
    "example_lactic_recovery",
    "test_butanol_077",
    "test_butanol_0773",
    "test_acrylic_056",
    "test_acrylic_water_limit",
    "test_acrylic_watered_9995",
)

REPRESENTATIVE_CASES = (
    "example_3mp_polisher",
    "test_nrtl_rk_total6",
    "test_nrtl_rk_total12",
    "test_nrtl_rk_total",
    "test_nrtl_rk_mixed",
    "test_nrtl_rk_partial",
    "test_nrtl_rk_mass",
    "test_nrtl_rk_multifeed",
    "test_unifac_homogeneous",
    "test_unifac_appearance",
    "test_unifac_disappearance",
    "test_binary_chloroform",
)

CASES = DIFFICULT_CASES + REPRESENTATIVE_CASES
MODES = ("baseline", "linear", "log_feed_anchor", "auto")


def worker(case: str, mode: str) -> None:
    column, inlets = prepare_case(case)
    if mode != "baseline":
        column.params["vlle_seed"] = "azeotropic"
        column.params["vlle_azeotropic_profile"] = mode
        column.params["max_iterations"] = max(
            180, int(column.params.get("max_iterations", 60))
        )
        column.params["max_jacobian_evaluations"] = max(
            180, int(column.params.get("max_jacobian_evaluations", 60))
        )
    started = time.perf_counter()
    try:
        result = column.solve(inlets)
    except Exception as exc:
        summary = {
            "case": case,
            "mode": mode,
            "success": False,
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        performance = result.performance
        summary = {
            "case": case,
            "mode": mode,
            "success": True,
            "elapsed_seconds": time.perf_counter() - started,
            "initializer": performance["initializer"],
            "jacobian_evaluations": performance["jacobian_evaluations"],
            "solver_iterations": performance["solver_iterations"],
            "function_evaluations": performance["function_evaluations"],
            "solver_work_basis": performance["solver_work_basis"],
            "mesh_residual": performance["mesh_residual"],
            "topology": performance["vlle_topology"],
            "candidate_source": performance.get(
                "vlle_azeotropic_candidate_source"
            ),
            "candidate_search_seconds": performance.get(
                "vlle_azeotropic_search_seconds", 0.0
            ),
            "candidates": performance.get("vlle_azeotropic_candidates", []),
            "attempts": performance.get("vlle_initializer_attempts", []),
            "distillate": result.outlet_streams["distillate"].composition,
            "bottoms": result.outlet_streams["bottoms"].composition,
        }
    print("BENCHMARK_JSON " + json.dumps(summary, sort_keys=True))


def run_worker(case: str, mode: str) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__)),
            "--worker",
            case,
            "--mode",
            mode,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    marker = next(
        (
            line for line in completed.stdout.splitlines()
            if line.startswith("BENCHMARK_JSON ")
        ),
        None,
    )
    if marker is not None:
        return marker
    return "BENCHMARK_JSON " + json.dumps({
        "case": case,
        "mode": mode,
        "success": False,
        "returncode": completed.returncode,
        "error": completed.stderr or completed.stdout,
    }, sort_keys=True)


def run_all(cases, modes, processes: int) -> None:
    with ThreadPoolExecutor(max_workers=processes) as executor:
        futures = [
            executor.submit(run_worker, case, mode)
            for case in cases
            for mode in modes
        ]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--mode", choices=MODES, default="auto")
    parser.add_argument(
        "--suite",
        choices=("difficult", "representative", "all"),
        default="difficult",
    )
    parser.add_argument("--all-modes", action="store_true")
    parser.add_argument("--processes", type=int, default=5)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.mode)
        return
    cases = {
        "difficult": DIFFICULT_CASES,
        "representative": REPRESENTATIVE_CASES,
        "all": CASES,
    }[args.suite]
    modes = MODES if args.all_modes else (args.mode,)
    run_all(cases, modes, max(1, args.processes))


if __name__ == "__main__":
    main()
