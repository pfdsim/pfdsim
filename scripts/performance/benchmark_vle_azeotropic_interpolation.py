#!/usr/bin/env python3
"""Compare VLE azeotropic composition-profile interpolation strategies."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe_azeotropic_initializer_matrix import prepare_case
from unit_operations_distillation import RigorousDistillation


CASES = (
    "example_ethanol_water",
    "example_pressure_swing_10bar",
    "example_pressure_swing_1bar",
    "example_mixed_acids",
    "example_ethanol_benzene_beer",
    "test_nitrile",
)
STRATEGIES = ("linear", "log_linear", "log_feed_anchor")


def worker(case: str, strategy: str) -> None:
    column, inlets = prepare_case(case)
    column.params["initializer"] = "azeotropic"
    original_initial = RigorousDistillation._initial_guess

    def transformed_initial(unit, *args, **kwargs):
        initial = original_initial(unit, *args, **kwargs)
        if unit is not column or strategy == "linear":
            return initial

        _inlet, comps, feed_z, stages, feed_stage = args[:5]
        pressures = args[7]
        T_min, T_max = args[12:14]
        top = dict(initial["x"][0])
        bottom = dict(initial["x"][-1])
        feed_index = min(max(int(feed_stage) - 1, 1), stages - 2)

        def geometric(left, right, fraction):
            return unit._normalize({
                comp: math.exp(
                    (1.0 - fraction)
                    * math.log(max(left.get(comp, 0.0), 1e-12))
                    + fraction
                    * math.log(max(right.get(comp, 0.0), 1e-12))
                )
                for comp in comps
            })

        for stage in range(stages):
            if strategy == "log_feed_anchor" and stage <= feed_index:
                fraction = stage / feed_index
                left, right = top, feed_z
            elif strategy == "log_feed_anchor":
                fraction = (stage - feed_index) / (stages - 1 - feed_index)
                left, right = feed_z, bottom
            else:
                fraction = stage / max(stages - 1, 1)
                left, right = top, bottom
            composition = geometric(left, right, fraction)
            initial["x"][stage] = composition
            initial["T"][stage] = unit._bubble_temperature_from_equation(
                composition,
                pressures[stage],
                T_min,
                T_max,
            )
        return initial

    started = time.perf_counter()
    with patch.object(
        RigorousDistillation, "_initial_guess", transformed_initial
    ):
        try:
            result = column.solve(inlets)
        except Exception as exc:
            summary = {
                "case": case,
                "strategy": strategy,
                "success": False,
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            performance = result.performance
            summary = {
                "case": case,
                "strategy": strategy,
                "success": True,
                "elapsed_seconds": time.perf_counter() - started,
                "jacobian_evaluations": performance["jacobian_evaluations"],
                "mesh_residual": performance["mesh_residual"],
                "distillate": result.outlet_streams["distillate"].composition,
                "bottoms": result.outlet_streams["bottoms"].composition,
            }
    print("BENCHMARK_JSON " + json.dumps(summary, sort_keys=True))


def run_worker(case: str, strategy: str) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__)),
            "--worker",
            case,
            "--strategy",
            strategy,
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
        "strategy": strategy,
        "success": False,
        "returncode": completed.returncode,
        "error": completed.stderr or completed.stdout,
    }, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--strategy", choices=STRATEGIES, default="linear")
    parser.add_argument("--all-strategies", action="store_true")
    parser.add_argument("--processes", type=int, default=5)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.strategy)
        return
    strategies = STRATEGIES if args.all_strategies else (args.strategy,)
    with ThreadPoolExecutor(max_workers=max(1, args.processes)) as executor:
        futures = [
            executor.submit(run_worker, case, strategy)
            for case in CASES
            for strategy in strategies
        ]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
