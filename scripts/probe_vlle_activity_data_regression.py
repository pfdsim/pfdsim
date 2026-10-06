#!/usr/bin/env python3
"""Isolate activity-data changes in the lactic azeotropic-seed regression.

Run with --baseline-ref pointing to a commit before the structural-data update.
The cases use current solver code, isolated interaction tables and estimator
caches, and the existing recovery-column benchmark's exact unperturbed feed.
No repository runtime table is changed. Full case logs/results stay under /tmp.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    "eos_binary_interactions_cas.json", "nrtl_binary_interactions_cas.json",
    "uniquac_binary_interactions_cas.json", "uniquac_rq_cas.json",
)


def required(mapping, key):
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"Missing required benchmark field: {key}")
    return value


def run_case(case):
    directory = Path(required(case, "directory"))
    code_root = Path(required(case, "code_root"))
    os.environ["PFDSIM_INTERACTION_DATA_DIR"] = str(directory / "data")
    sys.path.insert(0, str(code_root))
    with (directory / "worker.log").open("w", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            from scripts.benchmark_lactic_recovery_column import column_feed, result_summary
            from simulator import Simulator
            from thermodynamics_models import interaction_estimation
            from unit_operations_distillation import RigorousDistillation

            interaction_estimation._FIT_CACHE_PATH = directory / "estimation.sqlite"
            simulator = Simulator.from_file(code_root / "examples" / "lactic_acid_dehydration_pbr.pfd")
            start = time.perf_counter()
            simulator.initialize()
            thermo = required(simulator.thermo_packages, "global")
            if case.get("previous_propionic_r") is not None:
                if thermo.r.get("PA") != case.get("previous_propionic_r"):
                    raise ValueError("The previous propionic r control was not applied.")
            feed = column_feed(thermo, perturbed=False)
            params = dict(required(simulator.solver.units, "C-401").params)
            params.update(vlle_seed="azeotropic", vlle_projection_cycle_fallback=False)
            initialization_s = time.perf_counter() - start
            start = time.perf_counter()
            result = RigorousDistillation("VLLE-DATA-PROBE", thermo, params).solve({"feed": feed})
            report = result_summary(result, time.perf_counter() - start)
            # Persist the numerical result before optional diagnostic lookups.
            (directory / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            report.update(
                case=required(case, "name"), initialization_s=initialization_s,
                propionic_r=float(required(thermo.r, "PA")),
                structural_parameters={c: {"r":float(required(thermo.r, c)), "q":float(required(thermo.q, c))}
                                       for c in thermo.r if c in thermo.q},
                components_without_rq=sorted(set(thermo.components) - set(thermo.r)),
                initializer_attempts=required(result.performance, "vlle_initializer_attempts"),
                temperatures_C=required(result.performance, "stage_temperatures_C"),
                liquid_compositions=required(result.performance, "stage_liquid_compositions"),
                distillate_composition=dict(required(result.outlet_streams, "distillate").x),
                bottoms_composition=dict(required(result.outlet_streams, "bottoms").x),
                condenser_duty_kW=required(result.performance, "condenser_duty_kW"),
                reboiler_duty_kW=required(result.performance, "reboiler_duty_kW"),
            )
            parameters = {}
            for i, first in enumerate(thermo.components):
                for second in thermo.components[i + 1:]:
                    record = thermo._uniquac_interaction_for_components(first, second)
                    if record:
                        parameters[first + "/" + second] = {
                            key:value for key,value in record.items()
                            if key.startswith("tau") or key in ("a12_cal_per_mol", "a21_cal_per_mol", "model_variant", "use_q_prime")
                        }
            report["interaction_parameters"] = parameters
            (directory / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            return report


def compare(left, right):
    changed_parameters = []
    left_parameters, right_parameters = required(left, "interaction_parameters"), required(right, "interaction_parameters")
    for pair in sorted(left_parameters.keys() | right_parameters.keys()):
        a = left_parameters.get(pair)
        b = right_parameters.get(pair)
        if a != b:
            changed_parameters.append({"pair":pair, "left":a, "right":b})
    return {
        "left":required(left, "case"), "right":required(right, "case"),
        "max_stage_temperature_difference_K":max(abs(a-b) for a,b in zip(required(left, "temperatures_C"), required(right, "temperatures_C"))),
        "max_stage_liquid_mole_fraction_difference":max(
            abs(a.get(c, 0)-b.get(c, 0)) for a,b in zip(required(left, "liquid_compositions"), required(right, "liquid_compositions"))
            for c in a.keys() | b.keys()
        ),
        "reboiler_duty_difference_kW":required(right, "reboiler_duty_kW")-required(left, "reboiler_duty_kW"),
        "changed_interaction_parameters":changed_parameters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--code-root", type=Path, default=ROOT,
                        help="Use a fixed source snapshot while the main workspace is being edited.")
    parser.add_argument("--workers", type=int, choices=range(1, 6), default=2)
    args = parser.parse_args()
    code_root = args.code_root.resolve()
    output = Path(tempfile.mkdtemp(prefix="pfdsim-vlle-activity-data-"))
    baseline = {}
    for filename in TABLES:
        baseline[filename] = subprocess.check_output(
            ["git", "show", args.baseline_ref + ":data/" + filename], cwd=ROOT,
        )
    previous_r = required(required(required(json.loads(required(baseline, "uniquac_rq_cas.json")), "components"), "79-09-4"), "r")
    cache_source = code_root / "data" / "runtime" / "interaction_estimation_cache.sqlite"
    cache_seed = output / "estimation-seed.sqlite"
    if cache_source.is_file():
        with sqlite3.connect(cache_source.as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(cache_seed) as target:
                source.backup(target)
    cases = []
    for name in ("current", "current_previous_propionic_r", "baseline_data"):
        directory = output / name
        (directory / "data").mkdir(parents=True)
        for filename in TABLES:
            if name == "current_previous_propionic_r" and filename == "uniquac_rq_cas.json":
                continue
            if name == "baseline_data":
                (directory / "data" / filename).write_bytes(required(baseline, filename))
            else:
                shutil.copyfile(code_root / "data" / filename, directory / "data" / filename)
        if name == "current_previous_propionic_r":
            sys.path.insert(0, str(code_root))
            from scripts.activity_fitting import build_uniquac_rq_parameters as rq_builder
            records = rq_builder.thermochimica_1995_rq_records()
            adjusted = [{**item, "r":previous_r} if item.get("cas") == "79-09-4" else item for item in records]
            with patch.object(rq_builder, "thermochimica_1995_rq_records", return_value=adjusted):
                payload = rq_builder.build_uniquac_rq_payload()
            (directory / "data" / "uniquac_rq_cas.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if cache_seed.exists():
            shutil.copyfile(cache_seed, directory / "estimation.sqlite")
        cases.append({"name":name, "directory":str(directory), "code_root":str(code_root),
                      "previous_propionic_r":previous_r if name == "current_previous_propionic_r" else None})
    print("Artifacts:", output, flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(run_case, case):case for case in cases}
        for future in as_completed(pending):
            result = future.result()
            results[required(result, "case")] = result
            print(json.dumps({key:required(result, key) for key in (
                "case", "propionic_r", "jacobian_evaluations", "function_evaluations",
                "solver_iterations", "mesh_residual", "component_balance_error", "topology", "elapsed_s",
            )}), flush=True)
    comparisons = [compare(required(results, "current"), required(results, "current_previous_propionic_r")),
                   compare(required(results, "baseline_data"), required(results, "current_previous_propionic_r"))]
    (output / "comparison.json").write_text(json.dumps({"baseline_ref":args.baseline_ref, "comparisons":comparisons}, indent=2, sort_keys=True) + "\n")
    for item in comparisons:
        print(json.dumps({key:value for key,value in item.items() if key != "changed_interaction_parameters"}), flush=True)
    print("Complete report:", output / "comparison.json", flush=True)


if __name__ == "__main__":
    main()
