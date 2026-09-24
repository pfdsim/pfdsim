#!/usr/bin/env python3
"""Benchmark real column solves across three saved source versions.

Example (run from the repository root; output directory must not exist):
    python scripts/performance/benchmark_sparse_column_solves.py \
        --before-fixes /tmp/pfdsim-sparse-jacobian-before-trap-fixes.py \
        --output /tmp/pfdsim-column-comparison --repeats 3

Original uses HEAD for the four changed column modules. Sparse uses their
working-tree contents and the supplied pre-fix helper. Current snapshots all
five working-tree modules. Other dependencies and property data are shared.
Workers run sequentially in fresh processes, rotating version order between
repetitions. Setup/import time is excluded from column solve timings. Each
worker has a persistent log; completed records are appended immediately.
The first repetition is retained as warm-up and excluded from timing reports.
Use --start-repeat 4 --repeats 1 with an existing output directory to collect
an additional pass against its saved snapshots, without overwriting results.
"""

from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import resource
from statistics import median
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
MODULES = (
    "equilibrium_stage_column.py", "equilibrium_stage_vlle.py",
    "unit_operations_distillation.py", "unit_operations_separation.py",
)
VERSIONS = ("original", "sparse", "current")
CASES = {
    "benzene_toluene_20": "examples/benzene_toluene_20_stage_distillation_nrtl.pfd",
    "hydrocarbon_pr_42": "examples/biosteam_mesh_hydrocarbon_distillation.pfd (offline)",
    "partial_mass_12": "tests/test_unit_operations.py::test_rigorous_distillation_semi_analytic_jacobian_handles_partial_mass_spec",
    "partial_mass_fd_12": "same partial-mass test, colored finite differences",
    "cmo_20": "tests/test_unit_operations.py::test_methanol_water_hvap_variants_improve_mccabe_and_cmo_compositions",
    "cmo_hvap_20": "same CMO test, latent_heat_correction=True",
    "vlle_butanol_20": "tests/test_rigorous_distillation_vlle.py::test_butanol_water_projection_contracts_quickly_near_pure_bottoms",
    "acrylic_extraction_20": "examples/acrylic_acid_rigorous_extraction.pfd",
    "acrylic_extraction_flow_3": "tests/test_unit_operations.py::test_equation_oriented_rigorous_extractor_matches_split_sweep (flow-only)",
    "pyridine_extraction_20": "tests/test_unit_operations.py::test_auto_initializer_uses_coarse_grid_for_larger_extractors",
    "absorption_5": "tests/test_unit_operations.py::test_rigorous_absorber_concentrated_uniquac_acetaldehyde_case",
    "absorption_flow_5": "same absorber test, flow-only Jacobian",
    "absorption_fd_5": "same absorber test, colored finite differences",
    "stripping_5": "tests/test_unit_operations.py::test_rigorous_stripper_concentrated_uniquac_acetaldehyde_case",
    "trace_stripping_5": "examples/trace_organic_water_stripping_isothermal_unifnist.pfd",
}


def prepare_case(name):
    from simulator import Simulator
    from thermodynamics import create_thermodynamics
    from unit_operations_distillation import CMODistillation, RigorousDistillation
    from unit_operations_separation import (
        RigorousAbsorber, RigorousLiquidLiquidExtractor, RigorousStripper,
    )

    examples = {
        "benzene_toluene_20": ("benzene_toluene_20_stage_distillation_nrtl.pfd", "COL-1"),
        "hydrocarbon_pr_42": ("biosteam_mesh_hydrocarbon_distillation.pfd", "COL-1"),
        "acrylic_extraction_20": ("acrylic_acid_rigorous_extraction.pfd", "EX-100"),
        "trace_stripping_5": ("trace_organic_water_stripping_isothermal_unifnist.pfd", "STRIP-100"),
    }
    if name in examples:
        filename, unit_id = examples[name]
        simulator = Simulator.from_file(ROOT / "examples" / filename)
        simulator.pfd.metadata.online_lookup = False
        simulator.initialize()
        solver = simulator.solver
        solver._initialize_streams()
        inlets = {
            solver.stream_connections[stream_id][3]: solver._transition_stream_state(
                stream_id, solver.streams[stream_id], solver.unit_thermo_scopes[unit_id]
            )
            for stream_id in solver.unit_inlets[unit_id]
        }
        return solver.units[unit_id], inlets

    def state(thermo, temperature, pressure, flow, composition, phase="liquid"):
        return thermo.calculate_state(
            temperature, pressure, flow, composition, phase=phase, flash=False
        )

    if name.startswith("partial_mass"):
        thermo = create_thermodynamics(["methanol", "water"], "UNIFAC")
        feed = state(thermo, 298.15, 1.0, 100.0, {"methanol": 0.4, "water": 0.6})
        params = {
            "N_stages": 12, "feed_stage": 7, "reflux_ratio": 2.0,
            "D_mass_to_F_mass": 0.35, "P_condenser": 1.0,
            "P_drop_per_stage": 0.0, "condenser_type": "partial",
            "mesh_tolerance": 1e-6, "max_iterations": 100,
            "initializer": "cheap_estimate",
        }
        if "_fd_" in name:
            params["semi_analytic_flow_jacobian"] = False
        return RigorousDistillation(name, thermo, params), {"feed": feed}

    if name.startswith("cmo_"):
        thermo = create_thermodynamics(["methanol", "water"], "NRTL")
        composition = {"methanol": 0.3, "water": 0.7}
        feed = state(thermo, thermo.bubble_point_T(composition, 1.0), 1.0, 100.0, composition)
        params = {
            "N_stages": 20, "feed_stage": 15, "reflux_ratio": 1.0,
            "D_to_F": 0.3, "P_condenser": 1.0, "P_drop_per_stage": 0.0,
            "mesh_tolerance": 1e-7, "max_iterations": 160,
            "max_jacobian_evaluations": 160,
            "latent_heat_correction": name == "cmo_hvap_20",
        }
        return CMODistillation(name, thermo, params), {"feed": feed}

    if name == "vlle_butanol_20":
        thermo = create_thermodynamics(["butanol", "water"], "NRTL")
        feed = state(thermo, 298.15, 1.0, 100.0, {"butanol": 0.4, "water": 0.6})
        params = {
            "N_stages": 20, "feed_stage": 10, "reflux_ratio": 1.2,
            "D_to_F": 0.773, "P_condenser": 1.0, "P_drop_per_stage": 0.0,
            "condenser_type": "total", "stage_phase_model": "VLLE",
            "vlle_seed": "cheap", "max_iterations": 60,
            "max_jacobian_evaluations": 60,
        }
        return RigorousDistillation(name, thermo, params), {"feed": feed}

    if name in ("acrylic_extraction_flow_3", "pyridine_extraction_20"):
        acrylic = name == "acrylic_extraction_flow_3"
        components = (["diethyl ether", "n-hexane", "acrylic acid", "water"]
                      if acrylic else ["pyridine", "water", "diethyl ether"])
        thermo = create_thermodynamics(components, "UNIFAC" if acrylic else "UNIFNIST")

        def mass_stream(masses, temperature):
            # Preserve the test fixture's flow convention exactly.
            molar = {comp: mass * 1000.0 / thermo.props[comp].MW for comp, mass in masses.items()}
            flow = sum(molar.values())
            return state(thermo, temperature, 1.0, flow, {comp: v / flow for comp, v in molar.items()})

        if acrylic:
            feed = mass_stream({"acrylic acid": 200.0, "water": 1500.0}, 298.15)
            solvent = mass_stream({"diethyl ether": 250.0, "n-hexane": 250.0}, 298.15)
            params = {"N_stages": 3, "T": 25, "max_iterations": 80,
                      "solver_algorithm": "equation_oriented",
                      "semi_analytic_local_thermo_jacobian": False}
        else:
            feed = mass_stream({"pyridine": 100.0, "water": 900.0}, 303.15)
            solvent = mass_stream({"diethyl ether": 500.0}, 293.15)
            params = {"N_stages": 20, "mode": "adiabatic", "T": 298.15,
                      "max_iterations": 80, "max_jacobian_evaluations": 60}
        return RigorousLiquidLiquidExtractor(name, thermo, params), {"feed": feed, "solvent": solvent}

    thermo = create_thermodynamics(["water", "acetaldehyde", "nitrogen", "oxygen"], "UNIQUAC")
    params = {"N_stages": 5, "P_drop_per_stage": 0.0}
    if name.startswith("absorption"):
        gas = state(thermo, 298.15, 1.01325, 100.0,
                    {"acetaldehyde": 0.1, "nitrogen": 0.711, "oxygen": 0.189}, "vapor")
        liquid = state(thermo, 298.15, 1.01325, 400.0, {"water": 1.0})
        if "_flow_" in name:
            params["semi_analytic_local_thermo_jacobian"] = False
        if "_fd_" in name:
            params["semi_analytic_flow_jacobian"] = False
        return RigorousAbsorber(name, thermo, params), {"gas": gas, "water": liquid}
    if name == "stripping_5":
        liquid = state(thermo, 323.15, 1.01325, 200.0, {"water": 0.95, "acetaldehyde": 0.05})
        gas = state(thermo, 353.15, 1.01325, 100.0, {"nitrogen": 0.79, "oxygen": 0.21}, "vapor")
        return RigorousStripper(name, thermo, params), {"liquid": liquid, "air": gas}
    raise ValueError(name)


def install_timers():
    from equilibrium_stage_column import EquilibriumStageColumnMixin
    from unit_operations_separation import RigorousLiquidLiquidExtractor

    stats = {"jacobian_seconds": 0.0, "jacobian_calls": 0}

    def timed(function):
        if getattr(function, "_column_benchmark_timer", False):
            return function

        @wraps(function)
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                stats["jacobian_seconds"] += time.perf_counter() - start
                stats["jacobian_calls"] += 1
        wrapped._column_benchmark_timer = True
        return wrapped

    def instrument_solver(original):
        signature = inspect.signature(original)

        @wraps(original)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            callback = bound.arguments.get("jacobian")
            if callback is not None:
                bound.arguments["jacobian"] = timed(callback)
            return original(*bound.args, **bound.kwargs)
        return wrapped

    for cls in (EquilibriumStageColumnMixin, RigorousLiquidLiquidExtractor):
        cls._sparse_newton_solve = instrument_solver(cls._sparse_newton_solve)
        cls._finite_difference_jacobian = timed(cls._finite_difference_jacobian)
    return stats


def stream_record(stream):
    return {"F": stream.F, "T": stream.T, "P": stream.P,
            "composition": dict(stream.composition)}


def worker(args):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.overlay.resolve()))
    import numpy as np
    np.random.seed(20260924)
    record = {"case": args.worker, "version": args.version, "repeat": args.repeat}
    start = time.perf_counter()
    try:
        unit, inlets = prepare_case(args.worker)
        record["setup_seconds"] = time.perf_counter() - start
        record["inputs"] = {port: stream_record(stream) for port, stream in inlets.items()}
        stats = install_timers()
        record["peak_rss_before_solve_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        start = time.perf_counter()
        cpu_start = time.process_time()
        result = unit.solve(inlets)
        record.update(solve_seconds=time.perf_counter() - start,
                      solve_cpu_seconds=time.process_time() - cpu_start, **stats)
        record["performance"] = result.performance
        record["warnings"] = result.warnings
        record["heat_duty"] = result.heat_duty
        record["outputs"] = {port: stream_record(stream) for port, stream in result.outlet_streams.items()}
        components = set().union(*(s.composition for s in inlets.values()),
                                 *(s.composition for s in result.outlet_streams.values()))
        feed_flow = sum(s.F for s in inlets.values())
        record["component_balance_error"] = max(
            abs(sum(s.F * s.composition.get(comp, 0.0) for s in inlets.values())
                - sum(s.F * s.composition.get(comp, 0.0) for s in result.outlet_streams.values()))
            / max(feed_flow, 1.0) for comp in components
        )
        record["success"] = True
    except Exception as error:
        record.update(success=False, error=f"{type(error).__name__}: {error}",
                      elapsed_to_failure=time.perf_counter() - start)
        traceback.print_exc()
    record["peak_rss_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    with args.result.open("x") as handle:
        json.dump(record, handle, default=lambda obj: obj.item() if hasattr(obj, "item") else str(obj))
    print("RESULT", record["case"], record["version"], record["success"], flush=True)


def snapshot(output, before_fixes):
    helper = before_fixes.read_bytes()
    manifest = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "python": sys.version, "platform": platform.platform(), "sources": {},
                "cases": CASES, "seed": 20260924, "blas_threads": 1}
    current = {name: (ROOT / name).read_bytes() for name in (*MODULES, "sparse_jacobian.py")}
    for version in VERSIONS:
        directory = output / version
        directory.mkdir()
        for name, contents in current.items():
            if version == "original" and name in MODULES:
                contents = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
            elif version == "sparse" and name == "sparse_jacobian.py":
                contents = helper
            (directory / name).write_bytes(contents)
            manifest["sources"][f"{version}/{name}"] = hashlib.sha256(contents).hexdigest()
    # Retain the actual harness and the complete working diff with each run.
    (output / "benchmark_script.py").write_bytes(Path(__file__).read_bytes())
    (output / "working.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def summarize(output, report_name="summary.md", warmup_repeats=1):
    all_records = [json.loads(line) for path in sorted(output.glob("results*.jsonl"))
                   for line in path.read_text().splitlines()]
    records = [record for record in all_records if record["repeat"] > warmup_repeats]
    if not records:
        raise ValueError("No measured repetitions remain after excluding warm-up")
    lines = ["# Sparse column solve comparison", "",
             "Fresh-process column solves; setup excluded. Times are medians in seconds.", "",
             f"Repetitions 1–{warmup_repeats} are warm-up and excluded. "
             f"All {len(all_records)} raw records are retained; {len(records)} enter this report.", "",
             "| Case | Original | Sparse | Current | Current/original | Current/sparse | Max residual | Max component balance error |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case in dict.fromkeys(record["case"] for record in records):
        group = [record for record in records if record["case"] == case]
        medians = {}
        for version in VERSIONS:
            successes = [r["solve_seconds"] for r in group if r["version"] == version and r["success"]]
            medians[version] = median(successes) if successes else None
        residuals = [float(r["performance"].get("mesh_residual", r["performance"].get("mes_residual")))
                     for r in group if r["success"]
                     and ("mesh_residual" in r["performance"] or "mes_residual" in r["performance"])]
        balances = [r["component_balance_error"] for r in group if r["success"]]
        times = [f"{medians[v]:.4f}" if medians[v] is not None else "FAILED" for v in VERSIONS]
        ratios = [f"{medians['current'] / medians[v]:.3f}" if medians['current'] and medians[v] else "—"
                  for v in ("original", "sparse")]
        lines.append(f"| {case} | {' | '.join(times + ratios)} | "
                     f"{max(residuals, default=float('nan')):.3e} | {max(balances, default=float('nan')):.3e} |")
    failures = [r for r in records if not r["success"]]
    lines.extend(["", f"Measured records: {len(records)}; failed measured solves: {len(failures)}.",
                  f"Failed solves including warm-up: {sum(not r['success'] for r in all_records)}."])
    for record in failures:
        lines.append(f"- {record['case']} / {record['version']} / repeat {record['repeat']}: {record['error']}")
    details = {}
    for case in dict.fromkeys(record["case"] for record in records):
        group = [record for record in records if record["case"] == case]
        detail = {"versions": {}, "comparisons": {}}
        for version in VERSIONS:
            runs = [r for r in group if r["version"] == version and r["success"]]
            metrics = {}
            for key in ("solve_seconds", "solve_cpu_seconds", "jacobian_seconds",
                        "peak_rss_MiB", "component_balance_error"):
                values = [r[key] for r in runs]
                if values:
                    metrics[key] = {"median": median(values), "min": min(values), "max": max(values)}
            for key in ("mesh_residual", "mes_residual", "overall_energy_relative_error",
                        "max_stage_energy_relative_error", "component_balance_error",
                        "solver_iterations", "function_evaluations", "jacobian_evaluations",
                        "jacobian_method", "initializer", "vlle_topology"):
                values = [r["performance"].get(key) for r in runs]
                if any(value is not None for value in values):
                    label = "reported_component_balance_error" if key == "component_balance_error" else key
                    metrics[label] = values
            detail["versions"][version] = metrics
        for baseline in ("original", "sparse"):
            comparison = {"max_composition_absolute_delta": 0.0,
                          "max_temperature_K_delta": 0.0,
                          "max_flow_delta_over_feed": 0.0,
                          "max_heat_duty_relative_delta": 0.0,
                          "identical_inputs": True, "identical_output_ports": True,
                          "paired_successes": 0}
            for current in (r for r in group if r["version"] == "current" and r["success"]):
                reference = next((r for r in group if r["version"] == baseline
                                  and r["repeat"] == current["repeat"] and r["success"]), None)
                if reference is None:
                    continue
                comparison["paired_successes"] += 1
                comparison["identical_inputs"] &= current["inputs"] == reference["inputs"]
                comparison["identical_output_ports"] &= current["outputs"].keys() == reference["outputs"].keys()
                feed_flow = max(sum(s["F"] for s in current["inputs"].values()), 1.0)
                for port in current["outputs"].keys() & reference["outputs"].keys():
                    left, right = current["outputs"][port], reference["outputs"][port]
                    comparison["max_temperature_K_delta"] = max(
                        comparison["max_temperature_K_delta"], abs(left["T"] - right["T"]))
                    comparison["max_flow_delta_over_feed"] = max(
                        comparison["max_flow_delta_over_feed"], abs(left["F"] - right["F"]) / feed_flow)
                    for comp in left["composition"].keys() | right["composition"].keys():
                        comparison["max_composition_absolute_delta"] = max(
                            comparison["max_composition_absolute_delta"],
                            abs(left["composition"].get(comp, 0.0) - right["composition"].get(comp, 0.0)))
                comparison["max_heat_duty_relative_delta"] = max(
                    comparison["max_heat_duty_relative_delta"],
                    abs(current["heat_duty"] - reference["heat_duty"])
                    / max(abs(reference["heat_duty"]), 1.0))
            detail["comparisons"][baseline] = comparison
        details[case] = detail
    successes = [r for r in records if r["success"]]
    finite = all(math.isfinite(float(r["performance"].get("mesh_residual", r["performance"].get("mes_residual", float("nan")))))
                 and math.isfinite(r["component_balance_error"])
                 for r in successes)
    lines.extend(["", f"Reported residuals and balances finite: {finite}.", "",
                  "## Numerical differences (current versus baseline)", "",
                  "| Case | Baseline | Max composition delta | Max temperature delta, K | Max flow delta / feed | Identical inputs |",
                  "|---|---|---:|---:|---:|---|"])
    for case, detail in details.items():
        for baseline, comparison in detail["comparisons"].items():
            lines.append(f"| {case} | {baseline} | {comparison['max_composition_absolute_delta']:.3e} | "
                         f"{comparison['max_temperature_K_delta']:.3e} | {comparison['max_flow_delta_over_feed']:.3e} | "
                         f"{comparison['identical_inputs']} |")
    lines.extend(["", "## Timing variability and process memory", "",
                  "| Case | Version | Solve min–max, s | Median Jacobian time, s | Median process peak RSS, MiB |",
                  "|---|---|---:|---:|---:|"])
    for case, detail in details.items():
        for version, metrics in detail["versions"].items():
            if "solve_seconds" in metrics:
                timing = metrics["solve_seconds"]
                lines.append(f"| {case} | {version} | {timing['min']:.4f}–{timing['max']:.4f} | "
                             f"{metrics['jacobian_seconds']['median']:.4f} | {metrics['peak_rss_MiB']['median']:.1f} |")
    if all("solve_seconds" in d["versions"][v] for d in details.values() for v in VERSIONS):
        totals = {v: sum(d["versions"][v]["solve_seconds"]["median"] for d in details.values())
                  for v in VERSIONS}
        lines.extend(["", "Sum of per-case median solve times (one of each case): "
                      + ", ".join(f"{v}={value:.6f} s" for v, value in totals.items()) + "."])
        for baseline in ("original", "sparse"):
            geometric_ratio = math.exp(sum(
                math.log(d["versions"]["current"]["solve_seconds"]["median"]
                         / d["versions"][baseline]["solve_seconds"]["median"])
                for d in details.values()) / len(details))
            lines.append(f"Current/{baseline}: summed-time ratio={totals['current'] / totals[baseline]:.6f}; "
                         f"equal-case geometric mean ratio={geometric_ratio:.6f}.")
    # Exclusive writes preserve previously collected reports.
    with (output / (Path(report_name).stem + ".json")).open("x") as handle:
        json.dump(details, handle, indent=2)
    with (output / report_name).open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-fixes", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start-repeat", type=int, default=1)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--version", choices=VERSIONS)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--summarize", action="store_true",
                        help="Analyze existing --output/results.jsonl into new analysis.md/json files")
    args = parser.parse_args()
    if args.summarize:
        if not args.output:
            parser.error("--summarize requires --output")
        summarize(args.output, "analysis.md", args.warmup_repeats)
        return
    if args.worker:
        worker(args)
        return
    if not args.output or args.repeats < 1 or args.start_repeat < 1 or args.warmup_repeats < 0:
        parser.error("--output, positive --repeats/--start-repeat, and nonnegative --warmup-repeats are required")
    args.output = args.output.resolve()
    if args.start_repeat == 1:
        if not args.before_fixes:
            parser.error("--before-fixes is required when creating snapshots")
        args.before_fixes = args.before_fixes.resolve(strict=True)
        args.output.mkdir(parents=True, exist_ok=False)
        snapshot(args.output, args.before_fixes)
    else:
        manifest = json.loads((args.output / "manifest.json").read_text())
        for name, digest in manifest["sources"].items():
            if hashlib.sha256((args.output / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Saved source snapshot changed: {name}")
        for name in (*MODULES, "sparse_jacobian.py"):
            if (ROOT / name).read_bytes() != (args.output / "current" / name).read_bytes():
                raise ValueError(f"Working source changed since snapshot: {name}")
    with (args.output / f"benchmark_script-repeat{args.start_repeat}.py").open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    with (args.output / f"pass-{args.start_repeat}.json").open("x") as handle:
        json.dump({"start_repeat": args.start_repeat, "repeats": args.repeats,
                   "cases": args.cases, "warmup_repeats": args.warmup_repeats}, handle, indent=2)
    environment = dict(os.environ, PYTHONHASHSEED="20260924", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    results_name = "results.jsonl" if args.start_repeat == 1 else f"results-repeat{args.start_repeat}.jsonl"
    with (args.output / results_name).open("x") as results:
        for repeat in range(args.start_repeat - 1, args.start_repeat - 1 + args.repeats):
            for case in args.cases:
                order = VERSIONS[repeat % 3:] + VERSIONS[:repeat % 3]
                for version in order:
                    stem = f"{case}.{version}.{repeat + 1}"
                    result_path = args.output / f"{stem}.json"
                    command = [sys.executable, str(Path(__file__).resolve()), "--worker", case,
                               "--overlay", str(args.output / version), "--version", version,
                               "--repeat", str(repeat + 1), "--result", str(result_path)]
                    print(f"START {stem}", flush=True)
                    with (args.output / f"{stem}.log").open("x") as log:
                        completed = subprocess.run(command, cwd=ROOT, env=environment,
                                                   stdout=log, stderr=subprocess.STDOUT)
                    if completed.returncode:
                        # Never restart a failed/killed worker or continue after a signal.
                        print(f"STOP {stem}: exit {completed.returncode}; inspect its log", flush=True)
                        raise SystemExit(128 - completed.returncode if completed.returncode < 0 else completed.returncode)
                    record = json.loads(result_path.read_text())
                    results.write(json.dumps(record) + "\n")
                    results.flush()
                    print(f"DONE {stem} success={record['success']} "
                          f"solve_s={record.get('solve_seconds')} jac_s={record.get('jacobian_seconds')}", flush=True)
    report_name = "summary.md" if args.start_repeat == 1 else f"summary-repeat{args.start_repeat}.md"
    summarize(args.output, report_name, args.warmup_repeats)


if __name__ == "__main__":
    main()
