#!/usr/bin/env python3
"""Probe slightly overdrawn azeotropic first cuts with native column equations.

The cut capacity is min(z_feed[i]/x_azeotrope[i]) over participating species:
the largest D/F that could have exactly that azeotrope composition. Exceeding
it requests more distillate, not a composition across an azeotropic boundary.
Actual product composition remains free.

Example, from the repository root, with a new output directory:
  .venv/bin/python scripts/performance/probe_azeotropic_overdistillation.py \
      --output /tmp/azeotropic-overdraw --factors .98 1.005 1.02 1.05

Workers run sequentially, single-threaded BLAS, one fresh process per physical
case. Within a case the thermo caches are warmed and shared. Initializers and
methods rotate order across cuts. No random perturbations or production edits.
Native MESH retries and relaxed acceptance are disabled to isolate initialization
and globalization; --native-retries restores the Jacobian retry policy.
Each completed solve is persisted immediately. Native column
output validation still runs, with an additional local spinodal check on every
accepted liquid stage. This is not a global phase-stability proof.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_column_trust_region import trust_solve
from benchmark_sparse_column_solves import stream_record
from probe_azeotropic_initializer_matrix import prepare_case as regression_case

CASES = ("ethanol_water", "ipa_water", "nitrile_ternary", "ethanol_water_propanol")


def prepare_case(name):
    from thermodynamics import create_thermodynamics
    from unit_operations_distillation import RigorousDistillation

    if name == "ethanol_water":
        column, inlets = regression_case("example_ethanol_water")
        column.params.update(N_stages=30, feed_stage=21, reflux_ratio=8.)
        return column, inlets
    if name == "nitrile_ternary":
        return regression_case("test_nitrile")
    if name == "ipa_water":
        components, composition = ["isopropanol", "water"], {"isopropanol": .2, "water": .8}
        stages, feed_stage, reflux = 30, 20, 8.
    else:
        components = ["ethanol", "water", "1-propanol"]
        composition = {"ethanol": .1, "water": .85, "1-propanol": .05}
        stages, feed_stage, reflux = 40, 28, 8.
    thermo = create_thermodynamics(components, "NRTL")
    feed = thermo.calculate_state(298.15, 1., 100., composition, phase="liquid", flash=False)
    column = RigorousDistillation(name, thermo, {
        "N_stages": stages, "feed_stage": feed_stage, "reflux_ratio": reflux,
        "P_condenser": 1., "P_drop_per_stage": 0., "condenser_type": "total",
        "D_to_F": .2,
    })
    return column, {"feed": feed}


def candidate_metadata(column, feed):
    comps = list(feed.composition)
    pressure = float(column.get_param("P_condenser", feed.P))
    begin = time.perf_counter()
    candidates = column._vle_azeotrope_candidates(comps, pressure)
    records = []
    for candidate in candidates:
        item = dict(candidate)
        active = [comp for comp, value in item["composition"].items() if value > 1e-8]
        item["cut_capacity"] = min(feed.composition[c] / item["composition"][c] for c in active)
        item["limiting_component"] = min(active, key=lambda c: feed.composition[c] / item["composition"][c])
        item["K_values"] = column.thermo.K_values(item["T"], pressure, item["composition"])
        item["max_K_minus_one"] = max(abs(item["K_values"][c] - 1) for c in active)
        item["member_pure_bubble_T"] = {
            c: column.thermo.bubble_point_T({v: float(v == c) for v in comps}, pressure) for c in active
        }
        item["minimum_boiling"] = item["T"] < min(item["member_pure_bubble_T"].values())
        spinodal = getattr(column.thermo, "liquid_spinodal_stability", None)
        if spinodal is not None:
            item["local_liquid_stability"] = spinodal(item["T"], item["composition"])
        records.append(item)
    minimum = [c for c in records if c["minimum_boiling"] and 0 < c["cut_capacity"] < 1.]
    if not minimum:
        raise ValueError(f"no usable minimum-boiling first azeotrope: {records}")
    first = min(minimum, key=lambda c: c["T"])
    return {"components": comps, "pressure": pressure, "feed": stream_record(feed),
            "column_params": dict(column.params), "candidates": records, "first_azeotrope": first,
            "candidate_seconds": time.perf_counter() - begin}


def instrument(unit, method, attempts):
    from equilibrium_stage_column import EquilibriumStageColumnMixin
    native = EquilibriumStageColumnMixin._sparse_newton_solve

    def wrapped(self, residual, sparsity, x0, options, jacobian=None, step_event=None):
        import numpy as np
        # The native quality-context recursion must not create extra attempts.
        prior = getattr(self, "_quality_solver_aux_context_active", False)
        self._quality_solver_aux_context_active = True
        quality = getattr(self.thermo, "quality_context", None)
        entry = {"unit_id": self.unit_id, "size": len(x0), "options": dict(options),
                 "x0_sha256": hashlib.sha256(np.asarray(x0, dtype=float).tobytes()).hexdigest(),
                 "jacobian_seconds": 0., "rejected_trials": 0, "trust_trials": []}
        attempts.append(entry)
        begin = time.perf_counter()
        try:
            with quality(phase="solver_iteration", affects_result=False) if quality else nullcontext():
                initial = residual(x0)
                entry["initial_residual"] = float(np.linalg.norm(initial, ord=np.inf))
                entry["initial_merit"] = .5 * float(initial @ initial)
                if method == "line":
                    result = native(self, residual, sparsity, x0, options, jacobian=jacobian, step_event=step_event)
                else:
                    result = trust_solve(self, residual, sparsity, x0, options, jacobian,
                                         step_event, method, entry)
                entry.update({k: v for k, v in result.items() if k != "x"})
                return result
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            self._quality_solver_aux_context_active = prior
            entry["seconds"] = time.perf_counter() - begin
            print("ATTEMPT", self.unit_id, method, entry.get("success"),
                  entry.get("residual_norm"), entry.get("jacobian_evaluations"),
                  f"{entry['seconds']:.3f}s", flush=True)

    return patch.object(EquilibriumStageColumnMixin, "_sparse_newton_solve", wrapped)


def solve_job(base, inlets, metadata, args, factor, initializer, method, warm_profiles):
    from unit_operations_distillation import RigorousDistillation
    params = dict(base.params)
    params.update(D_to_F=factor * metadata["first_azeotrope"]["cut_capacity"],
                  initializer="cheap_estimate" if initializer == "continuation" else initializer,
                  mesh_tolerance=1e-6, acceptable_mesh_residual=1e-6,
                  max_iterations=args.budget, max_jacobian_evaluations=args.budget,
                  finite_difference_rel_step=1e-6, colored_jacobian_fallback=False,
                  stage_phase_model="VLE")
    if args.native_retries:
        params.pop("finite_difference_rel_step", None)
        params.pop("colored_jacobian_fallback", None)
    unit = RigorousDistillation(base.unit_id, base.thermo, params)
    if initializer == "continuation":
        unit.solve_context = {"recycle_evaluation": 0}
        if method in warm_profiles:
            unit._last_recycle_profile = warm_profiles[method]
    attempts = []
    record = {"case": args.worker, "factor": factor, "cut": params["D_to_F"],
              "unit_id": unit.unit_id,
              "initializer": initializer, "method": method, "budget": args.budget,
              "params": params, "attempts": attempts}
    begin = time.perf_counter()
    try:
        with instrument(unit, method, attempts):
            result = unit.solve(inlets)
        record.update(success=True, solve_seconds=time.perf_counter() - begin,
                      residual=result.performance["mesh_residual"], performance=result.performance,
                      outputs={p: stream_record(s) for p, s in result.outlet_streams.items()},
                      warnings=result.warnings, heat_duty=result.heat_duty)
        if initializer == "continuation":
            warm_profiles[method] = unit._last_recycle_profile
            record["used_initializer"] = result.performance["initializer"]
        flow = sum(s.F for s in inlets.values())
        components = metadata["components"]
        record["component_balance_error"] = max(abs(
            sum(s.F*s.composition.get(c, 0.) for s in inlets.values()) -
            sum(s.F*s.composition.get(c, 0.) for s in result.outlet_streams.values())) / flow for c in components)
        distillate = result.outlet_streams["distillate"]
        record["azeotrope_composition_delta"] = max(abs(
            distillate.composition.get(c, 0.) - metadata["first_azeotrope"]["composition"].get(c, 0.)) for c in components)
        spinodal = getattr(unit.thermo, "liquid_spinodal_stability", None)
        if spinodal is not None:
            record["stage_local_stability"] = [spinodal(T+273.15, x) for T, x in zip(
                result.performance["stage_temperatures_C"], result.performance["stage_liquid_compositions"])]
            record["locally_unstable_stages"] = [j+1 for j, value in enumerate(record["stage_local_stability"])
                                                  if value.get("locally_stable") is False]
    except Exception as error:
        record.update(success=False, solve_seconds=time.perf_counter() - begin,
                      error=f"{type(error).__name__}: {error}")
        if attempts:
            main_attempts = [a for a in attempts if a["unit_id"] == unit.unit_id]
            values = [a["residual_norm"] for a in main_attempts if a.get("residual_norm") is not None]
            record["residual"] = min(values) if values else None
        traceback.print_exc()
    record["total_jacobians"] = sum(a.get("jacobian_evaluations", 0) for a in attempts)
    record["total_functions"] = sum(a.get("function_evaluations", 0) for a in attempts)
    print(args.worker, factor, initializer, method, record["success"], record.get("residual"),
          f"{record['solve_seconds']:.3f}s", flush=True)
    return record


def worker(args):
    import numpy as np
    import scipy.optimize
    import scipy.sparse.linalg  # noqa: F401 -- warm the numerical machinery before timing

    random.seed(20261007)
    np.random.seed(20261007)
    unit, inlets = prepare_case(args.worker)
    metadata = candidate_metadata(unit, inlets["feed"])
    metadata["numpy"] = np.__version__
    with (args.output / f"{args.worker}-metadata.json").open("x") as handle:
        handle.write(json.dumps(metadata, indent=2, default=json_default) + "\n")
    if args.inspect:
        print(json.dumps(metadata, indent=2, default=json_default))
        return
    if args.audit_directory:
        audit_phases(unit, metadata, args)
        return
    warm_profiles = {}
    with (args.output / f"{args.worker}-results.jsonl").open("x") as handle:
        for index, factor in enumerate(args.factors):
            if not 0 < factor * metadata["first_azeotrope"]["cut_capacity"] < 1:
                raise ValueError("overdraw factor leaves no positive bottoms")
            initializers = args.initializers[index % len(args.initializers):] + args.initializers[:index % len(args.initializers)]
            for initializer in initializers:
                methods = args.methods[index % len(args.methods):] + args.methods[:index % len(args.methods)]
                for method in methods:
                    record = solve_job(unit, inlets, metadata, args, factor, initializer, method, warm_profiles)
                    handle.write(json.dumps(record, default=json_default) + "\n")
                    handle.flush()


def json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def audit_phases(unit, metadata, args):
    """Check first azeotrope and top/feed/bottom of one solution per cut."""
    source = args.audit_directory / f"{args.worker}-results.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines()]
    points = [{"kind": "first_azeotrope", "T": metadata["first_azeotrope"]["T"],
               "composition": metadata["first_azeotrope"]["composition"]}]
    for factor in dict.fromkeys(r["factor"] for r in records):
        successes = [r for r in records if r["factor"] == factor and r["success"]]
        if not successes:
            continue
        best = min(successes, key=lambda r: r["residual"])
        performance = best["performance"]
        N = int(performance["N_stages"])
        feed_stage = int(best["params"]["feed_stage"])
        for index in sorted({0, feed_stage - 1, N - 1}):
            points.append({"kind": "column_stage", "factor": factor,
                           "stage": index + 1, "initializer": best["initializer"], "method": best["method"],
                           "T": performance["stage_temperatures_C"][index] + 273.15,
                           "composition": performance["stage_liquid_compositions"][index]})
    with (args.output / f"{args.worker}-phase-audit.jsonl").open("x") as handle:
        for point in points:
            begin = time.perf_counter()
            has_lle, x1, x2, beta = unit.thermo.liquid_liquid_equilibrium(point["composition"], point["T"])
            point.update(has_lle=bool(has_lle), x1=x1, x2=x2, beta=beta,
                         seconds=time.perf_counter()-begin)
            handle.write(json.dumps(point, default=json_default)+"\n")
            handle.flush()
            print("PHASE", args.worker, point["kind"], point.get("factor"), point.get("stage"), has_lle, flush=True)


def summarize(output):
    """Produce a result table from persisted records, without new solves."""
    lines = ["# Azeotropic first-cut overdraw", "",
             "Residual tolerance: 1e-6; setup/candidate discovery excluded from solve times.", "",
             "| Case | First azeotrope | Temperature K | Capacity D/F |",
             "|---|---|---:|---:|"]
    for path in sorted(output.glob("*-metadata.json")):
        metadata = json.loads(path.read_text())
        first = metadata["first_azeotrope"]
        lines.append(f"| {path.name.removesuffix('-metadata.json')} | {first['name']} | {first['T']:.6f} | {first['cut_capacity']:.9f} |")
    records = [json.loads(line) for path in sorted(output.glob("*-results.jsonl"))
               for line in path.read_text().splitlines()]
    if records:
        lines += ["", "| Case | Cut / capacity | Initializer | Method | Success | Final residual | Total Jacobians | Solve seconds | Unstable stages |",
                  "|---|---:|---|---|---|---:|---:|---:|---|"]
        for row in records:
            residual = row.get("residual")
            if not row["success"] and row["initializer"] in ("cheap_estimate", "azeotropic", "continuation"):
                # Earlier logs stored the last retry; retain originals while
                # reporting the best completed main-MESH residual accurately.
                values = [a["residual_norm"] for a in row["attempts"] if a.get("residual_norm") is not None]
                residual = min(values) if values else residual
            residual_text = f"{residual:.3e}" if residual is not None else "unavailable"
            lines.append(f"| {row['case']} | {row['factor']:g} | {row['initializer']} | {row['method']} | "
                         f"{row['success']} | {residual_text} | {row['total_jacobians']} | {row['solve_seconds']:.4f} | "
                         f"{row.get('locally_unstable_stages', 'not checked')} |")
    with (output / "summary.md").open("x") as handle:
        handle.write("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Write a new summary.md from existing records, without solving")
    parser.add_argument("--audit-directory", type=Path,
                        help="Check phases in saved successes instead of running new column solves")
    parser.add_argument("--factors", nargs="+", type=float, default=[.98, 1.005, 1.02, 1.05])
    parser.add_argument("--initializers", nargs="+", choices=("cheap_estimate", "azeotropic", "cmo", "auto", "coarse_rigorous", "continuation"),
                        default=["cheap_estimate", "azeotropic"])
    parser.add_argument("--methods", nargs="+", choices=("line", "dogleg", "subspace"), default=["line", "dogleg", "subspace"])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--native-retries", action="store_true",
                        help="Retain native finite-difference step retries and colored-Jacobian fallback")
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args.output)
        return
    if args.budget < 1 or any(not math.isfinite(f) or f <= 0 for f in args.factors):
        parser.error("budget and finite factors must be positive")
    if "continuation" in args.initializers and args.factors != sorted(args.factors):
        parser.error("continuation requires increasing cut factors")
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"args": vars(args), "python": sys.version, "platform": platform.platform(),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "hash_seed": 0, "random_seed": 20261007, "blas_threads": 1}
    manifest["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        "equilibrium_stage_column.py", "unit_operations_distillation.py", "thermodynamics_models/activity.py",
        "scripts/performance/benchmark_column_trust_region.py", "scripts/performance/probe_azeotropic_initializer_matrix.py",
        str(Path(__file__).relative_to(ROOT)))}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    for name in (Path(__file__).name, "benchmark_column_trust_region.py", "probe_azeotropic_initializer_matrix.py"):
        (args.output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    for case in args.cases:
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output), "--worker", case,
                   "--budget", str(args.budget), "--factors", *map(str, args.factors),
                   "--initializers", *args.initializers, "--methods", *args.methods]
        if args.inspect:
            command.append("--inspect")
        if args.audit_directory:
            command.extend(["--audit-directory", str(args.audit_directory)])
        if args.native_retries:
            command.append("--native-retries")
        print("START", case, flush=True)
        with (args.output / f"{case}.log").open("x") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print("DONE", case, completed.returncode, flush=True)
        if completed.returncode:
            raise SystemExit(f"Worker failed ({completed.returncode}); inspect {case}.log")
    summarize(args.output)


if __name__ == "__main__":
    main()
