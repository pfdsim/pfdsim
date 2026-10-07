#!/usr/bin/env python3
"""Compare native column line search with script-only trust-region solvers.

Run with the repository's Python environment, e.g.:
  .venv/bin/python scripts/performance/benchmark_column_trust_region.py \
      --output /tmp/column-trust-native --repeats 3
  .venv/bin/python scripts/performance/benchmark_column_trust_region.py \
      --output /tmp/column-trust-stress --mode raw --noise .5 1.5 \
      --seeds 11 29 47 --cases partial_mass_12 absorption_5 stripping_5 cmo_20

Raw mode stops after the first nonlinear problem (no initializer/Jacobian
fallback rescue). End-to-end mode retains all production orchestration,
including VLLE events and post-solve checks. Noise is seeded additive Gaussian
noise in the native transformed variables, only on the first nonlinear call.
End-to-end jobs run in fresh sequential processes, with single-threaded BLAS.
Raw probes reuse one captured model per case to avoid repeated initialization;
methods rotate order, and initial vectors are identical for each paired start.
Every job has an incremental log and exclusive JSON result; the controller appends
JSONL records as jobs complete. Output directories must be new. No production
files are modified. Case construction is reused from the existing benchmark.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
from statistics import median
import subprocess
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_sparse_column_solves import CASES, install_timers, prepare_case, stream_record

METHODS = ("line", "dogleg", "dogleg_scaled", "subspace", "trf", "trf_unscaled")


class RawDone(BaseException):
    """Exit the public solve without triggering its initializer fallbacks."""


class BudgetDone(Exception):
    pass


def newton_direction(J, f):
    """Use the native solver's same sparse solves and diagonal shifts."""
    import numpy as np
    from scipy.sparse import csc_matrix, eye
    from scipy.sparse.linalg import MatrixRankWarning, spsolve

    matrix = csc_matrix(J)
    for shift in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                dx = spsolve(matrix if not shift else matrix + shift * eye(
                    *matrix.shape, format="csc"), -f)
            if np.all(np.isfinite(dx)):
                return dx
        except (RuntimeError, ValueError, MatrixRankWarning):
            pass
    return None


def dogleg_step(gauss_newton, gradient, J_scaled, radius):
    """Cauchy-to-Newton path inside a Euclidean trust region."""
    import numpy as np

    if gauss_newton is not None:
        Jp = J_scaled @ gauss_newton
        predicted = -float(gradient @ gauss_newton) - .5 * float(Jp @ Jp)
        # Dogleg assumes a Gauss-Newton minimizer. Native diagonal shifts
        # can violate that assumption; retain the valid Cauchy direction.
        if not math.isfinite(predicted) or predicted <= 0:
            gauss_newton = None
    if gauss_newton is not None and np.linalg.norm(gauss_newton) <= radius:
        # A shifted Newton solve can cease to be a descent direction.
        if np.dot(gradient, gauss_newton) < 0:
            return gauss_newton
    gg = float(gradient @ gradient)
    Jg = J_scaled @ gradient
    denominator = float(Jg @ Jg)
    if gg == 0 or not math.isfinite(gg) or denominator <= 0:
        return None
    cauchy = -(gg / denominator) * gradient
    cauchy_norm = float(np.linalg.norm(cauchy))
    if cauchy_norm >= radius:
        return -(radius / math.sqrt(gg)) * gradient
    if gauss_newton is None or np.dot(gradient, gauss_newton) >= 0:
        return cauchy
    difference = gauss_newton - cauchy
    a = float(difference @ difference)
    b = float(cauchy @ difference)
    c = float(cauchy @ cauchy) - radius**2
    tau = (-b + math.sqrt(max(b*b - a*c, 0.0))) / a
    return cauchy + tau * difference


def subspace_step(gauss_newton, gradient, J_scaled, radius):
    """Minimize the quadratic model in span(gradient, sparse Newton step).

    This retains sparse direct linear solves but, unlike dogleg, minimizes
    over the entire two-dimensional subspace. Only the small projected
    Hessian is dense. A scalar damping root enforces the trust radius.
    """
    import numpy as np
    from scipy.optimize import brentq

    gnorm = np.linalg.norm(gradient)
    if not math.isfinite(gnorm) or gnorm == 0:
        return None
    directions = [gradient / gnorm]
    if gauss_newton is not None and np.linalg.norm(gauss_newton) > 0:
        directions.append(gauss_newton / np.linalg.norm(gauss_newton))
    Q, singular, _ = np.linalg.svd(np.column_stack(directions), full_matrices=False)
    Q = Q[:, singular > singular[0] * 1e-12]
    A = J_scaled @ Q
    eigenvalues, eigenvectors = np.linalg.eigh(A.T @ A)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_gradient = eigenvectors.T @ (Q.T @ gradient)

    def coefficients(damping):
        denominator = eigenvalues + damping
        return -projected_gradient / np.maximum(denominator, np.finfo(float).tiny)

    unconstrained = coefficients(0.0)
    if np.linalg.norm(unconstrained) <= radius:
        return Q @ (eigenvectors @ unconstrained)
    upper = np.linalg.norm(projected_gradient) / radius
    damping = brentq(lambda value: np.linalg.norm(coefficients(value)) - radius,
                    0.0, upper, xtol=max(upper * 1e-14, 1e-300))
    return Q @ (eigenvectors @ coefficients(damping))


def trust_solve(unit, residual, sparsity, x0, options, jacobian, step_event,
                method, stats, radius_factor=1.0, lsmr_tolerance=1e-8):
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.sparse import csr_matrix
    from thermodynamics_models.common import ThermodynamicsError

    x = np.array(x0, dtype=float)
    f = residual(x)
    tolerance = options["mesh_tolerance"]
    acceptable = max(tolerance, options.get("acceptable_mesh_residual", tolerance))
    max_jacobians = min(options["max_jacobian_evaluations"], options["max_iterations"])
    groups = None
    njev, nfev, iterations = 0, 1, 0
    jacobian_method = "colored_finite_difference"
    message = "maximum Jacobian evaluations reached"
    current = [x, f]

    def build_jacobian(vector, value):
        nonlocal groups, njev, nfev, jacobian_method
        if njev >= max_jacobians:
            raise BudgetDone
        start = time.perf_counter()
        njev += 1
        try:
            result = jacobian(vector, value, options["finite_difference_rel_step"]) if jacobian else None
            if result is None:
                if groups is None:
                    groups = unit._color_jacobian_columns(sparsity)
                J, evaluations = unit._finite_difference_jacobian(
                    residual, vector, value, sparsity, groups,
                    options["finite_difference_rel_step"])
                jacobian_method = "colored_finite_difference"
            else:
                J, evaluations, *label = result
                jacobian_method = label[0] if label else "semi_analytic_flow"
            nfev += evaluations
            return csr_matrix(J)
        finally:
            stats["jacobian_seconds"] += time.perf_counter() - start

    def event(vector, value, direction):
        if step_event and direction is not None:
            step_event(vector, value, direction)

    try:
        if not np.all(np.isfinite(f)):
            raise ValueError("non-finite initial residual")
        if method in ("trf", "trf_unscaled") and np.linalg.norm(f, ord=np.inf) >= tolerance:
            # Keep the exact accepted state, rather than the latest FD/trial point.
            cache = [x.copy(), f.copy()]

            def fun(vector):
                nonlocal nfev
                nfev += 1
                try:
                    value = residual(vector)
                except ThermodynamicsError:
                    # A physical-domain failure at a trial point contracts
                    # the radius. Initial residual/Jacobian errors propagate.
                    stats["invalid_domain_trials"] = stats.get("invalid_domain_trials", 0) + 1
                    value = np.full_like(f, np.nan)
                cache[:] = [vector.copy(), value.copy()]
                return value

            def jac(vector):
                value = cache[1] if np.array_equal(vector, cache[0]) else fun(vector)
                current[:] = [vector.copy(), value.copy()]
                if np.linalg.norm(value, ord=np.inf) < tolerance:
                    raise BudgetDone
                J = build_jacobian(vector, value)
                # VLLE hooks require the untruncated Newton prediction, just as
                # they do in the native solver; preserve that interface.
                if step_event:
                    event(vector, value, newton_direction(J, value))
                return J

            def callback(intermediate_result):
                nonlocal iterations
                iterations += 1
                current[:] = [intermediate_result.x.copy(), intermediate_result.fun.copy()]
                if np.linalg.norm(intermediate_result.fun, ord=np.inf) < tolerance:
                    raise StopIteration

            result = least_squares(
                fun, x, jac=jac, method="trf", tr_solver="lsmr",
                x_scale=1.0 if method == "trf_unscaled" else "jac",
                tr_options={"atol": lsmr_tolerance, "btol": lsmr_tolerance},
                ftol=None, xtol=1e-12, gtol=None,
                max_nfev=options["max_iterations"] * options["line_search_steps"],
                callback=callback,
            )
            x, f = result.x, result.fun
            message = str(result.message)
        elif method not in ("trf", "trf_unscaled"):
            inverse_scale = np.ones_like(x)
            radius = radius_factor * math.sqrt(x.size)
            radius_max = 8.0 * math.sqrt(x.size)
            stall_best, stall_count = math.inf, 0
            for _ in range(options["max_iterations"]):
                norm = float(np.linalg.norm(f, ord=np.inf))
                if norm < tolerance:
                    message = "converged"
                    break
                if options.get("stall_iterations", 0):
                    progress = max(options.get("stall_relative_tolerance", 1e-4) * stall_best, 1e-12)
                    if not math.isfinite(stall_best) or norm < stall_best - progress:
                        stall_count = 0
                    else:
                        stall_count += 1
                    stall_best = min(stall_best, norm)
                    if stall_count >= options["stall_iterations"]:
                        message = "residual stalled"
                        break
                J = build_jacobian(x, f)
                dx = newton_direction(J, f)
                event(x, f, dx)
                if method == "dogleg_scaled":
                    column_norm = np.sqrt(np.asarray(J.power(2).sum(axis=0)).ravel())
                    # Same monotone Jacobian scaling convention as SciPy.
                    if njev == 1:
                        inverse_scale = np.where(column_norm > 0, column_norm, 1.0)
                    else:
                        inverse_scale = np.maximum(inverse_scale, column_norm)
                scale = 1.0 / inverse_scale
                J_scaled = J.multiply(scale).tocsr()
                gradient = np.asarray(J_scaled.T @ f).ravel()
                gn = dx * inverse_scale if dx is not None else None
                merit = 0.5 * float(f @ f)
                accepted = False
                for _ in range(options["line_search_steps"]):
                    step = subspace_step if method == "subspace" else dogleg_step
                    p = step(gn, gradient, J_scaled, radius)
                    if p is None:
                        message = "zero or invalid gradient"
                        break
                    Jp = J_scaled @ p
                    predicted = -float(gradient @ p) - 0.5 * float(Jp @ Jp)
                    if predicted <= 0 or not math.isfinite(predicted):
                        message = "nonpositive model reduction"
                        break
                    trial_x = x + scale * p
                    try:
                        trial_f = residual(trial_x)
                    except ThermodynamicsError:
                        stats["invalid_domain_trials"] = stats.get("invalid_domain_trials", 0) + 1
                        trial_f = np.full_like(f, np.nan)
                    nfev += 1
                    actual = merit - 0.5 * float(trial_f @ trial_f)
                    ratio = actual / predicted if np.all(np.isfinite(trial_f)) else -math.inf
                    step_norm = float(np.linalg.norm(p))
                    stats["trust_trials"].append({"radius": radius,
                                                 "ratio": ratio if math.isfinite(ratio) else None,
                                                 "step_norm": step_norm, "accepted": ratio > .1})
                    if ratio < .25:
                        radius = .25 * step_norm
                    elif ratio > .75 and step_norm > .95 * radius:
                        radius = min(2 * radius, radius_max)
                    if ratio > .1:
                        x, f = trial_x, trial_f
                        current[:] = [x, f]
                        iterations += 1
                        accepted = True
                        break
                    stats["rejected_trials"] += 1
                    if radius < 1e-12:
                        message = "trust radius underflow"
                        break
                if not accepted:
                    if message == "maximum Jacobian evaluations reached":
                        message = "trust region could not reduce the residual"
                    break
    except BudgetDone:
        x, f = current
        message = "maximum Jacobian evaluations reached"
    except Exception as exc:
        stats.update(function_evaluations=nfev, jacobian_evaluations=njev,
                     residual_norm=float(np.linalg.norm(current[1], ord=np.inf)))
        progress = getattr(exc, "add_solver_progress", None)
        if callable(progress):
            progress(iterations=iterations, function_evaluations=nfev, jacobian_evaluations=njev)
        raise
    norm = float(np.linalg.norm(f, ord=np.inf))
    return {"success": bool(norm < acceptable), "x": x, "residual_norm": norm,
            "iterations": iterations, "function_evaluations": nfev,
            "jacobian_evaluations": njev, "jacobian_method": jacobian_method,
            "message": "converged" if norm < tolerance else message}


def install(args, attempts, jacobian_stats):
    import numpy as np
    from equilibrium_stage_column import EquilibriumStageColumnMixin
    from unit_operations_separation import RigorousLiquidLiquidExtractor

    rng = np.random.default_rng(args.seed)
    first = True
    for cls in (EquilibriumStageColumnMixin, RigorousLiquidLiquidExtractor):
        original = cls._sparse_newton_solve

        def factory(native):
            @wraps(native)
            def wrapped(unit, residual, sparsity, x0, options, jacobian=None, step_event=None):
                nonlocal first
                vector = np.array(x0, dtype=float)
                if first:
                    if args.noise_value:
                        vector += args.noise_value * rng.standard_normal(vector.size)
                    first = False
                entry = {"size": vector.size, "options": dict(options),
                         "initial_x": vector.tolist(),
                         "x0_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                         "jacobian_seconds": 0.0, "rejected_trials": 0, "trust_trials": []}
                attempts.append(entry)
                context = getattr(unit.thermo, "quality_context", None)
                prior = getattr(unit, "_quality_solver_aux_context_active", False)
                unit._quality_solver_aux_context_active = True
                start = time.perf_counter()
                initial_jacobian_seconds = jacobian_stats["jacobian_seconds"]
                try:
                    with context(phase="solver_iteration", affects_result=False) if context else nullcontext():
                        if args.method == "line":
                            keywords = {"jacobian": jacobian}
                            if step_event is not None:
                                keywords["step_event"] = step_event
                            solution = native(unit, residual, sparsity, vector, options, **keywords)
                        else:
                            solution = trust_solve(unit, residual, sparsity, vector, options,
                                                   jacobian, step_event, args.method, entry,
                                                   args.radius_factor, args.lsmr_tolerance)
                        entry.update({key: value for key, value in solution.items() if key != "x"})
                        entry["strict_success"] = solution["residual_norm"] < options["mesh_tolerance"]
                except Exception as exc:
                    entry.update(success=False, strict_success=False,
                                 error=f"{type(exc).__name__}: {exc}")
                    for key in ("function_evaluations", "jacobian_evaluations", "residual_norm"):
                        if hasattr(exc, key):
                            entry[key] = getattr(exc, key)
                    raise
                finally:
                    if args.method == "line":
                        entry["jacobian_seconds"] = jacobian_stats["jacobian_seconds"] - initial_jacobian_seconds
                    entry["solver_seconds"] = time.perf_counter() - start
                    unit._quality_solver_aux_context_active = prior
                if args.mode == "raw":
                    raise RawDone
                return solution
            return wrapped
        cls._sparse_newton_solve = factory(original)


def raw_batch(args):
    """Replay the identical first nonlinear problem, without unit fallbacks."""
    import numpy as np
    import scipy
    import scipy.optimize
    import scipy.sparse.linalg
    from equilibrium_stage_column import EquilibriumStageColumnMixin
    from unit_operations_separation import RigorousLiquidLiquidExtractor

    np.random.seed(args.seed)
    random.seed(args.seed)
    jacobian_stats = install_timers()
    captured = []
    for cls in (EquilibriumStageColumnMixin, RigorousLiquidLiquidExtractor):
        original = cls._sparse_newton_solve

        def factory(native):
            def capture(unit, residual, sparsity, x0, options, jacobian=None, step_event=None):
                captured.append((native, unit, residual, sparsity, x0, options, jacobian, step_event))
                raise RawDone
            return capture
        cls._sparse_newton_solve = factory(original)
    unit, inlets = prepare_case(args.worker)
    try:
        unit.solve(inlets)
    except RawDone:
        pass
    if not captured:
        raise ValueError("case never entered the sparse Newton solver")
    native, unit, residual, sparsity, x0, options, jacobian, step_event = captured[0]
    if step_event is not None:
        raise ValueError("use end-to-end mode for stateful VLLE topology/projection events")
    context = getattr(unit.thermo, "quality_context", None)
    unit._quality_solver_aux_context_active = True
    with args.result.open("x") as handle:
        number = 0
        for repeat in range(1, args.repeats + 1):
            for noise in args.noise:
                for seed in args.seeds:
                    number += 1
                    vector = np.array(x0, dtype=float)
                    if noise:
                        vector += noise * np.random.default_rng(seed).standard_normal(vector.size)
                    methods = args.methods[number % len(args.methods):] + args.methods[:number % len(args.methods)]
                    for method in methods:
                        entry = {"size": vector.size, "options": dict(options), "jacobian_seconds": 0.0,
                                 "initial_x": vector.tolist(),
                                 "x0_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                                 "rejected_trials": 0, "trust_trials": []}
                        record = {"case": args.worker, "method": method, "mode": "raw", "noise": noise,
                                  "seed": seed, "repeat": repeat, "numpy": np.__version__,
                                  "scipy": scipy.__version__, "attempts": [entry],
                                  "radius_factor": args.radius_factor, "lsmr_tolerance": args.lsmr_tolerance}
                        start = time.perf_counter()
                        initial_jacobian_seconds = jacobian_stats["jacobian_seconds"]
                        try:
                            with context(phase="solver_iteration", affects_result=False) if context else nullcontext():
                                if method == "line":
                                    solution = native(unit, residual, sparsity, vector, options, jacobian=jacobian)
                                else:
                                    solution = trust_solve(unit, residual, sparsity, vector, options, jacobian,
                                                           None, method, entry, args.radius_factor,
                                                           args.lsmr_tolerance)
                            entry.update({k: v for k, v in solution.items() if k != "x"})
                            strict = solution["residual_norm"] < options["mesh_tolerance"]
                            entry["strict_success"] = strict
                            record.update(success=solution["success"], strict_success=strict,
                                          residual_norm=solution["residual_norm"])
                        except Exception as exc:
                            entry.update(success=False, strict_success=False)
                            record.update(success=False, strict_success=False,
                                          error=f"{type(exc).__name__}: {exc}")
                            traceback.print_exc()
                        entry["solver_seconds"] = time.perf_counter() - start
                        if method == "line":
                            entry["jacobian_seconds"] = jacobian_stats["jacobian_seconds"] - initial_jacobian_seconds
                        record.update(solve_seconds=entry["solver_seconds"], solver_seconds=entry["solver_seconds"],
                                      jacobian_evaluations=entry.get("jacobian_evaluations", 0),
                                      function_evaluations=entry.get("function_evaluations", 0))
                        handle.write(json.dumps(record, default=lambda obj: obj.item() if hasattr(obj, "item") else str(obj)) + "\n")
                        handle.flush()
                        print(args.worker, method, noise, seed, record["success"], flush=True)


def worker(args):
    import numpy as np
    import scipy
    # Import numerical machinery before timing for every method, including
    # line search. Otherwise lazy import cost penalizes trust-region methods.
    import scipy.optimize
    import scipy.sparse.linalg

    np.random.seed(args.seed)
    random.seed(args.seed)
    attempts = []
    record = {"case": args.worker, "method": args.method, "mode": args.mode,
              "noise": args.noise_value, "seed": args.seed, "repeat": args.repeat,
              "radius_factor": args.radius_factor, "lsmr_tolerance": args.lsmr_tolerance,
              "numpy": np.__version__, "scipy": scipy.__version__, "attempts": attempts}
    start = time.perf_counter()
    try:
        unit, inlets = prepare_case(args.worker)
        record["setup_seconds"] = time.perf_counter() - start
        record["inputs"] = {p: stream_record(s) for p, s in inlets.items()}
        install(args, attempts, install_timers())
        start = time.perf_counter()
        result = unit.solve(inlets)
        record.update(success=True, solve_seconds=time.perf_counter() - start,
                      performance=result.performance, warnings=result.warnings,
                      heat_duty=result.heat_duty,
                      outputs={p: stream_record(s) for p, s in result.outlet_streams.items()})
        record["residual_norm"] = result.performance.get(
            "mesh_residual", result.performance.get("mes_residual"))
        if attempts and record["residual_norm"] is not None:
            record["strict_success"] = record["residual_norm"] < attempts[-1]["options"]["mesh_tolerance"]
        record["newton_solver_used"] = bool(attempts)
        components = set().union(*(s.composition for s in inlets.values()),
                                 *(s.composition for s in result.outlet_streams.values()))
        feed_flow = max(sum(s.F for s in inlets.values()), 1.0)
        record["component_balance_error"] = max(abs(
            sum(s.F * s.composition.get(c, 0) for s in inlets.values()) -
            sum(s.F * s.composition.get(c, 0) for s in result.outlet_streams.values())
        ) / feed_flow for c in components)
        inlet_energy = sum(s.F * (s.H or 0.0) for s in inlets.values())
        outlet_energy = sum(s.F * (s.H or 0.0) for s in result.outlet_streams.values())
        record["external_energy_relative_error"] = abs(outlet_energy - inlet_energy - result.heat_duty) / max(
            abs(inlet_energy), abs(outlet_energy), abs(result.heat_duty), 1.0)
    except RawDone:
        record.update(success=attempts[-1]["success"],
                      strict_success=attempts[-1]["strict_success"],
                      solve_seconds=time.perf_counter() - start)
        record["residual_norm"] = attempts[-1]["residual_norm"]
    except Exception as exc:
        record.update(success=False, error=f"{type(exc).__name__}: {exc}",
                      solve_seconds=time.perf_counter() - start)
        traceback.print_exc()
    record["solver_seconds"] = sum(a["solver_seconds"] for a in attempts)
    record["jacobian_evaluations"] = sum(a.get("jacobian_evaluations", 0) for a in attempts)
    record["function_evaluations"] = sum(a.get("function_evaluations", 0) for a in attempts)
    with args.result.open("x") as handle:
        json.dump(record, handle, default=lambda obj: obj.item() if hasattr(obj, "item") else str(obj))
    print(args.worker, args.method, record["success"], f"{record['solve_seconds']:.3f}s", flush=True)


def summarize(output, warmup_repeats=0):
    all_records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    records = [r for r in all_records if r["repeat"] > warmup_repeats]
    lines = ["# Column trust-region comparison", "",
             "Single BLAS thread; setup excluded. End-to-end uses fresh sequential processes; raw reuses a model per case. All raw repetitions retained.",
             f"Repetitions <= {warmup_repeats} excluded from this table as warm-up.",
             "Strict means the requested infinity-norm residual; acceptable uses the native relaxed threshold.", "",
             "| Case | Noise | Method | Acceptable / runs | Strict / runs | Median solve s | Median solver s | Median J | Median f | Max final residual |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, noise in dict.fromkeys((r["case"], r["noise"]) for r in records):
        for method in dict.fromkeys(r["method"] for r in records):
            runs = [r for r in records if (r["case"], r["noise"], r["method"]) == (case, noise, method)]
            if not runs:
                continue
            successes = [r for r in runs if r["success"]]
            strict = sum(r.get("strict_success", False) for r in successes)
            finals = [r.get("residual_norm", r["attempts"][-1].get("residual_norm", math.nan)) for r in runs if r["attempts"]]
            def med(key):
                values = [r[key] for r in runs if r.get(key) is not None]
                return median(values) if values else math.nan
            lines.append(f"| {case} | {noise:g} | {method} | {len(successes)}/{len(runs)} | {strict}/{len(runs)} | "
                         f"{med('solve_seconds'):.4f} | {med('solver_seconds'):.4f} | "
                         f"{med('jacobian_evaluations'):g} | {med('function_evaluations'):g} | {max(finals, default=math.nan):.3e} |")
    lines += ["", "## Failures", ""]
    for r in records:
        if not r["success"]:
            last = r["attempts"][-1] if r["attempts"] else {}
            lines.append(f"- {r['case']} / {r['method']} / noise={r['noise']} / seed={r['seed']}: "
                         f"{r.get('error', last.get('message'))}; residual={last.get('residual_norm')}")
    with (output / "summary.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:-1]),
                        help="Unscaled TRF is opt-in because its PR-column probe is very expensive")
    parser.add_argument("--mode", choices=("end_to_end", "raw"), default="end_to_end")
    parser.add_argument("--noise", nargs="+", type=float, default=[0.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261006])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--radius-factor", type=float, default=1.0)
    parser.add_argument("--lsmr-tolerance", type=float, default=1e-8)
    parser.add_argument("--job-timeout", type=float, default=180.0)
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--batch-raw", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--noise-value", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.cases is None:
        args.cases = [case for case in CASES if args.mode != "raw" or case != "vlle_butanol_20"]
    if not args.worker and args.mode == "raw" and "vlle_butanol_20" in args.cases:
        parser.error("VLLE topology events require --mode end_to_end")
    if args.repeats < 1 or not 0 <= args.warmup_repeats < args.repeats:
        parser.error("require repeats >= 1 and 0 <= warmup-repeats < repeats")
    if not math.isfinite(args.radius_factor) or not 0 < args.radius_factor <= 8:
        parser.error("radius-factor must be finite and in (0, 8]")
    if not math.isfinite(args.lsmr_tolerance) or args.lsmr_tolerance < 0:
        parser.error("lsmr-tolerance must be finite and nonnegative")
    if any(not math.isfinite(value) or value < 0 for value in args.noise):
        parser.error("noise values must be finite and nonnegative")
    if any(seed < 0 for seed in args.seeds) or args.seed < 0:
        parser.error("seeds must be nonnegative")
    if not math.isfinite(args.job_timeout) or args.job_timeout <= 0:
        parser.error("job-timeout must be finite and positive")
    if args.batch_raw:
        raw_batch(args)
        return
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"python": sys.version, "platform": platform.platform(), "arguments": vars(args),
                "python_hash_seed": 0,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                    "equilibrium_stage_column.py", "equilibrium_stage_vlle.py",
                    "unit_operations_distillation.py", "unit_operations_separation.py",
                    "scripts/performance/benchmark_sparse_column_solves.py",
                    str(Path(__file__).relative_to(ROOT)))}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (args.output / "benchmark_script.py").write_bytes(Path(__file__).read_bytes())
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    number = 0
    with (args.output / "results.jsonl").open("x") as aggregate:
        if args.mode == "raw":
            for case in args.cases:
                result = args.output / f"{case}.jsonl"
                command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                           "--batch-raw", "--worker", case, "--result", str(result),
                           "--repeats", str(args.repeats), "--radius-factor", str(args.radius_factor),
                           "--lsmr-tolerance", str(args.lsmr_tolerance), "--methods", *args.methods,
                           "--noise", *map(str, args.noise), "--seeds", *map(str, args.seeds)]
                print(f"START raw batch {case}", flush=True)
                with (args.output / f"{case}.log").open("x") as log:
                    completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                               stderr=subprocess.STDOUT)
                if result.exists():
                    aggregate.write(result.read_text())
                    aggregate.flush()
                if completed.returncode:
                    raise SystemExit(f"Raw batch failed ({completed.returncode}); see {case}.log")
                print(f"DONE raw batch {case}", flush=True)
            summarize(args.output, args.warmup_repeats)
            print(f"Report: {args.output / 'summary.md'}", flush=True)
            return
        for repeat in range(1, args.repeats + 1):
            methods = args.methods[repeat % len(args.methods):] + args.methods[:repeat % len(args.methods)]
            for case in args.cases:
                for noise in args.noise:
                    for seed in args.seeds:
                        for method in methods:
                            number += 1
                            stem = f"{number:04d}-{case}-{method}-n{noise:g}-s{seed}-r{repeat}"
                            result = args.output / f"{stem}.json"
                            command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                                       "--worker", case, "--method", method, "--mode", args.mode,
                                       "--noise-value", str(noise), "--seed", str(seed), "--repeat", str(repeat),
                                       "--radius-factor", str(args.radius_factor),
                                       "--lsmr-tolerance", str(args.lsmr_tolerance), "--result", str(result)]
                            print(f"START {stem}", flush=True)
                            with (args.output / f"{stem}.log").open("x") as log:
                                try:
                                    completed = subprocess.run(command, cwd=ROOT, env=environment,
                                                               stdout=log, stderr=subprocess.STDOUT,
                                                               timeout=args.job_timeout)
                                    code = completed.returncode
                                except subprocess.TimeoutExpired:
                                    code = "self-imposed timeout"
                            if result.exists():
                                record = json.loads(result.read_text())
                            else:
                                record = {"case": case, "method": method, "noise": noise, "seed": seed,
                                          "repeat": repeat, "mode": args.mode, "success": False, "attempts": [],
                                          "error": f"worker exited without result: {code}",
                                          "solve_seconds": args.job_timeout, "solver_seconds": args.job_timeout,
                                          "jacobian_evaluations": None, "function_evaluations": None}
                            aggregate.write(json.dumps(record) + "\n")
                            aggregate.flush()
                            print(f"DONE {stem} success={record['success']} {record['solve_seconds']:.3f}s", flush=True)
                            if code in (137, -9, 143, -15):
                                raise SystemExit(f"Worker terminated ({code}); stopping experiment.")
    summarize(args.output, args.warmup_repeats)
    print(f"Report: {args.output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
