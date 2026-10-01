#!/usr/bin/env python3
"""Script-only experiment: analytic NRTL HE fused with activity evaluation.

From the repository root:
    set -o pipefail
    python scripts/performance/benchmark_fused_nrtl_excess_enthalpy.py \
        --output /tmp/pfdsim-fused-he \
        2>&1 | tee /tmp/pfdsim-fused-he.log

No production changes. The prototype differentiates the NRTL weighted sums,
supports every existing temperature/extrapolation mode, and respects activity
clipping. It patches only each experimental thermo instance. The MESH solver
and its outer finite-difference Jacobian are unchanged. "analytic" computes
the fused result for HE but discards gamma; "fused" reuses both through one shared
bounded state cache and mirrors the existing activity cache. The kernel
comparison has no Python dispatch or property caches.
Full solves use fresh instances and rotate variant order. One warm-up per case
and variant is excluded; imports/initial compilation are excluded as well.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from itertools import combinations
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from numba import njit

from compiled_activity import (
    _nrtl_activity_coefficients_numba as original_gamma,
    _nrtl_excess_enthalpy_numba as original_he,
)
from chemical_properties import ChemicalDatabase
from physical_constants import R_J_MOL_K
from simulator import Simulator
from thermodynamics import create_thermodynamics
from unit_operations_distillation import RigorousDistillation

SEED = 20260930
VARIANTS = ("original", "analytic", "fused")
CASES = ("benzene_toluene_20", "partial_mass_12", "ternary_20")
LONG_CASES = ("ternary_200", "junk_8_60", "junk_12_60")


@njit
def fused_nrtl(x, T, tau_mode, c, d, e, f, g, tref, energy, alpha, tmin, tmax):
    """Return gamma and analytic HE [kJ/kmol] using one set of NRTL matrices."""
    n = len(x)
    gamma = np.ones(n)
    total = 0.0
    for i in range(n):
        total += max(x[i], 0.0)
    if total <= 0:
        return gamma, 0.0
    xn = np.maximum(x, 0.0)/total
    tau = np.zeros((n, n))
    slope = np.zeros((n, n))
    G = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mode = tau_mode[i, j] % 10
            extrap = tau_mode[i, j] // 10
            b = T
            if extrap > 0:
                b = min(max(T, tmin[i, j]), tmax[i, j])
            if mode == 1:
                ref = tref[i, j]
                v = (c[i, j]+d[i, j]/b
                     +e[i, j]*((ref-b)/b+math.log(b/ref))
                     +f[i, j]*b+g[i, j]*b*b)
                dv = (-d[i, j]/(b*b)+e[i, j]*(b-ref)/(b*b)
                      +f[i, j]+2*g[i, j]*b)
                if b != T:
                    if extrap == 1:
                        dv = 0.0
                    elif extrap == 2:
                        vb = -b*b*dv
                        va = v+b*dv
                        v = va+vb/T
                        dv = -vb/(T*T)
                    elif extrap == 3:
                        vb = b*(2*v+b*dv)
                        vc = -b*b*(v+b*dv)
                        v = vb/T+vc/(T*T)
                        dv = -vb/(T*T)-2*vc/(T**3)
                    else:
                        vc = b*b*(3*v+b*dv)
                        vd = -(b**3)*(2*v+b*dv)
                        v = vc/(T*T)+vd/(T**3)
                        dv = -2*vc/(T**3)-3*vd/(T**4)
            elif mode == 2:
                # The existing energy/T mode clamps only for policy 1.
                b = min(max(T, tmin[i, j]), tmax[i, j]) if extrap == 1 else T
                v = energy[i, j]/b
                dv = -energy[i, j]/(b*b) if b == T else 0.0
            else:
                v, dv = 0.0, 0.0
            tau[i, j] = v
            slope[i, j] = dv
            G[i, j] = math.exp(-alpha[i, j]*v)
    denom = np.empty(n)
    denom_slope = np.empty(n)
    weighted = np.empty(n)
    weighted_slope = np.empty(n)
    for j in range(n):
        s, ds, a, da = 0.0, 0.0, 0.0, 0.0
        for i in range(n):
            z = xn[i]*G[i, j]
            dz = -alpha[i, j]*slope[i, j]*z
            s += z
            ds += dz
            a += z*tau[i, j]
            da += dz*tau[i, j]+z*slope[i, j]
        if abs(s) < 1e-30:
            s, ds = 1e-30, 0.0
        denom[j], denom_slope[j] = s, ds
        weighted[j] = a/s
        weighted_slope[j] = (da-weighted[j]*ds)/s
    # GE/(RT) = sum_i x_i weighted_i; HE = -RT^2 d(GE/RT)/dT.
    he_derivative = 0.0
    clipped = False
    for i in range(n):
        he_derivative += xn[i]*weighted_slope[i]
        ln = weighted[i]
        for j in range(n):
            ln += xn[j]*G[i, j]/denom[j]*(tau[i, j]-weighted[j])
        gamma[i] = max(math.exp(min(max(ln, -50.0), 50.0)), 1e-12)
        if ln <= math.log(1e-12) or ln >= 50.0:
            clipped = True
    if clipped:
        # GE identity assumes unclipped activities. At clipping, differentiate
        # the actual returned ln(gamma), matching the current HE definition.
        he_derivative = 0.0
        for i in range(n):
            ln, derivative = weighted[i], weighted_slope[i]
            for j in range(n):
                z = xn[j]*G[i, j]/denom[j]
                dz = z*(-alpha[i, j]*slope[i, j]-denom_slope[j]/denom[j])
                ln += z*(tau[i, j]-weighted[j])
                derivative += (dz*(tau[i, j]-weighted[j])
                               +z*(slope[i, j]-weighted_slope[j]))
            if math.log(1e-12) < ln < 50.0:
                he_derivative += xn[i]*derivative
    return gamma, -R_J_MOL_K*T*T*he_derivative


@njit
def kernel_loop(states, temperatures, repeats, variant, params):
    checksum = 0.0
    for k in range(repeats):
        i = k % len(states)
        x, T = states[i], temperatures[i]
        if variant == 0:
            gamma = original_gamma(x, T, *params)
            he = original_he(x, T, *params)
        elif variant == 1:
            gamma = original_gamma(x, T, *params)
            _, he = fused_nrtl(x, T, *params)
        else:
            gamma, he = fused_nrtl(x, T, *params)
        checksum += gamma[0]+he
    return checksum


def prepare_case(name, estimate_missing=False):
    if name == "benzene_toluene_20":
        sim = Simulator.from_file(ROOT / "examples/benzene_toluene_20_stage_distillation_nrtl.pfd")
        sim.initialize()
        solver = sim.solver
        solver._initialize_streams()
        unit_id = "COL-1"
        return solver.units[unit_id], {
            solver.stream_connections[sid][3]: solver._transition_stream_state(
                sid, solver.streams[sid], solver.unit_thermo_scopes[unit_id]
            ) for sid in solver.unit_inlets[unit_id]
        }
    if name in LONG_CASES:
        if name == "ternary_200":
            comps = ["methanol", "ethanol", "water"]
            z = dict(zip(comps, [0.25, 0.25, 0.5]))
            stages, reflux = 200, 2.0
        else:
            comps = ["methanol", "ethanol", "propanol", "isopropanol",
                     "butanol", "isobutanol", "acetone", "water"]
            if name == "junk_12_60":
                comps += ["pentanol", "hexanol", "2-butanol", "ethyl acetate"]
            z = {c: 0.30/(len(comps)-4) for c in comps}
            z.update(methanol=0.25, ethanol=0.25, acetone=0.15, water=0.05)
            stages, reflux = 60, 4.0
        db = ChemicalDatabase(enable_online=False)
        # Explicit synthetic input data: local textbook aliases omit MW for
        # butanol isomers, and bare pentanol/hexanol lack a resolved CAS identity.
        # Register isolated input records rather than changing database files.
        identities = {"isobutanol": ("78-83-1", 74.12),
                      "2-butanol": ("78-92-2", 74.12),
                      "pentanol": ("71-41-0", 88.15),
                      "hexanol": ("111-27-3", 102.177)}
        for component in comps:
            if component not in identities:
                continue
            cas, mw = identities[component]
            props = db.get_user_component(cas, fetch_online=False)
            if props is None:
                props = db.get_user_component(component, fetch_online=False)
            if props is None:
                raise ValueError(f"No offline properties for {component} / {cas}")
            props = deepcopy(props)
            props.CAS, props.MW = cas, mw
            props.property_sources["MW"] = {"method": "benchmark_component_override",
                                            "source": "explicit synthetic component input"}
            db.chemicals[component] = props
        rules = [{"model": "NRTL", "source": "UNIFAC", "policy": "missing_only",
                  "parameter_order": "source", "alpha": 0.3,
                  "Tmin_K": 293.15, "Tmax_K": 473.15}] if estimate_missing else None
        thermo = create_thermodynamics(comps, "NRTL", db, interaction_estimation=rules)
        if any(p.MW <= 0.0 for p in thermo.props.values()):
            raise ValueError("Synthetic column requires positive molecular weights")
        feed = thermo.calculate_state(298.15, 1.0, 100.0, z, phase="liquid", flash=False)
        params = {"N_stages": stages, "feed_stage": stages//2+1,
                  "reflux_ratio": reflux, "D_to_F": 0.45,
                  "P_condenser": 1.0, "P_drop_per_stage": 0.0,
                  "condenser_type": "total", "initializer": "cheap_estimate",
                  "mesh_tolerance": 1e-7, "max_iterations": 160,
                  "max_jacobian_evaluations": 100}
        return RigorousDistillation(name, thermo, params), {"feed": feed}
    partial = name == "partial_mass_12"
    comps = ["methanol", "water"] if partial else ["methanol", "ethanol", "water"]
    z = dict(zip(comps, [0.4, 0.6] if partial else [0.25, 0.25, 0.5]))
    thermo = create_thermodynamics(comps, "NRTL")
    feed = thermo.calculate_state(298.15, 1.0, 100.0, z, phase="liquid", flash=False)
    params = {"N_stages": 12 if partial else 20, "feed_stage": 7 if partial else 11,
              "reflux_ratio": 2.0, "P_condenser": 1.0, "P_drop_per_stage": 0.0,
              "condenser_type": "partial" if partial else "total",
              "mesh_tolerance": 1e-7, "max_iterations": 100,
              "max_jacobian_evaluations": 60, "initializer": "cheap_estimate"}
    params["D_mass_to_F_mass" if partial else "D_to_F"] = 0.35 if partial else 0.45
    return RigorousDistillation(name, thermo, params), {"feed": feed}


def install(thermo, variant):
    backend = thermo._compiled_activity_backend()
    params = backend.enthalpy_parameters()
    components = thermo.components
    cache = {}

    def key(T, composition):
        return float(T), tuple(float(composition.get(c, 0.0)) for c in components)

    def evaluate(T, composition):
        ck = key(T, composition)
        if len(cache) > 20000:
            cache.clear()
            thermo._activity_cache.clear()
        gamma, he = fused_nrtl(np.asarray(ck[1]), ck[0], *params)
        values = dict(zip(components, gamma.tolist()))
        state = values, float(he)
        cache[ck] = state
        # Preserve the authoritative cache's key shape. The compact shared
        # state key avoids constructing names twice for every property request.
        thermo._activity_cache[(ck[0], tuple(zip(components, ck[1])))] = values
        return state

    def activity(T, composition):
        ck = key(T, composition)
        state = cache.get(ck)
        if state is None:
            thermo._warn_activity_interaction_extrapolation(T)
            state = evaluate(T, composition)
        return dict(state[0])

    def excess(composition, T):
        thermo._warn_activity_interaction_extrapolation(T)
        ck = key(T, composition)
        if variant == "analytic":
            return float(fused_nrtl(np.asarray(ck[1]), ck[0], *params)[1])
        state = cache.get(ck)
        if state is None:
            state = evaluate(T, composition)
        return state[1]

    thermo.excess_enthalpy = excess
    if variant == "fused":
        thermo.activity_coefficients = activity

    def restore():
        # These are overrides on fresh instances; deleting restores class
        # dispatch and breaks the function -> thermo -> function cycles.
        del thermo.excess_enthalpy
        if variant == "fused":
            del thermo.activity_coefficients
    return restore


def validate(params, rng):
    records = []
    tests = [("actual", params, [280.0, 330.0, 380.0, 430.0])]
    # Every parameter representation and continuation policy, on both sides
    # and inside its interval. Avoid exact clamp kinks in derivative checks.
    n = len(params[0])
    for mode in (1, 2):
        for policy in range(5):
            arrays = [a.copy() for a in params]
            arrays[0][:] = mode+10*policy
            arrays[1][:] = rng.uniform(-0.5, 0.5, (n, n))
            arrays[2][:] = rng.uniform(-150, 150, (n, n))
            arrays[3][:] = rng.uniform(-0.5, 0.5, (n, n))
            arrays[4][:] = rng.uniform(-0.002, 0.002, (n, n))
            arrays[5][:] = rng.uniform(-1e-6, 1e-6, (n, n))
            arrays[6][:] = 298.15
            arrays[7][:] = rng.uniform(-150, 150, (n, n))
            arrays[8][:] = 0.3
            arrays[9][:], arrays[10][:] = 300.0, 400.0
            tests.append((f"mode={mode},policy={policy}", tuple(arrays), [250.0, 350.0, 450.0]))
    for sign in (-1, 1):
        arrays = [a.copy() for a in params]
        arrays[0][:], arrays[7][:], arrays[8][:] = 2, sign*30000.0, 0.0
        tests.append((f"activity_clip,sign={sign}", tuple(arrays), [280.0, 350.0, 430.0]))
    for label, arrays, temperatures in tests:
        for T in temperatures:
            for _ in range(8):
                x = rng.dirichlet(np.ones(n))
                gamma, he = fused_nrtl(x, T, *arrays)
                reference = original_gamma(x, T, *arrays)
                np.testing.assert_allclose(gamma, reference, rtol=3e-13, atol=3e-13)
                h = 0.01
                values = [np.dot(x, np.log(original_gamma(x, T+s*h, *arrays)))
                          for s in (-2, -1, 1, 2)]
                derivative = (values[0]-8*values[1]+8*values[2]-values[3])/(12*h)
                numeric_he = -R_J_MOL_K*T*T*derivative
                error = abs(he-numeric_he)/max(1.0, abs(he))
                if error > 2e-7:
                    raise AssertionError((label, T, he, numeric_he, error))
                records.append(error)
    # Nonpositive mixtures have the same all-ones gamma, zero HE convention.
    gamma, he = fused_nrtl(np.zeros(n), 350.0, *params)
    np.testing.assert_array_equal(gamma, np.ones(n))
    assert he == 0.0
    return {"checked_states": len(records), "max_scaled_he_error": max(records)}


def microbenchmark(params, rng, count):
    n = len(params[0])
    states = rng.dirichlet(np.ones(n), size=256)
    temperatures = rng.uniform(300.0, 420.0, 256)
    for variant in range(3):
        kernel_loop(states, temperatures, 1, variant, params)
    samples = {v: [] for v in VARIANTS}
    checksums = {}
    for repeat in range(5):
        for variant in np.roll(np.arange(3), repeat % 3):
            start = time.perf_counter()
            checksum = kernel_loop(states, temperatures, count, variant, params)
            samples[VARIANTS[variant]].append(time.perf_counter()-start)
            checksums[VARIANTS[variant]] = checksum
    return {"calls_per_sample": count, "seconds": {v: median(t) for v, t in samples.items()},
            "checksums": checksums}


def run_case(name, variant, estimate_missing=False):
    print(f"SETUP {name} {variant}", flush=True)
    unit, inlets = prepare_case(name, estimate_missing)
    restore = None
    if variant != "original":
        restore = install(unit.thermo, variant)
    # Do not charge a solve for collecting objects from an earlier variant.
    # Natural collection within this solve remains enabled for every variant.
    gc.collect()
    print(f"SOLVE {name} {variant}", flush=True)
    start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        result = unit.solve(inlets)
        cpu_elapsed = time.process_time()-cpu_start
        elapsed = time.perf_counter()-start
    finally:
        if restore is not None:
            restore()
    feed_flow = sum(s.F for s in inlets.values())
    outputs = result.outlet_streams
    balance = max(abs(sum(s.F*s.composition.get(c, 0) for s in inlets.values())
                      -sum(s.F*s.composition.get(c, 0) for s in outputs.values()))
                  / max(feed_flow, 1.0) for c in unit.thermo.components)
    if result.performance.get("jacobian_fallback"):
        raise AssertionError("Unexpected Jacobian fallback")
    return {"case": name, "variant": variant, "seconds": elapsed,
            "cpu_seconds": cpu_elapsed,
            "stages": unit.get_param("N_stages"), "components": unit.thermo.components,
            "residual": result.performance["mesh_residual"],
            "iterations": result.performance["solver_iterations"],
            "component_balance_error": balance, "feed_flow": feed_flow,
            "heat_duty": result.heat_duty, "warnings": result.warnings,
            "outputs": {p: {"F": s.F, "T": s.T, "x": s.composition}
                        for p, s in outputs.items()}}


def summarize(records, kernels):
    summary = {}
    variants = tuple(v for v in VARIANTS if any(r["variant"] == v for r in records))
    uncertainty_rng = np.random.default_rng(SEED)
    for case in dict.fromkeys(r["case"] for r in records):
        runs = [r for r in records if r["case"] == case and r["repeat"] > 0]
        timing = {v: median(r["seconds"] for r in runs if r["variant"] == v) for v in variants}
        deltas = {}
        for variant in variants[1:]:
            delta = {"composition": 0.0, "temperature_K": 0.0, "flow_over_feed": 0.0,
                     "heat_duty_relative": 0.0}
            for r in (r for r in runs if r["variant"] == variant):
                baseline = next(b for b in runs if b["variant"] == "original" and b["repeat"] == r["repeat"])
                for port, left in baseline["outputs"].items():
                    right = r["outputs"][port]
                    delta["temperature_K"] = max(delta["temperature_K"], abs(left["T"]-right["T"]))
                    delta["flow_over_feed"] = max(delta["flow_over_feed"], abs(left["F"]-right["F"])/baseline["feed_flow"])
                    delta["composition"] = max(delta["composition"], max(abs(v-right["x"][c]) for c, v in left["x"].items()))
                delta["heat_duty_relative"] = max(delta["heat_duty_relative"], abs(r["heat_duty"]-baseline["heat_duty"])/max(abs(baseline["heat_duty"]), 1.0))
            deltas[variant] = delta
        kernel = kernels[case]["seconds"]
        # Paired repetitions reduce drift; bootstrap makes small, noisy gains
        # visible as uncertainty rather than presenting every ratio as decisive.
        paired = {}
        for variant in variants[1:]:
            ratios = np.array([
                next(b["seconds"] for b in runs if b["variant"] == "original" and b["repeat"] == r["repeat"])/r["seconds"]
                for r in runs if r["variant"] == variant
            ])
            samples = ratios[uncertainty_rng.integers(0, len(ratios), size=(10000, len(ratios)))]
            interval = np.quantile(np.median(samples, axis=1), [0.025, 0.975])
            paired[variant] = {"median_speedup": float(np.median(ratios)),
                               "bootstrap_95_interval": interval.tolist()}
        summary[case] = {"solve_seconds": timing,
                         "solve_cpu_seconds": {v: median(r["cpu_seconds"] for r in runs if r["variant"] == v) for v in variants},
                         "stages": runs[0]["stages"], "components": runs[0]["components"],
                         "solve_speedups": {v: timing["original"]/timing[v] for v in variants[1:]},
                         "kernel_speedups": {v: kernel["original"]/kernel[v] for v in variants[1:]},
                         "paired_speedups": paired,
                         "max_residuals": {v: max(r["residual"] for r in runs if r["variant"] == v) for v in variants},
                         "iterations": {v: [r["iterations"] for r in runs if r["variant"] == v] for v in variants},
                         "solve_ranges": {v: [min(r["seconds"] for r in runs if r["variant"] == v), max(r["seconds"] for r in runs if r["variant"] == v)] for v in variants},
                         "max_component_balance_error": max(r["component_balance_error"] for r in runs),
                         "output_deltas": deltas}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--kernel-calls", type=int, default=100000)
    parser.add_argument("--cases", choices=CASES+LONG_CASES, nargs="+", default=list(CASES))
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--estimate-missing", action="store_true",
                        help="Fill missing synthetic-column NRTL pairs from UNIFAC at setup")
    args = parser.parse_args()
    if args.repeats < 1 or args.kernel_calls < 1:
        parser.error("Counts must be positive")
    if "original" not in args.variants or "fused" not in args.variants:
        parser.error("--variants must include original and fused")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.estimate_missing:
        from thermodynamics_models import interaction_estimation
        # Keep fitted data within this new artifact directory; don't mutate the
        # repository's existing runtime fit cache or its controlled JSONs.
        interaction_estimation._FIT_CACHE_PATH = args.output.resolve()/"estimated_interactions.sqlite"
    (args.output / "benchmark_script.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "manifest.json").write_text(json.dumps({
        "seed": SEED, "python": sys.version, "numpy": np.__version__, "threads": 1,
        "repeats": args.repeats, "cases": args.cases, "variants": args.variants,
        "estimate_missing": args.estimate_missing,
        "sources": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                    for name in ("compiled_activity.py", "thermodynamics_models/nrtl_uniquac.py",
                                 "equilibrium_stage_column.py", "unit_operations_distillation.py")},
    }, indent=2)+"\n")
    rng = np.random.default_rng(SEED)
    kernels, validations, records, compilation = {}, {}, [], []
    with (args.output / "results.jsonl").open("x") as handle:
        for case in args.cases:
            print(f"PREPARE {case}", flush=True)
            unit, _ = prepare_case(case, args.estimate_missing)
            thermo = unit.thermo
            missing = [pair for pair in combinations(thermo.components, 2)
                       if thermo._nrtl_interaction_for_components(*pair) is None]
            inventory = {"components": thermo.components,
                         "total_pairs": len(thermo.components)*(len(thermo.components)-1)//2,
                         "estimated_pairs": len(thermo.estimated_interaction_metadata),
                         "remaining_missing_pairs": missing,
                         "properties": {c: {"MW": p.MW, "CAS": p.CAS} for c,p in thermo.props.items()},
                         "estimation_quality": list(thermo.estimated_interaction_metadata.values())}
            (args.output/f"{case}.interactions.json").write_text(json.dumps(inventory, indent=2)+"\n")
            print(f"INTERACTIONS {case}: total={inventory['total_pairs']} estimated={inventory['estimated_pairs']} remaining_missing={len(missing)}", flush=True)
            params = unit.thermo._compiled_activity_backend().enthalpy_parameters()
            compile_start = time.perf_counter()
            fused_nrtl(np.full(len(params[0]), 1.0/len(params[0])), 350.0, *params)
            compilation.append({"case": case, "first_call_seconds": time.perf_counter()-compile_start})
            validations[case] = validate(params, rng)
            kernels[case] = microbenchmark(params, rng, args.kernel_calls)
            (args.output / f"{case}.kernel.json").write_text(json.dumps(kernels[case], indent=2)+"\n")
            print(f"VALIDATED {validations[case]}", flush=True)
            for repeat in range(args.repeats+1):
                offset = repeat % len(args.variants)
                order = args.variants[offset:]+args.variants[:offset]
                for variant in order:
                    r = run_case(case, variant, args.estimate_missing)
                    r["repeat"] = repeat
                    records.append(r)
                    handle.write(json.dumps(r)+"\n")
                    handle.flush()
                    print(f"DONE {case} {variant} repeat={repeat} {r['seconds']:.6f}s residual={r['residual']:.3e}", flush=True)
    summary = summarize(records, kernels)
    (args.output / "validation.json").write_text(json.dumps(validations, indent=2)+"\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output / "compilation.json").write_text(json.dumps(compilation, indent=2)+"\n")
    lines = ["NRTL analytic/fused HE experiment; single CPU thread.",
             "Times are medians; warm-ups excluded; setup/import/compilation excluded.", "",
             "| Case | Original ms | Analytic ms | Fused ms | Fused solve speedup | Fused kernel speedup |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, s in summary.items():
        t = s["solve_seconds"]
        analytic = f"{1000*t['analytic']:.3f}" if "analytic" in t else "—"
        lines.append(f"| {name} | {1000*t['original']:.3f} | {analytic} | {1000*t['fused']:.3f} | {s['solve_speedups']['fused']:.3f}x | {s['kernel_speedups']['fused']:.3f}x |")
    lines += ["", "Standalone analytic HE computes the fused state but discards gamma.",
              "Fused solves use a bounded shared gamma/HE cache and mirror the existing activity cache.",
              "The outer MESH Jacobian still uses its original finite differences."]
    (args.output / "summary.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
