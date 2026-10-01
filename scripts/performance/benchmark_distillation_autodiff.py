#!/usr/bin/env python3
"""Quick, script-only CPU experiment replacing MESH temperature/logit FD by AD.

Run from the repository root (the log persists both output streams):
    set -o pipefail
    python scripts/performance/benchmark_distillation_autodiff.py \
        --output /tmp/pfdsim-distillation-ad --repeats 3 \
        2>&1 | tee /tmp/pfdsim-distillation-ad.log

Requires the optional, already installed PyTorch; no production changes. Uses
float64 batched forward AD for NRTL/Raoult stage properties, plus exact custom
first derivatives of pure enthalpy integrals (Cp). The existing small internal
temperature difference defining NRTL excess enthalpy is preserved and itself
differentiated. This tests outer MESH derivatives, not an enthalpy model change.
The AD graph and backward passes are traced together with TorchScript by default;
use --backend eager to measure interpreter/dispatch overhead instead.
Use --backend inductor for CPU kernel fusion (longer compilation, cached in output).
Use --mode reverse to compare reverse mode (more outputs than inputs here).
The production residual, initializer, Newton solver and sparse assembly stay
authoritative; an AST patch replaces only the local finite-difference block in
memory. Unsupported thermodynamics and missing analytic Cp kernels fail loudly.

One warm-up solve per method/case is excluded. Measured order alternates, each
solve has fresh thermodynamic caches. Imports and initial backend compilation
are excluded. JSONL records flush after each solve; output directories must be
new. This is a small standalone experiment, not the repository performance suite.
"""

from __future__ import annotations

import argparse
import ast
from functools import wraps
import inspect
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys
import textwrap
import time

# Pin before importing numeric libraries; small stage blocks need one CPU thread.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from compiled_activity import _nrtl_activity_coefficients_numba
from equilibrium_stage_column import EquilibriumStageColumnMixin
from physical_constants import R_J_MOL_K
from simulator import Simulator
from thermodynamics import create_thermodynamics
from thermodynamics_models.nrtl_uniquac import NRTLThermodynamics
from unit_operations_distillation import RigorousDistillation

CASES = ("benzene_toluene_20", "partial_mass_12", "ternary_20")
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
np.random.seed(20260930)
torch.manual_seed(20260930)
GRAPH_CACHE = {}
COMPILATION_STATS = []


def prepare_case(name):
    if name == "benzene_toluene_20":
        sim = Simulator.from_file(
            ROOT / "examples/benzene_toluene_20_stage_distillation_nrtl.pfd"
        )
        sim.initialize()
        solver = sim.solver
        solver._initialize_streams()
        unit_id = "COL-1"
        inlets = {
            solver.stream_connections[sid][3]: solver._transition_stream_state(
                sid, solver.streams[sid], solver.unit_thermo_scopes[unit_id]
            ) for sid in solver.unit_inlets[unit_id]
        }
        return solver.units[unit_id], inlets
    partial = name == "partial_mass_12"
    comps = ["methanol", "water"] if partial else ["methanol", "ethanol", "water"]
    z = dict(zip(comps, [0.4, 0.6] if partial else [0.25, 0.25, 0.5]))
    thermo = create_thermodynamics(comps, "NRTL")
    feed = thermo.calculate_state(298.15, 1.0, 100.0, z, phase="liquid", flash=False)
    params = {
        "N_stages": 12 if partial else 20, "feed_stage": 7 if partial else 11,
        "reflux_ratio": 2.0, "P_condenser": 1.0, "P_drop_per_stage": 0.0,
        "condenser_type": "partial" if partial else "total",
        "mesh_tolerance": 1e-7, "max_iterations": 100,
        "max_jacobian_evaluations": 60, "initializer": "cheap_estimate",
    }
    params["D_mass_to_F_mass" if partial else "D_to_F"] = 0.35 if partial else 0.45
    return RigorousDistillation(name, thermo, params), {"feed": feed}


class LocalAD:
    """Independent stage graphs, batching derivative directions across stages."""

    def __init__(self, unit, arguments, backend_mode, mode):
        self.backend_mode = backend_mode
        self.mode = mode
        self.thermo = unit.thermo
        if type(self.thermo) is not NRTLThermodynamics:
            raise TypeError("This experiment supports plain NRTL/Raoult only")
        self.comps = arguments["comps"]
        if set(self.comps) != set(self.thermo.components):
            raise ValueError("Experiment requires all thermo components in the MESH model")
        self.N = arguments["N"]
        self.nc = len(self.comps)
        self.T_min = arguments["T_min"]
        self.span = arguments["T_max"] - self.T_min
        self.pressures = torch.tensor(arguments["pressures"])
        indices = [self.thermo.components.index(c) for c in self.comps]
        backend = self.thermo._compiled_activity_backend()
        self.params = {}
        for name in ("tau_mode", "tau_c", "tau_d", "tau_e", "tau_f", "tau_g",
                     "tau_tref", "tau_energy", "alpha", "interaction_tmin",
                     "interaction_tmax"):
            values = np.asarray(getattr(backend, name))[np.ix_(indices, indices)]
            self.params[name] = torch.tensor(values)
        # Diagonal reference temperatures can be zero; they are unused by NRTL.
        self.params["tau_tref"] = self.params["tau_tref"].clamp_min(1.0)
        self.psat = torch.tensor([self.thermo.get_Psat_coefficients(c) for c in self.comps])
        self.cpL = [self.thermo._liquid_cp_kernel(c) for c in self.comps]
        self.cpV = [self.thermo._ideal_gas_cp_kernel(c) for c in self.comps]
        if any(k is None for k in self.cpL + self.cpV):
            raise ValueError("Exact integral derivative requires analytic pure Cp kernels")
        self.seconds = 0.0
        self.calls = 0
        self.graph_key = (self.backend_mode, self.mode, self.N, self.nc, self.T_min, self.span,
                          tuple(arguments["pressures"]),
                          self.psat.numpy().tobytes(),
                          *(p.numpy().tobytes() for p in self.params.values()))

    def gamma(self, T, x):
        p = self.params
        mode = p["tau_mode"] % 10
        extrap = p["tau_mode"] // 10
        t = T[:, None, None]
        low, high = p["interaction_tmin"], p["interaction_tmax"]
        bound = torch.minimum(torch.maximum(t, low), high)
        b = torch.where(extrap > 0, bound, t)
        # Evaluate tangent continuations only at finite boundaries. For modes
        # with unrestricted limits, b=t and no continuation is selected.
        c, d, e, f, g, ref = (p["tau_" + n] for n in ("c", "d", "e", "f", "g", "tref"))
        v = c + d / b + e * ((ref - b) / b + torch.log(b / ref)) + f*b + g*b*b
        slope = -d/(b*b) + e*(b-ref)/(b*b) + f + 2*g*b
        cont2 = v + b*slope - b*b*slope/t
        cont3 = b*(2*v+b*slope)/t - b*b*(v+b*slope)/(t*t)
        cont4 = b*b*(3*v+b*slope)/(t*t) - b**3*(2*v+b*slope)/(t**3)
        v = torch.where((extrap == 2) & (b != t), cont2, v)
        v = torch.where((extrap == 3) & (b != t), cont3, v)
        v = torch.where((extrap == 4) & (b != t), cont4, v)
        # Energy/T uses the boundary only for the explicit clamp policy.
        # The tangent-continuation policies apply only to mode 1 in production.
        energy_temperature = torch.where(extrap == 1, bound, t)
        tau = torch.where(mode == 1, v, torch.where(
            mode == 2, p["tau_energy"]/energy_temperature, 0.0
        ))
        diagonal = torch.eye(self.nc, dtype=torch.bool)
        tau = torch.where(diagonal, 0.0, tau)
        G = torch.exp(-p["alpha"] * tau)
        denom = (x[:, :, None]*G).sum(dim=1).clamp_min(1e-30)
        weighted = (x[:, :, None]*tau*G).sum(dim=1)/denom
        ln = weighted + (x[:, None, :]*G/denom[:, None, :]
                         * (tau-weighted[:, None, :])).sum(dim=2)
        return torch.exp(ln.clamp(-50.0, 50.0)).clamp_min(1e-12)

    def vapor_pressure(self, T):
        A, B, C, D, E, F, G, H, Tc, power, slope, Tmin, lower_slope = (
            self.psat[:, i] for i in range(13)
        )
        t = T[:, None]
        b = torch.minimum(torch.maximum(t, Tmin), Tc)
        ln = A+B/b+C*torch.log(b)+D*b+E*b*b+F*b**5+G*b**3+H*((b/Tc)**power-1)
        ln = ln - torch.where(t < Tmin, Tmin*Tmin*lower_slope*(1/t-1/Tmin), 0.0)
        return torch.exp(ln) + torch.where(t > Tc, slope*(t-Tc), 0.0)

    def fields(self, u, pureL, pureV, cpL, cpV):
        N = self.N
        T = self.T_min + self.span*torch.sigmoid(u[:, 0].clamp(-60, 60))
        x = torch.softmax(torch.cat((u[:, 1:], torch.zeros((N, 1))), dim=1), dim=1)
        deltaT = (T-T.detach())[:, None]
        pureL = pureL + cpL*deltaT
        pureV = pureV + cpV*deltaT
        K = (self.gamma(T, x)*self.vapor_pressure(T)/self.pressures[:, None]).clamp(1e-6, 1e6)
        kx = K*x
        y = kx/kx.sum(dim=1, keepdim=True)
        dT = torch.maximum(torch.full_like(T, 1e-3), 1e-4*T)
        lo, hi = (T-dT).clamp_min(1.0), T+dT
        excess = -R_J_MOL_K*T*T*(x*(self.gamma(hi, x).log()-self.gamma(lo, x).log())
                                /(hi-lo)[:, None]).sum(dim=1)
        hL = (x*pureL).sum(dim=1)+excess
        hV = (y*pureV).sum(dim=1)
        bubble = kx.sum(dim=1)-1
        return torch.cat((x, y, hL[:, None], hV[:, None], bubble[:, None]), dim=1)

    def traceable_graph(self, u, pureL, pureV, cpL, cpV):
        if self.mode == "forward":
            tangents = []
            for j in range(self.nc):
                direction = torch.zeros_like(u)
                direction[:, j] = 1.0
                fields, tangent = torch.func.jvp(
                    lambda a: self.fields(a, pureL, pureV, cpL, cpV),
                    (u,), (direction,),
                )
                tangents.append(tangent)
            return fields, torch.stack(tangents, dim=2)
        fields = self.fields(u, pureL, pureV, cpL, cpV)
        # Trace the generated backward operations as well as the forward graph.
        derivatives = torch.stack([
            torch.autograd.grad(fields[:, i].sum(), u, retain_graph=True,
                                create_graph=False)[0]
            for i in range(2*self.nc+3)
        ], dim=1)
        return fields, derivatives

    def __call__(self, vector):
        start = time.perf_counter()
        self.calls += 1
        N, nc = self.N, self.nc
        local = np.column_stack((vector[:N], vector[N:N+N*(nc-1)].reshape(N, nc-1)))
        u = torch.tensor(local, requires_grad=True)
        temperatures = self.T_min + self.span/(1+np.exp(-np.clip(local[:, 0], -60, 60)))
        # Custom first-order primitive: preserve exact production integral values
        # and attach dH/dT=Cp, rather than porting every resolver integral to torch.
        pureL = torch.tensor([[self.thermo.enthalpy_liquid(c, float(t))*1000
                               for c in self.comps] for t in temperatures])
        pureV = torch.tensor([[self.thermo.enthalpy_ideal_gas(c, float(t))*1000
                               for c in self.comps] for t in temperatures])
        cpL = torch.tensor([[k.cp(float(t)) for k in self.cpL] for t in temperatures])
        cpV = torch.tensor([[k.cp(float(t)) for k in self.cpV] for t in temperatures])
        if self.backend_mode in ("script", "inductor"):
            if self.graph_key not in GRAPH_CACHE:
                compile_start = time.perf_counter()
                print(f"COMPILE START stages={N} components={nc}", flush=True)
                inputs = (u, pureL, pureV, cpL, cpV)
                # FX records saved tensors used by backward as graph values;
                # direct JIT tracing of autograd.grad treats some as constants.
                graph = make_fx(lambda a, b, c, d, e:
                                self.traceable_graph(a, b, c, d, e))(*inputs)
                graph.graph.eliminate_dead_code()
                graph.recompile()
                # The backward is already explicit in graph. Asking JIT to
                # differentiate that graph again adds unnecessary compiler work.
                with torch.no_grad(), torch.jit.optimized_execution(False):
                    detached = tuple(t.detach() for t in inputs)
                    if self.backend_mode == "script":
                        compiled = torch.jit.trace(graph, detached, check_trace=False)
                    else:
                        # The FX graph already exists; compile it directly rather
                        # than asking Dynamo to retrace forward-AD shape checks.
                        compiled = torch._inductor.compile(
                            graph, list(detached), options={"compile_threads": 2}
                        )
                        compiled(*detached)  # Include lazy compilation in cold time.
                    GRAPH_CACHE[self.graph_key] = compiled
                compilation_seconds = time.perf_counter()-compile_start
                COMPILATION_STATS.append({"stages": N, "components": nc,
                                          "seconds": compilation_seconds})
                print(f"COMPILE DONE {compilation_seconds:.3f}s", flush=True)
            with torch.no_grad(), torch.jit.optimized_execution(False):
                fields, derivatives = GRAPH_CACHE[self.graph_key](
                    u.detach(), pureL, pureV, cpL, cpV
                )
            return self.unpack(fields, derivatives, start)
        if self.mode == "forward":
            fields, derivatives = self.traceable_graph(u, pureL, pureV, cpL, cpV)
            return self.unpack(fields, derivatives, start)
        fields = self.fields(u, pureL, pureV, cpL, cpV)
        nout = fields.shape[1]
        # Summation over stages is safe because every graph is stage-local.
        derivatives = torch.autograd.grad(
            fields.sum(dim=0), u, torch.eye(nout), is_grads_batched=True
        )[0].permute(1, 0, 2)
        return self.unpack(fields, derivatives, start)

    def unpack(self, fields, derivatives, start):
        nc = self.nc
        derivatives = derivatives.detach().numpy()
        values = fields.detach().numpy()
        props = [{"y": dict(zip(self.comps, row[nc:2*nc])), "hL": row[2*nc],
                  "hV": row[2*nc+1], "bubble": row[2*nc+2]} for row in values]
        self.seconds += time.perf_counter()-start
        return props, derivatives


def validate_activity_kernel():
    """Check values and temperature derivatives for every NRTL policy."""
    rng = np.random.default_rng(20260930)
    ad = LocalAD.__new__(LocalAD)
    ad.nc = 3
    names = ("tau_mode", "tau_c", "tau_d", "tau_e", "tau_f", "tau_g",
             "tau_tref", "tau_energy", "alpha", "interaction_tmin", "interaction_tmax")
    checked = 0
    for mode in (0, 1, 2):
        for policy in range(5):
            shape = (ad.nc, ad.nc)
            arrays = (
                np.full(shape, mode + 10*policy, dtype=np.int64),
                rng.uniform(-0.5, 0.5, shape), rng.uniform(-150, 150, shape),
                rng.uniform(-0.5, 0.5, shape), rng.uniform(-0.002, 0.002, shape),
                rng.uniform(-1e-6, 1e-6, shape), np.full(shape, 298.15),
                rng.uniform(-150, 150, shape), np.full(shape, 0.3),
                np.full(shape, 300.0), np.full(shape, 400.0),
            )
            ad.params = {name: torch.tensor(a) for name, a in zip(names, arrays)}
            for temperature in (250.0, 350.0, 450.0):
                x = rng.dirichlet(np.ones(ad.nc))
                t = torch.tensor([temperature])
                values, derivative = torch.func.jvp(
                    lambda T: ad.gamma(T, torch.tensor(x[None, :])),
                    (t,), (torch.ones_like(t),),
                )
                reference = _nrtl_activity_coefficients_numba(x, temperature, *arrays)
                h = 0.01
                samples = [_nrtl_activity_coefficients_numba(x, temperature+s*h, *arrays)
                           for s in (-2, -1, 1, 2)]
                numeric = (samples[0]-8*samples[1]+8*samples[2]-samples[3])/(12*h)
                np.testing.assert_allclose(values.detach().numpy()[0], reference,
                                           rtol=3e-13, atol=3e-13)
                np.testing.assert_allclose(derivative.detach().numpy()[0], numeric,
                                           rtol=2e-7, atol=2e-10)
                checked += 1
    return {"checked_states": checked}


def patched_builder(original):
    """Reuse production sparse assembly verbatim, replacing its FD preamble."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    jac = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == "semi_analytic_flow_jacobian")
    assignments = [n for n in ast.walk(jac) if isinstance(n, ast.Assign)]
    props = next(n for n in assignments if ast.unparse(n.targets[0]) == "props")
    props.targets = ast.parse("props, local_derivatives = None").body[0].targets
    props.value = ast.parse("self._experiment_ad(vector)", mode="eval").body
    loop = next(n for n in ast.walk(jac) if isinstance(n, ast.For)
                and ast.unparse(n.target) == "col" and ast.unparse(n.iter) == "local_columns")
    # Keep every residual derivative contribution after the local property block.
    last = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "dbubble")
    preamble = ast.parse("""
local_index = 0 if col == stage else col - logits_start - stage*(nc-1) + 1
d = local_derivatives[stage, :, local_index]
dx = dict(zip(comps, d[:nc]))
dy = dict(zip(comps, d[nc:2*nc]))
dhL, dhV, dbubble = d[2*nc:]
""").body
    loop.body = preamble + loop.body[last+1:]
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, "<distillation-ad-experiment>", "exec"), namespace)
    return namespace[original.__name__]


def stream_record(s):
    return {"F": s.F, "T": s.T, "composition": dict(s.composition)}


def run_one(case, method, original, patched, backend, mode, validate=False):
    unit, inlets = prepare_case(case)
    stats = {"jacobian_seconds": 0.0, "jacobian_calls": 0,
             "jacobian_stage_property_seconds": 0.0, "jacobian_stage_property_calls": 0}
    checks = []
    signature = inspect.signature(original)
    inside_jac = False

    def build(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        model = original(*args, **kwargs)
        if not model["local_jacobian_available"]:
            raise ValueError("Case does not use local thermo Jacobian")
        if method == "ad" or validate:
            ad = LocalAD(unit, arguments, backend, mode)
            unit._experiment_ad = ad
            ad_model = patched(*args, **kwargs)
        # The closure cell is shared with residual: count only within Jacobians.
        jac = model["jacobian"]
        cell = dict(zip(jac.__code__.co_freevars, jac.__closure__))["stage_properties"]
        stage_original = cell.cell_contents

        @wraps(stage_original)
        def stage_timer(*a, **kw):
            start = time.perf_counter()
            try:
                return stage_original(*a, **kw)
            finally:
                if inside_jac:
                    stats["jacobian_stage_property_seconds"] += time.perf_counter()-start
                    stats["jacobian_stage_property_calls"] += 1
        cell.cell_contents = stage_timer
        selected = jac if method == "fd" else ad_model["jacobian"]

        def timed(vector, f0, step):
            nonlocal inside_jac
            # Correctness probe excluded from timings and performed on the
            # warm-up FD trajectory, including initial and late Newton states.
            if validate:
                reference = jac(vector, f0, step)[0].toarray()
                automatic = ad_model["jacobian"](vector, f0, step)[0].toarray()
                delta = automatic-reference
                checks.append({"scaled_max_entry_error": float(np.max(
                    np.abs(delta)/np.maximum(1.0, np.abs(automatic)))),
                    "relative_frobenius_error": float(np.linalg.norm(delta)
                        / max(np.linalg.norm(automatic), 1e-30))})
                if checks[-1]["scaled_max_entry_error"] > 2e-3:
                    raise AssertionError(f"AD/FD Jacobian disagrees: {checks[-1]}")
                # Also compare actual stage values to the production model.
                decoded = model["decode"](vector)
                values, _ = ad(vector)
                for s in range(arguments["N"]):
                    production = stage_original(s, decoded["T"][s], decoded["x"][s])
                    for field in ("hL", "hV", "bubble"):
                        np.testing.assert_allclose(values[s][field], production[field],
                                                   rtol=2e-10, atol=2e-7)
                    np.testing.assert_allclose(list(values[s]["y"].values()),
                                               [production["y"][c] for c in ad.comps],
                                               rtol=2e-10, atol=2e-12)
            start = time.perf_counter()
            inside_jac = True
            try:
                return selected(vector, f0, step)
            finally:
                inside_jac = False
                stats["jacobian_seconds"] += time.perf_counter()-start
                stats["jacobian_calls"] += 1
        model["jacobian"] = timed
        return model

    unit._build_mesh_model = lambda *a, **kw: build(unit, *a, **kw)
    start = time.perf_counter()
    result = unit.solve(inlets)
    elapsed = time.perf_counter()-start
    feed_flow = sum(s.F for s in inlets.values())
    outputs = result.outlet_streams
    balance = max(abs(sum(s.F*s.composition.get(c, 0) for s in inlets.values())
                      -sum(s.F*s.composition.get(c, 0) for s in outputs.values()))
                  / max(feed_flow, 1.0) for c in unit.thermo.components)
    if result.performance.get("jacobian_fallback"):
        raise AssertionError("Solver fell back to colored FD; comparison is invalid")
    return {"case": case, "method": method, "solve_seconds": elapsed, **stats,
            "ad_local_seconds": unit._experiment_ad.seconds if method == "ad" else None,
            "residual": result.performance["mesh_residual"],
            "iterations": result.performance["solver_iterations"],
            "component_balance_error": balance, "feed_flow": feed_flow,
            "outputs": {p: stream_record(s) for p, s in outputs.items()},
            "heat_duty": result.heat_duty, "checks": checks,
            "warnings": result.warnings}


def summarize(records, backend, mode):
    lines = ["Float64 CPU, one thread; medians of measured solves, warm-ups excluded.",
             f"AD = {backend} {mode}-mode PyTorch + exact pure-enthalpy Cp primitives.",
             "", "| Case | FD solve s | AD solve s | Solve speedup | FD Jacobian s | AD Jacobian s | Max residual (FD / AD) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    details = {}
    for case in dict.fromkeys(r["case"] for r in records):
        group = [r for r in records if r["case"] == case and r["repeat"] > 0]
        fd = [r for r in group if r["method"] == "fd"]
        ad = [r for r in group if r["method"] == "ad"]
        f, a = median(r["solve_seconds"] for r in fd), median(r["solve_seconds"] for r in ad)
        fj, aj = median(r["jacobian_seconds"] for r in fd), median(r["jacobian_seconds"] for r in ad)
        lines.append(f"| {case} | {f:.6f} | {a:.6f} | {f/a:.3f}x | {fj:.6f} | {aj:.6f} | "
                     f"{max(r['residual'] for r in fd):.3e} / {max(r['residual'] for r in ad):.3e} |")
        property_time = median(r["jacobian_stage_property_seconds"] for r in fd)
        deltas = {"composition": 0.0, "temperature_K": 0.0, "flow_over_feed": 0.0}
        for left, right in zip(fd, ad):
            for port, s in left["outputs"].items():
                t = right["outputs"][port]
                deltas["temperature_K"] = max(deltas["temperature_K"], abs(s["T"]-t["T"]))
                deltas["flow_over_feed"] = max(deltas["flow_over_feed"], abs(s["F"]-t["F"])/left["feed_flow"])
                for comp in s["composition"]:
                    deltas["composition"] = max(deltas["composition"], abs(s["composition"][comp]-t["composition"][comp]))
        details[case] = {
            "fd_solve_seconds": f, "ad_solve_seconds": a, "solve_speedup": f/a,
            "fd_jacobian_seconds": fj, "ad_jacobian_seconds": aj, "jacobian_speedup": fj/aj,
            "ad_local_seconds": median(r["ad_local_seconds"] for r in ad),
            "fd_stage_property_fraction": property_time/f,
            "zero_cost_properties_only_speedup": f/(f-property_time),
            "zero_cost_entire_jacobian_speedup_ceiling": f/(f-fj),
            "max_component_balance_error": max(r["component_balance_error"] for r in group),
            "output_deltas": deltas,
            "fd_iterations": [r["iterations"] for r in fd],
            "ad_iterations": [r["iterations"] for r in ad],
            "solve_ranges": {m: [min(r["solve_seconds"] for r in group if r["method"] == m),
                                  max(r["solve_seconds"] for r in group if r["method"] == m)]
                             for m in ("fd", "ad")},
        }
    lines += ["", json.dumps(details, indent=2), "",
              "Property-only estimate removes all Jacobian property evaluation costs;",
              "whole-Jacobian ceiling removes assembly and perturbation costs too.",
              "Both assume unchanged iterations and all other work; neither is measured AD.",
              "Initializer and final diagnostics are included in solve timings.",
              "Plain NRTL cases only: no claim about PR, UNIFAC, VLLE or CMO.",
              "NRTL's inner excess-enthalpy temperature difference remains unchanged."]
    return "\n".join(lines)+"\n", details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--backend", choices=("script", "eager", "inductor"), default="script")
    parser.add_argument("--mode", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    validation = validate_activity_kernel()
    (args.output / "activity_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    if args.backend == "inductor":
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.output.resolve() / "inductor-cache")
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "2"
    original = EquilibriumStageColumnMixin._build_mesh_model
    patched = patched_builder(original)
    records = []
    # New output directory: preserve the harness without overwriting artifacts.
    (args.output / "benchmark_script.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "manifest.json").write_text(json.dumps({
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "seed": 20260930, "threads": 1, "cases": args.cases,
        "repeats": args.repeats, "backend": args.backend, "mode": args.mode,
        "script": str(Path(__file__).relative_to(ROOT)),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("equilibrium_stage_column.py",
                                       "unit_operations_distillation.py",
                                       "compiled_activity.py",
                                       "thermodynamics_models/base.py",
                                       "thermodynamics_models/nrtl_uniquac.py")},
    }, indent=2)+"\n")
    with (args.output / "results.jsonl").open("x") as handle:
        for case in args.cases:
            for repeat in range(args.repeats+1):
                order = ("fd", "ad") if repeat % 2 == 0 else ("ad", "fd")
                for method in order:
                    print(f"START {case} {method} repeat={repeat}", flush=True)
                    record = run_one(case, method, original, patched, args.backend, args.mode,
                                     validate=repeat == 0 and method == "fd")
                    record["repeat"] = repeat
                    records.append(record)
                    handle.write(json.dumps(record)+"\n")
                    handle.flush()
                    print(f"DONE {record['solve_seconds']:.4f}s residual={record['residual']:.3e}", flush=True)
    report, details = summarize(records, args.backend, args.mode)
    (args.output / "summary.txt").write_text(report)
    (args.output / "summary.json").write_text(json.dumps(details, indent=2)+"\n")
    (args.output / "compilation.json").write_text(json.dumps(COMPILATION_STATS, indent=2)+"\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
