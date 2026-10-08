#!/usr/bin/env python3
"""Inspect binary overdraw failures and solve captured native MESH problems.

Example (new output directory, single-threaded BLAS):
  OPENBLAS_NUM_THREADS=1 .venv/bin/python \
      scripts/performance/diagnose_binary_azeotropic_overdraw.py \
      --output /tmp/binary-overdraw-diagnostic

No production edits. Reuses the overdraw fixtures, column model and Jacobians.
Captures failed iterates, residual labels, singular values and derivative checks.
Compares native line search with exact dense TRF on these small systems, using
both native and logarithmically interpolated stage-profile seeds. Every solve
is logged and persisted before continuing. Dense TRF is a diagnostic for these
122-variable columns, not a proposed replacement for the general sparse solver.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_azeotropic_overdistillation import candidate_metadata, json_default, prepare_case
from benchmark_column_trust_region import RawDone, newton_direction


def capture_problem(unit, inlets):
    from equilibrium_stage_column import EquilibriumStageColumnMixin
    from unit_operations_distillation import RigorousDistillation

    holder = {}
    build = EquilibriumStageColumnMixin._build_mesh_model
    initial = RigorousDistillation._initial_guess
    native = EquilibriumStageColumnMixin._sparse_newton_solve

    def model(self, *args, **kwargs):
        if self is not unit:
            return build(self,*args,**kwargs)
        holder["model_args"] = dict(inspect.signature(build).bind(self, *args, **kwargs).arguments)
        holder["model"] = build(self, *args, **kwargs)
        return holder["model"]

    def seed(self, *args, **kwargs):
        if self is not unit:
            return initial(self,*args,**kwargs)
        holder["initial_args"] = dict(inspect.signature(initial).bind(self, *args, **kwargs).arguments)
        holder["initial"] = initial(self, *args, **kwargs)
        return holder["initial"]

    def stop(self, residual, sparsity, x0, options, jacobian=None, step_event=None):
        if self is not unit:
            return native(self,residual,sparsity,x0,options,jacobian=jacobian,step_event=step_event)
        holder.update(x0=x0.copy(), options=dict(options), residual=residual, sparsity=sparsity,
                      jacobian=jacobian, step_event=step_event)
        raise RawDone

    with patch.object(EquilibriumStageColumnMixin, "_build_mesh_model", model), \
            patch.object(RigorousDistillation, "_initial_guess", seed), \
            patch.object(EquilibriumStageColumnMixin, "_sparse_newton_solve", stop):
        try:
            unit.solve(inlets)
        except RawDone:
            pass
    return holder


def diagnose(unit, problem, vector):
    import numpy as np
    residual = problem["residual"]
    f = residual(vector)
    J_sparse = problem["jacobian"](vector, f, 1e-6)[0]
    J = J_sparse.toarray()
    U, singular, Vt = np.linalg.svd(J, full_matrices=False)
    d = problem["model"]["decode"](vector)
    args = problem["model_args"]
    N = args["N"]
    alcohol = next(c for c in args["comps"] if c != "water")
    order = np.argsort(np.abs(f))[::-1][:12]
    gradient = J.T @ f
    # Seeded directional central differences avoid building an expensive full
    # central-difference Jacobian merely for a consistency check.
    checks = []
    rng = np.random.default_rng(20261007)
    for _ in range(3):
        direction = rng.normal(size=len(vector))
        direction /= np.linalg.norm(direction)
        h = 1e-5
        difference = (residual(vector+h*direction)-residual(vector-h*direction))/(2*h)
        predicted = J @ direction
        checks.append(float(np.linalg.norm(difference-predicted)/max(np.linalg.norm(difference), 1e-15)))
    direction = newton_direction(J_sparse,f)
    cap_diagnostics = None
    if direction is not None:
        largest = float(np.max(np.abs(direction)))
        cap_scale = min(1.,8./largest) if largest else 1.
        capped = direction*cap_scale
        trial = residual(vector+capped)
        cap_diagnostics = {"largest_newton_coordinate_step":largest,"cap_scale":cap_scale,
                           "largest_step_coordinate_index":int(np.argmax(np.abs(direction))),
                           "linear_solve_residual_inf":float(np.linalg.norm(J @ direction+f,ord=np.inf)),
                           "capped_trial_merit":.5*float(trial@trial),
                           "current_merit":.5*float(f@f),
                           "largest_capped_temperature_step":float(np.max(np.abs(capped[:N]))),
                           "largest_capped_flow_log_step":float(np.max(np.abs(capped[2*N:4*N])))}
    return {"residual": float(np.linalg.norm(f, ord=np.inf)), "merit": .5*float(f@f),
            "gradient_inf": float(np.linalg.norm(gradient, ord=np.inf)),
            "singular_values": singular.tolist(),
            "condition_number": float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
            "jacobian_direction_relative_errors": checks,
            "newton_step_cap": cap_diagnostics,
            "dominant_residuals": [{"label": problem["model"]["residual_labels"][int(i)],
                                    "value": float(f[i])} for i in order],
            "temperature_K": list(d["T"]), "alcohol_x": [x[alcohol] for x in d["x"]],
            "liquid_flows": list(d["L"]), "vapor_flows": list(d["V"]),
            "Q_cond": d["Q_cond"], "Q_reb": d["Q_reb"],
            "temperature_coordinates": vector[:N].tolist(),
            "composition_coordinates": vector[N:2*N].tolist(),
            "smallest_right_singular_vector": Vt[-1].tolist(),
            "native_vector": vector.tolist()}


def profile_seed(unit, problem, anchored):
    """Keep native endpoint/flow/duty estimates; interpolate compositions in log space."""
    import numpy as np
    initial = problem["initial"]
    args = problem["model_args"]
    N, comps = args["N"], args["comps"]
    feed_index = args["feed_index"]
    top, bottom = initial["x"][0], initial["x"][-1]
    xs, Ts = [], []
    for j in range(N):
        if anchored and j <= feed_index:
            left, right, weight = top, args["feed_z"], j/feed_index
        elif anchored:
            left, right = args["feed_z"], bottom
            weight = (j-feed_index)/(N-1-feed_index)
        else:
            left, right, weight = top, bottom, j/(N-1)
        logits = {c: (1-weight)*math.log(max(left[c], 1e-14))+weight*math.log(max(right[c], 1e-14)) for c in comps}
        max_logit = max(logits.values())
        weights = {c: math.exp(value-max_logit) for c, value in logits.items()}
        x = {c: value/sum(weights.values()) for c, value in weights.items()}
        xs.append(x)
        Ts.append(unit._bubble_temperature_from_equation(x,args["pressures"][j],args["T_min"],args["T_max"]))
    return np.asarray(unit._pack_variables(Ts,xs,initial["L"],initial["V"],initial["Q_cond"],initial["Q_reb"],
                                         comps,args["T_min"],args["T_max"],args["energy_scale"]))


def dense_solve(problem, seed, bounded, max_jacobians=200, physical=False, numerical=False):
    import numpy as np
    from scipy.optimize import least_squares

    count, functions = 0, 0
    N = problem["model_args"]["N"]
    native_seed = seed.copy()
    if physical:
        seed = seed.copy()
        seed[N:2*N] = 1/(1+np.exp(-seed[N:2*N]))
    current = [native_seed, problem["residual"](native_seed)]
    cache = [None, None]

    def to_native(vector):
        if not physical:
            return vector
        native = vector.copy()
        fractions = np.clip(vector[N:2*N],1e-30,1-1e-15)
        native[N:2*N] = np.log(fractions)-np.log1p(-fractions)
        return native

    def fun(vector):
        nonlocal functions
        functions += 1
        f = problem["residual"](to_native(vector))
        cache[:] = [vector.copy(), f]
        return f

    def jac(vector):
        nonlocal count, functions
        f = cache[1] if cache[0] is not None and np.array_equal(vector, cache[0]) else fun(vector)
        native = to_native(vector)
        current[:] = [native.copy(), f.copy()]
        if np.linalg.norm(f, ord=np.inf)<1e-8 or count >= max_jacobians:
            raise RawDone
        count += 1
        J, evaluations, *_ = problem["jacobian"](native,f,1e-6)
        functions += evaluations
        J = J.toarray()
        if physical:
            fractions = np.clip(vector[N:2*N],1e-30,1-1e-15)
            J[:,N:2*N] /= fractions*(1-fractions)
        return J

    def callback(intermediate_result):
        nonlocal count
        if numerical:
            count += 1
        current[:] = [to_native(intermediate_result.x), intermediate_result.fun.copy()]
        if np.linalg.norm(intermediate_result.fun,ord=np.inf)<1e-8 or count >= max_jacobians:
            raise StopIteration

    if bounded:
        args = problem["model_args"]
        N = args["N"]
        low, high = np.full(len(seed),-np.inf), np.full(len(seed),np.inf)
        for T, target in ((300., low), (420., high)):
            reduced = (T-args["T_min"])/(args["T_max"]-args["T_min"])
            target[:N] = math.log(reduced/(1-reduced))
        low[N:2*N], high[N:2*N] = (0.,1.) if physical else (-32.,32.)
        low[2*N:4*N], high[2*N:4*N] = math.log(1e-8), math.log(1e5)
        seed = np.maximum(np.minimum(seed,np.nextafter(high,low)),np.nextafter(low,high))
        bounds = (low,high)
    else:
        bounds = (-np.inf,np.inf)
    try:
        result = least_squares(fun,seed,jac="2-point" if numerical else jac,method="trf",tr_solver="exact",x_scale="jac",
                               bounds=bounds,gtol=1e-13,xtol=1e-13,ftol=None,max_nfev=max_jacobians*16,
                               callback=callback)
        vector, f, message = to_native(result.x), result.fun, str(result.message)
        if numerical:
            count = result.njev
    except RawDone:
        vector, f = current
        message = "residual target or Jacobian budget reached"
    return {"x": vector, "residual_norm": float(np.linalg.norm(f,ord=np.inf)),
            "jacobian_evaluations": count,"function_evaluations": functions,"message": message}


def replay_solution(unit, inlets, result):
    """Pass an independently obtained native vector through normal unit checks."""
    import numpy as np
    from equilibrium_stage_column import EquilibriumStageColumnMixin

    def solved(self,residual,sparsity,x0,options,jacobian=None,step_event=None):
        norm = float(np.linalg.norm(residual(result["x"]),ord=np.inf))
        return dict(result,success=norm<options["mesh_tolerance"],residual_norm=norm,
                    iterations=result["jacobian_evaluations"],jacobian_method="experimental_replay")
    with patch.object(EquilibriumStageColumnMixin,"_sparse_newton_solve",solved):
        checked = unit.solve(inlets)
    feed = inlets["feed"]
    error = max(abs(feed.F*feed.composition[c]-sum(s.F*s.composition.get(c,0.) for s in checked.outlet_streams.values()))/feed.F
                for c in feed.composition)
    stability = [unit.thermo.liquid_spinodal_stability(T+273.15,x) for T,x in zip(
        checked.performance["stage_temperatures_C"],checked.performance["stage_liquid_compositions"])]
    N = len(stability)
    phase_audit = []
    for index in sorted({0,int(unit.get_param("feed_stage"))-1,N-1}):
        x = checked.performance["stage_liquid_compositions"][index]
        T = checked.performance["stage_temperatures_C"][index]+273.15
        split,*_ = unit.thermo.liquid_liquid_equilibrium(x,T)
        phase_audit.append({"stage":index+1,"has_lle":bool(split)})
    return {"residual":checked.performance["mesh_residual"],"component_balance_error":error,
            "outputs":{p:{"F":s.F,"T":s.T,"composition":s.composition} for p,s in checked.outlet_streams.items()},
            "all_locally_stable":all(s["locally_stable"] for s in stability),"phase_audit":phase_audit,
            "performance":checked.performance}


def energy_precondition(unit, problem):
    """Invertible row transform: remove constant component enthalpy references."""
    import numpy as np
    from scipy.sparse import eye
    args = problem["model_args"]
    nc = len(args["comps"])
    P = eye(len(problem["x0"]),format="lil")
    multiplier = args["energy_scale"]/(args["flow_scale"]*50000.)
    for j in range(args["N"]):
        energy_row = j*(nc+2)+nc
        P[energy_row,energy_row] = multiplier
        for ci,c in enumerate(args["comps"]):
            reference = unit.thermo.mixture_enthalpy({comp:float(comp==c) for comp in args["comps"]},298.15,0.,P=args["pressures"][j])
            P[energy_row,j*(nc+2)+ci] = -reference*args["component_scales"][c]/args["energy_scale"]*multiplier
    P = P.tocsr()
    original_residual, original_jacobian = problem["residual"], problem["jacobian"]

    def residual(vector):
        return np.asarray(P @ original_residual(vector))

    def jacobian(vector,value,step):
        result = original_jacobian(vector,original_residual(vector),step)
        J,evaluations,*label = result
        return (P @ J,evaluations+1,*label)

    return dict(problem,residual=residual,jacobian=jacobian)


def reflux_ramp(base,inlets,metadata,args,handle):
    import numpy as np
    from unit_operations_distillation import RigorousDistillation
    for factor in args.factors:
        last = None
        for reflux in args.reflux_ramp:
            unit = RigorousDistillation(base.unit_id,base.thermo,dict(base.params,
                D_to_F=factor*metadata["first_azeotrope"]["cut_capacity"],reflux_ratio=reflux,initializer="cheap_estimate",
                mesh_tolerance=1e-8,acceptable_mesh_residual=1e-8,finite_difference_rel_step=1e-6,
                colored_jacobian_fallback=False,max_iterations=args.budget,max_jacobian_evaluations=args.budget))
            problem = capture_problem(unit,inlets)
            solve_problem = energy_precondition(unit,problem) if args.energy_precondition else problem
            seed = problem["x0"] if last is None else last
            record = {"case":args.current_case,"factor":factor,"reflux_ratio":reflux,
                      "seed":"reflux_ramp","method":"physical"}
            print("START ramp",args.current_case,factor,reflux,flush=True)
            begin = time.perf_counter()
            quality = getattr(unit.thermo,"quality_context",None)
            with quality(phase="solver_iteration",affects_result=False) if quality else nullcontext():
                result = dense_solve(solve_problem,seed,True,args.budget,physical=True)
                if args.energy_precondition:
                    result["preconditioned_residual_norm"] = result["residual_norm"]
                    result["residual_norm"] = float(np.linalg.norm(problem["residual"](result["x"]),ord=np.inf))
                record.update({k:v for k,v in result.items() if k!="x"})
                record["solve_seconds"] = time.perf_counter()-begin
                record["diagnostic"] = diagnose(unit,problem,result["x"])
            if result["residual_norm"]<1e-8:
                last = np.asarray(result["x"]).copy()
                record["validated_unit"] = replay_solution(unit,inlets,result)
            handle.write(json.dumps(record,default=json_default)+"\n")
            handle.flush()
            print("DONE ramp",args.current_case,factor,reflux,result["residual_norm"],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cases",nargs="+",choices=("ethanol_water","ipa_water"),default=["ethanol_water","ipa_water"])
    parser.add_argument("--factors",nargs="+",type=float,default=[1.005,1.02,1.05])
    parser.add_argument("--seeds",nargs="+",choices=("native","log","log_anchor","saved"),default=["native","log","log_anchor"])
    parser.add_argument("--methods",nargs="+",choices=("line","dense","dense_bounded","physical","physical_fd"),default=["line","dense","dense_bounded"])
    parser.add_argument("--restart-data",type=Path,help="JSONL diagnostic records from a completed run")
    parser.add_argument("--reflux-ramp",type=float,nargs="+",help="Continue validated physical-coordinate roots in reflux ratio")
    parser.add_argument("--energy-precondition",action="store_true",help="Remove reference enthalpy contributions using component balance rows")
    parser.add_argument("--seed-composition-floor",type=float,default=0.,help="Apply the same physical composition floor to every method's seed")
    parser.add_argument("--initializer",choices=("azeotropic","cheap_estimate","coarse_rigorous"),default="azeotropic")
    parser.add_argument("--audit-data",type=Path,help="Reevaluate saved vectors and unit checks without new nonlinear solves")
    parser.add_argument("--budget",type=int,default=200)
    args = parser.parse_args()
    if not 0 <= args.seed_composition_floor < .5:
        parser.error("seed composition floor must be in [0, .5)")
    if "saved" in args.seeds and args.restart_data is None:
        parser.error("saved seed requires --restart-data")
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"diagnostic_script.py").write_bytes(Path(__file__).read_bytes())
    random.seed(20261007)
    import numpy as np
    import scipy
    from equilibrium_stage_column import EquilibriumStageColumnMixin
    from unit_operations_distillation import RigorousDistillation
    np.random.seed(20261007)
    saved = [json.loads(line) for line in args.restart_data.read_text().splitlines()] if args.restart_data else []
    audited = [json.loads(line) for line in args.audit_data.read_text().splitlines()] if args.audit_data else []
    (args.output/"manifest.json").write_text(json.dumps({"args":vars(args),"python":sys.version,
        "numpy":np.__version__,"scipy":scipy.__version__,"hash_seed":os.environ.get("PYTHONHASHSEED"),
        "blas_threads":os.environ.get("OPENBLAS_NUM_THREADS"),
        "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            "equilibrium_stage_column.py","unit_operations_distillation.py","thermodynamics_models/activity.py",
            "scripts/performance/benchmark_column_trust_region.py","scripts/performance/probe_azeotropic_overdistillation.py")}},indent=2,default=str)+"\n")
    with (args.output/"results.jsonl").open("x") as handle:
        for name in args.cases:
            base,inlets = prepare_case(name)
            metadata = candidate_metadata(base,inlets["feed"])
            (args.output/(name+"-metadata.json")).write_text(json.dumps(metadata,indent=2,default=json_default)+"\n")
            if args.reflux_ramp:
                args.current_case = name
                reflux_ramp(base,inlets,metadata,args,handle)
                continue
            for factor in args.factors:
                unit = RigorousDistillation(base.unit_id,base.thermo,dict(base.params,
                    D_to_F=factor*metadata["first_azeotrope"]["cut_capacity"],initializer=args.initializer,
                    mesh_tolerance=1e-8,acceptable_mesh_residual=1e-8,finite_difference_rel_step=1e-6,
                    colored_jacobian_fallback=False,max_iterations=args.budget,max_jacobian_evaluations=args.budget))
                problem = capture_problem(unit,inlets)
                solve_problem = energy_precondition(unit,problem) if args.energy_precondition else problem
                if args.audit_data:
                    for row in audited:
                        if row["case"]!=name or row["factor"]!=factor:
                            continue
                        vector = np.array(row["diagnostic"]["native_vector"])
                        diagnostic = diagnose(unit,problem,vector)
                        record = {"case":name,"factor":factor,"seed":row["seed"],"method":row["method"],
                                  "diagnostic":diagnostic,"source_residual":row["residual_norm"]}
                        if diagnostic["residual"]<1e-8:
                            result = dict(x=vector,residual_norm=diagnostic["residual"],
                                          jacobian_evaluations=row.get("jacobian_evaluations",0),
                                          function_evaluations=row.get("function_evaluations",0),message="replay audit")
                            record["validated_unit"] = replay_solution(unit,inlets,result)
                        handle.write(json.dumps(record,default=json_default)+"\n")
                        handle.flush()
                        print("AUDIT",name,row["seed"],row["method"],diagnostic["residual"],flush=True)
                    continue
                for seed_name in args.seeds:
                    if seed_name=="saved":
                        candidates = [r for r in saved if r["case"]==name and r["factor"]==factor]
                        vector = np.array(min(candidates,key=lambda r:r["residual_norm"])["diagnostic"]["native_vector"])
                    else:
                        vector = problem["x0"] if seed_name=="native" else profile_seed(unit,problem,seed_name=="log_anchor")
                    if args.seed_composition_floor:
                        N = problem["model_args"]["N"]
                        vector = vector.copy()
                        fractions = np.clip(1/(1+np.exp(-vector[N:2*N])),args.seed_composition_floor,1-args.seed_composition_floor)
                        vector[N:2*N] = np.log(fractions)-np.log1p(-fractions)
                    for method in args.methods:
                        record = {"case":name,"factor":factor,"seed":seed_name,"method":method,
                                  "reflux_ratio":base.params["reflux_ratio"]}
                        print("START",name,factor,seed_name,method,flush=True)
                        begin = time.perf_counter()
                        quality = getattr(unit.thermo,"quality_context",None)
                        unit._quality_solver_aux_context_active = True
                        with quality(phase="solver_iteration",affects_result=False) if quality else nullcontext():
                            if method=="line":
                                result = EquilibriumStageColumnMixin._sparse_newton_solve(unit,solve_problem["residual"],problem["sparsity"],vector,problem["options"],jacobian=solve_problem["jacobian"])
                            else:
                                result = dense_solve(solve_problem,vector,method in ("dense_bounded","physical","physical_fd"),args.budget,
                                                     physical=method.startswith("physical"),numerical=method=="physical_fd")
                            if args.energy_precondition:
                                result["preconditioned_residual_norm"] = result["residual_norm"]
                                result["residual_norm"] = float(np.linalg.norm(problem["residual"](result["x"]),ord=np.inf))
                            record.update({k:v for k,v in result.items() if k!="x"})
                            record["solve_seconds"] = time.perf_counter()-begin
                            record["diagnostic"] = diagnose(unit,problem,result["x"])
                        if result["residual_norm"]<1e-8:
                            record["validated_unit"] = replay_solution(unit,inlets,result)
                        handle.write(json.dumps(record,default=json_default)+"\n")
                        handle.flush()
                        print("DONE",name,factor,seed_name,method,result["residual_norm"],record["solve_seconds"],flush=True)


if __name__=="__main__":
    main()
