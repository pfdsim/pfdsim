"""Compare current pretrained HANNA with five UNIFAC variants.

Reuse saved raw-binary and closely-agreeing NRTL/UNIQUAC states/predictions;
add measured ternary VLE/VLLE. Equal weight per binary pair or ternary system.
No fitting or nested pressure/temperature solves. Experimental and fitted
references remain separate. Cache official molecular embeddings once per SMILES.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.activity_fitting.benchmark_unifac_runtime_fits import METHOD_FILES, write_csv
from scripts.activity_fitting.benchmark_unifac_raw_vle import (
    PropertyDatabase, SourceCurves, is_acid, SOURCE,
)
from thermodynamics_models.activity import ActivityCoefficientThermodynamics
from thermodynamics_models.gamma_phi import GammaPhiVaporBackendMixin
from thermodynamics_models.unifac_models import UNIFACThermodynamics
from unifac import UNIFACModel, get_unifac_groups

METHODS = (*METHOD_FILES, "HANNA")
SEED = 20261013


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


class Hanna:
    """Batched use of the official ensemble, scalers and embedding routine."""

    def __init__(self, checkout):
        self.checkout = checkout
        sys.path.insert(0, str(checkout))
        from utils.HANNA_predictor import HANNA_Predictor
        self.predictor = HANNA_Predictor()
        self.predictor.ChemBERTA.eval()
        self.embeddings = {}

    def predict(self, smiles, temperatures, compositions, batch_size=128):
        for s in smiles:
            if s not in self.embeddings:
                self.embeddings[s] = self.predictor._get_scaled_embeddings_from_smiles([s])[0]
        temp = self.predictor.t_scaler.transform(np.asarray(temperatures).reshape(-1, 1))
        x = np.ascontiguousarray(compositions)
        output, energies = [], []
        for start in range(0, len(x), batch_size):
            end = min(len(x), start+batch_size)
            embeddings = torch.stack([self.embeddings[s] for s in smiles]).unsqueeze(0).repeat(end-start, 1, 1)
            lng, ge = self.predictor.model(torch.tensor(temp[start:end], dtype=torch.float32),
                                           torch.tensor(x[start:end, :-1], dtype=torch.float32), embeddings)
            output.extend(lng.detach().numpy().tolist())
            energies.extend(ge.detach().numpy().tolist())
        result, energies = np.array(output), np.array(energies)
        if not np.all(np.isfinite(result)):
            raise ValueError("nonfinite HANNA prediction")
        np.testing.assert_allclose(np.sum(x*result, axis=1), energies, atol=3e-6, rtol=2e-5)
        return result

    def checks(self):
        results = {}
        for smiles, x in ((["CCO", "O"], [[.2, .8], [.5, .5], [.8, .2]]),
                          (["CCO", "O", "CO"], [[.2, .3, .5], [.4, .3, .3]])):
            x = np.array(x)
            t = np.repeat(323.15, len(x))
            actual = self.predict(smiles, t, x)
            official, _ = self.predictor.predict(smiles, x, 323.15)
            np.testing.assert_allclose(actual, official, rtol=1e-5, atol=2e-6)
            permuted = self.predict(smiles[::-1], t, x[:, ::-1])[:, ::-1]
            np.testing.assert_allclose(actual, permuted, atol=4e-6, rtol=2e-5)
            xi = torch.tensor(x[:, :-1], dtype=torch.float32, requires_grad=True)
            embeddings = torch.stack([self.embeddings[s] for s in smiles]).unsqueeze(0).repeat(len(x), 1, 1)
            temperatures = torch.tensor(self.predictor.t_scaler.transform(t.reshape(-1, 1)), dtype=torch.float32)
            lng, ge = self.predictor.model(temperatures, xi, embeddings)
            derivatives = [torch.autograd.grad(lng[:, i].sum(), xi, retain_graph=True)[0] for i in range(len(smiles))]
            residual = sum(torch.tensor(x[:, i:i+1], dtype=torch.float32)*derivatives[i] for i in range(len(smiles)))
            maximum = float(residual.detach().abs().max())
            if maximum > 3e-5:
                raise ValueError(f"Gibbs-Duhem residual {maximum:g}")
            results[f"{len(smiles)}_components_Gibbs_Duhem_max"] = maximum
        pure = self.predict(["CCO", "O"], [323.15, 323.15], [[1, 0], [0, 1]])
        np.testing.assert_allclose([pure[0, 0], pure[1, 1]], 0, atol=1e-7)
        identical = self.predict(["CCO", "CCO"], [323.15], [[.2, .8]])
        np.testing.assert_allclose(identical, 0, atol=1e-7)
        results["official_interface_permutation_pure_identical_checks"] = "PASS"
        return results


class SharedVapor(GammaPhiVaporBackendMixin, ActivityCoefficientThermodynamics):
    """Existing vapor and pure-reference paths without a UNIFAC liquid model."""
    vapor_backend_model = "BV"

    def __init__(self, components, db, acid):
        super().__init__(components, db)
        self._initialize_gamma_phi_backend(components, db, second_virial_correlation="HOC" if acid else "TSONOPOULOS")


def equilibrium_residual(model, curves, measurement, lng):
    components = model.components
    t, p, x = measurement["T_K"], measurement["P_Pa"]/1e5, np.array(measurement["x"])
    doi = measurement["doi"]
    # Pure standard-state factors are independent of the liquid model and
    # composition. Compute them once per measured source/T/P, rather than
    # repeatedly clearing and rebuilding them for all six methods/phases.
    if not hasattr(model, "_benchmark_reference_cache"):
        model._benchmark_reference_cache = {}
    key = (doi, t, p)
    if key not in model._benchmark_reference_cache:
        for c in components:
            curves.evaluate(doi, c, t)
            if model.props[c].Tc is None or t >= model.props[c].Tc:
                raise ValueError("measured T at or above shared model critical temperature")
        model.Psat = MethodType(lambda self, c, temperature: curves.evaluate(doi, c, temperature), model)
        for cache in ("_psat_cache", "_phi_sat_cache", "_poynting_cache"):
            getattr(model, cache).clear()
        values = model._gamma_phi_reference_factors(t, p)
        model._benchmark_reference_cache[key] = np.array([values[c] for c in components])
    reference = model._benchmark_reference_cache[key]
    gamma = np.exp(lng)
    target = x*gamma*reference/p
    inferred = np.full(len(components), np.nan)
    if measurement.get("y") is not None:
        y = np.array(measurement["y"])
        phi = np.array(list(model.vapor_fugacity_coefficients(t, p, dict(zip(components, y))).values()))
        positive = (y > 0) & (x > 0)
        inferred[positive] = np.log(y[positive]*phi[positive]*p/(x[positive]*reference[positive]))
    else:
        y = target/target.sum()
        for _ in range(100):
            phi = np.array(list(model.vapor_fugacity_coefficients(t, p, dict(zip(components, y))).values()))
            amounts = target/phi
            updated = amounts/amounts.sum()
            if np.max(np.abs(updated-y)) < 1e-10:
                y = updated
                break
            y = .5*(y+updated)
        else:
            raise ValueError("fixed-P vapor composition failed to converge")
        phi = np.array(list(model.vapor_fugacity_coefficients(t, p, dict(zip(components, y))).values()))
    closure = float(np.sum(target/phi))
    return inferred, closure-1, math.log(closure)


def score(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            groups[r["system"]].append(r)
    pair_rows = []
    for system, rr in sorted(groups.items()):
        errors = [e for r in rr for e in r.get("errors", []) if math.isfinite(e)]
        closures = [r["closure"] for r in rr if "closure" in r]
        entry = {"system": system, "states": len(rr)}
        if errors:
            entry.update(MAE=float(np.mean(np.abs(errors))), MSE=float(np.mean(np.square(errors))), component_values=len(errors))
        if closures:
            entry.update(closure_AARD_percent=float(np.mean(np.abs(closures)))*100)
        pair_rows.append(entry)
    result = {"systems": len(groups), "states": sum(r["states"] for r in pair_rows)}
    good = [r for r in pair_rows if "MAE" in r]
    if good:
        result.update(ln_gamma_systems=len(good), component_values=sum(r["component_values"] for r in good),
                      MAE=float(np.mean([r["MAE"] for r in good])), median_system_MAE=float(np.median([r["MAE"] for r in good])),
                      RMS=float(np.sqrt(np.mean([r["MSE"] for r in good]))))
    closures = [r["closure_AARD_percent"] for r in pair_rows if "closure_AARD_percent" in r]
    if closures:
        result["mean_system_bubble_equation_AARD_percent"] = float(np.mean(closures))
    return result, pair_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hanna-root", type=Path, required=True)
    parser.add_argument("--binary-baseline", type=Path, required=True)
    parser.add_argument("--fit-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-predictions", type=Path, help="Reuse completed HANNA state predictions; reference calculations and aggregation remain checked")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hanna_root = args.hanna_root.resolve()
    os.chdir(hanna_root)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    started = time.perf_counter()
    model = Hanna(hanna_root)
    checks = model.checks()
    print("HANNA loaded; official-interface and thermodynamic checks passed", flush=True)
    source = json.loads(SOURCE.read_text())
    ternary_path = ROOT/"data/source/activity_fitting/hanna_ternary_vle_measurements.json"
    ternary = json.loads(ternary_path.read_text())["rows"]
    cached_predictions = {}
    resumed_provenance = {}
    if args.resume_predictions:
        previous_summary = json.loads((args.resume_predictions/"summary.json").read_text())
        for path in (SOURCE, ternary_path):
            if hashlib.sha256(path.read_bytes()).hexdigest() != previous_summary["input_sha256"][str(path.relative_to(ROOT))]:
                raise ValueError("reference input changed since cached HANNA evaluation")
        current_commit = subprocess.check_output(["git", "-C", str(hanna_root), "rev-parse", "HEAD"], text=True).strip()
        if current_commit != previous_summary["hanna_commit"]:
            raise ValueError("HANNA checkpoint repository changed")
        for r in read_csv(args.resume_predictions/"points.csv"):
            if r["method"] == "HANNA" and r["status"] == "ok":
                key = (r["system"], float(r["T_K"]), tuple(json.loads(r["x"])))
                cached_predictions[key] = np.array(json.loads(r["predicted_ln_gamma"]))
        resumed_provenance = {"previous_points_sha256": hashlib.sha256((args.resume_predictions/"points.csv").read_bytes()).hexdigest(),
                              "previous_summary_sha256": hashlib.sha256((args.resume_predictions/"summary.json").read_bytes()).hexdigest(),
                              "cached_HANNA_states": len(cached_predictions)}

    def predict_states(components, states):
        keys = [(" / ".join(components), float(t), tuple(x)) for t, x in states]
        missing = sorted(set(keys)-set(cached_predictions))
        if missing:
            predicted = model.predict([db.get(c).smiles for c in components], [k[1] for k in missing], [k[2] for k in missing])
            cached_predictions.update(zip(missing, predicted))
        return np.array([cached_predictions[k] for k in keys])
    def raw_key(r):
        return (r["doi"], r["cas1"], r["cas2"], float(r["T_K"]), float(r["P_Pa"]), float(r["x1"]))
    old = read_csv(args.binary_baseline/"points.csv")
    successful = {m: {raw_key(r): r for r in old if r["method"] == m and r["status"] == "ok"} for m in METHOD_FILES}
    common = set.intersection(*(set(v) for v in successful.values()))
    grouped = defaultdict(list)
    for i, r in enumerate(source["rows"]):
        if raw_key(r) in common:
            grouped[tuple((r["cas1"], r["cas2"]))].append({**r, "x": [r["x1"], 1-r["x1"]],
                         "y": [r["y1"], 1-r["y1"]] if "y1" in r else None,
                         "cohort": "raw_binary", "point_id": f"binary-{i}"})
    for i, r in enumerate(ternary):
        grouped[tuple(r["components"])].append({**r, "cohort": "raw_ternary", "point_id": f"ternary-{i}"})
    results, excluded, reference_drift = [], [], []
    db = PropertyDatabase()
    for number, (components, rr) in enumerate(sorted(grouped.items()), 1):
        try:
            smiles = [db.get(c).smiles for c in components]
            predictions = predict_states(components, [(r["T_K"], r["x"]) for r in rr])
            acid = is_acid(db, components)
            # All four acid pairs have measured vapor compositions and saved
            # association-corrected targets. Use those targets directly rather
            # than initializing geometry calculations merely to reproduce them.
            needs_vapor = len(components) == 3 or any(not successful["UNIFAC"][raw_key(r)].get("inferred_ln_gamma1") for r in rr)
            vapor = SharedVapor(list(components), db, acid) if needs_vapor else None
            curves = SourceCurves(source)
        except Exception as error:
            excluded.append({"system": " / ".join(components), "scope": "system", "reason": str(error)})
            continue
        unifac = {}
        if len(components) == 3:
            for method, filename in METHOD_FILES.items():
                try:
                    table = UNIFACModel(str(ROOT/"data"/filename))
                    groups = {c: get_unifac_groups(c, smiles=s, variant=method) for c, s in zip(components, smiles)}
                    if any(not g for g in groups.values()):
                        raise ValueError("missing structural groups")
                    main_groups = set(g.main_group for subgroups in groups.values() for sg in subgroups for g in (table.subgroups[sg],))
                    if any((i, j) not in table.interactions for i in main_groups for j in main_groups if i != j):
                        raise ValueError("missing group interaction")
                    cls = type("Benchmark"+method, (UNIFACThermodynamics,), {"unifac_variant": method})
                    unifac[method] = cls(list(components), db=db, unifac_groups=groups, unifac_data_path=str(ROOT/"data"/filename))
                except Exception as error:
                    excluded.append({"system": " / ".join(components), "method": method, "scope": "model", "reason": str(error)})
        for index, r in enumerate(rr):
            for method in METHODS:
                base = {"cohort": r["cohort"], "system": " / ".join(components), "method": method,
                        "point_id": r["point_id"], "doi": r["doi"], "T_K": r["T_K"], "P_Pa": r["P_Pa"], "x": r["x"], "y": r["y"]}
                try:
                    if method == "HANNA":
                        lng = predictions[index]
                    elif len(components) == 2:
                        previous = successful[method][raw_key(r)]
                        lng = np.array([float(previous[f"predicted_ln_gamma{i}"]) for i in (1, 2)])
                    else:
                        gamma = unifac[method].activity_coefficients(r["T_K"], dict(zip(components, r["x"])))
                        lng = np.log([gamma[c] for c in components])
                    if len(components) == 2 and all(successful["UNIFAC"][raw_key(r)].get(f"inferred_ln_gamma{i}") for i in (1, 2)):
                        inferred = np.array([float(successful["UNIFAC"][raw_key(r)][f"inferred_ln_gamma{i}"]) for i in (1, 2)])
                        closure_total = float(np.sum(np.array(r["y"])*np.exp(lng-inferred)))
                        closure, ln_closure = closure_total-1, math.log(closure_total)
                    else:
                        inferred, closure, ln_closure = equilibrium_residual(vapor, curves, r, lng)
                    if len(components) == 2 and method in METHOD_FILES and all(successful[method][raw_key(r)].get(f"inferred_ln_gamma{i}") for i in (1, 2)):
                        cached = np.array([float(successful[method][raw_key(r)][f"inferred_ln_gamma{i}"]) for i in (1, 2)])
                        reference_drift.append(float(np.max(np.abs(cached-inferred))))
                    errors = lng-inferred
                    results.append({**base, "status": "ok", "predicted_ln_gamma": lng.tolist(),
                                    "inferred_ln_gamma": [float(v) if math.isfinite(v) else None for v in inferred],
                                    "errors": [float(v) for v in errors if math.isfinite(v)], "closure": closure, "ln_closure": ln_closure})
                except Exception as error:
                    results.append({**base, "status": str(error)})
        print(f"Raw system {number}/{len(grouped)} complete", flush=True)
    write_csv(out/"raw_checkpoint.csv", results)
    agreement = read_csv(args.fit_baseline/"reference_agreement.csv")
    eligible = {r["pair"] for r in agreement if r["both_declared_ranges"] == "True" and float(r["mae_ln_gamma"]) <= .1}
    fit_points = read_csv(args.fit_baseline/"agreement_points.csv")
    fit_groups = defaultdict(list)
    for r in fit_points:
        if f"{r['cas1']} / {r['cas2']}" in eligible:
            fit_groups[(r["cas1"], r["cas2"])].append(r)
    for components, rr in sorted(fit_groups.items()):
        states = sorted({(float(r["T_K"]), float(r["x1"])) for r in rr})
        predictions = predict_states(components, [(r[0], [r[1], 1-r[1]]) for r in states])
        by_state = dict(zip(states, predictions))
        references = {}
        for r in rr:
            t, x = float(r["T_K"]), float(r["x1"])
            state = (r["reference"], t, x)
            ref = [float(r[f"reference_ln_gamma{i}"]) for i in (1, 2)]
            references[state] = ref
            if r["fully_parameterized"] != "True":
                continue
            pred = np.array([float(r[f"predicted_ln_gamma{i}"]) for i in (1, 2)])
            results.append({"cohort": "close_fits", "system": " / ".join(components), "method": r["method"],
                            "point_id": f"{state}", "T_K": t, "x": [x, 1-x], "reference": state[0], "status": "ok",
                            "predicted_ln_gamma": pred.tolist(), "inferred_ln_gamma": ref, "errors": (pred-ref).tolist()})
        for (reference, t, x), ref in references.items():
            pred = by_state[t, x]
            results.append({"cohort": "close_fits", "system": " / ".join(components), "method": "HANNA",
                            "point_id": f"{(reference,t,x)}", "T_K": t, "x": [x, 1-x], "reference": reference, "status": "ok",
                            "predicted_ln_gamma": pred.tolist(), "inferred_ln_gamma": ref, "errors": (pred-ref).tolist()})
    summaries, pair_scores = {}, []
    for cohort in ("raw_binary", "raw_ternary", "close_fits"):
        subset = [r for r in results if r["cohort"] == cohort]
        keys = {m: {(r["system"], r["point_id"]) for r in subset if r["method"] == m and r["status"] == "ok"} for m in METHODS}
        shared = set.intersection(*keys.values())
        summaries[cohort] = {}
        for m in METHODS:
            rr = [r for r in subset if r["method"] == m and (r["system"], r["point_id"]) in shared]
            totals, per_pair = score(rr)
            summaries[cohort][m] = totals
            pair_scores.extend({"cohort": cohort, "method": m, **r} for r in per_pair)
    pairwise = {}
    for cohort in ("raw_binary", "raw_ternary", "close_fits"):
        subset = [r for r in results if r["cohort"] == cohort and r["status"] == "ok"]
        keys = {m: {(r["system"], r["point_id"]) for r in subset if r["method"] == m} for m in METHODS}
        pairwise[cohort] = {}
        for m in METHOD_FILES:
            shared = keys[m] & keys["HANNA"]
            pairwise[cohort][m] = {method: score([r for r in subset if r["method"] == method and (r["system"], r["point_id"]) in shared])[0] for method in (m, "HANNA")}
    summary = {"protocol": __doc__, "scores": summaries, "pairwise": pairwise, "checks": checks,
               "resumed_provenance": resumed_provenance,
               "cached_raw_binary_points": len(common), "close_fit_eligible_pairs": len(eligible),
               "raw_ternary_input_states": len(ternary), "embeddings_cached": len(model.embeddings),
               "max_binary_inferred_reference_change": max(reference_drift, default=0),
               "ensemble_parameters": sum(p.numel() for p in model.predictor.model.parameters()),
               "encoder_parameters": sum(p.numel() for p in model.predictor.ChemBERTA.parameters()),
               "elapsed_seconds_excluding_imports": time.perf_counter()-started,
               "hanna_commit": subprocess.check_output(["git", "-C", str(hanna_root), "rev-parse", "HEAD"], text=True).strip(),
               "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE, ternary_path, Path(__file__))},
               "baseline_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.binary_baseline/"points.csv", args.fit_baseline/"agreement_points.csv", args.fit_baseline/"reference_agreement.csv")},
               "versions": {p: version(p) for p in ("torch", "numpy", "chemicals", "transformers", "rdkit", "scikit-learn")},
               "limitations": "Released ensemble trained on all available DDB data; overlap unknown. Fitted references are not measurements. Bubble-equation residual is not a solved pressure error. Common matched states only; no liquid fitting. Binary-pair and ternary-system weights equal within separate cohorts."}
    for row in results:
        for field in ("x", "y", "errors", "predicted_ln_gamma", "inferred_ln_gamma"):
            if field in row:
                row[field] = json.dumps(row[field], allow_nan=False)
    write_csv(out/"points.csv", results)
    write_csv(out/"pairs.csv", pair_scores)
    write_csv(out/"excluded.csv", excluded)
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
