"""Compare pretrained GRAPPA with Perry 2-8, Perry 2-10, and CoolProp.

Run with the Python environment containing GRAPPA's dependencies, chemicals,
and CoolProp. For example (the output directory must not already exist)::

    python scripts/psat_experiments/benchmark_grappa.py \
        --grappa-root /tmp/grappa-checkout --output-dir /tmp/grappa-results

Continuous references receive ten equally spaced temperatures, including the
endpoints of the intersection of their source/known-liquid domains with
250--600 K and reference pressures 1--1e7 Pa. Perry 2-10 uses its actual rows.
No GRAPPA parameters are fitted to or anchored on the reference values.
All selection decisions and evaluated points are saved. Sampling is deterministic.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import CoolProp.CoolProp as CP
import numpy as np
from chemicals.identifiers import search_chemical
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.optimize import brentq
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from compound_identity import parse_formula_counts
from perry_properties import PerryPropertyLibrary

T_LOW, T_HIGH = 250.0, 600.0
P_LOW, P_HIGH = 1.0, 1.0e7
SOURCES = ("Perry 2-8", "Perry 2-10", "CoolProp HEOS")


@dataclass
class Case:
    source: str
    cas: str
    name: str
    smiles: str
    temperatures: np.ndarray
    pressures_pa: np.ndarray
    tc: float | None
    tm: float | None


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["source", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def positive(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def clipped_samples(evaluate, low, high):
    """Clip a monotone continuous reference before uniformly sampling it."""
    low, high = max(T_LOW, low), min(T_HIGH, high)
    if high <= low:
        raise ValueError("no temperature overlap")
    p_low, p_high = evaluate(low), evaluate(high)
    if not (math.isfinite(p_low) and math.isfinite(p_high) and 0 < p_low < p_high):
        raise ValueError("invalid/non-increasing reference endpoints")
    if p_high <= P_LOW or p_low >= P_HIGH:
        raise ValueError("no pressure overlap")
    if p_low < P_LOW:
        low = brentq(lambda t: math.log(evaluate(t) / P_LOW), low, high)
    if p_high > P_HIGH:
        high = brentq(lambda t: math.log(evaluate(t) / P_HIGH), low, high)
    if high <= low:
        raise ValueError("no interval after pressure clipping")
    temperatures = np.linspace(low, high, 10)
    pressures = np.asarray([evaluate(float(t)) for t in temperatures])
    if not (np.all(np.isfinite(pressures)) and np.all(np.diff(pressures) > 0)):
        raise ValueError("invalid/non-increasing reference samples")
    if np.any(pressures < P_LOW * (1 - 1e-9)) or np.any(pressures > P_HIGH * (1 + 1e-9)):
        raise ValueError("reference pressure clipping failed")
    return temperatures, pressures


def metrics(rows):
    good = [r for r in rows if r["prediction_status"] == "ok"]
    result = {"points": len(rows), "compounds": len({r["cas"] for r in rows}),
              "prediction_failures": len(rows) - len(good)}
    if not good:
        return result
    absolute = np.asarray([r["absolute_percent_error"] for r in good])
    signed = np.asarray([r["signed_percent_error"] for r in good])
    logarithmic = np.asarray([r["ln_error"] for r in good])
    grouped = defaultdict(list)
    for r in good:
        grouped[r["cas"]].append(r["absolute_percent_error"])
    compound_errors = np.asarray([np.mean(v) for v in grouped.values()])
    result.update({
        "point_mean_absolute_percent_error": float(np.mean(absolute)),
        "point_median_absolute_percent_error": float(np.median(absolute)),
        "point_p90_absolute_percent_error": float(np.quantile(absolute, 0.9)),
        "point_p95_absolute_percent_error": float(np.quantile(absolute, 0.95)),
        "point_max_absolute_percent_error": float(np.max(absolute)),
        "point_signed_percent_bias": float(np.mean(signed)),
        "point_ln_mae": float(np.mean(np.abs(logarithmic))),
        "point_ln_rmse": float(np.sqrt(np.mean(logarithmic**2))),
        "compound_mean_absolute_percent_error": float(np.mean(compound_errors)),
        "compound_median_absolute_percent_error": float(np.median(compound_errors)),
        "compound_p90_absolute_percent_error": float(np.quantile(compound_errors, 0.9)),
        "compound_p95_absolute_percent_error": float(np.quantile(compound_errors, 0.95)),
        "point_fraction_within_10_percent": float(np.mean(absolute <= 10)),
        "point_fraction_within_20_percent": float(np.mean(absolute <= 20)),
        "point_fraction_within_50_percent": float(np.mean(absolute <= 50)),
    })
    return result


def pressure_bin(pressure):
    if pressure < 1000:
        return "1 Pa to <1 kPa"
    if pressure < 100000:
        return "1 to <100 kPa"
    return "100 kPa to 10 MPa"


def collect_cases(mol_to_pyg):
    perry_path = REPO_ROOT / "data/perry_properties.json"
    table_path = REPO_ROOT / "data/perry_table_2_10_vapor_pressure.json"
    perry = json.loads(perry_path.read_text())["chemicals"]
    table_payload = json.loads(table_path.read_text())
    table = table_payload["chemicals"]
    local = json.loads((REPO_ROOT / "data/chemicals.json").read_text())["chemicals"]
    local_by_cas = {row["CAS"]: row for row in local.values() if row.get("CAS")}
    library = PerryPropertyLibrary()
    audit, cases, identities = [], [], {}
    counts = {"Perry 2-8": len(perry), "Perry 2-10": len(table)}

    def excluded(source, cas, name, reason, **extra):
        audit.append({"source": source, "cas": cas, "name": name, "reason": reason, **extra})

    def identity(cas, source_formula=None):
        if cas not in identities:
            try:
                entry = search_chemical(cas)
                if entry.CASs != cas:
                    raise ValueError(f"CAS mismatch: resolved {entry.CASs}")
                mol = Chem.MolFromSmiles(entry.smiles)
                if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                    raise ValueError("invalid or disconnected molecular structure")
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                mol_to_pyg(mol, 293.15)  # Authoritative GRAPPA chemistry validation.
                identities[cas] = (smiles, rdMolDescriptors.CalcMolFormula(mol), None)
            except Exception as error:
                identities[cas] = (None, None, f"identity/model-input rejection: {error}")
        smiles, formula, error = identities[cas]
        if error:
            raise ValueError(error)
        if source_formula:
            source_counts = parse_formula_counts(source_formula)
            if source_counts is None or source_counts != parse_formula_counts(formula):
                raise ValueError(f"source/structure formula mismatch: {source_formula} versus {formula}")
        return smiles

    def bounds(cas):
        tm = positive(table.get(cas, {}).get("Tm_K")) or positive(local_by_cas.get(cas, {}).get("Tm"))
        tc = positive(perry.get(cas, {}).get("critical_constants", {}).get("Tc_K"))
        return tm, tc

    for cas, row in sorted(perry.items()):
        name = row.get("name") or cas
        try:
            smiles = identity(cas, row.get("formula"))
            correlations = row.get("vapor_pressure") or []
            if len(correlations) != 1:
                raise ValueError("expected exactly one Perry 2-8 correlation")
            correlation = correlations[0]
            if library.vapor_pressure_coefficients(correlation) is None:
                raise ValueError("invalid Perry coefficients")
            tm, tc = bounds(cas)
            low = max(float(correlation["T_min_K"]), tm or 0)
            high = min(float(correlation["T_max_K"]), tc * (1 - 1e-7) if tc else math.inf)

            def evaluate(t, cas=cas):
                value = library.vapor_pressure_bar(cas, t)
                if value is None:
                    raise ValueError("Perry evaluator rejected in-range temperature")
                return value.value * 100000.0

            temperatures, pressures = clipped_samples(evaluate, low, high)
            cases.append(Case("Perry 2-8", cas, name, smiles, temperatures, pressures, tc, tm))
        except (ValueError, KeyError, OverflowError, ZeroDivisionError) as error:
            excluded("Perry 2-8", cas, name, str(error))

    collisions = table_payload.get("metadata", {}).get("duplicate_cas_collisions", {})
    for cas, row in sorted(table.items()):
        name = row.get("name") or cas
        try:
            if cas in collisions:
                raise ValueError("ambiguous identity recorded in source metadata")
            smiles = identity(cas, row.get("formula"))
            tm, tc = bounds(cas)
            accepted = []
            for point in row.get("vapor_pressure", []):
                t, p = positive(point.get("T_K")), positive(point.get("P_bar"))
                p = p * 100000.0 if p else None
                reason = None
                if t is None or p is None:
                    reason = "invalid source point"
                elif not T_LOW <= t <= T_HIGH:
                    reason = "outside GRAPPA temperature range"
                elif not P_LOW <= p <= P_HIGH:
                    reason = "outside GRAPPA pressure range"
                elif tm and t < tm:
                    reason = "below known melting temperature"
                elif tc and t >= tc:
                    reason = "at/above known critical temperature"
                if reason:
                    excluded("Perry 2-10", cas, name, reason, temperature_K=t, pressure_Pa=p, scope="point")
                else:
                    accepted.append((t, p))
            if not accepted:
                raise ValueError("no eligible table points")
            accepted.sort()
            temperatures, pressures = np.asarray(accepted).T
            cases.append(Case("Perry 2-10", cas, name, smiles, temperatures, pressures, tc, tm))
        except (ValueError, KeyError) as error:
            excluded("Perry 2-10", cas, name, str(error))
    for row in table_payload.get("unresolved", []):
        excluded("Perry 2-10", "", row.get("name") or row.get("table_name") or "unresolved",
                 "source row has no validated CAS", scope="unresolved_source_row")

    fluids = sorted(CP.get_global_param_string("FluidsList").split(","))
    counts["CoolProp HEOS"] = len(fluids)
    seen_cas = set()
    for fluid in fluids:
        cas = ""
        try:
            if CP.get_fluid_param_string(fluid, "pure").lower() != "true":
                raise ValueError("pseudo-pure fluid/mixture")
            cas = CP.get_fluid_param_string(fluid, "CAS").strip()
            smiles = identity(cas)
            if cas in seen_cas:
                raise ValueError("duplicate CAS in CoolProp fluid list")
            seen_cas.add(cas)
            state = CP.AbstractState("HEOS", fluid)
            tc = float(state.T_critical())
            tm = float(state.Ttriple())
            low = max(tm, float(state.Tmin()))
            high = tc * (1 - 1e-7)

            def evaluate(t, state=state):
                state.update(CP.QT_INPUTS, 0.0, float(t))
                return float(state.p())

            temperatures, pressures = clipped_samples(evaluate, low, high)
            cases.append(Case("CoolProp HEOS", cas, fluid, smiles, temperatures, pressures, tc, tm))
        except (ValueError, KeyError, RuntimeError) as error:
            excluded("CoolProp HEOS", cas, fluid, str(error))
    return cases, audit, counts


def report(summary, compound_rows):
    lines = ["# GRAPPA vapor-pressure benchmark", "", summary["protocol"], "",
             "All percentage errors use reference pressure as the denominator. Compound error is the mean absolute percentage error over that compound's points; the headline is its median across compounds.", "",
             "| Source | Compounds | Points | Median compound error % | Mean compound error % | Point p95 error % | ln(P) MAE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for source, row in summary["sources"].items():
        if "compound_median_absolute_percent_error" not in row:
            continue
        lines.append(f"| {source} | {row['compounds']} | {row['points']} | {row['compound_median_absolute_percent_error']:.3f} | {row['compound_mean_absolute_percent_error']:.3f} | {row['point_p95_absolute_percent_error']:.3f} | {row['point_ln_mae']:.4f} |")
    lines += ["", "## Worst compounds by mean absolute percentage error", "",
              "| Source | CAS | Compound | Points | Mean error % |", "|---|---|---|---:|---:|"]
    for source in SOURCES:
        selected = sorted((r for r in compound_rows if r["source"] == source and r.get("point_mean_absolute_percent_error") is not None),
                          key=lambda r: r["point_mean_absolute_percent_error"], reverse=True)[:10]
        for row in selected:
            lines.append(f"| {source} | {row['cas']} | {row['name']} | {row['points']} | {row['point_mean_absolute_percent_error']:.3f} |")
    lines += ["", "## Limitations", "",
              "These are comparisons with tabulated reference values, fitted correlations, and a reference EOS. Sources and compounds overlap; do not add their counts as independent validation cases. GRAPPA's released model was trained on all its curated DDB data, and training-compound overlap is unknown. No fitting or reference-pressure anchoring was used.", "",
              "The compound metrics include valid predictions only; prediction-failure counts are reported separately in summary.json and all failed points remain in points.csv. Unknown melting temperatures are not invented; melting bounds are applied only when available. The near-critical endpoint is retained at Tc*(1-1e-7), without an additional reduced-temperature cutoff.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grappa-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    grappa_root = args.grappa_root.resolve()
    output_dir = args.output_dir.resolve()
    if not (grappa_root / "models/GRAPPA_state_dict.pt").is_file():
        parser.error("--grappa-root must contain models/GRAPPA_state_dict.pt")
    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Published utilities select CPU when CUDA is hidden.
    torch.set_num_threads(1)
    sys.path.insert(0, str(grappa_root))
    from src.utils import GRAPPAantoine, mol_to_pyg

    started = time.perf_counter()
    cases, audit, counts = collect_cases(mol_to_pyg)
    print(f"Reference collection: {len(cases)} source/compound cases, {sum(len(c.temperatures) for c in cases)} points", flush=True)
    reference_seconds = time.perf_counter() - started
    write_csv(output_dir / "excluded.csv", audit)
    smiles = sorted({case.smiles for case in cases})
    if not smiles:
        raise RuntimeError("No eligible reference cases")
    original_cwd = Path.cwd()
    started = time.perf_counter()
    try:
        os.chdir(grappa_root)  # Upstream model loader uses a relative checkpoint path.
        with torch.inference_mode():
            predictor = GRAPPAantoine()
            parameters = np.asarray(predictor(smiles), dtype=float)
    finally:
        os.chdir(original_cwd)
    inference_seconds = time.perf_counter() - started
    if parameters.shape != (len(smiles), 3) or not np.all(np.isfinite(parameters)):
        raise RuntimeError("Invalid model parameter output")
    parameters_by_smiles = dict(zip(smiles, parameters))
    points, compounds = [], []
    for case in cases:
        a, b, c = parameters_by_smiles[case.smiles]
        case_points = []
        for t, reference_pa in zip(case.temperatures, case.pressures_pa):
            row = {"source": case.source, "cas": case.cas, "name": case.name, "smiles": case.smiles,
                   "temperature_K": float(t), "reference_pressure_Pa": float(reference_pa),
                   "reference_pressure_bin": pressure_bin(reference_pa),
                   "Tc_K": case.tc, "Tm_or_Ttriple_K": case.tm,
                   "Tr": float(t / case.tc) if case.tc else None,
                   "A": float(a), "B": float(b), "C": float(c), "prediction_status": "ok"}
            try:
                if t + c <= 0:
                    raise ValueError("non-positive Antoine denominator")
                log_prediction_pa = a - b / (t + c) + math.log(1000.0)
                log_error = log_prediction_pa - math.log(reference_pa)
                prediction_pa = math.exp(log_prediction_pa)
                signed = 100.0 * math.expm1(log_error)
                if not math.isfinite(prediction_pa) or prediction_pa <= 0 or not math.isfinite(signed):
                    raise ValueError("invalid predicted pressure/error")
                row.update({"predicted_pressure_Pa": prediction_pa, "ln_error": log_error,
                            "signed_percent_error": signed, "absolute_percent_error": abs(signed)})
            except (ValueError, OverflowError) as error:
                row["prediction_status"] = str(error)
            case_points.append(row)
        compounds.append({"source": case.source, "cas": case.cas, "name": case.name, "smiles": case.smiles,
                          "sample_Tmin_K": float(case.temperatures.min()), "sample_Tmax_K": float(case.temperatures.max()),
                          "known_lower_phase_bound": case.tm is not None, **metrics(case_points)})
        points.extend(case_points)
    summaries = {source: metrics([r for r in points if r["source"] == source]) for source in SOURCES}
    stratified = {}
    for source in SOURCES:
        selected = [r for r in points if r["source"] == source]
        stratified[source] = {
            "pressure": {label: metrics([r for r in selected if r["reference_pressure_bin"] == label])
                         for label in ("1 Pa to <1 kPa", "1 to <100 kPa", "100 kPa to 10 MPa")},
            "reduced_temperature": {"Tr <= 0.85": metrics([r for r in selected if r["Tr"] is not None and r["Tr"] <= 0.85]),
                                    "Tr > 0.85": metrics([r for r in selected if r["Tr"] is not None and r["Tr"] > 0.85]),
                                    "Tc unknown": metrics([r for r in selected if r["Tr"] is None])},
        }
    source_cas = {source: {c.cas for c in cases if c.source == source} for source in SOURCES}
    summary = {
        "protocol": "10 uniformly spaced temperatures per continuous reference, including endpoints after source-domain, 250-600 K, 1-1e7 Pa, known melting/triple-point, and critical-point clipping. Perry 2-10 uses all eligible actual table points. One GRAPPA Antoine prediction per distinct canonical SMILES; pressures evaluated directly with NumPy-compatible scalar arithmetic. CPU, one thread, deterministic sampling; no fitting or anchoring.",
        "sources": summaries, "stratified": stratified, "input_records": counts,
        "unique_compounds_cas": len({c.cas for c in cases}), "unique_structures_smiles": len(smiles),
        "source_overlap": {f"{a} / {b}": len(source_cas[a] & source_cas[b]) for i, a in enumerate(SOURCES) for b in SOURCES[i+1:]},
        "exclusions_by_source_reason": {source: dict(Counter(r["reason"] for r in audit if r["source"] == source)) for source in SOURCES},
        "timing_seconds": {"reference_collection": reference_seconds, "model_load_and_antoine_prediction": inference_seconds,
                           "total_excluding_imports": time.perf_counter() - started + reference_seconds},
        "trainable_parameters": sum(p.numel() for p in predictor.model.parameters()),
        "versions": {name: version(name) for name in ("torch", "torch-geometric", "rdkit", "numpy", "chemicals", "CoolProp")},
        "grappa_commit": subprocess.check_output(["git", "-C", str(grappa_root), "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": hashlib.sha256((grappa_root / "models/GRAPPA_state_dict.pt").read_bytes()).hexdigest(),
        "source_file_sha256": {name: hashlib.sha256((REPO_ROOT / "data" / name).read_bytes()).hexdigest()
                               for name in ("perry_properties.json", "perry_table_2_10_vapor_pressure.json", "chemicals.json")},
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "argv": sys.argv,
    }
    write_csv(output_dir / "points.csv", points)
    write_csv(output_dir / "compounds.csv", compounds)
    write_csv(output_dir / "antoine_parameters.csv", [{"smiles": s, "A": float(v[0]), "B": float(v[1]), "C": float(v[2])} for s, v in parameters_by_smiles.items()])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output_dir / "report.md").write_text(report(summary, compounds))
    print(json.dumps({"sources": summaries, "timing_seconds": summary["timing_seconds"], "output_dir": str(output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
