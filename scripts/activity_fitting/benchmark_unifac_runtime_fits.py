"""Compare UNIFAC families with runtime-selected NRTL/UNIQUAC ln(gamma).

This is a fitted-model agreement benchmark, not validation against raw VLE/LLE.
Disabled and superseded interactions are excluded by the runtime loaders.
Bounded fits use Tmin, midpoint, Tmax; unspecified ranges use 298.15, 323.15,
348.15 K and are reported separately. Fifteen symmetric compositions probe
bulk and dilute limits; reference-only maxima of |gE/(RT)| add sensitive points.
Every UNIFAC method receives the same points for a given fitted reference.
Missing interactions are flagged even when the runtime zero fallback evaluates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from chemicals.identifiers import search_chemical
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chemical_properties import ChemicalProperties
from compiled_unifac import CompiledUNIFACBackend
from interaction_parameters import (
    _nrtl_interactions,
    _uniquac_interactions,
    activity_interaction_temperature_range,
)
from thermodynamics_models.nrtl_uniquac import NRTLThermodynamics, UNIQUACThermodynamics
from unifac import UNIFACModel, get_unifac_groups

METHOD_FILES = {
    "UNIFAC": "unifac_params.json",
    "UNIFAC2": "unifac2/UNIFAC2rq.csv",
    "UNIFDMD": "unifac_dmd.txt",
    "UNIFM2": "unifac2/UNIFM2rq.xlsx",
    "UNIFNIST": "nist_modified_unifac_params.json",
}
METHODS = tuple(METHOD_FILES)
BASE_X = np.array([.0001, .001, .01, .05, .1, .2, .35, .5, .65, .8, .9, .95, .99, .999, .9999])
FALLBACK_T = np.array([298.15, 323.15, 348.15])
SENSITIVITY_X = np.linspace(.02, .98, 49)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["reason"])
        writer.writeheader()
        writer.writerows(rows)


class IdentityDatabase:
    """Only identity is needed by the runtime activity-coefficient evaluators."""

    @lru_cache(maxsize=None)
    def get(self, cas):
        record = search_chemical(cas)
        if record.CASs != cas:
            raise ValueError(f"CAS resolves to a different identity: {cas} -> {record.CASs}")
        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError(f"Unresolved/disconnected molecular structure: {cas}")
        return ChemicalProperties(symbol=cas, name=record.common_name, formula=record.formula,
                                  CAS=cas, smiles=Chem.MolToSmiles(mol, isomericSmiles=True), MW=record.MW)


def reference_values(thermo, pair, temperatures, compositions):
    values = []
    for temperature, x1 in zip(temperatures, compositions):
        gamma = thermo.activity_coefficients(float(temperature), {pair[0]: float(x1), pair[1]: float(1-x1)})
        vector = np.array([gamma[c] for c in pair], dtype=float)
        if np.any(vector <= 0) or not np.all(np.isfinite(vector)):
            raise ValueError("Reference gamma is non-positive/non-finite")
        values.append(np.log(vector))
    return np.array(values)


def select_points(thermo, pair, record):
    bounds = activity_interaction_temperature_range(record)
    temperatures = np.unique(np.linspace(*bounds, 3)) if bounds else FALLBACK_T.copy()
    candidates_t = np.repeat(temperatures, len(SENSITIVITY_X))
    candidates_x = np.tile(SENSITIVITY_X, len(temperatures))
    lng = reference_values(thermo, pair, candidates_t, candidates_x)
    g_excess = candidates_x * lng[:, 0] + (1-candidates_x) * lng[:, 1]
    peaks = [SENSITIVITY_X[np.argmax(np.abs(g_excess[i*len(SENSITIVITY_X):(i+1)*len(SENSITIVITY_X)]))]
             for i in range(len(temperatures))]
    compositions = np.unique(np.round(np.r_[BASE_X, peaks], 10))
    t = np.repeat(temperatures, len(compositions))
    x = np.tile(compositions, len(temperatures))
    return t, x, reference_values(thermo, pair, t, x), bounds, temperatures, compositions


def case_metrics(errors, x, reference):
    absolute = np.abs(errors)
    bulk = (x >= .01) & (x <= .99)
    dilute = np.r_[absolute[x < .01, 0], absolute[x > .99, 1]]
    return {
        "states": len(x), "ln_gamma_values": errors.size,
        "mae_ln_gamma": float(absolute.mean()),
        "mse_ln_gamma": float(np.mean(errors**2)),
        "rmse_ln_gamma": float(np.sqrt(np.mean(errors**2))),
        "max_abs_ln_gamma": float(absolute.max()),
        "mean_signed_ln_gamma": float(errors.mean()),
        "bulk_mae_ln_gamma": float(absolute[bulk].mean()),
        "dilute_solute_mae_ln_gamma": float(dilute.mean()),
        "ideal_mae_ln_gamma": float(np.abs(reference).mean()),
        "reference_rms_ln_gamma": float(np.sqrt(np.mean(reference**2))),
    }


def aggregate(cases):
    if not cases:
        return {"reference_cases": 0}
    maes = np.array([r["mae_ln_gamma"] for r in cases])
    return {
        "reference_cases": len(cases), "states": sum(r["states"] for r in cases),
        "ln_gamma_values": sum(r["ln_gamma_values"] for r in cases),
        "mean_pair_mae_ln_gamma": float(maes.mean()),
        "median_pair_mae_ln_gamma": float(np.median(maes)),
        "p90_pair_mae_ln_gamma": float(np.quantile(maes, .9)),
        "p95_pair_mae_ln_gamma": float(np.quantile(maes, .95)),
        "pair_weighted_rmse_ln_gamma": float(np.sqrt(np.mean([r["mse_ln_gamma"] for r in cases]))),
        "mean_bulk_pair_mae_ln_gamma": float(np.mean([r["bulk_mae_ln_gamma"] for r in cases])),
        "median_dilute_solute_pair_mae_ln_gamma": float(np.median([r["dilute_solute_mae_ln_gamma"] for r in cases])),
        "mean_dilute_solute_pair_mae_ln_gamma": float(np.mean([r["dilute_solute_mae_ln_gamma"] for r in cases])),
        "fraction_pairs_mae_below_0_1": float(np.mean(maes < .1)),
        "fraction_pairs_mae_below_0_25": float(np.mean(maes < .25)),
        "fraction_pairs_better_than_ideal": float(np.mean([r["mae_ln_gamma"] < r["ideal_mae_ln_gamma"] for r in cases])),
    }


def aggregate_unique_pairs(cases):
    """Average fitted-reference scores within each binary before aggregation."""
    grouped = defaultdict(list)
    for row in cases:
        grouped[row["pair"]].append(row)
    fields = ("mae_ln_gamma", "mse_ln_gamma", "bulk_mae_ln_gamma",
              "dilute_solute_mae_ln_gamma", "ideal_mae_ln_gamma")
    averaged = [{**{k: float(np.mean([r[k] for r in rows])) for k in fields},
                 "states": sum(r["states"] for r in rows),
                 "ln_gamma_values": sum(r["ln_gamma_values"] for r in rows)}
                for rows in grouped.values()]
    result = aggregate(averaged)
    result["unique_binary_pairs"] = result.pop("reference_cases")
    result["fitted_reference_cases"] = len(cases)
    return result


def grand_totals(case_rows):
    good = [r for r in case_rows if r["status"] == "ok" and r["fully_parameterized"]]
    result = {}
    for label, selected in (
        ("common_all_five", good),
        ("common_all_five_no_reference_clipping", [r for r in good if not r["reference_clipped"]]),
        ("common_all_five_declared_no_clipping", [r for r in good if r["declared_temperature_range"] and not r["reference_clipped"]]),
    ):
        common = set.intersection(*({(r["reference"], r["pair"]) for r in selected if r["method"] == m} for m in METHODS))
        result[label] = {method: aggregate_unique_pairs([r for r in selected if r["method"] == method and (r["reference"], r["pair"]) in common]) for method in METHODS}
    families = {}
    for old, new in (("UNIFAC", "UNIFAC2"), ("UNIFDMD", "UNIFM2")):
        common = set.intersection(*({(r["reference"],r["pair"]) for r in good if r["method"] == m} for m in (old,new)))
        selected = [r for r in good if (r["reference"],r["pair"]) in common and r["method"] in (old,new)]
        errors = {method: defaultdict(list) for method in (old,new)}
        for row in selected:
            errors[row["method"]][row["pair"]].append(row["mae_ln_gamma"])
        pairs = errors[old].keys() & errors[new].keys()
        wins = sum(np.mean(errors[new][p]) < np.mean(errors[old][p]) and not math.isclose(np.mean(errors[new][p]), np.mean(errors[old][p]), rel_tol=1e-12, abs_tol=1e-12) for p in pairs)
        families[f"{old} versus {new}"] = {
            old: aggregate_unique_pairs([r for r in selected if r["method"] == old]),
            new: aggregate_unique_pairs([r for r in selected if r["method"] == new]),
            "new_model_unique_pair_wins": int(wins),
            "unique_pair_ties": sum(math.isclose(np.mean(errors[new][p]), np.mean(errors[old][p]), rel_tol=1e-12, abs_tol=1e-12) for p in pairs),
        }
    result["family_pairwise"] = families
    result["weighting"] = "Equal weight per distinct unordered CAS pair. Average each pair's NRTL/UNIQUAC MAE and MSE across its available included fitted references before aggregating across pairs; RMS is sqrt(mean pair MSE). Shared pairs are not counted twice."
    return result


def summaries(case_rows, reference_inventory, agreement, agreement_cases):
    result = {}
    agreement_pairs = {tuple(r["pair"].split(" / ")) for r in agreement if r["mae_ln_gamma"] <= .1 and r["both_declared_ranges"]}
    for reference in ("NRTL", "UNIQUAC"):
        eligible = [r for r in case_rows if r["reference"] == reference and r["status"] == "ok"]
        qualified = [r for r in eligible if r["fully_parameterized"]]
        common = set.intersection(*({r["pair"] for r in qualified if r["method"] == m} for m in METHODS))
        cohorts = {
            "all_runtime_fallbacks_flagged": eligible,
            "fully_parameterized_each_method": qualified,
            "common_all_five_methods": [r for r in qualified if r["pair"] in common],
            "common_all_five_declared_temperature_ranges": [r for r in qualified if r["pair"] in common and r["declared_temperature_range"]],
            "common_all_five_unspecified_temperature_ranges": [r for r in qualified if r["pair"] in common and not r["declared_temperature_range"]],
            "common_all_five_declared_no_clipping": [r for r in qualified if r["pair"] in common and r["declared_temperature_range"] and not r["reference_clipped"]],
            "common_all_five_no_reference_clipping": [r for r in qualified if r["pair"] in common and not r["reference_clipped"]],
        }
        result[reference] = {name: {method: aggregate([r for r in cohort if r["method"] == method]) for method in METHODS}
                             for name, cohort in cohorts.items()}
        shared = [r for r in agreement_cases if r["reference"] == reference and r["status"] == "ok" and r["fully_parameterized"]]
        shared_common = set.intersection(*({r["pair"] for r in shared if r["method"] == m} for m in METHODS))
        result[reference]["common_all_five_reference_agreement_below_0_1"] = {
            method: aggregate([r for r in shared if r["method"] == method and r["pair"] in shared_common and tuple(r["pair"].split(" / ")) in agreement_pairs])
            for method in METHODS
        }
        result[reference]["coverage"] = {
            method: {"ok_reference_cases": sum(r["method"] == method for r in eligible),
                     "fully_parameterized_cases": sum(r["method"] == method for r in qualified),
                     "fallback_cases": sum(r["method"] == method and not r["fully_parameterized"] for r in eligible),
                     "failed_cases": sum(r["reference"] == reference and r["method"] == method and r["status"] != "ok" for r in case_rows)}
            for method in METHODS
        }
        families = {}
        for old, new in (("UNIFAC", "UNIFAC2"), ("UNIFDMD", "UNIFM2")):
            old_rows = {r["pair"]: r for r in qualified if r["method"] == old}
            new_rows = {r["pair"]: r for r in qualified if r["method"] == new}
            matching = sorted(old_rows.keys() & new_rows.keys())
            families[f"{old} versus {new}"] = {
                old: aggregate([old_rows[p] for p in matching]), new: aggregate([new_rows[p] for p in matching]),
                "new_model_pair_wins": sum(new_rows[p]["mae_ln_gamma"] < old_rows[p]["mae_ln_gamma"] and not math.isclose(new_rows[p]["mae_ln_gamma"], old_rows[p]["mae_ln_gamma"], rel_tol=1e-12, abs_tol=1e-12) for p in matching),
                "ties": sum(math.isclose(new_rows[p]["mae_ln_gamma"], old_rows[p]["mae_ln_gamma"], rel_tol=1e-12, abs_tol=1e-12) for p in matching),
                "extra_fully_parameterized_pairs": len(new_rows.keys() - old_rows.keys()),
            }
        result[reference]["family_pairwise"] = families
    return result


def render(summary, rows):
    lines = ["# UNIFAC agreement with runtime NRTL/UNIQUAC fits", "", summary["protocol"], "",
             "Errors are dimensionless differences in natural-log activity coefficients. Each fitted pair has equal weight. RMS means square root of the mean per-pair MSE across both components and sampled states. No percentages of ln(gamma) are used."]
    if "grand_total" in summary:
        lines += ["", "## Grand total across NRTL and UNIQUAC", "", summary["grand_total"]["weighting"], "",
                  "| Population | Method | Unique pairs | Fitted references | Mean pair MAE | Median pair MAE | Pair-weighted RMS |", "|---|---|---:|---:|---:|---:|---:|"]
        for cohort in ("common_all_five", "common_all_five_no_reference_clipping", "common_all_five_declared_no_clipping"):
            for method, m in summary["grand_total"][cohort].items():
                if m["unique_binary_pairs"]:
                    lines.append(f"| {cohort} | {method} | {m['unique_binary_pairs']} | {m['fitted_reference_cases']} | {m['mean_pair_mae_ln_gamma']:.4f} | {m['median_pair_mae_ln_gamma']:.4f} | {m['pair_weighted_rmse_ln_gamma']:.4f} |")
    for cohort, title in (("common_all_five_methods", "Common fully parameterized pairs: all temperatures"),
                          ("common_all_five_declared_temperature_ranges", "Common fully parameterized pairs: declared temperature ranges"),
                          ("common_all_five_declared_no_clipping", "Common fully parameterized pairs: declared ranges, no reference clipping"),
                          ("common_all_five_unspecified_temperature_ranges", "Common fully parameterized pairs: unspecified temperature ranges"),
                          ("common_all_five_reference_agreement_below_0_1", "Common pairs with bounded NRTL/UNIQUAC agreement MAE <= 0.1")):
        lines += ["", "## " + title, "", "| Reference | Method | Pairs | States | Mean pair MAE | Median pair MAE | Pair-weighted RMS | Mean bulk MAE | Mean dilute-solute MAE |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for reference, data in summary["scores"].items():
            for method, m in data[cohort].items():
                if m["reference_cases"]:
                    lines.append(f"| {reference} | {method} | {m['reference_cases']} | {m['states']} | {m['mean_pair_mae_ln_gamma']:.4f} | {m['median_pair_mae_ln_gamma']:.4f} | {m['pair_weighted_rmse_ln_gamma']:.4f} | {m['mean_bulk_pair_mae_ln_gamma']:.4f} | {m['mean_dilute_solute_pair_mae_ln_gamma']:.4f} |")
    lines += ["", "## Coverage", "", "| Reference | Method | Evaluated pairs | Fully parameterized | Missing-interaction fallback | Failed |", "|---|---|---:|---:|---:|---:|"]
    for reference, data in summary["scores"].items():
        for method, m in data["coverage"].items():
            lines.append(f"| {reference} | {method} | {m['ok_reference_cases']} | {m['fully_parameterized_cases']} | {m['fallback_cases']} | {m['failed_cases']} |")
    lines += ["", "## Largest fully parameterized pair discrepancies", "", "| Reference | Method | Pair | MAE ln(gamma) | RMS ln(gamma) |", "|---|---|---|---:|---:|"]
    for reference in ("NRTL", "UNIQUAC"):
        for method in METHODS:
            worst = sorted((r for r in rows if r["reference"] == reference and r["method"] == method and r["status"] == "ok" and r["fully_parameterized"]), key=lambda r: r["mae_ln_gamma"], reverse=True)[:3]
            for r in worst:
                lines.append(f"| {reference} | {method} | {r['name1']} + {r['name2']} | {r['mae_ln_gamma']:.4f} | {r['rmse_ln_gamma']:.4f} |")
    lines += ["", "## Limits", "", "These are comparisons with fitted-model surfaces, not raw VLE/LLE validation. Fitted-model uncertainty and ML training overlap are unknown. Dilute limits and homogeneous model extensions inside miscibility gaps are deliberate sensitivity probes; fitting-composition coverage is generally not declared. Unspecified-temperature fits are a separate diagnostic cohort. The primary tables exclude incomplete interaction matrices; their runtime zero fallbacks are recorded separately. UNIQUAC uses the runtime r/q and q-prime basis, with fallback warnings recorded.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summarize-dir", type=Path, help="Reaggregate saved evaluations without repeating model calculations")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.summarize_dir:
        original = args.summarize_dir.resolve()
        summary = json.loads((original / "summary.json").read_text())
        rows = list(csv.DictReader((original / "pairs.csv").open()))
        for row in rows:
            for field in ("fully_parameterized", "declared_temperature_range", "reference_clipped"):
                row[field] = row[field] == "True"
            for field in ("states", "ln_gamma_values"):
                if row.get(field):
                    row[field] = int(row[field])
            for field in ("mae_ln_gamma", "mse_ln_gamma", "rmse_ln_gamma", "bulk_mae_ln_gamma", "dilute_solute_mae_ln_gamma", "ideal_mae_ln_gamma"):
                if row.get(field):
                    row[field] = float(row[field])
        for filename, expected in summary["input_sha256"].items():
            if filename != str(Path(__file__).relative_to(ROOT)) and sha256(ROOT / filename) != expected:
                raise RuntimeError(f"Input/implementation changed since saved evaluations: {filename}")
        summary["grand_total"] = grand_totals(rows)
        summary["evaluations_dir"] = str(original)
        summary["analysis_script_sha256"] = sha256(Path(__file__))
        summary["saved_evaluation_files_sha256"] = {name: sha256(original/name) for name in ("summary.json", "pairs.csv", "points.csv")}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        (output / "report.md").write_text(render(summary, rows))
        print(json.dumps(summary["grand_total"], indent=2), flush=True)
        return
    started = time.perf_counter()
    relevant = [ROOT / "data" / name for name in (
        "nrtl_binary_interactions_cas.json", "uniquac_binary_interactions_cas.json", "uniquac_rq_cas.json",
        *METHOD_FILES.values(), "unifac2/UNIFAC2int.csv", "unifac2/UNIFM2intA.xlsx", "unifac2/UNIFM2intB.xlsx")]
    relevant += [ROOT / name for name in ("interaction_parameters.py", "unifac.py", "unifac_fragmenter.py", "compiled_activity.py", "compiled_unifac.py", "thermodynamics_models/nrtl_uniquac.py", "scripts/activity_fitting/benchmark_unifac_runtime_fits.py")]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in relevant}
    selected = {"NRTL": _nrtl_interactions(), "UNIQUAC": _uniquac_interactions()}
    classes = {"NRTL": NRTLThermodynamics, "UNIQUAC": UNIQUACThermodynamics}
    db = IdentityDatabase()
    models = {method: UNIFACModel(str(ROOT / "data" / filename)) for method, filename in METHOD_FILES.items()}
    reference_cache, predictor_cache = {}, {}
    reference_inventory, case_rows, audit, checks = [], [], [], []

    @lru_cache(maxsize=None)
    def groups_for(method, cas):
        props = db.get(cas)
        return models[method]._resolve_groups(get_unifac_groups(props.name, smiles=props.smiles, variant=method, expected_mw=props.MW))

    def predictor(method, pair):
        key = (method, pair)
        if key not in predictor_cache:
            try:
                groups = {cas: groups_for(method, cas) for cas in pair}
                main_groups = {models[method].subgroups[g].main_group for value in groups.values() for g in value}
                missing = [(i, j) for i in sorted(main_groups) for j in sorted(main_groups) if i != j and (i, j) not in models[method].interactions and (i, j) not in models[method].interaction_coefficients]
                backend = CompiledUNIFACBackend.from_model(models[method], list(pair), groups)
                predictor_cache[key] = (backend, groups, missing, None)
            except (ValueError, KeyError, RuntimeError, ZeroDivisionError) as error:
                predictor_cache[key] = (None, None, None, str(error))
        return predictor_cache[key]

    def predict_values(method, pair, temperatures, compositions):
        backend, groups, _missing, failure = predictor(method, pair)
        if failure:
            raise ValueError(failure)
        values = []
        for temperature, x1 in zip(temperatures, compositions):
            gamma = backend.activity_coefficients([x1, 1-x1], temperature) if backend else models[method].activity_coefficients([groups[c] for c in pair], [x1, 1-x1], temperature)
            gamma = np.asarray(gamma)
            if np.any(gamma <= 0) or not np.all(np.isfinite(gamma)):
                raise ValueError("UNIFAC gamma is non-positive/non-finite")
            values.append(np.log(gamma))
        return np.array(values)

    point_fields = ["reference", "method", "cas1", "cas2", "T_K", "x1", "reference_ln_gamma1", "reference_ln_gamma2", "predicted_ln_gamma1", "predicted_ln_gamma2", "error_ln_gamma1", "error_ln_gamma2", "declared_temperature_range", "fully_parameterized", "reference_clipped"]
    with (output / "points.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=point_fields)
        writer.writeheader()
        for reference, records in selected.items():
            for index, (pair, record) in enumerate(sorted(records.items())):
                pair_label = " / ".join(pair)
                try:
                    props = [db.get(c) for c in pair]
                    thermo = classes[reference](list(pair), db=db)
                    t, x, ref, bounds, temperatures, compositions = select_points(thermo, pair, record)
                    reference_cache[reference, pair] = thermo
                    clipped = bool(np.any(np.isclose(ref, 50, atol=1e-10, rtol=0)) or np.any(np.isclose(ref, math.log(1e-12), atol=1e-10, rtol=0)))
                    inventory = {"reference": reference, "pair": pair_label, "name1": props[0].name, "name2": props[1].name,
                                 "declared_temperature_range": bounds is not None, "temperatures_K": json.dumps(temperatures.tolist()),
                                 "compositions_x1": json.dumps(compositions.tolist()), "states": len(t), "reference_clipped": clipped,
                                 "runtime_record": json.dumps(record, sort_keys=True), "warnings": "; ".join(thermo.warnings)}
                    reference_inventory.append(inventory)
                except (ValueError, KeyError, RuntimeError, OverflowError, ZeroDivisionError) as error:
                    audit.append({"reference": reference, "pair": pair_label, "scope": "reference", "reason": str(error)})
                    continue
                for method in METHODS:
                    backend, groups, missing, failure = predictor(method, pair)
                    base = {"reference": reference, "method": method, "pair": pair_label, "name1": props[0].name, "name2": props[1].name,
                            "declared_temperature_range": bounds is not None, "reference_clipped": clipped,
                            "fully_parameterized": not bool(missing) if missing is not None else False,
                            "missing_interactions": json.dumps(missing), "status": failure or "ok"}
                    if failure:
                        case_rows.append(base)
                        audit.append({"reference": reference, "pair": pair_label, "method": method, "scope": "prediction", "reason": failure})
                        continue
                    try:
                        pred = predict_values(method, pair, t, x)
                        error = pred - ref
                        if index % 40 == 0:
                            k = len(t)//2
                            native = np.log(models[method].activity_coefficients([groups[c] for c in pair], [x[k], 1-x[k]], t[k]))
                            np.testing.assert_allclose(pred[k], native, atol=2e-10, rtol=2e-10)
                            checks.append({"reference": reference, "method": method, "pair": pair_label, "max_native_ln_gamma_difference": float(np.max(np.abs(pred[k]-native)))})
                        case_rows.append({**base, **case_metrics(error, x, ref)})
                        for k in range(len(t)):
                            writer.writerow({"reference": reference, "method": method, "cas1": pair[0], "cas2": pair[1], "T_K": t[k], "x1": x[k],
                                             "reference_ln_gamma1": ref[k, 0], "reference_ln_gamma2": ref[k, 1], "predicted_ln_gamma1": pred[k, 0], "predicted_ln_gamma2": pred[k, 1],
                                             "error_ln_gamma1": error[k, 0], "error_ln_gamma2": error[k, 1], "declared_temperature_range": bounds is not None,
                                             "fully_parameterized": not bool(missing), "reference_clipped": clipped})
                    except (ValueError, KeyError, RuntimeError, OverflowError, ZeroDivisionError) as error:
                        case_rows.append({**base, "status": str(error)})
                        audit.append({"reference": reference, "pair": pair_label, "method": method, "scope": "prediction", "reason": str(error)})
                if (index+1) % 50 == 0:
                    handle.flush()
                    print(f"{reference}: {index+1}/{len(records)} runtime pairs", flush=True)

    agreement, agreement_cases, agreement_points = [], [], []
    for pair in sorted(selected["NRTL"].keys() & selected["UNIQUAC"].keys()):
        if ("NRTL", pair) not in reference_cache or ("UNIQUAC", pair) not in reference_cache:
            continue
        a = activity_interaction_temperature_range(selected["NRTL"][pair])
        b = activity_interaction_temperature_range(selected["UNIQUAC"][pair])
        if a and b:
            low, high = max(a[0], b[0]), min(a[1], b[1])
            if low > high:
                continue
            temperatures = np.unique(np.linspace(low, high, 3))
        elif a or b:
            temperatures = np.unique(np.linspace(*(a or b), 3))
        else:
            temperatures = FALLBACK_T
        t, x = np.repeat(temperatures, len(BASE_X)), np.tile(BASE_X, len(temperatures))
        try:
            av = reference_values(reference_cache["NRTL", pair], pair, t, x)
            bv = reference_values(reference_cache["UNIQUAC", pair], pair, t, x)
            agreement.append({"pair": " / ".join(pair), "name1": db.get(pair[0]).name, "name2": db.get(pair[1]).name,
                              "both_declared_ranges": bool(a and b), "temperatures_K": json.dumps(temperatures.tolist()),
                              **case_metrics(av-bv, x, bv)})
            if a and b:
                for method in METHODS:
                    _backend, _groups, missing, failure = predictor(method, pair)
                    common = {"method": method, "pair": " / ".join(pair), "name1": db.get(pair[0]).name, "name2": db.get(pair[1]).name,
                              "declared_temperature_range": True, "fully_parameterized": not bool(missing) if missing is not None else False}
                    if failure:
                        for reference in ("NRTL", "UNIQUAC"):
                            agreement_cases.append({**common, "reference": reference, "status": failure})
                        continue
                    try:
                        pred = predict_values(method, pair, t, x)
                        for reference, ref in (("NRTL", av), ("UNIQUAC", bv)):
                            error = pred-ref
                            agreement_cases.append({**common, "reference": reference, "status": "ok", **case_metrics(error, x, ref)})
                            for k in range(len(t)):
                                agreement_points.append({"reference": reference, "method": method, "cas1": pair[0], "cas2": pair[1], "T_K": t[k], "x1": x[k],
                                                         "reference_ln_gamma1": ref[k,0], "reference_ln_gamma2": ref[k,1], "predicted_ln_gamma1": pred[k,0], "predicted_ln_gamma2": pred[k,1],
                                                         "error_ln_gamma1": error[k,0], "error_ln_gamma2": error[k,1], "fully_parameterized": common["fully_parameterized"]})
                    except (ValueError, KeyError, RuntimeError, OverflowError, ZeroDivisionError) as error:
                        for reference in ("NRTL", "UNIQUAC"):
                            agreement_cases.append({**common, "reference": reference, "status": str(error)})
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            audit.append({"pair": " / ".join(pair), "scope": "reference_agreement", "reason": str(error)})
    scores = summaries(case_rows, reference_inventory, agreement, agreement_cases)
    shared_agreeing_pairs = set.intersection(*(
        {r["pair"] for r in agreement_cases if r["reference"] == reference and r["method"] == method and r["status"] == "ok" and r["fully_parameterized"]}
        for reference in ("NRTL", "UNIQUAC") for method in METHODS
    ))
    shared_agreeing_spread = [r for r in agreement if r["pair"] in shared_agreeing_pairs and r["both_declared_ranges"] and r["mae_ln_gamma"] <= .1]
    for filename, expected in hashes.items():
        if sha256(ROOT / filename) != expected:
            raise RuntimeError(f"Input/implementation changed during benchmark: {filename}")
    summary = {
        "protocol": "Runtime-selected enabled NRTL/UNIQUAC references, using their normal runtime activity-coefficient and structural-basis paths. Declared Tmin/midpoint/Tmax, otherwise 298.15/323.15/348.15 K. Fifteen symmetric x1 points from 0.0001 to 0.9999, plus reference-only |gE/(RT)| maxima on a 49-point interior grid at each temperature. Equal weight per fitted binary pair; both components scored. Missing UNIFAC interactions flagged; primary comparisons use fully parameterized common pairs. No raw-data fitting, vapor pressure, vapor fugacity, or phase-equilibrium solver enters the score.",
        "input_runtime_pairs": {r: len(v) for r, v in selected.items()}, "evaluated_references": dict(Counter(r["reference"] for r in reference_inventory)),
        "scores": scores, "grand_total": grand_totals(case_rows),
        "reference_agreement": {"all": aggregate(agreement), "both_declared_ranges": aggregate([r for r in agreement if r["both_declared_ranges"]]),
                                                  "agreeing_both_declared_ranges": aggregate([r for r in agreement if r["both_declared_ranges"] and r["mae_ln_gamma"] <= .1]),
                                                  "agreeing_shared_all_five": aggregate(shared_agreeing_spread)},
        "reference_clipped_cases": [r["reference"] + ": " + r["pair"] for r in reference_inventory if r["reference_clipped"]],
        "rq_fallback_cases": [r["pair"] for r in reference_inventory if r["reference"] == "UNIQUAC" and "estimated from UNIFAC" in r["warnings"]],
        "versions": {name: version(name) for name in ("numpy", "scipy", "rdkit", "chemicals", "numba")},
        "input_sha256": hashes, "elapsed_seconds_excluding_imports_and_final_writes": time.perf_counter()-started,
        "native_checks": len(checks), "max_native_ln_gamma_difference": max((r["max_native_ln_gamma_difference"] for r in checks), default=0),
        "argv": sys.argv,
    }
    write_csv(output / "pairs.csv", case_rows)
    write_csv(output / "references.csv", reference_inventory)
    write_csv(output / "reference_agreement.csv", agreement)
    write_csv(output / "agreement_pairs.csv", agreement_cases)
    write_csv(output / "agreement_points.csv", agreement_points)
    write_csv(output / "excluded.csv", audit)
    write_csv(output / "native_checks.csv", checks)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (output / "report.md").write_text(render(summary, case_rows))
    print(json.dumps({"evaluated_references": summary["evaluated_references"], "native_checks": len(checks), "elapsed_seconds": summary["elapsed_seconds_excluding_imports_and_final_writes"], "output_dir": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
