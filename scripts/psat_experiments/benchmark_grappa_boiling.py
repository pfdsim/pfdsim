"""GRAPPA vs the Tb-only pass of final_nannoolal_benchmark.py.

Reproduce its Perry 2-10 selection, offline identity resolution, explicit CAS
exclusion, validated Tb overrides, and Tm >= Tb - 2 K rejection. Recompute the
current unanchored Nannoolal estimates rather than using its stale report.
GRAPPA predicts Antoine parameters once per structure; invert at 101.325 kPa.
Never anchor either method to a reference Tb. Output directory must be new.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from rdkit import Chem
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from chemical_properties import ChemicalDatabase
from chemicals.identifiers import search_chemical
from nannoolal_method import estimate

BASELINE = ROOT / "scripts/nannoolal/final_nannoolal_benchmark.py"
TABLE = ROOT / "data/perry_table_2_10_vapor_pressure.json"
SEED = 20261012


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["cas", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows, method):
    errors = np.array([r[f"{method}_Tb_K"] - r["reference_Tb_K"] for r in rows])
    refs = np.array([r["reference_Tb_K"] for r in rows])
    if not len(errors):
        return {"compounds": 0}
    absolute = np.abs(errors)
    return {"compounds": len(errors), "MAE_K": float(absolute.mean()),
            "median_absolute_error_K": float(np.median(absolute)),
            "RMSE_K": float(np.sqrt(np.mean(errors**2))), "bias_K": float(errors.mean()),
            "p90_absolute_error_K": float(np.quantile(absolute, .9)),
            "p95_absolute_error_K": float(np.quantile(absolute, .95)),
            "max_absolute_error_K": float(absolute.max()),
            "MAPE_percent": float(np.mean(100 * absolute / refs)),
            "fraction_within_5_K": float(np.mean(absolute <= 5)),
            "fraction_within_10_K": float(np.mean(absolute <= 10))}


def paired(rows):
    result = {m: metrics(rows, m) for m in ("GRAPPA", "Nannoolal")}
    if not rows:
        return result
    delta = np.array([abs(r["GRAPPA_Tb_K"]-r["reference_Tb_K"])
                      - abs(r["Nannoolal_Tb_K"]-r["reference_Tb_K"]) for r in rows])
    rng = np.random.default_rng(SEED)
    boot = np.concatenate([delta[rng.integers(len(delta), size=(1000, len(delta)))].mean(axis=1)
                           for _ in range(20)])
    ties = np.isclose(delta, 0, rtol=0, atol=1e-10)
    result.update(GRAPPA_wins=int(np.sum((delta < 0) & ~ties)),
                  Nannoolal_wins=int(np.sum((delta > 0) & ~ties)), ties=int(ties.sum()),
                  MAE_difference_GRAPPA_minus_Nannoolal_K=float(delta.mean()),
                  paired_bootstrap_95_CI_MAE_difference_K=np.quantile(boot, [.025, .975]).tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grappa-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    grappa_root = args.grappa_root.resolve()
    started = time.perf_counter()
    # Read only the original benchmark's literal policy constants; importing
    # that file would execute unrelated critical-property benchmarks and write
    # over its stored output.
    constants = {}
    for node in ast.parse(BASELINE.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("EXCLUDE_CAS", "PFDSim_TB"):
                    constants[target.id] = ast.literal_eval(node.value)
    overrides, audit = {}, []
    for name, tb in constants["PFDSim_TB"].items():
        try:
            overrides[search_chemical(name).CASs] = tb
        except Exception as error:
            audit.append({"name": name, "stage": "override resolution", "reason": str(error)})
    payload = json.loads(TABLE.read_text())
    collisions = payload.get("metadata", {}).get("duplicate_cas_collisions", {})
    database = ChemicalDatabase(enable_online=False)
    rows = []
    for cas, record in payload["chemicals"].items():
        row = {"cas": cas, "name": record.get("name", cas)}
        if cas in constants["EXCLUDE_CAS"] or not record.get("Tb_K"):
            audit.append({**row, "stage": "original selection", "reason": "excluded CAS or missing Tb"})
            continue
        tb = overrides.get(cas) or record["Tb_K"]
        tm = record.get("Tm_K")
        if tm and tm >= tb - 2:
            audit.append({**row, "stage": "original selection", "reason": "Tm >= reference Tb - 2 K"})
            continue
        row.update(reference_Tb_K=tb, reference_source="PFDSim override" if cas in overrides else "Perry 2-10",
                   original_Perry_Tb_K=record["Tb_K"], Tm_K=tm,
                   reference_in_250_600_K=250 <= tb <= 600,
                   identity_collision=cas in collisions, Nannoolal_status="unavailable", GRAPPA_status="unavailable")
        try:
            identity = database.resolve_smiles_info(cas, fetch_online=False)
            smiles = identity.smiles if identity else None
            if not smiles:
                raise ValueError("missing offline SMILES")
            row["smiles"] = smiles
        except Exception as error:
            row["identity_status"] = str(error)
            rows.append(row)
            continue
        try:
            prediction = estimate(smiles)  # No supplied Tb.
            if not prediction.tb_K or not math.isfinite(prediction.tb_K):
                raise ValueError("no finite estimated Tb")
            row.update(Nannoolal_Tb_K=prediction.tb_K, Nannoolal_status="ok",
                       Nannoolal_warnings="; ".join(prediction.warnings))
        except Exception as error:
            row["Nannoolal_status"] = str(error)
        rows.append(row)
    print(f"Selected {len(rows)} references; current Nannoolal estimates: "
          f"{sum(r['Nannoolal_status']=='ok' for r in rows)}", flush=True)
    sys.path.insert(0, str(grappa_root))
    from src.utils import GRAPPAantoine, GRAPPAdirect, GRAPPAnormalbp, mol_to_pyg
    valid_smiles = set()
    for row in rows:
        if "smiles" not in row:
            continue
        try:
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                raise ValueError("invalid SMILES")
            mol_to_pyg(mol, 293.15)  # Authoritative model chemistry contract.
            valid_smiles.add(row["smiles"])
            row["GRAPPA_status"] = "input accepted"
        except Exception as error:
            row["GRAPPA_status"] = str(error)
    smiles = sorted(valid_smiles)
    torch.set_num_threads(1)
    previous = Path.cwd()
    model_start = time.perf_counter()
    try:
        os.chdir(grappa_root)  # Upstream loader uses a relative checkpoint path.
        with torch.inference_mode():
            predictor = GRAPPAantoine()
            parameters = dict(zip(smiles, predictor(smiles)))
        inference_seconds = time.perf_counter() - model_start
        for row in rows:
            if row["GRAPPA_status"] != "input accepted":
                continue
            a, b, c = parameters[row["smiles"]]
            try:
                denominator = a-math.log(101.325)
                tb = b/denominator-c
                if not all(math.isfinite(v) for v in (a, b, c, tb)) or b <= 0 or denominator <= 0 or tb <= 0:
                    raise ValueError("invalid monotone Antoine boiling-point inversion")
                row.update(GRAPPA_status="ok", GRAPPA_Tb_K=tb, A=a, B=b, C=c,
                           prediction_in_250_600_K=250 <= tb <= 600,
                           GRAPPA_error_K=tb-row["reference_Tb_K"])
                if row["Nannoolal_status"] == "ok":
                    row["Nannoolal_error_K"] = row["Nannoolal_Tb_K"]-row["reference_Tb_K"]
            except (ValueError, ZeroDivisionError) as error:
                row["GRAPPA_status"] = str(error)
        good = [r for r in rows if r["GRAPPA_status"] == "ok"]
        sample = [good[i] for i in np.linspace(0, len(good)-1, min(18, len(good)), dtype=int)]
        with torch.inference_mode():
            # Upstream normalbp subtracts an entire pressure list inside each
            # 32-item batch. Use <=32 inputs for its independent interface check;
            # our full inference inverts each returned A/B/C row separately.
            official = GRAPPAnormalbp()([r["smiles"] for r in sample])
            direct = GRAPPAdirect()([r["smiles"] for r in sample], [r["GRAPPA_Tb_K"] for r in sample])
        np.testing.assert_allclose(official, [r["GRAPPA_Tb_K"] for r in sample], rtol=2e-6)
        np.testing.assert_allclose(direct, 101.325, rtol=2e-5)
    finally:
        os.chdir(previous)
    matched = [r for r in rows if r["GRAPPA_status"] == r["Nannoolal_status"] == "ok"]
    cohorts = {"all_matched": matched,
               "reference_250_600_K": [r for r in matched if r["reference_in_250_600_K"]],
               "reference_outside_250_600_K": [r for r in matched if not r["reference_in_250_600_K"]],
               "excluding_identity_collisions": [r for r in matched if not r["identity_collision"]]}
    summary = {"protocol": __doc__, "cohorts": {k: paired(v) for k, v in cohorts.items()},
               "standalone": {m: metrics([r for r in rows if r[f"{m}_status"] == "ok"], m)
                              for m in ("GRAPPA", "Nannoolal")},
               "input_table_compounds": len(payload["chemicals"]), "selected_references": len(rows),
               "coverage_status": {m: dict(Counter(r[f"{m}_status"] for r in rows)) for m in ("GRAPPA", "Nannoolal")},
               "resolved_overrides": overrides, "seed": SEED, "bootstrap_replicates": 20000,
               "model_load_and_parameter_prediction_seconds": inference_seconds,
               "elapsed_seconds_excluding_imports": time.perf_counter()-started,
               "verification": f"{len(sample)} official boiling-point and direct pressure checks passed",
               "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (BASELINE, TABLE, ROOT / "nannoolal_method.py", ROOT / "chemical_properties.py", Path(__file__))},
               "checkpoint_sha256": hashlib.sha256((grappa_root / "models/GRAPPA_state_dict.pt").read_bytes()).hexdigest(),
               "versions": {p: version(p) for p in ("torch", "torch-geometric", "rdkit", "numpy", "chemicals")},
               "limitations": "Same selection as original benchmark; no extra chemistry, collision, or temperature filtering in headline. Predictions outside 250-600 K are extrapolations. Training overlap unknown. Current Nannoolal includes local calibration. Bootstrap treats compounds as independent, despite shared chemistry; exploratory interval only."}
    write_csv(output / "compounds.csv", rows)
    write_csv(output / "excluded.csv", audit)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"cohorts": summary["cohorts"], "standalone": summary["standalone"],
                      "timing": summary["elapsed_seconds_excluding_imports"], "verification": summary["verification"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
