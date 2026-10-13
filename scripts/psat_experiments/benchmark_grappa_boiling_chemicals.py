"""Expand the Perry boiling-point comparison with chemicals reference tables.

One reference per exact CAS, priority HEOS, CRC_ORG, CRC_INORG,
COMMON_CHEMISTRY, WEBBOOK. Exclude estimated JOBACK, mixed experimental/
predicted YAWS, and secondary WIKIDATA targets. Keep source tiers separate.
Reuse the completed Perry predictions where identity matches; never refit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from chemicals import phase_change
from chemicals.identifiers import int_to_CAS, search_chemical
import numpy as np
from rdkit import Chem
import torch

from benchmark_grappa_boiling import ROOT, metrics, paired, write_csv

sys.path.insert(0, str(ROOT))
from nannoolal_method import estimate

SOURCES = ("HEOS", "CRC_ORG", "CRC_INORG", "COMMON_CHEMISTRY", "WEBBOOK")


def predict_nannoolal(task):
    cas, tb, source, source_tm, old = task
    row = {"cas": cas, "reference_Tb_K": tb, "reference_source": source,
           "Tm_K": source_tm, "reference_in_250_600_K": 250 <= tb <= 600,
           "Nannoolal_status": "unavailable", "GRAPPA_status": "unavailable"}
    try:
        identity = search_chemical(cas)
        if identity.CASs != cas:
            raise ValueError("offline CAS identity mismatch")
        row["name"] = identity.common_name
        # Honor original benchmark's resolved graph for overlapping compounds.
        smiles = old.get("smiles") or identity.smiles
        if not smiles or Chem.MolFromSmiles(smiles) is None:
            raise ValueError("no valid offline molecular structure")
        row["smiles"] = smiles
        if old.get("smiles") == smiles:
            row["Nannoolal_status"] = old["Nannoolal_status"]
            if old["Nannoolal_status"] == "ok":
                row["Nannoolal_Tb_K"] = float(old["Nannoolal_Tb_K"])
                row["Nannoolal_warnings"] = old.get("Nannoolal_warnings", "")
        else:
            result = estimate(smiles)
            if not result.tb_K or not math.isfinite(result.tb_K):
                raise ValueError("no finite Nannoolal estimate")
            row.update(Nannoolal_Tb_K=result.tb_K, Nannoolal_status="ok",
                       Nannoolal_warnings="; ".join(result.warnings))
    except Exception as error:
        row["Nannoolal_status"] = str(error)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perry-results", type=Path, required=True)
    parser.add_argument("--grappa-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5, choices=range(1, 6))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    original = list(csv.DictReader((args.perry_results / "compounds.csv").open()))
    original_by_cas = {r["cas"]: r for r in original}
    phase_change.Tb("64-17-5", method="CRC_ORG")  # Load explicit source tables.
    selected, inventory, audit, conflicts = {}, {}, [], []
    for source in SOURCES:
        frame = phase_change.Tb_sources[source]
        inventory[source] = int(frame["Tb"].notna().sum())
        for index, record in frame.iterrows():
            tb = float(record["Tb"])
            if not math.isfinite(tb) or tb <= 0:
                continue
            cas = index if isinstance(index, str) else int_to_CAS(int(index))
            if cas in selected:
                previous = selected[cas]
                conflicts.append({"cas": cas, "preferred_source": previous[1], "preferred_Tb_K": previous[0],
                                  "other_source": source, "other_Tb_K": tb, "difference_K": tb-previous[0]})
                continue
            # Same-source melting point only; do not invent phase bounds.
            tm = float(record.get("Tm", math.nan))
            tm = tm if math.isfinite(tm) and tm > 0 else None
            if cas == "302-17-0" or (tm and tm >= tb-2):
                audit.append({"cas": cas, "source": source, "reason": "excluded decomposer or same-source Tm >= Tb-2 K"})
                continue
            selected[cas] = (tb, source, tm)
    tasks = [(cas, tb, source, tm, original_by_cas.get(cas, {}))
             for cas, (tb, source, tm) in sorted(selected.items())]
    print(f"Selected {len(tasks)} distinct chemicals references; {len(set(selected)-set(original_by_cas))} additional to Perry selection", flush=True)
    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(predict_nannoolal, tasks, chunksize=64))
    write_csv(out / "reference_and_nannoolal_checkpoint.csv", rows)
    print(f"Offline identity and Nannoolal pass complete: {sum(r['Nannoolal_status']=='ok' for r in rows)} estimates", flush=True)
    grappa_root = args.grappa_root.resolve()
    sys.path.insert(0, str(grappa_root))
    from src.utils import GRAPPAantoine, GRAPPAdirect, mol_to_pyg
    parameters = {r["smiles"]: [float(r[k]) for k in ("A", "B", "C")]
                  for r in original if r["GRAPPA_status"] == "ok"}
    needed = set()
    for r in rows:
        if "smiles" not in r:
            r["GRAPPA_status"] = "no resolved structure"
            continue
        try:
            mol_to_pyg(Chem.MolFromSmiles(r["smiles"]), 293.15)
            needed.add(r["smiles"])
            r["GRAPPA_status"] = "input accepted"
        except Exception as error:
            r["GRAPPA_status"] = str(error)
    missing = sorted(needed-set(parameters))
    model_start = time.perf_counter()
    previous = Path.cwd()
    try:
        os.chdir(grappa_root)
        with torch.inference_mode():
            model = GRAPPAantoine()
            if missing:
                parameters.update(zip(missing, model(missing)))
        model_seconds = time.perf_counter()-model_start
        for r in rows:
            if r["GRAPPA_status"] != "input accepted":
                continue
            a, b, c = parameters[r["smiles"]]
            try:
                denominator = a-math.log(101.325)
                tb = b/denominator-c
                if not all(math.isfinite(v) for v in (a, b, c, tb)) or b <= 0 or denominator <= 0 or tb <= 0:
                    raise ValueError("invalid monotone Antoine inversion")
                r.update(GRAPPA_status="ok", GRAPPA_Tb_K=tb, A=a, B=b, C=c,
                         prediction_in_250_600_K=250 <= tb <= 600)
            except (ValueError, ZeroDivisionError) as error:
                r["GRAPPA_status"] = str(error)
        good = [r for r in rows if r["GRAPPA_status"] == "ok"]
        samples = [good[i] for i in np.linspace(0, len(good)-1, 24, dtype=int)]
        with torch.inference_mode():
            pressures = GRAPPAdirect()([r["smiles"] for r in samples], [r["GRAPPA_Tb_K"] for r in samples])
        np.testing.assert_allclose(pressures, 101.325, rtol=2e-5)
    finally:
        os.chdir(previous)
    matched = [r for r in rows if r["GRAPPA_status"] == r["Nannoolal_status"] == "ok"]
    strict = [r for r in matched if r["reference_source"] in ("HEOS", "CRC_ORG", "CRC_INORG")]
    extra = [r for r in matched if r["cas"] not in original_by_cas]
    # Combined benchmark preserves the original Perry references and replaces
    # nothing; each additional CAS contributes one chemicals reference.
    original_matched = [{"cas": r["cas"], "reference_Tb_K": float(r["reference_Tb_K"]),
                         "GRAPPA_Tb_K": float(r["GRAPPA_Tb_K"]), "Nannoolal_Tb_K": float(r["Nannoolal_Tb_K"])}
                        for r in original if r["GRAPPA_status"] == r["Nannoolal_status"] == "ok"]
    populations = {"chemicals_all_matched": matched, "HEOS_CRC_matched": strict,
                   "chemicals_additional_to_Perry": extra,
                   "chemicals_reference_250_600_K": [r for r in matched if r["reference_in_250_600_K"]],
                   "combined_Perry_then_additional_chemicals": original_matched+extra}
    print(f"Inference complete: {len(good)} GRAPPA, {len(matched)} matched. Aggregating pairwise uncertainty.", flush=True)
    summary = {"protocol": __doc__, "source_priority": SOURCES, "inventory": inventory,
               "selected_reference_count": len(rows), "additional_reference_count": len(set(selected)-set(original_by_cas)),
               "cohorts": {k: paired(v) for k, v in populations.items()},
               "by_source": {source: {m: metrics([r for r in matched if r["reference_source"] == source], m)
                                      for m in ("GRAPPA", "Nannoolal")} for source in SOURCES},
               "standalone": {m: metrics([r for r in rows if r[f"{m}_status"] == "ok"], m) for m in ("GRAPPA", "Nannoolal")},
               "coverage_status": {m: dict(Counter(r[f"{m}_status"] for r in rows)) for m in ("GRAPPA", "Nannoolal")},
               "model_parameter_prediction_seconds": model_seconds,
               "new_structures_predicted": len(missing), "verification": "24 direct-pressure inversion checks passed",
               "elapsed_seconds_excluding_imports": time.perf_counter()-started,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "perry_points_sha256": hashlib.sha256((args.perry_results / "compounds.csv").read_bytes()).hexdigest(),
               "limitations": "Common Chemistry and WebBook flattened Tb fields do not retain full conditions/uncertainties; treat as supplementary compilations, not uniformly verified measurements. Source priority is deterministic, not selected by prediction error. Temperature extrapolation and training overlap unknown. Compound-level bootstrap ignores chemical-family dependence. Source tables available in installed chemicals version recorded by original benchmark."}
    write_csv(out / "compounds.csv", rows)
    write_csv(out / "excluded.csv", audit)
    write_csv(out / "reference_conflicts.csv", conflicts)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"counts": [len(rows), len(good), len(matched)], "cohorts": summary["cohorts"],
                      "by_source": summary["by_source"], "elapsed": summary["elapsed_seconds_excluding_imports"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
