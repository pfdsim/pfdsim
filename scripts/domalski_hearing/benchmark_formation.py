"""Benchmark Domalski--Hearing gas Hf and S against bundled local data.

The reference set is the strict ``source == "local"`` set used by pfdsim's
formation-property resolver:

* Perry Table 2-95 records in ``data/perry_properties.json``; and
* legacy curated ``data/chemicals.json`` records whose Hf and S provenance
  defaults explicitly to ``local``.

Detailed per-component results are written to JSON; stdout and the text report
contain only aggregate statistics, rejection counts, and the largest errors.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import RDLogger


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import domalski_hearing_method as dh  # noqa: E402
from chemicals.identifiers import search_chemical  # noqa: E402


PERRY_PATH = ROOT / "data" / "perry_properties.json"
CHEMICALS_PATH = ROOT / "data" / "chemicals.json"
JSON_OUTPUT = ROOT / "outputs" / "domalski_hearing_local_formation_benchmark.json"
TEXT_OUTPUT = ROOT / "outputs" / "domalski_hearing_local_formation_benchmark.txt"


def _strict_source(entry: dict[str, Any], key: str) -> str:
    property_sources = entry.get("property_sources") or {}
    direct = property_sources.get(key) or {}
    dataset = property_sources.get("dataset") or {}
    return direct.get("source") or dataset.get("source") or "local"


def load_reference_records() -> list[dict[str, Any]]:
    perry = json.loads(PERRY_PATH.read_text())["chemicals"]
    records: dict[str, dict[str, Any]] = {}
    for cas, entry in perry.items():
        formation = entry.get("formation_properties") or {}
        hf = formation.get("Hf_ideal_gas_J_per_kmol")
        entropy = formation.get("S_ideal_gas_J_per_kmol_K")
        if hf is None or entropy is None:
            continue
        records[cas] = {
            "identity": cas,
            "cas": cas,
            "name": entry.get("name") or cas,
            "formula": entry.get("formula") or "",
            "smiles": entry.get("smiles"),
            "reference_source": "Perry 9th Table 2-95",
            "reference_Hf_kJ_mol": float(hf) / 1.0e6,
            "reference_S_J_mol_K": float(entropy) / 1000.0,
        }

    bundled = json.loads(CHEMICALS_PATH.read_text())["chemicals"]
    for symbol, entry in bundled.items():
        if entry.get("Hf") is None or entry.get("S") is None:
            continue
        if _strict_source(entry, "Hf") != "local" or _strict_source(entry, "S") != "local":
            continue
        cas = str(entry.get("CAS") or "").strip()
        identity = cas or f"chemicals.json:{symbol}"
        if identity in records:
            continue
        records[identity] = {
            "identity": identity,
            "cas": cas,
            "name": entry.get("name") or symbol,
            "formula": entry.get("formula") or symbol,
            "smiles": entry.get("smiles"),
            "reference_source": "chemicals.json local",
            "reference_Hf_kJ_mol": float(entry["Hf"]),
            "reference_S_J_mol_K": float(entry["S"]),
        }

    return list(records.values())


def resolve_smiles(record: dict[str, Any]) -> tuple[str | None, str]:
    if record.get("smiles"):
        return str(record["smiles"]), "dataset"
    candidates = [record.get("cas"), record.get("name")]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            metadata = search_chemical(str(candidate))
        except Exception:
            continue
        smiles = str(getattr(metadata, "smiles", "") or "").strip()
        if smiles:
            return smiles, "chemicals.identifiers"
    return None, "missing"


def rejection_category(message: str) -> str:
    lower = message.lower()
    categories = (
        ("outside published element domain", "outside the published"),
        ("multiple molecular fragments", "one molecular fragment"),
        ("charged structure", "charge"),
        ("radical structure", "radical"),
        ("unassigned/unsupported group", "unclaimed"),
        ("unassigned/unsupported group", "does not publish"),
        ("unassigned/unsupported group", "no published"),
        ("unassigned/unsupported group", "not covered by table"),
        ("unassigned/unsupported group", "without one"),
        ("unassigned/unsupported group", "must have exactly"),
    )
    for category, needle in categories:
        if needle in lower:
            return category
    return "other fragmentation error"


def percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = fraction * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def metric_summary(
    rows: list[dict[str, Any]],
    error_key: str,
    reference_key: str,
) -> dict[str, Any]:
    errors = [float(row[error_key]) for row in rows if row.get(error_key) is not None]
    references = [
        float(row[reference_key]) for row in rows if row.get(error_key) is not None
    ]
    absolute = sorted(abs(value) for value in errors)
    if not errors:
        return {"n": 0}
    mean_reference = statistics.fmean(references)
    ss_residual = sum(value * value for value in errors)
    ss_total = sum((value - mean_reference) ** 2 for value in references)
    return {
        "n": len(errors),
        "bias": statistics.fmean(errors),
        "mae": statistics.fmean(absolute),
        "median_ae": statistics.median(absolute),
        "p90_ae": percentile(absolute, 0.90),
        "rmse": math.sqrt(statistics.fmean(value * value for value in errors)),
        "max_ae": absolute[-1],
        "r_squared": None if ss_total == 0.0 else 1.0 - ss_residual / ss_total,
    }


def benchmark() -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    records = load_reference_records()
    source_counts = Counter(record["reference_source"] for record in records)
    rejection_counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []

    for record in records:
        detail = dict(record)
        smiles, smiles_source = resolve_smiles(record)
        detail["smiles"] = smiles
        detail["smiles_source"] = smiles_source
        detail["status"] = "pending"
        detail["predicted_Hf_kJ_mol"] = None
        detail["predicted_S_J_mol_K"] = None
        detail["Hf_error"] = None
        detail["S_error"] = None

        if not smiles:
            detail["status"] = "missing structure"
            rejection_counts["missing structure"] += 1
            details.append(detail)
            continue

        try:
            estimate = dh.estimate(smiles)
        except dh.DomalskiHearingError as exc:
            category = rejection_category(str(exc))
            detail["status"] = category
            detail["rejection_message"] = str(exc)
            rejection_counts[category] += 1
            details.append(detail)
            continue

        # A fully claimed structure can currently reach the estimator with no
        # published groups (for example, elemental dihalogens or cyanogen).
        # The resulting zeros plus a symmetry term are not a group-additivity
        # prediction and must not enter the numerical error statistics.
        if not estimate.groups and not estimate.corrections:
            detail["status"] = "empty fragmentation"
            rejection_counts["empty fragmentation"] += 1
            details.append(detail)
            continue

        predicted_hf = estimate.gas.enthalpy_formation_kJ_mol
        predicted_s = estimate.gas.entropy_J_mol_K
        detail["predicted_Hf_kJ_mol"] = predicted_hf
        detail["predicted_S_J_mol_K"] = predicted_s
        detail["warnings"] = list(estimate.warnings)
        if predicted_hf is not None:
            detail["Hf_error"] = predicted_hf - record["reference_Hf_kJ_mol"]
        if predicted_s is not None:
            detail["S_error"] = predicted_s - record["reference_S_J_mol_K"]

        missing = []
        if predicted_hf is None:
            missing.append("Hf")
        if predicted_s is None:
            missing.append("S")
        if missing:
            label = "missing predicted " + "/".join(missing)
            detail["status"] = label
            rejection_counts[label] += 1
        else:
            detail["status"] = "predicted both"
        details.append(detail)

    hf_summary = metric_summary(details, "Hf_error", "reference_Hf_kJ_mol")
    s_summary = metric_summary(details, "S_error", "reference_S_J_mol_K")
    both_count = sum(
        row.get("Hf_error") is not None and row.get("S_error") is not None
        for row in details
    )
    structure_count = sum(row.get("smiles") is not None for row in details)
    fragmented_count = sum(
        row["status"].startswith("predicted") or row["status"].startswith("missing predicted")
        for row in details
    )
    worst_hf = sorted(
        (row for row in details if row.get("Hf_error") is not None),
        key=lambda row: abs(row["Hf_error"]),
        reverse=True,
    )[:10]
    worst_s = sorted(
        (row for row in details if row.get("S_error") is not None),
        key=lambda row: abs(row["S_error"]),
        reverse=True,
    )[:10]

    return {
        "benchmark": {
            "method": "Domalski--Hearing (1993), pfdsim implementation",
            "phase": "ideal gas",
            "temperature_K": 298.15,
            "reference_scope": "strict local Hf and S",
            "reference_count": len(records),
            "reference_source_counts": dict(sorted(source_counts.items())),
            "structure_count": structure_count,
            "fragmentation_success_count": fragmented_count,
            "both_properties_count": both_count,
            "rejection_counts": dict(rejection_counts.most_common()),
        },
        "Hf_kJ_mol": hf_summary,
        "S_J_mol_K": s_summary,
        "worst_Hf": worst_hf,
        "worst_S": worst_s,
        "records": details,
    }


def format_metric(label: str, units: str, metrics: dict[str, Any]) -> list[str]:
    if not metrics.get("n"):
        return [f"{label}: no predictions"]
    r_squared = metrics.get("r_squared")
    r_squared_text = "n/a" if r_squared is None else f"{r_squared:.4f}"
    return [
        f"{label} ({units}), n={metrics['n']}",
        f"  bias:      {metrics['bias']:.3f}",
        f"  MAE:       {metrics['mae']:.3f}",
        f"  median AE: {metrics['median_ae']:.3f}",
        f"  p90 AE:    {metrics['p90_ae']:.3f}",
        f"  RMSE:      {metrics['rmse']:.3f}",
        f"  max AE:    {metrics['max_ae']:.3f}",
        f"  R^2:       {r_squared_text}",
    ]


def format_report(result: dict[str, Any]) -> str:
    info = result["benchmark"]
    lines = [
        "Domalski--Hearing gas Hf/S benchmark against pfdsim local data",
        "Reference: ideal-gas standard properties at 298.15 K",
        "",
        f"Reference components:       {info['reference_count']}",
        f"Structures resolved:        {info['structure_count']}",
        f"Fragmentation successes:   {info['fragmentation_success_count']}",
        f"Predicted Hf and S:         {info['both_properties_count']}",
        "",
        "Reference sources:",
    ]
    for source, count in info["reference_source_counts"].items():
        lines.append(f"  {source}: {count}")
    lines.extend(["", "Non-numeric outcomes:"])
    for reason, count in info["rejection_counts"].items():
        lines.append(f"  {reason}: {count}")
    lines.extend(["", *format_metric("Hf", "kJ/mol", result["Hf_kJ_mol"])])
    lines.extend(["", *format_metric("S", "J/(mol K)", result["S_J_mol_K"])])
    lines.extend(["", "Largest absolute Hf errors:"])
    for row in result["worst_Hf"]:
        lines.append(
            f"  {row['name']} ({row['cas'] or row['identity']}): "
            f"{row['Hf_error']:+.3f} kJ/mol"
        )
    lines.extend(["", "Largest absolute S errors:"])
    for row in result["worst_S"]:
        lines.append(
            f"  {row['name']} ({row['cas'] or row['identity']}): "
            f"{row['S_error']:+.3f} J/(mol K)"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    result = benchmark()
    report = format_report(result)
    JSON_OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    TEXT_OUTPUT.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
