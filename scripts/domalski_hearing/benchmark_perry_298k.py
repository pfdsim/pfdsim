"""Benchmark Domalski--Hearing Hvap, Cpg, and Cpl against Perry at 298.15 K.

Perry correlations are evaluated only inside their published temperature
ranges. Domalski--Hearing Hvap is the gas-minus-liquid standard enthalpy of
formation; Cpg and Cpl are the corresponding gas and liquid Table-2 heat
capacities. Carboxylic acids are excluded from the Hvap comparison because
vapor association makes the Perry correlation nonrepresentative of the
method's isolated-monomer gas-minus-liquid quantity; they remain in both heat-
capacity comparisons.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import domalski_hearing_method as dh  # noqa: E402
from chemicals.identifiers import search_chemical  # noqa: E402
from perry_properties import PerryPropertyLibrary  # noqa: E402


TEMPERATURE_K = 298.15
PERRY_PATH = ROOT / "data" / "perry_properties.json"
JSON_OUTPUT = ROOT / "outputs" / "domalski_hearing_perry_298k_benchmark.json"
TEXT_OUTPUT = ROOT / "outputs" / "domalski_hearing_perry_298k_benchmark.txt"

PROPERTIES = {
    "Hvap_kJ_mol": "kJ/mol",
    "Cpg_J_mol_K": "J/(mol K)",
    "Cpl_J_mol_K": "J/(mol K)",
}
CARBOXYLIC_ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")


def resolve_smiles(cas: str, entry: dict[str, Any]) -> tuple[str | None, str]:
    smiles = str(entry.get("smiles") or "").strip()
    if smiles:
        return smiles, "dataset"
    for identifier in (cas, entry.get("name")):
        if not identifier:
            continue
        try:
            metadata = search_chemical(str(identifier))
        except Exception:
            continue
        smiles = str(getattr(metadata, "smiles", "") or "").strip()
        if smiles:
            return smiles, "chemicals.identifiers"
    return None, "missing"


def is_carboxylic_acid(smiles: str | None) -> bool:
    if not smiles or CARBOXYLIC_ACID is None:
        return False
    molecule = Chem.MolFromSmiles(smiles)
    return molecule is not None and molecule.HasSubstructMatch(CARBOXYLIC_ACID)


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


def metric_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    matched = [
        row for row in records
        if row["references"].get(key) is not None
        and row["predictions"].get(key) is not None
    ]
    references = [float(row["references"][key]) for row in matched]
    errors = [float(row["errors"][key]) for row in matched]
    absolute = sorted(abs(value) for value in errors)
    percentages = sorted(
        abs(error / reference) * 100.0
        for error, reference in zip(errors, references)
        if reference != 0.0
    )
    reference_count = sum(row["references"].get(key) is not None for row in records)
    prediction_count = sum(row["predictions"].get(key) is not None for row in records)
    if not matched:
        return {
            "reference_count": reference_count,
            "prediction_count": prediction_count,
            "matched_count": 0,
        }
    mean_reference = statistics.fmean(references)
    ss_residual = sum(value * value for value in errors)
    ss_total = sum((value - mean_reference) ** 2 for value in references)
    return {
        "reference_count": reference_count,
        "prediction_count": prediction_count,
        "matched_count": len(matched),
        "reference_coverage_percent": 100.0 * len(matched) / reference_count,
        "bias": statistics.fmean(errors),
        "mae": statistics.fmean(absolute),
        "median_ae": statistics.median(absolute),
        "p90_ae": percentile(absolute, 0.90),
        "rmse": math.sqrt(statistics.fmean(value * value for value in errors)),
        "max_ae": absolute[-1],
        "mape_percent": statistics.fmean(percentages),
        "median_ape_percent": statistics.median(percentages),
        "p90_ape_percent": percentile(percentages, 0.90),
        "r_squared": None if ss_total == 0.0 else 1.0 - ss_residual / ss_total,
    }


def evaluate_references(
    library: PerryPropertyLibrary,
    entry: dict[str, Any],
) -> dict[str, float | None]:
    hvap = library.heat_of_vaporization_value_from_entry(entry, TEMPERATURE_K)
    cpg = library.heat_capacity_value_from_entry(entry, TEMPERATURE_K, "gas")
    cpl = library.heat_capacity_value_from_entry(entry, TEMPERATURE_K, "liquid")
    return {
        "Hvap_kJ_mol": None if hvap is None else hvap[0],
        "Cpg_J_mol_K": None if cpg is None else cpg[0],
        "Cpl_J_mol_K": None if cpl is None else cpl[0],
    }


def evaluate_predictions(estimate: dh.DomalskiHearingResult) -> dict[str, float | None]:
    gas_hf = estimate.gas.enthalpy_formation_kJ_mol
    liquid_hf = estimate.liquid.enthalpy_formation_kJ_mol
    return {
        "Hvap_kJ_mol": (
            None if gas_hf is None or liquid_hf is None else gas_hf - liquid_hf
        ),
        "Cpg_J_mol_K": estimate.gas.heat_capacity_J_mol_K,
        "Cpl_J_mol_K": estimate.liquid.heat_capacity_J_mol_K,
    }


def benchmark() -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    perry = json.loads(PERRY_PATH.read_text())["chemicals"]
    library = PerryPropertyLibrary()
    status_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    excluded_carboxylic_acids = 0

    for cas, entry in perry.items():
        smiles, smiles_source = resolve_smiles(cas, entry)
        carboxylic_acid = is_carboxylic_acid(smiles)
        if carboxylic_acid:
            excluded_carboxylic_acids += 1
        references = evaluate_references(library, entry)
        excluded_properties = []
        if carboxylic_acid:
            references["Hvap_kJ_mol"] = None
            excluded_properties.append("Hvap_kJ_mol")
        row: dict[str, Any] = {
            "cas": cas,
            "name": entry.get("name") or cas,
            "formula": entry.get("formula") or "",
            "smiles": smiles,
            "smiles_source": smiles_source,
            "references": references,
            "predictions": {key: None for key in PROPERTIES},
            "errors": {key: None for key in PROPERTIES},
            "excluded_properties": excluded_properties,
            "status": "pending",
        }
        if not smiles:
            row["status"] = "missing structure"
            status_counts[row["status"]] += 1
            records.append(row)
            continue

        try:
            estimate = dh.estimate(smiles)
        except dh.DomalskiHearingError as exc:
            row["status"] = rejection_category(str(exc))
            row["rejection_message"] = str(exc)
            status_counts[row["status"]] += 1
            records.append(row)
            continue

        if not estimate.groups and not estimate.corrections:
            row["status"] = "empty fragmentation"
            status_counts[row["status"]] += 1
            records.append(row)
            continue

        row["predictions"] = evaluate_predictions(estimate)
        for key in excluded_properties:
            row["predictions"][key] = None
        row["warnings"] = list(estimate.warnings)
        missing = []
        for key in PROPERTIES:
            if key in excluded_properties:
                continue
            reference = references.get(key)
            prediction = row["predictions"].get(key)
            if reference is not None and prediction is not None:
                row["errors"][key] = prediction - reference
            if prediction is None:
                missing.append(key.split("_", 1)[0])
        row["status"] = "predicted all" if not missing else "missing " + "/".join(missing)
        status_counts[row["status"]] += 1
        records.append(row)

    metrics = {key: metric_summary(records, key) for key in PROPERTIES}
    worst = {
        key: sorted(
            (row for row in records if row["errors"].get(key) is not None),
            key=lambda row: abs(row["errors"][key]),
            reverse=True,
        )[:10]
        for key in PROPERTIES
    }
    return {
        "benchmark": {
            "method": "Domalski--Hearing (1993), pfdsim implementation",
            "reference": "Perry's Chemical Engineers' Handbook, 9th ed.",
            "temperature_K": TEMPERATURE_K,
            "perry_component_count": len(perry),
            "included_component_count": len(records),
            "hvap_excluded_carboxylic_acid_count": excluded_carboxylic_acids,
            "structure_count": sum(row["smiles"] is not None for row in records),
            "fragmentation_success_count": sum(
                row["status"].startswith("predicted") or row["status"].startswith("missing ")
                and row["status"] != "missing structure"
                for row in records
            ),
            "status_counts": dict(status_counts.most_common()),
            "reference_policy": "evaluate only within Perry correlation range",
            "hvap_definition": "Hf(gas) - Hf(liquid)",
            "exclusion_policy": "exclude neutral carboxylic-acid substructures from Hvap only",
        },
        "metrics": metrics,
        "worst": worst,
        "records": records,
    }


def format_metric(key: str, units: str, metrics: dict[str, Any]) -> list[str]:
    label = key.split("_", 1)[0]
    r_squared = metrics.get("r_squared")
    r_squared_text = "n/a" if r_squared is None else f"{r_squared:.4f}"
    return [
        f"{label} ({units})",
        f"  Perry references at 298.15 K: {metrics['reference_count']}",
        f"  matched predictions:          {metrics['matched_count']} "
        f"({metrics['reference_coverage_percent']:.1f}%)",
        f"  bias:                         {metrics['bias']:.3f}",
        f"  MAE:                          {metrics['mae']:.3f}",
        f"  median AE:                    {metrics['median_ae']:.3f}",
        f"  p90 AE:                       {metrics['p90_ae']:.3f}",
        f"  RMSE:                         {metrics['rmse']:.3f}",
        f"  max AE:                       {metrics['max_ae']:.3f}",
        f"  MAPE:                         {metrics['mape_percent']:.2f}%",
        f"  median APE:                   {metrics['median_ape_percent']:.2f}%",
        f"  p90 APE:                      {metrics['p90_ape_percent']:.2f}%",
        f"  R^2:                          {r_squared_text}",
    ]


def format_report(result: dict[str, Any]) -> str:
    info = result["benchmark"]
    lines = [
        "Domalski--Hearing benchmark against Perry correlations at 298.15 K",
        "Perry correlations are evaluated only within their published ranges.",
        "",
        f"Perry components:          {info['perry_component_count']}",
        f"Acids excluded from Hvap:  {info['hvap_excluded_carboxylic_acid_count']}",
        f"Components benchmarked:    {info['included_component_count']}",
        f"Structures resolved:       {info['structure_count']}",
        f"Fragmentation successes:  {info['fragmentation_success_count']}",
        "",
        "Structure/fragmentation outcomes:",
    ]
    for status, count in info["status_counts"].items():
        lines.append(f"  {status}: {count}")
    for key, units in PROPERTIES.items():
        lines.extend(["", *format_metric(key, units, result["metrics"][key])])
        lines.append("  largest absolute errors:")
        for row in result["worst"][key]:
            lines.append(
                f"    {row['name']} ({row['cas']}): "
                f"{row['errors'][key]:+.3f} {units}"
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
