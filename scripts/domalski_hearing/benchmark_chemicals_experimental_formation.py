"""Benchmark Domalski--Hearing gas Hf and S against ``chemicals`` data.

Only evaluated experimental compilations are admitted: ATcT, CRC, NIST
WebBook, and JANAF. Joback, Yaws, API-TDB, and TRC are excluded because the
``chemicals`` package documents estimated or computational content in them.
One highest-priority reference is selected per CAS and property.
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

import chemicals.reaction as reaction  # noqa: E402
import domalski_hearing_method as dh  # noqa: E402
from chemicals.identifiers import int_to_CAS, search_chemical  # noqa: E402


JSON_OUTPUT = ROOT / "outputs" / "domalski_hearing_chemicals_experimental_formation_benchmark.json"
TEXT_OUTPUT = ROOT / "outputs" / "domalski_hearing_chemicals_experimental_formation_benchmark.txt"
TEMPERATURE_K = 298.15
HF_PERCENT_ERROR_THRESHOLD_KJ_MOL = 300.0
CHEMICALS_PRESSURE_PA = 100_000.0
ENTROPY_PRESSURE_CORRECTION_J_MOL_K = dh.R_J_MOL_K * math.log(
    dh.PRESSURE_PA / CHEMICALS_PRESSURE_PA
)

SOURCE_PRIORITIES = {
    "Hf": ("ATCT_G", "CRC", "WEBBOOK", "JANAF"),
    "S": ("CRC", "WEBBOOK", "JANAF"),
}
SOURCE_LABELS = {
    "ATCT_G": "Active Thermochemical Tables 1.112",
    "CRC": "CRC Handbook",
    "WEBBOOK": "NIST WebBook",
    "JANAF": "JANAF 1998",
}
EXCLUDED_SOURCE_REASONS = {
    "API_TDB_G": "not exposed as an experimental Hfg method by chemicals",
    "TRC": "chemicals documents possible computational values",
    "YAWS": "chemicals documents the compilation as mostly estimated",
    "JOBACK": "group-contribution estimate",
}


def normalized_cas(index: Any) -> str:
    return str(index) if isinstance(index, str) else int_to_CAS(int(index))


def source_records(
    sources: dict[str, Any],
    property_name: str,
    column: str,
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for source in SOURCE_PRIORITIES[property_name]:
        dataframe = sources[source]
        for index, value in dataframe[column].items():
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            cas = normalized_cas(index)
            if cas in selected:
                continue
            selected[cas] = {
                "property": property_name,
                "cas": cas,
                "source": source,
                "source_label": SOURCE_LABELS[source],
                "reference": numeric / 1000.0 if property_name == "Hf" else numeric,
                "reference_units": "kJ/mol" if property_name == "Hf" else "J/(mol K)",
                "prediction": None,
                "error": None,
                "status": "pending",
            }
    return list(selected.values())


def load_reference_records() -> list[dict[str, Any]]:
    reaction._load_reaction_data()
    records = source_records(reaction.Hfg_sources, "Hf", "Hfg")
    records.extend(source_records(reaction.S0g_sources, "S", "S0g"))
    return records


def resolve_identity(cas: str) -> dict[str, str | None]:
    try:
        metadata = search_chemical(cas)
    except Exception:
        return {
            "name": cas,
            "formula": None,
            "smiles": None,
            "resolved_cas": None,
            "identity_status": "missing",
        }
    resolved_cas = str(getattr(metadata, "CASs", "") or "")
    if resolved_cas != cas:
        return {
            "name": cas,
            "formula": None,
            "smiles": None,
            "resolved_cas": resolved_cas or None,
            "identity_status": "CAS mismatch",
        }
    return {
        "name": str(getattr(metadata, "common_name", "") or cas),
        "formula": str(getattr(metadata, "formula", "") or "") or None,
        "smiles": str(getattr(metadata, "smiles", "") or "").strip() or None,
        "resolved_cas": resolved_cas,
        "identity_status": "exact CAS",
    }


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


def add_predictions(records: list[dict[str, Any]]) -> None:
    identity_cache: dict[str, dict[str, str | None]] = {}
    estimate_cache: dict[str, tuple[str, Any]] = {}
    for row in records:
        cas = row["cas"]
        identity = identity_cache.setdefault(cas, resolve_identity(cas))
        row.update(identity)
        smiles = identity["smiles"]
        if not smiles:
            row["status"] = (
                "identity CAS mismatch"
                if identity["identity_status"] == "CAS mismatch"
                else "missing structure"
            )
            continue

        if smiles not in estimate_cache:
            try:
                estimate = dh.estimate(smiles)
            except dh.DomalskiHearingError as exc:
                estimate_cache[smiles] = (rejection_category(str(exc)), str(exc))
            else:
                if not estimate.groups and not estimate.corrections:
                    estimate_cache[smiles] = ("empty fragmentation", None)
                else:
                    estimate_cache[smiles] = ("success", estimate)

        status, payload = estimate_cache[smiles]
        if status != "success":
            row["status"] = status
            if payload:
                row["rejection_message"] = payload
            continue

        estimate = payload
        if row["property"] == "Hf":
            prediction = estimate.gas.enthalpy_formation_kJ_mol
            applicability = dh.assess_hf_applicability(estimate)
            row["hf_applicable"] = applicability.applicable
            row["hf_applicability_reasons"] = list(applicability.reasons)
        else:
            prediction = estimate.gas.entropy_J_mol_K
            if prediction is not None:
                prediction += ENTROPY_PRESSURE_CORRECTION_J_MOL_K
        if prediction is None:
            row["status"] = "missing predicted value"
            row["warnings"] = list(estimate.warnings)
            continue
        row["prediction"] = float(prediction)
        row["error"] = float(prediction) - float(row["reference"])
        row["status"] = "matched"


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


def summarize(records: list[dict[str, Any]], property_name: str) -> dict[str, Any]:
    matched = [row for row in records if row["error"] is not None]
    status_counts = Counter(row["status"] for row in records)
    summary: dict[str, Any] = {
        "reference_count": len(records),
        "matched_count": len(matched),
        "coverage_percent": 100.0 * len(matched) / len(records) if records else 0.0,
        "status_counts": dict(status_counts.most_common()),
    }
    if not matched:
        return summary
    references = [float(row["reference"]) for row in matched]
    errors = [float(row["error"]) for row in matched]
    absolute = sorted(abs(error) for error in errors)
    mean_reference = statistics.fmean(references)
    ss_residual = sum(error * error for error in errors)
    ss_total = sum((reference - mean_reference) ** 2 for reference in references)
    summary.update({
        "bias": statistics.fmean(errors),
        "mae": statistics.fmean(absolute),
        "median_ae": statistics.median(absolute),
        "p90_ae": percentile(absolute, 0.90),
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
        "max_ae": absolute[-1],
        "r_squared": None if ss_total == 0.0 else 1.0 - ss_residual / ss_total,
    })
    if property_name == "S":
        percentages = sorted(
            abs(error / reference) * 100.0
            for error, reference in zip(errors, references)
            if reference != 0.0
        )
        summary.update({
            "mape_percent": statistics.fmean(percentages),
            "median_ape_percent": statistics.median(percentages),
            "p90_ape_percent": percentile(percentages, 0.90),
        })
    return summary


def summarize_hf_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in records if row["error"] is not None]
    lower_magnitude = [
        row for row in matched
        if abs(float(row["reference"])) <= HF_PERCENT_ERROR_THRESHOLD_KJ_MOL
    ]
    higher_magnitude = [
        row for row in matched
        if abs(float(row["reference"])) > HF_PERCENT_ERROR_THRESHOLD_KJ_MOL
    ]

    def absolute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"count": 0}
        references = [float(row["reference"]) for row in rows]
        errors = [float(row["error"]) for row in rows]
        absolute = sorted(abs(error) for error in errors)
        mean_reference = statistics.fmean(references)
        ss_total = sum(
            (reference - mean_reference) ** 2 for reference in references
        )
        return {
            "count": len(rows),
            "bias_kJ_mol": statistics.fmean(errors),
            "mae_kJ_mol": statistics.fmean(absolute),
            "median_ae_kJ_mol": statistics.median(absolute),
            "p90_ae_kJ_mol": percentile(absolute, 0.90),
            "rmse_kJ_mol": math.sqrt(
                statistics.fmean(error * error for error in errors)
            ),
            "max_ae_kJ_mol": absolute[-1],
            "r_squared": (
                None
                if ss_total == 0.0
                else 1.0 - sum(error * error for error in errors) / ss_total
            ),
        }

    def percentage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"count": 0}
        signed = [
            float(row["error"]) / abs(float(row["reference"])) * 100.0
            for row in rows
        ]
        absolute = sorted(abs(error) for error in signed)
        return {
            "count": len(rows),
            "bias_percent": statistics.fmean(signed),
            "mape_percent": statistics.fmean(absolute),
            "median_ape_percent": statistics.median(absolute),
            "p90_ape_percent": percentile(absolute, 0.90),
            "max_ape_percent": absolute[-1],
        }

    return {
        "reference_count": len(records),
        "matched_count": len(matched),
        "coverage_percent": 100.0 * len(matched) / len(records) if records else 0.0,
        "status_counts": dict(Counter(row["status"] for row in records).most_common()),
        "absolute_error_cohort": absolute_summary(lower_magnitude),
        "percentage_error_cohort": percentage_summary(higher_magnitude),
    }


def benchmark() -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    records = load_reference_records()
    add_predictions(records)
    metrics: dict[str, Any] = {}
    source_metrics: dict[str, Any] = {}
    worst: dict[str, Any] = {}
    hf_rows = [row for row in records if row["property"] == "Hf"]
    hf_numeric = [row for row in hf_rows if row["error"] is not None]
    hf_applicable = [row for row in hf_numeric if row.get("hf_applicable") is True]
    hf_non_applicable = [
        row for row in hf_numeric if row.get("hf_applicable") is False
    ]
    metrics["Hf"] = {
        "all": summarize_hf_split(hf_rows),
        "applicable": summarize_hf_split(hf_applicable),
        "non_applicable": summarize_hf_split(hf_non_applicable),
    }
    source_metrics["Hf"] = {
        source: {
            "all": summarize_hf_split([
                row for row in hf_rows if row["source"] == source
            ]),
            "applicable": summarize_hf_split([
                row for row in hf_applicable if row["source"] == source
            ]),
        }
        for source in SOURCE_PRIORITIES["Hf"]
    }
    worst["Hf_applicable_absolute"] = sorted(
        (
            row for row in hf_applicable
            if abs(float(row["reference"])) <= HF_PERCENT_ERROR_THRESHOLD_KJ_MOL
        ),
        key=lambda row: abs(row["error"]),
        reverse=True,
    )[:10]
    worst["Hf_applicable_percentage"] = sorted(
        (
            row for row in hf_applicable
            if abs(float(row["reference"])) > HF_PERCENT_ERROR_THRESHOLD_KJ_MOL
        ),
        key=lambda row: abs(row["error"] / row["reference"]),
        reverse=True,
    )[:10]

    s_rows = [row for row in records if row["property"] == "S"]
    metrics["S"] = summarize(s_rows, "S")
    source_metrics["S"] = {
        source: summarize(
            [row for row in s_rows if row["source"] == source],
            "S",
        )
        for source in SOURCE_PRIORITIES["S"]
    }
    worst["S"] = sorted(
        (row for row in s_rows if row["error"] is not None),
        key=lambda row: abs(row["error"]),
        reverse=True,
    )[:10]
    applicability_reasons = Counter(
        reason
        for row in hf_non_applicable
        for reason in row.get("hf_applicability_reasons", [])
    )
    return {
        "benchmark": {
            "method": "Domalski--Hearing (1993), pfdsim implementation",
            "reference": "chemicals package evaluated experimental compilations",
            "temperature_K": TEMPERATURE_K,
            "hf_percent_error_threshold_kJ_mol": HF_PERCENT_ERROR_THRESHOLD_KJ_MOL,
            "reference_pressure_Pa": CHEMICALS_PRESSURE_PA,
            "domalski_hearing_pressure_Pa": dh.PRESSURE_PA,
            "entropy_pressure_correction_J_mol_K": ENTROPY_PRESSURE_CORRECTION_J_MOL_K,
            "source_priorities": {
                key: list(value) for key, value in SOURCE_PRIORITIES.items()
            },
            "excluded_source_reasons": EXCLUDED_SOURCE_REASONS,
            "record_count": len(records),
            "unique_cas_count": len({row["cas"] for row in records}),
            "hf_numeric_count": len(hf_numeric),
            "hf_applicable_count": len(hf_applicable),
            "hf_non_applicable_count": len(hf_non_applicable),
            "hf_applicability_reason_counts": dict(applicability_reasons.most_common()),
        },
        "metrics": metrics,
        "source_metrics": source_metrics,
        "worst": worst,
        "records": records,
    }


def metric_lines(label: str, metric: dict[str, Any], property_name: str) -> list[str]:
    units = "kJ/mol" if property_name == "Hf" else "J/(mol K)"
    lines = [
        f"  {label}: refs={metric['reference_count']} matched={metric['matched_count']} "
        f"coverage={metric['coverage_percent']:.1f}%",
    ]
    if not metric.get("matched_count"):
        return lines
    lines.append(
        f"    bias={metric['bias']:+.3f} {units}  MAE={metric['mae']:.3f}  "
        f"median AE={metric['median_ae']:.3f}  p90 AE={metric['p90_ae']:.3f}  "
        f"RMSE={metric['rmse']:.3f}  R^2={metric['r_squared']:.4f}"
    )
    if property_name == "S":
        lines.append(
            f"    MAPE={metric['mape_percent']:.2f}%  "
            f"median APE={metric['median_ape_percent']:.2f}%  "
            f"p90 APE={metric['p90_ape_percent']:.2f}%"
        )
    return lines


def hf_split_lines(label: str, metric: dict[str, Any]) -> list[str]:
    absolute = metric["absolute_error_cohort"]
    percentage = metric["percentage_error_cohort"]
    lines = [
        f"  {label}: refs={metric['reference_count']} matched={metric['matched_count']} "
        f"coverage={metric['coverage_percent']:.1f}%",
    ]
    if absolute.get("count"):
        lines.append(
            f"    |Hf| <= {HF_PERCENT_ERROR_THRESHOLD_KJ_MOL:g}: n={absolute['count']}  "
            f"bias={absolute['bias_kJ_mol']:+.3f} kJ/mol  "
            f"MAE={absolute['mae_kJ_mol']:.3f}  "
            f"median AE={absolute['median_ae_kJ_mol']:.3f}  "
            f"p90 AE={absolute['p90_ae_kJ_mol']:.3f}"
        )
    if percentage.get("count"):
        lines.append(
            f"    |Hf| > {HF_PERCENT_ERROR_THRESHOLD_KJ_MOL:g}: n={percentage['count']}  "
            f"bias={percentage['bias_percent']:+.3f}%  "
            f"MAPE={percentage['mape_percent']:.3f}%  "
            f"median APE={percentage['median_ape_percent']:.3f}%  "
            f"p90 APE={percentage['p90_ape_percent']:.3f}%"
        )
    return lines


def format_report(result: dict[str, Any]) -> str:
    info = result["benchmark"]
    lines = [
        "Domalski--Hearing gas Hf/S benchmark against chemicals experimental data",
        "One highest-priority evaluated reference is selected per CAS and property.",
        "Estimated/computational source collections are excluded.",
        "",
        f"Reference records: {info['record_count']}",
        f"Unique CAS:        {info['unique_cas_count']}",
        f"S pressure correction to 1 bar: "
        f"{info['entropy_pressure_correction_J_mol_K']:+.6f} J/(mol K)",
    ]
    lines.extend(["", "Hf"])
    for cohort in ("all", "applicable", "non_applicable"):
        lines.extend(hf_split_lines(cohort, result["metrics"]["Hf"][cohort]))
    lines.append("  non-applicability reasons:")
    for reason, count in info["hf_applicability_reason_counts"].items():
        lines.append(f"    {reason}: {count}")
    lines.append("  applicable selected-source cohorts:")
    for source in SOURCE_PRIORITIES["Hf"]:
        lines.extend(hf_split_lines(
            source,
            result["source_metrics"]["Hf"][source]["applicable"],
        ))
    lines.append("  largest applicable absolute errors (|Hf| <= 300 kJ/mol):")
    for row in result["worst"]["Hf_applicable_absolute"]:
        lines.append(
            f"    {row['name']} ({row['cas']}, {row['source']}): "
            f"{row['error']:+.3f} kJ/mol"
        )
    lines.append("  largest applicable percentage errors (|Hf| > 300 kJ/mol):")
    for row in result["worst"]["Hf_applicable_percentage"]:
        lines.append(
            f"    {row['name']} ({row['cas']}, {row['source']}): "
            f"{row['error'] / abs(row['reference']) * 100.0:+.3f}%"
        )

    lines.extend(["", "S"])
    lines.extend(metric_lines("all", result["metrics"]["S"], "S"))
    lines.append("  selected-source cohorts:")
    for source in SOURCE_PRIORITIES["S"]:
        lines.extend(metric_lines(
            source,
            result["source_metrics"]["S"][source],
            "S",
        ))
    lines.append("  largest absolute errors:")
    for row in result["worst"]["S"]:
        lines.append(
            f"    {row['name']} ({row['cas']}, {row['source']}): "
            f"{row['error']:+.3f} J/(mol K)"
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
