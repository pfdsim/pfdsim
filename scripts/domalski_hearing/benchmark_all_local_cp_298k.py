"""Benchmark Domalski--Hearing Cpg and Cpl against all empirical local curves.

The benchmark reads pfdsim's canonical ideal-gas and ordinary-liquid heat-
capacity databases at 298.15 K. Perry and non-Perry records are reported as
separate cohorts. Adjusted Psi4 RRHO gas curves are excluded because they are
computational estimates rather than experimental or empirical correlations.
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import RDLogger


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import domalski_hearing_method as dh  # noqa: E402
from chemicals.heat_capacity import Zabransky_quasi_polynomial  # noqa: E402
from chemicals.identifiers import search_chemical  # noqa: E402


TEMPERATURE_K = 298.15
R_J_MOL_K = 8.31446261815324
GAS_DATABASE = ROOT / "data" / "ideal_gas_heat_capacity.sqlite"
LIQUID_DATABASE = ROOT / "data" / "liquid_heat_capacity.sqlite"
JSON_OUTPUT = ROOT / "outputs" / "domalski_hearing_all_local_cp_298k_benchmark.json"
TEXT_OUTPUT = ROOT / "outputs" / "domalski_hearing_all_local_cp_298k_benchmark.txt"
PERRY_SOURCE = "perry_9e"
NONEXPERIMENTAL_GAS_SOURCES = {"psi4_adjusted"}

PROPERTY_UNITS = {
    "Cpg": "J/(mol K)",
    "Cpl": "J/(mol K)",
}
REFERENCE_LIMITS = {
    "Cpg": (0.0, 2000.0),
    "Cpl": (3.0 * R_J_MOL_K, 2000.0),
}


def chebyshev_value(coefficients: list[float], x: float) -> float:
    b1 = 0.0
    b2 = 0.0
    for index in range(len(coefficients) - 1, 0, -1):
        b0 = 2.0 * x * b1 - b2 + float(coefficients[index])
        b2 = b1
        b1 = b0
    return x * b1 - b2 + float(coefficients[0])


def evaluate_gas_row(row: sqlite3.Row) -> float:
    coefficients = json.loads(row["cp_coefficients_json"])
    center = float(row["map_center_K"])
    scale = float(row["map_scale"])
    x = (TEMPERATURE_K - center) / (scale * (TEMPERATURE_K + center))
    return chebyshev_value(coefficients, x)


def evaluate_liquid_row(row: sqlite3.Row) -> float:
    coefficients = json.loads(row["cp_coefficients_json"])
    if row["model"] == "linear_chebyshev_liquid_cp_v1":
        center = float(row["map_center_K"])
        half_width = float(row["map_scale"])
        return chebyshev_value(coefficients, (TEMPERATURE_K - center) / half_width)
    if row["model"] == "native_zabransky_quasipolynomial_v1":
        return float(Zabransky_quasi_polynomial(
            TEMPERATURE_K,
            float(row["critical_temperature_K"]),
            *coefficients,
        ))
    raise ValueError(f"unsupported canonical liquid Cp model {row['model']!r}")


def plausible_reference(property_name: str, value: float) -> tuple[bool, str]:
    lower, upper = REFERENCE_LIMITS[property_name]
    if not math.isfinite(value):
        return False, "nonfinite reference"
    if value < lower:
        return False, f"reference below plausibility floor {lower:.6g} J/(mol K)"
    if value > upper:
        return False, f"reference above plausibility ceiling {upper:.6g} J/(mol K)"
    return True, ""


def load_reference_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    gas_uri = f"file:{GAS_DATABASE.resolve()}?mode=ro"
    with sqlite3.connect(gas_uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM canonical_ideal_gas_cp
            WHERE Tmin_fit_K <= ? AND Tmax_fit_K >= ?
            ORDER BY cas
            """,
            (TEMPERATURE_K, TEMPERATURE_K),
        ).fetchall()
    for row in rows:
        if row["source"] in NONEXPERIMENTAL_GAS_SOURCES:
            continue
        value = evaluate_gas_row(row)
        plausible, reason = plausible_reference("Cpg", value)
        record = {
            "property": "Cpg",
            "cas": row["cas"],
            "name": row["name"],
            "formula": row["formula"],
            "source": row["source"],
            "source_label": row["source_label"],
            "source_lineage": row["source_lineage"],
            "cohort": "Perry" if row["source"] == PERRY_SOURCE else "non-Perry",
            "reference": value,
            "prediction": None,
            "error": None,
            "status": "pending",
        }
        if not plausible:
            record["status"] = "excluded nonphysical reference"
            record["exclusion_reason"] = reason
            excluded.append(record)
        else:
            records.append(record)

    liquid_uri = f"file:{LIQUID_DATABASE.resolve()}?mode=ro"
    with sqlite3.connect(liquid_uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM canonical_liquid_cp
            WHERE Tmin_fit_K <= ? AND Tmax_fit_K >= ?
            ORDER BY cas
            """,
            (TEMPERATURE_K, TEMPERATURE_K),
        ).fetchall()
    for row in rows:
        value = evaluate_liquid_row(row)
        plausible, reason = plausible_reference("Cpl", value)
        record = {
            "property": "Cpl",
            "cas": row["cas"],
            "name": row["name"],
            "formula": row["formula"],
            "source": row["source"],
            "source_label": row["source_label"],
            "source_lineage": row["source_lineage"],
            "cohort": "Perry" if row["source"] == PERRY_SOURCE else "non-Perry",
            "reference": value,
            "prediction": None,
            "error": None,
            "status": "pending",
        }
        if not plausible:
            record["status"] = "excluded nonphysical reference"
            record["exclusion_reason"] = reason
            excluded.append(record)
        else:
            records.append(record)
    return records, excluded


def resolve_smiles(cas: str, name: str) -> str | None:
    for identifier in (cas, name):
        if not identifier:
            continue
        try:
            metadata = search_chemical(str(identifier))
        except Exception:
            continue
        smiles = str(getattr(metadata, "smiles", "") or "").strip()
        if smiles:
            return smiles
    return None


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
    smiles_cache: dict[str, str | None] = {}
    estimate_cache: dict[str, tuple[str, Any]] = {}
    for row in records:
        cas = row["cas"]
        if cas not in smiles_cache:
            smiles_cache[cas] = resolve_smiles(cas, row["name"])
        smiles = smiles_cache[cas]
        row["smiles"] = smiles
        if not smiles:
            row["status"] = "missing structure"
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
        prediction = (
            estimate.gas.heat_capacity_J_mol_K
            if row["property"] == "Cpg"
            else estimate.liquid.heat_capacity_J_mol_K
        )
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


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
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
    absolute = sorted(abs(value) for value in errors)
    percentages = sorted(
        abs(error / reference) * 100.0
        for error, reference in zip(errors, references)
    )
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
        "mape_percent": statistics.fmean(percentages),
        "median_ape_percent": statistics.median(percentages),
        "p90_ape_percent": percentile(percentages, 0.90),
        "r_squared": None if ss_total == 0.0 else 1.0 - ss_residual / ss_total,
    })
    return summary


def benchmark() -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    records, excluded_records = load_reference_records()
    add_predictions(records)
    metrics: dict[str, Any] = {}
    source_metrics: dict[str, Any] = {}
    worst_nonperry: dict[str, Any] = {}
    for property_name in PROPERTY_UNITS:
        property_rows = [row for row in records if row["property"] == property_name]
        metrics[property_name] = {
            "all": summarize(property_rows),
            "Perry": summarize([row for row in property_rows if row["cohort"] == "Perry"]),
            "non-Perry": summarize([
                row for row in property_rows if row["cohort"] == "non-Perry"
            ]),
        }
        sources = sorted({row["source"] for row in property_rows})
        source_metrics[property_name] = {
            source: {
                "source_label": next(
                    row["source_label"] for row in property_rows if row["source"] == source
                ),
                **summarize([row for row in property_rows if row["source"] == source]),
            }
            for source in sources
        }
        worst_nonperry[property_name] = sorted(
            (
                row for row in property_rows
                if row["cohort"] == "non-Perry" and row["error"] is not None
            ),
            key=lambda row: abs(row["error"]),
            reverse=True,
        )[:10]

    return {
        "benchmark": {
            "method": "Domalski--Hearing (1993), pfdsim implementation",
            "temperature_K": TEMPERATURE_K,
            "reference_scope": "canonical empirical local Cp records in range",
            "excluded_sources": sorted(NONEXPERIMENTAL_GAS_SOURCES),
            "record_count": len(records),
            "unique_cas_count": len({row["cas"] for row in records}),
            "excluded_nonphysical_reference_count": len(excluded_records),
            "reference_limits_J_mol_K": {
                key: {"minimum": limits[0], "maximum": limits[1]}
                for key, limits in REFERENCE_LIMITS.items()
            },
        },
        "metrics": metrics,
        "source_metrics": source_metrics,
        "worst_nonperry": worst_nonperry,
        "excluded_records": excluded_records,
        "records": records,
    }


def metric_line(label: str, metric: dict[str, Any]) -> str:
    if not metric.get("matched_count"):
        return (
            f"  {label:12s} refs={metric['reference_count']:4d} matched=   0 "
            f"coverage={metric['coverage_percent']:5.1f}%"
        )
    return (
        f"  {label:12s} refs={metric['reference_count']:4d} "
        f"matched={metric['matched_count']:4d} coverage={metric['coverage_percent']:5.1f}%  "
        f"MAPE={metric['mape_percent']:6.2f}% median={metric['median_ape_percent']:5.2f}% "
        f"p90={metric['p90_ape_percent']:6.2f}% MAE={metric['mae']:7.3f} "
        f"bias={metric['bias']:+7.3f}"
    )


def format_report(result: dict[str, Any]) -> str:
    info = result["benchmark"]
    lines = [
        "Domalski--Hearing Cpg/Cpl benchmark against all local empirical curves",
        "References are canonical records whose fitted range contains 298.15 K.",
        "Adjusted Psi4 RRHO records are excluded as computational estimates.",
        "",
        f"Reference records: {info['record_count']}",
        f"Unique CAS:        {info['unique_cas_count']}",
        f"Nonphysical refs excluded: {info['excluded_nonphysical_reference_count']}",
    ]
    if result["excluded_records"]:
        lines.append("Excluded references:")
        for row in result["excluded_records"]:
            lines.append(
                f"  {row['property']} {row['name']} ({row['cas']}, {row['source']}): "
                f"{row['reference']:.6g} J/(mol K); {row['exclusion_reason']}"
            )
    for property_name, units in PROPERTY_UNITS.items():
        lines.extend(["", f"{property_name} ({units})"])
        for cohort in ("all", "Perry", "non-Perry"):
            lines.append(metric_line(cohort, result["metrics"][property_name][cohort]))
        lines.append("  non-Perry sources:")
        for source, metric in result["source_metrics"][property_name].items():
            if source == PERRY_SOURCE:
                continue
            lines.append(metric_line(source, metric))
        lines.append("  largest non-Perry absolute errors:")
        for row in result["worst_nonperry"][property_name]:
            lines.append(
                f"    {row['name']} ({row['cas']}, {row['source']}): "
                f"{row['error']:+.3f} {units}"
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
