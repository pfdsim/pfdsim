#!/usr/bin/env python3
"""Benchmark pfdsim's Yoon-Thodos vapor viscosity against Perry 9th.

Every Perry vapor-viscosity curve is sampled uniformly over its full stated
temperature range.  Perry supplies MW, Tc, and Pc, each at quality 1.0, so the
comparison measures the Yoon-Thodos correlation and pfdsim's composition-class
quality factor without mixing in property-resolution fallbacks.

The human-readable report is written to stdout.  Use ``--output-json`` for a
machine-readable report containing aggregate metrics and every compound-level
result.  This script does not modify source data or runtime caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perry_properties import PerryPropertyLibrary  # noqa: E402
from property_resolver import (  # noqa: E402
    PropertyResolutionResult,
    PropertyResolver,
)


DEFAULT_DATABASE = ROOT / "data" / "perry_properties.json"


@dataclass(frozen=True)
class PointResult:
    cas: str
    temperature_K: float
    reduced_temperature: float
    range_position: float
    reference_Pa_s: float
    predicted_Pa_s: float
    signed_error_percent: float
    absolute_error_percent: float
    log_error: float
    chemical_class: str
    molecular_weight: float


@dataclass(frozen=True)
class CompoundResult:
    cas: str
    name: str
    formula: str
    chemical_class: str
    assigned_quality: float
    molecular_weight: float
    Tc_K: float
    Pc_bar: float
    Tmin_K: float
    Tmax_K: float
    Tr_min: float
    Tr_max: float
    points: int
    mape_percent: float
    median_ape_percent: float
    p90_ape_percent: float
    p95_ape_percent: float
    maximum_ape_percent: float
    mean_signed_error_percent: float
    median_signed_error_percent: float
    rms_log_error_percent: float
    geometric_bias_percent: float
    fraction_within_10_percent: float
    fraction_within_20_percent: float
    fraction_within_30_percent: float
    fraction_within_50_percent: float
    worst_temperature_K: float


class PerryInputResolver(PropertyResolver):
    """Run the production Yoon-Thodos method with fixed Perry criticals."""

    def __init__(self) -> None:
        super().__init__()
        self._benchmark_critical_results: dict[str, PropertyResolutionResult] = {}

    def set_critical_inputs(self, *, Tc_K: float, Pc_bar: float) -> None:
        self._benchmark_critical_results = {
            "Tc": PropertyResolutionResult(
                value=Tc_K,
                source="local",
                method="perry_critical_constant",
                quality=1.0,
                notes="Perry 9th Table 2-106; units K",
            ),
            "Pc": PropertyResolutionResult(
                value=Pc_bar,
                source="local",
                method="perry_critical_constant",
                quality=1.0,
                notes="Perry 9th Table 2-106; units bar",
            ),
        }

    def resolve_critical_properties(self, *args, **kwargs):
        return self._benchmark_critical_results


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def value_summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": max(values) if values else math.nan,
    }


def point_summary(points: Sequence[PointResult]) -> dict[str, float | int]:
    absolute = [point.absolute_error_percent for point in points]
    signed = [point.signed_error_percent for point in points]
    logs = [point.log_error for point in points]
    return {
        "points": len(points),
        "components": len({point.cas for point in points}),
        "mape_percent": mean(absolute),
        "median_ape_percent": percentile(absolute, 0.50),
        "p90_ape_percent": percentile(absolute, 0.90),
        "p95_ape_percent": percentile(absolute, 0.95),
        "p99_ape_percent": percentile(absolute, 0.99),
        "maximum_ape_percent": max(absolute) if absolute else math.nan,
        "mean_signed_error_percent": mean(signed),
        "median_signed_error_percent": percentile(signed, 0.50),
        "rms_log_error_percent": (
            100.0 * math.sqrt(mean([value * value for value in logs]))
            if logs else math.nan
        ),
        "geometric_bias_percent": (
            100.0 * math.expm1(mean(logs)) if logs else math.nan
        ),
        "underprediction_fraction": (
            sum(value < 0.0 for value in signed) / len(signed)
            if signed else math.nan
        ),
        "fraction_within_10_percent": (
            sum(value <= 10.0 for value in absolute) / len(absolute)
            if absolute else math.nan
        ),
        "fraction_within_20_percent": (
            sum(value <= 20.0 for value in absolute) / len(absolute)
            if absolute else math.nan
        ),
        "fraction_within_30_percent": (
            sum(value <= 30.0 for value in absolute) / len(absolute)
            if absolute else math.nan
        ),
        "fraction_within_50_percent": (
            sum(value <= 50.0 for value in absolute) / len(absolute)
            if absolute else math.nan
        ),
        "fraction_within_factor_two": (
            sum(abs(value) <= math.log(2.0) for value in logs) / len(logs)
            if logs else math.nan
        ),
    }


def chemical_class(notes: str) -> str:
    if "sparse-heteroatom multiplier" in notes:
        return "sparse_heteroatom"
    if "hydrocarbon multiplier" in notes:
        return "hydrocarbon"
    if "heteroatom-rich multiplier" in notes:
        return "heteroatom_rich"
    if "formula unavailable" in notes:
        return "formula_unavailable"
    return "unclassified"


def interval_label(value: float, edges: Sequence[float], unit: str = "") -> str:
    for low, high in zip(edges, edges[1:]):
        if low <= value < high:
            if math.isinf(low):
                return f"<{high:g}{unit}"
            if math.isinf(high):
                return f">={low:g}{unit}"
            return f"{low:g}-{high:g}{unit}"
    return "unbinned"


def grouped_summaries(
    points: Sequence[PointResult],
    key,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[PointResult]] = defaultdict(list)
    for point in points:
        groups[str(key(point))].append(point)
    return {
        label: point_summary(group)
        for label, group in sorted(groups.items())
    }


def evaluate_compound(
    resolver: PerryInputResolver,
    cas: str,
    row: dict,
    *,
    sample_points: int,
) -> tuple[CompoundResult, list[PointResult]]:
    viscosity_rows = row.get("vapor_viscosity") or []
    if len(viscosity_rows) != 1:
        raise ValueError(f"expected exactly one Perry vapor-viscosity row, got {len(viscosity_rows)}")
    correlation = viscosity_rows[0]
    critical = row.get("critical_constants") or {}
    Tc_K = float(critical["Tc_K"])
    Pc_bar = float(critical["Pc_MPa"]) * 10.0
    molecular_weight = float(
        correlation.get("molecular_weight") or critical["molecular_weight"]
    )
    Tmin_K = float(correlation["T_min_K"])
    Tmax_K = float(correlation["T_max_K"])
    if not (Tc_K > 0.0 and Pc_bar > 0.0 and molecular_weight > 0.0):
        raise ValueError("MW, Tc, and Pc must be positive")
    if not Tmax_K > Tmin_K:
        raise ValueError("Perry temperature range must have positive width")

    resolver.set_critical_inputs(Tc_K=Tc_K, Pc_bar=Pc_bar)
    props = {
        "CAS": cas,
        "name": row.get("name") or cas,
        "formula": row.get("formula") or "",
        "MW": molecular_weight,
        "Tc": Tc_K,
        "Pc": Pc_bar,
        "property_sources": {
            "MW": {
                "source": "local",
                "method": "perry_vapor_viscosity_table",
                "quality": 1.0,
            },
        },
    }

    points: list[PointResult] = []
    assigned_quality = math.nan
    class_name = "unclassified"
    for index in range(sample_points):
        position = index / (sample_points - 1)
        temperature = Tmin_K + (Tmax_K - Tmin_K) * position
        reference = PerryPropertyLibrary._eval_vapor_viscosity_Pa_s(
            correlation,
            temperature,
        )
        predicted_result = resolver._yoon_thodos_viscosity(
            cas,
            props,
            temperature,
            "vapor",
        )
        if reference is None or reference <= 0.0 or not math.isfinite(reference):
            raise ValueError(f"invalid Perry reference at {temperature:g} K")
        if predicted_result is None:
            raise ValueError(f"Yoon-Thodos returned no value at {temperature:g} K")
        predicted = float(predicted_result.value)
        if predicted <= 0.0 or not math.isfinite(predicted):
            raise ValueError(f"invalid Yoon-Thodos result at {temperature:g} K")
        if index == 0:
            assigned_quality = float(predicted_result.quality)
            class_name = chemical_class(predicted_result.notes)
        ratio = predicted / reference
        signed_error = 100.0 * (ratio - 1.0)
        points.append(
            PointResult(
                cas=cas,
                temperature_K=temperature,
                reduced_temperature=temperature / Tc_K,
                range_position=position,
                reference_Pa_s=reference,
                predicted_Pa_s=predicted,
                signed_error_percent=signed_error,
                absolute_error_percent=abs(signed_error),
                log_error=math.log(ratio),
                chemical_class=class_name,
                molecular_weight=molecular_weight,
            )
        )

    summary = point_summary(points)
    worst = max(points, key=lambda point: point.absolute_error_percent)
    compound = CompoundResult(
        cas=cas,
        name=str(row.get("name") or cas),
        formula=str(row.get("formula") or ""),
        chemical_class=class_name,
        assigned_quality=assigned_quality,
        molecular_weight=molecular_weight,
        Tc_K=Tc_K,
        Pc_bar=Pc_bar,
        Tmin_K=Tmin_K,
        Tmax_K=Tmax_K,
        Tr_min=Tmin_K / Tc_K,
        Tr_max=Tmax_K / Tc_K,
        points=len(points),
        mape_percent=float(summary["mape_percent"]),
        median_ape_percent=float(summary["median_ape_percent"]),
        p90_ape_percent=float(summary["p90_ape_percent"]),
        p95_ape_percent=float(summary["p95_ape_percent"]),
        maximum_ape_percent=float(summary["maximum_ape_percent"]),
        mean_signed_error_percent=float(summary["mean_signed_error_percent"]),
        median_signed_error_percent=float(summary["median_signed_error_percent"]),
        rms_log_error_percent=float(summary["rms_log_error_percent"]),
        geometric_bias_percent=float(summary["geometric_bias_percent"]),
        fraction_within_10_percent=float(summary["fraction_within_10_percent"]),
        fraction_within_20_percent=float(summary["fraction_within_20_percent"]),
        fraction_within_30_percent=float(summary["fraction_within_30_percent"]),
        fraction_within_50_percent=float(summary["fraction_within_50_percent"]),
        worst_temperature_K=worst.temperature_K,
    )
    return compound, points


def format_point_metrics(summary: dict[str, float | int]) -> str:
    return (
        f"n={summary['points']:6d} comps={summary['components']:3d} "
        f"MAPE={summary['mape_percent']:7.2f}% "
        f"MdAPE={summary['median_ape_percent']:7.2f}% "
        f"P90={summary['p90_ape_percent']:7.2f}% "
        f"P95={summary['p95_ape_percent']:7.2f}% "
        f"bias={summary['mean_signed_error_percent']:+7.2f}% "
        f"within20={100.0 * summary['fraction_within_20_percent']:5.1f}%"
    )


def append_group_table(
    lines: list[str],
    title: str,
    groups: dict[str, dict[str, float | int]],
) -> None:
    lines.extend(("", title))
    for label, summary in groups.items():
        lines.append(f"  {label:24s} {format_point_metrics(summary)}")


def build_report(
    *,
    database: Path,
    database_sha256: str,
    sample_points: int,
    total_records: int,
    compounds: Sequence[CompoundResult],
    points: Sequence[PointResult],
    exclusions: Counter,
    groups: dict[str, dict[str, dict[str, float | int]]],
    top: int,
) -> str:
    overall = point_summary(points)
    compound_mapes = [compound.mape_percent for compound in compounds]
    compound_summary = value_summary(compound_mapes)
    lines = [
        "Yoon-Thodos vapor-viscosity benchmark vs Perry 9th Table 2-138",
        f"database: {database}",
        f"database sha256: {database_sha256}",
        (
            f"coverage: {len(compounds)}/{total_records} compounds "
            f"({100.0 * len(compounds) / total_records:.2f}%), "
            f"{len(points)} sampled states, {sample_points} points/curve"
        ),
        "inputs: Perry MW/Tc/Pc at quality 1.0; full stated Perry temperature ranges",
        "prediction: production pfdsim Yoon-Thodos implementation; no fitted parameters",
        "",
        "OVERALL POINT-WEIGHTED",
        f"  {format_point_metrics(overall)}",
        (
            f"  P99={overall['p99_ape_percent']:.2f}% "
            f"max={overall['maximum_ape_percent']:.2f}% "
            f"median_bias={overall['median_signed_error_percent']:+.2f}% "
            f"geometric_bias={overall['geometric_bias_percent']:+.2f}% "
            f"RMS_log*100={overall['rms_log_error_percent']:.2f}"
        ),
        (
            f"  within10={100.0 * overall['fraction_within_10_percent']:.1f}% "
            f"within20={100.0 * overall['fraction_within_20_percent']:.1f}% "
            f"within30={100.0 * overall['fraction_within_30_percent']:.1f}% "
            f"within50={100.0 * overall['fraction_within_50_percent']:.1f}% "
            f"within_factor2={100.0 * overall['fraction_within_factor_two']:.1f}% "
            f"underpredicted={100.0 * overall['underprediction_fraction']:.1f}%"
        ),
        "",
        "COMPOUND-EQUAL DISTRIBUTION OF CURVE MAPE",
        (
            f"  n={compound_summary['count']} mean={compound_summary['mean']:.2f}% "
            f"median={compound_summary['median']:.2f}% "
            f"P90={compound_summary['p90']:.2f}% "
            f"P95={compound_summary['p95']:.2f}% "
            f"P99={compound_summary['p99']:.2f}% "
            f"max={compound_summary['maximum']:.2f}%"
        ),
    ]
    append_group_table(lines, "BY ASSIGNED QUALITY/COMPOSITION CLASS", groups["chemical_class"])
    append_group_table(lines, "BY REDUCED TEMPERATURE Tr", groups["reduced_temperature"])
    append_group_table(lines, "BY MOLECULAR WEIGHT", groups["molecular_weight"])
    append_group_table(lines, "BY POSITION IN PERRY TEMPERATURE RANGE", groups["range_position"])
    append_group_table(lines, "BY PERRY REFERENCE VISCOSITY", groups["reference_viscosity"])

    lines.extend(("", f"WORST {min(top, len(compounds))} COMPOUNDS BY CURVE MAPE"))
    for rank, compound in enumerate(
        sorted(compounds, key=lambda item: item.mape_percent, reverse=True)[:top],
        1,
    ):
        lines.append(
            f"  {rank:2d}. {compound.name[:31]:31s} {compound.cas:12s} "
            f"{compound.formula[:14]:14s} class={compound.chemical_class:18s} "
            f"q={compound.assigned_quality:.2f} MAPE={compound.mape_percent:8.2f}% "
            f"bias={compound.mean_signed_error_percent:+8.2f}% "
            f"max={compound.maximum_ape_percent:8.2f}%"
        )
    lines.extend(("", f"BEST {min(top, len(compounds))} COMPOUNDS BY CURVE MAPE"))
    for rank, compound in enumerate(
        sorted(compounds, key=lambda item: item.mape_percent)[:top],
        1,
    ):
        lines.append(
            f"  {rank:2d}. {compound.name[:31]:31s} {compound.cas:12s} "
            f"{compound.formula[:14]:14s} class={compound.chemical_class:18s} "
            f"q={compound.assigned_quality:.2f} MAPE={compound.mape_percent:8.2f}% "
            f"bias={compound.mean_signed_error_percent:+8.2f}% "
            f"max={compound.maximum_ape_percent:8.2f}%"
        )
    lines.extend(("", "EXCLUSIONS"))
    if exclusions:
        for reason, count in exclusions.most_common():
            lines.append(f"  {count:4d} {reason}")
    else:
        lines.append("  none")
    return "\n".join(lines)


def benchmark(args: argparse.Namespace) -> tuple[str, dict]:
    payload = json.loads(args.database.read_text(encoding="utf-8"))
    chemicals = payload["chemicals"]
    resolver = PerryInputResolver()
    compounds: list[CompoundResult] = []
    points: list[PointResult] = []
    exclusions: Counter = Counter()

    for cas, row in sorted(chemicals.items()):
        try:
            compound, compound_points = evaluate_compound(
                resolver,
                cas,
                row,
                sample_points=args.points,
            )
        except Exception as exc:
            exclusions[f"{type(exc).__name__}: {exc}"] += 1
            continue
        compounds.append(compound)
        points.extend(compound_points)

    if not compounds:
        raise RuntimeError("No Perry compounds were benchmarked successfully")

    tr_edges = (-math.inf, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, math.inf)
    mw_edges = (-math.inf, 50.0, 100.0, 150.0, 250.0, math.inf)
    position_edges = (-math.inf, 0.25, 0.50, 0.75, 1.0000001, math.inf)
    reference_uPa_edges = (-math.inf, 10.0, 20.0, 40.0, 80.0, math.inf)
    groups = {
        "chemical_class": grouped_summaries(points, lambda point: point.chemical_class),
        "reduced_temperature": grouped_summaries(
            points,
            lambda point: interval_label(point.reduced_temperature, tr_edges),
        ),
        "molecular_weight": grouped_summaries(
            points,
            lambda point: interval_label(point.molecular_weight, mw_edges, " g/mol"),
        ),
        "range_position": grouped_summaries(
            points,
            lambda point: interval_label(point.range_position, position_edges),
        ),
        "reference_viscosity": grouped_summaries(
            points,
            lambda point: interval_label(
                point.reference_Pa_s * 1.0e6,
                reference_uPa_edges,
                " uPa*s",
            ),
        ),
    }
    database_hash = sha256_path(args.database)
    overall = point_summary(points)
    compound_mape_distribution = value_summary(
        [compound.mape_percent for compound in compounds]
    )
    report_payload = {
        "schema_version": 1,
        "benchmark": "pfdsim_yoon_thodos_vs_perry_9_table_2_138",
        "source": {
            "database": str(args.database.resolve()),
            "sha256": database_hash,
            "records": len(chemicals),
        },
        "method": {
            "implementation": "PropertyResolver._yoon_thodos_viscosity",
            "inputs": "Perry MW, Tc, and Pc, each assigned quality 1.0",
            "sampling": "uniform inclusive sampling over every Perry curve's stated range",
            "points_per_curve": args.points,
        },
        "coverage": {
            "eligible_records": len(chemicals),
            "benchmarked_compounds": len(compounds),
            "sampled_states": len(points),
            "exclusions": dict(exclusions),
        },
        "overall_point_weighted": overall,
        "compound_curve_mape_distribution": compound_mape_distribution,
        "groups": groups,
        "compounds": [asdict(compound) for compound in compounds],
    }
    report = build_report(
        database=args.database,
        database_sha256=database_hash,
        sample_points=args.points,
        total_records=len(chemicals),
        compounds=compounds,
        points=points,
        exclusions=exclusions,
        groups=groups,
        top=args.top,
    )
    return report, report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--points",
        type=int,
        default=101,
        help="Uniform inclusive sample count per Perry curve (default: 101).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of best and worst compounds in the text report.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for complete machine-readable results.",
    )
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points must be at least 2")
    if args.top < 0:
        parser.error("--top cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    report, payload = benchmark(args)
    print(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
