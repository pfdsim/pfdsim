#!/usr/bin/env python3
"""Compare zero-dipole Reichenberg with Yoon-Thodos on Perry vapor viscosity.

The paired benchmark targets the two pfdsim classes for which Yoon-Thodos is
currently assigned quality 0.88: hydrocarbons and sparse-heteroatom compounds.
Perry supplies MW/Tc/Pc and the reference curves.  Molecular structures come
from the locally installed ``chemicals`` identifier database; no network data
or fitted parameters are used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from chemicals.identifiers import search_chemical
from rdkit import Chem
from rdkit.Chem import Descriptors


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perry_properties import PerryPropertyLibrary  # noqa: E402
from property_resolver import PropertyResolutionResult, PropertyResolver  # noqa: E402
import reichenberg_method as rm  # noqa: E402


DEFAULT_DATABASE = ROOT / "data" / "perry_properties.json"
TARGET_CLASSES = {"hydrocarbon", "sparse_heteroatom"}


@dataclass(frozen=True)
class CompoundResult:
    cas: str
    name: str
    formula: str
    chemical_class: str
    smiles: str
    groups: dict[str, int]
    contribution_sum: float
    Tmin_K: float
    Tmax_K: float
    Tr_min: float
    Tr_max: float
    yoon_mape_percent: float
    yoon_bias_percent: float
    yoon_p95_ape_percent: float
    yoon_max_ape_percent: float
    reichenberg_mape_percent: float
    reichenberg_bias_percent: float
    reichenberg_p95_ape_percent: float
    reichenberg_max_ape_percent: float
    mape_delta_percent_points: float


class PerryInputResolver(PropertyResolver):
    def __init__(self) -> None:
        super().__init__()
        self._critical: dict[str, PropertyResolutionResult] = {}

    def set_critical_inputs(self, Tc_K: float, Pc_bar: float) -> None:
        self._critical = {
            "Tc": PropertyResolutionResult(
                Tc_K, "local", "perry_critical_constant", 1.0, "units K"
            ),
            "Pc": PropertyResolutionResult(
                Pc_bar, "local", "perry_critical_constant", 1.0, "units bar"
            ),
        }

    def resolve_critical_properties(self, *args, **kwargs):
        return self._critical


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


def error_metrics(
    reference: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, float]:
    signed = [
        100.0 * (estimate / actual - 1.0)
        for actual, estimate in zip(reference, predicted)
    ]
    absolute = [abs(value) for value in signed]
    return {
        "mape": mean(absolute),
        "bias": mean(signed),
        "p95": percentile(absolute, 0.95),
        "maximum": max(absolute),
    }


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "components": len(values),
        "mean": mean(values),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "maximum": max(values) if values else math.nan,
        "within_5_percent": mean([value <= 5.0 for value in values]),
        "within_10_percent": mean([value <= 10.0 for value in values]),
        "within_20_percent": mean([value <= 20.0 for value in values]),
    }


def signed_distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "components": len(values),
        "mean": mean(values),
        "median": percentile(values, 0.50),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "minimum": min(values) if values else math.nan,
        "maximum": max(values) if values else math.nan,
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


def rejection_category(message: str) -> str:
    prefixes = (
        "element ",
        "unsupported carbon environment",
        "unsupported nitrogen environment",
        "unsupported oxygen environment",
        "non-ring or non-divalent sulfur",
        "nitrile without attached carbon skeleton",
        "charged or radical",
        "salts and disconnected",
    )
    for prefix in prefixes:
        if message.startswith(prefix):
            return prefix.rstrip()
    return message


def resolve_smiles(cas: str) -> str:
    try:
        metadata = search_chemical(cas)
    except Exception as exc:
        raise ValueError("SMILES unavailable") from exc
    smiles = getattr(metadata, "smiles", None)
    if not smiles:
        raise ValueError("SMILES unavailable")
    return str(smiles)


def benchmark(args: argparse.Namespace) -> tuple[dict, str]:
    chemicals = json.loads(args.database.read_text(encoding="utf-8"))["chemicals"]
    resolver = PerryInputResolver()
    eligible: Counter = Counter()
    covered: Counter = Counter()
    rejects: Counter = Counter()
    rejected_records = []
    results: list[CompoundResult] = []
    full_target_yoon_mapes: dict[str, list[float]] = defaultdict(list)

    for cas, row in sorted(chemicals.items()):
        critical = row["critical_constants"]
        correlation = row["vapor_viscosity"][0]
        Tc_K = float(critical["Tc_K"])
        Pc_bar = 10.0 * float(critical["Pc_MPa"])
        molecular_weight = float(critical["molecular_weight"])
        Tmin_K = float(correlation["T_min_K"])
        Tmax_K = float(correlation["T_max_K"])
        temperatures = [
            Tmin_K + (Tmax_K - Tmin_K) * index / (args.points - 1)
            for index in range(args.points)
        ]
        reference = [
            PerryPropertyLibrary._eval_vapor_viscosity_Pa_s(correlation, T)
            for T in temperatures
        ]
        if any(value is None or value <= 0.0 for value in reference):
            raise ValueError(f"Invalid Perry reference curve for {cas}")
        reference = [float(value) for value in reference]

        resolver.set_critical_inputs(Tc_K, Pc_bar)
        props = {
            "CAS": cas,
            "name": row.get("name") or cas,
            "formula": row.get("formula") or "",
            "MW": molecular_weight,
            "property_sources": {
                "MW": {
                    "source": "local",
                    "method": "perry_vapor_viscosity_table",
                    "quality": 1.0,
                }
            },
        }
        yoon_results = [
            resolver._yoon_thodos_viscosity(cas, props, T, "vapor")
            for T in temperatures
        ]
        if any(result is None for result in yoon_results):
            raise ValueError(f"Yoon-Thodos failed for {cas}")
        class_name = chemical_class(yoon_results[0].notes)
        if class_name not in TARGET_CLASSES:
            continue
        eligible[class_name] += 1
        yoon_values = [float(result.value) for result in yoon_results]
        yoon_metrics = error_metrics(reference, yoon_values)
        full_target_yoon_mapes[class_name].append(yoon_metrics["mape"])

        try:
            smiles = resolve_smiles(cas)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("SMILES unavailable")
            structure_mw = Descriptors.MolWt(mol)
            if abs(structure_mw / molecular_weight - 1.0) > 0.02:
                raise ValueError(
                    f"SMILES/Perry molecular-weight mismatch "
                    f"({structure_mw:g} vs {molecular_weight:g})"
                )
            fragmentation = rm.fragment(smiles)
            reichenberg_values = [
                rm.viscosity_Pa_s(
                    T,
                    molecular_weight,
                    Tc_K,
                    Pc_bar,
                    dipole_D=0.0,
                    fragmentation=fragmentation,
                )
                for T in temperatures
            ]
        except (ValueError, rm.ReichenbergFragmentationError) as exc:
            reason = rejection_category(str(exc))
            rejects[reason] += 1
            rejected_records.append(
                {
                    "cas": cas,
                    "name": row.get("name") or cas,
                    "formula": row.get("formula") or "",
                    "chemical_class": class_name,
                    "reason": str(exc),
                    "yoon_mape_percent": yoon_metrics["mape"],
                }
            )
            continue

        covered[class_name] += 1
        reichenberg_metrics = error_metrics(reference, reichenberg_values)
        results.append(
            CompoundResult(
                cas=cas,
                name=str(row.get("name") or cas),
                formula=str(row.get("formula") or ""),
                chemical_class=class_name,
                smiles=smiles,
                groups=dict(sorted(fragmentation.groups.items())),
                contribution_sum=fragmentation.contribution_sum,
                Tmin_K=Tmin_K,
                Tmax_K=Tmax_K,
                Tr_min=Tmin_K / Tc_K,
                Tr_max=Tmax_K / Tc_K,
                yoon_mape_percent=yoon_metrics["mape"],
                yoon_bias_percent=yoon_metrics["bias"],
                yoon_p95_ape_percent=yoon_metrics["p95"],
                yoon_max_ape_percent=yoon_metrics["maximum"],
                reichenberg_mape_percent=reichenberg_metrics["mape"],
                reichenberg_bias_percent=reichenberg_metrics["bias"],
                reichenberg_p95_ape_percent=reichenberg_metrics["p95"],
                reichenberg_max_ape_percent=reichenberg_metrics["maximum"],
                mape_delta_percent_points=(
                    reichenberg_metrics["mape"] - yoon_metrics["mape"]
                ),
            )
        )

    paired = {}
    for class_name in sorted(TARGET_CLASSES | {"all"}):
        subset = (
            results
            if class_name == "all"
            else [result for result in results if result.chemical_class == class_name]
        )
        yoon_mapes = [result.yoon_mape_percent for result in subset]
        reichenberg_mapes = [result.reichenberg_mape_percent for result in subset]
        deltas = [result.mape_delta_percent_points for result in subset]
        paired[class_name] = {
            "yoon_curve_mape_distribution": distribution(yoon_mapes),
            "reichenberg_curve_mape_distribution": distribution(reichenberg_mapes),
            "mape_delta_distribution": signed_distribution(deltas),
            "reichenberg_wins": sum(delta < -1.0e-12 for delta in deltas),
            "yoon_wins": sum(delta > 1.0e-12 for delta in deltas),
            "within_1_percentage_point": sum(abs(delta) <= 1.0 for delta in deltas),
        }

    full_yoon = {
        class_name: distribution(values)
        for class_name, values in sorted(full_target_yoon_mapes.items())
    }
    full_yoon["all"] = distribution(
        [value for values in full_target_yoon_mapes.values() for value in values]
    )
    payload = {
        "schema_version": 1,
        "benchmark": "zero_dipole_reichenberg_vs_yoon_thodos_on_perry",
        "database": str(args.database.resolve()),
        "points_per_curve": args.points,
        "target_classes": sorted(TARGET_CLASSES),
        "coverage": {
            "eligible": dict(eligible),
            "covered": dict(covered),
            "eligible_total": sum(eligible.values()),
            "covered_total": sum(covered.values()),
            "rejection_reasons": dict(rejects),
        },
        "full_target_yoon": full_yoon,
        "paired": paired,
        "compounds": [asdict(result) for result in results],
        "rejected_compounds": rejected_records,
    }
    report = build_report(payload, results, args.top)
    return payload, report


def format_distribution(summary: dict) -> str:
    return (
        f"n={summary['components']:3d} mean={summary['mean']:6.2f}% "
        f"median={summary['median']:6.2f}% P90={summary['p90']:6.2f}% "
        f"P95={summary['p95']:6.2f}% within5="
        f"{100.0 * summary['within_5_percent']:5.1f}% within10="
        f"{100.0 * summary['within_10_percent']:5.1f}%"
    )


def build_report(payload: dict, results: Sequence[CompoundResult], top: int) -> str:
    coverage = payload["coverage"]
    lines = [
        "Zero-dipole Reichenberg vs Yoon-Thodos on Perry vapor viscosity",
        f"sampling: {payload['points_per_curve']} points over each complete Perry range",
        (
            f"target coverage: {coverage['covered_total']}/{coverage['eligible_total']} "
            f"({100.0 * coverage['covered_total'] / coverage['eligible_total']:.1f}%)"
        ),
    ]
    for class_name in sorted(TARGET_CLASSES):
        eligible = coverage["eligible"].get(class_name, 0)
        covered = coverage["covered"].get(class_name, 0)
        lines.append(
            f"  {class_name}: {covered}/{eligible} "
            f"({100.0 * covered / eligible:.1f}%)"
        )
    lines.extend(("", "FULL TARGET Yoon-Thodos curve-MAPE distribution"))
    for class_name, summary in payload["full_target_yoon"].items():
        lines.append(f"  {class_name:20s} {format_distribution(summary)}")
    lines.extend(("", "PAIRED COMPARISON ON SUCCESSFULLY FRAGMENTED COMPOUNDS"))
    for class_name, comparison in payload["paired"].items():
        lines.append(f"  {class_name}")
        lines.append(
            f"    Yoon-Thodos {format_distribution(comparison['yoon_curve_mape_distribution'])}"
        )
        lines.append(
            f"    Reichenberg  {format_distribution(comparison['reichenberg_curve_mape_distribution'])}"
        )
        lines.append(
            f"    head-to-head: Reichenberg wins {comparison['reichenberg_wins']}, "
            f"Yoon wins {comparison['yoon_wins']}, within 1 point "
            f"{comparison['within_1_percentage_point']}"
        )
    lines.extend(("", f"BIGGEST {top} REICHENBERG IMPROVEMENTS (curve-MAPE points)"))
    for result in sorted(results, key=lambda item: item.mape_delta_percent_points)[:top]:
        lines.append(
            f"  {result.name[:31]:31s} {result.chemical_class:18s} "
            f"Y={result.yoon_mape_percent:6.2f}% R={result.reichenberg_mape_percent:6.2f}% "
            f"delta={result.mape_delta_percent_points:+7.2f}"
        )
    lines.extend(("", f"BIGGEST {top} REICHENBERG REGRESSIONS (curve-MAPE points)"))
    for result in sorted(
        results,
        key=lambda item: item.mape_delta_percent_points,
        reverse=True,
    )[:top]:
        lines.append(
            f"  {result.name[:31]:31s} {result.chemical_class:18s} "
            f"Y={result.yoon_mape_percent:6.2f}% R={result.reichenberg_mape_percent:6.2f}% "
            f"delta={result.mape_delta_percent_points:+7.2f}"
        )
    lines.extend(("", "REJECTION REASONS"))
    for reason, count in sorted(
        payload["coverage"]["rejection_reasons"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"  {count:3d} {reason}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points must be at least 2")
    if args.top < 0:
        parser.error("--top cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    payload, report = benchmark(args)
    print(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
