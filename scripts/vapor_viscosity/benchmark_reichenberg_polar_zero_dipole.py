#!/usr/bin/env python3
"""Compare zero-dipole Reichenberg with Yoon's heteroatom-rich Perry subset.

Water, ammonia, hydrogen fluoride, and hydrogen cyanide are explicitly
excluded. Carbon-containing structures are first passed through the strict
Reichenberg fragmenter. A failed structure uses Perry's inorganic prefactor
only when it has no C-H bond; otherwise it remains an explicit rejection.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import Descriptors


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perry_properties import PerryPropertyLibrary  # noqa: E402
import reichenberg_method as rm  # noqa: E402
from scripts.vapor_viscosity.benchmark_reichenberg_zero_dipole import (  # noqa: E402
    PerryInputResolver,
    chemical_class,
    distribution,
    error_metrics,
    format_distribution,
    rejection_category,
    resolve_smiles,
    signed_distribution,
)


DEFAULT_DATABASE = ROOT / "data" / "perry_properties.json"
EXCLUDED_CAS = {
    "7732-18-5": "water",
    "7664-41-7": "ammonia",
    "7664-39-3": "hydrogen fluoride",
    "74-90-8": "hydrogen cyanide",
}


@dataclass(frozen=True)
class CompoundResult:
    cas: str
    name: str
    formula: str
    branch: str
    smiles: str
    groups: dict[str, int]
    contribution_sum: float | None
    yoon_mape_percent: float
    yoon_bias_percent: float
    yoon_p95_ape_percent: float
    yoon_max_ape_percent: float
    reichenberg_mape_percent: float
    reichenberg_bias_percent: float
    reichenberg_p95_ape_percent: float
    reichenberg_max_ape_percent: float
    mape_delta_percent_points: float


def has_carbon_hydrogen_bond(mol: Chem.Mol) -> bool:
    return any(
        atom.GetSymbol() == "C" and atom.GetTotalNumHs() > 0
        for atom in mol.GetAtoms()
    )


def paired_summary(results: Sequence[CompoundResult]) -> dict:
    yoon = [result.yoon_mape_percent for result in results]
    reichenberg = [result.reichenberg_mape_percent for result in results]
    deltas = [result.mape_delta_percent_points for result in results]
    return {
        "yoon_curve_mape_distribution": distribution(yoon),
        "reichenberg_curve_mape_distribution": distribution(reichenberg),
        "mape_delta_distribution": signed_distribution(deltas),
        "reichenberg_wins": sum(delta < -1.0e-12 for delta in deltas),
        "yoon_wins": sum(delta > 1.0e-12 for delta in deltas),
        "within_1_percentage_point": sum(abs(delta) <= 1.0 for delta in deltas),
    }


def benchmark(args: argparse.Namespace) -> tuple[dict, str]:
    chemicals = json.loads(args.database.read_text(encoding="utf-8"))["chemicals"]
    resolver = PerryInputResolver()
    eligible = 0
    full_yoon_mapes = []
    results: list[CompoundResult] = []
    rejects: Counter = Counter()
    rejected_records = []

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
        if chemical_class(yoon_results[0].notes) != "heteroatom_rich":
            continue
        if cas in EXCLUDED_CAS:
            continue
        eligible += 1
        yoon_values = [float(result.value) for result in yoon_results]
        yoon_metrics = error_metrics(reference, yoon_values)
        full_yoon_mapes.append(yoon_metrics["mape"])

        smiles = ""
        branch = ""
        fragmentation = None
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
            try:
                fragmentation = rm.fragment(smiles)
                branch = "organic"
            except rm.ReichenbergFragmentationError:
                if has_carbon_hydrogen_bond(mol):
                    raise
                branch = "inorganic"

            reichenberg_values = [
                rm.viscosity_Pa_s(
                    T,
                    molecular_weight,
                    Tc_K,
                    Pc_bar,
                    dipole_D=0.0,
                    fragmentation=fragmentation,
                    inorganic=branch == "inorganic",
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
                    "reason": str(exc),
                    "yoon_mape_percent": yoon_metrics["mape"],
                }
            )
            continue

        reichenberg_metrics = error_metrics(reference, reichenberg_values)
        results.append(
            CompoundResult(
                cas=cas,
                name=str(row.get("name") or cas),
                formula=str(row.get("formula") or ""),
                branch=branch,
                smiles=smiles,
                groups=(
                    dict(sorted(fragmentation.groups.items()))
                    if fragmentation is not None
                    else {"inorganic_prefactor": 1}
                ),
                contribution_sum=(
                    fragmentation.contribution_sum
                    if fragmentation is not None
                    else None
                ),
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

    by_branch = {
        branch: paired_summary(
            [result for result in results if result.branch == branch]
        )
        for branch in ("organic", "inorganic")
        if any(result.branch == branch for result in results)
    }
    payload = {
        "schema_version": 1,
        "benchmark": "zero_dipole_reichenberg_vs_yoon_heteroatom_rich_perry",
        "database": str(args.database.resolve()),
        "points_per_curve": args.points,
        "excluded_cas": EXCLUDED_CAS,
        "coverage": {
            "eligible_after_exclusions": eligible,
            "covered": len(results),
            "organic": sum(result.branch == "organic" for result in results),
            "inorganic": sum(result.branch == "inorganic" for result in results),
            "rejection_reasons": dict(rejects),
        },
        "full_subset_yoon_curve_mape_distribution": distribution(full_yoon_mapes),
        "paired": paired_summary(results),
        "by_branch": by_branch,
        "compounds": [asdict(result) for result in results],
        "rejected_compounds": rejected_records,
    }
    return payload, build_report(payload, results, args.top)


def build_report(payload: dict, results: Sequence[CompoundResult], top: int) -> str:
    coverage = payload["coverage"]
    paired = payload["paired"]
    lines = [
        "Zero-dipole Reichenberg vs Yoon heteroatom-rich Perry subset",
        f"sampling: {payload['points_per_curve']} points over each complete Perry range",
        "excluded: water, ammonia, hydrogen fluoride, hydrogen cyanide",
        (
            f"coverage: {coverage['covered']}/{coverage['eligible_after_exclusions']} "
            f"({100.0 * coverage['covered'] / coverage['eligible_after_exclusions']:.1f}%), "
            f"organic={coverage['organic']}, inorganic={coverage['inorganic']}"
        ),
        "",
        "FULL SUBSET Yoon-Thodos",
        f"  {format_distribution(payload['full_subset_yoon_curve_mape_distribution'])}",
        "",
        "PAIRED COMPARISON",
        f"  Yoon-Thodos {format_distribution(paired['yoon_curve_mape_distribution'])}",
        f"  Reichenberg  {format_distribution(paired['reichenberg_curve_mape_distribution'])}",
        (
            f"  head-to-head: Reichenberg wins {paired['reichenberg_wins']}, "
            f"Yoon wins {paired['yoon_wins']}, within 1 point "
            f"{paired['within_1_percentage_point']}"
        ),
    ]
    for branch, comparison in payload["by_branch"].items():
        lines.extend(
            (
                "",
                f"BRANCH {branch}",
                f"  Yoon-Thodos {format_distribution(comparison['yoon_curve_mape_distribution'])}",
                f"  Reichenberg  {format_distribution(comparison['reichenberg_curve_mape_distribution'])}",
                (
                    f"  head-to-head: Reichenberg wins {comparison['reichenberg_wins']}, "
                    f"Yoon wins {comparison['yoon_wins']}"
                ),
            )
        )
    lines.extend(("", f"BIGGEST {top} REICHENBERG IMPROVEMENTS"))
    for result in sorted(results, key=lambda item: item.mape_delta_percent_points)[:top]:
        lines.append(
            f"  {result.name[:31]:31s} {result.branch:9s} "
            f"Y={result.yoon_mape_percent:7.2f}% R={result.reichenberg_mape_percent:7.2f}% "
            f"delta={result.mape_delta_percent_points:+8.2f}"
        )
    lines.extend(("", f"BIGGEST {top} REICHENBERG REGRESSIONS"))
    for result in sorted(
        results,
        key=lambda item: item.mape_delta_percent_points,
        reverse=True,
    )[:top]:
        lines.append(
            f"  {result.name[:31]:31s} {result.branch:9s} "
            f"Y={result.yoon_mape_percent:7.2f}% R={result.reichenberg_mape_percent:7.2f}% "
            f"delta={result.mape_delta_percent_points:+8.2f}"
        )
    lines.extend(("", "REJECTION REASONS"))
    for reason, count in sorted(
        coverage["rejection_reasons"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"  {count:3d} {reason}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--top", type=int, default=20)
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
