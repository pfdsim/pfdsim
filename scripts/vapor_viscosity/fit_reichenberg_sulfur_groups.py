#!/usr/bin/env python3
"""Fit provisional Reichenberg -SH and -S- groups against Perry curves.

The fit uses relative least squares over 101 uniformly sampled points per
Perry vapor-viscosity curve.  Generalization is measured by leaving out one
whole compound at a time, refitting the shared group value, and evaluating the
held-out curve.  Sulfur is replaced by its oxygen analogue only to reuse the
strict Table 2-173 fragmenter for the remaining molecular skeleton; the oxygen
group contribution is then removed before fitting sulfur.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from rdkit import Chem
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perry_properties import PerryPropertyLibrary  # noqa: E402
import reichenberg_method as rm  # noqa: E402
from scripts.vapor_viscosity.benchmark_reichenberg_zero_dipole import (  # noqa: E402
    PerryInputResolver,
    chemical_class,
    error_metrics,
    resolve_smiles,
)


DEFAULT_DATABASE = ROOT / "data" / "perry_properties.json"


@dataclass
class FitCompound:
    cas: str
    name: str
    formula: str
    smiles: str
    sulfur_group: str
    sulfur_environment: str
    base_contribution: float
    temperatures: list[float]
    reference: list[float]
    basis: list[float]
    yoon_mape_percent: float


@dataclass(frozen=True)
class ValidationResult:
    cas: str
    name: str
    formula: str
    sulfur_group: str
    sulfur_environment: str
    base_contribution: float
    individually_optimal_contribution: float
    leave_one_out_contribution: float
    yoon_mape_percent: float
    reichenberg_full_fit_mape_percent: float
    reichenberg_leave_one_out_mape_percent: float
    leave_one_out_bias_percent: float
    mape_delta_percent_points: float


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


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


def summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "compounds": len(values),
        "mean": mean(values),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "maximum": max(values) if values else math.nan,
        "within_5_percent": mean([value <= 5.0 for value in values]),
        "within_10_percent": mean([value <= 10.0 for value in values]),
    }


def classify_sulfur(mol: Chem.Mol) -> tuple[str, str, int]:
    sulfur_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S"]
    if len(sulfur_atoms) != 1:
        raise ValueError("requires exactly one sulfur atom")
    sulfur = sulfur_atoms[0]
    if sulfur.IsInRing():
        raise ValueError("ring sulfur already has a Perry contribution")
    carbon_neighbors = [
        atom for atom in sulfur.GetNeighbors() if atom.GetSymbol() == "C"
    ]
    if (
        sulfur.GetTotalNumHs() == 1
        and sulfur.GetDegree() == 1
        and len(carbon_neighbors) == 1
    ):
        carbon = carbon_neighbors[0]
        if carbon.GetIsAromatic():
            environment = "aryl_thiol"
        elif carbon.IsInRing():
            environment = "cycloalkyl_thiol"
        elif any(neighbor.GetIsAromatic() for neighbor in carbon.GetNeighbors()):
            environment = "benzyl_thiol"
        elif carbon.GetTotalNumHs() == 1:
            environment = "secondary_thiol"
        else:
            environment = "primary_thiol"
        return "SH", environment, sulfur.GetIdx()
    if (
        sulfur.GetTotalNumHs() == 0
        and sulfur.GetDegree() == 2
        and len(carbon_neighbors) == 2
    ):
        environment = (
            "branched_thioether"
            if any(carbon.GetTotalNumHs() <= 1 for carbon in carbon_neighbors)
            else "unbranched_thioether"
        )
        return "S", environment, sulfur.GetIdx()
    raise ValueError("unsupported non-ring sulfur environment")


def oxygen_proxy_fragmentation(
    mol: Chem.Mol,
    sulfur_group: str,
    sulfur_index: int,
) -> tuple[rm.ReichenbergFragmentation, float]:
    editable = Chem.RWMol(mol)
    editable.GetAtomWithIdx(sulfur_index).SetAtomicNum(8)
    proxy = editable.GetMol()
    Chem.SanitizeMol(proxy)
    fragmentation = rm.fragment(Chem.MolToSmiles(proxy))
    oxygen_group = "oh_alcohol" if sulfur_group == "SH" else "ether_o"
    base = fragmentation.contribution_sum - rm.GROUP_VALUES[oxygen_group]
    if base <= 0.0:
        raise ValueError("nonpositive non-sulfur skeleton contribution")
    return fragmentation, base


def reichenberg_basis(
    temperature_K: float,
    molecular_weight: float,
    Tc_K: float,
) -> float:
    Tr = temperature_K / Tc_K
    return (
        1.0e-7
        * math.sqrt(molecular_weight)
        * Tc_K
        * Tr
        / (1.0 + 0.36 * Tr * (Tr - 1.0)) ** (1.0 / 6.0)
    )


def predictions(compound: FitCompound, contribution: float) -> list[float]:
    denominator = compound.base_contribution + contribution
    if denominator <= 0.0:
        return [math.inf] * len(compound.basis)
    return [value / denominator for value in compound.basis]


def fit_contribution(compounds: Sequence[FitCompound]) -> float:
    lower = -min(compound.base_contribution for compound in compounds) + 1.0e-8

    def objective(contribution: float) -> float:
        squared = []
        for compound in compounds:
            estimates = predictions(compound, contribution)
            squared.extend(
                (estimate / reference - 1.0) ** 2
                for reference, estimate in zip(compound.reference, estimates)
            )
        return mean(squared)

    result = minimize_scalar(
        objective,
        bounds=(lower, 100.0),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    if not result.success:
        raise RuntimeError(f"Sulfur contribution fit failed: {result.message}")
    return float(result.x)


def load_compounds(args: argparse.Namespace) -> list[FitCompound]:
    chemicals = json.loads(args.database.read_text(encoding="utf-8"))["chemicals"]
    resolver = PerryInputResolver()
    compounds = []

    for cas, row in sorted(chemicals.items()):
        formula = str(row.get("formula") or "")
        if "S" not in formula:
            continue
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
            continue

        resolver.set_critical_inputs(Tc_K, Pc_bar)
        props = {
            "CAS": cas,
            "name": row.get("name") or cas,
            "formula": formula,
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
            continue
        if chemical_class(yoon_results[0].notes) != "sparse_heteroatom":
            continue

        try:
            smiles = resolve_smiles(cas)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            sulfur_group, environment, sulfur_index = classify_sulfur(mol)
            _, base = oxygen_proxy_fragmentation(
                mol,
                sulfur_group,
                sulfur_index,
            )
        except (ValueError, rm.ReichenbergFragmentationError):
            continue

        yoon_values = [float(result.value) for result in yoon_results]
        yoon_mape = error_metrics(
            [float(value) for value in reference],
            yoon_values,
        )["mape"]
        compounds.append(
            FitCompound(
                cas=cas,
                name=str(row.get("name") or cas),
                formula=formula,
                smiles=smiles,
                sulfur_group=sulfur_group,
                sulfur_environment=environment,
                base_contribution=base,
                temperatures=temperatures,
                reference=[float(value) for value in reference],
                basis=[
                    reichenberg_basis(T, molecular_weight, Tc_K)
                    for T in temperatures
                ],
                yoon_mape_percent=yoon_mape,
            )
        )
    return compounds


def validate_group(compounds: Sequence[FitCompound]) -> tuple[dict, list[ValidationResult]]:
    full_contribution = fit_contribution(compounds)
    validation = []
    individually_optimal = []
    leave_one_out_contributions = []

    for index, compound in enumerate(compounds):
        individual = fit_contribution([compound])
        training = list(compounds[:index]) + list(compounds[index + 1 :])
        leave_one_out = fit_contribution(training)
        full_metrics = error_metrics(
            compound.reference,
            predictions(compound, full_contribution),
        )
        loo_metrics = error_metrics(
            compound.reference,
            predictions(compound, leave_one_out),
        )
        individually_optimal.append(individual)
        leave_one_out_contributions.append(leave_one_out)
        validation.append(
            ValidationResult(
                cas=compound.cas,
                name=compound.name,
                formula=compound.formula,
                sulfur_group=compound.sulfur_group,
                sulfur_environment=compound.sulfur_environment,
                base_contribution=compound.base_contribution,
                individually_optimal_contribution=individual,
                leave_one_out_contribution=leave_one_out,
                yoon_mape_percent=compound.yoon_mape_percent,
                reichenberg_full_fit_mape_percent=full_metrics["mape"],
                reichenberg_leave_one_out_mape_percent=loo_metrics["mape"],
                leave_one_out_bias_percent=loo_metrics["bias"],
                mape_delta_percent_points=(
                    loo_metrics["mape"] - compound.yoon_mape_percent
                ),
            )
        )

    yoon_mapes = [result.yoon_mape_percent for result in validation]
    loo_mapes = [
        result.reichenberg_leave_one_out_mape_percent
        for result in validation
    ]
    group_summary = {
        "compounds": len(compounds),
        "full_fit_contribution": full_contribution,
        "individual_optimum": {
            "minimum": min(individually_optimal),
            "median": percentile(individually_optimal, 0.50),
            "maximum": max(individually_optimal),
        },
        "leave_one_out_contribution": {
            "minimum": min(leave_one_out_contributions),
            "median": percentile(leave_one_out_contributions, 0.50),
            "maximum": max(leave_one_out_contributions),
        },
        "yoon_curve_mape": summary(yoon_mapes),
        "reichenberg_leave_one_out_curve_mape": summary(loo_mapes),
        "reichenberg_wins": sum(
            result.mape_delta_percent_points < 0.0 for result in validation
        ),
        "yoon_wins": sum(
            result.mape_delta_percent_points > 0.0 for result in validation
        ),
    }
    return group_summary, validation


def build_report(payload: dict) -> str:
    lines = [
        "Provisional Reichenberg sulfur-group fit against Perry vapor viscosity",
        f"sampling: {payload['points_per_curve']} points per complete Perry curve",
        "fit: relative least squares; validation: leave one whole compound out",
    ]
    for group_name, group in payload["groups"].items():
        yoon = group["summary"]["yoon_curve_mape"]
        reichenberg = group["summary"]["reichenberg_leave_one_out_curve_mape"]
        individual = group["summary"]["individual_optimum"]
        loo = group["summary"]["leave_one_out_contribution"]
        lines.extend(
            (
                "",
                f"GROUP {group_name}",
                (
                    f"  n={group['summary']['compounds']} full-fit C="
                    f"{group['summary']['full_fit_contribution']:.6f}"
                ),
                (
                    f"  individual optimal C: min={individual['minimum']:.3f} "
                    f"median={individual['median']:.3f} max={individual['maximum']:.3f}"
                ),
                (
                    f"  leave-one-out fitted C: min={loo['minimum']:.3f} "
                    f"median={loo['median']:.3f} max={loo['maximum']:.3f}"
                ),
                (
                    f"  Yoon: mean={yoon['mean']:.3f}% median={yoon['median']:.3f}% "
                    f"P90={yoon['p90']:.3f}% within5="
                    f"{100.0 * yoon['within_5_percent']:.1f}%"
                ),
                (
                    f"  Reichenberg LOO: mean={reichenberg['mean']:.3f}% "
                    f"median={reichenberg['median']:.3f}% P90={reichenberg['p90']:.3f}% "
                    f"within5={100.0 * reichenberg['within_5_percent']:.1f}%"
                ),
                (
                    f"  head-to-head: Reichenberg wins "
                    f"{group['summary']['reichenberg_wins']}, Yoon wins "
                    f"{group['summary']['yoon_wins']}"
                ),
                "  compounds:",
            )
        )
        for result in group["compounds"]:
            lines.append(
                f"    {result['name'][:28]:28s} {result['sulfur_environment']:20s} "
                f"Copt={result['individually_optimal_contribution']:7.3f} "
                f"Cloo={result['leave_one_out_contribution']:7.3f} "
                f"Y={result['yoon_mape_percent']:6.2f}% "
                f"Rloo={result['reichenberg_leave_one_out_mape_percent']:6.2f}% "
                f"delta={result['mape_delta_percent_points']:+7.2f}"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    compounds = load_compounds(args)
    grouped: dict[str, list[FitCompound]] = defaultdict(list)
    for compound in compounds:
        grouped[compound.sulfur_group].append(compound)
    payload = {
        "schema_version": 1,
        "benchmark": "provisional_reichenberg_SH_and_S_group_fit",
        "database": str(args.database.resolve()),
        "points_per_curve": args.points,
        "objective": "pointwise relative least squares",
        "validation": "leave one complete compound out",
        "groups": {},
    }
    for group_name, group_compounds in sorted(grouped.items()):
        group_summary, validation = validate_group(group_compounds)
        payload["groups"][group_name] = {
            "summary": group_summary,
            "compounds": [asdict(result) for result in validation],
        }
    print(build_report(payload))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
