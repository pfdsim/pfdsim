#!/usr/bin/env python3
"""Test Reichenberg aldehyde-group refits against Perry vapor viscosity.

The published Perry Table 2-173 aldehyde contribution is compared with:

* one refitted shared aldehyde contribution; and
* an exploratory affine contribution versus the number of methylene groups.

Both alternatives are validated by leaving out one complete aldehyde curve,
refitting on the remaining homologs, and evaluating the held-out curve.  The
affine model diagnoses a chain-length interaction; it is not proposed as a
general Reichenberg group contribution without independent validation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.optimize import minimize, minimize_scalar


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reichenberg_method as rm  # noqa: E402
from perry_properties import PerryPropertyLibrary  # noqa: E402
from scripts.vapor_viscosity.benchmark_reichenberg_zero_dipole import (  # noqa: E402
    DEFAULT_DATABASE,
    PerryInputResolver,
    distribution,
    error_metrics,
    resolve_smiles,
)


PUBLISHED_ALDEHYDE_CONTRIBUTION = rm.GROUP_VALUES["aldehyde"]
AFFINE_REFERENCE_METHYLENES = 2


@dataclass(frozen=True)
class FitCompound:
    cas: str
    name: str
    formula: str
    smiles: str
    methylene_groups: int
    base_contribution: float
    temperatures_K: np.ndarray
    reference_viscosity_Pa_s: np.ndarray
    reichenberg_basis: np.ndarray
    yoon_mape_percent: float


@dataclass(frozen=True)
class ValidationResult:
    cas: str
    name: str
    formula: str
    methylene_groups: int
    individually_optimal_contribution: float
    constant_leave_one_out_contribution: float
    affine_leave_one_out_intercept: float
    affine_leave_one_out_methylene_slope: float
    published_mape_percent: float
    published_bias_percent: float
    constant_leave_one_out_mape_percent: float
    constant_leave_one_out_bias_percent: float
    affine_leave_one_out_mape_percent: float
    affine_leave_one_out_bias_percent: float
    yoon_mape_percent: float


def mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else math.nan


def basis_values(
    temperatures_K: np.ndarray,
    molecular_weight: float,
    critical_temperature_K: float,
) -> np.ndarray:
    reduced_temperature = temperatures_K / critical_temperature_K
    return (
        1.0e-7
        * math.sqrt(molecular_weight)
        * critical_temperature_K
        * reduced_temperature
        / (
            1.0
            + 0.36 * reduced_temperature * (reduced_temperature - 1.0)
        ) ** (1.0 / 6.0)
    )


def effective_contribution(
    compound: FitCompound,
    intercept: float,
    methylene_slope: float = 0.0,
) -> float:
    return (
        compound.base_contribution
        + float(intercept)
        + float(methylene_slope)
        * (compound.methylene_groups - AFFINE_REFERENCE_METHYLENES)
    )


def predictions(
    compound: FitCompound,
    intercept: float,
    methylene_slope: float = 0.0,
) -> np.ndarray:
    denominator = effective_contribution(compound, intercept, methylene_slope)
    if denominator <= 0.0 or not math.isfinite(denominator):
        return np.full_like(compound.reichenberg_basis, math.inf)
    return compound.reichenberg_basis / denominator


def relative_residuals(
    compounds: Sequence[FitCompound],
    intercept: float,
    methylene_slope: float = 0.0,
) -> np.ndarray:
    return np.concatenate([
        predictions(compound, intercept, methylene_slope)
        / compound.reference_viscosity_Pa_s
        - 1.0
        for compound in compounds
    ])


def fit_constant(compounds: Sequence[FitCompound]) -> float:
    lower = -min(compound.base_contribution for compound in compounds) + 1.0e-8

    def objective(contribution: float) -> float:
        residuals = relative_residuals(compounds, contribution)
        return float(np.mean(residuals**2))

    fit = minimize_scalar(
        objective,
        bounds=(lower, 50.0),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    if not fit.success:
        raise RuntimeError(f"constant aldehyde fit failed: {fit.message}")
    return float(fit.x)


def fit_affine(compounds: Sequence[FitCompound]) -> tuple[float, float]:
    minimum_base = min(compound.base_contribution for compound in compounds)

    def objective(parameters: np.ndarray) -> float:
        intercept, slope = (float(value) for value in parameters)
        denominators = [
            effective_contribution(compound, intercept, slope)
            for compound in compounds
        ]
        if any(value <= 0.0 or not math.isfinite(value) for value in denominators):
            return 1.0e12 + sum(max(0.0, -value) ** 2 for value in denominators)
        residuals = relative_residuals(compounds, intercept, slope)
        return float(np.mean(residuals**2))

    fit = minimize(
        objective,
        x0=np.asarray([PUBLISHED_ALDEHYDE_CONTRIBUTION, -1.0]),
        method="L-BFGS-B",
        bounds=(((-minimum_base + 1.0e-8), 50.0), (-5.0, 5.0)),
        options={"ftol": 1.0e-15, "gtol": 1.0e-12, "maxiter": 10000},
    )
    if not fit.success:
        raise RuntimeError(f"affine aldehyde fit failed: {fit.message}")
    return float(fit.x[0]), float(fit.x[1])


def curve_metrics(
    compound: FitCompound,
    intercept: float,
    slope: float = 0.0,
) -> dict[str, float]:
    return error_metrics(
        compound.reference_viscosity_Pa_s,
        predictions(compound, intercept, slope),
    )


def load_compounds(args: argparse.Namespace) -> list[FitCompound]:
    database = json.loads(args.database.read_text(encoding="utf-8"))["chemicals"]
    resolver = PerryInputResolver()
    compounds = []
    for cas, row in sorted(database.items()):
        try:
            smiles = resolve_smiles(cas)
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            fragmentation = rm.fragment(smiles)
        except (ValueError, rm.ReichenbergFragmentationError):
            continue
        groups = fragmentation.groups
        if not (
            groups.get("aldehyde") == 1
            and groups.get("ch3") == 1
            and set(groups) <= {"aldehyde", "ch2", "ch3"}
        ):
            continue

        critical = row["critical_constants"]
        correlation = row["vapor_viscosity"][0]
        molecular_weight = float(critical["molecular_weight"])
        structure_weight = float(Descriptors.MolWt(molecule))
        if abs(structure_weight / molecular_weight - 1.0) > 0.02:
            continue
        critical_temperature = float(critical["Tc_K"])
        critical_pressure_bar = 10.0 * float(critical["Pc_MPa"])
        temperatures = np.linspace(
            float(correlation["T_min_K"]),
            float(correlation["T_max_K"]),
            args.points,
        )
        reference = np.asarray([
            PerryPropertyLibrary._eval_vapor_viscosity_Pa_s(
                correlation,
                float(temperature),
            )
            for temperature in temperatures
        ])
        if np.any(~np.isfinite(reference)) or np.any(reference <= 0.0):
            continue

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
        resolver.set_critical_inputs(critical_temperature, critical_pressure_bar)
        yoon = [
            resolver._yoon_thodos_viscosity(cas, props, float(temperature), "vapor")
            for temperature in temperatures
        ]
        if any(result is None for result in yoon):
            continue
        yoon_mape = error_metrics(
            reference,
            [float(result.value) for result in yoon],
        )["mape"]
        compounds.append(
            FitCompound(
                cas=cas,
                name=str(row.get("name") or cas),
                formula=str(row.get("formula") or ""),
                smiles=smiles,
                methylene_groups=int(groups.get("ch2", 0)),
                base_contribution=(
                    fragmentation.contribution_sum
                    - PUBLISHED_ALDEHYDE_CONTRIBUTION
                ),
                temperatures_K=temperatures,
                reference_viscosity_Pa_s=reference,
                reichenberg_basis=basis_values(
                    temperatures,
                    molecular_weight,
                    critical_temperature,
                ),
                yoon_mape_percent=yoon_mape,
            )
        )
    return sorted(compounds, key=lambda compound: compound.methylene_groups)


def validate(compounds: Sequence[FitCompound]) -> tuple[dict, list[ValidationResult]]:
    full_constant = fit_constant(compounds)
    full_affine_intercept, full_affine_slope = fit_affine(compounds)
    results = []
    for index, compound in enumerate(compounds):
        training = list(compounds[:index]) + list(compounds[index + 1 :])
        individual = fit_constant([compound])
        constant_loo = fit_constant(training)
        affine_loo_intercept, affine_loo_slope = fit_affine(training)
        published = curve_metrics(compound, PUBLISHED_ALDEHYDE_CONTRIBUTION)
        constant = curve_metrics(compound, constant_loo)
        affine = curve_metrics(compound, affine_loo_intercept, affine_loo_slope)
        results.append(
            ValidationResult(
                cas=compound.cas,
                name=compound.name,
                formula=compound.formula,
                methylene_groups=compound.methylene_groups,
                individually_optimal_contribution=individual,
                constant_leave_one_out_contribution=constant_loo,
                affine_leave_one_out_intercept=affine_loo_intercept,
                affine_leave_one_out_methylene_slope=affine_loo_slope,
                published_mape_percent=published["mape"],
                published_bias_percent=published["bias"],
                constant_leave_one_out_mape_percent=constant["mape"],
                constant_leave_one_out_bias_percent=constant["bias"],
                affine_leave_one_out_mape_percent=affine["mape"],
                affine_leave_one_out_bias_percent=affine["bias"],
                yoon_mape_percent=compound.yoon_mape_percent,
            )
        )

    def metric_summary(field: str) -> dict:
        return distribution([float(getattr(result, field)) for result in results])

    summary = {
        "compounds": len(compounds),
        "published_contribution": PUBLISHED_ALDEHYDE_CONTRIBUTION,
        "full_constant_fit": full_constant,
        "full_affine_fit": {
            "intercept_at_two_methylenes": full_affine_intercept,
            "per_additional_methylene": full_affine_slope,
        },
        "individual_optimal_contribution": distribution([
            result.individually_optimal_contribution for result in results
        ]),
        "published_mape": metric_summary("published_mape_percent"),
        "constant_leave_one_out_mape": metric_summary(
            "constant_leave_one_out_mape_percent"
        ),
        "affine_leave_one_out_mape": metric_summary(
            "affine_leave_one_out_mape_percent"
        ),
        "yoon_mape": metric_summary("yoon_mape_percent"),
        "constant_improves_published": int(sum(
            result.constant_leave_one_out_mape_percent
            < result.published_mape_percent
            for result in results
        )),
        "affine_improves_published": int(sum(
            result.affine_leave_one_out_mape_percent
            < result.published_mape_percent
            for result in results
        )),
        "constant_beats_yoon": int(sum(
            result.constant_leave_one_out_mape_percent < result.yoon_mape_percent
            for result in results
        )),
        "affine_beats_yoon": int(sum(
            result.affine_leave_one_out_mape_percent < result.yoon_mape_percent
            for result in results
        )),
    }
    return summary, results


def build_report(payload: dict) -> str:
    summary = payload["summary"]
    affine = summary["full_affine_fit"]
    lines = [
        "Reichenberg aldehyde-group fit against Perry vapor viscosity",
        f"sampling: {payload['points_per_curve']} points per complete Perry curve",
        "objective: pointwise relative least squares",
        "validation: leave one complete aldehyde curve out",
        f"homologs: {summary['compounds']}",
        f"published aldehyde contribution: {summary['published_contribution']:.6f}",
        f"full shared-constant fit: {summary['full_constant_fit']:.6f}",
        (
            "full affine fit: effective aldehyde contribution = "
            f"{affine['intercept_at_two_methylenes']:.6f} "
            f"{affine['per_additional_methylene']:+.6f}*(n_CH2-2)"
        ),
        "",
        "CURVE-MAPE DISTRIBUTIONS",
    ]
    for label, key in (
        ("published", "published_mape"),
        ("shared constant LOO", "constant_leave_one_out_mape"),
        ("affine LOO", "affine_leave_one_out_mape"),
        ("Yoon-Thodos", "yoon_mape"),
    ):
        item = summary[key]
        lines.append(
            f"  {label:20s} mean={item['mean']:6.3f}% "
            f"median={item['median']:6.3f}% P90={item['p90']:6.3f}% "
            f"max={item['maximum']:6.3f}%"
        )
    lines.extend((
        "",
        (
            f"shared constant improves published: "
            f"{summary['constant_improves_published']}/{summary['compounds']}; "
            f"beats Yoon: {summary['constant_beats_yoon']}/{summary['compounds']}"
        ),
        (
            f"affine improves published: "
            f"{summary['affine_improves_published']}/{summary['compounds']}; "
            f"beats Yoon: {summary['affine_beats_yoon']}/{summary['compounds']}"
        ),
        "",
        "COMPOUNDS",
    ))
    for result in payload["compounds"]:
        lines.append(
            f"  {result['name'][:18]:18s} nCH2={result['methylene_groups']} "
            f"Copt={result['individually_optimal_contribution']:7.3f} "
            f"published={result['published_mape_percent']:6.3f}% "
            f"constantLOO={result['constant_leave_one_out_mape_percent']:6.3f}% "
            f"affineLOO={result['affine_leave_one_out_mape_percent']:6.3f}% "
            f"Yoon={result['yoon_mape_percent']:6.3f}%"
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
    if len(compounds) < 3:
        raise RuntimeError("at least three aldehyde homologs are required")
    summary, validation = validate(compounds)
    payload = {
        "schema_version": 1,
        "benchmark": "reichenberg_aldehyde_group_fit",
        "database": str(args.database.resolve()),
        "points_per_curve": args.points,
        "objective": "pointwise relative least squares",
        "validation": "leave one complete compound out",
        "summary": summary,
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
