#!/usr/bin/env python3
"""Benchmark the revised Govender liquid-conductivity model against Perry.

The revised model applies the defensible changes selected by the maintained
probes in this directory:

* refitted aliphatic COOH, ordinary-ketone, acyclic-thioether, and tertiary-
  amine contributions;
* one amine/alcohol correction per qualifying molecule; and
* second-order corrections for nonaromatic three-, four-, and five-membered
  rings (the four-member value is interpolated).

Both modes honor the production method's unconditional refusal of cyclic
thioethers and unsaturated cyclic ketones. Select ``--domain screened`` to
additionally refuse molecules with fewer than three carbon atoms. The default
``both`` reports the production and screened results together.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import govender_method as gm  # noqa: E402
from perry_properties import PerryPropertyLibrary  # noqa: E402
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    artifact_record,
    distribution,
    error_summary,
    format_metrics,
    grouped_summaries,
    interval_label,
)
from scripts.thermal_conductivity.liquid.benchmark_govender_perry import (  # noqa: E402
    DEFAULT_CONDUCTIVITY_DATABASE,
    DEFAULT_PERRY_DATABASE,
    evaluate_compound,
)


DOMAIN_MODES = ("all", "screened", "both")

# The production module is authoritative; these aliases are serialized in the
# benchmark artifact so its exact parameter set remains visible.
REVISED_GROUPS = gm.LOCAL_GROUP_REFITS
AMINE_ALCOHOL_CORRECTION = gm.LOCAL_AMINE_ALCOHOL_CORRECTION
RING_CORRECTIONS = gm.LOCAL_RING_CORRECTIONS


@dataclass(frozen=True)
class RevisedPoint:
    cas: str
    temperature_K: float
    temperature_over_tb: float
    reference_W_per_m_K: float
    predicted_W_per_m_K: float
    signed_error_percent: float
    absolute_error_percent: float


@dataclass(frozen=True)
class RevisedCompound:
    cas: str
    name: str
    formula: str
    smiles: str
    carbon_atoms: int
    tb_K: float
    Tmin_K: float
    Tmax_K: float
    points: int
    groups: dict[int, int]
    corrections: tuple[str, ...]
    mape_percent: float
    median_ape_percent: float
    p95_ape_percent: float
    maximum_ape_percent: float
    mean_signed_error_percent: float


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def nonaromatic_ring_counts(molecule: Chem.Mol) -> Counter[int]:
    return gm.nonaromatic_ring_counts(molecule)


def cyclic_thioether(molecule: Chem.Mol, groups: dict[int, int]) -> bool:
    return gm.is_cyclic_thioether(molecule, groups)


def cyclic_unsaturated_ketone(
    molecule: Chem.Mol, groups: dict[int, int]
) -> bool:
    return gm.is_unsaturated_cyclic_ketone(molecule, groups)


def domain_reasons(
    molecule: Chem.Mol, groups: dict[int, int]
) -> tuple[str, ...]:
    reasons = []
    carbon_atoms = sum(
        atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()
    )
    if carbon_atoms < 3:
        reasons.append("fewer than 3 carbon atoms")
    if cyclic_thioether(molecule, groups):
        reasons.append("cyclic thioether")
    if cyclic_unsaturated_ketone(molecule, groups):
        reasons.append("unsaturated cyclic ketone")
    return tuple(reasons)


def evaluate_revised(compound, base_points) -> tuple[RevisedCompound, list[RevisedPoint]]:
    molecule = Chem.MolFromSmiles(compound.smiles)
    estimate = gm.estimate_from_mol(
        molecule,
        tb=compound.tb_K,
        smiles=compound.smiles,
        local_refits=True,
    )
    groups = estimate.groups
    points = []
    for base in base_points:
        predicted = estimate.conductivity_W_m_K(base.temperature_K)
        signed = 100.0 * (predicted / base.reference_W_per_m_K - 1.0)
        points.append(RevisedPoint(
            cas=compound.cas,
            temperature_K=base.temperature_K,
            temperature_over_tb=base.temperature_K / compound.tb_K,
            reference_W_per_m_K=base.reference_W_per_m_K,
            predicted_W_per_m_K=predicted,
            signed_error_percent=signed,
            absolute_error_percent=abs(signed),
        ))
    summary = error_summary(points)
    return RevisedCompound(
        cas=compound.cas,
        name=compound.name,
        formula=compound.formula,
        smiles=compound.smiles,
        carbon_atoms=sum(
            atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()
        ),
        tb_K=compound.tb_K,
        Tmin_K=compound.Tmin_K,
        Tmax_K=compound.Tmax_K,
        points=len(points),
        groups=groups,
        corrections=tuple(
            warning for warning in estimate.warnings if "local" in warning
        ),
        mape_percent=float(summary["mape_percent"]),
        median_ape_percent=float(summary["median_ape_percent"]),
        p95_ape_percent=float(summary["p95_ape_percent"]),
        maximum_ape_percent=float(summary["maximum_ape_percent"]),
        mean_signed_error_percent=float(summary["mean_signed_error_percent"]),
    ), points


def load_candidates(args: argparse.Namespace) -> tuple[list[tuple], dict]:
    database = json.loads(
        args.conductivity_database.read_text(encoding="utf-8")
    )["chemicals"]
    perry = PerryPropertyLibrary(path=args.perry_database)
    candidates = []
    exclusions: Counter[str] = Counter()
    eligible = eligible_curves = 0
    for cas, entry in sorted(database.items()):
        rows = entry.get("liquid_thermal_conductivity") or []
        if not rows:
            continue
        eligible += 1
        eligible_curves += len(rows)
        try:
            compound, base_points = evaluate_compound(
                cas, entry, perry=perry,
                sample_points=args.points, tb_source="perry",
            )
            revised, revised_points = evaluate_revised(compound, base_points)
        except Exception as exc:
            exclusions[f"{type(exc).__name__}: {exc}"] += 1
            continue
        molecule = Chem.MolFromSmiles(compound.smiles)
        reasons = domain_reasons(molecule, revised.groups)
        candidates.append((compound, base_points, revised, revised_points, reasons))
    return candidates, {
        "eligible_liquid_conductivity_records": eligible,
        "eligible_conductivity_curves": eligible_curves,
        "method_exclusions": dict(exclusions),
    }


def compound_equal_metrics(compounds) -> dict[str, float | int]:
    mapes = [compound.mape_percent for compound in compounds]
    biases = [compound.mean_signed_error_percent for compound in compounds]
    mape_distribution = distribution(mapes)
    return {
        "compounds": len(compounds),
        "mean_curve_mape_percent": sum(mapes) / len(mapes),
        "median_curve_mape_percent": mape_distribution["median"],
        "mean_curve_bias_percent": sum(biases) / len(biases),
    }


def evaluate_mode(candidates: list[tuple], common: dict, mode: str) -> dict:
    selected = []
    domain_exclusions: Counter[str] = Counter()
    excluded_compounds = []
    for item in candidates:
        reasons = item[4]
        if mode == "screened" and reasons:
            for reason in reasons:
                domain_exclusions[reason] += 1
            excluded_compounds.append({
                "cas": item[0].cas,
                "name": item[0].name,
                "reasons": reasons,
            })
            continue
        selected.append(item)

    base_compounds = [item[0] for item in selected]
    base_points = [point for item in selected for point in item[1]]
    revised_compounds = [item[2] for item in selected]
    revised_points = [point for item in selected for point in item[3]]
    correction_counts = Counter(
        correction
        for compound in revised_compounds
        for correction in compound.corrections
    )
    ttb_edges = (-math.inf, 0.5, 0.75, 1.0, 1.25, math.inf)
    return {
        "domain": mode,
        "coverage": {
            **common,
            "benchmarked_compounds": len(revised_compounds),
            "benchmarked_conductivity_curves": sum(
                item[0].curves for item in selected
            ),
            "sampled_states": len(revised_points),
            "domain_excluded_compounds": len(excluded_compounds),
            "domain_exclusions": dict(domain_exclusions),
            "excluded_compounds": excluded_compounds,
        },
        "published_same_population": {
            "point_weighted": error_summary(base_points),
            "compound_equal": compound_equal_metrics(base_compounds),
        },
        "revised": {
            "point_weighted": error_summary(revised_points),
            "compound_equal": compound_equal_metrics(revised_compounds),
            "curve_mape_distribution": distribution([
                compound.mape_percent for compound in revised_compounds
            ]),
        },
        "by_temperature_over_tb": grouped_summaries(
            revised_points,
            lambda point: interval_label(point.temperature_over_tb, ttb_edges),
        ),
        "correction_usage_compounds": dict(correction_counts),
        "worst_compounds": [
            asdict(compound) for compound in sorted(
                revised_compounds,
                key=lambda item: item.mape_percent,
                reverse=True,
            )
        ],
    }


def build_report(payload: dict, top: int) -> str:
    lines = [
        "Revised Govender saturated-liquid conductivity benchmark vs Perry 9th Table 2-147",
        "Tb source: Perry normal point or vapor-pressure inversion",
        "Corrections: COOH, ordinary ketone, acyclic thioether, tertiary amine,",
        "             amine/alcohol, and nonaromatic 3-5 member rings",
    ]
    for mode in payload["modes"]:
        coverage = mode["coverage"]
        published = mode["published_same_population"]["point_weighted"]
        revised = mode["revised"]["point_weighted"]
        curve = mode["revised"]["compound_equal"]
        lines.extend((
            "",
            f"DOMAIN: {mode['domain']}",
            (
                f"  coverage: {coverage['benchmarked_compounds']}/"
                f"{coverage['eligible_liquid_conductivity_records']} compounds, "
                f"{coverage['sampled_states']} sampled states"
            ),
            f"  published same population: {format_metrics(published)}",
            f"  revised point-weighted:    {format_metrics(revised)}",
            (
                f"  revised compound-equal: n={curve['compounds']} "
                f"MAPE={curve['mean_curve_mape_percent']:.2f}% "
                f"median={curve['median_curve_mape_percent']:.2f}% "
                f"bias={curve['mean_curve_bias_percent']:+.2f}%"
            ),
        ))
        if mode["domain"] == "screened":
            lines.append(
                f"  domain exclusions: {coverage['domain_excluded_compounds']} unique"
            )
            for reason, count in sorted(coverage["domain_exclusions"].items()):
                lines.append(f"    {count:3d} {reason}")
        count = min(top, len(mode["worst_compounds"]))
        lines.append(f"  WORST {count}")
        for rank, compound in enumerate(mode["worst_compounds"][:top], 1):
            lines.append(
                f"    {rank:2d}. {compound['name'][:32]:32s} "
                f"{compound['cas']:12s} MAPE={compound['mape_percent']:7.2f}% "
                f"bias={compound['mean_signed_error_percent']:+7.2f}%"
            )
    lines.extend(("", "METHOD EXCLUSIONS BEFORE DOMAIN SCREENING"))
    exclusions = payload["modes"][0]["coverage"]["method_exclusions"]
    for reason, count in sorted(exclusions.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {count:3d} {reason}")
    return "\n".join(lines)


def benchmark(args: argparse.Namespace) -> tuple[dict, str]:
    candidates, common = load_candidates(args)
    requested = ("all", "screened") if args.domain == "both" else (args.domain,)
    payload = {
        "schema_version": 1,
        "benchmark": "revised_govender_vs_perry_9_table_2_147",
        "parameters": {
            "revised_groups": {str(key): value for key, value in REVISED_GROUPS.items()},
            "amine_alcohol_correction": AMINE_ALCOHOL_CORRECTION,
            "ring_corrections": {
                str(key): value for key, value in RING_CORRECTIONS.items()
            },
        },
        "domain_screen": {
            "unconditional_do_not_estimate": [
                "cyclic thioether",
                "unsaturated cyclic ketone",
            ],
            "minimum_carbon_atoms": 3,
            "screened_mode_additional_rule": "fewer than 3 carbon atoms",
        },
        "method_notes": {
            "group_58": "refit applies only to ketones outside the unsaturated cyclic class",
            "group_100": "refit applies only to acyclic thioethers",
            "ring_4": "arithmetic interpolation of fitted ring-3 and ring-5 corrections",
            "amine_alcohol": "one correction per molecule, not per OH group",
        },
        "modes": [evaluate_mode(candidates, common, mode) for mode in requested],
        "reproducibility": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("rdkit", "chemicals", "scipy")
            },
            "argv": sys.argv,
            "inputs": {
                "thermal_conductivity": artifact_record(args.conductivity_database),
                "perry_properties": artifact_record(args.perry_database),
                "govender_method": artifact_record(ROOT / "govender_method.py"),
                "uv_lock": artifact_record(ROOT / "uv.lock"),
            },
            "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
    }
    return payload, build_report(payload, args.top)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain", choices=DOMAIN_MODES, default="both",
        help="Applicability domain to report (default: both).",
    )
    parser.add_argument(
        "--conductivity-database", type=Path, default=DEFAULT_CONDUCTIVITY_DATABASE
    )
    parser.add_argument("--perry-database", type=Path, default=DEFAULT_PERRY_DATABASE)
    parser.add_argument(
        "--points", type=int, default=101,
        help="Uniform inclusive sample count per Perry curve (default: 101).",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points must be at least 2")
    if args.top < 0:
        parser.error("--top cannot be negative")
    return args


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    args = parse_args()
    payload, report = benchmark(args)
    print(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
