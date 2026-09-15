#!/usr/bin/env python3
"""Benchmark the Govender liquid-conductivity method against Perry 9th.

Perry Table 2-147 saturated-liquid correlations are sampled uniformly over
their complete declared temperature range.  The published Govender method is
evaluated with a Perry normal boiling point by default.  ``--tb-source
nannoolal`` instead exercises its fully structure-based path.

Structures come from the installed ``chemicals`` database and must agree with
Perry's molecular formula.  Unsupported compounds are reported as exclusions;
no values are fitted and no runtime property database is modified.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from chemicals.identifiers import search_chemical
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import govender_method  # noqa: E402
from compound_identity import parse_formula_counts  # noqa: E402
from perry_properties import PerryPropertyLibrary  # noqa: E402
from property_resolution.organic_classification import (  # noqa: E402
    classify_strict_molecular_organic,
)
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    DEFAULT_CONDUCTIVITY_DATABASE,
    artifact_record,
    conductivity_reference,
    distribution,
    error_summary,
    format_metrics,
    grouped_summaries,
    interval_label,
)


DEFAULT_PERRY_DATABASE = ROOT / "data" / "perry_properties.json"
TB_SOURCES = ("perry", "nannoolal")


@dataclass(frozen=True)
class PointResult:
    cas: str
    temperature_K: float
    temperature_over_tb: float
    reference_W_per_m_K: float
    predicted_W_per_m_K: float
    signed_error_percent: float
    absolute_error_percent: float
    caution_group: str


@dataclass(frozen=True)
class CompoundResult:
    cas: str
    name: str
    formula: str
    smiles: str
    tb_K: float
    tb_method: str
    Tmin_K: float
    Tmax_K: float
    curves: int
    points: int
    groups: dict[int, int]
    warnings: tuple[str, ...]
    mape_percent: float
    median_ape_percent: float
    p95_ape_percent: float
    maximum_ape_percent: float
    mean_signed_error_percent: float


def local_smiles(cas: str) -> str:
    try:
        return str(search_chemical(cas).smiles or "").strip()
    except Exception:
        return ""


def normalized_formula_counts(formula: str) -> dict[str, int]:
    counts = parse_formula_counts(formula)
    if not counts:
        raise ValueError("Perry molecular formula unavailable or unparseable")
    normalized = {str(element): int(count) for element, count in counts.items()}
    if "D" in normalized:
        normalized["H"] = normalized.get("H", 0) + normalized.pop("D")
    return normalized


def molecule_from_verified_smiles(smiles: str, formula: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        raise ValueError("local SMILES unavailable or invalid")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("disconnected molecular structure")
    actual = Counter(
        atom.GetSymbol() for atom in Chem.AddHs(molecule).GetAtoms()
    )
    expected = normalized_formula_counts(formula)
    if dict(actual) != expected:
        raise ValueError("Perry formula and hydrogen-complete SMILES disagree")
    return molecule


def boiling_point(
    perry: PerryPropertyLibrary,
    cas: str,
) -> tuple[float, str]:
    value = perry.normal_boiling_point_K(cas)
    if value is None:
        value = perry.table_2_10_normal_boiling_point_K(cas)
    if value is None:
        raise ValueError("Perry normal boiling point unavailable")
    return float(value.value), value.method


def evaluate_compound(
    cas: str,
    entry: dict,
    *,
    perry: PerryPropertyLibrary,
    sample_points: int,
    tb_source: str,
) -> tuple[CompoundResult, list[PointResult]]:
    rows = entry.get("liquid_thermal_conductivity") or []
    if not rows:
        raise ValueError("liquid-conductivity rows unavailable")
    name = str(entry.get("name") or cas)
    formula = str(entry.get("formula") or "")
    smiles = local_smiles(cas)
    classification = classify_strict_molecular_organic(
        cas=cas, formula=formula, smiles=smiles
    )
    if not classification.is_organic:
        raise ValueError(f"not strict organic: {classification.reason}")
    molecule = molecule_from_verified_smiles(smiles, formula)

    if tb_source == "perry":
        tb_K, tb_method = boiling_point(perry, cas)
        estimate = govender_method.estimate_from_mol(
            molecule, tb=tb_K, smiles=smiles, local_refits=False
        )
    else:
        estimate = govender_method.estimate_from_mol(
            molecule, smiles=smiles, local_refits=False
        )
        tb_K = estimate.tb_K
        tb_method = "nannoolal_method"

    caution = "yes" if any("limited data" in warning for warning in estimate.warnings) else "no"
    points: list[PointResult] = []
    for row in rows:
        lower = float(row["T_min_K"])
        upper = float(row["T_max_K"])
        if upper <= lower:
            raise ValueError("Perry conductivity range is empty")
        for index in range(sample_points):
            temperature = lower + (upper - lower) * index / (sample_points - 1)
            reference = conductivity_reference(row, temperature)
            predicted = estimate.conductivity_W_m_K(temperature)
            signed_error = 100.0 * (predicted / reference - 1.0)
            points.append(
                PointResult(
                    cas=cas,
                    temperature_K=temperature,
                    temperature_over_tb=temperature / tb_K,
                    reference_W_per_m_K=reference,
                    predicted_W_per_m_K=predicted,
                    signed_error_percent=signed_error,
                    absolute_error_percent=abs(signed_error),
                    caution_group=caution,
                )
            )

    summary = error_summary(points)
    return (
        CompoundResult(
            cas=cas,
            name=name,
            formula=formula,
            smiles=smiles,
            tb_K=tb_K,
            tb_method=tb_method,
            Tmin_K=min(float(row["T_min_K"]) for row in rows),
            Tmax_K=max(float(row["T_max_K"]) for row in rows),
            curves=len(rows),
            points=len(points),
            groups=estimate.groups,
            warnings=estimate.warnings,
            mape_percent=float(summary["mape_percent"]),
            median_ape_percent=float(summary["median_ape_percent"]),
            p95_ape_percent=float(summary["p95_ape_percent"]),
            maximum_ape_percent=float(summary["maximum_ape_percent"]),
            mean_signed_error_percent=float(summary["mean_signed_error_percent"]),
        ),
        points,
    )


def build_report(payload: dict, *, top: int) -> str:
    coverage = payload["coverage"]
    overall = payload["overall_point_weighted"]
    curve_distribution = payload["compound_curve_mape_distribution"]
    lines = [
        "Govender saturated-liquid thermal-conductivity benchmark vs Perry 9th Table 2-147",
        (
            f"Tb source: {payload['method']['tb_source']}; coverage: "
            f"{coverage['benchmarked_compounds']}/"
            f"{coverage['eligible_liquid_conductivity_records']} compounds, "
            f"{coverage['benchmarked_conductivity_curves']}/"
            f"{coverage['eligible_conductivity_curves']} curves, "
            f"{coverage['sampled_states']} sampled states"
        ),
        "Tb methods: " + ", ".join(
            f"{method}={count}" for method, count in sorted(payload["tb_methods"].items())
        ),
        "sampling: uniform inclusive points over every Perry curve's declared range",
        "",
        "OVERALL POINT-WEIGHTED",
        f"  {format_metrics(overall)}",
        (
            f"  P90={overall['p90_ape_percent']:.2f}% "
            f"max={overall['maximum_ape_percent']:.2f}% "
            f"within10={100.0 * overall['fraction_within_10_percent']:.1f}% "
            f"within30={100.0 * overall['fraction_within_30_percent']:.1f}%"
        ),
        "",
        "COMPOUND-EQUAL DISTRIBUTION OF CURVE MAPE",
        (
            f"  n={curve_distribution['count']} mean={curve_distribution['mean']:.2f}% "
            f"median={curve_distribution['median']:.2f}% "
            f"P90={curve_distribution['p90']:.2f}% "
            f"P95={curve_distribution['p95']:.2f}% "
            f"max={curve_distribution['maximum']:.2f}%"
        ),
        "",
        "BY USE OF A PAPER CAUTION GROUP",
    ]
    for label, summary in payload["groups"]["caution_group"].items():
        lines.append(f"  {label:12s} {format_metrics(summary)}")
    lines.extend(("", "BY T/Tb"))
    for label, summary in payload["groups"]["temperature_over_tb"].items():
        lines.append(f"  {label:12s} {format_metrics(summary)}")

    compounds = payload["compounds"]
    count = min(top, len(compounds))
    lines.extend(("", f"WORST {count} COMPOUNDS BY CURVE MAPE"))
    for rank, item in enumerate(
        sorted(compounds, key=lambda row: row["mape_percent"], reverse=True)[:top], 1
    ):
        lines.append(
            f"  {rank:2d}. {item['name'][:34]:34s} {item['cas']:12s} "
            f"{item['formula'][:15]:15s} MAPE={item['mape_percent']:7.2f}% "
            f"bias={item['mean_signed_error_percent']:+7.2f}%"
        )
    lines.extend(("", "EXCLUSIONS"))
    for reason, count in sorted(
        coverage["exclusions"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"  {count:4d} {reason}")
    if not coverage["exclusions"]:
        lines.append("  none")
    return "\n".join(lines)


def benchmark(args: argparse.Namespace) -> tuple[dict, str]:
    database = json.loads(
        args.conductivity_database.read_text(encoding="utf-8")
    )["chemicals"]
    perry = PerryPropertyLibrary(path=args.perry_database)
    compounds: list[CompoundResult] = []
    points: list[PointResult] = []
    exclusions: Counter[str] = Counter()
    eligible = 0
    eligible_curves = 0

    for cas, entry in sorted(database.items()):
        rows = entry.get("liquid_thermal_conductivity") or []
        if not rows:
            continue
        eligible += 1
        eligible_curves += len(rows)
        try:
            compound, compound_points = evaluate_compound(
                cas,
                entry,
                perry=perry,
                sample_points=args.points,
                tb_source=args.tb_source,
            )
        except Exception as exc:
            exclusions[f"{type(exc).__name__}: {exc}"] += 1
            continue
        compounds.append(compound)
        points.extend(compound_points)

    if not compounds:
        raise RuntimeError("No Perry compounds were benchmarked successfully")
    ttb_edges = (-math.inf, 0.5, 0.75, 1.0, 1.25, math.inf)
    payload = {
        "schema_version": 1,
        "benchmark": "govender_saturated_liquid_vs_perry_9_table_2_147",
        "method": {
            "name": "govender_2020",
            "tb_source": args.tb_source,
            "sampling_points_per_curve": args.points,
        },
        "reproducibility": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("rdkit", "chemicals")
            },
            "argv": sys.argv,
            "inputs": {
                "thermal_conductivity": artifact_record(args.conductivity_database),
                "perry_properties": artifact_record(args.perry_database),
                "uv_lock": artifact_record(ROOT / "uv.lock"),
            },
            "scripts": {
                "benchmark": artifact_record(Path(__file__)),
                "govender_method": artifact_record(ROOT / "govender_method.py"),
                "nannoolal_method": artifact_record(ROOT / "nannoolal_method.py"),
            },
        },
        "coverage": {
            "eligible_liquid_conductivity_records": eligible,
            "benchmarked_compounds": len(compounds),
            "eligible_conductivity_curves": eligible_curves,
            "benchmarked_conductivity_curves": sum(item.curves for item in compounds),
            "sampled_states": len(points),
            "exclusions": dict(exclusions),
        },
        "overall_point_weighted": error_summary(points),
        "compound_curve_mape_distribution": distribution(
            [compound.mape_percent for compound in compounds]
        ),
        "groups": {
            "caution_group": grouped_summaries(
                points, lambda point: point.caution_group
            ),
            "temperature_over_tb": grouped_summaries(
                points,
                lambda point: interval_label(point.temperature_over_tb, ttb_edges),
            ),
        },
        "tb_methods": dict(Counter(item.tb_method for item in compounds)),
        "compounds": [asdict(compound) for compound in compounds],
    }
    return payload, build_report(payload, top=args.top)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tb-source", choices=TB_SOURCES, default="perry",
        help="Normal-boiling-point source (default: perry).",
    )
    parser.add_argument(
        "--conductivity-database", type=Path, default=DEFAULT_CONDUCTIVITY_DATABASE
    )
    parser.add_argument("--perry-database", type=Path, default=DEFAULT_PERRY_DATABASE)
    parser.add_argument(
        "--points", type=int, default=101,
        help="Uniform inclusive sample count per Perry curve (default: 101).",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Worst compounds shown in the text report.",
    )
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
