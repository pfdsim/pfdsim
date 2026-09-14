#!/usr/bin/env python3
"""Benchmark viscosity/Cv gas-conductivity relations against Perry 9th.

The geometry-dependent relation is::

    k M / (mu Cv) = 2.5                                      monatomic
    k M / (mu Cv) = 1.3 + (R/Cv) (1.7614 - 0.3523/Tr)       linear
    k M / (mu Cv) = 1.15 + 2.033 (R/Cv)                     otherwise

The acentric-factor relation is::

    k M / (mu Cv) = 3.75 S (R/Cv)
    S = 1 + A(0.215 + 0.28288A - 1.061B + 0.26665C)
            / (0.6366 + BC + 1.061AB)
    A = Cv/R - 1.5
    B = 0.7862 - 0.7109 omega + 1.3168 omega**2
    C = 2 + 10.5 Tr**2

where ``Tr = T/Tc`` and ``Cv = Cp - R``. Select a relation with ``--method``.
The benchmark uses Perry Table
2-145 as the conductivity reference, Perry Table 2-138 for dilute-vapor
viscosity, pfdsim's canonical local ideal-gas Cp kernel, and Perry critical
temperature and molecular weight.  Each curve is sampled only over the
intersection of those sources' stated temperature ranges.

Molecular geometry is classified without quantum calculations.  Formula atom
counts identify monatomic and diatomic species; larger molecules use local
``chemicals`` SMILES and RDKit topology.  A molecule is linear only when its
hydrogen-complete graph is an unbranched chain whose internal atoms are
sp-hybridized.  Records without enough structural evidence are excluded.
That exclusion applies only to the geometry-dependent relation; geometry is
retained as a diagnostic grouping for the acentric-factor relation.

The text report is written to stdout.  ``--output-json`` optionally writes all
compound results and aggregate metrics.  The script is offline and does not
modify pfdsim's property databases or runtime cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from chemicals.identifiers import search_chemical
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compound_identity import parse_formula_counts  # noqa: E402
from perry_properties import PerryPropertyLibrary  # noqa: E402
from physical_constants import R_J_MOL_K  # noqa: E402
from property_resolution.ideal_gas_cp import (  # noqa: E402
    CANONICAL_DATABASE_PATH,
    load_bundled_kernel,
)


DEFAULT_CONDUCTIVITY_DATABASE = ROOT / "data" / "perry_thermal_conductivity.json"
DEFAULT_PERRY_DATABASE = ROOT / "data" / "perry_properties.json"
GEOMETRY_METHOD = "geometry_dependent"
ACENTRIC_METHOD = "acentric_factor"
METHODS = (GEOMETRY_METHOD, ACENTRIC_METHOD)


@dataclass(frozen=True)
class PointResult:
    cas: str
    geometry: str
    temperature_K: float
    reduced_temperature: float
    reference_W_per_m_K: float
    predicted_W_per_m_K: float
    viscosity_Pa_s: float
    cp_J_per_mol_K: float
    cv_J_per_mol_K: float
    dimensionless_factor: float
    signed_error_percent: float
    absolute_error_percent: float


@dataclass(frozen=True)
class CompoundResult:
    cas: str
    name: str
    formula: str
    geometry: str
    geometry_source: str
    cp_method: str
    Tmin_K: float
    Tmax_K: float
    conductivity_curves: int
    points: int
    mape_percent: float
    median_ape_percent: float
    p95_ape_percent: float
    maximum_ape_percent: float
    mean_signed_error_percent: float


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


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(ROOT))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": sha256_path(resolved)}


def error_summary(points: Sequence[PointResult]) -> dict[str, float | int]:
    absolute = [point.absolute_error_percent for point in points]
    signed = [point.signed_error_percent for point in points]
    return {
        "points": len(points),
        "components": len({point.cas for point in points}),
        "mape_percent": mean(absolute),
        "median_ape_percent": percentile(absolute, 0.50),
        "p90_ape_percent": percentile(absolute, 0.90),
        "p95_ape_percent": percentile(absolute, 0.95),
        "maximum_ape_percent": max(absolute) if absolute else math.nan,
        "mean_signed_error_percent": mean(signed),
        "fraction_within_10_percent": mean([value <= 10.0 for value in absolute]),
        "fraction_within_20_percent": mean([value <= 20.0 for value in absolute]),
        "fraction_within_30_percent": mean([value <= 30.0 for value in absolute]),
    }


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "maximum": max(values) if values else math.nan,
    }


def classify_geometry(formula: str, cas: str) -> tuple[str, str]:
    """Return ``(geometry, evidence)`` from formula counts and local topology."""
    isotope_normalized_formula = re.sub(r"D(?=\d|[A-Z]|$)", "H", formula)
    counts = parse_formula_counts(isotope_normalized_formula)
    if not counts:
        raise ValueError("molecular formula unavailable or unparseable")
    atom_count = sum(counts.values())
    if atom_count == 1:
        return "monatomic", "formula_atom_count"
    if atom_count == 2:
        return "linear", "formula_atom_count"

    try:
        metadata = search_chemical(cas)
    except Exception as exc:
        raise ValueError("local SMILES unavailable") from exc
    smiles = str(getattr(metadata, "smiles", "") or "").strip()
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        raise ValueError("local SMILES unavailable or invalid")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("disconnected molecular structure")

    molecule = Chem.AddHs(molecule)
    if molecule.GetNumAtoms() != atom_count:
        raise ValueError("formula and hydrogen-complete SMILES atom counts disagree")
    degrees = [atom.GetDegree() for atom in molecule.GetAtoms()]
    linear = (
        max(degrees) <= 2
        and sum(degree == 1 for degree in degrees) == 2
        and all(
            atom.GetHybridization() == Chem.HybridizationType.SP
            for atom in molecule.GetAtoms()
            if atom.GetDegree() == 2
        )
    )
    return ("linear" if linear else "nonlinear"), "rdkit_hydrogen_complete_topology"


def conductivity_reference(row: dict, temperature_K: float) -> float:
    coefficients = [float(value) for value in row.get("coefficients", [])]
    equation = int(row["equation_id"])
    if equation == 100:
        value = sum(
            coefficient * temperature_K**power
            for power, coefficient in enumerate(coefficients)
        )
    elif equation == 102 and len(coefficients) >= 2:
        c1, c2 = coefficients[:2]
        c3 = coefficients[2] if len(coefficients) > 2 else 0.0
        c4 = coefficients[3] if len(coefficients) > 3 else 0.0
        value = (
            c1 * temperature_K**c2 / (1.0 + c3 / temperature_K + c4 / temperature_K**2)
        )
    else:
        raise ValueError(f"unsupported conductivity equation {equation}")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Perry conductivity reference is nonpositive or nonfinite")
    return value


def relation_factor(
    method: str,
    geometry: str,
    cv: float,
    reduced_temperature: float,
    acentric_factor: float,
) -> float:
    if method == GEOMETRY_METHOD:
        if geometry == "monatomic":
            return 2.5
        if geometry == "linear":
            return 1.3 + (R_J_MOL_K / cv) * (1.7614 - 0.3523 / reduced_temperature)
        return 1.15 + 2.033 * R_J_MOL_K / cv
    if method == ACENTRIC_METHOD:
        a = cv / R_J_MOL_K - 1.5
        b = 0.7862 - 0.7109 * acentric_factor + 1.3168 * acentric_factor**2
        c = 2.0 + 10.5 * reduced_temperature**2
        s = 1.0 + a * (0.215 + 0.28288 * a - 1.061 * b + 0.26665 * c) / (
            0.6366 + b * c + 1.061 * a * b
        )
        return 3.75 * s * R_J_MOL_K / cv
    raise ValueError(f"unknown benchmark method {method!r}")


def overlapping_viscosity_row(
    rows: Sequence[dict],
    lower: float,
    upper: float,
) -> tuple[dict, float, float]:
    candidates = []
    for row in rows:
        overlap_min = max(lower, float(row["T_min_K"]))
        overlap_max = min(upper, float(row["T_max_K"]))
        if overlap_max > overlap_min:
            candidates.append(
                (overlap_max - overlap_min, row, overlap_min, overlap_max)
            )
    if not candidates:
        raise ValueError("no overlapping Perry vapor-viscosity range")
    _, row, overlap_min, overlap_max = max(candidates, key=lambda item: item[0])
    return row, overlap_min, overlap_max


def interval_label(value: float, edges: Sequence[float]) -> str:
    for lower, upper in zip(edges, edges[1:]):
        if lower <= value < upper:
            if math.isinf(lower):
                return f"<{upper:g}"
            if math.isinf(upper):
                return f">={lower:g}"
            return f"{lower:g}-{upper:g}"
    return "unbinned"


def grouped_summaries(points: Sequence[PointResult], key) -> dict[str, dict]:
    groups: dict[str, list[PointResult]] = defaultdict(list)
    for point in points:
        groups[str(key(point))].append(point)
    return {label: error_summary(group) for label, group in sorted(groups.items())}


def evaluate_compound(
    cas: str,
    thermal_entry: dict,
    perry_entry: dict,
    *,
    sample_points: int,
    method: str,
) -> tuple[CompoundResult, list[PointResult]]:
    conductivity_rows = thermal_entry.get("vapor_thermal_conductivity") or []
    if not conductivity_rows:
        raise ValueError("vapor-conductivity rows unavailable")
    critical = perry_entry.get("critical_constants") or {}
    Tc = float(critical["Tc_K"])
    acentric_factor = (
        float(critical["omega"]) if method == ACENTRIC_METHOD else math.nan
    )
    molecular_weight_g_per_mol = float(
        conductivity_rows[0].get("molecular_weight") or critical["molecular_weight"]
    )
    if Tc <= 0.0 or molecular_weight_g_per_mol <= 0.0:
        raise ValueError("Tc and molecular weight must be positive")

    formula = str(thermal_entry.get("formula") or perry_entry.get("formula") or "")
    try:
        geometry, geometry_source = classify_geometry(formula, cas)
    except ValueError:
        if method == GEOMETRY_METHOD:
            raise
        geometry = "unclassified"
        geometry_source = "unavailable_not_required"
    cp_kernel = load_bundled_kernel(cas)
    if cp_kernel is None:
        raise ValueError("canonical local ideal-gas Cp kernel unavailable")

    molecular_weight_kg_per_mol = molecular_weight_g_per_mol / 1000.0
    points: list[PointResult] = []
    evaluated_ranges = []
    unavailable_ranges = 0
    for conductivity_row in conductivity_rows:
        lower = max(float(conductivity_row["T_min_K"]), cp_kernel.Tmin)
        upper = min(float(conductivity_row["T_max_K"]), cp_kernel.Tmax)
        try:
            viscosity_row, lower, upper = overlapping_viscosity_row(
                perry_entry.get("vapor_viscosity") or [], lower, upper
            )
        except ValueError:
            unavailable_ranges += 1
            continue
        evaluated_ranges.append((lower, upper))
        for index in range(sample_points):
            temperature = lower + (upper - lower) * index / (sample_points - 1)
            reference = conductivity_reference(conductivity_row, temperature)
            viscosity = PerryPropertyLibrary._eval_vapor_viscosity_Pa_s(
                viscosity_row, temperature
            )
            if viscosity is None or viscosity <= 0.0 or not math.isfinite(viscosity):
                raise ValueError("Perry vapor viscosity is nonpositive or nonfinite")
            cp = cp_kernel.cp(temperature)
            cv = cp - R_J_MOL_K
            if cv <= 0.0 or not math.isfinite(cv):
                raise ValueError("Cp - R is nonpositive or nonfinite")
            reduced_temperature = temperature / Tc
            factor = relation_factor(
                method,
                geometry,
                cv,
                reduced_temperature,
                acentric_factor,
            )
            predicted = factor * viscosity * cv / molecular_weight_kg_per_mol
            if predicted <= 0.0 or not math.isfinite(predicted):
                raise ValueError("estimated conductivity is nonpositive or nonfinite")
            signed_error = 100.0 * (predicted / reference - 1.0)
            points.append(
                PointResult(
                    cas=cas,
                    geometry=geometry,
                    temperature_K=temperature,
                    reduced_temperature=reduced_temperature,
                    reference_W_per_m_K=reference,
                    predicted_W_per_m_K=predicted,
                    viscosity_Pa_s=viscosity,
                    cp_J_per_mol_K=cp,
                    cv_J_per_mol_K=cv,
                    dimensionless_factor=factor,
                    signed_error_percent=signed_error,
                    absolute_error_percent=abs(signed_error),
                )
            )
    if not evaluated_ranges:
        raise ValueError("no common conductivity, viscosity, and Cp range")
    if unavailable_ranges:
        raise ValueError(
            f"{unavailable_ranges} of {len(conductivity_rows)} conductivity rows "
            "lack a common viscosity/Cp range"
        )

    summary = error_summary(points)
    compound = CompoundResult(
        cas=cas,
        name=str(thermal_entry.get("name") or perry_entry.get("name") or cas),
        formula=formula,
        geometry=geometry,
        geometry_source=geometry_source,
        cp_method=cp_kernel.method,
        Tmin_K=min(lower for lower, _ in evaluated_ranges),
        Tmax_K=max(upper for _, upper in evaluated_ranges),
        conductivity_curves=len(evaluated_ranges),
        points=len(points),
        mape_percent=float(summary["mape_percent"]),
        median_ape_percent=float(summary["median_ape_percent"]),
        p95_ape_percent=float(summary["p95_ape_percent"]),
        maximum_ape_percent=float(summary["maximum_ape_percent"]),
        mean_signed_error_percent=float(summary["mean_signed_error_percent"]),
    )
    return compound, points


def format_metrics(summary: dict[str, float | int]) -> str:
    return (
        f"n={summary['points']:6d} comps={summary['components']:3d} "
        f"MAPE={summary['mape_percent']:7.2f}% "
        f"MdAPE={summary['median_ape_percent']:7.2f}% "
        f"P95={summary['p95_ape_percent']:7.2f}% "
        f"bias={summary['mean_signed_error_percent']:+7.2f}% "
        f"within20={100.0 * summary['fraction_within_20_percent']:5.1f}%"
    )


def build_report(payload: dict, *, top: int) -> str:
    coverage = payload["coverage"]
    overall = payload["overall_point_weighted"]
    curve_distribution = payload["compound_curve_mape_distribution"]
    compounds = payload["compounds"]
    lines = [
        (
            f"Gas {payload['method']['name']} thermal-conductivity benchmark "
            "vs Perry 9th Table 2-145"
        ),
        (
            f"coverage: {coverage['benchmarked_compounds']}/"
            f"{coverage['eligible_vapor_conductivity_records']} compounds, "
            f"{coverage['benchmarked_conductivity_curves']}/"
            f"{coverage['eligible_conductivity_curves']} curves, "
            f"{coverage['sampled_states']} sampled states"
        ),
        (
            "inputs: Perry vapor viscosity/MW/Tc"
            f"{'/omega' if payload['method']['name'] == ACENTRIC_METHOD else ''}; "
            "canonical local pfdsim ideal-gas Cp; Cv=Cp-R"
        ),
        f"relation: {payload['method']['relation']}",
        "sampling: uniform inclusive points over the common declared source range",
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
        "BY MOLECULAR GEOMETRY",
    ]
    for label, summary in payload["groups"]["geometry"].items():
        lines.append(f"  {label:12s} {format_metrics(summary)}")
    lines.extend(("", "BY REDUCED TEMPERATURE Tr"))
    for label, summary in payload["groups"]["reduced_temperature"].items():
        lines.append(f"  {label:12s} {format_metrics(summary)}")

    count = min(top, len(compounds))
    lines.extend(("", f"WORST {count} COMPOUNDS BY CURVE MAPE"))
    for rank, item in enumerate(
        sorted(compounds, key=lambda row: row["mape_percent"], reverse=True)[:top], 1
    ):
        lines.append(
            f"  {rank:2d}. {item['name'][:31]:31s} {item['cas']:12s} "
            f"{item['formula'][:14]:14s} {item['geometry']:9s} "
            f"MAPE={item['mape_percent']:7.2f}% "
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
    thermal = json.loads(args.conductivity_database.read_text(encoding="utf-8"))[
        "chemicals"
    ]
    perry = PerryPropertyLibrary(path=args.perry_database)
    compounds: list[CompoundResult] = []
    points: list[PointResult] = []
    exclusions: Counter = Counter()
    eligible = 0
    eligible_curves = 0

    for cas, thermal_entry in sorted(thermal.items()):
        if not thermal_entry.get("vapor_thermal_conductivity"):
            continue
        eligible += 1
        eligible_curves += len(thermal_entry["vapor_thermal_conductivity"])
        try:
            perry_entry = perry.get(cas, expand_identity=False)
            if perry_entry is None:
                raise ValueError("Perry property record unavailable")
            compound, compound_points = evaluate_compound(
                cas,
                thermal_entry,
                perry_entry,
                sample_points=args.points,
                method=args.method,
            )
        except Exception as exc:
            exclusions[f"{type(exc).__name__}: {exc}"] += 1
            continue
        compounds.append(compound)
        points.extend(compound_points)

    if not compounds:
        raise RuntimeError("No Perry compounds were benchmarked successfully")

    tr_edges = (-math.inf, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, math.inf)
    overall = error_summary(points)
    payload = {
        "schema_version": 3,
        "benchmark": f"gas_{args.method}_relation_vs_perry_9_table_2_145",
        "method": {
            "name": args.method,
            "cv": "canonical local ideal-gas Cp minus R",
            "relation": (
                "monatomic: 2.5; linear: 1.3 + (R/Cv)*(1.7614 - 0.3523/Tr); "
                "otherwise: 1.15 + 2.033*(R/Cv)"
                if args.method == GEOMETRY_METHOD
                else "3.75*S*(R/Cv), with A=Cv/R-1.5, "
                "B=0.7862-0.7109*omega+1.3168*omega^2, C=2+10.5*Tr^2"
            ),
            "sampling_points_per_curve": args.points,
        },
        "reproducibility": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "rdkit", "chemicals")
            },
            "argv": sys.argv,
            "inputs": {
                "thermal_conductivity": artifact_record(args.conductivity_database),
                "perry_properties": artifact_record(args.perry_database),
                "ideal_gas_heat_capacity": artifact_record(CANONICAL_DATABASE_PATH),
                "uv_lock": artifact_record(ROOT / "uv.lock"),
            },
            "scripts": {
                "benchmark": artifact_record(Path(__file__)),
                "reproduce": artifact_record(Path(__file__).with_name("reproduce.sh")),
            },
        },
        "coverage": {
            "eligible_vapor_conductivity_records": eligible,
            "benchmarked_compounds": len(compounds),
            "eligible_conductivity_curves": eligible_curves,
            "benchmarked_conductivity_curves": sum(
                compound.conductivity_curves for compound in compounds
            ),
            "sampled_states": len(points),
            "exclusions": dict(exclusions),
        },
        "overall_point_weighted": overall,
        "compound_curve_mape_distribution": distribution(
            [compound.mape_percent for compound in compounds]
        ),
        "groups": {
            "geometry": grouped_summaries(points, lambda point: point.geometry),
            "reduced_temperature": grouped_summaries(
                points,
                lambda point: interval_label(point.reduced_temperature, tr_edges),
            ),
        },
        "cp_methods": dict(Counter(compound.cp_method for compound in compounds)),
        "geometry_sources": dict(
            Counter(compound.geometry_source for compound in compounds)
        ),
        "compounds": [asdict(compound) for compound in compounds],
    }
    return payload, build_report(payload, top=args.top)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=METHODS,
        default=GEOMETRY_METHOD,
        help=f"Relation to benchmark (default: {GEOMETRY_METHOD}).",
    )
    parser.add_argument(
        "--conductivity-database", type=Path, default=DEFAULT_CONDUCTIVITY_DATABASE
    )
    parser.add_argument("--perry-database", type=Path, default=DEFAULT_PERRY_DATABASE)
    parser.add_argument(
        "--points",
        type=int,
        default=101,
        help="Uniform inclusive sample count per overlapping curve (default: 101).",
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Worst compounds shown in the text report."
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
