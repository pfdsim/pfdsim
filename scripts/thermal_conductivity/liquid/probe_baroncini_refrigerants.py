#!/usr/bin/env python3
"""Probe Baroncini refrigerant parameters against Perry liquid conductivity.

The broad population contains connected molecules composed only of carbon,
hydrogen, and at least one halogen.  Results are split by saturation/topology
and halogen family to expose applicability boundaries.  The general parameters
are A=0.494, a=0, b=0.5, and c=-0.167.  The authors' A=0.562 correction is
applied to R2x (C1H1X3) and non-mixed C1 refrigerants containing a single
halogen element type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perry_properties import PerryPropertyLibrary  # noqa: E402
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    conductivity_reference,
)
from scripts.thermal_conductivity.liquid.benchmark_baroncini_perry import (  # noqa: E402
    SAMPLE_POINTS,
    baroncini_conductivity_W_m_K,
    local_smiles,
    molecule_from_verified_smiles,
)


RDLogger.DisableLog("rdApp.*")

PARAMETERS = {"A": 0.494, "a": 0.0, "b": 0.5, "c": -0.167}
CORRECTED_A = 0.562
HALOGENS = {9: "F", 17: "Cl", 35: "Br", 53: "I"}
DEFAULT_CONDUCTIVITY_DATABASE = ROOT / "data" / "perry_thermal_conductivity.json"
DEFAULT_OUTPUT = (
    ROOT
    / "scripts"
    / "thermal_conductivity"
    / "liquid"
    / "results"
    / "baroncini_refrigerant_probe.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(molecule: Chem.Mol) -> dict | None:
    atomic_numbers = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
    halogen_numbers = atomic_numbers & set(HALOGENS)
    if 6 not in atomic_numbers or not halogen_numbers:
        return None
    if atomic_numbers - ({1, 6} | set(HALOGENS)):
        return None
    halogen_counts = Counter(
        HALOGENS[atom.GetAtomicNum()]
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() in HALOGENS
    )
    aromatic = any(atom.GetIsAromatic() for atom in molecule.GetAtoms())
    carbon_multiple_bonds = sum(
        bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        and bond.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
        for bond in molecule.GetBonds()
    )
    rings = len(molecule.GetRingInfo().AtomRings())
    if aromatic:
        topology = "aromatic"
    elif rings:
        topology = "cyclic"
    elif carbon_multiple_bonds:
        topology = "unsaturated_acyclic"
    else:
        topology = "saturated_acyclic"
    if len(halogen_counts) == 1:
        halogen_family = next(iter(halogen_counts))
    else:
        halogen_family = "mixed"
    return {
        "carbon_atoms": sum(
            atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()
        ),
        "hydrogen_atoms": sum(
            atom.GetAtomicNum() == 1 for atom in Chem.AddHs(molecule).GetAtoms()
        ),
        "halogen_counts": dict(sorted(halogen_counts.items())),
        "total_halogens": sum(halogen_counts.values()),
        "topology": topology,
        "halogen_family": halogen_family,
        "rings": rings,
        "carbon_multiple_bonds": carbon_multiple_bonds,
    }


def uses_authors_A_correction(classification: dict) -> bool:
    """Return whether the authors prescribe A=0.562 for this structure."""
    if classification["carbon_atoms"] != 1:
        return False
    hydrogens = classification["hydrogen_atoms"]
    halogens = classification["total_halogens"]
    is_R2x = hydrogens == 1 and halogens == 3
    is_nonmixed_C1 = len(classification["halogen_counts"]) == 1
    return is_R2x or is_nonmixed_C1


def metrics(errors: list[float]) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=float)
    absolute = np.abs(values)
    return {
        "points": len(values),
        "mape_percent": float(np.mean(absolute)),
        "median_ape_percent": float(np.median(absolute)),
        "p95_ape_percent": float(np.percentile(absolute, 95.0)),
        "maximum_ape_percent": float(np.max(absolute)),
        "mean_signed_error_percent": float(np.mean(values)),
        "within_20_percent": float(100.0 * np.mean(absolute <= 20.0)),
    }


def run_probe(conductivity_path: Path) -> dict:
    database = json.loads(conductivity_path.read_text())["chemicals"]
    perry = PerryPropertyLibrary()
    exclusions: Counter[str] = Counter()
    topology_errors: defaultdict[str, list[float]] = defaultdict(list)
    halogen_errors: defaultdict[str, list[float]] = defaultdict(list)
    carbon_count_errors: defaultdict[str, list[float]] = defaultdict(list)
    halogen_count_errors: defaultdict[str, list[float]] = defaultdict(list)
    all_errors = []
    all_baseline_errors = []
    compounds = []
    for cas, entry in sorted(database.items()):
        curves = entry.get("liquid_thermal_conductivity") or []
        if not curves:
            continue
        smiles = local_smiles(cas)
        try:
            molecule = molecule_from_verified_smiles(
                smiles, str(entry.get("formula") or "")
            )
        except ValueError as exc:
            exclusions[f"identity: {exc}"] += 1
            continue
        classification = classify(molecule)
        if classification is None:
            continue
        critical = perry.critical_properties(cas)
        if critical is None:
            exclusions["Perry critical temperature unavailable"] += 1
            continue
        critical_temperature = float(critical["Tc"].value)
        molecular_weight = float(curves[0]["molecular_weight"])
        errors = []
        baseline_errors = []
        correction_applies = uses_authors_A_correction(classification)
        corrected_parameters = {
            **PARAMETERS,
            "A": CORRECTED_A if correction_applies else PARAMETERS["A"],
        }
        for curve in curves:
            for temperature in np.linspace(
                float(curve["T_min_K"]),
                float(curve["T_max_K"]),
                SAMPLE_POINTS,
            ):
                temperature = float(temperature)
                if temperature > critical_temperature * (1.0 + 1.0e-10):
                    continue
                reference = conductivity_reference(curve, temperature)
                baseline = baroncini_conductivity_W_m_K(
                    temperature,
                    normal_boiling_temperature_K=1.0,
                    critical_temperature_K=critical_temperature,
                    molecular_weight_g_mol=molecular_weight,
                    **PARAMETERS,
                )
                predicted = baroncini_conductivity_W_m_K(
                    temperature,
                    normal_boiling_temperature_K=1.0,
                    critical_temperature_K=critical_temperature,
                    molecular_weight_g_mol=molecular_weight,
                    **corrected_parameters,
                )
                baseline_errors.append(100.0 * (baseline / reference - 1.0))
                errors.append(100.0 * (predicted / reference - 1.0))
        if not errors:
            exclusions["no subcritical conductivity states"] += 1
            continue
        all_errors.extend(errors)
        all_baseline_errors.extend(baseline_errors)
        topology_errors[str(classification["topology"])].extend(errors)
        halogen_errors[str(classification["halogen_family"])].extend(errors)
        carbon_bin = (
            "C1"
            if classification["carbon_atoms"] == 1
            else "C2"
            if classification["carbon_atoms"] == 2
            else "C3_plus"
        )
        halogen_bin = (
            f"X{classification['total_halogens']}"
            if classification["total_halogens"] <= 3
            else "X4_plus"
        )
        carbon_count_errors[carbon_bin].extend(errors)
        halogen_count_errors[halogen_bin].extend(errors)
        compounds.append(
            {
                "cas": cas,
                "name": entry.get("name") or cas,
                "formula": entry.get("formula") or "",
                "smiles": smiles,
                "classification": classification,
                "Tc_K": critical_temperature,
                "molecular_weight_g_mol": molecular_weight,
                "authors_A_correction": correction_applies,
                "A_used": corrected_parameters["A"],
                "baseline_errors": metrics(baseline_errors),
                "errors": metrics(errors),
            }
        )
    compounds.sort(key=lambda item: item["errors"]["mape_percent"], reverse=True)
    return {
        "parameters": {
            "general": PARAMETERS,
            "corrected_A": CORRECTED_A,
            "correction_domain": (
                "C1H1X3 (R2x) or non-mixed C1 with one halogen element type"
            ),
        },
        "coverage": {
            "compounds": len(compounds),
            "sampled_states": len(all_errors),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "baseline_overall": metrics(all_baseline_errors),
        "overall": metrics(all_errors),
        "by_topology": {
            topology: {
                "compounds": sum(
                    item["classification"]["topology"] == topology
                    for item in compounds
                ),
                **metrics(errors),
            }
            for topology, errors in sorted(topology_errors.items())
        },
        "by_halogen_family": {
            family: {
                "compounds": sum(
                    item["classification"]["halogen_family"] == family
                    for item in compounds
                ),
                **metrics(errors),
            }
            for family, errors in sorted(halogen_errors.items())
        },
        "by_carbon_count": {
            carbon_bin: {
                "compounds": sum(
                    (
                        "C1"
                        if item["classification"]["carbon_atoms"] == 1
                        else "C2"
                        if item["classification"]["carbon_atoms"] == 2
                        else "C3_plus"
                    )
                    == carbon_bin
                    for item in compounds
                ),
                **metrics(errors),
            }
            for carbon_bin, errors in sorted(carbon_count_errors.items())
        },
        "by_halogen_count": {
            halogen_bin: {
                "compounds": sum(
                    (
                        f"X{item['classification']['total_halogens']}"
                        if item["classification"]["total_halogens"] <= 3
                        else "X4_plus"
                    )
                    == halogen_bin
                    for item in compounds
                ),
                **metrics(errors),
            }
            for halogen_bin, errors in sorted(halogen_count_errors.items())
        },
        "compounds": compounds,
    }


def _line(values: dict) -> str:
    return (
        f"MAPE={values['mape_percent']:6.2f}% "
        f"MdAPE={values['median_ape_percent']:6.2f}% "
        f"P95={values['p95_ape_percent']:6.2f}% "
        f"bias={values['mean_signed_error_percent']:+6.2f}%"
    )


def report(result: dict) -> str:
    coverage = result["coverage"]
    lines = [
        "Baroncini refrigerant-parameter probe vs Perry 9th Table 2-147",
        f"coverage: {coverage['compounds']} compounds, {coverage['sampled_states']} states",
        "baseline A=0.494: " + _line(result["baseline_overall"]),
        "authors' corrected A: " + _line(result["overall"]),
        "",
        "BY TOPOLOGY",
    ]
    for topology, values in result["by_topology"].items():
        lines.append(
            f"  {topology:<22} n={values['compounds']:3d} {_line(values)}"
        )
    lines.extend(("", "BY HALOGEN FAMILY"))
    for family, values in result["by_halogen_family"].items():
        lines.append(f"  {family:<8} n={values['compounds']:3d} {_line(values)}")
    lines.extend(("", "BY CARBON COUNT"))
    for carbon_bin, values in result["by_carbon_count"].items():
        lines.append(
            f"  {carbon_bin:<8} n={values['compounds']:3d} {_line(values)}"
        )
    lines.extend(("", "BY HALOGEN COUNT"))
    for halogen_bin, values in result["by_halogen_count"].items():
        lines.append(
            f"  {halogen_bin:<8} n={values['compounds']:3d} {_line(values)}"
        )
    lines.extend(("", "COMPOUNDS, WORST FIRST"))
    for index, compound in enumerate(result["compounds"], 1):
        values = compound["errors"]
        baseline = compound["baseline_errors"]
        lines.append(
            f"  {index:2d}. {compound['name']:<30} {compound['cas']:<12} "
            f"{compound['classification']['topology']:<20} "
            f"{str(compound['classification']['halogen_counts']):<18} "
            f"MAPE={baseline['mape_percent']:6.2f}% -> {values['mape_percent']:6.2f}% "
            f"A={compound['A_used']:.3f}"
        )
    if coverage["exclusions"]:
        lines.extend(("", "EXCLUSIONS"))
        for reason, count in coverage["exclusions"].items():
            lines.append(f"  {count:3d} {reason}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conductivity-database",
        type=Path,
        default=DEFAULT_CONDUCTIVITY_DATABASE,
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_probe(args.conductivity_database)
    script = Path(__file__).resolve()
    base_script = ROOT / "scripts" / "thermal_conductivity" / "liquid" / "benchmark_baroncini_perry.py"
    result["reproducibility"] = {
        "python": platform.python_version(),
        "script": str(script.relative_to(ROOT)),
        "script_sha256": _sha256(script),
        "base_script_sha256": _sha256(base_script),
        "conductivity_database_sha256": _sha256(args.conductivity_database),
        "sample_points_per_curve": SAMPLE_POINTS,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(report(result))


if __name__ == "__main__":
    main()
