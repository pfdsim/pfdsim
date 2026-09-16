#!/usr/bin/env python3
"""Benchmark the supplied hydrocarbon liquid-conductivity model against Perry.

The model requires critical temperature and liquid molar volume at 20 degrees C::

    k = A M**B / V20 * (3 + 20*(1 - T/Tc)**(2/3))
                       / (3 + 20*(1 - 293.15/Tc)**(2/3))

``V20`` is in cm^3/mol and ``M`` is in g/mol.  All unbranched, acyclic
hydrocarbons use A=0.1811 and B=1.001; other hydrocarbons use A=0.4407 and
B=0.7717.  Perry correlations are used only where their stated liquid-density
range contains 293.15 K.  Conductivity curves are sampled at 101 evenly spaced
temperatures over their complete stated range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import govender_method  # noqa: E402
from modified_pachaiyappan import (  # noqa: E402
    OTHER_HYDROCARBON_PARAMETERS,
    REFERENCE_TEMPERATURE_K,
    STRAIGHT_CHAIN_PARAMETERS,
    conductivity_W_m_K as conductivity,
)
from perry_properties import PerryPropertyLibrary  # noqa: E402
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    conductivity_reference,
)
from scripts.thermal_conductivity.liquid.benchmark_govender_perry import (  # noqa: E402
    boiling_point,
    local_smiles,
    molecule_from_verified_smiles,
)


RDLogger.DisableLog("rdApp.*")

SAMPLE_POINTS = 101
DEFAULT_CONDUCTIVITY_DATABASE = ROOT / "data" / "perry_thermal_conductivity.json"
PRODUCTION_MODULE = ROOT / "modified_pachaiyappan.py"
DEFAULT_OUTPUT = (
    ROOT
    / "scripts"
    / "thermal_conductivity"
    / "liquid"
    / "results"
    / "hydrocarbon_model_perry_benchmark.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_hydrocarbon(molecule: Chem.Mol) -> bool:
    return all(atom.GetAtomicNum() in {1, 6} for atom in Chem.AddHs(molecule).GetAtoms())


def is_straight_chain(molecule: Chem.Mol) -> bool:
    """Return whether the carbon skeleton is acyclic and unbranched."""
    if molecule.GetRingInfo().NumRings():
        return False
    return all(
        sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) <= 2
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6
    )


def metrics(errors: list[float]) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=float)
    absolute = np.abs(values)
    return {
        "points": len(errors),
        "mape_percent": float(np.mean(absolute)),
        "median_ape_percent": float(np.median(absolute)),
        "p95_ape_percent": float(np.percentile(absolute, 95.0)),
        "mean_signed_error_percent": float(np.mean(values)),
        "within_10_percent": float(100.0 * np.mean(absolute <= 10.0)),
        "within_20_percent": float(100.0 * np.mean(absolute <= 20.0)),
    }


def format_metrics(values: dict[str, float | int]) -> str:
    return (
        f"MAPE={values['mape_percent']:.2f}% "
        f"MdAPE={values['median_ape_percent']:.2f}% "
        f"P95={values['p95_ape_percent']:.2f}% "
        f"bias={values['mean_signed_error_percent']:+.2f}% "
        f"within10={values['within_10_percent']:.1f}% "
        f"within20={values['within_20_percent']:.1f}%"
    )


def run_benchmark(conductivity_database: Path) -> dict:
    database = json.loads(conductivity_database.read_text())["chemicals"]
    perry = PerryPropertyLibrary()
    exclusions: Counter[str] = Counter()
    compounds = []
    all_errors: list[float] = []
    straight_errors: list[float] = []
    other_errors: list[float] = []
    common_hc_errors: list[float] = []
    common_govender_errors: list[float] = []

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
        if not is_hydrocarbon(molecule):
            continue

        critical = perry.critical_properties(cas)
        if critical is None:
            exclusions["critical properties unavailable"] += 1
            continue
        critical_temperature = float(critical["Tc"].value)
        if critical_temperature <= REFERENCE_TEMPERATURE_K:
            exclusions["Tc does not exceed 20 degrees C"] += 1
            continue
        volume = perry.liquid_molar_volume_m3_per_kmol(
            cas, REFERENCE_TEMPERATURE_K
        )
        if volume is None:
            exclusions["Perry V20 unavailable within stated range"] += 1
            continue

        molecular_weight = float(curves[0]["molecular_weight"])
        volume_cm3_mol = float(volume.value) * 1000.0
        straight_chain = is_straight_chain(molecule)
        errors = []
        references_and_temperatures = []
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
                predicted = conductivity(
                    temperature,
                    molecular_weight_g_mol=molecular_weight,
                    critical_temperature_K=critical_temperature,
                    molar_volume_20C_cm3_mol=volume_cm3_mol,
                    straight_chain=straight_chain,
                )
                errors.append(100.0 * (predicted / reference - 1.0))
                references_and_temperatures.append((temperature, reference))
        if not errors:
            exclusions["no subcritical conductivity states"] += 1
            continue

        common_govender = None
        try:
            tb_K, _ = boiling_point(perry, cas)
            estimate = govender_method.estimate_from_mol(
                molecule, tb=tb_K, smiles=smiles
            )
            govender_errors = [
                100.0
                * (estimate.conductivity_W_m_K(temperature) / reference - 1.0)
                for temperature, reference in references_and_temperatures
            ]
            common_hc_errors.extend(errors)
            common_govender_errors.extend(govender_errors)
            common_govender = metrics(govender_errors)
        except (ValueError, govender_method.GovenderError):
            pass

        all_errors.extend(errors)
        (straight_errors if straight_chain else other_errors).extend(errors)
        compounds.append(
            {
                "cas": cas,
                "name": entry.get("name") or cas,
                "formula": entry.get("formula") or "",
                "smiles": smiles,
                "straight_chain": straight_chain,
                "Tc_K": critical_temperature,
                "molecular_weight_g_mol": molecular_weight,
                "V20_cm3_mol": volume_cm3_mol,
                "hydrocarbon_model": metrics(errors),
                "govender_common_set": common_govender,
            }
        )

    compounds.sort(
        key=lambda item: item["hydrocarbon_model"]["mape_percent"], reverse=True
    )
    return {
        "model": {
            "straight_chain": {
                "A": STRAIGHT_CHAIN_PARAMETERS[0],
                "B": STRAIGHT_CHAIN_PARAMETERS[1],
            },
            "other_hydrocarbon": {
                "A": OTHER_HYDROCARBON_PARAMETERS[0],
                "B": OTHER_HYDROCARBON_PARAMETERS[1],
            },
            "reference_temperature_K": REFERENCE_TEMPERATURE_K,
            "straight_chain_definition": "acyclic unbranched carbon skeleton",
        },
        "coverage": {
            "compounds": len(compounds),
            "straight_chain_compounds": sum(
                bool(item["straight_chain"]) for item in compounds
            ),
            "other_hydrocarbon_compounds": sum(
                not item["straight_chain"] for item in compounds
            ),
            "sampled_states": len(all_errors),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "overall": metrics(all_errors),
        "straight_chain": metrics(straight_errors),
        "other_hydrocarbon": metrics(other_errors),
        "common_set_comparison": {
            "compounds": sum(
                item["govender_common_set"] is not None for item in compounds
            ),
            "hydrocarbon_model": metrics(common_hc_errors),
            "revised_govender": metrics(common_govender_errors),
        },
        "compounds": compounds,
    }


def report(result: dict) -> str:
    coverage = result["coverage"]
    comparison = result["common_set_comparison"]
    lines = [
        "Hydrocarbon-specific liquid thermal-conductivity model vs Perry 9th Table 2-147",
        (
            f"coverage: {coverage['compounds']} compounds "
            f"({coverage['straight_chain_compounds']} straight-chain, "
            f"{coverage['other_hydrocarbon_compounds']} other), "
            f"{coverage['sampled_states']} sampled states"
        ),
        f"overall:        {format_metrics(result['overall'])}",
        f"straight-chain: {format_metrics(result['straight_chain'])}",
        f"other HC:       {format_metrics(result['other_hydrocarbon'])}",
        "",
        f"COMMON SET WITH REVISED GOVENDER ({comparison['compounds']} compounds)",
        f"  hydrocarbon model: {format_metrics(comparison['hydrocarbon_model'])}",
        f"  revised Govender:  {format_metrics(comparison['revised_govender'])}",
        "",
        "WORST 20 COMPOUNDS",
    ]
    for index, compound in enumerate(result["compounds"][:20], 1):
        values = compound["hydrocarbon_model"]
        lines.append(
            f"  {index:2d}. {compound['name']:<34} {compound['cas']:<12} "
            f"MAPE={values['mape_percent']:7.2f}% "
            f"bias={values['mean_signed_error_percent']:+7.2f}%"
        )
    if coverage["exclusions"]:
        lines.extend(("", "HYDROCARBON EXCLUSIONS"))
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

    result = run_benchmark(args.conductivity_database)
    script = Path(__file__).resolve()
    result["reproducibility"] = {
        "python": platform.python_version(),
        "script": str(script.relative_to(ROOT)),
        "script_sha256": _sha256(script),
        "production_module": str(PRODUCTION_MODULE.relative_to(ROOT)),
        "production_module_sha256": _sha256(PRODUCTION_MODULE),
        "conductivity_database": str(args.conductivity_database.resolve()),
        "conductivity_database_sha256": _sha256(args.conductivity_database),
        "sample_points_per_curve": SAMPLE_POINTS,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(report(result))


if __name__ == "__main__":
    main()
