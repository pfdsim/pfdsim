#!/usr/bin/env python3
"""First-pass Baroncini liquid-conductivity benchmark against Perry.

The class definitions are deliberately structural and mutually exclusive:

* hydrocarbons are aromatic, olefinic, saturated cyclic, or saturated acyclic;
* alkynes are outside the supplied class table;
* C/H/O compounds enter an oxygenated class only when every oxygen atom is
  accounted for by that class's functional group;
* alcohols require OH attached to a nonaromatic, noncarbonyl carbon.

All classes use a=1.2 and c=0.167.  The refrigerant-specific alternatives are
not used.  Perry Table 2-147 curves are sampled uniformly over their complete
declared ranges, using Perry Tb, Tc, and molecular weight as model inputs.
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
from scripts.thermal_conductivity.liquid.benchmark_govender_perry import (  # noqa: E402
    boiling_point,
    local_smiles,
    molecule_from_verified_smiles,
)


RDLogger.DisableLog("rdApp.*")

TEMPERATURE_EXPONENT = 1.2
CRITICAL_TEMPERATURE_EXPONENT = 0.167
REDUCED_TEMPERATURE_EXPONENT = 0.38
SAMPLE_POINTS = 101
CLASS_PARAMETERS = {
    "saturated_hydrocarbons": {"A": 0.00350, "b": 0.5},
    "olefins": {"A": 0.0361, "b": 1.0},
    "cycloparaffins": {"A": 0.0310, "b": 1.0},
    "aromatics": {"A": 0.0346, "b": 1.0},
    "alcohols": {"A": 0.00339, "b": 0.5},
    "organic_acids": {"A": 0.00319, "b": 0.5},
    "ketones": {"A": 0.00383, "b": 0.5},
    "esters": {"A": 0.0415, "b": 1.0},
    "ethers": {"A": 0.0385, "b": 1.0},
}
DEFAULT_CONDUCTIVITY_DATABASE = ROOT / "data" / "perry_thermal_conductivity.json"
DEFAULT_OUTPUT = (
    ROOT
    / "scripts"
    / "thermal_conductivity"
    / "liquid"
    / "results"
    / "baroncini_perry_benchmark.json"
)

CARBOXYLIC_ACID = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")
ESTER = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][#6]")
KETONE = Chem.MolFromSmarts("[CX3](=[OX1])([#6])[#6]")
ALIPHATIC_ALCOHOL = Chem.MolFromSmarts("[OX2H1][#6;!a;!$(C=O)]")
ETHER = Chem.MolFromSmarts("[OD2]([#6])[#6]")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structural_details(molecule: Chem.Mol) -> dict[str, int | bool]:
    carbon_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6]
    return {
        "carbon_atoms": len(carbon_atoms),
        "oxygen_atoms": sum(
            atom.GetAtomicNum() == 8 for atom in molecule.GetAtoms()
        ),
        "rings": len(molecule.GetRingInfo().AtomRings()),
        "aromatic_atoms": sum(atom.GetIsAromatic() for atom in molecule.GetAtoms()),
        "carbon_double_bonds": sum(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetBeginAtom().GetAtomicNum() == 6
            and bond.GetEndAtom().GetAtomicNum() == 6
            for bond in molecule.GetBonds()
        ),
        "carbon_triple_bonds": sum(
            bond.GetBondType() == Chem.BondType.TRIPLE
            and bond.GetBeginAtom().GetAtomicNum() == 6
            and bond.GetEndAtom().GetAtomicNum() == 6
            for bond in molecule.GetBonds()
        ),
        "branched": any(
            sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) > 2
            for atom in carbon_atoms
        ),
        "acid_groups": len(molecule.GetSubstructMatches(CARBOXYLIC_ACID)),
        "ester_groups": len(molecule.GetSubstructMatches(ESTER)),
        "ketone_groups": len(molecule.GetSubstructMatches(KETONE)),
        "alcohol_groups": len(molecule.GetSubstructMatches(ALIPHATIC_ALCOHOL)),
        "ether_groups": len(molecule.GetSubstructMatches(ETHER)),
    }


def classify(molecule: Chem.Mol) -> tuple[str | None, dict[str, int | bool]]:
    details = structural_details(molecule)
    heavy_elements = {
        atom.GetAtomicNum() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
    }
    if heavy_elements == {6}:
        if details["aromatic_atoms"]:
            return "aromatics", details
        if details["carbon_triple_bonds"]:
            return None, details
        if details["carbon_double_bonds"]:
            return "olefins", details
        if details["rings"]:
            return "cycloparaffins", details
        return "saturated_hydrocarbons", details
    if not heavy_elements <= {6, 8}:
        return None, details

    oxygen_atoms = int(details["oxygen_atoms"])
    acid_groups = int(details["acid_groups"])
    ester_groups = int(details["ester_groups"])
    ketone_groups = int(details["ketone_groups"])
    alcohol_groups = int(details["alcohol_groups"])
    ether_groups = int(details["ether_groups"])
    if acid_groups and oxygen_atoms == 2 * acid_groups:
        return "organic_acids", details
    if ester_groups and oxygen_atoms == 2 * ester_groups:
        return "esters", details
    if ketone_groups and oxygen_atoms == ketone_groups:
        return "ketones", details
    if alcohol_groups and oxygen_atoms == alcohol_groups:
        return "alcohols", details
    if ether_groups and oxygen_atoms == ether_groups:
        return "ethers", details
    return None, details


def conductivity_W_m_K(
    temperature_K: float,
    *,
    normal_boiling_temperature_K: float,
    critical_temperature_K: float,
    molecular_weight_g_mol: float,
    chemical_class: str,
) -> float:
    if chemical_class not in CLASS_PARAMETERS:
        raise ValueError(f"unsupported Baroncini class {chemical_class!r}")
    parameters = CLASS_PARAMETERS[chemical_class]
    return baroncini_conductivity_W_m_K(
        temperature_K,
        normal_boiling_temperature_K=normal_boiling_temperature_K,
        critical_temperature_K=critical_temperature_K,
        molecular_weight_g_mol=molecular_weight_g_mol,
        A=parameters["A"],
        a=TEMPERATURE_EXPONENT,
        b=parameters["b"],
        c=CRITICAL_TEMPERATURE_EXPONENT,
    )


def baroncini_conductivity_W_m_K(
    temperature_K: float,
    *,
    normal_boiling_temperature_K: float,
    critical_temperature_K: float,
    molecular_weight_g_mol: float,
    A: float,
    a: float,
    b: float,
    c: float,
) -> float:
    """Evaluate the general Baroncini equation for supplied parameters."""
    if not 0.0 < temperature_K <= critical_temperature_K:
        raise ValueError("temperature must be positive and no greater than Tc")
    if normal_boiling_temperature_K <= 0.0 or molecular_weight_g_mol <= 0.0:
        raise ValueError("Tb and molecular weight must be positive")
    reduced_temperature = temperature_K / critical_temperature_K
    return (
        A
        * normal_boiling_temperature_K**a
        * molecular_weight_g_mol ** (-b)
        * critical_temperature_K ** (-c)
        * (1.0 - reduced_temperature) ** REDUCED_TEMPERATURE_EXPONENT
        * reduced_temperature ** (-1.0 / 6.0)
    )


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


def run_benchmark(conductivity_path: Path) -> dict:
    database = json.loads(conductivity_path.read_text())["chemicals"]
    perry = PerryPropertyLibrary()
    exclusions: Counter[str] = Counter()
    class_errors: defaultdict[str, list[float]] = defaultdict(list)
    all_errors = []
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
        chemical_class, details = classify(molecule)
        if chemical_class is None:
            exclusions["outside broad structural class definitions"] += 1
            continue
        try:
            boiling_temperature, boiling_method = boiling_point(perry, cas)
        except ValueError as exc:
            exclusions[str(exc)] += 1
            continue
        critical = perry.critical_properties(cas)
        if critical is None:
            exclusions["Perry critical temperature unavailable"] += 1
            continue
        critical_temperature = float(critical["Tc"].value)
        molecular_weight = float(curves[0]["molecular_weight"])
        errors = []
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
                predicted = conductivity_W_m_K(
                    temperature,
                    normal_boiling_temperature_K=boiling_temperature,
                    critical_temperature_K=critical_temperature,
                    molecular_weight_g_mol=molecular_weight,
                    chemical_class=chemical_class,
                )
                errors.append(100.0 * (predicted / reference - 1.0))
        if not errors:
            exclusions["no subcritical conductivity states"] += 1
            continue
        all_errors.extend(errors)
        class_errors[chemical_class].extend(errors)
        compounds.append(
            {
                "cas": cas,
                "name": entry.get("name") or cas,
                "formula": entry.get("formula") or "",
                "smiles": smiles,
                "class": chemical_class,
                "details": details,
                "Tb_K": boiling_temperature,
                "Tb_method": boiling_method,
                "Tc_K": critical_temperature,
                "molecular_weight_g_mol": molecular_weight,
                "errors": metrics(errors),
            }
        )
    compounds.sort(key=lambda item: item["errors"]["mape_percent"], reverse=True)
    return {
        "model": {
            "a": TEMPERATURE_EXPONENT,
            "c": CRITICAL_TEMPERATURE_EXPONENT,
            "reduced_temperature_exponent": REDUCED_TEMPERATURE_EXPONENT,
            "classes": CLASS_PARAMETERS,
        },
        "coverage": {
            "compounds": len(compounds),
            "sampled_states": len(all_errors),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "overall": metrics(all_errors),
        "classes": {
            chemical_class: {
                "compounds": sum(
                    item["class"] == chemical_class for item in compounds
                ),
                **metrics(class_errors[chemical_class]),
            }
            for chemical_class in CLASS_PARAMETERS
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
        "Baroncini liquid thermal-conductivity benchmark vs Perry 9th Table 2-147",
        f"coverage: {coverage['compounds']} compounds, {coverage['sampled_states']} states",
        "overall: " + _line(result["overall"]),
        "",
        "BY BROAD STRUCTURAL CLASS",
    ]
    for chemical_class, values in result["classes"].items():
        lines.append(
            f"  {chemical_class:<24} n={values['compounds']:3d} {_line(values)}"
        )
    lines.extend(("", "WORST 25 COMPOUNDS"))
    for index, compound in enumerate(result["compounds"][:25], 1):
        values = compound["errors"]
        lines.append(
            f"  {index:2d}. {compound['name']:<32} {compound['cas']:<12} "
            f"{compound['class']:<22} MAPE={values['mape_percent']:6.2f}% "
            f"bias={values['mean_signed_error_percent']:+6.2f}%"
        )
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
    result = run_benchmark(args.conductivity_database)
    script = Path(__file__).resolve()
    result["reproducibility"] = {
        "python": platform.python_version(),
        "script": str(script.relative_to(ROOT)),
        "script_sha256": _sha256(script),
        "conductivity_database_sha256": _sha256(args.conductivity_database),
        "sample_points_per_curve": SAMPLE_POINTS,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(report(result))


if __name__ == "__main__":
    main()
