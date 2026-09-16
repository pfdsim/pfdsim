#!/usr/bin/env python3
"""Final selected-domain Baroncini benchmark and input sensitivity analysis.

Selected oxygenated classes use the published Baroncini equation with the
validated exclusions and local adjustments documented by the companion probes.
Halogenated hydrocarbons require at least two carbon atoms and use the general
refrigerant parameters.  Ordinary hydrocarbons are deliberately excluded in
favor of Modified Pachaiyappan.

The sensitivity section compares trusted Perry/CoolProp Tb and Tc with:

* Nannoolal Tb while retaining trusted Tc;
* Nannoolal Tc anchored by trusted Tb;
* fully structure-based Nannoolal Tb and Tc.

Every sensitivity case uses the exact same compounds and temperature states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import CoolProp
import CoolProp.CoolProp as CP
import numpy as np
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nannoolal_method  # noqa: E402
import baroncini_method as bm  # noqa: E402
from perry_properties import PerryPropertyLibrary  # noqa: E402
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    conductivity_reference,
)
from scripts.thermal_conductivity.liquid.benchmark_baroncini_perry import (  # noqa: E402
    CLASS_PARAMETERS,
    SAMPLE_POINTS,
)
from scripts.thermal_conductivity.liquid.benchmark_govender_perry import (  # noqa: E402
    boiling_point,
    local_smiles,
    molecule_from_verified_smiles,
)


RDLogger.DisableLog("rdApp.*")

RESULTS = ROOT / "scripts" / "thermal_conductivity" / "liquid" / "results"
DEFAULT_BARONCINI_ARTIFACT = RESULTS / "baroncini_perry_benchmark.json"
DEFAULT_REFRIGERANT_ARTIFACT = RESULTS / "baroncini_refrigerant_probe.json"
DEFAULT_MIXED_COOLPROP_ARTIFACT = (
    RESULTS / "baroncini_mixed_halogen_coolprop_probe.json"
)
DEFAULT_NONMIXED_COOLPROP_ARTIFACT = (
    RESULTS / "baroncini_nonmixed_halogen_coolprop_probe.json"
)
DEFAULT_LOCAL_REFIT_ARTIFACT = RESULTS / "baroncini_ketone_refit_probe.json"
DEFAULT_CONDUCTIVITY_DATABASE = ROOT / "data" / "perry_thermal_conductivity.json"
DEFAULT_OUTPUT = RESULTS / "final_baroncini_benchmark.json"

KETONE = Chem.MolFromSmarts("[CX3](=[OX1])([#6])[#6]")
ALDEHYDE = Chem.MolFromSmarts("[CX3H1](=[OX1])[#6]")

KETONE_POSITION_LOG_INTERCEPT = bm.KETONE_POSITION_LOG_INTERCEPT
KETONE_POSITION_LOG_MW = bm.KETONE_POSITION_LOG_MW
KETONE_POSITION_SIDE_MIN = bm.KETONE_POSITION_SIDE_MIN
ALDEHYDE_A = bm.ALDEHYDE_PARAMETERS[0]
ALDEHYDE_B = bm.ALDEHYDE_PARAMETERS[2]
REFRIGERANT_PARAMETERS = dict(
    zip(("A", "a", "b", "c"), bm.PUBLISHED_PARAMETERS["halogenated_hydrocarbons"], strict=True)
)
NORMAL_BOILING_PRESSURE_PA = 101325.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def input_metrics(errors: list[float]) -> dict[str, float | int]:
    return metrics(errors)


def ketone_side_min(molecule: Chem.Mol) -> int:
    matches = molecule.GetSubstructMatches(KETONE)
    if len(matches) != 1:
        raise ValueError("expected one ketone group")
    carbonyl_index = matches[0][0]
    carbonyl = molecule.GetAtomWithIdx(carbonyl_index)
    attached = [
        atom.GetIdx()
        for atom in carbonyl.GetNeighbors()
        if atom.GetAtomicNum() == 6
    ]
    if len(attached) != 2:
        raise ValueError("ketone carbonyl requires two carbon neighbors")

    def count_side(start: int) -> int:
        visited = {carbonyl_index}
        pending = [start]
        count = 0
        while pending:
            index = pending.pop()
            if index in visited:
                continue
            visited.add(index)
            atom = molecule.GetAtomWithIdx(index)
            if atom.GetAtomicNum() == 6:
                count += 1
            pending.extend(
                neighbor.GetIdx()
                for neighbor in atom.GetNeighbors()
                if neighbor.GetIdx() not in visited
            )
        return count

    return min(count_side(index) for index in attached)


def ketone_carbonyl_in_ring(molecule: Chem.Mol) -> bool:
    matches = molecule.GetSubstructMatches(KETONE)
    return any(molecule.GetAtomWithIdx(match[0]).IsInRing() for match in matches)


def parameters_for_published_class(chemical_class: str) -> dict[str, float]:
    source = CLASS_PARAMETERS[chemical_class]
    return {"A": source["A"], "a": 1.2, "b": source["b"], "c": 0.167}


def selected_oxygenated_model(item: dict) -> tuple[str, dict, float, str] | None:
    """Return category, parameters, multiplier, and subtype for a Perry item."""
    chemical_class = item["class"]
    details = item["details"]
    molecule = Chem.MolFromSmiles(item["smiles"])
    if molecule is None:
        return None
    if chemical_class == "alcohols":
        if int(details["alcohol_groups"]) != 1:
            return None
        return "alcohols", parameters_for_published_class("alcohols"), 1.0, "published"
    if chemical_class == "organic_acids":
        if item["name"] == "Formic acid":
            return None
        return (
            "organic_acids",
            parameters_for_published_class("organic_acids"),
            1.0,
            "published_without_formic_acid",
        )
    if chemical_class == "esters":
        if int(details["ester_groups"]) != 1:
            return None
        return "esters", parameters_for_published_class("esters"), 1.0, "monoester"
    if chemical_class == "ethers":
        if int(details["rings"]) != 0:
            return None
        return "ethers", parameters_for_published_class("ethers"), 1.0, "acyclic"
    if chemical_class != "ketones":
        return None
    if len(molecule.GetSubstructMatches(KETONE)) != 1 or ketone_carbonyl_in_ring(molecule):
        return None
    parameters = parameters_for_published_class("ketones")
    if any(atom.GetIsAromatic() for atom in molecule.GetAtoms()):
        return "ketones", parameters, 1.0, "aromatic_published"
    if molecule.GetRingInfo().NumRings() != 0:
        return None
    side_min = ketone_side_min(molecule)
    factor = math.exp(
        KETONE_POSITION_LOG_INTERCEPT
        + KETONE_POSITION_LOG_MW * math.log(float(item["molecular_weight_g_mol"]))
        + KETONE_POSITION_SIDE_MIN * side_min
    )
    return "ketones", parameters, factor, "acyclic_aliphatic_local_refit"


def baroncini_value(
    record: dict,
    temperature_K: float,
    tb_K: float,
    tc_K: float,
) -> float:
    parameters = record["parameters"]
    classification = bm.BaronciniClassification(
        category=record["category"],
        subtype=record["subtype"],
        carbon_atoms=record.get("carbon_atoms", 0),
        parameters=(
            parameters["A"],
            parameters["a"],
            parameters["b"],
            parameters["c"],
        ),
        multiplier=record["multiplier"],
    )
    return bm.conductivity_W_m_K(
        temperature_K,
        normal_boiling_temperature_K=tb_K,
        critical_temperature_K=tc_K,
        molecular_weight_g_mol=record["molecular_weight_g_mol"],
        classification=classification,
    )


def perry_states(item: dict, database: dict) -> list[tuple[float, float]]:
    values = []
    for curve in database[item["cas"]]["liquid_thermal_conductivity"]:
        for temperature in np.linspace(
            float(curve["T_min_K"]), float(curve["T_max_K"]), SAMPLE_POINTS
        ):
            temperature = float(temperature)
            if temperature > float(item["Tc_K"]) * (1.0 + 1.0e-10):
                continue
            values.append((temperature, conductivity_reference(curve, temperature)))
    return values


def make_perry_record(
    item: dict,
    category: str,
    parameters: dict,
    multiplier: float,
    subtype: str,
    database: dict,
) -> dict:
    return {
        "source": "perry",
        "cas": item["cas"],
        "name": item["name"],
        "formula": item["formula"],
        "smiles": item["smiles"],
        "category": category,
        "subtype": subtype,
        "molecular_weight_g_mol": float(item["molecular_weight_g_mol"]),
        "carbon_atoms": sum(
            atom.GetAtomicNum() == 6
            for atom in Chem.MolFromSmiles(item["smiles"]).GetAtoms()
        ),
        "reference_Tb_K": float(item["Tb_K"]),
        "reference_Tb_method": item["Tb_method"],
        "reference_Tc_K": float(item["Tc_K"]),
        "parameters": parameters,
        "multiplier": multiplier,
        "states": perry_states(item, database),
    }


def oxygenated_records(artifact: dict, database: dict) -> tuple[list[dict], Counter]:
    records = []
    exclusions: Counter[str] = Counter()
    present_cas = set()
    for item in artifact["compounds"]:
        if item["class"] not in {
            "alcohols",
            "organic_acids",
            "ketones",
            "esters",
            "ethers",
        }:
            continue
        selected = selected_oxygenated_model(item)
        if selected is None:
            exclusions[f"excluded {item['class']} structure"] += 1
            continue
        category, parameters, multiplier, subtype = selected
        records.append(
            make_perry_record(
                item, category, parameters, multiplier, subtype, database
            )
        )
        present_cas.add(item["cas"])

    perry = PerryPropertyLibrary()
    for cas, entry in sorted(database.items()):
        if cas in present_cas or not entry.get("liquid_thermal_conductivity"):
            continue
        smiles = local_smiles(cas)
        try:
            molecule = molecule_from_verified_smiles(
                smiles, str(entry.get("formula") or "")
            )
        except ValueError:
            continue
        heavy_elements = {
            atom.GetAtomicNum()
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() != 1
        }
        if heavy_elements - {6, 8}:
            continue
        if sum(atom.GetAtomicNum() == 8 for atom in molecule.GetAtoms()) != 1:
            continue
        if len(molecule.GetSubstructMatches(ALDEHYDE)) != 1:
            continue
        if any(atom.GetIsAromatic() for atom in molecule.GetAtoms()):
            continue
        critical = perry.critical_properties(cas)
        if critical is None:
            continue
        try:
            tb_K, tb_method = boiling_point(perry, cas)
        except ValueError:
            continue
        item = {
            "cas": cas,
            "name": entry.get("name") or cas,
            "formula": entry.get("formula") or "",
            "smiles": smiles,
            "molecular_weight_g_mol": float(
                entry["liquid_thermal_conductivity"][0]["molecular_weight"]
            ),
            "Tb_K": tb_K,
            "Tb_method": tb_method,
            "Tc_K": float(critical["Tc"].value),
        }
        records.append(
            make_perry_record(
                item,
                "aldehydes",
                {"A": ALDEHYDE_A, "a": 1.2, "b": ALDEHYDE_B, "c": 0.167},
                1.0,
                "local_A_b_refit",
                database,
            )
        )
    return records, exclusions


def perry_halocarbon_records(
    artifact: dict, database: dict
) -> tuple[list[dict], Counter]:
    records = []
    exclusions: Counter[str] = Counter()
    perry = PerryPropertyLibrary()
    for item in artifact["compounds"]:
        carbon_atoms = int(item["classification"]["carbon_atoms"])
        if carbon_atoms < 2:
            exclusions["one-carbon halocarbon"] += 1
            continue
        try:
            tb_K, tb_method = boiling_point(perry, item["cas"])
        except ValueError:
            exclusions["normal boiling point unavailable"] += 1
            continue
        record = {
            **item,
            "Tb_K": tb_K,
            "Tb_method": tb_method,
        }
        records.append(
            make_perry_record(
                record,
                "halogenated_hydrocarbons",
                REFRIGERANT_PARAMETERS,
                1.0,
                "C2_plus_general_refrigerant",
                database,
            )
        )
    return records, exclusions


def coolprop_halocarbon_records(artifact: dict, source: str) -> list[dict]:
    if artifact["source"]["version"] != str(CoolProp.__version__):
        raise ValueError("CoolProp artifact version does not match runtime")
    records = []
    for item in artifact["compounds"]:
        details = item.get("details") or item.get("classification")
        if int(details["carbon_atoms"]) < 2:
            continue
        fluid = item["fluid"]
        tb_K = float(
            CP.PropsSI("T", "P", NORMAL_BOILING_PRESSURE_PA, "Q", 0.0, fluid)
        )
        states = [
            (
                float(temperature),
                float(
                    CP.PropsSI(
                        "CONDUCTIVITY", "T", float(temperature), "Q", 0.0, fluid
                    )
                ),
            )
            for temperature in np.linspace(
                float(item["Tmin_K"]), float(item["Tmax_K"]), SAMPLE_POINTS
            )
        ]
        records.append(
            {
                "source": source,
                "cas": item["cas"],
                "name": fluid,
                "formula": item["formula"],
                "smiles": item["smiles"],
                "category": "halogenated_hydrocarbons",
                "subtype": "C2_plus_general_refrigerant",
                "molecular_weight_g_mol": float(item["molecular_weight_g_mol"]),
                "carbon_atoms": int(details["carbon_atoms"]),
                "reference_Tb_K": tb_K,
                "reference_Tb_method": "coolprop_HEOS_normal_boiling_point",
                "reference_Tc_K": float(item["Tc_K"]),
                "parameters": REFRIGERANT_PARAMETERS,
                "multiplier": 1.0,
                "states": states,
            }
        )
    return records


def summarize_records(records: list[dict], error_key: str) -> dict:
    errors = [error for record in records for error in record[error_key]]
    return {"compounds": len(records), **metrics(errors)}


def base_benchmark(records: list[dict]) -> dict:
    for record in records:
        record["base_errors"] = [
            100.0
            * (
                baroncini_value(
                    record,
                    temperature,
                    record["reference_Tb_K"],
                    record["reference_Tc_K"],
                )
                / reference
                - 1.0
            )
            for temperature, reference in record["states"]
        ]
    categories = defaultdict(list)
    for record in records:
        categories[record["category"]].append(record)
    return {
        "overall": summarize_records(records, "base_errors"),
        "categories": {
            category: summarize_records(values, "base_errors")
            for category, values in sorted(categories.items())
        },
        "subtypes": {
            subtype: summarize_records(
                [record for record in records if record["subtype"] == subtype],
                "base_errors",
            )
            for subtype in sorted({record["subtype"] for record in records})
        },
    }


def sensitivity(records: list[dict]) -> dict:
    successful = []
    exclusions = []
    tb_errors = []
    anchored_tc_errors = []
    structure_tc_errors = []
    cases = (
        "reference",
        "nannoolal_Tb",
        "nannoolal_Tc_anchored",
        "nannoolal_structure_Tb_Tc",
    )
    for record in records:
        try:
            structure = nannoolal_method.estimate(record["smiles"])
            anchored = nannoolal_method.estimate(
                record["smiles"], tb=record["reference_Tb_K"]
            )
            if structure.tb_K is None or structure.tc_K is None or anchored.tc_K is None:
                raise ValueError("Nannoolal Tb or Tc unavailable")
            values = {
                "reference": (
                    record["reference_Tb_K"],
                    record["reference_Tc_K"],
                ),
                "nannoolal_Tb": (
                    float(structure.tb_K),
                    record["reference_Tc_K"],
                ),
                "nannoolal_Tc_anchored": (
                    record["reference_Tb_K"],
                    float(anchored.tc_K),
                ),
                "nannoolal_structure_Tb_Tc": (
                    float(structure.tb_K),
                    float(structure.tc_K),
                ),
            }
            maximum_temperature = min(tc for _tb, tc in values.values())
            common_states = [
                state
                for state in record["states"]
                if state[0] <= maximum_temperature * (1.0 + 1.0e-10)
            ]
            if not common_states:
                raise ValueError("no temperature states below every Tc")
            case_errors = {}
            for case, (tb_K, tc_K) in values.items():
                case_errors[case] = [
                    100.0
                    * (baroncini_value(record, temperature, tb_K, tc_K) / reference - 1.0)
                    for temperature, reference in common_states
                ]
        except (
            ValueError,
            TypeError,
            OverflowError,
            nannoolal_method.NannoolalError,
        ) as exc:
            exclusions.append(
                {
                    "source": record["source"],
                    "cas": record["cas"],
                    "name": record["name"],
                    "category": record["category"],
                    "reason": str(exc),
                }
            )
            continue
        tb_errors.append(100.0 * (float(structure.tb_K) / record["reference_Tb_K"] - 1.0))
        anchored_tc_errors.append(
            100.0 * (float(anchored.tc_K) / record["reference_Tc_K"] - 1.0)
        )
        structure_tc_errors.append(
            100.0 * (float(structure.tc_K) / record["reference_Tc_K"] - 1.0)
        )
        successful.append({**record, "sensitivity_errors": case_errors})

    categories = defaultdict(list)
    for record in successful:
        categories[record["category"]].append(record)

    def case_summary(selected: list[dict]) -> dict:
        return {
            case: metrics(
                [
                    error
                    for record in selected
                    for error in record["sensitivity_errors"][case]
                ]
            )
            for case in cases
        }

    return {
        "coverage": {
            "selected_compounds": len(records),
            "common_compounds": len(successful),
            "excluded_compounds": len(exclusions),
        },
        "input_errors": {
            "nannoolal_Tb_vs_reference": input_metrics(tb_errors),
            "nannoolal_Tc_anchored_vs_reference": input_metrics(anchored_tc_errors),
            "nannoolal_Tc_structure_vs_reference": input_metrics(structure_tc_errors),
        },
        "overall": case_summary(successful),
        "categories": {
            category: {"compounds": len(values), **case_summary(values)}
            for category, values in sorted(categories.items())
        },
        "exclusions": exclusions,
    }


def run_benchmark(
    conductivity_path: Path,
    baroncini_path: Path,
    refrigerant_path: Path,
    mixed_path: Path,
    nonmixed_path: Path,
    local_refit_path: Path,
) -> dict:
    database = json.loads(conductivity_path.read_text())["chemicals"]
    baroncini = json.loads(baroncini_path.read_text())
    refrigerants = json.loads(refrigerant_path.read_text())
    mixed = json.loads(mixed_path.read_text())
    nonmixed = json.loads(nonmixed_path.read_text())
    local_refits = json.loads(local_refit_path.read_text())
    oxygenated, oxygen_exclusions = oxygenated_records(baroncini, database)
    perry_halogenated, halogen_exclusions = perry_halocarbon_records(
        refrigerants, database
    )
    records = (
        oxygenated
        + perry_halogenated
        + coolprop_halocarbon_records(mixed, "coolprop_mixed")
        + coolprop_halocarbon_records(nonmixed, "coolprop_nonmixed_perry_absent")
    )
    seen = set()
    duplicates = []
    unique_records = []
    for record in records:
        if record["cas"] in seen:
            duplicates.append(record["cas"])
            continue
        seen.add(record["cas"])
        unique_records.append(record)
    records = unique_records
    base = base_benchmark(records)
    sensitivity_result = sensitivity(records)
    serialized_records = []
    for record in records:
        serialized_records.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"states", "base_errors"}
            }
            | {"base": metrics(record["base_errors"])}
        )
    serialized_records.sort(
        key=lambda item: item["base"]["mape_percent"], reverse=True
    )
    return {
        "domain": {
            "ordinary_hydrocarbons": "excluded; use Modified Pachaiyappan",
            "alcohols": "one nonphenolic alcohol group",
            "organic_acids": "mono- or dicarboxylic; formic acid excluded",
            "ketones": (
                "acyclic aliphatic monoketone local size/position correction; "
                "aromatic monoketone published parameters; cyclic carbonyl excluded"
            ),
            "aldehydes": "single aliphatic aldehyde oxygen; aromatic aldehydes excluded",
            "esters": "monoesters; diesters excluded",
            "ethers": "acyclic; cyclic ethers excluded",
            "halogenated_hydrocarbons": "at least two carbon atoms",
        },
        "local_adjustments": {
            "acyclic_aliphatic_ketone_log_multiplier": {
                "intercept": KETONE_POSITION_LOG_INTERCEPT,
                "log_molecular_weight": KETONE_POSITION_LOG_MW,
                "smaller_carbonyl_side_carbons": KETONE_POSITION_SIDE_MIN,
            },
            "aldehyde": {"A": ALDEHYDE_A, "b": ALDEHYDE_B},
        },
        "local_adjustment_validation": {
            "acyclic_aliphatic_ketones": {
                "compounds": local_refits["ketones"]["compounds"],
                "method": "whole-compound leave-one-out",
                "metrics": local_refits["ketones"]["models"]["A_b_position"][
                    "LOOCV"
                ],
            },
            "aldehydes": {
                "compounds": local_refits["aldehyde_hypothesis"]["compounds"],
                "method": "whole-compound leave-one-out",
                "metrics": local_refits["aldehyde_hypothesis"][
                    "aldehyde_only_models"
                ]["A_and_b"]["LOOCV"],
            },
        },
        "coverage": {
            "compounds": len(records),
            "duplicates_removed": duplicates,
            "oxygenated_exclusions": dict(sorted(oxygen_exclusions.items())),
            "halocarbon_exclusions": dict(sorted(halogen_exclusions.items())),
        },
        "base_benchmark": base,
        "input_sensitivity": sensitivity_result,
        "compounds": serialized_records,
    }


def _line(values: dict) -> str:
    return (
        f"MAPE={values['mape_percent']:6.2f}% "
        f"MdAPE={values['median_ape_percent']:6.2f}% "
        f"P95={values['p95_ape_percent']:6.2f}% "
        f"bias={values['mean_signed_error_percent']:+6.2f}%"
    )


def report(result: dict) -> str:
    base = result["base_benchmark"]
    lines = [
        "Final selected-domain Baroncini benchmark",
        f"coverage: {result['coverage']['compounds']} compounds",
        "overall: " + _line(base["overall"]),
        "",
        "PER CATEGORY",
    ]
    for category, values in base["categories"].items():
        lines.append(
            f"  {category:<28} n={values['compounds']:3d} {_line(values)}"
        )
    lines.extend(("", "LOCAL-ADJUSTMENT HELD-OUT VALIDATION"))
    for category, values in result["local_adjustment_validation"].items():
        lines.append(
            f"  {category:<28} n={values['compounds']:3d} "
            + _line(values["metrics"])
        )
    lines.extend(("", "INPUT SENSITIVITY — EXACT COMMON POPULATION"))
    sensitivity_result = result["input_sensitivity"]
    coverage = sensitivity_result["coverage"]
    lines.append(
        f"  coverage: {coverage['common_compounds']}/{coverage['selected_compounds']} compounds"
    )
    for name, values in sensitivity_result["input_errors"].items():
        lines.append(f"  {name:<40} {_line(values)}")
    lines.extend(("", "  CONDUCTIVITY OVERALL"))
    for case, values in sensitivity_result["overall"].items():
        lines.append(f"    {case:<34} {_line(values)}")
    lines.extend(("", "  CONDUCTIVITY BY CATEGORY (MAPE)"))
    for category, values in sensitivity_result["categories"].items():
        lines.append(
            f"    {category:<28} n={values['compounds']:3d} "
            + " ".join(
                f"{case}={values[case]['mape_percent']:.2f}%"
                for case in (
                    "reference",
                    "nannoolal_Tb",
                    "nannoolal_Tc_anchored",
                    "nannoolal_structure_Tb_Tc",
                )
            )
        )
    if sensitivity_result["exclusions"]:
        lines.extend(("", "NANNOOLAL SENSITIVITY EXCLUSIONS"))
        for item in sensitivity_result["exclusions"]:
            lines.append(
                f"  {item['category']:<28} {item['name']:<28} "
                f"{item['reason']}"
            )
    lines.extend(("", "WORST 20 IN SELECTED DOMAIN"))
    for index, item in enumerate(result["compounds"][:20], 1):
        lines.append(
            f"  {index:2d}. {item['name']:<28} {item['category']:<28} "
            f"MAPE={item['base']['mape_percent']:6.2f}%"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conductivity-database", type=Path, default=DEFAULT_CONDUCTIVITY_DATABASE
    )
    parser.add_argument(
        "--baroncini-artifact", type=Path, default=DEFAULT_BARONCINI_ARTIFACT
    )
    parser.add_argument(
        "--refrigerant-artifact", type=Path, default=DEFAULT_REFRIGERANT_ARTIFACT
    )
    parser.add_argument(
        "--mixed-coolprop-artifact",
        type=Path,
        default=DEFAULT_MIXED_COOLPROP_ARTIFACT,
    )
    parser.add_argument(
        "--nonmixed-coolprop-artifact",
        type=Path,
        default=DEFAULT_NONMIXED_COOLPROP_ARTIFACT,
    )
    parser.add_argument(
        "--local-refit-artifact", type=Path, default=DEFAULT_LOCAL_REFIT_ARTIFACT
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_benchmark(
        args.conductivity_database,
        args.baroncini_artifact,
        args.refrigerant_artifact,
        args.mixed_coolprop_artifact,
        args.nonmixed_coolprop_artifact,
        args.local_refit_artifact,
    )
    script = Path(__file__).resolve()
    dependencies = (
        args.baroncini_artifact,
        args.refrigerant_artifact,
        args.mixed_coolprop_artifact,
        args.nonmixed_coolprop_artifact,
        args.local_refit_artifact,
        ROOT / "baroncini_method.py",
        ROOT / "nannoolal_method.py",
    )
    result["reproducibility"] = {
        "python": platform.python_version(),
        "coolprop_version": str(CoolProp.__version__),
        "script": str(script.relative_to(ROOT)),
        "script_sha256": _sha256(script),
        "dependency_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in dependencies
        },
        "conductivity_database_sha256": _sha256(args.conductivity_database),
        "sample_points_per_curve": SAMPLE_POINTS,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(report(result))


if __name__ == "__main__":
    main()
