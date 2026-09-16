#!/usr/bin/env python3
"""Analyze Baroncini class scope and competing liquid-k methods.

This companion to ``benchmark_baroncini_perry.py`` answers four selection
questions using whole-compound validation:

* whether recommended oxygenated classes retain carbon-count trends;
* whether the monohydric-alcohol size trend admits a transferable correction;
* whether Baroncini or revised Govender performs better by oxygenated class;
* whether hydrocarbons should prefer production Modified Pachaiyappan.

Perry Tb, Tc, molecular weight, V20, and conductivity curves are used to
isolate the correlations.  Alcohol correction fits weight each compound
equally and use leave-one-compound-out validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import govender_method  # noqa: E402
import modified_pachaiyappan  # noqa: E402
from perry_properties import PerryPropertyLibrary  # noqa: E402
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    conductivity_reference,
)
from scripts.thermal_conductivity.liquid.benchmark_baroncini_perry import (  # noqa: E402
    SAMPLE_POINTS,
    conductivity_W_m_K as baroncini_conductivity,
)


RDLogger.DisableLog("rdApp.*")

OXYGENATED_CLASSES = (
    "alcohols",
    "organic_acids",
    "ketones",
    "esters",
    "ethers",
)
HYDROCARBON_CLASSES = (
    "saturated_hydrocarbons",
    "olefins",
    "cycloparaffins",
    "aromatics",
)
DEFAULT_BARONCINI_ARTIFACT = (
    ROOT
    / "scripts"
    / "thermal_conductivity"
    / "liquid"
    / "results"
    / "baroncini_perry_benchmark.json"
)
DEFAULT_CONDUCTIVITY_DATABASE = ROOT / "data" / "perry_thermal_conductivity.json"
DEFAULT_OUTPUT = (
    ROOT
    / "scripts"
    / "thermal_conductivity"
    / "liquid"
    / "results"
    / "baroncini_class_selection_analysis.json"
)


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
    }


def recommended_member(item: dict) -> bool:
    """Apply the oxygenated exclusions established in the initial probe."""
    details = item["details"]
    chemical_class = item["class"]
    if chemical_class == "alcohols":
        return int(details["alcohol_groups"]) == 1
    if chemical_class == "ketones":
        return int(details["rings"]) == 0
    if chemical_class == "esters":
        return int(details["ester_groups"]) == 1
    if chemical_class == "ethers":
        return int(details["rings"]) == 0
    return chemical_class == "organic_acids"


def states(item: dict, conductivity_database: dict) -> list[tuple[float, float]]:
    values = []
    for curve in conductivity_database[item["cas"]]["liquid_thermal_conductivity"]:
        for temperature in np.linspace(
            float(curve["T_min_K"]), float(curve["T_max_K"]), SAMPLE_POINTS
        ):
            temperature = float(temperature)
            if temperature > float(item["Tc_K"]) * (1.0 + 1.0e-10):
                continue
            values.append((temperature, conductivity_reference(curve, temperature)))
    return values


def baroncini_errors(item: dict, state_values: list[tuple[float, float]]) -> list[float]:
    return [
        100.0
        * (
            baroncini_conductivity(
                temperature,
                normal_boiling_temperature_K=float(item["Tb_K"]),
                critical_temperature_K=float(item["Tc_K"]),
                molecular_weight_g_mol=float(item["molecular_weight_g_mol"]),
                chemical_class=item["class"],
            )
            / reference
            - 1.0
        )
        for temperature, reference in state_values
    ]


def carbon_bin(carbon_atoms: int) -> str:
    if carbon_atoms <= 2:
        return "C1_C2"
    if carbon_atoms <= 4:
        return "C3_C4"
    if carbon_atoms <= 6:
        return "C5_C6"
    return "C7_plus"


def pearson(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) < 3 or np.std(values_x) == 0.0 or np.std(values_y) == 0.0:
        return None
    return float(np.corrcoef(values_x, values_y)[0, 1])


def size_analysis(items: list[dict]) -> dict:
    result = {}
    for chemical_class in OXYGENATED_CLASSES:
        members = [
            item
            for item in items
            if item["class"] == chemical_class and recommended_member(item)
        ]
        bins: defaultdict[str, list[dict]] = defaultdict(list)
        for item in members:
            bins[carbon_bin(int(item["details"]["carbon_atoms"]))].append(item)
        result[chemical_class] = {
            "compounds": len(members),
            "carbon_vs_signed_bias_pearson": pearson(
                [float(item["details"]["carbon_atoms"]) for item in members],
                [float(item["errors"]["mean_signed_error_percent"]) for item in members],
            ),
            "carbon_vs_mape_pearson": pearson(
                [float(item["details"]["carbon_atoms"]) for item in members],
                [float(item["errors"]["mape_percent"]) for item in members],
            ),
            "by_carbon_bin": {
                name: {
                    "compounds": len(values),
                    "mean_compound_mape_percent": float(
                        np.mean([item["errors"]["mape_percent"] for item in values])
                    ),
                    "mean_compound_bias_percent": float(
                        np.mean(
                            [
                                item["errors"]["mean_signed_error_percent"]
                                for item in values
                            ]
                        )
                    ),
                    "names": [item["name"] for item in values],
                }
                for name, values in sorted(bins.items())
            },
        }
    return result


def alcohol_features(item: dict, model: str) -> np.ndarray:
    carbon = float(item["details"]["carbon_atoms"])
    if model == "constant":
        return np.asarray([1.0])
    if model == "linear_carbon":
        return np.asarray([1.0, carbon])
    if model == "C6_hinge":
        return np.asarray([1.0, max(0.0, carbon - 5.0)])
    substitution = alcohol_substitution(item["smiles"])
    if model == "OH_substitution":
        return np.asarray(
            [
                1.0,
                float(substitution == "secondary"),
                float(substitution == "tertiary"),
            ]
        )
    if model == "primary_C6_plus":
        return np.asarray(
            [1.0, float(substitution == "primary" and carbon >= 6.0)]
        )
    raise ValueError(model)


def alcohol_substitution(smiles: str) -> str:
    """Classify the carbon bearing the sole aliphatic alcohol group."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid alcohol SMILES")
    matches = molecule.GetSubstructMatches(Chem.MolFromSmarts("[OX2H1][#6;!a;!$(C=O)]"))
    if len(matches) != 1:
        raise ValueError("expected one aliphatic alcohol group")
    carbon = molecule.GetAtomWithIdx(matches[0][1])
    carbon_neighbors = sum(
        neighbor.GetAtomicNum() == 6 for neighbor in carbon.GetNeighbors()
    )
    if carbon_neighbors <= 1:
        return "primary"
    if carbon_neighbors == 2:
        return "secondary"
    return "tertiary"


def alcohol_target(
    item: dict, state_values: list[tuple[float, float]]
) -> float:
    predictions = np.asarray(
        [
            baroncini_conductivity(
                temperature,
                normal_boiling_temperature_K=float(item["Tb_K"]),
                critical_temperature_K=float(item["Tc_K"]),
                molecular_weight_g_mol=float(item["molecular_weight_g_mol"]),
                chemical_class="alcohols",
            )
            for temperature, _reference in state_values
        ]
    )
    references = np.asarray([reference for _temperature, reference in state_values])
    return float(np.mean(np.log(references / predictions)))


def alcohol_correction_analysis(
    items: list[dict], conductivity_database: dict
) -> dict:
    alcohols = [
        item
        for item in items
        if item["class"] == "alcohols" and recommended_member(item)
    ]
    state_map = {item["cas"]: states(item, conductivity_database) for item in alcohols}
    targets = np.asarray(
        [alcohol_target(item, state_map[item["cas"]]) for item in alcohols]
    )
    baseline_errors = [
        error
        for item in alcohols
        for error in baroncini_errors(item, state_map[item["cas"]])
    ]
    models = {}
    substitution_counts = defaultdict(int)
    for item in alcohols:
        substitution_counts[alcohol_substitution(item["smiles"])] += 1
    for model in (
        "constant",
        "linear_carbon",
        "C6_hinge",
        "OH_substitution",
        "primary_C6_plus",
    ):
        feature_matrix = np.asarray([alcohol_features(item, model) for item in alcohols])
        corrected_errors = []
        correction_factors = []
        for held_out, item in enumerate(alcohols):
            training = np.arange(len(alcohols)) != held_out
            coefficients = np.linalg.lstsq(
                feature_matrix[training], targets[training], rcond=None
            )[0]
            correction_factor = float(
                math.exp(feature_matrix[held_out] @ coefficients)
            )
            correction_factors.append(correction_factor)
            for error in baroncini_errors(item, state_map[item["cas"]]):
                corrected_errors.append(
                    100.0 * (correction_factor * (1.0 + error / 100.0) - 1.0)
                )
        full_coefficients = np.linalg.lstsq(
            feature_matrix, targets, rcond=None
        )[0]
        models[model] = {
            "parameters": len(full_coefficients),
            "full_fit_log_coefficients": [float(value) for value in full_coefficients],
            "LOOCV_correction_factor_min": min(correction_factors),
            "LOOCV_correction_factor_max": max(correction_factors),
            "LOOCV": metrics(corrected_errors),
        }
    return {
        "compounds": len(alcohols),
        "substitution_counts": dict(sorted(substitution_counts.items())),
        "baseline": metrics(baseline_errors),
        "models": models,
    }


def govender_comparison(
    items: list[dict], conductivity_database: dict
) -> dict:
    by_class: defaultdict[str, list[dict]] = defaultdict(list)
    exclusions = []
    for item in items:
        if item["class"] not in OXYGENATED_CLASSES or not recommended_member(item):
            continue
        state_values = states(item, conductivity_database)
        try:
            estimate = govender_method.estimate(
                item["smiles"], tb=float(item["Tb_K"]), local_refits=True
            )
            govender_errors = [
                100.0
                * (estimate.conductivity_W_m_K(temperature) / reference - 1.0)
                for temperature, reference in state_values
            ]
        except (ValueError, TypeError, OverflowError, govender_method.GovenderError) as exc:
            exclusions.append(
                {"cas": item["cas"], "name": item["name"], "reason": str(exc)}
            )
            continue
        by_class[item["class"]].append(
            {
                "name": item["name"],
                "baroncini": baroncini_errors(item, state_values),
                "govender": govender_errors,
            }
        )
    summaries = {}
    for chemical_class, records in by_class.items():
        baroncini_values = [error for record in records for error in record["baroncini"]]
        govender_values = [error for record in records for error in record["govender"]]
        summaries[chemical_class] = {
            "compounds": len(records),
            "baroncini": metrics(baroncini_values),
            "govender": metrics(govender_values),
            "baroncini_compound_wins": int(sum(
                np.mean(np.abs(record["baroncini"]))
                < np.mean(np.abs(record["govender"]))
                for record in records
            )),
            "govender_compound_wins": int(sum(
                np.mean(np.abs(record["govender"]))
                < np.mean(np.abs(record["baroncini"]))
                for record in records
            )),
        }
    return {"classes": summaries, "exclusions": exclusions}


def hydrocarbon_comparison(
    items: list[dict], conductivity_database: dict, perry: PerryPropertyLibrary
) -> dict:
    records = []
    exclusions = []
    for item in items:
        if item["class"] not in HYDROCARBON_CLASSES:
            continue
        molecule = Chem.MolFromSmiles(item["smiles"])
        try:
            descriptors = modified_pachaiyappan.describe_hydrocarbon(molecule)
            volume = perry.liquid_molar_volume_m3_per_kmol(item["cas"], 293.15)
            if volume is None:
                raise ValueError("Perry V20 unavailable")
            volume_cm3_mol = float(volume.value) * 1000.0
        except (ValueError, modified_pachaiyappan.ModifiedPachaiyappanError) as exc:
            exclusions.append(
                {"cas": item["cas"], "name": item["name"], "reason": str(exc)}
            )
            continue
        state_values = states(item, conductivity_database)
        baroncini_values = baroncini_errors(item, state_values)
        pachaiyappan_values = []
        for temperature, reference in state_values:
            predicted = modified_pachaiyappan.conductivity_W_m_K(
                temperature,
                molecular_weight_g_mol=float(item["molecular_weight_g_mol"]),
                critical_temperature_K=float(item["Tc_K"]),
                molar_volume_20C_cm3_mol=volume_cm3_mol,
                straight_chain=descriptors.straight_chain,
            )
            pachaiyappan_values.append(100.0 * (predicted / reference - 1.0))
        records.append(
            {
                "name": item["name"],
                "class": item["class"],
                "baroncini": baroncini_values,
                "pachaiyappan": pachaiyappan_values,
            }
        )
    baroncini_values = [error for record in records for error in record["baroncini"]]
    pachaiyappan_values = [
        error for record in records for error in record["pachaiyappan"]
    ]
    by_class = {}
    for chemical_class in HYDROCARBON_CLASSES:
        subset = [record for record in records if record["class"] == chemical_class]
        if not subset:
            continue
        by_class[chemical_class] = {
            "compounds": len(subset),
            "baroncini": metrics(
                [error for record in subset for error in record["baroncini"]]
            ),
            "pachaiyappan": metrics(
                [error for record in subset for error in record["pachaiyappan"]]
            ),
        }
    return {
        "compounds": len(records),
        "baroncini": metrics(baroncini_values),
        "pachaiyappan": metrics(pachaiyappan_values),
        "baroncini_compound_wins": int(sum(
            np.mean(np.abs(record["baroncini"]))
            < np.mean(np.abs(record["pachaiyappan"]))
            for record in records
        )),
        "pachaiyappan_compound_wins": int(sum(
            np.mean(np.abs(record["pachaiyappan"]))
            < np.mean(np.abs(record["baroncini"]))
            for record in records
        )),
        "by_class": by_class,
        "exclusions": exclusions,
    }


def run_analysis(artifact_path: Path, conductivity_path: Path) -> dict:
    artifact = json.loads(artifact_path.read_text())
    conductivity_database = json.loads(conductivity_path.read_text())["chemicals"]
    items = artifact["compounds"]
    return {
        "recommended_domain": {
            "alcohols": "one alcohol group; phenols already excluded",
            "organic_acids": "mono- and dicarboxylic acids",
            "ketones": "acyclic ketones",
            "esters": "monoesters",
            "ethers": "acyclic ethers",
        },
        "oxygenated_size_analysis": size_analysis(items),
        "alcohol_correction": alcohol_correction_analysis(
            items, conductivity_database
        ),
        "oxygenated_baroncini_vs_govender": govender_comparison(
            items, conductivity_database
        ),
        "hydrocarbon_baroncini_vs_modified_pachaiyappan": hydrocarbon_comparison(
            items, conductivity_database, PerryPropertyLibrary()
        ),
    }


def _line(values: dict) -> str:
    return (
        f"MAPE={values['mape_percent']:6.2f}% "
        f"MdAPE={values['median_ape_percent']:6.2f}% "
        f"P95={values['p95_ape_percent']:6.2f}% "
        f"bias={values['mean_signed_error_percent']:+6.2f}%"
    )


def report(result: dict) -> str:
    lines = ["Baroncini class-selection analysis", "", "OXYGENATED SIZE TRENDS"]
    for chemical_class, values in result["oxygenated_size_analysis"].items():
        lines.append(
            f"  {chemical_class}: n={values['compounds']} "
            f"r(C,bias)={values['carbon_vs_signed_bias_pearson']} "
            f"r(C,MAPE)={values['carbon_vs_mape_pearson']}"
        )
        for bin_name, bin_values in values["by_carbon_bin"].items():
            lines.append(
                f"    {bin_name:<8} n={bin_values['compounds']:2d} "
                f"MAPE={bin_values['mean_compound_mape_percent']:6.2f}% "
                f"bias={bin_values['mean_compound_bias_percent']:+6.2f}%"
            )
    alcohol = result["alcohol_correction"]
    lines.extend(
        (
            "",
            f"ALCOHOL CORRECTION, {alcohol['compounds']} MONOHYDRIC COMPOUNDS",
            "  baseline: " + _line(alcohol["baseline"]),
        )
    )
    for name, values in alcohol["models"].items():
        lines.append(f"  {name:<15} LOOCV {_line(values['LOOCV'])}")
    lines.extend(("", "OXYGENATED BARONCINI VS GOVENDER"))
    comparison = result["oxygenated_baroncini_vs_govender"]
    for chemical_class, values in comparison["classes"].items():
        lines.extend(
            (
                f"  {chemical_class}: n={values['compounds']}",
                "    Baroncini: " + _line(values["baroncini"]),
                "    Govender:  " + _line(values["govender"]),
                f"    wins {values['baroncini_compound_wins']}:{values['govender_compound_wins']}",
            )
        )
    hydrocarbon = result["hydrocarbon_baroncini_vs_modified_pachaiyappan"]
    lines.extend(
        (
            "",
            f"HYDROCARBONS, EXACT COMMON SET n={hydrocarbon['compounds']}",
            "  Baroncini:             " + _line(hydrocarbon["baroncini"]),
            "  Modified Pachaiyappan: " + _line(hydrocarbon["pachaiyappan"]),
            f"  wins {hydrocarbon['baroncini_compound_wins']}:{hydrocarbon['pachaiyappan_compound_wins']}",
        )
    )
    for chemical_class, values in hydrocarbon["by_class"].items():
        lines.append(
            f"    {chemical_class:<24} n={values['compounds']:2d} "
            f"Baroncini={values['baroncini']['mape_percent']:6.2f}% "
            f"Pachaiyappan={values['pachaiyappan']['mape_percent']:6.2f}%"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baroncini-artifact", type=Path, default=DEFAULT_BARONCINI_ARTIFACT
    )
    parser.add_argument(
        "--conductivity-database",
        type=Path,
        default=DEFAULT_CONDUCTIVITY_DATABASE,
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_analysis(args.baroncini_artifact, args.conductivity_database)
    script = Path(__file__).resolve()
    dependencies = (
        args.baroncini_artifact,
        ROOT / "govender_method.py",
        ROOT / "modified_pachaiyappan.py",
    )
    result["reproducibility"] = {
        "python": platform.python_version(),
        "script": str(script.relative_to(ROOT)),
        "script_sha256": _sha256(script),
        "dependency_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in dependencies
        },
        "conductivity_database_sha256": _sha256(args.conductivity_database),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(report(result))


if __name__ == "__main__":
    main()
