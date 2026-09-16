#!/usr/bin/env python3
"""Probe transferable repairs for ketones and optional aldehyde applicability.

The published ketone parameters are A=0.00383 and b=0.5.  Candidate repairs
fit compound-equal logarithmic scale residuals and are evaluated by
leave-one-compound-out validation.  The primary candidate refits only A and b;
additional models test whether branching or carbonyl position explains error
that a molecular-size refit does not.  Because Baroncini publishes no aldehyde
row, a separate section tests the published ketone parameters on structurally
pure aldehydes, transfers the ketone refit without using aldehyde data, and
reports aldehyde-only A and A,b leave-one-compound-out fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
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
from scripts.thermal_conductivity.liquid.benchmark_baroncini_perry import (  # noqa: E402
    CLASS_PARAMETERS,
    SAMPLE_POINTS,
    conductivity_W_m_K,
)


RDLogger.DisableLog("rdApp.*")

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
    / "baroncini_ketone_refit_probe.json"
)
KETONE = Chem.MolFromSmarts("[CX3](=[OX1])([#6])[#6]")
ALDEHYDE = Chem.MolFromSmarts("[CX3H1](=[OX1])[#6]")
BENZALDEHYDE_EXTERNAL = {
    "cas": "100-52-7",
    "name": "Benzaldehyde",
    "smiles": "O=Cc1ccccc1",
    "molecular_weight_g_mol": 106.122,
    "Tb_K": 452.15,
    "Tb_source": "Perry 9th Table 2-10",
    "Tc_K": 693.0,
    "Tc_source": "ACS JCED 2015/IUPAC critical-property review Table 1",
    "observations": (
        (12.78, 0.15288),
        (15.56, 0.15187),
        (18.33, 0.15101),
        (21.11, 0.15000),
        (23.89, 0.14899),
        (26.67, 0.14798),
        (29.44, 0.14711),
        (32.22, 0.14610),
        (35.00, 0.14509),
        (37.78, 0.14423),
        (40.56, 0.14322),
        (43.33, 0.14221),
        (46.11, 0.14120),
        (48.89, 0.14033),
        (51.67, 0.13932),
        (54.44, 0.13831),
        (57.22, 0.13730),
        (60.00, 0.13644),
        (62.78, 0.13543),
        (65.56, 0.13442),
    ),
}


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


def ketone_descriptors(smiles: str) -> dict[str, float | bool]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid ketone SMILES")
    matches = molecule.GetSubstructMatches(KETONE)
    if len(matches) != 1:
        raise ValueError("expected exactly one ketone group")
    carbonyl_index = matches[0][0]
    carbonyl = molecule.GetAtomWithIdx(carbonyl_index)
    attached_carbons = [
        neighbor.GetIdx()
        for neighbor in carbonyl.GetNeighbors()
        if neighbor.GetAtomicNum() == 6
    ]
    if len(attached_carbons) != 2:
        raise ValueError("ketone carbonyl does not have two carbon neighbors")

    def side_carbons(start: int) -> int:
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

    sides = sorted(side_carbons(index) for index in attached_carbons)
    branched = any(
        sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) > 2
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6 and atom.GetIdx() != carbonyl_index
    )
    return {
        "side_min_carbons": float(sides[0]),
        "side_max_carbons": float(sides[1]),
        "branched": branched,
    }


def compound_states(item: dict, database: dict) -> list[tuple[float, float, float]]:
    values = []
    for curve in database[item["cas"]]["liquid_thermal_conductivity"]:
        for temperature in np.linspace(
            float(curve["T_min_K"]), float(curve["T_max_K"]), SAMPLE_POINTS
        ):
            temperature = float(temperature)
            if temperature > float(item["Tc_K"]) * (1.0 + 1.0e-10):
                continue
            reference = conductivity_reference(curve, temperature)
            prediction = conductivity_W_m_K(
                temperature,
                normal_boiling_temperature_K=float(item["Tb_K"]),
                critical_temperature_K=float(item["Tc_K"]),
                molecular_weight_g_mol=float(item["molecular_weight_g_mol"]),
                chemical_class="ketones",
            )
            values.append((temperature, reference, prediction))
    return values


def features(item: dict, model: str) -> np.ndarray:
    log_molecular_weight = math.log(float(item["molecular_weight_g_mol"]))
    if model == "A_only":
        return np.asarray([1.0])
    if model == "A_and_b":
        return np.asarray([1.0, log_molecular_weight])
    descriptors = item["ketone_descriptors"]
    if model == "A_b_branch":
        return np.asarray([1.0, log_molecular_weight, float(descriptors["branched"])])
    if model == "A_b_position":
        return np.asarray(
            [1.0, log_molecular_weight, float(descriptors["side_min_carbons"])]
        )
    if model == "A_b_position_branch":
        return np.asarray(
            [
                1.0,
                log_molecular_weight,
                float(descriptors["side_min_carbons"]),
                float(descriptors["branched"]),
            ]
        )
    raise ValueError(model)


def target(states: list[tuple[float, float, float]]) -> float:
    return float(np.mean([math.log(reference / prediction) for _, reference, prediction in states]))


def evaluate_model(items: list[dict], model: str) -> dict:
    feature_matrix = np.asarray([features(item, model) for item in items])
    targets = np.asarray([item["target"] for item in items])
    corrected_errors = []
    compound_results = []
    held_out_coefficients = []
    for held_out, item in enumerate(items):
        training = np.arange(len(items)) != held_out
        coefficients = np.linalg.lstsq(
            feature_matrix[training], targets[training], rcond=None
        )[0]
        held_out_coefficients.append(coefficients)
        factor = float(math.exp(feature_matrix[held_out] @ coefficients))
        errors = [
            100.0 * (factor * prediction / reference - 1.0)
            for _temperature, reference, prediction in item["states"]
        ]
        corrected_errors.extend(errors)
        compound_results.append(
            {
                "name": item["name"],
                "cas": item["cas"],
                "factor": factor,
                "errors": metrics(errors),
            }
        )
    full_coefficients = np.linalg.lstsq(feature_matrix, targets, rcond=None)[0]
    result = {
        "parameters": len(full_coefficients),
        "full_fit_log_coefficients": [float(value) for value in full_coefficients],
        "LOOCV_log_coefficient_ranges": [
            {
                "minimum": float(value),
                "maximum": float(maximum),
            }
            for value, maximum in zip(
                np.min(np.asarray(held_out_coefficients), axis=0),
                np.max(np.asarray(held_out_coefficients), axis=0),
                strict=True,
            )
        ],
        "LOOCV": metrics(corrected_errors),
        "compounds": compound_results,
    }
    if model in {"A_only", "A_and_b"}:
        published = CLASS_PARAMETERS["ketones"]
        result["full_fit_A"] = float(
            published["A"] * math.exp(full_coefficients[0])
        )
        result["full_fit_b"] = (
            float(published["b"] - full_coefficients[1])
            if model == "A_and_b"
            else float(published["b"])
        )
        if model == "A_and_b":
            held_out = np.asarray(held_out_coefficients)
            result["LOOCV_A_range"] = [
                float(published["A"] * math.exp(np.min(held_out[:, 0]))),
                float(published["A"] * math.exp(np.max(held_out[:, 0]))),
            ]
            result["LOOCV_b_range"] = [
                float(published["b"] - np.max(held_out[:, 1])),
                float(published["b"] - np.min(held_out[:, 1])),
            ]
    return result


def transferred_errors(items: list[dict], coefficients: list[float]) -> dict:
    """Apply a ketone-derived log-scale correction without fitting aldehydes."""
    errors = []
    compound_results = []
    for item in items:
        vector = np.asarray([1.0, math.log(float(item["molecular_weight_g_mol"]))])
        factor = float(math.exp(vector @ np.asarray(coefficients[:2])))
        compound_errors = [
            100.0 * (factor * prediction / reference - 1.0)
            for _temperature, reference, prediction in item["states"]
        ]
        errors.extend(compound_errors)
        compound_results.append(
            {
                "name": item["name"],
                "cas": item["cas"],
                "factor": factor,
                "errors": metrics(compound_errors),
            }
        )
    return {"errors": metrics(errors), "compounds": compound_results}


def aldehyde_items(database: dict) -> list[dict]:
    """Build the pure single-aldehyde Perry population omitted by class table."""
    perry = PerryPropertyLibrary()
    items = []
    for cas, entry in sorted(database.items()):
        curves = entry.get("liquid_thermal_conductivity") or []
        if not curves:
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
        critical = perry.critical_properties(cas)
        if critical is None:
            continue
        try:
            boiling_temperature, boiling_method = boiling_point(perry, cas)
        except ValueError:
            continue
        item = {
            "cas": cas,
            "name": entry.get("name") or cas,
            "formula": entry.get("formula") or "",
            "smiles": smiles,
            "Tb_K": boiling_temperature,
            "Tb_method": boiling_method,
            "Tc_K": float(critical["Tc"].value),
            "molecular_weight_g_mol": float(curves[0]["molecular_weight"]),
            "carbon_atoms": sum(
                atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()
            ),
            "unsaturated": any(
                bond.GetBeginAtom().GetAtomicNum() == 6
                and bond.GetEndAtom().GetAtomicNum() == 6
                and bond.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
                for bond in molecule.GetBonds()
            ),
        }
        item["states"] = compound_states(item, database)
        item["target"] = target(item["states"])
        items.append(item)
    return items


def benzaldehyde_external_validation(
    aldehyde_A: float,
    aldehyde_b: float,
    ketone_refit_A: float,
    ketone_refit_b: float,
) -> dict:
    """Evaluate fixed aldehyde parameters on supplied benzaldehyde data."""
    temperatures = np.asarray(
        [value[0] + 273.15 for value in BENZALDEHYDE_EXTERNAL["observations"]]
    )
    references = np.asarray(
        [value[1] for value in BENZALDEHYDE_EXTERNAL["observations"]]
    )
    def predictions(A: float, b: float) -> np.ndarray:
        return np.asarray(
            [
                conductivity_W_m_K(
                    float(temperature),
                    normal_boiling_temperature_K=BENZALDEHYDE_EXTERNAL["Tb_K"],
                    critical_temperature_K=BENZALDEHYDE_EXTERNAL["Tc_K"],
                    molecular_weight_g_mol=BENZALDEHYDE_EXTERNAL[
                        "molecular_weight_g_mol"
                    ],
                    chemical_class="ketones",
                )
                * (
                    A / CLASS_PARAMETERS["ketones"]["A"]
                    * BENZALDEHYDE_EXTERNAL["molecular_weight_g_mol"]
                    ** (CLASS_PARAMETERS["ketones"]["b"] - b)
                )
                for temperature in temperatures
            ]
        )

    variants = {
        "published_ketone_parameters": predictions(
            CLASS_PARAMETERS["ketones"]["A"], CLASS_PARAMETERS["ketones"]["b"]
        ),
        "ketone_A_b_refit_transferred": predictions(
            ketone_refit_A, ketone_refit_b
        ),
        "aldehyde_A_b_refit": predictions(aldehyde_A, aldehyde_b),
    }
    reference_slope = np.polyfit(temperatures, references, 1)[0]
    variant_results = {}
    for name, predicted_values in variants.items():
        errors = 100.0 * (predicted_values / references - 1.0)
        predicted_slope = np.polyfit(temperatures, predicted_values, 1)[0]
        variant_results[name] = {
            "errors": metrics(errors.tolist()),
            "predicted_slope_W_m_K2": float(predicted_slope),
            "slope_ratio_predicted_over_reference": float(
                predicted_slope / reference_slope
            ),
            "predictions": [
                {
                    "temperature_C": float(temperature - 273.15),
                    "reference_W_m_K": float(reference),
                    "predicted_W_m_K": float(prediction),
                    "signed_error_percent": float(error),
                }
                for temperature, reference, prediction, error in zip(
                    temperatures,
                    references,
                    predicted_values,
                    errors,
                    strict=True,
                )
            ],
        }
    return {
        key: value
        for key, value in BENZALDEHYDE_EXTERNAL.items()
        if key != "observations"
    } | {
        "points": len(temperatures),
        "Tmin_K": float(np.min(temperatures)),
        "Tmax_K": float(np.max(temperatures)),
        "reference_slope_W_m_K2": float(reference_slope),
        "variants": variant_results,
    }


def run_probe(artifact_path: Path, conductivity_path: Path) -> dict:
    artifact = json.loads(artifact_path.read_text())
    database = json.loads(conductivity_path.read_text())["chemicals"]
    items = []
    for source in artifact["compounds"]:
        if source["class"] != "ketones" or int(source["details"]["rings"]) != 0:
            continue
        item = dict(source)
        item["ketone_descriptors"] = ketone_descriptors(item["smiles"])
        item["states"] = compound_states(item, database)
        item["target"] = target(item["states"])
        items.append(item)
    baseline_errors = [
        100.0 * (prediction / reference - 1.0)
        for item in items
        for _temperature, reference, prediction in item["states"]
    ]
    models = {
        model: evaluate_model(items, model)
        for model in (
            "A_only",
            "A_and_b",
            "A_b_branch",
            "A_b_position",
            "A_b_position_branch",
        )
    }
    aldehydes = aldehyde_items(database)
    aldehyde_baseline_errors = [
        100.0 * (prediction / reference - 1.0)
        for item in aldehydes
        for _temperature, reference, prediction in item["states"]
    ]
    ketone_A_b_transfer = transferred_errors(
        aldehydes, models["A_and_b"]["full_fit_log_coefficients"]
    )
    ketone_position_transfer = transferred_errors(
        aldehydes, models["A_b_position"]["full_fit_log_coefficients"]
    )
    aldehyde_models = {
        model: evaluate_model(aldehydes, model)
        for model in ("A_only", "A_and_b")
    }
    return {
        "ketones": {
            "compounds": len(items),
            "published": {
                "A": CLASS_PARAMETERS["ketones"]["A"],
                "b": CLASS_PARAMETERS["ketones"]["b"],
                "errors": metrics(baseline_errors),
            },
            "models": models,
            "compound_descriptors": [
                {
                    "name": item["name"],
                    "cas": item["cas"],
                    "molecular_weight_g_mol": item["molecular_weight_g_mol"],
                    **item["ketone_descriptors"],
                    "published_errors": item["errors"],
                }
                for item in items
            ],
        },
        "aldehyde_hypothesis": {
            "compounds": len(aldehydes),
            "published_ketone_parameters": metrics(aldehyde_baseline_errors),
            "ketone_A_b_refit_transferred": ketone_A_b_transfer,
            "ketone_position_refit_transferred_with_side_min_zero": (
                ketone_position_transfer
            ),
            "aldehyde_only_models": aldehyde_models,
            "compound_descriptors": [
                {
                    "name": item["name"],
                    "cas": item["cas"],
                    "carbon_atoms": item["carbon_atoms"],
                    "unsaturated": item["unsaturated"],
                }
                for item in aldehydes
            ],
        },
        "benzaldehyde_external_validation": benzaldehyde_external_validation(
            aldehyde_models["A_and_b"]["full_fit_A"],
            aldehyde_models["A_and_b"]["full_fit_b"],
            models["A_and_b"]["full_fit_A"],
            models["A_and_b"]["full_fit_b"],
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
    ketones = result["ketones"]
    lines = [
        "Baroncini acyclic-ketone refit probe",
        f"compounds: {ketones['compounds']}",
        "published: " + _line(ketones["published"]["errors"]),
        "",
        "WHOLE-COMPOUND LEAVE-ONE-OUT",
    ]
    for name, values in ketones["models"].items():
        detail = ""
        if name in {"A_only", "A_and_b"}:
            detail = f" A={values['full_fit_A']:.8g} b={values['full_fit_b']:.6g}"
        lines.append(f"  {name:<22} {_line(values['LOOCV'])}{detail}")
    lines.extend(("", "A+b LOOCV BY COMPOUND"))
    for item in sorted(
        ketones["models"]["A_and_b"]["compounds"],
        key=lambda value: value["errors"]["mape_percent"],
        reverse=True,
    ):
        lines.append(
            f"  {item['name']:<28} factor={item['factor']:.5f} "
            f"MAPE={item['errors']['mape_percent']:6.2f}%"
        )
    aldehydes = result["aldehyde_hypothesis"]
    lines.extend(
        (
            "",
            f"OPTIONAL ALDEHYDE HYPOTHESIS, n={aldehydes['compounds']}",
            "  published ketone parameters: "
            + _line(aldehydes["published_ketone_parameters"]),
            "  transferred ketone A,b refit: "
            + _line(aldehydes["ketone_A_b_refit_transferred"]["errors"]),
            "  transferred ketone position refit (side=0): "
            + _line(
                aldehydes[
                    "ketone_position_refit_transferred_with_side_min_zero"
                ]["errors"]
            ),
            "  aldehyde-only A LOOCV: "
            + _line(aldehydes["aldehyde_only_models"]["A_only"]["LOOCV"]),
            "  aldehyde-only A,b LOOCV: "
            + _line(aldehydes["aldehyde_only_models"]["A_and_b"]["LOOCV"]),
        )
    )
    benzaldehyde = result["benzaldehyde_external_validation"]
    lines.extend(
        (
            "",
            "EXTERNAL BENZALDEHYDE VALIDATION — FIXED ALDEHYDE PARAMETERS",
            f"  points={benzaldehyde['points']} "
            f"T={benzaldehyde['Tmin_K']:.2f}–{benzaldehyde['Tmax_K']:.2f} K",
        )
    )
    for name, values in benzaldehyde["variants"].items():
        lines.append(
            f"  {name:<34} {_line(values['errors'])} "
            f"slope ratio={values['slope_ratio_predicted_over_reference']:.4f}"
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
    result = run_probe(args.baroncini_artifact, args.conductivity_database)
    script = Path(__file__).resolve()
    result["reproducibility"] = {
        "python": platform.python_version(),
        "script": str(script.relative_to(ROOT)),
        "script_sha256": _sha256(script),
        "baroncini_artifact_sha256": _sha256(args.baroncini_artifact),
        "conductivity_database_sha256": _sha256(args.conductivity_database),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(report(result))


if __name__ == "__main__":
    main()
