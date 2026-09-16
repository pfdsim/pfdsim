#!/usr/bin/env python3
"""Probe simple structure models for hydrocarbon molar volume at 20 degrees C.

Candidate linear models are fitted by relative-error weighted least squares and
evaluated by leave-one-compound-out validation.  Their held-out volume
predictions are also propagated through ``benchmark_hydrocarbon_model_perry``.
The selected C/H/ring model is finally tested on Perry hydrocarbons whose
liquid-density correlation does not cover 293.15 K; those conductivity curves
are not used to fit the volume model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perry_properties import PerryPropertyLibrary  # noqa: E402
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    conductivity_reference,
)
from scripts.thermal_conductivity.liquid.benchmark_hydrocarbon_model_perry import (  # noqa: E402
    REFERENCE_TEMPERATURE_K,
    SAMPLE_POINTS,
    conductivity,
    is_hydrocarbon,
    is_straight_chain,
    local_smiles,
    molecule_from_verified_smiles,
)


RDLogger.DisableLog("rdApp.*")

DEFAULT_BENCHMARK = (
    ROOT
    / "scripts"
    / "thermal_conductivity"
    / "liquid"
    / "results"
    / "hydrocarbon_model_perry_benchmark.json"
)
DEFAULT_CONDUCTIVITY_DATABASE = ROOT / "data" / "perry_thermal_conductivity.json"
DEFAULT_OUTPUT = (
    ROOT
    / "scripts"
    / "thermal_conductivity"
    / "liquid"
    / "results"
    / "hydrocarbon_v20_estimation_probe.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptors(molecule: Chem.Mol) -> dict[str, int]:
    carbon_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6]
    return {
        "carbon": len(carbon_atoms),
        "hydrogen": sum(atom.GetTotalNumHs() for atom in molecule.GetAtoms()),
        "branch_excess": sum(
            max(
                0,
                sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors())
                - 2,
            )
            for atom in carbon_atoms
        ),
        "rings": len(molecule.GetRingInfo().AtomRings()),
        "ring_carbons": sum(atom.IsInRing() for atom in carbon_atoms),
    }


FEATURES: dict[str, Callable[[dict[str, int]], list[float]]] = {
    "carbon_affine": lambda d: [1.0, d["carbon"]],
    "carbon_hydrogen_affine": lambda d: [1.0, d["carbon"], d["hydrogen"]],
    "carbon_hydrogen_branch": lambda d: [
        1.0,
        d["carbon"],
        d["hydrogen"],
        d["branch_excess"],
    ],
    "carbon_hydrogen_ring": lambda d: [
        1.0,
        d["carbon"],
        d["hydrogen"],
        d["rings"],
    ],
    "carbon_hydrogen_branch_ring": lambda d: [
        1.0,
        d["carbon"],
        d["hydrogen"],
        d["branch_excess"],
        d["rings"],
    ],
    "carbon_hydrogen_ring_carbons": lambda d: [
        1.0,
        d["carbon"],
        d["hydrogen"],
        d["ring_carbons"],
    ],
}


def fit_relative_linear(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Fit a linear volume model by least squares in relative residuals."""
    return np.linalg.lstsq(
        features / targets[:, np.newaxis],
        np.ones(len(targets)),
        rcond=None,
    )[0]


def error_metrics(errors: list[float] | np.ndarray) -> dict[str, float | int]:
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


def conductivity_errors(
    compound: dict,
    volume_cm3_mol: float,
    conductivity_database: dict,
) -> list[float]:
    errors = []
    for curve in conductivity_database[compound["cas"]][
        "liquid_thermal_conductivity"
    ]:
        for temperature in np.linspace(
            float(curve["T_min_K"]), float(curve["T_max_K"]), SAMPLE_POINTS
        ):
            temperature = float(temperature)
            if temperature > compound["Tc_K"] * (1.0 + 1.0e-10):
                continue
            reference = conductivity_reference(curve, temperature)
            predicted = conductivity(
                temperature,
                molecular_weight_g_mol=compound["molecular_weight_g_mol"],
                critical_temperature_K=compound["Tc_K"],
                molar_volume_20C_cm3_mol=volume_cm3_mol,
                straight_chain=compound["straight_chain"],
            )
            errors.append(100.0 * (predicted / reference - 1.0))
    return errors


def known_volume_rows(benchmark: dict) -> list[dict]:
    rows = []
    for compound in benchmark["compounds"]:
        molecule = Chem.MolFromSmiles(compound["smiles"])
        if molecule is None:
            continue
        rows.append(
            {
                **compound,
                "descriptors": descriptors(molecule),
            }
        )
    return rows


def evaluate_candidate(
    rows: list[dict],
    feature_function: Callable[[dict[str, int]], list[float]],
    conductivity_database: dict,
) -> tuple[dict, np.ndarray]:
    feature_matrix = np.asarray(
        [feature_function(row["descriptors"]) for row in rows], dtype=float
    )
    targets = np.asarray([row["V20_cm3_mol"] for row in rows], dtype=float)
    predictions = []
    conductivity_error_values = []
    for held_out in range(len(rows)):
        training = np.arange(len(rows)) != held_out
        coefficients = fit_relative_linear(
            feature_matrix[training], targets[training]
        )
        prediction = float(feature_matrix[held_out] @ coefficients)
        predictions.append(prediction)
        conductivity_error_values.extend(
            conductivity_errors(
                rows[held_out], prediction, conductivity_database
            )
        )
    volume_errors = 100.0 * (np.asarray(predictions) / targets - 1.0)
    return (
        {
            "parameters": feature_matrix.shape[1],
            "volume_LOOCV": error_metrics(volume_errors),
            "conductivity_LOOCV": error_metrics(conductivity_error_values),
        },
        volume_errors,
    )


def unavailable_volume_rows(
    conductivity_database: dict,
    perry: PerryPropertyLibrary,
) -> list[dict]:
    rows = []
    for cas, entry in sorted(conductivity_database.items()):
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
        if not is_hydrocarbon(molecule):
            continue
        critical = perry.critical_properties(cas)
        if critical is None:
            continue
        critical_temperature = float(critical["Tc"].value)
        if critical_temperature <= REFERENCE_TEMPERATURE_K:
            continue
        if perry.liquid_molar_volume_m3_per_kmol(cas, REFERENCE_TEMPERATURE_K):
            continue
        rows.append(
            {
                "cas": cas,
                "name": entry.get("name") or cas,
                "formula": entry.get("formula") or "",
                "smiles": smiles,
                "descriptors": descriptors(molecule),
                "straight_chain": is_straight_chain(molecule),
                "Tc_K": critical_temperature,
                "molecular_weight_g_mol": float(curves[0]["molecular_weight"]),
            }
        )
    return rows


def run_probe(benchmark_path: Path, conductivity_path: Path) -> dict:
    benchmark = json.loads(benchmark_path.read_text())
    conductivity_database = json.loads(conductivity_path.read_text())["chemicals"]
    rows = known_volume_rows(benchmark)
    candidates = {}
    for name, feature_function in FEATURES.items():
        candidates[name], _ = evaluate_candidate(
            rows, feature_function, conductivity_database
        )

    selected_rows = [row for row in rows if row["descriptors"]["carbon"] >= 3]
    selected_function = FEATURES["carbon_hydrogen_ring"]
    selected_validation, selected_errors = evaluate_candidate(
        selected_rows, selected_function, conductivity_database
    )
    feature_matrix = np.asarray(
        [selected_function(row["descriptors"]) for row in selected_rows],
        dtype=float,
    )
    targets = np.asarray(
        [row["V20_cm3_mol"] for row in selected_rows], dtype=float
    )
    coefficients = fit_relative_linear(feature_matrix, targets)
    selected_validation["coefficients"] = {
        "intercept": float(coefficients[0]),
        "carbon": float(coefficients[1]),
        "hydrogen": float(coefficients[2]),
        "ring": float(coefficients[3]),
    }
    selected_validation["compounds"] = len(selected_rows)
    selected_validation["worst_volume_LOOCV"] = [
        {
            "name": selected_rows[index]["name"],
            "cas": selected_rows[index]["cas"],
            "signed_error_percent": float(selected_errors[index]),
        }
        for index in np.argsort(np.abs(selected_errors))[::-1][:10]
    ]

    unavailable = unavailable_volume_rows(
        conductivity_database, PerryPropertyLibrary()
    )
    unavailable_errors = []
    for row in unavailable:
        feature_values = np.asarray(selected_function(row["descriptors"]), dtype=float)
        estimated_volume = float(feature_values @ coefficients)
        errors = conductivity_errors(row, estimated_volume, conductivity_database)
        unavailable_errors.extend(errors)
        row["estimated_V20_cm3_mol"] = estimated_volume
        row["conductivity"] = error_metrics(errors)

    observed_errors = []
    for row in rows:
        observed_errors.extend(
            conductivity_errors(row, row["V20_cm3_mol"], conductivity_database)
        )
    hybrid_errors = observed_errors + unavailable_errors
    return {
        "candidate_models_all_86": candidates,
        "selected_model_C_at_least_3": selected_validation,
        "missing_V20_holdout": {
            "compounds": len(unavailable),
            "conductivity": error_metrics(unavailable_errors),
            "records": unavailable,
        },
        "hybrid_observed_or_estimated_V20": {
            "compounds": len(rows) + len(unavailable),
            "observed_V20_compounds": len(rows),
            "estimated_V20_compounds": len(unavailable),
            "conductivity": error_metrics(hybrid_errors),
        },
    }


def _metric_line(values: dict) -> str:
    return (
        f"MAPE={values['mape_percent']:.2f}% "
        f"MdAPE={values['median_ape_percent']:.2f}% "
        f"P95={values['p95_ape_percent']:.2f}% "
        f"bias={values['mean_signed_error_percent']:+.2f}%"
    )


def report(result: dict) -> str:
    lines = [
        "Hydrocarbon V20 structure-model probe",
        "All model comparisons use leave-one-compound-out volume predictions.",
        "",
        "CANDIDATE MODELS ON ALL 86 COMPOUNDS",
    ]
    for name, candidate in result["candidate_models_all_86"].items():
        lines.append(
            f"  {name:<30} p={candidate['parameters']} "
            f"V20 {_metric_line(candidate['volume_LOOCV'])}; "
            f"k {_metric_line(candidate['conductivity_LOOCV'])}"
        )
    selected = result["selected_model_C_at_least_3"]
    lines.extend(
        (
            "",
            "SELECTED C/H/RING MODEL, C >= 3",
            f"  compounds={selected['compounds']}",
            f"  V20 LOOCV: {_metric_line(selected['volume_LOOCV'])}",
            f"  k LOOCV:   {_metric_line(selected['conductivity_LOOCV'])}",
            f"  coefficients: {selected['coefficients']}",
            "",
            "COMPOUNDS WITH NO IN-RANGE PERRY V20",
        )
    )
    for row in result["missing_V20_holdout"]["records"]:
        lines.append(
            f"  {row['name']:<30} V20={row['estimated_V20_cm3_mol']:7.2f} cm3/mol "
            f"k MAPE={row['conductivity']['mape_percent']:6.2f}%"
        )
    lines.extend(
        (
            "  combined missing-V20 conductivity: "
            + _metric_line(result["missing_V20_holdout"]["conductivity"]),
            "",
            "HYBRID OBSERVED/ESTIMATED V20",
            "  "
            + _metric_line(
                result["hybrid_observed_or_estimated_V20"]["conductivity"]
            ),
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--conductivity-database",
        type=Path,
        default=DEFAULT_CONDUCTIVITY_DATABASE,
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_probe(args.benchmark, args.conductivity_database)
    script = Path(__file__).resolve()
    result["reproducibility"] = {
        "python": platform.python_version(),
        "script": str(script.relative_to(ROOT)),
        "script_sha256": _sha256(script),
        "benchmark_sha256": _sha256(args.benchmark),
        "conductivity_database_sha256": _sha256(args.conductivity_database),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(report(result))


if __name__ == "__main__":
    main()
