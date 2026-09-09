#!/usr/bin/env python3
"""Compare ten-point affine-xTB calibration with direct Shomate fitting.

The identical 0.3%-scatter observations at 400--625 K are supplied to both
models. Accuracy is measured separately inside that narrow observation window
and over each canonical curve's full overlap with 273.15--1500 K.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from property_resolution.ideal_gas_cp import (  # noqa: E402
    DEFAULT_TMAX_K,
    DEFAULT_TMIN_K,
    load_bundled_kernel,
    rrho_ideal_gas_heat_capacity,
)
from property_resolution.resolver import PropertyResolver  # noqa: E402


DEFAULT_CANDIDATES = ROOT / "outputs" / "xtb_rrho_anchor_correction.csv"
DEFAULT_CSV = ROOT / "outputs" / "xtb_affine_vs_direct_shomate.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "xtb_affine_vs_direct_shomate_summary.json"


def canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("candidate SMILES is invalid")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def shomate_basis(temperatures: np.ndarray) -> np.ndarray:
    reduced = np.asarray(temperatures, dtype=float) / 1000.0
    return np.column_stack([
        np.ones_like(reduced),
        reduced,
        reduced**2,
        reduced**3,
        reduced**-2,
    ])


def fit_relative_shomate(
    observations: np.ndarray,
    anchor_basis: np.ndarray,
) -> np.ndarray:
    coefficients = np.empty((len(observations), anchor_basis.shape[1]))
    for index, values in enumerate(observations):
        relative_design = anchor_basis / values[:, None]
        scales = np.sqrt(np.mean(relative_design * relative_design, axis=0))
        scaled = np.linalg.lstsq(
            relative_design / scales,
            np.ones(len(values)),
            rcond=1.0e-10,
        )[0]
        coefficients[index] = scaled / scales
    return coefficients


def affine_xtb_predictions(
    observations: np.ndarray,
    anchor_xtb: np.ndarray,
    evaluation_xtb: np.ndarray,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(anchor_xtb)), anchor_xtb])
    coefficients = observations @ np.linalg.pinv(design).T
    evaluation = np.column_stack([
        np.ones(len(evaluation_xtb)),
        evaluation_xtb,
    ])
    return coefficients @ evaluation.T


def mape(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.mean(
        100.0 * np.abs(predicted / reference[None, :] - 1.0),
        axis=1,
    )


def compound_trials(
    resolver: PropertyResolver,
    row: dict[str, str],
    rng: np.random.Generator,
    *,
    noise_fraction: float,
    replicates: int,
    full_points: int,
    window_points: int,
    anchor_temperatures: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    reference = load_bundled_kernel(row["cas"])
    if reference is None:
        raise RuntimeError(f"missing canonical curve for {row['cas']}")
    kernel = resolver._xtb_rrho_ideal_gas_cp_kernel(
        row["name"],
        {
            "CAS": row["cas"],
            "name": row["name"],
            "formula": row["formula"],
            "smiles": row["smiles"],
        },
        allow_online=False,
    )
    if kernel is None:
        raise RuntimeError(f"xTB RRHO failed for {row['cas']}")
    artifact = resolver._load_xtb_rrho_artifact(
        f"smiles:{canonical_smiles(row['smiles'])}"
    )
    if artifact is None:
        raise RuntimeError(f"missing xTB artifact for {row['cas']}")

    lower = max(DEFAULT_TMIN_K, reference.Tmin)
    upper = min(DEFAULT_TMAX_K, reference.Tmax)
    full_temperatures = np.geomspace(lower, upper, full_points)
    window_temperatures = np.linspace(
        anchor_temperatures[0], anchor_temperatures[-1], window_points
    )
    full_reference = np.asarray([reference.cp(float(T)) for T in full_temperatures])
    window_reference = np.asarray([
        reference.cp(float(T)) for T in window_temperatures
    ])
    anchor_reference = np.asarray([
        reference.cp(float(T)) for T in anchor_temperatures
    ])
    observations = anchor_reference[None, :] * (
        1.0
        + rng.normal(
            0.0,
            noise_fraction,
            size=(replicates, len(anchor_temperatures)),
        )
    )
    frequencies = artifact["frequencies_cm_1"]
    geometry = artifact["geometry"]
    anchor_xtb = np.asarray(rrho_ideal_gas_heat_capacity(
        anchor_temperatures, frequencies, geometry
    ))
    full_xtb = np.asarray(rrho_ideal_gas_heat_capacity(
        full_temperatures, frequencies, geometry
    ))
    window_xtb = np.asarray(rrho_ideal_gas_heat_capacity(
        window_temperatures, frequencies, geometry
    ))

    affine_full = affine_xtb_predictions(observations, anchor_xtb, full_xtb)
    affine_window = affine_xtb_predictions(observations, anchor_xtb, window_xtb)
    coefficients = fit_relative_shomate(
        observations,
        shomate_basis(anchor_temperatures),
    )
    shomate_full = coefficients @ shomate_basis(full_temperatures).T
    shomate_window = coefficients @ shomate_basis(window_temperatures).T
    return {
        "affine_xtb": {
            "window_mape": mape(affine_window, window_reference),
            "full_mape": mape(affine_full, full_reference),
            "nonpositive": np.any(affine_full <= 0.0, axis=1),
        },
        "direct_shomate": {
            "window_mape": mape(shomate_window, window_reference),
            "full_mape": mape(shomate_full, full_reference),
            "nonpositive": np.any(shomate_full <= 0.0, axis=1),
        },
    }


def summarize(
    trials: dict[str, list[dict[str, np.ndarray]]],
) -> list[dict[str, Any]]:
    rows = []
    for method, compounds in trials.items():
        window = np.stack([item["window_mape"] for item in compounds])
        full = np.stack([item["full_mape"] for item in compounds])
        nonpositive = np.stack([item["nonpositive"] for item in compounds])
        replicate_window_means = np.mean(window, axis=0)
        replicate_full_means = np.mean(full, axis=0)
        rows.append({
            "method": method,
            "mean_window_mape_percent": float(np.mean(window)),
            "median_window_mape_percent": float(np.median(window)),
            "window_mean_p05_percent": float(np.percentile(replicate_window_means, 5.0)),
            "window_mean_p95_percent": float(np.percentile(replicate_window_means, 95.0)),
            "mean_full_range_mape_percent": float(np.mean(full)),
            "median_full_range_mape_percent": float(np.median(full)),
            "full_mean_p05_percent": float(np.percentile(replicate_full_means, 5.0)),
            "full_mean_p95_percent": float(np.percentile(replicate_full_means, 95.0)),
            "fraction_nonpositive_full_curves": float(np.mean(nonpositive)),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--noise-std-percent", type=float, default=0.3)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--anchor-start-K", type=float, default=400.0)
    parser.add_argument("--anchor-step-K", type=float, default=25.0)
    parser.add_argument("--anchor-count", type=int, default=10)
    parser.add_argument("--full-points", type=int, default=301)
    parser.add_argument("--window-points", type=int, default=181)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    rng = np.random.default_rng(args.seed)
    anchor_temperatures = (
        args.anchor_start_K
        + args.anchor_step_K * np.arange(args.anchor_count, dtype=float)
    )
    with args.candidates.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))

    trials: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix="pfdsim-shomate-comparison-") as directory:
        resolver = PropertyResolver()
        resolver.CACHE_DIR = Path(directory)
        for index, candidate in enumerate(candidates, 1):
            result = compound_trials(
                resolver,
                candidate,
                rng,
                noise_fraction=args.noise_std_percent / 100.0,
                replicates=args.replicates,
                full_points=args.full_points,
                window_points=args.window_points,
                anchor_temperatures=anchor_temperatures,
            )
            for method, values in result.items():
                trials[method].append(values)
            print(
                f"[{index:02d}/{len(candidates):02d}] "
                f"{candidate['cas']} {candidate['name']}",
                flush=True,
            )

    rows = summarize(trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "sample_size": len(candidates),
        "anchor_temperatures_K": anchor_temperatures.tolist(),
        "noise": {
            "distribution": "independent multiplicative normal",
            "standard_deviation_percent": args.noise_std_percent,
            "replicates": args.replicates,
            "seed": args.seed,
        },
        "direct_shomate_fit": (
            "relative least squares with scaled columns and rcond=1e-10"
        ),
        "results": rows,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
