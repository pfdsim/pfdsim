#!/usr/bin/env python3
"""Monte Carlo sparse-anchor xTB Cp corrections with experimental scatter.

Every sparse canonical Cp anchor receives independent multiplicative Gaussian
noise. Corrected curves are evaluated over each compound's full canonical
overlap with 273.15--1500 K, using the same 20-compound difficult cohort and
narrow anchor grids as ``benchmark_xtb_rrho_narrow_anchor_correction.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

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
from scripts.heat_capacity.gas.benchmark_xtb_rrho_narrow_anchor_correction import (  # noqa: E402
    ANCHOR_GRIDS,
)


DEFAULT_CANDIDATES = ROOT / "outputs" / "xtb_rrho_anchor_correction.csv"
DEFAULT_CSV = ROOT / "outputs" / "xtb_rrho_anchor_noise.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "xtb_rrho_anchor_noise_summary.json"


def canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("candidate SMILES is invalid")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def design_predictions(
    method: str,
    anchor_temperatures: np.ndarray,
    anchor_xtb: np.ndarray,
    noisy_reference: np.ndarray,
    evaluation_temperatures: np.ndarray,
    evaluation_xtb: np.ndarray,
) -> np.ndarray:
    if method == "constant_residual":
        offsets = np.mean(noisy_reference - anchor_xtb, axis=1)
        return evaluation_xtb[None, :] + offsets[:, None]
    if method == "total_cp_scale":
        scales = noisy_reference @ anchor_xtb / float(anchor_xtb @ anchor_xtb)
        return scales[:, None] * evaluation_xtb[None, :]
    if method == "affine_xtb":
        design = np.column_stack([np.ones(len(anchor_xtb)), anchor_xtb])
        coefficients = noisy_reference @ np.linalg.pinv(design).T
        evaluation = np.column_stack([
            np.ones(len(evaluation_xtb)),
            evaluation_xtb,
        ])
        return coefficients @ evaluation.T

    center = float(np.mean(anchor_temperatures))
    scale = float(np.ptp(anchor_temperatures))
    degree = 1 if method == "linear_temperature_residual" else 2
    anchor_x = (anchor_temperatures - center) / scale
    evaluation_x = (evaluation_temperatures - center) / scale
    design = np.polynomial.chebyshev.chebvander(anchor_x, degree)
    coefficients = (noisy_reference - anchor_xtb) @ np.linalg.pinv(design).T
    correction = coefficients @ np.polynomial.chebyshev.chebvander(
        evaluation_x, degree
    ).T
    return evaluation_xtb[None, :] + correction


def methods_for_count(count: int) -> tuple[str, ...]:
    methods = ["constant_residual", "total_cp_scale"]
    if count >= 2:
        methods.extend(("affine_xtb", "linear_temperature_residual"))
    if count >= 5:
        methods.append("quadratic_temperature_residual")
    return tuple(methods)


def compound_trials(
    resolver: PropertyResolver,
    source_row: dict[str, str],
    rng: np.random.Generator,
    *,
    noise_fraction: float,
    replicates: int,
    evaluation_points: int,
) -> tuple[float, dict[tuple[int, str], np.ndarray]]:
    reference = load_bundled_kernel(source_row["cas"])
    if reference is None:
        raise RuntimeError(f"missing canonical curve for {source_row['cas']}")
    kernel = resolver._xtb_rrho_ideal_gas_cp_kernel(
        source_row["name"],
        {
            "CAS": source_row["cas"],
            "name": source_row["name"],
            "formula": source_row["formula"],
            "smiles": source_row["smiles"],
        },
        allow_online=False,
    )
    if kernel is None:
        raise RuntimeError(f"xTB RRHO failed for {source_row['cas']}")
    artifact = resolver._load_xtb_rrho_artifact(
        f"smiles:{canonical_smiles(source_row['smiles'])}"
    )
    if artifact is None:
        raise RuntimeError(f"missing xTB artifact for {source_row['cas']}")

    lower = max(DEFAULT_TMIN_K, reference.Tmin)
    upper = min(DEFAULT_TMAX_K, reference.Tmax)
    temperatures = np.geomspace(lower, upper, evaluation_points)
    reference_values = np.asarray([reference.cp(float(T)) for T in temperatures])
    xtb_values = np.asarray(rrho_ideal_gas_heat_capacity(
        temperatures,
        artifact["frequencies_cm_1"],
        artifact["geometry"],
    ))
    baseline = float(np.mean(100.0 * np.abs(xtb_values / reference_values - 1.0)))
    results = {}
    for count, anchor_temperatures in ANCHOR_GRIDS.items():
        anchor_reference = np.asarray([
            reference.cp(float(T)) for T in anchor_temperatures
        ])
        anchor_xtb = np.asarray(rrho_ideal_gas_heat_capacity(
            anchor_temperatures,
            artifact["frequencies_cm_1"],
            artifact["geometry"],
        ))
        noisy_reference = anchor_reference[None, :] * (
            1.0
            + rng.normal(
                0.0,
                noise_fraction,
                size=(replicates, len(anchor_temperatures)),
            )
        )
        for method in methods_for_count(count):
            predicted = design_predictions(
                method,
                anchor_temperatures,
                anchor_xtb,
                noisy_reference,
                temperatures,
                xtb_values,
            )
            results[(count, method)] = np.mean(
                100.0 * np.abs(predicted / reference_values[None, :] - 1.0),
                axis=1,
            )
    return baseline, results


def summarize(
    baselines: Sequence[float],
    trials: dict[tuple[int, str], list[np.ndarray]],
) -> list[dict[str, Any]]:
    baseline_array = np.asarray(baselines)[:, None]
    rows = []
    for (count, method), molecule_trials in sorted(trials.items()):
        values = np.stack(molecule_trials)
        replicate_means = np.mean(values, axis=0)
        rows.append({
            "anchor_count": count,
            "anchor_temperatures_K": ";".join(
                f"{value:g}" for value in ANCHOR_GRIDS[count]
            ),
            "method": method,
            "mean_range_mape_percent": float(np.mean(values)),
            "median_range_mape_percent": float(np.median(values)),
            "p90_range_mape_percent": float(np.percentile(values, 90.0)),
            "mean_across_molecules_p05_percent": float(
                np.percentile(replicate_means, 5.0)
            ),
            "mean_across_molecules_p95_percent": float(
                np.percentile(replicate_means, 95.0)
            ),
            "fraction_of_trials_improving_baseline": float(
                np.mean(values < baseline_array)
            ),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--noise-std-percent", type=float, default=0.3)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--points", type=int, default=301)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    rng = np.random.default_rng(args.seed)
    with args.candidates.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))

    baselines = []
    trials: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix="pfdsim-anchor-noise-") as directory:
        resolver = PropertyResolver()
        resolver.CACHE_DIR = Path(directory)
        for index, candidate in enumerate(candidates, 1):
            baseline, compound = compound_trials(
                resolver,
                candidate,
                rng,
                noise_fraction=args.noise_std_percent / 100.0,
                replicates=args.replicates,
                evaluation_points=args.points,
            )
            baselines.append(baseline)
            for key, values in compound.items():
                trials[key].append(values)
            print(
                f"[{index:02d}/{len(candidates):02d}] {candidate['cas']} "
                f"{candidate['name']}: baseline={baseline:.3f}%",
                flush=True,
            )

    rows = summarize(baselines, trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "sample_size": len(candidates),
        "noise": {
            "distribution": "independent multiplicative normal",
            "standard_deviation_percent": args.noise_std_percent,
            "replicates": args.replicates,
            "seed": args.seed,
        },
        "evaluation_range": "each canonical curve's overlap with 273.15--1500 K",
        "evaluation_points_per_molecule": args.points,
        "baseline": {
            "mean_range_mape_percent": statistics.fmean(baselines),
            "median_range_mape_percent": statistics.median(baselines),
        },
        "corrections": rows,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
