#!/usr/bin/env python3
"""Fit portable kernels to Shomate-in-range/affine-xTB composite Cp curves.

Twenty observations from 300 through 680 K fit a direct Shomate curve. Outside
that window an affine-calibrated xTB RRHO curve is shifted independently at
each boundary to make Cp continuous. The complete composite is then fitted to
PFDsim's degree-8/12 rational-Chebyshev kernel contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from property_resolution.ideal_gas_cp import (  # noqa: E402
    DEFAULT_TMAX_K,
    DEFAULT_TMIN_K,
    fit_chebyshev_kernel,
    load_bundled_kernel,
    rrho_ideal_gas_heat_capacity,
)
from property_resolution.resolver import PropertyResolver  # noqa: E402
from scripts.heat_capacity.gas.benchmark_xtb_affine_vs_direct_shomate import (  # noqa: E402
    fit_relative_shomate,
    shomate_basis,
)


ANCHORS_K = np.arange(300.0, 680.0 + 0.1, 20.0)
FIT_MAPE_LIMIT_PERCENT = 0.01
FIT_MAX_LIMIT_PERCENT = 0.1
DEFAULT_CANDIDATES = ROOT / "outputs" / "xtb_rrho_anchor_correction.csv"
DEFAULT_CSV = ROOT / "outputs" / "xtb_shomate_composite_kernel.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "xtb_shomate_composite_kernel_summary.json"


def canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("candidate SMILES is invalid")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def composite_evaluator(
    shomate_coefficients: np.ndarray,
    affine_coefficients: np.ndarray,
    frequencies_cm_1: tuple[float, ...],
    geometry: str,
) -> Callable[[Any], Any]:
    def shomate(temperatures: np.ndarray) -> np.ndarray:
        return shomate_basis(temperatures) @ shomate_coefficients

    def affine_xtb(temperatures: np.ndarray) -> np.ndarray:
        xtb = np.asarray(rrho_ideal_gas_heat_capacity(
            temperatures, frequencies_cm_1, geometry
        ))
        return affine_coefficients[0] + affine_coefficients[1] * xtb

    lower_shift = float(shomate(np.asarray([ANCHORS_K[0]]))[0]
                        - affine_xtb(np.asarray([ANCHORS_K[0]]))[0])
    upper_shift = float(shomate(np.asarray([ANCHORS_K[-1]]))[0]
                        - affine_xtb(np.asarray([ANCHORS_K[-1]]))[0])

    def evaluate(temperatures: Any) -> Any:
        values = np.asarray(temperatures, dtype=float)
        flat = values.reshape(-1)
        result = np.empty_like(flat)
        lower = flat < ANCHORS_K[0]
        upper = flat > ANCHORS_K[-1]
        middle = ~(lower | upper)
        if np.any(lower):
            result[lower] = affine_xtb(flat[lower]) + lower_shift
        if np.any(middle):
            result[middle] = shomate(flat[middle])
        if np.any(upper):
            result[upper] = affine_xtb(flat[upper]) + upper_shift
        reshaped = result.reshape(values.shape)
        return float(reshaped) if reshaped.ndim == 0 else reshaped

    return evaluate


def source_data(
    resolver: PropertyResolver,
    row: dict[str, str],
) -> tuple[Any, tuple[float, ...], str, np.ndarray, np.ndarray]:
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
    anchor_reference = np.asarray([reference.cp(float(T)) for T in ANCHORS_K])
    anchor_xtb = np.asarray(rrho_ideal_gas_heat_capacity(
        ANCHORS_K,
        artifact["frequencies_cm_1"],
        artifact["geometry"],
    ))
    return (
        reference,
        tuple(artifact["frequencies_cm_1"]),
        str(artifact["geometry"]),
        anchor_reference,
        anchor_xtb,
    )


def fit_one(
    evaluator: Callable[[Any], Any],
    degree: int,
) -> tuple[float, float] | None:
    try:
        kernel = fit_chebyshev_kernel(
            evaluator,
            DEFAULT_TMIN_K,
            DEFAULT_TMAX_K,
            quality=0.90,
            source="composite experiment",
            method="shomate_affine_xtb_composite",
            degree=degree,
        )
    except (TypeError, ValueError, OverflowError):
        return None
    return kernel.fit_mape_percent, kernel.fit_max_error_percent


def realization(
    observations: np.ndarray,
    anchor_xtb: np.ndarray,
    frequencies: tuple[float, ...],
    geometry: str,
) -> dict[str, Any]:
    shomate_coefficients = fit_relative_shomate(
        observations[None, :], shomate_basis(ANCHORS_K)
    )[0]
    affine_design = np.column_stack([np.ones(len(anchor_xtb)), anchor_xtb])
    affine_coefficients = np.linalg.lstsq(
        affine_design, observations, rcond=None
    )[0]
    evaluator = composite_evaluator(
        shomate_coefficients,
        affine_coefficients,
        frequencies,
        geometry,
    )
    fits = {degree: fit_one(evaluator, degree) for degree in (8, 12)}
    selected_degree = None
    for degree in (8, 12):
        fit = fits[degree]
        if (
            fit is not None
            and fit[0] < FIT_MAPE_LIMIT_PERCENT
            and fit[1] < FIT_MAX_LIMIT_PERCENT
        ):
            selected_degree = degree
            break
    return {
        "degree8_mape_percent": fits[8][0] if fits[8] else None,
        "degree8_max_percent": fits[8][1] if fits[8] else None,
        "degree12_mape_percent": fits[12][0] if fits[12] else None,
        "degree12_max_percent": fits[12][1] if fits[12] else None,
        "selected_degree": selected_degree,
    }


def summarize(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "realizations": len(records),
        "selection_counts": dict(Counter(
            "rejected" if row["selected_degree"] is None else str(row["selected_degree"])
            for row in records
        )),
    }
    for degree in (8, 12):
        mape = [
            float(row[f"degree{degree}_mape_percent"])
            for row in records
            if row[f"degree{degree}_mape_percent"] is not None
        ]
        maximum = [
            float(row[f"degree{degree}_max_percent"])
            for row in records
            if row[f"degree{degree}_max_percent"] is not None
        ]
        result[f"degree_{degree}"] = {
            "successful_fits": len(mape),
            "median_mape_percent": statistics.median(mape) if mape else None,
            "p95_mape_percent": float(np.percentile(mape, 95.0)) if mape else None,
            "median_max_error_percent": statistics.median(maximum) if maximum else None,
            "p95_max_error_percent": float(np.percentile(maximum, 95.0)) if maximum else None,
        }
    result["label"] = label
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--noise-std-percent", type=float, default=0.3)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    rng = np.random.default_rng(args.seed)
    with args.candidates.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))

    noiseless = []
    noisy = []
    per_compound: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix="pfdsim-composite-fit-") as directory:
        resolver = PropertyResolver()
        resolver.CACHE_DIR = Path(directory)
        for index, row in enumerate(candidates, 1):
            _, frequencies, geometry, anchor_reference, anchor_xtb = source_data(
                resolver, row
            )
            clean = realization(anchor_reference, anchor_xtb, frequencies, geometry)
            clean.update({"cas": row["cas"], "name": row["name"], "kind": "noiseless"})
            noiseless.append(clean)
            for _ in range(args.replicates):
                observations = anchor_reference * (
                    1.0
                    + rng.normal(
                        0.0,
                        args.noise_std_percent / 100.0,
                        size=len(anchor_reference),
                    )
                )
                result = realization(observations, anchor_xtb, frequencies, geometry)
                result.update({"cas": row["cas"], "name": row["name"], "kind": "noisy"})
                noisy.append(result)
                per_compound[row["cas"]].append(result)
            print(
                f"[{index:02d}/{len(candidates):02d}] {row['cas']} {row['name']}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list((noiseless + noisy)[0]))
        writer.writeheader()
        writer.writerows(noiseless + noisy)
    summary = {
        "sample_size": len(candidates),
        "anchor_temperatures_K": ANCHORS_K.tolist(),
        "composite": (
            "direct Shomate on 300--680 K; affine xTB outside, shifted "
            "independently for Cp continuity at 300 and 680 K"
        ),
        "kernel_fit_range_K": [DEFAULT_TMIN_K, DEFAULT_TMAX_K],
        "fit_acceptance": {
            "mape_below_percent": FIT_MAPE_LIMIT_PERCENT,
            "maximum_error_below_percent": FIT_MAX_LIMIT_PERCENT,
        },
        "noise": {
            "distribution": "independent multiplicative normal",
            "standard_deviation_percent": args.noise_std_percent,
            "replicates_per_compound": args.replicates,
            "seed": args.seed,
        },
        "noiseless": summarize(noiseless, "one realization per compound"),
        "noisy": summarize(noisy, "all compound-realization pairs"),
        "compound_rejection_rates": {
            cas: float(np.mean([row["selected_degree"] is None for row in rows]))
            for cas, rows in per_compound.items()
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
