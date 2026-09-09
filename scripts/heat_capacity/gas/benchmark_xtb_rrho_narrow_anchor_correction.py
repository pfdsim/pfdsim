#!/usr/bin/env python3
"""Benchmark narrow-range sparse-data corrections to GFN2-xTB RRHO Cp.

The same 20 identity-clean compounds selected by the one-anchor experiment are
used. Sparse canonical reference points are restricted to deliberately narrow
online-data-like windows, while corrected curves are evaluated over each
canonical curve's full overlap with 273.15--1500 K.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

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


ANCHOR_GRIDS = {
    1: np.asarray([500.0]),
    2: np.asarray([500.0, 600.0]),
    5: np.arange(500.0, 600.0 + 0.1, 25.0),
    10: np.arange(400.0, 625.0 + 0.1, 25.0),
}
DEFAULT_CANDIDATES = ROOT / "outputs" / "xtb_rrho_anchor_correction.csv"
DEFAULT_CSV = ROOT / "outputs" / "xtb_rrho_narrow_anchor_correction.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "xtb_rrho_narrow_anchor_correction_summary.json"


def canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("candidate SMILES is invalid")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def relative_mse(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean((predicted / reference - 1.0) ** 2))


def range_mape(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean(100.0 * np.abs(predicted / reference - 1.0)))


def optimize_frequency_model(
    frequencies: Sequence[float],
    geometry: str,
    anchor_temperatures: np.ndarray,
    anchor_reference: np.ndarray,
    *,
    with_offset: bool,
) -> tuple[float, float]:
    def parameters(log_scale: float) -> tuple[float, float]:
        scale = math.exp(log_scale)
        prediction = np.asarray(rrho_ideal_gas_heat_capacity(
            anchor_temperatures,
            np.asarray(frequencies) * scale,
            geometry,
        ))
        if with_offset:
            weights = 1.0 / (anchor_reference * anchor_reference)
            offset = float(
                np.sum(weights * (anchor_reference - prediction))
                / np.sum(weights)
            )
        else:
            offset = 0.0
        return relative_mse(prediction + offset, anchor_reference), offset

    grid = np.linspace(math.log(0.05), math.log(20.0), 801)
    objectives = np.asarray([parameters(value)[0] for value in grid])
    best = int(np.argmin(objectives))
    left = grid[max(0, best - 1)]
    right = grid[min(len(grid) - 1, best + 1)]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    c = right - golden * (right - left)
    d = left + golden * (right - left)
    for _ in range(60):
        if parameters(c)[0] < parameters(d)[0]:
            right = d
            d = c
            c = right - golden * (right - left)
        else:
            left = c
            c = d
            d = left + golden * (right - left)
    log_scale = 0.5 * (left + right)
    _, offset = parameters(log_scale)
    return math.exp(log_scale), offset


def correction_models(
    anchor_temperatures: np.ndarray,
    anchor_reference: np.ndarray,
    anchor_xtb: np.ndarray,
    frequencies: Sequence[float],
    geometry: str,
) -> dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    residual = anchor_reference - anchor_xtb
    models: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "constant_residual": (
            lambda _temperatures, xtb, offset=float(np.mean(residual)): xtb + offset
        ),
    }
    if len(anchor_temperatures) >= 2:
        center = float(np.mean(anchor_temperatures))
        scale = float(np.ptp(anchor_temperatures))
        reduced = (anchor_temperatures - center) / scale
        linear = np.polynomial.chebyshev.chebfit(reduced, residual, 1)
        models["linear_temperature_residual"] = (
            lambda temperatures, xtb, coefficients=linear: xtb
            + np.polynomial.chebyshev.chebval((temperatures - center) / scale, coefficients)
        )
        design = np.column_stack([np.ones(len(anchor_xtb)), anchor_xtb])
        intercept, slope = np.linalg.lstsq(design, anchor_reference, rcond=None)[0]
        models["affine_xtb"] = (
            lambda _temperatures, xtb, intercept=float(intercept), slope=float(slope):
            intercept + slope * xtb
        )
    if len(anchor_temperatures) >= 5:
        center = float(np.mean(anchor_temperatures))
        scale = float(np.ptp(anchor_temperatures))
        reduced = (anchor_temperatures - center) / scale
        quadratic = np.polynomial.chebyshev.chebfit(reduced, residual, 2)
        models["quadratic_temperature_residual"] = (
            lambda temperatures, xtb, coefficients=quadratic: xtb
            + np.polynomial.chebyshev.chebval((temperatures - center) / scale, coefficients)
        )

    frequency_scale, _ = optimize_frequency_model(
        frequencies,
        geometry,
        anchor_temperatures,
        anchor_reference,
        with_offset=False,
    )
    models["frequency_scale"] = (
        lambda temperatures, _xtb, scale=frequency_scale: np.asarray(
            rrho_ideal_gas_heat_capacity(
                temperatures,
                np.asarray(frequencies) * scale,
                geometry,
            )
        )
    )
    if len(anchor_temperatures) >= 2:
        frequency_scale, offset = optimize_frequency_model(
            frequencies,
            geometry,
            anchor_temperatures,
            anchor_reference,
            with_offset=True,
        )
        models["frequency_scale_plus_offset"] = (
            lambda temperatures, _xtb, scale=frequency_scale, offset=offset:
            np.asarray(rrho_ideal_gas_heat_capacity(
                temperatures,
                np.asarray(frequencies) * scale,
                geometry,
            )) + offset
        )
    return models


def calculate_compound(
    resolver: PropertyResolver,
    source_row: dict[str, str],
    evaluation_points: int,
) -> list[dict[str, Any]]:
    reference = load_bundled_kernel(source_row["cas"])
    if reference is None:
        return []
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
        return []
    artifact = resolver._load_xtb_rrho_artifact(
        f"smiles:{canonical_smiles(source_row['smiles'])}"
    )
    if artifact is None:
        return []

    lower = max(DEFAULT_TMIN_K, reference.Tmin)
    upper = min(DEFAULT_TMAX_K, reference.Tmax)
    temperatures = np.geomspace(lower, upper, evaluation_points)
    reference_values = np.asarray([reference.cp(float(T)) for T in temperatures])
    xtb_values = np.asarray(rrho_ideal_gas_heat_capacity(
        temperatures,
        artifact["frequencies_cm_1"],
        artifact["geometry"],
    ))
    baseline = range_mape(xtb_values, reference_values)
    records = []
    for count, anchor_temperatures in ANCHOR_GRIDS.items():
        if anchor_temperatures[0] < lower or anchor_temperatures[-1] > upper:
            continue
        anchor_reference = np.asarray([
            reference.cp(float(T)) for T in anchor_temperatures
        ])
        anchor_xtb = np.asarray(rrho_ideal_gas_heat_capacity(
            anchor_temperatures,
            artifact["frequencies_cm_1"],
            artifact["geometry"],
        ))
        for method, model in correction_models(
            anchor_temperatures,
            anchor_reference,
            anchor_xtb,
            artifact["frequencies_cm_1"],
            artifact["geometry"],
        ).items():
            corrected = model(temperatures, xtb_values)
            records.append({
                "cas": source_row["cas"],
                "name": source_row["name"],
                "formula": source_row["formula"],
                "source": source_row["source"],
                "anchor_count": count,
                "anchor_temperatures_K": ";".join(f"{T:g}" for T in anchor_temperatures),
                "method": method,
                "baseline_range_mape_percent": baseline,
                "corrected_range_mape_percent": range_mape(corrected, reference_values),
                "minimum_corrected_cp_J_mol_K": float(np.min(corrected)),
            })
    return records


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    keys = sorted({(int(row["anchor_count"]), str(row["method"])) for row in records})
    for count, method in keys:
        cohort = [
            row for row in records
            if int(row["anchor_count"]) == count and row["method"] == method
        ]
        corrected = [float(row["corrected_range_mape_percent"]) for row in cohort]
        result[f"{count}_points:{method}"] = {
            "n": len(cohort),
            "mean_range_mape_percent": statistics.fmean(corrected),
            "median_range_mape_percent": statistics.median(corrected),
            "improved_molecules": sum(
                float(row["corrected_range_mape_percent"])
                < float(row["baseline_range_mape_percent"])
                for row in cohort
            ),
            "nonpositive_curves": sum(
                float(row["minimum_corrected_cp_J_mol_K"]) <= 0.0
                for row in cohort
            ),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--points", type=int, default=301)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    with args.candidates.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    records = []
    with tempfile.TemporaryDirectory(prefix="pfdsim-narrow-anchor-") as directory:
        resolver = PropertyResolver()
        resolver.CACHE_DIR = Path(directory)
        for index, candidate in enumerate(candidates, 1):
            compound_records = calculate_compound(resolver, candidate, args.points)
            records.extend(compound_records)
            print(
                f"[{index:02d}/{len(candidates):02d}] {candidate['cas']} "
                f"{candidate['name']}: rows={len(compound_records)}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    baseline = {
        row["cas"]: float(row["baseline_range_mape_percent"])
        for row in records
    }
    summary = {
        "sample_size": len(baseline),
        "evaluation_range": "each canonical curve's overlap with 273.15--1500 K",
        "evaluation_points_per_molecule": args.points,
        "anchor_grids_K": {
            str(count): values.tolist() for count, values in ANCHOR_GRIDS.items()
        },
        "baseline": {
            "mean_range_mape_percent": statistics.fmean(baseline.values()),
            "median_range_mape_percent": statistics.median(baseline.values()),
        },
        "corrections": summarize(records),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
