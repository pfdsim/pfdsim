#!/usr/bin/env python3
"""Test one-anchor corrections to GFN2-xTB RRHO ideal-gas Cp curves.

Twenty identity-clean compounds are selected from the paired xTB/Psi4 probe,
subject to an uncorrected dense-range MAPE above 1%.  Each correction uses only
the canonical Cp value at 500 K; all reported range metrics evaluate the full
273.15--1500 K overlap with the canonical empirical curve.
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


ANCHOR_TEMPERATURE_K = 500.0
DEFAULT_CANDIDATES = ROOT / "outputs" / "xtb_rrho_cp_psi4_extension.csv"
DEFAULT_CSV = ROOT / "outputs" / "xtb_rrho_anchor_correction.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "xtb_rrho_anchor_correction_summary.json"
METHODS = (
    "uncorrected",
    "additive_offset",
    "total_cp_scale",
    "vibrational_cp_scale",
    "frequency_scale",
)


def candidate_priority(row: dict[str, str]) -> tuple[float, str]:
    errors = [
        float(value)
        for key, value in row.items()
        if key.startswith("ape_") and value
    ]
    return (-(statistics.fmean(errors) if errors else 0.0), row["cas"])


def canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("candidate SMILES is invalid")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def bisect_frequency_scale(
    frequencies_cm_1: Sequence[float],
    geometry: str,
    target_cp: float,
) -> float | None:
    def residual(scale: float) -> float:
        return float(rrho_ideal_gas_heat_capacity(
            ANCHOR_TEMPERATURE_K,
            [scale * frequency for frequency in frequencies_cm_1],
            geometry,
        )) - target_cp

    lower = 1.0e-4
    upper = 100.0
    lower_value = residual(lower)
    upper_value = residual(upper)
    if lower_value == 0.0:
        return lower
    if upper_value == 0.0:
        return upper
    if lower_value * upper_value > 0.0:
        return None
    for _ in range(100):
        midpoint = math.sqrt(lower * upper)
        value = residual(midpoint)
        if abs(value) < 1.0e-10:
            return midpoint
        if lower_value * value <= 0.0:
            upper = midpoint
        else:
            lower = midpoint
            lower_value = value
    return math.sqrt(lower * upper)


def mean_ape(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.mean(100.0 * np.abs(predicted / reference - 1.0)))


def correction_curves(
    temperatures: np.ndarray,
    frequencies_cm_1: Sequence[float],
    geometry: str,
    target_anchor: float,
) -> tuple[dict[str, np.ndarray], dict[str, float | None]]:
    uncorrected = np.asarray(
        rrho_ideal_gas_heat_capacity(temperatures, frequencies_cm_1, geometry),
        dtype=float,
    )
    uncorrected_anchor = float(rrho_ideal_gas_heat_capacity(
        ANCHOR_TEMPERATURE_K,
        frequencies_cm_1,
        geometry,
    ))
    rigid_cp = float(rrho_ideal_gas_heat_capacity(
        ANCHOR_TEMPERATURE_K,
        (),
        geometry,
    ))
    vibrational_anchor = uncorrected_anchor - rigid_cp
    vibrational_scale = (
        (target_anchor - rigid_cp) / vibrational_anchor
        if vibrational_anchor > 0.0
        else None
    )
    frequency_scale = bisect_frequency_scale(
        frequencies_cm_1,
        geometry,
        target_anchor,
    )
    curves = {
        "uncorrected": uncorrected,
        "additive_offset": uncorrected + target_anchor - uncorrected_anchor,
        "total_cp_scale": uncorrected * target_anchor / uncorrected_anchor,
    }
    if vibrational_scale is not None and vibrational_scale > 0.0:
        curves["vibrational_cp_scale"] = (
            rigid_cp + vibrational_scale * (uncorrected - rigid_cp)
        )
    if frequency_scale is not None:
        curves["frequency_scale"] = np.asarray(
            rrho_ideal_gas_heat_capacity(
                temperatures,
                [frequency_scale * frequency for frequency in frequencies_cm_1],
                geometry,
            ),
            dtype=float,
        )
    return curves, {
        "anchor_offset_J_mol_K": target_anchor - uncorrected_anchor,
        "total_cp_scale": target_anchor / uncorrected_anchor,
        "vibrational_cp_scale": vibrational_scale,
        "frequency_scale": frequency_scale,
    }


def analyze_candidate(
    resolver: PropertyResolver,
    row: dict[str, str],
    points: int,
) -> dict[str, Any] | None:
    reference = load_bundled_kernel(row["cas"])
    if reference is None or not reference.covers(ANCHOR_TEMPERATURE_K):
        return None
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
        return None
    identity = f"smiles:{canonical_smiles(row['smiles'])}"
    artifact = resolver._load_xtb_rrho_artifact(identity)
    if artifact is None:
        return None
    lower = max(DEFAULT_TMIN_K, reference.Tmin)
    upper = min(DEFAULT_TMAX_K, reference.Tmax)
    if not lower <= ANCHOR_TEMPERATURE_K <= upper or upper <= lower:
        return None
    temperatures = np.geomspace(lower, upper, points)
    reference_values = np.asarray([reference.cp(float(T)) for T in temperatures])
    target_anchor = reference.cp(ANCHOR_TEMPERATURE_K)
    curves, parameters = correction_curves(
        temperatures,
        artifact["frequencies_cm_1"],
        artifact["geometry"],
        target_anchor,
    )
    metrics = {
        method: mean_ape(curve, reference_values)
        for method, curve in curves.items()
    }
    if metrics["uncorrected"] <= 1.0:
        return None
    return {
        "cas": row["cas"],
        "name": row["name"],
        "formula": row["formula"],
        "source": row["source"],
        "smiles": row["smiles"],
        "Tmin_K": lower,
        "Tmax_K": upper,
        "anchor_temperature_K": ANCHOR_TEMPERATURE_K,
        "reference_anchor_J_mol_K": target_anchor,
        "xtb_anchor_J_mol_K": float(rrho_ideal_gas_heat_capacity(
            ANCHOR_TEMPERATURE_K,
            artifact["frequencies_cm_1"],
            artifact["geometry"],
        )),
        **parameters,
        **{f"{method}_range_mape_percent": metrics.get(method) for method in METHODS},
    }


def summary_for(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    methods = {}
    for method in METHODS:
        key = f"{method}_range_mape_percent"
        values = [float(row[key]) for row in records if row.get(key) is not None]
        methods[method] = {
            "n": len(values),
            "mean_range_mape_percent": statistics.fmean(values) if values else None,
            "median_range_mape_percent": statistics.median(values) if values else None,
            "improved_molecules": (
                sum(value < float(row["uncorrected_range_mape_percent"])
                    for row, value in zip(
                        (row for row in records if row.get(key) is not None),
                        values,
                        strict=True,
                    ))
                if values
                else 0
            ),
        }
    return methods


def write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0]) if records else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--points", type=int, default=301)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    with args.candidates.open(newline="", encoding="utf-8") as handle:
        candidates = sorted(csv.DictReader(handle), key=candidate_priority)

    selected = []
    with tempfile.TemporaryDirectory(prefix="pfdsim-anchor-correction-") as directory:
        resolver = PropertyResolver()
        resolver.CACHE_DIR = Path(directory)
        for candidate in candidates:
            result = analyze_candidate(resolver, candidate, args.points)
            if result is None:
                continue
            selected.append(result)
            print(
                f"[{len(selected):02d}/{args.limit:02d}] {result['cas']} "
                f"{result['name']}: baseline="
                f"{result['uncorrected_range_mape_percent']:.3f}%",
                flush=True,
            )
            if len(selected) >= args.limit:
                break

    if len(selected) < args.limit:
        raise RuntimeError(
            f"only {len(selected)} candidates met the >1% dense-range criterion"
        )
    write_csv(args.output, selected)
    summary = {
        "sample_size": len(selected),
        "selection": (
            "identity-clean paired benchmark compounds ranked by sampled error, "
            "then admitted only when dense uncorrected range MAPE exceeds 1%"
        ),
        "range": "intersection of canonical source with 273.15--1500 K",
        "anchor_temperature_K": ANCHOR_TEMPERATURE_K,
        "evaluation_points_per_molecule": args.points,
        "methods": summary_for(selected),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
