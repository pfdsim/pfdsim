#!/usr/bin/env python3
"""Compare poly_x and Shomate fits of chemicals.json Zabransky Cpl curves."""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import bmat, csr_matrix, eye


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemicals import heat_capacity  # noqa: E402

from scripts.heat_capacity.gas.build_ideal_gas_heat_capacity_db import (  # noqa: E402
    fit_quality_penalty,
)


JSON_PATH = ROOT / "data" / "chemicals.json"
LIQUID_DATABASE = ROOT / "data" / "liquid_heat_capacity.sqlite"
TRAINING_POINTS = 1001
VALIDATION_POINTS = 10001

# Historical chemicals.json fits removed after the canonical liquid database
# became authoritative. They remain here only to reproduce the comparison
# which justified their removal.
REMOVED_CORRELATIONS = {
    "C3H8O3": {
        "equation": "poly_x",
        "selected_model": "ZABRANSKY_SPLINE_C",
        "Tmin_K": 293.15,
        "Tmax_K": 382.7,
        "coefficients": {
            "A": 218.4588511272281,
            "B": 47.702316945250146,
            "C": 1.3266202860155935e-11,
            "D": -2.7529714297431537e-11,
            "E": 1.748572765894705e-11,
        },
    },
    "C6H7N": {
        "equation": "poly_x",
        "selected_model": "ZABRANSKY_SPLINE_C",
        "Tmin_K": 270.2,
        "Tmax_K": 634.5,
        "coefficients": {
            "A": 191.04062721427505,
            "B": 12.210785440681327,
            "C": 10.978696307413871,
            "D": -3.6481999566663452,
            "E": 0.4585804730536751,
        },
    },
    "C5H5N": {
        "equation": "poly_x",
        "selected_model": "ZABRANSKY_SPLINE_C",
        "Tmin_K": 239.7,
        "Tmax_K": 558.0,
        "coefficients": {
            "A": 132.73657912744687,
            "B": 20.193169351327477,
            "C": 3.201500480733615,
            "D": -0.24189685450254098,
            "E": 0.2658082912083216,
        },
    },
    "C8H8O": {
        "equation": "poly_x",
        "selected_model": "ZABRANSKY_SPLINE_C",
        "Tmin_K": 298.15,
        "Tmax_K": 638.64,
        "coefficients": {
            "A": 204.85925535232656,
            "B": 32.90599143876807,
            "C": -1.6219999095433735,
            "D": 0.7526085272694366,
            "E": 8.888937197967475e-14,
        },
    },
}


def shomate_basis(temperatures: np.ndarray) -> np.ndarray:
    t = temperatures / 1000.0
    return np.column_stack((
        np.ones_like(t),
        t,
        t * t,
        t * t * t,
        1.0 / (t * t),
    ))


def fit_relative_shomate(
    temperatures: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    design = shomate_basis(temperatures)
    relative_design = design / values[:, None]
    coefficients, *_ = np.linalg.lstsq(
        relative_design,
        np.ones(len(temperatures)),
        rcond=1.0e-12,
    )
    return coefficients


def fit_quality_objective_shomate(
    temperatures: np.ndarray,
    values: np.ndarray,
    *,
    mean_weight: float = 0.185,
    maximum_weight: float = 0.031,
) -> np.ndarray:
    """Minimize a weighted mean-absolute-plus-maximum relative error."""
    relative_design = shomate_basis(temperatures) / values[:, None]
    point_count = len(temperatures)
    design = csr_matrix(relative_design)
    negative_identity = -eye(point_count, format="csr")
    zero_points = csr_matrix((point_count, 1))
    negative_maximum = csr_matrix(-np.ones((point_count, 1)))

    # |Xc - 1| <= u and |Xc - 1| <= m.
    constraints = bmat((
        (design, negative_identity, zero_points),
        (-design, negative_identity, zero_points),
        (design, None, negative_maximum),
        (-design, None, negative_maximum),
    ), format="csr")
    upper_bounds = np.concatenate((
        np.ones(point_count),
        -np.ones(point_count),
        np.ones(point_count),
        -np.ones(point_count),
    ))
    objective = np.concatenate((
        np.zeros(5),
        np.full(point_count, mean_weight / point_count),
        np.asarray((maximum_weight,)),
    ))
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=upper_bounds,
        bounds=[(None, None)] * 5 + [(0.0, None)] * (point_count + 1),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Shomate quality-objective fit failed: {result.message}")
    return np.asarray(result.x[:5], dtype=float)


def poly_x_values(correlation: dict, temperatures: np.ndarray) -> np.ndarray:
    x = (temperatures - 298.15) / 100.0
    coefficients = correlation.get("coefficients") or {}
    values = np.zeros_like(x)
    for name in reversed("ABCDEF"):
        values = values * x + float(coefficients.get(name, 0.0))
    return values


def error_metrics(predicted: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    errors = np.abs(predicted / reference - 1.0) * 100.0
    return float(np.mean(errors)), float(np.max(errors))


def source_audit(cas: str) -> tuple[float, float]:
    with closing(sqlite3.connect(LIQUID_DATABASE)) as connection:
        row = connection.execute(
            """
            SELECT base_quality, details_json
            FROM source_candidate_audit
            WHERE cas = ? AND source = 'zabransky_p_spline'
            """,
            (cas,),
        ).fetchone()
    if row is None:
        raise RuntimeError(f"missing Zabransky p-spline audit for {cas}")
    details = json.loads(row[1])
    critical_temperature = float(details["critical_temperature_K"])
    return float(row[0]), critical_temperature


def evaluate_domain(
    model,
    correlation: dict,
    Tmin: float,
    Tmax: float,
    base_quality: float,
) -> dict:
    training_T = np.linspace(Tmin, Tmax, TRAINING_POINTS)
    training_cp = np.asarray(
        [model.calculate(float(T)) for T in training_T], dtype=float,
    )
    coefficients = fit_relative_shomate(training_T, training_cp)
    quality_coefficients = fit_quality_objective_shomate(training_T, training_cp)
    lad_coefficients = fit_quality_objective_shomate(
        training_T,
        training_cp,
        mean_weight=1.0,
        maximum_weight=0.0,
    )

    validation_T = np.linspace(Tmin, Tmax, VALIDATION_POINTS)
    reference = np.asarray(
        [model.calculate(float(T)) for T in validation_T], dtype=float,
    )
    shomate = shomate_basis(validation_T) @ coefficients
    if np.any(~np.isfinite(shomate)) or np.any(shomate <= 0.0):
        raise RuntimeError("Shomate fit produced invalid heat capacity")
    shomate_mape, shomate_max = error_metrics(shomate, reference)
    shomate_penalty = fit_quality_penalty(shomate_max, shomate_mape)

    quality_shomate = shomate_basis(validation_T) @ quality_coefficients
    if np.any(~np.isfinite(quality_shomate)) or np.any(quality_shomate <= 0.0):
        raise RuntimeError("quality-objective Shomate fit produced invalid heat capacity")
    quality_mape, quality_max = error_metrics(quality_shomate, reference)
    quality_penalty = fit_quality_penalty(quality_max, quality_mape)

    lad_shomate = shomate_basis(validation_T) @ lad_coefficients
    if np.any(~np.isfinite(lad_shomate)) or np.any(lad_shomate <= 0.0):
        raise RuntimeError("LAD Shomate fit produced invalid heat capacity")
    lad_mape, lad_max = error_metrics(lad_shomate, reference)
    lad_penalty = fit_quality_penalty(lad_max, lad_mape)

    poly_x = poly_x_values(correlation, validation_T)
    poly_mape, poly_max = error_metrics(poly_x, reference)
    poly_penalty = fit_quality_penalty(poly_max, poly_mape)

    return {
        "Tmin": Tmin,
        "Tmax": Tmax,
        "coefficients": coefficients,
        "quality_coefficients": quality_coefficients,
        "lad_coefficients": lad_coefficients,
        "shomate_mape": shomate_mape,
        "shomate_max": shomate_max,
        "shomate_penalty": shomate_penalty,
        "shomate_quality": max(0.0, base_quality - shomate_penalty),
        "quality_mape": quality_mape,
        "quality_max": quality_max,
        "quality_penalty": quality_penalty,
        "quality_quality": max(0.0, base_quality - quality_penalty),
        "lad_mape": lad_mape,
        "lad_max": lad_max,
        "lad_penalty": lad_penalty,
        "lad_quality": max(0.0, base_quality - lad_penalty),
        "poly_mape": poly_mape,
        "poly_max": poly_max,
        "poly_penalty": poly_penalty,
        "poly_quality": max(0.0, base_quality - poly_penalty),
    }


def main() -> None:
    _ = heat_capacity.Cp_data_Poling
    chemicals = json.loads(JSON_PATH.read_text(encoding="utf-8"))["chemicals"]
    for symbol, correlation in REMOVED_CORRELATIONS.items():
        properties = chemicals[symbol]
        if "Cpl" in (properties.get("property_correlations") or {}):
            raise RuntimeError(
                f"{symbol} unexpectedly restored a duplicate runtime Cpl correlation"
            )
        cas = str(properties["CAS"])
        model = heat_capacity.zabransky_dict_iso_s[cas]
        base_quality, critical_temperature = source_audit(cas)
        json_Tmin = max(float(correlation["Tmin_K"]), float(model.Tmin))
        json_Tmax = min(float(correlation["Tmax_K"]), float(model.Tmax))
        admitted_Tmin = float(model.Tmin)
        admitted_Tmax = min(float(model.Tmax), 0.95 * critical_temperature)

        print(f"\n{properties['name']} ({symbol}, {cas})")
        print(f"base_quality={base_quality:.9f}")
        for label, Tmin, Tmax in (
            ("json_range", json_Tmin, json_Tmax),
            ("compiler_range", admitted_Tmin, admitted_Tmax),
        ):
            result = evaluate_domain(
                model, correlation, Tmin, Tmax, base_quality,
            )
            print(f"{label}={Tmin:g}-{Tmax:g} K")
            print(
                "  poly_x:   "
                f"MAPE={result['poly_mape']:.12g}% "
                f"max={result['poly_max']:.12g}% "
                f"penalty={result['poly_penalty']:.12g} "
                f"quality={result['poly_quality']:.12g}"
            )
            print(
                "  Shomate:  "
                f"MAPE={result['shomate_mape']:.12g}% "
                f"max={result['shomate_max']:.12g}% "
                f"penalty={result['shomate_penalty']:.12g} "
                f"quality={result['shomate_quality']:.12g}"
            )
            A, B, C, D, E = result["coefficients"]
            print(
                "  coefficients: "
                f"A={A:.15g}, B={B:.15g}, C={C:.15g}, "
                f"D={D:.15g}, E={E:.15g}"
            )
            print(
                "  optimized: "
                f"MAPE={result['quality_mape']:.12g}% "
                f"max={result['quality_max']:.12g}% "
                f"penalty={result['quality_penalty']:.12g} "
                f"quality={result['quality_quality']:.12g}"
            )
            A, B, C, D, E = result["quality_coefficients"]
            print(
                "  coefficients: "
                f"A={A:.15g}, B={B:.15g}, C={C:.15g}, "
                f"D={D:.15g}, E={E:.15g}"
            )
            print(
                "  LAD:       "
                f"MAPE={result['lad_mape']:.12g}% "
                f"max={result['lad_max']:.12g}% "
                f"penalty={result['lad_penalty']:.12g} "
                f"quality={result['lad_quality']:.12g}"
            )
            A, B, C, D, E = result["lad_coefficients"]
            print(
                "  coefficients: "
                f"A={A:.15g}, B={B:.15g}, C={C:.15g}, "
                f"D={D:.15g}, E={E:.15g}"
            )


if __name__ == "__main__":
    main()
