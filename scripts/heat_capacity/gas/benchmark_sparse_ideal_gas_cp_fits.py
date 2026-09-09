#!/usr/bin/env python3
"""Benchmark 2-9 point Cp fits with a held-out atom-Shomate prior.

Each canonical curve supplies evenly spaced, exact observations including its
admitted interval endpoints.  Identical molecular formulas remain in one of
five folds.  The atom prior for a compound is therefore always fitted without
that compound or any isomer sharing its formula.

The ridge penalty for each reported outer fold is selected using only the
other four held-out folds.  This makes the regularized-Shomate result nested
out-of-sample rather than selecting its penalty on the reported compounds.

This script is read-only and does not modify databases or runtime caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from chemicals.elements import simple_formula_parser


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_ideal_gas_cp_fallbacks as atom_benchmark  # noqa: E402


RIDGE_PENALTIES = (
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
)
CONVENTIONAL_ORGANIC_ELEMENTS = frozenset(
    {"H", "D", "T", "C", "O", "N", "F", "Cl", "P", "S", "Br", "I"}
)


def shomate_basis(temperatures: Sequence[float] | np.ndarray) -> np.ndarray:
    return atom_benchmark.temperature_basis("shomate", temperatures)


def atom_coefficients(component, fitted_parameters: np.ndarray) -> np.ndarray:
    category_parameters = fitted_parameters.reshape(len(component.counts), 5)
    return component.counts @ category_parameters


def is_conventional_organic(component) -> bool:
    elements = set(simple_formula_parser(component.formula))
    normalized = {"H" if element in {"D", "T"} else element for element in elements}
    return (
        {"C", "H"} <= normalized
        and elements <= CONVENTIONAL_ORGANIC_ELEMENTS
    )


def relative_linear_fit(temperatures: np.ndarray, values: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(temperatures), temperatures])
    return np.linalg.lstsq(design, values, rcond=None)[0]


def relative_calibration(
    atom_values: np.ndarray,
    observed_values: np.ndarray,
) -> float:
    relative_atom = atom_values / observed_values
    return float(np.dot(relative_atom, np.ones(len(relative_atom))) / np.dot(relative_atom, relative_atom))


def relative_residual_fit(
    temperatures: np.ndarray,
    atom_values: np.ndarray,
    observed_values: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    center = 0.5 * (temperatures[0] + temperatures[-1])
    half_width = max(0.5 * (temperatures[-1] - temperatures[0]), 1.0)
    reduced = (temperatures - center) / half_width
    design = np.column_stack([reduced**power for power in range(degree + 1)])
    relative_design = design / observed_values[:, None]
    relative_residual = (observed_values - atom_values) / observed_values
    return np.linalg.lstsq(relative_design, relative_residual, rcond=None)[0]


def residual_values(
    coefficients: np.ndarray,
    observation_temperatures: np.ndarray,
    evaluation_temperatures: np.ndarray,
) -> np.ndarray:
    center = 0.5 * (observation_temperatures[0] + observation_temperatures[-1])
    half_width = max(
        0.5 * (observation_temperatures[-1] - observation_temperatures[0]), 1.0
    )
    reduced = (evaluation_temperatures - center) / half_width
    return sum(
        coefficient * reduced**power
        for power, coefficient in enumerate(coefficients)
    )


def regularized_shomate_coefficients(
    prior: np.ndarray,
    observation_temperatures: np.ndarray,
    observed_values: np.ndarray,
    dense_temperatures: np.ndarray,
    atom_dense_values: np.ndarray,
    penalty: float,
) -> np.ndarray:
    observation_design = shomate_basis(observation_temperatures)
    atom_observations = observation_design @ prior
    relative_design = observation_design / observed_values[:, None]
    dense_relative_design = (
        shomate_basis(dense_temperatures) / atom_dense_values[:, None]
    )
    scales = np.sqrt(np.mean(dense_relative_design * dense_relative_design, axis=0))
    scales = np.where(scales > 1.0e-14, scales, 1.0)
    normalized_design = relative_design / scales
    relative_residual = (observed_values - atom_observations) / observed_values
    normal = normalized_design.T @ normalized_design + penalty * np.eye(5)
    normalized_delta = np.linalg.solve(
        normal, normalized_design.T @ relative_residual
    )
    return prior + normalized_delta / scales


def curve_mape(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(100.0 * np.abs(predicted / actual - 1.0)))


def metric_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "within_5_percent": float(np.mean(array <= 5.0)),
        "within_10_percent": float(np.mean(array <= 10.0)),
        "within_20_percent": float(np.mean(array <= 20.0)),
    }


def select_nested_penalties(
    ridge_errors: dict[tuple[int, float], list[float]],
    folds: int,
) -> dict[int, float]:
    selected = {}
    for outer_fold in range(folds):
        candidates = []
        for penalty in RIDGE_PENALTIES:
            training_errors = [
                error
                for fold in range(folds)
                if fold != outer_fold
                for error in ridge_errors[(fold, penalty)]
            ]
            candidates.append((float(np.mean(training_errors)), penalty))
        selected[outer_fold] = min(candidates)[1]
    return selected


def benchmark(args: argparse.Namespace) -> dict:
    components, excluded, _categories = atom_benchmark.load_components(
        args.database,
        folds=args.folds,
        grouping="formula",
        selenium_routing="separate",
        metal_routing="light_heavy",
    )
    fitted_atom_parameters = {
        fold: atom_benchmark.fit_atom_model(
            [component for component in components if component.fold != fold],
            "shomate",
            points=args.fit_points,
        )
        for fold in range(args.folds)
    }

    output = {
        "configuration": {
            "database": str(args.database),
            "formula_usable_components": len(components),
            "excluded_components": excluded,
            "folds": args.folds,
            "fit_points_per_atom_component": args.fit_points,
            "evaluation_points_per_component": args.test_points,
            "observation_positions": "equally spaced including interval endpoints",
            "observation_window_fraction": args.observation_window_fraction,
            "observation_window_position": args.observation_window_position,
            "runtime_range_K": [273.15, 1500.0],
            "ridge_penalties": list(RIDGE_PENALTIES),
            "observation_noise_standard_deviation_percent": args.noise_percent,
            "noise_seed": args.noise_seed,
        },
        "point_counts": {},
    }

    for point_count in range(2, 10):
        records = []
        ridge_errors: dict[tuple[int, float], list[float]] = defaultdict(list)
        for component in components:
            Tmin = max(component.Tmin, 273.15)
            Tmax = min(component.Tmax, 1500.0)
            if Tmin >= Tmax:
                continue
            evaluation_temperatures = np.linspace(Tmin, Tmax, args.test_points)
            actual = component.cp(evaluation_temperatures)
            interval_width = Tmax - Tmin
            observation_width = args.observation_window_fraction * interval_width
            observation_start = (
                Tmin
                + args.observation_window_position
                * (interval_width - observation_width)
            )
            observation_temperatures = np.linspace(
                observation_start,
                observation_start + observation_width,
                point_count,
            )
            observed = component.cp(observation_temperatures)
            if args.noise_percent:
                seed_text = f"{args.noise_seed}|{component.cas}|{point_count}"
                seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
                generator = np.random.default_rng(seed)
                observed = observed * (
                    1.0
                    + generator.normal(
                        0.0, args.noise_percent / 100.0, size=point_count
                    )
                )
                if np.any(observed <= 0.0):
                    raise ValueError("simulated observation noise produced nonpositive Cp")
            prior = atom_coefficients(
                component, fitted_atom_parameters[component.fold]
            )
            atom_curve = shomate_basis(evaluation_temperatures) @ prior
            atom_observations = shomate_basis(observation_temperatures) @ prior

            linear = relative_linear_fit(observation_temperatures, observed)
            linear_curve = linear[0] + linear[1] * evaluation_temperatures
            scale = relative_calibration(atom_observations, observed)
            scaled_curve = atom_curve * scale
            constant_residual = relative_residual_fit(
                observation_temperatures,
                atom_observations,
                observed,
                degree=0,
            )
            linear_residual = relative_residual_fit(
                observation_temperatures,
                atom_observations,
                observed,
                degree=1,
            )
            constant_residual_curve = atom_curve + residual_values(
                constant_residual,
                observation_temperatures,
                evaluation_temperatures,
            )
            linear_residual_curve = atom_curve + residual_values(
                linear_residual,
                observation_temperatures,
                evaluation_temperatures,
            )

            ridge_predictions = {}
            for penalty in RIDGE_PENALTIES:
                coefficients = regularized_shomate_coefficients(
                    prior,
                    observation_temperatures,
                    observed,
                    evaluation_temperatures,
                    atom_curve,
                    penalty,
                )
                curve = shomate_basis(evaluation_temperatures) @ coefficients
                ridge_predictions[penalty] = curve
                ridge_errors[(component.fold, penalty)].append(
                    curve_mape(curve, actual)
                )

            records.append(
                {
                    "component": component,
                    "actual": actual,
                    "curves": {
                        "linear": linear_curve,
                        "atom_prior": atom_curve,
                        "scaled_atom": scaled_curve,
                        "constant_residual": constant_residual_curve,
                        "linear_residual": linear_residual_curve,
                    },
                    "ridge": ridge_predictions,
                }
            )

        selected_penalties = select_nested_penalties(ridge_errors, args.folds)
        populations = {
            "all": records,
            "conventional_organic": [
                record
                for record in records
                if is_conventional_organic(record["component"])
            ],
        }
        point_result = {
            "selected_ridge_penalty_by_outer_fold": {
                str(fold): penalty for fold, penalty in selected_penalties.items()
            },
            "populations": {},
        }
        for population, selected_records in populations.items():
            method_errors = defaultdict(list)
            nonpositive = defaultdict(int)
            for record in selected_records:
                curves = dict(record["curves"])
                curves["regularized_shomate"] = record["ridge"][
                    selected_penalties[record["component"].fold]
                ]
                for method, curve in curves.items():
                    method_errors[method].append(
                        curve_mape(curve, record["actual"])
                    )
                    nonpositive[method] += int(np.any(curve <= 0.0))
            point_result["populations"][population] = {
                method: {
                    **metric_summary(errors),
                    "nonpositive_curves": nonpositive[method],
                }
                for method, errors in method_errors.items()
            }
        output["point_counts"][str(point_count)] = point_result
    return output


def print_summary(result: dict) -> None:
    print(
        "population|points|method|n|median|mean|p90|p95|within10|nonpositive"
    )
    methods = (
        "linear",
        "atom_prior",
        "scaled_atom",
        "constant_residual",
        "linear_residual",
        "regularized_shomate",
    )
    for population in ("all", "conventional_organic"):
        for point_count, point_result in result["point_counts"].items():
            metrics = point_result["populations"][population]
            for method in methods:
                values = metrics[method]
                print(
                    f"{population}|{point_count}|{method}|{values['n']}|"
                    f"{values['median']:.3f}%|{values['mean']:.3f}%|"
                    f"{values['p90']:.3f}%|{values['p95']:.3f}%|"
                    f"{100.0 * values['within_10_percent']:.1f}%|"
                    f"{values['nonpositive_curves']}"
                )
    print("RIDGE_PENALTIES points|outer_fold:penalty")
    for point_count, point_result in result["point_counts"].items():
        selected = point_result["selected_ridge_penalty_by_outer_fold"]
        print(
            f"{point_count}|"
            + ",".join(f"{fold}:{penalty:g}" for fold, penalty in selected.items())
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=atom_benchmark.DATABASE
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fit-points", type=int, default=41)
    parser.add_argument("--test-points", type=int, default=101)
    parser.add_argument(
        "--noise-percent",
        type=float,
        default=0.0,
        help="Deterministic Gaussian relative observation-noise standard deviation.",
    )
    parser.add_argument("--noise-seed", type=int, default=20260813)
    parser.add_argument(
        "--observation-window-fraction",
        type=float,
        default=1.0,
        help="Fraction of the admitted interval spanned by observations.",
    )
    parser.add_argument(
        "--observation-window-position",
        type=float,
        default=0.5,
        help="Window placement: 0=low end, 0.5=centered, 1=high end.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.observation_window_fraction <= 1.0:
        raise SystemExit("--observation-window-fraction must be in (0, 1]")
    if not 0.0 <= args.observation_window_position <= 1.0:
        raise SystemExit("--observation-window-position must be in [0, 1]")
    result = benchmark(args)
    print_summary(result)
    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
