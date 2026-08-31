#!/usr/bin/env python3
"""Benchmark atom-informed continuation outside a clustered Cp data range.

Ten observations are placed in a narrow low, middle, or high portion of each
component's admitted 273.15-1500 K interval.  A data-only Shomate curve is used
inside the observation window.  Outside, alternative methods either continue
that Shomate equation or multiply the held-out atom model by a constrained
continuation of the observed log residual ratio.

Formula-grouped outer folds keep every reported compound and all of its isomers
out of the atom-model training data.  Saturation hyperparameters are selected
for each outer fold using only the other four held-out folds.

This script is read-only and does not modify databases or runtime caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from chemicals.elements import simple_formula_parser


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_ideal_gas_cp_fallbacks as atom_benchmark  # noqa: E402


CONVENTIONAL_ORGANIC_ELEMENTS = frozenset(
    {"H", "D", "T", "C", "O", "N", "F", "Cl", "P", "S", "Br", "I"}
)
SATURATION_LENGTH_FACTORS = (0.5, 1.0, 2.0)
SATURATION_RATIO_CAPS = (1.25, 1.5, 2.0)


def is_conventional_organic(component) -> bool:
    elements = set(simple_formula_parser(component.formula))
    normalized = {"H" if element in {"D", "T"} else element for element in elements}
    return (
        {"C", "H"} <= normalized
        and elements <= CONVENTIONAL_ORGANIC_ELEMENTS
    )


def shomate_basis(temperatures: np.ndarray) -> np.ndarray:
    return atom_benchmark.temperature_basis("shomate", temperatures)


def relative_fit(design: np.ndarray, observations: np.ndarray) -> np.ndarray:
    relative_design = design / observations[:, None]
    return np.linalg.lstsq(
        relative_design, np.ones(len(observations)), rcond=1.0e-12
    )[0]


def relative_scale(atom_values: np.ndarray, observations: np.ndarray) -> float:
    relative_atom = atom_values / observations
    return float(
        np.dot(relative_atom, np.ones(len(relative_atom)))
        / np.dot(relative_atom, relative_atom)
    )


def deterministic_noise(
    values: np.ndarray,
    *,
    cas: str,
    position: float,
    standard_deviation_percent: float,
    seed: int,
) -> np.ndarray:
    if standard_deviation_percent == 0.0:
        return values
    text = f"{seed}|{cas}|{position:g}|{len(values)}"
    local_seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
    generator = np.random.default_rng(local_seed)
    result = values * (
        1.0
        + generator.normal(
            0.0, standard_deviation_percent / 100.0, size=len(values)
        )
    )
    if np.any(result <= 0.0):
        raise ValueError("simulated observation noise produced nonpositive Cp")
    return result


def log_residual_slope(
    temperatures: np.ndarray,
    observations: np.ndarray,
    atom_values: np.ndarray,
) -> float:
    centered = temperatures - float(np.mean(temperatures))
    design = np.column_stack([np.ones_like(centered), centered])
    coefficients = np.linalg.lstsq(
        design, np.log(observations / atom_values), rcond=None
    )[0]
    return float(coefficients[1])


def continuation_values(
    *,
    atom_curve: np.ndarray,
    atom_boundary: float,
    data_boundary: float,
    distances: np.ndarray,
    outward_log_slope: float,
    length: float,
    ratio_cap: float | None,
) -> np.ndarray:
    boundary_log_ratio = math.log(data_boundary / atom_boundary)
    if ratio_cap is None:
        log_ratio = boundary_log_ratio + outward_log_slope * distances
    else:
        excursion = float(
            np.clip(
                outward_log_slope * length,
                -math.log(ratio_cap),
                math.log(ratio_cap),
            )
        )
        log_ratio = boundary_log_ratio + excursion * (
            1.0 - np.exp(-distances / length)
        )
    return atom_curve * np.exp(log_ratio)


def evaluate_component(
    component,
    atom_parameters: np.ndarray,
    *,
    point_count: int,
    window_fraction: float,
    window_position: float,
    noise_percent: float,
    noise_seed: int,
) -> dict:
    Tmin = max(component.Tmin, 273.15)
    Tmax = min(component.Tmax, 1500.0)
    if Tmin >= Tmax:
        return {}
    interval_width = Tmax - Tmin
    observation_width = window_fraction * interval_width
    observation_min = Tmin + window_position * (interval_width - observation_width)
    observation_max = observation_min + observation_width
    observation_temperatures = np.linspace(
        observation_min, observation_max, point_count
    )
    observations = deterministic_noise(
        component.cp(observation_temperatures),
        cas=component.cas,
        position=window_position,
        standard_deviation_percent=noise_percent,
        seed=noise_seed,
    )

    category_parameters = atom_parameters.reshape(len(component.counts), 5)
    atom_coefficients = component.counts @ category_parameters
    atom_observations = shomate_basis(observation_temperatures) @ atom_coefficients
    data_coefficients = relative_fit(
        shomate_basis(observation_temperatures), observations
    )
    residual_slope = log_residual_slope(
        observation_temperatures, observations, atom_observations
    )

    temperatures = np.linspace(Tmin, Tmax, 301)
    truth = component.cp(temperatures)
    atom_curve = shomate_basis(temperatures) @ atom_coefficients
    data_curve = shomate_basis(temperatures) @ data_coefficients
    inside = (temperatures >= observation_min) & (temperatures <= observation_max)
    outside = ~inside

    methods = {
        "raw_shomate": data_curve.copy(),
        "global_scaled_atom": atom_curve
        * relative_scale(atom_observations, observations),
    }

    low_boundary_data = float(
        shomate_basis(np.asarray([observation_min]))[0] @ data_coefficients
    )
    high_boundary_data = float(
        shomate_basis(np.asarray([observation_max]))[0] @ data_coefficients
    )
    low_boundary_atom = float(
        shomate_basis(np.asarray([observation_min]))[0] @ atom_coefficients
    )
    high_boundary_atom = float(
        shomate_basis(np.asarray([observation_max]))[0] @ atom_coefficients
    )

    def piecewise(ratio_cap: float | None, length_factor: float) -> np.ndarray:
        result = data_curve.copy()
        low = temperatures < observation_min
        high = temperatures > observation_max
        if np.any(low):
            result[low] = continuation_values(
                atom_curve=atom_curve[low],
                atom_boundary=low_boundary_atom,
                data_boundary=low_boundary_data,
                distances=observation_min - temperatures[low],
                outward_log_slope=-residual_slope,
                length=max(length_factor * observation_width, 1.0),
                ratio_cap=ratio_cap,
            )
        if np.any(high):
            result[high] = continuation_values(
                atom_curve=atom_curve[high],
                atom_boundary=high_boundary_atom,
                data_boundary=high_boundary_data,
                distances=temperatures[high] - observation_max,
                outward_log_slope=residual_slope,
                length=max(length_factor * observation_width, 1.0),
                ratio_cap=ratio_cap,
            )
        return result

    methods["endpoint_scaled_atom"] = piecewise(1.0, 1.0)
    methods["unconstrained_log_residual"] = piecewise(None, 1.0)
    for length_factor in SATURATION_LENGTH_FACTORS:
        for ratio_cap in SATURATION_RATIO_CAPS:
            methods[f"saturating_{length_factor:g}_{ratio_cap:g}"] = piecewise(
                ratio_cap, length_factor
            )

    output = {
        "cas": component.cas,
        "formula": component.formula,
        "fold": component.fold,
        "organic": is_conventional_organic(component),
        "methods": {},
    }
    for name, predicted in methods.items():
        relative = 100.0 * np.abs((predicted - truth) / truth)
        output["methods"][name] = {
            "full_mape": float(np.mean(relative)),
            "inside_mape": float(np.mean(relative[inside])),
            "outside_mape": float(np.mean(relative[outside])),
            "nonpositive": bool(np.any(predicted <= 0.0)),
            "nonfinite": bool(np.any(~np.isfinite(predicted))),
        }
    return output


def summarize(values: list[float], nonpositive: int, nonfinite: int) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": len(array),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "within_10_percent": float(np.mean(array <= 10.0)),
        "nonpositive_curves": int(nonpositive),
        "nonfinite_curves": int(nonfinite),
    }


def benchmark(args: argparse.Namespace) -> dict:
    components, excluded, _ = atom_benchmark.load_components(
        args.database,
        folds=args.folds,
        grouping="formula",
        selenium_routing="separate",
        metal_routing="light_heavy",
    )
    atom_parameters = {
        fold: atom_benchmark.fit_atom_model(
            [component for component in components if component.fold != fold],
            "shomate",
            points=args.fit_points,
        )
        for fold in range(args.folds)
    }
    records = []
    for component in components:
        result = evaluate_component(
            component,
            atom_parameters[component.fold],
            point_count=args.points,
            window_fraction=args.observation_window_fraction,
            window_position=args.observation_window_position,
            noise_percent=args.noise_percent,
            noise_seed=args.noise_seed,
        )
        if result:
            records.append(result)

    candidates = [
        f"saturating_{length:g}_{cap:g}"
        for length in SATURATION_LENGTH_FACTORS
        for cap in SATURATION_RATIO_CAPS
    ]
    selected_by_fold = {}
    for outer_fold in range(args.folds):
        training = [record for record in records if record["fold"] != outer_fold]
        selected_by_fold[outer_fold] = min(
            candidates,
            key=lambda name: np.median(
                [record["methods"][name]["outside_mape"] for record in training]
            ),
        )

    populations = {}
    for population in ("all", "conventional_organic"):
        selected_records = (
            records
            if population == "all"
            else [record for record in records if record["organic"]]
        )
        method_names = (
            "raw_shomate",
            "global_scaled_atom",
            "endpoint_scaled_atom",
            "unconstrained_log_residual",
            "selected_saturating_residual",
        )
        metrics = {}
        for method in method_names:
            values = defaultdict(list)
            nonpositive = nonfinite = 0
            for record in selected_records:
                name = (
                    selected_by_fold[record["fold"]]
                    if method == "selected_saturating_residual"
                    else method
                )
                item = record["methods"][name]
                for metric in ("full_mape", "inside_mape", "outside_mape"):
                    values[metric].append(item[metric])
                nonpositive += int(item["nonpositive"])
                nonfinite += int(item["nonfinite"])
            metrics[method] = {
                metric: summarize(entries, nonpositive, nonfinite)
                for metric, entries in values.items()
            }
        populations[population] = metrics

    return {
        "configuration": {
            "database": str(args.database),
            "formula_usable_records": len(components),
            "excluded_records": excluded,
            "folds": args.folds,
            "points": args.points,
            "observation_window_fraction": args.observation_window_fraction,
            "observation_window_position": args.observation_window_position,
            "observation_noise_standard_deviation_percent": args.noise_percent,
            "noise_seed": args.noise_seed,
            "saturation_length_factors": list(SATURATION_LENGTH_FACTORS),
            "saturation_ratio_caps": list(SATURATION_RATIO_CAPS),
        },
        "selected_saturation_by_outer_fold": {
            str(fold): name for fold, name in selected_by_fold.items()
        },
        "populations": populations,
    }


def print_summary(result: dict) -> None:
    config = result["configuration"]
    print(
        "CONFIG "
        + " ".join(f"{key}={value}" for key, value in config.items())
    )
    print(
        "population|method|scope|n|median|mean|p90|p95|within10|"
        "nonpositive|nonfinite"
    )
    for population, methods in result["populations"].items():
        for method, scopes in methods.items():
            for scope, metric in scopes.items():
                print(
                    f"{population}|{method}|{scope}|{metric['n']}|"
                    f"{metric['median']:.3f}%|{metric['mean']:.3f}%|"
                    f"{metric['p90']:.3f}%|{metric['p95']:.3f}%|"
                    f"{100.0 * metric['within_10_percent']:.1f}%|"
                    f"{metric['nonpositive_curves']}|{metric['nonfinite_curves']}"
                )
    print("SELECTED", result["selected_saturation_by_outer_fold"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=atom_benchmark.DATABASE)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fit-points", type=int, default=41)
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--observation-window-fraction", type=float, default=0.2)
    parser.add_argument(
        "--observation-window-position", type=float, default=0.5
    )
    parser.add_argument("--noise-percent", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=20260814)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points < 5:
        raise SystemExit("--points must be at least 5 for a full Shomate fit")
    if not 0.0 < args.observation_window_fraction <= 1.0:
        raise SystemExit("--observation-window-fraction must be in (0, 1]")
    if not 0.0 <= args.observation_window_position <= 1.0:
        raise SystemExit("--observation-window-position must be in [0, 1]")
    result = benchmark(args)
    print_summary(result)
    if args.output_json:
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
