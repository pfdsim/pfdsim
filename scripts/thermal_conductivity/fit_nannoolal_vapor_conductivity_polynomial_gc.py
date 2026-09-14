#!/usr/bin/env python3
"""Fit direct quadratic/cubic first-order Nannoolal GC conductivity models.

The reported curve is k = A + B*T + C*T**2 (+ D*T**3). Internally, temperature
is centered and scaled as x=(T-500 K)/500 K. This preserves the exact
polynomial while avoiding the severe conditioning of raw Kelvin powers.

Each centered polynomial coefficient is an additive sum of first-order
Nannoolal group contributions. Ridge strength is selected by nested,
whole-compound cross-validation using compound-balanced relative squared
error. No viscosity, heat capacity, or Nannoolal second-order correction is
used in the first-order arm. The augmented arm admits both structural
corrections and interaction-pair counts meeting the compound-support cutoff.
Optional molar-mass features include global linear, quadratic, logarithmic,
and inverse-square-root terms, plus per-group linear or logarithmic mass
modulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.thermal_conductivity.fit_nannoolal_vapor_conductivity_gc import (  # noqa: E402
    DEFAULT_CONDUCTIVITY_DATABASE,
    Curve,
    feature_names,
    load_curves,
    local_smiles,
    percentile,
)
import nannoolal_method as nm  # noqa: E402


T_CENTER_K = 500.0
T_SCALE_K = 500.0
RIDGE_ALPHAS = (0.0, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
FIRST_ORDER = "first_order"
FIRST_PLUS_SECOND_ORDER = "first_plus_second_order"
GROUP_ORDERS = (FIRST_ORDER, FIRST_PLUS_SECOND_ORDER)
MASS_REFERENCE_G_PER_MOL = 100.0
BOILING_POINT_REFERENCE_K = 500.0
MASS_RULES = (
    "none",
    "global_linear",
    "global_quadratic",
    "global_log",
    "global_inverse_sqrt",
    "group_linear",
    "group_log",
)
BOILING_POINT_RULES = (
    "none",
    "global_linear",
    "global_quadratic",
    "global_log",
)


def second_order_groups_by_cas(curves: Sequence[Curve]) -> dict[str, dict[str, int]]:
    output = {}
    for cas in sorted({curve.cas for curve in curves}):
        molecule = Chem.MolFromSmiles(local_smiles(cas))
        if molecule is None:
            raise ValueError(f"Nannoolal benchmark SMILES disappeared for {cas}")
        fragmentation = nm.fragment(molecule)
        groups = {
            f"correction_{group}": int(count)
            for group, count in fragmentation.corrections.items()
            if count > 0
        }
        interaction_counts: Counter = Counter()
        for index, first in enumerate(fragmentation.interaction_classes):
            for second in fragmentation.interaction_classes[index + 1 :]:
                interaction_counts["".join(sorted((first, second)))] += 1
        groups.update(
            {
                f"interaction_{pair}": int(count)
                for pair, count in interaction_counts.items()
            }
        )
        output[cas] = groups
    return output


def regression_features(
    curves: Sequence[Curve],
    second_order: dict[str, dict[str, int]],
    group_order: str,
    minimum_compounds: int,
) -> list[str]:
    features = feature_names(curves, minimum_compounds)
    if group_order == FIRST_ORDER:
        return features
    support: Counter = Counter()
    for groups in second_order.values():
        support.update(groups.keys())
    return [
        *features,
        *sorted(
            group for group, count in support.items() if count >= minimum_compounds
        ),
    ]


def regression_group_count(
    curve: Curve,
    second_order: dict[str, dict[str, int]],
    feature: str,
) -> int:
    if feature.startswith(("correction_", "interaction_")):
        return second_order[curve.cas].get(feature, 0)
    return curve.groups.get(feature, 0)


def compound_folds(cases: Sequence[str], folds: int, salt: str) -> dict[str, int]:
    ordered = sorted(
        set(cases),
        key=lambda cas: hashlib.sha256(f"{salt}:{cas}".encode()).hexdigest(),
    )
    return {cas: index % folds for index, cas in enumerate(ordered)}


def build_data(
    curves: Sequence[Curve],
    second_order: dict[str, dict[str, int]],
    features: Sequence[str],
    degree: int,
    mass_rule: str = "none",
    mass_temperature_degree: int | None = None,
    boiling_point_rule: str = "none",
    boiling_point_temperature_degree: int | None = None,
) -> dict:
    base_groups = np.asarray(
        [
            [
                regression_group_count(curve, second_order, feature)
                for feature in features
            ]
            for curve in curves
        ],
        dtype=float,
    )
    molecular_weight = np.asarray([curve.molecular_weight for curve in curves])
    centered_mass = molecular_weight / MASS_REFERENCE_G_PER_MOL - 1.0
    log_mass = np.log(molecular_weight / MASS_REFERENCE_G_PER_MOL)
    labels = list(features)
    augmented = [base_groups]
    if mass_rule == "global_linear":
        augmented.append(centered_mass[:, None])
        labels.append("molar_mass_linear")
    elif mass_rule == "global_quadratic":
        augmented.extend((centered_mass[:, None], centered_mass[:, None] ** 2))
        labels.extend(("molar_mass_linear", "molar_mass_quadratic"))
    elif mass_rule == "global_log":
        augmented.append(log_mass[:, None])
        labels.append("molar_mass_log")
    elif mass_rule == "global_inverse_sqrt":
        augmented.append(
            (np.sqrt(MASS_REFERENCE_G_PER_MOL / molecular_weight) - 1.0)[:, None]
        )
        labels.append("molar_mass_inverse_sqrt")
    elif mass_rule == "group_linear":
        augmented.extend((centered_mass[:, None], base_groups * centered_mass[:, None]))
        labels.extend(
            ("molar_mass_linear", *(f"molar_mass_linear_x_{name}" for name in features))
        )
    elif mass_rule == "group_log":
        augmented.extend((log_mass[:, None], base_groups * log_mass[:, None]))
        labels.extend(
            ("molar_mass_log", *(f"molar_mass_log_x_{name}" for name in features))
        )
    elif mass_rule != "none":
        raise ValueError(f"unknown molar-mass rule {mass_rule!r}")
    if boiling_point_rule != "none":
        if any(curve.normal_boiling_point is None for curve in curves):
            raise ValueError("normal boiling point is unavailable")
        boiling_points = np.asarray(
            [curve.normal_boiling_point for curve in curves],
            dtype=float,
        )
        centered_boiling = boiling_points / BOILING_POINT_REFERENCE_K - 1.0
        if boiling_point_rule == "global_linear":
            augmented.append(centered_boiling[:, None])
            labels.append("boiling_point_linear")
        elif boiling_point_rule == "global_quadratic":
            augmented.extend(
                (centered_boiling[:, None], centered_boiling[:, None] ** 2)
            )
            labels.extend(("boiling_point_linear", "boiling_point_quadratic"))
        elif boiling_point_rule == "global_log":
            augmented.append(
                np.log(boiling_points / BOILING_POINT_REFERENCE_K)[:, None]
            )
            labels.append("boiling_point_log")
        else:
            raise ValueError(f"unknown boiling-point rule {boiling_point_rule!r}")
    groups_by_curve = np.column_stack((np.ones(len(curves)), *augmented))
    group_scales = np.sqrt(np.mean(groups_by_curve * groups_by_curve, axis=0))
    group_scales = np.where(group_scales > 1.0e-12, group_scales, 1.0)
    groups_by_curve /= group_scales
    curve_cases = np.asarray([curve.cas for curve in curves])
    curve_index = np.concatenate(
        [np.full(len(curve.temperatures), index) for index, curve in enumerate(curves)]
    )
    temperatures = np.concatenate([curve.temperatures for curve in curves])
    reference = np.concatenate([curve.reference for curve in curves])
    groups_by_point = groups_by_curve[curve_index]
    reduced_temperature = (temperatures - T_CENTER_K) / T_SCALE_K
    mass_indices = [
        index + 1
        for index, label in enumerate(labels)
        if label.startswith("molar_mass")
    ]
    boiling_point_indices = [
        index + 1
        for index, label in enumerate(labels)
        if label.startswith("boiling_point")
    ]
    blocks = []
    for power in range(degree + 1):
        block = groups_by_point * reduced_temperature[:, None] ** power
        if (
            mass_temperature_degree is not None
            and power > mass_temperature_degree
            and mass_indices
        ):
            block = block.copy()
            block[:, mass_indices] = 0.0
        if (
            boiling_point_temperature_degree is not None
            and power > boiling_point_temperature_degree
            and boiling_point_indices
        ):
            block = block.copy()
            block[:, boiling_point_indices] = 0.0
        blocks.append(block)
    return {
        "degree": degree,
        "width": groups_by_curve.shape[1],
        "features": np.asarray(["intercept", *labels]),
        "mass_rule": mass_rule,
        "mass_temperature_degree": mass_temperature_degree,
        "boiling_point_rule": boiling_point_rule,
        "boiling_point_temperature_degree": boiling_point_temperature_degree,
        "group_scales": group_scales,
        "curve_cases": curve_cases,
        "curve_index": curve_index,
        "point_cases": curve_cases[curve_index],
        "temperatures": temperatures,
        "reference": reference,
        "design": np.concatenate(blocks, axis=1),
    }


def point_weights(data: dict, point_mask: np.ndarray) -> np.ndarray:
    cases = data["point_cases"][point_mask]
    counts = Counter(cases)
    return np.asarray([1.0 / counts[cas] for cas in cases])


def fit(data: dict, curve_mask: np.ndarray, alpha: float) -> np.ndarray:
    point_mask = curve_mask[data["curve_index"]]
    reference = data["reference"][point_mask]
    design = data["design"][point_mask] / reference[:, None]
    weights = point_weights(data, point_mask)
    weight_sum = float(np.sum(weights))
    scales = np.sqrt(np.sum(weights[:, None] * design * design, axis=0) / weight_sum)
    scales = np.where(scales > 1.0e-14, scales, 1.0)
    standardized = design / scales
    gram = standardized.T @ (weights[:, None] * standardized) / weight_sum
    rhs = standardized.T @ weights / weight_sum
    penalty = alpha * np.identity(standardized.shape[1])
    for power in range(data["degree"] + 1):
        penalty[power * data["width"], power * data["width"]] = 0.0
    coefficients = np.linalg.solve(
        gram + penalty + 1.0e-12 * np.identity(gram.shape[0]),
        rhs,
    )
    return coefficients / scales


def predict(data: dict, coefficients: np.ndarray) -> np.ndarray:
    return data["design"] @ coefficients


def compound_equal_mape(
    data: dict,
    predictions: np.ndarray,
    curve_mask: np.ndarray,
) -> float:
    point_mask = curve_mask[data["curve_index"]]
    cases = data["point_cases"][point_mask]
    errors = np.abs(predictions[point_mask] / data["reference"][point_mask] - 1.0)
    return float(np.mean([np.mean(errors[cases == cas]) for cas in sorted(set(cases))]))


def select_alpha(
    data: dict,
    outer_training: np.ndarray,
    folds: int,
    salt: str,
) -> float:
    cases = data["curve_cases"]
    assignments = compound_folds(cases[outer_training], folds, salt)
    scores = []
    for alpha in RIDGE_ALPHAS:
        fold_scores = []
        for fold in range(folds):
            validation = np.asarray(
                [
                    outer_training[index] and assignments.get(cas) == fold
                    for index, cas in enumerate(cases)
                ]
            )
            coefficients = fit(data, outer_training & ~validation, alpha)
            fold_scores.append(
                compound_equal_mape(data, predict(data, coefficients), validation)
            )
        scores.append((sum(fold_scores) / len(fold_scores), alpha))
    return min(scores)[1]


def cross_validated_predictions(
    data: dict,
    outer_folds: int,
    inner_folds: int,
) -> tuple[np.ndarray, list[float]]:
    cases = data["curve_cases"]
    assignments = compound_folds(cases, outer_folds, "polynomial_gc_outer")
    predictions = np.full(len(data["reference"]), np.nan)
    alphas = []
    for fold in range(outer_folds):
        test_curves = np.asarray([assignments[cas] == fold for cas in cases])
        training_curves = ~test_curves
        alpha = select_alpha(
            data,
            training_curves,
            inner_folds,
            f"polynomial_gc_inner_{fold}",
        )
        alphas.append(alpha)
        coefficients = fit(data, training_curves, alpha)
        test_points = test_curves[data["curve_index"]]
        predictions[test_points] = predict(data, coefficients)[test_points]
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("cross-validation left nonfinite predictions")
    return predictions, alphas


def metrics(data: dict, predictions: np.ndarray) -> dict:
    signed = 100.0 * (predictions / data["reference"] - 1.0)
    absolute = np.abs(signed)
    curve_mapes = {}
    for index, cas in enumerate(data["curve_cases"]):
        mask = data["curve_index"] == index
        curve_mapes.setdefault(cas, []).append(float(np.mean(absolute[mask])))
    compound_mapes = {
        cas: float(np.mean(values)) for cas, values in curve_mapes.items()
    }
    worst_point = int(np.argmax(absolute))
    worst_compound = max(compound_mapes, key=compound_mapes.get)
    return {
        "points": len(predictions),
        "compounds": len(compound_mapes),
        "mape_percent": float(np.mean(absolute)),
        "median_ape_percent": float(np.median(absolute)),
        "p90_ape_percent": float(percentile(absolute, 0.90)),
        "p95_ape_percent": float(percentile(absolute, 0.95)),
        "maximum_ape_percent": float(np.max(absolute)),
        "mean_signed_error_percent": float(np.mean(signed)),
        "nonpositive_prediction_fraction": float(np.mean(predictions <= 0.0)),
        "compound_equal_mean_curve_mape_percent": float(
            np.mean(list(compound_mapes.values()))
        ),
        "compound_equal_median_curve_mape_percent": float(
            np.median(list(compound_mapes.values()))
        ),
        "worst_point": {
            "cas": str(data["point_cases"][worst_point]),
            "temperature_K": float(data["temperatures"][worst_point]),
            "signed_error_percent": float(signed[worst_point]),
        },
        "worst_compound": {
            "cas": worst_compound,
            "curve_mape_percent": compound_mapes[worst_compound],
        },
    }


def independent_curve_floor(data: dict) -> dict:
    predictions = np.empty(len(data["reference"]))
    degree = data["degree"]
    for curve_index in range(len(data["curve_cases"])):
        mask = data["curve_index"] == curve_index
        temperature = data["temperatures"][mask]
        reference = data["reference"][mask]
        reduced = (temperature - T_CENTER_K) / T_SCALE_K
        basis = np.column_stack([reduced**power for power in range(degree + 1)])
        coefficients = np.linalg.lstsq(
            basis / reference[:, None],
            np.ones(len(reference)),
            rcond=None,
        )[0]
        predictions[mask] = basis @ coefficients
    return metrics(data, predictions)


def raw_kelvin_coefficients(centered: Sequence[float]) -> list[float]:
    degree = len(centered) - 1
    raw = []
    for power in range(degree + 1):
        raw.append(
            sum(
                centered[source_power]
                * math.comb(source_power, power)
                * (-T_CENTER_K) ** (source_power - power)
                / T_SCALE_K**source_power
                for source_power in range(power, degree + 1)
            )
        )
    return raw


def coefficient_payload(data: dict, coefficients: np.ndarray) -> dict:
    width = data["width"]
    blocks = coefficients.reshape(data["degree"] + 1, width)
    output = {}
    names = ("A", "B", "C", "D")
    for feature_index, feature in enumerate(data["features"]):
        centered = blocks[:, feature_index] / data["group_scales"][feature_index]
        raw = raw_kelvin_coefficients(centered)
        output[str(feature)] = {
            names[power]: float(value) for power, value in enumerate(raw)
        }
    return output


def run(args: argparse.Namespace) -> tuple[dict, str]:
    curves, exclusions = load_curves(args)
    if not curves:
        raise RuntimeError("no Nannoolal-fragmentable Perry curves")
    if any(rule != "none" for rule in args.boiling_point_rules):
        missing_boiling = {
            curve.cas for curve in curves if curve.normal_boiling_point is None
        }
        if missing_boiling:
            exclusions["Perry normal boiling point unavailable"] += len(missing_boiling)
            curves = [curve for curve in curves if curve.cas not in missing_boiling]
    second_order = second_order_groups_by_cas(curves)
    features_by_order = {
        group_order: regression_features(
            curves,
            second_order,
            group_order,
            args.minimum_group_compounds,
        )
        for group_order in args.group_orders
    }
    results = {}
    for group_order in args.group_orders:
        features = features_by_order[group_order]
        for degree in args.degrees:
            for mass_rule in args.mass_rules:
                for boiling_point_rule in args.boiling_point_rules:
                    mass_degrees = (
                        args.mass_temperature_degrees
                        if mass_rule != "none"
                        else (None,)
                    )
                    boiling_degrees = (
                        args.boiling_point_temperature_degrees
                        if boiling_point_rule != "none"
                        else (None,)
                    )
                    for mass_temperature_degree, boiling_temperature_degree in product(
                        mass_degrees,
                        boiling_degrees,
                    ):
                        data = build_data(
                            curves,
                            second_order,
                            features,
                            degree,
                            mass_rule,
                            mass_temperature_degree,
                            boiling_point_rule,
                            boiling_temperature_degree,
                        )
                        predictions, outer_alphas = cross_validated_predictions(
                            data,
                            args.outer_folds,
                            args.inner_folds,
                        )
                        result = metrics(data, predictions)
                        all_curves = np.ones(len(curves), dtype=bool)
                        final_alpha = select_alpha(
                            data,
                            all_curves,
                            args.outer_folds,
                            f"polynomial_gc_final_{group_order}_degree_{degree}_"
                            f"{mass_rule}_massT_{mass_temperature_degree}_"
                            f"Tb_{boiling_point_rule}_TbT_{boiling_temperature_degree}",
                        )
                        final_coefficients = fit(data, all_curves, final_alpha)
                        result.update(
                            {
                                "degree": degree,
                                "group_order": group_order,
                                "mass_rule": mass_rule,
                                "mass_temperature_degree": mass_temperature_degree,
                                "boiling_point_rule": boiling_point_rule,
                                "boiling_point_temperature_degree": (
                                    boiling_temperature_degree
                                ),
                                "features": [str(value) for value in data["features"]],
                                "selected_outer_alphas": outer_alphas,
                                "final_alpha": final_alpha,
                                "independent_curve_floor": independent_curve_floor(
                                    data
                                ),
                                "full_fit_raw_kelvin_coefficients": coefficient_payload(
                                    data,
                                    final_coefficients,
                                ),
                            }
                        )
                        key = (
                            f"{group_order}_degree_{degree}_mass_{mass_rule}_"
                            f"massT_{mass_temperature_degree}_Tb_{boiling_point_rule}_"
                            f"TbT_{boiling_temperature_degree}"
                        )
                        results[key] = result

    best_key = min(results, key=lambda key: results[key]["mape_percent"])
    admitted_second_order = sorted(
        feature
        for feature in features_by_order.get(FIRST_PLUS_SECOND_ORDER, ())
        if feature.startswith(("correction_", "interaction_"))
    )
    payload = {
        "schema_version": 1,
        "model": "direct_first_order_nannoolal_vapor_conductivity_polynomial_gc",
        "equations": {
            "quadratic": "k=A+B*T+C*T^2",
            "cubic": "k=A+B*T+C*T^2+D*T^3",
            "internal_temperature": "(T-500 K)/500 K",
        },
        "molar_mass": {
            "reference_g_per_mol": MASS_REFERENCE_G_PER_MOL,
            "tested_rules": args.mass_rules,
            "tested_temperature_degrees": args.mass_temperature_degrees,
        },
        "normal_boiling_point": {
            "reference_K": BOILING_POINT_REFERENCE_K,
            "tested_rules": args.boiling_point_rules,
            "tested_temperature_degrees": args.boiling_point_temperature_degrees,
            "source": "Perry normal point, vapor-pressure inversion, or Table 2-10",
        },
        "group_basis": {
            "source": "nannoolal_method.fragment",
            "combination": "additive group counts",
            "minimum_compound_support": args.minimum_group_compounds,
            "tested_orders": args.group_orders,
            "features_by_order": features_by_order,
            "admitted_second_order": admitted_second_order,
        },
        "cross_validation": {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "fold_unit": "CAS compound",
            "ridge_alphas": RIDGE_ALPHAS,
            "fit_loss": "compound-equal weighted squared relative error",
            "selection_metric": "compound-equal MAPE",
        },
        "coverage": {
            "compounds": len({curve.cas for curve in curves}),
            "curves": len(curves),
            "sampled_states": sum(len(curve.temperatures) for curve in curves),
            "exclusions": dict(exclusions),
        },
        "best_model": best_key,
        "models": results,
    }
    lines = [
        "Direct polynomial first-order Nannoolal GC vapor conductivity",
        (
            f"coverage: {payload['coverage']['compounds']} compounds, "
            f"{payload['coverage']['curves']} curves, "
            f"{payload['coverage']['sampled_states']} states"
        ),
        ("groups: additive Nannoolal counts; no viscosity or heat capacity"),
        f"admitted second-order groups: {', '.join(admitted_second_order) or 'none'}",
        (
            f"validation: {args.outer_folds}-fold outer / "
            f"{args.inner_folds}-fold inner by whole compound"
        ),
        "",
        "HELD-OUT PERFORMANCE",
    ]
    for key, result in sorted(
        results.items(),
        key=lambda item: item[1]["mape_percent"],
    ):
        floor = result["independent_curve_floor"]
        lines.append(
            f"  {key:35s} MAPE={result['mape_percent']:7.2f}% "
            f"MdAPE={result['median_ape_percent']:7.2f}% "
            f"P95={result['p95_ape_percent']:7.2f}% "
            f"bias={result['mean_signed_error_percent']:+7.2f}% "
            f"nonpositive={100.0 * result['nonpositive_prediction_fraction']:5.2f}% "
            f"curve-floor={floor['mape_percent']:.3f}%"
        )
    best = results[best_key]
    lines.extend(
        (
            "",
            f"BEST: {best_key}",
            (
                f"  worst compound: {best['worst_compound']['cas']} "
                f"({best['worst_compound']['curve_mape_percent']:.2f}% curve MAPE)"
            ),
            (
                f"  worst point: {best['worst_point']['cas']} at "
                f"{best['worst_point']['temperature_K']:.2f} K, "
                f"{best['worst_point']['signed_error_percent']:+.2f}%"
            ),
            "",
            "EXCLUSIONS",
        )
    )
    for reason, count in exclusions.most_common():
        lines.append(f"  {count:4d} {reason}")
    return payload, "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conductivity-database",
        type=Path,
        default=DEFAULT_CONDUCTIVITY_DATABASE,
    )
    parser.add_argument("--points", type=int, default=31)
    parser.add_argument("--degrees", nargs="+", type=int, choices=(2, 3), default=(2,))
    parser.add_argument(
        "--group-orders",
        nargs="+",
        choices=GROUP_ORDERS,
        default=(FIRST_PLUS_SECOND_ORDER,),
    )
    parser.add_argument(
        "--mass-rules",
        nargs="+",
        choices=MASS_RULES,
        default=("global_linear",),
    )
    parser.add_argument(
        "--mass-temperature-degrees",
        nargs="+",
        type=int,
        choices=(0, 1, 2, 3),
        default=(1,),
        help="Highest polynomial power receiving each mass feature (default: 1).",
    )
    parser.add_argument(
        "--boiling-point-rules",
        nargs="+",
        choices=BOILING_POINT_RULES,
        default=("none",),
    )
    parser.add_argument(
        "--boiling-point-temperature-degrees",
        nargs="+",
        type=int,
        choices=(0, 1, 2, 3),
        default=(2,),
        help="Highest polynomial power receiving each Tb feature (default: 2).",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--minimum-group-compounds", type=int, default=3)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.points < 4:
        parser.error("--points must be at least 4")
    if args.outer_folds < 2 or args.inner_folds < 2:
        parser.error("outer and inner folds must both be at least 2")
    if args.minimum_group_compounds < 2:
        parser.error("--minimum-group-compounds must be at least 2")
    return args


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    args = parse_args()
    payload, report = run(args)
    print(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
