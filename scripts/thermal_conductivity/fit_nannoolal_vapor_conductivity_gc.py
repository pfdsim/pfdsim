#!/usr/bin/env python3
"""Fit a direct first-order Nannoolal GC model for vapor conductivity.

The curve is exactly representable as k = A*T**n/(1 + B/T), but is fitted in
a reference-temperature form for numerical conditioning:

    k = K0*(T/T0)**n * (1 + B/T0)/(1 + B/T)

For molecule i with first-order Nannoolal counts v_ig:

    log(K0_i) = a0 + sum_g(v_ig*a_g)
    B_i       = B_scale*softplus(b0 + sum_g(v_ig*b_g))
    n_i       = -0.5 + 3*sigmoid(c0 + sum_g(v_ig*c_g))
    A_i       = K0_i*(1 + B_i/T0)/T0**n_i

Only Fragmentation.groups and first-order silicon environments are used;
Nannoolal second-order corrections and interaction classes are deliberately
ignored. Evaluation uses nested whole-compound cross-validation against Perry
9th Table 2-145, with no viscosity or heat-capacity input.

The requested three-parameter curve is first fitted independently to each
Perry reference curve. Ridge regression then maps Nannoolal group vectors to
those latent curve parameters. This separates curve-form error from GC error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from chemicals.identifiers import search_chemical
from rdkit import Chem, RDLogger
from scipy.optimize import least_squares
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nannoolal_method as nm  # noqa: E402
from perry_properties import PerryPropertyLibrary  # noqa: E402
from property_resolution.organic_classification import (  # noqa: E402
    classify_strict_molecular_organic,
)
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    DEFAULT_CONDUCTIVITY_DATABASE,
    conductivity_reference,
    percentile,
)


REFERENCE_TEMPERATURE_K = 500.0
B_SCALE_K = 300.0
N_MIN = -0.5
N_MAX = 2.5
RIDGE_ALPHAS = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
GC_EXCLUDED_CAS = {
    "144-62-7": "manual GC exclusion: oxalic acid is a universal GC outlier",
}


@dataclass(frozen=True)
class Curve:
    cas: str
    name: str
    formula: str
    molecular_weight: float
    normal_boiling_point: float | None
    temperatures: np.ndarray
    reference: np.ndarray
    groups: dict[str, int]


def local_smiles(cas: str) -> str:
    try:
        return str(search_chemical(cas).smiles or "").strip()
    except Exception:
        return ""


def nannoolal_first_order_groups(smiles: str) -> dict[str, int]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("local SMILES unavailable or invalid")
    fragmentation = nm.fragment(molecule)
    groups = {str(group): int(count) for group, count in fragmentation.groups.items()}
    for environment in fragmentation.si_atoms:
        label = (
            f"Si_CH{environment['n_ch']}_"
            f"hal{int(environment['has_halogen'])}_o{int(environment['has_o'])}"
        )
        groups[label] = groups.get(label, 0) + 1
    return groups


def load_curves(args: argparse.Namespace) -> tuple[list[Curve], Counter]:
    chemicals = json.loads(args.conductivity_database.read_text(encoding="utf-8"))[
        "chemicals"
    ]
    curves = []
    exclusions: Counter = Counter()
    perry = PerryPropertyLibrary()
    for cas, entry in sorted(chemicals.items()):
        rows = entry.get("vapor_thermal_conductivity") or []
        if not rows:
            continue
        if cas in GC_EXCLUDED_CAS:
            exclusions[GC_EXCLUDED_CAS[cas]] += 1
            continue
        smiles = local_smiles(cas)
        classification = classify_strict_molecular_organic(
            cas=cas,
            formula=entry.get("formula"),
            smiles=smiles,
        )
        if not classification.is_organic:
            exclusions[f"not strict organic: {classification.reason}"] += 1
            continue
        try:
            groups = nannoolal_first_order_groups(smiles)
        except Exception as exc:
            exclusions[f"{type(exc).__name__}: {exc}"] += 1
            continue
        boiling = perry.normal_boiling_point_K(cas)
        if boiling is None:
            boiling = perry.table_2_10_normal_boiling_point_K(cas)
        normal_boiling_point = float(boiling.value) if boiling is not None else None
        for row in rows:
            temperatures = np.linspace(
                float(row["T_min_K"]),
                float(row["T_max_K"]),
                args.points,
            )
            reference = np.asarray(
                [conductivity_reference(row, float(T)) for T in temperatures]
            )
            curves.append(
                Curve(
                    cas=cas,
                    name=str(entry.get("name") or cas),
                    formula=str(entry.get("formula") or ""),
                    molecular_weight=float(row["molecular_weight"]),
                    normal_boiling_point=normal_boiling_point,
                    temperatures=temperatures,
                    reference=reference,
                    groups=groups,
                )
            )
    return curves, exclusions


def feature_names(curves: Sequence[Curve], minimum_compounds: int) -> list[str]:
    support: Counter = Counter()
    groups_by_cas = {curve.cas: curve.groups for curve in curves}
    for groups in groups_by_cas.values():
        support.update(groups.keys())
    return sorted(
        group for group, count in support.items() if count >= minimum_compounds
    )


def compound_folds(cases: Sequence[str], folds: int, salt: str) -> dict[str, int]:
    ordered = sorted(
        set(cases),
        key=lambda cas: hashlib.sha256(f"{salt}:{cas}".encode()).hexdigest(),
    )
    return {cas: index % folds for index, cas in enumerate(ordered)}


def fit_curve_target(curve: Curve) -> np.ndarray:
    """Fit one Perry curve to latent log(K0), raw-B, and raw-n parameters."""
    temperatures = curve.temperatures
    target = np.log(curve.reference)
    x = np.log(temperatures / REFERENCE_TEMPERATURE_K)
    slope, intercept = np.polyfit(x, target, 1)
    bounded_n = min(N_MAX - 1.0e-4, max(N_MIN + 1.0e-4, slope))
    fraction = (bounded_n - N_MIN) / (N_MAX - N_MIN)
    initial = np.asarray(
        [
            intercept,
            math.log(math.expm1(1.0)),
            math.log(fraction / (1.0 - fraction)),
        ]
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        log_k0, raw_b, raw_n = parameters
        B = B_SCALE_K * np.logaddexp(0.0, raw_b)
        n = N_MIN + (N_MAX - N_MIN) * expit(raw_n)
        return (
            log_k0
            + n * x
            + np.log1p(B / REFERENCE_TEMPERATURE_K)
            - np.log1p(B / temperatures)
            - target
        )

    result = least_squares(
        residual,
        initial,
        bounds=([-20.0, -20.0, -20.0], [5.0, 20.0, 20.0]),
        max_nfev=2000,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"curve fit failed: {result.message}")
    return np.asarray(result.x)


def matrices(
    curves: Sequence[Curve], features: Sequence[str], combining_rule: str
) -> dict[str, np.ndarray]:
    counts = np.asarray(
        [[curve.groups.get(feature, 0) for feature in features] for curve in curves],
        dtype=float,
    )
    totals = np.asarray([sum(curve.groups.values()) for curve in curves], dtype=float)
    if np.any(totals <= 0.0):
        raise ValueError("Nannoolal fragmentation produced no first-order groups")
    if combining_rule == "additive_counts":
        transformed = counts
        labels = list(features)
    elif combining_rule == "log_saturated_counts":
        transformed = np.log1p(counts)
        labels = list(features)
    elif combining_rule == "fractions_plus_size":
        transformed = np.column_stack((counts / totals[:, None], np.log(totals)))
        labels = [*features, "log_total_first_order_groups"]
    else:
        raise ValueError(f"unknown combining rule {combining_rule!r}")
    scales = np.sqrt(np.mean(transformed * transformed, axis=0))
    scales = np.where(scales > 1.0e-12, scales, 1.0)
    group_rows = np.column_stack((np.ones(len(curves)), transformed / scales))
    curve_cases = np.asarray([curve.cas for curve in curves])
    curve_targets = np.asarray([fit_curve_target(curve) for curve in curves])
    temperatures = np.concatenate([curve.temperatures for curve in curves])
    reference = np.concatenate([curve.reference for curve in curves])
    curve_index = np.concatenate(
        [np.full(len(curve.temperatures), index) for index, curve in enumerate(curves)]
    )
    return {
        "groups_by_curve": group_rows,
        "groups_by_point": group_rows[curve_index],
        "curve_cases": curve_cases,
        "curve_targets": curve_targets,
        "curve_index": curve_index,
        "point_cases": curve_cases[curve_index],
        "temperatures": temperatures,
        "reference": reference,
        "feature_labels": np.asarray(["intercept", *labels]),
        "feature_scales": np.asarray([1.0, *scales]),
    }


def unpack(
    parameters: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        parameters[:width],
        parameters[width : 2 * width],
        parameters[2 * width :],
    )


def curve_parameters(
    parameters: np.ndarray,
    groups_by_curve: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = groups_by_curve.shape[1]
    a, b, exponent = unpack(parameters, width)
    log_k0 = groups_by_curve @ a
    raw_b = groups_by_curve @ b
    raw_n = groups_by_curve @ exponent
    B = B_SCALE_K * np.logaddexp(0.0, raw_b)
    n = N_MIN + (N_MAX - N_MIN) * expit(raw_n)
    A = (
        np.exp(log_k0)
        * (1.0 + B / REFERENCE_TEMPERATURE_K)
        / REFERENCE_TEMPERATURE_K**n
    )
    return log_k0, B, n, A


def predict_log_k(parameters: np.ndarray, data: dict) -> np.ndarray:
    width = data["groups_by_point"].shape[1]
    a, b, exponent = unpack(parameters, width)
    groups = data["groups_by_point"]
    temperatures = data["temperatures"]
    raw_b = groups @ b
    B = B_SCALE_K * np.logaddexp(0.0, raw_b)
    n = N_MIN + (N_MAX - N_MIN) * expit(groups @ exponent)
    return (
        groups @ a
        + n * np.log(temperatures / REFERENCE_TEMPERATURE_K)
        + np.log1p(B / REFERENCE_TEMPERATURE_K)
        - np.log1p(B / temperatures)
    )


def curve_target_log_k(data: dict) -> np.ndarray:
    latents = data["curve_targets"][data["curve_index"]]
    temperatures = data["temperatures"]
    log_k0 = latents[:, 0]
    B = B_SCALE_K * np.logaddexp(0.0, latents[:, 1])
    n = N_MIN + (N_MAX - N_MIN) * expit(latents[:, 2])
    return (
        log_k0
        + n * np.log(temperatures / REFERENCE_TEMPERATURE_K)
        + np.log1p(B / REFERENCE_TEMPERATURE_K)
        - np.log1p(B / temperatures)
    )


def fit(data: dict, curve_mask: np.ndarray, alpha: float) -> np.ndarray:
    groups = data["groups_by_curve"][curve_mask]
    targets = data["curve_targets"][curve_mask]
    cases = data["curve_cases"][curve_mask]
    counts = Counter(cases)
    weights = np.asarray([1.0 / counts[cas] for cas in cases])
    weight_sum = float(np.sum(weights))
    gram = groups.T @ (weights[:, None] * groups) / weight_sum
    penalty = alpha * np.identity(groups.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + penalty + 1.0e-12 * np.identity(groups.shape[1]),
        groups.T @ (weights[:, None] * targets) / weight_sum,
    )
    return np.concatenate((coefficients[:, 0], coefficients[:, 1], coefficients[:, 2]))


def compound_equal_mape(
    data: dict,
    predicted_log_k: np.ndarray,
    curve_mask: np.ndarray,
) -> float:
    point_mask = curve_mask[data["curve_index"]]
    cases = data["point_cases"][point_mask]
    errors = np.abs(
        np.exp(predicted_log_k[point_mask]) / data["reference"][point_mask] - 1.0
    )
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
            parameters = fit(data, outer_training & ~validation, alpha)
            fold_scores.append(
                compound_equal_mape(data, predict_log_k(parameters, data), validation)
            )
        scores.append((sum(fold_scores) / len(fold_scores), alpha))
    return min(scores)[1]


def cross_validated_predictions(
    data: dict,
    outer_folds: int,
    inner_folds: int,
) -> tuple[np.ndarray, list[float]]:
    cases = data["curve_cases"]
    assignments = compound_folds(
        cases,
        outer_folds,
        "nannoolal_vapor_conductivity_outer",
    )
    predictions = np.full(len(data["reference"]), np.nan)
    selected_alphas = []
    for fold in range(outer_folds):
        test_curves = np.asarray([assignments[cas] == fold for cas in cases])
        training_curves = ~test_curves
        alpha = select_alpha(
            data,
            training_curves,
            inner_folds,
            f"nannoolal_vapor_conductivity_inner_{fold}",
        )
        selected_alphas.append(alpha)
        parameters = fit(data, training_curves, alpha)
        all_predictions = predict_log_k(parameters, data)
        test_points = test_curves[data["curve_index"]]
        predictions[test_points] = all_predictions[test_points]
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("cross-validation left nonfinite predictions")
    return predictions, selected_alphas


def metrics(data: dict, predicted_log_k: np.ndarray) -> dict:
    predicted = np.exp(predicted_log_k)
    signed = 100.0 * (predicted / data["reference"] - 1.0)
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
        "points": len(predicted),
        "compounds": len(compound_mapes),
        "mape_percent": float(np.mean(absolute)),
        "median_ape_percent": float(np.median(absolute)),
        "p90_ape_percent": percentile(absolute, 0.90),
        "p95_ape_percent": percentile(absolute, 0.95),
        "maximum_ape_percent": float(np.max(absolute)),
        "mean_signed_error_percent": float(np.mean(signed)),
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


def coefficient_payload(parameters: np.ndarray, data: dict) -> dict:
    width = data["groups_by_curve"].shape[1]
    scales = data["feature_scales"]
    a, b, exponent = unpack(parameters, width)
    return {
        str(label): {
            "log_K0": float(a[index] / scales[index]),
            "raw_B": float(b[index] / scales[index]),
            "raw_n": float(exponent[index] / scales[index]),
        }
        for index, label in enumerate(data["feature_labels"])
    }


def parameter_summary(parameters: np.ndarray, data: dict) -> dict:
    log_k0, B, n, A = curve_parameters(parameters, data["groups_by_curve"])
    return {
        "K0_W_per_m_K": {
            "minimum": float(np.min(np.exp(log_k0))),
            "median": float(np.median(np.exp(log_k0))),
            "maximum": float(np.max(np.exp(log_k0))),
        },
        "B_K": {
            "minimum": float(np.min(B)),
            "median": float(np.median(B)),
            "maximum": float(np.max(B)),
        },
        "n": {
            "minimum": float(np.min(n)),
            "median": float(np.median(n)),
            "maximum": float(np.max(n)),
        },
        "A_derived": {
            "minimum": float(np.min(A)),
            "median": float(np.median(A)),
            "maximum": float(np.max(A)),
        },
    }


def run(args: argparse.Namespace) -> tuple[dict, str]:
    curves, exclusions = load_curves(args)
    if not curves:
        raise RuntimeError("no Nannoolal-fragmentable Perry curves")
    features = feature_names(curves, args.minimum_group_compounds)
    results = {}
    curve_form_metrics = None
    for rule in args.combining_rules:
        data = matrices(curves, features, rule)
        if curve_form_metrics is None:
            curve_form_metrics = metrics(data, curve_target_log_k(data))
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
            f"nannoolal_vapor_conductivity_final_{rule}",
        )
        final_parameters = fit(data, all_curves, final_alpha)
        result.update(
            {
                "selected_outer_alphas": outer_alphas,
                "final_alpha": final_alpha,
                "coefficients": coefficient_payload(final_parameters, data),
                "full_fit_parameter_summary": parameter_summary(
                    final_parameters,
                    data,
                ),
            }
        )
        results[rule] = result

    best_rule = min(results, key=lambda rule: results[rule]["mape_percent"])
    payload = {
        "schema_version": 1,
        "model": "direct_first_order_nannoolal_vapor_thermal_conductivity_gc",
        "curve": {
            "equation": "k=A*T^n/(1+B/T)",
            "reference_temperature_K": REFERENCE_TEMPERATURE_K,
            "B_scale_K": B_SCALE_K,
            "n_bounds": [N_MIN, N_MAX],
            "parameterization": (
                "log(K0)=linear GC; B=B_scale*softplus(linear GC); "
                "n=-0.5+3*sigmoid(linear GC); "
                "A=K0*(1+B/T0)/T0^n"
            ),
        },
        "group_basis": {
            "source": "nannoolal_method.fragment",
            "terms": "first-order groups and first-order silicon environments only",
            "second_order_corrections": False,
            "feature_count": len(features),
            "features": features,
            "minimum_compound_support": args.minimum_group_compounds,
        },
        "cross_validation": {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "fold_unit": "CAS compound",
            "ridge_alphas": RIDGE_ALPHAS,
            "fit_loss": "compound-equal weighted squared log conductivity error",
            "selection_metric": "compound-equal MAPE",
        },
        "coverage": {
            "compounds": len({curve.cas for curve in curves}),
            "curves": len(curves),
            "sampled_states": sum(len(curve.temperatures) for curve in curves),
            "exclusions": dict(exclusions),
        },
        "independent_curve_form_fit": curve_form_metrics,
        "best_combining_rule": best_rule,
        "combining_rules": results,
    }
    lines = [
        "Direct first-order Nannoolal GC vapor-conductivity regression",
        (
            f"coverage: {payload['coverage']['compounds']} compounds, "
            f"{payload['coverage']['curves']} curves, "
            f"{payload['coverage']['sampled_states']} states"
        ),
        (
            f"groups: {len(features)} first-order features; "
            "no viscosity, heat capacity, or second-order groups"
        ),
        (
            f"validation: {args.outer_folds}-fold outer / "
            f"{args.inner_folds}-fold inner by whole compound"
        ),
        (
            "requested curve-form floor from independent per-curve fits: "
            f"MAPE={curve_form_metrics['mape_percent']:.3f}%, "
            f"P95={curve_form_metrics['p95_ape_percent']:.3f}%"
        ),
        "",
        "HELD-OUT PERFORMANCE BY COMBINING RULE",
    ]
    for rule, result in sorted(
        results.items(),
        key=lambda item: item[1]["mape_percent"],
    ):
        lines.append(
            f"  {rule:24s} MAPE={result['mape_percent']:7.2f}% "
            f"MdAPE={result['median_ape_percent']:7.2f}% "
            f"P95={result['p95_ape_percent']:7.2f}% "
            f"bias={result['mean_signed_error_percent']:+7.2f}% "
            f"curve_mean={result['compound_equal_mean_curve_mape_percent']:7.2f}%"
        )
    best = results[best_rule]
    lines.extend(
        (
            "",
            f"BEST: {best_rule}",
            (
                f"  worst compound: {best['worst_compound']['cas']} "
                f"({best['worst_compound']['curve_mape_percent']:.2f}% curve MAPE)"
            ),
            (
                f"  worst point: {best['worst_point']['cas']} at "
                f"{best['worst_point']['temperature_K']:.2f} K, "
                f"{best['worst_point']['signed_error_percent']:+.2f}%"
            ),
            f"  fitted parameter ranges: {best['full_fit_parameter_summary']}",
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
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--minimum-group-compounds", type=int, default=3)
    parser.add_argument(
        "--combining-rules",
        nargs="+",
        choices=("additive_counts", "log_saturated_counts", "fractions_plus_size"),
        default=("additive_counts", "log_saturated_counts", "fractions_plus_size"),
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.points < 3:
        parser.error("--points must be at least 3")
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
