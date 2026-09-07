#!/usr/bin/env python3
"""Benchmark resolved-dipole Reichenberg on the zero-dipole Perry population.

The eligible population and strict Reichenberg fragmentation are taken from
``benchmark_reichenberg_zero_dipole.py``.  Dipoles are resolved from the local
CCCBDB table first; compounds absent there use GFN2-xTB.  PBE0 cache entries
and functional-class dipole heuristics are deliberately excluded so the two
reported source populations remain unambiguous.

GFN2-xTB geometries and results use pfdsim's persistent dipole cache, so an
interrupted run can resume without repeating completed quantum calculations.
Progress is written to stderr and the human-readable report to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reichenberg_method as rm  # noqa: E402
from perry_properties import PerryPropertyLibrary  # noqa: E402
from property_resolver import PropertyResolver  # noqa: E402
from scripts.vapor_viscosity.benchmark_reichenberg_zero_dipole import (  # noqa: E402
    DEFAULT_DATABASE,
    benchmark as zero_dipole_benchmark,
    distribution,
    error_metrics,
    format_distribution,
    mean,
    signed_distribution,
)


ALLOWED_DIPOLE_METHODS = {
    "cccbdb_experimental_dipole": "CCCBDB",
    "gfn2_xtb_dipole": "GFN2-xTB",
}
PRACTICAL_MAPE_TIE_POINTS = 0.01
PRACTICAL_METHOD_TIE_POINTS = 0.10


class NISTThenXTBResolver(PropertyResolver):
    """Use CCCBDB first and specifically GFN2-xTB for experimental misses."""

    def _load_dipole_artifact(self, identity: str, artifact: str):
        # The production resolver normally treats a cached PBE0 result as a
        # monotonic upgrade over xTB.  This benchmark is explicitly CCCBDB vs
        # xTB, so do not admit a pre-existing PBE0 result into the xTB cohort.
        if artifact == "result_pvdz":
            return None
        return super()._load_dipole_artifact(identity, artifact)


@dataclass(frozen=True)
class CompoundResult:
    cas: str
    name: str
    formula: str
    chemical_class: str
    smiles: str
    molecular_weight: float
    heavy_atoms: int
    size_bin: str
    groups: dict[str, int]
    contribution_sum: float
    Tmin_K: float
    Tmax_K: float
    Tr_min: float
    Tr_max: float
    dipole_D: float
    reduced_dipole: float
    dipole_source: str
    dipole_method: str
    dipole_quality: float
    yoon_thodos_mape_percent: float
    zero_dipole_mape_percent: float
    zero_dipole_bias_percent: float
    zero_dipole_p95_ape_percent: float
    zero_dipole_max_ape_percent: float
    resolved_dipole_mape_percent: float
    resolved_dipole_bias_percent: float
    resolved_dipole_p95_ape_percent: float
    resolved_dipole_max_ape_percent: float
    resolved_minus_zero_mape_points: float
    dipole_term_mean_effect_percent: float
    dipole_term_max_effect_percent: float


def molecular_size_bin(heavy_atoms: int) -> str:
    if heavy_atoms <= 5:
        return "1-5 heavy atoms"
    if heavy_atoms <= 10:
        return "6-10 heavy atoms"
    if heavy_atoms <= 15:
        return "11-15 heavy atoms"
    return "16+ heavy atoms"


def paired_summary(results: Sequence[CompoundResult]) -> dict:
    zero = [result.zero_dipole_mape_percent for result in results]
    resolved = [result.resolved_dipole_mape_percent for result in results]
    deltas = [result.resolved_minus_zero_mape_points for result in results]
    effects = [result.dipole_term_max_effect_percent for result in results]
    return {
        "zero_dipole_curve_mape_distribution": distribution(zero),
        "resolved_dipole_curve_mape_distribution": distribution(resolved),
        "resolved_minus_zero_mape_distribution": signed_distribution(deltas),
        "maximum_dipole_effect_distribution": distribution(effects),
        "resolved_dipole_wins": sum(
            delta < -PRACTICAL_MAPE_TIE_POINTS for delta in deltas
        ),
        "zero_dipole_wins": sum(
            delta > PRACTICAL_MAPE_TIE_POINTS for delta in deltas
        ),
        "ties": sum(abs(delta) <= PRACTICAL_MAPE_TIE_POINTS for delta in deltas),
        "within_0_1_mape_point": sum(abs(delta) <= 0.1 for delta in deltas),
        "within_1_mape_point": sum(abs(delta) <= 1.0 for delta in deltas),
    }


def method_selection_summary(
    results: Sequence[CompoundResult],
    choose_reichenberg: Sequence[bool],
) -> dict:
    selected = []
    oracle = []
    decisive = 0
    correct = 0
    for result, use_reichenberg in zip(results, choose_reichenberg):
        selected.append(
            result.zero_dipole_mape_percent
            if use_reichenberg
            else result.yoon_thodos_mape_percent
        )
        oracle.append(
            min(
                result.zero_dipole_mape_percent,
                result.yoon_thodos_mape_percent,
            )
        )
        delta = (
            result.zero_dipole_mape_percent
            - result.yoon_thodos_mape_percent
        )
        if abs(delta) > PRACTICAL_METHOD_TIE_POINTS:
            decisive += 1
            correct += use_reichenberg == (delta < 0.0)
    return {
        "compounds": len(results),
        "selected_reichenberg": sum(choose_reichenberg),
        "selected_yoon_thodos": len(results) - sum(choose_reichenberg),
        "mean_selected_mape_percent": mean(selected),
        "mean_oracle_regret_points": mean(
            [value - best for value, best in zip(selected, oracle)]
        ),
        "decisive_compounds": decisive,
        "correct_decisive_compounds": correct,
        "decisive_accuracy": correct / decisive if decisive else None,
    }


def best_method_threshold(
    results: Sequence[CompoundResult],
    feature: str,
) -> tuple[float, dict]:
    ordered = sorted(results, key=lambda result: float(getattr(result, feature)))
    selected_sum = sum(result.zero_dipole_mape_percent for result in ordered)
    choices = [(selected_sum / len(ordered), float("-inf"))]
    index = 0
    while index < len(ordered):
        value = float(getattr(ordered[index], feature))
        while (
            index < len(ordered)
            and float(getattr(ordered[index], feature)) == value
        ):
            selected_sum += (
                ordered[index].yoon_thodos_mape_percent
                - ordered[index].zero_dipole_mape_percent
            )
            index += 1
        threshold = (
            float("inf")
            if index == len(ordered)
            else (value + float(getattr(ordered[index], feature))) / 2.0
        )
        choices.append((selected_sum / len(ordered), threshold))
    _score, threshold = min(choices, key=lambda item: (item[0], item[1]))
    selected = [
        float(getattr(result, feature)) >= threshold for result in results
    ]
    return threshold, method_selection_summary(results, selected)


def leave_one_out_method_threshold(
    results: Sequence[CompoundResult],
    feature: str,
) -> dict:
    selected = []
    thresholds = []
    for held_out, result in enumerate(results):
        training = [
            candidate for index, candidate in enumerate(results)
            if index != held_out
        ]
        threshold, _summary = best_method_threshold(training, feature)
        thresholds.append(threshold)
        selected.append(float(getattr(result, feature)) >= threshold)
    summary = method_selection_summary(results, selected)
    finite = [threshold for threshold in thresholds if math.isfinite(threshold)]
    summary["threshold_distribution"] = (
        signed_distribution(finite)
        if finite
        else {
            "components": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "minimum": None,
            "maximum": None,
        }
    )
    summary["always_reichenberg_thresholds"] = sum(
        threshold == float("-inf") for threshold in thresholds
    )
    summary["always_yoon_thodos_thresholds"] = sum(
        threshold == float("inf") for threshold in thresholds
    )
    return summary


def method_classifier_analysis(results: Sequence[CompoundResult]) -> dict:
    analysis = {
        "practical_tie_points": PRACTICAL_METHOD_TIE_POINTS,
        "current_structural": method_selection_summary(
            results,
            [
                result.chemical_class == "sparse_heteroatom"
                for result in results
            ],
        ),
        "all_yoon_thodos": method_selection_summary(
            results,
            [False] * len(results),
        ),
        "all_reichenberg": method_selection_summary(
            results,
            [True] * len(results),
        ),
        "oracle": method_selection_summary(
            results,
            [
                result.zero_dipole_mape_percent
                < result.yoon_thodos_mape_percent
                for result in results
            ],
        ),
        "features": {},
    }
    for feature in ("dipole_D", "reduced_dipole"):
        threshold, fitted = best_method_threshold(results, feature)
        analysis["features"][feature] = {
            "fitted_threshold": threshold if math.isfinite(threshold) else None,
            "fitted_rule": (
                "always_reichenberg"
                if threshold == float("-inf")
                else "always_yoon_thodos"
                if threshold == float("inf")
                else "reichenberg_at_or_above_threshold"
            ),
            "fitted": fitted,
            "leave_one_out": leave_one_out_method_threshold(results, feature),
        }
    raw_threshold = analysis["features"]["dipole_D"]["fitted_threshold"]
    raw_rule = analysis["features"]["dipole_D"]["fitted_rule"]

    def raw_dipole_choice(result: CompoundResult) -> bool:
        if raw_rule == "always_reichenberg":
            return True
        if raw_rule == "always_yoon_thodos":
            return False
        return result.dipole_D >= float(raw_threshold)

    analysis["by_source_at_fitted_dipole_cutoff"] = {}
    for source in sorted({result.dipole_source for result in results}):
        subset = [result for result in results if result.dipole_source == source]
        analysis["by_source_at_fitted_dipole_cutoff"][source] = {
            "current_structural": method_selection_summary(
                subset,
                [
                    result.chemical_class == "sparse_heteroatom"
                    for result in subset
                ],
            ),
            "dipole_cutoff": method_selection_summary(
                subset,
                [raw_dipole_choice(result) for result in subset],
            ),
        }
    return analysis


def is_unbranched_terminal_1_alkyne(result: CompoundResult) -> bool:
    groups = result.groups
    return bool(
        result.chemical_class == "hydrocarbon"
        and groups.get("alkyne_ch") == 1
        and groups.get("alkyne_c") == 1
        and groups.get("ch2", 0) >= 1
        and groups.get("ch3") == 1
        and set(groups) <= {"alkyne_ch", "alkyne_c", "ch2", "ch3"}
    )


def is_unbranched_c3_plus_aldehyde(result: CompoundResult) -> bool:
    groups = result.groups
    return bool(
        result.chemical_class == "sparse_heteroatom"
        and groups.get("aldehyde") == 1
        and groups.get("ch2", 0) >= 1
        and groups.get("ch3") == 1
        and set(groups) <= {"aldehyde", "ch2", "ch3"}
    )


def structure_only_method_choice(result: CompoundResult) -> bool:
    """Return true for zero-dipole Reichenberg, false for Yoon-Thodos."""
    if is_unbranched_c3_plus_aldehyde(result):
        return False
    if result.chemical_class == "sparse_heteroatom":
        return True
    return is_unbranched_terminal_1_alkyne(result)


def structure_rule_analysis(results: Sequence[CompoundResult]) -> dict:
    terminal_alkynes = [
        result for result in results if is_unbranched_terminal_1_alkyne(result)
    ]
    linear_aldehydes = [
        result for result in results if is_unbranched_c3_plus_aldehyde(result)
    ]

    def family_summary(
        family: Sequence[CompoundResult],
        expected_method: str,
    ) -> dict:
        deltas = [
            result.zero_dipole_mape_percent
            - result.yoon_thodos_mape_percent
            for result in family
        ]
        return {
            "compounds": len(family),
            "expected_method": expected_method,
            "support_or_practical_tie": sum(
                delta <= PRACTICAL_METHOD_TIE_POINTS
                if expected_method == "reichenberg"
                else delta >= -PRACTICAL_METHOD_TIE_POINTS
                for delta in deltas
            ),
            "mean_reichenberg_minus_yoon_mape_points": mean(deltas),
            "members": [
                {
                    "name": result.name,
                    "cas": result.cas,
                    "reichenberg_minus_yoon_mape_points": delta,
                }
                for result, delta in zip(family, deltas)
            ],
        }

    return {
        "rule": (
            "Reichenberg for sparse-heteroatom compounds except unbranched "
            "C3+ aldehydes; also Reichenberg for unbranched terminal "
            "1-alkynes with at least four carbons; Yoon-Thodos otherwise"
        ),
        "selection": method_selection_summary(
            results,
            [structure_only_method_choice(result) for result in results],
        ),
        "terminal_1_alkynes": family_summary(terminal_alkynes, "reichenberg"),
        "unbranched_c3_plus_aldehydes": family_summary(
            linear_aldehydes,
            "yoon_thodos",
        ),
    }


def _reference_curve(row: dict, points: int) -> tuple[list[float], list[float]]:
    correlation = row["vapor_viscosity"][0]
    minimum = float(correlation["T_min_K"])
    maximum = float(correlation["T_max_K"])
    temperatures = [
        minimum + (maximum - minimum) * index / (points - 1)
        for index in range(points)
    ]
    reference = [
        PerryPropertyLibrary._eval_vapor_viscosity_Pa_s(correlation, temperature)
        for temperature in temperatures
    ]
    if any(value is None or value <= 0.0 for value in reference):
        raise ValueError("Perry reference curve contains an invalid value")
    return temperatures, [float(value) for value in reference]


def _dipole_effect(
    zero_values: Sequence[float],
    resolved_values: Sequence[float],
) -> tuple[float, float]:
    effects = [
        100.0 * (resolved / zero - 1.0)
        for zero, resolved in zip(zero_values, resolved_values)
    ]
    return (
        sum(abs(effect) for effect in effects) / len(effects),
        max(abs(effect) for effect in effects),
    )


def benchmark(args: argparse.Namespace) -> tuple[dict, str]:
    base_payload, _ = zero_dipole_benchmark(args)
    base_results = base_payload["compounds"]
    if args.limit is not None:
        base_results = base_results[: args.limit]

    database = json.loads(args.database.read_text(encoding="utf-8"))["chemicals"]
    resolver = NISTThenXTBResolver()
    results: list[CompoundResult] = []
    failures: list[dict] = []
    failure_reasons: Counter = Counter()

    total = len(base_results)
    for index, base in enumerate(base_results, start=1):
        cas = str(base["cas"])
        name = str(base["name"])
        row = database[cas]
        critical = row["critical_constants"]
        molecular_weight = float(critical["molecular_weight"])
        Tc_K = float(critical["Tc_K"])
        Pc_bar = 10.0 * float(critical["Pc_MPa"])
        smiles = str(base["smiles"])
        props = {
            "CAS": cas,
            "name": name,
            "formula": base["formula"],
            "MW": molecular_weight,
            "smiles": smiles,
        }

        try:
            dipole = resolver.resolve_dipole_moment(
                cas,
                props,
                use_pvdz=False,
                allow_online=False,
            )
            source = ALLOWED_DIPOLE_METHODS.get(str(dipole.method))
            if source is None:
                raise ValueError(
                    f"unsupported dipole result {dipole.source}/{dipole.method}: "
                    f"{dipole.notes}"
                )
            dipole_D = float(dipole.value)
            if not math.isfinite(dipole_D) or dipole_D < 0.0:
                raise ValueError(f"invalid resolved dipole {dipole_D!r}")

            temperatures, reference = _reference_curve(row, args.points)
            fragmentation = rm.fragment(smiles)
            zero_values = [
                rm.viscosity_Pa_s(
                    temperature,
                    molecular_weight,
                    Tc_K,
                    Pc_bar,
                    dipole_D=0.0,
                    fragmentation=fragmentation,
                )
                for temperature in temperatures
            ]
            resolved_values = [
                rm.viscosity_Pa_s(
                    temperature,
                    molecular_weight,
                    Tc_K,
                    Pc_bar,
                    dipole_D=dipole_D,
                    fragmentation=fragmentation,
                )
                for temperature in temperatures
            ]
            zero_metrics = error_metrics(reference, zero_values)
            resolved_metrics = error_metrics(reference, resolved_values)
            mean_effect, max_effect = _dipole_effect(zero_values, resolved_values)
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError("benchmark SMILES no longer parses")
            heavy_atoms = int(molecule.GetNumHeavyAtoms())
            result = CompoundResult(
                cas=cas,
                name=name,
                formula=str(base["formula"]),
                chemical_class=str(base["chemical_class"]),
                smiles=smiles,
                molecular_weight=molecular_weight,
                heavy_atoms=heavy_atoms,
                size_bin=molecular_size_bin(heavy_atoms),
                groups={str(key): int(value) for key, value in base["groups"].items()},
                contribution_sum=float(base["contribution_sum"]),
                Tmin_K=float(base["Tmin_K"]),
                Tmax_K=float(base["Tmax_K"]),
                Tr_min=float(base["Tr_min"]),
                Tr_max=float(base["Tr_max"]),
                dipole_D=dipole_D,
                reduced_dipole=rm.reduced_dipole(dipole_D, Pc_bar, Tc_K),
                dipole_source=source,
                dipole_method=str(dipole.method),
                dipole_quality=float(dipole.quality),
                yoon_thodos_mape_percent=float(base["yoon_mape_percent"]),
                zero_dipole_mape_percent=zero_metrics["mape"],
                zero_dipole_bias_percent=zero_metrics["bias"],
                zero_dipole_p95_ape_percent=zero_metrics["p95"],
                zero_dipole_max_ape_percent=zero_metrics["maximum"],
                resolved_dipole_mape_percent=resolved_metrics["mape"],
                resolved_dipole_bias_percent=resolved_metrics["bias"],
                resolved_dipole_p95_ape_percent=resolved_metrics["p95"],
                resolved_dipole_max_ape_percent=resolved_metrics["maximum"],
                resolved_minus_zero_mape_points=(
                    resolved_metrics["mape"] - zero_metrics["mape"]
                ),
                dipole_term_mean_effect_percent=mean_effect,
                dipole_term_max_effect_percent=max_effect,
            )
            results.append(result)
            print(
                f"[{index:3d}/{total}] {cas} {name}: {dipole_D:.4f} D "
                f"({source}), MAPE delta {result.resolved_minus_zero_mape_points:+.3f}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failure_reasons[reason] += 1
            failures.append({
                "cas": cas,
                "name": name,
                "formula": str(base["formula"]),
                "smiles": smiles,
                "reason": reason,
            })
            print(
                f"[{index:3d}/{total}] {cas} {name}: FAILED {reason}",
                file=sys.stderr,
                flush=True,
            )

    def summaries_by(field: str) -> dict:
        labels = sorted({str(getattr(result, field)) for result in results})
        return {
            label: paired_summary(
                [result for result in results if str(getattr(result, field)) == label]
            )
            for label in labels
        }

    payload = {
        "schema_version": 1,
        "benchmark": "resolved_dipole_reichenberg_on_zero_dipole_perry_population",
        "database": str(args.database.resolve()),
        "points_per_curve": args.points,
        "limit": args.limit,
        "dipole_policy": "CCCBDB, then GFN2-xTB; no PBE0 or heuristic dipoles",
        "base_population": {
            "eligible_total": base_payload["coverage"]["eligible_total"],
            "fragmented_total": base_payload["coverage"]["covered_total"],
            "selected_total": total,
            "resolved_total": len(results),
            "failed_total": len(failures),
        },
        "dipole_source_counts": dict(Counter(result.dipole_source for result in results)),
        "overall": paired_summary(results),
        "by_dipole_source": summaries_by("dipole_source"),
        "by_chemical_class": summaries_by("chemical_class"),
        "by_size": summaries_by("size_bin"),
        "method_classifier_analysis": method_classifier_analysis(results),
        "structure_rule_analysis": structure_rule_analysis(results),
        "failure_reasons": dict(failure_reasons),
        "compounds": [asdict(result) for result in results],
        "failures": failures,
    }
    return payload, build_report(payload, results, args.top)


def _comparison_lines(label: str, summary: dict) -> list[str]:
    return [
        f"  {label}",
        f"    zero     {format_distribution(summary['zero_dipole_curve_mape_distribution'])}",
        f"    resolved {format_distribution(summary['resolved_dipole_curve_mape_distribution'])}",
        (
            f"    head-to-head: resolved wins {summary['resolved_dipole_wins']}, "
            f"zero wins {summary['zero_dipole_wins']}, ties {summary['ties']}, "
            f"tie threshold {PRACTICAL_MAPE_TIE_POINTS:g} point; "
            f"within 0.1 point {summary['within_0_1_mape_point']}"
        ),
    ]


def _method_selector_line(label: str, summary: dict) -> str:
    accuracy = summary["decisive_accuracy"]
    accuracy_text = "n/a" if accuracy is None else f"{100.0 * accuracy:.1f}%"
    return (
        f"  {label:24s} mean MAPE={summary['mean_selected_mape_percent']:7.3f}% "
        f"regret={summary['mean_oracle_regret_points']:7.3f} points "
        f"Reichenberg={summary['selected_reichenberg']:3d} "
        f"decisive accuracy={accuracy_text}"
    )


def _method_rule_text(feature: dict) -> str:
    rule = feature["fitted_rule"]
    if rule == "always_reichenberg":
        return "always Reichenberg"
    if rule == "always_yoon_thodos":
        return "always Yoon-Thodos"
    return f"Reichenberg at or above {feature['fitted_threshold']:.6g}"


def _optional_number(value: float | None) -> str:
    return "none" if value is None else f"{value:.6g}"


def build_report(
    payload: dict,
    results: Sequence[CompoundResult],
    top: int,
) -> str:
    population = payload["base_population"]
    lines = [
        "Resolved-dipole vs zero-dipole Reichenberg on Perry organics",
        f"sampling: {payload['points_per_curve']} points over each complete Perry range",
        f"dipoles: {payload['dipole_policy']}",
        (
            f"population: {population['resolved_total']}/{population['selected_total']} "
            f"resolved from {population['fragmented_total']} fragmented compounds "
            f"({population['eligible_total']} originally eligible)"
        ),
        "sources: " + ", ".join(
            f"{name}={count}"
            for name, count in sorted(payload["dipole_source_counts"].items())
        ),
        "",
        "OVERALL",
        *_comparison_lines("all", payload["overall"]),
    ]
    for title, key in (
        ("BY DIPOLE SOURCE", "by_dipole_source"),
        ("BY CHEMICAL CLASS", "by_chemical_class"),
        ("BY MOLECULAR SIZE", "by_size"),
    ):
        lines.extend(("", title))
        for label, summary in payload[key].items():
            lines.extend(_comparison_lines(label, summary))

    classifiers = payload["method_classifier_analysis"]
    lines.extend(("", "YOON-THODOS VS ZERO-DIPOLE REICHENBERG SELECTORS"))
    for label, key in (
        ("current structural", "current_structural"),
        ("always Yoon-Thodos", "all_yoon_thodos"),
        ("always Reichenberg", "all_reichenberg"),
        ("oracle", "oracle"),
    ):
        lines.append(_method_selector_line(label, classifiers[key]))
    for feature_name, feature in classifiers["features"].items():
        lines.append(
            _method_selector_line(f"fitted {feature_name}", feature["fitted"])
            + f"; {_method_rule_text(feature)}"
        )
        thresholds = feature["leave_one_out"]["threshold_distribution"]
        lines.append(
            _method_selector_line(
                f"LOO {feature_name}",
                feature["leave_one_out"],
            )
            + (
                f"; threshold median={_optional_number(thresholds['median'])}, "
                f"range={_optional_number(thresholds['minimum'])}-"
                f"{_optional_number(thresholds['maximum'])}"
            )
        )
    lines.append("  source check at fitted raw-dipole cutoff")
    for source, summaries in classifiers[
        "by_source_at_fitted_dipole_cutoff"
    ].items():
        current = summaries["current_structural"]
        dipole = summaries["dipole_cutoff"]
        lines.append(
            f"    {source:9s} current={current['mean_selected_mape_percent']:.3f}% "
            f"dipole={dipole['mean_selected_mape_percent']:.3f}% "
            f"accuracy={100.0 * dipole['decisive_accuracy']:.1f}%"
        )

    structure = payload["structure_rule_analysis"]
    lines.extend(("", "STRUCTURE-ONLY CANDIDATE"))
    lines.append(f"  {structure['rule']}")
    lines.append(_method_selector_line("candidate", structure["selection"]))
    for label, key in (
        ("terminal 1-alkynes", "terminal_1_alkynes"),
        ("unbranched C3+ aldehydes", "unbranched_c3_plus_aldehydes"),
    ):
        family = structure[key]
        lines.append(
            f"  {label}: {family['support_or_practical_tie']}/"
            f"{family['compounds']} support or practically tie for "
            f"{family['expected_method']}; mean R-Y delta="
            f"{family['mean_reichenberg_minus_yoon_mape_points']:+.3f} points"
        )

    lines.extend(("", f"BIGGEST {top} IMPROVEMENTS FROM RESOLVED DIPOLES"))
    for result in sorted(results, key=lambda item: item.resolved_minus_zero_mape_points)[:top]:
        lines.append(
            f"  {result.name[:29]:29s} {result.heavy_atoms:2d} heavy "
            f"{result.dipole_D:6.3f} D {result.dipole_source:9s} "
            f"zero={result.zero_dipole_mape_percent:6.2f}% "
            f"resolved={result.resolved_dipole_mape_percent:6.2f}% "
            f"delta={result.resolved_minus_zero_mape_points:+7.2f}"
        )
    lines.extend(("", f"BIGGEST {top} REGRESSIONS FROM RESOLVED DIPOLES"))
    for result in sorted(
        results,
        key=lambda item: item.resolved_minus_zero_mape_points,
        reverse=True,
    )[:top]:
        lines.append(
            f"  {result.name[:29]:29s} {result.heavy_atoms:2d} heavy "
            f"{result.dipole_D:6.3f} D {result.dipole_source:9s} "
            f"zero={result.zero_dipole_mape_percent:6.2f}% "
            f"resolved={result.resolved_dipole_mape_percent:6.2f}% "
            f"delta={result.resolved_minus_zero_mape_points:+7.2f}"
        )
    if payload["failures"]:
        lines.extend(("", "FAILURES"))
        for failure in payload["failures"]:
            lines.append(
                f"  {failure['cas']} {failure['name']}: {failure['reason']}"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--limit",
        type=int,
        help="deterministically benchmark only the first N fragmented compounds",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points must be at least 2")
    if args.top < 0:
        parser.error("--top cannot be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    payload, report = benchmark(args)
    print(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
