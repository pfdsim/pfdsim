#!/usr/bin/env python3
"""Validate Jossi polar/nonpolar branch selection against CoolProp viscosity.

For each CoolProp pure fluid with a transport model and a locally resolvable
neutral molecular structure, this benchmark samples a fixed supercritical
``(Tr, rho_r)`` grid.  CoolProp viscosity at nearly zero density supplies the
dilute-gas baseline and CoolProp viscosity at the sampled density supplies the
target, isolating the dense-gas increment that Jossi-Stiel-Thodos estimates.

The production structural association guard is retained: fluids for which
pfdsim deliberately applies neither Jossi branch are reported separately and
excluded from polar/nonpolar classifier fitting.  Remaining dipoles are taken
from CCCBDB first and GFN2-xTB only on misses.  The report compares the current
structural branch, raw-dipole and reduced-dipole thresholds, both constant
branches, and an oracle.  Leave-one-fluid-out results expose threshold
overfitting without any randomized split.

GFN2-xTB results use pfdsim's persistent dipole cache.  Progress is written to
stderr, while the final report is written to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import CoolProp.CoolProp as CP
from chemicals.identifiers import search_chemical
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from property_resolver import PropertyResolver  # noqa: E402
from scripts.vapor_viscosity.benchmark_reichenberg_resolved_dipole import (  # noqa: E402
    ALLOWED_DIPOLE_METHODS,
    NISTThenXTBResolver,
)
from scripts.vapor_viscosity.benchmark_reichenberg_zero_dipole import (  # noqa: E402
    distribution,
    signed_distribution,
)


REDUCED_TEMPERATURES = (1.05, 1.20, 1.50, 2.00)
REDUCED_DENSITIES = (0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.60)
DILUTE_REDUCED_DENSITY = 1.0e-8
MINIMUM_VALID_STATE_FRACTION = 0.75
PRACTICAL_BRANCH_TIE_POINTS = 0.10
CAS_PATTERN = re.compile(r"\d{2,7}-\d{2}-\d")


@dataclass(frozen=True)
class StateResult:
    reduced_temperature: float
    reduced_density: float
    temperature_K: float
    molar_density_mol_m3: float
    coolprop_viscosity_Pa_s: float
    dilute_viscosity_Pa_s: float
    nonpolar_viscosity_Pa_s: float
    polar_viscosity_Pa_s: float
    nonpolar_ape_percent: float
    polar_ape_percent: float


@dataclass(frozen=True)
class FluidResult:
    fluid: str
    cas: str
    formula: str
    smiles: str
    molecular_weight_g_mol: float
    critical_temperature_K: float
    critical_pressure_Pa: float
    critical_molar_density_mol_m3: float
    current_branch: str
    current_branch_basis: str
    dipole_D: float
    reduced_dipole: float
    dipole_source: str
    dipole_method: str
    dipole_quality: float
    valid_states: int
    possible_states: int
    nonpolar_mape_percent: float
    polar_mape_percent: float
    polar_minus_nonpolar_mape_points: float
    states: tuple[StateResult, ...]


def mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else math.nan


def absolute_percentage_error(actual: float, predicted: float) -> float:
    return 100.0 * abs(float(predicted) / float(actual) - 1.0)


def reduced_dipole(dipole_D: float, Pc_Pa: float, Tc_K: float) -> float:
    return 52.46 * float(dipole_D) ** 2 * (float(Pc_Pa) / 1.0e5) / float(Tc_K) ** 2


def jossi_viscosity(
    resolver: PropertyResolver,
    branch: str,
    reduced_density: float,
    dilute_viscosity_Pa_s: float,
    molecular_weight_g_mol: float,
    critical_temperature_K: float,
    critical_pressure_Pa: float,
) -> float:
    increment = resolver._jossi_dimensionless_increment(branch, reduced_density)
    if increment is None or increment < 0.0 or not math.isfinite(increment):
        raise ValueError(f"Jossi {branch} increment is unavailable")
    Pc_MPa = critical_pressure_Pa / 1.0e6
    xi = (
        2173.4
        * critical_temperature_K ** (1.0 / 6.0)
        * molecular_weight_g_mol ** -0.5
        * Pc_MPa ** (-2.0 / 3.0)
    )
    value = dilute_viscosity_Pa_s + increment / xi / 1000.0
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"Jossi {branch} viscosity is invalid")
    return value


def chemical_metadata(cas: str) -> tuple[str, str]:
    metadata = search_chemical(cas)
    smiles = str(getattr(metadata, "smiles", "") or "").strip()
    formula = str(getattr(metadata, "formula", "") or "").strip()
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        raise ValueError("SMILES unavailable")
    if len(Chem.GetMolFrags(molecule)) != 1 or Chem.GetFormalCharge(molecule) != 0:
        raise ValueError("structure is disconnected or charged")
    return formula, smiles


def coolprop_constants(fluid: str) -> tuple[float, float, float, float]:
    values = (
        float(CP.PropsSI("molar_mass", fluid)) * 1000.0,
        float(CP.PropsSI("Tcrit", fluid)),
        float(CP.PropsSI("pcrit", fluid)),
        float(CP.PropsSI("rhomolar_critical", fluid)),
    )
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("CoolProp critical constants are invalid")
    return values


def coolprop_viscosity(fluid: str, temperature_K: float, density: float) -> float:
    value = float(
        CP.PropsSI("VISCOSITY", "T", temperature_K, "Dmolar", density, fluid)
    )
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError("CoolProp viscosity is invalid")
    return value


def state_results(
    fluid: str,
    resolver: PropertyResolver,
    molecular_weight: float,
    Tc_K: float,
    Pc_Pa: float,
    rhoc_molar: float,
) -> tuple[StateResult, ...]:
    results = []
    for Tr in REDUCED_TEMPERATURES:
        temperature = Tr * Tc_K
        try:
            dilute = coolprop_viscosity(
                fluid,
                temperature,
                max(DILUTE_REDUCED_DENSITY * rhoc_molar, 1.0e-12),
            )
        except (TypeError, ValueError, OverflowError):
            continue
        for rho_r in REDUCED_DENSITIES:
            density = rho_r * rhoc_molar
            try:
                target = coolprop_viscosity(fluid, temperature, density)
                nonpolar = jossi_viscosity(
                    resolver,
                    "nonpolar",
                    rho_r,
                    dilute,
                    molecular_weight,
                    Tc_K,
                    Pc_Pa,
                )
                polar = jossi_viscosity(
                    resolver,
                    "polar",
                    rho_r,
                    dilute,
                    molecular_weight,
                    Tc_K,
                    Pc_Pa,
                )
            except (TypeError, ValueError, OverflowError):
                continue
            results.append(
                StateResult(
                    reduced_temperature=Tr,
                    reduced_density=rho_r,
                    temperature_K=temperature,
                    molar_density_mol_m3=density,
                    coolprop_viscosity_Pa_s=target,
                    dilute_viscosity_Pa_s=dilute,
                    nonpolar_viscosity_Pa_s=nonpolar,
                    polar_viscosity_Pa_s=polar,
                    nonpolar_ape_percent=absolute_percentage_error(target, nonpolar),
                    polar_ape_percent=absolute_percentage_error(target, polar),
                )
            )
    return tuple(results)


def threshold_candidates(results: Sequence[FluidResult], field: str) -> list[float]:
    values = sorted({float(getattr(result, field)) for result in results})
    return [float("-inf"), float("inf"), *(
        (left + right) / 2.0 for left, right in zip(values, values[1:])
    )]


def selection_summary(
    results: Sequence[FluidResult],
    choose_polar: Callable[[FluidResult, int], bool],
) -> dict:
    selected = []
    oracle = []
    decisive = 0
    correct = 0
    polar_count = 0
    for index, result in enumerate(results):
        polar = bool(choose_polar(result, index))
        polar_count += polar
        selected.append(
            result.polar_mape_percent if polar else result.nonpolar_mape_percent
        )
        oracle.append(min(result.polar_mape_percent, result.nonpolar_mape_percent))
        delta = result.polar_minus_nonpolar_mape_points
        if abs(delta) > PRACTICAL_BRANCH_TIE_POINTS:
            decisive += 1
            correct += polar == (delta < 0.0)
    return {
        "fluids": len(results),
        "selected_polar": polar_count,
        "selected_nonpolar": len(results) - polar_count,
        "mean_selected_mape_percent": mean(selected),
        "mean_oracle_regret_points": mean(
            [value - best for value, best in zip(selected, oracle)]
        ),
        "decisive_fluids": decisive,
        "correct_decisive_fluids": correct,
        "decisive_accuracy": correct / decisive if decisive else None,
    }


def best_threshold(results: Sequence[FluidResult], field: str) -> tuple[float, dict]:
    choices = []
    for threshold in threshold_candidates(results, field):
        summary = selection_summary(
            results,
            lambda result, _index, t=threshold: float(getattr(result, field)) >= t,
        )
        choices.append((summary["mean_selected_mape_percent"], threshold, summary))
    _score, threshold, summary = min(choices, key=lambda item: (item[0], item[1]))
    return threshold, summary


def leave_one_out_threshold(results: Sequence[FluidResult], field: str) -> dict:
    thresholds = []
    predictions: dict[int, bool] = {}
    for held_out in range(len(results)):
        training = [result for index, result in enumerate(results) if index != held_out]
        threshold, _summary = best_threshold(training, field)
        thresholds.append(threshold)
        predictions[held_out] = float(getattr(results[held_out], field)) >= threshold
    summary = selection_summary(results, lambda _result, index: predictions[index])
    finite_thresholds = [value for value in thresholds if math.isfinite(value)]
    summary["threshold_distribution"] = (
        signed_distribution(finite_thresholds)
        if finite_thresholds
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
    summary["always_nonpolar_thresholds"] = sum(
        value == float("inf") for value in thresholds
    )
    summary["always_polar_thresholds"] = sum(
        value == float("-inf") for value in thresholds
    )
    return summary


def classifier_analysis(results: Sequence[FluidResult]) -> dict:
    current = selection_summary(
        results,
        lambda result, _index: result.current_branch == "polar",
    )
    all_nonpolar = selection_summary(results, lambda _result, _index: False)
    all_polar = selection_summary(results, lambda _result, _index: True)
    oracle = selection_summary(
        results,
        lambda result, _index: result.polar_mape_percent < result.nonpolar_mape_percent,
    )
    features = {}
    for field in ("dipole_D", "reduced_dipole"):
        threshold, fitted = best_threshold(results, field)
        features[field] = {
            "fitted_threshold": threshold if math.isfinite(threshold) else None,
            "fitted_rule": (
                "always_nonpolar"
                if threshold == float("inf")
                else "always_polar"
                if threshold == float("-inf")
                else "polar_at_or_above_threshold"
            ),
            "fitted": fitted,
            "leave_one_out": leave_one_out_threshold(results, field),
        }
    return {
        "practical_tie_points": PRACTICAL_BRANCH_TIE_POINTS,
        "current_structural": current,
        "all_nonpolar": all_nonpolar,
        "all_polar": all_polar,
        "oracle": oracle,
        "features": features,
    }


def density_analysis(results: Sequence[FluidResult]) -> dict:
    summaries = {}
    for rho_r in REDUCED_DENSITIES:
        rows = []
        for result in results:
            states = [
                state for state in result.states
                if state.reduced_density == rho_r
            ]
            if not states:
                continue
            rows.append({
                "dipole_D": result.dipole_D,
                "nonpolar_mape_percent": mean(
                    [state.nonpolar_ape_percent for state in states]
                ),
                "polar_mape_percent": mean(
                    [state.polar_ape_percent for state in states]
                ),
            })
        dipoles = sorted({row["dipole_D"] for row in rows})
        thresholds = [float("-inf"), float("inf"), *(
            (left + right) / 2.0
            for left, right in zip(dipoles, dipoles[1:])
        )]
        choices = []
        for threshold in thresholds:
            selected = [
                row["polar_mape_percent"]
                if row["dipole_D"] >= threshold
                else row["nonpolar_mape_percent"]
                for row in rows
            ]
            choices.append((mean(selected), threshold))
        best_mape, best_cutoff = min(choices, key=lambda item: (item[0], item[1]))
        deltas = [
            row["polar_mape_percent"] - row["nonpolar_mape_percent"]
            for row in rows
        ]
        summaries[f"{rho_r:g}"] = {
            "fluids": len(rows),
            "mean_nonpolar_mape_percent": mean(
                [row["nonpolar_mape_percent"] for row in rows]
            ),
            "mean_polar_mape_percent": mean(
                [row["polar_mape_percent"] for row in rows]
            ),
            "polar_wins": sum(
                delta < -PRACTICAL_BRANCH_TIE_POINTS for delta in deltas
            ),
            "nonpolar_wins": sum(
                delta > PRACTICAL_BRANCH_TIE_POINTS for delta in deltas
            ),
            "ties": sum(
                abs(delta) <= PRACTICAL_BRANCH_TIE_POINTS for delta in deltas
            ),
            "best_dipole_selected_mape_percent": best_mape,
            "best_dipole_threshold": (
                best_cutoff if math.isfinite(best_cutoff) else None
            ),
            "best_dipole_rule": (
                "always_nonpolar"
                if best_cutoff == float("inf")
                else "always_polar"
                if best_cutoff == float("-inf")
                else "polar_at_or_above_threshold"
            ),
        }
    return summaries


def benchmark(args: argparse.Namespace) -> tuple[dict, str]:
    branch_resolver = PropertyResolver()
    dipole_resolver = NISTThenXTBResolver()
    results: list[FluidResult] = []
    guarded: list[dict] = []
    rejected: list[dict] = []
    rejection_reasons: Counter = Counter()
    fluids = sorted(CP.FluidsList())
    if args.limit is not None:
        fluids = fluids[: args.limit]

    possible_states = len(REDUCED_TEMPERATURES) * len(REDUCED_DENSITIES)
    minimum_states = math.ceil(MINIMUM_VALID_STATE_FRACTION * possible_states)
    total = len(fluids)
    for index, fluid in enumerate(fluids, start=1):
        try:
            cas = str(CP.get_fluid_param_string(fluid, "CAS") or "").strip()
            if CAS_PATTERN.fullmatch(cas) is None:
                raise ValueError("CoolProp fluid has no ordinary CAS number")
            formula, smiles = chemical_metadata(cas)
            molecular_weight, Tc_K, Pc_Pa, rhoc_molar = coolprop_constants(fluid)
            states = state_results(
                fluid,
                branch_resolver,
                molecular_weight,
                Tc_K,
                Pc_Pa,
                rhoc_molar,
            )
            if len(states) < minimum_states:
                raise ValueError(
                    f"CoolProp transport model produced only {len(states)}/{possible_states} "
                    "valid states"
                )
            props = {
                "CAS": cas,
                "name": fluid,
                "formula": formula,
                "smiles": smiles,
                "MW": molecular_weight,
            }
            branch, _factor, branch_basis = branch_resolver._jossi_polarity(fluid, props)
            if branch is None:
                guarded.append({
                    "fluid": fluid,
                    "cas": cas,
                    "formula": formula,
                    "smiles": smiles,
                    "reason": branch_basis,
                    "valid_states": len(states),
                })
                print(
                    f"[{index:3d}/{total}] {fluid}: GUARDED {branch_basis}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            dipole = dipole_resolver.resolve_dipole_moment(
                cas,
                props,
                use_pvdz=False,
                allow_online=False,
            )
            dipole_source = ALLOWED_DIPOLE_METHODS.get(str(dipole.method))
            if dipole_source is None:
                raise ValueError(
                    f"unsupported dipole result {dipole.source}/{dipole.method}: "
                    f"{dipole.notes}"
                )
            dipole_D = float(dipole.value)
            if dipole_D < 0.0 or not math.isfinite(dipole_D):
                raise ValueError("resolved dipole is invalid")
            nonpolar_mape = mean([state.nonpolar_ape_percent for state in states])
            polar_mape = mean([state.polar_ape_percent for state in states])
            result = FluidResult(
                fluid=fluid,
                cas=cas,
                formula=formula,
                smiles=smiles,
                molecular_weight_g_mol=molecular_weight,
                critical_temperature_K=Tc_K,
                critical_pressure_Pa=Pc_Pa,
                critical_molar_density_mol_m3=rhoc_molar,
                current_branch=branch,
                current_branch_basis=branch_basis,
                dipole_D=dipole_D,
                reduced_dipole=reduced_dipole(dipole_D, Pc_Pa, Tc_K),
                dipole_source=dipole_source,
                dipole_method=str(dipole.method),
                dipole_quality=float(dipole.quality),
                valid_states=len(states),
                possible_states=possible_states,
                nonpolar_mape_percent=nonpolar_mape,
                polar_mape_percent=polar_mape,
                polar_minus_nonpolar_mape_points=polar_mape - nonpolar_mape,
                states=states,
            )
            results.append(result)
            print(
                f"[{index:3d}/{total}] {fluid}: {dipole_D:.4f} D "
                f"({dipole_source}), current={branch}, "
                f"polar-nonpolar={result.polar_minus_nonpolar_mape_points:+.3f} points",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            rejection_reasons[reason] += 1
            rejected.append({"fluid": fluid, "reason": reason})
            print(
                f"[{index:3d}/{total}] {fluid}: REJECTED {reason}",
                file=sys.stderr,
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "benchmark": "jossi_polarity_classifier_against_coolprop_dense_viscosity",
        "coolprop_version": str(CP.get_global_param_string("version")),
        "reduced_temperatures": list(REDUCED_TEMPERATURES),
        "reduced_densities": list(REDUCED_DENSITIES),
        "dilute_reduced_density": DILUTE_REDUCED_DENSITY,
        "minimum_valid_state_fraction": MINIMUM_VALID_STATE_FRACTION,
        "limit": args.limit,
        "dipole_policy": "CCCBDB, then GFN2-xTB; no PBE0 or heuristic dipoles",
        "coverage": {
            "coolprop_fluids": len(fluids),
            "branchable_fluids": len(results),
            "association_guarded_fluids": len(guarded),
            "rejected_fluids": len(rejected),
            "dipole_source_counts": dict(
                Counter(result.dipole_source for result in results)
            ),
            "current_branch_counts": dict(
                Counter(result.current_branch for result in results)
            ),
            "rejection_reasons": dict(rejection_reasons),
        },
        "classifier_analysis": classifier_analysis(results),
        "by_reduced_density": density_analysis(results),
        "branch_error_distributions": {
            "nonpolar_mape_percent": distribution(
                [result.nonpolar_mape_percent for result in results]
            ),
            "polar_mape_percent": distribution(
                [result.polar_mape_percent for result in results]
            ),
            "polar_minus_nonpolar_mape_points": signed_distribution(
                [result.polar_minus_nonpolar_mape_points for result in results]
            ),
        },
        "fluids": [asdict(result) for result in results],
        "association_guarded": guarded,
        "rejected": rejected,
    }
    return payload, build_report(payload, results, args.top)


def classifier_line(label: str, summary: dict) -> str:
    accuracy = summary["decisive_accuracy"]
    accuracy_text = "n/a" if accuracy is None else f"{100.0 * accuracy:.1f}%"
    return (
        f"  {label:24s} mean MAPE={summary['mean_selected_mape_percent']:7.3f}% "
        f"regret={summary['mean_oracle_regret_points']:7.3f} points "
        f"polar={summary['selected_polar']:3d} "
        f"decisive accuracy={accuracy_text}"
    )


def threshold_text(value: float | None) -> str:
    return "none" if value is None else f"{value:.6g}"


def fitted_rule_text(feature: dict) -> str:
    rule = feature["fitted_rule"]
    if rule == "always_nonpolar":
        return "always nonpolar"
    if rule == "always_polar":
        return "always polar"
    return f"polar at dipole >= {threshold_text(feature['fitted_threshold'])}"


def build_report(
    payload: dict,
    results: Sequence[FluidResult],
    top: int,
) -> str:
    coverage = payload["coverage"]
    analysis = payload["classifier_analysis"]
    lines = [
        "Jossi polar/nonpolar classifier validation against CoolProp",
        f"CoolProp version: {payload['coolprop_version']}",
        "Tr grid: " + ", ".join(f"{value:g}" for value in REDUCED_TEMPERATURES),
        "rho_r grid: " + ", ".join(f"{value:g}" for value in REDUCED_DENSITIES),
        (
            f"coverage: branchable={coverage['branchable_fluids']}, "
            f"association-guarded={coverage['association_guarded_fluids']}, "
            f"rejected={coverage['rejected_fluids']} of "
            f"{coverage['coolprop_fluids']} CoolProp fluids"
        ),
        "dipoles: " + ", ".join(
            f"{name}={count}"
            for name, count in sorted(coverage["dipole_source_counts"].items())
        ),
        "current branches: " + ", ".join(
            f"{name}={count}"
            for name, count in sorted(coverage["current_branch_counts"].items())
        ),
        "",
        "BRANCH SELECTORS (per-fluid mean over valid states)",
        classifier_line("current structural", analysis["current_structural"]),
        classifier_line("always nonpolar", analysis["all_nonpolar"]),
        classifier_line("always polar", analysis["all_polar"]),
        classifier_line("oracle", analysis["oracle"]),
    ]
    for field, feature in analysis["features"].items():
        lines.append(
            classifier_line(f"fitted {field}", feature["fitted"])
            + f"; {fitted_rule_text(feature)}"
        )
        threshold_distribution = feature["leave_one_out"]["threshold_distribution"]
        lines.append(
            classifier_line(f"LOO {field}", feature["leave_one_out"])
            + (
                f" threshold median={threshold_text(threshold_distribution['median'])}, "
                f"range={threshold_text(threshold_distribution['minimum'])}-"
                f"{threshold_text(threshold_distribution['maximum'])}"
            )
        )

    lines.extend(("", "BY REDUCED DENSITY"))
    for rho_r, summary in payload["by_reduced_density"].items():
        rule = summary["best_dipole_rule"].replace("_", " ")
        if summary["best_dipole_threshold"] is not None:
            rule += f" at {summary['best_dipole_threshold']:.4g} D"
        lines.append(
            f"  rho_r={float(rho_r):4.2f} n={summary['fluids']:2d} "
            f"nonpolar={summary['mean_nonpolar_mape_percent']:7.3f}% "
            f"polar={summary['mean_polar_mape_percent']:7.3f}% "
            f"wins P/N/tie={summary['polar_wins']}/"
            f"{summary['nonpolar_wins']}/{summary['ties']}; best dipole rule: {rule}"
        )

    lines.extend(("", f"BIGGEST {top} POLAR-BRANCH IMPROVEMENTS"))
    for result in sorted(results, key=lambda item: item.polar_minus_nonpolar_mape_points)[:top]:
        lines.append(
            f"  {result.fluid[:24]:24s} {result.dipole_D:6.3f} D "
            f"{result.dipole_source:9s} current={result.current_branch:8s} "
            f"nonpolar={result.nonpolar_mape_percent:7.2f}% "
            f"polar={result.polar_mape_percent:7.2f}% "
            f"delta={result.polar_minus_nonpolar_mape_points:+8.2f}"
        )
    lines.extend(("", f"BIGGEST {top} POLAR-BRANCH REGRESSIONS"))
    for result in sorted(
        results,
        key=lambda item: item.polar_minus_nonpolar_mape_points,
        reverse=True,
    )[:top]:
        lines.append(
            f"  {result.fluid[:24]:24s} {result.dipole_D:6.3f} D "
            f"{result.dipole_source:9s} current={result.current_branch:8s} "
            f"nonpolar={result.nonpolar_mape_percent:7.2f}% "
            f"polar={result.polar_mape_percent:7.2f}% "
            f"delta={result.polar_minus_nonpolar_mape_points:+8.2f}"
        )
    if payload["association_guarded"]:
        lines.extend(("", "ASSOCIATION-GUARDED FLUIDS"))
        for item in payload["association_guarded"]:
            lines.append(f"  {item['fluid']}: {item['reason']}")
    lines.extend(("", "REJECTION REASONS"))
    for reason, count in sorted(
        coverage["rejection_reasons"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines.append(f"  {count:3d} {reason}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--limit",
        type=int,
        help="deterministically inspect only the first N CoolProp fluids",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
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
