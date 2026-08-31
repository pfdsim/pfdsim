#!/usr/bin/env python3
"""Probe Reichenberg dipole sensitivity and accuracy against Perry vapor viscosity.

The selected gas-phase dipole moments are experimental values from the NIST
CCCBDB table at https://cccbdb.nist.gov/diplistx.asp.  Perry 9th supplies the
reference vapor-viscosity curves and the MW/Tc/Pc inputs.  Every curve is
sampled uniformly over its complete stated temperature range.

The report is written to stdout.  ``--output-json`` optionally writes all
inputs, assignments, sensitivity results, and curve-level metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perry_properties import PerryPropertyLibrary  # noqa: E402
from property_resolver import (  # noqa: E402
    PropertyResolutionResult,
    PropertyResolver,
)


DEFAULT_DATABASE = ROOT / "data" / "perry_properties.json"
DIPOLE_SOURCE = "https://cccbdb.nist.gov/diplistx.asp"

# Perry 9th Table 2-173. Only groups used by this probe are repeated here.
REICHENBERG_GROUPS = {
    "CH3": 9.04,
    "CH2": 6.47,
    "OH_alcohol": 7.96,
    "C_carbonyl": 12.02,
    "CHO": 14.02,
    "NH2": 9.71,
    "CN": 18.13,
}


@dataclass(frozen=True)
class CompoundSpec:
    name: str
    cas: str
    dipole_D: float
    branch: str
    groups: tuple[tuple[str, int], ...] = ()
    assignment_note: str = ""


COMPOUNDS = (
    CompoundSpec("Water", "7732-18-5", 1.857, "inorganic"),
    CompoundSpec("Ammonia", "7664-41-7", 1.476, "inorganic"),
    CompoundSpec(
        "Methanol",
        "67-56-1",
        1.672,
        "organic",
        (("CH3", 1), ("OH_alcohol", 1)),
    ),
    CompoundSpec(
        "Ethanol",
        "64-17-5",
        1.679,
        "organic",
        (("CH3", 1), ("CH2", 1), ("OH_alcohol", 1)),
    ),
    CompoundSpec(
        "Acetone",
        "67-64-1",
        2.880,
        "organic",
        (("CH3", 2), ("C_carbonyl", 1)),
    ),
    CompoundSpec(
        "Acetonitrile",
        "75-05-8",
        3.919,
        "organic",
        (("CH3", 1), ("CN", 1)),
    ),
    CompoundSpec("Hydrogen fluoride", "7664-39-3", 1.827, "inorganic"),
    CompoundSpec(
        "Hydrogen cyanide",
        "74-90-8",
        2.980,
        "inorganic",
        assignment_note=(
            "Treated as inorganic because the organic -CN group requires an "
            "attached organic skeleton; the CN-only alternative is also reported."
        ),
    ),
    CompoundSpec(
        "Formaldehyde",
        "50-00-0",
        2.332,
        "organic",
        (("CHO", 1),),
    ),
    CompoundSpec(
        "Formamide",
        "75-12-7",
        3.730,
        "organic",
        (("C_carbonyl", 1), ("NH2", 1)),
    ),
    CompoundSpec("Sulfur dioxide", "7446-09-5", 1.633, "inorganic"),
)


@dataclass(frozen=True)
class ProbeResult:
    name: str
    cas: str
    formula: str
    branch: str
    groups: str
    group_sum: Optional[float]
    dipole_D: float
    reduced_dipole: float
    dipole_k: float
    Tmin_K: float
    Tmax_K: float
    Tr_min: float
    Tr_max: float
    yoon_thodos_mape_percent: float
    reichenberg_zero_mape_percent: float
    reichenberg_actual_mape_percent: float
    reichenberg_zero_bias_percent: float
    reichenberg_actual_bias_percent: float
    reichenberg_actual_max_ape_percent: float
    zero_dipole_mean_effect_percent: float
    zero_dipole_max_effect_percent: float
    dipole_5_percent_max_effect_percent: float
    dipole_10_percent_max_effect_percent: float
    dipole_20_percent_max_effect_percent: float
    maximum_local_dipole_elasticity: float
    alternate_organic_cn_mape_percent: Optional[float] = None


class PerryInputResolver(PropertyResolver):
    """Run production Yoon-Thodos with exact Perry critical inputs."""

    def __init__(self) -> None:
        super().__init__()
        self._benchmark_critical_results: dict[str, PropertyResolutionResult] = {}

    def set_critical_inputs(self, Tc_K: float, Pc_bar: float) -> None:
        self._benchmark_critical_results = {
            "Tc": PropertyResolutionResult(
                Tc_K, "local", "perry_critical_constant", 1.0, "units K"
            ),
            "Pc": PropertyResolutionResult(
                Pc_bar, "local", "perry_critical_constant", 1.0, "units bar"
            ),
        }

    def resolve_critical_properties(self, *args, **kwargs):
        return self._benchmark_critical_results


def group_sum(spec: CompoundSpec) -> Optional[float]:
    if spec.branch != "organic":
        return None
    return sum(REICHENBERG_GROUPS[name] * count for name, count in spec.groups)


def format_groups(spec: CompoundSpec) -> str:
    if not spec.groups:
        return "inorganic prefactor"
    return " + ".join(
        f"{count} {name}" if count != 1 else name
        for name, count in spec.groups
    )


def reichenberg_reduced_dipole(dipole_D: float, Pc_bar: float, Tc_K: float) -> float:
    return 52.46 * dipole_D**2 * Pc_bar / Tc_K**2


def reichenberg_dipole_factor(
    temperature_K: float,
    dipole_D: float,
    Pc_bar: float,
    Tc_K: float,
) -> float:
    Tr = temperature_K / Tc_K
    reduced_dipole = reichenberg_reduced_dipole(dipole_D, Pc_bar, Tc_K)
    k = 270.0 * reduced_dipole**4
    return (1.0 + k) / (Tr + k)


def reichenberg_A(
    *,
    molecular_weight: float,
    Tc_K: float,
    Pc_bar: float,
    branch: str,
    contribution_sum: Optional[float],
) -> float:
    if branch == "organic":
        if contribution_sum is None or contribution_sum <= 0.0:
            raise ValueError("Organic Reichenberg calculation requires a positive group sum")
        return 1.0e-7 * math.sqrt(molecular_weight) * Tc_K / contribution_sum
    if branch != "inorganic":
        raise ValueError(f"Unknown Reichenberg branch {branch!r}")
    Pc_Pa = Pc_bar * 1.0e5
    return (
        1.6104e-10
        * math.sqrt(molecular_weight)
        * Pc_Pa ** (2.0 / 3.0)
        * Tc_K ** (-1.0 / 6.0)
    )


def reichenberg_viscosity_Pa_s(
    temperature_K: float,
    *,
    dipole_D: float,
    molecular_weight: float,
    Tc_K: float,
    Pc_bar: float,
    branch: str,
    contribution_sum: Optional[float],
) -> float:
    Tr = temperature_K / Tc_K
    A = reichenberg_A(
        molecular_weight=molecular_weight,
        Tc_K=Tc_K,
        Pc_bar=Pc_bar,
        branch=branch,
        contribution_sum=contribution_sum,
    )
    temperature_factor = Tr**2 / (1.0 + 0.36 * Tr * (Tr - 1.0)) ** (1.0 / 6.0)
    return A * temperature_factor * reichenberg_dipole_factor(
        temperature_K,
        dipole_D,
        Pc_bar,
        Tc_K,
    )


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def curve_metrics(reference: Sequence[float], predicted: Sequence[float]) -> tuple[float, float, float]:
    signed = [100.0 * (estimate / actual - 1.0) for actual, estimate in zip(reference, predicted)]
    return mean([abs(value) for value in signed]), mean(signed), max(abs(value) for value in signed)


def maximum_relative_effect(
    temperatures: Sequence[float],
    *,
    assumed_dipoles: Sequence[float],
    actual_dipole: float,
    Pc_bar: float,
    Tc_K: float,
) -> float:
    return max(
        abs(
            100.0
            * (
                reichenberg_dipole_factor(T, assumed, Pc_bar, Tc_K)
                / reichenberg_dipole_factor(T, actual_dipole, Pc_bar, Tc_K)
                - 1.0
            )
        )
        for assumed in assumed_dipoles
        for T in temperatures
    )


def mean_relative_effect(
    temperatures: Sequence[float],
    *,
    assumed_dipole: float,
    actual_dipole: float,
    Pc_bar: float,
    Tc_K: float,
) -> float:
    return mean(
        [
            abs(
                100.0
                * (
                    reichenberg_dipole_factor(T, assumed_dipole, Pc_bar, Tc_K)
                    / reichenberg_dipole_factor(T, actual_dipole, Pc_bar, Tc_K)
                    - 1.0
                )
            )
            for T in temperatures
        ]
    )


def validate_perry_example() -> float:
    """Reproduce Perry's ethyl-acetate example on printed page 2-352."""
    value = reichenberg_viscosity_Pa_s(
        401.25,
        dipole_D=1.78,
        molecular_weight=88.1051,
        Tc_K=523.3,
        Pc_bar=38.8,
        branch="organic",
        contribution_sum=37.96,
    )
    if not math.isclose(value, 1.003e-5, rel_tol=5.0e-4):
        raise AssertionError(f"Perry ethyl-acetate validation failed: {value:g} Pa*s")
    return value


def evaluate(args: argparse.Namespace) -> tuple[list[ProbeResult], float]:
    chemicals = json.loads(args.database.read_text(encoding="utf-8"))["chemicals"]
    resolver = PerryInputResolver()
    results = []
    validation = validate_perry_example()

    for spec in COMPOUNDS:
        row = chemicals[spec.cas]
        critical = row["critical_constants"]
        correlation = row["vapor_viscosity"][0]
        Tc_K = float(critical["Tc_K"])
        Pc_bar = 10.0 * float(critical["Pc_MPa"])
        molecular_weight = float(critical["molecular_weight"])
        Tmin_K = float(correlation["T_min_K"])
        Tmax_K = float(correlation["T_max_K"])
        temperatures = [
            Tmin_K + (Tmax_K - Tmin_K) * index / (args.points - 1)
            for index in range(args.points)
        ]
        reference = [
            PerryPropertyLibrary._eval_vapor_viscosity_Pa_s(correlation, T)
            for T in temperatures
        ]
        if any(value is None or value <= 0.0 for value in reference):
            raise ValueError(f"Invalid Perry vapor-viscosity curve for {spec.name}")
        reference = [float(value) for value in reference]
        contribution_sum = group_sum(spec)

        zero = [
            reichenberg_viscosity_Pa_s(
                T,
                dipole_D=0.0,
                molecular_weight=molecular_weight,
                Tc_K=Tc_K,
                Pc_bar=Pc_bar,
                branch=spec.branch,
                contribution_sum=contribution_sum,
            )
            for T in temperatures
        ]
        actual = [
            reichenberg_viscosity_Pa_s(
                T,
                dipole_D=spec.dipole_D,
                molecular_weight=molecular_weight,
                Tc_K=Tc_K,
                Pc_bar=Pc_bar,
                branch=spec.branch,
                contribution_sum=contribution_sum,
            )
            for T in temperatures
        ]
        zero_mape, zero_bias, _ = curve_metrics(reference, zero)
        actual_mape, actual_bias, actual_max = curve_metrics(reference, actual)

        resolver.set_critical_inputs(Tc_K, Pc_bar)
        props = {
            "CAS": spec.cas,
            "name": spec.name,
            "formula": row.get("formula") or "",
            "MW": molecular_weight,
            "property_sources": {
                "MW": {
                    "source": "local",
                    "method": "perry_vapor_viscosity_table",
                    "quality": 1.0,
                }
            },
        }
        yoon = [
            resolver._yoon_thodos_viscosity(spec.cas, props, T, "vapor").value
            for T in temperatures
        ]
        yoon_mape, _, _ = curve_metrics(reference, yoon)

        reduced_dipole = reichenberg_reduced_dipole(spec.dipole_D, Pc_bar, Tc_K)
        k = 270.0 * reduced_dipole**4
        elasticities = [
            abs(
                8.0
                * k
                * (T / Tc_K - 1.0)
                / ((1.0 + k) * (T / Tc_K + k))
            )
            for T in temperatures
        ]
        alternate = None
        if spec.cas == "74-90-8":
            alternate_curve = [
                reichenberg_viscosity_Pa_s(
                    T,
                    dipole_D=spec.dipole_D,
                    molecular_weight=molecular_weight,
                    Tc_K=Tc_K,
                    Pc_bar=Pc_bar,
                    branch="organic",
                    contribution_sum=REICHENBERG_GROUPS["CN"],
                )
                for T in temperatures
            ]
            alternate, _, _ = curve_metrics(reference, alternate_curve)

        results.append(
            ProbeResult(
                name=spec.name,
                cas=spec.cas,
                formula=str(row.get("formula") or ""),
                branch=spec.branch,
                groups=format_groups(spec),
                group_sum=contribution_sum,
                dipole_D=spec.dipole_D,
                reduced_dipole=reduced_dipole,
                dipole_k=k,
                Tmin_K=Tmin_K,
                Tmax_K=Tmax_K,
                Tr_min=Tmin_K / Tc_K,
                Tr_max=Tmax_K / Tc_K,
                yoon_thodos_mape_percent=yoon_mape,
                reichenberg_zero_mape_percent=zero_mape,
                reichenberg_actual_mape_percent=actual_mape,
                reichenberg_zero_bias_percent=zero_bias,
                reichenberg_actual_bias_percent=actual_bias,
                reichenberg_actual_max_ape_percent=actual_max,
                zero_dipole_mean_effect_percent=mean_relative_effect(
                    temperatures,
                    assumed_dipole=0.0,
                    actual_dipole=spec.dipole_D,
                    Pc_bar=Pc_bar,
                    Tc_K=Tc_K,
                ),
                zero_dipole_max_effect_percent=maximum_relative_effect(
                    temperatures,
                    assumed_dipoles=(0.0,),
                    actual_dipole=spec.dipole_D,
                    Pc_bar=Pc_bar,
                    Tc_K=Tc_K,
                ),
                dipole_5_percent_max_effect_percent=maximum_relative_effect(
                    temperatures,
                    assumed_dipoles=(0.95 * spec.dipole_D, 1.05 * spec.dipole_D),
                    actual_dipole=spec.dipole_D,
                    Pc_bar=Pc_bar,
                    Tc_K=Tc_K,
                ),
                dipole_10_percent_max_effect_percent=maximum_relative_effect(
                    temperatures,
                    assumed_dipoles=(0.90 * spec.dipole_D, 1.10 * spec.dipole_D),
                    actual_dipole=spec.dipole_D,
                    Pc_bar=Pc_bar,
                    Tc_K=Tc_K,
                ),
                dipole_20_percent_max_effect_percent=maximum_relative_effect(
                    temperatures,
                    assumed_dipoles=(0.80 * spec.dipole_D, 1.20 * spec.dipole_D),
                    actual_dipole=spec.dipole_D,
                    Pc_bar=Pc_bar,
                    Tc_K=Tc_K,
                ),
                maximum_local_dipole_elasticity=max(elasticities),
                alternate_organic_cn_mape_percent=alternate,
            )
        )
    return results, validation


def build_report(results: Sequence[ProbeResult], validation: float, points: int) -> str:
    lines = [
        "Reichenberg dipole sensitivity and Perry vapor-viscosity probe",
        f"dipole source: {DIPOLE_SOURCE}",
        f"sampling: {points} uniform inclusive points over each complete Perry range",
        f"Perry ethyl-acetate example validation: {validation:.10g} Pa*s",
        "",
        "PERRY ACCURACY (curve MAPE)",
        "name|Yoon-Thodos|Reichenberg mu=0|Reichenberg actual mu|actual max APE",
    ]
    for result in results:
        lines.append(
            f"{result.name}|{result.yoon_thodos_mape_percent:.3f}%|"
            f"{result.reichenberg_zero_mape_percent:.3f}%|"
            f"{result.reichenberg_actual_mape_percent:.3f}%|"
            f"{result.reichenberg_actual_max_ape_percent:.3f}%"
        )
    lines.extend(
        (
            "",
            "DIPOLE-TERM SENSITIVITY OVER PERRY RANGE",
            "name|mu_D|mu*_r|zero mean|max zero|max +/-5%|max +/-10%|max +/-20%|max elasticity",
        )
    )
    for result in results:
        lines.append(
            f"{result.name}|{result.dipole_D:.3f}|{result.reduced_dipole:.5f}|"
            f"{result.zero_dipole_mean_effect_percent:.3f}%|"
            f"{result.zero_dipole_max_effect_percent:.3f}%|"
            f"{result.dipole_5_percent_max_effect_percent:.3f}%|"
            f"{result.dipole_10_percent_max_effect_percent:.3f}%|"
            f"{result.dipole_20_percent_max_effect_percent:.3f}%|"
            f"{result.maximum_local_dipole_elasticity:.3f}"
        )
    lines.extend(("", "STRUCTURAL ASSIGNMENTS"))
    for spec, result in zip(COMPOUNDS, results):
        suffix = f"; {spec.assignment_note}" if spec.assignment_note else ""
        contribution = (
            f"{result.group_sum:.2f}"
            if result.group_sum is not None
            else "-"
        )
        lines.append(
            f"{result.name}: {result.branch}; {result.groups}; "
            f"sum={contribution}{suffix}"
        )
    hcn = next(result for result in results if result.cas == "74-90-8")
    lines.append(
        f"Hydrogen cyanide organic CN-only alternative actual-dipole MAPE: "
        f"{hcn.alternate_organic_cn_mape_percent:.3f}%"
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    results, validation = evaluate(args)
    print(build_report(results, validation, args.points))
    if args.output_json is not None:
        payload = {
            "schema_version": 1,
            "dipole_source": DIPOLE_SOURCE,
            "database": str(args.database.resolve()),
            "points_per_curve": args.points,
            "perry_ethyl_acetate_validation_Pa_s": validation,
            "group_values": REICHENBERG_GROUPS,
            "results": [asdict(result) for result in results],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
