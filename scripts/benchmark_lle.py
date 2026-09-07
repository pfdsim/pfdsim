#!/usr/bin/env python3
"""Benchmark repeated liquid-liquid equilibrium solves.

The deterministic workload covers binary and multicomponent systems, several
activity-coefficient methods, miscible and phase-splitting feeds, broad
temperature ranges, dilute feeds, and near-critical conditions.  Model
construction and optional compiled-kernel preparation are reported separately
from the timed solves.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermodynamics import create_thermodynamics  # noqa: E402


@dataclass(frozen=True)
class State:
    temperature: float
    composition: dict[str, float]


@dataclass(frozen=True)
class Scenario:
    name: str
    method: str
    components: tuple[str, ...]
    states: tuple[State, ...]
    description: str


def binary_states(
    component_a: str,
    component_b: str,
    fractions_a: tuple[float, ...],
    temperatures: tuple[float, ...],
) -> tuple[State, ...]:
    return tuple(
        State(
            temperature=temperature,
            composition={component_a: fraction, component_b: 1.0 - fraction},
        )
        for temperature in temperatures
        for fraction in fractions_a
    )


def multicomponent_states(
    compositions: tuple[dict[str, float], ...],
    temperatures: tuple[float, ...],
) -> tuple[State, ...]:
    return tuple(
        State(temperature=temperature, composition=dict(composition))
        for temperature in temperatures
        for composition in compositions
    )


BUTANOL_WATER_FRACTIONS = (0.01, 0.05, 0.10, 0.30, 0.60)
HYDROCARBON_WATER_FRACTIONS = (0.001, 0.05, 0.25, 0.50, 0.95, 0.999)
GENERIC_BINARY_FRACTIONS = (0.01, 0.10, 0.30, 0.50, 0.90, 0.99)

NRTL_UNIQUAC_MULTICOMPONENT = (
    {"H2O": 0.45, "CH3OH": 0.01, "methyl acetate": 0.25, "(C2H5)2O": 0.29},
    {"H2O": 0.15, "CH3OH": 0.25, "methyl acetate": 0.30, "(C2H5)2O": 0.30},
    {"H2O": 0.70, "CH3OH": 0.05, "methyl acetate": 0.10, "(C2H5)2O": 0.15},
)
UNIFAC_MULTICOMPONENT = (
    {
        "diethyl ether": 0.121392237235,
        "n-hexane": 0.104651525863,
        "acrylic acid": 0.024974946990,
        "water": 0.748981289912,
    },
    {"diethyl ether": 0.30, "n-hexane": 0.10, "acrylic acid": 0.10, "water": 0.50},
    {"diethyl ether": 0.05, "n-hexane": 0.30, "acrylic acid": 0.15, "water": 0.50},
)
UNIFNIST_TERNARY = (
    {"water": 0.30, "methanol": 0.40, "benzene": 0.30},
    {"water": 0.15, "methanol": 0.10, "benzene": 0.75},
    {"water": 0.70, "methanol": 0.20, "benzene": 0.10},
)


SCENARIOS = (
    Scenario(
        "nrtl_butanol_water",
        "NRTL",
        ("1-butanol", "water"),
        binary_states(
            "1-butanol",
            "water",
            BUTANOL_WATER_FRACTIONS,
            (298.15, 313.15, 373.15, 397.50, 399.20),
        ),
        "binary miscibility gap from ambient conditions through the fitted UCST region",
    ),
    Scenario(
        "uniquac_butanol_water",
        "UNIQUAC",
        ("1-butanol", "water"),
        binary_states(
            "1-butanol",
            "water",
            BUTANOL_WATER_FRACTIONS,
            (298.15, 313.15, 373.15, 397.50, 399.00),
        ),
        "binary miscibility gap from ambient conditions through the fitted UCST region",
    ),
    Scenario(
        "nrtl_octane_water",
        "NRTL",
        ("n-octane", "water"),
        binary_states(
            "n-octane",
            "water",
            HYDROCARBON_WATER_FRACTIONS,
            (298.15, 413.15, 473.15, 523.15, 533.10),
        ),
        "strong binary split with dilute feeds and a wide temperature range",
    ),
    Scenario(
        "uniquac_octane_water",
        "UNIQUAC",
        ("n-octane", "water"),
        binary_states(
            "n-octane",
            "water",
            HYDROCARBON_WATER_FRACTIONS,
            (298.15, 413.15, 473.15, 523.15, 533.10),
        ),
        "strong binary split with dilute feeds and a wide temperature range",
    ),
    Scenario(
        "unifac_hexane_water",
        "UNIFAC",
        ("hexane", "water"),
        binary_states(
            "hexane",
            "water",
            HYDROCARBON_WATER_FRACTIONS,
            (288.15, 298.15, 323.15, 353.15),
        ),
        "compiled UNIFAC binary split including trace second phases",
    ),
    Scenario(
        "unifdmd_ethyl_acetate_water",
        "UNIFDMD",
        ("ethyl acetate", "water"),
        binary_states(
            "ethyl acetate",
            "water",
            GENERIC_BINARY_FRACTIONS,
            (288.15, 298.15, 323.15, 348.15),
        ),
        "modified-UNIFAC binary system spanning dilute and central feeds",
    ),
    Scenario(
        "unifnist_methanol_heptane",
        "UNIFNIST",
        ("methanol", "heptane"),
        binary_states(
            "methanol",
            "heptane",
            GENERIC_BINARY_FRACTIONS,
            (288.15, 298.15, 323.15, 348.15),
        ),
        "Dortmund UNIFAC binary system with polar/nonpolar phase behavior",
    ),
    Scenario(
        "nrtl_quaternary_extraction",
        "NRTL",
        ("H2O", "CH3OH", "methyl acetate", "(C2H5)2O"),
        multicomponent_states(
            NRTL_UNIQUAC_MULTICOMPONENT,
            (288.15, 298.15, 323.15, 348.15),
        ),
        "four-component extraction mixtures",
    ),
    Scenario(
        "uniquac_quaternary_extraction",
        "UNIQUAC",
        ("H2O", "CH3OH", "methyl acetate", "(C2H5)2O"),
        multicomponent_states(
            NRTL_UNIQUAC_MULTICOMPONENT,
            (288.15, 298.15, 323.15, 348.15),
        ),
        "four-component extraction mixtures",
    ),
    Scenario(
        "unifac_quaternary_extraction",
        "UNIFAC",
        ("diethyl ether", "n-hexane", "acrylic acid", "water"),
        multicomponent_states(
            UNIFAC_MULTICOMPONENT,
            (288.15, 298.15, 313.15, 333.15),
        ),
        "four-component extraction mixtures",
    ),
    Scenario(
        "unifdmd_quaternary_extraction",
        "UNIFDMD",
        ("diethyl ether", "n-hexane", "acrylic acid", "water"),
        multicomponent_states(
            UNIFAC_MULTICOMPONENT,
            (288.15, 298.15, 313.15, 333.15),
        ),
        "four-component extraction mixtures using modified UNIFAC",
    ),
    Scenario(
        "unifnist_ternary_extraction",
        "UNIFNIST",
        ("water", "methanol", "benzene"),
        multicomponent_states(
            UNIFNIST_TERNARY,
            (298.15, 315.15, 330.00, 345.15),
        ),
        "ternary cosolvent extraction mixtures",
    ),
)


def percentile_95(samples: list[float]) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def solve_pass(thermo, states: tuple[State, ...], max_iter: int, tol: float) -> tuple[int, float]:
    lle_count = 0
    checksum = 0.0
    for state in states:
        has_lle, phase1, phase2, beta = thermo.liquid_liquid_equilibrium(
            state.composition,
            state.temperature,
            max_iter=max_iter,
            tol=tol,
        )
        lle_count += int(has_lle)
        checksum += float(beta)
        if has_lle:
            first = next(iter(state.composition))
            checksum += phase1[first] + 0.5 * phase2[first]
    return lle_count, checksum


def run_scenario(
    scenario: Scenario,
    repeats: int,
    warmups: int,
    max_iter: int,
    tol: float,
) -> dict[str, object]:
    setup_started = time.perf_counter()
    with redirect_stdout(sys.stderr):
        thermo = create_thermodynamics(list(scenario.components), scenario.method)
        prepare = getattr(thermo, "prepare_compiled_backends", None)
        if callable(prepare):
            prepare(need_lle=True)
    setup_seconds = time.perf_counter() - setup_started

    warmup_lle_count = 0
    warmup_checksum = 0.0
    for _ in range(warmups):
        warmup_lle_count, warmup_checksum = solve_pass(
            thermo, scenario.states, max_iter, tol
        )

    samples = []
    lle_count = warmup_lle_count
    checksum = warmup_checksum
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            lle_count, checksum = solve_pass(thermo, scenario.states, max_iter, tol)
            samples.append(time.perf_counter() - started)
    finally:
        if was_enabled:
            gc.enable()

    state_count = len(scenario.states)
    solve_count = state_count * repeats
    total_seconds = sum(samples)
    per_solve_us = [sample * 1.0e6 / state_count for sample in samples]
    return {
        "name": scenario.name,
        "method": scenario.method,
        "description": scenario.description,
        "component_count": len(scenario.components),
        "state_count": state_count,
        "solve_count": solve_count,
        "lle_states_per_pass": lle_count,
        "single_phase_states_per_pass": state_count - lle_count,
        "setup_seconds": setup_seconds,
        "timed_seconds": total_seconds,
        "solves_per_second": solve_count / total_seconds,
        "median_us_per_solve": statistics.median(per_solve_us),
        "mean_us_per_solve": statistics.mean(per_solve_us),
        "p95_us_per_solve": percentile_95(per_solve_us),
        "min_us_per_solve": min(per_solve_us),
        "max_us_per_solve": max(per_solve_us),
        "checksum": checksum,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="timed passes through every state in each scenario (default: 10)",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="untimed warmup passes after backend preparation (default: 1)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="maximum iterations supplied to each LLE solve (default: 100)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="equilibrium tolerance supplied to each LLE solve (default: 1e-6)",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.name for scenario in SCENARIOS],
        help="scenario to run; may be supplied multiple times",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable JSON report instead of the text table",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="list available scenarios and exit",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.max_iter < 1:
        parser.error("--max-iter must be at least 1")
    if not math.isfinite(args.tol) or args.tol <= 0.0:
        parser.error("--tol must be a positive finite number")
    return args


def print_text(results: list[dict[str, object]], args: argparse.Namespace) -> None:
    total_states = sum(int(result["state_count"]) for result in results)
    total_solves = sum(int(result["solve_count"]) for result in results)
    total_seconds = sum(float(result["timed_seconds"]) for result in results)
    print("LLE solver benchmark")
    print(
        f"scenarios={len(results)} states/pass={total_states} repeats={args.repeats} "
        f"warmups={args.warmups} max_iter={args.max_iter} tol={args.tol:g}"
    )
    print()
    print(
        f"{'scenario':<38} {'method':<9} {'states':>6} {'LLE':>5} "
        f"{'median us':>12} {'p95 us':>12} {'solve/s':>11}"
    )
    for result in results:
        print(
            f"{result['name']:<38} {result['method']:<9} "
            f"{result['state_count']:>6} {result['lle_states_per_pass']:>5} "
            f"{result['median_us_per_solve']:>12.3f} "
            f"{result['p95_us_per_solve']:>12.3f} "
            f"{result['solves_per_second']:>11.1f}"
        )
    print()
    print(
        f"total solves={total_solves} timed={total_seconds:.6f} s "
        f"aggregate={total_solves / total_seconds:.1f} solves/s"
    )


def main() -> int:
    args = parse_args()
    if args.list_scenarios:
        for scenario in SCENARIOS:
            print(
                f"{scenario.name:<38} {scenario.method:<9} "
                f"{len(scenario.states):>3} states  {scenario.description}"
            )
        return 0

    selected = [
        scenario
        for scenario in SCENARIOS
        if args.scenario is None or scenario.name in args.scenario
    ]
    results = [
        run_scenario(
            scenario,
            repeats=args.repeats,
            warmups=args.warmups,
            max_iter=args.max_iter,
            tol=args.tol,
        )
        for scenario in selected
    ]
    if args.json:
        print(json.dumps({
            "benchmark": "liquid_liquid_equilibrium",
            "repeats": args.repeats,
            "warmups": args.warmups,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "results": results,
        }, indent=2, sort_keys=True))
    else:
        print_text(results, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
