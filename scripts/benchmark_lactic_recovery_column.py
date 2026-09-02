#!/usr/bin/env python3
"""Benchmark sequential recycle evaluations of the lactic recovery column."""

from __future__ import annotations

import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator import Simulator


BASE_COMPOSITION = {
    "H2O": 0.4393767665649053,
    "AA": 0.25149758937416866,
    "AcH": 9.632033783440319e-8,
    "PA": 0.007609656944632908,
    "MIBK": 0.3015158904858782,
}


def normalized(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    return {component: value / total for component, value in values.items()}


def column_feed(thermo, *, perturbed: bool):
    composition = dict(BASE_COMPOSITION)
    flow = 79.79197900206256
    temperature = 298.48253699169915
    if perturbed:
        composition["H2O"] += 2.0e-5
        composition["MIBK"] -= 2.0e-5
        composition = normalized(composition)
        flow *= 1.0002
        temperature += 0.01
    state = thermo.calculate_state(
        temperature,
        0.16,
        flow,
        composition,
        phase="liquid",
        flash=False,
    )
    state.thermo_scope = "global"
    return state


def result_summary(result, elapsed_s: float) -> dict:
    performance = result.performance
    return {
        "elapsed_s": elapsed_s,
        "initializer": performance["initializer"],
        "solver_iterations": performance["solver_iterations"],
        "function_evaluations": performance["function_evaluations"],
        "jacobian_evaluations": performance["jacobian_evaluations"],
        "mesh_residual": performance["mesh_residual"],
        "component_balance_error": performance["component_balance_error"],
        "topology": performance["vlle_topology"],
        "topology_solves": performance["vlle_topology_solves"],
        "topology_history": performance["vlle_topology_history"],
        "distillate_flow": result.outlet_streams["distillate"].F,
        "bottoms_flow": result.outlet_streams["bottoms"].F,
    }


def main() -> None:
    with redirect_stdout(sys.stderr):
        simulator = Simulator.from_file(
            ROOT / "examples" / "lactic_acid_dehydration_pbr.pfd"
        ).initialize()
    thermo = simulator.thermo_packages["global"]
    column = simulator.solver.units["C-401"]
    backend = thermo._compiled_lle_backend(298.15)
    lle_counts = {"attempts": 0, "fallbacks": 0}
    if backend is not None:
        original_split = backend.split

        def counted_split(*args, **kwargs):
            lle_counts["attempts"] += 1
            result = original_split(*args, **kwargs)
            if result is None:
                lle_counts["fallbacks"] += 1
            return result

        backend.split = counted_split

    results = []
    for evaluation, perturbed in ((1, False), (2, True)):
        column.solve_context = {
            "recycle_evaluation": evaluation,
            "recycle_final_pass": False,
            "expensive_diagnostics": True,
        }
        start = time.perf_counter()
        result = column.solve({"feed": column_feed(thermo, perturbed=perturbed)})
        elapsed = time.perf_counter() - start
        results.append(result_summary(result, elapsed))
    column.solve_context = {}

    print(json.dumps({
        "benchmark": "lactic_recovery_column_sequential_recycle",
        "second_feed_perturbation": {
            "relative_flow": 2.0e-4,
            "water_to_mibk_mole_fraction": 2.0e-5,
            "temperature_K": 0.01,
        },
        "evaluations": results,
        "compiled_lle": lle_counts,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
