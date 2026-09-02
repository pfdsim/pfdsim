#!/usr/bin/env python3
"""Benchmark sequential recycle evaluations of the lactic acid extractor."""

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


SOLVENT_FLOW = 27.875679167306025
SOLVENT_COMPOSITION = {
    "MIBK": 0.8833901899406459,
    "H2O": 0.11660840176852998,
    "LLA": 0.0,
    "AA": 1.111434316327607e-6,
    "AcH": 2.2318880457366943e-7,
    "PA": 7.366770325223948e-8,
    "PD23": 0.0,
    "CO": 0.0,
    "CO2": 0.0,
}
RAFFINATE_FLOW = 148.21997987554627
RAFFINATE_COMPOSITION = {
    "AA": 0.001972084844295868,
    "AcH": 9.491093062317083e-8,
    "PA": 2.4636888920771925e-8,
    "PD23": 4.713888401351184e-10,
    "CO": 1.7245465220667993e-14,
    "CO2": 1.8690122490497535e-13,
    "H2O": 0.9940071923979468,
    "MIBK": 0.0040206027383404555,
    "LLA": 4.334866063490966e-15,
}
EXTRACT_FLOW = 70.99336068351404
EXTRACT_COMPOSITION = {
    "AA": 0.28266711779660464,
    "AcH": 1.0258551438080538e-7,
    "PA": 0.008552762952818442,
    "PD23": 3.483532927057614e-10,
    "CO": 1.274428970400821e-14,
    "CO2": 1.3811880521329504e-13,
    "H2O": 0.3703094987982777,
    "MIBK": 0.33847051751827806,
    "LLA": 2.611552356458849e-15,
}


def normalized(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    return {component: value / total for component, value in values.items()}


def liquid_state(thermo, temperature, flow, composition):
    state = thermo.calculate_state(
        temperature,
        1.01325,
        flow,
        composition,
        phase="liquid",
        flash=False,
    )
    state.thermo_scope = "extraction"
    return state


def inlets(simulator, *, perturbed: bool):
    thermo = simulator.thermo_packages["extraction"]
    components = simulator.solver.components
    feed_component_flows = {
        component: (
            RAFFINATE_FLOW * RAFFINATE_COMPOSITION.get(component, 0.0)
            + EXTRACT_FLOW * EXTRACT_COMPOSITION.get(component, 0.0)
            - SOLVENT_FLOW * SOLVENT_COMPOSITION.get(component, 0.0)
        )
        for component in components
    }
    feed_flow = sum(feed_component_flows.values())
    feed_composition = {
        component: value / feed_flow
        for component, value in feed_component_flows.items()
    }
    solvent_flow = SOLVENT_FLOW
    solvent_composition = dict(SOLVENT_COMPOSITION)
    solvent_temperature = 298.1357425027178
    if perturbed:
        solvent_flow *= 1.0002
        solvent_composition["H2O"] += 2.0e-5
        solvent_composition["MIBK"] -= 2.0e-5
        solvent_composition = normalized(solvent_composition)
        solvent_temperature += 0.01
    return {
        "feed": liquid_state(thermo, 298.15, feed_flow, feed_composition),
        "solvent": liquid_state(
            thermo,
            solvent_temperature,
            solvent_flow,
            solvent_composition,
        ),
    }


def result_summary(result, elapsed_s: float) -> dict:
    performance = result.performance
    return {
        "elapsed_s": elapsed_s,
        "initializer": performance["initializer"],
        "solver_iterations": performance["solver_iterations"],
        "function_evaluations": performance["function_evaluations"],
        "jacobian_evaluations": performance["jacobian_evaluations"],
        "jacobian_method": performance["jacobian_method"],
        "mesh_residual": performance["mesh_residual"],
        "component_balance_error": performance["component_balance_error"],
        "raffinate_flow": result.outlet_streams["raffinate"].F,
        "extract_flow": result.outlet_streams["extract"].F,
    }


def main() -> None:
    with redirect_stdout(sys.stderr):
        simulator = Simulator.from_file(
            ROOT / "examples" / "lactic_acid_dehydration_pbr.pfd"
        ).initialize()
    extractor = simulator.solver.units["X-301"]
    results = []
    for evaluation, perturbed in ((1, False), (2, True)):
        extractor.solve_context = {
            "recycle_evaluation": evaluation,
            "recycle_final_pass": False,
            "expensive_diagnostics": True,
        }
        start = time.perf_counter()
        result = extractor.solve(inlets(simulator, perturbed=perturbed))
        elapsed = time.perf_counter() - start
        results.append(result_summary(result, elapsed))
    extractor.solve_context = {}
    print(json.dumps({
        "benchmark": "lactic_extractor_sequential_recycle",
        "second_solvent_perturbation": {
            "relative_flow": 2.0e-4,
            "water_to_mibk_mole_fraction": 2.0e-5,
            "temperature_K": 0.01,
        },
        "evaluations": results,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
