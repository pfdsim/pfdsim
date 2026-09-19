#!/usr/bin/env python3
"""Compare projected VLLE disappearance on acrylic-acid contraction cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.performance.vlle_projection_benchmark import (
    install_probe,
    print_summary,
    solve_and_summarize,
)
from simulator import Simulator
from unit_operations_distillation import RigorousDistillation


CASES = ("water_limit", "watered_9995")
FEED_FLOW = 70.66427321297607
FEED_Z = {
    "H2O": 0.37049321630854976,
    "MIBK": 0.33780317143923255,
    "AA": 0.28311082465910160,
    "PA": 0.008592684817684328,
}
AZEOTROPE = {
    "H2O": 0.6478905556344128,
    "MIBK": 0.3521094443655872,
}


def column_params(cut):
    return {
        "N_stages": 20,
        "feed_stage": 10,
        "reflux_ratio": 1.2,
        "D_to_F": cut,
        "P_condenser": 1.01325,
        "P_drop_per_stage": 0.0,
        "condenser_type": "total",
        "stage_phase_model": "VLLE",
        "vlle_seed": "cheap",
        "mesh_tolerance": 1e-5,
        "acceptable_mesh_residual": 1e-4,
        "max_iterations": 140,
        "max_jacobian_evaluations": 140,
        "vlle_colored_jacobian_fallback": False,
    }


def prepare_case(name):
    simulator = Simulator.from_file(
        ROOT / "examples/lactic_acid_dehydration_pbr.pfd"
    ).initialize()
    thermo = simulator.thermo_packages["global"]
    feed = thermo.calculate_state(
        298.3188792640525,
        1.01325,
        FEED_FLOW,
        FEED_Z,
        phase="liquid",
        flash=False,
    )

    if name == "water_limit":
        cut = 0.5718453727817714
    elif name == "watered_9995":
        component_flows = {
            component: feed.F * feed.composition[component]
            for component in FEED_Z
        }
        water_add = (
            component_flows["MIBK"]
            * AZEOTROPE["H2O"]
            / AZEOTROPE["MIBK"]
            - component_flows["H2O"]
        )
        mixed_flow = feed.F + water_add
        mixed_z = {
            "H2O": (component_flows["H2O"] + water_add) / mixed_flow,
            "MIBK": component_flows["MIBK"] / mixed_flow,
            "AA": component_flows["AA"] / mixed_flow,
            "PA": component_flows["PA"] / mixed_flow,
        }
        feed = thermo.calculate_state(
            feed.T,
            feed.P,
            mixed_flow,
            mixed_z,
            phase="liquid",
            flash=False,
        )
        target_distillate = (
            0.9995 * component_flows["MIBK"] / AZEOTROPE["MIBK"]
        )
        cut = target_distillate / mixed_flow
    else:
        raise ValueError(name)

    column = RigorousDistillation(
        f"BENCHMARK-{name.upper()}", thermo, column_params(cut)
    )
    return column, feed


def worker(case, mode, gate_fraction=None, gate_contraction_ratio=None):
    probe = install_probe(
        mode,
        gate_fraction=gate_fraction,
        gate_contraction_ratio=gate_contraction_ratio,
    )
    column, feed = prepare_case(case)
    print_summary(solve_and_summarize(case, mode, column, feed, probe))


def run_all():
    for case in CASES:
        for mode in ("baseline", "projected", "screened"):
            command = [sys.executable, str(Path(__file__)), "--worker", case, mode]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            marker = next(
                line for line in completed.stdout.splitlines()
                if line.startswith("BENCHMARK_JSON ")
            )
            summary = json.loads(marker.removeprefix("BENCHMARK_JSON "))
            print_summary(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("CASE", "MODE"))
    parser.add_argument("--gate-fraction", type=float)
    parser.add_argument("--gate-contraction-ratio", type=float)
    args = parser.parse_args()
    if args.worker:
        worker(
            *args.worker,
            gate_fraction=args.gate_fraction,
            gate_contraction_ratio=args.gate_contraction_ratio,
        )
    else:
        run_all()


if __name__ == "__main__":
    main()
