#!/usr/bin/env python3
"""Compare projected VLLE disappearance on three isolated example columns."""

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
from thermodynamics import create_thermodynamics
from unit_operations_distillation import RigorousDistillation


CASES = ("ethanol_benzene", "methylpyridine", "lactic", "butanol_0773")


def prepare_case(name):
    if name == "ethanol_benzene":
        simulator = Simulator.from_file(
            ROOT / "examples/ethanol_benzene_azeotropic_distillation_rigorous.pfd"
        ).initialize()
        thermo = simulator.thermo_packages["global"]
        column = simulator.solver.units["AZE-COL"]
        feed = thermo.calculate_state(
            326.31357674962914,
            1.0,
            17.0,
            {
                "water": 0.07643799099824727,
                "benzene": 0.3529411764705882,
                "ethanol": 0.5706208325311645,
            },
            phase="liquid",
            flash=False,
        )
        return column, feed
    if name == "methylpyridine":
        simulator = Simulator.from_file(
            ROOT / "examples/3methylpyridine_ether_extraction_recycle.pfd"
        ).initialize()
        thermo = simulator.thermo_packages["global"]
        column = simulator.solver.units["COL-200"]
        feed = thermo.calculate_state(
            365.49300080272144,
            1.0,
            2.7591076826414205,
            {
                "water": 0.386010390735681,
                "diethyl ether": 0.030231403146349166,
                "3-methylpyridine": 0.5837582061179698,
            },
            phase="liquid",
            flash=False,
        )
        return column, feed
    if name == "lactic":
        simulator = Simulator.from_file(
            ROOT / "examples/lactic_acid_dehydration_pbr.pfd"
        ).initialize()
        thermo = simulator.thermo_packages["global"]
        column = simulator.solver.units["C-401"]
        feed = thermo.calculate_state(
            298.48253699169915,
            0.16,
            79.79197900206256,
            {
                "H2O": 0.4393767665649053,
                "AA": 0.25149758937416866,
                "AcH": 9.632033783440319e-8,
                "PA": 0.007609656944632908,
                "MIBK": 0.3015158904858782,
            },
            phase="liquid",
            flash=False,
        )
        feed.thermo_scope = "global"
        column.solve_context = {
            "recycle_evaluation": 1,
            "recycle_final_pass": False,
            "expensive_diagnostics": True,
        }
        return column, feed
    if name == "butanol_0773":
        thermo = create_thermodynamics(["butanol", "water"], "NRTL")
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {"butanol": 0.4, "water": 0.6},
            phase="liquid",
            flash=False,
        )
        column = RigorousDistillation(
            "BUTANOL-0773",
            thermo,
            {
                "N_stages": 20,
                "feed_stage": 10,
                "reflux_ratio": 1.2,
                "D_to_F": 0.773,
                "P_condenser": 1.0,
                "P_drop_per_stage": 0.0,
                "condenser_type": "total",
                "stage_phase_model": "VLLE",
                "vlle_seed": "cheap",
                "max_iterations": 180,
                "max_jacobian_evaluations": 180,
            },
        )
        return column, feed
    raise ValueError(name)


def worker(case, mode, gate_fraction=None, gate_contraction_ratio=None):
    probe = install_probe(
        mode,
        gate_fraction=gate_fraction,
        gate_contraction_ratio=gate_contraction_ratio,
    )
    column, feed = prepare_case(case)
    summary = solve_and_summarize(case, mode, column, feed, probe)
    column.solve_context = {}
    print_summary(summary)


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
