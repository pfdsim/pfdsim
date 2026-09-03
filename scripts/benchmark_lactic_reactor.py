#!/usr/bin/env python3
"""Benchmark and optionally profile the isolated lactic-acid PBR."""

from __future__ import annotations

import argparse
import cProfile
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator import Simulator
from vapor_dimerization import MultiVaporDimerizationModel


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vdm-mode",
        choices=("fused", "readable"),
        default="fused",
        help="Use or disable the fused multi-acid vapor closure.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Write raw cProfile statistics for the timed reactor solve.",
    )
    parser.add_argument(
        "--association-enthalpy-mode",
        choices=("scalar", "state"),
        default="scalar",
        help="Use the compiled scalar path or reconstruct the complete state.",
    )
    return parser.parse_args()


def main() -> None:
    options = arguments()
    if options.vdm_mode == "readable":
        MultiVaporDimerizationModel.compiled_vapor_closure = (
            lambda *_args, **_kwargs: None
        )
    if options.association_enthalpy_mode == "state":
        MultiVaporDimerizationModel.compiled_association_enthalpy = (
            lambda *_args, **_kwargs: None
        )

    setup_start = time.perf_counter()
    with redirect_stdout(sys.stderr):
        simulator = Simulator.from_file(
            ROOT / "examples" / "lactic_acid_dehydration_pbr.pfd"
        ).initialize()
        solver = simulator.solver
        solver._initialize_streams()
    setup_elapsed = time.perf_counter() - setup_start

    reactor = solver.units["R-101"]
    inlet = solver.streams["Wet-Lactic-Feed"]
    profiler = cProfile.Profile() if options.profile else None
    start = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    result = reactor.solve({"in": inlet})
    if profiler is not None:
        profiler.disable()
    elapsed = time.perf_counter() - start
    if profiler is not None:
        profiler.dump_stats(str(options.profile))

    performance = result.performance
    print(json.dumps({
        "benchmark": "lactic_isolated_reactor",
        "vdm_mode": options.vdm_mode,
        "association_enthalpy_mode": options.association_enthalpy_mode,
        "profiled": profiler is not None,
        "setup_elapsed_s": setup_elapsed,
        "reactor_elapsed_s": elapsed,
        "solver_method": performance["solver_method"],
        "solver_evaluations": performance["solver_evaluations"],
        "profile_points": performance["profile_points"],
        "T_out_C": performance["T_out_C"],
        "P_out_bar": performance["P_out_bar"],
        "LLA_conversion": performance["component_conversions"]["LLA"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
