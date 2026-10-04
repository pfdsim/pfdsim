"""Command-line interface for running pfdsim process simulations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .pfd_parser import ParseError
else:
    from pfd_parser import ParseError
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .simulator import SimulationError, Simulator
else:
    from simulator import SimulationError, Simulator
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from . import __version__
else:
    from __init__ import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfdsim",
        description="Run a pfdsim process flow diagram. Use 'pfdsim fit --help' for parameter fitting.",
    )
    parser.add_argument("file", type=Path, help="input .pfd file")
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        const="",
        metavar="PATH",
        help="write a .pfr file; omit PATH to use the input filename",
    )
    parser.add_argument(
        "--pfr",
        "--full-pfr",
        action="store_true",
        help="print the complete .pfr report instead of the default summary",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print solver progress to stderr",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _output_path(input_path: Path, value: str) -> Path:
    if not value:
        return input_path.with_suffix(".pfr")

    output_path = Path(value)
    if output_path.suffix.lower() != ".pfr":
        if output_path.suffix:
            output_path = output_path.with_suffix(".pfr")
        else:
            output_path = Path(f"{output_path}.pfr")
    return output_path


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "unknown"
    return f"{value:.{digits}f}"


def _print_status(sim: Simulator) -> None:
    result = sim.result
    assert result is not None

    process_name = sim.pfd.metadata.process_name or "Unnamed Process"
    print(f"Process: {process_name}")
    print(f"Status: {'CONVERGED' if result.converged else 'NOT CONVERGED'}")
    print(f"Thermodynamics: {sim.thermo_method}")
    print(f"Iterations: {result.iterations}")
    print(f"Mass balance error: {result.mass_balance_error * 100:.4f}%")
    print(f"Energy balance error: {result.energy_balance_error * 100:.4f}%")

    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    else:
        print("Errors: none")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    else:
        print("Warnings: none")


def _print_final_outlets(sim: Simulator) -> None:
    result = sim.result
    assert result is not None

    final_streams = [stream for stream in sim.pfd.streams if stream.destination.is_product]
    print("\nFinal outlets:")
    if not final_streams:
        print("  none")
        return

    for stream in final_streams:
        state = result.streams.get(stream.id)
        print(f"  {stream.id}:")
        if state is None:
            print("    state unavailable")
            continue

        print(f"    temperature: {_format_float(state.T - 273.15, 2)} C")
        print(f"    pressure: {_format_float(state.P)} bar")
        print(f"    molar flow: {_format_float(state.F)} kmol/h")
        if state.MW is not None:
            print(f"    mass flow: {_format_float(state.mass_flow(), 2)} kg/h")
        print(f"    vapor fraction: {_format_float(state.vapor_fraction)}")
        print("    composition:")
        for component, fraction in sorted(
            state.composition.items(), key=lambda item: (-item[1], item[0])
        ):
            component_flow = state.F * fraction
            print(
                f"      {component}: x={fraction:.6f}, "
                f"F={component_flow:.4f} kmol/h"
            )


def _print_unit_duties(sim: Simulator) -> None:
    result = sim.result
    assert result is not None

    print("\nUnit duties:")
    if not sim.pfd.units:
        print("  none")
        return

    for unit in sim.pfd.units:
        unit_result = result.units.get(unit.id)
        print(f"  {unit.id} ({unit.unit_type}):")
        if unit_result is None:
            print("    results unavailable")
            continue

        heating, cooling, net_heat = sim._unit_heat_summary(unit_result)
        if heating > 0.0 or cooling > 0.0:
            print(f"    heating: {heating / 3600:.4f} kW")
            print(f"    cooling: {cooling / 3600:.4f} kW")
            print(f"    net heat: {net_heat / 3600:.4f} kW")
        else:
            print(f"    heat: {unit_result.heat_duty / 3600:.4f} kW")

        utility_work = sim._unit_utility_work(unit_result)
        print(f"    work: {utility_work / 3600:.4f} kW")
        if utility_work != unit_result.work:
            print(f"    process work: {unit_result.work / 3600:.4f} kW")


def _print_summary(sim: Simulator) -> None:
    _print_status(sim)
    _print_final_outlets(sim)
    _print_unit_duties(sim)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "fit":
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .fit_cli import main as fit_main
        else:
            from fit_cli import main as fit_main
        return fit_main(argv[1:])
    parser = _build_parser()
    args = parser.parse_args(argv)
    input_path = args.file.expanduser()

    output_path = None
    if args.output is not None:
        output_path = _output_path(input_path, args.output).expanduser()
        if _same_path(input_path, output_path):
            parser.error("input and output paths must be different")

    try:
        simulator = Simulator.from_file(str(input_path))
        progress_callback = None
        if args.verbose:
            progress_callback = lambda message: print(message, file=sys.stderr, flush=True)
        result = simulator.run(progress_callback=progress_callback)

        if args.pfr:
            print(simulator._generate_pfr())
        else:
            _print_summary(simulator)

        if output_path is not None:
            simulator.write_results(str(output_path))
            print(f"Wrote {output_path}", file=sys.stderr)

        return 0 if result.converged else 1
    except (OSError, UnicodeError, ParseError, SimulationError) as error:
        print(f"pfdsim: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
