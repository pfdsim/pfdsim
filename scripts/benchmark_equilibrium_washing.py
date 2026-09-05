"""Reproducible warm-filter benchmark, optionally against a Git revision.

Default input is the maintained melt-washing example. Pass a PFD explicitly
to benchmark another system. No random inputs or online lookup are introduced.
Use --revision HEAD before committing to measure the pre-change implementation.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pfd",
        nargs="?",
        default=str(repo / "examples/equilibrium_warm_melt_washing.pfd"),
    )
    parser.add_argument("--revision")
    parser.add_argument("--module-root", help=argparse.SUPPRESS)
    parser.add_argument("--cells", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--four-liquid-components",
        action="store_true",
        help="Extend the bundled example with methanol and acetone",
    )
    parser.add_argument("--thermo", help="Override the benchmark thermodynamic method")
    parser.add_argument(
        "--acid-bottoms",
        action="store_true",
        help="Isolated AA/PA/water/MIBK crystallizer and filter; no upstream flowsheet",
    )
    parser.add_argument(
        "--stack-after",
        type=float,
        help="Dump one diagnostic stack after this many seconds",
    )
    args = parser.parse_args()
    if args.stack_after:
        faulthandler.dump_traceback_later(args.stack_after)
    if args.revision:
        with tempfile.TemporaryDirectory(prefix="pfdsim-wash-benchmark-") as directory:
            for name in (
                "equilibrium_washing.py",
                "unit_operations_warm_filtration.py",
                "unit_operations_filtration.py",
                "liquid_mixture_viscosity.py",
            ):
                contents = subprocess.run(
                    ["git", "show", f"{args.revision}:{name}"],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                (Path(directory) / name).write_text(contents)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                str(Path(args.pfd).resolve()),
                "--module-root",
                directory,
            ]
            for name in ("cells", "steps"):
                if getattr(args, name) is not None:
                    command.extend([f"--{name}", str(getattr(args, name))])
            if args.four_liquid_components:
                command.append("--four-liquid-components")
            if args.thermo:
                command.extend(["--thermo", args.thermo])
            if args.acid_bottoms:
                command.append("--acid-bottoms")
            subprocess.run(command, check=True)
        return
    sys.path.insert(0, str(repo))
    if args.module_root:
        sys.path.insert(0, args.module_root)
    from pfd_parser import Parameter
    from simulator import Simulator

    started = time.perf_counter()
    if args.acid_bottoms:
        source = """PROCESS: Isolated acid bottoms warm washing benchmark
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: UNIQUAC
COMPONENTS:
    AA | Acrylic Acid | CAS=79-10-7, type=conventional_with_solid, PSD={distribution:lognormal, d50_um:400, GSD:1.6, basis:mass, classes:20}, particle_sphericity=0.75
    PA | Propionic Acid | CAS=79-09-4
    water | Water
    MIBK | 108-10-1
STREAM Feed : FEED -> C.in
    T = 295 [K]
    P = 1 [bar]
    F_mass = 100 [kg/h]
    w = AA:0.96, PA:0.038, water:0.001, MIBK:0.001
STREAM Slurry : C.out -> F.in
STREAM Wash : FEED -> F.wash
    T = 15.5 [C]
    P = 1 [bar]
    F_mass = 12 [kg/h]
    w = AA:1
STREAM Cake : F.cake -> PRODUCT
STREAM Filtrate : F.filtrate -> PRODUCT
UNIT C : Crystallizer
    T = 280 [K]
UNIT F : Filter
    washing_model = equilibrium
    cycle_time = 600 [s]
    porosity = 0.4
    capture_cut_size = 0 [um]
    medium_resistance = 1e9 [1/m]
    P_drop = 0.5 [bar]
    wash_cells = 10
    wash_steps = 100
    equilibrium_nucleus_diameter = 10 [um]
"""
        sim = Simulator.from_string(source)
    elif args.four_liquid_components:
        if (
            Path(args.pfd).resolve()
            != repo / "examples/equilibrium_warm_melt_washing.pfd"
        ):
            raise ValueError(
                "--four-liquid-components applies only to the bundled example"
            )
        source = Path(args.pfd).read_text()
        source = source.replace(
            "    ethanol | Ethanol",
            "    ethanol | Ethanol\n    methanol | Methanol\n    acetone | Acetone",
        )
        source = source.replace(
            "water:0.9, ethanol:0.1",
            "water:0.88, ethanol:0.08, methanol:0.02, acetone:0.02",
        )
        source = source.replace(
            "PROPERTY_CORRELATIONS:",
            "PROPERTY_CORRELATIONS:\n    methanol.mul | equation=exp_poly_x, A=-6.90775527898, B=-2, Tmin_K=220, Tmax_K=360\n    acetone.mul | equation=exp_poly_x, A=-6.90775527898, B=-2, Tmin_K=220, Tmax_K=360",
        )
        if args.thermo:
            source = source.replace(
                "THERMO_METHOD: IDEAL", "THERMO_METHOD: " + args.thermo
            )
        sim = Simulator.from_string(source)
    else:
        if args.thermo:
            from pfd_parser import parse_pfd

            pfd = parse_pfd(Path(args.pfd).read_text())
            pfd.metadata.thermo_method = args.thermo
            sim = Simulator(pfd)
        else:
            sim = Simulator.from_file(args.pfd)
    if sim.pfd.metadata.online_lookup:
        raise ValueError("Benchmark requires ONLINE_LOOKUP: false")
    for unit in sim.pfd.units:
        if unit.unit_type != "Filter":
            continue
        for name in ("cells", "steps"):
            value = getattr(args, name)
            if value is not None:
                parameter = "wash_" + name
                unit.params = [p for p in unit.params if p.name.lower() != parameter]
                unit.params.append(Parameter(name=parameter, value=str(value)))
    print("Initializing thermodynamics", file=sys.stderr, flush=True)
    sim.initialize()
    initialized = time.perf_counter()
    print("Solving isolated flowsheet", file=sys.stderr, flush=True)
    result = sim.run()
    if not result.converged:
        raise RuntimeError(result.errors)
    report = {
        "elapsed_s": time.perf_counter() - started,
        "initialization_s": initialized-started,
        "solve_s": time.perf_counter()-initialized,
        "mass_balance_error": result.mass_balance_error,
        "energy_balance_error": result.energy_balance_error,
        "filters": {},
    }
    for name, unit in result.units.items():
        if unit.performance.get("washing_model") == "equilibrium":
            report["filters"][name] = {
                key: unit.performance.get(key)
                for key in (
                    "area_m2",
                    "pressure_drop_bar",
                    "washing_time_s",
                    "cell_temperatures_K",
                    "net_solid_change_kmol_h",
                    "maximum_equilibrium_residual",
                    "wash_cells",
                    "wash_steps",
                )
            }
    print(json.dumps(report, indent=2))
    if args.stack_after:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
