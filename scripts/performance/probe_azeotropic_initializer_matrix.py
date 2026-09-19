#!/usr/bin/env python3
"""Probe explicit azeotropic initialization across isolated regression columns."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator import Simulator
from thermodynamics import create_thermodynamics
from unit_operations_distillation import RigorousDistillation


CASE_NAMES = (
    "example_ethanol_water",
    "example_pressure_swing_10bar",
    "example_pressure_swing_1bar",
    "example_mixed_acids",
    "example_ethanol_benzene_beer",
    "example_ethanol_benzene_entrainer",
    "example_lactic_recovery",
    "example_3mp_polisher",
    "test_nitrile",
    "test_nrtl_rk_total",
    "test_nrtl_rk_total6",
    "test_nrtl_rk_total12",
    "test_nrtl_rk_mixed",
    "test_nrtl_rk_partial",
    "test_nrtl_rk_mass",
    "test_nrtl_rk_multifeed",
    "test_unifac_homogeneous",
    "test_unifac_appearance",
    "test_unifac_disappearance",
    "test_binary_chloroform",
    "test_butanol_077",
    "test_butanol_0773",
    "test_acrylic_056",
    "test_acrylic_water_limit",
    "test_acrylic_watered_9995",
)


def state(thermo, T, P, F, composition, *, flash=False):
    return thermo.calculate_state(
        T, P, F, composition, phase="liquid", flash=flash
    )


def ordinary_params(**updates):
    params = {
        "N_stages": 20,
        "feed_stage": 10,
        "reflux_ratio": 2.0,
        "D_to_F": 0.4,
        "P_condenser": 1.0,
        "P_drop_per_stage": 0.0,
        "condenser_type": "total",
        "mesh_tolerance": 1e-5,
        "max_iterations": 140,
        "max_jacobian_evaluations": 140,
    }
    params.update(updates)
    return params


def nrtl_rk_case(condenser="total", stages=16):
    thermo = create_thermodynamics(
        ["ethanol", "water", "benzene"], "NRTL-RK"
    )
    feed = state(
        thermo,
        298.15,
        5.0,
        20.0,
        {"ethanol": 0.35, "water": 0.25, "benzene": 0.40},
    )
    params = ordinary_params(
        N_stages=stages,
        feed_stage=max(1, stages // 2),
        reflux_ratio=2.0,
        D_to_F=0.4,
        P_condenser=5.0,
        condenser_type=condenser,
        stage_phase_model="VLLE",
        max_iterations=100,
        max_jacobian_evaluations=100,
    )
    if condenser == "mixed":
        params["distillate_vapor_fraction"] = 0.25
    return RigorousDistillation("MATRIX-NRTL-RK", thermo, params), {"feed": feed}


def acrylic_case(kind):
    simulator = Simulator.from_file(
        ROOT / "examples/lactic_acid_dehydration_pbr.pfd"
    ).initialize()
    thermo = simulator.thermo_packages["global"]
    flow = 70.66427321297607
    composition = {
        "H2O": 0.37049321630854976,
        "MIBK": 0.33780317143923255,
        "AA": 0.28311082465910160,
        "PA": 0.008592684817684328,
    }
    feed = state(thermo, 298.3188792640525, 1.01325, flow, composition)
    if kind == "moderate":
        cut = 0.56
    elif kind == "water_limit":
        cut = 0.5718453727817714
    else:
        component_flows = {
            component: feed.F * feed.composition[component]
            for component in composition
        }
        azeotrope = {"H2O": 0.6478905556344128, "MIBK": 0.3521094443655872}
        water_add = (
            component_flows["MIBK"]
            * azeotrope["H2O"]
            / azeotrope["MIBK"]
            - component_flows["H2O"]
        )
        mixed_flow = feed.F + water_add
        feed = state(
            thermo,
            feed.T,
            feed.P,
            mixed_flow,
            {
                "H2O": (component_flows["H2O"] + water_add) / mixed_flow,
                "MIBK": component_flows["MIBK"] / mixed_flow,
                "AA": component_flows["AA"] / mixed_flow,
                "PA": component_flows["PA"] / mixed_flow,
            },
        )
        target = 0.9995 * component_flows["MIBK"] / azeotrope["MIBK"]
        cut = target / mixed_flow
    params = ordinary_params(
        N_stages=20,
        feed_stage=10,
        reflux_ratio=1.2,
        D_to_F=cut,
        P_condenser=1.01325,
        stage_phase_model="VLLE",
        acceptable_mesh_residual=1e-4,
        max_iterations=140,
        max_jacobian_evaluations=140,
        vlle_colored_jacobian_fallback=False,
    )
    params["vlle_seed"] = "cheap"
    return RigorousDistillation("MATRIX-ACRYLIC", thermo, params), {"feed": feed}


def prepare_case(name):
    if name == "example_ethanol_water":
        thermo = create_thermodynamics(["ethanol", "water"], "UNIFAC")
        feed = state(thermo, 298.15, 1.0, 100.0, {"ethanol": 0.1, "water": 0.9})
        params = ordinary_params(reflux_ratio=4.0, D_to_F=0.11)
        params["initializer"] = "cmo"
    elif name == "example_pressure_swing_10bar":
        simulator = Simulator.from_file(
            ROOT / "examples/ethanol_pressure_swing_recycle_wasteful.pfd"
        ).initialize()
        thermo = simulator.thermo_packages["global"]
        feed = state(
            thermo, 419.41, 10.0, 446.4479,
            {"C2H5OH": 0.900526, "H2O": 0.099474},
        )
        params = ordinary_params(
            N_stages=60, feed_stage=16, reflux_ratio=5.0, D_to_F=0.8,
            P_condenser=10.0, T_min=360.0, T_max=460.0,
            mesh_tolerance=1e-6, max_iterations=400,
            max_jacobian_evaluations=400,
        )
        params["initializer"] = "cheap_estimate"
    elif name == "example_pressure_swing_1bar":
        simulator = Simulator.from_file(
            ROOT / "examples/ethanol_pressure_swing_recycle_wasteful.pfd"
        ).initialize()
        thermo = simulator.thermo_packages["global"]
        feed = state(
            thermo, 350.56, 1.0, 357.1583,
            {"C2H5OH": 0.876671, "H2O": 0.123329}, flash=True,
        )
        params = ordinary_params(
            N_stages=60, feed_stage=34, reflux_ratio=4.5, D_to_F=0.97,
            P_condenser=1.0, T_min=330.0, T_max=390.0,
            mesh_tolerance=1e-6, max_iterations=400,
            max_jacobian_evaluations=400,
        )
        params["initializer"] = "coarse_rigorous"
    elif name == "example_mixed_acids":
        simulator = Simulator.from_file(
            ROOT / "examples/mixed_acid_dehydration_uniquac_vdm.pfd"
        ).initialize()
        thermo = simulator.thermo_packages["global"]
        feed = state(
            thermo, 375.350164, 1.01325, 100.0,
            {
                "H2O": 0.7160306973,
                "CH3COOH": 0.1074010275,
                "C2H5COOH": 0.0870644380,
                "C2H3COOH": 0.0895038372,
            },
        )
        params = ordinary_params(
            N_stages=30, feed_stage=15, reflux_ratio=2.0, D_to_F=0.7,
            P_condenser=1.01325, mesh_tolerance=1e-6,
            acceptable_mesh_residual=5e-5, T_min=300.0, T_max=430.0,
            max_iterations=120, max_jacobian_evaluations=120,
        )
        params["initializer"] = "cheap_estimate"
    elif name == "example_ethanol_benzene_beer":
        thermo = create_thermodynamics(["ethanol", "water"], "UNIFAC")
        feed = state(thermo, 298.15, 1.0, 100.0, {"ethanol": 0.1, "water": 0.9})
        params = ordinary_params(
            N_stages=30, feed_stage=21, reflux_ratio=8.0, D=11.0,
        )
        params.pop("D_to_F")
        params["initializer"] = "cmo"
    elif name == "example_ethanol_benzene_entrainer":
        thermo = create_thermodynamics(
            ["ethanol", "water", "benzene"], "UNIFAC"
        )
        feed = state(
            thermo, 326.31357674962914, 1.0, 17.0,
            {
                "water": 0.07643799099824727,
                "benzene": 0.3529411764705882,
                "ethanol": 0.5706208325311645,
            },
        )
        params = ordinary_params(
            N_stages=16, feed_stage=8, reflux_ratio=4.0, D=10.8,
            stage_phase_model="VLLE", max_iterations=300,
            max_jacobian_evaluations=300,
        )
        params.pop("D_to_F")
    elif name == "example_lactic_recovery":
        simulator = Simulator.from_file(
            ROOT / "examples/lactic_acid_dehydration_pbr.pfd"
        ).initialize()
        thermo = simulator.thermo_packages["global"]
        feed = state(
            thermo, 298.48253699169915, 0.16, 79.79197900206256,
            {
                "H2O": 0.4393767665649053,
                "AA": 0.25149758937416866,
                "AcH": 9.632033783440319e-8,
                "PA": 0.007609656944632908,
                "MIBK": 0.3015158904858782,
            },
        )
        feed.thermo_scope = "global"
        params = ordinary_params(
            N_stages=20, feed_stage=10, reflux_ratio=0.5,
            D_to_F=0.7405235022673647, P_condenser=0.16,
            stage_phase_model="VLLE", mesh_tolerance=1e-5,
            acceptable_mesh_residual=1e-4, max_iterations=180,
            max_jacobian_evaluations=180,
        )
        params["vlle_seed"] = "cheap"
    elif name == "example_3mp_polisher":
        simulator = Simulator.from_file(
            ROOT / "examples/3methylpyridine_ether_extraction_recycle.pfd"
        ).initialize()
        thermo = simulator.thermo_packages["global"]
        feed = state(
            thermo, 365.49300080272144, 1.0, 2.7591076826414205,
            {
                "water": 0.386010390735681,
                "diethyl ether": 0.030231403146349166,
                "3-methylpyridine": 0.5837582061179698,
            },
        )
        params = ordinary_params(
            N_stages=20, feed_stage=10, reflux_ratio=1.0,
            P_condenser=1.0, condenser_type="partial",
            stage_phase_model="VLLE", max_iterations=100,
            max_jacobian_evaluations=100,
            distillate_mass_fraction=0.230528,
        )
        params.pop("D_to_F")
    elif name == "test_nitrile":
        groups = {
            "acrylonitrile": {68: 1},
            "acetonitrile": {40: 1},
            "water": {16: 1},
        }
        thermo = create_thermodynamics(
            ["acrylonitrile", "acetonitrile", "water"],
            "UNIFNIST", None, groups,
        )
        composition = {
            "acrylonitrile": 0.6,
            "acetonitrile": 0.1,
            "water": 0.3,
        }
        pressure = 1.01325
        feed = state(
            thermo, thermo.bubble_point_T(composition, pressure), pressure,
            100.0, composition,
        )
        params = ordinary_params(
            N_stages=80, feed_stage=70, reflux_ratio=10.0,
            D_to_F=0.861, P_condenser=pressure, mesh_tolerance=1e-6,
            max_iterations=400, max_jacobian_evaluations=400,
        )
    elif name.startswith("test_nrtl_rk_"):
        suffix = name.removeprefix("test_nrtl_rk_")
        if suffix in ("total", "mixed", "partial"):
            stages = 12 if suffix in ("mixed", "partial") else 16
            return nrtl_rk_case(suffix, stages=stages)
        if suffix in ("total6", "total12"):
            return nrtl_rk_case("total", stages=int(suffix.removeprefix("total")))
        column, inlets = nrtl_rk_case("total")
        if suffix == "mass":
            composition = {
                "ethanol": 0.3108141416332551,
                "water": 0.2579577018625229,
                "benzene": 0.43122815650422197,
            }
            column.params.pop("D_to_F")
            column.params["distillate_mass_flow"] = (
                8.0 * column.thermo.mixture_MW(composition)
            )
            return column, inlets
        thermo = column.thermo
        feed = state(
            thermo, 298.15, 5.0, 12.0,
            {"ethanol": 0.40, "water": 0.20, "benzene": 0.40},
        )
        wash = state(
            thermo, 298.15, 5.0, 8.0,
            {"ethanol": 0.275, "water": 0.325, "benzene": 0.40},
        )
        column.params.update({
            "N_stages": 12,
            "feed_stage": 6,
            "feed_stages": {"feed": 7, "wash": 3},
        })
        return column, {"feed": feed, "wash": wash}
    elif name.startswith("test_unifac_"):
        thermo = create_thermodynamics(
            ["ethanol", "water", "benzene"], "UNIFAC"
        )
        if name == "test_unifac_homogeneous" or name == "test_unifac_disappearance":
            composition = {"ethanol": 0.75, "water": 0.10, "benzene": 0.15}
        else:
            composition = {"ethanol": 0.55, "water": 0.15, "benzene": 0.30}
        feed = state(thermo, 298.15, 1.0, 20.0, composition)
        params = ordinary_params(
            N_stages=12, feed_stage=6, reflux_ratio=2.0, D_to_F=0.4,
            stage_phase_model="VLLE", max_iterations=100,
            max_jacobian_evaluations=100,
        )
        if name == "test_unifac_appearance":
            params["vlle_seed"] = "cheap"
            params["vlle_initial_topology"] = "all_vle"
        elif name == "test_unifac_disappearance":
            params["vlle_initial_topology"] = "all_vlle"
    elif name == "test_binary_chloroform":
        thermo = create_thermodynamics(["water", "chloroform"], "UNIFNIST")
        feed = state(
            thermo, 298.15, 1.01325, 20.0,
            {"water": 0.5, "chloroform": 0.5},
        )
        params = ordinary_params(
            N_stages=6, feed_stage=3, reflux_ratio=2.0, D_to_F=0.5,
            P_condenser=1.01325, stage_phase_model="VLLE",
            max_iterations=120, max_jacobian_evaluations=120,
        )
        params["vlle_seed"] = "cheap"
    elif name.startswith("test_butanol_"):
        thermo = create_thermodynamics(["butanol", "water"], "NRTL")
        feed = state(
            thermo, 298.15, 1.0, 100.0,
            {"butanol": 0.4, "water": 0.6},
        )
        cut = 0.77 if name.endswith("077") else 0.773
        params = ordinary_params(
            N_stages=20, feed_stage=10, reflux_ratio=1.2, D_to_F=cut,
            stage_phase_model="VLLE", max_iterations=120,
            max_jacobian_evaluations=120,
        )
        if name.endswith("0773"):
            params["vlle_seed"] = "cheap"
    elif name == "test_acrylic_056":
        return acrylic_case("moderate")
    elif name == "test_acrylic_water_limit":
        return acrylic_case("water_limit")
    elif name == "test_acrylic_watered_9995":
        return acrylic_case("watered")
    else:
        raise ValueError(name)
    return RigorousDistillation(f"MATRIX-{name}", thermo, params), {"feed": feed}


def aggregate_for_candidates(column, inlets):
    stages = int(column.get_param("N_stages", column.get_param("stages", 10)))
    feed_stage = int(column.get_param("feed_stage", max(1, stages // 2)))
    inlet, _feeds = column._aggregate_feeds(inlets, stages, feed_stage)
    components = column._component_order(inlet)
    pressure = float(column.get_param("P_condenser", inlet.P))
    return inlet, components, pressure


def classify_error(error):
    message = str(error)
    if "could not build an azeotropic initializer" in message:
        return "no_seed"
    if "azeotropic initializer failed" in message:
        return "seed_construction_error"
    if "topology cycle" in message:
        return "topology_cycle"
    if "MESH solve failed" in message or "MES solve failed" in message:
        return "newton_failure"
    return "other_failure"


def worker(name, mode):
    column, inlets = prepare_case(name)
    inlet, components, pressure = aggregate_for_candidates(column, inlets)
    started = time.perf_counter()
    try:
        candidates = column._vle_azeotrope_candidates(components, pressure)
    except Exception as exc:
        candidates = []
        candidate_error = f"{type(exc).__name__}: {exc}"
    else:
        candidate_error = None

    if mode == "vle_azeotropic":
        if column._stage_phase_model() == "VLLE":
            column.params["vlle_seed"] = "homogeneous"
            column.params["vlle_homogeneous_initializer"] = "azeotropic"
        else:
            column.params["initializer"] = "azeotropic"
    try:
        result = column.solve(inlets)
    except Exception as exc:
        summary = {
            "case": name,
            "mode": mode,
            "success": False,
            "failure": classify_error(exc),
            "error": f"{type(exc).__name__}: {exc}",
            "candidate_count": len(candidates),
            "candidates": candidates,
            "candidate_error": candidate_error,
            "elapsed_seconds": time.perf_counter() - started,
        }
    else:
        performance = result.performance
        summary = {
            "case": name,
            "mode": mode,
            "success": True,
            "failure": None,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "candidate_error": candidate_error,
            "elapsed_seconds": time.perf_counter() - started,
            "initializer": performance.get("initializer"),
            "mesh_residual": performance.get("mesh_residual"),
            "jacobian_evaluations": performance.get("jacobian_evaluations"),
            "solver_work_basis": performance.get("solver_work_basis"),
            "attempts": performance.get("vlle_initializer_attempts", []),
            "topology": performance.get("vlle_topology", "VLE"),
            "distillate": result.outlet_streams["distillate"].composition,
            "bottoms": result.outlet_streams["bottoms"].composition,
        }
    print("MATRIX_JSON " + json.dumps(summary, sort_keys=True))


def run_worker_process(name, mode):
    completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__)),
                "--worker",
                name,
                "--mode",
                mode,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    marker = next(
        (
            line for line in completed.stdout.splitlines()
            if line.startswith("MATRIX_JSON ")
        ),
        None,
    )
    if marker is not None:
        return marker
    summary = {
        "case": name,
        "success": False,
        "failure": "worker_failure",
        "error": completed.stderr or completed.stdout,
        "returncode": completed.returncode,
    }
    return "MATRIX_JSON " + json.dumps(summary, sort_keys=True)


def run_all(processes, mode):
    with ThreadPoolExecutor(max_workers=processes) as executor:
        futures = {
            executor.submit(run_worker_process, name, mode): name
            for name in CASE_NAMES
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=CASE_NAMES)
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=("baseline", "vle_azeotropic"),
        default="vle_azeotropic",
    )
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.mode)
    else:
        run_all(max(1, args.processes), args.mode)


if __name__ == "__main__":
    main()
