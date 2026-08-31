#!/usr/bin/env python3
"""Probe equation-oriented VLLE stages in a rigorous distillation column.

This is a research harness, not production unit-operation code.  It compares:

* the existing homogeneous-VLE rigorous column;
* a cached nested LLE switch inside the existing column residual; and
* a pure equation-oriented active-set model with no nested phase split in its
  Newton residual.

The equation-oriented model assumes that two equilibrium liquid phases are
co-routed down the column as one aggregate liquid flow. A VLLE stage carries
two liquid compositions and a liquid phase fraction.  Component balances use
the aggregate liquid composition, liquid enthalpy is phase-fraction weighted,
and the vapor is in equilibrium with both liquids. Gamma-phi methods iterate
one shared vapor-fugacity state from the symmetric L1/L2 liquid-fugacity
target, matching the production rigorous-column architecture without adding
explicit vapor-composition variables to this probe. Separate liquid traffic, phase-specific
downcomers, side draws, and decanter condensers are deliberately outside it.

Examples
--------

Run the five bundled phase-topology cases with all three formulations::

    python scripts/probe_equation_oriented_vlle_column.py

Run only the mixed-topology case::

    python scripts/probe_equation_oriented_vlle_column.py --case mixed_top

Run only the fastest equation-oriented path::

    python scripts/probe_equation_oriented_vlle_column.py --formulation equation_oriented

List cases or retain a machine-readable report::

    python scripts/probe_equation_oriented_vlle_column.py --list-cases
    python scripts/probe_equation_oriented_vlle_column.py --json-output report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermodynamics import create_thermodynamics  # noqa: E402
from equilibrium_stage_vlle import shared_vlle_vapor_terms  # noqa: E402
from unit_operations_distillation import RigorousDistillation  # noqa: E402


@dataclass(frozen=True)
class ColumnCase:
    name: str
    description: str
    method: str
    components: tuple[str, ...]
    feed_z: dict[str, float]
    feed_T: float = 298.15
    feed_P: float = 1.0
    feed_F: float = 20.0
    stages: int = 12
    feed_stage: int = 6
    reflux_ratio: float = 2.0
    distillate_fraction: float = 0.4
    pressure_drop_per_stage: float = 0.0
    mesh_tolerance: float = 1e-5
    max_iterations: int = 80
    initial_topology: str = "screened"


CASES = (
    ColumnCase(
        name="no_split",
        description="high-ethanol ternary profile; homogeneous VLE control",
        method="UNIFAC",
        components=("ethanol", "water", "benzene"),
        feed_z={"ethanol": 0.75, "water": 0.10, "benzene": 0.15},
    ),
    ColumnCase(
        name="near_boundary",
        description="one top stage enters the liquid miscibility gap",
        method="UNIFAC",
        components=("ethanol", "water", "benzene"),
        feed_z={"ethanol": 0.55, "water": 0.30, "benzene": 0.15},
    ),
    ColumnCase(
        name="phase_disappearance",
        description="overactivated ghost liquids must collapse back to VLE",
        method="UNIFAC",
        components=("ethanol", "water", "benzene"),
        feed_z={"ethanol": 0.75, "water": 0.10, "benzene": 0.15},
        initial_topology="all_vlle",
    ),
    ColumnCase(
        name="mixed_top",
        description="upper stages split while lower stages remain homogeneous",
        method="UNIFAC",
        components=("ethanol", "water", "benzene"),
        feed_z={"ethanol": 0.55, "water": 0.15, "benzene": 0.30},
    ),
    ColumnCase(
        name="all_vlle",
        description="benzene-rich azeotropic profile; every stage splits",
        method="UNIFAC",
        components=("ethanol", "water", "benzene"),
        feed_z={"ethanol": 0.35, "water": 0.25, "benzene": 0.40},
        stages=16,
        feed_stage=8,
    ),
)


@dataclass
class RunSummary:
    formulation: str
    success: bool
    elapsed_s: float
    seed_elapsed_s: float = 0.0
    end_to_end_elapsed_s: Optional[float] = None
    residual: Optional[float] = None
    iterations: Optional[int] = None
    function_evaluations: Optional[int] = None
    jacobian_evaluations: Optional[int] = None
    topology: str = ""
    split_stages: int = 0
    outer_iterations: int = 0
    top_temperature_C: Optional[float] = None
    bottom_temperature_C: Optional[float] = None
    distillate: Optional[dict[str, float]] = None
    bottoms: Optional[dict[str, float]] = None
    component_balance_error: Optional[float] = None
    stability_checks: int = 0
    stability_cache_hits: int = 0
    message: str = ""


@dataclass
class ColumnProfile:
    T: list[float]
    aggregate_x: list[dict[str, float]]
    L: list[float]
    V: list[float]
    Q_cond: float
    Q_reb: float
    split_data: list[Optional[tuple[dict[str, float], dict[str, float], float]]]


@dataclass(frozen=True)
class StageLayout:
    start: int
    stop: int
    temperature: int
    x1: slice
    x2: Optional[slice]
    beta: Optional[int]
    liquid_flow: int
    vapor_flow: int
    active_vlle: bool


class ActiveSetChange(RuntimeError):
    """Request an immediate equation-oriented topology rebuild."""

    def __init__(
        self,
        profile: ColumnProfile,
        active: list[bool],
        reason: str,
    ):
        super().__init__(reason)
        self.profile = profile
        self.active = list(active)
        self.reason = reason


def normalize(composition: dict[str, float], components: tuple[str, ...]) -> dict[str, float]:
    values = {
        comp: max(float(composition.get(comp, 0.0)), 0.0)
        for comp in components
    }
    total = sum(values.values())
    if total <= 0.0:
        return {comp: 1.0 / len(components) for comp in components}
    return {comp: value / total for comp, value in values.items()}


def sigmoid(value: float) -> float:
    value = min(max(float(value), -60.0), 60.0)
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def logit(value: float) -> float:
    value = min(max(float(value), 1e-12), 1.0 - 1e-12)
    return math.log(value / (1.0 - value))


def composition_from_logits(values, components: tuple[str, ...]) -> dict[str, float]:
    logits = np.asarray(list(values) + [0.0], dtype=float)
    logits = np.clip(logits, -60.0, 60.0)
    logits -= np.max(logits)
    fractions = np.exp(logits)
    fractions /= np.sum(fractions)
    return {
        comp: float(fractions[index])
        for index, comp in enumerate(components)
    }


def composition_logits(composition: dict[str, float], components: tuple[str, ...]) -> list[float]:
    composition = normalize(composition, components)
    reference = max(composition.get(components[-1], 0.0), 1e-14)
    return [
        math.log(max(composition.get(comp, 0.0), 1e-14) / reference)
        for comp in components[:-1]
    ]


def topology_text(active: list[bool]) -> str:
    return "".join("L" if value else "." for value in active)


def phase_distance(x1: dict[str, float], x2: dict[str, float], components) -> float:
    return sum(abs(x1.get(comp, 0.0) - x2.get(comp, 0.0)) for comp in components)


class StabilityCache:
    """Exact-key cache for compiled liquid-stability/LLE checks."""

    def __init__(self, thermo, components: tuple[str, ...], tol: float = 1e-7):
        self.thermo = thermo
        self.components = components
        self.tol = tol
        self.calls = 0

        @lru_cache(maxsize=200000)
        def cached(T: float, values: tuple[float, ...]):
            self.calls += 1
            composition = {
                comp: float(values[index])
                for index, comp in enumerate(self.components)
            }
            return self.thermo.liquid_liquid_equilibrium(
                composition,
                float(T),
                max_iter=100,
                tol=self.tol,
            )

        self._cached = cached

    def split(self, T: float, composition: dict[str, float]):
        composition = normalize(composition, self.components)
        values = tuple(float(composition[comp]) for comp in self.components)
        has_lle, x1, x2, beta = self._cached(float(T), values)
        return (
            bool(has_lle),
            normalize(x1, self.components),
            normalize(x2, self.components),
            float(beta),
        )

    @property
    def hits(self) -> int:
        return int(self._cached.cache_info().hits)


def unit_params(case: ColumnCase) -> dict:
    return {
        "N_stages": case.stages,
        "feed_stage": case.feed_stage,
        "condenser_type": "total",
        "reflux_ratio": case.reflux_ratio,
        "D_to_F": case.distillate_fraction,
        "P_condenser": case.feed_P,
        "P_drop_per_stage": case.pressure_drop_per_stage,
        "initializer": "estimate",
        "mesh_tolerance": case.mesh_tolerance,
        "max_iterations": case.max_iterations,
        "max_jacobian_evaluations": case.max_iterations,
    }


def make_thermo_and_feed(case: ColumnCase):
    thermo = create_thermodynamics(list(case.components), case.method)
    feed = thermo.calculate_state(
        case.feed_T,
        case.feed_P,
        case.feed_F,
        case.feed_z,
        phase="liquid",
        flash=False,
    )
    # Exclude one-time Numba compilation from formulation timings.
    thermo.activity_coefficients(case.feed_T, case.feed_z)
    thermo.liquid_liquid_equilibrium(case.feed_z, case.feed_T, max_iter=100, tol=1e-7)
    return thermo, feed


def profile_from_unit_result(result, stages: int) -> ColumnProfile:
    performance = result.performance
    return ColumnProfile(
        T=[float(value) + 273.15 for value in performance["stage_temperatures_C"]],
        aggregate_x=[dict(value) for value in performance["stage_liquid_compositions"]],
        L=[float(value) for value in performance["liquid_flows"]],
        V=[float(value) for value in performance["vapor_flows"]],
        Q_cond=float(performance["condenser_duty_kW"]) * 3600.0,
        Q_reb=float(performance["reboiler_duty_kW"]) * 3600.0,
        split_data=[None] * stages,
    )


def cheap_initial_profile(case: ColumnCase, thermo, feed) -> ColumnProfile:
    """Build the production cheap-estimate profile without solving MESH."""
    helper = RigorousDistillation(
        f"{case.name}_cheap_seed",
        thermo,
        unit_params(case),
    )
    inlet, _feed_specs = helper._aggregate_feeds(
        {"feed": feed},
        case.stages,
        case.feed_stage,
    )
    components = helper._component_order(inlet)
    feed_z = helper._normalize({
        comp: inlet.composition.get(comp, 0.0)
        for comp in components
    })
    pressures = helper._pressure_profile(case.stages, inlet.P)
    T_min, T_max = helper._temperature_bounds(components)
    q_feed = helper._feed_thermal_condition(
        inlet,
        feed_z,
        pressures[case.feed_stage - 1],
        T_min,
        T_max,
    )
    distillate_spec = helper._distillate_spec(inlet)
    initial = helper._initial_guess(
        inlet,
        components,
        feed_z,
        case.stages,
        case.feed_stage,
        case.reflux_ratio,
        q_feed,
        pressures,
        "total",
        0.0,
        distillate_spec,
        [],
        T_min,
        T_max,
    )
    return ColumnProfile(
        T=[float(value) for value in initial["T"]],
        aggregate_x=[dict(value) for value in initial["x"]],
        L=[float(value) for value in initial["L"]],
        V=[float(value) for value in initial["V"]],
        Q_cond=float(initial["Q_cond"]),
        Q_reb=float(initial["Q_reb"]),
        split_data=[None] * case.stages,
    )


def coarse_initial_profile(case: ColumnCase, thermo, feed) -> ColumnProfile:
    """Solve a small homogeneous column and interpolate it to the full grid."""
    coarse_stages = max(4, int(round(case.stages / 3.0)))
    coarse_stages = min(coarse_stages, case.stages)
    coarse_feed_stage = 1 + round(
        (case.feed_stage - 1)
        * (coarse_stages - 1)
        / max(case.stages - 1, 1)
    )
    params = unit_params(case)
    params.update({
        "N_stages": coarse_stages,
        "feed_stage": coarse_feed_stage,
        "initializer": "estimate",
        "mesh_tolerance": max(case.mesh_tolerance, 1e-4),
        "max_iterations": min(case.max_iterations, 50),
        "max_jacobian_evaluations": min(case.max_iterations, 50),
    })
    result = RigorousDistillation(
        f"{case.name}_coarse_seed",
        thermo,
        params,
    ).solve({"feed": feed})
    coarse = profile_from_unit_result(result, coarse_stages)
    coarse_grid = np.linspace(0.0, 1.0, coarse_stages)
    full_grid = np.linspace(0.0, 1.0, case.stages)
    aggregate_x = []
    for position in full_grid:
        composition = {
            comp: float(np.interp(
                position,
                coarse_grid,
                [stage.get(comp, 0.0) for stage in coarse.aggregate_x],
            ))
            for comp in case.components
        }
        aggregate_x.append(normalize(composition, case.components))
    return ColumnProfile(
        T=[
            float(np.interp(position, coarse_grid, coarse.T))
            for position in full_grid
        ],
        aggregate_x=aggregate_x,
        L=[
            max(float(np.interp(position, coarse_grid, coarse.L)), 1e-12)
            for position in full_grid
        ],
        V=[
            max(float(np.interp(position, coarse_grid, coarse.V)), 1e-12)
            for position in full_grid
        ],
        Q_cond=coarse.Q_cond,
        Q_reb=coarse.Q_reb,
        split_data=[None] * case.stages,
    )


def component_balance_error(case: ColumnCase, distillate, bottoms) -> float:
    scale = max(case.feed_F, 1.0)
    return max(
        abs(
            case.feed_F * case.feed_z.get(comp, 0.0)
            - case.feed_F * case.distillate_fraction * distillate.get(comp, 0.0)
            - case.feed_F * (1.0 - case.distillate_fraction) * bottoms.get(comp, 0.0)
        ) / scale
        for comp in case.components
    )


def run_baseline(case: ColumnCase):
    thermo, feed = make_thermo_and_feed(case)
    started = time.perf_counter()
    try:
        result = RigorousDistillation(
            f"{case.name}_baseline",
            thermo,
            unit_params(case),
        ).solve({"feed": feed})
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return None, RunSummary(
            formulation="homogeneous",
            success=False,
            elapsed_s=elapsed,
            message=f"{type(exc).__name__}: {exc}",
        )
    elapsed = time.perf_counter() - started
    profile = profile_from_unit_result(result, case.stages)
    cache = StabilityCache(thermo, case.components)
    active = [
        cache.split(T, x)[0]
        for T, x in zip(profile.T, profile.aggregate_x)
    ]
    performance = result.performance
    summary = RunSummary(
        formulation="homogeneous",
        success=True,
        elapsed_s=elapsed,
        residual=float(performance["mesh_residual"]),
        iterations=int(performance["solver_iterations"]),
        function_evaluations=int(performance["function_evaluations"]),
        jacobian_evaluations=int(performance["jacobian_evaluations"]),
        topology=topology_text(active),
        split_stages=sum(active),
        top_temperature_C=float(performance["T_top_C"]),
        bottom_temperature_C=float(performance["T_bottom_C"]),
        distillate=dict(result.outlet_streams["distillate"].composition),
        bottoms=dict(result.outlet_streams["bottoms"].composition),
        component_balance_error=float(performance["component_balance_error"]),
        stability_checks=cache.calls,
        stability_cache_hits=cache.hits,
        message="post-solve topology screen only",
    )
    return result, summary


def run_nested_switch(case: ColumnCase) -> RunSummary:
    thermo, feed = make_thermo_and_feed(case)
    cache = StabilityCache(thermo, case.components)
    original_K = thermo.K_values
    original_H = thermo.mixture_enthalpy

    def switched_K(T: float, P: float, composition: dict[str, float]):
        has_lle, x1, x2, _beta = cache.split(T, composition)
        if not has_lle:
            return original_K(T, P, composition)
        K1 = original_K(T, P, x1)
        K2 = original_K(T, P, x2)
        return {
            comp: math.sqrt(
                max(K1.get(comp, 1.0) * x1.get(comp, 0.0), 1e-300)
                * max(K2.get(comp, 1.0) * x2.get(comp, 0.0), 1e-300)
            ) / max(composition.get(comp, 0.0), 1e-300)
            for comp in case.components
        }

    def switched_H(
        composition,
        T,
        vapor_fraction=1.0,
        x=None,
        y=None,
        P=1.01325,
    ):
        if vapor_fraction > 1e-12:
            return original_H(composition, T, vapor_fraction, x, y, P)
        has_lle, x1, x2, beta = cache.split(T, composition)
        if not has_lle:
            return original_H(composition, T, 0.0, x, y, P)
        h1 = original_H(x1, T, 0.0, P=P)
        h2 = original_H(x2, T, 0.0, P=P)
        return (1.0 - beta) * h1 + beta * h2

    thermo.K_values = switched_K
    thermo.mixture_enthalpy = switched_H
    started = time.perf_counter()
    try:
        result = RigorousDistillation(
            f"{case.name}_nested",
            thermo,
            unit_params(case),
        ).solve({"feed": feed})
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return RunSummary(
            formulation="nested_switch",
            success=False,
            elapsed_s=elapsed,
            stability_checks=cache.calls,
            stability_cache_hits=cache.hits,
            message=f"{type(exc).__name__}: {exc}",
        )

    elapsed = time.perf_counter() - started
    performance = result.performance
    active = [
        cache.split(float(T_C) + 273.15, x)[0]
        for T_C, x in zip(
            performance["stage_temperatures_C"],
            performance["stage_liquid_compositions"],
        )
    ]
    return RunSummary(
        formulation="nested_switch",
        success=True,
        elapsed_s=elapsed,
        residual=float(performance["mesh_residual"]),
        iterations=int(performance["solver_iterations"]),
        function_evaluations=int(performance["function_evaluations"]),
        jacobian_evaluations=int(performance["jacobian_evaluations"]),
        topology=topology_text(active),
        split_stages=sum(active),
        top_temperature_C=float(performance["T_top_C"]),
        bottom_temperature_C=float(performance["T_bottom_C"]),
        distillate=dict(result.outlet_streams["distillate"].composition),
        bottoms=dict(result.outlet_streams["bottoms"].composition),
        component_balance_error=float(performance["component_balance_error"]),
        stability_checks=cache.calls,
        stability_cache_hits=cache.hits,
    )


class EquationOrientedColumn:
    """Fixed-topology VLE/VLLE MESH model for the probing script."""

    def __init__(
        self,
        case: ColumnCase,
        thermo,
        feed,
        active: list[bool],
        profile: ColumnProfile,
        stability: StabilityCache,
        jacobian_mode: str,
        phase_fraction_min: float,
        phase_distance_min: float,
        topology_change_residual: float,
    ):
        self.case = case
        self.thermo = thermo
        self.feed = feed
        self.components = case.components
        self.nc = len(case.components)
        self.active = list(active)
        self.profile = profile
        self.stability = stability
        self.jacobian_mode = jacobian_mode
        self.phase_fraction_min = phase_fraction_min
        self.phase_distance_min = phase_distance_min
        self.topology_change_residual = topology_change_residual
        self.helper = RigorousDistillation(
            f"{case.name}_equation_oriented",
            thermo,
            unit_params(case),
        )
        self.pressures = [
            case.feed_P + stage * case.pressure_drop_per_stage
            for stage in range(case.stages)
        ]
        self.T_min, self.T_max = self.helper._temperature_bounds(list(case.components))
        self.temperature_span = self.T_max - self.T_min
        self.flow_scale = max(case.feed_F, 1.0)
        self.energy_scale = max(
            abs(case.feed_F * float(feed.H or 0.0)),
            case.feed_F * 50000.0,
            1.0,
        )
        component_floor = 1e-4
        self.component_scales = {
            comp: max(
                case.feed_F * case.feed_z.get(comp, 0.0),
                self.flow_scale * component_floor,
                1e-12,
            )
            for comp in case.components
        }
        self.layouts, self.Q_cond_index, self.Q_reb_index, self.n_vars = self._layouts()
        self.stage_row_counts = [
            self.nc + 2 + (self.nc if is_vlle else 0)
            for is_vlle in self.active
        ]
        self.stage_row_starts = []
        row_cursor = 0
        for count in self.stage_row_counts:
            self.stage_row_starts.append(row_cursor)
            row_cursor += count
        self.n_rows = row_cursor + 2

    def _layouts(self):
        layouts = []
        cursor = 0
        composition_size = self.nc - 1
        for is_vlle in self.active:
            start = cursor
            temperature = cursor
            cursor += 1
            x1 = slice(cursor, cursor + composition_size)
            cursor += composition_size
            x2 = None
            beta = None
            if is_vlle:
                x2 = slice(cursor, cursor + composition_size)
                cursor += composition_size
                beta = cursor
                cursor += 1
            liquid_flow = cursor
            vapor_flow = cursor + 1
            cursor += 2
            layouts.append(StageLayout(
                start=start,
                stop=cursor,
                temperature=temperature,
                x1=x1,
                x2=x2,
                beta=beta,
                liquid_flow=liquid_flow,
                vapor_flow=vapor_flow,
                active_vlle=is_vlle,
            ))
        Q_cond_index = cursor
        Q_reb_index = cursor + 1
        return layouts, Q_cond_index, Q_reb_index, cursor + 2

    def _initial_split(self, stage: int):
        previous = self.profile.split_data[stage]
        if previous is not None:
            return previous
        has_lle, x1, x2, beta = self.stability.split(
            self.profile.T[stage],
            self.profile.aggregate_x[stage],
        )
        if not has_lle:
            # Deliberately overactivated stages need a finite-distance ghost
            # pair so the equation-oriented solve can collapse it to the
            # trivial x1 == x2 solution and let the outer active set remove it.
            aggregate = normalize(
                self.profile.aggregate_x[stage],
                self.components,
            )
            ranked = sorted(
                self.components,
                key=lambda comp: aggregate[comp],
                reverse=True,
            )
            comp_a, comp_b = ranked[:2]
            delta = min(
                0.02,
                0.25 * aggregate[comp_a],
                0.25 * aggregate[comp_b],
            )
            if delta <= 1e-12:
                raise RuntimeError(
                    f"stage {stage + 1} cannot construct a ghost-liquid initializer"
                )
            x1 = dict(aggregate)
            x2 = dict(aggregate)
            x1[comp_a] += delta
            x1[comp_b] -= delta
            x2[comp_a] -= delta
            x2[comp_b] += delta
            return x1, x2, 0.5
        return x1, x2, beta

    def pack_initial(self) -> np.ndarray:
        vector = np.zeros(self.n_vars, dtype=float)
        for stage, layout in enumerate(self.layouts):
            scaled_T = (
                self.profile.T[stage] - self.T_min
            ) / self.temperature_span
            vector[layout.temperature] = logit(scaled_T)
            if layout.active_vlle:
                x1, x2, beta = self._initial_split(stage)
                vector[layout.x1] = composition_logits(x1, self.components)
                vector[layout.x2] = composition_logits(x2, self.components)
                vector[layout.beta] = logit(beta)
            else:
                vector[layout.x1] = composition_logits(
                    self.profile.aggregate_x[stage],
                    self.components,
                )
            vector[layout.liquid_flow] = math.log(max(self.profile.L[stage], 1e-12))
            vector[layout.vapor_flow] = math.log(max(self.profile.V[stage], 1e-12))
        vector[self.Q_cond_index] = self.profile.Q_cond / self.energy_scale
        vector[self.Q_reb_index] = self.profile.Q_reb / self.energy_scale
        return vector

    def decode(self, vector) -> dict:
        stages = [
            self.decode_stage(vector, stage)
            for stage in range(self.case.stages)
        ]
        return {
            "stages": stages,
            "Q_cond": float(vector[self.Q_cond_index] * self.energy_scale),
            "Q_reb": float(vector[self.Q_reb_index] * self.energy_scale),
        }

    def decode_stage(self, vector, stage: int) -> dict:
        layout = self.layouts[stage]
        T = self.T_min + self.temperature_span * sigmoid(vector[layout.temperature])
        x1 = composition_from_logits(vector[layout.x1], self.components)
        if layout.active_vlle:
            x2 = composition_from_logits(vector[layout.x2], self.components)
            beta = sigmoid(vector[layout.beta])
            aggregate = {
                comp: (1.0 - beta) * x1[comp] + beta * x2[comp]
                for comp in self.components
            }
        else:
            x2 = dict(x1)
            beta = 0.0
            aggregate = dict(x1)
        return {
            "T": float(T),
            "x1": x1,
            "x2": x2,
            "beta": float(beta),
            "aggregate_x": aggregate,
            "L": float(math.exp(min(max(vector[layout.liquid_flow], -40.0), 40.0))),
            "V": float(math.exp(min(max(vector[layout.vapor_flow], -40.0), 40.0))),
        }

    def stage_properties(self, stage: int, state: dict) -> dict:
        T = state["T"]
        P = self.pressures[stage]
        if not self.active[stage]:
            K = self.thermo.K_values(T, P, state["x1"])
            vapor_terms = {
                comp: max(K.get(comp, 1.0) * state["x1"].get(comp, 0.0), 1e-300)
                for comp in self.components
            }
            lle_residuals = []
            h_liquid = self.thermo.mixture_enthalpy(
                state["x1"], T, vapor_fraction=0.0, P=P
            )
        else:
            gamma1 = self.thermo.activity_coefficients(T, state["x1"])
            gamma2 = self.thermo.activity_coefficients(T, state["x2"])
            activities1 = {
                comp: max(state["x1"][comp] * gamma1.get(comp, 1.0), 1e-300)
                for comp in self.components
            }
            activities2 = {
                comp: max(state["x2"][comp] * gamma2.get(comp, 1.0), 1e-300)
                for comp in self.components
            }
            lle_residuals = [
                math.log(activities1[comp]) - math.log(activities2[comp])
                for comp in self.components
            ]
            vapor_terms = shared_vlle_vapor_terms(
                self.thermo,
                T,
                P,
                state["x1"],
                state["x2"],
                self.components,
                gamma1,
                gamma2,
            )
            h1 = self.thermo.mixture_enthalpy(
                state["x1"], T, vapor_fraction=0.0, P=P
            )
            h2 = self.thermo.mixture_enthalpy(
                state["x2"], T, vapor_fraction=0.0, P=P
            )
            h_liquid = (1.0 - state["beta"]) * h1 + state["beta"] * h2

        vapor_total = sum(vapor_terms.values())
        y = {
            comp: vapor_terms[comp] / max(vapor_total, 1e-300)
            for comp in self.components
        }
        h_vapor = self.thermo.mixture_enthalpy(
            y, T, vapor_fraction=1.0, P=P
        )
        return {
            "y": y,
            "bubble": vapor_total - 1.0,
            "lle": lle_residuals,
            "hL": float(h_liquid),
            "hV": float(h_vapor),
        }

    def residual(self, vector) -> np.ndarray:
        decoded = self.decode(vector)
        stages = decoded["stages"]
        props = [
            self.stage_properties(stage, state)
            for stage, state in enumerate(stages)
        ]
        values = []
        feed_index = self.case.feed_stage - 1
        D = stages[0]["V"]
        for stage, state in enumerate(stages):
            for comp in self.components:
                incoming = (
                    self.case.feed_F * self.case.feed_z.get(comp, 0.0)
                    if stage == feed_index else 0.0
                )
                if stage > 0:
                    incoming += (
                        stages[stage - 1]["L"]
                        * stages[stage - 1]["aggregate_x"].get(comp, 0.0)
                    )
                if stage < self.case.stages - 1:
                    incoming += (
                        stages[stage + 1]["V"]
                        * props[stage + 1]["y"].get(comp, 0.0)
                    )

                if stage == 0:
                    outgoing = (
                        state["L"] + D
                    ) * state["aggregate_x"].get(comp, 0.0)
                else:
                    outgoing = (
                        state["L"] * state["aggregate_x"].get(comp, 0.0)
                        + state["V"] * props[stage]["y"].get(comp, 0.0)
                    )
                values.append(
                    (incoming - outgoing) / self.component_scales[comp]
                )

            incoming_energy = 0.0
            if stage > 0:
                incoming_energy += stages[stage - 1]["L"] * props[stage - 1]["hL"]
            if stage < self.case.stages - 1:
                incoming_energy += stages[stage + 1]["V"] * props[stage + 1]["hV"]
            if stage == feed_index:
                incoming_energy += self.case.feed_F * float(self.feed.H)
            if stage == 0:
                incoming_energy += decoded["Q_cond"]
                outgoing_energy = (state["L"] + D) * props[stage]["hL"]
            else:
                if stage == self.case.stages - 1:
                    incoming_energy += decoded["Q_reb"]
                outgoing_energy = (
                    state["L"] * props[stage]["hL"]
                    + state["V"] * props[stage]["hV"]
                )
            values.append((incoming_energy - outgoing_energy) / self.energy_scale)
            values.extend(props[stage]["lle"])
            values.append(props[stage]["bubble"])

        specified_D = self.case.feed_F * self.case.distillate_fraction
        values.append((D - specified_D) / self.flow_scale)
        values.append(
            (stages[0]["L"] - self.case.reflux_ratio * D) / self.flow_scale
        )
        return np.asarray(values, dtype=float)

    def sparsity(self):
        matrix = lil_matrix((self.n_rows, self.n_vars), dtype=int)

        for stage, count in enumerate(self.stage_row_counts):
            rows = range(
                self.stage_row_starts[stage],
                self.stage_row_starts[stage] + count,
            )
            dependent_stages = [stage]
            if stage > 0:
                dependent_stages.append(stage - 1)
            if stage < self.case.stages - 1:
                dependent_stages.append(stage + 1)
            for row in rows:
                for dependent in dependent_stages:
                    layout = self.layouts[dependent]
                    matrix[row, layout.start:layout.stop] = 1
                if stage == 0:
                    matrix[row, self.Q_cond_index] = 1
                if stage == self.case.stages - 1:
                    matrix[row, self.Q_reb_index] = 1

        spec_row = self.n_rows - 2
        top = self.layouts[0]
        matrix[spec_row, top.vapor_flow] = 1
        matrix[spec_row + 1, top.liquid_flow] = 1
        matrix[spec_row + 1, top.vapor_flow] = 1
        return matrix.tocsr()

    def topology_for_decoded(self, decoded: dict) -> list[bool]:
        topology = []
        for stage, state in enumerate(decoded["stages"]):
            stable_split, _x1, _x2, _beta = self.stability.split(
                state["T"],
                state["aggregate_x"],
            )
            if self.active[stage]:
                topology.append(
                    stable_split
                    and self.phase_fraction_min
                    < state["beta"]
                    < 1.0 - self.phase_fraction_min
                    and phase_distance(
                        state["x1"],
                        state["x2"],
                        self.components,
                    ) > self.phase_distance_min
                )
            else:
                topology.append(stable_split)
        return topology

    def semi_analytic_jacobian(self, vector, _f0, rel_step: float):
        """Stage-local thermodynamic FD plus analytic MESH flow derivatives."""
        from scipy.sparse import lil_matrix

        decoded = self.decode(vector)
        updated_active = self.topology_for_decoded(decoded)
        residual_norm = float(np.linalg.norm(_f0, ord=np.inf))
        if (
            updated_active != self.active
            and residual_norm <= self.topology_change_residual
        ):
            raise ActiveSetChange(
                self.profile_from_decoded(decoded),
                updated_active,
                "stage stability changed at MESH residual "
                f"{residual_norm:.3e} before the next local Jacobian",
            )
        stages = decoded["stages"]
        props = [
            self.stage_properties(stage, state)
            for stage, state in enumerate(stages)
        ]
        matrix = lil_matrix((self.n_rows, self.n_vars), dtype=float)
        evaluations = 0

        def add(row: int, column: int, value: float) -> None:
            if value:
                matrix[row, column] = matrix[row, column] + value

        for stage, layout in enumerate(self.layouts):
            local_columns = [layout.temperature]
            local_columns.extend(range(layout.x1.start, layout.x1.stop))
            if layout.active_vlle:
                local_columns.extend(range(layout.x2.start, layout.x2.stop))
                local_columns.append(layout.beta)

            state = stages[stage]
            stage_props = props[stage]
            for column in local_columns:
                step = rel_step * max(abs(float(vector[column])), 1.0)
                trial = np.array(vector, dtype=float, copy=True)
                trial[column] += step
                perturbed_state = self.decode_stage(trial, stage)
                perturbed_props = self.stage_properties(
                    stage,
                    perturbed_state,
                )
                evaluations += 1

                dx = {
                    comp: (
                        perturbed_state["aggregate_x"].get(comp, 0.0)
                        - state["aggregate_x"].get(comp, 0.0)
                    ) / step
                    for comp in self.components
                }
                dy = {
                    comp: (
                        perturbed_props["y"].get(comp, 0.0)
                        - stage_props["y"].get(comp, 0.0)
                    ) / step
                    for comp in self.components
                }
                dhL = (perturbed_props["hL"] - stage_props["hL"]) / step
                dhV = (perturbed_props["hV"] - stage_props["hV"]) / step

                liquid_out_coefficient = (
                    state["L"] + state["V"]
                    if stage == 0 else state["L"]
                )
                for component_index, comp in enumerate(self.components):
                    scale = self.component_scales[comp]
                    local_row = self.stage_row_starts[stage] + component_index
                    add(
                        local_row,
                        column,
                        -(
                            liquid_out_coefficient * dx[comp]
                            + (state["V"] * dy[comp] if stage > 0 else 0.0)
                        ) / scale,
                    )
                    if stage < self.case.stages - 1:
                        next_row = self.stage_row_starts[stage + 1] + component_index
                        add(
                            next_row,
                            column,
                            state["L"] * dx[comp] / scale,
                        )
                    if stage > 0:
                        previous_row = self.stage_row_starts[stage - 1] + component_index
                        add(
                            previous_row,
                            column,
                            state["V"] * dy[comp] / scale,
                        )

                energy_row = self.stage_row_starts[stage] + self.nc
                add(
                    energy_row,
                    column,
                    -(
                        liquid_out_coefficient * dhL
                        + (state["V"] * dhV if stage > 0 else 0.0)
                    ) / self.energy_scale,
                )
                if stage < self.case.stages - 1:
                    add(
                        self.stage_row_starts[stage + 1] + self.nc,
                        column,
                        state["L"] * dhL / self.energy_scale,
                    )
                if stage > 0:
                    add(
                        self.stage_row_starts[stage - 1] + self.nc,
                        column,
                        state["V"] * dhV / self.energy_scale,
                    )

                equilibrium_row = energy_row + 1
                for index, base_value in enumerate(stage_props["lle"]):
                    derivative = (
                        perturbed_props["lle"][index] - base_value
                    ) / step
                    add(equilibrium_row + index, column, derivative)
                bubble_row = equilibrium_row + len(stage_props["lle"])
                add(
                    bubble_row,
                    column,
                    (perturbed_props["bubble"] - stage_props["bubble"]) / step,
                )

        for stage, layout in enumerate(self.layouts):
            state = stages[stage]
            stage_props = props[stage]
            liquid_column = layout.liquid_flow
            vapor_column = layout.vapor_flow
            liquid_derivative = state["L"]
            vapor_derivative = state["V"]

            for component_index, comp in enumerate(self.components):
                scale = self.component_scales[comp]
                local_row = self.stage_row_starts[stage] + component_index
                add(
                    local_row,
                    liquid_column,
                    -liquid_derivative * state["aggregate_x"].get(comp, 0.0) / scale,
                )
                if stage < self.case.stages - 1:
                    add(
                        self.stage_row_starts[stage + 1] + component_index,
                        liquid_column,
                        liquid_derivative * state["aggregate_x"].get(comp, 0.0) / scale,
                    )

                if stage == 0:
                    add(
                        local_row,
                        vapor_column,
                        -vapor_derivative * state["aggregate_x"].get(comp, 0.0) / scale,
                    )
                else:
                    add(
                        local_row,
                        vapor_column,
                        -vapor_derivative * stage_props["y"].get(comp, 0.0) / scale,
                    )
                    add(
                        self.stage_row_starts[stage - 1] + component_index,
                        vapor_column,
                        vapor_derivative * stage_props["y"].get(comp, 0.0) / scale,
                    )

            energy_row = self.stage_row_starts[stage] + self.nc
            add(
                energy_row,
                liquid_column,
                -liquid_derivative * stage_props["hL"] / self.energy_scale,
            )
            if stage < self.case.stages - 1:
                add(
                    self.stage_row_starts[stage + 1] + self.nc,
                    liquid_column,
                    liquid_derivative * stage_props["hL"] / self.energy_scale,
                )
            if stage == 0:
                add(
                    energy_row,
                    vapor_column,
                    -vapor_derivative * stage_props["hL"] / self.energy_scale,
                )
            else:
                add(
                    energy_row,
                    vapor_column,
                    -vapor_derivative * stage_props["hV"] / self.energy_scale,
                )
                add(
                    self.stage_row_starts[stage - 1] + self.nc,
                    vapor_column,
                    vapor_derivative * stage_props["hV"] / self.energy_scale,
                )

        add(self.stage_row_starts[0] + self.nc, self.Q_cond_index, 1.0)
        add(
            self.stage_row_starts[-1] + self.nc,
            self.Q_reb_index,
            1.0,
        )
        spec_row = self.n_rows - 2
        top = stages[0]
        top_layout = self.layouts[0]
        add(spec_row, top_layout.vapor_flow, top["V"] / self.flow_scale)
        add(
            spec_row + 1,
            top_layout.liquid_flow,
            top["L"] / self.flow_scale,
        )
        add(
            spec_row + 1,
            top_layout.vapor_flow,
            -self.case.reflux_ratio * top["V"] / self.flow_scale,
        )
        return matrix.tocsr(), evaluations, "semi_analytic_local_thermo"

    def solve(self) -> tuple[dict, dict]:
        x0 = self.pack_initial()
        options = {
            "mesh_tolerance": self.case.mesh_tolerance,
            "acceptable_mesh_residual": max(
                50.0 * self.case.mesh_tolerance,
                self.case.mesh_tolerance,
            ),
            "max_iterations": self.case.max_iterations,
            "max_jacobian_evaluations": self.case.max_iterations,
            "line_search_steps": 16,
            "finite_difference_rel_step": 1e-6,
        }
        solution = self.helper._sparse_newton_solve(
            self.residual,
            self.sparsity(),
            x0,
            options,
            jacobian=(
                self.semi_analytic_jacobian
                if self.jacobian_mode == "local" else None
            ),
        )
        if not solution["success"]:
            raise RuntimeError(
                f"EO MESH failed (residual {solution['residual_norm']:.3e}): "
                f"{solution['message']}"
            )
        decoded = self.decode(solution["x"])
        final_residual = self.residual(solution["x"])
        solution["residual_norm"] = float(np.linalg.norm(final_residual, ord=np.inf))
        return decoded, solution

    def profile_from_decoded(self, decoded: dict) -> ColumnProfile:
        stages = decoded["stages"]
        split_data = []
        for stage, state in enumerate(stages):
            if self.active[stage]:
                split_data.append((dict(state["x1"]), dict(state["x2"]), state["beta"]))
            else:
                split_data.append(None)
        return ColumnProfile(
            T=[state["T"] for state in stages],
            aggregate_x=[dict(state["aggregate_x"]) for state in stages],
            L=[state["L"] for state in stages],
            V=[state["V"] for state in stages],
            Q_cond=decoded["Q_cond"],
            Q_reb=decoded["Q_reb"],
            split_data=split_data,
        )


def run_equation_oriented(
    case: ColumnCase,
    baseline_result,
    baseline_elapsed_s: float,
    max_outer: int,
    phase_fraction_min: float,
    phase_distance_min: float,
    jacobian_mode: str,
    seed_mode: str,
    topology_change_residual: float,
) -> RunSummary:
    formulation = f"equation_oriented_{jacobian_mode}_{seed_mode}"
    thermo, feed = make_thermo_and_feed(case)
    stability = StabilityCache(thermo, case.components)
    seed_started = time.perf_counter()
    actual_seed_mode = seed_mode
    if seed_mode == "auto":
        profile = cheap_initial_profile(case, thermo, feed)
        cheap_active = [
            stability.split(T, x)[0]
            for T, x in zip(profile.T, profile.aggregate_x)
        ]
        if any(cheap_active):
            actual_seed_mode = "cheap"
            seed_elapsed_s = time.perf_counter() - seed_started
        else:
            actual_seed_mode = "homogeneous"
            if baseline_result is not None:
                profile = profile_from_unit_result(
                    baseline_result,
                    case.stages,
                )
                seed_elapsed_s = baseline_elapsed_s
            else:
                homogeneous = RigorousDistillation(
                    f"{case.name}_homogeneous_seed",
                    thermo,
                    unit_params(case),
                ).solve({"feed": feed})
                profile = profile_from_unit_result(
                    homogeneous,
                    case.stages,
                )
                seed_elapsed_s = time.perf_counter() - seed_started
    elif seed_mode == "cheap":
        profile = cheap_initial_profile(case, thermo, feed)
        seed_elapsed_s = time.perf_counter() - seed_started
    elif seed_mode == "coarse":
        profile = coarse_initial_profile(case, thermo, feed)
        seed_elapsed_s = time.perf_counter() - seed_started
    elif seed_mode == "homogeneous":
        if baseline_result is not None:
            profile = profile_from_unit_result(baseline_result, case.stages)
            seed_elapsed_s = baseline_elapsed_s
        else:
            homogeneous = RigorousDistillation(
                f"{case.name}_homogeneous_seed",
                thermo,
                unit_params(case),
            ).solve({"feed": feed})
            profile = profile_from_unit_result(homogeneous, case.stages)
            seed_elapsed_s = time.perf_counter() - seed_started
    else:
        return RunSummary(
            formulation=formulation,
            success=False,
            elapsed_s=0.0,
            message=f"unknown EO seed mode {seed_mode!r}",
        )
    formulation = f"equation_oriented_{jacobian_mode}_{actual_seed_mode}"
    started = time.perf_counter()
    screened_active = [
        stability.split(T, x)[0]
        for T, x in zip(profile.T, profile.aggregate_x)
    ]
    active = list(screened_active)
    if case.initial_topology == "all_vlle":
        active = [True] * case.stages
    elif case.initial_topology == "all_vle":
        active = [False] * case.stages
    elif case.initial_topology != "screened":
        return RunSummary(
            formulation=formulation,
            success=False,
            elapsed_s=0.0,
            seed_elapsed_s=seed_elapsed_s,
            end_to_end_elapsed_s=seed_elapsed_s,
            message=f"unknown initial topology {case.initial_topology!r}",
        )
    topology_history = [topology_text(active)]
    # A fixed-topology Newton system cannot carry an absent second liquid:
    # x1 == x2 makes its phase fraction indeterminate and the Jacobian
    # singular.  Remove such phases using the seed-profile stability screen
    # before equation assembly.  Newly unstable stages can still be added by
    # the post-solve active-set pass below.
    if case.initial_topology == "all_vlle" and active != screened_active:
        active = list(screened_active)
        topology_history.append(topology_text(active))
    total_iterations = 0
    total_functions = 0
    total_jacobians = 0
    outer_solves = 0
    decoded = None
    solution = None

    try:
        for outer in range(1, max_outer + 1):
            outer_solves += 1
            model = EquationOrientedColumn(
                case,
                thermo,
                feed,
                active,
                profile,
                stability,
                jacobian_mode,
                phase_fraction_min,
                phase_distance_min,
                topology_change_residual,
            )
            try:
                decoded, solution = model.solve()
            except ActiveSetChange as change:
                profile = change.profile
                active = list(change.active)
                for stage, is_active in enumerate(active):
                    if not is_active:
                        profile.split_data[stage] = None
                topology_history.append(topology_text(active))
                continue
            total_iterations += int(solution["iterations"])
            total_functions += int(solution["function_evaluations"])
            total_jacobians += int(solution["jacobian_evaluations"])
            profile = model.profile_from_decoded(decoded)

            new_active = model.topology_for_decoded(decoded)

            topology_history.append(topology_text(new_active))
            if new_active == active:
                break
            active = new_active
        else:
            raise RuntimeError(
                f"active set did not stabilize in {max_outer} outer iterations"
            )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return RunSummary(
            formulation=formulation,
            success=False,
            elapsed_s=elapsed,
            seed_elapsed_s=seed_elapsed_s,
            end_to_end_elapsed_s=seed_elapsed_s + elapsed,
            iterations=total_iterations,
            function_evaluations=total_functions,
            jacobian_evaluations=total_jacobians,
            topology=topology_history[-1],
            split_stages=topology_history[-1].count("L"),
            outer_iterations=outer_solves,
            stability_checks=stability.calls,
            stability_cache_hits=stability.hits,
            message=f"{type(exc).__name__}: {exc}; topology history={topology_history}",
        )

    elapsed = time.perf_counter() - started
    stages = decoded["stages"]
    D = stages[0]["V"]
    B = stages[-1]["L"]
    distillate = dict(stages[0]["aggregate_x"])
    bottoms = dict(stages[-1]["aggregate_x"])
    balance = max(
        abs(
            case.feed_F * case.feed_z.get(comp, 0.0)
            - D * distillate.get(comp, 0.0)
            - B * bottoms.get(comp, 0.0)
        ) / max(case.feed_F, 1.0)
        for comp in case.components
    )
    return RunSummary(
        formulation=formulation,
        success=True,
        elapsed_s=elapsed,
        seed_elapsed_s=seed_elapsed_s,
        end_to_end_elapsed_s=seed_elapsed_s + elapsed,
        residual=float(solution["residual_norm"]),
        iterations=total_iterations,
        function_evaluations=total_functions,
        jacobian_evaluations=total_jacobians,
        topology=topology_text(active),
        split_stages=sum(active),
        outer_iterations=outer_solves,
        top_temperature_C=float(stages[0]["T"] - 273.15),
        bottom_temperature_C=float(stages[-1]["T"] - 273.15),
        distillate=distillate,
        bottoms=bottoms,
        component_balance_error=float(balance),
        stability_checks=stability.calls,
        stability_cache_hits=stability.hits,
        message=f"topology history={topology_history}",
    )


def format_composition(composition: Optional[dict[str, float]]) -> str:
    if not composition:
        return "-"
    return ", ".join(f"{comp}={value:.5f}" for comp, value in composition.items())


def print_summary(summary: RunSummary) -> None:
    status = "OK" if summary.success else "ERROR"
    print(
        f"  {summary.formulation:<20} {status:<5} "
        f"time={summary.elapsed_s:8.4f}s "
        f"res={summary.residual if summary.residual is not None else float('nan'):.3e} "
        f"iters={summary.iterations if summary.iterations is not None else 0:3d} "
        f"outer={summary.outer_iterations:2d} "
        f"splits={summary.split_stages:2d} topology={summary.topology or '-'}"
    )
    if summary.seed_elapsed_s > 0.0:
        print(
            f"    seed={summary.seed_elapsed_s:.4f}s "
            f"end-to-end={summary.end_to_end_elapsed_s:.4f}s"
        )
    if summary.success:
        print(
            f"    Ttop={summary.top_temperature_C:.5f} C "
            f"Tbottom={summary.bottom_temperature_C:.5f} C "
            f"balance={summary.component_balance_error:.3e}"
        )
        print(f"    D: {format_composition(summary.distillate)}")
        print(f"    B: {format_composition(summary.bottoms)}")
    print(
        f"    stability checks={summary.stability_checks} "
        f"cache hits={summary.stability_cache_hits}"
    )
    if summary.message:
        print(f"    {summary.message}")


def selected_cases(names: list[str]) -> list[ColumnCase]:
    by_name = {case.name: case for case in CASES}
    if not names or "all" in names:
        return list(CASES)
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}")
    return [by_name[name] for name in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="named case to run; repeat for multiple cases (default: all)",
    )
    parser.add_argument(
        "--formulation",
        choices=("all", "homogeneous", "nested_switch", "equation_oriented"),
        default="all",
        help="formulation to report (default: all)",
    )
    parser.add_argument("--max-outer", type=int, default=8)
    parser.add_argument("--phase-fraction-min", type=float, default=1e-6)
    parser.add_argument("--phase-distance-min", type=float, default=1e-3)
    parser.add_argument(
        "--topology-change-residual",
        type=float,
        default=5e-2,
        help="largest MESH residual permitting an in-Newton topology rebuild",
    )
    parser.add_argument(
        "--eo-jacobian",
        choices=("local", "colored", "both"),
        default="local",
        help="equation-oriented Jacobian architecture (default: local)",
    )
    parser.add_argument(
        "--eo-seed",
        choices=("auto", "cheap", "coarse", "homogeneous", "all"),
        default="auto",
        help="equation-oriented initial profile (default: auto)",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_cases:
        for case in CASES:
            print(f"{case.name:<16} {case.description}")
        return 0
    if args.max_outer < 1:
        raise SystemExit("--max-outer must be positive")

    reports = []
    for case in selected_cases(args.case):
        print()
        print("=" * 100)
        print(f"{case.name}: {case.description}")
        print(
            f"method={case.method} stages={case.stages} feed_stage={case.feed_stage} "
            f"RR={case.reflux_ratio:g} D/F={case.distillate_fraction:g} z={case.feed_z}"
        )

        needs_baseline = (
            args.formulation in ("all", "homogeneous")
            or (
                args.formulation == "equation_oriented"
                and args.eo_seed in ("homogeneous", "all")
            )
        )
        if needs_baseline:
            baseline_result, baseline = run_baseline(case)
        else:
            baseline_result = None
            baseline = RunSummary(
                formulation="homogeneous",
                success=False,
                elapsed_s=0.0,
                message="not run",
            )
        summaries = []
        if args.formulation in ("all", "homogeneous"):
            summaries.append(baseline)
        if args.formulation in ("all", "nested_switch"):
            summaries.append(run_nested_switch(case))
        if args.formulation in ("all", "equation_oriented"):
            jacobian_modes = (
                ("local", "colored")
                if args.eo_jacobian == "both" else (args.eo_jacobian,)
            )
            seed_modes = (
                ("auto", "cheap", "coarse", "homogeneous")
                if args.eo_seed == "all" else (args.eo_seed,)
            )
            for seed_mode in seed_modes:
                for jacobian_mode in jacobian_modes:
                    summaries.append(run_equation_oriented(
                        case,
                        baseline_result,
                        baseline.elapsed_s,
                        args.max_outer,
                        args.phase_fraction_min,
                        args.phase_distance_min,
                        jacobian_mode,
                        seed_mode,
                        args.topology_change_residual,
                    ))
        for summary in summaries:
            print_summary(summary)
        reports.append({
            "case": asdict(case),
            "runs": [asdict(summary) for summary in summaries],
        })

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(reports, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_output}")

    failed = [
        run
        for report in reports
        for run in report["runs"]
        if not run["success"]
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
