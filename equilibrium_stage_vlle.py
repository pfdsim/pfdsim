"""Equation-oriented VLLE stages for rigorous distillation.

The model treats coexisting equilibrium liquids as co-routed phases with one
aggregate downward flow. Each active VLLE stage carries two liquid compositions
and a phase fraction; stable stages retain the smaller ordinary-VLE equation
set.  Stage topology is fixed within a Newton attempt and rebuilt only at
local-Jacobian boundaries or after convergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy.sparse import lil_matrix


@dataclass
class VLLEProfile:
    T: list[float]
    aggregate_x: list[dict[str, float]]
    L: list[float]
    V: list[float]
    Q_cond: float
    Q_reb: float
    split_data: list[Optional[tuple[dict[str, float], dict[str, float], float]]]


@dataclass(frozen=True)
class _StageLayout:
    start: int
    stop: int
    temperature: int
    x1: slice
    x2: Optional[slice]
    beta: Optional[int]
    liquid_flow: int
    vapor_flow: int
    active_vlle: bool


@dataclass
class VLLEColumnSolution:
    decoded: dict
    stage_properties: list[dict]
    solver: dict
    active: list[bool]
    topology_history: list[str]
    stability_checks: int
    stability_cache_hits: int
    outer_solves: int
    total_iterations: int
    total_function_evaluations: int
    total_jacobian_evaluations: int


class _ActiveSetChange(RuntimeError):
    def __init__(self, profile: VLLEProfile, active: list[bool]):
        super().__init__("VLLE stage topology changed")
        self.profile = profile
        self.active = list(active)


def _normalize(composition: dict[str, float], components) -> dict[str, float]:
    values = {
        comp: max(float(composition.get(comp, 0.0)), 0.0)
        for comp in components
    }
    total = sum(values.values())
    if total <= 0.0:
        return {comp: 1.0 / len(components) for comp in components}
    return {comp: value / total for comp, value in values.items()}


def _sigmoid(value: float) -> float:
    value = min(max(float(value), -60.0), 60.0)
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-12), 1.0 - 1e-12)
    return math.log(value / (1.0 - value))


def _composition_from_logits(values, components) -> dict[str, float]:
    logits = np.asarray(list(values) + [0.0], dtype=float)
    logits = np.clip(logits, -60.0, 60.0)
    logits -= np.max(logits)
    fractions = np.exp(logits)
    fractions /= np.sum(fractions)
    return {
        comp: float(fractions[index])
        for index, comp in enumerate(components)
    }


def _composition_logits(composition: dict[str, float], components) -> list[float]:
    composition = _normalize(composition, components)
    reference = max(composition.get(components[-1], 0.0), 1e-14)
    return [
        math.log(max(composition.get(comp, 0.0), 1e-14) / reference)
        for comp in components[:-1]
    ]


def topology_text(active: list[bool]) -> str:
    return "".join("L" if value else "." for value in active)


def shared_vlle_vapor_terms(
    thermo,
    T: float,
    P: float,
    x1: dict[str, float],
    x2: dict[str, float],
    components,
    gamma1: dict[str, float],
    gamma2: dict[str, float],
) -> dict[str, float]:
    """Return dimensionless vapor terms for one vapor shared by both liquids.

    Ideal-vapor activity models retain their ordinary K-value closure. For a
    gamma-phi model, form one symmetric liquid-fugacity target and iterate one
    shared vapor composition/EOS fugacity state. This makes the vapor used by
    the MESH equations common to L1 and L2 instead of geometrically averaging
    two independently iterated vapor states.
    """
    correction_active = getattr(thermo, "_vapor_phase_correction_active", None)
    uses_vapor_correction = bool(
        callable(correction_active) and correction_active()
    )
    vapor_phi = getattr(thermo, "vapor_fugacity_coefficients", None)
    reference_factors = getattr(
        thermo,
        "_liquid_fugacity_reference_factors",
        None,
    )
    if (
        not uses_vapor_correction
        or not callable(vapor_phi)
        or not callable(reference_factors)
    ):
        K1 = thermo.K_values(T, P, x1)
        K2 = thermo.K_values(T, P, x2)
        return {
            comp: math.sqrt(
                max(K1.get(comp, 1.0) * x1.get(comp, 0.0), 1e-300)
                * max(K2.get(comp, 1.0) * x2.get(comp, 0.0), 1e-300)
            )
            for comp in components
        }

    reference = reference_factors(T, P)
    target_fugacity = {
        comp: math.sqrt(
            max(
                x1.get(comp, 0.0)
                * gamma1.get(comp, 1.0)
                * reference.get(comp, 1.0),
                1e-300,
            )
            * max(
                x2.get(comp, 0.0)
                * gamma2.get(comp, 1.0)
                * reference.get(comp, 1.0),
                1e-300,
            )
        )
        for comp in components
    }
    y = _normalize(target_fugacity, components)
    vapor_terms = {
        comp: target_fugacity[comp] / max(P, 1e-300)
        for comp in components
    }
    for _ in range(12):
        try:
            phi_v = vapor_phi(T, P, y)
        except Exception:
            phi_v = {comp: 1.0 for comp in components}
        vapor_terms = {
            comp: max(
                target_fugacity[comp]
                / (max(float(phi_v.get(comp, 1.0)), 1e-8) * max(P, 1e-300)),
                1e-300,
            )
            for comp in components
        }
        y_new = _normalize(vapor_terms, components)
        change = max(
            abs(y_new[comp] - y.get(comp, 0.0)) for comp in components
        )
        y = y_new
        if change < 1e-10:
            break

    # Re-evaluate at the returned shared vapor composition so the bubble term
    # and post-solve fugacity audit use the same EOS state.
    try:
        phi_v = vapor_phi(T, P, y)
    except Exception:
        phi_v = {comp: 1.0 for comp in components}
    return {
        comp: max(
            target_fugacity[comp]
            / (max(float(phi_v.get(comp, 1.0)), 1e-8) * max(P, 1e-300)),
            1e-300,
        )
        for comp in components
    }


def three_phase_fugacity_residuals(
    thermo,
    T: float,
    P: float,
    x1: dict[str, float],
    x2: dict[str, float],
    y: dict[str, float],
    components,
) -> dict[str, float]:
    """Return maximum dimensionless log-fugacity residuals for one stage."""
    log_f1 = thermo._phase_log_fugacities(T, P, x1, "liquid")
    log_f2 = thermo._phase_log_fugacities(T, P, x2, "liquid")
    log_fv = thermo._phase_log_fugacities(T, P, y, "vapor")
    liquid_liquid = max(
        abs(log_f1[comp] - log_f2[comp]) for comp in components
    )
    vapor_liquid = max(
        max(
            abs(log_f1[comp] - log_fv[comp]),
            abs(log_f2[comp] - log_fv[comp]),
        )
        for comp in components
    )
    return {
        "liquid_liquid": float(liquid_liquid),
        "vapor_liquid": float(vapor_liquid),
        "overall": float(max(liquid_liquid, vapor_liquid)),
    }


class VLLEStabilityCache:
    """Exact-key cache around the model's compiled/readable LLE splitter."""

    def __init__(self, thermo, components, tol: float):
        self.thermo = thermo
        self.components = tuple(components)
        self.tol = float(tol)
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
        composition = _normalize(composition, self.components)
        values = tuple(float(composition[comp]) for comp in self.components)
        has_lle, x1, x2, beta = self._cached(float(T), values)
        return (
            bool(has_lle),
            _normalize(x1, self.components),
            _normalize(x2, self.components),
            float(beta),
        )

    @property
    def hits(self) -> int:
        return int(self._cached.cache_info().hits)


class EquationOrientedVLLEColumn:
    """One fixed VLE/VLLE stage topology and its sparse MESH equations."""

    def __init__(
        self,
        unit,
        inlet,
        feed_specs: list[dict],
        components: list[str],
        pressures: list[float],
        reflux_ratio: float,
        condenser_vapor_fraction: float,
        distillate_spec: dict,
        T_min: float,
        T_max: float,
        flow_scale: float,
        energy_scale: float,
        component_scales: dict[str, float],
        active: list[bool],
        profile: VLLEProfile,
        stability: VLLEStabilityCache,
        phase_fraction_min: float,
        phase_distance_min: float,
        topology_change_residual: float,
        solver_options: dict,
    ):
        self.unit = unit
        self.thermo = unit.thermo
        self.inlet = inlet
        self.feed_specs = feed_specs
        self.components = tuple(components)
        self.nc = len(self.components)
        self.N = len(pressures)
        self.pressures = [float(value) for value in pressures]
        self.reflux_ratio = float(reflux_ratio)
        self.condenser_vapor_fraction = float(condenser_vapor_fraction)
        self.distillate_spec = distillate_spec
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self.temperature_span = self.T_max - self.T_min
        self.flow_scale = float(flow_scale)
        self.energy_scale = float(energy_scale)
        self.component_scales = component_scales
        self.active = list(active)
        self.profile = profile
        self.stability = stability
        self.phase_fraction_min = float(phase_fraction_min)
        self.phase_distance_min = float(phase_distance_min)
        self.topology_change_residual = float(topology_change_residual)
        self.solver_options = solver_options
        self.layouts, self.Q_cond_index, self.Q_reb_index, self.n_vars = self._layouts()
        self.stage_row_counts = [
            self.nc + 2 + (self.nc if is_vlle else 0)
            for is_vlle in self.active
        ]
        self.stage_row_starts = []
        cursor = 0
        for count in self.stage_row_counts:
            self.stage_row_starts.append(cursor)
            cursor += count
        self.n_rows = cursor + 2
        self.stage_feeds = [[] for _ in range(self.N)]
        for feed in feed_specs:
            stage = int(feed["stage"])
            if 0 <= stage < self.N:
                self.stage_feeds[stage].append(feed)
        feed_mw = sum(
            inlet.composition.get(comp, 0.0) * self.thermo.props[comp].MW
            for comp in self.components
        )
        self.distillate_scale = (
            max(abs(float(distillate_spec["value"])), flow_scale * feed_mw, 1.0)
            if distillate_spec["kind"] == "mass" else flow_scale
        )

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
            layouts.append(_StageLayout(
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
        return layouts, cursor, cursor + 1, cursor + 2

    def _initial_split(self, stage: int):
        previous = self.profile.split_data[stage]
        if previous is not None:
            return previous
        has_lle, x1, x2, beta = self.stability.split(
            self.profile.T[stage],
            self.profile.aggregate_x[stage],
        )
        if not has_lle:
            raise RuntimeError(
                f"stage {stage + 1} is active VLLE but its initializer has no split"
            )
        return x1, x2, beta

    def pack_initial(self) -> np.ndarray:
        vector = np.zeros(self.n_vars, dtype=float)
        for stage, layout in enumerate(self.layouts):
            scaled_T = (
                self.profile.T[stage] - self.T_min
            ) / self.temperature_span
            vector[layout.temperature] = _logit(scaled_T)
            if layout.active_vlle:
                x1, x2, beta = self._initial_split(stage)
                vector[layout.x1] = _composition_logits(x1, self.components)
                vector[layout.x2] = _composition_logits(x2, self.components)
                vector[layout.beta] = _logit(beta)
            else:
                vector[layout.x1] = _composition_logits(
                    self.profile.aggregate_x[stage], self.components
                )
            vector[layout.liquid_flow] = math.log(max(self.profile.L[stage], 1e-12))
            vector[layout.vapor_flow] = math.log(max(self.profile.V[stage], 1e-12))
        vector[self.Q_cond_index] = self.profile.Q_cond / self.energy_scale
        vector[self.Q_reb_index] = self.profile.Q_reb / self.energy_scale
        return vector

    def decode_stage(self, vector, stage: int) -> dict:
        layout = self.layouts[stage]
        T = self.T_min + self.temperature_span * _sigmoid(vector[layout.temperature])
        x1 = _composition_from_logits(vector[layout.x1], self.components)
        if layout.active_vlle:
            x2 = _composition_from_logits(vector[layout.x2], self.components)
            beta = _sigmoid(vector[layout.beta])
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

    def decode(self, vector) -> dict:
        return {
            "stages": [self.decode_stage(vector, stage) for stage in range(self.N)],
            "Q_cond": float(vector[self.Q_cond_index] * self.energy_scale),
            "Q_reb": float(vector[self.Q_reb_index] * self.energy_scale),
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
            lle = []
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
            lle = [
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
            h1 = self.thermo.mixture_enthalpy(state["x1"], T, 0.0, P=P)
            h2 = self.thermo.mixture_enthalpy(state["x2"], T, 0.0, P=P)
            h_liquid = (1.0 - state["beta"]) * h1 + state["beta"] * h2
        vapor_total = sum(vapor_terms.values())
        y = {
            comp: vapor_terms[comp] / max(vapor_total, 1e-300)
            for comp in self.components
        }
        h_vapor = self.thermo.mixture_enthalpy(y, T, 1.0, P=P)
        return {
            "y": y,
            "bubble": vapor_total - 1.0,
            "lle": lle,
            "hL": float(h_liquid),
            "hV": float(h_vapor),
        }

    def _feed_component(self, stage: int, comp: str) -> float:
        return sum(
            feed["F"] * feed["z"].get(comp, 0.0)
            for feed in self.stage_feeds[stage]
        )

    def _feed_enthalpy(self, stage: int) -> float:
        return sum(feed["F"] * feed["H"] for feed in self.stage_feeds[stage])

    def residual(self, vector) -> np.ndarray:
        decoded = self.decode(vector)
        stages = decoded["stages"]
        props = [self.stage_properties(stage, state) for stage, state in enumerate(stages)]
        values = []
        D = stages[0]["V"]
        top_vapor_fraction = self.condenser_vapor_fraction
        for stage, state in enumerate(stages):
            for comp in self.components:
                incoming = self._feed_component(stage, comp)
                if stage > 0:
                    incoming += (
                        stages[stage - 1]["L"]
                        * stages[stage - 1]["aggregate_x"].get(comp, 0.0)
                    )
                if stage < self.N - 1:
                    incoming += (
                        stages[stage + 1]["V"]
                        * props[stage + 1]["y"].get(comp, 0.0)
                    )
                if stage == 0:
                    outgoing = (
                        (state["L"] + (1.0 - top_vapor_fraction) * D)
                        * state["aggregate_x"].get(comp, 0.0)
                        + top_vapor_fraction * D * props[stage]["y"].get(comp, 0.0)
                    )
                else:
                    outgoing = (
                        state["L"] * state["aggregate_x"].get(comp, 0.0)
                        + state["V"] * props[stage]["y"].get(comp, 0.0)
                    )
                values.append((incoming - outgoing) / self.component_scales[comp])

            incoming_energy = self._feed_enthalpy(stage)
            if stage > 0:
                incoming_energy += stages[stage - 1]["L"] * props[stage - 1]["hL"]
            if stage < self.N - 1:
                incoming_energy += stages[stage + 1]["V"] * props[stage + 1]["hV"]
            if stage == 0:
                incoming_energy += decoded["Q_cond"]
                outgoing_energy = (
                    (state["L"] + (1.0 - top_vapor_fraction) * D) * props[stage]["hL"]
                    + top_vapor_fraction * D * props[stage]["hV"]
                )
            else:
                if stage == self.N - 1:
                    incoming_energy += decoded["Q_reb"]
                outgoing_energy = (
                    state["L"] * props[stage]["hL"]
                    + state["V"] * props[stage]["hV"]
                )
            values.append((incoming_energy - outgoing_energy) / self.energy_scale)
            values.extend(props[stage]["lle"])
            values.append(props[stage]["bubble"])

        if self.distillate_spec["kind"] == "mass":
            product_mw = sum(
                (
                    (1.0 - top_vapor_fraction) * stages[0]["aggregate_x"].get(comp, 0.0)
                    + top_vapor_fraction * props[0]["y"].get(comp, 0.0)
                ) * self.thermo.props[comp].MW
                for comp in self.components
            )
            values.append(
                (D * product_mw - float(self.distillate_spec["value"]))
                / self.distillate_scale
            )
        else:
            values.append(
                (D - float(self.distillate_spec["value"])) / self.flow_scale
            )
        values.append(
            (stages[0]["L"] - self.reflux_ratio * D) / self.flow_scale
        )
        return np.asarray(values, dtype=float)

    def sparsity(self):
        matrix = lil_matrix((self.n_rows, self.n_vars), dtype=int)
        for stage, count in enumerate(self.stage_row_counts):
            rows = range(
                self.stage_row_starts[stage],
                self.stage_row_starts[stage] + count,
            )
            dependencies = [stage]
            if stage > 0:
                dependencies.append(stage - 1)
            if stage < self.N - 1:
                dependencies.append(stage + 1)
            for row in rows:
                for dependent in dependencies:
                    layout = self.layouts[dependent]
                    matrix[row, layout.start:layout.stop] = 1
                if stage == 0:
                    matrix[row, self.Q_cond_index] = 1
                if stage == self.N - 1:
                    matrix[row, self.Q_reb_index] = 1
        top = self.layouts[0]
        spec_row = self.n_rows - 2
        matrix[spec_row, top.start:top.stop] = 1
        matrix[spec_row + 1, top.liquid_flow] = 1
        matrix[spec_row + 1, top.vapor_flow] = 1
        return matrix.tocsr()

    def topology_for_decoded(self, decoded: dict) -> list[bool]:
        topology = []
        for stage, state in enumerate(decoded["stages"]):
            stable, _x1, _x2, _beta = self.stability.split(
                state["T"], state["aggregate_x"]
            )
            if self.active[stage]:
                distance = sum(
                    abs(state["x1"].get(comp, 0.0) - state["x2"].get(comp, 0.0))
                    for comp in self.components
                )
                topology.append(
                    stable
                    and self.phase_fraction_min
                    < state["beta"]
                    < 1.0 - self.phase_fraction_min
                    and distance > self.phase_distance_min
                )
            else:
                topology.append(stable)
        return topology

    def local_jacobian(self, vector, f0, rel_step: float):
        decoded = self.decode(vector)
        updated = self.topology_for_decoded(decoded)
        residual_norm = float(np.linalg.norm(f0, ord=np.inf))
        if updated != self.active and residual_norm <= self.topology_change_residual:
            raise _ActiveSetChange(self.profile_from_decoded(decoded), updated)

        stages = decoded["stages"]
        props = [self.stage_properties(stage, state) for stage, state in enumerate(stages)]
        matrix = lil_matrix((self.n_rows, self.n_vars), dtype=float)
        evaluations = 0
        top_vapor_fraction = self.condenser_vapor_fraction

        def add(row, column, value):
            if value:
                matrix[row, column] = matrix[row, column] + value

        for stage, layout in enumerate(self.layouts):
            columns = [layout.temperature]
            columns.extend(range(layout.x1.start, layout.x1.stop))
            if layout.active_vlle:
                columns.extend(range(layout.x2.start, layout.x2.stop))
                columns.append(layout.beta)
            state = stages[stage]
            base = props[stage]
            for column in columns:
                step = rel_step * max(abs(float(vector[column])), 1.0)
                trial = np.array(vector, dtype=float, copy=True)
                trial[column] += step
                changed_state = self.decode_stage(trial, stage)
                changed = self.stage_properties(stage, changed_state)
                evaluations += 1
                dx = {
                    comp: (
                        changed_state["aggregate_x"].get(comp, 0.0)
                        - state["aggregate_x"].get(comp, 0.0)
                    ) / step
                    for comp in self.components
                }
                dy = {
                    comp: (
                        changed["y"].get(comp, 0.0) - base["y"].get(comp, 0.0)
                    ) / step
                    for comp in self.components
                }
                dhL = (changed["hL"] - base["hL"]) / step
                dhV = (changed["hV"] - base["hV"]) / step
                liquid_coefficient = state["L"]
                vapor_coefficient = state["V"]
                if stage == 0:
                    liquid_coefficient += (1.0 - top_vapor_fraction) * state["V"]
                    vapor_coefficient = top_vapor_fraction * state["V"]
                for ci, comp in enumerate(self.components):
                    scale = self.component_scales[comp]
                    row = self.stage_row_starts[stage] + ci
                    add(
                        row,
                        column,
                        -(liquid_coefficient * dx[comp] + vapor_coefficient * dy[comp]) / scale,
                    )
                    if stage < self.N - 1:
                        add(
                            self.stage_row_starts[stage + 1] + ci,
                            column,
                            state["L"] * dx[comp] / scale,
                        )
                    if stage > 0:
                        add(
                            self.stage_row_starts[stage - 1] + ci,
                            column,
                            state["V"] * dy[comp] / scale,
                        )
                energy_row = self.stage_row_starts[stage] + self.nc
                add(
                    energy_row,
                    column,
                    -(liquid_coefficient * dhL + vapor_coefficient * dhV) / self.energy_scale,
                )
                if stage < self.N - 1:
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
                for index, value in enumerate(base["lle"]):
                    add(
                        equilibrium_row + index,
                        column,
                        (changed["lle"][index] - value) / step,
                    )
                add(
                    equilibrium_row + len(base["lle"]),
                    column,
                    (changed["bubble"] - base["bubble"]) / step,
                )
                if stage == 0 and self.distillate_spec["kind"] == "mass":
                    d_mw = sum(
                        self.thermo.props[comp].MW
                        * ((1.0 - top_vapor_fraction) * dx[comp] + top_vapor_fraction * dy[comp])
                        for comp in self.components
                    )
                    add(
                        self.n_rows - 2,
                        column,
                        state["V"] * d_mw / self.distillate_scale,
                    )

        for stage, layout in enumerate(self.layouts):
            state = stages[stage]
            item = props[stage]
            dL = state["L"]
            dV = state["V"]
            for ci, comp in enumerate(self.components):
                scale = self.component_scales[comp]
                row = self.stage_row_starts[stage] + ci
                add(row, layout.liquid_flow, -dL * state["aggregate_x"][comp] / scale)
                if stage < self.N - 1:
                    add(
                        self.stage_row_starts[stage + 1] + ci,
                        layout.liquid_flow,
                        dL * state["aggregate_x"][comp] / scale,
                    )
                if stage == 0:
                    product_comp = (
                        (1.0 - top_vapor_fraction) * state["aggregate_x"][comp]
                        + top_vapor_fraction * item["y"][comp]
                    )
                    add(row, layout.vapor_flow, -dV * product_comp / scale)
                else:
                    add(row, layout.vapor_flow, -dV * item["y"][comp] / scale)
                    add(
                        self.stage_row_starts[stage - 1] + ci,
                        layout.vapor_flow,
                        dV * item["y"][comp] / scale,
                    )
            energy_row = self.stage_row_starts[stage] + self.nc
            add(energy_row, layout.liquid_flow, -dL * item["hL"] / self.energy_scale)
            if stage < self.N - 1:
                add(
                    self.stage_row_starts[stage + 1] + self.nc,
                    layout.liquid_flow,
                    dL * item["hL"] / self.energy_scale,
                )
            if stage == 0:
                product_h = (
                    (1.0 - top_vapor_fraction) * item["hL"]
                    + top_vapor_fraction * item["hV"]
                )
                add(energy_row, layout.vapor_flow, -dV * product_h / self.energy_scale)
            else:
                add(energy_row, layout.vapor_flow, -dV * item["hV"] / self.energy_scale)
                add(
                    self.stage_row_starts[stage - 1] + self.nc,
                    layout.vapor_flow,
                    dV * item["hV"] / self.energy_scale,
                )

        add(self.stage_row_starts[0] + self.nc, self.Q_cond_index, 1.0)
        add(self.stage_row_starts[-1] + self.nc, self.Q_reb_index, 1.0)
        spec_row = self.n_rows - 2
        top = stages[0]
        top_props = props[0]
        top_layout = self.layouts[0]
        if self.distillate_spec["kind"] == "mass":
            product_mw = sum(
                (
                    (1.0 - top_vapor_fraction) * top["aggregate_x"][comp]
                    + top_vapor_fraction * top_props["y"][comp]
                ) * self.thermo.props[comp].MW
                for comp in self.components
            )
            add(spec_row, top_layout.vapor_flow, top["V"] * product_mw / self.distillate_scale)
        else:
            add(spec_row, top_layout.vapor_flow, top["V"] / self.flow_scale)
        add(spec_row + 1, top_layout.liquid_flow, top["L"] / self.flow_scale)
        add(
            spec_row + 1,
            top_layout.vapor_flow,
            -self.reflux_ratio * top["V"] / self.flow_scale,
        )
        return matrix.tocsr(), evaluations, "vlle_semi_analytic_local_thermo"

    def profile_from_decoded(self, decoded: dict) -> VLLEProfile:
        split_data = []
        for stage, state in enumerate(decoded["stages"]):
            if self.active[stage]:
                split_data.append((dict(state["x1"]), dict(state["x2"]), state["beta"]))
            else:
                split_data.append(None)
        return VLLEProfile(
            T=[state["T"] for state in decoded["stages"]],
            aggregate_x=[dict(state["aggregate_x"]) for state in decoded["stages"]],
            L=[state["L"] for state in decoded["stages"]],
            V=[state["V"] for state in decoded["stages"]],
            Q_cond=decoded["Q_cond"],
            Q_reb=decoded["Q_reb"],
            split_data=split_data,
        )

    def solve(self):
        x0 = self.pack_initial()
        solution = self.unit._sparse_newton_solve(
            self.residual,
            self.sparsity(),
            x0,
            self.solver_options,
            jacobian=self.local_jacobian,
        )
        if (
            not solution["success"]
            and self.unit._truthy_param(
                self.unit.get_param("vlle_colored_jacobian_fallback", True)
            )
        ):
            fallback = self.unit._sparse_newton_solve(
                self.residual,
                self.sparsity(),
                x0,
                self.solver_options,
                jacobian=None,
            )
            if (
                fallback["success"]
                or fallback["residual_norm"] < solution["residual_norm"]
            ):
                solution = fallback
        if not solution["success"]:
            raise RuntimeError(
                f"VLLE MESH failed (residual {solution['residual_norm']:.3e}): "
                f"{solution['message']}"
            )
        decoded = self.decode(solution["x"])
        solution["residual_norm"] = float(
            np.linalg.norm(self.residual(solution["x"]), ord=np.inf)
        )
        props = [
            self.stage_properties(stage, state)
            for stage, state in enumerate(decoded["stages"])
        ]
        return decoded, props, solution


def solve_vlle_active_set(
    unit,
    inlet,
    feed_specs,
    components,
    pressures,
    reflux_ratio,
    condenser_vapor_fraction,
    distillate_spec,
    T_min,
    T_max,
    flow_scale,
    energy_scale,
    component_scales,
    initial_profile: VLLEProfile,
    solver_options,
) -> VLLEColumnSolution:
    phase_fraction_min = float(unit.get_param("vlle_phase_fraction_min", 1e-6))
    phase_distance_min = float(unit.get_param("vlle_phase_distance_min", 1e-3))
    topology_change_residual = float(unit.get_param("vlle_topology_change_residual", 5e-2))
    max_outer = int(unit.get_param("vlle_max_topology_updates", 8))
    stability = VLLEStabilityCache(
        unit.thermo,
        components,
        float(unit.get_param("vlle_stability_tolerance", 1e-7)),
    )
    profile = initial_profile
    screened_active = [
        stability.split(T, x)[0]
        for T, x in zip(profile.T, profile.aggregate_x)
    ]
    initial_topology = str(
        unit.get_param("vlle_initial_topology", "screened")
    ).strip().lower().replace("-", "_")
    if initial_topology in ("screened", "auto"):
        active = list(screened_active)
    elif initial_topology in ("all_vle", "vle"):
        active = [False] * len(pressures)
    elif initial_topology in ("all_vlle", "vlle"):
        active = [True] * len(pressures)
    else:
        raise RuntimeError(
            "vlle_initial_topology must be screened, all_vle, or all_vlle"
        )
    history = [topology_text(active)]
    # Absent liquid phases make the fixed-topology equations singular.  A
    # deliberately overactivated seed is therefore reconciled against the
    # seed-profile stability result before its first Newton system is built.
    if initial_topology in ("all_vlle", "vlle") and active != screened_active:
        active = list(screened_active)
        history.append(topology_text(active))
    total_iterations = 0
    total_functions = 0
    total_jacobians = 0
    decoded = None
    props = None
    solution = None

    for outer in range(1, max_outer + 1):
        model = EquationOrientedVLLEColumn(
            unit,
            inlet,
            feed_specs,
            components,
            pressures,
            reflux_ratio,
            condenser_vapor_fraction,
            distillate_spec,
            T_min,
            T_max,
            flow_scale,
            energy_scale,
            component_scales,
            active,
            profile,
            stability,
            phase_fraction_min,
            phase_distance_min,
            topology_change_residual,
            solver_options,
        )
        try:
            decoded, props, solution = model.solve()
        except _ActiveSetChange as change:
            profile = change.profile
            active = list(change.active)
            for stage, is_active in enumerate(active):
                if not is_active:
                    profile.split_data[stage] = None
            history.append(topology_text(active))
            continue
        total_iterations += int(solution["iterations"])
        total_functions += int(solution["function_evaluations"])
        total_jacobians += int(solution["jacobian_evaluations"])
        profile = model.profile_from_decoded(decoded)
        updated = model.topology_for_decoded(decoded)
        history.append(topology_text(updated))
        if updated == active:
            return VLLEColumnSolution(
                decoded=decoded,
                stage_properties=props,
                solver=solution,
                active=active,
                topology_history=history,
                stability_checks=stability.calls,
                stability_cache_hits=stability.hits,
                outer_solves=outer,
                total_iterations=total_iterations,
                total_function_evaluations=total_functions,
                total_jacobian_evaluations=total_jacobians,
            )
        active = updated

    raise RuntimeError(
        f"VLLE active set did not stabilize in {max_outer} topology attempts; "
        f"history={history}"
    )
