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

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .sparse_jacobian import FixedPatternCSR, SparsePatternBuilder
    from .distillation_condenser import TotalCondenserBoundary
    from .stage_efficiency import VaporStageEfficiencies
else:
    from sparse_jacobian import FixedPatternCSR, SparsePatternBuilder
    from distillation_condenser import TotalCondenserBoundary
    from stage_efficiency import VaporStageEfficiencies


@dataclass
class VLLEProfile:
    T: list[float]
    aggregate_x: list[dict[str, float]]
    L: list[float]
    V: list[float]
    Q_cond: float
    Q_reb: float
    split_data: list[Optional[tuple[dict[str, float], dict[str, float], float]]]
    vapor_compositions: Optional[list[dict[str, float]]] = None


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
    topology_events: list[dict]
    work: dict[str, int]
    final_topology_projection_checks: int
    top_liquid_routing: dict
    projection_cycle_recoveries: int = 0


class _ActiveSetChange(RuntimeError):
    def __init__(
        self,
        profile: VLLEProfile,
        active: list[bool],
        *,
        reason: str,
        residual_norm: float,
    ):
        super().__init__("VLLE stage topology changed")
        self.profile = profile
        self.active = list(active)
        self.reason = str(reason)
        self.residual_norm = float(residual_norm)
        self.solver_iterations = 0
        self.function_evaluations = 0
        self.jacobian_evaluations = 0

    def add_solver_progress(
        self,
        *,
        iterations: int,
        function_evaluations: int,
        jacobian_evaluations: int,
    ) -> None:
        self.solver_iterations += int(iterations)
        self.function_evaluations += int(function_evaluations)
        self.jacobian_evaluations += int(jacobian_evaluations)


class VLLESolveFailure(RuntimeError):
    """Numerical failure retaining the work spent on this initializer."""

    def __init__(self, message: str, work: dict[str, int]):
        super().__init__(message)
        self.work = dict(work)


class VLLETopologyCycle(VLLESolveFailure):
    """Active-set cycle with structured topology history for recovery."""

    def __init__(self, message: str, history: list[str], work: dict[str, int]):
        super().__init__(message, work)
        self.history = list(history)


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


def _split_phase_metrics(
    has_lle: bool,
    x1: dict[str, float],
    x2: dict[str, float],
    beta: float,
    components,
) -> tuple[float, float]:
    if not has_lle:
        return 0.0, 0.0
    phase_fraction = min(max(float(beta), 0.0), max(1.0 - float(beta), 0.0))
    distance = sum(
        abs(float(x1.get(comp, 0.0)) - float(x2.get(comp, 0.0)))
        for comp in components
    )
    return float(phase_fraction), float(distance)


def _split_is_active(
    has_lle: bool,
    x1: dict[str, float],
    x2: dict[str, float],
    beta: float,
    components,
    phase_fraction_min: float,
    phase_distance_min: float,
) -> bool:
    phase_fraction, distance = _split_phase_metrics(
        has_lle,
        x1,
        x2,
        beta,
        components,
    )
    return bool(
        has_lle
        and phase_fraction > phase_fraction_min
        and distance > phase_distance_min
    )


def _shared_vlle_vapor_closure(
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
        return ({
            comp: math.sqrt(
                max(K1.get(comp, 1.0) * x1.get(comp, 0.0), 1e-300)
                * max(K2.get(comp, 1.0) * x2.get(comp, 0.0), 1e-300)
            )
            for comp in components
        }, None)

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
    fused_closure = getattr(thermo, "_vdm_vapor_terms_closure", None)
    if callable(fused_closure):
        solved = fused_closure(T, P, target_fugacity, components)
        if solved is not None:
            return dict(solved["vapor_terms"]), solved
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
    vapor_terms = {
        comp: max(
            target_fugacity[comp]
            / (max(float(phi_v.get(comp, 1.0)), 1e-8) * max(P, 1e-300)),
            1e-300,
        )
        for comp in components
    }
    returned_vapor = _normalize(vapor_terms, components)
    association_state = None
    state_method = getattr(thermo, "_vdm_vapor_association_state", None)
    if callable(state_method):
        association_state = state_method(T, P, returned_vapor)
    return vapor_terms, association_state


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
    """Return shared-vapor terms while hiding optional closure-state reuse."""
    return _shared_vlle_vapor_closure(
        thermo, T, P, x1, x2, components, gamma1, gamma2
    )[0]


def three_phase_fugacity_residuals(
    thermo,
    T: float,
    P: float,
    x1: dict[str, float],
    x2: dict[str, float],
    y: dict[str, float],
    components,
    *,
    include_vapor=True,
) -> dict[str, float]:
    """Return maximum dimensionless log-fugacity residuals for one stage."""
    log_f1 = thermo._phase_log_fugacities(T, P, x1, "liquid")
    log_f2 = thermo._phase_log_fugacities(T, P, x2, "liquid")
    liquid_liquid = max(
        abs(log_f1[comp] - log_f2[comp]) for comp in components
    )
    log_fv = thermo._phase_log_fugacities(T, P, y, "vapor") if include_vapor else None
    vapor_liquid = max(
        max(
            abs(log_f1[comp] - log_fv[comp]),
            abs(log_f2[comp] - log_fv[comp]),
        )
        for comp in components
    ) if include_vapor else 0.0
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
                allow_unconverged_candidate=True,
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


def route_top_liquids(components, options, state, props):
    """Shared phase-labelled product/reflux routing for equations and seed inventories."""
    x1, x2, beta = state['x1'], state['x2'], state['beta']
    h1, h2 = props['hL1'], props['hL2']
    if options is None:
        return {'distillate_x':state['aggregate_x'], 'reflux_x':state['aggregate_x'],
                'distillate_h':props['hL'], 'reflux_h':props['hL'],
                'liquid1_x':x1, 'liquid2_x':x2,
                'distillate_beta':beta, 'reflux_beta':beta,
                'reflux_stage_beta':beta,
                'condensate_beta':beta, 'withdrawal_fraction':None}
    selector = options['component']
    swapped = bool(selector and x2[selector] > x1[selector])
    if swapped:
        x1, x2, beta, h1, h2 = x2, x1, 1-beta, h2, h1
    f1, f2 = options['fractions']
    draw1, draw2 = (1-beta)*f1, beta*f2
    return1, return2 = (1-beta)*(1-f1), beta*(1-f2)

    def blend(first, second, fallback):
        amount = first+second
        if amount <= 0:
            return fallback, props['hL'], beta
        composition = {c:(first*x1[c]+second*x2[c])/amount for c in components}
        return composition, (first*h1+second*h2)/amount, second/amount

    xD, hD, betaD = blend(draw1, draw2, state['aggregate_x'])
    xR, hR, betaR = blend(return1, return2, state['aggregate_x'])
    return {'distillate_x':xD, 'reflux_x':xR, 'distillate_h':hD, 'reflux_h':hR,
            'liquid1_x':x1, 'liquid2_x':x2, 'distillate_beta':betaD,
            'reflux_beta':betaR, 'condensate_beta':beta,
            'reflux_stage_beta':1-betaR if swapped else betaR,
            'withdrawal_fraction':draw1+draw2}


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
        self.liquid_routing = unit._distillate_liquid_routing_options(self.components)
        self.nc = len(self.components)
        self.N = len(pressures)
        self.pressures = [float(value) for value in pressures]
        self.reflux_ratio = float(reflux_ratio)
        self.condenser_vapor_fraction = float(condenser_vapor_fraction)
        self.distillate_spec = distillate_spec
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self.temperature_span = self.T_max - self.T_min
        self.condenser_boundary = TotalCondenserBoundary(
            unit, self.components, self.pressures[0], T_min, T_max, allow_lle=True,
        )
        self.flow_scale = float(flow_scale)
        self.energy_scale = float(energy_scale)
        self.component_scales = component_scales
        self.active = list(active)
        self.profile = profile
        self.stability = stability
        self.phase_fraction_min = float(phase_fraction_min)
        self.phase_distance_min = float(phase_distance_min)
        self.phase_fraction_appearance_min = float(unit.get_param(
            "vlle_phase_fraction_appearance_min",
            max(10.0 * self.phase_fraction_min, 1e-5),
        ))
        self.phase_distance_appearance_min = float(unit.get_param(
            "vlle_phase_distance_appearance_min",
            self.phase_distance_min,
        ))
        self.topology_change_residual = float(topology_change_residual)
        self.topology_policy = str(
            unit.get_param("vlle_topology_policy", "adaptive")
        ).strip().lower().replace("-", "_")
        if self.topology_policy not in ("adaptive", "residual_gate"):
            raise RuntimeError(
                "vlle_topology_policy must be adaptive or residual_gate"
            )
        self.topology_progress_fraction = float(unit.get_param(
            "vlle_topology_progress_fraction",
            0.35,
        ))
        self.topology_candidate_streak = max(1, int(unit.get_param(
            "vlle_topology_candidate_streak",
            2,
        )))
        self.topology_stall_iterations = max(1, int(unit.get_param(
            "vlle_topology_stall_iterations",
            3,
        )))
        self.projection_gate_fraction = float(unit.get_param(
            "vlle_projection_gate_fraction",
            0.15,
        ))
        self.projection_enabled = unit._truthy_param(unit.get_param(
            "vlle_projection_enabled",
            True,
        ))
        self.projection_contraction_ratio = float(unit.get_param(
            "vlle_projection_contraction_ratio",
            0.5,
        ))
        self.projection_candidate_streak = max(1, int(unit.get_param(
            "vlle_projection_candidate_streak",
            2,
        )))
        if not 0.0 < self.projection_gate_fraction < 0.5:
            raise RuntimeError(
                "vlle_projection_gate_fraction must be between 0 and 0.5"
            )
        if not 0.0 < self.projection_contraction_ratio < 1.0:
            raise RuntimeError(
                "vlle_projection_contraction_ratio must be between 0 and 1"
            )
        self._topology_initial_residual = None
        self._topology_best_residual = math.inf
        self._topology_no_progress = 0
        self._topology_last_candidate = None
        self._topology_candidate_count = 0
        self._projection_last_candidate = None
        self._projection_candidate_count = 0
        self.projection_direction_assessments = 0
        self.projection_checks = 0
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
        self.core_rows = self.n_rows
        self.efficiency = VaporStageEfficiencies(unit,self.components,self.N,feed_specs,self.n_vars,self.n_rows)
        self.n_vars += self.efficiency.extra_size
        self.n_rows += self.efficiency.extra_size
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
        # Each EquationOrientedVLLEColumn represents one fixed active-set
        # topology.  A topology change constructs a new system and therefore a
        # new checked Jacobian pattern automatically.
        self._sparsity = self.sparsity()
        self._jacobian_pattern = FixedPatternCSR(self._sparsity)

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
        if self.efficiency.active:
            vapor_seed = self.profile.vapor_compositions
            if vapor_seed is None:
                vapor_seed = []
                for stage in range(self.N):
                    state = self.decode_stage(vector,stage)
                    state['actual_y'] = None
                    vapor_seed.append(self.stage_properties(stage,state)['equilibrium_y'])
            self.efficiency.pack(vector,vapor_seed)
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
            'actual_y': self.efficiency.decode(vector,stage),
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
        vapor_association_state = None
        if not self.active[stage]:
            closure_method = getattr(
                self.thermo, "_K_values_with_vapor_state", None
            )
            if callable(closure_method):
                K, vapor_association_state = closure_method(
                    T, P, state["x1"]
                )
            else:
                K = self.thermo.K_values(T, P, state["x1"])
            vapor_terms = {
                comp: max(K.get(comp, 1.0) * state["x1"].get(comp, 0.0), 1e-300)
                for comp in self.components
            }
            lle = []
            h_liquid = self.thermo.mixture_enthalpy(
                state["x1"], T, vapor_fraction=0.0, P=P
            )
            h1 = h2 = h_liquid
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
            vapor_terms, vapor_association_state = _shared_vlle_vapor_closure(
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
        reused_enthalpy = getattr(
            self.thermo, "_vapor_enthalpy_from_association_state", None
        )
        actual_y = state.get('actual_y')
        if actual_y is not None:
            h_vapor = self.thermo.mixture_enthalpy(actual_y,T,1.,P=P)
        elif vapor_association_state is not None and callable(reused_enthalpy):
            h_vapor = reused_enthalpy(y, T, vapor_association_state)
        else:
            h_vapor = self.thermo.mixture_enthalpy(y, T, 1.0, P=P)
        return {
            "y": actual_y if actual_y is not None else y,
            'equilibrium_y': y,
            "bubble": (self.condenser_boundary.residual(T, state['aggregate_x'], vapor_total-1.)
                       if stage == 0 else vapor_total-1.),
            "lle": lle,
            "hL": float(h_liquid),
            "hL1": float(h1),
            "hL2": float(h2),
            "hV": float(h_vapor),
        }

    def top_liquid_routing(self, state, props):
        """Return physical top-liquid product/reflux compositions and enthalpies."""
        return route_top_liquids(self.components, self.liquid_routing, state, props)

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
        routing = self.top_liquid_routing(stages[0], props[0])
        values = []
        D = stages[0]["V"]
        top_vapor_fraction = self.condenser_vapor_fraction
        for stage, state in enumerate(stages):
            for comp in self.components:
                incoming = self._feed_component(stage, comp)
                if stage > 0:
                    incoming_liquid_x = routing['reflux_x'] if stage == 1 else stages[stage-1]['aggregate_x']
                    incoming += (
                        stages[stage - 1]["L"]
                        * incoming_liquid_x.get(comp, 0.0)
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
                incoming_liquid_h = routing['reflux_h'] if stage == 1 else props[stage-1]['hL']
                incoming_energy += stages[stage - 1]["L"] * incoming_liquid_h
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
                    (1.0 - top_vapor_fraction) * routing['distillate_x'].get(comp, 0.0)
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
        if self.liquid_routing is None:
            values.append((stages[0]['L']-self.reflux_ratio*D)/self.flow_scale)
        else:
            liquid_product = (1-top_vapor_fraction)*D
            condensed = stages[0]['L']+liquid_product
            values.append((liquid_product-condensed*routing['withdrawal_fraction'])/self.flow_scale)
        values.extend(self.efficiency.residuals(props,[s['V'] for s in stages]))
        return np.asarray(values, dtype=float)

    def sparsity(self):
        matrix = SparsePatternBuilder((self.n_rows, self.n_vars))
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
                    matrix.mark_range(row, layout.start, layout.stop)
                    for column in self.efficiency.columns.get(dependent,()):
                        matrix.mark(row,column)
                if stage == 0:
                    matrix.mark(row, self.Q_cond_index)
                if stage == self.N - 1:
                    matrix.mark(row, self.Q_reb_index)
        top = self.layouts[0]
        spec_row = self.core_rows - 2
        matrix.mark_range(spec_row, top.start, top.stop)
        matrix.mark(spec_row + 1, top.liquid_flow)
        matrix.mark(spec_row + 1, top.vapor_flow)
        if self.liquid_routing is not None:
            matrix.mark_range(spec_row+1, top.start, top.stop)
        self.efficiency.mark_sparsity(matrix,lambda j:tuple(range(self.layouts[j].start,self.layouts[j].stop)))
        return matrix.tocsr()

    def topology_assessment(self, decoded: dict) -> tuple[list[bool], list[dict]]:
        topology = []
        details = []
        for stage, state in enumerate(decoded["stages"]):
            has_lle, split_x1, split_x2, split_beta = self.stability.split(
                state["T"], state["aggregate_x"]
            )
            phase_fraction, distance = _split_phase_metrics(
                has_lle,
                split_x1,
                split_x2,
                split_beta,
                self.components,
            )
            if self.active[stage]:
                fraction_min = self.phase_fraction_min
                distance_min = self.phase_distance_min
            else:
                fraction_min = self.phase_fraction_appearance_min
                distance_min = self.phase_distance_appearance_min
            mapped_active = bool(
                has_lle
                and phase_fraction > fraction_min
                and distance > distance_min
            )
            topology.append(mapped_active)
            details.append({
                "stage": stage,
                "current_active": bool(self.active[stage]),
                "mapped_active": mapped_active,
                "has_lle": bool(has_lle),
                "phase_fraction": phase_fraction,
                "phase_distance": distance,
            })
        return topology, details

    def topology_for_decoded(self, decoded: dict) -> list[bool]:
        topology, _details = self.topology_assessment(decoded)
        return topology

    def _adaptive_topology_change(
        self,
        updated: list[bool],
        details: list[dict],
        residual_norm: float,
    ) -> tuple[bool, str]:
        if updated == self.active:
            self._topology_last_candidate = None
            self._topology_candidate_count = 0
            return False, "unchanged"

        candidate_key = tuple(updated)
        if candidate_key == self._topology_last_candidate:
            self._topology_candidate_count += 1
        else:
            self._topology_last_candidate = candidate_key
            self._topology_candidate_count = 1

        if self._topology_initial_residual is None:
            self._topology_initial_residual = max(residual_norm, 1e-30)
        progress_tolerance = max(1e-8, 1e-3 * self._topology_best_residual)
        if residual_norm < self._topology_best_residual - progress_tolerance:
            self._topology_best_residual = residual_norm
            self._topology_no_progress = 0
        else:
            self._topology_no_progress += 1

        # Once Newton is already close, rebuilding is cheap and the mapped
        # equilibrium topology is more useful than further persistence tests.
        if residual_norm <= self.topology_change_residual:
            return True, "residual_floor"

        changed = [
            item
            for item in details
            if item["mapped_active"] != item["current_active"]
        ]
        contraction_only = bool(changed) and all(
            item["current_active"] and not item["mapped_active"]
            for item in changed
        )
        # An active stage is represented by the larger VLLE equation set.  If
        # an independent split projection says that phase is absent, retaining
        # it risks a singular vanishing-phase block, so contract immediately.
        if contraction_only:
            return True, "confident_phase_contraction"

        progress_limit = (
            self.topology_progress_fraction * self._topology_initial_residual
        )
        if (
            self._topology_candidate_count >= self.topology_candidate_streak
            and residual_norm <= progress_limit
        ):
            return True, "persistent_candidate_after_progress"

        if (
            self._topology_candidate_count >= self.topology_candidate_streak
            and self._topology_no_progress >= self.topology_stall_iterations
        ):
            return True, "persistent_candidate_at_stall"
        return False, "deferred_candidate"

    def local_jacobian(self, vector, f0, rel_step: float):
        decoded = self.decode(vector)
        updated, details = self.topology_assessment(decoded)
        residual_norm = float(np.linalg.norm(f0, ord=np.inf))
        if self._topology_initial_residual is None:
            self._topology_initial_residual = max(residual_norm, 1e-30)
            self._topology_best_residual = residual_norm
        if self.topology_policy == "residual_gate":
            change_topology = (
                updated != self.active
                and residual_norm <= self.topology_change_residual
            )
            reason = "residual_gate"
        else:
            change_topology, reason = self._adaptive_topology_change(
                updated,
                details,
                residual_norm,
            )
        if change_topology:
            raise _ActiveSetChange(
                self.profile_from_decoded(decoded),
                updated,
                reason=reason,
                residual_norm=residual_norm,
            )

        stages = decoded["stages"]
        props = [self.stage_properties(stage, state) for stage, state in enumerate(stages)]
        matrix = self._jacobian_pattern.empty()
        top_routing = self.top_liquid_routing(stages[0], props[0])
        evaluations = 0
        top_vapor_fraction = self.condenser_vapor_fraction
        vapor_flows = [s['V'] for s in stages]

        def add(row, column, value):
            matrix.add(row, column, value)

        for stage, layout in enumerate(self.layouts):
            columns = [layout.temperature]
            columns.extend(range(layout.x1.start, layout.x1.stop))
            if layout.active_vlle:
                columns.extend(range(layout.x2.start, layout.x2.stop))
                columns.append(layout.beta)
            columns.extend(self.efficiency.columns.get(stage,()))
            state = stages[stage]
            base = props[stage]
            for column in columns:
                step = rel_step * max(abs(float(vector[column])), 1.0)
                trial = np.array(vector, dtype=float, copy=True)
                trial[column] += step
                changed_state = self.decode_stage(trial, stage)
                if column in self.efficiency.columns.get(stage,()):
                    changed = self.efficiency.actual_properties(base,stage,changed_state['T'],
                        self.pressures[stage],changed_state['actual_y'])
                else:
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
                reflux_dx, reflux_dh = dx, dhL
                if stage == 0:
                    changed_routing = self.top_liquid_routing(changed_state, changed)
                    reflux_dx = {comp:(changed_routing['reflux_x'][comp]
                                      -top_routing['reflux_x'][comp])/step
                                 for comp in self.components}
                    reflux_dh = (changed_routing['reflux_h']-top_routing['reflux_h'])/step
                    if self.liquid_routing is not None:
                        condensed = state['L']+(1-top_vapor_fraction)*state['V']
                        derivative = (changed_routing['withdrawal_fraction']
                                      -top_routing['withdrawal_fraction'])/step
                        add(self.core_rows-1,column,-condensed*derivative/self.flow_scale)
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
                            state["L"] * reflux_dx[comp] / scale,
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
                        state["L"] * reflux_dh / self.energy_scale,
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
                self.efficiency.add_local_derivatives(add,stage,column,props,changed,step,vapor_flows)
                if stage == 0 and self.distillate_spec["kind"] == "mass":
                    d_mw = sum(
                        self.thermo.props[comp].MW
                        * ((1.0 - top_vapor_fraction)
                           * (changed_routing['distillate_x'][comp]-top_routing['distillate_x'][comp])/step
                           + top_vapor_fraction * dy[comp])
                        for comp in self.components
                    )
                    add(
                        self.core_rows - 2,
                        column,
                        state["V"] * d_mw / self.distillate_scale,
                    )

        for stage, layout in enumerate(self.layouts):
            state = stages[stage]
            item = props[stage]
            self.efficiency.add_flow_derivatives(add,stage,layout.vapor_flow,props,vapor_flows)
            dL = state["L"]
            dV = state["V"]
            reflux_x = top_routing['reflux_x'] if stage == 0 else state['aggregate_x']
            reflux_h = top_routing['reflux_h'] if stage == 0 else item['hL']
            for ci, comp in enumerate(self.components):
                scale = self.component_scales[comp]
                row = self.stage_row_starts[stage] + ci
                add(row, layout.liquid_flow, -dL * state["aggregate_x"][comp] / scale)
                if stage < self.N - 1:
                    add(
                        self.stage_row_starts[stage + 1] + ci,
                        layout.liquid_flow,
                        dL * reflux_x[comp] / scale,
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
                    dL * reflux_h / self.energy_scale,
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
        spec_row = self.core_rows - 2
        top = stages[0]
        top_props = props[0]
        top_layout = self.layouts[0]
        if self.distillate_spec["kind"] == "mass":
            product_mw = sum(
                (
                    (1.0 - top_vapor_fraction) * top_routing['distillate_x'][comp]
                    + top_vapor_fraction * top_props["y"][comp]
                ) * self.thermo.props[comp].MW
                for comp in self.components
            )
            add(spec_row, top_layout.vapor_flow, top["V"] * product_mw / self.distillate_scale)
        else:
            add(spec_row, top_layout.vapor_flow, top["V"] / self.flow_scale)
        if self.liquid_routing is None:
            add(spec_row+1, top_layout.liquid_flow, top['L']/self.flow_scale)
            add(spec_row+1, top_layout.vapor_flow, -self.reflux_ratio*top['V']/self.flow_scale)
        else:
            fraction = top_routing['withdrawal_fraction']
            add(spec_row+1, top_layout.liquid_flow, -fraction*top['L']/self.flow_scale)
            add(spec_row+1, top_layout.vapor_flow,
                (1-top_vapor_fraction)*(1-fraction)*top['V']/self.flow_scale)
        return matrix.tocsr(), evaluations, "vlle_semi_analytic_local_thermo"

    def projected_boundary_event(self, vector, residual, direction) -> None:
        """Contract phases whose raw Newton step repeatedly predicts absence."""
        self.projection_direction_assessments += 1
        predicted_vector = np.asarray(vector, dtype=float) + np.asarray(
            direction, dtype=float
        )
        updated = list(self.active)
        changed = []
        for stage, layout in enumerate(self.layouts):
            if not layout.active_vlle:
                continue
            current_logit = float(vector[layout.beta])
            beta_step = float(direction[layout.beta])
            current_beta = _sigmoid(current_logit)
            moving_to_boundary = (
                (current_beta >= 0.5 and beta_step > 0.0)
                or (current_beta < 0.5 and beta_step < 0.0)
            )
            if not moving_to_boundary:
                continue
            predicted_beta = _sigmoid(current_logit + beta_step)
            current_fraction = min(current_beta, 1.0 - current_beta)
            predicted_fraction = min(predicted_beta, 1.0 - predicted_beta)
            contraction_ratio = predicted_fraction / max(
                current_fraction, 1e-300
            )
            if predicted_fraction > self.projection_gate_fraction:
                continue
            if contraction_ratio > self.projection_contraction_ratio:
                continue

            predicted_state = self.decode_stage(predicted_vector, stage)
            has_lle, split_x1, split_x2, split_beta = self.stability.split(
                predicted_state["T"], predicted_state["aggregate_x"]
            )
            self.projection_checks += 1
            phase_fraction, phase_distance = _split_phase_metrics(
                has_lle,
                split_x1,
                split_x2,
                split_beta,
                self.components,
            )
            mapped_active = bool(
                has_lle
                and phase_fraction > self.phase_fraction_min
                and phase_distance > self.phase_distance_min
            )
            if not mapped_active:
                updated[stage] = False
                changed.append(stage)

        candidate = tuple(updated) if changed else None
        if candidate is not None and candidate == self._projection_last_candidate:
            self._projection_candidate_count += 1
        elif candidate is not None:
            self._projection_last_candidate = candidate
            self._projection_candidate_count = 1
        else:
            self._projection_last_candidate = None
            self._projection_candidate_count = 0

        if (
            candidate is not None
            and self._projection_candidate_count
            >= self.projection_candidate_streak
        ):
            decoded = self.decode(vector)
            raise _ActiveSetChange(
                self.profile_from_decoded(decoded),
                updated,
                reason="projected_newton_boundary",
                residual_norm=float(np.linalg.norm(residual, ord=np.inf)),
            )

    def reset_projected_boundary_candidate(self) -> None:
        self._projection_last_candidate = None
        self._projection_candidate_count = 0

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
            vapor_compositions=[dict(self.stage_properties(j,state)['y'])
                                for j,state in enumerate(decoded['stages'])] if self.efficiency.active else None,
        )

    def solve(self):
        x0 = self.pack_initial()
        try:
            sparsity = self._sparsity
        except AttributeError:
            # Some callers construct a solver-shaped test double without
            # running __init__.  Real systems cache this topology-specific
            # pattern, while those deliberately partial instances can retain
            # the original lazy behavior.
            sparsity = self.sparsity()
        solution = self.unit._sparse_newton_solve(
            self.residual,
            sparsity,
            x0,
            self.solver_options,
            jacobian=self.local_jacobian,
            step_event=(
                self.projected_boundary_event
                if self.projection_enabled
                else None
            ),
        )
        attempted_iterations = int(solution["iterations"])
        attempted_functions = int(solution["function_evaluations"])
        attempted_jacobians = int(solution["jacobian_evaluations"])
        if not solution["success"] and self.topology_policy == "adaptive":
            decoded = self.decode(solution["x"])
            updated = self.topology_for_decoded(decoded)
            if updated != self.active:
                change = _ActiveSetChange(
                    self.profile_from_decoded(decoded),
                    updated,
                    reason="failed_iterate_recovery",
                    residual_norm=float(solution["residual_norm"]),
                )
                change.add_solver_progress(
                    iterations=attempted_iterations,
                    function_evaluations=attempted_functions,
                    jacobian_evaluations=attempted_jacobians,
                )
                raise change
        if (
            not solution["success"]
            and self.unit._truthy_param(
                self.unit.get_param("vlle_colored_jacobian_fallback", True)
            )
        ):
            self.reset_projected_boundary_candidate()
            try:
                fallback = self.unit._sparse_newton_solve(
                    self.residual,
                    self.sparsity(),
                    x0,
                    self.solver_options,
                    jacobian=None,
                    step_event=(
                        self.projected_boundary_event
                        if self.projection_enabled
                        else None
                    ),
                )
            except _ActiveSetChange as change:
                change.add_solver_progress(
                    iterations=attempted_iterations,
                    function_evaluations=attempted_functions,
                    jacobian_evaluations=attempted_jacobians,
                )
                raise
            attempted_iterations += int(fallback["iterations"])
            attempted_functions += int(fallback["function_evaluations"])
            attempted_jacobians += int(fallback["jacobian_evaluations"])
            if (
                fallback["success"]
                or fallback["residual_norm"] < solution["residual_norm"]
            ):
                solution = fallback
        if not solution["success"] and self.topology_policy == "adaptive":
            decoded = self.decode(solution["x"])
            updated = self.topology_for_decoded(decoded)
            if updated != self.active:
                change = _ActiveSetChange(
                    self.profile_from_decoded(decoded),
                    updated,
                    reason="failed_iterate_recovery_after_fallback",
                    residual_norm=float(solution["residual_norm"]),
                )
                change.add_solver_progress(
                    iterations=attempted_iterations,
                    function_evaluations=attempted_functions,
                    jacobian_evaluations=attempted_jacobians,
                )
                raise change
        if not solution["success"]:
            raise VLLESolveFailure(
                f"VLLE MESH failed (residual {solution['residual_norm']:.3e}): "
                f"{solution['message']}",
                {
                    "solver_iterations": attempted_iterations,
                    "function_evaluations": attempted_functions,
                    "jacobian_evaluations": attempted_jacobians,
                },
            )
        decoded = self.decode(solution["x"])
        solution["residual_norm"] = float(
            np.linalg.norm(self.residual(solution["x"]), ord=np.inf)
        )
        solution["total_iterations"] = attempted_iterations
        solution["total_function_evaluations"] = attempted_functions
        solution["total_jacobian_evaluations"] = attempted_jacobians
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
    initial_active: Optional[list[bool]] = None,
) -> VLLEColumnSolution:
    phase_fraction_min = float(unit.get_param("vlle_phase_fraction_min", 1e-6))
    phase_distance_min = float(unit.get_param("vlle_phase_distance_min", 1e-3))
    phase_fraction_appearance_min = float(unit.get_param(
        "vlle_phase_fraction_appearance_min",
        max(10.0 * phase_fraction_min, 1e-5),
    ))
    phase_distance_appearance_min = float(unit.get_param(
        "vlle_phase_distance_appearance_min",
        phase_distance_min,
    ))
    topology_change_residual = float(unit.get_param(
        "vlle_topology_change_residual",
        5e-2,
    ))
    configured_max_outer = unit.get_param("vlle_max_topology_updates")
    max_outer = (
        max(8, min(len(pressures) + 4, 24))
        if configured_max_outer is None
        else int(configured_max_outer)
    )
    stability = VLLEStabilityCache(
        unit.thermo,
        components,
        float(unit.get_param("vlle_stability_tolerance", 1e-7)),
    )
    boundary = TotalCondenserBoundary(unit, components, pressures[0], T_min, T_max, allow_lle=True)
    if boundary.options:
        profile = VLLEProfile(
            T=list(initial_profile.T), aggregate_x=[dict(x) for x in initial_profile.aggregate_x],
            L=list(initial_profile.L), V=list(initial_profile.V),
            Q_cond=initial_profile.Q_cond, Q_reb=initial_profile.Q_reb,
            split_data=list(initial_profile.split_data),
            vapor_compositions=initial_profile.vapor_compositions,
        )
        profile.T[0] = boundary.seed_temperature(profile.aggregate_x[0], profile.T[0])
        split, x1, x2, beta = stability.split(profile.T[0], profile.aggregate_x[0])
        profile.split_data[0] = (x1, x2, beta) if split else None
        if initial_active is not None:
            initial_active = list(initial_active)
            initial_active[0] = _split_is_active(split, x1, x2, beta, components,
                                                phase_fraction_appearance_min, phase_distance_appearance_min)
        initial_profile = profile
    profile = initial_profile
    screened_active = None
    initial_topology = "previous_recycle" if initial_active is not None else str(
        unit.get_param("vlle_initial_topology", "screened")
    ).strip().lower().replace("-", "_")
    if initial_active is not None:
        if len(initial_active) != len(pressures):
            raise RuntimeError(
                "recycle VLLE topology length does not match the column stage count"
            )
        active = [bool(value) for value in initial_active]
    else:
        screened_active = []
        for T, x in zip(profile.T, profile.aggregate_x):
            has_lle, split_x1, split_x2, split_beta = stability.split(T, x)
            screened_active.append(_split_is_active(
                has_lle,
                split_x1,
                split_x2,
                split_beta,
                components,
                phase_fraction_appearance_min,
                phase_distance_appearance_min,
            ))
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
    topology_events = []
    # Absent liquid phases make the fixed-topology equations singular.  A
    # deliberately overactivated seed is therefore reconciled against the
    # seed-profile stability result before its first Newton system is built.
    if (
        screened_active is not None
        and initial_topology in ("all_vlle", "vlle")
        and active != screened_active
    ):
        previous = topology_text(active)
        active = list(screened_active)
        history.append(topology_text(active))
        topology_events.append({
            "from": previous,
            "to": topology_text(active),
            "reason": "initial_profile_screen",
            "residual_norm": None,
        })
    visited_topologies = {topology_text(active)}
    projection_enabled = unit._truthy_param(unit.get_param('vlle_projection_enabled',True))
    projection_cycle_recoveries = 0
    def recover_projection_cycle(previous,proposed,residual):
        nonlocal projection_enabled,projection_cycle_recoveries
        if (not projection_enabled or projection_cycle_recoveries
                or not unit._truthy_param(unit.get_param('vlle_projection_cycle_fallback',True))):
            return False
        projection_enabled = False
        projection_cycle_recoveries += 1
        # A different solver mode may validly revisit a topology. Reset only
        # this mode's cycle guard, retaining history, iterate and spent work.
        visited_topologies.clear()
        topology_events.append({'from':previous,'to':proposed,
            'reason':'disable_projection_on_cycle','residual_norm':float(residual)})
        return True
    total_iterations = 0
    total_functions = 0
    total_jacobians = 0
    total_projection_directions = 0
    total_projection_checks = 0
    decoded = None
    props = None
    solution = None
    outer = 0

    def work_snapshot():
        return {
            "solver_iterations": total_iterations,
            "function_evaluations": total_functions,
            "jacobian_evaluations": total_jacobians,
            "vlle_topology_solves": outer,
            "vlle_stability_checks": stability.calls,
            "vlle_stability_cache_hits": stability.hits,
            "vlle_projection_direction_assessments": total_projection_directions,
            "vlle_projection_checks": total_projection_checks,
        }

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
        model.projection_enabled = projection_enabled
        try:
            decoded, props, solution = model.solve()
        except _ActiveSetChange as change:
            total_iterations += change.solver_iterations
            total_functions += change.function_evaluations
            total_jacobians += change.jacobian_evaluations
            total_projection_directions += model.projection_direction_assessments
            total_projection_checks += model.projection_checks
            previous = topology_text(active)
            proposed = topology_text(change.active)
            if proposed in visited_topologies and not recover_projection_cycle(previous,proposed,change.residual_norm):
                raise VLLETopologyCycle(
                    "VLLE topology cycle detected while changing "
                    f"{previous} -> {proposed}; history={history}",
                    history,
                    work_snapshot(),
                ) from change
            profile = change.profile
            active = list(change.active)
            for stage, is_active in enumerate(active):
                if not is_active:
                    profile.split_data[stage] = None
            history.append(topology_text(active))
            topology_events.append({
                "from": previous,
                "to": topology_text(active),
                "reason": change.reason,
                "residual_norm": change.residual_norm,
                "solver_iterations": change.solver_iterations,
                "function_evaluations": change.function_evaluations,
                "jacobian_evaluations": change.jacobian_evaluations,
                "projection_direction_assessments": (
                    model.projection_direction_assessments
                ),
                "projection_checks": model.projection_checks,
            })
            visited_topologies.add(proposed)
            continue
        except VLLESolveFailure as failure:
            total_iterations += failure.work["solver_iterations"]
            total_functions += failure.work["function_evaluations"]
            total_jacobians += failure.work["jacobian_evaluations"]
            total_projection_directions += model.projection_direction_assessments
            total_projection_checks += model.projection_checks
            failure.work = work_snapshot()
            raise
        total_iterations += int(solution.get(
            "total_iterations", solution["iterations"]
        ))
        total_functions += int(solution.get(
            "total_function_evaluations", solution["function_evaluations"]
        ))
        total_jacobians += int(solution.get(
            "total_jacobian_evaluations", solution["jacobian_evaluations"]
        ))
        total_projection_directions += model.projection_direction_assessments
        total_projection_checks += model.projection_checks
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
                topology_events=topology_events,
                work=work_snapshot(),
                final_topology_projection_checks=model.projection_checks,
                top_liquid_routing=model.top_liquid_routing(decoded['stages'][0], props[0]),
                projection_cycle_recoveries=projection_cycle_recoveries,
            )
        topology_events.append({
            "from": topology_text(active),
            "to": topology_text(updated),
            "reason": "post_convergence_screen",
            "residual_norm": float(solution["residual_norm"]),
            "solver_iterations": int(solution["total_iterations"]),
            "function_evaluations": int(solution["total_function_evaluations"]),
            "jacobian_evaluations": int(solution["total_jacobian_evaluations"]),
            "projection_direction_assessments": (
                model.projection_direction_assessments
            ),
            "projection_checks": model.projection_checks,
        })
        proposed = topology_text(updated)
        if proposed in visited_topologies and not recover_projection_cycle(
                topology_text(active),proposed,solution['residual_norm']):
            raise VLLETopologyCycle(
                "VLLE topology cycle detected after convergence while changing "
                f"{topology_text(active)} -> {proposed}; history={history}",
                history,
                work_snapshot(),
            )
        visited_topologies.add(proposed)
        active = updated

    raise VLLESolveFailure(
        f"VLLE active set did not stabilize in {max_outer} topology attempts; "
        f"history={history}",
        work_snapshot(),
    )
