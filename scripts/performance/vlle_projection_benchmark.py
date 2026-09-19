"""Shared instrumentation for experimental VLLE boundary-projection benchmarks.

This module deliberately monkeypatches the sparse Newton solve only inside a
short-lived benchmark worker. It is not production integration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse.linalg

from equilibrium_stage_column import EquilibriumStageColumnMixin
from equilibrium_stage_vlle import (
    EquationOrientedVLLEColumn,
    _ActiveSetChange,
    _split_phase_metrics,
    topology_text,
)

DEFAULT_SCREENED_GATE_FRACTION = 0.15
DEFAULT_SCREENED_GATE_CONTRACTION_RATIO = 0.5


@dataclass
class ProbeState:
    policy: str
    gate_fraction: float | None = None
    gate_contraction_ratio: float | None = None
    current_model: object | None = None
    current_vector: np.ndarray | None = None
    current_values: np.ndarray | None = None
    current_jacobian_serial: int = 0
    current_supports_projection: bool = False
    records: list[dict] = field(default_factory=list)
    projection_events: list[dict] = field(default_factory=list)
    projection_direction_assessments: int = 0
    projected_stage_assessments: int = 0
    projection_seconds: float = 0.0
    beta_gate_trace: list[dict] = field(default_factory=list)
    screened_stage_checks: list[dict] = field(default_factory=list)


class ProjectedBoundaryEvent(BaseException):
    def __init__(self, profile, active, residual_norm, predicted_topology, streak):
        self.profile = profile
        self.active = list(active)
        self.residual_norm = float(residual_norm)
        self.predicted_topology = str(predicted_topology)
        self.streak = int(streak)


def install_probe(
    policy: str,
    gate_fraction: float | None = None,
    gate_contraction_ratio: float | None = None,
) -> ProbeState:
    """Install one process-local Newton probe and return its collected state."""
    if policy not in ("baseline", "projected", "screened"):
        raise ValueError(f"unknown projection policy: {policy}")
    if policy == "screened" and gate_fraction is None:
        gate_fraction = DEFAULT_SCREENED_GATE_FRACTION
    if policy == "screened" and gate_contraction_ratio is None:
        gate_contraction_ratio = DEFAULT_SCREENED_GATE_CONTRACTION_RATIO
    state = ProbeState(
        policy=policy,
        gate_fraction=gate_fraction,
        gate_contraction_ratio=gate_contraction_ratio,
    )
    original_newton = EquilibriumStageColumnMixin._sparse_newton_solve
    original_assessment = EquationOrientedVLLEColumn.topology_assessment
    original_spsolve = scipy.sparse.linalg.spsolve

    def projected_spsolve(matrix, rhs, *args, **kwargs):
        direction = original_spsolve(matrix, rhs, *args, **kwargs)
        model = state.current_model
        if (
            state.policy != "baseline"
            and state.current_supports_projection
            and isinstance(model, EquationOrientedVLLEColumn)
            and state.current_vector is not None
            and state.current_values is not None
        ):
            last_serial = getattr(model, "_benchmark_projection_serial", None)
            if last_serial != state.current_jacobian_serial:
                projection_started = time.perf_counter()
                model._benchmark_projection_serial = state.current_jacobian_serial
                state.projection_direction_assessments += 1
                predicted_vector = state.current_vector + direction
                if state.policy == "projected":
                    predicted = model.decode(predicted_vector)
                    predicted_active, details = original_assessment(
                        model, predicted
                    )
                    state.projected_stage_assessments += model.N
                    changed = [
                        item
                        for item in details
                        if item["mapped_active"] != item["current_active"]
                    ]
                    contraction_only = bool(changed) and all(
                        item["current_active"] and not item["mapped_active"]
                        for item in changed
                    )
                else:
                    gate_fraction = (
                        model.phase_fraction_min
                        if state.gate_fraction is None
                        else state.gate_fraction
                    )
                    (
                        predicted_active,
                        changed,
                        assessed_stages,
                        minimum_prediction,
                        stage_checks,
                    ) = _screened_contraction(
                        model,
                        state.current_vector,
                        predicted_vector,
                        direction,
                        gate_fraction,
                        state.gate_contraction_ratio,
                    )
                    state.projected_stage_assessments += assessed_stages
                    state.screened_stage_checks.extend({
                        "topology": topology_text(model.active),
                        "jacobian": state.current_jacobian_serial,
                        **item,
                    } for item in stage_checks)
                    if minimum_prediction is not None:
                        state.beta_gate_trace.append({
                            "topology": topology_text(model.active),
                            "jacobian": state.current_jacobian_serial,
                            **minimum_prediction,
                        })
                    contraction_only = bool(changed)
                candidate = tuple(predicted_active) if contraction_only else None
                if candidate is not None and candidate == getattr(
                    model, "_benchmark_projection_candidate", None
                ):
                    streak = getattr(model, "_benchmark_projection_streak", 0) + 1
                elif candidate is not None:
                    streak = 1
                else:
                    streak = 0
                model._benchmark_projection_candidate = candidate
                model._benchmark_projection_streak = streak
                if candidate is not None:
                    state.projection_events.append({
                        "from": topology_text(model.active),
                        "predicted": topology_text(predicted_active),
                        "streak": streak,
                        "jacobian": state.current_jacobian_serial,
                        "changed": [
                            {
                                "stage": item["stage"] + 1,
                                "phase_fraction": item["phase_fraction"],
                            }
                            for item in changed
                        ],
                    })
                if candidate is not None and streak >= 2:
                    accepted = model.decode(state.current_vector)
                    state.projection_seconds += (
                        time.perf_counter() - projection_started
                    )
                    raise ProjectedBoundaryEvent(
                        model.profile_from_decoded(accepted),
                        predicted_active,
                        np.linalg.norm(state.current_values, ord=np.inf),
                        topology_text(predicted_active),
                        streak,
                    )
                state.projection_seconds += (
                    time.perf_counter() - projection_started
                )
        return direction

    def counted_newton(
        self,
        residual,
        sparsity,
        x0,
        options,
        jacobian=None,
        step_event=None,
    ):
        quality_context = getattr(
            getattr(self, "thermo", None), "quality_context", None
        )
        quality_active = getattr(
            self, "_quality_solver_aux_context_active", False
        )
        if quality_context is not None and not quality_active:
            return original_newton(
                self, residual, sparsity, x0, options, jacobian=jacobian
            )

        model = getattr(residual, "__self__", None)
        state.current_model = model
        state.current_supports_projection = jacobian is not None
        active = getattr(model, "active", None)
        topology = topology_text(active) if active is not None else "VLE_SEED"
        jacobian_count = 0

        wrapped_jacobian = None
        if jacobian is not None:
            def wrapped_jacobian(vector, values, rel_step):
                nonlocal jacobian_count
                jacobian_count += 1
                state.current_jacobian_serial += 1
                state.current_vector = np.array(vector, dtype=float, copy=True)
                state.current_values = np.array(values, dtype=float, copy=True)
                return jacobian(vector, values, rel_step)

        try:
            result = original_newton(
                self,
                residual,
                sparsity,
                x0,
                options,
                jacobian=wrapped_jacobian,
                step_event=None,
            )
        except ProjectedBoundaryEvent as event:
            state.records.append({
                "topology": topology,
                "jacobians": jacobian_count,
                "outcome": "projected_boundary",
                "to": event.predicted_topology,
                "residual": event.residual_norm,
            })
            raise _ActiveSetChange(
                event.profile,
                event.active,
                reason="projected_newton_boundary",
                residual_norm=event.residual_norm,
            )
        except _ActiveSetChange as change:
            state.records.append({
                "topology": topology,
                "jacobians": jacobian_count,
                "outcome": change.reason,
                "to": topology_text(change.active),
                "residual": change.residual_norm,
            })
            raise

        state.records.append({
            "topology": topology,
            "jacobians": (
                jacobian_count
                if jacobian is not None
                else int(result["jacobian_evaluations"])
            ),
            "outcome": result["message"],
            "success": bool(result["success"]),
            "residual": float(result["residual_norm"]),
        })
        return result

    scipy.sparse.linalg.spsolve = projected_spsolve
    EquilibriumStageColumnMixin._sparse_newton_solve = counted_newton
    return state


def solve_and_summarize(case: str, mode: str, column, feed, probe: ProbeState):
    """Solve one isolated column and return comparable benchmark data."""
    column.params["vlle_projection_enabled"] = False
    started = time.perf_counter()
    result = column.solve({"feed": feed})
    elapsed = time.perf_counter() - started
    performance = result.performance

    return {
        "case": case,
        "mode": mode,
        "elapsed_seconds": elapsed,
        "mesh_residual": performance["mesh_residual"],
        "topology": performance["vlle_topology"],
        "topology_history": performance["vlle_topology_history"],
        "reported_solver_iterations": performance["solver_iterations"],
        "reported_jacobian_evaluations": performance["jacobian_evaluations"],
        "observed_jacobian_evaluations": sum(
            item["jacobians"] for item in probe.records
        ),
        "records": probe.records,
        "projection_events": probe.projection_events,
        "projection_direction_assessments": (
            probe.projection_direction_assessments
        ),
        "projected_stage_assessments": probe.projected_stage_assessments,
        "projection_seconds": probe.projection_seconds,
        "gate_fraction": probe.gate_fraction,
        "gate_contraction_ratio": probe.gate_contraction_ratio,
        "beta_gate_trace": probe.beta_gate_trace,
        "screened_stage_checks": probe.screened_stage_checks,
        "distillate": _stream_summary(result.outlet_streams["distillate"]),
        "bottoms": _stream_summary(result.outlet_streams["bottoms"]),
        "condenser_duty_kW": performance["condenser_duty_kW"],
        "reboiler_duty_kW": performance["reboiler_duty_kW"],
    }


def print_summary(summary: dict) -> None:
    print("BENCHMARK_JSON " + json.dumps(summary, sort_keys=True))


def _stream_summary(stream):
    return {
        "F": stream.F,
        "T": stream.T,
        "composition": stream.composition,
    }


def _screened_contraction(
    model,
    current_vector,
    predicted_vector,
    direction,
    gate_fraction,
    gate_contraction_ratio,
):
    """Project only active stages whose beta-logit step crosses disappearance."""
    predicted_active = list(model.active)
    changed = []
    assessed_stages = 0
    minimum_prediction = None
    stage_checks = []
    for stage, layout in enumerate(model.layouts):
        if not layout.active_vlle:
            continue
        current_logit = float(current_vector[layout.beta])
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
        contraction_ratio = predicted_fraction / max(current_fraction, 1e-300)
        if (
            minimum_prediction is None
            or predicted_fraction < minimum_prediction["phase_fraction"]
        ):
            minimum_prediction = {
                "stage": stage + 1,
                "current_fraction": current_fraction,
                "phase_fraction": predicted_fraction,
                "contraction_ratio": contraction_ratio,
                "beta_step": beta_step,
            }
        if predicted_fraction > gate_fraction:
            continue
        if (
            gate_contraction_ratio is not None
            and contraction_ratio > gate_contraction_ratio
        ):
            continue

        assessed_stages += 1
        predicted_state = model.decode_stage(predicted_vector, stage)
        has_lle, split_x1, split_x2, split_beta = model.stability.split(
            predicted_state["T"], predicted_state["aggregate_x"]
        )
        phase_fraction, phase_distance = _split_phase_metrics(
            has_lle,
            split_x1,
            split_x2,
            split_beta,
            model.components,
        )
        mapped_active = bool(
            has_lle
            and phase_fraction > model.phase_fraction_min
            and phase_distance > model.phase_distance_min
        )
        stage_checks.append({
            "stage": stage + 1,
            "current_fraction": current_fraction,
            "predicted_fraction": predicted_fraction,
            "contraction_ratio": contraction_ratio,
            "beta_step": beta_step,
            "projected_phase_fraction": phase_fraction,
            "projected_phase_distance": phase_distance,
            "projected_active": mapped_active,
        })
        if mapped_active:
            continue
        predicted_active[stage] = False
        changed.append({
            "stage": stage,
            "current_active": True,
            "mapped_active": False,
            "has_lle": bool(has_lle),
            "phase_fraction": phase_fraction,
            "phase_distance": phase_distance,
        })
    return (
        predicted_active,
        changed,
        assessed_stages,
        minimum_prediction,
        stage_checks,
    )


def _sigmoid(value):
    if value >= 0.0:
        exponential = np.exp(-value)
        return float(1.0 / (1.0 + exponential))
    exponential = np.exp(value)
    return float(exponential / (1.0 + exponential))
