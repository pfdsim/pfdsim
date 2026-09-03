"""Pure-solid solid-liquid equilibrium for crystallizer calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .common import P_REF, R, ThermodynamicsError


@dataclass
class PureSolidSLEResult:
    """Equilibrium allocation between one liquid solution and pure solids."""

    liquid_component_flows: dict[str, float]
    solid_component_flows: dict[str, float]
    liquid_composition: dict[str, float]
    liquid_activities: dict[str, float]
    saturation_activities: dict[str, float]
    saturation_residuals: dict[str, float]
    iterations: int
    converged: bool
    details: dict = field(default_factory=dict)


def _exp_for_report(value: float) -> float:
    """Exponentiate a log activity without overflowing diagnostics."""
    return math.exp(min(float(value), 700.0))


def _normalized_composition(
    thermo,
    composition: dict[str, float],
) -> dict[str, float]:
    values = {
        component: max(0.0, float(composition.get(component, 0.0)))
        for component in thermo.components
    }
    total = sum(values.values())
    if total <= 0.0:
        raise ThermodynamicsError("SLE mother-liquor composition is empty")
    return {
        component: value / total
        for component, value in values.items()
    }


def liquid_solution_activities(
    thermo,
    T: float,
    P: float,
    composition: dict[str, float],
) -> dict[str, float]:
    """Return liquid activities on the pure-liquid standard-state basis."""
    x = _normalized_composition(thermo, composition)
    activity_coefficients = getattr(thermo, 'activity_coefficients', None)
    if callable(activity_coefficients):
        gamma = activity_coefficients(float(T), x)
        return {
            component: x[component]
            * max(float(gamma.get(component, 1.0)), 1.0e-300)
            for component in thermo.components
        }

    fugacity_coefficients = getattr(thermo, 'fugacity_coefficients', None)
    if callable(fugacity_coefficients):
        mixture_phi = fugacity_coefficients(float(T), float(P), x, 'liquid')
        activities = {}
        for component in thermo.components:
            pure = {
                candidate: 1.0 if candidate == component else 0.0
                for candidate in thermo.components
            }
            pure_phi = fugacity_coefficients(
                float(T), float(P), pure, 'liquid'
            )
            activities[component] = (
                x[component]
                * max(float(mixture_phi.get(component, 1.0)), 1.0e-300)
                / max(float(pure_phi.get(component, 1.0)), 1.0e-300)
            )
        return activities

    return dict(x)


def pure_solid_log_saturation_activity(
    thermo,
    component: str,
    T: float,
    P: float = P_REF,
) -> float:
    """Return ``ln(a_sat)`` for a pure solid and subcooled liquid.

    The fusion Gibbs energy is integrated from the normal melting point using
    the resolved, temperature-dependent liquid and solid heat capacities. A
    pressure correction based on the pure solid/liquid molar-volume difference
    is included away from the thermochemical standard pressure.
    """
    temperature = float(T)
    pressure = float(P)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ThermodynamicsError(
            "SLE temperature must be positive and finite"
        )
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ThermodynamicsError("SLE pressure must be positive and finite")
    props = thermo.props.get(component)
    if props is None:
        raise ThermodynamicsError(
            f"Component '{component}' is unavailable for SLE"
        )
    if props.Tm is None or not math.isfinite(float(props.Tm)) or props.Tm <= 0.0:
        raise ThermodynamicsError(
            f"Cannot calculate SLE for '{component}' without a valid Tm"
        )
    if (
        props.Hfus is None
        or not math.isfinite(float(props.Hfus))
        or props.Hfus <= 0.0
    ):
        raise ThermodynamicsError(
            f"Cannot calculate SLE for '{component}' without a positive Hfus"
        )

    melting_temperature = float(props.Tm)
    heat_of_fusion = float(props.Hfus)
    thermo.mark_property_source_context_once(
        component, 'Tm', phase='solid_liquid_equilibrium'
    )
    thermo.mark_property_source_context_once(
        component, 'Hfus', phase='solid_liquid_equilibrium'
    )

    liquid_delta_h = thermo._integrate_liquid_cp(
        component, melting_temperature, temperature
    )
    solid_delta_h = thermo._integrate_solid_cp(
        component, melting_temperature, temperature
    )
    liquid_delta_s = thermo._integrate_cp_over_T(
        component, melting_temperature, temperature, 'liquid'
    )
    solid_delta_s = thermo._integrate_cp_over_T(
        component, melting_temperature, temperature, 'solid'
    )
    fusion_enthalpy = heat_of_fusion + liquid_delta_h - solid_delta_h
    fusion_entropy = (
        1000.0 * heat_of_fusion / melting_temperature
        + liquid_delta_s
        - solid_delta_s
    )
    fusion_gibbs = fusion_enthalpy - temperature * fusion_entropy / 1000.0
    log_activity = -1000.0 * fusion_gibbs / (R * temperature)

    if abs(pressure - P_REF) > 1.0e-12:
        solid_volume = thermo._solid_molar_volume(component, temperature)
        liquid_volume, _note = thermo._liquid_molar_volume_for_poynting(
            component, temperature
        )
        if liquid_volume is None or liquid_volume <= 0.0:
            raise ThermodynamicsError(
                f"Cannot calculate the SLE pressure correction for '{component}' "
                "without a liquid molar volume"
            )
        # Volumes are m3/kmol and pressure is bar. This product is kJ/kmol;
        # divide by R*T on the numerically equivalent J/mol basis.
        pressure_work_kJ_per_kmol = (
            (solid_volume - liquid_volume)
            * (pressure - P_REF)
            * 100.0
        )
        log_activity += pressure_work_kJ_per_kmol / (R * temperature)

    return float(log_activity)


def solve_pure_solid_sle(
    thermo,
    T: float,
    P: float,
    total_component_flows: dict[str, float],
    candidates: list[str] | tuple[str, ...],
    *,
    initial_solid_flows: dict[str, float] | None = None,
    tolerance: float = 1.0e-8,
    max_iterations: int = 500,
) -> PureSolidSLEResult:
    """Solve liquid/pure-solid equilibrium by complementarity residuals."""
    temperature = float(T)
    pressure = float(P)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ThermodynamicsError(
            "SLE temperature must be positive and finite"
        )
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ThermodynamicsError("SLE pressure must be positive and finite")
    total_flows = {}
    for component in thermo.components:
        flow = float(total_component_flows.get(component, 0.0))
        if not math.isfinite(flow) or flow < 0.0:
            raise ThermodynamicsError(
                f"SLE total flow for '{component}' must be finite and nonnegative"
            )
        total_flows[component] = flow
    candidate_names = list(dict.fromkeys(str(value) for value in candidates))
    unknown = sorted(set(candidate_names) - set(thermo.components))
    if unknown:
        raise ThermodynamicsError(
            "SLE candidates are not fluid-backend components: "
            + ', '.join(unknown)
        )
    present_candidates = [
        component
        for component in candidate_names
        if total_flows.get(component, 0.0) > 1.0e-15
    ]
    above_melting_candidates = []
    active_candidates = []
    for component in present_candidates:
        props = thermo.props.get(component)
        melting_temperature = getattr(props, 'Tm', None)
        try:
            above_melting = (
                melting_temperature is not None
                and temperature > float(melting_temperature)
            )
        except (TypeError, ValueError):
            above_melting = False
        if above_melting:
            above_melting_candidates.append(component)
        else:
            active_candidates.append(component)
    liquid_total = sum(total_flows.values())
    if liquid_total <= 0.0:
        raise ThermodynamicsError("SLE feed contains no fluid-capable material")

    log_saturation = {
        component: pure_solid_log_saturation_activity(
            thermo, component, temperature, pressure
        )
        for component in active_candidates
    }
    if not active_candidates:
        composition = {
            component: flow / liquid_total
            for component, flow in total_flows.items()
            if flow > 0.0
        }
        activities = liquid_solution_activities(
            thermo, temperature, pressure, composition
        )
        return PureSolidSLEResult(
            liquid_component_flows=total_flows,
            solid_component_flows={},
            liquid_composition=composition,
            liquid_activities=activities,
            saturation_activities={},
            saturation_residuals={},
            iterations=0,
            converged=True,
            details={
                'above_melting_candidates': above_melting_candidates,
                'solver_message': 'no subcooled solid candidates',
            },
        )

    all_liquid_composition = {
        component: flow / liquid_total
        for component, flow in total_flows.items()
        if flow > 0.0
    }
    all_liquid_activities = liquid_solution_activities(
        thermo, temperature, pressure, all_liquid_composition
    )
    all_liquid_residuals = {
        component: log_saturation[component]
        - math.log(
            max(all_liquid_activities.get(component, 0.0), 1.0e-300)
        )
        for component in active_candidates
    }
    if all(value >= -tolerance for value in all_liquid_residuals.values()):
        return PureSolidSLEResult(
            liquid_component_flows=total_flows,
            solid_component_flows={},
            liquid_composition=all_liquid_composition,
            liquid_activities=all_liquid_activities,
            saturation_activities={
                component: _exp_for_report(value)
                for component, value in log_saturation.items()
            },
            saturation_residuals=all_liquid_residuals,
            iterations=0,
            converged=True,
            details={
                'complementarity_residual': 0.0,
                'feasibility_residual': 0.0,
                'solver_message': 'stable all-liquid state',
                'above_melting_candidates': above_melting_candidates,
            },
        )

    upper = np.asarray(
        [total_flows[component] for component in active_candidates],
        dtype=float,
    )
    flow_scale = max(liquid_total, 1.0)
    initial = initial_solid_flows or {}
    x0 = np.asarray([
        min(max(float(initial.get(component, 0.0)), 0.0), upper[index])
        for index, component in enumerate(active_candidates)
    ])

    last = {}

    def evaluate(solid_values: np.ndarray):
        liquid_flows = dict(total_flows)
        for index, component in enumerate(active_candidates):
            liquid_flows[component] = max(
                0.0, total_flows[component] - float(solid_values[index])
            )
        remaining = sum(liquid_flows.values())
        if remaining <= max(1.0e-14, 1.0e-14 * liquid_total):
            raise ThermodynamicsError(
                "Pure-solid SLE reached an all-solid state; this first "
                "crystallizer model requires a liquid mother phase"
            )
        composition = {
            component: flow / remaining
            for component, flow in liquid_flows.items()
            if flow > 0.0
        }
        activities = liquid_solution_activities(
            thermo, temperature, pressure, composition
        )
        undersaturation = np.asarray([
            log_saturation[component]
            - math.log(max(activities.get(component, 0.0), 1.0e-300))
            for component in active_candidates
        ])
        scaled_solids = solid_values / flow_scale
        # Fischer-Burmeister residual for s >= 0, undersaturation >= 0,
        # and s*undersaturation = 0.
        residual = (
            np.sqrt(scaled_solids * scaled_solids + undersaturation * undersaturation)
            - scaled_solids
            - undersaturation
        )
        last.update({
            'liquid_flows': liquid_flows,
            'composition': composition,
            'activities': activities,
            'undersaturation': undersaturation,
        })
        return residual

    try:
        solved = least_squares(
            evaluate,
            x0,
            bounds=(np.zeros_like(upper), upper),
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=int(max_iterations),
        )
        residual = evaluate(solved.x)
    except ThermodynamicsError:
        raise
    except Exception as exc:
        raise ThermodynamicsError("Pure-solid SLE solve failed") from exc

    solids = {
        component: float(solved.x[index])
        for index, component in enumerate(active_candidates)
        if float(solved.x[index]) > max(1.0e-12, tolerance * flow_scale)
    }
    undersaturation = {
        component: float(last['undersaturation'][index])
        for index, component in enumerate(active_candidates)
    }
    complementarity_error = float(np.max(np.abs(residual)))
    feasibility_error = max(
        [-value for value in undersaturation.values()]
        + [
            abs(value)
            for component, value in undersaturation.items()
            if solids.get(component, 0.0) > 0.0
        ]
        + [0.0]
    )
    converged = bool(
        solved.success
        and complementarity_error <= tolerance
        and feasibility_error <= tolerance
    )
    if not converged:
        raise ThermodynamicsError(
            "Pure-solid SLE did not converge to a feasible equilibrium "
            f"(complementarity residual={complementarity_error:.3g}, "
            f"feasibility residual={feasibility_error:.3g})"
        )

    return PureSolidSLEResult(
        liquid_component_flows={
            component: float(flow)
            for component, flow in last['liquid_flows'].items()
            if float(flow) > 1.0e-15
        },
        solid_component_flows=solids,
        liquid_composition=dict(last['composition']),
        liquid_activities={
            component: float(value)
            for component, value in last['activities'].items()
        },
        saturation_activities={
            component: _exp_for_report(value)
            for component, value in log_saturation.items()
        },
        saturation_residuals=undersaturation,
        iterations=int(solved.nfev),
        converged=True,
        details={
            'complementarity_residual': complementarity_error,
            'feasibility_residual': feasibility_error,
            'solver_message': str(solved.message),
            'above_melting_candidates': above_melting_candidates,
        },
    )
