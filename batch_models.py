"""Adaptive homogeneous batch and semi-batch reactor balances."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional

import numpy as np
from scipy.integrate import solve_ivp

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .kinetic_models import (
        HomogeneousRateState,
        KineticReaction,
        KineticsError,
        evaluate_reaction_rate,
    )
else:
    from kinetic_models import (
        HomogeneousRateState,
        KineticReaction,
        KineticsError,
        evaluate_reaction_rate,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_basic import _ThermoStateSolver
else:
    from unit_operations_basic import _ThermoStateSolver


class _BatchTerminalEventError(KineticsError):
    """A deterministic trajectory constraint, not an integrator failure."""


@dataclass(frozen=True)
class BatchFeed:
    """One continuous-equivalent inlet apportioned to each batch."""

    name: str
    component_amounts_kmol: Mapping[str, float]
    molar_enthalpy_kJ_kmol: float
    start_h: Optional[float] = None
    stop_h: Optional[float] = None

    @property
    def scheduled(self) -> bool:
        return self.start_h is not None or self.stop_h is not None


@dataclass(frozen=True)
class BatchThermalModel:
    mode: str = 'isothermal'
    isothermal_temperature_K: Optional[float] = None
    specified_energy_kJ_batch: Optional[float] = None
    UA_W_K: Optional[float] = None
    jacket_temperature_K: Optional[float] = None


@dataclass(frozen=True)
class BatchNumericalControls:
    method: str = 'auto'
    relative_tolerance: float = 1.0e-7
    absolute_tolerance: float = 1.0e-9
    maximum_step_h: Optional[float] = None
    first_step_h: Optional[float] = None
    profile_points: int = 31


@dataclass(frozen=True)
class AdaptiveBatchSolution:
    outlet_component_amounts_kmol: dict[str, float]
    reaction_extents_kmol_batch: tuple[float, ...]
    outlet_state: object
    heat_per_batch_kJ: float
    profile: tuple[dict, ...]
    profile_states: tuple[object, ...]
    solver_method: str
    attempted_methods: tuple[str, ...]
    evaluations: int
    status: int
    message: str
    material_balance_residuals_kmol_batch: dict[str, float]
    energy_balance_residual_kJ_batch: float
    maximum_working_volume_m3: float


def _positive(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise KineticsError(f"{label} must be positive and finite")
    return number


def _bounded_temperature_guess(value: float) -> float:
    return min(max(float(value), 50.0), 5000.0)


def solve_adaptive_batch(
    feeds: tuple[BatchFeed, ...],
    reactions: tuple[KineticReaction, ...],
    thermo,
    phase: str,
    pressure_bar: float,
    reaction_time_h: float,
    thermal: BatchThermalModel,
    controls: BatchNumericalControls,
    usable_vessel_volume_m3: Optional[float] = None,
) -> AdaptiveBatchSolution:
    """Integrate one representative stirred-vessel reaction period.

    Unscheduled feeds form the initial charge. Scheduled feeds are introduced
    at a constant rate between ``start_h`` and ``stop_h``. Amounts are per
    representative batch; time is in hours.
    """
    if not reactions:
        raise KineticsError("Batch reactor requires at least one kinetic reaction")
    incompatible = [
        reaction.rate_unit for reaction in reactions
        if reaction.rate_output_basis != 'fluid_volume'
    ]
    if incompatible:
        raise KineticsError(
            "Batch reactor requires fluid-volume rate units; incompatible "
            "rate_unit(s): " + ', '.join(incompatible)
        )
    if not feeds:
        raise KineticsError("Batch reactor requires at least one feed")
    phase_name = str(phase).strip().lower()
    if phase_name == 'gas':
        phase_name = 'vapor'
    if phase_name not in {'vapor', 'liquid'}:
        raise KineticsError("Batch reactor requires phase='vapor' or phase='liquid'")
    pressure = _positive(pressure_bar, 'Batch reactor pressure')
    reaction_time = _positive(reaction_time_h, 'Batch reactor reaction time')
    usable_volume = (
        None if usable_vessel_volume_m3 is None
        else _positive(usable_vessel_volume_m3, 'Batch usable vessel volume')
    )
    components = tuple(str(component) for component in thermo.components)

    mode = str(thermal.mode).strip().lower()
    if mode not in {'isothermal', 'adiabatic', 'duty', 'jacketed'}:
        raise KineticsError(
            "Batch thermal mode must be isothermal, adiabatic, duty, or jacketed"
        )
    if mode == 'isothermal':
        _positive(thermal.isothermal_temperature_K, 'Batch isothermal temperature')
    elif mode == 'duty':
        if thermal.specified_energy_kJ_batch is None or not math.isfinite(
            float(thermal.specified_energy_kJ_batch)
        ):
            raise KineticsError("Batch duty mode requires finite Q_batch")
    elif mode == 'jacketed':
        _positive(thermal.UA_W_K, 'Batch UA')
        _positive(thermal.jacket_temperature_K, 'Batch jacket temperature')

    initial_amounts = np.zeros(len(components), dtype=float)
    initial_enthalpy = 0.0
    scheduled_feeds = []
    total_feed_amounts = np.zeros(len(components), dtype=float)
    total_feed_enthalpy = 0.0
    for feed in feeds:
        amounts = np.asarray([
            float(feed.component_amounts_kmol.get(component, 0.0))
            for component in components
        ], dtype=float)
        if np.any(~np.isfinite(amounts)) or np.any(amounts < 0.0):
            raise KineticsError(f"Batch feed {feed.name!r} has invalid component amounts")
        amount = float(np.sum(amounts))
        if amount <= 0.0:
            raise KineticsError(f"Batch feed {feed.name!r} has zero batch amount")
        feed_enthalpy = amount * float(feed.molar_enthalpy_kJ_kmol)
        if not math.isfinite(feed_enthalpy):
            raise KineticsError(f"Batch feed {feed.name!r} has invalid enthalpy")
        total_feed_amounts += amounts
        total_feed_enthalpy += feed_enthalpy
        if feed.scheduled:
            start = 0.0 if feed.start_h is None else float(feed.start_h)
            stop = reaction_time if feed.stop_h is None else float(feed.stop_h)
            if (
                not math.isfinite(start) or not math.isfinite(stop)
                or start < 0.0 or stop <= start or stop > reaction_time
            ):
                raise KineticsError(
                    f"Batch feed {feed.name!r} requires 0 <= start < stop <= t_reaction"
                )
            scheduled_feeds.append((feed.name, amounts, feed_enthalpy, start, stop))
        else:
            initial_amounts += amounts
            initial_enthalpy += feed_enthalpy
    if float(np.sum(initial_amounts)) <= 0.0:
        raise KineticsError(
            "Batch reactor requires a nonzero initial charge; not every inlet may be scheduled"
        )

    stoichiometry = np.asarray([
        [reaction.reaction.stoichiometry.get(component, 0.0) for reaction in reactions]
        for component in components
    ], dtype=float)
    state_solver = _ThermoStateSolver(thermo, 'batch transient state')
    initial_total = float(np.sum(initial_amounts))
    initial_composition = {
        component: float(initial_amounts[index] / initial_total)
        for index, component in enumerate(components)
        if initial_amounts[index] > 0.0
    }
    initial_H = initial_enthalpy / initial_total
    initial_state, initial_error = state_solver.state_at_enthalpy(
        pressure,
        initial_total,
        initial_composition,
        initial_H,
        298.15,
        force_phase=phase_name,
        include=('H',),
    )
    if abs(initial_error) > max(1.0e-5, abs(initial_H) * 1.0e-9):
        raise KineticsError(
            f"Batch initial-charge enthalpy residual is {initial_error:.6g} kJ/kmol"
        )

    enthalpy_index = len(components) if mode != 'isothermal' else None
    extent_start = len(components) + (1 if enthalpy_index is not None else 0)
    heat_index = extent_start + len(reactions) if mode != 'isothermal' else None
    initial_values = list(initial_amounts)
    if enthalpy_index is not None:
        initial_values.append(initial_enthalpy)
    initial_values.extend([0.0] * len(reactions))
    if heat_index is not None:
        initial_values.append(0.0)
    initial_values = np.asarray(initial_values, dtype=float)

    temperature_guess = (
        float(thermal.isothermal_temperature_K)
        if mode == 'isothermal' else float(initial_state.T)
    )

    def feed_rates(time_h: float):
        component_rate = np.zeros(len(components), dtype=float)
        enthalpy_rate = 0.0
        for _name, amounts, feed_enthalpy, start, stop in scheduled_feeds:
            if start <= time_h < stop or (
                math.isclose(time_h, stop, abs_tol=1.0e-12)
                and math.isclose(stop, reaction_time, abs_tol=1.0e-12)
            ):
                duration = stop - start
                component_rate += amounts / duration
                enthalpy_rate += feed_enthalpy / duration
        return component_rate, enthalpy_rate

    def point_from_values(time_h: float, values):
        amounts = np.asarray(values[:len(components)], dtype=float)
        clean = np.maximum(amounts, 0.0)
        total = float(np.sum(clean))
        if total <= 0.0:
            raise KineticsError("Batch trial state has zero total inventory")
        composition = {
            component: float(clean[index] / total)
            for index, component in enumerate(components)
            if clean[index] > 0.0
        }
        if mode == 'isothermal':
            temperature = float(thermal.isothermal_temperature_K)
            stream_state = None
        else:
            enthalpy_target = float(values[enthalpy_index]) / total
            stream_state, enthalpy_error = state_solver.state_at_enthalpy(
                pressure,
                total,
                composition,
                enthalpy_target,
                _bounded_temperature_guess(temperature_guess),
                force_phase=phase_name,
                include=('H',),
            )
            if abs(enthalpy_error) > max(1.0e-5, abs(enthalpy_target) * 1.0e-9):
                raise KineticsError(
                    f"Batch local enthalpy residual is {enthalpy_error:.6g} kJ/kmol"
                )
            temperature = float(stream_state.T)
        rate_state = HomogeneousRateState.from_flows(
            thermo,
            {component: float(clean[index]) for index, component in enumerate(components)},
            temperature,
            pressure,
            phase_name,
        )
        rates = np.asarray([
            evaluate_reaction_rate(reaction, rate_state, thermo)
            for reaction in reactions
        ], dtype=float)
        volume = total / rate_state.molar_density_kmol_m3
        return {
            'amounts': clean,
            'total': total,
            'composition': composition,
            'temperature': temperature,
            'stream_state': stream_state,
            'rate_state': rate_state,
            'rates': rates,
            'volume': volume,
        }

    def heat_rate(point) -> float:
        if mode == 'adiabatic':
            return 0.0
        if mode == 'duty':
            return float(thermal.specified_energy_kJ_batch) / reaction_time
        if mode == 'jacketed':
            return (
                float(thermal.UA_W_K)
                * (float(thermal.jacket_temperature_K) - point['temperature'])
                * 3.6
            )
        return 0.0

    if (
        usable_volume is not None
        and point_from_values(0.0, initial_values)['volume']
        > usable_volume * (1.0 + 1.0e-10)
    ):
        raise KineticsError("initial charge exceeds usable vessel volume")

    def rhs_with_feed(time_h: float, values, component_feed, enthalpy_feed):
        point = point_from_values(time_h, values)
        extent_rates = point['volume'] * point['rates']
        component_rates = component_feed + stoichiometry @ extent_rates
        derivatives = list(component_rates)
        if enthalpy_index is not None:
            q_rate = heat_rate(point)
            derivatives.append(enthalpy_feed + q_rate)
        derivatives.extend(float(rate) for rate in extent_rates)
        if heat_index is not None:
            derivatives.append(q_rate)
        return np.asarray(derivatives, dtype=float)

    relative_tolerance = _positive(
        controls.relative_tolerance, 'Batch relative tolerance'
    )
    absolute_base = _positive(controls.absolute_tolerance, 'Batch absolute tolerance')
    amount_scale = max(float(np.sum(total_feed_amounts)), 1.0)
    absolute_vector = [max(1.0e-12, amount_scale * absolute_base)] * len(components)
    mole_tolerance = max(1.0e-12 * amount_scale, 1.0e-14)
    if enthalpy_index is not None:
        absolute_vector.append(max(1.0e-4, abs(total_feed_enthalpy) * absolute_base))
    absolute_vector.extend(
        [max(1.0e-12, amount_scale * absolute_base)] * len(reactions)
    )
    if heat_index is not None:
        absolute_vector.append(max(1.0e-4, abs(total_feed_enthalpy) * absolute_base))

    requested_method = str(controls.method).strip().upper()
    aliases = {
        'AUTO': 'auto', 'RK45': 'RK45', 'BDF': 'BDF',
        'RADAU': 'Radau', 'LSODA': 'LSODA',
    }
    if requested_method not in aliases:
        raise KineticsError("Batch solver must be auto, RK45, BDF, Radau, or LSODA")
    if requested_method == 'AUTO':
        primary_method = (
            'LSODA'
            if mode != 'isothermal' or len(reactions) > 1
            or any(
                reaction.reversible
                or reaction.model in {'custom', 'custom_net'}
                for reaction in reactions
            )
            else 'RK45'
        )
    else:
        primary_method = aliases[requested_method]
    methods = [primary_method]
    if requested_method == 'AUTO' and primary_method == 'RK45':
        methods.append('LSODA')

    profile_points = int(controls.profile_points)
    if profile_points < 2 or profile_points > 501:
        raise KineticsError("Batch profile_points must be between 2 and 501")
    evaluation_grid = set(np.linspace(0.0, reaction_time, profile_points))
    boundaries = {0.0, reaction_time}
    for _name, _amounts, _enthalpy, start, stop in scheduled_feeds:
        boundaries.update((start, stop))
    evaluation_grid.update(boundaries)
    evaluation_grid = np.asarray(sorted(evaluation_grid), dtype=float)

    maximum_step = (
        math.inf if controls.maximum_step_h is None
        else _positive(controls.maximum_step_h, 'Batch maximum time step')
    )
    first_step = (
        None if controls.first_step_h is None
        else _positive(controls.first_step_h, 'Batch first time step')
    )
    if first_step is not None and first_step > reaction_time:
        raise KineticsError("Batch first step cannot exceed reaction time")

    attempted = []
    failures = []
    solved = None
    for method in methods:
        attempted.append(method)
        try:
            current = initial_values.copy()
            times = []
            columns = []
            evaluations = 0
            segment_boundaries = sorted(boundaries)
            messages = []
            for segment_index, (start, stop) in enumerate(zip(
                segment_boundaries[:-1], segment_boundaries[1:]
            )):
                segment_component_feed, segment_enthalpy_feed = feed_rates(
                    0.5 * (start + stop)
                )

                def segment_rhs(time_h, values):
                    return rhs_with_feed(
                        time_h,
                        values,
                        segment_component_feed,
                        segment_enthalpy_feed,
                    )

                def depleted_inventory(_time_h, values):
                    return float(np.min(values[:len(components)]) + mole_tolerance)

                depleted_inventory.terminal = True
                depleted_inventory.direction = -1.0

                integration_events = [depleted_inventory]
                if usable_volume is not None:
                    def vessel_capacity(_time_h, values):
                        return float(
                            usable_volume
                            - point_from_values(_time_h, values)['volume']
                        )

                    vessel_capacity.terminal = True
                    vessel_capacity.direction = -1.0
                    integration_events.append(vessel_capacity)

                segment_grid = evaluation_grid[
                    (evaluation_grid >= start) & (evaluation_grid <= stop)
                ]
                if segment_index:
                    segment_grid = segment_grid[segment_grid > start]
                result = solve_ivp(
                    segment_rhs,
                    (start, stop),
                    current,
                    method=method,
                    rtol=relative_tolerance,
                    atol=np.asarray(absolute_vector, dtype=float),
                    max_step=maximum_step,
                    first_step=(
                        min(first_step, stop - start)
                        if first_step is not None else None
                    ),
                    t_eval=segment_grid,
                    events=integration_events,
                )
                if not result.success:
                    raise KineticsError(result.message)
                if result.status == 1:
                    hit_index = next(
                        (
                            index for index, event_times in enumerate(result.t_events)
                            if len(event_times)
                        ),
                        0,
                    )
                    event_time = float(result.t_events[hit_index][-1])
                    if hit_index == 0:
                        raise _BatchTerminalEventError(
                            "component inventory depleted before the end of the "
                            f"reaction period at t={event_time:.6g} h"
                        )
                    raise _BatchTerminalEventError(
                        "trajectory exceeds usable vessel volume before the end "
                        f"of the reaction period at t={event_time:.6g} h"
                    )
                evaluations += int(result.nfev)
                messages.append(str(result.message))
                times.extend(float(value) for value in result.t)
                columns.extend(
                    result.y[:, index].copy()
                    for index in range(result.y.shape[1])
                )
                current = np.asarray(result.y[:, -1], dtype=float)
            solved = (
                np.asarray(times, dtype=float),
                np.column_stack(columns),
                current,
                evaluations,
                '; '.join(dict.fromkeys(messages)),
            )
            break
        except _BatchTerminalEventError:
            raise
        except (KineticsError, ValueError, FloatingPointError) as error:
            failures.append(f"{method}: {error}")
    if solved is None:
        raise KineticsError(
            "Batch adaptive integration failed (" + '; '.join(failures) + ')'
        )
    times, values_matrix, outlet_values, evaluations, message = solved

    profile_rows = []
    profile_states = []
    maximum_volume = 0.0
    for column, time_h in enumerate(times):
        values = values_matrix[:, column]
        point = point_from_values(float(time_h), values)
        if mode == 'isothermal':
            stream_state = thermo.calculate_state(
                point['temperature'], pressure, point['total'], point['composition'],
                phase=phase_name, flash=False, include=('H', 'rho'),
            )
        else:
            stream_state = point['stream_state']
            if stream_state.rho is None:
                stream_state = thermo.calculate_state(
                    stream_state.T, stream_state.P, stream_state.F,
                    stream_state.composition, phase=phase_name, flash=False,
                    include=('H', 'rho'),
                )
        charged_enthalpy = initial_enthalpy
        for _name, _amounts, feed_enthalpy, start, stop in scheduled_feeds:
            fraction = min(max((time_h - start) / (stop - start), 0.0), 1.0)
            charged_enthalpy += fraction * feed_enthalpy
        cumulative_heat = (
            point['total'] * stream_state.H - charged_enthalpy
            if mode == 'isothermal' else float(values[heat_index])
        )
        maximum_volume = max(maximum_volume, float(point['volume']))
        profile_states.append(stream_state)
        profile_rows.append({
            'time_h': float(time_h),
            'temperature_K': stream_state.T,
            'temperature_C': stream_state.T - 273.15,
            'pressure_bar': stream_state.P,
            'inventory_kmol': stream_state.F,
            'working_volume_m3': float(point['volume']),
            'composition': dict(stream_state.composition),
            'reaction_rates_kmol_m3_h': [float(rate) for rate in point['rates']],
            'reaction_extents_kmol_batch': [
                float(value) for value in values[extent_start:extent_start + len(reactions)]
            ],
            'cumulative_heat_kJ_batch': float(cumulative_heat),
        })

    outlet_point = point_from_values(reaction_time, outlet_values)
    outlet_state = profile_states[-1]
    outlet_amounts = {
        component: float(outlet_point['amounts'][index])
        for index, component in enumerate(components)
        if outlet_point['amounts'][index] > 0.0
    }
    extents = tuple(
        float(value)
        for value in outlet_values[extent_start:extent_start + len(reactions)]
    )
    calculated_amounts = total_feed_amounts + stoichiometry @ np.asarray(extents)
    material_residuals = {
        component: float(outlet_point['amounts'][index] - calculated_amounts[index])
        for index, component in enumerate(components)
    }
    heat_per_batch = (
        outlet_point['total'] * outlet_state.H - total_feed_enthalpy
        if mode == 'isothermal' else float(outlet_values[heat_index])
    )
    energy_residual = (
        outlet_point['total'] * outlet_state.H
        - total_feed_enthalpy
        - heat_per_batch
    )
    return AdaptiveBatchSolution(
        outlet_component_amounts_kmol=outlet_amounts,
        reaction_extents_kmol_batch=extents,
        outlet_state=outlet_state,
        heat_per_batch_kJ=float(heat_per_batch),
        profile=tuple(profile_rows),
        profile_states=tuple(profile_states),
        solver_method=attempted[-1],
        attempted_methods=tuple(attempted),
        evaluations=evaluations,
        status=0,
        message=message,
        material_balance_residuals_kmol_batch=material_residuals,
        energy_balance_residual_kJ_batch=float(energy_residual),
        maximum_working_volume_m3=float(maximum_volume),
    )
