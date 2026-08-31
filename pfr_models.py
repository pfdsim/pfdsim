"""Shared adaptive axial balances for homogeneous PFR and packed-bed flow."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

import numpy as np

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .axial_solver import (
        AxialEvent,
        AxialIntegrationError,
        AxialSolution,
        AxialSolverOptions,
        integrate_axial,
    )
else:
    from axial_solver import (
        AxialEvent,
        AxialIntegrationError,
        AxialSolution,
        AxialSolverOptions,
        integrate_axial,
    )
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
    from .transport_correlations import single_phase_flow
else:
    from transport_correlations import single_phase_flow
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_basic import _ThermoStateSolver
else:
    from unit_operations_basic import _ThermoStateSolver


@dataclass(frozen=True)
class PFRGeometry:
    """Resolved cylindrical reactor geometry."""

    volume_m3: float
    length_m: Optional[float]
    diameter_m: Optional[float]
    n_tubes: int = 1

    @property
    def flow_area_m2(self) -> Optional[float]:
        if self.diameter_m is None:
            return None
        return self.n_tubes * math.pi * self.diameter_m**2 / 4.0

    @property
    def heat_perimeter_m(self) -> Optional[float]:
        if self.diameter_m is None:
            return None
        return self.n_tubes * math.pi * self.diameter_m

    @property
    def uses_length_coordinate(self) -> bool:
        return self.length_m is not None and self.flow_area_m2 is not None


@dataclass(frozen=True)
class PFRPressureModel:
    model: str = 'none'
    specified_drop_bar: float = 0.0
    void_fraction: Optional[float] = None
    particle_diameter_m: Optional[float] = None
    roughness_m: float = 0.0
    friction_model: str = 'churchill'


@dataclass(frozen=True)
class PFRThermalModel:
    mode: str = 'isothermal'
    isothermal_temperature_K: Optional[float] = None
    specified_duty_kJ_h: Optional[float] = None
    U_W_m2_K: Optional[float] = None
    jacket_temperature_K: Optional[float] = None


@dataclass(frozen=True)
class PFRNumericalControls:
    method: str = 'auto'
    relative_tolerance: float = 1.0e-7
    absolute_tolerance: float = 1.0e-9
    maximum_step: Optional[float] = None
    first_step: Optional[float] = None
    profile_points: int = 31


@dataclass(frozen=True)
class AxialReactionMeasure:
    """Dimensional measure multiplying local rates along an axial coordinate."""

    basis: str
    total: float
    gradient: float


RateModifier = Callable[
    [KineticReaction, float, HomogeneousRateState, float],
    tuple[float, float],
]


@dataclass(frozen=True)
class AdaptiveAxialReactorSolution:
    outlet_component_flows: dict[str, float]
    reaction_extents_kmol_h: tuple[float, ...]
    outlet_state: object
    heat_duty_kJ_h: float
    residence_time_h: float
    profile: tuple[dict, ...]
    profile_states: tuple[object, ...]
    solver_method: str
    attempted_methods: tuple[str, ...]
    evaluations: int
    status: int
    message: str
    material_balance_residuals_kmol_h: dict[str, float]
    energy_balance_residual_kJ_h: float
    depletion_events: tuple[dict, ...] = ()
    warnings: tuple[str, ...] = ()


# Compatibility name retained for callers that imported the original PFR
# solution type before the axial core also served PackedBedReactor.
AdaptivePFRSolution = AdaptiveAxialReactorSolution


def _positive(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise KineticsError(f"{label} must be positive and finite") from error
    if not math.isfinite(number) or number <= 0.0:
        raise KineticsError(f"{label} must be positive and finite")
    return number


def _bounded_temperature_guess(value: float) -> float:
    return max(1.0, min(5000.0, float(value)))


def _solve_adaptive_axial_reactor(
    inlet,
    reactions: tuple[KineticReaction, ...],
    thermo,
    phase: str,
    geometry: PFRGeometry,
    thermal: PFRThermalModel,
    pressure: PFRPressureModel,
    controls: PFRNumericalControls,
    *,
    reaction_measure: Optional[AxialReactionMeasure] = None,
    rate_modifier: Optional[RateModifier] = None,
    reactor_name: str = 'PFR',
    residence_void_fraction: Optional[float] = None,
) -> AdaptiveAxialReactorSolution:
    """Integrate coupled reaction, energy, pressure, and residence-time balances."""
    if not reactions:
        raise KineticsError(f"{reactor_name} requires at least one kinetic reaction")
    phase_name = str(phase).strip().lower()
    if phase_name == 'gas':
        phase_name = 'vapor'
    if phase_name not in {'vapor', 'liquid'}:
        raise KineticsError(
            f"{reactor_name} requires phase='vapor' or phase='liquid'"
        )
    volume = _positive(geometry.volume_m3, f'{reactor_name} volume')
    pressure_model = str(pressure.model).strip().lower().replace('-', '_')
    if pressure_model not in {'none', 'specified', 'darcy', 'ergun'}:
        raise KineticsError(
            f"{reactor_name} pressure model must be none, specified, darcy, or Ergun"
        )
    thermal_mode = str(thermal.mode).strip().lower()
    if thermal_mode not in {'isothermal', 'adiabatic', 'duty', 'jacketed'}:
        raise KineticsError(
            f"{reactor_name} thermal mode must be isothermal, adiabatic, duty, or jacketed"
        )
    if inlet.H is None or not math.isfinite(float(inlet.H)):
        raise KineticsError(f"{reactor_name} inlet enthalpy must be finite")
    if thermal_mode == 'isothermal':
        _positive(
            thermal.isothermal_temperature_K,
            f'{reactor_name} isothermal temperature',
        )
    if thermal_mode == 'duty':
        if (
            thermal.specified_duty_kJ_h is None
            or not math.isfinite(float(thermal.specified_duty_kJ_h))
        ):
            raise KineticsError(
                f"{reactor_name} duty mode requires a finite specified duty"
            )
    if thermal_mode == 'jacketed':
        _positive(thermal.U_W_m2_K, f'{reactor_name} heat-transfer coefficient')
        _positive(thermal.jacket_temperature_K, f'{reactor_name} jacket temperature')
    if pressure_model == 'specified':
        drop = float(pressure.specified_drop_bar)
        if not math.isfinite(drop) or drop <= 0.0 or drop >= float(inlet.P):
            raise KineticsError(
                f"{reactor_name} specified pressure drop must be positive and below inlet pressure"
            )
    if pressure_model == 'darcy':
        if not math.isfinite(float(pressure.roughness_m)) or pressure.roughness_m < 0.0:
            raise KineticsError(
                f"{reactor_name} Darcy roughness must be finite and nonnegative"
            )
    if pressure_model == 'ergun':
        if (
            pressure.void_fraction is None
            or not 0.0 < float(pressure.void_fraction) < 1.0
        ):
            raise KineticsError(
                f"{reactor_name} Ergun void fraction must lie between zero and one"
            )
        _positive(
            pressure.particle_diameter_m,
            f'{reactor_name} Ergun particle diameter',
        )
    geometry_required = pressure_model in {'specified', 'darcy', 'ergun'} or thermal_mode == 'jacketed'
    if geometry_required and not geometry.uses_length_coordinate:
        raise KineticsError(
            "Calculated/distributed pressure drop and jacketed heat transfer "
            "require resolvable reactor length and diameter"
        )

    if geometry.uses_length_coordinate:
        coordinate_end = _positive(geometry.length_m, f'{reactor_name} length')
        volume_gradient = float(geometry.flow_area_m2)
        coordinate_kind = 'length'
    else:
        coordinate_end = volume
        volume_gradient = 1.0
        coordinate_kind = 'volume'

    if reaction_measure is None:
        reaction_measure = AxialReactionMeasure(
            'fluid_volume',
            volume,
            volume_gradient,
        )
    if residence_void_fraction is not None:
        if not 0.0 < float(residence_void_fraction) < 1.0:
            raise KineticsError(
                f"{reactor_name} residence void fraction must lie between zero and one"
            )
    measure_basis = str(reaction_measure.basis).strip().lower()
    if measure_basis not in {'fluid_volume', 'catalyst_mass'}:
        raise KineticsError(
            f"{reactor_name} reaction measure must be fluid_volume or catalyst_mass"
        )
    _positive(reaction_measure.total, f'{reactor_name} reaction measure total')
    measure_gradient = _positive(
        reaction_measure.gradient,
        f'{reactor_name} reaction measure gradient',
    )
    if coordinate_kind == 'length':
        integrated_measure = measure_gradient * coordinate_end
        if abs(integrated_measure / float(reaction_measure.total) - 1.0) > 1.0e-8:
            raise KineticsError(
                f"{reactor_name} reaction measure total and axial gradient are "
                "inconsistent"
            )
    mismatched = [
        reaction.rate_unit for reaction in reactions
        if reaction.rate_output_basis != measure_basis
    ]
    if mismatched:
        expected = (
            'catalyst-mass' if measure_basis == 'catalyst_mass' else 'fluid-volume'
        )
        raise KineticsError(
            f"{reactor_name} requires {expected} rate units; incompatible rate_unit(s): "
            + ', '.join(mismatched)
        )

    components = tuple(str(component) for component in thermo.components)
    inlet_flows = {
        component: float(inlet.component_flows().get(component, 0.0))
        for component in components
    }
    if any(not math.isfinite(value) or value < 0.0 for value in inlet_flows.values()):
        raise KineticsError(
            f"{reactor_name} inlet component flows must be finite and nonnegative"
        )
    n0 = np.asarray([inlet_flows[component] for component in components], dtype=float)
    stoichiometry = np.asarray([
        [reaction.reaction.stoichiometry.get(component, 0.0) for reaction in reactions]
        for component in components
    ], dtype=float)
    flow_scale = max(float(np.sum(n0)), 1.0)
    mole_tolerance = max(1.0e-12 * flow_scale, 1.0e-14)
    inlet_enthalpy_flow = float(inlet.F * inlet.H)
    state_solver = _ThermoStateSolver(thermo, f'{reactor_name} axial state')

    pressure_index = len(reactions) if pressure_model in {'darcy', 'ergun'} else None
    enthalpy_index = (
        len(reactions) + (1 if pressure_index is not None else 0)
        if thermal_mode != 'isothermal' else None
    )
    residence_index = (
        len(reactions)
        + (1 if pressure_index is not None else 0)
        + (1 if enthalpy_index is not None else 0)
    )
    initial_values = [0.0] * len(reactions)
    if pressure_index is not None:
        initial_values.append(float(inlet.P))
    if enthalpy_index is not None:
        initial_values.append(inlet_enthalpy_flow)
    initial_values.append(0.0)

    def flows_for(extents) -> np.ndarray:
        return n0 + stoichiometry @ np.asarray(extents, dtype=float)

    def pressure_for(coordinate: float, values) -> float:
        if pressure_index is not None:
            return float(values[pressure_index])
        if pressure_model == 'specified':
            return float(inlet.P - pressure.specified_drop_bar * coordinate / coordinate_end)
        return float(inlet.P)

    def enthalpy_flow_for(values) -> float:
        if enthalpy_index is None:
            raise KineticsError(
                f"Isothermal {reactor_name} has no integrated enthalpy state"
            )
        return float(values[enthalpy_index])

    def point_from_values(coordinate: float, values, *, pressure_override=None):
        extents = np.asarray(values[:len(reactions)], dtype=float)
        flows = flows_for(extents)
        clean = np.maximum(flows, 0.0)
        total = float(np.sum(clean))
        if total <= 0.0:
            raise KineticsError(
                f"{reactor_name} trial state has zero total molar flow"
            )
        composition = {
            component: float(clean[index] / total)
            for index, component in enumerate(components)
        }
        local_pressure = (
            float(pressure_override)
            if pressure_override is not None else pressure_for(coordinate, values)
        )
        if not math.isfinite(local_pressure) or local_pressure <= 1.0e-6:
            raise KineticsError(f"{reactor_name} local pressure is nonpositive")
        if thermal_mode == 'isothermal':
            temperature = float(thermal.isothermal_temperature_K)
            stream_state = None
        else:
            enthalpy_target = enthalpy_flow_for(values) / total
            stream_state, enthalpy_error = state_solver.state_at_enthalpy(
                local_pressure,
                total,
                composition,
                enthalpy_target,
                _bounded_temperature_guess(inlet.T),
                force_phase=phase_name,
                include=('H',),
            )
            if abs(enthalpy_error) > max(1.0e-5, abs(enthalpy_target) * 1.0e-9):
                raise KineticsError(
                    f"{reactor_name} local enthalpy solve residual is "
                    f"{enthalpy_error:.6g} kJ/kmol"
                )
            temperature = float(stream_state.T)
        rate_state = HomogeneousRateState.from_flows(
            thermo,
            {component: float(clean[index]) for index, component in enumerate(components)},
            temperature,
            local_pressure,
            phase_name,
        )
        intrinsic_rates = np.asarray([
            evaluate_reaction_rate(reaction, rate_state, thermo)
            for reaction in reactions
        ], dtype=float)
        effectiveness_factors = np.ones(len(reactions), dtype=float)
        rates = intrinsic_rates.copy()
        if rate_modifier is not None:
            modified = [
                rate_modifier(
                    reaction, float(rate), rate_state, float(coordinate)
                )
                for reaction, rate in zip(reactions, intrinsic_rates)
            ]
            rates = np.asarray([item[0] for item in modified], dtype=float)
            effectiveness_factors = np.asarray(
                [item[1] for item in modified], dtype=float
            )
        if np.any(~np.isfinite(rates)) or np.any(~np.isfinite(effectiveness_factors)):
            raise KineticsError(
                f"{reactor_name} rate modifier returned non-finite values"
            )
        # At a nonnegative-inventory boundary, suppress only reaction
        # directions that would consume an absent component. Producing and
        # independent reactions remain active, and a suppressed direction can
        # restart after another reaction creates positive inventory.
        boundary_components = np.flatnonzero(flows <= mole_tolerance)
        if len(boundary_components):
            rates = rates.copy()
            maximum_active_set_iterations = max(
                4,
                len(boundary_components) * max(1, len(reactions)) * 2,
            )
            for _iteration in range(maximum_active_set_iterations):
                changed = False
                for component_index in boundary_components:
                    contributions = (
                        stoichiometry[component_index, :] * rates
                    )
                    production = float(np.sum(contributions[contributions > 0.0]))
                    consumption = float(-np.sum(contributions[contributions < 0.0]))
                    if consumption <= production + 1.0e-15:
                        continue
                    factor = (
                        0.0 if production <= 0.0 else production / consumption
                    )
                    consuming = contributions < 0.0
                    rates[consuming] *= factor
                    changed = True
                if not changed:
                    break
            else:
                raise KineticsError(
                    f"{reactor_name} inventory active-set rate projection "
                    "did not converge"
                )
        return {
            'raw_flows': flows,
            'flows': clean,
            'total': total,
            'composition': composition,
            'pressure': local_pressure,
            'temperature': temperature,
            'stream_state': stream_state,
            'rate_state': rate_state,
            'intrinsic_rates': intrinsic_rates,
            'rates': rates,
            'effectiveness_factors': effectiveness_factors,
        }

    def heat_derivative(point) -> float:
        if thermal_mode == 'adiabatic':
            return 0.0
        if thermal_mode == 'duty':
            return float(thermal.specified_duty_kJ_h) / coordinate_end
        if thermal_mode == 'jacketed':
            return (
                float(thermal.U_W_m2_K)
                * float(geometry.heat_perimeter_m)
                * (float(thermal.jacket_temperature_K) - point['temperature'])
                * 3.6
            )
        return 0.0

    inlet_mass_flow_kg_s = float(inlet.mass_flow()) / 3600.0

    def hydraulic_values(point):
        density_molar = point['rate_state'].molar_density_kmol_m3
        mixture_mw = float(thermo.mixture_MW(point['composition']))
        density_mass = density_molar * mixture_mw
        viscosity = float(thermo.mixture_viscosity(
            point['composition'],
            point['temperature'],
            point['pressure'],
            1.0 if phase_name == 'vapor' else 0.0,
            x=point['composition'] if phase_name == 'liquid' else None,
            y=point['composition'] if phase_name == 'vapor' else None,
        ))
        area = float(geometry.flow_area_m2)
        if inlet_mass_flow_kg_s <= 0.0:
            raise KineticsError(
                f"{reactor_name} hydraulic model requires positive inlet mass flow"
            )
        velocity = inlet_mass_flow_kg_s / density_mass / area
        if pressure_model == 'darcy':
            flow = single_phase_flow(
                inlet_mass_flow_kg_s / geometry.n_tubes,
                density_mass,
                viscosity,
                float(geometry.diameter_m),
                roughness_m=float(pressure.roughness_m),
                friction_model=pressure.friction_model,
            )
            friction_gradient = float(flow.pressure_gradient.friction)
            reynolds = float(flow.reynolds_number)
            friction_factor = float(flow.darcy_friction_factor)
            velocity = float(flow.velocity_m_s)
        else:
            epsilon = float(pressure.void_fraction)
            particle = float(pressure.particle_diameter_m)
            friction_gradient = (
                150.0 * viscosity * (1.0 - epsilon) ** 2
                / (epsilon**3 * particle**2) * velocity
                + 1.75 * density_mass * (1.0 - epsilon)
                / (epsilon**3 * particle) * velocity**2
            )
            reynolds = density_mass * abs(velocity) * particle / viscosity
            friction_factor = None
        return {
            'density_kg_m3': density_mass,
            'viscosity_Pa_s': viscosity,
            'velocity_m_s': velocity,
            'friction_gradient_Pa_m': friction_gradient,
            'reynolds_number': reynolds,
            'darcy_friction_factor': friction_factor,
        }

    def balances(coordinate: float, values):
        point = point_from_values(coordinate, values)
        extent_derivatives = measure_gradient * point['rates']
        derivatives = list(float(value) for value in extent_derivatives)
        enthalpy_derivative = heat_derivative(point)

        if pressure_index is not None:
            hydraulic = hydraulic_values(point)
            ds = max(1.0e-7, min(1.0e-3, coordinate_end * 1.0e-5))
            projected = np.asarray(values, dtype=float).copy()
            projected[:len(reactions)] += extent_derivatives * ds
            if enthalpy_index is not None:
                projected[enthalpy_index] += enthalpy_derivative * ds
            projected_point = point_from_values(
                min(coordinate + ds, coordinate_end),
                projected,
                pressure_override=point['pressure'],
            )
            projected_hydraulic = hydraulic_values(projected_point)
            velocity_s = (
                projected_hydraulic['velocity_m_s'] - hydraulic['velocity_m_s']
            ) / ds
            dp_bar = max(1.0e-5, abs(point['pressure']) * 1.0e-6)
            p_low = max(1.0e-6, point['pressure'] - dp_bar)
            p_high = point['pressure'] + dp_bar
            low_hydraulic = hydraulic_values(point_from_values(
                coordinate, values, pressure_override=p_low
            ))
            high_hydraulic = hydraulic_values(point_from_values(
                coordinate, values, pressure_override=p_high
            ))
            velocity_p_pa = (
                high_hydraulic['velocity_m_s'] - low_hydraulic['velocity_m_s']
            ) / ((p_high - p_low) * 1.0e5)
            denominator = (
                1.0
                + hydraulic['density_kg_m3']
                * hydraulic['velocity_m_s']
                * velocity_p_pa
            )
            if not math.isfinite(denominator) or denominator <= 1.0e-6:
                raise KineticsError(
                    f"{reactor_name} pressure balance approached an "
                    "acceleration/choking singularity"
                )
            acceleration_numerator = (
                hydraulic['density_kg_m3']
                * hydraulic['velocity_m_s']
                * velocity_s
            )
            pressure_derivative_bar_m = -(
                hydraulic['friction_gradient_Pa_m'] + acceleration_numerator
            ) / denominator / 1.0e5
            derivatives.append(pressure_derivative_bar_m)

        if enthalpy_index is not None:
            derivatives.append(enthalpy_derivative)
        volumetric_flow_m3_h = (
            point['total'] / point['rate_state'].molar_density_kmol_m3
        )
        if residence_void_fraction is not None:
            residence_volume_gradient = (
                volume_gradient * float(residence_void_fraction)
            )
        elif pressure_model == 'ergun':
            residence_volume_gradient = (
                volume_gradient * float(pressure.void_fraction)
            )
        else:
            residence_volume_gradient = volume_gradient
        derivatives.append(residence_volume_gradient / volumetric_flow_m3_h)
        return tuple(derivatives)

    profile_points = int(controls.profile_points)
    if profile_points < 2 or profile_points > 501:
        raise KineticsError(
            f"{reactor_name} profile_points must be between 2 and 501"
        )
    evaluation_grid = tuple(
        coordinate_end * index / (profile_points - 1)
        for index in range(profile_points)
    )
    relative_tolerance = _positive(
        controls.relative_tolerance, f'{reactor_name} relative tolerance'
    )
    absolute_base = _positive(
        controls.absolute_tolerance, f'{reactor_name} absolute tolerance'
    )
    absolute_vector = [max(1.0e-12, flow_scale * absolute_base)] * len(reactions)
    if pressure_index is not None:
        absolute_vector.append(max(1.0e-10, inlet.P * absolute_base))
    if enthalpy_index is not None:
        absolute_vector.append(max(1.0e-4, abs(inlet_enthalpy_flow) * absolute_base))
    absolute_vector.append(max(1.0e-12, absolute_base))

    requested_method = str(controls.method).strip().upper()
    method_aliases = {
        'AUTO': 'auto', 'RK45': 'RK45', 'BDF': 'BDF',
        'RADAU': 'Radau', 'LSODA': 'LSODA',
    }
    if requested_method not in method_aliases:
        raise KineticsError(
            f"{reactor_name} solver must be auto, RK45, BDF, Radau, or LSODA"
        )
    if requested_method == 'AUTO':
        primary_method = (
            'LSODA'
            if thermal_mode != 'isothermal'
            or len(reactions) > 1
            or any(
                reaction.reversible
                or reaction.model in {'custom', 'custom_net'}
                for reaction in reactions
            )
            else 'RK45'
        )
    else:
        primary_method = method_aliases[requested_method]
    methods = [primary_method]
    if requested_method == 'AUTO' and primary_method == 'RK45':
        methods.append('LSODA')

    attempted = []
    solution = None
    depletion_records = ()
    failures = []

    def project_inventory_boundary(values, component_indices):
        corrected = np.asarray(values, dtype=float).copy()
        indices = tuple(sorted(set(int(index) for index in component_indices)))
        if not indices:
            return corrected
        matrix = stoichiometry[np.asarray(indices, dtype=int), :]
        target = -flows_for(corrected[:len(reactions)])[list(indices)]
        extent_correction, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        corrected[:len(reactions)] += extent_correction
        residual = flows_for(corrected[:len(reactions)])[list(indices)]
        if np.max(np.abs(residual), initial=0.0) > max(100.0 * mole_tolerance, 1.0e-12):
            raise AxialIntegrationError(
                f"{reactor_name} could not project a depletion event onto the "
                "nonnegative inventory boundary"
            )
        return corrected

    def integrate_with_depletion_segments(method: str):
        current_coordinate = 0.0
        current_values = np.asarray(initial_values, dtype=float)
        positions = np.empty(0, dtype=float)
        values = np.empty((len(initial_values), 0), dtype=float)
        event_positions: dict[str, list[float]] = {}
        records: list[dict] = []
        evaluations = 0
        final_status = 0
        final_message = 'The solver successfully reached the outlet.'
        maximum_segments = max(8, 4 * len(components) + 4)

        for _segment_index in range(maximum_segments):
            remaining = coordinate_end - current_coordinate
            if remaining <= max(1.0e-12, coordinate_end * 1.0e-12):
                break
            start_flows = flows_for(current_values[:len(reactions)])
            segment_events = []
            for component_index, flow in enumerate(start_flows):
                if flow <= mole_tolerance:
                    continue
                segment_events.append(AxialEvent(
                    f'component_depletion::{component_index}',
                    lambda _coordinate, state, index=component_index: float(
                        flows_for(state[:len(reactions)])[index] - mole_tolerance
                    ),
                    terminal=True,
                    direction=-1.0,
                ))
            if pressure_index is not None:
                segment_events.append(AxialEvent(
                    'minimum_pressure',
                    lambda _coordinate, state: float(
                        state[pressure_index] - 1.0e-4
                    ),
                    terminal=True,
                    direction=-1.0,
                ))
            local_grid = tuple(
                max(0.0, coordinate - current_coordinate)
                for coordinate in evaluation_grid
                if coordinate >= current_coordinate - 1.0e-12
            )
            segment = integrate_axial(
                lambda local_coordinate, state: balances(
                    current_coordinate + local_coordinate,
                    state,
                ),
                remaining,
                current_values,
                options=AxialSolverOptions(
                    method=method,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=tuple(absolute_vector),
                    maximum_step_m=controls.maximum_step,
                    first_step_m=(
                        controls.first_step
                        if current_coordinate == 0.0 else None
                    ),
                ),
                events=tuple(segment_events),
                evaluation_positions_m=local_grid,
            )
            evaluations += segment.evaluations
            global_positions = current_coordinate + segment.position_m
            if positions.size:
                keep = global_positions > positions[-1] + 1.0e-12
                positions = np.concatenate((positions, global_positions[keep]))
                values = np.column_stack((values, segment.values[:, keep]))
            else:
                positions = global_positions.copy()
                values = segment.values.copy()
            for name, local_positions in segment.event_positions_m.items():
                if len(local_positions):
                    event_positions.setdefault(name, []).extend(
                        float(current_coordinate + position)
                        for position in local_positions
                    )

            current_values = segment.outlet_values
            next_coordinate = current_coordinate + segment.outlet_position_m
            if not segment.terminated_by_event:
                current_coordinate = next_coordinate
                final_message = segment.message
                break
            if len(segment.event_positions_m.get('minimum_pressure', ())):
                current_coordinate = next_coordinate
                final_status = 1
                final_message = segment.message
                break

            explicit_hits = {
                int(name.split('::', 1)[1])
                for name, hit_positions in segment.event_positions_m.items()
                if name.startswith('component_depletion::') and len(hit_positions)
            }
            event_flows = flows_for(current_values[:len(reactions)])
            simultaneous_hits = {
                index for index, (start_flow, event_flow) in enumerate(
                    zip(start_flows, event_flows)
                )
                if start_flow > mole_tolerance
                and event_flow <= 10.0 * mole_tolerance
            }
            depleted_indices = tuple(sorted(explicit_hits | simultaneous_hits))
            if not depleted_indices:
                raise AxialIntegrationError(
                    f"{reactor_name} stopped at an unidentified inventory event"
                )
            current_values = project_inventory_boundary(
                current_values,
                depleted_indices,
            )
            if values.size:
                values[:, -1] = current_values
            measure_at_event = (
                next_coordinate * measure_gradient
                if coordinate_kind == 'length'
                else next_coordinate / coordinate_end * float(reaction_measure.total)
            )
            records.append({
                'coordinate': float(next_coordinate),
                'coordinate_kind': coordinate_kind,
                'components': tuple(components[index] for index in depleted_indices),
                'reaction_measure_at_event': float(measure_at_event),
                'remaining_reaction_measure': max(
                    0.0,
                    float(reaction_measure.total) - float(measure_at_event),
                ),
                'reaction_measure_basis': measure_basis,
            })
            if next_coordinate <= current_coordinate + max(
                1.0e-12,
                coordinate_end * 1.0e-12,
            ):
                raise AxialIntegrationError(
                    f"{reactor_name} depletion continuation made no axial progress"
                )
            current_coordinate = next_coordinate
        else:
            raise AxialIntegrationError(
                f"{reactor_name} exceeded the maximum number of depletion segments"
            )

        return AxialSolution(
            position_m=positions,
            values=values,
            event_positions_m={
                name: np.asarray(hit_positions, dtype=float)
                for name, hit_positions in event_positions.items()
            },
            evaluations=evaluations,
            status=final_status,
            message=final_message,
        ), tuple(records)

    for method in methods:
        attempted.append(method)
        try:
            solution, depletion_records = integrate_with_depletion_segments(
                method,
            )
            break
        except (AxialIntegrationError, ValueError) as error:
            failures.append(f"{method}: {error}")
    if solution is None:
        raise KineticsError(
            f"{reactor_name} adaptive integration failed ("
            + '; '.join(failures) + ')'
        )
    if solution.terminated_by_event:
        hits = [
            name for name, positions in solution.event_positions_m.items()
            if len(positions)
        ]
        raise KineticsError(
            f"{reactor_name} integration terminated before the outlet due to "
            + ', '.join(hits)
            + f" at coordinate {solution.outlet_position_m:.6g}"
        )

    profile_rows = []
    profile_states = []
    for column, coordinate in enumerate(solution.position_m):
        values = solution.values[:, column]
        point = point_from_values(float(coordinate), values)
        if thermal_mode == 'isothermal':
            stream_state = thermo.calculate_state(
                point['temperature'],
                point['pressure'],
                point['total'],
                point['composition'],
                phase=phase_name,
                flash=False,
                include=('H', 'rho'),
            )
        else:
            stream_state = point['stream_state']
            if stream_state.rho is None:
                stream_state = thermo.calculate_state(
                    stream_state.T,
                    stream_state.P,
                    stream_state.F,
                    stream_state.composition,
                    phase=phase_name,
                    flash=False,
                    include=('H', 'rho'),
                )
        profile_states.append(stream_state)
        cumulative_volume = (
            float(coordinate) * volume_gradient
            if coordinate_kind == 'length' else float(coordinate)
        )
        cumulative_measure = (
            float(coordinate) * measure_gradient
            if coordinate_kind == 'length'
            else float(coordinate) / coordinate_end * float(reaction_measure.total)
        )
        row = {
            'reactor_volume_m3': cumulative_volume,
            'temperature_K': stream_state.T,
            'temperature_C': stream_state.T - 273.15,
            'pressure_bar': stream_state.P,
            'total_flow_kmol_h': stream_state.F,
            'residence_time_h': float(values[residence_index]),
            'composition': dict(stream_state.composition),
            'reaction_measure_basis': measure_basis,
            'reaction_rates': [float(rate) for rate in point['rates']],
            'intrinsic_reaction_rates': [
                float(rate) for rate in point['intrinsic_rates']
            ],
            'effectiveness_factors': [
                float(value) for value in point['effectiveness_factors']
            ],
        }
        if measure_basis == 'fluid_volume':
            row['reaction_rates_kmol_m3_h'] = list(row['reaction_rates'])
        else:
            row['catalyst_mass_kg'] = cumulative_measure
            row['reaction_rates_kmol_kg_cat_h'] = list(row['reaction_rates'])
            row['intrinsic_reaction_rates_kmol_kg_cat_h'] = list(
                row['intrinsic_reaction_rates']
            )
        if coordinate_kind == 'length':
            row['position_m'] = float(coordinate)
        if pressure_model in {'darcy', 'ergun'}:
            hydraulic = hydraulic_values(point)
            row.update(hydraulic)
            stream_state.rho = hydraulic['density_kg_m3']
            stream_state.mu = hydraulic['viscosity_Pa_s']
        profile_rows.append(row)

    outlet_values = solution.outlet_values
    outlet_point = point_from_values(coordinate_end, outlet_values)
    outlet_state = profile_states[-1]
    outlet_flows = {
        component: float(outlet_point['flows'][index])
        for index, component in enumerate(components)
        if outlet_point['flows'][index] > mole_tolerance
    }
    extents = tuple(float(value) for value in outlet_values[:len(reactions)])
    calculated_flows = n0 + stoichiometry @ np.asarray(extents, dtype=float)
    material_residuals = {
        component: float(
            outlet_state.component_flows().get(component, 0.0)
            - calculated_flows[index]
        )
        for index, component in enumerate(components)
    }
    if thermal_mode == 'isothermal':
        heat_duty = outlet_state.F * outlet_state.H - inlet_enthalpy_flow
    else:
        heat_duty = float(outlet_values[enthalpy_index] - inlet_enthalpy_flow)
    energy_residual = (
        outlet_state.F * outlet_state.H - inlet_enthalpy_flow - heat_duty
    )
    finalized_depletion_records = []
    depletion_warnings = []
    for record in depletion_records:
        finalized = dict(record)
        coordinate = float(record['coordinate'])
        downstream_rows = [
            row for row in profile_rows
            if float(
                row.get('position_m', row['reactor_volume_m3'])
            ) >= coordinate - 1.0e-12
        ]
        rate_scale = max(
            (
                abs(rate)
                for row in profile_rows
                for rate in row['reaction_rates']
            ),
            default=1.0,
        )
        all_inactive = bool(downstream_rows) and all(
            abs(rate) <= max(1.0e-14, rate_scale * 1.0e-10)
            for row in downstream_rows
            for rate in row['reaction_rates']
        )
        finalized['all_reactions_inactive_downstream'] = all_inactive
        remaining = float(record['remaining_reaction_measure'])
        if measure_basis == 'catalyst_mass':
            finalized['remaining_catalyst_mass_kg'] = remaining
            remaining_text = f'{remaining:.6g} kg catalyst'
        else:
            finalized['remaining_reactor_volume_m3'] = remaining
            remaining_text = f'{remaining:.6g} m3 reactor volume'
        component_text = ', '.join(record['components'])
        coordinate_label = 'z' if coordinate_kind == 'length' else 'V'
        warning = (
            f"{reactor_name} component inventory depleted for {component_text} "
            f"at {coordinate_label}={coordinate:.6g}; reaction directions "
            "consuming absent components were suppressed while hydraulic and "
            "thermal balances continued to the outlet"
        )
        if all_inactive and remaining > 0.0:
            warning += f'; all reactions are inactive through the remaining {remaining_text}'
        depletion_warnings.append(warning)
        finalized_depletion_records.append(finalized)
    return AdaptiveAxialReactorSolution(
        outlet_component_flows=outlet_flows,
        reaction_extents_kmol_h=extents,
        outlet_state=outlet_state,
        heat_duty_kJ_h=heat_duty,
        residence_time_h=float(outlet_values[residence_index]),
        profile=tuple(profile_rows),
        profile_states=tuple(profile_states),
        solver_method=attempted[-1],
        attempted_methods=tuple(attempted),
        evaluations=solution.evaluations,
        status=solution.status,
        message=solution.message,
        material_balance_residuals_kmol_h=material_residuals,
        energy_balance_residual_kJ_h=float(energy_residual),
        depletion_events=tuple(finalized_depletion_records),
        warnings=tuple(depletion_warnings),
    )


def solve_adaptive_pfr(
    inlet,
    reactions: tuple[KineticReaction, ...],
    thermo,
    phase: str,
    geometry: PFRGeometry,
    thermal: PFRThermalModel,
    pressure: PFRPressureModel,
    controls: PFRNumericalControls,
) -> AdaptiveAxialReactorSolution:
    """Solve a fluid-volume-rate plug-flow reactor."""
    return _solve_adaptive_axial_reactor(
        inlet,
        reactions,
        thermo,
        phase,
        geometry,
        thermal,
        pressure,
        controls,
        reactor_name='PFR',
    )


def solve_adaptive_packed_bed(
    inlet,
    reactions: tuple[KineticReaction, ...],
    thermo,
    phase: str,
    geometry: PFRGeometry,
    thermal: PFRThermalModel,
    pressure: PFRPressureModel,
    controls: PFRNumericalControls,
    *,
    reaction_measure: AxialReactionMeasure,
    rate_modifier: Optional[RateModifier],
    residence_void_fraction: float,
) -> AdaptiveAxialReactorSolution:
    """Solve a fixed-bed reactor with catalyst-mass-rate kinetics."""
    return _solve_adaptive_axial_reactor(
        inlet,
        reactions,
        thermo,
        phase,
        geometry,
        thermal,
        pressure,
        controls,
        reaction_measure=reaction_measure,
        rate_modifier=rate_modifier,
        reactor_name='PackedBedReactor',
        residence_void_fraction=residence_void_fraction,
    )
