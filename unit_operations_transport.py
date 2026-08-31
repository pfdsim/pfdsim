"""Distributed transport unit operations."""

from dataclasses import dataclass
import math
from typing import Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .axial_solver import (
        AxialEvent,
        AxialSolverOptions,
        integrate_axial,
    )
else:
    from axial_solver import (
        AxialEvent,
        AxialSolverOptions,
        integrate_axial,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics import StreamState
else:
    from thermodynamics import StreamState
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .transport_correlations import (
        CircularConduitGeometry,
        TwoPhaseFlow,
        beggs_brill_two_phase_flow,
        single_phase_flow,
    )
else:
    from transport_correlations import (
        CircularConduitGeometry,
        TwoPhaseFlow,
        beggs_brill_two_phase_flow,
        single_phase_flow,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_base import UnitOperation, UnitOperationError, UnitResult
else:
    from unit_operations_base import UnitOperation, UnitOperationError, UnitResult
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_basic import _ThermoStateSolver
else:
    from unit_operations_basic import _ThermoStateSolver


PIPE_MATERIAL_ROUGHNESS_M = {
    'smooth': 0.0,
    'glass': 1.5e-6,
    'plastic': 1.5e-6,
    'pvc': 1.5e-6,
    'copper': 1.5e-6,
    'drawn_tubing': 1.5e-6,
    'stainless_steel': 1.5e-5,
    'commercial_steel': 4.5e-5,
    'carbon_steel': 4.5e-5,
    'steel': 4.5e-5,
    'wrought_iron': 4.5e-5,
    'galvanized_steel': 1.5e-4,
    'galvanized_iron': 1.5e-4,
    'cast_iron': 2.6e-4,
    'concrete': 3.0e-4,
}


class _TwoPhasePipeFlow(RuntimeError):
    pass


@dataclass(frozen=True)
class _PipeLocalState:
    state: StreamState
    phase: str
    density_kg_m3: float
    viscosity_pa_s: float | None
    surface_tension_n_m: float | None
    mass_quality: float | None
    velocity_m_s: float
    flow: object | None
    energy_residual_j_kg: float
    energy_iterations: int
    energy_iteration_residual_kj_per_kmol: float
    energy_iteration_converged: bool


class Pipe(UnitOperation):
    """Adiabatic one-dimensional pipe pressure-drop calculation."""

    _MINIMUM_PRESSURE_BAR = 1e-4

    _LENGTH_UNITS = {
        'm': 1.0,
        'meter': 1.0,
        'meters': 1.0,
        'metre': 1.0,
        'metres': 1.0,
        'mm': 1e-3,
        'cm': 1e-2,
        'km': 1e3,
        'ft': 0.3048,
        'foot': 0.3048,
        'feet': 0.3048,
        'in': 0.0254,
        'inch': 0.0254,
        'inches': 0.0254,
    }
    _VELOCITY_UNITS = {
        'm/s': 1.0,
        'm/sec': 1.0,
        'mps': 1.0,
        'ft/s': 0.3048,
        'ft/sec': 0.3048,
        'km/h': 1.0 / 3.6,
        'km/hr': 1.0 / 3.6,
    }

    def _parameter(self, names: tuple[str, ...]):
        matches = []
        lowered = {name.lower() for name in names}
        for key, value in self.params.items():
            if str(key).startswith('__unit__'):
                continue
            if str(key).lower() in lowered:
                matches.append((str(key), value, self.params.get(f'__unit__{key}')))
        if len(matches) > 1:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' received duplicate aliases for {names[0]}: "
                + ', '.join(match[0] for match in matches)
            )
        return matches[0] if matches else None

    @staticmethod
    def _normalized_unit(unit: Optional[str]) -> str:
        return str(unit or '').strip().lower().replace(' ', '')

    def _length_value(self, entry, label: str) -> float:
        name, raw, unit = entry
        value = float(raw)
        normalized = self._normalized_unit(unit)
        if normalized:
            factor = self._LENGTH_UNITS.get(normalized)
            if factor is None:
                raise UnitOperationError(
                    f"Pipe '{self.unit_id}' does not recognize {label} unit {unit!r}"
                )
            value *= factor
        if not math.isfinite(value) or value <= 0.0:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' requires positive {name}"
            )
        return value

    def _signed_length_value(self, entry, label: str) -> float:
        name, raw, unit = entry
        value = float(raw)
        normalized = self._normalized_unit(unit)
        if normalized:
            factor = self._LENGTH_UNITS.get(normalized)
            if factor is None:
                raise UnitOperationError(
                    f"Pipe '{self.unit_id}' does not recognize {label} unit {unit!r}"
                )
            value *= factor
        if not math.isfinite(value):
            raise UnitOperationError(f"Pipe '{self.unit_id}' requires finite {name}")
        return value

    def _velocity_value(self, entry) -> float:
        name, raw, unit = entry
        value = float(raw)
        normalized = self._normalized_unit(unit)
        if normalized:
            factor = self._VELOCITY_UNITS.get(normalized)
            if factor is None:
                raise UnitOperationError(
                    f"Pipe '{self.unit_id}' does not recognize velocity unit {unit!r}"
                )
            value *= factor
        if not math.isfinite(value) or value <= 0.0:
            raise UnitOperationError(f"Pipe '{self.unit_id}' requires positive {name}")
        return value

    def _length(self) -> float:
        entry = self._parameter(('length', 'pipe_length', 'L'))
        if entry is None:
            raise UnitOperationError(f"Pipe '{self.unit_id}' requires length")
        return self._length_value(entry, 'length')

    def _elevation(self, length_m: float) -> tuple[float, float, str]:
        orientation = self._parameter(('orientation', 'direction'))
        angle = self._parameter(('angle', 'inclination', 'inclination_angle'))
        elevation = self._parameter(('elevation_change', 'delta_z', 'elevation'))
        supplied = sum(item is not None for item in (orientation, angle, elevation))
        if supplied > 1:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' can specify only one of orientation, angle, "
                "or elevation_change"
            )
        if elevation is not None:
            delta_z = self._signed_length_value(elevation, 'elevation change')
            if abs(delta_z) > length_m + 1e-12:
                raise UnitOperationError(
                    f"Pipe '{self.unit_id}' elevation change cannot exceed its length"
                )
            slope = max(-1.0, min(1.0, delta_z / length_m))
            return slope, math.degrees(math.asin(slope)), 'elevation_change'
        if angle is not None:
            name, raw, unit = angle
            angle_value = float(raw)
            normalized = self._normalized_unit(unit)
            if normalized in ('rad', 'radian', 'radians'):
                angle_degrees = math.degrees(angle_value)
            elif normalized in ('', 'deg', 'degree', 'degrees'):
                angle_degrees = angle_value
            else:
                raise UnitOperationError(
                    f"Pipe '{self.unit_id}' does not recognize angle unit {unit!r}"
                )
            if not math.isfinite(angle_degrees) or abs(angle_degrees) > 90.0:
                raise UnitOperationError(
                    f"Pipe '{self.unit_id}' requires {name} between -90 and 90 degrees"
                )
            return math.sin(math.radians(angle_degrees)), angle_degrees, 'angle'
        if orientation is None:
            return 0.0, 0.0, 'horizontal_default'
        value = str(orientation[1]).strip().lower().replace('-', '_').replace(' ', '_')
        slopes = {
            'horizontal': 0.0,
            'level': 0.0,
            'vertical': 1.0,
            'vertical_up': 1.0,
            'up': 1.0,
            'uphill': 1.0,
            'vertical_down': -1.0,
            'down': -1.0,
            'downhill': -1.0,
        }
        if value not in slopes:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' unknown orientation {orientation[1]!r}"
            )
        slope = slopes[value]
        return slope, math.degrees(math.asin(slope)), value

    def _roughness(self, warnings: list[str]) -> tuple[float, str, str]:
        explicit = self._parameter(('roughness', 'absolute_roughness', 'epsilon'))
        material_entry = self._parameter(('material', 'pipe_material'))
        material = 'commercial_steel' if material_entry is None else str(material_entry[1])
        material_key = material.strip().lower().replace('-', '_').replace(' ', '_')
        if explicit is not None:
            if material_entry is not None:
                warnings.append(
                    f"Pipe '{self.unit_id}' explicit roughness overrides material {material!r}"
                )
            roughness = self._signed_length_value(explicit, 'roughness')
            if roughness < 0.0:
                raise UnitOperationError(
                    f"Pipe '{self.unit_id}' roughness cannot be negative"
                )
            return roughness, 'explicit', material_key
        if material_key not in PIPE_MATERIAL_ROUGHNESS_M:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' unknown material {material!r}; supported materials: "
                + ', '.join(sorted(PIPE_MATERIAL_ROUGHNESS_M))
            )
        return PIPE_MATERIAL_ROUGHNESS_M[material_key], 'material', material_key

    def _diameters(self, inlet_density: float, mass_flow_kg_s: float
                   ) -> tuple[float, float, str, Optional[float], str]:
        diameter = self._parameter(('diameter', 'D', 'pipe_diameter'))
        diameter_in = self._parameter(('diameter_in', 'inlet_diameter', 'D_in'))
        diameter_out = self._parameter(('diameter_out', 'outlet_diameter', 'D_out'))
        velocity = self._parameter(('velocity', 'target_velocity', 'inlet_velocity'))
        if diameter is not None and diameter_in is not None:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' cannot specify both diameter and diameter_in"
            )
        start_entry = diameter_in or diameter
        if start_entry is not None and velocity is not None:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' can specify diameter or velocity, not both"
            )
        target_velocity = None
        if start_entry is not None:
            diameter_start = self._length_value(start_entry, 'diameter')
            specification = start_entry[0]
        elif velocity is not None:
            target_velocity = self._velocity_value(velocity)
            area = mass_flow_kg_s / (inlet_density * target_velocity)
            diameter_start = math.sqrt(4.0 * area / math.pi)
            specification = velocity[0]
        else:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' requires diameter/diameter_in or velocity"
            )
        diameter_end = (
            diameter_start
            if diameter_out is None
            else self._length_value(diameter_out, 'outlet diameter')
        )
        profile_entry = self._parameter(('diameter_profile', 'profile', 'taper'))
        if profile_entry is None:
            profile = 'constant' if diameter_end == diameter_start else 'smooth'
        else:
            profile = str(profile_entry[1]).strip().lower().replace('-', '_')
        aliases = {'smooth_taper': 'smooth'}
        profile = aliases.get(profile, profile)
        if profile not in {'constant', 'linear', 'smooth', 'smoothstep'}:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' diameter_profile must be constant, linear, "
                "smooth, or smoothstep"
            )
        if profile == 'constant' and abs(diameter_end - diameter_start) > 1e-15:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' constant profile cannot have differing diameters"
            )
        return diameter_start, diameter_end, specification, target_velocity, profile

    @staticmethod
    def _diameter_function(diameter_in: float, diameter_out: float,
                           length_m: float, profile: str):
        def diameter_at(position_m: float) -> float:
            fraction = max(0.0, min(1.0, float(position_m) / length_m))
            if profile == 'constant':
                weight = 0.0
            elif profile in {'linear', 'smooth'}:
                weight = fraction
            else:
                weight = fraction * fraction * (3.0 - 2.0 * fraction)
            return diameter_in + weight * (diameter_out - diameter_in)

        return diameter_at

    @staticmethod
    def _phase_name(state: StreamState) -> Optional[str]:
        if state.vapor_fraction <= 1e-8:
            return 'liquid'
        if state.vapor_fraction >= 1.0 - 1e-8:
            return 'vapor'
        return None

    def _phase_model(self) -> str:
        value = self.get_param(
            'phase_model',
            self.get_param(
                'phase_calculation',
                self.get_param('flow_phase_model', 'auto'),
            ),
        )
        normalized = str(value).strip().lower().replace('-', '_').replace(' ', '_')
        aliases = {
            'automatic': 'auto',
            'single': 'single_phase',
            'singlephase': 'single_phase',
            'two': 'two_phase',
            'twophase': 'two_phase',
            'beggs': 'two_phase',
            'beggs_brill': 'two_phase',
            'beggsbrill': 'two_phase',
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {'auto', 'single_phase', 'two_phase'}:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' phase_model must be auto, single_phase, "
                "or two_phase"
            )
        return normalized

    @staticmethod
    def _caused_by(exc: BaseException, exception_type: type[BaseException]) -> bool:
        current = exc
        while current is not None:
            if isinstance(current, exception_type):
                return True
            current = current.__cause__ or current.__context__
        return False

    def _two_phase_result(self, inlet: StreamState, reason: str,
                          performance: dict, warnings: list[str]) -> UnitResult:
        warnings.append(
            f"Pipe '{self.unit_id}' encountered two-phase flow ({reason}); "
            "two-phase pressure drop is not implemented, so pressure drop was set to zero"
        )
        outlet = inlet.copy()
        performance.update({
            'model_status': 'two_phase_not_implemented',
            'pressure_drop_bar': 0.0,
            'P_in_bar': inlet.P,
            'P_out_bar': inlet.P,
            'two_phase_not_implemented': True,
        })
        return UnitResult(outlet_streams={'out': outlet}, performance=performance, warnings=warnings)

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if len(inlets) != 1:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' requires exactly one inlet stream"
            )
        inlet = next(iter(inlets.values()))
        if inlet.F <= 0.0:
            raise UnitOperationError(f"Pipe '{self.unit_id}' requires positive flow")
        if inlet.P <= 0.0:
            raise UnitOperationError(f"Pipe '{self.unit_id}' requires positive inlet pressure")

        warnings: list[str] = []
        length_m = self._length()
        slope, angle_degrees, orientation = self._elevation(length_m)
        roughness_m, roughness_source, material = self._roughness(warnings)
        friction_model = str(self.get_param('friction_model', 'churchill')).strip().lower()
        phase_model = self._phase_model()
        two_phase_model = str(
            self.get_param('two_phase_model', 'beggs_brill')
        ).strip().lower().replace('-', '_').replace(' ', '_')
        if two_phase_model in {'beggs', 'beggsbrill'}:
            two_phase_model = 'beggs_brill'
        if two_phase_model != 'beggs_brill':
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' two_phase_model must be beggs_brill"
            )
        surface_tension_method = str(
            self.get_param('surface_tension_method', 'auto')
        ).strip().lower().replace('-', '_').replace(' ', '_')
        surface_tension_method = {
            'butler_unifac': 'butler-unifac',
            'winterfeld_scriven_davis': 'wsd',
        }.get(surface_tension_method, surface_tension_method)
        if surface_tension_method not in {'auto', 'butler', 'butler-unifac', 'wsd'}:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' surface_tension_method must be auto, "
                "butler, butler-unifac, or wsd"
            )
        energy_tolerance_entry = self._parameter((
            'local_energy_tolerance',
            'energy_tolerance',
            'energy_iteration_tolerance',
            'energy_residual_tolerance',
        ))
        energy_iteration_tolerance = (
            1.0e-6
            if energy_tolerance_entry is None
            else float(energy_tolerance_entry[1])
        )
        if (
            energy_iteration_tolerance <= 0.0
            or not math.isfinite(energy_iteration_tolerance)
        ):
            name = energy_tolerance_entry[0] if energy_tolerance_entry else 'energy_tolerance'
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' requires positive finite {name}"
            )
        energy_iterations_entry = self._parameter((
            'local_energy_max_iterations',
            'energy_max_iterations',
            'energy_iteration_max_iterations',
            'energy_iterations',
            'energy_iteration_limit',
        ))
        energy_iterations_override = energy_iterations_entry is not None
        local_energy_max_iterations = (
            12
            if energy_iterations_entry is None
            else int(float(energy_iterations_entry[1]))
        )
        if local_energy_max_iterations < 1:
            name = (
                energy_iterations_entry[0]
                if energy_iterations_entry
                else 'energy_max_iterations'
            )
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' requires positive integer {name}"
            )
        composition = dict(inlet.composition)
        molecular_weight = self.thermo.mixture_MW(composition)
        if molecular_weight <= 0.0:
            raise UnitOperationError(f"Pipe '{self.unit_id}' cannot determine mixture MW")
        mass_flow_kg_s = inlet.F * molecular_weight / 3600.0
        base_performance = {
            'length_m': length_m,
            'elevation_change_m': length_m * slope,
            'inclination_angle_degrees': angle_degrees,
            'orientation': orientation,
            'roughness_m': roughness_m,
            'roughness_source': roughness_source,
            'material': material,
            'friction_model': friction_model,
            'phase_model': phase_model,
            'two_phase_model': two_phase_model,
            'energy_iteration_tolerance_kJ_per_kmol': energy_iteration_tolerance,
            'local_energy_max_iterations': local_energy_max_iterations,
            'local_energy_iterations_overridden': energy_iterations_override,
            'mass_flow_kg_s': mass_flow_kg_s,
        }
        allow_two_phase = phase_model in {'auto', 'two_phase'}
        inlet_specified_phase = self._phase_name(inlet)
        if inlet_specified_phase is None and not allow_two_phase:
            return self._two_phase_result(
                inlet, 'two-phase inlet', base_performance, warnings
            )
        if inlet_specified_phase is None and allow_two_phase:
            inlet_state = inlet.copy()
        else:
            try:
                inlet_state = self.thermo.calculate_state(
                    inlet.T, inlet.P, inlet.F, composition, include=('H', 'rho')
                )
            except Exception as exc:
                raise UnitOperationError(
                    f"Pipe '{self.unit_id}' could not evaluate its inlet state"
                ) from exc
        inlet_phase = self._phase_name(inlet_state)
        if inlet_phase is None:
            inlet_phase = 'two_phase'
        if inlet_phase == 'two_phase' and not allow_two_phase:
            return self._two_phase_result(
                inlet, 'equilibrium inlet is two-phase', base_performance, warnings
            )
        if phase_model == 'two_phase' and inlet_phase != 'two_phase':
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' phase_model=two_phase requires a two-phase inlet"
            )
        pooled_lle_warning = inlet_state.liquid2_fraction > 1.0e-10
        if pooled_lle_warning:
            warnings.append(
                f"Pipe '{self.unit_id}' retained two liquid phases but used their "
                "pooled-liquid composition for the current homogeneous transport "
                "correlations"
            )

        inlet_densities = self.thermo.transport_mixture_density(
            composition,
            inlet_state.T,
            inlet_state.P,
            inlet_state.vapor_fraction,
            inlet_state.x,
            inlet_state.y,
        )

        def vapor_mass_quality(state: StreamState) -> float:
            vapor_fraction = max(0.0, min(1.0, float(state.vapor_fraction)))
            liquid_fraction = 1.0 - vapor_fraction
            liquid_mw = self.thermo.mixture_MW(state.x or composition)
            vapor_mw = self.thermo.mixture_MW(state.y or composition)
            vapor_mass = vapor_fraction * vapor_mw
            liquid_mass = liquid_fraction * liquid_mw
            total_mass = vapor_mass + liquid_mass
            if total_mass <= 0.0:
                raise UnitOperationError(
                    f"Pipe '{self.unit_id}' cannot determine two-phase mass quality"
                )
            return vapor_mass / total_mass

        def bulk_two_phase_density(quality: float, liquid_density: float,
                                   vapor_density: float) -> float:
            specific_volume = (
                (1.0 - quality) / liquid_density
                + quality / vapor_density
            )
            if specific_volume <= 0.0:
                raise UnitOperationError(
                    "Pipe calculated invalid two-phase specific volume"
                )
            return 1.0 / specific_volume

        if inlet_phase == 'two_phase':
            if inlet_densities.liquid is None or inlet_densities.vapor is None:
                raise UnitOperationError(
                    f"Pipe '{self.unit_id}' cannot determine two-phase inlet densities"
                )
            inlet_quality = vapor_mass_quality(inlet_state)
            inlet_density = bulk_two_phase_density(
                inlet_quality,
                inlet_densities.liquid,
                inlet_densities.vapor,
            )
        else:
            inlet_density = (
                inlet_densities.vapor if inlet_phase == 'vapor' else inlet_densities.liquid
            )
        diameter_in, diameter_out, diameter_spec, target_velocity, profile = self._diameters(
            inlet_density, mass_flow_kg_s
        )
        diameter_at = self._diameter_function(
            diameter_in, diameter_out, length_m, profile
        )
        geometry = CircularConduitGeometry(
            diameter_m=diameter_at,
            roughness_m=roughness_m,
            elevation_gradient=slope,
        )
        base_performance.update({
            'diameter_specification': diameter_spec,
            'target_inlet_velocity_m_s': target_velocity,
            'diameter_in_m': diameter_in,
            'diameter_out_m': diameter_out,
            'diameter_profile': profile,
            'inlet_phase': inlet_phase,
        })

        area_in = geometry.area_at(0.0)
        inlet_velocity = mass_flow_kg_s / (inlet_density * area_in)
        if inlet_state.H is None:
            raise UnitOperationError(f"Pipe '{self.unit_id}' inlet enthalpy is unavailable")
        total_specific_energy = (
            inlet_state.H * 1000.0 / molecular_weight
            + inlet_velocity * inlet_velocity / 2.0
        )
        state_solver = _ThermoStateSolver(self.thermo, f"Pipe '{self.unit_id}'")
        last_temperature_seed = inlet.T

        def local_state(position_m: float, pressure_bar: float,
                        with_flow: bool = True,
                        check_equilibrium: bool = True) -> _PipeLocalState:
            nonlocal last_temperature_seed, pooled_lle_warning
            elevation_m = slope * position_m
            diameter_m = geometry.diameter_at(position_m)
            area_m2 = math.pi * diameter_m * diameter_m / 4.0
            velocity_guess = inlet_velocity * area_in / area_m2
            enthalpy_target = (
                total_specific_energy
                - 9.80665 * elevation_m
                - velocity_guess * velocity_guess / 2.0
            ) * molecular_weight / 1000.0
            state = None
            density = None
            velocity = None
            phase = None
            quality = None
            previous_target = None
            previous_residual = None
            energy_iterations = 0
            energy_iteration_residual = math.inf
            energy_iteration_converged = False
            for iteration in range(1, local_energy_max_iterations + 1):
                state, _ = state_solver.state_at_enthalpy(
                    pressure_bar,
                    inlet.F,
                    composition,
                    enthalpy_target,
                    last_temperature_seed if state is None else state.T,
                    force_phase=(
                        inlet_phase if phase_model == 'single_phase' else None
                    ),
                    include=('H',),
                )
                if state.liquid2_fraction > 1.0e-10 and not pooled_lle_warning:
                    pooled_lle_warning = True
                    warnings.append(
                        f"Pipe '{self.unit_id}' formed two liquid phases and used "
                        "their pooled-liquid composition for the current homogeneous "
                        "transport correlations"
                    )
                phase = self._phase_name(state)
                if phase is None:
                    if not allow_two_phase:
                        raise _TwoPhasePipeFlow(
                            f"vapor fraction {state.vapor_fraction:.6g} "
                            f"at {position_m:.4g} m"
                        )
                    phase = 'two_phase'
                elif inlet_phase != 'two_phase' and phase != inlet_phase:
                    raise _TwoPhasePipeFlow(
                        f"phase changed from {inlet_phase} to {phase} at {position_m:.4g} m"
                    )
                if phase_model == 'two_phase' and phase != 'two_phase':
                    raise UnitOperationError(
                        f"Pipe '{self.unit_id}' phase_model=two_phase found "
                        f"{phase} flow at {position_m:.4g} m"
                    )
                densities = self.thermo.transport_mixture_density(
                    composition,
                    state.T,
                    pressure_bar,
                    state.vapor_fraction,
                    state.x,
                    state.y,
                )
                if phase == 'two_phase':
                    if densities.liquid is None or densities.vapor is None:
                        raise UnitOperationError(
                            f"Pipe '{self.unit_id}' cannot determine two-phase densities "
                            f"at {position_m:.4g} m"
                        )
                    quality = vapor_mass_quality(state)
                    density = bulk_two_phase_density(
                        quality,
                        densities.liquid,
                        densities.vapor,
                    )
                else:
                    density = densities.vapor if phase == 'vapor' else densities.liquid
                velocity = mass_flow_kg_s / (density * area_m2)
                updated_target = (
                    total_specific_energy
                    - 9.80665 * elevation_m
                    - velocity * velocity / 2.0
                ) * molecular_weight / 1000.0
                residual = updated_target - enthalpy_target
                energy_iterations = iteration
                energy_iteration_residual = residual
                energy_iteration_converged = (
                    abs(residual) <= energy_iteration_tolerance
                )
                phase_iteration_limit = (
                    local_energy_max_iterations
                    if energy_iterations_override
                    else (1 if phase == 'two_phase' else local_energy_max_iterations)
                )
                if energy_iteration_converged:
                    enthalpy_target = updated_target
                    break
                if iteration >= phase_iteration_limit:
                    if phase == 'two_phase':
                        break
                    raise UnitOperationError(
                        f"Pipe '{self.unit_id}' local energy iteration did not converge "
                        f"after {iteration} iterations at {position_m:.4g} m "
                        f"(residual {residual:.4g} kJ/kmol)"
                    )
                next_target = updated_target
                if previous_target is not None and previous_residual is not None:
                    denominator = residual - previous_residual
                    if denominator and math.isfinite(denominator):
                        secant_target = (
                            enthalpy_target
                            - residual * (enthalpy_target - previous_target) / denominator
                        )
                        if math.isfinite(secant_target):
                            step = secant_target - enthalpy_target
                            max_step = max(
                                1.0,
                                10.0 * abs(residual),
                                abs(enthalpy_target) * 0.01,
                            )
                            if abs(step) > max_step:
                                step = math.copysign(max_step, step)
                            next_target = enthalpy_target + step
                previous_target = enthalpy_target
                previous_residual = residual
                enthalpy_target = next_target
            else:
                raise UnitOperationError(
                    f"Pipe '{self.unit_id}' local energy iteration did not converge"
                )
            if state is None or density is None or velocity is None:
                raise UnitOperationError(f"Pipe '{self.unit_id}' local state is unavailable")
            last_temperature_seed = state.T

            viscosity = None
            surface_tension = None
            flow = None
            if with_flow:
                if check_equilibrium and phase != 'two_phase':
                    try:
                        equilibrium = self.thermo.calculate_state(
                            state.T,
                            pressure_bar,
                            inlet.F,
                            composition,
                            include=(),
                        )
                    except Exception:
                        equilibrium = None
                    if equilibrium is not None:
                        equilibrium_phase = self._phase_name(equilibrium)
                        if equilibrium_phase is None:
                            if not allow_two_phase:
                                raise _TwoPhasePipeFlow(
                                    f"vapor fraction {equilibrium.vapor_fraction:.6g} "
                                    f"at {position_m:.4g} m"
                                )
                        elif equilibrium_phase != inlet_phase:
                            raise _TwoPhasePipeFlow(
                                f"phase changed from {inlet_phase} to {equilibrium_phase} "
                                f"at {position_m:.4g} m"
                            )

                viscosities = self.thermo.transport_mixture_viscosity(
                    composition,
                    state.T,
                    pressure_bar,
                    state.vapor_fraction,
                    state.x,
                    state.y,
                )
                if phase == 'two_phase':
                    densities = self.thermo.transport_mixture_density(
                        composition,
                        state.T,
                        pressure_bar,
                        state.vapor_fraction,
                        state.x,
                        state.y,
                    )
                    if (
                        densities.liquid is None or densities.vapor is None
                        or viscosities.liquid is None or viscosities.vapor is None
                    ):
                        raise UnitOperationError(
                            f"Pipe '{self.unit_id}' cannot determine two-phase "
                            f"transport properties at {position_m:.4g} m"
                        )
                    quality = vapor_mass_quality(state)
                    surface_tension = self.thermo.transport_mixture_surface_tension(
                        composition,
                        state.T,
                        pressure_bar,
                        state.vapor_fraction,
                        state.x,
                        state.y,
                        method=surface_tension_method,
                    )
                    viscosity = (
                        (1.0 - quality) * viscosities.liquid
                        + quality * viscosities.vapor
                    )
                    flow = beggs_brill_two_phase_flow(
                        mass_flow_kg_s,
                        quality,
                        densities.liquid,
                        densities.vapor,
                        viscosities.liquid,
                        viscosities.vapor,
                        surface_tension,
                        pressure_bar * 1.0e5,
                        diameter_m,
                        roughness_m=geometry.roughness_at(position_m),
                        elevation_gradient=geometry.elevation_gradient_at(position_m),
                        friction_model=friction_model,
                    )
                else:
                    viscosity = (
                        viscosities.vapor if phase == 'vapor' else viscosities.liquid
                    )
                    flow = single_phase_flow(
                        mass_flow_kg_s,
                        density,
                        viscosity,
                        diameter_m,
                        roughness_m=geometry.roughness_at(position_m),
                        elevation_gradient=geometry.elevation_gradient_at(position_m),
                        friction_model=friction_model,
                    )
            energy_residual = (
                (state.H or 0.0) * 1000.0 / molecular_weight
                + velocity * velocity / 2.0
                + 9.80665 * elevation_m
                - total_specific_energy
            )
            return _PipeLocalState(
                state=state,
                phase=phase,
                density_kg_m3=density,
                viscosity_pa_s=viscosity,
                surface_tension_n_m=surface_tension,
                mass_quality=quality,
                velocity_m_s=velocity,
                flow=flow,
                energy_residual_j_kg=energy_residual,
                energy_iterations=energy_iterations,
                energy_iteration_residual_kj_per_kmol=energy_iteration_residual,
                energy_iteration_converged=energy_iteration_converged,
            )

        def velocity_pressure_derivative(position_m: float, pressure_bar: float) -> float:
            pressure_step = max(1e-5, abs(pressure_bar) * 1e-4)
            pressure_low = max(1e-5, pressure_bar - pressure_step)
            pressure_high = pressure_bar + pressure_step
            velocity_low = local_state(position_m, pressure_low, with_flow=False).velocity_m_s
            velocity_high = local_state(position_m, pressure_high, with_flow=False).velocity_m_s
            return (velocity_high - velocity_low) / (
                (pressure_high - pressure_low) * 1e5
            )

        def velocity_position_derivative(position_m: float, pressure_bar: float) -> float:
            position_step = max(1e-5, min(0.01, length_m * 1e-4))
            position_low = max(0.0, position_m - position_step)
            position_high = min(length_m, position_m + position_step)
            if position_high <= position_low:
                return 0.0
            velocity_low = local_state(position_low, pressure_bar, with_flow=False).velocity_m_s
            velocity_high = local_state(position_high, pressure_bar, with_flow=False).velocity_m_s
            return (velocity_high - velocity_low) / (position_high - position_low)

        def balances(position_m: float, values):
            integrated_pressure_bar = float(values[0])
            pressure_exhausted = integrated_pressure_bar <= self._MINIMUM_PRESSURE_BAR
            pressure_bar = max(integrated_pressure_bar, self._MINIMUM_PRESSURE_BAR)
            local = local_state(
                position_m,
                pressure_bar,
                check_equilibrium=not pressure_exhausted,
            )
            if isinstance(local.flow, TwoPhaseFlow):
                gradient = local.flow.pressure_gradient
                return (
                    -gradient.total / 1e5,
                    gradient.friction / 1e5,
                    gradient.gravity / 1e5,
                    gradient.acceleration / 1e5,
                )
            velocity_p = velocity_pressure_derivative(position_m, pressure_bar)
            velocity_s = velocity_position_derivative(position_m, pressure_bar)
            density = local.density_kg_m3
            velocity = local.velocity_m_s
            friction = local.flow.pressure_gradient.friction
            gravity = local.flow.pressure_gradient.gravity
            denominator = 1.0 + density * velocity * velocity_p
            if denominator <= 1e-6 or not math.isfinite(denominator):
                raise UnitOperationError(
                    f"Pipe '{self.unit_id}' approached a choking/acceleration singularity "
                    f"at {position_m:.4g} m"
                )
            pressure_gradient_pa_m = -(
                friction + gravity + density * velocity * velocity_s
            ) / denominator
            acceleration = density * velocity * (
                velocity_p * pressure_gradient_pa_m + velocity_s
            )
            return (
                pressure_gradient_pa_m / 1e5,
                friction / 1e5,
                gravity / 1e5,
                acceleration / 1e5,
            )

        relative_tolerance = float(self.get_param('relative_tolerance', self.get_param('rtol', 1e-8)))
        absolute_tolerance = float(self.get_param('absolute_tolerance', self.get_param('atol', 1e-10)))
        max_step_entry = self._parameter(('maximum_step', 'max_step', 'max_step_m'))
        maximum_step = (
            min(10.0, length_m)
            if max_step_entry is None
            else self._length_value(max_step_entry, 'maximum step')
        )
        profile_points = int(self.get_param('profile_points', 11))
        if profile_points < 2 or profile_points > 201:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' profile_points must be between 2 and 201"
            )
        positions = tuple(
            length_m * index / (profile_points - 1)
            for index in range(profile_points)
        )
        if inlet.P <= self._MINIMUM_PRESSURE_BAR:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' has no available inlet pressure; "
                f"{inlet_phase} flow stops at 0 m of {length_m:g} m"
            )
        try:
            solution = integrate_axial(
                balances,
                length_m,
                (inlet.P, 0.0, 0.0, 0.0),
                options=AxialSolverOptions(
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                    maximum_step_m=maximum_step,
                ),
                events=(
                    AxialEvent(
                        'minimum_pressure',
                        lambda _position, values: (
                            values[0] - self._MINIMUM_PRESSURE_BAR
                        ),
                        direction=-1.0,
                    ),
                ),
                evaluation_positions_m=positions,
            )
        except Exception as exc:
            if self._caused_by(exc, _TwoPhasePipeFlow):
                return self._two_phase_result(
                    inlet,
                    'phase transition during axial integration',
                    base_performance,
                    warnings,
                )
            if isinstance(exc, UnitOperationError):
                raise
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' axial integration failed"
            ) from exc
        if solution.terminated_by_event:
            raise UnitOperationError(
                f"Pipe '{self.unit_id}' exhausted its available inlet pressure after "
                f"{solution.outlet_position_m:.4g} m of {length_m:g} m; "
                f"{inlet_phase} flow cannot reach the outlet at the specified flow rate"
            )

        pressure_out = float(solution.outlet_values[0])
        try:
            final_local = local_state(length_m, pressure_out)
        except _TwoPhasePipeFlow:
            return self._two_phase_result(
                inlet,
                'phase transition at outlet',
                base_performance,
                warnings,
            )
        outlet = self.thermo.calculate_state(
            final_local.state.T,
            pressure_out,
            inlet.F,
            composition,
        )
        profile_data = []
        two_phase_encountered = False
        for index, position_m in enumerate(solution.position_m):
            pressure_bar = float(solution.values[0, index])
            local = local_state(float(position_m), pressure_bar)
            is_two_phase = isinstance(local.flow, TwoPhaseFlow)
            two_phase_encountered = two_phase_encountered or is_two_phase
            row = {
                'position_m': float(position_m),
                'pressure_bar': pressure_bar,
                'temperature_K': local.state.T,
                'temperature_C': local.state.T - 273.15,
                'diameter_m': geometry.diameter_at(float(position_m)),
                'phase': local.phase,
                'vapor_fraction': local.state.vapor_fraction,
                'density_kg_m3': local.density_kg_m3,
                'viscosity_Pa_s': local.viscosity_pa_s,
                'velocity_m_s': local.velocity_m_s,
                'darcy_friction_factor': local.flow.darcy_friction_factor,
                'energy_residual_J_per_kg': local.energy_residual_j_kg,
                'energy_iterations': local.energy_iterations,
                'energy_iteration_residual_kJ_per_kmol': (
                    local.energy_iteration_residual_kj_per_kmol
                ),
                'energy_iteration_converged': local.energy_iteration_converged,
            }
            if is_two_phase:
                row.update({
                    'surface_tension_N_m': local.surface_tension_n_m,
                    'mass_quality': local.mass_quality,
                    'flow_regime': local.flow.flow_regime,
                    'liquid_holdup': local.flow.liquid_holdup,
                    'no_slip_liquid_fraction': local.flow.no_slip_liquid_fraction,
                    'superficial_liquid_velocity_m_s': (
                        local.flow.superficial_liquid_velocity_m_s
                    ),
                    'superficial_vapor_velocity_m_s': (
                        local.flow.superficial_vapor_velocity_m_s
                    ),
                    'froude_number': local.flow.froude_number,
                    'liquid_velocity_number': local.flow.liquid_velocity_number,
                    'reynolds_number': local.flow.no_slip_reynolds_number,
                    'no_slip_reynolds_number': local.flow.no_slip_reynolds_number,
                    'no_slip_friction_factor': local.flow.no_slip_friction_factor,
                    'two_phase_friction_multiplier': (
                        local.flow.two_phase_friction_multiplier
                    ),
                    'beggs_brill_acceleration_factor': local.flow.acceleration_factor,
                })
            else:
                row['reynolds_number'] = local.flow.reynolds_number
            profile_data.append(row)

        friction_drop = float(solution.outlet_values[1])
        gravity_drop = float(solution.outlet_values[2])
        acceleration_drop = float(solution.outlet_values[3])
        pressure_drop = inlet.P - pressure_out
        base_performance.update({
            'model_status': (
                'two_phase_beggs_brill_integrated'
                if two_phase_encountered
                else 'single_phase_integrated'
            ),
            'two_phase_not_implemented': False,
            'two_phase_encountered': two_phase_encountered,
            'P_in_bar': inlet.P,
            'P_out_bar': pressure_out,
            'pressure_drop_bar': pressure_drop,
            'friction_pressure_drop_bar': friction_drop,
            'gravity_pressure_drop_bar': gravity_drop,
            'acceleration_pressure_drop_bar': acceleration_drop,
            'pressure_drop_closure_bar': (
                pressure_drop - friction_drop - gravity_drop - acceleration_drop
            ),
            'T_in_C': inlet_state.T - 273.15,
            'T_out_C': outlet.T - 273.15,
            'velocity_in_m_s': profile_data[0]['velocity_m_s'],
            'velocity_out_m_s': profile_data[-1]['velocity_m_s'],
            'reynolds_in': profile_data[0]['reynolds_number'],
            'reynolds_out': profile_data[-1]['reynolds_number'],
            'friction_factor_in': profile_data[0]['darcy_friction_factor'],
            'friction_factor_out': profile_data[-1]['darcy_friction_factor'],
            'maximum_energy_residual_J_per_kg': max(
                abs(row['energy_residual_J_per_kg'])
                for row in profile_data
            ),
            'maximum_local_energy_iterations': max(
                int(row['energy_iterations']) for row in profile_data
            ),
            'maximum_local_energy_iteration_residual_kJ_per_kmol': max(
                abs(row['energy_iteration_residual_kJ_per_kmol'])
                for row in profile_data
            ),
            'integration_evaluations': solution.evaluations,
            'integration_relative_tolerance': relative_tolerance,
            'integration_absolute_tolerance': absolute_tolerance,
            'integration_maximum_step_m': maximum_step,
            'profile': profile_data,
        })
        return UnitResult(
            outlet_streams={'out': outlet},
            performance=base_performance,
            warnings=warnings,
        )
