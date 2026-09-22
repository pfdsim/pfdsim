"""
Basic stream handling, pressure-change, heat-transfer, and flash unit operations.
"""

import math
import inspect

from scipy.optimize import brentq

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .physical_constants import R_J_MOL_K
else:
    from physical_constants import R_J_MOL_K

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .pressure_standards import ATM_PRESSURE_BAR, THERMOCHEMICAL_STANDARD_PRESSURE_BAR
else:
    from pressure_standards import ATM_PRESSURE_BAR, THERMOCHEMICAL_STANDARD_PRESSURE_BAR
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics import StreamState
else:
    from thermodynamics import StreamState
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_conversions import (
        mass_flow_to_kg_per_hour,
        molar_flow_to_kmol_per_hour,
        pressure_to_bar,
    )
else:
    from unit_conversions import (
        mass_flow_to_kg_per_hour,
        molar_flow_to_kmol_per_hour,
        pressure_to_bar,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_base import UnitOperation, UnitOperationError, UnitResult
else:
    from unit_operations_base import UnitOperation, UnitOperationError, UnitResult


BTU_H_FT2_F_TO_W_M2_K = 5.6783


def _u_service(low_btu: float, high_btu: float, dirt: float | None = None) -> dict:
    low = low_btu * BTU_H_FT2_F_TO_W_M2_K
    high = high_btu * BTU_H_FT2_F_TO_W_M2_K
    record = {
        'low': low,
        'typical': 0.5 * (low + high),
        'high': high,
        'source': 'Seader Table 12.5',
    }
    if dirt is not None:
        record['dirt_factor_hr_ft2_F_per_Btu'] = dirt
        record['dirt_factor_m2_K_per_W'] = dirt * 0.1761
    return record


class _ThermoStateSolver:
    """Shared pressure/property state solves for unit operations."""

    def __init__(self, thermo, unit_label: str):
        self.thermo = thermo
        self.unit_label = unit_label
        self._ph_predictor_count = 0
        self._ph_predictor_mean = None
        self._ph_predictor_m2 = None
        self._ph_predictor_history = []
        self._ph_predictor_phase = None

    def _reset_ph_predictor(self, phase: str | None) -> None:
        self._ph_predictor_count = 0
        self._ph_predictor_mean = None
        self._ph_predictor_m2 = None
        self._ph_predictor_history = []
        self._ph_predictor_phase = phase

    def _ph_predictor_feature(
        self,
        P: float,
        H_target: float,
        composition: dict,
    ) -> tuple[float, ...]:
        return (
            float(H_target),
            math.log(max(float(P), 1.0e-300)),
            *(
                float(composition.get(component, 0.0))
                for component in self.thermo.components
            ),
        )

    def _ph_predictor_seed(
        self,
        feature: tuple[float, ...],
        fallback: float,
    ) -> float:
        history = self._ph_predictor_history
        if not history:
            return float(fallback)
        if len(history) == 1:
            return float(history[-1][1])
        count = self._ph_predictor_count
        scale = []
        floors = (100.0, 0.01, *([1.0e-4] * len(self.thermo.components)))
        for index, floor in enumerate(floors):
            variance = (
                self._ph_predictor_m2[index] / max(count - 1, 1)
                if self._ph_predictor_m2 is not None
                else 0.0
            )
            scale.append(max(math.sqrt(max(variance, 0.0)), floor))
        older_feature, older_temperature = history[-2]
        previous_feature, previous_temperature = history[-1]
        previous_delta = tuple(
            (new - old) / item_scale
            for new, old, item_scale in zip(
                previous_feature,
                older_feature,
                scale,
            )
        )
        current_delta = tuple(
            (new - old) / item_scale
            for new, old, item_scale in zip(
                feature,
                previous_feature,
                scale,
            )
        )
        denominator = sum(value * value for value in previous_delta)
        factor = (
            0.0
            if denominator <= 1.0e-20
            else sum(
                current * previous
                for current, previous in zip(current_delta, previous_delta)
            ) / denominator
        )
        prediction = previous_temperature + factor * (
            previous_temperature - older_temperature
        )
        maximum_jump = max(
            10.0,
            3.0 * abs(previous_temperature - older_temperature),
        )
        return max(
            1.0,
            min(
                5000.0,
                max(
                    previous_temperature - maximum_jump,
                    min(previous_temperature + maximum_jump, prediction),
                ),
            ),
        )

    def _update_ph_predictor(
        self,
        feature: tuple[float, ...],
        temperature: float,
    ) -> None:
        count = self._ph_predictor_count
        if count == 0:
            self._ph_predictor_mean = list(feature)
            self._ph_predictor_m2 = [0.0] * len(feature)
        else:
            new_count = count + 1
            for index, value in enumerate(feature):
                delta = value - self._ph_predictor_mean[index]
                self._ph_predictor_mean[index] += delta / new_count
                self._ph_predictor_m2[index] += delta * (
                    value - self._ph_predictor_mean[index]
                )
        self._ph_predictor_count = count + 1
        self._ph_predictor_history.append((feature, float(temperature)))
        if len(self._ph_predictor_history) > 2:
            self._ph_predictor_history.pop(0)

    def _calculate_trial_state(self, T: float, P: float, F: float,
                               composition: dict, include,
                               force_phase: str | None) -> StreamState:
        if force_phase in ('vapor', 'liquid'):
            return self.thermo.calculate_state(
                T, P, F, composition, phase=force_phase, flash=False, include=include
            )
        return self.thermo.calculate_state(T, P, F, composition, include=include)

    def _direct_state_compatible(self, state: StreamState,
                                 force_phase: str | None) -> bool:
        fluid_vapor_fraction = state.fluid_vapor_fraction
        if force_phase is not None and fluid_vapor_fraction is None:
            return False
        if force_phase == 'vapor' and fluid_vapor_fraction < 1.0 - 1e-8:
            return False
        if force_phase == 'liquid' and fluid_vapor_fraction > 1e-8:
            return False
        return True

    def _include_with(self, include, *required: str):
        if include is None:
            return None
        return tuple(dict.fromkeys(tuple(include) + tuple(required)))

    def _direct_state_at_entropy(self, P: float, F: float, composition: dict,
                                 S_target: float,
                                 force_phase: str | None) -> StreamState | None:
        solver = getattr(self.thermo, 'calculate_state_PS', None)
        if solver is None:
            return None
        try:
            state = solver(P, S_target, F, composition, include=('H', 'S', 'rho', 'Cp'))
        except (NotImplementedError, AttributeError):
            return None
        except Exception:
            return None
        if state.S is None or state.H is None:
            return None
        if not self._direct_state_compatible(state, force_phase):
            return None
        return state

    def _direct_state_at_enthalpy(self, P: float, F: float, composition: dict,
                                  H_target: float, force_phase: str | None,
                                  include=None,
                                  T_guess: float | None = None) -> StreamState | None:
        solver = getattr(self.thermo, 'calculate_state_PH', None)
        if solver is None:
            return None
        solver_include = self._include_with(include, 'H')
        if force_phase != self._ph_predictor_phase:
            self._reset_ph_predictor(force_phase)
        feature = self._ph_predictor_feature(P, H_target, composition)
        predictor_seed = self._ph_predictor_seed(
            feature,
            T_guess if T_guess is not None else 298.15,
        )
        supported_keywords = getattr(self, '_direct_ph_keywords', None)
        if supported_keywords is None:
            try:
                parameters = inspect.signature(solver).parameters
                accepts_keywords = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
                supported_keywords = {
                    name for name in ('include', 'phase', 'T_guess')
                    # A legacy forwarding wrapper may accept **kwargs even
                    # though its underlying backend has no phase/seed API.
                    if (name == 'include' and accepts_keywords) or (
                        name in parameters
                        and parameters[name].kind != inspect.Parameter.POSITIONAL_ONLY
                    )
                }
            except (TypeError, ValueError):
                supported_keywords = {'include'}
            self._direct_ph_keywords = supported_keywords
        keywords = {
            'include': solver_include,
            'phase': force_phase,
            'T_guess': predictor_seed,
        }
        try:
            state = solver(
                P, H_target, F, composition,
                **{key: value for key, value in keywords.items()
                   if key in supported_keywords},
            )
        except (NotImplementedError, AttributeError):
            return None
        except Exception:
            return None
        if (
            state.H is None
            or not math.isfinite(state.T)
            or state.T <= 0.0
            or not math.isfinite(state.H)
            or abs(state.H - H_target) > self._property_tolerance(H_target)
        ):
            return None
        if not self._direct_state_compatible(state, force_phase):
            return None
        self._update_ph_predictor(feature, state.T)
        return state

    def temperature_at_enthalpy(
        self,
        P: float,
        F: float,
        composition: dict,
        H_target: float,
        T_guess: float,
        force_phase: str | None = None,
    ) -> tuple[float, float]:
        """Solve PH for temperature without constructing a full stream state."""
        solver = getattr(self.thermo, 'temperature_at_PH', None)
        if callable(solver):
            if force_phase != self._ph_predictor_phase:
                self._reset_ph_predictor(force_phase)
            feature = self._ph_predictor_feature(P, H_target, composition)
            predictor_seed = self._ph_predictor_seed(feature, T_guess)
            try:
                temperature, residual = solver(
                    P,
                    H_target,
                    composition,
                    phase=force_phase,
                    T_guess=predictor_seed,
                )
                temperature = float(temperature)
                residual = float(residual)
                if (
                    math.isfinite(temperature) and temperature > 0.0
                    and math.isfinite(residual)
                    and abs(residual) <= self._property_tolerance(H_target)
                ):
                    self._update_ph_predictor(feature, temperature)
                    return temperature, residual
            except (NotImplementedError, AttributeError):
                pass
            except Exception:
                pass

        state, residual = self.state_at_enthalpy(
            P,
            F,
            composition,
            H_target,
            T_guess,
            force_phase=force_phase,
            include=('H',),
        )
        return float(state.T), float(residual)

    def _pure_saturation_temperature(self, comp: str, P: float,
                                     composition: dict, T_guess: float) -> float | None:
        props = getattr(self.thermo, 'props', {}).get(comp)
        Pc = getattr(props, 'Pc', None)
        if Pc is not None and P >= Pc:
            return None

        candidates = self._temperature_grid(T_guess)
        try:
            candidates.append(self.thermo.bubble_point_T(composition, P, T_guess))
        except Exception:
            pass
        grid = sorted(set(max(1.0, min(5000.0, float(T))) for T in candidates))

        def residual(T: float) -> float:
            try:
                return self.thermo.K_value(comp, T, P, composition) - 1.0
            except TypeError:
                return self.thermo.K_value(comp, T, P) - 1.0

        evaluated = []
        for T in grid:
            try:
                value = residual(T)
                if math.isfinite(value):
                    evaluated.append((T, value))
            except Exception:
                continue

        for T, value in evaluated:
            if abs(value) < 1e-8:
                return T
        for (T1, f1), (T2, f2) in zip(evaluated, evaluated[1:]):
            if f1 * f2 < 0.0:
                return brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=100)
        return None

    def _saturated_phase_states(self, P: float, F: float, composition: dict,
                                T_guess: float) -> tuple[StreamState, StreamState] | None:
        if any(
            composition.get(component, 0.0) > 1.0e-12
            for component in getattr(
                self.thermo, 'permanent_solid_components', ()
            )
        ):
            return None
        active = [comp for comp, z in composition.items() if z > 1e-10]
        if len(active) != 1:
            return None
        comp = active[0]
        try:
            T_sat = self._pure_saturation_temperature(comp, P, composition, T_guess)
            if T_sat is None or not math.isfinite(T_sat):
                return None
            liquid = self.thermo.calculate_state(
                T_sat, P, F, composition, phase='liquid', flash=False
            )
            vapor = self.thermo.calculate_state(
                T_sat, P, F, composition, phase='vapor', flash=False
            )
        except Exception:
            return None

        if liquid.H is None or liquid.S is None or vapor.H is None or vapor.S is None:
            return None
        return liquid, vapor

    def _two_phase_state(self, liquid: StreamState, vapor: StreamState,
                         vapor_fraction: float,
                         H: float | None = None,
                         S: float | None = None) -> StreamState:
        vapor_fraction = max(0.0, min(1.0, float(vapor_fraction)))
        composition = dict(liquid.composition)
        state = liquid.copy()
        state.vapor_fraction = vapor_fraction
        state.liquid1_fraction = 1.0 - vapor_fraction
        state.liquid2_fraction = 0.0
        state.x = dict(composition)
        state.x1 = dict(composition)
        state.x2 = None
        state.y = dict(composition)
        state.phase_status = 'pure_component_saturation'
        state.phase_stability = 'explicit_saturation_construction'
        if H is None:
            H = (1.0 - vapor_fraction) * (liquid.H or 0.0) + vapor_fraction * (vapor.H or 0.0)
        if S is None:
            S = (1.0 - vapor_fraction) * (liquid.S or 0.0) + vapor_fraction * (vapor.S or 0.0)
        state.H = H
        state.S = S
        state.MW = self.thermo.mixture_MW(composition)
        try:
            state.Cp = self.thermo.phase_weighted_mixture_Cp(
                composition,
                liquid.T,
                vapor_fraction,
                state.x,
                state.y,
                liquid.P,
            )
        except Exception:
            state.Cp = None
        try:
            state.rho = self.thermo.mixture_molar_density(
                composition,
                liquid.T,
                liquid.P,
                vapor_fraction,
                state.x,
                state.y,
            )
        except Exception:
            state.rho = None
        return state

    def _saturated_state_at_entropy(self, P: float, F: float, composition: dict,
                                    S_target: float, T_guess: float) -> StreamState | None:
        phases = self._saturated_phase_states(P, F, composition, T_guess)
        if phases is None:
            return None
        liquid, vapor = phases
        S_low = min(liquid.S, vapor.S)
        S_high = max(liquid.S, vapor.S)
        tolerance = max(1e-6, abs(S_target) * 1e-10)
        if S_target < S_low - tolerance or S_target > S_high + tolerance:
            return None

        denom = vapor.S - liquid.S
        vapor_fraction = 0.0 if abs(denom) < 1e-12 else (S_target - liquid.S) / denom
        return self._two_phase_state(liquid, vapor, vapor_fraction, S=S_target)

    def _saturated_state_at_enthalpy(self, P: float, F: float, composition: dict,
                                     H_target: float, T_guess: float) -> StreamState | None:
        phases = self._saturated_phase_states(P, F, composition, T_guess)
        if phases is None:
            return None
        liquid, vapor = phases
        H_low = min(liquid.H, vapor.H)
        H_high = max(liquid.H, vapor.H)
        tolerance = max(1e-6, abs(H_target) * 1e-10)
        if H_target < H_low - tolerance or H_target > H_high + tolerance:
            return None

        denom = vapor.H - liquid.H
        vapor_fraction = 0.0 if abs(denom) < 1e-12 else (H_target - liquid.H) / denom
        return self._two_phase_state(liquid, vapor, vapor_fraction, H=H_target)

    def _two_phase_envelope_temperatures(self, P: float, composition: dict,
                                         T_guess: float) -> tuple[float, float] | None:
        try:
            fluid_composition = self.thermo._split_permanent_solid_composition(
                composition
            )[2]
        except Exception:
            fluid_composition = composition
        active = [comp for comp, z in fluid_composition.items() if z > 1e-10]
        if len(active) <= 1:
            return None
        try:
            T_bubble = self.thermo.bubble_point_T(fluid_composition, P, T_guess)
            T_dew = self.thermo.dew_point_T(fluid_composition, P, T_guess)
        except Exception:
            return None
        if not math.isfinite(T_bubble) or not math.isfinite(T_dew):
            return None

        T_low, T_high = sorted((float(T_bubble), float(T_dew)))
        if T_high - T_low < 1e-6:
            return None
        return T_low, T_high

    def _property_tolerance(self, target: float) -> float:
        return max(1e-8, abs(target) * 1e-12)

    def _two_phase_envelope_state_at_property(self, P: float, F: float,
                                              composition: dict,
                                              property_name: str,
                                              target: float,
                                              T_guess: float) -> StreamState | None:
        envelope = self._two_phase_envelope_temperatures(P, composition, T_guess)
        if envelope is None:
            return None
        T_low, T_high = envelope

        def trial_state(T: float, include=None) -> StreamState:
            return self.thermo.calculate_state(T, P, F, composition, include=include)

        def property_value(state: StreamState) -> float:
            value = getattr(state, property_name)
            if value is None:
                raise UnitOperationError(f"{property_name} was not calculated")
            return value

        def residual(T: float) -> float:
            state = trial_state(T, include=(property_name,))
            return property_value(state) - target

        tolerance = self._property_tolerance(target)
        seed = max(T_low, min(T_high, T_guess))
        try:
            low_state = trial_state(T_low, include=(property_name,))
            high_state = trial_state(T_high, include=(property_name,))
            low_value = property_value(low_state)
            high_value = property_value(high_state)
        except Exception:
            low_state = high_state = None
            low_value = high_value = None
        if low_value is not None and high_value is not None:
            value_min = min(low_value, high_value)
            value_max = max(low_value, high_value)
            if target < value_min - tolerance or target > value_max + tolerance:
                return None
            if abs(low_value - target) <= tolerance:
                return trial_state(T_low)
            if abs(high_value - target) <= tolerance:
                return trial_state(T_high)

        if property_name == 'H':
            newton = self._newton_temperature_for_enthalpy(
                P,
                F,
                composition,
                target,
                seed,
                force_phase=None,
                T_bounds=(T_low, T_high),
                allow_two_phase=True,
            )
            if newton is not None:
                state = trial_state(newton)
                if abs(property_value(state) - target) <= tolerance:
                    return state

        candidates = [
            T_low,
            T_high,
            seed,
            T_low + 1e-6 * (T_high - T_low),
            T_high - 1e-6 * (T_high - T_low),
        ]
        step = 0.02 * (T_high - T_low)
        for index in range(1, 11):
            candidates.append(seed - index * step)
            candidates.append(seed + index * step)
        candidates.extend(
            T_low + (T_high - T_low) * index / 8.0
            for index in range(1, 8)
        )
        grid = sorted(set(
            max(T_low, min(T_high, float(T))) for T in candidates
            if math.isfinite(float(T))
        ))

        evaluated = []
        for T in grid:
            try:
                value = residual(T)
                if math.isfinite(value):
                    evaluated.append((T, value))
            except Exception:
                continue

        for T, value in evaluated:
            if abs(value) <= tolerance:
                return trial_state(T)
        brackets = []
        for (T1, f1), (T2, f2) in zip(evaluated, evaluated[1:]):
            if f1 * f2 < 0.0:
                brackets.append((T1, T2))
        if brackets:
            T1, T2 = min(
                brackets,
                key=lambda bracket: (
                    abs(0.5 * (bracket[0] + bracket[1]) - seed),
                    bracket[1] - bracket[0],
                ),
            )
            T_out = brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=100)
            state = trial_state(T_out)
            if abs(property_value(state) - target) <= tolerance:
                return state
        return None

    def _temperature_grid(self, *anchors: float,
                          T_bounds: tuple[float, float] | None = None) -> list[float]:
        lower = 1.0
        upper = 5000.0
        if T_bounds is not None:
            lower, upper = sorted(float(value) for value in T_bounds)
            lower = max(1.0, lower)
            upper = min(5000.0, upper)
            if upper <= lower:
                upper = min(5000.0, lower + 1e-6)

        candidates = [
            30.0, 50.0, 80.0, 120.0, 160.0, 200.0, 250.0, 298.15, 350.0,
            450.0, 600.0, 800.0, 1100.0, 1500.0, 2000.0, 2500.0, 3200.0,
            4000.0,
        ]
        if T_bounds is not None:
            span = upper - lower
            candidates = [
                lower,
                upper,
                lower + 0.01 * span,
                lower + 0.05 * span,
                lower + 0.10 * span,
                lower + 0.25 * span,
                lower + 0.50 * span,
                lower + 0.75 * span,
                lower + 0.90 * span,
                lower + 0.95 * span,
                lower + 0.99 * span,
            ]
        for value in anchors:
            if value is None or not math.isfinite(float(value)):
                continue
            value = float(value)
            candidates.extend([
                value * 0.25, value * 0.5, value * 0.75, value * 0.9,
                value, value * 1.1, value * 1.25, value * 1.5,
                value * 2.0, value * 3.0, value * 4.0,
                value - 20.0, value - 10.0, value - 2.0,
                value + 2.0, value + 10.0, value + 20.0,
            ])
        return sorted(set(max(lower, min(upper, float(T))) for T in candidates))

    def _solve_temperature(self, residual, candidates: list[float], label: str,
                           preferred_T: float | None = None,
                           residual_tolerance: float = 1e-8) -> float:
        evaluated = []
        for T in candidates:
            try:
                value = residual(T)
                if math.isfinite(value):
                    evaluated.append((T, value))
            except Exception:
                continue

        for T, value in evaluated:
            if abs(value) <= residual_tolerance:
                return T

        brackets = []
        for (T1, f1), (T2, f2) in zip(evaluated, evaluated[1:]):
            if f1 * f2 < 0.0:
                brackets.append((T1, T2))
        if brackets:
            if preferred_T is not None and math.isfinite(float(preferred_T)):
                seed = float(preferred_T)
                T1, T2 = min(
                    brackets,
                    key=lambda bracket: (
                        abs(0.5 * (bracket[0] + bracket[1]) - seed),
                        bracket[1] - bracket[0],
                    ),
                )
            else:
                T1, T2 = min(brackets, key=lambda bracket: bracket[1] - bracket[0])
            return brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=100)

        if evaluated:
            T_best, f_best = min(evaluated, key=lambda item: abs(item[1]))
            raise UnitOperationError(
                f"{self.unit_label} could not bracket {label}; "
                f"best residual {f_best:.4g} at {T_best:.2f} K"
            )
        raise UnitOperationError(f"{self.unit_label} could not evaluate {label}")

    def _clamp_temperature(self, T: float,
                           T_bounds: tuple[float, float] | None) -> float:
        T = max(1.0, min(5000.0, float(T)))
        if T_bounds is None:
            return T
        lower, upper = sorted(float(value) for value in T_bounds)
        lower = max(1.0, lower)
        upper = min(5000.0, upper)
        if upper <= lower:
            return lower
        epsilon = max(1e-8, 1e-10 * (upper - lower))
        return max(lower + epsilon, min(upper - epsilon, T))

    def _newton_temperature_for_enthalpy(self, P: float, F: float,
                                         composition: dict,
                                         H_target: float,
                                         T_guess: float,
                                         force_phase: str | None,
                                         T_bounds: tuple[float, float] | None = None,
                                         allow_two_phase: bool = False) -> float | None:
        if T_guess is None or not math.isfinite(float(T_guess)):
            return None

        tolerance = self._property_tolerance(H_target)
        T = self._clamp_temperature(float(T_guess), T_bounds)
        best = None

        for _iteration in range(12):
            try:
                state = self._calculate_trial_state(
                    T,
                    P,
                    F,
                    composition,
                    include=('H', 'Cp'),
                    force_phase=force_phase,
                )
            except Exception:
                return None
            if state.H is None:
                return None
            residual = state.H - H_target
            if best is None or abs(residual) < abs(best[1]):
                best = (T, residual)
            if abs(residual) <= tolerance:
                return T

            fluid_vapor_fraction = state.fluid_vapor_fraction
            in_two_phase = (
                fluid_vapor_fraction is not None
                and 1e-8 < fluid_vapor_fraction < 1.0 - 1e-8
            )
            if in_two_phase and not allow_two_phase:
                return None

            if in_two_phase and allow_two_phase:
                slope = self._finite_difference_enthalpy_slope(
                    T,
                    P,
                    F,
                    composition,
                    force_phase,
                    T_bounds,
                )
            else:
                Cp = state.Cp
                if Cp is None or not math.isfinite(float(Cp)) or abs(float(Cp)) < 1e-12:
                    return None
                slope = float(Cp)
            if slope is None or not math.isfinite(float(slope)) or abs(float(slope)) < 1e-12:
                return None

            step = residual / float(slope)
            if not math.isfinite(step):
                return None
            span = None
            if T_bounds is not None:
                lower, upper = sorted(float(value) for value in T_bounds)
                span = max(upper - lower, 1e-6)
            max_step = 0.35 * max(abs(T), 50.0) if span is None else 0.5 * span
            step = max(-max_step, min(max_step, step))

            accepted = False
            damping = 1.0
            for _trial in range(8):
                T_next = self._clamp_temperature(T - damping * step, T_bounds)
                if abs(T_next - T) <= max(1e-8, abs(T) * 1e-10):
                    return None
                try:
                    trial = self._calculate_trial_state(
                        T_next,
                        P,
                        F,
                        composition,
                        include=('H',),
                        force_phase=force_phase,
                    )
                    trial_residual = trial.H - H_target
                except Exception:
                    damping *= 0.5
                    continue
                if trial.H is not None and math.isfinite(trial_residual):
                    if abs(trial_residual) <= abs(residual) * 0.9 or abs(trial_residual) <= tolerance:
                        T = T_next
                        accepted = True
                        break
                damping *= 0.5
            if not accepted:
                return best[0] if best is not None and abs(best[1]) <= tolerance else None

        return best[0] if best is not None and abs(best[1]) <= tolerance else None

    def _newton_temperature_for_entropy(self, P: float, F: float,
                                        composition: dict,
                                        S_target: float,
                                        T_guess: float,
                                        force_phase: str | None,
                                        T_bounds: tuple[float, float] | None = None,
                                        allow_two_phase: bool = False) -> float | None:
        if T_guess is None or not math.isfinite(float(T_guess)):
            return None

        tolerance = self._property_tolerance(S_target)
        T = self._clamp_temperature(float(T_guess), T_bounds)
        best = None

        for _iteration in range(12):
            try:
                state = self._calculate_trial_state(
                    T,
                    P,
                    F,
                    composition,
                    include=('S', 'Cp'),
                    force_phase=force_phase,
                )
            except Exception:
                return None
            if state.S is None:
                return None
            residual = state.S - S_target
            if best is None or abs(residual) < abs(best[1]):
                best = (T, residual)
            if abs(residual) <= tolerance:
                return T

            fluid_vapor_fraction = state.fluid_vapor_fraction
            in_two_phase = (
                fluid_vapor_fraction is not None
                and 1e-8 < fluid_vapor_fraction < 1.0 - 1e-8
            )
            if in_two_phase and not allow_two_phase:
                return None

            if in_two_phase and allow_two_phase:
                slope = self._finite_difference_entropy_slope(
                    T,
                    P,
                    F,
                    composition,
                    force_phase,
                    T_bounds,
                )
            else:
                Cp = state.Cp
                if Cp is None or not math.isfinite(float(Cp)) or abs(float(Cp)) < 1e-12:
                    return None
                slope = self._cp_entropy_slope(Cp, T)
            if slope is None or not math.isfinite(float(slope)) or abs(float(slope)) < 1e-12:
                return None

            step = residual / float(slope)
            if not math.isfinite(step):
                return None
            span = None
            if T_bounds is not None:
                lower, upper = sorted(float(value) for value in T_bounds)
                span = max(upper - lower, 1e-6)
            max_step = 0.35 * max(abs(T), 50.0) if span is None else 0.5 * span
            step = max(-max_step, min(max_step, step))

            accepted = False
            damping = 1.0
            for _trial in range(8):
                T_next = self._clamp_temperature(T - damping * step, T_bounds)
                if abs(T_next - T) <= max(1e-8, abs(T) * 1e-10):
                    return None
                try:
                    trial = self._calculate_trial_state(
                        T_next,
                        P,
                        F,
                        composition,
                        include=('S',),
                        force_phase=force_phase,
                    )
                    trial_residual = trial.S - S_target
                except Exception:
                    damping *= 0.5
                    continue
                if trial.S is not None and math.isfinite(trial_residual):
                    if abs(trial_residual) <= abs(residual) * 0.9 or abs(trial_residual) <= tolerance:
                        T = T_next
                        accepted = True
                        break
                damping *= 0.5
            if not accepted:
                return best[0] if best is not None and abs(best[1]) <= tolerance else None

        return best[0] if best is not None and abs(best[1]) <= tolerance else None

    def _local_temperature_bracket_for_enthalpy(self, P: float, F: float,
                                                composition: dict,
                                                H_target: float,
                                                T_guess: float,
                                                force_phase: str | None,
                                                T_bounds: tuple[float, float] | None = None
                                                ) -> float | None:
        if T_guess is None or not math.isfinite(float(T_guess)):
            return None

        tolerance = self._property_tolerance(H_target)
        T_seed = self._clamp_temperature(float(T_guess), T_bounds)

        def residual(T: float) -> float:
            state = self._calculate_trial_state(
                T,
                P,
                F,
                composition,
                include=('H',),
                force_phase=force_phase,
            )
            if state.H is None:
                raise UnitOperationError("Enthalpy was not calculated")
            return state.H - H_target

        try:
            f_seed = residual(T_seed)
        except Exception:
            return None
        if not math.isfinite(f_seed):
            return None
        if abs(f_seed) <= tolerance:
            return T_seed

        direction = -1.0 if f_seed > 0.0 else 1.0
        widths = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
        previous_T = T_seed
        previous_f = f_seed
        for width in widths:
            T_trial = self._clamp_temperature(T_seed + direction * width, T_bounds)
            if abs(T_trial - previous_T) <= max(1e-10, abs(previous_T) * 1e-12):
                continue
            try:
                f_trial = residual(T_trial)
            except Exception:
                continue
            if not math.isfinite(f_trial):
                continue
            if abs(f_trial) <= tolerance:
                return T_trial
            if previous_f * f_trial < 0.0:
                return brentq(
                    residual,
                    min(previous_T, T_trial),
                    max(previous_T, T_trial),
                    xtol=1e-7,
                    rtol=1e-9,
                    maxiter=100,
                )
            previous_T = T_trial
            previous_f = f_trial
        return None

    def _local_temperature_bracket_for_entropy(self, P: float, F: float,
                                               composition: dict,
                                               S_target: float,
                                               T_guess: float,
                                               force_phase: str | None,
                                               T_bounds: tuple[float, float] | None = None
                                               ) -> float | None:
        if T_guess is None or not math.isfinite(float(T_guess)):
            return None

        tolerance = self._property_tolerance(S_target)
        T_seed = self._clamp_temperature(float(T_guess), T_bounds)

        def residual(T: float) -> float:
            state = self._calculate_trial_state(
                T,
                P,
                F,
                composition,
                include=('S',),
                force_phase=force_phase,
            )
            if state.S is None:
                raise UnitOperationError("Entropy was not calculated")
            return state.S - S_target

        try:
            f_seed = residual(T_seed)
        except Exception:
            return None
        if not math.isfinite(f_seed):
            return None
        if abs(f_seed) <= tolerance:
            return T_seed

        direction = -1.0 if f_seed > 0.0 else 1.0
        widths = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
        previous_T = T_seed
        previous_f = f_seed
        for width in widths:
            T_trial = self._clamp_temperature(T_seed + direction * width, T_bounds)
            if abs(T_trial - previous_T) <= max(1e-10, abs(previous_T) * 1e-12):
                continue
            try:
                f_trial = residual(T_trial)
            except Exception:
                continue
            if not math.isfinite(f_trial):
                continue
            if abs(f_trial) <= tolerance:
                return T_trial
            if previous_f * f_trial < 0.0:
                return brentq(
                    residual,
                    min(previous_T, T_trial),
                    max(previous_T, T_trial),
                    xtol=1e-7,
                    rtol=1e-9,
                    maxiter=100,
                )
            previous_T = T_trial
            previous_f = f_trial
        return None

    def _cp_entropy_slope(self, Cp: float, T: float) -> float:
        if T <= 0.0:
            return float('nan')
        return float(Cp) / T

    def _finite_difference_enthalpy_slope(self, T: float, P: float, F: float,
                                          composition: dict,
                                          force_phase: str | None,
                                          T_bounds: tuple[float, float] | None
                                          ) -> float | None:
        if T_bounds is None:
            delta = max(1e-3, abs(T) * 1e-5)
            T_minus = max(1.0, T - delta)
            T_plus = min(5000.0, T + delta)
        else:
            lower, upper = sorted(float(value) for value in T_bounds)
            span = max(upper - lower, 1e-6)
            delta = max(1e-5 * span, 1e-4)
            T_minus = max(lower, T - delta)
            T_plus = min(upper, T + delta)
            if T_minus == T_plus:
                return None

        try:
            minus = self._calculate_trial_state(
                T_minus,
                P,
                F,
                composition,
                include=('H',),
                force_phase=force_phase,
            )
            plus = self._calculate_trial_state(
                T_plus,
                P,
                F,
                composition,
                include=('H',),
                force_phase=force_phase,
            )
        except Exception:
            return None
        if minus.H is None or plus.H is None:
            return None
        return (plus.H - minus.H) / (T_plus - T_minus)

    def _finite_difference_entropy_slope(self, T: float, P: float, F: float,
                                         composition: dict,
                                         force_phase: str | None,
                                         T_bounds: tuple[float, float] | None
                                         ) -> float | None:
        if T_bounds is None:
            delta = max(1e-3, abs(T) * 1e-5)
            T_minus = max(1.0, T - delta)
            T_plus = min(5000.0, T + delta)
        else:
            lower, upper = sorted(float(value) for value in T_bounds)
            span = max(upper - lower, 1e-6)
            delta = max(1e-5 * span, 1e-4)
            T_minus = max(lower, T - delta)
            T_plus = min(upper, T + delta)
            if T_minus == T_plus:
                return None

        try:
            minus = self._calculate_trial_state(
                T_minus,
                P,
                F,
                composition,
                include=('S',),
                force_phase=force_phase,
            )
            plus = self._calculate_trial_state(
                T_plus,
                P,
                F,
                composition,
                include=('S',),
                force_phase=force_phase,
            )
        except Exception:
            return None
        if minus.S is None or plus.S is None:
            return None
        return (plus.S - minus.S) / (T_plus - T_minus)

    def _enthalpy_temperature_bounds(self, P: float, F: float, composition: dict,
                                     H_target: float, T_guess: float,
                                     force_phase: str | None
                                     ) -> tuple[float, float] | None:
        if force_phase is not None:
            return None
        envelope = self._two_phase_envelope_temperatures(P, composition, T_guess)
        if envelope is None:
            return None

        T_low, T_high = envelope
        try:
            low_state = self.thermo.calculate_state(T_low, P, F, composition, include=('H',))
            high_state = self.thermo.calculate_state(T_high, P, F, composition, include=('H',))
        except Exception:
            return None
        if low_state.H is None or high_state.H is None:
            return None

        H_min = min(low_state.H, high_state.H)
        H_max = max(low_state.H, high_state.H)
        tolerance = self._property_tolerance(H_target)
        if H_target < H_min - tolerance:
            return (1.0, T_low)
        if H_target > H_max + tolerance:
            return (T_high, 5000.0)
        return (T_low, T_high)

    def state_at_entropy(self, P: float, F: float, composition: dict,
                         S_target: float, T_guess: float,
                         force_phase: str | None = None) -> StreamState:
        direct = self._direct_state_at_entropy(P, F, composition, S_target, force_phase)
        if direct is not None:
            return direct

        if force_phase is None:
            saturated = self._saturated_state_at_entropy(
                P, F, composition, S_target, T_guess
            )
            if saturated is not None:
                return saturated

        T_newton = self._newton_temperature_for_entropy(
            P,
            F,
            composition,
            S_target,
            T_guess,
            force_phase,
        )
        if T_newton is not None:
            return self._calculate_trial_state(
                T_newton,
                P,
                F,
                composition,
                include=('H', 'S'),
                force_phase=force_phase,
            )

        T_local = self._local_temperature_bracket_for_entropy(
            P,
            F,
            composition,
            S_target,
            T_guess,
            force_phase,
        )
        if T_local is not None:
            return self._calculate_trial_state(
                T_local,
                P,
                F,
                composition,
                include=('H', 'S'),
                force_phase=force_phase,
            )

        if force_phase is None:
            envelope = self._two_phase_envelope_state_at_property(
                P, F, composition, 'S', S_target, T_guess
            )
            if envelope is not None:
                return envelope

        def residual(T: float) -> float:
            state = self._calculate_trial_state(
                T, P, F, composition, include=('S',), force_phase=force_phase
            )
            if state.S is None:
                raise UnitOperationError("Entropy was not calculated")
            return state.S - S_target

        T = self._solve_temperature(
            residual,
            self._temperature_grid(T_guess),
            f"isentropic outlet temperature at {P:g} bar",
            preferred_T=T_guess,
            residual_tolerance=self._property_tolerance(S_target),
        )
        return self._calculate_trial_state(
            T, P, F, composition, include=('H', 'S'), force_phase=force_phase
        )

    def state_at_enthalpy(self, P: float, F: float, composition: dict,
                          H_target: float, T_guess: float,
                          force_phase: str | None = None,
                          include=None,
                          T_bounds: tuple[float, float] | None = None
                          ) -> tuple[StreamState, float]:
        direct = self._direct_state_at_enthalpy(
            P,
            F,
            composition,
            H_target,
            force_phase,
            include=include,
            T_guess=T_guess,
        )
        if direct is not None and (
            T_bounds is None
            or min(T_bounds) <= direct.T <= max(T_bounds)
        ):
            return direct, (direct.H or 0.0) - H_target

        if force_phase is None:
            saturated = self._saturated_state_at_enthalpy(
                P, F, composition, H_target, T_guess
            )
            if saturated is not None:
                return saturated, 0.0

        T_newton = self._newton_temperature_for_enthalpy(
            P,
            F,
            composition,
            H_target,
            T_guess,
            force_phase,
            T_bounds=T_bounds,
        )
        if T_newton is not None:
            state = self._calculate_trial_state(
                T_newton,
                P,
                F,
                composition,
                include=self._include_with(include, 'H'),
                force_phase=force_phase,
            )
            return state, (state.H or 0.0) - H_target

        T_local = self._local_temperature_bracket_for_enthalpy(
            P,
            F,
            composition,
            H_target,
            T_guess,
            force_phase,
            T_bounds=T_bounds,
        )
        if T_local is not None:
            state = self._calculate_trial_state(
                T_local,
                P,
                F,
                composition,
                include=self._include_with(include, 'H'),
                force_phase=force_phase,
            )
            return state, (state.H or 0.0) - H_target

        if force_phase is None:
            envelope = self._two_phase_envelope_state_at_property(
                P, F, composition, 'H', H_target, T_guess
            )
            if envelope is not None:
                return envelope, (envelope.H or 0.0) - H_target

        if T_bounds is None:
            T_bounds = self._enthalpy_temperature_bounds(
                P,
                F,
                composition,
                H_target,
                T_guess,
                force_phase,
            )

            T_bounded_newton = self._newton_temperature_for_enthalpy(
                P,
                F,
                composition,
                H_target,
                T_guess,
                force_phase,
                T_bounds=T_bounds,
            )
            if T_bounded_newton is not None:
                state = self._calculate_trial_state(
                    T_bounded_newton,
                    P,
                    F,
                    composition,
                    include=self._include_with(include, 'H'),
                    force_phase=force_phase,
                )
                return state, (state.H or 0.0) - H_target

            T_bounded_local = self._local_temperature_bracket_for_enthalpy(
                P,
                F,
                composition,
                H_target,
                T_guess,
                force_phase,
                T_bounds=T_bounds,
            )
            if T_bounded_local is not None:
                state = self._calculate_trial_state(
                    T_bounded_local,
                    P,
                    F,
                    composition,
                    include=self._include_with(include, 'H'),
                    force_phase=force_phase,
                )
                return state, (state.H or 0.0) - H_target

        def residual(T: float) -> float:
            state = self._calculate_trial_state(
                T, P, F, composition, include=('H',), force_phase=force_phase
            )
            if state.H is None:
                raise UnitOperationError("Enthalpy was not calculated")
            return state.H - H_target

        T = self._solve_temperature(
            residual,
            self._temperature_grid(T_guess, T_bounds=T_bounds),
            f"outlet enthalpy at {P:g} bar",
            preferred_T=T_guess,
            residual_tolerance=self._property_tolerance(H_target),
        )
        state = self._calculate_trial_state(
            T,
            P,
            F,
            composition,
            include=self._include_with(include, 'H'),
            force_phase=force_phase,
        )
        return state, (state.H or 0.0) - H_target


class Mixer(UnitOperation):
    """Mix streams with optional outlet pressure, temperature, or duty specs."""

    supports_permanent_solids = True
    particle_size_behavior = 'nonselective'

    def particle_size_sources_for_outlet(self, outlet_name, inlets, result):
        return inlets.values()
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if not inlets:
            raise UnitOperationError(f"Mixer '{self.unit_id}' has no inlet streams")
        
        mode = str(self.get_param('mode', 'adiabatic')).strip().lower()
        P_drop = float(self.get_param('P_drop', 0))
        T_out_spec = self.get_temperature_param('T_out')
        if T_out_spec is None:
            T_out_spec = self.get_temperature_param('T')
        if T_out_spec is None:
            T_out_spec = self.get_temperature_param('temperature')
        Q_spec = self._heat_duty_spec()
        if T_out_spec is not None and Q_spec is not None:
            raise UnitOperationError(
                f"Mixer '{self.unit_id}' cannot specify both outlet temperature and heat duty"
            )

        P_out, pressure_policy = self._outlet_pressure(inlets, P_drop)
        adjusted_inlets = {
            port: self._let_down_to_pressure(stream, P_out)
            for port, stream in inlets.items()
        }
        
        # Sum up flows and component moles
        total_F = 0.0
        total_H = 0.0
        component_moles = {}
        
        for port, stream in adjusted_inlets.items():
            total_F += stream.F
            total_H += stream.F * self._stream_enthalpy(stream)
            
            for comp, z in stream.composition.items():
                moles = stream.F * z
                component_moles[comp] = component_moles.get(comp, 0) + moles
        
        if total_F <= 0:
            raise UnitOperationError(f"Mixer '{self.unit_id}' has zero total flow")
        
        # Calculate outlet composition
        composition = {comp: moles / total_F for comp, moles in component_moles.items()}

        T_guess = sum(s.T * s.F for s in adjusted_inlets.values()) / total_F
        Q = 0.0
        outlet = None

        if T_out_spec is not None:
            T_out = T_out_spec
        elif Q_spec is not None:
            Q = Q_spec
            H_target = (total_H + Q) / total_F
            outlet, _ = self._find_state_for_H(composition, P_out, total_F, H_target, T_guess)
            T_out = outlet.T
        elif mode in ('isothermal', 'fixed_t', 'fixed-temperature', 'fixed_temperature'):
            # Legacy behavior: no explicit T_out means hold the mixed outlet at
            # the flow-weighted post-letdown inlet temperature.
            T_out = T_guess
        else:
            H_target = total_H / total_F
            outlet, _ = self._find_state_for_H(composition, P_out, total_F, H_target, T_guess)
            T_out = outlet.T
        
        if outlet is None:
            outlet = self.thermo.calculate_state(T_out, P_out, total_F, composition)
        
        if T_out_spec is not None or mode in ('isothermal', 'fixed_t', 'fixed-temperature', 'fixed_temperature'):
            Q = total_F * self._stream_enthalpy(outlet) - total_H
        
        return UnitResult(
            outlet_streams={'out': outlet},
            heat_duty=Q,
            performance={
                'mode': mode,
                'n_inlets': len(inlets),
                'pressure_policy': pressure_policy,
                'P_out': P_out,
                'T_out_C': T_out - 273.15,
                'duty_kW': Q / 3600,
                'inlet_pressure_range_bar': [
                    min(stream.P for stream in inlets.values()),
                    max(stream.P for stream in inlets.values()),
                ],
            }
        )

    def _heat_duty_spec(self):
        for name in ('Q', 'duty', 'heat_duty'):
            value = self.get_param(name)
            if value is None:
                continue
            duty = float(value)
            unit = (self.get_param_unit(name) or '').strip().lower()
            if unit in ('w', 'watt', 'watts'):
                return duty * 3.6
            if unit in ('mw', 'megawatt', 'megawatts'):
                return duty * 3.6e6
            if unit in ('kj/h', 'kj/hr', 'kj per h', 'kj per hour'):
                return duty
            if unit in ('j/h', 'j/hr', 'j per h', 'j per hour'):
                return duty / 1000.0
            # Match existing heater convention: plain small duties are kW.
            return duty * 3600.0 if abs(duty) < 1e6 else duty
        return None

    def _outlet_pressure(self, inlets: dict[str, StreamState], P_drop: float) -> tuple[float, str]:
        explicit = self.get_param('P_out', self.get_param('P', self.get_param('pressure')))
        min_P = min(stream.P for stream in inlets.values())
        if explicit is None:
            P_out = min_P - P_drop
            policy = 'auto_valve_to_lowest_inlet_pressure'
        else:
            P_out = float(explicit)
            policy = 'specified_outlet_pressure'
            if P_out > min_P + 1e-9:
                raise UnitOperationError(
                    f"Mixer '{self.unit_id}' outlet pressure ({P_out:g} bar) is above "
                    f"the lowest inlet pressure ({min_P:g} bar); use an upstream pump "
                    "or compressor before mixing"
                )
        if P_out <= 0.0:
            raise UnitOperationError(
                f"Mixer '{self.unit_id}' outlet pressure must be positive"
            )
        return P_out, policy

    def _stream_enthalpy(self, stream: StreamState) -> float:
        if stream.H is not None:
            return stream.H
        return self.thermo.mixture_enthalpy(
            stream.composition,
            stream.T,
            stream.vapor_fraction,
            stream.x,
            stream.y,
            stream.P,
        )

    def _let_down_to_pressure(self, stream: StreamState, P_out: float) -> StreamState:
        if abs(stream.P - P_out) <= 1e-9:
            return stream.copy()
        if P_out > stream.P + 1e-9:
            raise UnitOperationError(
                f"Mixer '{self.unit_id}' cannot raise inlet stream pressure from "
                f"{stream.P:g} to {P_out:g} bar"
            )
        state, _ = self._find_state_for_H(
            stream.composition,
            P_out,
            stream.F,
            self._stream_enthalpy(stream),
            stream.T,
        )
        return state
    
    def _find_T_for_H(self, composition: dict, P: float, H_target: float, 
                      T_guess: float) -> float:
        """Find temperature that gives target enthalpy"""
        state, _ = self._find_state_for_H(composition, P, 1.0, H_target, T_guess)
        return state.T

    def _find_state_for_H(self, composition: dict, P: float, F: float,
                          H_target: float, T_guess: float) -> tuple[StreamState, float]:
        try:
            return _ThermoStateSolver(
                self.thermo,
                f"Mixer '{self.unit_id}'",
            ).state_at_enthalpy(
                P, F, composition, H_target, T_guess
            )
        except UnitOperationError as exc:
            raise UnitOperationError(
                f"Mixer '{self.unit_id}' could not satisfy target enthalpy at "
                f"{P:g} bar"
            ) from exc


class Splitter(UnitOperation):
    """Split or component-split a stream into named outlet streams."""

    supports_permanent_solids = True
    particle_size_behavior = 'nonselective'
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if not inlets:
            raise UnitOperationError(f"Splitter '{self.unit_id}' has no inlet stream")
        if len(inlets) != 1:
            raise UnitOperationError(
                f"Splitter '{self.unit_id}' requires exactly one inlet stream"
            )
        inlet = list(inlets.values())[0]

        component_splits = self._component_splits()
        unknown_components = [
            comp for comp in component_splits
            if comp not in inlet.composition
        ]
        if unknown_components:
            raise UnitOperationError(
                f"Splitter '{self.unit_id}' component_splits reference unknown "
                f"component(s): {', '.join(unknown_components)}"
            )
        outlet_names = self._outlet_names(component_splits)
        if component_splits:
            outlets, split_fracs, component_report = self._component_split_outlets(
                inlet, outlet_names, component_splits
            )
            mode = 'component_split'
        else:
            outlets, split_fracs = self._bulk_split_outlets(inlet, outlet_names)
            component_report = {}
            mode = 'bulk_split'

        inlet_enthalpy_flow = inlet.F * self._stream_enthalpy(inlet)
        outlet_enthalpy_flow = sum(
            outlet.F * self._stream_enthalpy(outlet)
            for outlet in outlets.values()
        )
        heat_duty = outlet_enthalpy_flow - inlet_enthalpy_flow
        
        return UnitResult(
            outlet_streams=outlets,
            heat_duty=heat_duty,
            performance={
                'mode': mode,
                'outlet_names': list(outlets),
                'split_fractions': split_fracs,
                'component_splits': component_report,
                'implicit_state_change_duty_kW': heat_duty / 3600,
            }
        )

    def _stream_enthalpy(self, stream: StreamState) -> float:
        if stream.H is not None:
            return stream.H
        return self.thermo.mixture_enthalpy(
            stream.composition,
            stream.T,
            stream.vapor_fraction,
            stream.x,
            stream.y,
            stream.P,
        )

    def _outlet_names(self, component_splits: dict[str, dict[str, float]]) -> list[str]:
        raw = self.get_param('outlets', self.get_param('outlet_names'))
        if raw is not None:
            names = [part.strip() for part in str(raw).replace(';', ',').split(',') if part.strip()]
            if len(names) < 2:
                raise UnitOperationError(
                    f"Splitter '{self.unit_id}' requires at least two outlet names"
                )
            return names

        names = []
        for splits in component_splits.values():
            for outlet in splits:
                if outlet not in names:
                    names.append(outlet)
        if len(names) >= 2:
            return names

        count = self._legacy_split_count()
        return ['out'] + [f'out{i}' for i in range(2, count + 1)]

    def _legacy_split_count(self) -> int:
        specified = 0
        for i in range(10):
            key = f'split_frac{i+1}' if i > 0 else 'split_frac'
            if self.get_param(key) is not None:
                specified += 1
        return max(2, specified + 1)

    def _bulk_split_outlets(self, inlet: StreamState, outlet_names: list[str]):
        flows = self._specified_outlet_flows(inlet, outlet_names)
        split_fracs = [flow / inlet.F if inlet.F > 0 else 0.0 for flow in flows]
        outlets = {}
        for name, flow in zip(outlet_names, flows):
            outlets[name] = self._make_outlet_state(inlet, name, flow, inlet.composition)
        return outlets, split_fracs

    def _specified_outlet_flows(self, inlet: StreamState, outlet_names: list[str]) -> list[float]:
        if inlet.F <= 0.0:
            raise UnitOperationError(f"Splitter '{self.unit_id}' inlet flow must be positive")

        fractions = self._fraction_specs(outlet_names)
        molar_flows = self._flow_specs(inlet, outlet_names, mass=False)
        mass_flows = self._flow_specs(inlet, outlet_names, mass=True)
        if (
            any(value is not None for value in molar_flows)
            and any(value is not None for value in mass_flows)
        ):
            raise UnitOperationError(
                f"Splitter '{self.unit_id}' cannot mix molar-flow and mass-flow split specs"
            )

        if any(value is not None for value in mass_flows):
            mw = inlet.MW or self.thermo.mixture_MW(inlet.composition)
            if mw <= 0.0:
                raise UnitOperationError(
                    f"Splitter '{self.unit_id}' cannot convert mass flow without mixture MW"
            )
            molar_flows = [None if value is None else value / mw for value in mass_flows]

        if any(value is not None for value in molar_flows):
            combined_flows = []
            for flow, frac in zip(molar_flows, fractions):
                if flow is not None and frac is not None:
                    raise UnitOperationError(
                        f"Splitter '{self.unit_id}' cannot specify both flow and fraction "
                        "for the same outlet"
                    )
                combined_flows.append(flow if flow is not None else (frac * inlet.F if frac is not None else None))
            return self._complete_flows(combined_flows, inlet.F)
        if any(value is not None for value in fractions):
            return [frac * inlet.F for frac in self._complete_fractions(fractions)]

        if len(outlet_names) == 2:
            legacy = self.get_param('split_frac')
            if legacy is not None:
                frac = float(legacy)
                return [frac * inlet.F, (1.0 - frac) * inlet.F]
        return [inlet.F / len(outlet_names) for _ in outlet_names]

    def _fraction_specs(self, outlet_names: list[str]) -> list[float | None]:
        values = [None for _ in outlet_names]
        bulk = self._parse_named_values(self.get_param('split_fracs', self.get_param('split_fractions')))
        for index, name in enumerate(outlet_names):
            value = self._first_param(
                f'split_frac_{name}',
                f'{name}_split_frac',
                f'frac_{name}',
                f'{name}_fraction',
            )
            if value is None:
                value = bulk.get(name)
            if value is None:
                key = f'split_frac{index + 1}' if index > 0 else 'split_frac'
                value = self.get_param(key)
            values[index] = None if value is None else float(value)
        return values

    def _flow_specs(self, inlet: StreamState, outlet_names: list[str], *, mass: bool) -> list[float | None]:
        values = [None for _ in outlet_names]
        if mass:
            bulk = self._parse_named_values(self.get_param('mass_flows', self.get_param('outlet_mass_flows')))
        else:
            bulk = self._parse_named_values(self.get_param('flows', self.get_param('outlet_flows')))
        for index, name in enumerate(outlet_names):
            if mass:
                value, unit = self._first_param_with_unit(
                    f'mass_flow_{name}', f'{name}_mass_flow',
                    f'F_mass_{name}', f'{name}_F_mass',
                )
            else:
                value, unit = self._first_param_with_unit(
                    f'F_{name}', f'{name}_F',
                    f'flow_{name}', f'{name}_flow',
                    f'molar_flow_{name}', f'{name}_molar_flow',
                )
            if value is None:
                value = bulk.get(name)
                unit = ''
            if value is None:
                values[index] = None
            elif mass:
                values[index] = self._convert_mass_flow(float(value), unit)
            else:
                values[index] = self._convert_molar_flow(float(value), unit)
        return values

    def _complete_fractions(self, fractions: list[float | None]) -> list[float]:
        specified = [float(value) for value in fractions if value is not None]
        if any(value < 0.0 for value in specified):
            raise UnitOperationError(f"Splitter '{self.unit_id}' split fractions must be nonnegative")
        total = sum(specified)
        if total > 1.0 + 1e-12:
            raise UnitOperationError(f"Splitter '{self.unit_id}' split fractions exceed 1")
        missing = [index for index, value in enumerate(fractions) if value is None]
        if missing:
            remainder = (1.0 - total) / len(missing)
            return [
                remainder if value is None else float(value)
                for value in fractions
            ]
        if abs(total - 1.0) > 1e-9:
            raise UnitOperationError(f"Splitter '{self.unit_id}' split fractions must sum to 1")
        return [float(value) for value in fractions]

    def _complete_flows(self, flows: list[float | None], total_flow: float) -> list[float]:
        specified = [float(value) for value in flows if value is not None]
        if any(value < 0.0 for value in specified):
            raise UnitOperationError(f"Splitter '{self.unit_id}' outlet flows must be nonnegative")
        total = sum(specified)
        if total > total_flow + 1e-9:
            raise UnitOperationError(f"Splitter '{self.unit_id}' outlet flows exceed inlet flow")
        missing = [index for index, value in enumerate(flows) if value is None]
        if missing:
            remainder = (total_flow - total) / len(missing)
            return [
                remainder if value is None else float(value)
                for value in flows
            ]
        if abs(total - total_flow) > max(1e-9, 1e-9 * total_flow):
            raise UnitOperationError(f"Splitter '{self.unit_id}' outlet flows must sum to inlet flow")
        return [float(value) for value in flows]

    def _component_split_outlets(self, inlet: StreamState, outlet_names: list[str],
                                 component_splits: dict[str, dict[str, float]]):
        base_fracs = [
            flow / inlet.F if inlet.F > 0 else 0.0
            for flow in self._specified_outlet_flows(inlet, outlet_names)
        ]
        outlet_moles = {name: {} for name in outlet_names}
        component_report = {}
        for comp, z in inlet.composition.items():
            comp_total = inlet.F * z
            splits = dict(component_splits.get(comp, {}))
            if not splits:
                splits = {name: frac for name, frac in zip(outlet_names, base_fracs)}
            fractions = self._complete_component_split(comp, outlet_names, splits)
            component_report[comp] = fractions
            for name, frac in fractions.items():
                outlet_moles[name][comp] = comp_total * frac

        outlets = {}
        split_fracs = []
        for name in outlet_names:
            total = sum(outlet_moles[name].values())
            split_fracs.append(total / inlet.F if inlet.F > 0.0 else 0.0)
            if total > 0.0:
                composition = {
                    comp: value / total
                    for comp, value in outlet_moles[name].items()
                    if value > 0.0
                }
            else:
                composition = dict(inlet.composition)
            outlets[name] = self._make_outlet_state(inlet, name, total, composition)
        return outlets, split_fracs, component_report

    def _complete_component_split(self, comp: str, outlet_names: list[str],
                                  splits: dict[str, float]) -> dict[str, float]:
        unknown = [name for name in splits if name not in outlet_names]
        if unknown:
            raise UnitOperationError(
                f"Splitter '{self.unit_id}' component split for {comp} references "
                f"unknown outlet(s): {', '.join(unknown)}"
            )
        values = {name: float(splits.get(name, 0.0)) for name in outlet_names}
        if any(value < 0.0 for value in values.values()):
            raise UnitOperationError(
                f"Splitter '{self.unit_id}' component split fractions must be nonnegative"
            )
        total = sum(values.values())
        if total > 1.0 + 1e-12:
            raise UnitOperationError(
                f"Splitter '{self.unit_id}' component split fractions for {comp} exceed 1"
            )
        if total < 1.0 - 1e-12:
            missing = [name for name in outlet_names if name not in splits]
            target = missing[-1] if missing else outlet_names[-1]
            values[target] += 1.0 - total
        return values

    def _component_splits(self) -> dict[str, dict[str, float]]:
        raw = self.get_param('component_splits', self.get_param('component_split'))
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return {
                str(comp): {str(outlet): float(frac) for outlet, frac in splits.items()}
                for comp, splits in raw.items()
            }
        result: dict[str, dict[str, float]] = {}
        for block in str(raw).split(';'):
            block = block.strip()
            if not block:
                continue
            if ':' not in block:
                raise UnitOperationError(
                    f"Splitter '{self.unit_id}' component_splits entries must use "
                    "component:outlet=fraction syntax"
                )
            comp, rest = block.split(':', 1)
            comp = comp.strip()
            splits = {}
            for item in rest.split(','):
                item = item.strip()
                if not item:
                    continue
                if '=' in item:
                    outlet, value = item.split('=', 1)
                elif ':' in item:
                    outlet, value = item.split(':', 1)
                else:
                    raise UnitOperationError(
                        f"Splitter '{self.unit_id}' component_splits entry '{item}' "
                        "must use outlet=fraction"
                    )
                splits[outlet.strip()] = float(value.strip())
            result[comp] = splits
        return result

    def _make_outlet_state(self, inlet: StreamState, outlet_name: str, flow: float,
                           composition: dict[str, float]) -> StreamState:
        T = self._outlet_temperature(inlet, outlet_name)
        P = self._outlet_pressure(inlet, outlet_name)
        phase = self._outlet_phase(outlet_name)
        changed = (
            abs(T - inlet.T) > 1e-12
            or abs(P - inlet.P) > 1e-12
            or phase is not None
            or set(composition) != set(inlet.composition)
            or any(abs(composition.get(comp, 0.0) - inlet.composition.get(comp, 0.0)) > 1e-12 for comp in composition)
        )
        if not changed:
            outlet = inlet.copy()
            scale = flow / inlet.F if inlet.F > 0.0 else 0.0
            outlet.F = flow
            outlet.solid_component_flows = {
                component: component_flow * scale
                for component, component_flow in inlet.solid_component_flows.items()
            }
            return outlet
        outlet = self.thermo.calculate_state(T, P, flow, composition, phase=phase, flash=(phase is None))
        vapor_fraction = self._first_param(
            f'vapor_fraction_{outlet_name}',
            f'{outlet_name}_vapor_fraction',
            f'VF_{outlet_name}',
            f'{outlet_name}_VF',
        )
        if vapor_fraction is not None:
            outlet.vapor_fraction = float(vapor_fraction)
        return outlet

    def _outlet_temperature(self, inlet: StreamState, outlet_name: str) -> float:
        value, unit, param_name = self._first_param_with_unit_and_name(
            f'T_{outlet_name}', f'{outlet_name}_T',
            f'T_out_{outlet_name}', f'{outlet_name}_T_out',
        )
        if value is None:
            return inlet.T
        value = float(value)
        if unit and self._param_was_preconverted_temperature(param_name):
            return value
        unit = (unit or '').strip().lower()
        if unit in ('c', '°c', 'celsius'):
            return value + 273.15
        if unit in ('f', '°f', 'fahrenheit'):
            return (value - 32.0) * 5.0 / 9.0 + 273.15
        return value + 273.15 if value < 200.0 else value

    def _outlet_pressure(self, inlet: StreamState, outlet_name: str) -> float:
        value, unit, param_name = self._first_param_with_unit_and_name(
            f'P_{outlet_name}', f'{outlet_name}_P',
            f'P_out_{outlet_name}', f'{outlet_name}_P_out',
        )
        pressure = inlet.P if value is None else self._convert_pressure(float(value), unit, param_name)
        drop, drop_unit, drop_param = self._first_param_with_unit_and_name(
            f'P_drop_{outlet_name}', f'{outlet_name}_P_drop',
        )
        if drop is None:
            drop, drop_unit, drop_param = self._first_param_with_unit_and_name('P_drop')
        pressure -= self._convert_pressure(float(drop or 0.0), drop_unit, drop_param)
        if pressure <= 0.0:
            raise UnitOperationError(
                f"Splitter '{self.unit_id}' outlet pressure for {outlet_name} must be positive"
            )
        return pressure

    def _outlet_phase(self, outlet_name: str):
        phase = self._first_param(
            f'phase_{outlet_name}',
            f'{outlet_name}_phase',
        )
        if phase is None:
            return None
        phase = str(phase).strip().lower()
        if phase in ('gas', 'vapour'):
            phase = 'vapor'
        if phase not in ('vapor', 'liquid'):
            raise UnitOperationError(
                f"Splitter '{self.unit_id}' outlet phase for {outlet_name} must be vapor or liquid"
            )
        return phase

    def _first_param(self, *names):
        for name in names:
            value = self.get_param(name)
            if value is not None:
                return value
        return None

    def _first_param_with_unit(self, *names):
        value, unit, _name = self._first_param_with_unit_and_name(*names)
        return value, unit

    def _first_param_with_unit_and_name(self, *names):
        for name in names:
            value = self.get_param(name)
            if value is not None:
                return value, self.get_param_unit(name), name
        return None, '', ''

    def _param_was_preconverted_temperature(self, name: str) -> bool:
        lower = (name or '').lower()
        return lower == 't' or 't_' in lower

    def _param_was_preconverted_pressure(self, name: str) -> bool:
        lower = (name or '').lower()
        return lower in {
            'p', 'p_out', 'p_top', 'p_bottom', 'p_condenser', 'p_reboiler',
            'p_drop', 'p_drop_hot', 'p_drop_cold', 'p_drop_per_stage',
            'stage_pressures', 'pressure_profile',
        } or 'pressure' in lower

    def _convert_pressure(self, value: float, unit: str | None, param_name: str) -> float:
        if unit and self._param_was_preconverted_pressure(param_name):
            return value
        return pressure_to_bar(value, unit)

    def _convert_mass_flow(self, value: float, unit: str | None) -> float:
        return mass_flow_to_kg_per_hour(value, unit)

    def _convert_molar_flow(self, value: float, unit: str | None) -> float:
        return molar_flow_to_kmol_per_hour(value, unit)

    def _parse_named_values(self, raw) -> dict[str, float]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return {str(key): float(value) for key, value in raw.items()}
        values = {}
        for item in str(raw).replace(';', ',').split(','):
            item = item.strip()
            if not item:
                continue
            if ':' in item:
                key, value = item.split(':', 1)
            elif '=' in item:
                key, value = item.split('=', 1)
            else:
                raise UnitOperationError(
                    f"Splitter '{self.unit_id}' named split item '{item}' must use name:value"
                )
            values[key.strip()] = float(value.strip())
        return values


class Pump(UnitOperation):
    """Increase liquid pressure"""
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if not inlets:
            raise UnitOperationError(f"Pump '{self.unit_id}' has no inlet stream")
        inlet = list(inlets.values())[0]
        
        P_out, pressure_spec = self._outlet_pressure(inlet.P)
        eta = float(self.get_param('eta', 0.75))
        eta_mech = float(self.get_param('eta_mech', 0.95))
        if not 0.0 < eta <= 1.0:
            raise UnitOperationError(
                f"Pump '{self.unit_id}' requires hydraulic efficiency eta in (0, 1]"
            )
        if not 0.0 < eta_mech <= 1.0:
            raise UnitOperationError(
                f"Pump '{self.unit_id}' requires mechanical efficiency eta_mech in (0, 1]"
            )
        
        warnings = []
        if inlet.vapor_fraction > 0.01:
            raise UnitOperationError(
                f"Pump '{self.unit_id}' inlet has vapor fraction {inlet.vapor_fraction:.2f} - "
                "pumps require liquid feed"
            )
        if inlet.vapor_fraction > 1e-6:
            warnings.append(
                f"Pump inlet has trace vapor fraction {inlet.vapor_fraction:.3g}; "
                "treating feed as liquid"
            )
        
        dP = P_out - inlet.P  # bar
        if dP <= 0.0:
            raise UnitOperationError(
                f"Pump '{self.unit_id}' outlet pressure ({P_out:g} bar) must be "
                f"greater than inlet pressure ({inlet.P:g} bar)"
            )
        
        # Incompressible liquid hydraulic work: integral(V dP). Stream rho is
        # molar density [kmol/m3], so V_molar is its reciprocal [m3/kmol].
        if inlet.rho and inlet.rho > 0:
            V_molar = 1.0 / inlet.rho  # m3/kmol
            hydraulic_work = V_molar * dP * 100  # bar to kPa, result in kJ/kmol
            liquid_volume_basis = 'inlet_stream_density'
        else:
            hydraulic_work = 0.1 * dP  # kJ/kmol rough estimate
            V_molar = None
            liquid_volume_basis = 'rough_default'
            warnings.append(
                f"Pump '{self.unit_id}' inlet density is unavailable; using rough "
                "liquid-volume estimate for hydraulic work"
            )
        
        fluid_work = hydraulic_work / eta
        shaft_work = fluid_work / eta_mech
        mechanical_loss = shaft_work - fluid_work
        
        # Adiabatic pump: hydraulic inefficiency appears as fluid enthalpy rise.
        # Mechanical losses are reported as shaft loss and are not added to the
        # process stream.
        H_in = self._stream_enthalpy(inlet)
        H_target = H_in + fluid_work
        outlet, enthalpy_error = self._find_liquid_state_for_H(
            inlet.composition,
            P_out,
            inlet.F,
            H_target,
            inlet.T,
        )
        if abs(enthalpy_error) > max(1e-5, abs(H_target) * 1e-8):
            warnings.append(
                f"Pump '{self.unit_id}' outlet enthalpy residual is "
                f"{enthalpy_error:.3g} kJ/kmol"
            )
        try:
            equilibrium_outlet = self.thermo.calculate_state(
                outlet.T,
                P_out,
                inlet.F,
                inlet.composition,
            )
            if equilibrium_outlet.vapor_fraction > 1e-6:
                warnings.append(
                    f"Pump '{self.unit_id}' equilibrium outlet check has vapor fraction "
                    f"{equilibrium_outlet.vapor_fraction:.3g}; forced liquid pump "
                    "calculation may be questionable"
                )
        except Exception:
            warnings.append(
                f"Pump '{self.unit_id}' could not run an equilibrium outlet phase check"
            )

        fluid_power = inlet.F * fluid_work
        shaft_power = inlet.F * shaft_work
        
        return UnitResult(
            outlet_streams={'out': outlet},
            work=fluid_power,
            performance={
                'pressure_spec': pressure_spec,
                'hydraulic_efficiency': eta,
                'efficiency': eta,
                'mechanical_efficiency': eta_mech,
                'pressure_rise': dP,
                'P_in_bar': inlet.P,
                'P_out_bar': P_out,
                'liquid_molar_volume_m3_per_kmol': V_molar,
                'liquid_volume_basis': liquid_volume_basis,
                'hydraulic_work_kJ_per_kmol': hydraulic_work,
                'fluid_work_kJ_per_kmol': fluid_work,
                'shaft_work_kJ_per_kmol': shaft_work,
                'mechanical_loss_kJ_per_kmol': mechanical_loss,
                'hydraulic_power_kW': inlet.F * hydraulic_work / 3600,
                'fluid_power_kW': fluid_power / 3600,
                'shaft_power_kW': shaft_power / 3600,
                'mechanical_loss_kW': inlet.F * mechanical_loss / 3600,
                'power_kW': shaft_power / 3600,
                'enthalpy_rise_kJ_per_kmol': outlet.H - H_in,
                'enthalpy_residual_kJ_per_kmol': enthalpy_error,
                'T_out_C': outlet.T - 273.15,
            },
            warnings=warnings,
        )

    def _outlet_pressure(self, P_in: float) -> tuple[float, str]:
        absolute_specs = [
            name for name in ('P_out', 'P', 'pressure')
            if self.get_param(name) is not None
        ]
        delta_specs = [
            name for name in ('delta_P', 'dP', 'P_rise', 'pressure_rise')
            if self.get_param(name) is not None
        ]
        ratio_specs = [
            name for name in ('pressure_ratio', 'P_ratio', 'ratio')
            if self.get_param(name) is not None
        ]
        spec_count = int(bool(absolute_specs)) + int(bool(delta_specs)) + int(bool(ratio_specs))
        if spec_count > 1:
            raise UnitOperationError(
                f"Pump '{self.unit_id}' can specify only one pressure target: "
                "P_out/P, delta_P, or pressure_ratio"
            )

        if absolute_specs:
            name = absolute_specs[0]
            return float(self.get_param(name)), name
        if delta_specs:
            name = delta_specs[0]
            delta = float(self.get_param(name))
            if delta <= 0.0:
                raise UnitOperationError(
                    f"Pump '{self.unit_id}' requires positive {name}"
                )
            return P_in + delta, name
        if ratio_specs:
            name = ratio_specs[0]
            ratio = float(self.get_param(name))
            if ratio <= 1.0:
                raise UnitOperationError(
                    f"Pump '{self.unit_id}' requires {name} > 1"
                )
            return P_in * ratio, name
        return P_in + 5.0, 'default_delta_P'

    def _stream_enthalpy(self, stream: StreamState) -> float:
        if stream.H is not None:
            return stream.H
        return self.thermo.mixture_enthalpy(
            stream.composition,
            stream.T,
            stream.vapor_fraction,
            stream.x,
            stream.y,
            stream.P,
        )

    def _find_liquid_state_for_H(self, composition: dict, P: float, F: float,
                                 H_target: float, T_guess: float) -> tuple[StreamState, float]:
        return _ThermoStateSolver(
            self.thermo,
            f"Pump '{self.unit_id}'",
        ).state_at_enthalpy(
            P, F, composition, H_target, T_guess, force_phase='liquid'
        )


class _IsentropicPressureMachine(UnitOperation):
    """Shared entropy/enthalpy solve helpers for compressors and expanders."""

    def _reject_unsupported_specs(self):
        if self.get_param('eta_poly') is not None:
            raise UnitOperationError(
                f"{self.__class__.__name__} '{self.unit_id}' does not yet support "
                "polytropic efficiency; use eta_isen"
            )

    def _efficiency(self, primary: str, default: float) -> float:
        value = float(self.get_param(primary, self.get_param('eta', default)))
        if not 0.0 < value <= 1.0:
            raise UnitOperationError(
                f"{self.__class__.__name__} '{self.unit_id}' requires {primary} in (0, 1]"
            )
        return value

    def _mechanical_efficiency(self, default: float = 0.98) -> float:
        value = float(self.get_param('eta_mech', default))
        if not 0.0 < value <= 1.0:
            raise UnitOperationError(
                f"{self.__class__.__name__} '{self.unit_id}' requires eta_mech in (0, 1]"
            )
        return value

    def _outlet_pressure(self, P_in: float, direction: str,
                         default_ratio: float) -> tuple[float, str]:
        absolute_specs = [
            name for name in ('P_out', 'P', 'pressure')
            if self.get_param(name) is not None
        ]
        delta_names = (
            ('delta_P', 'dP', 'P_rise', 'pressure_rise')
            if direction == 'increase'
            else ('delta_P', 'dP', 'P_drop', 'pressure_drop')
        )
        delta_specs = [name for name in delta_names if self.get_param(name) is not None]
        ratio_specs = [
            name for name in ('pressure_ratio', 'P_ratio', 'ratio')
            if self.get_param(name) is not None
        ]
        spec_count = int(bool(absolute_specs)) + int(bool(delta_specs)) + int(bool(ratio_specs))
        if spec_count > 1:
            raise UnitOperationError(
                f"{self.__class__.__name__} '{self.unit_id}' can specify only one "
                "pressure target: P_out/P, delta_P/dP, or pressure_ratio"
            )

        if absolute_specs:
            name = absolute_specs[0]
            P_out = float(self.get_param(name))
        elif delta_specs:
            name = delta_specs[0]
            delta = float(self.get_param(name))
            if delta <= 0.0:
                raise UnitOperationError(
                    f"{self.__class__.__name__} '{self.unit_id}' requires positive {name}"
                )
            P_out = P_in + delta if direction == 'increase' else P_in - delta
        elif ratio_specs:
            name = ratio_specs[0]
            ratio = float(self.get_param(name))
            if ratio <= 1.0:
                raise UnitOperationError(
                    f"{self.__class__.__name__} '{self.unit_id}' requires {name} > 1"
                )
            P_out = P_in * ratio if direction == 'increase' else P_in / ratio
        else:
            name = 'default_pressure_ratio'
            P_out = P_in * default_ratio if direction == 'increase' else P_in / default_ratio

        if P_out <= 0.0:
            raise UnitOperationError(
                f"{self.__class__.__name__} '{self.unit_id}' outlet pressure must be positive"
            )
        if direction == 'increase' and P_out <= P_in:
            raise UnitOperationError(
                f"Compressor '{self.unit_id}' outlet pressure ({P_out:g} bar) must be "
                f"greater than inlet pressure ({P_in:g} bar)"
            )
        if direction == 'decrease' and P_out >= P_in:
            raise UnitOperationError(
                f"Expander '{self.unit_id}' outlet pressure ({P_out:g} bar) must be "
                f"less than inlet pressure ({P_in:g} bar)"
            )
        return P_out, name

    def _stream_enthalpy(self, stream: StreamState) -> float:
        if stream.H is not None:
            return stream.H
        return self.thermo.mixture_enthalpy(
            stream.composition,
            stream.T,
            stream.vapor_fraction,
            stream.x,
            stream.y,
            stream.P,
        )

    def _stream_entropy(self, stream: StreamState) -> float:
        if stream.S is not None:
            return stream.S
        return self.thermo.mixture_entropy(
            stream.composition,
            stream.T,
            stream.vapor_fraction,
            stream.x,
            stream.y,
            stream.P,
        )

    def _state_at_entropy(self, P: float, F: float, composition: dict,
                          S_target: float, T_guess: float,
                          force_vapor: bool) -> StreamState:
        return _ThermoStateSolver(
            self.thermo,
            f"{self.__class__.__name__} '{self.unit_id}'",
        ).state_at_entropy(
            P,
            F,
            composition,
            S_target,
            T_guess,
            force_phase='vapor' if force_vapor else None,
        )

    def _state_at_enthalpy(self, P: float, F: float, composition: dict,
                           H_target: float, T_guess: float,
                           force_vapor: bool) -> tuple[StreamState, float]:
        return _ThermoStateSolver(
            self.thermo,
            f"{self.__class__.__name__} '{self.unit_id}'",
        ).state_at_enthalpy(
            P,
            F,
            composition,
            H_target,
            T_guess,
            force_phase='vapor' if force_vapor else None,
        )

    def _constant_cp_isentropic_temperature_seed(self, inlet: StreamState,
                                                 P_out: float,
                                                 fallback: float) -> float:
        try:
            Cp = inlet.Cp
            if Cp is None:
                Cp = self.thermo.mixture_Cp(
                    inlet.composition,
                    inlet.T,
                    inlet.vapor_fraction,
                )
            Cp_kJ_per_kmol_K = None if Cp is None else float(Cp)
            if (
                Cp_kJ_per_kmol_K is None
                or not math.isfinite(float(Cp_kJ_per_kmol_K))
                or Cp_kJ_per_kmol_K <= 0.0
                or inlet.P <= 0.0
                or P_out <= 0.0
            ):
                return fallback
            exponent = R_J_MOL_K / float(Cp_kJ_per_kmol_K)
            seed = inlet.T * (P_out / inlet.P) ** exponent
            if not math.isfinite(seed) or seed <= 0.0:
                return fallback
            return max(1.0, min(5000.0, seed))
        except Exception:
            return fallback


class Compressor(_IsentropicPressureMachine):
    """Increase gas pressure"""
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        inlet = list(inlets.values())[0]
        self._reject_unsupported_specs()

        P_out, pressure_spec = self._outlet_pressure(
            inlet.P, direction='increase', default_ratio=2.0
        )
        eta_isen = self._efficiency('eta_isen', 0.75)
        eta_mech = self._mechanical_efficiency(0.98)
        
        warnings = []
        if inlet.vapor_fraction < 0.9:
            warnings.append(
                f"Compressor inlet has liquid fraction {1-inlet.vapor_fraction:.2f} - "
                "may cause damage"
            )

        H_in = self._stream_enthalpy(inlet)
        S_in = self._stream_entropy(inlet)
        pressure_ratio = P_out / inlet.P
        isentropic_guess = self._constant_cp_isentropic_temperature_seed(
            inlet,
            P_out,
            inlet.T * pressure_ratio**0.25,
        )
        isentropic = self._state_at_entropy(
            P_out,
            inlet.F,
            inlet.composition,
            S_in,
            max(inlet.T, isentropic_guess),
            force_vapor=True,
        )
        dH_is = (isentropic.H or 0.0) - H_in
        if dH_is < -1e-6:
            raise UnitOperationError(
                f"Compressor '{self.unit_id}' calculated negative isentropic work"
            )

        fluid_work = max(0.0, dH_is) / eta_isen
        H_target = H_in + fluid_work
        actual_guess = inlet.T + max(0.0, isentropic.T - inlet.T) / eta_isen
        outlet, enthalpy_error = self._state_at_enthalpy(
            P_out,
            inlet.F,
            inlet.composition,
            H_target,
            actual_guess,
            force_vapor=True,
        )
        if abs(enthalpy_error) > max(1e-5, abs(H_target) * 1e-8):
            warnings.append(
                f"Compressor '{self.unit_id}' outlet enthalpy residual is "
                f"{enthalpy_error:.3g} kJ/kmol"
            )

        shaft_work = fluid_work / eta_mech
        mechanical_loss = shaft_work - fluid_work
        fluid_power = inlet.F * fluid_work
        shaft_power = inlet.F * shaft_work
        
        return UnitResult(
            outlet_streams={'out': outlet},
            work=fluid_power,
            performance={
                'pressure_spec': pressure_spec,
                'compression_ratio': pressure_ratio,
                'P_in_bar': inlet.P,
                'P_out_bar': P_out,
                'isentropic_efficiency': eta_isen,
                'mechanical_efficiency': eta_mech,
                'T_isentropic_C': isentropic.T - 273.15,
                'T_outlet_C': outlet.T - 273.15,
                'isentropic_work_kJ_per_kmol': max(0.0, dH_is),
                'fluid_work_kJ_per_kmol': fluid_work,
                'shaft_work_kJ_per_kmol': shaft_work,
                'mechanical_loss_kJ_per_kmol': mechanical_loss,
                'fluid_power_kW': fluid_power / 3600,
                'shaft_power_kW': shaft_power / 3600,
                'mechanical_loss_kW': inlet.F * mechanical_loss / 3600,
                'power_kW': shaft_power / 3600,
                'entropy_in_kJ_per_kmol_K': S_in,
                'entropy_out_kJ_per_kmol_K': outlet.S,
                'entropy_generation_kJ_per_kmol_K': None if outlet.S is None else outlet.S - S_in,
                'enthalpy_residual_kJ_per_kmol': enthalpy_error,
            },
            warnings=warnings
        )


class Expander(_IsentropicPressureMachine):
    """Reduce pressure and recover work (turbine)"""
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        inlet = list(inlets.values())[0]
        self._reject_unsupported_specs()

        P_out, pressure_spec = self._outlet_pressure(
            inlet.P, direction='decrease', default_ratio=2.0
        )
        eta_isen = self._efficiency('eta_isen', 0.80)
        eta_mech = self._mechanical_efficiency(0.98)
        warnings = []
        if inlet.vapor_fraction < 0.9:
            warnings.append(
                f"Expander inlet has liquid fraction {1-inlet.vapor_fraction:.2f} - "
                "turbine calculation may be questionable"
            )

        H_in = self._stream_enthalpy(inlet)
        S_in = self._stream_entropy(inlet)
        expansion_ratio = inlet.P / P_out
        isentropic_guess = self._constant_cp_isentropic_temperature_seed(
            inlet,
            P_out,
            inlet.T / expansion_ratio**0.25,
        )
        isentropic = self._state_at_entropy(
            P_out,
            inlet.F,
            inlet.composition,
            S_in,
            min(inlet.T, isentropic_guess),
            force_vapor=False,
        )
        dH_is = H_in - (isentropic.H or 0.0)
        if dH_is < -1e-6:
            raise UnitOperationError(
                f"Expander '{self.unit_id}' calculated negative isentropic work"
            )

        fluid_work_output = max(0.0, dH_is) * eta_isen
        H_target = H_in - fluid_work_output
        actual_guess = inlet.T - eta_isen * max(0.0, inlet.T - isentropic.T)
        outlet, enthalpy_error = self._state_at_enthalpy(
            P_out,
            inlet.F,
            inlet.composition,
            H_target,
            actual_guess,
            force_vapor=False,
        )
        if abs(enthalpy_error) > max(1e-5, abs(H_target) * 1e-8):
            warnings.append(
                f"Expander '{self.unit_id}' outlet enthalpy residual is "
                f"{enthalpy_error:.3g} kJ/kmol"
            )

        shaft_work_output = fluid_work_output * eta_mech
        mechanical_loss = fluid_work_output - shaft_work_output
        fluid_power_output = inlet.F * fluid_work_output
        shaft_power_output = inlet.F * shaft_work_output
        
        return UnitResult(
            outlet_streams={'out': outlet},
            work=-fluid_power_output,
            performance={
                'pressure_spec': pressure_spec,
                'expansion_ratio': expansion_ratio,
                'P_in_bar': inlet.P,
                'P_out_bar': P_out,
                'isentropic_efficiency': eta_isen,
                'mechanical_efficiency': eta_mech,
                'T_isentropic_C': isentropic.T - 273.15,
                'T_outlet_C': outlet.T - 273.15,
                'isentropic_work_output_kJ_per_kmol': max(0.0, dH_is),
                'fluid_work_output_kJ_per_kmol': fluid_work_output,
                'shaft_work_output_kJ_per_kmol': shaft_work_output,
                'mechanical_loss_kJ_per_kmol': mechanical_loss,
                'fluid_power_output_kW': fluid_power_output / 3600,
                'shaft_power_output_kW': shaft_power_output / 3600,
                'mechanical_loss_kW': inlet.F * mechanical_loss / 3600,
                'power_recovered_kW': shaft_power_output / 3600,
                'entropy_in_kJ_per_kmol_K': S_in,
                'entropy_out_kJ_per_kmol_K': outlet.S,
                'entropy_generation_kJ_per_kmol_K': None if outlet.S is None else outlet.S - S_in,
                'outlet_vapor_frac': outlet.vapor_fraction,
                'enthalpy_residual_kJ_per_kmol': enthalpy_error,
            },
            warnings=warnings,
        )


class Valve(_IsentropicPressureMachine):
    """Isenthalpic pressure reduction (throttling valve)"""
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        inlet = list(inlets.values())[0]
        
        P_out = float(self.get_param('P_out', self.get_param('P', inlet.P / 2)))
        
        if P_out >= inlet.P:
            raise UnitOperationError(
                f"Valve '{self.unit_id}' outlet pressure ({P_out} bar) must be "
                f"less than inlet ({inlet.P} bar)"
            )
        
        # Isenthalpic flash at the downstream pressure.
        H_target = self._stream_enthalpy(inlet)
        outlet, enthalpy_error = self._state_at_enthalpy(
            P_out,
            inlet.F,
            inlet.composition,
            H_target,
            inlet.T,
            force_vapor=False,
        )
        warnings = []
        if abs(enthalpy_error) > 10.0:
            warnings.append(
                f"Valve '{self.unit_id}' isenthalpic flash residual is "
                f"{enthalpy_error:.3g} kJ/kmol"
            )
        
        return UnitResult(
            outlet_streams={'out': outlet},
            warnings=warnings,
            performance={
                'pressure_drop': inlet.P - P_out,
                'inlet_vapor_frac': inlet.vapor_fraction,
                'outlet_vapor_frac': outlet.vapor_fraction,
                'joule_thomson_dT': outlet.T - inlet.T,
                'enthalpy_error_kJ_per_kmol': enthalpy_error,
            }
        )


class Heater(UnitOperation):
    """Add heat to stream"""
    supports_permanent_solids = True
    particle_size_behavior = 'nonselective'
    heat_direction = 1
    direction_label = 'heat'
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if not inlets:
            raise UnitOperationError(f"{self.__class__.__name__} '{self.unit_id}' has no inlet stream")
        inlet = list(inlets.values())[0]
        
        # Get specification (T_out, Q, or vap_frac)
        T_out = self.get_temperature_param('T_out')
        if T_out is None:
            T_out = self.get_temperature_param('T')
        Q = self._heat_duty_spec()
        vap_frac = self._vapor_fraction_spec()
        P_drop = float(self.get_param('P_drop', 0))
        if P_drop < 0.0:
            raise UnitOperationError(
                f"{self.__class__.__name__} '{self.unit_id}' requires nonnegative P_drop"
            )
        
        P_out = inlet.P - P_drop
        if P_out <= 0.0:
            raise UnitOperationError(
                f"{self.__class__.__name__} '{self.unit_id}' outlet pressure must be positive"
            )

        spec_count = sum(value is not None for value in (T_out, Q, vap_frac))
        if spec_count != 1:
            raise UnitOperationError(
                f"{self.__class__.__name__} '{self.unit_id}' requires exactly one of "
                "T_out/T, Q/duty/heat_duty, or vapor_fraction/vap_frac/VF"
            )

        henry_controls = (
            'henry_components',
            'water_component',
            'henry_water_mole_fraction_min',
            'henry_water_cutoff',
            'henry_dilute_mole_fraction_max',
            'henry_dilute_cutoff',
            'henry_pressure_warning_bar',
        )
        henry_requested = any(
            self.get_param(name) is not None for name in henry_controls
        )
        henry_helper = None
        aqueous_context = None
        henry_info = None
        henry_warnings = []
        composition = dict(inlet.composition)
        inlet_H = inlet.H
        if henry_requested:
            # Flash owns the shared single-stream Henry-aware TP/PV/PH state
            # construction. Reuse that exact implementation so a thermal unit
            # and its downstream separator cannot disagree on phase behavior.
            henry_helper = Flash(self.unit_id, self.thermo, self.params)
            composition = henry_helper._normalized_composition(composition)
            reference_T = T_out if T_out is not None else inlet.T
            aqueous_context, henry_info, henry_warnings = (
                henry_helper._flash_henry_context(
                    inlet,
                    composition,
                    reference_T,
                )
            )
            inlet_H = henry_helper._stream_enthalpy(inlet, aqueous_context)
        
        if T_out is not None:
            # Temperature specified
            if aqueous_context is not None:
                outlet = henry_helper._state_at_TP(
                    composition,
                    T_out,
                    P_out,
                    inlet.F,
                    aqueous_context,
                )
            else:
                outlet = self.thermo.calculate_state(
                    T_out, P_out, inlet.F, inlet.composition
                )
            Q = inlet.F * (outlet.H - inlet_H)  # kJ/h
            
        elif Q is not None:
            # Duty specified
            if inlet.F <= 0.0:
                if abs(Q) > 1e-12:
                    raise UnitOperationError(
                        f"{self.__class__.__name__} '{self.unit_id}' cannot apply nonzero "
                        "duty to a zero-flow stream"
                    )
                outlet = inlet.copy()
                outlet.P = P_out
                self._enforce_direction(Q)
                return self._result(outlet, Q, inlet, P_drop)

            H_out = inlet_H + Q / inlet.F
            if aqueous_context is not None:
                outlet = henry_helper._state_at_PH(
                    composition,
                    P_out,
                    H_out,
                    inlet.F,
                    inlet.T,
                    aqueous_context,
                )
                enthalpy_error = outlet.H - H_out
            else:
                outlet, enthalpy_error = self._state_for_enthalpy(
                    inlet.composition, P_out, inlet.F, H_out, inlet.T
                )
            if abs(enthalpy_error) > max(1e-5, abs(H_out) * 1e-8):
                raise UnitOperationError(
                    f"{self.__class__.__name__} '{self.unit_id}' outlet enthalpy "
                    f"residual is {enthalpy_error:.3g} kJ/kmol"
                )
            
        elif vap_frac is not None:
            # Vapor fraction specified
            vap_frac = float(vap_frac)
            if not 0.0 <= vap_frac <= 1.0:
                raise UnitOperationError(
                    f"{self.__class__.__name__} '{self.unit_id}' vapor_fraction must be between 0 and 1"
                )
            if aqueous_context is not None:
                outlet = henry_helper._state_at_PV(
                    composition,
                    P_out,
                    vap_frac,
                    inlet.F,
                    inlet.T,
                    aqueous_context,
                )
            else:
                direct_pq = getattr(self.thermo, 'calculate_state_PQ', None)
                if direct_pq is not None:
                    try:
                        outlet = direct_pq(P_out, vap_frac, inlet.F, inlet.composition)
                    except (NotImplementedError, AttributeError):
                        outlet = None
                else:
                    outlet = None
                if outlet is None:
                    T_out, x, y = self.thermo.flash_PV(inlet.composition, P_out, vap_frac)
                    outlet = self.thermo.calculate_state(
                        T_out, P_out, inlet.F, inlet.composition
                    )
                    outlet.vapor_fraction = vap_frac
                    outlet.x = x
                    outlet.y = y
            Q = inlet.F * (outlet.H - inlet_H)

        self.thermo._record_estimated_interaction_extrapolation(
            outlet.T,
            (
                (outlet.effective_liquid1_fraction, outlet.x1 or outlet.x or {}),
                (outlet.liquid2_fraction, outlet.x2 or {}),
            ),
        )
        if aqueous_context is not None and outlet.effective_liquid1_fraction > 1e-12:
            henry_warnings.extend(self._henry_solution_warnings(
                aqueous_context,
                henry_info,
                [outlet.x1 or outlet.x or {}],
                [outlet.T],
                [outlet.P],
            ))
 
        self._enforce_direction(Q)
        return self._result(
            outlet,
            Q,
            inlet,
            P_drop,
            henry_info=henry_info,
            warnings=henry_warnings,
        )

    def _result(self, outlet: StreamState, Q: float, inlet: StreamState,
                P_drop: float, *, henry_info=None, warnings=None) -> UnitResult:
        performance = {
            'T_in_C': inlet.T - 273.15,
            'T_out_C': outlet.T - 273.15,
            'P_in_bar': inlet.P,
            'P_out_bar': outlet.P,
            'P_drop_bar': P_drop,
            'duty_kW': Q / 3600,
        }
        if henry_info is not None:
            performance['henry'] = henry_info
        return UnitResult(
            outlet_streams={'out': outlet},
            heat_duty=Q,
            performance=performance,
            warnings=list(warnings or ()),
        )

    def _vapor_fraction_spec(self):
        return self.get_param(
            'vap_frac',
            self.get_param(
                'vapor_frac',
                self.get_param('vapor_fraction', self.get_param('VF')),
            ),
        )

    def _heat_duty_spec(self):
        for name in ('Q', 'duty', 'heat_duty'):
            value = self.get_param(name)
            if value is None:
                continue
            duty = float(value)
            unit = (self.get_param_unit(name) or '').strip().lower()
            if unit in ('w', 'watt', 'watts'):
                return duty * 3.6
            if unit in ('mw', 'megawatt', 'megawatts'):
                return duty * 3.6e6
            if unit in ('kw', 'kilowatt', 'kilowatts'):
                return duty * 3600.0
            if unit in ('kj/h', 'kj/hr', 'kj per h', 'kj per hour'):
                return duty
            if unit in ('j/h', 'j/hr', 'j per h', 'j per hour'):
                return duty / 1000.0
            return duty * 3600.0 if abs(duty) < 1e6 else duty
        return None

    def _enforce_direction(self, Q: float) -> None:
        tolerance = max(1e-9, abs(Q) * 1e-12)
        if self.heat_direction > 0 and Q <= tolerance:
            raise UnitOperationError(
                f"Heater '{self.unit_id}' must add heat; calculated duty is {Q / 3600:.6g} kW"
            )
        if self.heat_direction < 0 and Q >= -tolerance:
            raise UnitOperationError(
                f"Cooler '{self.unit_id}' must remove heat; calculated duty is {Q / 3600:.6g} kW"
            )

    def _state_for_enthalpy(self, composition: dict, P: float, F: float,
                            H_target: float, T_guess: float) -> tuple[StreamState, float]:
        return _ThermoStateSolver(
            self.thermo,
            f"{self.__class__.__name__} '{self.unit_id}'",
        ).state_at_enthalpy(P, F, composition, H_target, T_guess)

    def _find_T_for_H(self, composition: dict, P: float, H_target: float,
                      T_guess: float) -> float:
        state, _error = self._state_for_enthalpy(
            composition, P, 1.0, H_target, T_guess
        )
        return state.T


class Cooler(Heater):
    """Remove heat from stream (same math as heater, different context)"""
    heat_direction = -1
    direction_label = 'cool'


class HeatExchanger(UnitOperation):
    """Exchange heat between two streams"""

    supports_permanent_solids = True
    particle_size_behavior = 'nonselective'

    def particle_size_sources_for_outlet(self, outlet_name, inlets, result):
        performance = result.performance
        if outlet_name == performance.get('hot_out_port'):
            return (inlets[performance['hot_in_port']],)
        if outlet_name == performance.get('cold_out_port'):
            return (inlets[performance['cold_in_port']],)
        raise UnitOperationError(
            f"HeatExchanger '{self.unit_id}' cannot map particle population "
            f"for outlet '{outlet_name}'"
        )

    U_ESTIMATE_TABLE = {
        # Values are preliminary dirty-service estimates converted from
        # typical shell-and-tube U ranges in Btu/(hr-ft2-F) to W/(m2-K).
        'liquid_liquid': {'low': 250.0, 'typical': 600.0, 'high': 1400.0},
        'gas_liquid': {'low': 60.0, 'typical': 170.0, 'high': 450.0},
        'gas_gas': {'low': 30.0, 'typical': 85.0, 'high': 280.0},
        'condensing_vapor': {'low': 170.0, 'typical': 570.0, 'high': 1150.0},
        'steam_condensing': {'low': 2250.0, 'typical': 3400.0, 'high': 5700.0},
        'vaporizing': {'low': 850.0, 'typical': 1400.0, 'high': 2250.0},
        'water_boiling': {'low': 1400.0, 'typical': 2200.0, 'high': 3400.0},
    }

    SEADER_U_ESTIMATE_TABLE = {
        # Product and Process Design Principles, 4th ed., Table 12.5.
        # Native values are Btu/(h-ft2-F); _u_service converts to W/(m2-K).
        'aroclor_jet_fuel': _u_service(100, 150, 0.0015),
        'cutback_asphalt_water': _u_service(10, 20, 0.01),
        'demin_water_water': _u_service(300, 500, 0.001),
        'amine_water': _u_service(140, 200, 0.003),
        'fuel_oil_water': _u_service(15, 25, 0.007),
        'fuel_oil_oil': _u_service(10, 15, 0.008),
        'gasoline_water': _u_service(60, 100, 0.003),
        'heavy_oil_heavy_oil': _u_service(10, 40, 0.004),
        'heavy_oil_water': _u_service(15, 50, 0.005),
        'hydrogen_reformer_stream': _u_service(90, 120, 0.002),
        'kerosene_gas_oil_water': _u_service(25, 50, 0.005),
        'kerosene_gas_oil_oil': _u_service(20, 35, 0.005),
        'kerosene_jet_fuel_trichloroethylene': _u_service(40, 50, 0.0015),
        'jacket_water_water': _u_service(230, 300, 0.002),
        'lube_oil_low_water': _u_service(25, 50, 0.002),
        'lube_oil_high_water': _u_service(40, 80, 0.003),
        'lube_oil_oil': _u_service(11, 20, 0.006),
        'naphtha_water': _u_service(50, 70, 0.005),
        'naphtha_oil': _u_service(25, 35, 0.005),
        'organic_solvent_water': _u_service(50, 150, 0.003),
        'organic_solvent_brine': _u_service(35, 90, 0.003),
        'organic_solvent_organic': _u_service(20, 60, 0.002),
        'vegetable_oil_water': _u_service(20, 50, 0.004),
        'water_caustic': _u_service(100, 250, 0.003),
        'water_water': _u_service(200, 250, 0.003),
        'wax_distillate_water': _u_service(15, 25, 0.005),
        'wax_distillate_oil': _u_service(13, 23, 0.005),
        'alcohol_vapor_water': _u_service(100, 200, 0.002),
        'asphalt_dowtherm_vapor': _u_service(40, 60, 0.006),
        'dowtherm_vapor_tall_oil': _u_service(60, 80, 0.004),
        'dowtherm_vapor_dowtherm_liquid': _u_service(80, 120, 0.0015),
        'gas_plant_tar_steam': _u_service(40, 50, 0.0055),
        'high_boiling_hydrocarbon_water': _u_service(20, 50, 0.003),
        'low_boiling_hydrocarbon_water': _u_service(80, 200, 0.003),
        'hydrocarbon_partial_condenser_oil': _u_service(25, 45, 0.004),
        'organic_solvent_vapor_water': _u_service(100, 200, 0.003),
        'organic_solvent_high_nc_water': _u_service(20, 60, 0.003),
        'organic_solvent_low_nc_water': _u_service(50, 120, 0.003),
        'kerosene_condensing_water': _u_service(30, 65, 0.004),
        'kerosene_condensing_oil': _u_service(20, 30, 0.005),
        'naphtha_condensing_water': _u_service(50, 75, 0.005),
        'naphtha_condensing_oil': _u_service(20, 30, 0.005),
        'stabilizer_column_vapor_water': _u_service(80, 120, 0.003),
        'steam_feedwater': _u_service(400, 1000, 0.0005),
        'steam_no6_fuel_oil': _u_service(15, 25, 0.0055),
        'steam_no2_fuel_oil': _u_service(60, 90, 0.0025),
        'sulfur_dioxide_water': _u_service(150, 200, 0.003),
        'tall_oil_vapor_water': _u_service(20, 50, 0.004),
        'aromatic_vapor_water_azeotrope': _u_service(40, 80, 0.005),
        'compressed_air_water': _u_service(40, 80, 0.005),
        'atmospheric_air_water': _u_service(10, 50, 0.005),
        'water_compressed_air': _u_service(20, 40, 0.005),
        'water_atmospheric_air': _u_service(5, 20, 0.005),
        'water_hydrogen_natural_gas': _u_service(80, 125, 0.003),
        'ammonia_steam_vaporizer': _u_service(150, 300, 0.0015),
        'chlorine_steam_vaporizer': _u_service(150, 300, 0.0015),
        'chlorine_heat_transfer_oil_vaporizer': _u_service(40, 60, 0.0015),
        'propane_butane_steam_vaporizer': _u_service(200, 300, 0.0015),
        'water_steam_vaporizer': _u_service(250, 400, 0.0015),
    }
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        inlet_list = list(inlets.items())
        if len(inlet_list) != 2:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' requires exactly 2 inlet streams"
            )
        hot_port, hot, cold_port, cold = self._identify_hot_cold(inlet_list)

        U = self.get_param('U')  # W/m2-K
        A = self.get_param('A', self.get_param('area'))  # m2
        U_value, estimate_U = self._u_spec(U)
        A_value = self._area_spec(A)
        UA_available = self._ua_spec()
        flow_pattern = self._flow_pattern()
        curve_segments = self._curve_segments()
        allow_temperature_cross = self._truthy_param(
            self.get_param('allow_temperature_cross', False)
        )
        if self._truthy_param(self.get_param('estimate_U', False)):
            estimate_U = True

        T_hot_out = self._temperature_spec(
            'T_hot_out', 'hot_T_out', 'hot_out_T'
        )
        T_cold_out = self._temperature_spec(
            'T_cold_out', 'cold_T_out', 'cold_out_T'
        )
        T_tube_out = self._temperature_spec(
            'T_tube_out', 'tube_T_out', 'tube_out_T'
        )
        T_shell_out = self._temperature_spec(
            'T_shell_out', 'shell_T_out', 'shell_out_T'
        )
        Q_spec = self._heat_duty_spec()
        hot_vap_frac = self._vapor_fraction_spec('hot')
        cold_vap_frac = self._vapor_fraction_spec('cold')
        tube_vap_frac = self._vapor_fraction_spec('tube')
        shell_vap_frac = self._vapor_fraction_spec('shell')
        hot_force_phase = self._forced_phase_spec(
            'hot_phase', 'phase_hot', 'force_hot_phase'
        )
        cold_force_phase = self._forced_phase_spec(
            'cold_phase', 'phase_cold', 'force_cold_phase'
        )
        tube_force_phase = self._forced_phase_spec(
            'tube_phase', 'phase_tube', 'force_tube_phase'
        )
        shell_force_phase = self._forced_phase_spec(
            'shell_phase', 'phase_shell', 'force_shell_phase'
        )
        min_approach = float(self.get_param('min_approach', 10))  # K

        thermal_spec_count = sum([
            T_hot_out is not None,
            T_cold_out is not None,
            T_tube_out is not None,
            T_shell_out is not None,
            Q_spec is not None,
            hot_vap_frac is not None,
            cold_vap_frac is not None,
            tube_vap_frac is not None,
            shell_vap_frac is not None,
        ])
        design_mode = thermal_spec_count == 1
        rating_mode = thermal_spec_count == 0
        solid_bearing = any(
            stream.solid_component_flows
            for stream in (hot, cold)
        )
        if solid_bearing and (rating_mode or estimate_U):
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' supports permanent solids only "
                "with an explicit duty or outlet-state specification and without "
                "auto-U estimation; slurry/powder film coefficients are not modeled"
            )
        if thermal_spec_count > 1:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' requires exactly one of U+A, "
                "T_hot_out, T_cold_out, T_tube_out, T_shell_out, "
                "Q/duty/heat_duty, hot/cold vapor_fraction, or "
                "tube/shell vapor_fraction"
            )
        if not design_mode and not rating_mode:
            raise UnitOperationError(f"HeatExchanger '{self.unit_id}' has invalid thermal specification")
        if (T_tube_out is not None or tube_vap_frac is not None) and not self._has_physical_port('tube', hot_port, cold_port):
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' specifies a tube outlet but has no tube inlet port"
            )
        if (T_shell_out is not None or shell_vap_frac is not None) and not self._has_physical_port('shell', hot_port, cold_port):
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' specifies a shell outlet but has no shell inlet port"
            )
        if tube_force_phase is not None and not self._has_physical_port(
            'tube', hot_port, cold_port
        ):
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' specifies tube_phase but has no "
                "tube inlet port"
            )
        if shell_force_phase is not None and not self._has_physical_port(
            'shell', hot_port, cold_port
        ):
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' specifies shell_phase but has no "
                "shell inlet port"
            )

        T_hot_out = self._physical_spec_value(
            T_hot_out, T_tube_out, T_shell_out, hot_port
        )
        T_cold_out = self._physical_spec_value(
            T_cold_out, T_tube_out, T_shell_out, cold_port
        )
        hot_vap_frac = self._physical_spec_value(
            hot_vap_frac, tube_vap_frac, shell_vap_frac, hot_port
        )
        cold_vap_frac = self._physical_spec_value(
            cold_vap_frac, tube_vap_frac, shell_vap_frac, cold_port
        )
        hot_force_phase = self._physical_spec_value(
            hot_force_phase, tube_force_phase, shell_force_phase, hot_port
        )
        cold_force_phase = self._physical_spec_value(
            cold_force_phase, tube_force_phase, shell_force_phase, cold_port
        )
        if hot_force_phase is not None and hot_vap_frac is not None:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' cannot combine a forced hot-side "
                "phase with a hot-side vapor-fraction outlet specification"
            )
        if cold_force_phase is not None and cold_vap_frac is not None:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' cannot combine a forced cold-side "
                "phase with a cold-side vapor-fraction outlet specification"
            )
        self._validate_forced_phase_state(hot, hot_force_phase, hot_port)
        self._validate_forced_phase_state(cold, cold_force_phase, cold_port)

        P_drop_hot = self._pressure_drop_for_port(hot_port, 'hot')
        P_drop_cold = self._pressure_drop_for_port(cold_port, 'cold')
        P_hot_out = self._outlet_pressure(hot, P_drop_hot, hot_port)
        P_cold_out = self._outlet_pressure(cold, P_drop_cold, cold_port)

        warnings = []

        if design_mode and T_hot_out is not None:
            hot_out = self._state_at_temperature(
                hot.composition, P_hot_out, hot.F, T_hot_out, hot_force_phase
            )
            Q = hot.F * (self._stream_enthalpy(hot) - self._stream_enthalpy(hot_out))
            self._enforce_heat_transfer_direction(Q)
            cold_out = self._outlet_from_heat_added(
                cold, P_cold_out, Q, cold_force_phase
            )

        elif design_mode and T_cold_out is not None:
            cold_out = self._state_at_temperature(
                cold.composition, P_cold_out, cold.F, T_cold_out,
                cold_force_phase,
            )
            Q = cold.F * (self._stream_enthalpy(cold_out) - self._stream_enthalpy(cold))
            self._enforce_heat_transfer_direction(Q)
            hot_out = self._outlet_from_heat_removed(
                hot, P_hot_out, Q, hot_force_phase
            )

        elif design_mode and Q_spec is not None:
            Q = Q_spec
            self._enforce_heat_transfer_direction(Q)
            hot_out = self._outlet_from_heat_removed(
                hot, P_hot_out, Q, hot_force_phase
            )
            cold_out = self._outlet_from_heat_added(
                cold, P_cold_out, Q, cold_force_phase
            )

        elif design_mode and hot_vap_frac is not None:
            hot_out = self._state_for_vapor_fraction(
                hot.composition, P_hot_out, hot.F, hot_vap_frac
            )
            Q = hot.F * (self._stream_enthalpy(hot) - self._stream_enthalpy(hot_out))
            self._enforce_heat_transfer_direction(Q)
            cold_out = self._outlet_from_heat_added(
                cold, P_cold_out, Q, cold_force_phase
            )

        elif design_mode and cold_vap_frac is not None:
            cold_out = self._state_for_vapor_fraction(
                cold.composition, P_cold_out, cold.F, cold_vap_frac
            )
            Q = cold.F * (self._stream_enthalpy(cold_out) - self._stream_enthalpy(cold))
            self._enforce_heat_transfer_direction(Q)
            hot_out = self._outlet_from_heat_removed(
                hot, P_hot_out, Q, hot_force_phase
            )

        else:
            Q = self._rate_exchanger_by_curves(
                hot,
                cold,
                P_hot_out,
                P_cold_out,
                U_value,
                A_value,
                UA_available,
                estimate_U,
                curve_segments,
                flow_pattern,
                hot_force_phase,
                cold_force_phase,
            )
            hot_out = self._outlet_from_heat_removed(
                hot, P_hot_out, Q, hot_force_phase
            )
            cold_out = self._outlet_from_heat_added(
                cold, P_cold_out, Q, cold_force_phase
            )

        compute_full_curve = self._compute_full_curve_metrics(rating_mode)
        curve = None
        if compute_full_curve:
            try:
                curve = self._curve_metrics(
                    hot,
                    cold,
                    P_hot_out,
                    P_cold_out,
                    Q,
                    curve_segments,
                    flow_pattern,
                    include_u_estimates=estimate_U,
                    hot_force_phase=hot_force_phase,
                    cold_force_phase=cold_force_phase,
                )
            except Exception as exc:
                if self._curve_metrics_failure_is_deferrable(rating_mode):
                    warnings.append(
                        f"HeatExchanger '{self.unit_id}' deferred curve diagnostics "
                        f"during recycle iteration after: {exc}"
                    )
                else:
                    raise
        if curve is None:
            curve = self._endpoint_curve_metrics(hot, cold, hot_out, cold_out, flow_pattern)
        if curve['temperature_cross']:
            message = (
                f"Temperature cross detected in HeatExchanger '{self.unit_id}' "
                f"(minimum approach {curve['min_approach_K']:.4g} K)"
            )
            if rating_mode and not allow_temperature_cross:
                raise UnitOperationError(message)
            if not allow_temperature_cross:
                warnings.append(message)

        approach = curve['min_approach_K']
        if approach < min_approach:
            warnings.append(
                f"Minimum curve temperature approach {approach:.1f} K is below minimum {min_approach} K"
            )
        endpoint_approach = min(curve['endpoint_delta_T'])
        if endpoint_approach >= min_approach and approach < min_approach:
            warnings.append(
                "Internal heat-exchanger pinch is tighter than endpoint temperature checks"
            )
        if estimate_U:
            warnings.append(
                "HeatExchanger auto-U estimates are preliminary service-class values; "
                "replace with user-supplied U for design-quality work"
            )
        
        hot_out_port = self._outlet_port_name(hot_port, 'hot')
        cold_out_port = self._outlet_port_name(cold_port, 'cold')
        sizing = self._sizing_performance(
            U_value, A_value, UA_available, estimate_U, curve
        )
        profile = self._profile_performance(curve)
        
        return UnitResult(
            outlet_streams={
                hot_out_port: hot_out,
                cold_out_port: cold_out,
            },
            heat_duty=0.0,
            performance={
                'duty_kW': Q / 3600,
                'heat_transferred_kW': Q / 3600,
                'hot_side_duty_kW': -Q / 3600,
                'cold_side_duty_kW': Q / 3600,
                'LMTD': curve['equivalent_delta_T_K'],
                'equivalent_delta_T_K': curve['equivalent_delta_T_K'],
                'UA_required_W_per_K': curve['UA_required_W_per_K'],
                'UA_required_kW_per_K': (
                    curve['UA_required_W_per_K'] / 1000.0
                    if curve['UA_required_W_per_K'] is not None
                    else None
                ),
                'min_approach_K': approach,
                'approach_T': approach,
                'temperature_cross': curve['temperature_cross'],
                'curve_metrics_delayed': curve.get('metrics_delayed', False),
                'flow_pattern': flow_pattern,
                'curve_segments': curve_segments,
                'T_hot_in': hot.T - 273.15,
                'T_hot_out': hot_out.T - 273.15,
                'T_cold_in': cold.T - 273.15,
                'T_cold_out': cold_out.T - 273.15,
                'P_hot_in_bar': hot.P,
                'P_hot_out_bar': hot_out.P,
                'P_cold_in_bar': cold.P,
                'P_cold_out_bar': cold_out.P,
                'P_drop_hot_bar': P_drop_hot,
                'P_drop_cold_bar': P_drop_cold,
                'hot_in_port': hot_port,
                'cold_in_port': cold_port,
                'hot_out_port': hot_out_port,
                'cold_out_port': cold_out_port,
                'hot_phase_mode': hot_force_phase or 'auto',
                'cold_phase_mode': cold_force_phase or 'auto',
                **profile,
                **sizing,
            },
            warnings=warnings
        )

    def _identify_hot_cold(self, inlet_list: list[tuple[str, StreamState]]
                           ) -> tuple[str, StreamState, str, StreamState]:
        hot_matches = []
        cold_matches = []
        for port, stream in inlet_list:
            port_lower = port.lower()
            if 'hot' in port_lower:
                hot_matches.append((port, stream))
            if 'cold' in port_lower:
                cold_matches.append((port, stream))

        if len(hot_matches) > 1 or len(cold_matches) > 1:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' has ambiguous hot/cold inlet ports"
            )

        if hot_matches and cold_matches:
            hot_port, hot = hot_matches[0]
            cold_port, cold = cold_matches[0]
            if hot_port == cold_port:
                raise UnitOperationError(
                    f"HeatExchanger '{self.unit_id}' inlet port cannot be both hot and cold"
                )
            return hot_port, hot, cold_port, cold

        if hot_matches:
            hot_port, hot = hot_matches[0]
            cold_port, cold = next(
                (port, stream) for port, stream in inlet_list if port != hot_port
            )
            return hot_port, hot, cold_port, cold

        if cold_matches:
            cold_port, cold = cold_matches[0]
            hot_port, hot = next(
                (port, stream) for port, stream in inlet_list if port != cold_port
            )
            return hot_port, hot, cold_port, cold

        (port1, stream1), (port2, stream2) = inlet_list
        if stream1.T >= stream2.T:
            return port1, stream1, port2, stream2
        return port2, stream2, port1, stream1

    def _temperature_spec(self, *names: str):
        for name in names:
            value = self.get_temperature_param(name)
            if value is not None:
                return value
        return None

    def _physical_spec_value(self, role_value, tube_value, shell_value, port: str):
        if role_value is not None:
            return role_value
        port_lower = port.lower()
        if 'tube' in port_lower:
            return tube_value
        if 'shell' in port_lower:
            return shell_value
        return None

    def _forced_phase_spec(self, *names: str) -> str | None:
        value = None
        selected_name = None
        for name in names:
            candidate = self.get_param(name)
            if candidate is not None:
                value = candidate
                selected_name = name
                break
        if value is None:
            return None
        normalized = str(value).strip().lower().replace('-', '_')
        aliases = {
            '': None,
            'auto': None,
            'equilibrium': None,
            'flash': None,
            'none': None,
            'gas': 'vapor',
            'vapour': 'vapor',
            'vapor': 'vapor',
            'liquid': 'liquid',
        }
        if normalized not in aliases:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' {selected_name} must be auto, "
                "vapor, or liquid"
            )
        return aliases[normalized]

    def _validate_forced_phase_state(
        self,
        state: StreamState,
        force_phase: str | None,
        side: str,
    ) -> None:
        fluid_vapor_fraction = state.fluid_vapor_fraction
        if fluid_vapor_fraction is None:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' {side} inlet has no fluid phase "
                "for a forced fluid-phase specification"
            )
        if force_phase == 'vapor' and fluid_vapor_fraction < 1.0 - 1.0e-8:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' {side} inlet is incompatible "
                "with forced vapor phase"
            )
        if force_phase == 'liquid' and fluid_vapor_fraction > 1.0e-8:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' {side} inlet is incompatible "
                "with forced liquid phase"
            )

    def _state_at_temperature(
        self,
        composition: dict,
        P: float,
        F: float,
        T: float,
        force_phase: str | None,
    ) -> StreamState:
        if force_phase is not None:
            return self.thermo.calculate_state(
                T,
                P,
                F,
                composition,
                phase=force_phase,
                flash=False,
            )
        return self.thermo.calculate_state(T, P, F, composition)

    def _has_physical_port(self, side: str, *ports: str) -> bool:
        return any(side in port.lower() for port in ports)

    def _u_spec(self, value) -> tuple[float | None, bool]:
        if value is None:
            return None, False
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ('auto', 'estimated', 'estimate'):
                return None, True
        try:
            U = float(value)
        except (TypeError, ValueError) as exc:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' U must be numeric or auto"
            ) from exc
        if U <= 0.0:
            raise UnitOperationError(f"HeatExchanger '{self.unit_id}' U must be positive")
        unit = (self.get_param_unit('U') or '').strip().lower()
        if unit in ('btu/hr-ft2-f', 'btu/(hr-ft2-f)', 'btu/h-ft2-f',
                    'btu/hr/ft2/f', 'btu/(h ft2 f)'):
            U *= 5.6783
        return U, False

    def _area_spec(self, value) -> float | None:
        if value is None:
            return None
        A = float(value)
        if A <= 0.0:
            raise UnitOperationError(f"HeatExchanger '{self.unit_id}' area must be positive")
        return A

    def _ua_spec(self) -> float | None:
        for name in ('UA', 'UA_available'):
            value = self.get_param(name)
            if value is None:
                continue
            UA = float(value)
            if UA <= 0.0:
                raise UnitOperationError(f"HeatExchanger '{self.unit_id}' {name} must be positive")
            unit = (self.get_param_unit(name) or '').strip().lower()
            if unit in ('kw/k', 'kw per k', 'kw/kdeg', 'kw/degc', 'kw/c'):
                return UA * 1000.0
            if unit in ('w/k', 'w per k', 'w/degc', 'w/c', ''):
                return UA
            return UA * 1000.0 if abs(UA) < 1e4 else UA
        return None

    def _flow_pattern(self) -> str:
        pattern = self.get_param('flow_pattern', self.get_param('type', 'countercurrent'))
        pattern = str(pattern).strip().lower().replace('-', '_')
        aliases = {
            'counter': 'countercurrent',
            'counter_current': 'countercurrent',
            'counterflow': 'countercurrent',
            'counter_flow': 'countercurrent',
            'co_current': 'cocurrent',
            'co-current': 'cocurrent',
            'parallel': 'cocurrent',
            'parallel_flow': 'cocurrent',
        }
        pattern = aliases.get(pattern, pattern)
        if pattern in ('shell_tube', 'shell_and_tube', 'plate', 'double_pipe'):
            return 'countercurrent'
        if pattern not in ('countercurrent', 'cocurrent'):
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' flow_pattern must be countercurrent or cocurrent"
            )
        return pattern

    def _curve_segments(self) -> int:
        value = self.get_param('curve_segments', self.get_param('n_segments', 40))
        segments = int(value)
        if not 4 <= segments <= 400:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' curve_segments must be between 4 and 400"
            )
        return segments

    def _heat_duty_spec(self):
        for name in ('Q', 'duty', 'heat_duty'):
            value = self.get_param(name)
            if value is None:
                continue
            duty = float(value)
            unit = (self.get_param_unit(name) or '').strip().lower()
            if unit in ('w', 'watt', 'watts'):
                return duty * 3.6
            if unit in ('mw', 'megawatt', 'megawatts'):
                return duty * 3.6e6
            if unit in ('kw', 'kilowatt', 'kilowatts'):
                return duty * 3600.0
            if unit in ('kj/h', 'kj/hr', 'kj per h', 'kj per hour'):
                return duty
            if unit in ('j/h', 'j/hr', 'j per h', 'j per hour'):
                return duty / 1000.0
            return duty * 3600.0 if abs(duty) < 1e6 else duty
        return None

    def _rate_exchanger_by_curves(
        self,
        hot: StreamState,
        cold: StreamState,
        P_hot_out: float,
        P_cold_out: float,
        U_value: float | None,
        A_value: float | None,
        UA_available: float | None,
        estimate_U: bool,
        segments: int,
        flow_pattern: str,
        hot_force_phase: str | None = None,
        cold_force_phase: str | None = None,
    ) -> float:
        if hot.T <= cold.T:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' rating mode requires hot inlet temperature above cold inlet"
            )

        constant_UA = UA_available
        if constant_UA is None and U_value is not None and A_value is not None:
            constant_UA = U_value * A_value

        if constant_UA is not None:
            target = constant_UA
            mode = 'UA'
        elif estimate_U and A_value is not None:
            target = A_value
            mode = 'area'
        else:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' rating mode requires UA, U and A, "
                "or A with explicit auto-U estimation"
            )

        def objective(Q: float) -> float:
            if Q <= 1e-9:
                return -target
            curve = self._curve_metrics(
                hot, cold, P_hot_out, P_cold_out, Q, segments, flow_pattern,
                include_u_estimates=(mode == 'area'),
                hot_force_phase=hot_force_phase,
                cold_force_phase=cold_force_phase,
            )
            if curve['temperature_cross']:
                return float('inf')
            if mode == 'area':
                return curve['auto_U_area_required_m2'] - target
            return curve['UA_required_W_per_K'] - target

        delta_t = max(1e-6, hot.T - cold.T)
        capacity = constant_UA if constant_UA is not None else A_value * self.U_ESTIMATE_TABLE['liquid_liquid']['typical']
        Q_high = max(1.0, capacity * delta_t * 3.6)
        f_high = None
        for _ in range(80):
            try:
                f_high = objective(Q_high)
            except Exception:
                f_high = float('inf')
            if f_high > 0.0:
                break
            Q_high *= 2.0
        else:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' could not bracket rating duty"
            )

        if math.isinf(f_high):
            # Back off to the largest non-crossing interval, then use the cross
            # point as the upper bracket.
            pass

        return brentq(objective, 0.0, Q_high, xtol=1e-5, rtol=1e-9, maxiter=100)

    def _compute_full_curve_metrics(self, rating_mode: bool) -> bool:
        if rating_mode:
            return True
        context = getattr(self, 'solve_context', {}) or {}
        return context.get('expensive_diagnostics', True) is not False

    def _curve_metrics_failure_is_deferrable(self, rating_mode: bool) -> bool:
        if rating_mode:
            return False
        context = getattr(self, 'solve_context', {}) or {}
        return (
            context.get('recycle_evaluation') is not None
            and context.get('recycle_final_pass') is not True
        )

    def _endpoint_curve_metrics(
        self,
        hot_in: StreamState,
        cold_in: StreamState,
        hot_out: StreamState,
        cold_out: StreamState,
        flow_pattern: str,
    ) -> dict:
        if flow_pattern == 'countercurrent':
            endpoint_delta_t = (hot_in.T - cold_out.T, hot_out.T - cold_in.T)
        else:
            endpoint_delta_t = (hot_in.T - cold_in.T, hot_out.T - cold_out.T)
        min_approach = min(endpoint_delta_t)
        return {
            'UA_required_W_per_K': None,
            'equivalent_delta_T_K': None,
            'min_approach_K': min_approach,
            'endpoint_delta_T': endpoint_delta_t,
            'temperature_cross': any(delta <= 0.0 for delta in endpoint_delta_t),
            'node_delta_T_K': list(endpoint_delta_t),
            'metrics_delayed': True,
        }

    def _profile_performance(self, curve: dict) -> dict:
        keys = (
            'profile_heat_fraction',
            'profile_hot_T_C',
            'profile_cold_T_C',
            'profile_delta_T_K',
        )
        return {
            key: curve[key]
            for key in keys
            if key in curve
        }

    def _curve_metrics(
        self,
        hot: StreamState,
        cold: StreamState,
        P_hot_out: float,
        P_cold_out: float,
        Q: float,
        segments: int,
        flow_pattern: str,
        include_u_estimates: bool = False,
        hot_force_phase: str | None = None,
        cold_force_phase: str | None = None,
    ) -> dict:
        hot_states, cold_states = self._curve_states(
            hot, cold, P_hot_out, P_cold_out, Q, segments,
            hot_force_phase=hot_force_phase,
            cold_force_phase=cold_force_phase,
        )
        node_delta_t = []
        hot_profile_T_C = []
        cold_profile_T_C = []
        for i, hot_state in enumerate(hot_states):
            cold_state = cold_states[segments - i] if flow_pattern == 'countercurrent' else cold_states[i]
            hot_profile_T_C.append(hot_state.T - 273.15)
            cold_profile_T_C.append(cold_state.T - 273.15)
            node_delta_t.append(hot_state.T - cold_state.T)

        dQ_W = (Q / segments) / 3.6
        UA_required = 0.0
        auto_area = 0.0
        auto_segments = []
        service_counts = {}
        temperature_cross = any(delta <= 0.0 for delta in node_delta_t)

        for i in range(segments):
            dT1 = node_delta_t[i]
            dT2 = node_delta_t[i + 1]
            lmtd = self._lmtd(dT1, dT2)
            if lmtd is None:
                UA_segment = float('inf')
            else:
                UA_segment = dQ_W / lmtd
            UA_required += UA_segment

            if include_u_estimates:
                service_info = self._segment_service_info(
                    hot_states[i], hot_states[i + 1],
                    cold_states[i], cold_states[i + 1],
                )
                service = service_info['service']
                U_record = self._u_record_for_service(service_info)
                U_est = U_record['typical']
                area = UA_segment / U_est if math.isfinite(UA_segment) else float('inf')
                auto_area += area
                service_counts[service] = service_counts.get(service, 0) + 1
                auto_segments.append({
                    'index': i,
                    'service': service,
                    'fallback_service': service_info.get('fallback_service'),
                    'classification_confidence': service_info.get('confidence'),
                    'classification_basis': service_info.get('basis'),
                    'source': U_record.get('source'),
                    'U_low_W_m2_K': U_record.get('low'),
                    'U_W_m2_K': U_est,
                    'U_high_W_m2_K': U_record.get('high'),
                    'dirt_factor_hr_ft2_F_per_Btu': U_record.get('dirt_factor_hr_ft2_F_per_Btu'),
                    'dirt_factor_m2_K_per_W': U_record.get('dirt_factor_m2_K_per_W'),
                    'area_m2': area,
                    'UA_required_W_per_K': UA_segment,
                })

        equivalent_delta_T = (Q / 3.6) / UA_required if UA_required > 0 and math.isfinite(UA_required) else 0.0
        result = {
            'UA_required_W_per_K': UA_required,
            'equivalent_delta_T_K': equivalent_delta_T,
            'min_approach_K': min(node_delta_t),
            'endpoint_delta_T': (node_delta_t[0], node_delta_t[-1]),
            'temperature_cross': temperature_cross,
            'node_delta_T_K': node_delta_t,
            'profile_heat_fraction': [i / segments for i in range(segments + 1)],
            'profile_hot_T_C': hot_profile_T_C,
            'profile_cold_T_C': cold_profile_T_C,
            'profile_delta_T_K': node_delta_t,
        }
        if include_u_estimates:
            result.update({
                'auto_U_area_required_m2': auto_area,
                'auto_U_effective_W_m2_K': UA_required / auto_area if auto_area > 0 and math.isfinite(auto_area) else None,
                'auto_U_service_counts': service_counts,
                'auto_U_segments': auto_segments,
            })
        return result

    def _curve_states(
        self,
        hot: StreamState,
        cold: StreamState,
        P_hot_out: float,
        P_cold_out: float,
        Q: float,
        segments: int,
        hot_force_phase: str | None = None,
        cold_force_phase: str | None = None,
    ) -> tuple[list[StreamState], list[StreamState]]:
        H_hot_in = self._stream_enthalpy(hot)
        H_cold_in = self._stream_enthalpy(cold)
        hot_states = []
        cold_states = []
        T_hot_guess = hot.T
        T_cold_guess = cold.T
        for i in range(segments + 1):
            fraction = i / segments
            P_hot = hot.P + fraction * (P_hot_out - hot.P)
            P_cold = cold.P + fraction * (P_cold_out - cold.P)
            H_hot = H_hot_in - (Q * fraction) / hot.F
            H_cold = H_cold_in + (Q * fraction) / cold.F
            if i == 0:
                hot_state = hot.copy()
                cold_state = cold.copy()
            else:
                hot_state = self._state_for_enthalpy(
                    hot.composition, P_hot, hot.F, H_hot, T_hot_guess,
                    include=('H',), force_phase=hot_force_phase,
                )
                cold_state = self._state_for_enthalpy(
                    cold.composition, P_cold, cold.F, H_cold, T_cold_guess,
                    include=('H',), force_phase=cold_force_phase,
                )
            T_hot_guess = hot_state.T
            T_cold_guess = cold_state.T
            hot_states.append(hot_state)
            cold_states.append(cold_state)
        return hot_states, cold_states

    def _lmtd(self, dT1: float, dT2: float) -> float | None:
        if dT1 <= 0.0 or dT2 <= 0.0:
            return None
        if abs(dT1 - dT2) <= 1e-9:
            return 0.5 * (dT1 + dT2)
        return (dT1 - dT2) / math.log(dT1 / dT2)

    def _sizing_performance(
        self,
        U_value: float | None,
        A_value: float | None,
        UA_available: float | None,
        estimate_U: bool,
        curve: dict,
    ) -> dict:
        UA_required = curve['UA_required_W_per_K']
        performance = {}
        if UA_available is not None:
            performance['UA_available_W_per_K'] = UA_available
        if U_value is not None:
            performance['U_W_m2_K'] = U_value
            if UA_required is not None:
                performance['area_required_m2'] = UA_required / U_value
        if A_value is not None:
            performance['area_m2'] = A_value
            if UA_required is not None:
                performance['U_required_W_m2_K'] = UA_required / A_value
        if U_value is not None and A_value is not None and UA_available is None:
            UA_available = U_value * A_value
            performance['UA_available_W_per_K'] = UA_available
        if UA_available is not None and UA_required is not None:
            performance['UA_margin_W_per_K'] = UA_available - UA_required
            performance['UA_utilization'] = UA_required / UA_available

        if estimate_U:
            auto_area = curve.get('auto_U_area_required_m2')
            performance.update({
                'auto_U_estimation': True,
                'auto_U_area_required_m2': auto_area,
                'auto_U_effective_W_m2_K': curve.get('auto_U_effective_W_m2_K'),
                'auto_U_service_counts': curve.get('auto_U_service_counts', {}),
                'auto_U_segments': curve.get('auto_U_segments', []),
            })
            if auto_area and A_value is not None:
                performance['auto_U_area_margin_m2'] = A_value - auto_area
                performance['auto_U_area_utilization'] = auto_area / A_value
        else:
            performance['auto_U_estimation'] = False
        return performance

    def _segment_service(
        self,
        hot_start: StreamState,
        hot_end: StreamState,
        cold_start: StreamState,
        cold_end: StreamState,
    ) -> str:
        return self._segment_service_info(hot_start, hot_end, cold_start, cold_end)['service']

    def _segment_service_info(
        self,
        hot_start: StreamState,
        hot_end: StreamState,
        cold_start: StreamState,
        cold_end: StreamState,
    ) -> dict:
        hot_dvf = hot_end.vapor_fraction - hot_start.vapor_fraction
        cold_dvf = cold_end.vapor_fraction - cold_start.vapor_fraction

        hot_family = self._stream_family(hot_start, hot_end)
        cold_family = self._stream_family(cold_start, cold_end)

        if cold_dvf > 1e-4 and hot_dvf < -1e-4 and hot_family['family'] == 'water':
            return self._vaporizer_service_info(cold_family, hot_family)
        if hot_dvf < -1e-4:
            return self._condensing_service_info(hot_family, cold_family)
        if cold_dvf > 1e-4:
            return self._vaporizer_service_info(cold_family, hot_family)

        hot_phase = self._phase_label(hot_start, hot_end)
        cold_phase = self._phase_label(cold_start, cold_end)
        if hot_phase == 'liquid' and cold_phase == 'liquid':
            return self._liquid_liquid_service_info(hot_family, cold_family)
        if hot_phase == 'vapor' and cold_phase == 'vapor':
            return self._service_info('gas_gas', 'gas_gas', 'gas-gas; no Seader Table 12.5 gas-gas row')
        return self._gas_liquid_service_info(hot_family, cold_family, hot_start, cold_start)

    def _u_record_for_service(self, service_info: dict) -> dict:
        service = service_info.get('service')
        record = self.SEADER_U_ESTIMATE_TABLE.get(service)
        if record is not None:
            return record
        fallback = service_info.get('fallback_service') or service
        record = self.U_ESTIMATE_TABLE.get(fallback)
        if record is not None:
            return {
                **record,
                'source': 'generic fallback',
            }
        return {
            **self.U_ESTIMATE_TABLE['liquid_liquid'],
            'source': 'generic fallback',
        }

    def _service_info(
        self,
        service: str,
        fallback: str,
        basis: str,
        confidence: str = 'medium',
    ) -> dict:
        if service not in self.SEADER_U_ESTIMATE_TABLE and fallback not in self.U_ESTIMATE_TABLE:
            fallback = 'liquid_liquid'
        return {
            'service': service,
            'fallback_service': fallback,
            'basis': basis,
            'confidence': confidence,
        }

    def _liquid_liquid_service_info(self, family_a: dict, family_b: dict) -> dict:
        families = {family_a['family'], family_b['family']}
        if families == {'water'}:
            if min(family_a.get('water_mass_fraction', 0.0), family_b.get('water_mass_fraction', 0.0)) > 0.995:
                return self._service_info('demin_water_water', 'liquid_liquid', 'liquid-liquid; nearly pure water on both sides', 'high')
            return self._service_info('water_water', 'liquid_liquid', 'liquid-liquid; water-rich on both sides', 'high')

        water_side, other_side = self._split_family_pair(family_a, family_b, 'water')
        if water_side is not None:
            other = other_side['family']
            if other == 'caustic':
                return self._service_info('water_caustic', 'liquid_liquid', 'liquid-liquid; water with caustic solution', other_side['confidence'])
            if other == 'amine':
                return self._service_info('amine_water', 'liquid_liquid', 'liquid-liquid; amine solution with water', other_side['confidence'])
            if other == 'gasoline':
                return self._service_info('gasoline_water', 'liquid_liquid', other_side['basis'], other_side['confidence'])
            if other == 'naphtha':
                return self._service_info('naphtha_water', 'liquid_liquid', other_side['basis'], other_side['confidence'])
            if other in ('kerosene', 'gas_oil'):
                return self._service_info('kerosene_gas_oil_water', 'liquid_liquid', other_side['basis'], other_side['confidence'])
            if other == 'heavy_oil':
                return self._service_info('heavy_oil_water', 'liquid_liquid', other_side['basis'], other_side['confidence'])
            if other == 'fuel_oil':
                return self._service_info('fuel_oil_water', 'liquid_liquid', other_side['basis'], other_side['confidence'])
            if other == 'lube_oil_high':
                return self._service_info('lube_oil_high_water', 'liquid_liquid', other_side['basis'], other_side['confidence'])
            if other in ('lube_oil_low', 'oil'):
                return self._service_info('lube_oil_low_water', 'liquid_liquid', other_side['basis'], other_side['confidence'])
            if other == 'wax':
                return self._service_info('wax_distillate_water', 'liquid_liquid', other_side['basis'], other_side['confidence'])
            if other == 'vegetable_oil':
                return self._service_info('vegetable_oil_water', 'liquid_liquid', other_side['basis'], other_side['confidence'])
            if other in ('organic_solvent', 'alcohol', 'chlorinated_solvent'):
                return self._service_info('organic_solvent_water', 'liquid_liquid', other_side['basis'], other_side['confidence'])

        brine_side, other_side = self._split_family_pair(family_a, family_b, 'brine')
        if brine_side is not None and other_side['family'] in ('organic_solvent', 'alcohol', 'chlorinated_solvent'):
            return self._service_info('organic_solvent_brine', 'liquid_liquid', 'liquid-liquid; organic solvent with brine', other_side['confidence'])

        if family_a['family'] in ('organic_solvent', 'alcohol', 'chlorinated_solvent') and family_b['family'] in ('organic_solvent', 'alcohol', 'chlorinated_solvent'):
            return self._service_info('organic_solvent_organic', 'liquid_liquid', 'liquid-liquid; organic solvents on both sides', 'medium')
        if family_a['family'] == 'heavy_oil' and family_b['family'] == 'heavy_oil':
            return self._service_info('heavy_oil_heavy_oil', 'liquid_liquid', 'liquid-liquid; heavy oils on both sides', 'medium')
        if 'oil' in (family_a['family'], family_b['family']) or family_a['family'].endswith('oil') or family_b['family'].endswith('oil'):
            return self._service_info('fuel_oil_oil', 'liquid_liquid', 'liquid-liquid; oil-like liquid pair', 'low')
        return self._service_info('liquid_liquid', 'liquid_liquid', 'liquid-liquid; broad fallback', 'low')

    @staticmethod
    def _split_family_pair(family_a: dict, family_b: dict, target: str) -> tuple[dict | None, dict | None]:
        if family_a['family'] == target:
            return family_a, family_b
        if family_b['family'] == target:
            return family_b, family_a
        return None, None

    def _gas_liquid_service_info(
        self,
        hot_family: dict,
        cold_family: dict,
        hot_state: StreamState,
        cold_state: StreamState,
    ) -> dict:
        gas_family = hot_family if hot_family.get('phase_hint') == 'vapor' else cold_family
        liquid_family = cold_family if gas_family is hot_family else hot_family
        gas_state = hot_state if gas_family is hot_family else cold_state
        liquid_is_water = liquid_family['family'] in ('water', 'brine')
        if gas_family['family'] == 'air' and liquid_is_water:
            compressed = gas_state.P > 3.0
            service = 'compressed_air_water' if compressed else 'atmospheric_air_water'
            return self._service_info(service, 'gas_liquid', f"gas-liquid; {'compressed' if compressed else 'atmospheric'} air/N2 with water/brine", 'high')
        if gas_family['family'] in ('hydrogen', 'natural_gas', 'light_hydrocarbon') and liquid_family['family'] == 'water':
            return self._service_info('water_hydrogen_natural_gas', 'gas_liquid', 'gas-liquid; hydrogen/natural-gas-like mixture with water', gas_family['confidence'])
        return self._service_info('gas_liquid', 'gas_liquid', 'gas-liquid; broad fallback', 'low')

    def _condensing_service_info(self, vapor_family: dict, liquid_family: dict) -> dict:
        if vapor_family['family'] == 'water':
            if liquid_family['family'] == 'water':
                return self._service_info('steam_feedwater', 'steam_condensing', 'condensing steam heating water/feedwater', 'high')
            if liquid_family['family'] in ('fuel_oil', 'heavy_oil'):
                service = 'steam_no6_fuel_oil' if liquid_family['family'] == 'heavy_oil' else 'steam_no2_fuel_oil'
                return self._service_info(service, 'steam_condensing', f"condensing steam heating {liquid_family['family']}", liquid_family['confidence'])
            return self._service_info('steam_condensing', 'steam_condensing', 'condensing steam; generic process liquid', 'medium')
        if vapor_family['family'] == 'alcohol' and liquid_family['family'] == 'water':
            return self._service_info('alcohol_vapor_water', 'condensing_vapor', 'condensing alcohol vapor with water', vapor_family['confidence'])
        if vapor_family['family'] == 'sulfur_dioxide' and liquid_family['family'] == 'water':
            return self._service_info('sulfur_dioxide_water', 'condensing_vapor', 'condensing sulfur dioxide with water', 'high')
        if (
            vapor_family.get('aromatic_group_mass_fraction', 0.0) >= 0.20
            and vapor_family.get('water_mass_fraction', 0.0) >= 0.10
            and liquid_family['family'] == 'water'
        ):
            return self._service_info(
                'aromatic_vapor_water_azeotrope',
                'condensing_vapor',
                'condensing aromatic/water vapor stream against water',
                self._combined_confidence(vapor_family['confidence'], liquid_family['confidence']),
            )
        if vapor_family['family'] == 'dowtherm':
            if liquid_family['family'] == 'dowtherm':
                return self._service_info(
                    'dowtherm_vapor_dowtherm_liquid',
                    'condensing_vapor',
                    'condensing Dowtherm-like vapor against Dowtherm-like liquid',
                    self._combined_confidence(vapor_family['confidence'], liquid_family['confidence']),
                )
            if liquid_family['family'] == 'vegetable_oil':
                return self._service_info('dowtherm_vapor_tall_oil', 'condensing_vapor', 'condensing Dowtherm-like vapor against tall/vegetable oil', liquid_family['confidence'])
        if vapor_family['family'] in ('light_hydrocarbon', 'gasoline') and liquid_family['family'] == 'water':
            return self._service_info('low_boiling_hydrocarbon_water', 'condensing_vapor', vapor_family['basis'], vapor_family['confidence'])
        if vapor_family['family'] in ('naphtha', 'kerosene', 'gas_oil', 'heavy_oil') and liquid_family['family'] == 'water':
            service = 'high_boiling_hydrocarbon_water' if vapor_family['family'] in ('gas_oil', 'heavy_oil') else f"{vapor_family['family']}_condensing_water"
            if service not in self.SEADER_U_ESTIMATE_TABLE:
                service = 'high_boiling_hydrocarbon_water'
            return self._service_info(service, 'condensing_vapor', vapor_family['basis'], vapor_family['confidence'])
        if vapor_family['family'] in ('organic_solvent', 'alcohol', 'chlorinated_solvent', 'aromatic_solvent') and liquid_family['family'] in ('water', 'brine'):
            service = 'organic_solvent_vapor_water'
            if vapor_family.get('Tb_avg_K') and vapor_family['Tb_avg_K'] > 450.0:
                service = 'organic_solvent_high_nc_water'
            return self._service_info(service, 'condensing_vapor', vapor_family['basis'], vapor_family['confidence'])
        if vapor_family['family'] in ('light_hydrocarbon', 'naphtha', 'kerosene') and liquid_family['family'] in ('oil', 'fuel_oil', 'heavy_oil', 'lube_oil_low', 'lube_oil_high'):
            return self._service_info('hydrocarbon_partial_condenser_oil', 'condensing_vapor', 'condensing hydrocarbon vapor against oil-like liquid', 'medium')
        return self._service_info('condensing_vapor', 'condensing_vapor', 'condensing vapor; broad fallback', 'low')

    @staticmethod
    def _combined_confidence(*values: str) -> str:
        rank = {'low': 0, 'medium': 1, 'high': 2}
        inverse = {0: 'low', 1: 'medium', 2: 'high'}
        return inverse[min(rank.get(value, 0) for value in values)]

    def _vaporizer_service_info(self, vaporizing_family: dict, heating_family: dict) -> dict:
        if heating_family['family'] == 'water':
            family = vaporizing_family['family']
            if family == 'ammonia':
                return self._service_info('ammonia_steam_vaporizer', 'vaporizing', 'vaporizing ammonia with condensing steam', 'high')
            if family == 'chlorine':
                return self._service_info('chlorine_steam_vaporizer', 'vaporizing', 'vaporizing chlorine with condensing steam', 'high')
            if family in ('light_hydrocarbon', 'gasoline') or (vaporizing_family.get('C_avg') or 999.0) <= 4.5:
                return self._service_info('propane_butane_steam_vaporizer', 'vaporizing', vaporizing_family['basis'], vaporizing_family['confidence'])
            if family == 'water':
                return self._service_info('water_steam_vaporizer', 'water_boiling', 'vaporizing water with condensing steam', 'high')
        if vaporizing_family['family'] == 'water':
            return self._service_info('water_boiling', 'water_boiling', 'vaporizing water; broad fallback', 'medium')
        return self._service_info('vaporizing', 'vaporizing', 'vaporizing process fluid; broad fallback', 'low')

    def _stream_family(self, start: StreamState, end: StreamState) -> dict:
        T = 0.5 * (start.T + end.T)
        phase_hint = self._phase_label(start, end)
        composition = self._normalized_composition(start.composition)
        total_mass = 0.0
        scores = {
            'water': 0.0,
            'air': 0.0,
            'hydrogen': 0.0,
            'hydrocarbon': 0.0,
            'organic': 0.0,
            'caustic': 0.0,
            'amine': 0.0,
            'brine': 0.0,
            'ammonia': 0.0,
            'chlorine': 0.0,
            'sulfur_dioxide': 0.0,
            'vegetable_oil': 0.0,
            'dowtherm': 0.0,
        }
        hydrocarbon_moles = 0.0
        hydrocarbon_c_moles = 0.0
        tb_weight = 0.0
        tb_total = 0.0
        liquid_visc_weight = 0.0
        liquid_log_visc = 0.0
        dowtherm_biphenyl_mass = 0.0
        dowtherm_ether_mass = 0.0
        alcohol_group_mass = 0.0
        aromatic_group_mass = 0.0
        labels = []

        for comp, z in composition.items():
            if z <= 0.0:
                continue
            props = getattr(self.thermo, 'props', {}).get(comp)
            mw = float(getattr(props, 'MW', 0.0) or start.MW or 1.0)
            mass = z * max(mw, 1e-12)
            total_mass += mass
            category = self._component_category(comp, props)
            scores[category] = scores.get(category, 0.0) + mass
            markers = self._component_functional_markers(props)
            if 'alcohol' in markers:
                alcohol_group_mass += mass
            if 'aromatic' in markers:
                aromatic_group_mass += mass
            dowtherm_kind = self._dowtherm_component_kind(comp, props)
            if dowtherm_kind == 'biphenyl':
                dowtherm_biphenyl_mass += mass
            elif dowtherm_kind == 'diphenyl_ether':
                dowtherm_ether_mass += mass
            if category == 'hydrocarbon':
                counts = self._formula_counts_for_component(comp, props) or {}
                carbon = counts.get('C', 0)
                hydrocarbon_moles += z
                hydrocarbon_c_moles += z * carbon
            if props and getattr(props, 'Tb', None):
                tb_total += mass * float(props.Tb)
                tb_weight += mass
            if category in ('hydrocarbon', 'organic', 'vegetable_oil'):
                viscosity = self._component_viscosity(comp, props, T, phase_hint)
                if viscosity and viscosity > 0.0:
                    liquid_log_visc += mass * math.log(viscosity)
                    liquid_visc_weight += mass
            if props and getattr(props, 'name', None):
                labels.append(str(props.name).lower())
            labels.append(str(comp).lower())

        if total_mass <= 0.0:
            return {
                'family': 'unknown',
                'confidence': 'low',
                'basis': 'empty or unknown composition',
                'phase_hint': phase_hint,
            }
        mass_scores = {key: value / total_mass for key, value in scores.items()}
        family = max(mass_scores, key=mass_scores.get)
        dominant = mass_scores[family]
        confidence = 'high' if dominant >= 0.80 else 'medium' if dominant >= 0.50 else 'low'
        dowtherm_mass_fraction = (dowtherm_biphenyl_mass + dowtherm_ether_mass) / total_mass
        dowtherm_ether_fraction = (
            dowtherm_ether_mass / (dowtherm_biphenyl_mass + dowtherm_ether_mass)
            if (dowtherm_biphenyl_mass + dowtherm_ether_mass) > 0.0
            else None
        )
        if (
            dowtherm_mass_fraction >= 0.75
            and dowtherm_biphenyl_mass > 0.0
            and dowtherm_ether_mass > 0.0
            and dowtherm_ether_fraction is not None
            and 0.45 <= dowtherm_ether_fraction <= 0.90
        ):
            family = 'dowtherm'
            dominant = dowtherm_mass_fraction
            confidence = 'high' if dominant >= 0.90 else 'medium'
        elif mass_scores.get('water', 0.0) >= 0.50:
            aqueous_solutes = [
                ('caustic', mass_scores.get('caustic', 0.0)),
                ('amine', mass_scores.get('amine', 0.0)),
                ('brine', mass_scores.get('brine', 0.0)),
            ]
            solute_family, solute_score = max(aqueous_solutes, key=lambda item: item[1])
            if solute_score >= 0.02:
                family = solute_family
                dominant = mass_scores['water'] + solute_score
                confidence = 'high' if solute_score >= 0.08 else 'medium'
        C_avg = hydrocarbon_c_moles / hydrocarbon_moles if hydrocarbon_moles > 0.0 else None
        Tb_avg = tb_total / tb_weight if tb_weight > 0.0 else None
        mu_liq = math.exp(liquid_log_visc / liquid_visc_weight) if liquid_visc_weight > 0.0 else None
        alcohol_group_fraction = alcohol_group_mass / total_mass
        aromatic_group_fraction = aromatic_group_mass / total_mass

        if family == 'hydrocarbon':
            family = self._hydrocarbon_family(C_avg, Tb_avg, mu_liq, ' '.join(labels))
            if aromatic_group_fraction >= 0.50:
                family = 'aromatic_hydrocarbon'
        elif family == 'organic':
            if (
                alcohol_group_fraction >= 0.50
                or self._labels_contain(labels, ('alcohol', 'methanol', 'ethanol', 'propanol', 'butanol', 'glycol'))
            ):
                family = 'alcohol'
            elif aromatic_group_fraction >= 0.50:
                family = 'aromatic_solvent'
            elif self._labels_contain(labels, ('trichloroethylene', 'chloroform', 'chloride', 'chlorinated')):
                family = 'chlorinated_solvent'
            else:
                family = 'organic_solvent'
        elif family == 'brine' and mass_scores.get('water', 0.0) > 0.5:
            family = 'brine'
        elif family == 'air':
            family = 'air'
        elif family == 'water':
            family = 'water'

        basis_parts = [f"{family} family from mass-fraction scores"]
        if C_avg is not None:
            basis_parts.append(f"C_avg={C_avg:.2g}")
        if family == 'dowtherm' and dowtherm_ether_fraction is not None:
            basis_parts.append(f"diphenyl-ether fraction~{dowtherm_ether_fraction:.2g}")
        if alcohol_group_fraction > 0.0:
            basis_parts.append(f"UNIFAC alcohol-group mass fraction~{alcohol_group_fraction:.2g}")
        if aromatic_group_fraction > 0.0:
            basis_parts.append(f"UNIFAC aromatic-group mass fraction~{aromatic_group_fraction:.2g}")
        if Tb_avg is not None:
            basis_parts.append(f"Tb_avg={Tb_avg:.0f} K")
        if mu_liq is not None:
            basis_parts.append(f"mu_liq~{mu_liq:.2g} Pa*s")
        return {
            'family': family,
            'confidence': confidence,
            'basis': '; '.join(basis_parts),
            'phase_hint': phase_hint,
            'water_mass_fraction': mass_scores.get('water', 0.0),
            'alcohol_group_mass_fraction': alcohol_group_fraction,
            'aromatic_group_mass_fraction': aromatic_group_fraction,
            'C_avg': C_avg,
            'Tb_avg_K': Tb_avg,
            'liquid_viscosity_Pa_s': mu_liq,
            'mass_scores': mass_scores,
        }

    @staticmethod
    def _normalized_composition(composition: dict) -> dict:
        total = sum(max(float(value), 0.0) for value in composition.values())
        if total <= 0.0:
            return {}
        return {
            comp: max(float(value), 0.0) / total
            for comp, value in composition.items()
        }

    def _component_category(self, comp: str, props) -> str:
        counts = self._formula_counts_for_component(comp, props)
        if counts == {'H': 2, 'O': 1}:
            return 'water'
        if counts in ({'N': 2}, {'O': 2}, {'Ar': 1}):
            return 'air'
        if counts == {'H': 2}:
            return 'hydrogen'
        if counts == {'N': 1, 'H': 3}:
            return 'ammonia'
        if counts == {'Cl': 2}:
            return 'chlorine'
        if counts == {'S': 1, 'O': 2}:
            return 'sulfur_dioxide'
        if counts in ({'Na': 1, 'O': 1, 'H': 1}, {'K': 1, 'O': 1, 'H': 1}):
            return 'caustic'
        if counts in ({'Na': 1, 'Cl': 1}, {'K': 1, 'Cl': 1}, {'Ca': 1, 'Cl': 2}):
            return 'brine'
        if counts and self._formula_counts_look_caustic(counts):
            return 'caustic'
        if counts and self._formula_counts_look_brine_salt(counts):
            return 'brine'
        label = ' '.join(
            str(value).lower()
            for value in (comp, getattr(props, 'symbol', ''), getattr(props, 'name', ''))
            if value
        )
        if self._component_has_amine_marker(label, props):
            return 'amine'
        if self._labels_contain([label], ('vegetable oil', 'tall oil')):
            return 'vegetable_oil'
        if self._labels_contain([label], ('dowtherm', 'diphenyl ether', 'biphenyl')):
            return 'dowtherm'

        if counts:
            carbon = counts.get('C', 0)
            hydrogen = counts.get('H', 0)
            non_ch = {
                element: count
                for element, count in counts.items()
                if count and element not in ('C', 'H')
            }
            if carbon > 0 and hydrogen > 0 and not non_ch:
                return 'hydrocarbon'
            if carbon > 0:
                return 'organic'

        if self._labels_contain([label], ('water', 'steam')):
            return 'water'
        if self._labels_contain([label], ('nitrogen', 'oxygen', 'argon', 'air')):
            return 'air'
        if self._labels_contain([label], ('hydrogen',)):
            return 'hydrogen'
        if self._labels_contain([label], ('ammonia',)):
            return 'ammonia'
        if self._labels_contain([label], ('chlorine',)):
            return 'chlorine'
        if self._labels_contain([label], ('sulfur dioxide', 'sulphur dioxide')):
            return 'sulfur_dioxide'
        if self._labels_contain([label], ('caustic', 'sodium hydroxide', 'potassium hydroxide')):
            return 'caustic'
        if self._labels_contain([label], ('brine', 'sodium chloride', 'potassium chloride', 'salt')):
            return 'brine'
        return 'unknown'

    @staticmethod
    def _formula_counts_look_caustic(counts: dict) -> bool:
        cations = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Mg', 'Ca', 'Sr', 'Ba'}
        return (
            counts.get('C', 0) == 0
            and counts.get('O', 0) >= 1
            and counts.get('H', 0) >= 1
            and any(counts.get(element, 0) > 0 for element in cations)
        )

    @staticmethod
    def _formula_counts_look_brine_salt(counts: dict) -> bool:
        cations = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Mg', 'Ca', 'Sr', 'Ba'}
        halides = {'Cl', 'Br', 'I', 'F'}
        return (
            counts.get('C', 0) == 0
            and any(counts.get(element, 0) > 0 for element in cations)
            and any(counts.get(element, 0) > 0 for element in halides)
        )

    def _component_has_amine_marker(self, label: str, props) -> bool:
        if self._labels_contain([label], ('mea', 'dea', 'mdea', 'ethanolamine', 'diethanolamine', 'amine')):
            return True
        return 'amine' in self._component_functional_markers(props)

    def _component_functional_markers(self, props) -> set[str]:
        groups = self._component_unifac_groups(props)
        markers = set()
        if not isinstance(groups, dict):
            return markers
        group_keys = set(groups)
        group_text = ' '.join(str(key).lower() for key in groups)
        # This helper requests classic UNIFAC below. IDs 28-36 retain these
        # amine meanings in all three neutral parameter families; later IDs
        # are variant-specific and must not be guessed without model context.
        amine_group_numbers = (28, 29, 30, 31, 32, 33, 34, 35, 36)
        if (
            any(marker in group_text for marker in ('amine', 'nh2', 'ch3n'))
            or self._unifac_groups_include(
                group_keys,
                group_text,
                names=(),
                numbers=amine_group_numbers,
            )
        ):
            markers.add('amine')
        if self._unifac_groups_include(group_keys, group_text, names=('oh', 'ch3oh'), numbers=(14, 15, 81, 82)):
            markers.add('alcohol')
        if self._unifac_groups_include(group_keys, group_text, names=('ach', 'ac', 'acch3', 'acch2'), numbers=(9, 10, 11, 12)):
            markers.add('aromatic')
        return markers

    def _component_unifac_groups(self, props):
        groups = getattr(props, 'unifac_groups', None) or getattr(props, 'groups', None)
        if isinstance(groups, dict):
            return groups
        return self._unifac_groups_for_component(props)

    @staticmethod
    def _unifac_groups_include(group_keys: set, group_text: str, *, names: tuple[str, ...], numbers: tuple[int, ...]) -> bool:
        for key in group_keys:
            try:
                if int(key) in numbers:
                    return True
            except (TypeError, ValueError):
                pass
            key_text = str(key).lower()
            if key_text in names:
                return True
        return any(f'[{name}]' in group_text for name in names)

    def _unifac_groups_for_component(self, props):
        if props is None:
            return None
        cache = getattr(self, '_auto_u_unifac_group_cache', None)
        if cache is None:
            cache = {}
            self._auto_u_unifac_group_cache = cache
        cache_key = (
            str(getattr(props, 'name', '') or ''),
            str(getattr(props, 'symbol', '') or ''),
            str(getattr(props, 'formula', '') or ''),
            str(getattr(props, 'smiles', '') or ''),
        )
        if cache_key in cache:
            return cache[cache_key]
        identifiers = [
            getattr(props, 'name', None),
            getattr(props, 'symbol', None),
        ]
        smiles = getattr(props, 'smiles', None)
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .unifac import get_unifac_groups
            else:
                from unifac import get_unifac_groups
        except Exception:
            cache[cache_key] = None
            return None
        for identifier in identifiers:
            if not identifier:
                continue
            try:
                groups = get_unifac_groups(str(identifier), smiles=smiles)
                cache[cache_key] = groups
                return groups
            except Exception:
                continue
        cache[cache_key] = None
        return None

    def _dowtherm_component_kind(self, comp: str, props) -> str | None:
        label = ' '.join(
            str(value).lower()
            for value in (comp, getattr(props, 'symbol', ''), getattr(props, 'name', ''), getattr(props, 'formula', ''))
            if value
        )
        if 'diphenyl ether' in label or 'diphenyl oxide' in label:
            return 'diphenyl_ether'
        if 'biphenyl' in label or 'diphenyl' in label:
            return 'biphenyl'
        counts = self._formula_counts_for_component(comp, props)
        if counts == {'C': 12, 'H': 10, 'O': 1}:
            return 'diphenyl_ether'
        if counts == {'C': 12, 'H': 10}:
            return 'biphenyl'
        return None

    def _formula_counts_for_component(self, comp: str, props) -> dict | None:
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import parse_formula_counts
            else:
                from compound_identity import parse_formula_counts
        except Exception:
            return None
        for value in (getattr(props, 'formula', None), getattr(props, 'symbol', None), comp):
            if not value:
                continue
            counts = parse_formula_counts(str(value))
            if counts:
                return counts
        return None

    def _component_viscosity(self, comp: str, props, T: float, phase_hint: str) -> float | None:
        phase = 'liquid' if phase_hint != 'vapor' else 'vapor'
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            result = get_property_resolver().resolve_viscosity(
                comp,
                T,
                phase,
                props=props._resolver_props() if hasattr(props, '_resolver_props') else None,
            )
            return float(result.value)
        except Exception:
            return None

    @staticmethod
    def _labels_contain(labels, needles: tuple[str, ...]) -> bool:
        text = ' '.join(str(label).lower() for label in labels)
        return any(needle in text for needle in needles)

    def _hydrocarbon_family(
        self,
        C_avg: float | None,
        Tb_avg: float | None,
        mu_liq: float | None,
        label_text: str,
    ) -> str:
        label_text = label_text.lower()
        if 'wax' in label_text:
            return 'wax'
        if 'fuel oil' in label_text:
            return 'fuel_oil'
        if 'lube' in label_text:
            return 'lube_oil_high' if ('high' in label_text or (mu_liq or 0.0) >= 0.03) else 'lube_oil_low'
        if mu_liq is not None:
            if mu_liq >= 0.10:
                return 'heavy_oil'
            if mu_liq >= 0.03:
                return 'lube_oil_high'
            if mu_liq >= 0.01:
                return 'lube_oil_low'
        if C_avg is None:
            return 'natural_gas' if Tb_avg is not None and Tb_avg < 250.0 else 'oil'
        if C_avg <= 4.5:
            return 'light_hydrocarbon'
        if C_avg <= 7.5:
            return 'gasoline'
        if C_avg <= 10.5:
            return 'naphtha'
        if C_avg <= 16.0:
            return 'kerosene'
        if C_avg <= 24.0:
            return 'gas_oil'
        return 'heavy_oil'

    def _phase_label(self, start: StreamState, end: StreamState) -> str:
        vf = 0.5 * (start.vapor_fraction + end.vapor_fraction)
        if vf <= 1e-3:
            return 'liquid'
        if vf >= 1.0 - 1e-3:
            return 'vapor'
        return 'two_phase'

    def _is_pure_water(self, stream: StreamState) -> bool:
        if len(stream.composition) != 1:
            return False
        comp = next(iter(stream.composition)).strip().lower()
        return comp in ('h2o', 'water')

    def _vapor_fraction_spec(self, side: str):
        names = (
            f'{side}_vap_frac',
            f'{side}_vapor_frac',
            f'{side}_vapor_fraction',
            f'vap_frac_{side}',
            f'vapor_frac_{side}',
            f'vapor_fraction_{side}',
            f'vap_frac_{side}_out',
            f'vapor_frac_{side}_out',
            f'vapor_fraction_{side}_out',
            f'{side}_out_vap_frac',
            f'{side}_out_vapor_frac',
            f'{side}_out_vapor_fraction',
            f'VF_{side}',
            f'VF_{side}_out',
            f'{side}_VF',
            f'{side}_out_VF',
        )
        for name in names:
            value = self.get_param(name)
            if value is not None:
                vap_frac = float(value)
                if not 0.0 <= vap_frac <= 1.0:
                    raise UnitOperationError(
                        f"HeatExchanger '{self.unit_id}' vapor_fraction must be between 0 and 1"
                    )
                return vap_frac
        return None

    def _pressure_drop_for_port(self, port: str, thermal_role: str) -> float:
        port_lower = port.lower()
        names = []
        if 'tube' in port_lower:
            names.append('P_drop_tube')
        if 'shell' in port_lower:
            names.append('P_drop_shell')
        names.append(f'P_drop_{thermal_role}')
        names.append('P_drop')

        for name in names:
            value = self.get_param(name)
            if value is not None:
                P_drop = float(value)
                if P_drop < 0.0:
                    raise UnitOperationError(
                        f"HeatExchanger '{self.unit_id}' requires nonnegative {name}"
                    )
                return P_drop
        return 0.0

    def _outlet_pressure(self, stream: StreamState, P_drop: float, port: str) -> float:
        P_out = stream.P - P_drop
        if P_out <= 0.0:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' outlet pressure for port '{port}' "
                "must be positive"
            )
        return P_out

    def _outlet_port_name(self, inlet_port: str, fallback_side: str) -> str:
        port_lower = inlet_port.lower()
        if 'tube' in port_lower:
            return 'tube_out'
        if 'shell' in port_lower:
            return 'shell_out'
        if 'hot' in port_lower:
            return 'hot_out'
        if 'cold' in port_lower:
            return 'cold_out'
        if port_lower.endswith('_in'):
            return f"{inlet_port[:-3]}_out"
        if port_lower.endswith('in'):
            return f"{inlet_port[:-2]}out"
        return f'{fallback_side}_out'

    def _enforce_heat_transfer_direction(self, Q: float) -> None:
        tolerance = max(1e-9, abs(Q) * 1e-12)
        if Q <= tolerance:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' must transfer heat from hot side "
                f"to cold side; calculated duty is {Q / 3600:.6g} kW"
            )

    def _stream_enthalpy(self, stream: StreamState) -> float:
        if stream.H is not None:
            return stream.H
        return self.thermo.mixture_enthalpy(
            stream.composition,
            stream.T,
            stream.vapor_fraction,
            stream.x,
            stream.y,
            stream.P,
        )

    def _outlet_from_heat_added(self, stream: StreamState, P_out: float,
                                Q: float,
                                force_phase: str | None = None) -> StreamState:
        if stream.F <= 0.0:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' cannot transfer heat to a zero-flow stream"
            )
        H_target = self._stream_enthalpy(stream) + Q / stream.F
        return self._state_for_enthalpy(
            stream.composition, P_out, stream.F, H_target, stream.T,
            force_phase=force_phase,
        )

    def _outlet_from_heat_removed(self, stream: StreamState, P_out: float,
                                  Q: float,
                                  force_phase: str | None = None) -> StreamState:
        if stream.F <= 0.0:
            raise UnitOperationError(
                f"HeatExchanger '{self.unit_id}' cannot remove heat from a zero-flow stream"
            )
        H_target = self._stream_enthalpy(stream) - Q / stream.F
        return self._state_for_enthalpy(
            stream.composition, P_out, stream.F, H_target, stream.T,
            force_phase=force_phase,
        )

    def _state_for_vapor_fraction(self, composition: dict, P: float, F: float,
                                  vapor_fraction: float) -> StreamState:
        direct_pq = getattr(self.thermo, 'calculate_state_PQ', None)
        if direct_pq is not None:
            try:
                return direct_pq(P, vapor_fraction, F, composition)
            except (NotImplementedError, AttributeError):
                pass

        T, x, y = self.thermo.flash_PV(composition, P, vapor_fraction)
        state = self.thermo.calculate_state(T, P, F, composition)
        state.vapor_fraction = vapor_fraction
        state.x = x
        state.y = y
        return state

    def _state_for_enthalpy(self, composition: dict, P: float, F: float,
                            H_target: float, T_guess: float, include=None,
                            force_phase: str | None = None) -> StreamState:
        state, _error = _ThermoStateSolver(
            self.thermo,
            f"HeatExchanger '{self.unit_id}'",
        ).state_at_enthalpy(
            P, F, composition, H_target, T_guess, include=include,
            force_phase=force_phase,
        )
        return state

    def _rate_exchanger(self, hot: StreamState, cold: StreamState,
                        U: float, A: float, P_drop_hot: float, P_drop_cold: float
                        ) -> tuple[float, float, float]:
        """Compatibility wrapper for the curve-based rating calculation."""
        P_hot_out = self._outlet_pressure(hot, P_drop_hot, 'hot')
        P_cold_out = self._outlet_pressure(cold, P_drop_cold, 'cold')
        Q = self._rate_exchanger_by_curves(
            hot,
            cold,
            P_hot_out,
            P_cold_out,
            float(U),
            float(A),
            None,
            False,
            self._curve_segments(),
            self._flow_pattern(),
        )
        hot_out = self._outlet_from_heat_removed(hot, P_hot_out, Q)
        cold_out = self._outlet_from_heat_added(cold, P_cold_out, Q)
        return hot_out.T, cold_out.T, Q
    
    def _find_T_for_H(self, composition: dict, P: float, H_target: float,
                      T_guess: float) -> float:
        return self._state_for_enthalpy(composition, P, 1.0, H_target, T_guess).T


class Flash(UnitOperation):
    """Single-stage vapor-liquid flash separation"""

    supports_permanent_solids = True
    particle_size_behavior = 'nonselective'

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if len(inlets) != 1:
            raise UnitOperationError(
                f"Flash '{self.unit_id}' requires exactly 1 inlet stream"
            )
        inlet = next(iter(inlets.values()))
        if inlet.F <= 0.0:
            raise UnitOperationError(
                f"Flash '{self.unit_id}' requires positive inlet flow"
            )

        T = self.get_temperature_param('T')
        if T is None:
            T = self.get_temperature_param('temperature')
        P = self.get_param('P', self.get_param('pressure'))
        VF = self._vapor_fraction_spec()
        Q = self._heat_duty_spec()

        pressure_was_specified = P is not None
        nonpressure_specs = sum(1 for value in (T, VF, Q) if value is not None)
        pressure_inherited = P is None and nonpressure_specs == 1
        if pressure_inherited:
            P = inlet.P

        # Count specifications
        specs = sum(1 for x in [T, P, VF, Q] if x is not None)
        if specs != 2:
            raise UnitOperationError(
                f"Flash '{self.unit_id}' requires exactly 2 specifications "
                f"(T, P, VF, Q), got {specs}"
            )
        
        if T is not None:
            T = float(T)
            if not math.isfinite(T) or T <= 0.0:
                raise UnitOperationError(
                    f"Flash '{self.unit_id}' temperature must be positive"
                )
        if P is not None:
            P = float(P)
            if not math.isfinite(P) or P <= 0.0:
                raise UnitOperationError(
                    f"Flash '{self.unit_id}' pressure must be positive"
                )
        if VF is not None:
            VF = float(VF)
            if not 0.0 <= VF <= 1.0:
                raise UnitOperationError(
                    f"Flash '{self.unit_id}' vapor_fraction must be between 0 and 1"
                )

        composition = self._normalized_composition(inlet.composition)
        reference_T = T if T is not None else inlet.T
        aqueous_context, henry_info, henry_warnings = self._flash_henry_context(
            inlet,
            composition,
            reference_T,
        )
        inlet_H = self._stream_enthalpy(inlet, aqueous_context)
        H_target = inlet_H + Q / inlet.F if Q is not None else None
        state = None

        # Solve based on which specs are given
        if T is not None and P is not None:
            # TP flash
            state = self._state_at_TP(
                composition, T, P, inlet.F, aqueous_context
            )

        elif P is not None and VF is not None:
            # PV flash
            state = self._state_at_PV(
                composition, P, VF, inlet.F, inlet.T, aqueous_context
            )

        elif T is not None and VF is not None:
            # TV flash
            P = self._find_P_for_VF(
                composition, T, VF, inlet.P, aqueous_context
            )
            state = self._state_at_TP(
                composition, T, P, inlet.F, aqueous_context
            )

        elif P is not None and Q is not None:
            # PH flash, with heat duty converted to target outlet enthalpy.
            state = self._state_at_PH(
                composition, P, H_target, inlet.F, inlet.T, aqueous_context
            )

        elif T is not None and Q is not None:
            # TH flash - find pressure for target enthalpy.
            P = self._find_P_for_H_at_T(
                composition, T, H_target, inlet.P, inlet.F, aqueous_context
            )
            state = self._state_at_TP(
                composition, T, P, inlet.F, aqueous_context
            )

        elif VF is not None and Q is not None:
            # VH flash - find pressure; T follows from the PV flash.
            P = self._find_P_for_VF_and_H(
                composition, VF, H_target, inlet.P, inlet.F, aqueous_context
            )
            state = self._state_at_PV(
                composition, P, VF, inlet.F, inlet.T, aqueous_context
            )

        else:
            raise UnitOperationError(
                f"Flash '{self.unit_id}' specification combination not supported"
            )

        T = state.T
        P = state.P
        self.thermo._record_estimated_interaction_extrapolation(
            T,
            (
                (state.effective_liquid1_fraction, state.x1 or state.x or {}),
                (state.liquid2_fraction, state.x2 or {}),
            ),
        )
        vap_frac = max(0.0, min(1.0, state.vapor_fraction))
        x = self._normalized_composition(state.x) if state.x else {}
        y = self._normalized_composition(state.y) if state.y else {}

        # Create outlet streams
        F_vap = inlet.F * vap_frac
        phase_flows = state.phase_component_flows()
        nonvapor_component_flows = {}
        for phase_name in ('liquid1', 'liquid2', 'solid'):
            for component, flow in phase_flows[phase_name].items():
                nonvapor_component_flows[component] = (
                    nonvapor_component_flows.get(component, 0.0) + flow
                )
        F_liq = sum(nonvapor_component_flows.values())
        nonvapor_composition = (
            {
                component: flow / F_liq
                for component, flow in nonvapor_component_flows.items()
                if flow > 0.0
            }
            if F_liq > 0.0 else dict(x)
        )

        vap_out = self._phase_outlet(
            T, P, F_vap, y, 'vapor', aqueous_context
        )
        liq_out = self._phase_outlet(
            T, P, F_liq, nonvapor_composition, 'liquid', aqueous_context
        )
        # An ordinary flash has one physical liquid outlet. Under a universal
        # LLE-aware phase policy, retain both equilibrium liquids inside that
        # outlet rather than silently collapsing them or acting as a decanter.
        if state.liquid2_fraction > 1.0e-12 and F_liq > 0.0:
            nonvapor_fraction = 1.0 - state.vapor_fraction
            if nonvapor_fraction > 0.0:
                liq_out.liquid1_fraction = (
                    state.effective_liquid1_fraction / nonvapor_fraction
                )
                liq_out.liquid2_fraction = (
                    state.liquid2_fraction / nonvapor_fraction
                )
                liq_out.solid_fraction = state.solid_fraction / nonvapor_fraction
                liq_out.solid_composition = (
                    dict(state.solid_composition)
                    if state.solid_composition else None
                )
                liq_out.solid_component_flows = dict(state.solid_component_flows)
                liq_out.solid_particle_properties = {
                    component: dict(values)
                    for component, values in state.solid_particle_properties.items()
                }
                liq_out.x1 = dict(state.x1 or x)
                liq_out.x2 = dict(state.x2 or x)
                liq_out.x = dict(x)
                liq_out.fluid_phase_model = state.fluid_phase_model
                liq_out.phase_status = state.phase_status
                liq_out.phase_stability = state.phase_stability
                liq_out.phase_details = dict(state.phase_details)
                include = frozenset(
                    name
                    for name, value in (
                        ('H', liq_out.H),
                        ('S', liq_out.S),
                        ('Cp', liq_out.Cp),
                        ('rho', liq_out.rho),
                        ('mu', liq_out.mu),
                    )
                    if value is not None
                )
                self.thermo._populate_multifluid_state_properties(
                    liq_out,
                    include,
                )

        state_H = (
            state.H
            if state.H is not None
            else self._stream_enthalpy(state, aqueous_context)
        )
        Q_calc = inlet.F * (state_H - inlet_H)
        duty_residual = Q_calc - Q if Q is not None else None
        
        performance = {
            'T_C': T - 273.15,
            'P_bar': P,
            'pressure_source': (
                'upstream'
                if pressure_inherited else
                ('specified' if pressure_was_specified else 'solved')
            ),
            'vapor_fraction': vap_frac,
            'fluid_vapor_fraction': state.fluid_vapor_fraction,
            'duty_kW': Q_calc / 3600,
            'liquid_composition': x,
            'nonvapor_composition': nonvapor_composition,
            'vapor_composition': y,
            'vapor_flow_kmol_hr': F_vap,
            'liquid_flow_kmol_hr': F_liq,
            'liquid1_fraction_of_liquid_outlet': liq_out.effective_liquid1_fraction,
            'liquid2_fraction_of_liquid_outlet': liq_out.liquid2_fraction,
            'henry': henry_info,
        }
        if duty_residual is not None:
            performance['duty_residual_kW'] = duty_residual / 3600

        warnings = list(henry_warnings)
        if aqueous_context is not None and F_liq > 1e-12:
            warnings.extend(self._henry_solution_warnings(
                aqueous_context,
                henry_info,
                [x],
                [T],
                [P],
            ))

        return UnitResult(
            outlet_streams={
                'vapor_out': vap_out,
                'liquid_out': liq_out,
            },
            heat_duty=Q_calc,
            performance=performance,
            warnings=warnings,
        )

    def _vapor_fraction_spec(self):
        return self.get_param(
            'VF',
            self.get_param(
                'vap_frac',
                self.get_param('vapor_frac', self.get_param('vapor_fraction')),
            ),
        )

    def _heat_duty_spec(self):
        for name in ('Q', 'duty', 'heat_duty'):
            value = self.get_param(name)
            if value is None:
                continue
            duty = float(value)
            unit = (self.get_param_unit(name) or '').strip().lower()
            if unit in ('w', 'watt', 'watts'):
                return duty * 3.6
            if unit in ('mw', 'megawatt', 'megawatts'):
                return duty * 3.6e6
            if unit in ('kw', 'kilowatt', 'kilowatts'):
                return duty * 3600.0
            if unit in ('kj/h', 'kj/hr', 'kj per h', 'kj per hour'):
                return duty
            if unit in ('j/h', 'j/hr', 'j per h', 'j per hour'):
                return duty / 1000.0
            return duty * 3600.0 if abs(duty) < 1e6 else duty
        return None

    @staticmethod
    def _normalized_composition(composition: dict) -> dict:
        total = sum(max(float(value), 0.0) for value in composition.values())
        if total <= 0.0:
            raise UnitOperationError("Flash feed composition cannot be empty")
        return {
            comp: max(float(value), 0.0) / total
            for comp, value in composition.items()
        }

    def _flash_henry_context(self, inlet: StreamState, composition: dict,
                             reference_T: float):
        comps = list(self.thermo.components)
        water = self._henry_water_component(comps)
        total_available_moles = {
            comp: inlet.F * composition.get(comp, 0.0)
            for comp in comps
        }
        potential_liquid_moles = {
            comp: (
                0.0
                if comp != water
                and self._henry_noncondensable_candidate(comp, reference_T)
                else total_available_moles[comp]
            )
            for comp in comps
        }
        # A flash has no dedicated solvent inlet. Its conservative pre-solve
        # aqueous estimate is the condensable portion of the feed itself.
        solvent_liquid_moles = dict(potential_liquid_moles)
        return self._henry_context_from_estimates(
            comps,
            reference_T=reference_T,
            solvent_liquid_moles=solvent_liquid_moles,
            potential_liquid_moles=potential_liquid_moles,
            total_available_moles=total_available_moles,
            aqueous_loading_water_moles=potential_liquid_moles.get(water, 0.0),
        )

    def _stream_enthalpy(self, stream: StreamState,
                         aqueous_context=None) -> float:
        if aqueous_context is not None:
            (
                _normalized,
                fluid_fraction,
                fluid_composition,
                _solid_fraction,
                solid_composition,
            ) = self.thermo._split_permanent_solid_composition(
                stream.composition
            )
            fluid_H = self.thermo.aqueous_mixture_enthalpy(
                fluid_composition,
                stream.T,
                stream.fluid_vapor_fraction or 0.0,
                aqueous_context,
                x=stream.x,
                y=stream.y,
                P=stream.P,
            )
            return fluid_fraction * fluid_H + sum(
                stream.composition.get(component, 0.0)
                * 1000.0
                * self.thermo.enthalpy_solid(component, stream.T)
                for component in solid_composition
            )
        if stream.H is not None:
            return stream.H
        return self.thermo.mixture_enthalpy(
            stream.composition,
            stream.T,
            stream.vapor_fraction,
            stream.x,
            stream.y,
            stream.P,
        )

    def _state_at_TP(self, composition: dict, T: float, P: float, F: float,
                     aqueous_context=None) -> StreamState:
        if aqueous_context is not None:
            (
                normalized,
                fluid_fraction,
                fluid_composition,
                solid_fraction,
                solid_composition,
            ) = self.thermo._split_permanent_solid_composition(composition)
            if fluid_fraction <= 1.0e-15:
                return self.thermo.calculate_state(T, P, F, normalized)
            V, x, y = self.thermo.aqueous_flash_TP(
                fluid_composition,
                T,
                P,
                aqueous_context,
            )
            fluid_state = self._combined_state_from_flash(
                T, P, F * fluid_fraction, fluid_composition,
                V, x, y, aqueous_context
            )
            if solid_fraction <= 1.0e-15:
                return fluid_state
            return self.thermo._combine_permanent_solid_state(
                T=T,
                P=P,
                F=F,
                composition=normalized,
                fluid_fraction=fluid_fraction,
                solid_fraction=solid_fraction,
                solid_composition=solid_composition,
                fluid_state=fluid_state,
                include_set=self.thermo._normalize_state_include(None),
            )
        return self.thermo.calculate_state(T, P, F, composition)

    def _state_at_PV(self, composition: dict, P: float, VF: float, F: float,
                     T_guess: float, aqueous_context=None) -> StreamState:
        if aqueous_context is not None:
            solver = _ThermoStateSolver(
                self.thermo,
                f"Flash '{self.unit_id}'",
            )

            def residual(T_value: float) -> float:
                return self._state_at_TP(
                    composition,
                    T_value,
                    P,
                    F,
                    aqueous_context,
                ).vapor_fraction - VF

            T = solver._solve_temperature(
                residual,
                solver._temperature_grid(T_guess),
                f"Henry-aware vapor fraction {VF:g} at {P:g} bar",
                preferred_T=T_guess,
                residual_tolerance=1e-8,
            )
            return self._state_at_TP(
                composition, T, P, F, aqueous_context
            )
        direct_pq = getattr(self.thermo, 'calculate_state_PQ', None)
        if direct_pq is not None:
            try:
                return direct_pq(P, VF, F, composition)
            except (NotImplementedError, AttributeError):
                pass
            except Exception:
                if any(
                    composition.get(component, 0.0) > 0.0
                    for component in getattr(
                        self.thermo, 'permanent_solid_components', ()
                    )
                ):
                    raise
                if getattr(self.thermo, 'fluid_phase_model', 'VLE') != 'VLE':
                    raise
                pass
        T, x, y = self.thermo.flash_PV(composition, P, VF)
        return self._combined_state_from_flash(T, P, F, composition, VF, x, y)

    def _state_at_PH(self, composition: dict, P: float, H_target: float, F: float,
                     T_guess: float, aqueous_context=None) -> StreamState:
        if aqueous_context is not None:
            solver = _ThermoStateSolver(
                self.thermo,
                f"Flash '{self.unit_id}'",
            )

            def residual(T_value: float) -> float:
                state = self._state_at_TP(
                    composition,
                    T_value,
                    P,
                    F,
                    aqueous_context,
                )
                if state.H is None:
                    raise UnitOperationError("Enthalpy was not calculated")
                return state.H - H_target

            T = solver._solve_temperature(
                residual,
                solver._temperature_grid(T_guess),
                f"Henry-aware target enthalpy at {P:g} bar",
                preferred_T=T_guess,
                residual_tolerance=max(1e-6, abs(H_target) * 1e-8),
            )
            state = self._state_at_TP(
                composition, T, P, F, aqueous_context
            )
            tolerance = max(1e-6, abs(H_target) * 1e-8)
            if state.H is None or abs(state.H - H_target) > tolerance:
                residual_value = None if state.H is None else state.H - H_target
                raise UnitOperationError(
                    f"Flash '{self.unit_id}' Henry-aware PH solve residual is "
                    f"{residual_value!r} kJ/kmol"
                )
            return state
        state, residual = _ThermoStateSolver(
            self.thermo,
            f"Flash '{self.unit_id}'",
        ).state_at_enthalpy(P, F, composition, H_target, T_guess)
        tolerance = max(1e-6, abs(H_target) * 1e-8)
        if abs(residual) > tolerance:
            raise UnitOperationError(
                f"Flash '{self.unit_id}' PH solve residual is {residual:.4g} kJ/kmol"
            )
        return state

    def _combined_state_from_flash(self, T: float, P: float, F: float,
                                   composition: dict, VF: float,
                                   x: dict, y: dict,
                                   aqueous_context=None) -> StreamState:
        composition = self._normalized_composition(composition)
        x = self._normalized_composition(x)
        y = self._normalized_composition(y)
        VF = max(0.0, min(1.0, float(VF)))
        state = StreamState(
            T=T,
            P=P,
            F=F,
            composition=composition,
            vapor_fraction=VF,
            liquid1_fraction=1.0 - VF,
            x=x,
            x1=x,
            y=y,
            fluid_phase_model=getattr(self.thermo, 'fluid_phase_model', 'VLE'),
            phase_status='aqueous_vle' if aqueous_context is not None else 'ordinary_vle',
            phase_stability=(
                'aqueous_henry_vle_constrained'
                if aqueous_context is not None else 'vle_constrained'
            ),
        )
        state.MW = self.thermo.mixture_MW(composition)
        if aqueous_context is not None:
            state.H = self.thermo.aqueous_mixture_enthalpy(
                composition, T, VF, aqueous_context, x=x, y=y, P=P
            )
        else:
            state.H = self.thermo.mixture_enthalpy(
                composition, T, VF, x, y, P
            )
        try:
            state.S = self.thermo.mixture_entropy(composition, T, VF, x, y, P)
        except Exception:
            state.S = None
        try:
            state.Cp = self.thermo.phase_weighted_mixture_Cp(composition, T, VF, x, y, P)
        except Exception:
            state.Cp = None
        try:
            state.rho = self.thermo.mixture_molar_density(composition, T, P, VF, x, y)
        except Exception:
            state.rho = None
        return state

    def _phase_outlet(self, T: float, P: float, F: float,
                      composition: dict, phase: str,
                      aqueous_context=None) -> StreamState:
        try:
            state = self.thermo.calculate_state(
                T, P, F, composition, phase=phase, flash=False
            )
        except Exception:
            state = StreamState(
                T=T,
                P=P,
                F=F,
                composition=dict(composition),
                vapor_fraction=1.0 if phase == 'vapor' else 0.0,
                liquid1_fraction=0.0 if phase == 'vapor' else 1.0,
                x1=None if phase == 'vapor' else dict(composition),
                fluid_phase_model=getattr(
                    self.thermo, 'fluid_phase_model', 'VLE'
                ),
                phase_status=f'fallback_forced_{phase}',
                phase_stability='explicit_phase_constraint',
            )
            state.MW = self.thermo.mixture_MW(composition)
            try:
                state.H = self.thermo.mixture_enthalpy(
                    composition,
                    T,
                    state.vapor_fraction,
                    None if phase == 'vapor' else composition,
                    composition if phase == 'vapor' else None,
                    P,
                )
            except Exception:
                state.H = None
        if phase == 'vapor':
            state.x = None
            state.x1 = None
            state.y = dict(composition)
            state.vapor_fraction = 1.0
            state.liquid1_fraction = 0.0
        else:
            if state.solid_fraction <= 1.0e-15:
                state.x = dict(composition)
                state.x1 = dict(composition)
                state.y = None
                state.vapor_fraction = 0.0
                state.liquid1_fraction = 1.0
            if aqueous_context is not None:
                (
                    _normalized,
                    fluid_fraction,
                    fluid_composition,
                    _solid_fraction,
                    solid_composition,
                ) = self.thermo._split_permanent_solid_composition(composition)
                state.H = (
                    fluid_fraction
                    * self.thermo.aqueous_liquid_enthalpy(
                        fluid_composition, T, P, aqueous_context
                    )
                    + sum(
                        composition.get(component, 0.0)
                        * 1000.0
                        * self.thermo.enthalpy_solid(component, T)
                        for component in solid_composition
                    )
                )
        return state

    def _find_P_for_VF(self, composition: dict, T: float, VF_target: float,
                       P_guess: float, aqueous_context=None) -> float:
        """Find pressure for given vapor fraction at temperature."""
        def residual(P_value: float) -> float:
            return self._state_at_TP(
                composition,
                T,
                P_value,
                1.0,
                aqueous_context,
            ).vapor_fraction - VF_target

        return self._solve_pressure(
            residual,
            P_guess,
            f"vapor fraction {VF_target:g} at {T:.2f} K",
            residual_tolerance=1e-7,
        )

    def _find_P_for_H_at_T(self, composition: dict, T: float,
                           H_target: float, P_guess: float,
                           F: float, aqueous_context=None) -> float:
        def residual(P_value: float) -> float:
            state = self._state_at_TP(
                composition, T, P_value, F, aqueous_context
            )
            if state.H is None:
                raise UnitOperationError("Enthalpy was not calculated")
            return state.H - H_target

        return self._solve_pressure(
            residual,
            P_guess,
            f"target enthalpy at {T:.2f} K",
            residual_tolerance=max(1e-6, abs(H_target) * 1e-8),
        )

    def _find_P_for_VF_and_H(self, composition: dict, VF: float,
                             H_target: float, P_guess: float,
                             F: float, aqueous_context=None) -> float:
        def residual(P_value: float) -> float:
            state = self._state_at_PV(
                composition,
                P_value,
                VF,
                F,
                350.0,
                aqueous_context,
            )
            if state.H is None:
                raise UnitOperationError("Enthalpy was not calculated")
            return state.H - H_target

        return self._solve_pressure(
            residual,
            P_guess,
            f"target enthalpy at vapor fraction {VF:g}",
            residual_tolerance=max(1e-6, abs(H_target) * 1e-8),
        )

    def _solve_pressure(self, residual, P_guess: float, label: str,
                        residual_tolerance: float) -> float:
        P_guess = (
            float(P_guess)
            if P_guess and math.isfinite(float(P_guess))
            else THERMOCHEMICAL_STANDARD_PRESSURE_BAR
        )
        candidates = self._pressure_grid(P_guess)
        evaluated = []
        for P in candidates:
            try:
                value = residual(P)
                if math.isfinite(value):
                    evaluated.append((P, value))
            except Exception:
                continue
        near_roots = [
            (P, value)
            for P, value in evaluated
            if abs(value) <= residual_tolerance
        ]
        if near_roots:
            P, _value = min(
                near_roots,
                key=lambda item: abs(math.log(max(item[0], 1e-30) / P_guess)),
            )
            return P
        brackets = []
        for (P1, f1), (P2, f2) in zip(evaluated, evaluated[1:]):
            if f1 * f2 < 0.0:
                brackets.append((P1, P2))
        if brackets:
            P1, P2 = min(
                brackets,
                key=lambda bracket: (
                    abs(math.log(max(0.5 * (bracket[0] + bracket[1]), 1e-30) / P_guess)),
                    bracket[1] - bracket[0],
                ),
            )
            return brentq(residual, P1, P2, xtol=1e-8, rtol=1e-9, maxiter=100)
        if evaluated:
            P_best, f_best = min(evaluated, key=lambda item: abs(item[1]))
            raise UnitOperationError(
                f"Flash '{self.unit_id}' could not bracket {label}; "
                f"best residual {f_best:.4g} at {P_best:.4g} bar"
            )
        raise UnitOperationError(
            f"Flash '{self.unit_id}' could not evaluate pressure solve for {label}"
        )

    @staticmethod
    def _pressure_grid(P_guess: float) -> list[float]:
        P_guess = max(float(P_guess), 1e-6)
        candidates = [
            1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2,
            0.5, THERMOCHEMICAL_STANDARD_PRESSURE_BAR, ATM_PRESSURE_BAR,
            2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
            200.0, 500.0, 1000.0,
        ]
        for multiplier in (0.001, 0.003, 0.01, 0.03, 0.1, 0.25, 0.5, 0.8,
                           1.0, 1.25, 2.0, 4.0, 10.0, 30.0, 100.0):
            candidates.append(P_guess * multiplier)
        return sorted(set(max(1e-8, min(5000.0, float(P))) for P in candidates))
