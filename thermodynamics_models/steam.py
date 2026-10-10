import math
from typing import Iterable, Optional, Union

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..chemical_properties import ChemicalDatabase
else:
    from chemical_properties import ChemicalDatabase

from .common import (
    P_REF,
    STEAM_WATER_H_OFFSET,
    STEAM_WATER_MW,
    STEAM_WATER_S_OFFSET,
    ThermodynamicsError,
)
from .base import IdealThermodynamics, StreamState, TransportPhaseValues

class SteamThermodynamics(IdealThermodynamics):
    """
    Pure-water steam-table backend using CoolProp's IF97 implementation.

    The public interface uses pfdsim units and reference conventions:
    pressure in bar, temperature in K, enthalpy in kJ/kmol, entropy/Cp in
    kJ/kmol-K, and molar density in kmol/m3.
    """

    FLUID = 'IF97::Water'

    def __init__(self, components: list[str], db: Optional[ChemicalDatabase] = None):
        super().__init__(components, db)
        if len(components) != 1:
            raise ThermodynamicsError("STEAM thermodynamics supports pure H2O streams only")

        self.water_component = components[0]
        props = self.props[self.water_component]
        identifiers = {
            self.water_component.strip().lower(),
            getattr(props, 'symbol', '').strip().lower(),
            getattr(props, 'formula', '').strip().lower(),
            getattr(props, 'name', '').strip().lower(),
        }
        if not ({'h2o', 'water'} & identifiers):
            raise ThermodynamicsError("STEAM thermodynamics supports pure H2O streams only")

        try:
            from CoolProp.CoolProp import PhaseSI, PropsSI
        except Exception as exc:
            raise ThermodynamicsError(
                "STEAM thermodynamics requires CoolProp. Install it with "
                "`python -m pip install CoolProp`."
            ) from exc

        self._props_si = PropsSI
        self._phase_si = PhaseSI
        self._water_mw = STEAM_WATER_MW
        self._pcrit_bar = self._props('pcrit', '', 0.0, '', 0.0) / 1e5
        self._tcrit = self._props('Tcrit', '', 0.0, '', 0.0)
        self._tmin = self._props('Tmin', '', 0.0, '', 0.0)
        self._reference_H_offset = STEAM_WATER_H_OFFSET
        self._reference_S_offset = STEAM_WATER_S_OFFSET

    def _props(self, output: str, input1: str, value1: float,
               input2: str, value2: float) -> float:
        try:
            return float(self._props_si(output, input1, value1, input2, value2, self.FLUID))
        except Exception as exc:
            raise ThermodynamicsError(
                f"CoolProp IF97 failed for {output}({input1}={value1}, "
                f"{input2}={value2})"
            ) from exc

    def _phase(self, input1: str, value1: float, input2: str, value2: float) -> str:
        try:
            return str(self._phase_si(input1, value1, input2, value2, self.FLUID))
        except Exception:
            return ''

    @staticmethod
    def _pressure_to_pa(P_bar: float) -> float:
        return float(P_bar) * 1e5

    def _mass_enthalpy_to_molar(self, H_mass: float) -> float:
        return float(H_mass) * self._water_mw / 1000.0

    def _molar_enthalpy_to_mass(self, H_molar: float) -> float:
        return float(H_molar) * 1000.0 / self._water_mw

    def _mass_entropy_to_molar(self, S_mass: float) -> float:
        return float(S_mass) * self._water_mw / 1000.0

    def _molar_entropy_to_mass(self, S_molar: float) -> float:
        return float(S_molar) * 1000.0 / self._water_mw

    def _density_to_molar(self, density_mass: float) -> float:
        return float(density_mass) / self._water_mw

    def _molar_density_to_mass(self, density_molar: float) -> float:
        return float(density_molar) * self._water_mw

    def _pure_viscosity(self, comp: str, T: float, P: float, phase: str) -> float:
        """Pure-water dynamic viscosity [Pa*s] from the IF97 backend."""
        self._normalize_water_composition({comp: 1.0})
        phase_key = 'vapor' if phase in {'gas', 'vapor'} else 'liquid'
        cache_key = (phase_key, comp, float(T), float(P))
        cached = self._viscosity_cache.get(cache_key)
        if cached is not None:
            return cached

        use_saturation = False
        if P < self._pcrit_bar:
            T_sat = self._saturation_temperature(P)
            use_saturation = abs(T - T_sat) <= max(1e-6, 1e-8 * T_sat)
        if use_saturation:
            quality = 1.0 if phase_key == 'vapor' else 0.0
            value = self._props(
                'V', 'P', self._pressure_to_pa(P), 'Q', quality
            )
        else:
            value = self._props(
                'V', 'T', float(T), 'P', self._pressure_to_pa(P)
            )

        if value <= 0.0 or not math.isfinite(value):
            raise ThermodynamicsError(
                f"CoolProp IF97 returned invalid {phase_key} viscosity at "
                f"T={T:.1f} K and P={P:.4g} bar"
            )
        return self._set_limited_cache(self._viscosity_cache, cache_key, value)

    def transport_mixture_thermal_conductivity(
        self, composition: dict[str, float], T: float, P: float,
        vapor_fraction: float = 1.0, x: Optional[dict] = None,
        y: Optional[dict] = None,
    ) -> TransportPhaseValues:
        """Pressure-dependent liquid/vapor water conductivity from IF97."""
        self._normalize_water_composition(composition)
        vf = float(vapor_fraction)
        if not math.isfinite(vf) or not 0 <= vf <= 1:
            raise ThermodynamicsError('Conductivity requires vapor fraction between zero and one')
        saturated = P < self._pcrit_bar and abs(T-self._saturation_temperature(P)) <= max(1e-6, 1e-8*T)
        def value(quality):
            result = (self._props('L', 'P', self._pressure_to_pa(P), 'Q', quality) if saturated
                      else self._props('L', 'T', T, 'P', self._pressure_to_pa(P)))
            if not math.isfinite(result) or result <= 0:
                raise ThermodynamicsError('IF97 returned invalid thermal conductivity')
            return result
        return TransportPhaseValues(liquid=value(0) if vf < 1 else None,
                                    vapor=value(1) if vf > 0 else None)

    def _normalize_water_composition(self, composition: dict[str, float]) -> dict[str, float]:
        total = sum(max(float(value), 0.0) for value in composition.values())
        if total <= 0.0:
            raise ThermodynamicsError("STEAM thermodynamics requires a positive H2O composition")

        water_total = 0.0
        nonwater = []
        for comp, value in composition.items():
            fraction = max(float(value), 0.0) / total
            props = self.props.get(comp)
            identifiers = {
                str(comp).strip().lower(),
                getattr(props, 'symbol', '').strip().lower() if props else '',
                getattr(props, 'formula', '').strip().lower() if props else '',
                getattr(props, 'name', '').strip().lower() if props else '',
            }
            if {'h2o', 'water'} & identifiers:
                water_total += fraction
            elif fraction > 1e-12:
                nonwater.append(comp)

        if nonwater or abs(water_total - 1.0) > 1e-10:
            raise ThermodynamicsError("STEAM thermodynamics supports pure H2O streams only")
        return {self.water_component: 1.0}

    def _saturation_temperature(self, P: float) -> float:
        if P >= self._pcrit_bar:
            raise ThermodynamicsError("Water has no saturation temperature above the critical pressure")
        return self._props('T', 'P', self._pressure_to_pa(P), 'Q', 0.0)

    def _saturation_pressure(self, T: float) -> float:
        if T >= self._tcrit:
            raise ThermodynamicsError("Water has no saturation pressure above the critical temperature")
        return self._props('P', 'T', T, 'Q', 0.0) / 1e5

    def _quality_from_pair(self, input1: str, value1: float,
                           input2: str, value2: float) -> Optional[float]:
        if input1 == 'Q':
            return float(value1)
        if input2 == 'Q':
            return float(value2)
        try:
            quality = self._props('Q', input1, value1, input2, value2)
        except ThermodynamicsError:
            return None
        if math.isnan(quality) or quality < -0.5:
            return None
        return max(0.0, min(1.0, quality))

    def _vapor_fraction_from_TP(self, T: float, P: float,
                                phase: Optional[str] = None) -> tuple[float, Optional[float]]:
        phase_lower = phase.lower() if phase else None
        if phase_lower in ('vapor', 'gas'):
            try:
                T_sat = self._saturation_temperature(P)
                if abs(T - T_sat) <= max(1e-6, 1e-8 * T_sat):
                    return 1.0, 1.0
            except ThermodynamicsError:
                pass
            return 1.0, None
        if phase_lower == 'liquid':
            try:
                T_sat = self._saturation_temperature(P)
                if abs(T - T_sat) <= max(1e-6, 1e-8 * T_sat):
                    return 0.0, 0.0
            except ThermodynamicsError:
                pass
            return 0.0, None
        try:
            T_sat = self._saturation_temperature(P)
            if abs(T - T_sat) <= max(1e-6, 1e-8 * T_sat):
                self.add_warning(
                    "STEAM state specified by T and P lies on the saturation curve; "
                    "assuming saturated vapor unless a phase is forced."
                )
                return 1.0, 1.0
            return (0.0, None) if T < T_sat else (1.0, None)
        except ThermodynamicsError:
            phase_name = self._phase('T', T, 'P', self._pressure_to_pa(P)).lower()
            if 'liquid' in phase_name:
                return 0.0, None
            return 1.0, None

    def _forced_phase_conflicts_with_TP(self, T: float, P: float,
                                        forced_liquid: bool) -> bool:
        """True when the forced phase is not the stable IF97 phase at (T, P).

        CoolProp (T, P) lookups always return the stable phase, so a
        conflicting forced phase must be evaluated on the saturation curve
        at T instead of silently taking the other phase's properties.
        """
        if T >= self._tcrit or P >= self._pcrit_bar:
            return False
        try:
            T_sat = self._saturation_temperature(P)
        except ThermodynamicsError:
            return False
        if abs(T - T_sat) <= max(1e-6, 1e-8 * T_sat):
            return False
        return (T < T_sat) != forced_liquid

    def _phase_compositions(self, vapor_fraction: float,
                            composition: dict[str, float]) -> tuple[Optional[dict], Optional[dict]]:
        if vapor_fraction <= 1e-12:
            return dict(composition), None
        if vapor_fraction >= 1.0 - 1e-12:
            return None, dict(composition)
        return dict(composition), dict(composition)

    def _state_from_inputs(self, input1: str, value1: float,
                           input2: str, value2: float,
                           F: float,
                           composition: dict[str, float],
                           include: Optional[Union[str, Iterable[str]]] = None,
                           vapor_fraction: Optional[float] = None) -> StreamState:
        include_set = self._normalize_state_include(include)
        composition = self._normalize_water_composition(composition)

        T = self._props('T', input1, value1, input2, value2)
        P = self._props('P', input1, value1, input2, value2) / 1e5
        quality = self._quality_from_pair(input1, value1, input2, value2)
        if vapor_fraction is None:
            vapor_fraction = quality if quality is not None else self._vapor_fraction_from_TP(T, P)[0]
        vapor_fraction = max(0.0, min(1.0, float(vapor_fraction)))
        x, y = self._phase_compositions(vapor_fraction, composition)
        state = StreamState(
            T=T,
            P=P,
            F=F,
            composition=composition,
            vapor_fraction=vapor_fraction,
            x=x,
            y=y,
        )
        state.MW = self._water_mw

        if 'H' in include_set:
            state.H = (
                self._mass_enthalpy_to_molar(
                    self._props('H', input1, value1, input2, value2)
                )
                + self._reference_H_offset
            )
        if 'S' in include_set:
            state.S = (
                self._mass_entropy_to_molar(
                    self._props('S', input1, value1, input2, value2)
                )
                + self._reference_S_offset
            )
        if 'rho' in include_set:
            if 0.0 < vapor_fraction < 1.0:
                rho_liq = self._density_to_molar(
                    self._props('D', 'P', self._pressure_to_pa(P), 'Q', 0.0)
                )
                rho_vap = self._density_to_molar(
                    self._props('D', 'P', self._pressure_to_pa(P), 'Q', 1.0)
                )
                molar_volume = (1.0 - vapor_fraction) / rho_liq + vapor_fraction / rho_vap
                state.rho = 1.0 / molar_volume
            else:
                state.rho = self._density_to_molar(
                    self._props('D', input1, value1, input2, value2)
                )
        if 'Cp' in include_set:
            if 0.0 < vapor_fraction < 1.0:
                state.Cp = None
            else:
                state.Cp = self._mass_entropy_to_molar(
                    self._props('C', input1, value1, input2, value2)
                )
        if 'mu' in include_set:
            state.mu = self.mixture_viscosity(
                composition, T, P, vapor_fraction, x, y
            )
        return state

    def calculate_state(self, T: float, P: float, F: float,
                       composition: dict[str, float],
                       phase: Optional[str] = None,
                       flash: bool = True,
                       include: Optional[Union[str, Iterable[str]]] = None) -> StreamState:
        solid_state = self._permanent_solid_state_at_TP(
            T, P, F, composition, phase, flash, include
        )
        if solid_state is not None:
            return solid_state
        composition = self._normalize_water_composition(composition)
        phase_lower = phase.lower() if phase else None
        vapor_fraction, quality = self._vapor_fraction_from_TP(T, P, phase_lower)
        if quality is not None:
            return self._state_from_inputs(
                'P',
                self._pressure_to_pa(P),
                'Q',
                quality,
                F,
                composition,
                include=include,
                vapor_fraction=vapor_fraction,
            )
        if phase_lower is not None and self._forced_phase_conflicts_with_TP(
            T, P, phase_lower == 'liquid'
        ):
            state = self._state_from_inputs(
                'T',
                float(T),
                'Q',
                vapor_fraction,
                F,
                composition,
                include=include,
                vapor_fraction=vapor_fraction,
            )
            # Properties are the saturation-curve continuation at T; keep the
            # caller's pressure for stream bookkeeping.
            state.P = float(P)
            return state
        return self._state_from_inputs(
            'T',
            T,
            'P',
            self._pressure_to_pa(P),
            F,
            composition,
            include=include,
            vapor_fraction=vapor_fraction,
        )

    def temperature_at_PH(self, P, H, composition, *, phase=None, T_guess=None):
        """Use IF97's native inverse and forward properties without StreamStates."""
        self._normalize_water_composition(composition)
        phase = str(phase or '').strip().lower()
        if phase == 'gas':
            phase = 'vapor'
        if phase not in {'liquid', 'vapor'}:
            raise NotImplementedError('temperature-only steam PH requires a homogeneous phase')
        pressure = self._pressure_to_pa(P)
        target = self._molar_enthalpy_to_mass(float(H) - self._reference_H_offset)
        if not math.isfinite(target) or not math.isfinite(pressure) or pressure <= 0.0:
            raise ThermodynamicsError('steam PH requires finite inputs and positive pressure')
        # A PH temperature alone cannot describe a wet state. Check enthalpy
        # endpoints before using the inverse, including the forced phase.
        quality = 1.0 if phase == 'vapor' else 0.0
        saturated = False
        if P < self._pcrit_bar:
            boundary = self._props('H', 'P', pressure, 'Q', quality)
            saturated = abs(target - boundary) <= 1.0e-5
            if not saturated and ((phase == 'vapor' and target < boundary)
                                  or (phase == 'liquid' and target > boundary)):
                raise NotImplementedError('steam PH target is outside the requested homogeneous phase')
        temperature = self._props('T', 'P', pressure, 'H', target)
        if saturated:
            return temperature, self._mass_enthalpy_to_molar(boundary - target)

        def evaluate(T):
            return (
                self._mass_enthalpy_to_molar(self._props('H', 'T', T, 'P', pressure))
                + self._reference_H_offset,
                self._mass_entropy_to_molar(self._props('Cpmass', 'T', T, 'P', pressure)),
            )

        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..ph_solver import solve_caloric_temperature
        else:
            from ph_solver import solve_caloric_temperature
        temperature, residual, _evaluations, converged = solve_caloric_temperature(
            evaluate, (), float(H), temperature,
        )
        if not converged:
            raise ThermodynamicsError(f'steam PH residual {residual:.6g} kJ/kmol')
        return float(temperature), float(residual)

    def calculate_state_PH(self, P: float, H: float, F: float,
                           composition: dict[str, float],
                           include: Optional[Union[str, Iterable[str]]] = None) -> StreamState:
        return self._state_from_inputs(
            'P',
            self._pressure_to_pa(P),
            'H',
            self._molar_enthalpy_to_mass(H - self._reference_H_offset),
            F,
            composition,
            include=include,
        )

    def calculate_state_PS(self, P: float, S: float, F: float,
                           composition: dict[str, float],
                           include: Optional[Union[str, Iterable[str]]] = None) -> StreamState:
        return self._state_from_inputs(
            'P',
            self._pressure_to_pa(P),
            'S',
            self._molar_entropy_to_mass(S - self._reference_S_offset),
            F,
            composition,
            include=include,
        )

    def calculate_state_PQ(self, P: float, vapor_fraction: float, F: float,
                           composition: dict[str, float],
                           include: Optional[Union[str, Iterable[str]]] = None) -> StreamState:
        solid_state = self._permanent_solid_state_at_PQ(
            P, vapor_fraction, F, composition, include
        )
        if solid_state is not None:
            return solid_state
        return self._state_from_inputs(
            'P',
            self._pressure_to_pa(P),
            'Q',
            max(0.0, min(1.0, float(vapor_fraction))),
            F,
            composition,
            include=include,
            vapor_fraction=vapor_fraction,
        )

    def flash_TP(self, composition: dict[str, float], T: float, P: float) -> tuple[float, dict, dict]:
        composition = self._normalize_water_composition(composition)
        vapor_fraction, _quality = self._vapor_fraction_from_TP(T, P)
        if vapor_fraction <= 1e-12:
            return 0.0, dict(composition), dict(composition)
        if vapor_fraction >= 1.0 - 1e-12:
            return 1.0, dict(composition), dict(composition)
        return vapor_fraction, dict(composition), dict(composition)

    def Psat(self, comp: str, T: float) -> float:
        self._normalize_water_composition({comp: 1.0})
        return self._saturation_pressure(T)

    def bubble_point_T(self, composition: dict[str, float], P: float,
                       T_guess: float = 350.0) -> float:
        self._normalize_water_composition(composition)
        return self._saturation_temperature(P)

    def dew_point_T(self, composition: dict[str, float], P: float,
                    T_guess: float = 350.0) -> float:
        self._normalize_water_composition(composition)
        return self._saturation_temperature(P)

    def bubble_point_P(self, composition: dict[str, float], T: float) -> float:
        self._normalize_water_composition(composition)
        return self._saturation_pressure(T)

    def dew_point_P(self, composition: dict[str, float], T: float) -> float:
        self._normalize_water_composition(composition)
        return self._saturation_pressure(T)

    def K_values(self, T: float, P: float, composition: dict[str, float]) -> dict[str, float]:
        composition = self._normalize_water_composition(composition)
        return {self.water_component: self._saturation_pressure(T) / P}

    def K_value(self, comp: str, T: float, P: float,
                composition: Optional[dict[str, float]] = None) -> float:
        self._normalize_water_composition({comp: 1.0})
        return self._saturation_pressure(T) / P

    def mixture_Cp(self, composition: dict[str, float], T: float,
                   vapor_fraction: float = 1.0,
                   P: Optional[float] = None) -> Optional[float]:
        composition = self._normalize_water_composition(composition)
        if 0.0 < vapor_fraction < 1.0:
            return None
        phase = 'vapor' if vapor_fraction >= 0.5 else 'liquid'
        pressure = float(P) if P is not None and P > 0.0 else P_REF
        return self.calculate_state(
            T, pressure, 1.0, composition, phase=phase, flash=False, include=('Cp',)
        ).Cp

    def mixture_enthalpy(self, composition: dict[str, float], T: float,
                        vapor_fraction: float = 1.0,
                        x: Optional[dict] = None,
                        y: Optional[dict] = None,
                        P: float = P_REF) -> float:
        composition = self._normalize_water_composition(composition)
        if 0.0 < vapor_fraction < 1.0:
            return self.calculate_state_PQ(
                P, vapor_fraction, 1.0, composition, include=('H',)
            ).H
        phase = 'vapor' if vapor_fraction >= 0.5 else 'liquid'
        return self.calculate_state(
            T, P, 1.0, composition, phase=phase, flash=False, include=('H',)
        ).H

    def mixture_entropy(self, composition: dict[str, float], T: float,
                        vapor_fraction: float = 1.0,
                        x: Optional[dict] = None,
                        y: Optional[dict] = None,
                        P: float = P_REF) -> float:
        composition = self._normalize_water_composition(composition)
        if 0.0 < vapor_fraction < 1.0:
            return self.calculate_state_PQ(
                P, vapor_fraction, 1.0, composition, include=('S',)
            ).S
        phase = 'vapor' if vapor_fraction >= 0.5 else 'liquid'
        return self.calculate_state(
            T, P, 1.0, composition, phase=phase, flash=False, include=('S',)
        ).S

    def mixture_liquid_molar_volume(self, composition: dict[str, float], T: float) -> float:
        composition = self._normalize_water_composition(composition)
        density = self.calculate_state(
            T, P_REF, 1.0, composition, phase='liquid', flash=False, include=('rho',)
        ).rho
        return 1.0 / density

    def mixture_molar_density(self, composition: dict[str, float], T: float, P: float,
                              vapor_fraction: float = 1.0,
                              x: Optional[dict] = None,
                              y: Optional[dict] = None) -> float:
        composition = self._normalize_water_composition(composition)
        if 0.0 < vapor_fraction < 1.0:
            return self.calculate_state_PQ(
                P, vapor_fraction, 1.0, composition, include=('rho',)
            ).rho
        phase = 'vapor' if vapor_fraction >= 0.5 else 'liquid'
        return self.calculate_state(
            T, P, 1.0, composition, phase=phase, flash=False, include=('rho',)
        ).rho
