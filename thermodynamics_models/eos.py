import math
from typing import Iterable, Optional, Union

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..chemical_properties import ChemicalDatabase
else:
    from chemical_properties import ChemicalDatabase

from .common import (
    P_REF,
    T_REF,
    ThermodynamicsError,
    _solve_bubble_point_temperature,
    _solve_dew_point_temperature,
)
from .base import IdealThermodynamics, StreamState
from .henry import AqueousEquilibriumContext


def _phi_henry_aqueous_K_values(
    thermo,
    T: float,
    P: float,
    composition: dict[str, float],
    context: AqueousEquilibriumContext,
    cache_kind: str,
) -> dict[str, float]:
    """Hybrid phi-phi/phi-Henry K-values for a water-rich liquid."""
    thermo._warn_aqueous_henry_pressure(P, context)
    x = thermo._normalized_aqueous_composition(composition, thermo.components)
    cache_key = thermo._k_values_cache_key(cache_kind, T, P, x) + (context.cache_key(),)
    cached = thermo._get_cached_k_values(cache_key)
    if cached is not None:
        return cached

    phi_l = thermo.fugacity_coefficients(T, P, x, 'liquid')
    y = dict(x)
    K = {comp: 1.0 for comp in thermo.components}
    for _ in range(15):
        phi_v = thermo.fugacity_coefficients(T, P, y, 'vapor')
        K = {}
        for comp in thermo.components:
            vapor_phi = max(float(phi_v.get(comp, 1.0)), 1e-12)
            if comp in context.component_data:
                value = thermo._henry_ideal_vapor_K_value(comp, T, P, context) / vapor_phi
            else:
                value = max(float(phi_l.get(comp, 1.0)), 1e-12) / vapor_phi
            K[comp] = max(1e-12, min(1e12, value))
        y_new = {comp: x.get(comp, 0.0) * K[comp] for comp in thermo.components}
        y_total = sum(y_new.values())
        if y_total <= 0.0:
            break
        y_new = {comp: value / y_total for comp, value in y_new.items()}
        if max(abs(y_new[comp] - y.get(comp, 0.0)) for comp in thermo.components) < 1e-9:
            break
        y = y_new
    return thermo._set_cached_k_values(cache_key, K)

class _EOSCpDepartureMixin:
    """Real-gas Cp for EOS models: Cp = Cp_ideal + d(H_dep)/dT at fixed P/phase.

    The departure is a central finite difference of the departure enthalpy.
    Differentiating H_dep numerically covers every alpha variant without
    per-variant d2(alpha)/dT2 derivations and inherently smooths the
    second-derivative kink of switched alpha forms (e.g. Boston-Mathias
    at Tc) over the differencing window. Both phases use the EOS basis so
    Cp equals dH/dT of this model's own enthalpy; without a pressure the
    ideal/tabulated base-class path is used unchanged.
    """

    def _cp_departure(self, composition: dict[str, float], T: float,
                      P: float, phase: str) -> float:
        cache = getattr(self, '_cp_departure_cache', None)
        if cache is None:
            cache = self._cp_departure_cache = {}
        cache_key = (phase, float(T), float(P), self._composition_cache_key(composition))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        dT = max(0.05, 1e-4 * float(T))
        T_low = max(1.0, float(T) - dT)
        T_high = float(T) + dT
        try:
            h_low = self.departure_enthalpy(T_low, P, composition, phase)
            h_high = self.departure_enthalpy(T_high, P, composition, phase)
            value = (h_high - h_low) / (T_high - T_low)
        except Exception:
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        return self._set_limited_cache(cache, cache_key, value)

    def mixture_Cp(self, composition: dict[str, float], T: float,
                   vapor_fraction: float = 1.0,
                   P: Optional[float] = None) -> float:
        """Mixture Cp [kJ/kmol-K] consistent with the EOS enthalpy model."""
        if P is None or P <= 0.0:
            return super().mixture_Cp(composition, T, vapor_fraction, P)
        V = max(0.0, min(1.0, float(vapor_fraction)))
        cp_ideal = sum(
            z * self.Cp_ideal_gas(comp, T)
            for comp, z in composition.items()
        )  # J/mol-K is numerically kJ/kmol-K
        if V > 0.999:
            return cp_ideal + self._cp_departure(composition, T, float(P), 'vapor')
        if V < 0.001:
            return cp_ideal + self._cp_departure(composition, T, float(P), 'liquid')
        cp_vapor = cp_ideal + self._cp_departure(composition, T, float(P), 'vapor')
        cp_liquid = cp_ideal + self._cp_departure(composition, T, float(P), 'liquid')
        return V * cp_vapor + (1.0 - V) * cp_liquid


class RKThermodynamics(_EOSCpDepartureMixin, IdealThermodynamics):
    """
    Redlich-Kwong thermodynamic property calculator.
    
    Extends IdealThermodynamics with:
    - RK equation of state for compressibility
    - Fugacity coefficients for VLE
    - Phi-phi K-values for VLE
    - Departure functions for enthalpy/entropy
    """
    
    def __init__(
        self,
        components: list[str],
        db: Optional[ChemicalDatabase] = None,
        interaction_overrides: Optional[list[dict]] = None,
    ):
        """
        Initialize with list of component symbols.
        
        Args:
            components: List of chemical symbols
            db: Chemical database (uses default if None)
        """
        super().__init__(components, db, interaction_overrides)
        
        # Import RK module
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..rk_eos import RedlichKwong, RKError
        else:
            from rk_eos import RedlichKwong, RKError
        
        # Check that all components have critical properties
        missing_tc = []
        for comp in components:
            props = self.props[comp]
            if props.Tc is None or props.Pc is None:
                missing_tc.append(comp)
        
        if missing_tc:
            raise ThermodynamicsError(
                f"RK method requires critical properties (Tc, Pc) for all components. "
                f"Missing for: {', '.join(missing_tc)}"
            )
        
        # Create RK calculator
        try:
            self.rk = RedlichKwong(components, db)
        except RKError as e:
            raise ThermodynamicsError(f"Failed to initialize RK: {e}")
        for comp in components:
            self.mark_property_source_context(
                comp,
                ('Tc', 'Pc'),
                kind='thermo_model',
                phase='rk_eos_parameters',
                description='Redlich-Kwong EOS critical-property parameters',
                affects_result=True,
            )
        self.extend_warnings(getattr(self.rk, 'warnings', []))
    
    def compressibility_factor(self, T: float, P: float, 
                                composition: dict[str, float],
                                phase: str = 'vapor') -> float:
        """
        Calculate compressibility factor Z using RK.
        
        Args:
            T: Temperature [K]
            P: Pressure [bar]
            composition: Mole fractions
            phase: 'vapor' or 'liquid'
            
        Returns:
            Z value
        """
        return self.rk.compressibility_factor(T, P, composition, phase)
    
    def fugacity_coefficients(self, T: float, P: float,
                               composition: dict[str, float],
                               phase: str = 'vapor') -> dict[str, float]:
        """
        Calculate fugacity coefficients using RK.
        
        Args:
            T: Temperature [K]
            P: Pressure [bar]
            composition: Mole fractions
            phase: 'vapor' or 'liquid'
            
        Returns:
            Dictionary of fugacity coefficients
        """
        return self.rk.fugacity_coefficients(T, P, composition, phase)

    def molar_volume(self, T: float, P: float,
                     composition: dict[str, float],
                     phase: str = 'vapor') -> float:
        """Molar volume using RK [m3/kmol]."""
        return self.rk.molar_volume(T, P, composition, phase) / 1000.0
    
    def K_values(self, T: float, P: float,
                 composition: dict[str, float]) -> dict[str, float]:
        return self.rk.phi_phi_K_values(T, P, composition)

    def aqueous_K_values(self, T: float, P: float,
                         composition: dict[str, float],
                         context: AqueousEquilibriumContext) -> dict[str, float]:
        return _phi_henry_aqueous_K_values(
            self, T, P, composition, context, 'aqueous_rk_phi_henry'
        )

    def K_value(self, comp: str, T: float, P: float,
                x: Optional[dict[str, float]] = None) -> float:
        composition = x if x is not None else {comp: 1.0}
        return self.K_values(T, P, composition).get(comp, 1.0)

    def flash_TP(self, composition: dict[str, float], T: float, P: float) -> tuple[float, dict, dict]:
        return self._iterative_K_flash_TP(composition, T, P)

    def bubble_point_T(self, composition: dict[str, float], P: float,
                       T_guess: float = 350.0) -> float:
        return _solve_bubble_point_temperature(self, composition, P, T_guess)

    def dew_point_T(self, composition: dict[str, float], P: float,
                    T_guess: float = 350.0) -> float:
        return _solve_dew_point_temperature(self, composition, P, T_guess)
    
    def molar_volume_vapor(self, T: float, P: float, 
                           composition: dict[str, float]) -> float:
        """
        Vapor molar volume using RK [m3/kmol]
        """
        return self.molar_volume(T, P, composition, 'vapor')

    def vapor_molar_volume_for_density(self, T: float, P: float,
                                       composition: dict[str, float]) -> float:
        """Vapor molar volume from RK for stream density [m3/kmol]."""
        return self.molar_volume_vapor(T, P, composition)
    
    def molar_volume_liquid(self, T: float, P: float,
                            composition: dict[str, float]) -> float:
        """
        Liquid molar volume [m3/kmol] using the shared resolver-backed path.
        """
        return self.mixture_liquid_molar_volume(composition, T)
    
    def departure_enthalpy(self, T: float, P: float,
                           composition: dict[str, float],
                           phase: str = 'vapor') -> float:
        """
        Departure enthalpy (H - H_ig) [kJ/kmol]
        """
        # J/mol is numerically equal to kJ/kmol.
        return self.rk.departure_enthalpy(T, P, composition, phase)

    def departure_entropy(self, T: float, P: float,
                          composition: dict[str, float],
                          phase: str = 'vapor') -> float:
        """Departure entropy (S - S_ig) [kJ/kmol-K]."""
        S_dep = self.rk.departure_entropy(T, P, composition, phase)
        return S_dep

    def departure_gibbs(self, T: float, P: float,
                        composition: dict[str, float],
                        phase: str = 'vapor') -> float:
        """Departure Gibbs energy (G - G_ig) [kJ/kmol]."""
        return self.departure_enthalpy(T, P, composition, phase) - T * self.departure_entropy(
            T, P, composition, phase
        )

    def residual_enthalpy(self, T: float, P: float,
                          composition: dict[str, float],
                          phase: str = 'vapor') -> float:
        """Alias for RK departure enthalpy [kJ/kmol]."""
        return self.departure_enthalpy(T, P, composition, phase)
    
    def mixture_enthalpy(self, composition: dict[str, float], T: float,
                        vapor_fraction: float = 1.0,
                        x: Optional[dict] = None,
                        y: Optional[dict] = None,
                        P: float = P_REF) -> float:
        """
        Mixture molar enthalpy including departure [kJ/kmol]
        """
        def ideal_gas_mixture(comp: dict[str, float]) -> float:
            return 1000.0 * sum(
                zi * self.enthalpy_ideal_gas(component, T)
                for component, zi in comp.items()
            )
        
        if vapor_fraction > 0.999:
            return ideal_gas_mixture(composition) + self.departure_enthalpy(T, P, composition, 'vapor')
        elif vapor_fraction < 0.001:
            return ideal_gas_mixture(composition) + self.departure_enthalpy(T, P, composition, 'liquid')
        else:
            x = x or composition
            y = y or composition
            H_l = ideal_gas_mixture(x) + self.departure_enthalpy(T, P, x, 'liquid')
            H_v = ideal_gas_mixture(y) + self.departure_enthalpy(T, P, y, 'vapor')
            return vapor_fraction * H_v + (1 - vapor_fraction) * H_l

    def mixture_entropy(self, composition: dict[str, float], T: float,
                        vapor_fraction: float = 1.0,
                        x: Optional[dict] = None,
                        y: Optional[dict] = None,
                        P: float = P_REF) -> float:
        """Mixture molar entropy including RK departure [kJ/kmol-K]."""
        def ideal_gas_mixture(comp: dict[str, float]) -> float:
            return self._ideal_gas_mixture_entropy(comp, T, P)

        if vapor_fraction > 0.999:
            return ideal_gas_mixture(composition) + self.departure_entropy(T, P, composition, 'vapor')
        if vapor_fraction < 0.001:
            return ideal_gas_mixture(composition) + self.departure_entropy(T, P, composition, 'liquid')
        x = x or composition
        y = y or composition
        S_l = ideal_gas_mixture(x) + self.departure_entropy(T, P, x, 'liquid')
        S_v = ideal_gas_mixture(y) + self.departure_entropy(T, P, y, 'vapor')
        return vapor_fraction * S_v + (1.0 - vapor_fraction) * S_l
    
    def calculate_state(self, T: float, P: float, F: float,
                       composition: dict[str, float],
                       phase: Optional[str] = None,
                       flash: bool = True,
                       include: Optional[Union[str, Iterable[str]]] = None) -> StreamState:
        """
        Calculate complete stream state using RK.
        """
        state = super().calculate_state(
            T, P, F, composition, phase, flash,
            include=include,
        )
        return state


class CubicEOSThermodynamics(_EOSCpDepartureMixin, IdealThermodynamics):
    """Phi-phi VLE using a cubic EOS for both liquid and vapor fugacity."""

    def __init__(
        self,
        components: list[str],
        model: str,
        db: Optional[ChemicalDatabase] = None,
        interaction_overrides: Optional[list[dict]] = None,
        unifac_groups: Optional[dict] = None,
    ):
        super().__init__(components, db, interaction_overrides)
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..cubic_eos import CubicEOS, CubicEOSError
        else:
            from cubic_eos import CubicEOS, CubicEOSError

        try:
            self.cubic = CubicEOS(components, model, db, interaction_overrides, unifac_groups)
        except CubicEOSError as e:
            raise ThermodynamicsError(str(e)) from e
        self.model = self.cubic.model
        for comp in components:
            self.mark_property_source_context(
                comp,
                ('Tc', 'Pc', 'omega'),
                kind='thermo_model',
                phase=f'{str(self.model).lower()}_eos_parameters',
                description='Cubic EOS critical-property parameters',
                affects_result=True,
            )
        self.extend_warnings(getattr(self.cubic, 'warnings', []))

    def _compiled_pressure_enthalpy_backend(self):
        state = getattr(self, '_compiled_pressure_enthalpy_state', None)
        cubic_backend = self.cubic._compiled_backend
        if state is not None and state[0] is cubic_backend:
            return state[1]
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..compiled_ph import CompiledPHBackend
            else:
                from compiled_ph import CompiledPHBackend

            backend = (
                CompiledPHBackend.from_thermo(self, cubic_backend=cubic_backend)
                if cubic_backend is not None else None
            )
        except Exception:
            backend = None
        self._compiled_pressure_enthalpy_state = cubic_backend, backend
        return backend

    def calculate_state_PH(
        self,
        P: float,
        H: float,
        F: float,
        composition: dict[str, float],
        include: Optional[Union[str, Iterable[str]]] = None,
        *,
        phase: Optional[str] = None,
        T_guess: Optional[float] = None,
    ) -> StreamState:
        """Direct homogeneous PH state using compact compiled EOS properties."""
        temperature, _residual = self.temperature_at_PH(
            P,
            H,
            composition,
            phase=phase,
            T_guess=T_guess,
        )
        normalized = self.cubic._normalized_composition(composition)
        phase_name = str(phase).strip().lower()
        if phase_name == 'gas':
            phase_name = 'vapor'
        return self.calculate_state(
            temperature,
            P,
            F,
            normalized,
            phase=phase_name,
            flash=False,
            include=include,
        )

    def temperature_at_PH(
        self,
        P: float,
        H: float,
        composition: dict[str, float],
        *,
        phase: Optional[str] = None,
        T_guess: Optional[float] = None,
    ) -> tuple[float, float]:
        """Return homogeneous PH temperature and enthalpy residual."""
        phase_name = str(phase or '').strip().lower()
        if phase_name == 'gas':
            phase_name = 'vapor'
        if phase_name not in {'liquid', 'vapor'}:
            raise NotImplementedError(
                'direct cubic-EOS PH requires an explicit homogeneous phase'
            )
        backend = self._compiled_pressure_enthalpy_backend()
        if backend is None:
            raise NotImplementedError(
                'direct cubic-EOS PH requires compatible compiled Cp and EOS backends'
            )
        normalized = self.cubic._normalized_composition(composition)
        values = [normalized[component] for component in self.components]
        seed = float(T_guess) if T_guess is not None else T_REF
        temperature, residual, _evaluations, converged = (
            backend.solve_temperature(P, H, values, seed, phase_name)
        )
        if not converged:
            raise ThermodynamicsError(
                'compiled cubic-EOS PH temperature solve did not converge; '
                f'last residual {residual:.6g} kJ/kmol'
            )
        return float(temperature), float(residual)

    def compressibility_factor(self, T: float, P: float,
                                composition: dict[str, float],
                                phase: str = 'vapor') -> float:
        value = self.cubic.compressibility_factor(T, P, composition, phase)
        self.extend_warnings(self.cubic.warnings)
        return value

    def fugacity_coefficients(self, T: float, P: float,
                               composition: dict[str, float],
                               phase: str = 'vapor') -> dict[str, float]:
        values = self.cubic.fugacity_coefficients(T, P, composition, phase)
        self.extend_warnings(self.cubic.warnings)
        return values

    def molar_volume(self, T: float, P: float,
                     composition: dict[str, float],
                     phase: str = 'vapor') -> float:
        """Molar volume [m3/kmol]."""
        value = self.cubic.molar_volume(T, P, composition, phase) / 1000.0
        self.extend_warnings(self.cubic.warnings)
        return value

    def molar_volume_vapor(self, T: float, P: float,
                           composition: dict[str, float]) -> float:
        """Vapor molar volume [m3/kmol]."""
        return self.molar_volume(T, P, composition, 'vapor')

    def vapor_molar_volume_for_density(self, T: float, P: float,
                                       composition: dict[str, float]) -> float:
        """Vapor molar volume from cubic EOS for stream density [m3/kmol]."""
        return self.molar_volume_vapor(T, P, composition)

    def molar_volume_liquid(self, T: float, P: float,
                            composition: dict[str, float]) -> float:
        """Liquid molar volume [m3/kmol] using the shared resolver-backed path."""
        return self.mixture_liquid_molar_volume(composition, T)

    def departure_enthalpy(self, T: float, P: float,
                           composition: dict[str, float],
                           phase: str = 'vapor') -> float:
        """EOS residual/departure enthalpy [kJ/kmol]."""
        value = self.cubic.departure_enthalpy(T, P, composition, phase)
        self.extend_warnings(self.cubic.warnings)
        return value

    def departure_entropy(self, T: float, P: float,
                          composition: dict[str, float],
                          phase: str = 'vapor') -> float:
        """EOS residual/departure entropy [kJ/kmol-K]."""
        value = self.cubic.departure_entropy(T, P, composition, phase)
        self.extend_warnings(self.cubic.warnings)
        return value

    def departure_gibbs(self, T: float, P: float,
                        composition: dict[str, float],
                        phase: str = 'vapor') -> float:
        """EOS residual/departure Gibbs energy [kJ/kmol]."""
        value = self.cubic.departure_gibbs(T, P, composition, phase)
        self.extend_warnings(self.cubic.warnings)
        return value

    def residual_enthalpy(self, T: float, P: float,
                          composition: dict[str, float],
                          phase: str = 'vapor') -> float:
        """Alias for EOS departure enthalpy [kJ/kmol]."""
        return self.departure_enthalpy(T, P, composition, phase)

    def mixture_enthalpy(self, composition: dict[str, float], T: float,
                        vapor_fraction: float = 1.0,
                        x: Optional[dict] = None,
                        y: Optional[dict] = None,
                        P: float = P_REF) -> float:
        """Mixture molar enthalpy with cubic EOS departure correction [kJ/kmol]."""
        def ideal_gas_mixture(comp: dict[str, float]) -> float:
            return 1000.0 * sum(
                zi * self.enthalpy_ideal_gas(component, T)
                for component, zi in comp.items()
            )

        if vapor_fraction > 0.999:
            return ideal_gas_mixture(composition) + self.departure_enthalpy(T, P, composition, 'vapor')
        if vapor_fraction < 0.001:
            return ideal_gas_mixture(composition) + self.departure_enthalpy(T, P, composition, 'liquid')
        x = x or composition
        y = y or composition
        H_l = ideal_gas_mixture(x) + self.departure_enthalpy(T, P, x, 'liquid')
        H_v = ideal_gas_mixture(y) + self.departure_enthalpy(T, P, y, 'vapor')
        return vapor_fraction * H_v + (1.0 - vapor_fraction) * H_l

    def mixture_entropy(self, composition: dict[str, float], T: float,
                        vapor_fraction: float = 1.0,
                        x: Optional[dict] = None,
                        y: Optional[dict] = None,
                        P: float = P_REF) -> float:
        """Mixture molar entropy with cubic EOS departure correction [kJ/kmol-K]."""
        def ideal_gas_mixture(comp: dict[str, float]) -> float:
            return self._ideal_gas_mixture_entropy(comp, T, P)

        if vapor_fraction > 0.999:
            return ideal_gas_mixture(composition) + self.departure_entropy(T, P, composition, 'vapor')
        if vapor_fraction < 0.001:
            return ideal_gas_mixture(composition) + self.departure_entropy(T, P, composition, 'liquid')
        x = x or composition
        y = y or composition
        S_l = ideal_gas_mixture(x) + self.departure_entropy(T, P, x, 'liquid')
        S_v = ideal_gas_mixture(y) + self.departure_entropy(T, P, y, 'vapor')
        return vapor_fraction * S_v + (1.0 - vapor_fraction) * S_l

    def K_values(self, T: float, P: float,
                 composition: dict[str, float]) -> dict[str, float]:
        values = self.cubic.phi_phi_K_values(T, P, composition)
        self.extend_warnings(self.cubic.warnings)
        return values

    def aqueous_K_values(self, T: float, P: float,
                         composition: dict[str, float],
                         context: AqueousEquilibriumContext) -> dict[str, float]:
        return _phi_henry_aqueous_K_values(
            self, T, P, composition, context, f'aqueous_{self.model}_phi_henry'
        )

    def K_value(self, comp: str, T: float, P: float,
                x: Optional[dict[str, float]] = None) -> float:
        composition = x if x is not None else {comp: 1.0}
        return self.K_values(T, P, composition).get(comp, 1.0)

    def flash_TP(self, composition: dict[str, float], T: float, P: float) -> tuple[float, dict, dict]:
        return self._iterative_K_flash_TP(composition, T, P)

    def bubble_point_T(self, composition: dict[str, float], P: float,
                       T_guess: float = 350.0) -> float:
        return _solve_bubble_point_temperature(self, composition, P, T_guess)

    def dew_point_T(self, composition: dict[str, float], P: float,
                    T_guess: float = 350.0) -> float:
        return _solve_dew_point_temperature(self, composition, P, T_guess)

    def calculate_state(self, T: float, P: float, F: float,
                       composition: dict[str, float],
                       phase: Optional[str] = None,
                       flash: bool = True,
                       include: Optional[Union[str, Iterable[str]]] = None) -> StreamState:
        return super().calculate_state(
            T, P, F, composition, phase, flash,
            include=include,
        )
