import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import brentq, least_squares

from .common import (
    P_REF,
    R,
    R_BAR,
    ThermodynamicsError,
    _solve_bubble_point_temperature,
    _solve_dew_point_temperature,
)
from .base import FluidPhaseEquilibrium, IdealThermodynamics, StreamState
from .henry import AqueousEquilibriumContext


@dataclass
class VLLEFlashResult:
    """Reference vapor-liquid-liquid TP flash result."""

    phase_count: int
    status: str
    vapor_fraction: float
    liquid1_fraction: float
    liquid2_fraction: float
    y: dict[str, float]
    x1: dict[str, float]
    x2: dict[str, float]
    residual: float
    iterations: int
    extra: dict = field(default_factory=dict)


class ActivityCoefficientThermodynamics(IdealThermodynamics):
    """Shared VLE/LLE machinery for liquid activity-coefficient models."""

    def __init__(
        self,
        components: list[str],
        db=None,
        interaction_overrides: Optional[list[dict]] = None,
    ):
        super().__init__(components, db, interaction_overrides)
        self._activity_interaction_extrapolation_warning_keys: set[tuple] = set()

    @staticmethod
    def _activity_interaction_temperature_bounds(record: dict) -> tuple[float, float]:
        policy = record.get('do_not_extrapolate', False)
        if not isinstance(policy, bool):
            raise ThermodynamicsError(
                "Activity interaction do_not_extrapolate must be boolean"
            )
        if not policy:
            return -math.inf, math.inf
        if record.get('Tmin_K') is None or record.get('Tmax_K') is None:
            raise ThermodynamicsError(
                "Activity interaction do_not_extrapolate=true requires "
                "Tmin_K and Tmax_K"
            )
        try:
            low = float(record['Tmin_K'])
            high = float(record['Tmax_K'])
        except (TypeError, ValueError) as error:
            raise ThermodynamicsError(
                "Activity interaction Tmin_K and Tmax_K must be numeric"
            ) from error
        if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high <= low:
            raise ThermodynamicsError(
                "Activity interaction do_not_extrapolate=true requires finite "
                "0 < Tmin_K < Tmax_K"
            )
        return low, high

    def _activity_interaction_temperature(
        self,
        record: dict,
        component1: str,
        component2: str,
        T: float,
    ) -> float:
        low, high = self._activity_interaction_temperature_bounds(record)
        requested = float(T)
        effective = min(max(requested, low), high)
        if effective == requested:
            return requested
        boundary = 'Tmin_K' if requested < low else 'Tmax_K'
        key = (tuple(sorted((component1, component2))), boundary)
        if key not in self._activity_interaction_extrapolation_warning_keys:
            self._activity_interaction_extrapolation_warning_keys.add(key)
            self.add_warning(
                f"Activity interaction parameters for {component1}/{component2} "
                f"were evaluated at {effective:g} K instead of {requested:g} K: "
                f"do_not_extrapolate=true and the declared range is "
                f"{low:g}-{high:g} K."
            )
        return effective

    def _warn_activity_interaction_extrapolation(
        self,
        T: float,
        components=None,
    ) -> None:
        """Allow record-backed models to surface compiled-path clamp warnings."""

    def _validate_activity_interactions(self, resolver) -> None:
        for index, component1 in enumerate(self.components):
            for component2 in self.components[index + 1:]:
                record = resolver(component1, component2)
                if record is not None:
                    self._activity_interaction_temperature_bounds(record)

    def initialize(self) -> 'ActivityCoefficientThermodynamics':
        """Prepare persistent providers and the selected activity kernel."""
        super().initialize()
        if getattr(self, '_activity_runtime_initialized', False):
            return self
        try:
            compiled = getattr(self, '_compiled_unifac', None)
            backend_factory = getattr(self, '_compiled_activity_backend', None)
            if compiled is None and callable(backend_factory):
                compiled = backend_factory()
            compile_kernels = getattr(compiled, 'compile_kernels', None)
            if callable(compile_kernels):
                compile_kernels()
        except ThermodynamicsError:
            raise
        except Exception as error:
            raise ThermodynamicsError(
                f"Failed to initialize {type(self).__name__} activity "
                f"coefficient kernel: {error}"
            ) from error
        self._activity_runtime_initialized = True
        return self

    @property
    def vapor_eos(self):
        """Vapor-fugacity EOS backend for gamma-phi variants (None when unused)."""
        return getattr(self, '_vapor_eos_backend', None)

    @vapor_eos.setter
    def vapor_eos(self, backend) -> None:
        self._vapor_eos_backend = backend

    @property
    def rk(self):
        """Legacy alias for the vapor backend (may be an RK or PR model)."""
        return self.vapor_eos

    @rk.setter
    def rk(self, backend) -> None:
        self.vapor_eos = backend

    def vapor_fugacity_coefficients(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
    ) -> dict[str, float]:
        """Effective nominal-component vapor fugacity coefficients."""
        backend = self.vapor_eos
        if backend is None:
            return {comp: 1.0 for comp in self.components}
        return backend.fugacity_coefficients(T, P, composition, 'vapor')

    def _liquid_fugacity_reference_factors(
        self,
        T: float,
        P: float,
    ) -> dict[str, float]:
        """Pure-liquid reference fugacities [bar] used by this model's K-values."""
        if self.vapor_eos is not None:
            return self._gamma_phi_reference_factors(T, P)
        return {component: self.Psat(component, T) for component in self.components}

    def compiled_vlle_backend(self):
        """Return the optional compiled ideal-vapor VLLE backend, built lazily."""
        if getattr(self, '_compiled_vlle_initialized', False):
            return getattr(self, '_compiled_vlle', None)
        self._compiled_vlle_initialized = True
        self._compiled_vlle = None
        if hasattr(self, '_initialize_gamma_phi_backend') or hasattr(self, '_initialize_vdm'):
            return None
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..compiled_vlle import CompiledActivityVLLEBackend
            else:
                from compiled_vlle import CompiledActivityVLLEBackend
            self._compiled_vlle = CompiledActivityVLLEBackend.from_thermo(self)
        except Exception:
            self._compiled_vlle = None
        return self._compiled_vlle

    def _compiled_vlle_result_to_reference(self, result) -> VLLEFlashResult:
        return VLLEFlashResult(
            phase_count=result.phase_count,
            status=result.status,
            vapor_fraction=result.vapor_fraction,
            liquid1_fraction=result.liquid1_fraction,
            liquid2_fraction=result.liquid2_fraction,
            y={comp: float(result.y[index]) for index, comp in enumerate(self.components)},
            x1={comp: float(result.x1[index]) for index, comp in enumerate(self.components)},
            x2={comp: float(result.x2[index]) for index, comp in enumerate(self.components)},
            residual=result.residual,
            iterations=result.iterations,
        )

    def liquid_spinodal_stability(
        self,
        T: float,
        composition: dict[str, float],
        *,
        mole_fraction_floor: float = 1.0e-8,
        eigenvalue_tolerance: float = 1.0e-6,
    ) -> dict:
        """Cheap local liquid-stability test from the reduced Gibbs Hessian.

        This deliberately detects only spinodal instability. A nonnegative
        Hessian is not a global tangent-plane stability proof.
        """
        normalized = self._normalize_phase_composition(composition)
        active = [
            component for component in self.components
            if normalized.get(component, 0.0) > mole_fraction_floor
        ]
        if len(active) < 2:
            return {
                'locally_stable': True,
                'minimum_eigenvalue': math.inf,
                'active_components': tuple(active),
                'reason': 'fewer_than_two_active_liquid_components',
            }
        active_total = sum(normalized[component] for component in active)
        base = [normalized[component] / active_total for component in active]
        independent = list(base[:-1])
        minimum_fraction = min(base)
        step = min(1.0e-4, 0.08 * minimum_fraction)
        step = max(step, 1.0e-7)
        if minimum_fraction <= 2.5 * step:
            step = max(minimum_fraction / 4.0, 1.0e-10)

        compiled_activity = getattr(self, '_compiled_unifac', None)
        if compiled_activity is None:
            backend_factory = getattr(self, '_compiled_activity_backend', None)
            if callable(backend_factory):
                try:
                    compiled_activity = backend_factory(float(T))
                except Exception:
                    compiled_activity = None
        compiled_components = tuple(
            getattr(compiled_activity, 'components', ()) or ()
        )
        compiled_index = {
            component: index
            for index, component in enumerate(compiled_components)
        }
        if not all(component in compiled_index for component in active):
            compiled_activity = None

        def active_gammas(values):
            if compiled_activity is not None:
                full = [0.0] * len(compiled_components)
                for component, value in zip(active, values):
                    full[compiled_index[component]] = float(value)
                calculated = compiled_activity.activity_coefficients(full, float(T))
                return [
                    max(float(calculated[compiled_index[component]]), 1.0e-300)
                    for component in active
                ]
            trial = {component: 0.0 for component in self.components}
            for component, value in zip(active, values):
                trial[component] = float(value)
            gamma = self.activity_coefficients(float(T), trial)
            return [
                max(float(gamma.get(component, 1.0)), 1.0e-300)
                for component in active
            ]

        def reduced_gradient(coordinates):
            values = list(coordinates)
            values.append(1.0 - sum(coordinates))
            if any(value <= 0.0 for value in values):
                raise ThermodynamicsError(
                    "Spinodal finite difference left the composition simplex"
                )
            gamma_values = active_gammas(values)
            chemical_potentials = [
                math.log(float(value)) + math.log(gamma_value)
                for value, gamma_value in zip(values, gamma_values)
            ]
            reference = chemical_potentials[-1]
            return [value - reference for value in chemical_potentials[:-1]]

        dimension = len(independent)
        hessian = [[0.0] * dimension for _ in range(dimension)]
        for column in range(dimension):
            plus = list(independent)
            minus = list(independent)
            plus[column] += step
            minus[column] -= step
            gradient_plus = reduced_gradient(plus)
            gradient_minus = reduced_gradient(minus)
            for row in range(dimension):
                hessian[row][column] = (
                    gradient_plus[row] - gradient_minus[row]
                ) / (2.0 * step)

        # Thermodynamically consistent activity models produce a symmetric
        # Hessian; averaging suppresses harmless finite-difference asymmetry.
        for row in range(dimension):
            for column in range(row):
                value = 0.5 * (hessian[row][column] + hessian[column][row])
                hessian[row][column] = value
                hessian[column][row] = value

        if dimension == 1:
            minimum = float(hessian[0][0])
            direction = (1.0,)
        elif dimension == 2:
            a = float(hessian[0][0])
            b = float(hessian[0][1])
            c = float(hessian[1][1])
            discriminant = math.sqrt(max(0.0, (a - c) ** 2 + 4.0 * b * b))
            minimum = 0.5 * (a + c - discriminant)
            if abs(b) > 1.0e-30:
                vector = (b, minimum - a)
            elif a <= c:
                vector = (1.0, 0.0)
            else:
                vector = (0.0, 1.0)
            norm = math.hypot(*vector)
            direction = tuple(value / max(norm, 1.0e-300) for value in vector)
        else:
            import numpy as np

            eigenvalues, eigenvectors = np.linalg.eigh(
                np.asarray(hessian, dtype=float)
            )
            minimum_index = int(np.argmin(eigenvalues))
            minimum = float(eigenvalues[minimum_index])
            direction = tuple(
                float(value) for value in eigenvectors[:, minimum_index]
            )
        return {
            'locally_stable': minimum >= -abs(float(eigenvalue_tolerance)),
            'minimum_eigenvalue': minimum,
            'active_components': tuple(active),
            'unstable_direction': direction,
            'finite_difference_step': float(step),
            'activity_backend': (
                type(compiled_activity).__name__
                if compiled_activity is not None else 'python_dictionary'
            ),
            'reason': 'reduced_gibbs_hessian',
        }

    def _robust_vlle_result(
        self,
        composition: dict[str, float],
        T: float,
        P: float,
    ) -> VLLEFlashResult:
        """Use the compiled global-search splitter when eligible, then reference."""
        def admissible(result: VLLEFlashResult) -> bool:
            fractions = (
                result.vapor_fraction,
                result.liquid1_fraction,
                result.liquid2_fraction,
            )
            return (
                result.status != 'not_converged'
                and all(math.isfinite(float(value)) and float(value) >= -1.0e-10
                        for value in fractions)
                and abs(sum(float(value) for value in fractions) - 1.0) <= 1.0e-7
                and math.isfinite(float(result.residual))
            )

        backend = self.compiled_vlle_backend() if len(self.components) > 2 else None
        if backend is not None:
            try:
                compiled = backend.flash_TP(composition, T, P, max_iter=200)
                result = self._compiled_vlle_result_to_reference(compiled)
                if admissible(result):
                    return result
            except Exception:
                pass
        result = self.flash3_TP(composition, T, P, max_iter=200)
        if not admissible(result):
            raise ThermodynamicsError(
                "VLLE phase search did not return an admissible equilibrium state"
            )
        return result

    @staticmethod
    def _fluid_result_from_vlle(
        result: VLLEFlashResult,
        *,
        stability: str,
        extra: Optional[dict] = None,
    ) -> FluidPhaseEquilibrium:
        details = dict(result.extra or {})
        details.update(extra or {})
        return FluidPhaseEquilibrium(
            vapor_fraction=float(result.vapor_fraction),
            liquid1_fraction=float(result.liquid1_fraction),
            liquid2_fraction=float(result.liquid2_fraction),
            y=dict(result.y),
            x1=dict(result.x1),
            x2=dict(result.x2),
            status=str(result.status),
            stability=stability,
            extra=details,
        )

    def _adaptive_vle_candidate(
        self,
        composition: dict[str, float],
        T: float,
        P: float,
    ) -> FluidPhaseEquilibrium:
        """Fast constrained VLE seed used only by local-spinodal mode."""
        backend = self.compiled_vlle_backend() if len(self.components) > 2 else None
        compiled_vle = getattr(backend, 'flash_VLE_TP', None)
        if callable(compiled_vle):
            try:
                V, x_values, y_values = compiled_vle(
                    composition,
                    T,
                    P,
                    max_iter=100,
                )
                V = max(0.0, min(1.0, float(V)))
                x = {
                    component: float(x_values[index])
                    for index, component in enumerate(self.components)
                }
                y = {
                    component: float(y_values[index])
                    for index, component in enumerate(self.components)
                }
                return FluidPhaseEquilibrium(
                    vapor_fraction=V,
                    liquid1_fraction=1.0 - V,
                    liquid2_fraction=0.0,
                    y=y,
                    x1=x,
                    x2={},
                    status=(
                        'single_vapor' if V >= 1.0 - 1.0e-10
                        else 'single_liquid' if V <= 1.0e-10
                        else 'ordinary_vle'
                    ),
                    stability='vle_candidate_for_spinodal',
                    extra={'vle_backend': type(backend).__name__},
                )
            except Exception:
                pass
        return super()._fluid_phase_equilibrium_TP(composition, T, P)

    def _fluid_phase_equilibrium_TP(
        self,
        composition: dict[str, float],
        T: float,
        P: float,
    ) -> FluidPhaseEquilibrium:
        mode = self.fluid_phase_model
        if mode == 'VLE':
            return super()._fluid_phase_equilibrium_TP(composition, T, P)
        if mode == 'VLLE':
            result = self._robust_vlle_result(composition, T, P)
            return self._fluid_result_from_vlle(
                result,
                stability='global_vlle_search',
            )

        ordinary = self._adaptive_vle_candidate(composition, T, P)
        if ordinary.liquid1_fraction <= 1.0e-10:
            return FluidPhaseEquilibrium(
                **{
                    **ordinary.__dict__,
                    'stability': 'spinodal_not_applicable_no_liquid',
                }
            )
        spinodal = self.liquid_spinodal_stability(T, ordinary.x1)
        if spinodal['locally_stable']:
            return FluidPhaseEquilibrium(
                **{
                    **ordinary.__dict__,
                    'stability': 'locally_stable_spinodal_only',
                    'extra': {'spinodal': spinodal},
                }
            )
        result = self._robust_vlle_result(composition, T, P)
        return self._fluid_result_from_vlle(
            result,
            stability='spinodal_unstable_vlle_selected',
            extra={'spinodal': spinodal},
        )

    def calculate_state_PQ(
        self,
        P: float,
        vapor_fraction: float,
        F: float,
        composition: dict[str, float],
        include=None,
    ):
        """Pressure/quality state honoring the selected universal fluid policy."""
        solid_state = self._permanent_solid_state_at_PQ(
            P,
            vapor_fraction,
            F,
            composition,
            include,
        )
        if solid_state is not None:
            return solid_state
        if self.fluid_phase_model == 'VLE':
            return super().calculate_state_PQ(
                P,
                vapor_fraction,
                F,
                composition,
                include=include,
            )

        target = max(0.0, min(1.0, float(vapor_fraction)))
        stability = 'global_vlle_search'
        spinodal = None
        if self.fluid_phase_model == 'VL(L)E':
            ordinary = super().calculate_state_PQ(
                P,
                target,
                F,
                composition,
                include=(),
            )
            def finish_ordinary(stability_label: str, details=None):
                return self._apply_fluid_equilibrium_to_state(
                    ordinary,
                    FluidPhaseEquilibrium(
                        vapor_fraction=ordinary.vapor_fraction,
                        liquid1_fraction=ordinary.effective_liquid1_fraction,
                        liquid2_fraction=0.0,
                        y=dict(ordinary.y or ordinary.composition),
                        x1=dict(ordinary.x1 or ordinary.x or ordinary.composition),
                        x2={},
                        status=ordinary.phase_status,
                        stability=stability_label,
                        extra=dict(details or {}),
                    ),
                    self._normalize_state_include(include),
                )

            if ordinary.effective_liquid1_fraction <= 1.0e-10:
                return finish_ordinary(
                    'spinodal_not_applicable_no_liquid'
                )
            spinodal = self.liquid_spinodal_stability(
                ordinary.T,
                ordinary.x1 or ordinary.x or ordinary.composition,
            )
            if spinodal['locally_stable']:
                return finish_ordinary(
                    'locally_stable_spinodal_only',
                    {'spinodal': spinodal},
                )
            stability = 'spinodal_unstable_vlle_selected'

        T, result = self.flash3_PV(
            composition,
            P,
            target,
            max_iter=200,
        )
        equilibrium = self._fluid_result_from_vlle(
            result,
            stability=stability,
            extra={'spinodal': spinodal} if spinodal is not None else None,
        )
        total = sum(composition.values())
        normalized = (
            {component: value / total for component, value in composition.items()}
            if total > 0.0 else dict(composition)
        )
        state = StreamState(
            T=float(T),
            P=float(P),
            F=float(F),
            composition=normalized,
            fluid_phase_model=self.fluid_phase_model,
        )
        return self._apply_fluid_equilibrium_to_state(
            state,
            equilibrium,
            self._normalize_state_include(include),
        )

    def activity_coefficients(self, T: float, composition: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError

    def excess_enthalpy(self, composition: dict[str, float], T: float) -> float:
        """
        Excess enthalpy from the temperature derivative of activity coefficients [kJ/kmol].

        H^E = -R T^2 sum_i x_i (d ln gamma_i / dT)_x
        """
        x_total = sum(max(float(value), 0.0) for value in composition.values())
        if x_total <= 0.0:
            return 0.0
        x = {
            comp: max(float(composition.get(comp, 0.0)), 0.0) / x_total
            for comp in self.components
        }
        dT = max(1e-3, 1e-4 * T)
        T_low = max(1.0, T - dT)
        T_high = T + dT
        gamma_low = self.activity_coefficients(T_low, x)
        gamma_high = self.activity_coefficients(T_high, x)
        derivative_sum = 0.0
        for comp in self.components:
            if x.get(comp, 0.0) <= 0.0:
                continue
            ln_high = math.log(max(gamma_high.get(comp, 1.0), 1e-300))
            ln_low = math.log(max(gamma_low.get(comp, 1.0), 1e-300))
            derivative_sum += x[comp] * (ln_high - ln_low) / (T_high - T_low)
        return -R * T**2 * derivative_sum

    def excess_gibbs(self, composition: dict[str, float], T: float) -> float:
        """Liquid excess Gibbs energy [kJ/kmol]."""
        x_total = sum(max(float(value), 0.0) for value in composition.values())
        if x_total <= 0.0:
            return 0.0
        x = {
            comp: max(float(composition.get(comp, 0.0)), 0.0) / x_total
            for comp in self.components
        }
        gamma = self.activity_coefficients(T, x)
        return R * T * sum(
            x[comp] * math.log(max(gamma.get(comp, 1.0), 1e-300))
            for comp in self.components
            if x.get(comp, 0.0) > 0.0
        )

    def excess_entropy(self, composition: dict[str, float], T: float) -> float:
        """Liquid excess entropy [kJ/kmol-K] from S^E = (H^E - G^E) / T."""
        return (self.excess_enthalpy(composition, T) - self.excess_gibbs(composition, T)) / max(T, 1e-12)

    def _vapor_residual_enthalpy(self, composition: dict[str, float], T: float, P: float) -> float:
        """Vapor residual enthalpy from the gamma-phi vapor backend [kJ/kmol]."""
        rk_model = self.vapor_eos
        if rk_model is None:
            return 0.0
        try:
            return rk_model.departure_enthalpy(T, P, composition, 'vapor')
        except Exception as exc:
            self.add_warning(
                f"Could not calculate RK vapor residual enthalpy for "
                f"{self.__class__.__name__}; using zero residual enthalpy fallback "
                f"where unavailable ({exc})."
            )
            return 0.0

    def _vapor_residual_entropy(self, composition: dict[str, float], T: float,
                                P: float) -> float:
        """Vapor residual entropy from the gamma-phi vapor backend [kJ/kmol-K]."""
        rk_model = self.vapor_eos
        if rk_model is None or not hasattr(rk_model, 'departure_entropy'):
            return 0.0
        try:
            return rk_model.departure_entropy(T, P, composition, 'vapor')
        except Exception as exc:
            self.add_warning(
                f"Could not calculate RK vapor residual entropy for "
                f"{self.__class__.__name__}; using zero residual entropy fallback "
                f"where unavailable ({exc})."
            )
            return 0.0

    def _vapor_residual_cp(self, composition: dict[str, float], T: float,
                           P: float) -> float:
        """Vapor residual Cp [kJ/kmol-K] as d(H_residual)/dT at constant P.

        Differentiates the polymorphic vapor residual enthalpy, so gamma-phi
        models get the RK/PR departure slope and VDM models get the
        association-enthalpy slope with the same code path.
        """
        cache = getattr(self, '_vapor_residual_cp_cache', None)
        if cache is None:
            cache = self._vapor_residual_cp_cache = {}
        cache_key = (float(T), float(P), self._composition_cache_key(composition))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        dT = max(0.05, 1e-4 * float(T))
        T_low = max(1.0, float(T) - dT)
        T_high = float(T) + dT
        try:
            h_low = self._vapor_residual_enthalpy(composition, T_low, P)
            h_high = self._vapor_residual_enthalpy(composition, T_high, P)
            value = (h_high - h_low) / (T_high - T_low)
        except Exception:
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        return self._set_limited_cache(cache, cache_key, value)

    def _excess_cp(self, composition: dict[str, float], T: float) -> float:
        """Liquid excess heat capacity [kJ/kmol-K] as d(H^E)/dT."""
        cache = getattr(self, '_excess_cp_cache', None)
        if cache is None:
            cache = self._excess_cp_cache = {}
        cache_key = (float(T), self._composition_cache_key(composition))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        dT = max(0.25, 1e-3 * float(T))
        T_low = max(1.0, float(T) - dT)
        T_high = float(T) + dT
        try:
            value = (
                self.excess_enthalpy(composition, T_high)
                - self.excess_enthalpy(composition, T_low)
            ) / (T_high - T_low)
        except Exception:
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        return self._set_limited_cache(cache, cache_key, value)

    def mixture_Cp(self, composition: dict[str, float], T: float,
                   vapor_fraction: float = 1.0,
                   P: Optional[float] = None) -> float:
        """Mixture Cp [kJ/kmol-K] consistent with this model's enthalpy.

        Adds the liquid excess Cp d(H^E)/dT and, for gamma-phi/VDM variants,
        the vapor residual d(H_res)/dT, so Cp equals dH/dT phase by phase.
        """
        cp = super().mixture_Cp(composition, T, vapor_fraction, P)
        V = max(0.0, min(1.0, float(vapor_fraction)))
        if V < 0.999:
            cp += (1.0 - V) * self._excess_cp(composition, T)
        if (
            P is not None and P > 0.0 and V > 0.001
            and self._vapor_phase_correction_active()
        ):
            cp += V * self._vapor_residual_cp(composition, T, float(P))
        return cp

    def _gamma_phi_phi_sat(self, comp: str, T: float, Psat: float) -> float:
        """Pure saturated-vapor fugacity coefficient for gamma-phi models."""
        cache_key = (comp, float(T))
        cached = self._phi_sat_cache.get(cache_key)
        if cached is not None:
            return cached

        value = 1.0
        if self.vapor_eos is None:
            if len(self._phi_sat_cache) > 20000:
                self._phi_sat_cache.clear()
            self._phi_sat_cache[cache_key] = value
            return value
        try:
            phi_sat = self.vapor_eos.fugacity_coefficients(T, Psat, {comp: 1.0}, 'vapor')
            value = max(float(phi_sat.get(comp, 1.0)), 1e-12)
        except Exception:
            self.add_warning(
                f"Could not calculate saturated vapor fugacity coefficient for "
                f"{self._component_label(comp)} in the gamma-phi reference state; "
                "using phi_sat=1 where unavailable."
            )

        if len(self._phi_sat_cache) > 20000:
            self._phi_sat_cache.clear()
        self._phi_sat_cache[cache_key] = value
        return value

    def _gamma_phi_poynting_factor(self, comp: str, T: float, P: float, Psat: float) -> float:
        """Poynting correction using pure liquid molar volume at T."""
        cache_key = (comp, float(T), float(P))
        cached = self._poynting_cache.get(cache_key)
        if cached is not None:
            return cached

        value = 1.0
        volume_m3_per_kmol, endpoint_note = self._liquid_molar_volume_for_poynting(comp, T)
        if endpoint_note is not None:
            warning_key = (comp, endpoint_note)
            if warning_key not in self._poynting_endpoint_warning_keys:
                self._poynting_endpoint_warning_keys.add(warning_key)
                self.add_warning(
                    f"Liquid molar volume for {self._component_label(comp)} was evaluated "
                    f"outside the in-range density correlation for the gamma-phi Poynting "
                    f"correction ({endpoint_note})."
                )
        if volume_m3_per_kmol is None:
            self.add_warning(
                f"Could not calculate Poynting correction for {self._component_label(comp)}; "
                "using Poynting=1 where liquid molar volume is unavailable."
            )

        if volume_m3_per_kmol is not None:
            volume_m3_per_mol = float(volume_m3_per_kmol) / 1000.0
            exponent = volume_m3_per_mol * (P - Psat) / (R_BAR * T)
            value = math.exp(max(min(exponent, 50.0), -50.0))

        if len(self._poynting_cache) > 20000:
            self._poynting_cache.clear()
        self._poynting_cache[cache_key] = value
        return value

    def _gamma_phi_reference_factors(self, T: float, P: float) -> dict[str, float]:
        """
        Return phi_sat * Psat * Poynting for gamma-phi K-values [bar].

        The full modified Raoult expression is:
            K_i = gamma_i * phi_i^sat * Psat_i * Poynting_i / (phi_i^V * P)
        """
        reference = {}
        for comp in self.components:
            Psat = self.Psat(comp, T)
            phi_sat = self._gamma_phi_phi_sat(comp, T, Psat)
            poynting = self._gamma_phi_poynting_factor(comp, T, P, Psat)
            reference[comp] = phi_sat * Psat * poynting
        return reference

    def mixture_enthalpy(self, composition: dict[str, float], T: float,
                        vapor_fraction: float = 1.0,
                        x: Optional[dict] = None,
                        y: Optional[dict] = None,
                        P: float = P_REF) -> float:
        """
        Mixture molar enthalpy [kJ/kmol] with liquid excess enthalpy and,
        for gamma-phi variants, vapor residual enthalpy.
        """
        H = super().mixture_enthalpy(composition, T, vapor_fraction, x, y, P)
        V = max(0.0, min(1.0, float(vapor_fraction)))

        if V < 0.999:
            liquid_composition = x or composition
            H += (1.0 - V) * self.excess_enthalpy(liquid_composition, T)

        if V > 0.001:
            vapor_composition = y or composition
            H += V * self._vapor_residual_enthalpy(vapor_composition, T, P)

        return H

    def mixture_entropy(self, composition: dict[str, float], T: float,
                        vapor_fraction: float = 1.0,
                        x: Optional[dict] = None,
                        y: Optional[dict] = None,
                        P: float = P_REF) -> float:
        """
        Mixture molar entropy [kJ/kmol-K] with liquid excess entropy and,
        for gamma-phi variants, vapor residual entropy.
        """
        S = super().mixture_entropy(composition, T, vapor_fraction, x, y, P)
        V = max(0.0, min(1.0, float(vapor_fraction)))

        if V < 0.999:
            liquid_composition = x or composition
            S += (1.0 - V) * self.excess_entropy(liquid_composition, T)

        if V > 0.001:
            vapor_composition = y or composition
            S += V * self._vapor_residual_entropy(vapor_composition, T, P)

        return S

    def K_value(self, comp: str, T: float, P: float,
                x: Optional[dict[str, float]] = None) -> float:
        composition = x if x is not None else {comp: 1.0}
        return self.K_values(T, P, composition).get(comp, 1.0)

    def K_values(self, T: float, P: float,
                 composition: dict[str, float]) -> dict[str, float]:
        cache_key = self._k_values_cache_key('gamma_raoult', T, P, composition)
        cached = self._get_cached_k_values(cache_key)
        if cached is not None:
            return cached

        gamma = self.activity_coefficients(T, composition)
        K = {
            comp: max(
                1e-6,
                min(1e6, gamma.get(comp, 1.0) * self.Psat(comp, T) / max(P, 1e-12)),
            )
            for comp in self.components
        }
        return self._set_cached_k_values(cache_key, K)

    def aqueous_K_values(self, T: float, P: float,
                         composition: dict[str, float],
                         context: AqueousEquilibriumContext) -> dict[str, float]:
        """Ideal-vapor gamma/Henry K-values without evaluating Psat for Henry solutes."""
        self._warn_aqueous_henry_pressure(P, context)
        x = self._normalized_aqueous_composition(composition, self.components)
        cache_key = self._k_values_cache_key('aqueous_gamma', T, P, x) + (context.cache_key(),)
        cached = self._get_cached_k_values(cache_key)
        if cached is not None:
            return cached

        gamma = self.activity_coefficients(
            T,
            self._aqueous_bulk_activity_composition(x, context),
        )
        K = {}
        for comp in self.components:
            if comp in context.component_data:
                value = self._henry_ideal_vapor_K_value(comp, T, P, context)
            else:
                value = gamma.get(comp, 1.0) * self.Psat(comp, T) / max(P, 1e-12)
            K[comp] = max(1e-12, min(1e12, float(value)))
        return self._set_cached_k_values(cache_key, K)

    def _aqueous_bulk_activity_composition(
        self,
        composition: dict[str, float],
        context: AqueousEquilibriumContext,
    ) -> dict[str, float]:
        """Normalize the non-Henry liquid submixture used by gamma models."""
        bulk = {
            comp: max(float(composition.get(comp, 0.0)), 0.0)
            for comp in self.components
            if comp not in context.component_data
        }
        total = sum(bulk.values())
        if total <= 1.0e-15:
            raise ThermodynamicsError(
                "Aqueous Henry equilibrium requires a non-Henry bulk liquid component"
            )
        normalized = {comp: value / total for comp, value in bulk.items()}
        normalized.update({comp: 0.0 for comp in context.component_data})
        return normalized

    def _gamma_phi_K_values(self, T: float, P: float,
                            composition: dict[str, float]) -> dict[str, float]:
        cache_key = self._k_values_cache_key('gamma_phi_rk', T, P, composition)
        cached = self._get_cached_k_values(cache_key)
        if cached is not None:
            return cached

        x = {
            comp: max(float(composition.get(comp, 0.0)), 0.0)
            for comp in self.components
        }
        x_sum = sum(x.values())
        if x_sum <= 0.0:
            x = {comp: 1.0 / len(self.components) for comp in self.components}
        else:
            x = {comp: value / x_sum for comp, value in x.items()}

        gamma = self.activity_coefficients(T, x)
        reference = self._gamma_phi_reference_factors(T, P)
        y = {
            comp: x.get(comp, 0.0) * gamma.get(comp, 1.0) * reference[comp] / max(P, 1e-12)
            for comp in self.components
        }
        y_sum = sum(max(value, 0.0) for value in y.values())
        y = {comp: max(value, 0.0) / y_sum for comp, value in y.items()} if y_sum > 0 else dict(x)

        K = {}
        for _ in range(8):
            try:
                phi_v = self.vapor_eos.fugacity_coefficients(T, P, y, 'vapor')
            except Exception:
                phi_v = {comp: 1.0 for comp in self.components}
            K = {}
            for comp in self.components:
                phi = max(phi_v.get(comp, 1.0), 1e-8)
                value = gamma.get(comp, 1.0) * reference[comp] / (phi * max(P, 1e-12))
                K[comp] = max(1e-6, min(1e6, value))
            y_new = {comp: x.get(comp, 0.0) * K[comp] for comp in self.components}
            y_sum = sum(max(value, 0.0) for value in y_new.values())
            if y_sum <= 0.0:
                break
            y_new = {comp: max(value, 0.0) / y_sum for comp, value in y_new.items()}
            if max(abs(y_new[comp] - y.get(comp, 0.0)) for comp in self.components) < 1e-8:
                y = y_new
                break
            y = y_new
        return self._set_cached_k_values(cache_key, K)

    def _rachford_rice(self, z: dict[str, float], K: dict[str, float], V_guess: float) -> float:
        V = V_guess
        for _ in range(50):
            f = 0.0
            df = 0.0
            for comp in self.components:
                z_i = z.get(comp, 0.0)
                K_i = K.get(comp, 1.0)
                if z_i <= 0:
                    continue
                term = K_i - 1.0
                denom = 1.0 + V * term
                if abs(denom) < 1e-10:
                    denom = 1e-10
                f += z_i * term / denom
                df -= z_i * term * term / (denom * denom)
            if abs(df) < 1e-12:
                break
            V_new = V - f / df
            if V_new < 0:
                V_new = V / 2.0
            elif V_new > 1:
                V_new = (V + 1.0) / 2.0
            if abs(V_new - V) < 1e-8:
                break
            V = V_new
        return max(0.0, min(1.0, V))

    def flash_TP(self, composition: dict[str, float], T: float, P: float) -> tuple[float, dict, dict]:
        return self.flash(T, P, composition)

    def bubble_point_T(self, composition: dict[str, float], P: float,
                       T_guess: float = 350.0) -> float:
        T = _solve_bubble_point_temperature(self, composition, P, T_guess)
        self._record_estimated_interaction_extrapolation(
            T, ((1.0, composition),)
        )
        return T

    def dew_point_T(self, composition: dict[str, float], P: float,
                    T_guess: float = 350.0) -> float:
        T = _solve_dew_point_temperature(self, composition, P, T_guess)
        # The converged dew calculation necessarily exercised a liquid
        # activity model for every represented condensable pair. The vapor
        # composition is sufficient for relevance filtering here; warning
        # generation must not trigger another equilibrium iteration.
        self._record_estimated_interaction_extrapolation(
            T, ((1.0, composition),)
        )
        return T

    def bubble_point_T_vlle(self, composition: dict[str, float], P: float,
                            T_guess: float = 350.0,
                            max_iter: int = 100) -> float:
        """
        LLE-aware bubble temperature at fixed pressure.

        If the liquid is stable as one phase, this reduces to the ordinary
        bubble point.  If the liquid splits, the residual uses the bubble
        pressure of the equilibrium liquid phases, which should be identical
        for a converged LLE split.
        """
        z = self._normalize_phase_composition(composition)

        def residual(T: float) -> float:
            has_lle, x1, x2, _beta = self.liquid_liquid_equilibrium(
                z, T, max_iter=max_iter, tol=1e-8
            )
            if has_lle:
                p1 = self.bubble_point_P(self._normalize_phase_composition(x1), T)
                p2 = self.bubble_point_P(self._normalize_phase_composition(x2), T)
                return 0.5 * (p1 + p2) - P
            return self.bubble_point_P(z, T) - P

        ordinary = self.bubble_point_T(z, P, T_guess)
        return self._solve_temperature_residual(
            residual,
            T_guess=ordinary,
            centers=(ordinary, T_guess),
            label='VLLE bubble point',
        )

    def dew_point_T_vlle(self, composition: dict[str, float], P: float,
                         T_guess: float = 350.0,
                         max_iter: int = 100) -> float:
        """
        VLLE-aware dew temperature at fixed pressure.

        The residual is evaluated with the reference Flash3 TP solver near the
        dew boundary.  A local bracket around the ordinary dew point keeps this
        path cheap for normal column calculations while still detecting
        two-liquid incipient condensate when present.
        """
        y = self._normalize_phase_composition(composition)
        ordinary = self.dew_point_T(y, P, T_guess)
        target = 1.0 - 1e-8

        def residual(T: float) -> float:
            result = self.flash3_TP(y, T, P, max_iter=max_iter)
            return result.vapor_fraction - target

        return self._solve_temperature_residual(
            residual,
            T_guess=ordinary,
            centers=(ordinary, T_guess),
            label='VLLE dew point',
            exact_tolerance=2e-8,
        )

    def flash3_PV(self, composition: dict[str, float], P: float,
                  vapor_fraction: float, T_guess: float = 350.0,
                  max_iter: int = 100) -> tuple[float, VLLEFlashResult]:
        z = self._normalize_phase_composition(composition)
        target = max(0.0, min(1.0, float(vapor_fraction)))
        binary_invariant = self._binary_invariant_at_pressure(z, P, T_guess, max_iter)
        if binary_invariant is not None:
            T_binary, binary_result = binary_invariant
            bounds = binary_result.extra.get('vapor_fraction_bounds') if binary_result.extra else None
            if self._bounds_contain(bounds, target):
                return T_binary, self._with_binary_vapor_fraction(binary_result, z, target)

        if target <= 1e-10:
            T = self.bubble_point_T_vlle(z, P, T_guess, max_iter=max_iter)
            return T, self.flash3_TP(z, T, P, max_iter=max_iter)
        if target >= 1.0 - 1e-10:
            T = self.dew_point_T_vlle(z, P, T_guess, max_iter=max_iter)
            return T, self.flash3_TP(z, T, P, max_iter=max_iter)

        trials: dict[float, VLLEFlashResult] = {}

        def result_at(T: float) -> VLLEFlashResult:
            key = round(float(T), 10)
            result = trials.get(key)
            if result is None:
                result = self.flash3_TP(z, T, P, max_iter=max_iter)
                trials[key] = result
            return result

        def residual(T: float) -> float:
            return result_at(T).vapor_fraction - target

        def feed_has_lle_near_guess() -> bool:
            try:
                has_lle, _x1, _x2, _beta = self.liquid_liquid_equilibrium(
                    z,
                    max(1.0, min(5000.0, float(T_guess))),
                    max_iter=max_iter,
                    tol=1e-8,
                )
                return bool(has_lle)
            except Exception:
                return True

        prefer_vlle_branch = feed_has_lle_near_guess()

        def try_local_temperature_solve() -> tuple[Optional[tuple[float, VLLEFlashResult]], Optional[tuple[float, VLLEFlashResult]]]:
            samples: list[tuple[float, float, VLLEFlashResult]] = []
            fallback_match: tuple[float, VLLEFlashResult] | None = None
            guess = max(1.0, min(5000.0, float(T_guess)))

            for width in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0):
                points = [guess] if width <= 0.0 else [
                    max(1.0, guess - width),
                    min(5000.0, guess + width),
                ]
                for point in points:
                    try:
                        result = result_at(point)
                        f_value = result.vapor_fraction - target
                    except Exception:
                        continue
                    if not math.isfinite(f_value):
                        continue
                    if abs(f_value) <= 1e-8:
                        if result.phase_count == 3:
                            return (point, result), fallback_match
                        if fallback_match is None:
                            fallback_match = (point, result)
                    samples.append((point, f_value, result))

                phase3_samples = sorted(
                    (point, f_value)
                    for point, f_value, result in samples
                    if result.phase_count == 3
                )
                for (T1, f1), (T2, f2) in zip(phase3_samples, phase3_samples[1:]):
                    if f1 * f2 < 0.0:
                        T = brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=80)
                        return (T, result_at(T)), fallback_match

                if prefer_vlle_branch:
                    continue

                all_samples = sorted((point, f_value) for point, f_value, _result in samples)
                for (T1, f1), (T2, f2) in zip(all_samples, all_samples[1:]):
                    if f1 * f2 < 0.0:
                        T = brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=80)
                        result = result_at(T)
                        if result.phase_count == 3:
                            return (T, result), fallback_match
                        if fallback_match is None:
                            fallback_match = (T, result)
            return None, fallback_match

        local_result, local_fallback = try_local_temperature_solve()
        if local_result is not None:
            return local_result
        if local_fallback is not None and not prefer_vlle_branch:
            return local_fallback

        bubble = self.bubble_point_T_vlle(z, P, T_guess, max_iter=max_iter)
        dew = self.dew_point_T_vlle(z, P, T_guess, max_iter=max_iter)
        T_low, T_high = sorted((bubble, dew))

        binary_candidate = result_at(0.5 * (T_low + T_high))
        bounds = binary_candidate.extra.get('vapor_fraction_bounds') if binary_candidate.extra else None
        if bounds is not None and bounds[0] - 1e-10 <= target <= bounds[1] + 1e-10:
            adjusted = self._with_binary_vapor_fraction(binary_candidate, z, target)
            return 0.5 * (T_low + T_high), adjusted

        try:
            f_low = residual(T_low)
            if math.isfinite(f_low) and abs(f_low) <= 1e-8:
                return T_low, result_at(T_low)
            f_high = residual(T_high)
            if math.isfinite(f_high) and abs(f_high) <= 1e-8:
                return T_high, result_at(T_high)
            if math.isfinite(f_low) and math.isfinite(f_high) and f_low * f_high < 0.0:
                T = brentq(residual, T_low, T_high, xtol=1e-7, rtol=1e-9, maxiter=80)
                return T, result_at(T)
        except Exception:
            pass

        T = self._solve_bounded_scalar_residual(
            residual,
            T_low,
            T_high,
            preferred=0.5 * (T_low + T_high),
            label=f'VLLE PV vapor fraction {target:g}',
            exact_tolerance=1e-8,
        )
        result = result_at(T)
        if result.phase_count == 3 or local_fallback is None:
            return T, result
        return local_fallback

    def flash3_TV(self, composition: dict[str, float], T: float,
                  vapor_fraction: float, P_guess: float = P_REF,
                  max_iter: int = 100) -> tuple[float, VLLEFlashResult]:
        z = self._normalize_phase_composition(composition)
        target = max(0.0, min(1.0, float(vapor_fraction)))
        trials: dict[float, VLLEFlashResult] = {}
        compiled_backend = self.compiled_vlle_backend() if len(self.components) > 2 else None
        binary_invariant = self._binary_invariant_at_temperature(z, T, max_iter)
        if binary_invariant is not None:
            P_binary, binary_result = binary_invariant
            bounds = binary_result.extra.get('vapor_fraction_bounds') if binary_result.extra else None
            if self._bounds_contain(bounds, target):
                return P_binary, self._with_binary_vapor_fraction(binary_result, z, target)

        def result_at(P_value: float) -> VLLEFlashResult:
            key = round(float(P_value), 10)
            result = trials.get(key)
            if result is None:
                if compiled_backend is None:
                    result = self.flash3_TP(z, T, P_value, max_iter=max_iter)
                else:
                    result = self._compiled_vlle_result_to_reference(
                        compiled_backend.flash_TP(z, T, P_value, max_iter=max_iter)
                    )
                trials[key] = result
            return result

        def residual(P_value: float) -> float:
            return result_at(P_value).vapor_fraction - target

        def try_vlle_branch() -> tuple[float, VLLEFlashResult] | None:
            try:
                has_lle, x1, x2, _beta = self.liquid_liquid_equilibrium(
                    z, T, max_iter=max_iter, tol=1e-8
                )
            except Exception:
                return None
            if not has_lle:
                return None
            try:
                p_top = 0.5 * (
                    self.bubble_point_P(self._normalize_phase_composition(x1), T)
                    + self.bubble_point_P(self._normalize_phase_composition(x2), T)
                )
            except Exception:
                return None
            if not math.isfinite(p_top) or p_top <= 0.0:
                return None

            for pressure_factor in (0.98, 0.99, 0.995):
                P_previous = max(1e-8, float(p_top))
                previous = result_at(P_previous)
                f_previous = previous.vapor_fraction - target
                phase_previous = previous.phase_count
                if phase_previous == 3 and abs(f_previous) <= 1e-8:
                    return P_previous, previous

                for _ in range(80):
                    P_value = max(1e-8, P_previous * pressure_factor)
                    result = result_at(P_value)
                    f_value = result.vapor_fraction - target
                    phase = result.phase_count
                    if phase == 3 and abs(f_value) <= 1e-8:
                        return P_value, result
                    if phase == 3 and phase_previous == 3 and f_value * f_previous < 0.0:
                        P = brentq(residual, P_value, P_previous, xtol=1e-7, rtol=1e-9, maxiter=80)
                        return P, result_at(P)
                    if phase != 3 and phase_previous == 3:
                        if f_value * f_previous < 0.0:
                            P = brentq(residual, P_value, P_previous, xtol=1e-7, rtol=1e-9, maxiter=80)
                            root_result = result_at(P)
                            if root_result.phase_count == 3:
                                return P, root_result
                        if abs(f_previous) <= 5e-7:
                            return P_previous, previous
                        break

                    P_previous = P_value
                    previous = result
                    f_previous = f_value
                    phase_previous = phase
            return None

        branch_result = try_vlle_branch()
        if branch_result is not None:
            return branch_result

        def try_single_liquid_pressure_solve() -> tuple[float, VLLEFlashResult] | None:
            try:
                has_lle, _x1, _x2, _beta = self.liquid_liquid_equilibrium(
                    z, T, max_iter=max_iter, tol=1e-8
                )
            except Exception:
                return None
            if has_lle:
                return None

            guess = max(1e-8, float(P_guess))
            try:
                f_guess = residual(guess)
                if math.isfinite(f_guess) and abs(f_guess) <= 1e-8:
                    return guess, result_at(guess)
            except Exception:
                pass

            for low_factor, high_factor in (
                (0.9, 1.1),
                (0.8, 1.25),
                (0.67, 1.5),
                (0.5, 2.0),
                (0.25, 4.0),
                (0.1, 10.0),
            ):
                low = max(1e-8, guess * low_factor)
                high = max(low * 1.000001, guess * high_factor)
                try:
                    f_low = residual(low)
                    if math.isfinite(f_low) and abs(f_low) <= 1e-8:
                        return low, result_at(low)
                    f_high = residual(high)
                    if math.isfinite(f_high) and abs(f_high) <= 1e-8:
                        return high, result_at(high)
                    if math.isfinite(f_low) and math.isfinite(f_high) and f_low * f_high < 0.0:
                        P = brentq(residual, low, high, xtol=1e-7, rtol=1e-9, maxiter=80)
                        return P, result_at(P)
                except Exception:
                    continue
            return None

        single_liquid_result = try_single_liquid_pressure_solve()
        if single_liquid_result is not None:
            return single_liquid_result

        pressure_basis = [max(1e-6, float(P_guess)), P_REF]
        try:
            pressure_basis.append(max(1e-6, self.bubble_point_P(z, T)))
        except Exception:
            pass
        grid = self._pressure_grid(pressure_basis)
        P = self._solve_grid_scalar_residual(
            residual,
            grid,
            preferred=max(1e-6, float(P_guess)),
            label=f'VLLE TV vapor fraction {target:g}',
            exact_tolerance=1e-8,
        )
        return P, result_at(P)

    def _flash3_property_target_solve(
        self,
        composition: dict[str, float],
        P: float,
        target: float,
        T_guess: float,
        max_iter: int,
        property_of_result,
        binary_adjuster,
        exact_floor: float,
        solution_floor: float,
        label: str,
        residual_units: str,
    ) -> tuple[float, VLLEFlashResult, float]:
        """Shared temperature solve for a flash3 mixture-property target.

        ``property_of_result(T, P, result)`` evaluates the target property of
        a VLLE flash result. PH and PS flashes differ only in that callable,
        the binary-invariant adjuster, and their tolerance floors.
        """
        z = self._normalize_phase_composition(composition)
        trials: dict[float, VLLEFlashResult] = {}
        properties: dict[float, float] = {}
        compiled_backend = self.compiled_vlle_backend() if len(self.components) > 2 else None
        compiled_trials: dict[float, VLLEFlashResult] = {}
        compiled_properties: dict[float, float] = {}

        def result_at(T: float) -> VLLEFlashResult:
            key = round(float(T), 10)
            result = trials.get(key)
            if result is None:
                result = self.flash3_TP(z, T, P, max_iter=max_iter)
                trials[key] = result
            return result

        def property_at(T: float) -> float:
            key = round(float(T), 10)
            value = properties.get(key)
            if value is None:
                value = property_of_result(T, P, result_at(T))
                properties[key] = value
            return value

        def residual(T: float) -> float:
            return property_at(T) - target

        def compiled_result_at(T: float) -> VLLEFlashResult:
            if compiled_backend is None:
                raise RuntimeError("Compiled VLLE backend is unavailable")
            key = round(float(T), 10)
            result = compiled_trials.get(key)
            if result is None:
                result = self._compiled_vlle_result_to_reference(
                    compiled_backend.flash_TP(z, T, P, max_iter=max_iter)
                )
                compiled_trials[key] = result
            return result

        def compiled_property_at(T: float) -> float:
            key = round(float(T), 10)
            value = compiled_properties.get(key)
            if value is None:
                value = property_of_result(T, P, compiled_result_at(T))
                compiled_properties[key] = value
            return value

        def compiled_residual(T: float) -> float:
            return compiled_property_at(T) - target

        exact_tolerance = max(exact_floor, abs(target) * 1e-9)
        solution_tolerance = max(solution_floor, abs(target) * 1e-8)
        evaluated: dict[float, tuple[float, VLLEFlashResult]] = {}

        def sample(T_value: float) -> tuple[float, VLLEFlashResult]:
            key = round(float(T_value), 10)
            value = evaluated.get(key)
            if value is None:
                result = result_at(T_value)
                value = (property_at(T_value) - target, result)
                evaluated[key] = value
            return value

        binary_invariant = self._binary_invariant_at_pressure(z, P, T_guess, max_iter)
        if binary_invariant is not None:
            T_binary, binary_result = binary_invariant
            adjusted = binary_adjuster(binary_result, z, T_binary, P, target)
            if adjusted is not None:
                residual_value = property_of_result(T_binary, P, adjusted) - target
                return T_binary, adjusted, residual_value

        def phase3_branch_temperature_solve() -> float | None:
            try:
                bubble = self.bubble_point_T_vlle(z, P, T_guess, max_iter=max_iter)
                f_previous, previous = sample(bubble)
            except Exception:
                return None
            if previous.phase_count != 3 or not math.isfinite(f_previous):
                return None
            if abs(f_previous) <= exact_tolerance:
                return bubble

            def compiled_phase3_branch_temperature_solve() -> float | None:
                if compiled_backend is None:
                    return None

                previous_compiled_T: float | None = None
                previous_compiled_f: float | None = None
                for offset in (1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25):
                    point = bubble + offset
                    try:
                        f_value = compiled_residual(point)
                        result = compiled_result_at(point)
                    except Exception:
                        continue
                    if result.phase_count != 3 or not math.isfinite(f_value):
                        continue
                    if abs(f_value) <= exact_tolerance:
                        return point
                    previous_compiled_T = point
                    previous_compiled_f = f_value
                    break
                if previous_compiled_T is None or previous_compiled_f is None:
                    return None

                step = 0.25
                max_span = 8.0
                refinement_points = 5

                def bracket_compiled_phase3(T1: float, f1: float, T2: float, f2: float) -> float | None:
                    if f1 * f2 >= 0.0:
                        return None
                    T = brentq(compiled_residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=80)
                    try:
                        result = compiled_result_at(T)
                    except Exception:
                        return None
                    if result.phase_count == 3:
                        return T
                    return None

                while previous_compiled_T - bubble < max_span:
                    point = previous_compiled_T + step
                    try:
                        f_value = compiled_residual(point)
                        result = compiled_result_at(point)
                    except Exception:
                        previous_compiled_T = point
                        continue
                    if result.phase_count == 3 and math.isfinite(f_value):
                        if abs(f_value) <= exact_tolerance:
                            return point
                        bracketed = bracket_compiled_phase3(
                            previous_compiled_T,
                            previous_compiled_f,
                            point,
                            f_value,
                        )
                        if bracketed is not None:
                            return bracketed
                        previous_compiled_T = point
                        previous_compiled_f = f_value
                        continue

                    for refinement_index in range(1, refinement_points + 1):
                        refined = (
                            previous_compiled_T
                            + (point - previous_compiled_T) * refinement_index / (refinement_points + 1)
                        )
                        try:
                            f_refined = compiled_residual(refined)
                            refined_result = compiled_result_at(refined)
                        except Exception:
                            continue
                        if refined_result.phase_count != 3 or not math.isfinite(f_refined):
                            continue
                        if abs(f_refined) <= exact_tolerance:
                            return refined
                        bracketed = bracket_compiled_phase3(
                            previous_compiled_T,
                            previous_compiled_f,
                            refined,
                            f_refined,
                        )
                        if bracketed is not None:
                            return bracketed
                        previous_compiled_T = refined
                        previous_compiled_f = f_refined
                    return None
                return None

            compiled_T = compiled_phase3_branch_temperature_solve()
            if compiled_T is not None:
                return compiled_T

            previous_T = bubble
            step = 0.25
            max_span = 8.0
            refinement_points = 5

            def bracket_phase3(T1: float, f1: float, T2: float, f2: float) -> float | None:
                if f1 * f2 >= 0.0:
                    return None
                T = brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=80)
                try:
                    result = result_at(T)
                except Exception:
                    return None
                if result.phase_count == 3:
                    return T
                return None

            for index in range(1, int(max_span / step) + 2):
                point = bubble + index * step
                try:
                    f_value, result = sample(point)
                except Exception:
                    continue
                if result.phase_count == 3 and math.isfinite(f_value):
                    if abs(f_value) <= exact_tolerance:
                        return point
                    bracketed = bracket_phase3(previous_T, f_previous, point, f_value)
                    if bracketed is not None:
                        return bracketed
                    previous_T = point
                    f_previous = f_value
                    continue

                for refinement_index in range(1, refinement_points + 1):
                    refined = previous_T + (point - previous_T) * refinement_index / (refinement_points + 1)
                    try:
                        f_refined, refined_result = sample(refined)
                    except Exception:
                        continue
                    if refined_result.phase_count != 3 or not math.isfinite(f_refined):
                        continue
                    if abs(f_refined) <= exact_tolerance:
                        return refined
                    bracketed = bracket_phase3(previous_T, f_previous, refined, f_refined)
                    if bracketed is not None:
                        return bracketed
                    previous_T = refined
                    f_previous = f_refined
                return None
            return None

        def local_temperature_solve() -> float | None:
            samples: list[tuple[float, float, VLLEFlashResult]] = []
            exact_match: float | None = None
            guess = max(1.0, min(5000.0, float(T_guess)))
            for width in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0):
                points = [guess] if width <= 0.0 else [
                    max(1.0, guess - width),
                    min(5000.0, guess + width),
                ]
                for point in points:
                    try:
                        f_value, result = sample(point)
                    except Exception:
                        continue
                    if not math.isfinite(f_value):
                        continue
                    if abs(f_value) <= exact_tolerance:
                        if result.phase_count == 3:
                            return point
                        if exact_match is None:
                            exact_match = point
                    samples.append((point, f_value, result))

                phase3_samples = sorted(
                    (point, f_value)
                    for point, f_value, result in samples
                    if result.phase_count == 3
                )
                for (T1, f1), (T2, f2) in zip(phase3_samples, phase3_samples[1:]):
                    if f1 * f2 < 0.0:
                        return brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=80)
                if exact_match is not None:
                    return exact_match

            all_samples = sorted((point, f_value) for point, f_value, _result in samples)
            for (T1, f1), (T2, f2) in zip(all_samples, all_samples[1:]):
                if f1 * f2 < 0.0:
                    return brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=80)
            return None

        local_T = phase3_branch_temperature_solve()
        if local_T is None:
            local_T = local_temperature_solve()
        if local_T is not None:
            result = result_at(local_T)
            residual_value = property_at(local_T) - target
            if abs(residual_value) <= solution_tolerance:
                return local_T, result, residual_value

        centers = [float(T_guess)]
        try:
            centers.append(self.bubble_point_T_vlle(z, P, T_guess, max_iter=max_iter))
        except Exception:
            pass
        try:
            centers.append(self.dew_point_T_vlle(z, P, T_guess, max_iter=max_iter))
        except Exception:
            pass
        T = self._solve_temperature_residual(
            residual,
            T_guess=T_guess,
            centers=tuple(centers),
            label=label,
            exact_tolerance=exact_tolerance,
        )
        result = result_at(T)
        residual_value = property_at(T) - target
        if abs(residual_value) > solution_tolerance:
            raise ThermodynamicsError(
                f"{label} residual is {residual_value:.6g} {residual_units}"
            )
        return T, result, residual_value

    def flash3_PH(self, composition: dict[str, float], P: float,
                  H_target: float, T_guess: float = 350.0,
                  max_iter: int = 100) -> tuple[float, VLLEFlashResult, float]:
        return self._flash3_property_target_solve(
            composition, P, float(H_target), T_guess, max_iter,
            property_of_result=self._flash3_mixture_enthalpy,
            binary_adjuster=self._with_binary_enthalpy,
            exact_floor=1e-5,
            solution_floor=20.0,
            label='VLLE PH flash',
            residual_units='kJ/kmol',
        )

    def flash3_PS(self, composition: dict[str, float], P: float,
                  S_target: float, T_guess: float = 350.0,
                  max_iter: int = 100) -> tuple[float, VLLEFlashResult, float]:
        return self._flash3_property_target_solve(
            composition, P, float(S_target), T_guess, max_iter,
            property_of_result=self._flash3_mixture_entropy,
            binary_adjuster=self._with_binary_entropy,
            exact_floor=1e-7,
            solution_floor=0.1,
            label='VLLE PS flash',
            residual_units='kJ/kmol-K',
        )

    def _vapor_phase_correction_active(self) -> bool:
        """True when K-values include vapor-phase corrections (phi or VDM)."""
        return (
            self.vapor_eos is not None
            or bool(getattr(self, '_vdm_models', None))
        )

    def bubble_point_P(self, composition: dict[str, float], T: float) -> float:
        gamma = self.activity_coefficients(T, composition)
        ideal = sum(
            x * gamma.get(comp, 1.0) * self.Psat(comp, T)
            for comp, x in composition.items()
        )
        if not self._vapor_phase_correction_active():
            return ideal
        return self._bubble_point_P_from_K_values(composition, T, ideal)

    def _bubble_point_P_from_K_values(self, composition: dict[str, float],
                                      T: float, P_estimate: float) -> float:
        """Solve sum(x_i K_i(T, P)) = 1 so bubble P includes vapor corrections."""
        def residual(P_value: float) -> float:
            K = self.K_values(T, P_value, composition)
            return sum(
                composition.get(comp, 0.0) * K.get(comp, 1.0)
                for comp in self.components
            ) - 1.0

        P_center = max(float(P_estimate), 1e-8)
        values = []
        for factor in (1.0, 0.8, 1.25, 0.5, 2.0, 0.25, 4.0,
                       0.1, 10.0, 0.03, 30.0, 0.01, 100.0):
            P_try = P_center * factor
            try:
                f_try = residual(P_try)
            except Exception:
                continue
            if not math.isfinite(f_try):
                continue
            if abs(f_try) < 1e-10:
                return P_try
            values.append((P_try, f_try))
        values.sort()
        for (P1, f1), (P2, f2) in zip(values, values[1:]):
            if f1 * f2 < 0.0:
                try:
                    return brentq(residual, P1, P2, xtol=1e-12, rtol=1e-10, maxiter=100)
                except Exception:
                    continue
        if values:
            return min(values, key=lambda item: abs(item[1]))[0]
        return P_center

    def dew_point_P(self, composition: dict[str, float], T: float) -> float:
        y = composition
        x = dict(y)
        P_dew = P_REF
        correction = self._vapor_phase_correction_active()
        history: list[float] = []
        for _ in range(50):
            if correction:
                K = self.K_values(T, max(P_dew, 1e-8), x)
                volatility = {
                    comp: max(K.get(comp, 1.0) * max(P_dew, 1e-8), 1e-30)
                    for comp in y
                }
            else:
                gamma = self.activity_coefficients(T, x)
                volatility = {
                    comp: max(gamma.get(comp, 1.0) * self.Psat(comp, T), 1e-30)
                    for comp in y
                }
            sum_inv = sum(y_i / volatility[comp] for comp, y_i in y.items())
            P_new = 1.0 / max(sum_inv, 1e-30)
            history.append(P_new)
            if len(history) >= 3:
                # Aitken delta-squared acceleration of the linearly
                # convergent pressure substitution, with a step safeguard.
                p1, p2, p3 = history[-3:]
                denominator = (p3 - p2) - (p2 - p1)
                if abs(denominator) > 1e-300:
                    accelerated = p3 - (p3 - p2) ** 2 / denominator
                    if math.isfinite(accelerated) and 0.2 * p3 < accelerated < 5.0 * p3:
                        P_new = accelerated
                        history.clear()
            x_new = {
                comp: y_i * P_new / volatility[comp]
                for comp, y_i in y.items()
            }
            x_sum = sum(x_new.values())
            if x_sum > 0:
                x_new = {comp: value / x_sum for comp, value in x_new.items()}
            x_change = max(
                abs(x_new.get(comp, 0.0) - x.get(comp, 0.0))
                for comp in x_new
            ) if x_new else 0.0
            P_change = abs(P_new - P_dew)
            P_dew = P_new
            x = x_new
            if P_change <= 1e-10 * max(P_dew, 1e-30) and x_change <= 1e-12:
                break

        if correction:
            # Frozen-composition bracketed polish: the substitution converges
            # slowly when phi depends strongly on P, so finish on the dew
            # residual directly with the incipient liquid held fixed.
            def frozen_residual(P_value: float) -> float:
                K = self.K_values(T, max(P_value, 1e-8), x)
                return sum(
                    y_i / max(K.get(comp, 1.0), 1e-30)
                    for comp, y_i in y.items()
                ) - 1.0

            try:
                f_final = frozen_residual(P_dew)
                if abs(f_final) > 1e-9:
                    low = high = P_dew
                    f_low = f_high = f_final
                    for _ in range(60):
                        if f_low <= 0.0 <= f_high:
                            P_dew = brentq(
                                frozen_residual, low, high,
                                xtol=1e-12, rtol=1e-10, maxiter=100,
                            )
                            break
                        if f_low > 0.0:
                            low = max(1e-10, low * 0.7)
                            f_low = frozen_residual(low)
                        if f_high < 0.0:
                            high = high * 1.4
                            f_high = frozen_residual(high)
            except Exception:
                pass
        return P_dew

    def flash(self, T: float, P: float, z: dict[str, float],
              max_iter: int = 50, tol: float = 1e-6) -> tuple[float, dict, dict]:
        z = {
            comp: max(float(z.get(comp, 0.0)), 0.0)
            for comp in self.components
        }
        z_total = sum(z.values())
        if z_total <= 0.0:
            z = {comp: 1.0 / len(self.components) for comp in self.components}
        else:
            z = {comp: value / z_total for comp, value in z.items()}

        K = self.K_values(T, P, z)
        sum_zK = sum(z.get(comp, 0.0) * K.get(comp, 1.0) for comp in self.components)
        if sum_zK <= 1.0:
            return 0.0, dict(z), dict(z)

        V = 0.5
        for _ in range(max_iter):
            V = self._rachford_rice_bounded(z, K, V)
            x, y = self._flash_phase_compositions(z, K, V)
            K_new = self.K_values(T, P, x)
            max_change = 0.0
            for comp in self.components:
                if K.get(comp, 1.0) > 1e-10:
                    max_change = max(max_change, abs(K_new.get(comp, 1.0) - K[comp]) / K[comp])
            K = K_new
            if max_change < tol:
                break

        V = self._rachford_rice_bounded(z, K, V)
        return (V, *self._flash_phase_compositions(z, K, V))

    def _normalize_lle_composition(
        self,
        composition: dict[str, float],
    ) -> dict[str, float]:
        if not composition:
            raise ThermodynamicsError("LLE composition must not be empty")
        unknown = [comp for comp in composition if comp not in self.components]
        if unknown:
            raise ThermodynamicsError(
                "LLE composition contains unknown component(s): "
                + ", ".join(unknown)
            )
        normalized = {}
        for comp, raw_value in composition.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ThermodynamicsError(
                    f"LLE composition for {comp} must be numeric"
                ) from error
            if not math.isfinite(value) or value < 0.0:
                raise ThermodynamicsError(
                    f"LLE composition for {comp} must be finite and nonnegative"
                )
            normalized[comp] = value
        total = sum(normalized.values())
        if total <= 0.0:
            raise ThermodynamicsError("LLE composition must have a positive total")
        return {comp: value / total for comp, value in normalized.items()}

    @staticmethod
    def _lle_phase_difference_tolerance(tol: float) -> float:
        return max(1.0e-8, min(1.0e-4, 10.0 * float(tol)))

    @staticmethod
    def _validate_lle_solver_controls(max_iter: int, tol: float) -> None:
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
            raise ThermodynamicsError("LLE max_iter must be a positive integer")
        if not math.isfinite(float(tol)) or tol <= 0.0:
            raise ThermodynamicsError("LLE tolerance must be positive and finite")

    def liquid_liquid_equilibrium(self, composition: dict[str, float], T: float,
                                  max_iter: int = 100, tol: float = 1e-6,
                                  *, allow_unconverged_candidate: bool = False) -> tuple[bool, dict, dict, float]:
        self._validate_lle_solver_controls(max_iter, tol)
        z = self._normalize_lle_composition(composition)
        comps = list(z.keys())
        n_comp = len(comps)
        if n_comp < 2:
            return False, dict(z), dict(z), 0.0
        if n_comp == 2:
            binary_split = self._binary_liquid_liquid_equilibrium(
                z, T, max_iter, tol
            )
            if binary_split is not None:
                return binary_split
            return False, dict(z), dict(z), 0.0

        x1 = {}
        x2 = {}
        for i, comp in enumerate(comps):
            z_i = z[comp]
            if i == 0:
                x1[comp] = min(0.95, z_i * 1.5)
                x2[comp] = max(0.05, z_i * 0.5)
            else:
                x1[comp] = max(0.05, z_i * 0.5)
                x2[comp] = min(0.95, z_i * 1.5)
        s1 = sum(x1.values())
        s2 = sum(x2.values())
        x1 = {comp: value / s1 for comp, value in x1.items()}
        x2 = {comp: value / s2 for comp, value in x2.items()}
        beta = 0.5
        phase_tolerance = self._lle_phase_difference_tolerance(tol)

        for _ in range(max_iter):
            gamma1 = self.activity_coefficients(T, x1)
            gamma2 = self.activity_coefficients(T, x2)
            K = {
                comp: gamma1.get(comp, 1.0) / max(1e-10, gamma2.get(comp, 1.0))
                for comp in comps
            }
            max_diff = 0.0
            for comp in comps:
                act1 = x1[comp] * gamma1.get(comp, 1.0)
                act2 = x2[comp] * gamma2.get(comp, 1.0)
                max_diff = max(max_diff, abs(act1 - act2) / max(act1, act2, 1e-10))
            if max_diff < tol:
                phase_diff = sum(abs(x1[comp] - x2[comp]) for comp in comps) / n_comp
                if phase_diff < phase_tolerance:
                    return False, dict(z), dict(z), 0.0
                lever_denominator = sum(
                    (x2[comp] - x1[comp]) ** 2 for comp in comps
                )
                if lever_denominator > 1e-30:
                    beta = sum(
                        (x2[comp] - x1[comp]) * (z[comp] - x1[comp])
                        for comp in comps
                    ) / lever_denominator
                    beta = max(0.0, min(1.0, beta))
                if beta <= 1e-10 or beta >= 1.0 - 1e-10:
                    return False, dict(z), dict(z), 0.0
                return True, x1, x2, beta

            for _inner in range(20):
                f = sum(z[comp] * (K[comp] - 1.0) / (1.0 + beta * (K[comp] - 1.0)) for comp in comps)
                df = -sum(
                    z[comp] * (K[comp] - 1.0) ** 2 / (1.0 + beta * (K[comp] - 1.0)) ** 2
                    for comp in comps
                )
                if abs(df) < 1e-15:
                    break
                beta = max(1e-12, min(1.0 - 1e-12, beta - f / df))
                if abs(f) < 1e-10:
                    break

            for comp in comps:
                denom = 1.0 + beta * (K[comp] - 1.0)
                x1[comp] = z[comp] / max(denom, 1e-10)
                x2[comp] = K[comp] * x1[comp]
            s1 = sum(x1.values())
            s2 = sum(x2.values())
            if s1 > 0:
                x1 = {comp: value / s1 for comp, value in x1.items()}
            if s2 > 0:
                x2 = {comp: value / s2 for comp, value in x2.items()}

        if allow_unconverged_candidate:
            phase_diff = sum(
                abs(x1[comp] - x2[comp]) for comp in comps
            ) / n_comp
            if (
                phase_diff >= phase_tolerance
                and 1e-10 < beta < 1.0 - 1e-10
            ):
                return True, x1, x2, beta
        return False, dict(z), dict(z), 0.0

    def _binary_liquid_liquid_equilibrium(
        self,
        composition: dict[str, float],
        T: float,
        max_iter: int,
        tol: float,
        prefer_adaptive_starts: bool = False,
    ) -> Optional[tuple[bool, dict, dict, float]]:

        comps = list(composition.keys())
        comp_a, comp_b = comps

        def sigmoid(value: float) -> float:
            if value >= 0:
                exp_neg = math.exp(-value)
                return 1.0 / (1.0 + exp_neg)
            exp_pos = math.exp(value)
            return exp_pos / (1.0 + exp_pos)

        def logit(value: float) -> float:
            value = min(max(value, 1e-12), 1.0 - 1e-12)
            return math.log(value / (1.0 - value))

        def phase(x_a: float) -> dict[str, float]:
            x_a = min(max(float(x_a), 1e-12), 1.0 - 1e-12)
            return {comp_a: x_a, comp_b: 1.0 - x_a}

        def low_high_from_variables(variables) -> tuple[float, float]:
            # Span all ordered pairs 0 < x_low < x_high < 1: real miscibility
            # gaps need not straddle any particular composition.
            x_low = sigmoid(float(variables[0]))
            x_high = x_low + (1.0 - x_low) * sigmoid(float(variables[1]))
            return x_low, x_high

        def residual(variables):
            x_a_1, x_a_2 = low_high_from_variables(variables)
            x1 = phase(x_a_1)
            x2 = phase(x_a_2)
            gamma1 = self.activity_coefficients(T, x1)
            gamma2 = self.activity_coefficients(T, x2)
            return np.array([
                math.log(max(x1[comp_a] * gamma1.get(comp_a, 1.0), 1e-300))
                - math.log(max(x2[comp_a] * gamma2.get(comp_a, 1.0), 1e-300)),
                math.log(max(x1[comp_b] * gamma1.get(comp_b, 1.0), 1e-300))
                - math.log(max(x2[comp_b] * gamma2.get(comp_b, 1.0), 1e-300)),
            ])

        primary_starts = [
            (1e-5, 1.0 - 1e-5),
            (1e-4, 1.0 - 1e-4),
            (1e-3, 1.0 - 1e-3),
            (0.01, 0.99),
            (0.05, 0.95),
            (0.10, 0.90),
            (0.20, 0.80),
            (0.005, 0.35),
            (0.05, 0.55),
            (0.45, 0.95),
            (0.65, 0.995),
        ]
        z_a = composition.get(comp_a, 0.0)
        adaptive_starts = []
        for half_width in (0.005, 0.01, 0.025, 0.05, 0.10, 0.20):
            low = max(1.0e-8, z_a - half_width)
            high = min(1.0 - 1.0e-8, z_a + half_width)
            if high - low > 2.0e-8:
                adaptive_starts.append((low, high))

        phase_tolerance = self._lle_phase_difference_tolerance(tol)
        residual_limit = max(float(tol), 1.0e-9)

        def solve_from_starts(starts) -> Optional[tuple[float, float, float]]:
            best_local = None
            for start_1, start_2 in starts:
                x_low_start, x_high_start = sorted((start_1, start_2))
                solved = least_squares(
                    residual,
                    np.array([
                        logit(x_low_start),
                        logit(
                            (x_high_start - x_low_start)
                            / (1.0 - x_low_start)
                        ),
                    ], dtype=float),
                    method='trf',
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12,
                    max_nfev=max_iter,
                )
                residual_values = residual(solved.x)
                if not np.all(np.isfinite(residual_values)):
                    continue
                norm = float(np.linalg.norm(residual_values, ord=np.inf))
                x_a_1, x_a_2 = low_high_from_variables(solved.x)
                if not math.isfinite(x_a_1) or not math.isfinite(x_a_2):
                    continue
                if abs(x_a_1 - x_a_2) < phase_tolerance:
                    continue
                x_low, x_high = sorted((x_a_1, x_a_2))
                contains_feed = x_low - 1e-9 <= z_a <= x_high + 1e-9
                best_contains_feed = False
                if best_local is not None:
                    best_low, best_high = sorted((best_local[1], best_local[2]))
                    best_contains_feed = (
                        best_low - 1e-9 <= z_a <= best_high + 1e-9
                    )
                if best_local is None or (
                    contains_feed and not best_contains_feed
                ) or (
                    contains_feed == best_contains_feed and norm < best_local[0]
                ):
                    best_local = (norm, x_a_1, x_a_2)
                    if (
                        contains_feed
                        and norm <= min(residual_limit, 1.0e-9)
                    ):
                        break
            return best_local

        first_starts = (
            adaptive_starts if prefer_adaptive_starts else primary_starts
        )
        retry_starts = (
            primary_starts if prefer_adaptive_starts else adaptive_starts
        )
        best = solve_from_starts(first_starts)
        retry_attempted = False
        if best is None or best[0] > residual_limit:
            retry_best = solve_from_starts(retry_starts)
            retry_attempted = True
            if retry_best is not None and (
                best is None or retry_best[0] < best[0]
            ):
                best = retry_best

        def split_from_candidate(candidate):
            if candidate is None or candidate[0] > residual_limit:
                return None
            _, x_a_1, x_a_2 = candidate
            x_low, x_high = sorted((x_a_1, x_a_2))
            if z_a < x_low - 1e-9 or z_a > x_high + 1e-9:
                return None
            phase1 = phase(x_high)
            phase2 = phase(x_low)
            beta = min(max(
                (z_a - x_high) / (x_low - x_high), 0.0
            ), 1.0)
            if beta <= 1e-10 or beta >= 1.0 - 1e-10:
                return None
            return True, phase1, phase2, beta

        split = split_from_candidate(best)
        if split is not None:
            return split
        if not retry_attempted:
            split = split_from_candidate(solve_from_starts(retry_starts))
            if split is not None:
                return split
        return False, dict(composition), dict(composition), 0.0

    def flash3_TP(self, composition: dict[str, float], T: float, P: float,
                  max_iter: int = 100, tol: float = 1e-9) -> VLLEFlashResult:
        """
        Reference TP flash over vapor, one liquid, and two liquid phases.

        The equilibrium model follows the active activity-coefficient K-value
        implementation, so gamma-Raoult models use ideal vapor and gamma-phi
        variants include the vapor fugacity backend already exposed by
        ``K_values``.  The method is deliberately readable and diagnostic; the
        compiled VLLE backend should match this behavior before becoming a fast
        path.
        """
        z = self._normalize_phase_composition(composition)
        V, x_vle, y_vle = self.flash_TP(z, T, P)
        has_feed_lle, feed_x1, feed_x2, feed_beta = self.liquid_liquid_equilibrium(
            z, T, max_iter=max_iter, tol=max(tol, 1e-9)
        )

        if has_feed_lle:
            feed_x1 = self._normalize_phase_composition(feed_x1)
            feed_x2 = self._normalize_phase_composition(feed_x2)
            if len(self.components) == 2:
                binary = self._binary_invariant_vlle_result(z, T, P, feed_x1, feed_x2)
                if binary is not None:
                    return binary
            else:
                structured = self._structured_vlle_from_seeds(
                    z, T, P, feed_x1, feed_x2, feed_beta, V,
                    status='structured_vlle_feed_lle_seed',
                    max_iter=max_iter,
                    tol=tol,
                )
                if structured is not None:
                    return structured

        if V <= 1e-10:
            if has_feed_lle:
                l1 = max(0.0, min(1.0, 1.0 - float(feed_beta)))
                l2 = max(0.0, min(1.0, float(feed_beta)))
                total = l1 + l2
                if total > 0.0:
                    l1 /= total
                    l2 /= total
                return VLLEFlashResult(
                    phase_count=2,
                    status='lle_only',
                    vapor_fraction=0.0,
                    liquid1_fraction=l1,
                    liquid2_fraction=l2,
                    y=dict(z),
                    x1=feed_x1,
                    x2=feed_x2,
                    residual=self._lle_residual(T, feed_x1, feed_x2),
                    iterations=0,
                )
            return self._ordinary_vlle_result(z, V, x_vle, y_vle, 'single_liquid')

        if V >= 1.0 - 1e-10:
            if has_feed_lle:
                stable_vle = self._stable_vle_from_lle_seeds(
                    z,
                    T,
                    P,
                    (feed_x1, feed_x2),
                    max_iter=max_iter,
                    tol=tol,
                )
                if stable_vle is not None:
                    V, x_vle, y_vle, tpd, gibbs_delta = stable_vle
                    result = self._ordinary_vlle_result(
                        z, V, x_vle, y_vle, 'ordinary_vle'
                    )
                    result.extra.update({
                        'stability_seed': 'feed_lle_tpd',
                        'liquid_tpd': tpd,
                        'reduced_gibbs_delta': gibbs_delta,
                    })
                    return result
            return self._ordinary_vlle_result(z, V, x_vle, y_vle, 'single_vapor')

        liquid_seed = self._normalize_phase_composition(x_vle)
        has_liquid_lle, liquid_x1, liquid_x2, liquid_beta = self.liquid_liquid_equilibrium(
            liquid_seed, T, max_iter=max_iter, tol=max(tol, 1e-9)
        )
        if has_liquid_lle:
            liquid_x1 = self._normalize_phase_composition(liquid_x1)
            liquid_x2 = self._normalize_phase_composition(liquid_x2)
            if len(self.components) == 2:
                binary = self._binary_invariant_vlle_result(z, T, P, liquid_x1, liquid_x2)
                if binary is not None:
                    return binary
            else:
                structured = self._structured_vlle_from_seeds(
                    z, T, P, liquid_x1, liquid_x2, liquid_beta, V,
                    status='structured_vlle_vle_liquid_seed',
                    max_iter=max_iter,
                    tol=tol,
                )
                if structured is not None:
                    return structured

        return self._ordinary_vlle_result(z, V, x_vle, y_vle, 'ordinary_vle')

    def vlle_flash(self, composition: dict[str, float], T: float, P: float,
                   max_iter: int = 100) -> tuple[float, float, dict, dict, dict]:
        result = self.flash3_TP(composition, T, P, max_iter=max_iter)
        liquid2_fraction = result.liquid2_fraction
        return result.vapor_fraction, liquid2_fraction, result.y, result.x1, result.x2

    def _ordinary_vlle_result(self, z: dict[str, float], V: float,
                              x: dict[str, float], y: dict[str, float],
                              status: str) -> VLLEFlashResult:
        V = max(0.0, min(1.0, float(V)))
        phase_count = 1 if V <= 1e-10 or V >= 1.0 - 1e-10 else 2
        x = self._normalize_phase_composition(x)
        y = self._normalize_phase_composition(y)
        return VLLEFlashResult(
            phase_count=phase_count,
            status=status,
            vapor_fraction=V,
            liquid1_fraction=1.0 - V,
            liquid2_fraction=0.0,
            y=y,
            x1=x,
            x2=x,
            residual=0.0,
            iterations=0,
            extra={'overall_composition': dict(z)},
        )

    def _normalize_phase_composition(self, composition: dict[str, float]) -> dict[str, float]:
        raw = {
            comp: max(float(composition.get(comp, 0.0)), 0.0)
            for comp in self.components
        }
        total = sum(raw.values())
        if total <= 0.0:
            return {comp: 1.0 / len(self.components) for comp in self.components}
        return {comp: value / total for comp, value in raw.items()}

    def _structured_vlle_from_seeds(
        self,
        z: dict[str, float],
        T: float,
        P: float,
        x1_seed: dict[str, float],
        x2_seed: dict[str, float],
        beta_seed: float,
        vapor_seed: float,
        status: str,
        max_iter: int,
        tol: float,
    ) -> Optional[VLLEFlashResult]:
        x1 = self._normalize_phase_composition(x1_seed)
        x2 = self._normalize_phase_composition(x2_seed)
        seed_v = max(0.05, min(0.85, float(vapor_seed)))
        liquid_total = 1.0 - seed_v
        seed_l2 = liquid_total * max(0.0, min(1.0, float(beta_seed)))
        seed_l2 = max(1e-4, min(0.9, seed_l2))
        seed_l1 = max(1e-4, liquid_total - seed_l2)
        if seed_l1 + seed_l2 > 0.95:
            scale = 0.95 / (seed_l1 + seed_l2)
            seed_l1 *= scale
            seed_l2 *= scale

        best_failed = None
        previous_vector = None
        previous_residual = None
        last_delta = None
        cooldown = 0
        acceleration_attempts = 0
        acceleration_accepts = 0
        vector_size = 2 * len(self.components)
        inverse_jacobian = [
            [-1.0 if row == col else 0.0 for col in range(vector_size)]
            for row in range(vector_size)
        ]
        for iteration in range(1, max_iter + 1):
            current_vector = self._vlle_phase_pair_vector(x1, x2)
            k1 = self.K_values(T, P, x1)
            k2 = self.K_values(T, P, x2)
            solved = self._solve_three_phase_rr(z, k1, k2, seed_l1, seed_l2)
            ok, V, L1, y, x1_new, x2_new, rr_residual = solved
            if not ok:
                best_failed = rr_residual
                break

            L2 = 1.0 - V - L1
            mapped_vector = self._vlle_phase_pair_vector(x1_new, x2_new)
            residual_vector = [
                mapped_vector[index] - current_vector[index]
                for index in range(vector_size)
            ]
            composition_delta = max(abs(value) for value in residual_vector)
            if composition_delta < tol and rr_residual < max(1e-9, tol):
                phase_distance = sum(abs(x1_new[comp] - x2_new[comp]) for comp in self.components) / len(self.components)
                if phase_distance < 1e-4:
                    return None
                equilibrium_residual = self._vlle_equilibrium_residual(T, P, y, x1_new, x2_new)
                return VLLEFlashResult(
                    phase_count=3,
                    status=status,
                    vapor_fraction=V,
                    liquid1_fraction=L1,
                    liquid2_fraction=L2,
                    y=y,
                    x1=x1_new,
                    x2=x2_new,
                    residual=max(composition_delta, rr_residual, equilibrium_residual),
                    iterations=iteration,
                    extra={
                        'rr_residual': rr_residual,
                        'equilibrium_residual': equilibrium_residual,
                        'acceleration_attempts': acceleration_attempts,
                        'acceleration_accepts': acceleration_accepts,
                    },
                )

            next_vector = mapped_vector
            if cooldown > 0:
                cooldown -= 1
            elif iteration >= 3 and previous_vector is not None and previous_residual is not None:
                if last_delta is not None and composition_delta > 2.5 * max(last_delta, 1e-16):
                    cooldown = 2
                else:
                    acceleration_attempts += 1
                    trial_vector = self._broyden_vlle_phase_pair(
                        current_vector,
                        previous_vector,
                        residual_vector,
                        previous_residual,
                        inverse_jacobian,
                    )
                    ordinary_step = composition_delta
                    trial_step = max(
                        abs(trial_vector[index] - current_vector[index])
                        for index in range(vector_size)
                    )
                    if (
                        trial_step <= 4.0 * max(ordinary_step, 1e-14)
                        and self._valid_vlle_phase_pair_vector(trial_vector)
                    ):
                        next_vector = trial_vector
                        acceleration_accepts += 1

            previous_vector = current_vector
            previous_residual = residual_vector
            last_delta = composition_delta
            x1, x2 = self._vlle_phase_pair_from_vector(next_vector)
            seed_l1 = L1
            seed_l2 = L2

        if best_failed is not None and best_failed < 1e-6:
            return None
        return None

    def _vlle_phase_pair_vector(self, x1: dict[str, float], x2: dict[str, float]) -> list[float]:
        return [float(x1[comp]) for comp in self.components] + [
            float(x2[comp]) for comp in self.components
        ]

    def _vlle_phase_pair_from_vector(self, vector: list[float]) -> tuple[dict[str, float], dict[str, float]]:
        split = len(self.components)
        x1 = {
            comp: max(float(vector[index]), 0.0)
            for index, comp in enumerate(self.components)
        }
        x2 = {
            comp: max(float(vector[split + index]), 0.0)
            for index, comp in enumerate(self.components)
        }
        return self._normalize_phase_composition(x1), self._normalize_phase_composition(x2)

    def _valid_vlle_phase_pair_vector(self, vector: list[float]) -> bool:
        if any(not math.isfinite(value) for value in vector):
            return False
        x1, x2 = self._vlle_phase_pair_from_vector(vector)
        if min(min(x1.values()), min(x2.values())) <= 1e-12:
            return False
        phase_distance = sum(abs(x1[comp] - x2[comp]) for comp in self.components) / len(self.components)
        return phase_distance > 1e-5

    def _broyden_vlle_phase_pair(
        self,
        current_vector: list[float],
        previous_vector: list[float],
        residual_vector: list[float],
        previous_residual: list[float],
        inverse_jacobian: list[list[float]],
    ) -> list[float]:
        size = len(current_vector)
        step = [
            current_vector[index] - previous_vector[index]
            for index in range(size)
        ]
        residual_change = [
            residual_vector[index] - previous_residual[index]
            for index in range(size)
        ]
        denominator = sum(value * value for value in residual_change)
        if denominator > 1e-20:
            jacobian_times_change = [
                sum(inverse_jacobian[row][col] * residual_change[col] for col in range(size))
                for row in range(size)
            ]
            correction = [
                step[index] - jacobian_times_change[index]
                for index in range(size)
            ]
            for row in range(size):
                scale = correction[row] / denominator
                for col in range(size):
                    inverse_jacobian[row][col] += scale * residual_change[col]

        broyden_step = [
            -sum(inverse_jacobian[row][col] * residual_vector[col] for col in range(size))
            for row in range(size)
        ]
        return [
            current_vector[index] + broyden_step[index]
            for index in range(size)
        ]

    def _solve_three_phase_rr(
        self,
        z: dict[str, float],
        k1: dict[str, float],
        k2: dict[str, float],
        seed_l1: float,
        seed_l2: float,
    ) -> tuple[bool, float, float, dict[str, float], dict[str, float], dict[str, float], float]:
        components = self.components
        l1 = max(1e-12, float(seed_l1))
        l2 = max(1e-12, float(seed_l2))
        if l1 + l2 >= 1.0:
            scale = (1.0 - 1e-12) / (l1 + l2)
            l1 *= scale
            l2 *= scale

        def residual(l1_value: float, l2_value: float):
            v_value = 1.0 - l1_value - l2_value
            sum_x1 = 0.0
            sum_x2 = 0.0
            for comp in components:
                k1_i = max(float(k1.get(comp, 1.0)), 1e-14)
                k2_i = max(float(k2.get(comp, 1.0)), 1e-14)
                denom = v_value + l1_value / k1_i + l2_value / k2_i
                if denom <= 0.0:
                    return None
                y_i = z.get(comp, 0.0) / denom
                sum_x1 += y_i / k1_i
                sum_x2 += y_i / k2_i
            return sum_x1 - 1.0, sum_x2 - 1.0

        best = (float('inf'), l1, l2)
        for _ in range(50):
            f = residual(l1, l2)
            if f is None:
                break
            f1, f2 = f
            norm = max(abs(f1), abs(f2))
            if norm < best[0]:
                best = (norm, l1, l2)
            if norm < 1e-12:
                break

            jac00 = jac01 = jac10 = jac11 = 0.0
            v_value = 1.0 - l1 - l2
            valid = True
            for comp in components:
                z_i = z.get(comp, 0.0)
                k1_i = max(float(k1.get(comp, 1.0)), 1e-14)
                k2_i = max(float(k2.get(comp, 1.0)), 1e-14)
                d_l1 = 1.0 / k1_i - 1.0
                d_l2 = 1.0 / k2_i - 1.0
                denom = v_value + l1 / k1_i + l2 / k2_i
                if denom <= 0.0:
                    valid = False
                    break
                scale = -z_i / (denom * denom)
                jac00 += scale * d_l1 / k1_i
                jac01 += scale * d_l2 / k1_i
                jac10 += scale * d_l1 / k2_i
                jac11 += scale * d_l2 / k2_i
            if not valid:
                break
            det = jac00 * jac11 - jac01 * jac10
            if abs(det) < 1e-18:
                break
            step_l1 = (-f1 * jac11 + jac01 * f2) / det
            step_l2 = (jac10 * f1 - jac00 * f2) / det
            damping = 1.0
            accepted = False
            for _line in range(30):
                trial_l1 = l1 + damping * step_l1
                trial_l2 = l2 + damping * step_l2
                if trial_l1 > 0.0 and trial_l2 > 0.0 and trial_l1 + trial_l2 < 1.0:
                    trial = residual(trial_l1, trial_l2)
                    if trial is not None:
                        trial_norm = max(abs(trial[0]), abs(trial[1]))
                        if trial_norm < norm:
                            l1 = trial_l1
                            l2 = trial_l2
                            accepted = True
                            break
                damping *= 0.5
            if not accepted:
                break

        norm, l1, l2 = best
        V = 1.0 - l1 - l2
        if not math.isfinite(norm) or norm > 1e-8 or V <= 1e-10 or l1 <= 1e-10 or l2 <= 1e-10:
            empty = {comp: 0.0 for comp in components}
            return False, V, l1, empty, empty, empty, norm

        y = {}
        x1 = {}
        x2 = {}
        for comp in components:
            k1_i = max(float(k1.get(comp, 1.0)), 1e-14)
            k2_i = max(float(k2.get(comp, 1.0)), 1e-14)
            denom = V + l1 / k1_i + l2 / k2_i
            y[comp] = z.get(comp, 0.0) / denom
            x1[comp] = y[comp] / k1_i
            x2[comp] = y[comp] / k2_i
        return (
            True,
            V,
            l1,
            self._normalize_phase_composition(y),
            self._normalize_phase_composition(x1),
            self._normalize_phase_composition(x2),
            norm,
        )

    def _binary_invariant_vlle_result(
        self,
        z: dict[str, float],
        T: float,
        P: float,
        x1: dict[str, float],
        x2: dict[str, float],
    ) -> Optional[VLLEFlashResult]:
        if len(self.components) != 2:
            return None
        p1 = self.bubble_point_P(x1, T)
        p2 = self.bubble_point_P(x2, T)
        pressure_residual = max(abs(p1 - P), abs(p2 - P))
        if pressure_residual > max(5e-5, 1e-4 * P):
            return None

        k1 = self.K_values(T, P, x1)
        k2 = self.K_values(T, P, x2)
        y1 = self._normalize_phase_composition({
            comp: x1[comp] * k1.get(comp, 1.0)
            for comp in self.components
        })
        y2 = self._normalize_phase_composition({
            comp: x2[comp] * k2.get(comp, 1.0)
            for comp in self.components
        })
        vapor_residual = max(abs(y1[comp] - y2[comp]) for comp in self.components)
        if vapor_residual > 2e-4:
            return None
        y = self._normalize_phase_composition({
            comp: 0.5 * (y1[comp] + y2[comp])
            for comp in self.components
        })

        key = self.components[0]
        denom = x1[key] - x2[key]
        if abs(denom) < 1e-14:
            return None
        slope = (x2[key] - y[key]) / denom
        intercept = (z[key] - x2[key]) / denom

        candidates = [0.0, 1.0]
        if abs(slope) > 1e-14:
            candidates.append(-intercept / slope)
        slope_l2 = -1.0 - slope
        intercept_l2 = 1.0 - intercept
        if abs(slope_l2) > 1e-14:
            candidates.append(-intercept_l2 / slope_l2)

        valid = []
        ordered = sorted(candidates)
        for left, right in zip(ordered, ordered[1:]):
            mid = 0.5 * (left + right)
            if mid < -1e-12 or mid > 1.0 + 1e-12:
                continue
            l1 = intercept + slope * mid
            l2 = 1.0 - mid - l1
            if min(mid, l1, l2) >= -1e-10:
                valid.extend([max(0.0, left), min(1.0, right)])
        valid = [value for value in valid if 0.0 <= value <= 1.0]
        if not valid:
            return None
        v_low = min(valid)
        v_high = max(valid)
        V = 0.5 * (v_low + v_high)
        L1 = intercept + slope * V
        L2 = 1.0 - V - L1
        if min(V, L1, L2) <= 1e-10:
            return None

        return VLLEFlashResult(
            phase_count=3,
            status='binary_invariant_vlle',
            vapor_fraction=V,
            liquid1_fraction=L1,
            liquid2_fraction=L2,
            y=y,
            x1=x1,
            x2=x2,
            residual=max(pressure_residual, vapor_residual),
            iterations=1,
            extra={
                'pressure_residual_bar': pressure_residual,
                'vapor_composition_residual': vapor_residual,
                'vapor_fraction_bounds': (v_low, v_high),
                'phase_amounts_underdetermined': True,
            },
        )

    def _phase_log_fugacities(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str,
    ) -> dict[str, float]:
        comp = self._normalize_phase_composition(composition)
        if phase == 'vapor':
            try:
                phi = self.vapor_fugacity_coefficients(T, P, comp)
            except Exception:
                phi = {component: 1.0 for component in self.components}
            return {
                component: math.log(max(
                    comp.get(component, 0.0)
                    * max(float(phi.get(component, 1.0)), 1e-300)
                    * max(float(P), 1e-300),
                    1e-300,
                ))
                for component in self.components
            }

        gamma = self.activity_coefficients(T, comp)
        reference = self._liquid_fugacity_reference_factors(T, P)
        return {
            component: math.log(max(
                comp.get(component, 0.0)
                * max(float(gamma.get(component, 1.0)), 1e-300)
                * max(float(reference.get(component, 1.0)), 1e-300),
                1e-300,
            ))
            for component in self.components
        }

    def _liquid_tpd_against_vapor(
        self,
        T: float,
        P: float,
        vapor_composition: dict[str, float],
        liquid_seed: dict[str, float],
    ) -> float:
        x = self._normalize_phase_composition(liquid_seed)
        liquid_log_f = self._phase_log_fugacities(T, P, x, 'liquid')
        vapor_log_f = self._phase_log_fugacities(
            T, P, vapor_composition, 'vapor'
        )
        return sum(
            x.get(component, 0.0)
            * (liquid_log_f[component] - vapor_log_f[component])
            for component in self.components
        )

    def _vle_reduced_gibbs(
        self,
        T: float,
        P: float,
        vapor_fraction: float,
        liquid_composition: dict[str, float],
        vapor_composition: dict[str, float],
    ) -> float:
        V = max(0.0, min(1.0, float(vapor_fraction)))
        y = self._normalize_phase_composition(vapor_composition)
        vapor_log_f = self._phase_log_fugacities(T, P, y, 'vapor')
        value = V * sum(
            y.get(component, 0.0) * vapor_log_f[component]
            for component in self.components
        )
        if V >= 1.0 - 1e-12:
            return value
        x = self._normalize_phase_composition(liquid_composition)
        liquid_log_f = self._phase_log_fugacities(T, P, x, 'liquid')
        return value + (1.0 - V) * sum(
            x.get(component, 0.0) * liquid_log_f[component]
            for component in self.components
        )

    def _vle_from_liquid_seed(
        self,
        z: dict[str, float],
        T: float,
        P: float,
        liquid_seed: dict[str, float],
        max_iter: int,
        tol: float,
    ) -> tuple[float, dict[str, float], dict[str, float]] | None:
        K = self.K_values(T, P, self._normalize_phase_composition(liquid_seed))
        V = 0.5
        convergence_tolerance = max(float(tol), 1e-9)

        for _iteration in range(max_iter):
            V = self._rachford_rice_bounded(z, K, V)
            x, y = self._flash_phase_compositions(z, K, V)
            K_new = self.K_values(T, P, x)
            error = max(
                abs(math.log(
                    max(float(K_new.get(component, 1.0)), 1e-300)
                    / max(float(K.get(component, 1.0)), 1e-300)
                ))
                for component in self.components
            )
            K = {
                component: math.sqrt(
                    max(float(K.get(component, 1.0)), 1e-300)
                    * max(float(K_new.get(component, 1.0)), 1e-300)
                )
                for component in self.components
            }
            if error < convergence_tolerance:
                break
        else:
            return None

        V = self._rachford_rice_bounded(z, K, V)
        if V <= 1e-10 or V >= 1.0 - 1e-10:
            return None
        x, y = self._flash_phase_compositions(z, K, V)
        K_check = self.K_values(T, P, x)
        equilibrium_residual = max(
            abs(math.log(
                max(y.get(component, 0.0), 1e-300)
                / max(
                    x.get(component, 0.0)
                    * float(K_check.get(component, 1.0)),
                    1e-300,
                )
            ))
            for component in self.components
        )
        if equilibrium_residual > max(1e-7, 100.0 * convergence_tolerance):
            return None
        return V, x, y

    def _stable_vle_from_lle_seeds(
        self,
        z: dict[str, float],
        T: float,
        P: float,
        liquid_seeds: tuple[dict[str, float], dict[str, float]],
        max_iter: int,
        tol: float,
    ) -> tuple[float, dict[str, float], dict[str, float], float, float] | None:
        if hasattr(self, '_initialize_vdm'):
            return None

        scored_seeds = []
        for seed in liquid_seeds:
            normalized = self._normalize_phase_composition(seed)
            try:
                tpd = self._liquid_tpd_against_vapor(T, P, z, normalized)
            except Exception:
                continue
            if math.isfinite(tpd):
                scored_seeds.append((tpd, normalized))
        if not scored_seeds:
            return None

        tpd, seed = min(scored_seeds, key=lambda item: item[0])
        if tpd >= -max(1e-8, 10.0 * float(tol)):
            return None
        candidate = self._vle_from_liquid_seed(
            z, T, P, seed, max_iter=max_iter, tol=tol
        )
        if candidate is None:
            return None
        V, x, y = candidate
        candidate_gibbs = self._vle_reduced_gibbs(T, P, V, x, y)
        vapor_gibbs = self._vle_reduced_gibbs(T, P, 1.0, z, z)
        gibbs_delta = candidate_gibbs - vapor_gibbs
        if not math.isfinite(gibbs_delta) or gibbs_delta >= -1e-10:
            return None
        return V, x, y, tpd, gibbs_delta

    def _vlle_equilibrium_residual(self, T: float, P: float, y: dict[str, float],
                                   x1: dict[str, float], x2: dict[str, float]) -> float:
        k1 = self.K_values(T, P, x1)
        k2 = self.K_values(T, P, x2)
        residual = 0.0
        for comp in self.components:
            y_i = max(y.get(comp, 0.0), 1e-300)
            x1_i = max(x1.get(comp, 0.0), 1e-300)
            x2_i = max(x2.get(comp, 0.0), 1e-300)
            residual = max(
                residual,
                abs(math.log(y_i / max(x1_i * k1.get(comp, 1.0), 1e-300))),
                abs(math.log(y_i / max(x2_i * k2.get(comp, 1.0), 1e-300))),
            )
        return residual

    def _lle_residual(self, T: float, x1: dict[str, float], x2: dict[str, float]) -> float:
        gamma1 = self.activity_coefficients(T, x1)
        gamma2 = self.activity_coefficients(T, x2)
        residual = 0.0
        for comp in self.components:
            a1 = max(x1.get(comp, 0.0) * gamma1.get(comp, 1.0), 1e-300)
            a2 = max(x2.get(comp, 0.0) * gamma2.get(comp, 1.0), 1e-300)
            residual = max(residual, abs(math.log(a1 / a2)))
        return residual

    def _flash3_mixture_enthalpy(self, T: float, P: float,
                                 result: VLLEFlashResult) -> float:
        V = max(0.0, min(1.0, result.vapor_fraction))
        L1 = max(0.0, result.liquid1_fraction)
        L2 = max(0.0, result.liquid2_fraction)
        total = V + L1 + L2
        if total <= 0.0:
            return self.mixture_enthalpy(result.x1, T, 0.0, result.x1, None, P)
        V /= total
        L1 /= total
        L2 /= total
        h_vapor = self.mixture_enthalpy(result.y, T, 1.0, None, result.y, P)
        h_liquid1 = self.mixture_enthalpy(result.x1, T, 0.0, result.x1, None, P)
        h_liquid2 = self.mixture_enthalpy(result.x2, T, 0.0, result.x2, None, P)
        return V * h_vapor + L1 * h_liquid1 + L2 * h_liquid2

    def _flash3_mixture_entropy(self, T: float, P: float,
                                result: VLLEFlashResult) -> float:
        V = max(0.0, min(1.0, result.vapor_fraction))
        L1 = max(0.0, result.liquid1_fraction)
        L2 = max(0.0, result.liquid2_fraction)
        total = V + L1 + L2
        if total <= 0.0:
            return self.mixture_entropy(result.x1, T, 0.0, result.x1, None, P)
        V /= total
        L1 /= total
        L2 /= total
        s_vapor = self.mixture_entropy(result.y, T, 1.0, None, result.y, P)
        s_liquid1 = self.mixture_entropy(result.x1, T, 0.0, result.x1, None, P)
        s_liquid2 = self.mixture_entropy(result.x2, T, 0.0, result.x2, None, P)
        return V * s_vapor + L1 * s_liquid1 + L2 * s_liquid2

    def _with_binary_vapor_fraction(
        self,
        result: VLLEFlashResult,
        z: dict[str, float],
        vapor_fraction: float,
    ) -> VLLEFlashResult:
        bounds = result.extra.get('vapor_fraction_bounds') if result.extra else None
        if bounds is None:
            return result
        V = max(bounds[0], min(bounds[1], float(vapor_fraction)))
        key = self.components[0]
        denom = result.x1[key] - result.x2[key]
        if abs(denom) < 1e-14:
            return result
        L1 = (z[key] - V * result.y[key] - (1.0 - V) * result.x2[key]) / denom
        L2 = 1.0 - V - L1
        extra = dict(result.extra)
        extra['selected_vapor_fraction'] = V
        return VLLEFlashResult(
            phase_count=result.phase_count,
            status=result.status,
            vapor_fraction=V,
            liquid1_fraction=L1,
            liquid2_fraction=L2,
            y=dict(result.y),
            x1=dict(result.x1),
            x2=dict(result.x2),
            residual=result.residual,
            iterations=result.iterations,
            extra=extra,
        )

    def _binary_invariant_at_pressure(
        self,
        z: dict[str, float],
        P: float,
        T_guess: float,
        max_iter: int,
    ) -> tuple[float, VLLEFlashResult] | None:
        if len(self.components) != 2:
            return None
        try:
            T = self.bubble_point_T_vlle(z, P, T_guess=T_guess, max_iter=max_iter)
            result = self.flash3_TP(z, T, P, max_iter=max_iter)
        except Exception:
            return None
        if result.status != 'binary_invariant_vlle':
            return None
        return T, result

    def _binary_invariant_at_temperature(
        self,
        z: dict[str, float],
        T: float,
        max_iter: int,
    ) -> tuple[float, VLLEFlashResult] | None:
        if len(self.components) != 2:
            return None
        try:
            has_lle, x1, x2, _beta = self.liquid_liquid_equilibrium(
                z, T, max_iter=max_iter, tol=1e-8
            )
        except Exception:
            return None
        if not has_lle:
            return None
        try:
            P = 0.5 * (
                self.bubble_point_P(self._normalize_phase_composition(x1), T)
                + self.bubble_point_P(self._normalize_phase_composition(x2), T)
            )
            if not math.isfinite(P) or P <= 0.0:
                return None
            result = self.flash3_TP(z, T, P, max_iter=max_iter)
        except Exception:
            return None
        if result.status != 'binary_invariant_vlle':
            return None
        return P, result

    @staticmethod
    def _bounds_contain(bounds, value: float, tolerance: float = 1e-10) -> bool:
        if bounds is None:
            return False
        low, high = bounds
        return low - tolerance <= float(value) <= high + tolerance

    def _with_binary_enthalpy(
        self,
        result: VLLEFlashResult,
        z: dict[str, float],
        T: float,
        P: float,
        H_target: float,
    ) -> VLLEFlashResult | None:
        bounds = result.extra.get('vapor_fraction_bounds') if result.extra else None
        if bounds is None:
            return None

        low, high = bounds
        low_result = self._with_binary_vapor_fraction(result, z, low)
        high_result = self._with_binary_vapor_fraction(result, z, high)
        h_low = self._flash3_mixture_enthalpy(T, P, low_result)
        h_high = self._flash3_mixture_enthalpy(T, P, high_result)
        target = float(H_target)
        h_min = min(h_low, h_high)
        h_max = max(h_low, h_high)
        tolerance = max(1e-5, abs(target) * 1e-9)
        if target < h_min - tolerance or target > h_max + tolerance:
            return None
        if abs(h_high - h_low) <= 1e-12:
            V = 0.5 * (low + high)
        else:
            fraction = (target - h_low) / (h_high - h_low)
            V = low + fraction * (high - low)
        return self._with_binary_vapor_fraction(result, z, V)

    def _with_binary_entropy(
        self,
        result: VLLEFlashResult,
        z: dict[str, float],
        T: float,
        P: float,
        S_target: float,
    ) -> VLLEFlashResult | None:
        bounds = result.extra.get('vapor_fraction_bounds') if result.extra else None
        if bounds is None:
            return None

        low, high = bounds
        low_result = self._with_binary_vapor_fraction(result, z, low)
        high_result = self._with_binary_vapor_fraction(result, z, high)
        s_low = self._flash3_mixture_entropy(T, P, low_result)
        s_high = self._flash3_mixture_entropy(T, P, high_result)
        target = float(S_target)
        s_min = min(s_low, s_high)
        s_max = max(s_low, s_high)
        tolerance = max(1e-7, abs(target) * 1e-9)
        if target < s_min - tolerance or target > s_max + tolerance:
            return None
        if abs(s_high - s_low) <= 1e-14:
            V = 0.5 * (low + high)
        else:
            fraction = (target - s_low) / (s_high - s_low)
            V = low + fraction * (high - low)
        return self._with_binary_vapor_fraction(result, z, V)

    def _solve_temperature_residual(
        self,
        residual,
        T_guess: float,
        centers: tuple[float, ...],
        label: str,
        exact_tolerance: float = 1e-8,
    ) -> float:
        clean_centers = []
        for value in centers:
            try:
                T = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(T) and T > 0.0:
                clean_centers.append(max(1.0, min(5000.0, T)))
        if not clean_centers:
            clean_centers.append(max(1.0, min(5000.0, float(T_guess))))

        residual_cache = {}

        def cached_residual(T: float) -> float:
            key = round(float(T), 10)
            if key not in residual_cache:
                residual_cache[key] = float(residual(float(T)))
            return residual_cache[key]

        for center in clean_centers:
            try:
                f_center = cached_residual(center)
                if math.isfinite(f_center) and abs(f_center) <= exact_tolerance:
                    return center
            except Exception:
                pass
            for width in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0):
                low = max(1.0, center - width)
                high = min(5000.0, center + width)
                if high <= low:
                    continue
                try:
                    f_low = cached_residual(low)
                    if math.isfinite(f_low) and abs(f_low) <= exact_tolerance:
                        return low
                    f_high = cached_residual(high)
                    if math.isfinite(f_high) and abs(f_high) <= exact_tolerance:
                        return high
                    if math.isfinite(f_low) and math.isfinite(f_high) and f_low * f_high < 0.0:
                        return brentq(cached_residual, low, high, xtol=1e-7, rtol=1e-9, maxiter=80)
                except Exception:
                    continue

        grid = self._temperature_grid(clean_centers)
        return self._solve_grid_scalar_residual(
            cached_residual,
            grid,
            preferred=clean_centers[0],
            label=label,
            exact_tolerance=exact_tolerance,
        )

    def _solve_bounded_scalar_residual(
        self,
        residual,
        low: float,
        high: float,
        preferred: float,
        label: str,
        exact_tolerance: float = 1e-8,
    ) -> float:
        low = float(low)
        high = float(high)
        if high < low:
            low, high = high, low
        points = sorted(set([
            low,
            high,
            max(low, min(high, float(preferred))),
            *[low + (high - low) * index / 12.0 for index in range(1, 12)],
        ]))
        return self._solve_grid_scalar_residual(
            residual,
            points,
            preferred=preferred,
            label=label,
            exact_tolerance=exact_tolerance,
        )

    def _solve_grid_scalar_residual(
        self,
        residual,
        grid: list[float],
        preferred: float,
        label: str,
        exact_tolerance: float = 1e-8,
    ) -> float:
        evaluated = []
        for value in sorted(set(float(item) for item in grid if math.isfinite(float(item)))):
            if value <= 0.0:
                continue
            try:
                f = float(residual(value))
            except Exception:
                continue
            if not math.isfinite(f):
                continue
            if abs(f) <= exact_tolerance:
                return value
            evaluated.append((value, f))

        brackets = []
        for left, right in zip(evaluated, evaluated[1:]):
            if left[1] * right[1] < 0.0:
                brackets.append((left[0], right[0]))
        if brackets:
            low, high = min(
                brackets,
                key=lambda bracket: (
                    abs(0.5 * (bracket[0] + bracket[1]) - preferred),
                    bracket[1] - bracket[0],
                ),
            )
            return brentq(residual, low, high, xtol=1e-7, rtol=1e-9, maxiter=100)

        raise ThermodynamicsError(f"Could not bracket {label}")

    def _temperature_grid(self, centers: list[float]) -> list[float]:
        grid = [
            80.0, 120.0, 150.0, 200.0, 250.0, 298.15, 350.0,
            400.0, 500.0, 650.0, 800.0, 1000.0, 1500.0,
        ]
        for center in centers:
            grid.extend([
                center,
                center - 80.0,
                center - 40.0,
                center - 20.0,
                center - 10.0,
                center + 10.0,
                center + 20.0,
                center + 40.0,
                center + 80.0,
            ])
        for props in getattr(self, 'props', {}).values():
            for value in (getattr(props, 'Tb', None), getattr(props, 'Tc', None)):
                if value:
                    grid.extend([0.55 * float(value), float(value), 1.25 * float(value)])
        return sorted(set(max(1.0, min(5000.0, float(T))) for T in grid))

    def _pressure_grid(self, centers: list[float]) -> list[float]:
        grid = [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0, P_REF, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        for center in centers:
            if center <= 0.0 or not math.isfinite(center):
                continue
            for factor in (0.05, 0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0, 20.0):
                grid.append(max(1e-8, center * factor))
        return sorted(set(float(P) for P in grid if P > 0.0 and math.isfinite(P)))

    def generate_Txy_data(self, comp1: str, comp2: str, P: float,
                          n_points: int = 50) -> dict:
        """
        Generate T-x-y diagram data for a binary system (VLLE capable).
        
        Checks for liquid-liquid equilibrium and includes three-phase
        (VLLE) region if present.
        
        Args:
            comp1: First component (more volatile)
            comp2: Second component (less volatile)
            P: Pressure [bar]
            n_points: Number of points
            
        Returns:
            Dict with 'x', 'y', 'T_bubble', 'T_dew', 'azeotrope', 
            and optionally 'lle_region' data
        """
        x_data = []
        y_data = []
        T_bubble = []
        T_dew = []
        azeotrope = None
        lle_region = None
        
        # First check if system has LLE at a reference temperature
        # (use midpoint temperature estimate)
        T_ref = 298.15  # 25°C
        try:
            has_lle, x1_lle, x2_lle, beta = self.liquid_liquid_equilibrium(
                {comp1: 0.5, comp2: 0.5}, T_ref
            )
            if has_lle and x1_lle and x2_lle:
                # System has LLE - find the VLLE temperature
                # (where LLE phases are in equilibrium with vapor)
                lle_region = {
                    'exists': True,
                    'x1_phase1': x1_lle.get(comp1, 0),
                    'x1_phase2': x2_lle.get(comp1, 0),
                    'T_': None  # Will be found if VLLE exists
                }
        except:
            has_lle = False
        
        for i in range(n_points + 1):
            x1 = i / n_points
            x2 = 1 - x1
            
            if x1 < 0.001:
                x1 = 0.001
                x2 = 0.999
            if x1 > 0.999:
                x1 = 0.999
                x2 = 0.001
            
            composition = {comp1: x1, comp2: x2}
            
            # Bubble point
            try:
                T_bub = self.bubble_point_T(composition, P)
            except:
                # Skip this point if bubble point fails
                continue
            
            # Vapor composition at bubble point. Use the model-level K-values
            # so gamma-phi methods include vapor fugacity corrections.
            K = self.K_values(T_bub, P, composition)
            y1 = x1 * K.get(comp1, 1.0)
            y2 = x2 * K.get(comp2, 1.0)
            y_sum = y1 + y2
            if y_sum > 0:
                y1 /= y_sum
            
            # Dew point for same vapor composition
            y_comp = {comp1: y1, comp2: 1 - y1}
            try:
                T_dw = self.dew_point_T(y_comp, P)
            except:
                T_dw = T_bub  # Fallback
            
            x_data.append(x1)
            y_data.append(y1)
            T_bubble.append(T_bub - 273.15)  # Convert to °C
            T_dew.append(T_dw - 273.15)
            
            # Check for azeotrope (x ≈ y)
            if i > 0 and azeotrope is None and len(x_data) >= 2:
                prev_diff = x_data[-2] - y_data[-2]
                curr_diff = x1 - y1
                if prev_diff * curr_diff < 0:  # Sign change
                    # Interpolate
                    frac = abs(prev_diff) / (abs(prev_diff) + abs(curr_diff))
                    x_az = x_data[-2] + frac * (x1 - x_data[-2])
                    T_az = T_bubble[-2] + frac * (T_bubble[-1] - T_bubble[-2])
                    azeotrope = {
                        'x': round(x_az, 4),
                        'T': round(T_az, 2),
                    }
        
        # Classify azeotrope type using the full T_bubble endpoints
        if azeotrope is not None and len(T_bubble) >= 2:
            T_pure_comp1 = T_bubble[-1]  # x → 1 (nearly pure component 1)
            T_pure_comp2 = T_bubble[0]   # x → 0 (nearly pure component 2)
            T_min_pure = min(T_pure_comp1, T_pure_comp2)
            T_max_pure = max(T_pure_comp1, T_pure_comp2)
            if azeotrope['T'] < T_min_pure:
                azeotrope['type'] = 'minimum'
            elif azeotrope['T'] > T_max_pure:
                azeotrope['type'] = 'maximum'
            else:
                azeotrope['type'] = 'none'  # Should not happen

        result = {
            'x': x_data,
            'y': y_data,
            'T_bubble': T_bubble,
            'T_dew': T_dew,
            'azeotrope': azeotrope,
            'components': [comp1, comp2],
            'pressure_bar': P
        }
        
        # Include LLE information if detected
        if lle_region:
            result['lle_region'] = lle_region
        
        return result
    
    def generate_Pxy_data(self, comp1: str, comp2: str, T: float,
                          n_points: int = 50) -> dict:
        """
        Generate P-x-y diagram data for a binary system at constant T.
        """
        x_data = []
        y_data = []
        P_bubble = []
        P_dew = []
        
        for i in range(n_points + 1):
            x1 = i / n_points
            if x1 < 0.001: x1 = 0.001
            if x1 > 0.999: x1 = 0.999
            
            composition = {comp1: x1, comp2: 1 - x1}
            
            # Bubble point pressure
            P_bub = self.bubble_point_P(composition, T)

            # Vapor composition from model K-values so gamma-phi and VDM
            # vapor corrections stay consistent with the bubble curve.
            K = self.K_values(T, P_bub, composition)
            y1 = x1 * K.get(comp1, 1.0)
            
            # Dew point pressure
            y_comp = {comp1: y1, comp2: 1 - y1}
            P_dw = self.dew_point_P(y_comp, T)
            
            x_data.append(x1)
            y_data.append(y1)
            P_bubble.append(P_bub)
            P_dew.append(P_dw)
        
        return {
            'x': x_data,
            'y': y_data,
            'P_bubble': P_bubble,
            'P_dew': P_dew,
            'components': [comp1, comp2],
            'temperature_C': T - 273.15
        }
    
    def generate_xy_data(self, comp1: str, comp2: str, T: float = None, P: float = None,
                         n_points: int = 50) -> dict:
        """
        Generate x-y diagram data (equilibrium curve).
        
        Specify either T (isobaric) or P (isothermal).
        Returns equilibrium curve y vs x, plus diagonal y=x line.
        """
        if T is None and P is None:
            raise ValueError("Must specify either T or P")
        
        x_data = []
        y_data = []
        
        for i in range(n_points + 1):
            x1 = i / n_points
            if x1 < 0.001: x1 = 0.001
            if x1 > 0.999: x1 = 0.999
            
            composition = {comp1: x1, comp2: 1 - x1}
            
            if P is not None:
                # Isobaric - find bubble point T
                T_calc = self.bubble_point_T(composition, P)
            else:
                T_calc = T
            
            if P is not None:
                K = self.K_values(T_calc, P, composition)
                y1 = x1 * K.get(comp1, 1.0)
            else:
                P_bub = self.bubble_point_P(composition, T)
                K = self.K_values(T_calc, P_bub, composition)
                y1 = x1 * K.get(comp1, 1.0)
            
            x_data.append(x1)
            y_data.append(min(1.0, max(0.0, y1)))
        
        # Generate diagonal (y=x) line for reference
        diagonal_x = [0.0, 1.0]
        diagonal_y = [0.0, 1.0]
        
        return {
            'x': x_data,
            'y': y_data,
            'diagonal_x': diagonal_x,
            'diagonal_y': diagonal_y,
            'components': [comp1, comp2],
            'pressure_bar': P,
            'temperature_C': (T - 273.15) if T else None
        }


class VaporDimerizationActivityMixin:
    """Gamma-VDM VLE correction for user-declared or recognized associators."""

    def _initialize_vdm(self) -> None:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..vapor_dimerization import (
                GENERIC_MONOCARBOXYLIC_DELTA_H_J_MOL,
                GENERIC_MONOCARBOXYLIC_DELTA_S_J_MOL_K,
                VaporDimerizationModel,
                get_dimerization_params,
            )
        else:
            from vapor_dimerization import (
                GENERIC_MONOCARBOXYLIC_DELTA_H_J_MOL,
                GENERIC_MONOCARBOXYLIC_DELTA_S_J_MOL_K,
                VaporDimerizationModel,
                get_dimerization_params,
            )

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..compiled_vdm import compile_vdm_kernels
            else:
                from compiled_vdm import compile_vdm_kernels

            compile_vdm_kernels()
        except Exception:
            pass

        self._vdm_models = {}
        self._vdm_multi_model_cache = {}
        self._vdm_cross_residual_overrides = {}
        for record in getattr(self, 'interaction_overrides', ()):
            if str(record.get('model', '')).upper() != 'VDM':
                continue
            comp1 = record.get('component1')
            comp2 = record.get('component2')
            if not comp1 or not comp2 or comp1 == comp2:
                continue
            self._vdm_cross_residual_overrides[
                tuple(sorted((str(comp1), str(comp2))))
            ] = {
                'delta_H_residual_J_per_mol': float(
                    record['delta_H_residual_J_per_mol']
                ),
                'delta_S_residual_J_per_mol_K': float(
                    record['delta_S_residual_J_per_mol_K']
                ),
                'source': record.get('comment') or 'PFD VDM cross override',
            }
        for comp in self.components:
            identifiers = []
            props = self.props.get(comp)
            if props is not None:
                # Process symbols and user-component display names are local
                # aliases, never chemical identity inputs.
                identifiers.extend([props.CAS, props.formula])
            params = (
                getattr(props, 'vapor_dimerization', None)
                if props is not None else None
            )
            if params is None:
                for identifier in identifiers:
                    if not identifier:
                        continue
                    params = get_dimerization_params(str(identifier))
                    if params is not None:
                        break
            if params is None and props is not None:
                try:
                    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                        from ..vapor_dimerization import is_monocarboxylic_acid
                    else:
                        from vapor_dimerization import is_monocarboxylic_acid
                    structural_identifier = props.CAS or props.formula or ''
                    if is_monocarboxylic_acid(
                        structural_identifier,
                        smiles=getattr(props, 'smiles', None),
                    ):
                        params = {
                            'delta_H_J_per_mol': (
                                GENERIC_MONOCARBOXYLIC_DELTA_H_J_MOL
                            ),
                            'delta_S_J_per_mol_K': (
                                GENERIC_MONOCARBOXYLIC_DELTA_S_J_MOL_K
                            ),
                            'source': 'estimated (monocarboxylic acid default)',
                        }
                except Exception:
                    params = None
            if params is None:
                continue
            self._vdm_models[comp] = VaporDimerizationModel(
                monomer=comp,
                dimer=f"({comp})2",
                delta_H=float(params["delta_H_J_per_mol"]),
                delta_S=float(params["delta_S_J_per_mol_K"]),
            )
        for comp1, comp2 in self._vdm_cross_residual_overrides:
            missing = [
                comp for comp in (comp1, comp2)
                if comp not in self._vdm_models
            ]
            if missing:
                raise ThermodynamicsError(
                    f"VDM cross override for {comp1}/{comp2} requires active "
                    "homodimer parameters for both components; missing "
                    + ', '.join(missing)
                    + ". Add component VDM bundles or use recognized associators."
                )
        self._vdm_phi_sat_cache = {}
        self._vdm_assoc_reference_cache = {}
        self._vdm_assoc_reference_cp_cache = {}
        self._vdm_viscosity_warning_emitted = False

    def _vdm_vapor_association_state(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
    ) -> Optional[dict]:
        """Return the physical association state for a nominal vapor mixture."""
        nominal = self._normalized_positive_composition(composition)
        _acid, model = self._active_vdm_model(nominal)
        if model is None:
            return None
        return model.association_state(float(T), float(P), nominal, rk_model=None)

    @staticmethod
    def _vdm_physical_moles_per_nominal(state: Optional[dict]) -> float:
        """Physical vapor molecule count per nominal monomer-equivalent mole."""
        if state is None:
            return 1.0
        extent = sum(
            max(float(value), 0.0)
            for value in (state.get('extents') or {}).values()
        )
        factor = 1.0 - extent
        if not math.isfinite(factor) or factor <= 0.0 or factor > 1.0 + 1.0e-10:
            raise ThermodynamicsError(
                "VDM returned an invalid physical-moles-per-nominal factor"
            )
        return min(factor, 1.0)

    def vapor_molar_volume_for_density(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
    ) -> float:
        """VDM vapor volume [m3/nominal kmol] from the physical molecule count."""
        ideal_physical_volume = super().vapor_molar_volume_for_density(
            T,
            P,
            composition,
        )
        state = self._vdm_vapor_association_state(T, P, composition)
        return (
            self._vdm_physical_moles_per_nominal(state)
            * ideal_physical_volume
        )

    def mixture_viscosity(
        self,
        composition: dict[str, float],
        T: float,
        P: float,
        vapor_fraction: float = 1.0,
        x: Optional[dict] = None,
        y: Optional[dict] = None,
    ) -> float:
        """Use nominal-component vapor viscosity and warn only when requested."""
        V = max(0.0, min(1.0, float(vapor_fraction)))
        if V > 0.001 and not self._vdm_viscosity_warning_emitted:
            state = self._vdm_vapor_association_state(T, P, y or composition)
            association = 1.0 - self._vdm_physical_moles_per_nominal(state)
            if association > 1.0e-12:
                self.add_warning(
                    "VDM vapor viscosity uses nominal-component mixture viscosity; "
                    "internal physical dimer species are not included in the "
                    "transport mixing rule."
                )
                self._vdm_viscosity_warning_emitted = True
        return super().mixture_viscosity(
            composition,
            T,
            P,
            vapor_fraction,
            x=x,
            y=y,
        )

    def _active_vdm_model(self, composition: dict[str, float]):
        active = [
            comp for comp in self._vdm_models
            if max(float(composition.get(comp, 0.0)), 0.0) > 1e-12
        ]
        if not active:
            return None, None
        if len(active) > 1:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..vapor_dimerization import (
                                MultiVaporDimerizationModel,
                                get_cross_dimerization_residual,
                            )
            else:
                from vapor_dimerization import (
                                MultiVaporDimerizationModel,
                                get_cross_dimerization_residual,
                            )

            key = tuple(active)
            cached = self._vdm_multi_model_cache.get(key)
            if cached is not None:
                return None, cached
            residuals = {}
            for index, comp_i in enumerate(active):
                for comp_j in active[index + 1:]:
                    residual = self._vdm_cross_residual_overrides.get(
                        tuple(sorted((comp_i, comp_j)))
                    )
                    if residual is None:
                        residual = get_cross_dimerization_residual(comp_i, comp_j)
                    residuals[(comp_i, comp_j)] = residual
            model = MultiVaporDimerizationModel(
                {comp: self._vdm_models[comp] for comp in active},
                residuals,
            )
            self._vdm_multi_model_cache[key] = model
            return None, model
        comp = active[0]
        return comp, self._vdm_models[comp]

    def _vdm_phi_sat(self, comp: str, T: float) -> float:
        model = self._vdm_models.get(comp)
        if model is None:
            return 1.0
        key = (comp, float(T))
        cached = self._vdm_phi_sat_cache.get(key)
        if cached is not None:
            return cached
        Psat = self.Psat(comp, T)
        pure_y = {component: 0.0 for component in self.components}
        pure_y[comp] = 1.0
        phi = model.fugacity_coefficients(T, Psat, pure_y, rk_model=None)
        value = max(phi.get(comp, 1.0), 1e-12)
        if len(self._vdm_phi_sat_cache) > 20000:
            self._vdm_phi_sat_cache.clear()
        self._vdm_phi_sat_cache[key] = value
        return value

    def _vdm_single_acid_phi_sat(self, comp: str, T: float) -> float:
        """Pure saturated monomer fugacity correction for one ideal associator."""
        model = self._vdm_models.get(comp)
        if model is None:
            return 1.0
        key = (comp, float(T))
        cached = self._vdm_phi_sat_cache.get(key)
        if cached is not None:
            return cached

        Psat = self.Psat(comp, T)
        alpha = model._alpha_from_equilibrium(model.K_eq(T) * Psat, 1.0)
        value = max((1.0 - alpha) / max(1.0 - 0.5 * alpha, 1e-30), 1e-12)
        if len(self._vdm_phi_sat_cache) > 20000:
            self._vdm_phi_sat_cache.clear()
        self._vdm_phi_sat_cache[key] = value
        return value

    def fugacity_coefficients(self, T: float, P: float,
                               composition: dict[str, float],
                               phase: str = 'vapor') -> dict[str, float]:
        if phase.lower().startswith('l'):
            return {comp: 1.0 for comp in self.components}
        _, model = self._active_vdm_model(composition)
        if model is None:
            return {comp: 1.0 for comp in self.components}
        return model.fugacity_coefficients(T, P, composition, rk_model=None)

    def vapor_fugacity_coefficients(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
    ) -> dict[str, float]:
        """VDM nominal-component fugacity coefficients for equilibrium audits."""
        return self.fugacity_coefficients(T, P, composition, phase='vapor')

    def _vdm_vapor_terms_closure(
        self,
        T: float,
        P: float,
        target_fugacity: dict[str, float],
        components,
    ) -> Optional[dict]:
        """Return a fused multi-acid shared-vapor closure when available."""
        total = sum(max(float(target_fugacity.get(comp, 0.0)), 0.0) for comp in components)
        if total <= 0.0:
            return None
        initial_vapor = {
            comp: max(float(target_fugacity.get(comp, 0.0)), 0.0) / total
            for comp in components
        }
        _acid, model = self._active_vdm_model(initial_vapor)
        compiled_closure = getattr(model, 'compiled_vapor_closure', None)
        if not callable(compiled_closure):
            return None
        return compiled_closure(
            T,
            P,
            components,
            {
                comp: max(float(target_fugacity.get(comp, 0.0)), 0.0)
                / max(float(P), 1e-300)
                for comp in components
            },
            max_iter=12,
            tol=1e-10,
            phi_floor=1e-8,
        )

    def _vapor_enthalpy_from_association_state(
        self,
        composition: dict[str, float],
        T: float,
        association_state: dict,
    ) -> float:
        """Use an existing VDM closure state for nominal vapor enthalpy."""
        ideal = sum(
            fraction * self.enthalpy_ideal_gas(comp, T)
            for comp, fraction in composition.items()
        ) * 1000.0
        return ideal + float(association_state.get('association_enthalpy', 0.0))

    def _liquid_fugacity_reference_factors(
        self,
        T: float,
        P: float,
    ) -> dict[str, float]:
        """VDM saturated-monomer references used by the VDM K-value equations."""
        return {
            comp: self._vdm_phi_sat(comp, T) * self.Psat(comp, T)
            for comp in self.components
        }

    def _vdm_pure_saturated_association_enthalpy(self, comp: str, T: float) -> float:
        """Pure saturated-vapor association enthalpy [kJ/kmol nominal component]."""
        cache_key = (comp, float(T))
        cached = self._vdm_assoc_reference_cache.get(cache_key)
        if cached is not None:
            return cached

        value = 0.0
        model = self._vdm_models.get(comp)
        if model is not None:
            try:
                Psat = self.Psat(comp, T)
                state = model.association_state(T, Psat, {comp: 1.0}, rk_model=None)
                value = float(state.get("association_enthalpy", 0.0))
            except Exception as exc:
                self.add_warning(
                    f"Could not calculate pure saturated VDM association enthalpy "
                    f"reference for {self._component_label(comp)} at T={T:.2f} K; "
                    f"using zero reference fallback where unavailable ({exc})."
                )
                value = 0.0

        if len(self._vdm_assoc_reference_cache) > 20000:
            self._vdm_assoc_reference_cache.clear()
        self._vdm_assoc_reference_cache[cache_key] = value
        return value

    def _vdm_pure_saturated_association_cp(self, comp: str, T: float) -> float:
        """Temperature derivative of the pure saturated association reference."""
        cache_key = (comp, float(T))
        cached = self._vdm_assoc_reference_cp_cache.get(cache_key)
        if cached is not None:
            return cached
        dT = max(0.05, 1.0e-4 * float(T))
        T_low = max(1.0, float(T) - dT)
        T_high = float(T) + dT
        value = (
            self._vdm_pure_saturated_association_enthalpy(comp, T_high)
            - self._vdm_pure_saturated_association_enthalpy(comp, T_low)
        ) / (T_high - T_low)
        if not math.isfinite(value):
            raise ThermodynamicsError(
                f"VDM pure saturated association Cp reference is non-finite for "
                f"{self._component_label(comp)}"
            )
        if len(self._vdm_assoc_reference_cp_cache) > 20000:
            self._vdm_assoc_reference_cp_cache.clear()
        self._vdm_assoc_reference_cp_cache[cache_key] = value
        return value

    def enthalpy_liquid(self, comp: str, T: float) -> float:
        """Liquid enthalpy with the apparent-Hvap association reference."""
        value = super().enthalpy_liquid(comp, T)
        if comp in self._vdm_models:
            value += (
                self._vdm_pure_saturated_association_enthalpy(comp, T)
                / 1000.0
            )
        return value

    def Cp_liquid(self, comp: str, T: float) -> float:
        """Liquid Cp consistent with the shifted VDM liquid enthalpy."""
        value = super().Cp_liquid(comp, T)
        if comp in self._vdm_models:
            # kJ/kmol-K and J/mol-K are numerically identical.
            value += self._vdm_pure_saturated_association_cp(comp, T)
        return value

    def _vapor_residual_enthalpy(self, composition: dict[str, float], T: float, P: float) -> float:
        """
        Actual VDM vapor association enthalpy [kJ/kmol nominal mixture].

        Ideal-gas formation enthalpies use the zero-pressure monomer reference.
        The pure saturated association reference needed to preserve apparent
        Hvap therefore belongs on the liquid enthalpy, not in this vapor
        departure.
        """
        _, model = self._active_vdm_model(composition)
        if model is None:
            return super()._vapor_residual_enthalpy(composition, T, P)

        nominal = {
            comp: max(float(composition.get(comp, 0.0)), 0.0)
            for comp in self.components
        }
        total = sum(nominal.values())
        if total <= 0.0:
            return 0.0
        nominal = {comp: value / total for comp, value in nominal.items()}

        try:
            scalar_enthalpy = getattr(
                model, 'compiled_association_enthalpy', None
            )
            if callable(scalar_enthalpy):
                value = scalar_enthalpy(
                    T,
                    P,
                    self.components,
                    nominal,
                )
                if value is not None:
                    return float(value)
            state = model.association_state(T, P, nominal, rk_model=None)
            return float(state.get("association_enthalpy", 0.0))
        except Exception as exc:
            self.add_warning(
                f"Could not calculate VDM vapor association residual enthalpy for "
                f"{self.__class__.__name__}; using zero residual enthalpy fallback "
                f"where unavailable ({exc})."
            )
            return 0.0

    def _single_acid_vdm_K_values(
        self,
        acid: str,
        model,
        T: float,
        P: float,
        x: dict[str, float],
    ) -> dict[str, float]:
        """Scalar K-value solve for one VDM associator with ideal physical fugacities."""
        cache_key = self._k_values_cache_key('gamma_vdm_single_ideal', T, P, x)
        cached = self._get_cached_k_values(cache_key)
        if cached is not None:
            return cached

        gamma = self.activity_coefficients(T, x)
        base = {}
        for comp in self.components:
            phi_sat = self._vdm_single_acid_phi_sat(comp, T)
            base[comp] = max(
                1e-12,
                gamma.get(comp, 1.0) * phi_sat * self.Psat(comp, T) / max(P, 1e-12),
            )

        acid_x_base = x.get(acid, 0.0) * base.get(acid, 1e-12)
        inert_x_base = sum(
            x.get(comp, 0.0) * base.get(comp, 1e-12)
            for comp in self.components
            if comp != acid
        )
        if acid_x_base <= 0.0:
            K = {
                comp: float(max(1e-6, min(1e6, base.get(comp, 1.0))))
                for comp in self.components
            }
            return self._set_cached_k_values(cache_key, K)

        y_acid = acid_x_base / max(acid_x_base + inert_x_base, 1e-30)
        kappa_base = model.K_eq(T) * P
        for _ in range(15):
            alpha = model._alpha_from_equilibrium(kappa_base, y_acid)
            acid_numerator = acid_x_base / max(1.0 - alpha, 1e-30)
            vapor_sum = acid_numerator + inert_x_base
            if vapor_sum <= 0.0:
                break
            y_new = acid_numerator / vapor_sum
            if abs(y_new - y_acid) < 1e-9:
                y_acid = y_new
                break
            y_acid = y_new

        alpha = model._alpha_from_equilibrium(kappa_base, y_acid)
        denom = max(1.0 - y_acid * alpha / 2.0, 1e-30)
        K = {}
        for comp in self.components:
            if comp == acid:
                value = base[comp] * denom / max(1.0 - alpha, 1e-30)
            else:
                value = base[comp] * denom
            K[comp] = float(max(1e-6, min(1e6, value)))
        return self._set_cached_k_values(cache_key, K)

    def _K_values_with_vapor_state(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
    ) -> tuple[dict[str, float], Optional[dict]]:
        cache_key = self._k_values_cache_key('gamma_vdm', T, P, composition)
        cached = self._get_cached_k_values(cache_key)
        if cached is not None:
            return cached, None

        x = {
            comp: max(float(composition.get(comp, 0.0)), 0.0)
            for comp in self.components
        }
        total = sum(x.values())
        if total <= 0.0:
            x = {comp: 1.0 / len(self.components) for comp in self.components}
        else:
            x = {comp: value / total for comp, value in x.items()}

        acid, model = self._active_vdm_model(x)
        if model is None:
            return super().K_values(T, P, x), None
        if acid is not None:
            K = self._single_acid_vdm_K_values(acid, model, T, P, x)
            return self._set_cached_k_values(cache_key, K), None

        gamma = self.activity_coefficients(T, x)
        phi_sat = {comp: self._vdm_phi_sat(comp, T) for comp in self.components}
        base_K = {
            comp: (
                gamma.get(comp, 1.0)
                * phi_sat[comp]
                * self.Psat(comp, T)
                / max(P, 1e-12)
            )
            for comp in self.components
        }
        K = {
            comp: float(max(1e-6, min(1e6, base_K[comp])))
            for comp in self.components
        }
        compiled_closure = getattr(model, 'compiled_vapor_closure', None)
        if callable(compiled_closure):
            closure = compiled_closure(
                T,
                P,
                self.components,
                base_K,
                x,
                max_iter=15,
                tol=1e-9,
                phi_floor=1e-12,
            )
            if closure is not None:
                K = {
                    comp: float(closure['values'].get(comp, K[comp]))
                    for comp in self.components
                }
                return self._set_cached_k_values(cache_key, K), closure
        y_sum = sum(x[comp] * K[comp] for comp in self.components)
        y = {
            comp: x[comp] * K[comp] / y_sum
            for comp in self.components
        } if y_sum > 0.0 else dict(x)

        for _ in range(15):
            phi_v = model.fugacity_coefficients(T, P, y, rk_model=None)
            K_new = {}
            for comp in self.components:
                phi = max(phi_v.get(comp, 1.0), 1e-12)
                value = gamma.get(comp, 1.0) * phi_sat[comp] * self.Psat(comp, T) / (phi * max(P, 1e-12))
                K_new[comp] = float(max(1e-6, min(1e6, value)))
            y_new = {comp: x[comp] * K_new[comp] for comp in self.components}
            y_sum = sum(max(value, 0.0) for value in y_new.values())
            if y_sum <= 0.0:
                K = K_new
                break
            y_new = {comp: max(value, 0.0) / y_sum for comp, value in y_new.items()}
            if max(abs(y_new[comp] - y.get(comp, 0.0)) for comp in self.components) < 1e-9:
                K = K_new
                break
            y = y_new
            K = K_new

        return self._set_cached_k_values(cache_key, K), None

    def K_values(self, T: float, P: float,
                 composition: dict[str, float]) -> dict[str, float]:
        return self._K_values_with_vapor_state(T, P, composition)[0]

    def aqueous_K_values(self, T: float, P: float,
                         composition: dict[str, float],
                         context: AqueousEquilibriumContext) -> dict[str, float]:
        """VDM vapor association with frozen Henry liquid standard states."""
        self._warn_aqueous_henry_pressure(P, context)
        x = self._normalized_aqueous_composition(composition, self.components)
        acid, model = self._active_vdm_model(x)
        if model is None:
            return super().aqueous_K_values(T, P, x, context)

        cache_key = self._k_values_cache_key('aqueous_gamma_vdm', T, P, x) + (
            context.cache_key(),
        )
        cached = self._get_cached_k_values(cache_key)
        if cached is not None:
            return cached

        gamma = self.activity_coefficients(
            T,
            self._aqueous_bulk_activity_composition(x, context),
        )
        reference_K = {}
        for comp in self.components:
            if comp in context.component_data:
                value = self._henry_ideal_vapor_K_value(comp, T, P, context)
            else:
                phi_sat = (
                    self._vdm_single_acid_phi_sat(comp, T)
                    if acid is not None else self._vdm_phi_sat(comp, T)
                )
                value = (
                    gamma.get(comp, 1.0) * phi_sat * self.Psat(comp, T)
                    / max(P, 1e-12)
                )
            reference_K[comp] = max(1e-12, min(1e12, float(value)))

        y = {comp: x.get(comp, 0.0) * reference_K[comp] for comp in self.components}
        y_total = sum(y.values())
        y = (
            {comp: value / y_total for comp, value in y.items()}
            if y_total > 0.0 else dict(x)
        )
        K = dict(reference_K)
        for _ in range(15):
            phi_v = model.fugacity_coefficients(T, P, y, rk_model=None)
            K = {
                comp: max(
                    1e-12,
                    min(1e12, reference_K[comp] / max(phi_v.get(comp, 1.0), 1e-12)),
                )
                for comp in self.components
            }
            y_new = {comp: x.get(comp, 0.0) * K[comp] for comp in self.components}
            y_total = sum(y_new.values())
            if y_total <= 0.0:
                break
            y_new = {comp: value / y_total for comp, value in y_new.items()}
            if max(abs(y_new[comp] - y.get(comp, 0.0)) for comp in self.components) < 1e-9:
                break
            y = y_new
        return self._set_cached_k_values(cache_key, K)

    def _rk_gamma_phi_K_values(self, T: float, P: float,
                               composition: dict[str, float]) -> dict[str, float]:
        return self._gamma_phi_K_values(T, P, composition)

    def K_value(self, comp: str, T: float, P: float,
                x: Optional[dict[str, float]] = None) -> float:
        composition = x if x is not None else {comp: 1.0}
        return self.K_values(T, P, composition).get(comp, 1.0)

    def bubble_point_T(self, composition: dict[str, float], P: float,
                       T_guess: float = 350.0) -> float:
        return _solve_bubble_point_temperature(self, composition, P, T_guess)
