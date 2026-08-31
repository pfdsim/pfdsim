"""On-demand liquid-vapor interfacial-property calculations.

Mixture surface tension is intentionally kept outside the stream state and the
core thermodynamic property resolver.  Unit operations that need an
interfacial property can retain a :class:`MixtureSurfaceTensionCalculator` and
reuse its activity and warm-start caches across nearby states.

The activity-model contract is deliberately small: an object must expose
``activity_coefficients(T, composition)`` and return a component-keyed mapping
of gamma values.  Existing UNIFAC thermodynamics objects satisfy that contract,
including their cached fragmentation and optional compiled backend.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import threading
from typing import Literal, Mapping, Optional, Protocol, Sequence, runtime_checkable
import warnings as python_warnings

import numpy as np


R = 8.31446261815324  # J/mol/K
SURFACE_AREA_TABLE_PATH = (
    Path(__file__).resolve().parent
    / 'data'
    / 'mixture_surface_tension_areas_cas.json'
)
_SURFACE_AREA_TABLE = None
_SURFACE_AREA_NAME_INDEX = None
_SURFACE_AREA_TABLE_LOCK = threading.RLock()


class InterfacialPropertyError(ValueError):
    """Raised when an interfacial property cannot be calculated safely."""


class InterfacialPropertyWarning(RuntimeWarning):
    """Warning emitted for an explicitly approximate method fallback."""


@runtime_checkable
class ActivityCoefficientModel(Protocol):
    """Minimal activity-model interface used by the Butler calculation."""

    def activity_coefficients(
        self,
        T: float,
        composition: Mapping[str, float],
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class SurfaceComponent:
    """Pure-component inputs needed by mixture surface-tension methods.

    Units:
        ``Vc``: critical molar volume, m^3/mol
        ``rho_molar(T)``: pure liquid molar density, mol/m^3
        ``sigma(T)``: pure liquid surface tension, N/m

    The two callables may return either a number or an object with a numeric
    ``value`` attribute, such as ``PropertyResolutionResult``.
    """

    name: str
    Vc: Optional[float]
    rho_molar: object
    sigma: object
    CAS: Optional[str] = None


@dataclass(frozen=True)
class SurfaceTensionResult:
    """Result and numerical diagnostics for a mixture surface tension."""

    sigma: float
    surface_x: np.ndarray
    bulk_x: np.ndarray
    A: np.ndarray
    method: str
    solver: str
    area_method: str
    converged: bool
    residual_max_abs: float
    iterations: int
    activity_evaluations: int = 0
    activity_cache_hits: int = 0
    warm_start_used: Optional[str] = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _WarmState:
    T: float
    bulk_x: np.ndarray
    surface_x: np.ndarray


class MixtureSurfaceTensionCalculator:
    """Reusable Butler/WSD mixture surface-tension calculator.

        ``activity_model`` is normally the flowsheet's existing UNIFAC-NIST
        thermodynamics object. Reusing that object
    preserves its component fragmentation and compiled numeric backend.  This
    calculator adds an exact-state LRU around activity calls made repeatedly by
    the nonlinear Butler solve.
    """

    def __init__(
        self,
        components: Sequence[SurfaceComponent],
        activity_model: Optional[ActivityCoefficientModel] = None,
        *,
        activity_cache_size: int = 4096,
        pure_input_cache_size: int = 256,
        warm_start_cache_size: int = 64,
        warm_start_max_distance: float = 0.35,
        newton_max_components: int = 12,
    ):
        self.components = tuple(components)
        if not self.components:
            raise InterfacialPropertyError('At least one component is required.')
        self.names = tuple(str(component.name) for component in self.components)
        if any(not name for name in self.names):
            raise InterfacialPropertyError('Surface-component names cannot be empty.')
        if len(set(self.names)) != len(self.names):
            raise InterfacialPropertyError('Surface-component names must be unique.')
        if activity_model is not None and not hasattr(
            activity_model, 'activity_coefficients'
        ):
            raise InterfacialPropertyError(
                'activity_model must expose activity_coefficients(T, composition).'
            )
        variant = str(getattr(activity_model, 'unifac_variant', '') or '').upper()
        if variant.startswith('UNIF') and variant != 'UNIFNIST':
            raise InterfacialPropertyError(
                f'Butler mixture surface tension requires UNIFNIST activities; '
                f'received {variant}.'
            )

        self.activity_model = activity_model
        self.activity_cache_size = max(0, int(activity_cache_size))
        self.pure_input_cache_size = max(0, int(pure_input_cache_size))
        self.warm_start_cache_size = max(0, int(warm_start_cache_size))
        self.warm_start_max_distance = max(0.0, float(warm_start_max_distance))
        self.newton_max_components = max(2, int(newton_max_components))

        self._activity_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._pure_input_cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._warm_states: deque[_WarmState] = deque(
            maxlen=self.warm_start_cache_size or None
        )
        self._activity_evaluations = 0
        self._activity_cache_hits = 0
        self._lock = threading.RLock()

    def clear_caches(self) -> None:
        """Clear activity, pure-input, and retained Butler warm starts."""

        with self._lock:
            self._activity_cache.clear()
            self._pure_input_cache.clear()
            self._warm_states.clear()
            self._activity_evaluations = 0
            self._activity_cache_hits = 0

    def cache_info(self) -> dict[str, int]:
        """Return compact cache statistics for diagnostics and tests."""

        with self._lock:
            return {
                'activity_entries': len(self._activity_cache),
                'pure_input_entries': len(self._pure_input_cache),
                'warm_start_entries': len(self._warm_states),
                'activity_evaluations': self._activity_evaluations,
                'activity_cache_hits': self._activity_cache_hits,
            }

    def calculate(
        self,
        T: float,
        x: Mapping[str, float] | Sequence[float],
        *,
        method: Literal['auto', 'butler', 'butler-unifac', 'wsd'] = 'auto',
        zero_cutoff: float = 1.0e-15,
        residual_tol: float = 1.0e-8,
        max_iterations: int = 100,
        max_nfev: int = 300,
        warm_start: Optional[SurfaceTensionResult | Sequence[float]] = None,
        use_cached_warm_start: bool = True,
        remember_warm_start: bool = True,
    ) -> SurfaceTensionResult:
        """Calculate liquid-vapor surface tension for one liquid composition.

        ``method='auto'`` prefers Butler whenever an activity model is present.
        If Butler is unavailable, raises, or does not converge, auto mode uses
        WSD and emits :class:`InterfacialPropertyWarning`.  Explicit Butler mode
        never silently changes methods.
        """

        T = _positive_finite(T, 'Temperature')
        bulk_x = self._normalize_composition(x)
        zero_cutoff = max(0.0, float(zero_cutoff))
        residual_tol = _positive_finite(residual_tol, 'Residual tolerance')
        max_iterations = max(1, int(max_iterations))
        max_nfev = max(1, int(max_nfev))
        method_key = str(method).strip().lower()
        if method_key not in {'auto', 'butler', 'butler-unifac', 'wsd'}:
            raise InterfacialPropertyError(f'Unknown surface-tension method: {method!r}.')

        active = bulk_x > zero_cutoff
        if not np.any(active):
            raise InterfacialPropertyError(
                'No components remain after applying the mole-fraction cutoff.'
            )
        active_indices = np.flatnonzero(active)

        if len(active_indices) == 1:
            return self._pure_component_limit(T, bulk_x, int(active_indices[0]))

        if method_key == 'wsd':
            return self._wsd(
                T,
                bulk_x,
                active_indices,
                fallback_reason='WSD was explicitly requested',
            )

        if self.activity_model is None:
            if method_key != 'auto':
                raise InterfacialPropertyError(
                    'Butler surface tension requires an activity_model.'
                )
            return self._wsd(
                T,
                bulk_x,
                active_indices,
                fallback_reason='no activity model was supplied for Butler',
            )

        try:
            result = self._butler(
                T,
                bulk_x,
                active_indices,
                residual_tol=residual_tol,
                max_iterations=max_iterations,
                max_nfev=max_nfev,
                warm_start=warm_start,
                use_cached_warm_start=use_cached_warm_start,
                remember_warm_start=remember_warm_start,
            )
        except Exception as exc:
            if method_key != 'auto':
                if isinstance(exc, InterfacialPropertyError):
                    raise
                raise InterfacialPropertyError(
                    f'Butler surface-tension calculation failed: {exc}'
                ) from exc
            return self._wsd(
                T,
                bulk_x,
                active_indices,
                fallback_reason=f'Butler failed: {exc}',
            )

        if result.converged or method_key != 'auto':
            return result
        return self._wsd(
            T,
            bulk_x,
            active_indices,
            fallback_reason=(
                f'Butler did not converge; residual '
                f'{result.residual_max_abs:.3e} N/m'
            ),
            inherited_warnings=result.warnings,
        )

    def _normalize_composition(
        self,
        x: Mapping[str, float] | Sequence[float],
    ) -> np.ndarray:
        if isinstance(x, Mapping):
            unknown = set(x) - set(self.names)
            if unknown:
                raise InterfacialPropertyError(
                    f'Composition contains unknown components: {sorted(unknown)}.'
                )
            values = [x.get(name, 0.0) for name in self.names]
        else:
            values = x
        return normalize_mole_fractions(values)

    def _pure_component_limit(
        self,
        T: float,
        bulk_x: np.ndarray,
        index: int,
    ) -> SurfaceTensionResult:
        _, _, sigma_i, A_active, area_method = self._pure_inputs(
            T, np.array([index])
        )
        surface_x = np.zeros_like(bulk_x)
        surface_x[index] = 1.0
        A = np.full_like(bulk_x, np.nan)
        A[index] = A_active[0]
        return SurfaceTensionResult(
            sigma=float(sigma_i[0]),
            surface_x=surface_x,
            bulk_x=bulk_x.copy(),
            A=A,
            method='pure-component surface tension',
            solver='closed-form',
            area_method=area_method,
            converged=True,
            residual_max_abs=0.0,
            iterations=0,
        )

    def _wsd(
        self,
        T: float,
        bulk_x: np.ndarray,
        active_indices: np.ndarray,
        *,
        fallback_reason: str,
        inherited_warnings: tuple[str, ...] = (),
    ) -> SurfaceTensionResult:
        _, Vb, sigma_i, A_active, area_method = self._pure_inputs(
            T, active_indices
        )
        xb = normalize_mole_fractions(bulk_x[active_indices])
        weights = xb * Vb
        V_mix = float(np.sum(weights))
        if V_mix <= 0.0 or not math.isfinite(V_mix):
            raise InterfacialPropertyError('WSD mixture molar volume is invalid.')
        sigma_matrix = np.sqrt(np.outer(sigma_i, sigma_i))
        sigma_mix = float(
            np.sum(np.outer(weights, weights) * sigma_matrix) / (V_mix * V_mix)
        )
        if sigma_mix < 0.0 or not math.isfinite(sigma_mix):
            raise InterfacialPropertyError('WSD produced an invalid surface tension.')

        message = (
            'Using Winterfeld-Scriven-Davis mixture surface tension because '
            f'{fallback_reason}; WSD does not model surface-phase segregation '
            'and is least reliable for aqueous or strongly nonideal mixtures.'
        )
        python_warnings.warn(message, InterfacialPropertyWarning, stacklevel=3)
        A = np.full_like(bulk_x, np.nan)
        A[active_indices] = A_active
        effective_bulk_x = np.zeros_like(bulk_x)
        effective_bulk_x[active_indices] = xb
        return SurfaceTensionResult(
            sigma=sigma_mix,
            surface_x=effective_bulk_x,
            bulk_x=bulk_x.copy(),
            A=A,
            method='Winterfeld-Scriven-Davis',
            solver='closed-form',
            area_method=area_method,
            converged=True,
            residual_max_abs=0.0,
            iterations=0,
            warnings=tuple(inherited_warnings) + (message,),
        )

    def _butler(
        self,
        T: float,
        bulk_x: np.ndarray,
        active_indices: np.ndarray,
        *,
        residual_tol: float,
        max_iterations: int,
        max_nfev: int,
        warm_start: Optional[SurfaceTensionResult | Sequence[float]],
        use_cached_warm_start: bool,
        remember_warm_start: bool,
    ) -> SurfaceTensionResult:
        start_evaluations = self._activity_evaluations
        start_hits = self._activity_cache_hits
        _, _, sigma_i, A_active, area_method = self._pure_inputs(
            T, active_indices
        )
        xb = normalize_mole_fractions(bulk_x[active_indices])
        xb_full = np.zeros_like(bulk_x)
        xb_full[active_indices] = xb
        ln_gamma_bulk = self._ln_gamma(T, xb_full)[active_indices]

        xs0, warm_source = self._select_warm_start(
            T,
            bulk_x,
            active_indices,
            warm_start,
            use_cached_warm_start,
        )
        if xs0 is None:
            xs0 = xb.copy()
            xs0, _, _ = self._fixed_point_seed(
                T,
                xs0,
                xb,
                active_indices,
                sigma_i,
                A_active,
                ln_gamma_bulk,
                iterations=min(8, max_iterations),
            )

        def component_sigmas(xs: np.ndarray) -> np.ndarray:
            return self._butler_component_sigmas(
                T,
                xs,
                xb,
                active_indices,
                sigma_i,
                A_active,
                ln_gamma_bulk,
            )

        def residual_z(z: np.ndarray) -> np.ndarray:
            values = component_sigmas(_softmax_with_reference(z))
            return values[:-1] - values[-1]

        z0 = _softmax_inverse(xs0)
        solver = 'damped-newton'
        solver_warning = None
        if len(active_indices) <= self.newton_max_components:
            z, converged, iterations, newton_reason = _damped_newton(
                residual_z,
                z0,
                tol=residual_tol,
                max_iter=max_iterations,
            )
        else:
            z = z0
            converged = False
            iterations = 0
            newton_reason = (
                f'active component count {len(active_indices)} exceeds safe Newton limit '
                f'{self.newton_max_components}'
            )

        if not converged:
            try:
                from scipy.optimize import least_squares

                scale = 1.0e-3
                opt = least_squares(
                    lambda values: residual_z(values) / scale,
                    z,
                    method='trf',
                    jac='3-point',
                    x_scale='jac',
                    xtol=1.0e-12,
                    ftol=1.0e-12,
                    gtol=1.0e-12,
                    max_nfev=max_nfev,
                )
                z = np.asarray(opt.x, dtype=float)
                iterations = int(opt.nfev)
                converged = bool(
                    opt.success
                    and np.max(np.abs(residual_z(z))) <= residual_tol
                )
                solver = 'scipy-least-squares'
                if not converged:
                    solver_warning = (
                        f'SciPy least_squares did not converge after Newton fallback: '
                        f'{opt.message}'
                    )
            except Exception as exc:
                solver_warning = (
                    f'Newton was unavailable ({newton_reason}); SciPy least_squares '
                    f'failed or is unavailable ({exc!r}); trying fixed point.'
                )

        if not converged:
            current_residual = float(np.max(np.abs(residual_z(z))))
            xs_fp, fp_residual, fp_iterations = self._fixed_point_seed(
                T,
                _softmax_with_reference(z),
                xb,
                active_indices,
                sigma_i,
                A_active,
                ln_gamma_bulk,
                iterations=max_iterations,
            )
            if fp_residual <= current_residual:
                z = _softmax_inverse(xs_fp)
                iterations = fp_iterations
                converged = fp_residual <= residual_tol
                solver = 'damped-fixed-point'

        xs = _softmax_with_reference(z)
        component_values = component_sigmas(xs)
        sigma_mix = float(np.mean(component_values))
        residual_max_abs = float(np.max(np.abs(component_values - sigma_mix)))
        converged = bool(
            converged
            and residual_max_abs <= residual_tol
            and sigma_mix >= 0.0
            and math.isfinite(sigma_mix)
        )
        result_warnings = []
        if solver_warning:
            result_warnings.append(solver_warning)
        if not converged:
            result_warnings.append(
                f'Butler residual {residual_max_abs:.3e} N/m exceeded tolerance '
                f'{residual_tol:.3e} N/m.'
            )

        surface_x = np.zeros_like(bulk_x)
        surface_x[active_indices] = xs
        A = np.full_like(bulk_x, np.nan)
        A[active_indices] = A_active
        if converged and remember_warm_start:
            self._remember_warm_state(T, bulk_x, surface_x)

        return SurfaceTensionResult(
            sigma=sigma_mix,
            surface_x=surface_x,
            bulk_x=bulk_x.copy(),
            A=A,
            method=self._butler_method_name(),
            solver=solver,
            area_method=area_method,
            converged=converged,
            residual_max_abs=residual_max_abs,
            iterations=iterations,
            activity_evaluations=self._activity_evaluations - start_evaluations,
            activity_cache_hits=self._activity_cache_hits - start_hits,
            warm_start_used=warm_source,
            warnings=tuple(result_warnings),
        )

    def _butler_method_name(self) -> str:
        variant = getattr(self.activity_model, 'unifac_variant', None)
        if variant:
            return f'Butler-{variant}'
        return f'Butler-{self.activity_model.__class__.__name__}'

    def _butler_component_sigmas(
        self,
        T: float,
        xs: np.ndarray,
        xb: np.ndarray,
        active_indices: np.ndarray,
        sigma_i: np.ndarray,
        A: np.ndarray,
        ln_gamma_bulk: np.ndarray,
    ) -> np.ndarray:
        xs = normalize_mole_fractions(xs)
        xs_full = np.zeros(len(self.components), dtype=float)
        xs_full[active_indices] = xs
        ln_gamma_surface = self._ln_gamma(T, xs_full)[active_indices]
        return sigma_i + (R * T / A) * (
            np.log(np.maximum(xs, 1.0e-300))
            + ln_gamma_surface
            - np.log(np.maximum(xb, 1.0e-300))
            - ln_gamma_bulk
        )

    def _fixed_point_seed(
        self,
        T: float,
        xs: np.ndarray,
        xb: np.ndarray,
        active_indices: np.ndarray,
        sigma_i: np.ndarray,
        A: np.ndarray,
        ln_gamma_bulk: np.ndarray,
        *,
        iterations: int,
    ) -> tuple[np.ndarray, float, int]:
        residual = math.inf
        for iteration in range(1, iterations + 1):
            xs_full = np.zeros(len(self.components), dtype=float)
            xs_full[active_indices] = xs
            ln_gamma_surface = self._ln_gamma(T, xs_full)[active_indices]
            _, target = _solve_butler_sigma_given_surface_activity(
                T,
                xb,
                sigma_i,
                A,
                ln_gamma_bulk,
                ln_gamma_surface,
            )
            log_xs = 0.5 * np.log(np.maximum(xs, 1.0e-300))
            log_xs += 0.5 * np.log(np.maximum(target, 1.0e-300))
            xs = _normalize_from_logs(log_xs)
            values = self._butler_component_sigmas(
                T,
                xs,
                xb,
                active_indices,
                sigma_i,
                A,
                ln_gamma_bulk,
            )
            residual = float(np.max(np.abs(values - np.mean(values))))
        return xs, residual, iterations

    def _ln_gamma(self, T: float, x: np.ndarray) -> np.ndarray:
        if self.activity_model is None:
            raise InterfacialPropertyError('No activity model is available.')
        x = normalize_mole_fractions(x)
        key = (float(T), tuple(float(value) for value in x))
        with self._lock:
            cached = self._activity_cache.get(key)
            if cached is not None:
                self._activity_cache.move_to_end(key)
                self._activity_cache_hits += 1
                return cached.copy()

        composition = {
            name: float(x[index]) for index, name in enumerate(self.names)
        }
        gamma = self.activity_model.activity_coefficients(float(T), composition)
        if not isinstance(gamma, Mapping):
            raise InterfacialPropertyError(
                'activity_coefficients must return a component-keyed mapping.'
            )
        try:
            values = np.array([float(gamma[name]) for name in self.names], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise InterfacialPropertyError(
                'activity_coefficients did not return every component.'
            ) from exc
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise InterfacialPropertyError(
                'Activity coefficients must be finite and strictly positive.'
            )
        ln_gamma = np.log(values)
        with self._lock:
            self._activity_evaluations += 1
            if self.activity_cache_size:
                self._activity_cache[key] = ln_gamma.copy()
                self._activity_cache.move_to_end(key)
                while len(self._activity_cache) > self.activity_cache_size:
                    self._activity_cache.popitem(last=False)
        return ln_gamma

    def _pure_inputs(
        self,
        T: float,
        active_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        key = (float(T), tuple(int(index) for index in active_indices))
        with self._lock:
            cached = self._pure_input_cache.get(key)
            if cached is not None:
                self._pure_input_cache.move_to_end(key)
                Vc, Vb, sigma, A, area_method = cached
                return (
                    Vc.copy(),
                    Vb.copy(),
                    sigma.copy(),
                    A.copy(),
                    area_method,
                )

        selected = [self.components[int(index)] for index in active_indices]
        rho = np.array(
            [_numeric(_call_temperature_property(component.rho_molar, T)) for component in selected],
            dtype=float,
        )
        sigma = np.array(
            [_numeric(_call_temperature_property(component.sigma, T)) for component in selected],
            dtype=float,
        )
        if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
            raise InterfacialPropertyError(
                'All active liquid molar densities must be finite and positive.'
            )
        if np.any(~np.isfinite(sigma)) or np.any(sigma < 0.0):
            raise InterfacialPropertyError(
                'All active pure surface tensions must be finite and nonnegative.'
            )
        Vb = 1.0 / rho
        tabulated_areas = [_tabulated_surface_area(component) for component in selected]
        if all(value is not None for value in tabulated_areas):
            A = np.array(tabulated_areas, dtype=float)
            Vc = np.array(
                [
                    _optional_numeric(component.Vc, default=math.nan)
                    for component in selected
                ],
                dtype=float,
            )
            area_method = 'tabulated-cas'
        else:
            Vc = np.array([_numeric(component.Vc) for component in selected], dtype=float)
            if np.any(~np.isfinite(Vc)) or np.any(Vc <= 0.0):
                raise InterfacialPropertyError(
                    'All active critical molar volumes must be finite and positive '
                    'when Goldsack-White areas are required.'
                )
            A = goldsack_white_area(Vc, Vb)
            area_method = 'goldsack-white'
        result = Vc, Vb, sigma, A, area_method
        with self._lock:
            if self.pure_input_cache_size:
                self._pure_input_cache[key] = (
                    Vc.copy(),
                    Vb.copy(),
                    sigma.copy(),
                    A.copy(),
                    area_method,
                )
                self._pure_input_cache.move_to_end(key)
                while len(self._pure_input_cache) > self.pure_input_cache_size:
                    self._pure_input_cache.popitem(last=False)
        return result

    def _select_warm_start(
        self,
        T: float,
        bulk_x: np.ndarray,
        active_indices: np.ndarray,
        explicit: Optional[SurfaceTensionResult | Sequence[float]],
        use_cached: bool,
    ) -> tuple[Optional[np.ndarray], Optional[str]]:
        if explicit is not None:
            values = explicit.surface_x if isinstance(explicit, SurfaceTensionResult) else explicit
            array = np.asarray(values, dtype=float)
            if array.shape != bulk_x.shape or np.any(~np.isfinite(array)) or np.any(array < 0.0):
                raise InterfacialPropertyError(
                    'Explicit Butler warm start must be a finite, nonnegative full composition.'
                )
            active_values = array[active_indices]
            if float(np.sum(active_values)) <= 0.0:
                raise InterfacialPropertyError(
                    'Explicit Butler warm start has no active surface components.'
                )
            return normalize_mole_fractions(active_values), 'explicit'

        if not use_cached or not self.warm_start_cache_size:
            return None, None
        best = None
        best_distance = math.inf
        with self._lock:
            states = tuple(self._warm_states)
        for state in states:
            distance = abs(float(T) - state.T) / max(float(T), state.T, 1.0)
            distance += float(np.sum(np.abs(bulk_x - state.bulk_x)))
            if distance < best_distance:
                active_values = state.surface_x[active_indices]
                if float(np.sum(active_values)) > 0.0:
                    best = normalize_mole_fractions(active_values)
                    best_distance = distance
        if best is not None and best_distance <= self.warm_start_max_distance:
            return best, 'cache'
        return None, None

    def _remember_warm_state(
        self,
        T: float,
        bulk_x: np.ndarray,
        surface_x: np.ndarray,
    ) -> None:
        if not self.warm_start_cache_size:
            return
        state = _WarmState(float(T), bulk_x.copy(), surface_x.copy())
        with self._lock:
            self._warm_states.append(state)


def estimate_mixture_surface_tension(
    T: float,
    x: Mapping[str, float] | Sequence[float],
    components: Sequence[SurfaceComponent],
    *,
    activity_model: Optional[ActivityCoefficientModel] = None,
    method: Literal['auto', 'butler', 'butler-unifac', 'wsd'] = 'auto',
    **options,
) -> SurfaceTensionResult:
    """One-shot convenience wrapper.

    Repeated unit-operation calls should retain a
    :class:`MixtureSurfaceTensionCalculator` instead so caches and warm starts
    survive between states.
    """

    calculator = MixtureSurfaceTensionCalculator(
        components,
        activity_model=activity_model,
    )
    return calculator.calculate(T, x, method=method, **options)


def normalize_mole_fractions(
    x: Sequence[float] | np.ndarray,
    *,
    atol: float = 1.0e-14,
) -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise InterfacialPropertyError(
            'Mole fractions must be a nonempty one-dimensional sequence.'
        )
    if np.any(~np.isfinite(array)):
        raise InterfacialPropertyError('Mole fractions must be finite.')
    if np.any(array < -atol):
        raise InterfacialPropertyError('Mole fractions cannot be negative.')
    array = np.maximum(array, 0.0)
    total = float(np.sum(array))
    if total <= 0.0:
        raise InterfacialPropertyError(
            'At least one mole fraction must be positive.'
        )
    return array / total


def goldsack_white_area(Vc: np.ndarray, Vb: np.ndarray) -> np.ndarray:
    """Goldsack-White molar surface area in m^2/mol."""

    Vc = np.asarray(Vc, dtype=float)
    Vb = np.asarray(Vb, dtype=float)
    if Vc.shape != Vb.shape or np.any(Vc <= 0.0) or np.any(Vb <= 0.0):
        raise InterfacialPropertyError('Vc and Vb must be positive arrays of equal shape.')
    return 1.021e8 * Vc ** (6.0 / 15.0) * Vb ** (4.0 / 15.0)


def _solve_butler_sigma_given_surface_activity(
    T: float,
    xb: np.ndarray,
    sigma_i: np.ndarray,
    A: np.ndarray,
    ln_gamma_bulk: np.ndarray,
    ln_gamma_surface: np.ndarray,
) -> tuple[float, np.ndarray]:
    alpha = A / (R * T)
    log_b = (
        np.log(np.maximum(xb, 1.0e-300))
        + ln_gamma_bulk
        - ln_gamma_surface
        - alpha * sigma_i
    )

    def objective(sigma: float) -> float:
        return _logsumexp(log_b + alpha * sigma)

    lo = float(np.min(sigma_i) - 0.5)
    hi = float(np.max(sigma_i) + 0.5)
    step = 0.5
    for _ in range(100):
        if objective(lo) <= 0.0:
            break
        lo -= step
        step *= 2.0
    else:
        raise InterfacialPropertyError('Could not bracket the lower Butler root.')
    step = 0.5
    for _ in range(100):
        if objective(hi) >= 0.0:
            break
        hi += step
        step *= 2.0
    else:
        raise InterfacialPropertyError('Could not bracket the upper Butler root.')
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if objective(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    xs = _normalize_from_logs(log_b + alpha * sigma)
    return sigma, xs


def _damped_newton(
    residual,
    z0: np.ndarray,
    *,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, bool, int, str]:
    """Damped finite-difference Newton with conditioning and line-search guards."""

    z = np.asarray(z0, dtype=float).copy()
    current = np.asarray(residual(z), dtype=float)
    for iteration in range(max_iter + 1):
        norm = float(np.max(np.abs(current)))
        if norm <= tol:
            return z, True, iteration, 'converged'
        if iteration == max_iter:
            break
        size = len(z)
        jacobian = np.empty((size, size), dtype=float)
        for column in range(size):
            step = 1.0e-5 * max(1.0, abs(float(z[column])))
            upper = z.copy()
            lower = z.copy()
            upper[column] += step
            lower[column] -= step
            jacobian[:, column] = (
                np.asarray(residual(upper)) - np.asarray(residual(lower))
            ) / (2.0 * step)
        if np.any(~np.isfinite(jacobian)):
            return z, False, iteration, 'non-finite Newton Jacobian'
        try:
            condition = float(np.linalg.cond(jacobian))
            if not math.isfinite(condition) or condition > 1.0e10:
                return z, False, iteration, f'ill-conditioned Newton Jacobian ({condition:g})'
            delta = np.linalg.solve(jacobian, -current)
        except np.linalg.LinAlgError as exc:
            return z, False, iteration, f'singular Newton Jacobian ({exc})'
        max_step = float(np.max(np.abs(delta)))
        if max_step > 12.0:
            delta *= 12.0 / max_step

        accepted = False
        damping = 1.0
        while damping >= 1.0 / 256.0:
            candidate = z + damping * delta
            candidate_residual = np.asarray(residual(candidate), dtype=float)
            candidate_norm = float(np.max(np.abs(candidate_residual)))
            if math.isfinite(candidate_norm) and candidate_norm < norm:
                z = candidate
                current = candidate_residual
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            return z, False, iteration, 'Newton line search rejected every step'
    return z, False, max_iter, 'Newton iteration limit reached'


def _softmax_with_reference(z: np.ndarray) -> np.ndarray:
    values = np.empty(len(z) + 1, dtype=float)
    values[:-1] = z
    values[-1] = 0.0
    return _normalize_from_logs(values)


def _softmax_inverse(x: np.ndarray) -> np.ndarray:
    x = normalize_mole_fractions(x)
    return np.log(np.maximum(x[:-1], 1.0e-300)) - math.log(
        max(float(x[-1]), 1.0e-300)
    )


def _normalize_from_logs(log_values: np.ndarray) -> np.ndarray:
    values = np.asarray(log_values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise InterfacialPropertyError('Cannot normalize non-finite logarithms.')
    return np.exp(values - _logsumexp(values))


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _call_temperature_property(function, T: float):
    if not callable(function):
        raise InterfacialPropertyError('Temperature-dependent property must be callable.')
    return function(float(T))


def _numeric(value) -> float:
    if hasattr(value, 'value'):
        value = value.value
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise InterfacialPropertyError(f'Expected a numeric property value, got {value!r}.') from exc


def _optional_numeric(value, *, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return _numeric(value)
    except InterfacialPropertyError:
        return float(default)


def _normalized_area_name(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(value or '').casefold())


def _surface_area_table() -> tuple[dict, dict]:
    global _SURFACE_AREA_TABLE, _SURFACE_AREA_NAME_INDEX
    with _SURFACE_AREA_TABLE_LOCK:
        if _SURFACE_AREA_TABLE is not None and _SURFACE_AREA_NAME_INDEX is not None:
            return _SURFACE_AREA_TABLE, _SURFACE_AREA_NAME_INDEX
        try:
            payload = json.loads(SURFACE_AREA_TABLE_PATH.read_text())
            areas = payload.get('areas', {})
        except (OSError, TypeError, ValueError):
            areas = {}
        if not isinstance(areas, dict):
            areas = {}
        name_index = {}
        for cas, entry in areas.items():
            if not isinstance(entry, dict):
                continue
            for name in (entry.get('name'), *(entry.get('aliases') or [])):
                key = _normalized_area_name(name)
                if key:
                    name_index[key] = cas
        _SURFACE_AREA_TABLE = areas
        _SURFACE_AREA_NAME_INDEX = name_index
        return areas, name_index


def _tabulated_surface_area(component: SurfaceComponent) -> Optional[float]:
    areas, name_index = _surface_area_table()
    candidates = [component.CAS, component.name]
    entry = None
    for candidate in candidates:
        text = str(candidate or '').strip()
        if text in areas:
            entry = areas[text]
            break
        cas = name_index.get(_normalized_area_name(text))
        if cas:
            entry = areas.get(cas)
            break
    if not isinstance(entry, dict):
        return None
    try:
        value = float(entry['area_m2_per_mol'])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value > 0.0 and math.isfinite(value) else None


def _positive_finite(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InterfacialPropertyError(f'{label} must be finite and positive.') from exc
    if result <= 0.0 or not math.isfinite(result):
        raise InterfacialPropertyError(f'{label} must be finite and positive.')
    return result


__all__ = [
    'ActivityCoefficientModel',
    'InterfacialPropertyError',
    'InterfacialPropertyWarning',
    'MixtureSurfaceTensionCalculator',
    'SurfaceComponent',
    'SurfaceTensionResult',
    'estimate_mixture_surface_tension',
    'goldsack_white_area',
    'normalize_mole_fractions',
    'SURFACE_AREA_TABLE_PATH',
]
