"""Optional compiled VLLE flash backends.

The backends keep the hot activity, VLE, LLE, RR, and structured VLLE loops in
Numba arrays.  Ideal-vapor UNIFAC-family, NRTL, and UNIQUAC methods are
supported; gamma-phi variants retain the readable reference implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import brentq

try:
    from numba import njit, typeof
except Exception:  # pragma: no cover
    njit = None
    typeof = None

try:
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .compiled_unifac import _activity_coefficients_numba
    else:
        from compiled_unifac import _activity_coefficients_numba
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .compiled_lle import _lle_split_numba
    else:
        from compiled_lle import _lle_split_numba
except Exception:  # pragma: no cover
    _activity_coefficients_numba = None
    _lle_split_numba = None

try:
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .compiled_activity import (
                CompiledNRTLBackend,
                CompiledUNIQUACBackend,
                _nrtl_activity_coefficients_numba,
                _uniquac_activity_coefficients_numba,
            )
    else:
        from compiled_activity import (
                CompiledNRTLBackend,
                CompiledUNIQUACBackend,
                _nrtl_activity_coefficients_numba,
                _uniquac_activity_coefficients_numba,
            )
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .compiled_lle import _lle_split_nrtl_numba, _lle_split_uniquac_numba
    else:
        from compiled_lle import _lle_split_nrtl_numba, _lle_split_uniquac_numba
except Exception:  # pragma: no cover
    CompiledNRTLBackend = None
    CompiledUNIQUACBackend = None
    _nrtl_activity_coefficients_numba = None
    _uniquac_activity_coefficients_numba = None
    _lle_split_nrtl_numba = None
    _lle_split_uniquac_numba = None


@dataclass
class CompiledVLLEFlashResult:
    """Array-backed flash result from the experimental backend."""

    phase_count: int
    status_code: int
    vapor_fraction: float
    liquid1_fraction: float
    liquid2_fraction: float
    y: np.ndarray
    x1: np.ndarray
    x2: np.ndarray
    residual: float
    iterations: int

    @property
    def status(self) -> str:
        return {
            0: "single_liquid",
            1: "single_vapor",
            2: "ordinary_vle",
            3: "binary_invariant_vlle",
            4: "structured_vlle_feed_lle_seed",
            5: "structured_vlle_vle_liquid_seed",
            6: "lle_only",
            7: "not_converged",
        }.get(self.status_code, f"unknown_{self.status_code}")


def _wrap_compiled_vlle_result(raw) -> CompiledVLLEFlashResult:
    return CompiledVLLEFlashResult(
        phase_count=int(raw[0]),
        status_code=int(raw[1]),
        vapor_fraction=float(raw[2]),
        liquid1_fraction=float(raw[3]),
        liquid2_fraction=float(raw[4]),
        y=np.asarray(raw[5], dtype=np.float64),
        x1=np.asarray(raw[6], dtype=np.float64),
        x2=np.asarray(raw[7], dtype=np.float64),
        residual=float(raw[8]),
        iterations=int(raw[9]),
    )


@dataclass
class CompiledActivityVLLEBackend:
    """Compiled TP/PV VLLE backend for ideal-vapor activity models."""

    components: list[str]
    model_id: int
    integer_parameters: np.ndarray
    parameter_0: np.ndarray
    parameter_1: np.ndarray
    parameter_2: np.ndarray
    parameter_3: np.ndarray
    parameter_4: np.ndarray
    parameter_5: np.ndarray
    parameter_6: np.ndarray
    parameter_7: np.ndarray
    parameter_8: np.ndarray
    parameter_9: np.ndarray
    thermo: object = field(repr=False)
    compilation_complete: bool = False
    _psat_coefficients_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @classmethod
    def from_thermo(
        cls,
        thermo,
    ) -> "CompiledActivityVLLEBackend | None":
        if njit is None:
            return None
        method_name = thermo.__class__.__name__.upper()
        unifac_backend = getattr(thermo, "_compiled_unifac", None)
        if unifac_backend is not None and _activity_coefficients_numba is not None and _lle_split_numba is not None:
            backend = unifac_backend
            model_id = 0
            integer_parameters = np.asarray([[int(backend.variant_id)]], dtype=np.int64)
            parameters = (
                backend.nu,
                backend.r,
                backend.q,
                backend.subgroup_q,
                backend.interactions,
                backend.interactions_b,
                backend.interactions_c,
                np.zeros_like(backend.interactions),
                np.zeros_like(backend.interactions),
                np.zeros_like(backend.interactions),
            )
        elif "NRTL" in method_name and CompiledNRTLBackend is not None:
            backend = CompiledNRTLBackend.from_thermo(thermo)
            if backend is None:
                return None
            model_id = 1
            integer_parameters = np.asarray(backend.tau_mode, dtype=np.int64)
            parameters = (
                backend.tau_c,
                backend.tau_d,
                backend.tau_e,
                backend.tau_f,
                backend.tau_g,
                backend.tau_tref,
                backend.tau_energy,
                backend.alpha,
                backend.interaction_tmin,
                backend.interaction_tmax,
            )
        elif "UNIQUAC" in method_name and CompiledUNIQUACBackend is not None:
            backend = CompiledUNIQUACBackend.from_thermo(thermo)
            if backend is None:
                return None
            model_id = 2
            integer_parameters = np.asarray(backend.tau_mode, dtype=np.int64)
            bounds = np.zeros_like(backend.tau_a, dtype=np.float64)
            for i in range(bounds.shape[0]):
                for j in range(i + 1, bounds.shape[1]):
                    bounds[i, j] = backend.interaction_tmin[i, j]
                    bounds[j, i] = backend.interaction_tmax[i, j]
            parameters = (
                np.diag(np.asarray(backend.r, dtype=np.float64)),
                np.diag(np.asarray(backend.q, dtype=np.float64)),
                np.diag(np.asarray(backend.q_residual, dtype=np.float64)),
                backend.tau_a,
                backend.tau_b,
                backend.tau_c,
                backend.tau_d,
                backend.tau_e,
                backend.tau_tref,
                bounds,
            )
        else:
            return None

        return cls(
            components=list(backend.components),
            model_id=model_id,
            integer_parameters=integer_parameters,
            parameter_0=np.asarray(parameters[0], dtype=np.float64),
            parameter_1=np.asarray(parameters[1], dtype=np.float64),
            parameter_2=np.asarray(parameters[2], dtype=np.float64),
            parameter_3=np.asarray(parameters[3], dtype=np.float64),
            parameter_4=np.asarray(parameters[4], dtype=np.float64),
            parameter_5=np.asarray(parameters[5], dtype=np.float64),
            parameter_6=np.asarray(parameters[6], dtype=np.float64),
            parameter_7=np.asarray(parameters[7], dtype=np.float64),
            parameter_8=np.asarray(parameters[8], dtype=np.float64),
            parameter_9=np.asarray(parameters[9], dtype=np.float64),
            thermo=thermo,
        )

    def compile_kernels(self) -> None:
        if self.compilation_complete or typeof is None:
            return
        count = len(self.components)
        composition = np.zeros(count, dtype=np.float64)
        coeff_a = np.zeros(count, dtype=np.float64)
        coeff_b = np.zeros(count, dtype=np.float64)
        coeff_cd = np.zeros((count, 2), dtype=np.float64)

        def compile_for(dispatcher, *args):
            dispatcher.compile(tuple(typeof(argument) for argument in args))

        if self.model_id == 0:
            common = (
                self.parameter_0, self.parameter_1, self.parameter_2,
                self.parameter_3, self.parameter_4, self.parameter_5,
                self.parameter_6, int(self.integer_parameters[0, 0]),
                coeff_a, coeff_b, coeff_cd,
            )
            compile_for(
                _vlle_flash_tp_unifac_numba,
                composition, 298.15, 1.0, *common, 80, 1.0e-10,
            )
            compile_for(
                _vle_flash_unifac,
                composition, 298.15, 1.0, *common, 80, 1.0e-10,
            )
        else:
            common = self._kernel_args_with_coefficients(
                (coeff_a, coeff_b, coeff_cd)
            )
            compile_for(
                _vlle_flash_tp_activity_numba,
                composition, 298.15, 1.0, *common, 80, 1.0e-10,
            )
            compile_for(
                _vle_flash_activity,
                composition, 298.15, 1.0, *common, 80, 1.0e-10,
            )
        self.compilation_complete = True

    def _z_array(self, composition: dict[str, float]) -> np.ndarray:
        return np.asarray(
            [max(float(composition.get(comp, 0.0)), 0.0) for comp in self.components],
            dtype=np.float64,
        )

    def _kernel_args_with_coefficients(
        self,
        coeffs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple:
        return (
            int(self.model_id),
            self.integer_parameters,
            self.parameter_0,
            self.parameter_1,
            self.parameter_2,
            self.parameter_3,
            self.parameter_4,
            self.parameter_5,
            self.parameter_6,
            self.parameter_7,
            self.parameter_8,
            self.parameter_9,
            coeffs[0],
            coeffs[1],
            coeffs[2],
        )

    def _psat_coefficients_for_range(
        self,
        T_low: float,
        T_high: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _psat_coefficients_for_components(
            self.thermo,
            self.components,
            self._psat_coefficients_cache,
            T_low,
            T_high,
        )

    def _psat_coefficients_for_temperature(self, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = max(1.0, float(T))
        width = max(2.0, 0.005 * center)
        return self._psat_coefficients_for_range(center - width, center + width)

    def _flash_TP_with_coefficients(
        self,
        z: np.ndarray,
        T: float,
        P: float,
        coeffs: tuple[np.ndarray, np.ndarray, np.ndarray],
        max_iter: int,
        tol: float,
    ) -> CompiledVLLEFlashResult:
        if self.model_id == 0:
            raw = _vlle_flash_tp_unifac_numba(
                z,
                float(T),
                float(P),
                self.parameter_0,
                self.parameter_1,
                self.parameter_2,
                self.parameter_3,
                self.parameter_4,
                self.parameter_5,
                self.parameter_6,
                int(self.integer_parameters[0, 0]),
                coeffs[0],
                coeffs[1],
                coeffs[2],
                int(max_iter),
                float(tol),
            )
        else:
            raw = _vlle_flash_tp_activity_numba(
                z,
                float(T),
                float(P),
                *self._kernel_args_with_coefficients(coeffs),
                int(max_iter),
                float(tol),
            )
        return _wrap_compiled_vlle_result(raw)

    def flash_TP(self, composition: dict[str, float], T: float, P: float,
                 max_iter: int = 80, tol: float = 1e-10) -> CompiledVLLEFlashResult:
        if self.model_id != 0:
            self.thermo._warn_activity_interaction_extrapolation(T, self.components)
        z = self._z_array(composition)
        coeffs = self._psat_coefficients_for_temperature(float(T))
        return self._flash_TP_with_coefficients(z, T, P, coeffs, max_iter, tol)

    def flash_VLE_TP(self, composition: dict[str, float], T: float, P: float,
                     max_iter: int = 80, tol: float = 1e-10
                     ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compiled constrained VLE candidate for adaptive spinodal mode."""
        if self.model_id != 0:
            self.thermo._warn_activity_interaction_extrapolation(T, self.components)
        z = self._z_array(composition)
        coeffs = self._psat_coefficients_for_temperature(float(T))
        if self.model_id == 0:
            V, x, y = _vle_flash_unifac(
                z,
                float(T),
                float(P),
                self.parameter_0,
                self.parameter_1,
                self.parameter_2,
                self.parameter_3,
                self.parameter_4,
                self.parameter_5,
                self.parameter_6,
                int(self.integer_parameters[0, 0]),
                coeffs[0],
                coeffs[1],
                coeffs[2],
                int(max_iter),
                float(tol),
            )
        else:
            V, x, y = _vle_flash_activity(
                z,
                float(T),
                float(P),
                *self._kernel_args_with_coefficients(coeffs),
                int(max_iter),
                float(tol),
            )
        return float(V), np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def flash_PV(self, composition: dict[str, float], P: float, vapor_fraction: float,
                 T_low: float | None = None, T_high: float | None = None,
                 max_iter: int = 80, tol: float = 1e-9,
                 T_guess: float = 350.0) -> tuple[float, CompiledVLLEFlashResult]:
        z_dict = self._normalize_dict(composition)
        target = max(0.0, min(1.0, float(vapor_fraction)))
        trials: dict[float, CompiledVLLEFlashResult] = {}
        # Branch identity at bubble/dew boundaries must match a direct TP call;
        # do not let the looser outer PV tolerance alter the inner TP phase
        # classification.
        tp_tolerance = min(float(tol), 1.0e-10)

        def result_at(T: float) -> CompiledVLLEFlashResult:
            key = round(float(T), 10)
            result = trials.get(key)
            if result is None:
                result = self.flash_TP(
                    z_dict,
                    T,
                    P,
                    max_iter=max_iter,
                    tol=tp_tolerance,
                )
                trials[key] = result
            return result

        def residual(T: float) -> float:
            return result_at(T).vapor_fraction - target

        if target <= 1e-10:
            low = (
                self.thermo.bubble_point_T_vlle(z_dict, P, T_guess=T_guess, max_iter=max_iter)
                if T_low is None else float(T_low)
            )
            return low, result_at(low)
        if target >= 1.0 - 1e-10:
            high = (
                self.thermo.dew_point_T_vlle(z_dict, P, T_guess=T_guess, max_iter=max_iter)
                if T_high is None else float(T_high)
            )
            return high, result_at(high)

        samples: list[tuple[float, float, CompiledVLLEFlashResult]] = []
        exact_match: tuple[float, CompiledVLLEFlashResult] | None = None
        center = max(1.0, min(5000.0, float(T_guess)))
        for width in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0):
            points = [center] if width <= 0.0 else [
                max(1.0, center - width),
                min(5000.0, center + width),
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
                        return point, result
                    if exact_match is None:
                        exact_match = (point, result)
                samples.append((point, f_value, result))

            phase3_values = sorted(
                (point, f_value)
                for point, f_value, result in samples
                if result.phase_count == 3
            )
            for (T1, f1), (T2, f2) in zip(phase3_values, phase3_values[1:]):
                if f1 * f2 < 0.0:
                    T = brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=80)
                    return T, result_at(T)

            branch_risk = any(
                result.phase_count == 3 or result.status == 'lle_only'
                for _point, _f_value, result in samples
            )
            if exact_match is not None and not branch_risk:
                return exact_match

            if not branch_risk:
                all_values = sorted((point, f_value) for point, f_value, _result in samples)
                for (T1, f1), (T2, f2) in zip(all_values, all_values[1:]):
                    if f1 * f2 < 0.0:
                        T = brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=80)
                        return T, result_at(T)

        if exact_match is not None:
            return exact_match

        all_values = sorted((point, f_value) for point, f_value, _result in samples)
        for (T1, f1), (T2, f2) in zip(all_values, all_values[1:]):
            if f1 * f2 < 0.0:
                T = brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=80)
                return T, result_at(T)

        if T_low is None:
            T_low = self.thermo.bubble_point_T_vlle(z_dict, P, T_guess=T_guess, max_iter=max_iter)
        if T_high is None:
            T_high = self.thermo.dew_point_T_vlle(z_dict, P, T_guess=T_guess, max_iter=max_iter)
        low, high = sorted((float(T_low), float(T_high)))

        f_low = residual(low)
        if math.isfinite(f_low) and abs(f_low) <= 1e-8:
            return low, result_at(low)
        f_high = residual(high)
        if math.isfinite(f_high) and abs(f_high) <= 1e-8:
            return high, result_at(high)
        if math.isfinite(f_low) and math.isfinite(f_high) and f_low * f_high < 0.0:
            T = brentq(residual, low, high, xtol=1e-7, rtol=1e-9, maxiter=80)
            return T, result_at(T)

        grid = [low + (high - low) * index / 12.0 for index in range(13)]
        values = []
        for T in grid:
            try:
                f = residual(T)
            except Exception:
                continue
            if math.isfinite(f):
                values.append((T, f))
        for T, f in values:
            if abs(f) <= 1e-8:
                return T, result_at(T)
        brackets = [
            (T1, T2)
            for (T1, f1), (T2, f2) in zip(values, values[1:])
            if f1 * f2 < 0.0
        ]
        if brackets:
            preferred = 0.5 * (low + high)
            T1, T2 = min(brackets, key=lambda item: abs(0.5 * (item[0] + item[1]) - preferred))
            T = brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=80)
            return T, result_at(T)
        raise ValueError("Compiled VLLE PV target is not bracketed")

    def _normalize_dict(self, composition: dict[str, float]) -> dict[str, float]:
        values = {
            comp: max(float(composition.get(comp, 0.0)), 0.0)
            for comp in self.components
        }
        total = sum(values.values())
        if total <= 0.0:
            return {comp: 1.0 / len(self.components) for comp in self.components}
        return {comp: value / total for comp, value in values.items()}

def _fit_psat_surrogate_from_samples(
    samples: list[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    if len(samples) < 8:
        return None

    T = np.asarray([item[0] for item in samples], dtype=float)
    log_p = np.asarray([math.log(item[1]) for item in samples], dtype=float)
    design = np.column_stack((
        np.ones_like(T),
        1.0 / T,
        np.log(T),
        T * T,
    ))

    # The T**2 column is several orders of magnitude larger than 1/T.
    # Solve the scaled linear problem, then transform back to the physical
    # coefficient basis used by the compiled kernels.
    offsets = design.mean(axis=0)
    scales = design.std(axis=0)
    offsets[0] = 0.0
    scales[0] = 1.0
    scaled_design = (design - offsets) / scales
    scaled_coeffs, *_ = np.linalg.lstsq(scaled_design, log_p, rcond=None)
    coeffs = scaled_coeffs / scales
    coeffs[0] -= float(np.sum(scaled_coeffs[1:] * offsets[1:] / scales[1:]))

    residual = design @ coeffs - log_p
    if float(np.linalg.norm(residual, ord=np.inf)) > math.log(10.0) * 0.05:
        return None
    return tuple(float(value) for value in coeffs)


def _fit_psat_surrogate_from_thermo_psat(
    thermo,
    comp: str,
    fit_range: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    """Fit ln(P_bar) = A + B/T + C*ln(T) + D*T**2 from active Psat."""

    samples = []
    T_low = float(fit_range[0])
    T_high = float(fit_range[1])
    if not np.isfinite(T_low) or not np.isfinite(T_high) or T_high <= T_low:
        return None
    for T in np.linspace(T_low, T_high, 41):
        try:
            P = float(thermo.Psat(comp, float(T)))
        except Exception:
            continue
        if np.isfinite(P) and P > 0.0:
            samples.append((float(T), float(P)))
    if len(samples) < 8:
        return None

    # Antoine is monotonic in its normal operating region.  If the active
    # Psat path is nonmonotonic over this window, usually because the resolver
    # fell through to a rough corresponding-states fallback outside its domain,
    # do not hide that behavior behind a smooth surrogate.
    increasing_pairs = 0
    for (_T1, p1), (_T2, p2) in zip(samples, samples[1:]):
        if p2 > p1:
            increasing_pairs += 1
    if increasing_pairs < len(samples) - 1:
        return None

    return _fit_psat_surrogate_from_samples(samples)


def _temperature_window(T_low: float, T_high: float) -> tuple[float, float]:
    low = max(1.0, float(T_low))
    high = max(1.0, float(T_high))
    if high < low:
        low, high = high, low
    if high - low < 1e-9:
        center = low
        width = max(20.0, 0.04 * center)
        low = center - width
        high = center + width
    else:
        padding = max(1.0, 0.10 * (high - low))
        low -= padding
        high += padding
    return max(1.0, low), min(5000.0, high)


def _psat_coefficients_for_components(
    thermo,
    components: list[str],
    cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]],
    T_low: float,
    T_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low, high = _temperature_window(T_low, T_high)
    key = (round(low, 6), round(high, 6))
    cached = cache.get(key)
    if cached is not None:
        return cached

    a = []
    b = []
    c = []
    d = []
    for comp in components:
        coeffs = _fit_psat_surrogate_from_thermo_psat(thermo, comp, (low, high))
        props = getattr(thermo, "props", {}).get(comp)
        if (
            coeffs is None
            and props is not None
            and props.antoine_A is not None
            and props.antoine_B is not None
            and props.antoine_C is not None
            and (
                props.antoine_Tmin is None
                or props.antoine_Tmax is None
                or (low >= props.antoine_Tmin - 1e-9 and high <= props.antoine_Tmax + 1e-9)
            )
        ):
            samples = []
            for T in np.linspace(low, high, 41):
                T_c = float(T) - 273.15
                denom = float(props.antoine_C) + T_c
                if abs(denom) < 1e-12:
                    continue
                exponent = float(props.antoine_A) - float(props.antoine_B) / denom
                P = 10.0 ** exponent
                if np.isfinite(P) and P > 0.0:
                    samples.append((float(T), float(P)))
            coeffs = _fit_psat_surrogate_from_samples(samples)
        if coeffs is None:
            raise ValueError(
                f"Cannot build compiled Psat surrogate for {comp} over {low:g}-{high:g} K"
            )
        a.append(coeffs[0])
        b.append(coeffs[1])
        c.append(coeffs[2])
        d.append(coeffs[3])

    result = (
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        np.column_stack((
            np.asarray(c, dtype=np.float64),
            np.asarray(d, dtype=np.float64),
        )),
    )
    if len(cache) > 128:
        cache.clear()
    cache[key] = result
    return result


if njit is not None and _activity_coefficients_numba is not None and _lle_split_numba is not None:

    @njit(cache=True)
    def _norm(values):
        n = values.shape[0]
        out = np.empty(n, dtype=np.float64)
        total = 0.0
        for i in range(n):
            value = values[i]
            if value < 0.0:
                value = 0.0
            out[i] = value
            total += value
        if total <= 0.0:
            for i in range(n):
                out[i] = 1.0 / n
        else:
            for i in range(n):
                out[i] /= total
        return out


    @njit(cache=True)
    def _psat_array(T, antoine_a, antoine_b, antoine_c):
        n = antoine_a.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            log_p = antoine_a[i] + antoine_b[i] / T + antoine_c[i, 0] * math.log(T) + antoine_c[i, 1] * T * T
            if log_p > 115.0:
                log_p = 115.0
            elif log_p < -115.0:
                log_p = -115.0
            out[i] = math.exp(log_p)
        return out


    @njit(cache=True)
    def _k_values_unifac(x, T, P, nu, r, q, subgroup_q, interactions,
                         interactions_b, interactions_c, variant_id,
                         antoine_a, antoine_b, antoine_c):
        gamma = _activity_coefficients_numba(
            nu, r, q, subgroup_q, interactions, interactions_b, interactions_c,
            variant_id, x, T
        )
        psat = _psat_array(T, antoine_a, antoine_b, antoine_c)
        n = x.shape[0]
        K = np.empty(n, dtype=np.float64)
        if P < 1e-12:
            P = 1e-12
        for i in range(n):
            value = gamma[i] * psat[i] / P
            if value < 1e-8:
                value = 1e-8
            elif value > 1e8:
                value = 1e8
            K[i] = value
        return K


    @njit(cache=True)
    def _bubble_point_p_unifac(x, T, nu, r, q, subgroup_q, interactions,
                               interactions_b, interactions_c, variant_id,
                               antoine_a, antoine_b, antoine_c):
        x = _norm(x)
        gamma = _activity_coefficients_numba(
            nu, r, q, subgroup_q, interactions, interactions_b, interactions_c,
            variant_id, x, T
        )
        psat = _psat_array(T, antoine_a, antoine_b, antoine_c)
        total = 0.0
        for i in range(x.shape[0]):
            total += x[i] * gamma[i] * psat[i]
        return total

    @njit(cache=True)
    def _rr_vle(z, K, guess):
        f0 = 0.0
        f1 = 0.0
        for i in range(z.shape[0]):
            km1 = K[i] - 1.0
            f0 += z[i] * km1
            f1 += z[i] * km1 / K[i]
        if f0 <= 0.0:
            return 0.0
        if f1 >= 0.0:
            return 1.0
        V = guess
        if V < 0.0:
            V = 0.0
        elif V > 1.0:
            V = 1.0
        for _ in range(30):
            f = 0.0
            df = 0.0
            for i in range(z.shape[0]):
                km1 = K[i] - 1.0
                denom = 1.0 + V * km1
                if abs(denom) < 1e-14:
                    denom = 1e-14
                f += z[i] * km1 / denom
                df -= z[i] * km1 * km1 / (denom * denom)
            if abs(f) < 1e-12:
                return V
            if abs(df) < 1e-14:
                break
            new_V = V - f / df
            if new_V <= 0.0 or new_V >= 1.0:
                break
            if abs(new_V - V) < 1e-12:
                return new_V
            V = new_V
        low = 0.0
        high = 1.0
        for _ in range(80):
            mid = 0.5 * (low + high)
            f = 0.0
            for i in range(z.shape[0]):
                km1 = K[i] - 1.0
                f += z[i] * km1 / (1.0 + mid * km1)
            if abs(f) < 1e-13:
                return mid
            if f > 0.0:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)


    @njit(cache=True)
    def _vle_flash_unifac(z_input, T, P, nu, r, q, subgroup_q, interactions,
                          interactions_b, interactions_c, variant_id,
                          antoine_a, antoine_b, antoine_c, max_iter, tol):
        z = _norm(z_input)
        x = z.copy()
        y = z.copy()
        V = 0.5
        K = _k_values_unifac(x, T, P, nu, r, q, subgroup_q, interactions,
                             interactions_b, interactions_c, variant_id,
                             antoine_a, antoine_b, antoine_c)
        for _iteration in range(max_iter):
            old_x = x.copy()
            V = _rr_vle(z, K, V)
            for i in range(z.shape[0]):
                denom = 1.0 + V * (K[i] - 1.0)
                if abs(denom) < 1e-14:
                    denom = 1e-14
                x[i] = z[i] / denom
                y[i] = K[i] * x[i]
            x = _norm(x)
            y = _norm(y)
            K_new = _k_values_unifac(
                x, T, P, nu, r, q, subgroup_q, interactions,
                interactions_b, interactions_c, variant_id,
                antoine_a, antoine_b, antoine_c,
            )
            change = 0.0
            for i in range(z.shape[0]):
                diff = abs(x[i] - old_x[i])
                if diff > change:
                    change = diff
            K = K_new
            if change < tol:
                break
        V = _rr_vle(z, K, V)
        for i in range(z.shape[0]):
            denom = 1.0 + V * (K[i] - 1.0)
            if abs(denom) < 1e-14:
                denom = 1e-14
            x[i] = z[i] / denom
            y[i] = K[i] * x[i]
        x = _norm(x)
        y = _norm(y)
        return V, x, y


    @njit(cache=True)
    def _seeded_vle_unifac(z_input, T, P, liquid_seed, nu, r, q, subgroup_q,
                           interactions, interactions_b, interactions_c, variant_id,
                           antoine_a, antoine_b, antoine_c, max_iter, tol):
        z = _norm(z_input)
        x = _norm(liquid_seed)
        y = z.copy()
        K = _k_values_unifac(
            x, T, P, nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, antoine_a, antoine_b, antoine_c
        )
        V = 0.5
        convergence_tolerance = max(tol, 1e-9)
        converged = False
        for _iteration in range(max_iter):
            V = _rr_vle(z, K, V)
            for i in range(z.shape[0]):
                denom = 1.0 + V * (K[i] - 1.0)
                if abs(denom) < 1e-14:
                    denom = 1e-14
                x[i] = z[i] / denom
                y[i] = K[i] * x[i]
            x = _norm(x)
            y = _norm(y)
            K_new = _k_values_unifac(
                x, T, P, nu, r, q, subgroup_q, interactions, interactions_b,
                interactions_c, variant_id, antoine_a, antoine_b, antoine_c
            )
            error = 0.0
            for i in range(z.shape[0]):
                value = abs(math.log(max(K_new[i], 1e-300) / max(K[i], 1e-300)))
                if value > error:
                    error = value
                K[i] = math.sqrt(max(K[i], 1e-300) * max(K_new[i], 1e-300))
            if error < convergence_tolerance:
                converged = True
                break

        V = _rr_vle(z, K, V)
        for i in range(z.shape[0]):
            denom = 1.0 + V * (K[i] - 1.0)
            if abs(denom) < 1e-14:
                denom = 1e-14
            x[i] = z[i] / denom
            y[i] = K[i] * x[i]
        x = _norm(x)
        y = _norm(y)
        if not converged or V <= 1e-10 or V >= 1.0 - 1e-10:
            return False, V, x, y

        K_check = _k_values_unifac(
            x, T, P, nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, antoine_a, antoine_b, antoine_c
        )
        residual = 0.0
        for i in range(z.shape[0]):
            value = abs(math.log(
                max(y[i], 1e-300) / max(x[i] * K_check[i], 1e-300)
            ))
            if value > residual:
                residual = value
        if residual > max(1e-7, 100.0 * convergence_tolerance):
            return False, V, x, y
        return True, V, x, y


    @njit(cache=True)
    def _liquid_tpd_unifac(z_input, T, P, liquid_seed, nu, r, q, subgroup_q,
                           interactions, interactions_b, interactions_c, variant_id,
                           antoine_a, antoine_b, antoine_c):
        z = _norm(z_input)
        x = _norm(liquid_seed)
        K = _k_values_unifac(
            x, T, P, nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, antoine_a, antoine_b, antoine_c
        )
        value = 0.0
        for i in range(z.shape[0]):
            value += x[i] * math.log(
                max(x[i] * K[i], 1e-300) / max(z[i], 1e-300)
            )
        return value


    @njit(cache=True)
    def _reduced_gibbs_unifac(T, P, V, x_input, y_input, nu, r, q, subgroup_q,
                              interactions, interactions_b, interactions_c, variant_id,
                              antoine_a, antoine_b, antoine_c):
        x = _norm(x_input)
        y = _norm(y_input)
        pressure = max(P, 1e-300)
        value = 0.0
        for i in range(y.shape[0]):
            value += V * y[i] * math.log(max(y[i] * pressure, 1e-300))
        if V >= 1.0 - 1e-12:
            return value
        K = _k_values_unifac(
            x, T, P, nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, antoine_a, antoine_b, antoine_c
        )
        for i in range(x.shape[0]):
            value += (1.0 - V) * x[i] * math.log(
                max(x[i] * K[i] * pressure, 1e-300)
            )
        return value


    @njit(cache=True)
    def _stable_vle_from_lle_seeds_unifac(
        z, T, P, seed1, seed2, nu, r, q, subgroup_q, interactions,
        interactions_b, interactions_c, variant_id, antoine_a, antoine_b,
        antoine_c, max_iter, tol,
    ):
        tpd1 = _liquid_tpd_unifac(
            z, T, P, seed1, nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, antoine_a, antoine_b, antoine_c
        )
        tpd2 = _liquid_tpd_unifac(
            z, T, P, seed2, nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, antoine_a, antoine_b, antoine_c
        )
        seed = seed1 if tpd1 <= tpd2 else seed2
        tpd = min(tpd1, tpd2)
        empty = np.zeros(z.shape[0], dtype=np.float64)
        if not np.isfinite(tpd) or tpd >= -max(1e-8, 10.0 * tol):
            return False, 1.0, empty, empty

        ok, V, x, y = _seeded_vle_unifac(
            z, T, P, seed, nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, antoine_a, antoine_b, antoine_c,
            max_iter, tol
        )
        if not ok:
            return False, V, x, y
        candidate_gibbs = _reduced_gibbs_unifac(
            T, P, V, x, y, nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, antoine_a, antoine_b, antoine_c
        )
        vapor_gibbs = _reduced_gibbs_unifac(
            T, P, 1.0, z, z, nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, antoine_a, antoine_b, antoine_c
        )
        if not np.isfinite(candidate_gibbs) or candidate_gibbs - vapor_gibbs >= -1e-10:
            return False, V, x, y
        return True, V, x, y


    @njit(cache=True)
    def _solve_three_phase_rr(z, k1, k2, seed_l1, seed_l2):
        n = z.shape[0]

        l1 = seed_l1
        l2 = seed_l2
        if l1 < 1e-10:
            l1 = 1e-10
        if l2 < 1e-10:
            l2 = 1e-10
        if l1 + l2 > 1.0 - 1e-10:
            scale = (1.0 - 1e-10) / (l1 + l2)
            l1 *= scale
            l2 *= scale
        best_norm = 1e300
        best_l1 = l1
        best_l2 = l2

        for _iteration in range(35):
            v = 1.0 - l1 - l2
            f1 = 0.0
            f2 = 0.0
            valid = True
            for i in range(n):
                denom = v + l1 / k1[i] + l2 / k2[i]
                if denom <= 0.0:
                    valid = False
                    break
                y_i = z[i] / denom
                f1 += y_i / k1[i]
                f2 += y_i / k2[i]
            if not valid:
                break
            f1 -= 1.0
            f2 -= 1.0
            norm = max(abs(f1), abs(f2))
            if norm < best_norm:
                best_norm = norm
                best_l1 = l1
                best_l2 = l2
            if norm < 1e-10:
                break

            jac = np.zeros((2, 2), dtype=np.float64)
            for i in range(n):
                d_l1 = 1.0 / k1[i] - 1.0
                d_l2 = 1.0 / k2[i] - 1.0
                denom = v + l1 / k1[i] + l2 / k2[i]
                if denom <= 0.0:
                    valid = False
                    break
                scale = -z[i] / (denom * denom)
                jac[0, 0] += scale * d_l1 / k1[i]
                jac[0, 1] += scale * d_l2 / k1[i]
                jac[1, 0] += scale * d_l1 / k2[i]
                jac[1, 1] += scale * d_l2 / k2[i]
            if not valid:
                break
            det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
            if abs(det) < 1e-18:
                break
            d1 = (-f1 * jac[1, 1] + jac[0, 1] * f2) / det
            d2 = (jac[1, 0] * f1 - jac[0, 0] * f2) / det
            damping = 1.0
            accepted = False
            for _line in range(25):
                tl1 = l1 + damping * d1
                tl2 = l2 + damping * d2
                if tl1 > 0.0 and tl2 > 0.0 and tl1 + tl2 < 1.0:
                    tv = 1.0 - tl1 - tl2
                    tf1 = 0.0
                    tf2 = 0.0
                    good = True
                    for i in range(n):
                        denom = tv + tl1 / k1[i] + tl2 / k2[i]
                        if denom <= 0.0:
                            good = False
                            break
                        y_i = z[i] / denom
                        tf1 += y_i / k1[i]
                        tf2 += y_i / k2[i]
                    if good:
                        trial_norm = max(abs(tf1 - 1.0), abs(tf2 - 1.0))
                        if trial_norm < norm:
                            l1 = tl1
                            l2 = tl2
                            accepted = True
                            break
                damping *= 0.5
            if not accepted:
                break

        l1 = best_l1
        l2 = best_l2
        v = 1.0 - l1 - l2
        y = np.empty(n, dtype=np.float64)
        x1 = np.empty(n, dtype=np.float64)
        x2 = np.empty(n, dtype=np.float64)
        if best_norm > 1e-7 or v <= 1e-8 or l1 <= 1e-8 or l2 <= 1e-8:
            return False, v, l1, y, x1, x2, best_norm
        for i in range(n):
            denom = v + l1 / k1[i] + l2 / k2[i]
            y[i] = z[i] / denom
            x1[i] = y[i] / k1[i]
            x2[i] = y[i] / k2[i]
        return True, v, l1, _norm(y), _norm(x1), _norm(x2), best_norm


    @njit(cache=True)
    def _binary_invariant(z, T, P, x1, x2, nu, r, q, subgroup_q, interactions,
                          interactions_b, interactions_c, variant_id,
                          antoine_a, antoine_b, antoine_c):
        n = z.shape[0]
        y = np.zeros(n, dtype=np.float64)
        if n != 2:
            return False, 0.0, 0.0, y, 1e300
        p1 = _bubble_point_p_unifac(x1, T, nu, r, q, subgroup_q, interactions,
                                    interactions_b, interactions_c, variant_id,
                                    antoine_a, antoine_b, antoine_c)
        p2 = _bubble_point_p_unifac(x2, T, nu, r, q, subgroup_q, interactions,
                                    interactions_b, interactions_c, variant_id,
                                    antoine_a, antoine_b, antoine_c)
        residual = max(abs(p1 - P), abs(p2 - P))
        if residual > max(5e-5, 1e-4 * P):
            return False, 0.0, 0.0, y, residual
        k1 = _k_values_unifac(x1, T, P, nu, r, q, subgroup_q, interactions,
                              interactions_b, interactions_c, variant_id,
                              antoine_a, antoine_b, antoine_c)
        k2 = _k_values_unifac(x2, T, P, nu, r, q, subgroup_q, interactions,
                              interactions_b, interactions_c, variant_id,
                              antoine_a, antoine_b, antoine_c)
        y1 = _norm(x1 * k1)
        y2 = _norm(x2 * k2)
        if max(abs(y1[0] - y2[0]), abs(y1[1] - y2[1])) > 2e-4:
            return False, 0.0, 0.0, y, residual
        y = _norm(0.5 * (y1 + y2))
        denom = x1[0] - x2[0]
        if abs(denom) < 1e-14:
            return False, 0.0, 0.0, y, residual
        slope = (x2[0] - y[0]) / denom
        intercept = (z[0] - x2[0]) / denom
        v_low = 0.0
        v_high = 1.0
        # constraints: l1 >= 0, l2 >= 0, v in [0,1]
        if abs(slope) > 1e-14:
            root = -intercept / slope
            if slope > 0.0:
                if root > v_low:
                    v_low = root
            else:
                if root < v_high:
                    v_high = root
        slope_l2 = -1.0 - slope
        intercept_l2 = 1.0 - intercept
        if abs(slope_l2) > 1e-14:
            root = -intercept_l2 / slope_l2
            if slope_l2 > 0.0:
                if root > v_low:
                    v_low = root
            else:
                if root < v_high:
                    v_high = root
        if v_low < 0.0:
            v_low = 0.0
        if v_high > 1.0:
            v_high = 1.0
        if v_high <= v_low:
            return False, 0.0, 0.0, y, residual
        v = 0.5 * (v_low + v_high)
        l1 = intercept + slope * v
        l2 = 1.0 - v - l1
        if v <= 1e-8 or l1 <= 1e-8 or l2 <= 1e-8:
            return False, v, l1, y, residual
        return True, v, l1, y, residual


    @njit(cache=True)
    def _structured_vlle_from_seeds(z, T, P, x1_seed, x2_seed, beta_seed, v_seed,
                                    status_code, nu, r, q, subgroup_q, interactions,
                                    interactions_b, interactions_c, variant_id,
                                    antoine_a, antoine_b, antoine_c, max_iter, tol):
        n = z.shape[0]
        x1 = x1_seed.copy()
        x2 = x2_seed.copy()
        seed_v = v_seed
        if seed_v < 0.05:
            seed_v = 0.05
        elif seed_v > 0.85:
            seed_v = 0.85
        ltot = 1.0 - seed_v
        seed_l2 = ltot * beta_seed
        if seed_l2 < 1e-4:
            seed_l2 = 1e-4
        if seed_l2 > 0.9:
            seed_l2 = 0.9
        seed_l1 = ltot - seed_l2
        if seed_l1 < 1e-4:
            seed_l1 = 1e-4
        if seed_l1 + seed_l2 > 0.95:
            scale = 0.95 / (seed_l1 + seed_l2)
            seed_l1 *= scale
            seed_l2 *= scale

        y = z.copy()
        v = 0.0
        l1 = 0.0
        l2 = 0.0
        residual = 1e300
        vector_size = 2 * n
        previous_vector = np.zeros(vector_size, dtype=np.float64)
        previous_residual = np.zeros(vector_size, dtype=np.float64)
        has_previous = False
        last_delta = 1e300
        cooldown = 0
        acceleration_attempts = 0
        acceleration_accepts = 0
        inverse_jacobian = np.zeros((vector_size, vector_size), dtype=np.float64)
        for row in range(vector_size):
            inverse_jacobian[row, row] = -1.0

        for iteration in range(max_iter):
            current_vector = np.empty(vector_size, dtype=np.float64)
            for i in range(n):
                current_vector[i] = x1[i]
                current_vector[n + i] = x2[i]
            k1 = _k_values_unifac(x1, T, P, nu, r, q, subgroup_q, interactions,
                                  interactions_b, interactions_c, variant_id,
                                  antoine_a, antoine_b, antoine_c)
            k2 = _k_values_unifac(x2, T, P, nu, r, q, subgroup_q, interactions,
                                  interactions_b, interactions_c, variant_id,
                                  antoine_a, antoine_b, antoine_c)
            ok, v, l1, y_new, new_x1, new_x2, residual = _solve_three_phase_rr(
                z, k1, k2, seed_l1, seed_l2
            )
            if not ok:
                return False, 0, 7, 0.0, 0.0, 0.0, y, x1, x2, residual, iteration + 1
            l2 = 1.0 - v - l1
            delta = 0.0
            phase_diff = 0.0
            mapped_vector = np.empty(vector_size, dtype=np.float64)
            residual_vector = np.empty(vector_size, dtype=np.float64)
            for i in range(n):
                mapped_vector[i] = new_x1[i]
                mapped_vector[n + i] = new_x2[i]
            for i in range(vector_size):
                residual_vector[i] = mapped_vector[i] - current_vector[i]
                d = abs(residual_vector[i])
                if d > delta:
                    delta = d
            for i in range(n):
                phase_diff += abs(new_x1[i] - new_x2[i])
            phase_diff /= n
            if delta < tol and residual < 1e-9:
                if phase_diff < 1e-4:
                    return False, 0, 7, v, l1, l2, y_new, new_x1, new_x2, max(delta, residual), iteration + 1
                return True, 3, status_code, v, l1, l2, y_new, new_x1, new_x2, max(delta, residual), iteration + 1

            next_vector = mapped_vector
            if cooldown > 0:
                cooldown -= 1
            elif iteration >= 2 and has_previous:
                if delta > 2.5 * max(last_delta, 1e-16):
                    cooldown = 2
                else:
                    acceleration_attempts += 1
                    step = np.empty(vector_size, dtype=np.float64)
                    residual_change = np.empty(vector_size, dtype=np.float64)
                    denominator = 0.0
                    for i in range(vector_size):
                        step[i] = current_vector[i] - previous_vector[i]
                        residual_change[i] = residual_vector[i] - previous_residual[i]
                        denominator += residual_change[i] * residual_change[i]
                    if denominator > 1e-20:
                        jacobian_times_change = np.zeros(vector_size, dtype=np.float64)
                        for row in range(vector_size):
                            total = 0.0
                            for col in range(vector_size):
                                total += inverse_jacobian[row, col] * residual_change[col]
                            jacobian_times_change[row] = total
                        for row in range(vector_size):
                            scale_update = (step[row] - jacobian_times_change[row]) / denominator
                            for col in range(vector_size):
                                inverse_jacobian[row, col] += scale_update * residual_change[col]

                    trial_vector = np.empty(vector_size, dtype=np.float64)
                    trial_step = 0.0
                    for row in range(vector_size):
                        broyden_step = 0.0
                        for col in range(vector_size):
                            broyden_step -= inverse_jacobian[row, col] * residual_vector[col]
                        trial_vector[row] = current_vector[row] + broyden_step
                        d = abs(trial_vector[row] - current_vector[row])
                        if d > trial_step:
                            trial_step = d

                    valid_trial = True
                    for i in range(vector_size):
                        if not np.isfinite(trial_vector[i]):
                            valid_trial = False
                            break
                        if trial_vector[i] <= 1e-12:
                            valid_trial = False
                            break
                    if valid_trial and trial_step <= 4.0 * max(delta, 1e-14):
                        trial_x1 = np.empty(n, dtype=np.float64)
                        trial_x2 = np.empty(n, dtype=np.float64)
                        for i in range(n):
                            trial_x1[i] = trial_vector[i]
                            trial_x2[i] = trial_vector[n + i]
                        trial_x1 = _norm(trial_x1)
                        trial_x2 = _norm(trial_x2)
                        trial_phase_diff = 0.0
                        for i in range(n):
                            trial_phase_diff += abs(trial_x1[i] - trial_x2[i])
                        trial_phase_diff /= n
                        if trial_phase_diff > 1e-5:
                            next_vector = np.empty(vector_size, dtype=np.float64)
                            for i in range(n):
                                next_vector[i] = trial_x1[i]
                                next_vector[n + i] = trial_x2[i]
                            acceleration_accepts += 1

            for i in range(vector_size):
                previous_vector[i] = current_vector[i]
                previous_residual[i] = residual_vector[i]
            has_previous = True
            last_delta = delta
            x1_next = np.empty(n, dtype=np.float64)
            x2_next = np.empty(n, dtype=np.float64)
            for i in range(n):
                x1_next[i] = next_vector[i]
                x2_next[i] = next_vector[n + i]
            x1 = _norm(x1_next)
            x2 = _norm(x2_next)
            y = y_new
            seed_l1 = l1
            seed_l2 = l2
        return False, 0, 7, v, l1, l2, y, x1, x2, residual, max_iter


    @njit(cache=True)
    def _vlle_flash_tp_unifac_numba(z_input, T, P, nu, r, q, subgroup_q, interactions,
                                    interactions_b, interactions_c, variant_id,
                                    antoine_a, antoine_b, antoine_c, max_iter, tol):
        z = _norm(z_input)
        V, x_vle, y_vle = _vle_flash_unifac(
            z, T, P, nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, antoine_a, antoine_b, antoine_c,
            max_iter, tol
        )

        lle_converged, has_lle, feed_x1, feed_x2, feed_beta = _lle_split_numba(
            nu, r, q, subgroup_q, interactions, interactions_b, interactions_c,
            variant_id, z, T, max_iter, 1e-6
        )
        has_lle = lle_converged and has_lle
        if has_lle:
            if z.shape[0] == 2:
                ok, v, l1, y, residual = _binary_invariant(
                    z, T, P, feed_x1, feed_x2, nu, r, q, subgroup_q, interactions,
                    interactions_b, interactions_c, variant_id, antoine_a, antoine_b, antoine_c
                )
                if ok:
                    return 3, 3, v, l1, 1.0 - v - l1, y, feed_x1, feed_x2, residual, 1
            else:
                out = _structured_vlle_from_seeds(
                    z, T, P, feed_x1, feed_x2, feed_beta, V, 4,
                    nu, r, q, subgroup_q, interactions, interactions_b, interactions_c,
                    variant_id, antoine_a, antoine_b, antoine_c, max_iter, tol
                )
                if out[0]:
                    return out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10]

        if V >= 1.0 - 1e-10 and has_lle:
            stable, stable_v, stable_x, stable_y = _stable_vle_from_lle_seeds_unifac(
                z, T, P, feed_x1, feed_x2, nu, r, q, subgroup_q, interactions,
                interactions_b, interactions_c, variant_id, antoine_a, antoine_b,
                antoine_c, max_iter, tol
            )
            if stable:
                return 2, 2, stable_v, 1.0 - stable_v, 0.0, stable_y, stable_x, stable_x, 0.0, 0
        if V >= 1.0 - 1e-10:
            return 1, 1, 1.0, 0.0, 0.0, y_vle, x_vle, x_vle, 0.0, 0
        if V <= 1e-10:
            if has_lle:
                return 2, 6, 0.0, 1.0 - feed_beta, feed_beta, z, feed_x1, feed_x2, 0.0, 0
            return 1, 0, 0.0, 1.0, 0.0, z, z, z, 0.0, 0

        lle_converged, has_vle_lle, liquid_x1, liquid_x2, liquid_beta = _lle_split_numba(
            nu, r, q, subgroup_q, interactions, interactions_b, interactions_c,
            variant_id, x_vle, T, max_iter, 1e-6
        )
        has_vle_lle = lle_converged and has_vle_lle
        if has_vle_lle:
            out = _structured_vlle_from_seeds(
                z, T, P, liquid_x1, liquid_x2, liquid_beta, V, 5,
                nu, r, q, subgroup_q, interactions, interactions_b, interactions_c,
                variant_id, antoine_a, antoine_b, antoine_c, max_iter, tol
            )
            if out[0]:
                return out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10]

        return 2, 2, V, 1.0 - V, 0.0, y_vle, x_vle, x_vle, 0.0, 0


    @njit(cache=True)
    def _vlle_flash_pv_unifac_numba(z, P, vf_target, T_low, T_high, nu, r, q,
                                    subgroup_q, interactions, interactions_b,
                                    interactions_c, variant_id, antoine_a,
                                    antoine_b, antoine_c, max_iter, tol):
        low = T_low
        high = T_high
        raw_low = _vlle_flash_tp_unifac_numba(
            z, low, P, nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, antoine_a, antoine_b, antoine_c,
            max_iter, tol
        )
        raw_high = _vlle_flash_tp_unifac_numba(
            z, high, P, nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, antoine_a, antoine_b, antoine_c,
            max_iter, tol
        )
        f_low = raw_low[2] - vf_target
        f_high = raw_high[2] - vf_target
        if abs(f_low) < 1e-8:
            return low, raw_low
        if abs(f_high) < 1e-8:
            return high, raw_high
        if f_low * f_high > 0.0:
            if abs(f_low) <= abs(f_high):
                return low, raw_low
            return high, raw_high
        raw_mid = raw_low
        mid = low
        for _ in range(70):
            mid = 0.5 * (low + high)
            raw_mid = _vlle_flash_tp_unifac_numba(
                z, mid, P, nu, r, q, subgroup_q, interactions, interactions_b,
                interactions_c, variant_id, antoine_a, antoine_b, antoine_c,
                max_iter, tol
            )
            vf = raw_mid[2]
            f_mid = vf - vf_target
            if abs(f_mid) < 1e-8 or high - low < 1e-8:
                break
            if f_low * f_mid <= 0.0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        return mid, raw_mid


    @njit(cache=True)
    def _heterogeneous_boundary_residual(z, T, P, nu, r, q, subgroup_q, interactions,
                                         interactions_b, interactions_c, variant_id,
                                         antoine_a, antoine_b, antoine_c, max_iter, tol):
        lle_converged, has_lle, x1, x2, beta = _lle_split_numba(
            nu, r, q, subgroup_q, interactions, interactions_b, interactions_c,
            variant_id, z, T, max_iter, 1e-6
        )
        if not lle_converged or not has_lle:
            return False, 0.0
        p1 = _bubble_point_p_unifac(x1, T, nu, r, q, subgroup_q, interactions,
                                    interactions_b, interactions_c, variant_id,
                                    antoine_a, antoine_b, antoine_c)
        return True, p1 - P


    @njit(cache=True)
    def _heterogeneous_boundary_t_unifac_numba(z, P, T_low, T_high, step, nu, r, q,
                                               subgroup_q, interactions,
                                               interactions_b, interactions_c,
                                               variant_id, antoine_a, antoine_b,
                                               antoine_c, max_iter, tol):
        calls = 0
        last_T = 0.0
        last_f = 0.0
        have_last = False
        T = T_low
        bracket_low = 0.0
        bracket_high = 0.0
        while T <= T_high + 1e-12:
            ok, f = _heterogeneous_boundary_residual(
                z, T, P, nu, r, q, subgroup_q, interactions, interactions_b,
                interactions_c, variant_id, antoine_a, antoine_b, antoine_c,
                max_iter, tol
            )
            calls += 1
            if ok:
                if abs(f) < 1e-8:
                    return T, calls
                if have_last and f * last_f < 0.0:
                    bracket_low = last_T
                    bracket_high = T
                    break
                last_T = T
                last_f = f
                have_last = True
            T += step
        if bracket_low <= 0.0:
            return 0.0, calls
        low = bracket_low
        high = bracket_high
        f_low = last_f
        for _ in range(80):
            mid = 0.5 * (low + high)
            ok, f_mid = _heterogeneous_boundary_residual(
                z, mid, P, nu, r, q, subgroup_q, interactions, interactions_b,
                interactions_c, variant_id, antoine_a, antoine_b, antoine_c,
                max_iter, tol
            )
            calls += 1
            if not ok:
                low = mid
                continue
            if abs(f_mid) < 1e-9 or high - low < 1e-9:
                return mid, calls
            if f_low * f_mid <= 0.0:
                high = mid
            else:
                low = mid
                f_low = f_mid
        return 0.5 * (low + high), calls


    @njit(cache=True)
    def _bubble_residual_binary(x0, T, P, nu, r, q, subgroup_q, interactions,
                                interactions_b, interactions_c, variant_id,
                                antoine_a, antoine_b, antoine_c):
        x = np.empty(2, dtype=np.float64)
        x[0] = x0
        x[1] = 1.0 - x0
        return _bubble_point_p_unifac(x, T, nu, r, q, subgroup_q, interactions,
                                      interactions_b, interactions_c, variant_id,
                                      antoine_a, antoine_b, antoine_c) - P


    @njit(cache=True)
    def _bubble_t_binary(x0, P, T_low, T_high, nu, r, q, subgroup_q, interactions,
                         interactions_b, interactions_c, variant_id,
                         antoine_a, antoine_b, antoine_c):
        low = T_low
        high = T_high
        f_low = _bubble_residual_binary(x0, low, P, nu, r, q, subgroup_q, interactions,
                                        interactions_b, interactions_c, variant_id,
                                        antoine_a, antoine_b, antoine_c)
        f_high = _bubble_residual_binary(x0, high, P, nu, r, q, subgroup_q, interactions,
                                         interactions_b, interactions_c, variant_id,
                                         antoine_a, antoine_b, antoine_c)
        if f_low * f_high > 0.0:
            return 0.0
        for _ in range(70):
            mid = 0.5 * (low + high)
            f_mid = _bubble_residual_binary(x0, mid, P, nu, r, q, subgroup_q, interactions,
                                            interactions_b, interactions_c, variant_id,
                                            antoine_a, antoine_b, antoine_c)
            if abs(f_mid) < 1e-9 or high - low < 1e-9:
                return mid
            if f_low * f_mid <= 0.0:
                high = mid
            else:
                low = mid
                f_low = f_mid
        return 0.5 * (low + high)


    @njit(cache=True)
    def _homogeneous_binary_azeotrope_t_unifac_numba(P, T_low, T_high, nu, r, q,
                                                     subgroup_q, interactions,
                                                     interactions_b, interactions_c,
                                                     variant_id, antoine_a,
                                                     antoine_b, antoine_c, max_iter):
        calls = 0
        last_x = 0.0
        last_f = 0.0
        have_last = False
        bracket_low = 0.0
        bracket_high = 0.0
        for index in range(1, 1000):
            x0 = index / 1000.0
            T = _bubble_t_binary(x0, P, T_low, T_high, nu, r, q, subgroup_q,
                                 interactions, interactions_b, interactions_c,
                                 variant_id, antoine_a, antoine_b, antoine_c)
            calls += 1
            if T <= 0.0:
                continue
            x = np.empty(2, dtype=np.float64)
            x[0] = x0
            x[1] = 1.0 - x0
            K = _k_values_unifac(x, T, P, nu, r, q, subgroup_q, interactions,
                                 interactions_b, interactions_c, variant_id,
                                 antoine_a, antoine_b, antoine_c)
            f = np.log(K[0] / K[1])
            if abs(f) < 1e-8:
                return T, x, calls
            if have_last and f * last_f < 0.0:
                bracket_low = last_x
                bracket_high = x0
                break
            have_last = True
            last_x = x0
            last_f = f
        if bracket_low <= 0.0:
            return 0.0, np.zeros(2, dtype=np.float64), calls
        low = bracket_low
        high = bracket_high
        f_low = last_f
        root_x = 0.5 * (low + high)
        root_T = 0.0
        for _ in range(max_iter):
            root_x = 0.5 * (low + high)
            root_T = _bubble_t_binary(root_x, P, T_low, T_high, nu, r, q, subgroup_q,
                                      interactions, interactions_b, interactions_c,
                                      variant_id, antoine_a, antoine_b, antoine_c)
            calls += 1
            x = np.empty(2, dtype=np.float64)
            x[0] = root_x
            x[1] = 1.0 - root_x
            K = _k_values_unifac(x, root_T, P, nu, r, q, subgroup_q, interactions,
                                 interactions_b, interactions_c, variant_id,
                                 antoine_a, antoine_b, antoine_c)
            f_mid = np.log(K[0] / K[1])
            if abs(f_mid) < 1e-10 or high - low < 1e-12:
                return root_T, x, calls
            if f_low * f_mid <= 0.0:
                high = root_x
            else:
                low = root_x
                f_low = f_mid
        x = np.empty(2, dtype=np.float64)
        x[0] = root_x
        x[1] = 1.0 - root_x
        return root_T, x, calls

else:

    def _vlle_flash_tp_unifac_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Compiled VLLE backend is unavailable")


if (
    njit is not None
    and _nrtl_activity_coefficients_numba is not None
    and _uniquac_activity_coefficients_numba is not None
    and _lle_split_nrtl_numba is not None
    and _lle_split_uniquac_numba is not None
):

    @njit(cache=True)
    def _activity_gamma(model_id, x, T, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9):
        n = x.shape[0]
        if model_id == 1:
            return _nrtl_activity_coefficients_numba(
                x, T, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7,
                p8, p9,
            )
        r = np.empty(n, dtype=np.float64)
        q = np.empty(n, dtype=np.float64)
        q_residual = np.empty(n, dtype=np.float64)
        interaction_tmin = np.empty((n, n), dtype=np.float64)
        interaction_tmax = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            r[i] = p0[i, i]
            q[i] = p1[i, i]
            q_residual[i] = p2[i, i]
            for j in range(n):
                if i == j:
                    interaction_tmin[i, j] = -np.inf
                    interaction_tmax[i, j] = np.inf
                elif i < j:
                    interaction_tmin[i, j] = p9[i, j]
                    interaction_tmax[i, j] = p9[j, i]
                else:
                    interaction_tmin[i, j] = p9[j, i]
                    interaction_tmax[i, j] = p9[i, j]
        return _uniquac_activity_coefficients_numba(
            x, T, r, q, q_residual, integer_parameters,
            p3, p4, p5, p6, p7, p8,
            interaction_tmin, interaction_tmax,
        )


    @njit(cache=True)
    def _activity_lle_split(model_id, z, T, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                            max_iter, tol):
        if model_id == 1:
            return _lle_split_nrtl_numba(
                z, T, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7,
                p8, p9, max_iter, tol
            )
        n = z.shape[0]
        r = np.empty(n, dtype=np.float64)
        q = np.empty(n, dtype=np.float64)
        q_residual = np.empty(n, dtype=np.float64)
        interaction_tmin = np.empty((n, n), dtype=np.float64)
        interaction_tmax = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            r[i] = p0[i, i]
            q[i] = p1[i, i]
            q_residual[i] = p2[i, i]
            for j in range(n):
                if i == j:
                    interaction_tmin[i, j] = -np.inf
                    interaction_tmax[i, j] = np.inf
                elif i < j:
                    interaction_tmin[i, j] = p9[i, j]
                    interaction_tmax[i, j] = p9[j, i]
                else:
                    interaction_tmin[i, j] = p9[j, i]
                    interaction_tmax[i, j] = p9[i, j]
        return _lle_split_uniquac_numba(
            z, T, r, q, q_residual, integer_parameters,
            p3, p4, p5, p6, p7, p8,
            interaction_tmin, interaction_tmax, max_iter, tol
        )


    @njit(cache=True)
    def _k_values_activity(x, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4,
                           p5, p6, p7, p8, p9, antoine_a, antoine_b, antoine_c):
        gamma = _activity_gamma(
            model_id, x, T, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9
        )
        psat = _psat_array(T, antoine_a, antoine_b, antoine_c)
        out = np.empty(x.shape[0], dtype=np.float64)
        pressure = max(P, 1e-12)
        for i in range(x.shape[0]):
            value = gamma[i] * psat[i] / pressure
            out[i] = min(max(value, 1e-8), 1e8)
        return out


    @njit(cache=True)
    def _bubble_point_p_activity(x, T, model_id, integer_parameters, p0, p1, p2, p3,
                                 p4, p5, p6, p7, p8, p9,
                                 antoine_a, antoine_b, antoine_c):
        xn = _norm(x)
        gamma = _activity_gamma(
            model_id, xn, T, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9
        )
        psat = _psat_array(T, antoine_a, antoine_b, antoine_c)
        total = 0.0
        for i in range(xn.shape[0]):
            total += xn[i] * gamma[i] * psat[i]
        return total

    @njit(cache=True)
    def _vle_flash_activity(z_input, T, P, model_id, integer_parameters, p0, p1, p2,
                            p3, p4, p5, p6, p7, p8, p9,
                            antoine_a, antoine_b, antoine_c, max_iter, tol):
        z = _norm(z_input)
        x = z.copy()
        y = z.copy()
        V = 0.5
        K = _k_values_activity(
            x, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c
        )
        for _iteration in range(max_iter):
            old_x = x.copy()
            V = _rr_vle(z, K, V)
            for i in range(z.shape[0]):
                denom = 1.0 + V * (K[i] - 1.0)
                if abs(denom) < 1e-14:
                    denom = 1e-14
                x[i] = z[i] / denom
                y[i] = K[i] * x[i]
            x = _norm(x)
            y = _norm(y)
            K_new = _k_values_activity(
                x, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                antoine_a, antoine_b, antoine_c
            )
            change = 0.0
            for i in range(z.shape[0]):
                change = max(change, abs(x[i] - old_x[i]))
            K = K_new
            if change < tol:
                break
        V = _rr_vle(z, K, V)
        for i in range(z.shape[0]):
            denom = 1.0 + V * (K[i] - 1.0)
            if abs(denom) < 1e-14:
                denom = 1e-14
            x[i] = z[i] / denom
            y[i] = K[i] * x[i]
        x = _norm(x)
        y = _norm(y)
        return V, x, y


    @njit(cache=True)
    def _seeded_vle_activity(z_input, T, P, liquid_seed, model_id,
                             integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                             antoine_a, antoine_b, antoine_c, max_iter, tol):
        z = _norm(z_input)
        x = _norm(liquid_seed)
        y = z.copy()
        K = _k_values_activity(
            x, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c
        )
        V = 0.5
        convergence_tolerance = max(tol, 1e-9)
        converged = False
        for _iteration in range(max_iter):
            V = _rr_vle(z, K, V)
            for i in range(z.shape[0]):
                denom = 1.0 + V * (K[i] - 1.0)
                if abs(denom) < 1e-14:
                    denom = 1e-14
                x[i] = z[i] / denom
                y[i] = K[i] * x[i]
            x = _norm(x)
            y = _norm(y)
            K_new = _k_values_activity(
                x, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                antoine_a, antoine_b, antoine_c
            )
            error = 0.0
            for i in range(z.shape[0]):
                value = abs(math.log(max(K_new[i], 1e-300) / max(K[i], 1e-300)))
                if value > error:
                    error = value
                K[i] = math.sqrt(max(K[i], 1e-300) * max(K_new[i], 1e-300))
            if error < convergence_tolerance:
                converged = True
                break

        V = _rr_vle(z, K, V)
        for i in range(z.shape[0]):
            denom = 1.0 + V * (K[i] - 1.0)
            if abs(denom) < 1e-14:
                denom = 1e-14
            x[i] = z[i] / denom
            y[i] = K[i] * x[i]
        x = _norm(x)
        y = _norm(y)
        if not converged or V <= 1e-10 or V >= 1.0 - 1e-10:
            return False, V, x, y

        K_check = _k_values_activity(
            x, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c
        )
        residual = 0.0
        for i in range(z.shape[0]):
            value = abs(math.log(
                max(y[i], 1e-300) / max(x[i] * K_check[i], 1e-300)
            ))
            if value > residual:
                residual = value
        if residual > max(1e-7, 100.0 * convergence_tolerance):
            return False, V, x, y
        return True, V, x, y


    @njit(cache=True)
    def _liquid_tpd_activity(z_input, T, P, liquid_seed, model_id,
                             integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                             antoine_a, antoine_b, antoine_c):
        z = _norm(z_input)
        x = _norm(liquid_seed)
        K = _k_values_activity(
            x, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c
        )
        value = 0.0
        for i in range(z.shape[0]):
            value += x[i] * math.log(
                max(x[i] * K[i], 1e-300) / max(z[i], 1e-300)
            )
        return value


    @njit(cache=True)
    def _reduced_gibbs_activity(T, P, V, x_input, y_input, model_id,
                                integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                                antoine_a, antoine_b, antoine_c):
        x = _norm(x_input)
        y = _norm(y_input)
        pressure = max(P, 1e-300)
        value = 0.0
        for i in range(y.shape[0]):
            value += V * y[i] * math.log(max(y[i] * pressure, 1e-300))
        if V >= 1.0 - 1e-12:
            return value
        K = _k_values_activity(
            x, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c
        )
        for i in range(x.shape[0]):
            value += (1.0 - V) * x[i] * math.log(
                max(x[i] * K[i] * pressure, 1e-300)
            )
        return value


    @njit(cache=True)
    def _stable_vle_from_lle_seeds_activity(
        z, T, P, seed1, seed2, model_id, integer_parameters, p0, p1, p2,
        p3, p4, p5, p6, p7, p8, p9,
        antoine_a, antoine_b, antoine_c, max_iter, tol,
    ):
        tpd1 = _liquid_tpd_activity(
            z, T, P, seed1, model_id, integer_parameters, p0, p1, p2, p3,
            p4, p5, p6, p7, p8, p9, antoine_a, antoine_b, antoine_c
        )
        tpd2 = _liquid_tpd_activity(
            z, T, P, seed2, model_id, integer_parameters, p0, p1, p2, p3,
            p4, p5, p6, p7, p8, p9, antoine_a, antoine_b, antoine_c
        )
        seed = seed1 if tpd1 <= tpd2 else seed2
        tpd = min(tpd1, tpd2)
        empty = np.zeros(z.shape[0], dtype=np.float64)
        if not np.isfinite(tpd) or tpd >= -max(1e-8, 10.0 * tol):
            return False, 1.0, empty, empty

        ok, V, x, y = _seeded_vle_activity(
            z, T, P, seed, model_id, integer_parameters, p0, p1, p2, p3,
            p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c, max_iter, tol
        )
        if not ok:
            return False, V, x, y
        candidate_gibbs = _reduced_gibbs_activity(
            T, P, V, x, y, model_id, integer_parameters, p0, p1, p2, p3,
            p4, p5, p6, p7, p8, p9, antoine_a, antoine_b, antoine_c
        )
        vapor_gibbs = _reduced_gibbs_activity(
            T, P, 1.0, z, z, model_id, integer_parameters, p0, p1, p2, p3,
            p4, p5, p6, p7, p8, p9, antoine_a, antoine_b, antoine_c
        )
        if not np.isfinite(candidate_gibbs) or candidate_gibbs - vapor_gibbs >= -1e-10:
            return False, V, x, y
        return True, V, x, y


    @njit(cache=True)
    def _binary_invariant_activity(z, T, P, x1, x2, model_id, integer_parameters,
                                   p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, antoine_a, antoine_b,
                                   antoine_c):
        n = z.shape[0]
        y = np.zeros(n, dtype=np.float64)
        if n != 2:
            return False, 0.0, 0.0, y, 1e300
        p_x1 = _bubble_point_p_activity(
            x1, T, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c
        )
        p_x2 = _bubble_point_p_activity(
            x2, T, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c
        )
        pressure_residual = max(abs(p_x1 - P), abs(p_x2 - P))
        if pressure_residual > max(5e-4, 5e-4 * P):
            return False, 0.0, 0.0, y, pressure_residual
        k1 = _k_values_activity(
            x1, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c
        )
        k2 = _k_values_activity(
            x2, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c
        )
        y1 = _norm(x1 * k1)
        y2 = _norm(x2 * k2)
        vapor_residual = max(abs(y1[0] - y2[0]), abs(y1[1] - y2[1]))
        if vapor_residual > 2e-4:
            return False, 0.0, 0.0, y, max(pressure_residual, vapor_residual)
        y = _norm(0.5 * (y1 + y2))
        denom = x1[0] - x2[0]
        if abs(denom) < 1e-14:
            return False, 0.0, 0.0, y, pressure_residual
        slope = (x2[0] - y[0]) / denom
        intercept = (z[0] - x2[0]) / denom
        v_low = 0.0
        v_high = 1.0
        if abs(slope) > 1e-14:
            root = -intercept / slope
            if slope > 0.0:
                v_low = max(v_low, root)
            else:
                v_high = min(v_high, root)
        slope_l2 = -1.0 - slope
        intercept_l2 = 1.0 - intercept
        if abs(slope_l2) > 1e-14:
            root = -intercept_l2 / slope_l2
            if slope_l2 > 0.0:
                v_low = max(v_low, root)
            else:
                v_high = min(v_high, root)
        v_low = max(0.0, v_low)
        v_high = min(1.0, v_high)
        if v_high <= v_low:
            return False, 0.0, 0.0, y, pressure_residual
        v = 0.5 * (v_low + v_high)
        l1 = intercept + slope * v
        l2 = 1.0 - v - l1
        if min(v, l1, l2) <= 1e-8:
            return False, v, l1, y, pressure_residual
        return True, v, l1, y, max(pressure_residual, vapor_residual)


    @njit(cache=True)
    def _structured_vlle_activity(z, T, P, x1_seed, x2_seed, beta_seed, v_seed,
                                  status_code, model_id, integer_parameters, p0, p1,
                                  p2, p3, p4, p5, p6, p7, p8, p9,
                                  antoine_a, antoine_b, antoine_c,
                                  max_iter, tol):
        n = z.shape[0]
        x1 = _norm(x1_seed)
        x2 = _norm(x2_seed)
        seed_v = min(max(v_seed, 0.05), 0.85)
        liquid_total = 1.0 - seed_v
        seed_l2 = min(max(liquid_total * beta_seed, 1e-4), 0.9)
        seed_l1 = max(liquid_total - seed_l2, 1e-4)
        if seed_l1 + seed_l2 > 0.95:
            scale = 0.95 / (seed_l1 + seed_l2)
            seed_l1 *= scale
            seed_l2 *= scale
        size = 2 * n
        previous_vector = np.zeros(size, dtype=np.float64)
        previous_residual = np.zeros(size, dtype=np.float64)
        inverse_jacobian = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            inverse_jacobian[i, i] = -1.0
        has_previous = False
        last_delta = 1e300
        cooldown = 0
        y = z.copy()
        v = 0.0
        l1 = 0.0
        l2 = 0.0
        rr_residual = 1e300
        for iteration in range(max_iter):
            current = np.empty(size, dtype=np.float64)
            for i in range(n):
                current[i] = x1[i]
                current[n + i] = x2[i]
            k1 = _k_values_activity(
                x1, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                antoine_a, antoine_b, antoine_c
            )
            k2 = _k_values_activity(
                x2, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                antoine_a, antoine_b, antoine_c
            )
            ok, v, l1, y_new, x1_new, x2_new, rr_residual = _solve_three_phase_rr(
                z, k1, k2, seed_l1, seed_l2
            )
            if not ok:
                return False, 0, 7, 0.0, 0.0, 0.0, y, x1, x2, rr_residual, iteration + 1
            l2 = 1.0 - v - l1
            mapped = np.empty(size, dtype=np.float64)
            residual_vector = np.empty(size, dtype=np.float64)
            delta = 0.0
            phase_diff = 0.0
            for i in range(n):
                mapped[i] = x1_new[i]
                mapped[n + i] = x2_new[i]
                phase_diff += abs(x1_new[i] - x2_new[i])
            phase_diff /= n
            for i in range(size):
                residual_vector[i] = mapped[i] - current[i]
                delta = max(delta, abs(residual_vector[i]))
            if delta < tol and rr_residual < max(1e-9, tol):
                if phase_diff < 1e-4:
                    return False, 0, 7, v, l1, l2, y_new, x1_new, x2_new, max(delta, rr_residual), iteration + 1
                return True, 3, status_code, v, l1, l2, y_new, x1_new, x2_new, max(delta, rr_residual), iteration + 1

            next_vector = mapped
            if cooldown > 0:
                cooldown -= 1
            elif iteration >= 2 and has_previous:
                if delta > 2.5 * max(last_delta, 1e-16):
                    cooldown = 2
                else:
                    step = current - previous_vector
                    residual_change = residual_vector - previous_residual
                    denominator = 0.0
                    for i in range(size):
                        denominator += residual_change[i] * residual_change[i]
                    if denominator > 1e-20:
                        jacobian_times_change = inverse_jacobian @ residual_change
                        correction = step - jacobian_times_change
                        for row in range(size):
                            scale_update = correction[row] / denominator
                            for col in range(size):
                                inverse_jacobian[row, col] += scale_update * residual_change[col]
                    trial = current - inverse_jacobian @ residual_vector
                    trial_step = 0.0
                    valid = True
                    for i in range(size):
                        trial_step = max(trial_step, abs(trial[i] - current[i]))
                        if not np.isfinite(trial[i]) or trial[i] <= 1e-12:
                            valid = False
                    if valid and trial_step <= 4.0 * max(delta, 1e-14):
                        tx1 = _norm(trial[:n])
                        tx2 = _norm(trial[n:])
                        trial_phase_diff = 0.0
                        for i in range(n):
                            trial_phase_diff += abs(tx1[i] - tx2[i])
                        if trial_phase_diff / n > 1e-5:
                            next_vector = np.empty(size, dtype=np.float64)
                            for i in range(n):
                                next_vector[i] = tx1[i]
                                next_vector[n + i] = tx2[i]
            previous_vector = current
            previous_residual = residual_vector
            has_previous = True
            last_delta = delta
            x1 = _norm(next_vector[:n])
            x2 = _norm(next_vector[n:])
            y = y_new
            seed_l1 = l1
            seed_l2 = l2
        return False, 0, 7, v, l1, l2, y, x1, x2, rr_residual, max_iter


    @njit(cache=True)
    def _vlle_flash_tp_activity_numba(z_input, T, P, model_id, integer_parameters,
                                      p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, antoine_a, antoine_b,
                                      antoine_c, max_iter, tol):
        z = _norm(z_input)
        V, x_vle, y_vle = _vle_flash_activity(
            z, T, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c, max_iter, tol
        )
        lle_converged, has_lle, feed_x1, feed_x2, feed_beta = _activity_lle_split(
            model_id, z, T, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            max_iter, 1e-6
        )
        has_lle = lle_converged and has_lle
        if has_lle:
            if z.shape[0] == 2:
                ok, v, l1, y, residual = _binary_invariant_activity(
                    z, T, P, feed_x1, feed_x2, model_id, integer_parameters,
                    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, antoine_a, antoine_b, antoine_c
                )
                if ok:
                    return 3, 3, v, l1, 1.0 - v - l1, y, feed_x1, feed_x2, residual, 1
            else:
                out = _structured_vlle_activity(
                    z, T, P, feed_x1, feed_x2, feed_beta, V, 4, model_id,
                    integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, antoine_a,
                    antoine_b, antoine_c, max_iter, tol
                )
                if out[0]:
                    return out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10]
        if V >= 1.0 - 1e-10 and has_lle:
            stable, stable_v, stable_x, stable_y = _stable_vle_from_lle_seeds_activity(
                z, T, P, feed_x1, feed_x2, model_id, integer_parameters,
                p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, antoine_a, antoine_b, antoine_c,
                max_iter, tol
            )
            if stable:
                return 2, 2, stable_v, 1.0 - stable_v, 0.0, stable_y, stable_x, stable_x, 0.0, 0
        if V >= 1.0 - 1e-10:
            return 1, 1, 1.0, 0.0, 0.0, y_vle, x_vle, x_vle, 0.0, 0
        if V <= 1e-10:
            if has_lle:
                return 2, 6, 0.0, 1.0 - feed_beta, feed_beta, z, feed_x1, feed_x2, 0.0, 0
            return 1, 0, 0.0, 1.0, 0.0, z, z, z, 0.0, 0
        lle_converged, has_liquid_lle, liquid_x1, liquid_x2, liquid_beta = _activity_lle_split(
            model_id, x_vle, T, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            max_iter, 1e-6
        )
        has_liquid_lle = lle_converged and has_liquid_lle
        if has_liquid_lle:
            out = _structured_vlle_activity(
                z, T, P, liquid_x1, liquid_x2, liquid_beta, V, 5, model_id,
                integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, antoine_a,
                antoine_b, antoine_c, max_iter, tol
            )
            if out[0]:
                return out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10]
        return 2, 2, V, 1.0 - V, 0.0, y_vle, x_vle, x_vle, 0.0, 0


    @njit(cache=True)
    def _vlle_flash_pv_activity_numba(z, P, target, T_low, T_high, model_id,
                                      integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                                      antoine_a, antoine_b, antoine_c, max_iter, tol):
        low = T_low
        high = T_high
        low_result = _vlle_flash_tp_activity_numba(
            z, low, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c, max_iter, tol
        )
        high_result = _vlle_flash_tp_activity_numba(
            z, high, P, model_id, integer_parameters, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
            antoine_a, antoine_b, antoine_c, max_iter, tol
        )
        f_low = low_result[2] - target
        f_high = high_result[2] - target
        if abs(f_low) < 1e-8:
            return low, low_result
        if abs(f_high) < 1e-8:
            return high, high_result
        if f_low * f_high > 0.0:
            if abs(f_low) <= abs(f_high):
                return low, low_result
            return high, high_result
        result = low_result
        mid = low
        for _ in range(70):
            mid = 0.5 * (low + high)
            result = _vlle_flash_tp_activity_numba(
                z, mid, P, model_id, integer_parameters, p0, p1, p2, p3, p4,
                p5, p6, p7, p8, p9,
                antoine_a, antoine_b, antoine_c, max_iter, tol
            )
            f_mid = result[2] - target
            if abs(f_mid) < 1e-8 or high - low < 1e-8:
                break
            if f_low * f_mid <= 0.0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid
        return mid, result

else:

    def _vlle_flash_tp_activity_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Compiled activity VLLE backend is unavailable")

    def _vlle_flash_pv_activity_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Compiled activity VLLE backend is unavailable")
