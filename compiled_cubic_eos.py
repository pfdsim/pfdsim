"""Compiled kernels for fixed-component SRK/PR-family cubic EOS models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .physical_constants import R_BAR_CM3_MOL_K
else:
    from physical_constants import R_BAR_CM3_MOL_K

try:
    from numba import njit, typeof
except Exception:  # pragma: no cover - optional dependency fallback
    njit = None
    typeof = None


R_CM3 = R_BAR_CM3_MOL_K
ALPHA_SOAVE = 0
ALPHA_BOSTON_MATHIAS = 1
ALPHA_MATHIAS_COPEMAN = 2
ALPHA_PRSV1 = 3
ALPHA_PRSV2 = 4
ALPHA_TWU = 5
ALPHA_REDLICH_KWONG = 6
ALPHA_RKSMHV2_MATHIAS_COPEMAN = 7


def compiled_alpha_parameter_arrays(eos, *, mhv2: bool = False):
    """Build the shared fixed-component alpha arrays for compiled EOS kernels."""
    modes = []
    c1 = []
    c2 = []
    c3 = []
    params_values = [eos.params[component] for component in eos.components]
    for params in params_values:
        if mhv2 and eos._has_mc_constants(params):
            mode = ALPHA_RKSMHV2_MATHIAS_COPEMAN
            constants = (params.mc_c1, params.mc_c2, params.mc_c3)
        elif eos.use_twu and eos._has_twu_constants(params):
            mode = ALPHA_TWU
            constants = (params.twu_l, params.twu_m, params.twu_n)
        elif eos.use_prsv2 and eos._has_prsv_constants(params):
            mode = ALPHA_PRSV2
            constants = (params.kappa1, params.kappa2, params.kappa3)
        elif eos.use_prsv1 and eos._has_prsv_constants(params):
            mode = ALPHA_PRSV1
            constants = (params.kappa1, 0.0, 0.0)
        elif eos.use_mathias_copeman and eos._has_mc_constants(params):
            mode = ALPHA_MATHIAS_COPEMAN
            constants = (params.mc_c1, params.mc_c2, params.mc_c3)
        elif eos.use_boston_mathias or eos.use_mathias_copeman:
            mode = ALPHA_BOSTON_MATHIAS
            constants = (0.0, 0.0, 0.0)
        else:
            mode = ALPHA_SOAVE
            constants = (0.0, 0.0, 0.0)
        modes.append(mode)
        c1.append(float(constants[0]))
        c2.append(float(constants[1]))
        c3.append(float(constants[2]))
    return (
        params_values,
        np.asarray(modes, dtype=np.int64),
        np.asarray(c1, dtype=np.float64),
        np.asarray(c2, dtype=np.float64),
        np.asarray(c3, dtype=np.float64),
    )


@dataclass
class CompiledCubicEOSBackend:
    components: tuple[str, ...]
    Tc: np.ndarray
    Pc: np.ndarray
    omega: np.ndarray
    a0: np.ndarray
    pure_b: np.ndarray
    alpha_mode: np.ndarray
    c1: np.ndarray
    c2: np.ndarray
    c3: np.ndarray
    delta1: float
    delta2: float
    compilation_complete: bool = False

    @classmethod
    def from_eos(cls, eos) -> "CompiledCubicEOSBackend | None":
        if njit is None:
            return None
        params_values, modes, c1, c2, c3 = compiled_alpha_parameter_arrays(eos)
        delta1, delta2 = eos._delta_roots()
        backend = cls(
            components=tuple(eos.components),
            Tc=np.asarray([params.Tc for params in params_values], dtype=np.float64),
            Pc=np.asarray([params.Pc for params in params_values], dtype=np.float64),
            omega=np.asarray([params.omega for params in params_values], dtype=np.float64),
            a0=np.asarray([params.a0 for params in params_values], dtype=np.float64),
            pure_b=np.asarray([params.b for params in params_values], dtype=np.float64),
            alpha_mode=modes,
            c1=c1,
            c2=c2,
            c3=c3,
            delta1=float(delta1),
            delta2=float(delta2),
        )
        backend.compile_kernels()
        return backend

    @classmethod
    def from_rk(cls, rk) -> "CompiledCubicEOSBackend | None":
        if njit is None:
            return None
        params_values = [rk.params[component] for component in rk.components]
        backend = cls(
            components=tuple(rk.components),
            Tc=np.asarray([params.Tc for params in params_values], dtype=np.float64),
            Pc=np.asarray([params.Pc for params in params_values], dtype=np.float64),
            omega=np.asarray([
                rk.props[component].omega or 0.0
                for component in rk.components
            ], dtype=np.float64),
            a0=np.asarray([params.a for params in params_values], dtype=np.float64),
            pure_b=np.asarray([params.b for params in params_values], dtype=np.float64),
            alpha_mode=np.full(
                len(params_values), ALPHA_REDLICH_KWONG, dtype=np.int64
            ),
            c1=np.zeros(len(params_values), dtype=np.float64),
            c2=np.zeros(len(params_values), dtype=np.float64),
            c3=np.zeros(len(params_values), dtype=np.float64),
            delta1=1.0,
            delta2=0.0,
        )
        backend.compile_kernels()
        return backend

    def compile_kernels(self) -> None:
        """Compile/load kernels by signature without evaluating a state."""
        if self.compilation_complete or typeof is None:
            return
        count = len(self.components)
        composition = np.zeros(count, dtype=np.float64)
        matrix = np.zeros((count, count), dtype=np.float64)

        def compile_for(dispatcher, *args):
            dispatcher.compile(tuple(typeof(argument) for argument in args))

        common = (
            self.Tc, self.omega, self.a0, self.pure_b,
            self.alpha_mode, self.c1, self.c2, self.c3,
        )
        compile_for(
            _cubic_mixture_numba,
            300.0, composition, matrix, matrix, *common, self.delta1,
        )
        compile_for(
            _cubic_roots_state_numba,
            300.0, 1.0, composition, matrix, *common,
            self.delta1, self.delta2,
        )
        compile_for(
            _cubic_fugacity_numba,
            300.0, 1.0, composition, 0, matrix, *common,
            self.delta1, self.delta2,
        )
        compile_for(
            _cubic_departure_enthalpy_numba,
            300.0, 1.0, composition, 0, matrix, matrix, *common,
            self.delta1, self.delta2,
        )
        compile_for(
            _cubic_departure_entropy_numba,
            300.0, 1.0, composition, 0, matrix, matrix, *common,
            self.delta1, self.delta2,
        )
        compile_for(
            _cubic_phi_phi_k_numba,
            300.0, 1.0, composition, 2, matrix,
            self.Tc, self.Pc, self.omega, self.a0, self.pure_b,
            self.alpha_mode, self.c1, self.c2, self.c3,
            self.delta1, self.delta2,
        )
        self.compilation_complete = True

    def _arrays(self, composition, kij, dkij=None):
        x = np.asarray(composition, dtype=np.float64)
        kij_array = np.asarray(kij, dtype=np.float64).reshape(
            (len(self.components), len(self.components))
        )
        if dkij is None:
            dkij_array = np.zeros_like(kij_array)
        else:
            dkij_array = np.asarray(dkij, dtype=np.float64).reshape(kij_array.shape)
        return x, kij_array, dkij_array

    def mixture_parameters(self, T, composition, kij, dkij=None):
        x, kij_array, dkij_array = self._arrays(composition, kij, dkij)
        return _cubic_mixture_numba(
            float(T), x, kij_array, dkij_array,
            self.Tc, self.omega, self.a0, self.pure_b,
            self.alpha_mode, self.c1, self.c2, self.c3,
            self.delta1,
        )

    def compressibility_roots(self, T, P, composition, kij):
        x, kij_array, _ = self._arrays(composition, kij)
        roots, count = _cubic_roots_state_numba(
            float(T), float(P), x, kij_array,
            self.Tc, self.omega, self.a0, self.pure_b,
            self.alpha_mode, self.c1, self.c2, self.c3,
            self.delta1, self.delta2,
        )
        return roots[:count].copy()

    def fugacity_coefficients(self, T, P, composition, phase, kij):
        x, kij_array, _ = self._arrays(composition, kij)
        phase_id = 1 if str(phase).strip().lower() == 'vapor' else 0
        return _cubic_fugacity_numba(
            float(T), float(P), x, phase_id, kij_array,
            self.Tc, self.omega, self.a0, self.pure_b,
            self.alpha_mode, self.c1, self.c2, self.c3,
            self.delta1, self.delta2,
        )

    def departure_enthalpy(self, T, P, composition, phase, kij, dkij):
        x, kij_array, dkij_array = self._arrays(composition, kij, dkij)
        phase_id = 1 if str(phase).strip().lower() == 'vapor' else 0
        return float(_cubic_departure_enthalpy_numba(
            float(T), float(P), x, phase_id, kij_array, dkij_array,
            self.Tc, self.omega, self.a0, self.pure_b,
            self.alpha_mode, self.c1, self.c2, self.c3,
            self.delta1, self.delta2,
        ))

    def departure_entropy(self, T, P, composition, phase, kij, dkij):
        x, kij_array, dkij_array = self._arrays(composition, kij, dkij)
        phase_id = 1 if str(phase).strip().lower() == 'vapor' else 0
        return float(_cubic_departure_entropy_numba(
            float(T), float(P), x, phase_id, kij_array, dkij_array,
            self.Tc, self.omega, self.a0, self.pure_b,
            self.alpha_mode, self.c1, self.c2, self.c3,
            self.delta1, self.delta2,
        ))

    def phi_phi_K_values(self, T, P, composition, max_iter, kij):
        x, kij_array, _ = self._arrays(composition, kij)
        return _cubic_phi_phi_k_numba(
            float(T), float(P), x, int(max_iter), kij_array,
            self.Tc, self.Pc, self.omega, self.a0, self.pure_b,
            self.alpha_mode, self.c1, self.c2, self.c3,
            self.delta1, self.delta2,
        )


if njit is not None:

    @njit(cache=True)
    def _soave_m_numba(omega, is_pr):
        if is_pr:
            return 0.37464 + 1.54226 * omega - 0.26992 * omega * omega
        return 0.480 + 1.574 * omega - 0.176 * omega * omega


    @njit(cache=True)
    def _alpha_and_derivative_numba(T, Tc, omega, mode, c1, c2, c3, is_pr):
        if mode == ALPHA_REDLICH_KWONG:
            alpha = 1.0 / math.sqrt(T)
            return alpha, -0.5 / (T ** 1.5)
        Tr = T / Tc
        if Tr < 1.0e-12:
            Tr = 1.0e-12
        sqrt_Tr = math.sqrt(Tr)
        m_soave = _soave_m_numba(omega, is_pr)

        if mode == ALPHA_RKSMHV2_MATHIAS_COPEMAN:
            theta = 1.0 - sqrt_Tr
            if Tr <= 1.0:
                factor = 1.0 + c1 * theta + c2 * theta**2 + c3 * theta**3
                dfactor_dtheta = c1 + 2.0 * c2 * theta + 3.0 * c3 * theta**2
            else:
                factor = 1.0 + c1 * theta
                dfactor_dtheta = c1
            dtheta_dT = -1.0 / (2.0 * Tc * sqrt_Tr)
            return (
                factor * factor,
                2.0 * factor * dfactor_dtheta * dtheta_dT,
            )

        if mode == ALPHA_TWU:
            exponent_a = c3 * (c2 - 1.0)
            exponent_b = c3 * c2
            alpha = Tr ** exponent_a * math.exp(
                c1 * (1.0 - Tr ** exponent_b)
            )
            dln_dTr = (
                exponent_a / Tr
                - c1 * exponent_b * Tr ** (exponent_b - 1.0)
            )
            return alpha, alpha * dln_dTr / Tc

        if mode == ALPHA_PRSV1 or mode == ALPHA_PRSV2:
            kappa0 = (
                0.378893 + 1.4897153 * omega
                - 0.17131848 * omega * omega
                + 0.0196554 * omega * omega * omega
            )
            g = (1.0 + sqrt_Tr) * (0.7 - Tr)
            dg_dTr = (0.7 - Tr) / (2.0 * sqrt_Tr) - (1.0 + sqrt_Tr)
            if mode == ALPHA_PRSV2:
                h = c1 + c2 * (c3 - Tr) * (1.0 - sqrt_Tr)
                dh_dTr = c2 * (
                    -(1.0 - sqrt_Tr)
                    - (c3 - Tr) / (2.0 * sqrt_Tr)
                )
                kappa = kappa0 + h * g
                dkappa_dTr = dh_dTr * g + h * dg_dTr
            else:
                if Tr <= 0.7:
                    taper = 1.0
                    dtaper = 0.0
                elif Tr >= 1.0:
                    taper = 0.0
                    dtaper = 0.0
                else:
                    s = (Tr - 0.7) / 0.3
                    taper = 1.0 - s * s * (3.0 - 2.0 * s)
                    dtaper = -6.0 * s * (1.0 - s) / 0.3
                kappa = kappa0 + c1 * taper * g
                dkappa_dTr = c1 * (dtaper * g + taper * dg_dTr)
            theta = 1.0 - sqrt_Tr
            factor = 1.0 + kappa * theta
            dfactor_dTr = (
                dkappa_dTr * theta - kappa / (2.0 * sqrt_Tr)
            )
            return factor * factor, 2.0 * factor * dfactor_dTr / Tc

        if mode == ALPHA_MATHIAS_COPEMAN and Tr <= 1.0:
            theta = 1.0 - sqrt_Tr
            factor = 1.0 + c1 * theta + c2 * theta**2 + c3 * theta**3
            dfactor_dtheta = c1 + 2.0 * c2 * theta + 3.0 * c3 * theta**2
            dtheta_dT = -1.0 / (2.0 * Tc * sqrt_Tr)
            return (
                factor * factor,
                2.0 * factor * dfactor_dtheta * dtheta_dT,
            )

        if (mode == ALPHA_BOSTON_MATHIAS or mode == ALPHA_MATHIAS_COPEMAN) and Tr > 1.0:
            m_value = c1 if mode == ALPHA_MATHIAS_COPEMAN else m_soave
            coefficient = 1.0 - 2.0 / (2.0 + m_value)
            power = 1.0 + m_value / 2.0
            exponent = coefficient * (1.0 - Tr**power)
            alpha = math.exp(2.0 * exponent)
            dexponent_dT = (
                -coefficient * power * Tr ** (power - 1.0) / Tc
            )
            return alpha, alpha * 2.0 * dexponent_dT

        factor = 1.0 + m_soave * (1.0 - sqrt_Tr)
        return (
            factor * factor,
            -factor * m_soave / (Tc * sqrt_Tr),
        )


    @njit(cache=True)
    def _pure_parameters_numba(T, Tc, omega, a0, modes, c1, c2, c3, is_pr):
        count = Tc.shape[0]
        pure_a = np.empty(count, dtype=np.float64)
        pure_da = np.empty(count, dtype=np.float64)
        for component in range(count):
            alpha, derivative = _alpha_and_derivative_numba(
                T, Tc[component], omega[component], modes[component],
                c1[component], c2[component], c3[component], is_pr,
            )
            pure_a[component] = a0[component] * alpha
            pure_da[component] = a0[component] * derivative
        return pure_a, pure_da


    @njit(cache=True)
    def _cubic_mixture_numba(
        T, x, kij, dkij, Tc, omega, a0, pure_b,
        modes, c1, c2, c3, delta1,
    ):
        is_pr = delta1 > 2.0
        pure_a, pure_da = _pure_parameters_numba(
            T, Tc, omega, a0, modes, c1, c2, c3, is_pr
        )
        count = x.shape[0]
        a_mix = 0.0
        da_mix = 0.0
        b_mix = 0.0
        for i in range(count):
            b_mix += x[i] * pure_b[i]
            for j in range(count):
                base = math.sqrt(pure_a[i] * pure_a[j])
                a_mix += x[i] * x[j] * base * (1.0 - kij[i, j])
                if pure_a[i] > 0.0 and pure_a[j] > 0.0:
                    dbase = 0.5 * base * (
                        pure_da[i] / pure_a[i]
                        + pure_da[j] / pure_a[j]
                    )
                    da_mix += x[i] * x[j] * (
                        dbase * (1.0 - kij[i, j])
                        - base * dkij[i, j]
                    )
        return a_mix, b_mix, pure_a, pure_da, da_mix


    @njit(cache=True)
    def _cbrt_numba(value):
        return -(abs(value) ** (1.0 / 3.0)) if value < 0.0 else value ** (1.0 / 3.0)


    @njit(cache=True)
    def _monic_roots_numba(a, b, c, minimum):
        p = b - a * a / 3.0
        q = 2.0 * a**3 / 27.0 - a * b / 3.0 + c
        shift = a / 3.0
        discriminant = (0.5 * q) ** 2 + (p / 3.0) ** 3
        candidates = np.zeros(3, dtype=np.float64)
        candidate_count = 0
        if discriminant > 1.0e-14:
            root = (
                _cbrt_numba(-0.5 * q + math.sqrt(discriminant))
                + _cbrt_numba(-0.5 * q - math.sqrt(discriminant))
            )
            candidates[0] = root - shift
            candidate_count = 1
        elif abs(p) < 1.0e-14:
            candidates[0] = -shift
            candidate_count = 1
        else:
            argument = (3.0 * q / (2.0 * p)) * math.sqrt(-3.0 / p)
            argument = max(-1.0, min(1.0, argument))
            theta = math.acos(argument)
            radius = 2.0 * math.sqrt(-p / 3.0)
            for index in range(3):
                candidates[index] = radius * math.cos(
                    (theta + 2.0 * math.pi * index) / 3.0
                ) - shift
            candidate_count = 3
        roots = np.zeros(3, dtype=np.float64)
        count = 0
        for index in range(candidate_count):
            value = candidates[index]
            if value <= minimum:
                continue
            insert = count
            while insert > 0 and roots[insert - 1] > value:
                roots[insert] = roots[insert - 1]
                insert -= 1
            roots[insert] = value
            count += 1
        return roots, count


    @njit(cache=True)
    def _roots_from_mixture_numba(T, P, a_mix, b_mix, delta1, delta2):
        A = a_mix * P / (R_CM3 * R_CM3 * T * T)
        B = b_mix * P / (R_CM3 * T)
        u = delta1 + delta2
        w = delta1 * delta2
        return _monic_roots_numba(
            -(1.0 + B - u * B),
            A + w * B * B - u * B - u * B * B,
            -(A * B + w * B * B + w * B**3),
            B + 1.0e-12,
        )


    @njit(cache=True)
    def _cubic_roots_state_numba(
        T, P, x, kij, Tc, omega, a0, pure_b,
        modes, c1, c2, c3, delta1, delta2,
    ):
        zeros = np.zeros_like(kij)
        a_mix, b_mix, _, _, _ = _cubic_mixture_numba(
            T, x, kij, zeros, Tc, omega, a0, pure_b,
            modes, c1, c2, c3, delta1,
        )
        return _roots_from_mixture_numba(T, P, a_mix, b_mix, delta1, delta2)


    @njit(cache=True)
    def _cubic_fugacity_numba(
        T, P, x, phase_id, kij, Tc, omega, a0, pure_b,
        modes, c1, c2, c3, delta1, delta2,
    ):
        zeros = np.zeros_like(kij)
        a_mix, b_mix, pure_a, _, _ = _cubic_mixture_numba(
            T, x, kij, zeros, Tc, omega, a0, pure_b,
            modes, c1, c2, c3, delta1,
        )
        count_components = x.shape[0]
        phi = np.ones(count_components, dtype=np.float64)
        if a_mix <= 0.0 or b_mix <= 0.0:
            return phi
        roots, count = _roots_from_mixture_numba(
            T, P, a_mix, b_mix, delta1, delta2
        )
        Z = 1.0 if count == 0 else (roots[count - 1] if phase_id == 1 else roots[0])
        A = a_mix * P / (R_CM3 * R_CM3 * T * T)
        B = b_mix * P / (R_CM3 * T)
        delta_diff = delta1 - delta2
        log_arg = (Z + delta1 * B) / max(Z + delta2 * B, 1.0e-16)
        attraction_log = math.log(max(log_arg, 1.0e-16))
        for i in range(count_components):
            sum_aij = 0.0
            for j in range(count_components):
                sum_aij += x[j] * math.sqrt(pure_a[i] * pure_a[j]) * (1.0 - kij[i, j])
            bi_over_b = pure_b[i] / b_mix
            attraction = 2.0 * sum_aij / a_mix - bi_over_b
            ln_phi = bi_over_b * (Z - 1.0) - math.log(max(Z - B, 1.0e-16))
            ln_phi -= A / max(B * delta_diff, 1.0e-16) * attraction * attraction_log
            ln_phi = max(-50.0, min(50.0, ln_phi))
            phi[i] = max(math.exp(ln_phi), 1.0e-12)
        return phi


    @njit(cache=True)
    def _departure_terms_numba(
        T, P, x, phase_id, kij, dkij, Tc, omega, a0, pure_b,
        modes, c1, c2, c3, delta1, delta2,
    ):
        a_mix, b_mix, _, _, da_mix = _cubic_mixture_numba(
            T, x, kij, dkij, Tc, omega, a0, pure_b,
            modes, c1, c2, c3, delta1,
        )
        roots, count = _roots_from_mixture_numba(
            T, P, a_mix, b_mix, delta1, delta2
        )
        Z = 1.0 if count == 0 else (roots[count - 1] if phase_id == 1 else roots[0])
        B = b_mix * P / (R_CM3 * T)
        delta_diff = delta1 - delta2
        log_arg = (Z + delta1 * B) / max(Z + delta2 * B, 1.0e-16)
        attraction_log = math.log(max(log_arg, 1.0e-16))
        return a_mix, b_mix, da_mix, B, Z, delta_diff, attraction_log


    @njit(cache=True)
    def _cubic_departure_enthalpy_numba(
        T, P, x, phase_id, kij, dkij, Tc, omega, a0, pure_b,
        modes, c1, c2, c3, delta1, delta2,
    ):
        a_mix, b_mix, da_mix, _, Z, delta_diff, attraction_log = _departure_terms_numba(
            T, P, x, phase_id, kij, dkij, Tc, omega, a0, pure_b,
            modes, c1, c2, c3, delta1, delta2,
        )
        if b_mix <= 0.0:
            return 0.0
        value = R_CM3 * T * (Z - 1.0)
        value += (T * da_mix - a_mix) / max(
            b_mix * delta_diff, 1.0e-16
        ) * attraction_log
        return 0.1 * value


    @njit(cache=True)
    def _cubic_departure_entropy_numba(
        T, P, x, phase_id, kij, dkij, Tc, omega, a0, pure_b,
        modes, c1, c2, c3, delta1, delta2,
    ):
        _, b_mix, da_mix, B, Z, delta_diff, attraction_log = _departure_terms_numba(
            T, P, x, phase_id, kij, dkij, Tc, omega, a0, pure_b,
            modes, c1, c2, c3, delta1, delta2,
        )
        if b_mix <= 0.0:
            return 0.0
        value = R_CM3 * math.log(max(Z - B, 1.0e-16))
        value += da_mix / max(b_mix * delta_diff, 1.0e-16) * attraction_log
        return 0.1 * value


    @njit(cache=True)
    def _normalize_numba(values):
        total = 0.0
        for value in values:
            if value > 0.0:
                total += value
        result = np.empty(values.shape[0], dtype=np.float64)
        if total <= 0.0:
            for index in range(values.shape[0]):
                result[index] = 1.0 / values.shape[0]
        else:
            for index in range(values.shape[0]):
                result[index] = max(values[index], 0.0) / total
        return result


    @njit(cache=True)
    def _cubic_phi_phi_k_numba(
        T, P, x, max_iter, kij, Tc, Pc, omega, a0, pure_b,
        modes, c1, c2, c3, delta1, delta2,
    ):
        count_components = x.shape[0]
        K = np.empty(count_components, dtype=np.float64)
        for component in range(count_components):
            exponent = 5.373 * (1.0 + omega[component]) * (1.0 - Tc[component] / T)
            K[component] = max(
                1.0e-8,
                min(1.0e8, Pc[component] / max(P, 1.0e-12) * math.exp(exponent)),
            )
        roots, root_count = _cubic_roots_state_numba(
            T, P, x, kij, Tc, omega, a0, pure_b,
            modes, c1, c2, c3, delta1, delta2,
        )
        single_root = root_count < 2
        initial_K = K.copy()
        phi_l = _cubic_fugacity_numba(
            T, P, x, 0, kij, Tc, omega, a0, pure_b,
            modes, c1, c2, c3, delta1, delta2,
        )
        work = np.empty(count_components, dtype=np.float64)
        for component in range(count_components):
            work[component] = x[component] * K[component]
        y = _normalize_numba(work)
        for _ in range(max_iter):
            phi_v = _cubic_fugacity_numba(
                T, P, y, 1, kij, Tc, omega, a0, pure_b,
                modes, c1, c2, c3, delta1, delta2,
            )
            K_new = np.empty(count_components, dtype=np.float64)
            for component in range(count_components):
                value = phi_l[component] / max(phi_v[component], 1.0e-12)
                K_new[component] = max(1.0e-8, min(1.0e8, value))
                work[component] = x[component] * K_new[component]
            y_new = _normalize_numba(work)
            maximum_change = 0.0
            for component in range(count_components):
                maximum_change = max(
                    maximum_change,
                    abs(y_new[component] - y[component]),
                )
            if maximum_change < 1.0e-9:
                trivial = single_root
                for component in range(count_components):
                    if x[component] > 0.0 and abs(K_new[component] - 1.0) >= 1e-6:
                        trivial = False
                if trivial:
                    return initial_K
                return K_new
            for component in range(count_components):
                K[component] = 0.5 * K[component] + 0.5 * K_new[component]
                y[component] = y_new[component]
        return K

else:

    def _cubic_mixture_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError('Numba is not available')

    def _cubic_roots_state_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError('Numba is not available')

    def _cubic_fugacity_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError('Numba is not available')

    def _cubic_departure_enthalpy_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError('Numba is not available')

    def _cubic_departure_entropy_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError('Numba is not available')

    def _cubic_phi_phi_k_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError('Numba is not available')
