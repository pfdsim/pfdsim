"""Compiled fixed-component MHV2 kernels for cubic EOS models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .compiled_cubic_eos import (
        _normalize_numba,
        _pure_parameters_numba,
        _roots_from_mixture_numba,
        compiled_alpha_parameter_arrays,
    )
    from .compiled_unifac import _activity_coefficients_numba
    from .physical_constants import R_BAR_CM3_MOL_K
else:
    from compiled_cubic_eos import (
        _normalize_numba,
        _pure_parameters_numba,
        _roots_from_mixture_numba,
        compiled_alpha_parameter_arrays,
    )
    from compiled_unifac import _activity_coefficients_numba
    from physical_constants import R_BAR_CM3_MOL_K

try:
    from numba import njit, typeof
except Exception:  # pragma: no cover - optional dependency fallback
    njit = None
    typeof = None


R_CM3 = R_BAR_CM3_MOL_K


@dataclass
class CompiledMHV2EOSBackend:
    """Array-backed cubic EOS using a compiled MHV2 excess-Gibbs rule."""

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
    nu: np.ndarray
    r: np.ndarray
    q: np.ndarray
    subgroup_q: np.ndarray
    interactions: np.ndarray
    interactions_b: np.ndarray
    interactions_c: np.ndarray
    variant_id: int
    q1: float
    q2: float
    delta1: float
    delta2: float
    compilation_complete: bool = False

    @classmethod
    def from_eos(cls, eos) -> "CompiledMHV2EOSBackend | None":
        if njit is None or eos._ge_provider.backend is None:
            return None
        params, modes, c1, c2, c3 = compiled_alpha_parameter_arrays(
            eos,
            mhv2=True,
        )
        activity = eos._ge_provider.backend
        delta1, delta2 = eos._delta_roots()
        backend = cls(
            components=tuple(eos.components),
            Tc=np.asarray([value.Tc for value in params], dtype=np.float64),
            Pc=np.asarray([value.Pc for value in params], dtype=np.float64),
            omega=np.asarray([value.omega for value in params], dtype=np.float64),
            a0=np.asarray([value.a0 for value in params], dtype=np.float64),
            pure_b=np.asarray([value.b for value in params], dtype=np.float64),
            alpha_mode=modes,
            c1=c1,
            c2=c2,
            c3=c3,
            nu=activity.nu,
            r=activity.r,
            q=activity.q,
            subgroup_q=activity.subgroup_q,
            interactions=activity.interactions,
            interactions_b=activity.interactions_b,
            interactions_c=activity.interactions_c,
            variant_id=int(activity.variant_id),
            q1=float(eos._ge_mixing.q1),
            q2=float(eos._ge_mixing.q2),
            delta1=float(delta1),
            delta2=float(delta2),
        )
        backend.compile_kernels()
        return backend

    def _common(self):
        return (
            self.Tc, self.Pc, self.omega, self.a0, self.pure_b,
            self.alpha_mode, self.c1, self.c2, self.c3,
            self.nu, self.r, self.q, self.subgroup_q,
            self.interactions, self.interactions_b, self.interactions_c,
            self.variant_id, self.q1, self.q2, self.delta1, self.delta2,
        )

    def compile_kernels(self) -> None:
        if self.compilation_complete or typeof is None:
            return
        composition = np.full(
            len(self.components),
            1.0 / len(self.components),
            dtype=np.float64,
        )
        common = self._common()

        def compile_for(dispatcher, *args):
            dispatcher.compile(tuple(typeof(argument) for argument in args))

        compile_for(_mhv2_mechanical_numba, 300.0, composition, common)
        compile_for(_mhv2_caloric_numba, 300.0, composition, common)
        compile_for(_mhv2_roots_numba, 300.0, 1.0, composition, common)
        compile_for(_mhv2_fugacity_numba, 300.0, 1.0, composition, 0, common)
        compile_for(_mhv2_departure_enthalpy_numba, 300.0, 1.0, composition, 0, common)
        compile_for(_mhv2_departure_entropy_numba, 300.0, 1.0, composition, 0, common)
        compile_for(
            _mhv2_phi_phi_numba,
            300.0, 1.0, composition, 80, common,
        )
        self.compilation_complete = True

    @staticmethod
    def _composition(values):
        array = np.asarray(values, dtype=np.float64)
        total = float(array.sum())
        if total <= 0.0:
            return np.full_like(array, 1.0 / len(array))
        return np.maximum(array, 0.0) / total

    def mixture_parameters(self, T, composition, _kij=(), dkij=None):
        x = self._composition(composition)
        if dkij is None:
            a_mix, b_mix, pure_a, pure_da, _ = _mhv2_mechanical_numba(
                float(T), x, self._common()
            )
            return a_mix, b_mix, pure_a, pure_da, math.nan
        a_mix, b_mix, pure_a, pure_da, da_mix, _ = _mhv2_caloric_numba(
            float(T), x, self._common()
        )
        return a_mix, b_mix, pure_a, pure_da, da_mix

    def compressibility_roots(self, T, P, composition, _kij=()):
        roots, count = _mhv2_roots_numba(
            float(T), float(P), self._composition(composition), self._common()
        )
        return roots[:count].copy()

    def fugacity_coefficients(self, T, P, composition, phase, _kij=()):
        phase_id = 1 if str(phase).strip().lower() == 'vapor' else 0
        values, count = _mhv2_fugacity_numba(
            float(T), float(P), self._composition(composition), phase_id,
            self._common(),
        )
        if count == 0:
            raise ValueError('MHV2 cubic state has no physical root')
        return values

    def departure_enthalpy(self, T, P, composition, phase, _kij=(), _dkij=()):
        phase_id = 1 if str(phase).strip().lower() == 'vapor' else 0
        return float(_mhv2_departure_enthalpy_numba(
            float(T), float(P), self._composition(composition), phase_id,
            self._common(),
        ))

    def departure_entropy(self, T, P, composition, phase, _kij=(), _dkij=()):
        phase_id = 1 if str(phase).strip().lower() == 'vapor' else 0
        return float(_mhv2_departure_entropy_numba(
            float(T), float(P), self._composition(composition), phase_id,
            self._common(),
        ))

    def phi_phi_K_values(self, T, P, composition, max_iter, _kij=()):
        return _mhv2_phi_phi_numba(
            float(T), float(P), self._composition(composition), int(max_iter),
            self._common(),
        )


if njit is not None:

    @njit(cache=True)
    def _mhv2_ln_gamma_numba(
        T, x, nu, r, q, subgroup_q, interactions,
        interactions_b, interactions_c, variant_id,
    ):
        gamma = _activity_coefficients_numba(
            nu, r, q, subgroup_q, interactions,
            interactions_b, interactions_c, variant_id, x, T,
        )
        result = np.empty(gamma.shape[0], dtype=np.float64)
        for index in range(gamma.shape[0]):
            result[index] = math.log(gamma[index])
        return result


    @njit(cache=True)
    def _mhv2_mix_from_activity_numba(
        T, x, pure_a, pure_da, pure_b, ln_gamma, dln_gamma,
        q1, q2,
    ):
        count = x.shape[0]
        b_mix = 0.0
        for index in range(count):
            b_mix += x[index] * pure_b[index]
        pure_D = np.empty(count, dtype=np.float64)
        pure_dD = np.empty(count, dtype=np.float64)
        pure_q = np.empty(count, dtype=np.float64)
        size = np.empty(count, dtype=np.float64)
        rhs = 0.0
        for index in range(count):
            D_i = pure_a[index] / (pure_b[index] * R_CM3 * T)
            pure_D[index] = D_i
            pure_dD[index] = (
                pure_da[index] / (pure_b[index] * R_CM3 * T) - D_i / T
            )
            q_i = q1 * D_i + q2 * D_i * D_i
            pure_q[index] = q_i
            size_i = math.log(b_mix / pure_b[index])
            size[index] = size_i
            rhs += x[index] * (ln_gamma[index] + q_i + size_i)
        discriminant = q1 * q1 + 4.0 * q2 * rhs
        if not math.isfinite(discriminant) or discriminant <= 0.0:
            raise ValueError('MHV2 mixing equation has no physical branch')
        D_mix = 2.0 * rhs / (q1 - math.sqrt(discriminant))
        slope = q1 + 2.0 * q2 * D_mix
        dD_dT = 0.0
        sigma = np.empty(count, dtype=np.float64)
        for index in range(count):
            dD_dT += x[index] * (
                dln_gamma[index]
                + (q1 + 2.0 * q2 * pure_D[index]) * pure_dD[index]
            )
            sigma[index] = (
                pure_q[index] + q2 * D_mix * D_mix
                + ln_gamma[index] + size[index]
                + pure_b[index] / b_mix - 1.0
            ) / slope
        dD_dT /= slope
        a_mix = D_mix * b_mix * R_CM3 * T
        da_mix = b_mix * R_CM3 * (D_mix + T * dD_dT)
        return a_mix, b_mix, da_mix, sigma


    @njit(cache=True)
    def _mhv2_mechanical_numba(T, x, common):
        (
            Tc, _Pc, omega, a0, pure_b, modes, c1, c2, c3,
            nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, q1, q2, _delta1, _delta2,
        ) = common
        pure_a, pure_da = _pure_parameters_numba(
            T, Tc, omega, a0, modes, c1, c2, c3, False
        )
        ln_gamma = _mhv2_ln_gamma_numba(
            T, x, nu, r, q, subgroup_q, interactions,
            interactions_b, interactions_c, variant_id,
        )
        zeros = np.zeros(x.shape[0], dtype=np.float64)
        a_mix, b_mix, _, sigma = _mhv2_mix_from_activity_numba(
            T, x, pure_a, pure_da, pure_b, ln_gamma, zeros, q1, q2,
        )
        return a_mix, b_mix, pure_a, pure_da, sigma


    @njit(cache=True)
    def _mhv2_caloric_numba(T, x, common):
        (
            Tc, _Pc, omega, a0, pure_b, modes, c1, c2, c3,
            nu, r, q, subgroup_q, interactions, interactions_b,
            interactions_c, variant_id, q1, q2, _delta1, _delta2,
        ) = common
        pure_a, pure_da = _pure_parameters_numba(
            T, Tc, omega, a0, modes, c1, c2, c3, False
        )
        ln_gamma = _mhv2_ln_gamma_numba(
            T, x, nu, r, q, subgroup_q, interactions,
            interactions_b, interactions_c, variant_id,
        )
        step = T * 1.0e-4
        mm = _mhv2_ln_gamma_numba(
            T - 2.0 * step, x, nu, r, q, subgroup_q, interactions,
            interactions_b, interactions_c, variant_id,
        )
        m = _mhv2_ln_gamma_numba(
            T - step, x, nu, r, q, subgroup_q, interactions,
            interactions_b, interactions_c, variant_id,
        )
        p = _mhv2_ln_gamma_numba(
            T + step, x, nu, r, q, subgroup_q, interactions,
            interactions_b, interactions_c, variant_id,
        )
        pp = _mhv2_ln_gamma_numba(
            T + 2.0 * step, x, nu, r, q, subgroup_q, interactions,
            interactions_b, interactions_c, variant_id,
        )
        derivative = (mm - 8.0 * m + 8.0 * p - pp) / (12.0 * step)
        a_mix, b_mix, da_mix, sigma = _mhv2_mix_from_activity_numba(
            T, x, pure_a, pure_da, pure_b,
            ln_gamma, derivative, q1, q2,
        )
        return a_mix, b_mix, pure_a, pure_da, da_mix, sigma


    @njit(cache=True)
    def _mhv2_roots_numba(T, P, x, common):
        delta1, delta2 = common[19], common[20]
        a_mix, b_mix, _, _, _ = _mhv2_mechanical_numba(
            T, x, common
        )
        return _roots_from_mixture_numba(
            T, P, a_mix, b_mix, delta1, delta2
        )


    @njit(cache=True)
    def _mhv2_fugacity_numba(T, P, x, phase_id, common):
        pure_b = common[4]
        delta1, delta2 = common[19], common[20]
        a_mix, b_mix, _, _, sigma = _mhv2_mechanical_numba(
            T, x, common
        )
        roots, count = _roots_from_mixture_numba(
            T, P, a_mix, b_mix, delta1, delta2
        )
        phi = np.empty(x.shape[0], dtype=np.float64)
        if count == 0:
            for index in range(x.shape[0]):
                phi[index] = math.nan
            return phi, count
        Z = roots[count - 1] if phase_id == 1 else roots[0]
        B = b_mix * P / (R_CM3 * T)
        attraction_log = math.log(
            (Z + delta1 * B) / max(Z + delta2 * B, 1.0e-16)
        )
        delta_diff = delta1 - delta2
        for index in range(x.shape[0]):
            ln_phi = (
                pure_b[index] / b_mix * (Z - 1.0)
                - math.log(Z - B)
                - sigma[index] * attraction_log / delta_diff
            )
            phi[index] = math.exp(ln_phi)
        return phi, count


    @njit(cache=True)
    def _mhv2_departure_terms_numba(T, P, x, phase_id, common):
        delta1, delta2 = common[19], common[20]
        a_mix, b_mix, _, _, da_mix, _ = _mhv2_caloric_numba(
            T, x, common
        )
        roots, count = _roots_from_mixture_numba(
            T, P, a_mix, b_mix, delta1, delta2
        )
        if count == 0:
            raise ValueError('MHV2 cubic state has no physical root')
        Z = roots[count - 1] if phase_id == 1 else roots[0]
        B = b_mix * P / (R_CM3 * T)
        attraction_log = math.log(
            (Z + delta1 * B) / max(Z + delta2 * B, 1.0e-16)
        )
        return a_mix, b_mix, da_mix, B, Z, delta1 - delta2, attraction_log


    @njit(cache=True)
    def _mhv2_departure_enthalpy_numba(T, P, x, phase_id, common):
        a_mix, b_mix, da_mix, _, Z, delta_diff, attraction_log = (
            _mhv2_departure_terms_numba(T, P, x, phase_id, common)
        )
        value = R_CM3 * T * (Z - 1.0)
        value += (
            (T * da_mix - a_mix) / max(b_mix * delta_diff, 1.0e-16)
            * attraction_log
        )
        return 0.1 * value


    @njit(cache=True)
    def _mhv2_departure_entropy_numba(T, P, x, phase_id, common):
        _, b_mix, da_mix, B, Z, delta_diff, attraction_log = (
            _mhv2_departure_terms_numba(T, P, x, phase_id, common)
        )
        value = R_CM3 * math.log(max(Z - B, 1.0e-16))
        value += da_mix / max(b_mix * delta_diff, 1.0e-16) * attraction_log
        return 0.1 * value


    @njit(cache=True)
    def _mhv2_phi_phi_numba(T, P, x, max_iter, common):
        Tc = common[0]
        Pc = common[1]
        omega = common[2]
        count = x.shape[0]
        K = np.empty(count, dtype=np.float64)
        work = np.empty(count, dtype=np.float64)
        for index in range(count):
            exponent = 5.373 * (1.0 + omega[index]) * (1.0 - Tc[index] / T)
            K[index] = max(
                1.0e-8,
                min(1.0e8, Pc[index] / max(P, 1.0e-12) * math.exp(exponent)),
            )
            work[index] = x[index] * K[index]
        initial_K = K.copy()
        roots, root_count = _mhv2_roots_numba(T, P, x, common)
        single_root = root_count < 2
        phi_l, liquid_count = _mhv2_fugacity_numba(
            T, P, x, 0, common
        )
        if liquid_count == 0:
            return phi_l
        y = _normalize_numba(work)
        for _ in range(max_iter):
            phi_v, vapor_count = _mhv2_fugacity_numba(
                T, P, y, 1, common
            )
            if vapor_count == 0:
                return phi_v
            K_new = np.empty(count, dtype=np.float64)
            for index in range(count):
                K_new[index] = max(
                    1.0e-8,
                    min(1.0e8, phi_l[index] / max(phi_v[index], 1.0e-12)),
                )
                work[index] = x[index] * K_new[index]
            y_new = _normalize_numba(work)
            maximum_change = 0.0
            for index in range(count):
                maximum_change = max(
                    maximum_change,
                    abs(y_new[index] - y[index]),
                )
            if maximum_change < 1.0e-9:
                trivial = single_root
                for index in range(count):
                    if x[index] > 0.0 and abs(K_new[index] - 1.0) >= 1.0e-6:
                        trivial = False
                return initial_K if trivial else K_new
            for index in range(count):
                K[index] = 0.5 * K[index] + 0.5 * K_new[index]
                y[index] = y_new[index]
        return K

else:

    def _unavailable(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError('Numba is not available')

    _mhv2_mechanical_numba = _unavailable
    _mhv2_caloric_numba = _unavailable
    _mhv2_roots_numba = _unavailable
    _mhv2_fugacity_numba = _unavailable
    _mhv2_departure_enthalpy_numba = _unavailable
    _mhv2_departure_entropy_numba = _unavailable
    _mhv2_phi_phi_numba = _unavailable
