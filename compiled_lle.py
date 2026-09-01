"""
Compiled liquid-liquid equilibrium splitter for array-backed activity models.

This module mirrors ActivityCoefficientThermodynamics.liquid_liquid_equilibrium
for the multicomponent fixed-point split, but keeps the phase split and UNIFAC
activity calls inside Numba arrays. It is intentionally an optional fast path:
callers should fall back to the readable Python implementation whenever the
compiled backend is unavailable or the component set does not match exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from numba import njit, typeof
except Exception:  # pragma: no cover - exercised only without optional numba
    njit = None
    typeof = None

try:
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .compiled_unifac import _activity_coefficients_numba
    else:
        from compiled_unifac import _activity_coefficients_numba
except Exception:  # pragma: no cover - exercised only without optional backend
    _activity_coefficients_numba = None

try:
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .compiled_activity import (
                _nrtl_activity_coefficients_numba,
                _uniquac_activity_coefficients_numba,
            )
    else:
        from compiled_activity import (
                _nrtl_activity_coefficients_numba,
                _uniquac_activity_coefficients_numba,
            )
except Exception:  # pragma: no cover - exercised only without optional backend
    _nrtl_activity_coefficients_numba = None
    _uniquac_activity_coefficients_numba = None


@dataclass
class CompiledLLEBackend:
    """Compiled LLE splitter for one fixed UNIFAC component set."""

    components: list[str]
    nu: np.ndarray
    r: np.ndarray
    q: np.ndarray
    subgroup_q: np.ndarray
    interactions: np.ndarray
    interactions_b: np.ndarray
    interactions_c: np.ndarray
    variant_id: int
    compilation_complete: bool = False
    _component_index: dict[str, int] = field(init=False, repr=False)
    _ordered_arrays_cache: dict[
        tuple[str, ...],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._component_index = {
            comp: index
            for index, comp in enumerate(self.components)
        }

    @classmethod
    def from_unifac_backend(cls, backend) -> "CompiledLLEBackend | None":
        if njit is None or _activity_coefficients_numba is None or backend is None:
            return None
        return cls(
            components=list(backend.components),
            nu=np.asarray(backend.nu, dtype=np.float64),
            r=np.asarray(backend.r, dtype=np.float64),
            q=np.asarray(backend.q, dtype=np.float64),
            subgroup_q=np.asarray(backend.subgroup_q, dtype=np.float64),
            interactions=np.asarray(backend.interactions, dtype=np.float64),
            interactions_b=np.asarray(backend.interactions_b, dtype=np.float64),
            interactions_c=np.asarray(backend.interactions_c, dtype=np.float64),
            variant_id=int(backend.variant_id),
        )

    def compile_kernels(self) -> None:
        if self.compilation_complete or typeof is None:
            return
        composition = np.zeros(len(self.components), dtype=np.float64)
        args = (
            self.nu, self.r, self.q, self.subgroup_q, self.interactions,
            self.interactions_b, self.interactions_c, self.variant_id,
            composition, 298.15, 100, 1.0e-6,
        )
        _lle_split_numba.compile(tuple(typeof(argument) for argument in args))
        self.compilation_complete = True

    def _ordered_arrays(self, order: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        cached = self._ordered_arrays_cache.get(order)
        if cached is not None:
            return cached
        if set(order) != set(self.components):
            return None
        indices = []
        for comp in order:
            index = self._component_index.get(comp)
            if index is None:
                return None
            indices.append(index)
        index_array = np.asarray(indices, dtype=np.int64)
        arrays = (
            self.nu[index_array, :].copy(),
            self.r[index_array].copy(),
            self.q[index_array].copy(),
        )
        self._ordered_arrays_cache[order] = arrays
        return arrays

    def split(
        self,
        composition: dict[str, float],
        T: float,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> tuple[bool, dict[str, float], dict[str, float], float] | None:
        order = tuple(composition.keys())
        ordered_arrays = self._ordered_arrays(order)
        if ordered_arrays is None:
            return None

        nu, r, q = ordered_arrays
        z = np.asarray(
            [max(float(composition.get(comp, 0.0)), 0.0) for comp in order],
            dtype=np.float64,
        )
        has_lle, x1, x2, beta = _lle_split_numba(
            nu,
            r,
            q,
            self.subgroup_q,
            self.interactions,
            self.interactions_b,
            self.interactions_c,
            self.variant_id,
            z,
            float(T),
            int(max_iter),
            float(tol),
        )
        x1_dict = {comp: float(x1[index]) for index, comp in enumerate(order)}
        x2_dict = {comp: float(x2[index]) for index, comp in enumerate(order)}
        return bool(has_lle), x1_dict, x2_dict, float(beta)


@dataclass
class CompiledNRTLLLEBackend:
    """Compiled LLE splitter for one fixed NRTL component set."""

    components: list[str]
    tau_mode: np.ndarray
    tau_c: np.ndarray
    tau_d: np.ndarray
    tau_e: np.ndarray
    tau_f: np.ndarray
    tau_g: np.ndarray
    tau_tref: np.ndarray
    tau_energy: np.ndarray
    alpha: np.ndarray
    interaction_temperature_caps: np.ndarray
    compilation_complete: bool = False
    _component_index: dict[str, int] = field(init=False, repr=False)
    _ordered_arrays_cache: dict[
        tuple[str, ...],
        tuple[np.ndarray, ...],
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._component_index = {
            comp: index
            for index, comp in enumerate(self.components)
        }

    @classmethod
    def from_activity_backend(cls, backend) -> "CompiledNRTLLLEBackend | None":
        if njit is None or _nrtl_activity_coefficients_numba is None or backend is None:
            return None
        return cls(
            components=list(backend.components),
            tau_mode=np.asarray(backend.tau_mode, dtype=np.int64),
            tau_c=np.asarray(backend.tau_c, dtype=np.float64),
            tau_d=np.asarray(backend.tau_d, dtype=np.float64),
            tau_e=np.asarray(backend.tau_e, dtype=np.float64),
            tau_f=np.asarray(backend.tau_f, dtype=np.float64),
            tau_g=np.asarray(backend.tau_g, dtype=np.float64),
            tau_tref=np.asarray(backend.tau_tref, dtype=np.float64),
            tau_energy=np.asarray(backend.tau_energy, dtype=np.float64),
            alpha=np.asarray(backend.alpha, dtype=np.float64),
            interaction_temperature_caps=np.asarray(
                backend.interaction_temperature_caps,
                dtype=np.float64,
            ),
        )

    def compile_kernels(self) -> None:
        if self.compilation_complete or typeof is None:
            return
        composition = np.zeros(len(self.components), dtype=np.float64)
        args = (
            composition, 298.15, self.tau_mode, self.tau_c, self.tau_d,
            self.tau_e, self.tau_f, self.tau_g, self.tau_tref, self.tau_energy, self.alpha,
            self.interaction_temperature_caps,
            100, 1.0e-6,
        )
        _lle_split_nrtl_numba.compile(
            tuple(typeof(argument) for argument in args)
        )
        self.compilation_complete = True

    def _ordered_arrays(
        self,
        order: tuple[str, ...],
    ) -> tuple[np.ndarray, ...] | None:
        cached = self._ordered_arrays_cache.get(order)
        if cached is not None:
            return cached
        if set(order) != set(self.components):
            return None
        indices = []
        for comp in order:
            index = self._component_index.get(comp)
            if index is None:
                return None
            indices.append(index)
        index_array = np.asarray(indices, dtype=np.int64)
        arrays = (
            self.tau_mode[index_array, :][:, index_array].copy(),
            self.tau_c[index_array, :][:, index_array].copy(),
            self.tau_d[index_array, :][:, index_array].copy(),
            self.tau_e[index_array, :][:, index_array].copy(),
            self.tau_f[index_array, :][:, index_array].copy(),
            self.tau_g[index_array, :][:, index_array].copy(),
            self.tau_tref[index_array, :][:, index_array].copy(),
            self.tau_energy[index_array, :][:, index_array].copy(),
            self.alpha[index_array, :][:, index_array].copy(),
            self.interaction_temperature_caps[index_array].copy(),
        )
        self._ordered_arrays_cache[order] = arrays
        return arrays

    def split(
        self,
        composition: dict[str, float],
        T: float,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> tuple[bool, dict[str, float], dict[str, float], float] | None:
        order = tuple(composition.keys())
        ordered_arrays = self._ordered_arrays(order)
        if ordered_arrays is None:
            return None
        tau_mode, tau_c, tau_d, tau_e, tau_f, tau_g, tau_tref, tau_energy, alpha, interaction_temperature_caps = ordered_arrays
        z = np.asarray(
            [max(float(composition.get(comp, 0.0)), 0.0) for comp in order],
            dtype=np.float64,
        )
        has_lle, x1, x2, beta = _lle_split_nrtl_numba(
            z,
            float(T),
            tau_mode,
            tau_c,
            tau_d,
            tau_e,
            tau_f,
            tau_g,
            tau_tref,
            tau_energy,
            alpha,
            interaction_temperature_caps,
            int(max_iter),
            float(tol),
        )
        x1_dict = {comp: float(x1[index]) for index, comp in enumerate(order)}
        x2_dict = {comp: float(x2[index]) for index, comp in enumerate(order)}
        return bool(has_lle), x1_dict, x2_dict, float(beta)


@dataclass
class CompiledUNIQUACLLEBackend:
    """Compiled LLE splitter for one fixed UNIQUAC component set."""

    components: list[str]
    r: np.ndarray
    q: np.ndarray
    q_residual: np.ndarray
    tau_mode: np.ndarray
    tau_a: np.ndarray
    tau_b: np.ndarray
    tau_c: np.ndarray
    tau_d: np.ndarray
    tau_e: np.ndarray
    tau_tref: np.ndarray
    interaction_temperature_caps: np.ndarray
    compilation_complete: bool = False
    _component_index: dict[str, int] = field(init=False, repr=False)
    _ordered_arrays_cache: dict[
        tuple[str, ...],
        tuple[np.ndarray, ...],
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._component_index = {
            comp: index
            for index, comp in enumerate(self.components)
        }

    @classmethod
    def from_activity_backend(cls, backend) -> "CompiledUNIQUACLLEBackend | None":
        if njit is None or _uniquac_activity_coefficients_numba is None or backend is None:
            return None
        return cls(
            components=list(backend.components),
            r=np.asarray(backend.r, dtype=np.float64),
            q=np.asarray(backend.q, dtype=np.float64),
            q_residual=np.asarray(backend.q_residual, dtype=np.float64),
            tau_mode=np.asarray(backend.tau_mode, dtype=np.int64),
            tau_a=np.asarray(backend.tau_a, dtype=np.float64),
            tau_b=np.asarray(backend.tau_b, dtype=np.float64),
            tau_c=np.asarray(backend.tau_c, dtype=np.float64),
            tau_d=np.asarray(backend.tau_d, dtype=np.float64),
            tau_e=np.asarray(backend.tau_e, dtype=np.float64),
            tau_tref=np.asarray(backend.tau_tref, dtype=np.float64),
            interaction_temperature_caps=np.asarray(
                backend.interaction_temperature_caps,
                dtype=np.float64,
            ),
        )

    def compile_kernels(self) -> None:
        if self.compilation_complete or typeof is None:
            return
        composition = np.zeros(len(self.components), dtype=np.float64)
        args = (
            composition, 298.15, self.r, self.q, self.q_residual,
            self.tau_mode, self.tau_a, self.tau_b, self.tau_c,
            self.tau_d, self.tau_e, self.tau_tref,
            self.interaction_temperature_caps,
            100, 1.0e-6,
        )
        _lle_split_uniquac_numba.compile(
            tuple(typeof(argument) for argument in args)
        )
        self.compilation_complete = True

    def _ordered_arrays(
        self,
        order: tuple[str, ...],
    ) -> tuple[np.ndarray, ...] | None:
        cached = self._ordered_arrays_cache.get(order)
        if cached is not None:
            return cached
        if set(order) != set(self.components):
            return None
        indices = []
        for comp in order:
            index = self._component_index.get(comp)
            if index is None:
                return None
            indices.append(index)
        index_array = np.asarray(indices, dtype=np.int64)
        arrays = (
            self.r[index_array].copy(),
            self.q[index_array].copy(),
            self.q_residual[index_array].copy(),
            self.tau_mode[index_array, :][:, index_array].copy(),
            self.tau_a[index_array, :][:, index_array].copy(),
            self.tau_b[index_array, :][:, index_array].copy(),
            self.tau_c[index_array, :][:, index_array].copy(),
            self.tau_d[index_array, :][:, index_array].copy(),
            self.tau_e[index_array, :][:, index_array].copy(),
            self.tau_tref[index_array, :][:, index_array].copy(),
            self.interaction_temperature_caps[index_array].copy(),
        )
        self._ordered_arrays_cache[order] = arrays
        return arrays

    def split(
        self,
        composition: dict[str, float],
        T: float,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> tuple[bool, dict[str, float], dict[str, float], float] | None:
        order = tuple(composition.keys())
        ordered_arrays = self._ordered_arrays(order)
        if ordered_arrays is None:
            return None
        r, q, q_residual, tau_mode, tau_a, tau_b, tau_c, tau_d, tau_e, tau_tref, interaction_temperature_caps = ordered_arrays
        z = np.asarray(
            [max(float(composition.get(comp, 0.0)), 0.0) for comp in order],
            dtype=np.float64,
        )
        has_lle, x1, x2, beta = _lle_split_uniquac_numba(
            z,
            float(T),
            r,
            q,
            q_residual,
            tau_mode,
            tau_a,
            tau_b,
            tau_c,
            tau_d,
            tau_e,
            tau_tref,
            interaction_temperature_caps,
            int(max_iter),
            float(tol),
        )
        x1_dict = {comp: float(x1[index]) for index, comp in enumerate(order)}
        x2_dict = {comp: float(x2[index]) for index, comp in enumerate(order)}
        return bool(has_lle), x1_dict, x2_dict, float(beta)


if njit is not None:

    @njit(cache=True)
    def _normalize(values):
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
            equal = 1.0 / n
            for i in range(n):
                out[i] = equal
        else:
            for i in range(n):
                out[i] /= total
        return out

else:

    def _normalize(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Compiled LLE backend is unavailable")


if njit is not None and _activity_coefficients_numba is not None:

    @njit(cache=True)
    def _lle_split_numba(
        nu, r, q, subgroup_q, interactions, interactions_b, interactions_c,
        variant_id, z_input, T, max_iter, tol,
    ):
        z = _normalize(z_input)
        n = z.shape[0]
        x1 = np.empty(n, dtype=np.float64)
        x2 = np.empty(n, dtype=np.float64)
        if n == 2:
            x1[0] = 1.0 - 1e-5
            x1[1] = 1e-5
            x2[0] = 1e-5
            x2[1] = 1.0 - 1e-5
        else:
            for i in range(n):
                z_i = z[i]
                if i == 0:
                    x1[i] = min(0.95, z_i * 1.5)
                    x2[i] = max(0.05, z_i * 0.5)
                else:
                    x1[i] = max(0.05, z_i * 0.5)
                    x2[i] = min(0.95, z_i * 1.5)
        x1 = _normalize(x1)
        x2 = _normalize(x2)
        beta = 0.5
        K = np.empty(n, dtype=np.float64)
        max_diff = 0.0

        for _iteration in range(max_iter):
            gamma1 = _activity_coefficients_numba(
                nu, r, q, subgroup_q, interactions, interactions_b, interactions_c,
                variant_id, x1, T
            )
            gamma2 = _activity_coefficients_numba(
                nu, r, q, subgroup_q, interactions, interactions_b, interactions_c,
                variant_id, x2, T
            )

            max_diff = 0.0
            for i in range(n):
                gamma2_i = gamma2[i]
                if gamma2_i < 1e-10:
                    gamma2_i = 1e-10
                K[i] = gamma1[i] / gamma2_i
                act1 = x1[i] * gamma1[i]
                act2 = x2[i] * gamma2[i]
                scale = max(max(act1, act2), 1e-10)
                diff = abs(act1 - act2) / scale
                if diff > max_diff:
                    max_diff = diff

            if max_diff < tol:
                phase_diff = 0.0
                for i in range(n):
                    phase_diff += abs(x1[i] - x2[i])
                phase_diff /= n
                if phase_diff < 0.01:
                    return False, z, z, 0.0
                if beta <= 1e-10 or beta >= 1.0 - 1e-10:
                    return False, z, z, 0.0
                return True, x1, x2, beta

            for _inner in range(20):
                f = 0.0
                df = 0.0
                for i in range(n):
                    km1 = K[i] - 1.0
                    denom = 1.0 + beta * km1
                    f += z[i] * km1 / denom
                    df -= z[i] * km1 * km1 / (denom * denom)
                if abs(df) < 1e-15:
                    break
                beta = beta - f / df
                if beta < 1e-12:
                    beta = 1e-12
                elif beta > 1.0 - 1e-12:
                    beta = 1.0 - 1e-12
                if abs(f) < 1e-10:
                    break

            for i in range(n):
                denom = 1.0 + beta * (K[i] - 1.0)
                if denom < 1e-10:
                    denom = 1e-10
                x1[i] = z[i] / denom
                x2[i] = K[i] * x1[i]
            x1 = _normalize(x1)
            x2 = _normalize(x2)

        phase_diff = 0.0
        for i in range(n):
            phase_diff += abs(x1[i] - x2[i])
        phase_diff /= n
        if phase_diff < 0.01:
            return False, z, z, 0.0
        if beta <= 1e-10 or beta >= 1.0 - 1e-10:
            return False, z, z, 0.0
        return True, x1, x2, beta

else:

    def _lle_split_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Compiled LLE backend is unavailable")


if njit is not None and _nrtl_activity_coefficients_numba is not None:

    @njit(cache=True)
    def _lle_split_nrtl_numba(
        z_input, T, tau_mode, tau_c, tau_d, tau_e, tau_f, tau_g, tau_tref, tau_energy, alpha,
        interaction_temperature_caps,
        max_iter, tol,
    ):
        z = _normalize(z_input)
        n = z.shape[0]
        x1 = np.empty(n, dtype=np.float64)
        x2 = np.empty(n, dtype=np.float64)
        if n == 2:
            x1[0] = 1.0 - 1e-5
            x1[1] = 1e-5
            x2[0] = 1e-5
            x2[1] = 1.0 - 1e-5
        else:
            for i in range(n):
                z_i = z[i]
                if i == 0:
                    x1[i] = min(0.95, z_i * 1.5)
                    x2[i] = max(0.05, z_i * 0.5)
                else:
                    x1[i] = max(0.05, z_i * 0.5)
                    x2[i] = min(0.95, z_i * 1.5)
        x1 = _normalize(x1)
        x2 = _normalize(x2)
        beta = 0.5
        K = np.empty(n, dtype=np.float64)

        for _iteration in range(max_iter):
            gamma1 = _nrtl_activity_coefficients_numba(
                x1, T, tau_mode, tau_c, tau_d, tau_e, tau_f, tau_g, tau_tref, tau_energy, alpha,
                interaction_temperature_caps,
            )
            gamma2 = _nrtl_activity_coefficients_numba(
                x2, T, tau_mode, tau_c, tau_d, tau_e, tau_f, tau_g, tau_tref, tau_energy, alpha,
                interaction_temperature_caps,
            )

            max_diff = 0.0
            for i in range(n):
                gamma2_i = gamma2[i]
                if gamma2_i < 1e-10:
                    gamma2_i = 1e-10
                K[i] = gamma1[i] / gamma2_i
                act1 = x1[i] * gamma1[i]
                act2 = x2[i] * gamma2[i]
                scale = max(max(act1, act2), 1e-10)
                diff = abs(act1 - act2) / scale
                if diff > max_diff:
                    max_diff = diff

            if max_diff < tol:
                phase_diff = 0.0
                for i in range(n):
                    phase_diff += abs(x1[i] - x2[i])
                phase_diff /= n
                if phase_diff < 0.01:
                    return False, z, z, 0.0
                if beta <= 1e-10 or beta >= 1.0 - 1e-10:
                    return False, z, z, 0.0
                return True, x1, x2, beta

            for _inner in range(20):
                f = 0.0
                df = 0.0
                for i in range(n):
                    km1 = K[i] - 1.0
                    denom = 1.0 + beta * km1
                    f += z[i] * km1 / denom
                    df -= z[i] * km1 * km1 / (denom * denom)
                if abs(df) < 1e-15:
                    break
                beta = beta - f / df
                if beta < 1e-12:
                    beta = 1e-12
                elif beta > 1.0 - 1e-12:
                    beta = 1.0 - 1e-12
                if abs(f) < 1e-10:
                    break

            for i in range(n):
                denom = 1.0 + beta * (K[i] - 1.0)
                if denom < 1e-10:
                    denom = 1e-10
                x1[i] = z[i] / denom
                x2[i] = K[i] * x1[i]
            x1 = _normalize(x1)
            x2 = _normalize(x2)

        phase_diff = 0.0
        for i in range(n):
            phase_diff += abs(x1[i] - x2[i])
        phase_diff /= n
        if phase_diff < 0.01:
            return False, z, z, 0.0
        if beta <= 1e-10 or beta >= 1.0 - 1e-10:
            return False, z, z, 0.0
        return True, x1, x2, beta

else:

    def _lle_split_nrtl_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Compiled NRTL LLE backend is unavailable")


if njit is not None and _uniquac_activity_coefficients_numba is not None:

    @njit(cache=True)
    def _lle_split_uniquac_numba(
        z_input, T, r, q, q_residual, tau_mode, tau_a, tau_b, tau_c, tau_d,
        tau_e, tau_tref,
        interaction_temperature_caps, max_iter, tol
    ):
        z = _normalize(z_input)
        n = z.shape[0]
        x1 = np.empty(n, dtype=np.float64)
        x2 = np.empty(n, dtype=np.float64)
        if n == 2:
            x1[0] = 1.0 - 1e-5
            x1[1] = 1e-5
            x2[0] = 1e-5
            x2[1] = 1.0 - 1e-5
        else:
            for i in range(n):
                z_i = z[i]
                if i == 0:
                    x1[i] = min(0.95, z_i * 1.5)
                    x2[i] = max(0.05, z_i * 0.5)
                else:
                    x1[i] = max(0.05, z_i * 0.5)
                    x2[i] = min(0.95, z_i * 1.5)
        x1 = _normalize(x1)
        x2 = _normalize(x2)
        beta = 0.5
        K = np.empty(n, dtype=np.float64)

        for _iteration in range(max_iter):
            gamma1 = _uniquac_activity_coefficients_numba(
                x1, T, r, q, q_residual, tau_mode, tau_a, tau_b, tau_c, tau_d,
                tau_e, tau_tref,
                interaction_temperature_caps,
            )
            gamma2 = _uniquac_activity_coefficients_numba(
                x2, T, r, q, q_residual, tau_mode, tau_a, tau_b, tau_c, tau_d,
                tau_e, tau_tref,
                interaction_temperature_caps,
            )

            max_diff = 0.0
            for i in range(n):
                gamma2_i = gamma2[i]
                if gamma2_i < 1e-10:
                    gamma2_i = 1e-10
                K[i] = gamma1[i] / gamma2_i
                act1 = x1[i] * gamma1[i]
                act2 = x2[i] * gamma2[i]
                scale = max(max(act1, act2), 1e-10)
                diff = abs(act1 - act2) / scale
                if diff > max_diff:
                    max_diff = diff

            if max_diff < tol:
                phase_diff = 0.0
                for i in range(n):
                    phase_diff += abs(x1[i] - x2[i])
                phase_diff /= n
                if phase_diff < 0.01:
                    return False, z, z, 0.0
                if beta <= 1e-10 or beta >= 1.0 - 1e-10:
                    return False, z, z, 0.0
                return True, x1, x2, beta

            for _inner in range(20):
                f = 0.0
                df = 0.0
                for i in range(n):
                    km1 = K[i] - 1.0
                    denom = 1.0 + beta * km1
                    f += z[i] * km1 / denom
                    df -= z[i] * km1 * km1 / (denom * denom)
                if abs(df) < 1e-15:
                    break
                beta = beta - f / df
                if beta < 1e-12:
                    beta = 1e-12
                elif beta > 1.0 - 1e-12:
                    beta = 1.0 - 1e-12
                if abs(f) < 1e-10:
                    break

            for i in range(n):
                denom = 1.0 + beta * (K[i] - 1.0)
                if denom < 1e-10:
                    denom = 1e-10
                x1[i] = z[i] / denom
                x2[i] = K[i] * x1[i]
            x1 = _normalize(x1)
            x2 = _normalize(x2)

        phase_diff = 0.0
        for i in range(n):
            phase_diff += abs(x1[i] - x2[i])
        phase_diff /= n
        if phase_diff < 0.01:
            return False, z, z, 0.0
        if beta <= 1e-10 or beta >= 1.0 - 1e-10:
            return False, z, z, 0.0
        return True, x1, x2, beta

else:

    def _lle_split_uniquac_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Compiled UNIQUAC LLE backend is unavailable")
