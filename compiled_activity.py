"""Compiled NRTL and UNIQUAC activity-coefficient backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from .physical_constants import R_J_MOL_K
else:
    from physical_constants import R_J_MOL_K

try:
    from numba import njit, typeof

    if __package__ and __package__.split(".", 1)[0] == "pfdsim":
        from .compiled_cache import numba_cached
    else:
        from compiled_cache import numba_cached
except Exception:  # pragma: no cover - exercised only without optional numba
    njit = None
    typeof = None


@dataclass
class CompiledNRTLBackend:
    """Array-backed NRTL gamma calculator for a fixed component list."""

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
    interaction_tmin: np.ndarray
    interaction_tmax: np.ndarray
    compilation_complete: bool = False

    @classmethod
    def from_thermo(cls, thermo) -> "CompiledNRTLBackend | None":
        if njit is None:
            return None
        params = thermo._nrtl_parameter_matrices()
        backend = cls(
            components=list(thermo.components),
            tau_mode=np.asarray(params["tau_mode"], dtype=np.int64),
            tau_c=np.asarray(params["tau_c"], dtype=np.float64),
            tau_d=np.asarray(params["tau_d"], dtype=np.float64),
            tau_e=np.asarray(params["tau_e"], dtype=np.float64),
            tau_f=np.asarray(params["tau_f"], dtype=np.float64),
            tau_g=np.asarray(params["tau_g"], dtype=np.float64),
            tau_tref=np.asarray(params["tau_tref"], dtype=np.float64),
            tau_energy=np.asarray(params["tau_energy"], dtype=np.float64),
            alpha=np.asarray(params["alpha"], dtype=np.float64),
            interaction_tmin=np.asarray(params["interaction_tmin"], dtype=np.float64),
            interaction_tmax=np.asarray(params["interaction_tmax"], dtype=np.float64),
        )
        backend.compile_kernels()
        return backend

    def compile_kernels(self) -> None:
        if self.compilation_complete or typeof is None:
            return
        composition = np.zeros(len(self.components), dtype=np.float64)
        args = (
            composition,
            298.15,
            self.tau_mode,
            self.tau_c,
            self.tau_d,
            self.tau_e,
            self.tau_f,
            self.tau_g,
            self.tau_tref,
            self.tau_energy,
            self.alpha,
            self.interaction_tmin,
            self.interaction_tmax,
        )
        _nrtl_activity_coefficients_numba.compile(
            tuple(typeof(argument) for argument in args)
        )
        _nrtl_excess_enthalpy_numba.compile(
            tuple(typeof(argument) for argument in args)
        )
        self.compilation_complete = True

    def activity_coefficients(
        self, x: list[float] | np.ndarray, T: float
    ) -> list[float]:
        x_array = np.asarray(x, dtype=np.float64)
        return _nrtl_activity_coefficients_numba(
            x_array,
            float(T),
            self.tau_mode,
            self.tau_c,
            self.tau_d,
            self.tau_e,
            self.tau_f,
            self.tau_g,
            self.tau_tref,
            self.tau_energy,
            self.alpha,
            self.interaction_tmin,
            self.interaction_tmax,
        ).tolist()

    def excess_enthalpy(self, x: list[float] | np.ndarray, T: float) -> float:
        x_array = np.asarray(x, dtype=np.float64)
        return float(
            _nrtl_excess_enthalpy_numba(
                x_array,
                float(T),
                *self.enthalpy_parameters(),
            )
        )

    def enthalpy_parameters(self):
        """Shared numeric arguments for scalar and fused caloric kernels."""
        return (
            self.tau_mode,
            self.tau_c,
            self.tau_d,
            self.tau_e,
            self.tau_f,
            self.tau_g,
            self.tau_tref,
            self.tau_energy,
            self.alpha,
            self.interaction_tmin,
            self.interaction_tmax,
        )


@dataclass
class CompiledUNIQUACBackend:
    """Array-backed UNIQUAC gamma calculator for a fixed component list."""

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
    interaction_tmin: np.ndarray
    interaction_tmax: np.ndarray
    compilation_complete: bool = False

    @classmethod
    def from_thermo(cls, thermo, components=None) -> "CompiledUNIQUACBackend | None":
        if njit is None:
            return None
        selected_components = list(components or thermo.components)
        indices = [thermo.components.index(comp) for comp in selected_components]
        index_array = np.asarray(indices, dtype=np.int64)
        params = thermo._uniquac_parameter_matrices()
        backend = cls(
            components=selected_components,
            r=np.asarray(params["r_combinatorial"], dtype=np.float64)[index_array],
            q=np.asarray(params["q_combinatorial"], dtype=np.float64)[index_array],
            q_residual=np.asarray(params["q_residual"], dtype=np.float64)[index_array],
            tau_mode=np.asarray(params["tau_mode"], dtype=np.int64)[index_array, :][
                :, index_array
            ],
            tau_a=np.asarray(params["tau_a"], dtype=np.float64)[index_array, :][
                :, index_array
            ],
            tau_b=np.asarray(params["tau_b"], dtype=np.float64)[index_array, :][
                :, index_array
            ],
            tau_c=np.asarray(params["tau_c"], dtype=np.float64)[index_array, :][
                :, index_array
            ],
            tau_d=np.asarray(params["tau_d"], dtype=np.float64)[index_array, :][
                :, index_array
            ],
            tau_e=np.asarray(params["tau_e"], dtype=np.float64)[index_array, :][
                :, index_array
            ],
            tau_tref=np.asarray(params["tau_tref"], dtype=np.float64)[index_array, :][
                :, index_array
            ],
            interaction_tmin=np.asarray(params["interaction_tmin"], dtype=np.float64)[
                index_array, :
            ][:, index_array],
            interaction_tmax=np.asarray(params["interaction_tmax"], dtype=np.float64)[
                index_array, :
            ][:, index_array],
        )
        backend.compile_kernels()
        return backend

    def compile_kernels(self) -> None:
        if self.compilation_complete or typeof is None:
            return
        composition = np.zeros(len(self.components), dtype=np.float64)
        args = (
            composition,
            298.15,
            self.r,
            self.q,
            self.q_residual,
            self.tau_mode,
            self.tau_a,
            self.tau_b,
            self.tau_c,
            self.tau_d,
            self.tau_e,
            self.tau_tref,
            self.interaction_tmin,
            self.interaction_tmax,
        )
        _uniquac_activity_coefficients_numba.compile(
            tuple(typeof(argument) for argument in args)
        )
        _uniquac_excess_enthalpy_numba.compile(
            tuple(typeof(argument) for argument in args)
        )
        self.compilation_complete = True

    def activity_coefficients(
        self, x: list[float] | np.ndarray, T: float
    ) -> list[float]:
        x_array = np.asarray(x, dtype=np.float64)
        return _uniquac_activity_coefficients_numba(
            x_array,
            float(T),
            self.r,
            self.q,
            self.q_residual,
            self.tau_mode,
            self.tau_a,
            self.tau_b,
            self.tau_c,
            self.tau_d,
            self.tau_e,
            self.tau_tref,
            self.interaction_tmin,
            self.interaction_tmax,
        ).tolist()

    def excess_enthalpy(self, x: list[float] | np.ndarray, T: float) -> float:
        x_array = np.asarray(x, dtype=np.float64)
        return float(
            _uniquac_excess_enthalpy_numba(
                x_array,
                float(T),
                *self.enthalpy_parameters(),
            )
        )

    def enthalpy_parameters(self):
        """Shared numeric arguments for scalar and fused caloric kernels."""
        return (
            self.r,
            self.q,
            self.q_residual,
            self.tau_mode,
            self.tau_a,
            self.tau_b,
            self.tau_c,
            self.tau_d,
            self.tau_e,
            self.tau_tref,
            self.interaction_tmin,
            self.interaction_tmax,
        )


if njit is not None:
    _cached_kernel = numba_cached(njit)

    @_cached_kernel
    def _nrtl_activity_coefficients_numba(
        x,
        T,
        tau_mode,
        tau_c,
        tau_d,
        tau_e,
        tau_f,
        tau_g,
        tau_tref,
        tau_energy,
        alpha,
        interaction_tmin,
        interaction_tmax,
    ):
        n = x.shape[0]
        out = np.ones(n, dtype=np.float64)
        total = 0.0
        for i in range(n):
            if x[i] > 0.0:
                total += x[i]
        if total <= 0.0:
            return out

        xn = np.empty(n, dtype=np.float64)
        for i in range(n):
            value = x[i]
            if value < 0.0:
                value = 0.0
            xn[i] = value / total

        tau = np.zeros((n, n), dtype=np.float64)
        G = np.ones((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    tau[i, j] = 0.0
                    G[i, j] = 1.0
                    continue
                encoded_mode = tau_mode[i, j]
                mode = encoded_mode % 10
                extrapolation_mode = encoded_mode // 10
                interaction_T = T
                if extrapolation_mode == 1 and interaction_T < interaction_tmin[i, j]:
                    interaction_T = interaction_tmin[i, j]
                elif extrapolation_mode == 1 and interaction_T > interaction_tmax[i, j]:
                    interaction_T = interaction_tmax[i, j]
                if mode == 1:
                    tref = tau_tref[i, j]
                    boundary = interaction_T
                    if extrapolation_mode >= 2:
                        if T < interaction_tmin[i, j]:
                            boundary = interaction_tmin[i, j]
                        elif T > interaction_tmax[i, j]:
                            boundary = interaction_tmax[i, j]
                    tau_ij = (
                        tau_c[i, j]
                        + tau_d[i, j] / boundary
                        + tau_e[i, j]
                        * ((tref - boundary) / boundary + np.log(boundary / tref))
                        + tau_f[i, j] * boundary
                        + tau_g[i, j] * boundary * boundary
                    )
                    if extrapolation_mode >= 2 and boundary != T:
                        derivative = (
                            -tau_d[i, j] / (boundary * boundary)
                            + tau_e[i, j] * (boundary - tref) / (boundary * boundary)
                            + tau_f[i, j]
                            + 2.0 * tau_g[i, j] * boundary
                        )
                        if extrapolation_mode == 2:
                            outside_b = -boundary * boundary * derivative
                            outside_a = tau_ij + boundary * derivative
                            tau_ij = outside_a + outside_b / T
                        elif extrapolation_mode == 3:
                            outside_b = boundary * (
                                2.0 * tau_ij + boundary * derivative
                            )
                            outside_c = (
                                -boundary * boundary * (tau_ij + boundary * derivative)
                            )
                            tau_ij = outside_b / T + outside_c / (T * T)
                        else:
                            outside_c = (
                                boundary
                                * boundary
                                * (3.0 * tau_ij + boundary * derivative)
                            )
                            outside_d = -(boundary**3) * (
                                2.0 * tau_ij + boundary * derivative
                            )
                            tau_ij = outside_c / (T * T) + outside_d / (T * T * T)
                elif mode == 2:
                    tau_ij = tau_energy[i, j] / interaction_T
                else:
                    tau_ij = 0.0
                tau[i, j] = tau_ij
                G[i, j] = np.exp(-alpha[i, j] * tau_ij)

        denom_col = np.empty(n, dtype=np.float64)
        weighted_tau_col = np.empty(n, dtype=np.float64)
        for j in range(n):
            denom = 0.0
            weighted = 0.0
            for k in range(n):
                denom += xn[k] * G[k, j]
                weighted += xn[k] * tau[k, j] * G[k, j]
            if abs(denom) < 1e-30:
                denom = 1e-30
            denom_col[j] = denom
            weighted_tau_col[j] = weighted / denom

        for i in range(n):
            ln_gamma = 0.0
            for j in range(n):
                ln_gamma += xn[j] * tau[j, i] * G[j, i] / denom_col[i]
            for j in range(n):
                ln_gamma += (
                    xn[j] * G[i, j] / denom_col[j] * (tau[i, j] - weighted_tau_col[j])
                )
            if ln_gamma > 50.0:
                ln_gamma = 50.0
            elif ln_gamma < -50.0:
                ln_gamma = -50.0
            value = np.exp(ln_gamma)
            if value < 1e-12:
                value = 1e-12
            out[i] = value
        return out

    @_cached_kernel
    def _nrtl_excess_enthalpy_numba(
        x,
        T,
        tau_mode,
        tau_c,
        tau_d,
        tau_e,
        tau_f,
        tau_g,
        tau_tref,
        tau_energy,
        alpha,
        interaction_tmin,
        interaction_tmax,
    ):
        total = 0.0
        for value in x:
            if value > 0.0:
                total += value
        if total <= 0.0:
            return 0.0
        xn = np.empty(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            xn[i] = max(x[i], 0.0) / total
        dT = max(1.0e-3, 1.0e-4 * T)
        T_low = max(1.0, T - dT)
        T_high = T + dT
        gamma_low = _nrtl_activity_coefficients_numba(
            xn,
            T_low,
            tau_mode,
            tau_c,
            tau_d,
            tau_e,
            tau_f,
            tau_g,
            tau_tref,
            tau_energy,
            alpha,
            interaction_tmin,
            interaction_tmax,
        )
        gamma_high = _nrtl_activity_coefficients_numba(
            xn,
            T_high,
            tau_mode,
            tau_c,
            tau_d,
            tau_e,
            tau_f,
            tau_g,
            tau_tref,
            tau_energy,
            alpha,
            interaction_tmin,
            interaction_tmax,
        )
        derivative_sum = 0.0
        for i in range(xn.shape[0]):
            if xn[i] > 0.0:
                derivative_sum += (
                    xn[i]
                    * (
                        np.log(max(gamma_high[i], 1.0e-300))
                        - np.log(max(gamma_low[i], 1.0e-300))
                    )
                    / (T_high - T_low)
                )
        return -R_J_MOL_K * T * T * derivative_sum

    @_cached_kernel
    def _uniquac_activity_coefficients_numba(
        x,
        T,
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
        interaction_tmin,
        interaction_tmax,
    ):
        n = x.shape[0]
        out = np.ones(n, dtype=np.float64)
        total = 0.0
        for i in range(n):
            if x[i] > 0.0:
                total += x[i]
        if total <= 0.0:
            return out

        xn = np.empty(n, dtype=np.float64)
        for i in range(n):
            value = x[i]
            if value < 0.0:
                value = 0.0
            xn[i] = value / total

        tau = np.ones((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    tau[i, j] = 1.0
                    continue
                encoded_mode = tau_mode[i, j]
                mode = encoded_mode % 10
                extrapolation_mode = encoded_mode // 10
                interaction_T = T
                if extrapolation_mode == 1 and interaction_T < interaction_tmin[i, j]:
                    interaction_T = interaction_tmin[i, j]
                elif extrapolation_mode == 1 and interaction_T > interaction_tmax[i, j]:
                    interaction_T = interaction_tmax[i, j]
                if mode == 1:
                    tref = tau_tref[i, j]
                    boundary = interaction_T
                    if extrapolation_mode >= 2:
                        if T < interaction_tmin[i, j]:
                            boundary = interaction_tmin[i, j]
                        elif T > interaction_tmax[i, j]:
                            boundary = interaction_tmax[i, j]
                    exponent = (
                        tau_a[i, j]
                        + tau_b[i, j] / boundary
                        + tau_c[i, j]
                        * ((tref - boundary) / boundary + np.log(boundary / tref))
                        + tau_d[i, j] * boundary
                        + tau_e[i, j] * boundary * boundary
                    )
                    if extrapolation_mode >= 2 and boundary != T:
                        derivative = (
                            -tau_b[i, j] / (boundary * boundary)
                            + tau_c[i, j] * (boundary - tref) / (boundary * boundary)
                            + tau_d[i, j]
                            + 2.0 * tau_e[i, j] * boundary
                        )
                        if extrapolation_mode == 2:
                            outside_b = -boundary * boundary * derivative
                            outside_a = exponent + boundary * derivative
                            exponent = outside_a + outside_b / T
                        elif extrapolation_mode == 3:
                            outside_b = boundary * (
                                2.0 * exponent + boundary * derivative
                            )
                            outside_c = (
                                -boundary
                                * boundary
                                * (exponent + boundary * derivative)
                            )
                            exponent = outside_b / T + outside_c / (T * T)
                        else:
                            outside_c = (
                                boundary
                                * boundary
                                * (3.0 * exponent + boundary * derivative)
                            )
                            outside_d = -(boundary**3) * (
                                2.0 * exponent + boundary * derivative
                            )
                            exponent = outside_c / (T * T) + outside_d / (T * T * T)
                elif mode == 2:
                    exponent = -tau_a[i, j] / interaction_T
                else:
                    exponent = 0.0
                if exponent > 50.0:
                    exponent = 50.0
                elif exponent < -50.0:
                    exponent = -50.0
                tau[i, j] = np.exp(exponent)

        rx = 0.0
        qx = 0.0
        qx_residual = 0.0
        for i in range(n):
            rx += r[i] * xn[i]
            qx += q[i] * xn[i]
            qx_residual += q_residual[i] * xn[i]
        if abs(rx) < 1e-30:
            rx = 1e-30
        if abs(qx) < 1e-30:
            qx = 1e-30
        if abs(qx_residual) < 1e-30:
            qx_residual = 1e-30

        phi = np.empty(n, dtype=np.float64)
        theta_residual = np.empty(n, dtype=np.float64)
        ell = np.empty(n, dtype=np.float64)
        xl_sum = 0.0
        for i in range(n):
            phi[i] = r[i] * xn[i] / rx
            theta_residual[i] = q_residual[i] * xn[i] / qx_residual
            ell[i] = 5.0 * (r[i] - q[i]) - (r[i] - 1.0)
            xl_sum += xn[i] * ell[i]

        theta_tau_col = np.empty(n, dtype=np.float64)
        for i in range(n):
            total_i = 0.0
            for j in range(n):
                total_i += theta_residual[j] * tau[j, i]
            if abs(total_i) < 1e-30:
                total_i = 1e-30
            theta_tau_col[i] = total_i

        for i in range(n):
            phi_over_x = r[i] / rx
            if phi_over_x < 1e-30:
                phi_over_x = 1e-30
            theta_over_phi = q[i] * rx / (r[i] * qx)
            if theta_over_phi < 1e-30:
                theta_over_phi = 1e-30

            ln_gamma_c = (
                np.log(phi_over_x)
                + 5.0 * q[i] * np.log(theta_over_phi)
                + ell[i]
                - phi_over_x * xl_sum
            )
            residual_sum = 0.0
            for j in range(n):
                residual_sum += theta_residual[j] * tau[i, j] / theta_tau_col[j]
            ln_gamma_r = q_residual[i] * (1.0 - np.log(theta_tau_col[i]) - residual_sum)
            ln_gamma = ln_gamma_c + ln_gamma_r
            if ln_gamma > 50.0:
                ln_gamma = 50.0
            elif ln_gamma < -50.0:
                ln_gamma = -50.0
            value = np.exp(ln_gamma)
            if value < 1e-12:
                value = 1e-12
            out[i] = value
        return out

    @_cached_kernel
    def _uniquac_excess_enthalpy_numba(
        x,
        T,
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
        interaction_tmin,
        interaction_tmax,
    ):
        total = 0.0
        for value in x:
            if value > 0.0:
                total += value
        if total <= 0.0:
            return 0.0
        xn = np.empty(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            xn[i] = max(x[i], 0.0) / total
        dT = max(1.0e-3, 1.0e-4 * T)
        T_low = max(1.0, T - dT)
        T_high = T + dT
        gamma_low = _uniquac_activity_coefficients_numba(
            xn,
            T_low,
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
            interaction_tmin,
            interaction_tmax,
        )
        gamma_high = _uniquac_activity_coefficients_numba(
            xn,
            T_high,
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
            interaction_tmin,
            interaction_tmax,
        )
        derivative_sum = 0.0
        for i in range(xn.shape[0]):
            if xn[i] > 0.0:
                derivative_sum += (
                    xn[i]
                    * (
                        np.log(max(gamma_high[i], 1.0e-300))
                        - np.log(max(gamma_low[i], 1.0e-300))
                    )
                    / (T_high - T_low)
                )
        return -R_J_MOL_K * T * T * derivative_sum

else:

    def _nrtl_activity_coefficients_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Compiled NRTL backend is unavailable")

    def _nrtl_excess_enthalpy_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Compiled NRTL backend is unavailable")

    def _uniquac_activity_coefficients_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Compiled UNIQUAC backend is unavailable")

    def _uniquac_excess_enthalpy_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Compiled UNIQUAC backend is unavailable")
