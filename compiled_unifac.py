"""
Compiled UNIFAC activity-coefficient backend.

The reference UNIFAC implementation in unifac.py is intentionally readable and
dict-based. This module builds fixed numeric arrays for one component set and
uses Numba for the hot activity-coefficient kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, typeof
except Exception:  # pragma: no cover - exercised only without optional numba
    njit = None
    typeof = None


@dataclass
class CompiledUNIFACBackend:
    """Array-backed UNIFAC gamma calculator for a fixed component list."""

    components: list[str]
    nu: np.ndarray
    r: np.ndarray
    q: np.ndarray
    subgroup_q: np.ndarray
    interactions: np.ndarray
    interactions_b: np.ndarray
    interactions_c: np.ndarray
    variant_id: int = 0
    compilation_complete: bool = False

    @classmethod
    def from_model(
        cls,
        unifac_model,
        components: list[str],
        component_groups: dict[str, dict[str, int]],
    ) -> "CompiledUNIFACBackend | None":
        if njit is None:
            return None

        resolved_groups = [
            unifac_model._resolve_groups(component_groups[component])
            for component in components
        ]
        subgroup_numbers = sorted({
            subgroup
            for groups in resolved_groups
            for subgroup in groups
        })
        if not subgroup_numbers:
            return None

        subgroup_index = {
            subgroup: index
            for index, subgroup in enumerate(subgroup_numbers)
        }
        n_components = len(components)
        n_groups = len(subgroup_numbers)

        nu = np.zeros((n_components, n_groups), dtype=np.float64)
        r = np.zeros(n_components, dtype=np.float64)
        q = np.zeros(n_components, dtype=np.float64)
        subgroup_q = np.zeros(n_groups, dtype=np.float64)
        main_groups = np.zeros(n_groups, dtype=np.int64)

        for group_number, index in subgroup_index.items():
            subgroup = unifac_model.subgroups[group_number]
            subgroup_q[index] = float(subgroup.Q)
            main_groups[index] = int(subgroup.main_group)

        for component_index, groups in enumerate(resolved_groups):
            for group_number, count in groups.items():
                group_index = subgroup_index[group_number]
                count = float(count)
                nu[component_index, group_index] = count
                subgroup = unifac_model.subgroups[group_number]
                r[component_index] += subgroup.R * count
                q[component_index] += subgroup.Q * count

        interactions = np.zeros((n_groups, n_groups), dtype=np.float64)
        interactions_b = np.zeros((n_groups, n_groups), dtype=np.float64)
        interactions_c = np.zeros((n_groups, n_groups), dtype=np.float64)
        variant_id = 1 if getattr(unifac_model, "variant", "UNIFAC") in ("UNIFDMD", "UNIFM2", "UNIFNIST") else 0
        for i in range(n_groups):
            for j in range(n_groups):
                main_i = int(main_groups[i])
                main_j = int(main_groups[j])
                coeffs = unifac_model.interaction_coefficients.get((main_i, main_j))
                if coeffs is not None:
                    interactions[i, j] = float(coeffs[0])
                    interactions_b[i, j] = float(coeffs[1])
                    interactions_c[i, j] = float(coeffs[2])
                else:
                    interactions[i, j] = float(
                        unifac_model.get_interaction(main_i, main_j)
                    )

        backend = cls(
            components=list(components),
            nu=nu,
            r=r,
            q=q,
            subgroup_q=subgroup_q,
            interactions=interactions,
            interactions_b=interactions_b,
            interactions_c=interactions_c,
            variant_id=variant_id,
        )
        backend.compile_kernels()
        return backend

    def compile_kernels(self) -> None:
        if self.compilation_complete or typeof is None:
            return
        composition = np.zeros(len(self.components), dtype=np.float64)
        args = (
            self.nu, self.r, self.q, self.subgroup_q,
            self.interactions, self.interactions_b, self.interactions_c,
            self.variant_id, composition, 298.15,
        )
        _activity_coefficients_numba.compile(
            tuple(typeof(argument) for argument in args)
        )
        self.compilation_complete = True

    def activity_coefficients(self, x: list[float] | np.ndarray, T: float) -> list[float]:
        x_array = np.asarray(x, dtype=np.float64)
        gamma = _activity_coefficients_numba(
            self.nu,
            self.r,
            self.q,
            self.subgroup_q,
            self.interactions,
            self.interactions_b,
            self.interactions_c,
            self.variant_id,
            x_array,
            float(T),
        )
        return gamma.tolist()


if njit is not None:

    @njit(cache=True)
    def _group_activity_coefficients_numba(X, subgroup_q, interactions, interactions_b, interactions_c, T):
        n_groups = X.shape[0]
        theta = np.zeros(n_groups, dtype=np.float64)
        psi = np.empty((n_groups, n_groups), dtype=np.float64)
        ln_gamma = np.zeros(n_groups, dtype=np.float64)

        sum_qx = 0.0
        for k in range(n_groups):
            sum_qx += subgroup_q[k] * X[k]
        if sum_qx < 1e-10:
            sum_qx = 1e-10

        for k in range(n_groups):
            theta[k] = subgroup_q[k] * X[k] / sum_qx

        for m in range(n_groups):
            for n in range(n_groups):
                exponent = -(
                    interactions[m, n]
                    + interactions_b[m, n] * T
                    + interactions_c[m, n] * T * T
                ) / T
                psi[m, n] = math.exp(exponent)

        denominator = np.zeros(n_groups, dtype=np.float64)
        for m in range(n_groups):
            total = 0.0
            for n in range(n_groups):
                total += theta[n] * psi[n, m]
            if total < 1e-10:
                total = 1e-10
            denominator[m] = total

        for k in range(n_groups):
            sum_theta_psi_mk = denominator[k]
            term2 = 0.0
            for m in range(n_groups):
                term2 += theta[m] * psi[k, m] / denominator[m]
            ln_gamma[k] = subgroup_q[k] * (1.0 - math.log(sum_theta_psi_mk) - term2)

        return ln_gamma


    @njit(cache=True)
    def _activity_coefficients_numba(
        nu, r, q, subgroup_q, interactions, interactions_b, interactions_c,
        variant_id, x, T,
    ):
        n_components = nu.shape[0]
        n_groups = nu.shape[1]
        z_coordination = 10.0

        x_sum = 0.0
        for i in range(n_components):
            if x[i] > 0.0:
                x_sum += x[i]
        if x_sum < 1e-300:
            x_sum = 1.0

        x_norm = np.empty(n_components, dtype=np.float64)
        for i in range(n_components):
            value = x[i] / x_sum
            if value < 0.0:
                value = 0.0
            x_norm[i] = value

        sum_xr = 0.0
        sum_xq = 0.0
        sum_xr34 = 0.0
        for i in range(n_components):
            sum_xr += x_norm[i] * r[i]
            sum_xq += x_norm[i] * q[i]
            if variant_id == 1:
                sum_xr34 += x_norm[i] * (r[i] ** 0.75)

        if sum_xr < 1e-10:
            sum_xr = 1e-10
        if sum_xq < 1e-10:
            sum_xq = 1e-10
        if sum_xr34 < 1e-10:
            sum_xr34 = 1e-10

        l_value = np.empty(n_components, dtype=np.float64)
        sum_xl = 0.0
        if variant_id != 1:
            for i in range(n_components):
                l_value[i] = (z_coordination / 2.0) * (r[i] - q[i]) - (r[i] - 1.0)
                sum_xl += x_norm[i] * l_value[i]

        ln_gamma_c = np.zeros(n_components, dtype=np.float64)
        for i in range(n_components):
            if variant_id == 1:
                v_prime = (r[i] ** 0.75) / sum_xr34
                v_value = r[i] / sum_xr
                f_value = q[i] / sum_xq
                ratio = 1.0
                if f_value > 1e-10:
                    ratio = v_value / f_value
                if v_prime <= 0.0:
                    v_prime = 1e-10
                if ratio <= 0.0:
                    ratio = 1e-10
                ln_gamma_c[i] = (
                    1.0
                    - v_prime
                    + math.log(v_prime)
                    - 5.0 * q[i] * (1.0 - ratio + math.log(ratio))
                )
            else:
                phi_over_x = r[i] / sum_xr
                theta_over_phi = q[i] * sum_xr / (r[i] * sum_xq)
                if phi_over_x <= 0.0:
                    phi_over_x = 1e-10
                if theta_over_phi <= 0.0:
                    theta_over_phi = 1e-10

                ln_gamma_c[i] = (
                    math.log(phi_over_x)
                    + (z_coordination / 2.0) * q[i] * math.log(theta_over_phi)
                    + l_value[i]
                    - phi_over_x * sum_xl
                )

        X_mix = np.zeros(n_groups, dtype=np.float64)
        denominator = 0.0
        for i in range(n_components):
            total_groups_i = 0.0
            for k in range(n_groups):
                total_groups_i += nu[i, k]
            denominator += x_norm[i] * total_groups_i
            for k in range(n_groups):
                X_mix[k] += x_norm[i] * nu[i, k]

        if denominator < 1e-10:
            denominator = 1e-10
        for k in range(n_groups):
            X_mix[k] /= denominator

        ln_gamma_group_mix = _group_activity_coefficients_numba(
            X_mix, subgroup_q, interactions, interactions_b, interactions_c, T
        )

        gamma = np.empty(n_components, dtype=np.float64)
        for i in range(n_components):
            X_pure = np.zeros(n_groups, dtype=np.float64)
            pure_denominator = 0.0
            for k in range(n_groups):
                pure_denominator += nu[i, k]
            if pure_denominator < 1e-10:
                pure_denominator = 1e-10
            for k in range(n_groups):
                X_pure[k] = nu[i, k] / pure_denominator

            ln_gamma_group_pure = _group_activity_coefficients_numba(
                X_pure, subgroup_q, interactions, interactions_b, interactions_c, T
            )
            ln_gamma_r = 0.0
            for k in range(n_groups):
                if nu[i, k] != 0.0:
                    ln_gamma_r += nu[i, k] * (
                        ln_gamma_group_mix[k] - ln_gamma_group_pure[k]
                    )
            gamma[i] = math.exp(ln_gamma_c[i] + ln_gamma_r)

        return gamma

else:

    def _activity_coefficients_numba(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("Numba is not available")
