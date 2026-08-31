"""Compiled numerical kernel for the fixed-component PSRK model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, typeof
except Exception:  # pragma: no cover - exercised only without optional numba
    njit = None
    typeof = None


R_BAR_CM3_PER_MOL_K = 83.14462618


@dataclass
class CompiledPSRKBackend:
    """Dense array representation of one initialized :class:`PSRK` model."""

    components: tuple[str, ...]
    nu: np.ndarray
    r: np.ndarray
    q: np.ndarray
    ell: np.ndarray
    subgroup_q: np.ndarray
    interaction_a: np.ndarray
    interaction_b: np.ndarray
    interaction_c: np.ndarray
    Tc: np.ndarray
    Pc: np.ndarray
    omega: np.ndarray
    uses_mathias_copeman: np.ndarray
    c1: np.ndarray
    c2: np.ndarray
    c3: np.ndarray
    a0: np.ndarray
    pure_b: np.ndarray
    q1: float
    gas_constant: float
    compilation_complete: bool = False

    @classmethod
    def from_psrk(cls, model) -> "CompiledPSRKBackend | None":
        if njit is None:
            return None

        subgroup_ids = sorted({
            subgroup_id
            for component in model._component_data
            for subgroup_id, _ in component.subgroups
        })
        if not subgroup_ids:
            return None
        subgroup_index = {
            subgroup_id: index
            for index, subgroup_id in enumerate(subgroup_ids)
        }
        n_components = len(model._component_data)
        n_groups = len(subgroup_ids)

        nu = np.zeros((n_components, n_groups), dtype=np.float64)
        subgroup_q = np.zeros(n_groups, dtype=np.float64)
        main_groups = np.zeros(n_groups, dtype=np.int64)
        for subgroup_id, index in subgroup_index.items():
            subgroup = model._subgroups[subgroup_id]
            subgroup_q[index] = float(subgroup.Q)
            main_groups[index] = int(subgroup.main_group_id)

        component_data = model._component_data
        for component_index, component in enumerate(component_data):
            for subgroup_id, count in component.subgroups:
                nu[component_index, subgroup_index[subgroup_id]] = float(count)

        interaction_a = np.zeros((n_groups, n_groups), dtype=np.float64)
        interaction_b = np.zeros((n_groups, n_groups), dtype=np.float64)
        interaction_c = np.zeros((n_groups, n_groups), dtype=np.float64)
        for source in range(n_groups):
            for target in range(n_groups):
                if source == target or main_groups[source] == main_groups[target]:
                    continue
                coefficients = model._interactions[
                    (int(main_groups[source]), int(main_groups[target]))
                ]
                interaction_a[source, target] = float(coefficients[0])
                interaction_b[source, target] = float(coefficients[1])
                interaction_c[source, target] = float(coefficients[2])

        r = np.asarray([component.r for component in component_data], dtype=np.float64)
        q = np.asarray([component.q for component in component_data], dtype=np.float64)
        ell = 5.0 * (r - q) - (r - 1.0)
        backend = cls(
            components=tuple(model.components),
            nu=nu,
            r=r,
            q=q,
            ell=ell,
            subgroup_q=subgroup_q,
            interaction_a=interaction_a,
            interaction_b=interaction_b,
            interaction_c=interaction_c,
            Tc=np.asarray([component.Tc_K for component in component_data], dtype=np.float64),
            Pc=np.asarray([component.Pc_bar for component in component_data], dtype=np.float64),
            omega=np.asarray([component.omega for component in component_data], dtype=np.float64),
            uses_mathias_copeman=np.asarray(
                [component.uses_mathias_copeman for component in component_data],
                dtype=np.bool_,
            ),
            c1=np.asarray([component.c1 for component in component_data], dtype=np.float64),
            c2=np.asarray([component.c2 for component in component_data], dtype=np.float64),
            c3=np.asarray([component.c3 for component in component_data], dtype=np.float64),
            a0=np.asarray([component.a0 for component in component_data], dtype=np.float64),
            pure_b=np.asarray([component.b for component in component_data], dtype=np.float64),
            q1=float(model._mixing_rule.q1),
            gas_constant=R_BAR_CM3_PER_MOL_K,
        )
        backend.compile_kernels()
        return backend

    def compile_kernels(self) -> None:
        """Compile/load kernels by signature without evaluating a state."""
        if self.compilation_complete or typeof is None:
            return
        composition = np.zeros(len(self.components), dtype=np.float64)

        def compile_for(dispatcher, *args):
            dispatcher.compile(tuple(typeof(argument) for argument in args))

        activity_args = (
            composition, 300.0, self.nu, self.r, self.q, self.ell,
            self.subgroup_q, self.interaction_a, self.interaction_b,
            self.interaction_c,
        )
        mixture_tail = (
            self.nu, self.r, self.q, self.ell, self.subgroup_q,
            self.interaction_a, self.interaction_b, self.interaction_c,
            self.Tc, self.omega, self.uses_mathias_copeman,
            self.c1, self.c2, self.c3, self.a0, self.pure_b,
            self.q1, self.gas_constant,
        )
        compile_for(_psrk_activity_state_numba, *activity_args)
        compile_for(
            _psrk_mixture_parameters_numba,
            300.0, composition, *mixture_tail,
        )
        compile_for(
            _psrk_roots_numba,
            300.0, 1.0, composition, *mixture_tail,
        )
        compile_for(
            _psrk_fugacity_numba,
            300.0, 1.0, composition, 0, *mixture_tail,
        )
        compile_for(
            _psrk_departure_enthalpy_numba,
            300.0, 1.0, composition, 0, *mixture_tail,
        )
        compile_for(
            _psrk_phi_phi_k_numba,
            300.0, 1.0, composition, 2, *mixture_tail, self.Pc,
        )
        self.compilation_complete = True

    def excess_gibbs_state(
        self,
        composition,
        temperature: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(composition, dtype=np.float64)
        return _psrk_activity_state_numba(
            x,
            float(temperature),
            self.nu,
            self.r,
            self.q,
            self.ell,
            self.subgroup_q,
            self.interaction_a,
            self.interaction_b,
            self.interaction_c,
        )

    def mixture_parameters(self, temperature: float, composition):
        x = np.asarray(composition, dtype=np.float64)
        return _psrk_mixture_parameters_numba(
            float(temperature),
            x,
            self.nu,
            self.r,
            self.q,
            self.ell,
            self.subgroup_q,
            self.interaction_a,
            self.interaction_b,
            self.interaction_c,
            self.Tc,
            self.omega,
            self.uses_mathias_copeman,
            self.c1,
            self.c2,
            self.c3,
            self.a0,
            self.pure_b,
            self.q1,
            self.gas_constant,
        )

    def compressibility_roots(
        self,
        temperature: float,
        pressure: float,
        composition,
    ) -> np.ndarray:
        x = np.asarray(composition, dtype=np.float64)
        roots, count = _psrk_roots_numba(
            float(temperature),
            float(pressure),
            x,
            self.nu,
            self.r,
            self.q,
            self.ell,
            self.subgroup_q,
            self.interaction_a,
            self.interaction_b,
            self.interaction_c,
            self.Tc,
            self.omega,
            self.uses_mathias_copeman,
            self.c1,
            self.c2,
            self.c3,
            self.a0,
            self.pure_b,
            self.q1,
            self.gas_constant,
        )
        return roots[:count].copy()

    def fugacity_coefficients(
        self,
        temperature: float,
        pressure: float,
        composition,
        phase: str,
    ) -> np.ndarray:
        x = np.asarray(composition, dtype=np.float64)
        phase_id = 1 if str(phase).strip().lower() in {'vapor', 'vap', 'gas'} else 0
        return _psrk_fugacity_numba(
            float(temperature),
            float(pressure),
            x,
            phase_id,
            self.nu,
            self.r,
            self.q,
            self.ell,
            self.subgroup_q,
            self.interaction_a,
            self.interaction_b,
            self.interaction_c,
            self.Tc,
            self.omega,
            self.uses_mathias_copeman,
            self.c1,
            self.c2,
            self.c3,
            self.a0,
            self.pure_b,
            self.q1,
            self.gas_constant,
        )

    def departure_enthalpy(
        self,
        temperature: float,
        pressure: float,
        composition,
        phase: str,
    ) -> float:
        x = np.asarray(composition, dtype=np.float64)
        phase_id = 1 if str(phase).strip().lower() in {'vapor', 'vap', 'gas'} else 0
        return float(_psrk_departure_enthalpy_numba(
            float(temperature),
            float(pressure),
            x,
            phase_id,
            self.nu,
            self.r,
            self.q,
            self.ell,
            self.subgroup_q,
            self.interaction_a,
            self.interaction_b,
            self.interaction_c,
            self.Tc,
            self.omega,
            self.uses_mathias_copeman,
            self.c1,
            self.c2,
            self.c3,
            self.a0,
            self.pure_b,
            self.q1,
            self.gas_constant,
        ))

    def phi_phi_K_values(
        self,
        temperature: float,
        pressure: float,
        composition,
        max_iter: int,
    ) -> np.ndarray:
        x = np.asarray(composition, dtype=np.float64)
        return _psrk_phi_phi_k_numba(
            float(temperature),
            float(pressure),
            x,
            int(max_iter),
            self.nu,
            self.r,
            self.q,
            self.ell,
            self.subgroup_q,
            self.interaction_a,
            self.interaction_b,
            self.interaction_c,
            self.Tc,
            self.omega,
            self.uses_mathias_copeman,
            self.c1,
            self.c2,
            self.c3,
            self.a0,
            self.pure_b,
            self.q1,
            self.gas_constant,
            self.Pc,
        )


if njit is not None:

    @njit(cache=True)
    def _interaction_matrices_numba(T, a, b, c):
        n_groups = a.shape[0]
        psi = np.empty((n_groups, n_groups), dtype=np.float64)
        dpsi = np.empty((n_groups, n_groups), dtype=np.float64)
        for source in range(n_groups):
            for target in range(n_groups):
                if source == target:
                    psi[source, target] = 1.0
                    dpsi[source, target] = 0.0
                    continue
                exponent = -(
                    a[source, target]
                    + b[source, target] * T
                    + c[source, target] * T * T
                ) / T
                factor = math.exp(exponent)
                psi[source, target] = factor
                dpsi[source, target] = factor * (
                    a[source, target] / (T * T) - c[source, target]
                )
        return psi, dpsi


    @njit(cache=True)
    def _group_state_numba(amounts, subgroup_q, psi, dpsi):
        n_groups = amounts.shape[0]
        fractions = np.zeros(n_groups, dtype=np.float64)
        theta = np.zeros(n_groups, dtype=np.float64)
        active = np.zeros(n_groups, dtype=np.bool_)
        total = 0.0
        for group in range(n_groups):
            if amounts[group] > 0.0:
                active[group] = True
                total += amounts[group]

        for group in range(n_groups):
            if active[group]:
                fractions[group] = amounts[group] / total
        surface_total = 0.0
        for group in range(n_groups):
            if active[group]:
                surface_total += subgroup_q[group] * fractions[group]
        for group in range(n_groups):
            if active[group]:
                theta[group] = subgroup_q[group] * fractions[group] / surface_total

        denominator = np.ones(n_groups, dtype=np.float64)
        denominator_derivative = np.zeros(n_groups, dtype=np.float64)
        for target in range(n_groups):
            if not active[target]:
                continue
            value = 0.0
            derivative = 0.0
            for source in range(n_groups):
                if active[source]:
                    value += theta[source] * psi[source, target]
                    derivative += theta[source] * dpsi[source, target]
            denominator[target] = value
            denominator_derivative[target] = derivative

        values = np.zeros(n_groups, dtype=np.float64)
        derivatives = np.zeros(n_groups, dtype=np.float64)
        for target in range(n_groups):
            if not active[target]:
                continue
            term2 = 0.0
            derivative_term2 = 0.0
            for source in range(n_groups):
                if not active[source]:
                    continue
                denom = denominator[source]
                term2 += theta[source] * psi[target, source] / denom
                derivative_term2 += theta[source] * (
                    dpsi[target, source] / denom
                    - psi[target, source]
                    * denominator_derivative[source]
                    / (denom * denom)
                )
            values[target] = subgroup_q[target] * (
                1.0 - math.log(denominator[target]) - term2
            )
            derivatives[target] = subgroup_q[target] * (
                -denominator_derivative[target] / denominator[target]
                - derivative_term2
            )
        return values, derivatives


    @njit(cache=True)
    def _psrk_activity_state_numba(
        x, T, nu, r, q, ell, subgroup_q,
        interaction_a, interaction_b, interaction_c,
    ):
        n_components = x.shape[0]
        n_groups = nu.shape[1]
        psi, dpsi = _interaction_matrices_numba(
            T, interaction_a, interaction_b, interaction_c
        )

        sum_xr = 0.0
        sum_xq = 0.0
        sum_x_ell = 0.0
        for component in range(n_components):
            sum_xr += x[component] * r[component]
            sum_xq += x[component] * q[component]
            sum_x_ell += x[component] * ell[component]

        mixture_amounts = np.zeros(n_groups, dtype=np.float64)
        for component in range(n_components):
            for group in range(n_groups):
                mixture_amounts[group] += x[component] * nu[component, group]
        mixture_values, mixture_derivatives = _group_state_numba(
            mixture_amounts, subgroup_q, psi, dpsi
        )

        ln_gamma = np.empty(n_components, dtype=np.float64)
        dln_gamma = np.empty(n_components, dtype=np.float64)
        for component in range(n_components):
            phi_over_x = r[component] / sum_xr
            theta_over_phi = (
                q[component] * sum_xr / (r[component] * sum_xq)
            )
            combinatorial = (
                math.log(phi_over_x)
                + 5.0 * q[component] * math.log(theta_over_phi)
                + ell[component]
                - phi_over_x * sum_x_ell
            )
            pure_values, pure_derivatives = _group_state_numba(
                nu[component], subgroup_q, psi, dpsi
            )
            residual = 0.0
            residual_derivative = 0.0
            for group in range(n_groups):
                count = nu[component, group]
                if count == 0.0:
                    continue
                residual += count * (
                    mixture_values[group] - pure_values[group]
                )
                residual_derivative += count * (
                    mixture_derivatives[group] - pure_derivatives[group]
                )
            ln_gamma[component] = combinatorial + residual
            dln_gamma[component] = residual_derivative
        return ln_gamma, dln_gamma


    @njit(cache=True)
    def _psrk_mixture_parameters_numba(
        T, x, nu, r, q, ell, subgroup_q,
        interaction_a, interaction_b, interaction_c,
        Tc, omega, uses_mc, c1, c2, c3, a0, pure_b, q1, gas_constant,
    ):
        n_components = x.shape[0]
        ln_gamma, dln_gamma = _psrk_activity_state_numba(
            x, T, nu, r, q, ell, subgroup_q,
            interaction_a, interaction_b, interaction_c,
        )
        pure_a = np.empty(n_components, dtype=np.float64)
        pure_d = np.empty(n_components, dtype=np.float64)
        pure_dd_dT = np.empty(n_components, dtype=np.float64)
        for component in range(n_components):
            reduced_temperature = T / Tc[component]
            sqrt_reduced_temperature = math.sqrt(reduced_temperature)
            theta = 1.0 - sqrt_reduced_temperature
            dtheta_dT = -sqrt_reduced_temperature / (2.0 * T)
            if not uses_mc[component]:
                coefficient = (
                    0.48
                    + 1.574 * omega[component]
                    - 0.176 * omega[component] * omega[component]
                )
                factor = 1.0 + coefficient * theta
                dfactor_dtheta = coefficient
            elif reduced_temperature <= 1.0:
                factor = (
                    1.0
                    + c1[component] * theta
                    + c2[component] * theta * theta
                    + c3[component] * theta * theta * theta
                )
                dfactor_dtheta = (
                    c1[component]
                    + 2.0 * c2[component] * theta
                    + 3.0 * c3[component] * theta * theta
                )
            else:
                factor = 1.0 + c1[component] * theta
                dfactor_dtheta = c1[component]
            alpha = factor * factor
            dalpha_dT = 2.0 * factor * dfactor_dtheta * dtheta_dT
            pure_a[component] = a0[component] * alpha
            da_dT = a0[component] * dalpha_dT
            pure_d[component] = (
                pure_a[component]
                / (pure_b[component] * gas_constant * T)
            )
            pure_dd_dT[component] = (
                da_dT / (pure_b[component] * gas_constant * T)
                - pure_d[component] / T
            )

        b_mix = 0.0
        g_over_rt = 0.0
        dg_over_rt_dT = 0.0
        D = 0.0
        dD_dT = 0.0
        for component in range(n_components):
            b_mix += x[component] * pure_b[component]
            g_over_rt += x[component] * ln_gamma[component]
            dg_over_rt_dT += x[component] * dln_gamma[component]
            D += x[component] * pure_d[component]
            dD_dT += x[component] * pure_dd_dT[component]

        size_term = 0.0
        for component in range(n_components):
            size_term += x[component] * math.log(
                b_mix / pure_b[component]
            )
        D += (g_over_rt + size_term) / q1
        dD_dT += dg_over_rt_dT / q1

        sigma = np.empty(n_components, dtype=np.float64)
        for component in range(n_components):
            sigma[component] = pure_d[component] + (
                ln_gamma[component]
                + math.log(b_mix / pure_b[component])
                + pure_b[component] / b_mix
                - 1.0
            ) / q1
        return pure_a, pure_d, b_mix, D, dD_dT, sigma


    @njit(cache=True)
    def _cbrt_numba(value):
        if value < 0.0:
            return -(abs(value) ** (1.0 / 3.0))
        return value ** (1.0 / 3.0)


    @njit(cache=True)
    def _compressibility_roots_numba(A, B):
        # z^3 - z^2 + (A-B-B^2)z - AB = 0
        a = -1.0
        b = A - B - B * B
        c = -A * B
        p = b - a * a / 3.0
        q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c
        shift = a / 3.0
        discriminant = (0.5 * q) ** 2 + (p / 3.0) ** 3
        candidates = np.zeros(3, dtype=np.float64)
        candidate_count = 0
        tolerance = 1.0e-14
        if discriminant > tolerance:
            candidates[0] = (
                _cbrt_numba(-0.5 * q + math.sqrt(discriminant))
                + _cbrt_numba(-0.5 * q - math.sqrt(discriminant))
                - shift
            )
            candidate_count = 1
        elif abs(p) < tolerance:
            candidates[0] = -shift
            candidate_count = 1
        else:
            argument = (3.0 * q / (2.0 * p)) * math.sqrt(-3.0 / p)
            if argument < -1.0:
                argument = -1.0
            elif argument > 1.0:
                argument = 1.0
            theta = math.acos(argument)
            radius = 2.0 * math.sqrt(-p / 3.0)
            for index in range(3):
                candidates[index] = (
                    radius
                    * math.cos((theta + 2.0 * math.pi * index) / 3.0)
                    - shift
                )
            candidate_count = 3

        roots = np.zeros(3, dtype=np.float64)
        count = 0
        for index in range(candidate_count):
            value = candidates[index]
            if value > B + 1.0e-12:
                insert = count
                while insert > 0 and roots[insert - 1] > value:
                    roots[insert] = roots[insert - 1]
                    insert -= 1
                roots[insert] = value
                count += 1
        return roots, count


    @njit(cache=True)
    def _psrk_roots_numba(
        T, P, x, nu, r, q, ell, subgroup_q,
        interaction_a, interaction_b, interaction_c,
        Tc, omega, uses_mc, c1, c2, c3, a0, pure_b, q1, gas_constant,
    ):
        _, _, b_mix, D, _, _ = _psrk_mixture_parameters_numba(
            T, x, nu, r, q, ell, subgroup_q,
            interaction_a, interaction_b, interaction_c,
            Tc, omega, uses_mc, c1, c2, c3, a0, pure_b, q1, gas_constant,
        )
        B = b_mix * P / (gas_constant * T)
        return _compressibility_roots_numba(D * B, B)


    @njit(cache=True)
    def _psrk_fugacity_numba(
        T, P, x, phase_id, nu, r, q, ell, subgroup_q,
        interaction_a, interaction_b, interaction_c,
        Tc, omega, uses_mc, c1, c2, c3, a0, pure_b, q1, gas_constant,
    ):
        _, _, b_mix, D, _, sigma = _psrk_mixture_parameters_numba(
            T, x, nu, r, q, ell, subgroup_q,
            interaction_a, interaction_b, interaction_c,
            Tc, omega, uses_mc, c1, c2, c3, a0, pure_b, q1, gas_constant,
        )
        B = b_mix * P / (gas_constant * T)
        roots, count = _compressibility_roots_numba(D * B, B)
        Z = roots[count - 1] if phase_id == 1 else roots[0]
        attraction_log = math.log1p(B / Z)
        phi = np.empty(x.shape[0], dtype=np.float64)
        for component in range(x.shape[0]):
            ln_phi = (
                (pure_b[component] / b_mix) * (Z - 1.0)
                - math.log(Z - B)
                - sigma[component] * attraction_log
            )
            phi[component] = math.exp(ln_phi)
        return phi


    @njit(cache=True)
    def _psrk_departure_enthalpy_numba(
        T, P, x, phase_id, nu, r, q, ell, subgroup_q,
        interaction_a, interaction_b, interaction_c,
        Tc, omega, uses_mc, c1, c2, c3, a0, pure_b, q1, gas_constant,
    ):
        _, _, b_mix, D, dD_dT, _ = _psrk_mixture_parameters_numba(
            T, x, nu, r, q, ell, subgroup_q,
            interaction_a, interaction_b, interaction_c,
            Tc, omega, uses_mc, c1, c2, c3, a0, pure_b, q1, gas_constant,
        )
        B = b_mix * P / (gas_constant * T)
        roots, count = _compressibility_roots_numba(D * B, B)
        Z = roots[count - 1] if phase_id == 1 else roots[0]
        attraction_log = math.log1p(B / Z)
        value = gas_constant * T * (Z - 1.0)
        value += gas_constant * T * T * dD_dT * attraction_log
        return 0.1 * value


    @njit(cache=True)
    def _normalize_numba(values):
        result = np.empty(values.shape[0], dtype=np.float64)
        total = 0.0
        for index in range(values.shape[0]):
            if values[index] > 0.0:
                total += values[index]
        for index in range(values.shape[0]):
            value = values[index]
            result[index] = value / total if value > 0.0 else 0.0
        return result


    @njit(cache=True)
    def _psrk_phi_phi_k_numba(
        T, P, x, max_iter, nu, r, q, ell, subgroup_q,
        interaction_a, interaction_b, interaction_c,
        Tc, omega, uses_mc, c1, c2, c3, a0, pure_b, q1, gas_constant, Pc,
    ):
        n_components = x.shape[0]
        K = np.empty(n_components, dtype=np.float64)
        for component in range(n_components):
            exponent = 5.373 * (1.0 + omega[component]) * (
                1.0 - Tc[component] / T
            )
            value = Pc[component] / P * math.exp(exponent)
            if value < 1.0e-8:
                value = 1.0e-8
            elif value > 1.0e8:
                value = 1.0e8
            K[component] = value

        roots, count = _psrk_roots_numba(
            T, P, x, nu, r, q, ell, subgroup_q,
            interaction_a, interaction_b, interaction_c,
            Tc, omega, uses_mc, c1, c2, c3, a0, pure_b, q1, gas_constant,
        )
        if count < 2:
            return K
        phi_liquid = _psrk_fugacity_numba(
            T, P, x, 0, nu, r, q, ell, subgroup_q,
            interaction_a, interaction_b, interaction_c,
            Tc, omega, uses_mc, c1, c2, c3, a0, pure_b, q1, gas_constant,
        )
        y_unnormalized = np.empty(n_components, dtype=np.float64)
        for component in range(n_components):
            y_unnormalized[component] = x[component] * K[component]
        y = _normalize_numba(y_unnormalized)

        for _ in range(max_iter):
            roots, count = _psrk_roots_numba(
                T, P, y, nu, r, q, ell, subgroup_q,
                interaction_a, interaction_b, interaction_c,
                Tc, omega, uses_mc, c1, c2, c3, a0, pure_b, q1, gas_constant,
            )
            if count < 2:
                return K
            phi_vapor = _psrk_fugacity_numba(
                T, P, y, 1, nu, r, q, ell, subgroup_q,
                interaction_a, interaction_b, interaction_c,
                Tc, omega, uses_mc, c1, c2, c3, a0, pure_b, q1, gas_constant,
            )
            K_new = np.empty(n_components, dtype=np.float64)
            for component in range(n_components):
                value = phi_liquid[component] / phi_vapor[component]
                if value < 1.0e-8:
                    value = 1.0e-8
                elif value > 1.0e8:
                    value = 1.0e8
                K_new[component] = value
                y_unnormalized[component] = x[component] * value
            new_y = _normalize_numba(y_unnormalized)
            maximum_change = 0.0
            for component in range(n_components):
                change = abs(new_y[component] - y[component])
                if change > maximum_change:
                    maximum_change = change
            if maximum_change < 1.0e-9:
                return K_new
            for component in range(n_components):
                K[component] = 0.5 * K[component] + 0.5 * K_new[component]
                y[component] = new_y[component]
        return K

else:

    def _psrk_activity_state_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Numba is not available")

    def _psrk_mixture_parameters_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Numba is not available")

    def _psrk_roots_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Numba is not available")

    def _psrk_fugacity_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Numba is not available")

    def _psrk_departure_enthalpy_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Numba is not available")

    def _psrk_phi_phi_k_numba(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Numba is not available")
