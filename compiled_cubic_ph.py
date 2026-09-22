"""Compiled homogeneous pressure-enthalpy solves for cubic EOS packages."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .compiled_cubic_eos import (
        CompiledCubicEOSBackend,
        _cubic_departure_enthalpy_numba,
        _interaction_parameters_numba,
    )
    from .property_resolution.ideal_gas_cp import (
        AffineIdealGasCpKernel,
        ChebyshevCpKernel,
        PiecewiseIdealGasCpKernel,
        PolynomialCpKernel,
        ShomateCpKernel,
    )
    from .thermodynamics_models.common import T_REF
else:
    from compiled_cubic_eos import (
        CompiledCubicEOSBackend,
        _cubic_departure_enthalpy_numba,
        _interaction_parameters_numba,
    )
    from property_resolution.ideal_gas_cp import (
        AffineIdealGasCpKernel,
        ChebyshevCpKernel,
        PiecewiseIdealGasCpKernel,
        PolynomialCpKernel,
        ShomateCpKernel,
    )
    from thermodynamics_models.common import T_REF

try:
    from numba import njit, typeof
except Exception:  # pragma: no cover - optional dependency fallback
    njit = None
    typeof = None


CP_KIND_POLYNOMIAL = 1
CP_KIND_SHOMATE = 2
CP_KIND_CHEBYSHEV = 3


def _flatten_analytic_kernel(kernel, intercept=0.0, scale_factor=1.0):
    if isinstance(kernel, AffineIdealGasCpKernel):
        return _flatten_analytic_kernel(
            kernel.base_kernel,
            intercept + scale_factor * kernel.intercept,
            scale_factor * kernel.scale_factor,
        )
    if isinstance(kernel, PiecewiseIdealGasCpKernel):
        records = []
        for index, segment in enumerate(kernel.segments):
            lower = -math.inf if index == 0 else segment.Tmin
            upper = (
                math.inf if index == len(kernel.segments) - 1 else segment.Tmax
            )
            children = _flatten_analytic_kernel(
                segment,
                intercept,
                scale_factor,
            )
            for child in children:
                child['Tmin'] = max(lower, child['Tmin'])
                child['Tmax'] = min(upper, child['Tmax'])
                if child['Tmin'] < child['Tmax']:
                    records.append(child)
        return records
    if isinstance(kernel, PolynomialCpKernel):
        kind = CP_KIND_POLYNOMIAL
    elif isinstance(kernel, ShomateCpKernel):
        kind = CP_KIND_SHOMATE
    elif isinstance(kernel, ChebyshevCpKernel):
        kind = CP_KIND_CHEBYSHEV
    else:
        raise TypeError(
            f'unsupported analytic ideal-gas Cp kernel {type(kernel).__name__}'
        )
    return [{
        'kind': kind,
        # Native leaf curves extrapolate without conditioning. Their enclosing
        # piecewise kernels select intervals; only the outer kernel clamps T.
        'Tmin': -math.inf,
        'Tmax': math.inf,
        'intercept': float(intercept),
        'scale_factor': float(scale_factor),
        'coefficients': tuple(float(value) for value in kernel.coefficients),
        'center': float(getattr(kernel, 'center', 0.0)),
        'scale': float(getattr(kernel, 'scale', 0.0)),
        'h_polynomial': tuple(
            float(value) for value in getattr(kernel, 'h_polynomial', ())
        ),
        'h_logarithmic': float(
            getattr(kernel, 'h_log_coefficient', 0.0)
        ),
        'h_reciprocal': float(
            getattr(kernel, 'h_reciprocal_coefficient', 0.0)
        ),
    }]


@dataclass
class CompiledCubicPHBackend:
    """Compact ideal-gas plus EOS-departure evaluator and PH solver."""

    components: tuple[str, ...]
    formation_enthalpy: np.ndarray
    lower_temperature: np.ndarray
    upper_temperature: np.ndarray
    component_segment_offset: np.ndarray
    component_segment_count: np.ndarray
    segment_kind: np.ndarray
    segment_lower: np.ndarray
    segment_upper: np.ndarray
    segment_intercept: np.ndarray
    segment_scale_factor: np.ndarray
    segment_coefficients: np.ndarray
    segment_coefficient_count: np.ndarray
    segment_centers: np.ndarray
    segment_scales: np.ndarray
    segment_h_polynomials: np.ndarray
    segment_h_polynomial_count: np.ndarray
    segment_h_logarithmic: np.ndarray
    segment_h_reciprocal: np.ndarray
    cubic_backend: CompiledCubicEOSBackend
    compilation_complete: bool = False

    @classmethod
    def from_thermo(cls, thermo) -> "CompiledCubicPHBackend | None":
        if njit is None or not isinstance(
            thermo.cubic._compiled_backend,
            CompiledCubicEOSBackend,
        ):
            return None
        kernels = [
            thermo._ideal_gas_cp_kernel(component)
            for component in thermo.components
        ]
        try:
            component_records = [
                _flatten_analytic_kernel(kernel) for kernel in kernels
            ]
        except TypeError:
            return None
        records = []
        offsets = []
        counts = []
        for kernel, items in zip(kernels, component_records):
            items = sorted(items, key=lambda item: (item['Tmin'], item['Tmax']))
            clipped = []
            lower = float(kernel.extended_Tmin)
            upper = float(kernel.extended_Tmax)
            for item in items:
                record = dict(item)
                record['Tmin'] = max(lower, record['Tmin'])
                record['Tmax'] = min(upper, record['Tmax'])
                if record['Tmax'] >= record['Tmin']:
                    clipped.append(record)
            if not clipped:
                return None
            clipped[0]['Tmin'] = lower
            clipped[-1]['Tmax'] = upper
            offsets.append(len(records))
            counts.append(len(clipped))
            records.extend(clipped)
        coefficient_width = max(len(item['coefficients']) for item in records)
        polynomial_width = max(
            1,
            max(len(item['h_polynomial']) for item in records),
        )
        coefficients = np.zeros((len(records), coefficient_width))
        h_polynomials = np.zeros((len(records), polynomial_width))
        for index, item in enumerate(records):
            coefficients[index, :len(item['coefficients'])] = (
                item['coefficients']
            )
            h_polynomials[index, :len(item['h_polynomial'])] = (
                item['h_polynomial']
            )
        backend = cls(
            components=tuple(thermo.components),
            formation_enthalpy=np.asarray([
                float(thermo.props[component].Hf or 0.0)
                for component in thermo.components
            ]),
            lower_temperature=np.asarray([
                kernel.extended_Tmin for kernel in kernels
            ]),
            upper_temperature=np.asarray([
                kernel.extended_Tmax for kernel in kernels
            ]),
            component_segment_offset=np.asarray(offsets, dtype=np.int64),
            component_segment_count=np.asarray(counts, dtype=np.int64),
            segment_kind=np.asarray([
                item['kind'] for item in records
            ], dtype=np.int64),
            segment_lower=np.asarray([item['Tmin'] for item in records]),
            segment_upper=np.asarray([item['Tmax'] for item in records]),
            segment_intercept=np.asarray([
                item['intercept'] for item in records
            ]),
            segment_scale_factor=np.asarray([
                item['scale_factor'] for item in records
            ]),
            segment_coefficients=coefficients,
            segment_coefficient_count=np.asarray([
                len(item['coefficients']) for item in records
            ], dtype=np.int64),
            segment_centers=np.asarray([item['center'] for item in records]),
            segment_scales=np.asarray([item['scale'] for item in records]),
            segment_h_polynomials=h_polynomials,
            segment_h_polynomial_count=np.asarray([
                len(item['h_polynomial']) for item in records
            ], dtype=np.int64),
            segment_h_logarithmic=np.asarray([
                item['h_logarithmic'] for item in records
            ]),
            segment_h_reciprocal=np.asarray([
                item['h_reciprocal'] for item in records
            ]),
            cubic_backend=thermo.cubic._compiled_backend,
        )
        backend.compile_kernels()
        return backend

    def _cp_state(self):
        return (
            self.formation_enthalpy,
            self.lower_temperature,
            self.upper_temperature,
            self.component_segment_offset,
            self.component_segment_count,
            self.segment_kind,
            self.segment_lower,
            self.segment_upper,
            self.segment_intercept,
            self.segment_scale_factor,
            self.segment_coefficients,
            self.segment_coefficient_count,
            self.segment_centers,
            self.segment_scales,
            self.segment_h_polynomials,
            self.segment_h_polynomial_count,
            self.segment_h_logarithmic,
            self.segment_h_reciprocal,
        )

    def _cubic_state(self):
        backend = self.cubic_backend
        return (
            backend.Tc,
            backend.omega,
            backend.a0,
            backend.pure_b,
            backend.alpha_mode,
            backend.c1,
            backend.c2,
            backend.c3,
            backend.delta1,
            backend.delta2,
            backend.interaction_plan,
        )

    def compile_kernels(self) -> None:
        if self.compilation_complete or typeof is None:
            return
        composition = np.full(
            len(self.components),
            1.0 / len(self.components),
            dtype=np.float64,
        )
        arguments = (
            600.0,
            100.0,
            composition,
            1,
            self._cp_state(),
            self._cubic_state(),
        )
        _compact_enthalpy_cp_numba.compile(tuple(
            typeof(argument) for argument in arguments
        ))
        solve_arguments = (
            100.0,
            0.0,
            composition,
            600.0,
            1,
            self._cp_state(),
            self._cubic_state(),
        )
        _direct_ph_numba.compile(tuple(
            typeof(argument) for argument in solve_arguments
        ))
        self.compilation_complete = True

    @staticmethod
    def _composition(values):
        array = np.asarray(values, dtype=np.float64)
        total = float(np.sum(array))
        if total <= 0.0:
            return np.full_like(array, 1.0 / len(array))
        return np.maximum(array, 0.0) / total

    def enthalpy_cp(self, T, P, composition, phase):
        phase_id = 1 if str(phase).strip().lower() == 'vapor' else 0
        return _compact_enthalpy_cp_numba(
            float(T),
            float(P),
            self._composition(composition),
            phase_id,
            self._cp_state(),
            self._cubic_state(),
        )

    def solve_temperature(self, P, H, composition, T_guess, phase):
        phase_id = 1 if str(phase).strip().lower() == 'vapor' else 0
        return _direct_ph_numba(
            float(P),
            float(H),
            self._composition(composition),
            float(T_guess),
            phase_id,
            self._cp_state(),
            self._cubic_state(),
        )


if njit is not None:

    @njit(cache=True)
    def _polynomial_difference_numba(coefficients, count, first, second):
        if count <= 1 or first == second:
            return 0.0
        quotient = coefficients[count - 1]
        value = quotient
        for index in range(count - 2, 0, -1):
            quotient = coefficients[index] + first * quotient
            value = quotient + second * value
        return (second - first) * value


    @njit(cache=True)
    def _native_chebyshev_cp_numba(T, center, scale, coefficients, count):
        mapped = (T - center) / (scale * (T + center))
        first = 0.0
        second = 0.0
        for index in range(count - 1, 0, -1):
            current = 2.0 * mapped * first - second + coefficients[index]
            second = first
            first = current
        return mapped * first - second + coefficients[0]


    @njit(cache=True)
    def _native_chebyshev_delta_h_numba(
        first_temperature,
        second_temperature,
        center,
        scale,
        polynomial,
        polynomial_count,
        logarithmic,
        reciprocal,
    ):
        first = (
            (first_temperature - center)
            / (scale * (first_temperature + center))
        )
        second = (
            (second_temperature - center)
            / (scale * (second_temperature + center))
        )
        delta = second - first
        return (
            _polynomial_difference_numba(
                polynomial,
                polynomial_count,
                first,
                second,
            )
            + logarithmic
            * math.log1p((-scale * delta) / (1.0 - scale * first))
            + reciprocal
            * scale
            * delta
            / ((1.0 - scale * second) * (1.0 - scale * first))
        )


    @njit(cache=True)
    def _base_cp_numba(kind, T, coefficients, count, center, scale):
        if kind == CP_KIND_POLYNOMIAL:
            value = 0.0
            for index in range(count - 1, -1, -1):
                value = value * T + coefficients[index]
            return value
        if kind == CP_KIND_SHOMATE:
            reduced = T / 1000.0
            return (
                coefficients[0]
                + reduced * (
                    coefficients[1]
                    + reduced * (
                        coefficients[2] + reduced * coefficients[3]
                    )
                )
                + coefficients[4] / (reduced * reduced)
            )
        return _native_chebyshev_cp_numba(
            T,
            center,
            scale,
            coefficients,
            count,
        )


    @njit(cache=True)
    def _base_delta_h_numba(
        kind,
        first,
        second,
        coefficients,
        count,
        center,
        scale,
        polynomial,
        polynomial_count,
        logarithmic,
        reciprocal,
    ):
        if kind == CP_KIND_POLYNOMIAL:
            total = 0.0
            for power in range(count):
                exponent = power + 1
                total += coefficients[power] * (
                    second**exponent - first**exponent
                ) / exponent
            return total
        if kind == CP_KIND_SHOMATE:
            first_reduced = first / 1000.0
            second_reduced = second / 1000.0
            return 1000.0 * (
                coefficients[0] * (second_reduced - first_reduced)
                + coefficients[1]
                * (second_reduced**2 - first_reduced**2) / 2.0
                + coefficients[2]
                * (second_reduced**3 - first_reduced**3) / 3.0
                + coefficients[3]
                * (second_reduced**4 - first_reduced**4) / 4.0
                - coefficients[4]
                * (1.0 / second_reduced - 1.0 / first_reduced)
            )
        return _native_chebyshev_delta_h_numba(
            first,
            second,
            center,
            scale,
            polynomial,
            polynomial_count,
            logarithmic,
            reciprocal,
        )


    @njit(cache=True)
    def _segment_cp_numba(segment, T, state):
        (
            _formation_enthalpy, _lower_temperature, _upper_temperature,
            _component_segment_offset, _component_segment_count,
            segment_kind, _segment_lower, _segment_upper,
            segment_intercept, segment_scale_factor,
            segment_coefficients, segment_coefficient_count,
            segment_centers, segment_scales,
            _segment_h_polynomials, _segment_h_polynomial_count,
            _segment_h_logarithmic, _segment_h_reciprocal,
        ) = state
        return (
            segment_intercept[segment]
            + segment_scale_factor[segment]
            * _base_cp_numba(
                segment_kind[segment],
                T,
                segment_coefficients[segment],
                segment_coefficient_count[segment],
                segment_centers[segment],
                segment_scales[segment],
            )
        )


    @njit(cache=True)
    def _segment_delta_h_numba(segment, first, second, state):
        (
            _formation_enthalpy, _lower_temperature, _upper_temperature,
            _component_segment_offset, _component_segment_count,
            segment_kind, _segment_lower, _segment_upper,
            segment_intercept, segment_scale_factor,
            segment_coefficients, segment_coefficient_count,
            segment_centers, segment_scales,
            segment_h_polynomials, segment_h_polynomial_count,
            segment_h_logarithmic, segment_h_reciprocal,
        ) = state
        return (
            segment_intercept[segment] * (second - first)
            + segment_scale_factor[segment]
            * _base_delta_h_numba(
                segment_kind[segment],
                first,
                second,
                segment_coefficients[segment],
                segment_coefficient_count[segment],
                segment_centers[segment],
                segment_scales[segment],
                segment_h_polynomials[segment],
                segment_h_polynomial_count[segment],
                segment_h_logarithmic[segment],
                segment_h_reciprocal[segment],
            )
        )


    @njit(cache=True)
    def _component_native_cp_numba(component, T, state):
        offset = state[3][component]
        count = state[4][component]
        selected = offset
        for local_index in range(count):
            segment = offset + local_index
            if state[6][segment] <= T <= state[7][segment]:
                selected = segment
                break
            if T > state[7][segment]:
                selected = segment
        return _segment_cp_numba(selected, T, state)


    @njit(cache=True)
    def _component_native_delta_h_numba(component, first, second, state):
        offset = state[3][component]
        count = state[4][component]
        total = 0.0
        for local_index in range(count):
            segment = offset + local_index
            left = max(first, state[6][segment])
            right = min(second, state[7][segment])
            if right > left:
                total += _segment_delta_h_numba(
                    segment,
                    left,
                    right,
                    state,
                )
        return total


    @njit(cache=True)
    def _component_conditioned_delta_h_numba(
        component,
        first,
        second,
        state,
    ):
        if first == second:
            return 0.0
        sign = 1.0
        if second < first:
            first, second = second, first
            sign = -1.0
        lower = state[1][component]
        upper = state[2][component]
        total = 0.0
        cursor = first
        if cursor < lower:
            end = min(second, lower)
            total += _component_native_cp_numba(
                component,
                lower,
                state,
            ) * (end - cursor)
            cursor = end
        if cursor < second and cursor < upper:
            end = min(second, upper)
            total += _component_native_delta_h_numba(
                component,
                cursor,
                end,
                state,
            )
            cursor = end
        if cursor < second:
            total += _component_native_cp_numba(
                component,
                upper,
                state,
            ) * (second - cursor)
        return sign * total


    @njit(cache=True)
    def _ideal_enthalpy_cp_numba(T, composition, state):
        (
            formation_enthalpy,
            lower_temperature,
            upper_temperature,
            _component_segment_offset,
            _component_segment_count,
            _segment_kind,
            _segment_lower,
            _segment_upper,
            _segment_intercept,
            _segment_scale_factor,
            _segment_coefficients,
            _segment_coefficient_count,
            _segment_centers,
            _segment_scales,
            _segment_h_polynomials,
            _segment_h_polynomial_count,
            _segment_h_logarithmic,
            _segment_h_reciprocal,
        ) = state
        enthalpy = 0.0
        heat_capacity = 0.0
        for component in range(composition.shape[0]):
            evaluation_temperature = min(
                upper_temperature[component],
                max(lower_temperature[component], T),
            )
            cp_value = _component_native_cp_numba(
                component,
                evaluation_temperature,
                state,
            )
            delta_h = _component_conditioned_delta_h_numba(
                component,
                T_REF,
                T,
                state,
            )
            fraction = composition[component]
            enthalpy += fraction * (
                1000.0 * formation_enthalpy[component] + delta_h
            )
            heat_capacity += fraction * cp_value
        return enthalpy, heat_capacity


    @njit(cache=True)
    def _departure_enthalpy_numba(T, P, composition, phase_id, cubic_state):
        (
            Tc, omega, a0, pure_b, modes, c1, c2, c3,
            delta1, delta2, interaction_plan,
        ) = cubic_state
        kij, dkij = _interaction_parameters_numba(T, interaction_plan)
        return _cubic_departure_enthalpy_numba(
            T, P, composition, phase_id, kij, dkij,
            Tc, omega, a0, pure_b, modes, c1, c2, c3,
            delta1, delta2,
        )


    @njit(cache=True)
    def _compact_enthalpy_cp_numba(
        T,
        P,
        composition,
        phase_id,
        cp_state,
        cubic_state,
    ):
        ideal_enthalpy, ideal_cp = _ideal_enthalpy_cp_numba(
            T,
            composition,
            cp_state,
        )
        departure = _departure_enthalpy_numba(
            T,
            P,
            composition,
            phase_id,
            cubic_state,
        )
        delta = max(0.05, 1.0e-4 * T)
        low_temperature = max(1.0, T - delta)
        high_temperature = T + delta
        low = _departure_enthalpy_numba(
            low_temperature,
            P,
            composition,
            phase_id,
            cubic_state,
        )
        high = _departure_enthalpy_numba(
            high_temperature,
            P,
            composition,
            phase_id,
            cubic_state,
        )
        departure_cp = (high - low) / (high_temperature - low_temperature)
        return ideal_enthalpy + departure, ideal_cp + departure_cp


    @njit(cache=True)
    def _direct_ph_numba(
        P,
        target_enthalpy,
        composition,
        temperature_guess,
        phase_id,
        cp_state,
        cubic_state,
    ):
        tolerance = max(1.0e-8, abs(target_enthalpy) * 1.0e-12)
        temperature = max(1.0, min(5000.0, temperature_guess))
        evaluations = 0
        for _iteration in range(12):
            enthalpy, heat_capacity = _compact_enthalpy_cp_numba(
                temperature,
                P,
                composition,
                phase_id,
                cp_state,
                cubic_state,
            )
            evaluations += 1
            residual = enthalpy - target_enthalpy
            if abs(residual) <= tolerance:
                return temperature, residual, evaluations, True
            if not math.isfinite(heat_capacity) or abs(heat_capacity) < 1.0e-12:
                return temperature, residual, evaluations, False
            maximum_step = 0.35 * max(abs(temperature), 50.0)
            step = max(
                -maximum_step,
                min(maximum_step, residual / heat_capacity),
            )
            accepted = False
            damping = 1.0
            for _trial in range(8):
                candidate = max(
                    1.0,
                    min(5000.0, temperature - damping * step),
                )
                candidate_enthalpy, _ = _compact_enthalpy_cp_numba(
                    candidate,
                    P,
                    composition,
                    phase_id,
                    cp_state,
                    cubic_state,
                )
                evaluations += 1
                candidate_residual = candidate_enthalpy - target_enthalpy
                if (
                    abs(candidate_residual) <= abs(residual) * 0.9
                    or abs(candidate_residual) <= tolerance
                ):
                    temperature = candidate
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                return temperature, residual, evaluations, False
        enthalpy, _ = _compact_enthalpy_cp_numba(
            temperature,
            P,
            composition,
            phase_id,
            cp_state,
            cubic_state,
        )
        evaluations += 1
        residual = enthalpy - target_enthalpy
        return temperature, residual, evaluations, abs(residual) <= tolerance

else:

    def _unavailable(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError('Numba is not available')

    _compact_enthalpy_cp_numba = _unavailable
    _direct_ph_numba = _unavailable
