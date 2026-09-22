"""Shared compact caloric properties and homogeneous pressure-enthalpy solves."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

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
    from .property_resolution import liquid_cp
    from .physical_constants import R_J_MOL_K
    from .ph_solver import solve_caloric_temperature
    from .compiled_activity import (
        CompiledNRTLBackend, CompiledUNIQUACBackend,
        _nrtl_excess_enthalpy_numba, _uniquac_excess_enthalpy_numba,
    )
    from .compiled_unifac import CompiledUNIFACBackend, _excess_enthalpy_numba
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
    from property_resolution import liquid_cp
    from physical_constants import R_J_MOL_K
    from ph_solver import solve_caloric_temperature
    from compiled_activity import (
        CompiledNRTLBackend, CompiledUNIQUACBackend,
        _nrtl_excess_enthalpy_numba, _uniquac_excess_enthalpy_numba,
    )
    from compiled_unifac import CompiledUNIFACBackend, _excess_enthalpy_numba

try:
    from numba import njit, typeof
    from numba.extending import register_jitable
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .compiled_cache import numba_cached
    else:
        from compiled_cache import numba_cached
except Exception:  # pragma: no cover - optional dependency fallback
    njit = None
    typeof = None


CP_KIND_POLYNOMIAL = 1
CP_KIND_SHOMATE = 2
CP_KIND_CHEBYSHEV = 3
CP_KIND_LINEAR_CHEBYSHEV = 4
CP_KIND_ZABRANSKY = 5


def _flatten_analytic_kernel(kernel, intercept=0.0, scale_factor=1.0):
    if isinstance(kernel, liquid_cp.ScaledIdealGasLiquidCpKernel):
        return _flatten_analytic_kernel(
            kernel.ideal_gas_kernel, intercept, scale_factor * kernel.scale_factor,
        )
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
    if isinstance(kernel, (PolynomialCpKernel, liquid_cp.PolynomialLiquidCpKernel,
                           liquid_cp.ConstantLiquidCpKernel)):
        kind = CP_KIND_POLYNOMIAL
    elif isinstance(kernel, (ShomateCpKernel, liquid_cp.ShomateLiquidCpKernel)):
        kind = CP_KIND_SHOMATE
    elif isinstance(kernel, ChebyshevCpKernel):
        kind = CP_KIND_CHEBYSHEV
    elif isinstance(kernel, liquid_cp.LinearChebyshevLiquidCpKernel):
        kind = CP_KIND_LINEAR_CHEBYSHEV
    elif isinstance(kernel, liquid_cp.NativeZabranskyLiquidCpKernel):
        kind = CP_KIND_ZABRANSKY
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
        'coefficients': ((float(kernel.value),) if isinstance(
            kernel, liquid_cp.ConstantLiquidCpKernel
        ) else tuple(float(value) for value in kernel.coefficients)),
        'center': float(getattr(kernel, 'center', getattr(kernel, 'critical_temperature', 0.0))),
        'scale': float(getattr(kernel, 'scale', getattr(kernel, 'half_width', 0.0))),
        'h_polynomial': tuple(
            float(value) for value in getattr(kernel, 'h_polynomial', getattr(kernel, 'h_coefficients', ()))
        ),
        'h_logarithmic': float(
            getattr(kernel, 'h_log_coefficient', 0.0)
        ),
        'h_reciprocal': float(
            getattr(kernel, 'h_reciprocal_coefficient', 0.0)
        ),
    }]


@dataclass
class CompiledPHBackend:
    """Compact vapor/liquid reference curves with optional EOS departures."""

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
    cubic_backend: CompiledCubicEOSBackend | None
    compilation_complete: bool = False

    @classmethod
    def from_thermo(cls, thermo, *, phase='vapor', cubic_backend=None) -> "CompiledPHBackend | None":
        if njit is None:
            return None
        if cubic_backend is not None and not isinstance(cubic_backend, CompiledCubicEOSBackend):
            return None
        kernels = [
            (thermo._liquid_cp_kernel(component) if phase == 'liquid'
             else thermo._ideal_gas_cp_kernel(component))
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
                (thermo.enthalpy_ideal_gas(component, T_REF)
                 - thermo.Hvap_at_T(component, T_REF) if phase == 'liquid'
                 else thermo.enthalpy_ideal_gas(component, T_REF))
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
            cubic_backend=cubic_backend,
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
        if backend is None:
            return None
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

    def bind(self, P, composition, phase):
        """Bind the numeric state once for a Python-driven caloric solve."""
        arguments = (
            float(P), self._composition(composition),
            1 if phase == 'vapor' else 0, self._cp_state(), self._cubic_state(),
        )
        return lambda T: _compact_enthalpy_cp_numba(float(T), *arguments)

    def with_cubic_departure(self, cubic_backend):
        """Reuse resolved Cp arrays for a compatible gamma-phi vapor backend."""
        if not isinstance(cubic_backend, CompiledCubicEOSBackend):
            return None
        if tuple(cubic_backend.components) != self.components:
            return None
        backend = replace(self, cubic_backend=cubic_backend, compilation_complete=False)
        backend.compile_kernels()
        return backend

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


@dataclass
class CompiledActivityPHBackend:
    """A pure-liquid caloric backend fused with the existing excess kernels."""

    pure: CompiledPHBackend
    indices: np.ndarray
    activity_states: tuple

    @classmethod
    def from_backends(cls, pure, activity):
        states = [None, None, None]
        if isinstance(activity, CompiledNRTLBackend):
            index = 0
        elif isinstance(activity, CompiledUNIQUACBackend):
            index = 1
        elif isinstance(activity, CompiledUNIFACBackend):
            index = 2
        else:
            return None
        states[index] = activity.enthalpy_parameters()
        backend = cls(pure, np.asarray([
            pure.components.index(comp) for comp in activity.components
        ], dtype=np.int64), tuple(states))
        arguments = (
            0.0, np.zeros(len(pure.components)), 300.0,
            pure._cp_state(), backend.indices, backend.activity_states,
        )
        _direct_activity_ph_numba.compile(tuple(typeof(arg) for arg in arguments))
        return backend

    def solve_temperature(self, P, H, composition, T_guess, phase):
        return _direct_activity_ph_numba(
            float(H), self.pure._composition(composition), float(T_guess),
            self.pure._cp_state(), self.indices, self.activity_states,
        )


if njit is not None:

    _solve_caloric_temperature_numba = register_jitable(inline='always')(solve_caloric_temperature)
    _linear_chebyshev_value_numba = numba_cached(njit)(liquid_cp._chebyshev_value)
    _linear_chebyshev_difference_numba = numba_cached(njit)(liquid_cp._chebyshev_difference)
    _caloric_cached = numba_cached(njit, dependencies=(
        liquid_cp._chebyshev_difference, _cubic_departure_enthalpy_numba,
        _interaction_parameters_numba,
    ))

    _activity_cached = numba_cached(njit, dependencies=(
        solve_caloric_temperature, _nrtl_excess_enthalpy_numba,
        _uniquac_excess_enthalpy_numba, _excess_enthalpy_numba,
        liquid_cp._chebyshev_difference,
    ))

    @_activity_cached
    def _ph_excess_enthalpy_numba(T, composition, indices, nrtl, uniquac, unifac):
        active = composition[indices]
        fraction = np.sum(active)
        if fraction <= 0.0:
            return 0.0
        active = active / fraction
        if nrtl is not None:
            return fraction * _nrtl_excess_enthalpy_numba(active, T, *nrtl)
        if uniquac is not None:
            return fraction * _uniquac_excess_enthalpy_numba(active, T, *uniquac)
        if unifac is not None:
            delta = max(1.0e-3, 1.0e-4 * T)
            excess = _excess_enthalpy_numba(
                *unifac, active, T, max(1.0, T - delta), T + delta,
            )
            return fraction * excess
        return 0.0

    @_activity_cached
    def _activity_ph_enthalpy_cp_numba(T, composition, cp_state, indices, states):
        enthalpy, cp = _ideal_enthalpy_cp_numba(T, composition, cp_state)
        excess = _ph_excess_enthalpy_numba(T, composition, indices, *states)
        delta = max(0.25, 1.0e-3 * T)
        low_T, high_T = max(1.0, T - delta), T + delta
        low = _ph_excess_enthalpy_numba(low_T, composition, indices, *states)
        high = _ph_excess_enthalpy_numba(high_T, composition, indices, *states)
        return enthalpy + excess, cp + (high - low) / (high_T - low_T)

    @_activity_cached
    def _direct_activity_ph_numba(target, composition, seed, cp_state, indices, states):
        return _solve_caloric_temperature_numba(
            _activity_ph_enthalpy_cp_numba, (composition, cp_state, indices, states),
            target, seed,
        )

    @_caloric_cached
    def _polynomial_difference_numba(coefficients, count, first, second):
        if count <= 1 or first == second:
            return 0.0
        quotient = coefficients[count - 1]
        value = quotient
        for index in range(count - 2, 0, -1):
            quotient = coefficients[index] + first * quotient
            value = quotient + second * value
        return (second - first) * value


    @_caloric_cached
    def _native_chebyshev_cp_numba(T, center, scale, coefficients, count):
        mapped = (T - center) / (scale * (T + center))
        first = 0.0
        second = 0.0
        for index in range(count - 1, 0, -1):
            current = 2.0 * mapped * first - second + coefficients[index]
            second = first
            first = current
        return mapped * first - second + coefficients[0]


    @_caloric_cached
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


    @_caloric_cached
    def _base_cp_numba(kind, T, coefficients, count, center, scale):
        if kind == CP_KIND_LINEAR_CHEBYSHEV:
            return _linear_chebyshev_value_numba(coefficients[:count], (T - center) / scale)
        if kind == CP_KIND_ZABRANSKY:
            reduced = T / center
            gap = 1.0 - reduced
            if gap <= 0.0:
                raise ValueError('Zabransky liquid Cp is undefined at or above Tc')
            return R_J_MOL_K * (
                coefficients[0] * math.log(gap) + coefficients[1] / gap
                + coefficients[2] + coefficients[3] * reduced
                + coefficients[4] * reduced**2 + coefficients[5] * reduced**3
            )
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


    @_caloric_cached
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
        if kind == CP_KIND_LINEAR_CHEBYSHEV:
            return _linear_chebyshev_difference_numba(
                polynomial[:polynomial_count], (first - center) / scale,
                (second - center) / scale,
            )
        if kind == CP_KIND_ZABRANSKY:
            delta = second - first
            if second >= center or first >= center:
                raise ValueError('Zabransky liquid Cp is undefined at or above Tc')
            a1, a2, a3, a4, a5, a6 = coefficients[:6]
            primitive = np.array((
                0.0, a3 - a1, a4 / (2.0 * center),
                a5 / (3.0 * center**2), a6 / (4.0 * center**3),
            ))
            return R_J_MOL_K * (
                _polynomial_difference_numba(primitive, 5, first, second)
                + a1 * delta * math.log1p(-first / center)
                + (a1 * second - center * (a1 + a2))
                * math.log1p(-delta / (center - first))
            )
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


    @_caloric_cached
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


    @_caloric_cached
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


    @_caloric_cached
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


    @_caloric_cached
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


    @_caloric_cached
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


    @_caloric_cached
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


    @_caloric_cached
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


    @_caloric_cached
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
        if cubic_state is None:
            return ideal_enthalpy, ideal_cp
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


    @numba_cached(njit, dependencies=(solve_caloric_temperature, liquid_cp._chebyshev_difference,
                                     _cubic_departure_enthalpy_numba, _interaction_parameters_numba))
    def _direct_ph_numba(
        P,
        target_enthalpy,
        composition,
        temperature_guess,
        phase_id,
        cp_state,
        cubic_state,
    ):
        return _solve_caloric_temperature_numba(
            _compact_enthalpy_cp_numba,
            (P, composition, phase_id, cp_state, cubic_state),
            target_enthalpy, temperature_guess,
        )

else:

    def _unavailable(*_args, **_kwargs):  # pragma: no cover
        raise RuntimeError('Numba is not available')

    _compact_enthalpy_cp_numba = _unavailable
    _direct_ph_numba = _unavailable
    _direct_activity_ph_numba = _unavailable
