"""Probe Haber-Bosch PFR pressure-enthalpy state-solving strategies.

The script captures real homogeneous-vapor PH requests from one complete
Haber-Bosch solve, then replays a deterministic sample through progressively
more specialized implementations:

* the current full-state solver with its original inlet-temperature seed;
* the current solver seeded by the preceding converged PFR state;
* a direct scalar H/Cp Newton solve without intermediate StreamState objects;
* a compact numeric evaluator with prebound Chebyshev Cp arrays and EOS data.
* recent-state affine predictors using 1, 2, 3, 4, and 8 prior states;
* quadratic and cubic local trajectory predictors with several window sizes;
* continuously updated linear/quadratic RLS and divided-difference models;
* nearest-state predictors from the preceding recycle evaluation.

Example::

    python scripts/performance/probe_haber_bosch_ph.py \
        --samples 2000 \
        --capture-output /tmp/haber-ph-workload.json \
        --output /tmp/haber-ph-probe.json

The expensive capture can be reused later with ``--capture-input``. Production
classes are restored immediately after capture; no simulation behavior changes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from property_resolution.ideal_gas_cp import ChebyshevCpKernel
from simulator import Simulator
from thermodynamics_models.common import T_REF
from unit_operations_basic import _ThermoStateSolver
from unit_operations_reactors import KineticsPFR

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None


EXAMPLE = ROOT / 'examples' / 'haber_bosch_full.pfd'


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values):
    return {
        'minimum': min(values) if values else None,
        'median': statistics.median(values) if values else None,
        'p90': _percentile(values, 0.90),
        'p99': _percentile(values, 0.99),
        'maximum': max(values) if values else None,
        'mean': statistics.fmean(values) if values else None,
    }


def _sample_evenly(records, maximum):
    if len(records) <= maximum:
        return records
    indices = np.linspace(0, len(records) - 1, maximum, dtype=np.int64)
    return [records[int(index)] for index in indices]


def capture_workload():
    original_state_at_enthalpy = _ThermoStateSolver.state_at_enthalpy
    original_temperature_at_enthalpy = _ThermoStateSolver.temperature_at_enthalpy
    original_trial_state = _ThermoStateSolver._calculate_trial_state
    original_pfr_solve = KineticsPFR.solve
    records = []
    next_solver_id = 0
    active_pfr = []

    def pfr_solve(self, *args, **kwargs):
        context = dict(getattr(self, 'solve_context', {}) or {})
        active_pfr.append({
            'unit': self.unit_id,
            'recycle_evaluation': context.get('recycle_evaluation'),
            'recycle_final_pass': bool(context.get('recycle_final_pass')),
        })
        try:
            return original_pfr_solve(self, *args, **kwargs)
        finally:
            active_pfr.pop()

    def trial_state(self, *args, **kwargs):
        if 'PFR axial state' in self.unit_label:
            self._ph_probe_trial_states = getattr(
                self,
                '_ph_probe_trial_states',
                0,
            ) + 1
        return original_trial_state(self, *args, **kwargs)

    def capture_request(
        self,
        pressure,
        flow,
        composition,
        target,
        temperature_guess,
        force_phase=None,
        include=None,
        T_bounds=None,
        *,
        temperature_only=False,
    ):
        nonlocal next_solver_id
        original = (
            original_temperature_at_enthalpy
            if temperature_only else original_state_at_enthalpy
        )
        arguments = (
            self, pressure, flow, composition, target, temperature_guess,
            force_phase,
        )
        if not temperature_only:
            arguments += (include, T_bounds)
        if (
            'PFR axial state' not in self.unit_label
            or force_phase != 'vapor'
            or getattr(self, '_ph_probe_active', False)
        ):
            return original(*arguments)
        if not hasattr(self, '_ph_probe_solver_id'):
            self._ph_probe_solver_id = next_solver_id
            next_solver_id += 1
        before = getattr(self, '_ph_probe_trial_states', 0)
        previous_temperature = getattr(
            self,
            '_ph_probe_previous_temperature',
            None,
        )
        start = time.perf_counter()
        self._ph_probe_active = True
        try:
            result, error = original(*arguments)
        finally:
            self._ph_probe_active = False
        temperature = float(result if temperature_only else result.T)
        elapsed = time.perf_counter() - start
        self._ph_probe_previous_temperature = temperature
        pfr_context = active_pfr[-1] if active_pfr else {}
        records.append({
            'solver': self._ph_probe_solver_id,
            'unit': pfr_context.get('unit'),
            'recycle_evaluation': pfr_context.get('recycle_evaluation'),
            'recycle_final_pass': pfr_context.get('recycle_final_pass', False),
            'P': float(pressure),
            'F': float(flow),
            'composition': {
                str(component): float(value)
                for component, value in composition.items()
            },
            'H_target': float(target),
            'original_seed': float(temperature_guess),
            'previous_seed': (
                None
                if previous_temperature is None
                else float(previous_temperature)
            ),
            'temperature': temperature,
            'temperature_only': temperature_only,
            'enthalpy_error': float(error),
            'trial_states': (
                getattr(self, '_ph_probe_trial_states', 0) - before
            ),
            'seconds': elapsed,
        })
        return result, error

    def temperature_at_enthalpy(self, *args, **kwargs):
        return capture_request(self, *args, **kwargs, temperature_only=True)

    _ThermoStateSolver._calculate_trial_state = trial_state
    _ThermoStateSolver.state_at_enthalpy = capture_request
    _ThermoStateSolver.temperature_at_enthalpy = temperature_at_enthalpy
    KineticsPFR.solve = pfr_solve
    try:
        simulator = Simulator.from_file(str(EXAMPLE))
        simulator.initialize()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        result = simulator.run()
        capture = {
            'wall_seconds': time.perf_counter() - wall_start,
            'cpu_seconds': time.process_time() - cpu_start,
            'converged': result.converged,
            'errors': result.errors,
            'iterations': result.iterations,
            'mass_balance_error': result.mass_balance_error,
            'pfr_ph_requests': len(records),
            'pfr_trial_states': sum(row['trial_states'] for row in records),
            'ph_seconds': sum(row['seconds'] for row in records),
            'trial_states_per_request': _summary([
                row['trial_states'] for row in records
            ]),
            'request_seconds': _summary([
                row['seconds'] for row in records
            ]),
        }
    finally:
        _ThermoStateSolver._calculate_trial_state = original_trial_state
        _ThermoStateSolver.state_at_enthalpy = original_state_at_enthalpy
        _ThermoStateSolver.temperature_at_enthalpy = original_temperature_at_enthalpy
        KineticsPFR.solve = original_pfr_solve
    return capture, records


def _clear_thermo_caches(thermo):
    for name in (
        '_enthalpy_ideal_cache',
        '_cp_integral_cache',
        '_cp_ideal_cache',
        '_cp_departure_cache',
    ):
        cache = getattr(thermo, name, None)
        if cache is not None:
            cache.clear()
    cubic = thermo.cubic
    for name in (
        '_phi_phi_k_cache',
        '_pair_kij_values_cache',
        '_pair_dkij_values_cache',
        '_kij_cache',
    ):
        cache = getattr(cubic, name, None)
        if cache is not None:
            cache.clear()


def _seed(record, better_seed):
    if better_seed and record['previous_seed'] is not None:
        return float(record['previous_seed'])
    return float(record['original_seed'])


def replay_current(thermo, records, better_seed):
    _clear_thermo_caches(thermo)
    solver = _ThermoStateSolver(thermo, 'Haber-Bosch PH replay')
    temperatures = []
    residuals = []
    elapsed_values = []
    start = time.perf_counter()
    for record in records:
        # Compare the requested seeds, without the production predictor
        # replacing both strategies with the same history-based seed.
        solver._reset_ph_predictor('vapor')
        request_start = time.perf_counter()
        state, residual = solver.state_at_enthalpy(
            record['P'],
            record['F'],
            record['composition'],
            record['H_target'],
            _seed(record, better_seed),
            force_phase='vapor',
            include=('H',),
        )
        elapsed_values.append(time.perf_counter() - request_start)
        temperatures.append(float(state.T))
        residuals.append(float(residual))
    return _replay_payload(
        records,
        temperatures,
        residuals,
        elapsed_values,
        time.perf_counter() - start,
    )


def _direct_temperature(
    enthalpy,
    heat_capacity,
    pressure,
    composition,
    target,
    seed,
):
    tolerance = max(1e-8, abs(target) * 1e-12)
    temperature = max(1.0, min(5000.0, float(seed)))
    evaluations = 0
    for _iteration in range(12):
        value = enthalpy(temperature, pressure, composition)
        evaluations += 1
        residual = value - target
        if abs(residual) <= tolerance:
            return temperature, residual, evaluations
        slope = heat_capacity(temperature, pressure, composition)
        if not math.isfinite(slope) or abs(slope) < 1e-12:
            raise RuntimeError('nonfinite direct-PH heat capacity')
        step = max(
            -0.35 * max(abs(temperature), 50.0),
            min(0.35 * max(abs(temperature), 50.0), residual / slope),
        )
        accepted = False
        damping = 1.0
        for _trial in range(8):
            candidate = max(
                1.0,
                min(5000.0, temperature - damping * step),
            )
            candidate_value = enthalpy(candidate, pressure, composition)
            evaluations += 1
            candidate_residual = candidate_value - target
            if (
                abs(candidate_residual) <= abs(residual) * 0.9
                or abs(candidate_residual) <= tolerance
            ):
                temperature = candidate
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            raise RuntimeError('direct-PH Newton step was not accepted')
    value = enthalpy(temperature, pressure, composition)
    evaluations += 1
    residual = value - target
    if abs(residual) > tolerance:
        raise RuntimeError('direct-PH Newton solve did not converge')
    return temperature, residual, evaluations


class DirectThermoEvaluator:
    def __init__(self, thermo):
        self.thermo = thermo

    def enthalpy(self, temperature, pressure, composition):
        return self.thermo.mixture_enthalpy(
            composition,
            temperature,
            1.0,
            P=pressure,
        )

    def heat_capacity(self, temperature, pressure, composition):
        return self.thermo.mixture_Cp(
            composition,
            temperature,
            1.0,
            pressure,
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
    def _chebyshev_cp_numba(temperature, center, scale, coefficients, count):
        mapped = (temperature - center) / (scale * (temperature + center))
        first = 0.0
        second = 0.0
        for index in range(count - 1, 0, -1):
            current = 2.0 * mapped * first - second + coefficients[index]
            second = first
            first = current
        return mapped * first - second + coefficients[0]


    @njit(cache=True)
    def _chebyshev_delta_h_numba(
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
    def _compact_ideal_h_cp_numba(
        temperature,
        composition,
        formation_enthalpy,
        centers,
        scales,
        coefficients,
        coefficient_counts,
        h_polynomials,
        h_polynomial_counts,
        h_logarithmic,
        h_reciprocal,
    ):
        enthalpy = 0.0
        heat_capacity = 0.0
        for component in range(composition.shape[0]):
            cp_value = _chebyshev_cp_numba(
                temperature,
                centers[component],
                scales[component],
                coefficients[component],
                coefficient_counts[component],
            )
            delta_h = _chebyshev_delta_h_numba(
                T_REF,
                temperature,
                centers[component],
                scales[component],
                h_polynomials[component],
                h_polynomial_counts[component],
                h_logarithmic[component],
                h_reciprocal[component],
            )
            fraction = composition[component]
            enthalpy += fraction * (
                1000.0 * formation_enthalpy[component] + delta_h
            )
            heat_capacity += fraction * cp_value
        return enthalpy, heat_capacity


class CompactVaporEvaluator:
    def __init__(self, thermo):
        if njit is None or thermo.cubic._compiled_backend is None:
            raise RuntimeError('compact probe requires Numba cubic backend')
        self.thermo = thermo
        self.components = tuple(thermo.components)
        kernels = [thermo._ideal_gas_cp_kernel(item) for item in self.components]
        if not all(isinstance(item, ChebyshevCpKernel) for item in kernels):
            raise RuntimeError('compact probe requires Chebyshev Cp kernels')
        coefficient_width = max(len(item.coefficients) for item in kernels)
        h_width = max(len(item.h_polynomial) for item in kernels)
        self.formation_enthalpy = np.asarray([
            float(thermo.props[item].Hf or 0.0) for item in self.components
        ])
        self.centers = np.asarray([item.center for item in kernels])
        self.scales = np.asarray([item.scale for item in kernels])
        self.coefficients = np.zeros((len(kernels), coefficient_width))
        self.coefficient_counts = np.asarray([
            len(item.coefficients) for item in kernels
        ], dtype=np.int64)
        self.h_polynomials = np.zeros((len(kernels), h_width))
        self.h_polynomial_counts = np.asarray([
            len(item.h_polynomial) for item in kernels
        ], dtype=np.int64)
        self.h_logarithmic = np.asarray([
            item.h_log_coefficient for item in kernels
        ])
        self.h_reciprocal = np.asarray([
            item.h_reciprocal_coefficient for item in kernels
        ])
        for index, kernel in enumerate(kernels):
            self.coefficients[index, :len(kernel.coefficients)] = (
                kernel.coefficients
            )
            self.h_polynomials[index, :len(kernel.h_polynomial)] = (
                kernel.h_polynomial
            )
        self.backend = thermo.cubic._compiled_backend
        uniform = np.full(len(kernels), 1.0 / len(kernels))
        _compact_ideal_h_cp_numba(
            600.0,
            uniform,
            self.formation_enthalpy,
            self.centers,
            self.scales,
            self.coefficients,
            self.coefficient_counts,
            self.h_polynomials,
            self.h_polynomial_counts,
            self.h_logarithmic,
            self.h_reciprocal,
        )

    def _composition(self, composition):
        values = np.asarray([
            composition.get(component, 0.0) for component in self.components
        ])
        return values / values.sum()

    def _ideal(self, temperature, values):
        return _compact_ideal_h_cp_numba(
            temperature,
            values,
            self.formation_enthalpy,
            self.centers,
            self.scales,
            self.coefficients,
            self.coefficient_counts,
            self.h_polynomials,
            self.h_polynomial_counts,
            self.h_logarithmic,
            self.h_reciprocal,
        )

    def enthalpy(self, temperature, pressure, composition):
        values = self._composition(composition)
        ideal_enthalpy, _ = self._ideal(temperature, values)
        departure = self.backend.departure_enthalpy(
            temperature,
            pressure,
            values,
            'vapor',
        )
        return float(ideal_enthalpy + departure)

    def heat_capacity(self, temperature, pressure, composition):
        values = self._composition(composition)
        _, ideal_cp = self._ideal(temperature, values)
        delta = max(0.05, 1e-4 * temperature)
        low = self.backend.departure_enthalpy(
            max(1.0, temperature - delta),
            pressure,
            values,
            'vapor',
        )
        high = self.backend.departure_enthalpy(
            temperature + delta,
            pressure,
            values,
            'vapor',
        )
        departure_cp = high - low
        departure_cp /= temperature + delta - max(1.0, temperature - delta)
        return float(ideal_cp + departure_cp)


def _state_features(records, components):
    raw = np.asarray([
        [
            record['H_target'],
            math.log(max(record['P'], 1.0e-300)),
            *(
                record['composition'].get(component, 0.0)
                for component in components
            ),
        ]
        for record in records
    ], dtype=np.float64)
    center = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return (raw - center) / scale


def _recent_state_predictions(records, features, steps):
    history = {}
    predictions = []
    available = []
    for index, record in enumerate(records):
        prior_indices = history.setdefault(record['solver'], [])
        if not prior_indices:
            predictions.append(record['original_seed'])
            available.append(False)
            prior_indices.append(index)
            continue
        recent_indices = prior_indices[-steps:]
        base_index = recent_indices[-1]
        base_temperature = records[base_index]['temperature']
        if len(recent_indices) == 1:
            prediction = base_temperature
        else:
            matrix = np.asarray([
                features[item] - features[base_index]
                for item in recent_indices[:-1]
            ])
            response = np.asarray([
                records[item]['temperature'] - base_temperature
                for item in recent_indices[:-1]
            ])
            try:
                sensitivity = np.linalg.lstsq(
                    matrix,
                    response,
                    rcond=1.0e-10,
                )[0]
                prediction = base_temperature + float(
                    (features[index] - features[base_index]) @ sensitivity
                )
            except np.linalg.LinAlgError:
                prediction = base_temperature
        recent_temperatures = [
            records[item]['temperature'] for item in recent_indices
        ]
        recent_span = max(recent_temperatures) - min(recent_temperatures)
        maximum_jump = max(10.0, 3.0 * recent_span)
        prediction = max(
            1.0,
            min(
                5000.0,
                max(
                    base_temperature - maximum_jump,
                    min(base_temperature + maximum_jump, prediction),
                ),
            ),
        )
        predictions.append(prediction)
        available.append(True)
        prior_indices.append(index)
    return predictions, available


def _online_scaled_secant_predictions(records, components):
    models = {}
    predictions = []
    available = []
    floors = np.asarray([100.0, 0.01, *([1.0e-4] * len(components))])
    for record in records:
        feature = np.asarray([
            record['H_target'],
            math.log(max(record['P'], 1.0e-300)),
            *(
                record['composition'].get(component, 0.0)
                for component in components
            ),
        ])
        model = models.get(record['solver'])
        if model is None:
            models[record['solver']] = {
                'count': 1,
                'mean': feature.copy(),
                'm2': np.zeros_like(feature),
                'features': [feature.copy()],
                'temperatures': [record['temperature']],
            }
            predictions.append(record['original_seed'])
            available.append(False)
            continue

        count = model['count']
        if len(model['features']) < 2:
            prediction = model['temperatures'][-1]
            is_available = False
        else:
            variance = model['m2'] / max(count - 1, 1)
            scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), floors)
            previous_delta = (
                model['features'][-1] - model['features'][-2]
            ) / scale
            current_delta = (feature - model['features'][-1]) / scale
            denominator = float(previous_delta @ previous_delta)
            factor = (
                0.0
                if denominator <= 1.0e-20
                else float(current_delta @ previous_delta) / denominator
            )
            prediction = model['temperatures'][-1] + factor * (
                model['temperatures'][-1] - model['temperatures'][-2]
            )
            span = abs(
                model['temperatures'][-1] - model['temperatures'][-2]
            )
            maximum_jump = max(10.0, 3.0 * span)
            prediction = max(
                1.0,
                min(
                    5000.0,
                    max(
                        model['temperatures'][-1] - maximum_jump,
                        min(
                            model['temperatures'][-1] + maximum_jump,
                            prediction,
                        ),
                    ),
                ),
            )
            is_available = True
        predictions.append(prediction)
        available.append(is_available)

        new_count = count + 1
        delta = feature - model['mean']
        model['mean'] += delta / new_count
        model['m2'] += delta * (feature - model['mean'])
        model['count'] = new_count
        model['features'].append(feature.copy())
        model['temperatures'].append(record['temperature'])
        if len(model['features']) > 2:
            model['features'].pop(0)
            model['temperatures'].pop(0)
    return predictions, available


def _recent_polynomial_predictions(
    records,
    features,
    *,
    degree,
    steps,
):
    history = {}
    predictions = []
    available = []
    for index, record in enumerate(records):
        prior_indices = history.setdefault(record['solver'], [])
        if len(prior_indices) < degree + 1:
            prediction = (
                records[prior_indices[-1]]['temperature']
                if prior_indices
                else record['original_seed']
            )
            predictions.append(prediction)
            available.append(False)
            prior_indices.append(index)
            continue

        recent_indices = prior_indices[-steps:]
        base_index = recent_indices[-1]
        base_feature = features[base_index]
        centered = np.asarray([
            features[item] - base_feature for item in recent_indices
        ])
        try:
            _, singular_values, right = np.linalg.svd(
                centered,
                full_matrices=False,
            )
        except np.linalg.LinAlgError:
            singular_values = np.asarray([])
            right = np.empty((0, centered.shape[1]))
        if not len(singular_values) or singular_values[0] <= 1.0e-14:
            prediction = records[base_index]['temperature']
            predictions.append(prediction)
            available.append(False)
            prior_indices.append(index)
            continue

        direction = right[0]
        if len(recent_indices) >= 2:
            last_step = (
                features[base_index] - features[recent_indices[-2]]
            )
            if float(last_step @ direction) < 0.0:
                direction = -direction
        coordinates = centered @ direction
        coordinate_scale = max(float(np.max(np.abs(coordinates))), 1.0e-14)
        normalized = coordinates / coordinate_scale
        temperatures = np.asarray([
            records[item]['temperature'] for item in recent_indices
        ])
        vandermonde = np.vander(
            normalized,
            N=degree + 1,
            increasing=True,
        )
        try:
            coefficients = np.linalg.lstsq(
                vandermonde,
                temperatures,
                rcond=1.0e-10,
            )[0]
            current_coordinate = float(
                (features[index] - base_feature) @ direction
            ) / coordinate_scale
            powers = np.asarray([
                current_coordinate**power
                for power in range(degree + 1)
            ])
            prediction = float(powers @ coefficients)
        except np.linalg.LinAlgError:
            prediction = records[base_index]['temperature']

        recent_span = float(np.max(temperatures) - np.min(temperatures))
        maximum_jump = max(10.0, 3.0 * recent_span)
        base_temperature = records[base_index]['temperature']
        prediction = max(
            1.0,
            min(
                5000.0,
                max(
                    base_temperature - maximum_jump,
                    min(base_temperature + maximum_jump, prediction),
                ),
            ),
        )
        predictions.append(prediction)
        available.append(True)
        prior_indices.append(index)
    return predictions, available


def _previous_recycle_predictions(records, features, neighbors):
    groups = {}
    for index, record in enumerate(records):
        evaluation = record.get('recycle_evaluation')
        unit = record.get('unit')
        if (
            unit is None
            or evaluation is None
            or record.get('recycle_final_pass')
        ):
            continue
        groups.setdefault((unit, int(evaluation)), []).append(index)

    predictions = []
    available = []
    for index, record in enumerate(records):
        evaluation = record.get('recycle_evaluation')
        unit = record.get('unit')
        candidates = (
            groups.get((unit, int(evaluation) - 1), ())
            if unit is not None and evaluation is not None
            else ()
        )
        if not candidates:
            predictions.append(
                record['previous_seed']
                if record['previous_seed'] is not None
                else record['original_seed']
            )
            available.append(False)
            continue
        candidate_array = np.asarray(candidates, dtype=np.int64)
        deltas = features[candidate_array] - features[index]
        distances = np.einsum('ij,ij->i', deltas, deltas)
        count = min(neighbors, len(candidate_array))
        nearest_positions = np.argpartition(
            distances,
            count - 1,
        )[:count]
        nearest_indices = candidate_array[nearest_positions]
        nearest_distances = distances[nearest_positions]
        exact = np.flatnonzero(nearest_distances <= 1.0e-24)
        if len(exact):
            prediction = records[int(nearest_indices[int(exact[0])])][
                'temperature'
            ]
        else:
            weights = 1.0 / nearest_distances
            prediction = float(sum(
                weight * records[int(item)]['temperature']
                for weight, item in zip(weights, nearest_indices)
            ) / np.sum(weights))
        predictions.append(max(1.0, min(5000.0, prediction)))
        available.append(True)
    return predictions, available


def _recursive_basis(delta, quadratic):
    if not quadratic:
        return delta
    terms = list(delta)
    for left in range(len(delta)):
        for right in range(left, len(delta)):
            terms.append(delta[left] * delta[right])
    return np.asarray(terms)


def _recursive_least_squares_predictions(
    records,
    features,
    *,
    forgetting,
    quadratic,
):
    models = {}
    predictions = []
    available = []
    for index, record in enumerate(records):
        solver = record['solver']
        model = models.get(solver)
        if model is None:
            basis_width = len(_recursive_basis(features[index] * 0.0, quadratic))
            models[solver] = {
                'base_feature': features[index].copy(),
                'base_temperature': record['temperature'],
                'coefficients': np.zeros(basis_width),
                'covariance': np.eye(basis_width) * 10.0,
                'last_temperature': record['temperature'],
                'recent_temperatures': [record['temperature']],
            }
            predictions.append(record['original_seed'])
            available.append(False)
            continue

        delta = features[index] - model['base_feature']
        basis = _recursive_basis(delta, quadratic)
        prediction = (
            model['base_temperature']
            + float(basis @ model['coefficients'])
        )
        recent = model['recent_temperatures'][-8:]
        span = max(recent) - min(recent)
        maximum_jump = max(10.0, 3.0 * span)
        last_temperature = model['last_temperature']
        prediction = max(
            1.0,
            min(
                5000.0,
                max(
                    last_temperature - maximum_jump,
                    min(last_temperature + maximum_jump, prediction),
                ),
            ),
        )
        predictions.append(prediction)
        available.append(True)

        covariance = model['covariance']
        projected = covariance @ basis
        denominator = forgetting + float(basis @ projected)
        gain = projected / max(denominator, 1.0e-14)
        target_delta = record['temperature'] - model['base_temperature']
        error = target_delta - float(basis @ model['coefficients'])
        model['coefficients'] += gain * error
        model['covariance'] = (
            covariance - np.outer(gain, basis) @ covariance
        ) / forgetting
        model['last_temperature'] = record['temperature']
        model['recent_temperatures'].append(record['temperature'])
    return predictions, available


def _newton_polynomial_value(abscissas, values, target):
    coefficients = [float(value) for value in values]
    count = len(coefficients)
    for order in range(1, count):
        for index in range(count - 1, order - 1, -1):
            denominator = abscissas[index] - abscissas[index - order]
            if abs(denominator) <= 1.0e-14:
                return float(values[-1])
            coefficients[index] = (
                coefficients[index] - coefficients[index - 1]
            ) / denominator
    result = coefficients[-1]
    for index in range(count - 2, -1, -1):
        result = coefficients[index] + (
            target - abscissas[index]
        ) * result
    return result


def _arc_polynomial_predictions(records, features, degree):
    histories = {}
    predictions = []
    available = []
    for index, record in enumerate(records):
        history = histories.setdefault(record['solver'], [])
        if not history:
            coordinate = 0.0
        else:
            previous_index, previous_coordinate = history[-1]
            delta = features[index] - features[previous_index]
            distance = float(np.linalg.norm(delta))
            if len(history) >= 2:
                older_index = history[-2][0]
                direction = features[previous_index] - features[older_index]
                sign = -1.0 if float(delta @ direction) < 0.0 else 1.0
            else:
                sign = 1.0
            coordinate = previous_coordinate + sign * distance

        if len(history) < degree + 1:
            prediction = (
                records[history[-1][0]]['temperature']
                if history
                else record['original_seed']
            )
            available.append(False)
        else:
            recent = history[-(degree + 1):]
            abscissas = [item[1] for item in recent]
            temperatures = [
                records[item[0]]['temperature'] for item in recent
            ]
            prediction = _newton_polynomial_value(
                abscissas,
                temperatures,
                coordinate,
            )
            base_temperature = temperatures[-1]
            span = max(temperatures) - min(temperatures)
            maximum_jump = max(10.0, 3.0 * span)
            prediction = max(
                1.0,
                min(
                    5000.0,
                    max(
                        base_temperature - maximum_jump,
                        min(base_temperature + maximum_jump, prediction),
                    ),
                ),
            )
            available.append(True)
        predictions.append(prediction)
        history.append((index, coordinate))
    return predictions, available


def _corrected_temperature(temperature, residual, heat_capacity):
    step = residual / heat_capacity
    maximum = 0.35 * max(abs(temperature), 50.0)
    step = max(-maximum, min(maximum, step))
    return max(1.0, min(5000.0, temperature - step))


def probe_predictor(records, predictions, available, evaluator):
    initial_errors = []
    final_errors = []
    evaluations = []
    accepted_initial = 0
    accepted_after_one = 0
    accepted_after_two = 0
    fallbacks = 0
    failures = []
    start = time.perf_counter()

    for index, (record, predicted) in enumerate(zip(records, predictions)):
        target = record['H_target']
        tolerance = max(1.0e-8, abs(target) * 1.0e-12)
        initial_errors.append(abs(predicted - record['temperature']))
        count = 0
        try:
            value = evaluator.enthalpy(
                predicted,
                record['P'],
                record['composition'],
            )
            count += 1
            residual = value - target
            temperature = predicted
            if abs(residual) <= tolerance:
                accepted_initial += 1
            else:
                heat_capacity = evaluator.heat_capacity(
                    temperature,
                    record['P'],
                    record['composition'],
                )
                temperature = _corrected_temperature(
                    temperature,
                    residual,
                    heat_capacity,
                )
                residual = evaluator.enthalpy(
                    temperature,
                    record['P'],
                    record['composition'],
                ) - target
                count += 1
                if abs(residual) <= tolerance:
                    accepted_after_one += 1
                else:
                    heat_capacity = evaluator.heat_capacity(
                        temperature,
                        record['P'],
                        record['composition'],
                    )
                    temperature = _corrected_temperature(
                        temperature,
                        residual,
                        heat_capacity,
                    )
                    residual = evaluator.enthalpy(
                        temperature,
                        record['P'],
                        record['composition'],
                    ) - target
                    count += 1
                    if abs(residual) <= tolerance:
                        accepted_after_two += 1
                    else:
                        fallbacks += 1
                        temperature, residual, extra = _direct_temperature(
                            evaluator.enthalpy,
                            evaluator.heat_capacity,
                            record['P'],
                            record['composition'],
                            target,
                            temperature,
                        )
                        count += extra
            final_errors.append(abs(temperature - record['temperature']))
            evaluations.append(count)
        except Exception as error:
            failures.append({'index': index, 'error': str(error)})

    total = len(records)
    return {
        'requests': total,
        'predictor_available': sum(bool(value) for value in available),
        'predictor_coverage': sum(bool(value) for value in available) / total,
        'initial_temperature_absolute_error_K': _summary(initial_errors),
        'final_temperature_absolute_error_K': _summary(final_errors),
        'enthalpy_residual_evaluations': _summary(evaluations),
        'accepted_without_correction': accepted_initial,
        'accepted_after_one_correction': accepted_after_one,
        'accepted_after_two_corrections': accepted_after_two,
        'full_solver_fallbacks': fallbacks,
        'full_solver_fallback_fraction': fallbacks / total,
        'failure_count': len(failures),
        'failures': failures[:20],
        'probe_seconds': time.perf_counter() - start,
    }


def predictor_probes(records, evaluator):
    features = _state_features(records, evaluator.components)
    result = {}
    predictions, available = _online_scaled_secant_predictions(
        records,
        evaluator.components,
    )
    result['continuous_online_scaled_secant'] = probe_predictor(
        records,
        predictions,
        available,
        evaluator,
    )
    recent_predictions = {}
    recent_available = {}
    for steps in (1, 2, 3, 4, 8):
        predictions, available = _recent_state_predictions(
            records,
            features,
            steps,
        )
        recent_predictions[steps] = predictions
        recent_available[steps] = available
        result[f'recent_state_{steps}'] = probe_predictor(
            records,
            predictions,
            available,
            evaluator,
        )
    for degree, windows in ((2, (3, 4, 6, 8)), (3, (4, 6, 8, 12))):
        degree_name = 'quadratic' if degree == 2 else 'cubic'
        for steps in windows:
            predictions, available = _recent_polynomial_predictions(
                records,
                features,
                degree=degree,
                steps=steps,
            )
            result[f'recent_{degree_name}_{steps}'] = probe_predictor(
                records,
                predictions,
                available,
                evaluator,
            )
    for forgetting in (0.5, 0.8, 0.95, 0.99):
        label = str(forgetting).replace('.', '_')
        for quadratic in (False, True):
            predictions, available = _recursive_least_squares_predictions(
                records,
                features,
                forgetting=forgetting,
                quadratic=quadratic,
            )
            order = 'quadratic' if quadratic else 'linear'
            result[f'continuous_{order}_rls_{label}'] = probe_predictor(
                records,
                predictions,
                available,
                evaluator,
            )
    for degree, name in ((1, 'linear'), (2, 'quadratic'), (3, 'cubic')):
        predictions, available = _arc_polynomial_predictions(
            records,
            features,
            degree,
        )
        result[f'continuous_arc_{name}'] = probe_predictor(
            records,
            predictions,
            available,
            evaluator,
        )
    if records and 'unit' in records[0]:
        recycle_predictions = {}
        recycle_available = {}
        for neighbors in (1, 2, 4, 8):
            predictions, available = _previous_recycle_predictions(
                records,
                features,
                neighbors,
            )
            recycle_predictions[neighbors] = predictions
            recycle_available[neighbors] = available
            result[f'previous_recycle_{neighbors}'] = probe_predictor(
                records,
                predictions,
                available,
                evaluator,
            )
        hybrid_predictions = []
        hybrid_available = []
        for index, record in enumerate(records):
            if recent_available[2][index]:
                hybrid_predictions.append(recent_predictions[2][index])
                hybrid_available.append(True)
            elif recycle_available[1][index]:
                hybrid_predictions.append(recycle_predictions[1][index])
                hybrid_available.append(True)
            else:
                hybrid_predictions.append(record['original_seed'])
                hybrid_available.append(False)
        result['recent_state_2_then_previous_recycle_1'] = probe_predictor(
            records,
            hybrid_predictions,
            hybrid_available,
            evaluator,
        )
    return result


def replay_direct(thermo, records, evaluator, better_seed):
    _clear_thermo_caches(thermo)
    temperatures = []
    residuals = []
    elapsed_values = []
    evaluation_counts = []
    failures = []
    start = time.perf_counter()
    for index, record in enumerate(records):
        request_start = time.perf_counter()
        try:
            temperature, residual, evaluations = _direct_temperature(
                evaluator.enthalpy,
                evaluator.heat_capacity,
                record['P'],
                record['composition'],
                record['H_target'],
                _seed(record, better_seed),
            )
            final_state = thermo.calculate_state(
                temperature,
                record['P'],
                record['F'],
                record['composition'],
                phase='vapor',
                flash=False,
                include=('H',),
            )
            residual = float(final_state.H - record['H_target'])
            temperatures.append(float(temperature))
            residuals.append(residual)
            evaluation_counts.append(evaluations)
        except Exception as error:
            temperatures.append(float('nan'))
            residuals.append(float('nan'))
            evaluation_counts.append(0)
            failures.append({'index': index, 'error': str(error)})
        elapsed_values.append(time.perf_counter() - request_start)
    payload = _replay_payload(
        records,
        temperatures,
        residuals,
        elapsed_values,
        time.perf_counter() - start,
    )
    payload['enthalpy_evaluations'] = _summary(evaluation_counts)
    payload['failures'] = failures[:20]
    payload['failure_count'] = len(failures)
    return payload


def _replay_payload(records, temperatures, residuals, elapsed, total):
    temperature_errors = [
        abs(actual - record['temperature'])
        for actual, record in zip(temperatures, records)
        if math.isfinite(actual)
    ]
    finite_residuals = [abs(value) for value in residuals if math.isfinite(value)]
    return {
        'requests': len(records),
        'total_seconds': total,
        'microseconds_per_request': 1e6 * total / max(len(records), 1),
        'request_seconds': _summary(elapsed),
        'temperature_absolute_error_K': _summary(temperature_errors),
        'enthalpy_absolute_residual_kJ_kmol': _summary(finite_residuals),
    }


def load_capture(path):
    payload = json.loads(path.read_text())
    return payload['capture'], payload['records']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument('--capture-input', type=Path)
    capture.add_argument('--capture-output', type=Path)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')

    if args.capture_input is not None:
        capture_summary, captured_records = load_capture(args.capture_input)
    else:
        capture_summary, captured_records = capture_workload()
        if args.capture_output is not None:
            args.capture_output.write_text(json.dumps({
                'capture': capture_summary,
                'records': captured_records,
            }, indent=2) + '\n')
    records = _sample_evenly(captured_records, args.samples)

    simulator = Simulator.from_file(str(EXAMPLE))
    simulator.initialize()
    thermo = simulator.thermo
    direct = DirectThermoEvaluator(thermo)
    compact = CompactVaporEvaluator(thermo)

    probes = {
        'current_original_seed': replay_current(thermo, records, False),
        'current_previous_seed': replay_current(thermo, records, True),
        'direct_original_seed': replay_direct(
            thermo, records, direct, False
        ),
        'direct_previous_seed': replay_direct(
            thermo, records, direct, True
        ),
        'compact_original_seed': replay_direct(
            thermo, records, compact, False
        ),
        'compact_previous_seed': replay_direct(
            thermo, records, compact, True
        ),
    }
    predictors = predictor_probes(captured_records, compact)
    baseline = probes['current_original_seed']['total_seconds']
    for probe in probes.values():
        probe['speedup_vs_current_original_seed'] = (
            baseline / probe['total_seconds']
        )
    payload = {
        'example': EXAMPLE.name,
        'capture': capture_summary,
        'sample_count': len(records),
        'probes': probes,
        'predictors': predictors,
    }
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
