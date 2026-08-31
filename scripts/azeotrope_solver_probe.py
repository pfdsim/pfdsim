#!/usr/bin/env python3
"""Prototype azeotrope finders for activity-coefficient thermo models.

This is intentionally standalone.  It is a scratchpad for binary and ternary
azeotrope algorithms before moving any of the logic into thermodynamics.py.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermodynamics import create_thermodynamics  # noqa: E402


# Use auto group resolution by default.  Keep only overrides that are currently
# needed because the generic resolver picks a subgroup name unsupported by a
# specific backend.
GROUP_OVERRIDES = {
    'acrylonitrile': {'Acrylonitrile': 1},
}


@dataclass
class Azeotrope:
    kind: str
    components: tuple[str, ...]
    T: float
    x: dict[str, float]
    y: dict[str, float]
    residual: float
    elapsed_s: float
    extra: dict

    def as_dict(self) -> dict:
        return {
            'kind': self.kind,
            'components': self.components,
            'T_C': self.T - 273.15,
            'x': self.x,
            'y': self.y,
            'residual': self.residual,
            'elapsed_s': self.elapsed_s,
            **self.extra,
        }


def normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(float(value), 0.0) for value in values.values())
    if total <= 0.0:
        return {key: 1.0 / len(values) for key in values}
    return {key: max(float(value), 0.0) / total for key, value in values.items()}


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def ternary_x_from_vars(values: np.ndarray) -> np.ndarray:
    return softmax(np.array([values[0], values[1], 0.0], dtype=float))


def ternary_vars_from_x(x_values: np.ndarray) -> np.ndarray:
    x_values = np.maximum(np.asarray(x_values, dtype=float), 1e-14)
    x_values = x_values / np.sum(x_values)
    return np.array([
        math.log(x_values[0] / x_values[2]),
        math.log(x_values[1] / x_values[2]),
    ], dtype=float)


def bounded_T(theta: float, T_min: float, T_max: float) -> float:
    return 0.5 * (T_min + T_max) + 0.5 * (T_max - T_min) * math.tanh(float(theta))


def bounded_T_var(T: float, T_min: float, T_max: float) -> float:
    scaled = (2.0 * float(T) - T_min - T_max) / max(T_max - T_min, 1e-12)
    scaled = min(max(scaled, -0.999999999), 0.999999999)
    return math.atanh(scaled)


def sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def logit(value: float) -> float:
    value = min(max(float(value), 1e-14), 1.0 - 1e-14)
    return math.log(value / (1.0 - value))


def vapor_from_liquid(thermo, composition: dict[str, float], T: float, P: float):
    K = thermo.K_values(T, P, composition)
    raw = {comp: composition[comp] * K[comp] for comp in composition}
    return normalize(raw), K, sum(raw.values())


def pure_boiling_temperatures(thermo, comps: list[str], P: float) -> dict[str, float]:
    out = {}
    for comp in comps:
        out[comp] = thermo.bubble_point_T(
            {name: (1.0 if name == comp else 0.0) for name in comps},
            P,
        )
    return out


def binary_type(T: float, pure_T: dict[str, float]) -> str:
    lo = min(pure_T.values())
    hi = max(pure_T.values())
    if T < lo - 1e-5:
        return 'minimum'
    if T > hi + 1e-5:
        return 'maximum'
    return 'intermediate'


def find_binary_homogeneous(thermo, pair: tuple[str, str], P: float,
                            tol: float = 1e-9) -> list[Azeotrope]:
    start = time.perf_counter()
    comp_a, comp_b = pair

    def residual_at(x_a: float) -> float:
        x_a = min(max(float(x_a), 1e-12), 1.0 - 1e-12)
        x = {comp_a: x_a, comp_b: 1.0 - x_a}
        T = thermo.bubble_point_T(x, P)
        K = thermo.K_values(T, P, x)
        return math.log(max(K[comp_a], 1e-300) / max(K[comp_b], 1e-300))

    eps_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    bracket = None
    for eps in eps_values:
        try:
            f_lo = residual_at(eps)
            f_hi = residual_at(1.0 - eps)
        except Exception:
            continue
        if f_lo == 0.0:
            bracket = (eps, eps)
            break
        if f_hi == 0.0:
            bracket = (1.0 - eps, 1.0 - eps)
            break
        if f_lo * f_hi < 0.0:
            bracket = (eps, 1.0 - eps)
            break

    root = None
    if bracket is not None and bracket[0] != bracket[1]:
        root = brentq(residual_at, bracket[0], bracket[1], xtol=1e-13, rtol=1e-13, maxiter=80)
    elif bracket is not None:
        root = bracket[0]
    else:
        optimum = minimize_scalar(
            lambda value: abs(residual_at(value)),
            bounds=(1e-6, 1.0 - 1e-6),
            method='bounded',
            options={'xatol': 1e-12, 'maxiter': 120},
        )
        if optimum.success and abs(residual_at(optimum.x)) < tol:
            root = float(optimum.x)

    if root is None or root < 1e-5 or root > 1.0 - 1e-5:
        return []

    x = {comp_a: root, comp_b: 1.0 - root}
    T = thermo.bubble_point_T(x, P)
    y, K, bubble = vapor_from_liquid(thermo, x, T, P)
    pure_T = pure_boiling_temperatures(thermo, list(pair), P)
    return [Azeotrope(
        kind='binary_homogeneous',
        components=pair,
        T=T,
        x=x,
        y=y,
        residual=max(abs(math.log(max(K[comp_a], 1e-300) / max(K[comp_b], 1e-300))), abs(bubble - 1.0)),
        elapsed_s=time.perf_counter() - start,
        extra={
            'type': binary_type(T, pure_T),
            'pure_T_C': {comp: temp - 273.15 for comp, temp in pure_T.items()},
        },
    )]


def find_binary_heterogeneous(thermo, pair: tuple[str, str], P: float,
                              tol: float = 1e-8) -> list[Azeotrope]:
    start = time.perf_counter()
    comp_a, comp_b = pair
    pure_T = pure_boiling_temperatures(thermo, list(pair), P)
    T_min = max(250.0, min(pure_T.values()) - 50.0)
    T_max = min(650.0, max(pure_T.values()) + 80.0)

    def lle_at(T: float):
        has_lle, x1, x2, beta = thermo.liquid_liquid_equilibrium(
            {comp_a: 0.5, comp_b: 0.5},
            T,
            tol=tol,
        )
        if not has_lle:
            return None
        P1 = thermo.bubble_point_P(x1, T)
        P2 = thermo.bubble_point_P(x2, T)
        y, K, bubble = vapor_from_liquid(thermo, x1, T, max(P1, 1e-12))
        return x1, x2, y, P1, P2, K, bubble

    # Cheap early rejection: most binaries are not LLE-forming.  Avoid a full
    # temperature scan when equimolar liquid is single phase at a few broad
    # reference temperatures.
    precheck_temperatures = [
        298.15,
        min(pure_T.values()),
        0.5 * (min(pure_T.values()) + max(pure_T.values())),
    ]
    has_any_lle = False
    for T in precheck_temperatures:
        if not (T_min <= T <= T_max):
            continue
        try:
            if lle_at(T) is not None:
                has_any_lle = True
                break
        except Exception:
            pass
    if not has_any_lle:
        return []

    samples = []
    for T in np.linspace(T_min, T_max, 24):
        try:
            value = lle_at(float(T))
        except Exception:
            value = None
        if value is not None:
            samples.append((float(T), value[3] - P, value))

    roots = []
    for left, right in zip(samples[:-1], samples[1:]):
        T_left, f_left, _ = left
        T_right, f_right, _ = right
        if f_left == 0.0:
            roots.append(T_left)
        elif f_left * f_right < 0.0:
            def pressure_residual(T_value: float) -> float:
                value = lle_at(T_value)
                if value is None:
                    raise ValueError("LLE disappeared while bracketing VLLE")
                return value[3] - P
            roots.append(brentq(
                pressure_residual,
                T_left,
                T_right,
                xtol=1e-10,
                rtol=1e-12,
                maxiter=100,
            ))

    out = []
    for T in roots:
        value = lle_at(T)
        if value is None:
            continue
        x1, x2, y, P1, P2, K, bubble = value
        out.append(Azeotrope(
            kind='binary_heterogeneous',
            components=pair,
            T=T,
            x=y,
            y=y,
            residual=max(abs(P1 - P), abs(P2 - P)),
            elapsed_s=time.perf_counter() - start,
            extra={
                'liquid_phase_1': x1,
                'liquid_phase_2': x2,
                'P_phase_1_bar': P1,
                'P_phase_2_bar': P2,
                'type': binary_type(T, pure_T),
            },
        ))
    return out


def ternary_temperature_window(thermo, comps: list[str], P: float) -> tuple[float, float, list[float]]:
    pure_T = list(pure_boiling_temperatures(thermo, comps, P).values())
    return max(240.0, min(pure_T) - 60.0), min(700.0, max(pure_T) + 90.0), pure_T


def find_ternary_homogeneous(thermo, comps_tuple: tuple[str, str, str], P: float,
                             starts: int = 80, tol: float = 1e-8,
                             check_lle: bool = True) -> list[Azeotrope]:
    start_time = time.perf_counter()
    comps = list(comps_tuple)
    T_min, T_max, pure_T = ternary_temperature_window(thermo, comps, P)

    def residual(values: np.ndarray) -> np.ndarray:
        x_array = ternary_x_from_vars(values[:2])
        x = dict(zip(comps, x_array))
        T = bounded_T(values[2], T_min, T_max)
        try:
            K = thermo.K_values(T, P, x)
            bubble = sum(x[comp] * K[comp] for comp in comps)
            ref = comps[-1]
            return np.array([
                math.log(max(K[comps[0]], 1e-300) / max(K[ref], 1e-300)),
                math.log(max(K[comps[1]], 1e-300) / max(K[ref], 1e-300)),
                math.log(max(bubble, 1e-300)),
            ], dtype=float)
        except Exception:
            return np.ones(3, dtype=float) * 1e3

    rng = random.Random(20260620)
    x_starts = [
        np.array([0.15, 0.15, 0.70]),
        np.array([0.15, 0.70, 0.15]),
        np.array([0.70, 0.15, 0.15]),
        np.array([1.0 / 3.0] * 3),
    ]
    if starts > 4:
        for i in range(1, 6):
            for j in range(1, 6 - i):
                k = 6 - i - j
                if k > 0:
                    x_starts.append(np.array([i / 6.0, j / 6.0, k / 6.0], dtype=float))
    x_starts = x_starts[:max(1, starts)]
    for _ in range(max(0, starts - len(x_starts))):
        draw = np.array([rng.random() for _ in range(3)], dtype=float)
        x_starts.append(draw / np.sum(draw))

    if starts <= 4:
        T_starts = [sum(pure_T) / len(pure_T), min(pure_T), max(pure_T)]
    else:
        T_starts = pure_T + [
            sum(pure_T) / len(pure_T),
            T_min + 0.25 * (T_max - T_min),
            T_min + 0.75 * (T_max - T_min),
        ]
    roots = []
    for x0 in x_starts:
        for T0 in T_starts:
            variables = np.r_[ternary_vars_from_x(x0), bounded_T_var(T0, T_min, T_max)]
            solved = least_squares(
                residual,
                variables,
                method='trf',
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9,
                max_nfev=100,
            )
            norm = float(np.linalg.norm(residual(solved.x), ord=np.inf))
            x_array = ternary_x_from_vars(solved.x[:2])
            T = bounded_T(solved.x[2], T_min, T_max)
            if norm > tol or np.any(x_array < 1e-5) or np.any(x_array > 1.0 - 1e-5):
                continue
            duplicate = any(
                np.linalg.norm(x_array - root[0], ord=np.inf) < 2e-5 and abs(T - root[1]) < 2e-4
                for root in roots
            )
            if not duplicate:
                roots.append((x_array, T, norm))

    out = []
    pure_map = dict(zip(comps, pure_T))
    for x_array, T, norm in roots:
        x = dict(zip(comps, x_array))
        y, K, bubble = vapor_from_liquid(thermo, x, T, P)
        if check_lle:
            try:
                has_lle, x1, x2, beta = thermo.liquid_liquid_equilibrium(x, T, tol=1e-8)
            except Exception:
                has_lle, x1, x2, beta = False, x, x, 0.0
        else:
            has_lle, x1, x2, beta = False, x, x, 0.0
        out.append(Azeotrope(
            kind='ternary_homogeneous',
            components=comps_tuple,
            T=T,
            x=x,
            y=y,
            residual=max(norm, abs(bubble - 1.0)),
            elapsed_s=time.perf_counter() - start_time,
            extra={
                'type': binary_type(T, pure_map),
                'lle_at_root': bool(has_lle),
                'lle_phase_1': x1 if has_lle else None,
                'lle_phase_2': x2 if has_lle else None,
            },
        ))
    return out


def find_ternary_heterogeneous(thermo, comps_tuple: tuple[str, str, str], P: float,
                               starts: int = 80, tol: float = 1e-7,
                               binary_hetero_roots: list[Azeotrope] | None = None) -> list[Azeotrope]:
    start_time = time.perf_counter()
    comps = list(comps_tuple)
    T_min, T_max, pure_T = ternary_temperature_window(thermo, comps, P)

    def residual(values: np.ndarray) -> np.ndarray:
        x1_array = ternary_x_from_vars(values[:2])
        x2_array = ternary_x_from_vars(values[2:4])
        x1 = dict(zip(comps, x1_array))
        x2 = dict(zip(comps, x2_array))
        T = bounded_T(values[4], T_min, T_max)
        beta = sigmoid(values[5])
        try:
            gamma1 = thermo.activity_coefficients(T, x1)
            gamma2 = thermo.activity_coefficients(T, x2)
            K = thermo.K_values(T, P, x1)
            raw_y = np.array([x1[comp] * K[comp] for comp in comps], dtype=float)
            y_array = raw_y / np.sum(raw_y)
            z_array = beta * x1_array + (1.0 - beta) * x2_array
            bubble = float(np.sum(raw_y))
            return np.array([
                math.log(max(x1[comps[0]] * gamma1[comps[0]], 1e-300) / max(x2[comps[0]] * gamma2[comps[0]], 1e-300)),
                math.log(max(x1[comps[1]] * gamma1[comps[1]], 1e-300) / max(x2[comps[1]] * gamma2[comps[1]], 1e-300)),
                math.log(max(x1[comps[2]] * gamma1[comps[2]], 1e-300) / max(x2[comps[2]] * gamma2[comps[2]], 1e-300)),
                math.log(max(bubble, 1e-300)),
                y_array[0] - z_array[0],
                y_array[1] - z_array[1],
            ], dtype=float)
        except Exception:
            return np.ones(6, dtype=float) * 1e3

    rng = random.Random(20260621)
    starts_data = []
    for _ in range(starts):
        x1 = np.array([rng.random() for _ in range(3)], dtype=float)
        x2 = np.array([rng.random() for _ in range(3)], dtype=float)
        x1 /= np.sum(x1)
        x2 /= np.sum(x2)
        if np.linalg.norm(x1 - x2, ord=1) < 0.25:
            continue
        starts_data.append((x1, x2, rng.uniform(T_min, T_max), rng.uniform(0.05, 0.95)))

    binary_hetero_roots = binary_hetero_roots or []
    for root in binary_hetero_roots:
        pair = tuple(comps.index(comp) for comp in root.components)
        phase1 = np.array([root.extra['liquid_phase_1'].get(comp, 1e-12) for comp in comps], dtype=float)
        phase2 = np.array([root.extra['liquid_phase_2'].get(comp, 1e-12) for comp in comps], dtype=float)
        missing = [idx for idx in range(3) if idx not in pair][0]
        # Ternary heteroazeotropes often grow from a binary LLE tie-line after
        # the third component partitions unevenly.  Equal trace perturbations
        # tend to slide back to the binary edge, so deliberately try asymmetric
        # third-component loads.
        for trace_1, trace_2 in (
            (1e-4, 1e-4),
            (1e-3, 1e-2),
            (1e-2, 1e-3),
            (0.02, 0.08),
            (0.08, 0.02),
            (0.05, 0.20),
            (0.20, 0.05),
            (0.15, 0.35),
            (0.35, 0.15),
        ):
            p1 = (1.0 - trace_1) * phase1
            p2 = (1.0 - trace_2) * phase2
            p1[missing] = trace_1
            p2[missing] = trace_2
            for beta0 in (0.25, 0.5, 0.75):
                starts_data.append((p1 / np.sum(p1), p2 / np.sum(p2), root.T, beta0))

    roots = []
    for x1, x2, T0, beta0 in starts_data:
        variables = np.r_[
            ternary_vars_from_x(x1),
            ternary_vars_from_x(x2),
            bounded_T_var(T0, T_min, T_max),
            logit(beta0),
        ]
        solved = least_squares(
            residual,
            variables,
            method='trf',
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=220,
        )
        norm = float(np.linalg.norm(residual(solved.x), ord=np.inf))
        x1_array = ternary_x_from_vars(solved.x[:2])
        x2_array = ternary_x_from_vars(solved.x[2:4])
        T = bounded_T(solved.x[4], T_min, T_max)
        beta = sigmoid(solved.x[5])
        if norm > tol or np.linalg.norm(x1_array - x2_array, ord=1) < 1e-3:
            continue
        if np.any(x1_array < 1e-7) or np.any(x2_array < 1e-7):
            continue
        signature = np.r_[np.sort(np.vstack([x1_array, x2_array]), axis=0).flatten(), T]
        duplicate = any(np.linalg.norm(signature - root[0], ord=np.inf) < 1e-4 for root in roots)
        if not duplicate:
            roots.append((signature, x1_array, x2_array, T, beta, norm))

    out = []
    pure_map = dict(zip(comps, pure_T))
    for _signature, x1_array, x2_array, T, beta, norm in roots:
        x1 = dict(zip(comps, x1_array))
        x2 = dict(zip(comps, x2_array))
        y, K, bubble = vapor_from_liquid(thermo, x1, T, P)
        z = {
            comp: beta * x1[comp] + (1.0 - beta) * x2[comp]
            for comp in comps
        }
        out.append(Azeotrope(
            kind='ternary_heterogeneous',
            components=comps_tuple,
            T=T,
            x=z,
            y=y,
            residual=max(norm, abs(bubble - 1.0)),
            elapsed_s=time.perf_counter() - start_time,
            extra={
                'type': binary_type(T, pure_map),
                'liquid_phase_1': x1,
                'liquid_phase_2': x2,
                'liquid_phase_fraction_1': beta,
            },
        ))
    return out


def run_case(name: str, components: list[str], method: str, pressure: float,
             ternary_starts: int, hetero_starts: int,
             include_heterogeneous: bool = True) -> dict:
    groups = {comp: GROUP_OVERRIDES[comp] for comp in components if comp in GROUP_OVERRIDES}
    thermo = create_thermodynamics(components, method, unifac_groups=groups or None)
    case_start = time.perf_counter()
    results = []
    binary_hetero_roots = []
    for pair in itertools.combinations(components, 2):
        homogeneous = find_binary_homogeneous(thermo, pair, pressure)
        heterogeneous = (
            find_binary_heterogeneous(thermo, pair, pressure)
            if include_heterogeneous
            else []
        )
        results.extend(homogeneous)
        results.extend(heterogeneous)
        binary_hetero_roots.extend(heterogeneous)
    if len(components) == 3:
        trio = tuple(components)
        results.extend(find_ternary_homogeneous(
            thermo,
            trio,
            pressure,
            starts=ternary_starts,
            check_lle=include_heterogeneous,
        ))
        if include_heterogeneous and (binary_hetero_roots or hetero_starts > 0):
            results.extend(find_ternary_heterogeneous(
                thermo,
                trio,
                pressure,
                starts=hetero_starts,
                binary_hetero_roots=binary_hetero_roots,
            ))
    return {
        'case': name,
        'method': method,
        'pressure_bar': pressure,
        'components': components,
        'elapsed_s': time.perf_counter() - case_start,
        'results': [result.as_dict() for result in results],
    }


def default_cases() -> list[tuple[str, list[str], str, float]]:
    return [
        ('water_methanol_benzene', ['water', 'methanol', 'benzene'], 'UNIFAC', 1.01325),
        ('water_acetonitrile_acrylonitrile', ['water', 'acetonitrile', 'acrylonitrile'], 'UNIFNIST', 1.01325),
        ('water_ethanol_cyclohexane', ['water', 'ethanol', 'cyclohexane'], 'UNIFNIST', 1.01325),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ternary-starts', type=int, default=50)
    parser.add_argument('--hetero-starts', type=int, default=80)
    parser.add_argument(
        '--vle-only',
        action='store_true',
        help='Skip LLE/VLLE searches and return only homogeneous VLE azeotropes.',
    )
    args = parser.parse_args()

    for case in default_cases():
        name, components, method, pressure = case
        print(f'CASE {name} method={method} P={pressure:g} bar components={components}', flush=True)
        try:
            result = run_case(
                name,
                components,
                method,
                pressure,
                args.ternary_starts,
                args.hetero_starts,
                include_heterogeneous=not args.vle_only,
            )
            print(result, flush=True)
        except Exception as exc:
            print({'case': name, 'status': 'error', 'error': repr(exc)}, flush=True)


if __name__ == '__main__':
    main()
