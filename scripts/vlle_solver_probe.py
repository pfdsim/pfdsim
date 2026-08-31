#!/usr/bin/env python3
"""Prototype fast TP VLLE flash solver.

This is intentionally a scratch script.  It tests the structured solver shape
before any of the logic moves into thermodynamics.py.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermodynamics import create_thermodynamics  # noqa: E402


@dataclass
class VLLEResult:
    phase_count: int
    vapor_fraction: float
    liquid1_fraction: float
    liquid2_fraction: float
    y: dict[str, float]
    x1: dict[str, float]
    x2: dict[str, float]
    residual: float
    iterations: int
    status: str


def normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(float(value), 0.0) for value in values.values())
    if total <= 0.0:
        return {key: 1.0 / len(values) for key in values}
    return {key: max(float(value), 0.0) / total for key, value in values.items()}


def max_comp_delta(a: dict[str, float], b: dict[str, float], comps: list[str]) -> float:
    return max(abs(a.get(comp, 0.0) - b.get(comp, 0.0)) for comp in comps)


def two_phase_flash(thermo, z: dict[str, float], T: float, P: float) -> VLLEResult:
    V, x, y = thermo.flash_TP(z, T, P)
    phase_count = 1 if V <= 1e-10 or V >= 1.0 - 1e-10 else 2
    return VLLEResult(
        phase_count=phase_count,
        vapor_fraction=V,
        liquid1_fraction=1.0 - V,
        liquid2_fraction=0.0,
        y=normalize(y),
        x1=normalize(x),
        x2=normalize(x),
        residual=0.0,
        iterations=0,
        status='ordinary_vle_or_single_phase',
    )


def solve_three_phase_rr(
    comps: list[str],
    z: dict[str, float],
    k1: dict[str, float],
    k2: dict[str, float],
    seed: tuple[float, float] = (0.35, 0.35),
) -> tuple[bool, float, float, dict[str, float], dict[str, float], dict[str, float], float]:
    """Solve three-phase RR with vapor as reference phase.

    Given K1 = y/x1 and K2 = y/x2, solve for L1 and L2.  V follows from
    V = 1 - L1 - L2.
    """

    z_array = np.asarray([max(float(z.get(comp, 0.0)), 0.0) for comp in comps], dtype=float)
    z_array = z_array / max(float(np.sum(z_array)), 1e-300)
    k1_array = np.asarray([max(float(k1.get(comp, 1.0)), 1e-14) for comp in comps], dtype=float)
    k2_array = np.asarray([max(float(k2.get(comp, 1.0)), 1e-14) for comp in comps], dtype=float)

    def compositions(l1_value: float, l2_value: float):
        v_value = 1.0 - l1_value - l2_value
        denom = v_value + l1_value / k1_array + l2_value / k2_array
        if np.any(denom <= 0.0):
            return None
        y_array = z_array / denom
        x1_array = y_array / k1_array
        x2_array = y_array / k2_array
        return y_array, x1_array, x2_array

    def residual(l1_value: float, l2_value: float) -> np.ndarray | None:
        comp_arrays = compositions(l1_value, l2_value)
        if comp_arrays is None:
            return None
        _, x1_array, x2_array = comp_arrays
        return np.array([
            np.sum(x1_array) - 1.0,
            np.sum(x2_array) - 1.0,
        ], dtype=float)

    l1, l2 = seed
    l1 = min(max(float(l1), 1e-10), 1.0 - 2e-10)
    l2 = min(max(float(l2), 1e-10), 1.0 - l1 - 1e-10)
    best = (float('inf'), l1, l2)

    for _ in range(30):
        f = residual(l1, l2)
        if f is None:
            break
        norm = float(np.linalg.norm(f, ord=np.inf))
        if norm < best[0]:
            best = (norm, l1, l2)
        if norm < 1e-10:
            break

        eps = 1e-6
        jac = np.zeros((2, 2), dtype=float)
        for j, (dl1, dl2) in enumerate(((eps, 0.0), (0.0, eps))):
            fp = residual(l1 + dl1, l2 + dl2)
            fm = residual(l1 - dl1, l2 - dl2)
            if fp is None or fm is None:
                break
            jac[:, j] = (fp - fm) / (2.0 * eps)
        else:
            try:
                step = np.linalg.solve(jac, -f)
            except np.linalg.LinAlgError:
                break

            damping = 1.0
            accepted = False
            for _line in range(20):
                trial_l1 = l1 + damping * float(step[0])
                trial_l2 = l2 + damping * float(step[1])
                if trial_l1 > 0.0 and trial_l2 > 0.0 and trial_l1 + trial_l2 < 1.0:
                    trial_f = residual(trial_l1, trial_l2)
                    if trial_f is not None and np.linalg.norm(trial_f, ord=np.inf) < norm:
                        l1, l2 = trial_l1, trial_l2
                        accepted = True
                        break
                damping *= 0.5
            if accepted:
                continue
            break

    norm, l1, l2 = best
    if not math.isfinite(norm) or norm > 1e-7:
        return False, 0.0, 0.0, {}, {}, {}, norm

    v = 1.0 - l1 - l2
    if min(v, l1, l2) <= 1e-8:
        return False, v, l1, {}, {}, {}, norm

    y_array, x1_array, x2_array = compositions(l1, l2)
    y = normalize({comp: float(y_array[i]) for i, comp in enumerate(comps)})
    x1 = normalize({comp: float(x1_array[i]) for i, comp in enumerate(comps)})
    x2 = normalize({comp: float(x2_array[i]) for i, comp in enumerate(comps)})
    return True, v, l1, y, x1, x2, norm


def fast_vlle_flash(thermo, z: dict[str, float], T: float, P: float,
                    max_iter: int = 30) -> VLLEResult:
    comps = list(z.keys())
    z = normalize(z)

    V, x_vle, y_vle = thermo.flash_TP(z, T, P)
    has_feed_lle, feed_x1, feed_x2, feed_beta = thermo.liquid_liquid_equilibrium(z, T)
    if has_feed_lle:
        feed_x1 = normalize(feed_x1)
        feed_x2 = normalize(feed_x2)
        if len(comps) == 2:
            k1 = thermo.K_values(T, P, feed_x1)
            k2 = thermo.K_values(T, P, feed_x2)
            degenerate = binary_vlle_if_invariant(
                thermo, comps, z, T, P, feed_x1, feed_x2, k1, k2
            )
            if degenerate is not None:
                return degenerate
        else:
            structured = structured_vlle_from_liquid_seeds(
                thermo, comps, z, T, P, feed_x1, feed_x2, feed_beta, V, max_iter
            )
            if structured is not None:
                return structured
    if V >= 1.0 - 1e-10:
        return two_phase_flash(thermo, z, T, P)

    liquid_seed = normalize(x_vle if V > 1e-10 else z)
    has_lle, x1, x2, beta = thermo.liquid_liquid_equilibrium(liquid_seed, T)
    if not has_lle:
        return two_phase_flash(thermo, z, T, P)

    x1 = normalize(x1)
    x2 = normalize(x2)
    seed_l2 = max(1e-4, min(0.9, (1.0 - V) * beta))
    seed_l1 = max(1e-4, min(0.9 - seed_l2, (1.0 - V) * (1.0 - beta)))
    last_residual = float('inf')

    for iteration in range(1, max_iter + 1):
        k1 = thermo.K_values(T, P, x1)
        k2 = thermo.K_values(T, P, x2)
        if len(comps) == 2:
            degenerate = binary_vlle_if_invariant(thermo, comps, z, T, P, x1, x2, k1, k2)
            if degenerate is not None:
                return degenerate
        ok, v, l1, y, new_x1, new_x2, rr_residual = solve_three_phase_rr(
            comps, z, k1, k2, seed=(seed_l1, seed_l2)
        )
        last_residual = rr_residual
        if not ok:
            return two_phase_flash(thermo, z, T, P)

        l2 = 1.0 - v - l1
        delta = max(
            max_comp_delta(x1, new_x1, comps),
            max_comp_delta(x2, new_x2, comps),
        )
        x1, x2 = new_x1, new_x2
        seed_l1, seed_l2 = l1, l2
        if delta < 1e-9 and rr_residual < 1e-9:
            if max_comp_delta(x1, x2, comps) < 1e-4:
                return two_phase_flash(thermo, z, T, P)
            return VLLEResult(
                phase_count=3,
                vapor_fraction=v,
                liquid1_fraction=l1,
                liquid2_fraction=l2,
                y=y,
                x1=x1,
                x2=x2,
                residual=max(delta, rr_residual),
                iterations=iteration,
                status='structured_vlle',
            )

    return VLLEResult(
        phase_count=0,
        vapor_fraction=0.0,
        liquid1_fraction=0.0,
        liquid2_fraction=0.0,
        y={},
        x1=x1,
        x2=x2,
        residual=last_residual,
        iterations=max_iter,
        status='not_converged',
    )


def structured_vlle_from_liquid_seeds(
    thermo,
    comps: list[str],
    z: dict[str, float],
    T: float,
    P: float,
    x1: dict[str, float],
    x2: dict[str, float],
    beta: float,
    vapor_seed: float,
    max_iter: int,
) -> VLLEResult | None:
    seed_v = min(max(float(vapor_seed), 0.05), 0.85)
    liquid_total = 1.0 - seed_v
    seed_l2 = max(1e-4, min(0.9, liquid_total * max(0.0, min(1.0, beta))))
    seed_l1 = max(1e-4, min(0.9 - seed_l2, liquid_total - seed_l2))
    last_residual = float('inf')

    for iteration in range(1, max_iter + 1):
        k1 = thermo.K_values(T, P, x1)
        k2 = thermo.K_values(T, P, x2)
        ok, v, l1, y, new_x1, new_x2, rr_residual = solve_three_phase_rr(
            comps, z, k1, k2, seed=(seed_l1, seed_l2)
        )
        last_residual = rr_residual
        if not ok:
            return None

        l2 = 1.0 - v - l1
        delta = max(
            max_comp_delta(x1, new_x1, comps),
            max_comp_delta(x2, new_x2, comps),
        )
        x1, x2 = new_x1, new_x2
        seed_l1, seed_l2 = l1, l2
        if delta < 1e-9 and rr_residual < 1e-9:
            if max_comp_delta(x1, x2, comps) < 1e-4:
                return None
            return VLLEResult(
                phase_count=3,
                vapor_fraction=v,
                liquid1_fraction=l1,
                liquid2_fraction=l2,
                y=y,
                x1=x1,
                x2=x2,
                residual=max(delta, rr_residual),
                iterations=iteration,
                status='structured_vlle_feed_lle_seed',
            )

    return VLLEResult(
        phase_count=0,
        vapor_fraction=0.0,
        liquid1_fraction=0.0,
        liquid2_fraction=0.0,
        y={},
        x1=x1,
        x2=x2,
        residual=last_residual,
        iterations=max_iter,
        status='feed_lle_seed_not_converged',
    )


def binary_vlle_if_invariant(
    thermo,
    comps: list[str],
    z: dict[str, float],
    T: float,
    P: float,
    x1: dict[str, float],
    x2: dict[str, float],
    k1: dict[str, float],
    k2: dict[str, float],
) -> VLLEResult | None:
    """Detect the binary invariant VLLE case.

    A binary three-phase TP flash has fixed phase compositions at the invariant
    point, but material balance leaves a family of phase amounts.  Enthalpy or a
    specified phase fraction is needed to choose one member of that family.
    """

    try:
        p1 = thermo.bubble_point_P(x1, T)
        p2 = thermo.bubble_point_P(x2, T)
    except Exception:
        return None
    pressure_error = max(abs(p1 - P), abs(p2 - P))
    if pressure_error > max(5e-5, 1e-4 * P):
        return None

    y1 = normalize({comp: x1[comp] * k1[comp] for comp in comps})
    y2 = normalize({comp: x2[comp] * k2[comp] for comp in comps})
    if max_comp_delta(y1, y2, comps) > 2e-4:
        return None
    y = normalize({comp: 0.5 * (y1[comp] + y2[comp]) for comp in comps})

    key = comps[0]
    z_key = z[key]
    y_key = y[key]
    x1_key = x1[key]
    x2_key = x2[key]
    denom = x1_key - x2_key
    if abs(denom) < 1e-12:
        return None

    # L1(V) = [z - V*y - (1-V)*x2] / (x1-x2).  Find the V interval where
    # V, L1, and L2 are all nonnegative.
    candidates = [0.0, 1.0]
    slope = (x2_key - y_key) / denom
    intercept = (z_key - x2_key) / denom
    if abs(slope) > 1e-14:
        candidates.append(-intercept / slope)
    slope_l2 = -1.0 - slope
    intercept_l2 = 1.0 - intercept
    if abs(slope_l2) > 1e-14:
        candidates.append(-intercept_l2 / slope_l2)

    valid_v = []
    for left, right in zip(sorted(candidates), sorted(candidates)[1:]):
        mid = 0.5 * (left + right)
        if mid < -1e-12 or mid > 1.0 + 1e-12:
            continue
        l1 = intercept + slope * mid
        l2 = 1.0 - mid - l1
        if min(mid, l1, l2) >= -1e-10:
            valid_v.extend([max(0.0, left), min(1.0, right)])
    valid_v = [value for value in valid_v if 0.0 <= value <= 1.0]
    if not valid_v:
        return None
    v_low = min(valid_v)
    v_high = max(valid_v)
    v = 0.5 * (v_low + v_high)
    l1 = intercept + slope * v
    l2 = 1.0 - v - l1
    if min(v, l1, l2) <= 1e-8:
        return None

    return VLLEResult(
        phase_count=3,
        vapor_fraction=v,
        liquid1_fraction=l1,
        liquid2_fraction=l2,
        y=y,
        x1=x1,
        x2=x2,
        residual=pressure_error,
        iterations=1,
        status=f'binary_invariant_vlle_amounts_underdetermined; V_range=[{v_low:.6g}, {v_high:.6g}]',
    )


def heteroazeotrope_temperature(thermo, z: dict[str, float], P: float) -> float | None:
    def residual(T: float) -> float:
        has_lle, x1, _x2, _beta = thermo.liquid_liquid_equilibrium(z, T)
        if not has_lle:
            return float('nan')
        return thermo.bubble_point_P(x1, T) - P

    samples = []
    for T in np.linspace(300.0, 390.0, 91):
        try:
            value = residual(float(T))
        except Exception:
            continue
        if math.isfinite(value):
            samples.append((float(T), value))

    for (t1, f1), (t2, f2) in zip(samples, samples[1:]):
        if abs(f1) < 1e-7:
            return t1
        if f1 * f2 < 0.0:
            return brentq(residual, t1, t2, xtol=1e-9, rtol=1e-10, maxiter=80)
    return None


def print_result(label: str, result: VLLEResult, elapsed_ms: float) -> None:
    print(f"\n{label}")
    print(f"  status: {result.status}")
    print(f"  phase_count: {result.phase_count}")
    print(f"  fractions: V={result.vapor_fraction:.8f}, L1={result.liquid1_fraction:.8f}, L2={result.liquid2_fraction:.8f}")
    print(f"  residual: {result.residual:.3e}; iterations: {result.iterations}; time: {elapsed_ms:.3f} ms")
    print(f"  y : {result.y}")
    print(f"  x1: {result.x1}")
    print(f"  x2: {result.x2}")


def main() -> None:
    components = ['water', 'chloroform']
    z = {'water': 0.5, 'chloroform': 0.5}
    P = 1.01325
    thermo = create_thermodynamics(components, 'UNIFNIST')

    # Warm the compiled/cached activity and LLE paths.
    for _ in range(3):
        thermo.liquid_liquid_equilibrium(z, 330.0)
        thermo.flash_TP(z, 330.0, P)

    T_vlle = heteroazeotrope_temperature(thermo, z, P)
    print("Water/chloroform UNIFNIST TP VLLE probe")
    print(f"  pressure: {P:g} bar")
    print(f"  feed z: {z}")
    if T_vlle is None:
        print("  no heteroazeotrope bracket found in 300-390 K")
        temperatures = [320.0, 330.0, 340.0]
    else:
        print(f"  estimated heteroazeotrope T: {T_vlle:.5f} K ({T_vlle - 273.15:.5f} C)")
        temperatures = [T_vlle - 2.0, T_vlle, T_vlle + 2.0]

    for T in temperatures:
        start = time.perf_counter()
        result = fast_vlle_flash(thermo, z, T, P)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print_result(f"T = {T:.5f} K ({T - 273.15:.5f} C)", result, elapsed_ms)

    if T_vlle is not None:
        # Timing after warmup at the true three-phase point.
        samples = []
        for _ in range(200):
            start = time.perf_counter()
            fast_vlle_flash(thermo, z, T_vlle, P)
            samples.append((time.perf_counter() - start) * 1000.0)
        print("\nTiming at estimated VLLE point after warmup")
        print(f"  mean: {sum(samples) / len(samples):.3f} ms")
        print(f"  median: {sorted(samples)[len(samples) // 2]:.3f} ms")


if __name__ == '__main__':
    main()
