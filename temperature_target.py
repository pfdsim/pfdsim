"""Shared memoized local search for temperature-target property solves."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from scipy.optimize import brentq


def solve_temperature_residual(
    residual: Callable[[float], float],
    *,
    T_guess: float,
    centers: Iterable[float],
    label: str,
    exact_tolerance: float,
    grid_builder: Callable[[tuple[float, ...]], list[float]],
    grid_solver: Callable[
        [Callable[[float], float], list[float], float, str, float],
        float,
    ],
) -> float:
    """Solve a temperature residual locally before invoking a grid fallback.

    Residual values are retained for the duration of the solve, so endpoint
    checks, Brent iterations, and the fallback grid never repeat an identical
    property evaluation.
    """
    clean_centers = []
    for value in centers:
        try:
            temperature = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(temperature) and temperature > 0.0:
            clean_centers.append(max(1.0, min(5000.0, temperature)))
    if not clean_centers:
        clean_centers.append(max(1.0, min(5000.0, float(T_guess))))

    residual_cache: dict[float, float] = {}

    def cached_residual(temperature: float) -> float:
        key = round(float(temperature), 10)
        if key not in residual_cache:
            residual_cache[key] = float(residual(float(temperature)))
        return residual_cache[key]

    for center in clean_centers:
        try:
            center_residual = cached_residual(center)
            if (
                math.isfinite(center_residual)
                and abs(center_residual) <= exact_tolerance
            ):
                return center
        except Exception:
            pass
        for width in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0):
            low = max(1.0, center - width)
            high = min(5000.0, center + width)
            if high <= low:
                continue
            try:
                low_residual = cached_residual(low)
                if (
                    math.isfinite(low_residual)
                    and abs(low_residual) <= exact_tolerance
                ):
                    return low
                high_residual = cached_residual(high)
                if (
                    math.isfinite(high_residual)
                    and abs(high_residual) <= exact_tolerance
                ):
                    return high
                if (
                    math.isfinite(low_residual)
                    and math.isfinite(high_residual)
                    and low_residual * high_residual < 0.0
                ):
                    return brentq(
                        cached_residual,
                        low,
                        high,
                        xtol=1e-7,
                        rtol=1e-9,
                        maxiter=80,
                    )
            except Exception:
                continue

    grid = grid_builder(tuple(clean_centers))
    return grid_solver(
        cached_residual,
        grid,
        clean_centers[0],
        label,
        exact_tolerance,
    )
