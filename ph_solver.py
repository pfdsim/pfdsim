"""One damped caloric Newton solver, usable from Python and Numba."""

import math


def solve_caloric_temperature(evaluate, arguments, target, seed):
    """Return (T, H residual, evaluations, converged), never a false root.

    ``evaluate(T, *arguments)`` returns molar enthalpy and its temperature
    slope. Unsuccessful local solves are left to the caller's phase-aware
    bracket/flash machinery.
    """
    tolerance = max(1.0e-8, abs(target) * 1.0e-12)
    temperature = max(1.0, min(5000.0, seed))
    enthalpy, heat_capacity = evaluate(temperature, *arguments)
    evaluations = 1
    for _iteration in range(24):
        residual = enthalpy - target
        if not math.isfinite(residual):
            return temperature, residual, evaluations, False
        if abs(residual) <= tolerance:
            return temperature, residual, evaluations, True
        if not math.isfinite(heat_capacity) or abs(heat_capacity) < 1.0e-12:
            return temperature, residual, evaluations, False
        maximum_step = 0.35 * max(abs(temperature), 50.0)
        step = max(-maximum_step, min(maximum_step, residual / heat_capacity))
        accepted = False
        damping = 1.0
        for _trial in range(8):
            candidate = max(1.0, min(5000.0, temperature - damping * step))
            candidate_enthalpy, candidate_cp = evaluate(candidate, *arguments)
            evaluations += 1
            candidate_residual = candidate_enthalpy - target
            if math.isfinite(candidate_residual) and (
                abs(candidate_residual) <= abs(residual) * 0.9
                or abs(candidate_residual) <= tolerance
            ):
                temperature = candidate
                enthalpy, heat_capacity = candidate_enthalpy, candidate_cp
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            return temperature, residual, evaluations, False
    residual = enthalpy - target
    return temperature, residual, evaluations, abs(residual) <= tolerance
