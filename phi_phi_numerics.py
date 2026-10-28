"""Authoritative array arithmetic for EOS equilibrium, optionally compiled.

Compiled and interpreted callers execute these same numerical functions.
Thermodynamic phase callbacks are supplied by the owning property model.
Higher-order algorithms are inlined into fixed model entry points; only
numerical-argument signatures enter the persistent Numba cache.
"""

import math
import numpy as np

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .compiled_cache import numba_cached
else:
    from compiled_cache import numba_cached

try:
    from numba import njit, typeof
    from numba.extending import register_jitable
except Exception:  # Match the existing optional-backend import policy.
    njit = typeof = None


def kernel(function=None, *, cache=True, dependencies=()):
    if function is None:
        return lambda implementation: kernel(implementation, cache=cache, dependencies=dependencies)
    if njit is None:
        return function
    if not cache:
        # Inline the shared higher-order source into fixed model entry points.
        # A separately compiled generic dispatcher retains a runtime function
        # pointer, making the caller uncacheable even with numeric arguments.
        return register_jitable(inline='always')(function)
    return numba_cached(njit, dependencies=dependencies)(function)


def interpreted(function):
    return getattr(function, 'py_func', function)


@kernel
def cubic_root_kind_code(attraction, volume_over_b, critical_volume, critical_attraction):
    if attraction <= critical_attraction*(1+1e-12):
        return 0
    return 1 if volume_over_b < critical_volume else 2


@kernel
def phase_admissible(kind, phase):
    return kind >= 0 and kind != (2 if phase == 0 else 1)


@kernel
def preferred_phase(liquid_g, vapor_g, liquid_valid, vapor_valid, same_root):
    if liquid_valid and vapor_valid:
        return 1 if same_root or vapor_g <= liquid_g else 0
    if liquid_valid:
        return 0
    return 1 if vapor_valid else -1


@kernel(cache=False)
def pure_cubic_phase(state, data, T, P, z, component):
    """Cubic branch identity establishes whether the selectors share a root."""
    kl, pl = state(T, P, z, 0, data)
    kv, pv = state(T, P, z, 1, data)
    liquid = phase_admissible(kl, 0) and math.isfinite(pl[component])
    vapor = phase_admissible(kv, 1) and math.isfinite(pv[component])
    return preferred_phase(pl[component], pv[component], liquid, vapor, kl == kv == 0)

@kernel
def rachford_rice(z, K, V_guess=0.5):

    def residual(beta):
        value = 0.0
        for i in range(len(z)):
            term = K[i] - 1.0
            denom = 1.0 + beta * term
            if abs(denom) < 1e-14:
                denom = 1e-14 if denom >= 0 else -1e-14
            value += z[i] * term / denom
        return value
    if residual(0.0) <= 0:
        return 0.0
    if residual(1.0) >= 0:
        return 1.0
    beta = max(0.0, min(1.0, V_guess))
    for _ in range(20):
        value = residual(beta)
        if abs(value) < 1e-12:
            return beta
        derivative = 0.0
        for i in range(len(z)):
            term = K[i] - 1.0
            denom = 1.0 + beta * term
            if abs(denom) < 1e-14:
                denom = 1e-14 if denom >= 0 else -1e-14
            derivative -= z[i] * term * term / (denom * denom)
        if abs(derivative) < 1e-14:
            break
        updated = beta - value / derivative
        if not 0 < updated < 1:
            break
        if abs(updated - beta) < 1e-12:
            return updated
        beta = updated
    low, high = (0.0, 1.0)
    for _ in range(100):
        beta = 0.5 * (low + high)
        value = residual(beta)
        if abs(value) < 1e-13 or high - low < 1e-13:
            return beta
        if value > 0:
            low = beta
        else:
            high = beta
    return 0.5 * (low + high)


@kernel
def phase_compositions(z, K, beta):
    denom = 1.0+beta*(K-1.0)
    for i in range(len(z)):
        if abs(denom[i]) < 1e-14:
            denom[i] = 1e-14 if denom[i] >= 0 else -1e-14
    x = z/denom
    y = K*x
    sx, sy = x.sum(), y.sum()
    if sx > 0:
        x /= sx
    if sy > 0:
        y /= sy
    return x, y


@kernel(cache=False)
def split_iterations(state, data, T, P, z, log_K, max_iter, tol):
    """Return status (0=budget, 1=invalid, 2=converged) and best log K."""
    point, best, previous = (log_K.copy(), log_K.copy(), log_K.copy())
    best_residual, previous_residual, damping, has_previous = (math.inf, math.inf, 1.0, False)
    for _ in range(max_iter):
        K = np.exp(np.minimum(690.0, np.maximum(-690.0, point)))
        beta = rachford_rice(z, K)
        x, y = phase_compositions(z, K, beta)
        kl, pl = state(T, P, x, 0, data)
        kv, pv = state(T, P, y, 1, data)
        valid = phase_admissible(kl, 0) and phase_admissible(kv, 1)
        for i in range(len(z)):
            if z[i] > 0 and (not math.isfinite(pl[i]) or not math.isfinite(pv[i])):
                valid = False
        if not valid:
            if has_previous and damping == 1.0:
                point = 0.5 * (previous + point)
                damping = 0.5
                continue
            return (1, best)
        changes = np.zeros(len(z))
        residual = 0.0
        for i in range(len(z)):
            if z[i] > 0:
                changes[i] = pl[i] - pv[i] - point[i]
                residual = max(residual, abs(changes[i]))
        if residual < best_residual:
            best, best_residual = (point.copy(), residual)
        if residual < tol:
            return (2, point)
        if residual > previous_residual or (has_previous and residual > 0.8 * previous_residual and (np.sum(changes * (point - previous)) < 0)):
            damping = 0.5
        previous, previous_residual, has_previous = (point, residual, True)
        point = np.minimum(690.0, np.maximum(-690.0, point + damping * changes))
    return (0, best)

@kernel
def boundary_composition(values, z, active):
    logits = np.zeros(len(active))
    logits[:-1] = values[1:]
    shift = np.max(logits)
    norm = shift + math.log(np.exp(logits - shift).sum())
    log_other = logits - norm
    other = np.zeros(len(z))
    for i in range(len(active)):
        other[active[i]] = math.exp(log_other[i])
    return other, log_other


@kernel(cache=False)
def boundary_equations(state, data, values, z, active, fixed, bubble, variable_T):
    value = math.exp(values[0])
    T, P = (value, fixed) if variable_T else (fixed, value)
    other, log_other = boundary_composition(values, z, active)
    x, y = (z, other) if bubble else (other, z)
    _, pl = state(T, P, x, 0, data)
    _, pv = state(T, P, y, 1, data)
    result = np.zeros(len(active))
    for i in range(len(active)):
        j = active[i]
        difference = math.log(z[j]) - log_other[i]
        result[i] = (difference if bubble else -difference) + pl[j] - pv[j]
    return result


@kernel(cache=False)
def boundary_samples(residual, data, values, z, active, fixed, bubble, variable_T, low, high):
    """Batch bound-aware forward differences at fixed prepared temperature."""
    f0 = residual(data, values, z, active, fixed, bubble, variable_T)
    samples = np.zeros((len(values), len(values)))
    steps = np.zeros(len(values))
    for i in range(len(values)):
        h = 2.**-26*max(1., abs(values[i]))*(1. if values[i] >= 0 else -1.)
        if values[i]+h > high[i] or values[i]+h < low[i]:
            h = -h
        perturbed = values.copy()
        perturbed[i] += h
        steps[i] = perturbed[i]-values[i]
        # A temperature perturbation needs freshly prepared interaction data.
        # The owning solver supplies this column through its full callback.
        if i == 0 and variable_T:
            continue
        samples[:, i] = residual(data, perturbed, z, active, fixed, bubble, variable_T)
    return f0, samples, steps


@kernel
def jacobian_from_samples(f0, samples, steps):
    return (samples-f0[:, None])/steps[None, :]

@kernel(cache=False)
def tpd_search(state, data, T, P, z, reference_phi, seed, phase, max_iter, tol):
    """Status: 0=budget, 1=invalid branch, 2=stationary, 3=negative TPD."""
    w, previous = (seed.copy(), seed.copy())
    d = np.zeros(len(z))
    for i in range(len(z)):
        if z[i] > 0:
            d[i] = math.log(z[i]) + reference_phi[i]
    damping, previous_change, has_previous = (1.0, math.inf, False)
    for _ in range(max_iter):
        kind, phi = state(T, P, w, phase, data)
        valid = phase_admissible(kind, phase)
        for i in range(len(z)):
            if z[i] > 0 and not math.isfinite(phi[i]):
                valid = False
        if not valid:
            if has_previous and damping == 1.0:
                w = 0.5 * (previous + w)
                damping = 0.5
                continue
            return (1, 0.0, phi)
        tpd = 0.0
        for i in range(len(z)):
            if w[i] > 0 and z[i] > 0:
                tpd += w[i] * (math.log(w[i]) + phi[i] - d[i])
        if tpd < -max(1e-12, 10 * tol):
            return (3, tpd, phi)
        log_w = d - phi
        shift = -math.inf
        for i in range(len(z)):
            if z[i] > 0:
                shift = max(shift, log_w[i])
        target = np.zeros(len(z))
        for i in range(len(z)):
            if z[i] > 0:
                target[i] = math.exp(log_w[i] - shift)
        target /= target.sum()
        change = np.max(np.abs(target - w))
        if change < tol:
            return (2, tpd, phi)
        dot = np.sum((target - w) * (w - previous))
        if change > previous_change or (has_previous and change > 0.8 * previous_change and (dot < 0)):
            damping = 0.5
        previous, previous_change, has_previous = (w, change, True)
        w = target if damping == 1.0 else 0.5 * (w + target)
    return (0, 0.0, np.zeros(len(z)))


def compile_kernels(state, data, component_count):
    """Load specializations by signature without evaluating a fluid state."""
    if typeof is None:
        return
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .compiled_phi_phi import kernels_for
    else:
        from compiled_phi_phi import kernels_for
    bound = kernels_for(state)
    x = np.zeros(component_count, dtype=np.float64)
    matrix = np.zeros((component_count, component_count), dtype=np.float64)
    for function, arguments in (
        (bound.split_iterations, (data, 300., 1., x, x, 100, 1e-9)),
        (bound.tpd_search, (data, 300., 1., x, x, x, 0, 100, 1e-9)),
        (bound.boundary_equations, (data, x, x, np.arange(component_count), 1., True, True)),
        (bound.pure_cubic_phase, (data, 300., 1., x, 0)),
        (bound.boundary_samples, (data, x, x, np.arange(component_count),
                            1., True, True, x, x)),
        (boundary_composition, (x, x, np.arange(component_count))),
        (jacobian_from_samples, (x, matrix, x)),
        (cubic_root_kind_code, (1., 1., 1., 1.)),
        (preferred_phase, (1., 1., True, True, True)),
        (rachford_rice, (x, x, .5)),
        (phase_compositions, (x, x, .5)),
    ):
        function.compile(tuple(typeof(argument) for argument in arguments))
