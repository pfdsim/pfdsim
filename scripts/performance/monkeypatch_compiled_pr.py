"""Experimental monkeypatch benchmark for fixed-component compiled PR K-values.

This is deliberately not production integration. It compares fresh initialized
BioSTEAM column solves and separately measures scalar-versus-batched compiled
evaluation of the final 42 stage states.
"""

import math
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator import Simulator

from physical_constants import R_BAR_CM3_MOL_K

R = R_BAR_CM3_MOL_K
SQRT2 = math.sqrt(2.0)
D1 = 1.0 + SQRT2
D2 = 1.0 - SQRT2
DD = D1 - D2


@njit(cache=False)
def normalize(x):
    out = np.maximum(x, 0.0)
    total = np.sum(out)
    if total <= 0.0:
        out[:] = 1.0 / len(out)
    else:
        out /= total
    return out


@njit(cache=False)
def pure_a(T, Tc, omega, a0):
    result = np.empty(len(Tc))
    for i in range(len(Tc)):
        tr = max(T / Tc[i], 1e-12)
        m = 0.37464 + 1.54226 * omega[i] - 0.26992 * omega[i] * omega[i]
        factor = 1.0 + m * (1.0 - math.sqrt(tr))
        result[i] = a0[i] * factor * factor
    return result


@njit(cache=False)
def pure_a_and_derivative(T, Tc, omega, a0):
    values = np.empty(len(Tc))
    derivatives = np.empty(len(Tc))
    for i in range(len(Tc)):
        tr = max(T / Tc[i], 1e-12)
        sqrt_tr = math.sqrt(tr)
        m = 0.37464 + 1.54226 * omega[i] - 0.26992 * omega[i] * omega[i]
        factor = 1.0 + m * (1.0 - sqrt_tr)
        values[i] = a0[i] * factor * factor
        derivatives[i] = a0[i] * (-factor * m / (Tc[i] * sqrt_tr))
    return values, derivatives


@njit(cache=False)
def cubic_roots(A, B):
    u = D1 + D2
    w = D1 * D2
    ca = -(1.0 + B - u * B)
    cb = A + w * B * B - u * B - u * B * B
    cc = -(A * B + w * B * B + w * B * B * B)
    p = cb - ca * ca / 3.0
    q = 2.0 * ca * ca * ca / 27.0 - ca * cb / 3.0 + cc
    shift = ca / 3.0
    disc = (0.5 * q) ** 2 + (p / 3.0) ** 3
    roots = np.empty(3)
    count = 0
    if disc > 1e-14:
        sd = math.sqrt(disc)
        left = -0.5 * q + sd
        right = -0.5 * q - sd
        yl = math.copysign(abs(left) ** (1.0 / 3.0), left)
        yr = math.copysign(abs(right) ** (1.0 / 3.0), right)
        z = yl + yr - shift
        if z > B + 1e-12:
            roots[count] = z
            count += 1
    elif abs(p) < 1e-14:
        z = -shift
        if z > B + 1e-12:
            roots[count] = z
            count += 1
    else:
        arg = (3.0 * q / (2.0 * p)) * math.sqrt(-3.0 / p)
        arg = min(1.0, max(-1.0, arg))
        theta = math.acos(arg)
        radius = 2.0 * math.sqrt(-p / 3.0)
        for k in range(3):
            z = radius * math.cos((theta + 2.0 * math.pi * k) / 3.0) - shift
            if z > B + 1e-12:
                roots[count] = z
                count += 1
    if count == 0:
        return 0, 1.0, 1.0
    zmin = roots[0]
    zmax = roots[0]
    for i in range(1, count):
        zmin = min(zmin, roots[i])
        zmax = max(zmax, roots[i])
    return count, zmin, zmax


@njit(cache=False)
def mixture(T, x, Tc, omega, a0, b, kij):
    ai = pure_a(T, Tc, omega, a0)
    n = len(x)
    amix = 0.0
    bmix = 0.0
    sumaij = np.zeros(n)
    for i in range(n):
        bmix += x[i] * b[i]
        for j in range(n):
            aij = math.sqrt(ai[i] * ai[j]) * (1.0 - kij[i, j])
            amix += x[i] * x[j] * aij
            sumaij[i] += x[j] * aij
    return amix, bmix, ai, sumaij


@njit(cache=False)
def mixture_with_derivative(T, x, Tc, omega, a0, b, kij, dkij):
    ai, dai = pure_a_and_derivative(T, Tc, omega, a0)
    n = len(x)
    amix = 0.0
    damix = 0.0
    bmix = 0.0
    sumaij = np.zeros(n)
    for i in range(n):
        bmix += x[i] * b[i]
        for j in range(n):
            base = math.sqrt(ai[i] * ai[j])
            aij = base * (1.0 - kij[i, j])
            amix += x[i] * x[j] * aij
            sumaij[i] += x[j] * aij
            if ai[i] > 0.0 and ai[j] > 0.0:
                dbase = 0.5 * base * (dai[i] / ai[i] + dai[j] / ai[j])
                daij = dbase * (1.0 - kij[i, j]) - base * dkij[i, j]
                damix += x[i] * x[j] * daij
    return amix, damix, bmix, ai, sumaij


@njit(cache=False)
def fugacity(T, P, xraw, phase, Tc, Pc, omega, a0, b, kij):
    x = normalize(xraw.copy())
    amix, bmix, ai, sumaij = mixture(T, x, Tc, omega, a0, b, kij)
    phi = np.ones(len(x))
    if amix <= 0.0 or bmix <= 0.0:
        return phi, 1
    A = amix * P / (R * R * T * T)
    B = bmix * P / (R * T)
    count, zmin, zmax = cubic_roots(A, B)
    Z = zmax if phase == 1 else zmin
    logarg = (Z + D1 * B) / max(Z + D2 * B, 1e-16)
    alog = math.log(max(logarg, 1e-16))
    for i in range(len(x)):
        biob = b[i] / bmix
        attraction = 2.0 * sumaij[i] / amix - biob
        lnphi = biob * (Z - 1.0) - math.log(max(Z - B, 1e-16))
        lnphi -= A / max(B * DD, 1e-16) * attraction * alog
        phi[i] = max(math.exp(max(min(lnphi, 50.0), -50.0)), 1e-12)
    return phi, count


@njit(cache=False)
def departure_properties(T, P, xraw, phase, Tc, omega, a0, b, kij, dkij):
    x = normalize(xraw.copy())
    amix, damix, bmix, ai, sumaij = mixture_with_derivative(
        T, x, Tc, omega, a0, b, kij, dkij
    )
    if bmix <= 0.0:
        return 0.0, 0.0, 0.0
    A = amix * P / (R * R * T * T)
    B = bmix * P / (R * T)
    count, zmin, zmax = cubic_roots(A, B)
    Z = zmax if phase == 1 else zmin
    logarg = (Z + D1 * B) / max(Z + D2 * B, 1e-16)
    alog = math.log(max(logarg, 1e-16))
    h = R * T * (Z - 1.0)
    h += (T * damix - amix) / max(bmix * DD, 1e-16) * alog
    s = R * math.log(max(Z - B, 1e-16))
    s += damix / max(bmix * DD, 1e-16) * alog
    h *= 0.1
    s *= 0.1
    return h, s, h - T * s


@njit(cache=False)
def phi_phi(T, P, xraw, max_iter, Tc, Pc, omega, a0, b, kij):
    x = normalize(xraw.copy())
    n = len(x)
    K = np.empty(n)
    for i in range(n):
        exponent = 5.373 * (1.0 + omega[i]) * (1.0 - Tc[i] / T)
        K[i] = max(1e-8, min(1e8, Pc[i] / max(P, 1e-12) * math.exp(exponent)))
    phil, count = fugacity(T, P, x, 0, Tc, Pc, omega, a0, b, kij)
    if count < 2:
        return K
    y = normalize(x * K)
    for _ in range(max_iter):
        phiv, vcount = fugacity(T, P, y, 1, Tc, Pc, omega, a0, b, kij)
        if vcount < 2:
            return K
        knew = np.empty(n)
        for i in range(n):
            knew[i] = max(1e-8, min(1e8, phil[i] / max(phiv[i], 1e-12)))
        ynew = normalize(x * knew)
        error = 0.0
        for i in range(n):
            error = max(error, abs(ynew[i] - y[i]))
        if error < 1e-9:
            return knew
        K = 0.5 * K + 0.5 * knew
        y = ynew
    return K


@njit(cache=False)
def phi_phi_batch(Ts, Ps, X, max_iter, Tc, Pc, omega, a0, b, KIJ):
    result = np.empty_like(X)
    for state in range(len(Ts)):
        result[state] = phi_phi(
            Ts[state], Ps[state], X[state], max_iter,
            Tc, Pc, omega, a0, b, KIJ[state],
        )
    return result


path = str(ROOT / 'examples' / 'biosteam_mesh_hydrocarbon_distillation.pfd')


def eos_arrays(eos):
    components = list(eos.components)
    return (
        components,
        np.asarray([eos.params[c].Tc for c in components]),
        np.asarray([eos.params[c].Pc for c in components]),
        np.asarray([eos.params[c].omega for c in components]),
        np.asarray([eos.params[c].a0 for c in components]),
        np.asarray([eos.params[c].b for c in components]),
    )


def patch_eos(eos, compile_departure=False):
    components, Tc, Pc, omega, a0, b = eos_arrays(eos)

    def patched_phi_phi(self, T, P, liquid_composition, max_iter=80):
        x = np.asarray([liquid_composition.get(c, 0.0) for c in components])
        kij = np.asarray(self._pair_kij_values(float(T))).reshape(
            len(components), len(components)
        )
        values = phi_phi(
            float(T), float(P), x, int(max_iter),
            Tc, Pc, omega, a0, b, kij,
        )
        return {comp: float(values[i]) for i, comp in enumerate(components)}

    eos.phi_phi_K_values = MethodType(patched_phi_phi, eos)
    if compile_departure:
        def compiled_departures(self, T, P, composition, phase='vapor'):
            x = np.asarray([composition.get(c, 0.0) for c in components])
            kij = np.asarray(self._pair_kij_values(float(T))).reshape(
                len(components), len(components)
            )
            dkij = np.asarray(self._pair_dkij_dT_values(float(T))).reshape(
                len(components), len(components)
            )
            return departure_properties(
                float(T),
                float(P),
                x,
                1 if str(phase).lower() == 'vapor' else 0,
                Tc,
                omega,
                a0,
                b,
                kij,
                dkij,
            )

        def patched_enthalpy(self, T, P, composition, phase='vapor'):
            return float(compiled_departures(self, T, P, composition, phase)[0])

        def patched_entropy(self, T, P, composition, phase='vapor'):
            return float(compiled_departures(self, T, P, composition, phase)[1])

        def patched_gibbs(self, T, P, composition, phase='vapor'):
            return float(compiled_departures(self, T, P, composition, phase)[2])

        eos.departure_enthalpy = MethodType(patched_enthalpy, eos)
        eos.departure_entropy = MethodType(patched_entropy, eos)
        eos.departure_gibbs = MethodType(patched_gibbs, eos)


def initialized_simulator(compiled=False, compiled_departure=False):
    simulator = Simulator.from_file(path)
    simulator.initialize()
    if compiled:
        patch_eos(
            simulator.thermo.cubic,
            compile_departure=compiled_departure,
        )
    return simulator


# Seed objects provide immutable fixed-component arrays and warm only Numba's
# machine code. Every timed solve below receives a fresh Simulator and caches.
seed = initialized_simulator()
eos = seed.thermo.cubic
components, Tc, Pc, omega, a0, b = eos_arrays(eos)

# Warm JIT on an ordinary liquid-like column state.
xwarm = np.full(len(components), 1.0 / len(components))
kijwarm = np.asarray(eos._pair_kij_values(380.0)).reshape(len(components), len(components))
phi_phi(380.0, 3.6, xwarm, 80, Tc, Pc, omega, a0, b, kijwarm)
phi_phi_batch(
    np.asarray([380.0]), np.asarray([3.6]), np.asarray([xwarm]), 80,
    Tc, Pc, omega, a0, b, np.asarray([kijwarm]),
)
departure_properties(
    380.0,
    3.6,
    xwarm,
    0,
    Tc,
    omega,
    a0,
    b,
    kijwarm,
    np.zeros_like(kijwarm),
)

# Obtain independent reference solutions with fresh thermodynamic caches.
reference_times = []
reference_result = None
for _ in range(3):
    simulator = initialized_simulator()
    start = time.perf_counter()
    reference_result = simulator.run()
    reference_times.append(time.perf_counter() - start)
assert reference_result.converged and not reference_result.errors
perf = reference_result.units['COL-1'].performance

# Compare compiled and Python K values over final and perturbed stage states.
sample_indices = [0, 10, 20, 30, 41]
max_relative = 0.0
max_absolute = 0.0
max_departure_relative = 0.0
max_departure_absolute = 0.0
for stage in sample_indices:
    T = perf['stage_temperatures_C'][stage] + 273.15
    P = perf['stage_pressures_bar'][stage]
    composition = perf['stage_liquid_compositions'][stage]
    py = eos.phi_phi_K_values(T, P, composition)
    x = np.asarray([composition.get(c, 0.0) for c in components])
    kij = np.asarray(eos._pair_kij_values(T)).reshape(len(components), len(components))
    compiled = phi_phi(T, P, x, 80, Tc, Pc, omega, a0, b, kij)
    for i, comp in enumerate(components):
        max_absolute = max(max_absolute, abs(compiled[i] - py[comp]))
        max_relative = max(max_relative, abs(compiled[i] / max(py[comp], 1e-30) - 1.0))
    dkij = np.asarray(eos._pair_dkij_dT_values(T)).reshape(
        len(components), len(components)
    )
    for phase, phase_id in (('liquid', 0), ('vapor', 1)):
        python_values = (
            eos.departure_enthalpy(T, P, composition, phase),
            eos.departure_entropy(T, P, composition, phase),
            eos.departure_gibbs(T, P, composition, phase),
        )
        compiled_values = departure_properties(
            T, P, x, phase_id, Tc, omega, a0, b, kij, dkij
        )
        for python_value, compiled_value in zip(
            python_values, compiled_values
        ):
            difference = abs(compiled_value - python_value)
            max_departure_absolute = max(max_departure_absolute, difference)
            max_departure_relative = max(
                max_departure_relative,
                difference / max(abs(python_value), 1e-30),
            )

# Repeat with a fresh simulator and compiled K kernel for every solve.
compiled_times = []
compiled_result = None
for _ in range(3):
    simulator = initialized_simulator(compiled=True)
    start = time.perf_counter()
    compiled_result = simulator.run()
    compiled_times.append(time.perf_counter() - start)
assert compiled_result.converged and not compiled_result.errors

compiled_departure_times = []
compiled_departure_result = None
for _ in range(3):
    simulator = initialized_simulator(
        compiled=True,
        compiled_departure=True,
    )
    start = time.perf_counter()
    compiled_departure_result = simulator.run()
    compiled_departure_times.append(time.perf_counter() - start)
assert compiled_departure_result.converged and not compiled_departure_result.errors

# Full-result agreement at final product streams.
max_stream_relative = 0.0
for name in ('Distillate', 'Bottoms'):
    left = reference_result.streams[name]
    right = compiled_result.streams[name]
    for left_value, right_value in (
        (left.T, right.T),
        (left.F, right.F),
        (left.H, right.H),
    ):
        max_stream_relative = max(
            max_stream_relative,
            abs(right_value - left_value) / max(abs(left_value), 1e-30),
        )
    for comp in components:
        lv = left.composition.get(comp, 0.0)
        rv = right.composition.get(comp, 0.0)
        max_stream_relative = max(max_stream_relative, abs(rv - lv) / max(abs(lv), 1e-12))

max_departure_stream_relative = 0.0
for name in ('Distillate', 'Bottoms'):
    left = reference_result.streams[name]
    right = compiled_departure_result.streams[name]
    for left_value, right_value in (
        (left.T, right.T),
        (left.F, right.F),
        (left.H, right.H),
    ):
        max_departure_stream_relative = max(
            max_departure_stream_relative,
            abs(right_value - left_value) / max(abs(left_value), 1e-30),
        )
    for comp in components:
        lv = left.composition.get(comp, 0.0)
        rv = right.composition.get(comp, 0.0)
        max_departure_stream_relative = max(
            max_departure_stream_relative,
            abs(rv - lv) / max(abs(lv), 1e-12),
        )

# Replay the actual final 42 stage states to isolate Python dispatch vs batch.
Ts = np.asarray([value + 273.15 for value in perf['stage_temperatures_C']])
Ps = np.asarray(perf['stage_pressures_bar'])
X = np.asarray([
    [composition.get(comp, 0.0) for comp in components]
    for composition in perf['stage_liquid_compositions']
])
KIJ = np.asarray([
    np.asarray(eos._pair_kij_values(float(T))).reshape(len(components), len(components))
    for T in Ts
])
phi_phi_batch(Ts, Ps, X, 80, Tc, Pc, omega, a0, b, KIJ)

REPLAY = 100
start = time.perf_counter()
for _ in range(REPLAY):
    for stage in range(len(Ts)):
        phi_phi(Ts[stage], Ps[stage], X[stage], 80, Tc, Pc, omega, a0, b, KIJ[stage])
scalar_replay = time.perf_counter() - start
start = time.perf_counter()
for _ in range(REPLAY):
    phi_phi_batch(Ts, Ps, X, 80, Tc, Pc, omega, a0, b, KIJ)
batch_replay = time.perf_counter() - start

print('REFERENCE_TIMES', reference_times)
print('COMPILED_K_TIMES', compiled_times)
print('COMPILED_K_DEPARTURE_TIMES', compiled_departure_times)
print('REFERENCE_MEDIAN', float(np.median(reference_times)))
print('COMPILED_K_MEDIAN', float(np.median(compiled_times)))
print('COMPILED_K_DEPARTURE_MEDIAN', float(np.median(compiled_departure_times)))
print('FULL_SOLVE_SPEEDUP', float(np.median(reference_times) / np.median(compiled_times)))
print('FULL_SOLVE_SPEEDUP_WITH_DEPARTURE', float(np.median(reference_times) / np.median(compiled_departure_times)))
print('INCREMENTAL_DEPARTURE_SPEEDUP', float(np.median(compiled_times) / np.median(compiled_departure_times)))
print('K_MAX_RELATIVE_ERROR', max_relative)
print('K_MAX_ABSOLUTE_ERROR', max_absolute)
print('DEPARTURE_MAX_RELATIVE_ERROR', max_departure_relative)
print('DEPARTURE_MAX_ABSOLUTE_ERROR', max_departure_absolute)
print('STREAM_MAX_RELATIVE_ERROR', max_stream_relative)
print('DEPARTURE_STREAM_MAX_RELATIVE_ERROR', max_departure_stream_relative)
print('SCALAR_REPLAY_SECONDS', scalar_replay)
print('BATCH_REPLAY_SECONDS', batch_replay)
print('BATCH_SPEEDUP_OVER_COMPILED_SCALAR_DISPATCH', scalar_replay / batch_replay)
print('REFERENCE_SOLVER', reference_result.units['COL-1'].performance['solver_iterations'], reference_result.units['COL-1'].performance['function_evaluations'], reference_result.units['COL-1'].performance['jacobian_evaluations'])
print('COMPILED_SOLVER', compiled_result.units['COL-1'].performance['solver_iterations'], compiled_result.units['COL-1'].performance['function_evaluations'], compiled_result.units['COL-1'].performance['jacobian_evaluations'])
print('COMPILED_DEPARTURE_SOLVER', compiled_departure_result.units['COL-1'].performance['solver_iterations'], compiled_departure_result.units['COL-1'].performance['function_evaluations'], compiled_departure_result.units['COL-1'].performance['jacobian_evaluations'])
