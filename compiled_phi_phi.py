"""Prepared native EOS callbacks for the shared equilibrium numerics."""

import numpy as np
from collections import namedtuple

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from . import compiled_cubic_eos as cubic
    from . import compiled_psrk as psrk
    from . import compiled_mhv2 as mhv2
    from .phi_phi_numerics import kernel, cubic_root_kind_code
    from . import phi_phi_numerics as numerics
else:
    import compiled_cubic_eos as cubic
    import compiled_psrk as psrk
    import compiled_mhv2 as mhv2
    from phi_phi_numerics import kernel, cubic_root_kind_code
    import phi_phi_numerics as numerics

@kernel
def phase_cubic(T, P, x, phase, data):
    kij, Tc, omega, a0, bi, modes, c1, c2, c3, d1, d2, vc, ac = data
    a, b, _, _, _ = cubic._cubic_mixture_numba(T, x, kij, np.zeros_like(kij), Tc, omega, a0, bi, modes, c1, c2, c3, d1)
    roots, count = cubic._roots_from_mixture_numba(T, P, a, b, d1, d2)
    z = 1.0 if count == 0 else roots[0 if phase == 0 else count - 1]
    values = cubic._cubic_fugacity_numba(T, P, x, phase, kij, Tc, omega, a0, bi, modes, c1, c2, c3, d1, d2)
    kind = cubic_root_kind_code(a/(b*cubic.R_CM3*T), z*cubic.R_CM3*T/(b*P), vc, ac)
    return (kind, np.log(values))

@kernel
def phase_psrk(T, P, x, phase, data):
    tail, vc, ac = data
    _, _, b, D, _, _ = psrk._psrk_mixture_parameters_numba(T, x, *tail)
    B = b * P / (tail[-1] * T)
    roots, count = psrk._compressibility_roots_numba(D * B, B)
    if count == 0:
        return (-1, np.full(len(x), np.nan))
    z = roots[0 if phase == 0 else count - 1]
    values = psrk._psrk_fugacity_numba(T, P, x, phase, *tail)
    if not np.isfinite(values).all() or (values <= 0).any():
        return (-1, np.full(len(x), np.nan))
    kind = cubic_root_kind_code(D, z/B, vc, ac)
    return (kind, np.log(values))

@kernel
def phase_mhv2(T, P, x, phase, data):
    common, vc, ac = data
    a, b, _, _, _ = mhv2._mhv2_mechanical_numba(T, x, common)
    roots, count = cubic._roots_from_mixture_numba(T, P, a, b, common[19], common[20])
    if count == 0:
        return (-1, np.full(len(x), np.nan))
    z = roots[0 if phase == 0 else count - 1]
    values, _ = mhv2._mhv2_fugacity_numba(T, P, x, phase, common)
    kind = cubic_root_kind_code(a/(b*cubic.R_CM3*T), z*cubic.R_CM3*T/(b*P), vc, ac)
    return (kind, np.log(values))


# Cache only model-bound entry points with numerical arguments. Caching the
# generic algorithms by Dispatcher-valued signature produces mortal weakrefs
# in Numba's disk index and causes both reload misses and serialization errors.
# These wrappers delegate to the single authoritative numerical implementation.
# Fingerprint the shared source as well as the owning module: Numba does not
# otherwise invalidate these cached callers when imported arithmetic changes.
_model_kernel = kernel(dependencies=(numerics.split_iterations,))


@_model_kernel
def cubic_split(data, T, P, z, log_K, max_iter, tol):
    return numerics.split_iterations(phase_cubic, data, T, P, z, log_K, max_iter, tol)


@_model_kernel
def psrk_split(data, T, P, z, log_K, max_iter, tol):
    return numerics.split_iterations(phase_psrk, data, T, P, z, log_K, max_iter, tol)


@_model_kernel
def mhv2_split(data, T, P, z, log_K, max_iter, tol):
    return numerics.split_iterations(phase_mhv2, data, T, P, z, log_K, max_iter, tol)


@_model_kernel
def cubic_tpd(data, T, P, z, reference_phi, seed, phase, max_iter, tol):
    return numerics.tpd_search(phase_cubic, data, T, P, z, reference_phi, seed, phase, max_iter, tol)


@_model_kernel
def psrk_tpd(data, T, P, z, reference_phi, seed, phase, max_iter, tol):
    return numerics.tpd_search(phase_psrk, data, T, P, z, reference_phi, seed, phase, max_iter, tol)


@_model_kernel
def mhv2_tpd(data, T, P, z, reference_phi, seed, phase, max_iter, tol):
    return numerics.tpd_search(phase_mhv2, data, T, P, z, reference_phi, seed, phase, max_iter, tol)


@_model_kernel
def cubic_boundary(data, values, z, active, fixed, bubble, variable_T):
    return numerics.boundary_equations(phase_cubic, data, values, z, active, fixed, bubble, variable_T)


@_model_kernel
def psrk_boundary(data, values, z, active, fixed, bubble, variable_T):
    return numerics.boundary_equations(phase_psrk, data, values, z, active, fixed, bubble, variable_T)


@_model_kernel
def mhv2_boundary(data, values, z, active, fixed, bubble, variable_T):
    return numerics.boundary_equations(phase_mhv2, data, values, z, active, fixed, bubble, variable_T)


@_model_kernel
def cubic_samples(data, values, z, active, fixed, bubble, variable_T, low, high):
    return numerics.boundary_samples(cubic_boundary, data,
        values, z, active, fixed, bubble, variable_T, low, high)


@_model_kernel
def psrk_samples(data, values, z, active, fixed, bubble, variable_T, low, high):
    return numerics.boundary_samples(psrk_boundary, data,
        values, z, active, fixed, bubble, variable_T, low, high)


@_model_kernel
def mhv2_samples(data, values, z, active, fixed, bubble, variable_T, low, high):
    return numerics.boundary_samples(mhv2_boundary, data,
        values, z, active, fixed, bubble, variable_T, low, high)


@_model_kernel
def cubic_pure(data, T, P, z, component):
    return numerics.pure_cubic_phase(phase_cubic, data, T, P, z, component)


@_model_kernel
def psrk_pure(data, T, P, z, component):
    return numerics.pure_cubic_phase(phase_psrk, data, T, P, z, component)


@_model_kernel
def mhv2_pure(data, T, P, z, component):
    return numerics.pure_cubic_phase(phase_mhv2, data, T, P, z, component)


PhaseKernels = namedtuple('PhaseKernels',
    'split_iterations tpd_search boundary_equations boundary_samples pure_cubic_phase')
_KERNELS = {
    phase_cubic: PhaseKernels(cubic_split, cubic_tpd, cubic_boundary, cubic_samples, cubic_pure),
    phase_psrk: PhaseKernels(psrk_split, psrk_tpd, psrk_boundary, psrk_samples, psrk_pure),
    phase_mhv2: PhaseKernels(mhv2_split, mhv2_tpd, mhv2_boundary, mhv2_samples, mhv2_pure),
}


def kernels_for(state):
    return _KERNELS[state]


def prepare(thermo, T, critical_geometry, *, record_warnings=True):
    """Specialize only property models whose actual methods match the kernel.

    Subclasses and instance overrides of the thermodynamic callbacks remain
    authoritative: unsupported callbacks use the same interpreted numerics.
    """
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .thermodynamics_models.eos import PhiPhiThermodynamics, CubicEOSThermodynamics, RKThermodynamics
        from .thermodynamics_models.psrk_thermo import PSRKThermodynamics
    else:
        from thermodynamics_models.eos import PhiPhiThermodynamics, CubicEOSThermodynamics, RKThermodynamics
        from thermodynamics_models.psrk_thermo import PSRKThermodynamics
    if isinstance(thermo, CubicEOSThermodynamics):
        owner, adapter = thermo.cubic, CubicEOSThermodynamics
    elif isinstance(thermo, RKThermodynamics):
        owner, adapter = thermo.rk, RKThermodynamics
    elif isinstance(thermo, PSRKThermodynamics):
        owner, adapter = thermo.psrk, PSRKThermodynamics
    else:
        return None
    for name, expected in (
        ('fugacity_coefficients', adapter.fugacity_coefficients),
        ('_eos_root_kind', PhiPhiThermodynamics._eos_root_kind),
        ('_eos_reduced_parameters', adapter._eos_reduced_parameters),
        ('compressibility_factor', adapter.compressibility_factor),
        ('molar_volume', adapter.molar_volume),
    ):
        if getattr(getattr(thermo, name), '__func__', None) is not expected:
            return None
    backend = owner._compiled_backend
    if backend is None:
        return None
    if record_warnings and hasattr(owner, '_compiled_temperature_warnings'):
        owner._compiled_temperature_warnings(T)
        thermo.extend_warnings(owner.warnings)
    key = float(T), id(backend)
    previous = getattr(thermo, '_phi_phi_native_plan', None)
    if previous is not None and previous[:2] == key:
        return previous[2:]
    if isinstance(backend, cubic.CompiledCubicEOSBackend):
        vc, ac = critical_geometry(backend.delta1, backend.delta2)
        data = (backend._interactions(T, None)[0], backend.Tc, backend.omega, backend.a0,
                backend.pure_b, backend.alpha_mode, backend.c1, backend.c2, backend.c3,
                backend.delta1, backend.delta2, vc, ac)
        state = phase_cubic
    elif isinstance(backend, psrk.CompiledPSRKBackend):
        vc, ac = critical_geometry(1., 0.)
        tail = tuple(getattr(backend, name) for name in (
            'nu', 'r', 'q', 'ell', 'subgroup_q', 'interaction_a', 'interaction_b', 'interaction_c',
            'Tc', 'omega', 'uses_mathias_copeman', 'c1', 'c2', 'c3', 'a0', 'pure_b', 'q1', 'gas_constant'))
        state, data = phase_psrk, (tail, vc, ac)
    elif isinstance(backend, mhv2.CompiledMHV2EOSBackend):
        vc, ac = critical_geometry(backend.delta1, backend.delta2)
        state, data = phase_mhv2, (backend._common(), vc, ac)
    else:
        return None
    thermo._phi_phi_native_plan = *key, state, data
    return state, data
