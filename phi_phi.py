"""Shared composition iteration for phi-phi equilibrium property models."""

import math
from functools import lru_cache
from functools import partial

import numpy as np
from scipy.optimize import brentq
from scipy.optimize import least_squares
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from . import phi_phi_numerics as numerics
else:
    import phi_phi_numerics as numerics


def _spinodal_attraction(volume, delta1, delta2):
    return ((volume + delta1)**2 * (volume + delta2)**2
            / ((2 * volume + delta1 + delta2) * (volume - 1)**2))


@lru_cache(maxsize=2)
def _cubic_critical_volume(delta1, delta2):
    """Critical V/b and a/(bRT) from the minimum of the spinodal curve."""
    def slope(v):
        return (2 / (v + delta1) + 2 / (v + delta2)
                - 2 / (2 * v + delta1 + delta2) - 2 / (v - 1))
    volume = brentq(slope, 1 + 1e-8, 100.0)
    attraction = _spinodal_attraction(volume, delta1, delta2)
    return volume, attraction


def cubic_root_kind(attraction, volume_over_b, delta1=1.0, delta2=0.0):
    """Identify stable cubic branches without a heuristic Z cutoff.

    On subcritical isotherms the two spinodals straddle the critical V/b;
    stable liquid and vapor roots therefore lie on opposite sides of it.
    Above the mechanical critical temperature there is one continuous branch.
    """
    critical_volume, critical_attraction = _cubic_critical_volume(delta1, delta2)
    code = numerics.cubic_root_kind_code(attraction, volume_over_b, critical_volume, critical_attraction)
    return ('supercritical', 'liquid', 'vapor')[code]


def cubic_pressure_extrema(attraction, pressure_scale, delta1=1.0, delta2=0.0):
    """Spinodal pressures [bar], including negative liquid-side pressures."""
    critical_volume, critical_attraction = _cubic_critical_volume(delta1, delta2)
    if attraction <= critical_attraction:
        return ()

    def residual(v):
        return _spinodal_attraction(v, delta1, delta2) - attraction

    high = 2 * critical_volume
    while residual(high) < 0:
        high *= 2
    volumes = (brentq(residual, 1 + 1e-12, critical_volume, xtol=1e-14),
               brentq(residual, critical_volume, high, xtol=1e-14))
    return tuple(pressure_scale * (1/(v-1) - attraction/((v+delta1)*(v+delta2)))
                 for v in volumes)


def wilson_K(props, T, P, components):
    return {
        comp: max(1e-8, min(1e8, props[comp].Pc / max(P, 1e-12) * math.exp(
            5.373 * (1.0 + props[comp].omega) * (1.0 - props[comp].Tc / T))))
        for comp in components
    }


class PhiPhiFlashError(ValueError):
    """A constrained VLE flash could not establish a valid result."""


def _numerical_errors():
    # Backends expose their own exception types. Trial-state failures must
    # participate in the same recovery/error path as integration-layer errors.
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .cubic_eos import CubicEOSError
        from .rk_eos import RKError
        from .thermodynamics_models.common import ThermodynamicsError
        from .thermodynamics_models.psrk import PSRKError
    else:
        from cubic_eos import CubicEOSError
        from rk_eos import RKError
        from thermodynamics_models.common import ThermodynamicsError
        from thermodynamics_models.psrk import PSRKError
    return (ThermodynamicsError, CubicEOSError, RKError, PSRKError,
            ValueError, OverflowError, FloatingPointError)


def normalized_eos_composition(components, composition):
    """Validate an EOS feed before normalizing or dropping absent components."""
    if set(composition) - set(components):
        raise PhiPhiFlashError('EOS composition contains unknown components')
    try:
        fractions = [float(composition.get(c, 0.)) for c in components]
    except (TypeError, ValueError, OverflowError) as error:
        raise PhiPhiFlashError('EOS composition must be nonnegative and finite with a positive total') from error
    maximum = max(fractions, default=0.)
    if maximum <= 0 or any(not math.isfinite(value) or value < 0 for value in fractions):
        raise PhiPhiFlashError('EOS composition must be nonnegative and finite with a positive total')
    # Scaling first prevents finite component amounts from overflowing their
    # sum, while retaining the usual normalization for fractions and traces.
    fractions = [value/maximum for value in fractions]
    total = math.fsum(fractions)
    return {component: value/total for component, value in zip(components, fractions)}


def _admissible_phase_kind(root_kind, phase):
    return root_kind not in ('invalid', 'liquid' if phase == 'vapor' else 'vapor')


def _native_provider(thermo, T, *, record_warnings=True):
    if numerics.njit is None:
        return None
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .compiled_phi_phi import prepare
    else:
        from compiled_phi_phi import prepare
    return prepare(thermo, float(T), _cubic_critical_volume, record_warnings=record_warnings)


def initialize_phi_phi(thermo):
    native = _native_provider(thermo, 300., record_warnings=False)
    if native is not None:
        numerics.compile_kernels(*native, len(thermo.components))


def _bound_kernels(state):
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .compiled_phi_phi import kernels_for
    else:
        from compiled_phi_phi import kernels_for
    return kernels_for(state)


def _boundary_jacobian(thermo, values, z, active, fixed, bubble, variable_T,
                       bounds, equations, python_state):
    native = _native_provider(thermo, math.exp(values[0]) if variable_T else fixed)
    state, data = native if native is not None else (python_state, None)
    sampler = (_bound_kernels(state).boundary_samples if native is not None else
               partial(numerics.interpreted(numerics.boundary_samples),
                       partial(numerics.interpreted(numerics.boundary_equations), state)))
    f0, samples, steps = sampler(data, values, z, active, float(fixed),
                                 bubble, variable_T, bounds[0], bounds[1])
    if variable_T:
        # Do not freeze temperature-dependent kij (or other model data) in
        # dF/dlog(T). The full callback prepares the perturbed temperature.
        perturbed = values.copy()
        perturbed[0] += steps[0]
        samples[:, 0] = equations(perturbed)
    return numerics.jacobian_from_samples(f0, samples, steps)


def solve_phi_phi_flash(thermo, z, T, P, max_iter, tol):
    """VLE-only EOS flash, with root identity and opposite-phase stability.

    A cubic's largest root is not necessarily a vapor root. In particular,
    excess-Gibbs mixing rules can have two liquid compositions at equal
    fugacity. Those states must not be accepted by this VLE-only interface.
    """
    if not (math.isfinite(T) and math.isfinite(P) and T > 0 and P > 0):
        raise PhiPhiFlashError('EOS TP flash temperature/pressure must be positive and finite')
    z = normalized_eos_composition(thermo.components, z)
    active = [c for c in thermo.components if z[c] > 0.0]
    cache = getattr(thermo, '_phi_phi_flash_cache', None)
    if cache is None:
        cache = thermo._phi_phi_flash_cache = {}
    cache_key = (float(T), float(P), tuple(z.values()), max_iter, tol)
    cached = cache.get(cache_key)
    if cached is not None and cached[3].issubset(thermo._warning_keys):
        beta, x, y, _ = cached
        return beta, dict(x), dict(y)

    def complete(beta, x, y):
        # Retain independent dictionaries, and recompute if warning context
        # was cleared so a cache hit cannot suppress a physical diagnostic.
        thermo._set_limited_cache(cache, cache_key,
            (beta, dict(x), dict(y), frozenset(thermo._warning_keys)), limit=1024)
        return beta, x, y

    initial = wilson_K(thermo.props, T, P, thermo.components)
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .thermodynamics_models.common import ThermodynamicsValidityError
    else:
        from thermodynamics_models.common import ThermodynamicsValidityError
    numerical_errors = _numerical_errors()
    invalid_states = []
    kind_cache, phi_cache, phase_cache = {}, {}, {}

    def kind(composition, phase):
        key = (phase, tuple(composition[c] for c in thermo.components))
        if key in kind_cache:
            return kind_cache[key]
        try:
            value = thermo._eos_root_kind(T, P, composition, phase)
            kind_cache[key] = value
            return value
        except numerical_errors as error:
            # An invalid trial branch does not invalidate a different, stable
            # phase at this T/P. Preserve the error if no feed branch is valid.
            invalid_states[:] = [error]
            return 'invalid'

    def admissible(composition, phase):
        return _admissible_phase_kind(kind(composition, phase), phase)

    def log_phi(composition, phase):
        key = (phase, tuple(composition[c] for c in thermo.components))
        if key in phi_cache:
            return phi_cache[key]
        values = thermo.fugacity_coefficients(T, P, composition, phase)
        if any(not math.isfinite(values[c]) or values[c] <= 0 for c in active):
            raise PhiPhiFlashError('EOS TP flash encountered invalid fugacity coefficients')
        result = {c: math.log(values[c]) for c in active}
        phi_cache[key] = result
        return result

    array_z = np.array(list(z.values()))
    active_indices = np.flatnonzero(array_z > 0.)

    def python_state(temperature, pressure, values, phase, data):
        composition = dict(zip(thermo.components, values.tolist()))
        name = 'liquid' if phase == 0 else 'vapor'
        root = kind(composition, name)
        code = {'invalid': -1, 'liquid': 1, 'vapor': 2}.get(root, 0)
        if not numerics.phase_admissible(code, phase):
            return code, np.full(len(values), np.nan)
        try:
            logs = log_phi(composition, name)
        except numerical_errors as error:
            invalid_states[:] = [error]
            return -1, np.full(len(values), np.nan)
        return code, np.array([logs.get(c, 0.) for c in thermo.components])

    native = _native_provider(thermo, T)
    phase_state, phase_data = native if native is not None else (python_state, None)
    bound = _bound_kernels(phase_state) if native is not None else None
    if native is not None and len(active) == 1 and max_iter > 1:
        try:
            phase = bound.pure_cubic_phase(
                phase_data, float(T), float(P), array_z, int(active_indices[0]))
            if phase >= 0:
                return complete(float(phase), dict(z), dict(z))
        except numerical_errors as error:
            invalid_states[:] = [error]
    split_solver = (bound.split_iterations if bound is not None else
                    partial(numerics.interpreted(numerics.split_iterations), python_state))
    stability_solver = (bound.tpd_search if bound is not None else
                        partial(numerics.interpreted(numerics.tpd_search), python_state))

    def phases(log_K):
        key = tuple(log_K)
        if key in phase_cache:
            return phase_cache[key]
        K = dict(initial)
        K.update({c: math.exp(max(-690.0, min(690.0, value)))
                  for c, value in zip(active, log_K)})
        beta = thermo._rachford_rice_bounded(z, K)
        x, y = thermo._flash_phase_compositions(z, K, beta)
        result = beta, x, y
        phase_cache[key] = result
        return result

    def equations(log_K):
        _, x, y = phases(log_K)
        liquid, vapor = log_phi(x, 'liquid'), log_phi(y, 'vapor')
        return [liquid[c] - vapor[c] - value for c, value in zip(active, log_K)]

    def finish(log_K):
        beta, x, y = phases(log_K)
        if not admissible(x, 'liquid') or not admissible(y, 'vapor'):
            return None
        if not 0.0 < beta < 1.0:
            # Near K=1 the sign of an arbitrarily small iteration error can
            # choose either endpoint. Establish single-phase identity and
            # stability from the feed roots below, independently of that sign.
            return None
        material = max(abs((1-beta)*x[c] + beta*y[c] - z[c]) for c in active)
        if material > tol:
            raise PhiPhiFlashError(f'EOS TP flash failed its final material balance check: residual={material:g}')
        # The public audit also rejects a homogeneous K=1 "split".
        try:
            thermo._check_diagram_vle(x, y, T, P, fugacity_tolerance=tol)
        except numerical_errors:
            return None
        return beta, x, y

    def iterate(seed):
        log_K = np.log(np.array([seed.get(c, initial[c]) for c in thermo.components]))
        try:
            status, point = split_solver(phase_data, float(T), float(P),
                                        array_z, log_K, int(max_iter), float(tol))
        except numerical_errors as error:
            invalid_states[:] = [error]
            return None
        best = point[active_indices].tolist()
        if status == 1:
            return None
        if status == 2:
            return finish(best)
        if max_iter > 1:
            # Rescue slow/oscillating substitution from its best iterate.
            # Trial evaluations may cross a branch; only a physically valid
            # final root with independently checked balances is accepted.
            try:
                solution = least_squares(equations, best, bounds=(-690.0, 690.0),
                                         max_nfev=max_iter, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if max(abs(v) for v in equations(solution.x)) < tol:
                    return finish(solution.x)
            except numerical_errors as error:
                invalid_states[:] = [error]
                return None

        return None

    # Endpoint Wilson estimates are not phase decisions: go directly to the
    # feed-root and opposite-phase stability search instead of iterating a
    # homogeneous K=1 solution only to discard it. Instability seeds below
    # still run the fully audited split solver.
    result = None
    if len(active) > 1:
        initial_beta, _, _ = phases([math.log(initial[c]) for c in active])
        if 0.0 < initial_beta < 1.0:
            result = iterate(initial)
    if result is not None:
        return complete(*result)

    # Select the lower-Gibbs admissible feed root. This is a reference for
    # a constrained opposite-phase TPD search, not a global LLE calculation.
    feed = []
    for phase in ('liquid', 'vapor'):
        if admissible(z, phase):
            try:
                phi = log_phi(z, phase)
                feed.append((sum(z[c]*phi[c] for c in active), phase, phi))
            except numerical_errors as error:
                invalid_states[:] = [error]
    if not feed:
        if invalid_states:
            error = invalid_states[-1]
            if isinstance(error, ThermodynamicsValidityError):
                raise error
            raise PhiPhiFlashError(f'EOS TP flash has no valid feed root: {error}') from error
        raise PhiPhiFlashError('EOS TP flash has no admissible feed root')
    same_root = False
    if len(feed) == 2:
        liquid_volume = thermo.molar_volume(T, P, z, 'liquid')
        vapor_volume = thermo.molar_volume(T, P, z, 'vapor')
        same_root = abs(liquid_volume-vapor_volume) <= 1e-6 * max(liquid_volume, vapor_volume)
    # Identical single-root selectors have identical Gibbs energies. Use the
    # vapor/EOS-volume convention for that continuous supercritical branch;
    # a subcritical liquid root already excludes the vapor selector above.
    by_phase = {entry[1]: entry for entry in feed}
    phase_id = numerics.preferred_phase(
        by_phase.get('liquid', (math.inf,))[0], by_phase.get('vapor', (math.inf,))[0],
        'liquid' in by_phase, 'vapor' in by_phase, same_root)
    _, reference_phase, reference_phi = by_phase['vapor' if phase_id == 1 else 'liquid']
    # A pure fluid cannot form a distinct composition. The lower-Gibbs valid
    # feed root already establishes its equilibrium phase without K or TPD
    # composition iteration, including the supercritical volume convention.
    if len(active) == 1 and max_iter > 1:
        return complete(0.0 if reference_phase == 'liquid' else 1.0, dict(z), dict(z))
    trial_phase = 'vapor' if reference_phase == 'liquid' else 'liquid'

    def normalized(values):
        total = sum(values.values())
        return {c: values.get(c, 0.0) / total for c in thermo.components}

    wilson_trial = normalized({c: z[c] * (initial[c] if trial_phase == 'vapor'
                                        else 1 / initial[c]) for c in active})
    seeds = [wilson_trial, dict(z)]
    seeds.extend(normalized({c: 1e-4*z[c] + (1-1e-4 if c == main else 0.0)
                             for c in active}) for main in active)
    unstable = []
    unfinished = False
    reference_array = np.array([reference_phi.get(c, 0.) for c in thermo.components])
    for seed in seeds:
        try:
            status, tpd, phi = stability_solver(
                phase_data, float(T), float(P), array_z, reference_array,
                np.array([seed[c] for c in thermo.components]),
                int(trial_phase == 'vapor'), int(max_iter), float(tol))
        except numerical_errors as error:
            invalid_states[:] = [error]
            unfinished = True
            continue
        unfinished |= status == 0
        if status == 3:
            differences = reference_array-phi if reference_phase == 'liquid' else phi-reference_array
            K = dict(initial)
            K.update({c: math.exp(max(-690., min(690., differences[i])))
                      for i, c in enumerate(thermo.components) if z[c] > 0.})
            unstable.append((tpd, K))
    for _, seed in sorted(unstable, key=lambda item: item[0]):
        result = iterate(seed)
        if result is not None:
            return complete(*result)
    if unstable or unfinished or max_iter <= 1:
        raise PhiPhiFlashError(f'EOS TP flash did not converge after {max_iter} iterations per attempt')

    if reference_phase == 'liquid' and len(active) > 1:
        # Detect local liquid instability without pretending to solve LLE.
        # A negative Gibbs curvature proves instability; nonnegative sampled
        # directions do not prove global liquid stability.
        def gibbs(composition):
            phi = log_phi(composition, 'liquid')
            return sum(composition[c]*(math.log(composition[c])+phi[c]) for c in active)
        g0 = gibbs(z)
        for seed in seeds:
            direction = {c: seed[c]-z[c] for c in active}
            size = max(abs(v) for v in direction.values())
            if size < 1e-12:
                continue
            direction = {c: v/size for c, v in direction.items()}
            step = min(1e-4, *(z[c]/(4*abs(v)) for c, v in direction.items() if v))
            if step < 1e-5:
                # Trace species can force a step below the resolution of
                # this optional curvature diagnostic. Do not infer a sign
                # from cancellation (or divide by an underflowed step²).
                continue
            plus = {c: z[c]+step*direction.get(c, 0.) for c in thermo.components}
            minus = {c: z[c]-step*direction.get(c, 0.) for c in thermo.components}
            try:
                curvature = (gibbs(plus)+gibbs(minus)-2*g0) / step**2
            except numerical_errors:
                continue
            if curvature < -1e-4:
                thermo.add_warning(
                    'EOS VLE-constrained flash retained a locally unstable liquid; '
                    'EOS LLE/VLLE is not implemented. No vapor fraction is inferred '
                    'from the liquid instability.')
                break
    return complete(0.0 if reference_phase == 'liquid' else 1.0, dict(z), dict(z))


def solve_phi_phi_boundary(thermo, composition, fixed, guess, *, kind, variable):
    """Solve actual component fugacity equality for a bubble/dew T or P.

    The unknown phase uses normalized log-composition coordinates. Wilson
    values supply seeds only. Every accepted solution has distinct admissible
    phases and passes the same physical fugacity audit as a TP flash.
    """
    if not math.isfinite(fixed) or fixed <= 0:
        raise PhiPhiFlashError('EOS phase-boundary temperature/pressure must be positive and finite')
    if guess is not None and (not math.isfinite(guess) or guess <= 0):
        raise PhiPhiFlashError('EOS phase-boundary guess must be positive and finite')
    z = normalized_eos_composition(thermo.components, composition)
    fractions = tuple(z.values())
    active = [c for c in thermo.components if z[c] > 0]
    log_z = np.log([z[c] for c in active])
    array_z = np.array(list(z.values()))
    active_indices = np.flatnonzero(array_z > 0.)
    cache = getattr(thermo, '_phi_phi_boundary_cache', None)
    if cache is None:
        cache = thermo._phi_phi_boundary_cache = {}
    # A pure-fluid coexistence boundary is independent of its starting guess.
    # Preserve guess-specific searches for mixtures, which can have multiple
    # physical boundaries. All inputs were validated before this lookup.
    key = (kind, variable, float(fixed), guess if len(active) > 1 else None, tuple(fractions))
    if key in cache:
        return cache[key]

    numerical_errors = _numerical_errors()
    low, high = (1., 2000.) if variable == 'T' else (1e-8, 10000.)
    bounds = (np.array([math.log(low)] + [-1400.] * (len(active)-1)),
              np.array([math.log(high)] + [1400.] * (len(active)-1)))

    def state(value):
        return (value, fixed) if variable == 'T' else (fixed, value)

    def python_state(T, P, values, phase, data):
        composition = dict(zip(thermo.components, values.tolist()))
        phi = thermo.fugacity_coefficients(T, P, composition, 'liquid' if phase == 0 else 'vapor')
        return 0, np.array([math.log(phi[c]) if z[c] > 0 else 0. for c in thermo.components])

    def phases(values):
        T, P = state(math.exp(values[0]))
        other_array, log_other = numerics.boundary_composition(np.asarray(values), array_z, active_indices)
        other = dict(zip(thermo.components, other_array.tolist()))
        x, y = (z, other) if kind == 'bubble' else (other, z)
        log_x, log_y = (log_z, log_other) if kind == 'bubble' else (log_other, log_z)
        return T, P, x, y, log_x, log_y

    def equations(values):
        T_trial = math.exp(values[0]) if variable == 'T' else fixed
        native = _native_provider(thermo, T_trial)
        phase_state, data = native if native is not None else (python_state, None)
        solver = (_bound_kernels(phase_state).boundary_equations if native is not None else
                  partial(numerics.interpreted(numerics.boundary_equations), python_state))
        return solver(data, np.asarray(values), array_z, active_indices,
                      float(fixed), kind == 'bubble', variable == 'T')

    def jacobian(values):
        return _boundary_jacobian(thermo, values, array_z, active_indices, fixed,
                                  kind == 'bubble', variable == 'T', bounds, equations, python_state)

    def seed(value):
        T, P = state(value)
        K = wilson_K(thermo.props, T, P, active)
        other = log_z + (1 if kind == 'bubble' else -1) * np.log([K[c] for c in active])
        return np.r_[math.log(value), (other - other[-1])[:-1]]

    if variable == 'T':
        def wilson_residual(T):
            K = wilson_K(thermo.props, T, fixed, active)
            return sum(z[c] * (K[c] if kind == 'bubble' else 1/K[c]) for c in active) - 1
        try:
            wilson = brentq(wilson_residual, low, high)
        except ValueError:
            wilson = guess if guess is not None else sum(z[c]*thermo.props[c].Tc for c in active)
    else:
        K = wilson_K(thermo.props, fixed, 1., active)
        wilson = (sum(z[c]*K[c] for c in active) if kind == 'bubble'
                  else 1/sum(z[c]/K[c] for c in active))

    def branch_seeds():
        # Near criticality the two-root interval can be much narrower than a
        # coarse pressure/temperature grid. Seed from actual spinodal geometry
        # so a pure-fluid solve is not trapped on a homogeneous zero residual.
        if variable == 'P':
            extrema = sorted(thermo._eos_pressure_extrema(fixed, z))
            endpoints = [0., *extrema]
            for left, right in zip(endpoints, endpoints[1:]):
                if right > max(left, 0.):
                    yield (max(left, 0.) + right) / 2
            return
        lower = max(low, min(thermo.props[c].Tc for c in active) * .5)
        upper = min(high, max(thermo.props[c].Tc for c in active) * 2)
        for _ in range(12):
            if thermo._eos_pressure_extrema(lower, z):
                break
            lower = max(low, lower * .5)
        else:
            return
        while upper < high and thermo._eos_pressure_extrema(upper, z):
            upper = min(high, upper * 2)
        if thermo._eos_pressure_extrema(upper, z):
            return
        # Keep the lower endpoint strictly on the subcritical side.
        for _ in range(36):
            middle = (lower + upper) / 2
            if thermo._eos_pressure_extrema(middle, z):
                lower = middle
            else:
                upper = middle
        # At low T a BWR fluid can have extra spinodal pairs. Locate the
        # highest-temperature fold first, then count the ordinary pairs just
        # below it instead of treating a disappearing low-T loop as critical.
        critical = lower
        lower = max(low, .9 * critical)
        count = len(thermo._eos_pressure_extrema(lower, z))
        upper = critical
        for _ in range(30):
            middle = (lower + upper) / 2
            if len(thermo._eos_pressure_extrema(middle, z)) >= count:
                lower = middle
            else:
                upper = middle
        critical = lower

        # LKP's simple/reference fluids can have disjoint narrow intervals.
        # Seed each interval separately; averaging their pressures can land
        # on a homogeneous root of both fluids instead of a coexistence root.
        for index in range(0, count, 2):
            def midpoint_residual(T):
                extrema = thermo._eos_pressure_extrema(T, z)
                return (extrema[index] + extrema[index+1]) / 2 - fixed

            for factor in (.75, .9, .97):
                lower = max(low, critical * factor)
                if midpoint_residual(lower) * midpoint_residual(critical) < 0:
                    yield brentq(midpoint_residual, lower, critical, xtol=1e-9)
                    break

    def candidates():
        if guess is not None:
            yield guess
        yield wilson
        try:
            yield from branch_seeds()
        except numerical_errors:
            pass
        for factor in (.95, 1.05, .85, 1.15, .7, 1.3):
            yield wilson * factor

    tried = set()
    for value in candidates():
        value = min(high, max(low, float(value)))
        if value in tried:
            continue
        tried.add(value)
        try:
            solution = least_squares(equations, seed(value), bounds=bounds,
                                     jac=jacobian, max_nfev=150, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if max(abs(equations(solution.x))) > 1e-8:
                continue
            T, P, x, y, _, _ = phases(solution.x)
            if not all(_admissible_phase_kind(thermo._eos_root_kind(T, P, phase_composition, phase), phase)
                       for phase, phase_composition in (('liquid', x), ('vapor', y))):
                continue
            thermo._check_diagram_vle(x, y, T, P, fugacity_tolerance=1e-8)
            result = T if variable == 'T' else P
            return thermo._set_limited_cache(cache, key, result, limit=1024)
        except numerical_errors:
            continue
    raise PhiPhiFlashError(
        f'Could not establish a physical EOS {kind} point: no distinct admissible '
        'phases satisfying fugacity equality were found')


def iterate_phi_phi_K(backend, T, P, x, max_iter=80):
    """Estimate K for a fixed liquid, with Wilson extrapolation in one phase.

    The extrapolation helps bracket phase boundaries but is not an equilibrium
    solution. TP flashes must update K from their actual two phase fugacities.
    """
    components = backend.components
    single_root = len(backend.compressibility_roots(T, P, x)) < 2
    phi_l = backend.fugacity_coefficients(T, P, x, "liquid")
    K = wilson_K(backend.props, T, P, components)
    initial_K = dict(K)
    y = backend._normalized_composition({comp: x[comp] * K[comp] for comp in components})
    for _ in range(max_iter):
        phi_v = backend.fugacity_coefficients(T, P, y, "vapor")
        K_new = {
            comp: max(1e-8, min(1e8, phi_l.get(comp, 1.0) / max(phi_v.get(comp, 1.0), 1e-12)))
            for comp in components
        }
        y_new = backend._normalized_composition({comp: x[comp] * K_new[comp] for comp in components})
        if max(abs(y_new[comp] - y[comp]) for comp in components) < 1e-9:
            # As in the cubic EOS: Wilson extrapolation avoids a homogeneous
            # K=1 fixed point hiding a phase boundary when only one root exists.
            if single_root and all(abs(K_new[c] - 1.0) < 1e-6 for c in components if x[c] > 0.0):
                return initial_K
            return K_new
        K = {comp: 0.5 * K[comp] + 0.5 * K_new[comp] for comp in components}
        y = y_new
    return K
