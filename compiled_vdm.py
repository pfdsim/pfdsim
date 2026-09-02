"""
Compiled vapor-dimerization backend.

The readable VDM implementation in vapor_dimerization.py handles names,
cross-association metadata, fallbacks, and reporting. This module only
accelerates the hot numeric kernel for the common two-acid case:

    A + A <-> A2
    A + B <-> AB
    B + B <-> B2

The public wrapper returns ``None`` when Numba is unavailable or the numeric
solve fails, so callers can safely fall back to the generic SciPy machinery.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, typeof
except Exception:  # pragma: no cover - exercised only without optional numba
    njit = None
    typeof = None


_COMPILATION_COMPLETE = False


def compile_vdm_kernels() -> bool:
    """Compile/load VDM kernels by signature without solving a state."""
    global _COMPILATION_COMPLETE
    if _COMPILATION_COMPLETE:
        return True
    if njit is None or typeof is None:
        return False
    two_args = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    _solve_two_acid_true_moles_numba.compile(
        tuple(typeof(argument) for argument in two_args)
    )
    nominal = np.zeros(2, dtype=np.float64)
    pair_i = np.zeros(3, dtype=np.int64)
    pair_j = np.zeros(3, dtype=np.int64)
    pair_kappa = np.zeros(3, dtype=np.float64)
    n_args = (nominal, 1.0, pair_i, pair_j, pair_kappa)
    _solve_n_acid_true_moles_numba.compile(
        tuple(typeof(argument) for argument in n_args)
    )
    closure_args = (
        np.ones(3, dtype=np.float64),
        np.ones(3, dtype=np.float64) / 3.0,
        np.asarray([0, 1], dtype=np.int64),
        pair_i,
        pair_j,
        np.ones(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        12,
        1.0e-10,
        0,
        1.0e-8,
    )
    _vapor_closure_numba.compile(
        tuple(typeof(argument) for argument in closure_args)
    )
    _COMPILATION_COMPLETE = True
    return True


def solve_two_acid_true_moles(
    n_a: float,
    n_b: float,
    inert_total: float,
    k_aa: float,
    k_ab: float,
    k_bb: float,
) -> tuple[float, float, float, float, float, float] | None:
    """
    Solve the two-acid vapor association material balances.

    Parameters are the nominal acid moles ``n_a``/``n_b``, inert nominal total,
    and association coefficients ``k_aa``, ``k_ab``, ``k_bb``. The return value
    is ``(a, b, e_aa, e_ab, e_bb, total)`` where ``a`` and ``b`` are monomer
    moles, ``e_*`` are dimer extents, and ``total`` is the true vapor mole
    total after association.
    """
    if njit is None:
        return None
    ok, a, b, e_aa, e_ab, e_bb, total = _solve_two_acid_true_moles_numba(
        float(n_a),
        float(n_b),
        float(inert_total),
        float(k_aa),
        float(k_ab),
        float(k_bb),
    )
    if not ok:
        return None
    return a, b, e_aa, e_ab, e_bb, total


def solve_n_acid_true_moles(
    nominal: list[float] | np.ndarray,
    inert_total: float,
    pair_i: list[int] | np.ndarray,
    pair_j: list[int] | np.ndarray,
    pair_kappa: list[float] | np.ndarray,
) -> tuple[list[float], list[float], float] | None:
    """
    Solve the generic N-acid vapor association material balances.

    ``pair_i``/``pair_j``/``pair_kappa`` describe the upper-triangular acid
    pair list in the same order the caller wants extents returned.
    """
    if njit is None:
        return None
    ok, monomers, extents, total = _solve_n_acid_true_moles_numba(
        np.asarray(nominal, dtype=np.float64),
        float(inert_total),
        np.asarray(pair_i, dtype=np.int64),
        np.asarray(pair_j, dtype=np.int64),
        np.asarray(pair_kappa, dtype=np.float64),
    )
    if not ok:
        return None
    return monomers.tolist(), extents.tolist(), float(total)


def solve_vapor_closure(
    base_values: list[float] | np.ndarray,
    liquid_composition: list[float] | np.ndarray,
    acid_indices: list[int] | np.ndarray,
    pair_i: list[int] | np.ndarray,
    pair_j: list[int] | np.ndarray,
    pair_kappa: list[float] | np.ndarray,
    pair_delta_h: list[float] | np.ndarray,
    max_iter: int,
    tol: float,
    mode: int,
    phi_floor: float,
) -> dict | None:
    """Solve a multi-acid ideal-physical-fugacity vapor closure.

    ``mode=1`` reproduces the VLE K-value iteration, while ``mode=0``
    reproduces the shared-VLLE vapor-term iteration.
    """
    if njit is None:
        return None
    result = _vapor_closure_numba(
        np.asarray(base_values, dtype=np.float64),
        np.asarray(liquid_composition, dtype=np.float64),
        np.asarray(acid_indices, dtype=np.int64),
        np.asarray(pair_i, dtype=np.int64),
        np.asarray(pair_j, dtype=np.int64),
        np.asarray(pair_kappa, dtype=np.float64),
        np.asarray(pair_delta_h, dtype=np.float64),
        int(max_iter),
        float(tol),
        int(mode),
        float(phi_floor),
    )
    ok, values, vapor_terms, vapor, phi, extents, association_enthalpy, iterations = result
    if not ok:
        return None
    return {
        "values": values.tolist(),
        "vapor_terms": vapor_terms.tolist(),
        "vapor_composition": vapor.tolist(),
        "phi_total": phi.tolist(),
        "extents": extents.tolist(),
        "association_enthalpy": float(association_enthalpy),
        "iterations": int(iterations),
    }


if njit is not None:

    @njit(cache=True)
    def _normalize_positive(values):
        total = 0.0
        for value in values:
            if value > 0.0:
                total += value
        normalized = np.zeros(values.shape[0], dtype=np.float64)
        if total <= 0.0:
            return False, normalized
        for index in range(values.shape[0]):
            normalized[index] = max(values[index], 0.0) / total
        return True, normalized


    @njit(cache=True)
    def _association_arrays(
        vapor,
        acid_indices,
        pair_i,
        pair_j,
        pair_kappa,
        pair_delta_h,
    ):
        n_acids = acid_indices.shape[0]
        nominal = np.empty(n_acids, dtype=np.float64)
        acid_total = 0.0
        for acid in range(n_acids):
            value = max(vapor[acid_indices[acid]], 0.0)
            nominal[acid] = value
            acid_total += value
        inert_total = max(1.0 - acid_total, 0.0)

        if n_acids == 2 and pair_kappa.shape[0] == 3:
            solved = _solve_two_acid_true_moles_numba(
                nominal[0],
                nominal[1],
                inert_total,
                pair_kappa[0],
                pair_kappa[1],
                pair_kappa[2],
            )
            if not solved[0]:
                return (
                    False,
                    np.ones(vapor.shape[0], dtype=np.float64),
                    np.zeros(pair_kappa.shape[0], dtype=np.float64),
                    0.0,
                )
            monomers = np.asarray([solved[1], solved[2]], dtype=np.float64)
            extents = np.asarray([solved[3], solved[4], solved[5]], dtype=np.float64)
            total = solved[6]
        else:
            solved = _solve_n_acid_true_moles_numba(
                nominal,
                inert_total,
                pair_i,
                pair_j,
                pair_kappa,
            )
            if not solved[0]:
                return (
                    False,
                    np.ones(vapor.shape[0], dtype=np.float64),
                    np.zeros(pair_kappa.shape[0], dtype=np.float64),
                    0.0,
                )
            monomers = solved[1]
            extents = solved[2]
            total = solved[3]

        phi = np.ones(vapor.shape[0], dtype=np.float64)
        for acid in range(n_acids):
            nominal_value = nominal[acid]
            if nominal_value > 1.0e-30:
                phi[acid_indices[acid]] = max(
                    monomers[acid] / max(total, 1.0e-300) / nominal_value,
                    1.0e-30,
                )
        association_enthalpy = 0.0
        for pair in range(extents.shape[0]):
            association_enthalpy += extents[pair] * pair_delta_h[pair]
        return True, phi, extents, association_enthalpy


    @njit(cache=True)
    def _vapor_closure_numba(
        base_values,
        liquid_composition,
        acid_indices,
        pair_i,
        pair_j,
        pair_kappa,
        pair_delta_h,
        max_iter,
        tol,
        mode,
        phi_floor,
    ):
        n = base_values.shape[0]
        values = np.empty(n, dtype=np.float64)
        vapor_terms = np.empty(n, dtype=np.float64)
        for index in range(n):
            if mode == 1:
                values[index] = min(max(base_values[index], 1.0e-6), 1.0e6)
                vapor_terms[index] = max(liquid_composition[index], 0.0) * values[index]
            else:
                values[index] = max(base_values[index], 1.0e-300)
                vapor_terms[index] = values[index]
        valid, vapor = _normalize_positive(vapor_terms)
        if not valid:
            return (
                False, values, vapor_terms, vapor,
                np.ones(n, dtype=np.float64),
                np.zeros(pair_kappa.shape[0], dtype=np.float64),
                0.0, 0,
            )

        iterations = 0
        for iteration in range(max_iter):
            ok, phi, _extents, _enthalpy = _association_arrays(
                vapor, acid_indices, pair_i, pair_j, pair_kappa, pair_delta_h
            )
            if not ok:
                return (
                    False, values, vapor_terms, vapor, phi, _extents,
                    _enthalpy, iterations,
                )
            for index in range(n):
                denominator = max(phi[index], phi_floor)
                if mode == 1:
                    values[index] = min(
                        max(base_values[index] / denominator, 1.0e-6),
                        1.0e6,
                    )
                    vapor_terms[index] = (
                        max(liquid_composition[index], 0.0) * values[index]
                    )
                else:
                    values[index] = max(
                        base_values[index] / denominator,
                        1.0e-300,
                    )
                    vapor_terms[index] = values[index]
            valid, vapor_new = _normalize_positive(vapor_terms)
            if not valid:
                return (
                    False, values, vapor_terms, vapor, phi, _extents,
                    _enthalpy, iterations,
                )
            change = 0.0
            for index in range(n):
                difference = abs(vapor_new[index] - vapor[index])
                if difference > change:
                    change = difference
            iterations = iteration + 1
            if change < tol:
                vapor = vapor_new
                break
            vapor = vapor_new

        # Match the shared-VLLE path's final fugacity re-evaluation. VLE keeps
        # its final K values but still returns association enthalpy evaluated
        # at the vapor composition implied by those values.
        ok, phi, extents, association_enthalpy = _association_arrays(
            vapor, acid_indices, pair_i, pair_j, pair_kappa, pair_delta_h
        )
        if not ok:
            return (
                False, values, vapor_terms, vapor, phi, extents,
                association_enthalpy, iterations,
            )
        if mode == 0:
            for index in range(n):
                values[index] = max(
                    base_values[index] / max(phi[index], phi_floor),
                    1.0e-300,
                )
                vapor_terms[index] = values[index]
            valid, returned_vapor = _normalize_positive(vapor_terms)
            if not valid:
                return (
                    False, values, vapor_terms, vapor, phi, extents,
                    association_enthalpy, iterations,
                )
            vapor = returned_vapor
            ok, phi, extents, association_enthalpy = _association_arrays(
                vapor, acid_indices, pair_i, pair_j, pair_kappa, pair_delta_h
            )
            if not ok:
                return (
                    False, values, vapor_terms, vapor, phi, extents,
                    association_enthalpy, iterations,
                )
        return (
            True, values, vapor_terms, vapor, phi, extents,
            association_enthalpy, iterations,
        )

    @njit(cache=True)
    def _solve_two_acid_true_moles_numba(n_a, n_b, inert_total, k_aa, k_ab, k_bb):
        if n_a <= 0.0 or n_b <= 0.0:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0

        if k_aa < 0.0:
            k_aa = 0.0
        if k_ab < 0.0:
            k_ab = 0.0
        if k_bb < 0.0:
            k_bb = 0.0

        scale_a = n_a if n_a > 1e-12 else 1e-12
        scale_b = n_b if n_b > 1e-12 else 1e-12
        a = 0.5 * n_a
        b = 0.5 * n_b
        if a <= 1e-300:
            a = 1e-300
        if b <= 1e-300:
            b = 1e-300

        best_a = a
        best_b = b
        best_e_aa = 0.0
        best_e_ab = 0.0
        best_e_bb = 0.0
        best_total = 1.0
        best_norm = 1e300

        for _iteration in range(30):
            evaluated = _evaluate_two_acid_state(
                a, b, n_a, n_b, inert_total, k_aa, k_ab, k_bb, scale_a, scale_b
            )
            ok = evaluated[0]
            if not ok:
                return False, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0

            (
                _ok,
                residual_a,
                residual_b,
                j11,
                j12,
                j21,
                j22,
                e_aa,
                e_ab,
                e_bb,
                total,
            ) = evaluated
            norm = abs(residual_a) if abs(residual_a) > abs(residual_b) else abs(residual_b)
            if norm < best_norm:
                best_norm = norm
                best_a = a
                best_b = b
                best_e_aa = e_aa if e_aa > 0.0 else 0.0
                best_e_ab = e_ab if e_ab > 0.0 else 0.0
                best_e_bb = e_bb if e_bb > 0.0 else 0.0
                best_total = total
            if norm < 1e-11:
                return True, a, b, best_e_aa, best_e_ab, best_e_bb, total

            det = j11 * j22 - j12 * j21
            if not math.isfinite(det) or abs(det) < 1e-30:
                break

            delta_a = (-residual_a * j22 + j12 * residual_b) / det
            delta_b = (j21 * residual_a - j11 * residual_b) / det
            if not math.isfinite(delta_a) or not math.isfinite(delta_b):
                break

            accepted = False
            for attempt in range(20):
                step = 0.5 ** attempt
                trial_a = a + step * delta_a
                trial_b = b + step * delta_b
                if not (1e-300 < trial_a <= n_a and 1e-300 < trial_b <= n_b):
                    continue
                trial = _evaluate_two_acid_state(
                    trial_a,
                    trial_b,
                    n_a,
                    n_b,
                    inert_total,
                    k_aa,
                    k_ab,
                    k_bb,
                    scale_a,
                    scale_b,
                )
                if not trial[0]:
                    continue
                trial_residual_a = trial[1]
                trial_residual_b = trial[2]
                trial_norm = (
                    abs(trial_residual_a)
                    if abs(trial_residual_a) > abs(trial_residual_b)
                    else abs(trial_residual_b)
                )
                if trial_norm < norm:
                    a = trial_a
                    b = trial_b
                    accepted = True
                    break
            if not accepted:
                break

        if best_norm < 1e-8:
            return True, best_a, best_b, best_e_aa, best_e_ab, best_e_bb, best_total
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0


    @njit(cache=True)
    def _evaluate_two_acid_state(
        a,
        b,
        n_a,
        n_b,
        inert_total,
        k_aa,
        k_ab,
        k_bb,
        scale_a,
        scale_b,
    ):
        term_aa = k_aa * a * a
        term_ab = k_ab * a * b
        term_bb = k_bb * b * b
        association_sum = term_aa + term_ab + term_bb
        base_total = inert_total + a + b
        disc = base_total * base_total + 4.0 * association_sum
        if disc < 0.0:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0

        root = math.sqrt(disc)
        if root <= 1e-300:
            return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0

        total = 0.5 * (base_total + root)
        if total <= 1e-300:
            total = 1e-300

        e_aa = term_aa / total
        e_ab = term_ab / total
        e_bb = term_bb / total
        f_a = a + 2.0 * e_aa + e_ab - n_a
        f_b = b + 2.0 * e_bb + e_ab - n_b

        d_assoc_da = 2.0 * k_aa * a + k_ab * b
        d_assoc_db = k_ab * a + 2.0 * k_bb * b
        d_total_da = 0.5 * (1.0 + (base_total + 2.0 * d_assoc_da) / root)
        d_total_db = 0.5 * (1.0 + (base_total + 2.0 * d_assoc_db) / root)
        inv_total_sq = 1.0 / (total * total)

        de_aa_da = ((2.0 * k_aa * a) * total - term_aa * d_total_da) * inv_total_sq
        de_aa_db = (0.0 * total - term_aa * d_total_db) * inv_total_sq
        de_ab_da = ((k_ab * b) * total - term_ab * d_total_da) * inv_total_sq
        de_ab_db = ((k_ab * a) * total - term_ab * d_total_db) * inv_total_sq
        de_bb_da = (0.0 * total - term_bb * d_total_da) * inv_total_sq
        de_bb_db = ((2.0 * k_bb * b) * total - term_bb * d_total_db) * inv_total_sq

        j11 = (1.0 + 2.0 * de_aa_da + de_ab_da) / scale_a
        j12 = (2.0 * de_aa_db + de_ab_db) / scale_a
        j21 = (2.0 * de_bb_da + de_ab_da) / scale_b
        j22 = (1.0 + 2.0 * de_bb_db + de_ab_db) / scale_b
        residual_a = f_a / scale_a
        residual_b = f_b / scale_b
        return (
            True,
            residual_a,
            residual_b,
            j11,
            j12,
            j21,
            j22,
            e_aa,
            e_ab,
            e_bb,
            total,
        )


    @njit(cache=True)
    def _solve_n_acid_true_moles_numba(nominal, inert_total, pair_i, pair_j, pair_kappa):
        n = nominal.shape[0]
        n_pairs = pair_kappa.shape[0]
        for i in range(n):
            if nominal[i] <= 0.0:
                return False, np.zeros(n, dtype=np.float64), np.zeros(n_pairs, dtype=np.float64), 1.0

        scales = np.empty(n, dtype=np.float64)
        monomers = np.empty(n, dtype=np.float64)
        best_monomers = np.empty(n, dtype=np.float64)
        residual = np.empty(n, dtype=np.float64)
        trial_residual = np.empty(n, dtype=np.float64)
        jacobian = np.empty((n, n), dtype=np.float64)
        terms = np.empty(n_pairs, dtype=np.float64)
        extents = np.empty(n_pairs, dtype=np.float64)
        best_extents = np.zeros(n_pairs, dtype=np.float64)

        for i in range(n):
            scales[i] = nominal[i] if nominal[i] > 1e-12 else 1e-12
            monomers[i] = 0.5 * nominal[i]
            if monomers[i] <= 1e-300:
                monomers[i] = 1e-300
            best_monomers[i] = monomers[i]

        best_total = 1.0
        best_norm = 1e300

        for _iteration in range(50):
            ok, norm, total = _evaluate_n_acid_state(
                monomers,
                nominal,
                inert_total,
                pair_i,
                pair_j,
                pair_kappa,
                scales,
                residual,
                jacobian,
                terms,
                extents,
            )
            if not ok:
                return False, np.zeros(n, dtype=np.float64), np.zeros(n_pairs, dtype=np.float64), 1.0

            if norm < best_norm:
                best_norm = norm
                best_total = total
                for i in range(n):
                    best_monomers[i] = monomers[i]
                for pair in range(n_pairs):
                    best_extents[pair] = extents[pair] if extents[pair] > 0.0 else 0.0

            if norm < 1e-11:
                result_extents = np.empty(n_pairs, dtype=np.float64)
                for pair in range(n_pairs):
                    result_extents[pair] = extents[pair] if extents[pair] > 0.0 else 0.0
                return True, monomers.copy(), result_extents, total

            delta = _solve_linear_system(jacobian, residual)
            if delta.shape[0] != n:
                break

            accepted = False
            for attempt in range(24):
                step = 0.5 ** attempt
                trial = np.empty(n, dtype=np.float64)
                in_bounds = True
                for i in range(n):
                    trial[i] = monomers[i] - step * delta[i]
                    if not (1e-300 < trial[i] <= nominal[i]):
                        in_bounds = False
                        break
                if not in_bounds:
                    continue

                trial_jacobian = np.empty((n, n), dtype=np.float64)
                trial_terms = np.empty(n_pairs, dtype=np.float64)
                trial_extents = np.empty(n_pairs, dtype=np.float64)
                ok, trial_norm, _trial_total = _evaluate_n_acid_state(
                    trial,
                    nominal,
                    inert_total,
                    pair_i,
                    pair_j,
                    pair_kappa,
                    scales,
                    trial_residual,
                    trial_jacobian,
                    trial_terms,
                    trial_extents,
                )
                if ok and trial_norm < norm:
                    for i in range(n):
                        monomers[i] = trial[i]
                    accepted = True
                    break

            if not accepted:
                break

        if best_norm < 1e-8:
            return True, best_monomers, best_extents, best_total
        return False, np.zeros(n, dtype=np.float64), np.zeros(n_pairs, dtype=np.float64), 1.0


    @njit(cache=True)
    def _evaluate_n_acid_state(
        monomers,
        nominal,
        inert_total,
        pair_i,
        pair_j,
        pair_kappa,
        scales,
        residual,
        jacobian,
        terms,
        extents,
    ):
        n = nominal.shape[0]
        n_pairs = pair_kappa.shape[0]
        association_sum = 0.0
        base_total = inert_total
        for i in range(n):
            base_total += monomers[i]

        for pair in range(n_pairs):
            coeff = pair_kappa[pair]
            if coeff < 0.0:
                coeff = 0.0
            term = coeff * monomers[pair_i[pair]] * monomers[pair_j[pair]]
            terms[pair] = term
            association_sum += term

        disc = base_total * base_total + 4.0 * association_sum
        if disc < 0.0:
            return False, 1e300, 1.0
        root = math.sqrt(disc)
        if root <= 1e-300:
            return False, 1e300, 1.0

        total = 0.5 * (base_total + root)
        if total <= 1e-300:
            total = 1e-300

        for i in range(n):
            residual[i] = monomers[i] - nominal[i]
            for j in range(n):
                jacobian[i, j] = 1.0 if i == j else 0.0

        d_assoc = np.zeros(n, dtype=np.float64)
        d_terms = np.zeros((n_pairs, n), dtype=np.float64)
        for pair in range(n_pairs):
            i = pair_i[pair]
            j = pair_j[pair]
            coeff = pair_kappa[pair]
            if coeff < 0.0:
                coeff = 0.0
            extent = terms[pair] / total
            if extent < 0.0:
                extent = 0.0
            extents[pair] = extent

            if i == j:
                residual[i] += 2.0 * extent
                derivative = 2.0 * coeff * monomers[i]
                d_terms[pair, i] = derivative
                d_assoc[i] += derivative
            else:
                residual[i] += extent
                residual[j] += extent
                d_i = coeff * monomers[j]
                d_j = coeff * monomers[i]
                d_terms[pair, i] = d_i
                d_terms[pair, j] = d_j
                d_assoc[i] += d_i
                d_assoc[j] += d_j

        d_total = np.empty(n, dtype=np.float64)
        for i in range(n):
            d_total[i] = 0.5 * (1.0 + (base_total + 2.0 * d_assoc[i]) / root)

        inv_total_sq = 1.0 / (total * total)
        for pair in range(n_pairs):
            i = pair_i[pair]
            j = pair_j[pair]
            term = terms[pair]
            for col in range(n):
                d_extent = (d_terms[pair, col] * total - term * d_total[col]) * inv_total_sq
                if i == j:
                    jacobian[i, col] += 2.0 * d_extent
                else:
                    jacobian[i, col] += d_extent
                    jacobian[j, col] += d_extent

        norm = 0.0
        for i in range(n):
            residual[i] /= scales[i]
            for j in range(n):
                jacobian[i, j] /= scales[i]
            absolute = abs(residual[i])
            if absolute > norm:
                norm = absolute
        return True, norm, total


    @njit(cache=True)
    def _solve_linear_system(A, b):
        n = b.shape[0]
        matrix = A.copy()
        rhs = b.copy()

        for k in range(n):
            pivot = k
            pivot_abs = abs(matrix[k, k])
            for i in range(k + 1, n):
                value = abs(matrix[i, k])
                if value > pivot_abs:
                    pivot = i
                    pivot_abs = value
            if pivot_abs < 1e-30:
                return np.empty(0, dtype=np.float64)

            if pivot != k:
                for j in range(k, n):
                    tmp = matrix[k, j]
                    matrix[k, j] = matrix[pivot, j]
                    matrix[pivot, j] = tmp
                tmp = rhs[k]
                rhs[k] = rhs[pivot]
                rhs[pivot] = tmp

            pivot_value = matrix[k, k]
            for i in range(k + 1, n):
                factor = matrix[i, k] / pivot_value
                matrix[i, k] = 0.0
                for j in range(k + 1, n):
                    matrix[i, j] -= factor * matrix[k, j]
                rhs[i] -= factor * rhs[k]

        result = np.empty(n, dtype=np.float64)
        for reverse_index in range(n):
            i = n - 1 - reverse_index
            total = rhs[i]
            for j in range(i + 1, n):
                total -= matrix[i, j] * result[j]
            result[i] = total / matrix[i, i]
        return result

else:

    def _vapor_closure_numba(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("Numba is not available")

    def _solve_two_acid_true_moles_numba(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("Numba is not available")
