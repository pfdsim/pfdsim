from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Literal
import math

import numpy as np


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physical_constants import R_J_MOL_K

R = R_J_MOL_K  # J/mol/K


@dataclass(frozen=True)
class SurfaceComponent:
    """Pure-component data needed for mixture surface tension.

    Units:
        Vc: critical molar volume, m^3/mol
        rho_molar(T): pure liquid molar density, mol/m^3
        sigma(T): pure liquid surface tension, N/m
    """

    name: str
    Vc: float
    rho_molar: Callable[[float], float]
    sigma: Callable[[float], float]


@dataclass(frozen=True)
class SurfaceTensionResult:
    """Result from a mixture surface-tension calculation."""

    sigma: float                  # N/m
    surface_x: np.ndarray          # surface mole fractions
    bulk_x: np.ndarray             # normalized bulk mole fractions
    A: np.ndarray                  # molar surface areas, m^2/mol
    method: str
    converged: bool
    residual_max_abs: float        # N/m
    iterations: int
    warnings: tuple[str, ...] = ()


# UNIFAC callback signature.
#
# It should return ln(gamma), not gamma.
#
# Example adapter:
#
#     def unifac_ln_gamma(T, x, comps):
#         gammas = my_unifac.gamma(T, x, [c.name for c in comps])
#         return np.log(gammas)
#
UNIFACLnGamma = Callable[[float, np.ndarray, Sequence[SurfaceComponent]], np.ndarray]


def normalize_mole_fractions(x: Sequence[float], *, atol: float = 1e-14) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)

    if x_arr.ndim != 1:
        raise ValueError("Mole fractions must be a 1D sequence.")

    if len(x_arr) == 0:
        raise ValueError("At least one component is required.")

    if np.any(~np.isfinite(x_arr)):
        raise ValueError("Mole fractions must be finite.")

    if np.any(x_arr < -atol):
        raise ValueError("Mole fractions cannot be negative.")

    x_arr = np.maximum(x_arr, 0.0)
    total = float(np.sum(x_arr))

    if total <= 0.0:
        raise ValueError("At least one mole fraction must be positive.")

    return x_arr / total


def pure_surface_inputs(
    T: float,
    components: Sequence[SurfaceComponent],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Vc, Vb, sigma_i, and Goldsack-White A_i.

    Units:
        Vc: m^3/mol
        Vb: m^3/mol
        sigma_i: N/m
        A_i: m^2/mol
    """

    if T <= 0.0 or not math.isfinite(T):
        raise ValueError("Temperature must be finite and positive.")

    Vc = np.array([c.Vc for c in components], dtype=float)
    rho = np.array([c.rho_molar(T) for c in components], dtype=float)
    sigma_i = np.array([c.sigma(T) for c in components], dtype=float)

    if np.any(~np.isfinite(Vc)) or np.any(Vc <= 0.0):
        raise ValueError("All critical molar volumes Vc must be finite and positive.")

    if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError("All pure liquid molar densities rho_molar(T) must be finite and positive.")

    if np.any(~np.isfinite(sigma_i)) or np.any(sigma_i < 0.0):
        raise ValueError("All pure surface tensions sigma(T) must be finite and nonnegative.")

    Vb = 1.0 / rho
    A = goldsack_white_area(Vc, Vb)

    return Vc, Vb, sigma_i, A


def goldsack_white_area(Vc: np.ndarray, Vb: np.ndarray) -> np.ndarray:
    """Goldsack-White molar surface area.

    A_i = 1.021e8 * Vc_i^(6/15) * Vb_i^(4/15)

    If Vc and Vb are in m^3/mol, A_i comes out in m^2/mol.
    The same numerical coefficient also works for cm^3/mol -> cm^2/mol.

    Args:
        Vc: critical molar volumes, m^3/mol
        Vb: pure liquid molar volumes at T, m^3/mol

    Returns:
        A: molar surface areas, m^2/mol
    """

    Vc = np.asarray(Vc, dtype=float)
    Vb = np.asarray(Vb, dtype=float)

    if np.any(Vc <= 0.0) or np.any(Vb <= 0.0):
        raise ValueError("Vc and Vb must be positive.")

    return 1.021e8 * Vc ** (6.0 / 15.0) * Vb ** (4.0 / 15.0)


def winterfeld_scriven_davis_surface_tension(
    T: float,
    x: Sequence[float],
    components: Sequence[SurfaceComponent],
) -> SurfaceTensionResult:
    """Winterfeld-Scriven-Davis / DIPPR 7C mixture surface tension.

    Recommended mainly for nonaqueous ordinary liquid mixtures.

    sigma_m = sum_i sum_j [(x_i V_i)(x_j V_j) / V_m^2] * sqrt(sigma_i sigma_j)

    Args:
        T: K
        x: bulk mole fractions
        components: component data

    Returns:
        SurfaceTensionResult with surface_x equal to bulk_x because this method
        does not explicitly solve a surface phase.
    """

    x_arr = normalize_mole_fractions(x)

    if len(x_arr) != len(components):
        raise ValueError("Length of x must match length of components.")

    _, Vb, sigma_i, A = pure_surface_inputs(T, components)

    if len(x_arr) == 1:
        return SurfaceTensionResult(
            sigma=float(sigma_i[0]),
            surface_x=x_arr.copy(),
            bulk_x=x_arr.copy(),
            A=A,
            method="Winterfeld-Scriven-Davis",
            converged=True,
            residual_max_abs=0.0,
            iterations=0,
        )

    V_mix = float(np.dot(x_arr, Vb))
    w = x_arr * Vb

    sigma_matrix = np.sqrt(np.outer(sigma_i, sigma_i))
    sigma_mix = float(np.sum(np.outer(w, w) * sigma_matrix) / (V_mix * V_mix))

    return SurfaceTensionResult(
        sigma=sigma_mix,
        surface_x=x_arr.copy(),
        bulk_x=x_arr.copy(),
        A=A,
        method="Winterfeld-Scriven-Davis",
        converged=True,
        residual_max_abs=0.0,
        iterations=0,
    )


def butler_unifac_surface_tension(
    T: float,
    x: Sequence[float],
    components: Sequence[SurfaceComponent],
    unifac_ln_gamma: UNIFACLnGamma,
    *,
    zero_cutoff: float = 1e-15,
    residual_tol: float = 1e-8,
    max_nfev: int = 300,
) -> SurfaceTensionResult:
    """Butler surface-phase method using UNIFAC activity coefficients.

    Solves, for each component i:

        sigma = sigma_i + (R*T/A_i) * ln[
            (x_i^s * gamma_i^s) / (x_i^b * gamma_i^b)
        ]

    with:

        sum_i x_i^s = 1

    Args:
        T: temperature, K
        x: bulk mole fractions
        components: component data
        unifac_ln_gamma:
            callback returning ln(gamma_i) at T, x, components
        zero_cutoff:
            components with x <= zero_cutoff are removed from the active mixture
        residual_tol:
            convergence tolerance in N/m for equality of component Butler tensions
        max_nfev:
            max function evaluations for scipy least_squares

    Returns:
        SurfaceTensionResult
    """

    x_full = normalize_mole_fractions(x)

    if len(x_full) != len(components):
        raise ValueError("Length of x must match length of components.")

    active = x_full > zero_cutoff

    if not np.any(active):
        raise ValueError("No active components after applying zero_cutoff.")

    active_indices = np.flatnonzero(active)
    active_components = [components[i] for i in active_indices]
    x_active = normalize_mole_fractions(x_full[active])

    _, _, sigma_i, A = pure_surface_inputs(T, active_components)

    if len(active_components) == 1:
        surface_x_full = np.zeros_like(x_full)
        surface_x_full[active_indices[0]] = 1.0

        _, _, _, A_full = pure_surface_inputs(T, components)

        return SurfaceTensionResult(
            sigma=float(sigma_i[0]),
            surface_x=surface_x_full,
            bulk_x=x_full,
            A=A_full,
            method="Butler-UNIFAC",
            converged=True,
            residual_max_abs=0.0,
            iterations=0,
        )

    ln_gamma_bulk = _call_unifac_ln_gamma(T, x_active, active_components, unifac_ln_gamma)

    # Generate a decent initial guess before using scipy.
    xs0, fp_sigma, fp_residual, fp_iters = _butler_fixed_point_initial_guess(
        T=T,
        xb=x_active,
        sigma_i=sigma_i,
        A=A,
        ln_gamma_bulk=ln_gamma_bulk,
        components=active_components,
        unifac_ln_gamma=unifac_ln_gamma,
        max_iter=100,
        tol=max(1e-10, residual_tol),
    )

    converged = False
    iterations = fp_iters
    warnings: list[str] = []

    try:
        from scipy.optimize import least_squares

        z0 = _softmax_inverse(xs0)

        scale = 1e-3  # 1 mN/m scaling, helps optimizer conditioning

        def residual_z(z: np.ndarray) -> np.ndarray:
            xs = _softmax_with_reference(z)
            component_sigmas = _butler_component_sigmas(
                T=T,
                xs=xs,
                xb=x_active,
                sigma_i=sigma_i,
                A=A,
                ln_gamma_bulk=ln_gamma_bulk,
                components=active_components,
                unifac_ln_gamma=unifac_ln_gamma,
            )
            return (component_sigmas[:-1] - component_sigmas[-1]) / scale

        opt = least_squares(
            residual_z,
            z0,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=max_nfev,
        )

        xs_active = _softmax_with_reference(opt.x)
        component_sigmas = _butler_component_sigmas(
            T=T,
            xs=xs_active,
            xb=x_active,
            sigma_i=sigma_i,
            A=A,
            ln_gamma_bulk=ln_gamma_bulk,
            components=active_components,
            unifac_ln_gamma=unifac_ln_gamma,
        )

        sigma_mix = float(np.mean(component_sigmas))
        residual_max_abs = float(np.max(np.abs(component_sigmas - sigma_mix)))
        converged = bool(opt.success and residual_max_abs <= residual_tol)
        iterations = int(opt.nfev)

        if not opt.success:
            warnings.append(f"SciPy least_squares did not report success: {opt.message}")

    except Exception as exc:
        # Fixed-point fallback. It is not as reliable as least_squares, but often
        # works for ordinary mixtures.
        xs_active = xs0
        component_sigmas = _butler_component_sigmas(
            T=T,
            xs=xs_active,
            xb=x_active,
            sigma_i=sigma_i,
            A=A,
            ln_gamma_bulk=ln_gamma_bulk,
            components=active_components,
            unifac_ln_gamma=unifac_ln_gamma,
        )

        sigma_mix = float(np.mean(component_sigmas))
        residual_max_abs = float(np.max(np.abs(component_sigmas - sigma_mix)))
        converged = residual_max_abs <= residual_tol
        warnings.append(
            "Used fixed-point Butler fallback because SciPy least_squares failed "
            f"or is unavailable: {exc!r}"
        )

        if fp_residual > residual_tol:
            warnings.append(
                f"Fixed-point initializer residual was {fp_residual:.3e} N/m."
            )

    surface_x_full = np.zeros_like(x_full)
    surface_x_full[active_indices] = xs_active

    _, _, _, A_full = pure_surface_inputs(T, components)

    if not converged:
        warnings.append(
            f"Butler residual {residual_max_abs:.3e} N/m exceeded tolerance "
            f"{residual_tol:.3e} N/m."
        )

    return SurfaceTensionResult(
        sigma=sigma_mix,
        surface_x=surface_x_full,
        bulk_x=x_full,
        A=A_full,
        method="Butler-UNIFAC",
        converged=converged,
        residual_max_abs=residual_max_abs,
        iterations=iterations,
        warnings=tuple(warnings),
    )


def estimate_mixture_surface_tension(
    T: float,
    x: Sequence[float],
    components: Sequence[SurfaceComponent],
    *,
    unifac_ln_gamma: UNIFACLnGamma | None = None,
    method: Literal["auto", "butler-unifac", "wsd"] = "auto",
) -> SurfaceTensionResult:
    """Convenience dispatcher.

    Args:
        T: K
        x: bulk mole fractions
        components: component data
        unifac_ln_gamma:
            required for method="butler-unifac"; optional for method="auto"
        method:
            "auto":
                use Butler-UNIFAC if UNIFAC is supplied, otherwise WSD
            "butler-unifac":
                use Butler surface-phase method
            "wsd":
                use Winterfeld-Scriven-Davis

    Returns:
        SurfaceTensionResult
    """

    if method == "auto":
        if unifac_ln_gamma is not None:
            return butler_unifac_surface_tension(T, x, components, unifac_ln_gamma)
        return winterfeld_scriven_davis_surface_tension(T, x, components)

    if method == "butler-unifac":
        if unifac_ln_gamma is None:
            raise ValueError("method='butler-unifac' requires unifac_ln_gamma.")
        return butler_unifac_surface_tension(T, x, components, unifac_ln_gamma)

    if method == "wsd":
        return winterfeld_scriven_davis_surface_tension(T, x, components)

    raise ValueError(f"Unknown surface-tension method: {method!r}")


def _call_unifac_ln_gamma(
    T: float,
    x: np.ndarray,
    components: Sequence[SurfaceComponent],
    unifac_ln_gamma: UNIFACLnGamma,
) -> np.ndarray:
    ln_gamma = np.asarray(unifac_ln_gamma(T, x.copy(), components), dtype=float)

    if ln_gamma.shape != x.shape:
        raise ValueError(
            "UNIFAC callback must return an array with the same shape as x."
        )

    if np.any(~np.isfinite(ln_gamma)):
        raise ValueError("UNIFAC callback returned non-finite ln(gamma).")

    return ln_gamma


def _butler_component_sigmas(
    *,
    T: float,
    xs: np.ndarray,
    xb: np.ndarray,
    sigma_i: np.ndarray,
    A: np.ndarray,
    ln_gamma_bulk: np.ndarray,
    components: Sequence[SurfaceComponent],
    unifac_ln_gamma: UNIFACLnGamma,
) -> np.ndarray:
    xs = normalize_mole_fractions(xs)
    xb = normalize_mole_fractions(xb)

    eps = 1e-300
    xs_safe = np.maximum(xs, eps)
    xb_safe = np.maximum(xb, eps)

    ln_gamma_surface = _call_unifac_ln_gamma(T, xs, components, unifac_ln_gamma)

    return sigma_i + (R * T / A) * (
        np.log(xs_safe)
        + ln_gamma_surface
        - np.log(xb_safe)
        - ln_gamma_bulk
    )


def _butler_fixed_point_initial_guess(
    *,
    T: float,
    xb: np.ndarray,
    sigma_i: np.ndarray,
    A: np.ndarray,
    ln_gamma_bulk: np.ndarray,
    components: Sequence[SurfaceComponent],
    unifac_ln_gamma: UNIFACLnGamma,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float, float, int]:
    """Fixed-point initializer for Butler.

    Rearranges Butler to:

        x_i^s = x_i^b gamma_i^b / gamma_i^s
                * exp[(sigma - sigma_i) A_i / RT]

    Given gamma_i^s, sigma is found from sum_i x_i^s = 1.
    """

    xs = xb.copy()
    sigma_guess = float(np.dot(xb, sigma_i))
    residual_max_abs = math.inf

    for k in range(1, max_iter + 1):
        ln_gamma_surface = _call_unifac_ln_gamma(T, xs, components, unifac_ln_gamma)

        sigma_new, xs_new = _solve_butler_sigma_given_surface_activity(
            T=T,
            xb=xb,
            sigma_i=sigma_i,
            A=A,
            ln_gamma_bulk=ln_gamma_bulk,
            ln_gamma_surface=ln_gamma_surface,
        )

        # Log-space damping keeps xs positive and is usually calmer when one
        # component surface-enriches strongly.
        damping = 0.5
        log_xs = (1.0 - damping) * np.log(np.maximum(xs, 1e-300))
        log_xs += damping * np.log(np.maximum(xs_new, 1e-300))
        xs = _normalize_from_logs(log_xs)
        sigma_guess = sigma_new

        component_sigmas = _butler_component_sigmas(
            T=T,
            xs=xs,
            xb=xb,
            sigma_i=sigma_i,
            A=A,
            ln_gamma_bulk=ln_gamma_bulk,
            components=components,
            unifac_ln_gamma=unifac_ln_gamma,
        )

        sigma_avg = float(np.mean(component_sigmas))
        residual_max_abs = float(np.max(np.abs(component_sigmas - sigma_avg)))

        if residual_max_abs <= tol:
            return xs, sigma_guess, residual_max_abs, k

    return xs, sigma_guess, residual_max_abs, max_iter


def _solve_butler_sigma_given_surface_activity(
    *,
    T: float,
    xb: np.ndarray,
    sigma_i: np.ndarray,
    A: np.ndarray,
    ln_gamma_bulk: np.ndarray,
    ln_gamma_surface: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Solve scalar sigma when gamma_surface is held fixed."""

    alpha = A / (R * T)

    log_b = (
        np.log(np.maximum(xb, 1e-300))
        + ln_gamma_bulk
        - ln_gamma_surface
        - alpha * sigma_i
    )

    def f(sigma: float) -> float:
        # We need sum_i exp(log_b_i + alpha_i * sigma) = 1.
        # Use logsumexp for stability.
        return _logsumexp(log_b + alpha * sigma)

    # f(sigma) is monotonic increasing. Root is f(sigma) = 0.
    lo = float(np.min(sigma_i) - 0.5)
    hi = float(np.max(sigma_i) + 0.5)

    # Expand bracket if needed.
    step = 0.5
    for _ in range(100):
        if f(lo) <= 0.0:
            break
        lo -= step
        step *= 2.0
    else:
        raise RuntimeError("Could not bracket lower Butler sigma root.")

    step = 0.5
    for _ in range(100):
        if f(hi) >= 0.0:
            break
        hi += step
        step *= 2.0
    else:
        raise RuntimeError("Could not bracket upper Butler sigma root.")

    # Bisection.
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid

    sigma = 0.5 * (lo + hi)
    log_xs_unnorm = log_b + alpha * sigma
    xs = _normalize_from_logs(log_xs_unnorm)

    return sigma, xs


def _softmax_with_reference(z: np.ndarray) -> np.ndarray:
    """Softmax for n components using n-1 free variables.

    Last logit is fixed to zero to remove degeneracy.
    """

    z_full = np.empty(len(z) + 1, dtype=float)
    z_full[:-1] = z
    z_full[-1] = 0.0

    return _normalize_from_logs(z_full)


def _softmax_inverse(x: np.ndarray) -> np.ndarray:
    """Inverse of _softmax_with_reference."""

    x = normalize_mole_fractions(x)
    eps = 1e-300
    return np.log(np.maximum(x[:-1], eps)) - np.log(max(float(x[-1]), eps))


def _normalize_from_logs(log_values: np.ndarray) -> np.ndarray:
    log_values = np.asarray(log_values, dtype=float)

    if np.any(~np.isfinite(log_values)):
        raise ValueError("Cannot normalize non-finite log values.")

    lse = _logsumexp(log_values)
    return np.exp(log_values - lse)


def _logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))