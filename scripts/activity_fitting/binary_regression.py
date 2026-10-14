"""Differentiable binary activity equations and numerical regression checks.

These source-fitting equations accept complex parameter perturbations. Runtime
thermodynamic models keep their own validation, limits and phase conventions.
"""

import numpy as np
from scipy.optimize import least_squares


def log_activity(model, x1, tau12, tau21, *, alpha=0.3, r=None, q=None):
    """Binary NRTL or standard UNIQUAC ln(gamma), with a trailing phase axis."""
    x, t12, t21 = np.broadcast_arrays(x1, tau12, tau21)
    other = 1.0 - x
    if model == "NRTL":
        g12, g21 = np.exp(-alpha * t12), np.exp(-alpha * t21)
        den1, den2 = x + other * g21, other + x * g12
        first = other**2 * (t21 * (g21 / den1)**2 + t12 * g12 / den2**2)
        second = x**2 * (t12 * (g12 / den2)**2 + t21 * g21 / den1**2)
    elif model == "UNIQUAC":
        r, q = np.asarray(r), np.asarray(q)
        rx, qx = x * r[0] + other * r[1], x * q[0] + other * q[1]
        ell = 5.0 * (r - q) - (r - 1.0)
        xl = x * ell[0] + other * ell[1]
        theta1, theta2 = x * q[0] / qx, other * q[1] / qx
        den1, den2 = theta1 + theta2 * t21, theta1 * t12 + theta2
        first = (np.log(r[0] / rx) + 5 * q[0] * np.log(q[0] * rx / (r[0] * qx))
                 + ell[0] - r[0] * xl / rx
                 + q[0] * (1 - np.log(den1) - theta1 / den1 - theta2 * t12 / den2))
        second = (np.log(r[1] / rx) + 5 * q[1] * np.log(q[1] * rx / (r[1] * qx))
                  + ell[1] - r[1] * xl / rx
                  + q[1] * (1 - np.log(den2) - theta1 * t21 / den1 - theta2 / den2))
    else:
        raise ValueError(f"Unknown binary activity model: {model}")
    return np.stack((first, second), axis=-1)


def gibbs_temperature_derivative(model, x1, tau12, tau21, d12, d21, *, alpha=0.3, q=None):
    """Exact d(gE/RT)/dT at fixed composition and constant alpha/r/q."""
    x, t12, t21, d12, d21 = np.broadcast_arrays(x1, tau12, tau21, d12, d21)
    other = 1.0 - x
    if model == "NRTL":
        g12, g21 = np.exp(-alpha * t12), np.exp(-alpha * t21)
        den1, den2 = x + other * g21, other + x * g12
        return x * other * (d21 * g21 * (den1 - alpha * t21 * x) / den1**2
                            + d12 * g12 * (den2 - alpha * t12 * other) / den2**2)
    if model == "UNIQUAC":
        q = np.asarray(q)
        theta1 = x * q[0] / (x * q[0] + other * q[1])
        theta2 = 1 - theta1
        return -(x * q[0] * theta2 * d21 / (theta1 + theta2 * t21)
                 + other * q[1] * theta1 * d12 / (theta1 * t12 + theta2))
    raise ValueError(f"Unknown binary activity model: {model}")


def complex_jacobian(function, parameters):
    """Cancellation-free Jacobian for the differentiable source equations."""
    parameters = np.asarray(parameters, dtype=float)
    step = 1e-25
    columns = []
    for index in range(len(parameters)):
        perturbed = parameters.astype(complex)
        perturbed[index] += 1j * step
        columns.append(np.imag(function(perturbed)) / step)
    return np.column_stack(columns)


def central_jacobian(function, parameters, relative_step=1e-4):
    """Scale-aware derivative for real-only runtime property calculations.

    A larger, symmetric step avoids differentiating runtime solver roundoff
    at sqrt(machine epsilon). It is not used for the analytic source models.
    """
    parameters = np.asarray(parameters, dtype=float)
    columns = []
    for index in range(len(parameters)):
        shift = np.zeros_like(parameters)
        shift[index] = relative_step * max(1.0, abs(parameters[index]))
        columns.append((function(parameters + shift) - function(parameters - shift))
                       / (2 * shift[index]))
    return np.column_stack(columns)


def projected_gradient_cosine(residuals, jacobian, parameters, lower, upper):
    """Stationarity independent of parameter units and residual magnitude.

    Outward cost gradients at active bounds satisfy the constrained optimum.
    Zero Jacobian columns supply no local descent direction.
    """
    residuals, jacobian = np.asarray(residuals), np.asarray(jacobian)
    parameters = np.asarray(parameters)
    lower, upper = np.broadcast_arrays(lower, upper, parameters)[:2]
    gradient = jacobian.T @ residuals
    distance = 1e-10 * np.maximum(1.0, np.abs(parameters))
    active = (((parameters - lower <= distance) & (gradient >= 0))
              | ((upper - parameters <= distance) & (gradient <= 0)))
    norms = np.linalg.norm(jacobian, axis=0)
    normalized = np.divide(gradient, norms, out=np.zeros_like(gradient), where=norms > 0)
    normalized[active] = 0.0
    return float(np.max(np.abs(normalized)) / max(1.0, np.linalg.norm(residuals)))


def fit_regression(function, starts, bounds, *, max_nfev=1500, jacobian=complex_jacobian):
    """Choose a finite, converged, stationary fit from the supplied starts."""
    solutions, failures = [], []
    for initial in starts:
        try:
            result = least_squares(
                function, initial, bounds=bounds,
                jac=lambda values: jacobian(function, values),
                x_scale="jac", max_nfev=max_nfev,
                ftol=1e-15, xtol=1e-14, gtol=1e-13,
            )
            if not (result.success and np.isfinite(result.x).all()
                    and np.isfinite(result.fun).all() and np.isfinite(result.jac).all()):
                raise ValueError(str(result.message))
            stationarity = projected_gradient_cosine(result.fun, result.jac, result.x, *bounds)
            if stationarity > 1e-7:
                raise ValueError(f"projected gradient cosine={stationarity:g}")
            solutions.append(result)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            failures.append(str(error))
    if not solutions:
        raise RuntimeError("No converged stationary source regression: " + "; ".join(failures))
    return min(solutions, key=lambda result: float(result.fun @ result.fun))
