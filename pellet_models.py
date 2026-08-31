"""Analytical and generalized effectiveness factors for porous catalyst pellets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional


class PelletModelError(ValueError):
    """Raised when a pellet-effectiveness calculation is inadmissible."""


@dataclass(frozen=True)
class SphericalEffectiveness:
    effectiveness_factor: float
    thiele_modulus: float
    generalized_thiele_modulus: float
    apparent_order: float
    method: str
    center_concentration_ratio: Optional[float] = None
    radial_nodes: int = 0


@dataclass(frozen=True)
class SphericalPowerLawNetworkEffectiveness:
    effectiveness_factors: tuple[float, ...]
    thiele_moduli: tuple[float, ...]
    apparent_orders: tuple[float, ...]
    overall_limiting_species_effectiveness: float
    center_concentration_ratio: float
    radial_nodes: int
    method: str = 'rigorous_power_law_network_sphere'


def _positive_finite(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise PelletModelError(f"{label} must be positive and finite") from error
    if not math.isfinite(number) or number <= 0.0:
        raise PelletModelError(f"{label} must be positive and finite")
    return number


def first_order_spherical_effectiveness(thiele_modulus: float) -> float:
    """Exact isothermal first-order effectiveness factor for a sphere."""
    phi = float(thiele_modulus)
    if not math.isfinite(phi) or phi < 0.0:
        raise PelletModelError("Thiele modulus must be nonnegative and finite")
    if phi < 1.0e-5:
        phi2 = phi * phi
        return 1.0 - phi2 / 15.0 + 2.0 * phi2 * phi2 / 315.0
    if phi > 50.0:
        return 3.0 / phi * (1.0 - 1.0 / phi)
    return 3.0 / phi * (1.0 / math.tanh(phi) - 1.0 / phi)


def generalized_power_law_spherical_effectiveness(
    thiele_modulus: float,
    apparent_order: float,
) -> SphericalEffectiveness:
    """Generalized-Thiele approximation for a positive-order power law.

    ``thiele_modulus`` is based on the surface limiting-species consumption:
    ``R*sqrt(rho_particle*q_surface/(D_eff*C_surface))``. The generalized
    modulus preserves the exact first-order solution and the standard
    strong-diffusion asymptote for an n-th-order rate.
    """
    phi = float(thiele_modulus)
    if not math.isfinite(phi) or phi < 0.0:
        raise PelletModelError("Thiele modulus must be nonnegative and finite")
    order = _positive_finite(apparent_order, "Apparent reaction order")
    generalized = phi * math.sqrt((order + 1.0) / 2.0)
    eta = first_order_spherical_effectiveness(generalized)
    return SphericalEffectiveness(
        effectiveness_factor=eta,
        thiele_modulus=phi,
        generalized_thiele_modulus=generalized,
        apparent_order=order,
        method=(
            'exact_first_order_sphere'
            if abs(order - 1.0) <= 1.0e-12
            else 'generalized_thiele_power_law_sphere'
        ),
    )


def rigorous_power_law_spherical_effectiveness(
    thiele_modulus: float,
    apparent_order: float,
    *,
    relative_tolerance: float = 1.0e-6,
    maximum_nodes: int = 2000,
) -> SphericalEffectiveness:
    """Solve the isothermal one-species spherical power-law pellet BVP.

    The dimensionless concentration ``u=C/C_surface`` satisfies
    ``u'' + 2*u'/x = phi**2*u**n`` with center symmetry and ``u(1)=1``.
    Nonconverged fractional-order dead-core cases are rejected.
    """
    phi = float(thiele_modulus)
    if not math.isfinite(phi) or phi < 0.0:
        raise PelletModelError("Thiele modulus must be nonnegative and finite")
    order = _positive_finite(apparent_order, "Apparent reaction order")
    tolerance = _positive_finite(relative_tolerance, "Pellet BVP tolerance")
    nodes_limit = int(maximum_nodes)
    if nodes_limit < 50:
        raise PelletModelError("Pellet BVP maximum_nodes must be at least 50")
    if phi == 0.0:
        return SphericalEffectiveness(
            1.0, 0.0, 0.0, order, 'rigorous_power_law_sphere', 1.0, 2
        )
    if abs(order - 1.0) <= 1.0e-12:
        eta = first_order_spherical_effectiveness(phi)
        center = (
            phi / math.sinh(phi)
            if phi < 700.0 else 0.0
        )
        return SphericalEffectiveness(
            eta,
            phi,
            phi,
            order,
            'exact_first_order_sphere',
            center,
            2,
        )

    import numpy as np
    from scipy.integrate import solve_bvp

    generalized = phi * math.sqrt((order + 1.0) / 2.0)
    coordinate = np.linspace(0.0, 1.0, 51)
    if generalized < 50.0:
        concentration = np.ones_like(coordinate)
        positive = coordinate > 0.0
        concentration[positive] = (
            np.sinh(generalized * coordinate[positive])
            / (coordinate[positive] * math.sinh(generalized))
        )
        concentration[0] = generalized / math.sinh(generalized)
    else:
        concentration = np.exp(-generalized * (1.0 - coordinate))
        concentration[0] = 0.0
    gradient = np.gradient(concentration, coordinate)
    gradient[0] = 0.0
    initial = np.vstack((concentration, gradient))

    def equations(_coordinate, values):
        return np.vstack((
            values[1],
            phi * phi * np.maximum(values[0], 0.0) ** order,
        ))

    def boundaries(center, surface):
        return np.asarray((center[1], surface[0] - 1.0))

    singular = np.asarray(((0.0, 0.0), (0.0, -2.0)))
    solution = solve_bvp(
        equations,
        boundaries,
        coordinate,
        initial,
        S=singular,
        tol=tolerance,
        max_nodes=nodes_limit,
    )
    minimum = float(np.min(solution.y[0]))
    if solution.status != 0 or minimum < -max(1.0e-8, 10.0 * tolerance):
        detail = (
            "; fractional-order kinetics may have formed a dead core"
            if order < 1.0 else ""
        )
        raise PelletModelError(
            "Rigorous spherical pellet BVP did not converge: "
            + str(solution.message) + detail
        )
    integration_grid = np.linspace(0.0, 1.0, 1001)
    profile = np.maximum(solution.sol(integration_grid)[0], 0.0)
    eta = float(3.0 * np.trapezoid(
        profile**order * integration_grid**2,
        integration_grid,
    ))
    if not math.isfinite(eta) or eta <= 0.0 or eta > 1.0 + 1.0e-5:
        raise PelletModelError(
            f"Rigorous spherical pellet returned invalid effectiveness {eta:.6g}"
        )
    return SphericalEffectiveness(
        effectiveness_factor=min(eta, 1.0),
        thiele_modulus=phi,
        generalized_thiele_modulus=generalized,
        apparent_order=order,
        method='rigorous_power_law_sphere',
        center_concentration_ratio=max(float(solution.sol(0.0)[0]), 0.0),
        radial_nodes=len(solution.x),
    )


def rigorous_power_law_network_spherical_effectiveness(
    thiele_modulus_squared_terms: Iterable[float],
    apparent_orders: Iterable[float],
    *,
    relative_tolerance: float = 1.0e-6,
    maximum_nodes: int = 2000,
) -> SphericalPowerLawNetworkEffectiveness:
    """Solve one diffusing species consumed by several power-law pathways."""
    terms = tuple(float(value) for value in thiele_modulus_squared_terms)
    orders = tuple(float(value) for value in apparent_orders)
    if not terms or len(terms) != len(orders):
        raise PelletModelError(
            "Pellet network requires equal nonempty Thiele-term and order lists"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in terms):
        raise PelletModelError(
            "Pellet network Thiele-modulus-squared terms must be nonnegative"
        )
    for order in orders:
        _positive_finite(order, "Apparent reaction order")
    total_term = sum(terms)
    if total_term == 0.0:
        return SphericalPowerLawNetworkEffectiveness(
            effectiveness_factors=tuple(1.0 for _ in terms),
            thiele_moduli=tuple(0.0 for _ in terms),
            apparent_orders=orders,
            overall_limiting_species_effectiveness=1.0,
            center_concentration_ratio=1.0,
            radial_nodes=2,
        )
    if len(terms) == 1:
        result = rigorous_power_law_spherical_effectiveness(
            math.sqrt(terms[0]),
            orders[0],
            relative_tolerance=relative_tolerance,
            maximum_nodes=maximum_nodes,
        )
        return SphericalPowerLawNetworkEffectiveness(
            effectiveness_factors=(result.effectiveness_factor,),
            thiele_moduli=(result.thiele_modulus,),
            apparent_orders=orders,
            overall_limiting_species_effectiveness=result.effectiveness_factor,
            center_concentration_ratio=float(
                result.center_concentration_ratio
                if result.center_concentration_ratio is not None else 1.0
            ),
            radial_nodes=result.radial_nodes,
            method=result.method,
        )

    import numpy as np
    from scipy.integrate import solve_bvp

    tolerance = _positive_finite(relative_tolerance, "Pellet BVP tolerance")
    nodes_limit = int(maximum_nodes)
    if nodes_limit < 50:
        raise PelletModelError("Pellet BVP maximum_nodes must be at least 50")
    weighted_order = sum(
        term * order for term, order in zip(terms, orders)
    ) / total_term
    phi = math.sqrt(total_term)
    generalized = phi * math.sqrt((weighted_order + 1.0) / 2.0)
    coordinate = np.linspace(0.0, 1.0, 51)
    if generalized < 50.0:
        concentration = np.ones_like(coordinate)
        positive = coordinate > 0.0
        concentration[positive] = (
            np.sinh(generalized * coordinate[positive])
            / (coordinate[positive] * math.sinh(generalized))
        )
        concentration[0] = generalized / math.sinh(generalized)
    else:
        concentration = np.exp(-generalized * (1.0 - coordinate))
        concentration[0] = 0.0
    gradient = np.gradient(concentration, coordinate)
    gradient[0] = 0.0
    initial = np.vstack((concentration, gradient))

    def equations(_coordinate, values):
        bounded = np.maximum(values[0], 0.0)
        source = sum(
            term * bounded**order
            for term, order in zip(terms, orders)
        )
        return np.vstack((values[1], source))

    def boundaries(center, surface):
        return np.asarray((center[1], surface[0] - 1.0))

    solution = solve_bvp(
        equations,
        boundaries,
        coordinate,
        initial,
        S=np.asarray(((0.0, 0.0), (0.0, -2.0))),
        tol=tolerance,
        max_nodes=nodes_limit,
    )
    minimum = float(np.min(solution.y[0]))
    if solution.status != 0 or minimum < -max(1.0e-8, 10.0 * tolerance):
        detail = (
            "; fractional-order kinetics may have formed a dead core"
            if min(orders) < 1.0 else ""
        )
        raise PelletModelError(
            "Rigorous spherical pellet network BVP did not converge: "
            + str(solution.message) + detail
        )
    integration_grid = np.linspace(0.0, 1.0, 1001)
    profile = np.maximum(solution.sol(integration_grid)[0], 0.0)
    factors = tuple(float(3.0 * np.trapezoid(
        profile**order * integration_grid**2,
        integration_grid,
    )) for order in orders)
    if any(
        not math.isfinite(value) or value <= 0.0 or value > 1.0 + 1.0e-5
        for value in factors
    ):
        raise PelletModelError(
            "Rigorous spherical pellet network returned invalid effectiveness factors"
        )
    clipped = tuple(min(value, 1.0) for value in factors)
    overall = sum(
        term * factor for term, factor in zip(terms, clipped)
    ) / total_term
    return SphericalPowerLawNetworkEffectiveness(
        effectiveness_factors=clipped,
        thiele_moduli=tuple(math.sqrt(term) for term in terms),
        apparent_orders=orders,
        overall_limiting_species_effectiveness=overall,
        center_concentration_ratio=max(float(solution.sol(0.0)[0]), 0.0),
        radial_nodes=len(solution.x),
    )
