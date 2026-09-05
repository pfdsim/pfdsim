"""Cycle cake-filter hydraulics in SI units.

The constitutive assumptions and derivations are in docs/filtration.md.
These functions have no thermodynamic or flowsheet dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import gammaincc


def wash_profile(wash_ratio: float, cells: int) -> np.ndarray:
    """Mother-liquor tracer remaining in equal, perfectly mixed cells in series.

    The independent variable is delivered wash volume / total pore volume.
    Cell count is a physical mixing parameter, not a numerical mesh setting.
    """
    return gammaincc(np.arange(1, cells + 1), cells * wash_ratio)


def cycle_coefficients(
    *,
    dry_mass,
    filtrate_volume,
    pore_volume,
    wash_volume,
    alpha,
    medium_resistance,
    pressure_drop,
    feed_viscosity,
    wash_viscosity,
    wash_cells,
):
    """Return formation and washing (a, b), where duration = a/A**2 + b/A.

    Viscosity is volume-linearly interpolated between feed and wash liquids.
    Cake resistance uses the average cell viscosity; medium resistance uses
    the viscosity in the final cell, immediately above the medium.
    """
    formation = (
        feed_viscosity * alpha * dry_mass * filtrate_volume / (2 * pressure_drop),
        feed_viscosity * medium_resistance * filtrate_volume / pressure_drop,
    )
    if wash_volume == 0:
        return formation, (0.0, 0.0)
    ratio = wash_volume / pore_volume
    # Tracer integrals saturate near W=1; splitting the range prevents adaptive
    # quadrature from missing that transient for very large wash ratios.
    upper = min(ratio, 50.0)
    mean_integral = quad(
        lambda w: float(np.mean(wash_profile(w, wash_cells))),
        0.0,
        upper,
        epsabs=1e-11,
        epsrel=1e-10,
    )[0]
    exit_integral = quad(
        lambda w: float(wash_profile(w, wash_cells)[-1]),
        0.0,
        upper,
        epsabs=1e-11,
        epsrel=1e-10,
    )[0]
    delta_mu = feed_viscosity - wash_viscosity
    cake_integral = (
        wash_viscosity * wash_volume + delta_mu * pore_volume * mean_integral
    )
    medium_integral = (
        wash_viscosity * wash_volume + delta_mu * pore_volume * exit_integral
    )
    washing = (
        alpha * dry_mass * cake_integral / pressure_drop,
        medium_resistance * medium_integral / pressure_drop,
    )
    return formation, washing


def required_area(a: float, b: float, available_time: float) -> float:
    """Positive root of t A² - b A - a = 0."""
    return (b + math.hypot(b, 2 * math.sqrt(a * available_time))) / (2 * available_time)


def deliquor(
    *,
    duration,
    pore_volume,
    area,
    cake_resistance,
    medium_resistance,
    viscosity,
    pressure_drop,
    entry_pressure,
    residual_saturation,
    pore_index,
    relative_permeability_exponent,
):
    """Uniform-saturation Darcy drainage with a Brooks–Corey capillary law.

    Liquid leaves; gas is a pressure utility, with negligible gas resistance.
    No evaporation or further cake consolidation is represented. The liquid
    inventory is mixed for this stage. Returns final and equilibrium saturation.
    """
    equilibrium_effective = (
        (entry_pressure / pressure_drop) ** pore_index
        if entry_pressure < pressure_drop
        else 1.0
    )
    equilibrium = (
        residual_saturation + (1 - residual_saturation) * equilibrium_effective
    )
    if duration == 0 or pressure_drop <= entry_pressure:
        return 1.0, equilibrium

    def rhs(_time, state):
        saturation = max(equilibrium, min(1.0, float(state[0])))
        effective = (saturation - residual_saturation) / (1 - residual_saturation)
        capillary = entry_pressure * effective ** (-1 / pore_index)
        kr = effective**relative_permeability_exponent
        resistance = cake_resistance / kr + medium_resistance
        rate = area * max(0.0, pressure_drop - capillary) / (viscosity * resistance)
        return [-rate / pore_volume]

    solution = solve_ivp(
        rhs, (0.0, duration), [1.0], method="Radau", rtol=1e-8, atol=1e-10
    )
    if not solution.success:
        raise ValueError(f"Deliquoring integration failed: {solution.message}")
    saturation = float(solution.y[0, -1])
    if not math.isfinite(saturation) or saturation < equilibrium - 1e-7:
        raise ValueError(
            "Deliquoring integration violated the capillary equilibrium bound"
        )
    return max(equilibrium, min(1.0, saturation)), equilibrium
