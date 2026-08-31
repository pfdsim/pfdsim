#!/usr/bin/env python3
"""Reproduce selected PSRK curves from Horstmann et al. (2005).

The reference anchors below were digitized from the PSRK curves (solid lines),
not the experimental symbols, in Figures 2, 3, and 6 of Fluid Phase
Equilibria 227 (2005) 157-164.  Their precision is intentionally limited to
what the published raster figures support.  This script therefore validates
the implementation against the published model calculation; it is not a new
regression against experimental data.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermodynamics_models.psrk import PSRK, PSRKError, R_BAR_CM3_PER_MOL_K


HCN = "74-90-8"
WATER = "7732-18-5"
OXYGEN = "7782-44-7"
OZONE = "10028-15-6"
CHLORINE = "7782-50-5"
SULFUR_DIOXIDE = "7446-09-5"
METHANE = "74-82-8"


@dataclass(frozen=True)
class ValidationResult:
    label: str
    calculated: tuple[float, ...]
    reference: tuple[float, ...]
    metric: str
    value: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.value <= self.tolerance


def _logistic(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _mean_absolute_relative_percent(
    calculated: tuple[float, ...],
    reference: tuple[float, ...],
) -> float:
    return 100.0 * sum(
        abs(calculated_value / reference_value - 1.0)
        for calculated_value, reference_value in zip(calculated, reference)
    ) / len(reference)


def _bubble_state(
    model: PSRK,
    temperature: float,
    liquid_fraction_1: float,
    pressure_guess_bar: float,
    vapor_fraction_guess: float,
) -> tuple[float, float]:
    component_1, component_2 = model.components
    x1 = liquid_fraction_1

    # All Figure 2 systems use component 1 as the more volatile component.
    # Parameterizing y1 between x1 and one excludes the ever-present trivial
    # single-phase solution x1 == y1.
    def decode(values: np.ndarray) -> tuple[float, float]:
        pressure = math.exp(float(values[0]))
        fraction = _logistic(float(values[1]))
        return pressure, x1 + (1.0 - x1) * fraction

    def residual(values: np.ndarray) -> list[float]:
        pressure, y1 = decode(values)
        liquid = {component_1: x1, component_2: 1.0 - x1}
        vapor = {component_1: y1, component_2: 1.0 - y1}
        try:
            phi_liquid = model.fugacity_coefficients(
                temperature, pressure, liquid, "liquid"
            )
            phi_vapor = model.fugacity_coefficients(
                temperature, pressure, vapor, "vapor"
            )
            return [
                math.log(x1 * phi_liquid[component_1])
                - math.log(y1 * phi_vapor[component_1]),
                math.log((1.0 - x1) * phi_liquid[component_2])
                - math.log((1.0 - y1) * phi_vapor[component_2]),
            ]
        except (PSRKError, OverflowError, ValueError):
            return [100.0, 100.0]

    relative_vapor_guess = (
        vapor_fraction_guess - x1
    ) / (1.0 - x1)
    relative_vapor_guess = min(1.0 - 1.0e-8, max(1.0e-8, relative_vapor_guess))
    solution = least_squares(
        residual,
        [math.log(pressure_guess_bar), _logit(relative_vapor_guess)],
        bounds=([-14.0, -18.0], [14.0, 18.0]),
        max_nfev=5000,
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
    )
    if np.linalg.norm(residual(solution.x)) > 1.0e-7:
        raise RuntimeError("PSRK bubble-state solve did not converge")
    return decode(solution.x)


def _figure_2_bubble_pressures(
    components: tuple[str, str],
    temperature: float,
    liquid_fractions: tuple[float, ...],
    initial_pressure_bar: float,
) -> tuple[float, ...]:
    model = PSRK(components)
    pressure = initial_pressure_bar
    vapor_fraction = 0.9
    result: list[float] = []
    for liquid_fraction in liquid_fractions:
        pressure, vapor_fraction = _bubble_state(
            model,
            temperature,
            liquid_fraction,
            pressure,
            max(vapor_fraction, liquid_fraction + 0.01),
        )
        result.append(100.0 * pressure)  # bar -> kPa
    return tuple(result)


def _azeotropic_fraction(
    model: PSRK,
    temperature: float,
    pressure_guess_bar: float,
    fraction_guess: float,
) -> tuple[float, float]:
    component_1, component_2 = model.components

    def decode(values: np.ndarray) -> tuple[float, float]:
        return math.exp(float(values[0])), _logistic(float(values[1]))

    def residual(values: np.ndarray) -> list[float]:
        pressure, fraction = decode(values)
        composition = {
            component_1: fraction,
            component_2: 1.0 - fraction,
        }
        try:
            phi_liquid = model.fugacity_coefficients(
                temperature, pressure, composition, "liquid"
            )
            phi_vapor = model.fugacity_coefficients(
                temperature, pressure, composition, "vapor"
            )
            return [
                math.log(phi_liquid[component_1] / phi_vapor[component_1]),
                math.log(phi_liquid[component_2] / phi_vapor[component_2]),
            ]
        except (PSRKError, OverflowError, ValueError):
            return [100.0, 100.0]

    solution = least_squares(
        residual,
        [math.log(pressure_guess_bar), _logit(fraction_guess)],
        bounds=([-14.0, -12.0], [14.0, 12.0]),
        max_nfev=5000,
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
    )
    if np.linalg.norm(residual(solution.x)) > 1.0e-8:
        raise RuntimeError("PSRK azeotrope solve did not converge")
    return decode(solution.x)


def _figure_3_azeotropes(temperatures: tuple[float, ...]) -> tuple[float, ...]:
    model = PSRK((CHLORINE, SULFUR_DIOXIDE))
    pressure = 1.0
    fraction = 0.85
    result: list[float] = []
    for temperature in temperatures:
        pressure, fraction = _azeotropic_fraction(
            model, temperature, pressure, fraction
        )
        result.append(fraction)
    return tuple(result)


def _pure_saturation_pressure(model: PSRK, temperature: float) -> float:
    cas = model.components[0]
    critical_pressure = model._component_data[0].Pc_bar

    def root_count(pressure: float) -> int:
        mixture = model._mixture_parameters(temperature, (1.0,))
        B = mixture.b * pressure / (R_BAR_CM3_PER_MOL_K * temperature)
        return len(model._compressibility_roots(mixture.D * B, B))

    def residual(pressure: float) -> float:
        phi_liquid = model.fugacity_coefficients(
            temperature, pressure, {cas: 1.0}, "liquid"
        )[cas]
        phi_vapor = model.fugacity_coefficients(
            temperature, pressure, {cas: 1.0}, "vapor"
        )[cas]
        return math.log(phi_liquid / phi_vapor)

    previous: tuple[float, float] | None = None
    for pressure in np.geomspace(1.0e-7, 0.999 * critical_pressure, 1800):
        if root_count(float(pressure)) < 2:
            continue
        value = residual(float(pressure))
        if previous is not None and previous[1] * value < 0.0:
            return brentq(residual, previous[0], float(pressure), xtol=1.0e-12)
        previous = float(pressure), value
    raise RuntimeError(f"No PSRK saturation pressure found at {temperature:g} K")


def _figure_6_henry_coefficients(
    temperatures: tuple[float, ...],
) -> tuple[float, ...]:
    pure_water = PSRK((WATER,))
    methane_water = PSRK((METHANE, WATER))
    dilute_fraction = 1.0e-9
    result: list[float] = []
    for temperature in temperatures:
        pressure = _pure_saturation_pressure(pure_water, temperature)
        phi_methane = methane_water.fugacity_coefficients(
            temperature,
            pressure,
            {METHANE: dilute_fraction, WATER: 1.0 - dilute_fraction},
            "liquid",
        )[METHANE]
        result.append(phi_methane * pressure / 10.0)  # bar -> MPa
    return tuple(result)


def run_validation() -> list[ValidationResult]:
    hcn_fractions = (0.10, 0.20, 0.50, 0.80)
    hcn_calculated = _figure_2_bubble_pressures(
        (HCN, WATER), 291.15, hcn_fractions, 0.4
    )
    hcn_reference = (39.0, 52.5, 60.5, 67.0)

    oxygen_fractions = (0.20, 0.40, 0.80)
    oxygen_calculated = _figure_2_bubble_pressures(
        (OXYGEN, OZONE), 90.25, oxygen_fractions, 0.8
    )
    oxygen_reference = (79.5, 93.0, 93.0)

    azeotrope_temperatures = (228.15, 243.15, 273.15, 323.15)
    azeotrope_calculated = _figure_3_azeotropes(azeotrope_temperatures)
    azeotrope_reference = (0.871, 0.852, 0.821, 0.778)

    henry_temperatures = (300.0, 340.0, 380.0, 420.0, 450.0, 500.0, 550.0, 580.0)
    henry_calculated = _figure_6_henry_coefficients(henry_temperatures)
    henry_reference = (3940.0, 5860.0, 6320.0, 5500.0, 4430.0, 2630.0, 1310.0, 790.0)

    return [
        ValidationResult(
            "Fig. 2a HCN/water bubble pressure [kPa]",
            hcn_calculated,
            hcn_reference,
            "AAD [%]",
            _mean_absolute_relative_percent(hcn_calculated, hcn_reference),
            3.0,
        ),
        ValidationResult(
            "Fig. 2c oxygen/ozone bubble pressure [kPa]",
            oxygen_calculated,
            oxygen_reference,
            "AAD [%]",
            _mean_absolute_relative_percent(oxygen_calculated, oxygen_reference),
            3.0,
        ),
        ValidationResult(
            "Fig. 3b chlorine/SO2 azeotropic y1",
            azeotrope_calculated,
            azeotrope_reference,
            "max |delta y1|",
            max(
                abs(calculated - reference)
                for calculated, reference in zip(
                    azeotrope_calculated, azeotrope_reference
                )
            ),
            0.01,
        ),
        ValidationResult(
            "Fig. 6 methane/water Henry coefficient [MPa]",
            henry_calculated,
            henry_reference,
            "AAD [%]",
            _mean_absolute_relative_percent(henry_calculated, henry_reference),
            5.0,
        ),
    ]


def main() -> int:
    results = run_validation()
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}  {result.label}")
        print("  calculated:", ", ".join(f"{value:.7g}" for value in result.calculated))
        print("  paper curve:", ", ".join(f"{value:.7g}" for value in result.reference))
        print(
            f"  {result.metric}: {result.value:.6g} "
            f"(graphical tolerance {result.tolerance:g})"
        )
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
