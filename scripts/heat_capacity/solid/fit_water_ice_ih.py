#!/usr/bin/env python3
"""Fit portable water ice-Ih Cp and density correlations to IAPWS R10-06."""

from __future__ import annotations

import json
import warnings

import numpy as np
from iapws import _Ice
from iapws._iapws import M


TMIN_K = 100.0
TMAX_K = 273.16
PRESSURE_MPA = 0.101325
POINTS = 10_001


def fit(values: np.ndarray, x: np.ndarray) -> dict:
    coefficients = np.polynomial.polynomial.polyfit(x, values, 5)
    predicted = np.polynomial.polynomial.polyval(x, coefficients)
    relative = np.abs(predicted / values - 1.0)
    return {
        "coefficients": dict(zip("ABCDEF", map(float, coefficients), strict=True)),
        "maximum_absolute_error": float(np.max(np.abs(predicted - values))),
        "mean_relative_error_percent": float(np.mean(relative) * 100.0),
        "maximum_relative_error_percent": float(np.max(relative) * 100.0),
    }


def main() -> None:
    temperatures = np.linspace(TMIN_K, TMAX_K, POINTS)
    x = (temperatures - 298.15) / 100.0
    # At 1 atm, IAPWS flags the final 0.0075 K through the triple point as
    # metastable ice. Retain that published continuation in the portable fit.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Metastable ice")
        states = [
            _Ice(float(temperature), PRESSURE_MPA)
            for temperature in temperatures
        ]
    report = {
        "source": "IAPWS R10-06(2009) ice Ih via iapws",
        "iapws_pressure_MPa": PRESSURE_MPA,
        "Tmin_K": TMIN_K,
        "Tmax_K": TMAX_K,
        "points": POINTS,
        "equation": "A+B*x+C*x^2+D*x^3+E*x^4+F*x^5; x=(T-298.15 K)/100",
        "Cps_J_mol_K": fit(
            np.asarray([state["cp"] * M for state in states]),
            x,
        ),
        "rhos_kg_m3": fit(
            np.asarray([state["rho"] for state in states]),
            x,
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
