"""Shared analytic logarithmic property correlations and derivatives."""

import math


def dippr101_log_value(T, coefficients):
    """ln(value)=A+B/T+C ln(T)+D T**E, with T in kelvin."""
    return (
        coefficients.get("A", 0.0)
        + coefficients.get("B", 0.0) / T
        + coefficients.get("C", 0.0) * math.log(T)
        + coefficients.get("D", 0.0) * T ** coefficients.get("E", 1.0)
    )


def dippr101_log_derivative(T, coefficients):
    exponent = coefficients.get("E", 1.0)
    return (
        -coefficients.get("B", 0.0) / T**2
        + coefficients.get("C", 0.0) / T
        + coefficients.get("D", 0.0) * exponent * T ** (exponent - 1)
    )
