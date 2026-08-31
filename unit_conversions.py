"""Shared unit-conversion factors used by parsing and simulation."""

from typing import Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .pressure_standards import NORMAL_BOILING_PRESSURE_BAR
else:
    from pressure_standards import NORMAL_BOILING_PRESSURE_BAR


PRESSURE_UNIT_FACTORS_TO_BAR = {
    "": 1.0,
    "bar": 1.0,
    "barg": 1.0,
    "atm": NORMAL_BOILING_PRESSURE_BAR,
    "atmosphere": NORMAL_BOILING_PRESSURE_BAR,
    "atmospheres": NORMAL_BOILING_PRESSURE_BAR,
    "kpa": 0.01,
    "pa": 1.0e-5,
    "mpa": 10.0,
    "psi": 0.0689475729,
    "psia": 0.0689475729,
    "mmhg": 0.00133322368,
    "torr": 0.00133322368,
}


def pressure_unit_factor(
    unit: Optional[str],
    *,
    strict: bool = False,
) -> float:
    """Return the factor converting ``unit`` to bar."""
    normalized = (unit or "").strip().lower()
    factor = PRESSURE_UNIT_FACTORS_TO_BAR.get(normalized)
    if factor is None:
        if strict:
            raise ValueError(f"Unsupported pressure unit: {unit}")
        return 1.0
    return factor


def pressure_to_bar(
    value: float,
    unit: Optional[str],
    *,
    strict: bool = False,
) -> float:
    """Convert one numeric pressure to bar."""
    return float(value) * pressure_unit_factor(unit, strict=strict)


def temperature_to_kelvin(
    value: float,
    unit: Optional[str],
    *,
    infer_unitless_celsius_below: Optional[float] = None,
) -> float:
    """Convert a temperature to kelvin using existing PFD conventions."""
    temperature = float(value)
    normalized = (unit or "").strip().lower()
    if normalized in {"c", "°c", "celsius"}:
        return temperature + 273.15
    if normalized in {"f", "°f", "fahrenheit"}:
        return (temperature - 32.0) * 5.0 / 9.0 + 273.15
    if (
        not normalized
        and infer_unitless_celsius_below is not None
        and temperature < infer_unitless_celsius_below
    ):
        return temperature + 273.15
    return temperature


def mass_flow_to_kg_per_hour(
    value: float,
    unit: Optional[str],
) -> float:
    """Convert the supported mass-flow units to kg/h."""
    flow = float(value)
    normalized = (unit or "").strip().lower()
    if "lb" in normalized:
        return flow * 0.45359237
    if normalized in {"g/h", "g/hr", "gram/h", "grams/h"}:
        return flow / 1000.0
    if normalized == "kg/s":
        return flow * 3600.0
    return flow


def molar_flow_to_kmol_per_hour(
    value: float,
    unit: Optional[str],
) -> float:
    """Convert the supported molar-flow units to kmol/h."""
    flow = float(value)
    normalized = (unit or "").strip().lower()
    if normalized in {"mol/h", "mol/hr"}:
        return flow / 1000.0
    if normalized == "mol/s":
        return flow * 3.6
    if normalized == "kmol/s":
        return flow * 3600.0
    return flow
