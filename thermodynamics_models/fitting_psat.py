"""Portable custom Psat definitions normalized into existing PFD contracts."""

import math
from copy import deepcopy
from types import MethodType

from .common import ThermodynamicsError

if __package__.split(".", 1)[0] == "pfdsim":
    from ..unit_conversions import pressure_unit_factor
    from ..property_resolution.common import (
        AntoineCoefficients,
        PropertyResolutionResult,
    )
    from ..property_resolution.log_correlations import dippr101_log_value
    from ..property_resolution.vapor_pressure_adapter import (
        _psat_correlation_functions,
        _required_coefficients,
    )
else:
    from unit_conversions import pressure_unit_factor
    from property_resolution.common import AntoineCoefficients, PropertyResolutionResult
    from property_resolution.log_correlations import dippr101_log_value
    from property_resolution.vapor_pressure_adapter import (
        _psat_correlation_functions,
        _required_coefficients,
    )

PSAT_FORMS = {
    "antoine": {
        "label": "Antoine",
        "coefficients": "ABC",
        "equation": "log10(P/unit) = A − B/(T/unit + C)",
    },
    "dippr101": {
        "label": "DIPPR 101",
        "coefficients": "ABCDE",
        "equation": "ln(P/unit) = A + B/T + C ln(T) + D T^E; T in K",
    },
    "canonical_psat_af": {
        "label": "PFDSim canonical A–F",
        "coefficients": "ABCDEF",
        "equation": "ln(P/bar) = A + B/T + C ln(T) + D T + E T² + F T⁵",
    },
    "canonical_psat_ag": {
        "label": "PFDSim canonical A–G",
        "coefficients": "ABCDEFG",
        "equation": "A–F + G T³",
    },
    "canonical_psat_ah": {
        "label": "PFDSim canonical A–H",
        "coefficients": "ABCDEFGH",
        "equation": "A–G + H[(T/Tc)^n − 1]; n = −3, −5 or −7",
    },
}


def normalize_psat(spec):
    if spec is None:
        return None
    if not isinstance(spec, dict) or spec.keys() - {
        "form",
        "coefficients",
        "Tmin_K",
        "Tmax_K",
        "pressure_unit",
        "temperature_unit",
        "inverse_power",
        "source",
    }:
        raise ValueError(
            "Psat accepts form, coefficients, Tmin_K, Tmax_K, pressure_unit, temperature_unit, inverse_power and source."
        )
    form = spec.get("form")
    if form not in PSAT_FORMS:
        raise ValueError("Choose Antoine, DIPPR 101 or a PFDSim canonical Psat form.")

    def finite(value, label):
        if isinstance(value, bool):
            raise ValueError(f"Psat {label} must be finite.")
        try:
            value = float(value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Psat {label} must be finite.") from error
        if not math.isfinite(value):
            raise ValueError(f"Psat {label} must be finite.")
        return value

    raw = spec.get("coefficients")
    required = set(PSAT_FORMS[form]["coefficients"])
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError(
            f"Psat {form} requires exactly coefficients {', '.join(sorted(required))}."
        )
    coefficients = {key: finite(value, key) for key, value in raw.items()}
    low, high = [finite(spec.get(key), key) for key in ("Tmin_K", "Tmax_K")]
    if not 0 < low < high:
        raise ValueError("Psat requires 0 < Tmin_K < Tmax_K.")
    pressure_unit = spec.get("pressure_unit", "bar")
    if pressure_unit == "barg":
        raise ValueError("Psat pressure must be absolute, not gauge pressure.")
    pressure_unit_factor(pressure_unit, strict=True)
    if form.startswith("canonical") and pressure_unit != "bar":
        raise ValueError("PFDSim canonical coefficients are always for ln(P/bar).")
    temperature_unit = spec.get("temperature_unit", "C" if form == "antoine" else "K")
    if (
        temperature_unit not in ("C", "K")
        or form != "antoine"
        and temperature_unit != "K"
    ):
        raise ValueError(
            "Antoine supports Celsius or Kelvin denominators; DIPPR/canonical forms use Kelvin."
        )
    if form == "antoine":
        if (
            low - 273.15 + coefficients["C"] <= 0 <= high - 273.15 + coefficients["C"]
            and temperature_unit == "C"
            or low + coefficients["C"] <= 0 <= high + coefficients["C"]
            and temperature_unit == "K"
        ):
            raise ValueError(
                "Antoine has a singular denominator within its declared range."
            )
        if coefficients["B"] <= 0:
            raise ValueError(
                "Antoine B must be positive for increasing saturation pressure."
            )
    if form == "canonical_psat_ah" and spec.get("inverse_power") not in (-3, -5, -7):
        raise ValueError("Canonical A–H inverse_power must be −3, −5 or −7.")
    if form != "canonical_psat_ah" and "inverse_power" in spec:
        raise ValueError("inverse_power applies only to canonical A–H.")
    return {
        **spec,
        "form": form,
        "coefficients": coefficients,
        "Tmin_K": low,
        "Tmax_K": high,
        "pressure_unit": pressure_unit,
        "temperature_unit": temperature_unit,
    }


def apply_psat(component, spec):
    """Use the authoritative PFD/property-resolution implementation at runtime."""
    spec = normalize_psat(spec)
    if spec is None:
        return
    coefficient = dict(spec["coefficients"])
    low, high = spec["Tmin_K"], spec["Tmax_K"]
    if spec["form"] == "antoine":
        component.property_correlations.pop("Psat", None)
        component.antoine_A = coefficient["A"] + math.log10(
            pressure_unit_factor(spec["pressure_unit"], strict=True)
        )
        component.antoine_B = coefficient["B"]
        component.antoine_C = coefficient["C"] + (
            273.15 if spec["temperature_unit"] == "K" else 0
        )
        component.antoine_Tmin, component.antoine_Tmax = low, high
        component.antoine_source = spec.get("source") or "Fitting Psat override"
    else:
        for name in (
            "antoine_A",
            "antoine_B",
            "antoine_C",
            "antoine_Tmin",
            "antoine_Tmax",
            "antoine_source",
        ):
            setattr(component, name, None)
        if spec["form"] == "dippr101":
            coefficient["A"] += math.log(
                pressure_unit_factor(spec["pressure_unit"], strict=True)
            )
        correlation = {
            "equation": "dippr_eq101" if spec["form"] == "dippr101" else spec["form"],
            "coefficients": coefficient,
            "Tmin_K": low,
            "Tmax_K": high,
            "source": spec.get("source") or "Fitting Psat override",
        }
        if spec["form"] == "canonical_psat_ah":
            correlation["inverse_power"] = spec["inverse_power"]
        component.property_correlations["Psat"] = correlation


def has_supplied_psat(component):
    return "Psat" in component.property_correlations or component.antoine_A is not None


def supplied_psat_dependencies(component):
    equation = str(
        component.property_correlations.get("Psat", {}).get("equation", "")
    ).lower()
    return (
        {"Tc"}
        if equation == "canonical_psat_ah"
        else {"Tc", "Pc"}
        if equation in ("reduced_vapor_pressure", "psat_mercury")
        else set()
    )


class FittingPsatEvaluator:
    """Native supplied correlation, used only by the fitting context."""

    def __init__(self, component, properties):
        self.name = component.name
        self.correlation = deepcopy(component.property_correlations.get("Psat"))
        self.properties = deepcopy(properties.to_dict())
        self.values = {}
        if self.correlation is not None:
            self.equation = str(self.correlation["equation"]).lower()
            self.low = self.correlation.get("Tmin_K")
            self.high = self.correlation.get("Tmax_K")
            self.source = self.correlation.get("source") or "Provided Psat"
            self.quality = float(self.correlation.get("quality", 1))
            self.coefficients = _required_coefficients(self.correlation, self.equation)
            self.antoine = None
        else:
            self.equation = "antoine"
            self.low, self.high = component.antoine_Tmin, component.antoine_Tmax
            self.source = component.antoine_source or "Provided Antoine"
            self.quality = 1.0
            self.antoine = AntoineCoefficients(
                component.antoine_A,
                component.antoine_B,
                component.antoine_C,
                self.low,
                self.high,
            )
        if (
            self.low is None
            or self.high is None
            or not 0 < float(self.low) < float(self.high)
        ):
            raise ValueError(
                f"Supplied Psat for {self.name} needs a valid Tmin_K and Tmax_K for direct fitting evaluation."
            )
        self.low, self.high = float(self.low), float(self.high)

    def __call__(self, T):
        T = float(T)
        if not math.isfinite(T) or not self.low <= T <= self.high:
            raise ThermodynamicsError(
                f"Supplied Psat for {self.name}: {T:g} K is outside its declared range {self.low:g}–{self.high:g} K. Direct fitting evaluation does not extrapolate or clamp."
            )
        if T in self.values:
            return self.values[T]
        try:
            if self.antoine is not None:
                pressure = self.antoine.vapor_pressure(T)
            elif self.equation == "dippr_eq101":
                pressure = math.exp(dippr101_log_value(T, self.coefficients))
            elif self.equation in (
                "canonical_psat",
                "canonical_psat_af",
                "canonical_psat_ag",
                "canonical_psat_ah",
            ):
                # These are the supplied canonical equations themselves, with
                # no curve fit, critical anchoring or continuation policy.
                c = self.coefficients
                value = (
                    c["A"]
                    + c["B"] / T
                    + c["C"] * math.log(T)
                    + c["D"] * T
                    + c["E"] * T**2
                    + c["F"] * T**5
                )
                if self.equation in ("canonical_psat_ag", "canonical_psat_ah"):
                    value += c["G"] * T**3
                if self.equation == "canonical_psat_ah":
                    tc = self.correlation.get("Tc_K") or self.properties.get("Tc")
                    if tc is None or not math.isfinite(float(tc)) or float(tc) <= 0:
                        raise ValueError("Canonical A–H requires its actual Tc_K.")
                    value += c["H"] * (
                        (T / float(tc)) ** int(self.correlation["inverse_power"]) - 1
                    )
                pressure = math.exp(value)
            else:
                evaluate, _ = _psat_correlation_functions(
                    self.equation, self.coefficients, self.correlation, self.properties
                )
                pressure = math.exp(evaluate(T))
            if pressure is None or not math.isfinite(float(pressure)) or pressure <= 0:
                raise ValueError(
                    "Correlation returned nonpositive or nonfinite pressure."
                )
        except (ValueError, TypeError, OverflowError, ZeroDivisionError) as error:
            raise ThermodynamicsError(
                f"Cannot evaluate supplied Psat for {self.name} at {T:g} K: {error}"
            ) from error
        if len(self.values) > 2048:
            self.values.clear()
        self.values[T] = float(pressure)
        return self.values[T]

    def quality_samples(self, low, high):
        return [
            (
                T,
                PropertyResolutionResult(
                    self(T),
                    str(self.source),
                    f"fitter_direct_{self.equation}",
                    self.quality,
                    f"Supplied correlation evaluated directly; validity {self.low:g}–{self.high:g} K; pressure/bar",
                ),
            )
            for T in sorted({float(low), float(high), (float(low) + float(high)) / 2})
        ]


def install_fitting_psat(thermo, definition):
    """Override Psat on this fitter-owned instance only; export is unchanged."""
    evaluators = {
        component.symbol: FittingPsatEvaluator(
            component, thermo.props[component.symbol]
        )
        for component in definition.components
        if has_supplied_psat(component)
    }
    if not evaluators:
        return
    original = thermo.Psat

    def evaluate(self, component, T):
        direct = evaluators.get(component)
        return direct(T) if direct is not None else original(component, T)

    thermo.Psat = MethodType(evaluate, thermo)
    thermo._fitting_psat_evaluators = evaluators
    # Only fitter-owned per-state caches are invalidated; no shared providers,
    # coefficients, defaults or serialization contracts are changed.
    for name in (
        "_psat_cache",
        "_phi_sat_cache",
        "_poynting_cache",
        "_k_values_cache",
        "_pure_saturation_temperature_cache",
        "_vdm_phi_sat_cache",
        "_vdm_assoc_reference_cache",
    ):
        cache = getattr(thermo, name, None)
        if cache is not None:
            cache.clear()
