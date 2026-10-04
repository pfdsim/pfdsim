"""Portable custom Psat definitions normalized into existing PFD contracts."""

import math

if __package__.split(".", 1)[0] == "pfdsim":
    from ..unit_conversions import pressure_unit_factor
else:
    from unit_conversions import pressure_unit_factor

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
