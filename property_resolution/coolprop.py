"""Shared CoolProp identity and saturation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional


COOLPROP_ALIAS_PATH = Path(__file__).resolve().parent.parent / "data" / "coolprop_fluid_aliases.json"
COOLPROP_PROPERTY_QUALITY = 0.995

_COOLPROP_UNAVAILABLE = object()
_COOLPROP_MODULE = None
_COOLPROP_ALIAS_MAP = None
_COOLPROP_REFRIGERANT_CAS_ALIASES = None


@dataclass(frozen=True)
class CoolPropFluidReference:
    """A strict pure-fluid CoolProp match and its preferred backend."""

    fluid: str
    backend: str

    @property
    def qualified_name(self) -> str:
        return f"{self.backend}::{self.fluid}"

    @property
    def method_prefix(self) -> str:
        return f"coolprop_{self.backend}"


@dataclass(frozen=True)
class CoolPropMeltingLineDomain:
    """Declared HEOS fusion-line limits for one pure fluid."""

    P_min_pa: float
    P_max_pa: float
    T_min_K: float
    T_max_K: float

    def contains_pressure(self, pressure_pa: float) -> bool:
        try:
            pressure = float(pressure_pa)
        except (TypeError, ValueError):
            return False
        tolerance = 1.0e-10 * max(abs(self.P_min_pa), abs(self.P_max_pa), 1.0)
        return (
            math.isfinite(pressure)
            and self.P_min_pa - tolerance
            <= pressure
            <= self.P_max_pa + tolerance
        )

    def contains_temperature(self, temperature_K: float) -> bool:
        try:
            temperature = float(temperature_K)
        except (TypeError, ValueError):
            return False
        tolerance = 1.0e-10 * max(abs(self.T_min_K), abs(self.T_max_K), 1.0)
        return (
            math.isfinite(temperature)
            and self.T_min_K - tolerance
            <= temperature
            <= self.T_max_K + tolerance
        )


def coolprop_cas_key(identifier: Any) -> Optional[str]:
    """Return the compact form of a valid CAS number."""
    text = str(identifier or "").strip()
    match = re.fullmatch(r"(\d{2,7})-?(\d{2})-?(\d)", text)
    if match is None:
        return None
    body = match.group(1) + match.group(2)
    check_digit = sum(
        multiplier * int(digit)
        for multiplier, digit in enumerate(reversed(body), start=1)
    ) % 10
    if check_digit != int(match.group(3)):
        return None
    return body + match.group(3)


def coolprop_module():
    global _COOLPROP_MODULE
    if _COOLPROP_MODULE is _COOLPROP_UNAVAILABLE:
        return None
    if _COOLPROP_MODULE is not None:
        return _COOLPROP_MODULE
    try:
        import CoolProp.CoolProp as CP
    except Exception:
        _COOLPROP_MODULE = _COOLPROP_UNAVAILABLE
        return None
    _COOLPROP_MODULE = CP
    return CP


def coolprop_alias_map() -> Optional[dict[str, str]]:
    """Return the bundled exact-CAS-to-fluid map."""
    global _COOLPROP_ALIAS_MAP
    if _COOLPROP_ALIAS_MAP is _COOLPROP_UNAVAILABLE:
        return None
    if _COOLPROP_ALIAS_MAP is not None:
        return _COOLPROP_ALIAS_MAP
    try:
        payload = json.loads(COOLPROP_ALIAS_PATH.read_text())
        aliases = payload.get("aliases")
        if not isinstance(aliases, dict):
            raise ValueError("missing aliases")
        ambiguous_aliases = payload.get("ambiguous_aliases") or {}
        if not isinstance(ambiguous_aliases, dict):
            raise ValueError("invalid ambiguous aliases")
        _COOLPROP_ALIAS_MAP = {
            str(key): str(value)
            for key, value in aliases.items()
            if (
                key
                and value
                and key not in ambiguous_aliases
                and coolprop_cas_key(key) == key
            )
        }
    except Exception:
        _COOLPROP_ALIAS_MAP = _COOLPROP_UNAVAILABLE
        return None
    return _COOLPROP_ALIAS_MAP


def _hyphenated_cas(compact_cas: str) -> Optional[str]:
    """Return the conventional spelling of a validated compact CAS key."""
    if coolprop_cas_key(compact_cas) != compact_cas or len(compact_cas) < 5:
        return None
    return f"{compact_cas[:-3]}-{compact_cas[-3:-1]}-{compact_cas[-1]}"


def coolprop_cas_from_refrigerant_alias(identifier: Any) -> Optional[str]:
    """Resolve an unambiguous refrigerant designator to its exact CAS.

    Short refrigerant tags such as ``R125`` share a synonym namespace with
    ordinary chemical-name databases.  They must be interpreted in the
    refrigerant namespace before a generic name lookup; explicit CAS numbers
    and full chemical names remain unaffected.
    """
    global _COOLPROP_REFRIGERANT_CAS_ALIASES
    normalized = re.sub(r"[^a-z0-9]+", "", str(identifier or "").lower())
    if not re.fullmatch(r"r[a-z]?\d[a-z0-9]*", normalized):
        return None
    if _COOLPROP_REFRIGERANT_CAS_ALIASES is _COOLPROP_UNAVAILABLE:
        return None
    if _COOLPROP_REFRIGERANT_CAS_ALIASES is None:
        try:
            payload = json.loads(COOLPROP_ALIAS_PATH.read_text())
            aliases = payload.get("aliases")
            ambiguous_aliases = payload.get("ambiguous_aliases") or {}
            if not isinstance(aliases, dict) or not isinstance(ambiguous_aliases, dict):
                raise ValueError("invalid alias payload")

            cas_by_fluid: dict[str, set[str]] = {}
            for key, fluid in aliases.items():
                compact_cas = coolprop_cas_key(key)
                if compact_cas == key:
                    cas_by_fluid.setdefault(str(fluid), set()).add(compact_cas)

            resolved: dict[str, str] = {}
            for key, fluid in aliases.items():
                alias_key = re.sub(r"[^a-z0-9]+", "", str(key).lower())
                if (
                    not re.fullmatch(r"r[a-z]?\d[a-z0-9]*", alias_key)
                    or key in ambiguous_aliases
                ):
                    continue
                cas_values = cas_by_fluid.get(str(fluid), set())
                if len(cas_values) != 1:
                    continue
                conventional = _hyphenated_cas(next(iter(cas_values)))
                if conventional:
                    resolved[alias_key] = conventional
            _COOLPROP_REFRIGERANT_CAS_ALIASES = resolved
        except Exception:
            _COOLPROP_REFRIGERANT_CAS_ALIASES = _COOLPROP_UNAVAILABLE
            return None
    return _COOLPROP_REFRIGERANT_CAS_ALIASES.get(normalized)


def coolprop_fluid_from_candidates(
    candidates: Iterable[Any],
) -> Optional[str]:
    aliases = coolprop_alias_map()
    if aliases is None:
        return None
    for candidate in candidates:
        cas_key = coolprop_cas_key(candidate)
        if cas_key is None:
            continue
        fluid = aliases.get(cas_key)
        if fluid and coolprop_fluid_is_pure(fluid):
            return fluid
    return None


def coolprop_fluid_is_pure(fluid: str) -> bool:
    CP = coolprop_module()
    if CP is None:
        return False
    try:
        return str(CP.get_fluid_param_string(fluid, "pure")).lower() == "true"
    except Exception:
        return False


def coolprop_reference_from_candidates(
    candidates: Iterable[Any],
) -> Optional[CoolPropFluidReference]:
    fluid = coolprop_fluid_from_candidates(candidates)
    if not fluid:
        return None
    backend = "IF97" if fluid == "Water" else "HEOS"
    return CoolPropFluidReference(fluid=fluid, backend=backend)


def coolprop_reference_for(
    symbol: str,
    props: Optional[Mapping[str, Any]],
) -> Optional[CoolPropFluidReference]:
    """Return a CoolProp reference only from an already resolved exact CAS."""
    cas_candidates = []

    def append_cas(value: Any) -> None:
        cas_key = coolprop_cas_key(value)
        if cas_key is None:
            return
        if cas_key not in cas_candidates:
            cas_candidates.append(cas_key)

    if props is not None:
        for key in ("CAS", "cas"):
            append_cas(props.get(key))
    append_cas(symbol)
    return coolprop_reference_from_candidates(cas_candidates)


def coolprop_props_si(output: str, reference: CoolPropFluidReference) -> Optional[float]:
    CP = coolprop_module()
    if CP is None:
        return None
    try:
        value = float(CP.PropsSI(output, reference.qualified_name))
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def coolprop_saturation_temperature(
    reference: CoolPropFluidReference,
    pressure_pa: float,
) -> Optional[float]:
    CP = coolprop_module()
    if CP is None:
        return None
    try:
        value = float(CP.PropsSI("T", "P", float(pressure_pa), "Q", 0.0, reference.qualified_name))
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def coolprop_saturation_pressure(
    reference: CoolPropFluidReference,
    temperature_K: float,
) -> Optional[float]:
    CP = coolprop_module()
    if CP is None:
        return None
    try:
        value = float(CP.PropsSI("P", "T", float(temperature_K), "Q", 0.0, reference.qualified_name))
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def coolprop_melting_temperature(
    reference: CoolPropFluidReference,
    pressure_pa: float,
) -> Optional[float]:
    """Return an in-domain HEOS fusion-line temperature."""
    CP = coolprop_module()
    if CP is None:
        return None
    try:
        state = CP.AbstractState("HEOS", reference.fluid)
        if not state.has_melting_line():
            return None
        domain = CoolPropMeltingLineDomain(
            P_min_pa=float(state.melting_line(CP.iP_min, CP.iT, 0.0)),
            P_max_pa=float(state.melting_line(CP.iP_max, CP.iT, 0.0)),
            T_min_K=float(state.melting_line(CP.iT_min, CP.iP, 0.0)),
            T_max_K=float(state.melting_line(CP.iT_max, CP.iP, 0.0)),
        )
        if not domain.contains_pressure(pressure_pa):
            return None
        value = float(state.melting_line(CP.iT, CP.iP, float(pressure_pa)))
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def coolprop_melting_line_domain(
    reference: CoolPropFluidReference,
) -> Optional[CoolPropMeltingLineDomain]:
    """Return declared HEOS fusion-line limits without extrapolation."""
    CP = coolprop_module()
    if CP is None:
        return None
    try:
        state = CP.AbstractState("HEOS", reference.fluid)
        if not state.has_melting_line():
            return None
        domain = CoolPropMeltingLineDomain(
            P_min_pa=float(state.melting_line(CP.iP_min, CP.iT, 0.0)),
            P_max_pa=float(state.melting_line(CP.iP_max, CP.iT, 0.0)),
            T_min_K=float(state.melting_line(CP.iT_min, CP.iP, 0.0)),
            T_max_K=float(state.melting_line(CP.iT_max, CP.iP, 0.0)),
        )
    except Exception:
        return None
    values = (
        domain.P_min_pa,
        domain.P_max_pa,
        domain.T_min_K,
        domain.T_max_K,
    )
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in values)
        or domain.P_min_pa > domain.P_max_pa
        or domain.T_min_K > domain.T_max_K
    ):
        return None
    return domain


def coolprop_melting_pressure(
    reference: CoolPropFluidReference,
    temperature_K: float,
) -> Optional[float]:
    """Return in-domain HEOS fusion pressure at a temperature."""
    CP = coolprop_module()
    domain = coolprop_melting_line_domain(reference)
    if CP is None or domain is None or not domain.contains_temperature(temperature_K):
        return None
    try:
        state = CP.AbstractState("HEOS", reference.fluid)
        value = float(state.melting_line(CP.iP, CP.iT, float(temperature_K)))
    except Exception:
        return None
    if (
        not math.isfinite(value)
        or value <= 0.0
        or not domain.contains_pressure(value)
    ):
        return None
    return value
