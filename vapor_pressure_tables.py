"""Tabulated vapor-pressure interpolation for common compounds."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

from scipy.interpolate import PchipInterpolator


DATA_PATH = Path(__file__).parent / "data" / "vapor_pressure_tables.json"


def _pressure_to_bar(value: float, units: str) -> float:
    units_key = units.strip().lower()
    if units_key == "bar":
        return value
    if units_key == "mpa":
        return value * 10.0
    if units_key == "kpa":
        return value / 100.0
    if units_key == "pa":
        return value / 100000.0
    raise ValueError(f"Unsupported vapor-pressure table units: {units}")


@dataclass(frozen=True)
class VaporPressureTable:
    """A bounded cubic spline through tabulated saturation pressures."""

    key: str
    name: str
    source: str
    temperatures: tuple[float, ...]
    pressures_bar: tuple[float, ...]

    def __post_init__(self):
        if len(self.temperatures) != len(self.pressures_bar):
            raise ValueError(f"Temperature/pressure length mismatch for {self.key}")
        if len(self.temperatures) < 4:
            raise ValueError(f"At least four points are needed for cubic interpolation: {self.key}")
        pairs = sorted(zip(self.temperatures, self.pressures_bar))
        if any(T2 <= T1 for (T1, _), (T2, _) in zip(pairs, pairs[1:])):
            raise ValueError(f"Temperatures must be strictly increasing for {self.key}")
        if any(P <= 0 for _, P in pairs):
            raise ValueError(f"Pressures must be positive for {self.key}")
        object.__setattr__(self, "temperatures", tuple(T for T, _ in pairs))
        object.__setattr__(self, "pressures_bar", tuple(P for _, P in pairs))

    @property
    def T_min(self) -> float:
        return self.temperatures[0]

    @property
    def T_max(self) -> float:
        return self.temperatures[-1]

    def covers_temperature(self, T: float, tolerance: float = 1e-9) -> bool:
        return self.T_min - tolerance <= T <= self.T_max + tolerance

    @cached_property
    def _ln_pressure_spline(self):
        return PchipInterpolator(self.temperatures, [math.log(P) for P in self.pressures_bar])

    def vapor_pressure(self, T: float) -> Optional[float]:
        """Return pressure in bar, or None outside the table range."""
        if not self.covers_temperature(T):
            return None
        # Vapor pressure is nearly exponential in T, so interpolate ln(P) to
        # preserve positivity without changing the table's exact points.
        try:
            return math.exp(float(self._ln_pressure_spline(T)))
        except (ValueError, OverflowError):
            return None


class VaporPressureTableLibrary:
    """Lookup tabulated vapor-pressure data by canonical identity aliases."""

    def __init__(self, path: Path = DATA_PATH):
        self.path = path
        self.tables: dict[str, VaporPressureTable] = {}
        self.aliases: dict[str, str] = {}
        self._load()

    @staticmethod
    def _normalize(identifier: str) -> str:
        return "".join(ch.lower() for ch in str(identifier) if ch.isalnum())

    @staticmethod
    def _looks_like_formula(identifier: str) -> bool:
        text = str(identifier).strip()
        if not text or not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", text):
            return False
        return any(ch.isdigit() for ch in text)

    def _add_alias(self, alias: object, key: str):
        if alias:
            self.aliases[self._normalize(str(alias))] = key

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, "r") as f:
            payload = json.load(f)
        for key, entry in payload.get("tables", {}).items():
            pressure_units = entry.get("P_units", "bar")
            table = VaporPressureTable(
                key=key,
                name=entry.get("name", key),
                source=entry.get("source", self.path.name),
                temperatures=tuple(float(T) for T in entry["temperatures"]),
                pressures_bar=tuple(
                    _pressure_to_bar(float(P), pressure_units)
                    for P in entry["pressures"]
                ),
            )
            self.tables[key] = table
            for alias in (
                key,
                entry.get("name"),
                entry.get("CAS"),
                entry.get("cas"),
            ):
                self._add_alias(alias, key)

    def get(self, identifier: str, T: Optional[float] = None) -> Optional[VaporPressureTable]:
        key = self.aliases.get(self._normalize(identifier))
        if key:
            table = self.tables.get(key)
            if table and (T is None or table.covers_temperature(T)):
                return table
        if self._looks_like_formula(identifier):
            return None

        aliases = [identifier]
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import get_compound_identity_resolver
            else:
                from compound_identity import get_compound_identity_resolver
            aliases = get_compound_identity_resolver().candidate_identifiers(identifier)
        except Exception:
            pass
        for alias in aliases:
            key = self.aliases.get(self._normalize(alias))
            if not key:
                continue
            table = self.tables.get(key)
            if table and (T is None or table.covers_temperature(T)):
                return table
        return None


_LIBRARY: Optional[VaporPressureTableLibrary] = None


def get_vapor_pressure_table_library() -> VaporPressureTableLibrary:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = VaporPressureTableLibrary()
    return _LIBRARY
