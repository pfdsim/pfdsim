"""Runtime lookup and evaluators for extracted Perry property correlations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from scipy.interpolate import PchipInterpolator
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .compound_identity import get_compound_identity_resolver
    from .pressure_standards import NORMAL_BOILING_PRESSURE_BAR
else:
    from compound_identity import get_compound_identity_resolver
    from pressure_standards import NORMAL_BOILING_PRESSURE_BAR


DATA_PATH = Path(__file__).parent / "data" / "perry_properties.json"
HEAT_OF_FUSION_DATA_PATH = Path(__file__).parent / "data" / "perry_heat_of_fusion.json"
TABLE_2_10_DATA_PATH = Path(__file__).parent / "data" / "perry_table_2_10_vapor_pressure.json"
THERMAL_CONDUCTIVITY_DATA_PATH = (
    Path(__file__).parent / "data" / "perry_thermal_conductivity.json"
)


@dataclass(frozen=True)
class PerryEvaluation:
    value: float
    units: str
    source: str
    method: str
    correlation: dict[str, Any]


class PerryPropertyLibrary:
    """Lookup helper for Perry pure-component property correlations."""

    def __init__(
        self,
        path: str | Path = DATA_PATH,
        heat_of_fusion_path: str | Path = HEAT_OF_FUSION_DATA_PATH,
        table_2_10_path: str | Path = TABLE_2_10_DATA_PATH,
        thermal_conductivity_path: str | Path = THERMAL_CONDUCTIVITY_DATA_PATH,
    ):
        self.path = Path(path)
        self.heat_of_fusion_path = Path(heat_of_fusion_path)
        self.table_2_10_path = Path(table_2_10_path)
        self.thermal_conductivity_path = Path(thermal_conductivity_path)
        self._loaded = False
        self._fusion_loaded = False
        self._table_2_10_loaded = False
        self._thermal_conductivity_loaded = False
        self.metadata: dict[str, Any] = {}
        self.chemicals: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, str] = {}
        self.heat_of_fusion_metadata: dict[str, Any] = {}
        self.heat_of_fusion_chemicals: dict[str, dict[str, Any]] = {}
        self.heat_of_fusion_aliases: dict[str, str] = {}
        self.table_2_10_metadata: dict[str, Any] = {}
        self.table_2_10_chemicals: dict[str, dict[str, Any]] = {}
        self.table_2_10_aliases: dict[str, str] = {}
        self._table_2_10_vapor_pressure_splines: dict[str, Any] = {}
        self.thermal_conductivity_metadata: dict[str, Any] = {}
        self.thermal_conductivity_chemicals: dict[str, dict[str, Any]] = {}
        self.saturated_liquids: dict[str, dict[str, Any]] = {}
        self.saturated_liquid_aliases: dict[str, str] = {}

    @staticmethod
    def _normalize(identifier: str) -> str:
        return "".join(ch.lower() for ch in str(identifier) if ch.isalnum())

    def _add_alias(self, alias: object, cas: str) -> None:
        if alias:
            self.aliases.setdefault(self._normalize(str(alias)), cas)

    def _add_fusion_alias(self, alias: object, cas: str) -> None:
        if alias:
            self.heat_of_fusion_aliases.setdefault(self._normalize(str(alias)), cas)

    def _add_table_2_10_alias(self, alias: object, cas: str) -> None:
        if alias:
            self.table_2_10_aliases.setdefault(self._normalize(str(alias)), cas)

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return

        payload = json.loads(self.path.read_text())
        self.metadata = payload.get("metadata", {})
        self.chemicals = payload.get("chemicals", {})

        formula_to_cas: dict[str, list[str]] = {}
        for cas, entry in self.chemicals.items():
            self._add_alias(cas, cas)
            self._add_alias(entry.get("name"), cas)
            for name in entry.get("names", []):
                self._add_alias(name, cas)
            for number in entry.get("compound_numbers", []):
                self._add_alias(f"perry:{number}", cas)
            for formula in entry.get("formulas", []):
                formula_to_cas.setdefault(self._normalize(formula), []).append(cas)

        # Formula aliases are only safe when unique; otherwise isomers would be
        # silently collapsed.
        for formula, cas_values in formula_to_cas.items():
            unique = sorted(set(cas_values))
            if len(unique) == 1:
                self.aliases.setdefault(formula, unique[0])

    def _load_heat_of_fusion(self) -> None:
        if self._fusion_loaded:
            return
        self._fusion_loaded = True
        if not self.heat_of_fusion_path.exists():
            return

        payload = json.loads(self.heat_of_fusion_path.read_text())
        self.heat_of_fusion_metadata = payload.get("metadata", {})
        self.heat_of_fusion_chemicals = payload.get("chemicals", {})
        for cas, entry in self.heat_of_fusion_chemicals.items():
            self._add_fusion_alias(cas, cas)
            self._add_fusion_alias(entry.get("name"), cas)
            for row in entry.get("heat_of_fusion", []):
                self._add_fusion_alias(row.get("table_name"), cas)
                self._add_fusion_alias(row.get("resolved_query"), cas)

    def _get_heat_of_fusion_entry(self, identifier: str) -> Optional[dict[str, Any]]:
        self._load_heat_of_fusion()
        cas = self.heat_of_fusion_aliases.get(self._normalize(identifier))
        if cas:
            return self.heat_of_fusion_chemicals.get(cas)

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import get_compound_identity_resolver
            else:
                from compound_identity import get_compound_identity_resolver
            candidates = get_compound_identity_resolver().candidate_identifiers(
                identifier,
                allow_formula=False,
            )
        except Exception:
            candidates = []
        for candidate in candidates:
            cas = self.heat_of_fusion_aliases.get(self._normalize(candidate))
            if cas:
                return self.heat_of_fusion_chemicals.get(cas)
        return None

    def _load_table_2_10(self) -> None:
        if self._table_2_10_loaded:
            return
        self._table_2_10_loaded = True
        if not self.table_2_10_path.exists():
            return

        payload = json.loads(self.table_2_10_path.read_text())
        self.table_2_10_metadata = payload.get("metadata", {})
        self.table_2_10_chemicals = payload.get("chemicals", {})
        for cas, entry in self.table_2_10_chemicals.items():
            self._add_table_2_10_alias(cas, cas)
            for alias in (
                entry.get("name"),
                entry.get("table_name"),
                entry.get("resolved_query"),
            ):
                self._add_table_2_10_alias(alias, cas)

    def _get_table_2_10_entry(self, identifier: str) -> Optional[dict[str, Any]]:
        self._load_table_2_10()
        cas = self.table_2_10_aliases.get(self._normalize(identifier))
        if cas:
            return self.table_2_10_chemicals.get(cas)

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import get_compound_identity_resolver
            else:
                from compound_identity import get_compound_identity_resolver
            candidates = get_compound_identity_resolver().candidate_identifiers(
                identifier,
                allow_formula=False,
            )
        except Exception:
            candidates = []
        for candidate in candidates:
            cas = self.table_2_10_aliases.get(self._normalize(candidate))
            if cas:
                return self.table_2_10_chemicals.get(cas)
        return None

    def _load_thermal_conductivity(self) -> None:
        if self._thermal_conductivity_loaded:
            return
        self._thermal_conductivity_loaded = True
        if not self.thermal_conductivity_path.exists():
            return

        payload = json.loads(self.thermal_conductivity_path.read_text())
        self.thermal_conductivity_metadata = payload.get("metadata", {})
        self.thermal_conductivity_chemicals = payload.get("chemicals", {})
        self.saturated_liquids = payload.get("saturated_liquids", {})
        identities = get_compound_identity_resolver()
        for name, entry in self.saturated_liquids.items():
            for alias in (name, *entry.get("aliases", [])):
                self.saturated_liquid_aliases.setdefault(self._normalize(alias), name)
                cas = identities.resolve_cas(alias, allow_formula=False)
                if cas:
                    self.saturated_liquid_aliases.setdefault(
                        self._normalize(cas), name
                    )

    def get(self, identifier: str, expand_identity: bool = True) -> Optional[dict[str, Any]]:
        self._load()
        cas = self.aliases.get(self._normalize(identifier))
        if cas:
            return self.chemicals.get(cas)

        if not expand_identity:
            return None

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import get_compound_identity_resolver
            else:
                from compound_identity import get_compound_identity_resolver
            candidates = get_compound_identity_resolver().candidate_identifiers(
                identifier,
                allow_formula=False,
            )
        except Exception:
            candidates = []
        for candidate in candidates:
            cas = self.aliases.get(self._normalize(candidate))
            if cas:
                return self.chemicals.get(cas)
        return None

    @staticmethod
    def _covers(row: dict[str, Any], T: float) -> bool:
        return row.get("T_min_K", -math.inf) <= T <= row.get("T_max_K", math.inf)

    @staticmethod
    def _range_width(row: dict[str, Any]) -> float:
        return float(row.get("T_max_K", math.inf)) - float(row.get("T_min_K", -math.inf))

    def _select_in_range(
        self,
        entry: dict[str, Any],
        key: str,
        T: float,
        supported: Optional[set[int]] = None,
    ) -> Optional[dict[str, Any]]:
        rows = []
        for row in entry.get(key, []):
            equation_id = row.get("equation_id")
            if supported is not None and equation_id not in supported:
                continue
            if self._covers(row, T):
                rows.append(row)
        if not rows:
            return None
        return min(
            rows,
            key=lambda row: (
                self._range_width(row),
                abs((row.get("T_min_K", T) + row.get("T_max_K", T)) / 2.0 - T),
            ),
        )

    @staticmethod
    def _polynomial(coefficients: list[float], T: float) -> float:
        return sum(coef * T ** power for power, coef in enumerate(coefficients))

    @staticmethod
    def _safe_power(base: float, exponent: float) -> float:
        if base < 0.0 and abs(exponent - round(exponent)) > 1e-12:
            raise ValueError("fractional power of negative base")
        return base ** exponent

    @classmethod
    def _eval_liquid_density_mol_per_dm3(cls, row: dict[str, Any], T: float) -> Optional[float]:
        equation_id = row.get("equation_id")
        coeffs = row.get("coefficients", [])
        try:
            if equation_id == 100:
                return cls._polynomial(coeffs, T)
            if equation_id == 105 and len(coeffs) >= 4:
                C1, C2, C3, C4 = coeffs[:4]
                tau = 1.0 - T / C3
                return C1 / (C2 ** (1.0 + cls._safe_power(tau, C4)))
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        return None

    @classmethod
    def _eval_liquid_heat_capacity_J_per_kmol_K(cls, row: dict[str, Any], T: float) -> Optional[float]:
        equation_id = row.get("equation_id")
        coeffs = row.get("coefficients", [])
        try:
            if equation_id == 100:
                return cls._polynomial(coeffs, T)
        except (ValueError, OverflowError):
            return None
        return None

    @classmethod
    def _eval_ideal_gas_polynomial_J_per_kmol_K(cls, row: dict[str, Any], T: float) -> Optional[float]:
        try:
            return cls._polynomial(row.get("coefficients", []), T)
        except (ValueError, OverflowError):
            return None

    @staticmethod
    def _eval_ideal_gas_hyperbolic_J_per_kmol_K(row: dict[str, Any], T: float) -> Optional[float]:
        coeffs = row.get("coefficients", [])
        if len(coeffs) < 5:
            return None
        C1, C2, C3, C4, C5 = coeffs[:5]
        try:
            sinh_arg = C3 / T
            cosh_arg = C5 / T
            return (
                C1
                + C2 * (sinh_arg / math.sinh(sinh_arg)) ** 2
                + C4 * (cosh_arg / math.cosh(cosh_arg)) ** 2
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            return None

    @staticmethod
    def _eval_vapor_viscosity_Pa_s(row: dict[str, Any], T: float) -> Optional[float]:
        coeffs = row.get("coefficients", [])
        if len(coeffs) < 2:
            return None
        C1 = coeffs[0]
        C2 = coeffs[1]
        C3 = coeffs[2] if len(coeffs) > 2 else 0.0
        C4 = coeffs[3] if len(coeffs) > 3 else 0.0
        try:
            return C1 * T ** C2 / (1.0 + C3 / T + C4 / T**2)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None

    @classmethod
    def _eval_liquid_viscosity_Pa_s(cls, row: dict[str, Any], T: float) -> Optional[float]:
        equation_id = row.get("equation_id")
        coeffs = row.get("coefficients", [])
        try:
            if equation_id == 100 and len(coeffs) == 1:
                return coeffs[0]
            if equation_id == 101 and len(coeffs) >= 2:
                C1 = coeffs[0]
                C2 = coeffs[1]
                C3 = coeffs[2] if len(coeffs) > 2 else 0.0
                C4 = coeffs[3] if len(coeffs) > 3 else 0.0
                C5 = coeffs[4] if len(coeffs) > 4 else 1.0
                return math.exp(C1 + C2 / T + C3 * math.log(T) + C4 * T**C5)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        return None

    @staticmethod
    def _source(table: str) -> str:
        return f"Perry 9th Table {table}"

    @staticmethod
    def _vapor_pressure_ln_pa(
        coefficients: tuple[float, float, float, float, float],
        T: float,
    ) -> float:
        C1, C2, C3, C4, C5 = coefficients
        return C1 + C2 / T + C3 * math.log(T) + C4 * T**C5

    @classmethod
    def vapor_pressure_coefficients(
        cls,
        row: dict[str, Any],
    ) -> Optional[tuple[float, float, float, float, float]]:
        """Normalize one extracted Table 2-8 coefficient row."""
        try:
            raw = tuple(float(value) for value in row.get("coefficients", ()))
        except (TypeError, ValueError):
            return None
        if any(not math.isfinite(value) for value in raw):
            return None
        if len(raw) == 3:
            candidates = ((raw[0], raw[1], raw[2], 0.0, 1.0),)
        elif len(raw) == 5:
            candidates = (raw,)
        else:
            return None

        references = []
        for temperature_key, pressure_key in (
            ("T_min_K", "P_at_T_min_Pa"),
            ("T_max_K", "P_at_T_max_Pa"),
        ):
            try:
                temperature = float(row.get(temperature_key))
                pressure = float(row.get(pressure_key))
            except (TypeError, ValueError):
                continue
            if (
                math.isfinite(temperature)
                and temperature > 0.0
                and math.isfinite(pressure)
                and pressure > 0.0
            ):
                references.append((temperature, math.log(pressure)))

        def score(coefficients):
            if not references:
                return 0.0
            try:
                residuals = [
                    abs(cls._vapor_pressure_ln_pa(coefficients, T) - ln_pressure)
                    for T, ln_pressure in references
                ]
            except (ValueError, ZeroDivisionError, OverflowError):
                return math.inf
            return max(residuals)

        selected = min(candidates, key=score)
        return selected if math.isfinite(score(selected)) else None

    def vapor_pressure_correlations(
        self,
        identifier: str,
    ) -> tuple[dict[str, Any], ...]:
        """Return every extracted Perry Table 2-8 row for one compound."""
        entry = self.get(identifier)
        if not entry:
            return ()
        return tuple(dict(row) for row in entry.get("vapor_pressure", ()))

    def vapor_pressure_bar(self, identifier: str, T: float) -> Optional[PerryEvaluation]:
        entry = self.get(identifier)
        if not entry:
            return None
        row = self._select_in_range(entry, "vapor_pressure", T)
        if not row:
            return None
        coeffs = self.vapor_pressure_coefficients(row)
        if coeffs is None:
            return None
        try:
            P_pa = math.exp(self._vapor_pressure_ln_pa(coeffs, T))
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        if P_pa <= 0:
            return None
        return PerryEvaluation(
            value=P_pa / 100000.0,
            units="bar",
            source=self._source(row["source_table"]),
            method="perry_vapor_pressure",
            correlation=row,
        )

    def normal_boiling_point_K(self, identifier: str) -> Optional[PerryEvaluation]:
        entry = self.get(identifier)
        if not entry:
            return None

        row = entry.get("normal_points")
        if row and row.get("Tb_K") is not None:
            return PerryEvaluation(
                value=row["Tb_K"],
                units="K",
                source=self._source(row["source_table"]),
                method="perry_normal_boiling_point",
                correlation=row,
            )

        target_bar = NORMAL_BOILING_PRESSURE_BAR
        for vp_row in entry.get("vapor_pressure", []):
            t_low = vp_row.get("T_min_K")
            t_high = vp_row.get("T_max_K")
            if t_low is None or t_high is None:
                continue
            p_low = self.vapor_pressure_bar(entry.get("cas") or identifier, t_low)
            p_high = self.vapor_pressure_bar(entry.get("cas") or identifier, t_high)
            if not p_low or not p_high:
                continue
            if not (min(p_low.value, p_high.value) <= target_bar <= max(p_low.value, p_high.value)):
                continue

            lo, hi = float(t_low), float(t_high)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                p_mid = self.vapor_pressure_bar(entry.get("cas") or identifier, mid)
                if not p_mid:
                    break
                if (p_low.value <= target_bar) == (p_mid.value <= target_bar):
                    lo = mid
                    p_low = p_mid
                else:
                    hi = mid
            row = dict(vp_row)
            row["target_pressure_bar"] = target_bar
            return PerryEvaluation(
                value=0.5 * (lo + hi),
                units="K",
                source=self._source(vp_row["source_table"]),
                method="perry_vapor_pressure_normal_boiling_point",
                correlation=row,
            )
        return None

    def table_2_10_normal_boiling_point_K(self, identifier: str) -> Optional[PerryEvaluation]:
        entry = self._get_table_2_10_entry(identifier)
        if not entry or entry.get("Tb_K") is None:
            return None
        row = {
            "source_table": entry.get("source_table", "2-10"),
            "cas": entry.get("cas"),
            "name": entry.get("name"),
            "table_name": entry.get("table_name"),
            "Tb_K": entry["Tb_K"],
            "Tb_C_at_760_mmHg": entry.get("Tb_C_at_760_mmHg"),
        }
        return PerryEvaluation(
            value=entry["Tb_K"],
            units="K",
            source=self._source(row["source_table"]),
            method="perry_table_2_10_normal_boiling_point",
            correlation=row,
        )

    def table_2_10_normal_melting_point_K(self, identifier: str) -> Optional[PerryEvaluation]:
        entry = self._get_table_2_10_entry(identifier)
        if not entry or entry.get("Tm_K") is None:
            return None
        row = {
            "source_table": entry.get("source_table", "2-10"),
            "cas": entry.get("cas"),
            "name": entry.get("name"),
            "table_name": entry.get("table_name"),
            "Tm_K": entry["Tm_K"],
            "Tm_C": entry.get("Tm_C"),
        }
        return PerryEvaluation(
            value=entry["Tm_K"],
            units="K",
            source=self._source(row["source_table"]),
            method="perry_table_2_10_normal_melting_point",
            correlation=row,
        )

    def table_2_10_vapor_pressure_curve(
        self,
        identifier: str,
    ) -> Optional[tuple[PchipInterpolator, dict[str, Any]]]:
        """Return the validated log-pressure PCHIP and its source row."""
        entry = self._get_table_2_10_entry(identifier)
        if not entry:
            return None

        cas = entry.get("cas") or identifier
        spline = self._table_2_10_vapor_pressure_splines.get(cas)
        row = None
        if spline is None:
            points = entry.get("vapor_pressure", [])
            pairs = sorted(
                (
                    float(point["T_K"]),
                    float(point["P_bar"]),
                )
                for point in points
                if point.get("T_K") is not None and point.get("P_bar") is not None
            )
            if len(pairs) < 4:
                return None
            temperatures = tuple(T_value for T_value, _ in pairs)
            pressures = tuple(P_value for _, P_value in pairs)
            if any(T2 <= T1 for T1, T2 in zip(temperatures, temperatures[1:])):
                return None
            if any(P_value <= 0.0 for P_value in pressures):
                return None
            row = {
                "source_table": entry.get("source_table", "2-10"),
                "cas": entry.get("cas"),
                "name": entry.get("name"),
                "table_name": entry.get("table_name"),
                "T_min_K": temperatures[0],
                "T_max_K": temperatures[-1],
                "temperatures_K": temperatures,
                "pressures_bar": pressures,
                "points": points,
            }
            spline = (
                PchipInterpolator(temperatures, [math.log(P_value) for P_value in pressures]),
                row,
            )
            self._table_2_10_vapor_pressure_splines[cas] = spline
        return spline

    def table_2_10_vapor_pressure_bar(self, identifier: str, T: float) -> Optional[PerryEvaluation]:
        spline = self.table_2_10_vapor_pressure_curve(identifier)
        if spline is None:
            return None

        interpolator, row = spline
        if not (row["T_min_K"] <= T <= row["T_max_K"]):
            return None
        try:
            pressure = math.exp(float(interpolator(T)))
        except (ValueError, OverflowError):
            return None
        if pressure <= 0.0:
            return None
        return PerryEvaluation(
            value=pressure,
            units="bar",
            source=self._source(row["source_table"]),
            method="perry_table_2_10_vapor_pressure",
            correlation=row,
        )

    def normal_melting_point_K(self, identifier: str) -> Optional[PerryEvaluation]:
        entry = self.get(identifier)
        if entry:
            row = entry.get("normal_points")
            if row and row.get("Tm_K") is not None:
                return PerryEvaluation(
                    value=row["Tm_K"],
                    units="K",
                    source=self._source(row["source_table"]),
                    method="perry_normal_melting_point",
                    correlation=row,
                )

        fusion_entry = self._get_heat_of_fusion_entry(identifier)
        if fusion_entry:
            for row in fusion_entry.get("heat_of_fusion", []):
                if row.get("Tm_K") is not None:
                    return PerryEvaluation(
                        value=row["Tm_K"],
                        units="K",
                        source=self._source(row["source_table"]),
                        method="perry_heat_of_fusion_melting_point",
                        correlation=row,
                    )

        table_2_10_tm = self.table_2_10_normal_melting_point_K(identifier)
        if table_2_10_tm:
            return table_2_10_tm
        return None

    @staticmethod
    def _eval_heat_of_vaporization_J_per_kmol(row: dict[str, Any], T: float, Tc: float) -> Optional[float]:
        coeffs = row.get("coefficients", [])
        if len(coeffs) < 4 or Tc <= 0 or T >= Tc:
            return None
        try:
            C1, C2, C3, C4 = coeffs[:4]
            Tr = T / Tc
            exponent = C2 + C3 * Tr + C4 * Tr * Tr
            value = C1 * (1.0 - Tr) ** exponent
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        return value if value > 0 else None

    def heat_of_vaporization_kJ_per_mol(
        self,
        identifier: str,
        T: Optional[float] = None,
    ) -> Optional[PerryEvaluation]:
        entry = self.get(identifier)
        if not entry:
            return None
        if T is None:
            tb = self.normal_boiling_point_K(identifier)
            if not tb:
                return None
            T = tb.value
        critical = self.critical_properties(identifier)
        Tc = critical["Tc"].value if critical and "Tc" in critical else None
        if Tc is None:
            return None
        row = self._select_in_range(entry, "heat_of_vaporization", T)
        if not row:
            return None
        value = self._eval_heat_of_vaporization_J_per_kmol(row, T, Tc)
        if value is None:
            return None
        return PerryEvaluation(
            value=value / 1.0e6,
            units="kJ/mol",
            source=self._source(row["source_table"]),
            method="perry_heat_of_vaporization",
            correlation=row,
        )

    def heat_of_vaporization_value_from_entry(
        self,
        entry: dict[str, Any],
        T: float,
    ) -> Optional[tuple[float, dict[str, Any], str]]:
        """Evaluate a pre-resolved Perry heat-of-vaporization entry."""
        critical = entry.get("critical_constants") or {}
        Tc = critical.get("Tc_K")
        if Tc is None:
            return None
        row = self._select_in_range(entry, "heat_of_vaporization", T)
        if not row:
            return None
        value = self._eval_heat_of_vaporization_J_per_kmol(row, T, Tc)
        if value is None or value <= 0:
            return None
        return value / 1.0e6, row, "perry_heat_of_vaporization"

    def heat_of_fusion_records(
        self,
        identifier: str,
    ) -> tuple[PerryEvaluation, ...]:
        """Return every extracted Perry Table 2-68 transition record."""
        entry = self._get_heat_of_fusion_entry(identifier)
        if not entry:
            return ()
        return tuple(
            PerryEvaluation(
                value=row["Hfus_kJ_per_mol"],
                units="kJ/mol",
                source=self._source(row["source_table"]),
                method="perry_heat_of_fusion",
                correlation=row,
            )
            for row in entry.get("heat_of_fusion", [])
            if row.get("Hfus_kJ_per_mol") is not None
        )

    def heat_of_fusion_kJ_per_mol(
        self,
        identifier: str,
        transition_temperature_K: Optional[float] = None,
    ) -> Optional[PerryEvaluation]:
        records = self.heat_of_fusion_records(identifier)
        if not records:
            return None
        if transition_temperature_K is not None:
            paired = [
                record for record in records
                if record.correlation.get("Tm_K") is not None
            ]
            if paired:
                return min(
                    paired,
                    key=lambda record: abs(
                        float(record.correlation["Tm_K"])
                        - float(transition_temperature_K)
                    ),
                )
        return next(
            (
                record for record in records
                if record.correlation.get("Tm_K") is not None
            ),
            records[0],
        )

    def critical_properties(self, identifier: str) -> Optional[dict[str, PerryEvaluation]]:
        entry = self.get(identifier)
        if not entry or not entry.get("critical_constants"):
            return None
        row = entry["critical_constants"]
        source = self._source(row["source_table"])
        # Vc (and its derived Zc) may have been patched from a higher-quality
        # source where Perry's tabulated value was unreliable; report the patch
        # source honestly rather than the Perry table.
        patch = row.get("_vc_patch")
        vc_source = patch["source"] if patch else source
        vc_method = "perry_critical_patched" if patch else "perry_critical"
        return {
            "Tc": PerryEvaluation(row["Tc_K"], "K", source, "perry_critical", row),
            "Pc": PerryEvaluation(row["Pc_MPa"] * 10.0, "bar", source, "perry_critical", row),
            "Vc": PerryEvaluation(row["Vc_m3_per_kmol"] * 1000.0, "cm^3/mol", vc_source, vc_method, row),
            "omega": PerryEvaluation(row["omega"], "dimensionless", source, "perry_critical", row),
            "Zc": PerryEvaluation(row["Zc"], "dimensionless", vc_source, vc_method, row),
        }

    def liquid_molar_density_mol_per_dm3(self, identifier: str, T: float) -> Optional[PerryEvaluation]:
        entry = self.get(identifier)
        if not entry:
            return None
        row = self._select_in_range(entry, "liquid_density", T, supported={100, 105})
        if not row:
            return None
        value = self._eval_liquid_density_mol_per_dm3(row, T)
        if value is None or value <= 0:
            return None
        return PerryEvaluation(
            value=value,
            units="mol/dm^3",
            source=self._source(row["source_table"]),
            method=f"perry_density_eq{row.get('equation_id')}",
            correlation=row,
        )

    def liquid_molar_volume_m3_per_kmol(self, identifier: str, T: float) -> Optional[PerryEvaluation]:
        density = self.liquid_molar_density_mol_per_dm3(identifier, T)
        if density is None or density.value <= 0:
            return None
        # 1 mol/dm^3 equals 1 kmol/m^3, so 1/rho is both dm^3/mol and m^3/kmol.
        return PerryEvaluation(
            value=1.0 / density.value,
            units="m^3/kmol",
            source=density.source,
            method=density.method.replace("density", "molar_volume"),
            correlation=density.correlation,
        )

    def heat_capacity_J_per_mol_K(self, identifier: str, T: float, phase: str) -> Optional[PerryEvaluation]:
        phase_key = phase.strip().lower().replace("-", "_")
        entry = self.get(identifier)
        if not entry:
            return None

        evaluated = self.heat_capacity_value_from_entry(entry, T, phase_key)
        if evaluated is None:
            return None
        value, row, method = evaluated

        return PerryEvaluation(
            value=value,
            units="J/(mol*K)",
            source=self._source(row["source_table"]),
            method=method,
            correlation=row,
        )

    def heat_capacity_value_from_entry(
        self,
        entry: dict[str, Any],
        T: float,
        phase: str,
    ) -> Optional[tuple[float, dict[str, Any], str]]:
        """Evaluate a pre-resolved Perry heat-capacity entry without identity lookup."""
        phase_key = phase.strip().lower().replace("-", "_")

        if phase_key in {"liquid", "l"}:
            row = self._select_in_range(entry, "liquid_heat_capacity", T, supported={100})
            if not row:
                return None
            value = self._eval_liquid_heat_capacity_J_per_kmol_K(row, T)
            method = f"perry_liquid_cp_eq{row.get('equation_id')}"
        elif phase_key in {"gas", "vapor", "vapour", "ideal_gas", "ideal"}:
            row = self._select_in_range(entry, "ideal_gas_heat_capacity_polynomial", T)
            if row:
                value = self._eval_ideal_gas_polynomial_J_per_kmol_K(row, T)
                method = "perry_ideal_gas_cp_polynomial"
            else:
                row = self._select_in_range(entry, "ideal_gas_heat_capacity_hyperbolic", T)
                if not row:
                    return None
                value = self._eval_ideal_gas_hyperbolic_J_per_kmol_K(row, T)
                method = "perry_ideal_gas_cp_hyperbolic"
        else:
            return None

        if value is None or value <= 0:
            return None
        return value / 1000.0, row, method

    def viscosity_Pa_s(self, identifier: str, T: float, phase: str) -> Optional[PerryEvaluation]:
        entry = self.get(identifier)
        if not entry:
            return None
        return self.viscosity_Pa_s_from_entry(entry, T, phase)

    def viscosity_Pa_s_from_entry(
        self, entry: dict[str, Any], T: float, phase: str,
    ) -> Optional[PerryEvaluation]:
        """Evaluate a prepared entry using the same selection as identifier lookup."""
        phase_key = phase.strip().lower().replace("-", "_")

        if phase_key in {"gas", "vapor", "vapour", "ideal_gas", "ideal"}:
            row = self._select_in_range(entry, "vapor_viscosity", T)
            if not row:
                return None
            value = self._eval_vapor_viscosity_Pa_s(row, T)
            method = "perry_vapor_viscosity"
        elif phase_key in {"liquid", "l"}:
            row = self._select_in_range(entry, "liquid_viscosity", T, supported={100, 101})
            if not row:
                return None
            value = self._eval_liquid_viscosity_Pa_s(row, T)
            method = f"perry_liquid_viscosity_eq{row.get('equation_id')}"
        else:
            return None

        if value is None or value <= 0:
            return None
        return PerryEvaluation(
            value=value,
            units="Pa*s",
            source=self._source(row["source_table"]),
            method=method,
            correlation=row,
        )

    def thermal_conductivity_correlation(
        self,
        identifier: str,
        T: float,
        phase: str,
    ) -> Optional[dict[str, Any]]:
        """Return an in-range Perry conductivity row without evaluating it."""
        self._load_thermal_conductivity()
        entry = self.get(identifier, expand_identity=False)
        if not entry:
            return None
        thermal_entry = self.thermal_conductivity_chemicals.get(entry.get("cas"))
        if not thermal_entry:
            return None
        if phase == "vapor":
            key = "vapor_thermal_conductivity"
        elif phase == "liquid":
            key = "liquid_thermal_conductivity"
        else:
            return None
        return self._select_in_range(thermal_entry, key, T, supported={100, 102})

    def saturated_liquid_thermal_conductivity_W_per_m_K(
        self,
        identifier: str,
        T: float,
    ) -> Optional[PerryEvaluation]:
        """Interpolate one contiguous Table 2-146 conductivity run."""
        self._load_thermal_conductivity()
        name = self.saturated_liquid_aliases.get(self._normalize(identifier))
        if name is None:
            cas = get_compound_identity_resolver().resolve_cas(
                identifier, allow_formula=False
            )
            if cas:
                name = self.saturated_liquid_aliases.get(self._normalize(cas))
        if name is None:
            return None
        entry = self.saturated_liquids[name]
        for run in entry.get("runs", []):
            samples = [(float(T_i), float(value)) for T_i, value in run]
            if not samples or T < samples[0][0] or T > samples[-1][0]:
                continue
            for sample_T, value in samples:
                if math.isclose(T, sample_T, abs_tol=1.0e-9):
                    return PerryEvaluation(
                        value=value,
                        units="W/(m*K)",
                        source=self._source("2-146"),
                        method="perry_saturated_liquid_thermal_conductivity_tabulated",
                        correlation={"source_table": "2-146", "temperature_K": sample_T},
                    )
            for (lower_T, lower_value), (upper_T, upper_value) in zip(
                samples,
                samples[1:],
            ):
                if lower_T < T < upper_T:
                    fraction = (T - lower_T) / (upper_T - lower_T)
                    return PerryEvaluation(
                        value=lower_value + fraction * (upper_value - lower_value),
                        units="W/(m*K)",
                        source=self._source("2-146"),
                        method=(
                            "perry_saturated_liquid_thermal_conductivity_"
                            "linear_interpolation"
                        ),
                        correlation={
                            "source_table": "2-146",
                            "T_min_K": lower_T,
                            "T_max_K": upper_T,
                        },
                    )
        return None

    def formation_properties(self, identifier: str) -> Optional[dict[str, PerryEvaluation]]:
        entry = self.get(identifier)
        if not entry or not entry.get("formation_properties"):
            return None
        row = entry["formation_properties"]
        source = self._source(row["source_table"])

        def optional_eval(key: str, value: Optional[float], units: str) -> Optional[PerryEvaluation]:
            if value is None:
                return None
            return PerryEvaluation(value, units, source, "perry_formation_298K", row)

        result = {
            "Hf": optional_eval("Hf", row.get("Hf_ideal_gas_J_per_kmol", 0.0) / 1.0e6, "kJ/mol"),
            "Gf": optional_eval("Gf", row.get("Gf_ideal_gas_J_per_kmol", 0.0) / 1.0e6, "kJ/mol"),
            "S": optional_eval("S", row.get("S_ideal_gas_J_per_kmol_K", 0.0) / 1000.0, "J/(mol*K)"),
            "Hcomb": optional_eval(
                "Hcomb",
                row.get("net_Hcomb_J_per_kmol") / 1.0e6 if row.get("net_Hcomb_J_per_kmol") is not None else None,
                "kJ/mol",
            ),
        }
        return {key: value for key, value in result.items() if value is not None}


_library: Optional[PerryPropertyLibrary] = None


def get_perry_property_library() -> PerryPropertyLibrary:
    global _library
    if _library is None:
        _library = PerryPropertyLibrary()
    return _library
