"""CAS-keyed aqueous Henry constants and frozen equilibrium contexts."""

from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


HENRY_REFERENCE_TEMPERATURE_K = 298.15
HENRY_DEFAULT_TMIN_K = 278.15
HENRY_DEFAULT_TMAX_K = 323.15


@dataclass(frozen=True)
class HenryTemperatureCorrelation:
    """One resolved richer pure-water temperature correlation."""

    model: str
    A: float
    B: float
    C: float
    D: float
    E: float
    Tmin_K: float
    Tmax_K: float
    range_source: str
    rms_lnH: Optional[float]
    fit_quality: Optional[float]
    uncertainty_upper_percent: Optional[float]
    normalize_to_reference: bool
    raw_reference_value_difference_percent: Optional[float]
    raw_reference_slope_difference_K: Optional[float]
    raw_reference_slope_difference_percent: Optional[float]
    source: str
    doi: Optional[str]
    notes: Optional[str]


@dataclass(frozen=True)
class HenryConstantRecord:
    """One pure-water Henry correlation in Hcp form."""

    cas: str
    hcp_298: float
    B: Optional[float]
    quality_h: Optional[str]
    quality_b: Optional[str]
    quality_score_h: Optional[float]
    quality_score_b: Optional[float]
    h_more_temperature_dependence: bool = False
    b_more_temperature_dependence: bool = False
    vinf_298_cm3_per_mol: Optional[float] = None
    vinf_uncertainty_cm3_per_mol: Optional[float] = None
    vinf_source: Optional[str] = None
    vinf_doi: Optional[str] = None
    temperature_correlation: Optional[HenryTemperatureCorrelation] = None
    iupac: Optional[str] = None
    trivial: Optional[str] = None


@dataclass(frozen=True)
class HenryComponentData:
    """Resolved, possibly user-overridden Henry data for one component."""

    component: str
    cas: str
    hcp_298: float
    B: Optional[float]
    quality_h: Optional[str]
    quality_b: Optional[str]
    quality_score_h: Optional[float]
    quality_score_b: Optional[float]
    temperature_min_K: float
    temperature_max_K: float
    temperature_range_source: str
    vinf_cm3_per_mol: Optional[float]
    vinf_uncertainty_cm3_per_mol: Optional[float]
    vinf_estimated_relative_mae: Optional[float]
    vinf_source: Optional[str]
    vinf_method: Optional[str]
    vinf_quality: Optional[float]
    vinf_unavailable_reason: Optional[str]
    more_temperature_dependence_available: bool
    temperature_correlation: Optional[HenryTemperatureCorrelation]
    h_source: str
    b_source: Optional[str]


@dataclass(frozen=True)
class AqueousEquilibriumContext:
    """Frozen Henry component selection for one aqueous equilibrium solve."""

    water_component: str
    component_data: Mapping[str, HenryComponentData]
    solvent_molar_volume_ref_m3_per_kmol: float
    reference_temperature_K: float = HENRY_REFERENCE_TEMPERATURE_K
    pressure_warning_bar: float = 20.0

    @property
    def henry_components(self) -> frozenset[str]:
        return frozenset(self.component_data)

    def cache_key(self) -> tuple:
        return (
            self.water_component,
            tuple(sorted(
                (
                    comp,
                    data.hcp_298,
                    data.B,
                    data.temperature_min_K,
                    data.temperature_max_K,
                    data.vinf_cm3_per_mol,
                    data.vinf_method,
                    (
                        None
                        if data.temperature_correlation is None
                        else (
                            data.temperature_correlation.model,
                            data.temperature_correlation.A,
                            data.temperature_correlation.B,
                            data.temperature_correlation.C,
                            data.temperature_correlation.D,
                            data.temperature_correlation.E,
                            data.temperature_correlation.normalize_to_reference,
                        )
                    ),
                    data.h_source,
                    data.b_source,
                )
                for comp, data in self.component_data.items()
            )),
            self.solvent_molar_volume_ref_m3_per_kmol,
            self.reference_temperature_K,
        )


class HenryConstantDatabase:
    """Lazy in-memory view of the bundled SQLite Henry database."""

    def __init__(self, path: Optional[Path] = None):
        self.path = (
            Path(path)
            if path is not None
            else Path(__file__).resolve().parents[1] / "data" / "henry_constants.sqlite"
        )
        self._records: Optional[dict[str, HenryConstantRecord]] = None
        self._metadata: Optional[dict[str, str]] = None

    @staticmethod
    def _optional_float(value) -> Optional[float]:
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None

    def _load(self) -> dict[str, HenryConstantRecord]:
        if self._records is not None:
            return self._records
        if not self.path.exists():
            self._records = {}
            self._metadata = {}
            return self._records

        records = {}
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            self._metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM henry_metadata")
            }
            rows = connection.execute(
                """
                SELECT h.CAS, h.iupac, h.trivial, h.H, h.B,
                       h.quality_h, h.quality_b,
                       h.quality_score_h, h.quality_score_b,
                       h.H_more_temperature_dependence,
                       h.B_more_temperature_dependence,
                       v.Vinf_298_cm3_per_mol,
                       v.uncertainty_cm3_per_mol,
                       v.source AS vinf_source,
                       v.doi AS vinf_doi,
                       t.model AS temperature_model,
                       t.A AS temperature_A,
                       t.B AS temperature_B,
                       t.C AS temperature_C,
                       t.D AS temperature_D,
                       t.E AS temperature_E,
                       t.Tmin_K AS temperature_Tmin_K,
                       t.Tmax_K AS temperature_Tmax_K,
                       t.range_source AS temperature_range_source,
                       t.rms_lnH AS temperature_rms_lnH,
                       t.fit_quality AS temperature_fit_quality,
                       t.uncertainty_upper_percent AS temperature_uncertainty_upper_percent,
                       t.normalize_to_reference AS temperature_normalize_to_reference,
                       t.raw_reference_value_difference_percent,
                       t.raw_reference_slope_difference_K,
                       t.raw_reference_slope_difference_percent,
                       t.source AS temperature_source,
                       t.doi AS temperature_doi,
                       t.notes AS temperature_notes
                FROM henry_constants h
                LEFT JOIN henry_partial_molar_volumes v ON v.CAS = h.CAS
                LEFT JOIN henry_temperature_correlations t ON t.CAS = h.CAS
                WHERE h.H IS NOT NULL AND h.H > 0
                """
            )
            for row in rows:
                hcp = self._optional_float(row["H"])
                if hcp is None or hcp <= 0.0:
                    continue
                cas = str(row["CAS"] or "").strip()
                if not cas:
                    continue
                temperature_correlation = None
                if row["temperature_model"] is not None:
                    temperature_correlation = HenryTemperatureCorrelation(
                        model=str(row["temperature_model"]),
                        A=float(row["temperature_A"]),
                        B=float(row["temperature_B"]),
                        C=float(row["temperature_C"]),
                        D=float(row["temperature_D"]),
                        E=float(row["temperature_E"]),
                        Tmin_K=float(row["temperature_Tmin_K"]),
                        Tmax_K=float(row["temperature_Tmax_K"]),
                        range_source=str(row["temperature_range_source"]),
                        rms_lnH=self._optional_float(row["temperature_rms_lnH"]),
                        fit_quality=self._optional_float(row["temperature_fit_quality"]),
                        uncertainty_upper_percent=self._optional_float(
                            row["temperature_uncertainty_upper_percent"]
                        ),
                        normalize_to_reference=bool(
                            row["temperature_normalize_to_reference"]
                        ),
                        raw_reference_value_difference_percent=self._optional_float(
                            row["raw_reference_value_difference_percent"]
                        ),
                        raw_reference_slope_difference_K=self._optional_float(
                            row["raw_reference_slope_difference_K"]
                        ),
                        raw_reference_slope_difference_percent=self._optional_float(
                            row["raw_reference_slope_difference_percent"]
                        ),
                        source=str(row["temperature_source"]),
                        doi=str(row["temperature_doi"] or "").strip() or None,
                        notes=str(row["temperature_notes"] or "").strip() or None,
                    )
                records[cas] = HenryConstantRecord(
                    cas=cas,
                    hcp_298=hcp,
                    B=self._optional_float(row["B"]),
                    quality_h=str(row["quality_h"] or "").strip() or None,
                    quality_b=str(row["quality_b"] or "").strip() or None,
                    quality_score_h=self._optional_float(row["quality_score_h"]),
                    quality_score_b=self._optional_float(row["quality_score_b"]),
                    h_more_temperature_dependence=bool(row["H_more_temperature_dependence"]),
                    b_more_temperature_dependence=bool(row["B_more_temperature_dependence"]),
                    vinf_298_cm3_per_mol=self._optional_float(row["Vinf_298_cm3_per_mol"]),
                    vinf_uncertainty_cm3_per_mol=self._optional_float(
                        row["uncertainty_cm3_per_mol"]
                    ),
                    vinf_source=str(row["vinf_source"] or "").strip() or None,
                    vinf_doi=str(row["vinf_doi"] or "").strip() or None,
                    temperature_correlation=temperature_correlation,
                    iupac=str(row["iupac"] or "").strip() or None,
                    trivial=str(row["trivial"] or "").strip() or None,
                )
        self._records = records
        return records

    def get(self, cas: str) -> Optional[HenryConstantRecord]:
        key = str(cas or "").strip()
        return self._load().get(key) if key else None

    def __len__(self) -> int:
        return len(self._load())

    @property
    def metadata(self) -> dict[str, str]:
        self._load()
        return dict(self._metadata or {})


_HENRY_DATABASE: Optional[HenryConstantDatabase] = None


def get_henry_constant_database() -> HenryConstantDatabase:
    global _HENRY_DATABASE
    if _HENRY_DATABASE is None:
        _HENRY_DATABASE = HenryConstantDatabase()
    return _HENRY_DATABASE
