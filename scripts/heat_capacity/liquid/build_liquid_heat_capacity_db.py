#!/usr/bin/env python3
"""Compile ordinary-liquid heat-capacity sources into canonical CAS records.

This is an offline compiler; the runtime resolver consumes only its SQLite
output and does not import the compiler or its source dependencies. Every
selected non-native compound is represented by one linearly mapped Chebyshev
polynomial over one contiguous temperature range. Native Zabransky
quasipolynomials are preserved exactly.

Candidate selection is quality-first. Every physically valid source is fitted,
all independently agreeing sources receive a corroboration bonus, the candidate
with the highest resulting quality minus fitting penalty is selected, and exact
ties use this deterministic preference:

    Perry 9e > CoolProp saturated-state CPMOLAR > Zabransky p spline
    > Zabransky p quasipolynomial > Zabransky C spline
    > Zabransky C quasipolynomial > NIST WebBook Shomate

Corroboration can therefore change which source wins. Zabransky ``sat`` records
are excluded: they represent the derivative of saturated-liquid enthalpy along
the saturation curve, not ordinary constant-pressure liquid heat capacity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from numpy.polynomial import chebyshev as ncheb


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.heat_capacity.gas import build_ideal_gas_heat_capacity_db as common


OUTPUT = ROOT / "data" / "liquid_heat_capacity.sqlite"
SCHEMA_VERSION = 1
MODEL_NAME = "linear_chebyshev_liquid_cp_v1"
NATIVE_ZABRANSKY_MODEL = "native_zabransky_quasipolynomial_v1"
FIT_DEGREES = (8, 12, 16)
TRAINING_POINTS_PER_RANGE = 257
VALIDATION_POINTS_PER_RANGE = 1001
ENTROPY_PRIMITIVE_BASE_EXTRA_DEGREE = 16
COOLPROP_MAX_REDUCED_TEMPERATURE = 0.95
ZABRANSKY_AVERAGED_QUALITY_FACTOR = 0.985
MAX_FINAL_QUALITY = 0.995
CORROBORATION_BONUS_PER_LINEAGE = 0.005
MAX_CORROBORATION_BONUS = 0.015
SOURCE_COMPARISON_MEDIAN_LIMIT_PERCENT = 0.5
SOURCE_COMPARISON_P95_LIMIT_PERCENT = 1.5
STRONG_CONFLICT_PERCENT = 10.0

ZABRANSKY_CLASS_QUALITY = {
    "I": 0.995,
    "II": 0.990,
    "III": 0.980,
    "IV": 0.960,
    "V": 0.900,
    "VI": 0.700,
}

# Rank is used only when corroborated, fit-adjusted qualities are exactly equal.
SOURCE_SPECS = {
    "perry_9e": {
        "rank": 1,
        "label": "Perry 9th edition",
        "lineage": "perry_9e",
        "quality": 0.980,
    },
    "coolprop_liquid": {
        "rank": 2,
        "label": "CoolProp HEOS liquid CPMOLAR at Q=0",
        "lineage": "coolprop_heos",
        "quality": 0.980,
    },
    "zabransky_p_spline": {
        "rank": 3,
        "label": "Zabransky ordinary isobaric Cp spline",
        "lineage": "zabransky_1996",
        "quality": None,
    },
    "zabransky_p_quasi": {
        "rank": 4,
        "label": "Zabransky ordinary isobaric Cp quasipolynomial",
        "lineage": "zabransky_1996",
        "quality": None,
    },
    "zabransky_C_spline": {
        "rank": 5,
        "label": "Zabransky averaged ordinary Cp spline",
        "lineage": "zabransky_1996",
        "quality": None,
    },
    "zabransky_C_quasi": {
        "rank": 6,
        "label": "Zabransky averaged ordinary Cp quasipolynomial",
        "lineage": "zabransky_1996",
        "quality": None,
    },
    "webbook_shomate": {
        "rank": 7,
        "label": "NIST WebBook liquid Shomate",
        "lineage": "nist_webbook",
        "quality": 0.975,
    },
}


SourceSegment = common.SourceSegment
@dataclass(frozen=True)
class LinearChebyshevFit:
    Tmin: float
    Tmax: float
    degree: int
    center: float
    half_width: float
    coefficients: tuple[float, ...]
    fit_mape_percent: float
    fit_max_error_percent: float
    fit_h_max_error_percent: float
    fit_s_max_error_percent: float
    validation_points: int
    h_coefficients: tuple[float, ...]
    s_coefficients: tuple[float, ...]

    @property
    def model(self) -> str:
        return MODEL_NAME


@dataclass(frozen=True)
class NativeZabranskyFit:
    Tmin: float
    Tmax: float
    critical_temperature: float
    coefficients: tuple[float, ...]
    degree: None = None
    fit_mape_percent: float = 0.0
    fit_max_error_percent: float = 0.0
    fit_h_max_error_percent: float = 0.0
    fit_s_max_error_percent: float = 0.0
    validation_points: int = 0

    @property
    def model(self) -> str:
        return NATIVE_ZABRANSKY_MODEL


CompiledFit = LinearChebyshevFit | NativeZabranskyFit


@dataclass
class SourceCandidate:
    cas: str
    source: str
    source_key: str
    source_version: str
    name: str
    formula: str
    segments: tuple[SourceSegment, ...]
    source_hash: str
    details: dict[str, Any]
    base_quality_value: float
    status: str = "available"
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def rank(self) -> int:
        return int(SOURCE_SPECS[self.source]["rank"])

    @property
    def label(self) -> str:
        return str(SOURCE_SPECS[self.source]["label"])

    @property
    def lineage(self) -> str:
        return str(SOURCE_SPECS[self.source]["lineage"])

    @property
    def base_quality(self) -> float:
        return float(self.base_quality_value)

    @property
    def Tmin(self) -> float:
        return min(segment.Tmin for segment in self.segments)

    @property
    def Tmax(self) -> float:
        return max(segment.Tmax for segment in self.segments)

    @property
    def fingerprint(self) -> str:
        payload = {
            "source_hash": self.source_hash,
            "source": self.source,
            "source_key": self.source_key,
            "source_version": self.source_version,
            "details": self.details,
        }
        return hashlib.sha256(common.canonical_json(payload).encode("utf-8")).hexdigest()

    def values(self, temperatures: Sequence[float] | np.ndarray) -> np.ndarray:
        values_T = np.asarray(temperatures, dtype=float)
        result = np.full(values_T.shape, np.nan, dtype=float)
        for segment in self.segments:
            mask = (
                np.isnan(result)
                & (values_T >= segment.Tmin - 1e-10)
                & (values_T <= segment.Tmax + 1e-10)
            )
            if not np.any(mask):
                continue
            calculated = np.asarray(segment.evaluate_vector(values_T[mask]), dtype=float)
            result[mask] = calculated
        return result


@dataclass(frozen=True)
class CandidateAssessment:
    candidate: SourceCandidate
    fit: CompiledFit
    fit_quality_penalty: float
    corroboration_bonus: float
    selection_quality: float
    corroborated_lineages: tuple[str, ...]


@dataclass(frozen=True)
class Crosscheck:
    cas: str
    first_source: str
    second_source: str
    first_lineage: str
    second_lineage: str
    point_count: int
    Tmin: float
    Tmax: float
    median_error_percent: float
    p95_error_percent: float
    max_error_percent: float
    close_agreement: bool
    independent: bool


@dataclass(frozen=True)
class SelectedRecord:
    assessment: CandidateAssessment
    crosschecks: tuple[Crosscheck, ...]

    @property
    def candidate(self) -> SourceCandidate:
        return self.assessment.candidate

    @property
    def fit(self) -> CompiledFit:
        return self.assessment.fit

    @property
    def quality(self) -> float:
        return self.assessment.selection_quality

    @property
    def corroboration_bonus(self) -> float:
        return self.assessment.corroboration_bonus


@dataclass(frozen=True)
class Quarantine:
    cas: str
    source: str
    reason: str
    details: dict[str, Any]


def _fixed_quality(source: str) -> float:
    value = SOURCE_SPECS[source]["quality"]
    if value is None:
        raise ValueError(f"source {source} requires row-specific quality")
    return float(value)


def _zabransky_quality(uncertainties: Sequence[str], averaged: bool) -> float:
    if not uncertainties:
        raise ValueError("missing Zabransky uncertainty class")
    qualities = []
    for uncertainty in uncertainties:
        try:
            qualities.append(ZABRANSKY_CLASS_QUALITY[str(uncertainty).strip()])
        except KeyError as error:
            raise ValueError(f"unknown Zabransky uncertainty class {uncertainty!r}") from error
    result = min(qualities)
    if averaged:
        result *= ZABRANSKY_AVERAGED_QUALITY_FACTOR
    return result


def collect_perry_candidates() -> list[SourceCandidate]:
    from perry_properties import DATA_PATH, get_perry_property_library

    library = get_perry_property_library()
    library._load()
    source_hash = common.sha256_path(DATA_PATH)
    source_version = str(library.metadata.get("source_book") or "Perry 9th edition")
    candidates = []
    for cas, entry in sorted(library.chemicals.items()):
        if not common.valid_cas(cas):
            continue
        segments = []
        rows = []
        critical = entry.get("critical_constants") or {}
        critical_temperature = common.finite_float(critical.get("Tc_K"))
        for row in entry.get("liquid_heat_capacity", ()):
            if int(row.get("equation_id") or 0) != 100:
                continue
            Tmin = float(row["T_min_K"])
            Tmax_native = float(row["T_max_K"])
            Tmax = (
                min(Tmax_native, COOLPROP_MAX_REDUCED_TEMPERATURE * critical_temperature)
                if critical_temperature is not None else Tmax_native
            )
            if not Tmin < Tmax:
                continue

            def evaluate(Ts: np.ndarray, row: dict[str, Any] = row) -> np.ndarray:
                values = []
                for T in Ts:
                    value = library._eval_liquid_heat_capacity_J_per_kmol_K(row, float(T))
                    values.append(float(value) / 1000.0 if value is not None else math.nan)
                return np.asarray(values, dtype=float)

            segments.append(SourceSegment(
                Tmin,
                Tmax,
                evaluate,
                {
                    "kind": "perry_liquid_cp_eq100",
                    "row": row,
                    "native_Tmax_K": Tmax_native,
                    "critical_temperature_K": critical_temperature,
                },
            ))
            rows.append(row)
        if not segments:
            continue
        name, formula = common.identity_for_cas(
            cas,
            str(entry.get("name") or ""),
            str(entry.get("formula") or ((entry.get("formulas") or [""])[0])),
        )
        candidates.append(SourceCandidate(
            cas=cas,
            source="perry_9e",
            source_key=cas,
            source_version=source_version,
            name=name,
            formula=formula,
            segments=tuple(sorted(segments, key=lambda item: (item.Tmin, item.Tmax))),
            source_hash=source_hash,
            details={
                "rows": rows,
                "critical_temperature_K": critical_temperature,
                "maximum_reduced_temperature": (
                    COOLPROP_MAX_REDUCED_TEMPERATURE
                    if critical_temperature is not None else None
                ),
            },
            base_quality_value=_fixed_quality("perry_9e"),
        ))
    return candidates


def collect_coolprop_candidates() -> tuple[list[SourceCandidate], list[Quarantine]]:
    try:
        import CoolProp
        import CoolProp.CoolProp as CP
    except ImportError:
        return [], [Quarantine("", "coolprop_liquid", "CoolProp unavailable", {})]

    version = str(getattr(CoolProp, "__version__", "unknown"))
    candidates = []
    quarantines = []
    seen_cas = set()
    for fluid in sorted(CP.get_global_param_string("FluidsList").split(",")):
        try:
            if CP.get_fluid_param_string(fluid, "pure").strip().lower() != "true":
                continue
            cas = CP.get_fluid_param_string(fluid, "CAS").strip()
            if not common.valid_cas(cas) or cas in seen_cas:
                continue
            state = CP.AbstractState("HEOS", fluid)
            source_Tmin = max(float(state.Tmin()), float(state.Ttriple()))
            critical_temperature = float(state.T_critical())
            source_Tmax = min(float(state.Tmax()), COOLPROP_MAX_REDUCED_TEMPERATURE * critical_temperature)
            Tmin = source_Tmin + max(1e-6, abs(source_Tmin) * 1e-9)
            Tmax = source_Tmax
            if not (math.isfinite(Tmin) and math.isfinite(Tmax) and Tmin < Tmax):
                raise ValueError("invalid CoolProp liquid temperature range")

            def evaluate(Ts: np.ndarray, fluid: str = fluid) -> np.ndarray:
                return np.asarray(
                    CP.PropsSI(
                        "CPMOLAR", "T", np.asarray(Ts, dtype=float), "Q", 0.0, fluid,
                    ),
                    dtype=float,
                )

            probe = evaluate(np.linspace(Tmin, Tmax, 257))
            if np.any(~np.isfinite(probe)) or np.any(probe <= 0.0):
                raise ValueError("nonfinite or nonpositive liquid CPMOLAR band")
            name, formula = common.identity_for_cas(
                cas,
                fluid,
                CP.get_fluid_param_string(fluid, "formula").strip(),
            )
            source_hash = hashlib.sha256(
                (
                    f"CoolProp|{version}|HEOS|{fluid}|{cas}|CPMOLAR|Q=0|"
                    f"Trmax={COOLPROP_MAX_REDUCED_TEMPERATURE}"
                ).encode("utf-8")
            ).hexdigest()
            candidates.append(SourceCandidate(
                cas=cas,
                source="coolprop_liquid",
                source_key=fluid,
                source_version=f"CoolProp {version}",
                name=name,
                formula=formula,
                segments=(SourceSegment(
                    Tmin,
                    Tmax,
                    evaluate,
                    {
                        "kind": "CoolProp_CPMOLAR_Q0",
                        "fluid": fluid,
                        "backend": "HEOS",
                    },
                ),),
                source_hash=source_hash,
                details={
                    "fluid": fluid,
                    "backend": "HEOS",
                    "property": "CPMOLAR",
                    "state_convention": "ordinary isobaric Cp evaluated at saturated-liquid state Q=0",
                    "source_Tmin_K": source_Tmin,
                    "source_Tmax_K": float(state.Tmax()),
                    "critical_temperature_K": critical_temperature,
                    "maximum_reduced_temperature": COOLPROP_MAX_REDUCED_TEMPERATURE,
                },
                base_quality_value=_fixed_quality("coolprop_liquid"),
            ))
            seen_cas.add(cas)
        except Exception as error:
            cas = ""
            try:
                candidate_cas = CP.get_fluid_param_string(fluid, "CAS").strip()
                if common.valid_cas(candidate_cas):
                    cas = candidate_cas
            except Exception:
                pass
            quarantines.append(Quarantine(
                cas,
                "coolprop_liquid",
                "CoolProp liquid source rejected",
                {"fluid": fluid, "error": f"{type(error).__name__}: {error}"},
            ))
    return candidates, quarantines


def _parse_zabransky_rows(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            row = {key: str(value or "").strip() for key, value in raw.items()}
            row["Tmin"] = float(row["Tmin"])
            row["Tmax"] = float(row["Tmax"])
            row["spline_coefficients"] = tuple(
                float(row[key] or 0.0)
                for key in ("A1-spline", "A2-spline", "A3-spline", "A4-spline")
            )
            row["quasi_coefficients"] = tuple(
                float(row[key] or 0.0)
                for key in (
                    "A1-quasi", "A2-quasi", "A3-quasi",
                    "A4-quasi", "A5-quasi", "A6-quasi",
                )
            )
            row["Tc"] = common.finite_float(row["Tc"])
            result.append(row)
    return result


def collect_zabransky_candidates() -> tuple[list[SourceCandidate], list[Quarantine], dict[str, int]]:
    import chemicals
    import chemicals.heat_capacity as heat_capacity

    _ = heat_capacity.Cp_data_Poling
    path = Path(heat_capacity.folder) / "Zabransky.tsv"
    source_hash = common.sha256_path(path)
    version = f"chemicals {chemicals.__version__}; Zabransky 1996"
    grouped_splines: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_quasi: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    quarantines = []
    excluded_saturation_rows = 0
    raw_rows = _parse_zabransky_rows(path)
    critical_by_cas = {
        row["CASRN"]: row["Tc"]
        for row in raw_rows
        if common.valid_cas(row["CASRN"]) and row["Tc"] is not None and row["Tc"] > 0.0
    }
    for row in raw_rows:
        data_type = row["Data Type"]
        if data_type == "sat":
            excluded_saturation_rows += 1
            continue
        if data_type not in {"p", "C"}:
            quarantines.append(Quarantine(
                row["CASRN"], "zabransky", "unknown Zabransky data type", {"row": row},
            ))
            continue
        cas = row["CASRN"]
        if not common.valid_cas(cas):
            quarantines.append(Quarantine(
                cas, "zabransky", "invalid or missing Zabransky CAS", {"row": row},
            ))
            continue
        try:
            _zabransky_quality([row["Uncertainty"]], data_type == "C")
        except ValueError as error:
            quarantines.append(Quarantine(
                cas, "zabransky", str(error), {"row": row},
            ))
            continue
        if any(row["spline_coefficients"]):
            grouped_splines[(data_type, cas)].append(row)
        if any(row["quasi_coefficients"]):
            grouped_quasi[(data_type, cas)].append(row)

    candidates = []
    for (data_type, cas), rows in sorted(grouped_splines.items()):
        rows = sorted(rows, key=lambda item: (item["Tmin"], item["Tmax"]))
        source = f"zabransky_{data_type}_spline"
        segments = []
        for row in rows:
            coefficients = tuple(row["spline_coefficients"])
            critical_temperature = critical_by_cas.get(cas)
            Tmax = (
                min(row["Tmax"], COOLPROP_MAX_REDUCED_TEMPERATURE * critical_temperature)
                if critical_temperature is not None else row["Tmax"]
            )
            if not row["Tmin"] < Tmax:
                continue

            def evaluate(
                Ts: np.ndarray,
                coefficients: tuple[float, ...] = coefficients,
            ) -> np.ndarray:
                return np.asarray([
                    heat_capacity.Zabransky_cubic(float(T), *coefficients)
                    for T in Ts
                ], dtype=float)

            segments.append(SourceSegment(
                row["Tmin"], Tmax, evaluate,
                {
                    "kind": "Zabransky_cubic",
                    "coefficients": coefficients,
                    "uncertainty_class": row["Uncertainty"],
                    "native_Tmax_K": row["Tmax"],
                    "critical_temperature_K": critical_temperature,
                },
            ))
        if not segments:
            continue
        name, formula = common.identity_for_cas(cas, rows[0]["Name"], "")
        uncertainties = [row["Uncertainty"] for row in rows]
        candidates.append(SourceCandidate(
            cas=cas,
            source=source,
            source_key=f"{cas}|{data_type}|spline",
            source_version=version,
            name=name,
            formula=formula,
            segments=tuple(segments),
            source_hash=source_hash,
            details={
                "data_type": data_type,
                "data_semantics": (
                    "ordinary constant-pressure heat capacity"
                    if data_type == "p"
                    else "averaged ordinary heat capacity from less rigorous experiments"
                ),
                "equation_form": "piecewise cubic spline",
                "uncertainty_classes": uncertainties,
                "quality_uses_worst_segment_class": True,
                "averaged_quality_factor": (
                    ZABRANSKY_AVERAGED_QUALITY_FACTOR if data_type == "C" else 1.0
                ),
                "critical_temperature_K": critical_by_cas.get(cas),
                "maximum_reduced_temperature": (
                    COOLPROP_MAX_REDUCED_TEMPERATURE if cas in critical_by_cas else None
                ),
                "segments": [
                    {
                        "Tmin": row["Tmin"],
                        "Tmax": row["Tmax"],
                        "coefficients": row["spline_coefficients"],
                        "uncertainty_class": row["Uncertainty"],
                    }
                    for row in rows
                ],
            },
            base_quality_value=_zabransky_quality(uncertainties, data_type == "C"),
        ))

    for (data_type, cas), rows in sorted(grouped_quasi.items()):
        source = f"zabransky_{data_type}_quasi"
        if len(rows) != 1:
            quarantines.append(Quarantine(
                cas,
                source,
                "multiple Zabransky quasipolynomial rows",
                {"row_count": len(rows)},
            ))
            continue
        row = rows[0]
        coefficients = tuple(row["quasi_coefficients"])
        Tc = row["Tc"]
        if Tc is None or Tc <= row["Tmin"]:
            quarantines.append(Quarantine(
                cas,
                source,
                "invalid Zabransky quasipolynomial critical temperature",
                {"Tc": Tc, "Tmin": row["Tmin"], "Tmax": row["Tmax"]},
            ))
            continue
        Tmax = min(row["Tmax"], COOLPROP_MAX_REDUCED_TEMPERATURE * Tc)
        if not row["Tmin"] < Tmax:
            quarantines.append(Quarantine(
                cas,
                source,
                "empty Zabransky quasipolynomial range after 0.95 Tc cap",
                {"Tc": Tc, "Tmin": row["Tmin"], "Tmax": row["Tmax"]},
            ))
            continue

        def evaluate(
            Ts: np.ndarray,
            coefficients: tuple[float, ...] = coefficients,
            Tc: float = Tc,
        ) -> np.ndarray:
            return np.asarray([
                heat_capacity.Zabransky_quasi_polynomial(float(T), Tc, *coefficients)
                for T in Ts
            ], dtype=float)

        name, formula = common.identity_for_cas(cas, row["Name"], "")
        candidates.append(SourceCandidate(
            cas=cas,
            source=source,
            source_key=f"{cas}|{data_type}|quasi",
            source_version=version,
            name=name,
            formula=formula,
            segments=(SourceSegment(
                row["Tmin"], Tmax, evaluate,
                {
                    "kind": "Zabransky_quasipolynomial",
                    "coefficients": coefficients,
                    "critical_temperature_K": Tc,
                    "uncertainty_class": row["Uncertainty"],
                },
            ),),
            source_hash=source_hash,
            details={
                "data_type": data_type,
                "data_semantics": (
                    "ordinary constant-pressure heat capacity"
                    if data_type == "p"
                    else "averaged ordinary heat capacity from less rigorous experiments"
                ),
                "equation_form": "critical-temperature quasipolynomial",
                "uncertainty_classes": [row["Uncertainty"]],
                "averaged_quality_factor": (
                    ZABRANSKY_AVERAGED_QUALITY_FACTOR if data_type == "C" else 1.0
                ),
                "coefficients": coefficients,
                "critical_temperature_K": Tc,
                "native_Tmax_K": row["Tmax"],
                "maximum_reduced_temperature": COOLPROP_MAX_REDUCED_TEMPERATURE,
            },
            base_quality_value=_zabransky_quality(
                [row["Uncertainty"]], data_type == "C",
            ),
        ))
    report = {
        "ordinary_candidate_count": len(candidates),
        "excluded_saturation_row_count": excluded_saturation_rows,
    }
    return candidates, quarantines, report


def collect_webbook_candidates() -> tuple[list[SourceCandidate], list[Quarantine]]:
    import chemicals
    import chemicals.heat_capacity as heat_capacity

    _ = heat_capacity.Cp_data_Poling
    folder = Path(heat_capacity.folder)
    version = str(chemicals.__version__)
    webbook_path = folder / "webbook_shomate_coefficients.json"
    candidates = []
    quarantines = []

    for cas, model in sorted(heat_capacity.WebBook_Shomate_liquids.items()):
        if not common.valid_cas(cas):
            continue
        segments = common.model_segments(model, "Shomate")
        name, formula = common.identity_for_cas(cas)
        candidates.append(SourceCandidate(
            cas=cas,
            source="webbook_shomate",
            source_key=cas,
            source_version=f"chemicals {version}; NIST WebBook snapshot",
            name=name,
            formula=formula,
            segments=segments,
            source_hash=common.sha256_path(webbook_path),
            details={
                "segments": [
                    segment.details | {"Tmin": segment.Tmin, "Tmax": segment.Tmax}
                    for segment in segments
                ],
            },
            base_quality_value=_fixed_quality("webbook_shomate"),
        ))

    return candidates, quarantines


def _segment_handoff_details(candidate: SourceCandidate) -> dict[str, Any]:
    segments = sorted(candidate.segments, key=lambda item: (item.Tmin, item.Tmax))
    gaps = []
    jumps = []
    for first, second in zip(segments[:-1], segments[1:]):
        if second.Tmin > first.Tmax:
            gaps.append(second.Tmin - first.Tmax)
            continue
        boundary = max(second.Tmin, first.Tmin)
        if boundary > min(first.Tmax, second.Tmax):
            continue
        first_value = float(first.evaluate_vector(np.asarray([boundary]))[0])
        second_value = float(second.evaluate_vector(np.asarray([boundary]))[0])
        if first_value > 0.0 and second_value > 0.0:
            jumps.append(abs(first_value / second_value - 1.0) * 100.0)
    return {
        "maximum_gap_K": max(gaps, default=0.0),
        "maximum_handoff_jump_percent": max(jumps, default=0.0),
    }


def validate_source(candidate: SourceCandidate) -> tuple[bool, str, dict[str, Any]]:
    temperatures = common.source_grid(candidate, candidate.Tmin, 401)
    values = candidate.values(temperatures)
    if len(values) == 0:
        return False, "empty source range", {}
    if np.any(~np.isfinite(values)):
        return False, "nonfinite heat capacity in source range", {}
    if np.any(values <= 0.0):
        index = int(np.argmin(values))
        return False, "nonpositive heat capacity in source range", {
            "T_K": float(temperatures[index]),
            "Cp_J_mol_K": float(values[index]),
        }
    handoff = _segment_handoff_details(candidate)
    if handoff["maximum_handoff_jump_percent"] > STRONG_CONFLICT_PERCENT:
        return False, "source has a discontinuous segment handoff", handoff
    return True, "", {
        "minimum_Cp_J_mol_K": float(np.min(values)),
        "maximum_Cp_J_mol_K": float(np.max(values)),
        "validation_points": len(values),
        **handoff,
    }


def _linear_source_grid(candidate: SourceCandidate, points_per_range: int) -> np.ndarray:
    grids = []
    for segment in candidate.segments:
        if segment.Tmin < segment.Tmax:
            grids.append(np.linspace(segment.Tmin, segment.Tmax, points_per_range))
    return np.unique(np.concatenate(grids)) if grids else np.asarray([], dtype=float)


def evaluate_linear_fit(
    fit: LinearChebyshevFit,
    temperatures: Sequence[float] | np.ndarray,
) -> np.ndarray:
    values_T = np.asarray(temperatures, dtype=float)
    x = (values_T - fit.center) / fit.half_width
    return ncheb.chebval(x, fit.coefficients)


def integrate_h_linear(fit: LinearChebyshevFit, T1: float, T2: float) -> float:
    x1 = (T1 - fit.center) / fit.half_width
    x2 = (T2 - fit.center) / fit.half_width
    return float(
        ncheb.chebval(x2, fit.h_coefficients)
        - ncheb.chebval(x1, fit.h_coefficients)
    )


def integrate_s_linear(fit: LinearChebyshevFit, T1: float, T2: float) -> float:
    x1 = (T1 - fit.center) / fit.half_width
    x2 = (T2 - fit.center) / fit.half_width
    return float(
        ncheb.chebval(x2, fit.s_coefficients)
        - ncheb.chebval(x1, fit.s_coefficients)
    )


def _validate_linear_integrals(
    candidate: SourceCandidate,
    fit: LinearChebyshevFit,
) -> tuple[float, float]:
    h_errors = []
    s_errors = []
    for segment in candidate.segments:
        dense_T = np.linspace(segment.Tmin, segment.Tmax, 1601)
        source = segment.evaluate_vector(dense_T)
        indexes = (0, 400, 800, 1200, 1600)
        for start_index in range(len(indexes) - 1):
            for end_index in range(start_index + 1, len(indexes)):
                left = indexes[start_index]
                right = indexes[end_index]
                source_H = float(np.trapezoid(source[left:right + 1], dense_T[left:right + 1]))
                source_S = float(np.trapezoid(
                    source[left:right + 1] / dense_T[left:right + 1],
                    dense_T[left:right + 1],
                ))
                fit_H = integrate_h_linear(fit, float(dense_T[left]), float(dense_T[right]))
                fit_S = integrate_s_linear(fit, float(dense_T[left]), float(dense_T[right]))
                if abs(source_H) > 1e-12:
                    h_errors.append(abs(fit_H / source_H - 1.0) * 100.0)
                if abs(source_S) > 1e-12:
                    s_errors.append(abs(fit_S / source_S - 1.0) * 100.0)
    return max(h_errors, default=0.0), max(s_errors, default=0.0)


def _fit_linear_coefficients(
    candidate: SourceCandidate,
    degree: int,
) -> Optional[LinearChebyshevFit]:
    training_T = _linear_source_grid(candidate, TRAINING_POINTS_PER_RANGE)
    validation_T = _linear_source_grid(candidate, VALIDATION_POINTS_PER_RANGE)
    if len(training_T) < degree + 1 or len(validation_T) < degree + 1:
        return None
    training_values = candidate.values(training_T)
    validation_values = candidate.values(validation_T)
    if (
        np.any(~np.isfinite(training_values)) or np.any(training_values <= 0.0)
        or np.any(~np.isfinite(validation_values)) or np.any(validation_values <= 0.0)
    ):
        return None
    center = 0.5 * (candidate.Tmin + candidate.Tmax)
    half_width = 0.5 * (candidate.Tmax - candidate.Tmin)
    if not (half_width > 0.0 and math.isfinite(center) and math.isfinite(half_width)):
        return None
    x_training = (training_T - center) / half_width
    vandermonde = ncheb.chebvander(x_training, degree)
    weighted = vandermonde / training_values[:, None]
    column_scale = np.linalg.norm(weighted, axis=0)
    if np.any(column_scale == 0.0):
        return None
    scaled_coefficients, *_ = np.linalg.lstsq(
        weighted / column_scale,
        np.ones(len(training_T)),
        rcond=None,
    )
    coefficients = scaled_coefficients / column_scale
    x_validation = (validation_T - center) / half_width
    predicted = ncheb.chebval(x_validation, coefficients)
    if np.any(~np.isfinite(predicted)) or np.any(predicted <= 0.0):
        return None
    continuous_T = np.linspace(candidate.Tmin, candidate.Tmax, VALIDATION_POINTS_PER_RANGE)
    continuous_cp = ncheb.chebval((continuous_T - center) / half_width, coefficients)
    if np.any(~np.isfinite(continuous_cp)) or np.any(continuous_cp <= 0.0):
        return None
    errors = np.abs(predicted / validation_values - 1.0) * 100.0

    h_coefficients = ncheb.chebint(coefficients) * half_width
    span_ratio = candidate.Tmax / candidate.Tmin
    span_extra_degree = 4 * max(0, math.ceil(math.log2(span_ratio)))
    entropy_degree = degree + ENTROPY_PRIMITIVE_BASE_EXTRA_DEGREE + span_extra_degree
    entropy_nodes = np.cos(
        np.pi * (np.arange(entropy_degree + 1) + 0.5) / (entropy_degree + 1)
    )
    entropy_temperatures = center + half_width * entropy_nodes
    entropy_values = ncheb.chebval(entropy_nodes, coefficients) / entropy_temperatures
    entropy_integrand_coefficients = ncheb.chebfit(
        entropy_nodes, entropy_values, entropy_degree,
    )
    s_coefficients = ncheb.chebint(entropy_integrand_coefficients) * half_width

    provisional = LinearChebyshevFit(
        Tmin=candidate.Tmin,
        Tmax=candidate.Tmax,
        degree=degree,
        center=center,
        half_width=half_width,
        coefficients=tuple(float(value) for value in coefficients),
        fit_mape_percent=float(np.mean(errors)),
        fit_max_error_percent=float(np.max(errors)),
        fit_h_max_error_percent=math.inf,
        fit_s_max_error_percent=math.inf,
        validation_points=len(validation_T),
        h_coefficients=tuple(float(value) for value in h_coefficients),
        s_coefficients=tuple(float(value) for value in s_coefficients),
    )
    h_error, s_error = _validate_linear_integrals(candidate, provisional)
    return LinearChebyshevFit(
        **{
            **provisional.__dict__,
            "fit_h_max_error_percent": h_error,
            "fit_s_max_error_percent": s_error,
        }
    )


def canonicalize(candidate: SourceCandidate) -> Optional[LinearChebyshevFit]:
    """Fit the complete native liquid range without ever raising Tmin."""
    best_fit = None
    for degree in FIT_DEGREES:
        fit = _fit_linear_coefficients(candidate, degree)
        if fit is None:
            continue
        if best_fit is None or fit.fit_max_error_percent < best_fit.fit_max_error_percent:
            best_fit = fit
        if fit.fit_max_error_percent <= common.FIT_MAX_RELATIVE_ERROR_PERCENT + 1e-10:
            return fit
    # Preserve the best complete-range curve even above the target. Its fit
    # diagnostics lower selection/final quality rather than sacrificing the
    # low-temperature liquid range.
    return best_fit


def compile_candidate(candidate: SourceCandidate) -> Optional[CompiledFit]:
    if candidate.source in {"zabransky_p_quasi", "zabransky_C_quasi"}:
        coefficients = tuple(float(value) for value in candidate.details["coefficients"])
        critical_temperature = float(candidate.details["critical_temperature_K"])
        return NativeZabranskyFit(
            Tmin=candidate.Tmin,
            Tmax=candidate.Tmax,
            critical_temperature=critical_temperature,
            coefficients=coefficients,
            validation_points=401,
        )
    return canonicalize(candidate)


def compiled_model(fit: CompiledFit) -> str:
    return fit.model if isinstance(fit, NativeZabranskyFit) else MODEL_NAME


def compiled_payload(fit: CompiledFit) -> dict[str, Any]:
    if isinstance(fit, NativeZabranskyFit):
        return {
            "model": NATIVE_ZABRANSKY_MODEL,
            "Tmin_K": fit.Tmin,
            "Tmax_K": fit.Tmax,
            "critical_temperature_K": fit.critical_temperature,
            "coefficients": fit.coefficients,
        }
    return {
        "model": MODEL_NAME,
        "Tmin_K": fit.Tmin,
        "Tmax_K": fit.Tmax,
        "degree": fit.degree,
        "map_center_K": fit.center,
        "map_half_width_K": fit.half_width,
        "cp_coefficients": fit.coefficients,
        "h_chebyshev_coefficients": fit.h_coefficients,
        "s_chebyshev_coefficients": fit.s_coefficients,
    }


def integrate_h_zabransky_native(fit: NativeZabranskyFit, T1: float, T2: float) -> float:
    if T1 == T2:
        return 0.0
    if T2 < T1:
        return -integrate_h_zabransky_native(fit, T2, T1)
    a1, a2, a3, a4, a5, a6 = fit.coefficients
    Tc = fit.critical_temperature
    delta = T2 - T1
    log_critical_ratio = math.log1p(-delta / (Tc - T1))
    log_reduced_gap_1 = math.log1p(-T1 / Tc)
    polynomial = common.polynomial_difference(
        (
            0.0,
            a3 - a1,
            a4 / (2.0 * Tc),
            a5 / (3.0 * Tc * Tc),
            a6 / (4.0 * Tc * Tc * Tc),
        ),
        T1,
        T2,
    )
    logarithmic = (
        a1 * delta * log_reduced_gap_1
        + (a1 * T2 - Tc * (a1 + a2)) * log_critical_ratio
    )
    return common.R * (polynomial + logarithmic)


def integrate_s_zabransky_native(fit: NativeZabranskyFit, T1: float, T2: float) -> float:
    if T1 == T2:
        return 0.0
    if T2 < T1:
        return -integrate_s_zabransky_native(fit, T2, T1)
    from scipy.special import spence

    a1, a2, a3, a4, a5, a6 = fit.coefficients
    Tc = fit.critical_temperature
    delta = T2 - T1
    log_temperature_ratio = math.log1p(delta / T1)
    log_critical_ratio = math.log1p(-delta / (Tc - T1))
    # scipy.special.spence(1-x) is Li_2(x).
    dilogarithm_difference = float(spence(1.0 - T2 / Tc) - spence(1.0 - T1 / Tc))
    polynomial = common.polynomial_difference(
        (
            0.0,
            a4 / Tc,
            a5 / (2.0 * Tc * Tc),
            a6 / (3.0 * Tc * Tc * Tc),
        ),
        T1,
        T2,
    )
    return common.R * (
        (a3 + a2) * log_temperature_ratio
        - a1 * dilogarithm_difference
        - a2 * log_critical_ratio
        + polynomial
    )


def compare_sources(first: SourceCandidate, second: SourceCandidate) -> Optional[dict[str, Any]]:
    comparison = common.compare_sources(first, second)
    if comparison is None:
        return None
    comparison = dict(comparison)
    comparison["close"] = bool(
        comparison["median_error_percent"] <= SOURCE_COMPARISON_MEDIAN_LIMIT_PERCENT
        and comparison["p95_error_percent"] <= SOURCE_COMPARISON_P95_LIMIT_PERCENT
    )
    comparison["strong_conflict"] = bool(
        comparison["median_error_percent"] > STRONG_CONFLICT_PERCENT
        and comparison["p95_error_percent"] > STRONG_CONFLICT_PERCENT
    )
    return comparison


def adjudicate_zabransky_form_conflicts(
    by_cas: Mapping[str, list[SourceCandidate]],
    quarantines: list[Quarantine],
) -> None:
    pairs = (
        ("zabransky_p_spline", "zabransky_p_quasi"),
        ("zabransky_C_spline", "zabransky_C_quasi"),
    )
    for cas, candidates in by_cas.items():
        by_source = {item.source: item for item in candidates if item.status == "available"}
        for first_source, second_source in pairs:
            first = by_source.get(first_source)
            second = by_source.get(second_source)
            if first is None or second is None:
                continue
            mutual = compare_sources(first, second)
            if mutual is None or not mutual["strong_conflict"]:
                continue
            independent = [
                item for item in candidates
                if item.status == "available" and item.lineage != "zabransky_1996"
            ]
            first_support = []
            second_support = []
            for alternative in independent:
                first_comparison = compare_sources(first, alternative)
                second_comparison = compare_sources(second, alternative)
                if first_comparison is not None and first_comparison["close"]:
                    first_support.append(alternative.source)
                if second_comparison is not None and second_comparison["close"]:
                    second_support.append(alternative.source)
            details = {
                "forms_comparison": mutual,
                "first_source": first_source,
                "second_source": second_source,
                "first_independent_support": first_support,
                "second_independent_support": second_support,
            }
            if first_support and not second_support:
                rejected = second
            elif second_support and not first_support:
                rejected = first
            else:
                for candidate in (first, second):
                    candidate.status = "internal_conflict"
                    candidate.reason = "unresolved catastrophic Zabransky form conflict"
                    quarantines.append(Quarantine(
                        cas, candidate.source, candidate.reason, details,
                    ))
                continue
            rejected.status = "outlier"
            rejected.reason = "Zabransky form conflicts with independently corroborated alternative"
            quarantines.append(Quarantine(cas, rejected.source, rejected.reason, details))


def obvious_consensus_outlier(
    candidate: SourceCandidate,
    alternatives: Sequence[SourceCandidate],
) -> tuple[bool, dict[str, Any]]:
    conflicting = []
    for alternative in alternatives:
        comparison = compare_sources(candidate, alternative)
        if comparison is not None and comparison["strong_conflict"]:
            conflicting.append((alternative, comparison))
    for first_index, (first, first_comparison) in enumerate(conflicting):
        for second, second_comparison in conflicting[first_index + 1:]:
            if first.lineage == second.lineage:
                continue
            mutual = compare_sources(first, second)
            if mutual is not None and mutual["close"]:
                return True, {
                    "candidate_vs_first": {"source": first.source, **first_comparison},
                    "candidate_vs_second": {"source": second.source, **second_comparison},
                    "alternatives_mutual": {
                        "first_source": first.source,
                        "second_source": second.source,
                        **mutual,
                    },
                }
    return False, {}


def pairwise_crosschecks(
    candidates: Sequence[SourceCandidate],
) -> tuple[Crosscheck, ...]:
    """Compare each eligible source pair once, before source selection."""
    result = []
    ordered = sorted(candidates, key=lambda item: (item.rank, item.source_key))
    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1:]:
            comparison = compare_sources(first, second)
            if comparison is None:
                continue
            result.append(Crosscheck(
                first.cas,
                first.source,
                second.source,
                first.lineage,
                second.lineage,
                comparison["point_count"],
                comparison["Tmin"],
                comparison["Tmax"],
                comparison["median_error_percent"],
                comparison["p95_error_percent"],
                comparison["max_error_percent"],
                comparison["close"],
                first.lineage != second.lineage,
            ))
    return tuple(result)


def corroborated_quality(
    candidate: SourceCandidate,
    fit: CompiledFit,
    checks: Sequence[Crosscheck],
) -> tuple[float, float, float, tuple[str, ...]]:
    corroborated_lineages = set()
    for check in checks:
        if not check.close_agreement or not check.independent:
            continue
        if check.first_source == candidate.source:
            corroborated_lineages.add(check.second_lineage)
        elif check.second_source == candidate.source:
            corroborated_lineages.add(check.first_lineage)
    bonus = min(
        MAX_CORROBORATION_BONUS,
        CORROBORATION_BONUS_PER_LINEAGE * len(corroborated_lineages),
    )
    penalty = common.fit_quality_penalty(
        fit.fit_max_error_percent,
        fit.fit_mape_percent,
    )
    quality = max(
        0.0,
        min(MAX_FINAL_QUALITY, candidate.base_quality + bonus) - penalty,
    )
    return quality, bonus, penalty, tuple(sorted(corroborated_lineages))


def compile_records() -> tuple[
    list[SelectedRecord],
    list[SourceCandidate],
    list[CandidateAssessment],
    list[Crosscheck],
    list[Quarantine],
    dict[str, Any],
]:
    candidates = collect_perry_candidates()
    coolprop, quarantines = collect_coolprop_candidates()
    zabransky, zabransky_quarantines, zabransky_report = collect_zabransky_candidates()
    nist, nist_quarantines = collect_webbook_candidates()
    candidates.extend(coolprop)
    candidates.extend(zabransky)
    candidates.extend(nist)
    quarantines.extend(zabransky_quarantines)
    quarantines.extend(nist_quarantines)
    candidates.sort(key=lambda item: (item.cas, item.rank, item.source_key))

    by_cas: dict[str, list[SourceCandidate]] = defaultdict(list)
    validation_details = {}
    for candidate in candidates:
        by_cas[candidate.cas].append(candidate)
        valid, reason, details = validate_source(candidate)
        validation_details[(candidate.cas, candidate.source)] = details
        if not valid:
            candidate.status = "invalid"
            candidate.reason = reason
            quarantines.append(Quarantine(candidate.cas, candidate.source, reason, details))

    adjudicate_zabransky_form_conflicts(by_cas, quarantines)

    # Resolve only obvious consensus outliers. Determine all flags against the
    # same candidate snapshot so iteration order cannot influence arbitration.
    outliers = []
    for cas, cas_candidates in by_cas.items():
        available = [item for item in cas_candidates if item.status == "available"]
        for candidate in available:
            alternatives = [item for item in available if item is not candidate]
            outlier, details = obvious_consensus_outlier(candidate, alternatives)
            if outlier:
                outliers.append((candidate, details))
    for candidate, details in outliers:
        if candidate.status != "available":
            continue
        candidate.status = "outlier"
        candidate.reason = "source is an obvious multi-source consensus outlier"
        quarantines.append(Quarantine(
            candidate.cas, candidate.source, candidate.reason, details,
        ))

    fits_by_candidate = {}
    for candidate in candidates:
        if candidate.status != "available":
            continue
        fit = compile_candidate(candidate)
        if fit is None:
            candidate.status = "fit_failed"
            candidate.reason = "degree 8/12/16 fit could not produce a finite positive canonical curve"
            quarantines.append(Quarantine(
                candidate.cas,
                candidate.source,
                candidate.reason,
                {"Tmin": candidate.Tmin, "Tmax": candidate.Tmax},
            ))
            continue
        fits_by_candidate[id(candidate)] = fit

    assessments = []
    selected_records = []
    all_crosschecks = []
    unavailable = []
    for cas, cas_candidates in sorted(by_cas.items()):
        eligible_candidates = [
            candidate for candidate in cas_candidates if id(candidate) in fits_by_candidate
        ]
        if not eligible_candidates:
            unavailable.append(cas)
            continue
        checks = pairwise_crosschecks(eligible_candidates)
        all_crosschecks.extend(checks)
        eligible = []
        for candidate in eligible_candidates:
            fit = fits_by_candidate[id(candidate)]
            quality, bonus, penalty, lineages = corroborated_quality(candidate, fit, checks)
            assessment = CandidateAssessment(
                candidate, fit, penalty, bonus, quality, lineages,
            )
            assessments.append(assessment)
            eligible.append(assessment)
        # Python's stable tuple ordering makes the exact-quality tie policy
        # explicit. Source rank precedes range breadth by request.
        selected = min(
            eligible,
            key=lambda item: (
                -item.selection_quality,
                item.candidate.rank,
                -(item.fit.Tmax - item.fit.Tmin),
                item.candidate.source_key,
            ),
        )
        selected.candidate.status = "selected"
        selected.candidate.reason = "highest corroborated quality after fitting penalty"
        for assessment in eligible:
            if assessment is selected:
                continue
            assessment.candidate.status = "not_selected"
            assessment.candidate.reason = "lower corroborated quality or exact-quality tie priority"
        selected_checks = tuple(
            check for check in checks
            if selected.candidate.source in (check.first_source, check.second_source)
        )
        selected_records.append(SelectedRecord(selected, selected_checks))

    report = {
        "candidate_count": len(candidates),
        "unique_cas_count": len(by_cas),
        "selected_count": len(selected_records),
        "unavailable_count": len(unavailable),
        "unavailable_cas": unavailable,
        "selected_by_source": dict(Counter(record.candidate.source for record in selected_records)),
        "candidate_statuses": dict(Counter(candidate.status for candidate in candidates)),
        "candidate_sources": dict(Counter(candidate.source for candidate in candidates)),
        "compiled_models": dict(Counter(compiled_model(record.fit) for record in selected_records)),
        "fit_degrees": dict(Counter(
            record.fit.degree for record in selected_records if record.fit.degree is not None
        )),
        "corroborated_count": sum(record.corroboration_bonus > 0.0 for record in selected_records),
        "corroborated_candidate_count": sum(
            assessment.corroboration_bonus > 0.0 for assessment in assessments
        ),
        "maximum_fit_error_percent": max(
            (record.fit.fit_max_error_percent for record in selected_records), default=0.0,
        ),
        "maximum_fit_h_error_percent": max(
            (record.fit.fit_h_max_error_percent for record in selected_records), default=0.0,
        ),
        "maximum_fit_s_error_percent": max(
            (record.fit.fit_s_max_error_percent for record in selected_records), default=0.0,
        ),
        "zabransky": zabransky_report,
        "validation_details": validation_details,
    }
    return selected_records, candidates, assessments, all_crosschecks, quarantines, report


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        f"""
        PRAGMA foreign_keys = ON;
        PRAGMA user_version = {SCHEMA_VERSION};

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );

        CREATE TABLE canonical_liquid_cp (
            cas TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            formula TEXT NOT NULL,
            model TEXT NOT NULL,
            source TEXT NOT NULL,
            source_label TEXT NOT NULL,
            source_rank INTEGER NOT NULL,
            source_key TEXT NOT NULL,
            source_version TEXT NOT NULL,
            source_lineage TEXT NOT NULL,
            source_fingerprint TEXT NOT NULL,
            quality REAL NOT NULL CHECK (quality >= 0.0 AND quality <= 0.995),
            base_quality REAL NOT NULL,
            selection_quality REAL NOT NULL,
            corroboration_bonus REAL NOT NULL,
            fit_quality_penalty REAL NOT NULL CHECK (fit_quality_penalty >= 0.0),
            Tmin_source_K REAL NOT NULL,
            Tmax_source_K REAL NOT NULL,
            Tmin_fit_K REAL NOT NULL,
            Tmax_fit_K REAL NOT NULL,
            Tmin_raised_K REAL NOT NULL,
            model_payload_json TEXT NOT NULL,
            critical_temperature_K REAL,
            degree INTEGER CHECK (degree IS NULL OR degree IN (8, 12, 16)),
            map_center_K REAL,
            map_scale REAL,
            cp_coefficients_json TEXT,
            h_polynomial_json TEXT,
            h_log_coefficient REAL,
            h_reciprocal_coefficient REAL,
            s_polynomial_json TEXT,
            s_log_minus_coefficient REAL,
            s_log_plus_coefficient REAL,
            fit_mape_percent REAL NOT NULL,
            fit_max_error_percent REAL NOT NULL,
            fit_h_max_error_percent REAL NOT NULL,
            fit_s_max_error_percent REAL NOT NULL,
            validation_points INTEGER NOT NULL,
            corroboration_count INTEGER NOT NULL,
            independent_corroboration_count INTEGER NOT NULL,
            corroboration_json TEXT NOT NULL,
            source_details_json TEXT NOT NULL,
            notes_json TEXT NOT NULL
        );

        CREATE TABLE source_candidate_audit (
            cas TEXT NOT NULL,
            source TEXT NOT NULL,
            source_label TEXT NOT NULL,
            source_rank INTEGER NOT NULL,
            source_key TEXT NOT NULL,
            source_version TEXT NOT NULL,
            source_lineage TEXT NOT NULL,
            source_fingerprint TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT NOT NULL,
            Tmin_K REAL NOT NULL,
            Tmax_K REAL NOT NULL,
            base_quality REAL NOT NULL,
            selection_quality REAL,
            corroboration_bonus REAL,
            fit_quality_penalty REAL,
            compiled_model TEXT,
            fit_degree INTEGER,
            fit_mape_percent REAL,
            fit_max_error_percent REAL,
            details_json TEXT NOT NULL,
            notes_json TEXT NOT NULL,
            PRIMARY KEY (cas, source)
        );

        CREATE TABLE source_crosscheck (
            cas TEXT NOT NULL,
            first_source TEXT NOT NULL,
            second_source TEXT NOT NULL,
            first_lineage TEXT NOT NULL,
            second_lineage TEXT NOT NULL,
            point_count INTEGER NOT NULL,
            Tmin_K REAL NOT NULL,
            Tmax_K REAL NOT NULL,
            median_error_percent REAL NOT NULL,
            p95_error_percent REAL NOT NULL,
            max_error_percent REAL NOT NULL,
            close_agreement INTEGER NOT NULL,
            independent INTEGER NOT NULL,
            PRIMARY KEY (cas, first_source, second_source)
        );

        CREATE TABLE source_quarantine (
            id INTEGER PRIMARY KEY,
            cas TEXT NOT NULL,
            source TEXT NOT NULL,
            reason TEXT NOT NULL,
            details_json TEXT NOT NULL
        );

        CREATE INDEX idx_canonical_liquid_cp_source
        ON canonical_liquid_cp (source);
        CREATE INDEX idx_canonical_liquid_cp_quality
        ON canonical_liquid_cp (quality);
        CREATE INDEX idx_liquid_candidate_status
        ON source_candidate_audit (status);
        CREATE INDEX idx_liquid_crosscheck_close
        ON source_crosscheck (close_agreement, independent);
        CREATE INDEX idx_liquid_quarantine_cas
        ON source_quarantine (cas);
        """
    )


def write_database(
    output: Path,
    selected: Sequence[SelectedRecord],
    candidates: Sequence[SourceCandidate],
    assessments: Sequence[CandidateAssessment],
    crosschecks: Sequence[Crosscheck],
    quarantines: Sequence[Quarantine],
    report: dict[str, Any],
) -> Optional[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if output.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = Path(tempfile.gettempdir()) / f"{output.stem}.backup-{timestamp}{output.suffix}"
        shutil.copy2(output, backup)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".tmp.sqlite", dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    assessment_by_candidate = {id(item.candidate): item for item in assessments}
    try:
        with closing(sqlite3.connect(temporary)) as connection:
            with connection:
                create_schema(connection)
                metadata = {
                    "schema_version": SCHEMA_VERSION,
                    "model": MODEL_NAME,
                    "built_at_utc": datetime.now(timezone.utc).isoformat(),
                    "fit_degrees": FIT_DEGREES,
                    "entropy_primitive_degree_policy": (
                        "Cp degree + 16 + 4*ceil(log2(Tmax/Tmin))"
                    ),
                    "fit_max_relative_error_percent": common.FIT_MAX_RELATIVE_ERROR_PERCENT,
                    "fit_zero_penalty_max_error_percent": common.FIT_ZERO_PENALTY_MAX_ERROR_PERCENT,
                    "fit_zero_penalty_mape_percent": common.FIT_ZERO_PENALTY_MAPE_PERCENT,
                    "fit_max_error_quality_weight": common.FIT_MAX_ERROR_QUALITY_WEIGHT,
                    "fit_mape_quality_weight": common.FIT_MAPE_QUALITY_WEIGHT,
                    "fit_Tmin_can_be_raised": False,
                    "selection_policy": (
                        "all fitted candidates receive independent-lineage corroboration; "
                        "highest corroborated quality minus fit penalty; exact tie by source rank"
                    ),
                    "exact_quality_tie_priority": [
                        source for source, _spec in sorted(
                            SOURCE_SPECS.items(), key=lambda item: item[1]["rank"]
                        )
                    ],
                    "source_specs": SOURCE_SPECS,
                    "zabransky_class_quality": ZABRANSKY_CLASS_QUALITY,
                    "zabransky_averaged_quality_factor": ZABRANSKY_AVERAGED_QUALITY_FACTOR,
                    "coolprop_maximum_reduced_temperature": COOLPROP_MAX_REDUCED_TEMPERATURE,
                    "excluded_semantics": ["Zabransky saturation-line heat capacity"],
                    "report": {
                        key: value for key, value in report.items()
                        if key != "validation_details"
                    },
                }
                connection.executemany(
                    "INSERT INTO metadata (key, value_json) VALUES (?, ?)",
                    [(key, common.canonical_json(value)) for key, value in metadata.items()],
                )
                for record in selected:
                    candidate = record.candidate
                    fit = record.fit
                    independent_lineages = record.assessment.corroborated_lineages
                    if isinstance(fit, NativeZabranskyFit):
                        critical_temperature = fit.critical_temperature
                        degree = map_center = map_scale = None
                        cp_coefficients = common.canonical_json(fit.coefficients)
                        h_polynomial = s_polynomial = None
                        h_log_coefficient = h_reciprocal_coefficient = None
                        s_log_minus_coefficient = s_log_plus_coefficient = None
                    else:
                        critical_temperature = common.finite_float(
                            candidate.details.get("critical_temperature_K")
                        )
                        degree = fit.degree
                        map_center = fit.center
                        map_scale = fit.half_width
                        cp_coefficients = common.canonical_json(fit.coefficients)
                        h_polynomial = common.canonical_json(fit.h_coefficients)
                        h_log_coefficient = 0.0
                        h_reciprocal_coefficient = 0.0
                        s_polynomial = common.canonical_json(fit.s_coefficients)
                        s_log_minus_coefficient = 0.0
                        s_log_plus_coefficient = 0.0
                    connection.execute(
                        """
                        INSERT INTO canonical_liquid_cp (
                            cas, name, formula, model, source, source_label, source_rank,
                            source_key, source_version, source_lineage, source_fingerprint,
                            quality, base_quality, selection_quality,
                            corroboration_bonus, fit_quality_penalty,
                            Tmin_source_K, Tmax_source_K, Tmin_fit_K, Tmax_fit_K, Tmin_raised_K,
                            model_payload_json, critical_temperature_K,
                            degree, map_center_K, map_scale, cp_coefficients_json,
                            h_polynomial_json, h_log_coefficient, h_reciprocal_coefficient,
                            s_polynomial_json, s_log_minus_coefficient, s_log_plus_coefficient,
                            fit_mape_percent, fit_max_error_percent,
                            fit_h_max_error_percent, fit_s_max_error_percent,
                            validation_points, corroboration_count,
                            independent_corroboration_count, corroboration_json,
                            source_details_json, notes_json
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            candidate.cas, candidate.name, candidate.formula, compiled_model(fit),
                            candidate.source, candidate.label, candidate.rank,
                            candidate.source_key, candidate.source_version, candidate.lineage,
                            candidate.fingerprint, record.quality, candidate.base_quality,
                            record.assessment.selection_quality, record.corroboration_bonus,
                            record.assessment.fit_quality_penalty,
                            candidate.Tmin, candidate.Tmax,
                            fit.Tmin, fit.Tmax, fit.Tmin - candidate.Tmin,
                            common.canonical_json(compiled_payload(fit)),
                            critical_temperature, degree, map_center, map_scale,
                            cp_coefficients, h_polynomial, h_log_coefficient,
                            h_reciprocal_coefficient, s_polynomial,
                            s_log_minus_coefficient, s_log_plus_coefficient,
                            fit.fit_mape_percent, fit.fit_max_error_percent,
                            fit.fit_h_max_error_percent, fit.fit_s_max_error_percent,
                            fit.validation_points,
                            sum(check.close_agreement for check in record.crosschecks),
                            len(independent_lineages),
                            common.canonical_json([check.__dict__ for check in record.crosschecks]),
                            common.canonical_json(candidate.details),
                            common.canonical_json(candidate.notes),
                        ),
                    )
                connection.executemany(
                    """
                    INSERT INTO source_candidate_audit (
                        cas, source, source_label, source_rank, source_key, source_version,
                        source_lineage, source_fingerprint, status, reason, Tmin_K, Tmax_K,
                        base_quality, selection_quality, corroboration_bonus,
                        fit_quality_penalty, compiled_model, fit_degree,
                        fit_mape_percent, fit_max_error_percent, details_json, notes_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            candidate.cas, candidate.source, candidate.label, candidate.rank,
                            candidate.source_key, candidate.source_version, candidate.lineage,
                            candidate.fingerprint, candidate.status, candidate.reason,
                            candidate.Tmin, candidate.Tmax, candidate.base_quality,
                            (
                                assessment_by_candidate[id(candidate)].selection_quality
                                if id(candidate) in assessment_by_candidate else None
                            ),
                            (
                                assessment_by_candidate[id(candidate)].corroboration_bonus
                                if id(candidate) in assessment_by_candidate else None
                            ),
                            (
                                assessment_by_candidate[id(candidate)].fit_quality_penalty
                                if id(candidate) in assessment_by_candidate else None
                            ),
                            (
                                compiled_model(assessment_by_candidate[id(candidate)].fit)
                                if id(candidate) in assessment_by_candidate else None
                            ),
                            (
                                assessment_by_candidate[id(candidate)].fit.degree
                                if id(candidate) in assessment_by_candidate else None
                            ),
                            (
                                assessment_by_candidate[id(candidate)].fit.fit_mape_percent
                                if id(candidate) in assessment_by_candidate else None
                            ),
                            (
                                assessment_by_candidate[id(candidate)].fit.fit_max_error_percent
                                if id(candidate) in assessment_by_candidate else None
                            ),
                            common.canonical_json(candidate.details),
                            common.canonical_json(candidate.notes),
                        )
                        for candidate in candidates
                    ],
                )
                connection.executemany(
                    """
                    INSERT INTO source_crosscheck (
                        cas, first_source, second_source, first_lineage,
                        second_lineage, point_count, Tmin_K, Tmax_K,
                        median_error_percent, p95_error_percent, max_error_percent,
                        close_agreement, independent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item.cas, item.first_source, item.second_source,
                            item.first_lineage, item.second_lineage, item.point_count,
                            item.Tmin, item.Tmax, item.median_error_percent,
                            item.p95_error_percent, item.max_error_percent,
                            int(item.close_agreement), int(item.independent),
                        )
                        for item in crosschecks
                    ],
                )
                connection.executemany(
                    """
                    INSERT INTO source_quarantine (cas, source, reason, details_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (item.cas, item.source, item.reason, common.canonical_json(item.details))
                        for item in quarantines
                    ],
                )
                integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
                if integrity != "ok":
                    raise RuntimeError(f"temporary database integrity check failed: {integrity}")
        os.replace(temporary, output)
        output.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()
    return backup


def validate_database(path: Path) -> dict[str, Any]:
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"database integrity check failed: {integrity}")
        rows = connection.execute("SELECT * FROM canonical_liquid_cp ORDER BY cas").fetchall()
        if not rows:
            raise RuntimeError("canonical liquid database contains no rows")
        maximum_primitive_error = 0.0
        for row in rows:
            expected_selection = max(
                0.0,
                min(MAX_FINAL_QUALITY, row["base_quality"] + row["corroboration_bonus"])
                - row["fit_quality_penalty"],
            )
            if abs(row["selection_quality"] - expected_selection) > 1e-12:
                raise RuntimeError(f"selection quality mismatch for {row['cas']}")
            if abs(row["quality"] - expected_selection) > 1e-12:
                raise RuntimeError(f"final quality mismatch for {row['cas']}")
            payload = json.loads(row["model_payload_json"])
            if row["model"] == MODEL_NAME:
                coefficients = tuple(json.loads(row["cp_coefficients_json"]))
                h_coefficients = tuple(json.loads(row["h_polynomial_json"]))
                s_coefficients = tuple(json.loads(row["s_polynomial_json"]))
                if len(coefficients) != row["degree"] + 1:
                    raise RuntimeError(f"coefficient count mismatch for {row['cas']}")
                fit = LinearChebyshevFit(
                    row["Tmin_fit_K"], row["Tmax_fit_K"], row["degree"],
                    row["map_center_K"], row["map_scale"], coefficients,
                    row["fit_mape_percent"], row["fit_max_error_percent"],
                    row["fit_h_max_error_percent"], row["fit_s_max_error_percent"],
                    row["validation_points"], h_coefficients, s_coefficients,
                )
                evaluate_cp = lambda values_T, fit=fit: evaluate_linear_fit(fit, values_T)
                delta_h = lambda first, second, fit=fit: integrate_h_linear(fit, first, second)
                delta_s = lambda first, second, fit=fit: integrate_s_linear(fit, first, second)
            elif row["model"] == NATIVE_ZABRANSKY_MODEL:
                import chemicals.heat_capacity as heat_capacity

                coefficients = tuple(float(value) for value in payload["coefficients"])
                critical_temperature = float(payload["critical_temperature_K"])
                fit = NativeZabranskyFit(
                    row["Tmin_fit_K"], row["Tmax_fit_K"], critical_temperature,
                    coefficients, validation_points=row["validation_points"],
                )
                evaluate_cp = lambda values_T, fit=fit: np.asarray([
                    heat_capacity.Zabransky_quasi_polynomial(
                        float(T), fit.critical_temperature, *fit.coefficients,
                    )
                    for T in values_T
                ])
                delta_h = lambda first, second, fit=fit: integrate_h_zabransky_native(
                    fit, first, second,
                )
                delta_s = lambda first, second, fit=fit: integrate_s_zabransky_native(
                    fit, first, second,
                )
            else:
                raise RuntimeError(f"unknown stored model for {row['cas']}: {row['model']}")
            temperatures = np.geomspace(fit.Tmin, fit.Tmax, 33)
            cp = evaluate_cp(temperatures)
            if np.any(~np.isfinite(cp)) or np.any(cp <= 0.0):
                raise RuntimeError(f"invalid stored heat-capacity curve for {row['cas']}")
            for first, second in zip(temperatures[:-1], temperatures[1:]):
                midpoint = math.sqrt(first * second)
                delta = max(1e-4, midpoint * 1e-6)
                numerical_cp = delta_h(midpoint - delta, midpoint + delta) / (2.0 * delta)
                reference_cp = float(evaluate_cp([midpoint])[0])
                error = abs(numerical_cp / reference_cp - 1.0)
                maximum_primitive_error = max(maximum_primitive_error, error)
                if error > 2e-7:
                    raise RuntimeError(f"enthalpy primitive mismatch for {row['cas']}")
                numerical_entropy_cp = (
                    delta_s(midpoint - delta, midpoint + delta) / (2.0 * delta) * midpoint
                )
                error = abs(numerical_entropy_cp / reference_cp - 1.0)
                maximum_primitive_error = max(maximum_primitive_error, error)
                if error > 2e-7:
                    raise RuntimeError(f"entropy primitive mismatch for {row['cas']}")

        # Verify quality-first selection and exact-quality source tie handling.
        for row in rows:
            candidates = connection.execute(
                """
                SELECT source, source_rank, source_key, selection_quality
                FROM source_candidate_audit
                WHERE cas = ? AND selection_quality IS NOT NULL
                """,
                (row["cas"],),
            ).fetchall()
            expected = min(
                candidates,
                key=lambda item: (
                    -item["selection_quality"], item["source_rank"], item["source_key"],
                ),
            )
            if expected["source"] != row["source"]:
                raise RuntimeError(f"quality-first source selection mismatch for {row['cas']}")
        return {
            "integrity": integrity,
            "record_count": len(rows),
            "maximum_primitive_relative_error": maximum_primitive_error,
            "selected_by_source": dict(Counter(row["source"] for row in rows)),
            "quality_minimum": min(row["quality"] for row in rows),
            "quality_mean": sum(row["quality"] for row in rows) / len(rows),
            "quality_maximum": max(row["quality"] for row in rows),
        }


def self_test() -> None:
    if _zabransky_quality(["I"], False) != 0.995:
        raise AssertionError("Zabransky p quality mapping failed")
    expected = 0.990 * ZABRANSKY_AVERAGED_QUALITY_FACTOR
    if abs(_zabransky_quality(["II"], True) - expected) > 1e-15:
        raise AssertionError("Zabransky averaged quality mapping failed")
    if _zabransky_quality(["II", "V"], False) != 0.900:
        raise AssertionError("piecewise Zabransky quality must use worst segment")
    if SOURCE_SPECS["perry_9e"]["rank"] >= SOURCE_SPECS["coolprop_liquid"]["rank"]:
        raise AssertionError("Perry must win an exact quality tie over CoolProp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help=f"SQLite output path (default: {OUTPUT})",
    )
    arguments = parser.parse_args()
    self_test()
    selected, candidates, assessments, crosschecks, quarantines, report = compile_records()
    backup = write_database(
        arguments.output,
        selected,
        candidates,
        assessments,
        crosschecks,
        quarantines,
        report,
    )
    validation = validate_database(arguments.output)
    summary = {
        "output": str(arguments.output),
        "backup": str(backup) if backup else None,
        "report": {key: value for key, value in report.items() if key != "validation_details"},
        "quarantine_count": len(quarantines),
        "validation": validation,
    }
    print(json.dumps(common.sanitize_json(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
