#!/usr/bin/env python3
"""Compile ideal-gas heat-capacity sources into one-fit CAS records.

This is an offline source compiler.  It does not modify or integrate with the
runtime property resolver.  Every selected compound is represented by one
rationally mapped Chebyshev polynomial over one contiguous temperature range.

Source priority is deliberately explicit:

    Perry 9e > CoolProp CP0MOLAR > TRC > NIST WebBook Shomate
    > JANAF > Poling polynomial > adjusted Psi4 RRHO

Degree 8 is attempted first, followed by degree 12.  Only when both fail the
maximum-relative-error contract is Tmin raised and the same sequence retried.
The generated SQLite database retains all source audits, cross-checks, and
quarantines alongside the selected canonical records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
from numpy.polynomial import chebyshev as ncheb
from numpy.polynomial import polynomial as npoly


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUTPUT = ROOT / "data" / "ideal_gas_heat_capacity.sqlite"
SCHEMA_VERSION = 3
MODEL_NAME = "rational_chebyshev_ideal_gas_cp_v1"
FIT_DEGREES = (8, 12)
FIT_MAX_RELATIVE_ERROR_PERCENT = 0.1
FIT_ZERO_PENALTY_MAX_ERROR_PERCENT = 0.1
FIT_ZERO_PENALTY_MAPE_PERCENT = 0.01
FIT_MAX_ERROR_QUALITY_WEIGHT = 0.031
FIT_MAPE_QUALITY_WEIGHT = 0.185
FIT_TMIN_STEP_K = 5.0
FIT_TMIN_RAISE_LIMIT_K = 273.15
TRAINING_POINTS_PER_RANGE = 161
VALIDATION_POINTS_PER_RANGE = 801
COMPARISON_POINTS_PER_RANGE = 101
R = 8.31446261815324
CAS_PATTERN = re.compile(r"^[1-9][0-9]{1,6}-[0-9]{2}-[0-9]$")

# The chemicals 1.5.2 metadata row for this CAS is an unrelated complex
# organic identity.  NIST WebBook and PubChem both identify it as the ClO
# radical, consistent with the TRC, Shomate, and JANAF heat-capacity sources.
SOURCE_IDENTITY_OVERRIDES = {
    "14989-30-1": ("Monochlorine monoxide", "ClO"),
}


SOURCE_SPECS = {
    "perry_9e": {
        "rank": 1,
        "label": "Perry 9th edition",
        "lineage": "perry_9e",
        "quality": 0.980,
    },
    "coolprop_cp0": {
        "rank": 2,
        "label": "CoolProp CP0MOLAR",
        "lineage": "coolprop_heos",
        "quality": 0.980,
    },
    "trc_1994": {
        "rank": 3,
        "label": "TRC gas-state correlation (1994)",
        "lineage": "trc_1994",
        "quality": 0.975,
    },
    "webbook_shomate": {
        "rank": 4,
        "label": "NIST WebBook Shomate",
        "lineage": "nist_webbook",
        "quality": 0.975,
    },
    "janaf_1998": {
        "rank": 5,
        "label": "JANAF 1998",
        "lineage": "nist_webbook",
        "quality": 0.970,
    },
    "poling_2001": {
        "rank": 6,
        "label": "Poling et al. gas polynomial",
        "lineage": "poling_2001",
        "quality": 0.970,
    },
    "psi4_adjusted": {
        "rank": 7,
        "label": "Adjusted Psi4 RRHO",
        "lineage": "psi4_2022a",
        "quality": 0.900,
    },
}


@dataclass(frozen=True)
class SourceSegment:
    Tmin: float
    Tmax: float
    evaluate_vector: Callable[[np.ndarray], np.ndarray] = field(compare=False, repr=False)
    details: dict[str, Any] = field(default_factory=dict)


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
        return float(SOURCE_SPECS[self.source]["quality"])

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
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def values(self, temperatures: Sequence[float] | np.ndarray) -> np.ndarray:
        values_T = np.asarray(temperatures, dtype=float)
        result = np.full(values_T.shape, np.nan, dtype=float)
        # Segment order is meaningful for Perry's low-temperature polynomial
        # handoff.  The first source segment owns an overlapping temperature.
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
class CanonicalFit:
    Tmin: float
    Tmax: float
    degree: int
    center: float
    scale: float
    coefficients: tuple[float, ...]
    fit_mape_percent: float
    fit_max_error_percent: float
    fit_h_max_error_percent: float
    fit_s_max_error_percent: float
    validation_points: int
    h_polynomial: tuple[float, ...]
    h_log_coefficient: float
    h_reciprocal_coefficient: float
    s_polynomial: tuple[float, ...]
    s_log_minus_coefficient: float
    s_log_plus_coefficient: float


@dataclass(frozen=True)
class Crosscheck:
    cas: str
    selected_source: str
    comparison_source: str
    selected_lineage: str
    comparison_lineage: str
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
    candidate: SourceCandidate
    fit: CanonicalFit
    quality: float
    corroboration_bonus: float
    fit_quality_penalty: float
    crosschecks: tuple[Crosscheck, ...]


@dataclass(frozen=True)
class Quarantine:
    cas: str
    source: str
    reason: str
    details: dict[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(
        sanitize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sanitize_json(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return sanitize_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def valid_cas(value: Any) -> bool:
    return bool(CAS_PATTERN.fullmatch(str(value or "").strip()))


def identity_for_cas(cas: str, fallback_name: str = "", fallback_formula: str = "") -> tuple[str, str]:
    if cas in SOURCE_IDENTITY_OVERRIDES:
        return SOURCE_IDENTITY_OVERRIDES[cas]
    try:
        from chemicals.identifiers import search_chemical

        metadata = search_chemical(cas)
        metadata_name = str(getattr(metadata, "common_name", "") or "")
        metadata_formula = str(getattr(metadata, "formula", "") or "")
        source_name = str(fallback_name or "").strip()
        source_formula = str(fallback_formula or "").strip()
        normalized_source_name = re.sub(r"[^a-z0-9]", "", source_name.lower())
        normalized_metadata_name = re.sub(r"[^a-z0-9]", "", metadata_name.lower())
        names_compatible = bool(
            not source_name
            or not metadata_name
            or normalized_source_name == normalized_metadata_name
            or normalized_source_name in normalized_metadata_name
            or normalized_metadata_name in normalized_source_name
        )
        return (
            source_name or metadata_name or cas,
            source_formula or (metadata_formula if names_compatible else ""),
        )
    except Exception:
        return fallback_name or cas, fallback_formula or ""


def vectorized_scalar(function: Callable[[float], float]) -> Callable[[np.ndarray], np.ndarray]:
    def evaluate(temperatures: np.ndarray) -> np.ndarray:
        return np.asarray([function(float(T)) for T in temperatures], dtype=float)

    return evaluate


def model_segments(model: Any, details_kind: str) -> tuple[SourceSegment, ...]:
    models = tuple(getattr(model, "models", (model,)))
    result = []
    for item in models:
        coeffs = tuple(float(value) for value in getattr(item, "coeffs", ()))
        result.append(SourceSegment(
            Tmin=float(item.Tmin),
            Tmax=float(item.Tmax),
            evaluate_vector=vectorized_scalar(item.calculate),
            details={"kind": details_kind, "coefficients": coeffs},
        ))
    return tuple(result)


def collect_perry_candidates() -> list[SourceCandidate]:
    from perry_properties import DATA_PATH, get_perry_property_library

    library = get_perry_property_library()
    library._load()
    source_hash = sha256_path(DATA_PATH)
    source_version = str(library.metadata.get("source_book") or "Perry 9th edition")
    candidates = []
    for cas, entry in sorted(library.chemicals.items()):
        if not valid_cas(cas):
            continue
        segments = []
        rows = []
        for row in entry.get("ideal_gas_heat_capacity_polynomial", ()):
            def evaluate(Ts: np.ndarray, row: dict[str, Any] = row) -> np.ndarray:
                return np.asarray([
                    library._eval_ideal_gas_polynomial_J_per_kmol_K(row, float(T)) / 1000.0
                    for T in Ts
                ], dtype=float)

            segments.append(SourceSegment(
                float(row["T_min_K"]), float(row["T_max_K"]), evaluate,
                {"kind": "perry_polynomial", "row": row},
            ))
            rows.append({"kind": "polynomial", **row})
        for row in entry.get("ideal_gas_heat_capacity_hyperbolic", ()):
            def evaluate(Ts: np.ndarray, row: dict[str, Any] = row) -> np.ndarray:
                return np.asarray([
                    library._eval_ideal_gas_hyperbolic_J_per_kmol_K(row, float(T)) / 1000.0
                    for T in Ts
                ], dtype=float)

            segments.append(SourceSegment(
                float(row["T_min_K"]), float(row["T_max_K"]), evaluate,
                {"kind": "perry_hyperbolic", "row": row},
            ))
            rows.append({"kind": "hyperbolic", **row})
        if not segments:
            continue
        name, formula = identity_for_cas(
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
            segments=tuple(segments),
            source_hash=source_hash,
            details={"rows": rows},
        ))
    return candidates


def collect_coolprop_candidates() -> tuple[list[SourceCandidate], list[Quarantine]]:
    try:
        import CoolProp
        import CoolProp.CoolProp as CP
    except ImportError:
        return [], [Quarantine("", "coolprop_cp0", "CoolProp unavailable", {})]

    version = str(getattr(CoolProp, "__version__", "unknown"))
    candidates = []
    quarantines = []
    seen_cas = set()
    for fluid in sorted(CP.get_global_param_string("FluidsList").split(",")):
        try:
            if CP.get_fluid_param_string(fluid, "pure").strip().lower() != "true":
                continue
            cas = CP.get_fluid_param_string(fluid, "CAS").strip()
            if not valid_cas(cas) or cas in seen_cas:
                continue
            state = CP.AbstractState("HEOS", fluid)
            source_Tmin = float(state.Tmin())
            source_Tmax = float(state.Tmax())
            # Several pure-fluid backends reject a PT call at the exact triple
            # endpoint even though CP0MOLAR is otherwise valid immediately above.
            Tmin = max(50.0, source_Tmin + max(1e-6, abs(source_Tmin) * 1e-9))
            Tmax = source_Tmax
            if not (math.isfinite(Tmin) and math.isfinite(Tmax) and Tmin < Tmax):
                raise ValueError("invalid CoolProp temperature range")

            def evaluate(Ts: np.ndarray, fluid: str = fluid) -> np.ndarray:
                return np.asarray(
                    CP.PropsSI("CP0MOLAR", "T", np.asarray(Ts, dtype=float), "P", 101325.0, fluid),
                    dtype=float,
                )

            probe_T = np.geomspace(Tmin, Tmax, 257)
            probe = evaluate(probe_T)
            if np.any(~np.isfinite(probe)) or np.any(probe <= 0.0):
                raise ValueError("nonfinite or nonpositive CP0MOLAR band")
            name, formula = identity_for_cas(
                cas,
                fluid,
                CP.get_fluid_param_string(fluid, "formula").strip(),
            )
            source_hash = hashlib.sha256(
                f"CoolProp|{version}|HEOS|{fluid}|{cas}".encode("utf-8")
            ).hexdigest()
            candidates.append(SourceCandidate(
                cas=cas,
                source="coolprop_cp0",
                source_key=fluid,
                source_version=f"CoolProp {version}",
                name=name,
                formula=formula,
                segments=(SourceSegment(
                    Tmin, Tmax, evaluate,
                    {"fluid": fluid, "backend": "HEOS", "property": "CP0MOLAR"},
                ),),
                source_hash=source_hash,
                details={
                    "fluid": fluid,
                    "backend": "HEOS",
                    "property": "CP0MOLAR",
                    "source_Tmin_K": source_Tmin,
                    "source_Tmax_K": source_Tmax,
                    "sample_pressure_Pa": 101325.0,
                },
            ))
            seen_cas.add(cas)
        except Exception as error:
            cas = ""
            try:
                candidate_cas = CP.get_fluid_param_string(fluid, "CAS").strip()
                if valid_cas(candidate_cas):
                    cas = candidate_cas
            except Exception:
                pass
            quarantines.append(Quarantine(
                cas, "coolprop_cp0", "CoolProp source rejected",
                {"fluid": fluid, "error": f"{type(error).__name__}: {error}"},
            ))
    return candidates, quarantines


def collect_chemicals_candidates() -> tuple[list[SourceCandidate], list[Quarantine]]:
    import chemicals
    import chemicals.heat_capacity as heat_capacity

    # Trigger the lazy-loaded datasets once in the offline compiler.
    _ = heat_capacity.Cp_data_Poling
    folder = Path(heat_capacity.folder)
    version = str(chemicals.__version__)
    paths = {
        "trc_1994": folder / "TRC Thermodynamics of Organic Compounds in the Gas State.tsv",
        "webbook_shomate": folder / "webbook_shomate_coefficients.json",
        "janaf_1998": folder / "JANAF_1998_gas_Cp.json",
        "poling_2001": folder / "PolingDatabank.tsv",
        "psi4_adjusted": folder / "psi4_adjusted_characteristic_temperatures.json",
    }
    hashes = {key: sha256_path(path) for key, path in paths.items()}
    candidates = []
    quarantines = []

    for cas, row in heat_capacity.TRC_gas_data.sort_index().iterrows():
        cas = str(cas)
        if not valid_cas(cas):
            continue
        coeffs = tuple(float(row[f"a{index}"]) for index in range(8))

        def evaluate(Ts: np.ndarray, coeffs: tuple[float, ...] = coeffs) -> np.ndarray:
            return np.asarray([
                heat_capacity.TRCCp(float(T), *coeffs) for T in Ts
            ], dtype=float)

        name, formula = identity_for_cas(cas, str(row["Chemical"]), "")
        candidates.append(SourceCandidate(
            cas, "trc_1994", cas, f"chemicals {version}; TRC 1994", name, formula,
            (SourceSegment(float(row["Tmin"]), float(row["Tmax"]), evaluate,
                           {"kind": "TRCCp", "coefficients": coeffs}),),
            hashes["trc_1994"],
            {"coefficients": coeffs, "I": row["I"], "J": row["J"], "Hfg": row["Hfg"]},
        ))

    for cas, model in sorted(heat_capacity.WebBook_Shomate_gases.items()):
        if not valid_cas(cas):
            continue
        segments = model_segments(model, "Shomate")
        name, formula = identity_for_cas(cas)
        candidates.append(SourceCandidate(
            cas, "webbook_shomate", cas, f"chemicals {version}; NIST WebBook snapshot",
            name, formula, segments, hashes["webbook_shomate"],
            {"segments": [segment.details | {"Tmin": segment.Tmin, "Tmax": segment.Tmax}
                          for segment in segments]},
        ))

    for cas, table in sorted(heat_capacity.Cp_dict_JANAF_gas.items()):
        if not valid_cas(cas):
            continue
        temperatures = np.asarray(table[0], dtype=float)
        values = np.asarray(table[1], dtype=float)
        keep = np.isfinite(temperatures) & np.isfinite(values) & (temperatures > 0.0) & (values > 0.0)
        temperatures = temperatures[keep]
        values = values[keep]
        if len(temperatures) < 2:
            quarantines.append(Quarantine(cas, "janaf_1998", "insufficient positive JANAF points", {}))
            continue

        def evaluate(
            Ts: np.ndarray,
            temperatures: np.ndarray = temperatures,
            values: np.ndarray = values,
        ) -> np.ndarray:
            return np.interp(np.asarray(Ts, dtype=float), temperatures, values)

        name, formula = identity_for_cas(cas)
        candidates.append(SourceCandidate(
            cas, "janaf_1998", cas, f"chemicals {version}; JANAF 1998 snapshot",
            name, formula,
            (SourceSegment(float(temperatures[0]), float(temperatures[-1]), evaluate,
                           {"kind": "JANAF_table", "point_count": len(temperatures)}),),
            hashes["janaf_1998"],
            {
                "point_count": len(temperatures),
                "Tmin": float(temperatures[0]),
                "Tmax": float(temperatures[-1]),
            },
        ))

    for cas, row in heat_capacity.Cp_data_Poling.dropna(subset=["a0"]).sort_index().iterrows():
        cas = str(cas)
        if not valid_cas(cas):
            continue
        coeffs = tuple(float(row[f"a{index}"]) for index in range(5))
        Tmin = finite_float(row["Tmin"])
        Tmax = finite_float(row["Tmax"])
        if Tmin is None or Tmax is None:
            # The only range-free rows are constant monatomic-gas polynomials.
            Tmin, Tmax = 50.0, 5000.0

        def evaluate(Ts: np.ndarray, coeffs: tuple[float, ...] = coeffs) -> np.ndarray:
            return np.asarray([
                heat_capacity.Poling(float(T), *coeffs) for T in Ts
            ], dtype=float)

        name, formula = identity_for_cas(cas, str(row["Chemical"]), "")
        candidates.append(SourceCandidate(
            cas, "poling_2001", cas, f"chemicals {version}; Poling databank",
            name, formula,
            (SourceSegment(Tmin, Tmax, evaluate,
                           {"kind": "Poling", "coefficients": coeffs}),),
            hashes["poling_2001"],
            {
                "coefficients": coeffs,
                "Cpg_298_J_mol_K": finite_float(row["Cpg"]),
                "reported_Tmin": finite_float(row["Tmin"]),
                "reported_Tmax": finite_float(row["Tmax"]),
            },
        ))

    from chemicals.elements import simple_formula_parser
    from chemicals.identifiers import search_chemical

    for cas, thetas_raw in sorted(
        heat_capacity.Cp_dict_characteristic_temperatures_adjusted_psi4_2022a.items()
    ):
        if not valid_cas(cas):
            continue
        thetas = tuple(float(value) for value in thetas_raw)
        try:
            metadata = search_chemical(cas)
            formula = str(metadata.formula)
            atom_count = int(sum(simple_formula_parser(formula).values()))
        except Exception as error:
            quarantines.append(Quarantine(
                cas, "psi4_adjusted", "Psi4 identity/formula unavailable",
                {"error": f"{type(error).__name__}: {error}"},
            ))
            continue
        if len(thetas) == 3 * atom_count - 5:
            linear = True
        elif len(thetas) == 3 * atom_count - 6:
            linear = False
        else:
            quarantines.append(Quarantine(
                cas, "psi4_adjusted", "incomplete Psi4 vibrational mode list",
                {"atom_count": atom_count, "mode_count": len(thetas)},
            ))
            continue

        def evaluate(
            Ts: np.ndarray,
            thetas: tuple[float, ...] = thetas,
            linear: bool = linear,
        ) -> np.ndarray:
            return np.asarray([
                heat_capacity.Cpg_statistical_mechanics(float(T), thetas, linear=linear)
                for T in Ts
            ], dtype=float)

        name, formula = identity_for_cas(cas, str(metadata.common_name), formula)
        candidates.append(SourceCandidate(
            cas, "psi4_adjusted", cas, f"chemicals {version}; adjusted Psi4 2022a",
            name, formula,
            (SourceSegment(50.0, 3000.0, evaluate,
                           {"kind": "RRHO", "linear": linear, "mode_count": len(thetas)}),),
            hashes["psi4_adjusted"],
            {"linear": linear, "mode_count": len(thetas), "characteristic_temperatures_K": thetas},
        ))

    return candidates, quarantines


def source_grid(candidate: SourceCandidate, Tmin: float, points_per_range: int) -> np.ndarray:
    points = []
    for segment in candidate.segments:
        lo = max(float(Tmin), segment.Tmin)
        hi = segment.Tmax
        if not (lo < hi):
            continue
        points.append(np.geomspace(lo, hi, points_per_range))
    if not points:
        return np.asarray([], dtype=float)
    return np.unique(np.concatenate(points))


def rational_map_parameters(Tmin: float, Tmax: float) -> tuple[float, float]:
    center = math.sqrt(Tmin * Tmax)
    scale = (Tmax - center) / (Tmax + center)
    if not (center > 0.0 and 0.0 < scale < 1.0):
        raise ValueError("invalid rational-map parameters")
    return center, scale


def rational_map(T: Sequence[float] | np.ndarray, center: float, scale: float) -> np.ndarray:
    temperatures = np.asarray(T, dtype=float)
    return (temperatures - center) / (scale * (temperatures + center))


def evaluate_cp_fit(fit: CanonicalFit, T: Sequence[float] | np.ndarray) -> np.ndarray:
    x = rational_map(T, fit.center, fit.scale)
    return ncheb.chebval(x, fit.coefficients)


def primitive_terms(
    coefficients: Sequence[float],
    center: float,
    scale: float,
) -> tuple[tuple[float, ...], float, float, tuple[float, ...], float, float]:
    power = np.asarray(ncheb.cheb2poly(np.asarray(coefficients, dtype=float)), dtype=float)

    h_numerator = power * (2.0 * center * scale)
    h_quotient, h_remainder = npoly.polydiv(
        h_numerator,
        np.asarray([1.0, -2.0 * scale, scale * scale]),
    )
    h_remainder = np.pad(h_remainder, (0, max(0, 2 - len(h_remainder))))
    h_A = -h_remainder[1] / scale
    h_B = h_remainder[0] - h_A
    h_polynomial = tuple(float(value) for value in npoly.polyint(h_quotient))
    h_log = float(-h_A / scale)
    h_reciprocal = float(h_B / scale)

    s_numerator = power * (2.0 * scale)
    s_quotient, s_remainder = npoly.polydiv(
        s_numerator,
        np.asarray([1.0, 0.0, -(scale * scale)]),
    )
    s_remainder = np.pad(s_remainder, (0, max(0, 2 - len(s_remainder))))
    s_A = 0.5 * (s_remainder[0] + s_remainder[1] / scale)
    s_B = 0.5 * (s_remainder[0] - s_remainder[1] / scale)
    s_polynomial = tuple(float(value) for value in npoly.polyint(s_quotient))
    s_log_minus = float(-s_A / scale)
    s_log_plus = float(s_B / scale)
    return (
        h_polynomial,
        h_log,
        h_reciprocal,
        s_polynomial,
        s_log_minus,
        s_log_plus,
    )


def evaluate_h_primitive(fit: CanonicalFit, T: float) -> float:
    x = float(rational_map([T], fit.center, fit.scale)[0])
    return (
        float(npoly.polyval(x, fit.h_polynomial))
        + fit.h_log_coefficient * math.log1p(-fit.scale * x)
        + fit.h_reciprocal_coefficient / (1.0 - fit.scale * x)
    )


def evaluate_s_primitive(fit: CanonicalFit, T: float) -> float:
    x = float(rational_map([T], fit.center, fit.scale)[0])
    return (
        float(npoly.polyval(x, fit.s_polynomial))
        + fit.s_log_minus_coefficient * math.log1p(-fit.scale * x)
        + fit.s_log_plus_coefficient * math.log1p(fit.scale * x)
    )


def polynomial_difference(coefficients: Sequence[float], x1: float, x2: float) -> float:
    """Evaluate P(x2)-P(x1) without subtracting two large primitive values."""
    values = tuple(float(value) for value in coefficients)
    if len(values) <= 1 or x1 == x2:
        return 0.0
    # Synthetic division constructs Q(x)=(P(x)-P(x1))/(x-x1).
    quotient = [0.0] * (len(values) - 1)
    quotient[-1] = values[-1]
    for index in range(len(values) - 2, 0, -1):
        quotient[index - 1] = values[index] + x1 * quotient[index]
    return (x2 - x1) * float(npoly.polyval(x2, quotient))


def integrate_h_fit(fit: CanonicalFit, T1: float, T2: float) -> float:
    """Return the stable analytic integral of Cp dT in J/mol."""
    if T1 == T2:
        return 0.0
    if T2 < T1:
        return -integrate_h_fit(fit, T2, T1)
    x1, x2 = rational_map([T1, T2], fit.center, fit.scale)
    delta_x = float(x2 - x1)
    scale = fit.scale
    polynomial = polynomial_difference(fit.h_polynomial, float(x1), float(x2))
    log_difference = math.log1p((-scale * delta_x) / (1.0 - scale * x1))
    reciprocal_difference = (
        scale * delta_x
        / ((1.0 - scale * x2) * (1.0 - scale * x1))
    )
    return (
        polynomial
        + fit.h_log_coefficient * log_difference
        + fit.h_reciprocal_coefficient * reciprocal_difference
    )


def integrate_s_fit(fit: CanonicalFit, T1: float, T2: float) -> float:
    """Return the stable analytic integral of Cp/T dT in J/(mol*K)."""
    if T1 == T2:
        return 0.0
    if T2 < T1:
        return -integrate_s_fit(fit, T2, T1)
    x1, x2 = rational_map([T1, T2], fit.center, fit.scale)
    delta_x = float(x2 - x1)
    scale = fit.scale
    polynomial = polynomial_difference(fit.s_polynomial, float(x1), float(x2))
    minus_log_difference = math.log1p((-scale * delta_x) / (1.0 - scale * x1))
    plus_log_difference = math.log1p((scale * delta_x) / (1.0 + scale * x1))
    return (
        polynomial
        + fit.s_log_minus_coefficient * minus_log_difference
        + fit.s_log_plus_coefficient * plus_log_difference
    )


def fit_coefficients(
    candidate: SourceCandidate,
    Tmin: float,
    degree: int,
) -> Optional[CanonicalFit]:
    Tmax = candidate.Tmax
    training_T = source_grid(candidate, Tmin, TRAINING_POINTS_PER_RANGE)
    validation_T = source_grid(candidate, Tmin, VALIDATION_POINTS_PER_RANGE)
    if len(training_T) < degree + 1 or len(validation_T) < degree + 1:
        return None
    training_values = candidate.values(training_T)
    validation_values = candidate.values(validation_T)
    if (
        np.any(~np.isfinite(training_values))
        or np.any(training_values <= 0.0)
        or np.any(~np.isfinite(validation_values))
        or np.any(validation_values <= 0.0)
    ):
        return None
    center, scale = rational_map_parameters(Tmin, Tmax)
    x_training = rational_map(training_T, center, scale)
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
    predicted = ncheb.chebval(rational_map(validation_T, center, scale), coefficients)
    if np.any(~np.isfinite(predicted)) or np.any(predicted <= 0.0):
        return None
    # Perry can contain a small unsupported handoff gap between its low-T
    # polynomial and high-T hyperbolic row.  The canonical curve is deliberately
    # contiguous, so validate finite positive behavior through the whole span,
    # not only where source samples exist.
    continuous_T = np.geomspace(Tmin, Tmax, VALIDATION_POINTS_PER_RANGE)
    continuous_values = ncheb.chebval(
        rational_map(continuous_T, center, scale), coefficients,
    )
    if np.any(~np.isfinite(continuous_values)) or np.any(continuous_values <= 0.0):
        return None
    errors = np.abs(predicted / validation_values - 1.0) * 100.0
    primitives = primitive_terms(coefficients, center, scale)
    provisional = CanonicalFit(
        Tmin=Tmin,
        Tmax=Tmax,
        degree=degree,
        center=center,
        scale=scale,
        coefficients=tuple(float(value) for value in coefficients),
        fit_mape_percent=float(np.mean(errors)),
        fit_max_error_percent=float(np.max(errors)),
        fit_h_max_error_percent=math.inf,
        fit_s_max_error_percent=math.inf,
        validation_points=len(validation_T),
        h_polynomial=primitives[0],
        h_log_coefficient=primitives[1],
        h_reciprocal_coefficient=primitives[2],
        s_polynomial=primitives[3],
        s_log_minus_coefficient=primitives[4],
        s_log_plus_coefficient=primitives[5],
    )
    h_errors, s_errors = validate_integrals(candidate, provisional)
    return CanonicalFit(
        **{
            **provisional.__dict__,
            "fit_h_max_error_percent": h_errors,
            "fit_s_max_error_percent": s_errors,
        }
    )


def validate_integrals(candidate: SourceCandidate, fit: CanonicalFit) -> tuple[float, float]:
    h_errors = []
    s_errors = []
    for segment in candidate.segments:
        lo = max(fit.Tmin, segment.Tmin)
        hi = min(fit.Tmax, segment.Tmax)
        if not lo < hi:
            continue
        dense_T = np.geomspace(lo, hi, 1601)
        source = segment.evaluate_vector(dense_T)
        fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
        indexes = sorted({int(round(fraction * (len(dense_T) - 1))) for fraction in fractions})
        for start_index in range(len(indexes) - 1):
            for end_index in range(start_index + 1, len(indexes)):
                left = indexes[start_index]
                right = indexes[end_index]
                if right <= left:
                    continue
                source_H = float(np.trapezoid(source[left:right + 1], dense_T[left:right + 1]))
                source_S = float(np.trapezoid(
                    source[left:right + 1] / dense_T[left:right + 1],
                    dense_T[left:right + 1],
                ))
                fit_H = integrate_h_fit(fit, float(dense_T[left]), float(dense_T[right]))
                fit_S = integrate_s_fit(fit, float(dense_T[left]), float(dense_T[right]))
                if abs(source_H) > 1e-12:
                    h_errors.append(abs(fit_H / source_H - 1.0) * 100.0)
                if abs(source_S) > 1e-12:
                    s_errors.append(abs(fit_S / source_S - 1.0) * 100.0)
    return max(h_errors, default=0.0), max(s_errors, default=0.0)


def Tmin_candidates(candidate: SourceCandidate) -> list[float]:
    native = candidate.Tmin
    upper = min(FIT_TMIN_RAISE_LIMIT_K, candidate.Tmax - 1.0)
    values = [native]
    if upper > native:
        count = int(math.floor((upper - native) / FIT_TMIN_STEP_K))
        values.extend(native + FIT_TMIN_STEP_K * index for index in range(1, count + 1))
        values.append(upper)
    return sorted({float(value) for value in values if value < candidate.Tmax})


def canonicalize(candidate: SourceCandidate) -> Optional[CanonicalFit]:
    best_fit = None
    for Tmin in Tmin_candidates(candidate):
        for degree in FIT_DEGREES:
            fit = fit_coefficients(candidate, Tmin, degree)
            if fit is None:
                continue
            if best_fit is None or fit.fit_max_error_percent < best_fit.fit_max_error_percent:
                best_fit = fit
            if fit.fit_max_error_percent <= FIT_MAX_RELATIVE_ERROR_PERCENT + 1e-10:
                return fit
    # A fit above the target remains useful.  Its maximum relative fitting
    # error is converted directly into a quality penalty instead of discarding
    # the only available source for a CAS identity.
    return best_fit


def physical_heat_capacity_floor(candidate: SourceCandidate) -> tuple[float, str]:
    try:
        from chemicals.elements import simple_formula_parser
        counts = simple_formula_parser(candidate.formula)
        if not counts.get("C") or not counts.get("H"):
            return 0.0, "not an organic molecular compound"
    except Exception:
        return 0.0, "identity unavailable"
    try:
        from chemicals.identifiers import search_chemical
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        molecule = Chem.MolFromSmiles(str(search_chemical(candidate.cas).smiles or ""))
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            return 0.0, "structure unavailable or disconnected"
        if any(atom.GetFormalCharge() != 0 for atom in molecule.GetAtoms()):
            return 0.0, "formally charged structure"
        heavy_atoms = molecule.GetNumHeavyAtoms()
        if heavy_atoms <= 3:
            return 0.0, "small organic molecule"
        definitely_nonlinear = (
            molecule.GetRingInfo().NumRings() > 0
            or any(atom.GetDegree() >= 3 for atom in molecule.GetAtoms())
        )
        if definitely_nonlinear:
            return 4.0 * R, "neutral nonlinear organic rigid-rotor/translational floor"
        return 3.5 * R, "neutral possibly-linear organic rigid-rotor/translational floor"
    except Exception:
        return 0.0, "structure validation unavailable"


def validate_source(candidate: SourceCandidate) -> tuple[bool, str, dict[str, Any]]:
    temperatures = source_grid(candidate, candidate.Tmin, 401)
    values = candidate.values(temperatures)
    if len(values) == 0:
        return False, "empty source range", {}
    if np.any(~np.isfinite(values)):
        return False, "nonfinite heat capacity in source range", {}
    if np.any(values <= 0.0):
        index = int(np.argmin(values))
        return False, "nonpositive heat capacity in source range", {
            "T_K": float(temperatures[index]), "Cp_J_mol_K": float(values[index]),
        }
    floor, floor_basis = physical_heat_capacity_floor(candidate)
    floor_mask = temperatures >= max(250.0, candidate.Tmin)
    if floor > 0.0 and np.any(floor_mask):
        admitted_values = values[floor_mask]
        if float(np.min(admitted_values)) < 0.98 * floor:
            local_index = int(np.argmin(admitted_values))
            admitted_T = temperatures[floor_mask]
            return False, "source violates conservative ideal-gas heat-capacity floor", {
                "T_K": float(admitted_T[local_index]),
                "Cp_J_mol_K": float(admitted_values[local_index]),
                "floor_J_mol_K": floor,
                "floor_basis": floor_basis,
            }
    return True, "", {
        "minimum_Cp_J_mol_K": float(np.min(values)),
        "maximum_Cp_J_mol_K": float(np.max(values)),
        "validation_points": len(values),
    }


def comparison_temperatures(first: SourceCandidate, second: SourceCandidate) -> np.ndarray:
    grids = []
    for first_segment in first.segments:
        for second_segment in second.segments:
            lo = max(first_segment.Tmin, second_segment.Tmin)
            hi = min(first_segment.Tmax, second_segment.Tmax)
            if lo < hi:
                grids.append(np.geomspace(lo, hi, COMPARISON_POINTS_PER_RANGE))
    return np.unique(np.concatenate(grids)) if grids else np.asarray([], dtype=float)


def compare_sources(first: SourceCandidate, second: SourceCandidate) -> Optional[dict[str, Any]]:
    temperatures = comparison_temperatures(first, second)
    if len(temperatures) < 5:
        return None
    first_values = first.values(temperatures)
    second_values = second.values(temperatures)
    keep = (
        np.isfinite(first_values) & np.isfinite(second_values)
        & (first_values > 0.0) & (second_values > 0.0)
    )
    if np.count_nonzero(keep) < 5:
        return None
    temperatures = temperatures[keep]
    errors = np.abs(first_values[keep] / second_values[keep] - 1.0) * 100.0
    return {
        "point_count": len(errors),
        "Tmin": float(np.min(temperatures)),
        "Tmax": float(np.max(temperatures)),
        "median_error_percent": float(np.median(errors)),
        "p95_error_percent": float(np.percentile(errors, 95.0)),
        "max_error_percent": float(np.max(errors)),
        "close": bool(np.median(errors) <= 0.5 and np.percentile(errors, 95.0) <= 1.5),
        "strong_conflict": bool(np.median(errors) > 10.0 and np.percentile(errors, 95.0) > 10.0),
    }


def obvious_cross_source_outlier(
    candidate: SourceCandidate,
    alternatives: Sequence[SourceCandidate],
) -> tuple[bool, dict[str, Any]]:
    conflicting = []
    for alternative in alternatives:
        if alternative.lineage == candidate.lineage:
            continue
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


def validate_poling_self_consistency(
    candidate: SourceCandidate,
    higher_candidates: Sequence[SourceCandidate],
) -> tuple[bool, str, dict[str, Any]]:
    constant = finite_float(candidate.details.get("Cpg_298_J_mol_K"))
    if constant is None or not (candidate.Tmin <= 298.15 <= candidate.Tmax):
        return True, "", {}
    polynomial = float(candidate.values([298.15])[0])
    difference = abs(polynomial / constant - 1.0) * 100.0
    if difference <= 5.0:
        return True, "", {"polynomial_vs_constant_percent": difference}
    comparisons = []
    for higher in higher_candidates:
        reference = float(higher.values([298.15])[0])
        if not math.isfinite(reference) or reference <= 0.0:
            continue
        comparisons.append({
            "source": higher.source,
            "polynomial_error_percent": abs(polynomial / reference - 1.0) * 100.0,
            "constant_error_percent": abs(constant / reference - 1.0) * 100.0,
        })
    if any(
        item["polynomial_error_percent"] <= 3.0 and item["constant_error_percent"] > 10.0
        for item in comparisons
    ):
        candidate.notes.append(
            "Poling polynomial retained: a higher-priority curve corroborates it while the companion 298 K constant conflicts"
        )
        return True, "", {
            "polynomial_vs_constant_percent": difference,
            "higher_source_comparisons": comparisons,
        }
    return False, "Poling polynomial conflicts with companion 298 K value", {
        "polynomial_Cp_298": polynomial,
        "constant_Cp_298": constant,
        "difference_percent": difference,
        "higher_source_comparisons": comparisons,
    }


def crosschecks_for_fit(
    selected: SourceCandidate,
    fit: CanonicalFit,
    alternatives: Sequence[SourceCandidate],
) -> tuple[Crosscheck, ...]:
    result = []
    for alternative in alternatives:
        temperatures = []
        for segment in alternative.segments:
            lo = max(fit.Tmin, segment.Tmin)
            hi = min(fit.Tmax, segment.Tmax)
            if lo < hi:
                temperatures.append(np.geomspace(lo, hi, COMPARISON_POINTS_PER_RANGE))
        if not temperatures:
            continue
        values_T = np.unique(np.concatenate(temperatures))
        alternative_values = alternative.values(values_T)
        selected_values = evaluate_cp_fit(fit, values_T)
        keep = (
            np.isfinite(alternative_values) & np.isfinite(selected_values)
            & (alternative_values > 0.0) & (selected_values > 0.0)
        )
        if np.count_nonzero(keep) < 5:
            continue
        values_T = values_T[keep]
        errors = np.abs(selected_values[keep] / alternative_values[keep] - 1.0) * 100.0
        median = float(np.median(errors))
        p95 = float(np.percentile(errors, 95.0))
        close = median <= 0.5 and p95 <= 1.5
        result.append(Crosscheck(
            selected.cas,
            selected.source,
            alternative.source,
            selected.lineage,
            alternative.lineage,
            len(errors),
            float(np.min(values_T)),
            float(np.max(values_T)),
            median,
            p95,
            float(np.max(errors)),
            close,
            alternative.lineage != selected.lineage,
        ))
    return tuple(sorted(result, key=lambda item: SOURCE_SPECS[item.comparison_source]["rank"]))


def quality_with_corroboration(
    candidate: SourceCandidate,
    fit: CanonicalFit,
    checks: Sequence[Crosscheck],
) -> tuple[float, float, float]:
    independent_lineages = {
        check.comparison_lineage
        for check in checks
        if check.close_agreement and check.independent
    }
    bonus = min(0.015, 0.005 * len(independent_lineages))
    fit_penalty = fit_quality_penalty(
        fit.fit_max_error_percent, fit.fit_mape_percent,
    )
    quality = max(0.0, min(0.995, candidate.base_quality + bonus) - fit_penalty)
    return quality, bonus, fit_penalty


def fit_quality_penalty(max_error_percent: float, mape_percent: float) -> float:
    if (
        max_error_percent < FIT_ZERO_PENALTY_MAX_ERROR_PERCENT
        and mape_percent < FIT_ZERO_PENALTY_MAPE_PERCENT
    ):
        return 0.0
    return (
        FIT_MAX_ERROR_QUALITY_WEIGHT * max_error_percent
        + FIT_MAPE_QUALITY_WEIGHT * mape_percent
    )


def compile_records() -> tuple[
    list[SelectedRecord],
    list[SourceCandidate],
    list[Crosscheck],
    list[Quarantine],
    dict[str, Any],
]:
    candidates = collect_perry_candidates()
    coolprop, quarantines = collect_coolprop_candidates()
    chemicals_candidates, chemicals_quarantines = collect_chemicals_candidates()
    candidates.extend(coolprop)
    candidates.extend(chemicals_candidates)
    quarantines.extend(chemicals_quarantines)

    candidates.sort(key=lambda item: (item.cas, item.rank, item.source_key))
    by_cas: dict[str, list[SourceCandidate]] = {}
    validation_details: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        by_cas.setdefault(candidate.cas, []).append(candidate)
        valid, reason, details = validate_source(candidate)
        validation_details[(candidate.cas, candidate.source)] = details
        if not valid:
            candidate.status = "invalid"
            candidate.reason = reason
            quarantines.append(Quarantine(candidate.cas, candidate.source, reason, details))

    # Poling has a companion 298 K point that catches coefficient transcription
    # failures, but the point itself is occasionally the corrupt field.  A
    # higher-priority curve arbitrates those disagreements when one exists.
    for cas, cas_candidates in by_cas.items():
        valid_higher = [item for item in cas_candidates if item.status == "available"]
        for candidate in cas_candidates:
            if candidate.source != "poling_2001" or candidate.status != "available":
                continue
            higher = [
                item for item in valid_higher
                if item.rank < candidate.rank and item.Tmin <= 298.15 <= item.Tmax
            ]
            valid, reason, details = validate_poling_self_consistency(candidate, higher)
            validation_details[(candidate.cas, candidate.source)]["poling_self_check"] = details
            if not valid:
                candidate.status = "invalid"
                candidate.reason = reason
                quarantines.append(Quarantine(candidate.cas, candidate.source, reason, details))

    selected_records = []
    all_crosschecks = []
    unavailable = []
    for cas, cas_candidates in sorted(by_cas.items()):
        selected_candidate = None
        selected_fit = None
        for candidate in sorted(cas_candidates, key=lambda item: item.rank):
            if candidate.status != "available":
                continue
            alternatives = [
                item for item in cas_candidates
                if item is not candidate and item.status == "available"
            ]
            outlier, details = obvious_cross_source_outlier(candidate, alternatives)
            if outlier:
                candidate.status = "outlier"
                candidate.reason = "source is an obvious cross-source outlier"
                quarantines.append(Quarantine(candidate.cas, candidate.source, candidate.reason, details))
                continue
            fit = canonicalize(candidate)
            if fit is None:
                candidate.status = "fit_failed"
                candidate.reason = "degree 8/12 fit could not produce a finite positive canonical curve"
                quarantines.append(Quarantine(
                    candidate.cas, candidate.source, candidate.reason,
                    {"Tmin": candidate.Tmin, "Tmax": candidate.Tmax},
                ))
                continue
            candidate.status = "selected"
            selected_candidate = candidate
            selected_fit = fit
            break
        if selected_candidate is None or selected_fit is None:
            unavailable.append(cas)
            continue
        for candidate in cas_candidates:
            if candidate.status == "available":
                candidate.status = "not_selected"
                candidate.reason = "lower source priority"
        alternatives = [
            item for item in cas_candidates
            if item is not selected_candidate and item.status == "not_selected"
        ]
        checks = crosschecks_for_fit(selected_candidate, selected_fit, alternatives)
        quality, bonus, fit_penalty = quality_with_corroboration(
            selected_candidate, selected_fit, checks,
        )
        selected_records.append(SelectedRecord(
            selected_candidate, selected_fit, quality, bonus, fit_penalty, checks,
        ))
        all_crosschecks.extend(checks)

    report = {
        "candidate_count": len(candidates),
        "unique_cas_count": len(by_cas),
        "selected_count": len(selected_records),
        "unavailable_count": len(unavailable),
        "unavailable_cas": unavailable,
        "quarantine_count": len(quarantines),
        "selected_by_source": dict(Counter(record.candidate.source for record in selected_records)),
        "selected_by_degree": dict(Counter(str(record.fit.degree) for record in selected_records)),
        "raised_Tmin_count": sum(record.fit.Tmin > record.candidate.Tmin + 1e-9 for record in selected_records),
        "corroborated_count": sum(record.corroboration_bonus > 0.0 for record in selected_records),
        "fit_max_error_percent": max(
            (record.fit.fit_max_error_percent for record in selected_records), default=0.0
        ),
        "fit_above_target_count": sum(
            record.fit.fit_max_error_percent > FIT_MAX_RELATIVE_ERROR_PERCENT + 1e-10
            for record in selected_records
        ),
        "fit_max_h_integral_error_percent": max(
            (record.fit.fit_h_max_error_percent for record in selected_records), default=0.0
        ),
        "fit_max_s_integral_error_percent": max(
            (record.fit.fit_s_max_error_percent for record in selected_records), default=0.0
        ),
        "validation_details": validation_details,
    }
    return selected_records, candidates, all_crosschecks, quarantines, report


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        PRAGMA user_version = 3;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );

        CREATE TABLE canonical_ideal_gas_cp (
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
            corroboration_bonus REAL NOT NULL,
            fit_quality_penalty REAL NOT NULL CHECK (fit_quality_penalty >= 0.0),
            Tmin_source_K REAL NOT NULL,
            Tmax_source_K REAL NOT NULL,
            Tmin_fit_K REAL NOT NULL,
            Tmax_fit_K REAL NOT NULL,
            Tmin_raised_K REAL NOT NULL,
            degree INTEGER NOT NULL CHECK (degree IN (8, 12)),
            map_center_K REAL NOT NULL,
            map_scale REAL NOT NULL,
            cp_coefficients_json TEXT NOT NULL,
            h_polynomial_json TEXT NOT NULL,
            h_log_coefficient REAL NOT NULL,
            h_reciprocal_coefficient REAL NOT NULL,
            s_polynomial_json TEXT NOT NULL,
            s_log_minus_coefficient REAL NOT NULL,
            s_log_plus_coefficient REAL NOT NULL,
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
            details_json TEXT NOT NULL,
            notes_json TEXT NOT NULL,
            PRIMARY KEY (cas, source)
        );

        CREATE TABLE source_crosscheck (
            cas TEXT NOT NULL,
            selected_source TEXT NOT NULL,
            comparison_source TEXT NOT NULL,
            selected_lineage TEXT NOT NULL,
            comparison_lineage TEXT NOT NULL,
            point_count INTEGER NOT NULL,
            Tmin_K REAL NOT NULL,
            Tmax_K REAL NOT NULL,
            median_error_percent REAL NOT NULL,
            p95_error_percent REAL NOT NULL,
            max_error_percent REAL NOT NULL,
            close_agreement INTEGER NOT NULL,
            independent INTEGER NOT NULL,
            PRIMARY KEY (cas, selected_source, comparison_source)
        );

        CREATE TABLE source_quarantine (
            id INTEGER PRIMARY KEY,
            cas TEXT NOT NULL,
            source TEXT NOT NULL,
            reason TEXT NOT NULL,
            details_json TEXT NOT NULL
        );

        CREATE INDEX idx_canonical_ideal_gas_cp_source
        ON canonical_ideal_gas_cp (source);
        CREATE INDEX idx_canonical_ideal_gas_cp_quality
        ON canonical_ideal_gas_cp (quality);
        CREATE INDEX idx_source_candidate_audit_status
        ON source_candidate_audit (status);
        CREATE INDEX idx_source_crosscheck_close
        ON source_crosscheck (close_agreement, independent);
        CREATE INDEX idx_source_quarantine_cas
        ON source_quarantine (cas);
        """
    )


def write_database(
    output: Path,
    selected: Sequence[SelectedRecord],
    candidates: Sequence[SourceCandidate],
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
    try:
        with sqlite3.connect(temporary) as connection:
            create_schema(connection)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "model": MODEL_NAME,
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "fit_degrees": FIT_DEGREES,
                "fit_max_relative_error_percent": FIT_MAX_RELATIVE_ERROR_PERCENT,
                "fit_zero_penalty_max_error_percent": FIT_ZERO_PENALTY_MAX_ERROR_PERCENT,
                "fit_zero_penalty_mape_percent": FIT_ZERO_PENALTY_MAPE_PERCENT,
                "fit_max_error_quality_weight": FIT_MAX_ERROR_QUALITY_WEIGHT,
                "fit_mape_quality_weight": FIT_MAPE_QUALITY_WEIGHT,
                "fit_Tmin_step_K": FIT_TMIN_STEP_K,
                "fit_Tmin_raise_limit_K": FIT_TMIN_RAISE_LIMIT_K,
                "source_priority": [
                    source for source, _spec in sorted(
                        SOURCE_SPECS.items(), key=lambda item: item[1]["rank"]
                    )
                ],
                "source_specs": SOURCE_SPECS,
                "report": {key: value for key, value in report.items() if key != "validation_details"},
            }
            connection.executemany(
                "INSERT INTO metadata (key, value_json) VALUES (?, ?)",
                [(key, canonical_json(value)) for key, value in metadata.items()],
            )
            for record in selected:
                candidate = record.candidate
                fit = record.fit
                independent_lineages = {
                    check.comparison_lineage
                    for check in record.crosschecks
                    if check.close_agreement and check.independent
                }
                connection.execute(
                    """
                    INSERT INTO canonical_ideal_gas_cp (
                        cas, name, formula, model, source, source_label, source_rank,
                        source_key, source_version, source_lineage, source_fingerprint,
                        quality, base_quality, corroboration_bonus,
                        fit_quality_penalty,
                        Tmin_source_K, Tmax_source_K, Tmin_fit_K, Tmax_fit_K, Tmin_raised_K,
                        degree, map_center_K, map_scale, cp_coefficients_json,
                        h_polynomial_json, h_log_coefficient, h_reciprocal_coefficient,
                        s_polynomial_json, s_log_minus_coefficient, s_log_plus_coefficient,
                        fit_mape_percent, fit_max_error_percent,
                        fit_h_max_error_percent, fit_s_max_error_percent,
                        validation_points, corroboration_count, independent_corroboration_count,
                        corroboration_json, source_details_json, notes_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        candidate.cas, candidate.name, candidate.formula, MODEL_NAME,
                        candidate.source, candidate.label, candidate.rank,
                        candidate.source_key, candidate.source_version, candidate.lineage,
                        candidate.fingerprint, record.quality, candidate.base_quality,
                        record.corroboration_bonus, record.fit_quality_penalty,
                        candidate.Tmin, candidate.Tmax,
                        fit.Tmin, fit.Tmax, fit.Tmin - candidate.Tmin, fit.degree,
                        fit.center, fit.scale, canonical_json(fit.coefficients),
                        canonical_json(fit.h_polynomial), fit.h_log_coefficient,
                        fit.h_reciprocal_coefficient, canonical_json(fit.s_polynomial),
                        fit.s_log_minus_coefficient, fit.s_log_plus_coefficient,
                        fit.fit_mape_percent, fit.fit_max_error_percent,
                        fit.fit_h_max_error_percent, fit.fit_s_max_error_percent,
                        fit.validation_points,
                        sum(check.close_agreement for check in record.crosschecks),
                        len(independent_lineages),
                        canonical_json([check.__dict__ for check in record.crosschecks]),
                        canonical_json(candidate.details), canonical_json(candidate.notes),
                    ),
                )
            connection.executemany(
                """
                INSERT INTO source_candidate_audit (
                    cas, source, source_label, source_rank, source_key, source_version,
                    source_lineage, source_fingerprint, status, reason, Tmin_K, Tmax_K,
                    base_quality, details_json, notes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        candidate.cas, candidate.source, candidate.label, candidate.rank,
                        candidate.source_key, candidate.source_version, candidate.lineage,
                        candidate.fingerprint, candidate.status, candidate.reason,
                        candidate.Tmin, candidate.Tmax, candidate.base_quality,
                        canonical_json(candidate.details), canonical_json(candidate.notes),
                    )
                    for candidate in candidates
                ],
            )
            connection.executemany(
                """
                INSERT INTO source_crosscheck (
                    cas, selected_source, comparison_source, selected_lineage,
                    comparison_lineage, point_count, Tmin_K, Tmax_K,
                    median_error_percent, p95_error_percent, max_error_percent,
                    close_agreement, independent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.cas, item.selected_source, item.comparison_source,
                        item.selected_lineage, item.comparison_lineage, item.point_count,
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
                    (item.cas, item.source, item.reason, canonical_json(item.details))
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
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"database integrity check failed: {integrity}")
        rows = connection.execute("SELECT * FROM canonical_ideal_gas_cp ORDER BY cas").fetchall()
        if not rows:
            raise RuntimeError("canonical database contains no rows")
        maximum_primitive_error = 0.0
        for row in rows:
            expected_penalty = fit_quality_penalty(
                row["fit_max_error_percent"], row["fit_mape_percent"],
            )
            expected_quality = max(
                0.0,
                min(0.995, row["base_quality"] + row["corroboration_bonus"])
                - expected_penalty,
            )
            if abs(row["fit_quality_penalty"] - expected_penalty) > 1e-12:
                raise RuntimeError(f"fit quality penalty mismatch for {row['cas']}")
            if abs(row["quality"] - expected_quality) > 1e-12:
                raise RuntimeError(f"quality mismatch for {row['cas']}")
            if (
                row["Tmin_source_K"] < FIT_TMIN_RAISE_LIMIT_K
                and row["Tmin_fit_K"] > FIT_TMIN_RAISE_LIMIT_K + 1e-10
            ):
                raise RuntimeError(f"raised Tmin exceeds 0 C for {row['cas']}")
            coefficients = tuple(json.loads(row["cp_coefficients_json"]))
            h_polynomial = tuple(json.loads(row["h_polynomial_json"]))
            s_polynomial = tuple(json.loads(row["s_polynomial_json"]))
            if len(coefficients) != row["degree"] + 1:
                raise RuntimeError(f"coefficient count mismatch for {row['cas']}")
            fit = CanonicalFit(
                row["Tmin_fit_K"], row["Tmax_fit_K"], row["degree"],
                row["map_center_K"], row["map_scale"], coefficients,
                row["fit_mape_percent"], row["fit_max_error_percent"],
                row["fit_h_max_error_percent"], row["fit_s_max_error_percent"],
                row["validation_points"], h_polynomial,
                row["h_log_coefficient"], row["h_reciprocal_coefficient"],
                s_polynomial, row["s_log_minus_coefficient"], row["s_log_plus_coefficient"],
            )
            temperatures = np.geomspace(fit.Tmin, fit.Tmax, 33)
            cp = evaluate_cp_fit(fit, temperatures)
            if np.any(~np.isfinite(cp)) or np.any(cp <= 0.0):
                raise RuntimeError(f"invalid stored heat-capacity curve for {row['cas']}")
            for first, second in zip(temperatures[:-1], temperatures[1:]):
                midpoint = math.sqrt(first * second)
                delta = max(1e-4, midpoint * 1e-6)
                numerical_cp = integrate_h_fit(fit, midpoint - delta, midpoint + delta) / (2.0 * delta)
                reference_cp = float(evaluate_cp_fit(fit, [midpoint])[0])
                primitive_error = abs(numerical_cp / reference_cp - 1.0)
                maximum_primitive_error = max(maximum_primitive_error, primitive_error)
                numerical_cp_over_T = integrate_s_fit(
                    fit, midpoint - delta, midpoint + delta
                ) / (2.0 * delta)
                entropy_error = abs(numerical_cp_over_T / (reference_cp / midpoint) - 1.0)
                maximum_primitive_error = max(maximum_primitive_error, entropy_error)
        return {
            "integrity": integrity,
            "row_count": len(rows),
            "maximum_primitive_derivative_relative_error": maximum_primitive_error,
            "quarantine_count": connection.execute(
                "SELECT count(*) FROM source_quarantine"
            ).fetchone()[0],
            "crosscheck_count": connection.execute(
                "SELECT count(*) FROM source_crosscheck"
            ).fetchone()[0],
        }


def self_test() -> None:
    Tmin, Tmax = 100.0, 1500.0
    center, scale = rational_map_parameters(Tmin, Tmax)
    endpoints = rational_map([Tmin, Tmax], center, scale)
    if not np.allclose(endpoints, [-1.0, 1.0], rtol=0.0, atol=1e-13):
        raise AssertionError(f"rational map endpoint failure: {endpoints}")

    exact_coefficients = np.asarray([80.0, 22.0, 3.0, -1.0, 0.25])

    def exact_synthetic(Ts: np.ndarray) -> np.ndarray:
        return ncheb.chebval(rational_map(Ts, center, scale), exact_coefficients)

    candidate = SourceCandidate(
        "999-99-9", "trc_1994", "synthetic", "self-test", "synthetic", "C2H6",
        (SourceSegment(Tmin, Tmax, exact_synthetic),), "synthetic", {},
    )
    fit = fit_coefficients(candidate, Tmin, 8)
    if fit is None or fit.fit_max_error_percent > 1e-8:
        raise AssertionError("exact synthetic canonical fit failed")

    def realistic_synthetic(Ts: np.ndarray) -> np.ndarray:
        Ts = np.asarray(Ts, dtype=float)
        return 25.0 + 0.08 * Ts + 12000.0 / (Ts * Ts)

    realistic_candidate = SourceCandidate(
        "999-99-8", "trc_1994", "realistic", "self-test", "realistic", "C2H6",
        (SourceSegment(Tmin, Tmax, realistic_synthetic),), "synthetic", {},
    )
    realistic_fit = fit_coefficients(realistic_candidate, Tmin, 8)
    if realistic_fit is None or realistic_fit.fit_max_error_percent > FIT_MAX_RELATIVE_ERROR_PERCENT:
        raise AssertionError("realistic synthetic fit contract failed")
    if fit_quality_penalty(0.099, 0.0099) != 0.0:
        raise AssertionError("fit quality exemption failed")
    expected_penalty = 0.031 * 0.5 + 0.185 * 0.02
    if abs(fit_quality_penalty(0.5, 0.02) - expected_penalty) > 1e-15:
        raise AssertionError("dual-error fit quality penalty failed")

    left, right = 250.0, 900.0
    grid = np.linspace(left, right, 20001)
    numerical_H = float(np.trapezoid(exact_synthetic(grid), grid))
    numerical_S = float(np.trapezoid(exact_synthetic(grid) / grid, grid))
    fitted_H = integrate_h_fit(fit, left, right)
    fitted_S = integrate_s_fit(fit, left, right)
    if abs(fitted_H / numerical_H - 1.0) > 1e-8:
        raise AssertionError("enthalpy primitive self-test failed")
    if abs(fitted_S / numerical_S - 1.0) > 1e-8:
        raise AssertionError("entropy primitive self-test failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    self_test()
    if args.validate_only:
        result = validate_database(args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    selected, candidates, crosschecks, quarantines, report = compile_records()
    backup = write_database(
        args.output, selected, candidates, crosschecks, quarantines, report,
    )
    validation = validate_database(args.output)
    summary = {
        **{key: value for key, value in report.items() if key != "validation_details"},
        "output": str(args.output),
        "backup": str(backup) if backup else None,
        "database_validation": validation,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
