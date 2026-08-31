#!/usr/bin/env python3
"""Build the runtime Henry database from Sander's official 5.0.0 archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import os
import re
import sqlite3
import tempfile
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_DATA = DATA / "source"
SOURCE = SOURCE_DATA / "henry_5.0.0_sql.zip"
OUTPUT = DATA / "henry_constants.sqlite"
TABLE_NAME = "henry_constants"
SOURCE_VALUES_TABLE = "henry_source_values"
METADATA_TABLE = "henry_metadata"
PARTIAL_MOLAR_VOLUME_TABLE = "henry_partial_molar_volumes"
TEMPERATURE_CORRELATION_TABLE = "henry_temperature_correlations"
TEMPERATURE_SOURCE_TABLE = "henry_temperature_source_correlations"
TEMPERATURE_REJECTION_TABLE = "henry_temperature_correlation_rejections"
BROCKBANK_TABLE_C2 = SOURCE_DATA / "henry" / "brockbank2013_table_c2.csv"

SOURCE_VERSION = "5.0.0"
SOURCE_URL = "https://www.henrys-law.org/henry_data/henry_5.0.0_sql.zip"
SOURCE_DOI = "10.5194/acp-23-10901-2023"
HENRY_REFERENCE_TEMPERATURE_K = 298.15
SOURCE_ARCHIVE_SHA256 = "d90b7264a55ebb2c42fcca4b3483d0cc5361caa177224685cc66f3f7d7fb9d1b"
EXPECTED_HENRY_ROWS = 46_434
EXPECTED_SPECIES_ROWS = 10_173
EXPECTED_SEAWATER_ROWS = 98
EXPECTED_ELIGIBLE_THREE_PARAMETER_ROWS = 238
TEMPERATURE_VALUE_COMPATIBILITY_PERCENT = 25.0
TEMPERATURE_SLOPE_COMPATIBILITY_FRACTION = 0.25
TEMPERATURE_SLOPE_COMPATIBILITY_MIN_K = 500.0
WATER_CONCENTRATION_REF_MOL_M3 = 55344.17427375269
WATER_DLN_CONCENTRATION_DINVT_REF_K = 23.020784850829262

VINF_SOURCE = "Zhou and Battino (2001)"
VINF_DOI = "10.1021/je000215o"
# Infinite-dilution partial molar volumes in water at 298.15 K [cm3/mol].
VINF_298_RECORDS = (
    ("1333-74-0", "hydrogen", 23.1, 1.1),
    ("7782-44-7", "oxygen", 32.1, 0.1),
    ("7727-37-9", "nitrogen", 33.1, 1.6),
    ("7440-37-1", "argon", 32.7, 0.4),
    ("7440-59-7", "helium", 24.6, 3.0),
    ("74-82-8", "methane", 32.0, 2.0),
    ("74-84-0", "ethane", 49.6, 1.7),
    ("74-85-1", "ethylene", 45.4, 1.3),
    ("74-98-6", "propane", 75.0, 3.1),
    ("106-97-8", "n-butane", 74.5, 2.0),
    ("75-28-5", "isobutane", 74.9, 0.5),
    ("75-71-8", "dichlorodifluoromethane", 88.3, 3.1),
    ("75-72-9", "chlorotrifluoromethane", 80.7, 3.0),
)

# IAPWS G7-04 wide-range pure-water Henry volatility correlations. Values are
# (CAS, component, A, B, C, Tmin_K, Tmax_K, RMS deviation in ln(kH)).
IAPWS_G7_04_RECORDS = (
    ("7440-59-7", "helium", -3.52839, 7.12983, 4.47770, 273.21, 553.18, 0.0341),
    ("7440-01-9", "neon", -3.18301, 5.31448, 5.43774, 273.20, 543.36, 0.0577),
    ("7440-37-1", "argon", -8.40954, 4.29587, 10.52779, 273.19, 568.36, 0.0443),
    ("7439-90-9", "krypton", -8.97358, 3.61508, 11.29963, 273.19, 525.56, 0.0434),
    ("7440-63-3", "xenon", -14.21635, 4.00041, 15.60999, 273.22, 574.85, 0.0363),
    ("1333-74-0", "hydrogen", -4.73284, 6.08954, 6.06066, 273.15, 636.09, 0.0517),
    ("7727-37-9", "nitrogen", -9.67578, 4.72162, 11.70585, 278.12, 636.46, 0.0372),
    ("7782-44-7", "oxygen", -9.44833, 4.43822, 11.42005, 274.15, 616.52, 0.0377),
    ("630-08-0", "carbon monoxide", -10.52862, 5.13259, 12.01421, 278.15, 588.67, 0.0039),
    ("124-38-9", "carbon dioxide", -8.55445, 4.01195, 9.52345, 274.19, 642.66, 0.0528),
    ("7783-06-4", "hydrogen sulfide", -4.51499, 5.23538, 4.42126, 273.15, 533.09, 0.0408),
    ("74-82-8", "methane", -10.44708, 4.66491, 12.12986, 275.46, 633.11, 0.0386),
    ("74-84-0", "ethane", -19.67563, 4.51222, 20.62567, 275.44, 473.46, 0.0259),
    ("2551-62-4", "sulfur hexafluoride", -16.56118, 2.15289, 20.35440, 283.14, 505.55, 0.0505),
)

CHAPOY_PROPANE_CORRELATION = {
    "CAS": "74-98-6",
    "component": "propane",
    "model": "chapoy_dippr101",
    "A": 552.64799,
    "B": -21334.4,
    "C": -85.89736,
    "D": 0.078453,
    "E": 1.0,
    "Tmin_K": 277.62,
    "Tmax_K": 368.16,
    "range_source": "Chapoy et al. (2004)",
    "rms_lnH": None,
    "fit_quality": 0.985,
    "uncertainty_upper_percent": None,
    "source": "Chapoy et al. (2004) propane-water correlation",
    "doi": "10.1016/j.fluid.2004.08.040",
    "source_row_ids": None,
    "source_literature_keys": "3129",
    "curvature_outlier_ids": None,
    "notes": (
        "Experimental kH correlation in kPa; reported AAD 1.5%; normalized "
        "at runtime to resolved Hcp_298 and B"
    ),
}

_THREE_PARAMETER_HCP_RE = re.compile(
    r"exp\("
    r"(?P<A>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?)"
    r"(?P<B>[+-](?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?)/T"
    r"(?P<C>[+-](?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?)ln\(T\)"
    r"\)",
    re.IGNORECASE,
)

# Sander orders types from most to least reliable. Bands prevent a large number
# of estimates from overwhelming a smaller body of measured/reviewed data;
# within a band the individual type weights preserve Sander's ordering.
QUALITY_TIERS = {
    "L": 0,
    "M": 0,
    "V": 1,
    "R": 1,
    "T": 2,
    "X": 3,
    "C": 3,
    "Q": 4,
    "E": 4,
    "?": 5,
}
TYPE_WEIGHTS = {
    "L": 1.00,
    "M": 0.88,
    "V": 0.72,
    "R": 0.64,
    "T": 0.50,
    "X": 0.36,
    "C": 0.30,
    "Q": 0.22,
    "E": 0.16,
    "?": 0.10,
}
TYPE_QUALITY = {
    "L": 0.98,
    "M": 0.92,
    "V": 0.82,
    "R": 0.76,
    "T": 0.66,
    "X": 0.55,
    "C": 0.48,
    "Q": 0.38,
    "E": 0.30,
    "?": 0.16,
}
EVIDENCE_WEIGHTS = {
    "L": 1.00,
    "M": 1.00,
    "V": 0.75,
    "R": 0.65,
    "T": 0.45,
    "X": 0.25,
    "C": 0.20,
    "Q": 0.15,
    "E": 0.10,
    "?": 0.05,
}
PRIMARY_EVIDENCE_TYPES = {"M", "V", "R"}
EXPLICIT_REVIEW_FAMILIES = {
    # Successive NASA/JPL kinetic and photochemical data evaluations.
    "1945": "jpl-atmospheric-evaluations",
    "2626": "jpl-atmospheric-evaluations",
    "3245": "jpl-atmospheric-evaluations",
    "3500": "jpl-atmospheric-evaluations",
    # Related hydration-function evaluations from the same research program.
    "3616": "plyasunov-shock-hydration-series",
    "3617": "plyasunov-shock-hydration-series",
    "3618": "plyasunov-shock-hydration-series",
    "3621": "plyasunov-shock-hydration-series",
    "3673": "plyasunov-shock-hydration-series",
    # Successive critical compilations by the same authors.
    "747": "staudinger-roberts-reviews",
    "1525": "staudinger-roberts-reviews",
}

# These notes identify a different solvent, a non-dilute or reactive apparent
# constant, or another standard state that the pure-water Henry model cannot
# represent. Version pinning and the source hash make this explicit audit list
# safe: a new upstream release must be reviewed before it can pass validation.
UNSUITABLE_NOTE_LABELS = {
    "759": "concentrated_brine",
    "3917": "heavy_water_solvent",
    "3521": "aqueous_ethanol_solvent",
    "769": "concentrated_solution_extrapolation",
    "201": "concentrated_solution_extrapolation",
    "752": "concentrated_solution_extrapolation",
    "2838": "reactive_non_henry_definition",
    "3668": "ph_specific_effective_constant",
    "3670": "ph_specific_effective_constant",
    "heffcl2": "reactive_effective_constant",
    "3857_cl2": "reactive_effective_constant",
    "3925": "non_dilute_total_solubility",
    "287hocl": "ph_specific_value",
    "3807": "not_intrinsic_henry_constant",
    "3809": "not_intrinsic_henry_constant",
    "jpl_cl2o": "not_intrinsic_henry_constant",
    "3857_cl2o": "not_intrinsic_henry_constant",
    "190_cl2o": "not_intrinsic_henry_constant",
    "hchodiol": "reactive_effective_constant",
    "rchodiol": "reactive_effective_constant",
    "3134-107-22-2-range": "salt_and_fulvic_acid_solution",
    "infty": "irreversible_reaction_effective_constant",
}
B_ONLY_UNSUITABLE_NOTE_LABELS = {
    "3525": "ph_specific_temperature_dependence",
    "3411": "ph_specific_temperature_dependence",
    "3940": "non_water_temperature_dependence",
}

_SCIENTIFIC_HTML_RE = re.compile(
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))×10<sup>([+-]?\d+)</sup>"
)


@dataclass(frozen=True)
class SourceDataset:
    rows: tuple[dict[str, object], ...]
    upstream_henry_rows: int
    upstream_species_rows: int
    excluded_seawater_rows: int
    excluded_wrong_rows: int
    excluded_missing_cas_rows: int
    archive_sha256: str


@dataclass(frozen=True)
class Candidate:
    row: dict[str, object]
    value: float
    weight: float
    provenance_multiplier: float


@dataclass(frozen=True)
class Aggregate:
    value: Optional[float]
    grade: Optional[str]
    quality_score: Optional[float]
    agreement_score: Optional[float]
    raw_agreement_score: Optional[float]
    effective_evidence: float
    direct_evidence: float
    review_evidence: float
    strongest_source_quality: Optional[float]
    source_count: int
    selected_rows: tuple[dict[str, object], ...]
    outlier_rows: tuple[dict[str, object], ...]


def nullable_text(value: object) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def parse_float(value: object) -> Optional[float]:
    """Parse exact finite values from the source's plain or HTML notation."""
    text = html.unescape(str(value or "")).strip().replace("−", "-")
    if not text or text.startswith((">", "<")) or "∞" in text:
        return None
    text = re.sub(r"\s+", "", text)
    text = _SCIENTIFIC_HTML_RE.sub(r"\1e\2", text)
    if re.search(r"<[^>]+>", text):
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def parse_three_parameter_hcp(note_text: object) -> Optional[tuple[float, float, float]]:
    """Parse Sander's Hcp=exp(A+B/T+C*ln(T)) note representation."""
    text = html.unescape(str(note_text or "")).replace("−", "-")
    text = re.sub(r"<[^>]+>", "", text)
    compact = re.sub(r"\s+|&nbsp;", "", text, flags=re.IGNORECASE)
    match = _THREE_PARAMETER_HCP_RE.search(compact)
    if match is None:
        return None
    values = tuple(float(match.group(name)) for name in ("A", "B", "C"))
    return values if all(math.isfinite(value) for value in values) else None


def _brockbank_ranges(path: Path = BROCKBANK_TABLE_C2) -> dict[str, tuple[float, float]]:
    ranges: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            low = float(row["Tmin_K"])
            high = float(row["Tmax_K"])
            if math.isfinite(low) and math.isfinite(high) and 0.0 < low <= high:
                ranges[str(row["CAS"]).strip()].append((low, high))
    resolved = {}
    for cas, values in ranges.items():
        unique = set(values)
        if len(unique) == 1:
            resolved[cas] = next(iter(unique))
    return resolved


def _brockbank_temperature_correlations(
    path: Path = BROCKBANK_TABLE_C2,
) -> dict[str, dict[str, object]]:
    """Load multi-temperature experimental DIPPR-101 kH regressions."""
    if not path.exists():
        return {}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["equation"]) != 101 or "E" not in str(row["data_type"]):
                continue
            Tmin = float(row["Tmin_K"])
            Tmax = float(row["Tmax_K"])
            if not (math.isfinite(Tmin) and math.isfinite(Tmax) and 0.0 < Tmin < Tmax):
                continue
            grouped[str(row["CAS"]).strip()].append({
                "component": str(row["name"]),
                "A": float(row["A"]),
                "B": float(row["B"]),
                "C": float(row["C"]),
                "D": float(row["D"]),
                "E": float(row["E"]),
                "Tmin_K": Tmin,
                "Tmax_K": Tmax,
                "uncertainty_upper_percent": float(row["uncertainty_upper_percent"]),
                "data_type": str(row["data_type"]),
            })
    return {
        cas: rows[0]
        for cas, rows in grouped.items()
        if len(rows) == 1
    }


def _dippr101_compatibility(
    correlation: Mapping[str, object],
    resolved_hcp: float,
    resolved_B: float,
) -> dict[str, object]:
    T = HENRY_REFERENCE_TEMPERATURE_K
    ln_volatility_kPa = (
        float(correlation["A"])
        + float(correlation["B"]) / T
        + float(correlation["C"]) * math.log(T)
        + float(correlation["D"]) * T ** float(correlation["E"])
    )
    raw_hcp = (
        WATER_CONCENTRATION_REF_MOL_M3
        / (math.exp(ln_volatility_kPa) * 1000.0)
    )
    raw_B = (
        WATER_DLN_CONCENTRATION_DINVT_REF_K
        - float(correlation["B"])
        + float(correlation["C"]) * T
        + float(correlation["D"])
        * float(correlation["E"])
        * T ** (float(correlation["E"]) + 1.0)
    )
    value_difference_percent = 100.0 * (raw_hcp / float(resolved_hcp) - 1.0)
    slope_difference_K = raw_B - float(resolved_B)
    slope_limit_K = max(
        TEMPERATURE_SLOPE_COMPATIBILITY_MIN_K,
        TEMPERATURE_SLOPE_COMPATIBILITY_FRACTION * abs(float(resolved_B)),
    )
    compatible = (
        abs(value_difference_percent) <= TEMPERATURE_VALUE_COMPATIBILITY_PERCENT
        and abs(slope_difference_K) <= slope_limit_K
    )
    return {
        "compatible": compatible,
        "raw_reference_value_difference_percent": value_difference_percent,
        "raw_reference_slope_difference_K": slope_difference_K,
        "raw_reference_slope_difference_percent": (
            100.0 * slope_difference_K / abs(float(resolved_B))
            if abs(float(resolved_B)) > 1e-12 else None
        ),
        "slope_limit_K": slope_limit_K,
    }


def _iapws_water_psat_bar(T: float) -> float:
    reduced_temperature = float(T) / 647.096
    tau = max(0.0, 1.0 - reduced_temperature)
    coefficients = (
        (-7.85951783, 1.0),
        (1.84408259, 1.5),
        (-11.7866497, 3.0),
        (22.6807411, 3.5),
        (-15.9618719, 4.0),
        (1.80122502, 7.5),
    )
    exponent = sum(a * tau ** b for a, b in coefficients) / reduced_temperature
    return 220.64 * math.exp(exponent)


def _iapws_compatibility(
    correlation: Mapping[str, object],
    resolved_hcp: float,
    resolved_B: float,
) -> dict[str, object]:
    def raw_hcp(T: float) -> float:
        reduced_temperature = T / 647.096
        tau = max(0.0, 1.0 - reduced_temperature)
        ln_ratio = (
            float(correlation["A"]) / reduced_temperature
            + float(correlation["B"]) * tau ** 0.355 / reduced_temperature
            + float(correlation["C"]) * reduced_temperature ** -0.41 * math.exp(tau)
        )
        concentration = WATER_CONCENTRATION_REF_MOL_M3 * math.exp(
            WATER_DLN_CONCENTRATION_DINVT_REF_K
            * (1.0 / T - 1.0 / HENRY_REFERENCE_TEMPERATURE_K)
        )
        return concentration / (_iapws_water_psat_bar(T) * math.exp(ln_ratio) * 1.0e5)

    T = HENRY_REFERENCE_TEMPERATURE_K
    dT = 0.05
    raw_reference = raw_hcp(T)
    raw_B = (
        math.log(raw_hcp(T + dT)) - math.log(raw_hcp(T - dT))
    ) / (1.0 / (T + dT) - 1.0 / (T - dT))
    value_difference_percent = 100.0 * (raw_reference / float(resolved_hcp) - 1.0)
    slope_difference_K = raw_B - float(resolved_B)
    slope_limit_K = max(
        TEMPERATURE_SLOPE_COMPATIBILITY_MIN_K,
        TEMPERATURE_SLOPE_COMPATIBILITY_FRACTION * abs(float(resolved_B)),
    )
    return {
        "compatible": (
            abs(value_difference_percent) <= TEMPERATURE_VALUE_COMPATIBILITY_PERCENT
            and abs(slope_difference_K) <= slope_limit_K
        ),
        "raw_reference_value_difference_percent": value_difference_percent,
        "raw_reference_slope_difference_K": slope_difference_K,
        "raw_reference_slope_difference_percent": (
            100.0 * slope_difference_K / abs(float(resolved_B))
            if abs(float(resolved_B)) > 1e-12 else None
        ),
        "slope_limit_K": slope_limit_K,
    }


def _hcp_three_parameter_compatibility(
    A: float,
    B: float,
    C: float,
    resolved_hcp: float,
    resolved_B: float,
) -> dict[str, object]:
    T = HENRY_REFERENCE_TEMPERATURE_K
    raw_hcp = math.exp(A + B / T + C * math.log(T))
    raw_B = B - C * T
    value_difference_percent = 100.0 * (raw_hcp / float(resolved_hcp) - 1.0)
    slope_difference_K = raw_B - float(resolved_B)
    slope_limit_K = max(
        TEMPERATURE_SLOPE_COMPATIBILITY_MIN_K,
        TEMPERATURE_SLOPE_COMPATIBILITY_FRACTION * abs(float(resolved_B)),
    )
    return {
        "compatible": (
            abs(value_difference_percent) <= TEMPERATURE_VALUE_COMPATIBILITY_PERCENT
            and abs(slope_difference_K) <= slope_limit_K
        ),
        "raw_reference_value_difference_percent": value_difference_percent,
        "raw_reference_slope_difference_K": slope_difference_K,
        "raw_reference_slope_difference_percent": (
            100.0 * slope_difference_K / abs(float(resolved_B))
            if abs(float(resolved_B)) > 1e-12 else None
        ),
        "slope_limit_K": slope_limit_K,
    }


def _normalization_anchor_is_a_range(aggregate: Mapping[str, object]) -> bool:
    return (
        aggregate.get("quality_h") is not None
        and aggregate.get("quality_b") is not None
        and float(aggregate["quality_h"]) >= 0.84
        and float(aggregate["quality_b"]) >= 0.84
    )


def _row_temperature_range(
    row: dict[str, object],
    brockbank_ranges: dict[str, tuple[float, float]],
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    literature = str(row.get("literature_id") or "")
    cas = str(row.get("casrn") or "").strip()
    if literature == "3518" and cas in brockbank_ranges:
        low, high = brockbank_ranges[cas]
        return low, high, "Brockbank (2013) Table C.2"
    if literature == "3681":
        return 273.15, 368.15, "Schwardt et al. (2021), 0-95 C"
    if literature == "3129":
        return 277.62, 368.16, "Chapoy et al. (2004)"
    return None, None, None


def source_temperature_correlations(
    dataset: SourceDataset,
) -> list[dict[str, object]]:
    """Return every explicit three-parameter Hcp fit embedded in source notes."""
    brockbank_ranges = _brockbank_ranges()
    records = []
    for row in dataset.rows:
        coefficients = parse_three_parameter_hcp(row.get("note_texts"))
        if coefficients is None:
            continue
        Tmin, Tmax, range_source = _row_temperature_range(row, brockbank_ranges)
        records.append({
            "source_row_id": int(row["id"]),
            "CAS": str(row["casrn"]).strip(),
            "model": "exp_a_b_over_t_c_log_t",
            "A": coefficients[0],
            "B": coefficients[1],
            "C": coefficients[2],
            "Tmin_K": Tmin,
            "Tmax_K": Tmax,
            "range_source": range_source,
            "htype": str(row["htype"]),
            "literature_key": nullable_text(row.get("literature_id")),
            "selected_for_B": int(row.get("_b_status") == "selected"),
            "note_labels": nullable_text(row.get("note_labels")),
        })
    if (
        dataset.archive_sha256 == SOURCE_ARCHIVE_SHA256
        and len(records) != EXPECTED_ELIGIBLE_THREE_PARAMETER_ROWS
    ):
        raise ValueError(
            f"Parsed {len(records)} Sander three-parameter correlations; "
            f"expected {EXPECTED_ELIGIBLE_THREE_PARAMETER_ROWS} eligible pure-water rows"
        )
    return records


def _note_labels(row: dict[str, object]) -> set[str]:
    return {
        label.strip().lower()
        for label in str(row.get("note_labels") or "").split("|")
        if label.strip()
    }


def _field_suitability_reason(row: dict[str, object], field: str) -> Optional[str]:
    labels = _note_labels(row)
    if any("seawater" in label for label in labels):
        return "seawater"
    for label in labels:
        if label in UNSUITABLE_NOTE_LABELS:
            return UNSUITABLE_NOTE_LABELS[label]
    note_text = html.unescape(str(row.get("note_texts") or "")).lower()
    h_is_explicitly_intrinsic = field == "h" and bool(
        labels & B_ONLY_UNSUITABLE_NOTE_LABELS.keys()
    )
    if not h_is_explicitly_intrinsic and (
        "value at ph" in note_text
        or "measured at low ph" in note_text
        or re.search(r"\bat ph\s*=", note_text)
    ):
        return "ph_specific_value"
    if field == "b":
        for label in labels:
            if label in B_ONLY_UNSUITABLE_NOTE_LABELS:
                return B_ONLY_UNSUITABLE_NOTE_LABELS[label]
    return None


def _source_temperature(row: dict[str, object]) -> Optional[float]:
    for label in _note_labels(row):
        match = re.fullmatch(r"at(\d+(?:\.\d+)?)k", label)
        if match:
            return float(match.group(1))
    return None


def _provenance_multiplier(row: dict[str, object], field: str) -> float:
    labels = _note_labels(row)
    multiplier = 1.0
    if "hightextrapol" in labels:
        multiplier *= 0.75
    if "atroomt" in labels:
        multiplier *= 0.85
    if "whichref" in labels:
        multiplier *= 0.85
    if "cdep" in labels:
        multiplier *= 0.75
    if any(
        "consicheck" in label or "pc_problem" in label or "inconsistent" in label
        for label in labels
    ):
        multiplier *= 0.75
    if field == "b":
        if "htdep_r2_lessthan_0.9" in labels:
            multiplier *= 0.55
        if "htdep_r2_lessthan_0.5" in labels:
            multiplier *= 0.25
    return multiplier


def _normalized_h_value(row: dict[str, object], value: float) -> tuple[Optional[float], Optional[str], float]:
    labels = _note_labels(row)
    if "unknownt" in labels:
        return None, "unknown_temperature", 0.0
    temperature = _source_temperature(row)
    if temperature is None:
        return value, None, 1.0
    delta = abs(temperature - 298.15)
    B = parse_float(row.get("mindhr"))
    if B is not None and _field_suitability_reason(row, "b") is None and delta <= 40.0:
        exponent = -B * (1.0 / temperature - 1.0 / 298.15)
        return value * math.exp(max(min(exponent, 100.0), -100.0)), None, 0.95
    if delta <= 7.0:
        return value, None, max(0.65, 1.0 - delta / 25.0)
    return None, "temperature_not_298K", 0.0


def _candidate(row: dict[str, object], field: str) -> tuple[Optional[Candidate], str]:
    raw_field = "hominus" if field == "h" else "mindhr"
    value = parse_float(row.get(raw_field))
    if value is None or (field == "h" and value <= 0.0):
        return None, "missing_or_nonfinite"
    reason = _field_suitability_reason(row, field)
    if reason is not None:
        return None, f"excluded:{reason}"
    temperature_multiplier = 1.0
    if field == "h":
        value, reason, temperature_multiplier = _normalized_h_value(row, value)
        if value is None:
            return None, f"excluded:{reason}"
    provenance = _provenance_multiplier(row, field) * temperature_multiplier
    weight = TYPE_WEIGHTS[str(row["htype"])] * provenance
    return Candidate(row, float(value), weight, provenance), "candidate"


def _weighted_median(candidates: list[Candidate], transform=lambda value: value) -> float:
    ordered = sorted((transform(item.value), item.weight) for item in candidates)
    midpoint = sum(weight for _value, weight in ordered) / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= midpoint:
            return value
    return ordered[-1][0]


def _independence_adjusted(candidates: list[Candidate]) -> list[Candidate]:
    counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        key = nullable_text(candidate.row.get("literature_id")) or f"row:{candidate.row['id']}"
        counts[key] += 1
    return [
        Candidate(
            candidate.row,
            candidate.value,
            candidate.weight / counts[
                nullable_text(candidate.row.get("literature_id")) or f"row:{candidate.row['id']}"
            ],
            candidate.provenance_multiplier,
        )
        for candidate in candidates
    ]


def _outliers(candidates: list[Candidate], field: str) -> set[int]:
    if len(candidates) < 3:
        return set()
    transform = math.log if field == "h" else (lambda value: value)
    center = _weighted_median(candidates, transform)
    deviations = [
        Candidate(item.row, abs(transform(item.value) - center), item.weight, item.provenance_multiplier)
        for item in candidates
    ]
    mad = _weighted_median(deviations)
    if field == "h":
        threshold = max(math.log(3.0), 3.5 * 1.4826 * mad)
    else:
        threshold = max(1000.0, 0.35 * abs(center), 3.5 * 1.4826 * mad)
    return {
        int(item.row["id"])
        for item in candidates
        if abs(transform(item.value) - center) > threshold
    }


def _agreement(candidates: list[Candidate], value: float, field: str) -> float:
    if len(candidates) <= 1:
        return 0.75
    total_weight = sum(item.weight for item in candidates)
    if field == "h":
        variance = sum(
            item.weight * (math.log(item.value) - math.log(value)) ** 2
            for item in candidates
        ) / total_weight
        return max(0.0, 1.0 - math.sqrt(variance) / math.log(5.0))
    variance = sum(
        item.weight * (item.value - value) ** 2 for item in candidates
    ) / total_weight
    scale = max(abs(value), 2000.0)
    return max(0.0, 1.0 - math.sqrt(variance) / (0.75 * scale))


def _review_family_key(row: dict[str, object]) -> str:
    literature_key = nullable_text(row.get("literature_id")) or f"row:{row['id']}"
    explicit = EXPLICIT_REVIEW_FAMILIES.get(literature_key)
    if explicit is not None:
        return explicit
    quotation = html.unescape(str(row.get("literature_quotation") or ""))
    quotation = re.sub(r"<[^>]+>", " ", quotation)
    author_text = re.split(r"\(\d{4}", quotation, maxsplit=1)[0]
    lead_author = re.split(
        r"\bet\s+al\.?\b|\band\b|&|,",
        author_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    normalized = re.sub(r"[^a-z0-9]+", "-", lead_author.lower()).strip("-")
    return f"lead-author:{normalized}" if normalized else f"literature:{literature_key}"


def _evidence_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Return dependency-adjusted evidence with diminishing review-family support."""
    reference_counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        reference = nullable_text(candidate.row.get("literature_id")) or f"row:{candidate.row['id']}"
        reference_counts[reference] += 1

    evidence = []
    for candidate in candidates:
        reference = nullable_text(candidate.row.get("literature_id")) or f"row:{candidate.row['id']}"
        weight = (
            EVIDENCE_WEIGHTS[str(candidate.row["htype"])]
            * candidate.provenance_multiplier
            / reference_counts[reference]
        )
        evidence.append(
            Candidate(
                candidate.row,
                candidate.value,
                weight,
                candidate.provenance_multiplier,
            )
        )

    non_reviews = [
        candidate for candidate in evidence if str(candidate.row["htype"]) != "L"
    ]
    review_families: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in evidence:
        if str(candidate.row["htype"]) == "L":
            review_families[_review_family_key(candidate.row)].append(candidate)

    ranked_families = []
    for family, members in review_families.items():
        family_strength = max(
            EVIDENCE_WEIGHTS["L"] * member.provenance_multiplier
            for member in members
        )
        ranked_families.append((family_strength, family, members))
    ranked_families.sort(key=lambda item: (-item[0], item[1]))

    reviews = []
    for rank, (family_strength, _family, members) in enumerate(ranked_families):
        family_contribution = family_strength * (0.5 ** rank)
        family_row_weight = sum(member.weight for member in members)
        scale = family_contribution / family_row_weight
        reviews.extend(
            Candidate(
                member.row,
                member.value,
                member.weight * scale,
                member.provenance_multiplier,
            )
            for member in members
        )
    return [*non_reviews, *reviews]


def _confidence_score(
    retained: list[Candidate],
    all_top_band: list[Candidate],
    outlier_ids: set[int],
    value: float,
    field: str,
) -> tuple[float, float, float, float, float, float, float]:
    """Calculate monotonic evidence confidence separately from value aggregation."""
    evidence = _evidence_candidates(retained)
    total_evidence = sum(candidate.weight for candidate in evidence)
    direct_evidence = sum(
        candidate.weight
        for candidate in evidence
        if str(candidate.row["htype"]) in PRIMARY_EVIDENCE_TYPES
    )
    review_evidence = sum(
        candidate.weight
        for candidate in evidence
        if str(candidate.row["htype"]) == "L"
    )
    strongest = max(
        TYPE_QUALITY[str(candidate.row["htype"])]
        * (0.85 + 0.15 * candidate.provenance_multiplier)
        for candidate in evidence
    )
    raw_agreement = _agreement(evidence, value, field)
    information = 1.0 - math.exp(-total_evidence / 1.5)
    agreement = 0.5 + (raw_agreement - 0.5) * information
    support = information

    all_evidence = _evidence_candidates(all_top_band)
    all_evidence_weight = sum(candidate.weight for candidate in all_evidence)
    outlier_weight = sum(
        candidate.weight
        for candidate in all_evidence
        if int(candidate.row["id"]) in outlier_ids
    )
    outlier_fraction = (
        outlier_weight / all_evidence_weight if all_evidence_weight > 0.0 else 0.0
    )

    score = 0.60 * strongest + 0.25 * agreement + 0.15 * support
    score *= 1.0 - 0.30 * outlier_fraction

    # Reviews summarize evidence but repeated editions are not independent
    # experiments. High grades require increasing amounts of primary evidence.
    if direct_evidence < 0.5 and review_evidence > 0.0:
        review_only_cap = 0.839999 if review_evidence <= 1.000001 else 0.899999
        score = min(score, review_only_cap)
    elif direct_evidence < 1.5:
        score = min(score, 0.899999)
    if not (
        direct_evidence >= 4.0
        or (review_evidence > 0.0 and direct_evidence >= 2.5)
    ):
        score = min(score, 0.949999)
    if agreement < 0.90 or outlier_ids:
        score = min(score, 0.949999)
    if agreement < 0.65:
        score = min(score, 0.839999)

    return (
        max(0.0, min(1.0, score)),
        agreement,
        raw_agreement,
        total_evidence,
        direct_evidence,
        review_evidence,
        strongest,
    )


def _grade(score: float) -> str:
    for threshold, grade in (
        (0.95, "A+"), (0.90, "A"), (0.84, "A-"),
        (0.78, "B+"), (0.72, "B"), (0.66, "B-"),
        (0.58, "C+"), (0.50, "C"), (0.42, "C-"),
        (0.34, "D+"), (0.27, "D"), (0.20, "D-"),
    ):
        if score >= threshold:
            return grade
    return "F"


def aggregate_field(rows: list[dict[str, object]], field: str) -> Aggregate:
    status_key = f"_{field}_status"
    candidates = []
    for row in rows:
        candidate, status = _candidate(row, field)
        row[status_key] = status
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return Aggregate(None, None, None, None, None, 0.0, 0.0, 0.0, None, 0, (), ())

    best_band = min(QUALITY_TIERS[str(item.row["htype"])] for item in candidates)
    top_band = []
    for item in candidates:
        if QUALITY_TIERS[str(item.row["htype"])] == best_band:
            top_band.append(item)
        else:
            item.row[status_key] = "excluded:lower_reliability_band"
    top_band = _independence_adjusted(top_band)

    outlier_ids = _outliers(top_band, field)
    retained = []
    outlier_rows = []
    for item in top_band:
        if int(item.row["id"]) in outlier_ids:
            item.row[status_key] = "excluded:statistical_outlier"
            outlier_rows.append(item.row)
        else:
            item.row[status_key] = "selected"
            retained.append(item)
    if not retained:
        raise RuntimeError(f"Outlier filtering removed every {field} candidate")

    total_weight = sum(item.weight for item in retained)
    if field == "h":
        value = math.exp(sum(item.weight * math.log(item.value) for item in retained) / total_weight)
    else:
        value = sum(item.weight * item.value for item in retained) / total_weight
    independent_sources = {
        nullable_text(item.row.get("literature_id")) or f"row:{item.row['id']}"
        for item in retained
    }
    (
        quality,
        agreement,
        raw_agreement,
        effective_evidence,
        direct_evidence,
        review_evidence,
        strongest_source_quality,
    ) = _confidence_score(retained, top_band, outlier_ids, value, field)
    return Aggregate(
        value=value,
        grade=_grade(quality),
        quality_score=quality,
        agreement_score=agreement,
        raw_agreement_score=raw_agreement,
        effective_evidence=effective_evidence,
        direct_evidence=direct_evidence,
        review_evidence=review_evidence,
        strongest_source_quality=strongest_source_quality,
        source_count=len(independent_sources),
        selected_rows=tuple(item.row for item in retained),
        outlier_rows=tuple(outlier_rows),
    )


def build_record(cas: str, rows: list[dict[str, object]]) -> tuple[object, ...]:
    rows_by_id = sorted(rows, key=lambda row: int(row["id"]))
    identity_row = rows_by_id[0]
    h = aggregate_field(rows_by_id, "h")
    b = aggregate_field(rows_by_id, "b")
    selected_rows = sorted(
        {int(row["id"]): row for row in [*h.selected_rows, *b.selected_rows]}.values(),
        key=lambda row: int(row["id"]),
    ) or rows_by_id
    paired_ids = {int(row["id"]) for row in h.selected_rows} & {
        int(row["id"]) for row in b.selected_rows
    }
    pair_fraction = (
        len(paired_ids) / len(b.selected_rows) if b.selected_rows and h.selected_rows else None
    )
    return (
        cas,
        nullable_text(identity_row.get("iupac")),
        nullable_text(identity_row.get("formula")),
        nullable_text(identity_row.get("trivial")),
        nullable_text(identity_row.get("casrn")),
        nullable_text(identity_row.get("inchikey")),
        h.value,
        b.value,
        h.grade,
        b.grade,
        h.quality_score,
        b.quality_score,
        h.agreement_score,
        b.agreement_score,
        h.raw_agreement_score,
        b.raw_agreement_score,
        h.effective_evidence,
        b.effective_evidence,
        h.direct_evidence,
        b.direct_evidence,
        h.review_evidence,
        b.review_evidence,
        h.strongest_source_quality,
        b.strongest_source_quality,
        int(any('moretdep' in str(row.get('note_labels') or '').lower()
                for row in h.selected_rows)),
        int(any('moretdep' in str(row.get('note_labels') or '').lower()
                for row in b.selected_rows)),
        h.source_count,
        b.source_count,
        pair_fraction,
        int(selected_rows[0]["id"]),
        int(selected_rows[-1]["id"]),
        ",".join(str(row["id"]) for row in h.selected_rows) or None,
        ",".join(str(row["id"]) for row in b.selected_rows) or None,
        ",".join(str(row["id"]) for row in h.outlier_rows) or None,
        ",".join(str(row["id"]) for row in b.outlier_rows) or None,
    )


def _curvature_outliers(candidates: list[Candidate]) -> set[int]:
    if len(candidates) < 3:
        return set()
    center = _weighted_median(candidates)
    deviations = [
        Candidate(item.row, abs(item.value - center), item.weight, item.provenance_multiplier)
        for item in candidates
    ]
    mad = _weighted_median(deviations)
    threshold = max(5.0, 3.5 * 1.4826 * mad)
    return {
        int(item.row["id"])
        for item in candidates
        if abs(item.value - center) > threshold
    }


def resolved_temperature_correlations(
    records: list[tuple[object, ...]],
    dataset: SourceDataset,
    source_correlations: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Resolve one active richer temperature model per eligible CAS."""
    aggregate_by_cas = {
        str(record[0]): {
            "H": record[6],
            "B": record[7],
            "quality_h": record[10],
            "quality_b": record[11],
        }
        for record in records
    }
    active = []
    rejections = []
    iapws_cas = set()
    for cas, component, A, B, C, Tmin, Tmax, rms in IAPWS_G7_04_RECORDS:
        aggregate = aggregate_by_cas.get(cas)
        if not aggregate or aggregate["H"] is None or aggregate["B"] is None:
            continue
        normalize = _normalization_anchor_is_a_range(aggregate)
        resolved = {
            "CAS": cas,
            "component": component,
            "model": "iapws_g7_04",
            "A": A,
            "B": B,
            "C": C,
            "D": 0.0,
            "E": 0.0,
            "Tmin_K": Tmin,
            "Tmax_K": Tmax,
            "range_source": "IAPWS G7-04 Table 2",
            "rms_lnH": rms,
            "fit_quality": math.exp(-rms),
            "uncertainty_upper_percent": None,
            "source": "IAPWS G7-04",
            "doi": "10.1063/1.1564818",
            "source_row_ids": None,
            "source_literature_keys": None,
            "curvature_outlier_ids": None,
            "normalize_to_reference": int(normalize),
            "notes": (
                "IAPWS wide-range kH correlation; "
                + (
                    "normalized at runtime to A-range resolved Hcp_298 and B"
                    if normalize else "used without normalization because an anchor is below A-range"
                )
            ),
        }
        compatibility = _iapws_compatibility(
            resolved,
            float(aggregate["H"]),
            float(aggregate["B"]),
        )
        resolved.update({
            key: compatibility[key]
            for key in (
                "raw_reference_value_difference_percent",
                "raw_reference_slope_difference_K",
                "raw_reference_slope_difference_percent",
            )
        })
        if not normalize or compatibility["compatible"]:
            active.append(resolved)
            iapws_cas.add(cas)
        else:
            rejections.append({
                **resolved,
                "reason": "reference_value_or_slope_incompatible",
                "slope_limit_K": compatibility["slope_limit_K"],
            })

    dedicated_cas = set(iapws_cas)
    chapoy = dict(CHAPOY_PROPANE_CORRELATION)
    chapoy_aggregate = aggregate_by_cas.get(str(chapoy["CAS"]))
    if chapoy_aggregate and chapoy_aggregate["H"] is not None and chapoy_aggregate["B"] is not None:
        normalize = _normalization_anchor_is_a_range(chapoy_aggregate)
        chapoy["normalize_to_reference"] = int(normalize)
        chapoy["notes"] = (
            "Experimental kH correlation in kPa; reported AAD 1.5%; "
            + (
                "normalized at runtime to A-range resolved Hcp_298 and B"
                if normalize else "used without normalization because an anchor is below A-range"
            )
        )
        compatibility = _dippr101_compatibility(
            chapoy,
            float(chapoy_aggregate["H"]),
            float(chapoy_aggregate["B"]),
        )
        chapoy.update({
            key: compatibility[key]
            for key in (
                "raw_reference_value_difference_percent",
                "raw_reference_slope_difference_K",
                "raw_reference_slope_difference_percent",
            )
        })
        if not normalize or compatibility["compatible"]:
            active.append(chapoy)
            dedicated_cas.add(str(chapoy["CAS"]))
        else:
            rejections.append({
                **chapoy,
                "reason": "reference_value_or_slope_incompatible",
                "slope_limit_K": compatibility["slope_limit_K"],
            })

    brockbank_cas = set()
    for cas, correlation in sorted(_brockbank_temperature_correlations().items()):
        if cas in dedicated_cas:
            continue
        aggregate = aggregate_by_cas.get(cas)
        if not aggregate or aggregate["H"] is None or aggregate["B"] is None:
            continue
        normalize = _normalization_anchor_is_a_range(aggregate)
        uncertainty = float(correlation["uncertainty_upper_percent"])
        resolved = {
            "CAS": cas,
            "component": correlation["component"],
            "model": "brockbank_dippr101",
            "A": correlation["A"],
            "B": correlation["B"],
            "C": correlation["C"],
            "D": correlation["D"],
            "E": correlation["E"],
            "Tmin_K": correlation["Tmin_K"],
            "Tmax_K": correlation["Tmax_K"],
            "range_source": "Brockbank (2013) Table C.2",
            "rms_lnH": None,
            "fit_quality": max(0.2, 1.0 - uncertainty / 100.0),
            "uncertainty_upper_percent": uncertainty,
            "source": "Brockbank (2013) experimental DIPPR-101 regression",
            "doi": None,
            "source_row_ids": None,
            "source_literature_keys": "3518",
            "curvature_outlier_ids": None,
            "normalize_to_reference": int(normalize),
            "notes": (
                f"Table C.2 data type {correlation['data_type']}; kH in kPa; "
                + (
                    "normalized at runtime to A-range resolved Hcp_298 and B"
                    if normalize else "used without normalization because an anchor is below A-range"
                )
            ),
        }
        compatibility = _dippr101_compatibility(
            correlation,
            float(aggregate["H"]),
            float(aggregate["B"]),
        )
        resolved.update({
            key: compatibility[key]
            for key in (
                "raw_reference_value_difference_percent",
                "raw_reference_slope_difference_K",
                "raw_reference_slope_difference_percent",
            )
        })
        if not normalize or compatibility["compatible"]:
            active.append(resolved)
            brockbank_cas.add(cas)
        else:
            rejections.append({
                **resolved,
                "reason": "reference_value_or_slope_incompatible",
                "slope_limit_K": compatibility["slope_limit_K"],
            })

    rows_by_id = {int(row["id"]): row for row in dataset.rows}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for correlation in source_correlations:
        if correlation["selected_for_B"]:
            grouped[str(correlation["CAS"])].append(correlation)

    for cas, correlations in sorted(grouped.items()):
        if cas in dedicated_cas or cas in brockbank_cas:
            continue
        aggregate = aggregate_by_cas.get(cas)
        if not aggregate or aggregate["H"] is None or aggregate["B"] is None:
            continue
        candidates = []
        correlation_by_id = {}
        for correlation in correlations:
            row = rows_by_id[int(correlation["source_row_id"])]
            b_candidate, _status = _candidate(row, "b")
            if b_candidate is None:
                continue
            candidate = Candidate(
                row=row,
                value=float(correlation["C"]),
                weight=b_candidate.weight,
                provenance_multiplier=b_candidate.provenance_multiplier,
            )
            candidates.append(candidate)
            correlation_by_id[int(row["id"])] = correlation
        candidates = _independence_adjusted(candidates)
        outlier_ids = _curvature_outliers(candidates)
        retained = [
            candidate for candidate in candidates
            if int(candidate.row["id"]) not in outlier_ids
        ]
        if not retained:
            continue
        total_weight = sum(candidate.weight for candidate in retained)
        curvature = sum(candidate.weight * candidate.value for candidate in retained) / total_weight
        raw_A = sum(
            candidate.weight
            * float(correlation_by_id[int(candidate.row["id"])]["A"])
            for candidate in retained
        ) / total_weight
        raw_B = sum(
            candidate.weight
            * float(correlation_by_id[int(candidate.row["id"])]["B"])
            for candidate in retained
        ) / total_weight
        compatibility = _hcp_three_parameter_compatibility(
            raw_A,
            raw_B,
            curvature,
            float(aggregate["H"]),
            float(aggregate["B"]),
        )
        normalize = _normalization_anchor_is_a_range(aggregate)
        normalized_B = float(aggregate["B"]) + curvature * HENRY_REFERENCE_TEMPERATURE_K
        normalized_A = (
            math.log(float(aggregate["H"]))
            - normalized_B / HENRY_REFERENCE_TEMPERATURE_K
            - curvature * math.log(HENRY_REFERENCE_TEMPERATURE_K)
        )
        coefficient_A = normalized_A if normalize else raw_A
        coefficient_B = normalized_B if normalize else raw_B

        ranges = [
            (
                float(correlation_by_id[int(candidate.row["id"])]["Tmin_K"]),
                float(correlation_by_id[int(candidate.row["id"])]["Tmax_K"]),
                str(correlation_by_id[int(candidate.row["id"])]["range_source"]),
            )
            for candidate in retained
            if correlation_by_id[int(candidate.row["id"])]["Tmin_K"] is not None
            and correlation_by_id[int(candidate.row["id"])]["Tmax_K"] is not None
        ]
        if ranges:
            Tmin = max(item[0] for item in ranges)
            Tmax = min(item[1] for item in ranges)
            if Tmin > Tmax:
                strongest = max(retained, key=lambda candidate: candidate.weight)
                chosen = correlation_by_id[int(strongest.row["id"])]
                Tmin = float(chosen["Tmin_K"] or HENRY_REFERENCE_TEMPERATURE_K - 20.0)
                Tmax = float(chosen["Tmax_K"] or HENRY_REFERENCE_TEMPERATURE_K + 25.0)
            range_source = "intersection: " + "; ".join(sorted({item[2] for item in ranges}))
        else:
            Tmin = HENRY_REFERENCE_TEMPERATURE_K - 20.0
            Tmax = HENRY_REFERENCE_TEMPERATURE_K + 25.0
            range_source = "default_5_to_50C; source range unavailable"

        retained_ids = [int(candidate.row["id"]) for candidate in retained]
        literature_keys = sorted({
            str(candidate.row.get("literature_id") or "")
            for candidate in retained
            if candidate.row.get("literature_id")
        })
        component = str(rows_by_id[retained_ids[0]].get("iupac") or cas)
        fit_quality = sum(
            candidate.weight
            * TYPE_QUALITY[str(candidate.row["htype"])]
            * candidate.provenance_multiplier
            for candidate in retained
        ) / total_weight
        resolved = {
            "CAS": cas,
            "component": component,
            "model": "exp_a_b_over_t_c_log_t",
            "A": coefficient_A,
            "B": coefficient_B,
            "C": curvature,
            "D": 0.0,
            "E": 0.0,
            "Tmin_K": Tmin,
            "Tmax_K": Tmax,
            "range_source": range_source,
            "rms_lnH": None,
            "fit_quality": fit_quality,
            "uncertainty_upper_percent": None,
            "source": "Sander 5.0 embedded three-parameter fits",
            "doi": "10.5194/acp-23-10901-2023",
            "source_row_ids": ",".join(map(str, sorted(retained_ids))),
            "source_literature_keys": ",".join(literature_keys) or None,
            "curvature_outlier_ids": (
                ",".join(map(str, sorted(outlier_ids))) if outlier_ids else None
            ),
            "normalize_to_reference": int(normalize),
            "raw_reference_value_difference_percent": (
                compatibility["raw_reference_value_difference_percent"]
            ),
            "raw_reference_slope_difference_K": (
                compatibility["raw_reference_slope_difference_K"]
            ),
            "raw_reference_slope_difference_percent": (
                compatibility["raw_reference_slope_difference_percent"]
            ),
            "notes": (
                "Curvature combined from selected source fits; "
                + (
                    "A and B normalized to A-range resolved Hcp_298 and local derivative"
                    if normalize else "raw combined source fit retained because an anchor is below A-range"
                )
            ),
        }
        if not normalize or compatibility["compatible"]:
            active.append(resolved)
        else:
            rejections.append({
                **resolved,
                "reason": "reference_value_or_slope_incompatible",
                "slope_limit_K": compatibility["slope_limit_K"],
            })
    return (
        sorted(active, key=lambda record: str(record["CAS"])),
        sorted(rejections, key=lambda record: (str(record["CAS"]), str(record["model"]))),
    )


def _archive_sql(path: Path) -> tuple[str, str]:
    archive_bytes = path.read_bytes()
    digest = hashlib.sha256(archive_bytes).hexdigest()
    with zipfile.ZipFile(path) as archive:
        sql_members = [name for name in archive.namelist() if name.lower().endswith(".sql")]
        if sql_members != ["henry-5.0.0.sql"]:
            raise ValueError(
                f"Expected only henry-5.0.0.sql in {path}, found {sql_members!r}"
            )
        sql_text = archive.read(sql_members[0]).decode("utf-8")
    return sql_text, digest


def _sqlite_compatible_source_sql(sql_text: str) -> str:
    if "CREATE TABLE henry " not in sql_text or "CREATE TABLE henry_notes " not in sql_text:
        raise ValueError("Source archive does not contain the expected Henry 5.0 schema")
    converted = sql_text.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
    converted = converted.replace(" CASCADE;", ";")
    converted = converted.replace(
        "(SELECT CURRVAL('henry_id_seq'))",
        "(SELECT MAX(id) FROM henry)",
    )
    if "CURRVAL(" in converted or "SERIAL PRIMARY KEY" in converted:
        raise ValueError("Unhandled PostgreSQL construct in Henry source archive")
    return converted


@contextmanager
def _source_connection(path: Path) -> Iterator[tuple[sqlite3.Connection, str]]:
    sql_text, digest = _archive_sql(path)
    with tempfile.TemporaryDirectory(prefix="pfdsim-henry-source-") as directory:
        staging_path = Path(directory) / "source.sqlite"
        connection = sqlite3.connect(staging_path)
        try:
            connection.executescript(_sqlite_compatible_source_sql(sql_text))
            yield connection, digest
        finally:
            connection.close()


def load_source_dataset(path: Path) -> SourceDataset:
    path = Path(path)
    with _source_connection(path) as (connection, digest):
        upstream_henry_rows = connection.execute("SELECT COUNT(*) FROM henry").fetchone()[0]
        upstream_species_rows = connection.execute("SELECT COUNT(*) FROM species").fetchone()[0]

        cursor = connection.execute(
            """
            SELECT
                h.id,
                h.Hominus AS hominus,
                h.mindHR AS mindhr,
                h.htype,
                s.id AS species_id,
                l.bibtexkey AS literature_id,
                l.quotation AS literature_quotation,
                s.iupac,
                s.formula,
                s.trivial,
                s.casrn,
                s.inchikey,
                s.subcat_id,
                GROUP_CONCAT(n.notelabel, '|') AS note_labels,
                GROUP_CONCAT(n.notetext, ' || ') AS note_texts
            FROM henry h
            JOIN species s ON s.id = h.species_id
            LEFT JOIN literature l ON l.id = h.literature_id
            LEFT JOIN henry_notes hn ON hn.henry_id = h.id
            LEFT JOIN notes n ON n.id = hn.notes_id
            GROUP BY h.id
            ORDER BY h.id
            """
        )
        columns = [item[0] for item in cursor.description]
        raw_rows = cursor.fetchall()

    eligible_rows = []
    excluded_seawater = 0
    excluded_wrong = 0
    excluded_missing_cas = 0
    for values in raw_rows:
        row = dict(zip(columns, values))
        note_labels = str(row.get("note_labels") or "").lower().split("|")
        if any("seawater" in label for label in note_labels):
            excluded_seawater += 1
            continue
        htype = nullable_text(row.get("htype"))
        if htype == "W":
            excluded_wrong += 1
            continue
        if htype not in QUALITY_TIERS:
            raise ValueError(f"Unexpected htype {htype!r} in source row {row['id']!r}")
        if nullable_text(row.get("casrn")) is None:
            excluded_missing_cas += 1
            continue
        eligible_rows.append(row)

    if path.resolve() == SOURCE.resolve():
        expected = {
            "archive SHA-256": (digest, SOURCE_ARCHIVE_SHA256),
            "Henry row count": (upstream_henry_rows, EXPECTED_HENRY_ROWS),
            "species row count": (upstream_species_rows, EXPECTED_SPECIES_ROWS),
            "seawater row count": (excluded_seawater, EXPECTED_SEAWATER_ROWS),
        }
        mismatches = [
            f"{label}: got {actual!r}, expected {wanted!r}"
            for label, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if mismatches:
            raise ValueError("Henry 5.0 source validation failed: " + "; ".join(mismatches))

    return SourceDataset(
        rows=tuple(eligible_rows),
        upstream_henry_rows=upstream_henry_rows,
        upstream_species_rows=upstream_species_rows,
        excluded_seawater_rows=excluded_seawater,
        excluded_wrong_rows=excluded_wrong,
        excluded_missing_cas_rows=excluded_missing_cas,
        archive_sha256=digest,
    )


def source_records(dataset: SourceDataset) -> list[tuple[object, ...]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in dataset.rows:
        groups[str(row["casrn"]).strip()].append(row)
    return [build_record(cas, rows) for cas, rows in sorted(groups.items())]


def _metadata(
    dataset: SourceDataset,
    record_count: int,
    temperature_source_count: int,
    temperature_correlation_count: int,
    temperature_rejection_count: int,
) -> dict[str, str]:
    metadata = {
        "schema_version": "6",
        "source_name": "Sander Henry's law constants",
        "source_version": SOURCE_VERSION,
        "source_url": SOURCE_URL,
        "source_doi": SOURCE_DOI,
        "source_archive_sha256": dataset.archive_sha256,
        "upstream_henry_rows": str(dataset.upstream_henry_rows),
        "upstream_species_rows": str(dataset.upstream_species_rows),
        "excluded_seawater_rows": str(dataset.excluded_seawater_rows),
        "excluded_wrong_rows": str(dataset.excluded_wrong_rows),
        "excluded_missing_cas_rows": str(dataset.excluded_missing_cas_rows),
        "eligible_source_rows": str(len(dataset.rows)),
        "aggregated_cas_records": str(record_count),
        "aggregation_policy": (
            "best_reliability_band;independent-reference-weighted;"
            "H_geometric_mean;B_arithmetic_mean;MAD_outlier_filter"
        ),
        "H_outlier_rule": "abs(ln(H)-weighted_median)>max(ln(3),3.5*1.4826*MAD)",
        "B_outlier_rule": "abs(B-weighted_median)>max(1000K,35%center,3.5*1.4826*MAD)",
        "quality_score_policy": (
            "60% strongest-source quality + 25% small-sample-adjusted agreement + "
            "15% monotonic evidence support;L reviews clustered by lineage with "
            "1,1/2,1/4,... family contributions;"
            "primary-evidence grade gates;outlier conflict penalty"
        ),
        "Vinf_source": VINF_SOURCE,
        "Vinf_source_doi": VINF_DOI,
        "Vinf_reference_temperature_K": "298.15",
        "Vinf_fallback_cm3_per_mol": "10.74 + 0.2683 * Vc_cm3_per_mol",
        "temperature_source_correlations": str(temperature_source_count),
        "resolved_temperature_correlations": str(temperature_correlation_count),
        "rejected_temperature_correlations": str(temperature_rejection_count),
        "temperature_correlation_models": (
            "iapws_g7_04;chapoy_dippr101;brockbank_dippr101;"
            "exp_a_b_over_t_c_log_t"
        ),
        "temperature_compatibility_policy": (
            "abs(raw_Hcp_298/resolved_Hcp_298-1)<=25%;"
            "abs(raw_B-resolved_B)<=max(500K,25%abs(resolved_B))"
        ),
    }
    for field in ("h", "b"):
        counts: dict[str, int] = defaultdict(int)
        for row in dataset.rows:
            counts[str(row.get(f"_{field}_status") or "not_evaluated")] += 1
        for status, count in sorted(counts.items()):
            metadata[f"{field.upper()}_status:{status}"] = str(count)
    return metadata


def _source_value_record(row: dict[str, object]) -> tuple[object, ...]:
    h_candidate, _h_candidate_status = _candidate(row, "h")
    b_candidate, _b_candidate_status = _candidate(row, "b")
    return (
        int(row["id"]),
        str(row["casrn"]).strip(),
        nullable_text(row.get("hominus")),
        nullable_text(row.get("mindhr")),
        h_candidate.value if h_candidate else None,
        b_candidate.value if b_candidate else None,
        str(row["htype"]),
        nullable_text(row.get("literature_id")),
        (
            _review_family_key(row)
            if str(row.get("htype") or "") == "L"
            else None
        ),
        nullable_text(row.get("note_labels")),
        str(row.get("_h_status") or "not_evaluated"),
        str(row.get("_b_status") or "not_evaluated"),
        h_candidate.weight if h_candidate else None,
        b_candidate.weight if b_candidate else None,
    )


def _write_database_file(
    records: list[tuple[object, ...]],
    dataset: SourceDataset,
    output: Path,
) -> None:
    temperature_sources = source_temperature_correlations(dataset)
    temperature_correlations, temperature_rejections = resolved_temperature_correlations(
        records, dataset, temperature_sources
    )
    with sqlite3.connect(output) as connection:
        connection.execute(
            f"""
            CREATE TABLE {TABLE_NAME} (
                CAS TEXT PRIMARY KEY,
                iupac TEXT,
                formula TEXT,
                trivial TEXT,
                casrn TEXT,
                inchikey TEXT,
                H REAL,
                B REAL,
                quality_h TEXT,
                quality_b TEXT,
                quality_score_h REAL,
                quality_score_b REAL,
                agreement_score_h REAL,
                agreement_score_b REAL,
                raw_agreement_score_h REAL,
                raw_agreement_score_b REAL,
                effective_evidence_h REAL NOT NULL,
                effective_evidence_b REAL NOT NULL,
                direct_evidence_h REAL NOT NULL,
                direct_evidence_b REAL NOT NULL,
                review_evidence_h REAL NOT NULL,
                review_evidence_b REAL NOT NULL,
                strongest_source_quality_h REAL,
                strongest_source_quality_b REAL,
                H_more_temperature_dependence INTEGER NOT NULL,
                B_more_temperature_dependence INTEGER NOT NULL,
                source_count_h INTEGER NOT NULL,
                source_count_b INTEGER NOT NULL,
                paired_B_fraction REAL,
                id_start INTEGER NOT NULL,
                id_end INTEGER NOT NULL,
                H_source_ids TEXT,
                B_source_ids TEXT,
                H_outlier_ids TEXT,
                B_outlier_ids TEXT
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE {PARTIAL_MOLAR_VOLUME_TABLE} (
                CAS TEXT PRIMARY KEY,
                component TEXT NOT NULL,
                Vinf_298_cm3_per_mol REAL NOT NULL,
                uncertainty_cm3_per_mol REAL NOT NULL,
                reference_temperature_K REAL NOT NULL,
                source TEXT NOT NULL,
                doi TEXT NOT NULL,
                method TEXT NOT NULL
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE {TEMPERATURE_SOURCE_TABLE} (
                source_row_id INTEGER PRIMARY KEY,
                CAS TEXT NOT NULL,
                model TEXT NOT NULL,
                A REAL NOT NULL,
                B REAL NOT NULL,
                C REAL NOT NULL,
                Tmin_K REAL,
                Tmax_K REAL,
                range_source TEXT,
                htype TEXT NOT NULL,
                literature_key TEXT,
                selected_for_B INTEGER NOT NULL,
                note_labels TEXT
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE {TEMPERATURE_CORRELATION_TABLE} (
                CAS TEXT PRIMARY KEY,
                component TEXT NOT NULL,
                model TEXT NOT NULL,
                A REAL NOT NULL,
                B REAL NOT NULL,
                C REAL NOT NULL,
                D REAL NOT NULL,
                E REAL NOT NULL,
                Tmin_K REAL NOT NULL,
                Tmax_K REAL NOT NULL,
                range_source TEXT NOT NULL,
                rms_lnH REAL,
                fit_quality REAL,
                uncertainty_upper_percent REAL,
                normalize_to_reference INTEGER NOT NULL,
                raw_reference_value_difference_percent REAL,
                raw_reference_slope_difference_K REAL,
                raw_reference_slope_difference_percent REAL,
                source TEXT NOT NULL,
                doi TEXT,
                source_row_ids TEXT,
                source_literature_keys TEXT,
                curvature_outlier_ids TEXT,
                notes TEXT
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE {TEMPERATURE_REJECTION_TABLE} (
                CAS TEXT NOT NULL,
                component TEXT NOT NULL,
                model TEXT NOT NULL,
                reason TEXT NOT NULL,
                raw_reference_value_difference_percent REAL,
                raw_reference_slope_difference_K REAL,
                raw_reference_slope_difference_percent REAL,
                value_limit_percent REAL NOT NULL,
                slope_limit_K REAL NOT NULL,
                source TEXT NOT NULL,
                doi TEXT,
                PRIMARY KEY (CAS, model)
            )
            """
        )
        connection.executemany(
            f"""
            INSERT INTO {TABLE_NAME} (
                CAS, iupac, formula, trivial, casrn, inchikey,
                H, B, quality_h, quality_b,
                quality_score_h, quality_score_b,
                agreement_score_h, agreement_score_b,
                raw_agreement_score_h, raw_agreement_score_b,
                effective_evidence_h, effective_evidence_b,
                direct_evidence_h, direct_evidence_b,
                review_evidence_h, review_evidence_b,
                strongest_source_quality_h, strongest_source_quality_b,
                H_more_temperature_dependence, B_more_temperature_dependence,
                source_count_h, source_count_b, paired_B_fraction,
                id_start, id_end, H_source_ids, B_source_ids,
                H_outlier_ids, B_outlier_ids
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        connection.execute(
            f"""
            CREATE TABLE {SOURCE_VALUES_TABLE} (
                id INTEGER PRIMARY KEY,
                CAS TEXT NOT NULL,
                H_raw TEXT,
                B_raw TEXT,
                H REAL,
                B REAL,
                htype TEXT NOT NULL,
                literature_key TEXT,
                review_family TEXT,
                note_labels TEXT,
                H_status TEXT NOT NULL,
                B_status TEXT NOT NULL,
                H_weight REAL,
                B_weight REAL
            )
            """
        )
        connection.executemany(
            f"""
            INSERT INTO {SOURCE_VALUES_TABLE} (
                id, CAS, H_raw, B_raw, H, B, htype, literature_key, review_family, note_labels,
                H_status, B_status, H_weight, B_weight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_source_value_record(row) for row in dataset.rows],
        )
        connection.execute(
            f"CREATE TABLE {METADATA_TABLE} (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.executemany(
            f"INSERT INTO {METADATA_TABLE} (key, value) VALUES (?, ?)",
            sorted(_metadata(
                dataset,
                len(records),
                len(temperature_sources),
                len(temperature_correlations),
                len(temperature_rejections),
            ).items()),
        )
        connection.executemany(
            f"""
            INSERT INTO {PARTIAL_MOLAR_VOLUME_TABLE} (
                CAS, component, Vinf_298_cm3_per_mol,
                uncertainty_cm3_per_mol, reference_temperature_K,
                source, doi, method
            ) VALUES (?, ?, ?, ?, 298.15, ?, ?, 'measured')
            """,
            [
                (cas, component, value, uncertainty, VINF_SOURCE, VINF_DOI)
                for cas, component, value, uncertainty in VINF_298_RECORDS
            ],
        )
        connection.executemany(
            f"""
            INSERT INTO {TEMPERATURE_SOURCE_TABLE} (
                source_row_id, CAS, model, A, B, C, Tmin_K, Tmax_K,
                range_source, htype, literature_key, selected_for_B, note_labels
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["source_row_id"], row["CAS"], row["model"],
                    row["A"], row["B"], row["C"], row["Tmin_K"], row["Tmax_K"],
                    row["range_source"], row["htype"], row["literature_key"],
                    row["selected_for_B"], row["note_labels"],
                )
                for row in temperature_sources
            ],
        )
        connection.executemany(
            f"""
            INSERT INTO {TEMPERATURE_CORRELATION_TABLE} (
                CAS, component, model, A, B, C, D, E, Tmin_K, Tmax_K,
                range_source, rms_lnH, fit_quality, uncertainty_upper_percent,
                normalize_to_reference, raw_reference_value_difference_percent,
                raw_reference_slope_difference_K, raw_reference_slope_difference_percent,
                source, doi,
                source_row_ids, source_literature_keys, curvature_outlier_ids, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["CAS"], row["component"], row["model"],
                    row["A"], row["B"], row["C"], row["D"], row["E"],
                    row["Tmin_K"], row["Tmax_K"], row["range_source"],
                    row["rms_lnH"], row["fit_quality"],
                    row["uncertainty_upper_percent"], row["normalize_to_reference"],
                    row["raw_reference_value_difference_percent"],
                    row["raw_reference_slope_difference_K"],
                    row["raw_reference_slope_difference_percent"],
                    row["source"], row["doi"],
                    row["source_row_ids"],
                    row["source_literature_keys"], row["curvature_outlier_ids"],
                    row["notes"],
                )
                for row in temperature_correlations
            ],
        )
        connection.executemany(
            f"""
            INSERT INTO {TEMPERATURE_REJECTION_TABLE} (
                CAS, component, model, reason,
                raw_reference_value_difference_percent,
                raw_reference_slope_difference_K,
                raw_reference_slope_difference_percent,
                value_limit_percent, slope_limit_K, source, doi
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["CAS"], row["component"], row["model"], row["reason"],
                    row["raw_reference_value_difference_percent"],
                    row["raw_reference_slope_difference_K"],
                    row["raw_reference_slope_difference_percent"],
                    TEMPERATURE_VALUE_COMPATIBILITY_PERCENT,
                    row["slope_limit_K"], row["source"], row["doi"],
                )
                for row in temperature_rejections
            ],
        )
        connection.execute(
            f"""
            CREATE UNIQUE INDEX idx_{TABLE_NAME}_inchikey_unique
            ON {TABLE_NAME} (inchikey)
            WHERE inchikey IS NOT NULL
            """
        )
        connection.execute(f"CREATE INDEX idx_{TABLE_NAME}_formula ON {TABLE_NAME} (formula)")
        connection.execute(f"CREATE INDEX idx_{TABLE_NAME}_quality_h ON {TABLE_NAME} (quality_h)")
        connection.execute(f"CREATE INDEX idx_{TABLE_NAME}_quality_b ON {TABLE_NAME} (quality_b)")
        connection.execute(f"CREATE INDEX idx_{SOURCE_VALUES_TABLE}_cas ON {SOURCE_VALUES_TABLE} (CAS)")
        connection.execute(
            f"CREATE INDEX idx_{TEMPERATURE_SOURCE_TABLE}_cas "
            f"ON {TEMPERATURE_SOURCE_TABLE} (CAS)"
        )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"Generated Henry database failed integrity check: {integrity}")


def write_database(
    records: list[tuple[object, ...]],
    dataset: SourceDataset,
    output: Path,
) -> None:
    """Write and validate a replacement before atomically publishing it."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        _write_database_file(records, dataset, temporary_path)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help=f"source archive (default: {SOURCE})")
    parser.add_argument("--output", type=Path, default=OUTPUT, help=f"output SQLite path (default: {OUTPUT})")
    args = parser.parse_args()

    dataset = load_source_dataset(args.source)
    records = source_records(dataset)
    write_database(records, dataset, args.output)
    print(
        f"Wrote {len(records)} CAS records to {args.output} from Henry {SOURCE_VERSION}; "
        f"excluded {dataset.excluded_seawater_rows} seawater rows"
    )


if __name__ == "__main__":
    main()
