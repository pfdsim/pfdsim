#!/usr/bin/env python3
"""Extract selected property tables from Perry's Chemical Engineers' Handbook.

The Perry tables are correlation tables, not just scalar property lookups. This
script preserves table IDs, equation IDs, coefficients, units, and fit ranges so
property resolution code can decide how to evaluate and prioritize them later.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOURCE_DATA_DIR = DATA_DIR / "source"
PDF = SOURCE_DATA_DIR / (
    "perrys-chemical-engineers-handjob-9th-edition-2021-9nbsped-"
    "9780071834094-0071834095-9780071834087-0071834087_compress.pdf"
)
OUTPUT = DATA_DIR / "perry_properties.json"
FUSION_OUTPUT = DATA_DIR / "perry_heat_of_fusion.json"

CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d+)*(?:\.\d*)?|\.\d+)(?:E[-+]?\d+)?", re.I)
FORMULA_FRAGMENT_RE = re.compile(r"(?:[A-Z][a-z]?|\d+)+")
FUSION_ROW_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9α-ωΑ-Ω(][A-Za-z0-9α-ωΑ-Ω ,.'()/-]*?)\s+"
    r"(?P<formula>(?:[A-Z][a-z]?\s*\d*)+)\s+"
    r"(?P<tm>[-+]?\d+(?:\.\d+)?)(?P<tm_note>\(\?\))?"
    r"(?:,\s*(?P<secondary_tm>[-+]?\d+(?:\.\d+)?))?\s+"
    r"(?P<hfus>\d+(?:\.\d+)?)\s*$"
)
FUSION_MISSING_TM_ROW_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9α-ωΑ-Ω(][A-Za-z0-9α-ωΑ-Ω ,.'()/-]*?)\s+"
    r"(?P<formula>(?:[A-Z][a-z]?\s*\d*)+)\s{4,}"
    r"(?P<hfus>\d+(?:\.\d+)?)\s*$"
)
GREEK_NAME_REPLACEMENTS = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "Α": "alpha",
    "Β": "beta",
    "Γ": "gamma",
    "Δ": "delta",
}
ATOMIC_WEIGHTS = {
    "H": 1.00794,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984032,
    "P": 30.973761,
    "S": 32.065,
    "Cl": 35.453,
    "Br": 79.904,
    "I": 126.90447,
}
VAPOR_PRESSURE_COEFFICIENT_CORRECTIONS = {
    ("2-8", "60-35-5"): [125.81, -12376.0, -14.589, 5.0824e-06, 2.0],
}
CRITICAL_CONSTANT_CORRECTIONS: dict[str, dict[str, Any]] = {
    "107-10-8": {
        "Vc_m3_per_kmol": 0.23,
        "Zc": 0.2639,
        "source": "chemicals (CRC)",
        "reason": "Perry Vc unreliable for this amine (original implied Zc=0.298)",
    },
    "107-15-3": {
        "Vc_m3_per_kmol": 0.204,
        "Zc": 0.2603,
        "source": "chemicals (CRC)",
        "reason": "Perry Vc unreliable for this amine (original implied Zc=0.337)",
    },
    "108-18-9": {
        "Vc_m3_per_kmol": 0.407,
        "Zc": 0.2995,
        "source": "chemicals (CRC)",
        "reason": "Perry Vc unreliable for this amine (original implied Zc=0.308)",
    },
    "74-89-5": {
        "Vc_m3_per_kmol": 0.141,
        "Zc": 0.2942,
        "source": "chemicals (IUPAC)",
        "reason": "Perry Vc unreliable for this amine (original implied Zc=0.321)",
    },
    "75-04-7": {
        "Vc_m3_per_kmol": 0.18,
        "Zc": 0.2667,
        "source": "chemicals (IUPAC)",
        "reason": "Perry Vc unreliable for this amine (original implied Zc=0.307)",
    },
}

# Perry Table 2-68 contains a handful of printed name/formula errors.  Keep
# these corrections explicit and auditable rather than relaxing identity
# matching for the entire table.
FUSION_SOURCE_ERRATA: dict[tuple[str, str], dict[str, str]] = {
    ("3methylpentane", "C8H18"): {
        "table_name": "3-Methylheptane",
        "note": (
            "Perry prints 3-Methylpentane with formula C8H18 between the "
            "2-, 3-, and 4-methylheptane rows; corrected to 3-Methylheptane"
        ),
    },
    ("camphene", "C10H12"): {
        "formula": "C10H16",
        "note": "Perry formula C10H12 corrected to the molecular formula C10H16",
    },
    ("benzoicacid", "C7H8O2"): {
        "formula": "C7H6O2",
        "note": "Perry formula C7H8O2 corrected to the molecular formula C7H6O2",
    },
    ("stearicacid", "C18H30O2"): {
        "formula": "C18H36O2",
        "note": "Perry formula C18H30O2 corrected to the molecular formula C18H36O2",
    },
    ("glutaricacid", "C6H8O4"): {
        "formula": "C5H8O4",
        "note": "Perry formula C6H8O4 corrected to the molecular formula C5H8O4",
    },
}

# Historical names and stereochemical materials absent from the installed
# local identity catalog.  Values are intentionally CAS/name identities only;
# all thermophysical numbers continue to come from Perry.
FUSION_IDENTITY_OVERRIDES: dict[str, dict[str, str]] = {
    "allocinnamicacid": {
        "cas": "102-94-3",
        "name": "cis-cinnamic acid",
        "formula": "C9H8O2",
    },
    "carvoximed": {
        "cas": "2051-55-0",
        "name": "d-carvoxime",
        "formula": "C10H15NO",
    },
    "carvoximel": {
        "cas": "80124-30-7",
        "name": "l-carvoxime",
        "formula": "C10H15NO",
    },
    "carvoximedl": {
        "cas": "55658-55-4",
        "name": "dl-carvoxime",
        "formula": "C10H15NO",
    },
    "bromolhydrate": {
        "cas": "507-42-6",
        "name": "bromal hydrate",
        "formula": "C2H3Br3O2",
    },
    "bromochlorbenzeneo": {
        "cas": "694-80-4",
        "name": "1-bromo-2-chlorobenzene",
        "formula": "C6H4BrCl",
    },
    "bromochlorbenzenem": {
        "cas": "108-37-2",
        "name": "1-bromo-3-chlorobenzene",
        "formula": "C6H4BrCl",
    },
    "bromochlorbenzenep": {
        "cas": "106-39-8",
        "name": "1-bromo-4-chlorobenzene",
        "formula": "C6H4BrCl",
    },
    "dibromophenol24": {
        "cas": "615-58-7",
        "name": "2,4-dibromophenol",
        "formula": "C6H4Br2O",
    },
    "diiodobenzeneo": {
        "cas": "615-42-9",
        "name": "1,2-diiodobenzene",
        "formula": "C6H4I2",
    },
    "diiodobenzenem": {
        "cas": "626-00-6",
        "name": "1,3-diiodobenzene",
        "formula": "C6H4I2",
    },
    "diiodobenzenep": {
        "cas": "624-38-4",
        "name": "1,4-diiodobenzene",
        "formula": "C6H4I2",
    },
    "dinitrotoluene24": {
        "cas": "121-14-2",
        "name": "2,4-dinitrotoluene",
        "formula": "C7H6N2O4",
    },
    "tribromophenol246": {
        "cas": "118-79-6",
        "name": "2,4,6-tribromophenol",
        "formula": "C6H3Br3O",
    },
    "trinitrotoluene246": {
        "cas": "118-96-7",
        "name": "2,4,6-trinitrotoluene",
        "formula": "C7H5N3O6",
    },
    "dimethyltartratedl": {
        "cas": "608-69-5",
        "name": "dimethyl dl-tartrate",
        "formula": "C6H10O6",
    },
    "dimethyltartrated": {
        "cas": "13171-64-7",
        "name": "dimethyl d-tartrate",
        "formula": "C6H10O6",
    },
    "dimethylpyrone": {
        "cas": "1004-36-0",
        "name": "2,6-dimethyl-4-pyrone",
        "formula": "C7H8O2",
    },
    "elaidicacid": {
        "cas": "112-79-8",
        "name": "elaidic acid",
        "formula": "C18H34O2",
    },
    "tetrachloroxyleneo": {
        "cas": "877-08-7",
        "name": "tetrachloro-o-xylene",
        "formula": "C8H6Cl4",
    },
    "tetrachloroxylenep": {
        "cas": "877-10-1",
        "name": "tetrachloro-p-xylene",
        "formula": "C8H6Cl4",
    },
    "xylenedibromideo": {
        "cas": "91-13-4",
        "name": "o-xylylene dibromide",
        "formula": "C8H8Br2",
    },
    "xylenedibromidem": {
        "cas": "626-15-3",
        "name": "m-xylylene dibromide",
        "formula": "C8H8Br2",
    },
    "xylenedichlorideo": {
        "cas": "612-12-4",
        "name": "o-xylylene dichloride",
        "formula": "C8H8Cl2",
    },
    "xylenedichloridem": {
        "cas": "626-16-4",
        "name": "m-xylylene dichloride",
        "formula": "C8H8Cl2",
    },
    "xylenedichloridep": {
        "cas": "623-25-6",
        "name": "p-xylylene dichloride",
        "formula": "C8H8Cl2",
    },
}

FUSION_BARE_NAME_CONTINUATIONS = frozenset({
    "alcohol",
    "anhydride",
    "dichloride (o-)",
    "ether",
    "fumarate",
    "hydrate",
    "oxalate",
    "phenylpropiolate",
    "pyrone",
    "succinate",
})


@dataclass(frozen=True)
class TableSpec:
    key: str
    table: str
    pages: tuple[int, int]
    has_equation_id: bool
    kind: str
    title: str
    units: dict[str, str]
    coefficient_multipliers: tuple[float, ...] | None = None
    endpoint_value_multiplier: float = 1.0


TABLES = [
    TableSpec(
        key="vapor_pressure",
        table="2-8",
        pages=(88, 94),
        has_equation_id=False,
        kind="vapor_pressure",
        title="Vapor pressure of inorganic and organic liquids",
        units={
            "equation": "ln(P/Pa) = C1 + C2/T + C3*ln(T) + C4*T**C5",
            "temperature": "K",
            "pressure": "Pa",
        },
    ),
    TableSpec(
        key="liquid_density",
        table="2-32",
        pages=(128, 134),
        has_equation_id=True,
        kind="generic_correlation",
        title="Densities of inorganic and organic liquids",
        units={"density": "mol/dm^3", "temperature": "K"},
    ),
    TableSpec(
        key="liquid_heat_capacity",
        table="2-72",
        pages=(172, 178),
        has_equation_id=True,
        kind="generic_correlation",
        title="Heat capacities of inorganic and organic liquids",
        units={
            "heat_capacity": "J/(kmol*K)",
            "temperature": "K",
            "printed_endpoint_values": "Cp * 1E-05",
        },
        endpoint_value_multiplier=1.0e5,
    ),
    TableSpec(
        key="ideal_gas_heat_capacity_polynomial",
        table="2-74",
        pages=(182, 183),
        has_equation_id=False,
        kind="generic_correlation",
        title="Ideal-gas heat capacity fit to a polynomial",
        units={"heat_capacity": "J/(kmol*K)", "temperature": "K"},
    ),
    TableSpec(
        key="ideal_gas_heat_capacity_hyperbolic",
        table="2-75",
        pages=(184, 190),
        has_equation_id=False,
        kind="generic_correlation",
        title="Ideal-gas heat capacity fit to hyperbolic functions",
        units={
            "heat_capacity": "J/(kmol*K)",
            "temperature": "K",
            "printed_coefficients": "C1*1E-05, C2*1E-05, C3*1E-03, C4*1E-05, C5",
            "printed_endpoint_values": "Cp * 1E-05",
        },
        coefficient_multipliers=(1.0e5, 1.0e5, 1.0e3, 1.0e5, 1.0),
        endpoint_value_multiplier=1.0e5,
    ),
    TableSpec(
        key="critical_constants",
        table="2-106",
        pages=(218, 224),
        has_equation_id=False,
        kind="critical",
        title="Critical constants and acentric factors",
        units={
            "critical_temperature": "K",
            "critical_pressure": "MPa",
            "critical_volume": "m^3/kmol",
            "critical_compressibility": "dimensionless",
            "acentric_factor": "dimensionless",
        },
    ),
    TableSpec(
        key="formation_properties",
        table="2-95",
        pages=(202, 208),
        has_equation_id=False,
        kind="formation",
        title="Ideal-gas formation properties and entropy at 298.15 K",
        units={
            "enthalpy_of_formation": "J/kmol",
            "gibbs_energy_of_formation": "J/kmol",
            "entropy": "J/(kmol*K)",
            "net_enthalpy_of_combustion": "J/kmol",
        },
    ),
    TableSpec(
        key="heat_of_vaporization",
        table="2-69",
        pages=(155, 162),
        has_equation_id=False,
        kind="generic_correlation",
        title="Heats of vaporization of inorganic and organic liquids",
        units={
            "equation": "DeltaHv = C1*(1 - Tr)**(C2 + C3*Tr + C4*Tr**2)",
            "enthalpy": "J/kmol",
            "temperature": "K",
            "printed_coefficients": "C1*1E-07, C2, C3, C4",
            "printed_endpoint_values": "DeltaHv * 1E-07",
        },
        coefficient_multipliers=(1.0e7, 1.0, 1.0, 1.0),
        endpoint_value_multiplier=1.0e7,
    ),
    TableSpec(
        key="vapor_viscosity",
        table="2-138",
        pages=(302, 308),
        has_equation_id=False,
        kind="generic_correlation",
        title="Vapor viscosity of inorganic and organic substances",
        units={"viscosity": "Pa*s", "temperature": "K"},
    ),
    TableSpec(
        key="liquid_viscosity",
        table="2-139",
        pages=(309, 316),
        has_equation_id=True,
        kind="generic_correlation",
        title="Viscosity of inorganic and organic liquids",
        units={"viscosity": "Pa*s", "temperature": "K"},
    ),
]

FUSION_TABLE = TableSpec(
    key="heat_of_fusion",
    table="2-68",
    pages=(153, 155),
    has_equation_id=False,
    kind="organic_heat_of_fusion",
    title="Heats of fusion of organic compounds",
    units={
        "melting_point": "degC",
        "heat_of_fusion": "cal/g",
    },
)


def normalize_text(text: str) -> str:
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00b7", "*").replace("\u2219", "*")
    return text


def parse_number(token: str) -> float:
    return float(token.replace(",", ""))


def numbers_after(text: str) -> list[float]:
    return [parse_number(match.group(0)) for match in NUMBER_RE.finditer(normalize_text(text))]


def normalize_key(text: object) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def formula_composition(formula: object) -> dict[str, int] | None:
    """Return elemental counts for the plain formulas printed by Perry."""
    value = formula_text(formula)
    if not value:
        return None
    counts: dict[str, int] = {}
    pos = 0
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", value):
        if match.start() != pos:
            return None
        element = match.group(1)
        if element not in ATOMIC_WEIGHTS:
            return None
        counts[element] = counts.get(element, 0) + int(match.group(2) or "1")
        pos = match.end()
    if pos != len(value):
        return None
    return counts or None


def normalize_formula(formula: object) -> str:
    """Canonicalize formulas elementally, including repeated atom tokens."""
    counts = formula_composition(formula)
    if counts is None:
        return normalize_key(str(formula).replace("*", "").replace("·", ""))
    elements = []
    if "C" in counts:
        elements.append("C")
    if "H" in counts:
        elements.append("H")
    elements.extend(sorted(element for element in counts if element not in {"C", "H"}))
    return "".join(
        element + (str(counts[element]) if counts[element] != 1 else "")
        for element in elements
    )


def formula_text(formula: object) -> str:
    return "".join(str(formula).replace("*", "").replace("·", "").split())


def is_formula_fragment(token: str) -> bool:
    return bool(FORMULA_FRAGMENT_RE.fullmatch(token))


def split_formula_suffix(parts: list[str]) -> tuple[str, str] | None:
    if len(parts) < 2:
        return None

    formula_parts: list[str] = []
    idx = len(parts) - 1
    while idx >= 0 and is_formula_fragment(parts[idx]):
        formula_parts.insert(0, parts[idx])
        idx -= 1

    if not formula_parts:
        formula_parts = [parts[-1]]
        idx = len(parts) - 2

    name = " ".join(parts[: idx + 1]).strip()
    formula = formula_text("".join(formula_parts))
    if not name or not formula:
        return None
    return name, formula


def comparable_name(text: object) -> str:
    value = str(text)
    for source, target in GREEK_NAME_REPLACEMENTS.items():
        value = value.replace(source, target)
    return value.lower()


def name_tokens(text: object) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", comparable_name(text)))


def names_compatible(source: object, candidate: object) -> bool:
    source_key = normalize_key(comparable_name(source))
    candidate_key = normalize_key(comparable_name(candidate))
    if not source_key or not candidate_key:
        return False
    if source_key == candidate_key:
        return True

    source_tokens = name_tokens(source)
    candidate_tokens = name_tokens(candidate)
    if len(source_tokens) < 2 or len(candidate_tokens) < 2:
        return False
    return source_tokens == candidate_tokens


def formula_mw(formula: str) -> float | None:
    """Return a simple molecular-weight estimate for plain organic formulas."""
    counts = formula_composition(formula)
    if not counts:
        return None
    return sum(ATOMIC_WEIGHTS[element] * count for element, count in counts.items())


def fusion_column_boundary(page: str) -> int | None:
    """Locate the start of the second visual column in a Table 2-68 page."""
    for line in page.splitlines():
        for heading in ("Nonhydrocarbon compounds", "Hydrocarbon compounds"):
            starts = [match.start() for match in re.finditer(re.escape(heading), line)]
            if len(starts) >= 2:
                # The repeated heading is indented within each fixed-width
                # column and separated by Perry's three-space table gutter.
                # Removing that gutter gives the actual fixed column width.
                return starts[1] - starts[0] - 3
    return None


def fusion_name_qualifier(fragment: str) -> str | None:
    """Normalize a suffix-only positional, stereochemical, or form label."""
    value = comparable_name(fragment).strip().replace(" ", "")
    value = value.strip("()")
    value = value.rstrip("-")
    aliases = {
        "o": "o",
        "m": "m",
        "p": "p",
        "t": "t",
        "n": "n",
        "alpha": "alpha",
        "beta": "beta",
        "gamma": "gamma",
        "d": "d",
        "l": "l",
        "dl": "dl",
        "cis": "cis",
        "trans": "trans",
    }
    return aliases.get(value)


def fusion_name_without_trailing_qualifiers(name: str) -> str:
    value = " ".join(str(name).split())
    while re.search(
        r"\s*\((?:[omptn]|alpha|beta|gamma|d|l|dl|cis|trans)-?\)\s*$",
        comparable_name(value),
    ):
        value = re.sub(r"\s*\([^()]+\)\s*$", "", value).strip()
    return value


def reconstruct_fusion_name(fragment: str, previous_name: str | None) -> tuple[str, str | None]:
    """Reconstruct one abbreviated Table 2-68 name within a visual column."""
    fragment = " ".join(fragment.split())
    qualifier = fusion_name_qualifier(fragment)
    if qualifier is not None and previous_name:
        base = fusion_name_without_trailing_qualifiers(previous_name)
        return f"{base} ({qualifier}-)", fragment

    qualifier_fragments = re.findall(r"\(([^()]*)\)", fragment)
    if qualifier_fragments and previous_name:
        qualifiers = [fusion_name_qualifier(item) for item in qualifier_fragments]
        leftover = re.sub(r"\([^()]*\)", "", fragment).strip()
        if not leftover and all(qualifiers):
            base = fusion_name_without_trailing_qualifiers(previous_name)
            suffix = " ".join(f"({item}-)" for item in qualifiers)
            return f"{base} {suffix}", fragment

    if comparable_name(fragment) in FUSION_BARE_NAME_CONTINUATIONS and previous_name:
        previous_base = fusion_name_without_trailing_qualifiers(previous_name)
        if comparable_name(fragment) == "anhydride" and comparable_name(previous_base).endswith("acid"):
            prefix = previous_base
        else:
            prefix = previous_base.split()[0]
        return f"{prefix} {fragment}", fragment
    return fragment, None


def apply_fusion_source_errata(table_name: str, formula: str) -> tuple[str, str, str | None]:
    correction = FUSION_SOURCE_ERRATA.get((normalize_key(table_name), formula_text(formula)))
    if not correction:
        return table_name, formula_text(formula), None
    return (
        correction.get("table_name", table_name),
        formula_text(correction.get("formula", formula)),
        correction["note"],
    )


def fusion_record_from_fields(
    *,
    table_name: str,
    formula: str,
    tm_text: str | None,
    hfus_text: str,
    secondary_tm_text: str | None,
    tm_note: str | None,
    line_number: int,
    raw_line: str,
    column: int,
    source_name_fragment: str | None = None,
) -> dict[str, Any] | None:
    source_table_name = table_name
    source_formula = formula_text(formula)
    table_name, formula, correction_note = apply_fusion_source_errata(
        table_name,
        source_formula,
    )
    if normalize_key(table_name) in {"hydrocarboncompounds", "nonhydrocarboncompounds"}:
        return None

    mw = formula_mw(formula)
    if not mw:
        return None
    hfus_cal_per_g = float(hfus_text)
    tm_c = float(tm_text) if tm_text is not None else None
    record: dict[str, Any] = {
        "source_table": FUSION_TABLE.table,
        "table_name": table_name,
        "formula": formula,
        "Tm_K": tm_c + 273.15 if tm_c is not None else None,
        "Tm_C": tm_c,
        "Hfus_cal_per_g": hfus_cal_per_g,
        "Hfus_kJ_per_mol": hfus_cal_per_g * 4.184 * mw / 1000.0,
        "molecular_weight_estimate": mw,
        "source_line_number": line_number,
        "source_column": column,
        "source_line": raw_line,
    }
    if source_name_fragment is not None:
        record["source_name_fragment"] = source_name_fragment
    if table_name != source_table_name:
        record["source_table_name"] = source_table_name
    if formula != source_formula:
        record["source_formula"] = source_formula
    if correction_note:
        record["source_correction"] = correction_note
    if tm_note:
        record["Tm_note"] = tm_note
    if secondary_tm_text:
        secondary_tm_c = float(secondary_tm_text)
        record["secondary_Tm_C"] = secondary_tm_c
        record["secondary_Tm_K"] = secondary_tm_c + 273.15
    qualifiers = re.findall(r"\(([^()]*)\)", comparable_name(table_name))
    if qualifiers:
        record["source_name_qualifiers"] = [item.strip() for item in qualifiers]
    return record


def fusion_records_from_layout(text: str) -> list[dict[str, Any]]:
    """Parse both visual columns while retaining abbreviated-name context."""
    records: list[dict[str, Any]] = []
    line_number = 0
    for page_number, page in enumerate(text.split("\f"), 1):
        boundary = fusion_column_boundary(page)
        previous_names: dict[int, str | None] = {1: None, 2: None}
        for raw_line in page.splitlines():
            line_number += 1
            columns = (
                (raw_line,) if boundary is None
                else (raw_line[:boundary], raw_line[boundary:])
            )
            for column_number, column_text in enumerate(columns, 1):
                if not column_text.strip():
                    continue
                match = FUSION_ROW_RE.search(column_text)
                missing_tm = False
                if match is None:
                    match = FUSION_MISSING_TM_ROW_RE.search(column_text)
                    missing_tm = match is not None
                if match is None:
                    continue

                fragment = " ".join(match.group("name").split())
                table_name, source_fragment = reconstruct_fusion_name(
                    fragment,
                    previous_names[column_number],
                )
                record = fusion_record_from_fields(
                    table_name=table_name,
                    formula=match.group("formula"),
                    tm_text=None if missing_tm else match.group("tm"),
                    hfus_text=match.group("hfus"),
                    secondary_tm_text=(
                        None if missing_tm else match.group("secondary_tm")
                    ),
                    tm_note=None if missing_tm else match.group("tm_note"),
                    line_number=line_number,
                    raw_line=" ".join(raw_line.split()),
                    column=column_number,
                    source_name_fragment=source_fragment,
                )
                if record is None:
                    continue
                record["source_page"] = FUSION_TABLE.pages[0] + page_number - 1
                previous_names[column_number] = record["table_name"]
                records.append(record)
    return records


def pdftotext(first: int, last: int) -> str:
    return subprocess.check_output(
        ["pdftotext", "-layout", "-f", str(first), "-l", str(last), str(PDF), "-"],
        text=True,
        errors="ignore",
    )


def split_identity(line: str, has_equation_id: bool) -> tuple[dict[str, Any], str] | None:
    cas_match = CAS_RE.search(line)
    if not cas_match:
        return None

    prefix_re = (
        r"^\s*(?P<equation_id>\d{3})\s+(?P<compound_number>\d+)\s+"
        if has_equation_id
        else r"^\s*(?P<compound_number>\d+)\s+"
    )
    prefix = re.match(prefix_re, line)
    if not prefix:
        return None

    left = line[prefix.end() : cas_match.start()].strip()
    parsed_identity = split_formula_suffix(left.split())
    if parsed_identity is None:
        return None
    name, formula = parsed_identity

    identity = prefix.groupdict()
    if identity.get("equation_id") is not None:
        identity["equation_id"] = int(identity["equation_id"])
    identity["compound_number"] = int(identity["compound_number"])
    identity["cas"] = cas_match.group(0)
    identity["formula"] = formula
    identity["name"] = name
    return identity, line[cas_match.end() :]


def base_correlation(spec: TableSpec, identity: dict[str, Any], nums: list[float]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_table": spec.table,
        "compound_number": identity["compound_number"],
    }
    if "equation_id" in identity:
        record["equation_id"] = identity["equation_id"]
    if nums:
        record["molecular_weight"] = nums[0]
    if len(nums) >= 5:
        printed_coefficients = nums[1:-4]
        if spec.coefficient_multipliers:
            coefficients = [
                value * spec.coefficient_multipliers[idx]
                for idx, value in enumerate(printed_coefficients)
            ]
            record["printed_coefficients"] = printed_coefficients
            record["coefficient_multipliers"] = list(spec.coefficient_multipliers[: len(printed_coefficients)])
            record["coefficients"] = coefficients
        else:
            record["coefficients"] = printed_coefficients
        record["T_min_K"] = nums[-4]
        if spec.endpoint_value_multiplier != 1.0:
            record["printed_value_at_T_min"] = nums[-3]
            record["value_at_T_min"] = nums[-3] * spec.endpoint_value_multiplier
        else:
            record["value_at_T_min"] = nums[-3]
        record["T_max_K"] = nums[-2]
        if spec.endpoint_value_multiplier != 1.0:
            record["printed_value_at_T_max"] = nums[-1]
            record["value_at_T_max"] = nums[-1] * spec.endpoint_value_multiplier
        else:
            record["value_at_T_max"] = nums[-1]
    else:
        record["raw_numbers"] = nums
    return record


def add_density_volume_endpoints(record: dict[str, Any]) -> None:
    """Add molar-volume endpoints for Perry molar-density correlations."""
    rho_min = record.get("value_at_T_min")
    rho_max = record.get("value_at_T_max")
    if rho_min:
        record["molar_volume_at_T_min_dm3_per_mol"] = 1.0 / rho_min
    if rho_max:
        record["molar_volume_at_T_max_dm3_per_mol"] = 1.0 / rho_max


def parse_property(spec: TableSpec, identity: dict[str, Any], nums: list[float]) -> dict[str, Any] | None:
    if spec.kind == "critical":
        if len(nums) < 6:
            return None
        record = {
            "source_table": spec.table,
            "compound_number": identity["compound_number"],
            "molecular_weight": nums[0],
            "Tc_K": nums[1],
            "Pc_MPa": nums[2],
            "Vc_m3_per_kmol": nums[3],
            "Zc": nums[4],
            "omega": nums[5],
        }
        correction = CRITICAL_CONSTANT_CORRECTIONS.get(identity["cas"])
        if correction:
            record["_vc_patch"] = {
                "date": "2026-07-09",
                "original_Vc_m3_per_kmol": record["Vc_m3_per_kmol"],
                "original_Zc": record["Zc"],
                "reason": correction["reason"],
                "source": correction["source"],
            }
            record["Vc_m3_per_kmol"] = correction["Vc_m3_per_kmol"]
            record["Zc"] = correction["Zc"]
        return record

    if spec.kind == "formation":
        if len(nums) < 5:
            return None
        return {
            "source_table": spec.table,
            "compound_number": identity["compound_number"],
            "molecular_weight": nums[0],
            "Hf_ideal_gas_J_per_kmol": nums[1] * 1.0e7,
            "Gf_ideal_gas_J_per_kmol": nums[2] * 1.0e7,
            "S_ideal_gas_J_per_kmol_K": nums[3] * 1.0e5,
            "net_Hcomb_J_per_kmol": nums[4] * 1.0e9 if len(nums) > 4 else None,
            "scaled_table_values": {
                "Hf_times_1e_minus_7": nums[1],
                "Gf_times_1e_minus_7": nums[2],
                "S_times_1e_minus_5": nums[3],
                "Hcomb_times_1e_minus_9": nums[4] if len(nums) > 4 else None,
            },
        }

    if spec.kind == "vapor_pressure":
        if len(nums) < 7:
            return None
        coeffs = nums[:-4]
        coeffs = VAPOR_PRESSURE_COEFFICIENT_CORRECTIONS.get(
            (spec.table, identity["cas"]),
            coeffs,
        )
        return {
            "source_table": spec.table,
            "compound_number": identity["compound_number"],
            "equation": spec.units["equation"],
            "coefficients": coeffs,
            "T_min_K": nums[-4],
            "P_at_T_min_Pa": nums[-3],
            "T_max_K": nums[-2],
            "P_at_T_max_Pa": nums[-1],
        }

    if len(nums) < 5:
        return None
    record = base_correlation(spec, identity, nums)
    if spec.key == "liquid_density":
        add_density_volume_endpoints(record)
    return record


def merge_identity(compound: dict[str, Any], identity: dict[str, Any]) -> None:
    compound.setdefault("cas", identity["cas"])
    compound.setdefault("name", identity["name"])
    compound.setdefault("formula", identity["formula"])
    names = compound.setdefault("names", [])
    if identity["name"] not in names:
        names.append(identity["name"])
    formulas = compound.setdefault("formulas", [])
    if identity["formula"] not in formulas:
        formulas.append(identity["formula"])
    numbers = compound.setdefault("compound_numbers", [])
    if identity["compound_number"] not in numbers:
        numbers.append(identity["compound_number"])


def table_2_10_identity_map() -> dict[str, list[dict[str, Any]]]:
    path = DATA_DIR / "perry_table_2_10_vapor_pressure.json"
    if not path.exists():
        return {}

    payload = json.loads(path.read_text())
    by_formula: dict[str, list[dict[str, Any]]] = {}
    for cas, entry in payload.get("chemicals", {}).items():
        formula_key = normalize_formula(entry.get("formula"))
        if not formula_key:
            continue
        aliases = [
            entry.get("name"),
            entry.get("table_name"),
            entry.get("resolved_query"),
        ]
        by_formula.setdefault(formula_key, []).append(
            {
                "cas": cas,
                "name": entry.get("name"),
                "formula": entry.get("formula"),
                "aliases": [alias for alias in aliases if alias],
            }
        )
    return by_formula


def resolve_fusion_record(
    record: dict[str, Any],
    by_formula: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str | None]:
    candidates = by_formula.get(normalize_formula(record["formula"]), [])
    matches: list[tuple[dict[str, Any], str]] = []
    for candidate in candidates:
        for alias in candidate["aliases"]:
            if names_compatible(record["table_name"], alias):
                matches.append((candidate, alias))
                break

    unique_cas = sorted({candidate["cas"] for candidate, _ in matches})
    if len(unique_cas) != 1:
        if not candidates:
            return None, "formula_not_found"
        if not matches:
            return None, "name_mismatch"
        return None, "ambiguous_name"

    candidate, matched_alias = next(
        (candidate, alias)
        for candidate, alias in matches
        if candidate["cas"] == unique_cas[0]
    )
    resolved = dict(record)
    resolved.update(
        {
            "cas": candidate["cas"],
            "name": candidate["name"],
            "resolved_query": matched_alias,
            "resolution_source": "Perry Table 2-10 CAS map",
        }
    )
    return resolved, None


def fusion_name_candidates(table_name: str) -> list[str]:
    candidates = [table_name]
    normalized_name = comparable_name(table_name)
    if normalized_name != table_name:
        candidates.append(normalized_name)
    suffix_match = re.search(
        r"\((?P<prefix>o|m|p|t|n|alpha|beta|gamma|d|l|dl|cis|trans)-?\)",
        normalized_name,
    )
    if suffix_match:
        prefix = suffix_match.group("prefix")
        base = re.sub(r"\s*\([^()]+\)\s*", " ", table_name).strip()
        prefix_aliases = {
            "o": ["o", "ortho"],
            "m": ["m", "meta"],
            "p": ["p", "para"],
            "t": ["t", "tert"],
            "n": ["n"],
            "alpha": ["alpha", "1"],
            "beta": ["beta", "2"],
            "gamma": ["gamma"],
            "d": ["d"],
            "l": ["l"],
            "dl": ["dl"],
            "cis": ["cis"],
            "trans": ["trans"],
        }[prefix]
        for alias in prefix_aliases:
            candidates.append(f"{alias}-{base}")
            candidates.append(f"{alias} {base}")
        candidates.append(base)

    unique: list[str] = []
    for candidate in candidates:
        candidate = " ".join(candidate.split())
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def resolve_fusion_record_with_chemicals(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from chemicals.identifiers import search_chemical
    except Exception:
        return None, "chemicals_unavailable"

    errors = []
    for candidate_name in fusion_name_candidates(record["table_name"]):
        try:
            chemical = search_chemical(candidate_name)
        except Exception as exc:
            errors.append(type(exc).__name__)
            continue
        if normalize_formula(getattr(chemical, "formula", "")) != normalize_formula(record["formula"]):
            errors.append("formula_mismatch")
            continue

        if not getattr(chemical, "CASs", None):
            errors.append("missing_cas")
            continue
        resolved = dict(record)
        resolved.update(
            {
                "cas": chemical.CASs,
                "name": getattr(chemical, "common_name", None) or record["table_name"],
                "resolved_query": candidate_name,
                "resolution_source": "chemicals.identifiers",
            }
        )
        return resolved, None
    return None, "chemicals_" + (errors[-1] if errors else "not_found")


def resolve_fusion_record_override(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    identity = FUSION_IDENTITY_OVERRIDES.get(normalize_key(record["table_name"]))
    if identity is None:
        return None
    if normalize_formula(identity["formula"]) != normalize_formula(record["formula"]):
        return None
    resolved = dict(record)
    resolved.update({
        "cas": identity["cas"],
        "name": identity["name"],
        "resolved_query": identity["name"],
        "resolution_source": "audited Perry Table 2-68 identity override",
    })
    return resolved


def require_chemicals_for_fusion_resolution() -> None:
    try:
        import chemicals  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Table 2-68 heat-of-fusion extraction requires the 'chemicals' package. "
            "Install requirements.txt or run with PYTHONPATH pointing at a chemicals install."
        ) from exc


def extract_organic_heat_of_fusion_payload(spec: TableSpec) -> dict[str, Any]:
    require_chemicals_for_fusion_resolution()
    text = normalize_text(pdftotext(*spec.pages)).split("TABLE 2-69", 1)[0]
    raw_records = fusion_records_from_layout(text)
    by_formula = table_2_10_identity_map()
    chemicals: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    resolution_sources: set[str] = set()

    for record in raw_records:
        resolved = resolve_fusion_record_override(record)
        reason = None
        if resolved is None:
            resolved, reason = resolve_fusion_record_with_chemicals(record)
        if resolved is None:
            resolved, reason = resolve_fusion_record(record, by_formula)
        if resolved is None:
            unresolved_record = dict(record)
            unresolved_record["reason"] = reason
            unresolved.append(unresolved_record)
            continue

        cas = resolved["cas"]
        resolution_sources.add(resolved["resolution_source"])
        entry = chemicals.setdefault(
            cas,
            {
                "cas": cas,
                "name": resolved["name"],
                "formula": resolved["formula"],
                "heat_of_fusion": [],
            },
        )
        if normalize_formula(entry["formula"]) != normalize_formula(resolved["formula"]):
            raise ValueError(
                f"Fusion identity collision for CAS {cas}: "
                f"{entry['formula']} versus {resolved['formula']}"
            )
        entry["heat_of_fusion"].append(resolved)

    return {
        "metadata": {
            "source": PDF.name,
            "source_book": "Perry's Chemical Engineers' Handbook, 9th ed.",
            "source_table": spec.table,
            "pages": list(spec.pages),
            "title": spec.title,
            "units": spec.units,
            "rows_extracted": len(raw_records),
            "rows_resolved": sum(len(entry["heat_of_fusion"]) for entry in chemicals.values()),
            "unique_cas_resolved": len(chemicals),
            "rows_unresolved": len(unresolved),
            "resolution_sources": sorted(resolution_sources),
            "resolver": (
                "layout_context+elemental_formula+chemicals+"
                "strict_perry_table_2_10+audited_overrides"
            ),
            "extraction_notes": [
                "Table 2-68 has no CAS column, so it is stored separately from the CAS-keyed Perry correlation file.",
                "The two visual columns are parsed independently and abbreviated continuation names inherit their column context.",
                "CAS resolution requires elemental formula agreement plus an exact local identity, strict Perry alias, or audited override.",
                "Printed Perry name/formula errors are retained in source_* fields and corrected through an explicit errata contract.",
                "Unmatched and ambiguous rows are retained under unresolved instead of being merged by formula alone.",
            ],
        },
        "chemicals": dict(sorted(chemicals.items())),
        "unresolved": unresolved,
        "raw_records": raw_records,
    }


def extract_table(spec: TableSpec, chemicals: dict[str, dict[str, Any]]) -> int:
    text = normalize_text(pdftotext(*spec.pages))
    count = 0
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line or "TABLE 2-" in line or "Cmpd." in line or "Formula" in line:
            continue
        parsed = split_identity(line, spec.has_equation_id)
        if not parsed:
            continue
        identity, tail = parsed
        nums = numbers_after(tail)
        prop = parse_property(spec, identity, nums)
        if prop is None:
            continue

        compound = chemicals.setdefault(identity["cas"], {})
        merge_identity(compound, identity)
        if spec.kind in {"critical", "formation"}:
            compound[spec.key] = prop
        else:
            compound.setdefault(spec.key, []).append(prop)
        count += 1
    return count


def main() -> None:
    if not PDF.exists():
        raise SystemExit(f"Missing Perry PDF: {PDF}")

    chemicals: dict[str, dict[str, Any]] = {}
    table_counts: dict[str, int] = {}
    for spec in TABLES:
        table_counts[spec.key] = extract_table(spec, chemicals)

    payload = {
        "metadata": {
            "source": PDF.name,
            "source_book": "Perry's Chemical Engineers' Handbook, 9th ed.",
            "extraction_notes": [
                "Correlations are stored with Perry table IDs, coefficients, units, and fit ranges.",
                "Formation table values were expanded from Perry's printed scale factors to SI per kmol.",
                "Molar density correlations use Perry Table 2-32 units of mol/dm^3; endpoint molar volumes are included as 1/rho in dm^3/mol.",
            ],
            "tables": {
                spec.key: {
                    "table": spec.table,
                    "pages": list(spec.pages),
                    "title": spec.title,
                    "units": spec.units,
                    "rows_extracted": table_counts[spec.key],
                }
                for spec in TABLES
            },
        },
        "chemicals": dict(sorted(chemicals.items())),
    }

    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fusion_payload = extract_organic_heat_of_fusion_payload(FUSION_TABLE)
    FUSION_OUTPUT.write_text(json.dumps(fusion_payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {FUSION_OUTPUT.relative_to(ROOT)}")
    print(json.dumps(table_counts, indent=2, sort_keys=True))
    print(f"Unique CAS entries: {len(chemicals)}")


if __name__ == "__main__":
    main()
