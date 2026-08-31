#!/usr/bin/env python3
"""Extract Perry Table 2-10 as a CAS-keyed tabulated vapor-pressure dataset.

Perry Table 2-10 has no CAS column and contains many isomers.  It is therefore
kept separate from the main Perry correlation database and resolved by name.
The default resolver uses the optional local ``chemicals`` package, avoiding
large online lookup batches against PubChem/PUGREST.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemical_properties import ChemicalDatabase
from compound_identity import parse_formula_counts
from pressure_standards import NORMAL_BOILING_PRESSURE_BAR

DATA_DIR = ROOT / "data"
SOURCE_DATA_DIR = DATA_DIR / "source"
PDF = SOURCE_DATA_DIR / (
    "perrys-chemical-engineers-handjob-9th-edition-2021-9nbsped-"
    "9780071834094-0071834095-9780071834087-0071834087_compress.pdf"
)
OUTPUT = DATA_DIR / "perry_table_2_10_vapor_pressure.json"

SOURCE_TABLE = "2-10"
PAGES = (98, 112)
PRESSURE_COLUMNS_MMHG = (1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0, 200.0, 400.0, 760.0)
NUMBER_RE = re.compile(r"[-+\u2212]?(?:\d+(?:\.\d*)?|\.\d+)")
FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\s*\d*|\(|\)|\d+)+$")
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
SOURCE_TEMPERATURE_CORRECTIONS_C = {
    # Perry Table 2-10 has two printed sign errors that violate monotonic Psat.
    # The corrected signs are consistent with neighboring table values.
    (103, "Diethyl ether", "C4H10O", 40.0): -27.7,
    (105, "Ethyl formate", "C3H6O2", 100.0): 5.4,
}
_THREAD_LOCAL = threading.local()
_CHEMICALS_SEARCH_LOCK = threading.Lock()


@dataclass
class PerryTableRow:
    raw_name: str
    query_name: str
    formula: str
    pressure_temperatures_C: dict[float, float]
    melting_point_C: Optional[float]
    page: int
    raw_line: str

    @property
    def tb_C(self) -> Optional[float]:
        return self.pressure_temperatures_C.get(760.0)


@dataclass
class ResolvedIdentity:
    cas: str
    name: str
    formula: str


def normalize_text(text: str) -> str:
    return (
        text.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00b7", "*")
        .replace("\u2219", "*")
    )


def pdftotext(first: int, last: int) -> str:
    return subprocess.check_output(
        ["pdftotext", "-layout", "-f", str(first), "-l", str(last), str(PDF), "-"],
        text=True,
        errors="ignore",
    )


def parse_number(token: str) -> float:
    return float(token.replace("\u2212", "-").replace("+", ""))


def normalize_formula(formula: str) -> str:
    return re.sub(r"\s+", "", formula).replace("*", "")


def formula_matches(expected: str, actual: str) -> bool:
    expected_counts = parse_formula_counts(normalize_formula(expected))
    actual_counts = parse_formula_counts(normalize_formula(actual))
    return expected_counts is not None and expected_counts == actual_counts


def split_identity(prefix: str) -> tuple[str, str] | None:
    parts = [part.strip() for part in re.split(r"\s{2,}", prefix.strip()) if part.strip()]
    if len(parts) < 2:
        return None
    formula = normalize_formula(parts[-1])
    if not FORMULA_RE.match(formula):
        return None
    name = " ".join(parts[:-1]).strip()
    if not name:
        return None
    return name, formula


def looks_like_formula_continuation(text: str) -> bool:
    formula = normalize_formula(text)
    return bool(formula and FORMULA_RE.match(formula))


def is_table_number(line: str, match: re.Match[str], data_column_start: int) -> bool:
    if match.start() < data_column_start:
        return False
    if match.start() > 0 and line[match.start() - 1].isalpha():
        return False
    return True


def apply_source_temperature_corrections(row: PerryTableRow) -> None:
    for pressure in tuple(row.pressure_temperatures_C):
        key = (row.page, row.query_name, row.formula, pressure)
        if key in SOURCE_TEMPERATURE_CORRECTIONS_C:
            row.pressure_temperatures_C[pressure] = SOURCE_TEMPERATURE_CORRECTIONS_C[key]


def page_columns(text: str) -> tuple[dict[float, int], Optional[int]]:
    for line in text.splitlines():
        if "Compound" not in line or "760" not in line or "point" not in line:
            continue
        pressure_positions = {
            float(match.group(0)): match.start()
            for match in re.finditer(r"\b(?:1|5|10|20|40|60|100|200|400|760)\b", line)
        }
        if all(column in pressure_positions for column in PRESSURE_COLUMNS_MMHG):
            return pressure_positions, line.find("point")
    raise ValueError("Could not locate Perry Table 2-10 column header")


def column_for_number(
    start: int,
    end: int,
    pressure_positions: dict[float, int],
    melting_position: Optional[int],
) -> str | float | None:
    center = (start + end - 1) / 2.0
    candidates: list[tuple[float, str | float]] = [
        (abs(center - position), pressure)
        for pressure, position in pressure_positions.items()
    ]
    if melting_position is not None and melting_position >= 0:
        candidates.append((abs(center - melting_position), "melting"))
    distance, column = min(candidates)
    if distance > 7.0:
        return None
    return column


def table_rows() -> list[PerryTableRow]:
    rows: list[PerryTableRow] = []
    parent_prefix = ""

    for page in range(PAGES[0], PAGES[1] + 1):
        text = normalize_text(pdftotext(page, page))
        pressure_positions, melting_position = page_columns(text)
        data_column_start = min(pressure_positions.values()) - 6
        last_row: Optional[PerryTableRow] = None
        base_indent: Optional[int] = None
        pending_name: Optional[str] = None
        pending_line: Optional[str] = None
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("TABLE 2-") or "Pressure, mmHg" in stripped:
                continue
            if "Temperature" in stripped or stripped.startswith("Name"):
                continue
            if "PHYSICAL AnD CHEMICAL DATA" in stripped or stripped == "Melting":
                continue
            if stripped.startswith("*Compiled") or stripped.startswith("(Continued"):
                continue

            matches = [
                match
                for match in NUMBER_RE.finditer(line)
                if is_table_number(line, match, data_column_start)
            ]
            if not matches:
                if not stripped:
                    continue
                if (
                    last_row is not None
                    and (
                        stripped.startswith("(")
                        or last_row.raw_name.count("(") > last_row.raw_name.count(")")
                        or last_row.raw_name.count("[") > last_row.raw_name.count("]")
                    )
                ):
                    last_row.raw_name = f"{last_row.raw_name} {stripped}".strip()
                    last_row.query_name = f"{last_row.query_name} {stripped}".strip()
                    last_row.raw_line = f"{last_row.raw_line}\n{line}"
                    continue
                if last_row is not None and looks_like_formula_continuation(stripped):
                    last_row.formula = normalize_formula(f"{last_row.formula}{stripped}")
                    last_row.raw_line = f"{last_row.raw_line}\n{line}"
                    continue
                pending_name = f"{pending_name} {stripped}".strip() if pending_name else stripped
                pending_line = f"{pending_line}\n{line}" if pending_line else line
                continue
            if len(matches) < 4:
                continue
            first_number = matches[0]
            identity = split_identity(line[: first_number.start()])
            if identity is None:
                continue

            raw_name, formula = identity
            raw_line_for_row = line
            used_pending_name = False
            if pending_name:
                raw_name = f"{pending_name} {raw_name}".strip()
                raw_line_for_row = f"{pending_line}\n{line}" if pending_line else line
                pending_name = None
                pending_line = None
                used_pending_name = True

            pressure_temperatures_C: dict[float, float] = {}
            melting_point_C: Optional[float] = None
            for match in matches:
                column = column_for_number(
                    match.start(),
                    match.end(),
                    pressure_positions,
                    melting_position,
                )
                if column is None:
                    continue
                value = parse_number(match.group(0))
                if column == "melting":
                    melting_point_C = value
                else:
                    pressure_temperatures_C[float(column)] = value

            if len(pressure_temperatures_C) < 4:
                continue

            compact_name = raw_name.strip()
            indent = len(line) - len(line.lstrip())
            if base_indent is None:
                base_indent = indent
            starts_continuation = bool(
                not used_pending_name
                and
                parent_prefix
                and (
                    compact_name.startswith("(")
                    or indent > base_indent + 1
                )
            )
            query_name = f"{parent_prefix} {compact_name}".strip() if starts_continuation else compact_name
            if not starts_continuation:
                parent_prefix = compact_name.split()[0]

            row = (
                PerryTableRow(
                    raw_name=compact_name,
                    query_name=query_name,
                    formula=formula,
                    pressure_temperatures_C=pressure_temperatures_C,
                    melting_point_C=melting_point_C,
                    page=page,
                    raw_line=raw_line_for_row,
                )
            )
            apply_source_temperature_corrections(row)
            rows.append(row)
            last_row = row

    return rows


def name_candidates(row: PerryTableRow) -> list[str]:
    result: list[str] = []

    def add(value: str):
        value = re.sub(r"\s+", " ", value).strip(" ,;")
        if value and value not in result:
            result.append(value)

    def add_variants(value: str):
        values = [value]
        greek = (
            ("\u03b1", "alpha"),
            ("\u03b2", "beta"),
            ("\u03b3", "gamma"),
            ("\u0391", "alpha"),
            ("\u0392", "beta"),
            ("\u0393", "gamma"),
        )
        normalized = value
        for source, replacement in greek:
            normalized = normalized.replace(source, replacement)
        normalized = (
            normalized
            .replace("\u2032", "'")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        values.append(normalized)
        values.append(re.sub(r"\s+\([^()]*\)", "", normalized))
        values.append(normalized.replace("-", " "))

        for candidate in values:
            add(candidate)
            for parenthetical in re.findall(r"\s\(([^()]+)\)", candidate):
                add(parenthetical)

    add_variants(row.query_name)
    add_variants(row.raw_name)
    return result


def resolve_row_online(row: PerryTableRow, database: ChemicalDatabase):
    formula_rejections: list[dict[str, str]] = []
    for candidate in name_candidates(row):
        props = database.get(candidate, fetch_online=True)
        if not props or not props.CAS:
            continue
        if props.formula and not formula_matches(row.formula, props.formula):
            formula_rejections.append(
                {
                    "query": candidate,
                    "resolved_name": props.name or "",
                    "resolved_cas": props.CAS or "",
                    "resolved_formula": props.formula,
                }
            )
            continue
        return (
            ResolvedIdentity(props.CAS.strip(), props.name or candidate, props.formula or ""),
            candidate,
            formula_rejections,
        )
    return None, None, formula_rejections


def dashed_cas(value: str) -> str:
    value = str(value or "").strip()
    if CAS_RE.match(value):
        return value
    digits = re.sub(r"\D", "", value)
    if len(digits) < 4:
        return value
    return f"{digits[:-3]}-{digits[-3:-1]}-{digits[-1]}"


def resolve_row_chemicals(row: PerryTableRow):
    try:
        from chemicals import identifiers as ids
    except ImportError as exc:
        raise RuntimeError(
            "The 'chemicals' package is required for --resolver chemicals. "
            "Install it or run with PYTHONPATH pointing at the temporary target."
        ) from exc

    formula_rejections: list[dict[str, str]] = []
    for candidate in name_candidates(row):
        try:
            with _CHEMICALS_SEARCH_LOCK:
                props = ids.search_chemical(candidate, autoload=True, cache=True)
        except ValueError:
            continue
        if not props:
            continue
        cas = dashed_cas(getattr(props, "CASs", "") or getattr(props, "CAS", ""))
        if not cas:
            continue
        formula = getattr(props, "formula", "") or ""
        if formula and not formula_matches(row.formula, formula):
            formula_rejections.append(
                {
                    "query": candidate,
                    "resolved_name": getattr(props, "common_name", "") or "",
                    "resolved_cas": cas,
                    "resolved_formula": formula,
                }
            )
            continue
        name = (
            getattr(props, "common_name", "")
            or getattr(props, "iupac_name", "")
            or candidate
        )
        return ResolvedIdentity(cas, name, formula), candidate, formula_rejections
    return None, None, formula_rejections


def worker_database() -> ChemicalDatabase:
    database = getattr(_THREAD_LOCAL, "database", None)
    if database is None:
        database = ChemicalDatabase(enable_online=True)
        _THREAD_LOCAL.database = database
    return database


def resolve_row(row: PerryTableRow, resolver: str):
    if resolver == "chemicals":
        return resolve_row_chemicals(row)
    if resolver == "online":
        return resolve_row_online(row, worker_database())
    raise ValueError(f"Unknown resolver: {resolver}")


def pressure_bar(mmHg: float) -> float:
    return NORMAL_BOILING_PRESSURE_BAR * mmHg / 760.0


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def source_temperature_corrections_payload() -> list[dict[str, Any]]:
    return [
        {
            "source_page": page,
            "table_name": name,
            "formula": formula,
            "P_mmHg": pressure,
            "corrected_T_C": temperature,
            "reason": "Printed Perry Table 2-10 sign error; corrected to preserve monotonic Psat.",
        }
        for (page, name, formula, pressure), temperature
        in sorted(SOURCE_TEMPERATURE_CORRECTIONS_C.items())
    ]


def row_payload(row: PerryTableRow, identity: ResolvedIdentity, query: str) -> dict[str, Any]:
    points = [
        {
            "P_mmHg": mmHg,
            "P_bar": pressure_bar(mmHg),
            "T_C": temperature_C,
            "T_K": temperature_C + 273.15,
        }
        for mmHg, temperature_C in sorted(row.pressure_temperatures_C.items())
        if math.isfinite(temperature_C)
    ]
    payload: dict[str, Any] = {
        "cas": identity.cas,
        "name": identity.name,
        "table_name": row.query_name,
        "resolved_query": query,
        "formula": row.formula,
        "source_table": SOURCE_TABLE,
        "source_page": row.page,
        "vapor_pressure": points,
    }
    if row.raw_name != row.query_name:
        payload["raw_table_name"] = row.raw_name
    if row.tb_C is not None:
        payload["Tb_C_at_760_mmHg"] = row.tb_C
        payload["Tb_K"] = row.tb_C + 273.15
    if row.melting_point_C is not None:
        payload["Tm_C"] = row.melting_point_C
        payload["Tm_K"] = row.melting_point_C + 273.15
    return payload


def resolve_row_record(
    index: int,
    row: PerryTableRow,
    resolver: str,
) -> tuple[int, Optional[str], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    identity, query, formula_rejections = resolve_row(row, resolver)
    if not identity or not query:
        unresolved = {
            "table_name": row.query_name,
            "query_name": row.query_name,
            "formula": row.formula,
            "source_page": row.page,
            "reason": "formula_mismatch" if formula_rejections else "not_found",
        }
        if row.raw_name != row.query_name:
            unresolved["raw_table_name"] = row.raw_name
        if formula_rejections:
            unresolved["formula_rejections"] = formula_rejections
        return index, None, None, unresolved

    cas = identity.cas.strip()
    if not CAS_RE.match(cas):
        return index, None, None, {
            "table_name": row.query_name,
            "query_name": row.query_name,
            "formula": row.formula,
            "source_page": row.page,
            "reason": f"invalid CAS {cas!r}",
        }

    return index, cas, row_payload(row, identity, query), None


def row_key_from_row(row: PerryTableRow) -> tuple[str, str, int]:
    return row.query_name, normalize_formula(row.formula), row.page


def row_key_from_record(record: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(record.get("query_name") or record.get("table_name") or ""),
        normalize_formula(str(record.get("formula") or "")),
        int(record.get("source_page") or 0),
    )


def fill_existing_with_online(
    existing_path: Path,
    output_path: Path,
    delay_s: float,
) -> None:
    existing = json.loads(existing_path.read_text())
    unresolved_records = existing.get("unresolved", [])
    unresolved_keys = {row_key_from_record(record) for record in unresolved_records}

    database = ChemicalDatabase(enable_online=True)
    rows = table_rows()
    chemicals = dict(existing.get("chemicals", {}))
    duplicate_cas = {
        str(cas): list(names)
        for cas, names in existing.get("metadata", {}).get("duplicate_cas_collisions", {}).items()
    }
    still_unresolved_by_index: dict[int, dict[str, Any]] = {}
    newly_resolved = 0
    duplicate_online = 0

    target_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if row_key_from_row(row) in unresolved_keys
    ]

    for offset, (index, row) in enumerate(target_rows, start=1):
        identity, query, formula_rejections = resolve_row_online(row, database)
        if identity and query and CAS_RE.match(identity.cas.strip()):
            payload = row_payload(row, identity, query)
            cas = identity.cas.strip()
            if cas in chemicals:
                duplicate_cas.setdefault(cas, [chemicals[cas]["table_name"]]).append(payload["table_name"])
                duplicate_online += 1
            else:
                chemicals[cas] = payload
                newly_resolved += 1
        else:
            unresolved = {
                "table_name": row.query_name,
                "query_name": row.query_name,
                "formula": row.formula,
                "source_page": row.page,
                "reason": "formula_mismatch" if formula_rejections else "not_found",
            }
            if row.raw_name != row.query_name:
                unresolved["raw_table_name"] = row.raw_name
            if formula_rejections:
                unresolved["formula_rejections"] = formula_rejections
            still_unresolved_by_index[index] = unresolved

        if delay_s > 0.0 and offset < len(target_rows):
            time.sleep(delay_s)

    metadata = dict(existing.get("metadata", {}))
    rows_unresolved = len(still_unresolved_by_index)
    metadata.update(
        {
            "rows_resolved": metadata.get("rows_extracted", len(rows)) - rows_unresolved,
            "unique_cas_resolved": len(chemicals),
            "rows_unresolved": rows_unresolved,
            "pubchem_fill": {
                "input_unresolved_rows": len(unresolved_records),
                "rows_attempted": len(target_rows),
                "newly_resolved_unique_cas": newly_resolved,
                "duplicate_cas_from_online": duplicate_online,
                "delay_s": delay_s,
            },
            "source_temperature_corrections_C": source_temperature_corrections_payload(),
            "duplicate_cas_collisions": duplicate_cas,
        }
    )
    notes = list(metadata.get("notes", []))
    note = "Remaining unresolved rows were retried sequentially through PubChem."
    if note not in notes:
        notes.append(note)
    metadata["notes"] = notes

    output = {
        "metadata": metadata,
        "chemicals": dict(sorted(chemicals.items())),
        "unresolved": [
            still_unresolved_by_index[index]
            for index in sorted(still_unresolved_by_index)
        ],
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {display_path(output_path)}")
    print(
        json.dumps(
            {
                "input_unresolved_rows": len(unresolved_records),
                "rows_attempted": len(target_rows),
                "newly_resolved_unique_cas": newly_resolved,
                "duplicate_cas_from_online": duplicate_online,
                "rows_unresolved": rows_unresolved,
                "unique_cas_resolved": len(chemicals),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--fill-existing-online",
        type=Path,
        help="Load an existing Table 2-10 JSON and retry only unresolved rows through PubChem sequentially.",
    )
    parser.add_argument(
        "--online-delay",
        type=float,
        default=0.25,
        help="Seconds to wait between sequential PubChem unresolved-row retries.",
    )
    parser.add_argument(
        "--resolver",
        choices=("chemicals", "online"),
        default="chemicals",
        help="Identity resolver to use for CAS assignment.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent identity-resolution workers.",
    )
    args = parser.parse_args()

    if args.fill_existing_online:
        fill_existing_with_online(args.fill_existing_online, args.output, max(0.0, args.online_delay))
        return

    rows = table_rows()
    if args.resolver == "chemicals":
        # Force the local metadata database to load before worker threads start.
        resolve_row_chemicals(PerryTableRow("Water", "Water", "H2O", {760.0: 100.0}, 0.0, 0, "Water H2O 100 0"))

    resolved_by_index: dict[int, tuple[str, dict[str, Any]]] = {}
    unresolved_by_index: dict[int, dict[str, Any]] = {}
    workers = max(1, args.workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(resolve_row_record, index, row, args.resolver)
            for index, row in enumerate(rows)
        ]
        for future in as_completed(futures):
            index, cas, payload, unresolved = future.result()
            if unresolved is not None:
                unresolved_by_index[index] = unresolved
                continue
            if cas is None or payload is None:
                continue
            resolved_by_index[index] = (cas, payload)

    chemicals: dict[str, dict[str, Any]] = {}
    duplicate_cas: dict[str, list[str]] = {}
    for index in sorted(resolved_by_index):
        cas, payload = resolved_by_index[index]
        if cas in chemicals:
            duplicate_cas.setdefault(cas, [chemicals[cas]["table_name"]]).append(payload["table_name"])
            continue
        chemicals[cas] = payload

    unresolved = [
        unresolved_by_index[index]
        for index in sorted(unresolved_by_index)
    ]
    rows_resolved = len(rows) - len(unresolved)

    output = {
        "metadata": {
            "source": PDF.name,
            "source_book": "Perry's Chemical Engineers' Handbook, 9th ed.",
            "source_table": SOURCE_TABLE,
            "pages": list(PAGES),
            "title": "Vapor pressures of organic compounds, up to 1 atm",
            "pressure_columns_mmHg": list(PRESSURE_COLUMNS_MMHG),
            "temperature_units": "degC in source; K in converted fields",
            "pressure_units": "mmHg in source; bar in converted fields",
            "rows_extracted": len(rows),
            "rows_resolved": rows_resolved,
            "unique_cas_resolved": len(chemicals),
            "rows_unresolved": len(unresolved),
            "resolver": args.resolver,
            "resolution_workers": workers,
            "source_temperature_corrections_C": source_temperature_corrections_payload(),
            "duplicate_cas_collisions": duplicate_cas,
            "notes": [
                "Rows are resolved by name and keyed only by validated CAS numbers.",
                "Rows whose resolved formula does not match the Perry formula are left unresolved.",
                "Rows without a 760 mmHg column keep their lower-pressure data but do not include Tb_K.",
                "This table is separate from perry_properties.json because Perry Table 2-10 has no CAS column.",
            ],
        },
        "chemicals": dict(sorted(chemicals.items())),
        "unresolved": unresolved,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {display_path(args.output)}")
    print(
        json.dumps(
            {
                "rows_extracted": len(rows),
                "rows_resolved": rows_resolved,
                "unique_cas_resolved": len(chemicals),
                "rows_unresolved": len(unresolved),
                "duplicate_cas": len(duplicate_cas),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
