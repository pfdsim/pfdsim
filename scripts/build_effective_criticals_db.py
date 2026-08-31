#!/usr/bin/env python3
"""Build a CAS-keyed SQLite table of effective critical properties."""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_DATA = DATA / "source"
SOURCE = SOURCE_DATA / "effective_criticals.txt"
OUTPUT = DATA / "effective_criticals.sqlite"
TABLE_NAME = "effective_criticals"

R_BAR_L = 0.0831446261815324
GRADE_OFFSETS = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}


def parse_float(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def parse_quality(value: Any) -> tuple[str, str, str, str, str]:
    text = str(value or "").strip()
    property_grades, separator, fit_grade = text.partition("/")
    if separator != "/" or len(property_grades) != 3 or not fit_grade:
        raise ValueError(f"Invalid quality code {text!r}")

    suffix = ""
    if fit_grade[-1].isdigit():
        suffix = fit_grade[-1]
        fit_grade = fit_grade[:-1]

    grades = (*property_grades, fit_grade)
    unexpected = [grade for grade in grades if grade not in GRADE_OFFSETS]
    if unexpected:
        raise ValueError(f"Invalid quality grade(s) {unexpected!r} in {text!r}")

    return property_grades[0], property_grades[1], property_grades[2], fit_grade, suffix


def quality_score(
    category: str,
    own_grade: str,
    fit_grade: str,
    suffix: str,
    cap_grade: str,
) -> float:
    score = 0.95 if suffix == "1" else 0.96
    score -= 0.04 * GRADE_OFFSETS[own_grade]
    score -= 0.02 * GRADE_OFFSETS[fit_grade]
    if cap_grade == "D":
        score = min(score, 0.75)
    if suffix == "2" and category == "Vc":
        score = min(score, 0.70)
    if suffix == "3" and category in {"Tc", "Pc"}:
        score = min(score, 0.70)
    return round(score, 6)


def omega_quality_score(fit_grade: str, suffix: str) -> float:
    score = 0.95 if suffix == "1" else 0.96
    score -= 0.05 * GRADE_OFFSETS[fit_grade]
    if fit_grade == "D":
        score = min(score, 0.75)
    return round(score, 6)


def zc_quality_score(tc_quality: float, pc_quality: float, vc_quality: float) -> float:
    return min(tc_quality, pc_quality, vc_quality)


def build_record(row: list[str]) -> tuple[object, ...]:
    if len(row) != 9:
        raise ValueError(f"Expected 9 fields, got {len(row)}: {row!r}")

    name, _source, quality, cas, mw_text, tc_text, pc_text, rhoc_text, omega_text = row
    mw = parse_float(mw_text)
    tc = parse_float(tc_text)
    pc_kpa = parse_float(pc_text)
    rhoc = parse_float(rhoc_text)
    omega = parse_float(omega_text)
    if mw is None or tc is None or pc_kpa is None or rhoc is None or omega is None:
        raise ValueError(f"Invalid numeric value in row for CAS {cas!r}")
    if tc <= 0 or pc_kpa <= 0 or rhoc <= 0:
        raise ValueError(f"Critical constants must be positive in row for CAS {cas!r}")

    pc_bar = pc_kpa / 100.0
    vc = mw * 1000.0 / rhoc
    zc = pc_bar * (vc / 1000.0) / (R_BAR_L * tc)

    tc_grade, pc_grade, vc_grade, fit_grade, suffix = parse_quality(quality)
    tc_quality = quality_score("Tc", tc_grade, fit_grade, suffix, tc_grade)
    pc_quality = quality_score("Pc", pc_grade, fit_grade, suffix, pc_grade)
    vc_quality = quality_score("Vc", vc_grade, fit_grade, suffix, vc_grade)
    zc_quality = zc_quality_score(tc_quality, pc_quality, vc_quality)
    omega_quality = omega_quality_score(fit_grade, suffix)

    return (
        cas.strip(),
        name.strip(),
        mw,
        tc,
        pc_bar,
        vc,
        zc,
        omega,
        tc_quality,
        pc_quality,
        vc_quality,
        zc_quality,
        omega_quality,
    )


def source_records(path: Path) -> list[tuple[object, ...]]:
    records = []
    seen_cas = set()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return records
        for line_number, row in enumerate(reader, start=2):
            record = build_record(row)
            cas = record[0]
            if not cas:
                raise ValueError(f"Missing CAS on source line {line_number}")
            if cas in seen_cas:
                raise ValueError(f"Duplicate CAS {cas!r} on source line {line_number}")
            seen_cas.add(cas)
            records.append(record)
    return records


def write_database(records: list[tuple[object, ...]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with sqlite3.connect(output) as connection:
        connection.execute(
            f"""
            CREATE TABLE {TABLE_NAME} (
                CAS TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                MW REAL NOT NULL,
                Tc REAL NOT NULL,
                Pc REAL NOT NULL,
                Vc REAL NOT NULL,
                Zc REAL NOT NULL,
                omega REAL NOT NULL,
                Tc_quality REAL NOT NULL,
                Pc_quality REAL NOT NULL,
                Vc_quality REAL NOT NULL,
                Zc_quality REAL NOT NULL,
                omega_quality REAL NOT NULL
            )
            """
        )
        connection.executemany(
            f"""
            INSERT INTO {TABLE_NAME} (
                CAS, name, MW, Tc, Pc, Vc, Zc, omega,
                Tc_quality, Pc_quality, Vc_quality, Zc_quality, omega_quality
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        connection.execute(f"CREATE INDEX idx_{TABLE_NAME}_name ON {TABLE_NAME} (name)")
        connection.execute(f"CREATE INDEX idx_{TABLE_NAME}_tc_quality ON {TABLE_NAME} (Tc_quality)")
        connection.execute(f"CREATE INDEX idx_{TABLE_NAME}_pc_quality ON {TABLE_NAME} (Pc_quality)")
        connection.execute(f"CREATE INDEX idx_{TABLE_NAME}_vc_quality ON {TABLE_NAME} (Vc_quality)")
        connection.execute(f"CREATE INDEX idx_{TABLE_NAME}_zc_quality ON {TABLE_NAME} (Zc_quality)")
        connection.execute(f"CREATE INDEX idx_{TABLE_NAME}_omega_quality ON {TABLE_NAME} (omega_quality)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help=f"source TSV path (default: {SOURCE})")
    parser.add_argument("--output", type=Path, default=OUTPUT, help=f"output SQLite path (default: {OUTPUT})")
    args = parser.parse_args()

    records = source_records(args.source)
    write_database(records, args.output)
    print(f"Wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
