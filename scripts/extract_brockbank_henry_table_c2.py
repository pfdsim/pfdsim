#!/usr/bin/env python3
"""Extract Brockbank (2013) Appendix Table C.2 into a reproducible CSV."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "source"
    / "henry"
    / "Aqueous Henrys Law Constants Infinite Dilution Activity Coeffic.pdf"
)
DEFAULT_OUTPUT = ROOT / "data" / "source" / "henry" / "brockbank2013_table_c2.csv"

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?"
ROW_RE = re.compile(
    rf"(?P<CAS>\d{{2,7}}-\d{{2}}-\d)\s+"
    rf"(?P<equation>100|101)\s+"
    rf"(?P<A>{NUMBER})\s+(?P<B>{NUMBER})\s+(?P<C>{NUMBER})\s+"
    rf"(?P<D>{NUMBER})\s+(?P<E>{NUMBER})\s+"
    rf"(?P<Tmin_K>[\d.]+)\s+(?P<Tmax_K>[\d.]+)\s+"
    rf"<\s*(?P<uncertainty_upper_percent>\d+)%\s+(?P<data_type>\w+)"
)
EXPECTED_ROWS = 409


def extract_rows(pdf_path: Path) -> list[dict[str, object]]:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    text = result.stdout
    start = text.find("Table C.2: kH regression summary (cont.)")
    end = text.find("Table C.3:", start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not locate Brockbank Appendix Table C.2 boundaries")

    rows = []
    for line in text[start:end].splitlines():
        match = ROW_RE.search(line)
        if match is None:
            continue
        values = match.groupdict()
        rows.append({
            "name": re.sub(r"^\d{3}\s+", "", line[:match.start()].strip()),
            "CAS": values["CAS"],
            "equation": int(values["equation"]),
            "A": float(values["A"]),
            "B": float(values["B"]),
            "C": float(values["C"]),
            "D": float(values["D"]),
            "E": float(values["E"]),
            "Tmin_K": float(values["Tmin_K"]),
            "Tmax_K": float(values["Tmax_K"]),
            "uncertainty_upper_percent": int(values["uncertainty_upper_percent"]),
            "data_type": values["data_type"],
        })
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Extracted {len(rows)} Brockbank Table C.2 rows; expected {EXPECTED_ROWS}"
        )
    return rows


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = extract_rows(args.source)
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
