#!/usr/bin/env python3
"""Extract Table 1 from ACS JCED 2015 critical-property review.

Source paper:
    Vapor-Liquid Critical Properties of Elements and Compounds. 12.
    Review of Recent Data for Hydrocarbons and Non-hydrocarbons
    J. Chem. Eng. Data 2015, 60, 3444-3482
    DOI: 10.1021/acs.jced.5b00571

The generated JSON is keyed by CASRN and stores recommended Table 1 values.
CAS resolution prefers CASRN headings in the article's Table 2, with
``chemicals`` name lookup and explicit PDF-text corrections as fallbacks.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = (
    ROOT
    / "data"
    / "reference"
    / "pure-component-properties"
    / "2016-ambrose-vapor-liquid-critical-properties-review.pdf"
)
DEFAULT_OUTPUT = ROOT / "data" / "acs_jced_5b00571_table1.json"

SOURCE = {
    "title": (
        "Vapor-Liquid Critical Properties of Elements and Compounds. 12. "
        "Review of Recent Data for Hydrocarbons and Non-hydrocarbons"
    ),
    "journal": "J. Chem. Eng. Data",
    "year": 2015,
    "volume": 60,
    "pages": "3444-3482",
    "doi": "10.1021/acs.jced.5b00571",
    "table": "Table 1",
}

ROW_RE = re.compile(
    r"^\s*(?P<name>\S.*?\S)\s{2,}"
    r"(?P<molar_mass>\d{1,4}\.\d{2,3})\s+"
    r"(?P<rest>.*\S)\s*$"
)
NUMBER_RE = re.compile(r"(?<![A-Za-z])\(?\d+(?:\.\d+)?\)?")
CAS_RE = re.compile(r"\d{2,7}-\d{2}-\d{1,2}")
HEADING_RE = re.compile(
    r"^\s*(?P<name>.+?):\s+Molar\s+[Mm]ass,\s+"
    r"(?P<molar_mass>\d+(?:\.\d+)?)(?:\s*g)?;\s+"
    r"(?P<formula>[^;]+);\s+CASRN\s+"
    r"(?P<cas>\d{2,7}-\d{2}-\d{1,2})"
)

UNICODE_REPLACEMENTS = {
    "η": "eta",
    "γ": "gamma",
    "−": "-",
    "–": "-",
    "—": "-",
    "′": "'",
    "’": "'",
}

# Corrections for PDF/Table wording that does not exactly match a resolvable
# Table 2 heading. These are intentionally narrow and source-note-bearing.
MANUAL_CAS_CORRECTIONS = {
    "docasane": {
        "CAS": "629-97-0",
        "note": "Table 1 text spells docasane; Table 2 heading/common name is docosane.",
    },
    "heptybenzene": {
        "CAS": "1078-71-3",
        "note": "Table 1 text appears to omit the l in heptylbenzene.",
    },
    "4oxa17heptanediol": {
        "CAS": "25265-71-8",
        "note": "Resolved from Table 2 heading alias Dipropylene Glycol.",
    },
    "butylpropenoate": {
        "CAS": "141-32-2",
        "note": "Resolved from Table 2 heading BUTYL 2-PROPENOATE (Butyl Acrylate).",
    },
    "z9methyl2octadecenoate": {
        "CAS": "112-62-9",
        "note": "Resolved from Table 2 heading (Z)-9-METHYL OCTADECENOATE (Methyl Oleate).",
    },
    "5phenylpentanoic": {
        "CAS": "2270-20-4",
        "note": "Table 1 text omits acid; resolved as 5-phenylpentanoic acid.",
    },
    "bis2ethylhexyl12benzenedicarboxylate": {
        "CAS": "117-81-7",
        "note": "Resolved from wrapped Table 2 heading for bis(2-ethylhexyl) phthalate.",
    },
    "bisetacyclopentadienyliron": {
        "CAS": "102-54-5",
        "note": "Resolved from bis(eta-cyclopentadienyl) iron / ferrocene.",
    },
}

HEADING_CAS_CORRECTIONS = {
    "98-06-06": {
        "CAS": "98-06-6",
        "note": "PDF text extraction includes an extra zero in tert-butylbenzene CASRN.",
    },
}

HEADING_NAME_CAS_CORRECTIONS = {
    "dipentyl12benzenedicarboxylate": {
        "CAS": "131-18-0",
        "note": (
            "Article text extraction reports the dibutyl phthalate CAS for "
            "dipentyl phthalate; corrected with chemicals metadata."
        ),
    },
}

# Corrections to values misprinted in the article itself (not extraction bugs).
VALUE_CORRECTIONS = {
    "111-46-6": {
        "critical_properties": {
            "Tc_K": 753.0,
            "Tc_uncertainty_K": 8.0,
            "Pc_MPa": 4.77,
            "Pc_uncertainty_MPa": 0.2,
        },
        "note": (
            "Article Table 1 prints the 3-oxa-1,5-pentanediol (diethylene "
            "glycol) row with Tc/pc identical to the preceding "
            "2,2-dimethyl-1,3-propanediol row (687 K, 4.20 MPa) -- a "
            "typesetting duplication; the values also break the DEG/TEG/"
            "tetraEG homologous trend.  Replaced with Nikitin's pulse-heating "
            "measurements: Tc = 753 +/- 8 K, pc = 4.77 +/- 0.2 MPa "
            "(verified by PFDSim, 2026-07-09)."
        ),
    },
}


def normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def display_text(text: str) -> str:
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def run_pdftotext(pdf_path: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), tmp.name],
            check=True,
        )
        return Path(tmp.name).read_text()


def heading_variants(name: str) -> set[str]:
    variants = {name}
    # Preserve leading stereochemical parentheticals, but remove trailing aliases.
    variants.add(re.sub(r"(?<!^)\([^)]*\)", "", name).strip())
    variants.add(re.sub(r"\([^)]*\)$", "", name).strip())
    variants.update(alias.strip() for alias in re.findall(r"\(([^()]*)\)", name))
    return {variant for variant in variants if variant}


def parse_article_headings(lines: list[str]) -> dict[str, dict[str, Any]]:
    headings: dict[str, dict[str, Any]] = {}

    for index, line in enumerate(lines):
        candidates = [line]
        if line.rstrip().endswith(":") and index + 1 < len(lines):
            candidates.append(line.rstrip() + " " + lines[index + 1].strip())

        for candidate in candidates:
            match = HEADING_RE.match(candidate)
            if not match:
                continue

            raw_cas = match.group("cas")
            cas = raw_cas
            notes: list[str] = []
            if raw_cas in HEADING_CAS_CORRECTIONS:
                correction = HEADING_CAS_CORRECTIONS[raw_cas]
                cas = correction["CAS"]
                notes.append(correction["note"])

            name = display_text(match.group("name"))
            correction_key_candidates = {
                normalize_key(name),
                normalize_key(re.sub(r"(?<!^)\([^)]*\)", "", name).strip()),
                normalize_key(re.sub(r"\([^)]*\)$", "", name).strip()),
            }
            correction_key = next(
                (
                    key for key in correction_key_candidates
                    if key in HEADING_NAME_CAS_CORRECTIONS
                ),
                None,
            )
            if correction_key:
                correction = HEADING_NAME_CAS_CORRECTIONS[correction_key]
                cas = correction["CAS"]
                notes.append(correction["note"])

            record = {
                "article_table2_name": name,
                "formula": match.group("formula").strip(),
                "molar_mass_g_mol": float(match.group("molar_mass")),
                "CAS": cas,
                "article_table2_line": index + 1,
            }
            if notes:
                record["cas_notes"] = notes

            for variant in heading_variants(name):
                headings.setdefault(normalize_key(variant), record)

    return headings


def find_table1_bounds(lines: list[str]) -> tuple[int, int]:
    start = next(
        i for i, line in enumerate(lines)
        if "Table 1. Recommended Critical Properties" in line
    )
    end = next(
        i for i, line in enumerate(lines[start + 1 :], start + 1)
        if "Uncertainties in the experimental values" in line
    )
    return start, end


def parse_table1_rows(lines: list[str]) -> list[dict[str, Any]]:
    start, end = find_table1_bounds(lines)
    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(lines[start:end], start + 1):
        match = ROW_RE.match(line)
        if not match:
            continue

        values = [
            float(number.group(0).strip("()"))
            for number in NUMBER_RE.finditer(match.group("rest"))
        ]
        if len(values) < 2:
            continue

        name = display_text(match.group("name"))
        original_name = name
        molar_mass = float(match.group("molar_mass"))
        notes: list[str] = []

        if name == "cyclopentane" and abs(molar_mass - 68.119) < 1e-3:
            name = "cyclopentene"
            notes.append(
                "Table 1 text extracts/prints cyclopentane with C5H8 molar mass; "
                "resolved as cyclopentene."
            )

        row = {
            "line": line_number,
            "name": name,
            "original_name": original_name,
            "molar_mass_g_mol": molar_mass,
            "Tc_K": values[0],
            "Tc_uncertainty_K": values[1],
        }
        if len(values) >= 4:
            row["Pc_MPa"] = values[2]
            row["Pc_uncertainty_MPa"] = values[3]
        if len(values) >= 6:
            row["rho_c_g_cm3"] = values[4]
            row["rho_c_uncertainty_g_cm3"] = values[5]
        if len(values) >= 7:
            row["Vc_cm3_mol"] = values[6]
        if len(values) >= 8:
            row["Zc"] = values[7]
        if notes:
            row["extraction_notes"] = notes
        rows.append(row)

    return rows


def chemicals_lookup(name: str) -> dict[str, Any] | None:
    try:
        from chemicals import identifiers
    except Exception:
        return None

    for candidate in (name, name.replace("eta", "η")):
        try:
            metadata = identifiers.pubchem_db.search_name(candidate)
        except Exception:
            metadata = None
        if metadata is None:
            continue
        return {
            "CAS": getattr(metadata, "CASs", None),
            "formula": getattr(metadata, "formula", None),
        }
    return None


def chemicals_formula_for_cas(cas: str) -> str | None:
    try:
        from chemicals import identifiers
        metadata = identifiers.pubchem_db.search_CAS(cas)
    except Exception:
        return None
    return getattr(metadata, "formula", None) if metadata is not None else None


def resolve_cas(
    row: dict[str, Any],
    headings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = normalize_key(row["name"])
    notes: list[str] = list(row.get("extraction_notes", []))

    if key in headings:
        heading = headings[key]
        notes.extend(heading.get("cas_notes", []))
        return {
            **row,
            "CAS": heading["CAS"],
            "formula": heading.get("formula"),
            "cas_resolution_method": "article_table2_heading",
            "article_table2_name": heading.get("article_table2_name"),
            "article_table2_line": heading.get("article_table2_line"),
            **({"extraction_notes": notes} if notes else {}),
        }

    if key in MANUAL_CAS_CORRECTIONS:
        correction = MANUAL_CAS_CORRECTIONS[key]
        notes.append(correction["note"])
        cas = correction["CAS"]
        return {
            **row,
            "CAS": cas,
            "formula": chemicals_formula_for_cas(cas),
            "cas_resolution_method": "manual_pdf_name_correction",
            "extraction_notes": notes,
        }

    lookup = chemicals_lookup(row["name"])
    if lookup and lookup.get("CAS"):
        return {
            **row,
            "CAS": lookup["CAS"],
            "formula": lookup.get("formula"),
            "cas_resolution_method": "chemicals_search_name",
            **({"extraction_notes": notes} if notes else {}),
        }

    raise LookupError(f"Could not resolve CAS for Table 1 row {row['line']}: {row['name']}")


def merge_entries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_cas: dict[str, dict[str, Any]] = {}

    for row in rows:
        cas = row["CAS"]
        entry = {
            "name": row["name"],
            "article_name": row["original_name"],
            "formula": row.get("formula"),
            "molar_mass_g_mol": row["molar_mass_g_mol"],
            "critical_properties": {
                "Tc_K": row["Tc_K"],
                "Tc_uncertainty_K": row["Tc_uncertainty_K"],
            },
            "cas_resolution_method": row["cas_resolution_method"],
            "source_table1_line": row["line"],
        }
        if "Pc_MPa" in row:
            entry["critical_properties"]["Pc_MPa"] = row["Pc_MPa"]
            entry["critical_properties"]["Pc_uncertainty_MPa"] = row["Pc_uncertainty_MPa"]
        if "rho_c_g_cm3" in row:
            entry["critical_properties"]["rho_c_g_cm3"] = row["rho_c_g_cm3"]
            entry["critical_properties"]["rho_c_uncertainty_g_cm3"] = row[
                "rho_c_uncertainty_g_cm3"
            ]
        if "Vc_cm3_mol" in row:
            entry["critical_properties"]["Vc_cm3_mol"] = row["Vc_cm3_mol"]
        if "Zc" in row:
            entry["critical_properties"]["Zc"] = row["Zc"]
        for optional_key in ("article_table2_name", "article_table2_line", "extraction_notes"):
            if optional_key in row:
                entry[optional_key] = row[optional_key]

        if cas in by_cas:
            existing = by_cas[cas]
            if existing["critical_properties"] == entry["critical_properties"]:
                existing.setdefault("duplicate_table1_rows", []).append(entry)
                existing.setdefault("extraction_notes", []).append(
                    f"Duplicate Table 1 row with same values at line {row['line']}."
                )
                continue
            existing.setdefault("alternate_table1_rows", []).append(entry)
            existing.setdefault("extraction_notes", []).append(
                "Table 1 contains another row with the same CAS but different "
                f"values at line {row['line']}; preserved under alternate_table1_rows."
            )
            continue
        by_cas[cas] = entry

    return dict(sorted(by_cas.items()))


def build_payload(pdf_path: Path) -> dict[str, Any]:
    text = run_pdftotext(pdf_path)
    lines = text.splitlines()
    headings = parse_article_headings(lines)
    table1_rows = parse_table1_rows(lines)
    resolved_rows = [resolve_cas(row, headings) for row in table1_rows]
    chemicals = merge_entries(resolved_rows)

    for cas, correction in VALUE_CORRECTIONS.items():
        entry = chemicals.get(cas)
        if entry is None:
            continue
        entry["critical_properties"].update(correction["critical_properties"])
        notes = entry.setdefault("extraction_notes", [])
        if correction["note"] not in notes:
            notes.append(correction["note"])

    property_counts = Counter()
    for row in table1_rows:
        property_counts["Tc"] += 1
        if "Pc_MPa" in row:
            property_counts["Pc"] += 1
        if "Vc_cm3_mol" in row:
            property_counts["Vc"] += 1

    return {
        "metadata": {
            "source": SOURCE,
            "pdf": str(pdf_path.relative_to(ROOT) if pdf_path.is_relative_to(ROOT) else pdf_path),
            "extraction_method": "pdftotext -layout plus Table 2 CAS heading resolution",
            "units": {
                "molar_mass": "g/mol",
                "Tc": "K",
                "Pc": "MPa",
                "critical_density": "g/cm^3",
                "Vc": "cm^3/mol",
                "Zc": "dimensionless",
            },
            "table1_row_count": len(table1_rows),
            "cas_key_count": len(chemicals),
            "property_counts": dict(property_counts),
        },
        "chemicals": chemicals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(args.pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    counts = payload["metadata"]["property_counts"]
    print(f"Wrote {args.output}")
    print(
        "Rows: {rows}; CAS keys: {cas}; Tc: {Tc}; Pc: {Pc}; Vc: {Vc}".format(
            rows=payload["metadata"]["table1_row_count"],
            cas=payload["metadata"]["cas_key_count"],
            **counts,
        )
    )


if __name__ == "__main__":
    main()
