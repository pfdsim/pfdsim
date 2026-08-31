#!/usr/bin/env python3
"""Compile CRC constant solid molar volumes into portable SQLite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from chemicals import volume as source_volume
from chemicals.identifiers import check_CAS


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "data" / "solid_volume.sqlite"
SCHEMA_VERSION = 1
CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


def valid_cas(value: str) -> bool:
    text = str(value or "").strip()
    return bool(CAS_PATTERN.fullmatch(text)) and check_CAS(text)


def classify_form(name: str) -> tuple[str, str, bool]:
    lowered = str(name or "").casefold()
    if "hydrate" in lowered:
        return "hydrate", "", True
    if "solvate" in lowered:
        return "solvate", "", True
    if any(token in lowered for token in ("vitreous", "glass", "amorphous")):
        return "glass", "", True
    for token in ("diamond", "graphite", "alpha", "beta", "gamma", "delta", "rutile", "anatase", "brookite"):
        if token in lowered:
            return "crystalline", token, True
    return "crystalline", "", True


def build_rows() -> tuple[list[dict], list[dict]]:
    rows, quarantines = [], []
    for cas, row in source_volume.rho_data_CRC_inorg_s_const.iterrows():
        cas = str(cas)
        name = str(row.Chemical or "").strip()
        try:
            native = float(row.Vm)
        except (TypeError, ValueError):
            native = 0.0
        if not valid_cas(cas):
            quarantines.append({"identifier": cas, "reason": "invalid CAS", "name": name})
            continue
        if not (native > 0.0):
            quarantines.append({"identifier": cas, "reason": "nonpositive molar volume", "name": name})
            continue
        form, polymorph, default = classify_form(name)
        value = native * 1000.0  # m^3/mol -> m^3/kmol
        fingerprint = hashlib.sha256(
            json.dumps([cas, name, native], separators=(",", ":")).encode()
        ).hexdigest()
        rows.append({
            "cas": cas, "name": name,
            "molar_volume_m3_per_kmol": value,
            "material_form": form, "polymorph": polymorph,
            "is_default_form": int(default), "quality": 0.94,
            "source": "crc_solid_inorganic_constant_volume",
            "source_label": "CRC Handbook constant solid molar volume",
            "source_fingerprint": fingerprint,
        })
    return rows, quarantines


def write_database(path: Path, rows: list[dict], quarantines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="solid-volume-", suffix=".sqlite", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with sqlite3.connect(temporary) as connection:
            connection.executescript("""
                PRAGMA journal_mode=DELETE;
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
                CREATE TABLE canonical_solid_volume(
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cas TEXT NOT NULL, name TEXT NOT NULL,
                    molar_volume_m3_per_kmol REAL NOT NULL,
                    material_form TEXT NOT NULL, polymorph TEXT NOT NULL,
                    is_default_form INTEGER NOT NULL, quality REAL NOT NULL,
                    source TEXT NOT NULL, source_label TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL
                );
                CREATE UNIQUE INDEX idx_solid_volume_cas_form
                ON canonical_solid_volume(cas, material_form, polymorph);
                CREATE TABLE source_candidate_audit(
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier TEXT NOT NULL, source TEXT NOT NULL,
                    status TEXT NOT NULL, reason TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
            """)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "record_count": len(rows), "quarantine_count": len(quarantines),
                "material_forms": dict(Counter(item["material_form"] for item in rows)),
                "native_source_units": "m^3/mol",
                "runtime_units": "m^3/kmol",
                "thermal_expansion_policy": "zero; quality penalized with distance from 298.15 K",
            }
            connection.executemany(
                "INSERT INTO metadata(key,value_json) VALUES (?,?)",
                [(key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()],
            )
            connection.executemany("""
                INSERT INTO canonical_solid_volume(
                    cas,name,molar_volume_m3_per_kmol,material_form,polymorph,
                    is_default_form,quality,source,source_label,source_fingerprint
                ) VALUES (:cas,:name,:molar_volume_m3_per_kmol,:material_form,:polymorph,
                          :is_default_form,:quality,:source,:source_label,:source_fingerprint)
            """, rows)
            connection.executemany(
                "INSERT INTO source_candidate_audit(identifier,source,status,reason,details_json) VALUES (?,?,?,?,?)",
                [(row["cas"], row["source"], "admitted", "", json.dumps({"name": row["name"]})) for row in rows]
                + [(item["identifier"], "crc_solid_inorganic_constant_volume", "quarantined", item["reason"], json.dumps(item)) for item in quarantines],
            )
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("solid-volume SQLite integrity check failed")
        temporary.replace(path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows, quarantines = build_rows()
    write_database(args.output, rows, quarantines)
    print(json.dumps({
        "output": str(args.output), "records": len(rows),
        "quarantines": len(quarantines),
        "material_forms": dict(Counter(row["material_form"] for row in rows)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
