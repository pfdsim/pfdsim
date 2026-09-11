#!/usr/bin/env python3
"""Forward cached phase-change provider payloads to Hvap quality contract v9."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from property_resolution.runtime_cache import (  # noqa: E402
    RUNTIME_PROPERTY_CACHE_PATH,
    cache_key_metadata,
)


def _replace_exact(value, old: float, new: float):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return new if numeric == old else value


def _forward_payload(provider: str, payload: dict) -> tuple[dict, int]:
    forwarded = json.loads(json.dumps(payload))
    changed = 0
    records = forwarded.get("Hvap_records")
    if not isinstance(records, list):
        return forwarded, changed
    for record in records:
        if not isinstance(record, dict):
            continue
        quality = record.get("quality")
        updated = quality
        if provider == "nist" and record.get("source") == "nist_phase_change":
            updated = _replace_exact(updated, 0.90, 0.91)
            updated = _replace_exact(updated, 0.93, 0.94)
        elif provider == "pubchem" and record.get("source") == "pubchem":
            if record.get("T_ref") is not None:
                updated = _replace_exact(updated, 0.88, 0.89)
            elif record.get("basis") == "normal_boiling_point":
                updated = _replace_exact(updated, 0.76, 0.82)
        if updated != quality:
            record["quality"] = updated
            changed += 1
    return forwarded, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=RUNTIME_PROPERTY_CACHE_PATH)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    database = args.database.expanduser()
    with sqlite3.connect(database, timeout=30.0) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM runtime_json_cache
            WHERE namespace = 'property_resolver'
              AND contract_version = 8
              AND cache_family IN ('phase_nist', 'phase_pubchem')
            ORDER BY cache_key
            """
        ).fetchall()
        forwarded = []
        changed_records = 0
        for row in rows:
            payload = json.loads(row["payload_json"])
            new_payload, changes = _forward_payload(row["provider"], payload)
            new_key = row["cache_key"].replace("_v8_", "_v9_", 1)
            metadata = cache_key_metadata(new_key, new_payload)
            forwarded.append((row, new_key, new_payload, metadata))
            changed_records += changes

        existing = connection.execute(
            """
            SELECT COUNT(*) FROM runtime_json_cache
            WHERE namespace = 'property_resolver'
              AND contract_version = 9
              AND cache_family IN ('phase_nist', 'phase_pubchem')
            """
        ).fetchone()[0]
        print(
            f"source_rows={len(rows)} changed_hvap_records={changed_records} "
            f"existing_target_rows={existing} apply={args.apply}"
        )
        if not args.apply:
            return 0
        if existing:
            raise RuntimeError(
                "Version-9 phase-change rows already exist; refusing to overwrite them"
            )

        connection.execute("BEGIN IMMEDIATE")
        for row, new_key, payload, metadata in forwarded:
            connection.execute(
                """
                INSERT INTO runtime_json_cache (
                    namespace, cache_key, cache_family, provider,
                    contract_version, identifier_key, payload_json,
                    provenance_json, is_missing, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["namespace"],
                    new_key,
                    metadata["cache_family"],
                    metadata["provider"],
                    metadata["contract_version"],
                    metadata["identifier_key"],
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ),
                    metadata["provenance_json"],
                    row["is_missing"],
                    row["created_at_utc"],
                    row["updated_at_utc"],
                ),
            )
        connection.commit()
        target = connection.execute(
            """
            SELECT COUNT(*) FROM runtime_json_cache
            WHERE namespace = 'property_resolver'
              AND contract_version = 9
              AND cache_family IN ('phase_nist', 'phase_pubchem')
            """
        ).fetchone()[0]
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"inserted={target} integrity={integrity}")
        return 0 if target == len(rows) and integrity == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
