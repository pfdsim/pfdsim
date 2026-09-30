#!/usr/bin/env python3
"""Forward auditable, unaffected canonical-Psat cache rows across versions.

Older rows without handoff history are reported as unverified and left in
place. Existing destination rows are never replaced or deleted.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from property_resolution.vapor_pressure import (  # noqa: E402
    CANONICAL_PSAT_CACHE_PATH,
    CANONICAL_PSAT_CACHE_VERSION,
)


DEFAULT_TRUSTED_ANTOINE_PATH = ROOT / "data" / "trusted_other_antoine.json"


def normalize_identifier(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def trusted_antoine_identifiers(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    identifiers = set()
    for record in payload.get("correlations", []):
        identity = record.get("identity", {})
        values = (
            identity.get("cas"),
            identity.get("name"),
            *identity.get("aliases", []),
        )
        identifiers.update(
            normalized
            for value in values
            if (normalized := normalize_identifier(value))
        )
    return identifiers


def row_matches_identifiers(row: sqlite3.Row, identifiers: set[str]) -> bool:
    return any(
        normalize_identifier(row[field]) in identifiers
        for field in ("cas", "component_name", "symbol")
        if field in row.keys() and row[field]
    )


def row_handoff_review_reason(row: sqlite3.Row) -> str | None:
    """Require both retained-source provenance and complete arbitration history."""
    if "provenance_json" not in row.keys() or not row["provenance_json"]:
        return "unverified"
    try:
        provenance = json.loads(row["provenance_json"])
    except (TypeError, ValueError):
        return "unverified"
    if not isinstance(provenance, list) or any(
        not isinstance(item, dict)
        or item.get("segment_type")
        not in ("canonical_override", "pinned", "completion", "fallback")
        for item in provenance
    ):
        return "unverified"
    # Distinct Antoine ranges can share one method name. Counting method names
    # would forward precisely those curves without applying the new budget.
    if (
        sum(
            item.get("segment_type") in {"canonical_override", "pinned"}
            for item in provenance
        )
        >= 2
    ):
        return "multiple_pinned"

    if "metadata_json" not in row.keys() or not row["metadata_json"]:
        return "unverified"
    try:
        metadata = json.loads(row["metadata_json"])
    except (TypeError, ValueError):
        return "unverified"
    decisions = (
        metadata.get("handoff_decisions") if isinstance(metadata, dict) else None
    )
    if not isinstance(decisions, list) or any(
        not isinstance(item, dict)
        or item.get("action") not in ("direct", "bridge", "reject")
        for item in decisions
    ):
        return "unverified"
    # A rejected source can become active when the new policy changes the
    # rejection order, even if only one pinned source survived the old policy.
    if any(item["action"] == "reject" for item in decisions):
        return "rejected_sources"
    return None


def forward_cache(
    database: Path,
    trusted_antoine_path: Path,
    from_version: int,
    to_version: int,
    *,
    apply: bool,
) -> dict[str, object]:
    if from_version == to_version:
        raise ValueError("Source and target cache versions must differ")
    affected_identifiers = trusted_antoine_identifiers(trusted_antoine_path)
    with sqlite3.connect(database, timeout=30.0) as connection:
        connection.row_factory = sqlite3.Row
        columns = [
            str(row[1])
            for row in connection.execute("PRAGMA table_info(canonical_psat_cache)")
        ]
        required = {
            "cache_version",
            "component_key",
            "input_fingerprint",
            "cas",
            "component_name",
            "symbol",
        }
        missing = sorted(required - set(columns))
        if missing:
            raise RuntimeError(
                f"canonical_psat_cache lacks required columns: {missing}"
            )
        source_rows = connection.execute(
            "SELECT * FROM canonical_psat_cache WHERE cache_version = ?",
            (int(from_version),),
        ).fetchall()
        identity_affected = [
            row
            for row in source_rows
            if row_matches_identifiers(row, affected_identifiers)
        ]
        arbitration_reviews = [
            (row, reason)
            for row in source_rows
            if (reason := row_handoff_review_reason(row)) is not None
        ]
        arbitration_affected = [row for row, _reason in arbitration_reviews]
        affected_keys = {
            (row["component_key"], row["input_fingerprint"])
            for row in (*identity_affected, *arbitration_affected)
        }
        affected = [
            row
            for row in source_rows
            if (row["component_key"], row["input_fingerprint"]) in affected_keys
        ]
        forwardable = [
            row
            for row in source_rows
            if (row["component_key"], row["input_fingerprint"]) not in affected_keys
        ]
        existing_rows = connection.execute(
            "SELECT * FROM canonical_psat_cache WHERE cache_version = ?",
            (int(to_version),),
        ).fetchall()
        existing = {
            (row["component_key"], row["input_fingerprint"]): row
            for row in existing_rows
        }
        inserts = []
        already_identical = 0
        existing_target_kept = 0
        comparison_columns = [
            column
            for column in columns
            if column not in {"cache_version", "created_at_utc", "updated_at_utc"}
        ]
        for row in forwardable:
            values = dict(row)
            values["cache_version"] = int(to_version)
            key = (values["component_key"], values["input_fingerprint"])
            present = existing.get(key)
            if present is None:
                inserts.append(values)
                continue
            if all(present[column] == values[column] for column in comparison_columns):
                already_identical += 1
                continue
            # A row already computed under the newer contract is authoritative;
            # forwarding must fill gaps, never replace newer results.
            existing_target_kept += 1

        if apply and inserts:
            placeholders = ", ".join("?" for _ in columns)
            column_sql = ", ".join(columns)
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                f"INSERT INTO canonical_psat_cache ({column_sql}) "
                f"VALUES ({placeholders})",
                [[row[column] for column in columns] for row in inserts],
            )
            connection.execute(f"PRAGMA user_version = {int(to_version)}")
            connection.commit()

        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        return {
            "database": str(database),
            "from_version": int(from_version),
            "to_version": int(to_version),
            "trusted_identifier_count": len(affected_identifiers),
            "source_rows": len(source_rows),
            "affected_rows_skipped": len(affected),
            "trusted_identifier_rows_skipped": len(identity_affected),
            "multi_pinned_rows_skipped": sum(
                reason == "multiple_pinned" for _row, reason in arbitration_reviews
            ),
            "rejected_source_rows_skipped": sum(
                reason == "rejected_sources" for _row, reason in arbitration_reviews
            ),
            "unverified_handoff_rows_skipped": sum(
                reason == "unverified" for _row, reason in arbitration_reviews
            ),
            "rows_to_insert": len(inserts),
            "already_identical": already_identical,
            "existing_target_kept": existing_target_kept,
            "applied": bool(apply),
            "integrity": integrity,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=CANONICAL_PSAT_CACHE_PATH)
    parser.add_argument(
        "--trusted-antoine",
        type=Path,
        default=DEFAULT_TRUSTED_ANTOINE_PATH,
    )
    parser.add_argument(
        "--from-version",
        type=int,
        default=CANONICAL_PSAT_CACHE_VERSION - 1,
    )
    parser.add_argument(
        "--to-version",
        type=int,
        default=CANONICAL_PSAT_CACHE_VERSION,
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = forward_cache(
        args.database.expanduser(),
        args.trusted_antoine.expanduser(),
        args.from_version,
        args.to_version,
        apply=args.apply,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["integrity"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
