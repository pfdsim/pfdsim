#!/usr/bin/env python3
"""One-time cleanup for stale online property-cache entries in SQLite.

This removes Perry-derived fields from cached online ChemicalProperties objects
when the current strict Perry identity lookup cannot match the compound by
symbol, name, or CAS. It is intended for repairing cache entries created before
ambiguous molecular formulas were blocked from isomer expansion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perry_properties import get_perry_property_library  # noqa: E402
from property_resolution.runtime_cache import (  # noqa: E402
    LEGACY_ONLINE_COMPONENT_CACHE_DIR,
    RUNTIME_PROPERTY_CACHE_PATH,
    SQLiteJSONCache,
)


def is_perry_source(source: object) -> bool:
    if not isinstance(source, dict):
        return False
    return any(
        "perry" in str(source.get(key, "")).lower()
        for key in ("source", "method", "notes")
    )


def has_exact_perry_match(payload: dict, library) -> bool:
    for key in ("symbol", "name", "CAS"):
        value = payload.get(key)
        if value and library.get(str(value)):
            return True
    return False


def sanitize_payload(payload: dict, library) -> list[str]:
    sources = payload.get("property_sources")
    if not isinstance(sources, dict):
        return []

    perry_fields = [field for field, source in sources.items() if is_perry_source(source)]
    if not perry_fields or has_exact_perry_match(payload, library):
        return []

    changed = []
    for field in perry_fields:
        if field in payload and payload[field] is not None:
            payload[field] = None
            changed.append(field)
        sources.pop(field, None)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-path",
        default=str(RUNTIME_PROPERTY_CACHE_PATH),
        help="Runtime property SQLite database to scan.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write cleaned cache rows. Without this, only report changes.",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_path).expanduser()
    if not cache_path.exists():
        print(f"Cache database does not exist: {cache_path}")
        return 0

    cache = SQLiteJSONCache(cache_path, 'online_property_fetcher')
    cache.migrate_json_directory(
        LEGACY_ONLINE_COMPONENT_CACHE_DIR,
        migration_name='online-property-fetcher-cache',
    )
    library = get_perry_property_library()
    changed_rows: list[tuple[str, list[str]]] = []

    for cache_key, payload in cache.items():
        if not isinstance(payload, dict) or payload.get("_missing"):
            continue

        changed = sanitize_payload(payload, library)
        if not changed:
            continue
        changed_rows.append((cache_key, changed))
        if args.apply:
            cache.set(cache_key, payload)

    mode = "updated" if args.apply else "would update"
    print(f"{mode} {len(changed_rows)} cache row(s)")
    for cache_key, fields in changed_rows[:20]:
        print(f"  {cache_key}: {', '.join(fields)}")
    if len(changed_rows) > 20:
        print(f"  ... {len(changed_rows) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
