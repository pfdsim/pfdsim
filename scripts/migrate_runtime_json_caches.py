#!/usr/bin/env python3
"""Import every legacy runtime JSON cache into the shared SQLite database."""

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
    LEGACY_ONLINE_COMPONENT_CACHE_DIR,
    LEGACY_PROPERTY_CACHE_DIR,
    RUNTIME_PROPERTY_CACHE_PATH,
    SQLiteJSONCache,
)


LEGACY_ZRA_PATH = ROOT / 'data' / 'liquid_volume_zra_cache.json'


def _migrate_zra(database: Path) -> dict[str, int]:
    cache = SQLiteJSONCache(database, 'liquid_volume_zra_v1')
    imported = skipped = errors = 0
    try:
        payload = json.loads(LEGACY_ZRA_PATH.read_text())
        fits = payload.get('fits') if isinstance(payload, dict) else None
    except (OSError, TypeError, ValueError):
        fits = None
    if not isinstance(fits, dict):
        return {'imported': 0, 'skipped': 0, 'errors': int(LEGACY_ZRA_PATH.exists())}
    for key, fit in fits.items():
        if not isinstance(fit, dict):
            errors += 1
        elif cache.get(str(key)) is not None:
            skipped += 1
        else:
            cache.set(str(key), fit)
            imported += 1
    return {'imported': imported, 'skipped': skipped, 'errors': errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--database',
        type=Path,
        default=RUNTIME_PROPERTY_CACHE_PATH,
        help='Destination SQLite database.',
    )
    args = parser.parse_args()
    database = args.database.expanduser()

    resolver_cache = SQLiteJSONCache(database, 'property_resolver')
    component_cache = SQLiteJSONCache(database, 'online_property_fetcher')
    results = {
        'property_resolver': resolver_cache.migrate_json_directory(
            LEGACY_PROPERTY_CACHE_DIR,
            migration_name='property-resolver-cache',
        ),
        'online_property_fetcher': component_cache.migrate_json_directory(
            LEGACY_ONLINE_COMPONENT_CACHE_DIR,
            migration_name='online-property-fetcher-cache',
        ),
        'liquid_volume_zra_v1': _migrate_zra(database),
    }

    with sqlite3.connect(database) as connection:
        integrity = str(connection.execute('PRAGMA integrity_check').fetchone()[0])
        counts = dict(connection.execute(
            'SELECT namespace, count(*) FROM runtime_json_cache GROUP BY namespace'
        ))
    for namespace, result in results.items():
        print(f'{namespace}: {result}')
    print(f'rows: {counts}')
    print(f'integrity: {integrity}')
    return 0 if integrity == 'ok' and all(not item['errors'] for item in results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
