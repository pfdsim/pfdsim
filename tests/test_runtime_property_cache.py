import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from property_resolution.resolver import PropertyResolver
from property_resolution.runtime_cache import (
    SQLiteJSONCache,
    runtime_cache_path_for_legacy_directory,
    runtime_cache_path_for_legacy_file,
)
from property_resolution.cache_expiration import (
    ONLINE_SOURCE_CACHE_TTL_DAYS,
    SELECTED_PROPERTY_CACHE_TTL_DAYS,
    cache_timestamp_is_fresh,
    sqlite_cache_ttl_days,
)


class RuntimePropertyCacheTests(unittest.TestCase):
    def test_global_sqlite_cache_ttl_policy(self):
        self.assertEqual(
            sqlite_cache_ttl_days(
                'property_cache.sqlite',
                namespace='property_resolver',
            ),
            ONLINE_SOURCE_CACHE_TTL_DAYS,
        )
        self.assertEqual(
            sqlite_cache_ttl_days(
                'property_cache.sqlite',
                namespace='liquid_volume_zra_v1',
            ),
            SELECTED_PROPERTY_CACHE_TTL_DAYS,
        )
        self.assertEqual(
            sqlite_cache_ttl_days(
                'property_cache.sqlite',
                namespace='ideal_gas_cp_derived_v1',
            ),
            SELECTED_PROPERTY_CACHE_TTL_DAYS,
        )
        self.assertEqual(
            sqlite_cache_ttl_days('saturation_properties_cache.sqlite'),
            SELECTED_PROPERTY_CACHE_TTL_DAYS,
        )
        self.assertEqual(
            sqlite_cache_ttl_days('canonical_psat_cache.sqlite'),
            SELECTED_PROPERTY_CACHE_TTL_DAYS,
        )
        self.assertIsNone(sqlite_cache_ttl_days('smiles_cache.sqlite'))
        self.assertIsNone(sqlite_cache_ttl_days('effective_criticals.sqlite'))
        self.assertIsNone(sqlite_cache_ttl_days('henry_constants.sqlite'))
        self.assertIsNone(sqlite_cache_ttl_days('ideal_gas_heat_capacity.sqlite'))
        self.assertIsNone(sqlite_cache_ttl_days('liquid_heat_capacity.sqlite'))

        now = datetime(2026, 8, 12, tzinfo=timezone.utc)
        self.assertTrue(cache_timestamp_is_fresh(
            (now - timedelta(days=30)).isoformat(),
            ttl_days=30,
            now=now,
        ))
        self.assertFalse(cache_timestamp_is_fresh(
            (now - timedelta(days=30, seconds=1)).isoformat(),
            ttl_days=30,
            now=now,
        ))
        self.assertFalse(cache_timestamp_is_fresh(
            'not-a-timestamp',
            ttl_days=30,
            now=now,
        ))

    def test_online_rows_expire_at_90_days_and_derived_rows_at_30(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'property_cache.sqlite')
            online = SQLiteJSONCache(path, 'property_resolver')
            derived = SQLiteJSONCache(path, 'liquid_volume_zra_v1')
            online.set('online-fresh-at-31', {'value': 31})
            online.set('online-stale-at-91', {'value': 91})
            derived.set('derived-stale-at-31', {'value': 31})
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    UPDATE runtime_json_cache
                    SET updated_at_utc = datetime('now', '-31 days')
                    WHERE cache_key IN ('online-fresh-at-31', 'derived-stale-at-31')
                    """
                )
                connection.execute(
                    """
                    UPDATE runtime_json_cache
                    SET updated_at_utc = datetime('now', '-91 days')
                    WHERE cache_key = 'online-stale-at-91'
                    """
                )

            self.assertEqual(online.get('online-fresh-at-31'), {'value': 31})
            self.assertIsNone(online.get('online-stale-at-91'))
            self.assertIsNone(derived.get('derived-stale-at-31'))
            with sqlite3.connect(path) as connection:
                remaining = {
                    row[0] for row in connection.execute(
                        'SELECT cache_key FROM runtime_json_cache'
                    )
                }
            self.assertEqual(remaining, {'online-fresh-at-31'})

    def test_roundtrip_records_query_provenance_and_negative_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'property_cache.sqlite')
            cache = SQLiteJSONCache(path, 'property_resolver')
            cache.set('phase_pubchem_v5_A/B', {
                'Tm': 321.2,
                '_sources': {'Tm': 'pubchem_melting_point_consensus'},
                '_qualities': {'Tm': 0.97},
            })
            cache.set('cp_nist_missing', {'_missing': True, 'reason': 'not_found'})

            self.assertEqual(cache.get('phase_pubchem_v5_A/B')['Tm'], 321.2)
            self.assertTrue(cache.get('cp_nist_missing')['_missing'])
            with sqlite3.connect(path) as connection:
                connection.row_factory = sqlite3.Row
                phase = connection.execute(
                    "SELECT * FROM runtime_json_cache WHERE cache_key = ?",
                    ('phase_pubchem_v5_A/B',),
                ).fetchone()
                missing = connection.execute(
                    "SELECT * FROM runtime_json_cache WHERE cache_key = ?",
                    ('cp_nist_missing',),
                ).fetchone()

            self.assertEqual(phase['namespace'], 'property_resolver')
            self.assertEqual(phase['cache_family'], 'phase_pubchem')
            self.assertEqual(phase['provider'], 'pubchem')
            self.assertEqual(phase['contract_version'], 5)
            self.assertEqual(phase['identifier_key'], 'A/B')
            self.assertEqual(
                json.loads(phase['provenance_json'])['_qualities']['Tm'],
                0.97,
            )
            self.assertEqual(missing['is_missing'], 1)
            self.assertEqual(connection_is_ok(path), 'ok')

    def test_prefix_iteration_is_namespaced_and_key_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'property_cache.sqlite')
            first = SQLiteJSONCache(path, 'property_resolver')
            second = SQLiteJSONCache(path, 'other')
            first.set('antoine_ethanol', {'source': 'NIST WebBook', 'A': 1})
            first.set('antoine_ethanol_canonical_0', {'source': 'NIST WebBook', 'A': 2})
            first.set('antoine_methanol', {'source': 'NIST WebBook', 'A': 3})
            second.set('antoine_ethanol', {'A': 99})

            self.assertEqual(
                [key for key, _payload in first.items(prefix='antoine_ethanol')],
                ['antoine_ethanol', 'antoine_ethanol_canonical_0'],
            )

    def test_concurrent_writers_preserve_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'property_cache.sqlite')

            def write(index):
                SQLiteJSONCache(path, 'parallel').set(
                    f'key-{index}',
                    {'index': index, 'source': 'fixture'},
                )

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(write, range(64)))

            cache = SQLiteJSONCache(path, 'parallel')
            self.assertEqual(cache.count(), 64)
            self.assertEqual(cache.get('key-37')['index'], 37)
            self.assertEqual(connection_is_ok(path), 'ok')

    def test_legacy_url_encoded_json_migration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory, 'legacy')
            legacy.mkdir()
            (legacy / 'phase_pubchem_v5_A%2FB.json').write_text(
                json.dumps({'Tm': 321.2})
            )
            (legacy / 'negative.json').write_text(
                json.dumps({'_missing': True, 'reason': 'not_found'})
            )
            path = Path(directory, 'property_cache.sqlite')
            cache = SQLiteJSONCache(path, 'property_resolver')

            first = cache.migrate_json_directory(legacy, migration_name='fixture')
            second = cache.migrate_json_directory(legacy, migration_name='fixture')

            self.assertEqual(first, {
                'imported': 2,
                'skipped': 0,
                'errors': 0,
                'already_completed': 0,
            })
            self.assertEqual(second['already_completed'], 1)
            self.assertEqual(cache.get('phase_pubchem_v5_A/B')['Tm'], 321.2)
            self.assertEqual(cache.count(), 2)

    def test_failed_legacy_migration_can_be_retried_after_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory, 'legacy')
            legacy.mkdir()
            broken = legacy / 'broken.json'
            broken.write_text('{not json')
            cache = SQLiteJSONCache(Path(directory, 'cache.sqlite'), 'fixture')

            first = cache.migrate_json_directory(legacy, migration_name='retry')
            broken.write_text(json.dumps({'value': 42}))
            second = cache.migrate_json_directory(legacy, migration_name='retry')

            self.assertEqual(first['errors'], 1)
            self.assertEqual(first['already_completed'], 0)
            self.assertEqual(second['errors'], 0)
            self.assertEqual(second['imported'], 1)
            self.assertEqual(cache.get('broken'), {'value': 42})

    def test_legacy_zra_mapping_imports_into_derived_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory, 'liquid_volume_zra_cache.json')
            legacy.write_text(json.dumps({
                'version': 1,
                'fits': {
                    'provided_rhol:fixture': {
                        'Z_RA': 0.25,
                        'kind': 'provided_rhol',
                        'sample_count': 15,
                        'source': 'fixture correlation',
                    },
                },
            }))
            resolver = PropertyResolver()
            with patch.object(
                PropertyResolver,
                'LIQUID_VOLUME_ZRA_CACHE_PATH',
                legacy,
            ):
                payload = resolver._load_liquid_volume_zra_cache()

            self.assertEqual(
                payload['fits']['provided_rhol:fixture']['Z_RA'],
                0.25,
            )
            with sqlite3.connect(legacy.with_suffix('.sqlite')) as connection:
                row = connection.execute(
                    """
                    SELECT payload_json FROM runtime_json_cache
                    WHERE namespace = 'liquid_volume_zra_v1'
                      AND cache_key = 'provided_rhol:fixture'
                    """
                ).fetchone()
            self.assertEqual(json.loads(row[0])['source'], 'fixture correlation')

    def test_custom_legacy_locations_map_to_isolated_sqlite_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            custom_dir = Path(directory, 'cache')
            custom_file = Path(directory, 'zra.json')
            self.assertEqual(
                runtime_cache_path_for_legacy_directory(
                    custom_dir,
                    default_legacy_directory=Path(directory, 'default'),
                ),
                custom_dir / 'property_cache.sqlite',
            )
            self.assertEqual(
                runtime_cache_path_for_legacy_file(
                    custom_file,
                    default_legacy_file=Path(directory, 'default.json'),
                ),
                Path(directory, 'zra.sqlite'),
            )


def connection_is_ok(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        return str(connection.execute('PRAGMA integrity_check').fetchone()[0])


if __name__ == '__main__':
    unittest.main()
