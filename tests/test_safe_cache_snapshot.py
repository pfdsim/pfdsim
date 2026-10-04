import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from property_resolution.runtime_cache import SQLiteJSONCache
from scripts.snapshot_test_cache import snapshot


class SafeCacheSnapshotTests(unittest.TestCase):
    def test_export_excludes_negative_derived_obsolete_and_unknown_alias_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'source.sqlite')
            cache = SQLiteJSONCache(path, 'property_resolver')
            cache.set('phase_nist_v9_71-36-3', {'Tb': 390.85, '_sources': {'Tb': 'nist'}})
            cache.set('phase_nist_v9_X', {'Tb': 337.85, '_sources': {'Tb': 'nist'}})
            cache.set('phase_nist_v8_71-36-3', {'Tb': 390.85})
            cache.set('phase_nist_v9_ethanol', {'_missing': True, 'reason': 'not_found'})
            SQLiteJSONCache(path, 'ideal_gas_cp_derived_v1').set('answer', {'value': 12.0})
            exported = snapshot(path)
            self.assertEqual(len(exported['records']), 1)
            self.assertEqual(exported['records'][0]['keys'], ['phase_nist_v9_71-36-3'])
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(connection.execute('SELECT count(*) FROM runtime_json_cache').fetchone()[0], 5)

    def test_checked_in_baseline_contains_sources_and_no_selected_answers(self):
        fixture = json.loads((Path(__file__).parent/'fixtures'/'runtime_cache.json').read_text())
        self.assertGreater(len(fixture['records']), 0)
        for record in fixture['records']:
            self.assertIn(record['namespace'], {'property_resolver', 'online_property_fetcher'})
            self.assertFalse(record['payload'].get('_missing'))
            self.assertTrue(record['captured_at_utc'])
            self.assertNotIn('X', record['keys'])
