import io
from pathlib import Path
import unittest
import pytest

from chemical_properties import ChemicalDatabase, OnlinePropertyFetcher, SmilesResolution, get_database
from property_resolver import PropertyResolver, get_property_resolver
from property_resolution.runtime_locks import runtime_lock
from thermodynamics_models import interaction_estimation
from tests.cache_isolation import isolated_runtime_caches


@pytest.mark.parametrize('attempt', [0, 1])
def test_pytest_functions_start_without_previous_aliases(attempt):
    # Each parameter is a separate function test. Both must start empty even
    # though the preceding invocation wrote the same disk key and singleton.
    database = get_database(enable_online=False)
    assert database._cached_smiles('X') is None
    assert 'fixture' not in get_property_resolver()._online_cache
    database._cache_smiles(SmilesResolution(
        'CO', 'chemicals', 'fixture', .99, f'pytest invocation {attempt}', 'X',
    ), ['X'])
    get_property_resolver()._online_cache['fixture'] = {'Hf':-201.59}


class RuntimeCacheIsolationTests(unittest.TestCase):
    def test_all_default_cache_owners_use_the_active_directory(self):
        with isolated_runtime_caches() as root:
            database = ChemicalDatabase(enable_online=True)
            resolver = PropertyResolver()
            self.assertEqual(database._smiles_cache_path, root/'smiles_cache.sqlite')
            self.assertEqual(database.online_fetcher.cache_dir, root/'components')
            self.assertEqual(database.online_fetcher._sqlite_cache().path,
                             root/'components'/'property_cache.sqlite')
            self.assertEqual(resolver._runtime_json_cache().path,
                             root/'source'/'property_cache.sqlite')
            self.assertEqual(resolver._liquid_volume_zra_store().path,
                             root/'source'/'property_cache.sqlite')
            self.assertEqual(resolver.SATURATION_PROPERTIES_CACHE_PATH,
                             root/'saturation_properties_cache.sqlite')
            self.assertEqual(resolver.initialize_canonical_vapor_pressure_disk_cache(),
                             root/'canonical_psat_cache.sqlite')
            self.assertEqual(interaction_estimation._fit_cache().path,
                             root/'interaction_estimation_cache.sqlite')
            with runtime_lock('fixture', 'key') as lock:
                self.assertEqual(lock.path, root/'locks.sqlite')

    def test_scopes_do_not_reuse_disk_or_singleton_state(self):
        original_database = get_database(enable_online=False)
        original_resolver = get_property_resolver()
        with isolated_runtime_caches() as first_root:
            database = get_database(enable_online=False)
            resolver = get_property_resolver()
            database._cache_smiles(SmilesResolution(
                'CO', 'chemicals', 'fixture', .99, 'synthetic test alias', 'X',
            ), ['X'])
            resolver._runtime_json_cache().set('fixture', {'Hf':-201.59})
            resolver._online_cache['fixture'] = {'Hf':-201.59}
            self.assertIsNotNone(database._cached_smiles('X'))
        self.assertFalse(first_root.exists())
        with isolated_runtime_caches() as second_root:
            self.assertNotEqual(first_root, second_root)
            self.assertIsNot(get_database(enable_online=False), database)
            self.assertIsNot(get_property_resolver(), resolver)
            self.assertIsNone(get_database()._cached_smiles('X'))
            self.assertIsNone(get_property_resolver()._runtime_json_cache().get('fixture'))
            self.assertNotIn('fixture', get_property_resolver()._online_cache)
        self.assertIs(get_database(), original_database)
        self.assertIs(get_property_resolver(), original_resolver)

    def test_explicit_cache_paths_remain_available_for_persistence_tests(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory, isolated_runtime_caches():
            path = Path(directory)
            fetcher = OnlinePropertyFetcher(cache_dir=path/'online')
            self.assertEqual(fetcher.cache_dir, path/'online')
            resolver = PropertyResolver()
            resolver.CACHE_DIR = path/'properties'
            resolver.CANONICAL_PSAT_CACHE_PATH = path/'canonical.sqlite'
            self.assertEqual(resolver._runtime_json_cache().path,
                             path/'properties'/'property_cache.sqlite')
            self.assertEqual(resolver.initialize_canonical_vapor_pressure_disk_cache(),
                             path/'canonical.sqlite')

    def test_unittest_methods_have_distinct_caches(self):
        paths = []

        class WritesAlias(unittest.TestCase):
            def runTest(self):
                database = get_database(enable_online=False)
                paths.append(database._smiles_cache_path)
                database._cache_smiles(SmilesResolution(
                    'CO', 'chemicals', 'fixture', .99, 'fixture', 'X',
                ), ['X'])

        class ReadsAlias(unittest.TestCase):
            def runTest(self):
                database = get_database(enable_online=False)
                paths.append(database._smiles_cache_path)
                self.assertIsNone(database._cached_smiles('X'))

        suite = unittest.TestSuite([WritesAlias(), ReadsAlias()])
        result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        self.assertTrue(result.wasSuccessful(), result.failures+result.errors)
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(not path.exists() for path in paths))
