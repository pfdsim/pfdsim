"""Test-only isolation of writable property caches and their singleton owners."""

from contextlib import contextmanager, ExitStack
from functools import wraps
import importlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


_active_scopes = 0


@contextmanager
def isolated_runtime_caches():
    """Use fresh caches while retaining explicit paths configured by a test."""
    global _active_scopes
    prefix = 'pfdsim.' if __package__.split('.', 1)[0] == 'pfdsim' else ''
    prefixes = {prefix}
    for candidate in ('', 'pfdsim.'):
        if candidate+'chemical_properties' in sys.modules:
            prefixes.add(candidate)

    with TemporaryDirectory(prefix='pfdsim-test-cache-') as directory, ExitStack() as stack:
        root = Path(directory)
        for selected in sorted(prefixes):
            stack.enter_context(_patch_runtime_caches(root, selected))
        _active_scopes += 1
        try:
            yield root
        finally:
            _active_scopes -= 1


@contextmanager
def _patch_runtime_caches(root, prefix):
    """Patch each loaded import namespace without introducing its counterpart."""
    chemical = importlib.import_module(prefix+'chemical_properties')
    facade = importlib.import_module(prefix+'property_resolver')
    base = importlib.import_module(prefix+'property_resolution.base')
    cache = importlib.import_module(prefix+'property_resolution.runtime_cache')
    locks = importlib.import_module(prefix+'property_resolution.runtime_locks')
    adapter = importlib.import_module(prefix+'property_resolution.vapor_pressure_adapter')
    estimation = importlib.import_module(prefix+'thermodynamics_models.interaction_estimation')

    with ExitStack() as stack:
        source_directory = root/'source'
        component_directory = root/'components'
        property_path = source_directory/'property_cache.sqlite'
        original_database_init = chemical.ChemicalDatabase.__init__
        original_fetcher_init = chemical.OnlinePropertyFetcher.__init__

        @wraps(original_database_init)
        def database_init(self, *args, **kwargs):
            original_database_init(self, *args, **kwargs)
            self._smiles_cache_path = root/'smiles_cache.sqlite'

        @wraps(original_fetcher_init)
        def fetcher_init(self, cache_dir=None):
            original_fetcher_init(self, component_directory if cache_dir is None else cache_dir)

        for owner, name, value in (
            (chemical.ChemicalDatabase, '__init__', database_init),
            (chemical.OnlinePropertyFetcher, '__init__', fetcher_init),
            (chemical, '_db', None),
            (facade, '_resolver', None),
            (cache, 'RUNTIME_PROPERTY_CACHE_PATH', property_path),
            (adapter, 'ANTOINE_CACHE_DIR', source_directory),
            (estimation, '_FIT_CACHE_PATH', root/'interaction_estimation_cache.sqlite'),
            (estimation, '_FIT_CACHE', None),
        ):
            stack.enter_context(patch.object(owner, name, value))
        for owner in (base.PropertyResolverBase, facade.PropertyResolver):
            for name, value in (
                ('CACHE_DIR', source_directory),
                ('RUNTIME_CACHE_PATH', property_path),
                ('LIQUID_VOLUME_ZRA_CACHE_PATH', property_path),
            ):
                stack.enter_context(patch.object(owner, name, value))
        for name, value in (
            ('SATURATION_PROPERTIES_CACHE_PATH', root/'saturation_properties_cache.sqlite'),
            ('CANONICAL_PSAT_CACHE_PATH', root/'canonical_psat_cache.sqlite'),
        ):
            stack.enter_context(patch.object(facade.PropertyResolver, name, value))
        # runtime_lock's default path was bound when its function was defined.
        stack.enter_context(patch.object(locks.runtime_lock, '__kwdefaults__', {
            **locks.runtime_lock.__kwdefaults__, 'path':root/'locks.sqlite',
        }))
        yield


def install_unittest_cache_isolation():
    """Apply the same isolation to explicit unittest runs and default workers."""
    if getattr(unittest.TestCase.run, '_pfdsim_cache_isolation', False):
        return
    original_test_run = unittest.TestCase.run
    original_suite_run = unittest.TestSuite.run

    @wraps(original_test_run)
    def test_run(self, result=None):
        if not type(self).__module__.split('.')[-1].startswith('test_'):
            return original_test_run(self, result)
        with isolated_runtime_caches():
            return original_test_run(self, result)

    @wraps(original_suite_run)
    def suite_run(self, result, debug=False):
        # Module/class fixtures run outside TestCase.run and need safe caches,
        # too. Nested suites use their enclosing scope; methods get fresh ones.
        if _active_scopes:
            return original_suite_run(self, result, debug)
        with isolated_runtime_caches():
            return original_suite_run(self, result, debug)

    test_run._pfdsim_cache_isolation = True
    unittest.TestCase.run = test_run
    unittest.TestSuite.run = suite_run
