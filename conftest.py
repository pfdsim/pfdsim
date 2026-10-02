import os

import pytest

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .tests.cache_isolation import isolated_runtime_caches
else:
    from tests.cache_isolation import isolated_runtime_caches


@pytest.fixture(scope='session', autouse=True)
def runtime_cache_session():
    """Protect shared class fixtures before individual test scopes begin."""
    with isolated_runtime_caches():
        yield


@pytest.fixture(autouse=True)
def runtime_cache_test(runtime_cache_session):
    """Give pytest function tests fresh disk and in-memory caches."""
    with isolated_runtime_caches():
        yield


@pytest.fixture(autouse=True)
def disable_automatic_xtb_rrho(monkeypatch):
    """Keep ordinary tests from accidentally launching optional QM work.

    Focused RRHO tests explicitly patch this dependency probe with the state
    they need. Real-backend behavior is exercised by the benchmark scripts and
    dedicated manual smoke probes rather than implicitly by unrelated tests.
    """
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .property_resolution.heat_capacity import HeatCapacityMixin
    else:
        from property_resolution.heat_capacity import HeatCapacityMixin

    monkeypatch.setattr(
        HeatCapacityMixin,
        '_xtb_rrho_dependency_state',
        staticmethod(lambda: {
            'tblite': 'missing',
            'ase': 'missing',
            'rdkit': 'missing',
        }),
    )


def pytest_cmdline_main(config):
    args = list(config.invocation_params.args)
    if args or os.environ.get('PFDSIM_PARALLEL_DEFAULT_ACTIVE'):
        return None

    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .tests.default_parallel import run_parallel_default
    else:
        from tests.default_parallel import run_parallel_default

    return run_parallel_default()
