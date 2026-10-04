import os
import unittest

import pytest

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .tests.cache_isolation import isolated_runtime_caches, uses_empty_runtime_cache
    from .tests.reporting import serializable_subtest_context
    from .tests.default_parallel import WORKER_COUNT
else:
    from tests.cache_isolation import isolated_runtime_caches, uses_empty_runtime_cache
    from tests.reporting import serializable_subtest_context
    from tests.default_parallel import WORKER_COUNT


@pytest.fixture(scope='session', autouse=True)
def runtime_cache_session():
    """Protect shared class fixtures before individual test scopes begin."""
    with isolated_runtime_caches():
        yield


@pytest.fixture(autouse=True)
def runtime_cache_test(runtime_cache_session, request):
    """Give pytest function tests fresh disk and in-memory caches."""
    if isinstance(request.instance, unittest.TestCase):
        # TestCase.run supplies the method scope, including nested unittest
        # suites; avoid making a redundant baseline copy around it.
        yield
    else:
        seeded = not (uses_empty_runtime_cache(request.function)
                      or getattr(request.cls, '_pfdsim_empty_runtime_cache', False))
        real_xtb = (getattr(request.function, '_pfdsim_real_xtb_dependencies', False)
                    or getattr(request.cls, '_pfdsim_real_xtb_dependencies', False))
        with isolated_runtime_caches(seeded=seeded, real_xtb=real_xtb):
            yield


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    args = list(config.invocation_params.args)
    # xdist re-enters this hook in each worker with the controller's original
    # (possibly empty) arguments. Only the controller may create workers.
    if (args or hasattr(config, 'workerinput')
            or os.environ.get('PYTEST_XDIST_WORKER')
            or os.environ.get('PFDSIM_PARALLEL_DEFAULT_ACTIVE')):
        return None

    # Continue through pytest so function tests and autouse fixtures are never
    # bypassed. Dynamic scheduling replaces obsolete per-module estimates.
    if not config.pluginmanager.hasplugin('xdist'):
        raise pytest.UsageError('The default suite requires pytest-xdist (install the dev dependencies)')
    config.option.numprocesses = WORKER_COUNT
    config.option.dist = 'load'
    return None


@pytest.hookimpl(wrapper=True)
def pytest_report_to_serializable():
    data = yield
    # Pytest's built-in subtest serializer retains arbitrary Python kwargs,
    # while xdist's transport only accepts primitive values. Normalize just
    # diagnostic context; outcomes, tracebacks, and test objects are untouched.
    if data is not None and '_subtest.context' in data:
        data['_subtest.context'] = serializable_subtest_context(data['_subtest.context'])
    return data
