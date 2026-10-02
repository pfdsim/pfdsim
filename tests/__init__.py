import os
import unittest

from .cache_isolation import install_unittest_cache_isolation

install_unittest_cache_isolation()


class ParallelDefaultDiscovery(unittest.TestCase):
    def test_default_suite(self):
        from tests.default_parallel import run_parallel_default

        self.assertEqual(run_parallel_default(), 0)


def load_tests(loader, tests, pattern):
    if os.environ.get('PFDSIM_PARALLEL_DEFAULT_ACTIVE'):
        return tests
    suite = unittest.TestSuite()
    suite.addTest(ParallelDefaultDiscovery('test_default_suite'))
    return suite
