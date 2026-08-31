import socket
import unittest

from property_resolution.common import OnlineAttemptState
from property_resolution.resolver import PropertyResolver
from tests.live_provider import run_optional_live_provider


class OptionalLiveProviderTests(unittest.TestCase):
    def test_tracker_transient_warns_and_returns_incomplete(self):
        resolver = PropertyResolver()

        def transient():
            resolver._record_online_attempt_state(
                OnlineAttemptState.TRANSIENT_FAILURE
            )
            return 'fallback'

        with self.assertWarns(RuntimeWarning):
            result = run_optional_live_provider(
                resolver,
                transient,
                label='fixture live provider',
            )
        self.assertFalse(result.completed)
        self.assertEqual(result.value, 'fallback')
        self.assertEqual(result.state, OnlineAttemptState.TRANSIENT_FAILURE)

    def test_dns_exception_warns_and_returns_incomplete(self):
        resolver = PropertyResolver()
        with self.assertWarns(RuntimeWarning):
            result = run_optional_live_provider(
                resolver,
                lambda: (_ for _ in ()).throw(
                    socket.gaierror('fixture DNS failure')
                ),
                label='fixture DNS provider',
            )
        self.assertFalse(result.completed)
        self.assertIsNone(result.value)
        self.assertEqual(result.state, OnlineAttemptState.TRANSIENT_FAILURE)

    def test_non_connectivity_failures_are_not_hidden(self):
        resolver = PropertyResolver()
        with self.assertRaisesRegex(AssertionError, 'fixture parser failure'):
            run_optional_live_provider(
                resolver,
                lambda: (_ for _ in ()).throw(
                    AssertionError('fixture parser failure')
                ),
                label='fixture parser',
            )


if __name__ == '__main__':
    unittest.main()
