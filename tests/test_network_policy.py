import unittest
import importlib
import urllib.error
import urllib.request
from unittest.mock import patch

from tests.network_policy import allow_live_providers
from tests.live_provider import run_optional_live_provider
from property_resolver import PropertyResolver


class ProviderNetworkPolicyTests(unittest.TestCase):
    def test_import_modes_share_the_same_transport_policy(self):
        self.assertIs(importlib.import_module('tests.network_policy'),
                      importlib.import_module('pfdsim.tests.network_policy'))

    def test_unmocked_provider_access_is_blocked_before_transport(self):
        with patch('tests.network_policy._urlopen') as transport:
            with self.assertRaisesRegex(urllib.error.URLError, 'disabled in ordinary tests'):
                urllib.request.urlopen('https://pubchem.ncbi.nlm.nih.gov/rest/pug')
            transport.assert_not_called()

    def test_mocked_provider_responses_remain_usable(self):
        with patch('urllib.request.urlopen', return_value='fixture response') as transport:
            self.assertEqual(urllib.request.urlopen('https://pubchem.ncbi.nlm.nih.gov/'),
                             'fixture response')
            transport.assert_called_once()

    def test_live_scope_authorizes_only_the_named_provider_and_restores_policy(self):
        with patch('tests.network_policy._urlopen', return_value='live result') as transport:
            with allow_live_providers('pubchem'):
                self.assertEqual(urllib.request.urlopen(
                    urllib.request.Request('https://pubchem.ncbi.nlm.nih.gov/')),
                    'live result')
                with self.assertRaises(urllib.error.URLError):
                    urllib.request.urlopen('https://webbook.nist.gov/')
            with self.assertRaises(urllib.error.URLError):
                urllib.request.urlopen('https://pubchem.ncbi.nlm.nih.gov/')
            transport.assert_called_once()

    def test_optional_live_helper_enables_provider_transport(self):
        with patch('tests.network_policy._urlopen', return_value='response') as transport:
            result = run_optional_live_provider(
                PropertyResolver(),
                lambda: urllib.request.urlopen('https://pubchem.ncbi.nlm.nih.gov/'),
                label='fixture live operation',
            )
            self.assertTrue(result.completed)
            self.assertEqual(result.value, 'response')
            transport.assert_called_once()
