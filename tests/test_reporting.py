from enum import Enum
import json
import unittest

from tests.reporting import serializable_subtest_context


class DiagnosticState(Enum):
    READY = 'ready'


class SubtestReportingTests(unittest.TestCase):
    def test_complex_labels_become_portable_without_mutating_assertion_inputs(self):
        original = {'msg': None, 'kwargs': {
            'cls': dict, 'state': DiagnosticState.READY,
            'nested': [{'classes': (int, str)}],
        }}
        result = serializable_subtest_context(original)
        self.assertEqual(result['kwargs']['cls'], 'builtins.dict')
        self.assertEqual(result['kwargs']['state'], 'DiagnosticState.READY')
        self.assertEqual(result['kwargs']['nested'], [{'classes': ('builtins.int', 'builtins.str')}])
        self.assertTrue(json.dumps(result))
        self.assertIs(original['kwargs']['cls'], dict)
        self.assertIs(original['kwargs']['state'], DiagnosticState.READY)
