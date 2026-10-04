import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def configuration(*args, worker=False):
    config = SimpleNamespace(
        invocation_params=SimpleNamespace(args=args),
        option=SimpleNamespace(numprocesses=None, dist='no'),
        pluginmanager=SimpleNamespace(hasplugin=lambda name: name == 'xdist'),
    )
    if worker:
        config.workerinput = {'workerid': 'gw0'}
    return config


class DefaultPytestWorkerTests(unittest.TestCase):
    def setUp(self):
        self.hook = importlib.import_module('pfdsim.conftest').pytest_cmdline_main
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop('PYTEST_XDIST_WORKER', None)
        os.environ.pop('PFDSIM_PARALLEL_DEFAULT_ACTIVE', None)

    def test_bare_controller_configures_six_workers(self):
        config = configuration()
        self.assertIsNone(self.hook(config))
        self.assertEqual(config.option.numprocesses, 6)
        self.assertEqual(config.option.dist, 'load')

    def test_bare_worker_never_spawns_more_workers(self):
        config = configuration(worker=True)
        self.assertIsNone(self.hook(config))
        self.assertIsNone(config.option.numprocesses)
        self.assertEqual(config.option.dist, 'no')

    def test_worker_environment_prevents_recursive_spawning(self):
        os.environ['PYTEST_XDIST_WORKER'] = 'gw0'
        config = configuration()
        self.hook(config)
        self.assertIsNone(config.option.numprocesses)

    def test_explicit_worker_count_and_focused_arguments_are_preserved(self):
        config = configuration('-n', '2', 'tests/test_cache_isolation.py')
        config.option.numprocesses = 2
        self.hook(config)
        self.assertEqual(config.option.numprocesses, 2)
        self.assertEqual(config.option.dist, 'no')

    def test_bare_pytest_starts_exactly_six_workers_and_completes(self):
        # A tiny independent suite exercises the actual xdist startup path
        # without recursively running pfdsim's full suite or its heavy fixtures.
        with tempfile.TemporaryDirectory(prefix='pfdsim-worker-startup-') as directory:
            root = Path(directory)
            (root/'conftest.py').write_text(
                'from pfdsim.conftest import pytest_cmdline_main, pytest_report_to_serializable\n')
            (root/'test_workers.py').write_text(
                'import os\nfrom pathlib import Path\n\n'
                + '\n'.join(
                    f'def test_worker_{index}():\n'
                    '    assert os.environ["PYTEST_XDIST_WORKER_COUNT"] == "6"\n'
                    '    with Path("workers.txt").open("a") as output:\n'
                    '        output.write(os.environ["PYTEST_XDIST_WORKER"] + "\\n")\n'
                    for index in range(12)
                )
                + '\nfrom enum import Enum\nimport unittest\n'
                'class State(Enum):\n    READY = "ready"\n'
                'class SubtestLabels(unittest.TestCase):\n'
                '    def test_complex_labels(self):\n'
                '        with self.subTest(cls=dict, state=State.READY):\n'
                '            self.assertEqual(State.READY.value, "ready")\n'
            )
            env = dict(os.environ)
            for name in ('PYTEST_XDIST_WORKER', 'PYTEST_XDIST_WORKER_COUNT',
                         'PYTEST_XDIST_TESTRUNUID', 'PYTEST_ADDOPTS'):
                env.pop(name, None)
            parent = str(Path(__file__).resolve().parents[2])
            env['PYTHONPATH'] = os.pathsep.join(filter(None, (parent, env.get('PYTHONPATH'))))
            completed = subprocess.run(
                [sys.executable, '-m', 'pytest'], cwd=root, env=env,
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            self.assertIn('13 passed', completed.stdout)
            self.assertEqual(set((root/'workers.txt').read_text().splitlines()),
                             {f'gw{index}' for index in range(6)})
