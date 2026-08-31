import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent


def run_python(code: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, '-c', code],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


class PackageImportTests(unittest.TestCase):
    def assert_success(self, completed: subprocess.CompletedProcess) -> None:
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    @unittest.skipUnless(
        importlib.util.find_spec('flask') is not None,
        'requires the optional pfdsim[web] dependency group',
    )
    def test_package_imports_use_only_package_module_identities(self):
        completed = run_python(
            """
import sys
import pfdsim.simulator as simulator
import pfdsim.unit_operations as unit_operations
import pfdsim.batch_models as batch_models
import pfdsim.pellet_models as pellet_models
import pfdsim.wsgi as wsgi
from pfdsim.thermodynamics import create_thermodynamics

assert simulator.Simulator.__module__ == 'pfdsim.simulator'
assert batch_models.solve_adaptive_batch.__module__ == 'pfdsim.batch_models'
assert pellet_models.first_order_spherical_effectiveness.__module__ == (
    'pfdsim.pellet_models'
)
assert unit_operations.UNIT_CLASSES['BatchReactor'].__module__ == (
    'pfdsim.unit_operations_reactors'
)
assert unit_operations.UNIT_CLASSES['PackedBedReactor'].__module__ == (
    'pfdsim.unit_operations_reactors'
)
assert wsgi.app.import_name == 'pfdsim.app'
assert not ({
    'simulator', 'unit_operations', 'batch_models', 'pellet_models',
    'thermodynamics',
} & sys.modules.keys())

for method, components, phase in (
    ('IDEAL', ['N2', 'O2'], 'vapor'),
    ('PR', ['CO', 'H2', 'CH3OH'], 'vapor'),
    ('NRTL', ['water', 'butanol'], 'liquid'),
    ('PSRK', ['CO', 'H2', 'CH3OH'], 'vapor'),
):
    thermo = create_thermodynamics(components, method)
    composition = {component: 1.0 / len(components) for component in components}
    state = thermo.calculate_state(
        350.0 if method != 'PSRK' else 500.0,
        1.0 if method in {'IDEAL', 'NRTL'} else 20.0,
        1.0,
        composition,
        phase=phase,
        flash=False,
    )
    assert state.H is not None
""",
            PARENT,
        )
        self.assert_success(completed)

    def test_repository_imports_use_only_top_level_module_identities(self):
        completed = run_python(
            """
import sys
import simulator
import unit_operations
import batch_models
import pellet_models
import thermodynamics

assert simulator.Simulator.__module__ == 'simulator'
assert batch_models.solve_adaptive_batch.__module__ == 'batch_models'
assert pellet_models.first_order_spherical_effectiveness.__module__ == (
    'pellet_models'
)
assert unit_operations.UNIT_CLASSES['BatchReactor'].__module__ == (
    'unit_operations_reactors'
)
assert unit_operations.UNIT_CLASSES['PackedBedReactor'].__module__ == (
    'unit_operations_reactors'
)
assert not ({
    'pfdsim.simulator', 'pfdsim.unit_operations',
    'pfdsim.batch_models', 'pfdsim.pellet_models', 'pfdsim.thermodynamics',
} & sys.modules.keys())
""",
            ROOT,
        )
        self.assert_success(completed)

    def test_package_cli_modules_are_executable(self):
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        expected = f"pfdsim {project['project']['version']}"
        for module in ('pfdsim', 'pfdsim.cli'):
            with self.subTest(module=module):
                completed = subprocess.run(
                    [sys.executable, '-m', module, '--version'],
                    cwd=PARENT,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=60,
                )
                self.assert_success(completed)
                self.assertEqual(completed.stdout.strip(), expected)


if __name__ == '__main__':
    unittest.main()
