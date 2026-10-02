import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent


def run_python(
    code: str,
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    module = 'tests.cache_isolation' if cwd == ROOT else 'pfdsim.tests.cache_isolation'
    code = (
        f'from {module} import isolated_runtime_caches\n'
        'with isolated_runtime_caches():\n'
        + textwrap.indent(code, '    ')
    )
    return subprocess.run(
        [sys.executable, '-c', code],
        cwd=cwd,
        env=env,
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
assert unit_operations.UNIT_CLASSES['Crystallizer'].__module__ == (
    'pfdsim.unit_operations_solids'
)
assert unit_operations.UNIT_CLASSES['LayerCrystallizer'].__module__ == (
    'pfdsim.unit_operations_solids'
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
    ('RKSMHV2', ['CO', 'H2', 'CH3OH'], 'vapor'),
    ('UNIFLBY', ['ethanol', 'water'], 'liquid'),
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
assert unit_operations.UNIT_CLASSES['Crystallizer'].__module__ == (
    'unit_operations_solids'
)
assert unit_operations.UNIT_CLASSES['LayerCrystallizer'].__module__ == (
    'unit_operations_solids'
)
assert not ({
    'pfdsim.simulator', 'pfdsim.unit_operations',
    'pfdsim.batch_models', 'pfdsim.pellet_models', 'pfdsim.thermodynamics',
} & sys.modules.keys())
""",
            ROOT,
        )
        self.assert_success(completed)

    @unittest.skipUnless(
        importlib.util.find_spec('numba') is not None,
        'requires the optional Numba dependency',
    )
    def test_numba_caches_are_isolated_between_import_modes(self):
        probe = """
import importlib
import numpy as np

prefix = {prefix!r}
unifac = importlib.import_module(prefix + 'compiled_unifac')
activity = importlib.import_module(prefix + 'compiled_activity')
lle = importlib.import_module(prefix + 'compiled_lle')
vlle = importlib.import_module(prefix + 'compiled_vlle')

nu = np.ones((1, 1), dtype=np.float64)
vector = np.ones(1, dtype=np.float64)
matrix = np.zeros((1, 1), dtype=np.float64)
antoine_cd = np.zeros((1, 2), dtype=np.float64)

gamma = unifac._activity_coefficients_numba(
    nu, vector, vector, vector, matrix, matrix, matrix,
    0, vector, 298.15,
)
assert np.all(np.isfinite(gamma))

split = lle._lle_split_numba(
    nu, vector, vector, vector, matrix, matrix, matrix,
    0, vector, 298.15, 2, 1.0e-6,
)
assert len(split) == 5

k_values = vlle._k_values_unifac(
    vector, 298.15, 1.0, nu, vector, vector, vector,
    matrix, matrix, matrix, 0, vector, vector, antoine_cd,
)
assert np.all(np.isfinite(k_values))

integer_parameters = np.zeros((1, 1), dtype=np.int64)
tmin = np.full((1, 1), -np.inf)
tmax = np.full((1, 1), np.inf)
nrtl_parameters = (integer_parameters,) + (matrix,) * 8 + (tmin, tmax)
uniquac_parameters = (
    vector, vector, vector, integer_parameters,
) + (matrix,) * 6 + (tmin, tmax)

for model, parameters in (
    ('nrtl', nrtl_parameters), ('uniquac', uniquac_parameters),
):
    gamma_kernel = getattr(activity, '_' + model + '_activity_coefficients_numba')
    enthalpy_kernel = getattr(activity, '_' + model + '_excess_enthalpy_numba')
    split_kernel = getattr(lle, '_lle_split_' + model + '_numba')
    assert np.allclose(gamma_kernel(vector, 298.15, *parameters), 1.0)
    assert np.isclose(enthalpy_kernel(vector, 298.15, *parameters), 0.0)
    split = split_kernel(vector, 298.15, *parameters, 2, 1.0e-6)
    assert len(split) == 5

for model_id, parameters in (
    (1, (matrix,) * 8 + (tmin, tmax)),
    (2, (nu,) * 3 + (matrix,) * 7),
):
    gamma = vlle._activity_gamma(
        model_id, vector, 298.15, integer_parameters, *parameters,
    )
    assert np.allclose(gamma, 1.0)

for module, dispatcher in (
    (unifac, unifac._activity_coefficients_numba),
    (lle, lle._lle_split_numba),
    (vlle, vlle._k_values_unifac),
    (vlle, vlle._activity_gamma),
):
    assert dispatcher.py_func.__qualname__.startswith(module.__name__ + '.')

assert unifac._activity_coefficients_numba.py_func.__qualname__ == (
    unifac.__name__ + '._activity_coefficients_numba'
)
assert lle._normalize.py_func.__qualname__ == lle.__name__ + '._normalize'

def dependency_tag(module, dispatcher):
    relative_name = dispatcher.py_func.__qualname__.removeprefix(
        module.__name__ + '.'
    )
    tag, _function_name = relative_name.split('.', 1)
    assert len(tag) == 12
    assert all(character in '0123456789abcdef' for character in tag)
    return tag

assert dependency_tag(lle, lle._lle_split_numba)
assert dependency_tag(vlle, vlle._k_values_unifac) != dependency_tag(
    vlle, vlle._activity_gamma
)
"""
        modes = (
            ('', ROOT),
            ('pfdsim.', PARENT),
        )
        for order in (modes, tuple(reversed(modes))):
            with self.subTest(order=tuple(prefix or 'top-level' for prefix, _ in order)):
                with tempfile.TemporaryDirectory(prefix='pfdsim-numba-cache-') as cache_dir:
                    env = dict(os.environ)
                    env['NUMBA_CACHE_DIR'] = cache_dir
                    # Repeat both modes to exercise restoration of their own caches.
                    for prefix, cwd in order + order:
                        completed = run_python(
                            probe.format(prefix=prefix),
                            cwd,
                            env=env,
                        )
                        self.assert_success(completed)

    @unittest.skipUnless(
        importlib.util.find_spec('numba') is not None,
        'requires the optional Numba dependency',
    )
    def test_compiled_lle_survives_unavailable_unifac_backend(self):
        completed = run_python(
            """
import builtins
import numpy as np

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'compiled_unifac':
        raise ImportError('simulated optional backend failure')
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import compiled_lle
builtins.__import__ = original_import

assert compiled_lle._activity_coefficients_numba is None
normalized = compiled_lle._normalize(np.asarray([1.0, 3.0]))
assert np.allclose(normalized, [0.25, 0.75])
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
