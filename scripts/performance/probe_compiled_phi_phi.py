#!/usr/bin/env python3
"""Compare the shared interpreted/compiled EOS algorithm and pure-root fusion.

Use new output directories and persist both output streams. --calibrate uses
only the performance suite's six-sample C reference, never its test workload.
Compilation is prepared outside ready-solve timing. No random inputs are
created; optional state matrices explicitly seed their samples.
"""

import argparse
import ast
from contextlib import contextmanager
import json
import inspect
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from numba import njit, typeof
import phi_phi
from phi_phi_numerics import phase_admissible, boundary_equations
from simulator import Simulator
from scripts.performance.diagnose_eos_flash_regression import run
from scripts.performance.probe_eos_flash_acceleration import check_matrix

ORIGINAL_FLASH = phi_phi.solve_phi_phi_flash
ORIGINAL_PROVIDER = phi_phi._native_provider
ORIGINAL_BOUNDARY = phi_phi.solve_phi_phi_boundary


@njit(cache=True)
def boundary_samples(state, data, values, z, active, fixed, bubble, variable_T, low, high):
    f0 = boundary_equations(state, data, values, z, active, fixed, bubble, variable_T)
    samples = np.zeros((len(values), len(values)))
    steps = np.zeros(len(values))
    for i in range(len(values)):
        h = 2.**-26*max(1., abs(values[i]))*(1. if values[i] >= 0 else -1.)
        if values[i]+h > high[i] or values[i]+h < low[i]:
            h = -h
        perturbed = values.copy()
        perturbed[i] += h
        steps[i] = perturbed[i]-values[i]
        if i == 0 and variable_T:
            continue
        samples[:, i] = boundary_equations(state, data, perturbed, z, active, fixed, bubble, variable_T)
    return f0, samples, steps


def boundary_jacobian(thermo, values, z, active, fixed, kind, variable, bounds, equations):
    native = ORIGINAL_PROVIDER(thermo, math.exp(values[0]) if variable == 'T' else fixed)
    if native is None:
        from scipy.optimize._numdiff import approx_derivative
        return approx_derivative(equations, values, method='2-point', bounds=bounds)
    f0, samples, steps = boundary_samples(*native, values, np.array(list(z.values())),
        np.array([thermo.components.index(c) for c in active]), float(fixed), kind == 'bubble',
        variable == 'T', np.array(bounds[0]), np.array(bounds[1]))
    if variable == 'T':
        perturbed = values.copy()
        perturbed[0] += steps[0]
        samples[:, 0] = equations(perturbed)
    return (samples-f0[:, None])/steps[None, :]


def jacobian_boundary():
    tree = ast.parse(inspect.getsource(ORIGINAL_BOUNDARY))
    class Transform(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'least_squares':
                jac = ast.parse('lambda values: _jac(thermo, values, z, active, fixed, kind, variable, bounds, equations)', mode='eval').body
                existing = next((k for k in node.keywords if k.arg == 'jac'), None)
                if existing is None:
                    node.keywords.append(ast.keyword(arg='jac', value=jac))
                else:
                    existing.value = jac
            return node
    tree = Transform().visit(tree)
    namespace = dict(vars(phi_phi), _jac=boundary_jacobian)
    exec(compile(ast.fix_missing_locations(tree), '<probe-boundary-jacobian>', 'exec'), namespace)
    return namespace['solve_phi_phi_boundary']


@njit(cache=True)
def pure_root(state, data, T, P, z, component):
    kl, pl = state(T, P, z, 0, data)
    kv, pv = state(T, P, z, 1, data)
    liquid = phase_admissible(kl, 0) and math.isfinite(pl[component])
    vapor = phase_admissible(kv, 1) and math.isfinite(pv[component])
    if liquid and vapor:
        if kl == 0 and kv == 0:
            return 1
        return 0 if pl[component] < pv[component] else 1
    if liquid:
        return 0
    return 1 if vapor else -1


@contextmanager
def scope(mode):
    previous = (phi_phi._native_provider, phi_phi.solve_phi_phi_flash, Simulator.initialize,
                phi_phi.normalized_eos_composition, phi_phi.solve_phi_phi_boundary)
    def scalar_validation(components, composition):
        if set(composition)-set(components):
            raise phi_phi.PhiPhiFlashError('EOS composition contains unknown components')
        try:
            fractions = [float(composition.get(c, 0.)) for c in components]
        except (TypeError, ValueError, OverflowError) as error:
            raise phi_phi.PhiPhiFlashError('EOS composition must be nonnegative and finite with a positive total') from error
        maximum = max(fractions, default=0.)
        if maximum <= 0 or any(not math.isfinite(v) or v < 0 for v in fractions):
            raise phi_phi.PhiPhiFlashError('EOS composition must be nonnegative and finite with a positive total')
        fractions = [v/maximum for v in fractions]
        total = math.fsum(fractions)
        return {c: v/total for c, v in zip(components, fractions)}
    def pure_flash(thermo, composition, T, P, max_iter, tol):
        z = phi_phi.normalized_eos_composition(thermo.components, composition)
        active = [c for c in thermo.components if z[c] > 0]
        if len(active) == 1 and max_iter > 1 and T > 0 and P > 0 and math.isfinite(T) and math.isfinite(P):
            native = ORIGINAL_PROVIDER(thermo, T)
            if native is not None:
                phase = pure_root(*native, float(T), float(P), np.array(list(z.values())),
                                  thermo.components.index(active[0]))
                if phase >= 0:
                    return float(phase), dict(z), dict(z)
        return ORIGINAL_FLASH(thermo, z, T, P, max_iter, tol)
    def initialize(self, *args, **kwargs):
        result = previous[2](self, *args, **kwargs)
        for thermo in getattr(self, 'thermo_packages', {'global': self.thermo}).values():
            native = ORIGINAL_PROVIDER(thermo, 300., record_warnings=False)
            if native is not None:
                values = (*native, 300., 1., np.zeros(len(thermo.components)), 0)
                pure_root.compile(tuple(typeof(v) for v in values))
                x = np.zeros(len(thermo.components))
                jac_values = (*native, x, x, np.arange(len(x)), 1., True, True, x, x)
                boundary_samples.compile(tuple(typeof(v) for v in jac_values))
        return result
    if mode == 'python':
        phi_phi._native_provider = lambda *args, **kwargs: None
    elif mode == 'compiled-pure':
        phi_phi.solve_phi_phi_flash = pure_flash
        Simulator.initialize = initialize
    elif mode == 'scalar-validation':
        phi_phi.normalized_eos_composition = scalar_validation
    elif mode == 'compiled-jacobian':
        phi_phi.solve_phi_phi_boundary = jacobian_boundary()
        Simulator.initialize = initialize
    try:
        yield
    finally:
        (phi_phi._native_provider, phi_phi.solve_phi_phi_flash, Simulator.initialize,
         phi_phi.normalized_eos_composition, phi_phi.solve_phi_phi_boundary) = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples', nargs='*', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--modes', nargs='+', choices=('python', 'compiled', 'compiled-pure', 'scalar-validation', 'compiled-jacobian'), default=('python', 'compiled'))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--matrix-points', type=int, default=0)
    parser.add_argument('--tests', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    calibration = None
    if args.calibrate:
        from tests.performance_all_examples import _measure_cpu_reference
        start_scale, samples = _measure_cpu_reference()
        calibration = {'start_scale': start_scale, 'start_samples': samples}
        print('CPU_REFERENCE_START', start_scale, flush=True)
    failures, reports = 0, []
    for mode in args.modes:
        output = args.output / mode
        output.mkdir()
        print('MODE', mode, flush=True)
        with scope(mode):
            if args.matrix_points:
                def reference(thermo, *values):
                    provider = phi_phi._native_provider
                    phi_phi._native_provider = lambda *args, **kwargs: None
                    try:
                        result = ORIGINAL_FLASH(thermo, *values)
                    finally:
                        phi_phi._native_provider = provider
                        getattr(thermo, '_phi_phi_flash_cache', {}).clear()
                    return result
                failures += check_matrix(args.matrix_points, output, reference)
            if args.tests:
                import pytest
                status = pytest.main(['tests/test_phi_phi.py', 'tests/test_cubic_phase_equilibrium.py',
                                     'tests/test_lkp.py', 'tests/test_mhv2.py'])
                if status:
                    return int(status)
            for path in args.examples:
                for repeat in range(args.repeats):
                    report = run(path, 'current', repeat, {}, output, False)
                    report['mode'] = mode
                    reports.append(report)
                    failures += int(not report['converged'] or bool(report['errors']))
    if calibration is not None:
        end_scale, samples = _measure_cpu_reference()
        calibration.update(end_scale=end_scale, end_samples=samples)
        calibration['mean_scale'] = (calibration['start_scale']+end_scale)/2
        print('CPU_REFERENCE_END', end_scale, flush=True)
    summary = []
    for mode in args.modes:
        for path in args.examples:
            rows = [r for r in reports if r['mode'] == mode and r['example'] == path.name]
            if not rows:
                continue
            record = {'mode': mode, 'example': path.name,
                      'median_cpu': statistics.median(r['cpu_seconds'] for r in rows),
                      'median_wall': statistics.median(r['wall_seconds'] for r in rows)}
            if calibration is not None:
                record['normalized_wall'] = record['median_wall']/calibration['mean_scale']
            summary.append(record)
    with (args.output/'summary.json').open('x') as stream:
        json.dump({'calibration': calibration, 'timings': summary}, stream, indent=2)
    print('SUMMARY', json.dumps(summary), flush=True)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
