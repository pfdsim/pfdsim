#!/usr/bin/env python3
"""Attribute ready-solve regressions to the shared EOS TP flash.

Example (output directory must be new)::

    .venv/bin/python scripts/performance/diagnose_eos_flash_regression.py \
        examples/methanol_synthesis.pfd --output /tmp/eos-flash-study \
        2>&1 | tee /tmp/eos-flash-study.log

The head-flash diagnostic restores only the reference revision's
IdealThermodynamics._iterative_K_flash_TP in memory. All other working-tree
code stays active. This is an attribution experiment, not an alternative
production solver: the historical solver may produce different physical
results. Reports include balances and stream states to expose that difference.
The head-boundaries mode instead restores the four reference bubble/dew
temperature/pressure methods; head-flash-boundaries restores both groups.
Use --profile for separate profiled runs; profiler times are not comparable
to the uninstrumented performance-suite timings. No random inputs are used.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import cProfile
import json
from pathlib import Path
import pstats
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulator import Simulator
from thermodynamics_models import base, common, eos, phase_diagrams
import phi_phi


def reference_function(revision, module, name, owner=None):
    path = module.__name__.replace('.', '/') + '.py'
    source = subprocess.run(
        ['git', 'show', f'{revision}:{path}'],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    tree = ast.parse(source)
    container = (next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == owner)
                 if owner else tree)
    method = next(node for node in container.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[method], type_ignores=[]),
                 f'git:{revision}:{path}', 'exec'), namespace)
    return namespace[method.name]


@contextmanager
def flash_variant(mode, original):
    patches = []
    def replace(owner, name, function):
        patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, function)
    if mode.startswith('head-flash'):
        replace(base.IdealThermodynamics, '_iterative_K_flash_TP', original['flash'])
    if mode.endswith('boundaries'):
        for module in (base, common, eos):
            for name in ('_solve_bubble_point_temperature', '_solve_dew_point_temperature'):
                replace(module, name, original[name])
        for name in ('bubble_point_P', 'dew_point_P'):
            replace(phase_diagrams.PhaseDiagramMixin, name, original[name])
    try:
        yield
    finally:
        for owner, name, previous in reversed(patches):
            setattr(owner, name, previous)


def run(path, mode, repeat, original, output, profile):
    stem = f'{path.stem}-{mode}-{repeat}'
    with flash_variant(mode, original):
        simulator = Simulator.from_file(path)
        simulator.initialize()
        profiler = cProfile.Profile() if profile else None
        flash_states = Counter()
        shared_flash = phi_phi.solve_phi_phi_flash
        def counted_flash(thermo, composition, T, P, max_iter, tol):
            flash_states[(id(thermo), T, P, tuple(composition.items()), max_iter, tol)] += 1
            return shared_flash(thermo, composition, T, P, max_iter, tol)
        if profiler:
            phi_phi.solve_phi_phi_flash = counted_flash
            profiler.enable()
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        try:
            result = simulator.run()
        finally:
            if profiler:
                profiler.disable()
                phi_phi.solve_phi_phi_flash = shared_flash
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
        if profiler:
            profiler.dump_stats(str(output / f'{stem}.pstats'))
            with (output / f'{stem}-profile.txt').open('x') as stream:
                pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(60)
        report = {
            'example': path.name, 'mode': mode, 'repeat': repeat,
            'profiled': profile, 'wall_seconds': wall_seconds, 'cpu_seconds': cpu_seconds,
            'converged': result.converged, 'errors': result.errors,
            'mass_balance_error': result.mass_balance_error,
            'energy_balance_error': result.energy_balance_error,
            'compiled_backend': type(getattr(
                getattr(simulator.thermo, 'cubic', getattr(simulator.thermo, 'psrk', None)),
                '_compiled_backend', None)).__name__,
            'thermo_warning_count': len(simulator.thermo.warnings),
            'thermo_warnings': list(simulator.thermo.warnings),
            'streams': {
                name: {key: getattr(state, key, None)
                       for key in ('T', 'P', 'F', 'H', 'vapor_fraction', 'composition', 'phase_status')}
                for name, state in result.streams.items()
            },
        }
        if profiler:
            report['flash_state_counts'] = {
                'calls': sum(flash_states.values()), 'unique_states': len(flash_states),
                'repeat_calls': sum(flash_states.values()) - len(flash_states),
                'most_repeated': flash_states.most_common(10),
            }
            report['profile_costs'] = [
                {'file': filename, 'line': line, 'function': name,
                 'calls': values[1], 'self_seconds': values[2], 'inclusive_seconds': values[3]}
                for (filename, line, name), values in sorted(
                    pstats.Stats(profiler).stats.items(), key=lambda item: item[1][3], reverse=True)
                if 'phi_phi.py' in filename or name in (
                    '_iterative_K_flash_TP', 'fugacity_coefficients', '_eos_root_kind')
            ]
        with (output / f'{stem}.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({k: v for k, v in report.items()
                          if k not in ('streams', 'profile_costs', 'flash_state_counts', 'thermo_warnings')},
                         allow_nan=False), flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', default='HEAD')
    parser.add_argument('--modes', nargs='+',
                        choices=('current', 'head-flash', 'head-boundaries', 'head-flash-boundaries'),
                        default=('current', 'head-flash'))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    original = {'flash': reference_function(args.reference, base,
                '_iterative_K_flash_TP', 'IdealThermodynamics')}
    if any(mode.endswith('boundaries') for mode in args.modes):
        for name in ('_solve_bubble_point_temperature', '_solve_dew_point_temperature'):
            original[name] = reference_function(args.reference, common, name)
        for name in ('bubble_point_P', 'dew_point_P'):
            original[name] = reference_function(args.reference, phase_diagrams, name, 'PhaseDiagramMixin')
    failures = 0
    for path in args.examples:
        for repeat in range(args.repeats):
            for mode in args.modes:
                report = run(path, mode, repeat, original, args.output, args.profile)
                failures += int(not report['converged'] or bool(report['errors']))
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
