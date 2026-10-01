#!/usr/bin/env python3
"""Measure/profile initialized adsorption solves and capture SQLite warning origins.

Example (run before and after a change, using distinct artifact directories):
    set -o pipefail
    python scripts/performance/benchmark_adsorption.py --output /tmp/adsorption-before \
        2>&1 | tee /tmp/adsorption-before.log > /dev/null

No randomness; one CPU thread. Initialization and one warm-up are excluded from
timings. Each measured solve uses a fresh Simulator instance. Profiling and
optional tracemalloc warning collection are separate from timing measurements.
"""

import argparse
from collections import Counter
import cProfile
import gc
import hashlib
import json
import os
from pathlib import Path
import pstats
from statistics import median
import sys
import time
import tracemalloc
import warnings

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from simulator import Simulator

CASES = ('air_3a_molecular_sieve_drying', 'dcm_3a_molecular_sieve_drying',
         'ethanol_3a_molecular_sieve_drying', 'competitive_13x_adsorption',
         'propane_propylene_4a_adsorption')


def prepare(case):
    return Simulator.from_file(ROOT / 'examples' / (case + '.pfd')).initialize()


def record(result):
    if not result.converged or result.errors:
        raise AssertionError(result.errors)
    return {
        'mass_balance_error': result.mass_balance_error,
        'energy_balance_error': result.energy_balance_error,
        'streams': {name: {'F': s.F, 'T': s.T, 'x': s.composition, 'H': s.H}
                    for name, s in result.streams.items()},
        'units': {name: {'heat_duty': r.heat_duty, 'performance': r.performance}
                  for name, r in result.units.items()},
    }


def warning_origins(case):
    gc.collect()
    tracemalloc.start(8)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ResourceWarning)
        sim = prepare(case)
        sim.run()
        del sim
        gc.collect()
    counts = Counter()
    for warning in captured:
        if not issubclass(warning.category, ResourceWarning):
            continue
        trace = tracemalloc.get_object_traceback(warning.source)
        origin = str(trace[-1]) if trace else f'{warning.filename}:{warning.lineno}'
        counts[origin] += 1
    tracemalloc.stop()
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--cases', choices=CASES, nargs='+', default=CASES)
    parser.add_argument('--warnings', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'manifest.json').write_text(json.dumps({
        'python':sys.version, 'numpy':np.__version__, 'scipy':scipy.__version__,
        'threads':1, 'repeats':args.repeats, 'cases':list(args.cases),
        'sources':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                   for name in ('adsorption_models.py', 'molecular_sieve.py',
                                'chemical_properties.py', 'property_resolution/runtime_cache.py',
                                'property_resolution/runtime_locks.py')},
    }, indent=2)+'\n')
    summary = {}
    for case in args.cases:
        samples, records = [], []
        for repeat in range(args.repeats+1):
            sim = prepare(case)
            gc.collect()
            start = time.perf_counter()
            result = sim.run()
            elapsed = time.perf_counter()-start
            if repeat:
                samples.append(elapsed)
                records.append(record(result))
            del sim, result
        sim = prepare(case)
        profile = cProfile.Profile()
        result = profile.runcall(sim.run)
        record(result)
        profile.dump_stats(args.output / (case+'.prof'))
        with (args.output / (case+'.profile.txt')).open('w') as handle:
            pstats.Stats(profile, stream=handle).strip_dirs().sort_stats('cumulative').print_stats(40)
        del sim, result
        entry = {'median_seconds': median(samples), 'samples_seconds': samples,
                 'results': records}
        if args.warnings:
            entry['resource_warning_origins'] = warning_origins(case)
        summary[case] = entry
        (args.output / 'results.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(case, 'median_seconds=', entry['median_seconds'],
              'resource_warning_origins=', entry.get('resource_warning_origins', {}), flush=True)


if __name__ == '__main__':
    main()
