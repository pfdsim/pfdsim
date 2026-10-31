#!/usr/bin/env python3
"""Probe disk-cache reuse with fixed callbacks and numeric-only entry points.

    .venv/bin/python scripts/performance/probe_stable_numba_cache.py \
        --output /tmp/stable-numba-cache 2>&1 | tee /tmp/stable-numba-cache.log

The shared higher-order algorithm is compiled in memory. Tiny cached model
entry points bind its callbacks at module scope. Tests use fresh subprocesses
with zero deserialized-dispatcher retention and an isolated cache directory.
No production code/cache edits, random inputs or numerical special cases.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from numba import njit


@njit(cache=False)
def shared_algorithm(callback, value):
    return callback(value)*2


@njit(cache=True)
def callback_a(value):
    return value+1


@njit(cache=True)
def callback_b(value):
    return value-1


@njit(cache=True)
def entry_a(value):
    return shared_algorithm(callback_a, value)


@njit(cache=True)
def entry_b(value):
    return shared_algorithm(callback_b, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', choices=('seed', 'reload', 'grow', 'reload-grown'))
    args = parser.parse_args()
    if args.worker:
        start = time.perf_counter()
        values = (2.,) if args.worker in ('seed', 'reload') else (2., 2)
        for value in values:
            assert entry_a(value) == 6
            assert entry_b(value) == 2
        result = {'phase': args.worker, 'seconds': time.perf_counter()-start,
                  'entry_cache_hits': sum(sum(f._cache_hits.values()) for f in (entry_a, entry_b)),
                  'entry_cache_misses': sum(sum(f._cache_misses.values()) for f in (entry_a, entry_b))}
        print(json.dumps(result), flush=True)
        return 0
    if args.output is None:
        parser.error('--output is required')
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, 'NUMBA_CACHE_DIR': str(args.output/'cache'),
                   'NUMBA_FUNCTION_CACHE_SIZE': '0'}
    reports = []
    for phase in ('seed', 'reload', 'grow', 'reload-grown'):
        with (args.output/f'{phase}.log').open('x') as stream:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', phase],
                                       env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f'{phase} failed; inspect its persisted log')
        report = json.loads((args.output/f'{phase}.log').read_text().splitlines()[-1])
        reports.append(report)
        print(json.dumps(report), flush=True)
    assert reports[1]['entry_cache_hits'] == 2 and reports[1]['entry_cache_misses'] == 0
    assert reports[2]['entry_cache_hits'] == 2 and reports[2]['entry_cache_misses'] == 2
    assert reports[3]['entry_cache_hits'] == 4 and reports[3]['entry_cache_misses'] == 0
    with (args.output/'summary.json').open('x') as stream:
        json.dump(reports, stream, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
