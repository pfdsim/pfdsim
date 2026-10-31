#!/usr/bin/env python3
"""Separate first/repeated utility solves from initialization and JIT loading.

Run in a fresh process with a new output directory and persist both streams::

    .venv/bin/python scripts/performance/profile_utility_first_use.py \
        --output /tmp/utility-first-use 2>&1 | tee /tmp/utility-first-use.log

No random inputs or production modifications. Instrumentation affects absolute
timings, so use the phase/function breakdown for attribution, not baselines.
"""

import argparse
from contextlib import contextmanager
import cProfile
import functools
import json
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulator import Simulator
from unit_operations_basic import Heater, Cooler


@contextmanager
def unit_timings(rows):
    originals = []
    depth = 0
    def instrument(original):
        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            nonlocal depth
            outer = depth == 0
            depth += 1
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                return original(self, *args, **kwargs)
            finally:
                depth -= 1
                if outer:
                    rows.append({'unit': self.unit_id, 'wall_seconds': time.perf_counter()-wall,
                                 'cpu_seconds': time.process_time()-cpu})
        return wrapped
    for cls in (Heater, Cooler):
        originals.append((cls, cls.solve))
        cls.solve = instrument(cls.solve)
    try:
        yield
    finally:
        for cls, original in originals:
            cls.solve = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for repeat in range(2):
        wall, cpu = time.perf_counter(), time.process_time()
        simulator = Simulator.from_file(ROOT/'examples/utility_surface_sizing.pfd')
        simulator.initialize()
        initialization = {'wall_seconds': time.perf_counter()-wall, 'cpu_seconds': time.process_time()-cpu}
        units = []
        profiler = cProfile.Profile()
        with unit_timings(units):
            wall, cpu = time.perf_counter(), time.process_time()
            profiler.enable()
            result = simulator.run()
            profiler.disable()
            solve = {'wall_seconds': time.perf_counter()-wall, 'cpu_seconds': time.process_time()-cpu}
        profiler.dump_stats(str(args.output/f'run-{repeat}.pstats'))
        stats = pstats.Stats(profiler)
        with (args.output/f'run-{repeat}-profile.txt').open('x') as stream:
            pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(80)
        costs = [{'file': filename, 'line': line, 'function': name, 'calls': values[1],
                  'self_seconds': values[2], 'inclusive_seconds': values[3]}
                 for (filename, line, name), values in sorted(stats.stats.items(),
                                                            key=lambda item: item[1][3], reverse=True)]
        row = {'repeat': repeat, 'initialization': initialization, 'solve': solve, 'units': units,
               'converged': result.converged, 'errors': result.errors, 'costs': costs}
        with (args.output/f'run-{repeat}.json').open('x') as stream:
            json.dump(row, stream, indent=2)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'costs'}), flush=True)
    with (args.output/'summary.json').open('x') as stream:
        json.dump(rows, stream, indent=2)
    return int(any(not r['converged'] or r['errors'] for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
