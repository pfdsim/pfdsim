#!/usr/bin/env python3
"""Explain canonical Psat cache misses during two fresh utility simulations.

Run in a fresh process with a new output directory; persist both streams::

    .venv/bin/python scripts/performance/diagnose_utility_psat_cache.py \
        --output /tmp/utility-psat-cache 2>&1 | tee /tmp/utility-psat-cache.log

Only diagnostic wrappers are installed. Cache admission, keys and evaluations
are unchanged. Reports expose changed field names and cache decisions rather
than dumping property snapshots or serialized curve payloads. No randomness.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from property_resolution.resolver import PropertyResolver
from simulator import Simulator


@contextmanager
def trace_cache(rows):
    names = ('_canonical_vapor_pressure_runtime', '_build_canonical_vapor_pressure_runtime',
             '_load_persistent_canonical_vapor_pressure_runtime',
             '_store_persistent_canonical_vapor_pressure_runtime', '_online_attempt_is_persistable')
    originals = {name: getattr(PropertyResolver, name) for name in names}
    stack, previous = [], {}
    def runtime(self, symbol, props, *, allow_online, minimum_pressure_bar):
        key = self._canonical_vapor_pressure_cache_key(symbol, props, allow_online, minimum_pressure_bar)
        fingerprint = self._canonical_persistent_input_fingerprint(props, key)
        label = symbol, allow_online, minimum_pressure_bar
        old = previous.get(label)
        row = {'symbol': symbol, 'allow_online': allow_online,
               'memory_hit_before': key in getattr(self, '_canonical_vapor_pressure_curves', {}),
               'same_memory_key_as_previous': old is not None and old[0] == key,
               'same_persistent_fingerprint_as_previous': old is not None and old[1] == fingerprint,
               'changed_input_fields': ([] if old is None else
                   [name for name in props.keys() | old[2].keys() if props.get(name) != old[2].get(name)]),
               'builds': 0, 'persistent_hits': 0, 'stores': 0, 'admission': []}
        previous[label] = key, fingerprint, deepcopy(props)
        start = time.perf_counter()
        stack.append(row)
        rows.append(row)
        try:
            return originals['_canonical_vapor_pressure_runtime'](
                self, symbol, props, allow_online=allow_online, minimum_pressure_bar=minimum_pressure_bar)
        finally:
            row['wall_seconds'] = time.perf_counter()-start
            row['memory_stored_after'] = key in getattr(self, '_canonical_vapor_pressure_curves', {})
            after = self._canonical_vapor_pressure_cache_key(symbol, props, allow_online, minimum_pressure_bar)
            row['inputs_changed_during_resolution'] = after != key
            stack.pop()
    def build(self, *args, **kwargs):
        if stack:
            stack[-1]['builds'] += 1
        return originals['_build_canonical_vapor_pressure_runtime'](self, *args, **kwargs)
    def load(self, *args, **kwargs):
        result = originals['_load_persistent_canonical_vapor_pressure_runtime'](self, *args, **kwargs)
        if stack:
            stack[-1]['persistent_hits'] += int(result is not None)
        return result
    def store(self, *args, **kwargs):
        if stack:
            stack[-1]['stores'] += 1
        return originals['_store_persistent_canonical_vapor_pressure_runtime'](self, *args, **kwargs)
    def admission(allow_online, state):
        result = originals['_online_attempt_is_persistable'](allow_online, state)
        if stack:
            stack[-1]['admission'].append({'allow_online': allow_online, 'state': str(state), 'accepted': result})
        return result
    PropertyResolver._canonical_vapor_pressure_runtime = runtime
    PropertyResolver._build_canonical_vapor_pressure_runtime = build
    PropertyResolver._load_persistent_canonical_vapor_pressure_runtime = load
    PropertyResolver._store_persistent_canonical_vapor_pressure_runtime = store
    PropertyResolver._online_attempt_is_persistable = staticmethod(admission)
    try:
        yield
    finally:
        for name, method in originals.items():
            setattr(PropertyResolver, name, staticmethod(method) if name == '_online_attempt_is_persistable' else method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with trace_cache(rows):
        for repeat in range(2):
            start = len(rows)
            sim = Simulator.from_file(ROOT/'examples/utility_surface_sizing.pfd')
            sim.initialize()
            result = sim.run()
            if not result.converged or result.errors:
                raise RuntimeError(result.errors)
            for row in rows[start:]:
                row['repeat'] = repeat
                print(json.dumps(row), flush=True)
            with (args.output/f'run-{repeat}.json').open('x') as stream:
                json.dump(rows[start:], stream, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
