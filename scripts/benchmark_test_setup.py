"""Measure common unit-operation setup with cold and copied safe caches.

Both modes use the offline test policy. This measures initialization rather
than solver changes; source fixtures can legitimately change selected property
values compared with a cold offline fallback. Each mode must be deterministic.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.cache_isolation import isolated_runtime_caches
from tests.test_unit_operations import UnitOperationSmokeTests


def measure(seeded, repeats):
    durations = []
    expected = None
    # Exclude imports, template construction, and first numerical compilation.
    for index in range(repeats + 1):
        started = time.perf_counter()
        with isolated_runtime_caches(seeded=seeded):
            case = UnitOperationSmokeTests()
            case.setUp()
            signature = {
                name: {'H': getattr(case, name).H, 'S': getattr(case, name).S}
                for name in ('liquid', 'liquid2', 'gas', 'hot', 'cold', 'lle_feed',
                             'solvent', 'reactive_feed')
            }
            elapsed = time.perf_counter() - started
        if expected is not None and signature != expected:
            raise AssertionError('Independent cache scopes produced inconsistent stream states')
        expected = signature
        if index:
            durations.append(elapsed)
    return {'seconds': durations, 'median_seconds': statistics.median(durations),
            'stream_states': expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    report = {
        'python': sys.version,
        'source_fixture_sha256': hashlib.sha256(
            (ROOT/'tests'/'fixtures'/'runtime_cache.json').read_bytes()).hexdigest(),
        'cold': measure(False, args.repeats),
        'safe_copy': measure(True, args.repeats),
    }
    report['setup_speedup'] = (report['cold']['median_seconds'] /
                               report['safe_copy']['median_seconds'])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
