#!/usr/bin/env python3
"""Small isolated reproduction of expired callbacks in Numba disk-cache keys.

    .venv/bin/python scripts/performance/probe_numba_callback_cache.py \
        --output /tmp/numba-callback-probe \
        2>&1 | tee /tmp/numba-callback-probe.log

The optional retention experiment patches only the child probe process to
hold deserialized dispatchers strongly. It is attribution, not a production
Numba patch. Each comparison starts from a copy of the identical seeded cache.
No repository caches are changed or inspected. No random inputs are used.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def worker(args):
    # Set cache location before importing Numba or defining dispatchers.
    os.environ['NUMBA_CACHE_DIR'] = str(args.cache)
    from numba import njit
    from numba.core.dispatcher import _MemoMixin
    pinned = []
    original = _MemoMixin._set_uuid
    if args.retain:
        def retain(self, uuid):
            pinned.append(self)
            return original(self, uuid)
        _MemoMixin._set_uuid = retain
    @njit(cache=True)
    def apply(callback, value):
        return callback(value)
    def callback(offset):
        @njit(cache=True)
        def evaluate(value):
            return value+offset
        return evaluate
    callbacks = [callback(i) for i in range(args.callbacks)]
    # Keep all *current* callbacks alive. The defect concerns reconstructed
    # callbacks from old index entries, not these ordinary module references.
    start = time.perf_counter()
    report = {'phase': args.worker, 'retention': args.retain,
              'callback_count': len(callbacks), 'numba_retention_limit': _MemoMixin._recent.maxlen}
    try:
        selected = callbacks if args.worker == 'seed' else callbacks[:1]
        for i, function in enumerate(selected):
            assert apply(function, 1.) == i+1.
        report.update(status='ok', cache_hits=sum(apply._cache_hits.values()),
                      cache_misses=sum(apply._cache_misses.values()))
    except Exception as error:
        report.update(status='error', error_type=type(error).__name__, error=str(error).split('\n')[0],
                      traceback_frames=[{'file': Path(f.filename).name, 'line': f.lineno, 'function': f.name}
                                        for f in traceback.extract_tb(error.__traceback__)])
    report.update(seconds=time.perf_counter()-start, pinned_dispatchers=len(pinned))
    print(json.dumps(report), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--callbacks', type=int, default=132)
    parser.add_argument('--worker', choices=('seed', 'reload'))
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--retain', action='store_true')
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.output is None:
        parser.error('--output is required')
    args.output.mkdir(parents=True, exist_ok=False)
    seeded = args.output/'seed-cache'
    reports = []
    def run(label, cache, phase, retain=False):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', phase,
                   '--cache', str(cache), '--callbacks', str(args.callbacks)]
        if retain:
            command.append('--retain')
        with (args.output/f'{label}.log').open('x') as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        lines = (args.output/f'{label}.log').read_text().splitlines()
        report = next((json.loads(line) for line in reversed(lines) if line.startswith('{')), None)
        if completed.returncode != 0 or report is None:
            raise RuntimeError(f'{label} failed; inspect its persisted log')
        reports.append(report)
        print(label, json.dumps(report), flush=True)
        return report
    if run('seed', seeded, 'seed')['status'] != 'ok':
        return 1
    for label, retain in (('ordinary-reload', False), ('retained-reload', True)):
        cache = args.output/(label+'-cache')
        shutil.copytree(seeded, cache)
        run(label, cache, 'reload', retain)
    with (args.output/'summary.json').open('x') as stream:
        json.dump(reports, stream, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
