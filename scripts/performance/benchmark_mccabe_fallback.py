#!/usr/bin/env python3
"""Compare the legacy and finite McCabe-Thiele fallback on a pressure-swing case.

Run from the repository root, persisting both output streams:
    set -o pipefail
    python scripts/performance/benchmark_mccabe_fallback.py --output /tmp/mccabe-fallback \
        2>&1 | tee /tmp/mccabe-fallback.log > /dev/null

One CPU thread; fresh initialized simulations; imports, setup and one warm-up
per variant excluded. Variant order alternates. No randomness. Temporary test
cache isolation protects the existing runtime databases. The legacy search is
reproduced only in this experiment; production has one shared implementation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys
import time
from unittest.mock import patch
import warnings

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from simulator import Simulator
from tests.cache_isolation import isolated_runtime_caches
from unit_operations_distillation import McCabeThieleDistillation


def legacy_search(excess, samples, bounds):
    return float(minimize_scalar(
        lambda x:max(excess(x), 0.), bounds=bounds, method='bounded',
        options={'xatol':1e-10},
    ).x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    case = ROOT/'examples'/'ethanol_pressure_swing_recycle_wasteful.pfd'
    updated = McCabeThieleDistillation._minimum_binary_stage_excess
    searches = {'legacy':legacy_search, 'finite':updated}
    (args.output/'manifest.json').write_text(json.dumps({
        'python':sys.version, 'numpy':np.__version__, 'scipy':scipy.__version__,
        'repeats':args.repeats, 'threads':1,
        'case':str(case.relative_to(ROOT)),
        'source_sha256':hashlib.sha256((ROOT/'unit_operations_distillation.py').read_bytes()).hexdigest(),
    }, indent=2)+'\n')
    records = []
    with isolated_runtime_caches(), (args.output/'results.jsonl').open('x') as handle:
        for repeat in range(args.repeats+1):
            order = ('legacy', 'finite') if repeat % 2 == 0 else ('finite', 'legacy')
            for variant in order:
                sim = Simulator.from_file(case).initialize()
                evaluations = 0
                def search(excess, samples, bounds):
                    def evaluate(x):
                        nonlocal evaluations
                        evaluations += 1
                        return excess(x)
                    return searches[variant](evaluate, samples, bounds)
                with patch.object(McCabeThieleDistillation, '_minimum_binary_stage_excess', staticmethod(search)), \
                     warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', RuntimeWarning)
                    start = time.perf_counter()
                    result = sim.run()
                    elapsed = time.perf_counter()-start
                if not result.converged or result.errors:
                    raise AssertionError(result.errors)
                row = {'repeat':repeat, 'variant':variant, 'seconds':elapsed,
                       'fallback_objective_evaluations':evaluations,
                       'scalar_warnings':sum('invalid value encountered in scalar' in str(w.message) for w in caught),
                       'mass_balance_error':result.mass_balance_error,
                       'energy_balance_error':result.energy_balance_error,
                       'column_residuals':{name:r.performance['mesh_residual'] for name,r in result.units.items()
                                           if 'mesh_residual' in r.performance},
                       'streams':{name:{'F':s.F, 'T':s.T, 'x':s.composition} for name,s in result.streams.items()}}
                records.append(row)
                handle.write(json.dumps(row)+'\n')
                handle.flush()
                print(variant, repeat, elapsed, 'warnings=',row['scalar_warnings'], flush=True)
    summary = {variant:{'median_seconds':median(r['seconds'] for r in records if r['variant']==variant and r['repeat']>0),
                        'scalar_warnings':sum(r['scalar_warnings'] for r in records if r['variant']==variant and r['repeat']>0),
                        'fallback_evaluations':[r['fallback_objective_evaluations'] for r in records if r['variant']==variant and r['repeat']>0]}
               for variant in searches}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
