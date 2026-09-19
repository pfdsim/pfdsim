"""Seeded warm-kernel timings and fugacity residuals for EOS K iteration.

Run from the repository: python scripts/benchmark_phi_phi.py --output /tmp/eos.json
Each timed call has a new temperature so memoization cannot hide solver cost.
Initialization/JIT time is reported separately from the calculation loop.
"""

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chemical_properties import ChemicalDatabase
from thermodynamics import create_thermodynamics


def benchmark(calls=100):
    rng = random.Random(391991)
    results = []
    db = ChemicalDatabase(enable_online=False)
    for method, components, T, P, fraction in (
        ('PR', ['C3H8', 'C4H10'], 323.15, 10.0, .45),
        ('SRK', ['CO2', 'water'], 350., 50., .03),
        ('PSRK', ['CO2', 'water'], 350., 50., .03),
        ('RKSMHV2', ['CO2', 'water'], 350., 50., .03),
    ):
        start = time.perf_counter()
        model = create_thermodynamics(components, method, db)
        x = {components[0]: fraction, components[1]: 1-fraction}
        model.K_values(T, P, x)
        init_seconds = time.perf_counter()-start
        temperatures = [T+rng.uniform(-2., 2.) for _ in range(calls)]
        start = time.perf_counter()
        ks = [model.K_values(t, P, x) for t in temperatures]
        elapsed = time.perf_counter()-start
        residuals = []
        for t, K in zip(temperatures, ks):
            total = sum(x[c]*K[c] for c in components)
            y = {c: x[c]*K[c]/total for c in components}
            pl = model.fugacity_coefficients(t, P, x, 'liquid')
            pv = model.fugacity_coefficients(t, P, y, 'vapor')
            residuals.append(max(abs(math.log(K[c]*pv[c]/pl[c])) for c in components))
        results.append(dict(method=method, init_seconds=init_seconds,
                            calls=calls, seconds=elapsed, microseconds_per_call=elapsed*1e6/calls,
                            max_log_fugacity_residual=max(residuals), K=ks[0]))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calls', type=int, default=100)
    args = parser.parse_args()
    results = benchmark(args.calls)
    args.output.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))
