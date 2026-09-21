"""Benchmark ready-solve CPU time for phi-phi-heavy PFD examples.

Run from the repository root, for example::

    python scripts/performance/benchmark_phi_phi_examples.py \
        --output /tmp/phi-phi-examples.json

Each repeat constructs and initializes a new simulator before timing only
``Simulator.run()``. Process CPU time is the primary metric so unrelated host
contention is not attributed to the example under test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulator import Simulator


DEFAULT_EXAMPLES = (
    ROOT / 'examples' / 'equilibrium_methanol_synthesis_recycle.pfd',
    ROOT / 'examples' / 'methanol_synthesis.pfd',
    ROOT / 'examples' / 'methanol_synthesis_psrk.pfd',
)


def benchmark(path: Path, repeats: int) -> dict:
    samples = []
    for _ in range(repeats):
        simulator = Simulator.from_file(path)
        simulator.initialize()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        result = simulator.run()
        samples.append({
            'wall_seconds': time.perf_counter() - wall_start,
            'cpu_seconds': time.process_time() - cpu_start,
            'converged': result.converged,
            'errors': result.errors,
        })
    cpu_values = [sample['cpu_seconds'] for sample in samples]
    return {
        'example': path.name,
        'repeats': repeats,
        'median_cpu_seconds': statistics.median(cpu_values),
        'samples': samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples', nargs='*', type=Path)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    paths = args.examples or DEFAULT_EXAMPLES
    payload = [benchmark(path, args.repeats) for path in paths]
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
