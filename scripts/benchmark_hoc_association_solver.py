#!/usr/bin/env python3
"""Measure HOC association cost and error across concentration/strength limits."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from thermodynamics_models.second_virial import (  # noqa: E402
    ChemicalAssociationSecondVirialVaporBackend,
)


class IdenticalAcidProvider:
    """Identically interacting labels with an analytical dimerization solution."""

    name = 'identical acids'
    components = ('A', 'B')
    associating_components = components

    def __init__(self, kappa):
        self.kappa = kappa

    def physical_second_virial_matrix(self, T, order=0):
        return ((0., 0.), (0., 0.))

    def association_constant_matrix(self, T, order=0):
        k = self.kappa if order == 0 else 0.
        return ((k, 2. * k), (2. * k, k))


def benchmark(repeats):
    rows = []
    for label, kappa, trace in (
        ('zero association', 0., 0.3),
        ('weak association', 1e-9, 0.3),
        ('ordinary mixture', 10., 0.3),
        ('strong association', 1e6, 0.3),
        ('trace acid', 10., 1e-100),
        ('absent acid', 10., 0.),
    ):
        provider = IdenticalAcidProvider(kappa)
        composition = {'A': 1. - trace, 'B': trace}
        expected_phi = 2. / (1. + math.sqrt(1. + 4. * kappa))
        for fallback_only in (False, True):
            backend = ChemicalAssociationSecondVirialVaporBackend(
                provider.components, provider,
            )
            context = patch(
                'compiled_vdm.solve_n_acid_true_moles', return_value=None,
            ) if fallback_only else nullcontext()
            with context:
                # Exclude imports and JIT compilation from steady-state timings.
                backend.association_state(400., 1., composition)
                durations = []
                max_error = 0.
                for _ in range(repeats):
                    backend._association_state_cache.clear()
                    started = time.perf_counter()
                    state = backend.association_state(400., 1., composition)
                    durations.append(time.perf_counter() - started)
                    max_error = max(max_error, *(
                        abs(value / expected_phi - 1.)
                        for value in state['chemical_phi'].values()
                    ))
            if max_error > 1e-8:
                raise RuntimeError(f'{label}: relative fugacity error {max_error:g}')
            rows.append({
                'case': label,
                'fallback_only': fallback_only,
                'median_uncached_microseconds': 1e6 * statistics.median(durations),
                'max_relative_fugacity_error': max_error,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    print(json.dumps(benchmark(args.repeats), indent=2))


if __name__ == '__main__':
    main()
