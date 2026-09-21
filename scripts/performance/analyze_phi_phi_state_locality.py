"""Measure locality between uncached phi-phi solves in real PFD examples.

Run from the repository root, for example::

    python scripts/performance/analyze_phi_phi_state_locality.py \
        examples/equilibrium_methanol_synthesis_recycle.pfd \
        examples/methanol_synthesis.pfd \
        examples/methanol_synthesis_psrk.pfd \
        --output /tmp/phi-phi-locality.json

The script leaves production behavior unchanged. It records only calls that
miss the exact EOS K-value cache, then reports how close each state is to the
previous miss handled by the same EOS object.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cubic_eos import CubicEOS
from rk_eos import RedlichKwong
from simulator import Simulator
from thermodynamics_models.psrk import PSRK


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class LocalityRecorder:
    def __init__(self) -> None:
        self.previous = {}
        self.rows = defaultdict(list)
        self.calls = defaultdict(int)
        self.hits = defaultdict(int)

    def record(self, label, instance, temperature, pressure, composition, result):
        object_key = (label, id(instance))
        wilson = instance._wilson_K(float(temperature), float(pressure))
        wilson_values = tuple(
            float(wilson[component]) for component in instance.components
        )
        current = (
            float(temperature),
            float(pressure),
            tuple(float(value) for value in composition),
            tuple(float(value) for value in result),
            wilson_values,
        )
        previous = self.previous.get(object_key)
        if previous is not None:
            old_temperature, old_pressure, old_composition, old_result, _ = previous
            self.rows[label].append({
                'relative_temperature_change': abs(
                    current[0] - old_temperature
                ) / max(current[0], old_temperature, 1.0),
                'log_pressure_change': abs(math.log(
                    max(current[1], 1.0e-300)
                    / max(old_pressure, 1.0e-300)
                )),
                'composition_l1_change': sum(
                    abs(new - old)
                    for new, old in zip(current[2], old_composition)
                ),
                'log_k_max_change': max(
                    abs(math.log(max(new, 1.0e-300) / max(old, 1.0e-300)))
                    for new, old in zip(current[3], old_result)
                ),
                'previous_k_to_current_wilson_log_distance': max(
                    abs(math.log(max(old, 1.0e-300) / max(seed, 1.0e-300)))
                    for old, seed in zip(old_result, current[4])
                ),
                'current_k_to_wilson_log_distance': max(
                    abs(math.log(max(new, 1.0e-300) / max(seed, 1.0e-300)))
                    for new, seed in zip(current[3], current[4])
                ),
            })
        self.previous[object_key] = current

    def summary(self):
        result = {}
        labels = sorted(set(self.calls) | set(self.rows))
        for label in labels:
            rows = self.rows[label]
            metrics = {}
            for metric in (
                'relative_temperature_change',
                'log_pressure_change',
                'composition_l1_change',
                'log_k_max_change',
                'previous_k_to_current_wilson_log_distance',
                'current_k_to_wilson_log_distance',
            ):
                values = [row[metric] for row in rows]
                metrics[metric] = {
                    'median': _percentile(values, 0.5),
                    'p90': _percentile(values, 0.9),
                    'p99': _percentile(values, 0.99),
                    'maximum': max(values) if values else None,
                }
            eligibility = {}
            for composition_limit in (0.01, 0.05, 0.10, 0.25, 0.50):
                eligible = [
                    row
                    for row in rows
                    if row['relative_temperature_change'] <= 0.05
                    and row['log_pressure_change'] <= 0.10
                    and row['composition_l1_change'] <= composition_limit
                ]
                k_changes = [row['log_k_max_change'] for row in eligible]
                eligibility[str(composition_limit)] = {
                    'pairs': len(eligible),
                    'fraction': len(eligible) / len(rows) if rows else 0.0,
                    'log_k_max_change_median': _percentile(k_changes, 0.5),
                    'log_k_max_change_p99': _percentile(k_changes, 0.99),
                    'log_k_max_change_maximum': (
                        max(k_changes) if k_changes else None
                    ),
                }
            guarded = [
                row
                for row in rows
                if row['relative_temperature_change'] <= 0.05
                and row['log_pressure_change'] <= 0.10
                and row['composition_l1_change'] <= 0.01
                and row['previous_k_to_current_wilson_log_distance'] <= 4.0
            ]
            guarded_changes = [row['log_k_max_change'] for row in guarded]
            result[label] = {
                'calls': self.calls[label],
                'exact_cache_hits': self.hits[label],
                'cache_misses': self.calls[label] - self.hits[label],
                'consecutive_miss_pairs': len(rows),
                'metrics': metrics,
                'warm_start_eligibility': eligibility,
                'guarded_warm_start': {
                    'pairs': len(guarded),
                    'fraction': len(guarded) / len(rows) if rows else 0.0,
                    'log_k_max_change_median': _percentile(
                        guarded_changes, 0.5
                    ),
                    'log_k_max_change_p99': _percentile(
                        guarded_changes, 0.99
                    ),
                    'log_k_max_change_maximum': (
                        max(guarded_changes) if guarded_changes else None
                    ),
                },
            }
        return result


def _install_wrappers(recorder: LocalityRecorder):
    originals = []

    def install(cls, label, normalize, cache_key):
        original = cls.phi_phi_K_values
        originals.append((cls, original))

        def wrapped(self, temperature, pressure, composition, max_iter=80):
            values = normalize(self, composition)
            key = cache_key(self, temperature, pressure, values, max_iter)
            recorder.calls[label] += 1
            if key in self._phi_phi_k_cache:
                recorder.hits[label] += 1
                return original(
                    self, temperature, pressure, composition, max_iter
                )
            result = original(
                self, temperature, pressure, composition, max_iter
            )
            recorder.record(
                label,
                self,
                temperature,
                pressure,
                values,
                [result[component] for component in self.components],
            )
            return result

        cls.phi_phi_K_values = wrapped

    install(
        CubicEOS,
        'cubic',
        lambda self, composition: tuple(
            self._normalized_composition(composition)[component]
            for component in self.components
        ),
        lambda self, temperature, pressure, values, max_iter: (
            float(temperature),
            float(pressure),
            self._composition_cache_key(dict(zip(self.components, values))),
            int(max_iter),
        ),
    )
    install(
        RedlichKwong,
        'rk',
        lambda self, composition: tuple(
            self._normalized_composition(composition)[component]
            for component in self.components
        ),
        lambda self, temperature, pressure, values, max_iter: (
            float(temperature),
            float(pressure),
            self._composition_cache_key(dict(zip(self.components, values))),
            int(max_iter),
        ),
    )
    install(
        PSRK,
        'psrk',
        lambda self, composition: tuple(
            self._normalize_composition(composition)
        ),
        lambda self, temperature, pressure, values, max_iter: (
            float(temperature),
            float(pressure),
            tuple(values),
            int(max_iter),
        ),
    )
    return originals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('examples', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    recorder = LocalityRecorder()
    originals = _install_wrappers(recorder)
    example_results = []
    try:
        for path in args.examples:
            simulator = Simulator.from_file(path)
            simulator.initialize()
            start = time.perf_counter()
            result = simulator.run()
            example_results.append({
                'example': path.name,
                'seconds': time.perf_counter() - start,
                'converged': result.converged,
                'errors': result.errors,
            })
    finally:
        for cls, original in originals:
            cls.phi_phi_K_values = original

    payload = {
        'examples': example_results,
        'locality': recorder.summary(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
