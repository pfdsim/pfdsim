"""Compare homogeneous PH temperature solves with full trial-state iteration.

Run from any directory, e.g. python scripts/performance/probe_non_eos_ph.py
--samples 60 --output /tmp/non-eos-ph.json. No random workload is used.
Compilation is reported separately; numerical residuals use the authoritative
mixture enthalpy, independently of the accelerated evaluator.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from thermodynamics import create_thermodynamics
from unit_operations_basic import _ThermoStateSolver


def clear_caloric_caches(thermo):
    for name in (
        '_enthalpy_ideal_cache', '_enthalpy_liquid_cache', '_cp_ideal_cache',
        '_cp_liquid_cache', '_cp_integral_cache', '_excess_cp_cache',
        '_vapor_residual_cp_cache',
    ):
        cache = getattr(thermo, name, None)
        if cache is not None:
            cache.clear()


def probe(model, phase, samples):
    components = ['water', 'ethanol']
    if 'VDM' in model:
        components = ['CH3COOH', 'H2O']
    if model == 'STEAM':
        components = ['H2O']
    thermo = create_thermodynamics(components, model)
    fraction = float(phase == 'vapor')
    pressure = 2.0
    start_temperature = 330.0 if phase == 'liquid' else 430.0
    records = []
    for index in range(samples):
        x = 0.4 + 0.04 * math.sin(index * 0.13)
        composition = ({components[0]: 1.0} if len(components) == 1 else
                       {components[0]: x, components[1]: 1.0 - x})
        temperature = start_temperature + 0.15 * index
        target = thermo.mixture_enthalpy(composition, temperature, fraction, P=pressure)
        records.append((composition, target))
    warm_start = time.perf_counter()
    thermo.temperature_at_PH(pressure, records[0][1], records[0][0],
                             phase=phase, T_guess=start_temperature - 20.0)
    initialization_seconds = time.perf_counter() - warm_start
    results = {}
    for accelerated in (False, True):
        clear_caloric_caches(thermo)
        solver = _ThermoStateSolver(thermo, 'non-EOS benchmark')
        outputs = []
        begin = time.perf_counter()
        # Disable only the new direct entry point in the baseline; retain the
        # existing unit-operation Newton/bracketing algorithms unchanged.
        existing_ph = thermo.calculate_state_PH if model == 'STEAM' else None
        with patch.object(thermo, 'calculate_state_PH', existing_ph):
            for composition, target in records:
                if accelerated:
                    temperature, residual = solver.temperature_at_enthalpy(
                        pressure, 1.0, composition, target, start_temperature - 20.0,
                        force_phase=phase,
                    )
                else:
                    state, residual = solver.state_at_enthalpy(
                        pressure, 1.0, composition, target, start_temperature - 20.0,
                        force_phase=phase, include=('H',),
                    )
                    temperature = state.T
                outputs.append(temperature)
        elapsed = time.perf_counter() - begin
        errors = [
            abs(thermo.mixture_enthalpy(comp, T, fraction, P=pressure) - target)
            for T, (comp, target) in zip(outputs, records)
        ]
        results['accelerated' if accelerated else 'baseline'] = {
            'seconds': elapsed, 'microseconds_per_request': 1.0e6 * elapsed / samples,
            'maximum_enthalpy_residual_kJ_kmol': max(errors),
        }
    return {
        'model': model, 'phase': phase, 'samples': samples,
        'initialization_seconds': initialization_seconds, **results,
        'speedup': results['baseline']['seconds'] / results['accelerated']['seconds'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=60)
    parser.add_argument('--models', nargs='+', default=[
        'IDEAL', 'NRTL', 'UNIQUAC', 'UNIFAC', 'UNIFDMD', 'NRTL-PR', 'NRTL-VDM', 'STEAM',
    ])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    results = [probe(model, phase, args.samples) for model in args.models
               for phase in ('liquid', 'vapor')]
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    for row in results:
        print(f"{row['model']:12} {row['phase']:6} {row['speedup']:6.2f}x "
              f"residual={row['accelerated']['maximum_enthalpy_residual_kJ_kmol']:.3g}")


if __name__ == '__main__':
    main()
