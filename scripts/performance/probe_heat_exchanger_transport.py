#!/usr/bin/env python3
"""Reproduce geometry/utility sizing and grid checks with isolated caches.

Run from the repository root (outputs must be new):
  .venv/bin/python scripts/performance/probe_heat_exchanger_transport.py \
      --output /tmp/exchanger-transport-study \
      2>&1 | tee /tmp/exchanger-transport-study.log

Sequential, deterministic cases; no random inputs. Complete metrics and errors
are persisted per case. Timing is exploratory, not a controlled benchmark.
"""

import argparse
import json
import os
import re
from pathlib import Path
import sys
import time
import traceback

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from chemical_properties import ChemicalDatabase
from simulator import Simulator
from tests.cache_isolation import isolated_runtime_caches
from thermodynamics import create_thermodynamics
from unit_operations_basic import HeatExchanger


def simulate_example(name, segments):
    source = (ROOT/'examples'/name).read_text()
    source = re.sub(r'curve_segments = \d+', f'curve_segments = {segments}', source)
    simulator = Simulator.from_string(source)
    result = simulator.run()
    if not result.converged:
        raise RuntimeError('Example did not converge')
    return {key: value['performance'] for key, value in simulator.get_results_dict()['units'].items()}


def phase_change_case(segments):
    thermo = create_thermodynamics(['water'], 'STEAM', db=ChemicalDatabase(enable_online=False))
    hot = thermo.calculate_state_PQ(5, .8, 20, {'water': 1})
    cold = thermo.calculate_state_PQ(3, .05, 100, {'water': 1})
    result = HeatExchanger('PHASE', thermo, {
        'U_model': 'double_pipe', 'wall_material': 'stainless_steel_304',
        'tube_inner_diameter': .02, 'tube_outer_diameter': .024,
        'shell_inner_diameter': .04, 'Q': 20, 'curve_segments': segments,
    }).solve({'shell_in': hot, 'tube_in': cold})
    performance = result.performance
    performance['hot_energy_residual_kW'] = (
        hot.F*(hot.H-result.outlet_streams['shell_out'].H)/3600-20)
    performance['cold_energy_residual_kW'] = (
        cold.F*(result.outlet_streams['tube_out'].H-cold.H)/3600-20)
    performance['vapor_fraction_hot_out'] = result.outlet_streams['shell_out'].vapor_fraction
    performance['vapor_fraction_cold_out'] = result.outlet_streams['tube_out'].vapor_fraction
    return {'PHASE': performance}


def compact(performance):
    keys = ('duty_kW', 'U_W_m2_K', 'area_required_m2', 'area_m2', 'min_approach_K',
            'LMTD_correction', 'utility_mass_flow_kg_h', 'electric_power_kW',
            'fuel_mass_flow_kg_h', 'utility_area_preliminary', 'vapor_fraction_hot_out',
            'vapor_fraction_cold_out', 'hot_energy_residual_kW', 'cold_energy_residual_kW')
    result = {key: performance[key] for key in keys if key in performance}
    nodes = performance.get('calculated_U_nodes', [])
    if nodes:
        result['maximum_wall_residual_K'] = max(row['wall_temperature_residual_K'] for row in nodes)
        result['sample_count'] = len(nodes)
        result['film_correlations'] = sorted({side['correlation'] for row in nodes
            for key, side in row.items() if key in ('tube', 'shell', 'annulus')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--segments', type=int, default=40)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = {
        'double_pipe': lambda n: simulate_example('double_pipe_calculated_u.pfd', n),
        'shell_tube': lambda n: simulate_example('shell_tube_calculated_u.pfd', n),
        'utilities': lambda n: simulate_example('utility_surface_sizing.pfd', n),
        'phase_change': phase_change_case,
    }
    manifest = {'segments': args.segments, 'randomness': 'none', 'cases': {}}
    failures = 0
    with isolated_runtime_caches():
        for case, run in cases.items():
            record = {}
            try:
                for n in ((args.segments,) if case == 'utilities' else (args.segments, 2*args.segments)):
                    started = time.perf_counter()
                    metrics = run(n)
                    record[str(n)] = {'seconds': time.perf_counter()-started, 'units': metrics}
                summary = {n: {unit: compact(p) for unit, p in value['units'].items()}
                           for n, value in record.items()}
                manifest['cases'][case] = {'status': 'ok', 'summary': summary}
            except Exception as error:
                failures += 1
                record['error'] = str(error)
                record['traceback'] = traceback.format_exc()
                manifest['cases'][case] = {'status': 'error', 'error': str(error)}
            with (args.output/f'{case}.json').open('x') as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
                stream.write('\n')
            print(case, json.dumps(manifest['cases'][case], allow_nan=False), flush=True)
    with (args.output/'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
