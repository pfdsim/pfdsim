#!/usr/bin/env python3
"""Compare saturated, subcooled, and absolute-temperature native VLLE columns.

Run from the repository root, persisting both output streams:
    set -o pipefail
    python scripts/probe_subcooled_total_condenser.py --output /tmp/subcooled.json \
        2>&1 | tee /tmp/subcooled.log

Uses the compact butanol example, one CPU thread, isolated runtime caches,
and no randomness. No monkeypatching or external recycle is used. This is a
numerical correctness probe rather than a timing benchmark.
"""

import argparse
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from pfd_parser import Parameter
from simulator import Simulator
from tests.cache_isolation import isolated_runtime_caches


def solve(parameter=None):
    sim = Simulator.from_file(ROOT/'examples'/'butanol_water_phase_selective_distillation.pfd')
    if parameter is not None:
        sim.pfd.units[0].params.append(parameter)
    result = sim.run()
    if not result.converged or result.errors:
        raise AssertionError(result.errors)
    feed,bottom = result.streams['Feed'],result.streams['Butanol-Bottoms']
    performance = result.units['COL-1'].performance
    if performance['mesh_residual'] > 1e-7 or performance['vlle_max_log_fugacity_residual'] > 1e-6:
        raise AssertionError('Column failed residual/fugacity checks')
    return {'bottoms_purity':bottom.composition['butanol'],
            'butanol_recovery':bottom.F*bottom.composition['butanol']/(feed.F*feed.composition['butanol']),
            'mass_balance_error':result.mass_balance_error,
            'energy_balance_error':result.energy_balance_error,
            'performance':performance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--subcooling',type=float,default=5.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with isolated_runtime_caches():
        saturated = solve()
        cooled = solve(Parameter('condenser_subcooling',args.subcooling,'K'))
        absolute = solve(Parameter('condenser_temperature',
                                  cooled['performance']['condenser_temperature_K'],'K'))
    for key in ('bottoms_purity','butanol_recovery'):
        if abs(cooled[key]-absolute[key]) > 1e-7:
            raise AssertionError(f'Absolute/subcooling mismatch: {key}')
    results = {'saturated':saturated,'subcooled':cooled,'absolute_temperature':absolute}
    with args.output.open('x') as handle:
        json.dump(results,handle,indent=2)
        handle.write('\n')
    for label,row in results.items():
        p = row['performance']
        print(label,'purity=',row['bottoms_purity'],'recovery=',row['butanol_recovery'],
              'top_C=',p['T_top_C'],'RR=',p['reflux_ratio'],
              'mesh=',p['mesh_residual'],'fugacity=',p['vlle_max_log_fugacity_residual'],flush=True)
    print('RESULTS:',args.output,flush=True)


if __name__ == '__main__':
    main()
