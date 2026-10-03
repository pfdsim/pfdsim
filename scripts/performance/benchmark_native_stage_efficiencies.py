#!/usr/bin/env python3
"""Time public rigorous VLE/VLLE solves with native Murphree efficiencies.

    set -o pipefail
    python scripts/performance/benchmark_native_stage_efficiencies.py --output /tmp/native-eta \
        2>&1 | tee /tmp/native-eta.log > /dev/null

No monkeypatching. One thread, sequential timeout-bounded workers, isolated
caches, fresh units and profiles, alternating order, one excluded warm-up and
three measured solves. Timings include unit initialization and audits, exclude
imports and property hydration. No RNG. Controls include omitted efficiency
and explicit E=1 to verify retained equilibrium behavior.
"""

import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from tests.cache_isolation import isolated_runtime_caches
from thermodynamics import create_thermodynamics
from unit_operations_distillation import RigorousDistillation

CASES = ('vle_methanol','vle_pr_partial_vapor','vlle_butanol_selective',
         'vlle_butanol_copooled','vlle_rk_partial_vapor')


def inputs(case):
    params = {'N_stages':20,'feed_stage':10,'P_condenser':1.,'P_drop_per_stage':0.,
              'mesh_tolerance':1e-7,'acceptable_mesh_residual':1e-7,
              'max_iterations':80,'max_jacobian_evaluations':80}
    method,T,phase = 'NRTL',298.15,'liquid'
    if case.startswith('vle_'):
        z = {'methanol':.4,'water':.6}
        params.update(reflux_ratio=2.,D_to_F=.35)
        if case == 'vle_pr_partial_vapor':
            method,z,T,phase = 'PR',{'benzene':.5,'toluene':.5},425.,'vapor'
            params.update(condenser_type='partial',D_to_F=.45)
    else:
        z = {'butanol':.4,'water':.6}
        params['stage_phase_model'] = 'VLLE'
        if case == 'vlle_butanol_copooled':
            params.update(reflux_ratio=1.2,D_to_F=.773,vlle_seed='cheap')
        elif case == 'vlle_rk_partial_vapor':
            method,T,phase = 'NRTL-RK',430.,'vapor'
            params.update(condenser_type='partial',reflux_ratio=2.,D_to_F=.65)
        else:
            params.update(D_to_F=.6119275124597854,distillate_liquid1_component='butanol',
                          distillate_liquid1_fraction=0.,distillate_liquid2_fraction=1.)
    thermo = create_thermodynamics(list(z),method)
    feed = thermo.calculate_state(T,1.,100.,z,phase=phase,flash=False)
    return thermo,feed,params


def worker(case,output,repeats):
    variants = ('default','E1.0','E0.7','E0.5')
    rows = []
    with isolated_runtime_caches(),(output/f'{case}.jsonl').open('x') as handle:
        thermo,feed,params = inputs(case)
        for repeat in range(repeats+1):
            for variant in variants if repeat % 2 == 0 else variants[::-1]:
                settings = dict(params)
                if variant != 'default':
                    settings['stage_efficiency'] = float(variant[1:])
                unit = RigorousDistillation('COL',thermo,settings)
                start = time.perf_counter()
                result = unit.solve({'feed':feed})
                elapsed = time.perf_counter()-start
                p = result.performance
                D,B = result.outlet_streams['distillate'],result.outlet_streams['bottoms']
                balance = {c:feed.F*feed.composition[c]-D.F*D.composition[c]-B.F*B.composition[c]
                           for c in thermo.components}
                energy = D.F*D.H+B.F*B.H-feed.F*feed.H-3600*(p['condenser_duty_kW']+p['reboiler_duty_kW'])
                if p['mesh_residual'] > 1e-7 or p['max_murphree_residual'] > 1e-7:
                    raise AssertionError('Residual target not met')
                if max(abs(v) for v in balance.values()) > 1e-5 or abs(energy) > .1:
                    raise AssertionError('External balances failed')
                row = {'case':case,'variant':variant,'repeat':repeat,'warmup':repeat==0,'seconds':elapsed,
                       'mesh_residual':p['mesh_residual'],'max_murphree_residual':p['max_murphree_residual'],
                       'iterations':p['solver_iterations'],'jacobian_method':p['jacobian_method'],
                       'component_balance_kmol_h':balance,'energy_balance_kJ_h':energy,
                       'distillate_x':D.composition,'bottoms_x':B.composition,
                       'reflux_ratio':p['reflux_ratio'],'topology':p.get('vlle_topology'),
                       'projection_cycle_recoveries':p.get('vlle_projection_cycle_recoveries',0)}
                handle.write(json.dumps(row)+'\n')
                handle.flush()
                rows.append(row)
                print(case,variant,repeat,'s=',elapsed,'residual=',p['mesh_residual'],flush=True)
    summary = {v:{'median_seconds':median(r['seconds'] for r in rows if r['variant']==v and not r['warmup']),
                  'representative':next(r for r in rows if r['variant']==v and not r['warmup'])} for v in variants}
    for v in variants:
        summary[v]['relative_to_default'] = summary[v]['median_seconds']/summary['default']['median_seconds']
    (output/f'{case}.json').write_text(json.dumps(summary,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--timeout',type=float,default=90.)
    parser.add_argument('--worker',choices=CASES)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error('repeats and timeout must be positive')
    if args.worker:
        worker(args.worker,args.output,args.repeats)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'manifest.json').write_text(json.dumps({'cases':args.cases,'repeats':args.repeats,
        'threads':1,'timeout':args.timeout,'seed':'deterministic, no RNG','python':sys.version,
        'timing':'public unit.solve including initialization and audits; excludes hydration'},indent=2)+'\n')
    statuses = {}
    for case in args.cases:
        with (args.output/f'{case}.log').open('x') as handle:
            try:
                result = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',case,
                    '--output',str(args.output),'--repeats',str(args.repeats)],
                    stdout=handle,stderr=subprocess.STDOUT,timeout=args.timeout,check=False)
                status = {'returncode':result.returncode}
            except subprocess.TimeoutExpired:
                status = {'status':'timeout'}
        statuses[case] = status
        (args.output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
        print(case,status,flush=True)
    summary = {c:json.loads((args.output/f'{c}.json').read_text()) for c in args.cases if (args.output/f'{c}.json').exists()}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('RESULTS',args.output,flush=True)
    if any(s.get('returncode') != 0 for s in statuses.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
