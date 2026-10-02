#!/usr/bin/env python3
"""Search top-liquid routing for the 20-stage butanol/water VLLE column.

The feed is 100 kmol/h, 40 mol% 1-butanol, at 298.15 K and 1 bar, on stage
10 of 20. The total condenser can independently withdraw fractions of each
equilibrium liquid, returning the remainder internally. No external recycle
or purge. Stage equilibrium/stability and the active-set solver remain the
production implementations; only top routing and its specification are patched.

Every case runs in a fresh spawned process with an individual timeout. At most
five processes run concurrently, each with one numerical-library thread. Logs,
case results and source provenance are persisted. No randomness.

    set -o pipefail
    python scripts/probe_butanol_phase_selective_distillation.py \
        --output /tmp/butanol-phase-routing --workers 5 --timeout 45 \
        2>&1 | tee /tmp/butanol-phase-routing.log > /dev/null
"""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import textwrap
import traceback
from unittest.mock import patch

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import equilibrium_stage_vlle as vlle
from tests.cache_isolation import isolated_runtime_caches
from thermodynamics import create_thermodynamics
from unit_operations_distillation import RigorousDistillation

Column = vlle.EquationOrientedVLLEColumn
FEED_FLOW = 100.
FEED_BUTANOL = .4
TARGET_PURITY = .995


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2,
                              default=lambda scalar:scalar.item())+'\n')


def routing(model, top):
    second_is_rich = top['x2']['butanol'] > top['x1']['butanol']
    rich = top['x2'] if second_is_rich else top['x1']
    aqueous = top['x1'] if second_is_rich else top['x2']
    fraction = top['beta'] if second_is_rich else 1-top['beta']
    total = top['L']+top['V']
    rich_flow, aqueous_flow = total*fraction, total*(1-fraction)
    f_a, f_o = model.unit.params['_probe_aqueous_draw'], model.unit.params['_probe_organic_draw']
    draws = (f_a*aqueous_flow, f_o*rich_flow)
    returns = ((1-f_a)*aqueous_flow, (1-f_o)*rich_flow)

    def blend(flows, fallback):
        amount = sum(flows)
        if amount <= 0:
            return dict(fallback)
        return {c:(flows[0]*aqueous[c]+flows[1]*rich[c])/amount for c in model.components}

    xD = blend(draws, aqueous)
    xR = blend(returns, rich)
    h_a = model.thermo.mixture_enthalpy(aqueous, top['T'], 0., P=model.pressures[0])
    h_o = model.thermo.mixture_enthalpy(rich, top['T'], 0., P=model.pressures[0])
    hD = (draws[0]*h_a+draws[1]*h_o)/sum(draws) if sum(draws)>0 else h_a
    hR = (returns[0]*h_a+returns[1]*h_o)/sum(returns) if sum(returns)>0 else h_o
    return {'aqueous_x':aqueous, 'organic_x':rich, 'organic_fraction':fraction,
            'distillate_x':xD, 'reflux_x':xR, 'hD':hD, 'hR':hR,
            'predicted_distillate_flow':sum(draws), 'predicted_reflux_flow':sum(returns),
            'aqueous_draw_flow':draws[0], 'organic_draw_flow':draws[1],
            'aqueous_reflux_flow':returns[0], 'organic_reflux_flow':returns[1]}


def solve_case(case):
    captured = {}
    original_residual, original_sparsity = Column.residual, Column.sparsity
    original_jacobian = Column.local_jacobian
    original_active_set = vlle.solve_vlle_active_set

    def residual(model, vector):
        values = original_residual(model, vector).copy()
        top = model.decode(vector)['stages'][0]
        route = routing(model, top)
        # All condensed material leaves the top as either reflux or product;
        # the existing top balance already uses its equilibrium aggregate.
        # The stage below must instead receive the actual routed reflux.
        for i,c in enumerate(model.components):
            values[model.stage_row_starts[1]+i] += (
                top['L']*(route['reflux_x'][c]-top['aggregate_x'][c])
                /model.component_scales[c])
        aggregate_h = model.stage_properties(0, top)['hL']
        values[model.stage_row_starts[1]+model.nc] += (
            top['L']*(route['hR']-aggregate_h)/model.energy_scale)
        values[-1] = (top['V']-route['predicted_distillate_flow'])/model.flow_scale
        return values

    def sparsity(model):
        matrix = original_sparsity(model).tolil()
        top = model.layouts[0]
        matrix[-1,top.start:top.stop] = 1
        return matrix.tocsr()

    # Retain the production active-set checks at each Jacobian boundary. Only
    # its derivative assembly is replaced; omitting those checks can leave
    # ghost phases and a singular Newton system near nearly pure bottoms.
    tree = ast.parse(textwrap.dedent(inspect.getsource(original_jacobian)))
    function = tree.body[0]
    end = next(i for i,node in enumerate(function.body)
               if isinstance(node,ast.Assign) and ast.unparse(node.targets[0])=='stages')
    function.body = function.body[:end]+[ast.Return(value=ast.Constant(value=None))]
    ast.fix_missing_locations(tree)
    namespace = dict(original_jacobian.__globals__)
    exec(compile(tree,'<VLLE-topology-checks>','exec'),namespace)
    assessment = namespace[function.name]

    def jacobian(model, vector, f0, rel_step):
        assessment(model,vector,f0,rel_step)
        groups = getattr(model,'_probe_colors',None)
        if groups is None:
            groups = model.unit._color_jacobian_columns(model._sparsity)
            model._probe_colors = groups
        matrix,evaluations = model.unit._finite_difference_jacobian(
            model.residual,vector,f0,model._sparsity,groups,rel_step)
        return matrix,evaluations,'probe_colored_finite_difference'

    class Captured(Exception):
        pass

    def active_set(*args, **kwargs):
        captured['solved'] = original_active_set(*args, **kwargs)
        raise Captured()

    with isolated_runtime_caches():
        thermo = create_thermodynamics(['butanol', 'water'], 'NRTL')
        feed = thermo.calculate_state(298.15, 1., FEED_FLOW,
            {'butanol':FEED_BUTANOL, 'water':1-FEED_BUTANOL}, phase='liquid', flash=False)
        params = {'N_stages':20, 'feed_stage':10, 'D_to_F':case['cut'],
                  'P_condenser':1., 'P_drop_per_stage':0., 'condenser_type':'total',
                  'stage_phase_model':'VLLE', 'vlle_seed':'cheap', 'reflux_ratio':1.2,
                  'mesh_tolerance':case.get('tolerance',1e-7),
                  'max_iterations':80, 'max_jacobian_evaluations':80,
                  '_probe_aqueous_draw':case['aqueous_draw'],
                  '_probe_organic_draw':case['organic_draw']}
        unit = RigorousDistillation(case['name'], thermo, params)
        if case.get('warm_from'):
            previous = json.loads(Path(case['warm_from']).read_text())
            states = previous['stages']
            profile = {'T':[s['T'] for s in states],
                       'x':[s['aggregate_x'] for s in states],
                       'x1':[s['x1'] for s in states], 'x2':[s['x2'] for s in states],
                       'beta':[s['beta'] for s in states],
                       'L':[s['L'] for s in states], 'V':[s['V'] for s in states],
                       'Q_cond':previous['condenser_duty_kW']*3600,
                       'Q_reb':previous['reboiler_duty_kW']*3600,
                       'vlle_topology':previous['topology']}
            # This is a numerical starting profile, not an external stream.
            unit._recycle_profile_initial_guess = lambda *args:profile
        start = time.perf_counter()
        native = case.get('native',False)
        with patch.object(Column,'residual',original_residual if native else residual), \
             patch.object(Column,'sparsity',original_sparsity if native else sparsity), \
             patch.object(Column,'local_jacobian',original_jacobian if native else jacobian), \
             patch.object(vlle,'solve_vlle_active_set',active_set):
            try:
                unit.solve({'feed':feed})
            except Captured:
                pass
        elapsed = time.perf_counter()-start
        solved = captured['solved']
        stages = solved.decoded['stages']
        top,bottom = stages[0],stages[-1]
        from types import SimpleNamespace
        route = routing(SimpleNamespace(unit=unit,thermo=thermo,pressures=[1.],components=thermo.components),top)
        D,B = top['V'],bottom['L']
        xD,xB = route['distillate_x'],bottom['aggregate_x']
        hB = solved.stage_properties[-1]['hL']
        balance = {c:feed.F*feed.composition[c]-D*xD[c]-B*xB[c] for c in thermo.components}
        energy = D*route['hD']+B*hB-feed.F*feed.H-solved.decoded['Q_cond']-solved.decoded['Q_reb']
        energy_scale = max(abs(D*route['hD'])+abs(B*hB)+abs(feed.F*feed.H),1.)
        audit = [vlle.three_phase_fugacity_residuals(thermo,s['T'],1.,s['x1'],s['x2'],p['y'],thermo.components)
                 for s,p,active in zip(stages,solved.stage_properties,solved.active) if active]
        fugacity = max((a['overall'] for a in audit),default=0.)
        balanced = (max(abs(v) for v in balance.values())/FEED_FLOW<1e-6
                    and abs(energy)/energy_scale<1e-6)
        valid = bool(solved.active[0] and balanced and fugacity<1e-5
                     and solved.solver['residual_norm']<1e-5)
        return {**case,'status':'ok' if valid else 'audit_failed', 'seconds':elapsed,
                'bottoms_purity':xB['butanol'], 'butanol_recovery':B*xB['butanol']/(FEED_FLOW*FEED_BUTANOL),
                'butanol_loss_kmol_h':D*xD['butanol'], 'distillate_flow':D,'bottoms_flow':B,
                'distillate_x':xD,'bottoms_x':xB,'top_T_C':top['T']-273.15,
                'reflux_flow':top['L'],'reflux_ratio':top['L']/D,'routing':route,
                'condenser_duty_kW':solved.decoded['Q_cond']/3600,
                'reboiler_duty_kW':solved.decoded['Q_reb']/3600,
                'mesh_residual':solved.solver['residual_norm'],'max_log_fugacity_residual':fugacity,
                'component_balance_residuals_kmol_h':balance,'energy_balance_residual_kJ_h':energy,
                'relative_energy_residual':abs(energy)/energy_scale,
                'topology':''.join('L' if a else '.' for a in solved.active),
                'topology_history':solved.topology_history,'work':solved.work,'stages':stages}


def child_case(case, result_path, log_path):
    with Path(log_path).open('x') as log, redirect_stdout(log), redirect_stderr(log):
        try:
            print('CASE:',case,flush=True)
            result = solve_case(case)
            print('RESULT:',{k:v for k,v in result.items() if k not in ('stages','routing')},flush=True)
        except Exception as error:
            traceback.print_exc()
            result = {**case,'status':'failed','error':str(error)}
        write_json(Path(result_path),result)


def run_case(case, output, timeout):
    result_path,log_path = output/(case['name']+'.json'),output/(case['name']+'.log')
    process = multiprocessing.get_context('spawn').Process(target=child_case,
        args=(case,str(result_path),str(log_path)))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(3)
        if process.is_alive():
            process.kill()
            process.join()
        result = {**case,'status':'timeout','timeout_seconds':timeout}
        write_json(result_path,result)
    elif result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {**case,'status':'process_failed','exitcode':process.exitcode}
        write_json(result_path,result)
    return result


def batch(cases, output, workers, timeout, records):
    with ThreadPoolExecutor(max_workers=workers) as pool, (output/'results.jsonl').open('a') as handle:
        pending = {pool.submit(run_case,c,output,timeout):c for c in cases}
        for future in as_completed(pending):
            row = future.result()
            records.append(row)
            handle.write(json.dumps(row)+'\n')
            handle.flush()
            print(row['name'],row['status'],'purity=',row.get('bottoms_purity'),
                  'recovery=',row.get('butanol_recovery'),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--timeout',type=float,default=45.)
    parser.add_argument('--refine-from',type=Path,
                        help='Reuse an existing scan and refine from its successful profiles')
    args = parser.parse_args()
    if not 1<=args.workers<=5 or args.timeout<=0:
        parser.error('Use 1-5 workers and a positive timeout')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'manifest.json',{
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'workers':args.workers,'timeout_seconds':args.timeout,'threads_per_case':1,
        'refine_from':str(args.refine_from) if args.refine_from else None,
        'sources':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                   for name in ('equilibrium_stage_vlle.py','unit_operations_distillation.py',
                                'data/nrtl_binary_interactions_cas.json',
                                str(Path(__file__).relative_to(ROOT)))}})
    if args.refine_from:
        seed_directory = args.refine_from
        baseline = json.loads((seed_directory/'baseline.json').read_text())
        records = [baseline]+[json.loads(line)
            for line in (seed_directory/'results.jsonl').read_text().splitlines()]
    else:
        seed_directory = args.output
        control = {'name':'baseline','cut':.773,'aqueous_draw':1/2.2,'organic_draw':1/2.2,'native':True}
        baseline = run_case(control,args.output,args.timeout)
        records = [baseline]
    write_json(args.output/'baseline.json',baseline)
    if baseline['status']!='ok':
        raise RuntimeError('Baseline failed: '+str(baseline))
    aqueous_butanol = baseline['routing']['aqueous_x']['butanol']
    organic_butanol = baseline['routing']['organic_x']['butanol']
    # With only the aqueous phase exported, the binary invariant fixes xD.
    # Overall butanol balance then determines the cut for any target purity.
    def cut_for(purity,xD):
        return (purity-FEED_BUTANOL)/(purity-xD)
    near_targets = (.99505,.9955,.997,.999)
    cases = []
    for a in (1.,.8,.6,1/2.2,.3):
        for purity in near_targets:
            cases.append({'name':f'a{a:.4f}_pure{purity:.5f}',
                'cut':cut_for(purity,aqueous_butanol),'aqueous_draw':a,'organic_draw':0.,
                'target_purity':purity})
    # Include less selective routing controls; their larger overhead butanol
    # content requires a larger cut to approach high-purity bottoms.
    for k in (.02,.1,.25):
        for cut in (.65,.70,.75):
            cases.append({'name':f'organic{k:.2f}_cut{cut:.2f}',
                          'cut':cut,'aqueous_draw':1.,'organic_draw':k})
    print('BASELINE:',{k:v for k,v in baseline.items() if k not in ('stages','routing')},flush=True)
    print('TOP PHASE BUTANOL:',aqueous_butanol,organic_butanol,
          'PURE-AQUEOUS TARGET CUTS:',[cut_for(p,aqueous_butanol) for p in near_targets],flush=True)
    if not args.refine_from:
        batch(cases,args.output,args.workers,args.timeout,records)
    refinements = []
    for a in (1.,.8,.6,1/2.2,.3):
        seeds = [r for r in records if r['status']=='ok' and r['organic_draw']==0.
                 and abs(r['aqueous_draw']-a)<1e-10]
        if not seeds:
            continue
        seed = max(seeds,key=lambda r:r['butanol_recovery'])
        for purity in (.99505,.9955,baseline['bottoms_purity'],.999):
            refinements.append({'name':f'warm_a{a:.4f}_pure{purity:.8f}',
                'cut':cut_for(purity,aqueous_butanol),'aqueous_draw':a,'organic_draw':0.,
                'target_purity':purity,'warm_from':str(seed_directory/(seed['name']+'.json'))})
    if refinements:
        batch(refinements,args.output,args.workers,args.timeout,records)
    eligible = [r for r in records if r['status']=='ok' and r['bottoms_purity']>TARGET_PURITY]
    if eligible:
        best = max(eligible,key=lambda r:(round(r['butanol_recovery'],10),-r['reboiler_duty_kW']))
        confirm = {k:best[k] for k in ('cut','aqueous_draw','organic_draw')}
        seed_path = args.output/(best['name']+'.json')
        if not seed_path.exists():
            seed_path = seed_directory/(best['name']+'.json')
        confirm.update(name='best_tight_confirmation',tolerance=1e-9,
                       warm_from=str(seed_path))
        verified = run_case(confirm,args.output,args.timeout)
        records.append(verified)
        write_json(args.output/'best.json',verified)
    else:
        best = verified = None
    summary = {'baseline':baseline,'best_scan_case':best,'confirmation':verified,
               'case_count':len(records),'statuses':{s:sum(r['status']==s for r in records)
                    for s in set(r['status'] for r in records)},
               'aqueous_phase_butanol':aqueous_butanol,'organic_phase_butanol':organic_butanol,
               'ideal_cut_at_99_5_purity':cut_for(TARGET_PURITY,aqueous_butanol),
               'ideal_recovery_at_99_5_purity':(1-cut_for(TARGET_PURITY,aqueous_butanol))*TARGET_PURITY/FEED_BUTANOL}
    write_json(args.output/'summary.json',summary)
    print('SUMMARY:',{k:v for k,v in summary.items() if k not in ('baseline','best_scan_case','confirmation')},flush=True)
    if verified:
        print('CONFIRMATION:',{k:v for k,v in verified.items() if k not in ('stages','routing')},flush=True)


if __name__=='__main__':
    main()
