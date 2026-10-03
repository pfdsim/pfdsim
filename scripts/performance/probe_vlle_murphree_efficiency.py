#!/usr/bin/env python3
"""Script-only VLLE efficiency probe, retaining native phase-topology changes.

    set -o pipefail
    python scripts/performance/probe_vlle_murphree_efficiency.py --output /tmp/vlle-efficiency \
        2>&1 | tee /tmp/vlle-efficiency.log > /dev/null

The liquids equilibrate immediately. Interior-tray vapor approaches their
common equilibrium vapor via a scalar Murphree efficiency. Condensers and
reboilers remain equilibrium devices. Vapor feeds enter the driving force as
flow-weighted mixtures with rising vapor. Actual vapor enthalpies enter MESH.
One CPU thread, sequential timing workers, fixed seeds, bounded runtimes, and
incremental logs/results. No production file is modified or new API exposed.
"""

import argparse
import json
from functools import lru_cache
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time
from unittest.mock import patch

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import equilibrium_stage_vlle as vlle
from sparse_jacobian import FixedPatternCSR, SparsePatternBuilder
from tests.cache_isolation import isolated_runtime_caches
from thermodynamics import create_thermodynamics
from unit_operations_distillation import RigorousDistillation
from scripts.performance.probe_vle_murphree_efficiency import (
    logits, softmax, incoming_vapor, vapor_feed_inventory,
)

NativeColumn = vlle.EquationOrientedVLLEColumn
NativeActiveSet = vlle.solve_vlle_active_set
CASES = ('butanol_selective','butanol_subcooled','butanol_copooled','butanol_partial',
         'butanol_vapor_feed','butanol_selective_rk','butanol_partial_vapor_rk','ternary_vlle')
VARIANTS = ('native','E1.0','E0.9','E0.7','E0.5','E0.3','profile')


class EfficiencyColumn(NativeColumn):
    """Script extension of the authoritative native liquid-equilibrium model."""

    def __init__(self,*args,**kwargs):
        self.ready = False
        super().__init__(*args,**kwargs)
        self.comps = self.components
        self.core_size,self.core_rows,self.core_pattern = self.n_vars,self.n_rows,self._sparsity
        self.n_vars += (self.N-2)*(self.nc-1)
        self.n_rows = self.n_vars
        self.efficiencies = np.ones(self.N)
        requested = self.unit.probe_efficiency
        self.efficiencies[1:-1] = (np.linspace(.5,.9,self.N-2) if requested == 'profile' else float(requested))
        self.feed_vapor_flow,self.feed_vapor_components = vapor_feed_inventory(
            self.feed_specs,self.components,self.N)
        # Only within a single fixed-topology solve: the second derivative
        # pass can reuse the exact same local thermodynamic evaluations.
        self.cached_properties = lru_cache(maxsize=8192)(self.compute_properties)
        self.ready = True
        self._sparsity = self.sparsity()
        self._jacobian_pattern = FixedPatternCSR(self._sparsity)

    def y_columns(self,j):
        if not 1 <= j < self.N-1:
            return []
        start = self.core_size+(j-1)*(self.nc-1)
        return list(range(start,start+self.nc-1))

    def thermal_columns(self,j):
        layout = self.layouts[j]
        cols = [layout.temperature,*range(layout.x1.start,layout.x1.stop)]
        if layout.active_vlle:
            cols.extend(range(layout.x2.start,layout.x2.stop))
            cols.append(layout.beta)
        return cols

    def efficiency_row(self,j):
        return self.core_rows+(j-1)*(self.nc-1)

    def decode_stage(self,vector,j):
        state = super().decode_stage(vector,j)
        if self.ready and 1 <= j < self.N-1:
            state['probe_y'] = softmax(vector[self.y_columns(j)],self.components)
        return state

    def pack_initial(self):
        vector = super().pack_initial()
        old_y = getattr(self.profile,'probe_vapor_compositions',None)
        for j in range(1,self.N-1):
            state = self.decode_stage(vector,j)
            y = old_y[j] if old_y is not None else super().stage_properties(j,state)['y']
            vector[self.y_columns(j)] = logits(y,self.components)
        return vector

    def compute_properties(self,j,T,x1_values,x2_values,beta,y_values):
        x1,x2 = dict(zip(self.components,x1_values)),dict(zip(self.components,x2_values))
        state = {'T':T,'x1':x1,'x2':x2,'beta':beta,
                 'aggregate_x':{c:(1-beta)*x1[c]+beta*x2[c] for c in self.components}}
        props = super().stage_properties(j,state)
        props['probe_equilibrium_y'] = dict(props['y'])
        if y_values:
            props['y'] = dict(zip(self.components,y_values))
            # The vapor EOS/enthalpy state belongs to the ACTUAL outgoing gas,
            # not to the hypothetical vapor used by the liquid equilibrium.
            props['hV'] = self.thermo.mixture_enthalpy(props['y'],T,1.,P=self.pressures[j])
        return props

    def stage_properties(self,j,state):
        if not self.ready:
            return super().stage_properties(j,state)
        return self.cached_properties(j,float(state['T']),tuple(state['x1'][c] for c in self.components),
            tuple(state['x2'][c] for c in self.components),float(state['beta']),
            tuple(state['probe_y'][c] for c in self.components) if 'probe_y' in state else ())

    def sparsity(self):
        if not self.ready:
            return super().sparsity()
        pattern = SparsePatternBuilder((self.n_rows,self.n_vars))
        core = self.core_pattern.tocoo()
        for row,col in zip(core.row,core.col):
            pattern.mark(int(row),int(col))
        for j in range(1,self.N-1):
            for col in self.y_columns(j):
                for neighbor in (j-1,j):
                    start = self.stage_row_starts[neighbor]
                    for row in range(start,start+self.nc+1):
                        pattern.mark(row,col)
            for row in range(self.efficiency_row(j),self.efficiency_row(j)+self.nc-1):
                for neighbor in (j,j+1):
                    for col in [*self.thermal_columns(neighbor),*self.y_columns(neighbor)]:
                        pattern.mark(row,col)
                pattern.mark(row,self.layouts[j+1].vapor_flow)
        return pattern.tocsr()

    def efficiency_residuals(self,decoded,props):
        states = decoded['stages']
        values = []
        for j in range(1,self.N-1):
            yin,_ = incoming_vapor(self,j,states[j+1]['V'],props[j+1]['y'])
            E = self.efficiencies[j]
            values.extend(props[j]['y'][c]-yin[c]-E*(props[j]['probe_equilibrium_y'][c]-yin[c])
                          for c in self.components[:-1])
        return values

    def residual(self,vector):
        values = super().residual(vector)
        decoded = self.decode(vector)
        props = [self.stage_properties(j,s) for j,s in enumerate(decoded['stages'])]
        return np.asarray([*values,*self.efficiency_residuals(decoded,props)])

    def profile_from_decoded(self,decoded):
        profile = super().profile_from_decoded(decoded)
        props = [self.stage_properties(j,s) for j,s in enumerate(decoded['stages'])]
        profile.probe_vapor_compositions = [dict(p['y']) for p in props]
        return profile

    def local_jacobian(self,vector,f0,rel_step):
        # Native spec-row offsets refer to the core equations. Its checked
        # matrix pattern is already expanded; restore the offset afterwards.
        full_rows = self.n_rows
        self.n_rows = self.core_rows
        try:
            core,evaluations,_ = super().local_jacobian(vector,f0,rel_step)
        finally:
            self.n_rows = full_rows
        matrix = self._jacobian_pattern.empty()
        coo = core.tocoo()
        for row,col,value in zip(coo.row,coo.col,coo.data):
            matrix.set(int(row),int(col),float(value))
        decoded = self.decode(vector)
        states = decoded['stages']
        props = [self.stage_properties(j,s) for j,s in enumerate(states)]
        for j in range(self.N):
            for col in [*self.thermal_columns(j),*self.y_columns(j)]:
                step = rel_step*max(abs(float(vector[col])),1.)
                trial = vector.copy()
                trial[col] += step
                state = self.decode_stage(trial,j)
                changed = self.stage_properties(j,state)
                evaluations += 1
                dy = {c:(changed['y'][c]-props[j]['y'][c])/step for c in self.components}
                dyeq = {c:(changed['probe_equilibrium_y'][c]-props[j]['probe_equilibrium_y'][c])/step
                        for c in self.components}
                if col in self.y_columns(j):
                    dh = (changed['hV']-props[j]['hV'])/step
                    for neighbor,sign in ((j,-1.),(j-1,1.)):
                        start = self.stage_row_starts[neighbor]
                        for i,c in enumerate(self.components):
                            matrix.add(start+i,col,sign*states[j]['V']*dy[c]/self.component_scales[c])
                        matrix.add(start+self.nc,col,sign*states[j]['V']*dh/self.energy_scale)
                for i,c in enumerate(self.components[:-1]):
                    if 1 <= j < self.N-1:
                        matrix.add(self.efficiency_row(j)+i,col,dy[c]-self.efficiencies[j]*dyeq[c])
                    if 2 <= j < self.N:
                        _,weight = incoming_vapor(self,j-1,states[j]['V'],props[j]['y'])
                        matrix.add(self.efficiency_row(j-1)+i,col,
                                   -(1-self.efficiencies[j-1])*weight*dy[c])
            if 2 <= j < self.N:
                yin,_ = incoming_vapor(self,j-1,states[j]['V'],props[j]['y'])
                total = states[j]['V']+self.feed_vapor_flow[j-1]
                for i,c in enumerate(self.components[:-1]):
                    matrix.add(self.efficiency_row(j-1)+i,self.layouts[j].vapor_flow,
                        -(1-self.efficiencies[j-1])*states[j]['V']*(props[j]['y'][c]-yin[c])/total)
        return matrix.tocsr(),evaluations,'probe_vlle_murphree_local_thermo'


def setup(case,topology_policy,warm_seed,no_projection):
    method = 'NRTL-RK' if case.endswith('_rk') else 'NRTL'
    z = {'butanol':.4,'water':.6}
    if case == 'ternary_vlle':
        method,z = 'UNIFAC',{'ethanol':.35,'water':.25,'benzene':.4}
    thermo = create_thermodynamics(list(z),method)
    vapor = 'vapor' in case
    feed = thermo.calculate_state(430. if vapor else 298.15,1.,100.,z,
                                  phase='vapor' if vapor else 'liquid',flash=False)
    params = {'N_stages':20,'feed_stage':10,'P_condenser':1.,'P_drop_per_stage':0.,
              'stage_phase_model':'VLLE','mesh_tolerance':1e-7,'max_iterations':80,
              'max_jacobian_evaluations':80,'vlle_colored_jacobian_fallback':False}
    if case == 'ternary_vlle':
        params.update(N_stages=16,feed_stage=8,condenser_type='total',reflux_ratio=2.,D_to_F=.4)
    elif 'partial' in case:
        params.update(condenser_type='partial',reflux_ratio=2.,D_to_F=.65)
    elif 'copooled' in case:
        params.update(condenser_type='total',reflux_ratio=1.2,D_to_F=.773,vlle_seed='cheap')
    else:
        params.update(condenser_type='total',D_to_F=.6119275124597854,
                      distillate_liquid1_component='butanol',distillate_liquid1_fraction=0.,
                      distillate_liquid2_fraction=1.)
    if 'subcooled' in case:
        params['condenser_subcooling'] = 5.
    captured = []
    def capture(*args,**kwargs):
        solved = NativeActiveSet(*args,**kwargs)
        captured.append((args,kwargs,solved))
        return solved
    unit = RigorousDistillation('SETUP',thermo,params)
    with patch.object(vlle,'solve_vlle_active_set',capture):
        result = unit.solve({'feed':feed})
    args,kwargs,equilibrium = captured[-1]
    args = list(args)
    kwargs = dict(kwargs)
    if topology_policy:
        params['vlle_topology_policy'] = topology_policy
    if no_projection:
        params['vlle_projection_enabled'] = False
    if warm_seed:
        stages = equilibrium.decoded['stages']
        args[13] = vlle.VLLEProfile(
            T=[s['T'] for s in stages],aggregate_x=[dict(s['aggregate_x']) for s in stages],
            L=[s['L'] for s in stages],V=[s['V'] for s in stages],
            Q_cond=equilibrium.decoded['Q_cond'],Q_reb=equilibrium.decoded['Q_reb'],
            split_data=[(dict(s['x1']),dict(s['x2']),s['beta']) if active else None
                        for s,active in zip(stages,equilibrium.active)],
        )
        kwargs['initial_active'] = list(equilibrium.active)
    args[2] = [dict(f) for f in args[2]]
    for f in args[2]:
        if feed.y is not None:
            f['probe_vapor_y'] = dict(feed.y)
    args[14] = dict(args[14],mesh_tolerance=1e-7,acceptable_mesh_residual=1e-7)
    return thermo,feed,params,args,kwargs,result.performance['initializer']


def audit(unit,feed,solved):
    stages,props = solved.decoded['stages'],solved.stage_properties
    comps = tuple(unit.thermo.components)
    route = solved.top_liquid_routing
    fraction = unit._condenser_vapor_fraction(unit.get_param('condenser_type','total'))
    D,B = stages[0]['V'],stages[-1]['L']
    product = {c:(1-fraction)*route['distillate_x'][c]+fraction*props[0]['y'][c] for c in comps}
    hD = (1-fraction)*route['distillate_h']+fraction*props[0]['hV']
    mass = {c:feed.F*feed.composition[c]-D*product[c]-B*stages[-1]['aggregate_x'][c] for c in comps}
    energy = feed.F*feed.H+solved.decoded['Q_cond']+solved.decoded['Q_reb']-D*hD-B*props[-1]['hL']
    liquid_error,reference_error,eff_error,split_error = 0.,0.,0.,0.
    minimum_distance,minimum_fraction = 2.,1.
    flows,vapor_components = vapor_feed_inventory(unit.probe_feeds,comps,len(stages))
    inlet_model = type('Inlet',(),{'feed_vapor_flow':flows,'feed_vapor_components':vapor_components,'comps':comps})()
    for j,(state,p) in enumerate(zip(stages,props)):
        reference = p.get('probe_equilibrium_y',p['y'])
        if solved.active[j]:
            distance = sum(abs(state['x1'][c]-state['x2'][c]) for c in comps)
            amount = min(state['beta'],1-state['beta'])
            minimum_distance,minimum_fraction = min(minimum_distance,distance),min(minimum_fraction,amount)
            if distance < 1e-3 or amount < 1e-6:
                raise AssertionError(f'Degenerate liquid phases on stage {j+1}: distance={distance}, fraction={amount}')
            has_split,first,second,beta = unit.thermo.liquid_liquid_equilibrium(
                state['aggregate_x'],state['T'],max_iter=100,tol=1e-8)
            if not has_split:
                raise AssertionError(f'Independent stability check finds no liquid split on stage {j+1}')
            direct = max(abs(state['x1'][c]-first[c]) for c in comps)
            direct = max(direct,max(abs(state['x2'][c]-second[c]) for c in comps),abs(state['beta']-beta))
            swapped = max(abs(state['x1'][c]-second[c]) for c in comps)
            swapped = max(swapped,max(abs(state['x2'][c]-first[c]) for c in comps),abs(state['beta']-(1-beta)))
            split_error = max(split_error,min(direct,swapped))
            errors = vlle.three_phase_fugacity_residuals(unit.thermo,state['T'],1.,state['x1'],state['x2'],
                reference,comps,include_vapor=not (j == 0 and unit.get_param('condenser_subcooling') is not None))
            liquid_error = max(liquid_error,errors['liquid_liquid'])
            reference_error = max(reference_error,errors['vapor_liquid'])
        if 1 <= j < len(stages)-1:
            E = (np.linspace(.5,.9,len(stages)-2)[j-1] if unit.probe_efficiency == 'profile'
                 else float(unit.probe_efficiency))
            yin,_ = incoming_vapor(inlet_model,j,stages[j+1]['V'],props[j+1]['y'])
            eff_error = max(eff_error,max(abs(p['y'][c]-yin[c]-E*(reference[c]-yin[c])) for c in comps))
    if (max(abs(v) for v in mass.values()) > 1e-4 or liquid_error > 1e-6
            or reference_error > 1e-6 or eff_error > 1e-6 or split_error > 1e-4):
        raise AssertionError(f'Physical audit failed: mass={mass}, LL={liquid_error}, reference={reference_error}, E={eff_error}')
    report = {'component_balance_kmol_h':mass,'energy_balance_kJ_h':float(energy),
            'max_independent_liquid_split_error':float(split_error),
            'minimum_liquid_phase_distance':float(minimum_distance),
            'minimum_liquid_phase_fraction':float(minimum_fraction),
            'max_liquid_fugacity_residual':float(liquid_error),'max_reference_vapor_fugacity_residual':float(reference_error),
            'max_murphree_residual':float(eff_error),'distillate_x':product,
            'bottoms_x':stages[-1]['aggregate_x'],
            'component_recoveries_bottoms':{c:float(B*stages[-1]['aggregate_x'][c]/(feed.F*feed.composition[c])) for c in comps},
            'reflux_ratio':float(stages[0]['L']/D),'top_C':float(stages[0]['T']-273.15),
            'topology':''.join('L' if x else '.' for x in solved.active),
            'topology_history':solved.topology_history,'work':solved.work,
            'aggregate_liquids':[s['aggregate_x'] for s in stages]}
    if 'butanol' in comps:
        report['butanol_recovery'] = report['component_recoveries_bottoms']['butanol']
    scale = max(abs(feed.F*feed.H),feed.F*50000.,1.)
    report['relative_energy_balance'] = float(abs(energy)/scale)
    if report['relative_energy_balance'] > 1e-6:
        raise AssertionError(f'Energy audit failed: {energy}')
    return report


def validate_jacobian(unit,args,kwargs,efficiency):
    unit.probe_efficiency = efficiency
    stability = vlle.VLLEStabilityCache(unit.thermo,args[3],1e-7)
    profile = args[13]
    active = [vlle._split_is_active(*stability.split(T,x),args[3],1e-5,1e-3)
              for T,x in zip(profile.T,profile.aggregate_x)]
    model = EfficiencyColumn(*args[:13],active,profile,stability,1e-6,1e-3,0.,args[14])
    vector = model.pack_initial()
    f0 = model.residual(vector)
    analytic = model.local_jacobian(vector,f0,1e-6)[0].toarray()
    numeric,_ = unit._finite_difference_jacobian(model.residual,vector,f0,model._sparsity,
                                                unit._color_jacobian_columns(model._sparsity),1e-6)
    error = float(np.max(abs(analytic-numeric.toarray())/np.maximum(1.,abs(numeric.toarray()))))
    if error > 1e-5:
        raise AssertionError(f'Jacobian error {error}')
    return error


def worker(case,output,repeats,topology_policy,warm_seed,no_projection,variants):
    with isolated_runtime_caches():
        thermo,feed,params,args,kwargs,initializer = setup(case,topology_policy,warm_seed,no_projection)
        check_unit = RigorousDistillation('CHECK',thermo,params)
        args[0] = check_unit
        jacobian_error = max(validate_jacobian(check_unit,args,kwargs,E) for E in (1.,'profile'))
        rows = []
        native = None
        with (output/f'{case}.jsonl').open('x') as handle:
            for repeat in range(repeats+1):
                for variant in (variants if repeat % 2 == 0 else variants[::-1]):
                    unit = RigorousDistillation('PROBE',thermo,params)
                    unit.probe_efficiency = 1. if variant == 'native' else ('profile' if variant == 'profile' else float(variant[1:]))
                    unit.probe_feeds = args[2]
                    call_args = [unit,*args[1:]]
                    start = time.perf_counter()
                    try:
                        with patch.object(vlle,'EquationOrientedVLLEColumn',NativeColumn if variant == 'native' else EfficiencyColumn):
                            solved = NativeActiveSet(*call_args,**kwargs)
                        elapsed = time.perf_counter()-start
                        row = {'case':case,'variant':variant,'repeat':repeat,'warmup':repeat==0,
                               'success':True,'seconds':elapsed,'residual':float(solved.solver['residual_norm']),
                               'unknowns':len(solved.solver['x'])}
                        row.update(audit(unit,feed,solved))
                        if variant == 'native':
                            native = row
                        elif variant == 'E1.0' and native is not None:
                            row['E1_native_max_x_difference'] = max(abs(a[c]-b[c]) for a,b in zip(
                                row['aggregate_liquids'],native['aggregate_liquids']) for c in thermo.components)
                    except (RuntimeError,AssertionError) as error:
                        row = {'case':case,'variant':variant,'repeat':repeat,'warmup':repeat==0,
                               'success':False,'seconds':time.perf_counter()-start,'error':str(error)}
                    handle.write(json.dumps(row)+'\n')
                    handle.flush()
                    rows.append(row)
                    print(case,variant,repeat,'ok=',row['success'],'s=',row['seconds'],
                          'res=',row.get('residual'),row.get('error',''),flush=True)
        native_median = median(r['seconds'] for r in rows if r['variant']=='native' and not r['warmup'])
        summary = {'case':case,'params':params,'initializer':initializer,'jacobian_error':jacobian_error,'variants':{}}
        for variant in variants:
            samples = [r for r in rows if r['variant']==variant and not r['warmup']]
            summary['variants'][variant] = {'successes':sum(r['success'] for r in samples),
                'repeats':repeats,'median_seconds':median(r['seconds'] for r in samples),
                'relative_to_native':median(r['seconds'] for r in samples)/native_median,
                'representative':samples[0]}
        (output/f'{case}.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('SUMMARY',case,{v:(r['successes'],r['median_seconds']) for v,r in summary['variants'].items()},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--timeout',type=float,default=90.)
    parser.add_argument('--worker',choices=CASES)
    parser.add_argument('--topology-policy',choices=('adaptive','residual_gate'))
    parser.add_argument('--warm-seed',action='store_true',help='Recovery probe: start from the native converged equilibrium profile')
    parser.add_argument('--no-projection',action='store_true',help='Disable the existing projected phase-contraction heuristic for diagnosis')
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    args = parser.parse_args()
    if 'native' not in args.variants:
        parser.error('Include native for a comparable timing baseline')
    if args.worker:
        worker(args.worker,args.output,args.repeats,args.topology_policy,args.warm_seed,args.no_projection,args.variants)
        return
    if args.repeats < 1 or args.timeout <= 0:
        parser.error('repeats/timeout must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'manifest.json').write_text(json.dumps({'cases':args.cases,'variants':args.variants,'repeats':args.repeats,
        'threads':1,'timeout':args.timeout,'seed':'no RNG',
        'topology_policy':args.topology_policy or 'native default',
        'projection_enabled':not args.no_projection,
        'timing':'active-set coupled solve, excludes property hydration and initializer construction',
        'initialization':('native converged equilibrium profile' if args.warm_seed
                          else 'same native initializer profile, not a converged-state continuation')},indent=2)+'\n')
    statuses = {}
    for case in args.cases:
        command = [sys.executable,str(Path(__file__).resolve()),'--worker',case,
                   '--output',str(args.output),'--repeats',str(args.repeats),'--variants',*args.variants]
        if args.topology_policy:
            command.extend(['--topology-policy',args.topology_policy])
        if args.warm_seed:
            command.append('--warm-seed')
        if args.no_projection:
            command.append('--no-projection')
        with (args.output/f'{case}.log').open('x') as handle:
            try:
                run = subprocess.run(command,
                    stdout=handle,stderr=subprocess.STDOUT,timeout=args.timeout,check=False)
                status = {'returncode':run.returncode}
            except subprocess.TimeoutExpired:
                status = {'status':'timeout'}
        statuses[case] = status
        (args.output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
        print('CASE',case,status,flush=True)
    summary = {c:json.loads((args.output/f'{c}.json').read_text()) for c in args.cases if (args.output/f'{c}.json').exists()}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('RESULTS',args.output,flush=True)
    if any(s.get('returncode') != 0 for s in statuses.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
