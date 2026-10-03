#!/usr/bin/env python3
"""Script-only sparse VLE Murphree experiment; does not modify production.

    set -o pipefail
    python scripts/performance/probe_vle_murphree_efficiency.py --output /tmp/murphree \
        2>&1 | tee /tmp/murphree.log > /dev/null

One thread, sequential timing workers, bounded worker runtime, fresh generic
profiles, one untimed warm-up per variant, alternating variant order, no RNG.
Imports, property hydration, initialization and audits are excluded from solve
timings. A common vapor efficiency applies to every component on an interior
tray; condensers/reboilers remain equilibrium boundaries. Single-liquid VLE,
partial condensers, fully vaporized feeds, and EOS methods can be probed.
There is no VLLE implementation here.
"""

import argparse
import json
import math
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

import numpy as np
from sparse_jacobian import FixedPatternCSR, SparsePatternBuilder
from tests.cache_isolation import isolated_runtime_caches
from thermodynamics import create_thermodynamics
from unit_operations_distillation import RigorousDistillation


CASES = ('methanol_water_20','methanol_water_40','ternary_20','ethanol_water_20',
         'subcooled_methanol_water_20','partial_methanol_water_20',
         'vapor_methanol_water_20','partial_vapor_methanol_water_20',
         'gamma_phi_methanol_water_20','eos_partial_vapor_20')


def vapor_feed_inventory(feeds, components, stages):
    flow = np.zeros(stages)
    components_in = [dict.fromkeys(components,0.) for _ in range(stages)]
    for feed in feeds:
        amount = feed['F']*feed['vapor_fraction']
        if amount <= 0:
            continue
        y = feed.get('probe_vapor_y')
        if y is None:
            if feed['vapor_fraction'] < 1.-1e-12:
                raise ValueError('Two-phase feeds require their actual vapor composition')
            y = feed['z']
        stage = int(feed['stage'])
        flow[stage] += amount
        for c in components:
            components_in[stage][c] += amount*y.get(c,0.)
    return flow,components_in


def incoming_vapor(model, stage, below_flow, below_y):
    total = below_flow+model.feed_vapor_flow[stage]
    return ({c:(below_flow*below_y[c]+model.feed_vapor_components[stage][c])/total
             for c in model.comps},below_flow/total)


def softmax(logits, components):
    values = np.asarray([*logits,0.],dtype=float)
    values = np.exp(values-values.max())
    values /= values.sum()
    return dict(zip(components,map(float,values)))


def logits(x, components):
    return [math.log(max(x[c],1e-14)/max(x[components[-1]],1e-14)) for c in components[:-1]]


class MurphreeProbe:
    """Augment the native VLE state with interior-tray vapor compositions.

    Uses native stage thermodynamics, temperature boundaries and sparse Newton.
    The local Jacobian differentiates thermodynamics only on the changed stage;
    flow/heat derivatives and neighbor connections are assembled analytically.
    """

    def __init__(self, unit, base, feeds, components, N, RR, pressures, D, energy_scale, scales, flow_scale):
        self.unit,self.base,self.feeds = unit,base,feeds
        self.comps,self.N,self.nc = components,N,len(components)
        self.RR,self.pressures,self.D = RR,pressures,D
        self.condenser_vapor_fraction = unit._condenser_vapor_fraction(unit.get_param('condenser_type','total'))
        self.energy_scale,self.scales,self.flow_scale = energy_scale,scales,flow_scale
        self.core_size = base['sparsity'].shape[1]
        self.nvars = self.core_size+(N-2)*(self.nc-1)
        self.core_rows = N*(self.nc+2)+2
        self.T_min,self.T_max = unit._temperature_bounds(components)
        self.efficiencies = np.ones(N)
        self.L_start = N*self.nc
        self.V_start = self.L_start+N
        self.Q_start = self.V_start+N
        self.pattern = self.sparsity()
        self.fixed = FixedPatternCSR(self.pattern)
        self.feed_components = np.zeros((N,self.nc))
        self.feed_energy = np.zeros(N)
        self.feed_vapor_flow,self.feed_vapor_components = vapor_feed_inventory(feeds,components,N)
        for f in feeds:
            stage = int(f['stage'])
            self.feed_components[stage] += f['F']*np.asarray([f['z'].get(c,0.) for c in components])
            self.feed_energy[stage] += f['F']*f['H']

    def y_columns(self, stage):
        if not 1 <= stage < self.N-1:
            return []
        start = self.core_size+(stage-1)*(self.nc-1)
        return list(range(start,start+self.nc-1))

    def local_columns(self, stage):
        start = self.N+stage*(self.nc-1)
        return [stage,*range(start,start+self.nc-1),*self.y_columns(stage)]

    def stage_columns(self, stage):
        return [*self.local_columns(stage),self.L_start+stage,self.V_start+stage]

    def efficiency_row(self, stage):
        return self.core_rows+(stage-1)*(self.nc-1)

    def sparsity(self):
        pattern = SparsePatternBuilder((self.nvars,self.nvars))
        for stage in range(self.N):
            start = stage*(self.nc+2)
            for row in range(start,start+self.nc+1):
                for neighbor in range(max(0,stage-1),min(self.N,stage+2)):
                    for col in self.stage_columns(neighbor):
                        pattern.mark(row,col)
            for col in self.local_columns(stage):
                pattern.mark(start+self.nc+1,col)
            if stage == 0:
                pattern.mark(start+self.nc,self.Q_start)
            elif stage == self.N-1:
                pattern.mark(start+self.nc,self.Q_start+1)
            if 1 <= stage < self.N-1:
                for row in range(self.efficiency_row(stage),self.efficiency_row(stage)+self.nc-1):
                    pattern.mark(row,self.V_start+stage+1)
                    for neighbor in (stage,stage+1):
                        for col in self.local_columns(neighbor):
                            pattern.mark(row,col)
        pattern.mark(self.core_rows-2,self.V_start)
        pattern.mark(self.core_rows-1,self.L_start)
        pattern.mark(self.core_rows-1,self.V_start)
        return pattern.tocsr()

    def pack(self, core):
        state = self.base['decode'](core)
        values = list(core)
        for stage in range(1,self.N-1):
            eq = self.base['stage_properties'](stage,state['T'][stage],state['x'][stage])
            values.extend(logits(eq['y'],self.comps))
        return np.asarray(values)

    def decode(self, vector):
        state = self.base['decode'](vector[:self.core_size])
        state['actual_y'] = [None]*self.N
        for stage in range(1,self.N-1):
            state['actual_y'][stage] = softmax(vector[self.y_columns(stage)],self.comps)
        return state

    def properties(self, stage, state, equilibrium=None, actual_hV=None):
        T,x = state['T'][stage],state['x'][stage]
        eq = (self.base['stage_properties'](stage,T,x) if equilibrium is None else equilibrium)
        y = state['actual_y'][stage] if 1 <= stage < self.N-1 else eq['y']
        hV = (actual_hV if actual_hV is not None else
              self.unit.thermo.mixture_enthalpy(y,float(T),1.,P=self.pressures[stage])
              if 1 <= stage < self.N-1 else eq['hV'])
        return {'equilibrium':eq,'y':y,'hV':hV,'hL':eq['hL'],'bubble':eq['bubble']}

    def residual(self, vector):
        s = self.decode(vector)
        props = [self.properties(j,s) for j in range(self.N)]
        return self.residual_from_state(s,props)

    def residual_from_state(self, s, props):
        L,V,D = s['L'],s['V'],s['V'][0]
        values = np.zeros(self.nvars)
        for j in range(self.N):
            row = j*(self.nc+2)
            for i,c in enumerate(self.comps):
                incoming = self.feed_components[j,i]
                if j > 0:
                    incoming += L[j-1]*s['x'][j-1][c]
                if j < self.N-1:
                    incoming += V[j+1]*props[j+1]['y'][c]
                outgoing = ((L[0]+(1-self.condenser_vapor_fraction)*D)*s['x'][0][c]
                            +self.condenser_vapor_fraction*D*props[0]['y'][c] if j == 0
                            else L[j]*s['x'][j][c]+V[j]*props[j]['y'][c])
                values[row+i] = (incoming-outgoing)/self.scales[c]
            incoming_h = self.feed_energy[j]
            if j > 0:
                incoming_h += L[j-1]*props[j-1]['hL']
            if j < self.N-1:
                incoming_h += V[j+1]*props[j+1]['hV']
            if j == 0:
                incoming_h += s['Q_cond']
                outgoing_h = ((L[0]+(1-self.condenser_vapor_fraction)*D)*props[0]['hL']
                              +self.condenser_vapor_fraction*D*props[0]['hV'])
            else:
                if j == self.N-1:
                    incoming_h += s['Q_reb']
                outgoing_h = L[j]*props[j]['hL']+V[j]*props[j]['hV']
            values[row+self.nc] = (incoming_h-outgoing_h)/self.energy_scale
            values[row+self.nc+1] = props[j]['bubble']
            if 1 <= j < self.N-1:
                E = self.efficiencies[j]
                yin,_ = incoming_vapor(self,j,V[j+1],props[j+1]['y'])
                for i,c in enumerate(self.comps[:-1]):
                    values[self.efficiency_row(j)+i] = props[j]['y'][c]-yin[c]-E*(props[j]['equilibrium']['y'][c]-yin[c])
        values[self.core_rows-2] = (D-self.D)/self.flow_scale
        values[self.core_rows-1] = (L[0]-self.RR*D)/self.flow_scale
        return values

    def jacobian(self, vector, _f0, step_size):
        s = self.decode(vector)
        props = [self.properties(j,s) for j in range(self.N)]
        L,V,D = s['L'],s['V'],s['V'][0]
        matrix = self.fixed.empty()
        add = matrix.add
        evaluations = 0
        for j in range(self.N):
            row = j*(self.nc+2)
            for col in self.local_columns(j):
                step = step_size*max(abs(float(vector[col])),1.)
                trial = vector.copy()
                trial[col] += step
                # Decode only the perturbed stage, avoiding an O(N^2) pass
                # through every composition for a local-thermo Jacobian.
                T,x,y = s['T'][j],s['x'][j],s['actual_y'][j]
                xstart = self.N+j*(self.nc-1)
                if col == j:
                    theta = min(max(float(trial[j]),-60.),60.)
                    T = self.T_min+(self.T_max-self.T_min)/(1.+math.exp(-theta))
                elif col in self.y_columns(j):
                    y = softmax(trial[self.y_columns(j)],self.comps)
                else:
                    x = softmax(trial[xstart:xstart+self.nc-1],self.comps)
                changed_state = {'T':{j:T},'x':{j:x},'actual_y':{j:y}}
                changed = self.properties(j,changed_state,
                    equilibrium=props[j]['equilibrium'] if col in self.y_columns(j) else None,
                    actual_hV=(props[j]['hV'] if 1 <= j < self.N-1
                               and xstart <= col < xstart+self.nc-1 else None))
                evaluations += 1
                dx = {c:(changed_state['x'][j][c]-s['x'][j][c])/step for c in self.comps}
                dy = {c:(changed['y'][c]-props[j]['y'][c])/step for c in self.comps}
                dyeq = {c:(changed['equilibrium']['y'][c]-props[j]['equilibrium']['y'][c])/step
                        for c in self.comps}
                dhL,dhV = (changed['hL']-props[j]['hL'])/step,(changed['hV']-props[j]['hV'])/step
                liquid,vapor = ((L[0]+(1-self.condenser_vapor_fraction)*D,
                                 self.condenser_vapor_fraction*D) if j == 0 else (L[j],V[j]))
                for i,c in enumerate(self.comps):
                    add(row+i,col,-(liquid*dx[c]+vapor*dy[c])/self.scales[c])
                    if j > 0:
                        add((j-1)*(self.nc+2)+i,col,V[j]*dy[c]/self.scales[c])
                    if j < self.N-1:
                        add((j+1)*(self.nc+2)+i,col,L[j]*dx[c]/self.scales[c])
                add(row+self.nc,col,-(liquid*dhL+vapor*dhV)/self.energy_scale)
                if j > 0:
                    add((j-1)*(self.nc+2)+self.nc,col,V[j]*dhV/self.energy_scale)
                if j < self.N-1:
                    add((j+1)*(self.nc+2)+self.nc,col,L[j]*dhL/self.energy_scale)
                add(row+self.nc+1,col,(changed['bubble']-props[j]['bubble'])/step)
                for i,c in enumerate(self.comps[:-1]):
                    if 1 <= j < self.N-1:
                        add(self.efficiency_row(j)+i,col,dy[c]-self.efficiencies[j]*dyeq[c])
                    if 2 <= j < self.N:
                        _,weight = incoming_vapor(self,j-1,V[j],props[j]['y'])
                        add(self.efficiency_row(j-1)+i,col,-(1-self.efficiencies[j-1])*weight*dy[c])
            for col,flow,is_liquid in ((self.L_start+j,L[j],True),(self.V_start+j,V[j],False)):
                comp = s['x'][j] if is_liquid else props[j]['y']
                h = props[j]['hL'] if is_liquid else props[j]['hV']
                if not is_liquid and j == 0:
                    fraction = self.condenser_vapor_fraction
                    comp = {c:(1-fraction)*s['x'][0][c]+fraction*props[0]['y'][c] for c in self.comps}
                    h = (1-fraction)*props[0]['hL']+fraction*props[0]['hV']
                for i,c in enumerate(self.comps):
                    add(row+i,col,-flow*comp[c]/self.scales[c])
                    neighbor = j+1 if is_liquid else j-1
                    if 0 <= neighbor < self.N:
                        add(neighbor*(self.nc+2)+i,col,flow*comp[c]/self.scales[c])
                add(row+self.nc,col,-flow*h/self.energy_scale)
                neighbor = j+1 if is_liquid else j-1
                if 0 <= neighbor < self.N:
                    add(neighbor*(self.nc+2)+self.nc,col,flow*h/self.energy_scale)
                if not is_liquid and 2 <= j < self.N:
                    yin,_ = incoming_vapor(self,j-1,V[j],props[j]['y'])
                    total = V[j]+self.feed_vapor_flow[j-1]
                    for i,c in enumerate(self.comps[:-1]):
                        add(self.efficiency_row(j-1)+i,col,
                            -(1-self.efficiencies[j-1])*V[j]*(props[j]['y'][c]-yin[c])/total)
        add(self.core_rows-2,self.V_start,D/self.flow_scale)
        add(self.core_rows-1,self.L_start,L[0]/self.flow_scale)
        add(self.core_rows-1,self.V_start,-self.RR*D/self.flow_scale)
        add(self.nc,self.Q_start,1.)
        add((self.N-1)*(self.nc+2)+self.nc,self.Q_start+1,1.)
        return matrix.tocsr(),evaluations,'probe_murphree_local_thermo'

    def audit(self, solution):
        s = self.decode(solution['x'])
        p = [self.properties(j,s) for j in range(self.N)]
        maximum = 0.
        for j in range(1,self.N-1):
            yin,_ = incoming_vapor(self,j,s['V'][j+1],p[j+1]['y'])
            maximum = max(maximum,max(abs(p[j]['y'][c]-yin[c]
                -self.efficiencies[j]*(p[j]['equilibrium']['y'][c]-yin[c])) for c in self.comps))
        fraction = self.condenser_vapor_fraction
        product_x = {c:(1-fraction)*s['x'][0][c]+fraction*p[0]['y'][c] for c in self.comps}
        product_h = (1-fraction)*p[0]['hL']+fraction*p[0]['hV']
        balances = {c:float(sum(f['F']*f['z'][c] for f in self.feeds)
                    -s['V'][0]*product_x[c]-s['L'][-1]*s['x'][-1][c]) for c in self.comps}
        energy = (sum(f['F']*f['H'] for f in self.feeds)+s['Q_cond']+s['Q_reb']
                  -s['V'][0]*product_h-s['L'][-1]*p[-1]['hL'])
        return {'max_murphree_residual':float(maximum),
                'component_balance_kmol_h':balances,'energy_balance_kJ_h':float(energy),
                'relative_energy_balance':float(abs(energy)/self.energy_scale),
                'distillate_x':product_x,'bottoms_x':s['x'][-1],
                'top_C':float(s['T'][0]-273.15),'bottom_C':float(s['T'][-1]-273.15),
                'condenser_kW':float(s['Q_cond']/3600.),'reboiler_kW':float(s['Q_reb']/3600.),
                'temperatures_K':list(map(float,s['T'])),
                'liquid_compositions':s['x'],'vapor_compositions':[v['y'] for v in p]}


def prepare(case):
    N = 40 if case == 'methanol_water_40' else 20
    if case == 'ternary_20':
        method,z,T,RR,cut = 'IDEAL',{'methanol':.2,'ethanol':.3,'water':.5},360.,2.,.4
    elif case == 'ethanol_water_20':
        method,z,T,RR,cut = 'UNIFAC',{'ethanol':.1,'water':.9},353.15,4.,.11
    else:
        method,z,T,RR,cut = 'NRTL',{'methanol':.4,'water':.6},298.15,2.,.35
    if case == 'gamma_phi_methanol_water_20':
        method = 'NRTL-RK'
    if case == 'eos_partial_vapor_20':
        method,z,T,RR,cut = 'PR',{'benzene':.5,'toluene':.5},425.,2.,.45
    vapor_feed = case.startswith('vapor_') or 'partial_vapor' in case
    if vapor_feed and method != 'PR':
        T,cut = 400.,.45
    condenser = 'partial' if 'partial' in case else 'total'
    fraction = 1. if condenser == 'partial' else 0.
    t = create_thermodynamics(list(z),method)
    f = t.calculate_state(T,1.,100.,z,phase='vapor' if vapor_feed else 'liquid',flash=False)
    params = {'N_stages':N,'feed_stage':N//2,'reflux_ratio':RR,'D_to_F':cut,
              'P_condenser':1.,'P_drop_per_stage':0.,'initializer':'estimate','condenser_type':condenser}
    if case.startswith('subcooled'):
        params['condenser_subcooling'] = 5.
    u = RigorousDistillation('PROBE',t,params)
    inlet,feeds = u._aggregate_feeds({'feed':f},N,N//2)
    comps = u._component_order(inlet)
    Tmin,Tmax = u._temperature_bounds(comps)
    q = u._feed_thermal_condition(inlet,z,1.,Tmin,Tmax)
    energy_scale = max(abs(f.F*f.H),f.F*50000.,1.)
    scales = {c:max(f.F*z[c],f.F*1e-4) for c in comps}
    spec = {'kind':'molar','value':f.F*cut}
    initial = u._initial_guess(inlet,comps,z,N,N//2,RR,q,[1.]*N,condenser,fraction,spec,[],Tmin,Tmax)
    base = u._build_mesh_model(inlet,feeds,comps,z,N,N//2-1,RR,[1.]*N,condenser,fraction,[],spec,
                               f.F,energy_scale,scales,Tmin,Tmax)
    initial['T'][0] = base['condenser_boundary'].seed_temperature(initial['x'][0],initial['T'][0])
    core = u._pack_variables(initial['T'],initial['x'],initial['L'],initial['V'],
                             initial['Q_cond'],initial['Q_reb'],comps,Tmin,Tmax,energy_scale)
    probe = MurphreeProbe(u,base,feeds,comps,N,RR,[1.]*N,f.F*cut,energy_scale,scales,f.F)
    return u,base,core,probe,params,method


def worker(case, output, repeats):
    options = {'mesh_tolerance':1e-7,'acceptable_mesh_residual':1e-7,'max_iterations':80,
               'max_jacobian_evaluations':80,'line_search_steps':16,'finite_difference_rel_step':1e-6}
    with isolated_runtime_caches():
        u,base,core,probe,params,method = prepare(case)
        seed = probe.pack(core)
        f0 = probe.residual(seed)
        analytic = probe.jacobian(seed,f0,1e-6)[0].toarray()
        numeric,_ = u._finite_difference_jacobian(probe.residual,seed,f0,probe.pattern,
                         u._color_jacobian_columns(probe.pattern),1e-6)
        error = float(np.max(abs(analytic-numeric.toarray())/np.maximum(1.,abs(numeric.toarray()))))
        if error > 1e-5:
            raise AssertionError(f'Jacobian validation failed: {error}')
        # Also check non-unit efficiencies, including the incoming-vapor terms.
        probe.efficiencies[1:-1] = np.linspace(.5,.9,probe.N-2)
        f0 = probe.residual(seed)
        analytic = probe.jacobian(seed,f0,1e-6)[0].toarray()
        numeric,_ = u._finite_difference_jacobian(probe.residual,seed,f0,probe.pattern,
                         u._color_jacobian_columns(probe.pattern),1e-6)
        error = max(error,float(np.max(abs(analytic-numeric.toarray())/np.maximum(1.,abs(numeric.toarray())))))
        if error > 1e-5:
            raise AssertionError(f'Non-unit-efficiency Jacobian validation failed: {error}')
        variants = ['native','E1.0','E0.9','E0.7','E0.5','E0.3','profile']
        records = []
        native_state = None
        with (output/f'{case}.jsonl').open('x') as handle:
            for repeat in range(repeats+1):
                order = variants if repeat % 2 == 0 else variants[::-1]
                for variant in order:
                    if variant == 'native':
                        residual,pattern,jacobian,x0 = base['residual'],base['sparsity'],base['jacobian'],core
                    else:
                        probe.efficiencies[1:-1] = (np.linspace(.5,.9,probe.N-2) if variant == 'profile'
                                                   else float(variant[1:]))
                        residual,pattern,jacobian,x0 = probe.residual,probe.pattern,probe.jacobian,seed
                    start = time.perf_counter()
                    solved = u._sparse_newton_solve(residual,pattern,x0,options,jacobian=jacobian)
                    seconds = time.perf_counter()-start
                    row = {'case':case,'variant':variant,'repeat':repeat,'warmup':repeat==0,
                           'seconds':seconds,'success':bool(solved['success']),
                           'residual':float(solved['residual_norm']),'iterations':int(solved['iterations']),
                           'function_evaluations':int(solved['function_evaluations']),
                           'jacobian_evaluations':int(solved['jacobian_evaluations']),
                           'message':solved['message'],'unknowns':len(x0)}
                    if variant == 'native':
                        state = base['decode'](solved['x'])
                        native_state = state if solved['success'] else None
                        ptop = base['stage_properties'](0,state['T'][0],state['x'][0])
                        row['distillate_x'] = {c:(1-probe.condenser_vapor_fraction)*state['x'][0][c]
                            +probe.condenser_vapor_fraction*ptop['y'][c] for c in probe.comps}
                        row['bottoms_x'] = state['x'][-1]
                    else:
                        row.update(probe.audit(solved))
                        if solved['success']:
                            if (row['max_murphree_residual'] > 1e-6
                                    or max(abs(v) for v in row['component_balance_kmol_h'].values()) > 1e-4
                                    or row['relative_energy_balance'] > 1e-6):
                                raise AssertionError('Converged state failed independent balance/efficiency audits')
                        if variant == 'E1.0' and native_state is not None and solved['success']:
                            s = probe.decode(solved['x'])
                            row['E1_native_max_x_difference'] = max(abs(s['x'][j][c]-native_state['x'][j][c])
                                for j in range(probe.N) for c in probe.comps)
                            if row['E1_native_max_x_difference'] > 1e-5:
                                raise AssertionError('Unit-efficiency solution disagrees with native VLE')
                    handle.write(json.dumps(row)+'\n')
                    handle.flush()
                    records.append(row)
                    print(case,variant,repeat,'ok=',row['success'],'s=',seconds,'res=',row['residual'],flush=True)
        summary = {'case':case,'method':method,'params':params,'jacobian_error':error,'variants':{}}
        native_median = median(r['seconds'] for r in records if r['variant']=='native' and not r['warmup'])
        for variant in variants:
            rows = [r for r in records if r['variant']==variant and not r['warmup']]
            summary['variants'][variant] = {
                'successful_repeats':sum(r['success'] for r in rows),'repeats':repeats,
                'median_seconds':median(r['seconds'] for r in rows),
                'relative_to_native':median(r['seconds'] for r in rows)/native_median,
                'iterations':[r['iterations'] for r in rows],
                'maximum_residual':max(r['residual'] for r in rows),
                'representative':rows[0],
            }
        (output/f'{case}.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('SUMMARY',case,{v:(r['successful_repeats'],r['median_seconds'])
              for v,r in summary['variants'].items()},flush=True)


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
    manifest = {'python':sys.version,'cases':args.cases,'repeats':args.repeats,
                'worker_timeout_seconds':args.timeout,'threads':1,
                'seed':'deterministic; no RNG','timing':'raw sparse MESH solve; excludes setup and warm-up',
                'initialization':'same fresh generic equilibrium profile for every solve',
                'options':'one scalar per tray; total/partial condensers; liquid/vapor feeds; no side draws'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    statuses = {}
    for case in args.cases:
        command = [sys.executable,str(Path(__file__).resolve()),'--worker',case,
                   '--output',str(args.output),'--repeats',str(args.repeats)]
        # Workers run sequentially so concurrent cases do not distort timings.
        # Complete stdout/stderr survive worker errors/timeouts in this log.
        with (args.output/f'{case}.log').open('x') as handle:
            try:
                result = subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,
                                        timeout=args.timeout,check=False)
                status = {'returncode':result.returncode}
            except subprocess.TimeoutExpired:
                status = {'status':'timeout'}
        statuses[case] = status
        (args.output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
        print('CASE',case,status,flush=True)
    summary = {case:json.loads((args.output/f'{case}.json').read_text())
               for case in args.cases if (args.output/f'{case}.json').exists()}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('RESULTS',args.output,flush=True)
    if any(status.get('returncode') != 0 for status in statuses.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
