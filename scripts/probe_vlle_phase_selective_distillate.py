#!/usr/bin/env python3
"""Script-only VLLE overhead routing experiment for the failed decanter test.

Use the same 16-stage UNIFAC column, withdrawing all water-rich top liquid as
8 kmol/h distillate and returning the benzene-rich liquid after a 2% purge.
The equilibrium/stability/active-set implementation stays authoritative.
Only top component/energy routing and the reflux specification are patched.
Both liquids remain co-routed on the other stages. A colored finite-difference
Jacobian differentiates the patched balances rather than the old analytic one.
The unmodified production output adapter is bypassed; results and balances are
assembled from the actual phase-specific solution.

From the repository root, preserving both output streams:
    set -o pipefail
    python scripts/probe_vlle_phase_selective_distillate.py \
        --output /tmp/vlle-phase-selective-overhead.json \
        2>&1 | tee /tmp/vlle-phase-selective-overhead.log > /dev/null

No randomness; one numeric-library thread. Existing property caches are protected
by test-only cache isolation. No production files or runtime data are rewritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
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

PURGE_FRACTION = .02
Column = vlle.EquationOrientedVLLEColumn


def routing(model, top):
    rich_is_second = top['x2']['benzene'] > top['x1']['benzene']
    reflux_x = top['x2'] if rich_is_second else top['x1']
    distillate_x = top['x1'] if rich_is_second else top['x2']
    fraction = top['beta'] if rich_is_second else 1-top['beta']
    hR = model.thermo.mixture_enthalpy(reflux_x, top['T'], 0., P=model.pressures[0])
    hD = model.thermo.mixture_enthalpy(distillate_x, top['T'], 0., P=model.pressures[0])
    return {'reflux_x':reflux_x, 'distillate_x':distillate_x,
            'reflux_fraction':fraction, 'hR':hR, 'hD':hD,
            'raw_reflux':top['L']/(1-PURGE_FRACTION)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    captured = {}
    original_residual = Column.residual
    original_sparsity = Column.sparsity
    original_newton = RigorousDistillation._sparse_newton_solve
    original_active_set = vlle.solve_vlle_active_set

    def residual(model, vector):
        values = original_residual(model, vector).copy()
        top = model.decode(vector)['stages'][0]
        route = routing(model, top)
        D, L, raw = top['V'], top['L'], route['raw_reflux']
        h_aggregate = model.stage_properties(0, top)['hL']
        for index, comp in enumerate(model.components):
            old_out = (L+D)*top['aggregate_x'][comp]
            new_out = raw*route['reflux_x'][comp]+D*route['distillate_x'][comp]
            values[index] += (old_out-new_out)/model.component_scales[comp]
            values[model.stage_row_starts[1]+index] += (
                L*(route['reflux_x'][comp]-top['aggregate_x'][comp])
                / model.component_scales[comp]
            )
        values[model.nc] += (
            (L+D)*h_aggregate-raw*route['hR']-D*route['hD']
        )/model.energy_scale
        values[model.stage_row_starts[1]+model.nc] += (
            L*(route['hR']-h_aggregate)
        )/model.energy_scale
        # The top beta describes the entire condensate, while L is net reflux.
        # Replace the fixed reflux-ratio equation with the phase-flow constraint.
        values[-1] = (raw-route['reflux_fraction']*(raw+D))/model.flow_scale
        return values

    def sparsity(model):
        matrix = original_sparsity(model).tolil()
        top = model.layouts[0]
        matrix[-1, top.start:top.stop] = 1
        return matrix.tocsr()

    def newton(unit, residual_function, sparsity_matrix, x0, options, **kwargs):
        owner = getattr(residual_function, '__self__', None)
        if isinstance(owner, Column):
            kwargs['jacobian'] = None
        return original_newton(unit, residual_function, sparsity_matrix, x0, options, **kwargs)

    class SolutionCaptured(Exception):
        pass

    def active_set(*args, **kwargs):
        captured['solution'] = original_active_set(*args, **kwargs)
        captured['feeds'] = args[2]
        # Production assumes co-routed distillate in its output adapter. Stop
        # before that adapter can mislabel products or audit the wrong outlets.
        raise SolutionCaptured()

    with isolated_runtime_caches():
        thermo = create_thermodynamics(['ethanol', 'water', 'benzene'], 'UNIFAC')
        feed = thermo.calculate_state(298.15, 1., 20.,
            {'ethanol':.35, 'water':.25, 'benzene':.4}, phase='liquid', flash=False)
        params = {'N_stages':16, 'feed_stage':8, 'condenser_type':'total',
                  'stage_phase_model':'VLLE', 'D_to_F':.4, 'P_condenser':1.,
                  'P_drop_per_stage':0., 'mesh_tolerance':1e-5,
                  'max_iterations':80, 'max_jacobian_evaluations':80}
        unit = RigorousDistillation('AZD', thermo, params)
        print('PARAMETERS:', params, 'PURGE:', PURGE_FRACTION, flush=True)
        start = time.perf_counter()
        with patch.object(Column, 'residual', residual), \
             patch.object(Column, 'sparsity', sparsity), \
             patch.object(RigorousDistillation, '_sparse_newton_solve', newton), \
             patch.object(vlle, 'solve_vlle_active_set', active_set):
            try:
                unit.solve({'feed':feed})
            except SolutionCaptured:
                pass
        elapsed = time.perf_counter()-start
        solved = captured['solution']
        stages = solved.decoded['stages']
        if not solved.active[0]:
            raise AssertionError('Phase-selective withdrawal requires two top liquids')
        top, bottom = stages[0], stages[-1]
        # Construct an adapter-shaped object solely for the routing calculation.
        from types import SimpleNamespace
        route = routing(SimpleNamespace(thermo=thermo, pressures=[1.]), top)
        D, B = top['V'], bottom['L']
        purge = route['raw_reflux']-top['L']
        if route['reflux_x']['benzene'] <= route['distillate_x']['benzene']:
            raise AssertionError('No distinct benzene-rich reflux phase')
        products = {
            'distillate':{'F':D, 'T_K':top['T'], 'x':route['distillate_x'], 'H':route['hD']},
            'purge':{'F':purge, 'T_K':top['T'], 'x':route['reflux_x'], 'H':route['hR']},
            'bottoms':{'F':B, 'T_K':bottom['T'], 'x':bottom['aggregate_x'],
                       'H':solved.stage_properties[-1]['hL']},
        }
        components = thermo.components
        balances = {c:feed.F*feed.composition[c]-sum(p['F']*p['x'][c] for p in products.values())
                    for c in components}
        energy = (sum(p['F']*p['H'] for p in products.values())-feed.F*feed.H
                  -solved.decoded['Q_cond']-solved.decoded['Q_reb'])
        fugacity_audits = [
            {'stage':i+1, **vlle.three_phase_fugacity_residuals(
                thermo, state['T'], 1., state['x1'], state['x2'], props['y'], components)}
            for i,(state,props,active) in enumerate(zip(stages, solved.stage_properties, solved.active))
            if active
        ]
        report = {'elapsed_seconds':elapsed, 'parameters':params,
                  'python':sys.version, 'numpy':np.__version__, 'scipy':scipy.__version__,
                  'sources':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                             for name in ('equilibrium_stage_vlle.py', 'unit_operations_distillation.py')},
                  'solver':{k:v for k,v in solved.solver.items() if k != 'x'},
                  'work':solved.work, 'topology_history':solved.topology_history,
                  'stages':stages, 'products':products, 'reflux':route,
                  'net_reflux_flow':top['L'], 'effective_reflux_ratio':top['L']/D,
                  'purge_fraction':PURGE_FRACTION,
                  'condenser_duty_kW':solved.decoded['Q_cond']/3600,
                  'reboiler_duty_kW':solved.decoded['Q_reb']/3600,
                  'component_balance_residuals_kmol_h':balances,
                  'energy_balance_residual_kJ_h':energy, 'fugacity_audits':fugacity_audits}
        args.output.write_text(json.dumps(report, indent=2, default=lambda value:value.item())+'\n')
        print('ELAPSED:', elapsed, flush=True)
        print('MESH RESIDUAL:', solved.solver['residual_norm'], flush=True)
        print('TOPOLOGY:', solved.topology_history, flush=True)
        print('PRODUCTS:', json.dumps(products), flush=True)
        print('REFLUX:', json.dumps(route), 'NET FLOW:', top['L'], flush=True)
        print('BALANCES:', balances, 'ENERGY:', energy, flush=True)
        print('FUGACITY AUDITS:', json.dumps(fugacity_audits), flush=True)


if __name__ == '__main__':
    main()
