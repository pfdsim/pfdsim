#!/usr/bin/env python3
"""Add one measured atmospheric VLLE point to published butanol/water fit inputs.

Read-only research probe. The source store supplies the exact published requests,
coefficients, bounds and structural basis; no source or runtime data is written.
An LLE-only refit is included to distinguish solver refinement from adding VLLE.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from activity_fit_store import DEFAULT_ACTIVITY_FITS_PATH
from scripts.activity_fitting.probe_butanol_joint_models import metric
from thermodynamics_models.interaction_fitting import parse_observations, prepare_fit

VLLE_POINT = {
    'id': 'atmospheric-vlle', 'kind': 'VLLE',
    'T_K': 365.73, 'P_bar': 1.013,
    'x1_alpha': .021, 'x1_beta': .362, 'y1': .246,
    'source': 'Gomis et al., J. Chem. Eng. Data 2013, 58, 3320–3326; '
              '10.1021/je400501z; binary endpoint, all fractions on butanol mole basis',
}


def published_inputs(path, model):
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)) as db:
        rows = db.execute(
            "SELECT id,result FROM fits WHERE status='published' AND model=? "
            "AND ((cas1='71-36-3' AND cas2='7732-18-5') OR "
            "(cas2='71-36-3' AND cas1='7732-18-5'))", (model,),
        ).fetchall()
    if len(rows) != 1:
        raise ValueError(f'Expected exactly one published butanol/water {model} fit')
    identifier, encoded = rows[0]
    report = json.loads(encoded)
    if report['component_cas'] != ['71-36-3', '7732-18-5']:
        raise ValueError('Published request must put butanol first')
    return identifier, report


def evaluate(problem, values, original_rows):
    report = problem.report(values, original_rows)
    problem.install(values)
    point = parse_observations([deepcopy(VLLE_POINT)])[0]
    errors, raw, _ = problem.row_errors(values, point)
    try:
        invariant = problem.predict_vlle(P=point['P_bar'], T_guess=point['T_K'])
        invariant['T_error_K'] = invariant['T_K'] - point['T_K']
        for field in ('x1_alpha', 'x1_beta', 'y1'):
            invariant[field+'_error'] = invariant[field] - point[field]
    except Exception as error:
        invariant = {'error': str(error)}
    residuals = problem.residuals(values, original_rows)
    return {
        'LLE_aqueous_RMSE': metric(report, 'LLE', 'x1_alpha'),
        'LLE_organic_RMSE': metric(report, 'LLE', 'x1_beta'),
        'UCST_temperature_abs_error_K': metric(report, 'UCST', 'T_K', 'MAE'),
        'UCST_composition_abs_error': metric(report, 'UCST', 'x1', 'MAE'),
        'original_data_objective': float(residuals @ residuals),
        'unphysical_original_observations': [
            (p['id'], p['kind']) for p in report['points'] if not p['physical']
        ],
        'VLLE_point_log_fugacity_residuals': raw,
        'VLLE_point_objective_at_unit_weight': float(errors @ errors),
        'predicted_atmospheric_invariant': invariant,
    }


def fit_case(task):
    model, published, weight, max_nfev, fit_alpha = task
    request = deepcopy(published['request'])
    request.update(starts=1, max_nfev=max_nfev, seed=1729, cv={'method':'none','folds':5})
    # Retain the published unbounded-law limits, fixed alpha, R/Q, pressure
    # correlations, row weights, missing-branch handling and continuation.
    request['initial'] = deepcopy(published['coefficients'])
    if model == 'NRTL' and fit_alpha:
        request['fit_alpha'] = True
        request['initial']['alpha12'] = published['parameters']['alpha12']
    if weight is not None:
        request['observations'].append({**VLLE_POINT, 'weight': weight})
    label = 'LLE-only-refit' if weight is None else f'LLE-plus-VLLE-weight-{weight:g}'
    problem = prepare_fit(request)
    original_rows = [r for r in problem.request['observations'] if r['id'] != VLLE_POINT['id']]
    baseline = evaluate(problem, problem.initial.copy(), original_rows)
    start = time.process_time()
    result = problem.solve(problem.request['observations'])
    cpu_seconds = time.process_time()-start
    values = result.pop('values')
    outcome = evaluate(problem, values, original_rows)
    return {
        'model':model, 'case':label, 'VLLE_weight':weight,
        'fit_form':problem.request['form'], 'fit_alpha':problem.request['fit_alpha'],
        'fixed_alpha':problem.request['alpha'] if model=='NRTL' and not fit_alpha else None,
        'training_observations':len(problem.request['observations']),
        'optimizer':result, 'fit_CPU_seconds':cpu_seconds,
        'published_baseline':baseline, 'candidate':outcome,
        'parameters':problem.parameters(values),
        'coefficients':dict(zip(problem.names,(values*np.asarray(problem.scales)).tolist())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-store',type=Path,default=DEFAULT_ACTIVITY_FITS_PATH)
    parser.add_argument('--models',nargs='+',choices=['NRTL','UNIQUAC'],default=['NRTL','UNIQUAC'])
    parser.add_argument('--vlle-weights',nargs='+',type=float,default=[1.])
    parser.add_argument('--max-nfev',type=int,default=150)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--skip-control',action='store_true',help='Reuse the already measured LLE-only control.')
    parser.add_argument('--fit-nrtl-alpha',action='store_true',help='Explore a free NRTL alpha in its normal bounds.')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():parser.error('Choose a new output filename; existing results are preserved.')
    if args.max_nfev < 1 or any(w <= 0 or not np.isfinite(w) for w in args.vlle_weights):
        parser.error('Evaluation count and VLLE weights must be positive and finite.')
    tasks=[]
    identifiers={}
    for model in args.models:
        identifier,published=published_inputs(args.source_store,model)
        identifiers[model]=identifier
        weights = args.vlle_weights if args.skip_control else [None,*args.vlle_weights]
        tasks.extend((model,published,weight,args.max_nfev,args.fit_nrtl_alpha) for weight in weights)
    results=[]
    with ProcessPoolExecutor(max_workers=min(max(args.workers,1),5)) as pool:
        for outcome in pool.map(fit_case,tasks):
            results.append(outcome)
            print(outcome['model'],outcome['case'],'nfev',outcome['optimizer']['nfev'],
                  'converged',outcome['optimizer']['success'],flush=True)
    args.output.write_text(json.dumps({'VLLE_point':VLLE_POINT,'published_fit_ids':identifiers,
                                      'results':results},indent=2,allow_nan=False)+'\n')
    print('Saved candidate-only report',args.output,flush=True)


if __name__=='__main__':main()
