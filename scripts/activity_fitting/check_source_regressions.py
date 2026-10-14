"""Reproduce source-fit quality, predictions and CPU timings across environments.

Developer-only: uses the same isolated source fixtures as the numerical tests.
Writes a new report, never approved source data or runtime interaction tables.
"""

import argparse
from contextlib import nullcontext
import importlib
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.activity_fitting.binary_regression import (
    central_jacobian, complex_jacobian, projected_gradient_cosine,
)
from tests.cache_isolation import isolated_runtime_caches
from tests.fit_source_comparison import assert_source_inputs_equal

SOURCES = {
    'fendu': ('fit_fendu_2025_activity_parameters', 'fendu_2025_activity_refits'),
    'mohsen': ('fit_mohsen_nia_2010_pentanol_propionic', 'mohsen_nia_2010_pentanol_propionic_association'),
    'cyclohexane': ('fit_moreau_ovejero_2012_2007_activity', 'moreau_ovejero_pentanol_cyclohexane_joint_fit'),
    'toluene': ('fit_moreau_2012_pentanol_toluene_activity', 'moreau_2012_pentanol_toluene_joint_fit'),
    'hexane': ('fit_moreau_ovejero_2016_2007_pentanol_hexane_activity', 'moreau_ovejero_2016_2007_pentanol_hexane_joint_fit'),
}


def regression(label, function, parameters, previous, bounds, predictions, jacobian=complex_jacobian):
    residual = function(parameters)
    old = function(previous)
    quality = projected_gradient_cosine(residual, jacobian(function, parameters), parameters, *bounds)
    if not np.isfinite(residual).all() or quality > 1e-7:
        raise RuntimeError(f'{label}: invalid regression, stationarity={quality:g}')
    if residual @ residual > old @ old + 1e-8 * max(1., old @ old):
        raise RuntimeError(f'{label}: objective regressed')
    return {'id': label, 'objective': float(residual @ residual),
            'approved_parameters_objective': float(old @ old), 'stationarity': quality,
            'parameters': parameters.tolist(),
            'predictions': {key: np.asarray(values).tolist() for key, values in predictions.items()}}


def check_source(name):
    module_name, filename = SOURCES[name]
    module = importlib.import_module('scripts.activity_fitting.' + module_name)
    source_path = ROOT/'data/source/activity_fitting'/(filename+'.json')
    approved = json.loads(source_path.read_text())
    with isolated_runtime_caches():
        start = time.process_time()
        generated = module.build_payload()
        elapsed = time.process_time() - start
        assert_source_inputs_equal(unittest.TestCase(), generated, approved, refitted_antoine=name == 'fendu')
        reports = []
        if name == 'fendu':
            for system_name, system in module.SYSTEMS.items():
                rows = module._rows(system)
                for model, form, specs in (
                    ('NRTL', 'A_plus_B_over_T_fixed_alpha', module.NRTL_SPECS),
                    ('UNIQUAC', 'A_plus_B_over_T', module.UNIQUAC_SPECS),
                ):
                    current = generated['analyses'][system_name]['candidate_fits'][model][form]
                    previous = approved['analyses'][system_name]['candidate_fits'][model][form]
                    parameters = np.array(current['parameters'])
                    function = lambda p: module._pressure_residuals(model, form, p, rows, system)
                    predictions = {'pressure_kPa': (1+function(parameters))*np.array(rows)[:, 2]}
                    reports.append(regression(system_name+'/'+model, function, parameters,
                                              np.array(previous['parameters']), specs[form][1:], predictions))
        elif name == 'mohsen':
            for model, fit in generated['analysis']['vdm_refits'].items():
                engine = module.MutableVDMModel(model)
                def function(p):
                    return module.objective(engine, np.r_[p, .1] if model == 'NRTL' else p)
                parameters = np.array(fit['parameters'][:2])
                previous = np.array(approved['analysis']['vdm_refits'][model]['parameters'][:2])
                points = module.predictions(model, 'VDM', fit['parameters'])
                predictions = {'temperature_K': [p['calculated_T_K'] for p in points],
                               'vapor_fraction': [p['calculated_y1'] for p in points]}
                reports.append(regression(model, function, parameters, previous, (-5000., 5000.),
                                          predictions, jacobian=central_jacobian))
        else:
            common = module.common if hasattr(module, 'common') else module
            context = module.configured_common_fit() if hasattr(module, 'configured_common_fit') else nullcontext()
            with context:
                for record in generated['interactions']:
                    model = record['model']
                    alpha = record.get('alpha12') if model == 'NRTL' else None
                    def candidate(payload):
                        return next(c for c in payload['model_selection'] if
                                    (c['model'], c['alpha'], c['form']) == (model, alpha, 'ABH'))
                    parameters = np.array(candidate(generated)['parameters'])
                    previous = np.array(candidate(approved)['parameters'])
                    function = lambda p: common.objective(model, alpha, True, p)
                    x = np.linspace(.001, .999, 101)
                    predictions = {'pressure_kPa': common.source_bubble(model, x, parameters, alpha, True)[0],
                                   'enthalpy_J_mol': np.concatenate([
                                       common.excess_enthalpy(model, x, T, parameters, alpha, True)
                                       for T in sorted(set(common.HE) | {record['Tmax_K']})])}
                    lower = np.array([-10., -10000., -500., -10., -10000., -500.])
                    reports.append(regression(model, function, parameters, previous, (lower, -lower), predictions))
    return {'source': name, 'cpu_seconds': elapsed, 'regressions': reports,
            'approved_source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--sources', nargs='+', choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new report path to preserve the previous run.')
    np.random.seed(4182)
    payload = {'seed': 4182, 'environment': {'python': sys.version.split()[0],
               'numpy': np.__version__, 'scipy': scipy.__version__, 'tblite': version('tblite')},
               'sources': []}
    for name in args.sources:
        source = check_source(name)
        payload['sources'].append(source)
        print(name, 'validated', len(source['regressions']), 'fits', flush=True)
    if args.reference:
        previous = json.loads(args.reference.read_text())
        input_versions = {s['source']: s['approved_source_sha256'] for s in previous['sources']}
        for source in payload['sources']:
            if input_versions[source['source']] != source['approved_source_sha256']:
                raise ValueError('Approved source changed between reports: ' + source['source'])
        index = {(s['source'], r['id']): r for s in previous['sources'] for r in s['regressions']}
        differences = []
        for source in payload['sources']:
            for current in source['regressions']:
                reference = index[(source['source'], current['id'])]
                differences.append({'source': source['source'], 'id': current['id'],
                    'relative_objective_difference': abs(current['objective']-reference['objective']) / max(np.finfo(float).tiny, abs(reference['objective'])),
                    'prediction_differences': {k: float(np.max(np.abs(np.array(v)-reference['predictions'][k])))
                                               for k, v in current['predictions'].items()}})
        payload['reference_comparison'] = differences
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
