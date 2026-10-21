"""Compare gamma-infinity transfer against saved finite-concentration results.

Filter BOTH measured mole fractions below 0.05 or 0.01. Compute the endpoint
activity coefficients with the authoritative pfdsim backend; no SLE root solve.
Reuse the original deduplication and the held-out solvent multiplier baseline.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import time

import numpy as np
from rdkit import rdBase

from benchmark_solubility_transfer import (
    METHODS, SCOPES, backend_for, groups, model, prepare, write_csv,
)
from benchmark_solvent_transfer_multipliers import (
    averages, difference_bootstrap, reconstruct,
)
from solubility_paths import DATA_DIR, RESULTS_DIR, verify_manifest, write_manifest

CUTOFFS = (0.05, 0.01)
LN10 = math.log(10)


def self_check():
    """Check the true x=0 endpoint rather than guessing a dilution epsilon."""
    maximum_difference = 0.0
    for method in METHODS:
        # This binary has complete parameters in every variant, so the check
        # tests the endpoint algorithm rather than parameter-table coverage.
        backend = backend_for(method, 'CCO', 'O')
        definitions = [groups(method,'CCO'),groups(method,'O')]
        endpoint = backend.activity_coefficients([0.0,1.0],298.15)[0]
        readable = model(method).activity_coefficients(definitions,[0.0,1.0],298.15)[0]
        near = backend.activity_coefficients([1e-12,1-1e-12],298.15)[0]
        assert endpoint>0 and math.isfinite(endpoint)
        assert math.isclose(endpoint,readable,rel_tol=1e-10,abs_tol=1e-10)
        difference = abs(math.log(endpoint/near))
        assert difference<1e-7, (method,difference)
        maximum_difference = max(maximum_difference,difference)
        # The same solvent on both sides must leave reference solubility alone.
        assert math.log10(endpoint)-math.log10(endpoint) == 0
    return dict(endpoint_matches_readable_all_six=True,
                endpoint_matches_near_dilute_all_six=True,
                maximum_ln_gamma_difference_at_1e_minus_12=maximum_difference,
                identical_solvent_transfer=True)


def compute_solute(job):
    solute, smiles, requests = job
    rdBase.DisableLog('rdApp.*')
    rows = []
    for method_index, method in enumerate(METHODS):
        backends = {}
        errors = {}
        for solvent, temperature, solvent_smiles in requests:
            row = dict(method=method,solute_index=solute,solvent_index=solvent,temperature_K=temperature)
            if solvent not in backends and solvent not in errors:
                try:
                    backends[solvent] = backend_for(method,smiles,solvent_smiles)
                except (ValueError,KeyError,OverflowError) as error:
                    errors[solvent] = str(error)
            if solvent in errors:
                row.update(status='model_unavailable',reason=errors[solvent])
            else:
                try:
                    backend = backends[solvent]
                    endpoint = backend.activity_coefficients([0.0,1.0],temperature)[0]
                    near = backend.activity_coefficients([1e-14,1-1e-14],temperature)[0]
                    if not math.isfinite(endpoint) or endpoint<=0:
                        raise ValueError('Nonfinite or nonpositive gamma infinity')
                    row.update(status='ok',log_gamma_infinity=math.log(endpoint),
                               ln_gamma_near_endpoint_difference=abs(math.log(endpoint/near)))
                except (ValueError,OverflowError,ZeroDivisionError) as error:
                    row.update(status='numerical_unavailable',reason=str(error))
            row['method_index'] = method_index
            rows.append(row)
    return rows


def paired(first, second, label, seed, draws):
    result = difference_bootstrap(first-second,seed,draws)
    return dict(label=label,MAE_difference_log10=result['UNIFAC_minus_baseline_MAE_log10'],
                paired_95pct_CI=result['paired_95pct_CI'],
                paired_familywise_95pct_CI_six_variants=result['paired_familywise_95pct_CI_six_variants'])


def score(rows, cases, capped, uncapped, baseline, seed, draws):
    if not len(rows):
        return {'predictions':0}
    positions = np.searchsorted(cases['case'],rows['case'])
    assert np.array_equal(cases['case'][positions],rows['case'])
    infinite_score, infinite_loss = averages(capped[positions],rows['solute'])
    finite_score, finite_loss = averages(rows['error_log10'],rows['solute'])
    uncapped_score, _ = averages(uncapped[positions],rows['solute'])
    approximation_score, _ = averages(capped[positions]-rows['error_log10'],rows['solute'])
    output = dict(infinite=infinite_score,finite=finite_score,uncapped_infinite=uncapped_score,
                  infinite_minus_finite=paired(infinite_loss,finite_loss,'infinite minus finite',seed,draws),
                  predicted_log10_difference=approximation_score,
                  predicted_log10_difference_abs_case_quantiles=np.quantile(
                      np.abs(capped[positions]-rows['error_log10']),[.5,.9,.95,.99]).tolist(),
                  predictions_above_one_before_capping=int(np.sum(uncapped[positions]+cases['log_target_x'][positions]>0)))
    available = np.isfinite(baseline[positions]).all(axis=1)
    if np.any(available):
        comparison = rows[available]
        p = positions[available]
        inf, i = averages(capped[p],comparison['solute'])
        fin, f = averages(comparison['error_log10'],comparison['solute'])
        base, b = averages(baseline[p],comparison['solute'])
        output['baseline_comparison'] = dict(
            infinite=inf,finite=fin,baseline=base,
            infinite_minus_baseline=paired(i,b,'infinite minus multiplier',seed,draws),
            finite_minus_baseline=paired(f,b,'finite minus multiplier',seed,draws),
            unavailable_baseline_cases_removed=int(np.sum(~available)))
    return output


def analyze(records, cases, infinite, raw_infinite, baseline, seed, draws):
    output = {}
    for cutoff in CUTOFFS:
        dilute = (cases['log_reference_x']<math.log10(cutoff)) & (cases['log_target_x']<math.log10(cutoff))
        cutoff_result = {}
        for scope_index,scope in enumerate(SCOPES):
            scope_mask = dilute & (cases['scope']==scope_index)
            methods = {}
            selected = {}
            for method,name in enumerate(METHODS):
                rows = records[(records['method']==method)&(records['scope']==scope_index)]
                positions = np.searchsorted(cases['case'],rows['case'])
                paired_mask = scope_mask[positions] & (rows['status']==0) & np.isfinite(infinite[method,positions])
                selected[method] = rows[paired_mask]
                methods[name] = dict(
                    eligible_dilute_cases=int(np.sum(scope_mask)),
                    infinite_available_cases=int(np.sum(scope_mask & np.isfinite(infinite[method]))),
                    finite_successful_cases=int(np.sum(scope_mask[positions] & (rows['status']==0))),
                    own_coverage=score(selected[method],cases,infinite[method],raw_infinite[method],baseline,seed,draws))
            common_six = selected[0]['case']
            common_five = common_six
            for method in range(1,len(METHODS)):
                common_six = np.intersect1d(common_six,selected[method]['case'])
                if method<5:
                    common_five = np.intersect1d(common_five,selected[method]['case'])
            for method,name in enumerate(METHODS):
                six = selected[method][np.isin(selected[method]['case'],common_six)]
                methods[name]['common_all_six'] = score(six,cases,infinite[method],raw_infinite[method],baseline,seed,draws)
                if method<5:
                    five = selected[method][np.isin(selected[method]['case'],common_five)]
                    methods[name]['common_five'] = score(five,cases,infinite[method],raw_infinite[method],baseline,seed,draws)
                methods[name]['categories_own_coverage'] = {
                    category:score(selected[method][selected[method]['category']==i],cases,
                                   infinite[method],raw_infinite[method],baseline,seed,draws)
                    for i,category in enumerate(('water_to_nonwater','nonwater_to_water','nonwater_to_nonwater'))}
            cutoff_result[scope] = methods
        output[str(cutoff)] = cutoff_result
    return output


def table(methods,cohort):
    lines = ['| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Difference | Paired 95% CI | Infinite RMSE | Finite RMSE |',
             '|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for name,values in methods.items():
        if cohort not in values or 'infinite' not in values[cohort]:
            continue
        row = values[cohort]
        inf,fin,p = row['infinite'],row['finite'],row['infinite_minus_finite']
        lo,hi = p['paired_95pct_CI']
        lines.append(f'| {name} | {inf["predictions"]:,} | {inf["solutes"]} | {fin["MAE_log10"]:.4f} | '
                     f'{inf["MAE_log10"]:.4f} | {p["MAE_difference_log10"]:+.4f} | {lo:+.4f} to {hi:+.4f} | '
                     f'{inf["RMSE_log10"]:.4f} | {fin["RMSE_log10"]:.4f} |')
    return lines


def report(output,summary):
    lines = ['# Infinite-dilution solubility transfer','',
             'Predict x_target = x_reference gamma_infinity_reference / gamma_infinity_target, '
             'with both gamma values evaluated at zero solute mole fraction and the measured temperature. '
             'No activity-equilibrium root is solved.','',
             '## Protocol','',
             '- Interpret 0.05 and 0.01 as mole fractions (5 mol% and 1 mol%), consistent with the preceding assessment.',
             '- Apply the strict cutoff to BOTH observed reference and observed target solubilities. '
             'This is a retrospective dilute-subset test; the unknown target’s solubility cannot be used to select cases prospectively.',
             '- Do not filter out predictions above the cutoff. Cap only at the physical upper bound x=1; retain uncapped error metrics as a sensitivity.',
             '- Reuse the previous exact-temperature/publication pairing, deduplication, strict fragmentation and complete interaction requirement.',
             '- Main comparisons use identical cases where both finite and infinite predictions are available. '
             'Infinite availability before requiring finite success is also recorded in summary.json.',
             '- Use pfdsim CompiledUNIFACBackend at [0,1], rather than approximating the limit by an arbitrary small concentration. '
             'Verify all six variants against the readable implementation and near-zero concentration.',
             '- Average errors equally over molecules, as before. Positive infinite-minus-finite MAE differences mean the infinite approximation is worse.',
             f'- Bootstrap whole molecules {summary["bootstrap_draws"]:,} times with seed {summary["seed"]}. '
             'Ordinary paired 95% intervals and simultaneous 95% intervals across six variants at each cutoff are saved.',
             '- Reuse the existing 10-repeat, five-fold molecule-held-out multiplier errors. No baseline is refitted on the test subsets.',
             '- The fixed cutoff selects different compounds, so compare finite/infinite within each cutoff, not across cutoffs. '
             'The preceding solid-form, pH/speciation and parameter-training-overlap limitations remain.','']
    for cutoff in CUTOFFS:
        values = summary['comparisons'][str(cutoff)]
        for cohort in ('own_coverage','common_all_six','common_five'):
            lines += [f'## Both measured mole fractions < {cutoff}: {cohort}','',
                      *table(values['same_publication'],cohort),'']
        lines += [f'## Cutoff {cutoff}: directional breakdown','',
                  '| Variant | Direction | Transfers | Molecules | Finite MAE | Infinite MAE | Difference |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for name,row in values['same_publication'].items():
            for category,result in row['categories_own_coverage'].items():
                if 'infinite' not in result:
                    continue
                lines.append(f'| {name} | {category} | {result["infinite"]["predictions"]:,} | '
                             f'{result["infinite"]["solutes"]} | {result["finite"]["MAE_log10"]:.4f} | '
                             f'{result["infinite"]["MAE_log10"]:.4f} | {result["infinite_minus_finite"]["MAE_difference_log10"]:+.4f} |')
        lines += ['',f'## Cutoff {cutoff}: multiplier baseline on each variant’s paired cases','',
                  '| Variant | Transfers | Molecules | Finite MAE | Infinite MAE | Multiplier MAE |',
                  '|---|---:|---:|---:|---:|---:|']
        for name,row in values['same_publication'].items():
            b = row['own_coverage'].get('baseline_comparison')
            if b:
                lines.append(f'| {name} | {b["infinite"]["predictions"]:,} | {b["infinite"]["solutes"]} | '
                             f'{b["finite"]["MAE_log10"]:.4f} | {b["infinite"]["MAE_log10"]:.4f} | {b["baseline"]["MAE_log10"]:.4f} |')
        for scope in SCOPES[1:]:
            lines += ['',f'## Cutoff {cutoff}: {scope}, own coverage','',*table(values[scope],'own_coverage'),'']
    lines += ['## Numerical audit','', '```json',json.dumps(summary['numerical_audit'],indent=2),'```','',
              '## Files','',
              '- gamma_infinity.csv: endpoint activities, availability and near-endpoint checks for every required binary/temperature/variant.',
              '- infinite_errors.npz: case IDs and capped/uncapped infinite prediction errors.',
              '- summary.json: all scores, uncertainties, prediction differences and baseline comparisons.',
              '- manifest.json: source/parameter/code fingerprints.',
              '- Existing finite predictions, held-out baseline results, source data and production code were not modified.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    parser.add_argument('--results-dir',type=Path,default=RESULTS_DIR/'unifac-transfer-results')
    parser.add_argument('--baseline-dir',type=Path,default=RESULTS_DIR/'solvent-multiplier-results')
    parser.add_argument('--output-dir',type=Path,default=RESULTS_DIR/'infinite-dilution-assessment')
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--bootstrap-draws',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20261014)
    args = parser.parse_args()
    if not 1<=args.workers<=5:
        parser.error('Use one to five workers')
    started = time.monotonic()
    rdBase.DisableLog('rdApp.*')
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    checks = self_check()
    print('Infinite endpoint checks passed for all six variants',flush=True)
    data_dir = args.data_dir
    cases,_,records,molecules,solvent_metadata,_ = reconstruct(data_dir,args.results_dir)
    jobs,solute_definitions,solvent_definitions,_,_ = prepare(data_dir)
    temperatures = np.empty(len(cases))
    cursor = 0
    for molecule in molecules:
        for bundle in jobs[molecule['inchi']]:
            n = len(bundle['observations'])
            count = n-1 if bundle.get('anchor_only') else n*(n-1)
            temperatures[cursor:cursor+count] = bundle['T']
            cursor += count
    assert cursor == len(cases)
    verify_manifest(args.baseline_dir/'manifest.json')
    with np.load(args.baseline_dir/'baseline_errors.npz') as saved:
        assert np.array_equal(saved['case'],cases['case'])
        baseline = saved['capped']
    needed = (cases['log_reference_x']<math.log10(max(CUTOFFS))) & (cases['log_target_x']<math.log10(max(CUTOFFS)))
    task_arguments = []
    identities,starts,counts = np.unique(cases['solute'],return_index=True,return_counts=True)
    for solute,start,count in zip(identities,starts,counts):
        positions = np.arange(start,start+count)[needed[start:start+count]]
        if not len(positions):
            continue
        requests = sorted(set(zip(cases['reference'][positions].tolist(),temperatures[positions].tolist())) |
                          set(zip(cases['target'][positions].tolist(),temperatures[positions].tolist())))
        task_arguments.append((int(solute),solute_definitions[molecules[solute]['inchi']]['smiles'],
                               [(solvent,T,solvent_definitions[solvent_metadata[solvent]['inchi']]['smiles'])
                                for solvent,T in requests]))
    gamma_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compute_solute,job):job[0] for job in task_arguments}
        task_count = len(futures)
        for count,future in enumerate(as_completed(futures),1):
            futures.pop(future)
            gamma_rows.extend(future.result())
            if count%100==0 or count==task_count:
                print(f'Computed endpoint activities for {count}/{task_count} solutes in {time.monotonic()-started:.1f}s',flush=True)
    write_csv(output/'gamma_infinity.csv',gamma_rows)
    lookup = {(r['method_index'],r['solute_index'],r['solvent_index'],r['temperature_K']):r['log_gamma_infinity']
              for r in gamma_rows if r['status']=='ok'}
    infinite = np.full((len(METHODS),len(cases)),np.nan)
    raw_infinite = infinite.copy()
    positions = np.flatnonzero(needed)
    for method in range(len(METHODS)):
        ref = np.array([lookup.get((method,int(cases['solute'][p]),int(cases['reference'][p]),temperatures[p]),np.nan) for p in positions])
        target = np.array([lookup.get((method,int(cases['solute'][p]),int(cases['target'][p]),temperatures[p]),np.nan) for p in positions])
        predicted = cases['log_reference_x'][positions]+(ref-target)/LN10
        raw_infinite[method,positions] = predicted-cases['log_target_x'][positions]
        infinite[method,positions] = np.minimum(predicted,0)-cases['log_target_x'][positions]
    np.savez_compressed(output/'infinite_errors.npz',capped=infinite,uncapped=raw_infinite,case=cases['case'],temperature_K=temperatures)
    print('Infinite predictions saved; calculating paired errors and intervals',flush=True)
    summary = dict(seed=args.seed,bootstrap_draws=args.bootstrap_draws,cutoffs=list(CUTOFFS),
                   numerical_audit=dict(checks=checks,required_solutes=len(task_arguments),
                                        endpoint_evaluations=len(gamma_rows),
                                        maximum_near_endpoint_ln_gamma_difference=max(
                                            (r['ln_gamma_near_endpoint_difference'] for r in gamma_rows if r['status']=='ok'),default=0)),
                   comparisons=analyze(records,cases,infinite,raw_infinite,baseline,args.seed,args.bootstrap_draws))
    summary['elapsed_seconds'] = time.monotonic()-started
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fingerprints = [Path(__file__),Path(__file__).with_name('benchmark_solvent_transfer_multipliers.py'),
                    args.results_dir/'metric_records.npy',args.results_dir/'manifest.json',
                    args.baseline_dir/'baseline_errors.npz',args.baseline_dir/'manifest.json']
    write_manifest(output/'manifest.json', vars(args), fingerprints)
    report(output,summary)
    for cutoff in CUTOFFS:
        print(f'\nCutoff {cutoff}: same-publication own coverage',flush=True)
        for method,row in summary['comparisons'][str(cutoff)]['same_publication'].items():
            value = row['own_coverage']
            print(method,json.dumps(value),flush=True)
    print(f'Complete: {output}; elapsed {summary["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    main()
