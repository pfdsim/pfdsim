"""Molecule-held-out evaluation of per-solvent bias-corrected UNIFAC.

Reuse saved predictions. Fit corrections and a domain-matched solvent-only
baseline on exactly the same training cases. Existing study folds and its
broad-population solvent baseline are retained for comparability.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from rdkit import rdBase

from benchmark_solubility_transfer import METHODS, write_csv
from benchmark_solvent_transfer_multipliers import averages, difference_bootstrap, fit, reconstruct
from solubility_paths import DATA_DIR, RESULTS_DIR, verify_manifest, write_manifest

SCENARIOS = ('finite_all','finite_0.05','infinite_0.05','finite_0.01','infinite_0.01')
MODES = ('relative_solvent','target_only')
FAMILY_SIZE = len(SCENARIOS)*len(METHODS)*len(MODES)
SCORING_DATA = None


def normal_equations(reference,target,solute,folds,error,shift,n_solvents,n_folds):
    """Equal molecule weights; identical design for bias and mean-shift fits."""
    _,inverse,counts = np.unique(solute,return_inverse=True,return_counts=True)
    weight = 1/counts[inverse]
    assignment = folds[solute]
    a = np.zeros((n_folds,n_solvents,n_solvents))
    b_error = np.zeros((n_folds,n_solvents))
    b_shift = np.zeros_like(b_error)
    for row,column,sign in ((reference,reference,1),(target,target,1),
                            (reference,target,-1),(target,reference,-1)):
        np.add.at(a,(assignment,row,column),sign*weight)
    for endpoint,sign in ((target,1),(reference,-1)):
        np.add.at(b_error,(assignment,endpoint),sign*weight*error)
        np.add.at(b_shift,(assignment,endpoint),sign*weight*shift)
    # Target-only sensitivity follows the previous bias diagnostic: average
    # references/temperatures within each molecule/target, then over molecules.
    _,target_inverse,target_counts = np.unique(solute*n_solvents+target,return_inverse=True,return_counts=True)
    target_weight = 1/target_counts[target_inverse]
    target_sum = np.zeros((n_folds,n_solvents))
    target_n = np.zeros_like(target_sum)
    np.add.at(target_sum,(assignment,target),target_weight*error)
    np.add.at(target_n,(assignment,target),target_weight)
    return a,b_error,b_shift,target_sum,target_n


def fit_training(normals,held_out,anchor):
    a,e,d,target_sum,target_n = normals
    train_a = a.sum(axis=0)-a[held_out]
    correction,components,residual_c = fit(train_a,e.sum(axis=0)-e[held_out],anchor)
    baseline,base_components,residual_b = fit(train_a,d.sum(axis=0)-d[held_out],anchor)
    assert np.array_equal(components,base_components)
    sums = target_sum.sum(axis=0)-target_sum[held_out]
    counts = target_n.sum(axis=0)-target_n[held_out]
    marginal = np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>1e-10)
    return correction,baseline,marginal,components,max(residual_c,residual_b)


def self_check():
    reference = np.array([0,0,1,1,0,2,0,1,2,0,1,2])
    target = np.array([1,2,0,2,2,1,2,0,1,1,2,0])
    solute = np.repeat(np.arange(4),3)
    folds = np.array([0,1,1,0])
    d = np.array([.2,-.3,.1,.7,.5,-.2,.4,.8,-.6,.3,-.4,.2])
    solvent_model = np.array([0.,.6,-.8])
    g = solvent_model[target]-solvent_model[reference]
    e = g-d
    normal = normal_equations(reference,target,solute,folds,e,d,4,2)
    c,b,_,components,residual = fit_training(normal,0,0)
    test = folds[solute]==0
    corrected_shift = g[test]-(c[target[test]]-c[reference[test]])
    baseline_shift = b[target[test]]-b[reference[test]]
    assert np.allclose(corrected_shift,baseline_shift,atol=1e-12)
    assert components[3]==-1 and np.isnan(c[3])
    changed = e.copy()
    changed[test] += 10000
    n2 = normal_equations(reference,target,solute,folds,changed,d,4,2)
    c2,_,_,_,_ = fit_training(n2,0,0)
    assert np.allclose(c[:3],c2[:3],atol=1e-10)
    # A perfect solute-dependent predictor with a constant solvent bias is
    # recovered after estimating that bias from the other molecules.
    bias = np.array([0.,.3,-.5])
    perfect_error = bias[target]-bias[reference]
    n3 = normal_equations(reference,target,solute,folds,perfect_error,d,4,2)
    c3,_,_,_,_ = fit_training(n3,0,0)
    assert np.allclose(perfect_error[test]-(c3[target[test]]-c3[reference[test]]),0,atol=1e-12)
    return dict(solvent_only_UNIFAC_reduces_to_matched_baseline=True,
                held_out_error_perturbation_does_not_change_fit=True,
                known_bias_recovered=True,unseen_solvent_unavailable=True,
                maximum_synthetic_residual=residual)


def calibrate(task):
    scenario,method,rows,case_subset,folds,n_solvents,anchor,solvent_names,output = task
    output = Path(output)
    repeats,n_folds = folds.shape[0],int(folds.max())+1
    n = len(rows)
    corrected = np.full((n,repeats),np.nan)
    target_corrected = corrected.copy()
    baseline = corrected.copy()
    parameters,audit = [],[]
    reference,target = case_subset['reference'],case_subset['target']
    solute = rows['solute']
    error = rows['error_log10']
    actual = case_subset['log_target_x']
    model_prediction = actual+error
    shift = actual-case_subset['log_reference_x']
    for repeat in range(repeats):
        normals = normal_equations(reference,target,solute,folds[repeat],error,shift,n_solvents,n_folds)
        for fold in range(n_folds):
            c,b,marginal,components,residual = fit_training(normals,fold,anchor)
            test = np.flatnonzero(folds[repeat,solute]==fold)
            r,t = reference[test],target[test]
            supported = (components[r]>=0)&(components[r]==components[t])
            positions = test[supported]
            r,t = reference[positions],target[positions]
            prediction = model_prediction[positions]-(c[t]-c[r])
            corrected[positions,repeat] = np.minimum(prediction,0)-actual[positions]
            prediction_baseline = case_subset['log_reference_x'][positions]+b[t]-b[r]
            baseline[positions,repeat] = np.minimum(prediction_baseline,0)-actual[positions]
            target_supported = np.isfinite(marginal[t])
            marginal_positions = positions[target_supported]
            marginal_prediction = model_prediction[marginal_positions]-marginal[target[marginal_positions]]
            target_corrected[marginal_positions,repeat] = np.minimum(marginal_prediction,0)-actual[marginal_positions]
            audit.append(dict(repeat=repeat,fold=fold,held_out_cases=len(test),supported_cases=len(positions),
                              training_molecules=int(np.unique(solute[folds[repeat,solute]!=fold]).size),
                              maximum_normal_equation_residual=residual,
                              relative_predictions_capped=int(np.sum(prediction>0))))
            for solvent in np.flatnonzero(components>=0):
                parameters.append(dict(repeat=repeat,fold=fold,solvent_index=int(solvent),
                                       solvent=solvent_names[solvent],component=int(components[solvent]),
                                       correction_log10=float(c[solvent]),baseline_log10=float(b[solvent]),
                                       target_only_bias_log10=float(marginal[solvent]) if np.isfinite(marginal[solvent]) else ''))
    stem = scenario+'_'+method
    np.savez_compressed(output/(stem+'.npz'),case=rows['case'],solute=solute,
                        uncorrected=error,relative_solvent=corrected,
                        target_only=target_corrected,matched_baseline=baseline)
    write_csv(output/(stem+'_coefficients.csv'),parameters)
    return dict(scenario=scenario,method=method,archive=stem+'.npz',audit=audit,
                candidate_cases=n,candidate_molecules=int(np.unique(solute).size))


def paired_result(first,second,label,seed,draws):
    raw = difference_bootstrap(first-second,seed,draws,requested_family_size=FAMILY_SIZE)
    ci = raw['paired_requested_family_95pct_CI']
    return dict(label=label,MAE_difference_log10=raw['UNIFAC_minus_baseline_MAE_log10'],
                paired_95pct_CI=raw['paired_95pct_CI'],familywise_95pct_CI=ci,
                comparison_family_size=FAMILY_SIZE,convincingly_better=ci[1]<0,
                convincingly_worse=ci[0]>0,
                split_difference_min=raw['split_MAE_difference_min'],
                split_difference_max=raw['split_MAE_difference_max'])


def initialize_scoring(cases,broad,seed,draws):
    global SCORING_DATA
    SCORING_DATA = cases,broad,seed,draws


def evaluate(task):
    scenario,method,archive,cohorts = task
    cases,broad,seed,draws = SCORING_DATA
    with np.load(archive) as saved:
        values = {key:saved[key] for key in saved.files}
    positions = np.searchsorted(cases['case'],values['case'])
    results = {}
    for mode in MODES:
        mode_results = {}
        for cohort,ids in cohorts[mode].items():
            eligible = np.isin(values['case'],ids)
            eligible &= np.isfinite(values[mode]).all(axis=1)&np.isfinite(values['matched_baseline']).all(axis=1)
            eligible &= np.isfinite(broad[positions]).all(axis=1)
            if not np.any(eligible):
                mode_results[cohort] = dict(predictions=0)
                continue
            solute = values['solute'][eligible]
            corrected,c = averages(values[mode][eligible],solute)
            uncorrected,u = averages(values['uncorrected'][eligible],solute)
            matched,m = averages(values['matched_baseline'][eligible],solute)
            previous,p = averages(broad[positions[eligible]],solute)
            mode_results[cohort] = dict(
                corrected=corrected,uncorrected=uncorrected,matched_baseline=matched,
                original_broad_baseline=previous,
                corrected_minus_matched=paired_result(c,m,'corrected minus domain-matched baseline',seed,draws),
                corrected_minus_original_baseline=paired_result(c,p,'corrected minus original broad baseline',seed,draws),
                corrected_minus_uncorrected=paired_result(c,u,'corrected minus uncorrected UNIFAC',seed,draws))
        results[mode] = mode_results
    return scenario,method,results


def result_table(results,mode,cohort):
    lines = ['| Variant | Transfers | Molecules | Uncorrected MAE | Corrected MAE | Matched baseline MAE | Original baseline MAE | Corrected − matched | Familywise 95% CI |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for method,row in results.items():
        value = row[mode].get(cohort,{})
        if 'corrected' not in value:
            continue
        p = value['corrected_minus_matched']
        lo,hi = p['familywise_95pct_CI']
        lines.append(f'| {method} | {value["corrected"]["predictions"]:,} | {value["corrected"]["solutes"]} | '
                     f'{value["uncorrected"]["MAE_log10"]:.4f} | {value["corrected"]["MAE_log10"]:.4f} | '
                     f'{value["matched_baseline"]["MAE_log10"]:.4f} | {value["original_broad_baseline"]["MAE_log10"]:.4f} | '
                     f'{p["MAE_difference_log10"]:+.4f} | {lo:+.4f} to {hi:+.4f} |')
    return lines


def render_report(output,summary):
    lines = ['# Held-out solvent-bias correction versus solvent-only transfer','',
             '## Protocol','',
             '- Main correction: log10(x_corrected) = log10(x_UNIFAC) − (b_target − b_reference). '
             'Fit one constant log bias per solvent using training molecule prediction errors. '
             'Solvent offsets have an arbitrary gauge; only differences are used.',
             '- Sensitivity: subtract the target solvent’s mean signed error only. Average within each training molecule/target first, then over molecules.',
             '- Fit every identifiable solvent coefficient within each training fold. Do not select coefficients using the previous full-dataset significance analysis.',
             '- Reuse the previous five-fold chemical-identity splits, repeated ten times. All temperatures, publications and solvents of each held-out molecule are excluded from fitting.',
             '- Domain-matched baseline: fit log10(x_target/x_reference) using the SAME training transfers, chemical coverage, equal-molecule weights and solvent design matrix as the relative-bias correction. '
             'Thus it is a direct test of whether UNIFAC adds useful molecule-dependent information beyond fitted solvent averages.',
             '- Also report the original broader-population mean-multiplier baseline, with its saved held-out predictions. '
             'That baseline used equal molecule/group/pair weights and all eligible training compounds; it is not the principal controlled comparison.',
             '- Corrections and matched baseline use least squares with equal total weight per molecule. '
             'Each available directed transfer within a molecule has equal weight. No ridge penalty or hyperparameter search is used.',
             '- Use saved reported UNIFAC predictions, including the physical x≤1 bound. After correction, cap x again at 1. '
             'This is statistical prediction calibration; activity coefficients and phase-equilibrium models are not modified or re-solved.',
             '- Infinite-dilution and finite dilute scenarios use both observed mole fractions <0.05 or <0.01. '
             'This is retrospective selection and cannot use an unknown target’s measured solubility prospectively.',
             '- Unseen solvents or disconnected training solvent components are unavailable. Comparisons require all ten repeats to be supported and use identical cases within each table row.',
             '- Primary metric: mean absolute log10 error, first averaged within each molecule and then equally over molecules. '
             'Score each split’s prediction before averaging losses; do not ensemble predictions before scoring.',
             f'- Resample whole test molecules and split repetitions {summary["bootstrap_draws"]:,} times, seed {summary["seed"]}. '
             f'Simultaneous intervals account for {FAMILY_SIZE} primary corrected-versus-matched-baseline comparisons '
             '(six variants × five prespecified scenarios × two correction forms). Negative differences favor corrected UNIFAC.',
             '- Shared-case cohorts and comparisons to the broad baseline/uncorrected model are additional diagnostics; their displayed intervals use the same conservative family size. '
             'They are not an additional globally adjusted family of all exploratory comparisons.',
             '- Confidence intervals condition on the fitted cross-validation predictions and do not prove parameter-training disjointness or independence between chemical scaffolds.', '']
    for scenario in SCENARIOS:
        lines += [f'## {scenario}: relative solvent correction, full usable coverage','',
                  *result_table(summary['results'][scenario],'relative_solvent','own_coverage'),'']
        for cohort in ('common_all_six','common_five'):
            lines += [f'## {scenario}: relative solvent correction, {cohort}','',
                      *result_table(summary['results'][scenario],'relative_solvent',cohort),'']
        lines += [f'## {scenario}: target-only correction sensitivity','',
                  *result_table(summary['results'][scenario],'target_only','own_coverage'),'']
    lines += ['## Numerical checks','', '```json',json.dumps(summary['checks'],indent=2),'```','',
              '## Interpretation limits','',
              'Coverage remains variant-dependent. Own-coverage rows use different compounds across variants; shared-case tables isolate that difference. '
              'Training-domain matching is necessary because the original broad baseline can be miscalibrated on a narrower UNIFAC-supported chemical population. '
              'Unknown pH, apparent versus intrinsic aqueous solubility, solid forms and possible overlap with historical UNIFAC parameter-training data remain limitations. '
              'Calibration is evaluated on other molecules from the same dataset, not on a separately collected external study.','',
              '## Files','',
              '- One .npz per scenario/variant stores every held-out corrected and matched-baseline error, with case IDs.',
              '- *_coefficients.csv stores every fold’s relative correction, target-only correction and matched-baseline solvent factors.',
              '- summary.json contains MAE, RMSE, signed bias, fold error, factor-2/10 accuracy and paired intervals.',
              '- audit.json records fold training/support counts and normal-equation residuals.',
              '- manifest.json stores portable source/input fingerprints. Previous results, datasets and production code are unchanged.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    parser.add_argument('--results-dir',type=Path,default=RESULTS_DIR/'unifac-transfer-results')
    parser.add_argument('--baseline-dir',type=Path,default=RESULTS_DIR/'solvent-multiplier-results')
    parser.add_argument('--infinite-dir',type=Path,default=RESULTS_DIR/'infinite-dilution-assessment')
    parser.add_argument('--output-dir',type=Path,default=RESULTS_DIR/'bias-corrected-transfer-results')
    parser.add_argument('--bootstrap-draws',type=int,default=20000)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--seed',type=int,default=20261014)
    args = parser.parse_args()
    if not 1<=args.workers<=5 or args.bootstrap_draws<1000:
        parser.error('Use 1–5 workers and at least 1,000 bootstrap draws')
    started = time.monotonic()
    rdBase.DisableLog('rdApp.*')
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    checks = self_check()
    print('Calibration recovery, fair-baseline identity and leakage checks passed',flush=True)
    cases,_,records,molecules,solvents,_ = reconstruct(args.data_dir,args.results_dir)
    verify_manifest(args.baseline_dir/'manifest.json')
    verify_manifest(args.infinite_dir/'manifest.json')
    with np.load(args.baseline_dir/'baseline_errors.npz') as saved:
        assert np.array_equal(saved['case'],cases['case'])
        broad = saved['capped']
    with np.load(args.infinite_dir/'infinite_errors.npz') as saved:
        assert np.array_equal(saved['case'],cases['case'])
        infinite = saved['capped']
    split_table = pd.read_csv(args.baseline_dir/'folds.csv')
    repeats = int(split_table.repeat.max())+1
    folds = np.full((repeats,len(molecules)),-1,dtype=np.int16)
    for row in split_table.itertuples():
        assert molecules[row.index]['inchi']==row.inchi
        folds[row.repeat,row.index] = row.fold
    assert np.all(folds>=0)
    anchor = next(row['index'] for row in solvents if row['name']=='water')
    names = [row['name'] for row in solvents]
    tasks = []
    for scenario in SCENARIOS:
        cutoff = None if scenario=='finite_all' else float(scenario.split('_')[1])
        for method_index,method in enumerate(METHODS):
            rows = records[(records['scope']==0)&(records['method']==method_index)&(records['status']==0)]
            positions = np.searchsorted(cases['case'],rows['case'])
            if cutoff is not None:
                keep = (cases['log_reference_x'][positions]<np.log10(cutoff)) & (cases['log_target_x'][positions]<np.log10(cutoff))
                keep &= np.isfinite(infinite[method_index,positions])
                rows,positions = rows[keep].copy(),positions[keep]
            else:
                rows = rows.copy()
            if scenario.startswith('infinite'):
                rows['error_log10'] = infinite[method_index,positions]
            order = np.argsort(rows['case'])
            rows,positions = rows[order],positions[order]
            tasks.append((scenario,method,rows,cases[positions],folds,len(solvents),anchor,names,str(output)))
    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(calibrate,task):task[:2] for task in tasks}
        for count,future in enumerate(as_completed(futures),1):
            futures.pop(future)
            job = future.result()
            jobs.append(job)
            print(f'Calibrated {count}/{len(tasks)}: {job["scenario"]} {job["method"]} in {time.monotonic()-started:.1f}s',flush=True)
    (output/'audit.json').write_text(json.dumps(jobs,indent=2)+'\n')
    cohorts = {}
    for scenario in SCENARIOS:
        cohorts[scenario] = {mode:{} for mode in MODES}
        supported = {mode:{} for mode in MODES}
        for method in METHODS:
            with np.load(output/(scenario+'_'+method+'.npz')) as saved:
                p = np.searchsorted(cases['case'],saved['case'])
                for mode in MODES:
                    valid = np.isfinite(saved[mode]).all(axis=1)&np.isfinite(saved['matched_baseline']).all(axis=1)
                    valid &= np.isfinite(broad[p]).all(axis=1)
                    supported[mode][method] = saved['case'][valid]
        for mode in MODES:
            all_six = supported[mode][METHODS[0]]
            five = all_six
            for method in METHODS[1:]:
                all_six = np.intersect1d(all_six,supported[mode][method])
                if method!='UNIFLBY':
                    five = np.intersect1d(five,supported[mode][method])
            cohorts[scenario][mode]['common_all_six'] = all_six
            cohorts[scenario][mode]['common_five'] = five
    evaluation = []
    for scenario in SCENARIOS:
        for method in METHODS:
            selected = {mode:dict(cohorts[scenario][mode]) for mode in MODES}
            for mode in MODES:
                with np.load(output/(scenario+'_'+method+'.npz')) as saved:
                    selected[mode]['own_coverage'] = saved['case']
                if method=='UNIFLBY':
                    selected[mode].pop('common_five')
            evaluation.append((scenario,method,output/(scenario+'_'+method+'.npz'),selected))
    results = {scenario:{} for scenario in SCENARIOS}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_scoring,
                             initargs=(cases,broad,args.seed,args.bootstrap_draws)) as pool:
        futures = {pool.submit(evaluate,task):task[:2] for task in evaluation}
        for count,future in enumerate(as_completed(futures),1):
            futures.pop(future)
            scenario,method,value = future.result()
            results[scenario][method] = value
            print(f'Scored {count}/{len(evaluation)}: {scenario} {method} in {time.monotonic()-started:.1f}s',flush=True)
    results = {scenario:{method:results[scenario][method] for method in METHODS} for scenario in SCENARIOS}
    summary = dict(seed=args.seed,bootstrap_draws=args.bootstrap_draws,folds=int(folds.max())+1,repeats=repeats,
                   primary_comparison_family_size=FAMILY_SIZE,checks=checks,results=results,
                   elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    dependencies = [Path(__file__),Path(__file__).with_name('benchmark_solvent_transfer_multipliers.py'),
                    Path(__file__).with_name('benchmark_solubility_transfer.py'),args.results_dir/'manifest.json',
                    args.results_dir/'metric_records.npy',args.baseline_dir/'manifest.json',
                    args.baseline_dir/'folds.csv',args.baseline_dir/'baseline_errors.npz',
                    args.infinite_dir/'manifest.json',args.infinite_dir/'infinite_errors.npz',
                    *[args.data_dir/name for name in ('BigSolDBv2.0.csv','curated-solubility-dataset.csv','BigSolDBv2.0_densities.csv')]]
    write_manifest(output/'manifest.json',vars(args),dependencies)
    render_report(output,summary)
    print('\n'+'\n'.join(result_table(results['finite_all'],'relative_solvent','own_coverage')),flush=True)
    print(f'Complete: {output}; elapsed {summary["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    main()
