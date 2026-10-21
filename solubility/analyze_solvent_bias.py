"""Diagnose solvent-dependent signed biases in the saved UNIFAC assessments.

This is an analysis, not a fit of improved activity coefficients or a claim
about held-out performance after correction. It never reruns UNIFAC models.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from rdkit import rdBase
from scipy import stats

from benchmark_solubility_transfer import METHODS, prepare, write_csv
from benchmark_solvent_transfer_multipliers import fit, reconstruct
from solubility_paths import DATA_DIR, RESULTS_DIR, verify_manifest, write_manifest


def holm(values):
    """Familywise correction, valid without independent solvent tests."""
    values = np.asarray(values,dtype=float)
    order = np.argsort(values)
    adjusted = np.minimum(1,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    result = np.empty(len(values))
    result[order] = adjusted
    return result


def grouped_bias(task):
    scenario,method,role,solvent,values,n_rows,minimum,seed,draws = task
    n = len(values)
    mean = float(np.mean(values))
    row = dict(scenario=scenario,method=method,role=role,solvent_index=solvent,
               transfers=n_rows,molecules=n,bias_log10=mean,geometric_prediction_ratio=10**mean,
               median_molecule_bias_log10=float(np.median(values)),
               fraction_molecules_overpredicted=float(np.mean(values>0)),
               eligible_for_inference=n>=minimum)
    if n<minimum:
        return row
    standard_error = float(np.std(values,ddof=1)/math.sqrt(n))
    if standard_error == 0:
        p = 1.0 if mean==0 else 0.0
    else:
        p = float(2*stats.t.sf(abs(mean/standard_error),df=n-1))
    critical = float(stats.t.ppf(.975,df=n-1))
    rng = np.random.default_rng(seed+solvent)
    boot = np.empty(draws)
    for start in range(0,draws,200):
        stop = min(start+200,draws)
        boot[start:stop] = np.mean(values[rng.integers(0,n,(stop-start,n))],axis=1)
    positives = int(np.sum(values>0))
    nonzero = int(np.sum(values!=0))
    sign_p = float(stats.binomtest(positives,nonzero,.5).pvalue) if nonzero else 1.0
    row.update(standard_error_log10=standard_error,p_mean=p,p_sign=sign_p,
               t_CI95_low=mean-critical*standard_error,t_CI95_high=mean+critical*standard_error,
               bootstrap_CI95_low=float(np.quantile(boot,.025)),
               bootstrap_CI95_high=float(np.quantile(boot,.975)))
    return row


def reverse_case_ids(jobs,molecules,cases):
    reverse = np.full(len(cases),-1,dtype=np.int64)
    cursor = 0
    for molecule in molecules:
        case_offset = molecule['index']*1000000
        local_offset = 0
        for bundle in jobs[molecule['inchi']]:
            n = len(bundle['observations'])
            if bundle.get('anchor_only'):
                count = n-1
            else:
                count = n*(n-1)
                for reference in range(n):
                    for target in range(n):
                        if reference==target:
                            continue
                        index = reference*(n-1)+(target if target<reference else target-1)
                        inverse = target*(n-1)+(reference if reference<target else reference-1)
                        reverse[cursor+index] = case_offset+local_offset+inverse
            local_offset += count
            cursor += count
    assert cursor==len(cases)
    paired = reverse>=0
    positions = np.searchsorted(cases['case'],reverse[paired])
    assert np.array_equal(cases['reference'][positions],cases['target'][paired])
    assert np.array_equal(cases['target'][positions],cases['reference'][paired])
    return reverse


def relative_bias(task):
    """Reference-adjusted solvent projection with compound-cluster covariance."""
    method,df,minimum,ethanol,seed,draws = task
    # Every row is one unordered, bidirectionally available pair. The odd
    # component removes the finite solver's direction-independent asymmetry.
    count = pd.concat([df[['solute','reference']].rename(columns={'reference':'solvent'}),
                       df[['solute','target']].rename(columns={'target':'solvent'})]).drop_duplicates().groupby('solvent').size()
    core = sorted(count[count>=minimum].index.tolist())
    if ethanol not in core:
        return [],dict(method=method,status='ethanol_anchor_not_supported')
    df = df[df.reference.isin(core)&df.target.isin(core)].copy()
    if not len(df):
        return [],dict(method=method,status='no_core_pairs')
    fit_count = pd.concat([df[['solute','reference']].rename(columns={'reference':'solvent'}),
                           df[['solute','target']].rename(columns={'target':'solvent'})]).drop_duplicates().groupby('solvent').size()
    mapping = {solvent:i for i,solvent in enumerate(core)}
    identities,inverse = np.unique(df.solute.to_numpy(),return_inverse=True)
    reference = df.reference.map(mapping).to_numpy()
    target = df.target.map(mapping).to_numpy()
    errors = df.error.to_numpy()
    weights = 1/np.bincount(inverse)[inverse]
    n_solutes,n_solvents = len(identities),len(core)
    matrices = np.zeros((n_solutes,n_solvents,n_solvents))
    vectors = np.zeros((n_solutes,n_solvents))
    np.add.at(matrices,(inverse,reference,reference),weights)
    np.add.at(matrices,(inverse,target,target),weights)
    np.add.at(matrices,(inverse,reference,target),-weights)
    np.add.at(matrices,(inverse,target,reference),-weights)
    np.add.at(vectors,(inverse,target),weights*errors)
    np.add.at(vectors,(inverse,reference),-weights*errors)
    normal,rhs = matrices.sum(axis=0),vectors.sum(axis=0)
    anchor = mapping[ethanol]
    beta,components,residual = fit(normal,rhs,anchor)
    connected = np.flatnonzero(components==components[anchor])
    free = connected[connected!=anchor]
    inverse_normal = np.linalg.inv(normal[np.ix_(free,free)])
    score = vectors[:,free]-np.einsum('ijk,k->ij',matrices[:,free][:,:,connected],beta[connected])
    covariance = inverse_normal@(score.T@score)@inverse_normal*(n_solutes/(n_solutes-1))
    standard_errors = np.sqrt(np.maximum(np.diag(covariance),0))
    rng = np.random.default_rng(seed)
    boot = []
    disconnected_draws = 0
    free_matrices = matrices[:,free][:,:,free]
    free_vectors = vectors[:,free]
    for _ in range(draws):
        multiplicities = np.bincount(rng.integers(0,n_solutes,n_solutes),minlength=n_solutes)
        a = np.einsum('i,ijk->jk',multiplicities,free_matrices)
        b = multiplicities@free_vectors
        try:
            if np.linalg.matrix_rank(a)<len(free):
                disconnected_draws += 1
                continue
            boot.append(np.linalg.solve(a,b))
        except np.linalg.LinAlgError:
            disconnected_draws += 1
    boot = np.array(boot)
    rows = []
    for j,index in enumerate(free):
        se = float(standard_errors[j])
        p = float(2*stats.t.sf(abs(beta[index]/se),df=n_solutes-1)) if se else float(beta[index]==0)
        bounds = np.quantile(boot[:,j],[.025,.975]) if len(boot) else [math.nan,math.nan]
        row = dict(method=method,solvent_index=core[index],bias_relative_to_ethanol_log10=float(beta[index]),
                   relative_prediction_ratio=10**float(beta[index]),
                   standard_error_cluster_robust=se,
                   bootstrap_CI95_low=float(bounds[0]),bootstrap_CI95_high=float(bounds[1]),
                   molecules=int(fit_count.get(core[index],0)),fit_molecules=n_solutes,
                   available_molecules_before_core_pair_filter=int(count[core[index]]),
                   eligible_for_inference=int(fit_count.get(core[index],0))>=minimum)
        if row['eligible_for_inference']:
            row['p_mean'] = p
        rows.append(row)
    return rows,dict(method=method,status='ok',anchor='ethanol',core_solvents=len(core),
                     connected_solvents=len(connected),fit_molecules=n_solutes,unordered_pairs=len(df),
                     bootstrap_draws=len(boot),disconnected_bootstrap_draws=disconnected_draws,
                     maximum_normal_equation_residual=residual)


def self_check():
    assert np.allclose(holm([.001,.02,.8]),[.003,.04,.8])
    truth = np.array([0.,.3,-.7])
    rows = []
    for solute in range(30):
        for a,b in ((0,1),(0,2),(1,2)):
            rows.append(dict(solute=solute,reference=a,target=b,error=truth[b]-truth[a]))
    estimated,audit = relative_bias(('synthetic',pd.DataFrame(rows),20,0,1234,20))
    assert audit['status']=='ok'
    assert all(abs(row['bias_relative_to_ethanol_log10']-truth[row['solvent_index']])<1e-12 for row in estimated)
    return dict(holm_known_result=True,relative_network_recovers_known_bias=True)


def annotate_tests(rows,group_fields):
    groups = {}
    for row in rows:
        if 'p_mean' in row:
            key = tuple(row.get(field) for field in group_fields)
            groups.setdefault(key,[]).append(row)
    for selected in groups.values():
        corrected = holm([row['p_mean'] for row in selected])
        for row,p in zip(selected,corrected):
            row['p_mean_Holm'] = float(p)
            row['significant_mean_bias_Holm_0_05'] = bool(p<.05)
            row['multiplicity_family_tests'] = len(selected)
        if 'p_sign' in selected[0]:
            corrected = holm([row['p_sign'] for row in selected])
            for row,p in zip(selected,corrected):
                row['p_sign_Holm'] = float(p)


def report(output,rows,relative,summary):
    primary = [r for r in rows if r['scenario']=='finite_all' and r['role']=='target' and r.get('significant_mean_bias_Holm_0_05')]
    lines = ['# Solvent-dependent UNIFAC prediction bias','',
             'Signed error is log10(predicted solubility / measured solubility). '
             'Negative bias means underprediction; positive bias means overprediction. '
             '10**bias is the molecule-balanced geometric prediction/measurement ratio, not the mean arithmetic relative error.','',
             '## Main protocol','',
             '- Reuse the saved same-publication, exact-temperature 20–30 °C predictions, deduplication and failure/coverage screens. No models are rerun.',
             '- For each variant and target solvent, average signed errors within each molecule first, over all its reference solvents, temperatures and publications. '
             'Then average those molecule means equally. Reference-solvent effects are calculated separately using the same rule.',
             f'- Require at least {summary["minimum_molecules"]} distinct molecules for significance tests. Sparse solvent results are retained descriptively.',
             '- Test the molecule-level mean against zero using a two-sided one-sample t test. Bootstrap whole molecule means for descriptive 95% intervals. '
             'The t test assumes independent molecules and uses an estimated standard error; small or heavy-tailed samples should be interpreted cautiously.',
             '- Apply Holm familywise correction across ALL tested solvents, both roles, and all six variants, separately for each prespecified scenario. '
             'A significant result requires corrected p<0.05. Bootstrap intervals are descriptive ordinary 95% intervals and are not multiplicity-adjusted.',
             '- Also report the median molecule error, fraction of molecules overpredicted, and exact sign-test p values with their own Holm correction. '
             'These distinguish a mean shift driven by large errors from consistent directional bias across molecules.',
             '- Marginal solvent bias depends on the tested molecules and reference-solvent mixture. It does not prove an intrinsic error in one UNIFAC interaction parameter.','',
             '## Significant target-solvent biases, finite concentration','',
             '| Variant | Target solvent | Molecules | Bias log10 | Geometric prediction ratio | Bootstrap 95% CI | Holm p |',
             '|---|---|---:|---:|---:|---|---:|']
    for row in sorted(primary,key=lambda r:(r['method'],r['bias_log10'])):
        lines.append(f'| {row["method"]} | {row["solvent"]} | {row["molecules"]} | {row["bias_log10"]:+.3f} | '
                     f'{row["geometric_prediction_ratio"]:.3f} | {row["bootstrap_CI95_low"]:+.3f} to '
                     f'{row["bootstrap_CI95_high"]:+.3f} | {row["p_mean_Holm"]:.3g} |')
    lines += ['','## Reference-adjusted relative solvent effects','',
              '- Use only solvent pairs where both prediction directions succeeded. Retain each unordered pair once and use '
              'd=(error_reference_to_target-error_target_to_reference)/2, removing the direction-independent part of finite-solver error.',
              '- Fit d≈b_target-b_reference by least squares, weighting each molecule equally. '
              'Retain core solvents with sufficient distinct molecules and anchor ethanol to zero.',
              '- These coefficients describe relative solvent effects. Absolute solvent errors are unidentifiable from transfer measurements. '
              'The projection adjusts for reference-solvent mixture through the shared comparison network; it does not eliminate chemical-class imbalance or solvent–solute interactions.',
              '- Test coefficients using compound-cluster robust sandwich covariance and t reference distribution. '
              'Apply Holm correction across all solvent coefficients and all six variants. Molecule-bootstrap refits provide descriptive 95% intervals.',
              '- Coefficients are diagnostic in-sample summaries. They are not used to correct predictions or establish held-out accuracy after correction.','',
              '| Variant | Solvent | Bias relative to ethanol | Geometric relative ratio | Bootstrap 95% CI | Holm p |',
              '|---|---|---:|---:|---|---:|']
    for row in sorted(relative,key=lambda r:(r['method'],r['bias_relative_to_ethanol_log10'])):
        if row.get('significant_mean_bias_Holm_0_05'):
            lines.append(f'| {row["method"]} | {row["solvent"]} | {row["bias_relative_to_ethanol_log10"]:+.3f} | '
                         f'{row["relative_prediction_ratio"]:.3f} | {row["bootstrap_CI95_low"]:+.3f} to '
                         f'{row["bootstrap_CI95_high"]:+.3f} | {row["p_mean_Holm"]:.3g} |')
    lines += ['','## Dilution sensitivity','',
              'finite_0.05 and infinite_0.05 use identical finite-success cases with both measured mole fractions <0.05. '
              'finite_0.01 and infinite_0.01 similarly use <0.01. These subsets have different molecule populations. '
              'All per-solvent signed means, sample sizes and tests are in solvent_bias.csv.','',
              '```json',json.dumps(summary['significance_counts'],indent=2),'```','',
              '## Limits','',
              'Molecules can share chemical scaffolds and data sources, so molecule-level inference does not guarantee independence between chemical families. '
              'Aqueous measurements may be apparent rather than intrinsic neutral-species solubilities; pH is unavailable. '
              'Solid forms and some temperatures/pressures may differ despite the matching protocol. '
              'UNIFAC parameter-training overlap with underlying publications is unverified. '
              'Model-dependent availability and stability rejection select the evaluated populations. '
              'Solvent biases are therefore diagnostic of this dataset and protocol, not universal solvent constants.','',
              '## Reproducibility and files','',
              '- solvent_bias.csv: all target/reference results, including sparse descriptive entries.',
              '- molecule_solvent_bias.csv: molecule-level means underlying the marginal tests.',
              '- relative_solvent_bias.csv: reference-adjusted network coefficients.',
              '- summary.json: counts, largest effects and numerical audit.',
              '- manifest.json: exact arguments and source/result fingerprints.',
              '- Production code, source datasets, previous results and UNIFAC parameters are unchanged.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = RESULTS_DIR
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    parser.add_argument('--results-dir',type=Path,default=base/'unifac-transfer-results')
    parser.add_argument('--infinite-dir',type=Path,default=base/'infinite-dilution-assessment')
    parser.add_argument('--output-dir',type=Path,default=base/'solvent-bias-assessment')
    parser.add_argument('--minimum-molecules',type=int,default=20)
    parser.add_argument('--bootstrap-draws',type=int,default=10000)
    parser.add_argument('--relative-bootstrap-draws',type=int,default=2000)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--seed',type=int,default=20261014)
    args = parser.parse_args()
    if not 1<=args.workers<=5 or args.minimum_molecules<3:
        parser.error('Use 1–5 workers and at least three molecules per formal test')
    started = time.monotonic()
    rdBase.DisableLog('rdApp.*')
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    checks = self_check()
    cases,_,records,molecules,solvents,_ = reconstruct(args.data_dir,args.results_dir)
    jobs,_,_,_,_ = prepare(args.data_dir)
    reverse_ids = reverse_case_ids(jobs,molecules,cases)
    verify_manifest(args.infinite_dir/'manifest.json')
    with np.load(args.infinite_dir/'infinite_errors.npz') as saved:
        assert np.array_equal(saved['case'],cases['case'])
        infinite = saved['capped']
    solvent_names = {row['index']:row['name'] for row in solvents}
    ethanol = next(row['index'] for row in solvents if row['name']=='ethanol')
    marginal_tasks,molecule_rows,relative_tasks = [],[],[]
    for method_index,method in enumerate(METHODS):
        r = records[(records['method']==method_index)&(records['scope']==0)&(records['status']==0)]
        positions = np.searchsorted(cases['case'],r['case'])
        frame = pd.DataFrame(dict(case=r['case'],solute=r['solute'],
                                  reference=cases['reference'][positions],target=cases['target'][positions],
                                  error=r['error_log10'],position=positions))
        reverse = reverse_ids[positions]
        # Saved rows follow completion order, so explicitly sort before lookup.
        sorted_records = np.sort(r,order='case')
        reverse_positions = np.searchsorted(sorted_records['case'],reverse)
        valid = (reverse_positions<len(sorted_records)) & (frame['case'].to_numpy()<reverse)
        valid[valid] &= sorted_records['case'][reverse_positions[valid]]==reverse[valid]
        symmetric = frame[valid].copy()
        symmetric['error'] = (r['error_log10'][valid]-sorted_records['error_log10'][reverse_positions[valid]])/2
        relative_tasks.append((method,symmetric,args.minimum_molecules,ethanol,args.seed,args.relative_bootstrap_draws))
        scenarios = {'finite_all':frame}
        for cutoff in (.05,.01):
            mask = ((cases['log_reference_x'][positions]<math.log10(cutoff)) &
                    (cases['log_target_x'][positions]<math.log10(cutoff)) &
                    np.isfinite(infinite[method_index,positions]))
            finite = frame[mask].copy()
            dilute = finite.copy()
            dilute['error'] = infinite[method_index,positions[mask]]
            scenarios[f'finite_{cutoff}'] = finite
            scenarios[f'infinite_{cutoff}'] = dilute
        for scenario,df in scenarios.items():
            for role in ('target','reference'):
                grouped = df.groupby([role,'solute']).error.agg(['mean','size']).reset_index()
                for solvent,subset in grouped.groupby(role):
                    values = subset['mean'].to_numpy()
                    marginal_tasks.append((scenario,method,role,int(solvent),values,
                                           int(subset['size'].sum()),args.minimum_molecules,args.seed,args.bootstrap_draws))
                    molecule_rows.extend(dict(scenario=scenario,method=method,role=role,solvent_index=int(solvent),
                                              solvent=solvent_names[int(solvent)],solute_index=int(row.solute),
                                              bias_log10=float(row.mean),transfers=int(row.size))
                                         for row in subset.itertuples())
    print(f'Prepared {len(marginal_tasks)} solvent/role/variant/scenario groups',flush=True)
    rows,relative,audit = [],[],[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(grouped_bias,task):'marginal' for task in marginal_tasks}
        futures.update({pool.submit(relative_bias,task):'relative' for task in relative_tasks})
        total = len(futures)
        for count,future in enumerate(as_completed(futures),1):
            kind = futures.pop(future)
            result = future.result()
            if kind=='marginal':
                rows.append(result)
            else:
                coefficients,numerical_audit = result
                relative.extend(coefficients);audit.append(numerical_audit)
            if count%500==0 or count==total:
                print(f'Completed {count}/{total} diagnostics in {time.monotonic()-started:.1f}s',flush=True)
    annotate_tests(rows,['scenario'])
    annotate_tests(relative,[])
    for row in rows+relative:
        row['solvent'] = solvent_names[row['solvent_index']]
    rows.sort(key=lambda r:(r['scenario'],r['method'],r['role'],r['solvent']))
    relative.sort(key=lambda r:(r['method'],r['solvent']))
    write_csv(output/'solvent_bias.csv',rows)
    write_csv(output/'molecule_solvent_bias.csv',molecule_rows)
    write_csv(output/'relative_solvent_bias.csv',relative)
    counts = {}
    for scenario in sorted({row['scenario'] for row in rows}):
        counts[scenario] = {}
        for method in METHODS:
            counts[scenario][method] = {}
            for role in ('target','reference'):
                selected = [r for r in rows if r['scenario']==scenario and r['method']==method and r['role']==role]
                counts[scenario][method][role] = dict(tested=sum(r['eligible_for_inference'] for r in selected),
                                                     significant=sum(r.get('significant_mean_bias_Holm_0_05',False) for r in selected))
    summary = dict(minimum_molecules=args.minimum_molecules,seed=args.seed,bootstrap_draws=args.bootstrap_draws,
                   checks=checks,relative_numerical_audit=audit,significance_counts=counts,
                   primary_significant_target_biases=[r for r in rows if r['scenario']=='finite_all' and r['role']=='target'
                                                      and r.get('significant_mean_bias_Holm_0_05')],
                   significant_reference_adjusted_biases=[r for r in relative if r.get('significant_mean_bias_Holm_0_05')],
                   elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    files = [Path(__file__),Path(__file__).with_name('benchmark_solvent_transfer_multipliers.py'),
             args.results_dir/'metric_records.npy',args.results_dir/'manifest.json',
             args.infinite_dir/'infinite_errors.npz',args.infinite_dir/'manifest.json']
    write_manifest(output/'manifest.json', vars(args), files)
    report(output,rows,relative,summary)
    print(json.dumps(summary['significance_counts']['finite_all'],indent=2),flush=True)
    print(f'Complete: {output}; elapsed {summary["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    main()
