"""Compare Nannoolal structural counts with aqueous atom/UNIFAC regressions.

Use existing molecular labels, atom/UNIFAC features, folds and nested model
selection. Native Nannoolal groups are the only newly extracted descriptors.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from benchmark_aqueous_count_regression import regression_job
from benchmark_solubility_transfer import write_csv
from benchmark_solvent_transfer_multipliers import averages, difference_bootstrap
from solubility_paths import RESULTS_DIR, ROOT, verify_manifest, write_manifest
import nannoolal_method as nm

LOWER_LOGS = -30.
UPPER_LOGS = 3.


def extract(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('Invalid molecular structure')
    fragmentation = nm.fragment(molecule)
    first = Counter({f'group_{int(g)}':int(n) for g,n in fragmentation.groups.items()})
    # Silicon environments are stored separately by the authoritative native
    # fragmenter; retain every recorded environment without borrowing a
    # property-specific choice of silicon subgroup mapping.
    for environment in fragmentation.si_atoms:
        label = (f'Si_CH{environment["n_ch"]}_'
                 f'hal{int(environment["has_halogen"])}_O{int(environment["has_o"])}')
        first[label] += 1
    corrected = dict(first)
    corrected.update({f'correction_{int(g)}':int(n) for g,n in fragmentation.corrections.items()})
    # Confirm the native heavy-atom coverage contract including Si atoms.
    claimed = [a for _,atoms in fragmentation.instances for a in atoms]
    assert len(claimed)==len(set(claimed))
    assert len(claimed)+len(fragmentation.si_atoms)==molecule.GetNumHeavyAtoms()
    return dict(first),corrected,fragmentation.do_not_estimate or '',fragmentation.notes


def extract_batch(batch):
    rdBase.DisableLog('rdApp.*')
    output = []
    for index,smiles in batch:
        try:
            first,corrected,flag,notes = extract(smiles)
            output.append(dict(index=index,status='ok',first=first,corrected=corrected,
                               property_refusal_flag=flag,notes='; '.join(notes)))
        except (ValueError,RuntimeError) as error:
            output.append(dict(index=index,status='unavailable',reason=str(error)))
    return output


def make_matrix(extracted,key,n_rows):
    labels = sorted({g for row in extracted if row['status']=='ok' for g in row[key]})
    positions = {g:i for i,g in enumerate(labels)}
    matrix = np.zeros((n_rows,len(labels)))
    for row in extracted:
        if row['status']=='ok':
            for g,value in row[key].items():
                matrix[row['index'],positions[g]] = value
    return matrix,labels


def valid_predictions(values):
    return np.isfinite(values)&(values>=LOWER_LOGS)&(values<=UPPER_LOGS)


def self_check():
    first,_,_,_ = extract('CCO')
    assert first=={'group_1':1,'group_7':1,'group_36':1}
    _,corrected,_,_ = extract('Cc1ccccc1C')
    assert corrected['correction_127']==1
    first,corrected,_,_ = extract('Cl[SiH](Cl)Cl')
    assert sum(value for label,value in first.items() if label.startswith('Si_'))==1
    assert corrected['correction_217']==1
    assert valid_predictions(np.array([-31.,-30.,0.,3.,4.,np.nan])).tolist()==[False,True,True,True,False,False]
    return dict(published_ethanol_group_assignment=True,ortho_correction=True,
                silicon_retained=True,extreme_predictions_invalid=True)


def paired_comparison(task):
    pool,first,second,first_path,second_path,seed,draws,family_size = task
    with np.load(first_path) as a,np.load(second_path) as b:
        assert np.array_equal(a['index'],b['index'])
        assert np.array_equal(a['target'],b['target'])
        a_valid = valid_predictions(a['predictions'])
        b_valid = valid_predictions(b['predictions'])
        keep = np.all(a_valid&b_valid,axis=1)
        if not np.any(keep):
            return dict(pool=pool,first=first,second=second,molecules=0)
        first_errors = a['predictions'][keep]-a['target'][keep,None]
        second_errors = b['predictions'][keep]-b['target'][keep,None]
        differences = np.abs(first_errors)-np.abs(second_errors)
        boot = difference_bootstrap(differences,seed,draws,requested_family_size=family_size)
        first_score,_ = averages(first_errors,a['index'][keep])
        second_score,_ = averages(second_errors,b['index'][keep])
        return dict(pool=pool,first=first,second=second,molecules=int(keep.sum()),
                    excluded_invalid_molecules=int((~keep).sum()),
                    first_invalid_predictions=int((~a_valid).sum()),second_invalid_predictions=int((~b_valid).sum()),
                    first_MAE=first_score['MAE_log10'],second_MAE=second_score['MAE_log10'],
                    MAE_difference=boot['UNIFAC_minus_baseline_MAE_log10'],CI95=boot['paired_95pct_CI'],
                    familywise_CI95=boot['paired_requested_family_95pct_CI'],
                    first_convincingly_better=boot['paired_requested_family_95pct_CI'][1]<0)


def score_pool(task):
    pool,names,output,raw = task
    output = Path(output)
    saved = {}
    common_valid = None
    result = {}
    for name in names:
        with np.load(output/(pool+'__'+name+'.npz')) as archive:
            values = {key:archive[key] for key in archive.files}
        valid = valid_predictions(values['predictions'])
        molecule_valid = np.all(valid,axis=1)
        common_valid = molecule_valid.copy() if common_valid is None else common_valid&molecule_valid
        score = dict(raw[name],invalid_predictions=int((~valid).sum()),
                     invalid_prediction_fraction=float((~valid).mean()),
                     molecules_invalid_any_repeat=int((~molecule_valid).sum()))
        if np.any(molecule_valid):
            error = values['predictions'][molecule_valid]-values['target'][molecule_valid,None]
            metrics,_ = averages(error,values['index'][molecule_valid])
            metrics['R2'] = 1-float(np.mean(error**2))/float(np.var(values['target'][molecule_valid]))
            score['own_valid'] = metrics
        result[name] = score
        saved[name] = values
    for name,values in saved.items():
        error = values['predictions'][common_valid]-values['target'][common_valid,None]
        metrics,_ = averages(error,values['index'][common_valid])
        metrics['R2'] = 1-float(np.mean(error**2))/float(np.var(values['target'][common_valid]))
        result[name]['common_valid'] = metrics
    return pool,result


def table(models):
    lines = ['| Model | Common valid molecules | MAE log10 | RMSE log10 | R² | Invalid predictions / total |',
             '|---|---:|---:|---:|---:|---:|']
    for name,row in models.items():
        score = row['common_valid']
        lines.append(f'| {name} | {score["solutes"]:,} | {score["MAE_log10"]:.4f} | {score["RMSE_log10"]:.4f} | '
                     f'{score["R2"]:.4f} | {row["invalid_predictions"]} / {row["solutes"]*5} |')
    return lines


def report(output,summary):
    lines = ['# Nannoolal groups for absolute aqueous solubility','',
             'Predict log10(S / (mol/L)) from molecular counts. This compares learned regressions on structural encodings, '
             'not Nannoolal boiling-point predictions or UNIFAC activity coefficients.','',
             '## Method','',
             '- Reuse the previous 10,421 deduplicated water labels, atom/UNIFAC feature matrices and five-fold molecular splits repeated five times. '
             'The principal domain is the same neutral, single-component, carbon-containing population of 9,044 structures.',
             '- Extract Nannoolal counts using pfdsim nannoolal_method.fragment. First-order silicon environments are retained with each native recorded environment attribute.',
             '- Compare first-order group counts, and first-order counts plus native second-order structural corrections. '
             'No group-interaction pair features, estimated pure-component properties or property contribution values are used.',
             '- Native property-specific do_not_estimate flags (e.g. COOH/NH2 coexistence) are recorded but not automatically excluded: '
             'we fit aqueous-solubility coefficients independently, rather than invoking those pure-property equations. Failed structural assignments are unavailable. '
             'Native local-extension structural group 219 is retained and disclosed in the fragmentation audit; no fitted property values enter the regression.',
             '- Use the identical regression_job implementation, training-only scaling, OLS, three-fold inner CV, ridge grid and optional nonlinear size bases as the preceding atom/UNIFAC assessment. '
             'Hyperparameters are selected on training validation errors only.',
             '- Each pool refits all included regressions on exactly the same chemical identities and outer molecular folds. '
             'Pairwise UNIFAC intersections and the all-encoding common intersection control coverage differences.',
             f'- A prediction is invalid if nonfinite or outside [{LOWER_LOGS:g}, {UPPER_LOGS:g}] log10 mol/L. '
             'This deliberately generous numerical screen is not a rigorous physical bound or a fitted threshold. '
             'Do not clip predictions or delete their original outputs. Invalid counts accompany every score.',
             '- Main tables score the same molecules where EVERY model in the pool is valid in EVERY repeat. '
             'Paired comparisons require both compared models valid in every repeat and explicitly report exclusions. '
             'Conditional accuracy with exclusions is not full-population accuracy; compare invalid rates as well.',
             f'- Paired bootstrap uses {summary["bootstrap_draws"]:,} draws of whole molecules and split repetitions, seed {summary["seed"]}. '
             f'Familywise intervals cover all {summary["comparison_family_size"]} displayed comparisons. '
             'Bootstrap comparisons run in five or fewer processes; no model is selected by test performance.',
             '- Sparse descriptors absent in training receive zero slope and are counted by the shared regression implementation. '
             'Absolute solubility includes solid-state and ionization effects; random molecule CV is not scaffold-disjoint or external validation.','',
             '## Coverage','', '```json',json.dumps(summary['coverage'],indent=2),'```','']
    for pool,models in summary['scores'].items():
        lines += [f'## {pool}','',*table(models),'']
    lines += ['## Paired comparisons','',
              '| Population | First model | Second model | Molecules | MAE difference | Familywise 95% CI | Invalid molecules removed |',
              '|---|---|---|---:|---:|---|---:|']
    for row in summary['paired']:
        if not row['molecules']:
            continue
        lo,hi = row['familywise_CI95']
        lines.append(f'| {row["pool"]} | {row["first"]} | {row["second"]} | {row["molecules"]:,} | '
                     f'{row["MAE_difference"]:+.4f} | {lo:+.4f} to {hi:+.4f} | {row["excluded_invalid_molecules"]} |')
    lines += ['','## Files','',
              '- fragmentation.csv: structural assignment coverage, native property flags, local-extension notes and failure reasons.',
              '- features.npz and feature_metadata.json: first-order and corrected Nannoolal count matrices.',
              '- *_folds.csv and *_coefficients.csv: nested choices and descriptive full-data refits.',
              '- model .npz archives: all raw held-out predictions, including invalid values; prior study outputs are unchanged.',
              '- summary.json, paired_comparisons.csv and manifest.json: full scores, uncertainty and portable provenance.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',type=Path,default=RESULTS_DIR/'aqueous-count-regression')
    parser.add_argument('--output-dir',type=Path,default=RESULTS_DIR/'nannoolal-aqueous-regression')
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--bootstrap-draws',type=int,default=20000)
    args = parser.parse_args()
    if not 1<=args.workers<=5 or args.bootstrap_draws<1000:
        parser.error('Use one to five workers and at least 1,000 bootstrap draws')
    started = time.monotonic()
    rdBase.DisableLog('rdApp.*')
    checks = self_check()
    verify_manifest(args.prepared_dir/'manifest.json')
    previous = json.loads((args.prepared_dir/'summary.json').read_text())
    metadata = json.loads((args.prepared_dir/'feature_metadata.json').read_text())
    molecules = pd.read_csv(args.prepared_dir/'molecules.csv')
    with np.load(args.prepared_dir/'features.npz') as archive:
        old = {key:archive[key] for key in archive.files}
    y = molecules.logS_mol_L.to_numpy()
    clean = molecules.neutral_single.to_numpy(dtype=bool)&(old['elements'][:,5]>0)
    split = pd.read_csv(args.prepared_dir/'folds.csv')
    outer = np.full((previous['repeats'],len(y)),-1,dtype=int)
    outer[split.repeat,split['index']] = split.fold
    assert np.all(outer>=0)
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    inputs = [(int(i),molecules.smiles[i]) for i in np.flatnonzero(clean)]
    batches = [inputs[i:i+48] for i in range(0,len(inputs),48)]
    extracted = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_batch,batch):i for i,batch in enumerate(batches)}
        for count,future in enumerate(as_completed(futures),1):
            futures.pop(future)
            extracted.extend(future.result())
            if count%25==0 or count==len(batches):
                print(f'Fragmented {len(extracted)}/{len(inputs)} molecules in {time.monotonic()-started:.1f}s',flush=True)
    first,first_labels = make_matrix(extracted,'first',len(y))
    corrected,corrected_labels = make_matrix(extracted,'corrected',len(y))
    available = np.zeros(len(y),dtype=bool)
    available[[row['index'] for row in extracted if row['status']=='ok']] = True
    write_csv(output/'fragmentation.csv',({key:value for key,value in row.items() if key not in ('first','corrected')}
                                         for row in sorted(extracted,key=lambda r:r['index'])))
    np.savez_compressed(output/'features.npz',first_order=first,with_corrections=corrected,available=available)
    (output/'feature_metadata.json').write_text(json.dumps(dict(first_order=first_labels,with_corrections=corrected_labels),indent=2)+'\n')
    variants = list(metadata['aliases'])
    pools = {'Nannoolal_own':available}
    pools.update({'with_'+v:available&old[v+'_available'] for v in variants})
    pools['common_all'] = available&np.logical_and.reduce([old[v+'_available'] for v in variants])
    tasks,names = [],{}
    for population,mask in pools.items():
        ids = np.flatnonzero(mask)
        specifications = [('size_selected','size_selected',np.empty((len(y),0)),[]),
                          ('element_counts_OLS','ols',old['elements'],metadata['elements']),
                          ('element_counts_adaptive','adaptive',old['elements'],metadata['elements']),
                          ('Nannoolal_first_OLS','ols',first,first_labels),
                          ('Nannoolal_first_adaptive','adaptive',first,first_labels),
                          ('Nannoolal_corrected_OLS','ols',corrected,corrected_labels),
                          ('Nannoolal_corrected_adaptive','adaptive',corrected,corrected_labels)]
        group_variants = variants if population=='common_all' else [population.removeprefix('with_')] if population.startswith('with_') else []
        for variant in group_variants:
            specifications += [(variant+'_subgroup_OLS','ols',old[variant+'_subgroups'],metadata['groups'][variant+'_subgroups']),
                               (variant+'_subgroup_adaptive','adaptive',old[variant+'_subgroups'],metadata['groups'][variant+'_subgroups'])]
        names[population] = [name for name,_,_,_ in specifications]
        for name,kind,x,labels in specifications:
            tasks.append((population,name,kind,x[ids],y[ids],old['heavy_atoms'][ids],old['total_atoms'][ids],
                          ids,outer[:,ids],previous['seed'],labels,str(output)))
    coverage = dict(eligible_neutral_organic=int(clean.sum()),Nannoolal_available=int(available.sum()),
                    first_order_columns=len(first_labels),with_corrections_columns=len(corrected_labels),
                    native_property_refusal_flags=sum(bool(row.get('property_refusal_flag')) for row in extracted),
                    local_extension_219_molecules=sum('group_219' in row.get('first',{}) for row in extracted),
                    pool_counts={population:int(mask.sum()) for population,mask in pools.items()})
    print(json.dumps(coverage,indent=2),flush=True)
    raw = {population:{} for population in pools}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(regression_job,task):task[:2] for task in tasks}
        for count,future in enumerate(as_completed(futures),1):
            futures.pop(future)
            population,name,score = future.result()
            raw[population][name] = score
            if count%10==0 or count==len(tasks):
                print(f'Fitted {count}/{len(tasks)} models in {time.monotonic()-started:.1f}s',flush=True)
    scores = {}
    for population in pools:
        key,value = score_pool((population,names[population],str(output),raw[population]))
        scores[key] = {name:value[name] for name in names[population]}
    comparisons = []
    for population in pools:
        comparisons += [(population,'Nannoolal_first_adaptive','element_counts_adaptive'),
                        (population,'Nannoolal_corrected_adaptive','Nannoolal_first_adaptive'),
                        (population,'Nannoolal_corrected_OLS','Nannoolal_first_OLS')]
        group_variants = variants if population=='common_all' else [population.removeprefix('with_')] if population.startswith('with_') else []
        for variant in group_variants:
            for nannoolal in ('first','corrected'):
                for suffix in ('OLS','adaptive'):
                    comparisons.append((population,'Nannoolal_'+nannoolal+'_'+suffix,variant+'_subgroup_'+suffix))
    paired = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(paired_comparison,(population,a,b,output/(population+'__'+a+'.npz'),
                                                  output/(population+'__'+b+'.npz'),previous['seed'],
                                                  args.bootstrap_draws,len(comparisons))):(population,a,b)
                   for population,a,b in comparisons}
        for count,future in enumerate(as_completed(futures),1):
            futures.pop(future)
            paired.append(future.result())
            if count%10==0 or count==len(comparisons):
                print(f'Compared {count}/{len(comparisons)} model pairs in {time.monotonic()-started:.1f}s',flush=True)
    paired.sort(key=lambda row:(row['pool'],row['first'],row['second']))
    write_csv(output/'paired_comparisons.csv',paired)
    summary = dict(seed=previous['seed'],folds=previous['folds'],repeats=previous['repeats'],checks=checks,
                   bootstrap_draws=args.bootstrap_draws,comparison_family_size=len(comparisons),
                   invalid_logS_range=[LOWER_LOGS,UPPER_LOGS],coverage=coverage,scores=scores,paired=paired,
                   elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_manifest(output/'manifest.json',vars(args),[
        Path(__file__),ROOT/'nannoolal_method.py',ROOT/'physical_constants.py',
        Path(__file__).with_name('benchmark_aqueous_count_regression.py'),
        Path(__file__).with_name('benchmark_solvent_transfer_multipliers.py'),
        args.prepared_dir/'manifest.json',args.prepared_dir/'features.npz',
        args.prepared_dir/'feature_metadata.json',args.prepared_dir/'molecules.csv',args.prepared_dir/'folds.csv'])
    report(output,summary)
    print('\n'+'\n'.join(table(scores['common_all'])),flush=True)
    print(f'Complete: {output}; elapsed {summary["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    main()
