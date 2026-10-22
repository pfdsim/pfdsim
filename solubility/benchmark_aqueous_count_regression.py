"""Predict absolute room-temperature aqueous log solubility from counts.

OLS baselines and nested-validation ridge/size models use only molecular atom
or UNIFAC group counts. Group regressions need no interaction parameters.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time

import numpy as np
from rdkit import Chem, rdBase
from scipy.linalg import svd

from benchmark_solubility_transfer import chemical, model, write_csv
from benchmark_solvent_transfer_multipliers import averages, difference_bootstrap
from solubility_dependencies import SHARED_FILES
from solubility_paths import DATA_DIR, RESULTS_DIR, ROOT, write_manifest
from unifac import get_unifac_groups

FAMILIES = (
    ('UNIFAC',('UNIFAC','UNIFAC2')),
    ('UNIFDMD',('UNIFDMD','UNIFM2')),
    ('UNIFNIST',('UNIFNIST',)),
    ('UNIFLBY',('UNIFLBY',)),
)
ALPHAS = np.array([0.,.01,.1,1.,10.,100.,1000.])
SIZE_FORMS = ('linear','log','surface','quadratic','flexible')
AUGMENTATIONS = ('none','log','surface','quadratic')


def data_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def prepare_water(data_dir):
    compounds = {}
    aq = defaultdict(list)
    big = defaultdict(list)
    audit = Counter()
    for row in data_rows(data_dir/'curated-solubility-dataset.csv'):
        identity = chemical(row['SMILES'],row['InChI'])
        if identity is None:
            audit['AqSolDB_invalid_identity_excluded'] += 1
            continue
        value = float(row['Solubility'])
        if not np.isfinite(value):
            audit['AqSolDB_invalid_value_excluded'] += 1
            continue
        key,smiles,neutral_single = identity
        compounds.setdefault(key,dict(inchi=key,smiles=smiles,name=row['Name'],neutral_single=neutral_single))
        aq[key].append(value)
        audit['AqSolDB_input_rows'] += 1
        if Chem.MolFromSmiles(row['SMILES']) is None:
            audit['AqSolDB_InChI_recoveries'] += 1
    for row in data_rows(data_dir/'BigSolDBv2.0.csv'):
        if row['Solvent'].strip().casefold()!='water' or not 293.15<=float(row['Temperature_K'])<=303.15:
            continue
        assert row['SMILES_Solvent']=='O'
        identity = chemical(row['SMILES_Solute'])
        if identity is None or not row['LogS(mol/L)']:
            audit['BigSolDB_water_invalid_identity_or_value_excluded'] += 1
            continue
        value = float(row['LogS(mol/L)'])
        if not np.isfinite(value):
            continue
        key,smiles,neutral_single = identity
        compounds.setdefault(key,dict(inchi=key,smiles=smiles,name=row['Compound_Name'],neutral_single=neutral_single))
        big[key,row['Source'],float(row['Temperature_K'])].append(value)
        audit['BigSolDB_room_temperature_water_rows'] += 1
    by_source = defaultdict(list)
    for (key,source,temperature),values in big.items():
        by_source[key,source].append((temperature,float(np.median(values))))
    by_compound = defaultdict(list)
    for (key,_),values in by_source.items():
        distance = min(abs(t-298.15) for t,_ in values)
        nearest = [(t,y) for t,y in values if abs(abs(t-298.15)-distance)<1e-9]
        by_compound[key].append(float(np.median([y for _,y in nearest])))
    big_values = {key:float(np.median(values)) for key,values in by_compound.items()}
    rows = []
    for key,metadata in sorted(compounds.items()):
        row = dict(metadata,index=len(rows))
        row['source'] = 'AqSolDB' if key in aq else 'BigSolDB_only'
        row['logS_mol_L'] = float(np.median(aq[key])) if key in aq else big_values[key]
        row['AqSolDB_logS'] = float(np.median(aq[key])) if key in aq else ''
        row['BigSolDB_logS'] = big_values.get(key,'')
        rows.append(row)
    overlap = sorted(set(aq)&set(big_values))
    difference = np.array([float(np.median(aq[key]))-big_values[key] for key in overlap])
    audit.update(compounds=len(rows),AqSolDB_unique=len(aq),BigSolDB_unique=len(big_values),
                 overlapping_compounds=len(overlap),BigSolDB_only=len(set(big_values)-set(aq)))
    audit = dict(audit)
    audit['overlap_median_absolute_logS_difference'] = float(np.median(np.abs(difference)))
    audit['overlap_absolute_logS_difference_95pct'] = float(np.quantile(np.abs(difference),.95))
    return rows,audit


def featurize_batch(batch):
    rdBase.DisableLog('rdApp.*')
    output = []
    for index,smiles in batch:
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        elements = np.bincount([a.GetAtomicNum() for a in molecule.GetAtoms()],minlength=119)[1:]
        assert len(elements)==118
        results = {}
        for representative,variants in FAMILIES:
            try:
                raw = get_unifac_groups('structure:'+smiles,smiles=smiles,variant=representative)
            except (ValueError,RuntimeError) as error:
                for variant in variants:
                    results[variant] = dict(status='fragmentation_unavailable',reason=str(error))
                continue
            for variant in variants:
                try:
                    m = model(variant)
                    subgroups = m._resolve_groups(raw)
                    main = Counter()
                    for group,count in subgroups.items():
                        main[m.subgroups[group].main_group] += count
                    results[variant] = dict(status='ok',subgroups=subgroups,main_groups=dict(main))
                except (ValueError,KeyError) as error:
                    results[variant] = dict(status='group_table_unavailable',reason=str(error))
        output.append((index,elements,results))
    return output


def feature_matrices(rows,features):
    elements = np.array([features[i][0] for i in range(len(rows))],dtype=float)
    total = elements.sum(axis=1)
    heavy = elements[:,1:].sum(axis=1)
    matrices,coverage,labels,aliases = {},{}, {},{}
    for representative,variants in FAMILIES:
        unique = []
        for variant in variants:
            available = np.array([features[i][1][variant]['status']=='ok' for i in range(len(rows))])
            for level in ('subgroups','main_groups'):
                ids = sorted({group for i in np.flatnonzero(available) for group in features[i][1][variant][level]})
                matrix = np.zeros((len(rows),len(ids)))
                lookup = {group:j for j,group in enumerate(ids)}
                for i in np.flatnonzero(available):
                    for group,count in features[i][1][variant][level].items():
                        matrix[i,lookup[group]] = count
                matrices[variant,level] = matrix
                if level=='subgroups':
                    labels[variant,level] = [f'{g}:{model(variant).subgroups[g].name}' for g in ids]
                else:
                    names = {s.main_group:s.main_group_name for s in model(variant).subgroups.values()}
                    labels[variant,level] = [f'{g}:{names[g]}' for g in ids]
            coverage[variant] = available
            equivalent = next((v for v in unique if np.array_equal(coverage[v],available)
                               and all([name.split(':',1)[0] for name in labels[v,l]]==
                                       [name.split(':',1)[0] for name in labels[variant,l]]
                                       and np.array_equal(matrices[v,l],matrices[variant,l])
                                       for l in ('subgroups','main_groups'))),None)
            if equivalent:
                aliases[equivalent].append(variant)
            else:
                unique.append(variant)
                aliases[variant] = [variant]
    return elements,total,heavy,matrices,coverage,labels,aliases


def size_basis(n,form):
    if form=='linear':
        return n[:,None]
    if form=='log':
        return np.log1p(n)[:,None]
    if form=='surface':
        return (n**(2/3))[:,None]
    if form=='quadratic':
        return np.column_stack([n,n*n])
    if form=='flexible':
        return np.column_stack([n,np.log1p(n),n**(2/3)])
    raise ValueError(form)


def augmented(x,heavy,form):
    if form=='none':
        return x
    return np.column_stack([x,size_basis(heavy,form)])


def ridge_grid(train,y,test,alphas):
    """Fit an unpenalized intercept; train-only scaling; one SVD per grid."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    active = scale>1e-12
    ymean = float(y.mean())
    coefficients = np.zeros((train.shape[1],len(alphas)))
    if np.any(active):
        z = (train[:,active]-mean[active])/scale[active]
        u,s,v = svd(z,full_matrices=False,check_finite=False)
        threshold = np.finfo(float).eps*max(z.shape)*s[0]
        factors = np.empty((len(s),len(alphas)))
        for j,alpha in enumerate(alphas):
            factors[:,j] = (np.divide(1,s,out=np.zeros_like(s),where=s>threshold) if alpha==0
                            else s/(s*s+alpha))
        coefficients[active] = (v.T@(factors*(u.T@(y-ymean))[:,None]))/scale[active,None]
    intercept = ymean-mean@coefficients
    prediction = test@coefficients+intercept
    unseen = np.any(np.abs(test[:,~active]-mean[~active])>1e-12,axis=1) if np.any(~active) else np.zeros(len(test),dtype=bool)
    return prediction,coefficients,intercept,unseen


def candidates(kind):
    if kind=='size_selected':
        return [(f'{base}_{form}',base,form) for base in ('heavy','total') for form in SIZE_FORMS]
    return [(form,'counts',form) for form in AUGMENTATIONS]


def candidate_matrix(x,heavy,total,kind,candidate):
    _,base,form = candidate
    return size_basis(heavy if base=='heavy' else total,form) if kind=='size_selected' else augmented(x,heavy,form)


def choose_model(x,y,heavy,total,kind,inner,alphas):
    options = candidates(kind)
    errors = np.zeros((len(options),len(alphas)))
    for i,candidate in enumerate(options):
        matrix = candidate_matrix(x,heavy,total,kind,candidate)
        for fold in np.unique(inner):
            train = inner!=fold
            validation = ~train
            prediction,_,_,_ = ridge_grid(matrix[train],y[train],matrix[validation],alphas)
            errors[i] += np.sum((prediction-y[validation,None])**2,axis=0)
    index = np.unravel_index(np.argmin(errors),errors.shape)
    return options[index[0]],float(alphas[index[1]])


def regression_job(task):
    pool,name,kind,x,y,heavy,total,ids,outer,seed,labels,output = task
    repeat_count,n_folds = len(outer),int(outer.max())+1
    predictions = np.empty((len(y),repeat_count))
    unseen = np.zeros_like(predictions,dtype=bool)
    choices = []
    for repeat in range(repeat_count):
        for fold in range(n_folds):
            train = outer[repeat]!=fold
            test = ~train
            if kind=='mean':
                predictions[test,repeat] = y[train].mean()
                continue
            candidate = ('none','counts','none')
            alpha = 0.
            if kind in ('adaptive','size_selected'):
                rng = np.random.default_rng(seed+repeat*n_folds+fold)
                inner = np.empty(np.sum(train),dtype=int)
                inner[rng.permutation(len(inner))] = np.arange(len(inner))%3
                grid = np.array([0.]) if kind=='size_selected' else ALPHAS
                candidate,alpha = choose_model(x[train],y[train],heavy[train],total[train],kind,inner,grid)
            matrix = candidate_matrix(x,heavy,total,kind,candidate) if kind in ('adaptive','size_selected') else x
            pred,b,intercept,unseen_test = ridge_grid(matrix[train],y[train],matrix[test],np.array([alpha]))
            predictions[test,repeat] = pred[:,0]
            unseen[test,repeat] = unseen_test
            train_prediction = matrix[train]@b[:,0]+intercept[0]
            choices.append(dict(repeat=repeat,fold=fold,size_basis=candidate[0],ridge_alpha=alpha,
                                training_RMSE=float(np.sqrt(np.mean((train_prediction-y[train])**2))),
                                unseen_test_molecules=int(unseen_test.sum())))
    error = predictions-y[:,None]
    score,_ = averages(error,ids)
    score['R2'] = 1-float(np.mean(error**2))/float(np.var(y))
    score['unseen_feature_molecules'] = int(np.any(unseen,axis=1).sum())
    score['choices'] = dict(Counter(c['size_basis'] for c in choices))
    score['ridge_choices'] = dict(Counter(str(c['ridge_alpha']) for c in choices))
    # Full-data coefficients are descriptive only, never used for CV scores.
    coefficient_rows = []
    if kind!='mean':
        candidate,alpha = ('none','counts','none'),0.
        if kind in ('adaptive','size_selected'):
            rng = np.random.default_rng(seed)
            inner = np.empty(len(y),dtype=int)
            inner[rng.permutation(len(y))] = np.arange(len(y))%3
            candidate,alpha = choose_model(x,y,heavy,total,kind,inner,np.array([0.]) if kind=='size_selected' else ALPHAS)
        matrix = candidate_matrix(x,heavy,total,kind,candidate) if kind in ('adaptive','size_selected') else x
        _,b,intercept,_ = ridge_grid(matrix,y,matrix,np.array([alpha]))
        feature_names = list(labels)
        if kind=='size_selected':
            feature_names = [f'{candidate[0]}_{j}' for j in range(matrix.shape[1])]
        else:
            feature_names += [f'size_{candidate[0]}_{j}' for j in range(matrix.shape[1]-len(feature_names))]
        coefficient_rows = [dict(feature='intercept',coefficient=float(intercept[0]))]
        coefficient_rows += [dict(feature=label,coefficient=float(value)) for label,value in zip(feature_names,b[:,0])]
        score['descriptive_refit'] = dict(size_basis=candidate[0],ridge_alpha=alpha)
    stem = pool+'__'+name
    destination = Path(output)
    np.savez_compressed(destination/(stem+'.npz'),index=ids,target=y,predictions=predictions,unseen=unseen)
    write_csv(destination/(stem+'_folds.csv'),choices)
    write_csv(destination/(stem+'_coefficients.csv'),coefficient_rows)
    return pool,name,score


def self_check():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40,3))
    x = np.column_stack([x,x[:,0]*2,np.ones(40)])
    y = 1.2+x[:,0]*.7-x[:,1]*.3
    test = x[:7]
    predicted,b,intercept,_ = ridge_grid(x,y,test,np.array([0.,1.]))
    assert np.allclose(predicted[:,0],y[:7],atol=1e-10)
    assert np.all(np.isfinite(b)) and np.all(np.isfinite(intercept))
    n = np.arange(1.,45.)
    size_y = 2-.3*n**(2/3)
    matrix = size_basis(n,'surface')
    prediction,_,_,_ = ridge_grid(matrix[:30],size_y[:30],matrix[30:],np.array([0.]))
    assert np.allclose(prediction[:,0],size_y[30:],atol=1e-10)
    inner = np.arange(30)%3
    candidate,alpha = choose_model(np.empty((30,0)),size_y[:30],n[:30],n[:30]*2,'size_selected',inner,np.array([0.]))
    chosen = candidate_matrix(np.empty((44,0)),n,n*2,'size_selected',candidate)
    prediction,_,_,_ = ridge_grid(chosen[:30],size_y[:30],chosen[30:],np.array([alpha]))
    assert np.allclose(prediction[:,0],size_y[30:],atol=1e-8)
    return dict(rank_deficient_OLS_prediction=True,nonlinear_size_recovery=True,
                nested_training_only_size_selection=True)


def format_table(models):
    lines = ['| Model | Molecules | MAE log10 | RMSE log10 | R² | Within factor 2 | Within factor 10 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in models.items():
        lines.append(f'| {name} | {row["solutes"]:,} | {row["MAE_log10"]:.4f} | {row["RMSE_log10"]:.4f} | '
                     f'{row["R2"]:.4f} | {100*row["within_factor_2"]:.1f}% | {100*row["within_factor_10"]:.1f}% |')
    return lines


def render_report(output,summary):
    lines = ['# Absolute room-temperature aqueous solubility from atom and UNIFAC counts','',
             'Predict log10(S / (mol/L)), with one target per chemical identity. This is absolute solubility; '
             'solid-state and ionization effects are included in the experimental target.','',
             '## Data and features','',
             '- AqSolDB supplies the curated label wherever available. Add BigSolDB-only water compounds measured at 293.15–303.15 K. '
             'For BigSolDB, median duplicates within source/temperature, use the temperature(s) nearest 298.15 K per publication, then median across publications.',
             '- Deduplicate by full standard InChI, preserving stereochemistry and salt/protonation distinctions. Recover invalid AqSolDB SMILES from its supplied InChI where possible.',
             '- Include ions and multicomponent salts in the atom-count population; preserve the entire supplied chemical identity. UNIFAC’s native fragmenter determines its own domain.',
             '- Count every element, including explicit and implicit hydrogens. Scalar size baselines use total atom count or heavy-atom count.',
             '- Test size bases N, log(1+N), N^(2/3), [N,N²], and [N,log(1+N),N^(2/3)], using heavy or total atom count. These are candidate size relationships, not a universal physical law.',
             '- Count UNIFAC subgroups and main groups with pfdsim’s authoritative fragmenter and group tables. No solvent interaction parameters, gamma calculations, melting points, logP values, or other precomputed descriptors are model features.',
             '- Compare ordinary least squares with adaptive ridge regressions augmented by optional log, surface-like, or quadratic heavy-atom size terms.',
             '- Equivalent variant feature encodings are identified after extraction and evaluated once. The aliases and coverage audit are saved.','',
             '## Validation','',
             f'- {summary["folds"]}-fold molecule CV, repeated {summary["repeats"]} times, seed {summary["seed"]}. '
             'Each chemical identity has a single target and never appears in both training and test within a split.',
             '- Select size form and ridge strength using three-fold inner CV on OUTER TRAINING data only, minimizing validation squared log error. '
             'Feature centering/scaling and removal of constant columns are fitted within each training split.',
             '- Plain OLS results are reported separately, without choosing them after seeing test performance. SVD handles dependent feature columns.',
             '- Feature columns absent/constant in training receive zero fitted slope; test extrapolations involving such features are retained and counted, not silently discarded.',
             '- Report pooled out-of-fold MAE, RMSE and R², averaging losses across repetitions. No test predictions are ensembled before scoring.',
             '- Group-coverage pools refit atom-count and size baselines on the SAME molecules, using identical outer folds. '
             'The common pool compares all distinct UNIFAC encodings on exactly the same chemical population.',
             '- Full-data coefficient refits are descriptive only. Their values do not establish individual atomic/group causal effects or statistical significance, especially with dependent features.',
             '- Labels may have unknown pH/solid form; random molecule splits do not establish unseen-scaffold or independently collected external-data accuracy.','',
             '## Dataset audit','', '```json',json.dumps(summary['data_audit'],indent=2),'```','',
             '## UNIFAC encoding coverage and aliases','', '```json',json.dumps(summary['coverage'],indent=2),'```','']
    for pool,models in summary['scores'].items():
        lines += [f'## {pool}','',*format_table(models),'']
    lines += ['## Size choices and paired comparisons','',
              'Training-fold selection frequencies are in summary.json. Paired comparisons resample entire molecules and split repetitions; '
              'their intervals use a family size covering both OLS/adaptive group-versus-element comparisons on own and common populations.','',
              '```json',json.dumps(summary['paired_comparisons'],indent=2),'```','',
              '## Files','',
              '- molecules.csv: deduplicated structures, labels and source-overlap targets.',
              '- fragmentation.csv: every variant’s success/failure per compound.',
              '- features.npz: atom and group count matrices; feature_metadata.json labels their columns.',
              '- *_folds.csv: chosen size transforms/ridge penalties and training errors for every outer fit.',
              '- *_coefficients.csv: descriptive full-data refit coefficients, with zero slopes for nonidentifiable columns.',
              '- *.npz per model: every molecule’s held-out prediction across repeats.',
              '- summary.json, manifest.json: results and portable fingerprints.',
              '- Inputs and pfdsim production implementations are unchanged.','']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    parser.add_argument('--output-dir',type=Path,default=RESULTS_DIR/'aqueous-count-regression')
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--seed',type=int,default=20261022)
    args = parser.parse_args()
    if not 1<=args.workers<=5 or args.folds<3 or args.repeats<2:
        parser.error('Use 1–5 workers, at least three folds and two repeats')
    started = time.monotonic()
    rdBase.DisableLog('rdApp.*')
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    checks = self_check()
    rows,audit = prepare_water(args.data_dir)
    write_csv(output/'molecules.csv',rows)
    print(json.dumps(audit,indent=2),flush=True)
    batches = [[(i,rows[i]['smiles']) for i in range(start,min(start+64,len(rows)))] for start in range(0,len(rows),64)]
    features = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(featurize_batch,b):i for i,b in enumerate(batches)}
        for count,future in enumerate(as_completed(futures),1):
            futures.pop(future)
            for index,elements,assignments in future.result():
                features[index] = elements,assignments
            if count%20==0 or count==len(batches):
                print(f'Featurized {len(features)}/{len(rows)} compounds in {time.monotonic()-started:.1f}s',flush=True)
    elements,total,heavy,matrices,available,labels,aliases = feature_matrices(rows,features)
    periodic = Chem.GetPeriodicTable()
    element_labels = [f'element_{z}_{periodic.GetElementSymbol(z)}' for z in range(1,119)]
    np.savez_compressed(output/'features.npz',elements=elements,total_atoms=total,heavy_atoms=heavy,
                        **{v+'_'+level:matrix for (v,level),matrix in matrices.items()},
                        **{v+'_available':mask for v,mask in available.items()})
    (output/'feature_metadata.json').write_text(json.dumps(dict(elements=element_labels,aliases=aliases,
                                                              groups={v+'_'+level:names for (v,level),names in labels.items()}),indent=2)+'\n')
    fragment_rows = [dict(index=i,variant=v,**{k:value for k,value in info.items() if k not in ('subgroups','main_groups')})
                     for i in range(len(rows)) for v,info in features[i][1].items()]
    write_csv(output/'fragmentation.csv',fragment_rows)
    rng = np.random.default_rng(args.seed)
    outer = np.empty((args.repeats,len(rows)),dtype=np.int16)
    for repeat in range(args.repeats):
        outer[repeat,rng.permutation(len(rows))] = np.arange(len(rows))%args.folds
    write_csv(output/'folds.csv',(dict(repeat=r,index=i,fold=int(outer[r,i])) for r in range(args.repeats) for i in range(len(rows))))
    pools = {'all_water':np.ones(len(rows),dtype=bool)}
    pools.update({variant:available[variant] for variant in aliases})
    pools['common_UNIFAC'] = np.logical_and.reduce([available[v] for v in aliases])
    y = np.array([r['logS_mol_L'] for r in rows])
    tasks = []
    for pool,mask in pools.items():
        ids = np.flatnonzero(mask)
        specs = [('training_mean','mean',np.empty((len(rows),0)),[]),
                 ('heavy_atoms_OLS','ols',heavy[:,None],['heavy_atoms']),
                 ('total_atoms_OLS','ols',total[:,None],['total_atoms']),
                 ('size_selected','size_selected',np.empty((len(rows),0)),[]),
                 ('element_counts_OLS','ols',elements,element_labels),
                 ('element_counts_adaptive','adaptive',elements,element_labels)]
        if pool!='all_water':
            variants = list(aliases) if pool=='common_UNIFAC' else [pool]
            for variant in variants:
                specs += [(variant+'_main_OLS','ols',matrices[variant,'main_groups'],labels[variant,'main_groups']),
                          (variant+'_subgroup_OLS','ols',matrices[variant,'subgroups'],labels[variant,'subgroups']),
                          (variant+'_subgroup_adaptive','adaptive',matrices[variant,'subgroups'],labels[variant,'subgroups'])]
        for name,kind,x,names in specs:
            tasks.append((pool,name,kind,x[ids],y[ids],heavy[ids],total[ids],ids,outer[:,ids],args.seed,names,str(output)))
    scores = {pool:{} for pool in pools}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(regression_job,task):task[:2] for task in tasks}
        for count,future in enumerate(as_completed(futures),1):
            futures.pop(future)
            name,model_name,score = future.result()
            scores[name][model_name] = score
            print(f'Fitted {count}/{len(tasks)}: {name}/{model_name} in {time.monotonic()-started:.1f}s',flush=True)
    paired = []
    family_size = 4*len(aliases)
    for variant in aliases:
        for pool in (variant,'common_UNIFAC'):
            for suffix,atomic in (('OLS','element_counts_OLS'),('adaptive','element_counts_adaptive')):
                group_name = variant+'_subgroup_'+suffix
                with np.load(output/(pool+'__'+group_name+'.npz')) as g,np.load(output/(pool+'__'+atomic+'.npz')) as a:
                    assert np.array_equal(g['index'],a['index'])
                    delta = np.abs(g['predictions']-g['target'][:,None])-np.abs(a['predictions']-a['target'][:,None])
                    comparison = difference_bootstrap(delta,args.seed,10000,requested_family_size=family_size)
                    paired.append(dict(pool=pool,group_model=group_name,atomic_model=atomic,
                                       molecules=len(g['index']),group_minus_atomic_MAE=comparison['UNIFAC_minus_baseline_MAE_log10'],
                                       CI95=comparison['paired_95pct_CI'],
                                       familywise_CI95=comparison['paired_requested_family_95pct_CI']))
    scores = {pool:{task[1]:scores[pool][task[1]] for task in tasks if task[0]==pool} for pool in pools}
    coverage = {variant:dict(aliases=names,molecules=int(available[variant].sum()),
                            subgroups=matrices[variant,'subgroups'].shape[1],main_groups=matrices[variant,'main_groups'].shape[1])
                for variant,names in aliases.items()}
    summary = dict(folds=args.folds,repeats=args.repeats,seed=args.seed,checks=checks,data_audit=audit,
                   coverage=coverage,common_molecules=int(pools['common_UNIFAC'].sum()),
                   scores=scores,paired_comparisons=paired,elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_manifest(output/'manifest.json',vars(args),[
        Path(__file__),Path(__file__).with_name('benchmark_solubility_transfer.py'),
        Path(__file__).with_name('benchmark_solvent_transfer_multipliers.py'),
        *[ROOT/path for path in SHARED_FILES],
        args.data_dir/'BigSolDBv2.0.csv',args.data_dir/'curated-solubility-dataset.csv'])
    render_report(output,summary)
    print('\n'+'\n'.join(format_table(scores['all_water'])),flush=True)
    print(f'Complete: {output}; elapsed {summary["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    main()
