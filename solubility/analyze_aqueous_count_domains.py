"""Chemical-domain sensitivity using cached aqueous count features.

Retain the full-data benchmark, and refit on neutral, connected organic
identities. Reuse its regression implementation, features and molecule folds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from benchmark_aqueous_count_regression import format_table,regression_job
from benchmark_solubility_transfer import write_csv
from solubility_paths import RESULTS_DIR,verify_manifest,write_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',type=Path,default=RESULTS_DIR/'aqueous-count-regression')
    parser.add_argument('--output-dir',type=Path,default=RESULTS_DIR/'aqueous-count-domain-check')
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    if not 1<=args.workers<=5:
        parser.error('Use one to five workers')
    started=time.monotonic()
    verify_manifest(args.prepared_dir/'manifest.json')
    previous=json.loads((args.prepared_dir/'summary.json').read_text())
    metadata=json.loads((args.prepared_dir/'feature_metadata.json').read_text())
    molecules=pd.read_csv(args.prepared_dir/'molecules.csv')
    with np.load(args.prepared_dir/'features.npz') as saved:
        features={name:saved[name] for name in saved.files}
    y=molecules.logS_mol_L.to_numpy()
    elements,total,heavy=features['elements'],features['total_atoms'],features['heavy_atoms']
    clean=molecules.neutral_single.to_numpy(dtype=bool)&(elements[:,5]>0)
    split=pd.read_csv(args.prepared_dir/'folds.csv')
    outer=np.full((previous['repeats'],len(y)),-1,dtype=int)
    outer[split.repeat,split['index']]=split.fold
    assert np.all(outer>=0)
    variants=list(metadata['aliases'])
    pools={'neutral_single_organic':clean}
    pools.update({'neutral_'+v:clean&features[v+'_available'] for v in variants})
    pools['neutral_common']=clean&np.logical_and.reduce([features[v+'_available'] for v in variants])
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    tasks=[]
    for pool,mask in pools.items():
        ids=np.flatnonzero(mask)
        specifications=[('training_mean','mean',np.empty((len(y),0)),[]),
                        ('heavy_atoms_OLS','ols',heavy[:,None],['heavy_atoms']),
                        ('total_atoms_OLS','ols',total[:,None],['total_atoms']),
                        ('size_selected','size_selected',np.empty((len(y),0)),[]),
                        ('element_counts_OLS','ols',elements,metadata['elements']),
                        ('element_counts_adaptive','adaptive',elements,metadata['elements'])]
        groups=variants if pool=='neutral_common' else [pool.removeprefix('neutral_')] if pool!='neutral_single_organic' else []
        for variant in groups:
            specifications.extend([
                (variant+'_main_OLS','ols',features[variant+'_main_groups'],metadata['groups'][variant+'_main_groups']),
                (variant+'_subgroup_OLS','ols',features[variant+'_subgroups'],metadata['groups'][variant+'_subgroups']),
                (variant+'_subgroup_adaptive','adaptive',features[variant+'_subgroups'],metadata['groups'][variant+'_subgroups'])])
        for name,kind,x,labels in specifications:
            tasks.append((pool,name,kind,x[ids],y[ids],heavy[ids],total[ids],ids,outer[:,ids],previous['seed'],labels,str(output)))
    print('Domain molecule counts:',{p:int(m.sum()) for p,m in pools.items()},flush=True)
    scores={pool:{} for pool in pools}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(regression_job,t):t[:2] for t in tasks}
        for count,future in enumerate(as_completed(futures),1):
            futures.pop(future)
            pool,name,score=future.result()
            scores[pool][name]=score
            if count%10==0 or count==len(tasks):
                print(f'Fitted {count}/{len(tasks)} domain models in {time.monotonic()-started:.1f}s',flush=True)
    scores={pool:{t[1]:scores[pool][t[1]] for t in tasks if t[0]==pool} for pool in pools}
    write_csv(output/'domain_membership.csv',(dict(index=i,inchi=molecules.inchi[i],neutral_single_organic=bool(clean[i])) for i in range(len(y))))
    summary=dict(domain='neutral, single connected component, at least one carbon atom',
                 domain_molecules=int(clean.sum()),excluded_from_domain=int((~clean).sum()),
                 seed=previous['seed'],folds=previous['folds'],repeats=previous['repeats'],
                 scores=scores,elapsed_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_manifest(output/'manifest.json',vars(args),[
        Path(__file__),Path(__file__).with_name('benchmark_aqueous_count_regression.py'),
        args.prepared_dir/'manifest.json',args.prepared_dir/'features.npz',
        args.prepared_dir/'feature_metadata.json',args.prepared_dir/'molecules.csv',args.prepared_dir/'folds.csv'])
    lines=['# Aqueous count regressions: chemical-domain check','',
           'The full-data assessment is retained unchanged in ../aqueous-count-regression/. '
           'This companion refits the same models on neutral, single-component organic identities, '
           'defined structurally before fitting, without filtering on solubility or prediction error.','',
           f'This domain contains {int(clean.sum()):,} records; {int((~clean).sum()):,} records are outside it. '
           'The same molecular folds, nested size/ridge selection and training-only scaling are used. '
           'All group and atom baselines within a pool are refitted on identical molecules. '
           'Neutral representations can still ionize in water; unknown pH and solid form remain limitations.','']
    for pool,models in scores.items():
        lines += [f'## {pool}','',*format_table(models),'']
    lines += ['## Why retain both populations?','',
              'The broad dataset contains salts, inorganic species and disconnected mixtures. '
              'Its strongest adaptive-atom extrapolation failure is a 24-fragment, 388-heavy-atom record: '
              'several held-out predictions exceed log10(S/mol/L)=79, which is physically impossible. '
              'Such results are retained as failures of this empirical regression rather than hidden by clipping. '
              'The structural domain check distinguishes ordinary molecular prediction from that broader input population.','',
              'Full-data coefficients are descriptive refits, not held-out predictions. '
              'Original data, prior predictions, features and production code were unchanged.','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    print('\n'+'\n'.join(format_table(scores['neutral_single_organic'])),flush=True)
    print(f'Complete: {output}; elapsed {summary["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__':
    main()
