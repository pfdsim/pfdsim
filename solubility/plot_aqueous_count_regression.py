"""Render saved aqueous count-regression results without refitting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_aqueous_count_regression import size_basis
from solubility_paths import RESULTS_DIR,verify_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=RESULTS_DIR/'aqueous-count-domain-check')
    parser.add_argument('--prepared-dir',type=Path,default=RESULTS_DIR/'aqueous-count-regression')
    args=parser.parse_args()
    verify_manifest(args.results_dir/'manifest.json')
    verify_manifest(args.prepared_dir/'manifest.json')
    for stem in ('size_relationship','predicted_vs_measured'):
        for suffix in ('png','svg'):
            if (args.results_dir/f'{stem}.{suffix}').exists():
                raise FileExistsError(args.results_dir/f'{stem}.{suffix}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((args.results_dir/'summary.json').read_text())
    full_pool='neutral_single_organic'
    with np.load(args.results_dir/(full_pool+'__size_selected.npz')) as predictions:
        ids=predictions['index']
        y=predictions['target']
    with np.load(args.prepared_dir/'features.npz') as features:
        heavy,total=features['heavy_atoms'][ids],features['total_atoms'][ids]
    choice=summary['scores'][full_pool]['size_selected']['descriptive_refit']['size_basis']
    count_kind,form=choice.split('_',1)
    n=heavy if count_kind=='heavy' else total
    curve=np.linspace(n.min(),n.max(),400)
    coefficients=pd.read_csv(args.results_dir/(full_pool+'__size_selected_coefficients.csv')).coefficient.to_numpy()
    fitted=size_basis(curve,form)@coefficients[1:]+coefficients[0]
    fig,axis=plt.subplots(figsize=(8,5),layout='constrained')
    axis.hexbin(n,y,gridsize=55,mincnt=1,bins='log',cmap='Blues')
    axis.plot(curve,fitted,color='#b94a35',label=f'Selected full-data curve: {choice}')
    if count_kind=='heavy':
        linear=pd.read_csv(args.results_dir/(full_pool+'__heavy_atoms_OLS_coefficients.csv')).coefficient.to_numpy()
        axis.plot(curve,linear[0]+linear[1]*curve,color='#25675c',linestyle='--',label='Linear full-data curve')
    axis.set_xlabel('Heavy atom count' if count_kind=='heavy' else 'Atom count including H')
    axis.set_ylabel('Measured log10(S / mol/L)')
    axis.set_title(f'Aqueous solubility versus size · {len(ids):,} neutral organic molecules\nCurves are descriptive fits; predictive scores use held-out molecules')
    axis.legend(fontsize=9)
    axis.spines[['top','right']].set_visible(False)
    for suffix in ('png','svg'):
        fig.savefig(args.results_dir/f'size_relationship.{suffix}',dpi=180)
    plt.close(fig)
    pool='neutral_UNIFDMD'
    models=('size_selected','element_counts_adaptive','UNIFDMD_subgroup_adaptive')
    labels=('Size only','Element counts + selected size','Dortmund groups + selected size')
    saved=[]
    for name in models:
        with np.load(args.results_dir/(pool+'__'+name+'.npz')) as data:
            saved.append((data['index'],data['target'],data['predictions']))
    assert all(np.array_equal(saved[0][0],r[0]) for r in saved)
    lower=min(min(r[1].min(),r[2].min()) for r in saved)-.3
    upper=max(max(r[1].max(),r[2].max()) for r in saved)+.3
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained',sharex=True,sharey=True)
    for axis,name,label,(_,target,prediction) in zip(axes,models,labels,saved):
        axis.hexbin(np.repeat(target,prediction.shape[1]),prediction.ravel(),gridsize=45,mincnt=1,bins='log',cmap='Blues')
        axis.plot([lower,upper],[lower,upper],color='#777777',linestyle='--',linewidth=1)
        score=summary['scores'][pool][name]
        axis.set_title(f'{label}\nMAE={score["MAE_log10"]:.3f}; R²={score["R2"]:.3f}')
        axis.set_xlabel('Measured log10(S / mol/L)')
        axis.set_xlim(lower,upper)
        axis.set_ylim(lower,upper)
        axis.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Held-out predicted log10(S / mol/L)')
    fig.suptitle(f'Identical {len(saved[0][0]):,}-molecule population and folds · all five CV repeats shown',fontsize=12)
    for suffix in ('png','svg'):
        fig.savefig(args.results_dir/f'predicted_vs_measured.{suffix}',dpi=180)
    plt.close(fig)
    print(f'Figures saved in {args.results_dir}')


if __name__=='__main__':
    main()
