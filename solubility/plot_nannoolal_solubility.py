"""Render saved Nannoolal feature accuracy and coverage; never refit models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from solubility_paths import RESULTS_DIR, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,
                        default=RESULTS_DIR/'nannoolal-aqueous-regression')
    args = parser.parse_args()
    root = args.results_dir
    verify_manifest(root/'manifest.json')
    for suffix in ('png','svg'):
        if (root/f'comparison.{suffix}').exists():
            raise FileExistsError(root/f'comparison.{suffix}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads((root/'summary.json').read_text())
    scores = summary['scores']['common_all']
    names = ['element_counts_adaptive','UNIFAC_subgroup_adaptive','UNIFDMD_subgroup_adaptive',
             'UNIFNIST_subgroup_adaptive','UNIFLBY_subgroup_adaptive',
             'Nannoolal_first_adaptive','Nannoolal_corrected_adaptive']
    labels = ['Element counts','Original UNIFAC','Dortmund','NIST','Lyngby',
              'Nannoolal first order','Nannoolal + corrections']
    values = [scores[name]['common_valid']['MAE_log10'] for name in names]
    n = scores[names[0]]['common_valid']['solutes']
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    positions = np.arange(len(names))
    colors = ['#777777']+['#428d83']*4+['#2957a4']*2
    axes[0].barh(positions,values,color=colors)
    axes[0].set_yticks(positions,labels)
    axes[0].invert_yaxis()
    for y,value in zip(positions,values):
        axes[0].text(value+.006,y,f'{value:.3f}',va='center',fontsize=9)
    axes[0].set_xlim(0,max(values)*1.15)
    axes[0].set_xlabel('Held-out mean absolute log10 solubility error')
    axes[0].set_title(f'Identical {n:,}-molecule population\nNested size/ridge selection; valid in every repeat')
    # The pairwise populations measure overlap, not each UNIFAC's full domain.
    # Retrieve full UNIFAC neutral-domain counts from the prepared feature set.
    manifest = json.loads((root/'manifest.json').read_text())
    prepared = (root/manifest['arguments']['prepared_dir']).resolve()
    import pandas as pd
    molecules = pd.read_csv(prepared/'molecules.csv')
    with np.load(prepared/'features.npz') as features:
        clean = molecules.neutral_single.to_numpy(dtype=bool)&(features['elements'][:,5]>0)
        coverage = [int((clean&features[v+'_available']).sum()) for v in ('UNIFAC','UNIFDMD','UNIFNIST','UNIFLBY')]
    coverage.append(summary['coverage']['Nannoolal_available'])
    coverage_labels = ['Original UNIFAC','Dortmund','NIST','Lyngby','Nannoolal']
    axes[1].barh(np.arange(5),coverage,color=['#428d83']*4+['#2957a4'])
    axes[1].set_yticks(np.arange(5),coverage_labels)
    axes[1].invert_yaxis()
    for y,value in enumerate(coverage):
        axes[1].text(value+60,y,f'{value:,}',va='center',fontsize=9)
    axes[1].set_xlim(0,len(molecules)*.85)
    axes[1].set_xlabel('Neutral organic molecules with structural assignments')
    axes[1].set_title(f'Coverage among {int(clean.sum()):,} eligible molecules\nNo property-value or interaction requirements')
    for axis in axes:
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='x',alpha=.2)
    fig.suptitle('Absolute aqueous solubility: Nannoolal versus UNIFAC group features',fontsize=13)
    for suffix in ('png','svg'):
        fig.savefig(root/f'comparison.{suffix}',dpi=180)
    plt.close(fig)
    print(f'Figures saved in {root}')


if __name__=='__main__':
    main()
