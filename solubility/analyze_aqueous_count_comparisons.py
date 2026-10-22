"""Paired molecule-level uncertainty for the saved count-regression studies.

Never refit models: use each saved out-of-fold prediction and its original
repeat index. Neutral-domain results are the main molecular-domain comparison;
the unfiltered release is a separately reported sensitivity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_solubility_transfer import write_csv
from benchmark_solvent_transfer_multipliers import difference_bootstrap
from solubility_paths import RESULTS_DIR, verify_manifest, write_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=RESULTS_DIR/'aqueous-count-domain-check')
    parser.add_argument('--output-dir',type=Path,default=RESULTS_DIR/'aqueous-count-comparisons')
    parser.add_argument('--draws',type=int,default=20000)
    args=parser.parse_args()
    verify_manifest(args.results_dir/'manifest.json')
    summary=json.loads((args.results_dir/'summary.json').read_text())
    comparisons=[]
    for pool,models in summary['scores'].items():
        comparisons += [(pool,'size_selected','heavy_atoms_OLS'),
                        (pool,'element_counts_OLS','size_selected'),
                        (pool,'element_counts_adaptive','element_counts_OLS')]
        group_models=[name for name in models if name.endswith('_subgroup_adaptive')]
        for name in group_models:
            prefix=name.removesuffix('_subgroup_adaptive')
            comparisons += [(pool,prefix+'_subgroup_OLS','element_counts_OLS'),
                            (pool,name,'element_counts_adaptive'),
                            (pool,name,prefix+'_subgroup_OLS')]
        if pool=='neutral_common':
            comparisons += [(pool,'UNIFNIST_subgroup_adaptive','UNIFDMD_subgroup_adaptive'),
                            (pool,'UNIFDMD_subgroup_adaptive','UNIFAC_subgroup_adaptive'),
                            (pool,'UNIFNIST_subgroup_adaptive','UNIFAC_subgroup_adaptive')]
    destination=args.output_dir.resolve()
    destination.mkdir(parents=True,exist_ok=False)
    results=[]
    input_files=[args.results_dir/'manifest.json',args.results_dir/'summary.json']
    for pool,first,second in comparisons:
        a_path=args.results_dir/(pool+'__'+first+'.npz')
        b_path=args.results_dir/(pool+'__'+second+'.npz')
        with np.load(a_path) as a,np.load(b_path) as b:
            assert np.array_equal(a['index'],b['index'])
            assert np.array_equal(a['target'],b['target'])
            first_error=a['predictions']-a['target'][:,None]
            second_error=b['predictions']-b['target'][:,None]
            differences=np.abs(first_error)-np.abs(second_error)
            boot=difference_bootstrap(differences,summary['seed'],args.draws,
                                      requested_family_size=len(comparisons))
            ci=boot['paired_requested_family_95pct_CI']
            results.append(dict(pool=pool,first=first,second=second,molecules=len(a['index']),
                                MAE_first=float(np.abs(first_error).mean()),
                                MAE_second=float(np.abs(second_error).mean()),
                                MAE_difference=boot['UNIFAC_minus_baseline_MAE_log10'],
                                paired_CI95=boot['paired_95pct_CI'],familywise_CI95=ci,
                                first_convincingly_better=ci[1]<0,
                                first_convincingly_worse=ci[0]>0))
        input_files.extend((a_path,b_path))
    output=dict(comparisons=results,comparison_family_size=len(comparisons),
                bootstrap_draws=args.draws,seed=summary['seed'])
    (destination/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    write_csv(destination/'comparisons.csv',results)
    write_manifest(destination/'manifest.json',vars(args),[
        Path(__file__),Path(__file__).with_name('benchmark_solvent_transfer_multipliers.py'),*input_files])
    lines=['# Paired comparisons of aqueous count regressions','',
           'Use saved held-out errors on identical molecules/folds. Negative differences favor the first model. '
           f'Bootstrap whole molecules and split repetitions {args.draws:,} times, seed {summary["seed"]}; '
           f'familywise intervals account for all {len(comparisons)} comparisons below. '
           'This assesses random-molecule generalization within this dataset, not unseen-scaffold or independently collected external data.','',
           '| Population | First model | Second model | Molecules | MAE difference | Familywise 95% CI |',
           '|---|---|---|---:|---:|---|']
    for row in results:
        lo,hi=row['familywise_CI95']
        lines.append(f'| {row["pool"]} | {row["first"]} | {row["second"]} | {row["molecules"]:,} | '
                     f'{row["MAE_difference"]:+.4f} | {lo:+.4f} to {hi:+.4f} |')
    lines += ['','Regularization and size augmentation are selected jointly for the adaptive count models. '
              'Their advantage over OLS does not isolate the effect of the nonlinear size term alone. '
              'Model choices used inner training CV; the paired statistics do not select or refit models.','']
    (destination/'REPORT.md').write_text('\n'.join(lines))
    print(f'Analyzed {len(comparisons)} paired comparisons: {destination}')
    for row in results:
        if row['pool'] in ('neutral_single_organic','neutral_common'):
            print(json.dumps(row))


if __name__=='__main__':
    main()
