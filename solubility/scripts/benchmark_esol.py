"""Frozen ESOL predictions on the expanded Nannoolal evaluation population."""

from __future__ import annotations

import argparse
import ast
from collections import namedtuple
import csv
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_nannoolal_solubility import valid_predictions
from benchmark_solubility_transfer import chemical, write_csv
from benchmark_solvent_transfer_multipliers import difference_bootstrap
from solubility_paths import RESULTS_DIR, STUDY_DIR, verify_manifest, write_manifest
from summarize_nannoolal_metrics import headline_metrics

REFERENCE = STUDY_DIR / 'provenance' / 'esol-reference'
LABELS = {'ESOL_RDKit': 'ESOL, fixed RDKit coefficients',
          'ESOL_original_RDKit': 'Original ESOL coefficients + RDKit descriptors',
          'Nannoolal': 'Expanded Nannoolal, held-out'}


def reference_calculator(path):
    """Load only the pinned upstream class, avoiding its demo/training imports.

    No ESOL logic or coefficients are reimplemented. Its exact class requires
    only RDKit and namedtuple, whereas the full script imports scikit-learn
    and includes network-enabled demos and refitting routines.
    """
    tree = ast.parse(path.read_text())
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'ESOLCalculator']
    if len(classes) != 1:
        raise ValueError('Expected exactly one upstream ESOLCalculator class')
    namespace = dict(Chem=Chem, Descriptors=Descriptors, Crippen=Crippen,
                     Lipinski=Lipinski, namedtuple=namedtuple)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['ESOLCalculator']()


def public_training_identities(path):
    """Audit public Delaney calibration structures with and without stereochemistry."""
    exact, connectivity = set(), set()
    rows = failures = 0
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            rows += 1
            identity = chemical(row['SMILES'])
            if identity is None:
                failures += 1
                continue
            exact.add(identity[0])
            molecule = Chem.MolFromSmiles(identity[1])
            Chem.RemoveStereochemistry(molecule)
            connectivity.add(Chem.MolToInchi(molecule))
    return exact, connectivity, dict(rows=rows, invalid_identities=failures,
                                     unique_exact=len(exact), unique_without_stereo=len(connectivity))


def render(output, summary):
    lines = ['# Frozen ESOL versus expanded Nannoolal', '',
             'Target: absolute aqueous log10(S / (mol/L)). The requested population is '
             'exactly the 8,128 molecules valid for expanded Nannoolal in all five saved CV repeats. '
             'ESOL coefficients are frozen; no local fitting, bias correction, or model selection is performed.', '',
             '## Matched metrics', '',
             '| Population | Model | Molecules | MAE | RMSE | Predictive R² | Pearson r | Bias | Within ×10 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for population, models in summary['scores'].items():
        for name, row in models.items():
            m = row['metrics']
            lines.append(f'| {population} | {LABELS[name]} | {row["valid_all_repeats"]:,} | '
                         f'{m["MAE_log10"]:.4f} | {m["RMSE_log10"]:.4f} | {m["R2"]:.4f} | '
                         f'{m["Pearson_r"]:.4f} | {m["bias_log10"]:+.4f} | {100*m["within_factor_10"]:.1f}% |')
    lines += ['', '## Paired MAE differences: Nannoolal minus ESOL', '',
              '| Population | ESOL variant | Difference | Paired 95% interval | Familywise 95% interval |',
              '|---|---|---:|---|---|']
    for row in summary['paired']:
        lines.append(f'| {row["population"]} | {row["model"]} | {row["difference"]:+.4f} | '
                     f'{row["CI95"]} | {row["familywise_CI95"]} |')
    lines += ['', '## Descriptor and coefficient definitions', '',
              '- Main model: Pat Walters’s fixed RDKit ESOL coefficients, applied using the pinned upstream class. '
              'logS = 0.26121066137801696 − 0.0066138847738667125 MW − 0.7416739523408995 MolLogP '
              '+ 0.003451545565957996 rotors − 0.42624840441316975 aromatic proportion.',
              '- Sensitivity only: original Delaney coefficients with the same RDKit descriptors: '
              'logS = 0.16 − 0.63 MolLogP − 0.0062 MW + 0.066 rotors − 0.74 aromatic proportion. '
              'This is not an exact recreation of the original Daylight-descriptor model; '
              'the upstream implementation explicitly discourages treating it as such.',
              '- RDKit average molecular weight (MolWt, not exact mass), Wildman–Crippen MolLogP, '
              'Lipinski.NumRotatableBonds, and the upstream aromatic SMARTS match count divided '
              'by Mol.GetNumAtoms(). Normal SMILES parsing uses implicit hydrogen; any retained '
              'explicit/isotopic atoms are handled exactly as in the reference. No AddHs, '
              'neutralization, tautomer standardization, or largest-fragment substitution is applied.',
              '- Nannoolal is the saved support-aware expanded model, with five-fold molecular CV '
              'repeated five times and training-only vocabulary/size/ridge selection. '
              'ESOL is one deterministic prediction per molecule; its prediction is repeated only '
              'to align losses with the five Nannoolal CV repeats, not to create independent observations.',
              '- All score rows use identical molecules across both ESOL equations and Nannoolal. '
              'Nonfinite predictions or predictions outside [-30, 3] are invalid, never clipped; '
              'raw predictions and failure reasons are retained. Pearson correlation and predictive '
              'R² are distinct; scores use individual held-out predictions, not a CV ensemble.', '',
              '## Calibration overlap and limitations', '',
              '```json', json.dumps(summary['audit'], indent=2), '```', '',
              'The public delaney.csv accompanies the upstream coefficient-refit routine. '
              'Its intersections are reported by full standard InChI and, conservatively, after '
              'removing stereochemistry. The exclusion sensitivity removes the latter union; '
              'it does not prove external validation or exclude overlap with Delaney’s full '
              'original 2,874-molecule training set, which includes proprietary structures. '
              'Nannoolal CV training populations are unchanged; the exclusion only selects '
              'test cases. Unknown pH, solution species, solid form, and scaffold generalization '
              'remain limitations.',
              f'The paired bootstrap resamples molecules and Nannoolal repeat indices '
              f'({summary["bootstrap_draws"]:,} draws, seed {summary["seed"]}), conditioning on saved '
              'predictions. Familywise intervals cover the two frozen equations on both populations '
              '(four comparisons). They do not include uncertainty from refitting either model.', '',
              '## Plots and provenance', '',
              '![Identical-molecule prediction comparison](correlation.png)', '',
              'All five Nannoolal predictions are displayed individually; ESOL’s deterministic '
              'predictions are replicated for identical hexagon density weighting. '
              'Axes, bins, and color scale are shared; no valid points are cropped.', '',
              '- predictions.csv: every requested molecule, descriptors, both ESOL predictions, '
              'all five Nannoolal predictions, validity and public-calibration overlap.',
              '- summary.json and headline_metrics.csv: full-precision metrics, counts and uncertainty.',
              '- manifest.json: portable input/output fingerprints and RDKit version.',
              '- Reference implementation, calibration table and MIT notice: '
              '../../provenance/esol-reference/. No production code or existing predictions changed.', '',
              '[Original paper](https://doi.org/10.1021/ci034243x); '
              f'[pinned RDKit implementation](https://github.com/PatWalters/solubility/blob/{summary["reference"]["commit"]}/esol.py).', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=RESULTS_DIR / 'expanded-nannoolal-regression')
    parser.add_argument('--prepared-dir', type=Path, default=RESULTS_DIR / 'aqueous-count-regression')
    parser.add_argument('--reference-dir', type=Path, default=REFERENCE)
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR / 'esol-comparison')
    parser.add_argument('--draws', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20261023)
    args = parser.parse_args()
    if args.draws < 1000:
        parser.error('Use at least 1000 bootstrap draws')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rdBase.DisableLog('rdApp.*')
    verify_manifest(args.results_dir / 'manifest.json')
    verify_manifest(args.prepared_dir / 'manifest.json')
    with np.load(args.results_dir / 'predictions.npz') as saved:
        keep = np.all(valid_predictions(saved['expanded_adaptive']), axis=1)
        ids, target, nannoolal = saved['index'][keep], saved['target'][keep], saved['expanded_adaptive'][keep]
    previous = json.loads((args.results_dir / 'summary.json').read_text())
    if len(ids) != previous['scores']['expanded_adaptive']['valid_all_repeats']:
        raise ValueError('Expanded evaluation cohort mismatch')
    with (args.prepared_dir / 'molecules.csv').open(newline='') as stream:
        molecules = {int(row['index']): row for row in csv.DictReader(stream)}
    calculator = reference_calculator(args.reference_dir / 'esol.py')
    exact, connectivity, train_audit = public_training_identities(args.reference_dir / 'delaney.csv')
    arrays = {name: np.full(len(ids), np.nan) for name in ('ESOL_RDKit', 'ESOL_original_RDKit')}
    rows = []
    overlaps, parse_failures = [], 0
    for position, index in enumerate(ids):
        row = molecules[int(index)]
        if not np.isclose(float(row['logS_mol_L']), target[position], atol=1e-12, rtol=1e-12):
            raise ValueError(f'Target mismatch for molecule {index}')
        molecule = Chem.MolFromSmiles(row['smiles'])
        record = dict(index=int(index), name=row['name'], smiles=row['smiles'], inchi=row['inchi'],
                      measured_logS=float(target[position]), status='ok', reason='')
        for repeat in range(nannoolal.shape[1]):
            record[f'Nannoolal_repeat_{repeat}'] = float(nannoolal[position, repeat])
        try:
            if molecule is None:
                raise ValueError('RDKit could not parse the saved structure')
            desc = calculator.calc_esol_descriptors(molecule)
            record.update(MW=desc.mw, MolLogP=desc.logp, rotors=desc.rotors, aromatic_proportion=desc.ap)
            arrays['ESOL_RDKit'][position] = calculator.calc_esol(molecule)
            arrays['ESOL_original_RDKit'][position] = calculator.calc_esol_orig(molecule)
        except (ValueError, RuntimeError, ZeroDivisionError) as error:
            parse_failures += 1
            record.update(status='descriptor_failure', reason=str(error))
        record['public_training_exact_overlap'] = row['inchi'] in exact
        if molecule is not None:
            Chem.RemoveStereochemistry(molecule)
            overlap = Chem.MolToInchi(molecule) in connectivity
        else:
            overlap = False
        record['public_training_overlap_without_stereo'] = overlap
        overlaps.append(overlap)
        for name, values in arrays.items():
            record[name] = float(values[position]) if np.isfinite(values[position]) else ''
            record[name + '_valid'] = bool(valid_predictions(values[position]))
        rows.append(record)
    print(f'Calculated both frozen ESOL equations for {len(ids):,} requested molecules', flush=True)
    predictions = {name: np.repeat(values[:, None], nannoolal.shape[1], axis=1)
                   for name, values in arrays.items()}
    predictions['Nannoolal'] = nannoolal
    common = np.logical_and.reduce([np.all(valid_predictions(p), axis=1) for p in predictions.values()])
    populations = {'same_expanded_cohort': common,
                   'excluding_public_training_structures': common & ~np.array(overlaps)}
    scores, paired, metric_rows = {}, [], []
    for population, mask in populations.items():
        scores[population] = {name: headline_metrics(p[mask], target[mask], ids[mask])
                              for name, p in predictions.items()}
        for name, row in scores[population].items():
            metric_rows.append(dict(population=population, model=name, **row['metrics']))
        for name in arrays:
            delta = np.abs(nannoolal[mask] - target[mask, None]) - np.abs(predictions[name][mask] - target[mask, None])
            result = difference_bootstrap(delta, args.seed, args.draws, requested_family_size=4)
            paired.append(dict(population=population, model=name, molecules=int(mask.sum()),
                               difference=result['UNIFAC_minus_baseline_MAE_log10'],
                               CI95=result['paired_95pct_CI'],
                               familywise_CI95=result['paired_requested_family_95pct_CI']))
    audit = dict(requested_molecules=len(ids), descriptor_failures=parse_failures,
                 invalid_ESOL_predictions={name: int((~valid_predictions(values)).sum()) for name, values in arrays.items()},
                 matched_valid_molecules=int(common.sum()), public_calibration_table=train_audit,
                 exact_public_training_overlap=sum(r['public_training_exact_overlap'] for r in rows),
                 public_training_overlap_without_stereo=int(sum(overlaps)),
                 remaining_after_overlap_exclusion=int(populations['excluding_public_training_structures'].sum()))
    summary = dict(reference=json.loads((args.reference_dir / 'source.json').read_text()),
                   rdkit_version=rdBase.rdkitVersion, bootstrap_draws=args.draws, seed=args.seed,
                   audit=audit, scores=scores, paired=paired)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'predictions.csv', rows)
    write_csv(output / 'headline_metrics.csv', metric_rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    # Reuse the established figure renderer, whose labels live in its own module.
    import summarize_nannoolal_metrics as renderer
    renderer.LABELS.update(LABELS)
    renderer.plot_panels(output, 'correlation', list(predictions), target, predictions,
                         [common] * len(predictions), scores['same_expanded_cohort'],
                         f'Frozen ESOL versus held-out Nannoolal: identical {common.sum():,} molecules')
    render(output, summary)
    sources = [Path(__file__), args.results_dir / 'manifest.json', args.results_dir / 'summary.json',
               args.results_dir / 'predictions.npz', args.prepared_dir / 'manifest.json',
               args.prepared_dir / 'molecules.csv', *sorted(args.reference_dir.iterdir()),
               STUDY_DIR / 'summarize_nannoolal_metrics.py',
               STUDY_DIR / 'benchmark_solvent_transfer_multipliers.py',
               STUDY_DIR / 'benchmark_nannoolal_solubility.py',
               STUDY_DIR / 'benchmark_expanded_nannoolal.py', *sorted(output.iterdir())]
    write_manifest(output / 'manifest.json', vars(args), sources, rdkit_version=rdBase.rdkitVersion)
    print(json.dumps(dict(audit=audit, scores=scores, paired=paired), indent=2), flush=True)
    print(f'Completed {output}', flush=True)


if __name__ == '__main__':
    main()
