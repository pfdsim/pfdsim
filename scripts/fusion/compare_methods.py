#!/usr/bin/env python3
"""Rank saved fusion predictions on identical populations and targets.

This postprocessing benchmark does not rerun models, fit parameters, remove
numeric outliers, or use different populations for competing methods. Bootstrap
intervals resample whole compounds with an explicit seed; they describe these
datasets, not independent validation or a universal chemical population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from benchmark_fusion import compound_summary
from prepare_data import write_json

HERE = Path(__file__).resolve().parent
GENERAL = ['chickos_perry_1991', 'chickos_structure_1999',
           'dy1996_author_descriptors', 'jain_author_descriptors_2004', 'walden_54_4']
LABELS = {'chickos_perry_1991': 'Perry Chickos 1991', 'chickos_structure_1999': 'Updated Chickos 1999',
          'dy1996_author_descriptors': 'Dannenfelser 1996', 'jain_author_descriptors_2004': 'Jain 2004',
          'walden_54_4': 'Walden baseline', 'compernolle_entropy_2011': 'Acid entropy x Tm',
          'compernolle_enthalpy_2011': 'Acid direct enthalpy'}


def compound_key(row):
    return row.get('cas') or row.get('smiles') or row.get('name') or row['id']


def disjoint_test_rows(saved):
    """Use the full training identity audit frozen with these predictions."""
    training = set(saved['audit']['identity_resolved_training_CAS'])
    return [r for r in saved['rows']['paper'] if r['fit_role'] == 'test'
            and not r['paper_excluded'] and not r['has_solid_solid_transition']
            and r['cas'] not in training]


def compare(rows, methods, target, rng, bootstrap_samples):
    common = [r for r in rows if all(r['predictions'].get(m, {}).get('value') is not None for m in methods)]
    metrics = {m: compound_summary(common, m, target) for m in methods}
    if not common:
        return {'observations': 0, 'compounds': 0, 'methods': metrics}
    groups = defaultdict(list)
    for row in common:
        groups[compound_key(row)].append(row)
    errors = {m: np.asarray([statistics.fmean(
        100 * abs(r['predictions'][m]['value'] - r[target]) / r[target]
        for r in entries) for entries in groups.values()]) for m in methods}
    pairwise = {}
    n = len(groups)
    # Use the same resampled compounds for every method within this population.
    indices = rng.integers(0, n, size=(bootstrap_samples, n))
    for left, right in combinations(methods, 2):
        difference = errors[right] - errors[left]
        sampled = difference[indices].mean(axis=1)
        pairwise[f'{left} vs {right}'] = {
            'positive_difference_favors': left,
            'right_minus_left_MAPE_percentage_points': float(difference.mean()),
            'bootstrap_95_percent_interval': np.quantile(sampled, [.025, .975]).tolist(),
            'left_lower_error_compounds': int(np.sum(difference > 1e-9)),
            'right_lower_error_compounds': int(np.sum(difference < -1e-9)),
            'tied_compounds': int(np.sum(np.abs(difference) <= 1e-9)),
        }
    classes = sorted({c for r in common for c in r.get('chemical_classes', [])})
    return {'observations': len(common), 'compounds': n, 'target': target,
            'transition_status_counts': dict(Counter(r.get('transition_status', 'source-defined') for r in common)),
            'methods': metrics, 'paired_differences': pairwise,
            'chemical_classes_exploratory': {c: {m: compound_summary(
                [r for r in common if c in r['chemical_classes']], m, target) for m in methods} for c in classes}}


def format_report(report):
    lines = ['Directly comparable fusion-method rankings',
             'Every method in a panel uses identical observations, measured Tm, and target.',
             'MAPE and MAE are compound-equal; bootstrap differences are percentage points.',
             'Handbook/development data and bootstrap intervals do not establish external validation.', '']
    for label, panel in report['panels'].items():
        lines.append(f'{label}: {panel["compounds"]} compounds, {panel["observations"]} observations')
        for method, summary in panel['methods'].items():
            if not summary['n_compounds']:
                continue
            lines.append(f'  {LABELS[method]:25s} MAPE={summary["compound_equal_mape_percent"]:6.2f}% '
                         f'MAE={summary["compound_equal_mae"]:6.3f} kJ/mol '
                         f'median={summary["median_compound_mape_percent"]:6.2f}% '
                         f'P90={summary["p90_compound_mape_percent"]:6.2f}%')
        for pair, summary in panel.get('paired_differences', {}).items():
            if pair != 'chickos_perry_1991 vs chickos_structure_1999':
                continue
            low, high = summary['bootstrap_95_percent_interval']
            lines.append(f'  Updated minus Perry MAPE: {summary["right_minus_left_MAPE_percentage_points"]:+.3f} pp; '
                         f'paired bootstrap 95% [{low:+.3f}, {high:+.3f}] pp; '
                         f'Perry wins {summary["left_lower_error_compounds"]}, '
                         f'updated wins {summary["right_lower_error_compounds"]}, '
                         f'ties {summary["tied_compounds"]}')
        lines.append('')
    return '\n'.join(lines).rstrip('\n') + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'results/benchmark.json')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error('--bootstrap-samples must be at least 100')
    saved = json.loads(args.input.read_text())
    rng = np.random.default_rng(args.seed)
    panels = {}
    for dataset in ['crc_handbook', 'perry_2_68']:
        source = [r for r in saved['rows']['handbooks'] if r['dataset'] == dataset
                  and r['transition_status'] != 'reported_multiple']
        panels[f'{dataset}: all general methods common'] = compare(source, GENERAL, 'fusion_enthalpy_kJ_mol', rng, args.bootstrap_samples)
        panels[f'{dataset}: confirmed single-transition common'] = compare(
            [r for r in source if r['transition_status'] == 'single_transition_in_chickos'],
            GENERAL, 'fusion_enthalpy_kJ_mol', rng, args.bootstrap_samples)
    test = disjoint_test_rows(saved)
    panels['disjoint resolved-CAS test: all general methods common'] = compare(test, GENERAL, 'fusion_enthalpy_kJ_mol', rng, args.bootstrap_samples)
    panels['disjoint resolved-CAS test: Perry vs updated'] = compare(test, GENERAL[:2], 'fusion_enthalpy_kJ_mol', rng, args.bootstrap_samples)
    panels['disjoint resolved-CAS test: updated vs Jain'] = compare(
        test, ['chickos_structure_1999', 'jain_author_descriptors_2004'], 'fusion_enthalpy_kJ_mol', rng, args.bootstrap_samples)
    acid_methods = ['chickos_perry_1991', 'chickos_structure_1999',
                    'compernolle_entropy_2011', 'compernolle_enthalpy_2011', 'walden_54_4']
    panels['acid development: common total-enthalpy observations'] = compare(saved['rows']['acids'], acid_methods, 'total_enthalpy_kJ_mol', rng, args.bootstrap_samples)
    report = {'panels': panels, 'seed': args.seed, 'bootstrap_samples': args.bootstrap_samples,
              'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': ['Each panel is a separate population; scores across panels are not a method ranking.',
                              'Bootstrap intervals resample compounds, not underlying literature sources; shared lineage remains.',
                              'Class splits are exploratory, overlapping, and not validated switching rules.',
                              'Acid panel targets summed transition enthalpy and is development data.']}
    write_json(args.output_dir / 'comparison.json', report)
    text = format_report(report)
    path = args.output_dir / 'comparison.txt'
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        backup = Path('/tmp') / f'pfdsim-fusion-comparison-{digest}.txt.backup'
        if not backup.exists():
            shutil.copy2(path, backup)
    path.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
