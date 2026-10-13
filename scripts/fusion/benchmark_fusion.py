#!/usr/bin/env python3
"""Compare fusion estimators with measured Tm; no online property resolution.

Run: python scripts/fusion/benchmark_fusion.py
The experimental final-fusion, total-transition-entropy, and acid total-enthalpy
panels have distinct targets. All paired comparisons use identical populations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from chemicals import miscdata, phase_change
from chemicals.identifiers import search_chemical
from compound_identity import parse_formula_counts
from fusion_methods import NotApplicable, acid_estimate, chickos, descriptor_entropy, flexibility, jain_domain, molecule
from prepare_data import write_json


METHODS = ['walden_54_4', 'chickos_perry_1991', 'chickos_structure_1999', 'chickos_published_1999',
           'dy1996_author_descriptors', 'jain_author_descriptors_2004', 'compernolle_entropy_2011',
           'compernolle_enthalpy_2011']


def load_json(path):
    return json.loads(path.read_text())


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(pairs):
    """Mean absolute relative error uses each compound once, not each source row."""
    if not pairs:
        return {'n': 0}
    errors = [prediction - reference for reference, prediction in pairs]
    relative = [e / r for e, (r, _) in zip(errors, pairs, strict=True)]
    absolute_relative = [abs(e) * 100 for e in relative]
    return {'n': len(pairs), 'mae': statistics.fmean(abs(e) for e in errors),
            'rmse': math.sqrt(statistics.fmean(e * e for e in errors)),
            'mape_percent': statistics.fmean(absolute_relative),
            'median_ape_percent': statistics.median(absolute_relative),
            'p90_ape_percent': quantile(absolute_relative, .90),
            'p95_ape_percent': quantile(absolute_relative, .95),
            'bias_percent': statistics.fmean(relative) * 100,
            'within_10_percent': statistics.fmean(abs(e) <= .10 for e in relative) * 100,
            'within_20_percent': statistics.fmean(abs(e) <= .20 for e in relative) * 100}


def compound_summary(rows, method, target):
    groups = defaultdict(list)
    for row in rows:
        value = row['predictions'].get(method, {}).get('value')
        if value is None:
            continue
        key = row.get('cas') or row.get('smiles') or row.get('name') or row['id']
        groups[key].append((row[target], value))
    individual = [summarize(pairs) for pairs in groups.values()]
    if not individual:
        return {'n_compounds': 0, 'n_observations': 0}
    return {'n_compounds': len(individual), 'n_observations': sum(s['n'] for s in individual),
            'compound_equal_mae': statistics.fmean(s['mae'] for s in individual),
            'compound_equal_mape_percent': statistics.fmean(s['mape_percent'] for s in individual),
            'median_compound_mape_percent': statistics.median(s['mape_percent'] for s in individual),
            'p90_compound_mape_percent': quantile([s['mape_percent'] for s in individual], .9),
            'compound_equal_bias_percent': statistics.fmean(s['bias_percent'] for s in individual)}


def source_descriptors(rows):
    by_cas = defaultdict(list)
    for row in rows:
        if row.get('cas'):
            by_cas[row['cas']].append(row)
    result = {}
    for cas, entries in by_cas.items():
        choices = {(row['sigma'], row['tau']) for row in entries}
        if len(choices) == 1:
            result[cas] = {'sigma': entries[0]['sigma'], 'tau': entries[0]['tau'],
                           'source_rows': [row['source_row'] for row in entries]}
    return result


def dy_descriptors(rows):
    by_cas = defaultdict(list)
    for row in rows:
        if row.get('cas') and not row['polymer']:
            by_cas[row['cas']].append(row)
    return {cas: {'ln_sigma': entries[0]['ln_sigma'], 'ln_phi': entries[0]['ln_phi'],
                  'source_ids': [r['id'] for r in entries]}
            for cas, entries in by_cas.items()
            if len({(r['ln_sigma'], r['ln_phi']) for r in entries}) == 1}


def classifications(smiles):
    try:
        mol = molecule(smiles)
    except NotApplicable:
        return ['unsupported_identity']
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    groups = []
    if symbols <= {'C', 'H'}:
        groups.append('hydrocarbon')
    if symbols & {'F', 'Cl', 'Br', 'I'}:
        groups.append('halogenated')
    if 'O' in symbols:
        groups.append('oxygen_containing')
    if 'N' in symbols:
        groups.append('nitrogen_containing')
    if 'S' in symbols:
        groups.append('sulfur_containing')
    if any(a.GetIsAromatic() for a in mol.GetAtoms()):
        groups.append('aromatic')
    if mol.GetRingInfo().NumRings():
        groups.append('cyclic')
    return groups or ['other']


def estimate_row(row, descriptors, original_descriptors):
    T = row['melting_temperature_K']
    smiles = row.get('smiles')
    predictions = {'walden_54_4': {'value': 54.4 * T / 1000, 'entropy_J_mol_K': 54.4}}
    for method in METHODS[1:]:
        try:
            if method == 'chickos_published_1999':
                entropy = row.get('published_prediction_entropy_J_mol_K')
                if entropy is None:
                    raise NotApplicable('published prediction absent for this observation')
                predictions[method] = {'value': entropy * T / 1000,
                                       'entropy_J_mol_K': entropy,
                                       'origin': 'published rounded group-additivity prediction; descriptor replay'}
                continue
            if not smiles:
                raise NotApplicable('molecular structure unavailable')
            if method.startswith('chickos'):
                result = chickos(smiles, updated=method.endswith('1999'))
            elif method == 'dy1996_author_descriptors':
                desc = original_descriptors.get(row.get('cas'))
                if desc is None:
                    raise NotApplicable('no unique original DY1996 descriptors for this CAS')
                value = 50 - 8.314 * desc['ln_sigma'] + 8.314 * desc['ln_phi']
                from fusion_methods import Estimate
                result = Estimate(value, {**desc, 'descriptor_origin': 'DY1996 rounded published log descriptors'})
            elif method == 'jain_author_descriptors_2004':
                desc = descriptors.get(row.get('cas'))
                if desc is None:
                    raise NotApplicable('no unique author-assigned Jain descriptors for this CAS')
                # Admission is structural, and the author descriptors are never
                # selected by closeness to the experimental target.
                jain_domain(molecule(smiles))
                result = descriptor_entropy(desc['sigma'], desc['tau'])
            else:
                result = acid_estimate(smiles, use_entropy='entropy_' in method,
                                       melting_temperature_K=T)
            predictions[method] = {'value': result.enthalpy_kJ_mol(T),
                                   'entropy_J_mol_K': result.entropy_J_mol_K,
                                   'details': result.details}
        except NotApplicable as error:
            predictions[method] = {'value': None, 'rejection': str(error)}
    return {**row, 'predictions': predictions,
            'chemical_classes': classifications(smiles) if smiles else ['unresolved']}


def load_handbooks(paper):
    """Perry provides paired Tm/Hfus; CRC is a same-CAS handbook cross-table join."""
    result, excluded = [], []
    known = defaultdict(list)
    for row in paper:
        if row.get('cas') and not row['numeric_audit_flags']:
            known[row['cas']].append(row)
    perry = load_json(ROOT / 'data/perry_heat_of_fusion.json')['chemicals']
    for cas, entry in sorted(perry.items()):
        for index, source in enumerate(entry['heat_of_fusion']):
            row = {'id': f'perry:{cas}:{index}', 'cas': cas, 'dataset': 'perry_2_68',
                   'name': source['table_name'], 'formula': source['formula'],
                   'melting_temperature_K': source.get('Tm_K'),
                   'fusion_enthalpy_kJ_mol': source['Hfus_kJ_per_mol'],
                   'raw_source': source}
            result.append(row)
    # Trigger the public lazy table loaders without consulting any resolver.
    phase_change.Hfus('71-43-2', method='CRC')
    for cas, source in phase_change.Hfus_data_CRC.iterrows():
        if cas not in miscdata.CRC_organic_data.index:
            excluded.append({'dataset': 'CRC', 'cas': cas, 'reason': 'no CRC organic melting point'})
            continue
        T = miscdata.CRC_organic_data.loc[cas, 'Tm']
        if not math.isfinite(float(T)):
            excluded.append({'dataset': 'CRC', 'cas': cas, 'reason': 'missing CRC melting point'})
            continue
        result.append({'id': f'crc:{cas}', 'cas': cas, 'dataset': 'crc_handbook',
                       'name': source['Chemical'], 'formula': source['Formula'],
                       'melting_temperature_K': float(T), 'fusion_enthalpy_kJ_mol': float(source['Hfus']) / 1000,
                       'pairing': 'same CAS across CRC fusion and organic-constants tables; form unreported'})
    accepted = []
    for row in result:
        try:
            if row['melting_temperature_K'] is None or not row['melting_temperature_K'] > 0 or row['fusion_enthalpy_kJ_mol'] <= 0:
                raise NotApplicable('invalid/unpaired Tm/Hfus')
            meta = search_chemical(row['cas'])
            if meta.CASs != row['cas'] or parse_formula_counts(meta.formula) != parse_formula_counts(row['formula']):
                raise NotApplicable('CAS/formula identity mismatch')
            molecule(meta.smiles)
            row['smiles'] = meta.smiles
            matches = [r for r in known[row['cas']] if abs(r['melting_temperature_K'] - row['melting_temperature_K']) <= 3.0]
            statuses = {r['has_solid_solid_transition'] for r in matches}
            row['transition_status'] = ('reported_multiple' if True in statuses else
                                        'single_transition_in_chickos' if statuses == {False} else 'unknown')
            row['chickos_fit_roles'] = sorted({r['fit_role'] for r in known[row['cas']]})
            row['lineage_warning'] = 'Handbook data can share measurements with model-development literature; independence not established.'
            accepted.append(row)
        except (NotApplicable, ValueError, KeyError) as error:
            excluded.append({'id': row['id'], 'dataset': row['dataset'], 'cas': row['cas'], 'reason': str(error)})
    return accepted, excluded


def make_panel(rows, target, methods=METHODS):
    summaries = {}
    for method in methods:
        pairs = [(r[target], r['predictions'][method]['value']) for r in rows
                 if r['predictions'][method]['value'] is not None]
        summaries[method] = {'observation_metrics': summarize(pairs),
                             'compound_metrics': compound_summary(rows, method, target),
                             'rejections': dict(Counter(r['predictions'][method].get('rejection') for r in rows
                                                       if r['predictions'][method]['value'] is None))}
    pairwise = {}
    for left, right in itertools.combinations(methods, 2):
        common = [r for r in rows if all(r['predictions'][m]['value'] is not None for m in [left, right])]
        if common:
            pairwise[f'{left} vs {right}'] = {m: compound_summary(common, m, target) for m in [left, right]}
    classes = sorted({c for r in rows for c in r.get('chemical_classes', [])})
    return {'n_observations': len(rows), 'target': target, 'methods': summaries,
            'paired_comparisons': pairwise,
            'by_chemical_class': {c: {m: compound_summary([r for r in rows if c in r['chemical_classes']], m, target)
                                     for m in methods} for c in classes}}


def entropy_panel(jain_rows):
    result = []
    for row in jain_rows:
        current = dict(row)
        predictions = {}
        for method, revised in [('jain_author_descriptors_2004', True), ('dy1996_equation_on_jain_descriptors', False)]:
            estimate = descriptor_entropy(row['sigma'], row['tau'], revised=revised)
            predictions[method] = {'value': estimate.entropy_J_mol_K}
        predictions['chickos_published_1999'] = {'value': row['published_chickos_entropy_J_mol_K']}
        predictions['walden_54_4'] = {'value': 54.4}
        current['predictions'] = predictions
        current['chemical_classes'] = classifications(row['smiles']) if row.get('smiles') else ['unresolved']
        result.append(current)
    methods = list(result[0]['predictions'])
    panel = make_panel(result, 'total_entropy_J_mol_K', methods)
    panel['replay_maximum_absolute_difference_J_mol_K'] = max(abs(
        r['predictions']['jain_author_descriptors_2004']['value'] - r['published_jain_entropy_J_mol_K']) for r in result)
    panel['warning'] = 'Chickos published values are a replay on its development lineage. '
    panel['warning'] += 'DY1996 column changes the flexibility base on Jain descriptors; '
    panel['warning'] += 'it is a controlled equation comparison, not a faithful replay of DY1996 descriptor assignments.'
    panel['tau_le_12'] = make_panel([r for r in result if r['tau'] <= 12], 'total_entropy_J_mol_K', methods)
    return panel, result


def dy_entropy_panel(rows):
    transformed = []
    for row in rows:
        if row['total_entropy_J_mol_K'] <= 0:
            continue
        transformed.append({**row, 'chemical_classes': classifications(row['smiles']) if row.get('smiles') else ['unresolved'],
                            'predictions': {'dy1996_author_descriptors': {'value': row['rounded_descriptor_entropy_J_mol_K']},
                                            'dy1996_published_prediction': {'value': row['published_prediction_entropy_J_mol_K']},
                                            'walden_54_4': {'value': 54.4}}})
    methods = ['dy1996_author_descriptors', 'dy1996_published_prediction', 'walden_54_4']
    accepted = [r for r in transformed if not r['paper_excluded'] and not r['polymer']]
    panel = make_panel(accepted, 'total_entropy_J_mol_K', methods)
    panel['warning'] = 'Original source-assigned log descriptors are rounded to 0.1; OCR-rejected rows remain in the extraction audit.'
    panel['excluded_diagnostic'] = make_panel([r for r in transformed if r['paper_excluded']], 'total_entropy_J_mol_K', methods)
    differences = [abs(r['rounded_descriptor_entropy_J_mol_K'] - r['published_prediction_entropy_J_mol_K']) for r in accepted]
    panel['rounded_descriptor_replay_maximum_difference'] = max(differences)
    panel['rounded_descriptor_replay_median_difference'] = statistics.median(differences)
    return panel, transformed


def format_report(report):
    lines = ['Fusion estimation benchmark with measured melting temperatures',
             'MAE: kJ/mol for enthalpy panels; J/(mol K) for entropy panels.',
             'MAPE is relative absolute error. Compound-equal rows prevent repeated measurements from dominating.',
             'Handbook/source-development populations are not independent validation.', '']
    for name, panel in report['panels'].items():
        lines += [name, f'  observations: {panel["n_observations"]}; target: {panel["target"]}']
        if 'warning' in panel:
            lines.append('  ' + panel['warning'])
        for method, summary in panel['methods'].items():
            obs = summary['observation_metrics']
            comp = summary['compound_metrics']
            if not obs['n']:
                lines.append(f'  {method}: no admitted cases')
                continue
            lines.append(f'  {method:40s} n={obs["n"]:4d} compounds={comp["n_compounds"]:4d} '
                         f'MAE={obs["mae"]:7.3f} MAPE={obs["mape_percent"]:6.2f}% '
                         f'compound MAPE={comp["compound_equal_mape_percent"]:6.2f}% '
                         f'P90={obs["p90_ape_percent"]:6.2f}% bias={obs["bias_percent"]:+6.2f}%')
        comparison = panel['paired_comparisons'].get('chickos_perry_1991 vs chickos_structure_1999')
        if comparison:
            lines.append('  SAME-POPULATION PERRY VS UPDATED:')
            for m, s in comparison.items():
                lines.append(f'    {m}: {s["n_compounds"]} compounds; MAPE={s["compound_equal_mape_percent"]:.2f}%')
        lines.append('')
    lines += ['AUDIT', json.dumps(report['audit'], indent=2, sort_keys=True), '',
              'All rows, group assignments, exclusions, paired comparisons and source hashes are in benchmark.json.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    paper_data = load_json(HERE / 'data/chickos_1999_records.json')
    jain_data = load_json(HERE / 'data/jain_2004_records.json')
    acids_data = load_json(HERE / 'data/compernolle_2011_records.json')
    dy_data = load_json(HERE / 'data/dannenfelser_1996_records.json')
    descriptors = source_descriptors(jain_data['records'])
    original_descriptors = dy_descriptors(dy_data['records'])
    paper = [estimate_row(r, descriptors, original_descriptors) for r in paper_data['records']
             if not r['numeric_audit_flags'] and r.get('smiles')]
    handbooks, excluded = load_handbooks(paper_data['records'])
    handbooks = [estimate_row(r, descriptors, original_descriptors) for r in handbooks]
    acids = [estimate_row({**r, 'dataset': 'compernolle_2011', 'cas': None}, descriptors, original_descriptors)
             for r in acids_data['records']]
    panels = {}
    for role in ['training', 'test']:
        for status, condition in [('single_transition', lambda r: not r['has_solid_solid_transition']),
                                  ('multiple_transition_diagnostic', lambda r: r['has_solid_solid_transition'])]:
            population = [r for r in paper if r['fit_role'] == role and not r['paper_excluded'] and condition(r)]
            panels[f'chickos_{role}_{status}'] = make_panel(population, 'fusion_enthalpy_kJ_mol')
    training_cas = {r['cas'] for r in paper_data['records']
                    if r.get('cas') and r['fit_role'] == 'training' and not r['paper_excluded']}
    test_single = [r for r in paper if r['fit_role'] == 'test' and not r['paper_excluded']
                   and not r['has_solid_solid_transition']]
    overlap = [r for r in test_single if r['cas'] in training_cas]
    panels['chickos_test_single_transition_disjoint_resolved_CAS'] = make_panel(
        [r for r in test_single if r['cas'] not in training_cas], 'fusion_enthalpy_kJ_mol')
    panels['chickos_test_single_transition_disjoint_resolved_CAS']['warning'] = (
        'Disjoint from identity-resolved training CAS only; unresolved training identities prevent a universal independence guarantee.')
    panels['chickos_paper_excluded_diagnostic'] = make_panel([r for r in paper if r['paper_excluded']], 'fusion_enthalpy_kJ_mol')
    for dataset in ['perry_2_68', 'crc_handbook']:
        rows = [r for r in handbooks if r['dataset'] == dataset]
        panels[f'{dataset}_all'] = make_panel(rows, 'fusion_enthalpy_kJ_mol')
        panels[f'{dataset}_no_reported_multiple_transition'] = make_panel(
            [r for r in rows if r['transition_status'] != 'reported_multiple'], 'fusion_enthalpy_kJ_mol')
    panels['acid_development_total_enthalpy_diagnostic'] = make_panel(acids, 'total_enthalpy_kJ_mol')
    panels['jain_total_entropy_replay'], entropy_rows = entropy_panel(jain_data['records'])
    panels['dy1996_total_entropy_replay'], dy_rows = dy_entropy_panel(dy_data['records'])
    structure_replay = []
    for row in paper:
        computed = row['predictions']['chickos_structure_1999']
        published = row.get('published_prediction_entropy_J_mol_K')
        if computed['value'] is not None and published is not None:
            difference = computed['entropy_J_mol_K'] - published
            structure_replay.append({'id': row['id'], 'name': row['name'], 'cas': row.get('cas'),
                                     'difference_J_mol_K': difference,
                                     'computed_entropy_J_mol_K': computed['entropy_J_mol_K'],
                                     'published_entropy_J_mol_K': published,
                                     'published_group_expression': row['group_expression_lines']})
    replay_abs = [abs(r['difference_J_mol_K']) for r in structure_replay]
    descriptor_audit = []
    for row in jain_data['records']:
        if not row.get('smiles'):
            continue
        try:
            inferred = flexibility(row['smiles'])
        except NotApplicable:
            continue
        descriptor_audit.append({'id': row['id'], 'name': row['name'], 'source_tau': row['tau'],
                                 'inferred': inferred, 'agrees': inferred['tau'] == row['tau']})
    files = sorted((HERE / 'data').glob('*.json')) + [Path(__file__), HERE / 'fusion_methods.py', HERE / 'prepare_data.py',
                                                     ROOT / 'data/perry_heat_of_fusion.json']
    import chemicals
    chemical_dir = Path(chemicals.__file__).parent
    external = [chemical_dir / 'Phase Change/CRC Handbook Heat of Fusion.tsv',
                chemical_dir / 'Misc/Physical Constants of Organic Compounds.csv']
    inputs = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p in external:
        if p.exists():
            inputs['chemicals/' + p.relative_to(chemical_dir).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    report = {'panels': panels, 'rows': {'paper': paper, 'handbooks': handbooks, 'acids': acids,
                                      'entropy': entropy_rows, 'dy_entropy': dy_rows},
              'handbook_exclusions': excluded, 'input_sha256': inputs,
              'implementation_audit': {'structure_replay': structure_replay,
                                       'automatic_flexibility': descriptor_audit},
              'environment': {'python': platform.python_version(),
                              'packages': {n: importlib.metadata.version(n) for n in ['chemicals', 'rdkit', 'numpy']}},
              'audit': {'paper_paired_observations': len(paper_data['records']),
                        'paper_numeric_flags': dict(Counter(f for r in paper_data['records'] for f in r['numeric_audit_flags'])),
                        'paper_identity_status': dict(Counter(r['identity_status'] for r in paper_data['records'])),
                        'paper_incomplete_blocks': len(paper_data['incomplete_blocks']),
                        'handbook_exclusions': len(excluded), 'unique_author_descriptor_CAS': len(descriptors),
                        'unique_DY1996_descriptor_CAS': len(original_descriptors),
                        'identity_resolved_training_CAS': sorted(training_cas),
                        'paper_designated_test_training_CAS_overlap': [{'cas': r['cas'], 'name': r['name']} for r in overlap],
                        'DY1996_OCR_rejected_rows': len(dy_data['unparsed_rows']),
                        '1999_structure_replay': {'n': len(replay_abs), 'median_absolute_difference_J_mol_K': statistics.median(replay_abs),
                                                  'within_0_5_J_mol_K': sum(v <= .5 for v in replay_abs)},
                        'Jain_automatic_flexibility': {'n': len(descriptor_audit),
                                                       'disagreements': sum(not r['agrees'] for r in descriptor_audit)},
                        'jain_replay_maximum_difference_J_mol_K': panels['jain_total_entropy_replay']['replay_maximum_absolute_difference_J_mol_K']}}
    write_json(args.output_dir / 'benchmark.json', report)
    text = format_report(report)
    # Reuse the backup policy for the plain text report too.
    text_path = args.output_dir / 'benchmark.txt'
    if text_path.exists():
        import shutil
        digest = hashlib.sha256(text_path.read_bytes()).hexdigest()[:16]
        shutil.copy2(text_path, Path('/tmp') / f'pfdsim-fusion-benchmark-{digest}.txt.backup')
    text_path.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
