#!/usr/bin/env python3
"""Extract auditable offline benchmark inputs from the published sources.

Requires pdftotext and xlrd only at preparation time. Run with
`uv run --with xlrd python scripts/fusion/prepare_data.py`. Benchmark execution
uses the resulting ordinary JSON files and needs neither XLS nor PDF readers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCES = ROOT / 'data/reference/property_estimation'
sys.path.insert(0, str(ROOT))

from chemicals.identifiers import search_chemical
from compound_identity import parse_formula_counts


def write_json(path: Path, payload) -> None:
    """Keep a recoverable copy before replacing any existing untracked output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        backup = Path('/tmp') / f'pfdsim-fusion-{path.name}-{digest}.backup'
        if not backup.exists():
            shutil.copy2(path, backup)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def pdf_text(path: Path) -> str:
    return subprocess.run(['pdftotext', '-layout', str(path), '-'], check=True,
                          capture_output=True, text=True).stdout


def source_metadata(path: Path, doi: str, url: str = '') -> dict:
    try:
        location = path.relative_to(ROOT).as_posix()
    except ValueError:
        location = path.name
    return {'path': location, 'doi': doi, 'url': url,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def canonical_name(name: str) -> str:
    return (name.replace('共', '(').replace('兲', ')').replace('␤', 'beta')
            .replace('␣', 'alpha').replace('␻', 'omega').replace('⫺', '-').strip())


def identity(name: str, formula: str) -> dict:
    """Resolve locally, with source-formula checks; never fall back by formula."""
    clean = re.sub(r'\s+', '', formula.replace('*', ''))
    expected = parse_formula_counts(clean)
    if not expected:
        return {'identity_status': 'source formula not parseable'}
    try:
        metadata = search_chemical(canonical_name(name), autoload=True, cache=True)
    except Exception:
        return {'identity_status': 'name not resolved locally'}
    if parse_formula_counts(metadata.formula) != expected:
        return {'identity_status': 'source/resolved formula mismatch',
                'resolved_name': metadata.common_name, 'resolved_formula': metadata.formula}
    return {'identity_status': 'name/formula matched', 'cas': metadata.CASs,
            'smiles': metadata.smiles, 'resolved_name': metadata.common_name}


NUM = re.compile(r'(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?')
FORMULA = re.compile(r'^([A-Z][A-Za-z0-9*()·+−\-]*)\s{2,}(.+?)\s*$')


def extract_chickos(text: str) -> dict:
    """Preserve final-transition rows separately from total-transition sums.

    The seven source columns are located from each page's printed header.
    Rounded published totals are cross-checked against the transition rows.
    A record with inconsistent totals remains in the audit, with a flag.
    """
    blocks = []
    current = None
    header_positions = None
    table = None
    for page_number, page in enumerate(text.split('\f'), 1):
        table_match = re.search(r'TABLE (5|6|7)\.', page)
        if table_match:
            table = int(table_match.group(1))
        if table not in {5, 7}:
            continue
        lines = page.splitlines()
        # References start after Table 7: no new data are admitted without a header.
        header = next((line for line in lines if len(re.findall(r'共(?:expt|calcd)兲', line)) == 6), None)
        if header is None:
            continue
        headings = list(re.finditer(r'共(?:expt|calcd)兲', header))
        temp = re.search(r'T\s*[(共]', header)
        if temp is None:
            continue
        header_positions = [temp.start()] + [h.start() for h in headings]
        for line_number, line in enumerate(lines, 1):
            match = FORMULA.match(line)
            if match and parse_formula_counts(match.group(1).replace('*', '')):
                if current:
                    blocks.append(current)
                current = {'formula': match.group(1).replace('*', ''),
                           'name': canonical_name(match.group(2)), 'table': table,
                           'paper_excluded': '*' in match.group(1),
                           'source_page': page_number, 'transitions': [],
                           'group_expression_lines': [], 'notes': []}
                continue
            if current is None:
                continue
            if re.search(r'\b[A-F]\d+\b', line) and 'TABLE' not in line:
                current['group_expression_lines'].append(line.strip())
                continue
            if not re.match(r'^\s+\d+(?:\.\d*)?\s{2,}', line):
                if any(t in line for t in ['Group value', 'Forms plastic', 'No prediction', 'Excluded', 'liquid crystal']):
                    current['notes'].append(line.strip())
                continue
            numbers = list(NUM.finditer(line))
            values = [None] * 7
            ambiguity = False
            for number in numbers:
                if number.start() > header_positions[-1] + 15:
                    continue
                col = min(range(7), key=lambda k: abs(number.start() - header_positions[k]))
                if values[col] is not None:
                    ambiguity = True
                values[col] = float(number.group())
            if values[0] is None or values[1] is None:
                continue
            current['transitions'].append({
                'temperature_K': values[0], 'enthalpy_kJ_mol': values[1],
                'entropy_J_mol_K': values[2], 'total_entropy_J_mol_K': values[3],
                'published_prediction_entropy_J_mol_K': values[4],
                'total_enthalpy_kJ_mol': values[5],
                'published_prediction_enthalpy_kJ_mol': values[6],
                'column_collision': ambiguity, 'source_page': page_number,
                'source_line': line_number, 'raw_row': line.strip(),
            })
    if current:
        blocks.append(current)
    records = []
    incomplete = []
    for block_number, block in enumerate(blocks):
        resolved = identity(block['name'], block['formula'])
        pending = []
        for row_number, row in enumerate(block['transitions']):
            pending.append(row)
            if row['total_entropy_J_mol_K'] is None or row['total_enthalpy_kJ_mol'] is None:
                continue
            final_temperature = row['temperature_K']
            flags = []
            if any(r['column_collision'] for r in pending):
                flags.append('ambiguous numeric column assignment')
            if not all(0 < r['temperature_K'] <= final_temperature and r['enthalpy_kJ_mol'] > 0 for r in pending):
                flags.append('invalid transition order/value')
            enthalpy_sum = sum(r['enthalpy_kJ_mol'] for r in pending)
            entropy_sum = sum(r['enthalpy_kJ_mol'] * 1000.0 / r['temperature_K'] for r in pending)
            if not math.isclose(enthalpy_sum, row['total_enthalpy_kJ_mol'], rel_tol=.025, abs_tol=.15):
                flags.append('transition enthalpies disagree with published total')
            if not math.isclose(entropy_sum, row['total_entropy_J_mol_K'], rel_tol=.03, abs_tol=.6):
                flags.append('transition entropies disagree with published total')
            record = {k: v for k, v in block.items() if k != 'transitions'}
            record.update(resolved)
            record.update({
                'id': f'chickos1999:{block["table"]}:{block_number}:{row_number}',
                'dataset': 'chickos_1999', 'melting_temperature_K': final_temperature,
                'fusion_enthalpy_kJ_mol': row['enthalpy_kJ_mol'],
                'total_enthalpy_kJ_mol': row['total_enthalpy_kJ_mol'],
                'total_entropy_J_mol_K': row['total_entropy_J_mol_K'],
                'published_prediction_entropy_J_mol_K': row['published_prediction_entropy_J_mol_K'],
                'transitions': pending, 'has_solid_solid_transition': len(pending) > 1,
                'numeric_audit_flags': flags,
                'fit_role': 'training' if block['table'] == 5 else 'test',
            })
            records.append(record)
            pending = []
        if pending or not block['transitions']:
            incomplete.append({'name': block['name'], 'formula': block['formula'],
                               'page': block['source_page'], 'unpaired_rows': pending})
    if len(records) < 1500:
        raise RuntimeError(f'Chickos extraction unexpectedly small: {len(records)}')
    return {'records': records, 'incomplete_blocks': incomplete,
            'block_count': len(blocks), 'extraction_policy':
            'Source seven-column layout; final row carries both experimental totals; '
            'all earlier rows retained as solid-solid transitions; mismatches audited.'}


def extract_jain(path: Path) -> dict:
    import xlrd
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)
    if sheet.nrows != 1801 or sheet.ncols != 7:
        raise RuntimeError('Unexpected Jain supplement dimensions')
    records = []
    for i in range(2, sheet.nrows):
        formula, name, sigma, tau, observed, jain, chickos = sheet.row_values(i)
        row = {'id': f'jain2004:{i + 1}', 'source_row': i + 1, 'formula': formula.strip(),
               'name': name.strip(), 'sigma': float(sigma), 'tau': float(tau),
               'total_entropy_J_mol_K': float(observed),
               'published_jain_entropy_J_mol_K': float(jain),
               'published_chickos_entropy_J_mol_K': float(chickos)}
        row.update(identity(row['name'], row['formula']))
        records.append(row)
    return {'records': records, 'column_order': ['formula', 'name', 'sigma', 'tau',
                                               'observed', 'Jain Eq VI', 'Chickos 1999'],
            'target': 'total transition entropy, not necessarily final fusion entropy',
            'lineage': 'Experimental and Chickos predictions taken from Chickos 1999; '
                       'not an independent test set for Chickos.'}


def extract_acids(text: str) -> dict:
    from rdkit import Chem
    records = []
    current = None
    rejected_smiles = []
    for page_number, page in enumerate(text.split('\f'), 1):
        if 'Table S2' in page:
            break
        for line_number, line in enumerate(page.splitlines(), 1):
            tokens = list(re.finditer(r'\S+', line))
            smiles_token = next((t for t in tokens if t.group().startswith('OC(=O)')), None)
            if smiles_token:
                if Chem.MolFromSmiles(smiles_token.group()) is None:
                    rejected_smiles.append({'page': page_number, 'line': line_number, 'raw_row': line.strip()})
                    current = None
                    continue
                current = {'name': line[:smiles_token.start()].strip(),
                           'smiles': smiles_token.group()}
                tail = line[smiles_token.end():]
            elif current:
                # Continuation rows carry another explicitly paired measurement.
                tail = line
            else:
                continue
            pair = re.search(r'(?<!\S)(\d+(?:\.\d+)?)\s{2,}(\d+(?:\.\d+)?)\s{2,}([^\d\s].*)$', tail)
            if pair is None:
                continue
            T, H = float(pair[1]), float(pair[2])
            if not 150 <= T <= 900 or not 0 < H < 300:
                continue
            records.append({**current, 'id': f'acids2011:{page_number}:{line_number}',
                            'melting_temperature_K': T, 'total_enthalpy_kJ_mol': H,
                            'source_page': page_number, 'source_line': line_number,
                            'raw_row': line.strip(), 'reference_text': pair[3],
                            'stereo_row_label': line[:pair.start()].strip() if not smiles_token else ''})
    if len(records) < 30:
        raise RuntimeError(f'Acid supplement extraction unexpectedly small: {len(records)}')
    return {'records': records, 'rejected_source_smiles': rejected_smiles,
            'target': 'Supplement sums enthalpies across all reported transitions; '
                      'not guaranteed to be final fusion enthalpy.',
            'lineage': 'Compernolle development observations, not an independent test.'}


def extract_dannenfelser(text: str) -> dict:
    records = []
    unparsed = []
    for page_number, page in enumerate(text.split('\f'), 1):
        header = next((line for line in page.splitlines() if 'EVAL' in line and 'DIFF' in line), None)
        positions = None
        if header:
            tokens = list(re.finditer(r'\S+', header))
            start = next(i for i, token in enumerate(tokens) if token.group() == 'EVAL') + 1
            if len(tokens) > start + 2:
                positions = [tokens[start + i].start() for i in range(3)]
        for line_number, line in enumerate(page.splitlines(), 1):
            # OCR columns: formula, name, optional evaluation grade, observed,
            # predicted, difference, ln(sigma), ln(phi), integer reference.
            m = re.search(r'((?:[-+]?\d+\.\d+\s+){4,5})(\d+)\s*$', line)
            if m is None:
                if re.search(r'\d+\.\d+\s+\d+\s*$', line) and 'FORMULA' not in line:
                    unparsed.append({'page': page_number, 'line': line_number, 'raw_row': line})
                continue
            front = line[:m.start()].strip()
            source_flag = front[0] if front.startswith(('*', '+')) else ''
            front = front.lstrip('*+ ').strip()
            grade_match = re.search(r'\s([ABCD])$', front)
            grade = grade_match[1] if grade_match else ''
            if grade_match:
                front = front[:grade_match.start()]
            fields = re.split(r'\s{2,}', front, maxsplit=1)
            if len(fields) != 2:
                unparsed.append({'page': page_number, 'line': line_number, 'raw_row': line})
                continue
            formula, name = fields
            numbers = [float(v) for v in m[1].split()]
            if len(numbers) == 5:
                observed, predicted, difference, log_sigma, log_phi = numbers
                prediction_origin = 'printed'
            else:
                if positions is None:
                    unparsed.append({'page': page_number, 'line': line_number, 'raw_row': line})
                    continue
                numeric_tokens = list(re.finditer(r'[-+]?\d+\.\d+', m[1]))
                assigned = {}
                for token in numeric_tokens[:2]:
                    location = m.start(1) + token.start()
                    col = min(range(3), key=lambda j: abs(location - positions[j]))
                    assigned[col] = float(token.group())
                if len(assigned) != 2:
                    unparsed.append({'page': page_number, 'line': line_number, 'raw_row': line})
                    continue
                missing = ({0, 1, 2} - set(assigned)).pop()
                if missing == 0:
                    assigned[0] = assigned[1] + assigned[2]
                elif missing == 1:
                    assigned[1] = assigned[0] - assigned[2]
                else:
                    assigned[2] = assigned[0] - assigned[1]
                observed, predicted, difference = [assigned[i] for i in range(3)]
                log_sigma, log_phi = numbers[-2:]
                prediction_origin = f'printed layout locates missing column {missing}; reconstructed from observed-predicted=difference'
            ref = int(m[2])
            row = {'id': f'dy1996:{page_number}:{line_number}', 'name': name.strip(),
                   'formula': formula.replace('*', '').strip(), 'source_page': page_number,
                   'raw_row': line.strip(), 'paper_excluded': bool(source_flag),
                   'source_flag': source_flag, 'prediction_origin': prediction_origin,
                   'evaluation_grade': grade or 'unreported', 'reference_number': int(ref),
                   'total_entropy_J_mol_K': float(observed),
                   'published_prediction_entropy_J_mol_K': float(predicted),
                   'ln_sigma': float(log_sigma), 'ln_phi': float(log_phi),
                   'published_difference': float(difference),
                   'rounded_descriptor_entropy_J_mol_K': 50 - 8.314 * float(log_sigma) + 8.314 * float(log_phi)}
            row.update(identity(row['name'], row['formula']))
            # Polymers use no intercept; handled explicitly by source notation.
            row['polymer'] = '(' in row['formula'] and 'n' in row['formula']
            if row['polymer']:
                row['rounded_descriptor_entropy_J_mol_K'] -= 50
            records.append(row)
    if len(records) < 1000:
        raise RuntimeError(f'DY supplement extraction unexpectedly small: {len(records)}')
    return {'records': records, 'unparsed_rows': unparsed,
            'target': 'total transition entropy',
            'descriptor_policy': 'Published ln(sigma) and ln(phi) are rounded to 0.1; '
                                 'replay uses those printed numbers and reports rounding discrepancies.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chickos-pdf', type=Path, default=REFERENCES / '1999-chickos-acree-liebman-solid-liquid-phase-change-enthalpies-entropies.pdf')
    parser.add_argument('--jain-xls', type=Path, default=REFERENCES / '2004-jain-yang-yalkowsky-total-entropy-melting-supporting-information.xls')
    args = parser.parse_args()
    chickos = extract_chickos(pdf_text(args.chickos_pdf))
    jain = extract_jain(args.jain_xls)
    write_json(HERE / 'data/chickos_1999_records.json', chickos)
    write_json(HERE / 'data/jain_2004_records.json', jain)
    acids = extract_acids(pdf_text(REFERENCES / '2011-compernolle-ceulemans-muller-fusion-properties-functionalised-acids-supporting-information.pdf'))
    write_json(HERE / 'data/compernolle_2011_records.json', acids)
    dy = extract_dannenfelser(pdf_text(REFERENCES / '1996-dannenfelser-yalkowsky-total-entropy-melting-supporting-information.pdf'))
    write_json(HERE / 'data/dannenfelser_1996_records.json', dy)
    sources = [
        (args.chickos_pdf, '10.1063/1.556045', 'https://srd.nist.gov/jpcrdreprint/1.556045.pdf'),
        (args.jain_xls, '10.1021/ie0497745', 'https://ndownloader.figshare.com/files/5170753'),
        (REFERENCES / '1996-dannenfelser-yalkowsky-total-entropy-melting-supporting-information.pdf', '10.1021/ie940581z', ''),
        (REFERENCES / '2011-compernolle-ceulemans-muller-fusion-properties-functionalised-acids.pdf', '10.5194/acp-11-8385-2011', 'https://acp.copernicus.org/articles/11/8385/2011/acp-11-8385-2011.pdf'),
        (REFERENCES / '2011-compernolle-ceulemans-muller-fusion-properties-functionalised-acids-supporting-information.pdf', '10.5194/acp-11-8385-2011', 'https://acp.copernicus.org/articles/11/8385/2011/acp-11-8385-2011-supplement.pdf'),
        (REFERENCES / '2004-jain-yang-yalkowsky-total-entropy-melting.pdf', '10.1021/ie0497745', ''),
        (REFERENCES / '1996-dannenfelser-yalkowsky-total-entropy-melting.pdf', '10.1021/ie940581z', ''),
        (ROOT / 'data/source/perrys-chemical-engineers-handjob-9th-edition-2021-9nbsped-9780071834094-0071834095-9780071834087-0071834087_compress.pdf', '', ''),
    ]
    manifest = {'sources': [source_metadata(*entry) for entry in sources],
                'formulas': {'Perry': 'pp. 2-335/336, Eqs. 2-42/43/44, Tables 2-162/163; 1991 Chickos parameters',
                             'Chickos1999': 'Eqs. 2-7, Tables 1/2; reference transitions Tables 5/7',
                             'DY1996': 'Eqs. 7/8/10 and original supplementary log descriptors',
                             'Jain2004': 'Eqs. III/IV/VI, Table 1 and 1799-row supplement',
                             'Compernolle2011': 'Eqs. 20/21, Table 1 and supplement Table S1'},
                'extraction_environment': {'python': sys.version.split()[0]}}
    write_json(HERE / 'data/source_manifest.json', manifest)
    print(f'Chickos: {len(chickos["records"])} paired observations from {chickos["block_count"]} blocks')
    print(f'Jain: {len(jain["records"])} total-entropy observations')
    print(f'Acid supplement: {len(acids["records"])} paired observations')
    print(f'Dannenfelser: {len(dy["records"])} total-entropy observations; {len(dy["unparsed_rows"])} OCR rejects')


if __name__ == '__main__':
    main()
