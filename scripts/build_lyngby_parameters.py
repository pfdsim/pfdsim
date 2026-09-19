"""Build Lyngby UNIFAC runtime data from the checked-in source tables.

The source is the MIT-licensed thermo project's transcription of Larsen,
Rasmussen and Fredenslund (1987), DOI 10.1021/ie00071a018. See docs/lyngby.md.
"""

import csv
import json
from pathlib import Path


def build(*, gas_extension=False):
    root = Path(__file__).resolve().parents[1]
    source = root / 'data' / 'source'
    subgroups = []
    with (source / 'lyngby_subgroups.csv').open() as handle:
        for row in csv.DictReader(handle):
            subgroups.append({
                **row,
                'number': int(row['number']),
                'main_group': int(row['main_group']),
                'R': float(row['R']),
                'Q': float(row['Q']),
            })
    interactions = []
    with (source / 'lyngby_interactions.tsv').open() as handle:
        for i, j, a, b, c in csv.reader(handle, delimiter='\t'):
            interactions.append(dict(i=int(i), j=int(j), a1=float(a),
                                     a2=float(b), a3=float(c)))
    assert len(subgroups) == 45
    assert len({(r['i'], r['j']) for r in interactions}) == len(interactions)
    result = {
        'metadata': {
            'model': 'Lyngby modified UNIFAC',
            'doi': '10.1021/ie00071a018',
            'source': 'CalebBell/thermo: LUFSG and UNIFAC Lyngby interaction parameters.tsv',
            'temperature_K': 298.15,
            'interaction_form': 'a1 + a2*(T-T0) + a3*(T*log(T0/T)+T-T0)',
        },
        'subgroups': subgroups,
        'interactions': interactions,
    }
    if gas_extension:
        result['metadata']['gas_extension_doi'] = '10.1021/ie00056a041'
        result['gas_components'] = {}
        with (source / 'mhv2_gas_groups.csv').open() as handle:
            for row in csv.DictReader(handle):
                main = int(row['main_group'])
                number = main + 24  # 46..58, after the 45 solvent subgroups.
                subgroups.append(dict(number=number, name=row['name'],
                                      main_group=main, main_group_name=row['name'],
                                      R=float(row['R']), Q=float(row['Q'])))
                result['gas_components'][row['CAS']] = number
        lines = [line.split() for line in
                 (source / 'mhv2_gas_interactions.txt').read_text().splitlines()
                 if line.strip() and not line.startswith('#')]
        for a_row, b_row in zip(lines[::2], lines[1::2]):
            i = int(a_row[0])
            assert a_row[0] == b_row[0]
            columns = list(range(22, 35)) if i < 22 else [1, 2, 3, 4, 5, 6, 7, 9, 10]
            assert len(a_row) == len(b_row) == len(columns) + 1
            for j, a, b in zip(columns, a_row[1:], b_row[1:]):
                assert (a == 'na') == (b == 'na')
                if a != 'na':
                    interactions.append(dict(i=i, j=j, a1=float(a), a2=float(b), a3=0.0))
        for i in range(22, 35):
            for j in range(22, 35):
                interactions.append(dict(i=i, j=j, a1=0.0, a2=0.0, a3=0.0))
        # Table III footnote: CH2-family groups within alcohol molecules.
        # Clone the alkyl main group; only its H2/N2 interactions differ.
        for row in list(interactions):
            if row['i'] == 1 or row['j'] == 1:
                interactions.append({**row,
                    'i': 35 if row['i'] == 1 else row['i'],
                    'j': 35 if row['j'] == 1 else row['j']})
        overrides = {(35, 22): (986.0, -2.133), (22, 35): (-129.2, -2.418),
                     (35, 24): (355.2, 0.0), (24, 35): (139.4, 0.0)}
        for row in interactions:
            if (row['i'], row['j']) in overrides:
                row['a1'], row['a2'] = overrides[(row['i'], row['j'])]
        interactions.extend(dict(i=i, j=j, a1=0.0, a2=0.0, a3=0.0)
                            for i, j in ((1, 35), (35, 1), (35, 35)))
        for sg in subgroups[:4]:
            subgroups.append({**sg, 'number': sg['number'] + 58,
                              'name': sg['name'] + "(alcohol)",
                              'main_group': 35, 'main_group_name': "CH2(alcohol)"})
        with (source / 'mhv2_alpha.csv').open() as handle:
            result['alpha'] = {row['CAS']: {key: float(row[key]) for key in ('c1', 'c2', 'c3')}
                               for row in csv.DictReader(handle)}
        assert len({(r['i'], r['j']) for r in interactions}) == len(interactions)
    return result


if __name__ == '__main__':
    for gas_extension, filename in ((False, 'lyngby_unifac.json'), (True, 'mhv2_unifac.json')):
        target = Path(__file__).resolve().parents[1] / 'data' / filename
        target.write_text(json.dumps(build(gas_extension=gas_extension), indent=2) + '\n')
        print(f'Built {target.name}')
