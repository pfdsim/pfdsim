"""Lyngby table access and exact translation of shared structural groups."""

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path


def canonical_lyngby_method(method):
    """Normalize this model family's names consistently across entry points."""
    aliases = {
        'RKSMHV2': 'RKSMHV2', 'RKS-MHV2': 'RKSMHV2', 'SRK-MHV2': 'RKSMHV2',
        'UNIFLBY': 'UNIFLBY', 'UNIF-LBY': 'UNIFLBY', 'UNIFAC-LBY': 'UNIFLBY',
        'UNIFAC-LYNGBY': 'UNIFLBY',
    }
    return aliases.get(str(method).strip().upper().replace('_', '-'), str(method).upper())


@lru_cache(maxsize=2)
def parameter_table(gas_extension=False):
    filename = 'mhv2_unifac.json' if gas_extension else 'lyngby_unifac.json'
    return json.loads((Path(__file__).resolve().parent / 'data' / filename).read_text())


def convert_classic_groups(groups):
    """Translate structural identities, never classic subgroup numbers verbatim.

    Lyngby splits aromatic alkyl groups and primary amines into their atoms'
    constituent groups. Unsupported chemistry is rejected rather than guessed.
    """
    classic = _classic_group_names()
    names = {sg['name'].upper(): sg['number'] for sg in parameter_table()['subgroups']}
    splits = {
        'ACCH3': {'AC': 1, 'CH3': 1}, 'ACCH2': {'AC': 1, 'CH2': 1},
        'ACCH': {'AC': 1, 'CH': 1}, 'CH3NH2': {'CH3': 1, 'NH2': 1},
        'CH2NH2': {'CH2': 1, 'NH2': 1}, 'CHNH2': {'CH': 1, 'NH2': 1},
        'ACNH2': {'AC': 1, 'ANH2': 1}, 'DOH': {'CH2': 2, 'OH': 2},
    }
    result = Counter()
    for key, count in groups.items():
        name = classic.get(int(key)) if str(key).isdigit() else str(key).upper()
        for part, multiplier in splits.get(name, {name: 1}).items():
            if part not in names:
                raise ValueError(f'No Lyngby UNIFAC subgroup for {name!r}')
            result[names[part]] += count * multiplier
    return dict(result)


@lru_cache(maxsize=1)
def _classic_group_names():
    path = Path(__file__).resolve().parent / 'data' / 'unifac_params.json'
    data = json.loads(path.read_text())
    return {sg['number']: ('CH-O' if sg['number'] == 26 else sg['name'].upper())
            for sg in data['subgroups']}
