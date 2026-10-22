"""Exercise study-only repairs, disjoint ownership and support backoff."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from expanded_nannoolal_features import extract_study,fit_vocabulary,encode,nm
from rdkit import Chem


def test_sulfonamide_priority_is_repaired_without_changing_production():
    registry=list(nm._ASSIGNERS)
    row=extract_study('CS(=O)(=O)N')
    assert row['residual_atoms']==0
    assert row['native']['native_105']==1
    assert nm._ASSIGNERS==registry
    try:
        nm.fragment(Chem.MolFromSmiles('CS(=O)(=O)N'))
    except nm.FragmentationError:
        pass
    else:
        raise AssertionError('The production fragmenter was changed')


def test_overlapping_urea_amides_form_disjoint_systems():
    for smiles in ('O=C1NCC(=O)N1','O=C1NC(=O)CC(=O)N1','CC(=O)NC(=O)N'):
        row=extract_study(smiles)
        assert row['amide_systems']==1
        assert row['residual_atoms']==0
        atoms=[a for unit in row['owners'] for a in unit]
        assert len(atoms)==len(set(atoms))==Chem.MolFromSmiles(smiles).GetNumAtoms()


def synthetic(label):
    return dict(status='candidate_complete',native={'native_1':1},
                units=[dict(options=(f'residual_detail:{label}','residual_core:N'),atoms=[0])])


def test_support_counts_molecules_and_backs_off_before_discarding():
    rows=[synthetic('a') for _ in range(12)]+[synthetic('b') for _ in range(10)]
    vocabulary,_,supported,_=fit_vocabulary(rows,20)
    assert np.all(supported)
    assert vocabulary.accepted=={'residual_core:N'}
    assert vocabulary.support['residual_core:N']==22
    # A test-only detailed environment must not enter the fitted vocabulary.
    matrix,ok,_=encode([synthetic('unseen')],vocabulary)
    assert ok.tolist()==[True]
    assert matrix.sum()==2


def test_unsupported_companions_do_not_inflate_regressable_support():
    rows=[synthetic('a') for _ in range(20)]
    for i,row in enumerate(rows):
        row['units'].append(dict(options=(f'rare_detail:{i}',f'rare_core:{i}'),atoms=[1]))
    vocabulary,_,supported,_=fit_vocabulary(rows,20)
    assert not np.any(supported)
    assert not vocabulary.accepted


def test_native_successes_retain_counts_and_singleton_boundaries_stay_out():
    row=extract_study('CCO')
    assert row['native']=={'native_1':1,'native_7':1,'native_36':1}
    assert not row['units']
    assert extract_study('C')['status']=='unsupported_boundary'


def test_complete_native_partitions_are_preserved_for_multiple_amides():
    smiles='CC(=O)NCCNC(=O)C'
    original=nm.fragment(Chem.MolFromSmiles(smiles))
    row=extract_study(smiles)
    assert row['native_complete']
    assert row['amide_systems']==0
    assert row['native']=={'native_'+str(group):count for group,count in original.groups.items()}
