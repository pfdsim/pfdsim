"""Study-only conflict repair and train-supported residual group features.

Production Nannoolal remains unchanged. Reuse its function code and matching
registry in an isolated namespace with the sulfonamide priority repaired.
Replace overlapping multi-carbonyl/N systems by disjoint structural units;
encode remaining unassigned atoms by detailed/backoff local environments.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import FunctionType

import numpy as np
from rdkit import Chem

from benchmark_nannoolal_solubility import nm

NATIVE_AMIDE_GROUPS = {48,49,50,99,100}


def isolated_fragmenter():
    namespace = dict(nm.__dict__)
    # Group 82's broad C-S(=O)2 match otherwise consumes the S/O atoms
    # needed by the more specific sulfonamide group 105.
    registry = [(16.5 if gid==105 else priority,gid,pattern,claim,predicate)
                for priority,gid,pattern,claim,predicate in nm._ASSIGNERS]
    namespace['_ASSIGNERS'] = sorted(registry,key=lambda item:item[0])
    return FunctionType(nm.fragment.__code__,namespace,'study_fragment',nm.fragment.__defaults__)


STUDY_FRAGMENT = isolated_fragmenter()


def native_state(molecule,fragmenter=STUDY_FRAGMENT):
    try:
        result = fragmenter(molecule)
        return result.mol,result,[],''
    except nm.FragmentationError as error:
        tb = error.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code is fragmenter.__code__:
                local = tb.tb_frame.f_locals
                return local['mol'],local.get('frag'),[a.GetIdx() for a in local.get('unclaimed',[])],str(error)
            tb = tb.tb_next
        raise RuntimeError('Native failure state unavailable') from error


def amide_systems(molecule,instances):
    """Merge overlapping native amide candidates and any intersecting units."""
    candidates = []
    for _,gid,pattern,claim,predicate in nm._ASSIGNERS:
        if gid not in NATIVE_AMIDE_GROUPS:
            continue
        for match in molecule.GetSubstructMatches(pattern):
            if predicate is not None and not predicate(molecule,match):
                continue
            atoms = set(match if claim is None else claim(molecule,match) if callable(claim)
                        else (match[i] for i in claim))
            candidates.append(atoms)
    # Connected hypergraph components: matching and native assigned units.
    units = candidates+[set(atoms) for _,atoms in instances]
    components = []
    for atoms in units:
        merged = set(atoms)
        changed = True
        while changed:
            changed = False
            kept = []
            for old in components:
                if old&merged:
                    merged |= old
                    changed = True
                else:
                    kept.append(old)
            components = kept
        components.append(merged)
    systems = []
    for atoms in components:
        carbonyls = [i for i in atoms if molecule.GetAtomWithIdx(i).GetAtomicNum()==6
                     and any(b.GetBondType()==Chem.BondType.DOUBLE and b.GetOtherAtomIdx(i) in atoms
                             and b.GetOtherAtom(molecule.GetAtomWithIdx(i)).GetAtomicNum()==8
                             for b in molecule.GetAtomWithIdx(i).GetBonds())]
        if len(carbonyls)>=2 and any(atoms&candidate for candidate in candidates):
            systems.append(atoms)
    return systems


def atom_options(atom):
    def neighbor_label(bond,detailed):
        neighbor = bond.GetOtherAtom(atom)
        text = f'{neighbor.GetSymbol()}:a{int(neighbor.GetIsAromatic())}:q{neighbor.GetFormalCharge()}:{bond.GetBondType()}'
        if detailed:
            text += f':H{neighbor.GetTotalNumHs()}:d{neighbor.GetDegree()}'
        return text
    center = (f'{atom.GetSymbol()}:a{int(atom.GetIsAromatic())}:q{atom.GetFormalCharge()}:'
              f'H{atom.GetTotalNumHs()}:r{int(atom.IsInRing())}:rad{atom.GetNumRadicalElectrons()}')
    detailed = 'residual_detail:'+center+'|'+';'.join(sorted(neighbor_label(b,True) for b in atom.GetBonds()))
    coarse = 'residual_core:'+center+'|'+';'.join(sorted(neighbor_label(b,False) for b in atom.GetBonds()))
    return detailed,coarse


def system_options(molecule,atoms):
    counts = Counter(molecule.GetAtomWithIdx(i).GetSymbol() for i in atoms)
    carbonyl_count = sum(1 for i in atoms if molecule.GetAtomWithIdx(i).GetAtomicNum()==6
                         and any(b.GetBondType()==Chem.BondType.DOUBLE
                                 and b.GetOtherAtom(molecule.GetAtomWithIdx(i)).GetAtomicNum()==8
                                 for b in molecule.GetAtomWithIdx(i).GetBonds()))
    n_h = sorted(a.GetTotalNumHs() for i in atoms if (a:=molecule.GetAtomWithIdx(i)).GetAtomicNum()==7)
    aromatic = sum(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)
    ring = any(molecule.GetAtomWithIdx(i).IsInRing() for i in atoms)
    core = ('amide_system_core:'+','.join(f'{element}{n}' for element,n in sorted(counts.items()))+
            f'|CO={carbonyl_count}|NH={n_h}|aromatic={aromatic}|ring={int(ring)}')
    boundary = sorted(f'{molecule.GetAtomWithIdx(i).GetSymbol()}-{neighbor.GetSymbol()}:'
                      f'a{int(neighbor.GetIsAromatic())}:{bond.GetBondType()}'
                      for i in atoms for bond in molecule.GetAtomWithIdx(i).GetBonds()
                      if (neighbor:=bond.GetOtherAtom(molecule.GetAtomWithIdx(i))).GetIdx() not in atoms)
    return core.replace('_core:','_detail:')+'|boundary='+';'.join(boundary),core


def extract_study(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('Invalid SMILES')
    active,fragmentation,missing,error = native_state(molecule,nm.fragment)
    native_complete = fragmentation is not None and not error
    if not native_complete:
        active,fragmentation,missing,error = native_state(molecule)
    if fragmentation is None:
        # Preserve the native minimum-size refusal; its single methane
        # example cannot support a new regression group.
        return dict(status='unsupported_boundary',reason=error,native={},units=[],owners=[])
    instances = [(str(gid),set(atoms)) for gid,atoms in fragmentation.instances]
    # Preserve complete native units. Only failed native assignments need
    # replacement by the experimental composite representation.
    systems = [] if not missing else amide_systems(active,instances)
    occupied = set().union(*systems) if systems else set()
    native = Counter()
    owners = []
    for gid,atoms in instances:
        if atoms&occupied:
            assert atoms<=occupied
            continue
        native['native_'+gid] += 1
        owners.append(sorted(atoms))
    for atom in active.GetAtoms():
        if atom.GetAtomicNum()==14:
            n_ch = sum(n.GetAtomicNum()==6 for n in atom.GetNeighbors())+atom.GetTotalNumHs()
            label = (f'native_Si_CH{n_ch}_hal{int(any(n.GetAtomicNum() in (9,17,35,53) for n in atom.GetNeighbors()))}'
                     f'_O{int(any(n.GetAtomicNum()==8 for n in atom.GetNeighbors()))}')
            native[label] += 1
            owners.append([atom.GetIdx()])
    units = []
    for atoms in systems:
        units.append(dict(options=system_options(active,atoms),atoms=sorted(atoms),kind='amide_system'))
        owners.append(sorted(atoms))
    covered = {atom for ownership in owners for atom in ownership}
    residual = [atom for atom in active.GetAtoms() if atom.GetIdx() not in covered]
    for atom in residual:
        units.append(dict(options=atom_options(atom),atoms=[atom.GetIdx()],kind='residual_atom'))
        owners.append([atom.GetIdx()])
    assigned = [i for group in owners for i in group]
    assert len(assigned)==len(set(assigned))==active.GetNumAtoms()
    assert set(assigned)==set(range(active.GetNumAtoms()))
    return dict(status='candidate_complete',native=dict(native),units=units,owners=owners,
                native_complete=native_complete,repaired_native_complete=not missing,amide_systems=len(systems),
                residual_atoms=len(residual),native_error=error)


@dataclass
class SupportedVocabulary:
    labels: list[str]
    accepted: set[str]
    support: dict[str,int]
    minimum: int


def encode(rows,vocabulary):
    positions = {label:i for i,label in enumerate(vocabulary.labels)}
    matrix = np.zeros((len(rows),len(positions)))
    supported = np.array([row['status']=='candidate_complete' for row in rows])
    choices = []
    for index,row in enumerate(rows):
        counts = Counter(row['native'])
        for unit in row['units']:
            label = next((candidate for candidate in unit['options'] if candidate in vocabulary.accepted),None)
            if label is None:
                supported[index] = False
            else:
                counts[label] += 1
        for label,n in counts.items():
            if label in positions:
                matrix[index,positions[label]] = n
            elif label.startswith('native_'):
                supported[index] = False
        choices.append(dict(counts))
    return matrix,supported,choices


def fit_vocabulary(rows,minimum=20):
    """Require support among fully representable training molecules.

    Reject undersupported *selected* groups, then back off and recompute.
    This avoids counting occurrences that were actually assigned to a more
    detailed group, or molecules unavailable because of another rare group.
    """
    candidates = {label for row in rows for unit in row['units'] for label in unit['options']}
    native = {label for row in rows for label in row['native']}
    eligible = np.array([row['status']=='candidate_complete' for row in rows])
    while True:
        possible = Counter()
        for row,ok in zip(rows,eligible):
            if ok:
                possible.update({label for unit in row['units'] for label in unit['options']})
        accepted = {label for label in candidates if possible[label]>=minimum}
        temporary = SupportedVocabulary(sorted(native|accepted),accepted,{},minimum)
        _,supported,chosen = encode(rows,temporary)
        new_eligible = eligible&supported
        if not np.array_equal(eligible,new_eligible):
            eligible = new_eligible
            continue
        actual = Counter(label for row,ok in zip(chosen,eligible) if ok for label in row if label in accepted)
        rare = {label for label,n in actual.items() if n<minimum}
        if rare:
            detailed = {label for label in rare if '_detail:' in label}
            candidates -= detailed or rare
            continue
        retained_native = {label for row,ok in zip(rows,eligible) if ok for label in row['native']}
        labels = sorted(retained_native|set(actual))
        vocabulary = SupportedVocabulary(labels,set(actual),dict(actual),minimum)
        matrix,supported,chosen = encode(rows,vocabulary)
        assert np.array_equal(supported,eligible)
        assert all(n>=minimum for n in actual.values())
        return vocabulary,matrix,supported,chosen
