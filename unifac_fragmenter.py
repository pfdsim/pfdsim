"""Native RDKit fragmenter for the UNIFAC parameter families.

The fragmenter assigns every heavy atom exactly once.  Functional and
parameter-set-specific groups are claimed before the carbon skeleton; if an
atom cannot be represented by the selected parameter family the molecule is
rejected instead of being silently approximated with another UNIFAC variant.

The public result uses subgroup numbers, so it can be passed directly to
``UNIFACModel._resolve_groups``.  The three supported families are the
original UNIFAC table (``UNIFAC``/``UNIFAC2``), Dortmund modified UNIFAC
(``UNIFDMD``/``UNIFM2``), and NIST modified UNIFAC (``UNIFNIST``).
"""

from __future__ import annotations

from collections import Counter

from rdkit import Chem


CLASSIC_VARIANTS = frozenset({'UNIFAC', 'UNIFAC2'})
DORTMUND_VARIANTS = frozenset({'UNIFDMD', 'UNIFM2'})
NIST_VARIANTS = frozenset({'UNIFNIST'})
SUPPORTED_VARIANTS = CLASSIC_VARIANTS | DORTMUND_VARIANTS | NIST_VARIANTS


class UNIFACFragmentationError(ValueError):
    """The structure cannot be represented unambiguously by the group table."""


def _canonical_variant(variant: str) -> str:
    value = str(variant).upper().strip()
    if value not in SUPPORTED_VARIANTS:
        raise UNIFACFragmentationError(f'unsupported UNIFAC variant: {variant}')
    return value


def fragment(smiles: str, variant: str = 'UNIFAC') -> dict[int, int]:
    """Return ``{subgroup_number: count}`` for a SMILES structure.

    The implementation intentionally has a strict domain.  Unsupported ring
    heterocycles and exotic functional groups raise
    :class:`UNIFACFragmentationError`; callers must not substitute Dortmund
    groups into NIST or classic parameter sets after such a failure.
    """
    if str(variant).upper().strip() == 'UNIFLBY':
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .lyngby_parameters import convert_classic_groups
        else:
            from lyngby_parameters import convert_classic_groups
        return convert_classic_groups(fragment(smiles, 'UNIFAC'))
    variant = _canonical_variant(variant)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise UNIFACFragmentationError(f'could not parse SMILES {smiles!r}')
    if not mol.GetNumHeavyAtoms():
        raise UNIFACFragmentationError('structure has no heavy atoms')
    canonical = Chem.MolToSmiles(mol, canonical=True)

    groups: Counter[int] = Counter()
    claimed: set[int] = set()

    def reject(reason: str) -> None:
        raise UNIFACFragmentationError(reason)

    def take(group: int, *atoms: int, count: int = 1) -> None:
        atom_set = set(atoms)
        overlap = atom_set & claimed
        if overlap:
            reject(
                f'internal overlap while assigning subgroup {group}: '
                f'atoms {sorted(overlap)} were already claimed'
            )
        claimed.update(atom_set)
        groups[group] += count

    def available(*atoms: int) -> bool:
        return not (set(atoms) & claimed)

    def matches(smarts: str):
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:  # pragma: no cover - static rule programming error
            raise RuntimeError(f'invalid internal SMARTS: {smarts}')
        return mol.GetSubstructMatches(pattern, uniquify=True)

    def atom(idx: int):
        return mol.GetAtomWithIdx(idx)

    def carbon_neighbors(idx: int, excluded: set[int] | None = None):
        excluded = excluded or set()
        return [
            nb for nb in atom(idx).GetNeighbors()
            if nb.GetSymbol() == 'C' and nb.GetIdx() not in excluded
        ]

    # Dortmund whole-ion subgroups. These groups describe the complete ionic
    # core, so exact canonical structures are appropriate rather than atom
    # fragments borrowed from neutral chemistry.
    if variant in DORTMUND_VARIANTS:
        ionic_examples = {
            '[B-](F)(F)(F)F': {195: 1},
            '[P-](F)(F)(F)(F)(F)F': {211: 1},
            'O=S(=O)([O-])[O-]': {209: 1},
            'O=S(=O)(O)[O-]': {210: 1},
            'O=S(=O)([O-])C(F)(F)F': {197: 1},
            'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F': {179: 1},
            '[NH2+]1CCCC1': {189: 1},
            'c1cc[nH+]cc1': {196: 1},
            'C[n+]1cc(C)n(C)c1': {1: 3, 178: 1},
            'C[n+]1ccn(C)c1': {1: 2, 184: 1},
            'Cc1cccc[nH+]1': {1: 1, 220: 1},
        }
        canonical_ions = {
            Chem.MolToSmiles(Chem.MolFromSmiles(value), canonical=True): result
            for value, result in ionic_examples.items()
        }
        ionic = canonical_ions.get(canonical)
        if ionic is not None:
            return dict(ionic)

    nitro_atoms: set[int] = set()
    for nitro_smarts in ('[#6][NX3+](=[OX1])[OX1-]', '[#6][NX3](=[OX1])=[OX1]'):
        for nitro_match in mol.GetSubstructMatches(Chem.MolFromSmarts(nitro_smarts)):
            nitro_atoms.update(nitro_match[1:])

    for current in mol.GetAtoms():
        if current.GetFormalCharge() or current.GetNumRadicalElectrons():
            if current.GetIdx() not in nitro_atoms:
                reject('charged and radical structures require explicit UNIFAC groups')
        if current.GetSymbol() not in {'C', 'H', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'Si'}:
            reject(f'element {current.GetSymbol()} is unsupported')

    is_nist = variant in NIST_VARIANTS
    is_modified = variant not in CLASSIC_VARIANTS

    # --- NIST-only whole-molecule and high-specificity groups -------------
    if is_nist and canonical == Chem.MolToSmiles(Chem.MolFromSmiles('OCC(O)CO'), canonical=True):
        take(204, *(a.GetIdx() for a in mol.GetAtoms()))
        return dict(groups)

    # NIST's dedicated ethylene-glycol group is also present in the classic
    # and Dortmund tables as DOH, subgroup 62.
    if canonical == Chem.MolToSmiles(Chem.MolFromSmiles('OCCO'), canonical=True):
        take(62, *(a.GetIdx() for a in mol.GetAtoms()))
        return dict(groups)

    exact_common = {
        Chem.MolToSmiles(Chem.MolFromSmiles('CS(C)=O'), canonical=True): {67: 1},
        Chem.MolToSmiles(Chem.MolFromSmiles('S=C=S'), canonical=True): {58: 1},
        Chem.MolToSmiles(Chem.MolFromSmiles('O=Cc1ccco1'), canonical=True): {61: 1},
        Chem.MolToSmiles(Chem.MolFromSmiles('C=CC#N'), canonical=True): {68: 1},
        Chem.MolToSmiles(Chem.MolFromSmiles('CN(C)C=O'), canonical=True): {72: 1},
    }
    exact = exact_common.get(canonical)
    if exact is not None:
        claimed.update(a.GetIdx() for a in mol.GetAtoms())
        groups.update(exact)
        return dict(sorted(groups.items()))

    if variant in CLASSIC_VARIANTS:
        classic_exact = {
            Chem.MolToSmiles(Chem.MolFromSmiles('CN1CCCC(=O)1'), canonical=True):
                {85: 1},
            Chem.MolToSmiles(Chem.MolFromSmiles('C1COCCN1'), canonical=True):
                {105: 1},
        }
        exact = classic_exact.get(canonical)
        if exact is not None:
            return dict(exact)

    if is_modified:
        modified_exact = {
            Chem.MolToSmiles(Chem.MolFromSmiles('C1COCOC1'), canonical=True):
                {78: 1, 83: 2},
            Chem.MolToSmiles(Chem.MolFromSmiles('C1OCOCO1'), canonical=True):
                {84: 3},
        }
        exact = modified_exact.get(canonical)
        if exact is not None:
            return dict(exact)

    if is_nist:
        exact_nist = {
            Chem.MolToSmiles(Chem.MolFromSmiles('ClC(Cl)=C(Cl)Cl'), canonical=True):
                {179: 1},
        }
        exact = exact_nist.get(canonical)
        if exact is not None:
            claimed.update(a.GetIdx() for a in mol.GetAtoms())
            groups.update(exact)
            return dict(sorted(groups.items()))

    # Classic and NIST silicon groups share the same structural split but use
    # different subgroup-number ranges. Dortmund has no silicon family.
    silicon_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == 'Si']
    if silicon_atoms:
        if variant in DORTMUND_VARIANTS:
            reject('silicon groups are absent from Dortmund modified UNIFAC')
        plain = ({3: 78, 2: 79, 1: 80, 0: 81}
                 if variant in CLASSIC_VARIANTS
                 else {3: 170, 2: 171, 1: 172, 0: 173})
        oxygenated = ({2: 82, 1: 83, 0: 84}
                      if variant in CLASSIC_VARIANTS
                      else {2: 174, 1: 175, 0: 176})
        for silicon in sorted(silicon_atoms,
                              key=lambda current: (current.GetTotalNumHs(),
                                                   current.GetIdx())):
            sidx = silicon.GetIdx()
            if sidx in claimed:
                continue
            available_oxygen = [
                nb for nb in silicon.GetNeighbors()
                if nb.GetSymbol() == 'O' and nb.GetIdx() not in claimed
            ]
            h_count = silicon.GetTotalNumHs()
            if available_oxygen and h_count in oxygenated:
                take(oxygenated[h_count], sidx, available_oxygen[0].GetIdx())
            else:
                subgroup = plain.get(h_count)
                if subgroup is None:
                    reject('silicon substitution pattern is absent from this table')
                take(subgroup, sidx)

    if is_modified:
        for match in matches('[#6][OX2][CX3](=[OX1])[OX2][#6]'):
            left, oxygen1, carbonyl, carbonyl_o, oxygen2, right = match
            if not available(*match):
                continue
            if atom(left).GetIsAromatic() or atom(right).GetIsAromatic():
                continue  # NIST aromatic carbonate groups are handled below.
            h_pair = tuple(sorted((atom(left).GetTotalNumHs(),
                                   atom(right).GetTotalNumHs()), reverse=True))
            subgroup = {(3, 3): 112, (2, 2): 113, (3, 2): 114}.get(h_pair)
            if subgroup is not None:
                take(subgroup, *match)

    # N,N-diethylformamide is the HCON(CH2)2 group in all three tables and
    # must precede the generic tertiary-amide rules.
    for match in matches('[CX3H1](=[OX1])[NX3]([CX4H2])[CX4H2]'):
        if available(*match):
            take(73, *match)

    # Variant-specific acyclic amides.
    for match in matches('[CX3](=[OX1])[NX3]'):
        carbonyl, oxygen, nitrogen = match
        if not available(*match) or atom(carbonyl).IsInRing() or atom(nitrogen).IsInRing():
            continue
        n_atom = atom(nitrogen)
        substituents = [
            nb for nb in n_atom.GetNeighbors()
            if nb.GetSymbol() == 'C' and nb.GetIdx() != carbonyl
        ]
        carbonyl_h = atom(carbonyl).GetTotalNumHs()
        subgroup = None
        atoms_to_take = [carbonyl, oxygen, nitrogen]
        if carbonyl_h == 1 and is_modified and n_atom.GetTotalNumHs() == 1 \
                and len(substituents) == 1:
            host = substituents[0]
            subgroup = {3: 93, 2: 94}.get(host.GetTotalNumHs())
            atoms_to_take.append(host.GetIdx())
        elif carbonyl_h == 0:
            if n_atom.GetTotalNumHs() == 2 and not substituents:
                subgroup = 94 if variant in CLASSIC_VARIANTS \
                    else 91 if variant in DORTMUND_VARIANTS else None
            elif n_atom.GetTotalNumHs() == 1 and len(substituents) == 1:
                host = substituents[0]
                table = ({3: 95, 2: 96} if variant in CLASSIC_VARIANTS
                         else {3: 92, 2: 100})
                subgroup = table.get(host.GetTotalNumHs())
                atoms_to_take.append(host.GetIdx())
            elif n_atom.GetTotalNumHs() == 0 and len(substituents) == 2:
                h_pair = tuple(sorted((nb.GetTotalNumHs() for nb in substituents),
                                      reverse=True))
                table = ({(3, 3): 97, (3, 2): 98, (2, 2): 99}
                         if variant in CLASSIC_VARIANTS
                         else {(3, 3): 101, (3, 2): 102, (2, 2): 103})
                subgroup = table.get(h_pair)
                atoms_to_take.extend(nb.GetIdx() for nb in substituents)
        if subgroup is not None:
            take(subgroup, *atoms_to_take)

    if is_modified:
        for match in matches('[CX3;R](=[OX1])[NX3;R][CX4;!R]'):
            carbonyl, oxygen, nitrogen, external = match
            if not available(*match):
                continue
            subgroup = {3: 86, 2: 87, 1: 88, 0: 89}.get(
                atom(external).GetTotalNumHs()
            )
            if subgroup is not None:
                take(subgroup, *match)

    if variant in CLASSIC_VARIANTS:
        for match in matches('[OX2H1][CX4H2][CX4][OX2H0]'):
            hydroxyl_o, carbon1, carbon2, ether_o = match
            if not available(*match):
                continue
            subgroup = 100 if atom(carbon2).GetTotalNumHs() == 2 else 101 \
                if atom(carbon2).GetTotalNumHs() == 1 else None
            if subgroup is not None:
                take(subgroup, *match)

    # Acetals: the central carbon and its two oxygens form the NIST group;
    # substituent carbons remain for ordinary skeleton assignment.
    if is_nist:
        for match in matches('[CX4]([OX2][#6])([OX2][#6])'):
            center, oxygen1, _sub1, oxygen2, _sub2 = match
            if not available(center, oxygen1, oxygen2):
                continue
            hydrogens = atom(center).GetTotalNumHs()
            subgroup = {2: 309, 1: 186, 0: 152}.get(hydrogens)
            if subgroup is None:
                reject('unsupported acetal carbon environment')
            take(subgroup, center, oxygen1, oxygen2)

        # Aldoxime and ketoxime.  The source reused number 309 for CH=NOH;
        # pfdsim preserves it under collision-free internal number 1309.
        for match in matches('[CX3]=[NX2][OX2H1]'):
            carbon, nitrogen, oxygen = match
            if not available(*match):
                continue
            take(1309 if atom(carbon).GetTotalNumHs() else 177,
                 carbon, nitrogen, oxygen)

        # Vicinal diols use the dedicated NIST/modified-UNIFAC DOH groups.
        for match in matches('[CX4]([OX2H1])-[CX4]([OX2H1])'):
            carbon1, oxygen1, carbon2, oxygen2 = match
            if not available(*match):
                continue
            h1, h2 = atom(carbon1).GetTotalNumHs(), atom(carbon2).GetTotalNumHs()
            pair = tuple(sorted((h1, h2), reverse=True))
            subgroup = {
                (2, 1): 205,
                (1, 1): 206,
                (2, 0): 207,
                (1, 0): 208,
                (0, 0): 209,
            }.get(pair)
            if subgroup is not None:
                take(subgroup, carbon1, oxygen1, carbon2, oxygen2)

        # Isocyanates include the directly attached carbon or aromatic ring
        # atom together with N=C=O.
        for match in matches('[#6][NX2]=[CX2]=[OX1]'):
            host, nitrogen, central, oxygen = match
            if not available(*match):
                continue
            host_atom = atom(host)
            if host_atom.GetIsAromatic():
                subgroup = 120
            else:
                subgroup = {3: 117, 2: 118, 1: 119}.get(
                    host_atom.GetTotalNumHs()
                )
            if subgroup is None:
                reject('isocyanate host cannot be represented by NIST UNIFAC')
            take(subgroup, *match)

        # Carbonates: both alkoxy host atoms are part of the composite group.
        for match in matches('[#6][OX2][CX3](=[OX1])[OX2][#6]'):
            left, oxygen1, carbonyl, carbonyl_o, oxygen2, right = match
            if not available(*match):
                continue
            left_atom, right_atom = atom(left), atom(right)
            aryl_count = int(left_atom.GetIsAromatic()) + int(right_atom.GetIsAromatic())
            if aryl_count == 2:
                subgroup = 200
            elif aryl_count == 1:
                alkyl = right_atom if left_atom.GetIsAromatic() else left_atom
                subgroup = 199 if alkyl.GetTotalNumHs() == 2 else None
            else:
                h_pair = tuple(sorted((left_atom.GetTotalNumHs(),
                                       right_atom.GetTotalNumHs()), reverse=True))
                subgroup = {(3, 3): 112, (2, 2): 113, (3, 2): 114}.get(h_pair)
            if subgroup is None:
                reject('carbonate substituents are absent from NIST UNIFAC')
            take(subgroup, *match)

        # Cyclic amides and esters use whole ring-functional groups.  The
        # remaining saturated ring carbons are assigned c-CH2/c-CH later.
        for match in matches('[CX3;R](=[OX1])[NX3H1;R]'):
            if available(*match):
                take(125, *match)
        for match in matches('[CX3;R](=[OX1])[OX2;R]'):
            if available(*match):
                take(126, *match)

        # Glycol ethers are single NIST groups spanning ether O, two adjacent
        # carbons, and the terminal hydroxyl O.
        for match in matches('[OX2]([#6])[CX4]-[CX4][OX2H1]'):
            ether_o, _ether_other, carbon1, carbon2, hydroxyl_o = match
            idxs = (ether_o, carbon1, carbon2, hydroxyl_o)
            if not available(*idxs):
                continue
            pair = (atom(carbon1).GetTotalNumHs(), atom(carbon2).GetTotalNumHs())
            subgroup = {(2, 2): 131, (1, 2): 132, (2, 1): 133}.get(pair)
            if subgroup is not None:
                take(subgroup, *idxs)

    # --- carbonyl families -------------------------------------------------
    if variant in CLASSIC_VARIANTS:
        for match in matches('[NX2]=[CX2]=[OX1]'):
            if available(*match):
                take(109, *match)

    # Anhydrides claim both carbonyl carbons, both carbonyl oxygens, and the
    # bridging oxygen.  NIST has subgroup 121; classic/Dortmund do not.
    for match in matches('[CX3](=[OX1])[OX2][CX3](=[OX1])'):
        if not available(*match):
            continue
        if not is_nist:
            reject('anhydrides require NIST subgroup 121 or explicit groups')
        take(121, *match)

    # NIST aryl esters are a distinct family and must precede generic esters.
    if is_nist:
        for match in matches('[c][OX2][CX3](=[OX1])[CX4]'):
            aryl, ester_o, carbonyl, carbonyl_o, alkyl = match
            if not available(*match):
                continue
            subgroup = {3: 127, 2: 128, 1: 129, 0: 130}.get(
                atom(alkyl).GetTotalNumHs()
            )
            if subgroup is None:
                reject('unsupported aryl-ester alkyl group')
            take(subgroup, *match)

    if is_nist:
        for match in matches('[c][CX3](=[OX1])[OX2H1]'):
            if available(*match):
                take(124, *match)

    # Carboxylic acids.
    for match in matches('[CX3](=[OX1])[OX2H1]'):
        carbon, oxygen, hydroxyl = match
        if not available(*match):
            continue
        take(43 if atom(carbon).GetTotalNumHs() else 42, *match)

    # Esters.  The subgroup includes the acyl-side carbon, not the O-side
    # carbon (ethyl acetate = CH3COO + CH2 + CH3).
    for match in matches('[CX3](=[OX1])[OX2][CX4]'):
        carbonyl, carbonyl_o, ester_o, o_carbon = match
        if not available(*match):
            continue
        if atom(carbonyl).GetTotalNumHs():
            subgroup = 23
            take(subgroup, carbonyl, carbonyl_o, ester_o)
            continue
        acyl_carbons = carbon_neighbors(carbonyl, {o_carbon})
        if len(acyl_carbons) != 1:
            reject('ester acyl side cannot be represented uniquely')
        acyl_carbon = acyl_carbons[0]
        subgroup = {3: 21, 2: 22}.get(acyl_carbon.GetTotalNumHs())
        if subgroup is None:
            take(77, carbonyl, carbonyl_o, ester_o)
        else:
            take(subgroup, carbonyl, carbonyl_o, ester_o,
                 acyl_carbon.GetIdx())

    # NIST aromatic aldehydes and ketones have dedicated ring-attached groups.
    if is_nist:
        for match in matches('[c][CX3H1]=[OX1]'):
            if available(*match):
                take(123, *match)
        for match in matches('[c][CX3](=[OX1])[#6]'):
            aryl, carbonyl, oxygen, _substituent = match
            if available(aryl, carbonyl, oxygen):
                take(178, aryl, carbonyl, oxygen)

    # Aldehydes.
    if is_nist:
        for match in matches('[CX3H2]=[OX1]'):
            if available(*match):
                take(308, *match)
    for match in matches('[CX3H1]=[OX1]'):
        if not available(*match):
            continue
        subgroup = 308 if is_nist and not carbon_neighbors(match[0]) else 20
        take(subgroup, *match)

    # Ketones.  CH3CO/CH2CO/CHCO/CCO include one alpha carbon plus C=O.
    for match in matches('[#6][CX3](=[OX1])[#6]'):
        left, carbonyl, oxygen, right = match
        if not available(carbonyl, oxygen):
            continue
        candidates = [idx for idx in (left, right) if available(idx)]
        if not candidates:
            reject('ketone has no available alpha-carbon subgroup')
        if is_nist and any(atom(idx).GetTotalNumHs() <= 1 for idx in candidates):
            alpha = min(candidates,
                        key=lambda idx: (atom(idx).GetTotalNumHs(), idx))
        else:
            alpha = max(candidates,
                        key=lambda idx: (atom(idx).GetTotalNumHs(), -idx))
        h_count = atom(alpha).GetTotalNumHs()
        subgroup = {3: 18, 2: 19}.get(h_count)
        if is_nist:
            subgroup = {3: 18, 2: 19, 1: 301, 0: 302}.get(h_count)
        if subgroup is None:
            reject('ketone alpha carbon cannot be represented by this table')
        take(subgroup, alpha, carbonyl, oxygen)

    # --- nitrogen and sulfur functional groups ----------------------------
    if is_nist:
        # Whole conjugated nitrile groups precede ordinary CHxCN assignment.
        for match in matches('[c][CX2]#[NX1]'):
            if available(*match):
                take(116, *match)
        for match in matches('[CX3H2]=[CX3H1][CX2]#[NX1]'):
            if available(*match):
                take(68, *match)

    # Nitriles include the adjacent carbon in CH3CN/CH2CN/etc.
    for match in matches('[#6][CX2]#[NX1]'):
        alpha, nitrile_c, nitrogen = match
        if not available(*match):
            continue
        h_count = atom(alpha).GetTotalNumHs()
        subgroup = {3: 40, 2: 41}.get(h_count)
        if is_nist:
            subgroup = {3: 40, 2: 41, 1: 303, 0: 304}.get(h_count)
        if subgroup is None:
            reject('nitrile alpha carbon cannot be represented by this table')
        take(subgroup, *match)

    # Nitro groups include their attached carbon.
    nitro_seen: set[int] = set()
    for smarts in ('[#6][NX3+](=[OX1])[OX1-]', '[#6][NX3](=[OX1])=[OX1]'):
        for match in matches(smarts):
            host, nitrogen, oxygen1, oxygen2 = match
            if nitrogen in nitro_seen or not available(*match):
                continue
            nitro_seen.add(nitrogen)
            if atom(host).GetIsAromatic():
                subgroup = 57
            else:
                h_count = atom(host).GetTotalNumHs()
                subgroup = {3: 54, 2: 55, 1: 56}.get(h_count)
                if is_nist:
                    subgroup = {3: 54, 2: 55, 1: 56, 0: 305}.get(h_count)
            if subgroup is None:
                reject('nitro-bearing carbon cannot be represented by this table')
            take(subgroup, *match)

    # Aromatic amines.  NIST distinguishes primary, secondary, tertiary,
    # and diaryl environments; each group includes the aromatic host carbon.
    if is_nist:
        for nitrogen in [a for a in mol.GetAtoms() if a.GetSymbol() == 'N'
                         and not a.GetIsAromatic()]:
            nidx = nitrogen.GetIdx()
            if nidx in claimed:
                continue
            aryl = [nb for nb in nitrogen.GetNeighbors() if nb.GetIsAromatic()]
            alkyl = [nb for nb in nitrogen.GetNeighbors()
                     if nb.GetSymbol() == 'C' and not nb.GetIsAromatic()]
            if len(aryl) == 2 and nitrogen.GetTotalNumHs() == 1:
                take(306, nidx, aryl[0].GetIdx())
                continue
            if len(aryl) != 1:
                continue
            aryl_idx = aryl[0].GetIdx()
            if nitrogen.GetTotalNumHs() == 1 and len(alkyl) == 1:
                host = alkyl[0]
                subgroup = {3: 156, 2: 157, 1: 158}.get(host.GetTotalNumHs())
                if subgroup is not None:
                    take(subgroup, nidx, aryl_idx, host.GetIdx())
                    continue
            if nitrogen.GetTotalNumHs() == 0 and len(alkyl) == 2:
                h_pair = tuple(sorted((a.GetTotalNumHs() for a in alkyl),
                                      reverse=True))
                subgroup = {(3, 3): 153, (3, 2): 154, (2, 2): 155}.get(h_pair)
                if subgroup is not None:
                    take(subgroup, nidx, aryl_idx,
                         *(a.GetIdx() for a in alkyl))
                    continue
            if nitrogen.GetTotalNumHs() == 0 and len(alkyl) == 2:
                take(307, nidx, aryl_idx)

    # Primary aromatic amines.
    for match in matches('[c][NX3H2]'):
        if available(*match):
            take(36, *match)

    if is_nist:
        # Benzylic tertiary primary amine (NIST ACN) and cycloalkyl amine.
        for match in matches('[c][CX4]([NX3H2])([CH3])[CH3]'):
            aryl, host, nitrogen, _methyl1, _methyl2 = match
            if available(aryl, host, nitrogen):
                take(307, aryl, host, nitrogen)
        for match in matches('[CX4;R][NX3H2]'):
            if available(*match):
                take(180, *match)

        # Saturated cyclic amines.  Each subgroup consumes N, one adjacent
        # ring carbon, and (for tertiary N) its exocyclic substituent.
        for nitrogen in [a for a in mol.GetAtoms() if a.GetSymbol() == 'N'
                         and a.IsInRing() and not a.GetIsAromatic()]:
            nidx = nitrogen.GetIdx()
            if nidx in claimed:
                continue
            ring_hosts = [nb for nb in nitrogen.GetNeighbors()
                          if nb.GetSymbol() == 'C' and nb.IsInRing()
                          and nb.GetIdx() not in claimed]
            if not ring_hosts:
                continue
            ring_host = min(ring_hosts,
                            key=lambda a: (a.GetTotalNumHs(), a.GetIdx()))
            ring_h = ring_host.GetTotalNumHs()
            external = [nb for nb in nitrogen.GetNeighbors()
                        if nb.GetSymbol() == 'C' and not nb.IsInRing()]
            if nitrogen.GetTotalNumHs() == 1 and not external:
                subgroup = {2: 188, 1: 162, 0: 163}.get(ring_h)
                atoms_to_take = (nidx, ring_host.GetIdx())
            elif nitrogen.GetTotalNumHs() == 0 and len(external) == 1:
                external_host = external[0]
                table = ({3: 189, 2: 190, 1: 191} if ring_h == 2
                         else {3: 164, 2: 165, 1: 166} if ring_h == 1
                         else {})
                subgroup = table.get(external_host.GetTotalNumHs())
                atoms_to_take = (nidx, ring_host.GetIdx(),
                                 external_host.GetIdx())
            else:
                subgroup = None
                atoms_to_take = ()
            if subgroup is not None:
                take(subgroup, *atoms_to_take)

    # Aliphatic primary, secondary, and tertiary amines.  One attached carbon
    # is included in the subgroup; remaining substituent carbons are skeleton.
    for nitrogen in [a for a in mol.GetAtoms() if a.GetSymbol() == 'N']:
        nidx = nitrogen.GetIdx()
        if nidx in claimed:
            continue
        if nitrogen.GetIsAromatic() or nitrogen.IsInRing():
            continue
        attached = [nb for nb in nitrogen.GetNeighbors() if nb.GetSymbol() == 'C']
        if not attached:
            reject('nitrogen without a carbon substituent is unsupported')
        host = max(attached, key=lambda a: (a.GetTotalNumHs(), -a.GetIdx()))
        h_count = host.GetTotalNumHs()
        n_h = nitrogen.GetTotalNumHs()
        table = (
            ({3: 28, 2: 29, 1: 30, 0: 85} if is_modified
             else {3: 28, 2: 29, 1: 30}) if n_h == 2 else
            {3: 31, 2: 32, 1: 33} if n_h == 1 else
            {3: 34, 2: 35}
        )
        subgroup = table.get(h_count)
        if subgroup is None:
            reject('amine substituent cannot be represented by this table')
        take(subgroup, nidx, host.GetIdx())

    # Thiols include sulfur and the host carbon.
    for match in matches('[#6][SX2H1]'):
        host, sulfur = match
        if not available(*match):
            continue
        h_count = atom(host).GetTotalNumHs()
        subgroup = {3: 59, 2: 60}.get(h_count)
        if is_nist:
            subgroup = {3: 59, 2: 60, 1: 192, 0: 193}.get(h_count)
        if atom(host).GetIsAromatic() and is_nist:
            subgroup = 194
        if subgroup is None:
            reject('thiol host cannot be represented by this table')
        take(subgroup, *match)

    if is_modified:
        for match in matches('[#6;R][SX4;R](=[OX1])(=[OX1])[#6;R]'):
            left, sulfur, oxygen1, oxygen2, right = match
            if not available(*match):
                continue
            h_pair = tuple(sorted((atom(left).GetTotalNumHs(),
                                   atom(right).GetTotalNumHs()), reverse=True))
            subgroup = 110 if h_pair == (2, 2) else 111 if h_pair == (2, 1) else None
            if subgroup is not None:
                take(subgroup, *match)

    if variant in DORTMUND_VARIANTS:
        for match in matches('[SX2][SX2]'):
            if available(*match):
                take(201, *match)

    if not is_nist:
        sulfide_table = ({3: 102, 2: 103, 1: 104}
                         if variant in CLASSIC_VARIANTS
                         else {3: 122, 2: 123, 1: 124})
        for sulfur in [a for a in mol.GetAtoms() if a.GetSymbol() == 'S'
                       and not a.GetIsAromatic() and a.GetTotalNumHs() == 0]:
            sidx = sulfur.GetIdx()
            if sidx in claimed:
                continue
            hosts = [nb for nb in sulfur.GetNeighbors() if nb.GetSymbol() == 'C']
            if len(hosts) != 2:
                continue
            host = max(hosts, key=lambda current: (current.GetTotalNumHs(),
                                                   -current.GetIdx()))
            subgroup = sulfide_table.get(host.GetTotalNumHs())
            if subgroup is not None:
                take(subgroup, sidx, host.GetIdx())

    if is_nist:
        # Sulfones consume sulfur and both oxygens; aromatic substitution has
        # its own group, while sulfolane uses the two adjacent ring carbons.
        for match in matches('[#6][SX4](=[OX1])(=[OX1])[#6]'):
            left, sulfur, oxygen1, oxygen2, right = match
            if not available(*match):
                continue
            if atom(left).GetIsAromatic() or atom(right).GetIsAromatic():
                aryl = left if atom(left).GetIsAromatic() else right
                take(122, aryl, sulfur, oxygen1, oxygen2)
            elif atom(left).IsInRing() and atom(right).IsInRing():
                take(110, *match)

        # Sulfides include sulfur plus one host carbon; aromatic sulfide ACS
        # claims the aromatic host instead of the alkyl side.
        for sulfur in [a for a in mol.GetAtoms() if a.GetSymbol() == 'S'
                       and not a.GetIsAromatic() and a.GetTotalNumHs() == 0]:
            sidx = sulfur.GetIdx()
            if sidx in claimed:
                continue
            hosts = [nb for nb in sulfur.GetNeighbors() if nb.GetSymbol() == 'C']
            if len(hosts) != 2:
                continue
            aromatic = [host for host in hosts if host.GetIsAromatic()]
            if aromatic:
                take(187, sidx, aromatic[0].GetIdx())
                continue
            host = min(hosts, key=lambda a: (a.GetTotalNumHs(), a.GetIdx()))
            subgroup = {3: 134, 2: 135, 1: 136, 0: 137}.get(
                host.GetTotalNumHs()
            )
            if subgroup is not None:
                take(subgroup, sidx, host.GetIdx())

        # Peroxides include O-O and one attached host carbon.
        for match in matches('[#6][OX2][OX2][#6]'):
            left, oxygen1, oxygen2, right = match
            if not available(*match):
                continue
            hosts = [atom(left), atom(right)]
            aromatic = [host for host in hosts if host.GetIsAromatic()]
            if aromatic:
                subgroup = 142
                host = aromatic[0]
            else:
                host = min(hosts, key=lambda a: (a.GetTotalNumHs(), a.GetIdx()))
                subgroup = {3: 138, 2: 139, 1: 140, 0: 141}.get(
                    host.GetTotalNumHs()
                )
            if subgroup is not None:
                take(subgroup, oxygen1, oxygen2, host.GetIdx())

    # --- oxygen groups ----------------------------------------------------
    # Phenol includes the aromatic host carbon.
    for match in matches('[c][OX2H1]'):
        if available(*match):
            take(17, *match)

    # Methanol is a single composite subgroup.
    if canonical == 'CO' and not claimed:
        take(15, *(a.GetIdx() for a in mol.GetAtoms()))
        return dict(groups)

    # Alcohol oxygen only; the host carbon remains a skeleton group.
    for match in matches('[#6][OX2H1]'):
        host, oxygen = match
        if oxygen in claimed:
            continue
        if not available(oxygen):
            continue
        if is_modified:
            h_count = atom(host).GetTotalNumHs()
            subgroup = 14 if h_count >= 2 else 81 if h_count == 1 else 82
        else:
            subgroup = 14
        take(subgroup, oxygen)

    # Water (isolated neutral O with two hydrogens).
    for oxygen in [a for a in mol.GetAtoms() if a.GetSymbol() == 'O']:
        if oxygen.GetIdx() not in claimed and oxygen.GetDegree() == 0 \
                and oxygen.GetTotalNumHs() == 2:
            take(16, oxygen.GetIdx())

    # Three-membered epoxide rings precede the generic ring-ether rule.
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 3:
            continue
        ring_atoms = [atom(idx) for idx in ring]
        oxygens = [a for a in ring_atoms if a.GetSymbol() == 'O']
        carbons = [a for a in ring_atoms if a.GetSymbol() == 'C']
        if len(oxygens) != 1 or len(carbons) != 2:
            continue
        idxs = tuple(ring)
        if not available(*idxs):
            continue
        if not is_modified:
            reject('epoxides require Dortmund/NIST groups or explicit groups')
        h_pair = tuple(sorted((carbons[0].GetTotalNumHs(),
                               carbons[1].GetTotalNumHs()), reverse=True))
        if variant in DORTMUND_VARIANTS:
            subgroup = {(2, 2): 119, (2, 1): 107, (2, 0): 153,
                        (1, 1): 109, (1, 0): 108}.get(h_pair)
        else:
            subgroup = {(2, 1): 107, (1, 1): 109}.get(h_pair)
        if subgroup is None:
            reject('epoxide substitution pattern is absent from this table')
        take(subgroup, *idxs)

    # Ring ethers.  Modified UNIFAC's THF group includes the oxygen and both
    # adjacent ring methylenes; classic THF includes O plus one methylene.
    for oxygen in [a for a in mol.GetAtoms()
                   if a.GetSymbol() == 'O' and a.IsInRing()
                   and not a.GetIsAromatic()]:
        oidx = oxygen.GetIdx()
        if oidx in claimed or oxygen.GetTotalNumHs() != 0:
            continue
        neighbors = [nb for nb in oxygen.GetNeighbors() if nb.GetSymbol() == 'C' and nb.IsInRing()]
        if len(neighbors) != 2:
            reject('unsupported ring-ether environment')
        if is_modified:
            idxs = (oidx, neighbors[0].GetIdx(), neighbors[1].GetIdx())
        else:
            chosen = max(neighbors, key=lambda a: (a.GetTotalNumHs(), -a.GetIdx()))
            idxs = (oidx, chosen.GetIdx())
        if available(*idxs):
            take(27, *idxs)

    # Acyclic ethers include O plus one attached carbon.  This rule fixes the
    # historical DME error: COC = CH3 + CH3O, never 2 CH3 + CH3O.
    for oxygen in [a for a in mol.GetAtoms() if a.GetSymbol() == 'O']:
        oidx = oxygen.GetIdx()
        if oidx in claimed or oxygen.GetTotalNumHs() != 0 \
                or oxygen.GetIsAromatic():
            continue
        neighbors = [nb for nb in oxygen.GetNeighbors() if nb.GetSymbol() == 'C']
        if len(neighbors) != 2:
            reject('unsupported ether oxygen environment')
        host = max(neighbors, key=lambda a: (a.GetTotalNumHs(), -a.GetIdx()))
        h_count = host.GetTotalNumHs()
        subgroup = {3: 24, 2: 25, 1: 26}.get(h_count)
        if subgroup is None or not available(oidx, host.GetIdx()):
            reject('ether carbon cannot be represented without overlap')
        take(subgroup, oidx, host.GetIdx())

    # --- aromatic heterocycles -------------------------------------------
    # Original UNIFAC uses whole-ring pyridine and thiophene subgroups;
    # Dortmund/NIST replaced them with segment groups.
    if variant in CLASSIC_VARIANTS:
        for ring in mol.GetRingInfo().AtomRings():
            ring_atoms = [atom(idx) for idx in ring]
            if not all(current.GetIsAromatic() for current in ring_atoms):
                continue
            symbols = Counter(current.GetSymbol() for current in ring_atoms)
            if len(ring) == 6 and symbols == Counter({'C': 5, 'N': 1}):
                n_h = sum(current.GetTotalNumHs() for current in ring_atoms
                          if current.GetSymbol() == 'C')
                subgroup = {5: 37, 4: 38, 3: 39}.get(n_h)
            elif len(ring) == 5 and symbols == Counter({'C': 4, 'S': 1}):
                n_h = sum(current.GetTotalNumHs() for current in ring_atoms
                          if current.GetSymbol() == 'C')
                subgroup = {4: 106, 3: 107, 2: 108}.get(n_h)
            else:
                continue
            if subgroup is None:
                reject('substituted aromatic heterocycle is absent from classic UNIFAC')
            if available(*ring):
                take(subgroup, *ring)

    if is_nist:
        for nitrogen in [a for a in mol.GetAtoms()
                         if a.GetSymbol() == 'N' and a.GetIsAromatic()
                         and a.GetTotalNumHs() == 1]:
            nidx = nitrogen.GetIdx()
            if nidx in claimed:
                continue
            neighbors = [nb for nb in nitrogen.GetNeighbors()
                         if nb.GetSymbol() == 'C' and nb.GetIsAromatic()]
            if len(neighbors) != 2:
                reject('unsupported pyrrole nitrogen environment')
            n_h = sum(nb.GetTotalNumHs() for nb in neighbors)
            subgroup = {2: 196, 1: 197, 0: 198}.get(n_h)
            if subgroup is None:
                reject('pyrrole substitution pattern is absent from NIST UNIFAC')
            take(subgroup, nidx, neighbors[0].GetIdx(), neighbors[1].GetIdx())

    # Pyridine: N plus its two adjacent aromatic carbons is one AC2...N group.
    for nitrogen in [a for a in mol.GetAtoms() if a.GetSymbol() == 'N' and a.GetIsAromatic()]:
        nidx = nitrogen.GetIdx()
        if nidx in claimed:
            continue
        neighbors = [nb for nb in nitrogen.GetNeighbors() if nb.GetIsAromatic() and nb.GetSymbol() == 'C']
        if len(neighbors) != 2:
            reject('unsupported aromatic-nitrogen environment')
        idxs = (nidx, neighbors[0].GetIdx(), neighbors[1].GetIdx())
        if not available(*idxs):
            reject('overlapping aromatic-nitrogen group')
        n_h = sum(nb.GetTotalNumHs() for nb in neighbors)
        subgroup = {2: 37, 1: 38, 0: 39}.get(n_h)
        if subgroup is None:
            reject('pyridine subgroup cannot be selected')
        take(subgroup, *idxs)

    # Furan and thiophene use the analogous heteroatom-plus-two-carbons group.
    sulfur_ids = (106, 107, 108) if variant in CLASSIC_VARIANTS else (104, 105, 106)
    for symbol, subgroup_ids in (('O', (159, 160, 161)), ('S', sulfur_ids)):
        for hetero in [a for a in mol.GetAtoms() if a.GetSymbol() == symbol and a.GetIsAromatic()]:
            hidx = hetero.GetIdx()
            if hidx in claimed:
                continue
            if symbol == 'O' and not is_nist:
                reject('furan requires explicit classic/Dortmund groups')
            neighbors = [nb for nb in hetero.GetNeighbors() if nb.GetSymbol() == 'C' and nb.GetIsAromatic()]
            if len(neighbors) != 2:
                reject(f'unsupported aromatic {symbol} environment')
            idxs = (hidx, neighbors[0].GetIdx(), neighbors[1].GetIdx())
            if not available(*idxs):
                reject(f'overlapping aromatic {symbol} group')
            n_h = sum(nb.GetTotalNumHs() for nb in neighbors)
            subgroup = subgroup_ids[2 - n_h]
            take(subgroup, *idxs)
    # --- unsaturation, halogens, aromatic substituents, skeleton ----------
    # Carbon-carbon triple bonds are one composite subgroup.
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.TRIPLE:
            continue
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if left.GetSymbol() != 'C' or right.GetSymbol() != 'C':
            continue
        idxs = (left.GetIdx(), right.GetIdx())
        if available(*idxs):
            take(65 if left.GetTotalNumHs() or right.GetTotalNumHs() else 66, *idxs)

    # Acyclic C=C pairs.
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if left.GetSymbol() != 'C' or right.GetSymbol() != 'C' \
                or left.GetIsAromatic() or left.IsInRing() or right.IsInRing():
            continue
        idxs = (left.GetIdx(), right.GetIdx())
        if not available(*idxs):
            continue
        h_pair = tuple(sorted((left.GetTotalNumHs(), right.GetTotalNumHs()), reverse=True))
        subgroup = {(2, 1): 5, (1, 1): 6, (2, 0): 7,
                    (1, 0): 8, (0, 0): 70}.get(h_pair)
        if subgroup is None:
            reject('unsupported carbon-carbon double-bond environment')
        take(subgroup, *idxs)

    # Cyclic C=C pairs use NIST's c-C=C family.
    if is_nist:
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            left, right = bond.GetBeginAtom(), bond.GetEndAtom()
            if left.GetSymbol() != 'C' or right.GetSymbol() != 'C' \
                    or not left.IsInRing() or not right.IsInRing():
                continue
            idxs = (left.GetIdx(), right.GetIdx())
            if not available(*idxs):
                continue
            h_pair = tuple(sorted((left.GetTotalNumHs(),
                                   right.GetTotalNumHs()), reverse=True))
            subgroup = {(1, 1): 201, (1, 0): 202, (0, 0): 203}.get(h_pair)
            if subgroup is None:
                reject('cyclic alkene substitution pattern is unsupported')
            take(subgroup, *idxs)
    else:
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            left, right = bond.GetBeginAtom(), bond.GetEndAtom()
            if left.GetSymbol() != 'C' or right.GetSymbol() != 'C' \
                    or not left.IsInRing() or not right.IsInRing():
                continue
            idxs = (left.GetIdx(), right.GetIdx())
            if not available(*idxs):
                continue
            h_pair = tuple(sorted((left.GetTotalNumHs(),
                                   right.GetTotalNumHs()), reverse=True))
            subgroup = {(2, 1): 5, (1, 1): 6, (2, 0): 7,
                        (1, 0): 8, (0, 0): 70}.get(h_pair)
            if subgroup is None:
                reject('ring alkene substitution pattern is unsupported')
            take(subgroup, *idxs)

    # Aromatic carbon with a directly attached alkyl carbon (toluene family).
    for aryl in [a for a in mol.GetAtoms() if a.GetSymbol() == 'C' and a.GetIsAromatic()]:
        aidx = aryl.GetIdx()
        if aidx in claimed:
            continue
        substituents = [
            nb for nb in aryl.GetNeighbors()
            if nb.GetSymbol() == 'C' and not nb.GetIsAromatic()
            and nb.GetIdx() not in claimed
        ]
        if len(substituents) != 1:
            continue
        host = substituents[0]
        subgroup = {3: 11, 2: 12, 1: 13}.get(host.GetTotalNumHs())
        if is_nist and host.GetTotalNumHs() == 0:
            subgroup = 195
        if subgroup is not None:
            take(subgroup, aidx, host.GetIdx())

    # Halogenated carbons are composite subgroups.  Mixed F/Cl environments
    # outside the explicitly represented classic family are rejected.
    for chlorine in [a for a in mol.GetAtoms() if a.GetSymbol() == 'Cl'
                     and a.GetIdx() not in claimed]:
        if len(chlorine.GetNeighbors()) != 1:
            continue
        host = chlorine.GetNeighbors()[0]
        if host.GetSymbol() == 'C' and any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(host).GetSymbol() == 'C'
                for bond in host.GetBonds()):
            take(69, chlorine.GetIdx())

    for carbon in [a for a in mol.GetAtoms() if a.GetSymbol() == 'C']:
        cidx = carbon.GetIdx()
        if cidx in claimed:
            continue
        halogens = [nb for nb in carbon.GetNeighbors()
                    if nb.GetSymbol() in {'F', 'Cl', 'Br', 'I'}
                    and nb.GetIdx() not in claimed]
        if not halogens:
            continue
        symbols = Counter(nb.GetSymbol() for nb in halogens)
        idxs = (cidx, *(nb.GetIdx() for nb in halogens))
        standalone_halogen = (
            len(halogens) == 1 and halogens[0].GetSymbol() in {'Br', 'I'}
        )
        if standalone_halogen:
            take(64 if halogens[0].GetSymbol() == 'Br' else 63,
                 halogens[0].GetIdx())
            continue
        if carbon.GetIsAromatic():
            if len(halogens) != 1:
                reject('polyhalogenated aromatic carbon is unsupported')
            subgroup = {'Cl': 53, 'F': 71, 'Br': 64, 'I': 63}.get(halogens[0].GetSymbol())
        elif set(symbols) == {'Cl'}:
            subgroup = {
                (1, 2): 44, (1, 1): 45, (1, 0): 46,
                (2, 2): 47, (2, 1): 48, (2, 0): 49,
                (3, 1): 50, (3, 0): 51, (4, 0): 52,
            }.get((symbols['Cl'], carbon.GetTotalNumHs()))
        elif variant in CLASSIC_VARIANTS and set(symbols) <= {'F', 'Cl'} \
                and 'F' in symbols and 'Cl' in symbols:
            key = (symbols['F'], symbols['Cl'], carbon.GetTotalNumHs())
            subgroup = {
                (1, 3, 0): 86,
                (1, 2, 0): 87,
                (1, 2, 1): 88,
                (1, 1, 1): 89,
                (2, 1, 0): 90,
                (2, 1, 1): 91,
                (3, 1, 0): 92,
                (2, 2, 0): 93,
            }.get(key)
        elif is_nist and set(symbols) <= {'F', 'Cl'} and 'F' in symbols:
            key = (symbols['F'], symbols['Cl'], carbon.GetTotalNumHs())
            subgroup = {
                (1, 0, 1): 143,
                (1, 1, 0): 144,
                (1, 2, 0): 145,
                (2, 0, 1): 146,
                (2, 1, 1): 147,
                (2, 2, 0): 148,
                (3, 0, 1): 149,
                (3, 1, 0): 150,
                (4, 0, 0): 151,
            }.get(key)
            if subgroup is None and not symbols['Cl']:
                subgroup = {3: 74, 2: 75, 1: 76}.get(symbols['F'])
        elif set(symbols) == {'F'}:
            subgroup = {3: 74, 2: 75, 1: 76}.get(symbols['F'])
        else:
            subgroup = None
        if subgroup is None:
            reject(f'unsupported halogen environment {dict(symbols)}')
        take(subgroup, *idxs)

    # Remaining carbon atoms are ordinary skeleton subgroups.
    for carbon in [a for a in mol.GetAtoms() if a.GetSymbol() == 'C']:
        idx = carbon.GetIdx()
        if idx in claimed:
            continue
        if carbon.GetIsAromatic():
            take(9 if carbon.GetTotalNumHs() else 10, idx)
            continue
        if carbon.GetHybridization() != Chem.HybridizationType.SP3:
            reject(f'unassigned unsaturated carbon atom {idx}')
        h_count = carbon.GetTotalNumHs()
        if is_modified and carbon.IsInRing():
            subgroup = {2: 78, 1: 79, 0: 80}.get(h_count)
        else:
            subgroup = {3: 1, 2: 2, 1: 3, 0: 4}.get(h_count)
        if subgroup is None:
            reject(f'unsupported carbon environment at atom {idx}')
        take(subgroup, idx)

    leftovers = [
        current for current in mol.GetAtoms()
        if current.GetAtomicNum() > 1 and current.GetIdx() not in claimed
    ]
    if leftovers:
        detail = ', '.join(f'{a.GetSymbol()}({a.GetIdx()})' for a in leftovers[:8])
        reject(f'unassigned heavy atom(s): {detail}')
    if not groups:
        reject('no UNIFAC groups were assigned')
    return dict(sorted(groups.items()))


def fragment_safe(smiles: str, variant: str = 'UNIFAC') -> dict[int, int] | None:
    """Return a fragmentation or ``None`` when the strict engine rejects it."""
    try:
        return fragment(smiles, variant)
    except (UNIFACFragmentationError, ValueError):
        return None
