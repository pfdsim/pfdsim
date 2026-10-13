"""Offline fusion estimators for the benchmark; never imported by production.

Chickos values are entropy contributions, not enthalpy contributions. Both
versions require the measured melting temperature. Published total-transition
entropy must not be confused with the entropy of the final fusion transition.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem

R = 8.314  # Value used in the source papers, J/(mol K).


class NotApplicable(ValueError):
    """A structure is outside an implemented, source-defined assignment."""


@dataclass(frozen=True)
class Estimate:
    entropy_J_mol_K: float
    details: dict

    def enthalpy_kJ_mol(self, melting_temperature_K: float) -> float:
        return self.entropy_J_mol_K * melting_temperature_K / 1000.0


@lru_cache(maxsize=8192)
def molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise NotApplicable("invalid SMILES")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NotApplicable("disconnected structure/salt/solvate")
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()):
        raise NotApplicable("net-charged molecule")
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise NotApplicable("radical")
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        raise NotApplicable("no carbon skeleton")
    return mol


@lru_cache(maxsize=None)
def pattern(smarts: str):
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise RuntimeError(f"Invalid benchmark SMARTS: {smarts}")
    return query


def ring_systems(mol):
    """Single/fused/spiro/conjugated ring systems, not RDKit's SSSR count."""
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        group = set(ring)
        touching = [s for s in systems if s & group]
        for other in touching:
            group |= other
            systems.remove(other)
        systems.append(group)
    # The Jain paper counts conjugated rings together; biphenyl is one system.
    changed = True
    while changed:
        changed = False
        for i, first in enumerate(systems):
            for second in systems[i + 1:]:
                joined = any(
                    b.GetBeginAtomIdx() in first and b.GetEndAtomIdx() in second
                    or b.GetEndAtomIdx() in first and b.GetBeginAtomIdx() in second
                    for b in mol.GetBonds() if b.GetIsConjugated()
                )
                if joined:
                    first |= second
                    systems.remove(second)
                    changed = True
                    break
            if changed:
                break
    return systems


def jain_domain(mol):
    if any(sum(mol.GetAtomWithIdx(i).GetSymbol() == 'C' for i in r) >= 8
           and not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
           for r in mol.GetRingInfo().AtomRings()):
        raise NotApplicable('Jain excludes nonaromatic rings with >=8 carbons')


def flexibility(smiles: str, *, revised: bool = True) -> dict:
    """DY 1996 Eq. 7 / Jain 2004 Eq. IV, including terminal exclusions."""
    mol = molecule(smiles)
    if any(a.GetSymbol() not in {"C", "H", "N", "O", "S", "F", "Cl", "Br", "I"}
           for a in mol.GetAtoms()):
        raise NotApplicable("flexibility outside CHNOS/halogen descriptor domain")
    if revised:
        jain_domain(mol)
    nitro_n = {m[0] for m in mol.GetSubstructMatches(pattern("[N+](=O)[O-]"))}
    sulfonyl_s = {m[0] for m in mol.GetSubstructMatches(pattern("S(=O)(=O)"))}
    sp3 = sp2 = 0
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True))
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if atom.IsInRing() or atom.GetDegree() <= 1:
            continue
        symbol = atom.GetSymbol()
        if symbol not in {"C", "N", "O", "S"}:
            continue
        # Cyano is a freely rotating terminal group, not a chain atom.
        if any(b.GetBondType() == Chem.BondType.TRIPLE for b in atom.GetBonds()):
            continue
        neighbors = atom.GetNeighbors()
        # Both papers exclude tert-butyl; Jain also explicitly excludes CF3 etc.
        identical_terminals = Counter(ranks[n.GetIdx()] for n in neighbors
                                      if n.GetDegree() == 1)
        if symbol == "C" and max(identical_terminals.values(), default=0) == 3:
            if revised or any(n.GetSymbol() == "C" and n.GetTotalNumHs() == 3
                              for n in neighbors):
                continue
        if i in nitro_n or i in sulfonyl_s:
            sp2 += 1
        elif symbol in {'O', 'S'} and atom.GetTotalValence() == 2:
            sp3 += 1  # Empirical O/S chain descriptor, including conjugated ethers.
        elif symbol == 'N' and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()):
            sp3 += 1  # Amide resonance does not turn the paper's NH/N descriptor into =N.
        elif atom.GetHybridization() == Chem.HybridizationType.SP3:
            sp3 += 1
        elif atom.GetHybridization() == Chem.HybridizationType.SP2:
            sp2 += 1
        else:
            raise NotApplicable(f"unassigned flexibility center: {symbol}")
    rings = len(ring_systems(mol))
    tau = max(0.0, sp3 + 0.5 * sp2 + 0.5 * rings - 1.0)
    if revised and tau > 18:
        raise NotApplicable("Jain published validation excludes tau >18")
    return {"sp3": sp3, "sp2": sp2, "ring_systems": rings, "tau": tau}


def descriptor_entropy(sigma: float, tau: float, *, revised: bool = True) -> Estimate:
    """Use source-assigned descriptors without approximating symmetry."""
    if not math.isfinite(sigma) or sigma < 1 or not math.isfinite(tau) or tau < 0:
        raise NotApplicable("invalid symmetry/flexibility descriptor")
    if revised and tau > 18:
        raise NotApplicable("Jain tau >18")
    base = 2.435 if revised else 2.85
    value = 50.0 - R * math.log(sigma) + R * tau * math.log(base)
    if value <= 0:
        raise NotApplicable("nonpositive predicted entropy")
    return Estimate(value, {"sigma": sigma, "tau": tau, "flexibility_base": base,
                            "descriptor_origin": "published author assignment"})


# Pattern atoms are ALL group atoms: no arbitrary neighboring carbon is claimed.
# Specific composite groups precede their constituent groups. Ring variants are
# selected by the center's ring membership. Unsupported groups are rejected.
FUNCTIONAL_GROUPS = (
    ("urea", "[N]-[C](=[O])-[N]"),
    ("carbonate", "[O;H0]-[C](=[O])-[O;H0]"),
    ("carbamate", "[O;H0]-[C](=[O])-[N]"),
    ("imide", "[C](=[O])-[N]-[C](=[O])"),
    ("anhydride", "[C](=[O])-[O;H0]-[C](=[O])"),
    ("sulfonamide", "[S](=[O])(=[O])-[N]"),
    ("sulfonic_acid", "[S](=[O])(=[O])-[O;H1]"),
    ("sulfonate", "[S](=[O])(=[O])-[O;H0]"),
    ("nitramine", "[N]-[N+](=[O])-[O-]"),
    ("nitrosamine", "[N]-[N]=[O]"),
    ("nitrate", "[O;H0]-[N+](=[O])-[O-]"),
    ("nitro", "[N+](=[O])-[O-]"),
    ("acid", "[C](=[O])-[O;H1]"),
    ("ester", "[C](=[O])-[O;H0]"),
    ("acyl_chloride", "[C](=[O])-[Cl]"),
    ("amide", "[C](=[O])-[N]"),
    ("thiourea", "[N]-[C](=[S])-[N]"),
    ("thioamide", "[C](=[S])-[N]"),
    ("isocyanate", "[N]=[C]=[O]"),
    ("oxime", "[N]=[C]"),  # Special handling below; C remains a chain center.
    ("nitrile", "[C]#[N]"),
    ("disulfide", "[S]-[S]"),
    ("peroxide", "[O]-[O]"),
    ("sulfone", "[S](=[O])(=[O])"),
    ("sulfoxide", "[S](=[O])"),
    ("carbonyl", "[C]=[O]"),
)


def _functional_assignment(mol):
    groups = []
    claimed = set()
    special_carbon = set()
    for kind, smarts in FUNCTIONAL_GROUPS:
        if kind == "oxime":
            continue  # Generic imines are assigned as nitrogen centers below.
        for match in mol.GetSubstructMatches(pattern(smarts)):
            if claimed.intersection(match):
                continue
            atoms = [mol.GetAtomWithIdx(i) for i in match]
            center = next((a for a in atoms if a.GetSymbol() == "C"
                           and any(b.GetBondType() == Chem.BondType.DOUBLE
                                   and b.GetOtherAtom(a).GetSymbol() in {"O", "S"}
                                   for b in a.GetBonds())), atoms[0])
            ring = center.IsInRing()
            subtype = kind
            if kind in {"amide", "carbamate", "imide", "sulfonamide", "thioamide"}:
                nitrogen = next(a for a in atoms if a.GetSymbol() == "N")
                subtype += f"_{nitrogen.GetTotalNumHs()}H"
                ring = center.IsInRing() and nitrogen.IsInRing()
                if kind == 'amide' and center.GetTotalNumHs() == 1 and nitrogen.GetTotalNumHs() == 0:
                    subtype = 'formamide_0H'
            elif kind in {"urea", "thiourea"}:
                hydrogens = sorted(a.GetTotalNumHs() for a in atoms if a.GetSymbol() == "N")
                subtype += "_" + "".join(map(str, hydrogens))
                ring = all(a.IsInRing() for a in atoms if a.GetSymbol() in {"C", "N"})
            elif kind in {"carbonate", "anhydride", "ester", "formate"}:
                ring = all(a.IsInRing() for a in atoms if a.GetDegree() > 1)
                if kind == "ester" and center.GetTotalNumHs() == 1:
                    subtype = "formate"
            elif kind == "carbonyl":
                subtype = "aldehyde" if center.GetTotalNumHs() == 1 else "ketone"
            groups.append((subtype, ring, set(match)))
            claimed.update(match)
            special_carbon.update(i for i in match if mol.GetAtomWithIdx(i).GetSymbol() == "C")
    for atom in mol.GetAtoms():
        i, symbol = atom.GetIdx(), atom.GetSymbol()
        if i in claimed or symbol in {"C", "H"}:
            continue
        ring = atom.IsInRing()
        if symbol in {"F", "Cl", "Br", "I"}:
            kind = symbol
        elif symbol == "O":
            if atom.GetTotalNumHs() == 1:
                kind = "phenol" if any(n.GetIsAromatic() for n in atom.GetNeighbors()) else "alcohol"
            elif atom.GetDegree() == 2:
                kind = "ether"
            else:
                raise NotApplicable("unsupported oxygen environment")
        elif symbol == "N":
            if any(n.GetSymbol() not in {"C", "H"} for n in atom.GetNeighbors()):
                raise NotApplicable("composite N-O/N-N environment requires a dedicated published group")
            if atom.GetIsAromatic() and atom.GetTotalNumHs() == 0:
                kind = "aromatic_n"
            elif atom.GetTotalValence() == 3:
                kind = f"amine_{atom.GetTotalNumHs()}H"
                if any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()):
                    kind = "imine"
            else:
                raise NotApplicable("unsupported nitrogen environment")
        elif symbol == "S" and atom.GetDegree() == 1 and atom.GetTotalNumHs() == 1:
            kind = "thiol"
        elif symbol == "S" and atom.GetDegree() == 2:
            kind = "sulfide"
        else:
            raise NotApplicable(f"unsupported functional center {symbol}")
        groups.append((kind, ring, {i}))
        claimed.add(i)
    return groups, special_carbon


# Perry Tables 2-162/163; (entropy contribution, carbon-bearing coefficient).
PERRY_CARBON = {
    "sp3_3": (18.33, 1), "sp3_2": (9.41, 1), "sp3_1": (-16.91, .69),
    "sp3_0": (-38.70, .67), "sp2_2": (14.56, 1), "sp2_1": (4.85, 3.23),
    "sp2_0": (-11.38, 1), "sp_1": (10.88, 1), "sp_0": (2.18, 1),
    "ar_H": (6.44, 1), "ar_sp3": (-10.33, 1), "ar_sp2": (-4.27, 1),
    "ar_sp": (-2.51, 1), "ring_sp3_1": (-15.98, .76),
    "ring_sp3_0": (-32.97, 1), "ring_sp2_1": (-4.35, .62),
    "ring_sp2_0": (-11.72, .86), "ring_sp_0": (-5.36, 1),
}
UPDATED_CARBON = {
    "sp3_3": (17.6, 1), "sp3_2": (7.1, 1), "sp3_1": (-16.4, .60),
    "sp3_0": (-34.8, .66), "sp2_2": (17.3, 1), "sp2_1": (5.3, .75),
    "sp2_0": (-10.7, 1), "sp_1": (14.9, 1), "sp_0": (-2.8, 1),
    "ar_H": (7.4, 1), "ar_sp3": (-9.6, 1), "ar_sp2": (-7.5, 1),
    "ar_internal": (-.7, 1), "ring_sp3_1": (-14.7, 1),
    "ring_sp3_0": (-34.6, 1), "ring_sp2_1": (-1.6, 1.92),
    "ring_sp2_0": (-12.3, 1), "ring_sp_0": (-4.7, 1),
}
PERRY_FUNCTIONAL = {
    "alcohol": (1.13, [1, 12.6, 18.9, 26.4]), "phenol": (16.57, [1]),
    "ether": (1.09, [1]), "ether_ring": (1.34, [1]),
    "ketone": (3.14, [1]), "ketone_ring": (-1.88, [1]),
    "aldehyde": (19.66, [1]), "acid": (14.90, [1, 1.83, 1.88, 1.72]),
    "ester": (3.68, [1]), "amine_2H": (16.23, [1]),
    "amine_2H_ar": (15.48, [1]), "amine_1H": (-2.18, [1]),
    "amine_1H_ring": (1.84, [1]), "amine_0H": (-15.90, [1]),
    "amine_0H_ring": (-17.07, [1]), "imine_ring": (1.67, [1]),
    "aromatic_n": (7.32, [1]), "nitrile": (9.62, [1, 1.4]),
    "nitro": (17.36, [1]), "amide_2H": (26.19, [1]),
    "amide_1H": (-.42, [1]), "thiol": (17.99, [1]),
    "sulfide": (7.20, [1, 1, 1, .36]), "sulfide_ring": (2.18, [1]),
    "sulfone": (3.26, [1]), "F_sp3": (14.73, [1]),
    "F_sp2": (13.01, [1]), "F_ring": (15.90, [1]),
    "Cl": (8.37, [1, 2, 2, 1.93]), "Br": (17.95, [1, 1, 1, .82]),
    "I": (16.95, [1]),
}
UPDATED_FUNCTIONAL = {
    "alcohol": (1.7, [1, 10.4, 9.7, 13.1, 12.1, 13.1]), "phenol": (20.3, [1]),
    "ether": (4.71, [1]), "ether_ring": (1.2, [1]),
    "peroxide": (10.6, [1]), "peroxide_ring": (27.7, [1]),
    "ketone": (4.6, [1]), "ketone_ring": (-1.4, [1]),
    "aldehyde": (21.5, [1]), "acid": (13.4, [1, 1.21, 2.25, 2.25, 2.25, 2.25]),
    "formate": (4.2, [1]), "ester": (7.7, [1]), "ester_ring": (3.1, [1]),
    "carbonate_ring": (1.3, [1]),
    "anhydride": (10.0, [1]), "anhydride_ring": (2.3, [1]),
    "acyl_chloride": (25.8, [1]), "aromatic_n": (10.9, [1]),
    "imine": (-1.8, [1]), "imine_ring": (.5, [1]),
    "amine_0H": (-22.2, [1]), "amine_0H_ring": (-19.3, [1]),
    "amine_1H": (-5.3, [1]), "amine_1H_ring": (2.2, [1]),
    "amine_2H": (21.4, [1]), "nitro": (17.7, [1]),
    "nitrate": (24.4, [1]), "nitrile": (17.7, [1]),
    "amide_0H": (-11.2, [1]), "amide_0H_ring": (-21.7, [1]),
    "formamide_0H": (6.9, [1]),
    "amide_1H": (1.5, [1]), "amide_1H_ring": (2.7, [1]), "amide_2H": (27.9, [1]),
    "urea_00": (-19.3, [1]), "urea_01": (.2, [1, -12.8, -24, 6]),
    "urea_02": (19.5, [1]), "urea_11": (1.5, [1]), "urea_12": (22.5, [1]),
    "carbamate_0H": (-23.12, [1]), "carbamate_0H_ring": (-5.2, [1]),
    "carbamate_1H": (10.6, [1]), "carbamate_1H_ring": (19.7, [1]),
    "carbamate_2H": (27.9, [1]), "imide_1H": (7.7, [1]),
    "imide_1H_ring": (1.4, [1]), "imide_0H_ring": (1.1, [1]),
    "sulfide": (2.1, [1]), "sulfide_ring": (2.9, [1]),
    "disulfide": (9.6, [1]), "disulfide_ring": (-6.4, [1]),
    "thiol": (23.0, [1]), "sulfoxide": (14.1, [1]),
    "sulfone": (.3, [1]), "sulfone_ring": (-10.4, [1]),
    "sulfonate": (7.9, [1]), "sulfonic_acid": (1.8, [1]),
    "sulfonamide_0H": (-11.3, [1]), "sulfonamide_1H": (6.3, [1]),
    "sulfonamide_1H_ring": (-.4, [1]), "sulfonamide_2H": (28.4, [1]),
    "thiourea_11": (14.4, [1]), "thiourea_12": (23.1, [1]),
    "thioamide_2H": (30.0, [1]), "isocyanate": (23.1, [1]),
    "F_sp2": (19.5, [1]), "F_ar": (16.6, [1]),
    "F_sp3_3": (13.3, [1]), "F_sp3_2": (16.4, [1]),
    "F_sp3_1": (12.7, [1]), "F_ring": (17.5, [1]),
    "Cl": (10.8, [1, 1.5, 1.5, 1.5, 1.5, 1.5]), "Br": (17.5, [1]),
    "I": (19.4, [1]),
}


@lru_cache(maxsize=16384)
def chickos(smiles: str, *, updated: bool = True) -> Estimate:
    """Structure-derived Perry/1991 or updated-1999 group entropy estimate."""
    mol = molecule(smiles)
    groups, consumed_c = _functional_assignment(mol)
    carbons = UPDATED_CARBON if updated else PERRY_CARBON
    functional = UPDATED_FUNCTIONAL if updated else PERRY_FUNCTIONAL
    # Only benzenoid six-membered rings and their N analogues are aromatic
    # under the paper's convention. Other aromatic heterocycles use ring terms.
    rings = list(mol.GetRingInfo().AtomRings())
    nonaromatic = [r for r in rings if not (len(r) == 6 and all(
        mol.GetAtomWithIdx(i).GetIsAromatic() and mol.GetAtomWithIdx(i).GetSymbol() in {"C", "N"}
        for i in r))]
    ring_atoms = set().union(*map(set, nonaromatic)) if nonaromatic else set()
    ring_bonds = {b.GetIdx() for b in mol.GetBonds() if any(
        b.GetBeginAtomIdx() in r and b.GetEndAtomIdx() in r for r in nonaromatic)}
    # Count independent cycles of the selected nonaromatic subgraph.
    components = 0
    unseen = set(ring_atoms)
    while unseen:
        stack = [unseen.pop()]
        components += 1
        while stack:
            i = stack.pop()
            for b in mol.GetAtomWithIdx(i).GetBonds():
                j = b.GetOtherAtomIdx(i)
                if b.GetIdx() in ring_bonds and j in unseen:
                    unseen.remove(j)
                    stack.append(j)
    nr = len(ring_bonds) - len(ring_atoms) + components
    ring_base, ring_increment = (33.4, 3.7) if updated else (35.19, 4.289)
    total = ring_base * nr + ring_increment * (len(ring_atoms) - 3 * nr)
    t = sum(kind != "F" for kind, _, _ in groups) + int(any(kind == "F" for kind, _, _ in groups))
    entries = []
    for kind, ring, ids in groups:
        atom = mol.GetAtomWithIdx(min(ids))
        key = kind
        if kind == "F":
            neighbor = atom.GetNeighbors()[0]
            if neighbor.IsInRing() and not neighbor.GetIsAromatic():
                key = "F_ring"
            elif neighbor.GetIsAromatic() and updated:
                key = "F_ar"
            elif neighbor.GetHybridization() == Chem.HybridizationType.SP3:
                count = sum(n.GetSymbol() == "F" for n in neighbor.GetNeighbors())
                key = f"F_sp3_{count}" if updated else "F_sp3"
            else:
                key = "F_sp2"
        elif kind == "amine_2H" and not updated and any(
            n.GetIsAromatic() for n in atom.GetNeighbors()):
            key = "amine_2H_ar"
        elif kind == "aromatic_n" and atom.GetIdx() in ring_atoms:
            if atom.GetTotalNumHs():
                key = 'amine_1H_ring'
            elif atom.GetDegree() == 3:
                key = 'amine_0H_ring'
            else:
                key = 'imine_ring'
        elif ring:
            if key + "_ring" not in functional and kind not in {'F', 'Cl', 'Br', 'I', 'aromatic_n'}:
                raise NotApplicable(f"no {'1999' if updated else 'Perry'} cyclic functional group: {key}")
            if key + "_ring" in functional:
                key += "_ring"
        if key not in functional:
            raise NotApplicable(f"no {'1999' if updated else 'Perry'} group assignment: {key}")
        g, coefficients = functional[key]
        if len(coefficients) > 1:
            if t > len(coefficients) and not updated:
                raise NotApplicable("Perry table lacks substitution coefficient beyond its columns")
            c = coefficients[min(max(t, 1), len(coefficients)) - 1]
        else:
            c = 1.0
        entries.append({"group": key, "atoms": sorted(ids), "coefficient": c, "contribution": c * g})
        total += c * g
    methylenes = []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if atom.GetSymbol() != "C" or i in consumed_c:
            continue
        h = atom.GetTotalNumHs()
        bearing = any(n.GetSymbol() != "C" or n.GetIdx() in consumed_c for n in atom.GetNeighbors())
        hybrid = {Chem.HybridizationType.SP3: "sp3", Chem.HybridizationType.SP2: "sp2",
                  Chem.HybridizationType.SP: "sp"}.get(atom.GetHybridization())
        if not hybrid:
            raise NotApplicable("unsupported carbon hybridization")
        if i in ring_atoms:
            if hybrid == "sp3" and h == 2:
                continue
            key = f"ring_{hybrid}_{h}"
        elif atom.GetIsAromatic():
            if h:
                key = "ar_H"
            else:
                neighbors = [n for n in atom.GetNeighbors() if not n.GetIsAromatic()]
                if not neighbors:
                    memberships = sum(i in r for r in rings)
                    key = "ar_internal" if updated and memberships >= 3 else "ar_sp2"
                elif any(n.GetSymbol() != 'C' or n.GetIdx() in consumed_c for n in neighbors):
                    key = 'ar_sp2'  # Functional substituents use the table's non-C environment.
                elif all(n.GetHybridization() == Chem.HybridizationType.SP3 for n in neighbors):
                    key = "ar_sp3"
                elif any(n.GetHybridization() == Chem.HybridizationType.SP for n in neighbors):
                    key = "ar_sp"
                else:
                    key = "ar_sp2"
        else:
            key = f"{hybrid}_{h}"
        if key not in carbons:
            raise NotApplicable(f"unsupported carbon environment: {key}")
        g, c = carbons[key]
        c = c if bearing else 1.0
        entry = {"group": key, "atoms": [i], "coefficient": c, "contribution": c * g}
        entries.append(entry)
        total += c * g
        if key == "sp3_2" and updated:
            methylenes.append(i)
    if updated and methylenes:
        # Source section 5 counts physical ring atoms when comparing chain
        # length with the rest of a cyclic molecule, not the compact ring formula.
        remainder = mol.GetNumHeavyAtoms() - len(methylenes)
        # A functional group contributes one remaining group, irrespective of
        # its atom count or whether it contains carbon. Keep physical ring
        # atoms for cyclic groups, as required by source section 5.
        remainder -= sum(
            len(ids) - max(1, sum(mol.GetAtomWithIdx(i).IsInRing() for i in ids))
            for _, _, ids in groups
        )
        unseen = set(methylenes)
        eligible = set()
        while unseen:
            chain = {unseen.pop()}
            stack = list(chain)
            while stack:
                for n in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                    if n.GetIdx() in unseen:
                        unseen.remove(n.GetIdx())
                        chain.add(n.GetIdx())
                        stack.append(n.GetIdx())
            if len(chain) >= remainder:
                eligible |= chain
        for entry in entries:
            if entry["group"] == "sp3_2" and entry["atoms"][0] in eligible:
                extra = entry["contribution"] * .31
                total += extra
                entry["contribution"] += extra
                entry["coefficient"] *= 1.31
    if total <= 0 or not math.isfinite(total):
        raise NotApplicable("nonpositive predicted entropy")
    return Estimate(total, {"functional_group_count": t, "nonaromatic_ring_count": nr,
                            "nonaromatic_ring_atoms": len(ring_atoms), "groups": entries})


def acid_estimate(smiles: str, *, use_entropy: bool, melting_temperature_K: float) -> Estimate:
    """Compernolle 2011 Eqs. 20/21, using DY effective torsions."""
    mol = molecule(smiles)
    if any(a.GetSymbol() not in {"C", "O", "H"} for a in mol.GetAtoms()):
        raise NotApplicable("acid model outside CHO domain")
    carbon_count = sum(a.GetSymbol() == "C" for a in mol.GetAtoms())
    if not 2 <= carbon_count <= 10:
        raise NotApplicable("acid model outside C2-C10")
    if any(a.GetIsAromatic() for a in mol.GetAtoms()) or any(
        b.GetBondType() != Chem.BondType.SINGLE
        and not (b.GetBondType() == Chem.BondType.DOUBLE and {b.GetBeginAtom().GetSymbol(),
                                                               b.GetEndAtom().GetSymbol()} == {"C", "O"})
        for b in mol.GetBonds()):
        raise NotApplicable("acid model requires saturated carbon skeleton")
    acid_matches = mol.GetSubstructMatches(pattern("C(=O)[O;H1]"))
    if not acid_matches:
        raise NotApplicable("acid model requires a carboxylic acid")
    acid_atoms = set().union(*map(set, acid_matches))
    other_oh = sum(a.GetSymbol() == "O" and a.GetTotalNumHs() == 1 and a.GetIdx() not in acid_atoms
                   for a in mol.GetAtoms())
    other_co = sum(a.GetIdx() not in acid_atoms for a in mol.GetAtoms()
                   if a.GetSymbol() == "C" and any(b.GetBondType() == Chem.BondType.DOUBLE
                                                  and b.GetOtherAtom(a).GetSymbol() == "O" for b in a.GetBonds()))
    allowed_o = acid_atoms | {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O" and (
        a.GetTotalNumHs() == 1 or any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds()))}
    if any(a.GetSymbol() == "O" and a.GetIdx() not in allowed_o for a in mol.GetAtoms()):
        raise NotApplicable("acid model does not include ethers/esters")
    if len(acid_matches) + other_oh + other_co < 2:
        raise NotApplicable("acid model requires at least two functional groups")
    if carbon_count == 2 and len(acid_matches) == 2:
        raise NotApplicable("oxalic acid excluded by source")
    tau = flexibility(smiles, revised=False)["tau"]
    carbon_degrees = [sum(n.GetSymbol() == "C" for n in a.GetNeighbors())
                      for a in mol.GetAtoms() if a.GetSymbol() == "C"]
    # The source's even-chain stability term describes unsubstituted linear
    # alpha,omega-diacids. Extra OH/oxo groups destroy that packing pattern.
    even = int(len(acid_matches) == 2 and not other_oh and not other_co
               and not mol.GetRingInfo().NumRings() and max(carbon_degrees) <= 2
               and carbon_count % 2 == 0)
    features = [1.0, tau, other_oh + other_co, len(acid_matches), even]
    coefficients = [-8.508, 6.520, 23.890, 25.379, 18.153] if use_entropy else [-12.895, 1.807, 11.552, 15.327, 12.358]
    value = sum(a * b for a, b in zip(features, coefficients, strict=True))
    entropy = value if use_entropy else value * 1000.0 / melting_temperature_K
    if entropy <= 0:
        raise NotApplicable("nonpositive acid-model prediction")
    return Estimate(entropy, {"features": features, "coefficient_basis": "entropy" if use_entropy else "enthalpy"})
