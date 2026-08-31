"""Reichenberg low-pressure vapor viscosity and strict RDKit fragmentation.

Equations and group values follow Perry 9th, Eqs. 2-84 through 2-87 and
Table 2-173. Unsupported or ambiguous structures are rejected rather than
silently approximated.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem


GROUP_VALUES = {
    "ch3": 9.04,
    "ch2": 6.47,
    "ch": 2.67,
    "c": -1.53,
    "alkene_ch2": 7.68,
    "alkene_ch": 5.53,
    "alkene_c": 1.78,
    "alkyne_ch": 7.41,
    "alkyne_c": 5.24,
    "ring_ch2": 6.91,
    "ring_ch": 1.16,
    "ring_c": 0.23,
    "ring_alkene_ch": 5.90,
    "ring_alkene_c": 3.59,
    "f": 4.46,
    "cl": 10.06,
    "br": 12.83,
    "oh_alcohol": 7.96,
    "ether_o": 3.59,
    "carbonyl": 12.02,
    "aldehyde": 14.02,
    "carboxylic_acid": 18.65,
    "ester": 13.41,
    "amine_nh2": 9.71,
    "amine_nh": 3.68,
    "ring_imine_n": 4.97,
    "nitrile": 18.13,
    "ring_s": 8.86,
}

_ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "F", "Cl", "Br"}


class ReichenbergFragmentationError(ValueError):
    """The molecule cannot be represented by Perry Table 2-173."""


@dataclass(frozen=True)
class ReichenbergFragmentation:
    smiles: str
    groups: Counter

    @property
    def contribution_sum(self) -> float:
        return sum(GROUP_VALUES[name] * count for name, count in self.groups.items())


@dataclass(frozen=True)
class ReichenbergStructureProfile:
    heavy_atoms: int
    carbon_atoms: int
    heteroatoms: int
    has_carbon_hydrogen_bond: bool


def _reject(reason: str) -> None:
    raise ReichenbergFragmentationError(reason)


def structure_profile(smiles: str) -> ReichenbergStructureProfile:
    """Return the structural counts used to select a vapor-viscosity branch."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        _reject("unparseable SMILES")
    if len(Chem.GetMolFrags(mol)) != 1:
        _reject("salts and disconnected structures unsupported")
    carbon_atoms = sum(atom.GetSymbol() == "C" for atom in mol.GetAtoms())
    heavy_atoms = mol.GetNumHeavyAtoms()
    return ReichenbergStructureProfile(
        heavy_atoms=heavy_atoms,
        carbon_atoms=carbon_atoms,
        heteroatoms=heavy_atoms - carbon_atoms,
        has_carbon_hydrogen_bond=any(
            atom.GetSymbol() == "C" and atom.GetTotalNumHs() > 0
            for atom in mol.GetAtoms()
        ),
    )


def fragment(smiles: str) -> ReichenbergFragmentation:
    """Fragment one neutral organic molecule with full heavy-atom accounting."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        _reject("unparseable SMILES")
    if len(Chem.GetMolFrags(mol)) != 1:
        _reject("salts and disconnected structures unsupported")

    assigned = [False] * mol.GetNumAtoms()
    groups: Counter = Counter()

    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in _ALLOWED_ELEMENTS:
            _reject(f"element {atom.GetSymbol()} unsupported")
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            _reject("charged or radical species unsupported")
    if not any(atom.GetSymbol() == "C" for atom in mol.GetAtoms()):
        _reject("no carbon (use inorganic Reichenberg prefactor)")

    def take(*indices: int) -> None:
        for index in indices:
            if assigned[index]:
                _reject(f"atom {index} claimed twice (fragmenter bug)")
            assigned[index] = True

    def matches(smarts: str):
        pattern = Chem.MolFromSmarts(smarts)
        return [
            match
            for match in mol.GetSubstructMatches(pattern)
            if not any(assigned[index] for index in match)
        ]

    # Whole carbonyl functional groups, from most specific to most general.
    for match in matches("[CX3](=[OX1])[OX2H1]"):
        if any(assigned[index] for index in match):
            continue
        take(*match)
        groups["carboxylic_acid"] += 1
    for match in matches("[CX3](=[OX1])[OX2][#6]"):
        if any(assigned[index] for index in match[:3]):
            continue
        take(*match[:3])
        groups["ester"] += 1
    for match in matches("[CX3H1]=[OX1]"):
        if any(assigned[index] for index in match):
            continue
        take(*match)
        groups["aldehyde"] += 1
    for match in matches("[CX3]=[OX1]"):
        if any(assigned[index] for index in match):
            continue
        take(*match)
        groups["carbonyl"] += 1

    # Nitrile consumes both C and N; HCN has no attached organic skeleton.
    for match in matches("[CX2]#[NX1]"):
        if any(assigned[index] for index in match):
            continue
        carbon = mol.GetAtomWithIdx(match[0])
        if not any(
            neighbor.GetSymbol() == "C" and neighbor.GetIdx() != match[1]
            for neighbor in carbon.GetNeighbors()
        ):
            _reject("nitrile without attached carbon skeleton")
        take(*match)
        groups["nitrile"] += 1

    # Oxygen not already consumed by a carbonyl group.
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        if assigned[index] or atom.GetSymbol() != "O":
            continue
        hydrogens = atom.GetTotalNumHs()
        neighbors = atom.GetNeighbors()
        if hydrogens == 1 and len(neighbors) == 1:
            take(index)
            groups["oh_alcohol"] += 1
        elif hydrogens == 0 and len(neighbors) == 2:
            if any(neighbor.GetSymbol() == "O" for neighbor in neighbors):
                _reject("peroxide unsupported")
            take(index)
            groups["ether_o"] += 1
        else:
            _reject("unsupported oxygen environment")

    # Nitrogen not already consumed by nitrile.
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        if assigned[index] or atom.GetSymbol() != "N":
            continue
        hydrogens = atom.GetTotalNumHs()
        if atom.IsInRing():
            if atom.GetIsAromatic() and hydrogens == 0:
                take(index)
                groups["ring_imine_n"] += 1
            else:
                _reject("unsupported ring nitrogen environment")
        elif not atom.GetIsAromatic() and hydrogens == 2:
            take(index)
            groups["amine_nh2"] += 1
        elif not atom.GetIsAromatic() and hydrogens == 1:
            take(index)
            groups["amine_nh"] += 1
        else:
            _reject("unsupported nitrogen environment")

    # Perry provides sulfur only for a divalent ring sulfur.
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        if assigned[index] or atom.GetSymbol() != "S":
            continue
        if atom.IsInRing() and atom.GetDegree() == 2 and atom.GetTotalNumHs() == 0:
            take(index)
            groups["ring_s"] += 1
        else:
            _reject("non-ring or non-divalent sulfur unsupported")

    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        if assigned[index] or atom.GetSymbol() not in {"F", "Cl", "Br"}:
            continue
        take(index)
        groups[atom.GetSymbol().lower()] += 1

    # Remaining atoms must be carbon-skeleton groups.
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        if assigned[index] or atom.GetSymbol() != "C":
            continue
        hydrogens = atom.GetTotalNumHs()
        hybridization = atom.GetHybridization()
        if atom.GetIsAromatic():
            name = {1: "ring_alkene_ch", 0: "ring_alkene_c"}.get(hydrogens)
        elif atom.IsInRing() and hybridization == Chem.HybridizationType.SP3:
            name = {2: "ring_ch2", 1: "ring_ch", 0: "ring_c"}.get(hydrogens)
        elif atom.IsInRing() and hybridization == Chem.HybridizationType.SP2:
            name = {1: "ring_alkene_ch", 0: "ring_alkene_c"}.get(hydrogens)
        elif not atom.IsInRing() and hybridization == Chem.HybridizationType.SP3:
            name = {3: "ch3", 2: "ch2", 1: "ch", 0: "c"}.get(hydrogens)
        elif not atom.IsInRing() and hybridization == Chem.HybridizationType.SP2:
            name = {2: "alkene_ch2", 1: "alkene_ch", 0: "alkene_c"}.get(hydrogens)
        elif not atom.IsInRing() and hybridization == Chem.HybridizationType.SP:
            name = {1: "alkyne_ch", 0: "alkyne_c"}.get(hydrogens)
        else:
            name = None
        if name is None:
            _reject(
                f"unsupported carbon environment at atom {index} "
                f"(H={hydrogens}, hybridization={hybridization}, ring={atom.IsInRing()})"
            )
        take(index)
        groups[name] += 1

    leftovers = [
        f"{atom.GetSymbol()}{atom.GetIdx()}"
        for atom in mol.GetAtoms()
        if not assigned[atom.GetIdx()]
    ]
    if leftovers:
        _reject(f"unassigned atoms: {', '.join(leftovers)}")
    if not groups:
        _reject("no Reichenberg groups found")
    carbon_skeleton_groups = {
        "ch3", "ch2", "ch", "c", "alkene_ch2", "alkene_ch", "alkene_c",
        "alkyne_ch", "alkyne_c", "ring_ch2", "ring_ch", "ring_c",
        "ring_alkene_ch", "ring_alkene_c",
    }
    if groups.get("nitrile") and not set(groups).intersection(carbon_skeleton_groups):
        _reject("nitrile groups without an unconsumed carbon skeleton")
    contribution = sum(GROUP_VALUES[name] * count for name, count in groups.items())
    if contribution <= 0.0 or not math.isfinite(contribution):
        _reject(f"nonpositive group contribution sum {contribution:g}")
    return ReichenbergFragmentation(smiles=smiles, groups=groups)


def reduced_dipole(dipole_D: float, Pc_bar: float, Tc_K: float) -> float:
    return 52.46 * float(dipole_D) ** 2 * float(Pc_bar) / float(Tc_K) ** 2


def organic_A(molecular_weight: float, Tc_K: float, contribution_sum: float) -> float:
    if molecular_weight <= 0.0 or Tc_K <= 0.0 or contribution_sum <= 0.0:
        raise ValueError("M, Tc, and the Reichenberg contribution sum must be positive")
    return 1.0e-7 * math.sqrt(molecular_weight) * Tc_K / contribution_sum


def inorganic_A(molecular_weight: float, Tc_K: float, Pc_bar: float) -> float:
    if molecular_weight <= 0.0 or Tc_K <= 0.0 or Pc_bar <= 0.0:
        raise ValueError("M, Tc, and Pc must be positive")
    return (
        1.6104e-10
        * math.sqrt(molecular_weight)
        * (Pc_bar * 1.0e5) ** (2.0 / 3.0)
        * Tc_K ** (-1.0 / 6.0)
    )


def viscosity_Pa_s(
    temperature_K: float,
    molecular_weight: float,
    Tc_K: float,
    Pc_bar: float,
    *,
    dipole_D: float = 0.0,
    fragmentation: Optional[ReichenbergFragmentation] = None,
    inorganic: bool = False,
) -> float:
    """Return Reichenberg dilute-gas viscosity in Pa*s."""
    if temperature_K <= 0.0:
        raise ValueError("Temperature must be positive")
    if inorganic:
        A = inorganic_A(molecular_weight, Tc_K, Pc_bar)
    elif fragmentation is not None:
        A = organic_A(molecular_weight, Tc_K, fragmentation.contribution_sum)
    else:
        raise ValueError("Organic Reichenberg viscosity requires fragmentation")
    Tr = temperature_K / Tc_K
    base = 1.0 + 0.36 * Tr * (Tr - 1.0)
    if base <= 0.0:
        raise ValueError("Reichenberg temperature factor is nonpositive")
    mu_star = reduced_dipole(dipole_D, Pc_bar, Tc_K)
    polar = 270.0 * mu_star**4
    value = (
        A
        * Tr**2
        / base ** (1.0 / 6.0)
        * (1.0 + polar)
        / (Tr + polar)
    )
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError("Reichenberg viscosity is nonphysical")
    return value
