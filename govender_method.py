"""Govender, Rarey and Ramjugernath (2020) saturated-liquid conductivity.

Implements Table 1 and equations (11)-(12) of J. Chem. Eng. Data 65,
1300-1312, DOI 10.1021/acs.jced.9b00741.  ``tb`` may be supplied in K;
otherwise the published structure-only Nannoolal boiling-point estimate is
used.  The method predicts the saturation-line value, not pressure effects or
the critical enhancement.  Ethylene and propylene glycol were omitted from
the paper's fit because of anomalous temperature trends; higher diols were
included, but their low-temperature predictions warrant caution.

Locally refitted contributions and second-order corrections are enabled by
default.  Their data, fitting probes, cross-validation reports, and final Perry
benchmark are maintained under ``scripts/thermal_conductivity/liquid/``. Pass
``local_refits=False`` to reproduce the published coefficients and equations.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from rdkit import Chem

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from . import nannoolal_method
else:
    import nannoolal_method


class GovenderError(ValueError):
    """Molecule or state cannot be evaluated with the published method."""


# Table 1: ink number -> (Delta A, Delta B).  The paper's appendix evaluates
# lambda(T) = exp(ln(n) * sum(B) / n) + sum(A)/Tb * (1 - T/Tb).
CONTRIBUTIONS: dict[int, tuple[float, float]] = {
    1: (16.16, -2.589), 2: (14.42, -2.754), 3: (17.40, -2.637),
    4: (2.33, -0.636), 5: (-16.18, 1.095), 6: (-34.82, 2.756),
    7: (-3.94, -0.919), 8: (-20.36, 0.818), 9: (-37.15, 1.966),
    10: (4.67, -1.218), 11: (-9.75, 0.547), 12: (-39.11, 2.764),
    14: (-14.43, -0.288), 15: (-34.63, 1.170),
    16: (6.17, -1.220), 17: (-11.00, 0.726), 18: (-12.04, 0.322),
    19: (-6.19, -0.086), 20: (4.83, -1.096), 21: (10.37, -1.217),
    24: (7.65, -1.860), 26: (13.33, -2.164), 27: (-10.10, 1.201),
    29: (15.01, -2.813), 32: (17.880, -2.5138),
    33: (-8.434, 0.2742),
    35: (28.679, -1.4986), 37: (31.255, -2.2799),
    38: (19.057, -2.1214), 39: (14.832, -2.3602),
    40: (16.852, -2.2692), 41: (20.216, -2.5325),
    42: (23.059, -1.9803), 43: (21.743, -2.4678),
    44: (16.272, -2.4102), 45: (6.135, -2.7263),
    46: (12.348, -2.9920), 47: (1.147, -3.3472),
    48: (13.727, -3.1523), 49: (2.150, -0.1041),
    50: (-4.051, -1.5286), 51: (7.507, 0.3052),
    52: (12.776, -1.4955), 53: (4.982, -1.6203),
    54: (6.527, -1.0405), 55: (23.158, -2.8564),
    58: (0.546, -0.2229), 59: (12.744, -2.4582),
    61: (25.492, -1.1897), 66: (18.249, -2.0575),
    70: (26.620, -3.0101), 72: (11.442, 0.8726),
    75: (28.361, -3.1016), 76: (29.557, -3.4053),
    80: (5.175, -0.9976), 81: (75.000, -2.2536),
    82: (9.835, 0.3294), 84: (-12.576, 1.1656),
    88: (22.279, -0.8005), 89: (25.543, -2.6877),
    93: (9.092, 0.5287), 95: (3.692, -0.5891),
    100: (11.579, 0.0759), 110: (-26.549, 3.2233),
    145: (13.232, -2.7571), 149: (14.202, -2.6812),
    153: (3.074, -0.1443), 175: (7.450, -1.0202),
}

# Locally refitted Delta A / Delta B pairs. These are conditional replacements
# for the published first-order values above, not competing calculation paths.
# See scripts/thermal_conductivity/liquid/ for sources and validation artifacts.
LOCAL_GROUP_REFITS: dict[int, tuple[float, float]] = {
    53: (18.37799753, -3.21353177),    # aliphatic COOH; 18 Perry compounds
    58: (4.38710797, -0.65722719),     # ordinary ketones; excludes quinones
    84: (-9.96173266, 2.21614816),     # joint tertiary-amine fit
    100: (-0.38907215, -0.23836755),  # acyclic thioethers
}

# Added once per molecule containing an amine and an aliphatic alcohol group.
LOCAL_AMINE_ALCOHOL_CORRECTION = (29.01858325, -0.33071887)

# Added once per nonaromatic SSSR ring. The four-member pair is the midpoint
# of independently fitted three- and five-member corrections and reproduced
# held-out cyclobutane without including it in either endpoint fit.
LOCAL_RING_CORRECTIONS: dict[int, tuple[float, float]] = {
    3: (8.24341049, -1.58125233),
    4: (7.75770332, -1.09958823),
    5: (7.27199616, -0.61792414),
}

PAPER_CAUTION_GROUPS = frozenset({32, 33, 47, 50, 61, 88, 95, 100, 110})
# The paper explicitly discloses only one supporting component for these
# exact classes. Other caution groups had sparse but non-singleton support.
PAPER_SINGLE_COMPONENT_GROUPS = frozenset({61, 88, 95, 110})
_HALOGENS = {9, 17, 35, 53}
_ELECTRONEGATIVE = {7, 8, 9, 17, 35, 53}
_AMINE_GROUPS = frozenset({80, 81, 82, 84, 93})
_ALCOHOL_GROUPS = frozenset({49, 153})
_CYCLIC_KETONE = Chem.MolFromSmarts("[CX3;R](=[OX1])([#6])[#6]")
assert _CYCLIC_KETONE is not None


@dataclass(frozen=True)
class GovenderResult:
    smiles: str
    tb_K: float
    tb_source: str
    n_heavy_atoms: int
    groups: dict[int, int]
    sum_A: float
    sum_B: float
    lambda_ref_W_m_K: float
    local_refits: bool
    warnings: tuple[str, ...]

    def conductivity_W_m_K(self, temperature_K: float) -> float:
        """Saturated-liquid conductivity at ``temperature_K`` [W/(m K)]."""
        if not math.isfinite(temperature_K) or temperature_K <= 0:
            raise GovenderError("temperature_K must be positive and finite")
        value = (self.lambda_ref_W_m_K
                 + self.sum_A / self.tb_K * (1 - temperature_K / self.tb_K))
        if value <= 0:
            raise GovenderError("non-positive prediction; outside the method range")
        return value


def _carbon_group(atom: Chem.Atom) -> int:
    """Classify a carbon left after the larger functional groups are claimed."""
    neighbors = atom.GetNeighbors()
    hydrogen = atom.GetTotalNumHs()
    has_e = any(nb.GetAtomicNum() in _ELECTRONEGATIVE for nb in neighbors)
    if atom.GetIsAromatic():
        aromatic_bonds = sum(b.GetIsAromatic() for b in atom.GetBonds())
        if aromatic_bonds == 3:
            return 19
        if hydrogen:
            return 16
        return 18 if has_e else 17
    if any(b.GetBondType() == Chem.BondType.DOUBLE
           and b.GetOtherAtom(atom).GetAtomicNum() == 6 for b in atom.GetBonds()):
        if atom.IsInRing():
            return 21
        if hydrogen == 2:
            return 26
        # Table 1 counts carbon substituents *besides* the C=C partner:
        # 2-heptene uses 20; the doubly substituted carbon of
        # 2-methyl-2-pentene uses 27.
        carbon_substituents = sum(
            nb.GetAtomicNum() == 6
            and atom.GetOwningMol().GetBondBetweenAtoms(
                atom.GetIdx(), nb.GetIdx()).GetBondType() != Chem.BondType.DOUBLE
            for nb in neighbors)
        return 27 if carbon_substituents >= 2 else 20
    if atom.GetHybridization() != Chem.HybridizationType.SP3:
        raise GovenderError(f"no published carbon group for atom {atom.GetIdx()}")
    if atom.IsInRing():
        by_h = {2: 24 if has_e else 10,
                1: 14 if has_e else 11,
                0: 15 if has_e else 12}
        if hydrogen in by_h:
            return by_h[hydrogen]
    else:
        if hydrogen == 3:
            host = neighbors[0]
            if host.GetIsAromatic():
                return 3
            if host.IsInRing() and host.GetAtomicNum() == 6:
                return 29
            return 2 if has_e else 1
        by_h = {2: 7 if has_e else 4,
                1: 8 if has_e else 5,
                0: 9 if has_e else 6}
        if hydrogen in by_h:
            return by_h[hydrogen]
    raise GovenderError(f"no published carbon group for atom {atom.GetIdx()}")


def _halogen_group(atom: Chem.Atom) -> int:
    host = atom.GetNeighbors()[0]
    if host.GetAtomicNum() != 6:
        raise GovenderError("halogen is not attached to carbon")
    z = atom.GetAtomicNum()
    other_halogens = sum(nb.GetAtomicNum() in _HALOGENS
                         for nb in host.GetNeighbors()) - 1
    if z == 9:
        if host.GetIsAromatic():
            return 37
        return {0: 35, 1: 38}.get(other_halogens, 39)
    if z == 17:
        if host.GetIsAromatic():
            return 41
        vinyl = any(b.GetBondType() == Chem.BondType.DOUBLE
                    and b.GetOtherAtom(host).GetAtomicNum() == 6
                    for b in host.GetBonds())
        if vinyl:
            return 149 if other_halogens else 42
        return {0: 40, 1: 43}.get(other_halogens, 44)
    if z == 35:
        if host.GetIsAromatic():
            return 46
        return 145 if other_halogens >= 2 else 45
    return 47


def _functional_group(gid: int, mol: Chem.Mol, atoms: tuple[int, ...],
                      carbon_count: int) -> tuple[int, int] | None:
    """Map a Nannoolal functional-group instance to (Govender id, frequency)."""
    if gid in {33, 34, 35, 36}:
        return (49 if carbon_count <= 7 else 153, carbon_count)
    if gid == 37:
        return (50, 1)
    if gid == 44:
        carbonyl = mol.GetAtomWithIdx(atoms[0])
        aromatic = any(nb.GetIsAromatic() for nb in carbonyl.GetNeighbors())
        return (48 if aromatic else 53, 1)
    if gid in {73, 79, 83, 97}:
        core = mol.GetAtomWithIdx(atoms[0])
        if gid == 73 and any(
            nb.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(core.GetIdx(), nb.GetIdx()).GetBondType()
            == Chem.BondType.SINGLE
            and (nb.GetDegree() != 2 or not any(
                other.GetAtomicNum() == 6 for other in nb.GetNeighbors()))
            for nb in core.GetNeighbors()
        ):
            raise GovenderError("phosphate group requires a triester")
        if gid == 79 and any(
            mol.GetAtomWithIdx(i).GetAtomicNum() == 8
            and mol.GetAtomWithIdx(i).GetTotalNumHs() > 0 for i in atoms
        ):
            raise GovenderError("carbonate group requires a diester")
        if gid == 83 and sum(nb.GetAtomicNum() == 6
                              for nb in core.GetNeighbors()) != 4:
            raise GovenderError("stannane group requires four carbon neighbors")
        if gid == 97 and core.GetIsAromatic():
            raise GovenderError("aromatic NH has no published Govender group")
    direct = {
        38: 51, 45: 54, 46: 55, 47: 54, 48: 72, 50: 70,
        51: 58, 52: 59, 57: 89, 65: 52, 67: 88,
        68: 75, 69: 76, 73: 95, 79: 61, 83: 110,
        90: 59, 92: 58, 97: 93, 103: 66,
        40: 80, 41: 81, 42: 82, 43: 84, 54: 100,
    }
    return (direct[gid], 1) if gid in direct else None


def _fragment(mol: Chem.Mol) -> tuple[Chem.Mol, Counter[int]]:
    """Return the normalized molecule and its Govender groups together."""
    try:
        base = nannoolal_method.fragment(mol)
    except nannoolal_method.FragmentationError as exc:
        raise GovenderError(str(exc)) from exc
    # Functional instances refer to this representation: Nannoolal demotes
    # carbonyl-bearing aromatic rings before assigning their groups.
    mol = base.mol
    if len(Chem.GetMolFrags(mol)) != 1 or Chem.GetFormalCharge(mol) != 0:
        raise GovenderError("method requires one neutral organic molecule")
    if base.do_not_estimate or 219 in base.groups:
        raise GovenderError("molecule has no published Govender fragmentation")

    groups: Counter[int] = Counter()
    covered: set[int] = set()
    carbon_count = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
    oh_count = 0
    for gid, atoms in base.instances:
        if gid == 39:  # Epoxide: Nannoolal claims the ring; Govender prices O.
            oxygen = next(i for i in atoms
                          if mol.GetAtomWithIdx(i).GetAtomicNum() == 8)
            groups[51] += 1
            covered.add(oxygen)
            continue
        mapped = _functional_group(gid, mol, atoms, carbon_count)
        if mapped is None:
            continue
        target, frequency = mapped
        groups[target] += frequency
        covered.update(atoms)
        if gid in {33, 34, 35, 36}:
            oh_count += 1
            if carbon_count <= 7:
                groups[175] += 1

    # Nannoolal has additional conjugation groups claiming all four carbons;
    # Govender's 32/33 replace only the middle two.  All remaining carbons
    # receive their ordinary single-atom group below.
    for gid, atoms in base.instances:
        if gid not in {88, 89}:
            continue
        if len(atoms) != 4:
            raise GovenderError("unexpected conjugated-diene fragmentation")
        middle = atoms[1:3]
        if any(i in covered for i in middle):
            raise GovenderError("overlapping conjugated-diene groups")
        groups[32 if gid == 88 else 33] += 1
        covered.update(middle)

    for atom in mol.GetAtoms():
        if atom.GetIdx() in covered:
            continue
        z = atom.GetAtomicNum()
        if z == 6:
            groups[_carbon_group(atom)] += 1
        elif z in _HALOGENS:
            groups[_halogen_group(atom)] += 1
        else:
            raise GovenderError(
                f"no published Govender group for {atom.GetSymbol()} atom "
                f"{atom.GetIdx()}")
    if oh_count > 2 or (oh_count == 2 and carbon_count <= 3):
        raise GovenderError("small glycols and polyols are outside the fitted model")
    return mol, groups


def nonaromatic_ring_counts(mol: Chem.Mol) -> Counter[int]:
    """Return SSSR nonaromatic-ring frequencies keyed by ring size."""
    return Counter(
        len(ring)
        for ring in mol.GetRingInfo().AtomRings()
        if not all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
    )


def is_cyclic_thioether(mol: Chem.Mol, groups: dict[int, int]) -> bool:
    """Whether a group-100 sulfur atom belongs to a ring."""
    return 100 in groups and any(
        atom.GetAtomicNum() == 16 and atom.IsInRing()
        for atom in mol.GetAtoms()
    )


def is_unsaturated_cyclic_ketone(
    mol: Chem.Mol, groups: dict[int, int]
) -> bool:
    """Whether a cyclic ketone carbon belongs to a carbon-unsaturated ring."""
    if 58 not in groups:
        return False
    ketone_carbons = {
        match[0] for match in mol.GetSubstructMatches(_CYCLIC_KETONE)
    }
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms = set(ring)
        if not ring_atoms & ketone_carbons:
            continue
        for first, second in zip(ring, (*ring[1:], ring[0])):
            bond = mol.GetBondBetweenAtoms(first, second)
            if (
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetBeginAtom().GetAtomicNum() == 6
                and bond.GetEndAtom().GetAtomicNum() == 6
            ):
                return True
    return False


def _contribution_sums(
    mol: Chem.Mol,
    groups: dict[int, int],
    local_refits: bool,
    warnings: list[str],
) -> tuple[float, float]:
    sum_a = sum(
        CONTRIBUTIONS[gid][0] * frequency for gid, frequency in groups.items()
    )
    sum_b = sum(
        CONTRIBUTIONS[gid][1] * frequency for gid, frequency in groups.items()
    )
    if not local_refits:
        return sum_a, sum_b

    for gid, refitted in LOCAL_GROUP_REFITS.items():
        frequency = groups.get(gid, 0)
        if not frequency:
            continue
        published = CONTRIBUTIONS[gid]
        sum_a += frequency * (refitted[0] - published[0])
        sum_b += frequency * (refitted[1] - published[1])
        warnings.append(
            f"group {gid} uses a local conductivity refit; see "
            "scripts/thermal_conductivity/liquid/"
        )

    present = set(groups)
    if present & _AMINE_GROUPS and present & _ALCOHOL_GROUPS:
        sum_a += LOCAL_AMINE_ALCOHOL_CORRECTION[0]
        sum_b += LOCAL_AMINE_ALCOHOL_CORRECTION[1]
        warnings.append(
            "amine-alcohol uses a local second-order correction; see "
            "scripts/thermal_conductivity/liquid/"
        )

    for size, frequency in sorted(nonaromatic_ring_counts(mol).items()):
        correction = LOCAL_RING_CORRECTIONS.get(size)
        if correction is None:
            continue
        sum_a += frequency * correction[0]
        sum_b += frequency * correction[1]
        warnings.append(
            f"{size}-member nonaromatic ring uses a local second-order "
            "correction; see scripts/thermal_conductivity/liquid/"
        )
    return sum_a, sum_b


def estimate(
    smiles: str,
    tb: float | None = None,
    local_refits: bool = True,
) -> GovenderResult:
    """Build a reusable saturated-liquid conductivity estimate from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise GovenderError(f"could not parse SMILES {smiles!r}")
    return estimate_from_mol(
        mol, tb=tb, smiles=smiles, local_refits=local_refits
    )


def estimate_from_mol(
    mol: Chem.Mol,
    tb: float | None = None,
    smiles: str | None = None,
    local_refits: bool = True,
) -> GovenderResult:
    """Build an estimate from an RDKit molecule; ``tb`` is in kelvin."""
    supplied_tb = tb is not None
    if tb is not None and (not math.isfinite(tb) or tb <= 0):
        raise GovenderError("tb must be positive and finite")
    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
    mol, groups = _fragment(mol)
    if is_cyclic_thioether(mol, groups):
        raise GovenderError(
            "cyclic thioether: do not estimate with the Govender method"
        )
    if is_unsaturated_cyclic_ketone(mol, groups):
        raise GovenderError(
            "unsaturated cyclic ketone: do not estimate with the Govender method"
        )
    if tb is None:
        # The Nannoolal Tb estimate is the existing authoritative
        # structure-only route.  Do not use its local refits here.
        tb = nannoolal_method.estimate_from_mol(mol, local_refits=False).tb_K
        if tb is None:
            raise GovenderError("Nannoolal could not estimate the boiling point; supply tb")
    n = mol.GetNumHeavyAtoms()
    caution = sorted(groups.keys() & PAPER_CAUTION_GROUPS)
    warnings = []
    if caution:
        warnings.append(f"paper marks groups {caution} as based on limited data")
    single_component = sorted(groups.keys() & PAPER_SINGLE_COMPONENT_GROUPS)
    if single_component:
        warnings.append(
            f"paper reports only one supporting component for groups "
            f"{single_component}"
        )
    if sum(a.GetAtomicNum() == 8 and a.GetDegree() == 1
           and a.GetTotalNumHs() == 1 for a in mol.GetAtoms()) == 2:
        warnings.append("higher diol: low-temperature hydrogen-bond anomaly is not modeled")
    sum_a, sum_b = _contribution_sums(
        mol, groups, local_refits, warnings
    )
    reference = math.exp(math.log(n) * sum_b / n)
    return GovenderResult(
        smiles=smiles,
        tb_K=tb, tb_source="provided" if supplied_tb else "estimated",
        n_heavy_atoms=n, groups=dict(sorted(groups.items())),
        sum_A=sum_a, sum_B=sum_b, lambda_ref_W_m_K=reference,
        local_refits=local_refits,
        warnings=tuple(warnings),
    )
