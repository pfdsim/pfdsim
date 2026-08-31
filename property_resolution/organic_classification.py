"""Central strict molecular-organic and hydrogen-bond-donor classification."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..compound_identity import parse_formula_counts
else:
    from compound_identity import parse_formula_counts


HALOGEN_ELEMENTS = frozenset({"F", "Cl", "Br", "I"})

# Conventional inorganic carbon compounds which otherwise resemble organics
# under a simple carbon-plus-hydrogen formula rule.
KNOWN_INORGANIC_CARBON_CAS = frozenset({
    "74-90-8",    # hydrogen cyanide
    "630-08-0",   # carbon monoxide
    "124-38-9",   # carbon dioxide
    "75-15-0",    # carbon disulfide
    "463-58-1",   # carbonyl sulfide
    "460-19-5",   # cyanogen
    "504-64-3",   # carbon suboxide
    "463-79-6",   # carbonic acid
    "75-44-5",    # phosgene
    "353-50-4",   # carbonyl fluoride
})


def _count_signature(counts: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    normalized = []
    for element, raw_count in counts.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError, OverflowError):
            return ()
        if count <= 0 or float(raw_count) != count:
            return ()
        normalized.append((str(element), count))
    return tuple(sorted(normalized))


KNOWN_INORGANIC_CARBON_SIGNATURES = frozenset({
    _count_signature(counts)
    for counts in (
        {"C": 1, "H": 1, "N": 1},       # HCN
        {"C": 1, "O": 1},                 # CO
        {"C": 1, "O": 2},                 # CO2
        {"C": 1, "S": 2},                 # CS2
        {"C": 1, "O": 1, "S": 1},       # COS
        {"C": 2, "N": 2},                 # cyanogen
        {"C": 3, "O": 2},                 # carbon suboxide
        {"C": 1, "H": 2, "O": 3},       # carbonic acid
        {"C": 1, "H": 1, "N": 1, "O": 1},  # HNCO/HCNO family
    )
})


@dataclass(frozen=True)
class StrictOrganicClassification:
    is_organic: bool
    reason: str


@dataclass(frozen=True)
class HydrogenBondDonorProfile:
    alcohol_oh: int = 0
    nitrogen_nh: int = 0
    carboxylic_acid_oh: int = 0
    phenol_oh: int = 0
    thiol_sh: int = 0
    other: int = 0

    @property
    def onh_count(self) -> int:
        return (
            self.alcohol_oh
            + self.nitrogen_nh
            + self.carboxylic_acid_oh
            + self.phenol_oh
        )

    @property
    def total_count(self) -> int:
        return self.onh_count + self.thiol_sh + self.other

    @property
    def is_polyol(self) -> bool:
        return self.alcohol_oh >= 2 and self.alcohol_oh == self.total_count

    @property
    def onh_classes(self) -> tuple[str, ...]:
        classes = []
        if self.is_polyol:
            classes.append("polyol")
        elif self.alcohol_oh:
            classes.append("alcohol")
        if self.nitrogen_nh:
            classes.append("nitrogen")
        if self.carboxylic_acid_oh:
            classes.append("acid")
        if self.phenol_oh:
            classes.append("phenol")
        return tuple(classes)

    @property
    def all_classes(self) -> tuple[str, ...]:
        classes = list(self.onh_classes)
        if self.thiol_sh:
            classes.append("thiol")
        if self.other:
            classes.append("other")
        return tuple(classes)

    def gc_fractions(self) -> Optional[dict[str, float]]:
        if self.onh_count <= 0 or self.thiol_sh or self.other:
            return None
        if self.is_polyol:
            return {"polyol": 1.0, "nitrogen": 0.0, "acid": 0.0, "phenol": 0.0}
        denominator = float(self.onh_count)
        return {
            "polyol": 0.0,
            "nitrogen": self.nitrogen_nh / denominator,
            "acid": self.carboxylic_acid_oh / denominator,
            "phenol": self.phenol_oh / denominator,
        }


def is_strict_organic_formula_counts(counts: Mapping[str, Any]) -> bool:
    signature = _count_signature(counts)
    if not signature or signature in KNOWN_INORGANIC_CARBON_SIGNATURES:
        return False
    normalized = dict(signature)
    if normalized.get("C", 0) <= 0:
        return False
    # One-carbon carbonate/bicarbonate species are conventional inorganic
    # compounds; ordinary organic carbonates contain additional carbons.
    if normalized.get("C") == 1 and normalized.get("O", 0) >= 3:
        return False
    halogen_count = sum(normalized.get(element, 0) for element in HALOGEN_ELEMENTS)
    # Carbonyl dihalides are retained in the explicit inorganic family.
    if (
        normalized.get("C") == 1
        and normalized.get("O") == 1
        and halogen_count == 2
        and sum(normalized.values()) == 4
    ):
        return False
    return normalized.get("H", 0) > 0 or halogen_count > 0


def classify_strict_molecular_organic(
    *,
    cas: Any = None,
    formula: Any = None,
    smiles: Any = None,
) -> StrictOrganicClassification:
    cas_text = str(cas or "").strip()
    if cas_text in KNOWN_INORGANIC_CARBON_CAS:
        return StrictOrganicClassification(False, "known inorganic carbon compound")

    counts = parse_formula_counts(str(formula)) if formula else None
    if counts and not is_strict_organic_formula_counts(counts):
        return StrictOrganicClassification(False, "formula is outside strict molecular-organic domain")

    smiles_text = str(smiles or "").strip()
    if smiles_text:
        try:
            from rdkit import Chem

            molecule = Chem.MolFromSmiles(smiles_text)
        except Exception:
            molecule = None
        if molecule is None:
            if counts:
                return StrictOrganicClassification(
                    True,
                    "strict molecular-organic formula; SMILES unavailable",
                )
            return StrictOrganicClassification(False, "SMILES could not be parsed")
        if len(Chem.GetMolFrags(molecule)) != 1:
            return StrictOrganicClassification(False, "disconnected or salt-like structure")
        if sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()) != 0:
            return StrictOrganicClassification(False, "net-charged structure")
        if not any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()):
            return StrictOrganicClassification(False, "structure contains no carbon")

    if counts:
        return StrictOrganicClassification(True, "strict molecular-organic formula")
    if smiles_text:
        return StrictOrganicClassification(True, "strict neutral molecular-organic structure")
    return StrictOrganicClassification(False, "molecular identity is insufficient")


def _donor_atom_class(atom) -> str:
    atomic_number = atom.GetAtomicNum()
    if atomic_number == 8:
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIsAromatic():
                return "phenol_oh"
            if neighbor.GetAtomicNum() == 6 and any(
                bond.GetBondTypeAsDouble() >= 1.9
                and bond.GetOtherAtom(neighbor).GetAtomicNum() in (8, 16)
                for bond in neighbor.GetBonds()
                if bond.GetOtherAtom(neighbor).GetIdx() != atom.GetIdx()
            ):
                return "carboxylic_acid_oh"
        return "alcohol_oh"
    if atomic_number == 7:
        return "nitrogen_nh"
    if atomic_number == 16:
        return "thiol_sh"
    return "other"


@lru_cache(maxsize=1)
def _hydrogen_bond_feature_factory():
    import os
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    return ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    )


@lru_cache(maxsize=4096)
def hydrogen_bond_donor_profile(smiles: str) -> Optional[HydrogenBondDonorProfile]:
    text = str(smiles or "").strip()
    if not text:
        return None
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            return None
        factory = _hydrogen_bond_feature_factory()
        atom_ids = set()
        for feature in factory.GetFeaturesForMol(molecule):
            if feature.GetFamily() == "Donor":
                atom_ids.update(feature.GetAtomIds())
        counts = {
            "alcohol_oh": 0,
            "nitrogen_nh": 0,
            "carboxylic_acid_oh": 0,
            "phenol_oh": 0,
            "thiol_sh": 0,
            "other": 0,
        }
        for atom_id in atom_ids:
            counts[_donor_atom_class(molecule.GetAtomWithIdx(atom_id))] += 1
        return HydrogenBondDonorProfile(**counts)
    except Exception:
        return None
