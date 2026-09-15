"""PFDSim-modified Stiel-Thodos dilute-gas thermal conductivity.

This standalone production model implements the final relation developed in:

* ``scripts/thermal_conductivity/benchmark_gas_viscosity_cv_relation.py``
* ``scripts/thermal_conductivity/experiment_functional_group_corrections.py``

The complete investigation, fitted coefficients, validation results, and
reproduction command are documented in
``scripts/thermal_conductivity/README.md``. The property resolver uses it only
as a dilute-vapor fallback after supplied and Perry correlations. Production
coefficient literals are rounded to five significant figures; the canonical
artifact retains full fit precision.

Inputs are resolved elsewhere: temperature [K], dilute-gas viscosity [Pa*s],
ideal-gas Cv [J/(mol*K)], molecular weight [g/mol], critical temperature [K],
an explicit molecular geometry class, molecular formula, optional CAS number,
and a neutral connected SMILES for molecules requiring structural group
classification. The result is thermal conductivity in W/(m*K). Fitted group
corrections are restricted to pfdsim's strict molecular-organic domain;
inorganics receive the unmodified base result.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from .compound_identity import parse_formula_counts
    from .physical_constants import R_J_MOL_K
    from .property_resolution.organic_classification import (
        classify_strict_molecular_organic,
    )
else:
    from compound_identity import parse_formula_counts
    from physical_constants import R_J_MOL_K
    from property_resolution.organic_classification import (
        classify_strict_molecular_organic,
    )


class PFDSimModifiedStielThodosError(ValueError):
    """The modified Stiel-Thodos model cannot evaluate the supplied state."""


@dataclass(frozen=True)
class GroupCorrection:
    """One binary group correction ``A + B/T`` with ``B`` in kelvin."""

    A: float
    B_K: float


@dataclass(frozen=True)
class PFDSimModifiedStielThodosEvaluation:
    """Auditable result of one PFDSim-modified Stiel-Thodos evaluation."""

    value_W_per_m_K: float
    base_value_W_per_m_K: float
    correction_factor: float
    correction_exponent: float
    reduced_temperature: float
    geometry: str
    active_groups: tuple[str, ...]
    organic_correction_eligible: bool
    organic_classification_reason: str


GROUP_CORRECTIONS: Mapping[str, GroupCorrection] = MappingProxyType(
    {
        "alcohol": GroupCorrection(A=0.21989, B_K=-76.607),
        "aldehyde": GroupCorrection(A=0.20855, B_K=-78.969),
        "aliphatic_ring": GroupCorrection(
            A=0.029391,
            B_K=-17.068,
        ),
        "alkene": GroupCorrection(A=0.071467, B_K=-21.455),
        "alkyne": GroupCorrection(A=0.044307, B_K=-21.510),
        "aromatic_ring": GroupCorrection(
            A=0.0013274,
            B_K=16.477,
        ),
        "chlorine": GroupCorrection(A=0.32856, B_K=-114.42),
        "hydrocarbon_only": GroupCorrection(
            A=0.25035,
            B_K=-70.837,
        ),
        "ketone": GroupCorrection(A=0.57891, B_K=-222.27),
        "thiol": GroupCorrection(A=-0.00062287, B_K=21.590),
        "acid_straight_long_C7_plus": GroupCorrection(
            A=0.81214,
            B_K=-359.36,
        ),
        "acid_straight_medium_C4_C6": GroupCorrection(
            A=0.26548,
            B_K=9.3632,
        ),
        "acid_straight_short_C1_C3": GroupCorrection(
            A=-0.69671,
            B_K=747.60,
        ),
        "acid_unsaturated_or_aromatic_monocarboxylic": GroupCorrection(
            A=-0.031682,
            B_K=30.260,
        ),
    }
)

ZEROED_CORRECTION_GROUPS = frozenset(
    {
        "acid_branched_saturated_monocarboxylic",
        "acid_dicarboxylic",
        "amine",
        "bromine",
        "ester",
        "ether",
        "fluorine",
    }
)

_SMARTS = {
    "alcohol": "[CX4][OX2H1]",
    "aldehyde": "[CX3;H1,H2](=O)",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "amine": "[NX3;H0,H1,H2;!$(N[C,S,P]=O);!$([N+])]",
    "carboxylic_acid": "[CX3](=[OX1])[OX2H1]",
    "ketone": "[#6][CX3](=O)[#6]",
    "thiol": "[SX2H1]",
}
_PATTERNS = MappingProxyType(
    {name: Chem.MolFromSmarts(smarts) for name, smarts in _SMARTS.items()}
)


def _positive_finite(name: str, value: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise PFDSimModifiedStielThodosError(
            f"{name} must be a positive finite number"
        ) from exc
    if normalized <= 0.0 or not math.isfinite(normalized):
        raise PFDSimModifiedStielThodosError(f"{name} must be a positive finite number")
    return normalized


def _normalized_geometry(geometry: str) -> str:
    key = str(geometry or "").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "monatomic": "monatomic",
        "monoatomic": "monatomic",
        "linear": "linear",
        "nonlinear": "nonlinear",
    }
    normalized = aliases.get(key)
    if normalized is None:
        raise PFDSimModifiedStielThodosError(
            f"Unsupported molecular geometry {geometry!r}; expected monatomic, "
            "linear, or nonlinear"
        )
    return normalized


def _validated_molecule(smiles: str) -> Chem.Mol:
    text = str(smiles or "").strip()
    molecule = Chem.MolFromSmiles(text) if text else None
    if molecule is None:
        raise PFDSimModifiedStielThodosError("SMILES is empty or unparseable")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise PFDSimModifiedStielThodosError(
            "PFDSim-modified Stiel-Thodos requires one connected molecule"
        )
    if sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()) != 0:
        raise PFDSimModifiedStielThodosError(
            "PFDSim-modified Stiel-Thodos does not support charged molecules"
        )
    return molecule


def _element_counts(molecule: Chem.Mol) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom in molecule.GetAtoms():
        element = atom.GetSymbol()
        counts[element] = counts.get(element, 0) + 1
    return counts


def _hydrogen_complete_for_formula(
    molecule: Chem.Mol,
    formula_counts: Mapping[str, int] | None,
) -> Chem.Mol:
    complete = Chem.AddHs(molecule)
    if _element_counts(complete) != formula_counts:
        raise PFDSimModifiedStielThodosError(
            "Molecular formula and structure have different element counts"
        )
    return complete


def classify_molecular_geometry(formula: str, smiles: str = "") -> tuple[str, str]:
    """Classify geometry from formula and hydrogen-complete molecular topology."""
    normalized_formula = re.sub(r"D(?=\d|[A-Z]|$)", "H", str(formula or ""))
    formula_counts = parse_formula_counts(normalized_formula)
    if not formula_counts:
        raise PFDSimModifiedStielThodosError(
            "Molecular formula is unavailable or invalid"
        )
    atom_count = sum(formula_counts.values())
    if atom_count == 1:
        return "monatomic", "formula_atom_count"
    if atom_count == 2:
        return "linear", "formula_atom_count"

    molecule = _hydrogen_complete_for_formula(
        _validated_molecule(smiles), formula_counts
    )
    degrees = [atom.GetDegree() for atom in molecule.GetAtoms()]
    linear = (
        max(degrees) <= 2
        and sum(degree == 1 for degree in degrees) == 2
        and all(
            atom.GetHybridization() == Chem.HybridizationType.SP
            for atom in molecule.GetAtoms()
            if atom.GetDegree() == 2
        )
    )
    return (
        "linear" if linear else "nonlinear",
        "rdkit_hydrogen_complete_topology",
    )


def _has_match(molecule: Chem.Mol, name: str) -> bool:
    return molecule.HasSubstructMatch(_PATTERNS[name])


def _acid_group(molecule: Chem.Mol) -> str | None:
    acid_count = len(molecule.GetSubstructMatches(_PATTERNS["carboxylic_acid"]))
    if acid_count == 0:
        return None
    if acid_count >= 2:
        return "acid_dicarboxylic"

    carbon_atoms = [atom for atom in molecule.GetAtoms() if atom.GetSymbol() == "C"]
    branched = any(
        sum(neighbor.GetSymbol() == "C" for neighbor in atom.GetNeighbors()) >= 3
        for atom in carbon_atoms
    )
    unsaturated_or_aromatic = any(
        atom.GetIsAromatic()
        or any(
            bond.GetBondTypeAsDouble() > 1.0
            and {
                bond.GetBeginAtom().GetSymbol(),
                bond.GetEndAtom().GetSymbol(),
            }
            != {"C", "O"}
            for bond in atom.GetBonds()
        )
        for atom in carbon_atoms
    )
    if unsaturated_or_aromatic:
        return "acid_unsaturated_or_aromatic_monocarboxylic"
    if branched:
        return "acid_branched_saturated_monocarboxylic"
    if len(carbon_atoms) <= 3:
        return "acid_straight_short_C1_C3"
    if len(carbon_atoms) <= 6:
        return "acid_straight_medium_C4_C6"
    return "acid_straight_long_C7_plus"


def _correction_profile(
    smiles: str,
    formula: str,
    cas: str,
) -> tuple[tuple[str, ...], bool, str]:
    formula_text = str(formula or "").strip()
    if not formula_text:
        raise PFDSimModifiedStielThodosError(
            "Molecular formula is required for the organic correction gate"
        )
    formula_identity = classify_strict_molecular_organic(
        cas=cas,
        formula=formula_text,
    )
    if not formula_identity.is_organic:
        return (), False, formula_identity.reason
    molecule = _validated_molecule(smiles)
    _hydrogen_complete_for_formula(molecule, parse_formula_counts(formula_text))
    identity = classify_strict_molecular_organic(
        cas=cas,
        formula=formula_text,
        smiles=smiles,
    )
    if not identity.is_organic:
        raise PFDSimModifiedStielThodosError(
            f"Organic formula and molecular structure are inconsistent: {identity.reason}"
        )
    groups = {
        name
        for name in (
            "alcohol",
            "aldehyde",
            "alkene",
            "alkyne",
            "amine",
            "ketone",
            "thiol",
        )
        if _has_match(molecule, name)
    }
    if rdMolDescriptors.CalcNumAliphaticRings(molecule) > 0:
        groups.add("aliphatic_ring")
    if rdMolDescriptors.CalcNumAromaticRings(molecule) > 0:
        groups.add("aromatic_ring")
    if any(atom.GetSymbol() == "Cl" for atom in molecule.GetAtoms()):
        groups.add("chlorine")
    heavy_elements = {
        atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetSymbol() != "H"
    }
    if heavy_elements == {"C"}:
        groups.add("hydrocarbon_only")
    acid_group = _acid_group(molecule)
    if acid_group in GROUP_CORRECTIONS:
        groups.add(acid_group)
    return (
        tuple(name for name in GROUP_CORRECTIONS if name in groups),
        True,
        identity.reason,
    )


def correction_groups(
    smiles: str,
    formula: str,
    cas: str = "",
) -> tuple[str, ...]:
    """Return active binary groups after the strict molecular-organic gate."""
    groups, _eligible, _reason = _correction_profile(smiles, formula, cas)
    return groups


def evaluate_pfdsim_modified_stiel_thodos(
    temperature_K: float,
    viscosity_Pa_s: float,
    cv_J_per_mol_K: float,
    molecular_weight_g_per_mol: float,
    critical_temperature_K: float,
    geometry: str,
    smiles: str,
    formula: str,
    cas: str = "",
) -> PFDSimModifiedStielThodosEvaluation:
    """Evaluate the PFDSim-modified Stiel-Thodos dilute-gas method."""
    temperature = _positive_finite("temperature_K", temperature_K)
    viscosity = _positive_finite("viscosity_Pa_s", viscosity_Pa_s)
    cv = _positive_finite("cv_J_per_mol_K", cv_J_per_mol_K)
    molecular_weight = _positive_finite(
        "molecular_weight_g_per_mol", molecular_weight_g_per_mol
    )
    critical_temperature = _positive_finite(
        "critical_temperature_K", critical_temperature_K
    )
    normalized_geometry = _normalized_geometry(geometry)
    reduced_temperature = temperature / critical_temperature

    if normalized_geometry == "monatomic":
        stiel_thodos_factor = 2.5
    elif normalized_geometry == "linear":
        stiel_thodos_factor = 1.3 + (R_J_MOL_K / cv) * (
            1.7614 - 0.3523 / reduced_temperature
        )
    else:
        stiel_thodos_factor = 1.15 + 2.033 * R_J_MOL_K / cv
    if stiel_thodos_factor <= 0.0 or not math.isfinite(stiel_thodos_factor):
        raise PFDSimModifiedStielThodosError(
            "Stiel-Thodos dimensionless factor is nonpositive or nonfinite"
        )

    molecular_weight_kg_per_mol = molecular_weight / 1000.0
    base_value = stiel_thodos_factor * viscosity * cv / molecular_weight_kg_per_mol
    active_groups, correction_eligible, classification_reason = _correction_profile(
        smiles,
        formula,
        cas,
    )
    correction_exponent = sum(
        GROUP_CORRECTIONS[group].A + GROUP_CORRECTIONS[group].B_K / temperature
        for group in active_groups
    )
    try:
        correction_factor = math.exp(correction_exponent)
    except OverflowError as exc:
        raise PFDSimModifiedStielThodosError(
            "Functional-group correction overflowed"
        ) from exc
    value = base_value * correction_factor
    if value <= 0.0 or not math.isfinite(value):
        raise PFDSimModifiedStielThodosError(
            "PFDSim-modified Stiel-Thodos conductivity is nonphysical"
        )
    return PFDSimModifiedStielThodosEvaluation(
        value_W_per_m_K=value,
        base_value_W_per_m_K=base_value,
        correction_factor=correction_factor,
        correction_exponent=correction_exponent,
        reduced_temperature=reduced_temperature,
        geometry=normalized_geometry,
        active_groups=active_groups,
        organic_correction_eligible=correction_eligible,
        organic_classification_reason=classification_reason,
    )


__all__ = [
    "GROUP_CORRECTIONS",
    "ZEROED_CORRECTION_GROUPS",
    "GroupCorrection",
    "PFDSimModifiedStielThodosError",
    "PFDSimModifiedStielThodosEvaluation",
    "classify_molecular_geometry",
    "correction_groups",
    "evaluate_pfdsim_modified_stiel_thodos",
]
