"""Modified Pachaiyappan liquid thermal conductivity for hydrocarbons.

The hydrocarbon correlation is attributed to Pachaiyappan, V., S. H.
Ibrahim, and N. R. Kuloor, *Chemical Engineering* 74(4) (1967), 140.  This
implementation uses the supplied modified parameter pairs and a local
structure estimate for the 20 degree C liquid molar volume when sufficiently
reliable density data are unavailable.

The local volume model and validation are documented under
``scripts/thermal_conductivity/liquid/``.  It is intentionally private to this
conductivity method; it is not a general-purpose liquid-volume estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors


REFERENCE_TEMPERATURE_K = 293.15
MINIMUM_CARBON_ATOMS = 3
STRAIGHT_CHAIN_PARAMETERS = (0.1811, 1.001)
OTHER_HYDROCARBON_PARAMETERS = (0.4407, 0.7717)

# Relative-error fit on 84 Perry hydrocarbons with at least three carbons.
# See probe_hydrocarbon_v20_estimation.py and its result artifacts.
LOCAL_V20_INTERCEPT = 28.08212031899058
LOCAL_V20_CARBON = 8.263762641940371
LOCAL_V20_HYDROGEN = 3.8098180968093285
LOCAL_V20_RING = -10.37762344628454


class ModifiedPachaiyappanError(ValueError):
    """Raised when the modified Pachaiyappan method is inapplicable."""


@dataclass(frozen=True)
class HydrocarbonDescriptors:
    carbon_atoms: int
    hydrogen_atoms: int
    rings: int
    straight_chain: bool


@dataclass(frozen=True)
class ModifiedPachaiyappanResult:
    value_W_per_m_K: float
    temperature_K: float
    critical_temperature_K: float
    molecular_weight_g_mol: float
    molar_volume_20C_cm3_mol: float
    descriptors: HydrocarbonDescriptors


def describe_hydrocarbon(molecule: Chem.Mol) -> HydrocarbonDescriptors:
    """Return the structural descriptors required by the correlation."""
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ModifiedPachaiyappanError("a single connected molecule is required")
    if any(atom.GetAtomicNum() not in {1, 6} for atom in molecule.GetAtoms()):
        raise ModifiedPachaiyappanError("method is restricted to hydrocarbons")
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
    if carbon_atoms < MINIMUM_CARBON_ATOMS:
        raise ModifiedPachaiyappanError(
            f"method requires at least {MINIMUM_CARBON_ATOMS} carbon atoms"
        )
    hydrogen_atoms = sum(
        atom.GetAtomicNum() == 1 for atom in Chem.AddHs(molecule).GetAtoms()
    )
    rings = len(molecule.GetRingInfo().AtomRings())
    straight_chain = rings == 0 and all(
        sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) <= 2
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6
    )
    return HydrocarbonDescriptors(
        carbon_atoms=carbon_atoms,
        hydrogen_atoms=hydrogen_atoms,
        rings=rings,
        straight_chain=straight_chain,
    )


def estimate_molar_volume_20C_cm3_mol(
    descriptors: HydrocarbonDescriptors,
) -> float:
    """Return the method-local hypothetical-liquid V20 estimate [cm3/mol]."""
    value = (
        LOCAL_V20_INTERCEPT
        + LOCAL_V20_CARBON * descriptors.carbon_atoms
        + LOCAL_V20_HYDROGEN * descriptors.hydrogen_atoms
        + LOCAL_V20_RING * descriptors.rings
    )
    if value <= 0.0:
        raise ModifiedPachaiyappanError("estimated V20 is non-positive")
    return value


def conductivity_W_m_K(
    temperature_K: float,
    *,
    molecular_weight_g_mol: float,
    critical_temperature_K: float,
    molar_volume_20C_cm3_mol: float,
    straight_chain: bool,
) -> float:
    """Evaluate the modified Pachaiyappan correlation in W/(m*K)."""
    if not 0.0 < temperature_K <= critical_temperature_K:
        raise ModifiedPachaiyappanError(
            "temperature must be positive and no greater than Tc"
        )
    if critical_temperature_K <= REFERENCE_TEMPERATURE_K:
        raise ModifiedPachaiyappanError("Tc must exceed 293.15 K")
    if molecular_weight_g_mol <= 0.0 or molar_volume_20C_cm3_mol <= 0.0:
        raise ModifiedPachaiyappanError(
            "molecular weight and V20 must be positive"
        )
    A, B = (
        STRAIGHT_CHAIN_PARAMETERS
        if straight_chain
        else OTHER_HYDROCARBON_PARAMETERS
    )
    numerator = 3.0 + 20.0 * (
        1.0 - temperature_K / critical_temperature_K
    ) ** (2.0 / 3.0)
    denominator = 3.0 + 20.0 * (
        1.0 - REFERENCE_TEMPERATURE_K / critical_temperature_K
    ) ** (2.0 / 3.0)
    return A * molecular_weight_g_mol**B / molar_volume_20C_cm3_mol * (
        numerator / denominator
    )


def evaluate_from_mol(
    molecule: Chem.Mol,
    temperature_K: float,
    *,
    critical_temperature_K: float,
    molar_volume_20C_cm3_mol: float,
) -> ModifiedPachaiyappanResult:
    """Validate a hydrocarbon molecule and evaluate its conductivity."""
    descriptors = describe_hydrocarbon(molecule)
    molecular_weight = float(Descriptors.MolWt(molecule))
    value = conductivity_W_m_K(
        temperature_K,
        molecular_weight_g_mol=molecular_weight,
        critical_temperature_K=critical_temperature_K,
        molar_volume_20C_cm3_mol=molar_volume_20C_cm3_mol,
        straight_chain=descriptors.straight_chain,
    )
    return ModifiedPachaiyappanResult(
        value_W_per_m_K=value,
        temperature_K=temperature_K,
        critical_temperature_K=critical_temperature_K,
        molecular_weight_g_mol=molecular_weight,
        molar_volume_20C_cm3_mol=molar_volume_20C_cm3_mol,
        descriptors=descriptors,
    )
