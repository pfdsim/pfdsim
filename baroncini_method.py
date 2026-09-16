"""Selected-domain Baroncini saturated-liquid thermal conductivity.

The base equation is

    k = A Tb**a M**(-b) Tc**(-c) (1 - Tr)**0.38 Tr**(-1/6)

with temperatures in K, molecular weight in g/mol, and conductivity in
W/(m*K). Published class coefficients are combined with the exclusions and
locally validated ketone/aldehyde adjustments documented under
``scripts/thermal_conductivity/liquid/``.

Ordinary hydrocarbons are intentionally outside this implementation because
production Modified Pachaiyappan is substantially more accurate for them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors


REDUCED_TEMPERATURE_EXPONENT = 0.38

PUBLISHED_PARAMETERS = {
    "alcohols": (0.00339, 1.2, 0.5, 0.167),
    "organic_acids": (0.00319, 1.2, 0.5, 0.167),
    "ketones": (0.00383, 1.2, 0.5, 0.167),
    "esters": (0.0415, 1.2, 1.0, 0.167),
    "ethers": (0.0385, 1.2, 1.0, 0.167),
    "halogenated_hydrocarbons": (0.494, 0.0, 0.5, -0.167),
}

ALDEHYDE_PARAMETERS = (0.011004414117711718, 1.2, 0.7284702063174113, 0.167)
KETONE_POSITION_LOG_INTERCEPT = 1.0234172949294005
KETONE_POSITION_LOG_MW = -0.24303994126694795
KETONE_POSITION_SIDE_MIN = 0.04093658586474451

HALOGEN_ATOMIC_NUMBERS = frozenset({9, 17, 35, 53})
CARBOXYLIC_ACID = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")
ESTER = Chem.MolFromSmarts("[CX3](=[OX1])[OX2][#6]")
KETONE = Chem.MolFromSmarts("[CX3](=[OX1])([#6])[#6]")
ALDEHYDE = Chem.MolFromSmarts("[CX3H1](=[OX1])[#6]")
ALIPHATIC_ALCOHOL = Chem.MolFromSmarts("[OX2H1][#6;!a;!$(C=O)]")
ETHER = Chem.MolFromSmarts("[OD2]([#6])[#6]")


class BaronciniError(ValueError):
    """Raised when Baroncini is inapplicable or receives invalid inputs."""


@dataclass(frozen=True)
class BaronciniClassification:
    category: str
    subtype: str
    carbon_atoms: int
    parameters: tuple[float, float, float, float]
    multiplier: float = 1.0

    @property
    def requires_boiling_point(self) -> bool:
        return self.parameters[1] != 0.0


@dataclass(frozen=True)
class BaronciniResult:
    value_W_per_m_K: float
    temperature_K: float
    normal_boiling_temperature_K: float | None
    critical_temperature_K: float
    molecular_weight_g_mol: float
    classification: BaronciniClassification


def _connected_molecule(molecule: Chem.Mol) -> None:
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise BaronciniError("a single connected molecule is required")


def _carbon_count(molecule: Chem.Mol) -> int:
    return sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())


def _oxygen_count(molecule: Chem.Mol) -> int:
    return sum(atom.GetAtomicNum() == 8 for atom in molecule.GetAtoms())


def _ketone_side_min(molecule: Chem.Mol, carbonyl_index: int) -> int:
    carbonyl = molecule.GetAtomWithIdx(carbonyl_index)
    attached = [
        atom.GetIdx()
        for atom in carbonyl.GetNeighbors()
        if atom.GetAtomicNum() == 6
    ]
    if len(attached) != 2:
        raise BaronciniError("ketone carbonyl requires two carbon neighbors")

    def count_side(start: int) -> int:
        visited = {carbonyl_index}
        pending = [start]
        count = 0
        while pending:
            index = pending.pop()
            if index in visited:
                continue
            visited.add(index)
            atom = molecule.GetAtomWithIdx(index)
            if atom.GetAtomicNum() == 6:
                count += 1
            pending.extend(
                neighbor.GetIdx()
                for neighbor in atom.GetNeighbors()
                if neighbor.GetIdx() not in visited
            )
        return count

    return min(count_side(index) for index in attached)


def classify_molecule(molecule: Chem.Mol) -> BaronciniClassification:
    """Classify a molecule in the validated production Baroncini domain."""
    _connected_molecule(molecule)
    atomic_numbers = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
    carbon_atoms = _carbon_count(molecule)
    halogens = atomic_numbers & HALOGEN_ATOMIC_NUMBERS
    if (
        carbon_atoms >= 2
        and halogens
        and not atomic_numbers - ({1, 6} | HALOGEN_ATOMIC_NUMBERS)
    ):
        return BaronciniClassification(
            category="halogenated_hydrocarbons",
            subtype="C2_plus_general_refrigerant",
            carbon_atoms=carbon_atoms,
            parameters=PUBLISHED_PARAMETERS["halogenated_hydrocarbons"],
        )

    heavy_elements = {number for number in atomic_numbers if number != 1}
    if not heavy_elements <= {6, 8} or 6 not in heavy_elements or 8 not in heavy_elements:
        raise BaronciniError("molecule is outside the selected Baroncini classes")
    oxygen_atoms = _oxygen_count(molecule)
    acid_matches = molecule.GetSubstructMatches(CARBOXYLIC_ACID)
    ester_matches = molecule.GetSubstructMatches(ESTER)
    ketone_matches = molecule.GetSubstructMatches(KETONE)
    aldehyde_matches = molecule.GetSubstructMatches(ALDEHYDE)
    alcohol_matches = molecule.GetSubstructMatches(ALIPHATIC_ALCOHOL)
    ether_matches = molecule.GetSubstructMatches(ETHER)

    if acid_matches and oxygen_atoms == 2 * len(acid_matches):
        if len(acid_matches) > 2:
            raise BaronciniError("more than two carboxylic-acid groups are unsupported")
        if carbon_atoms == 1:
            raise BaronciniError("formic acid is outside the selected Baroncini domain")
        return BaronciniClassification(
            category="organic_acids",
            subtype=("dicarboxylic" if len(acid_matches) == 2 else "monocarboxylic"),
            carbon_atoms=carbon_atoms,
            parameters=PUBLISHED_PARAMETERS["organic_acids"],
        )

    if len(ester_matches) == 1 and oxygen_atoms == 2:
        return BaronciniClassification(
            category="esters",
            subtype="monoester",
            carbon_atoms=carbon_atoms,
            parameters=PUBLISHED_PARAMETERS["esters"],
        )
    if ester_matches:
        raise BaronciniError("diesters and mixed ester functionality are unsupported")

    if len(ketone_matches) == 1 and oxygen_atoms == 1:
        carbonyl_index = ketone_matches[0][0]
        if molecule.GetAtomWithIdx(carbonyl_index).IsInRing():
            raise BaronciniError("cyclic-carbonyl ketones are unsupported")
        if any(atom.GetIsAromatic() for atom in molecule.GetAtoms()):
            return BaronciniClassification(
                category="ketones",
                subtype="aromatic_monoketone_published",
                carbon_atoms=carbon_atoms,
                parameters=PUBLISHED_PARAMETERS["ketones"],
            )
        if molecule.GetRingInfo().NumRings():
            raise BaronciniError("nonaromatic ring-containing ketones are unsupported")
        molecular_weight = float(Descriptors.MolWt(molecule))
        smaller_side = _ketone_side_min(molecule, carbonyl_index)
        multiplier = math.exp(
            KETONE_POSITION_LOG_INTERCEPT
            + KETONE_POSITION_LOG_MW * math.log(molecular_weight)
            + KETONE_POSITION_SIDE_MIN * smaller_side
        )
        return BaronciniClassification(
            category="ketones",
            subtype="acyclic_aliphatic_monoketone_local_refit",
            carbon_atoms=carbon_atoms,
            parameters=PUBLISHED_PARAMETERS["ketones"],
            multiplier=multiplier,
        )
    if ketone_matches:
        raise BaronciniError("multiple or mixed ketone functionality is unsupported")

    if len(aldehyde_matches) == 1 and oxygen_atoms == 1:
        if any(atom.GetIsAromatic() for atom in molecule.GetAtoms()):
            raise BaronciniError("aromatic aldehydes are unsupported")
        return BaronciniClassification(
            category="aldehydes",
            subtype="aliphatic_single_aldehyde_local_refit",
            carbon_atoms=carbon_atoms,
            parameters=ALDEHYDE_PARAMETERS,
        )
    if aldehyde_matches:
        raise BaronciniError("multiple or mixed aldehyde functionality is unsupported")

    if len(alcohol_matches) == 1 and oxygen_atoms == 1:
        return BaronciniClassification(
            category="alcohols",
            subtype="monohydric_nonphenolic",
            carbon_atoms=carbon_atoms,
            parameters=PUBLISHED_PARAMETERS["alcohols"],
        )
    if alcohol_matches:
        raise BaronciniError("diols, polyols, and mixed alcohol functionality are unsupported")

    if ether_matches and oxygen_atoms == len(ether_matches):
        if molecule.GetRingInfo().NumRings():
            raise BaronciniError("cyclic ethers are unsupported")
        return BaronciniClassification(
            category="ethers",
            subtype="acyclic_ether",
            carbon_atoms=carbon_atoms,
            parameters=PUBLISHED_PARAMETERS["ethers"],
        )
    raise BaronciniError("molecule is outside the selected Baroncini classes")


def conductivity_W_m_K(
    temperature_K: float,
    *,
    normal_boiling_temperature_K: float | None,
    critical_temperature_K: float,
    molecular_weight_g_mol: float,
    classification: BaronciniClassification,
) -> float:
    """Evaluate the selected Baroncini class in W/(m*K)."""
    if not 0.0 < temperature_K < critical_temperature_K:
        raise BaronciniError("temperature must be positive and below Tc")
    if molecular_weight_g_mol <= 0.0 or not math.isfinite(molecular_weight_g_mol):
        raise BaronciniError("molecular weight must be positive and finite")
    A, a, b, c = classification.parameters
    if classification.requires_boiling_point:
        if (
            normal_boiling_temperature_K is None
            or normal_boiling_temperature_K <= 0.0
            or not math.isfinite(normal_boiling_temperature_K)
        ):
            raise BaronciniError("this Baroncini class requires a valid Tb")
        tb_factor = normal_boiling_temperature_K**a
    else:
        tb_factor = 1.0
    reduced_temperature = temperature_K / critical_temperature_K
    value = (
        classification.multiplier
        * A
        * tb_factor
        * molecular_weight_g_mol ** (-b)
        * critical_temperature_K ** (-c)
        * (1.0 - reduced_temperature) ** REDUCED_TEMPERATURE_EXPONENT
        * reduced_temperature ** (-1.0 / 6.0)
    )
    if not math.isfinite(value) or value <= 0.0:
        raise BaronciniError("Baroncini produced a non-positive prediction")
    return value


def evaluate_from_mol(
    molecule: Chem.Mol,
    temperature_K: float,
    *,
    normal_boiling_temperature_K: float | None,
    critical_temperature_K: float,
) -> BaronciniResult:
    """Classify a molecule and evaluate its selected Baroncini equation."""
    classification = classify_molecule(molecule)
    molecular_weight = float(Descriptors.MolWt(molecule))
    value = conductivity_W_m_K(
        temperature_K,
        normal_boiling_temperature_K=normal_boiling_temperature_K,
        critical_temperature_K=critical_temperature_K,
        molecular_weight_g_mol=molecular_weight,
        classification=classification,
    )
    return BaronciniResult(
        value_W_per_m_K=value,
        temperature_K=temperature_K,
        normal_boiling_temperature_K=normal_boiling_temperature_K,
        critical_temperature_K=critical_temperature_K,
        molecular_weight_g_mol=molecular_weight,
        classification=classification,
    )


def estimate(
    smiles: str,
    temperature_K: float,
    *,
    normal_boiling_temperature_K: float | None,
    critical_temperature_K: float,
) -> BaronciniResult:
    """Parse SMILES and evaluate the selected Baroncini equation."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise BaronciniError(f"could not parse SMILES {smiles!r}")
    return evaluate_from_mol(
        molecule,
        temperature_K,
        normal_boiling_temperature_K=normal_boiling_temperature_K,
        critical_temperature_K=critical_temperature_K,
    )
