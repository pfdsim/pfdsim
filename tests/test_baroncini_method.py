import math

import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from baroncini_method import (
    ALDEHYDE_PARAMETERS,
    KETONE_POSITION_LOG_INTERCEPT,
    KETONE_POSITION_LOG_MW,
    KETONE_POSITION_SIDE_MIN,
    BaronciniError,
    classify_molecule,
    conductivity_W_m_K,
    evaluate_from_mol,
)


@pytest.mark.parametrize(
    ("smiles", "category", "subtype"),
    (
        ("CCCO", "alcohols", "monohydric_nonphenolic"),
        ("CC(=O)O", "organic_acids", "monocarboxylic"),
        ("O=C(O)CC(=O)O", "organic_acids", "dicarboxylic"),
        ("CC(=O)OCC", "esters", "monoester"),
        ("CCOCC", "ethers", "acyclic_ether"),
        ("CCCC=O", "aldehydes", "aliphatic_single_aldehyde_local_refit"),
        ("CC(=O)CC", "ketones", "acyclic_aliphatic_monoketone_local_refit"),
        ("O=C(c1ccccc1)c1ccccc1", "ketones", "aromatic_monoketone_published"),
        ("CCCl", "halogenated_hydrocarbons", "C2_plus_general_refrigerant"),
        ("FC(Cl)C(F)F", "halogenated_hydrocarbons", "C2_plus_general_refrigerant"),
    ),
)
def test_selected_classification(smiles, category, subtype):
    result = classify_molecule(Chem.MolFromSmiles(smiles))

    assert result.category == category
    assert result.subtype == subtype


@pytest.mark.parametrize(
    "smiles",
    (
        "Oc1ccccc1",
        "OCCO",
        "O=CO",
        "COC(=O)c1ccccc1C(=O)OC",
        "C1CCOC1",
        "O=C1CCCCC1",
        "O=Cc1ccccc1",
        "CCl",
        "CCC",
    ),
)
def test_selected_domain_rejections(smiles):
    with pytest.raises(BaronciniError):
        classify_molecule(Chem.MolFromSmiles(smiles))


def test_aliphatic_ketone_multiplier_matches_selected_refit():
    molecule = Chem.MolFromSmiles("CC(=O)CC")
    molecular_weight = Descriptors.MolWt(molecule)

    result = classify_molecule(molecule)

    expected = math.exp(
        KETONE_POSITION_LOG_INTERCEPT
        + KETONE_POSITION_LOG_MW * math.log(molecular_weight)
        + KETONE_POSITION_SIDE_MIN
    )
    assert math.isclose(result.multiplier, expected, rel_tol=1e-14)


def test_aldehyde_uses_local_parameters():
    result = classify_molecule(Chem.MolFromSmiles("CCCC=O"))

    assert result.parameters == ALDEHYDE_PARAMETERS


def test_equation_matches_manual_evaluation():
    molecule = Chem.MolFromSmiles("CCOCC")
    classification = classify_molecule(molecule)
    temperature = 300.0
    tb = 307.6
    tc = 466.7
    molecular_weight = Descriptors.MolWt(molecule)
    A, a, b, c = classification.parameters
    reduced_temperature = temperature / tc
    expected = (
        A
        * tb**a
        * molecular_weight ** (-b)
        * tc ** (-c)
        * (1.0 - reduced_temperature) ** 0.38
        * reduced_temperature ** (-1.0 / 6.0)
    )

    value = conductivity_W_m_K(
        temperature,
        normal_boiling_temperature_K=tb,
        critical_temperature_K=tc,
        molecular_weight_g_mol=molecular_weight,
        classification=classification,
    )

    assert math.isclose(value, expected, rel_tol=1e-14)


def test_halocarbon_does_not_require_boiling_point():
    result = evaluate_from_mol(
        Chem.MolFromSmiles("CCCl"),
        250.0,
        normal_boiling_temperature_K=None,
        critical_temperature_K=460.0,
    )

    assert result.value_W_per_m_K > 0.0
    assert not result.classification.requires_boiling_point


def test_oxygenated_class_requires_boiling_point():
    with pytest.raises(BaronciniError, match="requires a valid Tb"):
        evaluate_from_mol(
            Chem.MolFromSmiles("CCOCC"),
            250.0,
            normal_boiling_temperature_K=None,
            critical_temperature_K=460.0,
        )


def test_temperature_must_be_below_critical():
    classification = classify_molecule(Chem.MolFromSmiles("CCOCC"))
    with pytest.raises(BaronciniError, match="below Tc"):
        conductivity_W_m_K(
            460.0,
            normal_boiling_temperature_K=307.6,
            critical_temperature_K=460.0,
            molecular_weight_g_mol=74.12,
            classification=classification,
        )
