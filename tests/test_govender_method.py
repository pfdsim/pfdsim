"""Published worked examples for Govender et al., JCED 65 (2020) 1300."""

import math
import os
import sys

import pytest
from rdkit import Chem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import govender_method as gm
from govender_method import GovenderError, estimate, estimate_from_mol


@pytest.mark.parametrize(
    "smiles,tb,temperature,groups,conductivity",
    [
        ("CCc1ccccc1", 409.3, 277.34,
         {1: 1, 4: 1, 16: 5, 17: 1}, 0.1372),
        ("C1=CC=CC1", 313.8, 283.15,
         {10: 1, 21: 2, 32: 1}, 0.1509),
        ("c1ccc2ccccc2c1", 491.3, 438.83,
         {16: 8, 19: 2}, 0.1096),
        ("c1ccccc1-c1ccccc1", 529.0, 353.3,
         {16: 10, 17: 2}, 0.1329),
        ("FC(F)(Br)C(Cl)(F)Br", 366.02, 323.15,
         {9: 2, 39: 3, 44: 1, 145: 2}, 0.0604),
    ],
)
def test_published_appendix_examples(smiles, tb, temperature, groups, conductivity):
    result = estimate(smiles, tb=tb, local_refits=False)
    assert result.groups == groups
    assert result.tb_source == "provided"
    assert result.conductivity_W_m_K(temperature) == pytest.approx(
        conductivity, abs=0.0001)
    assert result.conductivity_W_m_K(tb) == pytest.approx(result.lambda_ref_W_m_K)


def test_short_alcohol_size_term_and_paper_arithmetic_discrepancy():
    result = estimate("CC(C)O", tb=355.5, local_refits=False)
    assert result.groups == {1: 2, 8: 1, 49: 3, 175: 1}
    assert result.sum_A == pytest.approx(25.86)
    assert result.sum_B == pytest.approx(-5.6925)
    # Table 11g prints 0.1468, but its coefficients and displayed equation
    # evaluate to 0.1498 at 303.15 K. Follow the equation and coefficients.
    assert result.conductivity_W_m_K(303.15) == pytest.approx(0.1498, abs=0.0001)


def test_long_alcohol_has_no_short_chain_constant():
    result = estimate("CCCCCCCCO", tb=468.0)
    assert result.groups[153] == 8
    assert 49 not in result.groups
    assert 175 not in result.groups


def test_higher_diol_is_supported_with_temperature_caution():
    result = estimate("OCCCCO", tb=560.0)
    assert result.groups[49] == 8
    assert result.groups[175] == 2
    assert any("low-temperature" in warning for warning in result.warnings)


def test_alkene_carbon_substituent_count():
    assert estimate("CC=CCC", tb=350).groups == {1: 2, 4: 1, 20: 2}
    assert estimate("CC(C)=CCC", tb=350).groups == {
        1: 3, 4: 1, 20: 1, 27: 1}


def test_internal_tb_uses_existing_nannoolal_estimator():
    result = estimate("CCc1ccccc1")
    assert result.tb_source == "estimated"
    assert result.tb_K > 0
    assert estimate_from_mol(Chem.MolFromSmiles("CCc1ccccc1"), tb=409.3).tb_source == "provided"


@pytest.mark.parametrize(
    "smiles,required_group",
    [
        ("CC(=O)O", 53), ("CC(=O)OC", 54), ("COC=O", 55),
        ("CC(=O)C", 58), ("CC=O", 59), ("COC(=O)OC", 61),
        ("CC1COC(=O)O1", 66), ("CC(=O)N", 70),
        ("CN(C)C=O", 72), ("C[N+](=O)[O-]", 75),
        ("c1ccncc1", 88), ("CC#N", 89),
        ("COP(=O)(OC)OC", 95), ("C[Sn](C)(C)C", 110),
        ("C1CO1", 51),
    ],
)
def test_other_published_functional_groups(smiles, required_group):
    assert required_group in estimate(smiles, tb=350).groups


def test_unpublished_group_and_invalid_states_refused():
    with pytest.raises(GovenderError, match="small glycols"):
        estimate("OCCO", tb=470.5)
    with pytest.raises(GovenderError, match="no published carbon group"):
        estimate("C#CC", tb=298.0)
    with pytest.raises(GovenderError, match="tb must"):
        estimate("CCO", tb=math.nan)
    with pytest.raises(GovenderError, match="temperature_K must"):
        estimate("CCO", tb=351.4).conductivity_W_m_K(0)
    with pytest.raises(GovenderError, match="carbonate group requires a diester"):
        estimate("COC(=O)O", tb=350)


def test_local_group_refits_default_on_and_can_be_disabled():
    cases = [
        ("CC(=O)O", 53),
        ("CC(=O)C", 58),
        ("CCN(CC)CC", 84),
        ("CCSCC", 100),
    ]
    for smiles, gid in cases:
        local = estimate(smiles, tb=400.0)
        published = estimate(smiles, tb=400.0, local_refits=False)
        frequency = local.groups[gid]
        expected_A = (
            published.sum_A
            + frequency * (
                gm.LOCAL_GROUP_REFITS[gid][0]
                - gm.CONTRIBUTIONS[gid][0]
            )
        )
        expected_B = (
            published.sum_B
            + frequency * (
                gm.LOCAL_GROUP_REFITS[gid][1]
                - gm.CONTRIBUTIONS[gid][1]
            )
        )
        assert local.local_refits is True
        assert published.local_refits is False
        assert local.sum_A == pytest.approx(expected_A)
        assert local.sum_B == pytest.approx(expected_B)
        assert any(f"group {gid}" in warning for warning in local.warnings)


def test_paper_single_component_groups_are_explicit():
    assert gm.PAPER_SINGLE_COMPONENT_GROUPS == {61, 88, 95, 110}
    assert gm.PAPER_SINGLE_COMPONENT_GROUPS < gm.PAPER_CAUTION_GROUPS


def test_local_ring_and_amine_alcohol_corrections():
    local = estimate("C1CO1", tb=400.0)
    published = estimate("C1CO1", tb=400.0, local_refits=False)
    assert local.sum_A == pytest.approx(
        published.sum_A + gm.LOCAL_RING_CORRECTIONS[3][0]
    )
    assert local.sum_B == pytest.approx(
        published.sum_B + gm.LOCAL_RING_CORRECTIONS[3][1]
    )

    local = estimate("CN(C)CCO", tb=400.0)
    published = estimate("CN(C)CCO", tb=400.0, local_refits=False)
    group_refit = gm.LOCAL_GROUP_REFITS[84]
    group_published = gm.CONTRIBUTIONS[84]
    interaction = gm.LOCAL_AMINE_ALCOHOL_CORRECTION
    assert local.sum_A == pytest.approx(
        published.sum_A + group_refit[0] - group_published[0] + interaction[0]
    )
    assert local.sum_B == pytest.approx(
        published.sum_B + group_refit[1] - group_published[1] + interaction[1]
    )


@pytest.mark.parametrize("local_refits", [False, True])
def test_do_not_estimate_domains_are_unconditional(local_refits):
    with pytest.raises(GovenderError, match="cyclic thioether: do not estimate"):
        estimate("C1CCSC1", tb=400.0, local_refits=local_refits)
    with pytest.raises(
        GovenderError, match="unsaturated cyclic ketone: do not estimate"
    ):
        estimate("O=C1C=CC(=O)C=C1", tb=400.0, local_refits=local_refits)


@pytest.mark.parametrize("local_refits", [False, True])
def test_aromatic_carbonyl_cannot_bypass_cyclic_ketone_refusal(local_refits):
    with pytest.raises(GovenderError, match="unsaturated cyclic ketone"):
        estimate("O=c1ccoc2ccccc12", tb=400.0, local_refits=local_refits)


@pytest.mark.parametrize("local_refits", [False, True])
@pytest.mark.parametrize(
    "smiles,expected_groups",
    [
        ("O=c1cccco1", {21: 2, 32: 1, 54: 1}),
        ("O=c1ccccn1C", {2: 1, 21: 2, 32: 1, 72: 1}),
    ],
)
def test_aromatic_carbonyl_groups_use_normalized_molecule(
    smiles, expected_groups, local_refits
):
    molecule = Chem.MolFromSmiles(smiles)
    original_smiles = Chem.MolToSmiles(molecule)
    original_aromaticity = [atom.GetIsAromatic() for atom in molecule.GetAtoms()]
    result = estimate_from_mol(molecule, tb=400.0, local_refits=local_refits)

    assert result.groups == expected_groups
    assert result.smiles == original_smiles
    assert Chem.MolToSmiles(molecule) == original_smiles
    assert [atom.GetIsAromatic() for atom in molecule.GetAtoms()] == original_aromaticity
