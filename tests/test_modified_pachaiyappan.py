import math

import pytest
from rdkit import Chem

from modified_pachaiyappan import (
    LOCAL_V20_CARBON,
    LOCAL_V20_HYDROGEN,
    LOCAL_V20_INTERCEPT,
    LOCAL_V20_RING,
    STRAIGHT_CHAIN_PARAMETERS,
    ModifiedPachaiyappanError,
    conductivity_W_m_K,
    describe_hydrocarbon,
    estimate_molar_volume_20C_cm3_mol,
)


def test_reference_temperature_reduces_to_base_relation():
    molecular_weight = 100.0
    volume = 120.0
    A, B = STRAIGHT_CHAIN_PARAMETERS

    value = conductivity_W_m_K(
        293.15,
        molecular_weight_g_mol=molecular_weight,
        critical_temperature_K=500.0,
        molar_volume_20C_cm3_mol=volume,
        straight_chain=True,
    )

    assert math.isclose(value, A * molecular_weight**B / volume, rel_tol=1e-14)


@pytest.mark.parametrize(
    ("smiles", "carbon", "hydrogen", "rings", "straight"),
    (
        ("CCC", 3, 8, 0, True),
        ("C=CCC", 4, 8, 0, True),
        ("CC(C)C", 4, 10, 0, False),
        ("c1ccccc1", 6, 6, 1, False),
        ("c1ccc(-c2ccccc2)cc1", 12, 10, 2, False),
    ),
)
def test_hydrocarbon_descriptors(smiles, carbon, hydrogen, rings, straight):
    descriptors = describe_hydrocarbon(Chem.MolFromSmiles(smiles))

    assert descriptors.carbon_atoms == carbon
    assert descriptors.hydrogen_atoms == hydrogen
    assert descriptors.rings == rings
    assert descriptors.straight_chain is straight


@pytest.mark.parametrize("smiles", ("CC", "CCO", "C.CC"))
def test_domain_rejects_small_heteroatom_and_disconnected_molecules(smiles):
    with pytest.raises(ModifiedPachaiyappanError):
        describe_hydrocarbon(Chem.MolFromSmiles(smiles))


def test_local_volume_estimate_uses_carbon_hydrogen_and_each_ring():
    descriptors = describe_hydrocarbon(Chem.MolFromSmiles("c1ccccc1"))

    value = estimate_molar_volume_20C_cm3_mol(descriptors)

    expected = (
        LOCAL_V20_INTERCEPT
        + 6 * LOCAL_V20_CARBON
        + 6 * LOCAL_V20_HYDROGEN
        + LOCAL_V20_RING
    )
    assert math.isclose(value, expected, rel_tol=1e-14)


def test_correlation_rejects_invalid_critical_domain():
    with pytest.raises(ModifiedPachaiyappanError, match="Tc must exceed"):
        conductivity_W_m_K(
            280.0,
            molecular_weight_g_mol=44.0,
            critical_temperature_K=290.0,
            molar_volume_20C_cm3_mol=90.0,
            straight_chain=True,
        )
    with pytest.raises(ModifiedPachaiyappanError, match="no greater than Tc"):
        conductivity_W_m_K(
            510.0,
            molecular_weight_g_mol=44.0,
            critical_temperature_K=500.0,
            molar_volume_20C_cm3_mol=90.0,
            straight_chain=True,
        )
