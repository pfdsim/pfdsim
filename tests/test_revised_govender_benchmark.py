import os
import sys

import pytest
from rdkit import Chem


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import govender_method as gm
from scripts.thermal_conductivity.liquid.benchmark_revised_govender_perry import (
    cyclic_thioether,
    cyclic_unsaturated_ketone,
    domain_reasons,
)


def molecule_and_groups(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    molecule, groups = gm._fragment(molecule)
    return molecule, dict(groups)


def test_screened_domain_classes_are_structural():
    molecule, groups = molecule_and_groups("C=C")
    assert domain_reasons(molecule, groups) == ("fewer than 3 carbon atoms",)

    molecule, groups = molecule_and_groups("C1CCSC1")
    assert cyclic_thioether(molecule, groups)
    assert "cyclic thioether" in domain_reasons(molecule, groups)

    molecule, groups = molecule_and_groups("CCSCC")
    assert not cyclic_thioether(molecule, groups)

    molecule, groups = molecule_and_groups("O=C1C=CC(=O)C=C1")
    assert cyclic_unsaturated_ketone(molecule, groups)
    assert "unsaturated cyclic ketone" in domain_reasons(molecule, groups)

    molecule, groups = molecule_and_groups("O=C1CCCCC1")
    assert not cyclic_unsaturated_ketone(molecule, groups)


def test_revised_parameters_apply_only_qualified_refits():
    local = gm.estimate("CC(=O)C", tb=400.0)
    published = gm.estimate("CC(=O)C", tb=400.0, local_refits=False)
    assert local.sum_A != published.sum_A
    assert any("group 58" in warning for warning in local.warnings)

    with pytest.raises(gm.GovenderError, match="unsaturated cyclic ketone"):
        gm.estimate("O=C1C=CC(=O)C=C1", tb=400.0)

    local = gm.estimate("CCSCC", tb=400.0)
    published = gm.estimate("CCSCC", tb=400.0, local_refits=False)
    assert local.sum_A != published.sum_A
    assert any("group 100" in warning for warning in local.warnings)

    with pytest.raises(gm.GovenderError, match="cyclic thioether"):
        gm.estimate("C1CCSC1", tb=400.0)


def test_ring_and_amine_alcohol_corrections_are_added_once():
    local = gm.estimate("C1CO1", tb=400.0)
    published = gm.estimate("C1CO1", tb=400.0, local_refits=False)
    assert local.sum_A == pytest.approx(
        published.sum_A + gm.LOCAL_RING_CORRECTIONS[3][0]
    )
    assert local.sum_B == pytest.approx(
        published.sum_B + gm.LOCAL_RING_CORRECTIONS[3][1]
    )
    assert any("3-member" in warning for warning in local.warnings)

    local = gm.estimate("CN(C)CCO", tb=400.0)
    published = gm.estimate("CN(C)CCO", tb=400.0, local_refits=False)
    assert local.sum_A == pytest.approx(
        published.sum_A
        + gm.LOCAL_GROUP_REFITS[84][0] - gm.CONTRIBUTIONS[84][0]
        + gm.LOCAL_AMINE_ALCOHOL_CORRECTION[0]
    )
    assert local.sum_B == pytest.approx(
        published.sum_B
        + gm.LOCAL_GROUP_REFITS[84][1] - gm.CONTRIBUTIONS[84][1]
        + gm.LOCAL_AMINE_ALCOHOL_CORRECTION[1]
    )
    assert sum("amine-alcohol" in warning for warning in local.warnings) == 1
