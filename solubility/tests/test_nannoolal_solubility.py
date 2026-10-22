"""Check native feature fidelity and invalid-prediction screening."""

from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark_nannoolal_solubility import extract,self_check


def test_native_groups_corrections_and_silicon_are_retained():
    result = self_check()
    assert all(result.values())


def test_property_refusal_is_recorded_without_inventing_property_features():
    first,corrected,flag,_ = extract('NCC(=O)O')
    assert flag
    assert first
    assert all(label.startswith(('group_','Si_')) for label in first)
    assert all(label.startswith(('group_','Si_','correction_')) for label in corrected)
