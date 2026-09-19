"""Lyngby parameter conventions, native groups, and numerical invariants."""

import math
from pathlib import Path

import numpy as np
import pytest

from compiled_unifac import CompiledUNIFACBackend
from lyngby_parameters import parameter_table
from thermodynamics import create_thermodynamics
from unifac import UNIFACModel, get_unifac_groups
from unifac_fragmenter import fragment


@pytest.fixture(scope='module')
def model():
    return UNIFACModel(str(Path(__file__).resolve().parents[1] / 'data' / 'lyngby_unifac.json'))


def test_parameter_conventions_and_identity(model):
    assert len(model.subgroups) == 45
    assert len(parameter_table()['interactions']) == 307
    assert model.subgroups[13].R == 1.0  # Methanol differs from classic UNIFAC.
    assert model.subgroups[22].name == 'CH-O'
    assert model.subgroup_by_name['CHO'].number == 17  # Aldehyde, not ether.
    for T in (250.0, 298.15, 450.0):
        a, b, c = (972.8, 0.2687, 8.773)  # Published CH2 -> OH.
        expected = math.exp(-(a + b*(T-298.15) + c*(T*math.log(298.15/T)+T-298.15))/T)
        assert model._interaction_psi(1, 4, T) == pytest.approx(expected)
    with pytest.raises(ValueError, match='unavailable'):
        model.get_interaction(1, 99)


@pytest.mark.parametrize('smiles,expected', [
    ('CCO', {1: 1, 2: 1, 12: 1}),
    ('CO', {13: 1}), ('O', {14: 1}),
    ('Cc1ccccc1', {1: 1, 10: 5, 11: 1}),
    ('CCN', {1: 1, 2: 1, 24: 1}),
    ('OCCO', {2: 2, 12: 2}),
])
def test_native_groups(smiles, expected):
    assert fragment(smiles, 'UNIFLBY') == expected


def test_known_and_native_groups_agree():
    for name, smiles in [('ethanol', 'CCO'), ('toluene', 'Cc1ccccc1'), ('ethylamine', 'CCN')]:
        assert get_unifac_groups(name, variant='UNIFLBY') == fragment(smiles, 'UNIFLBY')
    with pytest.raises(ValueError):
        get_unifac_groups('methane', variant='UNIFLBY')


@pytest.mark.parametrize('T', [280.0, 350.0, 500.0])
def test_compiled_reference_and_gibbs_duhem(model, T):
    groups = {'ethanol': {1: 1, 2: 1, 12: 1}, 'water': {14: 1}}
    backend = CompiledUNIFACBackend.from_model(model, list(groups), groups)
    for x in ([.4, .6], [1.0, 0.0], [1e-10, 1.0-1e-10]):
        reference = model.activity_coefficients(list(groups.values()), x, T)
        if backend is not None:
            assert backend.activity_coefficients(x, T) == pytest.approx(reference, rel=1e-12)
    assert model.activity_coefficients([{14: 1}], [1.0], T) == pytest.approx([1.0])
    h = 1e-5
    plus = np.log(model.activity_coefficients(list(groups.values()), [.4+h, .6-h], T))
    minus = np.log(model.activity_coefficients(list(groups.values()), [.4-h, .6+h], T))
    assert np.dot([.4, .6], (plus-minus)/(2*h)) == pytest.approx(0.0, abs=1e-8)


def test_standalone_factory_and_excess_enthalpy():
    model = create_thermodynamics(['ethanol', 'water'], 'UNIF-LBY')
    assert model.unifac_variant == 'UNIFLBY'
    assert model._compiled_unifac is not None
    x = {'ethanol': .4, 'water': .6}
    T, h = 350., .01
    gp = model.activity_coefficients(T+h, x)
    gm = model.activity_coefficients(T-h, x)
    from physical_constants import R_J_MOL_K
    expected = -R_J_MOL_K*T*T*sum(x[c]*math.log(gp[c]/gm[c])/(2*h) for c in x)
    assert model.excess_enthalpy(x, T) == pytest.approx(expected, rel=1e-6)


def test_lyngby_liquid_phase_split_and_vlle_routing():
    from chemical_properties import ChemicalDatabase
    model = create_thermodynamics(['hexane', 'water'], 'UNIFLBY',
                                  ChemicalDatabase(enable_online=False))
    z = {'hexane': .5, 'water': .5}
    result = model.flash3_TP(z, 298.15, 1.01325)
    assert result.status == 'lle_only'
    assert result.phase_count == 2
    assert result.residual < 1e-6
    g1 = model.activity_coefficients(298.15, result.x1)
    g2 = model.activity_coefficients(298.15, result.x2)
    for c in z:
        assert abs(math.log(result.x1[c]*g1[c]/(result.x2[c]*g2[c]))) < 1e-6


def test_lyngby_numeric_groups_are_not_classic_transport_ids():
    from liquid_mixture_viscosity import _utm_group_key_from_unifac_group
    assert _utm_group_key_from_unifac_group(12, variant='UNIFLBY') == 'std:14'
    assert _utm_group_key_from_unifac_group(10, variant='UNIFLBY') == 'std:9'
