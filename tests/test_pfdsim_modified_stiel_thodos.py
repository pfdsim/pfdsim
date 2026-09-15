import math

import pytest

from pfdsim_modified_stiel_thodos import (
    GROUP_CORRECTIONS,
    ZEROED_CORRECTION_GROUPS,
    PFDSimModifiedStielThodosError,
    classify_molecular_geometry,
    correction_groups,
    evaluate_pfdsim_modified_stiel_thodos,
)
from physical_constants import R_J_MOL_K


def test_monatomic_base_relation_without_group_correction():
    result = evaluate_pfdsim_modified_stiel_thodos(
        temperature_K=300.0,
        viscosity_Pa_s=2.0e-5,
        cv_J_per_mol_K=1.5 * R_J_MOL_K,
        molecular_weight_g_per_mol=4.0,
        critical_temperature_K=5.2,
        geometry="monatomic",
        smiles="[He]",
        formula="He",
    )

    expected = 2.5 * 2.0e-5 * (1.5 * R_J_MOL_K) / 0.004
    assert result.base_value_W_per_m_K == pytest.approx(expected)
    assert result.value_W_per_m_K == pytest.approx(expected)
    assert result.correction_factor == 1.0
    assert result.active_groups == ()
    assert not result.organic_correction_eligible


def test_linear_base_relation_and_geometry_alias():
    temperature = 400.0
    critical_temperature = 200.0
    cv = 2.5 * R_J_MOL_K
    viscosity = 1.5e-5
    molecular_weight = 28.0
    result = evaluate_pfdsim_modified_stiel_thodos(
        temperature,
        viscosity,
        cv,
        molecular_weight,
        critical_temperature,
        "linear",
        "N#N",
        "N2",
    )

    factor = 1.3 + R_J_MOL_K / cv * (1.7614 - 0.3523 / 2.0)
    expected = factor * viscosity * cv / (molecular_weight / 1000.0)
    assert result.base_value_W_per_m_K == pytest.approx(expected)
    assert result.value_W_per_m_K == pytest.approx(expected)
    assert result.reduced_temperature == 2.0


@pytest.mark.parametrize(
    ("formula", "smiles", "expected"),
    (
        ("He", "", "monatomic"),
        ("D2", "", "linear"),
        ("CO2", "O=C=O", "linear"),
        ("H2O", "O", "nonlinear"),
        ("C2H6O", "CCO", "nonlinear"),
    ),
)
def test_geometry_matches_benchmark_topology(formula, smiles, expected):
    geometry, _source = classify_molecular_geometry(formula, smiles)
    assert geometry == expected


def test_geometry_rejects_formula_structure_mismatch():
    with pytest.raises(PFDSimModifiedStielThodosError, match="different element counts"):
        classify_molecular_geometry("C2H6", "CCO")


def test_overlapping_binary_groups_apply_once_each():
    temperature = 500.0
    result = evaluate_pfdsim_modified_stiel_thodos(
        temperature_K=temperature,
        viscosity_Pa_s=2.0e-5,
        cv_J_per_mol_K=60.0,
        molecular_weight_g_per_mol=104.15,
        critical_temperature_K=617.0,
        geometry="non-linear",
        smiles="C=Cc1ccccc1",
        formula="C8H8",
    )

    assert result.active_groups == (
        "alkene",
        "aromatic_ring",
        "hydrocarbon_only",
    )
    exponent = sum(
        GROUP_CORRECTIONS[group].A + GROUP_CORRECTIONS[group].B_K / temperature
        for group in result.active_groups
    )
    assert result.correction_exponent == pytest.approx(exponent)
    assert result.correction_factor == pytest.approx(math.exp(exponent))
    assert result.value_W_per_m_K == pytest.approx(
        result.base_value_W_per_m_K * math.exp(exponent)
    )


@pytest.mark.parametrize(
    ("smiles", "formula", "expected"),
    (
        ("CCC(=O)O", "C3H6O2", "acid_straight_short_C1_C3"),
        ("CCCCCC(=O)O", "C6H12O2", "acid_straight_medium_C4_C6"),
        ("CCCCCCC(=O)O", "C7H14O2", "acid_straight_long_C7_plus"),
        (
            "C=CC(=O)O",
            "C3H4O2",
            "acid_unsaturated_or_aromatic_monocarboxylic",
        ),
    ),
)
def test_active_acid_subclasses(smiles, formula, expected):
    assert expected in correction_groups(smiles, formula)


@pytest.mark.parametrize(
    ("smiles", "formula"),
    (
        ("CC(C)C(=O)O", "C4H8O2"),  # Branched saturated acid.
        ("OC(=O)CC(=O)O", "C3H4O4"),  # Dicarboxylic acid.
        ("CCBr", "C2H5Br"),
        ("CC(=O)OC", "C3H6O2"),
        ("COC", "C2H6O"),
        ("CCF", "C2H5F"),
    ),
)
def test_zeroed_groups_do_not_apply_a_correction(smiles, formula):
    assert correction_groups(smiles, formula) == ()


@pytest.mark.parametrize(
    ("smiles", "formula", "cas"),
    (
        ("Cl", "ClH", "7647-01-0"),
        ("ClCl", "Cl2", "7782-50-5"),
        ("N(F)(F)F", "F3N", "7783-54-2"),
        ("NN", "N2H4", "302-01-2"),
        ("[H]C#N", "CHN", "74-90-8"),
    ),
)
def test_inorganics_are_gated_out_of_all_group_corrections(smiles, formula, cas):
    assert correction_groups(smiles, formula, cas) == ()


def test_organic_chlorine_still_uses_the_chlorine_correction():
    assert correction_groups("CCCl", "C2H5Cl") == ("chlorine",)


def test_condensed_formula_matches_implicit_smiles_hydrogens():
    assert correction_groups("CCO", "C2H5OH") == ("alcohol",)


@pytest.mark.parametrize(
    ("smiles", "formula"),
    (
        ("CCO", "C2H6"),
        ("C", "C2H6"),
        ("CCO", "C2H8O"),
    ),
)
def test_organic_formula_and_smiles_element_counts_must_match(smiles, formula):
    with pytest.raises(PFDSimModifiedStielThodosError, match="different element counts"):
        correction_groups(smiles, formula)


def test_inorganic_gate_does_not_require_smiles():
    assert correction_groups("", "Cl2") == ()


def test_zeroed_group_names_have_no_fitted_coefficients():
    assert ZEROED_CORRECTION_GROUPS.isdisjoint(GROUP_CORRECTIONS)


def test_coefficients_are_immutable():
    with pytest.raises(TypeError):
        GROUP_CORRECTIONS["alcohol"] = GROUP_CORRECTIONS["ketone"]


def test_production_coefficients_have_five_significant_figures_or_fewer():
    for correction in GROUP_CORRECTIONS.values():
        assert float(format(correction.A, ".5g")) == correction.A
        assert float(format(correction.B_K, ".5g")) == correction.B_K


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("temperature_K", 0.0),
        ("viscosity_Pa_s", math.nan),
        ("cv_J_per_mol_K", -1.0),
        ("molecular_weight_g_per_mol", math.inf),
        ("critical_temperature_K", 0.0),
    ),
)
def test_invalid_numeric_inputs_are_rejected(field, value):
    inputs = {
        "temperature_K": 300.0,
        "viscosity_Pa_s": 1.0e-5,
        "cv_J_per_mol_K": 20.0,
        "molecular_weight_g_per_mol": 30.0,
        "critical_temperature_K": 200.0,
        "geometry": "nonlinear",
        "smiles": "C",
        "formula": "CH4",
    }
    inputs[field] = value
    with pytest.raises(PFDSimModifiedStielThodosError, match="positive finite"):
        evaluate_pfdsim_modified_stiel_thodos(**inputs)


def test_invalid_geometry_and_structure_are_rejected():
    inputs = {
        "temperature_K": 300.0,
        "viscosity_Pa_s": 1.0e-5,
        "cv_J_per_mol_K": 20.0,
        "molecular_weight_g_per_mol": 30.0,
        "critical_temperature_K": 200.0,
        "geometry": "spherical",
        "smiles": "C",
        "formula": "CH4",
    }
    with pytest.raises(PFDSimModifiedStielThodosError, match="geometry"):
        evaluate_pfdsim_modified_stiel_thodos(**inputs)

    inputs.update(geometry="nonlinear", smiles="C.C")
    with pytest.raises(PFDSimModifiedStielThodosError, match="connected"):
        evaluate_pfdsim_modified_stiel_thodos(**inputs)
