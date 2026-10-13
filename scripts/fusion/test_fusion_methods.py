"""Source-example and benchmark-integrity checks, kept with the scripts."""

import json
import math
from pathlib import Path

import pytest

from benchmark_fusion import source_descriptors, summarize
from fusion_methods import NotApplicable, acid_estimate, chickos, flexibility
from prepare_data import extract_chickos

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize('smiles,T,expected', [
    ('c1ccc2sccc2c1', 304.5, 13.05),
    ('OCc1ccco1', 258.52, 12.22),
    ('C/C=C\\C=O', 158.38, 9.27),
])
def test_perry_worked_fusion_examples(smiles, T, expected):
    assert chickos(smiles, updated=False).enthalpy_kJ_mol(T) == pytest.approx(expected, abs=.01)


@pytest.mark.parametrize('smiles,expected', [
    ('C=Cc1ccccc1', 17.3 + 5.3 - 7.5 + 5 * 7.4),
    ('C=CCCCCC', 17.3 + 5.3 + 4 * 7.1 * 1.31 + 17.6),
    ('c1ccc2cnccc2c1', 7 * 7.4 + 2 * -7.5 + 10.9),
    ('FC(F)(F)C#N', -34.8 * .66 + 3 * 13.3 + 17.7),
    ('CC(=O)NC(C)C(N)=O', 2 * 17.6 + -16.4 * .60 + 1.5 + 27.9),
    ('Cn1cccc1', 33.4 + 2 * 3.7 + 17.6 + 2 * -1.6 + 2 * -1.6 * 1.92 - 19.3),
])
def test_updated_group_arithmetic_from_paper(smiles, expected):
    assert chickos(smiles).entropy_J_mol_K == pytest.approx(expected)


def test_formates_use_the_specific_published_group():
    # Table 2(a), A37 is formate (4.2), distinct from ordinary ester A38 (7.7).
    assert chickos('CCOC=O').entropy_J_mol_K == pytest.approx(17.6 + 7.1 + 4.2)
    assert chickos('CCOC(C)=O').entropy_J_mol_K == pytest.approx(2 * 17.6 + 7.1 + 7.7)
    with pytest.raises(NotApplicable, match='Perry.*formate'):
        chickos('CCOC=O', updated=False)


@pytest.mark.parametrize('smiles,expected', [
    ('CCCC[N+](=O)[O-]', 17.6 + 3 * 7.1 * 1.31 + 17.7),
    ('CCCCS(=O)(=O)C', 2 * 17.6 + 3 * 7.1 * 1.31 + .3),
    ('CCCCOC', 2 * 17.6 + 3 * 7.1 * 1.31 + 4.71),
])
def test_methylene_comparison_counts_functional_groups_not_their_atoms(smiles, expected):
    assert chickos(smiles).entropy_J_mol_K == pytest.approx(expected)


def test_methylene_comparison_retains_each_cyclic_functional_group_ring_atom():
    # A five-CH2 substituent is shorter than a six-membered lactone ring;
    # collapsing the ring's ester into one group would incorrectly apply 1.31.
    estimate = chickos('CCCCCCC1CCCC(=O)O1')
    chain = [g for g in estimate.details['groups'] if g['group'] == 'sp3_2']
    assert len(chain) == 5
    assert all(g['coefficient'] == 1 for g in chain)


@pytest.mark.parametrize('smiles,sp3,sp2,rings,tau', [
    ('CCCCC', 3, 0, 0, 2), ('CC(C)CC', 2, 0, 0, 1),
    ('CC(C)(C)C', 1, 0, 0, 0), ('CCCCCO', 4, 0, 0, 3),
    ('CCC(=O)CC', 2, 1, 0, 1.5), ('C1CCCC1', 0, 0, 1, 0),
    ('Cc1ccccc1', 0, 0, 1, 0), ('CC(C)(C)c1ccccc1', 0, 0, 1, 0),
    ('CCCCc1ccccc1', 3, 0, 1, 2.5),
    ('O=C(O)c1ccccc1', 0, 1, 1, 0),
])
def test_jain_published_flexibility_examples(smiles, sp3, sp2, rings, tau):
    assert flexibility(smiles) == {'sp3': sp3, 'sp2': sp2, 'ring_systems': rings, 'tau': tau}


@pytest.mark.parametrize('smiles,T,expected', [
    ('O=C(O)CO', 351.3, 14.9),
    ('O=C(O)CCC(=O)O', 458, 33.7),
    ('O=C(O)C(O)CC(=O)O', 403, 32.9),
    ('O=C(O)CC(=O)O', 405.9, 19.6),
])
def test_acid_supplement_predictions(smiles, T, expected):
    result = acid_estimate(smiles, use_entropy=False, melting_temperature_K=T)
    assert result.enthalpy_kJ_mol(T) == pytest.approx(expected, abs=.06)


def test_unsupported_functionality_is_not_silently_grouped():
    with pytest.raises(NotApplicable, match='carbonate'):
        chickos('COC(=O)OC')
    with pytest.raises(NotApplicable, match='cyclic'):
        chickos('O=C1NCCC1', updated=False)
    with pytest.raises(NotApplicable, match='composite'):
        chickos('CON')
    with pytest.raises(NotApplicable, match='disconnected'):
        chickos('CC(=O)[O-].[Na+]')


def test_stored_table_preserves_final_fusion_separately():
    rows = json.loads((HERE / 'data/chickos_1999_records.json').read_text())['records']
    case = next(r for r in rows if r['name'] == 'bromotrichloromethane')
    assert case['fusion_enthalpy_kJ_mol'] == 2.03
    assert case['total_enthalpy_kJ_mol'] == 7.2
    assert case['has_solid_solid_transition']
    assert len(case['transitions']) == 3
    assert not case['numeric_audit_flags']


def test_descriptor_join_never_selects_on_experimental_entropy():
    rows = [{'cas': 'x', 'sigma': 2, 'tau': 0, 'source_row': 1},
            {'cas': 'x', 'sigma': 3, 'tau': 0, 'source_row': 2}]
    assert source_descriptors(rows) == {}


def test_summary_keeps_signed_bias_and_absolute_errors():
    result = summarize([(10, 12), (20, 16)])
    assert result['mae'] == 3
    assert result['rmse'] == pytest.approx(math.sqrt(10))
    assert result['mape_percent'] == 20
    assert result['bias_percent'] == 0


def test_extractor_rejects_unexpected_source_shape():
    with pytest.raises(RuntimeError, match='unexpectedly small'):
        extract_chickos('not the source table')


def test_original_supplement_recovers_either_missing_numeric_column():
    rows = json.loads((HERE / 'data/dannenfelser_1996_records.json').read_text())['records']
    trichloro = next(r for r in rows if '1,1,1-trichloroethane' in r['name'] and r['source_page'] == 5)
    assert trichloro['total_entropy_J_mol_K'] == 42.2
    assert trichloro['published_prediction_entropy_J_mol_K'] == pytest.approx(40.9)
    diamino = next(r for r in rows if r['name'] == '1,2-diaminoethane'
                   and 'missing column 0' in r['prediction_origin'])
    assert diamino['total_entropy_J_mol_K'] == pytest.approx(82.0)
    assert diamino['published_prediction_entropy_J_mol_K'] == 44.2


def test_stored_replays_respect_source_rounding():
    rows = json.loads((HERE / 'data/dannenfelser_1996_records.json').read_text())['records']
    assert max(abs(r['rounded_descriptor_entropy_J_mol_K'] - r['published_prediction_entropy_J_mol_K'])
               for r in rows) < .5
