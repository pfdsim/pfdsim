"""Human table recognition, explicit ambiguity controls and append semantics."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from thermodynamics_models.interaction_fitting import (
    inspect_observations,
    parse_observations,
    normalize_fit_request,
)
from thermodynamics_models.fitting_data import cell_number
from .fitting_import_samples import WIKIPEDIA_TXY, UNLABELED_TXY


def test_wikipedia_multiline_heading_and_missing_pressure():
    proposal = inspect_observations(WIKIPEDIA_TXY, components=["acetone", "water"])
    assert not proposal["ready"]
    assert proposal["needs_review"]
    assert len(proposal["raw_rows"]) == 18
    assert [column["role"] for column in proposal["columns"]] == [
        "temperature",
        "x1",
        "y1",
    ]
    assert proposal["settings"]["temperature_unit"] == "C"
    assert proposal["settings"]["composition_basis"] == "mole_percent"
    assert proposal["reference_component"] == "acetone"
    assert "pressure" not in proposal["settings"]
    assert any("pressure" in issue for issue in proposal["issues"])
    assert proposal["observations"] is None


def test_wikipedia_import_converts_and_preserves_endpoints():
    proposal = inspect_observations(
        WIKIPEDIA_TXY,
        components=["acetone", "water"],
        import_options={"pressure": 1, "pressure_unit": "atm"},
    )
    assert proposal["ready"], proposal["issues"]
    assert len(proposal["observations"]) == 16
    assert len(proposal["excluded"]) == 2
    first = proposal["observations"][0]
    assert first["T_K"] == pytest.approx(360.95)
    assert first["P_bar"] == 1.01325
    assert first["x1"] == 0.01
    assert first["y1"] == 0.335
    assert proposal["original_text"] == WIKIPEDIA_TXY


def test_component_two_is_complemented_without_changing_temperature():
    proposal = inspect_observations(
        WIKIPEDIA_TXY,
        components=["water", "acetone"],
        import_options={"pressure": 1, "pressure_unit": "atm"},
    )
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["x1"] == 0.99
    assert proposal["observations"][0]["y1"] == pytest.approx(0.665)
    mismatched = inspect_observations(
        WIKIPEDIA_TXY, components=["ethanol", "water"], import_options={"pressure": 1}
    )
    assert not mismatched["ready"]
    assert any("component" in issue for issue in mismatched["issues"])


@pytest.mark.parametrize(
    "header,field",
    [
        (r"\(\gamma^\infty_{\mathrm{water\ in\ 1\text{-}butanol}}\)", "gamma2_inf"),
        (r"\(\gamma^\infty_{\mathrm{1\text{-}butanol\ in\ water}}\)", "gamma1_inf"),
    ],
)
def test_named_infinite_dilution_headers_follow_component_order(header, field):
    proposal = inspect_observations(
        f"T (K)\t{header}\n298.15\t5.06\n343.15\t3.27",
        components=["1-butanol", "water"],
    )
    assert proposal["ready"], proposal["issues"]
    assert [column["role"] for column in proposal["columns"]] == ["temperature", field]
    assert all(field in row for row in proposal["observations"])
    other = ({"gamma1_inf", "gamma2_inf"} - {field}).pop()
    assert all(other not in row for row in proposal["observations"])
    if field == "gamma2_inf":
        assert any("named solute" in note for note in proposal["notes"])


@pytest.mark.parametrize("layout", ["original", "reordered", "fractions", "reversed"])
def test_unlabeled_numeric_txy_proposes_roles_and_units(layout):
    rows = [line.split() for line in UNLABELED_TXY.splitlines()]
    expected = ["temperature", "x1", "y1"]
    if layout == "reordered":
        rows = [[row[1], row[2], row[0]] for row in rows]
        expected = ["x1", "y1", "temperature"]
    elif layout == "reversed":
        rows = [[row[0], row[2], row[1]] for row in rows]
        expected = ["temperature", "y1", "x1"]
    elif layout == "fractions":
        rows = [
            [row[0], str(float(row[1]) / 100), str(float(row[2]) / 100)] for row in rows
        ]
    raw = "\n".join(" ".join(row) for row in rows)
    proposal = inspect_observations(
        raw, import_options={"pressure": 1, "pressure_unit": "atm"}
    )
    assert proposal["ready"], proposal["issues"]
    assert proposal["needs_review"]
    assert [column["role"] for column in proposal["columns"]] == expected
    assert proposal["observations"][0]["x1"] == 0.01
    assert proposal["observations"][0]["y1"] == 0.335
    assert any("suggest" in note for note in proposal["notes"])


def test_ambiguous_columns_can_be_mapped_without_reformatting():
    raw = "300 0.2 -200\n320 0.5 -300"
    proposal = inspect_observations(
        raw,
        import_options={
            "kind": "HE",
            "mapping": ["temperature", "x1", "enthalpy"],
            "temperature_unit": "K",
            "enthalpy_unit": "J/mol",
            "composition_basis": "mole_fraction",
            "composition_component": 1,
        },
    )
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][1]["HE_J_mol"] == -300


def test_caption_supplies_pressure_and_book_headers_convert_enthalpy():
    raw = "Table 2: VLE at 760 mmHg\nTemperature (°C)\tLiquid\tVapor\n70\t0.2\t0.6\n60\t0.8\t0.9\nReferences: Smith (2020)."
    proposal = inspect_observations(raw)
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["P_bar"] == pytest.approx(1.01325)
    heat = "Excess enthalpy at T=298.15 K\nMole fraction\tH^E / kJ/mol\n0.2\t0.25\n0.5\t0.40"
    proposal = inspect_observations(heat)
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][1]["T_K"] == 298.15
    assert proposal["observations"][1]["HE_J_mol"] == 400


def test_html_group_headers_and_separate_tables_are_reviewable():
    html = """<table><tr><th rowspan="2">T (°C)</th><th colspan="2">mole percent acetone</th></tr>
<tr><th>liquid</th><th>vapor</th></tr><tr><td>70</td><td>20</td><td>60</td></tr><tr><td>60</td><td>80</td><td>90</td></tr></table>"""
    proposal = inspect_observations(
        html, components=["acetone", "water"], import_options={"pressure": 1}
    )
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["x1"] == 0.2
    both = html + html
    proposal = inspect_observations(
        both, import_options={"table": 1, "pressure": 1, "composition_component": 1}
    )
    assert len(proposal["tables"]) == 2
    assert proposal["table"] == 1


@pytest.mark.parametrize("layout", ["rows", "columns"])
def test_flattened_pdf_numbers_can_be_reassembled(layout):
    rows = [line.split() for line in UNLABELED_TXY.splitlines()]
    values = (
        [cell for row in rows for cell in row]
        if layout == "rows"
        else [row[column] for column in range(3) for row in rows]
    )
    proposal = inspect_observations(
        "\n".join(values), import_options={"pressure": 1, "pressure_unit": "atm"}
    )
    assert proposal["ready"], proposal["issues"]
    assert proposal["layout"] == layout
    assert len(proposal["observations"]) == 16


def test_scientific_notation_uncertainty_and_decimal_comma():
    assert cell_number("1.25 × 10⁻³") == 0.00125
    assert cell_number("−2.5[4]") == -2.5
    assert cell_number("298.15(2)") == 298.15
    assert cell_number("0,25 ± 0,01") == 0.25
    proposal = inspect_observations("T_K;x1;HE_J_mol\n300;0,25;100,5")
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["HE_J_mol"] == 100.5


def test_mass_percent_requires_weights_and_converts_both_phases():
    raw = "T (K)\tLiquid mass percent\tVapor mass percent\n330\t20\t60\n350\t80\t90"
    proposal = inspect_observations(raw, import_options={"pressure": 1})
    assert not proposal["ready"]
    assert any("molecular weights" in issue for issue in proposal["issues"])
    proposal = inspect_observations(
        raw, import_options={"pressure": 1, "molecular_weights": [58, 18]}
    )
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["x1"] == pytest.approx(
        (0.2 / 58) / (0.2 / 58 + 0.8 / 18)
    )


def test_mapping_changes_are_revalidated_and_bad_rows_can_be_excluded():
    raw = "A\tB\tC\n300\t0.2\t200\n320\t0.5\t—"
    options = {
        "kind": "HE",
        "mapping": ["temperature", "x1", "enthalpy"],
        "temperature_unit": "K",
        "enthalpy_unit": "J/mol",
        "composition_basis": "mole_fraction",
        "exclude_rows": [1],
    }
    proposal = inspect_observations(raw, import_options=options)
    assert proposal["ready"], proposal["issues"]
    assert len(proposal["observations"]) == 1
    assert proposal["excluded"][0]["row"] == 1


def test_fit_request_and_typed_json_use_one_normalizer():
    request = normalize_fit_request(
        {
            "components": ["acetone", "water"],
            "observations": WIKIPEDIA_TXY,
            "import_options": {"pressure": 1, "pressure_unit": "atm"},
        }
    )
    assert len(request["observations"]) == 16
    assert request["import_report"][0]["original_text"] == WIKIPEDIA_TXY
    with pytest.raises(ValueError, match="unknown fields"):
        parse_observations(
            json.dumps([{"T_K": 300, "x1": 0.5, "P_bar": 1, "weigth": 2}])
        )


def test_json_matrix_is_a_reviewable_table_instead_of_an_invalid_observation():
    rows = [
        [float(cell) for cell in line.split()] for line in UNLABELED_TXY.splitlines()
    ]
    proposal = inspect_observations(
        json.dumps(rows), import_options={"pressure": 1, "pressure_unit": "atm"}
    )
    assert proposal["ready"], proposal["issues"]
    assert proposal["needs_review"]
    assert len(proposal["observations"]) == 16
    assert json.loads(proposal["original_text"]) == rows
    assert (
        len(
            parse_observations(
                rows, import_options={"pressure": 1, "pressure_unit": "atm"}
            )
        )
        == 16
    )


def test_preview_cli_preserves_missing_conditions_without_fitting(tmp_path, capsys):
    from cli import main

    source = tmp_path / "paste.txt"
    source.write_text(WIKIPEDIA_TXY)
    assert (
        main(
            ["fit", str(source), "--components", "acetone", "water", "--preview-import"]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["observations"] is None
    assert (
        main(
            [
                "fit",
                str(source),
                "--components",
                "acetone",
                "water",
                "--preview-import",
                "--pressure",
                "1",
                "--pressure-unit",
                "atm",
            ]
        )
        == 0
    )
    assert len(json.loads(capsys.readouterr().out)["observations"]) == 16


def test_append_and_bulk_sigma_do_not_mutate_existing_observations(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the shared browser data-model check")
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        root / "static/js/fitting-observations.js",
        tmp_path / "fitting-observations.mjs",
    )
    script = """import assert from 'node:assert/strict';
import {appendObservations, mergeSigma, repairObservationIds} from './fitting-observations.mjs';
const first = [{id:'1',group:'1',source:'',sigma:{HE_J_mol:50},HE_J_mol:200}];
const input = [{id:'1',group:'1',source:'',sigma:{log_gamma:0.1},gamma1_inf:2}];
const added = appendObservations(first,input,{log_gamma:0.02});
assert.equal(added.rows.length,2); assert.equal(added.rows[1].id,'2');
assert.equal(added.rows[1].group,'2'); assert.equal(first.length,1);
assert.equal(first[0].sigma.HE_J_mol,50); assert.equal(input[0].sigma.log_gamma,0.1);
assert.equal(added.rows[1].sigma.log_gamma,0.02);
const repaired = repairObservationIds([{id:'29dbbdae-22ef-45a8-999b-e81208dc40ad',group:'29dbbdae-22ef-45a8-999b-e81208dc40ad',source:''}]);
assert.equal(repaired.rows[0].id,'1'); assert.equal(repaired.rows[0].group,'1');
assert.equal(mergeSigma(2,{HE_J_mol:50}).log_gamma,2);
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
