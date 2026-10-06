"""Explicit projection of wide tables through the shared observation converter."""

import json

import pytest

from thermodynamics_models.interaction_fitting import (
    inspect_observations,
    parse_observations,
)


HE_PASTE = """Acetone (1) + l-propanol (2)
0.058 290.4 316.3 325.6 309.3 283.4
0.110 508.8 561.8 581.0 559.1 517.0
0.152 658.4 731.2 766.4 738.6 692.0
0.202 819.7 898.0 952.0 927.6 884.0
0.256 964.4 1052.3 1115.2 1100.3 1050.7
0.305 1065.2 1166.0 1234.3 1225.1 1178.0
0.352 1143.7 1240.9 1315.4 1312.5 1268.7
0.405 1203.1 1303.5 1376.0 1383.2 1338.4
0.460 1243.7 1342.6 1413.8 1419.9 1384.4
0.508 1251.1 1352.8 1412.3 1425.6 1396.3
0.560 1228.2 1336.4 1383.4 1398.9 1377.4
0.608 1197.3 1301.4 1334.6 1349.3 1331.2
0.663 1131.7 1232.4 1243.0 1261.3 1242.9
0.707 1060.1 1153.5 1144.3 1166.4 1150.2
0.752 965.6 1050.9 1028.6 1046.7 1030.4
0.799 844.6 920.0 878.6 901.7 884.5
0.852 676.7 736.7 677.1 706.9 689.5
0.896 508.8 553.3 491.2 520.5 502.2
0.945 289.3 313.1 271.3 286.7 274.0"""


def he_options():
    # These are explicit synthetic test conditions, not inferred temperatures
    # from the supplied paper paste (which does not state its temperatures).
    return {
        "kind": "HE",
        "mapping": ["x1"] + ["enthalpy"] * 5,
        "shared_columns": [0],
        "temperature_unit": "K",
        "enthalpy_unit": "J/mol",
        "composition_basis": "mole_fraction",
        "series": [
            {
                "name": f"HE curve {index + 1}",
                "columns": [index + 1],
                "temperature": 298.15 + 10 * index,
            }
            for index in range(5)
        ],
    }


def test_original_he_paste_never_expands_without_explicit_series_mode():
    proposal = inspect_observations(HE_PASTE, import_options={"kind": "HE"})
    assert not proposal["ready"]
    assert "series" not in proposal
    assert proposal["series_available"]
    assert len(proposal["raw_rows"]) == 19


@pytest.mark.parametrize("format", ["plain", "tsv", "csv", "markdown", "html", "json"])
def test_supplied_19_by_5_he_table_expands_to_95_exact_observations(format):
    rows = [line.split() for line in HE_PASTE.splitlines()[1:]]
    headers = ["x1"] + [f"HE column {index + 1}" for index in range(5)]
    if format == "plain":
        text = HE_PASTE
    elif format == "tsv":
        text = "\n".join("\t".join(row) for row in rows)
    elif format == "csv":
        text = ",".join(headers) + "\n" + "\n".join(",".join(row) for row in rows)
    elif format == "markdown":
        text = (
            "|"
            + "|".join(headers)
            + "|\n|"
            + "|".join(["---"] * 6)
            + "|\n"
            + "\n".join("|" + "|".join(row) + "|" for row in rows)
        )
    elif format == "html":
        text = (
            "<table><tr>"
            + "".join(f"<th>{cell}</th>" for cell in headers)
            + "</tr>"
            + "".join(
                "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                for row in rows
            )
            + "</table>"
        )
    else:
        text = json.dumps([[float(cell) for cell in row] for row in rows])
    proposal = inspect_observations(text, import_options=he_options())
    assert proposal["ready"], proposal["issues"]
    assert len(proposal["observations"]) == 95
    for series in range(5):
        for row in range(19):
            point = proposal["observations"][series * 19 + row]
            assert point["x1"] == float(rows[row][0])
            assert point["HE_J_mol"] == float(rows[row][series + 1])
            assert point["T_K"] == pytest.approx(298.15 + 10 * series)
            assert proposal["observation_sources"][series * 19 + row]["row"] == row
    assert len({point["id"] for point in proposal["observations"]}) == 95
    assert all(series["observations"] == 19 for series in proposal["series_reports"])


def test_missing_series_temperatures_block_the_whole_import():
    options = he_options()
    options["series"][2].pop("temperature")
    proposal = inspect_observations(HE_PASTE, import_options=options)
    assert not proposal["ready"]
    assert proposal["observations"] is None
    assert any(
        "HE curve 3" in issue and "temperature" in issue for issue in proposal["issues"]
    )


def test_repeated_txy_pairs_have_individual_pressures_and_units():
    raw = "x1\tT (°C)\ty1\tT (K)\ty1\n0.1\t70\t0.6\t363.15\t0.65\n0.3\t65\t0.72\t358.15\t0.77\n0.6\t60\t0.9\t353.15\t0.92"
    options = {
        "kind": "VLE",
        "mapping": ["x1", "temperature", "y1", "temperature", "y1"],
        "shared_columns": [0],
        "composition_basis": "mole_fraction",
        "series": [
            {
                "columns": [1, 2],
                "pressure": 1,
                "pressure_unit": "atm",
                "temperature_unit": "C",
            },
            {
                "columns": [3, 4],
                "pressure": 200,
                "pressure_unit": "kpa",
                "temperature_unit": "K",
                "validation_only": True,
            },
        ],
    }
    proposal = inspect_observations(raw, import_options=options)
    assert proposal["ready"], proposal["issues"]
    assert len(proposal["observations"]) == 6
    assert proposal["observations"][0]["P_bar"] == 1.01325
    assert proposal["observations"][3]["P_bar"] == 2
    assert proposal["observations"][0]["T_K"] == 343.15
    assert proposal["observations"][3]["T_K"] == 363.15
    assert proposal["observations"][3]["validation_only"]
    assert not inspect_observations(raw)["ready"]


def test_nonshared_txy_triples_and_repeated_pxy_use_same_projection():
    triples = "T_K x1 y1 T_K x1 y1\n330 .1 .6 350 .2 .7\n340 .3 .75 360 .4 .85"
    opts = {
        "kind": "VLE",
        "mapping": ["temperature", "x1", "y1"] * 2,
        "temperature_unit": "K",
        "composition_basis": "mole_fraction",
        "series": [
            {"columns": [0, 1, 2], "pressure": 1},
            {"columns": [3, 4, 5], "pressure": 2},
        ],
    }
    assert len(parse_observations(triples, import_options=opts)) == 4
    pxy = "x1 P_kPa y1 P_atm y1\n.1 100 .6 2 .65\n.3 150 .75 3 .8"
    opts = {
        "kind": "VLE",
        "mapping": ["x1", "pressure", "y1", "pressure", "y1"],
        "shared_columns": [0],
        "composition_basis": "mole_fraction",
        "temperature_unit": "K",
        "series": [
            {"columns": [1, 2], "temperature": 300, "pressure_unit": "kpa"},
            {"columns": [3, 4], "temperature": 320, "pressure_unit": "atm"},
        ],
    }
    parsed = parse_observations(pxy, import_options=opts)
    assert parsed[0]["P_bar"] == 1
    assert parsed[2]["P_bar"] == 2.0265


def test_missing_measurement_only_excludes_its_own_series_point():
    raw = "x1 H1 H2\n.1 100 200\n.3 — 400\n.5 300 600"
    options = {
        "kind": "HE",
        "mapping": ["x1", "enthalpy", "enthalpy"],
        "shared_columns": [0],
        "temperature_unit": "K",
        "enthalpy_unit": "J/mol",
        "composition_basis": "mole_fraction",
        "series": [
            {"columns": [1], "temperature": 300},
            {"columns": [2], "temperature": 320},
        ],
    }
    proposal = inspect_observations(raw, import_options=options)
    assert proposal["ready"], proposal["issues"]
    assert len(proposal["observations"]) == 5
    assert [item["observations"] for item in proposal["series_reports"]] == [2, 3]
    assert proposal["series_reports"][0]["excluded"][0]["row"] == 1
    assert any(
        point["x1"] == 0.3 and point["HE_J_mol"] == 400
        for point in proposal["observations"]
    )


def test_shared_source_strings_and_single_measurements_are_not_truncated():
    raw = "source gamma1_inf gamma1_inf\nPaperA 2 3\nPaperB 4 5\nPaperC 6 7"
    opts = {
        "kind": "GAMMA_INF",
        "mapping": ["source", "gamma1_inf", "gamma1_inf"],
        "shared_columns": [0],
        "temperature_unit": "K",
        "series": [
            {"columns": [1], "temperature": 300},
            {"columns": [2], "temperature": 320},
        ],
    }
    points = parse_observations(raw, import_options=opts)
    assert len(points) == 6
    assert [point["source"] for point in points[:3]] == ["PaperA", "PaperB", "PaperC"]


def test_series_assignments_are_validated_and_ids_are_namespaced():
    opts = he_options()
    opts["series"][1]["columns"] = [1]
    with pytest.raises(ValueError, match="own columns"):
        inspect_observations(HE_PASTE, import_options=opts)
    raw = "id x1 HE_J_mol HE_J_mol\nA .1 100 200\nB .2 300 400"
    opts = {
        "kind": "HE",
        "mapping": ["id", "x1", "enthalpy", "enthalpy"],
        "shared_columns": [0, 1],
        "temperature_unit": "K",
        "enthalpy_unit": "J/mol",
        "composition_basis": "mole_fraction",
        "series": [
            {"columns": [2], "temperature": 300},
            {"columns": [3], "temperature": 320},
        ],
    }
    parsed = parse_observations(raw, import_options=opts)
    assert {point["id"] for point in parsed} == {"A.s1", "B.s1", "A.s2", "B.s2"}


def test_leading_numeric_condition_header_is_reviewable_and_preserved():
    raw = "HE / J/mol at temperatures in K\nx1\t298.15\t308.15\n.1\t100\t200\n.2\t300\t400"
    proposal = inspect_observations(raw, import_options={"kind": "HE"})
    assert proposal["series_header_hints"][1].get("condition_value") == 298.15
    opts = {
        "kind": "HE",
        "mapping": ["x1", "enthalpy", "enthalpy"],
        "shared_columns": [0],
        "temperature_unit": "K",
        "enthalpy_unit": "J/mol",
        "composition_basis": "mole_fraction",
        "series": [
            {"columns": [1], "temperature": 298.15},
            {"columns": [2], "temperature": 308.15},
        ],
    }
    parsed = inspect_observations(raw, import_options=opts)
    assert parsed["ready"], parsed["issues"]
    assert len(parsed["observations"]) == 4
    assert parsed["original_text"] == raw


@pytest.mark.parametrize("row", [0, 1])
def test_condition_heading_edits_and_exclusions_use_preview_data_rows(row):
    raw = "HE / J/mol at temperatures in K\nx1\t298.15\t308.15\n.1\t100\t200\n.2\t300\t400"
    options = {
        "kind": "HE", "mapping": ["x1", "enthalpy", "enthalpy"],
        "shared_columns": [0], "temperature_unit": "K",
        "enthalpy_unit": "J/mol", "composition_basis": "mole_fraction",
        "series": [{"columns": [1], "temperature": 298.15},
                   {"columns": [2], "temperature": 308.15}],
    }
    preview = inspect_observations(raw, import_options=options)
    edit = {"row": row, "column": 1, "value": "450"}
    edited = inspect_observations(raw, import_options={**options, "cell_edits": [edit]})
    assert edited["ready"], edited["issues"]
    assert edited["raw_rows"][row][1] == "450"
    assert edited["raw_rows"][1 - row] == preview["raw_rows"][1 - row]
    assert edited["observations"][row]["HE_J_mol"] == 450
    assert edited["cell_edits"] == [edit]
    assert edited["original_text"] == raw
    excluded = inspect_observations(raw, import_options={
        **options, "cell_edits": [edit], "exclude_rows": [1 - row],
    })
    assert excluded["ready"], excluded["issues"]
    assert len(excluded["observations"]) == 2
    assert excluded["observations"][0]["HE_J_mol"] == 450
    assert all(point["x1"] == (0.1 if row == 0 else 0.2) for point in excluded["observations"])
    assert excluded["line_numbers"] == [3, 4]
    assert {source["row"] for source in excluded["observation_sources"]} == {row}


def test_optional_blank_vapor_columns_preserve_later_source_rows():
    raw = "T_K\tx1\ty1\n330\t.1\t\n340\t.3\t.75"
    points = parse_observations(raw, import_options={"pressure": 1})
    assert len(points) == 2
    assert "y1" not in points[0]


def test_series_header_conditions_are_hints_not_automatically_applied():
    raw = (
        "x1\tHE / J/mol at 298.15 K\tHE / J/mol at 308.15 K\n.1\t100\t200\n.3\t300\t400"
    )
    proposal = inspect_observations(raw, import_options={"kind": "HE"})
    assert not proposal["ready"] and "series" not in proposal
    assert proposal["series_header_hints"][1]["temperature"] == 298.15
    assert proposal["series_header_hints"][2]["temperature"] == 308.15


def test_explicit_series_weights_sigma_and_validation_flags_are_preserved():
    opts = he_options()
    opts["series"][3].update(weight=2.5, sigma={"HE_J_mol": 15}, validation_only=True)
    proposal = inspect_observations(HE_PASTE, import_options=opts)
    assert proposal["ready"]
    points = proposal["observations"][3 * 19 : 4 * 19]
    assert all(
        point["weight"] == 2.5
        and point["sigma"] == {"HE_J_mol": 15}
        and point["validation_only"]
        for point in points
    )
