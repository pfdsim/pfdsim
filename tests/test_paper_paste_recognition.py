"""Paper/website notation must preserve the measured quantity and its units."""

import pytest

from thermodynamics_models.interaction_fitting import inspect_observations
from .fitting_import_samples import OCR_ACETONE_WATER_HE


def test_percentage_uncertainty_does_not_rescale_mole_fractions():
    paste = "VLE at 60 kPa; expanded uncertainties at 95% confidence\nT (K)\tx1\ty1\n342.0\t0.8986\t0.8922\n341.14\t0.7865\t0.7613"
    proposal = inspect_observations(paste)
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["x1"] == 0.8986
    assert proposal["observations"][0]["P_bar"] == 0.6


def test_compact_pdf_header_and_uncertainty_columns():
    # The notation/ordering is used by Barbieri et al., JCT 198 (2024) 107342.
    paste = "P [kPa] T [K] x₁ y₁\n60 342.00 0.8986 0.8922\n60 341.14 0.7865 0.7613"
    proposal = inspect_observations(paste)
    assert proposal["ready"], proposal["issues"]
    assert [column["role"] for column in proposal["columns"]] == [
        "pressure",
        "temperature",
        "x1",
        "y1",
    ]
    assert proposal["observations"][0]["P_bar"] == 0.6
    uncertain = "VLE at 80 kPa\nT/K\tu(T)/K\tx₁ (exp)\ty₁ (exp)\ty₁ (calc)\n349.45\t0.04\t0.9848\t0.9787\t0.97\n349.26\t0.04\t0.9709\t0.9578\t0.95"
    proposal = inspect_observations(uncertain)
    assert proposal["ready"], proposal["issues"]
    assert [column["role"] for column in proposal["columns"]] == [
        "temperature",
        "ignore",
        "x1",
        "y1",
        "ignore",
    ]
    assert proposal["observations"][0]["y1"] == 0.9787


@pytest.mark.parametrize(
    "unit,factor",
    [("J mol⁻¹", 1), ("kJ mol−1", 1000), ("cal/mol", 4.184), ("kcal mol^-1", 4184)],
)
def test_excess_molar_enthalpy_and_energy_units(unit, factor):
    paste = (
        f"Excess molar enthalpy at T=298.15 K\nx₁\tHₘᴱ / {unit}\n0.2\t0.25\n0.5\t0.40"
    )
    proposal = inspect_observations(paste)
    assert proposal["ready"], proposal["issues"]
    assert proposal["settings"]["kind"] == "HE"
    assert proposal["observations"][1]["HE_J_mol"] == pytest.approx(0.4 * factor)


def test_html_keeps_conditions_outside_the_table():
    paste = """<h2>Excess enthalpy at T = 298.15 K</h2>
    <p>Experimental measurements</p><table>
    <tr><th>x<sub>1</sub></th><th>H<sup>E</sup> / J mol<sup>−1</sup></th></tr>
    <tr><td>0.2</td><td>200</td></tr><tr><td>0.5</td><td>400</td></tr></table>"""
    proposal = inspect_observations(paste)
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["T_K"] == 298.15
    assert proposal["observations"][1]["HE_J_mol"] == 400


def test_liquid_phase_primes_and_component_two_compositions():
    paste = "Mutual solubility\nT / K\tx₂′\tx₂″\n300\t0.03\t0.9\n320\t0.04\t0.88"
    proposal = inspect_observations(paste)
    assert proposal["ready"], proposal["issues"]
    assert proposal["settings"]["kind"] == "LLE"
    assert proposal["observations"][0]["x1_alpha"] == pytest.approx(0.1)
    assert proposal["observations"][0]["x1_beta"] == pytest.approx(0.97)


def test_each_column_keeps_its_composition_scale_and_component():
    paste = "VLE at 100 kPa\nT_K\tx₁ / mole %\ty₂ / mole fraction\n330\t20\t0.4\n340\t50\t0.2"
    proposal = inspect_observations(paste)
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["x1"] == 0.2
    assert proposal["observations"][0]["y1"] == 0.6
    assert any("different" in note for note in proposal["notes"])


def test_decimal_comma_space_separated_values_and_pdf_minus():
    paste = "Excess enthalpy at T=298.15 K\nx1 H^E / J mol⁻¹\n0,2 − 200,5\n0,5 - 300,5"
    proposal = inspect_observations(paste)
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["HE_J_mol"] == -200.5


def test_critical_temperature_only_tables_are_not_flattened_into_txy():
    for paste in ("UCST\nT_K\n330\n340", "kind,T_K\nUCST,330\nUCST,340"):
        proposal = inspect_observations(paste)
        assert proposal["ready"], proposal["issues"]
        assert [row["T_K"] for row in proposal["observations"]] == [330, 340]
        assert all(row["kind"] == "UCST" for row in proposal["observations"])


def test_repeated_series_infers_each_energy_unit_from_its_header():
    paste = "x1\tH^E / J mol⁻¹\tH^E / kJ mol⁻¹\n0.2\t200\t0.25\n0.5\t400\t0.45"
    proposal = inspect_observations(
        paste,
        import_options={
            "shared_columns": [0],
            "series": [
                {"columns": [1], "temperature": 300, "temperature_unit": "K"},
                {"columns": [2], "temperature": 320, "temperature_unit": "K"},
            ],
        },
    )
    assert proposal["ready"], proposal["issues"]
    assert [row["HE_J_mol"] for row in proposal["observations"]] == [200, 400, 250, 450]


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 1, 2, 0)])
def test_ocr_headers_use_magnitudes_without_guessing_pressure_units(order):
    headings = ["??", "x?", "v?", "???"]
    rows = [[350, 0.1, 0.3, 80], [355, 0.4, 0.65, 85], [360, 0.8, 0.9, 90]]
    paste = (
        "\t".join(headings[i] for i in order)
        + "\n"
        + "\n".join("\t".join(str(row[i]) for i in order) for row in rows)
    )
    proposal = inspect_observations(paste)
    assert [column["role"] for column in proposal["columns"]] == [
        ["temperature", "x1", "y1", "pressure"][i] for i in order
    ]
    assert not proposal["ready"]
    assert any("pressure unit" in issue for issue in proposal["issues"])
    proposal = inspect_observations(paste, import_options={"pressure_unit": "kpa"})
    assert proposal["ready"], proposal["issues"]
    assert proposal["needs_review"]
    assert proposal["observations"][0]["T_K"] == 350
    assert proposal["observations"][0]["P_bar"] == 0.8


def test_ocr_inference_keeps_recognizable_pressure_and_liquid_headers():
    paste = "P / kPa\tx1\t???\t???\n80\t0.1\t350\t0.3\n85\t0.4\t355\t0.65\n90\t0.8\t360\t0.9"
    proposal = inspect_observations(paste)
    assert proposal["ready"], proposal["issues"]
    assert [column["role"] for column in proposal["columns"]] == [
        "pressure",
        "x1",
        "temperature",
        "y1",
    ]
    assert proposal["observations"][0]["P_bar"] == 0.8


def test_numeric_inference_respects_lle_caption():
    paste = "Mutual solubility (LLE)\n???\t???\t???\n300\t0.02\t0.95\n320\t0.04\t0.9"
    proposal = inspect_observations(paste)
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["kind"] == "LLE"
    assert proposal["observations"][0]["x1_alpha"] == 0.02


@pytest.mark.parametrize(
    "header,expected", [("ln γ₁∞", 2.718281828459045), ("log10 γ₁∞", 10)]
)
def test_logarithmic_activity_coefficients_are_converted(header, expected):
    proposal = inspect_observations(f"T_K\t{header}\n300\t1\n320\t2")
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["gamma1_inf"] == pytest.approx(expected)
    ambiguous = inspect_observations("T_K\tlog γ₁∞\n300\t1\n320\t2")
    assert not ambiguous["ready"]
    assert any("base" in issue for issue in ambiguous["issues"])


def test_sparse_typed_rows_and_mixed_objectives_preserve_metadata():
    critical = inspect_observations(
        "kind,T_K,source,validation_only\nUCST,330,Paper A,false\nLCST,340,Paper B,true"
    )
    assert critical["ready"], critical["issues"]
    assert [row["source"] for row in critical["observations"]] == ["Paper A", "Paper B"]
    assert critical["observations"][1]["validation_only"]
    mixed = inspect_observations(
        "kind,T_K,x1,P_bar,HE_J_mol\nVLE,350,0.3,1,\nHE,300,0.5,,200"
    )
    assert mixed["ready"], mixed["issues"]
    assert [row["kind"] for row in mixed["observations"]] == ["VLE", "HE"]
    assert "P_bar" not in mixed["observations"][1]


def test_ragged_ocr_calorimetry_keeps_heading_and_all_temperature_columns():
    proposal = inspect_observations(OCR_ACETONE_WATER_HE)
    assert proposal["settings"]["kind"] == "HE"
    assert proposal["settings"]["enthalpy_unit"] == "J/mol"
    assert len(proposal["tables"]) == 1
    assert len(proposal["raw_rows"]) == 23
    assert [column["role"] for column in proposal["columns"]] == ["x1"] + [
        "enthalpy"
    ] * 5
    assert [
        hint.get("temperature") for hint in proposal["series_header_hints"][1:]
    ] == [283.15, 298.15, 323.15, 343.15, 363.15]
    assert all(
        hint.get("temperature_unit") == "K"
        for hint in proposal["series_header_hints"][1:]
    )
    assert [row["row"] for row in proposal["ambiguous_rows"]] == [0, 1, 3, 9, 10]
    assert not proposal["ready"]
    assert proposal["original_text"] == OCR_ACETONE_WATER_HE


def he_series_options(temperatures=(283.15, 298.15, 323.15, 343.15, 363.15)):
    return {
        "shared_columns": [0],
        "series": [
            {
                "columns": [index + 1],
                "temperature": temperature,
                "temperature_unit": "K",
            }
            for index, temperature in enumerate(temperatures)
        ],
    }


def test_ragged_he_series_requires_review_before_any_import():
    options = he_series_options()
    blocked = inspect_observations(OCR_ACETONE_WATER_HE, import_options=options)
    assert not blocked["ready"]
    assert blocked["observations"] is None
    assert any("column positions are ambiguous" in issue for issue in blocked["issues"])
    assert all(series["observations"] == 18 for series in blocked["series_reports"])
    options["exclude_rows"] = [0, 1, 3, 9, 10]
    checked = inspect_observations(OCR_ACETONE_WATER_HE, import_options=options)
    assert checked["ready"], checked["issues"]
    assert len(checked["observations"]) == 90
    assert sorted({row["T_K"] for row in checked["observations"]}) == [
        283.15,
        298.15,
        323.15,
        343.15,
        363.15,
    ]
    at_last_temperature = [
        row for row in checked["observations"] if row["T_K"] == 363.15
    ]
    assert (
        next(row for row in at_last_temperature if row["x1"] == 0.151)["HE_J_mol"]
        == -2.2
    )
    assert (
        next(
            row
            for row in checked["observations"]
            if row["T_K"] == 283.15 and row["x1"] == 0.248
        )["HE_J_mol"]
        == 743.9
    )


@pytest.mark.parametrize("format", ["plain", "tsv", "csv", "markdown", "html"])
def test_shared_measurement_headings_and_ragged_rows_across_formats(format):
    heading = ["xI", "h E / J mol - 1"]
    conditions = [
        f"T = {temperature}K"
        for temperature in (283.15, 298.15, 323.15, 343.15, 363.15)
    ]
    rows = [
        ["0.1", "-100", "-80", "-60", "-40", "-20"],
        ["0.2", "-150", "-100"],
        ["0.3", "-200", "-160", "-120", "-80", "-40"],
    ]
    if format == "html":
        paste = (
            "<table><tr><th rowspan='2'>x₁</th><th colspan='5'>Hᴱ / J mol⁻¹</th></tr><tr>"
            + "".join(f"<th>{condition}</th>" for condition in conditions)
            + "</tr>"
            + "".join(
                "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
                for row in rows
            )
            + "</table>"
        )
    else:
        delimiter = {"plain": " ", "tsv": "\t", "csv": ",", "markdown": "|"}[format]
        lines = [heading, conditions, *rows]
        paste = "\n".join(delimiter.join(row) for row in lines)
    proposal = inspect_observations(paste)
    assert proposal["settings"]["kind"] == "HE"
    assert proposal["settings"]["enthalpy_unit"] == "J/mol"
    assert len(proposal["raw_rows"]) == 3
    assert [
        hint.get("temperature") for hint in proposal["series_header_hints"][1:]
    ] == [283.15, 298.15, 323.15, 343.15, 363.15]
    assert [item["row"] for item in proposal["ambiguous_rows"]] == [1]
    options = {**he_series_options(), "exclude_rows": [1]}
    checked = inspect_observations(paste, import_options=options)
    assert checked["ready"], checked["issues"]
    assert len(checked["observations"]) == 10


@pytest.mark.parametrize(
    "heading",
    [
        "T/K 283.15 298.15 323.15 343.15 363.15",
        "T_K\t283.15\t298.15\t323.15\t343.15\t363.15",
    ],
)
def test_one_shared_temperature_unit_and_separate_ocr_quantity_labels(heading):
    paste = (
        "xl\nhE / J mol⁻¹\n"
        + heading
        + "\n0.1 -100 -80 -60 -40 -20\n0.2 -150 -100\n0.3 -200 -160 -120 -80 -40"
    )
    proposal = inspect_observations(paste)
    assert proposal["settings"]["kind"] == "HE"
    assert [
        hint.get("temperature") for hint in proposal["series_header_hints"][1:]
    ] == [283.15, 298.15, 323.15, 343.15, 363.15]
    assert len(proposal["raw_rows"]) == 3


def test_explicit_blank_columns_preserve_known_series_positions():
    paste = "x1\tH^E / J mol⁻¹\nT=283.15K\tT=298.15K\tT=323.15K\tT=343.15K\tT=363.15K\n0.1\t\t100\t\t200\t\n0.3\t-200\t-160\t-120\t-80\t-40"
    proposal = inspect_observations(paste, import_options=he_series_options())
    assert proposal["ready"], proposal["issues"]
    assert not proposal["ambiguous_rows"]
    assert len(proposal["observations"]) == 7
    sparse = [row for row in proposal["observations"] if row["x1"] == 0.1]
    assert [(row["T_K"], row["HE_J_mol"]) for row in sparse] == [
        (298.15, 100),
        (343.15, 200),
    ]


def test_adjacent_decimal_fraction_cells_are_not_merged():
    proposal = inspect_observations("VLE at 1 bar\nT_K x1 y1\n350 0.3 .4\n360 0.5 .6")
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["x1"] == 0.3
    assert proposal["observations"][0]["y1"] == 0.4


def test_temperature_annotations_do_not_change_the_measured_quantity():
    paste = "x1\tH^E / J mol⁻¹ at T = 283.15 K\tH^E / J mol⁻¹ at T = 298.15 K\n0.1\t-100\t-80\n0.2\t-150\n0.3\t-200\t-160"
    proposal = inspect_observations(paste)
    assert proposal["settings"]["kind"] == "HE"
    assert [column["role"] for column in proposal["columns"]] == [
        "x1",
        "enthalpy",
        "enthalpy",
    ]
    assert [
        hint.get("temperature") for hint in proposal["series_header_hints"][1:]
    ] == [283.15, 298.15]
    assert len(proposal["raw_rows"]) == 3
    assert [item["row"] for item in proposal["ambiguous_rows"]] == [1]


@pytest.mark.parametrize("heading,temperatures,unit", [
    ("T = 10 °C T = 25 °C", [10, 25], "C"),
    ("T = 50 °F T = 77 °F", [50, 77], "F"),
    ("T = 283.15 K T = 298.15 K", [283.15, 298.15], "K"),
    ("T / °C 10 25", [10, 25], "C"),
    ("T / °F 50 77", [50, 77], "F"),
    ("T / K 283.15 298.15", [283.15, 298.15], "K"),
])
def test_repeated_temperature_conditions_and_measurement_labels_are_preserved(
    heading, temperatures, unit,
):
    paste = f"x l h E / J mol - 1\n{heading}\n0.1 -100 -80\n0.2 -150\n0.3 -200 -160"
    proposal = inspect_observations(paste)
    assert proposal["settings"]["kind"] == "HE"
    assert [column["role"] for column in proposal["columns"]] == [
        "x1", "enthalpy", "enthalpy",
    ]
    hints = proposal["series_header_hints"][1:]
    assert [
        hint.get("temperature") for hint in hints
    ] == temperatures
    assert [
        hint.get("temperature_unit") for hint in hints
    ] == [unit, unit]
    options = {
        "shared_columns": [0],
        "exclude_rows": [1],
        "series": [
            {"columns": [index], "temperature": hint["temperature"],
             "temperature_unit": hint["temperature_unit"]}
            for index, hint in enumerate(hints, start=1)
        ],
    }
    checked = inspect_observations(paste, import_options=options)
    assert checked["ready"], checked["issues"]
    assert sorted({row["T_K"] for row in checked["observations"]}) == pytest.approx([283.15, 298.15])
    assert [(row["x1"], row["HE_J_mol"]) for row in checked["observations"]] == [
        (0.1, -100), (0.3, -200), (0.1, -80), (0.3, -160),
    ]


def test_roman_liquid_phase_labels_remain_lle_endpoints():
    proposal = inspect_observations(
        "Mutual solubility (LLE)\nT_K\txI\txII\n300\t0.02\t0.95\n320\t0.04\t0.9"
    )
    assert proposal["ready"], proposal["issues"]
    assert [column["role"] for column in proposal["columns"]] == [
        "temperature",
        "x1_alpha",
        "x1_beta",
    ]


@pytest.mark.parametrize(
    "composition", ["xI (mole fraction)", "x_l / mole fraction", "x l"]
)
def test_ocr_composition_labels_with_units_keep_the_shared_axis(composition):
    paste = f"{composition}\th E / J mol - 1\nT=283.15K\tT=298.15K\n0.1\t-100\t-80\n0.2\t-150\n0.3\t-200\t-160"
    proposal = inspect_observations(paste)
    assert proposal["settings"]["kind"] == "HE"
    assert [column["role"] for column in proposal["columns"]] == [
        "x1",
        "enthalpy",
        "enthalpy",
    ]
    assert [
        hint.get("temperature") for hint in proposal["series_header_hints"][1:]
    ] == [283.15, 298.15]
    assert len(proposal["raw_rows"]) == 3
