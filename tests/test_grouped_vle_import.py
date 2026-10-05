"""Grouped equilibrium tables retain conditions, measured columns and units."""

import csv
from html import escape
import io

import pytest

from thermodynamics_models.interaction_fitting import (
    inspect_observations,
    parse_observations,
)
from .fitting_import_samples import PRESSURE_GROUPED_TXY


COUNTS = [15, 16, 17, 14, 13]
PRESSURES = [1.013, 0.8, 0.6, 0.4, 0.2]


def grouped_table(
    axis="T",
    *,
    format="csv",
    junk=False,
    captions=None,
    multiline=False,
    reordered=False,
):
    """Make realistic variants of the supplied five-series CSV for shared tests."""
    original = list(csv.reader(io.StringIO(PRESSURE_GROUPED_TXY)))
    captions = captions or (
        [f"Pressure (bar): {p}" for p in PRESSURES]
        if axis == "T"
        else [f"Temperature [K]: {300 + 10 * group}" for group in range(5)]
    )
    pressure_headers = ["P,bar", "P/kPa", "Pressure (atm)", "P [bar]", "p / Pa"]
    order = [1, 0, 2] if reordered else [0, 1, 2]
    rows = [[], []]
    for group in range(5):
        heading = "T,\nK" if multiline else "T,K"
        fields = [heading if axis == "T" else pressure_headers[group], "X1", "Y1"]
        fields = [fields[index] for index in order]
        if junk:
            fields.insert(1, "Author's fitted HE")
        rows[0].extend([captions[group]] + [""] * (len(fields) - 1))
        rows[1].extend(fields)
    for source in original[2:]:
        row = []
        for group in range(5):
            cells = source[group * 3 : group * 3 + 3]
            if axis == "P" and cells[0] != "-":
                pressure = float(cells[0]) / 100
                cells[0] = str(
                    [
                        pressure,
                        pressure * 100,
                        pressure / 1.01325,
                        pressure,
                        pressure * 1e5,
                    ][group]
                )
            cells = [cells[index] for index in order]
            if junk:
                cells.insert(1, "999")
            row.extend(cells)
        rows.append(row)
    if format == "html":
        span = 4 if junk else 3
        # Group captions span the measured and modeled columns in real copied HTML.
        text = (
            "<table><tr>"
            + "".join(
                f'<th colspan="{span}">{escape(caption)}</th>' for caption in captions
            )
            + "</tr>"
        )
        for index, row in enumerate(rows[1:]):
            tag = "th" if index == 0 else "td"
            text += (
                "<tr>"
                + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in row)
                + "</tr>"
            )
        return text + "</table>"
    output = io.StringIO()
    csv.writer(
        output, delimiter={"csv": ",", "tsv": "\t", "semicolon": ";"}[format]
    ).writerows(rows)
    return output.getvalue()


def assert_grouped_points(proposal, axis="T", temperatures=None):
    assert proposal["ready"], proposal["issues"]
    assert proposal["settings"]["kind"] == "VLE"
    assert len(proposal["observations"]) == sum(COUNTS) == 75
    assert [series["observations"] for series in proposal["series_reports"]] == COUNTS
    assert len(proposal["excluded"]) == 10
    source = list(csv.reader(io.StringIO(PRESSURE_GROUPED_TXY)))[2:]
    temperatures = temperatures or [300 + 10 * group for group in range(5)]
    for group, report in enumerate(proposal["series_reports"]):
        for row_index, observation_index in enumerate(report["observation_indices"]):
            point = proposal["observations"][observation_index]
            cells = source[row_index][3 * group : 3 * group + 3]
            assert point["kind"] == "VLE" and "HE_J_mol" not in point
            assert point["x1"] == float(cells[1])
            assert point["y1"] == float(cells[2])
            assert point["T_K"] == pytest.approx(
                float(cells[0]) if axis == "T" else temperatures[group]
            )
            assert point["P_bar"] == pytest.approx(
                PRESSURES[group] if axis == "T" else float(cells[0]) / 100
            )
            assert (
                proposal["observation_sources"][observation_index]["row"] == row_index
            )


@pytest.mark.parametrize("kind", [None, "VLE"])
def test_exact_pressure_grouped_csv_is_vle_with_75_measurements(kind):
    options = {"kind": kind} if kind else None
    proposal = inspect_observations(PRESSURE_GROUPED_TXY, import_options=options)
    assert_grouped_points(proposal)
    assert proposal["series_inferred"] and proposal["needs_review"]
    assert [column["role"] for column in proposal["columns"]] == [
        "temperature",
        "x1",
        "y1",
    ] * 5
    assert [series["pressure"] for series in proposal["series"]] == PRESSURES
    assert all(series["pressure_unit"] == "bar" for series in proposal["series"])
    assert all(
        proposal["columns"][index]["settings"]["temperature_unit"] == "K"
        for index in range(0, 15, 3)
    )
    assert parse_observations(PRESSURE_GROUPED_TXY) == proposal["observations"]
    assert proposal["original_text"] == PRESSURE_GROUPED_TXY


@pytest.mark.parametrize("axis", ["T", "P"])
@pytest.mark.parametrize("format", ["csv", "tsv", "semicolon", "html"])
@pytest.mark.parametrize("junk", [False, True])
def test_both_grouped_layouts_ignore_inserted_fitted_columns(axis, format, junk):
    text = grouped_table(axis, format=format, junk=junk)
    proposal = inspect_observations(text)
    assert_grouped_points(proposal, axis)
    width = 4 if junk else 3
    assert [spec["columns"] for spec in proposal["series"]] == [
        [
            width * group,
            width * group + (2 if junk else 1),
            width * group + (3 if junk else 2),
        ]
        for group in range(5)
    ]
    if junk:
        for index in range(1, 20, 4):
            assert proposal["columns"][index]["role"] == "ignore"
            assert proposal["columns"][index]["reason"] == "calculated column"
        assert any(
            "Unassigned columns are ignored" in note for note in proposal["notes"]
        )


@pytest.mark.parametrize("axis", ["T", "P"])
def test_reordered_measured_columns_do_not_change_groups(axis):
    proposal = inspect_observations(grouped_table(axis, junk=True, reordered=True))
    assert_grouped_points(proposal, axis)


@pytest.mark.parametrize("format", ["csv", "html"])
def test_group_captions_do_not_hide_component_indices(format):
    text = (
        grouped_table(format=format, reordered=True)
        .replace("X1", "X_2")
        .replace("Y1", "Y2")
    )
    proposal = inspect_observations(text)
    assert proposal["ready"], proposal["issues"]
    assert len(proposal["observations"]) == 75
    assert proposal["observations"][0]["x1"] == pytest.approx(1 - 0.981)
    assert proposal["observations"][0]["y1"] == pytest.approx(1 - 0.984)


@pytest.mark.parametrize(
    "caption",
    [
        "P={value} bar",
        "Pressure (bar): {value}",
        "P / bar = {value}",
        "p,bar={value}",
        "P = {value}bar",
        "P ~ {value} bar",
        "Ｐ＝{value}ｂａｒ",
        "{value} bar",
        "Pressure: [bar] = {value}",
        "P={value}[bar]",
    ],
)
def test_awkward_pressure_condition_headers(caption):
    proposal = inspect_observations(
        grouped_table(captions=[caption.format(value=value) for value in PRESSURES])
    )
    assert_grouped_points(proposal)


@pytest.mark.parametrize(
    "caption",
    [
        "T={value} K",
        "Temperature [K]: {value}",
        "T/K = {value}",
        "t, K = {value}",
        "T ( K ) : {value}",
        "Ｔ＝{value}Ｋ",
        "{value} K",
        "{value} kelvin",
        "Temperature: [K] = {value}",
        "T={value}[K]",
        "{value} [K]",
    ],
)
def test_awkward_temperature_condition_headers(caption):
    proposal = inspect_observations(
        grouped_table(
            "P", captions=[caption.format(value=300 + 10 * group) for group in range(5)]
        )
    )
    assert_grouped_points(proposal, "P")


def test_celsius_and_fahrenheit_condition_units_are_converted():
    captions = ["T=25 °C", "Temperature / K = 310", "T=116.33 °F", "T,C=55", "T,K=340"]
    proposal = inspect_observations(grouped_table("P", captions=captions))
    assert_grouped_points(proposal, "P", [298.15, 310, 320, 328.15, 340])


@pytest.mark.parametrize("format", ["csv", "tsv", "semicolon"])
def test_quoted_multiline_measurement_headers(format):
    text = grouped_table(format=format, multiline=True)
    proposal = inspect_observations(text)
    assert_grouped_points(proposal)
    assert (
        proposal["line_numbers"][0] == 8
    )  # The five quoted temperature headers span six physical lines.


def test_explicit_kelvin_is_not_reinterpreted_as_celsius_from_magnitude():
    text = 'P=1 bar,,,P=.8 bar,,\n"T,K",X1,Y1,"T,K",X1,Y1\n150,.2,.4,160,.3,.5'
    proposal = inspect_observations(text)
    assert proposal["ready"], proposal["issues"]
    assert [row["T_K"] for row in proposal["observations"]] == [150, 160]


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Temperature [K]", [150, 160]),
        ("Temp_K", [150, 160]),
        ("T,[K]", [150, 160]),
        ("T,°K", [150, 160]),
        ("Temperature (C)", [423.15, 433.15]),
        ("Temp_F", [(150 - 32) * 5 / 9 + 273.15, (160 - 32) * 5 / 9 + 273.15]),
    ],
)
def test_measured_temperature_units_are_explicit_even_with_awkward_labels(
    label, expected
):
    output = io.StringIO()
    csv.writer(output).writerows(
        [
            ["P=1 bar", "", "", "P=.8 bar", "", ""],
            [label, "X1", "Y1", label, "X1", "Y1"],
            [150, 0.2, 0.4, 160, 0.3, 0.5],
        ]
    )
    proposal = inspect_observations(output.getvalue())
    assert proposal["ready"], proposal["issues"]
    assert [row["T_K"] for row in proposal["observations"]] == pytest.approx(expected)


@pytest.mark.parametrize("axis", ["T", "P"])
def test_unknown_condition_units_require_review_instead_of_guessing(axis):
    symbol = "P" if axis == "T" else "T"
    values = PRESSURES if axis == "T" else [300 + 10 * group for group in range(5)]
    proposal = inspect_observations(
        grouped_table(axis, captions=[f"{symbol}={value}" for value in values])
    )
    assert not proposal["ready"]
    assert any("unit" in issue.lower() for issue in proposal["issues"])


@pytest.mark.parametrize("axis", ["T", "P"])
def test_explicit_common_units_resolve_unitless_group_conditions(axis):
    symbol = "P" if axis == "T" else "T"
    values = PRESSURES if axis == "T" else [300 + 10 * group for group in range(5)]
    key = "pressure_unit" if axis == "T" else "temperature_unit"
    proposal = inspect_observations(
        grouped_table(axis, captions=[f"{symbol}={value}" for value in values]),
        import_options={key: "bar" if axis == "T" else "K"},
    )
    assert_grouped_points(proposal, axis)


def test_missing_pressure_for_one_group_does_not_borrow_another_groups_value():
    captions = [f"P={value} bar" for value in PRESSURES]
    captions[2] = "Series 3"
    proposal = inspect_observations(grouped_table(captions=captions))
    assert not proposal["ready"]
    assert any(
        "Series 3" in issue and "pressure" in issue for issue in proposal["issues"]
    )
    assert "pressure" not in proposal["series"][2]


def test_conflicting_condition_units_require_review():
    proposal = inspect_observations(grouped_table(captions=["P (kPa)=1 bar"] * 5))
    assert not proposal["ready"]
    assert all("pressure" not in spec for spec in proposal["series"])


@pytest.mark.parametrize(
    "table",
    [
        "T_K x1 HE_J_mol\n300 .2 20",
        "T_K gamma1_inf\n300 2",
    ],
)
def test_irrelevant_pressure_caption_does_not_require_pressure_units(table):
    proposal = inspect_observations("P=1\n" + table)
    assert proposal["ready"], proposal["issues"]
    assert "P_bar" not in proposal["observations"][0]


@pytest.mark.parametrize("value", [None, "yes", 1])
def test_infer_series_requires_a_boolean(value):
    with pytest.raises(ValueError, match="infer_series must be true or false"):
        inspect_observations(
            PRESSURE_GROUPED_TXY, import_options={"infer_series": value}
        )


@pytest.mark.parametrize(
    "junk", ["Y1 fitted", "HE_model", "Y1calc", "HE_std", "predicted y1"]
)
def test_modeled_and_uncertainty_headers_are_ignored_without_relabeling_vle(junk):
    text = grouped_table(junk=True).replace("Author's fitted HE", junk)
    assert_grouped_points(inspect_observations(text))


def test_group_caption_can_mark_a_column_as_fitted_even_with_a_measured_leaf_label():
    text = """<table><tr><th colspan="3">P=1 bar</th><th>Author fitted</th><th colspan="3">P=.8 bar</th></tr>
<tr><th>T,K</th><th>X1</th><th>Y1</th><th>Y1</th><th>T,K</th><th>X1</th><th>Y1</th></tr>
<tr><td>330</td><td>.2</td><td>.4</td><td>999</td><td>320</td><td>.3</td><td>.5</td></tr></table>"""
    proposal = inspect_observations(text)
    assert proposal["ready"], proposal["issues"]
    assert proposal["columns"][3]["role"] == "ignore"
    assert [spec["columns"] for spec in proposal["series"]] == [[0, 1, 2], [4, 5, 6]]


def test_single_series_mode_can_be_selected_explicitly_and_edits_are_preserved():
    proposal = inspect_observations(
        PRESSURE_GROUPED_TXY, import_options={"infer_series": False}
    )
    assert not proposal["ready"] and "series" not in proposal
    assert proposal["series_suggestion"]["layout"] == "TxyGroups"
    mapping = ["temperature", "x1", "y1"] + ["ignore"] * 12
    selected = inspect_observations(
        PRESSURE_GROUPED_TXY,
        import_options={
            "infer_series": False,
            "mapping": mapping,
            "pressure": 1.013,
            "exclude_rows": [15, 16],
            "cell_edits": [{"row": 0, "column": 0, "value": "330.85"}],
        },
    )
    assert selected["ready"], selected["issues"]
    assert len(selected["observations"]) == 15
    assert selected["observations"][0]["T_K"] == 330.85


def test_measured_pressure_units_survive_explicit_repeated_series_refresh():
    text = grouped_table("P", junk=True)
    initial = inspect_observations(text)
    refreshed = inspect_observations(
        text,
        import_options={
            "kind": "VLE",
            "series": initial["series"],
            "mapping": [column["role"] for column in initial["columns"]],
            "shared_columns": [],
        },
    )
    assert_grouped_points(refreshed, "P")
    assert [
        column["settings"].get("pressure_unit")
        for column in refreshed["columns"]
        if column["role"] == "pressure"
    ] == ["bar", "kpa", "atm", "bar", "pa"]
