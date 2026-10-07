"""Missing measurements, PDF sign alignment, cell corrections and dimensions."""

import json

import numpy as np
import pytest

from thermodynamics_models.interaction_fitting import (
    fitting_catalog,
    fit_interactions,
    inspect_observations,
    parse_observations,
    prepare_fit,
)


MARKERS = [
    "",
    " ",
    "?",
    "??",
    "-",
    "--",
    "—",
    "–",
    "−",
    "..",
    "...",
    "…",
    "None",
    "NULL",
    "nil",
    "NA",
    "n/a",
    "N.A.",
    "NaN",
    "<NA>",
    "#N/A",
    "ND",
    "n.d.",
    "missing",
    "unknown",
    "unavailable",
    "not available",
    "not measured",
    "not reported",
]


@pytest.mark.parametrize("marker", [None, *MARKERS])
def test_structured_missing_values_are_omitted_without_losing_metadata(marker):
    row = {
        "kind": "VLE",
        "T_K": 300,
        "P_bar": 1,
        "x1": 0.2,
        "y1": marker,
        "weight": marker,
        "pin_tolerance": marker,
        "sigma": marker,
        "source": "None",
        "group": "unknown",
    }
    result = parse_observations([row])[0]
    assert "y1" not in result
    assert result["weight"] == 1
    assert result["pin_tolerance"] == 1e-5
    assert result["sigma"] == {}
    assert result["source"] == "None" and result["group"] == "unknown"
    assert "y1" in row  # Normalization must not rewrite the caller's data.


@pytest.mark.parametrize("marker", MARKERS)
@pytest.mark.parametrize("format", ["csv", "tsv", "html", "matrix"])
def test_missing_cells_in_import_formats(marker, format):
    headers = ["kind", "T_K", "P_bar", "x1", "y1"]
    cells = ["VLE", "300", "1", ".2", marker]
    if format == "html":
        text = "<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
        # Escape marker text so <NA> remains cell content.
        from html import escape

        text += (
            "<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr></table>"
        )
    elif format == "matrix":
        text = {"headers": headers, "rows": [cells]}
    else:
        delimiter = "," if format == "csv" else "\t"
        text = delimiter.join(headers) + "\n" + delimiter.join(cells)
    proposal = inspect_observations(text)
    assert proposal["ready"], proposal["issues"]
    assert len(proposal["observations"]) == 1
    assert "y1" not in proposal["observations"][0]
    assert proposal["raw_rows"][0][-1] == marker.strip()


@pytest.mark.parametrize("marker", [None, *MARKERS])
@pytest.mark.parametrize("field", ["T_K", "P_bar", "x1"])
def test_required_conditions_remain_required(marker, field):
    row = {"kind": "VLE", "T_K": 300, "P_bar": 1, "x1": 0.2, field: marker}
    with pytest.raises(ValueError, match="required|missing"):
        parse_observations([row])


@pytest.mark.parametrize("marker", ["wrong", "Infinity", float("inf"), float("nan")])
def test_bad_numbers_are_not_silently_discarded(marker):
    with pytest.raises(ValueError, match="finite"):
        parse_observations(
            [{"kind": "VLE", "T_K": 300, "P_bar": 1, "x1": 0.2, "y1": marker}]
        )


def test_zero_and_negative_measurements_are_not_missing():
    rows = parse_observations(
        [
            {"kind": "HE", "T_K": 300, "x1": 0.2, "HE_J_mol": 0, "weight": 0},
            {"kind": "HE", "T_K": 300, "x1": 0.3, "HE_J_mol": -20},
        ]
    )
    assert rows[0]["HE_J_mol"] == rows[0]["weight"] == 0
    assert rows[1]["HE_J_mol"] == -20


@pytest.mark.parametrize(
    "row,optional",
    [
        ({"kind": "GAMMA_INF", "T_K": 300, "gamma2_inf": 2}, "gamma1_inf"),
        ({"kind": "UCST", "T_K": 300}, "x1"),
        ({"kind": "LCST", "T_K": 300}, "x1"),
        ({"kind": "AZEOTROPE", "T_K": 300, "P_bar": 1, "x1": 0.2}, "y1"),
        ({"kind": "LLE", "T_K": 300, "x1_alpha": 0.1, "x1_beta": 0.9}, "P_bar"),
        ({"kind": "VLLE", "T_K": 300, "P_bar": 1}, "y1"),
    ],
)
def test_missing_optional_measurements_across_observation_kinds(row, optional):
    assert optional not in parse_observations([{**row, optional: "None"}])[0]


def test_missing_vlle_endpoints_preserve_the_pair_requirement():
    row = {"kind": "VLLE", "T_K": 300, "P_bar": 1, "x1_alpha": "?", "x1_beta": None}
    assert "x1_alpha" not in parse_observations([row])[0]
    with pytest.raises(ValueError, match="both liquid endpoints"):
        parse_observations([{**row, "x1_alpha": 0.1}])
    with pytest.raises(ValueError, match="GAMMA_INF needs"):
        parse_observations(
            [{"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": "?", "gamma2_inf": None}]
        )


@pytest.mark.parametrize("marker", [None, *MARKERS])
@pytest.mark.parametrize("known,unknown,value", [
    ("x1_alpha", "x1_beta", 0.8),
    ("x1_beta", "x1_alpha", 0.1),
])
def test_lle_accepts_either_endpoint_without_relabeling_it(marker, known, unknown, value):
    original = {"kind": "LLE", "T_K": 300, known: value, unknown: marker}
    row = parse_observations([original])[0]
    assert row[known] == value
    assert unknown not in row
    assert unknown in original


@pytest.mark.parametrize("endpoint", ["x1_alpha", "x1_beta"])
@pytest.mark.parametrize("value", [0, 1, -0.1, 1.1, float("nan"), float("inf")])
def test_one_sided_lle_rejects_invalid_endpoints(endpoint, value):
    with pytest.raises(ValueError, match="finite|strictly between"):
        parse_observations([{"kind": "LLE", "T_K": 300, endpoint: value}])


def test_lle_still_needs_a_temperature_and_one_measured_endpoint():
    with pytest.raises(ValueError, match="LLE needs"):
        parse_observations([{"kind": "LLE", "T_K": 300, "x1_alpha": "—", "x1_beta": None}])
    with pytest.raises(ValueError, match="required"):
        parse_observations([{"kind": "LLE", "x1_beta": 0.9}])


@pytest.mark.parametrize("endpoint", ["x1_alpha", "x1_beta"])
def test_lle_import_accepts_one_endpoint_column(endpoint):
    proposal = inspect_observations(f"kind,T_K,{endpoint}\nLLE,300,.2\nLLE,310,.3")
    assert proposal["ready"], proposal["issues"]
    assert len(proposal["observations"]) == 2
    assert proposal["observations"][0][endpoint] == 0.2


def test_sparse_lle_repeated_series_keep_one_sided_rows():
    proposal = inspect_observations(
        "T_K\tA\tB\n300\t.1\t—\n310\t?\t.8\n320\t—\t—",
        import_options={
            "kind": "LLE", "mapping": ["temperature", "x1_alpha", "x1_beta"],
            "shared_columns": [0], "temperature_unit": "K",
            "composition_basis": "mole_fraction", "composition_component": 1,
            "series": [{"columns": [1, 2]}],
        },
    )
    assert proposal["ready"], proposal["issues"]
    assert len(proposal["observations"]) == 2
    assert "x1_beta" not in proposal["observations"][0]
    assert "x1_alpha" not in proposal["observations"][1]
    assert len(proposal["series_reports"][0]["excluded"]) == 1


def test_one_sided_lle_residuals_do_not_reward_collapsed_ideal_phases():
    problem = prepare_fit({
        "components": ["ethanol", "water"], "form": "constant",
        "observations": [{"kind": "LLE", "T_K": 300, "x1_alpha": 0.2}],
    })
    values = problem.initial.copy()
    index = problem.lle_names["1"]
    values[index] = problem.lower[index]
    scaled, _, prediction = problem.row_errors(values, problem.request["observations"][0])
    assert prediction["evaluation_x1_beta"] > prediction["evaluation_x1_alpha"]
    # An ideal liquid has no coexistence: even a tiny latent gap must retain
    # the nonzero curvature limit of the log-activity residuals.
    assert np.linalg.norm(scaled[:2]) > 100


def test_fitted_lle_endpoint_crossing_a_tangent_grid_node_keeps_residual_size():
    problem = prepare_fit({
        "components": ["ethanol", "water"], "form": "constant",
        "observations": [{"kind": "LLE", "T_K": 300, "x1_alpha": 0.2}],
    })
    lengths = []
    for endpoint in (0.5, 0.50001):
        values = problem.initial.copy()
        values[problem.lle_names["1"]] = (endpoint - 0.2) / 0.8
        errors, _, _ = problem.row_errors(values, problem.request["observations"][0])
        lengths.append(len(errors))
    assert lengths == [45, 45]


@pytest.mark.parametrize("model,coefficients", [("NRTL", [2.5, 2.5]), ("UNIQUAC", [-2.5, -2.5])])
def test_one_sided_lle_fit_and_validation_use_stable_runtime_endpoints(model, coefficients):
    reference = prepare_fit({
        "components": ["ethanol", "water"], "model": model, "form": "constant",
        "observations": [{"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 1}],
    })
    reference.install(np.array(coefficients))
    truth = reference.predict_lle(300)
    a, b = truth["x1_alpha"], truth["x1_beta"]
    request = {
        "components": ["ethanol", "water"], "model": model, "form": "constant", "starts": 1,
        "initial": {"12.constant": coefficients[0] - 0.1, "21.constant": coefficients[1] + 0.1},
        "observations": [
            {"kind": "LLE", "T_K": 300, "x1_alpha": a, "x1_beta": None},
            {"kind": "LLE", "T_K": 300, "x1_alpha": "—", "x1_beta": b},
            {"kind": "LLE", "T_K": 300, "x1_alpha": a, "validation_only": True},
            {"kind": "LLE", "T_K": 300, "x1_beta": b, "validation_only": True},
        ],
    }
    problem = prepare_fit(request)
    assert set(problem.lle_names) == {"1", "2"}
    result = fit_interactions(request)
    assert result["success"], result["warnings"]
    assert result["optimizer"]["objective"] < 1e-7
    for point in result["points"]:
        assert point["physical"]
        assert point["predicted"]["split"]
        assert point["predicted"]["x1_alpha"] == pytest.approx(a, abs=1e-5)
        assert point["predicted"]["x1_beta"] == pytest.approx(b, abs=1e-5)
    for metrics in (result["physical_metrics"], result["validation_only"]["physical_metrics"]):
        assert metrics["LLE"]["x1_alpha"]["n"] == 1
        assert metrics["LLE"]["x1_beta"]["n"] == 1


def test_one_sided_lle_cross_validation_keeps_latent_coordinates_in_training_only():
    reference = prepare_fit({
        "components": ["ethanol", "water"], "form": "constant",
        "observations": [{"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 1}],
    })
    reference.install(np.array([2.5, 2.5]))
    state = reference.predict_lle(300)
    a, b = state["x1_alpha"], state["x1_beta"]
    result = fit_interactions({
        "components": ["ethanol", "water"], "form": "constant", "starts": 1,
        "cv": {"method": "kfold", "folds": 3},
        "initial": {
            "12.constant": 2.5, "21.constant": 2.5,
            "lle_gap.lower": (b - a) / (1 - a), "lle_gap.upper": (b - a) / b,
        },
        "bounds": {"lle_gap.lower": [1e-6, .999999], "lle_gap.upper": [1e-6, .999999]},
        "observations": [
            {"id": "anchor", "kind": "LLE", "T_K": 300, "x1_alpha": a, "x1_beta": b},
            {"id": "lower", "kind": "LLE", "T_K": 300, "x1_alpha": a},
            {"id": "upper", "kind": "LLE", "T_K": 300, "x1_beta": b},
        ],
    })
    assert result["success"], result["warnings"]
    for fold in result["cross_validation"]["folds"]:
        assert fold["success"], fold
        expected_latent = len(set(fold["training_ids"]) & {"lower", "upper"})
        assert fold["parameter_count"] == 2 + expected_latent
        assert len(fold["held_out_ids"]) == 1
        point = fold["points"][0]
        assert point["physical"]
        assert point["predicted"]["x1_alpha"] == pytest.approx(a, abs=1e-5)
        assert point["predicted"]["x1_beta"] == pytest.approx(b, abs=1e-5)


def test_an_entire_missing_data_row_is_retained_for_correction():
    proposal = inspect_observations("T_K\tP_bar\tx1\ty1\nNone\t?\tNA\tnull")
    assert not proposal["ready"]
    assert proposal["raw_rows"] == [["None", "?", "NA", "null"]]


@pytest.mark.parametrize(
    "marker", ["?", "None", "unknown", "not measured", "NOT  REPORTED"]
)
def test_whitespace_tables_keep_missing_marker_positions(marker):
    proposal = inspect_observations(
        f"T_K gamma1_inf gamma2_inf\n300 {marker} 2\n310 3 4"
    )
    assert proposal["ready"], proposal["issues"]
    assert "gamma1_inf" not in proposal["observations"][0]
    assert proposal["observations"][1]["gamma2_inf"] == 4


@pytest.mark.parametrize("marker", MARKERS)
def test_sparse_repeated_series_only_omit_the_missing_measurement(marker):
    text = f"x1\tH1\tH2\n.1\t100\t200\n.3\t{marker}\t400\n.5\t300\t600"
    proposal = inspect_observations(
        text,
        import_options={
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
        },
    )
    assert proposal["ready"], proposal["issues"]
    assert [s["observations"] for s in proposal["series_reports"]] == [2, 3]
    assert len(proposal["observations"]) == 5
    assert proposal["raw_rows"][1][1] == marker.strip()


@pytest.mark.parametrize("sign", ["-", "−", "–"])
def test_pdf_dashes_follow_known_column_count(sign):
    header = "T_K x1 HE_J_mol weight pin_tolerance\n"
    negative = inspect_observations(header + f"300 .2 {sign} 20 1 .00001")
    assert negative["ready"], negative["issues"]
    assert negative["observations"][0]["HE_J_mol"] == -20
    missing = inspect_observations(header + f"300 .2 {sign} 1 .00001")
    assert missing["raw_rows"][0] == ["300", ".2", sign, "1", ".00001"]
    assert not missing["ready"]
    assert any("HE_J_mol" in issue for issue in missing["issues"])


@pytest.mark.parametrize("delimiter", [",", "\t", " | "])
def test_explicit_cell_boundaries_preserve_dashes(delimiter):
    text = delimiter.join(["kind", "T_K", "gamma1_inf", "gamma2_inf"])
    text += "\n" + delimiter.join(["GAMMA_INF", "300", "-", "2"])
    result = parse_observations(text)[0]
    assert "gamma1_inf" not in result and result["gamma2_inf"] == 2


def test_detached_signs_inside_explicit_cells_and_decimal_commas():
    assert (
        parse_observations("kind;T_K;x1;HE_J_mol\nHE;300;0,2;- 20")[0]["HE_J_mol"]
        == -20
    )
    proposal = inspect_observations("T_K x1 HE_J_mol\n300 0,2 − 2,5e1")
    assert proposal["ready"], proposal["issues"]
    assert proposal["observations"][0]["HE_J_mol"] == -25


@pytest.mark.parametrize("order", ["before", "after"])
def test_headerless_sign_alignment_can_use_other_complete_rows(order):
    lines = ["300 .2 - 20", "310 .3 30"]
    if order == "after":
        lines.reverse()
    proposal = inspect_observations(
        "\n".join(lines),
        import_options={
            "kind": "HE",
            "temperature_unit": "K",
            "enthalpy_unit": "J/mol",
            "mapping": ["temperature", "x1", "enthalpy"],
        },
    )
    assert proposal["ready"], proposal["issues"]
    assert sorted(p["HE_J_mol"] for p in proposal["observations"]) == [-20, 30]


def test_ambiguous_signs_require_complete_explicit_row_alignment():
    text = "T_K x1 HE_J_mol weight pin_tolerance\n300 .2 - 20 - .00001"
    proposal = inspect_observations(text)
    assert not proposal["ready"] and len(proposal["ambiguous_rows"]) == 1
    assert proposal["ambiguous_rows"][0]["values"] == [
        "300",
        ".2",
        "-",
        "20",
        "-",
        ".00001",
    ]
    edits = [
        {"row": 0, "column": col, "value": val}
        for col, val in enumerate(["300", ".2", "-20", "?", ".00001"])
    ]
    partial = inspect_observations(text, import_options={"cell_edits": edits[:3]})
    assert not partial["ready"] and partial["ambiguous_rows"]
    resolved = inspect_observations(text, import_options={"cell_edits": edits})
    assert resolved["ready"], resolved["issues"]
    assert not resolved["ambiguous_rows"]
    assert resolved["observations"][0]["HE_J_mol"] == -20
    assert resolved["observations"][0]["weight"] == 1
    assert resolved["original_text"] == text
    assert resolved["cell_edits"] == edits


def test_cell_corrections_fix_missing_required_values_without_rewriting_source():
    text = "kind,T_K,x1,HE_J_mol\nHE,300,.2,None"
    proposal = inspect_observations(text)
    assert not proposal["ready"] and proposal["raw_rows"][0][-1] == "None"
    result = inspect_observations(
        text, import_options={"cell_edits": [{"row": 0, "column": 3, "value": "-20"}]}
    )
    assert result["ready"] and result["observations"][0]["HE_J_mol"] == -20
    assert result["original_text"] == text


@pytest.mark.parametrize(
    "edits",
    [
        None,
        {},
        [{"row": -1, "column": 0, "value": "2"}],
        [{"row": True, "column": 0, "value": "2"}],
        [{"row": 0, "column": 10, "value": "2"}],
        [{"row": 0, "column": 0, "value": None}],
        [{"row": 0, "column": 0, "value": "2", "extra": 1}],
    ],
)
def test_invalid_cell_corrections_are_rejected(edits):
    with pytest.raises(ValueError, match="cell edit|cell_edits"):
        inspect_observations(
            "T_K gamma1_inf\n300 2", import_options={"cell_edits": edits}
        )


FLAT_OPTIONS = {
    "kind": "HE",
    "mapping": ["temperature", "x1", "enthalpy"],
    "temperature_unit": "K",
    "enthalpy_unit": "J/mol",
    "composition_basis": "mole_fraction",
}


def test_single_line_table_requests_dimensions_before_interpretation():
    text = "300 .2 20 310 .3 30"
    result = inspect_observations(text)
    assert not result["ready"] and result["flattened"] and result["dimensions_needed"]
    assert result["value_count"] == 6
    assert result["row_count"] is result["column_count"] is None
    assert any("row and column counts" in issue for issue in result["issues"])
    assert result["original_text"] == text


@pytest.mark.parametrize(
    "layout,text",
    [("rows", "300 .2 -20 310 .3 30"), ("columns", "300 310 .2 .3 -20 30")],
)
def test_single_line_dimensions_support_both_orders(layout, text):
    result = inspect_observations(
        text,
        import_options={
            **FLAT_OPTIONS,
            "row_count": 2,
            "column_count": 3,
            "layout": layout,
        },
    )
    assert result["ready"], result["issues"]
    assert result["row_count"] == 2 and result["column_count"] == 3
    assert [(p["T_K"], p["x1"], p["HE_J_mol"]) for p in result["observations"]] == [
        (300, 0.2, -20),
        (310, 0.3, 30),
    ]


def test_single_line_over_column_limit_still_requests_dimensions():
    result = inspect_observations(" ".join(["300", ".2", "20"] * 20))
    assert result["dimensions_needed"] and result["value_count"] == 60


@pytest.mark.parametrize("prefix", ["Table 1\n", "T_K x1 HE_J_mol\n"])
def test_flattened_data_line_with_caption_or_header_requests_dimensions(prefix):
    text = prefix + "300 .2 20 310 .3 30"
    result = inspect_observations(text)
    assert not result["ready"] and result["dimensions_needed"]
    result = inspect_observations(
        text, import_options={**FLAT_OPTIONS, "row_count": 2, "column_count": 3}
    )
    assert result["ready"], result["issues"]
    assert len(result["observations"]) == 2


def test_wrong_dimensions_remain_reviewable_and_one_dimension_does_not_guess_other():
    for dimensions in [
        {"column_count": 3},
        {"row_count": 2},
        {"row_count": 3, "column_count": 3},
    ]:
        result = inspect_observations(
            "300 .2 20 310 .3 30", import_options={**FLAT_OPTIONS, **dimensions}
        )
        assert not result["ready"] and result["dimensions_needed"]


@pytest.mark.parametrize(
    "dimensions",
    [
        {"row_count": 0, "column_count": 3},
        {"row_count": 2001, "column_count": 3},
        {"row_count": True, "column_count": 3},
        {"row_count": 2, "column_count": 31},
        {"row_count": 2, "column_count": 2.5},
        {"row_count": "invalid", "column_count": 3},
        {"row_count": float("inf"), "column_count": 3},
        {"row_count": 2, "column_count": []},
    ],
)
def test_dimension_bounds(dimensions):
    with pytest.raises(ValueError, match="row_count|column_count"):
        inspect_observations("300 .2 20 310 .3 30", import_options=dimensions)


def test_flattened_missing_indicators_and_detached_negatives():
    result = inspect_observations(
        "300 .2 - 20 310 .3 30",
        import_options={**FLAT_OPTIONS, "row_count": 2, "column_count": 3},
    )
    assert result["ready"], result["issues"]
    assert result["observations"][0]["HE_J_mol"] == -20
    result = inspect_observations(
        "300 2 ? 310 None 3",
        import_options={
            "kind": "GAMMA_INF",
            "mapping": ["temperature", "gamma1_inf", "gamma2_inf"],
            "row_count": 2,
            "column_count": 3,
            "temperature_unit": "K",
        },
    )
    assert result["ready"], result["issues"]
    assert "gamma2_inf" not in result["observations"][0]
    assert "gamma1_inf" not in result["observations"][1]


def test_ambiguous_flattened_signs_require_review_after_dimensions():
    result = inspect_observations(
        "300 .2 - 20 - .00001", import_options={"row_count": 1, "column_count": 5}
    )
    assert not result["ready"] and not result["dimensions_needed"]
    assert len(result["ambiguous_rows"]) == 1
    assert result["raw_rows"] == [[""] * 5]


def test_flattened_multiline_rows_are_validated_when_supplied():
    text = "300\n.2\n20\n310\n.3\n30"
    result = inspect_observations(
        text, import_options={**FLAT_OPTIONS, "row_count": 3, "column_count": 3}
    )
    assert not result["ready"] and result["dimensions_needed"]
    result = inspect_observations(
        text,
        import_options={
            **FLAT_OPTIONS,
            "row_count": 2,
            "column_count": 3,
            "layout": "rows",
        },
    )
    assert result["ready"], result["issues"]


@pytest.mark.parametrize(
    "values",
    [
        ["300", "2", "?", "310", "None", "3"],
        ["None", "2", "3", "310", "2", "3"],
    ],
)
def test_multiline_flattening_preserves_missing_value_positions(values):
    result = inspect_observations(
        "\n".join(values),
        import_options={
            "kind": "GAMMA_INF",
            "mapping": ["temperature", "gamma1_inf", "gamma2_inf"],
            "row_count": 2,
            "column_count": 3,
            "layout": "rows",
            "temperature_unit": "K",
        },
    )
    assert result["raw_rows"] == [values[:3], values[3:]]
    if values[0] == "None":
        assert not result["ready"]
    else:
        assert result["ready"], result["issues"]
        assert "gamma2_inf" not in result["observations"][0]


def test_multiline_flattening_uses_dimensions_for_detached_signs():
    result = inspect_observations(
        "300\n.2\n-\n20\n310\n.3\n30",
        import_options={
            **FLAT_OPTIONS,
            "row_count": 2,
            "column_count": 3,
            "layout": "rows",
        },
    )
    assert result["ready"], result["issues"]
    assert result["observations"][0]["HE_J_mol"] == -20


def test_single_point_series_do_not_request_dimensions_again():
    result = inspect_observations(
        "A\tB\n2\t3",
        import_options={
            "kind": "GAMMA_INF",
            "mapping": ["gamma1_inf", "gamma1_inf"],
            "temperature_unit": "K",
            "series": [
                {"columns": [0], "temperature": 300},
                {"columns": [1], "temperature": 310},
            ],
        },
    )
    assert result["ready"], result["issues"]
    assert len(result["observations"]) == 2


def test_catalog_shares_all_markers_with_the_browser():
    assert {marker.strip().lower() for marker in MARKERS} <= set(
        fitting_catalog()["missing_tokens"]
    )
    json.dumps(fitting_catalog())


def test_missing_measurements_do_not_contribute_fitting_residuals():
    rows = [
        {"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 1, "gamma2_inf": marker}
        for marker in MARKERS
    ]
    rows.append({"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 1, "gamma2_inf": 2})
    problem = prepare_fit(
        {
            "components": ["ethanol", "water"],
            "model": "NRTL",
            "form": "constant",
            "observations": rows,
        }
    )
    errors = problem.residuals(np.zeros(2), problem.request["observations"])
    assert len(errors) == len(MARKERS) + 2
    np.testing.assert_allclose(errors[:-1], 0, atol=1e-12)
    assert abs(errors[-1]) > 0.1
