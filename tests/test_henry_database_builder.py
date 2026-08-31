import math

import pytest

from scripts.build_henry_constants_db import (
    _candidate,
    _brockbank_temperature_correlations,
    aggregate_field,
    parse_float,
    parse_three_parameter_hcp,
)


def source_row(
    row_id,
    H,
    B="",
    *,
    htype="M",
    literature=None,
    literature_quotation="",
    note_labels="",
    note_texts="",
):
    return {
        "id": row_id,
        "hominus": H,
        "mindhr": B,
        "htype": htype,
        "literature_id": literature or str(row_id),
        "literature_quotation": literature_quotation,
        "note_labels": note_labels,
        "note_texts": note_texts,
    }


def test_parse_float_handles_official_html_scientific_notation():
    assert parse_float("1.3&times;10<sup>&#8722;5</sup>") == pytest.approx(1.3e-5)
    assert parse_float("2.4&times;10<sup>3</sup>") == pytest.approx(2.4e3)
    assert parse_float("&gt; 2.3E-10") is None
    assert parse_float("&#8734;") is None


def test_parse_three_parameter_hcp_from_sander_note():
    note = (
        "H<sub>s</sub><sup>cp</sup>= exp( &#8722;130.91491 "
        "+6700.12242/T +17.04684&nbsp;ln(T)) mol m-3 Pa-1"
    )
    assert parse_three_parameter_hcp(note) == pytest.approx(
        (-130.91491, 6700.12242, 17.04684)
    )
    assert parse_three_parameter_hcp("More detailed temperature data exist.") is None


def test_brockbank_extract_selects_only_multitemperature_experimental_curves():
    correlations = _brockbank_temperature_correlations()
    assert len(correlations) == 314
    assert correlations['67-64-1']['component'] == 'ACETONE'
    assert correlations['67-64-1']['Tmin_K'] == 273.0
    assert correlations['67-64-1']['Tmax_K'] == 373.15
    assert correlations['67-64-1']['uncertainty_upper_percent'] == 10.0


@pytest.mark.parametrize(
    ("note_labels", "note_texts", "reason"),
    [
        ("seawater", "Solubility in sea water.", "seawater"),
        ("759", "Natural brines.", "concentrated_brine"),
        ("3917", "D2O solvent.", "heavy_water_solvent"),
        ("3521", "60% aqueous ethanol.", "aqueous_ethanol_solvent"),
        ("715pH4", "Value at pH = 4.", "ph_specific_value"),
        ("HCHOdiol", "Effective hydrated value.", "reactive_effective_constant"),
    ],
)
def test_non_pure_water_and_context_specific_values_are_excluded(
    note_labels,
    note_texts,
    reason,
):
    row = source_row(
        1,
        "1.0",
        "2000",
        note_labels=note_labels,
        note_texts=note_texts,
    )
    candidate, status = _candidate(row, "h")
    assert candidate is None
    assert status == f"excluded:{reason}"


def test_off_reference_value_is_normalized_with_paired_temperature_coefficient():
    row = source_row(1, "2.0", "2000", note_labels="at288K")
    candidate, status = _candidate(row, "h")
    expected = 2.0 * math.exp(-2000.0 * (1.0 / 288.0 - 1.0 / 298.15))
    assert status == "candidate"
    assert candidate.value == pytest.approx(expected)
    assert candidate.provenance_multiplier < 1.0


def test_unknown_or_distant_unadjustable_temperature_is_excluded():
    unknown, unknown_status = _candidate(
        source_row(1, "2.0", note_labels="unknownT"), "h"
    )
    distant, distant_status = _candidate(
        source_row(2, "2.0", note_labels="at273K"), "h"
    )
    assert unknown is None
    assert unknown_status == "excluded:unknown_temperature"
    assert distant is None
    assert distant_status == "excluded:temperature_not_298K"


def test_intrinsic_h_is_kept_when_only_temperature_dependence_is_ph_specific():
    row = source_row(
        1,
        "2.0",
        "4100",
        note_labels="3411",
        note_texts=(
            "The value at the reference temperature is intrinsic, but the "
            "temperature dependence refers to an effective constant at pH = 3.08."
        ),
    )
    h_candidate, h_status = _candidate(row, "h")
    b_candidate, b_status = _candidate(row, "b")
    assert h_candidate.value == pytest.approx(2.0)
    assert h_status == "candidate"
    assert b_candidate is None
    assert b_status == "excluded:ph_specific_value"


def test_log_space_outlier_is_excluded_and_auditable():
    rows = [
        source_row(1, "1.0"),
        source_row(2, "1.1"),
        source_row(3, "100.0"),
    ]
    aggregate = aggregate_field(rows, "h")
    assert aggregate.value == pytest.approx(math.sqrt(1.1))
    assert [row["id"] for row in aggregate.outlier_rows] == [3]
    assert rows[2]["_h_status"] == "excluded:statistical_outlier"


def test_agreement_materially_affects_quality_grade():
    agreeing = aggregate_field(
        [source_row(1, "0.9"), source_row(2, "1.0"), source_row(3, "1.1")],
        "h",
    )
    disagreeing = aggregate_field(
        [source_row(4, "1.0"), source_row(5, "100.0")],
        "h",
    )
    assert agreeing.agreement_score > 0.85
    assert disagreeing.agreement_score < 0.2
    assert agreeing.quality_score > disagreeing.quality_score + 0.2


def test_duplicate_rows_from_one_reference_do_not_inflate_support():
    aggregate = aggregate_field(
        [
            source_row(1, "1.0", literature="same"),
            source_row(2, "1.1", literature="same"),
            source_row(3, "0.95", literature="independent"),
        ],
        "h",
    )
    assert aggregate.source_count == 2


def test_lower_reliability_band_cannot_overwhelm_measurements():
    rows = [source_row(1, "1.0", htype="M")]
    rows.extend(source_row(index, "100.0", htype="Q") for index in range(2, 22))
    aggregate = aggregate_field(rows, "h")
    assert aggregate.value == pytest.approx(1.0)
    assert all(row["_h_status"] == "excluded:lower_reliability_band" for row in rows[1:])


def test_repeated_reviews_alone_cannot_receive_an_a_grade():
    aggregate = aggregate_field(
        [
            source_row(
                1,
                "1.0",
                htype="L",
                literature="review-v1",
                literature_quotation="Same Author et al. (2015)",
            ),
            source_row(
                2,
                "1.0",
                htype="L",
                literature="review-v2",
                literature_quotation="Same Author et al. (2019)",
            ),
        ],
        "h",
    )
    assert aggregate.review_evidence == pytest.approx(1.0)
    assert aggregate.direct_evidence == 0.0
    assert aggregate.grade == "B+"


def test_independent_review_families_add_diminishing_confidence():
    aggregate = aggregate_field(
        [
            source_row(
                1,
                "1.0",
                htype="L",
                literature="review-a",
                literature_quotation="Alpha et al. (2010)",
            ),
            source_row(
                2,
                "1.0",
                htype="L",
                literature="review-b",
                literature_quotation="Beta et al. (2020)",
            ),
        ],
        "h",
    )
    assert aggregate.review_evidence == pytest.approx(1.5)
    assert aggregate.direct_evidence == 0.0
    assert aggregate.grade == "A-"


def test_successive_jpl_editions_share_one_review_family():
    aggregate = aggregate_field(
        [
            source_row(
                1,
                "1.0",
                htype="L",
                literature="3245",
                literature_quotation="Burkholder et al. (2015)",
            ),
            source_row(
                2,
                "1.0",
                htype="L",
                literature="3500",
                literature_quotation="Burkholder et al. (2019)",
            ),
            source_row(
                3,
                "1.0",
                htype="L",
                literature="2626",
                literature_quotation="Sander et al. (2011)",
            ),
        ],
        "h",
    )
    assert aggregate.review_evidence == pytest.approx(1.0)
    assert aggregate.grade == "B+"


def test_agreeing_primary_evidence_monotonically_improves_confidence():
    rows = [
        source_row(1, "1.0", htype="L", literature="review-v1"),
        source_row(2, "1.0", htype="L", literature="review-v2"),
    ]
    scores = [aggregate_field([dict(row) for row in rows], "h").quality_score]
    for index in range(3, 9):
        rows.append(source_row(index, "1.0", htype="M"))
        scores.append(aggregate_field([dict(row) for row in rows], "h").quality_score)
    assert scores == sorted(scores)
    assert scores[-1] > 0.95


def test_highly_conflicted_evidence_can_reduce_confidence():
    agreeing = aggregate_field(
        [source_row(index, "1.0", htype="M") for index in range(1, 6)],
        "h",
    )
    conflicted = aggregate_field(
        [
            source_row(1, "1.0", htype="M"),
            source_row(2, "1.1", htype="M"),
            source_row(3, "0.9", htype="M"),
            source_row(4, "10.0", htype="M"),
            source_row(5, "0.1", htype="M"),
        ],
        "h",
    )
    assert conflicted.quality_score < agreeing.quality_score
