"""Current registry coverage and warnings that match actual solver contracts."""

from pathlib import Path
import pytest

from pfd_parser import parse_pfd, Unit
from dof_analyzer import (
    DOFAnalyzer,
    SpecificationStatus,
    analyze_dof,
    get_unit_info,
    list_unit_types,
)
from unit_syntax import UNIT_TYPE_ALIASES
from unit_settings import unit_setting_schema

ROOT = Path(__file__).resolve().parents[1]


def test_every_registered_type_and_alias_has_current_rules():
    assert set(list_unit_types()) == set(UNIT_TYPE_ALIASES.values())
    for alias, canonical in UNIT_TYPE_ALIASES.items():
        assert get_unit_info(alias) == get_unit_info(canonical)
        assert get_unit_info(canonical)


@pytest.mark.parametrize(
    "unit_type",
    [
        "RigorousDistillation",
        "CMODistillation",
        "McCabeThieleDistillation",
        "ShortcutDistillation",
        "RigorousExtractor",
        "ShortcutExtractor",
        "RigorousAbsorber",
        "RigorousStripper",
    ],
)
def test_default_closed_columns_have_no_spurious_missing_specs(unit_type):
    pfd = parse_pfd(f"UNIT U : {unit_type}\n")
    result = DOFAnalyzer(pfd)._analyze_unit(pfd.units[0])
    assert result.status is SpecificationStatus.OK
    assert result.dof == 0


def test_all_examples_have_no_unknown_registered_unit_warnings():
    for path in (ROOT / "examples").glob("*.pfd"):
        result = analyze_dof(parse_pfd(path.read_text()))
        assert not any("Unknown unit type" in warning for warning in result.warnings), (
            path.name
        )


def test_recycle_initial_guesses_are_not_over_constraints():
    text = (ROOT / "examples/3methylpyridine_ether_extraction_recycle.pfd").read_text()
    pfd = parse_pfd(text)
    result = analyze_dof(pfd)
    assert not any("over-constrain" in warning for warning in result.warnings)
    assert (
        next(r for r in result.stream_results if r.entity_id == "Recycle").status
        is SpecificationStatus.OK
    )


def test_genuine_missing_and_conflicting_targets_remain_errors():
    missing = analyze_dof(parse_pfd("UNIT U : Pump\n"))
    conflicting = analyze_dof(
        parse_pfd("UNIT U : Heater\n    T_out = 80 [C]\n    Q = 10 [kW]\n")
    )
    assert missing.unit_results[0].status is SpecificationStatus.UNDER_SPECIFIED
    assert missing.overall_status is SpecificationStatus.UNDER_SPECIFIED
    assert conflicting.unit_results[0].status is SpecificationStatus.OVER_SPECIFIED
    custom = Unit("X", "NotRegistered", ports=[])
    pfd = parse_pfd("PROCESS: unknown\n")
    pfd.units.append(custom)
    assert "Unknown unit type" in DOFAnalyzer(pfd)._analyze_unit(custom).message


@pytest.mark.parametrize("unit_type", sorted(set(UNIT_TYPE_ALIASES.values())))
def test_guided_settings_cover_every_unit(unit_type):
    schema = unit_setting_schema(unit_type)
    assert schema
    assert all(
        item["label"] and item["section"] in {"operation", "equipment", "solver"}
        for item in schema
    )
    for primary in get_unit_info(unit_type)["primary_specs"]:
        assert any(
            item["name"].lower() == primary.lower() and item["section"] == "operation"
            for item in schema
        )


def test_stage_count_and_reflux_are_primary_named_controls():
    for unit_type in [
        "RigorousDistillation",
        "CMODistillation",
        "McCabeThieleDistillation",
        "ShortcutDistillation",
    ]:
        schema = {item["name"].lower(): item for item in unit_setting_schema(unit_type)}
        assert schema["n_stages"]["section"] == "operation"
        assert schema["reflux_ratio"]["section"] == "operation"
        assert schema["n_stages"]["label"] == "Number of stages"
        assert schema["reflux_ratio"]["label"] == "Reflux ratio (L/D)"
