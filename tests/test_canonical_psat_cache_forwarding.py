import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest

from property_resolution.vapor_pressure_canonical import (
    PsatHandoffCoordinator,
    PsatHandoffPolicy,
    PsatHandoffRequirement,
    PsatSegment,
    PsatSegmentProvenance,
    PsatSegmentType,
)
from scripts.forward_canonical_psat_cache import forward_cache


def _create_cache(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE canonical_psat_cache (
                cache_version INTEGER NOT NULL,
                component_key TEXT NOT NULL,
                input_fingerprint TEXT NOT NULL,
                symbol TEXT NOT NULL,
                cas TEXT NOT NULL,
                component_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{"handoff_decisions": []}',
                PRIMARY KEY (cache_version, component_key, input_fingerprint)
            )
            """
        )
        connection.executemany(
            "INSERT INTO canonical_psat_cache "
            "(cache_version, component_key, input_fingerprint, symbol, cas, "
            "component_name, payload, provenance_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    14,
                    "affected",
                    "a",
                    "E2H",
                    "104-76-7",
                    "2-ethyl-1-hexanol",
                    "old",
                    '[{"method": "antoine", "segment_type": "pinned"}]',
                ),
                (
                    14,
                    "safe",
                    "b",
                    "BENZ",
                    "71-43-2",
                    "benzene",
                    "unchanged",
                    '[{"method": "antoine", "segment_type": "pinned"}]',
                ),
            ],
        )


def test_forward_cache_derives_affected_identities_from_trusted_data(tmp_path):
    database = tmp_path / "canonical.sqlite"
    trusted = tmp_path / "trusted.json"
    _create_cache(database)
    trusted.write_text(
        json.dumps(
            {
                "correlations": [
                    {
                        "identity": {
                            "cas": "104-76-7",
                            "name": "2-ethyl-1-hexanol",
                            "aliases": ["2-ethylhexanol"],
                        }
                    }
                ]
            }
        )
    )

    dry_run = forward_cache(database, trusted, 14, 15, apply=False)
    assert dry_run["affected_rows_skipped"] == 1
    assert dry_run["rows_to_insert"] == 1
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM canonical_psat_cache WHERE cache_version = 15"
            ).fetchone()[0]
            == 0
        )

    applied = forward_cache(database, trusted, 14, 15, apply=True)
    assert applied["rows_to_insert"] == 1
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT component_key, payload FROM canonical_psat_cache "
            "WHERE cache_version = 15"
        ).fetchall()
    assert rows == [("safe", "unchanged")]

    repeated = forward_cache(database, trusted, 14, 15, apply=True)
    assert repeated["rows_to_insert"] == 0
    assert repeated["already_identical"] == 1


def test_forward_cache_skips_rows_with_multiple_pinned_methods(tmp_path):
    database = tmp_path / "canonical.sqlite"
    trusted = tmp_path / "trusted.json"
    trusted.write_text('{"correlations": []}')
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE canonical_psat_cache (
                cache_version INTEGER NOT NULL,
                component_key TEXT NOT NULL,
                input_fingerprint TEXT NOT NULL,
                symbol TEXT NOT NULL,
                cas TEXT NOT NULL,
                component_name TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{"handoff_decisions": []}',
                PRIMARY KEY (cache_version, component_key, input_fingerprint)
            )
            """
        )
        connection.executemany(
            "INSERT INTO canonical_psat_cache "
            "(cache_version, component_key, input_fingerprint, symbol, cas, "
            "component_name, provenance_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    15,
                    "affected",
                    "a",
                    "IBOH",
                    "78-83-1",
                    "isobutanol",
                    json.dumps(
                        [
                            {"method": "perry_2_10", "segment_type": "pinned"},
                            {"method": "textbook_antoine", "segment_type": "pinned"},
                        ]
                    ),
                ),
                (
                    15,
                    "safe",
                    "b",
                    "BENZ",
                    "71-43-2",
                    "benzene",
                    json.dumps(
                        [
                            {"method": "perry_2_8", "segment_type": "pinned"},
                            {"method": "aw_upper", "segment_type": "completion"},
                        ]
                    ),
                ),
            ],
        )

    report = forward_cache(database, trusted, 15, 16, apply=True)
    assert report["multi_pinned_rows_skipped"] == 1
    assert report["affected_rows_skipped"] == 1
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT component_key FROM canonical_psat_cache WHERE cache_version = 16"
        ).fetchall()
    assert rows == [("safe",)]


def test_forward_cache_skips_same_method_handoffs_and_unusable_provenance(tmp_path):
    database = tmp_path / "canonical.sqlite"
    trusted = tmp_path / "trusted.json"
    trusted.write_text('{"correlations": []}')
    _create_cache(database)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM canonical_psat_cache")
        for index, provenance in enumerate(
            (
                '[{"method": "antoine", "segment_type": "pinned", "source": "low-range"}, '
                '{"method": "antoine", "segment_type": "pinned", "source": "high-range"}]',
                "",
                "invalid JSON",
                "null",
                "{}",
                '["invalid item"]',
            )
        ):
            connection.execute(
                "INSERT INTO canonical_psat_cache "
                "(cache_version, component_key, input_fingerprint, symbol, cas, "
                "component_name, payload, provenance_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (14, str(index), "a", "BENZ", "71-43-2", "benzene", "old", provenance),
            )
    report = forward_cache(database, trusted, 14, 17, apply=True)
    assert report["affected_rows_skipped"] == 6
    assert report["rows_to_insert"] == 0
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM canonical_psat_cache WHERE cache_version = 17"
            ).fetchone()[0]
            == 0
        )


def test_forward_cache_skips_a_single_survivor_when_rejection_order_can_change(
    tmp_path,
):
    def segment(name, priority, low, high, offset, slope):
        return PsatSegment(
            source=name,
            method=name,
            segment_type=PsatSegmentType.PINNED,
            priority=priority,
            T_min=low,
            T_max=high,
            ln_pressure_function=lambda T: 0.1 * T + offset(T),
            derivative_function=lambda T: 0.1 + slope(T),
            quality=0.95,
            handoff_requirement=(
                PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP_IF_AVAILABLE
            ),
        )

    sources = (
        segment("A", 700, 300, 400, lambda T: 0.1, lambda T: 0.0),
        segment(
            "B",
            600,
            200,
            350,
            lambda T: 0.1 + 0.0002 * (T - 330) ** 2,
            lambda T: 0.0004 * (T - 330),
        ),
        segment(
            "C",
            500,
            200,
            500,
            lambda T: max(390 - T, 0) / 900,
            lambda T: -1 / 900 if T < 390 else 0.0,
        ),
        segment("D", 800, 390, 500, lambda T: 0.0, lambda T: 0.0),
    )
    old = PsatHandoffCoordinator(
        200,
        500,
        PsatHandoffPolicy(max_direct_handoff_compensation_integral_K=1e6),
    ).coordinate(sources)
    current = PsatHandoffCoordinator(200, 500).coordinate(sources)
    assert [item.segment.method for item in old.assembly.slices] == ["D"]
    assert [item.segment.method for item in current.assembly.slices] == ["C", "D"]

    database, trusted = tmp_path / "canonical.sqlite", tmp_path / "trusted.json"
    trusted.write_text('{"correlations": []}')
    _create_cache(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE canonical_psat_cache SET provenance_json=?, metadata_json=? "
            "WHERE component_key='affected'",
            (
                json.dumps(
                    [
                        asdict(PsatSegmentProvenance.from_slice(item))
                        for item in old.assembly.slices
                    ]
                ),
                json.dumps(old.cache_metadata),
            ),
        )
    report = forward_cache(database, trusted, 14, 17, apply=True)
    assert report["multi_pinned_rows_skipped"] == 0
    assert report["rejected_source_rows_skipped"] == 1
    assert report["unverified_handoff_rows_skipped"] == 0
    assert report["rows_to_insert"] == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT component_key FROM canonical_psat_cache WHERE cache_version=17"
        ).fetchall() == [("safe",)]
        assert (
            connection.execute(
                "SELECT count(*) FROM canonical_psat_cache WHERE cache_version=14"
            ).fetchone()[0]
            == 2
        )


@pytest.mark.parametrize(
    "metadata",
    [
        "",
        "invalid JSON",
        "null",
        "[]",
        "{}",
        '{"handoff_decisions": null}',
        '{"handoff_decisions": ["invalid item"]}',
        '{"handoff_decisions": [{"action": "unknown"}]}',
        '{"handoff_decisions": [{"action": ["reject"]}]}',
    ],
)
def test_forward_cache_reports_missing_or_invalid_history_without_deleting_rows(
    tmp_path, metadata
):
    database, trusted = tmp_path / "canonical.sqlite", tmp_path / "trusted.json"
    trusted.write_text('{"correlations": []}')
    _create_cache(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE canonical_psat_cache SET metadata_json=? "
            "WHERE component_key='affected'",
            (metadata,),
        )
    report = forward_cache(database, trusted, 14, 17, apply=True)
    assert report["unverified_handoff_rows_skipped"] == 1
    assert report["rows_to_insert"] == 1
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM canonical_psat_cache WHERE cache_version=14"
            ).fetchone()[0]
            == 2
        )


def test_forward_cache_preserves_existing_destination_results(tmp_path):
    database, trusted = tmp_path / "canonical.sqlite", tmp_path / "trusted.json"
    trusted.write_text('{"correlations": []}')
    _create_cache(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO canonical_psat_cache SELECT 17, component_key, "
            "input_fingerprint, symbol, cas, component_name, 'newer result', "
            "provenance_json, metadata_json FROM canonical_psat_cache "
            "WHERE component_key='safe'"
        )
    report = forward_cache(database, trusted, 14, 17, apply=True)
    assert report["existing_target_kept"] == 1
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT payload FROM canonical_psat_cache "
                "WHERE cache_version=17 AND component_key='safe'"
            ).fetchone()[0]
            == "newer result"
        )
