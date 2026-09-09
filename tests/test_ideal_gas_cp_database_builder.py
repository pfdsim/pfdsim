import unittest
import sqlite3
from unittest.mock import patch

import numpy as np

from scripts.heat_capacity.gas import build_ideal_gas_heat_capacity_db as builder


class CanonicalCasQuarantineTests(unittest.TestCase):
    def test_production_database_quarantines_known_redirects(self):
        with sqlite3.connect(builder.OUTPUT) as connection:
            selected = connection.execute(
                """
                SELECT count(*) FROM canonical_ideal_gas_cp
                WHERE cas IN ('12075-35-3', '12070-15-4')
                """
            ).fetchone()[0]
            quarantined = connection.execute(
                """
                SELECT count(DISTINCT cas) FROM source_quarantine
                WHERE cas IN ('12075-35-3', '12070-15-4')
                  AND reason = 'source CAS resolves to a different canonical CAS'
                """
            ).fetchone()[0]
        self.assertEqual(selected, 0)
        self.assertEqual(quarantined, 2)

    def test_compile_quarantines_redirected_source_cas(self):
        candidate = builder.SourceCandidate(
            cas="12075-35-3",
            source="trc_1994",
            source_key="12075-35-3",
            source_version="test",
            name="allene",
            formula="C3H4",
            segments=(
                builder.SourceSegment(
                    100.0,
                    1000.0,
                    lambda temperatures: np.full_like(temperatures, 40.0),
                ),
            ),
            source_hash="test",
            details={},
        )
        redirect = {
            "source_cas": "12075-35-3",
            "resolved_canonical_cas": "463-49-0",
            "resolved_name": "allene",
            "resolved_formula": "C3H4",
        }
        with (
            patch.object(builder, "collect_perry_candidates", return_value=[candidate]),
            patch.object(builder, "collect_coolprop_candidates", return_value=([], [])),
            patch.object(builder, "collect_chemicals_candidates", return_value=([], [])),
            patch.object(builder, "noncanonical_cas_details", return_value=redirect),
        ):
            selected, candidates, crosschecks, quarantines, report = (
                builder.compile_records()
            )

        self.assertEqual(selected, [])
        self.assertEqual(crosschecks, [])
        self.assertEqual(candidates, [candidate])
        self.assertEqual(candidate.status, "quarantined")
        self.assertEqual(
            candidate.reason,
            "source CAS resolves to a different canonical CAS",
        )
        self.assertEqual(candidate.details["identity_quarantine"], redirect)
        self.assertEqual(len(quarantines), 1)
        self.assertEqual(report["selected_count"], 0)
        self.assertEqual(report["quarantine_count"], 1)

    def test_verified_identity_override_is_not_redirected(self):
        self.assertIsNone(builder.noncanonical_cas_details("14989-30-1"))


if __name__ == "__main__":
    unittest.main()
