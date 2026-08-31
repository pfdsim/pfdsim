import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.warm_coolprop_triple_corroboration import (
    RateLimitedUrlopen,
    RequestBudgetExhausted,
    classify_result,
    provisional_targets,
)


class CoolPropTripleWarmTests(unittest.TestCase):
    def test_rate_limiter_spaces_and_caps_actual_requests(self):
        clock = [10.0]
        starts = []

        def monotonic():
            return clock[0]

        def sleep(seconds):
            clock[0] += seconds

        def original(request, *_args, **_kwargs):
            starts.append((request, clock[0]))
            clock[0] += 0.1
            return request

        limiter = RateLimitedUrlopen(
            original,
            maximum_requests=2,
            minimum_interval_seconds=1.05,
            monotonic=monotonic,
            sleep=sleep,
        )
        self.assertEqual(limiter('first'), 'first')
        self.assertEqual(limiter('second'), 'second')
        with self.assertRaises(RequestBudgetExhausted):
            limiter('third')
        self.assertEqual(limiter.request_count, 2)
        self.assertGreaterEqual(starts[1][1] - starts[0][1], 1.05)

    def test_result_classification_distinguishes_all_outcomes(self):
        def triple(value, method, quality):
            return {
                'Tt': SimpleNamespace(
                    value=value,
                    method=method,
                    quality=quality,
                ),
            }

        self.assertEqual(
            classify_result(triple(100.0, 'coolprop_HEOS_triple_point', 0.995)),
            'coolprop_confirmed',
        )
        self.assertEqual(
            classify_result(triple(100.0, 'coolprop_HEOS_triple_point', 0.89)),
            'still_provisional',
        )
        self.assertEqual(
            classify_result(triple(100.0, 'nist_triple_point', 0.96)),
            'external_selected',
        )
        self.assertEqual(
            classify_result(triple(None, 'none', 0.0)),
            'rejected_or_missing',
        )

    def test_resume_targets_use_latest_component_result(self):
        from property_resolution.phase_change import PhaseChangeMixin

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    CREATE TABLE resolved_phase_point_cache (
                        cache_version INTEGER,
                        resolution_kind TEXT,
                        component_key TEXT,
                        cas TEXT,
                        component_name TEXT,
                        Tt_value REAL,
                        Pt_value REAL,
                        Tt_quality REAL,
                        Pt_quality REAL,
                        updated_at_utc TEXT
                    )
                    """
                )
                rows = (
                    ('cas:1', '1-11-1', 'confirmed', 0.89, '2026-01-01'),
                    ('cas:1', '1-11-1', 'confirmed', 0.995, '2026-01-02'),
                    ('cas:2', '2-22-2', 'remaining', 0.89, '2026-01-01'),
                )
                for key, cas, name, quality, updated in rows:
                    connection.execute(
                        """
                        INSERT INTO resolved_phase_point_cache VALUES (
                            ?, 'triple_point', ?, ?, ?, 200.0, 0.1,
                            ?, ?, ?
                        )
                        """,
                        (
                            PhaseChangeMixin.PHASE_POINT_CACHE_VERSION,
                            key,
                            cas,
                            name,
                            quality,
                            quality,
                            updated,
                        ),
                    )
            targets = provisional_targets(path)
        self.assertEqual([item.cas for item in targets], ['2-22-2'])


if __name__ == '__main__':
    unittest.main()
