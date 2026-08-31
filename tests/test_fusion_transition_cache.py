import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from property_resolution.common import (
    FusionTransitionRecord,
    OnlineAttemptState,
)
from property_resolution.resolver import PropertyResolver


class FusionTransitionCacheTests(unittest.TestCase):
    @staticmethod
    def fixture_props(tm=300.0):
        return {
            'symbol': 'XFUS',
            'name': 'fusion cache fixture',
            'formula': 'C6H6',
            'CAS': '999-99-9',
            'Tm': tm,
            'fusion_transitions': [
                {
                    'enthalpy_kJ_mol': 10.0,
                    'temperature_K': 300.0,
                    'material_form': 'anhydrous',
                    'source': 'provided',
                    'method': 'fixture_anhydrous',
                    'quality': 0.97,
                },
                {
                    'enthalpy_kJ_mol': 20.0,
                    'temperature_K': 350.0,
                    'material_form': 'hydrate',
                    'form_label': 'monohydrate',
                    'source': 'provided',
                    'method': 'fixture_hydrate',
                    'quality': 0.98,
                },
            ],
            'property_sources': {
                'Tm': {
                    'source': 'provided',
                    'method': 'fixture_tm',
                    'quality': 0.97,
                },
            },
        }

    def test_persists_structured_records_and_selected_scalar(self):
        props = self.fixture_props()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            first = PropertyResolver()
            first.SATURATION_PROPERTIES_CACHE_PATH = path
            records = first.resolve_fusion_transitions(
                'XFUS', props, allow_online=False,
            )
            self.assertEqual(len(records), 2)

            with sqlite3.connect(path) as connection:
                connection.row_factory = sqlite3.Row
                row = connection.execute(
                    'SELECT * FROM resolved_fusion_transition_cache'
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(
                    row['cache_version'],
                    first.FUSION_TRANSITION_CACHE_VERSION,
                )
                self.assertEqual(
                    row['online_phase_contract_version'],
                    first.ONLINE_PHASE_CHANGE_CACHE_VERSION,
                )
                self.assertEqual(row['component_key'], 'cas:999-99-9')
                self.assertEqual(
                    row['online_attempt_state'],
                    OnlineAttemptState.NOT_NEEDED.value,
                )
                self.assertEqual(row['selected_Hfus_value'], 10.0)
                self.assertEqual(row['selected_Hfus_method'], 'fixture_anhydrous')
                self.assertEqual(len(json.loads(row['records_json'])), 2)
                selected = json.loads(row['selected_records_json'])
                self.assertEqual(len(selected), 1)
                self.assertEqual(selected[0]['material_form'], 'anhydrous')
                metadata = json.loads(row['input_metadata_json'])
                self.assertEqual(metadata['units']['Hfus'], 'kJ/mol')
                self.assertEqual(
                    metadata['online_phase_contract_version'],
                    first.ONLINE_PHASE_CHANGE_CACHE_VERSION,
                )
                self.assertEqual(
                    len(metadata['perry_heat_of_fusion_sha256']),
                    64,
                )

            second = PropertyResolver()
            second.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                second,
                '_resolve_fusion_transitions_uncached',
                side_effect=AssertionError('persistent cache was not read'),
            ):
                loaded = second.resolve_fusion_transitions(
                    'XFUS', props, allow_online=False,
                )
            self.assertEqual(loaded, records)
            self.assertTrue(all(
                isinstance(record, FusionTransitionRecord)
                for record in loaded
            ))

    def test_tm_and_transition_metadata_change_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = Path(
                directory, 'saturation.sqlite',
            )
            resolver.resolve_fusion_transitions(
                'XFUS', self.fixture_props(300.0), allow_online=False,
            )
            changed = self.fixture_props(301.0)
            changed['fusion_transitions'][0]['form_label'] = 'form I'
            resolver.resolve_fusion_transitions(
                'XFUS', changed, allow_online=False,
            )
            with sqlite3.connect(resolver.SATURATION_PROPERTIES_CACHE_PATH) as connection:
                count = connection.execute(
                    'SELECT count(*) FROM resolved_fusion_transition_cache'
                ).fetchone()[0]
            self.assertEqual(count, 2)

    def test_fusion_rows_expire_after_30_days(self):
        props = self.fixture_props()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            first = PropertyResolver()
            first.SATURATION_PROPERTIES_CACHE_PATH = path
            first.resolve_fusion_transitions(
                'XFUS', props, allow_online=False,
            )
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    UPDATE resolved_fusion_transition_cache
                    SET updated_at_utc = datetime('now', '-31 days')
                    """
                )

            replacement = (FusionTransitionRecord(
                enthalpy_kJ_mol=11.0,
                temperature_K=300.0,
                source='fixture',
                method='rebuilt_after_ttl',
                quality=0.95,
            ),)
            second = PropertyResolver()
            second.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                second,
                '_resolve_fusion_transitions_uncached',
                return_value=replacement,
            ) as build:
                actual = second.resolve_fusion_transitions(
                    'XFUS', props, allow_online=False,
                )
            self.assertEqual(build.call_count, 1)
            self.assertEqual(actual, replacement)

    def test_transient_online_result_is_not_persisted(self):
        props = self.fixture_props()
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = Path(
                directory, 'saturation.sqlite',
            )

            def transient(*_args, **_kwargs):
                resolver._record_online_attempt_state(
                    OnlineAttemptState.TRANSIENT_FAILURE
                )
                return ()

            with patch.object(
                resolver,
                '_resolve_fusion_transitions_uncached',
                side_effect=transient,
            ):
                self.assertEqual(
                    resolver.resolve_fusion_transitions(
                        'XFUS', props, allow_online=True,
                    ),
                    (),
                )

            with sqlite3.connect(resolver.SATURATION_PROPERTIES_CACHE_PATH) as connection:
                resolver._ensure_phase_point_cache_schema(connection)
                count = connection.execute(
                    'SELECT count(*) FROM resolved_fusion_transition_cache'
                ).fetchone()[0]
            self.assertEqual(count, 0)

    def test_pfd_scalar_and_future_structured_overrides_bypass_all_caches(self):
        for structured in (False, True):
            with self.subTest(structured=structured), tempfile.TemporaryDirectory() as directory:
                resolver = PropertyResolver()
                resolver.SATURATION_PROPERTIES_CACHE_PATH = Path(
                    directory, 'saturation.sqlite',
                )
                props = self.fixture_props()
                if structured:
                    props['fusion_transitions'][0].update({
                        'source': 'pfd',
                        'method': 'pfd_component_override',
                    })
                else:
                    props['Hfus'] = 12.0
                    props['property_sources']['Hfus'] = {
                        'source': 'pfd',
                        'method': 'pfd_component_override',
                        'quality': 1.0,
                    }

                with patch.object(
                    resolver,
                    '_resolve_fusion_transitions_uncached',
                    wraps=resolver._resolve_fusion_transitions_uncached,
                ) as build:
                    first = resolver.resolve_fusion_transitions(
                        'XFUS', props, allow_online=False,
                    )
                    second = resolver.resolve_fusion_transitions(
                        'XFUS', props, allow_online=False,
                    )
                self.assertEqual(first, second)
                self.assertEqual(build.call_count, 2)
                self.assertFalse(hasattr(
                    resolver, '_resolved_fusion_transition_cache',
                ))

                resolver.initialize_phase_point_disk_cache()
                with sqlite3.connect(resolver.SATURATION_PROPERTIES_CACHE_PATH) as connection:
                    count = connection.execute(
                        'SELECT count(*) FROM resolved_fusion_transition_cache'
                    ).fetchone()[0]
                self.assertEqual(count, 0)


if __name__ == '__main__':
    unittest.main()
