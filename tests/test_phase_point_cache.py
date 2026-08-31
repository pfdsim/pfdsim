import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from property_resolution.common import OnlineAttemptState, PropertyResolutionResult
from property_resolution.resolver import PropertyResolver
from tests.live_provider import run_optional_live_provider


class PhasePointCacheTests(unittest.TestCase):
    @staticmethod
    def fixture_props():
        return {
            'symbol': 'C2H5OH',
            'name': 'Ethanol',
            'formula': 'C2H6O',
            'CAS': '64-17-5',
            'MW': 46.06844,
            'Tm': 159.0,
            'Tc': 514.71,
            'Hfus': 5.02,
            'property_sources': {
                'Tm': {
                    'source': 'local',
                    'method': 'fixture_melting_point',
                    'quality': 0.96,
                },
                'Hfus': {
                    'source': 'local',
                    'method': 'fixture_heat_of_fusion',
                    'quality': 0.95,
                },
            },
        }

    @staticmethod
    def fixture_tb():
        return PropertyResolutionResult(
            value=351.57040446751455,
            source='local',
            method='coolprop_HEOS_boiling_point',
            quality=0.995,
            notes='complete boiling-point provenance',
        )

    @staticmethod
    def fixture_tm():
        return PropertyResolutionResult(
            value=158.38389460326692,
            source='local',
            method='coolprop_HEOS_melting_point',
            quality=0.995,
            notes='complete melting-point provenance',
        )

    @staticmethod
    def fixture_triple():
        return {
            'Tt': PropertyResolutionResult(
                value=159.1,
                source='local',
                method='coolprop_HEOS_triple_point',
                quality=0.995,
                notes='complete triple-temperature provenance',
            ),
            'Pt': PropertyResolutionResult(
                value=7.2e-9,
                source='local',
                method='coolprop_HEOS_triple_point',
                quality=0.995,
                notes='complete triple-pressure provenance',
            ),
        }

    def test_persists_complete_boiling_melting_and_triple_point_contracts(self):
        props = self.fixture_props()
        expected_tb = self.fixture_tb()
        expected_tm = self.fixture_tm()
        expected_triple = self.fixture_triple()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            first = PropertyResolver()
            first.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                first,
                '_resolve_boiling_point_uncached',
                return_value=expected_tb,
            ) as build_tb, patch.object(
                first,
                '_resolve_melting_point_uncached',
                return_value=expected_tm,
            ) as build_tm, patch.object(
                first,
                '_resolve_triple_point_uncached',
                return_value=expected_triple,
            ) as build_triple:
                actual_tb = first.resolve_boiling_point(
                    'C2H5OH',
                    props,
                    allow_online=False,
                )
                actual_tm = first.resolve_melting_point(
                    'C2H5OH',
                    props,
                    allow_online=False,
                )
                actual_triple = first.resolve_triple_point(
                    'C2H5OH',
                    props,
                    allow_online=False,
                )
            self.assertEqual(build_tb.call_count, 1)
            self.assertEqual(build_tm.call_count, 1)
            self.assertEqual(build_triple.call_count, 1)
            self.assertEqual(actual_tb, expected_tb)
            self.assertEqual(actual_tm, expected_tm)
            self.assertEqual(actual_triple, expected_triple)

            with sqlite3.connect(path) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    """
                    SELECT * FROM resolved_phase_point_cache
                    ORDER BY resolution_kind
                    """
                ).fetchall()
                self.assertEqual(len(rows), 3)
                by_kind = {row['resolution_kind']: row for row in rows}
                boiling = by_kind['boiling_point']
                melting = by_kind['melting_point']
                triple = by_kind['triple_point']
                self.assertEqual(boiling['component_key'], 'cas:64-17-5')
                self.assertEqual(
                    boiling['online_attempt_state'],
                    OnlineAttemptState.NOT_NEEDED.value,
                )
                self.assertEqual(boiling['Tb_method'], expected_tb.method)
                self.assertEqual(melting['Tm_value'], expected_tm.value)
                self.assertEqual(melting['Tm_method'], expected_tm.method)
                self.assertEqual(triple['Tt_method'], expected_triple['Tt'].method)
                self.assertEqual(triple['Pt_notes'], expected_triple['Pt'].notes)
                metadata = json.loads(triple['input_metadata_json'])
                self.assertEqual(
                    metadata['future_triple_point_estimation_inputs'],
                    ['Tm', 'Tb', 'Tc', 'Hfus'],
                )
                self.assertEqual(
                    metadata['properties']['property_sources']['Hfus']['method'],
                    'fixture_heat_of_fusion',
                )
                plan = connection.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT * FROM resolved_phase_point_cache
                    WHERE cache_version = ?
                      AND resolution_kind = ?
                      AND component_key = ?
                      AND input_fingerprint = ?
                    """,
                    (
                        first.PHASE_POINT_CACHE_VERSION,
                        triple['resolution_kind'],
                        triple['component_key'],
                        triple['input_fingerprint'],
                    ),
                ).fetchall()
                self.assertIn('USING INDEX', plan[0]['detail'])
                self.assertNotIn('SCAN', plan[0]['detail'])

            second = PropertyResolver()
            second.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                second,
                '_resolve_boiling_point_uncached',
                side_effect=AssertionError('boiling cache was not checked first'),
            ), patch.object(
                second,
                '_resolve_melting_point_uncached',
                side_effect=AssertionError('melting cache was not checked first'),
            ), patch.object(
                second,
                '_resolve_triple_point_uncached',
                side_effect=AssertionError('triple cache was not checked first'),
            ):
                loaded_tb = second.resolve_boiling_point(
                    'C2H5OH', props, allow_online=False
                )
                loaded_tm = second.resolve_melting_point(
                    'C2H5OH', props, allow_online=False
                )
                loaded_triple = second.resolve_triple_point(
                    'C2H5OH', props, allow_online=False
                )
            self.assertEqual(loaded_tb, expected_tb)
            self.assertEqual(loaded_tm, expected_tm)
            self.assertEqual(loaded_triple, expected_triple)

    def test_phase_point_rows_expire_after_30_days(self):
        props = self.fixture_props()
        expected = self.fixture_tm()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            first = PropertyResolver()
            first.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                first,
                '_resolve_melting_point_uncached',
                return_value=expected,
            ):
                first.resolve_melting_point(
                    'C2H5OH', props, allow_online=False,
                )
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    UPDATE resolved_phase_point_cache
                    SET updated_at_utc = datetime('now', '-31 days')
                    """
                )

            replacement = PropertyResolutionResult(
                value=160.0,
                source='fixture',
                method='rebuilt_after_ttl',
                quality=0.95,
                notes='expired row was ignored',
            )
            second = PropertyResolver()
            second.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                second,
                '_resolve_melting_point_uncached',
                return_value=replacement,
            ) as build:
                actual = second.resolve_melting_point(
                    'C2H5OH', props, allow_online=False,
                )
            self.assertEqual(build.call_count, 1)
            self.assertEqual(actual, replacement)

    def test_melting_point_input_contract_and_mode_invalidate_rows(self):
        props = self.fixture_props()
        expected = self.fixture_tm()
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = (
                Path(directory, 'saturation.sqlite')
            )
            with patch.object(
                resolver,
                '_resolve_melting_point_uncached',
                return_value=expected,
            ) as build:
                resolver.resolve_melting_point(
                    'C2H5OH', props, allow_online=False
                )
                changed = dict(props)
                changed['Tm'] = 160.0
                resolver.resolve_melting_point(
                    'C2H5OH', changed, allow_online=False
                )
                changed_source = dict(props)
                changed_source['property_sources'] = {
                    **props['property_sources'],
                    'Tm': {
                        **props['property_sources']['Tm'],
                        'quality': 0.94,
                    },
                }
                resolver.resolve_melting_point(
                    'C2H5OH', changed_source, allow_online=False
                )
                resolver.resolve_melting_point(
                    'C2H5OH', props, allow_online=True
                )
            self.assertEqual(build.call_count, 4)

    def test_online_attempt_state_retries_transient_fallbacks(self):
        props = {
            'symbol': 'XRETRY',
            'name': 'online retry fixture',
            'formula': 'C2H6O',
            'CAS': '999-99-9',
            'MW': 46.0,
            'smiles': 'CCO',
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = path

            def transient(_identifier):
                raise LookupError('temporary provider failure')

            with patch.object(
                resolver,
                '_fetch_phase_change_pubchem',
                side_effect=transient,
            ), patch.object(
                resolver,
                '_fetch_phase_change_nist',
                side_effect=transient,
            ):
                fallback = resolver.resolve_boiling_point(
                    'XRETRY', props, allow_online=True
                )
            self.assertEqual(fallback.method, 'nannoolal_tb')
            self.assertFalse(path.exists())
            self.assertFalse(resolver._resolved_phase_point_cache)

            with patch.object(
                resolver,
                '_fetch_phase_change_pubchem',
                return_value={
                    'Tb': 400.0,
                    '_sources': {'Tb': 'pubchem'},
                },
            ), patch.object(
                resolver,
                '_fetch_phase_change_nist',
                return_value=None,
            ):
                recovered = resolver.resolve_boiling_point(
                    'XRETRY', props, allow_online=True
                )
            self.assertEqual(recovered.value, 400.0)
            self.assertEqual(recovered.method, 'pubchem')
            with sqlite3.connect(path) as connection:
                row = connection.execute(
                    'SELECT online_attempt_state, Tb_value '
                    'FROM resolved_phase_point_cache'
                ).fetchone()
            self.assertEqual(row, (
                OnlineAttemptState.COMPLETE_WITH_DATA.value,
                400.0,
            ))

    def test_online_attempt_states_cache_negative_and_not_needed_results(self):
        props = {
            'symbol': 'XNEG',
            'name': 'complete negative fixture',
            'formula': 'C2H6O',
            'CAS': '999-99-8',
            'MW': 46.0,
            'smiles': 'CCO',
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                resolver, '_fetch_phase_change_pubchem', return_value=None
            ), patch.object(
                resolver, '_fetch_phase_change_nist', return_value=None
            ):
                estimated = resolver.resolve_boiling_point(
                    'XNEG', props, allow_online=True
                )
            self.assertEqual(estimated.method, 'nannoolal_tb')
            with sqlite3.connect(path) as connection:
                state = connection.execute(
                    'SELECT online_attempt_state '
                    'FROM resolved_phase_point_cache'
                ).fetchone()[0]
            self.assertEqual(
                state,
                OnlineAttemptState.COMPLETE_NO_DATA.value,
            )

            hard_props = {
                **props,
                'CAS': '999-99-7',
                'Tb': 390.0,
                'property_sources': {
                    'Tb': {
                        'source': 'local',
                        'method': 'hard_fixture_tb',
                        'quality': 0.98,
                    },
                },
            }
            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                side_effect=AssertionError('online provider was not needed'),
            ):
                hard = resolver.resolve_boiling_point(
                    'XNEG-HARD', hard_props, allow_online=True
                )
            self.assertEqual(hard.method, 'hard_fixture_tb')
            with sqlite3.connect(path) as connection:
                state = connection.execute(
                    "SELECT online_attempt_state "
                    "FROM resolved_phase_point_cache "
                    "WHERE component_key = 'cas:999-99-7'"
                ).fetchone()[0]
            self.assertEqual(state, OnlineAttemptState.NOT_NEEDED.value)

    def test_offline_estimate_records_not_attempted(self):
        props = {
            'name': 'offline estimate fixture',
            'CAS': '999-99-4',
            'formula': 'C2H6O',
            'MW': 46.0,
            'smiles': 'CCO',
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = path
            result = resolver.resolve_boiling_point(
                'XOFFLINE', props, allow_online=False
            )
            self.assertEqual(result.method, 'nannoolal_tb')
            with sqlite3.connect(path) as connection:
                state = connection.execute(
                    'SELECT online_attempt_state '
                    'FROM resolved_phase_point_cache'
                ).fetchone()[0]
            self.assertEqual(
                state,
                OnlineAttemptState.NOT_ATTEMPTED.value,
            )

    def test_partial_transient_phase_payload_does_not_cache_missing_hvap(self):
        props = {
            'name': 'partial phase fixture',
            'CAS': '999-99-6',
        }
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.CACHE_DIR = Path(directory, 'source_cache')
            identifiers = resolver._identifier_candidates('XPARTIAL', props)
            cache_key = f"hvap_tb_v2_{'|'.join(identifiers)}"
            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                return_value={
                    'Tb': 400.0,
                    '_online_attempt_state': (
                        OnlineAttemptState.TRANSIENT_FAILURE.value
                    ),
                },
            ):
                self.assertIsNone(
                    resolver.get_hvap_online('XPARTIAL', props)
                )
            self.assertIsNone(resolver._get_cache(cache_key))

            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                return_value={
                    'Hvap': 42.0,
                    '_sources': {'Hvap': 'fixture'},
                    '_online_attempt_state': (
                        OnlineAttemptState.COMPLETE_WITH_DATA.value
                    ),
                },
            ):
                self.assertEqual(
                    resolver.get_hvap_online('XPARTIAL', props),
                    42.0,
                )
            self.assertEqual(resolver._get_cache(cache_key)['Hvap'], 42.0)

    def test_online_attempt_state_not_attempted_requires_retry(self):
        resolver = PropertyResolver()
        with resolver._online_attempt_scope(True) as attempt:
            self.assertIsNone(resolver._fetch_phase_change_online('', {}))
        self.assertEqual(
            attempt.state,
            OnlineAttemptState.NOT_ATTEMPTED,
        )
        self.assertFalse(resolver._online_attempt_is_persistable(
            True,
            attempt.state,
        ))

    def test_transient_state_propagates_to_enclosing_cache_builds(self):
        resolver = PropertyResolver()
        with resolver._online_attempt_scope(True) as outer:
            with resolver._online_attempt_scope(True) as inner:
                resolver._record_online_attempt_state(
                    OnlineAttemptState.TRANSIENT_FAILURE
                )
        self.assertEqual(inner.state, OnlineAttemptState.TRANSIENT_FAILURE)
        self.assertEqual(outer.state, OnlineAttemptState.TRANSIENT_FAILURE)

    def test_confirmed_coolprop_triple_cache_records_not_needed(self):
        props = {
            'CAS': '1333-74-0',
            'name': 'hydrogen',
            'Tm': 13.99,
            'property_sources': {
                'Tm': {
                    'source': 'local',
                    'method': 'experimental_hydrogen_tm',
                    'quality': 0.98,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            first = PropertyResolver()
            first.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                first,
                '_fetch_phase_change_online',
                side_effect=AssertionError('confirmed CoolProp probed online'),
            ):
                triple = first.resolve_triple_point(
                    'hydrogen', props, allow_online=True
                )
            self.assertEqual(triple['Tt'].method, 'coolprop_HEOS_triple_point')
            self.assertEqual(triple['Tt'].quality, 0.995)
            with sqlite3.connect(path) as connection:
                state = connection.execute(
                    "SELECT online_attempt_state "
                    "FROM resolved_phase_point_cache "
                    "WHERE resolution_kind = 'triple_point'"
                ).fetchone()[0]
            self.assertEqual(state, OnlineAttemptState.NOT_NEEDED.value)

    def test_provisional_coolprop_triple_caches_completed_online_absence(self):
        props = {'CAS': '7440-59-7', 'name': 'helium'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            first = PropertyResolver()
            first.SATURATION_PROPERTIES_CACHE_PATH = path

            def complete_negative(*_args, **_kwargs):
                first._record_online_attempt_state(
                    OnlineAttemptState.COMPLETE_NO_DATA
                )
                return None

            with patch.object(
                first,
                '_fetch_phase_change_online',
                side_effect=complete_negative,
            ):
                triple = first.resolve_triple_point(
                    'helium', props, allow_online=True
                )
            self.assertEqual(triple['Tt'].quality, 0.89)
            with sqlite3.connect(path) as connection:
                row = connection.execute(
                    "SELECT online_attempt_state, Tt_quality "
                    "FROM resolved_phase_point_cache "
                    "WHERE resolution_kind = 'triple_point'"
                ).fetchone()
            self.assertEqual(row, (
                OnlineAttemptState.COMPLETE_NO_DATA.value,
                0.89,
            ))

            second = PropertyResolver()
            second.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                second,
                '_fetch_phase_change_online',
                side_effect=AssertionError('complete negative was reprobed'),
            ):
                loaded = second.resolve_triple_point(
                    'helium', props, allow_online=True
                )
            self.assertEqual(loaded, triple)

    def test_provisional_coolprop_triple_retries_transient_corroboration(self):
        props = {'CAS': '7440-59-7', 'name': 'helium'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = path

            def transient(*_args, **_kwargs):
                resolver._record_online_attempt_state(
                    OnlineAttemptState.TRANSIENT_FAILURE
                )
                return None

            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                side_effect=transient,
            ):
                provisional = resolver.resolve_triple_point(
                    'helium', props, allow_online=True
                )
            self.assertEqual(provisional['Tt'].quality, 0.89)
            self.assertFalse(path.exists())

            def conflicting(*_args, **_kwargs):
                resolver._record_online_attempt_state(
                    OnlineAttemptState.COMPLETE_WITH_DATA
                )
                return {
                    'Tt': 3.0,
                    'Pt': 0.08,
                    '_sources': {'Tt': 'online_fixture', 'Pt': 'online_fixture'},
                    '_qualities': {'Tt': 0.96, 'Pt': 0.94},
                }

            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                side_effect=conflicting,
            ):
                recovered = resolver.resolve_triple_point(
                    'helium', props, allow_online=True
                )
            self.assertEqual(recovered['Tt'].value, 3.0)
            self.assertEqual(recovered['Tt'].method, 'online_fixture')

    def test_pfd_phase_override_is_memory_only(self):
        props = {
            'name': 'PFD boiling fixture',
            'CAS': '999-99-5',
            'Tb': 410.0,
            'property_sources': {
                'Tb': {
                    'source': 'pfd',
                    'method': 'pfd_component_override',
                    'quality': 1.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = path
            result = resolver.resolve_boiling_point(
                'PFD-TB', props, allow_online=True
            )
            self.assertEqual(result.value, 410.0)
            self.assertEqual(result.method, 'pfd_component_override')
            self.assertTrue(resolver._resolved_phase_point_cache)
            self.assertFalse(path.exists())

    def test_future_estimation_inputs_and_modes_invalidate_rows(self):
        props = self.fixture_props()
        expected = self.fixture_triple()
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = (
                Path(directory, 'saturation.sqlite')
            )
            with patch.object(
                resolver,
                '_resolve_triple_point_uncached',
                return_value=expected,
            ) as build:
                resolver.resolve_triple_point(
                    'C2H5OH', props, allow_online=False
                )
                for field_name, value in (
                    ('Tm', 160.0),
                    ('Tb', 352.0),
                    ('Tc', 515.0),
                    ('Hfus', 5.2),
                ):
                    changed = dict(props)
                    changed[field_name] = value
                    if field_name == 'Tm':
                        changed['property_sources'] = {
                            **props['property_sources'],
                            'Tm': {
                                'source': 'provided',
                                'method': 'pfd_component_override',
                                'quality': 1.0,
                            },
                        }
                    resolver.resolve_triple_point(
                        'C2H5OH', changed, allow_online=False
                    )
                resolver.resolve_triple_point(
                    'C2H5OH', props, allow_online=True
                )
            self.assertEqual(build.call_count, 6)

    def test_canonical_psat_backfills_and_reuses_missing_triple_pressure(self):
        props = {
            'symbol': 'XPT',
            'name': 'triple pressure backfill fixture',
            'CAS': '999-98-1',
            'Tm': 245.0,
            'Tt': 250.0,
            'Tc': 500.0,
            'Pc': 40.0,
            'property_sources': {
                'Tm': {
                    'source': 'local',
                    'method': 'fixture_tm',
                    'quality': 0.95,
                },
                'Tt': {
                    'source': 'local',
                    'method': 'fixture_tt',
                    'quality': 0.92,
                    'notes': 'selected triple temperature',
                },
            },
        }
        partial = {
            'Tt': PropertyResolutionResult(
                value=250.0,
                source='local',
                method='fixture_tt',
                quality=0.92,
                notes='selected triple temperature',
            ),
            'Pt': PropertyResolutionResult(
                value=None,
                source='missing',
                method='none',
                quality=0.0,
                notes='Pt unavailable',
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            first = PropertyResolver()
            first.SATURATION_PROPERTIES_CACHE_PATH = path

            def complete_negative(*_args, **_kwargs):
                first._record_online_attempt_state(
                    OnlineAttemptState.COMPLETE_NO_DATA
                )
                return partial

            with patch.object(
                first,
                '_resolve_triple_point_uncached',
                side_effect=complete_negative,
            ):
                unresolved = first.resolve_triple_point(
                    'XPT', props, allow_online=True
                )
            self.assertIsNone(unresolved['Pt'].value)

            runtime = SimpleNamespace(
                curve=SimpleNamespace(
                    P_critical_bar=40.0,
                    covers_temperature=lambda T: 250.0 <= T <= 500.0,
                ),
                evaluator=lambda _T: PropertyResolutionResult(
                    value=0.0123,
                    source='calculated',
                    method='deep_fixture_completion',
                    quality=0.71,
                    notes='local deep-vacuum quality',
                ),
            )
            cache_key = first._canonical_vapor_pressure_cache_key(
                'XPT', props, True, 1.0e-3
            )
            derived = first._maybe_backfill_triple_pressure(
                'XPT',
                props,
                runtime,
                cache_key=cache_key,
                allow_online=True,
            )
            self.assertIsNotNone(derived)
            self.assertEqual(derived.value, 0.0123)
            self.assertEqual(derived.quality, 0.71)

            with sqlite3.connect(path) as connection:
                primary = connection.execute(
                    """
                    SELECT Pt_value, online_attempt_state
                    FROM resolved_phase_point_cache
                    WHERE resolution_kind = 'triple_point'
                    """
                ).fetchone()
                backfill = connection.execute(
                    """
                    SELECT Pt_value, Pt_quality, psat_method
                    FROM canonical_triple_pressure_backfill
                    """
                ).fetchone()
            self.assertEqual(primary, (
                None,
                OnlineAttemptState.COMPLETE_NO_DATA.value,
            ))
            self.assertEqual(backfill, (
                0.0123,
                0.71,
                'deep_fixture_completion',
            ))

            second = PropertyResolver()
            second.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                second,
                '_resolve_triple_point_uncached',
                side_effect=AssertionError(
                    'complete-no-data triple providers were probed again'
                ),
            ):
                loaded = second.resolve_triple_point(
                    'XPT', props, allow_online=True
                )
            self.assertEqual(loaded['Tt'], partial['Tt'])
            self.assertEqual(loaded['Pt'].value, 0.0123)
            self.assertEqual(
                loaded['Pt'].method,
                'canonical_psat_at_triple_temperature',
            )
            self.assertEqual(loaded['Pt'].quality, 0.71)
            self.assertIn(
                'canonical Psat cache version=',
                loaded['Pt'].notes,
            )

    def test_reported_triple_pressure_precedes_cached_psat_backfill(self):
        props = {
            'symbol': 'XPT2',
            'name': 'triple pressure priority fixture',
            'CAS': '999-98-2',
            'Tm': 245.0,
            'Tt': 250.0,
            'Tc': 500.0,
            'Pc': 40.0,
            'property_sources': {
                'Tm': {
                    'source': 'local',
                    'method': 'fixture_tm',
                    'quality': 0.95,
                },
                'Tt': {
                    'source': 'local',
                    'method': 'fixture_tt',
                    'quality': 0.92,
                },
            },
        }
        Tt = PropertyResolutionResult(
            value=250.0,
            source='local',
            method='fixture_tt',
            quality=0.92,
            notes='',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = path
            Tt = resolver._source_result_for_value(props, 'Tt', units='K')
            self.assertIsNotNone(Tt)
            resolver._store_canonical_triple_pressure_backfill(
                'XPT2',
                props,
                triple_temperature=Tt,
                psat_result=PropertyResolutionResult(
                    value=0.0123,
                    source='calculated',
                    method='fixture_completion',
                    quality=0.70,
                    notes='',
                ),
                allow_online=True,
                canonical_input_fingerprint='fixture-canonical-fingerprint',
            )
            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                return_value={
                    'Pt': 0.02,
                    '_sources': {'Pt': 'reported_fixture'},
                    '_qualities': {'Pt': 0.94},
                    '_notes': {'Pt': 'reported triple pressure'},
                },
            ):
                resolved = resolver.resolve_triple_point(
                    'XPT2', props, allow_online=True
                )
            self.assertEqual(resolved['Pt'].value, 0.02)
            self.assertEqual(resolved['Pt'].method, 'reported_fixture')
            self.assertEqual(resolved['Pt'].quality, 0.94)

    def test_transient_triple_probe_retries_ahead_of_cached_psat_backfill(self):
        props = {
            'symbol': 'XPT3',
            'name': 'transient triple pressure fixture',
            'CAS': '999-98-3',
            'Tm': 245.0,
            'Tt': 250.0,
            'property_sources': {
                'Tm': {
                    'source': 'local',
                    'method': 'fixture_tm',
                    'quality': 0.95,
                },
                'Tt': {
                    'source': 'local',
                    'method': 'fixture_tt',
                    'quality': 0.92,
                },
            },
        }
        Tt = PropertyResolutionResult(
            value=250.0,
            source='local',
            method='fixture_tt',
            quality=0.92,
            notes='',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = path
            Tt = resolver._source_result_for_value(props, 'Tt', units='K')
            self.assertIsNotNone(Tt)
            resolver._store_canonical_triple_pressure_backfill(
                'XPT3',
                props,
                triple_temperature=Tt,
                psat_result=PropertyResolutionResult(
                    value=0.0123,
                    source='calculated',
                    method='fixture_completion',
                    quality=0.70,
                    notes='',
                ),
                allow_online=True,
                canonical_input_fingerprint='fixture-canonical-fingerprint',
            )

            def transient(*_args, **_kwargs):
                resolver._record_online_attempt_state(
                    OnlineAttemptState.TRANSIENT_FAILURE
                )
                return None

            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                side_effect=transient,
            ):
                fallback = resolver.resolve_triple_point(
                    'XPT3', props, allow_online=True
                )
            self.assertEqual(
                fallback['Pt'].method,
                'canonical_psat_at_triple_temperature',
            )
            with sqlite3.connect(path) as connection:
                selected_count = connection.execute(
                    "SELECT count(*) FROM resolved_phase_point_cache "
                    "WHERE resolution_kind = 'triple_point'"
                ).fetchone()[0]
            self.assertEqual(selected_count, 0)

            def reported(*_args, **_kwargs):
                resolver._record_online_attempt_state(
                    OnlineAttemptState.COMPLETE_WITH_DATA
                )
                return {
                    'Pt': 0.02,
                    '_sources': {'Pt': 'reported_fixture'},
                    '_qualities': {'Pt': 0.94},
                }

            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                side_effect=reported,
            ):
                recovered = resolver.resolve_triple_point(
                    'XPT3', props, allow_online=True
                )
            self.assertEqual(recovered['Pt'].value, 0.02)
            self.assertEqual(recovered['Pt'].method, 'reported_fixture')

    def test_live_benzophenone_ttriple_backfills_pt_from_canonical_psat(self):
        from chemical_properties import ChemicalDatabase

        component = ChemicalDatabase(enable_online=False).get(
            'benzophenone',
            fetch_online=False,
        )
        self.assertIsNotNone(component)
        props = component.to_dict()
        props['Tt'] = None
        props['Pt'] = None
        props['property_sources'].pop('Tt', None)
        props['property_sources'].pop('Pt', None)

        with tempfile.TemporaryDirectory() as directory:
            first = PropertyResolver()
            first.CACHE_DIR = Path(directory, 'source_cache')
            first.SATURATION_PROPERTIES_CACHE_PATH = Path(
                directory,
                'saturation.sqlite',
            )
            first.CANONICAL_PSAT_CACHE_PATH = Path(
                directory,
                'canonical.sqlite',
            )
            live_triple = run_optional_live_provider(
                first,
                lambda: first.resolve_triple_point(
                    component.symbol,
                    props,
                    allow_online=True,
                ),
                label='live benzophenone triple-point lookup',
            )
            if not live_triple.completed:
                return
            triple = live_triple.value
            self.assertIsNotNone(triple['Tt'].value)
            self.assertAlmostEqual(
                triple['Tt'].value,
                321.03,
                delta=0.1,
            )
            self.assertIsNone(triple['Pt'].value)
            props['Tt'] = triple['Tt'].value
            props['property_sources']['Tt'] = {
                'source': triple['Tt'].source,
                'method': triple['Tt'].method,
                'quality': triple['Tt'].quality,
                'notes': triple['Tt'].notes,
            }

            live_psat = run_optional_live_provider(
                first,
                lambda: first.resolve_vapor_pressure(
                    component.symbol,
                    400.0,
                    props,
                    allow_online=True,
                ),
                label='live benzophenone canonical Psat construction',
            )
            if not live_psat.completed:
                return
            self.assertGreater(live_psat.value.value, 0.0)

            second = PropertyResolver()
            second.CACHE_DIR = first.CACHE_DIR
            second.SATURATION_PROPERTIES_CACHE_PATH = (
                first.SATURATION_PROPERTIES_CACHE_PATH
            )
            second.CANONICAL_PSAT_CACHE_PATH = first.CANONICAL_PSAT_CACHE_PATH
            with patch(
                'property_resolution.online_phase_change.urllib.request.urlopen',
                side_effect=AssertionError(
                    'completed live provider response caused another HTTP probe'
                ),
            ):
                cached = second.resolve_triple_point(
                    component.symbol,
                    props,
                    allow_online=True,
                )
            self.assertEqual(cached['Tt'], triple['Tt'])
            self.assertGreater(cached['Pt'].value, 0.0)
            self.assertEqual(
                cached['Pt'].method,
                'canonical_psat_at_triple_temperature',
            )
            with sqlite3.connect(
                first.SATURATION_PROPERTIES_CACHE_PATH
            ) as connection:
                count = connection.execute(
                    'SELECT count(*) '
                    'FROM canonical_triple_pressure_backfill'
                ).fetchone()[0]
            self.assertEqual(count, 1)


if __name__ == '__main__':
    unittest.main()
