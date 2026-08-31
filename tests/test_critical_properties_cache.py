import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from property_resolution.common import OnlineAttemptState, PropertyResolutionResult
from property_resolution.resolver import PropertyResolver
from property_resolution.vapor_pressure_adapter import (
    DIRECT_PSAT_INPUT_METHODS,
    PsatCanonicalizationAdapter,
    PsatCanonicalizationInputs,
)
from property_resolution.vapor_pressure_canonical import (
    PsatHandoffRequirement,
    PsatPriority,
    PsatSegment,
    PsatSegmentType,
)


class CriticalPropertiesCacheTests(unittest.TestCase):
    @staticmethod
    def fixture_props():
        return {
            'symbol': 'C4H10O',
            'name': '2-Butanol',
            'formula': 'C4H10O',
            'CAS': '78-92-2',
            'MW': 74.123,
            'Tb': 372.88356662483557,
            'property_sources': {
                'Tb': {
                    'source': 'local',
                    'method': 'perry_normal_boiling_point',
                    'quality': 0.98,
                    'notes': 'Perry fixture',
                },
            },
        }

    @staticmethod
    def fixture_results():
        values = {
            'Tc': 535.9,
            'Pc': 41.885,
            'Vc': 269.0,
            'Zc': 0.253,
            'omega': 0.577,
        }
        return {
            name: PropertyResolutionResult(
                value=value,
                source='local',
                method=f'fixture_{name}',
                quality=0.97,
                notes=f'complete {name} provenance',
            )
            for name, value in values.items()
        }

    def test_persists_and_reloads_complete_versioned_result_contract(self):
        props = self.fixture_props()
        expected = self.fixture_results()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'critical_properties.sqlite')
            first = PropertyResolver()
            first.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                first,
                '_resolve_critical_properties_uncached',
                return_value=expected,
            ) as build:
                actual = first.resolve_critical_properties(
                    'C4H10O',
                    props,
                    allow_online=False,
                    allow_estimation=True,
                )
            self.assertEqual(build.call_count, 1)
            self.assertEqual(actual, expected)

            with sqlite3.connect(path) as connection:
                connection.row_factory = sqlite3.Row
                row = connection.execute(
                    'SELECT * FROM resolved_critical_properties_cache'
                ).fetchone()
                self.assertEqual(row['cache_version'], first.CRITICAL_PROPERTIES_CACHE_VERSION)
                self.assertEqual(row['component_key'], 'cas:78-92-2')
                self.assertEqual(row['cas'], '78-92-2')
                self.assertEqual(row['component_name'], '2-Butanol')
                self.assertEqual(row['allow_online'], 0)
                self.assertEqual(row['allow_estimation'], 1)
                self.assertEqual(
                    row['online_attempt_state'],
                    OnlineAttemptState.NOT_NEEDED.value,
                )
                self.assertTrue(row['created_at_utc'].endswith('Z'))
                self.assertTrue(row['updated_at_utc'].endswith('Z'))
                for name, result in expected.items():
                    self.assertEqual(row[f'{name}_value'], result.value)
                    self.assertEqual(row[f'{name}_source'], result.source)
                    self.assertEqual(row[f'{name}_method'], result.method)
                    self.assertEqual(row[f'{name}_quality'], result.quality)
                    self.assertEqual(row[f'{name}_notes'], result.notes)
                payload = json.loads(row['results_json'])
                metadata = json.loads(row['input_metadata_json'])
                self.assertEqual(payload['Tc']['method'], 'fixture_Tc')
                self.assertEqual(
                    metadata['properties']['property_sources']['Tb']['notes'],
                    'Perry fixture',
                )
                self.assertEqual(metadata['units']['Pc'], 'bar')
                plan = connection.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT * FROM resolved_critical_properties_cache
                    WHERE cache_version = ?
                      AND component_key = ?
                      AND input_fingerprint = ?
                    """,
                    (
                        first.CRITICAL_PROPERTIES_CACHE_VERSION,
                        row['component_key'],
                        row['input_fingerprint'],
                    ),
                ).fetchall()
                self.assertIn('USING INDEX', plan[0]['detail'])
                self.assertNotIn('SCAN', plan[0]['detail'])

            second = PropertyResolver()
            second.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                second,
                '_resolve_critical_properties_uncached',
                side_effect=AssertionError('persistent cache was not checked first'),
            ):
                loaded = second.resolve_critical_properties(
                    'C4H10O',
                    props,
                    allow_online=False,
                    allow_estimation=True,
                )
            self.assertEqual(loaded, expected)

    def test_critical_rows_expire_after_30_days(self):
        props = self.fixture_props()
        expected = self.fixture_results()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'saturation.sqlite')
            first = PropertyResolver()
            first.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                first,
                '_resolve_critical_properties_uncached',
                return_value=expected,
            ):
                first.resolve_critical_properties(
                    'C4H10O', props, allow_online=False,
                )
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    UPDATE resolved_critical_properties_cache
                    SET updated_at_utc = datetime('now', '-31 days')
                    """
                )

            replacement = dict(expected)
            replacement['Tc'] = PropertyResolutionResult(
                540.0, 'fixture', 'rebuilt_after_ttl', 0.95, '',
            )
            second = PropertyResolver()
            second.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                second,
                '_resolve_critical_properties_uncached',
                return_value=replacement,
            ) as build:
                actual = second.resolve_critical_properties(
                    'C4H10O', props, allow_online=False,
                )
            self.assertEqual(build.call_count, 1)
            self.assertEqual(actual, replacement)

    def test_missing_tb_is_resolved_before_critical_cache_fingerprint_and_build(self):
        props = {
            'symbol': 'XTB',
            'name': 'critical Tb dependency fixture',
            'formula': 'C7H8',
            'MW': 92.14,
        }
        expected = self.fixture_results()
        missing = PropertyResolutionResult(None, 'missing', 'none', 0.0, '')
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = Path(
                directory,
                'saturation.sqlite',
            )
            captured = {}

            def build(_symbol, current, **_options):
                captured.update(current)
                return expected

            with patch.object(
                resolver,
                'resolve_triple_point',
                return_value={'Tt': missing, 'Pt': missing},
            ) as triple, patch.object(
                resolver,
                'resolve_boiling_point',
                return_value=PropertyResolutionResult(
                    383.75,
                    'local',
                    'fixture_resolved_tb',
                    0.97,
                    'source-backed fixture Tb',
                ),
            ) as boiling, patch.object(
                resolver,
                '_resolve_critical_properties_uncached',
                side_effect=build,
            ):
                actual = resolver.resolve_critical_properties(
                    'XTB',
                    props,
                    allow_online=False,
                    allow_estimation=True,
                )

            self.assertEqual(actual, expected)
            self.assertEqual(triple.call_count, 1)
            self.assertEqual(boiling.call_count, 1)
            self.assertEqual(captured['Tb'], 383.75)
            self.assertEqual(
                captured['property_sources']['Tb']['method'],
                'fixture_resolved_tb',
            )
            with sqlite3.connect(resolver.SATURATION_PROPERTIES_CACHE_PATH) as connection:
                metadata = json.loads(connection.execute(
                    'SELECT input_metadata_json FROM resolved_critical_properties_cache'
                ).fetchone()[0])
            self.assertEqual(metadata['properties']['Tb'], 383.75)

    def test_transient_online_critical_fallback_is_not_cached(self):
        props = self.fixture_props()
        expected = self.fixture_results()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'critical_properties.sqlite')
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = path

            def transient_build(*_args, **_kwargs):
                resolver._record_online_attempt_state(
                    OnlineAttemptState.TRANSIENT_FAILURE
                )
                return expected

            with patch.object(
                resolver,
                '_resolve_critical_properties_uncached',
                side_effect=transient_build,
            ):
                first = resolver.resolve_critical_properties(
                    'C4H10O', props, allow_online=True
                )
            self.assertEqual(first, expected)
            self.assertFalse(path.exists())
            self.assertFalse(resolver._resolved_critical_properties_cache)

            def complete_build(*_args, **_kwargs):
                resolver._record_online_attempt_state(
                    OnlineAttemptState.COMPLETE_NO_DATA
                )
                return expected

            with patch.object(
                resolver,
                '_resolve_critical_properties_uncached',
                side_effect=complete_build,
            ):
                second = resolver.resolve_critical_properties(
                    'C4H10O', props, allow_online=True
                )
            self.assertEqual(second, expected)
            with sqlite3.connect(path) as connection:
                state = connection.execute(
                    'SELECT online_attempt_state '
                    'FROM resolved_critical_properties_cache'
                ).fetchone()[0]
            self.assertEqual(
                state,
                OnlineAttemptState.COMPLETE_NO_DATA.value,
            )

    def test_online_critical_cache_records_not_needed(self):
        props = self.fixture_props()
        expected = self.fixture_results()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'critical_properties.sqlite')
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                resolver,
                '_resolve_critical_properties_uncached',
                return_value=expected,
            ):
                actual = resolver.resolve_critical_properties(
                    'C4H10O', props, allow_online=True
                )
            self.assertEqual(actual, expected)
            with sqlite3.connect(path) as connection:
                state = connection.execute(
                    'SELECT online_attempt_state '
                    'FROM resolved_critical_properties_cache'
                ).fetchone()[0]
            self.assertEqual(state, OnlineAttemptState.NOT_NEEDED.value)

    def test_pfd_critical_override_is_memory_only(self):
        props = self.fixture_props()
        props['Tc'] = 540.0
        props['property_sources'] = {
            **props['property_sources'],
            'Tc': {
                'source': 'pfd',
                'method': 'pfd_component_override',
                'quality': 1.0,
            },
        }
        expected = self.fixture_results()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'critical_properties.sqlite')
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                resolver,
                '_resolve_critical_properties_uncached',
                return_value=expected,
            ):
                actual = resolver.resolve_critical_properties(
                    'C4H10O', props, allow_online=True
                )
            self.assertEqual(actual, expected)
            self.assertTrue(resolver._resolved_critical_properties_cache)
            self.assertFalse(path.exists())

    @staticmethod
    def hexafluorobenzene_props(*, omega=None, omega_quality=None, omega_method=None):
        props = {
            'symbol': 'C6F6',
            'name': 'Hexafluorobenzene',
            'CAS': '392-56-3',
            'formula': 'C6F6',
            'MW': 186.056,
            'Tb': 353.4,
            'Tc': 516.7,
            'Pc': 32.8,
            'Vc': 337.1,
            'Zc': 0.257,
            'property_sources': {
                'Tb': {
                    'source': 'online',
                    'method': 'nist_phase_change',
                    'quality': 0.93,
                },
                **{
                    name: {
                        'source': 'local',
                        'method': 'acs_jced_5b00571_table1',
                        'quality': 0.99,
                    }
                    for name in ('Tc', 'Pc', 'Vc', 'Zc')
                },
            },
        }
        if omega is not None:
            props['omega'] = omega
            props['property_sources']['omega'] = {
                'source': 'provided',
                'method': omega_method or 'hydrated_omega',
                'quality': omega_quality,
            }
        return props

    def test_psat_definition_omega_competes_by_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            for existing_quality, existing_method, expected_method in (
                (None, None, 'psat_definition_at_Tr_0_7'),
                (0.91, 'effective_critical', 'psat_definition_at_Tr_0_7'),
                (0.93, 'equal_quality_hydrated_omega', 'equal_quality_hydrated_omega'),
                (0.96, 'higher_quality_hydrated_omega', 'higher_quality_hydrated_omega'),
                (1.00, 'pfd_component_override', 'pfd_component_override'),
            ):
                with self.subTest(
                    existing_quality=existing_quality,
                    existing_method=existing_method,
                ):
                    resolver = PropertyResolver()
                    resolver.SATURATION_PROPERTIES_CACHE_PATH = Path(
                        directory,
                        f'{existing_method or "missing"}.sqlite',
                    )
                    props = self.hexafluorobenzene_props(
                        omega=(0.390378 if existing_quality is not None else None),
                        omega_quality=existing_quality,
                        omega_method=existing_method,
                    )
                    result = resolver.resolve_critical_properties(
                        'C6F6',
                        props,
                        allow_online=False,
                        allow_estimation=True,
                    )['omega']

                    self.assertEqual(result.method, expected_method)
                    if expected_method == 'psat_definition_at_Tr_0_7':
                        self.assertAlmostEqual(result.value, 0.39684728727052443)
                        self.assertAlmostEqual(result.quality, 0.93)
                        self.assertIn('property dependencies=()', result.notes)
                        self.assertIn('quality penalty=0.02', result.notes)
                    else:
                        self.assertEqual(result.value, 0.390378)
                        self.assertEqual(result.quality, existing_quality)

    def test_thiodiglycol_rejected_perry_cannot_supply_omega(self):
        props = {
            'symbol': 'C4H10O2S',
            'name': 'thiodiglycol',
            'CAS': '111-48-8',
            'formula': 'C4H10O2S',
            'MW': 122.186,
            'smiles': 'OCCSCCO',
            'Tb': 555.15,
            'property_sources': {
                'Tb': {
                    'source': 'online',
                    'method': 'pubchem',
                    'quality': 0.95,
                },
                'smiles': {
                    'source': 'local',
                    'method': 'explicit',
                    'quality': 0.99,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = Path(
                directory,
                'critical.sqlite',
            )
            results = resolver.resolve_critical_properties(
                'C4H10O2S',
                props,
                allow_online=False,
                allow_estimation=True,
            )
        omega = results['omega']
        self.assertEqual(omega.method, 'lee_kesler')
        self.assertAlmostEqual(omega.value, 0.95313787069723)
        self.assertAlmostEqual(omega.quality, 0.64)
        self.assertNotEqual(omega.method, 'psat_definition_at_Tr_0_7')
        self.assertLess(omega.value, 1.0)

    def test_thiodiglycol_soft_criticals_reject_unvalidated_psat_omega(self):
        props = {
            'symbol': 'C4H10O2S',
            'name': 'thiodiglycol',
            'CAS': '111-48-8',
            'formula': 'C4H10O2S',
            'MW': 122.186,
            'smiles': 'OCCSCCO',
            'property_sources': {
                'smiles': {
                    'source': 'local',
                    'method': 'explicit',
                    'quality': 0.99,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = Path(
                directory,
                'critical.sqlite',
            )
            results = resolver.resolve_critical_properties(
                'C4H10O2S',
                props,
                allow_online=False,
                allow_estimation=True,
            )

        self.assertEqual(results['Tc'].method, 'nannoolal_tc')
        self.assertEqual(results['Pc'].method, 'nannoolal_pc')
        self.assertEqual(results['omega'].method, 'lee_kesler')
        self.assertAlmostEqual(results['omega'].value, 0.95313787069723)
        self.assertLess(results['omega'].value, 1.0)

    def test_soft_criticals_require_hard_tb_validated_psat_for_omega(self):
        resolver = PropertyResolver()
        props = {
            'symbol': 'XSOFT',
            'name': 'soft critical fixture',
            'Tc': 500.0,
            'Pc': 40.0,
            'property_sources': {
                'Tc': {
                    'source': 'estimated',
                    'method': 'nannoolal_tc',
                    'quality': 0.85,
                },
                'Pc': {
                    'source': 'estimated',
                    'method': 'nannoolal_pc',
                    'quality': 0.80,
                },
            },
        }
        critical = {
            name: PropertyResolutionResult(
                props[name],
                'estimated',
                f'nannoolal_{name.lower()}',
                props['property_sources'][name]['quality'],
                '',
            )
            for name in ('Tc', 'Pc')
        }

        def segment(status):
            return PsatSegment(
                source='fixture',
                method='fixture_psat',
                segment_type=PsatSegmentType.PINNED,
                priority=int(PsatPriority.PERRY_2_10),
                T_min=300.0,
                T_max=400.0,
                ln_pressure_function=lambda T: 0.01 * T - 4.0,
                derivative_function=lambda _T: 0.01,
                quality=0.95 if status == 'hard_tb_validated' else 0.90,
                metadata={
                    'property_dependencies': (),
                    'tb_validation_required': True,
                    'tb_validation_status': status,
                    'quality_basis': (
                        'hard_tb_validation'
                        if status == 'hard_tb_validated'
                        else 'standalone_unvalidated'
                    ),
                },
            )

        for status, admitted in (
            ('validation_unavailable', False),
            ('hard_tb_validated', True),
        ):
            with self.subTest(status=status), patch.object(
                PsatCanonicalizationAdapter,
                'collect_inputs',
                return_value=PsatCanonicalizationInputs(
                    segments=(segment(status),),
                ),
            ):
                result = resolver._independent_psat_omega_result(
                    'XSOFT',
                    props,
                    critical,
                )
            self.assertEqual(result is not None, admitted)

    def test_psat_omega_quality_uses_shared_validation_tier(self):
        resolver = PropertyResolver()
        props = {
            'symbol': 'XTIER',
            'name': 'omega tier fixture',
            'Tc': 500.0,
            'Pc': 40.0,
            'property_sources': {
                name: {
                    'source': 'local',
                    'method': 'hard_fixture',
                    'quality': 0.99,
                }
                for name in ('Tc', 'Pc')
            },
        }
        critical = {
            name: PropertyResolutionResult(
                props[name], 'local', 'hard_fixture', 0.99, ''
            )
            for name in ('Tc', 'Pc')
        }

        def direct_segment(quality, status, basis):
            return PsatSegment(
                source='Perry 9th Table 2-10',
                method='perry_2_10_vapor_pressure',
                segment_type=PsatSegmentType.PINNED,
                priority=int(PsatPriority.PERRY_2_10),
                T_min=300.0,
                T_max=400.0,
                ln_pressure_function=lambda T: 0.01 * T - 4.0,
                derivative_function=lambda _T: 0.01,
                quality=quality,
                handoff_requirement=(
                    PsatHandoffRequirement.NONE
                    if status == 'hard_tb_validated'
                    else PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP_IF_AVAILABLE
                ),
                metadata={
                    'property_dependencies': (),
                    'tb_validation_required': True,
                    'tb_validation_status': status,
                    'quality_basis': basis,
                    'higher_preference_overlap_quality': 0.93,
                },
            )

        standalone = direct_segment(
            0.90,
            'validation_unavailable',
            'standalone_unvalidated',
        )
        hard = direct_segment(
            0.95,
            'hard_tb_validated',
            'hard_tb_validation',
        )
        higher = PsatSegment(
            source='higher reference',
            method='higher_reference',
            segment_type=PsatSegmentType.PINNED,
            priority=int(PsatPriority.PERRY_2_8),
            T_min=300.0,
            T_max=340.0,
            ln_pressure_function=lambda T: 0.01 * T - 4.0,
            derivative_function=lambda _T: 0.01,
            quality=0.98,
            metadata={
                'property_dependencies': (),
                'tb_validation_required': False,
                'tb_validation_status': 'exempt_priority',
                'quality_basis': 'exempt_priority',
            },
        )
        cases = (
            ((standalone,), 0.88),
            ((standalone, higher), 0.91),
            ((hard,), 0.93),
        )
        for segments, expected_quality in cases:
            with self.subTest(expected_quality=expected_quality), patch.object(
                PsatCanonicalizationAdapter,
                'collect_inputs',
                return_value=PsatCanonicalizationInputs(segments=segments),
            ):
                result = resolver._independent_psat_omega_result(
                    'XTIER', props, critical
                )
                self.assertIsNotNone(result)
                self.assertEqual(
                    result.method,
                    'psat_definition_at_Tr_0_7',
                )
                self.assertAlmostEqual(result.quality, expected_quality)

    def test_omega_recovery_rejects_omega_dependent_or_untagged_psat(self):
        resolver = PropertyResolver()
        props = self.hexafluorobenzene_props()
        base_results = {
            name: PropertyResolutionResult(
                props[name],
                'local',
                'fixture',
                0.99,
                '',
            )
            for name in ('Tc', 'Pc')
        }

        for metadata in (
            {'property_dependencies': ('omega',)},
            {},
        ):
            with self.subTest(metadata=metadata):
                segment = PsatSegment(
                    source='fixture',
                    method='fixture_psat',
                    segment_type=PsatSegmentType.PINNED,
                    priority=900,
                    T_min=300.0,
                    T_max=400.0,
                    ln_pressure_function=lambda _T: 0.25,
                    derivative_function=lambda _T: 0.01,
                    quality=0.99,
                    metadata=metadata,
                )
                with patch.object(
                    PsatCanonicalizationAdapter,
                    'collect_inputs',
                    return_value=PsatCanonicalizationInputs(
                        segments=(segment,),
                    ),
                ):
                    result = resolver._independent_psat_omega_result(
                        'C6F6',
                        props,
                        base_results,
                    )
                self.assertIsNone(result)

    def test_direct_psat_segments_declare_property_dependencies(self):
        props = self.hexafluorobenzene_props()
        inputs = PsatCanonicalizationAdapter(
            props,
            input_methods=DIRECT_PSAT_INPUT_METHODS,
        ).collect_inputs(
            T_min=200.0,
            T_critical=props['Tc'],
            P_critical_bar=props['Pc'],
            T_boiling=props['Tb'],
        )
        self.assertTrue(inputs.segments)
        self.assertTrue(all(
            item.metadata.get('property_dependencies') is not None
            for item in inputs.segments
        ))
        self.assertTrue(all(
            'omega' not in item.metadata['property_dependencies']
            for item in inputs.segments
        ))

    def test_declared_dependency_and_temperature_range_control_omega_recovery(self):
        resolver = PropertyResolver()
        props = {
            'symbol': 'XPSAT',
            'name': 'dependency fixture',
            'Tc': 500.0,
            'Pc': 50.0,
            'property_sources': {
                name: {
                    'source': 'local',
                    'method': f'fixture_{name}',
                    'quality': 0.99,
                }
                for name in ('Tc', 'Pc')
            },
            'property_correlations': {
                'Psat': {
                    'equation': 'exp_poly_x',
                    'coefficients': {'A': 0.0, 'B': 0.1},
                    'Tmin_K': 300.0,
                    'Tmax_K': 400.0,
                    'quality': 0.97,
                    'source': 'fixture direct Psat',
                    'property_dependencies': ['omega'],
                },
            },
        }
        results = {
            name: PropertyResolutionResult(
                props[name],
                'local',
                f'fixture_{name}',
                0.99,
                '',
            )
            for name in ('Tc', 'Pc')
        }

        self.assertIsNone(resolver._independent_psat_omega_result(
            'XPSAT', props, results
        ))

        independent = json.loads(json.dumps(props))
        independent['property_correlations']['Psat']['property_dependencies'] = []
        recovered = resolver._independent_psat_omega_result(
            'XPSAT', independent, results
        )
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.method, 'psat_definition_at_Tr_0_7')
        self.assertAlmostEqual(recovered.quality, 0.95)
        expected_ln_pressure = 0.1 * ((350.0 - 298.15) / 100.0)
        expected_omega = -(
            expected_ln_pressure - math.log(50.0)
        ) / math.log(10.0) - 1.0
        self.assertAlmostEqual(recovered.value, expected_omega)

        outside = json.loads(json.dumps(independent))
        outside['property_correlations']['Psat']['Tmax_K'] = 340.0
        self.assertIsNone(resolver._independent_psat_omega_result(
            'XPSAT', outside, results
        ))

    def test_input_contract_and_cache_version_prevent_incompatible_reuse(self):
        props = self.fixture_props()
        expected = self.fixture_results()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'critical_properties.sqlite')
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                resolver,
                '_resolve_critical_properties_uncached',
                return_value=expected,
            ) as build:
                resolver.resolve_critical_properties(
                    'C4H10O', props, allow_online=False
                )
                changed = dict(props)
                changed['Tc'] = 540.0
                changed['property_sources'] = {
                    **props['property_sources'],
                    'Tc': {
                        'source': 'pfd',
                        'method': 'pfd_component_override',
                        'quality': 1.0,
                    },
                }
                resolver.resolve_critical_properties(
                    'C4H10O', changed, allow_online=False
                )
                resolver.resolve_critical_properties(
                    'C4H10O', props, allow_online=True
                )
            self.assertEqual(build.call_count, 3)

            stale = PropertyResolver()
            stale.SATURATION_PROPERTIES_CACHE_PATH = path
            stale.CRITICAL_PROPERTIES_CACHE_VERSION += 1
            with patch.object(
                stale,
                '_resolve_critical_properties_uncached',
                return_value=expected,
            ) as rebuild:
                stale.resolve_critical_properties(
                    'C4H10O', props, allow_online=False
                )
            self.assertEqual(rebuild.call_count, 1)


if __name__ == '__main__':
    unittest.main()
