import math
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from chemical_properties import ChemicalDatabase
from property_resolution.resolver import PropertyResolver
from property_resolution.vapor_pressure import _CanonicalVaporPressureRuntime
from property_resolution.vapor_pressure_adapter import (
    PsatCanonicalizationAdapter,
)
from property_resolution.common import (
    AntoineCoefficients,
    OnlineAttemptState,
    PropertyResolutionError,
)
from property_resolution.vapor_pressure_canonical import PsatSegmentType


class CanonicalVaporPressureResolutionTests(unittest.TestCase):
    def assertClose(self, actual, expected, *, rel=1e-8, abs_tol=1e-12):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol),
            f"{actual!r} != {expected!r}",
        )

    @staticmethod
    def local_props(identifier):
        component = ChemicalDatabase(enable_online=False).get(
            identifier,
            fetch_online=False,
        )
        if component is None:
            raise AssertionError(f"Missing local test component {identifier!r}")
        return component, component.to_dict()

    def test_canonicalizes_once_and_reuses_cached_evaluator(self):
        resolver = PropertyResolver()
        _component, props = self.local_props("ethanol")
        original_build = resolver._build_canonical_vapor_pressure_runtime

        with tempfile.TemporaryDirectory() as tmpdir:
            resolver.CANONICAL_PSAT_CACHE_PATH = (
                Path(tmpdir) / "canonical_psat.sqlite"
            )
            with patch.object(
                resolver,
                "_build_canonical_vapor_pressure_runtime",
                wraps=original_build,
            ) as build:
                first = resolver.resolve_vapor_pressure(
                    "ethanol",
                    300.0,
                    props,
                    allow_online=False,
                )
                coefficients = resolver.resolve_vapor_pressure_coefficients(
                    "ethanol",
                    props,
                    allow_online=False,
                )
                second = resolver.resolve_vapor_pressure(
                    "ethanol",
                    310.0,
                    props,
                    allow_online=False,
                )

        self.assertEqual(build.call_count, 1)
        self.assertEqual(len(resolver._canonical_vapor_pressure_curves), 1)
        self.assertEqual(coefficients.shape, (13,))
        self.assertGreater(first.value, 0.0)
        self.assertGreater(second.value, first.value)
        (
            A, B, C, D, E, F, G, H, Tc, inverse_power,
            supercritical_slope, T_min, lower_continuation_slope,
        ) = coefficients
        expected_ln_pressure = (
            A
            + B / 300.0
            + C * math.log(300.0)
            + D * 300.0
            + E * 300.0**2
            + F * 300.0**5
            + G * 300.0**3
        )
        if H != 0.0:
            expected_ln_pressure += H * (
                (300.0 / Tc) ** int(inverse_power) - 1.0
            )
        self.assertClose(first.value, math.exp(expected_ln_pressure), rel=1e-11)
        runtime = next(iter(resolver._canonical_vapor_pressure_curves.values()))
        self.assertClose(
            supercritical_slope,
            runtime.curve.supercritical_slope,
            rel=1e-12,
        )
        self.assertClose(T_min, runtime.curve.T_min, rel=1e-12)
        self.assertClose(
            lower_continuation_slope,
            runtime.curve.lower_continuation_slope,
            rel=1e-12,
        )

        coefficients[0] = math.nan
        fresh = resolver.resolve_vapor_pressure_coefficients(
            "ethanol",
            props,
            allow_online=False,
        )
        self.assertTrue(math.isfinite(fresh[0]))

    def test_completed_canonical_curve_persists_pt_from_local_quality_at_tt(self):
        _component, props = self.local_props('ethanol')
        props['Pt'] = None
        props['property_sources'].pop('Pt', None)
        with tempfile.TemporaryDirectory() as tmpdir:
            saturation_path = Path(tmpdir) / 'saturation.sqlite'
            canonical_path = Path(tmpdir) / 'canonical.sqlite'
            first = PropertyResolver()
            first.SATURATION_PROPERTIES_CACHE_PATH = saturation_path
            first.CANONICAL_PSAT_CACHE_PATH = canonical_path
            first.resolve_vapor_pressure(
                'ethanol',
                300.0,
                props,
                allow_online=False,
            )
            with sqlite3.connect(saturation_path) as connection:
                stored = connection.execute(
                    """
                    SELECT Pt_value, Pt_quality,
                           psat_source, psat_method, psat_quality
                    FROM canonical_triple_pressure_backfill
                    """
                ).fetchone()
            self.assertIsNotNone(stored)
            Pt_value, Pt_quality, source, method, psat_quality = stored
            self.assertGreater(Pt_value, 0.0)
            self.assertEqual(source, 'local')
            self.assertEqual(method, 'coolprop_HEOS_psat')
            self.assertEqual(Pt_quality, psat_quality)
            self.assertEqual(Pt_quality, 0.995)

            second = PropertyResolver()
            second.SATURATION_PROPERTIES_CACHE_PATH = saturation_path
            second.CANONICAL_PSAT_CACHE_PATH = canonical_path
            with patch.object(
                second,
                '_build_canonical_vapor_pressure_runtime',
                side_effect=AssertionError('persistent canonical curve missed'),
            ):
                second.resolve_vapor_pressure(
                    'ethanol',
                    310.0,
                    props,
                    allow_online=False,
                )
            with sqlite3.connect(saturation_path) as connection:
                count = connection.execute(
                    'SELECT count(*) '
                    'FROM canonical_triple_pressure_backfill'
                ).fetchone()[0]
            self.assertEqual(count, 1)

    def test_triple_pressure_backfill_expires_after_30_days(self):
        _component, props = self.local_props('ethanol')
        props['Pt'] = None
        props['property_sources'].pop('Pt', None)
        with tempfile.TemporaryDirectory() as directory:
            saturation_path = Path(directory) / 'saturation.sqlite'
            canonical_path = Path(directory) / 'canonical.sqlite'
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = saturation_path
            resolver.CANONICAL_PSAT_CACHE_PATH = canonical_path
            resolver.resolve_vapor_pressure(
                'ethanol', 300.0, props, allow_online=False,
            )
            triple = resolver.resolve_triple_point(
                'ethanol', props, allow_online=False,
            )
            component_key = resolver._phase_point_component_identity(
                'ethanol', props,
            )[0]
            with sqlite3.connect(saturation_path) as connection:
                connection.execute(
                    """
                    UPDATE canonical_triple_pressure_backfill
                    SET created_at_utc = datetime('now', '-31 days')
                    """
                )

            fresh = PropertyResolver()
            fresh.SATURATION_PROPERTIES_CACHE_PATH = saturation_path
            fresh.CANONICAL_PSAT_CACHE_PATH = canonical_path
            self.assertIsNone(fresh._load_canonical_triple_pressure_backfill(
                component_key=component_key,
                triple_temperature=triple['Tt'],
                allow_online=False,
            ))

    def test_transient_antoine_lookup_retries_in_same_resolver(self):
        resolver = PropertyResolver()
        props = {'name': 'retry Antoine', 'CAS': '999-99-3'}
        with tempfile.TemporaryDirectory() as tmpdir:
            resolver.CACHE_DIR = Path(tmpdir) / 'source_cache'
            with patch.object(
                resolver,
                '_fetch_antoine_nist',
                side_effect=LookupError('temporary NIST failure'),
            ), patch.object(
                resolver,
                '_fetch_antoine_pubchem',
                side_effect=LookupError('temporary PubChem failure'),
            ):
                self.assertIsNone(resolver.get_antoine_online(
                    'XRETRY-ANTOINE', props=props
                ))

            expected = AntoineCoefficients(
                A=4.0,
                B=1200.0,
                C=-40.0,
                T_min=250.0,
                T_max=450.0,
                source='fixture',
                P_units='bar',
            )
            with patch.object(
                resolver,
                '_fetch_antoine_nist',
                return_value=expected,
            ), patch.object(
                resolver,
                '_fetch_antoine_pubchem',
                return_value=None,
            ):
                recovered = resolver.get_antoine_online(
                    'XRETRY-ANTOINE', props=props
                )
            self.assertEqual(recovered, expected)

    def test_canonical_runtime_persists_full_versioned_curve_contract(self):
        _component, props = self.local_props("ethanol")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "canonical_psat.sqlite"
            first_resolver = PropertyResolver()
            first_resolver.CANONICAL_PSAT_CACHE_PATH = path
            first = first_resolver.resolve_vapor_pressure_coefficients(
                "ethanol",
                props,
                allow_online=False,
            )
            first_runtime = next(iter(
                first_resolver._canonical_vapor_pressure_curves.values()
            ))

            with sqlite3.connect(path) as connection:
                connection.row_factory = sqlite3.Row
                row = connection.execute(
                    "SELECT * FROM canonical_psat_cache"
                ).fetchone()
                self.assertEqual(
                    connection.execute("PRAGMA user_version").fetchone()[0],
                    first_resolver.CANONICAL_PSAT_CACHE_VERSION,
                )

            self.assertIsNotNone(row)
            self.assertEqual(
                row["cache_version"],
                first_resolver.CANONICAL_PSAT_CACHE_VERSION,
            )
            self.assertEqual(row["cas"], "64-17-5")
            self.assertEqual(
                row["online_attempt_state"],
                OnlineAttemptState.NOT_NEEDED.value,
            )
            self.assertTrue(row["created_at_utc"].endswith("Z"))
            self.assertTrue(row["updated_at_utc"].endswith("Z"))
            self.assertEqual(row["form"], first_runtime.curve.form.value)
            self.assertEqual(len(json.loads(row["provenance_json"])), len(
                first_runtime.curve.provenance
            ))
            self.assertEqual(
                json.loads(row["diagnostics_json"])["sample_count"],
                first_runtime.curve.diagnostics.sample_count,
            )
            self.assertEqual(
                json.loads(row["metadata_json"])["domain_selection"]["basis"],
                first_runtime.curve.metadata["domain_selection"]["basis"],
            )

            second_resolver = PropertyResolver()
            second_resolver.CANONICAL_PSAT_CACHE_PATH = path
            with patch.object(
                second_resolver,
                "_build_canonical_vapor_pressure_runtime",
                side_effect=AssertionError("persistent cache was not used"),
            ):
                second = second_resolver.resolve_vapor_pressure_coefficients(
                    "ethanol",
                    props,
                    allow_online=False,
                )
            self.assertEqual(tuple(first), tuple(second))
            second_runtime = next(iter(
                second_resolver._canonical_vapor_pressure_curves.values()
            ))
            self.assertEqual(
                second_runtime.curve.provenance,
                first_runtime.curve.provenance,
            )
            self.assertEqual(
                second_runtime.curve.diagnostics,
                first_runtime.curve.diagnostics,
            )
            self.assertEqual(
                second_resolver._canonical_cache_json(
                    second_runtime.curve.metadata
                ),
                first_resolver._canonical_cache_json(
                    first_runtime.curve.metadata
                ),
            )

            versioned_resolver = PropertyResolver()
            versioned_resolver.CANONICAL_PSAT_CACHE_PATH = path
            versioned_resolver.CANONICAL_PSAT_CACHE_VERSION = (
                first_resolver.CANONICAL_PSAT_CACHE_VERSION + 1
            )
            original_build = (
                versioned_resolver._build_canonical_vapor_pressure_runtime
            )
            with patch.object(
                versioned_resolver,
                "_build_canonical_vapor_pressure_runtime",
                wraps=original_build,
            ) as build:
                versioned_resolver.resolve_vapor_pressure_coefficients(
                    "ethanol",
                    props,
                    allow_online=False,
                )
            self.assertEqual(build.call_count, 1)
            with sqlite3.connect(path) as connection:
                versions = {
                    item[0]
                    for item in connection.execute(
                        "SELECT cache_version FROM canonical_psat_cache"
                    )
                }
            self.assertEqual(versions, {
                first_resolver.CANONICAL_PSAT_CACHE_VERSION,
                versioned_resolver.CANONICAL_PSAT_CACHE_VERSION,
            })

    def test_canonical_psat_rows_expire_after_30_days(self):
        _component, props = self.local_props('ethanol')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'canonical.sqlite')
            first = PropertyResolver()
            first.CANONICAL_PSAT_CACHE_PATH = path
            first.resolve_vapor_pressure_coefficients(
                'ethanol', props, allow_online=False,
            )
            with sqlite3.connect(path) as connection:
                connection.execute(
                    """
                    UPDATE canonical_psat_cache
                    SET updated_at_utc = datetime('now', '-31 days')
                    """
                )

            second = PropertyResolver()
            second.CANONICAL_PSAT_CACHE_PATH = path
            original = second._build_canonical_vapor_pressure_runtime
            with patch.object(
                second,
                '_build_canonical_vapor_pressure_runtime',
                wraps=original,
            ) as build:
                second.resolve_vapor_pressure_coefficients(
                    'ethanol', props, allow_online=False,
                )
            self.assertEqual(build.call_count, 1)

    def test_transient_online_canonical_build_is_not_cached(self):
        _component, props = self.local_props('ethanol')
        with tempfile.TemporaryDirectory() as tmpdir:
            template = PropertyResolver()
            template.SATURATION_PROPERTIES_CACHE_PATH = (
                Path(tmpdir) / 'template_saturation.sqlite'
            )
            runtime = template._build_canonical_vapor_pressure_runtime(
                'ethanol',
                props,
                allow_online=False,
                minimum_pressure_bar=1.0e-7,
            )

            path = Path(tmpdir) / 'canonical_psat.sqlite'
            resolver = PropertyResolver()
            resolver.CANONICAL_PSAT_CACHE_PATH = path

            def transient_build(*_args, **_kwargs):
                resolver._record_online_attempt_state(
                    OnlineAttemptState.TRANSIENT_FAILURE
                )
                return runtime

            with patch.object(
                resolver,
                '_build_canonical_vapor_pressure_runtime',
                side_effect=transient_build,
            ):
                first = resolver.resolve_vapor_pressure_coefficients(
                    'ethanol', props, allow_online=True
                )
            self.assertEqual(tuple(first), tuple(runtime.coefficients))
            self.assertFalse(path.exists())
            self.assertFalse(resolver._canonical_vapor_pressure_curves)

            def complete_build(*_args, **_kwargs):
                resolver._record_online_attempt_state(
                    OnlineAttemptState.COMPLETE_WITH_DATA
                )
                return runtime

            with patch.object(
                resolver,
                '_build_canonical_vapor_pressure_runtime',
                side_effect=complete_build,
            ):
                second = resolver.resolve_vapor_pressure_coefficients(
                    'ethanol', props, allow_online=True
                )
            self.assertEqual(tuple(second), tuple(runtime.coefficients))
            with sqlite3.connect(path) as connection:
                state = connection.execute(
                    'SELECT online_attempt_state FROM canonical_psat_cache'
                ).fetchone()[0]
            self.assertEqual(
                state,
                OnlineAttemptState.COMPLETE_WITH_DATA.value,
            )

    def test_online_canonical_cache_records_not_needed(self):
        _component, props = self.local_props('ethanol')
        with tempfile.TemporaryDirectory() as tmpdir:
            template = PropertyResolver()
            template.SATURATION_PROPERTIES_CACHE_PATH = (
                Path(tmpdir) / 'template_saturation.sqlite'
            )
            runtime = template._build_canonical_vapor_pressure_runtime(
                'ethanol',
                props,
                allow_online=False,
                minimum_pressure_bar=1.0e-7,
            )
            path = Path(tmpdir) / 'canonical_psat.sqlite'
            resolver = PropertyResolver()
            resolver.CANONICAL_PSAT_CACHE_PATH = path
            with patch.object(
                resolver,
                '_build_canonical_vapor_pressure_runtime',
                return_value=runtime,
            ):
                resolver.resolve_vapor_pressure_coefficients(
                    'ethanol', props, allow_online=True
                )
            with sqlite3.connect(path) as connection:
                state = connection.execute(
                    'SELECT online_attempt_state FROM canonical_psat_cache'
                ).fetchone()[0]
            self.assertEqual(state, OnlineAttemptState.NOT_NEEDED.value)

    def test_canonical_completeness_uses_selected_tb_validation_metadata(self):
        _component, props = self.local_props('ethanol')
        with tempfile.TemporaryDirectory() as tmpdir:
            template = PropertyResolver()
            template.SATURATION_PROPERTIES_CACHE_PATH = (
                Path(tmpdir) / 'template_saturation.sqlite'
            )
            base_runtime = template._build_canonical_vapor_pressure_runtime(
                'ethanol',
                props,
                allow_online=False,
                minimum_pressure_bar=1.0e-7,
            )
            base_item = base_runtime.curve.provenance[0]

            def runtime_with(*metadata_items):
                provenance = tuple(
                    replace(
                        base_item,
                        source=f'fixture-{index}',
                        method=f'fixture-{index}',
                        segment_type=PsatSegmentType.PINNED.value,
                        metadata=metadata,
                    )
                    for index, metadata in enumerate(metadata_items)
                )
                curve = replace(
                    base_runtime.curve,
                    provenance=provenance,
                )
                return _CanonicalVaporPressureRuntime(
                    curve,
                    base_runtime.evaluator,
                )

            pending = {
                'tb_validation_required': True,
                'tb_validation_status': 'validation_unavailable',
                'quality_basis': 'standalone_unvalidated',
            }
            safe = {
                'tb_validation_required': False,
                'tb_validation_status': 'exempt_priority',
                'quality_basis': 'exempt_priority',
            }
            promoted = {
                'tb_validation_required': True,
                'tb_validation_status': 'validation_unavailable',
                'quality_basis': 'higher_preference_overlap',
            }
            hard = {
                'tb_validation_required': True,
                'tb_validation_status': 'hard_tb_validated',
                'quality_basis': 'hard_tb_validation',
            }
            cases = (
                ((pending,), OnlineAttemptState.NOT_ATTEMPTED),
                ((safe, pending), OnlineAttemptState.NOT_ATTEMPTED),
                ((promoted,), OnlineAttemptState.NOT_NEEDED),
                ((hard,), OnlineAttemptState.NOT_NEEDED),
                ((safe,), OnlineAttemptState.NOT_NEEDED),
            )
            for index, (metadata_items, expected_state) in enumerate(cases):
                with self.subTest(expected_state=expected_state):
                    resolver = PropertyResolver()
                    resolver.CANONICAL_PSAT_CACHE_PATH = Path(
                        tmpdir,
                        f'canonical-{index}.sqlite',
                    )
                    runtime = runtime_with(*metadata_items)
                    with patch.object(
                        resolver,
                        '_build_canonical_vapor_pressure_runtime',
                        return_value=runtime,
                    ):
                        resolver.resolve_vapor_pressure_coefficients(
                            'ethanol', props, allow_online=False
                        )
                    with sqlite3.connect(
                        resolver.CANONICAL_PSAT_CACHE_PATH
                    ) as connection:
                        state = connection.execute(
                            'SELECT online_attempt_state '
                            'FROM canonical_psat_cache'
                        ).fetchone()[0]
                    self.assertEqual(state, expected_state.value)

            for state in (
                OnlineAttemptState.COMPLETE_NO_DATA,
                OnlineAttemptState.COMPLETE_WITH_DATA,
            ):
                with self.subTest(online_state=state):
                    resolver = PropertyResolver()
                    resolver.CANONICAL_PSAT_CACHE_PATH = Path(
                        tmpdir,
                        f'online-{state.value}.sqlite',
                    )
                    runtime = runtime_with(pending)

                    def build(*_args, **_kwargs):
                        resolver._record_online_attempt_state(state)
                        return runtime

                    with patch.object(
                        resolver,
                        '_build_canonical_vapor_pressure_runtime',
                        side_effect=build,
                    ):
                        resolver.resolve_vapor_pressure_coefficients(
                            'ethanol', props, allow_online=True
                        )
                    with sqlite3.connect(
                        resolver.CANONICAL_PSAT_CACHE_PATH
                    ) as connection:
                        stored = connection.execute(
                            'SELECT online_attempt_state '
                            'FROM canonical_psat_cache'
                        ).fetchone()[0]
                    self.assertEqual(stored, state.value)

    def test_pfd_canonical_override_is_memory_only(self):
        _component, props = self.local_props('ethanol')
        props = dict(props)
        props['property_correlations'] = {
            **dict(props.get('property_correlations') or {}),
            'Psat': {
                'equation': 'canonical_af',
                'coefficients': {
                    'A': 1.0, 'B': -1000.0, 'C': 0.0,
                    'D': 0.0, 'E': 0.0, 'F': 0.0,
                },
                'Tmin_K': 150.0,
                'Tmax_K': 514.71,
                '_pfd_override': True,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            template = PropertyResolver()
            template.SATURATION_PROPERTIES_CACHE_PATH = (
                Path(tmpdir) / 'template_saturation.sqlite'
            )
            runtime = template._build_canonical_vapor_pressure_runtime(
                'ethanol',
                self.local_props('ethanol')[1],
                allow_online=False,
                minimum_pressure_bar=1.0e-7,
            )
            path = Path(tmpdir) / 'canonical_psat.sqlite'
            resolver = PropertyResolver()
            resolver.CANONICAL_PSAT_CACHE_PATH = path
            with patch.object(
                resolver,
                '_build_canonical_vapor_pressure_runtime',
                return_value=runtime,
            ):
                resolver.resolve_vapor_pressure_coefficients(
                    'ethanol', props, allow_online=True
                )
            self.assertTrue(resolver._canonical_vapor_pressure_curves)
            self.assertFalse(path.exists())

    def test_checked_evaluator_enforces_physical_temperature_domain(self):
        resolver = PropertyResolver()
        component, props = self.local_props("acetic acid")
        resolver.resolve_vapor_pressure(
            "acetic acid",
            300.0,
            props,
            allow_online=False,
        )
        runtime = next(iter(resolver._canonical_vapor_pressure_curves.values()))
        curve = runtime.curve

        self.assertClose(curve.T_min, component.Tm, rel=1e-9)
        self.assertClose(
            curve.pressure_bar(curve.T_critical),
            component.Pc,
            rel=1e-9,
        )
        with self.assertRaises(PropertyResolutionError):
            runtime.evaluator(curve.T_min - 1.0e-6)
        with self.assertRaises(PropertyResolutionError):
            runtime.evaluator(curve.T_critical + 1.0e-6)

    def test_sublimator_uses_triple_anchor_without_normal_boiling_point(self):
        resolver = PropertyResolver()
        component, props = self.local_props("CO2")

        coefficients = resolver.resolve_vapor_pressure_coefficients(
            "CO2",
            props,
            allow_online=False,
        )
        runtime = next(iter(resolver._canonical_vapor_pressure_curves.values()))
        curve = runtime.curve

        self.assertEqual(coefficients.shape, (13,))
        self.assertIsNone(component.Tb)
        self.assertIsNone(curve.T_boiling)
        self.assertClose(curve.T_min, component.Tt, rel=1e-12)
        self.assertClose(
            curve.pressure_bar(component.Tt),
            component.Pt,
            rel=1e-10,
        )
        self.assertClose(
            curve.pressure_bar(component.Tc),
            component.Pc,
            rel=1e-10,
        )

    def test_boiling_point_below_liquid_domain_is_not_a_fit_anchor(self):
        resolver = PropertyResolver()
        component, props = self.local_props("74-86-2")

        coefficients = resolver.resolve_vapor_pressure_coefficients(
            component.symbol,
            props,
            allow_online=False,
        )
        runtime = next(iter(resolver._canonical_vapor_pressure_curves.values()))
        curve = runtime.curve

        self.assertEqual(coefficients.shape, (13,))
        self.assertLess(component.Tb, component.Tm)
        self.assertClose(curve.T_min, component.Tm, rel=1e-12)
        self.assertIsNone(curve.T_boiling)
        self.assertClose(
            curve.pressure_bar(component.Tc),
            component.Pc,
            rel=1e-10,
        )

    def test_melting_point_domain_keeps_low_pressure_liquid_psat_available(self):
        resolver = PropertyResolver()
        component, props = self.local_props("C3H8O3")

        result = resolver.resolve_vapor_pressure(
            "C3H8O3",
            318.15,
            props,
            allow_online=False,
        )
        runtime = next(iter(resolver._canonical_vapor_pressure_curves.values()))
        curve = runtime.curve

        self.assertIsNone(component.Tt)
        self.assertClose(curve.T_min, component.Tm, rel=1e-12)
        self.assertEqual(
            curve.metadata["domain_selection"]["basis"],
            "melting_point",
        )
        self.assertLess(result.value, 0.001)
        self.assertGreater(result.value, 0.0)

    def test_reports_quality_from_source_slice_at_temperature(self):
        resolver = PropertyResolver()
        _component, props = self.local_props("acetic acid")
        resolver.resolve_vapor_pressure(
            "acetic acid",
            300.0,
            props,
            allow_online=False,
        )
        runtime = next(iter(resolver._canonical_vapor_pressure_curves.values()))
        provenance = runtime.curve.provenance
        quality_penalty = runtime.curve.metadata["fit_quality_penalty"]

        self.assertGreaterEqual(len(provenance), 2)
        self.assertGreater(
            len({round(item.quality, 12) for item in provenance}),
            1,
        )
        for item in provenance:
            temperature = 0.5 * (item.T_min + item.T_max)
            resolved = runtime.evaluator(temperature)
            self.assertEqual(resolved.source, item.source)
            self.assertEqual(resolved.method, item.method)
            self.assertClose(
                resolved.quality,
                max(0.0, item.quality - quality_penalty),
            )

    def test_fit_quality_penalty_is_shared_by_curve_and_all_point_results(self):
        resolver = PropertyResolver()
        props = {
            "Tc": 560.0,
            "Pc": 35.0,
            "omega": 0.25,
            "Tb": 355.0,
            "Hvap": 36.0,
            "property_sources": {
                field_name: {
                    "source": "provided",
                    "method": "pfd_component_override",
                    "quality": 1.0,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb", "Hvap")
            },
        }
        resolver.resolve_vapor_pressure(
            "XNO",
            300.0,
            props,
            allow_online=False,
        )
        runtime = next(iter(resolver._canonical_vapor_pressure_curves.values()))
        curve = runtime.curve
        penalty = curve.metadata["fit_quality_penalty"]
        original_quality = curve.metadata["fit_original_overall_quality"]

        self.assertGreater(penalty, 0.0)
        self.assertClose(
            curve.quality,
            max(0.0, original_quality - penalty),
        )
        self.assertTrue(curve.metadata["fit_warnings"])
        for item in curve.provenance:
            temperature = 0.5 * (item.T_min + item.T_max)
            resolved = runtime.evaluator(temperature)
            self.assertClose(
                resolved.quality,
                max(0.0, item.quality - penalty),
            )

    def test_physical_domain_controls_form_independently_of_pressure_fallback(self):
        resolver = PropertyResolver()
        _component, props = self.local_props("ethanol")
        ordinary = resolver.resolve_vapor_pressure_coefficients(
            "ethanol",
            props,
            allow_online=False,
        )
        ordinary_runtime = next(
            iter(resolver._canonical_vapor_pressure_curves.values())
        )
        deep_props = {
            **props,
            "_psat_minimum_pressure_bar": 1.0e-5,
        }
        deep = resolver.resolve_vapor_pressure_coefficients(
            "ethanol",
            deep_props,
            allow_online=False,
        )
        runtimes = tuple(resolver._canonical_vapor_pressure_curves.values())
        deep_runtime = min(runtimes, key=lambda item: item.curve.T_min)

        self.assertEqual(len(runtimes), 2)
        self.assertClose(
            deep_runtime.curve.T_min,
            ordinary_runtime.curve.T_min,
            rel=1e-12,
        )
        self.assertEqual(ordinary_runtime.curve.metadata[
            "domain_selection"
        ]["basis"], "triple_point")
        self.assertEqual(deep_runtime.curve.metadata[
            "domain_selection"
        ]["basis"], "triple_point")
        self.assertEqual(deep.shape, (13,))
        self.assertEqual(tuple(deep), tuple(ordinary))

    def test_broad_interval_uses_lower_ambrose_walton_omega(self):
        physical_omega_bound = PropertyResolver._canonical_broad_T_min(
            500.0,
            40.0,
            0.7,
            0.001,
        )
        minimum_omega_bound = PropertyResolver._canonical_broad_T_min(
            500.0,
            40.0,
            None,
            0.001,
        )

        self.assertGreater(physical_omega_bound, 0.0)
        self.assertLess(physical_omega_bound, 500.0)
        self.assertGreater(minimum_omega_bound, 0.0)
        self.assertLess(minimum_omega_bound, physical_omega_bound)

    def test_coefficients_preserve_direct_pfd_a_h_payload(self):
        resolver = PropertyResolver()
        Tc = 500.0
        Pc = 20.0
        B = -2000.0
        H = -0.001
        props = {
            "symbol": "X",
            "name": "Test fluid",
            "Tc": Tc,
            "Pc": Pc,
            "Tb": 300.0,
            "property_sources": {
                "Tc": {
                    "source": "provided",
                    "method": "pfd_component_override",
                    "quality": 1.0,
                },
                "Pc": {
                    "source": "provided",
                    "method": "pfd_component_override",
                    "quality": 1.0,
                },
                "Tb": {
                    "source": "provided",
                    "method": "pfd_component_override",
                    "quality": 1.0,
                },
            },
            "property_correlations": {
                "Psat": {
                    "_pfd_override": True,
                    "equation": "canonical_psat_ah",
                    "coefficients": {
                        "A": math.log(Pc) - B / Tc,
                        "B": B,
                        "C": 0.0,
                        "D": 0.0,
                        "E": 0.0,
                        "F": 0.0,
                        "G": 0.0,
                        "H": H,
                    },
                    "inverse_power": -3,
                    "Tmin_K": 1.0,
                    "Tmax_K": Tc,
                    "Tc_K": Tc,
                    "Pc_bar": Pc,
                    "Tb_K": 300.0,
                },
            },
        }

        coefficients = resolver.resolve_vapor_pressure_coefficients(
            "X",
            props,
            allow_online=False,
        )

        self.assertEqual(coefficients.shape, (13,))
        self.assertEqual(coefficients[7], H)
        self.assertEqual(coefficients[8], Tc)
        self.assertEqual(coefficients[9], -3.0)
        self.assertGreater(coefficients[10], 0.0)
        runtime = next(iter(resolver._canonical_vapor_pressure_curves.values()))
        self.assertEqual(coefficients[11], runtime.curve.T_min)
        self.assertGreater(coefficients[12], 0.0)

    def test_initialization_fetches_and_reuses_all_nist_antoine_rows(self):
        html = """
            <h2>Antoine Equation Parameters</h2>
            <table>
              <tr><td>280.0 to 340.0</td><td>5.1</td><td>1200.0</td><td>-40.0</td></tr>
              <tr><td>330.0 to 410.0</td><td>4.8</td><td>1100.0</td><td>-55.0</td></tr>
            </table>
        """
        rows = PropertyResolver._parse_nist_antoine_rows(html)
        self.assertEqual(len(rows), 2)
        self.assertClose(rows[0].C, 233.15)

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            resolver = PropertyResolver()
            resolver.CACHE_DIR = cache_dir
            component = {"symbol": "X", "name": "test fluid"}
            with patch(
                "property_resolution.vapor_pressure_adapter.ANTOINE_CACHE_DIR",
                cache_dir,
            ), patch.object(
                resolver,
                "_fetch_antoine_nist_rows",
                return_value=rows,
            ) as fetch:
                resolver._prime_nist_antoine_cache("test fluid", component)
                resolver._prime_nist_antoine_cache("test fluid", component)
                self.assertEqual(fetch.call_count, 1)

                cached_resolver = PropertyResolver()
                cached_resolver.CACHE_DIR = cache_dir
                with patch.object(
                    cached_resolver,
                    "_fetch_antoine_nist_rows",
                    side_effect=AssertionError(
                        "persistent NIST cache was not reused"
                    ),
                ):
                    cached_resolver._prime_nist_antoine_cache(
                        "test fluid",
                        component,
                    )

                adapter = PsatCanonicalizationAdapter(component)
                inputs = adapter.collect_inputs()

            nist_segments = [
                segment
                for segment in inputs.segments
                if segment.method == "cached_nist_antoine"
            ]
            self.assertEqual(len(nist_segments), 2)

    def test_live_resolver_no_longer_exposes_pointwise_fallback_chain(self):
        resolver = PropertyResolver()

        self.assertFalse(hasattr(resolver, "_resolve_vapor_pressure_pointwise"))
        self.assertFalse(hasattr(resolver, "_clausius_clapeyron"))
        self.assertFalse(hasattr(resolver, "_lee_kesler_psat"))
        self.assertFalse(hasattr(resolver, "_ambrose_walton_psat"))


if __name__ == "__main__":
    unittest.main()
