import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from unittest.mock import patch

from chemical_properties import ChemicalDatabase
from property_resolution import (
    AntoineCoefficients,
    CanonicalPsatForm,
    DEFAULT_PSAT_MINIMUM_PRESSURE_BAR,
    PropertyResolutionResult,
    PsatAdapterError,
    PsatBoundaryConditions,
    PsatCanonicalizationAdapter,
    PsatCanonicalizationError,
    PsatCanonicalizationInputs,
    PsatCanonicalizer,
    PsatCompletionCoordinator,
    PsatDerivativeBasis,
    PsatHandoffCoordinator,
    PsatHandoffRequirement,
    PsatEndpoint,
    PsatAnchorRegistry,
    PsatPriority,
    PsatSegment,
    PsatSegmentAssembler,
    PsatSegmentType,
    trim_psat_segment_to_validity,
    validate_antoine_boiling_point,
)
from property_resolution.runtime_cache import SQLiteJSONCache
from property_resolution.common import (
    CACHED_NIST_ANTOINE_PROFILE,
    DirectPsatConfidenceProfile,
    HIGH_QUALITY_ANTOINE_PROFILE,
    STANDARD_DIRECT_TABLE_PROFILE,
    admit_direct_psat_segment,
    validate_psat_boiling_point,
)
from pressure_standards import NORMAL_BOILING_PRESSURE_BAR
import property_resolution.vapor_pressure_adapter as vapor_pressure_adapter


from physical_constants import R_J_MOL_K


class VaporPressureCanonicalizationAdapterTests(unittest.TestCase):
    def test_trusted_other_antoine_uses_normalized_priority_550_source(self):
        adapter = PsatCanonicalizationAdapter(
            {
                "CAS": "104-76-7",
                "name": "2-ethyl-1-hexanol",
            }
        )
        inputs = vapor_pressure_adapter._collect_trusted_other_antoine_inputs(
            adapter,
            T_boiling=None,
        )
        self.assertEqual(inputs.metadata["trusted_other_antoine_row_count"], 1)
        self.assertEqual(len(inputs.segments), 1)
        segment = inputs.segments[0]
        self.assertEqual(segment.priority, PsatPriority.HIGH_QUALITY_ANTOINE)
        self.assertEqual(segment.method, "trusted_other_antoine")
        self.assertEqual(segment.source, "10.1016/j.fluid.2005.03.034")
        self.assertEqual((segment.T_min, segment.T_max), (353.2, 458.2))
        self.assertAlmostEqual(
            math.exp(segment.ln_pressure(400.0)),
            0.1457431733298181,
            places=12,
        )
        self.assertEqual(
            segment.metadata["data_file"],
            "data/trusted_other_antoine.json",
        )
        self.assertEqual(segment.metadata["source"]["table"], "Table 5")

    def test_provider_neutral_tb_validator_contract(self):
        target = NORMAL_BOILING_PRESSURE_BAR
        evaluations = []

        def pressure(value):
            evaluations.append(value)
            return target

        exact = validate_psat_boiling_point(pressure, 300.0, 400.0, 350.0)
        self.assertTrue(exact.accepted)
        self.assertEqual(exact.pressure_bar, target)
        self.assertEqual(evaluations, [350.0])

        for factor, accepted in ((1.019999, True), (1.020001, False)):
            with self.subTest(factor=factor):
                result = validate_psat_boiling_point(
                    lambda _T, factor=factor: target * factor,
                    300.0,
                    400.0,
                    350.0,
                )
                self.assertEqual(result.accepted, accepted)

        gross = validate_psat_boiling_point(lambda _T: 0.1, 300.0, 400.0, 350.0)
        self.assertFalse(gross.accepted)
        self.assertGreater(gross.relative_error, 0.80)

        for boiling in (298.9, 401.1):
            with self.subTest(outside=boiling):
                calls = []
                outside = validate_psat_boiling_point(
                    lambda T: calls.append(T),
                    300.0,
                    400.0,
                    boiling,
                )
                self.assertFalse(outside.covers_boiling_point)
                self.assertEqual(calls, [])

        endpoint = validate_psat_boiling_point(
            lambda _T: target,
            300.0,
            400.0,
            299.5,
        )
        self.assertTrue(endpoint.covers_boiling_point)
        self.assertTrue(endpoint.accepted)

        for invalid_pressure in (math.nan, math.inf, 0.0, -1.0, None):
            with self.subTest(invalid_pressure=invalid_pressure):
                invalid = validate_psat_boiling_point(
                    lambda _T, value=invalid_pressure: value,
                    300.0,
                    400.0,
                    350.0,
                )
                self.assertTrue(invalid.covers_boiling_point)
                self.assertFalse(invalid.accepted)
                self.assertIsNone(invalid.pressure_bar)

        for bounds in ((400.0, 300.0), (300.0, 300.0), (-1.0, 300.0)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                validate_psat_boiling_point(
                    lambda _T: target,
                    bounds[0],
                    bounds[1],
                    350.0,
                )

    def test_direct_psat_priority_boundary_and_confidence_profiles(self):
        self.assertEqual(
            (
                STANDARD_DIRECT_TABLE_PROFILE.standalone_quality,
                STANDARD_DIRECT_TABLE_PROFILE.higher_priority_overlap_quality,
                STANDARD_DIRECT_TABLE_PROFILE.hard_tb_quality,
            ),
            (0.90, 0.93, 0.95),
        )
        self.assertEqual(
            (
                CACHED_NIST_ANTOINE_PROFILE.standalone_quality,
                CACHED_NIST_ANTOINE_PROFILE.higher_priority_overlap_quality,
                CACHED_NIST_ANTOINE_PROFILE.hard_tb_quality,
            ),
            (0.87, 0.90, 0.95),
        )
        self.assertEqual(
            (
                HIGH_QUALITY_ANTOINE_PROFILE.standalone_quality,
                HIGH_QUALITY_ANTOINE_PROFILE.higher_priority_overlap_quality,
                HIGH_QUALITY_ANTOINE_PROFILE.hard_tb_quality,
            ),
            (0.95, 0.96, 0.97),
        )
        for profile in (
            STANDARD_DIRECT_TABLE_PROFILE,
            CACHED_NIST_ANTOINE_PROFILE,
            HIGH_QUALITY_ANTOINE_PROFILE,
        ):
            self.assertLessEqual(
                profile.standalone_quality,
                profile.higher_priority_overlap_quality,
            )
            self.assertLessEqual(
                profile.higher_priority_overlap_quality,
                profile.hard_tb_quality,
            )
        with self.assertRaises(ValueError):
            DirectPsatConfidenceProfile(0.95, 0.90, 0.97)

        for priority in (400, 500, 550, 599, 600, 700, 800, 900, 1000):
            with self.subTest(priority=priority):
                decision = admit_direct_psat_segment(
                    priority=priority,
                    intrinsic_quality=0.98,
                    confidence_profile=(
                        STANDARD_DIRECT_TABLE_PROFILE if priority < 600 else None
                    ),
                    qualified_Tb=350.0,
                    pressure_at_temperature=lambda _T: 0.1,
                    T_min=300.0,
                    T_max=400.0,
                    source_label="priority fixture",
                )
                if priority < 600:
                    self.assertFalse(decision.admitted)
                    self.assertEqual(
                        decision.validation_status,
                        "hard_tb_conflict",
                    )
                else:
                    self.assertTrue(decision.admitted)
                    self.assertEqual(
                        decision.validation_status,
                        "exempt_priority",
                    )
                    self.assertEqual(decision.selected_quality, 0.98)

    @staticmethod
    def pfd_component(equation, coefficients, **correlation_fields):
        return {
            "symbol": "X",
            "name": "PFD test component",
            "Tc": 500.0,
            "Pc": math.exp(3.0),
            "Tb": 350.0,
            "property_correlations": {
                "Psat": {
                    "equation": equation,
                    "coefficients": coefficients,
                    "quality": 1.0,
                    "_pfd_override": True,
                    **correlation_fields,
                }
            },
        }

    def test_shared_antoine_tb_validation_reports_coverage_and_residual(self):
        coefficients = AntoineCoefficients(
            A=3.974761287430068,
            B=1152.368,
            C=227.129,
            T_min=244.15,
            T_max=362.15,
        )

        accepted = validate_antoine_boiling_point(coefficients, 336.421)
        outside = validate_antoine_boiling_point(coefficients, 400.0)

        self.assertTrue(accepted.covers_boiling_point)
        self.assertTrue(accepted.accepted)
        self.assertIsNotNone(accepted.pressure_bar)
        self.assertLess(accepted.relative_error, 0.02)
        self.assertFalse(outside.covers_boiling_point)
        self.assertFalse(outside.accepted)
        self.assertIsNone(outside.pressure_bar)
        self.assertIsNone(outside.relative_error)

    def test_shared_antoine_tb_validation_rejects_pressure_mismatch(self):
        validation = validate_antoine_boiling_point(
            AntoineCoefficients(
                A=5.0,
                B=1000.0,
                C=200.0,
                T_min=250.0,
                T_max=400.0,
            ),
            350.0,
        )

        self.assertTrue(validation.covers_boiling_point)
        self.assertFalse(validation.accepted)
        self.assertGreater(validation.relative_error, 0.02)

    def test_chemicals_json_loads_triple_point_fields_with_provenance(self):
        payload = {
            "chemicals": {
                "TP": {
                    "name": "Triple point test",
                    "formula": "TP",
                    "Tc": 400.0,
                    "Tt": 123.45,
                    "Pt": 0.0042,
                    "Tm": 125.0,
                    "property_sources": {
                        "dataset": {
                            "source": "test dataset",
                            "method": "chemicals_json",
                            "quality": 0.97,
                        }
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chemicals.json"
            path.write_text(json.dumps(payload))
            component = ChemicalDatabase(
                db_path=path,
                enable_online=False,
            ).chemicals["TP"]

        self.assertEqual(component.Tt, 123.45)
        self.assertEqual(component.Pt, 0.0042)
        self.assertEqual(component.to_dict()["Tt"], 123.45)
        self.assertEqual(component.to_dict()["Pt"], 0.0042)
        self.assertEqual(component.property_sources["Tt"]["quality"], 0.97)
        self.assertEqual(component.property_sources["Pt"]["quality"], 0.97)
        selection = PsatCanonicalizationAdapter(component).resolve_domain()
        self.assertEqual(selection.T_min, 123.45)
        self.assertEqual(selection.basis, "triple_point")

    def test_triple_point_precedes_melting_point_and_pressure_floor(self):
        calls = []
        component = {
            "Tc": 514.71,
            "Tt": 159.05,
            "Tm": 160.0,
            "property_sources": {
                "Tt": {
                    "source": "CoolProp",
                    "method": "HEOS_triple_point",
                    "quality": 0.995,
                    "notes": "pure-fluid HEOS",
                }
            },
        }
        adapter = PsatCanonicalizationAdapter(
            component,
            psat_at_temperature=lambda item, temperature: calls.append(
                (item, temperature)
            ),
            tsat_at_pressure=lambda item, pressure: calls.append((item, pressure)),
        )

        selection = adapter.resolve_domain()

        self.assertEqual(selection.T_min, 159.05)
        self.assertEqual(selection.basis, "triple_point")
        self.assertEqual(selection.source, "CoolProp")
        self.assertEqual(selection.method, "HEOS_triple_point")
        self.assertEqual(selection.quality, 0.995)
        self.assertIn("pure-fluid HEOS", selection.notes)
        self.assertIsNone(selection.pressure_bar)
        self.assertEqual(selection.rejected_candidates, ())
        self.assertEqual(calls, [])

    def test_subtriple_boiling_candidate_selects_triple_point_anchor(self):
        component = {
            "symbol": "CO2",
            "CAS": "124-38-9",
            "Tb": 185.1,
            "Tt": 216.592,
            "Pt": 5.17964,
            "property_sources": {
                "Tt": {"quality": 0.995},
                "Pt": {"quality": 0.99},
            },
        }
        adapter = PsatCanonicalizationAdapter(component)

        selection = adapter.select_phase_change_anchor(
            T_boiling=component["Tb"],
            boiling_quality=0.10,
        )

        self.assertTrue(selection.uses_sublimation_anchor)
        self.assertIsNone(selection.T_boiling)
        self.assertEqual(selection.boiling_quality, 0.0)
        self.assertIsNotNone(selection.triple_anchor)
        self.assertEqual(selection.triple_anchor.temperature, component["Tt"])
        self.assertEqual(
            math.exp(selection.triple_anchor.ln_pressure),
            component["Pt"],
        )
        self.assertEqual(selection.triple_anchor.quality, 0.99)
        self.assertEqual(selection.triple_anchor.context["anchor_name"], "Tt")

        missing_normal = adapter.select_phase_change_anchor(
            T_boiling=None,
            boiling_quality=0.0,
        )
        self.assertTrue(missing_normal.uses_sublimation_anchor)
        self.assertIsNone(missing_normal.T_boiling)
        self.assertIsNotNone(missing_normal.triple_anchor)
        self.assertEqual(
            math.exp(missing_normal.triple_anchor.ln_pressure),
            component["Pt"],
        )

        ordinary = adapter.select_phase_change_anchor(
            T_boiling=250.0,
            boiling_quality=0.95,
        )
        self.assertFalse(ordinary.uses_sublimation_anchor)
        self.assertEqual(ordinary.T_boiling, 250.0)
        self.assertIsNone(ordinary.triple_anchor)

    def test_soft_triple_pair_sets_domain_without_changing_boiling_topology(self):
        component = {
            "symbol": "XSOFT-TRIPLE",
            "Tc": 500.0,
            "Pc": 40.0,
            "Tb": 240.0,
            "Tt": 250.0,
            "Pt": 1.5,
            "property_sources": {
                "Tb": {
                    "source": "local",
                    "method": "fixture_tb",
                    "quality": 0.95,
                },
                "Tt": {
                    "source": "local",
                    "method": "fixture_provisional_triple",
                    "quality": 0.89,
                },
                "Pt": {
                    "source": "local",
                    "method": "fixture_provisional_triple",
                    "quality": 0.89,
                },
            },
        }
        adapter = PsatCanonicalizationAdapter(
            component,
            tsat_at_pressure=lambda *_args: 200.0,
        )
        domain = adapter.resolve_domain(T_critical=500.0)
        selection = adapter.select_phase_change_anchor(
            T_boiling=240.0,
            boiling_quality=0.95,
        )
        self.assertEqual(domain.basis, "triple_point")
        self.assertEqual(domain.T_min, 250.0)
        self.assertEqual(selection.T_boiling, 240.0)
        self.assertFalse(selection.uses_sublimation_anchor)
        self.assertIsNone(selection.triple_anchor)

        selection_without_tb = adapter.select_phase_change_anchor(
            T_boiling=None,
            boiling_quality=0.0,
        )
        self.assertIsNone(selection_without_tb.T_boiling)
        self.assertFalse(selection_without_tb.uses_sublimation_anchor)
        self.assertIsNone(selection_without_tb.triple_anchor)

    def test_physical_domain_resolution_does_not_probe_psat(self):
        calls = []
        selection = PsatCanonicalizationAdapter(
            {
                "symbol": "ETOH",
                "name": "ethanol",
                "CAS": "64-17-5",
                "Tc": 514.7092848812961,
                "Tm": 160.0,
            },
            psat_at_temperature=lambda item, temperature: calls.append(
                (item, temperature)
            ),
        ).resolve_domain()

        self.assertEqual(selection.T_min, 160.0)
        self.assertEqual(selection.basis, "melting_point")
        self.assertEqual(selection.method, "Tm_field")
        self.assertIsNone(selection.pressure_bar)
        self.assertEqual(calls, [])

    def test_object_without_triple_point_uses_melting_point(self):
        component = SimpleNamespace(
            Tc=500.0,
            Tm=178.0,
            property_sources={
                "Tm": {
                    "source": "local",
                    "method": "perry_melting_point",
                    "quality": 0.95,
                }
            },
        )

        selection = PsatCanonicalizationAdapter(
            component,
            psat_at_temperature=lambda _item, _temperature: 0.01,
        ).resolve_domain()

        self.assertEqual(selection.T_min, 178.0)
        self.assertEqual(selection.basis, "melting_point")
        self.assertEqual(selection.quality, 0.95)
        self.assertIsNone(selection.pressure_bar)
        self.assertEqual(selection.rejected_candidates, ("Tt unavailable",))

    def test_pfd_psat_override_ignores_lookup_phase_points(self):
        component = {
            "Tc": 500.0,
            "Tt": 180.0,
            "Pt": 0.002,
            "Tm": 190.0,
            "property_correlations": {
                "Psat": {
                    "equation": "poly_x",
                    "coefficients": {"A": 0.1, "B": 0.1},
                    "_pfd_override": True,
                },
            },
            "property_sources": {
                "Tt": {"source": "local", "method": "coolprop_Tt"},
                "Pt": {"source": "local", "method": "coolprop_Pt"},
                "Tm": {"source": "local", "method": "perry_Tm"},
            },
        }

        selection = PsatCanonicalizationAdapter(
            component,
            tsat_at_pressure=lambda _item, _pressure: 230.0,
        ).resolve_domain()

        self.assertEqual(selection.basis, "pressure_floor")
        self.assertEqual(selection.T_min, 230.0)
        self.assertEqual(
            selection.rejected_candidates,
            (
                "Tt ignored because it is not part of the PFD Psat override",
                "Tm ignored because it is not part of the PFD Psat override",
            ),
        )

    def test_invalid_triple_point_falls_through_to_melting_point(self):
        adapter = PsatCanonicalizationAdapter(
            {"Tc": 500.0, "Tt": 600.0, "Tm": 200.0},
            psat_at_temperature=lambda _item, _temperature: 0.01,
        )

        selection = adapter.resolve_domain()

        self.assertEqual(selection.T_min, 200.0)
        self.assertEqual(selection.basis, "melting_point")
        self.assertIn("Tt invalid", selection.rejected_candidates[0])

    def test_pressure_floor_receives_component_and_exact_default_pressure(self):
        calls = []
        component = {"Tc": 450.0, "symbol": "X"}

        def tsat_at_pressure(item, pressure):
            calls.append((item, pressure))
            return 145.0

        selection = PsatCanonicalizationAdapter(
            component,
            tsat_at_pressure=tsat_at_pressure,
        ).resolve_domain()

        self.assertEqual(
            calls,
            [(component, DEFAULT_PSAT_MINIMUM_PRESSURE_BAR)],
        )
        self.assertEqual(selection.T_min, 145.0)
        self.assertEqual(selection.basis, "pressure_floor")
        self.assertEqual(selection.pressure_bar, 1.0e-3)
        self.assertEqual(selection.source, "calculated")
        self.assertEqual(selection.method, "tsat_at_pressure")
        self.assertIsNone(selection.quality)
        self.assertIn("0.001 bar", selection.notes)

    def test_pressure_floor_preserves_resolution_result_provenance(self):
        component = {"Tc": 400.0, "Tt": None, "Tm": float("nan")}
        adapter = PsatCanonicalizationAdapter(
            component,
            tsat_at_pressure=lambda _item, _pressure: PropertyResolutionResult(
                value=133.0,
                source="completion",
                method="nannoolal_inverse",
                quality=0.82,
                notes="anchored at Tb",
            ),
        )

        selection = adapter.resolve_domain()

        self.assertEqual(selection.T_min, 133.0)
        self.assertEqual(selection.source, "completion")
        self.assertEqual(selection.method, "nannoolal_inverse")
        self.assertEqual(selection.quality, 0.82)
        self.assertIn("anchored at Tb", selection.notes)
        self.assertEqual(
            selection.rejected_candidates,
            ("Tt unavailable", "Tm invalid for 0 < T < Tc: nan"),
        )

    def test_explicit_critical_temperature_can_override_component_value(self):
        adapter = PsatCanonicalizationAdapter(
            {"Tc": None, "Tm": 150.0},
            psat_at_temperature=lambda _item, _temperature: 0.01,
        )

        selection = adapter.resolve_domain(T_critical=400.0)

        self.assertEqual(selection.T_min, 150.0)

    def test_missing_all_sources_reports_every_failure(self):
        with self.assertRaisesRegex(
            PsatAdapterError,
            "Tt unavailable; Tm unavailable; Tsat pressure fallback unavailable",
        ):
            PsatCanonicalizationAdapter({"Tc": 500.0}).resolve_domain()

    def test_invalid_pressure_floor_temperature_is_rejected(self):
        adapter = PsatCanonicalizationAdapter(
            {"Tc": 500.0},
            tsat_at_pressure=lambda _item, _pressure: 500.0,
        )

        with self.assertRaisesRegex(
            PsatAdapterError,
            r"Tsat\(0.001 bar\) invalid",
        ):
            adapter.resolve_domain()

    def test_triple_point_below_default_pressure_still_sets_domain(self):
        calls = []
        component = {
            "Tc": 500.0,
            "Tt": 180.0,
            "Pt": 1.0e-6,
            "property_sources": {
                "Tt": {"method": "pfd_component_override", "quality": 1.0},
                "Pt": {"method": "pfd_component_override", "quality": 1.0},
            },
        }

        def tsat_at_pressure(item, pressure):
            calls.append((item, pressure))
            return PropertyResolutionResult(
                value=220.0,
                source="completion",
                method="floor_inverse",
                quality=0.9,
            )

        selection = PsatCanonicalizationAdapter(
            component,
            tsat_at_pressure=tsat_at_pressure,
        ).resolve_domain()

        self.assertEqual(calls, [])
        self.assertEqual(selection.T_min, 180.0)
        self.assertEqual(selection.basis, "triple_point")
        self.assertEqual(selection.pressure_bar, 1.0e-6)
        self.assertIn("Pt=1e-06 bar", selection.notes)

    def test_triple_point_without_pressure_does_not_probe_psat(self):
        calls = []
        component = {"Tc": 500.0, "Tt": 180.0, "Tm": 190.0}

        def psat_at_temperature(item, temperature):
            calls.append((item, temperature))
            return PropertyResolutionResult(
                value=1.0e-5,
                source="resolution",
                method="resolved_psat",
                quality=0.96,
            )

        selection = PsatCanonicalizationAdapter(
            component,
            psat_at_temperature=psat_at_temperature,
            tsat_at_pressure=lambda _item, pressure: (
                220.0 if pressure == 0.001 else None
            ),
        ).resolve_domain()

        self.assertEqual(calls, [])
        self.assertEqual(selection.basis, "triple_point")
        self.assertEqual(selection.T_min, 180.0)
        self.assertIsNone(selection.pressure_bar)

    def test_melting_point_precedes_configured_pressure_fallback(self):
        calls = []
        component = {"Tc": 500.0, "Tm": 200.0}

        selection = PsatCanonicalizationAdapter(
            component,
            minimum_pressure_bar=0.02,
            psat_at_temperature=lambda _item, _temperature: 0.01,
            tsat_at_pressure=lambda item, pressure: (
                calls.append((item, pressure)) or 230.0
            ),
        ).resolve_domain()

        self.assertEqual(calls, [])
        self.assertEqual(selection.T_min, 200.0)
        self.assertEqual(selection.basis, "melting_point")
        self.assertIsNone(selection.pressure_bar)

    def test_explicit_deep_vacuum_pressure_bound_is_honored(self):
        selection = PsatCanonicalizationAdapter(
            {
                "symbol": "ETOH",
                "name": "ethanol",
                "CAS": "64-17-5",
                "Tc": 514.7092848812961,
            },
            minimum_pressure_bar=1.0e-6,
            tsat_at_pressure=lambda _item, pressure: (
                184.27865279559805 if pressure == 1.0e-6 else None
            ),
        ).resolve_domain()

        self.assertEqual(selection.basis, "pressure_floor")
        self.assertEqual(selection.pressure_bar, 1.0e-6)
        self.assertAlmostEqual(selection.T_min, 184.27865279559805)

    def test_pressure_floor_failure_keeps_original_exception(self):
        failure = RuntimeError("inverse did not converge")

        def fail(_item, _pressure):
            raise failure

        adapter = PsatCanonicalizationAdapter(
            {"Tc": 500.0},
            tsat_at_pressure=fail,
        )
        with self.assertRaises(PsatAdapterError) as context:
            adapter.resolve_domain()

        self.assertIs(context.exception.__cause__, failure)
        self.assertIn("inverse did not converge", str(context.exception))

    def test_invalid_adapter_inputs_fail_before_provider_access(self):
        calls = []
        with self.assertRaises(PsatAdapterError):
            PsatCanonicalizationAdapter(
                {"Tc": 500.0},
                minimum_pressure_bar=0.0,
                tsat_at_pressure=lambda item, pressure: calls.append((item, pressure)),
            )

        adapter = PsatCanonicalizationAdapter(
            {"Tc": float("nan")},
            tsat_at_pressure=lambda item, pressure: calls.append((item, pressure)),
        )
        with self.assertRaises(PsatAdapterError):
            adapter.resolve_domain()

        self.assertEqual(calls, [])

    def test_selection_is_direct_canonicalizer_input(self):
        adapter = PsatCanonicalizationAdapter(
            {"Tc": 500.0, "Tm": 180.0},
            psat_at_temperature=lambda _item, _temperature: 0.01,
        )
        selection = adapter.resolve_domain()

        canonicalizer = PsatCanonicalizer(
            **selection.canonicalizer_input(),
            T_critical=500.0,
            P_critical_bar=40.0,
            critical_quality=0.95,
            T_boiling=350.0,
            boiling_quality=0.95,
        )

        self.assertEqual(canonicalizer.T_min, 180.0)

    def test_full_domain_pfd_af_is_a_direct_canonical_override(self):
        coefficients = {
            "A": 5.0,
            "B": -1000.0,
            "C": 0.0,
            "D": 0.0,
            "E": 0.0,
            "F": 0.0,
            "G": 0.0,
            "H": 0.1,
        }
        component = self.pfd_component(
            "canonical_psat_ah",
            coefficients,
            Tmin_K=200.0,
            Tmax_K=500.0,
            Tc_K=500.0,
            Pc_bar=math.exp(3.0),
            Tb_K=350.0,
            inverse_power=-7,
        )

        inputs = PsatCanonicalizationAdapter(component).collect_inputs(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=math.exp(3.0),
            T_boiling=350.0,
        )

        self.assertEqual(inputs.segments, ())
        self.assertIsNotNone(inputs.canonical_override)
        curve = inputs.canonical_override
        self.assertEqual(
            curve.coefficients,
            tuple(coefficients[name] for name in "ABCDEFGH"),
        )
        self.assertEqual(curve.inverse_power, -7)
        self.assertEqual(curve.form, CanonicalPsatForm.AH)
        self.assertEqual(curve.quality, 1.0)
        expected_ln_pressure = 2.5 + 0.1 * ((400.0 / 500.0) ** -7 - 1.0)
        expected_derivative = (
            1000.0 / 400.0**2 + 0.1 * -7.0 / 500.0 * (400.0 / 500.0) ** -8
        )
        self.assertAlmostEqual(curve.ln_pressure(400.0), expected_ln_pressure)
        self.assertAlmostEqual(curve.dln_pressure_dT(400.0), expected_derivative)
        self.assertEqual(curve.provenance[0].priority, PsatPriority.PFD_OVERRIDE)
        self.assertEqual(curve.provenance[0].segment_type, "canonical_override")

        canonicalizer = PsatCanonicalizer(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=math.exp(3.0),
            critical_quality=0.95,
            T_boiling=350.0,
            boiling_quality=0.95,
        )
        result = canonicalizer.canonicalize(**inputs.canonicalizer_arguments())
        self.assertIs(result.curve, curve)
        self.assertIsNone(result.assembly)
        self.assertIsNone(result.completion)

    def test_narrow_pfd_af_is_a_pinned_segment_instead_of_an_override(self):
        coefficients = {
            "A": 5.0,
            "B": -1000.0,
            "C": 0.0,
            "D": 0.0,
            "E": 0.0,
            "F": 0.0,
            "G": 0.0,
        }
        component = self.pfd_component(
            "canonical_psat",
            coefficients,
            Tmin_K=250.0,
            Tmax_K=450.0,
        )

        inputs = PsatCanonicalizationAdapter(component).collect_inputs(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=math.exp(3.0),
        )

        self.assertIsNone(inputs.canonical_override)
        self.assertEqual(len(inputs.segments), 1)
        segment = inputs.segments[0]
        self.assertEqual(segment.segment_type, PsatSegmentType.PINNED)
        self.assertEqual(segment.priority, PsatPriority.PFD_OVERRIDE)
        self.assertEqual(segment.quality, 1.0)
        self.assertAlmostEqual(segment.ln_pressure(400.0), 2.5)

    def test_pfd_canonical_psat_defaults_to_a_f(self):
        coefficients = {
            "A": 5.0,
            "B": -1000.0,
            "C": 0.0,
            "D": 0.0,
            "E": 0.0,
            "F": 1.0e-14,
        }
        inputs = PsatCanonicalizationAdapter(
            self.pfd_component(
                "canonical_psat",
                coefficients,
                Tmin_K=200.0,
                Tmax_K=500.0,
            )
        ).collect_inputs(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=math.exp(3.0 + 1.0e-14 * 500.0**5),
        )

        curve = inputs.canonical_override
        self.assertEqual(curve.form, CanonicalPsatForm.AF)
        self.assertEqual(curve.G, 0.0)
        self.assertEqual(curve.H, 0.0)
        self.assertAlmostEqual(
            curve.ln_pressure(400.0),
            5.0 - 1000.0 / 400.0 + 1.0e-14 * 400.0**5,
        )

    def test_explicit_pfd_a_g_form_is_supported(self):
        coefficients = {
            "A": 5.0,
            "B": -1000.0,
            "C": 0.0,
            "D": 0.0,
            "E": 0.0,
            "F": 1.0e-14,
            "G": 1.0e-9,
        }
        inputs = PsatCanonicalizationAdapter(
            self.pfd_component(
                "canonical_psat_ag",
                coefficients,
                Tmin_K=200.0,
                Tmax_K=500.0,
            )
        ).collect_inputs(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=math.exp(3.0 + 1.0e-14 * 500.0**5 + 1.0e-9 * 500.0**3),
        )

        curve = inputs.canonical_override
        self.assertEqual(curve.form, CanonicalPsatForm.AG)
        self.assertAlmostEqual(
            curve.ln_pressure(400.0),
            5.0 - 1000.0 / 400.0 + 1.0e-14 * 400.0**5 + 1.0e-9 * 400.0**3,
        )

    def test_legacy_pfd_psat_is_first_hard_pin_layer(self):
        coefficients = {
            "A": -2.0,
            "B": 0.01,
        }
        component = self.pfd_component(
            "exp_poly_x",
            coefficients,
            Tmin_K=300.0,
            Tmax_K=400.0,
        )
        pfd_inputs = PsatCanonicalizationAdapter(component).collect_inputs(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=math.exp(-2.0 + 0.01 * ((500.0 - 298.15) / 100.0)),
        )

        def baseline_ln_pressure(T):
            return -2.0 + 0.01 * ((T - 298.15) / 100.0)

        lower_priority = PsatSegment(
            source="CoolProp",
            method="synthetic_heos",
            segment_type=PsatSegmentType.PINNED,
            priority=int(PsatPriority.COOLPROP_HEOS),
            T_min=200.0,
            T_max=500.0,
            ln_pressure_function=baseline_ln_pressure,
            derivative_function=lambda _T: 0.0001,
            quality=0.995,
        )
        inputs = pfd_inputs.merged(
            PsatCanonicalizationInputs(
                segments=(lower_priority,),
            )
        )
        canonicalizer = PsatCanonicalizer(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=math.exp(baseline_ln_pressure(500.0)),
            critical_quality=0.95,
        )

        result = canonicalizer.canonicalize(**inputs.canonicalizer_arguments())

        self.assertEqual(
            [item.segment.priority for item in result.assembly.slices],
            [
                PsatPriority.COOLPROP_HEOS,
                PsatPriority.PFD_OVERRIDE,
                PsatPriority.COOLPROP_HEOS,
            ],
        )
        self.assertEqual(
            [(item.T_min, item.T_max) for item in result.assembly.slices],
            [(200.0, 300.0), (300.0, 400.0), (400.0, 500.0)],
        )
        pfd_segment = pfd_inputs.segments[0]
        self.assertAlmostEqual(pfd_segment.dln_pressure_dT(350.0), 0.0001)

    def test_coolprop_psat_segment_covers_triple_to_critical(self):
        component = {
            "symbol": "ETOH",
            "name": "ethanol",
            "CAS": "64-17-5",
        }

        inputs = PsatCanonicalizationAdapter(component).collect_inputs()

        segment = next(
            item for item in inputs.segments if item.method == "coolprop_HEOS_psat"
        )
        self.assertEqual(segment.method, "coolprop_HEOS_psat")
        self.assertEqual(segment.priority, PsatPriority.COOLPROP_HEOS)
        self.assertAlmostEqual(segment.T_min, 159.1)
        self.assertAlmostEqual(segment.T_max, 514.7092848812961)
        self.assertAlmostEqual(segment.quality, 0.995)
        self.assertAlmostEqual(
            segment.P_min_bar,
            segment.pressure_bar(segment.T_min),
            places=18,
        )
        self.assertGreater(segment.pressure_bar(351.57040446751455), 1.0)
        self.assertLess(segment.pressure_bar(351.57040446751455), 1.02)
        self.assertEqual(segment.metadata["qualified_name"], "HEOS::Ethanol")

    def test_coolprop_segment_canonicalizes_its_complete_domain(self):
        inputs = PsatCanonicalizationAdapter(
            {
                "symbol": "ETOH",
                "name": "ethanol",
                "CAS": "64-17-5",
            }
        ).collect_inputs()
        segment = inputs.segments[0]
        canonicalizer = PsatCanonicalizer(
            T_min=segment.T_min,
            T_critical=segment.T_max,
            P_critical_bar=segment.P_max_bar,
            critical_quality=0.995,
            T_boiling=351.57040446751455,
            boiling_quality=0.995,
        )

        result = canonicalizer.canonicalize(**inputs.canonicalizer_arguments())

        self.assertTrue(result.curve.diagnostics.monotonic)
        self.assertLess(result.curve.diagnostics.mard_percent, 0.2)
        self.assertLess(
            result.curve.diagnostics.max_absolute_relative_error_percent,
            0.5,
        )

    def test_coolprop_uses_if97_for_water(self):
        inputs = PsatCanonicalizationAdapter(
            {
                "symbol": "H2O",
                "name": "water",
                "CAS": "7732-18-5",
            }
        ).collect_inputs()

        segment = inputs.segments[0]
        self.assertEqual(segment.method, "coolprop_IF97_psat")
        self.assertEqual(segment.metadata["qualified_name"], "IF97::Water")
        self.assertAlmostEqual(segment.T_min, 273.16)
        self.assertAlmostEqual(segment.P_min_bar, 0.00611657)

    def test_unsupported_component_has_no_coolprop_segment(self):
        inputs = PsatCanonicalizationAdapter(
            {
                "symbol": "HNO3",
                "name": "nitric acid",
                "CAS": "7697-37-2",
            }
        ).collect_inputs()

        self.assertEqual(inputs, PsatCanonicalizationInputs())

    def test_coolprop_requires_an_exact_resolved_cas(self):
        components = (
            {"symbol": "NA", "name": "sodium", "formula": "Na"},
            {"symbol": "C2H6O", "name": "unknown"},
            {"symbol": "Co", "name": "cobalt"},
            {"symbol": "R11", "name": "custom solvent"},
            {"symbol": "ETOH", "name": "ethanol"},
            {"symbol": "ETOH", "CAS": "64-17-4"},
        )

        for component in components:
            with self.subTest(component=component):
                inputs = PsatCanonicalizationAdapter(component).collect_inputs()
                self.assertFalse(
                    any(
                        segment.method.startswith("coolprop_")
                        for segment in inputs.segments
                    )
                )
                if component.get("name") == "cobalt":
                    self.assertFalse(
                        any(
                            segment.method == "perry_2_8_vapor_pressure"
                            for segment in inputs.segments
                        )
                    )

    def test_coolprop_accepts_compact_exact_cas(self):
        inputs = PsatCanonicalizationAdapter(
            {
                "symbol": "ETOH",
                "CAS": "64175",
            }
        ).collect_inputs()

        self.assertEqual(inputs.segments[0].metadata["qualified_name"], "HEOS::Ethanol")

    def test_non_pfd_psat_correlation_is_collected_as_local_segment(self):
        component = self.pfd_component(
            "exp_poly_x",
            {"A": 1.0},
            Tmin_K=250.0,
            Tmax_K=450.0,
        )
        component["property_correlations"]["Psat"].pop("_pfd_override")
        component["property_correlations"]["Psat"]["quality"] = 0.97

        inputs = PsatCanonicalizationAdapter(component).collect_inputs(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=40.0,
        )

        self.assertEqual(len(inputs.segments), 1)
        segment = inputs.segments[0]
        self.assertEqual(segment.method, "local_psat_exp_poly_x")
        self.assertEqual(segment.priority, PsatPriority.LOCAL_CORRELATION)
        self.assertEqual(segment.quality, 0.97)
        self.assertEqual((segment.T_min, segment.T_max), (250.0, 450.0))
        self.assertAlmostEqual(segment.ln_pressure(350.0), 1.0)

    def test_local_reduced_psat_uses_declared_internal_criticals(self):
        coefficients = {
            "A": -7.0,
            "B": 1.2,
            "C": -3.0,
            "D": -1.5,
        }
        component = {
            "symbol": "X",
            "Tc": 500.0,
            "Pc": 40.0,
            "property_correlations": {
                "Psat": {
                    "equation": "reduced_vapor_pressure",
                    "Tmin_K": 250.0,
                    "Tmax_K": 450.0,
                    "Tc_K": 600.0,
                    "Pc_Pa": 5_000_000.0,
                    "coefficients": coefficients,
                },
            },
        }

        segment = PsatCanonicalizationAdapter(component).collect_inputs().segments[0]
        temperature = 350.0
        Tr = temperature / 600.0
        tau = 1.0 - Tr
        numerator = (
            coefficients["A"] * tau
            + coefficients["B"] * tau**1.5
            + coefficients["C"] * tau**3
            + coefficients["D"] * tau**6
        )
        expected = math.log(50.0) + numerator / Tr

        self.assertAlmostEqual(segment.ln_pressure(temperature), expected)
        self.assertEqual(segment.metadata["reducing_Tc_K"], 600.0)
        self.assertEqual(segment.metadata["reducing_Pc_bar"], 50.0)

    def test_local_reduced_psat_requires_complete_self_contained_parameters(self):
        base = {
            "symbol": "X",
            "Tc": 500.0,
            "Pc": 40.0,
            "property_correlations": {
                "Psat": {
                    "equation": "reduced_vapor_pressure",
                    "Tmin_K": 250.0,
                    "Tmax_K": 450.0,
                    "Tc_K": 600.0,
                    "Pc_Pa": 5_000_000.0,
                    "coefficients": {"A": -7.0, "B": 1.2, "C": -3.0},
                },
            },
        }
        with self.assertRaisesRegex(PsatAdapterError, "requires coefficient D"):
            PsatCanonicalizationAdapter(base).collect_inputs()

        missing_Tc = json.loads(json.dumps(base))
        missing_Tc["property_correlations"]["Psat"]["coefficients"]["D"] = -1.5
        missing_Tc["property_correlations"]["Psat"].pop("Tc_K")
        with self.assertRaisesRegex(PsatAdapterError, "Tc_K"):
            PsatCanonicalizationAdapter(missing_Tc).collect_inputs()

    def test_nitric_acid_local_a_f_correlation_is_a_bounded_segment(self):
        payload = json.loads(
            (
                Path(__file__).resolve().parent.parent / "data" / "chemicals.json"
            ).read_text()
        )
        component = payload["chemicals"]["HNO3"]

        inputs = PsatCanonicalizationAdapter(component).collect_inputs()

        self.assertEqual(len(inputs.segments), 1)
        segment = inputs.segments[0]
        self.assertEqual(segment.method, "local_psat_canonical_psat_af")
        self.assertEqual(segment.priority, PsatPriority.LOCAL_CORRELATION)
        self.assertEqual(segment.quality, 0.97)
        self.assertEqual((segment.T_min, segment.T_max), (231.55, 376.1))
        self.assertAlmostEqual(segment.pressure_bar(356.04), 1.013166489463462)
        self.assertGreater(segment.dln_pressure_dT(356.04), 0.0)

    def test_no_hard_fallback_does_not_replace_local_measured_segment(self):
        payload = json.loads(
            (
                Path(__file__).resolve().parent.parent / "data" / "chemicals.json"
            ).read_text()
        )
        component = payload["chemicals"]["C3H6O3"]

        inputs = PsatCanonicalizationAdapter(component).collect_inputs(
            T_min=330.4,
            T_critical=662.557,
            P_critical_bar=57.4699,
        )

        self.assertEqual(len(inputs.segments), 1)
        segment = inputs.segments[0]
        self.assertEqual(segment.method, "local_psat_exp_poly_x")
        self.assertAlmostEqual(
            segment.pressure_bar(348.4),
            0.0007742650823540126,
            places=12,
        )
        self.assertNotIn("no_hard_nannoolal", inputs.metadata)

    def test_curated_reduced_psat_records_preserve_internal_criticals(self):
        payload = json.loads(
            (
                Path(__file__).resolve().parent.parent / "data" / "chemicals.json"
            ).read_text()
        )["chemicals"]
        expected = {
            "C3H8O3": (824.854, 62.607, 850.0, 75.0),
            "C6H7N": (698.8, 53.1, 705.0, 56.3),
            "C5H5N": (620.0, 56.5, 620.0, 56.5),
            "C3H6O_PO": (488.11, 54.366, 488.11, 54.366),
            "C8H8O": (709.6, 40.1, 709.6, 40.1),
        }

        for key, values in expected.items():
            with self.subTest(key=key):
                component = payload[key]
                correlation = component["property_correlations"]["Psat"]
                self.assertEqual(
                    (
                        component["Tc"],
                        component["Pc"],
                        correlation["Tc_K"],
                        correlation["Pc_Pa"] / 100000.0,
                    ),
                    values,
                )

    def test_curated_critical_qualities_override_dataset_default(self):
        database = ChemicalDatabase(enable_online=False)
        expected_qualities = {
            "C3H8O3": {
                "Tc": 0.88,
                "Pc": 0.88,
                "Vc": 0.92,
                "Zc": 0.88,
                "omega": 0.86,
            },
            "C3H6O_PO": {
                "Tc": 0.995,
                "Pc": 0.995,
                "Vc": 0.995,
                "Zc": 0.995,
            },
        }
        for key, expected in expected_qualities.items():
            with self.subTest(key=key):
                component = database.chemicals[key]
                for field_name, expected_quality in expected.items():
                    provenance = component.property_sources[field_name]
                    self.assertEqual(provenance["quality"], expected_quality)
        glycerol = database.chemicals["C3H8O3"]
        self.assertEqual(glycerol.Vc, 240.22)
        self.assertEqual(
            glycerol.property_sources["Vc"]["method"], "effective_critical"
        )
        self.assertEqual(
            glycerol.property_sources["omega"]["method"], "effective_critical"
        )

    def test_coolprop_is_collected_before_local_correlation(self):
        component = {
            "symbol": "ETOH",
            "name": "ethanol",
            "CAS": "64-17-5",
            "Tc": 514.7092848812961,
            "property_correlations": {
                "Psat": {
                    "equation": "exp_poly_x",
                    "Tmin_K": 250.0,
                    "Tmax_K": 400.0,
                    "coefficients": {"A": -2.0, "B": 0.01},
                    "quality": 0.97,
                },
            },
        }

        inputs = PsatCanonicalizationAdapter(component).collect_inputs()

        self.assertGreaterEqual(len(inputs.segments), 5)
        self.assertEqual(
            [segment.priority for segment in inputs.segments[:5]],
            [
                PsatPriority.COOLPROP_HEOS,
                PsatPriority.LOCAL_CORRELATION,
                PsatPriority.PERRY_2_8,
                PsatPriority.HIGH_QUALITY_ANTOINE,
                PsatPriority.PERRY_2_10,
            ],
        )

    def test_perry_2_8_is_an_analytic_bounded_segment(self):
        from perry_properties import get_perry_property_library

        component = {
            "symbol": "ETOH",
            "name": "ethanol",
            "CAS": "64-17-5",
        }
        inputs = PsatCanonicalizationAdapter(component).collect_inputs()
        segment = next(
            item
            for item in inputs.segments
            if item.method == "perry_2_8_vapor_pressure"
        )

        self.assertEqual(segment.priority, PsatPriority.PERRY_2_8)
        self.assertEqual(segment.quality, 0.98)
        self.assertEqual((segment.T_min, segment.T_max), (159.05, 514.0))
        temperature = 350.0
        expected = (
            get_perry_property_library()
            .vapor_pressure_bar(
                "64-17-5",
                temperature,
            )
            .value
        )
        self.assertAlmostEqual(segment.pressure_bar(temperature), expected)
        step = 1.0e-3
        numerical_slope = (
            segment.ln_pressure(temperature + step)
            - segment.ln_pressure(temperature - step)
        ) / (2.0 * step)
        self.assertAlmostEqual(
            segment.dln_pressure_dT(temperature),
            numerical_slope,
            places=9,
        )
        self.assertEqual(
            segment.handoff_requirement,
            PsatHandoffRequirement.NONE,
        )

    def test_perry_2_10_is_a_bounded_log_pchip_segment(self):
        from perry_properties import get_perry_property_library

        component = {
            "symbol": "C7H6O",
            "name": "benzaldehyde",
            "CAS": "100-52-7",
        }
        inputs = PsatCanonicalizationAdapter(component).collect_inputs()
        segment = next(
            item
            for item in inputs.segments
            if item.method == "perry_2_10_vapor_pressure"
        )

        self.assertEqual(segment.priority, PsatPriority.PERRY_2_10)
        self.assertEqual(segment.quality, 0.90)
        self.assertEqual(
            segment.metadata["tb_validation_status"],
            "validation_unavailable",
        )
        self.assertEqual(
            segment.derivative_basis,
            PsatDerivativeBasis.INTERPOLATED,
        )
        self.assertAlmostEqual(segment.T_min, 299.35)
        self.assertAlmostEqual(segment.T_max, 452.15)
        self.assertAlmostEqual(segment.P_max_bar, 1.01325)
        temperature = 400.0
        expected = (
            get_perry_property_library()
            .table_2_10_vapor_pressure_bar(
                "100-52-7",
                temperature,
            )
            .value
        )
        self.assertAlmostEqual(segment.pressure_bar(temperature), expected)
        step = 1.0e-3
        numerical_slope = (
            segment.ln_pressure(temperature + step)
            - segment.ln_pressure(temperature - step)
        ) / (2.0 * step)
        self.assertAlmostEqual(
            segment.dln_pressure_dT(temperature),
            numerical_slope,
            places=9,
        )
        self.assertEqual(segment.metadata["interpolation"], "log_pressure_pchip")

    def test_perry_2_10_uses_shared_tb_admission_and_confidence(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_perry_2_10_inputs,
        )

        def collect(component, boiling=None):
            return PsatCanonicalizationAdapter(
                component,
                input_methods=(_collect_perry_2_10_inputs,),
            ).collect_inputs(T_boiling=boiling)

        benzaldehyde = {
            "name": "benzaldehyde",
            "CAS": "100-52-7",
            "Tb": 452.15,
            "property_sources": {
                "Tb": {
                    "source": "local",
                    "method": "perry_normal_boiling_point",
                    "quality": 0.97,
                },
            },
        }
        accepted_segment = collect(benzaldehyde, 452.15).segments[0]
        self.assertEqual(accepted_segment.quality, 0.95)
        self.assertEqual(
            accepted_segment.metadata["tb_validation_status"],
            "hard_tb_validated",
        )
        self.assertEqual(
            accepted_segment.metadata["quality_basis"],
            "hard_tb_validation",
        )
        self.assertEqual(
            accepted_segment.handoff_requirement,
            PsatHandoffRequirement.NONE,
        )

        thiodiglycol = {
            "name": "thiodiglycol",
            "CAS": "111-48-8",
            "Tb": 555.15,
            "property_sources": {
                "Tb": {
                    "source": "online",
                    "method": "pubchem",
                    "quality": 0.95,
                },
            },
        }
        rejected = collect(thiodiglycol, 555.15)
        self.assertEqual(rejected.segments, ())
        self.assertEqual(len(rejected.warnings), 1)
        warning = rejected.warnings[0]
        self.assertIn("Perry 9th Table 2-10", warning)
        self.assertIn("Tb=555.15 K", warning)
        self.assertIn("Psat(Tb)=", warning)
        self.assertIn("87.2496%", warning)
        self.assertIn("allowed tolerance=2%", warning)

        soft = {
            **thiodiglycol,
            "Tb": 551.7,
            "property_sources": {
                "Tb": {
                    "source": "estimated",
                    "method": "nannoolal_tb",
                    "quality": 0.80,
                },
            },
        }
        soft_segment = collect(soft, soft["Tb"]).segments[0]
        self.assertEqual(soft_segment.quality, 0.90)
        self.assertEqual(
            soft_segment.metadata["tb_validation_status"],
            "validation_unavailable",
        )
        self.assertEqual(
            soft_segment.metadata["quality_basis"],
            "standalone_unvalidated",
        )
        self.assertNotIn("tb_validation_pending", soft_segment.metadata)

        missing = dict(thiodiglycol)
        missing.pop("Tb")
        missing.pop("property_sources")
        missing_segment = collect(missing).segments[0]
        self.assertEqual(missing_segment.quality, 0.90)
        self.assertEqual(
            missing_segment.metadata["tb_validation_status"],
            "validation_unavailable",
        )

        outside = {**thiodiglycol, "Tb": 600.0}
        outside_segment = collect(outside, 600.0).segments[0]
        self.assertEqual(outside_segment.quality, 0.90)
        self.assertEqual(
            outside_segment.metadata["tb_validation_status"],
            "hard_tb_outside_range",
        )

        higher = replace(
            soft_segment,
            source="higher reference",
            method="higher_reference",
            priority=int(PsatPriority.PERRY_2_8),
            T_max=soft_segment.T_max - 20.0,
            quality=0.98,
            handoff_requirement=PsatHandoffRequirement.NONE,
            metadata={
                "tb_validation_required": False,
                "tb_validation_status": "exempt_priority",
                "quality_basis": "exempt_priority",
            },
        )
        promoted_result = PsatHandoffCoordinator(
            soft_segment.T_min,
            soft_segment.T_max,
        ).coordinate((soft_segment, higher))
        promoted = next(
            item
            for item in promoted_result.segments
            if item.method == "perry_2_10_vapor_pressure"
        )
        self.assertEqual(promoted.quality, 0.93)
        self.assertEqual(
            promoted.metadata["quality_basis"],
            "higher_preference_overlap",
        )

        same_priority = replace(
            higher,
            method="same_priority_reference",
            priority=int(PsatPriority.PERRY_2_10),
        )
        same_result = PsatHandoffCoordinator(
            soft_segment.T_min,
            soft_segment.T_max,
        ).coordinate((soft_segment, same_priority))
        same = next(
            item
            for item in same_result.segments
            if item.method == "perry_2_10_vapor_pressure"
        )
        self.assertEqual(same.quality, 0.90)
        self.assertEqual(
            same.metadata["quality_basis"],
            "standalone_unvalidated",
        )

    def test_priority_exempt_direct_sources_ignore_conflicting_tb(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_curated_antoine_inputs,
            _collect_perry_2_8_inputs,
        )

        hard_tb = {
            "source": "local",
            "method": "hard_fixture_tb",
            "quality": 0.99,
        }
        perry = PsatCanonicalizationAdapter(
            {
                "name": "benzene",
                "CAS": "71-43-2",
                "Tb": 300.0,
                "property_sources": {"Tb": hard_tb},
            },
            input_methods=(_collect_perry_2_8_inputs,),
        ).collect_inputs(T_boiling=300.0)
        self.assertTrue(perry.segments)
        self.assertTrue(
            all(
                item.metadata["tb_validation_status"] == "exempt_priority"
                and item.metadata["tb_validation_required"] is False
                and item.quality == 0.98
                for item in perry.segments
            )
        )

        curated = PsatCanonicalizationAdapter(
            {
                "symbol": "X",
                "source": "local",
                "Tb": 350.0,
                "antoine_A": 5.0,
                "antoine_B": 1000.0,
                "antoine_C": 200.0,
                "antoine_Tmin": 250.0,
                "antoine_Tmax": 400.0,
                "property_sources": {"Tb": hard_tb},
            },
            input_methods=(_collect_curated_antoine_inputs,),
        ).collect_inputs(T_boiling=350.0)
        self.assertEqual(len(curated.segments), 1)
        self.assertEqual(curated.warnings, ())
        self.assertEqual(
            curated.segments[0].metadata["tb_validation_status"],
            "exempt_priority",
        )
        self.assertFalse(curated.segments[0].metadata["tb_validation_required"])

        pfd = PsatCanonicalizationAdapter(
            self.pfd_component(
                "exp_poly_x",
                {"A": -2.0},
                Tmin_K=250.0,
                Tmax_K=450.0,
            )
        ).collect_inputs(
            T_min=250.0,
            T_critical=500.0,
            P_critical_bar=40.0,
            T_boiling=350.0,
        )
        self.assertEqual(len(pfd.segments), 1)
        self.assertEqual(
            pfd.segments[0].metadata["tb_validation_status"],
            "exempt_priority",
        )
        self.assertFalse(pfd.segments[0].metadata["tb_validation_required"])

    def test_perry_2_8_precedes_2_10_when_both_are_available(self):
        inputs = PsatCanonicalizationAdapter(
            {
                "name": "ethanol",
                "CAS": "64-17-5",
            }
        ).collect_inputs()
        perry_segments = [
            segment
            for segment in inputs.segments
            if segment.method.startswith("perry_2_")
        ]

        self.assertEqual(
            [segment.priority for segment in perry_segments],
            [PsatPriority.PERRY_2_8, PsatPriority.PERRY_2_10],
        )

    def test_no_hard_aw_routes_by_tb_and_omega_availability(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_no_hard_fallback_inputs,
        )

        def collect(include_tb, include_omega):
            component = {
                "CAS": ("71-43-2" if include_tb else "not-a-real-compound"),
                "Tc": 500.0,
                "Pc": 50.0,
                "property_sources": {
                    field_name: {
                        "source": "reported",
                        "method": "reported",
                        "quality": 0.98,
                    }
                    for field_name in ("Tc", "Pc")
                },
            }
            if include_tb:
                component["Tb"] = 330.0
                component["property_sources"]["Tb"] = {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
            if include_omega:
                component["omega"] = 0.4
                component["property_sources"]["omega"] = {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
            adapter = PsatCanonicalizationAdapter(
                component,
                hvap_at_temperature=lambda _component, _T: None,
                input_methods=(_collect_no_hard_fallback_inputs,),
            )
            return adapter.collect_inputs(
                T_min=200.0,
                T_critical=500.0,
                P_critical_bar=50.0,
                T_boiling=330.0 if include_tb else None,
            )

        for (
            include_tb,
            include_omega,
            expected_route,
            expected_factor,
            expected_factors,
        ) in (
            (
                True,
                True,
                "tb_variable_omega",
                0.91,
                {0.91, 0.88, 0.85},
            ),
            (
                True,
                False,
                "tb_effective_constant_omega",
                0.90,
                {0.90, 0.88, 0.85, 0.80},
            ),
            (
                False,
                True,
                "physical_omega",
                0.91,
                {0.91, 0.88, 0.85, 0.80},
            ),
        ):
            with self.subTest(
                include_tb=include_tb,
                include_omega=include_omega,
            ):
                inputs = collect(include_tb, include_omega)
                relation = next(
                    item
                    for item in inputs.relations
                    if item.method == "no_hard_ambrose_walton"
                )
                self.assertTrue(relation.requires_no_hard_segments)
                self.assertEqual(
                    relation.metadata["no_hard_route"],
                    expected_route,
                )
                self.assertEqual(
                    {item.metadata["quality_factor"] for item in inputs.relations},
                    expected_factors,
                )
                anchors = PsatAnchorRegistry.from_tb_tc(
                    T_critical=500.0,
                    P_critical_bar=50.0,
                    critical_quality=0.98,
                    T_boiling=330.0 if include_tb else None,
                    boiling_quality=0.98 if include_tb else None,
                )
                bound = relation.bind(
                    PsatBoundaryConditions(
                        T_min=200.0,
                        T_max=500.0,
                        anchors=anchors.points,
                    )
                )
                self.assertEqual(
                    bound.metadata["no_hard_route"],
                    expected_route,
                )
                self.assertAlmostEqual(
                    bound.pressure_bar(500.0),
                    50.0,
                    places=10,
                )
                self.assertAlmostEqual(
                    bound.quality,
                    0.98 * expected_factor,
                )
                if include_tb:
                    self.assertAlmostEqual(
                        bound.pressure_bar(330.0),
                        1.01325,
                        places=10,
                    )
                if expected_route == "tb_variable_omega":
                    self.assertGreater(
                        bound.raw_dln_pressure_dT(200.0),
                        0.0,
                    )
                    self.assertLess(
                        math.exp(bound.raw_ln_pressure(200.0)),
                        bound.pressure_bar(330.0),
                    )
                    transition = 350.0
                    step = 1.0e-4
                    self.assertAlmostEqual(
                        bound.dln_pressure_dT(transition - step),
                        bound.dln_pressure_dT(transition + step),
                        places=5,
                    )

    def test_no_hard_constant_omega_with_tb_uses_point_nine_upper_factor(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_no_hard_fallback_inputs,
        )

        component = {
            "CAS": "71-43-2",
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "Tb": 360.0,
            "property_sources": {
                field_name: {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb")
            },
        }
        inputs = PsatCanonicalizationAdapter(
            component,
            hvap_at_temperature=lambda _component, _T: None,
            input_methods=(_collect_no_hard_fallback_inputs,),
        ).collect_inputs(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=50.0,
            T_boiling=360.0,
        )

        self.assertEqual(
            inputs.relations[0].metadata["no_hard_route"],
            "tb_effective_constant_omega",
        )
        self.assertEqual(
            {item.metadata["quality_factor"] for item in inputs.relations},
            {0.90, 0.88, 0.85, 0.80},
        )
        self.assertAlmostEqual(
            inputs.relations[0].quality,
            0.98 * 0.90,
        )

    def test_no_hard_aw_uses_estimated_tb_when_tb_and_omega_are_missing(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_no_hard_fallback_inputs,
        )

        class Resolver:
            @staticmethod
            def resolve_boiling_point(
                _identifier,
                _props,
                *,
                allow_online,
                allow_estimation,
            ):
                self.assertFalse(allow_online)
                if not allow_estimation:
                    return None
                return PropertyResolutionResult(
                    value=340.0,
                    source="estimated",
                    method="test_tb_estimate",
                    quality=0.80,
                )

        adapter = PsatCanonicalizationAdapter(
            {
                "CAS": "71-43-2",
                "Tc": 500.0,
                "Pc": 50.0,
                "property_sources": {
                    field_name: {
                        "source": "reported",
                        "method": "reported",
                        "quality": 0.98,
                    }
                    for field_name in ("Tc", "Pc")
                },
            },
            hvap_at_temperature=lambda _component, _T: None,
            input_methods=(_collect_no_hard_fallback_inputs,),
        )
        adapter._property_resolver = Resolver()

        inputs = adapter.collect_inputs(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=50.0,
        )
        relation = inputs.relations[0]

        self.assertEqual(
            relation.metadata["no_hard_route"],
            "tb_effective_constant_omega",
        )
        self.assertTrue(relation.metadata["tb_was_estimated"])
        self.assertEqual(inputs.additional_anchors[0].temperature, 340.0)
        self.assertAlmostEqual(relation.quality, 0.80 * 0.90)
        self.assertTrue(
            any("No qualified Tb or omega" in warning for warning in inputs.warnings)
        )

    def test_no_hard_aw_admits_soft_criticals_and_omega_with_warnings(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_no_hard_fallback_inputs,
        )

        component = {
            "CAS": "71-43-2",
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "Tb": 330.0,
            "property_sources": {
                "Tc": {
                    "source": "soft",
                    "method": "soft_tc",
                    "quality": 0.92,
                },
                "Pc": {
                    "source": "soft",
                    "method": "soft_pc",
                    "quality": 0.91,
                },
                "omega": {
                    "source": "soft",
                    "method": "soft_omega",
                    "quality": 0.905,
                },
                "Tb": {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                },
            },
        }

        inputs = PsatCanonicalizationAdapter(
            component,
            hvap_at_temperature=lambda _component, _T: None,
            input_methods=(_collect_no_hard_fallback_inputs,),
        ).collect_inputs(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=50.0,
            T_boiling=330.0,
        )

        self.assertEqual(
            inputs.relations[0].metadata["no_hard_route"],
            "tb_variable_omega",
        )
        self.assertAlmostEqual(
            inputs.relations[0].quality,
            0.905 * 0.91,
        )
        self.assertEqual(len(inputs.warnings), 3)
        self.assertTrue(any("Tc admitted" in item for item in inputs.warnings))
        self.assertTrue(any("Pc admitted" in item for item in inputs.warnings))
        self.assertTrue(any("omega admitted" in item for item in inputs.warnings))

    def test_soft_property_quality_is_shared_by_hard_completion_routes(self):
        from property_resolution.vapor_pressure_adapter import (
            _peng_robinson_delta_z_or_ideal,
            _collect_anchored_ambrose_walton_inputs,
            _collect_anchored_ambrose_walton_lower_inputs,
            _collect_deep_vacuum_inputs,
        )

        component = {
            "CAS": "71-43-2",
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "Tb": 330.0,
            "property_sources": {
                "Tc": {
                    "source": "soft",
                    "method": "soft_tc",
                    "quality": 0.92,
                },
                "Pc": {
                    "source": "soft",
                    "method": "soft_pc",
                    "quality": 0.91,
                },
                "omega": {
                    "source": "soft",
                    "method": "soft_omega",
                    "quality": 0.905,
                },
                "Tb": {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                },
            },
        }
        inputs = PsatCanonicalizationAdapter(
            component,
            hvap_at_temperature=lambda _component, _T: PropertyResolutionResult(
                value=40.0,
                source="reported",
                method="provided_hvap_fit",
                quality=0.97,
            ),
            input_methods=(
                _collect_anchored_ambrose_walton_inputs,
                _collect_anchored_ambrose_walton_lower_inputs,
                _collect_deep_vacuum_inputs,
            ),
        ).collect_inputs(
            T_min=220.0,
            T_critical=500.0,
            P_critical_bar=50.0,
            T_boiling=330.0,
        )

        self.assertEqual(
            {item.method for item in inputs.relations},
            {
                "anchored_ambrose_walton_upper",
                "anchored_ambrose_walton_lower",
                "resolved_hvap_clapeyron_lower",
                "dynamic_omega_deep_fallback",
            },
        )
        self.assertTrue(any("Tc admitted" in item for item in inputs.warnings))
        self.assertTrue(any("Pc admitted" in item for item in inputs.warnings))
        self.assertTrue(any("omega admitted" in item for item in inputs.warnings))

        component["property_sources"]["Tc"]["quality"] = 0.899
        rejected = PsatCanonicalizationAdapter(
            component,
            hvap_at_temperature=lambda _component, _T: PropertyResolutionResult(
                value=40.0,
                source="reported",
                method="provided_hvap_fit",
                quality=0.97,
            ),
            input_methods=(
                _collect_anchored_ambrose_walton_inputs,
                _collect_anchored_ambrose_walton_lower_inputs,
                _collect_deep_vacuum_inputs,
            ),
        ).collect_inputs(
            T_min=220.0,
            T_critical=500.0,
            P_critical_bar=50.0,
            T_boiling=330.0,
        )
        self.assertEqual(
            {item.method for item in rejected.relations},
            {
                "anchored_ambrose_walton_upper",
                "anchored_ambrose_walton_lower",
                "resolved_hvap_clapeyron_lower",
                "dynamic_omega_deep_fallback",
            },
        )
        self.assertTrue(any("below quality 0.90" in item for item in rejected.warnings))
        self.assertTrue(
            all(
                item.metadata["critical_pr_quality_factor"] == 0.97
                for item in rejected.relations
                if item.method == "resolved_hvap_clapeyron_lower"
            )
        )
        clapeyron = next(
            item
            for item in rejected.relations
            if item.method == "resolved_hvap_clapeyron_lower"
        )
        boundary_temperature = 300.0
        boundary_slope = 40000.0 / (
            R_J_MOL_K
            * boundary_temperature**2
            * _peng_robinson_delta_z_or_ideal(
                boundary_temperature,
                0.25,
                500.0,
                50.0,
                0.4,
            )
        )
        bound = clapeyron.bind(
            PsatBoundaryConditions(
                T_min=220.0,
                T_max=boundary_temperature,
                right=PsatEndpoint(
                    temperature=boundary_temperature,
                    ln_pressure=math.log(0.25),
                    dln_pressure_dT=boundary_slope,
                    source="hard",
                    method="hard_boundary",
                    quality=0.96,
                    segment_type=PsatSegmentType.PINNED,
                    derivative_basis=PsatDerivativeBasis.ANALYTIC,
                ),
            )
        )
        self.assertEqual(
            bound.metadata["deep_completion_route"],
            "peng_robinson_delta_z",
        )
        self.assertEqual(
            bound.metadata["critical_pr_quality_factor"],
            0.97,
        )

    def test_lower_aw_uses_soft_omega_only_without_a_hard_second_anchor(self):
        from property_resolution.vapor_pressure_adapter import (
            _ambrose_walton_dln_pressure_dT,
            _ambrose_walton_ln_pressure,
            _collect_anchored_ambrose_walton_lower_inputs,
        )

        component = {
            "CAS": "not-a-real-compound",
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "property_sources": {
                "Tc": {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                },
                "Pc": {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                },
                "omega": {
                    "source": "soft",
                    "method": "soft_omega",
                    "quality": 0.91,
                },
            },
        }
        inputs = PsatCanonicalizationAdapter(
            component,
            hvap_at_temperature=lambda _component, _T: None,
            input_methods=(_collect_anchored_ambrose_walton_lower_inputs,),
        ).collect_inputs(
            T_min=220.0,
            T_critical=500.0,
            P_critical_bar=50.0,
        )
        relation = inputs.relations[0]
        boundary_temperature = 300.0
        right = PsatEndpoint(
            temperature=boundary_temperature,
            ln_pressure=_ambrose_walton_ln_pressure(
                boundary_temperature,
                500.0,
                50.0,
                0.45,
            ),
            dln_pressure_dT=_ambrose_walton_dln_pressure_dT(
                boundary_temperature,
                500.0,
                0.45,
            ),
            source="hard",
            method="numerical_hard_boundary",
            quality=0.96,
            segment_type=PsatSegmentType.PINNED,
            derivative_basis=PsatDerivativeBasis.NUMERICAL,
        )

        bound = relation.bind(
            PsatBoundaryConditions(
                T_min=220.0,
                T_max=boundary_temperature,
                right=right,
            )
        )

        self.assertEqual(
            bound.metadata["bound_lower_aw_route"],
            "endpoint_to_soft_physical_omega_at_Tr_0.7",
        )
        self.assertAlmostEqual(
            bound.quality,
            0.91 * 0.94 * bound.metadata["span_quality_factor"],
        )
        self.assertTrue(any("omega admitted" in warning for warning in inputs.warnings))

    def test_no_hard_nannoolal_hands_off_c1_to_aw_at_tr_08(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_no_hard_fallback_inputs,
        )

        component = {
            "CAS": "not-a-real-compound",
            "smiles": "CCCC",
            "Tc": 500.0,
            "Pc": 50.0,
            "Tb": 350.0,
            "property_sources": {
                "Tc": {
                    "source": "estimated",
                    "method": "soft_tc",
                    "quality": 0.89,
                },
                "Pc": {
                    "source": "estimated",
                    "method": "soft_pc",
                    "quality": 0.89,
                },
                "Tb": {
                    "source": "estimated",
                    "method": "soft_tb",
                    "quality": 0.80,
                },
                "smiles": {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.99,
                },
            },
        }
        inputs = PsatCanonicalizationAdapter(
            component,
            input_methods=(_collect_no_hard_fallback_inputs,),
        ).collect_inputs(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=50.0,
            T_boiling=350.0,
        )
        upper_relation = next(
            item
            for item in inputs.relations
            if item.method == "no_hard_nannoolal_upper_aw"
        )
        middle_relation = next(
            item for item in inputs.relations if item.method == "no_hard_nannoolal"
        )
        deep_relation = next(
            item for item in inputs.relations if item.method == "no_hard_nannoolal_deep"
        )
        anchors = PsatAnchorRegistry.from_tb_tc(
            T_critical=500.0,
            P_critical_bar=50.0,
            critical_quality=0.89,
            T_boiling=350.0,
            boiling_quality=0.80,
        )
        upper = upper_relation.bind(
            PsatBoundaryConditions(
                T_min=400.0,
                T_max=500.0,
                anchors=anchors.points,
            )
        )
        middle = middle_relation.bind(
            PsatBoundaryConditions(
                T_min=200.0,
                T_max=400.0,
                anchors=anchors.points,
            )
        )
        deep = deep_relation.bind(
            PsatBoundaryConditions(
                T_min=200.0,
                T_max=400.0,
                anchors=anchors.points,
            )
        )

        self.assertTrue(
            all(relation.requires_no_hard_segments for relation in inputs.relations)
        )
        self.assertAlmostEqual(
            middle.pressure_bar(350.0),
            1.01325,
            places=10,
        )
        self.assertAlmostEqual(
            upper.pressure_bar(500.0),
            50.0,
            places=10,
        )
        handoff = 400.0
        self.assertAlmostEqual(
            middle.raw_ln_pressure(handoff),
            upper.raw_ln_pressure(handoff),
            places=12,
        )
        self.assertAlmostEqual(
            middle.raw_dln_pressure_dT(handoff),
            upper.raw_dln_pressure_dT(handoff),
            places=12,
        )
        self.assertEqual(
            upper.metadata["nannoolal_handoff_reduced_temperature"],
            0.8,
        )
        self.assertAlmostEqual(middle.quality, 0.80 * 0.85)
        self.assertAlmostEqual(deep.quality, 0.80 * 0.75)
        self.assertAlmostEqual(
            upper.quality,
            min(0.80 * 0.85, 0.89) * 0.91 * 0.95,
        )
        self.assertTrue(
            any(
                "Qualified Tc/Pc were unavailable" in warning
                for warning in inputs.warnings
            )
        )

    def test_all_estimated_no_hard_route_canonicalizes_end_to_end(self):
        from property_resolution import (
            log_pressure_weighted_psat_quality,
        )

        component = {
            "CAS": "not-a-real-compound",
            "smiles": "CCCC",
            "Tc": 500.0,
            "Pc": 50.0,
            "Tb": 350.0,
            "property_sources": {
                field_name: {
                    "source": "estimated",
                    "method": "test_estimate",
                    "quality": quality,
                }
                for field_name, quality in (
                    ("Tc", 0.80),
                    ("Pc", 0.80),
                    ("Tb", 0.80),
                    ("smiles", 0.90),
                )
            },
        }
        inputs = PsatCanonicalizationAdapter(component).collect_inputs(
            T_min=220.0,
            T_critical=500.0,
            P_critical_bar=50.0,
            T_boiling=350.0,
        )

        result = PsatCanonicalizer(
            T_min=220.0,
            T_critical=500.0,
            P_critical_bar=50.0,
            critical_quality=0.80,
            T_boiling=350.0,
            boiling_quality=0.80,
        ).canonicalize(
            (),
            inputs.relations,
            additional_anchors=inputs.additional_anchors,
            metadata={"input_warnings": inputs.warnings},
        )

        self.assertEqual(
            [item.method for item in result.completion.generated_segments],
            [
                "no_hard_nannoolal_upper_aw",
                "no_hard_nannoolal",
                "no_hard_nannoolal_deep",
            ],
        )
        self.assertEqual(result.curve.form, CanonicalPsatForm.AF)
        self.assertAlmostEqual(
            result.curve.pressure_bar(350.0),
            1.01325,
            places=8,
        )
        self.assertAlmostEqual(
            result.curve.pressure_bar(500.0),
            50.0,
            places=8,
        )
        expected_quality, breakdown = log_pressure_weighted_psat_quality(
            result.completion.assembly,
        )
        self.assertAlmostEqual(
            result.curve.metadata["fit_original_overall_quality"],
            expected_quality,
        )
        self.assertAlmostEqual(
            result.curve.quality,
            max(
                0.0,
                expected_quality - result.curve.metadata["fit_quality_penalty"],
            ),
        )
        self.assertEqual(
            result.curve.metadata["quality_aggregation"]["basis"],
            "log_pressure_weighted_segment_average",
        )
        self.assertEqual(
            result.curve.metadata["quality_aggregation"]["middle_pressure_weight"],
            3.0,
        )
        self.assertEqual(len(breakdown), 3)
        self.assertEqual(
            {round(item.quality, 6) for item in result.curve.provenance},
            {
                round(0.80 * 0.75, 6),
                round(0.80 * 0.85, 6),
                round(
                    min(0.80 * 0.85, 0.80) * 0.91 * 0.95,
                    6,
                ),
            },
        )
        self.assertTrue(result.curve.diagnostics.monotonic)
        self.assertTrue(
            math.isfinite(result.curve.diagnostics.p95_absolute_relative_error_percent)
        )
        self.assertTrue(result.curve.diagnostics.monotonic)
        self.assertTrue(
            any(
                "Qualified Tc/Pc were unavailable" in warning
                for warning in result.curve.metadata["input_warnings"]
            )
        )

    def test_unusable_soft_critical_nannoolal_falls_back_to_no_hard_aw(self):
        cases = (
            {
                "label": "Tb above fixed Nannoolal handoff",
                "CAS": "112-79-8",
                "smiles": "CCCCCCCCC=CCCCCCCCC(=O)O",
                "Tm": 317.55,
                "Tb": 657.5454194851694,
                "Tc": 813.7411534229033,
                "Pc": 13.45061269399174,
                "qualities": (0.75, 0.80, 0.80),
            },
            {
                "label": "Nannoolal do-not-estimate functional groups",
                "CAS": "150-13-0",
                "smiles": "C1=CC(=CC=C1C(=O)O)N",
                "Tm": 461.65,
                "Tb": 582.0673095421237,
                "Tc": 873.1009643131855,
                "Pc": 36.16286372673303,
                "qualities": (0.44, 0.55, 0.55),
            },
        )

        for case in cases:
            with self.subTest(case["label"]):
                tc_quality, pc_quality, tb_quality = case["qualities"]
                component = {
                    key: case[key] for key in ("CAS", "smiles", "Tm", "Tb", "Tc", "Pc")
                }
                component["property_sources"] = {
                    "Tm": {
                        "source": "local",
                        "method": "paired_fusion_transition",
                        "quality": 0.95,
                    },
                    "Tc": {
                        "source": "estimated",
                        "method": "test_tc",
                        "quality": tc_quality,
                    },
                    "Pc": {
                        "source": "estimated",
                        "method": "test_pc",
                        "quality": pc_quality,
                    },
                    "Tb": {
                        "source": "estimated",
                        "method": "test_tb",
                        "quality": tb_quality,
                    },
                    "smiles": {
                        "source": "reported",
                        "method": "test_smiles",
                        "quality": 0.99,
                    },
                }
                adapter = PsatCanonicalizationAdapter(component)
                domain = adapter.resolve_domain(T_critical=case["Tc"])
                inputs = adapter.collect_inputs(
                    T_min=domain.T_min,
                    T_critical=case["Tc"],
                    P_critical_bar=case["Pc"],
                )

                self.assertTrue(inputs.relations)
                self.assertIn(
                    "no_hard_ambrose_walton",
                    {item.method for item in inputs.relations},
                )
                self.assertTrue(
                    any(
                        "usable no-hard Nannoolal" in warning
                        for warning in inputs.warnings
                    )
                )

                result = PsatCanonicalizer(
                    T_min=domain.T_min,
                    T_critical=case["Tc"],
                    P_critical_bar=case["Pc"],
                    critical_quality=min(tc_quality, pc_quality),
                ).canonicalize(
                    (),
                    inputs.relations,
                    additional_anchors=inputs.additional_anchors,
                    metadata={"input_warnings": inputs.warnings},
                )
                self.assertAlmostEqual(result.curve.T_min, case["Tm"])
                self.assertGreater(result.curve.pressure_bar(case["Tm"]), 0.0)
                self.assertEqual(
                    {item.method for item in result.completion.generated_segments},
                    {"no_hard_ambrose_walton"},
                )
                self.assertAlmostEqual(
                    result.curve.pressure_bar(case["Tc"]),
                    case["Pc"],
                )

    def test_no_hard_aw_then_pr_clapeyron_completes_real_perry_curve(self):
        from scipy.optimize import brentq

        from perry_properties import get_perry_property_library
        from property_resolution.vapor_pressure_adapter import (
            _collect_deep_vacuum_inputs,
            _collect_no_hard_fallback_inputs,
            _collect_perry_2_8_inputs,
        )

        library = get_perry_property_library()
        library._load()
        entry = library.get("ethylbenzene", expand_identity=False)
        critical = entry["critical_constants"]
        Tc = float(critical["Tc_K"])
        Pc_bar = float(critical["Pc_MPa"]) * 10.0
        Tb = float(library.normal_boiling_point_K(entry["cas"]).value)
        component = {
            "name": "ethylbenzene",
            "CAS": entry["cas"],
            "Tc": Tc,
            "Pc": Pc_bar,
            "omega": float(critical["omega"]),
            "Tb": Tb,
            "property_sources": {
                field_name: {
                    "source": "Perry 9th",
                    "method": "perry",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb")
            },
        }
        reference = (
            PsatCanonicalizationAdapter(
                component,
                input_methods=(_collect_perry_2_8_inputs,),
            )
            .collect_inputs(
                T_critical=Tc,
                P_critical_bar=Pc_bar,
                T_boiling=Tb,
            )
            .segments[0]
        )

        def crossing(pressure):
            return brentq(
                lambda T: reference.raw_ln_pressure(T) - math.log(pressure),
                reference.T_min,
                min(reference.T_max, Tc),
            )

        T_min = crossing(0.001)
        inputs = PsatCanonicalizationAdapter(
            component,
            input_methods=(
                _collect_no_hard_fallback_inputs,
                _collect_deep_vacuum_inputs,
            ),
        ).collect_inputs(
            T_min=T_min,
            T_critical=Tc,
            P_critical_bar=Pc_bar,
            T_boiling=Tb,
        )
        assembler = PsatSegmentAssembler(T_min, Tc)
        anchors = PsatAnchorRegistry.from_tb_tc(
            T_critical=Tc,
            P_critical_bar=Pc_bar,
            critical_quality=0.98,
            T_boiling=Tb,
            boiling_quality=0.98,
        )
        for anchor in inputs.additional_anchors:
            anchors.add(anchor)

        completion = PsatCompletionCoordinator(
            assembler,
            inputs.relations,
            anchors,
        ).complete()

        self.assertEqual(
            [item.method for item in completion.generated_segments],
            [
                "no_hard_ambrose_walton",
                "resolved_hvap_clapeyron_lower",
            ],
        )
        self.assertEqual(
            completion.generated_segments[0].metadata["no_hard_route"],
            "tb_variable_omega",
        )
        self.assertEqual(
            completion.generated_segments[1].metadata["deep_completion_route"],
            "peng_robinson_delta_z",
        )
        self.assertAlmostEqual(
            math.exp(completion.assembly.ln_pressure(Tb)),
            1.01325,
            places=10,
        )
        self.assertAlmostEqual(
            math.exp(completion.assembly.ln_pressure(Tc)),
            Pc_bar,
            places=10,
        )
        for pressure in (0.25, 0.05, 0.001):
            with self.subTest(pressure=pressure):
                prediction = math.exp(
                    completion.assembly.ln_pressure(crossing(pressure))
                )
                self.assertLess(
                    abs(prediction / pressure - 1.0),
                    0.015,
                )

    def test_deep_fallback_waits_for_tb_effective_no_hard_switch(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_deep_vacuum_inputs,
            _collect_no_hard_fallback_inputs,
        )

        for cas, expected_route in (
            ("64-19-7", "calibrated_monoacid_dimer"),
            ("71-43-2", "peng_robinson_delta_z"),
        ):
            with self.subTest(cas=cas):
                component = {
                    "name": "synthetic component",
                    "CAS": cas,
                    "Tc": 500.0,
                    "Pc": 50.0,
                    "omega": 0.4,
                    "Tb": 360.0,
                    "property_sources": {
                        field_name: {
                            "source": "reported",
                            "method": "reported",
                            "quality": 0.98,
                        }
                        for field_name in ("Tc", "Pc", "omega", "Tb")
                    },
                }
                inputs = PsatCanonicalizationAdapter(
                    component,
                    hvap_at_temperature=lambda _component, _T: PropertyResolutionResult(
                        value=30.0,
                        source="reported",
                        method="provided_hvap_fit",
                        quality=0.97,
                    ),
                    input_methods=(
                        _collect_no_hard_fallback_inputs,
                        _collect_deep_vacuum_inputs,
                    ),
                ).collect_inputs(
                    T_min=220.0,
                    T_critical=500.0,
                    P_critical_bar=50.0,
                    T_boiling=360.0,
                )
                fallback = next(
                    item
                    for item in inputs.relations
                    if item.method == "dynamic_omega_deep_fallback"
                )
                self.assertEqual(fallback.P_max_bar, 0.25)

                anchors = PsatAnchorRegistry.from_tb_tc(
                    T_critical=500.0,
                    P_critical_bar=50.0,
                    critical_quality=0.98,
                    T_boiling=360.0,
                    boiling_quality=0.98,
                )
                for anchor in inputs.additional_anchors:
                    anchors.add(anchor)
                completion = PsatCompletionCoordinator(
                    PsatSegmentAssembler(220.0, 500.0),
                    inputs.relations,
                    anchors,
                ).complete()

                self.assertEqual(
                    [item.method for item in completion.generated_segments],
                    [
                        "no_hard_ambrose_walton",
                        "no_hard_ambrose_walton",
                        "resolved_hvap_clapeyron_lower",
                    ],
                )
                self.assertEqual(
                    [
                        item.metadata.get("no_hard_quality_band")
                        for item in completion.generated_segments[:2]
                    ],
                    ["tb_to_critical", "quarter_bar_to_tb"],
                )
                deep = completion.generated_segments[-1]
                self.assertEqual(
                    deep.metadata["deep_completion_route"],
                    expected_route,
                )
                self.assertAlmostEqual(
                    deep.pressure_bar(deep.T_max),
                    0.25,
                    places=10,
                )
                self.assertNotIn(
                    "dynamic_omega_deep_fallback",
                    {item.method for item in completion.generated_segments},
                )
                self.assertTrue(
                    any(
                        "Deep AW fallback boundary is above its qualified switch"
                        in item
                        for item in completion.rejected_attempts
                    )
                )

    def test_deep_fallback_does_not_preempt_no_hard_nannoolal(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_deep_vacuum_inputs,
            _collect_no_hard_fallback_inputs,
        )

        component = {
            "CAS": "106-97-8",
            "smiles": "CCCC",
            "Tc": 500.0,
            "Pc": 50.0,
            "Tb": 350.0,
            "property_sources": {
                "Tc": {
                    "source": "estimated",
                    "method": "soft_tc",
                    "quality": 0.89,
                },
                "Pc": {
                    "source": "estimated",
                    "method": "soft_pc",
                    "quality": 0.89,
                },
                "Tb": {
                    "source": "estimated",
                    "method": "soft_tb",
                    "quality": 0.80,
                },
                "smiles": {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.99,
                },
            },
        }
        inputs = PsatCanonicalizationAdapter(
            component,
            hvap_at_temperature=lambda _component, _T: PropertyResolutionResult(
                value=30.0,
                source="reported",
                method="provided_hvap_fit",
                quality=0.97,
            ),
            input_methods=(
                _collect_no_hard_fallback_inputs,
                _collect_deep_vacuum_inputs,
            ),
        ).collect_inputs(
            T_min=220.0,
            T_critical=500.0,
            P_critical_bar=50.0,
            T_boiling=350.0,
        )
        anchors = PsatAnchorRegistry.from_tb_tc(
            T_critical=500.0,
            P_critical_bar=50.0,
            critical_quality=0.89,
            T_boiling=350.0,
            boiling_quality=0.80,
        )
        for anchor in inputs.additional_anchors:
            anchors.add(anchor)
        completion = PsatCompletionCoordinator(
            PsatSegmentAssembler(220.0, 500.0),
            inputs.relations,
            anchors,
        ).complete()

        self.assertEqual(
            [item.method for item in completion.generated_segments],
            [
                "no_hard_nannoolal_upper_aw",
                "no_hard_nannoolal",
                "resolved_hvap_clapeyron_lower",
            ],
        )
        deep = completion.generated_segments[-1]
        self.assertEqual(
            deep.metadata["deep_completion_route"],
            "ideal_delta_z",
        )
        self.assertAlmostEqual(
            deep.pressure_bar(deep.T_max),
            0.05,
            places=10,
        )
        self.assertNotIn(
            "dynamic_omega_deep_fallback",
            {item.method for item in completion.generated_segments},
        )
        self.assertTrue(
            any(
                "Deep AW fallback boundary is above its qualified switch" in item
                for item in completion.rejected_attempts
            )
        )

    def test_log_temperature_hermite_fills_a_two_hard_source_middle_gap(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_log_temperature_middle_gap_inputs,
        )

        def ln_pressure(T):
            coordinate = math.log(T / 300.0)
            return (
                math.log(0.5)
                + 8.0 * coordinate
                + 0.5 * coordinate**2
                - 0.1 * coordinate**3
            )

        def derivative(T):
            coordinate = math.log(T / 300.0)
            return (8.0 + coordinate - 0.3 * coordinate**2) / T

        relation = (
            PsatCanonicalizationAdapter(
                {},
                input_methods=(_collect_log_temperature_middle_gap_inputs,),
            )
            .collect_inputs(
                T_min=250.0,
                T_critical=500.0,
            )
            .relations[0]
        )
        self.assertEqual(
            relation.priority,
            PsatPriority.MIDDLE_COMPLETION,
        )
        self.assertGreater(
            relation.priority,
            PsatPriority.AMBROSE_WALTON,
        )
        self.assertEqual(
            relation.allowed_left_segment_types,
            (
                PsatSegmentType.PINNED,
                PsatSegmentType.CANONICAL_OVERRIDE,
            ),
        )
        self.assertEqual(
            relation.allowed_right_segment_types,
            relation.allowed_left_segment_types,
        )

        left = PsatSegment(
            source="left",
            method="left_hard",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=250.0,
            T_max=320.0,
            ln_pressure_function=ln_pressure,
            derivative_function=derivative,
            quality=0.96,
        )
        right = PsatSegment(
            source="right",
            method="right_hard",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=400.0,
            T_max=500.0,
            ln_pressure_function=ln_pressure,
            derivative_function=derivative,
            quality=0.94,
        )
        assembler = PsatSegmentAssembler(250.0, 500.0)
        assembler.extend((left, right))

        completion = PsatCompletionCoordinator(
            assembler,
            (relation,),
        ).complete()
        middle = completion.generated_segments[0]

        self.assertEqual(
            middle.method,
            "log_temperature_hermite_middle",
        )
        self.assertAlmostEqual(middle.quality, 0.94 * 0.97)
        self.assertEqual(
            middle.metadata["bridge_coordinate"],
            "log_temperature",
        )
        self.assertEqual(
            middle.metadata["quality_factor"],
            0.97,
        )
        for index in range(41):
            temperature = 320.0 + 80.0 * index / 40.0
            with self.subTest(temperature=temperature):
                self.assertAlmostEqual(
                    middle.ln_pressure(temperature),
                    ln_pressure(temperature),
                    places=12,
                )
                self.assertAlmostEqual(
                    middle.dln_pressure_dT(temperature),
                    derivative(temperature),
                    places=12,
                )
        self.assertTrue(
            all(
                assessment.accepted
                for assessment in completion.assembly.assess_junctions()
            )
        )

    def test_log_temperature_middle_completion_reconstructs_perry_gap(self):
        from scipy.optimize import brentq

        from perry_properties import get_perry_property_library
        from property_resolution.vapor_pressure_adapter import (
            _collect_log_temperature_middle_gap_inputs,
            _collect_perry_2_8_inputs,
        )

        library = get_perry_property_library()
        library._load()
        entry = library.get("ethylbenzene", expand_identity=False)
        critical = entry["critical_constants"]
        Tc = float(critical["Tc_K"])
        Pc_bar = float(critical["Pc_MPa"]) * 10.0
        component = {
            "name": "ethylbenzene",
            "CAS": entry["cas"],
            "Tc": Tc,
            "Pc": Pc_bar,
            "property_sources": {
                field_name: {
                    "source": "Perry 9th",
                    "method": "perry",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc")
            },
        }
        inputs = PsatCanonicalizationAdapter(
            component,
            input_methods=(
                _collect_perry_2_8_inputs,
                _collect_log_temperature_middle_gap_inputs,
            ),
        ).collect_inputs(
            T_min=float(entry["vapor_pressure"][0]["T_min_K"]),
            T_critical=Tc,
            P_critical_bar=Pc_bar,
        )
        source = inputs.segments[0]

        def crossing(pressure):
            return brentq(
                lambda T: source.raw_ln_pressure(T) - math.log(pressure),
                source.T_min,
                source.T_max,
            )

        left_temperature = crossing(0.5)
        right_temperature = crossing(5.0)
        assembler = PsatSegmentAssembler(
            source.T_min,
            source.T_max,
        )
        assembler.extend(
            (
                source.clipped(source.T_min, left_temperature),
                source.clipped(right_temperature, source.T_max),
            )
        )
        completion = PsatCompletionCoordinator(
            assembler,
            inputs.relations,
        ).complete()
        middle = completion.generated_segments[0]
        errors = []
        for index in range(101):
            temperature = (
                left_temperature
                + (right_temperature - left_temperature) * index / 100.0
            )
            errors.append(
                abs(
                    math.exp(
                        middle.raw_ln_pressure(temperature)
                        - source.raw_ln_pressure(temperature)
                    )
                    - 1.0
                )
            )

        self.assertEqual(
            middle.method,
            "log_temperature_hermite_middle",
        )
        self.assertLess(sum(errors) / len(errors), 0.001)
        self.assertLess(max(errors), 0.002)
        self.assertAlmostEqual(middle.quality, 0.98 * 0.97)

    def test_anchored_aw_declares_a_hard_c1_upper_completion(self):
        component = {
            "name": "limited-range compound",
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "property_sources": {
                field_name: {
                    "source": "physical source",
                    "method": "reported",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega")
            },
        }
        inputs = PsatCanonicalizationAdapter(component).collect_inputs(
            T_min=250.0,
            T_critical=500.0,
            P_critical_bar=50.0,
        )
        relation = next(
            item
            for item in inputs.relations
            if item.method == "anchored_ambrose_walton_upper"
        )

        self.assertEqual(relation.priority, PsatPriority.AMBROSE_WALTON)
        self.assertEqual(relation.T_min, 250.0)
        self.assertEqual(relation.T_max, 500.0)
        self.assertAlmostEqual(relation.quality, 0.98 * 0.91)
        self.assertEqual(
            relation.allowed_left_segment_types,
            (
                PsatSegmentType.PINNED,
                PsatSegmentType.CANONICAL_OVERRIDE,
            ),
        )
        self.assertEqual(relation.required_anchor_names, ("Tc",))
        self.assertEqual(
            relation.left_derivative_requirement.value,
            "required",
        )
        self.assertEqual(
            relation.metadata["omega_anchor_reduced_temperature"],
            0.7,
        )

    def test_lower_aw_uses_dynamic_omega_for_an_analytic_hard_derivative(self):
        from property_resolution.vapor_pressure_adapter import (
            _ambrose_walton_dln_pressure_dT,
            _ambrose_walton_ln_pressure,
            _ambrose_walton_omega_sensitivity,
        )

        component = {
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "Tb": 350.0,
            "property_sources": {
                field_name: {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb")
            },
        }
        relation = next(
            item
            for item in PsatCanonicalizationAdapter(
                component,
                hvap_at_temperature=lambda _component, _T: PropertyResolutionResult(
                    value=40.0,
                    source="provided",
                    method="provided_hvap_fit",
                    quality=0.98,
                ),
            )
            .collect_inputs(
                T_min=200.0,
                T_critical=500.0,
                P_critical_bar=50.0,
                T_boiling=350.0,
            )
            .relations
            if item.method == "anchored_ambrose_walton_lower"
        )
        hard_temperature = 330.0
        endpoint_omega = 0.45
        omega_slope = 1.0e-4
        endpoint_ln_pressure = _ambrose_walton_ln_pressure(
            hard_temperature,
            500.0,
            50.0,
            endpoint_omega,
        )
        endpoint_slope = (
            _ambrose_walton_dln_pressure_dT(
                hard_temperature,
                500.0,
                endpoint_omega,
            )
            + _ambrose_walton_omega_sensitivity(
                hard_temperature,
                500.0,
                endpoint_omega,
            )
            * omega_slope
        )
        hard = PsatSegment(
            source="measured",
            method="analytic_hard_curve",
            segment_type=PsatSegmentType.PINNED,
            priority=800,
            T_min=hard_temperature,
            T_max=380.0,
            ln_pressure_function=lambda T: (
                endpoint_ln_pressure + endpoint_slope * (T - hard_temperature)
            ),
            derivative_function=lambda _T: endpoint_slope,
            quality=0.97,
        )
        boiling = PsatEndpoint.fixed_pressure_point(
            350.0,
            1.01325,
            source="fixed_anchor",
            method="normal_boiling_point",
            quality=0.98,
            context={"anchor_name": "Tb"},
        )
        bound = relation.bind(
            PsatBoundaryConditions(
                T_min=200.0,
                T_max=hard_temperature,
                right=PsatEndpoint.from_segment(hard, hard_temperature),
                anchors=(boiling,),
            )
        )
        trimmed = trim_psat_segment_to_validity(bound)

        self.assertEqual(
            relation.right_derivative_requirement.value,
            "required",
        )
        self.assertEqual(relation.P_min_bar, 0.25)
        self.assertEqual(
            bound.metadata["bound_lower_aw_route"],
            "endpoint_slope_dynamic_omega",
        )
        self.assertIsNone(bound.metadata["bound_correction_basis"])
        self.assertAlmostEqual(
            bound.metadata["bound_omega_slope_per_K"],
            omega_slope,
            places=12,
        )
        self.assertAlmostEqual(
            bound.ln_pressure(hard_temperature),
            endpoint_ln_pressure,
            places=12,
        )
        self.assertAlmostEqual(
            bound.dln_pressure_dT(hard_temperature),
            endpoint_slope,
            places=12,
        )
        self.assertAlmostEqual(trimmed.pressure_bar(trimmed.T_min), 0.25)
        self.assertAlmostEqual(
            bound.quality,
            0.97 * 0.95 * bound.metadata["span_quality_factor"],
        )

    def test_lower_aw_uses_tb_anchor_for_an_interpolated_hard_derivative(self):
        from property_resolution.vapor_pressure_adapter import (
            _ambrose_walton_dln_pressure_dT,
            _ambrose_walton_ln_pressure,
            _ambrose_walton_omega_from_pressure,
            _ambrose_walton_omega_sensitivity,
        )

        component = {
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "Tb": 350.0,
            "property_sources": {
                field_name: {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb")
            },
        }
        relation = next(
            item
            for item in PsatCanonicalizationAdapter(
                component,
                hvap_at_temperature=lambda _component, _T: PropertyResolutionResult(
                    value=40.0,
                    source="provided",
                    method="provided_hvap_fit",
                    quality=0.98,
                ),
            )
            .collect_inputs(
                T_min=200.0,
                T_critical=500.0,
                P_critical_bar=50.0,
                T_boiling=350.0,
            )
            .relations
            if item.method == "anchored_ambrose_walton_lower"
        )
        hard_temperature = 330.0
        endpoint_omega = 0.68
        endpoint_ln_pressure = _ambrose_walton_ln_pressure(
            hard_temperature,
            500.0,
            50.0,
            endpoint_omega,
        )
        boiling_omega = _ambrose_walton_omega_from_pressure(
            350.0,
            math.log(1.01325),
            500.0,
            50.0,
            preferred_omega=0.4,
        )
        omega_slope = (boiling_omega - endpoint_omega) / (350.0 - hard_temperature)
        endpoint_slope = (
            _ambrose_walton_dln_pressure_dT(
                hard_temperature,
                500.0,
                endpoint_omega,
            )
            + _ambrose_walton_omega_sensitivity(
                hard_temperature,
                500.0,
                endpoint_omega,
            )
            * omega_slope
            + 0.0001
        )
        hard = PsatSegment(
            source="table",
            method="interpolated_hard_curve",
            segment_type=PsatSegmentType.PINNED,
            priority=500,
            T_min=hard_temperature,
            T_max=380.0,
            ln_pressure_function=lambda T: (
                endpoint_ln_pressure + endpoint_slope * (T - hard_temperature)
            ),
            derivative_function=lambda _T: endpoint_slope,
            derivative_basis=PsatDerivativeBasis.INTERPOLATED,
            quality=0.95,
        )
        boiling = PsatEndpoint.fixed_pressure_point(
            350.0,
            1.01325,
            source="fixed_anchor",
            method="normal_boiling_point",
            quality=0.97,
            context={"anchor_name": "Tb"},
        )
        bound = relation.bind(
            PsatBoundaryConditions(
                T_min=200.0,
                T_max=hard_temperature,
                right=PsatEndpoint.from_segment(hard, hard_temperature),
                anchors=(boiling,),
            )
        )
        trimmed = trim_psat_segment_to_validity(bound)

        self.assertEqual(
            bound.metadata["bound_lower_aw_route"],
            "endpoint_to_tb_effective_omega",
        )
        self.assertEqual(
            bound.metadata["bound_correction_basis"],
            "affine_inverse_temperature",
        )
        self.assertEqual(
            bound.metadata["hard_derivative_basis"],
            "interpolated",
        )
        self.assertAlmostEqual(
            bound.ln_pressure(hard_temperature),
            endpoint_ln_pressure,
            places=12,
        )
        self.assertAlmostEqual(
            bound.dln_pressure_dT(hard_temperature),
            endpoint_slope,
            places=12,
        )
        self.assertAlmostEqual(trimmed.pressure_bar(trimmed.T_min), 0.25)
        self.assertAlmostEqual(
            bound.quality,
            0.95 * 0.94 * bound.metadata["span_quality_factor"],
        )

    def test_lower_aw_localizes_an_above_tb_hard_slope(self):
        from property_resolution.vapor_pressure_adapter import (
            _ambrose_walton_dln_pressure_dT,
            _ambrose_walton_ln_pressure,
            _ambrose_walton_omega_from_pressure,
        )

        component = {
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "Tb": 350.0,
            "property_sources": {
                field_name: {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb")
            },
        }
        relation = next(
            item
            for item in PsatCanonicalizationAdapter(component)
            .collect_inputs(
                T_min=200.0,
                T_critical=500.0,
                P_critical_bar=50.0,
                T_boiling=350.0,
            )
            .relations
            if item.method == "anchored_ambrose_walton_lower"
        )
        hard_temperature = 370.0
        endpoint_omega = 0.44
        endpoint_ln_pressure = _ambrose_walton_ln_pressure(
            hard_temperature,
            500.0,
            50.0,
            endpoint_omega,
        )
        endpoint_slope = (
            _ambrose_walton_dln_pressure_dT(
                hard_temperature,
                500.0,
                endpoint_omega,
            )
            + 0.001
        )
        hard = PsatSegment(
            source="measured",
            method="above_tb_hard_curve",
            segment_type=PsatSegmentType.PINNED,
            priority=800,
            T_min=hard_temperature,
            T_max=400.0,
            ln_pressure_function=lambda T: (
                endpoint_ln_pressure + endpoint_slope * (T - hard_temperature)
            ),
            derivative_function=lambda _T: endpoint_slope,
            quality=0.97,
        )
        boiling = PsatEndpoint.fixed_pressure_point(
            350.0,
            1.01325,
            source="fixed_anchor",
            method="normal_boiling_point",
            quality=0.96,
            context={"anchor_name": "Tb"},
        )
        bound = relation.bind(
            PsatBoundaryConditions(
                T_min=200.0,
                T_max=hard_temperature,
                right=PsatEndpoint.from_segment(hard, hard_temperature),
                anchors=(boiling,),
            )
        )

        self.assertEqual(
            bound.metadata["bound_lower_aw_route"],
            "endpoint_to_tb_effective_omega",
        )
        self.assertEqual(
            bound.metadata["bound_correction_basis"],
            "Tb_localized_quintic",
        )
        self.assertAlmostEqual(bound.pressure_bar(350.0), 1.01325)
        self.assertAlmostEqual(
            bound.ln_pressure(hard_temperature),
            endpoint_ln_pressure,
            places=12,
        )
        self.assertAlmostEqual(
            bound.dln_pressure_dT(hard_temperature),
            endpoint_slope,
            places=12,
        )
        omega_tb = _ambrose_walton_omega_from_pressure(
            350.0,
            math.log(1.01325),
            500.0,
            50.0,
            preferred_omega=0.4,
        )
        self.assertAlmostEqual(
            bound.metadata["bound_anchor_omega"],
            omega_tb,
        )
        step = 1.0e-3
        left_slope = (bound.ln_pressure(350.0) - bound.ln_pressure(350.0 - step)) / step
        right_slope = (
            bound.ln_pressure(350.0 + step) - bound.ln_pressure(350.0)
        ) / step
        self.assertLess(abs(left_slope - right_slope), 2.0e-6)
        self.assertAlmostEqual(
            bound.quality,
            0.96 * 0.94 * bound.metadata["span_quality_factor"],
        )

    def test_lower_aw_reproduces_perry_tb_to_switch_benchmark(self):
        from scipy.optimize import brentq

        from perry_properties import get_perry_property_library
        from property_resolution.vapor_pressure_adapter import (
            _collect_anchored_ambrose_walton_lower_inputs,
            _collect_perry_2_8_inputs,
        )

        library = get_perry_property_library()
        library._load()
        curve_mards = []
        for entry in library.chemicals.values():
            if not entry.get("vapor_pressure") or not entry.get("critical_constants"):
                continue
            critical = entry["critical_constants"]
            Tc = float(critical["Tc_K"])
            Pc_bar = float(critical["Pc_MPa"]) * 10.0
            Tb_result = library.normal_boiling_point_K(entry["cas"])
            if Tb_result is None:
                continue
            Tb = float(Tb_result.value)
            component = {
                "name": entry.get("name"),
                "CAS": entry.get("cas"),
                "Tc": Tc,
                "Pc": Pc_bar,
                "Tb": Tb,
                "property_sources": {
                    field_name: {
                        "source": "Perry 9th",
                        "method": "perry",
                        "quality": 0.98,
                    }
                    for field_name in ("Tc", "Pc", "Tb")
                },
            }
            inputs = PsatCanonicalizationAdapter(
                component,
                input_methods=(
                    _collect_perry_2_8_inputs,
                    _collect_anchored_ambrose_walton_lower_inputs,
                ),
            ).collect_inputs(
                T_min=float(entry["vapor_pressure"][0]["T_min_K"]),
                T_critical=Tc,
                P_critical_bar=Pc_bar,
                T_boiling=Tb,
            )
            source = next(
                item
                for item in inputs.segments
                if item.method == "perry_2_8_vapor_pressure"
            )
            if not source.covers_temperature(Tb):
                continue
            target = math.log(0.25)
            if (
                source.raw_ln_pressure(source.T_min) > target
                or source.raw_ln_pressure(Tb) < target
            ):
                continue
            relation = next(
                item
                for item in inputs.relations
                if item.method == "anchored_ambrose_walton_lower"
            )
            boiling = PsatEndpoint.fixed_pressure_point(
                Tb,
                1.01325,
                source="fixed_anchor",
                method="normal_boiling_point",
                quality=0.98,
                context={"anchor_name": "Tb"},
            )
            bound = relation.bind(
                PsatBoundaryConditions(
                    T_min=source.T_min,
                    T_max=Tb,
                    right=PsatEndpoint.from_segment(source, Tb),
                    anchors=(boiling,),
                )
            )
            trim_psat_segment_to_validity(bound)
            reference_switch_temperature = brentq(
                lambda T: source.raw_ln_pressure(T) - target,
                source.T_min,
                Tb,
            )
            errors = []
            for index in range(161):
                ln_pressure = (
                    math.log(0.25)
                    + (math.log(1.01325) - math.log(0.25)) * index / 160.0
                )
                if index == 0:
                    temperature = reference_switch_temperature
                elif index == 160:
                    temperature = Tb
                else:
                    temperature = brentq(
                        lambda T: source.raw_ln_pressure(T) - ln_pressure,
                        reference_switch_temperature,
                        Tb,
                    )
                errors.append(
                    abs(
                        math.exp(bound.raw_ln_pressure(temperature))
                        / math.exp(ln_pressure)
                        - 1.0
                    )
                )
            curve_mards.append(sum(errors) / len(errors))

        ordered = sorted(curve_mards)
        p95_index = round(0.95 * (len(ordered) - 1))
        self.assertEqual(len(curve_mards), 334)
        self.assertLess(median(curve_mards), 0.0015)
        self.assertLess(ordered[p95_index], 0.0065)
        self.assertLess(max(curve_mards), 0.035)

    def test_hvap_quality_selects_deep_switch_pressure(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_anchored_ambrose_walton_lower_inputs,
            _collect_deep_vacuum_inputs,
        )

        component = {
            "CAS": "71-43-2",
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "Tb": 350.0,
            "property_sources": {
                field_name: {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb")
            },
        }

        def collect(quality, method="provided_hvap_fit"):
            adapter = PsatCanonicalizationAdapter(
                component,
                hvap_at_temperature=lambda _component, T: PropertyResolutionResult(
                    value=45.0 * (1.0 - T / 500.0) ** 0.38,
                    source="source-backed",
                    method=method,
                    quality=quality,
                ),
                input_methods=(
                    _collect_anchored_ambrose_walton_lower_inputs,
                    _collect_deep_vacuum_inputs,
                ),
            )
            return adapter.collect_inputs(
                T_min=220.0,
                T_critical=500.0,
                P_critical_bar=50.0,
                T_boiling=350.0,
            )

        for quality, expected_switch in (
            (0.97, 0.25),
            (0.95, 0.25),
            (0.949999, 0.10),
            (0.92, 0.10),
            (0.90, 0.10),
            (0.899999, 0.05),
            (0.85, 0.05),
            (0.80, 0.05),
        ):
            with self.subTest(quality=quality):
                inputs = collect(quality)
                lower_aw = next(
                    item
                    for item in inputs.relations
                    if item.method == "anchored_ambrose_walton_lower"
                )
                clapeyron = next(
                    item
                    for item in inputs.relations
                    if item.method == "resolved_hvap_clapeyron_lower"
                )
                self.assertEqual(lower_aw.P_min_bar, expected_switch)
                self.assertEqual(clapeyron.P_max_bar, expected_switch)

        one_point = collect(0.99, method="watson_hvap")
        self.assertEqual(
            next(
                item
                for item in one_point.relations
                if item.method == "anchored_ambrose_walton_lower"
            ).P_min_bar,
            0.001,
        )
        self.assertFalse(
            any(
                item.method == "resolved_hvap_clapeyron_lower"
                for item in one_point.relations
            )
        )
        multi_point = collect(
            0.92,
            method="nist_hvap_watson_fit",
        )
        self.assertEqual(
            next(
                item
                for item in multi_point.relations
                if item.method == "resolved_hvap_clapeyron_lower"
            ).P_max_bar,
            0.10,
        )

        unavailable = collect(0.799999)
        self.assertEqual(
            next(
                item
                for item in unavailable.relations
                if item.method == "anchored_ambrose_walton_lower"
            ).P_min_bar,
            0.001,
        )
        self.assertFalse(
            any(
                item.method == "resolved_hvap_clapeyron_lower"
                for item in unavailable.relations
            )
        )

    def test_frozen_hvap_requires_one_complete_stable_source_method(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_anchored_ambrose_walton_lower_inputs,
            _collect_deep_vacuum_inputs,
        )

        component = {
            "CAS": "71-43-2",
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "Tb": 350.0,
            "property_sources": {
                field_name: {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb")
            },
        }

        def collect(callback):
            return PsatCanonicalizationAdapter(
                component,
                hvap_at_temperature=callback,
                input_methods=(
                    _collect_anchored_ambrose_walton_lower_inputs,
                    _collect_deep_vacuum_inputs,
                ),
            ).collect_inputs(
                T_min=220.0,
                T_critical=500.0,
                P_critical_bar=50.0,
                T_boiling=350.0,
            )

        def result(T, *, source="stable", method="provided_hvap_fit", quality=0.97):
            return PropertyResolutionResult(
                value=45.0 * (1.0 - T / 500.0) ** 0.38,
                source=source,
                method=method,
                quality=quality,
            )

        minimum_quality = collect(
            lambda _component, T: result(
                T,
                quality=0.89 if T < 230.0 else 0.97,
            )
        )
        self.assertEqual(
            next(
                item
                for item in minimum_quality.relations
                if item.method == "resolved_hvap_clapeyron_lower"
            ).P_max_bar,
            0.05,
        )

        rejected_callbacks = {
            "source switch": lambda _component, T: result(
                T,
                source="lower" if T < 300.0 else "upper",
            ),
            "method switch": lambda _component, T: result(
                T,
                method=("provided_hvap_fit" if T < 300.0 else "nist_hvap_watson_fit"),
            ),
            "incomplete range": lambda _component, T: None if T < 230.0 else result(T),
            "unsupported method": lambda _component, T: result(
                T,
                method="trouton_rule",
            ),
        }
        for label, callback in rejected_callbacks.items():
            with self.subTest(label=label):
                inputs = collect(callback)
                self.assertFalse(
                    any(
                        item.method == "resolved_hvap_clapeyron_lower"
                        for item in inputs.relations
                    )
                )
                self.assertEqual(
                    next(
                        item
                        for item in inputs.relations
                        if item.method == "anchored_ambrose_walton_lower"
                    ).P_min_bar,
                    0.001,
                )

    def test_deep_clapeyron_uses_pr_then_ideal_delta_z(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_deep_vacuum_inputs,
            _peng_robinson_delta_z_or_ideal,
        )

        def component(include_omega):
            values = {
                "CAS": "123-45-6",
                "Tc": 500.0,
                "Pc": 50.0,
                "Tb": 350.0,
                "property_sources": {
                    field_name: {
                        "source": "reported",
                        "method": "reported",
                        "quality": 0.98,
                    }
                    for field_name in ("Tc", "Pc", "Tb")
                },
            }
            if include_omega:
                values["omega"] = 0.4
                values["property_sources"]["omega"] = {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
            return values

        for include_omega, expected_route in (
            (True, "peng_robinson_delta_z"),
            (False, "ideal_delta_z"),
        ):
            with self.subTest(route=expected_route):
                adapter = PsatCanonicalizationAdapter(
                    component(include_omega),
                    hvap_at_temperature=lambda _component, _T: PropertyResolutionResult(
                        value=40.0,
                        source="source-backed",
                        method="provided_hvap_fit",
                        quality=0.97,
                    ),
                    input_methods=(_collect_deep_vacuum_inputs,),
                )
                relation = next(
                    item
                    for item in adapter.collect_inputs(
                        T_min=240.0,
                        T_critical=500.0,
                        P_critical_bar=50.0,
                        T_boiling=350.0,
                    ).relations
                    if item.method == "resolved_hvap_clapeyron_lower"
                )
                boundary_temperature = 300.0
                delta_z = (
                    _peng_robinson_delta_z_or_ideal(
                        boundary_temperature,
                        0.25,
                        500.0,
                        50.0,
                        0.4,
                    )
                    if include_omega
                    else 1.0
                )
                boundary_slope = 40000.0 / (
                    R_J_MOL_K * boundary_temperature**2 * delta_z
                )
                right = PsatEndpoint(
                    temperature=boundary_temperature,
                    ln_pressure=math.log(0.25),
                    dln_pressure_dT=boundary_slope,
                    source="hard",
                    method="hard_boundary",
                    quality=0.96,
                    segment_type=PsatSegmentType.PINNED,
                    derivative_basis=PsatDerivativeBasis.ANALYTIC,
                )
                bound = relation.bind(
                    PsatBoundaryConditions(
                        T_min=240.0,
                        T_max=boundary_temperature,
                        right=right,
                    )
                )

                self.assertEqual(
                    bound.metadata["deep_completion_route"],
                    expected_route,
                )
                self.assertAlmostEqual(bound.pressure_bar(300.0), 0.25)
                self.assertAlmostEqual(
                    bound.dln_pressure_dT(300.0),
                    boundary_slope,
                    places=10,
                )
                self.assertLess(bound.pressure_bar(240.0), 0.25)
                self.assertAlmostEqual(
                    bound.quality,
                    0.96 * 0.95,
                )
                self.assertEqual(
                    bound.metadata["quality_factor"],
                    0.95,
                )

    def test_deep_clapeyron_pr_root_loss_falls_back_to_ideal_delta_z(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_deep_vacuum_inputs,
            _peng_robinson_delta_z_or_ideal,
        )

        component = {
            "CAS": "71-43-2",
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "Tb": 350.0,
            "property_sources": {
                field_name: {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb")
            },
        }
        relation = next(
            item
            for item in PsatCanonicalizationAdapter(
                component,
                hvap_at_temperature=lambda _component, _T: PropertyResolutionResult(
                    value=40.0,
                    source="source-backed",
                    method="provided_hvap_fit",
                    quality=0.97,
                ),
                input_methods=(_collect_deep_vacuum_inputs,),
            )
            .collect_inputs(
                T_min=240.0,
                T_critical=500.0,
                P_critical_bar=50.0,
                T_boiling=350.0,
            )
            .relations
            if item.method == "resolved_hvap_clapeyron_lower"
        )
        boundary_temperature = 300.0
        ideal_slope = 40000.0 / (R_J_MOL_K * boundary_temperature**2)
        right = PsatEndpoint(
            temperature=boundary_temperature,
            ln_pressure=math.log(0.25),
            dln_pressure_dT=ideal_slope,
            source="hard",
            method="hard_boundary",
            quality=0.96,
            segment_type=PsatSegmentType.PINNED,
            derivative_basis=PsatDerivativeBasis.ANALYTIC,
        )

        with patch(
            "cubic_eos.CubicEOS._solve_monic_cubic",
            return_value=[1.0],
        ):
            self.assertEqual(
                _peng_robinson_delta_z_or_ideal(
                    boundary_temperature,
                    0.25,
                    500.0,
                    50.0,
                    0.4,
                ),
                1.0,
            )
            bound = relation.bind(
                PsatBoundaryConditions(
                    T_min=240.0,
                    T_max=boundary_temperature,
                    right=right,
                )
            )
            self.assertAlmostEqual(
                bound.dln_pressure_dT(boundary_temperature),
                ideal_slope,
                places=11,
            )
            self.assertLess(bound.pressure_bar(240.0), 0.25)

        self.assertEqual(
            bound.metadata["deep_completion_route"],
            "peng_robinson_delta_z",
        )

    def test_deep_clapeyron_declares_per_segment_slope_exception(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_deep_vacuum_inputs,
            _peng_robinson_delta_z_or_ideal,
        )

        component = {
            "CAS": "71-43-2",
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "Tb": 350.0,
            "property_sources": {
                field_name: {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb")
            },
        }
        relation = next(
            item
            for item in PsatCanonicalizationAdapter(
                component,
                hvap_at_temperature=lambda _component, _T: PropertyResolutionResult(
                    value=40.0,
                    source="source-backed",
                    method="provided_hvap_fit",
                    quality=0.97,
                ),
                input_methods=(_collect_deep_vacuum_inputs,),
            )
            .collect_inputs(
                T_min=240.0,
                T_critical=500.0,
                P_critical_bar=50.0,
                T_boiling=350.0,
            )
            .relations
            if item.method == "resolved_hvap_clapeyron_lower"
        )
        boundary_temperature = 300.0
        model_slope = 40000.0 / (
            R_J_MOL_K
            * boundary_temperature**2
            * _peng_robinson_delta_z_or_ideal(
                boundary_temperature,
                0.25,
                500.0,
                50.0,
                0.4,
            )
        )

        def endpoint(relative_mismatch):
            return PsatEndpoint(
                temperature=boundary_temperature,
                ln_pressure=math.log(0.25),
                dln_pressure_dT=model_slope / (1.0 + relative_mismatch),
                source="hard",
                method="hard_boundary",
                quality=0.96,
                segment_type=PsatSegmentType.PINNED,
                derivative_basis=PsatDerivativeBasis.ANALYTIC,
            )

        accepted = relation.bind(
            PsatBoundaryConditions(
                T_min=240.0,
                T_max=boundary_temperature,
                right=endpoint(0.099),
            )
        )
        self.assertAlmostEqual(
            accepted.metadata["incoming_slope_mismatch"],
            0.099,
            places=9,
        )
        self.assertAlmostEqual(accepted.quality, 0.96 * 0.95)
        self.assertTrue(accepted.allow_junction_slope_mismatch)

        excepted = relation.bind(
            PsatBoundaryConditions(
                T_min=240.0,
                T_max=boundary_temperature,
                right=endpoint(0.101),
            )
        )
        self.assertAlmostEqual(excepted.quality, 0.96 * 0.95)
        self.assertTrue(excepted.allow_junction_slope_mismatch)
        self.assertEqual(endpoint(0.101).quality, 0.96)

    def test_deep_clapeyron_calibrates_monoacid_dimerization(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_deep_vacuum_inputs,
        )

        component = {
            "name": "acetic acid",
            "CAS": "64-19-7",
            "Tc": 592.0,
            "Pc": 57.9,
            "omega": 0.47,
            "Tb": 391.0,
            "property_sources": {
                field_name: {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb")
            },
        }
        adapter = PsatCanonicalizationAdapter(
            component,
            hvap_at_temperature=lambda _component, _T: PropertyResolutionResult(
                value=25.0,
                source="source-backed",
                method="provided_hvap_fit",
                quality=0.97,
            ),
            input_methods=(_collect_deep_vacuum_inputs,),
        )
        relation = next(
            item
            for item in adapter.collect_inputs(
                T_min=260.0,
                T_critical=592.0,
                P_critical_bar=57.9,
                T_boiling=391.0,
            ).relations
            if item.method == "resolved_hvap_clapeyron_lower"
        )
        boundary_temperature = 330.0
        required_enthalpy = 40000.0
        boundary_slope = required_enthalpy / (R_J_MOL_K * boundary_temperature**2)
        right = PsatEndpoint(
            temperature=boundary_temperature,
            ln_pressure=math.log(0.25),
            dln_pressure_dT=boundary_slope,
            source="hard",
            method="hard_boundary",
            quality=0.96,
            segment_type=PsatSegmentType.PINNED,
            derivative_basis=PsatDerivativeBasis.ANALYTIC,
        )
        bound = relation.bind(
            PsatBoundaryConditions(
                T_min=260.0,
                T_max=boundary_temperature,
                right=right,
            )
        )

        self.assertEqual(
            bound.metadata["deep_completion_route"],
            "calibrated_monoacid_dimer",
        )
        self.assertAlmostEqual(
            bound.metadata["dimer_extent_at_boundary"],
            0.375,
        )
        self.assertAlmostEqual(
            bound.dln_pressure_dT(boundary_temperature),
            boundary_slope,
            places=10,
        )
        self.assertLess(bound.pressure_bar(260.0), 0.25)
        self.assertAlmostEqual(
            bound.quality,
            0.96 * 0.90,
        )
        self.assertEqual(
            bound.metadata["quality_factor"],
            0.90,
        )

    def test_deep_clapeyron_slope_exception_keeps_clapeyron_selected(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_deep_vacuum_inputs,
        )

        component = {
            "CAS": "71-43-2",
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "Tb": 350.0,
            "property_sources": {
                field_name: {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb")
            },
        }
        inputs = PsatCanonicalizationAdapter(
            component,
            hvap_at_temperature=lambda _component, _T: PropertyResolutionResult(
                value=40.0,
                source="source-backed",
                method="provided_hvap_fit",
                quality=0.97,
            ),
            input_methods=(_collect_deep_vacuum_inputs,),
        ).collect_inputs(
            T_min=240.0,
            T_critical=500.0,
            P_critical_bar=50.0,
            T_boiling=350.0,
        )
        hard = PsatSegment(
            source="hard",
            method="hard_boundary",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=300.0,
            T_max=350.0,
            ln_pressure_function=lambda T: math.log(0.25) + 0.01 * (T - 300.0),
            derivative_function=lambda _T: 0.01,
            quality=0.96,
            derivative_basis=PsatDerivativeBasis.ANALYTIC,
        )
        assembler = PsatSegmentAssembler(240.0, 350.0)
        assembler.add(hard)

        completion = PsatCompletionCoordinator(
            assembler,
            inputs.relations,
        ).complete()

        self.assertEqual(
            [item.method for item in completion.generated_segments],
            ["resolved_hvap_clapeyron_lower"],
        )
        self.assertEqual(
            completion.generated_segments[0].metadata["deep_completion_route"],
            "peng_robinson_delta_z",
        )
        self.assertTrue(
            completion.generated_segments[0].metadata[
                "junction_slope_exception_penalty_applied"
            ]
        )
        self.assertAlmostEqual(
            completion.generated_segments[0].quality,
            0.96 * 0.95 * 0.90,
        )
        self.assertEqual(
            completion.generated_segments[0].metadata[
                "junction_slope_exception_original_quality"
            ],
            0.96 * 0.95,
        )
        self.assertEqual(
            completion.generated_segments[0].metadata[
                "junction_slope_exception_quality_factor"
            ],
            0.90,
        )
        self.assertEqual(hard.quality, 0.96)
        self.assertEqual(len(completion.junction_warnings), 1)
        self.assertIn(
            "accepted by slope-mismatch exception",
            completion.junction_warnings[0],
        )

    def test_perry_hard_aw_pr_clapeyron_completes_to_deep_vacuum(self):
        from scipy.optimize import brentq

        from perry_properties import get_perry_property_library
        from property_resolution.vapor_pressure_adapter import (
            _collect_anchored_ambrose_walton_lower_inputs,
            _collect_deep_vacuum_inputs,
            _collect_perry_2_8_inputs,
        )

        library = get_perry_property_library()
        library._load()
        entry = library.get("ethylbenzene", expand_identity=False)
        critical = entry["critical_constants"]
        Tc = float(critical["Tc_K"])
        Pc_bar = float(critical["Pc_MPa"]) * 10.0
        Tb = float(library.normal_boiling_point_K("ethylbenzene").value)
        component = {
            "name": "ethylbenzene",
            "CAS": entry["cas"],
            "Tc": Tc,
            "Pc": Pc_bar,
            "omega": float(critical["omega"]),
            "Tb": Tb,
            "property_sources": {
                field_name: {
                    "source": "Perry 9th",
                    "method": "perry",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega", "Tb")
            },
        }
        source_inputs = PsatCanonicalizationAdapter(
            component,
            input_methods=(_collect_perry_2_8_inputs,),
        ).collect_inputs(
            T_critical=Tc,
            P_critical_bar=Pc_bar,
            T_boiling=Tb,
        )
        source = source_inputs.segments[0]

        def crossing(pressure):
            return brentq(
                lambda T: source.raw_ln_pressure(T) - math.log(pressure),
                source.T_min,
                min(source.T_max, Tc),
            )

        T_min = crossing(0.001)
        hard_temperature = crossing(0.5)
        inputs = PsatCanonicalizationAdapter(
            component,
            input_methods=(
                _collect_perry_2_8_inputs,
                _collect_anchored_ambrose_walton_lower_inputs,
                _collect_deep_vacuum_inputs,
            ),
        ).collect_inputs(
            T_min=T_min,
            T_critical=Tc,
            P_critical_bar=Pc_bar,
            T_boiling=Tb,
        )
        hard = inputs.segments[0].clipped(
            hard_temperature,
            inputs.segments[0].T_max,
        )
        assembler = PsatSegmentAssembler(T_min, hard.T_max)
        assembler.add(hard)
        completion = PsatCompletionCoordinator(
            assembler,
            inputs.relations,
            PsatAnchorRegistry.from_tb_tc(
                T_critical=Tc,
                P_critical_bar=Pc_bar,
                critical_quality=0.98,
                T_boiling=Tb,
                boiling_quality=0.98,
            ),
        ).complete()

        self.assertEqual(
            [segment.method for segment in completion.generated_segments],
            [
                "anchored_ambrose_walton_lower",
                "resolved_hvap_clapeyron_lower",
            ],
        )
        self.assertEqual(
            completion.generated_segments[-1].metadata["deep_completion_route"],
            "peng_robinson_delta_z",
        )
        predicted = math.exp(completion.assembly.ln_pressure(T_min))
        self.assertLess(abs(predicted / 0.001 - 1.0), 0.02)

    def test_anchored_aw_derives_omega_when_only_criticals_are_admissible(self):
        component = {
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "property_sources": {
                "Tc": {"source": "reported", "method": "reported", "quality": 0.98},
                "Pc": {"source": "reported", "method": "reported", "quality": 0.98},
                "omega": {
                    "source": "calculated",
                    "method": "lee_kesler",
                    "quality": 0.96,
                },
            },
        }
        inputs = PsatCanonicalizationAdapter(component).collect_inputs(
            T_critical=500.0,
            P_critical_bar=50.0,
        )
        relation = next(
            item
            for item in inputs.relations
            if item.method == "anchored_ambrose_walton_upper"
        )
        self.assertIsNone(relation.metadata["omega"])
        self.assertEqual(
            relation.metadata["omega_basis"],
            "resolved_from_selected_hard_curve",
        )
        self.assertEqual(
            relation.assembly_anchor_requirements[0].temperature,
            350.0,
        )

        component["property_sources"]["Tc"]["quality"] = 0.92
        inputs = PsatCanonicalizationAdapter(component).collect_inputs(
            T_critical=500.0,
            P_critical_bar=50.0,
        )
        self.assertTrue(
            any(
                item.method == "anchored_ambrose_walton_upper"
                for item in inputs.relations
            )
        )
        self.assertTrue(any("Tc admitted" in warning for warning in inputs.warnings))

    def test_anchored_aw_uses_selected_hard_psat_at_tr_07_for_omega(self):
        from perry_properties import get_perry_property_library

        library = get_perry_property_library()
        critical = library.critical_properties("ethanol")
        Tc = critical["Tc"].value
        Pc_bar = critical["Pc"].value
        component = {
            "name": "ethanol",
            "CAS": "64-17-5",
            "Tc": Tc,
            "Pc": Pc_bar,
            "omega": critical["omega"].value,
            "property_sources": {
                "Tc": {
                    "source": "Perry 9th",
                    "method": "perry_critical",
                    "quality": 0.98,
                },
                "Pc": {
                    "source": "Perry 9th",
                    "method": "perry_critical",
                    "quality": 0.98,
                },
                "omega": {
                    "source": "calculated",
                    "method": "lee_kesler",
                    "quality": 0.90,
                },
            },
        }
        inputs = PsatCanonicalizationAdapter(component).collect_inputs(
            T_critical=Tc,
            P_critical_bar=Pc_bar,
        )
        source = next(
            item
            for item in inputs.segments
            if item.method == "perry_2_8_vapor_pressure"
        )
        relation = next(
            item
            for item in inputs.relations
            if item.method == "anchored_ambrose_walton_upper"
        )
        hard = source.clipped(source.T_min, 0.8 * Tc)
        assembler = PsatSegmentAssembler(source.T_min, Tc)
        assembler.add(hard)
        completion = PsatCompletionCoordinator(
            assembler,
            (relation,),
            PsatAnchorRegistry.from_tb_tc(
                T_critical=Tc,
                P_critical_bar=Pc_bar,
                critical_quality=0.98,
            ),
        ).complete()

        generated = completion.generated_segments[0]
        self.assertEqual(
            generated.method,
            "anchored_ambrose_walton_upper",
        )
        self.assertAlmostEqual(
            generated.ln_pressure(0.8 * Tc),
            source.ln_pressure(0.8 * Tc),
            places=12,
        )
        self.assertAlmostEqual(
            generated.dln_pressure_dT(0.8 * Tc),
            source.dln_pressure_dT(0.8 * Tc),
            places=12,
        )
        self.assertAlmostEqual(generated.pressure_bar(Tc), Pc_bar)
        self.assertEqual(
            generated.metadata["bound_omega_basis"],
            "physical_definition_from_hard_segment_at_Tr_0.7",
        )
        self.assertEqual(generated.metadata["bound_omega_quality"], 0.98)
        self.assertAlmostEqual(generated.quality, 0.98 * 0.91)

    def test_anchored_aw_matches_hard_boundary_and_critical_point(self):
        component = {
            "Tc": 500.0,
            "Pc": 50.0,
            "omega": 0.4,
            "property_sources": {
                field_name: {
                    "source": "reported",
                    "method": "reported",
                    "quality": 0.98,
                }
                for field_name in ("Tc", "Pc", "omega")
            },
        }
        relation = next(
            item
            for item in PsatCanonicalizationAdapter(component)
            .collect_inputs(
                T_critical=500.0,
                P_critical_bar=50.0,
            )
            .relations
            if item.method == "anchored_ambrose_walton_upper"
        )
        hard_segment = PsatSegment(
            source="measured",
            method="hard_curve",
            segment_type=PsatSegmentType.PINNED,
            priority=800,
            T_min=350.0,
            T_max=400.0,
            ln_pressure_function=lambda T: -4.0 + 0.02 * (T - 350.0),
            derivative_function=lambda _T: 0.02,
            quality=0.97,
        )
        left = PsatEndpoint.from_segment(hard_segment, 400.0)
        critical = PsatEndpoint.fixed_pressure_point(
            500.0,
            50.0,
            source="fixed_anchor",
            method="critical_point",
            quality=0.98,
            context={"anchor_name": "Tc"},
        )
        completion = relation.bind(
            PsatBoundaryConditions(
                T_min=400.0,
                T_max=500.0,
                left=left,
                anchors=(critical,),
            )
        )

        self.assertAlmostEqual(
            completion.ln_pressure(400.0),
            left.ln_pressure,
            places=12,
        )
        self.assertAlmostEqual(
            completion.dln_pressure_dT(400.0),
            left.dln_pressure_dT,
            places=12,
        )
        self.assertAlmostEqual(completion.pressure_bar(500.0), 50.0)
        self.assertEqual(completion.quality, 0.98 * 0.91)
        step = 1.0e-3
        numerical_derivative = (
            completion.ln_pressure(450.0 + step) - completion.ln_pressure(450.0 - step)
        ) / (2.0 * step)
        self.assertAlmostEqual(
            completion.dln_pressure_dT(450.0),
            numerical_derivative,
            places=9,
        )
        with self.assertRaisesRegex(
            PsatCanonicalizationError,
            "terminal gap ending at Tc",
        ):
            relation.bind(
                PsatBoundaryConditions(
                    T_min=400.0,
                    T_max=450.0,
                    left=left,
                    anchors=(critical,),
                )
            )

    def test_anchored_aw_reproduces_full_perry_upper_benchmark(self):
        from perry_properties import get_perry_property_library
        from property_resolution.vapor_pressure_adapter import (
            _collect_anchored_ambrose_walton_inputs,
            _collect_perry_2_8_inputs,
        )

        library = get_perry_property_library()
        library._load()
        curve_mards = []
        curve_maxima = []
        acid_mards = []
        acid_maxima = []
        for entry in library.chemicals.values():
            if not entry.get("vapor_pressure") or not entry.get("critical_constants"):
                continue
            critical = entry["critical_constants"]
            Tc = float(critical["Tc_K"])
            Pc_bar = float(critical["Pc_MPa"]) * 10.0
            component = {
                "name": entry.get("name"),
                "CAS": entry.get("cas"),
                "Tc": Tc,
                "Pc": Pc_bar,
                "omega": float(critical["omega"]),
                "property_sources": {
                    field_name: {
                        "source": "Perry 9th",
                        "method": "perry_critical",
                        "quality": 0.98,
                    }
                    for field_name in ("Tc", "Pc", "omega")
                },
            }
            inputs = PsatCanonicalizationAdapter(
                component,
                input_methods=(
                    _collect_perry_2_8_inputs,
                    _collect_anchored_ambrose_walton_inputs,
                ),
            ).collect_inputs(
                T_critical=Tc,
                P_critical_bar=Pc_bar,
            )
            source_segment = next(
                item
                for item in inputs.segments
                if item.method == "perry_2_8_vapor_pressure"
            )
            relation = next(
                item
                for item in inputs.relations
                if item.method == "anchored_ambrose_walton_upper"
            )
            coefficients = library.vapor_pressure_coefficients(
                entry["vapor_pressure"][0]
            )
            T_handoff = 0.8 * Tc
            left = PsatEndpoint.from_segment(source_segment, T_handoff)
            critical_endpoint = PsatEndpoint.fixed_pressure_point(
                Tc,
                Pc_bar,
                source="fixed_anchor",
                method="critical_point",
                quality=0.98,
                context={"anchor_name": "Tc"},
            )
            completion = relation.bind(
                PsatBoundaryConditions(
                    T_min=T_handoff,
                    T_max=Tc,
                    left=left,
                    anchors=(critical_endpoint,),
                )
            )
            errors = []
            previous_pressure = None
            for index in range(401):
                temperature = T_handoff + (Tc - T_handoff) * index / 400.0
                C1, C2, C3, C4, C5 = coefficients
                reference = (
                    math.exp(
                        C1
                        + C2 / temperature
                        + C3 * math.log(temperature)
                        + C4 * temperature**C5
                    )
                    / 100000.0
                )
                predicted = completion.pressure_bar(temperature)
                errors.append(abs(predicted / reference - 1.0))
                if previous_pressure is not None:
                    self.assertGreater(predicted, previous_pressure)
                previous_pressure = predicted
            curve_mard = sum(errors) / len(errors)
            curve_maximum = max(errors)
            curve_mards.append(curve_mard)
            curve_maxima.append(curve_maximum)
            if "acid" in str(entry.get("name") or "").lower():
                acid_mards.append(curve_mard)
                acid_maxima.append(curve_maximum)

        ordered_mards = sorted(curve_mards)
        p95_index = round(0.95 * (len(ordered_mards) - 1))
        self.assertEqual(len(curve_mards), 345)
        self.assertLess(median(curve_mards), 0.0040)
        self.assertLess(ordered_mards[p95_index], 0.0195)
        self.assertLess(max(curve_mards), 0.069)
        self.assertLess(max(curve_maxima), 0.125)

        ordered_acid_mards = sorted(acid_mards)
        acid_p95_index = round(0.95 * (len(ordered_acid_mards) - 1))
        self.assertEqual(len(acid_mards), 22)
        self.assertLess(median(acid_mards), 0.010)
        self.assertLess(ordered_acid_mards[acid_p95_index], 0.023)
        self.assertLess(max(acid_mards), 0.026)
        self.assertLess(max(acid_maxima), 0.045)

    def test_endpoint_inferred_omega_preserves_tb_upper_accuracy(self):
        from perry_properties import get_perry_property_library
        from property_resolution.vapor_pressure_adapter import (
            _collect_anchored_ambrose_walton_inputs,
            _collect_perry_2_8_inputs,
        )

        library = get_perry_property_library()
        library._load()
        curve_mards = []
        for entry in library.chemicals.values():
            if not entry.get("vapor_pressure") or not entry.get("critical_constants"):
                continue
            critical = entry["critical_constants"]
            Tc = float(critical["Tc_K"])
            Pc_bar = float(critical["Pc_MPa"]) * 10.0
            Tb_result = library.normal_boiling_point_K(entry["cas"])
            if Tb_result is None:
                continue
            Tb = float(Tb_result.value)
            if not float(entry["vapor_pressure"][0]["T_min_K"]) <= Tb < 0.7 * Tc:
                continue
            component = {
                "name": entry.get("name"),
                "CAS": entry.get("cas"),
                "Tc": Tc,
                "Pc": Pc_bar,
                "omega": float(critical["omega"]),
                "property_sources": {
                    "Tc": {
                        "source": "Perry 9th",
                        "method": "perry_critical",
                        "quality": 0.98,
                    },
                    "Pc": {
                        "source": "Perry 9th",
                        "method": "perry_critical",
                        "quality": 0.98,
                    },
                    "omega": {
                        "source": "calculated",
                        "method": "lee_kesler",
                        "quality": 0.80,
                    },
                },
            }
            inputs = PsatCanonicalizationAdapter(
                component,
                input_methods=(
                    _collect_perry_2_8_inputs,
                    _collect_anchored_ambrose_walton_inputs,
                ),
            ).collect_inputs(
                T_min=Tb,
                T_critical=Tc,
                P_critical_bar=Pc_bar,
            )
            source = inputs.segments[0]
            relation = inputs.relations[0]
            left = PsatEndpoint.from_segment(source, Tb)
            critical_endpoint = PsatEndpoint.fixed_pressure_point(
                Tc,
                Pc_bar,
                source="fixed_anchor",
                method="critical_point",
                quality=0.98,
                context={"anchor_name": "Tc"},
            )
            completion = relation.bind(
                PsatBoundaryConditions(
                    T_min=Tb,
                    T_max=Tc,
                    left=left,
                    anchors=(critical_endpoint,),
                )
            )
            penalty = 1.0 - (0.7 - Tb / Tc) / 2.0
            self.assertEqual(
                completion.metadata["bound_omega_basis"],
                "aw_inversion_from_hard_endpoint",
            )
            self.assertAlmostEqual(
                completion.metadata["omega_quality_penalty"],
                penalty,
            )
            self.assertAlmostEqual(
                completion.metadata["bound_omega_quality"],
                left.quality * penalty,
            )
            self.assertAlmostEqual(
                completion.quality,
                left.quality * penalty * 0.91,
            )
            coefficients = library.vapor_pressure_coefficients(
                entry["vapor_pressure"][0]
            )
            C1, C2, C3, C4, C5 = coefficients
            errors = []
            for index in range(401):
                temperature = Tb + (Tc - Tb) * index / 400.0
                reference = (
                    math.exp(
                        C1
                        + C2 / temperature
                        + C3 * math.log(temperature)
                        + C4 * temperature**C5
                    )
                    / 100000.0
                )
                errors.append(
                    abs(completion.pressure_bar(temperature) / reference - 1.0)
                )
            curve_mards.append(sum(errors) / len(errors))

        ordered = sorted(curve_mards)
        p95_index = round(0.95 * (len(ordered) - 1))
        self.assertEqual(len(curve_mards), 276)
        self.assertLess(median(curve_mards), 0.0060)
        self.assertLess(ordered[p95_index], 0.027)
        self.assertLess(max(curve_mards), 0.065)

    def test_ethanol_rejects_bad_nist_then_uses_anchored_aw_upper(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_anchored_ambrose_walton_inputs,
            _collect_cached_nist_antoine_inputs,
            _collect_textbook_antoine_inputs,
        )

        component = {
            "name": "ethanol",
            "CAS": "64-17-5",
            "Tb": 351.57040446751455,
            "Tc": 514.0,
            "Pc": 61.37,
            "omega": 0.643558,
            "property_sources": {
                "Tb": {
                    "source": "reported",
                    "method": "normal_boiling_point",
                    "quality": 0.97,
                },
                **{
                    field_name: {
                        "source": "reported",
                        "method": "critical_property",
                        "quality": 0.98,
                    }
                    for field_name in ("Tc", "Pc", "omega")
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            SQLiteJSONCache(
                cache_dir / "property_cache.sqlite",
                "property_resolver",
            ).set(
                "antoine_64-17-5_413.00K",
                {
                    "A": 4.92531,
                    "B": 1432.526,
                    "C": 211.33099999999996,
                    "T_min": 364.8,
                    "T_max": 513.91,
                    "source": "NIST WebBook",
                    "P_units": "bar",
                },
            )
            with patch(
                "property_resolution.vapor_pressure_adapter.ANTOINE_CACHE_DIR",
                cache_dir,
            ):
                inputs = PsatCanonicalizationAdapter(
                    component,
                    input_methods=(
                        _collect_textbook_antoine_inputs,
                        _collect_cached_nist_antoine_inputs,
                        _collect_anchored_ambrose_walton_inputs,
                    ),
                ).collect_inputs(
                    T_min=276.15,
                    T_critical=514.0,
                    P_critical_bar=61.37,
                    T_boiling=component["Tb"],
                )

        result = PsatCanonicalizer(
            T_min=276.15,
            T_critical=514.0,
            P_critical_bar=61.37,
            critical_quality=0.98,
            T_boiling=component["Tb"],
            boiling_quality=0.97,
        ).canonicalize(**inputs.canonicalizer_arguments())
        methods = [item.segment.method for item in result.assembly.slices]
        self.assertEqual(
            methods,
            ["textbook_antoine", "anchored_ambrose_walton_upper"],
        )
        self.assertTrue(
            any(
                decision.rejected_method == "cached_nist_antoine"
                for decision in result.completion.handoffs.decisions
            )
        )
        temperature = 398.15
        assembled_pressure = math.exp(result.assembly.ln_pressure(temperature))
        self.assertAlmostEqual(assembled_pressure, 5.013502252524025)
        canonical_pressure = result.curve.pressure_bar(temperature)
        self.assertGreater(canonical_pressure, 4.9)
        self.assertLess(
            abs(canonical_pressure / assembled_pressure - 1.0),
            0.006,
        )

    def test_smith_antoine_uses_the_high_quality_550_layer(self):
        inputs = PsatCanonicalizationAdapter(
            {
                "name": "1-Butanol",
            }
        ).collect_inputs()
        segment = next(
            item for item in inputs.segments if item.method == "textbook_antoine"
        )

        self.assertEqual(PsatPriority.HIGH_QUALITY_ANTOINE, 550)
        self.assertEqual(segment.priority, PsatPriority.HIGH_QUALITY_ANTOINE)
        self.assertEqual(segment.quality, 0.95)
        self.assertEqual((segment.T_min, segment.T_max), (310.15, 411.15))
        self.assertEqual(
            segment.handoff_requirement,
            PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP_IF_AVAILABLE,
        )
        self.assertEqual(
            segment.metadata["tb_validation_status"],
            "validation_unavailable",
        )
        self.assertLess(
            abs(segment.pressure_bar(390.75) / 1.01325 - 1.0),
            0.02,
        )

        validated_segment = next(
            item
            for item in PsatCanonicalizationAdapter(
                {
                    "name": "1-Butanol",
                }
            )
            .collect_inputs(T_boiling=390.75)
            .segments
            if item.method == "textbook_antoine"
        )
        self.assertEqual(validated_segment.quality, 0.97)
        self.assertEqual(
            validated_segment.metadata["tb_validation_status"],
            "hard_tb_validated",
        )
        self.assertEqual(
            validated_segment.handoff_requirement,
            PsatHandoffRequirement.NONE,
        )

        outside_tb_segment = next(
            item
            for item in PsatCanonicalizationAdapter(
                {
                    "name": "1-Butanol",
                }
            )
            .collect_inputs(T_boiling=500.0)
            .segments
            if item.method == "textbook_antoine"
        )
        self.assertEqual(
            outside_tb_segment.handoff_requirement,
            PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP_IF_AVAILABLE,
        )

    def test_large_smith_perry_endpoint_mismatch_rejects_lower_priority_source(self):
        inputs = PsatCanonicalizationAdapter(
            {
                "name": "iso-Butanol",
                "symbol": "C4H10O",
                "formula": "C4H10O",
                "CAS": "78-83-1",
                "Tb": 380.95,
            }
        ).collect_inputs(T_boiling=380.95)
        result = PsatHandoffCoordinator(264.15, 401.15).coordinate(inputs.segments)

        self.assertEqual(
            [segment.method for segment in result.rejected_segments],
            ["perry_2_10_vapor_pressure"],
        )
        self.assertEqual(
            [item.segment.method for item in result.assembly.slices],
            ["textbook_antoine"],
        )
        self.assertIn(
            "direct handoff relocation exceeds compensation budget",
            result.decisions[0].reason,
        )

    def test_hydrated_smith_antoine_is_not_promoted_to_curated_priority(self):
        component = {
            "name": "1-Butanol",
            "source": "local",
            "Tb": 390.75,
            "antoine_A": 4.650959413659159,
            "antoine_B": 1395.140622500463,
            "antoine_C": 182.739,
            "antoine_Tmin": 310.15,
            "antoine_Tmax": 411.15,
            "antoine_source": "Smith8 Appendix B Table B.2",
            "property_sources": {
                "Antoine": {
                    "source": "local",
                    "method": "antoine",
                    "quality": 0.95,
                },
            },
        }

        inputs = PsatCanonicalizationAdapter(component).collect_inputs()
        antoine_segments = [
            segment for segment in inputs.segments if "antoine" in segment.method
        ]

        self.assertNotIn(
            "local_antoine",
            [segment.method for segment in antoine_segments],
        )
        textbook_segment = next(
            segment
            for segment in antoine_segments
            if segment.method == "textbook_antoine"
        )
        self.assertEqual(
            textbook_segment.priority,
            PsatPriority.HIGH_QUALITY_ANTOINE,
        )

    def test_antoine_table_rows_use_strict_priority_400_validation(self):
        heptanal = PsatCanonicalizationAdapter(
            {
                "name": "heptanal",
                "CAS": "111-71-7",
                "Tb": 426.0,
            }
        ).collect_inputs()
        heptanal_segment = next(
            segment
            for segment in heptanal.segments
            if segment.method == "antoine_table"
        )
        self.assertEqual(heptanal_segment.priority, PsatPriority.OTHER_ANTOINE)
        self.assertEqual(heptanal_segment.quality, 0.95)
        self.assertEqual(
            heptanal_segment.handoff_requirement,
            PsatHandoffRequirement.NONE,
        )

        nonanal = PsatCanonicalizationAdapter(
            {
                "name": "nonanal",
                "CAS": "124-19-6",
                "Tb": 465.3722222222222,
            }
        ).collect_inputs()
        nonanal_segment = next(
            segment for segment in nonanal.segments if segment.method == "antoine_table"
        )
        self.assertEqual(nonanal_segment.quality, 0.90)
        self.assertEqual(
            nonanal_segment.handoff_requirement,
            PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP_IF_AVAILABLE,
        )
        self.assertEqual(
            nonanal_segment.metadata["tb_validation_status"],
            "hard_tb_outside_range",
        )

    def test_soft_tb_cannot_validate_or_reject_hard_antoine_data(self):
        from property_resolution.vapor_pressure_adapter import (
            _collect_antoine_table_inputs,
        )

        soft_component = {
            "symbol": "C6F6",
            "name": "hexafluorobenzene",
            "CAS": "392-56-3",
            "Tb": 348.23794864774715,
            "property_sources": {
                "Tb": {
                    "source": "estimated",
                    "method": "nannoolal_tb",
                    "quality": 0.80,
                },
            },
        }
        soft = PsatCanonicalizationAdapter(
            soft_component,
            input_methods=(_collect_antoine_table_inputs,),
        ).collect_inputs(T_boiling=soft_component["Tb"])
        unvalidated = next(
            item for item in soft.segments if item.method == "antoine_table"
        )

        self.assertEqual(unvalidated.quality, 0.90)
        self.assertEqual(
            unvalidated.metadata["tb_validation_status"],
            "validation_unavailable",
        )
        self.assertEqual(soft.warnings, ())

        hard_component = {
            **soft_component,
            "Tb": 353.4,
            "property_sources": {
                "Tb": {
                    "source": "online",
                    "method": "nist_phase_change",
                    "quality": 0.93,
                },
            },
        }
        hard = PsatCanonicalizationAdapter(
            hard_component,
            input_methods=(_collect_antoine_table_inputs,),
        ).collect_inputs(T_boiling=hard_component["Tb"])
        validated = next(
            item for item in hard.segments if item.method == "antoine_table"
        )

        self.assertEqual(validated.quality, 0.95)
        self.assertEqual(
            validated.metadata["tb_validation_status"],
            "hard_tb_validated",
        )

    def test_unvalidated_antoine_quality_tracks_strictly_higher_overlap(self):
        def segment(method, priority, quality, requirement, **metadata):
            return PsatSegment(
                source=method,
                method=method,
                segment_type=PsatSegmentType.PINNED,
                priority=priority,
                T_min=300.0,
                T_max=400.0,
                ln_pressure_function=lambda T: 0.01 * T - 4.0,
                derivative_function=lambda _T: 0.01,
                quality=quality,
                handoff_requirement=requirement,
                metadata=metadata,
            )

        unvalidated = segment(
            "unvalidated_antoine",
            int(PsatPriority.OTHER_ANTOINE),
            0.90,
            PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP_IF_AVAILABLE,
            tb_validation_status="validation_unavailable",
            quality_basis="standalone_unvalidated",
            higher_preference_overlap_quality=0.93,
        )
        standalone = PsatHandoffCoordinator(300.0, 400.0).coordinate((unvalidated,))
        standalone_segment = next(
            item for item in standalone.segments if item.method == "unvalidated_antoine"
        )
        self.assertEqual(standalone_segment.quality, 0.90)
        self.assertNotIn("overlap_validation_source", standalone_segment.metadata)

        same_priority = segment(
            "same_priority_antoine",
            int(PsatPriority.OTHER_ANTOINE),
            0.92,
            PsatHandoffRequirement.NONE,
        )
        same_layer = PsatHandoffCoordinator(300.0, 420.0).coordinate(
            (
                replace(unvalidated, T_max=420.0),
                replace(same_priority, T_max=380.0),
            )
        )
        same_layer_segment = next(
            item for item in same_layer.segments if item.method == "unvalidated_antoine"
        )
        self.assertEqual(same_layer_segment.quality, 0.90)
        self.assertNotIn("overlap_validation_source", same_layer_segment.metadata)

        higher_priority = segment(
            "perry_reference",
            int(PsatPriority.PERRY_2_10),
            0.98,
            PsatHandoffRequirement.NONE,
        )
        # The higher-priority source owns the overlap, so expose the promotion
        # on a tail where the Antoine row remains selected.
        tailed_antoine = replace(
            unvalidated,
            T_min=300.0,
            T_max=420.0,
        )
        tailed_reference = replace(
            higher_priority,
            T_min=300.0,
            T_max=380.0,
        )
        corroborated = PsatHandoffCoordinator(300.0, 420.0).coordinate(
            (tailed_antoine, tailed_reference)
        )
        promoted = next(
            item
            for item in corroborated.segments
            if item.method == "unvalidated_antoine"
        )
        self.assertEqual(promoted.quality, 0.93)
        self.assertEqual(
            promoted.metadata["quality_basis"],
            "higher_preference_overlap",
        )
        self.assertEqual(
            promoted.metadata["overlap_validation_method"],
            "perry_reference",
        )

    def test_cached_nist_antoine_uses_priority_400_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            Tb = 390.0
            B = 1000.0
            C = 200.0
            A = math.log10(1.01325) + B / (C + Tb - 273.15)
            SQLiteJSONCache(
                cache_dir / "property_cache.sqlite",
                "property_resolver",
            ).set(
                "antoine_999-99-9",
                {
                    "A": A,
                    "B": B,
                    "C": C,
                    "T_min": 300.0,
                    "T_max": 410.0,
                    "source": "NIST WebBook",
                    "P_units": "bar",
                },
            )
            with patch(
                "property_resolution.vapor_pressure_adapter.ANTOINE_CACHE_DIR",
                cache_dir,
            ):
                inputs = PsatCanonicalizationAdapter(
                    {
                        "name": "cache-only compound",
                        "CAS": "999-99-9",
                        "Tb": Tb,
                    }
                ).collect_inputs()

        segment = next(
            item for item in inputs.segments if item.method == "cached_nist_antoine"
        )
        self.assertEqual(segment.priority, PsatPriority.OTHER_ANTOINE)
        self.assertEqual(segment.quality, 0.95)
        self.assertEqual(
            segment.handoff_requirement,
            PsatHandoffRequirement.NONE,
        )
        self.assertTrue(segment.metadata["cache_only"])

    def test_curated_chemicals_json_antoine_rows_are_pinned_segments(self):
        database = ChemicalDatabase(enable_online=False)
        for key, expected_range in (
            ("3-methylpentane", (244.15, 362.15)),
            ("H2SO4", (298.15, 610.0)),
        ):
            with self.subTest(key=key):
                component = database.chemicals[key]
                inputs = PsatCanonicalizationAdapter(component).collect_inputs()

                segment = next(
                    item for item in inputs.segments if item.method == "local_antoine"
                )
                self.assertEqual(segment.method, "local_antoine")
                self.assertEqual(segment.priority, PsatPriority.LOCAL_ANTOINE)
                self.assertEqual(segment.quality, 0.98)
                self.assertEqual(
                    segment.handoff_requirement,
                    PsatHandoffRequirement.NONE,
                )
                self.assertEqual((segment.T_min, segment.T_max), expected_range)
                self.assertGreater(segment.dln_pressure_dT(component.Tb), 0.0)
                self.assertLess(
                    abs(segment.pressure_bar(component.Tb) - 1.01325) / 1.01325,
                    0.02,
                )

    def test_curated_antoine_follows_local_psat_correlation(self):
        component = {
            "symbol": "X",
            "source": "local",
            "Tc": 500.0,
            "antoine_A": 4.0,
            "antoine_B": 1000.0,
            "antoine_C": 200.0,
            "antoine_Tmin": 250.0,
            "antoine_Tmax": 400.0,
            "property_correlations": {
                "Psat": {
                    "equation": "exp_poly_x",
                    "Tmin_K": 225.0,
                    "Tmax_K": 450.0,
                    "coefficients": {"A": -2.0},
                    "quality": 0.97,
                },
            },
        }

        inputs = PsatCanonicalizationAdapter(component).collect_inputs()

        self.assertEqual(
            [segment.priority for segment in inputs.segments],
            [PsatPriority.LOCAL_CORRELATION, PsatPriority.LOCAL_ANTOINE],
        )

    def test_online_antoine_is_not_promoted_but_pfd_antoine_stays_top_priority(self):
        base = {
            "symbol": "X",
            "antoine_A": 4.0,
            "antoine_B": 1000.0,
            "antoine_C": 200.0,
            "antoine_Tmin": 250.0,
            "antoine_Tmax": 400.0,
        }
        online = {
            **base,
            "property_sources": {
                "Antoine": {
                    "source": "online",
                    "method": "nist_antoine",
                    "quality": 0.90,
                },
            },
        }
        pfd = {
            **base,
            "property_sources": {
                field_name: {
                    "source": "provided",
                    "method": "pfd_component_override",
                    "quality": 1.0,
                }
                for field_name in (
                    "antoine_A",
                    "antoine_B",
                    "antoine_C",
                    "antoine_Tmin",
                    "antoine_Tmax",
                )
            },
        }

        self.assertEqual(
            PsatCanonicalizationAdapter(online).collect_inputs(),
            PsatCanonicalizationInputs(),
        )
        pfd_inputs = PsatCanonicalizationAdapter(pfd).collect_inputs(
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=40.0,
        )
        self.assertEqual(len(pfd_inputs.segments), 1)
        segment = pfd_inputs.segments[0]
        self.assertEqual(segment.method, "pfd_antoine")
        self.assertEqual(segment.priority, PsatPriority.PFD_OVERRIDE)
        self.assertEqual(segment.quality, 1.0)
        self.assertEqual((segment.T_min, segment.T_max), (250.0, 400.0))

    def test_online_antoine_on_a_local_component_is_not_promoted(self):
        component = {
            "symbol": "X",
            "source": "local",
            "antoine_A": 4.0,
            "antoine_B": 1000.0,
            "antoine_C": 200.0,
            "antoine_Tmin": 250.0,
            "antoine_Tmax": 400.0,
            "antoine_source": "NIST Chemistry WebBook",
            "property_sources": {
                "Antoine": {
                    "source": "online",
                    "method": "nist_antoine",
                    "quality": 0.90,
                },
            },
        }

        self.assertEqual(
            PsatCanonicalizationAdapter(component).collect_inputs(),
            PsatCanonicalizationInputs(),
        )

    def test_curated_antoine_is_exempt_when_tb_is_outside_range(self):
        component = {
            "symbol": "X",
            "source": "local",
            "Tb": 450.0,
            "antoine_A": 4.0,
            "antoine_B": 1000.0,
            "antoine_C": 200.0,
            "antoine_Tmin": 250.0,
            "antoine_Tmax": 400.0,
        }

        segment = PsatCanonicalizationAdapter(component).collect_inputs().segments[0]

        self.assertEqual(
            segment.handoff_requirement,
            PsatHandoffRequirement.NONE,
        )
        self.assertEqual(
            segment.metadata["tb_validation_status"],
            "exempt_priority",
        )
        self.assertFalse(segment.metadata["tb_validation_required"])

    def test_programmatic_incomplete_pfd_antoine_violates_parser_invariant(self):
        component = {
            "symbol": "X",
            "antoine_A": 4.0,
            "antoine_B": 1000.0,
            "antoine_C": 200.0,
            "antoine_Tmin": 250.0,
            "antoine_Tmax": 400.0,
            "property_sources": {
                "antoine_A": {
                    "source": "provided",
                    "method": "pfd_component_override",
                    "quality": 1.0,
                },
            },
        }

        with self.assertRaisesRegex(
            PsatAdapterError,
            "PFDParser should have rejected",
        ):
            PsatCanonicalizationAdapter(component).collect_inputs(
                T_min=200.0,
                T_critical=500.0,
                P_critical_bar=40.0,
            )

    def test_unprovenanced_antoine_is_not_promoted_as_curated(self):
        component = {
            "symbol": "X",
            "antoine_A": 4.0,
            "antoine_B": 1000.0,
            "antoine_C": 200.0,
            "antoine_Tmin": 250.0,
            "antoine_Tmax": 400.0,
        }

        self.assertEqual(
            PsatCanonicalizationAdapter(component).collect_inputs(),
            PsatCanonicalizationInputs(),
        )

    def test_curated_antoine_that_misses_tb_remains_exempt(self):
        component = {
            "symbol": "X",
            "source": "local",
            "Tb": 350.0,
            "antoine_A": 5.0,
            "antoine_B": 1000.0,
            "antoine_C": 200.0,
            "antoine_Tmin": 250.0,
            "antoine_Tmax": 400.0,
        }

        inputs = PsatCanonicalizationAdapter(component).collect_inputs()

        segment = next(
            item for item in inputs.segments if item.method == "local_antoine"
        )
        self.assertEqual(inputs.warnings, ())
        self.assertEqual(
            segment.metadata["tb_validation_status"],
            "exempt_priority",
        )

    def test_pfd_canonical_contract_rejects_missing_coefficients_and_conflicts(self):
        missing = self.pfd_component(
            "canonical_psat",
            {"A": 1.0, "B": -1000.0},
        )
        with self.assertRaisesRegex(PsatAdapterError, "requires coefficient C"):
            PsatCanonicalizationAdapter(missing).collect_inputs(
                T_min=200.0,
                T_critical=500.0,
                P_critical_bar=40.0,
            )

        conflict = self.pfd_component(
            "canonical_psat",
            {name: 0.0 for name in "ABCDEFG"},
            Tc_K=510.0,
        )
        with self.assertRaisesRegex(PsatAdapterError, "conflicts with component value"):
            PsatCanonicalizationAdapter(conflict).collect_inputs(
                T_min=200.0,
                T_critical=500.0,
                P_critical_bar=40.0,
            )

        nonzero_extension = self.pfd_component(
            "canonical_psat",
            {**{name: 0.0 for name in "ABCDEF"}, "G": 1.0},
        )
        with self.assertRaisesRegex(PsatAdapterError, "A-F.*nonzero G"):
            PsatCanonicalizationAdapter(nonzero_extension).collect_inputs(
                T_min=200.0,
                T_critical=500.0,
                P_critical_bar=40.0,
            )

        missing_inverse = self.pfd_component(
            "canonical_psat_ah",
            {name: 0.0 for name in "ABCDEFGH"},
        )
        with self.assertRaisesRegex(PsatAdapterError, "requires inverse_power"):
            PsatCanonicalizationAdapter(missing_inverse).collect_inputs(
                T_min=200.0,
                T_critical=500.0,
                P_critical_bar=40.0,
            )

    def test_provider_input_bundle_rejects_multiple_direct_overrides(self):
        curve = (
            PsatCanonicalizationAdapter(
                self.pfd_component(
                    "canonical_psat",
                    {
                        "A": 5.0,
                        "B": -1000.0,
                        "C": 0.0,
                        "D": 0.0,
                        "E": 0.0,
                        "F": 0.0,
                        "G": 0.0,
                    },
                )
            )
            .collect_inputs(
                T_min=200.0,
                T_critical=500.0,
                P_critical_bar=math.exp(3.0),
            )
            .canonical_override
        )

        with self.assertRaisesRegex(PsatAdapterError, "Multiple direct"):
            PsatCanonicalizationInputs(canonical_override=curve).merged(
                PsatCanonicalizationInputs(canonical_override=curve)
            )

    def test_input_resolution_chain_accepts_new_methods_without_adapter_changes(self):
        calls = []

        def custom_method(adapter, **state):
            calls.append((adapter.component, state))
            return PsatCanonicalizationInputs(metadata={"custom": True})

        component = {"symbol": "X"}
        inputs = PsatCanonicalizationAdapter(
            component,
            input_methods=(custom_method,),
        ).collect_inputs(T_min=200.0, T_critical=500.0, P_critical_bar=40.0)

        self.assertEqual(inputs.metadata, {"custom": True})
        self.assertEqual(calls[0][0], component)
        self.assertEqual(calls[0][1]["T_min"], 200.0)


if __name__ == "__main__":
    unittest.main()
