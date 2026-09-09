import math
import unittest

from property_resolution import (
    BoundaryConditionedPsatEvaluation,
    BoundaryConditionedPsatSegment,
    CanonicalPsatCurve,
    CanonicalPsatForm,
    CanonicalPsatFitPolicy,
    CanonicalPsatFitter,
    PsatAnchorRegistry,
    PsatAnchorRequirement,
    PsatAssemblyAnchorRequirement,
    PsatBoundaryConditions,
    PsatBoundaryRequirement,
    PsatCanonicalizationError,
    PsatCanonicalizer,
    PsatCompletionCoordinator,
    PsatDerivativeBasis,
    PsatDerivativeRequirement,
    PsatEndpoint,
    PsatGapConsistency,
    PsatHandoffAction,
    PsatHandoffCoordinator,
    PsatHandoffRequirement,
    PsatJunctionPolicy,
    PsatPriority,
    PsatSample,
    PsatSegment,
    PsatSegmentAssembler,
    PsatSegmentProvenance,
    PsatSegmentType,
    assess_psat_gap_consistency,
    make_c1_bridge_segment,
    probe_psat_segment,
    trim_psat_segment_to_validity,
)
from property_resolution.vapor_pressure_canonical import (
    CanonicalPsatFitDiagnostics,
    CanonicalPsatSliceFitDiagnostics,
)


class VaporPressureCanonicalArchitectureTests(unittest.TestCase):
    def make_segment(
        self,
        method,
        T_min,
        T_max,
        priority,
        offset=0.0,
        segment_type=PsatSegmentType.PINNED,
        derivative=True,
    ):
        return PsatSegment(
            source="test",
            method=method,
            segment_type=segment_type,
            priority=priority,
            T_min=T_min,
            T_max=T_max,
            ln_pressure_function=lambda T: T / 100.0 + offset,
            derivative_function=(lambda _T: 0.01) if derivative else None,
            quality=0.95,
        )

    def test_segment_retains_provider_supplied_context(self):
        segment = PsatSegment(
            source="CoolProp",
            method="HEOS",
            segment_type=PsatSegmentType.PINNED,
            priority=PsatPriority.COOLPROP_HEOS,
            T_min=160.0,
            T_max=514.0,
            ln_pressure_function=lambda T: T / 100.0,
            context={"identifier": "ethanol", "backend": "HEOS"},
        )

        self.assertEqual(segment.context["identifier"], "ethanol")
        self.assertEqual(segment.context["backend"], "HEOS")

    def test_segment_supports_analytic_and_numerical_derivatives(self):
        analytic = self.make_segment("analytic", 200.0, 500.0, 10)
        self.assertAlmostEqual(analytic.dln_pressure_dT(350.0), 0.01)
        self.assertEqual(
            analytic.derivative_basis,
            PsatDerivativeBasis.ANALYTIC,
        )
        self.assertEqual(
            PsatEndpoint.from_segment(
                analytic,
                350.0,
            ).derivative_basis,
            PsatDerivativeBasis.ANALYTIC,
        )

        numerical = PsatSegment(
            source="test",
            method="quadratic",
            segment_type=PsatSegmentType.PINNED,
            priority=10,
            T_min=200.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 1.0e-4 * T**2,
        )
        self.assertAlmostEqual(numerical.dln_pressure_dT(350.0), 0.07, places=8)
        self.assertAlmostEqual(numerical.dln_pressure_dT(200.0), 0.04, places=7)
        self.assertAlmostEqual(numerical.dln_pressure_dT(500.0), 0.10, places=7)
        self.assertEqual(
            numerical.derivative_basis,
            PsatDerivativeBasis.NUMERICAL,
        )

        interpolated = PsatSegment(
            source="test",
            method="interpolated",
            segment_type=PsatSegmentType.PINNED,
            priority=10,
            T_min=200.0,
            T_max=500.0,
            ln_pressure_function=lambda T: T / 100.0,
            derivative_function=lambda _T: 0.01,
            derivative_basis=PsatDerivativeBasis.INTERPOLATED,
        )
        self.assertEqual(
            PsatEndpoint.from_segment(
                interpolated,
                350.0,
            ).derivative_basis,
            PsatDerivativeBasis.INTERPOLATED,
        )

    def test_segment_rejects_invalid_contract_values(self):
        with self.assertRaises(PsatCanonicalizationError):
            PsatSegment(
                source="test",
                method="invalid",
                segment_type=PsatSegmentType.PINNED,
                priority=10,
                T_min=500.0,
                T_max=200.0,
                ln_pressure_function=lambda T: T,
            )
        with self.assertRaises(PsatCanonicalizationError):
            PsatSegment(
                source="test",
                method="invalid",
                segment_type=PsatSegmentType.PINNED,
                priority=10,
                T_min=200.0,
                T_max=500.0,
                ln_pressure_function=lambda T: T,
                quality=1.1,
            )

    def test_priority_arbitration_clips_lower_priority_segments(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        base = self.make_segment(
            "base",
            200.0,
            500.0,
            PsatPriority.PERRY_2_8,
        )
        override = self.make_segment(
            "override",
            300.0,
            400.0,
            PsatPriority.PFD_OVERRIDE,
            offset=0.1,
        )
        assembler.extend((base, override))

        assembly = assembler.assemble_pinned()

        self.assertTrue(assembly.covers_target())
        self.assertEqual(
            [item.segment.method for item in assembly.slices],
            ["base", "override", "base"],
        )
        self.assertEqual(
            [(item.T_min, item.T_max) for item in assembly.slices],
            [(200.0, 300.0), (300.0, 400.0), (400.0, 500.0)],
        )
        self.assertEqual(assembly.slice_at(300.0).segment.method, "override")
        self.assertAlmostEqual(assembly.ln_pressure(350.0), 3.6)

    def test_equal_priority_uses_provider_insertion_order(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        first = self.make_segment("first", 200.0, 500.0, 10)
        second = self.make_segment("second", 250.0, 450.0, 10, offset=1.0)
        assembler.extend((first, second))

        assembly = assembler.assemble()

        self.assertEqual(len(assembly.slices), 1)
        self.assertEqual(assembly.slices[0].segment.method, "first")

    def test_pinned_assembly_ignores_completion_and_reports_gaps(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        assembler.add(self.make_segment("lower", 200.0, 260.0, 50))
        assembler.add(self.make_segment("upper", 400.0, 500.0, 50))
        assembler.add(self.make_segment(
            "completion",
            260.0,
            400.0,
            PsatPriority.AMBROSE_WALTON,
            segment_type=PsatSegmentType.COMPLETION,
        ))

        pinned = assembler.assemble_pinned()
        complete = assembler.assemble()

        self.assertFalse(pinned.covers_target())
        self.assertEqual(
            [(gap.T_min, gap.T_max) for gap in pinned.coverage_gaps()],
            [(260.0, 400.0)],
        )
        self.assertTrue(complete.covers_target())

    def test_gap_consistency_finds_a_monotonic_c1_connection(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        assembler.extend((
            self.make_segment("lower", 200.0, 260.0, 50),
            self.make_segment("upper", 400.0, 500.0, 50),
        ))

        assessment = assess_psat_gap_consistency(
            assembler.assemble_pinned(),
        )[0]

        self.assertEqual(assessment.status, PsatGapConsistency.C1_CONNECTABLE)
        self.assertTrue(assessment.continuation_possible)
        self.assertAlmostEqual(assessment.secant_dln_pressure_dT, 0.01)

    def test_gap_consistency_rejects_reversed_pressure_boundaries(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        assembler.extend((
            self.make_segment("lower", 200.0, 260.0, 50),
            self.make_segment("upper", 400.0, 500.0, 50, offset=-3.0),
        ))

        assessment = assess_psat_gap_consistency(
            assembler.assemble_pinned(),
        )[0]

        self.assertEqual(assessment.status, PsatGapConsistency.INCONSISTENT)
        self.assertFalse(assessment.continuation_possible)
        self.assertIn("higher pressure", assessment.reason)

    def test_gap_consistency_marks_a_one_sided_gap_open_ended(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        assembler.add(self.make_segment("upper", 300.0, 500.0, 50))

        assessment = assess_psat_gap_consistency(
            assembler.assemble_pinned(),
        )[0]

        self.assertEqual(assessment.status, PsatGapConsistency.OPEN_ENDED)
        self.assertFalse(assessment.continuation_possible)

    def test_junctions_report_value_and_slope_mismatch(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        assembler.extend((
            self.make_segment("left", 200.0, 350.0, 20),
            self.make_segment("right", 350.0, 500.0, 20, offset=0.2),
        ))

        junction = assembler.assemble().junctions()[0]

        self.assertAlmostEqual(junction.temperature, 350.0)
        self.assertAlmostEqual(junction.delta_ln_pressure, 0.2)
        self.assertAlmostEqual(junction.relative_pressure_mismatch, math.expm1(0.2))
        self.assertAlmostEqual(junction.delta_dln_pressure_dT, 0.0)
        self.assertAlmostEqual(junction.relative_slope_mismatch, 0.0)
        self.assertFalse(junction.assess().accepted)
        self.assertTrue(junction.assess(PsatJunctionPolicy(
            max_absolute_relative_pressure_mismatch=0.25,
        )).accepted)
        with self.assertRaises(PsatCanonicalizationError):
            assembler.assemble().require_compatible_junctions()

    def test_segment_slope_exception_warns_without_changing_neighbor_quality(self):
        left = PsatSegment(
            source="test",
            method="less_reliable_completion",
            segment_type=PsatSegmentType.COMPLETION,
            priority=20,
            T_min=200.0,
            T_max=350.0,
            ln_pressure_function=lambda T: 3.5 + 0.02 * (T - 350.0),
            derivative_function=lambda _T: 0.02,
            quality=0.70,
            allow_junction_slope_mismatch=True,
        )
        right = PsatSegment(
            source="test",
            method="hard_source",
            segment_type=PsatSegmentType.PINNED,
            priority=20,
            T_min=350.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 3.5 + 0.01 * (T - 350.0),
            derivative_function=lambda _T: 0.01,
            quality=0.98,
        )
        assembler = PsatSegmentAssembler(200.0, 500.0)
        assembler.extend((left, right))

        result = PsatCompletionCoordinator(
            assembler,
            (),
        ).complete()
        assessment = result.assembly.assess_junctions()[0]

        self.assertTrue(assessment.accepted)
        self.assertEqual(assessment.reasons, ())
        self.assertEqual(len(assessment.warnings), 1)
        self.assertIn(
            "accepted by slope-mismatch exception on less_reliable_completion",
            assessment.warnings[0],
        )
        self.assertEqual(len(result.junction_warnings), 1)
        self.assertEqual(left.quality, 0.70)
        self.assertEqual(right.quality, 0.98)
        self.assertAlmostEqual(
            result.assembly.slices[0].quality,
            0.70 * 0.90,
        )
        self.assertEqual(
            result.assembly.slices[1].quality,
            0.98,
        )
        self.assertEqual(
            result.assembly.slices[0].segment.metadata[
                "junction_slope_exception_original_quality"
            ],
            0.70,
        )
        self.assertEqual(
            result.assembly.slices[0].segment.metadata[
                "junction_slope_exception_quality_factor"
            ],
            0.90,
        )

        mismatched_pressure = PsatSegment(
            source="test",
            method="pressure_mismatched_completion",
            segment_type=PsatSegmentType.COMPLETION,
            priority=20,
            T_min=200.0,
            T_max=350.0,
            ln_pressure_function=lambda T: 3.6 + 0.02 * (T - 350.0),
            derivative_function=lambda _T: 0.02,
            quality=0.70,
            allow_junction_slope_mismatch=True,
        )
        pressure_assembler = PsatSegmentAssembler(200.0, 500.0)
        pressure_assembler.extend((mismatched_pressure, right))
        pressure_assessment = (
            pressure_assembler.assemble().assess_junctions()[0]
        )
        self.assertFalse(pressure_assessment.accepted)
        self.assertTrue(any(
            "pressure mismatch" in reason
            for reason in pressure_assessment.reasons
        ))
        self.assertEqual(len(pressure_assessment.warnings), 1)

    def test_handoff_coordinator_moves_boundary_to_compatible_overlap_point(self):
        left = PsatSegment(
            source="higher",
            method="higher_source",
            segment_type=PsatSegmentType.PINNED,
            priority=700,
            T_min=200.0,
            T_max=400.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.98,
        )
        right = PsatSegment(
            source="lower",
            method="lower_source",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=300.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T + 4.0e-5 * (T - 350.0) ** 2,
            derivative_function=lambda T: 0.01 + 8.0e-5 * (T - 350.0),
            quality=0.95,
        )

        result = PsatHandoffCoordinator(200.0, 500.0).coordinate((left, right))

        self.assertTrue(result.assembly.covers_target())
        self.assertEqual(result.bridge_segments, ())
        self.assertEqual(result.rejected_segments, ())
        self.assertEqual(result.decisions[-1].action, PsatHandoffAction.DIRECT)
        boundary = result.assembly.slices[0].T_max
        self.assertLess(boundary, 400.0)
        self.assertGreater(boundary, 350.0)
        self.assertAlmostEqual(result.assembly.slices[1].T_min, boundary)
        self.assertTrue(result.assembly.assess_junctions()[0].accepted)

    def test_handoff_coordinator_builds_narrow_monotonic_c1_bridge(self):
        left = PsatSegment(
            source="higher",
            method="higher_source",
            segment_type=PsatSegmentType.PINNED,
            priority=700,
            T_min=200.0,
            T_max=400.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.98,
        )
        right = PsatSegment(
            source="lower",
            method="lower_source",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=300.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T + 0.022,
            derivative_function=lambda _T: 0.01,
            quality=0.95,
        )

        result = PsatHandoffCoordinator(200.0, 500.0).coordinate((left, right))

        self.assertTrue(result.assembly.covers_target())
        self.assertEqual(result.rejected_segments, ())
        self.assertEqual(result.decisions[-1].action, PsatHandoffAction.BRIDGE)
        self.assertEqual(len(result.bridge_segments), 1)
        self.assertEqual(
            [item.segment.method for item in result.assembly.slices],
            [
                "higher_source",
                "c1_handoff_higher_source_to_lower_source",
                "lower_source",
            ],
        )
        self.assertEqual(len(result.assembly.junctions()), 2)
        self.assertTrue(all(
            assessment.accepted
            for assessment in result.assembly.assess_junctions()
        ))
        self.assertTrue(probe_psat_segment(result.bridge_segments[0]).monotonic)

    def test_handoff_bridge_searches_beyond_a_bad_priority_boundary(self):
        left = PsatSegment(
            source="higher",
            method="higher_source",
            segment_type=PsatSegmentType.PINNED,
            priority=700,
            T_min=200.0,
            T_max=400.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.98,
        )
        minimum_offset = math.log1p(0.022)
        curvature = (math.log1p(0.03) - minimum_offset) / 2500.0
        right = PsatSegment(
            source="lower",
            method="lower_source",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=300.0,
            T_max=500.0,
            ln_pressure_function=(
                lambda T: 0.01 * T
                + minimum_offset
                + curvature * (T - 350.0) ** 2
            ),
            derivative_function=(
                lambda T: 0.01 + 2.0 * curvature * (T - 350.0)
            ),
            quality=0.95,
        )

        result = PsatHandoffCoordinator(200.0, 500.0).coordinate((left, right))

        self.assertEqual(result.rejected_segments, ())
        self.assertEqual(result.decisions[-1].action, PsatHandoffAction.BRIDGE)
        bridge = result.bridge_segments[0]
        self.assertLess(bridge.T_max, 400.0)
        self.assertGreater(bridge.T_min, 300.0)
        self.assertTrue(probe_psat_segment(bridge).monotonic)

    def test_required_overlap_accepts_reasonable_higher_preference_source(self):
        conditional = PsatSegment(
            source="conditional",
            method="conditional_antoine",
            segment_type=PsatSegmentType.PINNED,
            priority=700,
            T_min=200.0,
            T_max=400.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.98,
            handoff_requirement=(
                PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP
            ),
        )
        higher = PsatSegment(
            source="higher",
            method="higher_source",
            segment_type=PsatSegmentType.PINNED,
            priority=800,
            T_min=300.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.97,
        )

        result = PsatHandoffCoordinator(200.0, 500.0).coordinate(
            (conditional, higher),
        )

        self.assertEqual(result.rejected_segments, ())
        self.assertTrue(result.assembly.covers_target())

    def test_required_overlap_is_not_satisfied_across_a_gap(self):
        conditional = PsatSegment(
            source="conditional",
            method="conditional_antoine",
            segment_type=PsatSegmentType.PINNED,
            priority=700,
            T_min=200.0,
            T_max=300.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.98,
            handoff_requirement=(
                PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP
            ),
        )
        higher = PsatSegment(
            source="higher",
            method="higher_source",
            segment_type=PsatSegmentType.PINNED,
            priority=800,
            T_min=320.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.97,
        )

        result = PsatHandoffCoordinator(200.0, 500.0).coordinate(
            (conditional, higher),
        )

        self.assertEqual(result.rejected_segments, (conditional,))
        self.assertEqual(result.decisions[0].action, PsatHandoffAction.REJECT)
        self.assertIn("higher-preference", result.decisions[0].reason)

    def test_optional_overlap_gets_a_pass_as_the_highest_hard_source(self):
        conditional = PsatSegment(
            source="conditional",
            method="high_quality_antoine",
            segment_type=PsatSegmentType.PINNED,
            priority=700,
            T_min=200.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.98,
            handoff_requirement=(
                PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP_IF_AVAILABLE
            ),
        )
        lower = PsatSegment(
            source="lower",
            method="lower_source",
            segment_type=PsatSegmentType.PINNED,
            priority=500,
            T_min=300.0,
            T_max=450.0,
            ln_pressure_function=lambda T: 0.02 * T,
            derivative_function=lambda _T: 0.02,
            quality=0.95,
        )

        result = PsatHandoffCoordinator(200.0, 500.0).coordinate(
            (conditional, lower),
        )

        self.assertEqual(result.rejected_segments, ())
        self.assertEqual(
            [item.segment.method for item in result.assembly.slices],
            ["high_quality_antoine"],
        )

    def test_optional_overlap_is_required_when_a_higher_source_exists(self):
        conditional = PsatSegment(
            source="conditional",
            method="high_quality_antoine",
            segment_type=PsatSegmentType.PINNED,
            priority=700,
            T_min=200.0,
            T_max=300.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.98,
            handoff_requirement=(
                PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP_IF_AVAILABLE
            ),
        )
        higher = PsatSegment(
            source="higher",
            method="higher_source",
            segment_type=PsatSegmentType.PINNED,
            priority=800,
            T_min=320.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.97,
        )

        result = PsatHandoffCoordinator(200.0, 500.0).coordinate(
            (conditional, higher),
        )

        self.assertEqual(result.rejected_segments, (conditional,))
        self.assertEqual(
            [item.segment.method for item in result.assembly.slices],
            ["higher_source"],
        )

    def test_required_overlap_validation_propagates_down_preference_levels(self):
        def segment(method, priority, T_min, T_max, requirement):
            return PsatSegment(
                source=method,
                method=method,
                segment_type=PsatSegmentType.PINNED,
                priority=priority,
                T_min=T_min,
                T_max=T_max,
                ln_pressure_function=lambda T: 0.01 * T,
                derivative_function=lambda _T: 0.01,
                quality=0.95,
                handoff_requirement=requirement,
            )

        low = segment(
            "low_antoine",
            700,
            200.0,
            375.0,
            PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP,
        )
        middle = segment(
            "middle_antoine",
            800,
            275.0,
            450.0,
            PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP,
        )
        high = segment(
            "trusted_source",
            900,
            350.0,
            500.0,
            PsatHandoffRequirement.NONE,
        )

        result = PsatHandoffCoordinator(200.0, 500.0).coordinate(
            (low, middle, high),
        )

        self.assertEqual(result.rejected_segments, ())
        self.assertTrue(result.assembly.covers_target())

    def test_unvalidated_required_overlap_cannot_validate_a_lower_level(self):
        low = PsatSegment(
            source="low",
            method="low_antoine",
            segment_type=PsatSegmentType.PINNED,
            priority=700,
            T_min=200.0,
            T_max=375.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.95,
            handoff_requirement=(
                PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP
            ),
        )
        middle = PsatSegment(
            source="middle",
            method="middle_antoine",
            segment_type=PsatSegmentType.PINNED,
            priority=800,
            T_min=275.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.95,
            handoff_requirement=(
                PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP
            ),
        )

        result = PsatHandoffCoordinator(200.0, 500.0).coordinate((low, middle))

        self.assertEqual(
            {segment.method for segment in result.rejected_segments},
            {"low_antoine", "middle_antoine"},
        )
        self.assertEqual(result.assembly.slices, ())

    def test_handoff_smoothing_rejects_locally_inconsistent_overlap(self):
        left = PsatSegment(
            source="higher",
            method="higher_source",
            segment_type=PsatSegmentType.PINNED,
            priority=700,
            T_min=200.0,
            T_max=400.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.98,
        )
        inconsistent = PsatSegment(
            source="lower",
            method="inconsistent_source",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=300.0,
            T_max=500.0,
            ln_pressure_function=(
                lambda T: 0.01 * T + 0.022 + 4.0e-4 * (400.0 - T) ** 2
            ),
            derivative_function=lambda T: 0.01 - 8.0e-4 * (400.0 - T),
            quality=0.95,
        )

        result = PsatHandoffCoordinator(200.0, 500.0).coordinate(
            (left, inconsistent),
        )

        self.assertEqual(result.bridge_segments, ())
        self.assertEqual(result.rejected_segments, (inconsistent,))
        self.assertEqual(result.decisions[0].action, PsatHandoffAction.REJECT)

    def test_handoff_coordinator_rejects_lower_preference_and_promotes_next_source(self):
        left = PsatSegment(
            source="higher",
            method="higher_source",
            segment_type=PsatSegmentType.PINNED,
            priority=700,
            T_min=200.0,
            T_max=400.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.98,
        )
        conflicting = PsatSegment(
            source="middle",
            method="conflicting_source",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=300.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T + 0.30,
            derivative_function=lambda _T: 0.01,
            quality=0.95,
        )
        fallback = PsatSegment(
            source="lower",
            method="compatible_fallback",
            segment_type=PsatSegmentType.PINNED,
            priority=500,
            T_min=300.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.90,
        )

        result = PsatHandoffCoordinator(200.0, 500.0).coordinate(
            (left, conflicting, fallback),
        )

        self.assertTrue(result.assembly.covers_target())
        self.assertEqual(result.rejected_segments, (conflicting,))
        self.assertEqual(result.decisions[0].action, PsatHandoffAction.REJECT)
        self.assertEqual(result.decisions[0].rejected_method, "conflicting_source")
        self.assertEqual(
            [item.segment.method for item in result.assembly.slices],
            ["higher_source", "compatible_fallback"],
        )

    def test_handoff_coordinator_rejects_invalid_hard_segment_before_arbitration(self):
        invalid = PsatSegment(
            source="higher",
            method="invalid_source",
            segment_type=PsatSegmentType.PINNED,
            priority=700,
            T_min=200.0,
            T_max=500.0,
            ln_pressure_function=lambda T: -(T - 350.0) ** 2,
            derivative_function=lambda T: -2.0 * (T - 350.0),
            quality=0.98,
        )
        fallback = PsatSegment(
            source="lower",
            method="valid_fallback",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=200.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
            quality=0.95,
        )

        result = PsatHandoffCoordinator(200.0, 500.0).coordinate(
            (invalid, fallback),
        )

        self.assertEqual(result.rejected_segments, (invalid,))
        self.assertEqual(result.decisions[0].action, PsatHandoffAction.REJECT)
        self.assertIn("invalid hard-pinned segment", result.decisions[0].reason)
        self.assertEqual(
            [item.segment.method for item in result.assembly.slices],
            ["valid_fallback"],
        )

    def test_completion_coordinator_applies_handoffs_before_gap_completion(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        assembler.extend((
            PsatSegment(
                source="higher",
                method="higher_source",
                segment_type=PsatSegmentType.PINNED,
                priority=700,
                T_min=200.0,
                T_max=400.0,
                ln_pressure_function=lambda T: 0.01 * T,
                derivative_function=lambda _T: 0.01,
                quality=0.98,
            ),
            PsatSegment(
                source="lower",
                method="lower_source",
                segment_type=PsatSegmentType.PINNED,
                priority=600,
                T_min=300.0,
                T_max=500.0,
                ln_pressure_function=lambda T: 0.01 * T + 0.022,
                derivative_function=lambda _T: 0.01,
                quality=0.95,
            ),
        ))

        result = PsatCompletionCoordinator(assembler, ()).complete()

        self.assertTrue(result.assembly.covers_target())
        self.assertIsNotNone(result.handoffs)
        self.assertEqual(len(result.handoffs.bridge_segments), 1)
        self.assertEqual(result.generated_segments, ())
        self.assertTrue(all(
            assessment.accepted
            for assessment in result.assembly.assess_junctions()
        ))

    def test_boundary_conditioned_segment_declares_range_and_boundary_shape(self):
        right = PsatEndpoint(
            temperature=400.0,
            ln_pressure=2.0,
            dln_pressure_dT=0.01,
            source="source",
            method="right",
            quality=0.9,
        )

        def evaluation_factory(conditions):
            anchor = conditions.right
            return (
                lambda T: anchor.ln_pressure + anchor.dln_pressure_dT * (T - anchor.temperature),
                lambda _T: anchor.dln_pressure_dT,
            )

        deferred = BoundaryConditionedPsatSegment(
            source="calculated",
            method="clapeyron_like",
            segment_type=PsatSegmentType.COMPLETION,
            priority=PsatPriority.CLAPEYRON,
            quality=0.85,
            boundary_requirement=PsatBoundaryRequirement.RIGHT,
            T_min=150.0,
            T_max=450.0,
            evaluation_factory=evaluation_factory,
            context={"model": "frozen_hvap"},
        )
        segment = deferred.bind(PsatBoundaryConditions(
            T_min=200.0,
            T_max=400.0,
            right=right,
        ))

        self.assertEqual(segment.T_min, 200.0)
        self.assertEqual(segment.T_max, 400.0)
        self.assertAlmostEqual(segment.quality, 0.85)
        self.assertAlmostEqual(segment.ln_pressure(400.0), 2.0)
        self.assertEqual(segment.context["model"], "frozen_hvap")

        with self.assertRaises(PsatCanonicalizationError):
            deferred.bind(PsatBoundaryConditions(
                T_min=100.0,
                T_max=400.0,
                right=right,
            ))
        with self.assertRaises(PsatCanonicalizationError):
            deferred.bind(PsatBoundaryConditions(200.0, 400.0))

    def test_boundary_condition_can_require_a_hard_pinned_source(self):
        deferred = BoundaryConditionedPsatSegment(
            source="calculated",
            method="hard_source_completion",
            segment_type=PsatSegmentType.COMPLETION,
            priority=PsatPriority.PRIMARY_COMPLETION,
            quality=0.9,
            boundary_requirement=PsatBoundaryRequirement.LEFT,
            T_min=300.0,
            T_max=500.0,
            evaluation_factory=lambda conditions: (
                lambda T: (
                    conditions.left.ln_pressure
                    + conditions.left.dln_pressure_dT
                    * (T - conditions.left.temperature)
                ),
                lambda _T: conditions.left.dln_pressure_dT,
            ),
            allowed_left_segment_types=(
                PsatSegmentType.PINNED,
                PsatSegmentType.CANONICAL_OVERRIDE,
            ),
        )
        pinned = self.make_segment("pinned", 300.0, 400.0, 800)
        pinned_endpoint = PsatEndpoint.from_segment(pinned, 400.0)
        bound = deferred.bind(PsatBoundaryConditions(
            T_min=400.0,
            T_max=500.0,
            left=pinned_endpoint,
        ))

        self.assertEqual(
            pinned_endpoint.segment_type,
            PsatSegmentType.PINNED,
        )
        self.assertEqual(bound.method, "hard_source_completion")

        fixed_endpoint = PsatEndpoint.fixed_pressure_point(
            400.0,
            math.exp(pinned_endpoint.ln_pressure),
            source="fixed_anchor",
            method="fixed",
            quality=0.95,
        )
        with self.assertRaisesRegex(
            PsatCanonicalizationError,
            "received fixed anchor",
        ):
            deferred.bind(PsatBoundaryConditions(
                T_min=400.0,
                T_max=500.0,
                left=fixed_endpoint,
            ))

    def test_completion_supplies_requested_interior_assembly_anchor(self):
        hard = self.make_segment("hard", 300.0, 400.0, 800)
        assembler = PsatSegmentAssembler(300.0, 500.0)
        assembler.add(hard)
        seen = {}

        def evaluation_factory(conditions):
            interior = conditions.anchor("interior")
            seen["anchor"] = interior
            left = conditions.left
            return BoundaryConditionedPsatEvaluation(
                ln_pressure_function=lambda T: (
                    left.ln_pressure + left.dln_pressure_dT * (
                    T - left.temperature
                    )
                ),
                derivative_function=lambda _T: left.dln_pressure_dT,
                quality=0.72,
                metadata={"bound_parameter": 42.0},
            )

        relation = BoundaryConditionedPsatSegment(
            source="calculated",
            method="interior_anchored",
            segment_type=PsatSegmentType.COMPLETION,
            priority=PsatPriority.PRIMARY_COMPLETION,
            quality=0.9,
            boundary_requirement=PsatBoundaryRequirement.LEFT,
            T_min=350.0,
            T_max=500.0,
            evaluation_factory=evaluation_factory,
            assembly_anchor_requirements=(
                PsatAssemblyAnchorRequirement(
                    name="interior",
                    temperature=350.0,
                    allowed_segment_types=(PsatSegmentType.PINNED,),
                ),
            ),
        )

        result = PsatCompletionCoordinator(
            assembler,
            (relation,),
        ).complete()

        self.assertTrue(result.assembly.covers_target())
        self.assertEqual(seen["anchor"].temperature, 350.0)
        self.assertEqual(
            seen["anchor"].segment_type,
            PsatSegmentType.PINNED,
        )
        self.assertEqual(seen["anchor"].method, "hard")
        self.assertEqual(result.generated_segments[0].quality, 0.72)
        self.assertEqual(
            result.generated_segments[0].metadata["bound_parameter"],
            42.0,
        )

    def test_fixed_tb_tc_anchors_are_value_only_and_globally_available(self):
        anchors = PsatAnchorRegistry.from_tb_tc(
            T_critical=500.0,
            P_critical_bar=10.0,
            critical_quality=0.96,
            T_boiling=350.0,
            boiling_quality=0.98,
        )

        boiling = anchors.named("Tb")
        critical = anchors.named("Tc")

        self.assertAlmostEqual(boiling.temperature, 350.0)
        self.assertAlmostEqual(math.exp(boiling.ln_pressure), 1.01325)
        self.assertIsNone(boiling.dln_pressure_dT)
        self.assertAlmostEqual(critical.temperature, 500.0)
        self.assertAlmostEqual(math.exp(critical.ln_pressure), 10.0)
        self.assertIsNone(critical.dln_pressure_dT)

    def test_derivative_contract_distinguishes_value_only_boundaries(self):
        right = PsatEndpoint.fixed_pressure_point(
            400.0,
            2.0,
            source="fixed",
            method="fixed_point",
            quality=0.95,
        )
        conditions = PsatBoundaryConditions(200.0, 400.0, right=right)

        def factory(boundaries):
            return (
                lambda T: boundaries.right.ln_pressure + 0.01 * (T - 400.0),
                lambda _T: 0.01,
            )

        required = BoundaryConditionedPsatSegment(
            source="calculated",
            method="slope_matched",
            segment_type=PsatSegmentType.COMPLETION,
            priority=10,
            quality=0.9,
            boundary_requirement=PsatBoundaryRequirement.RIGHT,
            T_min=200.0,
            T_max=400.0,
            evaluation_factory=factory,
            right_derivative_requirement=PsatDerivativeRequirement.REQUIRED,
        )
        optional = BoundaryConditionedPsatSegment(
            source="calculated",
            method="value_anchored",
            segment_type=PsatSegmentType.COMPLETION,
            priority=10,
            quality=0.9,
            boundary_requirement=PsatBoundaryRequirement.RIGHT,
            T_min=200.0,
            T_max=400.0,
            evaluation_factory=factory,
            right_derivative_requirement=PsatDerivativeRequirement.OPTIONAL,
        )

        with self.assertRaises(PsatCanonicalizationError):
            required.bind(conditions)
        self.assertAlmostEqual(optional.bind(conditions).ln_pressure(400.0), math.log(2.0))

        right_with_name = PsatEndpoint.fixed_pressure_point(
            400.0,
            2.0,
            source="fixed",
            method="fixed_point",
            quality=0.95,
            context={"anchor_name": "Tb"},
        )
        anchor_conditions = PsatBoundaryConditions(
            200.0,
            400.0,
            anchors=(right_with_name,),
        )
        anchor_slope_required = BoundaryConditionedPsatSegment(
            source="calculated",
            method="anchor_slope_required",
            segment_type=PsatSegmentType.COMPLETION,
            priority=10,
            quality=0.9,
            boundary_requirement=PsatBoundaryRequirement.NONE,
            T_min=200.0,
            T_max=400.0,
            evaluation_factory=factory,
            anchor_requirements=(PsatAnchorRequirement(
                "Tb",
                PsatDerivativeRequirement.REQUIRED,
            ),),
        )
        with self.assertRaises(PsatCanonicalizationError):
            anchor_slope_required.bind(anchor_conditions)

    def test_pressure_validity_trims_a_monotonic_relation_at_crossings(self):
        segment = PsatSegment(
            source="test",
            method="pressure_bounded",
            segment_type=PsatSegmentType.COMPLETION,
            priority=10,
            T_min=200.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.02 * (T - 300.0) + math.log(0.25),
            derivative_function=lambda _T: 0.02,
            P_min_bar=0.25,
            P_max_bar=2.0,
        )

        trimmed = trim_psat_segment_to_validity(segment)

        self.assertAlmostEqual(trimmed.T_min, 300.0, places=8)
        self.assertAlmostEqual(trimmed.pressure_bar(trimmed.T_min), 0.25, places=10)
        self.assertAlmostEqual(trimmed.pressure_bar(trimmed.T_max), 2.0, places=10)
        with self.assertRaises(PsatCanonicalizationError):
            segment.ln_pressure(250.0)

    def test_completion_coordinator_cross_binds_pressure_limited_relations(self):
        T_min = 200.0
        T_boiling = 350.0
        T_critical = 500.0
        P_critical_bar = 10.0
        slope = math.log(P_critical_bar / 1.01325) / (T_critical - T_boiling)
        intercept = math.log(1.01325) - slope * T_boiling
        switch_temperature = (math.log(0.25) - intercept) / slope
        observed_anchors = []

        def upper_factory(conditions):
            observed_anchors.append((
                conditions.anchor("Tb"),
                conditions.anchor("Tc"),
            ))
            return (
                lambda T: intercept + slope * T,
                lambda _T: slope,
            )

        def lower_factory(conditions):
            boundary = conditions.right
            continuation_slope = (
                boundary.dln_pressure_dT
                if boundary.dln_pressure_dT is not None
                else slope
            )
            return (
                lambda T: boundary.ln_pressure
                + continuation_slope * (T - boundary.temperature),
                lambda _T: continuation_slope,
            )

        upper = BoundaryConditionedPsatSegment(
            source="calculated",
            method="upper_relation",
            segment_type=PsatSegmentType.COMPLETION,
            priority=300,
            quality=0.95,
            boundary_requirement=PsatBoundaryRequirement.NONE,
            T_min=T_min,
            T_max=T_critical,
            P_min_bar=0.25,
            P_max_bar=P_critical_bar,
            evaluation_factory=upper_factory,
            required_anchor_names=("Tb", "Tc"),
        )
        lower = BoundaryConditionedPsatSegment(
            source="calculated",
            method="lower_relation",
            segment_type=PsatSegmentType.COMPLETION,
            priority=290,
            quality=0.92,
            boundary_requirement=PsatBoundaryRequirement.RIGHT,
            T_min=T_min,
            T_max=T_critical,
            P_max_bar=0.25,
            evaluation_factory=lower_factory,
            right_derivative_requirement=PsatDerivativeRequirement.OPTIONAL,
        )
        temperatures = [
            T_min + (T_critical - T_min) * index / 120.0
            for index in range(121)
        ]
        canonicalized = PsatCanonicalizer(
            T_min=T_min,
            T_critical=T_critical,
            P_critical_bar=P_critical_bar,
            critical_quality=0.96,
            T_boiling=T_boiling,
            boiling_quality=0.98,
            fit_policy=CanonicalPsatFitPolicy(
                max_mard_percent=1.0e-5,
                max_absolute_relative_error_percent=1.0e-4,
            ),
        ).canonicalize(
            (),
            (upper, lower),
            temperatures=temperatures,
        )
        result = canonicalized.completion

        self.assertTrue(result.assembly.covers_target())
        self.assertEqual(
            [item.segment.method for item in result.assembly.slices],
            ["lower_relation", "upper_relation"],
        )
        self.assertAlmostEqual(result.assembly.slices[0].T_max, switch_temperature, places=7)
        self.assertAlmostEqual(result.assembly.slices[1].T_min, switch_temperature, places=7)
        self.assertAlmostEqual(
            result.assembly.slices[0].segment.pressure_bar(switch_temperature),
            0.25,
            places=9,
        )
        self.assertEqual(
            result.assembly.slices[0].segment.context["right_boundary_method"],
            "upper_relation",
        )
        self.assertGreaterEqual(len(observed_anchors), 1)
        self.assertTrue(all(
            boiling.dln_pressure_dT is None and critical.dln_pressure_dT is None
            for boiling, critical in observed_anchors
        ))
        self.assertTrue(all(item.accepted for item in result.assembly.assess_junctions()))

        curve = canonicalized.curve
        self.assertAlmostEqual(curve.pressure_bar(T_boiling), 1.01325, places=10)
        self.assertAlmostEqual(curve.pressure_bar(T_critical), P_critical_bar, places=10)
        self.assertTrue(curve.diagnostics.monotonic)

    def test_completion_coordinator_fills_a_missing_middle_from_both_sides(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        lower = PsatSegment(
            source="source",
            method="lower_source",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=200.0,
            T_max=300.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
        )
        upper = PsatSegment(
            source="source",
            method="upper_source",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=400.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
        )
        assembler.extend((lower, upper))

        def middle_factory(conditions):
            left = conditions.left
            right = conditions.right
            span = right.temperature - left.temperature
            slope = (right.ln_pressure - left.ln_pressure) / span
            return (
                lambda T: left.ln_pressure + slope * (T - left.temperature),
                lambda _T: slope,
            )

        middle = BoundaryConditionedPsatSegment(
            source="calculated",
            method="middle_completion",
            segment_type=PsatSegmentType.COMPLETION,
            priority=300,
            quality=0.90,
            boundary_requirement=PsatBoundaryRequirement.BOTH,
            T_min=250.0,
            T_max=450.0,
            evaluation_factory=middle_factory,
            left_derivative_requirement=PsatDerivativeRequirement.REQUIRED,
            right_derivative_requirement=PsatDerivativeRequirement.REQUIRED,
        )

        result = PsatCompletionCoordinator(assembler, (middle,)).complete()

        self.assertTrue(result.assembly.covers_target())
        self.assertEqual(
            [item.segment.method for item in result.assembly.slices],
            ["lower_source", "middle_completion", "upper_source"],
        )
        self.assertEqual(
            [(item.T_min, item.T_max) for item in result.assembly.slices],
            [(200.0, 300.0), (300.0, 400.0), (400.0, 500.0)],
        )
        self.assertTrue(all(item.accepted for item in result.assembly.assess_junctions()))

    def test_no_hard_relation_runs_only_for_an_empty_hard_assembly(self):
        relation = BoundaryConditionedPsatSegment(
            source="calculated",
            method="no_hard_fallback",
            segment_type=PsatSegmentType.FALLBACK,
            priority=200,
            quality=0.80,
            boundary_requirement=PsatBoundaryRequirement.NONE,
            T_min=200.0,
            T_max=500.0,
            evaluation_factory=lambda _conditions: (
                lambda T: 0.01 * T,
                lambda _T: 0.01,
            ),
            requires_no_hard_segments=True,
        )
        empty = PsatSegmentAssembler(200.0, 500.0)

        completed = PsatCompletionCoordinator(
            empty,
            (relation,),
        ).complete()

        self.assertEqual(
            [item.method for item in completed.generated_segments],
            ["no_hard_fallback"],
        )

        with_hard = PsatSegmentAssembler(200.0, 500.0)
        with_hard.add(PsatSegment(
            source="hard",
            method="upper_hard",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=300.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
        ))

        skipped = PsatCompletionCoordinator(
            with_hard,
            (relation,),
        ).complete(require_complete=False)

        self.assertEqual(skipped.generated_segments, ())
        self.assertEqual(
            [
                (gap.T_min, gap.T_max)
                for gap in skipped.assembly.coverage_gaps()
            ],
            [(200.0, 300.0)],
        )

    def test_completion_coordinator_fills_only_a_missing_bottom(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        upper = PsatSegment(
            source="source",
            method="complete_upper",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=300.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
        )
        assembler.add(upper)
        calls = []

        def lower_factory(conditions):
            calls.append(conditions)
            right = conditions.right
            return (
                lambda T: right.ln_pressure + 0.01 * (T - right.temperature),
                lambda _T: 0.01,
            )

        lower = BoundaryConditionedPsatSegment(
            source="calculated",
            method="bottom_completion",
            segment_type=PsatSegmentType.COMPLETION,
            priority=300,
            quality=0.90,
            boundary_requirement=PsatBoundaryRequirement.RIGHT,
            T_min=150.0,
            T_max=350.0,
            evaluation_factory=lower_factory,
        )

        result = PsatCompletionCoordinator(assembler, (lower,)).complete()

        self.assertTrue(result.assembly.covers_target())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].T_min, 200.0)
        self.assertEqual(calls[0].T_max, 300.0)
        self.assertEqual(
            [item.segment.method for item in result.assembly.slices],
            ["bottom_completion", "complete_upper"],
        )

    def test_completion_coordinator_fills_only_a_missing_top(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        assembler.add(PsatSegment(
            source="source",
            method="complete_lower",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=200.0,
            T_max=300.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
        ))

        def upper_factory(conditions):
            left = conditions.left
            return (
                lambda T: left.ln_pressure + 0.01 * (T - left.temperature),
                lambda _T: 0.01,
            )

        upper = BoundaryConditionedPsatSegment(
            source="calculated",
            method="top_completion",
            segment_type=PsatSegmentType.COMPLETION,
            priority=300,
            quality=0.90,
            boundary_requirement=PsatBoundaryRequirement.LEFT,
            T_min=250.0,
            T_max=550.0,
            evaluation_factory=upper_factory,
        )

        result = PsatCompletionCoordinator(assembler, (upper,)).complete()

        self.assertTrue(result.assembly.covers_target())
        self.assertEqual(
            [item.segment.method for item in result.assembly.slices],
            ["complete_lower", "top_completion"],
        )

    def test_completion_coordinator_uses_lower_priority_relation_after_rejection(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        assembler.add(PsatSegment(
            source="source",
            method="upper_source",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=300.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
        ))

        def factory(conditions):
            right = conditions.right
            return (
                lambda T: right.ln_pressure + 0.01 * (T - right.temperature),
                lambda _T: 0.01,
            )

        rejected = BoundaryConditionedPsatSegment(
            source="calculated",
            method="missing_anchor_method",
            segment_type=PsatSegmentType.COMPLETION,
            priority=400,
            quality=0.95,
            boundary_requirement=PsatBoundaryRequirement.RIGHT,
            T_min=200.0,
            T_max=300.0,
            evaluation_factory=factory,
            required_anchor_names=("unavailable",),
        )
        fallback = BoundaryConditionedPsatSegment(
            source="calculated",
            method="usable_fallback",
            segment_type=PsatSegmentType.FALLBACK,
            priority=100,
            quality=0.70,
            boundary_requirement=PsatBoundaryRequirement.RIGHT,
            T_min=200.0,
            T_max=300.0,
            evaluation_factory=factory,
        )

        result = PsatCompletionCoordinator(
            assembler,
            (rejected, fallback),
        ).complete()

        self.assertTrue(result.assembly.covers_target())
        self.assertEqual(result.generated_segments[0].method, "usable_fallback")
        self.assertTrue(any("missing_anchor_method" in item for item in result.rejected_attempts))

    def test_completion_coordinator_does_nothing_to_an_already_complete_curve(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        assembler.add(PsatSegment(
            source="source",
            method="already_complete",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=200.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
        ))
        calls = []

        def unused_factory(conditions):
            calls.append(conditions)
            return lambda T: 0.01 * T, lambda _T: 0.01

        unused = BoundaryConditionedPsatSegment(
            source="calculated",
            method="must_not_run",
            segment_type=PsatSegmentType.FALLBACK,
            priority=100,
            quality=0.5,
            boundary_requirement=PsatBoundaryRequirement.NONE,
            T_min=200.0,
            T_max=500.0,
            evaluation_factory=unused_factory,
        )

        result = PsatCompletionCoordinator(assembler, (unused,)).complete()

        self.assertTrue(result.assembly.covers_target())
        self.assertEqual(result.generated_segments, ())
        self.assertEqual(calls, [])
        self.assertEqual(len(result.assembly.slices), 1)

    def test_completion_coordinator_reports_an_unfillable_gap(self):
        assembler = PsatSegmentAssembler(200.0, 500.0)
        assembler.add(PsatSegment(
            source="source",
            method="upper_only",
            segment_type=PsatSegmentType.PINNED,
            priority=600,
            T_min=400.0,
            T_max=500.0,
            ln_pressure_function=lambda T: 0.01 * T,
            derivative_function=lambda _T: 0.01,
        ))

        with self.assertRaisesRegex(
            PsatCanonicalizationError,
            "uncovered gaps: 200-400 K",
        ):
            PsatCompletionCoordinator(assembler, ()).complete()

    def test_c1_bridge_preserves_endpoint_values_and_derivatives(self):
        left = PsatEndpoint(
            temperature=300.0,
            ln_pressure=0.0,
            dln_pressure_dT=0.01,
            source="left",
            method="left_curve",
            quality=0.95,
        )
        right = PsatEndpoint(
            temperature=400.0,
            ln_pressure=1.0,
            dln_pressure_dT=0.01,
            source="right",
            method="right_curve",
            quality=0.90,
        )

        bridge = make_c1_bridge_segment(
            left,
            right,
            source="calculated",
            method="hermite_bridge",
        )
        probe = probe_psat_segment(bridge)

        self.assertAlmostEqual(bridge.ln_pressure(300.0), 0.0)
        self.assertAlmostEqual(bridge.ln_pressure(400.0), 1.0)
        self.assertAlmostEqual(bridge.dln_pressure_dT(300.0), 0.01)
        self.assertAlmostEqual(bridge.dln_pressure_dT(400.0), 0.01)
        self.assertAlmostEqual(bridge.quality, 0.90)
        self.assertTrue(probe.finite)
        self.assertTrue(probe.monotonic)

    def test_log_temperature_c1_bridge_recovers_a_cubic_log_temperature_curve(self):
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
            return (
                8.0 + coordinate - 0.3 * coordinate**2
            ) / T

        left = PsatEndpoint(
            temperature=300.0,
            ln_pressure=ln_pressure(300.0),
            dln_pressure_dT=derivative(300.0),
            source="left",
            method="left_curve",
            quality=0.96,
        )
        right = PsatEndpoint(
            temperature=500.0,
            ln_pressure=ln_pressure(500.0),
            dln_pressure_dT=derivative(500.0),
            source="right",
            method="right_curve",
            quality=0.94,
        )

        bridge = make_c1_bridge_segment(
            left,
            right,
            source="calculated",
            method="log_temperature_bridge",
            coordinate="log_temperature",
        )

        for index in range(41):
            temperature = 300.0 + 200.0 * index / 40.0
            with self.subTest(temperature=temperature):
                self.assertAlmostEqual(
                    bridge.ln_pressure(temperature),
                    ln_pressure(temperature),
                    places=12,
                )
                self.assertAlmostEqual(
                    bridge.dln_pressure_dT(temperature),
                    derivative(temperature),
                    places=12,
                )
        self.assertEqual(
            bridge.metadata["bridge_coordinate"],
            "log_temperature",
        )
        self.assertEqual(bridge.quality, 0.94)

    def test_provenance_contains_only_the_selected_slice(self):
        segment = self.make_segment("source", 200.0, 500.0, 10)
        assembler = PsatSegmentAssembler(250.0, 450.0)
        assembler.add(segment)
        item = assembler.assemble().slices[0]

        provenance = PsatSegmentProvenance.from_slice(item)

        self.assertEqual(provenance.T_min, 250.0)
        self.assertEqual(provenance.T_max, 450.0)
        self.assertEqual(provenance.segment_type, "pinned")

    def test_curve_quality_is_log_pressure_weighted_with_middle_emphasis(self):
        from property_resolution import (
            log_pressure_weighted_psat_quality,
        )

        def segment(method, T_min, T_max, P_min, P_max, quality):
            slope = math.log(P_max / P_min) / (T_max - T_min)
            return PsatSegment(
                source="test",
                method=method,
                segment_type=PsatSegmentType.COMPLETION,
                priority=200,
                T_min=T_min,
                T_max=T_max,
                ln_pressure_function=lambda T, T_min=T_min,
                P_min=P_min, slope=slope: (
                    math.log(P_min) + slope * (T - T_min)
                ),
                derivative_function=lambda _T, slope=slope: slope,
                quality=quality,
            )

        assembler = PsatSegmentAssembler(200.0, 500.0)
        assembler.extend((
            segment("deep", 200.0, 300.0, 0.001, 0.1, 0.50),
            segment("middle", 300.0, 400.0, 0.1, 5.0, 0.90),
            segment("upper", 400.0, 500.0, 5.0, 50.0, 0.70),
        ))

        quality, breakdown = log_pressure_weighted_psat_quality(
            assembler.assemble(),
        )

        deep_weight = math.log(0.1 / 0.001)
        middle_weight = 3.0 * math.log(5.0 / 0.1)
        upper_weight = math.log(50.0 / 5.0)
        expected = (
            0.50 * deep_weight
            + 0.90 * middle_weight
            + 0.70 * upper_weight
        ) / (deep_weight + middle_weight + upper_weight)
        self.assertAlmostEqual(quality, expected)
        self.assertEqual(len(breakdown), 3)
        self.assertGreater(quality, (0.50 + 0.90 + 0.70) / 3.0)

    def test_canonical_curve_evaluates_the_a_through_h_form(self):
        curve = CanonicalPsatCurve(
            A=1.0,
            B=-100.0,
            C=0.2,
            D=0.003,
            E=1.0e-6,
            F=1.0e-15,
            G=1.0e-9,
            H=0.05,
            inverse_power=-3,
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=10.0,
            T_boiling=350.0,
        )
        T = 325.0
        expected_ln_pressure = (
            1.0
            - 100.0 / T
            + 0.2 * math.log(T)
            + 0.003 * T
            + 1.0e-6 * T**2
            + 1.0e-15 * T**5
            + 1.0e-9 * T**3
            + 0.05 * ((T / 500.0) ** -3 - 1.0)
        )
        expected_derivative = (
            100.0 / T**2
            + 0.2 / T
            + 0.003
            + 2.0e-6 * T
            + 5.0e-15 * T**4
            + 3.0e-9 * T**2
            + 0.05 * -3.0 / 500.0 * (T / 500.0) ** -4
        )

        self.assertAlmostEqual(curve.ln_pressure(T), expected_ln_pressure)
        self.assertAlmostEqual(curve.pressure_bar(T), math.exp(expected_ln_pressure))
        self.assertAlmostEqual(curve.dln_pressure_dT(T), expected_derivative)
        expected_critical_derivative = (
            100.0 / 500.0**2
            + 0.2 / 500.0
            + 0.003
            + 2.0e-6 * 500.0
            + 5.0e-15 * 500.0**4
            + 3.0e-9 * 500.0**2
            + 0.05 * -3.0 / 500.0
        )
        self.assertAlmostEqual(
            curve.supercritical_slope,
            curve.P_critical_bar * expected_critical_derivative,
        )
        expected_lower_derivative = (
            100.0 / 200.0**2
            + 0.2 / 200.0
            + 0.003
            + 2.0e-6 * 200.0
            + 5.0e-15 * 200.0**4
            + 3.0e-9 * 200.0**2
            + 0.05 * -3.0 / 500.0 * (200.0 / 500.0) ** -4
        )
        self.assertAlmostEqual(
            curve.lower_continuation_slope,
            expected_lower_derivative,
        )
        self.assertEqual(
            curve.coefficients,
            (1.0, -100.0, 0.2, 0.003, 1.0e-6, 1.0e-15, 1.0e-9, 0.05),
        )
        self.assertEqual(curve.form, CanonicalPsatForm.AH)

        with self.assertRaises(PsatCanonicalizationError):
            curve.ln_pressure(501.0)

    def test_canonical_curve_reports_anchor_residuals_without_rewriting_itself(self):
        curve = CanonicalPsatCurve(
            A=math.log(5.0),
            B=0.0,
            C=0.0,
            D=0.0,
            E=0.0,
            F=0.0,
            G=0.0,
            T_min=200.0,
            T_critical=500.0,
            P_critical_bar=5.0,
            T_boiling=350.0,
        )

        self.assertAlmostEqual(curve.critical_anchor_log_residual, 0.0)
        self.assertAlmostEqual(
            curve.boiling_anchor_log_residual,
            math.log(5.0) - math.log(1.01325),
        )

    def test_constrained_fitter_recovers_samples_and_exact_anchors(self):
        T_min = 200.0
        T_boiling = 350.0
        T_critical = 500.0
        P_critical_bar = 10.0
        boiling_ln_pressure = math.log(1.01325)
        critical_ln_pressure = math.log(P_critical_bar)
        B = (
            boiling_ln_pressure - critical_ln_pressure
        ) / (1.0 / T_boiling - 1.0 / T_critical)
        A = critical_ln_pressure - B / T_critical

        def source_ln_pressure(T):
            return A + B / T

        samples = tuple(
            PsatSample(
                temperature=T_min + (T_critical - T_min) * index / 60.0,
                ln_pressure=source_ln_pressure(
                    T_min + (T_critical - T_min) * index / 60.0
                ),
                source="synthetic",
                method="two_term",
            )
            for index in range(61)
        )
        fitter = CanonicalPsatFitter(CanonicalPsatFitPolicy(
            max_mard_percent=1.0e-6,
            max_p95_absolute_relative_error_percent=1.0e-6,
            max_absolute_relative_error_percent=1.0e-5,
        ))

        curve = fitter.fit(
            samples,
            T_min=T_min,
            T_critical=T_critical,
            P_critical_bar=P_critical_bar,
            T_boiling=T_boiling,
            quality=0.97,
        )

        self.assertAlmostEqual(curve.pressure_bar(T_boiling), 1.01325, places=10)
        self.assertAlmostEqual(curve.pressure_bar(T_critical), P_critical_bar, places=10)
        self.assertLess(curve.diagnostics.mard_percent, 1.0e-6)
        self.assertLess(curve.diagnostics.max_anchor_log_residual, 1.0e-10)
        self.assertTrue(curve.diagnostics.monotonic)
        self.assertEqual(curve.quality, 0.97)
        self.assertEqual(curve.form, CanonicalPsatForm.AF)
        self.assertEqual(curve.metadata["attempted_forms"], ("A-F",))

    def test_fitter_retries_a_g_before_a_h(self):
        T_min = 50.0
        T_boiling = 300.0
        T_critical = 500.0
        P_critical_bar = 20.0
        G = 3.0e-8
        boiling_ln_pressure = math.log(1.01325)
        critical_ln_pressure = math.log(P_critical_bar)
        B = (
            boiling_ln_pressure
            - critical_ln_pressure
            - G * (T_boiling**3 - T_critical**3)
        ) / (1.0 / T_boiling - 1.0 / T_critical)
        A = critical_ln_pressure - B / T_critical - G * T_critical**3

        def source_ln_pressure(T):
            return A + B / T + G * T**3

        samples = tuple(
            PsatSample(
                temperature=T_min + (T_critical - T_min) * index / 200.0,
                ln_pressure=source_ln_pressure(
                    T_min + (T_critical - T_min) * index / 200.0
                ),
            )
            for index in range(201)
        )

        curve = CanonicalPsatFitter().fit(
            samples,
            T_min=T_min,
            T_critical=T_critical,
            P_critical_bar=P_critical_bar,
            T_boiling=T_boiling,
        )

        self.assertEqual(curve.form, CanonicalPsatForm.AG)
        self.assertEqual(curve.metadata["attempted_forms"], ("A-F", "A-G"))
        self.assertEqual(curve.metadata["attempted_inverse_powers"], ())
        self.assertLess(curve.diagnostics.max_absolute_relative_error_percent, 1.0e-8)

    def test_pfd_local_fidelity_forces_a_g_retry(self):
        T_min = 200.0
        T_boiling = 350.0
        T_critical = 500.0
        P_critical_bar = 20.0
        G = 3.0e-8
        boiling_ln_pressure = math.log(1.01325)
        critical_ln_pressure = math.log(P_critical_bar)
        B = (
            boiling_ln_pressure
            - critical_ln_pressure
            - G * (T_boiling**3 - T_critical**3)
        ) / (1.0 / T_boiling - 1.0 / T_critical)
        A = critical_ln_pressure - B / T_critical - G * T_critical**3

        def source_ln_pressure(T):
            return A + B / T + G * T**3

        def source_derivative(T):
            return -B / T**2 + 3.0 * G * T**2

        assembler = PsatSegmentAssembler(T_min, T_critical)
        for lower, upper, priority, quality, method in (
            (200.0, 300.0, 300, 0.80, "lower_completion"),
            (300.0, 400.0, 1000, 1.00, "pfd_psat_poly_x"),
            (400.0, 500.0, 300, 0.80, "upper_completion"),
        ):
            assembler.add(PsatSegment(
                source=("provided" if priority == 1000 else "calculated"),
                method=method,
                segment_type=PsatSegmentType.PINNED,
                priority=priority,
                quality=quality,
                T_min=lower,
                T_max=upper,
                ln_pressure_function=source_ln_pressure,
                derivative_function=source_derivative,
            ))

        curve = CanonicalPsatFitter(CanonicalPsatFitPolicy(
            pfd_slice_mard_percent=1.0e-3,
            pfd_slice_max_error_percent=3.0e-3,
        )).fit_assembly(
            assembler.assemble(),
            (
                T_min
                + (T_critical - T_min) * index / 400.0
                for index in range(401)
            ),
            P_critical_bar=P_critical_bar,
            T_boiling=T_boiling,
        )

        self.assertEqual(curve.form, CanonicalPsatForm.AG)
        self.assertEqual(curve.metadata["attempted_forms"], ("A-F", "A-G"))
        pfd_diagnostics = next(
            item
            for item in curve.metadata["slice_fit_diagnostics"]
            if item["priority"] == PsatPriority.PFD_OVERRIDE
        )
        self.assertLess(pfd_diagnostics["mard_percent"], 1.0e-8)
        self.assertLess(
            pfd_diagnostics["max_absolute_relative_error_percent"],
            1.0e-8,
        )

    def test_fit_tolerance_scales_only_with_overall_quality(self):
        policy = CanonicalPsatFitPolicy()
        full = PsatSegmentProvenance(
            source="local",
            method="full",
            segment_type="pinned",
            priority=900,
            quality=0.995,
            T_min=200.0,
            T_max=500.0,
        )
        extended = PsatSegmentProvenance(
            source="calculated",
            method="extension",
            segment_type="completion",
            priority=300,
            quality=0.995,
            T_min=200.0,
            T_max=500.0,
        )

        full_tolerance = policy.tolerance_for(
            quality=0.995,
            provenance=(full,),
            T_min=200.0,
            T_critical=500.0,
        )
        extended_tolerance = policy.tolerance_for(
            quality=0.995,
            provenance=(extended,),
            T_min=200.0,
            T_critical=500.0,
        )
        lower_quality_tolerance = policy.tolerance_for(
            quality=0.80,
            provenance=(full,),
            T_min=200.0,
            T_critical=500.0,
        )

        self.assertEqual(
            full_tolerance.profile,
            "quality_scaled_one_shot",
        )
        self.assertEqual(
            extended_tolerance.profile,
            "quality_scaled_assembled",
        )
        self.assertAlmostEqual(
            full_tolerance.mard_threshold_percent,
            0.1,
        )
        self.assertAlmostEqual(
            full_tolerance.max_error_threshold_percent,
            0.5,
        )
        self.assertEqual(
            full_tolerance.mard_threshold_percent,
            extended_tolerance.mard_threshold_percent,
        )
        self.assertEqual(
            full_tolerance.max_error_threshold_percent,
            extended_tolerance.max_error_threshold_percent,
        )
        self.assertAlmostEqual(
            lower_quality_tolerance.mard_threshold_percent,
            4.0,
        )
        self.assertAlmostEqual(
            lower_quality_tolerance.max_error_threshold_percent,
            20.0,
        )

    def test_pfd_slice_fit_contract_is_local_to_priority_1000(self):
        def diagnostics(priority, mard, maximum):
            return CanonicalPsatSliceFitDiagnostics(
                source="provided",
                method="pfd_psat_poly_x",
                priority=priority,
                quality=1.0,
                T_min=290.0,
                T_max=395.0,
                sample_count=201,
                mard_percent=mard,
                p95_absolute_relative_error_percent=maximum,
                max_absolute_relative_error_percent=maximum,
            )

        policy = CanonicalPsatFitPolicy()

        self.assertEqual(
            policy.pfd_slice_rejection_reasons((
                diagnostics(PsatPriority.COOLPROP_HEOS, 5.0, 10.0),
                diagnostics(PsatPriority.PFD_OVERRIDE, 0.20, 0.90),
            )),
            (),
        )
        reasons = policy.pfd_slice_rejection_reasons((
            diagnostics(PsatPriority.PFD_OVERRIDE, 0.30, 1.10),
        ))
        self.assertEqual(len(reasons), 2)
        self.assertIn("MARD", reasons[0])
        self.assertIn("maximum error", reasons[1])

    def test_fit_tolerance_penalty_matches_quality_scaled_example(self):
        tolerance = CanonicalPsatFitPolicy().tolerance_for(
            quality=0.96,
            provenance=(),
            T_min=200.0,
            T_critical=500.0,
        )
        diagnostics = CanonicalPsatFitDiagnostics(
            sample_count=100,
            mard_percent=0.9,
            p95_absolute_relative_error_percent=2.0,
            max_absolute_relative_error_percent=4.5,
            max_anchor_log_residual=0.0,
            monotonic=True,
            reduced_condition_number=1.0,
        )

        self.assertAlmostEqual(tolerance.mard_threshold_percent, 0.8)
        self.assertAlmostEqual(tolerance.max_error_threshold_percent, 4.0)
        mard_excess, max_error_excess = tolerance.excess_fractions(
            diagnostics
        )
        self.assertAlmostEqual(mard_excess, 0.001)
        self.assertAlmostEqual(max_error_excess, 0.005)
        self.assertAlmostEqual(tolerance.quality_penalty(diagnostics), 0.015)
        self.assertIn("quality penalty=0.015", tolerance.warning(
            diagnostics,
            original_quality=0.96,
        ))

    def test_fitter_retries_inverse_powers_when_error_thresholds_are_exceeded(self):
        T_min = 180.0
        T_boiling = 300.0
        T_critical = 500.0
        P_critical_bar = 20.0
        H = -0.05
        boiling_ln_pressure = math.log(1.01325)
        critical_ln_pressure = math.log(P_critical_bar)
        boiling_inverse = (T_boiling / T_critical) ** -7 - 1.0
        B = (
            boiling_ln_pressure
            - critical_ln_pressure
            - H * boiling_inverse
        ) / (1.0 / T_boiling - 1.0 / T_critical)
        A = critical_ln_pressure - B / T_critical

        def source_ln_pressure(T):
            return A + B / T + H * ((T / T_critical) ** -7 - 1.0)

        samples = tuple(
            PsatSample(
                temperature=T_min + (T_critical - T_min) * index / 100.0,
                ln_pressure=source_ln_pressure(
                    T_min + (T_critical - T_min) * index / 100.0
                ),
            )
            for index in range(101)
        )

        curve = CanonicalPsatFitter().fit(
            samples,
            T_min=T_min,
            T_critical=T_critical,
            P_critical_bar=P_critical_bar,
            T_boiling=T_boiling,
        )

        self.assertEqual(curve.inverse_power, -7)
        self.assertEqual(curve.form, CanonicalPsatForm.AH)
        self.assertEqual(curve.metadata["attempted_forms"], ("A-F", "A-G", "A-H"))
        self.assertEqual(curve.metadata["attempted_inverse_powers"], (-3, -5, -7))
        self.assertLess(curve.diagnostics.max_absolute_relative_error_percent, 1.0)
        self.assertLess(curve.diagnostics.mard_percent, 0.2)

    def test_fitter_penalizes_remaining_high_error_and_warns(self):
        T_min = 200.0
        T_critical = 500.0
        samples = tuple(
            PsatSample(
                temperature=T_min + (T_critical - T_min) * index / 80.0,
                ln_pressure=0.01 * (
                    T_min + (T_critical - T_min) * index / 80.0
                )
                + 0.03 * math.sin(index * math.pi / 4.0),
            )
            for index in range(81)
        )

        curve = CanonicalPsatFitter().fit(
            samples,
            T_min=T_min,
            T_critical=T_critical,
            P_critical_bar=math.exp(0.01 * T_critical),
        )

        self.assertGreater(curve.metadata["fit_quality_penalty"], 0.0)
        self.assertLess(curve.quality, 1.0)
        self.assertTrue(curve.metadata["fit_warnings"])
        self.assertIn(
            "quality-scaled error thresholds",
            curve.metadata["fit_warnings"][0],
        )


if __name__ == "__main__":
    unittest.main()
