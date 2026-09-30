"""Component-aware adapter for canonical vapor-pressure construction."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Optional

from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..pressure_standards import NORMAL_BOILING_PRESSURE_BAR
else:
    from pressure_standards import NORMAL_BOILING_PRESSURE_BAR

from .common import (
    CACHED_NIST_ANTOINE_PROFILE,
    DirectPsatAdmissionDecision,
    DirectPsatConfidenceProfile,
    HIGH_QUALITY_ANTOINE_PROFILE,
    PropertyResolutionResult,
    R,
    STANDARD_DIRECT_TABLE_PROFILE,
    admit_direct_psat_segment,
)
from .coolprop import (
    COOLPROP_PROPERTY_QUALITY,
    coolprop_props_si,
    coolprop_reference_for,
    coolprop_saturation_pressure,
)
from .runtime_cache import (
    LEGACY_PROPERTY_CACHE_DIR,
    SQLiteJSONCache,
    runtime_cache_path_for_legacy_directory,
)
from .vapor_pressure_canonical import (
    BoundaryConditionedPsatEvaluation,
    BoundaryConditionedPsatSegment,
    CanonicalPsatForm,
    CanonicalPsatCurve,
    PsatAssemblyAnchorRequirement,
    PsatBoundaryConditions,
    PsatBoundaryRequirement,
    PsatCanonicalizationError,
    PsatDerivativeBasis,
    PsatDerivativeRequirement,
    PsatEndpoint,
    PsatHandoffRequirement,
    PsatPriority,
    PsatSegment,
    PsatSegmentProvenance,
    PsatSegmentType,
    make_c1_bridge_segment,
)


DEFAULT_PSAT_MINIMUM_PRESSURE_BAR = 1.0e-3
PERRY_2_8_PSAT_QUALITY = 0.98
AMBROSE_WALTON_QUALITY_FACTOR = 0.91
LOWER_AMBROSE_WALTON_QUALITY_FACTOR = 0.95
LOWER_AMBROSE_WALTON_ANCHORED_QUALITY_FACTOR = 0.94
MIDDLE_LOG_T_HERMITE_QUALITY_FACTOR = 0.97
NO_HARD_AW_UPPER_QUALITY_FACTOR = 0.91
NO_HARD_TB_EFFECTIVE_UPPER_QUALITY_FACTOR = 0.90
NO_HARD_AW_MIDDLE_QUALITY_FACTOR = 0.88
NO_HARD_AW_LOW_QUALITY_FACTOR = 0.85
NO_HARD_AW_DEEPEST_QUALITY_FACTOR = 0.80
NANNOOLAL_PSAT_QUALITY_FACTOR = 0.85
NANNOOLAL_DEEP_PSAT_QUALITY_FACTOR = 0.75
NANNOOLAL_UPPER_AW_EXTRA_QUALITY_FACTOR = 0.95
LOW_QUALITY_CRITICAL_PR_QUALITY_FACTOR = 0.97
LOWER_AW_SPAN_PENALTY_PER_DECADE = 0.03
LOWER_AW_MINIMUM_SPAN_QUALITY_FACTOR = 0.85
HVAP_HIGH_QUALITY_THRESHOLD = 0.95
HVAP_MEDIUM_QUALITY_THRESHOLD = 0.90
HVAP_MINIMUM_COMPLETION_QUALITY = 0.80
HVAP_HIGH_QUALITY_SWITCH_PRESSURE_BAR = 0.25
HVAP_MEDIUM_QUALITY_SWITCH_PRESSURE_BAR = 0.10
HVAP_LOW_QUALITY_SWITCH_PRESSURE_BAR = 0.05
DEEP_CLAPEYRON_QUALITY_FACTOR = 0.95
DIMER_CLAPEYRON_QUALITY_FACTOR = 0.90
DEEP_AW_FALLBACK_QUALITY_FACTOR = 0.90
DIMER_ENTHALPY_J_MOL = -60500.0
DIMER_SUPPRESSED_ENTROPY_J_MOL_K = -1000.0
AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY = 0.90
AMBROSE_WALTON_PREFERRED_PROPERTY_QUALITY = 0.93
AMBROSE_WALTON_HARD_BOUNDARY_MINIMUM_PROPERTY_QUALITY = 0.80
PSAT_MINIMUM_BOILING_POINT_QUALITY = 0.90
AMBROSE_WALTON_MINIMUM_REDUCED_TEMPERATURE = 0.70
ANTOINE_CACHE_DIR = Path(__file__).resolve().parent.parent / ".property_cache"
TRUSTED_OTHER_ANTOINE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "trusted_other_antoine.json"
)


class PsatAdapterError(ValueError):
    """Raised when component data cannot satisfy an adapter contract."""


@dataclass(frozen=True)
class PsatDomainSelection:
    """Selected canonical Psat lower bound and its provenance."""

    T_min: float
    basis: str
    source: str
    method: str
    quality: Optional[float] = None
    notes: str = ""
    pressure_bar: Optional[float] = None
    rejected_candidates: tuple[str, ...] = ()

    def canonicalizer_input(self) -> dict[str, float]:
        """Return the domain argument consumed by ``PsatCanonicalizer``."""
        return {"T_min": self.T_min}


@dataclass(frozen=True)
class PsatPhaseChangeAnchorSelection:
    """Boiling or sublimation anchor selected from component phase points."""

    T_boiling: Optional[float]
    boiling_quality: float
    uses_sublimation_anchor: bool = False
    triple_anchor: Optional[PsatEndpoint] = None


@dataclass(frozen=True)
class PsatCanonicalizationInputs:
    """Provider contributions ready for ``PsatCanonicalizer.canonicalize``."""

    segments: tuple[PsatSegment, ...] = ()
    relations: tuple[BoundaryConditionedPsatSegment, ...] = ()
    additional_anchors: tuple[PsatEndpoint, ...] = ()
    canonical_override: Optional[CanonicalPsatCurve] = None
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def merged(self, other: "PsatCanonicalizationInputs") -> "PsatCanonicalizationInputs":
        """Combine provider layers while preserving their insertion order."""
        if self.canonical_override is not None and other.canonical_override is not None:
            raise PsatAdapterError("Multiple direct canonical Psat overrides were supplied")
        metadata = dict(self.metadata)
        metadata.update(other.metadata)
        warnings = tuple(dict.fromkeys((*self.warnings, *other.warnings)))
        return PsatCanonicalizationInputs(
            segments=self.segments + other.segments,
            relations=self.relations + other.relations,
            additional_anchors=self.additional_anchors + other.additional_anchors,
            canonical_override=self.canonical_override or other.canonical_override,
            warnings=warnings,
            metadata=metadata,
        )

    def canonicalizer_arguments(self) -> dict[str, Any]:
        """Return keyword arguments accepted by ``canonicalize``."""
        metadata = dict(self.metadata)
        if self.warnings:
            metadata["input_warnings"] = self.warnings
        return {
            "segments": self.segments,
            "relations": self.relations,
            "additional_anchors": self.additional_anchors,
            "canonical_override": self.canonical_override,
            "metadata": metadata,
        }


@dataclass(frozen=True)
class _FrozenHvapPlan:
    evaluator_J_mol: Callable[[float], float] = field(
        repr=False,
        compare=False,
    )
    quality: float
    source: str
    method: str
    relation_class: str
    T_min: float
    T_max: float
    switch_pressure_bar: float
    sample_count: int


TsatAtPressure = Callable[[Any, float], Any]
PsatAtTemperature = Callable[[Any, float], Any]
HvapAtTemperature = Callable[[Any, float], Any]
PsatInputMethod = Callable[..., PsatCanonicalizationInputs]


class PsatCanonicalizationAdapter:
    """Translate component/resolver data into canonicalizer inputs.

    The provider-neutral canonicalizer never reads component objects. This
    adapter owns that boundary and is the intended home for future source
    segment collection and boundary-conditioned completion relations.
    """

    def __init__(
        self,
        component: Any,
        *,
        psat_at_temperature: Optional[PsatAtTemperature] = None,
        tsat_at_pressure: Optional[TsatAtPressure] = None,
        hvap_at_temperature: Optional[HvapAtTemperature] = None,
        allow_online_hvap: bool = False,
        minimum_pressure_bar: float = DEFAULT_PSAT_MINIMUM_PRESSURE_BAR,
        input_methods: Optional[tuple[PsatInputMethod, ...]] = None,
    ):
        self.component = component
        self.psat_at_temperature = psat_at_temperature
        self.tsat_at_pressure = tsat_at_pressure
        self.hvap_at_temperature = hvap_at_temperature
        self.allow_online_hvap = bool(allow_online_hvap)
        self._property_resolver = None
        self._boiling_point_candidates: dict[
            tuple[Optional[float], bool],
            Optional[PropertyResolutionResult],
        ] = {}
        self._no_hard_boiling_candidate = None
        self._domain_selection: Optional[PsatDomainSelection] = None
        self._frozen_hvap_plans: dict[
            tuple[float, float, float],
            Optional[_FrozenHvapPlan],
        ] = {}
        self.minimum_pressure_bar = _positive_finite(
            minimum_pressure_bar,
            "minimum_pressure_bar",
        )
        self.input_methods = tuple(
            DEFAULT_PSAT_INPUT_METHODS if input_methods is None else input_methods
        )
        if any(not callable(method) for method in self.input_methods):
            raise PsatAdapterError("Psat input methods must be callable")

    def resolve_domain(
        self,
        *,
        T_critical: Optional[float] = None,
    ) -> PsatDomainSelection:
        """Select ``Tt``, then ``Tm``, then the pressure-floor fallback."""
        critical_candidate = (
            self.component_value("Tc") if T_critical is None else T_critical
        )
        if isinstance(critical_candidate, PropertyResolutionResult):
            critical_candidate = critical_candidate.value
        T_critical = _positive_finite(critical_candidate, "T_critical")
        rejected: list[str] = []
        has_pfd_psat_override = _pfd_psat_correlation(
            self.component
        ) is not None

        for field_name, basis in (
            ("Tt", "triple_point"),
            ("Tm", "melting_point"),
        ):
            if (
                has_pfd_psat_override
                and str(
                    self.property_source(field_name).get("method") or ""
                ).lower() != "pfd_component_override"
            ):
                rejected.append(
                    f"{field_name} ignored because it is not part of the "
                    "PFD Psat override"
                )
                continue
            raw_value = self.component_value(field_name)
            if raw_value is None:
                rejected.append(f"{field_name} unavailable")
                continue
            candidate = self._candidate_from_field(field_name, raw_value)
            temperature = _valid_subcritical_temperature(
                candidate.value,
                T_critical,
            )
            if temperature is None:
                rejected.append(
                    f"{field_name} invalid for 0 < T < Tc: {candidate.value!r}"
                )
                continue
            pressure_bar = None
            pressure_note = ""
            if field_name == "Tt":
                raw_pressure = self.component_value("Pt")
                pressure_is_compatible = (
                    not has_pfd_psat_override
                    or str(
                        self.property_source("Pt").get("method") or ""
                    ).lower() == "pfd_component_override"
                )
                if raw_pressure is not None and pressure_is_compatible:
                    pressure_candidate = self._candidate_from_field(
                        "Pt",
                        raw_pressure,
                    )
                    pressure_bar = _positive_resolution_value(
                        pressure_candidate
                    )
                    if pressure_bar is not None:
                        pressure_note = (
                            f"Pt={pressure_bar:g} bar from "
                            f"{pressure_candidate.source}/"
                            f"{pressure_candidate.method}"
                        )
            notes = "; ".join(
                part for part in (candidate.notes, pressure_note)
                if part
            )
            selection = PsatDomainSelection(
                T_min=temperature,
                basis=basis,
                source=candidate.source,
                method=candidate.method,
                quality=_optional_quality(candidate.quality),
                notes=notes,
                pressure_bar=pressure_bar,
                rejected_candidates=tuple(rejected),
            )
            self._domain_selection = selection
            return selection

        if self.tsat_at_pressure is None:
            rejected.append("Tsat pressure fallback unavailable")
            raise PsatAdapterError("; ".join(rejected))

        try:
            raw_tsat = self.tsat_at_pressure(
                self.component,
                self.minimum_pressure_bar,
            )
        except Exception as error:
            rejected.append(
                f"Tsat({self.minimum_pressure_bar:g} bar) failed: {error}"
            )
            raise PsatAdapterError("; ".join(rejected)) from error

        candidate = _candidate_from_tsat(raw_tsat)
        temperature = _valid_subcritical_temperature(candidate.value, T_critical)
        if temperature is None:
            rejected.append(
                f"Tsat({self.minimum_pressure_bar:g} bar) invalid for "
                f"0 < T < Tc: {candidate.value!r}"
            )
            raise PsatAdapterError("; ".join(rejected))

        pressure_note = (
            f"lower Psat pressure floor {self.minimum_pressure_bar:g} bar"
        )
        notes = "; ".join(
            part for part in (candidate.notes, pressure_note) if part
        )
        selection = PsatDomainSelection(
            T_min=temperature,
            basis="pressure_floor",
            source=candidate.source,
            method=candidate.method,
            quality=_optional_quality(candidate.quality),
            notes=notes,
            pressure_bar=self.minimum_pressure_bar,
            rejected_candidates=tuple(rejected),
        )
        self._domain_selection = selection
        return selection

    @property
    def completion_pressure_floor(self) -> Optional[float]:
        """Pressure limit for completion, absent on a physical domain."""
        if (
            self._domain_selection is not None
            and self._domain_selection.basis != "pressure_floor"
        ):
            return None
        return self.minimum_pressure_bar

    def collect_inputs(
        self,
        *,
        T_min: Optional[float] = None,
        T_critical: Optional[float] = None,
        P_critical_bar: Optional[float] = None,
        T_boiling: Optional[float] = None,
    ) -> PsatCanonicalizationInputs:
        """Collect every configured provider contribution in chain order."""
        qualified_boiling = _qualified_boiling_point_candidate(
            self,
            explicit_value=T_boiling,
        )
        T_boiling = (
            _positive_resolution_value(qualified_boiling)
            if qualified_boiling is not None
            else None
        )
        inputs = PsatCanonicalizationInputs()
        for method in self.input_methods:
            if method is _collect_no_hard_fallback_inputs and (
                inputs.canonical_override is not None or inputs.segments
            ):
                continue
            contribution = method(
                self,
                T_min=T_min,
                T_critical=T_critical,
                P_critical_bar=P_critical_bar,
                T_boiling=T_boiling,
            )
            if not isinstance(contribution, PsatCanonicalizationInputs):
                raise PsatAdapterError(
                    f"Psat input method {getattr(method, '__name__', method)!r} "
                    "returned an invalid contribution"
                )
            inputs = inputs.merged(contribution)
            if inputs.canonical_override is not None:
                break
        return inputs

    def component_value(self, field_name: str) -> Any:
        """Read a component field from either a mapping or an object."""
        if self.component is None:
            return None
        if isinstance(self.component, Mapping):
            return self.component.get(field_name)
        return getattr(self.component, field_name, None)

    def property_source(self, field_name: str) -> Mapping[str, Any]:
        """Return stored per-property provenance when the component has it."""
        sources = self.component_value("property_sources")
        if not isinstance(sources, Mapping):
            return {}
        metadata = sources.get(field_name)
        return metadata if isinstance(metadata, Mapping) else {}

    def select_phase_change_anchor(
        self,
        *,
        T_boiling: Optional[float],
        boiling_quality: float,
    ) -> PsatPhaseChangeAnchorSelection:
        """Replace a sub-triple-point boiling candidate with a Tt/Pt anchor."""
        boiling_candidate = _property_candidate(
            self,
            "Tb",
            explicit_value=T_boiling,
        )
        if (
            T_boiling is not None
            and (
                boiling_quality < PSAT_MINIMUM_BOILING_POINT_QUALITY
                or _resolution_is_estimated(boiling_candidate)
            )
        ):
            T_boiling = None
            boiling_quality = 0.0
        has_pfd_psat_override = _pfd_psat_correlation(
            self.component
        ) is not None
        triple_is_compatible = (
            not has_pfd_psat_override
            or str(
                self.property_source("Tt").get("method") or ""
            ).lower() == "pfd_component_override"
        )
        triple_candidate = _property_candidate(self, "Tt")
        triple_is_qualified = (
            triple_candidate is not None
            and _optional_quality(triple_candidate.quality) is not None
            and float(triple_candidate.quality)
            >= PSAT_MINIMUM_BOILING_POINT_QUALITY
            and not _resolution_is_estimated(triple_candidate)
        )
        T_triple = (
            _component_numeric_value(self.component, "Tt")
            if triple_is_compatible and triple_is_qualified
            else None
        )
        pressure_is_compatible = (
            not has_pfd_psat_override
            or str(
                self.property_source("Pt").get("method") or ""
            ).lower() == "pfd_component_override"
        )
        pressure_candidate = _property_candidate(self, "Pt")
        pressure_is_qualified = (
            pressure_candidate is not None
            and _optional_quality(pressure_candidate.quality) is not None
            and float(pressure_candidate.quality)
            >= PSAT_MINIMUM_BOILING_POINT_QUALITY
            and not _resolution_is_estimated(pressure_candidate)
        )
        P_triple_bar = (
            _component_numeric_value(self.component, "Pt")
            if pressure_is_compatible and pressure_is_qualified
            else None
        )
        replaces_subtriple_boiling = (
            T_boiling is not None
            and T_triple is not None
            and T_boiling < T_triple
        )
        anchors_missing_normal_boiling = (
            T_boiling is None
            and T_triple is not None
            and P_triple_bar is not None
            and P_triple_bar >= NORMAL_BOILING_PRESSURE_BAR - 1.0e-9
        )
        if not (
            replaces_subtriple_boiling
            or anchors_missing_normal_boiling
        ):
            return PsatPhaseChangeAnchorSelection(
                T_boiling=T_boiling,
                boiling_quality=boiling_quality,
            )

        triple_anchor = None
        if P_triple_bar is not None and P_triple_bar > 0.0:
            qualities = tuple(
                quality
                for quality in (
                    _optional_quality(
                        self.property_source(field_name).get("quality")
                    )
                    for field_name in ("Tt", "Pt")
                )
                if quality is not None
            )
            context = {
                key: value
                for key in ("symbol", "name", "CAS")
                if (value := self.component_value(key)) not in (None, "")
            }
            context["anchor_name"] = "Tt"
            triple_anchor = PsatEndpoint.fixed_pressure_point(
                T_triple,
                P_triple_bar,
                source="fixed_anchor",
                method="triple_point",
                quality=min(qualities, default=0.0),
                context=context,
            )

        return PsatPhaseChangeAnchorSelection(
            T_boiling=None,
            boiling_quality=0.0,
            uses_sublimation_anchor=True,
            triple_anchor=triple_anchor,
        )

    def _candidate_from_field(
        self,
        field_name: str,
        raw_value: Any,
    ) -> PropertyResolutionResult:
        if isinstance(raw_value, PropertyResolutionResult):
            return raw_value
        metadata = self.property_source(field_name)
        return PropertyResolutionResult(
            value=raw_value,
            source=str(metadata.get("source") or "component"),
            method=str(metadata.get("method") or f"{field_name}_field"),
            quality=metadata.get("quality"),
            notes=str(metadata.get("notes") or ""),
        )


def _collect_pfd_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_min: Optional[float] = None,
    T_critical: Optional[float] = None,
    P_critical_bar: Optional[float] = None,
    T_boiling: Optional[float] = None,
) -> PsatCanonicalizationInputs:
    """Translate a PFD Psat override into the highest-priority source layer."""
    correlation = _pfd_psat_correlation(adapter.component)
    if correlation is None:
        return _collect_pfd_antoine_inputs(
            adapter,
            T_min=T_min,
            T_critical=T_critical,
            T_boiling=T_boiling,
        )

    target_T_min = _positive_finite(T_min, "T_min")
    target_T_critical = _positive_finite(T_critical, "T_critical")
    target_P_critical = _positive_finite(P_critical_bar, "P_critical_bar")
    if target_T_min >= target_T_critical:
        raise PsatAdapterError("PFD Psat target requires T_min < T_critical")
    target_T_boiling = None
    if T_boiling is not None:
        target_T_boiling = _positive_finite(T_boiling, "T_boiling")
        if not target_T_min <= target_T_boiling < target_T_critical:
            raise PsatAdapterError("PFD Psat target boiling point is outside the domain")

    equation = str(correlation.get("equation") or "").strip().lower()
    if not equation:
        raise PsatAdapterError("PFD Psat correlation requires an equation")
    coefficients = _required_coefficients(correlation, equation)
    declared_T_min = _correlation_float(
        correlation,
        "Tmin_K",
        default=target_T_min,
    )
    declared_T_max = _correlation_float(
        correlation,
        "Tmax_K",
        "Tc_K",
        default=target_T_critical,
    )
    if declared_T_min <= 0.0 or declared_T_max <= declared_T_min:
        raise PsatAdapterError("PFD Psat correlation requires 0 < Tmin_K < Tmax_K")

    ln_pressure, derivative = _psat_correlation_functions(
        equation,
        coefficients,
        correlation,
        adapter.component,
    )
    context = _component_context(adapter.component)
    metadata = {
        "provider": "pfd",
        "equation": equation,
        "property_dependencies": _psat_correlation_property_dependencies(
            equation,
            correlation,
        ),
        "canonical_form": (
            _CANONICAL_PSAT_EQUATIONS[equation].value
            if equation in _CANONICAL_PSAT_EQUATIONS
            else None
        ),
        "declared_T_min": declared_T_min,
        "declared_T_max": declared_T_max,
    }
    pfd_admission = admit_direct_psat_segment(
        priority=int(PsatPriority.PFD_OVERRIDE),
        intrinsic_quality=1.0,
        confidence_profile=None,
        qualified_Tb=None,
        pressure_at_temperature=lambda _temperature: None,
        T_min=declared_T_min,
        T_max=declared_T_max,
        source_label="PFD Psat override",
        exemption_priority=int(PsatPriority.PERRY_2_8),
    )
    metadata.update(_direct_psat_admission_metadata(pfd_admission, None))

    if equation in _CANONICAL_PSAT_EQUATIONS and (
        declared_T_min <= target_T_min
        and declared_T_max >= target_T_critical
    ):
        _validate_pfd_critical_metadata(
            correlation,
            target_T_critical,
            target_P_critical,
            target_T_boiling,
        )
        curve = CanonicalPsatCurve(
            A=coefficients["A"],
            B=coefficients["B"],
            C=coefficients["C"],
            D=coefficients["D"],
            E=coefficients["E"],
            F=coefficients["F"],
            G=coefficients["G"],
            T_min=target_T_min,
            T_critical=target_T_critical,
            P_critical_bar=target_P_critical,
            H=coefficients["H"],
            inverse_power=_canonical_inverse_power(
                correlation,
                coefficients,
                _CANONICAL_PSAT_EQUATIONS[equation],
            ),
            form=_CANONICAL_PSAT_EQUATIONS[equation],
            T_boiling=target_T_boiling,
            quality=1.0,
            provenance=(PsatSegmentProvenance(
                source="provided",
                method="pfd_canonical_psat",
                segment_type=PsatSegmentType.CANONICAL_OVERRIDE.value,
                priority=int(PsatPriority.PFD_OVERRIDE),
                quality=1.0,
                T_min=target_T_min,
                T_max=target_T_critical,
                context=context,
                metadata=metadata,
            ),),
            metadata=metadata,
        )
        return PsatCanonicalizationInputs(
            canonical_override=curve,
            metadata={"pfd_psat": "canonical_override"},
        )

    segment = PsatSegment(
        source="provided",
        method=f"pfd_psat_{equation}",
        segment_type=PsatSegmentType.PINNED,
        priority=int(PsatPriority.PFD_OVERRIDE),
        T_min=declared_T_min,
        T_max=declared_T_max,
        ln_pressure_function=ln_pressure,
        derivative_function=derivative,
        quality=1.0,
        P_min_bar=_optional_positive_correlation_float(correlation, "Pmin_bar"),
        P_max_bar=_optional_positive_correlation_float(correlation, "Pmax_bar"),
        context=context,
        metadata=metadata,
    )
    return PsatCanonicalizationInputs(
        segments=(segment,),
        metadata={"pfd_psat": "pinned_segment"},
    )


def _collect_pfd_antoine_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_min: Optional[float],
    T_critical: Optional[float],
    T_boiling: Optional[float] = None,
) -> PsatCanonicalizationInputs:
    overridden_fields, aggregate_override = _pfd_antoine_override_state(adapter)
    if not overridden_fields and not aggregate_override:
        return PsatCanonicalizationInputs()
    required_fields = frozenset((
        "antoine_A",
        "antoine_B",
        "antoine_C",
        "antoine_Tmin",
        "antoine_Tmax",
    ))
    if not aggregate_override and overridden_fields != required_fields:
        missing_provenance = ", ".join(sorted(required_fields - overridden_fields))
        raise PsatAdapterError(
            "Parsed PFD Antoine invariant has partial override provenance; "
            f"missing PFD override markers for {missing_provenance}. "
            "PFDParser should have rejected this override"
        )
    values = [
        _required_pfd_component_number(adapter, field_name)
        for field_name in ("antoine_A", "antoine_B", "antoine_C")
    ]
    declared_T_min = _required_pfd_component_number(adapter, "antoine_Tmin")
    declared_T_max = _required_pfd_component_number(adapter, "antoine_Tmax")
    ln_pressure, derivative, P_min_bar, P_max_bar = _antoine_functions(
        *values,
        declared_T_min,
        declared_T_max,
        label="PFD Antoine",
    )
    segment = PsatSegment(
        source="provided",
        method="pfd_antoine",
        segment_type=PsatSegmentType.PINNED,
        priority=int(PsatPriority.PFD_OVERRIDE),
        T_min=declared_T_min,
        T_max=declared_T_max,
        ln_pressure_function=ln_pressure,
        derivative_function=derivative,
        quality=1.0,
        P_min_bar=P_min_bar,
        P_max_bar=P_max_bar,
        context=_component_context(adapter.component),
        metadata={
            "provider": "pfd",
            "equation": "antoine",
            "property_dependencies": (),
            "pressure_units": "bar",
            "temperature_basis": "Celsius inside Antoine denominator",
        },
    )
    segment, warning = _admit_direct_psat_segment(
        adapter,
        segment,
        T_boiling=T_boiling,
        confidence_profile=None,
        source_label="PFD Antoine",
    )
    if segment is None:
        return PsatCanonicalizationInputs(warnings=(str(warning),))
    return PsatCanonicalizationInputs(
        segments=(segment,),
        metadata={"pfd_psat": "antoine_segment"},
    )


def _collect_coolprop_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_boiling: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    """Return a full-range CoolProp Psat segment when the component is supported."""
    reference = _coolprop_reference(adapter.component)
    if reference is None:
        return PsatCanonicalizationInputs()

    T_triple = coolprop_props_si("Ttriple", reference)
    T_critical = coolprop_props_si("Tcrit", reference)
    P_triple_reported_pa = coolprop_props_si("ptriple", reference)
    P_critical_pa = coolprop_props_si("pcrit", reference)
    if (
        T_triple is None
        or T_critical is None
        or P_triple_reported_pa is None
        or P_critical_pa is None
        or not (0.0 < T_triple < T_critical)
        or P_triple_reported_pa <= 0.0
        or P_critical_pa <= 0.0
    ):
        return PsatCanonicalizationInputs()

    def ln_pressure(temperature: float) -> float:
        pressure_pa = coolprop_saturation_pressure(reference, temperature)
        if pressure_pa is None:
            raise PsatAdapterError(
                f"CoolProp {reference.qualified_name} Psat failed at "
                f"{temperature:g} K"
            )
        return math.log(pressure_pa / 100000.0)

    P_triple_evaluated_pa = coolprop_saturation_pressure(reference, T_triple)
    P_critical_evaluated_pa = coolprop_saturation_pressure(reference, T_critical)
    if (
        P_triple_evaluated_pa is None
        or P_critical_evaluated_pa is None
        or P_triple_evaluated_pa <= 0.0
        or P_critical_evaluated_pa <= 0.0
    ):
        return PsatCanonicalizationInputs()

    context = {
        **_component_context(adapter.component),
        "coolprop_fluid": reference.fluid,
        "coolprop_backend": reference.backend,
    }
    segment = PsatSegment(
        source="local",
        method=f"{reference.method_prefix}_psat",
        segment_type=PsatSegmentType.PINNED,
        priority=int(PsatPriority.COOLPROP_HEOS),
        T_min=float(T_triple),
        T_max=float(T_critical),
        ln_pressure_function=ln_pressure,
        quality=COOLPROP_PROPERTY_QUALITY,
        P_min_bar=float(P_triple_evaluated_pa) / 100000.0,
        P_max_bar=float(P_critical_evaluated_pa) / 100000.0,
        context=context,
        metadata={
            "provider": "coolprop",
            "property_dependencies": (),
            "backend": reference.backend,
            "fluid": reference.fluid,
            "qualified_name": reference.qualified_name,
            "reported_triple_pressure_bar": (
                float(P_triple_reported_pa) / 100000.0
            ),
            "evaluated_triple_pressure_bar": (
                float(P_triple_evaluated_pa) / 100000.0
            ),
            "reported_critical_pressure_bar": (
                float(P_critical_pa) / 100000.0
            ),
        },
    )
    segment, warning = _admit_direct_psat_segment(
        adapter,
        segment,
        T_boiling=T_boiling,
        confidence_profile=None,
        source_label=f"CoolProp {reference.qualified_name}",
    )
    if segment is None:
        return PsatCanonicalizationInputs(warnings=(str(warning),))
    return PsatCanonicalizationInputs(
        segments=(segment,),
        metadata={"coolprop_psat": reference.qualified_name},
    )


def _collect_local_correlation_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_boiling: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    """Translate a curated chemicals.json Psat correlation into a pinned segment."""
    correlation = _local_psat_correlation(adapter.component)
    if correlation is None:
        return PsatCanonicalizationInputs()

    equation = str(correlation.get("equation") or "").strip().lower()
    if not equation:
        raise PsatAdapterError("Local Psat correlation requires an equation")
    coefficients = _required_coefficients(correlation, equation)
    declared_T_min = _correlation_float(correlation, "Tmin_K")
    declared_T_max = _correlation_float(correlation, "Tmax_K")
    if declared_T_min <= 0.0 or declared_T_max <= declared_T_min:
        raise PsatAdapterError(
            "Local Psat correlation requires 0 < Tmin_K < Tmax_K"
        )

    reducing_Tc_K = None
    reducing_Pc_bar = None
    if equation in {"reduced_vapor_pressure", "psat_mercury"}:
        reducing_Tc_K = _correlation_float(correlation, "Tc_K")
        reducing_Pc_bar = _critical_pressure_bar(
            correlation,
            adapter.component,
            require_explicit=True,
        )
        if declared_T_max > reducing_Tc_K:
            raise PsatAdapterError(
                "Local reduced Psat correlation cannot extend above its reducing Tc_K"
            )

    ln_pressure, derivative = _psat_correlation_functions(
        equation,
        coefficients,
        correlation,
        adapter.component,
    )
    quality = _correlation_quality(correlation, default=0.96)
    citation = str(
        correlation.get("source")
        or correlation.get("selected_model")
        or "chemicals.json"
    )
    metadata = {
        "provider": "chemicals.json",
        "equation": equation,
        "property_dependencies": _psat_correlation_property_dependencies(
            equation,
            correlation,
        ),
        "canonical_form": (
            _CANONICAL_PSAT_EQUATIONS[equation].value
            if equation in _CANONICAL_PSAT_EQUATIONS
            else None
        ),
        "declared_T_min": declared_T_min,
        "declared_T_max": declared_T_max,
        "selected_model": correlation.get("selected_model"),
        "quality_note": correlation.get("quality_note"),
        "citation": citation,
        "reducing_Tc_K": reducing_Tc_K,
        "reducing_Pc_bar": reducing_Pc_bar,
    }
    segment = PsatSegment(
        source="local",
        method=f"local_psat_{equation}",
        segment_type=PsatSegmentType.PINNED,
        priority=int(PsatPriority.LOCAL_CORRELATION),
        T_min=declared_T_min,
        T_max=declared_T_max,
        ln_pressure_function=ln_pressure,
        derivative_function=derivative,
        quality=quality,
        P_min_bar=_optional_positive_correlation_float(correlation, "Pmin_bar"),
        P_max_bar=_optional_positive_correlation_float(correlation, "Pmax_bar"),
        context=_component_context(adapter.component),
        metadata=metadata,
    )
    segment, warning = _admit_direct_psat_segment(
        adapter,
        segment,
        T_boiling=T_boiling,
        confidence_profile=None,
        source_label=f"local direct Psat {citation}",
    )
    if segment is None:
        return PsatCanonicalizationInputs(warnings=(str(warning),))
    return PsatCanonicalizationInputs(
        segments=(segment,),
        metadata={"local_psat_correlation": citation},
    )


def _collect_curated_antoine_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_boiling: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    """Translate a curated chemicals.json Antoine row into a pinned segment."""
    source_metadata = _curated_antoine_source_metadata(adapter)
    if source_metadata is None:
        return PsatCanonicalizationInputs()
    coefficients = _curated_antoine_coefficients(adapter)
    if coefficients is None:
        return PsatCanonicalizationInputs()
    A, B, C, T_min, T_max = coefficients
    ln_pressure, derivative, P_min_bar, P_max_bar = _antoine_functions(
        A,
        B,
        C,
        T_min,
        T_max,
        label="Curated Antoine",
    )

    quality = _property_source_quality(source_metadata, default=0.98)
    citation = str(
        adapter.component_value("antoine_source")
        or source_metadata.get("source")
        or "curated chemicals.json Antoine"
    )
    handoff_requirement = PsatHandoffRequirement.NONE
    segment = PsatSegment(
        source="local",
        method="local_antoine",
        segment_type=PsatSegmentType.PINNED,
        priority=int(PsatPriority.LOCAL_ANTOINE),
        T_min=T_min,
        T_max=T_max,
        ln_pressure_function=ln_pressure,
        derivative_function=derivative,
        quality=quality,
        P_min_bar=P_min_bar,
        P_max_bar=P_max_bar,
        context=_component_context(adapter.component),
        handoff_requirement=handoff_requirement,
        metadata={
            "provider": "chemicals.json",
            "equation": "antoine",
            "property_dependencies": (),
            "citation": citation,
            "pressure_units": "bar",
            "temperature_basis": "Celsius inside Antoine denominator",
        },
    )
    segment, warning = _admit_direct_psat_segment(
        adapter,
        segment,
        T_boiling=T_boiling,
        confidence_profile=None,
        source_label="curated chemicals.json Antoine",
    )
    if segment is None:
        return _discarded_input_warning(adapter, str(warning))
    return PsatCanonicalizationInputs(
        segments=(segment,),
        metadata={"curated_antoine": citation},
    )


def _collect_textbook_antoine_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_boiling: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    """Translate a Smith Appendix B Antoine row into the quality-550 layer."""
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..textbook_properties import get_textbook_property_library
        else:
            from textbook_properties import get_textbook_property_library
    except ImportError:
        return PsatCanonicalizationInputs()
    library = get_textbook_property_library()
    entry = None
    resolved_identifier = None
    for field_name in ("name", "symbol", "formula"):
        identifier = adapter.component_value(field_name)
        if identifier in (None, ""):
            continue
        entry = library.get(str(identifier))
        if entry:
            resolved_identifier = str(identifier)
            break
    if not entry or any(entry.get(key) is None for key in (
        "antoine_A",
        "antoine_B",
        "antoine_C",
        "antoine_Tmin",
        "antoine_Tmax",
    )):
        return PsatCanonicalizationInputs()
    A = float(entry["antoine_A"])
    B = float(entry["antoine_B"])
    C = float(entry["antoine_C"])
    T_min = float(entry["antoine_Tmin"])
    T_max = float(entry["antoine_Tmax"])
    citation = str(entry.get("source_table") or "Smith8 Appendix B")
    ln_pressure, derivative, P_min_bar, P_max_bar = _antoine_functions(
        A,
        B,
        C,
        T_min,
        T_max,
        label="Smith Appendix B Antoine",
    )
    segment = PsatSegment(
        source=citation,
        method="textbook_antoine",
        segment_type=PsatSegmentType.PINNED,
        priority=int(PsatPriority.HIGH_QUALITY_ANTOINE),
        T_min=T_min,
        T_max=T_max,
        ln_pressure_function=ln_pressure,
        derivative_function=derivative,
        quality=HIGH_QUALITY_ANTOINE_PROFILE.hard_tb_quality,
        P_min_bar=P_min_bar,
        P_max_bar=P_max_bar,
        context=_component_context(adapter.component),
        handoff_requirement=(
            PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP_IF_AVAILABLE
        ),
        metadata={
            "provider": "Smith Appendix B",
            "equation": "antoine",
            "property_dependencies": (),
            "citation": citation,
            "resolved_identifier": resolved_identifier,
            "pressure_units": "bar",
            "temperature_basis": "Celsius inside Antoine denominator",
        },
    )
    segment, warning = _admit_direct_psat_segment(
        adapter,
        segment,
        T_boiling=T_boiling,
        confidence_profile=HIGH_QUALITY_ANTOINE_PROFILE,
        source_label="Smith Appendix B Antoine",
    )
    if segment is None:
        return _discarded_input_warning(adapter, str(warning))
    return PsatCanonicalizationInputs(
        segments=(segment,),
        metadata={"textbook_antoine": citation},
    )


def _other_antoine_segment(
    adapter: PsatCanonicalizationAdapter,
    *,
    source: str,
    method: str,
    A: float,
    B: float,
    C: float,
    T_min: float,
    T_max: float,
    T_boiling: Optional[float],
    confidence_profile: DirectPsatConfidenceProfile,
    metadata: Mapping[str, Any],
    priority: int = int(PsatPriority.OTHER_ANTOINE),
) -> tuple[Optional[PsatSegment], Optional[str]]:
    try:
        ln_pressure, derivative, P_min_bar, P_max_bar = _antoine_functions(
            A,
            B,
            C,
            T_min,
            T_max,
            label=f"{source} Antoine",
        )
    except PsatAdapterError as error:
        return None, str(error)
    segment_metadata = dict(metadata)
    segment_metadata.update({
        "provider": source,
        "equation": "antoine",
        "property_dependencies": (),
        "pressure_units": "bar",
        "temperature_basis": "Celsius inside Antoine denominator",
    })
    segment = PsatSegment(
        source=source,
        method=method,
        segment_type=PsatSegmentType.PINNED,
        priority=int(priority),
        T_min=T_min,
        T_max=T_max,
        ln_pressure_function=ln_pressure,
        derivative_function=derivative,
        quality=confidence_profile.standalone_quality,
        P_min_bar=P_min_bar,
        P_max_bar=P_max_bar,
        context=_component_context(adapter.component),
        handoff_requirement=(
            PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP_IF_AVAILABLE
        ),
        metadata=segment_metadata,
    )
    return _admit_direct_psat_segment(
        adapter,
        segment,
        T_boiling=T_boiling,
        confidence_profile=confidence_profile,
        source_label=f"{source} Antoine row",
    )


def _collect_trusted_other_antoine_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_boiling: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    """Load normalized, source-attributed Antoine rows at priority 550."""
    if not TRUSTED_OTHER_ANTOINE_PATH.exists():
        return PsatCanonicalizationInputs()
    payload = json.loads(TRUSTED_OTHER_ANTOINE_PATH.read_text())

    identifiers = {
        re.sub(r"[^a-z0-9]+", "", str(value).lower())
        for field in ("CAS", "cas", "name", "symbol")
        if (value := adapter.component_value(field)) not in (None, "")
    }
    segments = []
    warnings = []
    for entry in payload.get("correlations", []):
        identity = entry.get("identity", {})
        entry_identifiers = {
            re.sub(r"[^a-z0-9]+", "", str(value).lower())
            for value in (
                identity.get("cas"),
                identity.get("name"),
                *identity.get("aliases", []),
            )
            if value not in (None, "")
        }
        if identifiers.isdisjoint(entry_identifiers):
            continue
        correlation = entry.get("correlation", {})
        if correlation.get("equation") != "antoine_log10_bar_degC":
            warnings.append(
                f"Unsupported trusted Antoine equation for {entry.get('record_id')}"
            )
            continue
        source = entry.get("source", {})
        citation = str(
            source.get("doi")
            or source.get("title")
            or entry.get("record_id")
            or "trusted literature Antoine"
        )
        segment, warning = _other_antoine_segment(
            adapter,
            source=citation,
            method="trusted_other_antoine",
            A=float(correlation["A"]),
            B=float(correlation["B"]),
            C=float(correlation["C"]),
            T_min=float(correlation["Tmin_K"]),
            T_max=float(correlation["Tmax_K"]),
            T_boiling=T_boiling,
            confidence_profile=HIGH_QUALITY_ANTOINE_PROFILE,
            metadata={
                "record_id": entry.get("record_id"),
                "data_file": "data/trusted_other_antoine.json",
                "citation": citation,
                "source": source,
                "provenance": entry.get("provenance", {}),
            },
            priority=int(PsatPriority.HIGH_QUALITY_ANTOINE),
        )
        if segment is not None:
            segments.append(segment)
        if warning:
            warnings.append(warning)
    return PsatCanonicalizationInputs(
        segments=_ordered_other_antoine_segments(segments, T_boiling),
        warnings=tuple(warnings),
        metadata=(
            {"trusted_other_antoine_row_count": len(segments)}
            if segments
            else {}
        ),
    )


def _ordered_other_antoine_segments(
    items: list[PsatSegment],
    T_boiling: Optional[float],
) -> tuple[PsatSegment, ...]:
    def distance_to_boiling(segment: PsatSegment) -> float:
        if T_boiling is None:
            return -segment.T_max
        if segment.T_min <= T_boiling <= segment.T_max:
            return 0.0
        return min(
            abs(segment.T_min - T_boiling),
            abs(segment.T_max - T_boiling),
        )

    return tuple(sorted(
        items,
        key=lambda segment: (
            segment.handoff_requirement is not PsatHandoffRequirement.NONE,
            distance_to_boiling(segment),
            -segment.T_max,
            segment.T_min,
        ),
    ))


def _collect_antoine_table_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_boiling: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..antoine_properties import get_antoine_table
        else:
            from antoine_properties import get_antoine_table
    except ImportError:
        return PsatCanonicalizationInputs()
    table = get_antoine_table()
    entries = []
    seen = set()
    for field_name in ("name", "symbol", "formula"):
        identifier = adapter.component_value(field_name)
        if identifier in (None, ""):
            continue
        for entry in table.entries(str(identifier)):
            key = (
                entry.record_id,
                entry.A,
                entry.B,
                entry.C,
                entry.T_min,
                entry.T_max,
            )
            if key not in seen:
                seen.add(key)
                entries.append(entry)
    segments = []
    warnings = []
    for entry in entries:
        segment, warning = _other_antoine_segment(
            adapter,
            source="data/antoine.txt",
            method="antoine_table",
            A=entry.A,
            B=entry.B,
            C=entry.C,
            T_min=entry.T_min,
            T_max=entry.T_max,
            T_boiling=T_boiling,
            confidence_profile=STANDARD_DIRECT_TABLE_PROFILE,
            metadata={
                "record_id": entry.record_id,
                "formula": entry.formula,
                "name": entry.name,
            },
        )
        if segment is not None:
            segments.append(segment)
        if warning:
            warnings.append(warning)
    return PsatCanonicalizationInputs(
        segments=_ordered_other_antoine_segments(segments, T_boiling),
        warnings=tuple(warnings),
        metadata=(
            {"antoine_table_row_count": len(segments)}
            if segments
            else {}
        ),
    )


def _cached_nist_antoine_payloads(component: Any):
    identifiers = []
    for field_name in ("CAS", "cas", "name"):
        value = _component_value(component, field_name)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text and text not in identifiers:
            identifiers.append(text)
    cache = SQLiteJSONCache(
        runtime_cache_path_for_legacy_directory(
            ANTOINE_CACHE_DIR,
            default_legacy_directory=LEGACY_PROPERTY_CACHE_DIR,
        ),
        'property_resolver',
    )
    cache.migrate_json_directory(
        ANTOINE_CACHE_DIR,
        migration_name='property-resolver-cache',
    )
    seen_keys = set()
    for identifier in identifiers:
        prefix = f"antoine_{identifier}"
        for cache_key, payload in cache.items(prefix=prefix):
            if cache_key in seen_keys:
                continue
            if cache_key != prefix and not cache_key.startswith(prefix + '_'):
                continue
            seen_keys.add(cache_key)
            if payload.get("source") == "NIST WebBook":
                yield identifier, cache_key, payload


def _collect_cached_nist_antoine_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_boiling: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    rows = []
    seen = set()
    for identifier, cache_key, payload in _cached_nist_antoine_payloads(
        adapter.component
    ):
        try:
            values = tuple(
                float(payload[key])
                for key in ("A", "B", "C", "T_min", "T_max")
            )
        except (KeyError, TypeError, ValueError):
            continue
        if values in seen:
            continue
        seen.add(values)
        rows.append((identifier, cache_key, values))
    segments = []
    warnings = []
    for identifier, cache_key, values in rows:
        A, B, C, T_min, T_max = values
        segment, warning = _other_antoine_segment(
            adapter,
            source="NIST WebBook",
            method="cached_nist_antoine",
            A=A,
            B=B,
            C=C,
            T_min=T_min,
            T_max=T_max,
            T_boiling=T_boiling,
            confidence_profile=CACHED_NIST_ANTOINE_PROFILE,
            metadata={
                "resolved_identifier": identifier,
                "cache_key": cache_key,
                "cache_database": "property_cache.sqlite",
                "cache_only": True,
            },
        )
        if segment is not None:
            segments.append(segment)
        if warning:
            warnings.append(warning)
    return PsatCanonicalizationInputs(
        segments=_ordered_other_antoine_segments(segments, T_boiling),
        warnings=tuple(warnings),
        metadata=(
            {"cached_nist_antoine_row_count": len(segments)}
            if segments
            else {}
        ),
    )


def _collect_perry_2_8_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_boiling: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    """Translate every resolved Perry Table 2-8 Psat row into a segment."""
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..perry_properties import get_perry_property_library
        else:
            from perry_properties import get_perry_property_library
    except ImportError:
        return PsatCanonicalizationInputs()
    library = get_perry_property_library()
    entry = None
    resolved_identifier = None
    for identifier in _component_identifier_candidates(adapter.component):
        entry = library.get(identifier)
        if entry:
            resolved_identifier = identifier
            break
    if not entry:
        return PsatCanonicalizationInputs()

    segments = []
    warnings = []
    for row_index, row in enumerate(entry.get("vapor_pressure", ())):
        try:
            segment = _perry_2_8_segment(
                adapter,
                library,
                entry,
                row,
                row_index,
                resolved_identifier,
            )
        except PsatAdapterError as error:
            warnings.append(str(error))
            continue
        segment, warning = _admit_direct_psat_segment(
            adapter,
            segment,
            T_boiling=T_boiling,
            confidence_profile=None,
            source_label="Perry 9th Table 2-8",
        )
        if segment is not None:
            segments.append(segment)
        if warning:
            warnings.append(warning)
    return PsatCanonicalizationInputs(
        segments=tuple(segments),
        warnings=tuple(warnings),
        metadata=(
            {"perry_2_8_cas": entry.get("cas")}
            if segments
            else {}
        ),
    )


def _perry_2_8_segment(
    adapter: PsatCanonicalizationAdapter,
    library: Any,
    entry: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index: int,
    resolved_identifier: Optional[str],
) -> PsatSegment:
    label = f"Perry 2-8 row {row_index} for {entry.get('name') or entry.get('cas')}"
    try:
        T_min = float(row.get("T_min_K"))
        T_max = float(row.get("T_max_K"))
    except (TypeError, ValueError) as error:
        raise PsatAdapterError(f"{label} has an invalid temperature range") from error
    if not all(math.isfinite(value) for value in (T_min, T_max)):
        raise PsatAdapterError(f"{label} has a non-finite temperature range")
    if T_min <= 0.0 or T_max <= T_min:
        raise PsatAdapterError(f"{label} requires 0 < Tmin < Tmax")
    coefficients = library.vapor_pressure_coefficients(dict(row))
    if coefficients is None:
        raise PsatAdapterError(f"{label} has unusable coefficients")
    C1, C2, C3, C4, C5 = coefficients
    ln_pa_to_ln_bar = math.log(100000.0)

    def ln_pressure(T: float) -> float:
        try:
            return (
                C1
                + C2 / T
                + C3 * math.log(T)
                + C4 * T**C5
                - ln_pa_to_ln_bar
            )
        except (ValueError, ZeroDivisionError, OverflowError) as error:
            raise PsatAdapterError(f"{label} failed at {T:g} K") from error

    def derivative(T: float) -> float:
        try:
            return (
                -C2 / T**2
                + C3 / T
                + C4 * C5 * T ** (C5 - 1.0)
            )
        except (ValueError, ZeroDivisionError, OverflowError) as error:
            raise PsatAdapterError(f"{label} derivative failed at {T:g} K") from error

    try:
        P_min_bar = math.exp(ln_pressure(T_min))
        P_max_bar = math.exp(ln_pressure(T_max))
    except OverflowError as error:
        raise PsatAdapterError(f"{label} endpoint pressure overflowed") from error
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (P_min_bar, P_max_bar)
    ):
        raise PsatAdapterError(f"{label} endpoint pressures are invalid")
    context = _component_context(adapter.component)
    context.update({
        "perry_cas": entry.get("cas"),
        "perry_name": entry.get("name"),
    })
    return PsatSegment(
        source="Perry 9th Table 2-8",
        method="perry_2_8_vapor_pressure",
        segment_type=PsatSegmentType.PINNED,
        priority=int(PsatPriority.PERRY_2_8),
        T_min=T_min,
        T_max=T_max,
        ln_pressure_function=ln_pressure,
        derivative_function=derivative,
        quality=PERRY_2_8_PSAT_QUALITY,
        P_min_bar=P_min_bar,
        P_max_bar=P_max_bar,
        context=context,
        metadata={
            "provider": "Perry",
            "property_dependencies": (),
            "table": row.get("source_table", "2-8"),
            "equation": row.get("equation"),
            "row_index": row_index,
            "resolved_identifier": resolved_identifier,
            "original_coefficients": tuple(row.get("coefficients", ())),
            "normalized_coefficients": coefficients,
            "declared_T_min": T_min,
            "declared_T_max": T_max,
            "reported_P_min_Pa": row.get("P_at_T_min_Pa"),
            "reported_P_max_Pa": row.get("P_at_T_max_Pa"),
        },
    )


def _collect_perry_2_10_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_boiling: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    """Translate a resolved Perry Table 2-10 table into a log-PCHIP segment."""
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..perry_properties import get_perry_property_library
        else:
            from perry_properties import get_perry_property_library
    except ImportError:
        return PsatCanonicalizationInputs()
    library = get_perry_property_library()
    curve = None
    resolved_identifier = None
    for identifier in _component_identifier_candidates(adapter.component):
        curve = library.table_2_10_vapor_pressure_curve(identifier)
        if curve is not None:
            resolved_identifier = identifier
            break
    if curve is None:
        return PsatCanonicalizationInputs()
    interpolator, row = curve
    derivative_interpolator = interpolator.derivative()
    T_min = float(row["T_min_K"])
    T_max = float(row["T_max_K"])
    pressures = tuple(float(value) for value in row["pressures_bar"])

    def ln_pressure(T: float) -> float:
        value = float(interpolator(T))
        if not math.isfinite(value):
            raise PsatAdapterError(
                f"Perry 2-10 Psat returned a non-finite value at {T:g} K"
            )
        return value

    def derivative(T: float) -> float:
        value = float(derivative_interpolator(T))
        if not math.isfinite(value):
            raise PsatAdapterError(
                f"Perry 2-10 Psat derivative is non-finite at {T:g} K"
            )
        return value

    context = _component_context(adapter.component)
    context.update({
        "perry_cas": row.get("cas"),
        "perry_name": row.get("name"),
    })
    segment = PsatSegment(
        source="Perry 9th Table 2-10",
        method="perry_2_10_vapor_pressure",
        segment_type=PsatSegmentType.PINNED,
        priority=int(PsatPriority.PERRY_2_10),
        T_min=T_min,
        T_max=T_max,
        ln_pressure_function=ln_pressure,
        derivative_function=derivative,
        derivative_basis=PsatDerivativeBasis.INTERPOLATED,
        quality=STANDARD_DIRECT_TABLE_PROFILE.hard_tb_quality,
        P_min_bar=pressures[0],
        P_max_bar=pressures[-1],
        context=context,
        metadata={
            "provider": "Perry",
            "property_dependencies": (),
            "table": row.get("source_table", "2-10"),
            "interpolation": "log_pressure_pchip",
            "resolved_identifier": resolved_identifier,
            "point_count": len(pressures),
            "temperatures_K": tuple(row["temperatures_K"]),
            "pressures_bar": pressures,
            "declared_T_min": T_min,
            "declared_T_max": T_max,
        },
    )
    segment, warning = _admit_direct_psat_segment(
        adapter,
        segment,
        T_boiling=T_boiling,
        confidence_profile=STANDARD_DIRECT_TABLE_PROFILE,
        source_label="Perry 9th Table 2-10",
    )
    if segment is None:
        return PsatCanonicalizationInputs(warnings=(str(warning),))
    return PsatCanonicalizationInputs(
        segments=(segment,),
        metadata={"perry_2_10_cas": row.get("cas")},
    )


def _frozen_hvap_plan(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_min: Optional[float],
    T_critical: Optional[float],
    T_boiling: Optional[float],
) -> Optional[_FrozenHvapPlan]:
    if T_min is None or T_critical is None or T_boiling is None:
        return None
    try:
        lower_temperature = float(T_min)
        critical_temperature = float(T_critical)
        boiling_temperature = float(T_boiling)
    except (TypeError, ValueError):
        return None
    if not (
        0.0
        < lower_temperature
        < boiling_temperature
        < critical_temperature
    ):
        return None
    cache_key = (
        lower_temperature,
        critical_temperature,
        boiling_temperature,
    )
    if cache_key in adapter._frozen_hvap_plans:
        return adapter._frozen_hvap_plans[cache_key]

    sample_count = 41
    temperatures = tuple(
        lower_temperature
        + (boiling_temperature - lower_temperature)
        * index
        / (sample_count - 1)
        for index in range(sample_count)
    )
    results = []
    for temperature in temperatures:
        result = _resolved_hvap_at_temperature(
            adapter,
            temperature,
        )
        value = (
            _finite_resolution_value(result)
            if result is not None
            else None
        )
        quality = (
            _optional_quality(result.quality)
            if result is not None
            else None
        )
        if (
            value is None
            or value <= 0.0
            or quality is None
        ):
            adapter._frozen_hvap_plans[cache_key] = None
            return None
        results.append((result, value, quality))

    methods = {
        str(result.method or "").strip().lower()
        for result, _value, _quality in results
    }
    sources = {
        str(result.source or "").strip().lower()
        for result, _value, _quality in results
    }
    if len(methods) != 1 or len(sources) != 1:
        adapter._frozen_hvap_plans[cache_key] = None
        return None
    method = next(iter(methods))
    relation_class = _hvap_relation_class(method)
    if relation_class is None:
        adapter._frozen_hvap_plans[cache_key] = None
        return None
    quality = min(item[2] for item in results)
    if quality < HVAP_MINIMUM_COMPLETION_QUALITY:
        adapter._frozen_hvap_plans[cache_key] = None
        return None

    switch_pressure = _hvap_switch_pressure(
        quality,
        relation_class,
    )
    if adapter.completion_pressure_floor is not None:
        switch_pressure = max(
            switch_pressure,
            adapter.completion_pressure_floor,
        )
    log_values = [
        math.log(value * 1000.0)
        for _result, value, _quality in results
    ]
    interpolator = PchipInterpolator(
        temperatures,
        log_values,
        extrapolate=False,
    )

    def evaluator_J_mol(temperature: float) -> float:
        if not lower_temperature <= temperature <= boiling_temperature:
            raise PsatCanonicalizationError(
                "Frozen Hvap relation is outside its sampled range"
            )
        value = math.exp(float(interpolator(temperature)))
        if not math.isfinite(value) or value <= 0.0:
            raise PsatCanonicalizationError(
                "Frozen Hvap relation returned an invalid value"
            )
        return value

    representative = results[-1][0]
    plan = _FrozenHvapPlan(
        evaluator_J_mol=evaluator_J_mol,
        quality=quality,
        source=str(representative.source),
        method=str(representative.method),
        relation_class=relation_class,
        T_min=lower_temperature,
        T_max=boiling_temperature,
        switch_pressure_bar=switch_pressure,
        sample_count=sample_count,
    )
    adapter._frozen_hvap_plans[cache_key] = plan
    return plan


def _resolved_hvap_at_temperature(
    adapter: PsatCanonicalizationAdapter,
    temperature: float,
) -> Optional[PropertyResolutionResult]:
    if adapter.hvap_at_temperature is not None:
        raw_value = adapter.hvap_at_temperature(
            adapter.component,
            temperature,
        )
        if isinstance(raw_value, PropertyResolutionResult):
            return raw_value
        if isinstance(raw_value, Mapping) and "value" in raw_value:
            return PropertyResolutionResult(
                value=raw_value.get("value"),
                source=str(raw_value.get("source") or "provided"),
                method=str(raw_value.get("method") or "provided_hvap_fit"),
                quality=raw_value.get("quality"),
                notes=str(raw_value.get("notes") or ""),
            )
        return None

    identifiers = _component_identifier_candidates(adapter.component)
    if not identifiers:
        return None
    if adapter._property_resolver is None:
        from .resolver import PropertyResolver

        adapter._property_resolver = PropertyResolver()
    props = (
        dict(adapter.component)
        if isinstance(adapter.component, Mapping)
        else dict(vars(adapter.component))
    )
    try:
        result = adapter._property_resolver.resolve_hvap(
            identifiers[0],
            props,
            T=temperature,
            allow_online=adapter.allow_online_hvap,
            allow_estimation=False,
        )
    except (ArithmeticError, LookupError, TypeError, ValueError):
        return None
    return result if isinstance(result, PropertyResolutionResult) else None


def _hvap_relation_class(method: str) -> Optional[str]:
    normalized = str(method or "").strip().lower()
    if normalized == "nist_hvap_watson_fit":
        return "multi_point_fit"
    if (
        normalized == "provided_hvap_fit"
        or "perry_2_69" in normalized
        or "heat_of_vaporization" in normalized
    ):
        return "broad_direct"
    return None


def _hvap_switch_pressure(
    quality: float,
    relation_class: str,
) -> float:
    if quality >= HVAP_HIGH_QUALITY_THRESHOLD:
        return HVAP_HIGH_QUALITY_SWITCH_PRESSURE_BAR
    if quality >= HVAP_MEDIUM_QUALITY_THRESHOLD:
        return HVAP_MEDIUM_QUALITY_SWITCH_PRESSURE_BAR
    return HVAP_LOW_QUALITY_SWITCH_PRESSURE_BAR


def _upper_eased_linear_weight(
    coordinate: float,
    endpoint_width: float = 0.05,
) -> tuple[float, float]:
    if coordinate <= 0.0:
        return 0.0, 0.0
    if coordinate <= 1.0 - endpoint_width:
        return coordinate, 1.0
    if coordinate >= 1.0:
        return 1.0, 0.0
    local = (1.0 - coordinate) / endpoint_width
    polynomial = (
        6.0 * local**3
        - 8.0 * local**4
        + 3.0 * local**5
    )
    derivative = (
        18.0 * local**2
        - 32.0 * local**3
        + 15.0 * local**4
    )
    return 1.0 - endpoint_width * polynomial, derivative


def _require_matching_critical_anchor(
    conditions: PsatBoundaryConditions,
    *,
    Tc: float,
    Pc_bar: float,
    method: str,
) -> None:
    critical = conditions.anchor("Tc")
    if critical is None:
        raise PsatCanonicalizationError(
            f"{method} requires the canonical critical anchor"
        )
    tolerance = 1.0e-8 * max(Tc, 1.0)
    if (
        abs(critical.temperature - Tc) > tolerance
        or abs(critical.ln_pressure - math.log(Pc_bar)) > 1.0e-8
    ):
        raise PsatCanonicalizationError(
            f"{method} critical properties conflict with the canonical anchor"
        )


def _collect_no_hard_fallback_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_min: Optional[float] = None,
    T_critical: Optional[float] = None,
    P_critical_bar: Optional[float] = None,
    T_boiling: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    if _pfd_psat_correlation(adapter.component) is not None:
        return PsatCanonicalizationInputs()
    critical_temperature = _property_candidate(
        adapter,
        "Tc",
        explicit_value=T_critical,
    )
    critical_pressure = _property_candidate(
        adapter,
        "Pc",
        explicit_value=P_critical_bar,
    )
    Tc = (
        _positive_resolution_value(critical_temperature)
        if critical_temperature is not None
        else None
    )
    Pc_bar = (
        _positive_resolution_value(critical_pressure)
        if critical_pressure is not None
        else None
    )
    try:
        relation_T_min = float(T_min)
    except (TypeError, ValueError):
        return PsatCanonicalizationInputs()
    if (
        Tc is None
        or Pc_bar is None
        or not 0.0 < relation_T_min < Tc
    ):
        return PsatCanonicalizationInputs()
    critical_qualities = (
        _optional_quality(
            None if critical_temperature is None
            else critical_temperature.quality
        ),
        _optional_quality(
            None if critical_pressure is None
            else critical_pressure.quality
        ),
    )
    criticals_are_admissible = (
        all(quality is not None for quality in critical_qualities)
        and min(critical_qualities)
        >= AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY
    )
    nannoolal_fallback_warning = None
    if not criticals_are_admissible:
        nannoolal_inputs = _collect_no_hard_nannoolal_inputs(
            adapter,
            T_min=relation_T_min,
            T_critical=Tc,
            P_critical_bar=Pc_bar,
            T_boiling=T_boiling,
            critical_temperature=critical_temperature,
            critical_pressure=critical_pressure,
        )
        if (
            nannoolal_inputs.canonical_override is not None
            or nannoolal_inputs.segments
            or nannoolal_inputs.relations
        ):
            return nannoolal_inputs
        nannoolal_fallback_warning = (
            "Qualified Tc/Pc and a usable no-hard Nannoolal Psat route "
            "were unavailable; falling back to quality-weighted no-hard "
            "Ambrose-Walton"
        )

    warnings = [
        warning
        for warning in (
            _property_admission_warning(
                "Tc",
                critical_temperature,
            ),
            _property_admission_warning(
                "Pc",
                critical_pressure,
            ),
        )
        if warning is not None
    ]
    if nannoolal_fallback_warning is not None:
        warnings.append(nannoolal_fallback_warning)
    trusted_tb = _boiling_point_candidate(
        adapter,
        explicit_value=T_boiling,
        allow_estimation=False,
    )
    trusted_tb_value = (
        _positive_resolution_value(trusted_tb)
        if trusted_tb is not None
        else None
    )
    trusted_tb_quality = (
        _optional_quality(trusted_tb.quality)
        if trusted_tb is not None
        else None
    )
    trusted_tb_is_admissible = (
        trusted_tb_value is not None
        and trusted_tb_value < Tc
        and trusted_tb_quality is not None
        and trusted_tb_quality
        >= AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY
        and not _resolution_is_estimated(trusted_tb)
    )
    if trusted_tb_is_admissible:
        warning = _property_admission_warning(
            "Tb",
            trusted_tb,
        )
        if warning is not None:
            warnings.append(warning)

    omega_candidate = _property_candidate(adapter, "omega")
    omega = (
        _finite_resolution_value(omega_candidate)
        if omega_candidate is not None
        else None
    )
    omega_quality = (
        _optional_quality(omega_candidate.quality)
        if omega_candidate is not None
        else None
    )
    omega_is_admissible = (
        omega is not None
        and -0.5 <= omega <= 2.0
        and omega_quality is not None
        and omega_quality
        >= AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY
    )
    route = None
    consumed_tb = None
    consumed_omega_quality = None
    effective_omega_reference = None
    omega_slope = 0.0
    transition_temperature = (
        AMBROSE_WALTON_MINIMUM_REDUCED_TEMPERATURE * Tc
    )
    if trusted_tb_is_admissible:
        consumed_tb = trusted_tb
        effective_omega_reference = _ambrose_walton_omega_from_pressure(
            trusted_tb_value,
            math.log(1.01325),
            Tc,
            Pc_bar,
            preferred_omega=omega,
        )
        if (
            omega_is_admissible
            and trusted_tb_value < transition_temperature
        ):
            route = "tb_variable_omega"
            consumed_omega_quality = omega_quality
            omega_slope = (
                (omega - effective_omega_reference)
                / (transition_temperature - trusted_tb_value)
            )
        else:
            route = "tb_effective_constant_omega"
    elif omega_is_admissible:
        route = "physical_omega"
        consumed_omega_quality = omega_quality
        effective_omega_reference = omega
    else:
        estimated_tb = _boiling_point_candidate(
            adapter,
            explicit_value=T_boiling,
            allow_estimation=True,
        )
        estimated_tb_value = (
            _positive_resolution_value(estimated_tb)
            if estimated_tb is not None
            else None
        )
        if (
            estimated_tb_value is None
            or not relation_T_min < estimated_tb_value < Tc
        ):
            return PsatCanonicalizationInputs()
        consumed_tb = estimated_tb
        trusted_tb_value = estimated_tb_value
        trusted_tb_quality = _optional_quality(estimated_tb.quality)
        effective_omega_reference = _ambrose_walton_omega_from_pressure(
            trusted_tb_value,
            math.log(1.01325),
            Tc,
            Pc_bar,
        )
        route = "tb_effective_constant_omega"
        warnings.append(
            "No qualified Tb or omega was available for no-hard Psat; "
            f"using Tb from {estimated_tb.source}/{estimated_tb.method} "
            f"at quality {trusted_tb_quality}"
        )

    if effective_omega_reference is None or route is None:
        return PsatCanonicalizationInputs()
    if consumed_omega_quality is not None:
        warning = _property_admission_warning(
            "omega",
            omega_candidate,
        )
        if warning is not None:
            warnings.append(warning)
    adapter._no_hard_boiling_candidate = consumed_tb
    boiling_temperature = (
        _positive_resolution_value(consumed_tb)
        if consumed_tb is not None
        else None
    )
    hvap_plan = _frozen_hvap_plan(
        adapter,
        T_min=relation_T_min,
        T_critical=Tc,
        T_boiling=boiling_temperature,
    )
    lower_switch_pressure = (
        adapter.completion_pressure_floor
        if hvap_plan is None
        else hvap_plan.switch_pressure_bar
    )

    def effective_omega(T: float) -> tuple[float, float]:
        if route != "tb_variable_omega":
            return effective_omega_reference, 0.0
        coordinate = (
            (T - trusted_tb_value)
            / (transition_temperature - trusted_tb_value)
        )
        weight, weight_derivative = _upper_eased_linear_weight(
            coordinate,
        )
        omega_value = (
            effective_omega_reference
            + (omega - effective_omega_reference) * weight
        )
        omega_derivative = (
            (omega - effective_omega_reference)
            * weight_derivative
            / (transition_temperature - trusted_tb_value)
        )
        return omega_value, omega_derivative

    def ln_pressure(T: float) -> float:
        omega_value, _omega_derivative = effective_omega(T)
        return _ambrose_walton_ln_pressure(
            T,
            Tc,
            Pc_bar,
            omega_value,
        )

    def derivative(T: float) -> float:
        omega_value, omega_derivative = effective_omega(T)
        return (
            _ambrose_walton_dln_pressure_dT(
                T,
                Tc,
                omega_value,
            )
            + _ambrose_walton_omega_sensitivity(
                T,
                Tc,
                omega_value,
            )
            * omega_derivative
        )

    quality_inputs = [*critical_qualities]
    if consumed_tb is not None:
        quality_inputs.append(
            _optional_quality(consumed_tb.quality)
        )
    if consumed_omega_quality is not None:
        quality_inputs.append(consumed_omega_quality)
    quality_inputs = [
        quality for quality in quality_inputs
        if quality is not None
    ]
    base_quality = min(quality_inputs)
    if route == "tb_variable_omega":
        quality_bands = (
            (
                0.25,
                Pc_bar,
                NO_HARD_AW_UPPER_QUALITY_FACTOR,
                "variable_omega_upper",
            ),
            (
                0.05,
                0.25,
                NO_HARD_AW_MIDDLE_QUALITY_FACTOR,
                "variable_omega_middle",
            ),
            (
                adapter.completion_pressure_floor,
                0.05,
                NO_HARD_AW_LOW_QUALITY_FACTOR,
                "variable_omega_low",
            ),
        )
    elif consumed_tb is not None:
        quality_bands = (
            (
                1.01325,
                Pc_bar,
                NO_HARD_TB_EFFECTIVE_UPPER_QUALITY_FACTOR,
                "tb_to_critical",
            ),
            (
                0.25,
                1.01325,
                NO_HARD_AW_MIDDLE_QUALITY_FACTOR,
                "quarter_bar_to_tb",
            ),
            (
                0.05,
                0.25,
                NO_HARD_AW_LOW_QUALITY_FACTOR,
                "low_pressure",
            ),
            (
                adapter.completion_pressure_floor,
                0.05,
                NO_HARD_AW_DEEPEST_QUALITY_FACTOR,
                "deepest_pressure",
            ),
        )
    else:
        quality_bands = (
            (
                1.01325,
                Pc_bar,
                NO_HARD_AW_UPPER_QUALITY_FACTOR,
                "omega_only_upper",
            ),
            (
                0.25,
                1.01325,
                NO_HARD_AW_MIDDLE_QUALITY_FACTOR,
                "omega_only_middle",
            ),
            (
                0.05,
                0.25,
                NO_HARD_AW_LOW_QUALITY_FACTOR,
                "omega_only_low",
            ),
            (
                adapter.completion_pressure_floor,
                0.05,
                NO_HARD_AW_DEEPEST_QUALITY_FACTOR,
                "omega_only_deepest",
            ),
        )
    common_context = {
        **_component_context(adapter.component),
        "Tc_source": critical_temperature.source,
        "Tc_method": critical_temperature.method,
        "Tc_quality": critical_temperature.quality,
        "Pc_source": critical_pressure.source,
        "Pc_method": critical_pressure.method,
        "Pc_quality": critical_pressure.quality,
        "Tb_source": (
            None if consumed_tb is None else consumed_tb.source
        ),
        "Tb_method": (
            None if consumed_tb is None else consumed_tb.method
        ),
        "Tb_quality": (
            None if consumed_tb is None else consumed_tb.quality
        ),
        "omega_source": (
            None if omega_candidate is None else omega_candidate.source
        ),
        "omega_method": (
            None if omega_candidate is None else omega_candidate.method
        ),
        "omega_quality": omega_quality,
    }
    relations = []
    for nominal_lower, upper_pressure, quality_factor, band_name in quality_bands:
        if nominal_lower is None:
            lower_pressure = lower_switch_pressure
        elif lower_switch_pressure is None:
            lower_pressure = nominal_lower
        else:
            lower_pressure = max(
                nominal_lower,
                lower_switch_pressure,
            )
        if (
            lower_pressure is not None
            and lower_pressure >= upper_pressure * (1.0 - 1.0e-12)
        ):
            continue
        relation_quality = base_quality * quality_factor

        def bind_no_hard_aw(
            conditions,
            relation_quality=relation_quality,
            quality_factor=quality_factor,
            band_name=band_name,
        ):
            _require_matching_critical_anchor(
                conditions,
                Tc=Tc,
                Pc_bar=Pc_bar,
                method="No-hard Ambrose-Walton",
            )
            return BoundaryConditionedPsatEvaluation(
                ln_pressure_function=ln_pressure,
                derivative_function=derivative,
                quality=relation_quality,
                metadata={
                    "no_hard_route": route,
                    "no_hard_quality_band": band_name,
                    "tb_was_estimated": (
                        consumed_tb is not None
                        and _resolution_is_estimated(consumed_tb)
                    ),
                    "effective_omega_reference": (
                        effective_omega_reference
                    ),
                    "effective_omega_reference_temperature_K": (
                        None
                        if consumed_tb is None
                        else _positive_resolution_value(consumed_tb)
                    ),
                    "omega_slope_per_K": omega_slope,
                    "transition_temperature_K": transition_temperature,
                    "switch_pressure_bar": lower_switch_pressure,
                    "switch_basis": (
                        "frozen_hvap_quality"
                        if hvap_plan is not None
                        else "continue_aw_to_domain_floor"
                    ),
                    "quality_factor": quality_factor,
                },
            )

        relations.append(BoundaryConditionedPsatSegment(
            source="calculated",
            method="no_hard_ambrose_walton",
            segment_type=PsatSegmentType.COMPLETION,
            priority=int(PsatPriority.NO_HARD_FALLBACK),
            quality=relation_quality,
            boundary_requirement=PsatBoundaryRequirement.NONE,
            T_min=relation_T_min,
            T_max=Tc,
            evaluation_factory=bind_no_hard_aw,
            P_min_bar=lower_pressure,
            P_max_bar=upper_pressure,
            required_anchor_names=("Tc",),
            context=common_context,
            metadata={
                "provider": "Ambrose-Walton",
                "completion": "no-hard full fallback",
                "no_hard_route": route,
                "no_hard_quality_band": band_name,
                "tb_was_estimated": (
                    consumed_tb is not None
                    and _resolution_is_estimated(consumed_tb)
                ),
                "switch_pressure_bar": lower_switch_pressure,
                "quality_factor": quality_factor,
            },
            inherit_boundary_quality=False,
            requires_no_hard_segments=True,
        ))
    additional_anchors = ()
    if consumed_tb is not None and boiling_temperature is not None:
        additional_anchors = (
            PsatEndpoint.fixed_pressure_point(
                boiling_temperature,
                1.01325,
                source=str(consumed_tb.source),
                method=str(consumed_tb.method),
                quality=float(consumed_tb.quality),
                context={"anchor_name": "Tb"},
            ),
        )
    return PsatCanonicalizationInputs(
        relations=tuple(relations),
        additional_anchors=additional_anchors,
        warnings=tuple(warnings),
        metadata={
            "no_hard_ambrose_walton": True,
            "no_hard_route": route,
        },
    )


def _collect_no_hard_nannoolal_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_min: float,
    T_critical: float,
    P_critical_bar: float,
    T_boiling: Optional[float],
    critical_temperature: Optional[PropertyResolutionResult],
    critical_pressure: Optional[PropertyResolutionResult],
) -> PsatCanonicalizationInputs:
    boiling = _boiling_point_candidate(
        adapter,
        explicit_value=T_boiling,
        allow_estimation=True,
    )
    Tb = (
        _positive_resolution_value(boiling)
        if boiling is not None
        else None
    )
    smiles = _smiles_candidate(adapter)
    if (
        Tb is None
        or not T_min < Tb < T_critical
        or smiles is None
        or not smiles.value
    ):
        return PsatCanonicalizationInputs()
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..nannoolal_method import estimate_psat
        else:
            from nannoolal_method import estimate_psat

        estimate = estimate_psat(str(smiles.value), tb=Tb)
    except (ArithmeticError, LookupError, TypeError, ValueError):
        return PsatCanonicalizationInputs()
    if estimate.db is None or estimate.tb_K is None:
        return PsatCanonicalizationInputs()

    def native_ln_pressure(T: float) -> float:
        pressure_kPa = estimate.psat_kPa(T)
        if (
            pressure_kPa is None
            or not math.isfinite(pressure_kPa)
            or pressure_kPa <= 0.0
        ):
            raise PsatCanonicalizationError(
                "No-hard Nannoolal returned invalid pressure"
            )
        return math.log(pressure_kPa / 100.0)

    def native_derivative(T: float) -> float:
        enthalpy = estimate.dhvap_J_mol(T)
        if (
            enthalpy is None
            or not math.isfinite(enthalpy)
            or enthalpy <= 0.0
        ):
            raise PsatCanonicalizationError(
                "No-hard Nannoolal returned invalid derivative"
            )
        return enthalpy / (R * T**2)

    handoff_temperature = 0.8 * T_critical
    if not Tb < handoff_temperature < T_critical:
        return PsatCanonicalizationInputs()
    try:
        handoff_ln_pressure = native_ln_pressure(
            handoff_temperature
        )
        handoff_slope = native_derivative(handoff_temperature)
        upper_omega = _ambrose_walton_omega_from_pressure(
            handoff_temperature,
            handoff_ln_pressure,
            T_critical,
            P_critical_bar,
        )
        upper_ln_pressure, upper_derivative = (
            _c1_anchored_aw_upper_functions(
                left_temperature=handoff_temperature,
                left_ln_pressure=handoff_ln_pressure,
                left_slope=handoff_slope,
                Tc=T_critical,
                Pc_bar=P_critical_bar,
                omega=upper_omega,
            )
        )
    except PsatCanonicalizationError:
        return PsatCanonicalizationInputs()

    nannoolal_quality_inputs = [
        _optional_quality(boiling.quality),
        _optional_quality(smiles.quality),
    ]
    nannoolal_quality_inputs = [
        quality for quality in nannoolal_quality_inputs
        if quality is not None
    ]
    nannoolal_base_quality = min(
        nannoolal_quality_inputs or [0.5]
    )
    middle_quality = (
        nannoolal_base_quality
        * NANNOOLAL_PSAT_QUALITY_FACTOR
    )
    deep_quality = (
        nannoolal_base_quality
        * NANNOOLAL_DEEP_PSAT_QUALITY_FACTOR
    )
    critical_quality_inputs = [
        _optional_quality(
            None if critical_temperature is None
            else critical_temperature.quality
        ),
        _optional_quality(
            None if critical_pressure is None
            else critical_pressure.quality
        ),
    ]
    critical_quality_inputs = [
        quality for quality in critical_quality_inputs
        if quality is not None
    ]
    upper_quality_factor = (
        AMBROSE_WALTON_QUALITY_FACTOR
        * NANNOOLAL_UPPER_AW_EXTRA_QUALITY_FACTOR
    )
    upper_quality = (
        min(middle_quality, *(critical_quality_inputs or [middle_quality]))
        * upper_quality_factor
    )
    warnings = [
        "Qualified Tc/Pc were unavailable for no-hard Psat; "
        "using Tb-centered Nannoolal through 0.8Tc and anchored C1 "
        "Ambrose-Walton to the available critical point"
    ]
    if _resolution_is_estimated(boiling):
        warnings.append(
            f"Nannoolal Psat uses estimated Tb from "
            f"{boiling.source}/{boiling.method} at quality {boiling.quality}"
        )
    warnings.extend(str(item) for item in estimate.warnings)

    common_context = {
        **_component_context(adapter.component),
        "Tc_source": (
            None
            if critical_temperature is None
            else critical_temperature.source
        ),
        "Tc_method": (
            None
            if critical_temperature is None
            else critical_temperature.method
        ),
        "Tc_quality": (
            None
            if critical_temperature is None
            else critical_temperature.quality
        ),
        "Pc_source": (
            None
            if critical_pressure is None
            else critical_pressure.source
        ),
        "Pc_method": (
            None
            if critical_pressure is None
            else critical_pressure.method
        ),
        "Pc_quality": (
            None
            if critical_pressure is None
            else critical_pressure.quality
        ),
        "Tb_source": boiling.source,
        "Tb_method": boiling.method,
        "Tb_quality": boiling.quality,
        "smiles_source": smiles.source,
        "smiles_method": smiles.method,
        "smiles_quality": smiles.quality,
    }

    def require_critical(conditions):
        _require_matching_critical_anchor(
            conditions,
            Tc=T_critical,
            Pc_bar=P_critical_bar,
            method="No-hard Nannoolal/Ambrose-Walton",
        )

    def bind_no_hard_nannoolal_upper(conditions):
        require_critical(conditions)
        return BoundaryConditionedPsatEvaluation(
            ln_pressure_function=upper_ln_pressure,
            derivative_function=upper_derivative,
            quality=upper_quality,
            metadata={
                "no_hard_route": "nannoolal_c1_aw_upper",
                "nannoolal_handoff_temperature_K": (
                    handoff_temperature
                ),
                "nannoolal_handoff_reduced_temperature": 0.8,
                "upper_aw_omega": upper_omega,
                "quality_factor": upper_quality_factor,
                "nannoolal_handoff_quality": middle_quality,
                "nannoolal_warnings": tuple(estimate.warnings),
            },
        )

    def bind_no_hard_nannoolal_middle(conditions):
        require_critical(conditions)
        return BoundaryConditionedPsatEvaluation(
            ln_pressure_function=native_ln_pressure,
            derivative_function=native_derivative,
            quality=middle_quality,
            metadata={
                "no_hard_route": "tb_centered_nannoolal",
                "no_hard_quality_band": "nannoolal_middle",
                "quality_factor": NANNOOLAL_PSAT_QUALITY_FACTOR,
                "nannoolal_warnings": tuple(estimate.warnings),
            },
        )

    def bind_no_hard_nannoolal_deep(conditions):
        require_critical(conditions)
        return BoundaryConditionedPsatEvaluation(
            ln_pressure_function=native_ln_pressure,
            derivative_function=native_derivative,
            quality=deep_quality,
            metadata={
                "no_hard_route": "tb_centered_nannoolal",
                "no_hard_quality_band": "nannoolal_deep",
                "quality_factor": (
                    NANNOOLAL_DEEP_PSAT_QUALITY_FACTOR
                ),
                "nannoolal_warnings": tuple(estimate.warnings),
            },
        )

    handoff_pressure = math.exp(handoff_ln_pressure)
    relations = [
        BoundaryConditionedPsatSegment(
            source="estimated",
            method="no_hard_nannoolal_upper_aw",
            segment_type=PsatSegmentType.COMPLETION,
            priority=int(PsatPriority.NANNOOLAL),
            quality=upper_quality,
            boundary_requirement=PsatBoundaryRequirement.NONE,
            T_min=handoff_temperature,
            T_max=T_critical,
            evaluation_factory=bind_no_hard_nannoolal_upper,
            P_min_bar=handoff_pressure,
            P_max_bar=P_critical_bar,
            required_anchor_names=("Tc",),
            context=common_context,
            metadata={
                "provider": "Ambrose-Walton",
                "completion": "Nannoolal C1 upper fallback",
                "no_hard_route": "nannoolal_c1_aw_upper",
                "quality_factor": upper_quality_factor,
            },
            inherit_boundary_quality=False,
            requires_no_hard_segments=True,
        ),
        BoundaryConditionedPsatSegment(
            source="estimated",
            method="no_hard_nannoolal",
            segment_type=PsatSegmentType.COMPLETION,
            priority=int(PsatPriority.NANNOOLAL),
            quality=middle_quality,
            boundary_requirement=PsatBoundaryRequirement.NONE,
            T_min=T_min,
            T_max=handoff_temperature,
            evaluation_factory=bind_no_hard_nannoolal_middle,
            P_min_bar=0.05,
            P_max_bar=handoff_pressure,
            required_anchor_names=("Tc",),
            context=common_context,
            metadata={
                "provider": "Nannoolal",
                "completion": "no-hard middle fallback",
                "no_hard_route": "tb_centered_nannoolal",
                "no_hard_quality_band": "nannoolal_middle",
                "quality_factor": NANNOOLAL_PSAT_QUALITY_FACTOR,
            },
            inherit_boundary_quality=False,
            requires_no_hard_segments=True,
        ),
    ]
    if (
        adapter.completion_pressure_floor is None
        or adapter.completion_pressure_floor < 0.05
    ):
        relations.append(BoundaryConditionedPsatSegment(
            source="estimated",
            method="no_hard_nannoolal_deep",
            segment_type=PsatSegmentType.COMPLETION,
            priority=int(PsatPriority.NANNOOLAL),
            quality=deep_quality,
            boundary_requirement=PsatBoundaryRequirement.NONE,
            T_min=T_min,
            T_max=handoff_temperature,
            evaluation_factory=bind_no_hard_nannoolal_deep,
            P_min_bar=adapter.completion_pressure_floor,
            P_max_bar=0.05,
            required_anchor_names=("Tc",),
            context=common_context,
            metadata={
                "provider": "Nannoolal",
                "completion": "no-hard deep fallback",
                "no_hard_route": "tb_centered_nannoolal",
                "no_hard_quality_band": "nannoolal_deep",
                "quality_factor": (
                    NANNOOLAL_DEEP_PSAT_QUALITY_FACTOR
                ),
            },
            inherit_boundary_quality=False,
            requires_no_hard_segments=True,
        ))
    adapter._no_hard_boiling_candidate = boiling
    return PsatCanonicalizationInputs(
        relations=tuple(relations),
        additional_anchors=(
            PsatEndpoint.fixed_pressure_point(
                Tb,
                1.01325,
                source=str(boiling.source),
                method=str(boiling.method),
                quality=float(boiling.quality),
                context={"anchor_name": "Tb"},
            ),
        ),
        warnings=tuple(dict.fromkeys(warnings)),
        metadata={
            "no_hard_nannoolal": True,
            "no_hard_route": "tb_centered_nannoolal",
        },
    )


def _collect_log_temperature_middle_gap_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_min: Optional[float] = None,
    T_critical: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    try:
        relation_T_min = float(T_min)
        relation_T_max = float(T_critical)
    except (TypeError, ValueError):
        return PsatCanonicalizationInputs()
    if not 0.0 < relation_T_min < relation_T_max:
        return PsatCanonicalizationInputs()

    def bind_middle_gap(conditions):
        left = conditions.left
        right = conditions.right
        if left is None or right is None:
            raise PsatCanonicalizationError(
                "Middle-gap completion requires two hard boundaries"
            )
        if right.ln_pressure <= left.ln_pressure:
            raise PsatCanonicalizationError(
                "Middle-gap completion requires increasing boundary pressure"
            )
        quality = (
            min(left.quality, right.quality)
            * MIDDLE_LOG_T_HERMITE_QUALITY_FACTOR
        )
        pressure_span_decades = (
            (right.ln_pressure - left.ln_pressure) / math.log(10.0)
        )
        bridge = make_c1_bridge_segment(
            left,
            right,
            source="calculated",
            method="log_temperature_hermite_middle",
            priority=int(PsatPriority.MIDDLE_COMPLETION),
            quality=quality,
            coordinate="log_temperature",
            metadata={
                "provider": "Hermite",
                "completion": "two-hard-boundary middle gap",
                "temperature_span_K": (
                    right.temperature - left.temperature
                ),
                "pressure_span_decades": pressure_span_decades,
                "quality_factor": (
                    MIDDLE_LOG_T_HERMITE_QUALITY_FACTOR
                ),
            },
        )
        return BoundaryConditionedPsatEvaluation(
            ln_pressure_function=bridge.raw_ln_pressure,
            derivative_function=bridge.raw_dln_pressure_dT,
            quality=quality,
            context=bridge.context,
            metadata=bridge.metadata,
        )

    relation = BoundaryConditionedPsatSegment(
        source="calculated",
        method="log_temperature_hermite_middle",
        segment_type=PsatSegmentType.COMPLETION,
        priority=int(PsatPriority.MIDDLE_COMPLETION),
        quality=MIDDLE_LOG_T_HERMITE_QUALITY_FACTOR,
        boundary_requirement=PsatBoundaryRequirement.BOTH,
        T_min=relation_T_min,
        T_max=relation_T_max,
        evaluation_factory=bind_middle_gap,
        left_derivative_requirement=PsatDerivativeRequirement.REQUIRED,
        right_derivative_requirement=PsatDerivativeRequirement.REQUIRED,
        allowed_left_segment_types=(
            PsatSegmentType.PINNED,
            PsatSegmentType.CANONICAL_OVERRIDE,
        ),
        allowed_right_segment_types=(
            PsatSegmentType.PINNED,
            PsatSegmentType.CANONICAL_OVERRIDE,
        ),
        metadata={
            "provider": "Hermite",
            "completion": "two-hard-boundary middle gap",
            "bridge_coordinate": "log_temperature",
            "quality_factor": MIDDLE_LOG_T_HERMITE_QUALITY_FACTOR,
        },
        inherit_boundary_quality=False,
    )
    return PsatCanonicalizationInputs(
        relations=(relation,),
        metadata={"log_temperature_hermite_middle": True},
    )


def _collect_anchored_ambrose_walton_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_min: Optional[float] = None,
    T_critical: Optional[float] = None,
    P_critical_bar: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    """Provide a C1 Ambrose-Walton completion above a hard source segment."""
    critical_temperature = _property_candidate(
        adapter,
        "Tc",
        explicit_value=T_critical,
    )
    critical_pressure = _property_candidate(
        adapter,
        "Pc",
        explicit_value=P_critical_bar,
    )
    if critical_temperature is None or critical_pressure is None:
        return PsatCanonicalizationInputs()

    Tc = _positive_resolution_value(critical_temperature)
    Pc_bar = _positive_resolution_value(critical_pressure)
    critical_qualities = (
        _optional_quality(critical_temperature.quality),
        _optional_quality(critical_pressure.quality),
    )
    if (
        Tc is None
        or Pc_bar is None
        or any(quality is None for quality in critical_qualities)
        or min(critical_qualities)
        < AMBROSE_WALTON_HARD_BOUNDARY_MINIMUM_PROPERTY_QUALITY
    ):
        return PsatCanonicalizationInputs()
    critical_quality_factor = (
        LOW_QUALITY_CRITICAL_PR_QUALITY_FACTOR
        if min(critical_qualities)
        < AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY
        else 1.0
    )
    warnings = [
        warning
        for warning in (
            _property_admission_warning(
                "Tc",
                critical_temperature,
            ),
            _property_admission_warning(
                "Pc",
                critical_pressure,
            ),
        )
        if warning is not None
    ]
    if critical_quality_factor < 1.0:
        warnings.append(
            "Anchored upper Ambrose-Walton is using Tc/Pc below quality "
            f"{AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY:.2f}; retained with "
            f"quality factor {critical_quality_factor:.3f}"
        )

    acentric_factor = _property_candidate(adapter, "omega")
    omega = (
        _finite_resolution_value(acentric_factor)
        if acentric_factor is not None
        else None
    )
    omega_quality = (
        _optional_quality(acentric_factor.quality)
        if acentric_factor is not None
        else None
    )
    omega_is_admissible = (
        omega is not None
        and -0.5 <= omega <= 2.0
        and omega_quality is not None
        and omega_quality
        >= AMBROSE_WALTON_PREFERRED_PROPERTY_QUALITY
        and not _resolution_is_estimated(acentric_factor)
    )
    fixed_omega = omega if omega_is_admissible else None
    relation_quality_inputs = list(critical_qualities)
    if omega_is_admissible:
        relation_quality_inputs.append(omega_quality)
    relation_quality = (
        min(relation_quality_inputs)
        * AMBROSE_WALTON_QUALITY_FACTOR
        * critical_quality_factor
    )
    omega_anchor_temperature = (
        AMBROSE_WALTON_MINIMUM_REDUCED_TEMPERATURE * Tc
    )
    relation_T_min = omega_anchor_temperature
    requested_T_min = _valid_subcritical_temperature(T_min, Tc)
    if requested_T_min is None:
        boiling_candidate = _qualified_boiling_point_candidate(
            adapter,
            explicit_value=None,
        )
        requested_T_min = _valid_subcritical_temperature(
            _positive_resolution_value(boiling_candidate),
            Tc,
        )
    if requested_T_min is not None:
        relation_T_min = requested_T_min
    critical_context = {
        "Tc_source": critical_temperature.source,
        "Tc_method": critical_temperature.method,
        "Tc_quality": critical_temperature.quality,
        "Pc_source": critical_pressure.source,
        "Pc_method": critical_pressure.method,
        "Pc_quality": critical_pressure.quality,
        "omega_source": (
            acentric_factor.source if acentric_factor is not None else None
        ),
        "omega_method": (
            acentric_factor.method if acentric_factor is not None else None
        ),
        "omega_quality": (
            acentric_factor.quality if acentric_factor is not None else None
        ),
        "omega_basis": (
            "admissible_component_property"
            if omega_is_admissible
            else "resolved_from_selected_hard_curve"
        ),
    }

    def bind_anchored_aw(conditions):
        left = conditions.left
        critical = conditions.anchor("Tc")
        if left is None or critical is None:
            raise PsatCanonicalizationError(
                "Anchored Ambrose-Walton requires a hard left boundary and Tc anchor"
            )
        tolerance = 1.0e-8 * max(Tc, 1.0)
        if abs(critical.temperature - Tc) > tolerance:
            raise PsatCanonicalizationError(
                "Anchored Ambrose-Walton Tc conflicts with the canonical critical anchor"
            )
        if abs(critical.ln_pressure - math.log(Pc_bar)) > 1.0e-8:
            raise PsatCanonicalizationError(
                "Anchored Ambrose-Walton Pc conflicts with the canonical critical anchor"
            )
        if abs(conditions.T_max - Tc) > tolerance:
            raise PsatCanonicalizationError(
                "Anchored Ambrose-Walton only completes the terminal gap ending at Tc"
            )
        effective_omega = fixed_omega
        effective_omega_quality = omega_quality
        effective_omega_basis = "admissible_component_property"
        omega_quality_penalty = 1.0
        if effective_omega is None:
            omega_anchor = conditions.anchor("AW_omega_at_Tr_0.7")
            if omega_anchor is not None:
                effective_omega = _acentric_factor_from_definition(
                    omega_anchor.ln_pressure,
                    Pc_bar,
                )
                effective_omega_quality = omega_anchor.quality
                effective_omega_basis = (
                    "physical_definition_from_hard_segment_at_Tr_0.7"
                )
            else:
                effective_omega = _ambrose_walton_omega_from_pressure(
                    left.temperature,
                    left.ln_pressure,
                    Tc,
                    Pc_bar,
                    preferred_omega=omega,
                )
                endpoint_reduced_temperature = left.temperature / Tc
                omega_quality_penalty = max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - (
                            AMBROSE_WALTON_MINIMUM_REDUCED_TEMPERATURE
                            - endpoint_reduced_temperature
                        )
                        / 2.0,
                    ),
                )
                effective_omega_quality = (
                    left.quality * omega_quality_penalty
                )
                effective_omega_basis = "aw_inversion_from_hard_endpoint"
        bound_quality = (
            min(
                *critical_qualities,
                effective_omega_quality,
            )
            * AMBROSE_WALTON_QUALITY_FACTOR
            * critical_quality_factor
        )
        ln_pressure, derivative = _c1_anchored_aw_upper_functions(
            left_temperature=left.temperature,
            left_ln_pressure=left.ln_pressure,
            left_slope=left.dln_pressure_dT,
            Tc=Tc,
            Pc_bar=Pc_bar,
            omega=effective_omega,
        )

        return BoundaryConditionedPsatEvaluation(
            ln_pressure_function=ln_pressure,
            derivative_function=derivative,
            quality=bound_quality,
            context={
                "bound_omega": effective_omega,
                "bound_omega_quality": effective_omega_quality,
                "bound_omega_basis": effective_omega_basis,
            },
            metadata={
                "bound_omega": effective_omega,
                "bound_omega_quality": effective_omega_quality,
                "bound_omega_basis": effective_omega_basis,
                "omega_quality_penalty": omega_quality_penalty,
                "critical_quality_factor": critical_quality_factor,
                "omega_endpoint_reduced_temperature": (
                    left.temperature / Tc
                    if effective_omega_basis == "aw_inversion_from_hard_endpoint"
                    else None
                ),
            },
        )

    relation = BoundaryConditionedPsatSegment(
        source="calculated",
        method="anchored_ambrose_walton_upper",
        segment_type=PsatSegmentType.COMPLETION,
        priority=int(PsatPriority.AMBROSE_WALTON),
        quality=relation_quality,
        boundary_requirement=PsatBoundaryRequirement.LEFT,
        T_min=relation_T_min,
        T_max=Tc,
        evaluation_factory=bind_anchored_aw,
        P_max_bar=Pc_bar,
        left_derivative_requirement=PsatDerivativeRequirement.REQUIRED,
        required_anchor_names=("Tc",),
        assembly_anchor_requirements=(
            ()
            if omega_is_admissible
            else (
                PsatAssemblyAnchorRequirement(
                    name="AW_omega_at_Tr_0.7",
                    temperature=omega_anchor_temperature,
                    allowed_segment_types=(
                        PsatSegmentType.PINNED,
                        PsatSegmentType.CANONICAL_OVERRIDE,
                    ),
                    required=False,
                ),
            )
        ),
        allowed_left_segment_types=(
            PsatSegmentType.PINNED,
            PsatSegmentType.CANONICAL_OVERRIDE,
        ),
        context={
            **_component_context(adapter.component),
            **critical_context,
        },
        metadata={
            "provider": "Ambrose-Walton",
            "completion": "C1 hard-boundary correction",
            "minimum_reduced_temperature": (
                relation_T_min / Tc
            ),
            "omega_anchor_reduced_temperature": (
                AMBROSE_WALTON_MINIMUM_REDUCED_TEMPERATURE
            ),
            "Tc_K": Tc,
            "Pc_bar": Pc_bar,
            "omega": fixed_omega,
            "omega_basis": critical_context["omega_basis"],
            "quality_factor": AMBROSE_WALTON_QUALITY_FACTOR,
            "critical_quality_factor": critical_quality_factor,
        },
    )
    return PsatCanonicalizationInputs(
        relations=(relation,),
        warnings=tuple(warnings),
        metadata={"anchored_ambrose_walton_upper": True},
    )


def _collect_anchored_ambrose_walton_lower_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_min: Optional[float] = None,
    T_critical: Optional[float] = None,
    P_critical_bar: Optional[float] = None,
    T_boiling: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    """Provide a hard-boundary-matched AW completion down to 0.25 bar."""
    critical_temperature = _property_candidate(
        adapter,
        "Tc",
        explicit_value=T_critical,
    )
    critical_pressure = _property_candidate(
        adapter,
        "Pc",
        explicit_value=P_critical_bar,
    )
    if critical_temperature is None or critical_pressure is None:
        return PsatCanonicalizationInputs()

    Tc = _positive_resolution_value(critical_temperature)
    Pc_bar = _positive_resolution_value(critical_pressure)
    critical_qualities = (
        _optional_quality(critical_temperature.quality),
        _optional_quality(critical_pressure.quality),
    )
    if (
        Tc is None
        or Pc_bar is None
        or any(quality is None for quality in critical_qualities)
        or min(critical_qualities)
        < AMBROSE_WALTON_HARD_BOUNDARY_MINIMUM_PROPERTY_QUALITY
    ):
        return PsatCanonicalizationInputs()
    critical_quality_factor = (
        LOW_QUALITY_CRITICAL_PR_QUALITY_FACTOR
        if min(critical_qualities)
        < AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY
        else 1.0
    )
    warnings = [
        warning
        for warning in (
            _property_admission_warning(
                "Tc",
                critical_temperature,
            ),
            _property_admission_warning(
                "Pc",
                critical_pressure,
            ),
        )
        if warning is not None
    ]
    if critical_quality_factor < 1.0:
        warnings.append(
            "Anchored lower Ambrose-Walton is using Tc/Pc below quality "
            f"{AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY:.2f}; retained with "
            f"quality factor {critical_quality_factor:.3f}"
        )

    acentric_factor = _property_candidate(adapter, "omega")
    omega = (
        _finite_resolution_value(acentric_factor)
        if acentric_factor is not None
        else None
    )
    omega_quality = (
        _optional_quality(acentric_factor.quality)
        if acentric_factor is not None
        else None
    )
    omega_is_admissible = (
        omega is not None
        and -0.5 <= omega <= 2.0
        and omega_quality is not None
        and omega_quality
        >= AMBROSE_WALTON_PREFERRED_PROPERTY_QUALITY
        and not _resolution_is_estimated(acentric_factor)
    )
    soft_omega_is_admissible = (
        omega is not None
        and -0.5 <= omega <= 2.0
        and omega_quality is not None
        and AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY
        <= omega_quality
        < AMBROSE_WALTON_PREFERRED_PROPERTY_QUALITY
    )
    if soft_omega_is_admissible:
        warning = _property_admission_warning(
            "omega",
            acentric_factor,
        )
        if warning is not None:
            warnings.append(warning)
    fixed_omega = omega if omega_is_admissible else None
    relation_T_min = _valid_subcritical_temperature(T_min, Tc)
    if relation_T_min is None:
        relation_T_min = max(1.0, 0.15 * Tc)
    omega_anchor_temperature = (
        AMBROSE_WALTON_MINIMUM_REDUCED_TEMPERATURE * Tc
    )
    relation_quality = (
        min(critical_qualities)
        * LOWER_AMBROSE_WALTON_QUALITY_FACTOR
        * critical_quality_factor
    )
    boiling_candidate = _qualified_boiling_point_candidate(
        adapter,
        explicit_value=T_boiling,
    )
    boiling_temperature = _positive_resolution_value(boiling_candidate)
    hvap_plan = _frozen_hvap_plan(
        adapter,
        T_min=T_min,
        T_critical=Tc,
        T_boiling=boiling_temperature,
    )
    lower_switch_pressure = (
        adapter.completion_pressure_floor
        if hvap_plan is None
        else hvap_plan.switch_pressure_bar
    )

    def bind_lower_aw(conditions):
        right = conditions.right
        if right is None:
            raise PsatCanonicalizationError(
                "Lower Ambrose-Walton requires a hard right boundary"
            )
        if (
            lower_switch_pressure is not None
            and right.ln_pressure <= math.log(
                lower_switch_pressure
            ) + 1.0e-10
        ):
            raise PsatCanonicalizationError(
                "Lower Ambrose-Walton requires a hard boundary above "
                f"{lower_switch_pressure:g} bar"
            )
        tolerance = 1.0e-8 * max(Tc, 1.0)
        if right.temperature >= Tc - tolerance:
            raise PsatCanonicalizationError(
                "Lower Ambrose-Walton requires a subcritical hard boundary"
            )

        endpoint_omega = _ambrose_walton_omega_from_pressure(
            right.temperature,
            right.ln_pressure,
            Tc,
            Pc_bar,
            preferred_omega=fixed_omega,
        )
        boiling_anchor = conditions.anchor("Tb")
        use_separate_anchor = (
            right.derivative_basis is not PsatDerivativeBasis.ANALYTIC
            or (
                boiling_anchor is not None
                and right.temperature > boiling_anchor.temperature + tolerance
            )
        )
        route = "endpoint_slope_dynamic_omega"
        quality_factor = LOWER_AMBROSE_WALTON_QUALITY_FACTOR
        consumed_anchor_quality = None
        anchor_temperature = None
        anchor_omega = None
        omega_slope = None
        correction_basis = None

        if use_separate_anchor:
            if (
                boiling_anchor is not None
                and abs(
                    boiling_anchor.temperature - right.temperature
                ) > tolerance
            ):
                anchor_temperature = boiling_anchor.temperature
                anchor_omega = _ambrose_walton_omega_from_pressure(
                    boiling_anchor.temperature,
                    boiling_anchor.ln_pressure,
                    Tc,
                    Pc_bar,
                    preferred_omega=fixed_omega,
                )
                consumed_anchor_quality = boiling_anchor.quality
                route = "endpoint_to_tb_effective_omega"
            elif fixed_omega is not None and abs(
                omega_anchor_temperature - right.temperature
            ) > tolerance:
                anchor_temperature = omega_anchor_temperature
                anchor_omega = fixed_omega
                consumed_anchor_quality = omega_quality
                route = "endpoint_to_physical_omega_at_Tr_0.7"
            else:
                omega_anchor = conditions.anchor(
                    "lower_AW_omega_at_Tr_0.7"
                )
                if (
                    omega_anchor is not None
                    and abs(
                        omega_anchor.temperature - right.temperature
                    ) > tolerance
                ):
                    anchor_temperature = omega_anchor.temperature
                    anchor_omega = _acentric_factor_from_definition(
                        omega_anchor.ln_pressure,
                        Pc_bar,
                    )
                    consumed_anchor_quality = omega_anchor.quality
                    route = "endpoint_to_hard_psat_at_Tr_0.7"
                elif soft_omega_is_admissible and abs(
                    omega_anchor_temperature - right.temperature
                ) > tolerance:
                    anchor_temperature = omega_anchor_temperature
                    anchor_omega = omega
                    consumed_anchor_quality = omega_quality
                    route = "endpoint_to_soft_physical_omega_at_Tr_0.7"
                else:
                    raise PsatCanonicalizationError(
                        "Lower Ambrose-Walton needs Tb or omega at Tr=0.7 "
                        "for a non-analytic hard derivative"
                    )

            omega_slope = (
                anchor_omega - endpoint_omega
            ) / (
                anchor_temperature - right.temperature
            )
            quality_factor = LOWER_AMBROSE_WALTON_ANCHORED_QUALITY_FACTOR
        else:
            sensitivity = _ambrose_walton_omega_sensitivity(
                right.temperature,
                Tc,
                endpoint_omega,
            )
            if abs(sensitivity) <= 1.0e-10:
                raise PsatCanonicalizationError(
                    "Lower Ambrose-Walton cannot infer an omega slope "
                    "at the hard boundary"
                )
            constant_omega_slope = _ambrose_walton_dln_pressure_dT(
                right.temperature,
                Tc,
                endpoint_omega,
            )
            omega_slope = (
                right.dln_pressure_dT - constant_omega_slope
            ) / sensitivity

        def effective_omega(T: float) -> float:
            return endpoint_omega + omega_slope * (
                T - right.temperature
            )

        def baseline_ln_pressure(T: float) -> float:
            return _ambrose_walton_ln_pressure(
                T,
                Tc,
                Pc_bar,
                effective_omega(T),
            )

        def baseline_derivative(T: float) -> float:
            local_omega = effective_omega(T)
            return (
                _ambrose_walton_dln_pressure_dT(
                    T,
                    Tc,
                    local_omega,
                )
                + _ambrose_walton_omega_sensitivity(
                    T,
                    Tc,
                    local_omega,
                )
                * omega_slope
            )

        baseline_boundary_slope = baseline_derivative(right.temperature)
        slope_delta = right.dln_pressure_dT - baseline_boundary_slope
        correction = lambda _T: 0.0
        correction_derivative = lambda _T: 0.0
        if use_separate_anchor and boiling_anchor is not None and (
            right.temperature > boiling_anchor.temperature + tolerance
        ):
            span = right.temperature - boiling_anchor.temperature
            value_delta = (
                right.ln_pressure
                - baseline_ln_pressure(right.temperature)
            )
            endpoint_coordinate_slope = slope_delta * span
            coefficient_3 = (
                10.0 * value_delta
                - 4.0 * endpoint_coordinate_slope
            )
            coefficient_4 = (
                7.0 * endpoint_coordinate_slope
                - 15.0 * value_delta
            )
            coefficient_5 = (
                6.0 * value_delta
                - 3.0 * endpoint_coordinate_slope
            )

            def correction(T: float) -> float:
                if T <= boiling_anchor.temperature:
                    return 0.0
                coordinate = (
                    T - boiling_anchor.temperature
                ) / span
                return (
                    coefficient_3 * coordinate**3
                    + coefficient_4 * coordinate**4
                    + coefficient_5 * coordinate**5
                )

            def correction_derivative(T: float) -> float:
                if T <= boiling_anchor.temperature:
                    return 0.0
                coordinate = (
                    T - boiling_anchor.temperature
                ) / span
                return (
                    3.0 * coefficient_3 * coordinate**2
                    + 4.0 * coefficient_4 * coordinate**3
                    + 5.0 * coefficient_5 * coordinate**4
                ) / span

            correction_basis = "Tb_localized_quintic"
        elif use_separate_anchor:
            coefficient = -slope_delta * right.temperature**2
            correction = lambda T: coefficient * (
                1.0 / T - 1.0 / right.temperature
            )
            correction_derivative = lambda T: -coefficient / T**2
            correction_basis = "affine_inverse_temperature"

        def ln_pressure(T: float) -> float:
            return baseline_ln_pressure(T) + correction(T)

        def derivative(T: float) -> float:
            return baseline_derivative(T) + correction_derivative(T)

        quality_inputs = [*critical_qualities, right.quality]
        if consumed_anchor_quality is not None:
            quality_inputs.append(consumed_anchor_quality)
        lower_span_pressure = lower_switch_pressure
        if lower_span_pressure is None:
            lower_span_pressure = math.exp(
                ln_pressure(conditions.T_min)
            )
        pressure_span_decades = max(
            0.0,
            math.log10(
                math.exp(right.ln_pressure)
                / lower_span_pressure
            ),
        )
        span_quality_factor = max(
            LOWER_AW_MINIMUM_SPAN_QUALITY_FACTOR,
            1.0
            - LOWER_AW_SPAN_PENALTY_PER_DECADE
            * pressure_span_decades,
        )
        bound_quality = (
            min(quality_inputs)
            * quality_factor
            * span_quality_factor
            * critical_quality_factor
        )
        return BoundaryConditionedPsatEvaluation(
            ln_pressure_function=ln_pressure,
            derivative_function=derivative,
            quality=bound_quality,
            context={
                "bound_endpoint_omega": endpoint_omega,
                "bound_omega_slope_per_K": omega_slope,
                "bound_lower_aw_route": route,
            },
            metadata={
                "bound_endpoint_omega": endpoint_omega,
                "bound_omega_slope_per_K": omega_slope,
                "bound_lower_aw_route": route,
                "bound_anchor_temperature_K": anchor_temperature,
                "bound_anchor_omega": anchor_omega,
                "bound_correction_basis": correction_basis,
                "hard_derivative_basis": right.derivative_basis.value,
                "quality_factor": quality_factor,
                "pressure_span_decades": pressure_span_decades,
                "span_quality_factor": span_quality_factor,
                "critical_quality_factor": critical_quality_factor,
            },
        )

    relation = BoundaryConditionedPsatSegment(
        source="calculated",
        method="anchored_ambrose_walton_lower",
        segment_type=PsatSegmentType.COMPLETION,
        priority=int(PsatPriority.AMBROSE_WALTON),
        quality=relation_quality,
        boundary_requirement=PsatBoundaryRequirement.RIGHT,
        T_min=relation_T_min,
        T_max=Tc,
        evaluation_factory=bind_lower_aw,
        P_min_bar=lower_switch_pressure,
        right_derivative_requirement=PsatDerivativeRequirement.REQUIRED,
        assembly_anchor_requirements=(
            ()
            if fixed_omega is not None
            else (
                PsatAssemblyAnchorRequirement(
                    name="lower_AW_omega_at_Tr_0.7",
                    temperature=omega_anchor_temperature,
                    allowed_segment_types=(
                        PsatSegmentType.PINNED,
                        PsatSegmentType.CANONICAL_OVERRIDE,
                    ),
                    required=False,
                ),
            )
        ),
        allowed_right_segment_types=(
            PsatSegmentType.PINNED,
            PsatSegmentType.CANONICAL_OVERRIDE,
        ),
        context={
            **_component_context(adapter.component),
            "Tc_source": critical_temperature.source,
            "Tc_method": critical_temperature.method,
            "Tc_quality": critical_temperature.quality,
            "Pc_source": critical_pressure.source,
            "Pc_method": critical_pressure.method,
            "Pc_quality": critical_pressure.quality,
        },
        metadata={
            "provider": "Ambrose-Walton",
            "completion": "hard-boundary lower continuation",
            "switch_pressure_bar": lower_switch_pressure,
            "switch_basis": (
                "frozen_hvap_quality"
                if hvap_plan is not None
                else "continue_aw_to_domain_floor"
            ),
            "hvap_quality": (
                None if hvap_plan is None else hvap_plan.quality
            ),
            "hvap_method": (
                None if hvap_plan is None else hvap_plan.method
            ),
            "Tc_K": Tc,
            "Pc_bar": Pc_bar,
            "dynamic_omega_quality_factor": (
                LOWER_AMBROSE_WALTON_QUALITY_FACTOR
            ),
            "separate_anchor_quality_factor": (
                LOWER_AMBROSE_WALTON_ANCHORED_QUALITY_FACTOR
            ),
            "critical_quality_factor": critical_quality_factor,
        },
        inherit_boundary_quality=False,
    )
    return PsatCanonicalizationInputs(
        relations=(relation,),
        warnings=tuple(warnings),
        metadata={"anchored_ambrose_walton_lower": True},
    )


def _collect_deep_vacuum_inputs(
    adapter: PsatCanonicalizationAdapter,
    *,
    T_min: Optional[float] = None,
    T_critical: Optional[float] = None,
    P_critical_bar: Optional[float] = None,
    T_boiling: Optional[float] = None,
    **_state: Any,
) -> PsatCanonicalizationInputs:
    critical_temperature = _property_candidate(
        adapter,
        "Tc",
        explicit_value=T_critical,
    )
    critical_pressure = _property_candidate(
        adapter,
        "Pc",
        explicit_value=P_critical_bar,
    )
    if critical_temperature is None or critical_pressure is None:
        return PsatCanonicalizationInputs()
    Tc = _positive_resolution_value(critical_temperature)
    Pc_bar = _positive_resolution_value(critical_pressure)
    critical_qualities = (
        _optional_quality(critical_temperature.quality),
        _optional_quality(critical_pressure.quality),
    )
    boiling_candidate = adapter._no_hard_boiling_candidate
    boiling_temperature = (
        _positive_resolution_value(boiling_candidate)
        if boiling_candidate is not None
        else T_boiling
    )
    if boiling_temperature is None:
        boiling_temperature = _component_numeric_value(
            adapter.component,
            "Tb",
        )
    plan = _frozen_hvap_plan(
        adapter,
        T_min=T_min,
        T_critical=Tc,
        T_boiling=boiling_temperature,
    )
    if (
        plan is None
        or Tc is None
        or Pc_bar is None
    ):
        return PsatCanonicalizationInputs()
    low_quality_criticals = any(
        quality is None
        or quality < AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY
        for quality in critical_qualities
    )
    critical_pr_quality_factor = (
        LOW_QUALITY_CRITICAL_PR_QUALITY_FACTOR
        if low_quality_criticals
        else 1.0
    )
    admitted_critical_qualities = [
        quality for quality in critical_qualities
        if quality is not None
        and quality >= AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY
    ]
    warnings = [
        warning
        for warning in (
            _property_admission_warning(
                "Tc",
                critical_temperature,
            ),
            _property_admission_warning(
                "Pc",
                critical_pressure,
            ),
        )
        if warning is not None
    ]
    if low_quality_criticals:
        warnings.append(
            "Deep PR/Clapeyron is using Tc/Pc below quality 0.90; "
            f"retained with quality factor "
            f"{LOW_QUALITY_CRITICAL_PR_QUALITY_FACTOR:.3f}"
        )

    acentric_factor = _property_candidate(adapter, "omega")
    omega = (
        _finite_resolution_value(acentric_factor)
        if acentric_factor is not None
        else None
    )
    omega_quality = (
        _optional_quality(acentric_factor.quality)
        if acentric_factor is not None
        else None
    )
    omega_is_admissible = (
        omega is not None
        and -0.5 <= omega <= 2.0
        and omega_quality is not None
        and omega_quality >= AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY
    )
    physical_omega = omega if omega_is_admissible else None
    if omega_is_admissible:
        warning = _property_admission_warning(
            "omega",
            acentric_factor,
        )
        if warning is not None:
            warnings.append(warning)
    acid_identifier = _monocarboxylic_acid_identifier(adapter.component)
    is_monoacid = acid_identifier is not None

    def bind_clapeyron(conditions):
        right = conditions.right
        if right is None:
            raise PsatCanonicalizationError(
                "Deep-vacuum Clapeyron requires a right boundary"
            )
        boundary_pressure = math.exp(right.ln_pressure)
        if boundary_pressure > plan.switch_pressure_bar * (1.0 + 1.0e-8):
            raise PsatCanonicalizationError(
                "Deep-vacuum Clapeyron boundary is above its qualified switch"
            )
        if not plan.T_min <= conditions.T_min < right.temperature <= plan.T_max:
            raise PsatCanonicalizationError(
                "Frozen Hvap does not cover the complete deep-vacuum interval"
            )
        boundary_hvap = plan.evaluator_J_mol(right.temperature)
        route = "calibrated_monoacid_dimer"
        dimer_entropy = None
        dimer_extent_at_boundary = None
        delta_z_basis = None

        if is_monoacid:
            required_enthalpy = (
                R
                * right.temperature**2
                * right.dln_pressure_dT
            )
            if required_enthalpy <= boundary_hvap:
                dimer_entropy = DIMER_SUPPRESSED_ENTROPY_J_MOL_K
                dimer_extent_at_boundary = 0.0
            else:
                dimer_extent_at_boundary = min(
                    0.499999,
                    max(
                        0.0,
                        1.0 - boundary_hvap / required_enthalpy,
                    ),
                )
                pressure_function = (
                    dimer_extent_at_boundary
                    * (1.0 - dimer_extent_at_boundary)
                    / (1.0 - 2.0 * dimer_extent_at_boundary) ** 2
                )
                equilibrium_constant = (
                    pressure_function / boundary_pressure
                )
                dimer_entropy = (
                    R * math.log(equilibrium_constant)
                    + DIMER_ENTHALPY_J_MOL / right.temperature
                )
        else:
            route = (
                "peng_robinson_delta_z"
                if physical_omega is not None
                else "ideal_delta_z"
            )
            delta_z_basis = route

        def slope(T: float, ln_pressure: float) -> float:
            hvap = plan.evaluator_J_mol(T)
            if is_monoacid:
                extent = _dimer_extent(
                    T,
                    ln_pressure,
                    dimer_entropy,
                )
                return hvap / (
                    R * T**2 * (1.0 - extent)
                )
            delta_z = (
                1.0
                if physical_omega is None
                else _peng_robinson_delta_z_or_ideal(
                    T,
                    math.exp(ln_pressure),
                    Tc,
                    Pc_bar,
                    physical_omega,
                )
            )
            return hvap / (R * T**2 * delta_z)

        boundary_slope = slope(
            right.temperature,
            right.ln_pressure,
        )
        relative_slope_mismatch = abs(
            boundary_slope / right.dln_pressure_dT - 1.0
        )

        solution = solve_ivp(
            lambda T, state: [slope(T, state[0])],
            (right.temperature, conditions.T_min),
            [right.ln_pressure],
            rtol=2.0e-9,
            atol=2.0e-11,
            dense_output=True,
            max_step=max(
                (right.temperature - conditions.T_min) / 50.0,
                0.1,
            ),
        )
        if not solution.success:
            raise PsatCanonicalizationError(
                "Deep-vacuum Clapeyron integration failed"
            )

        def ln_pressure(T: float) -> float:
            value = float(solution.sol(T)[0])
            if not math.isfinite(value):
                raise PsatCanonicalizationError(
                    "Deep-vacuum Clapeyron returned a non-finite pressure"
                )
            return value

        def derivative(T: float) -> float:
            return slope(T, ln_pressure(T))

        quality_inputs = [plan.quality, right.quality]
        quality_inputs.extend(
            quality for quality in critical_qualities
            if quality is not None
            and quality >= AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY
        )
        if physical_omega is not None and not is_monoacid:
            quality_inputs.append(omega_quality)
        quality_factor = (
            DIMER_CLAPEYRON_QUALITY_FACTOR
            if is_monoacid
            else DEEP_CLAPEYRON_QUALITY_FACTOR
        ) * critical_pr_quality_factor
        return BoundaryConditionedPsatEvaluation(
            ln_pressure_function=ln_pressure,
            derivative_function=derivative,
            quality=min(quality_inputs) * quality_factor,
            context={
                "deep_completion_route": route,
                "hvap_source": plan.source,
                "hvap_method": plan.method,
            },
            metadata={
                "deep_completion_route": route,
                "delta_z_basis": delta_z_basis,
                "incoming_slope_mismatch": relative_slope_mismatch,
                "dimer_entropy_J_mol_K": dimer_entropy,
                "dimer_extent_at_boundary": dimer_extent_at_boundary,
                "acid_identifier": acid_identifier,
                "hvap_source": plan.source,
                "hvap_method": plan.method,
                "hvap_quality": plan.quality,
                "hvap_relation_class": plan.relation_class,
                "hvap_sample_count": plan.sample_count,
                "quality_factor": quality_factor,
                "critical_pr_quality_factor": (
                    critical_pr_quality_factor
                ),
            },
            allow_junction_slope_mismatch=True,
        )

    clapeyron = BoundaryConditionedPsatSegment(
        source="calculated",
        method="resolved_hvap_clapeyron_lower",
        segment_type=PsatSegmentType.COMPLETION,
        priority=int(PsatPriority.CLAPEYRON),
        quality=(
            plan.quality
            * DEEP_CLAPEYRON_QUALITY_FACTOR
            * critical_pr_quality_factor
        ),
        boundary_requirement=PsatBoundaryRequirement.RIGHT,
        T_min=plan.T_min,
        T_max=Tc,
        evaluation_factory=bind_clapeyron,
        P_max_bar=plan.switch_pressure_bar,
        right_derivative_requirement=PsatDerivativeRequirement.REQUIRED,
        allowed_right_segment_types=(
            PsatSegmentType.PINNED,
            PsatSegmentType.CANONICAL_OVERRIDE,
            PsatSegmentType.COMPLETION,
        ),
        metadata={
            "provider": "Clapeyron",
            "switch_pressure_bar": plan.switch_pressure_bar,
            "hvap_source": plan.source,
            "hvap_method": plan.method,
            "hvap_quality": plan.quality,
            "hvap_relation_class": plan.relation_class,
            "critical_pr_quality_factor": (
                critical_pr_quality_factor
            ),
            "delta_z_policy": (
                "monoacid_dimer"
                if is_monoacid
                else "peng_robinson_or_ideal"
            ),
        },
        inherit_boundary_quality=False,
    )

    def bind_aw_fallback(conditions):
        right = conditions.right
        if right is None:
            raise PsatCanonicalizationError(
                "Deep AW fallback requires a right boundary"
            )
        boundary_pressure = math.exp(right.ln_pressure)
        if boundary_pressure > plan.switch_pressure_bar * (1.0 + 1.0e-8):
            raise PsatCanonicalizationError(
                "Deep AW fallback boundary is above its qualified switch"
            )
        endpoint_omega = _ambrose_walton_omega_from_pressure(
            right.temperature,
            right.ln_pressure,
            Tc,
            Pc_bar,
            preferred_omega=physical_omega,
        )
        sensitivity = _ambrose_walton_omega_sensitivity(
            right.temperature,
            Tc,
            endpoint_omega,
        )
        if abs(sensitivity) <= 1.0e-10:
            raise PsatCanonicalizationError(
                "Deep AW fallback cannot infer an omega slope"
            )
        omega_slope = (
            right.dln_pressure_dT
            - _ambrose_walton_dln_pressure_dT(
                right.temperature,
                Tc,
                endpoint_omega,
            )
        ) / sensitivity

        def effective_omega(T: float) -> float:
            return endpoint_omega + omega_slope * (
                T - right.temperature
            )

        def ln_pressure(T: float) -> float:
            return _ambrose_walton_ln_pressure(
                T,
                Tc,
                Pc_bar,
                effective_omega(T),
            )

        def derivative(T: float) -> float:
            local_omega = effective_omega(T)
            return (
                _ambrose_walton_dln_pressure_dT(
                    T,
                    Tc,
                    local_omega,
                )
                + _ambrose_walton_omega_sensitivity(
                    T,
                    Tc,
                    local_omega,
                )
                * omega_slope
            )

        return BoundaryConditionedPsatEvaluation(
            ln_pressure_function=ln_pressure,
            derivative_function=derivative,
            quality=(
                min([right.quality, *admitted_critical_qualities])
                * DEEP_AW_FALLBACK_QUALITY_FACTOR
                * critical_pr_quality_factor
            ),
            metadata={
                "deep_completion_route": "dynamic_omega_fallback",
                "bound_endpoint_omega": endpoint_omega,
                "bound_omega_slope_per_K": omega_slope,
                "quality_factor": DEEP_AW_FALLBACK_QUALITY_FACTOR,
                "critical_pr_quality_factor": (
                    critical_pr_quality_factor
                ),
            },
        )

    aw_fallback = BoundaryConditionedPsatSegment(
        source="calculated",
        method="dynamic_omega_deep_fallback",
        segment_type=PsatSegmentType.FALLBACK,
        priority=int(PsatPriority.CLAPEYRON) - 10,
        quality=(
            min(admitted_critical_qualities or [1.0])
            * DEEP_AW_FALLBACK_QUALITY_FACTOR
            * critical_pr_quality_factor
        ),
        boundary_requirement=PsatBoundaryRequirement.RIGHT,
        T_min=plan.T_min,
        T_max=Tc,
        evaluation_factory=bind_aw_fallback,
        P_max_bar=plan.switch_pressure_bar,
        right_derivative_requirement=PsatDerivativeRequirement.REQUIRED,
        allowed_right_segment_types=(
            PsatSegmentType.PINNED,
            PsatSegmentType.CANONICAL_OVERRIDE,
            PsatSegmentType.COMPLETION,
        ),
        metadata={
            "provider": "Ambrose-Walton",
            "completion": "deep-vacuum fallback",
        },
        inherit_boundary_quality=False,
    )
    return PsatCanonicalizationInputs(
        relations=(clapeyron, aw_fallback),
        warnings=tuple(warnings),
        metadata={
            "deep_vacuum_hvap_method": plan.method,
            "deep_vacuum_switch_pressure_bar": plan.switch_pressure_bar,
        },
    )


def _pfd_psat_correlation(component: Any) -> Optional[Mapping[str, Any]]:
    correlations = _component_value(component, "property_correlations")
    if not isinstance(correlations, Mapping):
        return None
    correlation = correlations.get("Psat")
    if not isinstance(correlation, Mapping) or not correlation.get("_pfd_override"):
        return None
    return correlation


def _local_psat_correlation(component: Any) -> Optional[Mapping[str, Any]]:
    correlations = _component_value(component, "property_correlations")
    if not isinstance(correlations, Mapping):
        return None
    correlation = correlations.get("Psat")
    if (
        not isinstance(correlation, Mapping)
        or correlation.get("_pfd_override")
    ):
        return None
    return correlation


def _curated_antoine_coefficients(
    adapter: PsatCanonicalizationAdapter,
) -> Optional[tuple[float, float, float, float, float]]:
    fields = ("antoine_A", "antoine_B", "antoine_C", "antoine_Tmin", "antoine_Tmax")
    raw_values = tuple(adapter.component_value(field_name) for field_name in fields)
    if all(value is None for value in raw_values):
        return None
    if any(value is None for value in raw_values):
        raise PsatAdapterError(
            "Curated Antoine segment requires A, B, C, Tmin, and Tmax"
        )
    source_records = tuple(
        adapter.property_source(field_name)
        for field_name in (*fields, "Antoine")
    )
    if any(
        str(record.get("method") or "").lower() == "pfd_component_override"
        for record in source_records
    ):
        return None
    citation = str(adapter.component_value("antoine_source") or "").lower()
    if "data/antoine.txt" in citation or (
        "smith" in citation and "appendix b" in citation
    ):
        return None

    values = []
    for field_name, raw_value in zip(fields, raw_values):
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise PsatAdapterError(
                f"Curated Antoine {field_name} must be finite"
            ) from error
        if not math.isfinite(value):
            raise PsatAdapterError(f"Curated Antoine {field_name} must be finite")
        values.append(value)
    if values[3] <= 0.0 or values[4] <= values[3]:
        raise PsatAdapterError("Curated Antoine requires 0 < Tmin < Tmax")
    return tuple(values)


def _pfd_antoine_override_state(
    adapter: PsatCanonicalizationAdapter,
) -> tuple[frozenset[str], bool]:
    fields = (
        "antoine_A",
        "antoine_B",
        "antoine_C",
        "antoine_Tmin",
        "antoine_Tmax",
    )
    overridden = frozenset(
        field_name
        for field_name in fields
        if str(adapter.property_source(field_name).get("method") or "").lower()
        == "pfd_component_override"
    )
    aggregate_override = (
        str(adapter.property_source("Antoine").get("method") or "").lower()
        == "pfd_component_override"
    )
    return overridden, aggregate_override


def _curated_antoine_source_metadata(
    adapter: PsatCanonicalizationAdapter,
) -> Optional[Mapping[str, Any]]:
    overridden_fields, aggregate_override = _pfd_antoine_override_state(adapter)
    if overridden_fields or aggregate_override:
        return None
    antoine_record = adapter.property_source("Antoine")
    dataset_record = adapter.property_source("dataset")
    field_records = tuple(
        adapter.property_source(field_name)
        for field_name in (
            "antoine_A",
            "antoine_B",
            "antoine_C",
            "antoine_Tmin",
            "antoine_Tmax",
        )
    )
    citation = str(adapter.component_value("antoine_source") or "").strip().lower()
    citation_is_online = (
        citation.startswith("nist chemistry webbook")
        or citation.startswith("pubchem")
    )
    if any(
        _antoine_provenance_is_online(record)
        for record in (antoine_record, *field_records)
    ) or citation_is_online:
        return None
    component_source = str(adapter.component_value("source") or "").strip().lower()
    source_name = str(antoine_record.get("source") or "").strip().lower()
    method_name = str(antoine_record.get("method") or "").strip().lower()
    if source_name in {"local", "provided", "exact"}:
        return antoine_record
    if method_name in {"curated_antoine", "chemicals_json"}:
        return antoine_record
    if component_source == "local":
        return antoine_record
    if antoine_record:
        return None
    if dataset_record:
        return dataset_record
    return None


def _antoine_provenance_is_online(record: Mapping[str, Any]) -> bool:
    source_name = str(record.get("source") or "").strip().lower()
    method_name = str(record.get("method") or "").strip().lower()
    return source_name == "online" or any(
        token in method_name
        for token in ("online", "nist", "pubchem", "webbook")
    )


def _discarded_input_warning(
    adapter: PsatCanonicalizationAdapter,
    message: str,
) -> PsatCanonicalizationInputs:
    context = _component_context(adapter.component)
    identifier = context.get("symbol") or context.get("name") or context.get("CAS")
    warning = f"{message} for {identifier}" if identifier else message
    return PsatCanonicalizationInputs(warnings=(warning,))


def _antoine_functions(
    A: float,
    B: float,
    C: float,
    T_min: float,
    T_max: float,
    *,
    label: str,
) -> tuple[Callable[[float], float], Callable[[float], float], float, float]:
    if T_min <= 0.0 or T_max <= T_min:
        raise PsatAdapterError(f"{label} requires 0 < Tmin < Tmax")
    denominator_min = C + T_min - 273.15
    denominator_max = C + T_max - 273.15
    if B <= 0.0 or denominator_min == 0.0 or denominator_max == 0.0:
        raise PsatAdapterError(f"{label} correlation is not monotonic and finite")
    if min(denominator_min, denominator_max) < 0.0 < max(
        denominator_min,
        denominator_max,
    ):
        raise PsatAdapterError(f"{label} correlation is singular in its range")
    ln10 = math.log(10.0)

    def ln_pressure(T: float) -> float:
        denominator = C + T - 273.15
        if denominator == 0.0:
            raise PsatAdapterError(f"{label} Psat is singular at {T:g} K")
        return ln10 * (A - B / denominator)

    def derivative(T: float) -> float:
        denominator = C + T - 273.15
        if denominator == 0.0:
            raise PsatAdapterError(f"{label} Psat is singular at {T:g} K")
        return ln10 * B / denominator**2

    try:
        P_min_bar = math.exp(ln_pressure(T_min))
        P_max_bar = math.exp(ln_pressure(T_max))
    except OverflowError as error:
        raise PsatAdapterError(f"{label} endpoint pressure overflowed") from error
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (P_min_bar, P_max_bar)
    ):
        raise PsatAdapterError(f"{label} endpoint pressures are invalid")
    return ln_pressure, derivative, P_min_bar, P_max_bar


def _coolprop_reference(component: Any):
    component_props = {
        key: value
        for key in ("CAS", "cas")
        if (value := _component_value(component, key)) not in (None, "")
    }
    symbol = _component_value(component, "symbol") or ""
    return coolprop_reference_for(str(symbol), component_props)


def _validate_pfd_critical_metadata(
    correlation: Mapping[str, Any],
    T_critical: float,
    P_critical_bar: float,
    T_boiling: Optional[float],
) -> None:
    explicit_Tc = _optional_correlation_float(correlation, "Tc_K")
    explicit_Pc_bar = _optional_correlation_float(correlation, "Pc_bar")
    explicit_Pc_pa = _optional_correlation_float(correlation, "Pc_Pa")
    explicit_Tb = _optional_correlation_float(correlation, "Tb_K")
    comparisons = (
        ("Tc_K", explicit_Tc, T_critical),
        ("Pc_bar", explicit_Pc_bar, P_critical_bar),
        ("Pc_Pa", explicit_Pc_pa, P_critical_bar * 100000.0),
        ("Tb_K", explicit_Tb, T_boiling),
    )
    for name, explicit, target in comparisons:
        if explicit is None or target is None:
            continue
        tolerance = 1.0e-9 * max(abs(explicit), abs(target), 1.0)
        if abs(explicit - target) > tolerance:
            raise PsatAdapterError(
                f"PFD Psat {name}={explicit:g} conflicts with component value "
                f"{target:g}"
            )


def _component_value(component: Any, field_name: str) -> Any:
    if component is None:
        return None
    if isinstance(component, Mapping):
        return component.get(field_name)
    return getattr(component, field_name, None)


DEFAULT_PSAT_INPUT_METHODS: tuple[PsatInputMethod, ...] = (
    _collect_pfd_inputs,
    _collect_coolprop_inputs,
    _collect_local_correlation_inputs,
    _collect_curated_antoine_inputs,
    _collect_perry_2_8_inputs,
    _collect_trusted_other_antoine_inputs,
    _collect_textbook_antoine_inputs,
    _collect_perry_2_10_inputs,
    _collect_cached_nist_antoine_inputs,
    _collect_antoine_table_inputs,
    _collect_no_hard_fallback_inputs,
    _collect_log_temperature_middle_gap_inputs,
    _collect_anchored_ambrose_walton_inputs,
    _collect_anchored_ambrose_walton_lower_inputs,
    _collect_deep_vacuum_inputs,
)


DIRECT_PSAT_INPUT_METHODS: tuple[PsatInputMethod, ...] = (
    _collect_pfd_inputs,
    _collect_coolprop_inputs,
    _collect_local_correlation_inputs,
    _collect_curated_antoine_inputs,
    _collect_perry_2_8_inputs,
    _collect_trusted_other_antoine_inputs,
    _collect_textbook_antoine_inputs,
    _collect_perry_2_10_inputs,
    _collect_cached_nist_antoine_inputs,
    _collect_antoine_table_inputs,
)


def _property_candidate(
    adapter: PsatCanonicalizationAdapter,
    field_name: str,
    *,
    explicit_value: Optional[float] = None,
) -> Optional[PropertyResolutionResult]:
    raw_value = adapter.component_value(field_name)
    if raw_value is None and explicit_value is None:
        return None
    candidate = adapter._candidate_from_field(
        field_name,
        explicit_value if raw_value is None else raw_value,
    )
    if explicit_value is None:
        return candidate
    try:
        value = float(explicit_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return PropertyResolutionResult(
        value=value,
        source=candidate.source,
        method=candidate.method,
        quality=candidate.quality,
        notes=candidate.notes,
    )


def _property_admission_warning(
    property_name: str,
    candidate: Optional[PropertyResolutionResult],
) -> Optional[str]:
    if candidate is None:
        return None
    quality = _optional_quality(candidate.quality)
    if (
        quality is None
        or quality < AMBROSE_WALTON_MINIMUM_PROPERTY_QUALITY
        or quality >= AMBROSE_WALTON_PREFERRED_PROPERTY_QUALITY
    ):
        return None
    return (
        f"{property_name} admitted for Psat completion at quality "
        f"{quality:.3f} below preferred "
        f"{AMBROSE_WALTON_PREFERRED_PROPERTY_QUALITY:.3f}; "
        f"source={candidate.source}, method={candidate.method}"
    )


def _qualified_boiling_point_candidate(
    adapter: PsatCanonicalizationAdapter,
    *,
    explicit_value: Optional[float],
) -> Optional[PropertyResolutionResult]:
    """Return only a hard Tb allowed to anchor or validate Psat inputs."""
    candidate = _property_candidate(
        adapter,
        "Tb",
        explicit_value=explicit_value,
    )
    quality = (
        _optional_quality(candidate.quality)
        if candidate is not None
        else None
    )
    if (
        candidate is None
        or _positive_resolution_value(candidate) is None
        or (
            quality is not None
            and quality < PSAT_MINIMUM_BOILING_POINT_QUALITY
        )
        or _resolution_is_estimated(candidate)
    ):
        return None
    return candidate


def _admit_direct_psat_segment(
    adapter: PsatCanonicalizationAdapter,
    segment: PsatSegment,
    *,
    T_boiling: Optional[float],
    confidence_profile: Optional[DirectPsatConfidenceProfile],
    source_label: Optional[str] = None,
) -> tuple[Optional[PsatSegment], Optional[str]]:
    """Apply shared priority/Tb admission and emit coherent metadata."""
    boiling_candidate = _qualified_boiling_point_candidate(
        adapter,
        explicit_value=T_boiling,
    )
    boiling_temperature = _positive_resolution_value(boiling_candidate)

    def source_pressure(temperature: float) -> float:
        return math.exp(float(segment.ln_pressure_function(temperature)))

    decision = admit_direct_psat_segment(
        priority=segment.priority,
        intrinsic_quality=segment.quality,
        confidence_profile=confidence_profile,
        qualified_Tb=boiling_temperature,
        pressure_at_temperature=source_pressure,
        T_min=segment.T_min,
        T_max=segment.T_max,
        source_label=source_label or f"{segment.source}/{segment.method}",
        existing_exempt_handoff=segment.handoff_requirement.value,
        exemption_priority=int(PsatPriority.PERRY_2_8),
    )
    if not decision.admitted:
        return None, decision.warning
    metadata = dict(segment.metadata)
    metadata.update(_direct_psat_admission_metadata(
        decision,
        confidence_profile,
    ))
    return replace(
        segment,
        quality=float(decision.selected_quality),
        handoff_requirement=PsatHandoffRequirement(
            decision.handoff_requirement
        ),
        metadata=metadata,
    ), None


def _direct_psat_admission_metadata(
    decision: DirectPsatAdmissionDecision,
    confidence_profile: Optional[DirectPsatConfidenceProfile],
) -> dict[str, Any]:
    metadata = {
        "tb_validation_required": decision.tb_validation_required,
        "tb_validation_status": decision.validation_status,
        "tb_validation_pressure_bar": decision.pressure_bar,
        "tb_validation_relative_error": decision.relative_error,
        "quality_basis": decision.quality_basis,
    }
    if confidence_profile is not None:
        metadata.update({
            "standalone_unvalidated_quality": (
                confidence_profile.standalone_quality
            ),
            "higher_preference_overlap_quality": (
                confidence_profile.higher_priority_overlap_quality
            ),
            "hard_tb_validated_quality": confidence_profile.hard_tb_quality,
        })
    return metadata


def _boiling_point_candidate(
    adapter: PsatCanonicalizationAdapter,
    *,
    explicit_value: Optional[float],
    allow_estimation: bool,
) -> Optional[PropertyResolutionResult]:
    cache_key = (
        None if explicit_value is None else float(explicit_value),
        bool(allow_estimation),
    )
    if cache_key in adapter._boiling_point_candidates:
        return adapter._boiling_point_candidates[cache_key]
    candidate = _property_candidate(
        adapter,
        "Tb",
        explicit_value=explicit_value,
    )
    if (
        candidate is not None
        and _positive_resolution_value(candidate) is not None
        and _optional_quality(candidate.quality) is not None
        and (
            allow_estimation
            or _qualified_boiling_point_candidate(
                adapter,
                explicit_value=explicit_value,
            ) is not None
        )
    ):
        adapter._boiling_point_candidates[cache_key] = candidate
        return candidate

    identifiers = _component_identifier_candidates(adapter.component)
    if not identifiers:
        adapter._boiling_point_candidates[cache_key] = None
        return None
    if adapter._property_resolver is None:
        from .resolver import PropertyResolver

        adapter._property_resolver = PropertyResolver()
    props = (
        dict(adapter.component)
        if isinstance(adapter.component, Mapping)
        else dict(vars(adapter.component))
    )
    try:
        resolved = adapter._property_resolver.resolve_boiling_point(
            identifiers[0],
            props,
            allow_online=False,
            allow_estimation=allow_estimation,
        )
    except (ArithmeticError, LookupError, TypeError, ValueError):
        resolved = None
    if (
        not isinstance(resolved, PropertyResolutionResult)
        or _positive_resolution_value(resolved) is None
        or _optional_quality(resolved.quality) is None
        or (
            not allow_estimation
            and (
                _optional_quality(resolved.quality)
                < PSAT_MINIMUM_BOILING_POINT_QUALITY
                or _resolution_is_estimated(resolved)
            )
        )
    ):
        resolved = None
    adapter._boiling_point_candidates[cache_key] = resolved
    return resolved


def _smiles_candidate(
    adapter: PsatCanonicalizationAdapter,
) -> Optional[PropertyResolutionResult]:
    for field_name in ("smiles", "SMILES"):
        value = adapter.component_value(field_name)
        if value:
            return adapter._candidate_from_field(field_name, value)
    identifiers = _component_identifier_candidates(adapter.component)
    if not identifiers:
        return None
    if adapter._property_resolver is None:
        from .resolver import PropertyResolver

        adapter._property_resolver = PropertyResolver()
    props = (
        dict(adapter.component)
        if isinstance(adapter.component, Mapping)
        else dict(vars(adapter.component))
    )
    try:
        result = adapter._property_resolver._resolve_smiles_result(
            identifiers[0],
            props,
            allow_online=False,
        )
    except (ArithmeticError, LookupError, TypeError, ValueError):
        return None
    return (
        result
        if isinstance(result, PropertyResolutionResult) and result.value
        else None
    )


def _finite_resolution_value(
    candidate: PropertyResolutionResult,
) -> Optional[float]:
    try:
        value = float(candidate.value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _resolution_is_estimated(candidate: PropertyResolutionResult) -> bool:
    source = str(candidate.source or "").strip().lower()
    method = str(candidate.method or "").strip().lower()
    return (
        source in {"calculated", "estimated"}
        or method in {"lee_kesler", "estimated", "calculated"}
        or "estimate" in method
    )


def _ambrose_walton_reduced_terms(
    reduced_temperature: float,
) -> tuple[tuple[float, float], ...]:
    if (
        math.isfinite(reduced_temperature)
        and 1.0 < reduced_temperature <= 1.0 + 1.0e-12
    ):
        reduced_temperature = 1.0
    if not math.isfinite(reduced_temperature) or not 0.0 < reduced_temperature <= 1.0:
        raise PsatAdapterError(
            "Ambrose-Walton requires 0 < reduced temperature <= 1"
        )
    tau = 1.0 - reduced_temperature
    coefficient_sets = (
        (-5.97616, 1.29874, -0.60394, -1.06841),
        (-5.03365, 1.11505, -5.41217, -7.46628),
        (-0.64771, 2.41539, -4.26979, 3.25259),
    )
    powers = (1.0, 1.5, 2.5, 5.0)
    terms = []
    for coefficients in coefficient_sets:
        numerator = sum(
            coefficient * tau**power
            for coefficient, power in zip(coefficients, powers)
        )
        derivative_tau = sum(
            coefficient * power * tau ** (power - 1.0)
            for coefficient, power in zip(coefficients, powers)
        )
        value = numerator / reduced_temperature
        derivative_reduced_temperature = (
            -derivative_tau * reduced_temperature - numerator
        ) / reduced_temperature**2
        terms.append((value, derivative_reduced_temperature))
    return tuple(terms)


def _ambrose_walton_ln_pressure(
    T: float,
    Tc: float,
    Pc_bar: float,
    omega: float,
) -> float:
    terms = _ambrose_walton_reduced_terms(T / Tc)
    return (
        math.log(Pc_bar)
        + terms[0][0]
        + omega * terms[1][0]
        + omega**2 * terms[2][0]
    )


def _acentric_factor_from_definition(
    ln_pressure_bar: float,
    Pc_bar: float,
) -> float:
    omega = -(ln_pressure_bar - math.log(Pc_bar)) / math.log(10.0) - 1.0
    if not math.isfinite(omega) or not -0.5 <= omega <= 2.0:
        raise PsatCanonicalizationError(
            "Acentric factor inferred from hard Psat at Tr=0.7 is outside -0.5 to 2"
        )
    return omega


def _ambrose_walton_omega_from_pressure(
    T: float,
    ln_pressure_bar: float,
    Tc: float,
    Pc_bar: float,
    *,
    preferred_omega: Optional[float] = None,
) -> float:
    terms = _ambrose_walton_reduced_terms(T / Tc)
    f0, f1, f2 = (item[0] for item in terms)
    target = ln_pressure_bar - math.log(Pc_bar)
    if abs(f2) <= 1.0e-14:
        if abs(f1) <= 1.0e-14:
            raise PsatCanonicalizationError(
                "Ambrose-Walton cannot infer omega at the hard endpoint"
            )
        roots = ((target - f0) / f1,)
    else:
        discriminant = f1**2 - 4.0 * f2 * (f0 - target)
        if discriminant < 0.0:
            raise PsatCanonicalizationError(
                "Ambrose-Walton omega has no real root at the hard endpoint"
            )
        root = math.sqrt(discriminant)
        roots = (
            (-f1 + root) / (2.0 * f2),
            (-f1 - root) / (2.0 * f2),
        )
    plausible = [value for value in roots if -0.5 <= value <= 2.0]
    if not plausible:
        raise PsatCanonicalizationError(
            "Ambrose-Walton omega inferred at the hard endpoint is outside -0.5 to 2"
        )
    if preferred_omega is not None and math.isfinite(preferred_omega):
        return min(plausible, key=lambda value: abs(value - preferred_omega))
    return min(plausible, key=abs)


def _ambrose_walton_dln_pressure_dT(
    T: float,
    Tc: float,
    omega: float,
) -> float:
    terms = _ambrose_walton_reduced_terms(T / Tc)
    return (
        terms[0][1]
        + omega * terms[1][1]
        + omega**2 * terms[2][1]
    ) / Tc


def _ambrose_walton_omega_sensitivity(
    T: float,
    Tc: float,
    omega: float,
) -> float:
    terms = _ambrose_walton_reduced_terms(T / Tc)
    return terms[1][0] + 2.0 * omega * terms[2][0]


def _c1_anchored_aw_upper_functions(
    *,
    left_temperature: float,
    left_ln_pressure: float,
    left_slope: float,
    Tc: float,
    Pc_bar: float,
    omega: float,
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    span = Tc - left_temperature
    if span <= 1.0e-8 * max(Tc, 1.0):
        raise PsatCanonicalizationError(
            "Anchored Ambrose-Walton requires a nonzero interval below Tc"
        )
    baseline_at_left = _ambrose_walton_ln_pressure(
        left_temperature,
        Tc,
        Pc_bar,
        omega,
    )
    baseline_slope_at_left = _ambrose_walton_dln_pressure_dT(
        left_temperature,
        Tc,
        omega,
    )
    value_delta = left_ln_pressure - baseline_at_left
    slope_delta = left_slope - baseline_slope_at_left
    correction_coefficient = span * slope_delta + 2.0 * value_delta

    def ln_pressure(T: float) -> float:
        coordinate = (T - left_temperature) / span
        correction = (1.0 - coordinate) ** 2 * (
            value_delta + correction_coefficient * coordinate
        )
        return (
            _ambrose_walton_ln_pressure(
                T,
                Tc,
                Pc_bar,
                omega,
            )
            + correction
        )

    def derivative(T: float) -> float:
        coordinate = (T - left_temperature) / span
        correction_derivative = (
            -2.0
            * (1.0 - coordinate)
            * (value_delta + correction_coefficient * coordinate)
            + (1.0 - coordinate) ** 2 * correction_coefficient
        ) / span
        return (
            _ambrose_walton_dln_pressure_dT(T, Tc, omega)
            + correction_derivative
        )

    return ln_pressure, derivative


def _peng_robinson_delta_z_or_ideal(
    T: float,
    pressure_bar: float,
    Tc: float,
    Pc_bar: float,
    omega: float,
) -> float:
    try:
        reduced_temperature = T / Tc
        reduced_pressure = pressure_bar / Pc_bar
        if not (
            0.0 < reduced_temperature < 1.0
            and reduced_pressure > 0.0
        ):
            return 1.0
        kappa = (
            0.37464
            + 1.54226 * omega
            - 0.26992 * omega**2
        )
        alpha = (
            1.0
            + kappa
            * (1.0 - math.sqrt(reduced_temperature))
        ) ** 2
        A = (
            0.45724
            * alpha
            * reduced_pressure
            / reduced_temperature**2
        )
        B = (
            0.07780
            * reduced_pressure
            / reduced_temperature
        )
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..cubic_eos import CubicEOS
        else:
            from cubic_eos import CubicEOS

        roots = sorted(
            root
            for root in CubicEOS._solve_monic_cubic(
                -(1.0 - B),
                A - 3.0 * B**2 - 2.0 * B,
                -(A * B - B**2 - B**3),
            )
            if root > B + 1.0e-12
        )
        if len(roots) < 2:
            return 1.0
        delta_z = roots[-1] - roots[0]
        return (
            delta_z
            if math.isfinite(delta_z) and delta_z > 0.0
            else 1.0
        )
    except (ArithmeticError, TypeError, ValueError):
        return 1.0


def _dimer_extent(
    T: float,
    ln_pressure: float,
    entropy_J_mol_K: float,
) -> float:
    exponent = (
        entropy_J_mol_K / R
        - DIMER_ENTHALPY_J_MOL / (R * T)
        + ln_pressure
    )
    pressure_product = math.exp(
        min(700.0, max(-700.0, exponent))
    )
    return 0.5 * (
        1.0
        - 1.0 / math.sqrt(1.0 + 4.0 * pressure_product)
    )


def _monocarboxylic_acid_identifier(
    component: Any,
) -> Optional[str]:
    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..vapor_dimerization import is_monocarboxylic_acid
        else:
            from vapor_dimerization import is_monocarboxylic_acid
    except ImportError:
        return None
    smiles = (
        _component_value(component, "smiles")
        or _component_value(component, "SMILES")
    )
    for identifier in _component_identifier_candidates(component):
        try:
            if is_monocarboxylic_acid(
                identifier,
                smiles=(str(smiles) if smiles else None),
            ):
                return identifier
        except (ArithmeticError, LookupError, TypeError, ValueError):
            continue
    return None


def _candidate_from_psat(raw_value: Any) -> PropertyResolutionResult:
    if isinstance(raw_value, PropertyResolutionResult):
        return raw_value
    if isinstance(raw_value, Mapping) and "value" in raw_value:
        return PropertyResolutionResult(
            value=raw_value.get("value"),
            source=str(raw_value.get("source") or "calculated"),
            method=str(raw_value.get("method") or "psat_at_temperature"),
            quality=raw_value.get("quality"),
            notes=str(raw_value.get("notes") or ""),
        )
    return PropertyResolutionResult(
        value=raw_value,
        source="calculated",
        method="psat_at_temperature",
        quality=None,
        notes="",
    )


def _candidate_from_tsat(raw_value: Any) -> PropertyResolutionResult:
    if isinstance(raw_value, PropertyResolutionResult):
        return raw_value
    if isinstance(raw_value, Mapping) and "value" in raw_value:
        return PropertyResolutionResult(
            value=raw_value.get("value"),
            source=str(raw_value.get("source") or "calculated"),
            method=str(raw_value.get("method") or "tsat_at_pressure"),
            quality=raw_value.get("quality"),
            notes=str(raw_value.get("notes") or ""),
        )
    return PropertyResolutionResult(
        value=raw_value,
        source="calculated",
        method="tsat_at_pressure",
        quality=None,
        notes="",
    )


def _positive_resolution_value(
    candidate: Optional[PropertyResolutionResult],
) -> Optional[float]:
    if candidate is None:
        return None
    try:
        value = float(candidate.value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _positive_finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise PsatAdapterError(
            f"{name} must be a positive finite value"
        ) from error
    if not math.isfinite(number) or number <= 0.0:
        raise PsatAdapterError(f"{name} must be a positive finite value")
    return number


def _valid_subcritical_temperature(value: Any, T_critical: float) -> Optional[float]:
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(temperature)
        or temperature <= 0.0
        or temperature >= T_critical
    ):
        return None
    return temperature


def _optional_quality(value: Any) -> Optional[float]:
    try:
        quality = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
        return None
    return quality


_CANONICAL_PSAT_EQUATIONS = {
    "canonical_psat": CanonicalPsatForm.AF,
    "canonical_psat_af": CanonicalPsatForm.AF,
    "canonical_psat_ag": CanonicalPsatForm.AG,
    "canonical_psat_ah": CanonicalPsatForm.AH,
}
_SUPPORTED_PINNED_PSAT_EQUATIONS = frozenset({
    "poly_x",
    "exp_poly_x",
    "reduced_vapor_pressure",
    "psat_mercury",
        *_CANONICAL_PSAT_EQUATIONS,
})


def _psat_correlation_property_dependencies(
    equation: str,
    correlation: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    """Declare selected-property inputs consumed by a direct Psat evaluator."""
    declared = (
        correlation.get('property_dependencies')
        if isinstance(correlation, Mapping)
        else None
    )
    if declared is not None:
        if isinstance(declared, str):
            declared = (declared,)
        return tuple(sorted({str(item).strip() for item in declared if str(item).strip()}))
    normalized = str(equation or "").strip().lower()
    if normalized in {"reduced_vapor_pressure", "psat_mercury"}:
        return ("Tc", "Pc")
    if normalized == "canonical_psat_ah":
        return ("Tc",)
    return ()


def _required_coefficients(
    correlation: Mapping[str, Any],
    equation: str,
) -> dict[str, float]:
    if equation not in _SUPPORTED_PINNED_PSAT_EQUATIONS:
        raise PsatAdapterError(f"Unsupported PFD Psat equation {equation!r}")
    raw_coefficients = correlation.get("coefficients")
    coefficients = raw_coefficients if isinstance(raw_coefficients, Mapping) else {}
    form = _CANONICAL_PSAT_EQUATIONS.get(equation)
    if form:
        required = ["A", "B", "C", "D", "E", "F"]
    elif equation == "psat_mercury":
        required = ["A", "B", "C", "D", "E", "F"]
    elif equation == "reduced_vapor_pressure":
        required = ["A", "B", "C", "D"]
    else:
        required = ["A"]
    if form in {CanonicalPsatForm.AG, CanonicalPsatForm.AH}:
        required.append("G")
    if form is CanonicalPsatForm.AH:
        required.append("H")
    result = {}
    for name in ("A", "B", "C", "D", "E", "F", "G", "H"):
        raw_value = coefficients.get(name)
        if raw_value is None:
            if name in required:
                label = "PFD canonical Psat" if form else f"Psat {equation}"
                raise PsatAdapterError(f"{label} requires coefficient {name}")
            raw_value = 0.0
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise PsatAdapterError(
                f"PFD Psat coefficient {name} must be finite"
            ) from error
        if not math.isfinite(value):
            raise PsatAdapterError(f"PFD Psat coefficient {name} must be finite")
        result[name] = value
    if form is CanonicalPsatForm.AF and (result["G"] != 0.0 or result["H"] != 0.0):
        raise PsatAdapterError("PFD canonical A-F Psat cannot contain nonzero G or H")
    if form is CanonicalPsatForm.AG and result["H"] != 0.0:
        raise PsatAdapterError("PFD canonical A-G Psat cannot contain nonzero H")
    return result


def _psat_correlation_functions(
    equation: str,
    coefficients: Mapping[str, float],
    correlation: Mapping[str, Any],
    component: Any,
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    if equation in _CANONICAL_PSAT_EQUATIONS:
        form = _CANONICAL_PSAT_EQUATIONS[equation]
        inverse_power = _canonical_inverse_power(
            correlation,
            coefficients,
            form,
        )
        T_critical = _correlation_float(
            correlation,
            "Tc_K",
            default=_component_numeric_value(component, "Tc"),
        )

        def canonical_ln_pressure(T: float) -> float:
            value = (
                coefficients["A"]
                + coefficients["B"] / T
                + coefficients["C"] * math.log(T)
                + coefficients["D"] * T
                + coefficients["E"] * T**2
                + coefficients["F"] * T**5
            )
            if form in {CanonicalPsatForm.AG, CanonicalPsatForm.AH}:
                value += coefficients["G"] * T**3
            if form is CanonicalPsatForm.AH:
                value += coefficients["H"] * (
                    (T / T_critical) ** inverse_power - 1.0
                )
            return value

        def canonical_derivative(T: float) -> float:
            value = (
                -coefficients["B"] / T**2
                + coefficients["C"] / T
                + coefficients["D"]
                + 2.0 * coefficients["E"] * T
                + 5.0 * coefficients["F"] * T**4
            )
            if form in {CanonicalPsatForm.AG, CanonicalPsatForm.AH}:
                value += 3.0 * coefficients["G"] * T**2
            if form is CanonicalPsatForm.AH:
                value += (
                    coefficients["H"]
                    * inverse_power
                    / T_critical
                    * (T / T_critical) ** (inverse_power - 1)
                )
            return value

        return canonical_ln_pressure, canonical_derivative

    if equation in {"poly_x", "exp_poly_x"}:
        def polynomial(T: float) -> float:
            x = (T - 298.15) / 100.0
            return sum(
                coefficients[name] * x**power
                for power, name in enumerate(("A", "B", "C", "D", "E", "F"))
            )

        def polynomial_derivative(T: float) -> float:
            x = (T - 298.15) / 100.0
            return sum(
                power * coefficients[name] * x ** (power - 1)
                for power, name in enumerate(("A", "B", "C", "D", "E", "F"))
                if power
            ) / 100.0

        if equation == "exp_poly_x":
            return polynomial, polynomial_derivative

        def poly_ln_pressure(T: float) -> float:
            pressure = polynomial(T)
            if not math.isfinite(pressure) or pressure <= 0.0:
                raise PsatAdapterError(
                    f"PFD poly_x Psat returned invalid pressure at {T:g} K"
                )
            return math.log(pressure)

        def poly_ln_derivative(T: float) -> float:
            pressure = polynomial(T)
            if not math.isfinite(pressure) or pressure <= 0.0:
                raise PsatAdapterError(
                    f"PFD poly_x Psat returned invalid pressure at {T:g} K"
                )
            return polynomial_derivative(T) / pressure

        return poly_ln_pressure, poly_ln_derivative

    if equation == "psat_mercury":
        Tc_hg = _correlation_float(
            correlation,
            "Tc_K",
            default=_component_numeric_value(component, "Tc"),
        )
        Pc_bar_hg = _critical_pressure_bar(correlation, component)

        def _mercury_series(tau: float) -> float:
            return (
                coefficients["A"] * tau
                + coefficients["B"] * tau**1.89
                + coefficients["C"] * tau**2.0
                + coefficients["D"] * tau**8.0
                + coefficients["E"] * tau**8.5
                + coefficients["F"] * tau**9.0
            )

        def _mercury_series_derivative(tau: float) -> float:
            # dS/dtau of the term series above.
            return (
                coefficients["A"]
                + 1.89 * coefficients["B"] * tau**0.89
                + 2.0 * coefficients["C"] * tau
                + 8.0 * coefficients["D"] * tau**7.0
                + 8.5 * coefficients["E"] * tau**7.5
                + 9.0 * coefficients["F"] * tau**8.0
            )

        def mercury_ln_pressure(T: float) -> float:
            Tr = T / Tc_hg
            tau = 1.0 - Tr
            if Tr <= 0.0 or tau < 0.0:
                raise PsatAdapterError(
                    f"PFD mercury vapor pressure is invalid at {T:g} K"
                )
            return math.log(Pc_bar_hg) + _mercury_series(tau) / Tr

        def mercury_derivative(T: float) -> float:
            Tr = T / Tc_hg
            tau = 1.0 - Tr
            if Tr <= 0.0 or tau < 0.0:
                raise PsatAdapterError(
                    f"PFD mercury vapor pressure is invalid at {T:g} K"
                )
            series = _mercury_series(tau)
            series_derivative = _mercury_series_derivative(tau)
            # ln(P) = ln(Pc) + S(tau)/Tr; d(ln P)/dT with dTr/dT = 1/Tc.
            return (-series_derivative * Tr - series) / (Tr**2 * Tc_hg)

        return mercury_ln_pressure, mercury_derivative

    Tc = _correlation_float(
        correlation,
        "Tc_K",
        default=_component_numeric_value(component, "Tc"),
    )
    Pc_bar = _critical_pressure_bar(correlation, component)

    def reduced_ln_pressure(T: float) -> float:
        Tr = T / Tc
        tau = 1.0 - Tr
        if Tr <= 0.0 or tau < 0.0:
            raise PsatAdapterError(
                f"PFD reduced vapor pressure is invalid at {T:g} K"
            )
        numerator = (
            coefficients["A"] * tau
            + coefficients["B"] * tau**1.5
            + coefficients["C"] * tau**3
            + coefficients["D"] * tau**6
        )
        return math.log(Pc_bar) + numerator / Tr

    def reduced_derivative(T: float) -> float:
        Tr = T / Tc
        tau = 1.0 - Tr
        if Tr <= 0.0 or tau < 0.0:
            raise PsatAdapterError(
                f"PFD reduced vapor pressure is invalid at {T:g} K"
            )
        numerator = (
            coefficients["A"] * tau
            + coefficients["B"] * tau**1.5
            + coefficients["C"] * tau**3
            + coefficients["D"] * tau**6
        )
        derivative_Tr = -(
            coefficients["A"]
            + 1.5 * coefficients["B"] * math.sqrt(tau)
            + 3.0 * coefficients["C"] * tau**2
            + 6.0 * coefficients["D"] * tau**5
        )
        return (derivative_Tr * Tr - numerator) / (Tr**2 * Tc)

    return reduced_ln_pressure, reduced_derivative


def _correlation_quality(
    correlation: Mapping[str, Any],
    *,
    default: float,
) -> float:
    raw_quality = correlation.get("quality", default)
    try:
        quality = float(raw_quality)
    except (TypeError, ValueError) as error:
        raise PsatAdapterError("Psat correlation quality must be finite") from error
    if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
        raise PsatAdapterError("Psat correlation quality must be between 0 and 1")
    return quality


def _property_source_quality(
    source_metadata: Mapping[str, Any],
    *,
    default: float,
) -> float:
    raw_quality = source_metadata.get("quality", default)
    try:
        quality = float(raw_quality)
    except (TypeError, ValueError) as error:
        raise PsatAdapterError("Property source quality must be finite") from error
    if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
        raise PsatAdapterError("Property source quality must be between 0 and 1")
    return quality


def _canonical_inverse_power(
    correlation: Mapping[str, Any],
    coefficients: Mapping[str, float],
    form: CanonicalPsatForm,
) -> Optional[int]:
    raw_power = correlation.get("inverse_power")
    if form is not CanonicalPsatForm.AH:
        if raw_power not in (None, ""):
            raise PsatAdapterError(
                f"PFD canonical {form.value} Psat cannot declare inverse_power"
            )
        return None
    if raw_power in (None, ""):
        raise PsatAdapterError("PFD canonical A-H Psat requires inverse_power")
    try:
        power = int(raw_power)
    except (TypeError, ValueError) as error:
        raise PsatAdapterError(
            "PFD canonical Psat inverse_power must be -3, -5, or -7"
        ) from error
    if float(raw_power) != power or power not in {-3, -5, -7}:
        raise PsatAdapterError(
            "PFD canonical Psat inverse_power must be -3, -5, or -7"
        )
    return power


def _component_context(component: Any) -> dict[str, Any]:
    def value(name: str) -> Any:
        if isinstance(component, Mapping):
            return component.get(name)
        return getattr(component, name, None)

    return {
        key: item
        for key in ("symbol", "name", "CAS")
        if (item := value(key)) not in (None, "")
    }


def _component_identifier_candidates(component: Any) -> tuple[str, ...]:
    for field_name in ("CAS", "cas"):
        value = _component_value(component, field_name)
        if value not in (None, ""):
            text = str(value).strip()
            return (text,) if text else ()
    name = _component_value(component, "name")
    if name not in (None, ""):
        text = str(name).strip()
        return (text,) if text else ()
    return ()


def _component_numeric_value(component: Any, field_name: str) -> Optional[float]:
    raw_value = (
        component.get(field_name)
        if isinstance(component, Mapping)
        else getattr(component, field_name, None)
    )
    if isinstance(raw_value, PropertyResolutionResult):
        raw_value = raw_value.value
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _required_pfd_component_number(
    adapter: PsatCanonicalizationAdapter,
    field_name: str,
) -> float:
    value = _component_numeric_value(adapter.component, field_name)
    if value is None:
        raise PsatAdapterError(
            f"Parsed PFD Antoine invariant is missing finite {field_name}; "
            "PFDParser should have rejected this override"
        )
    return value


def _correlation_float(
    correlation: Mapping[str, Any],
    *names: str,
    default: Optional[float] = None,
) -> float:
    value = _optional_correlation_float(correlation, *names)
    if value is None:
        value = default
    if value is None or not math.isfinite(float(value)):
        joined = " or ".join(names)
        raise PsatAdapterError(f"PFD Psat correlation requires finite {joined}")
    return float(value)


def _optional_correlation_float(
    correlation: Mapping[str, Any],
    *names: str,
) -> Optional[float]:
    for name in names:
        raw_value = correlation.get(name)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise PsatAdapterError(f"PFD Psat {name} must be finite") from error
        if not math.isfinite(value):
            raise PsatAdapterError(f"PFD Psat {name} must be finite")
        return value
    return None


def _optional_positive_correlation_float(
    correlation: Mapping[str, Any],
    name: str,
) -> Optional[float]:
    value = _optional_correlation_float(correlation, name)
    if value is not None and value <= 0.0:
        raise PsatAdapterError(f"PFD Psat {name} must be positive")
    return value


def _critical_pressure_bar(
    correlation: Mapping[str, Any],
    component: Any,
    *,
    require_explicit: bool = False,
) -> float:
    pressure_bar = _optional_correlation_float(correlation, "Pc_bar")
    if pressure_bar is not None:
        return _positive_finite(pressure_bar, "PFD Psat Pc_bar")
    pressure_pa = _optional_correlation_float(correlation, "Pc_Pa")
    if pressure_pa is not None:
        return _positive_finite(pressure_pa / 100000.0, "PFD Psat Pc_Pa")
    if require_explicit:
        raise PsatAdapterError(
            "Local reduced Psat correlation requires explicit Pc_bar or Pc_Pa"
        )
    return _positive_finite(
        _component_numeric_value(component, "Pc"),
        "PFD Psat component Pc",
    )


__all__ = [
    "DEFAULT_PSAT_MINIMUM_PRESSURE_BAR",
    "DEFAULT_PSAT_INPUT_METHODS",
    "PsatAtTemperature",
    "PsatAdapterError",
    "PsatCanonicalizationAdapter",
    "PsatCanonicalizationInputs",
    "PsatDomainSelection",
    "PsatInputMethod",
    "TsatAtPressure",
]
