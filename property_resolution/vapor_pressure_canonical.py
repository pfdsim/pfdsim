"""Provider-agnostic architecture for canonical vapor-pressure curves."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import brentq


LnPressureFunction = Callable[[float], float]
LnPressureDerivative = Callable[[float], float]


class PsatCanonicalizationError(ValueError):
    """Raised when a segment or canonical curve violates its contract."""


class PsatSegmentType(str, Enum):
    """Role a segment plays while a complete saturation curve is assembled."""

    CANONICAL_OVERRIDE = "canonical_override"
    PINNED = "pinned"
    COMPLETION = "completion"
    FALLBACK = "fallback"

    @property
    def is_hard_pinned(self) -> bool:
        return self in {self.CANONICAL_OVERRIDE, self.PINNED}


class PsatDerivativeBasis(str, Enum):
    """How a segment's local ``dln(P)/dT`` relation was obtained."""

    ANALYTIC = "analytic"
    INTERPOLATED = "interpolated"
    NUMERICAL = "numerical"


class PsatHandoffRequirement(str, Enum):
    """Additional source validation required before a segment may be selected."""

    NONE = "none"
    HIGHER_PREFERENCE_OVERLAP = "higher_preference_overlap"
    HIGHER_PREFERENCE_OVERLAP_IF_AVAILABLE = (
        "higher_preference_overlap_if_available"
    )


class CanonicalPsatForm(str, Enum):
    """Supported compact runtime representations, ordered by complexity."""

    AF = "A-F"
    AG = "A-G"
    AH = "A-H"


class PsatPriority(IntEnum):
    """Default priorities; providers may use any integer priority."""

    PFD_OVERRIDE = 1000
    COOLPROP_HEOS = 900
    LOCAL_CORRELATION = 800
    LOCAL_ANTOINE = 700
    PERRY_2_8 = 600
    HIGH_QUALITY_ANTOINE = 550
    PERRY_2_10 = 500
    OTHER_ANTOINE = 400
    MIDDLE_COMPLETION = 320
    PRIMARY_COMPLETION = 300
    AMBROSE_WALTON = 300
    CLAPEYRON = 300
    NO_HARD_FALLBACK = 200
    LAST_FALLBACK = 100
    NANNOOLAL = 100


@dataclass(frozen=True)
class PsatSegment:
    """A bounded source, completion, or fallback relation for ``ln(P/bar)``."""

    source: str
    method: str
    segment_type: PsatSegmentType
    priority: int
    T_min: float
    T_max: float
    ln_pressure_function: LnPressureFunction = field(repr=False, compare=False)
    derivative_function: Optional[LnPressureDerivative] = field(
        default=None,
        repr=False,
        compare=False,
    )
    quality: float = 1.0
    P_min_bar: Optional[float] = None
    P_max_bar: Optional[float] = None
    context: Mapping[str, Any] = field(default_factory=dict, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    handoff_requirement: PsatHandoffRequirement = PsatHandoffRequirement.NONE
    derivative_basis: Optional[PsatDerivativeBasis] = None
    allow_junction_slope_mismatch: bool = False

    def __post_init__(self) -> None:
        if not str(self.source).strip():
            raise PsatCanonicalizationError("A Psat segment source is required")
        if not str(self.method).strip():
            raise PsatCanonicalizationError("A Psat segment method is required")
        if not isinstance(self.segment_type, PsatSegmentType):
            try:
                object.__setattr__(self, "segment_type", PsatSegmentType(self.segment_type))
            except (TypeError, ValueError) as error:
                raise PsatCanonicalizationError("Invalid Psat segment type") from error
        if not isinstance(self.handoff_requirement, PsatHandoffRequirement):
            try:
                object.__setattr__(
                    self,
                    "handoff_requirement",
                    PsatHandoffRequirement(self.handoff_requirement),
                )
            except (TypeError, ValueError) as error:
                raise PsatCanonicalizationError(
                    "Invalid Psat segment handoff requirement"
                ) from error
        _validate_temperature_range(self.T_min, self.T_max)
        _validate_pressure_range(self.P_min_bar, self.P_max_bar)
        if not isinstance(self.priority, int):
            raise PsatCanonicalizationError("Psat segment priority must be an integer")
        if not isinstance(self.allow_junction_slope_mismatch, bool):
            raise PsatCanonicalizationError(
                "Psat junction slope mismatch allowance must be boolean"
            )
        if not math.isfinite(float(self.quality)) or not 0.0 <= self.quality <= 1.0:
            raise PsatCanonicalizationError("Psat segment quality must be between 0 and 1")
        if not callable(self.ln_pressure_function):
            raise PsatCanonicalizationError("A Psat segment evaluator is required")
        if self.derivative_function is not None and not callable(self.derivative_function):
            raise PsatCanonicalizationError("Psat segment derivative must be callable")
        derivative_basis = self.derivative_basis
        if derivative_basis is None:
            derivative_basis = (
                PsatDerivativeBasis.ANALYTIC
                if self.derivative_function is not None
                else PsatDerivativeBasis.NUMERICAL
            )
            object.__setattr__(self, "derivative_basis", derivative_basis)
        elif not isinstance(derivative_basis, PsatDerivativeBasis):
            try:
                object.__setattr__(
                    self,
                    "derivative_basis",
                    PsatDerivativeBasis(derivative_basis),
                )
            except (TypeError, ValueError) as error:
                raise PsatCanonicalizationError(
                    "Invalid Psat derivative basis"
                ) from error

    @property
    def is_hard_pinned(self) -> bool:
        return self.segment_type.is_hard_pinned

    def covers_temperature(self, T: float, tolerance: float = 1.0e-9) -> bool:
        return self.T_min - tolerance <= T <= self.T_max + tolerance

    def covers_pressure(self, pressure_bar: float, tolerance: float = 1.0e-9) -> bool:
        if not math.isfinite(pressure_bar) or pressure_bar <= 0.0:
            return False
        if self.P_min_bar is not None and pressure_bar < self.P_min_bar * (1.0 - tolerance):
            return False
        if self.P_max_bar is not None and pressure_bar > self.P_max_bar * (1.0 + tolerance):
            return False
        return True

    def raw_ln_pressure(self, T: float) -> float:
        self._require_temperature(T)
        value = float(self.ln_pressure_function(float(T)))
        if not math.isfinite(value):
            raise PsatCanonicalizationError(
                f"{self.method} returned a non-finite ln(P) at {T:g} K"
            )
        return value

    def ln_pressure(self, T: float) -> float:
        value = self.raw_ln_pressure(T)
        self._require_pressure(value, T)
        return value

    def pressure_bar(self, T: float) -> float:
        try:
            value = math.exp(self.ln_pressure(T))
        except OverflowError as error:
            raise PsatCanonicalizationError(
                f"{self.method} overflowed while evaluating pressure at {T:g} K"
            ) from error
        if not math.isfinite(value) or value <= 0.0:
            raise PsatCanonicalizationError(
                f"{self.method} returned an invalid pressure at {T:g} K"
            )
        return value

    def dln_pressure_dT(self, T: float) -> float:
        self.ln_pressure(T)
        return self.raw_dln_pressure_dT(T)

    def raw_dln_pressure_dT(self, T: float) -> float:
        self._require_temperature(T)
        if self.derivative_function is not None:
            value = float(self.derivative_function(float(T)))
        else:
            value = self._numerical_derivative(float(T))
        if not math.isfinite(value):
            raise PsatCanonicalizationError(
                f"{self.method} returned a non-finite derivative at {T:g} K"
            )
        return value

    def clipped(self, T_min: float, T_max: float) -> "PsatSegment":
        _validate_temperature_range(T_min, T_max)
        if T_min < self.T_min or T_max > self.T_max:
            raise PsatCanonicalizationError("A clipped segment exceeds its source range")
        return replace(self, T_min=float(T_min), T_max=float(T_max))

    def _require_temperature(self, T: float) -> None:
        if not math.isfinite(float(T)) or not self.covers_temperature(float(T)):
            raise PsatCanonicalizationError(
                f"Temperature {T!r} K is outside {self.method} range "
                f"{self.T_min:g}-{self.T_max:g} K"
            )

    def _require_pressure(self, ln_pressure: float, T: float) -> None:
        ln_min = math.log(self.P_min_bar) if self.P_min_bar is not None else -math.inf
        ln_max = math.log(self.P_max_bar) if self.P_max_bar is not None else math.inf
        tolerance = 1.0e-9
        if ln_pressure < ln_min - tolerance or ln_pressure > ln_max + tolerance:
            raise PsatCanonicalizationError(
                f"{self.method} pressure at {T:g} K is outside declared range "
                f"{self.P_min_bar or 0:g}-{self.P_max_bar or math.inf:g} bar"
            )

    def _numerical_derivative(self, T: float) -> float:
        width = self.T_max - self.T_min
        step = min(max(1.0e-5 * max(abs(T), 1.0), 1.0e-6), width / 4.0)
        if T - step >= self.T_min and T + step <= self.T_max:
            return (
                self.ln_pressure_function(T + step)
                - self.ln_pressure_function(T - step)
            ) / (2.0 * step)
        if T + 2.0 * step <= self.T_max:
            return (
                -3.0 * self.ln_pressure_function(T)
                + 4.0 * self.ln_pressure_function(T + step)
                - self.ln_pressure_function(T + 2.0 * step)
            ) / (2.0 * step)
        if T - 2.0 * step >= self.T_min:
            return (
                3.0 * self.ln_pressure_function(T)
                - 4.0 * self.ln_pressure_function(T - step)
                + self.ln_pressure_function(T - 2.0 * step)
            ) / (2.0 * step)
        raise PsatCanonicalizationError(
            f"Cannot estimate a derivative inside {self.method} range"
        )


@dataclass(frozen=True)
class PsatSegmentSlice:
    """The interval of a segment selected by priority arbitration."""

    segment: PsatSegment
    T_min: float
    T_max: float

    def __post_init__(self) -> None:
        _validate_temperature_range(self.T_min, self.T_max)
        if self.T_min < self.segment.T_min or self.T_max > self.segment.T_max:
            raise PsatCanonicalizationError("A segment slice exceeds its source range")

    @property
    def priority(self) -> int:
        return self.segment.priority

    @property
    def quality(self) -> float:
        return self.segment.quality

    def covers_temperature(self, T: float, tolerance: float = 1.0e-9) -> bool:
        return self.T_min - tolerance <= T <= self.T_max + tolerance

    def ln_pressure(self, T: float) -> float:
        if not self.covers_temperature(T):
            raise PsatCanonicalizationError("Temperature is outside the selected slice")
        return self.segment.ln_pressure(T)

    def dln_pressure_dT(self, T: float) -> float:
        if not self.covers_temperature(T):
            raise PsatCanonicalizationError("Temperature is outside the selected slice")
        return self.segment.dln_pressure_dT(T)


@dataclass(frozen=True)
class PsatGap:
    """An uncovered temperature interval requiring completion."""

    T_min: float
    T_max: float

    def __post_init__(self) -> None:
        _validate_temperature_range(self.T_min, self.T_max)


@dataclass(frozen=True)
class PsatEndpoint:
    """Value, slope, quality, and context at one segment boundary."""

    temperature: float
    ln_pressure: float
    dln_pressure_dT: Optional[float]
    source: str
    method: str
    quality: float
    context: Mapping[str, Any] = field(default_factory=dict, compare=False)
    segment_type: Optional[PsatSegmentType] = None
    derivative_basis: Optional[PsatDerivativeBasis] = None

    def __post_init__(self) -> None:
        values = (self.temperature, self.ln_pressure, self.quality)
        if any(not math.isfinite(value) for value in values):
            raise PsatCanonicalizationError("Psat endpoint values must be finite")
        if self.dln_pressure_dT is not None and not math.isfinite(self.dln_pressure_dT):
            raise PsatCanonicalizationError("Psat endpoint derivative must be finite")
        derivative_basis = self.derivative_basis
        if self.dln_pressure_dT is None:
            if derivative_basis is not None:
                raise PsatCanonicalizationError(
                    "A value-only Psat endpoint cannot declare a derivative basis"
                )
        elif derivative_basis is None:
            object.__setattr__(
                self,
                "derivative_basis",
                PsatDerivativeBasis.NUMERICAL,
            )
        elif not isinstance(derivative_basis, PsatDerivativeBasis):
            try:
                object.__setattr__(
                    self,
                    "derivative_basis",
                    PsatDerivativeBasis(derivative_basis),
                )
            except (TypeError, ValueError) as error:
                raise PsatCanonicalizationError(
                    "Invalid Psat endpoint derivative basis"
                ) from error
        if self.temperature <= 0.0:
            raise PsatCanonicalizationError("Psat endpoint temperature must be positive")
        if not 0.0 <= self.quality <= 1.0:
            raise PsatCanonicalizationError("Psat endpoint quality must be between 0 and 1")
        if not str(self.source).strip() or not str(self.method).strip():
            raise PsatCanonicalizationError("Psat endpoint source and method are required")

    @property
    def has_derivative(self) -> bool:
        return self.dln_pressure_dT is not None

    @classmethod
    def from_segment(cls, segment: PsatSegment, temperature: float) -> "PsatEndpoint":
        return cls(
            temperature=float(temperature),
            ln_pressure=segment.ln_pressure(temperature),
            dln_pressure_dT=segment.dln_pressure_dT(temperature),
            source=segment.source,
            method=segment.method,
            quality=segment.quality,
            derivative_basis=segment.derivative_basis,
            context=dict(segment.context),
            segment_type=segment.segment_type,
        )

    @classmethod
    def fixed_pressure_point(
        cls,
        temperature: float,
        pressure_bar: float,
        *,
        source: str,
        method: str,
        quality: float,
        context: Optional[Mapping[str, Any]] = None,
    ) -> "PsatEndpoint":
        if not math.isfinite(pressure_bar) or pressure_bar <= 0.0:
            raise PsatCanonicalizationError("Fixed Psat boundary pressure must be positive")
        return cls(
            temperature=float(temperature),
            ln_pressure=math.log(pressure_bar),
            dln_pressure_dT=None,
            source=source,
            method=method,
            quality=quality,
            context=dict(context or {}),
        )

    @classmethod
    def from_slice(
        cls,
        item: PsatSegmentSlice,
        temperature: float,
    ) -> "PsatEndpoint":
        if not item.covers_temperature(temperature):
            raise PsatCanonicalizationError("Endpoint temperature is outside the slice")
        return cls.from_segment(item.segment, temperature)


class PsatBoundaryRequirement(str, Enum):
    """Accepted boundary shapes for a deferred completion relation."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    EITHER = "either"


class PsatDerivativeRequirement(str, Enum):
    """Derivative data required from a neighboring boundary point."""

    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


@dataclass(frozen=True)
class PsatAnchorRequirement:
    """A named fixed point and the derivative data required with it."""

    name: str
    derivative_requirement: PsatDerivativeRequirement = PsatDerivativeRequirement.NONE

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise PsatCanonicalizationError("A named Psat anchor requirement is required")


@dataclass(frozen=True)
class PsatAssemblyAnchorRequirement:
    """An interior point requested from the currently selected assembly."""

    name: str
    temperature: float
    derivative_requirement: PsatDerivativeRequirement = PsatDerivativeRequirement.NONE
    allowed_segment_types: tuple[PsatSegmentType, ...] = ()
    required: bool = True

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise PsatCanonicalizationError("An assembly anchor name is required")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise PsatCanonicalizationError(
                "An assembly anchor temperature must be positive and finite"
            )
        if any(not isinstance(item, PsatSegmentType) for item in self.allowed_segment_types):
            raise PsatCanonicalizationError(
                "Assembly anchor segment types must be PsatSegmentType values"
            )


@dataclass(frozen=True)
class PsatBoundaryConditions:
    """Known neighboring values supplied when an uncovered gap is completed."""

    T_min: float
    T_max: float
    left: Optional[PsatEndpoint] = None
    right: Optional[PsatEndpoint] = None
    anchors: tuple[PsatEndpoint, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        _validate_temperature_range(self.T_min, self.T_max)
        tolerance = 1.0e-7 * max(self.T_max - self.T_min, 1.0)
        if self.left is not None and abs(self.left.temperature - self.T_min) > tolerance:
            raise PsatCanonicalizationError(
                "The left boundary condition must lie at the gap lower endpoint"
            )
        if self.right is not None and abs(self.right.temperature - self.T_max) > tolerance:
            raise PsatCanonicalizationError(
                "The right boundary condition must lie at the gap upper endpoint"
            )

    def anchor(self, name: str) -> Optional[PsatEndpoint]:
        normalized = str(name).strip().lower()
        for item in self.anchors:
            anchor_name = str(item.context.get("anchor_name") or "").strip().lower()
            if anchor_name == normalized:
                return item
        return None


@dataclass(frozen=True)
class BoundaryConditionedPsatEvaluation:
    """Evaluators plus metadata resolved only after boundaries are known."""

    ln_pressure_function: LnPressureFunction = field(repr=False, compare=False)
    derivative_function: Optional[LnPressureDerivative] = field(
        default=None,
        repr=False,
        compare=False,
    )
    quality: Optional[float] = None
    context: Mapping[str, Any] = field(default_factory=dict, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    allow_junction_slope_mismatch: Optional[bool] = None

    def __post_init__(self) -> None:
        if not callable(self.ln_pressure_function):
            raise PsatCanonicalizationError(
                "A bound Psat evaluation requires a pressure evaluator"
            )
        if self.derivative_function is not None and not callable(
            self.derivative_function
        ):
            raise PsatCanonicalizationError(
                "A bound Psat derivative must be callable"
            )
        if self.quality is not None and (
            not math.isfinite(self.quality) or not 0.0 <= self.quality <= 1.0
        ):
            raise PsatCanonicalizationError(
                "A bound Psat quality must be between 0 and 1"
            )
        if (
            self.allow_junction_slope_mismatch is not None
            and not isinstance(self.allow_junction_slope_mismatch, bool)
        ):
            raise PsatCanonicalizationError(
                "A bound junction slope mismatch allowance must be boolean"
            )


BoundaryEvaluationFactory = Callable[
    [PsatBoundaryConditions],
    tuple[LnPressureFunction, Optional[LnPressureDerivative]]
    | BoundaryConditionedPsatEvaluation,
]


@dataclass(frozen=True)
class BoundaryConditionedPsatSegment:
    """A deferred segment relation that becomes evaluable after gap binding."""

    source: str
    method: str
    segment_type: PsatSegmentType
    priority: int
    quality: float
    boundary_requirement: PsatBoundaryRequirement
    T_min: float
    T_max: float
    evaluation_factory: BoundaryEvaluationFactory = field(repr=False, compare=False)
    P_min_bar: Optional[float] = None
    P_max_bar: Optional[float] = None
    left_derivative_requirement: PsatDerivativeRequirement = PsatDerivativeRequirement.NONE
    right_derivative_requirement: PsatDerivativeRequirement = PsatDerivativeRequirement.NONE
    required_anchor_names: tuple[str, ...] = ()
    anchor_requirements: tuple[PsatAnchorRequirement, ...] = ()
    assembly_anchor_requirements: tuple[PsatAssemblyAnchorRequirement, ...] = ()
    allowed_left_segment_types: tuple[PsatSegmentType, ...] = ()
    allowed_right_segment_types: tuple[PsatSegmentType, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    inherit_boundary_quality: bool = True
    allow_junction_slope_mismatch: bool = False
    requires_no_hard_segments: bool = False

    def __post_init__(self) -> None:
        if not str(self.source).strip() or not str(self.method).strip():
            raise PsatCanonicalizationError(
                "Boundary-conditioned source and method are required"
            )
        if self.segment_type not in {
            PsatSegmentType.COMPLETION,
            PsatSegmentType.FALLBACK,
        }:
            raise PsatCanonicalizationError(
                "Boundary-conditioned segments must be completion or fallback relations"
            )
        _validate_temperature_range(self.T_min, self.T_max)
        _validate_pressure_range(self.P_min_bar, self.P_max_bar)
        if not isinstance(self.priority, int):
            raise PsatCanonicalizationError("Psat segment priority must be an integer")
        if not isinstance(self.allow_junction_slope_mismatch, bool):
            raise PsatCanonicalizationError(
                "Boundary-conditioned junction slope mismatch allowance "
                "must be boolean"
            )
        if not isinstance(self.requires_no_hard_segments, bool):
            raise PsatCanonicalizationError(
                "Boundary-conditioned no-hard-segment requirement must be boolean"
            )
        if not math.isfinite(self.quality) or not 0.0 <= self.quality <= 1.0:
            raise PsatCanonicalizationError("Psat segment quality must be between 0 and 1")
        if not callable(self.evaluation_factory):
            raise PsatCanonicalizationError("A boundary evaluation factory is required")
        for label, segment_types in (
            ("left", self.allowed_left_segment_types),
            ("right", self.allowed_right_segment_types),
        ):
            if any(not isinstance(item, PsatSegmentType) for item in segment_types):
                raise PsatCanonicalizationError(
                    f"Allowed {label} boundary segment types must be PsatSegmentType values"
                )

    def bind(self, conditions: PsatBoundaryConditions) -> PsatSegment:
        tolerance = 1.0e-9 * max(self.T_max - self.T_min, 1.0)
        if (
            conditions.T_min < self.T_min - tolerance
            or conditions.T_max > self.T_max + tolerance
        ):
            raise PsatCanonicalizationError(
                f"Requested gap {conditions.T_min:g}-{conditions.T_max:g} K is outside "
                f"{self.method} validity range {self.T_min:g}-{self.T_max:g} K"
            )
        self._require_boundaries(conditions)
        evaluation = self.evaluation_factory(conditions)
        dynamic_context = {}
        dynamic_metadata = {}
        allow_junction_slope_mismatch = self.allow_junction_slope_mismatch
        if isinstance(evaluation, BoundaryConditionedPsatEvaluation):
            evaluator = evaluation.ln_pressure_function
            derivative = evaluation.derivative_function
            quality = self.quality if evaluation.quality is None else evaluation.quality
            dynamic_context.update(evaluation.context)
            dynamic_metadata.update(evaluation.metadata)
            if evaluation.allow_junction_slope_mismatch is not None:
                allow_junction_slope_mismatch = (
                    evaluation.allow_junction_slope_mismatch
                )
        elif isinstance(evaluation, tuple) and len(evaluation) == 2:
            evaluator, derivative = evaluation
            quality = self.quality
        else:
            raise PsatCanonicalizationError(
                "A boundary evaluation factory must return evaluator and derivative "
                "or BoundaryConditionedPsatEvaluation"
            )
        endpoints = [item for item in (conditions.left, conditions.right) if item]
        required_anchor_names = {
            str(name).strip().lower() for name in self.required_anchor_names
        }
        required_anchor_names.update(
            str(requirement.name).strip().lower()
            for requirement in self.anchor_requirements
        )
        required_anchor_names.update(
            str(requirement.name).strip().lower()
            for requirement in self.assembly_anchor_requirements
        )
        endpoints.extend(
            item for item in conditions.anchors
            if str(item.context.get("anchor_name") or "").strip().lower()
            in required_anchor_names
        )
        if self.inherit_boundary_quality and endpoints:
            quality = min(quality, *(item.quality for item in endpoints))
        context = dict(self.context)
        context.update(conditions.context)
        context.update(dynamic_context)
        if conditions.left is not None:
            context.setdefault("left_boundary_source", conditions.left.source)
            context.setdefault("left_boundary_method", conditions.left.method)
        if conditions.right is not None:
            context.setdefault("right_boundary_source", conditions.right.source)
            context.setdefault("right_boundary_method", conditions.right.method)
        return PsatSegment(
            source=self.source,
            method=self.method,
            segment_type=self.segment_type,
            priority=self.priority,
            T_min=conditions.T_min,
            T_max=conditions.T_max,
            ln_pressure_function=evaluator,
            derivative_function=derivative,
            quality=quality,
            P_min_bar=self.P_min_bar,
            P_max_bar=self.P_max_bar,
            context=context,
            metadata={**self.metadata, **dynamic_metadata},
            allow_junction_slope_mismatch=allow_junction_slope_mismatch,
        )

    def _require_boundaries(self, conditions: PsatBoundaryConditions) -> None:
        has_left = conditions.left is not None
        has_right = conditions.right is not None
        accepted = {
            PsatBoundaryRequirement.NONE: True,
            PsatBoundaryRequirement.LEFT: has_left,
            PsatBoundaryRequirement.RIGHT: has_right,
            PsatBoundaryRequirement.BOTH: has_left and has_right,
            PsatBoundaryRequirement.EITHER: has_left or has_right,
        }[self.boundary_requirement]
        if not accepted:
            raise PsatCanonicalizationError(
                f"{self.method} requires {self.boundary_requirement.value} boundary conditions"
            )
        derivative_requirements = (
            ("left", conditions.left, self.left_derivative_requirement),
            ("right", conditions.right, self.right_derivative_requirement),
        )
        for label, endpoint, requirement in derivative_requirements:
            if requirement is PsatDerivativeRequirement.REQUIRED and (
                endpoint is None or not endpoint.has_derivative
            ):
                raise PsatCanonicalizationError(
                    f"{self.method} requires a derivative at the {label} boundary"
                )
        boundary_type_requirements = (
            ("left", conditions.left, self.allowed_left_segment_types),
            ("right", conditions.right, self.allowed_right_segment_types),
        )
        for label, endpoint, allowed_types in boundary_type_requirements:
            if (
                endpoint is not None
                and allowed_types
                and endpoint.segment_type not in allowed_types
            ):
                allowed = ", ".join(item.value for item in allowed_types)
                actual = (
                    endpoint.segment_type.value
                    if endpoint.segment_type is not None
                    else "fixed anchor"
                )
                raise PsatCanonicalizationError(
                    f"{self.method} requires a {label} boundary from "
                    f"{allowed}; received {actual}"
                )
        missing_anchors = [
            name for name in self.required_anchor_names
            if conditions.anchor(name) is None
        ]
        missing_anchors.extend(
            requirement.name
            for requirement in self.anchor_requirements
            if conditions.anchor(requirement.name) is None
        )
        missing_anchors.extend(
            requirement.name
            for requirement in self.assembly_anchor_requirements
            if requirement.required and conditions.anchor(requirement.name) is None
        )
        if missing_anchors:
            raise PsatCanonicalizationError(
                f"{self.method} requires fixed anchor(s): {', '.join(missing_anchors)}"
            )
        missing_anchor_derivatives = [
            requirement.name
            for requirement in self.anchor_requirements
            if requirement.derivative_requirement is PsatDerivativeRequirement.REQUIRED
            and not conditions.anchor(requirement.name).has_derivative
        ]
        if missing_anchor_derivatives:
            raise PsatCanonicalizationError(
                f"{self.method} requires anchor derivative(s): "
                f"{', '.join(missing_anchor_derivatives)}"
            )
        invalid_assembly_anchor_types = []
        for requirement in self.assembly_anchor_requirements:
            endpoint = conditions.anchor(requirement.name)
            if (
                endpoint is not None
                and requirement.allowed_segment_types
                and endpoint.segment_type not in requirement.allowed_segment_types
            ):
                invalid_assembly_anchor_types.append(requirement.name)
        if invalid_assembly_anchor_types:
            raise PsatCanonicalizationError(
                f"{self.method} requires assembly anchor(s) from allowed segment "
                f"types: {', '.join(invalid_assembly_anchor_types)}"
            )
        missing_assembly_anchor_derivatives = [
            requirement.name
            for requirement in self.assembly_anchor_requirements
            if (
                conditions.anchor(requirement.name) is not None
                and requirement.derivative_requirement
                is PsatDerivativeRequirement.REQUIRED
                and not conditions.anchor(requirement.name).has_derivative
            )
        ]
        if missing_assembly_anchor_derivatives:
            raise PsatCanonicalizationError(
                f"{self.method} requires assembly anchor derivative(s): "
                f"{', '.join(missing_assembly_anchor_derivatives)}"
            )


@dataclass(frozen=True)
class PsatJunctionPolicy:
    """Acceptance limits for directly adjoining source segments."""

    max_absolute_relative_pressure_mismatch: float = 0.02
    max_absolute_relative_slope_mismatch: float = 0.10
    slope_mismatch_exception_quality_factor: float = 0.90
    require_positive_slopes: bool = True

    def __post_init__(self) -> None:
        for value in (
            self.max_absolute_relative_pressure_mismatch,
            self.max_absolute_relative_slope_mismatch,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise PsatCanonicalizationError(
                    "Junction tolerances must be nonnegative"
                )
        if (
            not math.isfinite(self.slope_mismatch_exception_quality_factor)
            or not 0.0 <= self.slope_mismatch_exception_quality_factor <= 1.0
        ):
            raise PsatCanonicalizationError(
                "Junction slope-exception quality factor must be between 0 and 1"
            )


class PsatHandoffAction(str, Enum):
    """Outcome selected for one adjoining pair of source segments."""

    DIRECT = "direct"
    BRIDGE = "bridge"
    REJECT = "reject"


class PsatGapConsistency(str, Enum):
    """Diagnostic classification for an uncovered interval between sources."""

    OPEN_ENDED = "open_ended"
    C1_CONNECTABLE = "c1_connectable"
    STRESSED = "stressed"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True)
class PsatGapAssessment:
    """Whether two source boundaries can plausibly admit a smooth continuation."""

    gap: PsatGap
    status: PsatGapConsistency
    left: Optional[PsatSegmentSlice]
    right: Optional[PsatSegmentSlice]
    secant_dln_pressure_dT: Optional[float] = None
    left_slope_ratio: Optional[float] = None
    right_slope_ratio: Optional[float] = None
    continuation_possible: bool = False
    reason: str = ""


@dataclass(frozen=True)
class PsatHandoffPolicy:
    """Generic overlap-search and smoothing policy for pinned source segments."""

    direct_junction_policy: PsatJunctionPolicy = field(
        default_factory=PsatJunctionPolicy,
    )
    smoothing_max_absolute_relative_pressure_mismatch: float = 0.025
    smoothing_max_absolute_relative_slope_mismatch: float = 0.15
    search_sample_count: int = 201
    bridge_width_fractions: tuple[float, ...] = (
        0.10,
        0.20,
        0.35,
        0.50,
        0.75,
        1.00,
    )
    bridge_probe_points: int = 201

    def __post_init__(self) -> None:
        if self.search_sample_count < 3:
            raise PsatCanonicalizationError(
                "Psat handoff search requires at least three samples"
            )
        if self.bridge_probe_points < 3:
            raise PsatCanonicalizationError(
                "Psat handoff bridge probing requires at least three samples"
            )
        limits = (
            self.smoothing_max_absolute_relative_pressure_mismatch,
            self.smoothing_max_absolute_relative_slope_mismatch,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in limits):
            raise PsatCanonicalizationError(
                "Psat handoff smoothing tolerances must be nonnegative"
            )
        if (
            self.smoothing_max_absolute_relative_pressure_mismatch
            < self.direct_junction_policy.max_absolute_relative_pressure_mismatch
            or self.smoothing_max_absolute_relative_slope_mismatch
            < self.direct_junction_policy.max_absolute_relative_slope_mismatch
        ):
            raise PsatCanonicalizationError(
                "Psat handoff smoothing tolerances cannot be stricter than direct tolerances"
            )
        if not self.bridge_width_fractions or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in self.bridge_width_fractions
        ):
            raise PsatCanonicalizationError(
                "Psat handoff bridge widths must lie in (0, 1]"
            )


@dataclass(frozen=True)
class PsatHandoffDecision:
    """Auditable decision for one source-to-source handoff."""

    action: PsatHandoffAction
    left_method: str
    right_method: str
    overlap_T_min: Optional[float]
    overlap_T_max: Optional[float]
    handoff_temperature: Optional[float] = None
    bridge_T_min: Optional[float] = None
    bridge_T_max: Optional[float] = None
    rejected_method: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True)
class PsatHandoffResult:
    """Coordinated pinned assembly and its retained concrete segments."""

    assembly: "PsatAssembly"
    segments: tuple[PsatSegment, ...]
    bridge_segments: tuple[PsatSegment, ...]
    rejected_segments: tuple[PsatSegment, ...]
    decisions: tuple[PsatHandoffDecision, ...]
    gap_assessments: tuple[PsatGapAssessment, ...] = ()


@dataclass(frozen=True)
class PsatJunctionAssessment:
    """Result of applying a junction policy to adjacent segments."""

    junction: "PsatJunction"
    accepted: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PsatJunction:
    """Value and derivative mismatch between adjacent selected segments."""

    temperature: float
    left: PsatSegmentSlice
    right: PsatSegmentSlice
    delta_ln_pressure: float
    delta_dln_pressure_dT: float

    @property
    def relative_pressure_mismatch(self) -> float:
        try:
            return math.expm1(self.delta_ln_pressure)
        except OverflowError:
            return math.copysign(math.inf, self.delta_ln_pressure)

    @property
    def relative_slope_mismatch(self) -> float:
        left_slope = self.left.dln_pressure_dT(self.temperature)
        right_slope = self.right.dln_pressure_dT(self.temperature)
        scale = max(abs(left_slope), abs(right_slope), 1.0e-15)
        return self.delta_dln_pressure_dT / scale

    def assess(
        self,
        policy: Optional[PsatJunctionPolicy] = None,
    ) -> PsatJunctionAssessment:
        policy = policy or PsatJunctionPolicy()
        reasons = []
        warnings = []
        pressure_mismatch = abs(self.relative_pressure_mismatch)
        slope_mismatch = abs(self.relative_slope_mismatch)
        left_slope = self.left.dln_pressure_dT(self.temperature)
        right_slope = self.right.dln_pressure_dT(self.temperature)
        if pressure_mismatch > policy.max_absolute_relative_pressure_mismatch:
            reasons.append(
                f"relative pressure mismatch {pressure_mismatch:.6g} exceeds "
                f"{policy.max_absolute_relative_pressure_mismatch:.6g}"
            )
        if slope_mismatch > policy.max_absolute_relative_slope_mismatch:
            message = (
                f"relative slope mismatch {slope_mismatch:.6g} exceeds "
                f"{policy.max_absolute_relative_slope_mismatch:.6g}"
            )
            excepting_methods = tuple(
                item.segment.method
                for item in (self.left, self.right)
                if item.segment.allow_junction_slope_mismatch
            )
            if excepting_methods:
                warnings.append(
                    f"{message}; accepted by slope-mismatch exception on "
                    f"{', '.join(excepting_methods)}"
                )
            else:
                reasons.append(message)
        if policy.require_positive_slopes and (left_slope <= 0.0 or right_slope <= 0.0):
            reasons.append("one or both endpoint slopes are nonpositive")
        return PsatJunctionAssessment(
            self,
            not reasons,
            tuple(reasons),
            tuple(warnings),
        )


@dataclass(frozen=True)
class PsatAssembly:
    """Priority-selected source intervals before completion or regression."""

    target_T_min: float
    target_T_max: float
    slices: tuple[PsatSegmentSlice, ...]

    def __post_init__(self) -> None:
        _validate_temperature_range(self.target_T_min, self.target_T_max)
        ordered = tuple(sorted(self.slices, key=lambda item: (item.T_min, item.T_max)))
        if ordered != self.slices:
            object.__setattr__(self, "slices", ordered)
        for item in self.slices:
            if item.T_min < self.target_T_min or item.T_max > self.target_T_max:
                raise PsatCanonicalizationError("A selected slice exceeds the target range")
        for left, right in zip(self.slices, self.slices[1:]):
            if right.T_min < left.T_max - 1.0e-9:
                raise PsatCanonicalizationError("Selected Psat slices overlap internally")

    def coverage_gaps(self, tolerance: float = 1.0e-9) -> tuple[PsatGap, ...]:
        gaps = []
        cursor = self.target_T_min
        for item in self.slices:
            if item.T_max <= cursor + tolerance:
                cursor = max(cursor, item.T_max)
                continue
            if item.T_min > cursor + tolerance:
                gaps.append(PsatGap(cursor, item.T_min))
            cursor = max(cursor, item.T_max)
        if cursor < self.target_T_max - tolerance:
            gaps.append(PsatGap(cursor, self.target_T_max))
        return tuple(gaps)

    def covers_target(self, tolerance: float = 1.0e-9) -> bool:
        return not self.coverage_gaps(tolerance)

    def slice_at(self, T: float) -> Optional[PsatSegmentSlice]:
        candidates = [item for item in self.slices if item.covers_temperature(T)]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.priority)

    def ln_pressure(self, T: float) -> float:
        item = self.slice_at(T)
        if item is None:
            raise PsatCanonicalizationError(f"No assembled Psat segment covers {T:g} K")
        return item.ln_pressure(T)

    def dln_pressure_dT(self, T: float) -> float:
        item = self.slice_at(T)
        if item is None:
            raise PsatCanonicalizationError(f"No assembled Psat segment covers {T:g} K")
        return item.dln_pressure_dT(T)

    def junctions(self, tolerance: float = 1.0e-9) -> tuple[PsatJunction, ...]:
        junctions = []
        for left, right in zip(self.slices, self.slices[1:]):
            if abs(left.T_max - right.T_min) > tolerance:
                continue
            temperature = 0.5 * (left.T_max + right.T_min)
            left_ln_pressure = left.ln_pressure(temperature)
            right_ln_pressure = right.ln_pressure(temperature)
            left_slope = left.dln_pressure_dT(temperature)
            right_slope = right.dln_pressure_dT(temperature)
            junctions.append(PsatJunction(
                temperature=temperature,
                left=left,
                right=right,
                delta_ln_pressure=right_ln_pressure - left_ln_pressure,
                delta_dln_pressure_dT=right_slope - left_slope,
            ))
        return tuple(junctions)

    def assess_junctions(
        self,
        policy: Optional[PsatJunctionPolicy] = None,
        tolerance: float = 1.0e-9,
    ) -> tuple[PsatJunctionAssessment, ...]:
        return tuple(
            junction.assess(policy)
            for junction in self.junctions(tolerance)
        )

    def require_compatible_junctions(
        self,
        policy: Optional[PsatJunctionPolicy] = None,
        tolerance: float = 1.0e-9,
    ) -> None:
        rejected = [
            assessment
            for assessment in self.assess_junctions(policy, tolerance)
            if not assessment.accepted
        ]
        if not rejected:
            return
        details = []
        for assessment in rejected:
            junction = assessment.junction
            details.append(
                f"{junction.left.segment.method}->{junction.right.segment.method} "
                f"at {junction.temperature:g} K: {', '.join(assessment.reasons)}"
            )
        raise PsatCanonicalizationError(
            "Incompatible Psat segment junctions: " + "; ".join(details)
        )

    def junction_warnings(
        self,
        policy: Optional[PsatJunctionPolicy] = None,
        tolerance: float = 1.0e-9,
    ) -> tuple[str, ...]:
        warnings = []
        for assessment in self.assess_junctions(policy, tolerance):
            junction = assessment.junction
            warnings.extend(
                f"{junction.left.segment.method}->{junction.right.segment.method} "
                f"at {junction.temperature:g} K: {warning}"
                for warning in assessment.warnings
            )
        return tuple(warnings)


class PsatSegmentAssembler:
    """Fill a target interval by descending priority without replacing winners."""

    def __init__(self, target_T_min: float, target_T_max: float):
        _validate_temperature_range(target_T_min, target_T_max)
        self.target_T_min = float(target_T_min)
        self.target_T_max = float(target_T_max)
        self._segments: list[PsatSegment] = []

    @property
    def segments(self) -> tuple[PsatSegment, ...]:
        return tuple(self._segments)

    def add(self, segment: PsatSegment) -> None:
        if not isinstance(segment, PsatSegment):
            raise TypeError("PsatSegmentAssembler accepts PsatSegment instances")
        self._segments.append(segment)

    def extend(self, segments: Iterable[PsatSegment]) -> None:
        for segment in segments:
            self.add(segment)

    def replace_segments(self, segments: Iterable[PsatSegment]) -> None:
        """Replace candidate segments after a coordination stage."""
        replacement = tuple(segments)
        if any(not isinstance(segment, PsatSegment) for segment in replacement):
            raise TypeError("PsatSegmentAssembler accepts PsatSegment instances")
        self._segments = list(replacement)

    def assemble(
        self,
        segment_types: Optional[Sequence[PsatSegmentType]] = None,
        tolerance: float = 1.0e-9,
    ) -> PsatAssembly:
        allowed_types = set(segment_types) if segment_types is not None else None
        indexed = [
            (index, segment)
            for index, segment in enumerate(self._segments)
            if allowed_types is None or segment.segment_type in allowed_types
        ]
        indexed.sort(key=lambda item: (-item[1].priority, item[0]))
        uncovered = [(self.target_T_min, self.target_T_max)]
        selected = []
        for _index, segment in indexed:
            next_uncovered = []
            for gap_min, gap_max in uncovered:
                selected_min = max(gap_min, segment.T_min)
                selected_max = min(gap_max, segment.T_max)
                if selected_max <= selected_min + tolerance:
                    next_uncovered.append((gap_min, gap_max))
                    continue
                selected.append(PsatSegmentSlice(segment, selected_min, selected_max))
                if selected_min > gap_min + tolerance:
                    next_uncovered.append((gap_min, selected_min))
                if selected_max < gap_max - tolerance:
                    next_uncovered.append((selected_max, gap_max))
            uncovered = next_uncovered
            if not uncovered:
                break
        selected.sort(key=lambda item: (item.T_min, item.T_max, -item.priority))
        return PsatAssembly(
            target_T_min=self.target_T_min,
            target_T_max=self.target_T_max,
            slices=tuple(selected),
        )

    def assemble_pinned(self, tolerance: float = 1.0e-9) -> PsatAssembly:
        return self.assemble(
            segment_types=(
                PsatSegmentType.CANONICAL_OVERRIDE,
                PsatSegmentType.PINNED,
            ),
            tolerance=tolerance,
        )


class PsatAnchorRegistry:
    """Named fixed Psat points available to every completion relation."""

    def __init__(self, points: Iterable[PsatEndpoint] = ()):
        self._points: list[PsatEndpoint] = []
        for point in points:
            self.add(point)

    @property
    def points(self) -> tuple[PsatEndpoint, ...]:
        return tuple(self._points)

    def add(self, point: PsatEndpoint) -> None:
        if not isinstance(point, PsatEndpoint):
            raise TypeError("PsatAnchorRegistry accepts PsatEndpoint instances")
        name = str(point.context.get("anchor_name") or "").strip().lower()
        if name:
            existing = [
                item for item in self._points
                if str(item.context.get("anchor_name") or "").strip().lower() == name
            ]
            if existing and max(item.quality for item in existing) > point.quality:
                return
            self._points = [item for item in self._points if item not in existing]
        self._points.append(point)

    def named(self, name: str) -> Optional[PsatEndpoint]:
        normalized = str(name).strip().lower()
        candidates = [
            item for item in self._points
            if str(item.context.get("anchor_name") or "").strip().lower() == normalized
        ]
        return max(candidates, key=lambda item: item.quality) if candidates else None

    def at_temperature(
        self,
        temperature: float,
        tolerance: float = 1.0e-7,
    ) -> Optional[PsatEndpoint]:
        candidates = [
            item for item in self._points
            if abs(item.temperature - temperature) <= tolerance
        ]
        return max(candidates, key=lambda item: item.quality) if candidates else None

    @classmethod
    def from_tb_tc(
        cls,
        *,
        T_critical: float,
        P_critical_bar: float,
        critical_quality: float,
        T_boiling: Optional[float] = None,
        boiling_quality: Optional[float] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> "PsatAnchorRegistry":
        common_context = dict(context or {})
        critical_context = dict(common_context)
        critical_context["anchor_name"] = "Tc"
        points = [PsatEndpoint.fixed_pressure_point(
            T_critical,
            P_critical_bar,
            source="fixed_anchor",
            method="critical_point",
            quality=critical_quality,
            context=critical_context,
        )]
        if T_boiling is not None:
            boiling_context = dict(common_context)
            boiling_context["anchor_name"] = "Tb"
            points.append(PsatEndpoint.fixed_pressure_point(
                T_boiling,
                1.01325,
                source="fixed_anchor",
                method="normal_boiling_point",
                quality=(
                    critical_quality if boiling_quality is None else boiling_quality
                ),
                context=boiling_context,
            ))
        return cls(points)


@dataclass(frozen=True)
class PsatCompletionResult:
    """Completed assembly plus generated segments and rejected attempts."""

    assembly: PsatAssembly
    generated_segments: tuple[PsatSegment, ...]
    rejected_attempts: tuple[str, ...]
    handoffs: Optional[PsatHandoffResult] = None
    junction_warnings: tuple[str, ...] = ()


class PsatCompletionCoordinator:
    """Bind deferred relations into uncovered intervals by declared contracts."""

    def __init__(
        self,
        assembler: PsatSegmentAssembler,
        relations: Iterable[BoundaryConditionedPsatSegment],
        anchors: Optional[PsatAnchorRegistry] = None,
        *,
        segment_probe_points: int = 201,
        handoff_policy: Optional[PsatHandoffPolicy] = None,
    ):
        self.assembler = assembler
        self.relations = tuple(relations)
        self.anchors = anchors or PsatAnchorRegistry()
        self.segment_probe_points = segment_probe_points
        self.handoff_policy = handoff_policy

    def complete(
        self,
        *,
        require_complete: bool = True,
        junction_policy: Optional[PsatJunctionPolicy] = None,
        max_iterations: int = 100,
    ) -> PsatCompletionResult:
        generated = []
        hard_segments = tuple(
            segment
            for segment in self.assembler.segments
            if segment.segment_type.is_hard_pinned
        )
        other_segments = tuple(
            segment
            for segment in self.assembler.segments
            if not segment.segment_type.is_hard_pinned
        )
        handoffs = PsatHandoffCoordinator(
            self.assembler.target_T_min,
            self.assembler.target_T_max,
            self.handoff_policy or PsatHandoffPolicy(
                direct_junction_policy=junction_policy or PsatJunctionPolicy(),
            ),
        ).coordinate(hard_segments)
        self.assembler.replace_segments((*handoffs.segments, *other_segments))
        rejected = [
            (
                f"handoff {decision.left_method}->{decision.right_method}: "
                f"rejected {decision.rejected_method}; {decision.reason}"
            )
            for decision in handoffs.decisions
            if decision.action is PsatHandoffAction.REJECT
        ]
        ordered_relations = sorted(
            enumerate(self.relations),
            key=lambda item: (-item[1].priority, item[0]),
        )
        for _iteration in range(max_iterations):
            assembly = self.assembler.assemble()
            gaps = assembly.coverage_gaps()
            if not gaps:
                assembly, generated = (
                    self._apply_junction_slope_exception_penalties(
                        assembly,
                        generated,
                        junction_policy,
                    )
                )
                assembly.require_compatible_junctions(junction_policy)
                return PsatCompletionResult(
                    assembly,
                    tuple(generated),
                    tuple(rejected),
                    handoffs,
                    assembly.junction_warnings(junction_policy),
                )
            progress = False
            for gap in gaps:
                for _index, relation in ordered_relations:
                    try:
                        conditions = self._conditions_for(gap, assembly, relation)
                        if conditions is None:
                            continue
                        segment = relation.bind(conditions)
                        segment = trim_psat_segment_to_validity(
                            segment,
                            sample_count=self.segment_probe_points,
                        )
                        probe = probe_psat_segment(
                            segment,
                            sample_count=self.segment_probe_points,
                        )
                        if not probe.finite or not probe.monotonic:
                            raise PsatCanonicalizationError(
                                "generated segment is not finite and strictly monotonic"
                            )
                    except PsatCanonicalizationError as error:
                        rejected.append(
                            f"{relation.method} for {gap.T_min:g}-{gap.T_max:g} K: {error}"
                        )
                        continue
                    self.assembler.add(segment)
                    generated.append(segment)
                    progress = True
                    break
                if progress:
                    break
            if not progress:
                break

        assembly = self.assembler.assemble()
        if require_complete and not assembly.covers_target():
            gaps = ", ".join(
                f"{gap.T_min:g}-{gap.T_max:g} K"
                for gap in assembly.coverage_gaps()
            )
            raise PsatCanonicalizationError(
                f"Unable to complete Psat target range; uncovered gaps: {gaps}"
            )
        if assembly.covers_target():
            assembly, generated = (
                self._apply_junction_slope_exception_penalties(
                    assembly,
                    generated,
                    junction_policy,
                )
            )
            assembly.require_compatible_junctions(junction_policy)
        return PsatCompletionResult(
            assembly,
            tuple(generated),
            tuple(rejected),
            handoffs,
            (
                assembly.junction_warnings(junction_policy)
                if assembly.covers_target()
                else ()
            ),
        )

    def _apply_junction_slope_exception_penalties(
        self,
        assembly: PsatAssembly,
        generated: list[PsatSegment],
        policy: Optional[PsatJunctionPolicy],
    ) -> tuple[PsatAssembly, list[PsatSegment]]:
        active_policy = policy or PsatJunctionPolicy()
        affected_mismatches: dict[int, float] = {}
        for assessment in assembly.assess_junctions(active_policy):
            if not assessment.warnings:
                continue
            mismatch = abs(assessment.junction.relative_slope_mismatch)
            for item in (
                assessment.junction.left,
                assessment.junction.right,
            ):
                if item.segment.allow_junction_slope_mismatch:
                    segment_id = id(item.segment)
                    affected_mismatches[segment_id] = max(
                        mismatch,
                        affected_mismatches.get(segment_id, 0.0),
                    )
        if not affected_mismatches:
            return assembly, generated

        replacements = {}
        updated_segments = []
        for segment in self.assembler.segments:
            segment_id = id(segment)
            if (
                segment_id not in affected_mismatches
                or segment.metadata.get(
                    "junction_slope_exception_penalty_applied"
                )
            ):
                updated_segments.append(segment)
                continue
            metadata = dict(segment.metadata)
            metadata.update({
                "junction_slope_exception_penalty_applied": True,
                "junction_slope_exception_original_quality": segment.quality,
                "junction_slope_exception_quality_factor": (
                    active_policy.slope_mismatch_exception_quality_factor
                ),
                "junction_slope_exception_maximum_mismatch": (
                    affected_mismatches[segment_id]
                ),
            })
            replacement = replace(
                segment,
                quality=(
                    segment.quality
                    * active_policy.slope_mismatch_exception_quality_factor
                ),
                metadata=metadata,
            )
            replacements[segment_id] = replacement
            updated_segments.append(replacement)

        if not replacements:
            return assembly, generated
        self.assembler.replace_segments(updated_segments)
        return (
            self.assembler.assemble(),
            [
                replacements.get(id(segment), segment)
                for segment in generated
            ],
        )

    def _conditions_for(
        self,
        gap: PsatGap,
        assembly: PsatAssembly,
        relation: BoundaryConditionedPsatSegment,
    ) -> Optional[PsatBoundaryConditions]:
        if relation.requires_no_hard_segments and any(
            item.segment.is_hard_pinned
            for item in assembly.slices
        ):
            return None
        overlap_min = max(gap.T_min, relation.T_min)
        overlap_max = min(gap.T_max, relation.T_max)
        if overlap_max <= overlap_min:
            return None
        left_at_gap = self._boundary_at(gap.T_min, assembly, prefer_left=True)
        right_at_gap = self._boundary_at(gap.T_max, assembly, prefer_left=False)
        requirement = relation.boundary_requirement
        if requirement is PsatBoundaryRequirement.BOTH:
            if left_at_gap is None or right_at_gap is None:
                return None
            if overlap_min > gap.T_min or overlap_max < gap.T_max:
                return None
            T_min, T_max = gap.T_min, gap.T_max
            left, right = left_at_gap, right_at_gap
        elif requirement is PsatBoundaryRequirement.LEFT:
            if left_at_gap is None or overlap_min > gap.T_min:
                return None
            T_min, T_max = gap.T_min, overlap_max
            left, right = left_at_gap, None
        elif requirement is PsatBoundaryRequirement.RIGHT:
            if right_at_gap is None or overlap_max < gap.T_max:
                return None
            T_min, T_max = overlap_min, gap.T_max
            left, right = None, right_at_gap
        elif requirement is PsatBoundaryRequirement.EITHER:
            if left_at_gap is not None and overlap_min <= gap.T_min:
                T_min, T_max = gap.T_min, overlap_max
                left, right = left_at_gap, None
            elif right_at_gap is not None and overlap_max >= gap.T_max:
                T_min, T_max = overlap_min, gap.T_max
                left, right = None, right_at_gap
            else:
                return None
        else:
            T_min, T_max = overlap_min, overlap_max
            left = left_at_gap if abs(T_min - gap.T_min) <= 1.0e-9 else None
            right = right_at_gap if abs(T_max - gap.T_max) <= 1.0e-9 else None
        anchors = list(self.anchors.points)
        for requirement in relation.assembly_anchor_requirements:
            item = assembly.slice_at(requirement.temperature)
            if item is None:
                continue
            endpoint = PsatEndpoint.from_slice(item, requirement.temperature)
            anchor_context = dict(endpoint.context)
            anchor_context["anchor_name"] = requirement.name
            anchors.append(PsatEndpoint(
                temperature=endpoint.temperature,
                ln_pressure=endpoint.ln_pressure,
                dln_pressure_dT=endpoint.dln_pressure_dT,
                source=endpoint.source,
                method=endpoint.method,
                quality=endpoint.quality,
                derivative_basis=endpoint.derivative_basis,
                context=anchor_context,
                segment_type=endpoint.segment_type,
            ))
        return PsatBoundaryConditions(
            T_min=T_min,
            T_max=T_max,
            left=left,
            right=right,
            anchors=tuple(anchors),
            context={
                "target_T_min": self.assembler.target_T_min,
                "target_T_max": self.assembler.target_T_max,
            },
        )

    def _boundary_at(
        self,
        temperature: float,
        assembly: PsatAssembly,
        *,
        prefer_left: bool,
    ) -> Optional[PsatEndpoint]:
        tolerance = 1.0e-7 * max(assembly.target_T_max - assembly.target_T_min, 1.0)
        candidates = []
        for item in assembly.slices:
            boundary_temperature = item.T_max if prefer_left else item.T_min
            if abs(boundary_temperature - temperature) <= tolerance:
                candidates.append(PsatEndpoint.from_slice(item, boundary_temperature))
        if candidates:
            return max(candidates, key=lambda item: item.quality)
        return self.anchors.at_temperature(temperature, tolerance)


def trim_psat_segment_to_validity(
    segment: PsatSegment,
    sample_count: int = 201,
) -> PsatSegment:
    """Trim a monotonic segment to its declared pressure-valid interval."""
    if sample_count < 3:
        raise PsatCanonicalizationError("At least three validity probe points are required")
    temperatures = np.linspace(segment.T_min, segment.T_max, sample_count)
    try:
        ln_pressures = np.asarray([
            segment.raw_ln_pressure(float(temperature))
            for temperature in temperatures
        ])
        slopes = np.asarray([
            segment.raw_dln_pressure_dT(float(temperature))
            for temperature in temperatures
        ])
    except (ArithmeticError, PsatCanonicalizationError, TypeError, ValueError) as error:
        raise PsatCanonicalizationError(
            "Unable to probe segment validity"
        ) from error
    if (
        np.any(~np.isfinite(ln_pressures))
        or np.any(~np.isfinite(slopes))
        or np.min(slopes) <= 0.0
        or np.any(np.diff(ln_pressures) <= 0.0)
    ):
        raise PsatCanonicalizationError(
            "Pressure validity trimming requires a strictly monotonic relation"
        )

    T_min = segment.T_min
    T_max = segment.T_max
    if segment.P_min_bar is not None:
        target = math.log(segment.P_min_bar)
        lower_value = segment.raw_ln_pressure(T_min)
        upper_value = segment.raw_ln_pressure(T_max)
        if upper_value < target - 1.0e-10:
            raise PsatCanonicalizationError("Segment never reaches its minimum pressure")
        if lower_value < target:
            if abs(upper_value - target) <= 1.0e-10:
                T_min = T_max
            else:
                T_min = brentq(
                    lambda T: segment.raw_ln_pressure(T) - target,
                    T_min,
                    T_max,
                    xtol=1.0e-10,
                    rtol=1.0e-12,
                )
    if segment.P_max_bar is not None:
        target = math.log(segment.P_max_bar)
        lower_value = segment.raw_ln_pressure(T_min)
        upper_value = segment.raw_ln_pressure(T_max)
        if lower_value > target + 1.0e-10:
            raise PsatCanonicalizationError("Segment starts above its maximum pressure")
        if upper_value > target:
            if abs(lower_value - target) <= 1.0e-10:
                T_max = T_min
            else:
                T_max = brentq(
                    lambda T: segment.raw_ln_pressure(T) - target,
                    T_min,
                    T_max,
                    xtol=1.0e-10,
                    rtol=1.0e-12,
                )
    if T_max <= T_min + 1.0e-10:
        raise PsatCanonicalizationError("Pressure validity leaves no segment interval")
    return segment.clipped(T_min, T_max)


@dataclass(frozen=True)
class PsatSegmentProbe:
    """Finite and monotonicity diagnostics for one bounded segment."""

    sample_count: int
    finite: bool
    monotonic: bool
    minimum_dln_pressure_dT: float
    maximum_dln_pressure_dT: float
    minimum_ln_pressure: float
    maximum_ln_pressure: float


def probe_psat_segment(
    segment: PsatSegment,
    sample_count: int = 201,
) -> PsatSegmentProbe:
    if sample_count < 3:
        raise PsatCanonicalizationError("At least three segment probe points are required")
    temperatures = np.linspace(segment.T_min, segment.T_max, sample_count)
    ln_pressures = []
    slopes = []
    try:
        for temperature in temperatures:
            ln_pressures.append(segment.ln_pressure(float(temperature)))
            slopes.append(segment.dln_pressure_dT(float(temperature)))
    except (ArithmeticError, PsatCanonicalizationError, TypeError, ValueError):
        return PsatSegmentProbe(
            sample_count=sample_count,
            finite=False,
            monotonic=False,
            minimum_dln_pressure_dT=math.nan,
            maximum_dln_pressure_dT=math.nan,
            minimum_ln_pressure=math.nan,
            maximum_ln_pressure=math.nan,
        )
    finite = all(math.isfinite(value) for value in (*ln_pressures, *slopes))
    monotonic = finite and min(slopes) > 0.0 and all(
        right > left
        for left, right in zip(ln_pressures, ln_pressures[1:])
    )
    return PsatSegmentProbe(
        sample_count=sample_count,
        finite=finite,
        monotonic=monotonic,
        minimum_dln_pressure_dT=min(slopes),
        maximum_dln_pressure_dT=max(slopes),
        minimum_ln_pressure=min(ln_pressures),
        maximum_ln_pressure=max(ln_pressures),
    )


def make_c1_bridge_segment(
    left: PsatEndpoint,
    right: PsatEndpoint,
    *,
    source: str,
    method: str,
    priority: int = PsatPriority.PRIMARY_COMPLETION,
    quality: Optional[float] = None,
    baseline: Optional[PsatSegment] = None,
    context: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    coordinate: str = "temperature",
    require_monotonic: bool = True,
    probe_points: int = 201,
) -> PsatSegment:
    """Create a cubic C1 bridge, optionally as a correction to a baseline."""
    if right.temperature <= left.temperature:
        raise PsatCanonicalizationError("A C1 bridge requires ordered endpoints")
    if not left.has_derivative or not right.has_derivative:
        raise PsatCanonicalizationError(
            "A C1 bridge requires derivatives at both endpoints"
        )
    if baseline is not None and (
        not baseline.covers_temperature(left.temperature)
        or not baseline.covers_temperature(right.temperature)
    ):
        raise PsatCanonicalizationError("The C1 bridge baseline does not cover the gap")

    if coordinate == "temperature":
        coordinate_function = lambda T: T
        coordinate_derivative = lambda _T: 1.0
    elif coordinate == "log_temperature":
        coordinate_function = math.log
        coordinate_derivative = lambda T: 1.0 / T
    else:
        raise PsatCanonicalizationError(
            f"Unsupported C1 bridge coordinate {coordinate!r}"
        )
    left_coordinate = coordinate_function(left.temperature)
    right_coordinate = coordinate_function(right.temperature)
    coordinate_span = right_coordinate - left_coordinate
    if baseline is None:
        left_value = left.ln_pressure
        right_value = right.ln_pressure
        left_temperature_slope = left.dln_pressure_dT
        right_temperature_slope = right.dln_pressure_dT
    else:
        left_value = left.ln_pressure - baseline.ln_pressure(left.temperature)
        right_value = right.ln_pressure - baseline.ln_pressure(right.temperature)
        left_temperature_slope = (
            left.dln_pressure_dT
            - baseline.dln_pressure_dT(left.temperature)
        )
        right_temperature_slope = (
            right.dln_pressure_dT
            - baseline.dln_pressure_dT(right.temperature)
        )
    left_coordinate_slope = (
        left_temperature_slope
        / coordinate_derivative(left.temperature)
    )
    right_coordinate_slope = (
        right_temperature_slope
        / coordinate_derivative(right.temperature)
    )

    def correction(T: float) -> float:
        fraction = (
            coordinate_function(T) - left_coordinate
        ) / coordinate_span
        fraction2 = fraction * fraction
        fraction3 = fraction2 * fraction
        h00 = 2.0 * fraction3 - 3.0 * fraction2 + 1.0
        h10 = fraction3 - 2.0 * fraction2 + fraction
        h01 = -2.0 * fraction3 + 3.0 * fraction2
        h11 = fraction3 - fraction2
        return (
            h00 * left_value
            + h10 * coordinate_span * left_coordinate_slope
            + h01 * right_value
            + h11 * coordinate_span * right_coordinate_slope
        )

    def correction_derivative(T: float) -> float:
        fraction = (
            coordinate_function(T) - left_coordinate
        ) / coordinate_span
        fraction2 = fraction * fraction
        dh00 = 6.0 * fraction2 - 6.0 * fraction
        dh10 = 3.0 * fraction2 - 4.0 * fraction + 1.0
        dh01 = -6.0 * fraction2 + 6.0 * fraction
        dh11 = 3.0 * fraction2 - 2.0 * fraction
        coordinate_slope = (
            dh00 * left_value
            + dh10 * coordinate_span * left_coordinate_slope
            + dh01 * right_value
            + dh11 * coordinate_span * right_coordinate_slope
        ) / coordinate_span
        return coordinate_slope * coordinate_derivative(T)

    def evaluate(T: float) -> float:
        base = baseline.ln_pressure(T) if baseline is not None else 0.0
        return base + correction(T)

    def derivative(T: float) -> float:
        base = baseline.dln_pressure_dT(T) if baseline is not None else 0.0
        return base + correction_derivative(T)

    segment_quality = min(left.quality, right.quality)
    if baseline is not None:
        segment_quality = min(segment_quality, baseline.quality)
    if quality is not None:
        segment_quality = quality
    bridge_context = dict(context or {})
    bridge_context.setdefault("left_source", left.source)
    bridge_context.setdefault("left_method", left.method)
    bridge_context.setdefault("right_source", right.source)
    bridge_context.setdefault("right_method", right.method)
    if baseline is not None:
        bridge_context.setdefault("baseline_source", baseline.source)
        bridge_context.setdefault("baseline_method", baseline.method)
    bridge_metadata = dict(metadata or {})
    bridge_metadata.setdefault("bridge_coordinate", coordinate)
    segment = PsatSegment(
        source=source,
        method=method,
        segment_type=PsatSegmentType.COMPLETION,
        priority=int(priority),
        T_min=left.temperature,
        T_max=right.temperature,
        ln_pressure_function=evaluate,
        derivative_function=derivative,
        quality=segment_quality,
        context=bridge_context,
        metadata=bridge_metadata,
    )
    probe = probe_psat_segment(segment, probe_points)
    if not probe.finite:
        raise PsatCanonicalizationError("The C1 bridge produced non-finite values")
    if require_monotonic and not probe.monotonic:
        raise PsatCanonicalizationError(
            "The C1 bridge is not strictly monotonic over its range"
        )
    return segment


def assess_psat_gap_consistency(
    assembly: PsatAssembly,
    *,
    probe_points: int = 201,
) -> tuple[PsatGapAssessment, ...]:
    """Diagnose gaps without selecting or constructing a completion method."""
    if probe_points < 3:
        raise PsatCanonicalizationError(
            "Psat gap consistency probing requires at least three points"
        )
    assessments = []
    for gap in assembly.coverage_gaps():
        left = next(
            (
                item for item in reversed(assembly.slices)
                if abs(item.T_max - gap.T_min) <= 1.0e-9
            ),
            None,
        )
        right = next(
            (
                item for item in assembly.slices
                if abs(item.T_min - gap.T_max) <= 1.0e-9
            ),
            None,
        )
        if left is None or right is None:
            assessments.append(PsatGapAssessment(
                gap=gap,
                status=PsatGapConsistency.OPEN_ENDED,
                left=left,
                right=right,
                reason="only one source boundary is available",
            ))
            continue

        left_endpoint = PsatEndpoint.from_slice(left, gap.T_min)
        right_endpoint = PsatEndpoint.from_slice(right, gap.T_max)
        span = gap.T_max - gap.T_min
        pressure_span = right_endpoint.ln_pressure - left_endpoint.ln_pressure
        secant_slope = pressure_span / span
        left_slope = left_endpoint.dln_pressure_dT
        right_slope = right_endpoint.dln_pressure_dT
        left_ratio = (
            left_slope / secant_slope
            if secant_slope != 0.0 and left_slope is not None
            else None
        )
        right_ratio = (
            right_slope / secant_slope
            if secant_slope != 0.0 and right_slope is not None
            else None
        )
        if pressure_span <= 0.0:
            assessments.append(PsatGapAssessment(
                gap=gap,
                status=PsatGapConsistency.INCONSISTENT,
                left=left,
                right=right,
                secant_dln_pressure_dT=secant_slope,
                left_slope_ratio=left_ratio,
                right_slope_ratio=right_ratio,
                reason="the upper-temperature boundary does not have higher pressure",
            ))
            continue
        if left_slope is None or right_slope is None:
            assessments.append(PsatGapAssessment(
                gap=gap,
                status=PsatGapConsistency.STRESSED,
                left=left,
                right=right,
                secant_dln_pressure_dT=secant_slope,
                left_slope_ratio=left_ratio,
                right_slope_ratio=right_ratio,
                continuation_possible=True,
                reason="pressure ordering is consistent but a boundary slope is unavailable",
            ))
            continue
        if left_slope <= 0.0 or right_slope <= 0.0:
            assessments.append(PsatGapAssessment(
                gap=gap,
                status=PsatGapConsistency.INCONSISTENT,
                left=left,
                right=right,
                secant_dln_pressure_dT=secant_slope,
                left_slope_ratio=left_ratio,
                right_slope_ratio=right_ratio,
                reason="one or both source boundary slopes are nonpositive",
            ))
            continue
        try:
            make_c1_bridge_segment(
                left_endpoint,
                right_endpoint,
                source="calculated",
                method="gap_consistency_probe",
                quality=min(left.quality, right.quality),
                probe_points=probe_points,
            )
        except PsatCanonicalizationError as error:
            assessments.append(PsatGapAssessment(
                gap=gap,
                status=PsatGapConsistency.STRESSED,
                left=left,
                right=right,
                secant_dln_pressure_dT=secant_slope,
                left_slope_ratio=left_ratio,
                right_slope_ratio=right_ratio,
                continuation_possible=True,
                reason=f"ordered boundaries require a more flexible continuation: {error}",
            ))
            continue
        assessments.append(PsatGapAssessment(
            gap=gap,
            status=PsatGapConsistency.C1_CONNECTABLE,
            left=left,
            right=right,
            secant_dln_pressure_dT=secant_slope,
            left_slope_ratio=left_ratio,
            right_slope_ratio=right_ratio,
            continuation_possible=True,
            reason="a monotonic cubic C1 continuation connects both boundaries",
        ))
    return tuple(assessments)


class PsatHandoffCoordinator:
    """Choose compatible overlap handoffs before fallback completion runs."""

    def __init__(
        self,
        target_T_min: float,
        target_T_max: float,
        policy: Optional[PsatHandoffPolicy] = None,
    ):
        _validate_temperature_range(target_T_min, target_T_max)
        self.target_T_min = float(target_T_min)
        self.target_T_max = float(target_T_max)
        self.policy = policy or PsatHandoffPolicy()

    def coordinate(
        self,
        segments: Iterable[PsatSegment],
    ) -> PsatHandoffResult:
        """Coordinate hard-pinned candidates and materialize selected slices."""
        candidates = tuple(segments)
        if any(not segment.segment_type.is_hard_pinned for segment in candidates):
            raise PsatCanonicalizationError(
                "Psat handoff coordination accepts only hard-pinned segments"
            )
        insertion_order = {id(segment): index for index, segment in enumerate(candidates)}
        active = []
        rejected_segments = []
        decisions = []
        for segment in candidates:
            probe = probe_psat_segment(
                segment,
                sample_count=self.policy.bridge_probe_points,
            )
            if probe.finite and probe.monotonic:
                active.append(segment)
                continue
            rejected_segments.append(segment)
            reasons = []
            if not probe.finite:
                reasons.append("non-finite value or derivative")
            if not probe.monotonic:
                reasons.append("not strictly monotonic")
            decisions.append(PsatHandoffDecision(
                action=PsatHandoffAction.REJECT,
                left_method=segment.method,
                right_method=segment.method,
                overlap_T_min=None,
                overlap_T_max=None,
                rejected_method=segment.method,
                reason="invalid hard-pinned segment: " + ", ".join(reasons),
            ))

        while True:
            assembler = PsatSegmentAssembler(self.target_T_min, self.target_T_max)
            assembler.extend(active)
            assembly = assembler.assemble_pinned()
            coordinated, rejected, new_decisions = self._coordinate_assembly(
                assembly,
                insertion_order,
            )
            if rejected is None:
                rejected = self._unsatisfied_handoff_requirement(
                    active,
                    coordinated,
                    insertion_order,
                )
                if rejected is not None:
                    decisions.append(PsatHandoffDecision(
                        action=PsatHandoffAction.REJECT,
                        left_method=rejected.method,
                        right_method=rejected.method,
                        overlap_T_min=None,
                        overlap_T_max=None,
                        rejected_method=rejected.method,
                        reason=(
                            "required reasonable overlap with a higher-preference "
                            "hard-pinned segment was unavailable"
                        ),
                    ))
                    rejected_segments.append(rejected)
                    active = [
                        segment for segment in active if segment is not rejected
                    ]
                    continue
                decisions.extend(new_decisions)
                coordinated = self._promote_overlap_validated_quality(
                    coordinated,
                    active,
                )
                materialized, final_assembly = self._materialize(coordinated)
                bridges = tuple(
                    segment
                    for segment in materialized
                    if segment.metadata.get("handoff_bridge")
                )
                return PsatHandoffResult(
                    assembly=final_assembly,
                    segments=materialized,
                    bridge_segments=bridges,
                    rejected_segments=tuple(rejected_segments),
                    decisions=tuple(decisions),
                    gap_assessments=assess_psat_gap_consistency(
                        final_assembly,
                        probe_points=self.policy.bridge_probe_points,
                    ),
                )
            decisions.extend(
                decision
                for decision in new_decisions
                if decision.action is PsatHandoffAction.REJECT
            )
            rejected_segments.append(rejected)
            active = [segment for segment in active if segment is not rejected]

    def _coordinate_assembly(
        self,
        assembly: PsatAssembly,
        insertion_order: Mapping[int, int],
    ) -> tuple[PsatAssembly, Optional[PsatSegment], tuple[PsatHandoffDecision, ...]]:
        slices = list(assembly.slices)
        decisions = []
        index = 0
        while index < len(slices) - 1:
            left = slices[index]
            right = slices[index + 1]
            if abs(left.T_max - right.T_min) > 1.0e-9:
                index += 1
                continue
            overlap = self._overlap(left, right)
            current_assessment = self._assessment(left, right, left.T_max)
            if current_assessment is not None and current_assessment.accepted:
                decisions.append(self._direct_decision(
                    left,
                    right,
                    left.T_max,
                    "existing priority boundary is compatible",
                ))
                index += 1
                continue

            if overlap is not None:
                direct_temperature = self._direct_handoff_temperature(
                    left,
                    right,
                    overlap,
                )
                if direct_temperature is not None:
                    slices[index] = replace(left, T_max=direct_temperature)
                    slices[index + 1] = replace(right, T_min=direct_temperature)
                    decisions.append(self._direct_decision(
                        left,
                        right,
                        direct_temperature,
                        "compatible point selected inside source overlap",
                    ))
                    index += 1
                    continue

                bridge = self._smooth_bridge(left, right, overlap)
                if bridge is not None:
                    bridge_segment, bridge_T_min, bridge_T_max = bridge
                    slices[index] = replace(left, T_max=bridge_T_min)
                    slices[index + 1] = PsatSegmentSlice(
                        bridge_segment,
                        bridge_T_min,
                        bridge_T_max,
                    )
                    slices.insert(
                        index + 2,
                        replace(right, T_min=bridge_T_max),
                    )
                    decisions.append(PsatHandoffDecision(
                        action=PsatHandoffAction.BRIDGE,
                        left_method=left.segment.method,
                        right_method=right.segment.method,
                        overlap_T_min=overlap[0],
                        overlap_T_max=overlap[1],
                        bridge_T_min=bridge_T_min,
                        bridge_T_max=bridge_T_max,
                        reason="narrowest successful monotonic C1 bridge inside overlap",
                    ))
                    index += 2
                    continue

            rejected = self._lower_preference_segment(
                left.segment,
                right.segment,
                insertion_order,
            )
            reason = (
                "no compatible direct handoff or monotonic C1 overlap bridge"
                if overlap is not None
                else "incompatible touching segments have no overlap available for smoothing"
            )
            decisions.append(PsatHandoffDecision(
                action=PsatHandoffAction.REJECT,
                left_method=left.segment.method,
                right_method=right.segment.method,
                overlap_T_min=overlap[0] if overlap is not None else None,
                overlap_T_max=overlap[1] if overlap is not None else None,
                rejected_method=rejected.method,
                reason=reason,
            ))
            return assembly, rejected, tuple(decisions)

        return PsatAssembly(
            target_T_min=assembly.target_T_min,
            target_T_max=assembly.target_T_max,
            slices=tuple(slices),
        ), None, tuple(decisions)

    @staticmethod
    def _overlap(
        left: PsatSegmentSlice,
        right: PsatSegmentSlice,
    ) -> Optional[tuple[float, float]]:
        overlap_T_min = max(left.segment.T_min, right.segment.T_min, left.T_min)
        overlap_T_max = min(left.segment.T_max, right.segment.T_max, right.T_max)
        if overlap_T_max <= overlap_T_min + 1.0e-9:
            return None
        return overlap_T_min, overlap_T_max

    def _direct_handoff_temperature(
        self,
        left: PsatSegmentSlice,
        right: PsatSegmentSlice,
        overlap: tuple[float, float],
    ) -> Optional[float]:
        candidates = self._search_temperatures(overlap, left.T_max)
        accepted = []
        for temperature in candidates:
            assessment = self._assessment(left, right, temperature)
            if assessment is None or not assessment.accepted:
                continue
            accepted.append((
                abs(temperature - left.T_max),
                abs(assessment.junction.relative_pressure_mismatch),
                abs(assessment.junction.relative_slope_mismatch),
                temperature,
            ))
        return min(accepted)[-1] if accepted else None

    def _smooth_bridge(
        self,
        left: PsatSegmentSlice,
        right: PsatSegmentSlice,
        overlap: tuple[float, float],
    ) -> Optional[tuple[PsatSegment, float, float]]:
        pressure_limit = (
            self.policy.smoothing_max_absolute_relative_pressure_mismatch
        )
        slope_limit = self.policy.smoothing_max_absolute_relative_slope_mismatch
        overlap_T_min, overlap_T_max = overlap
        for width_fraction, bridge_T_min, bridge_T_max in self._bridge_intervals(
            overlap,
            left.T_max,
        ):
            if (
                bridge_T_min <= left.T_min + 1.0e-9
                or bridge_T_max >= right.T_max - 1.0e-9
                or not self._sources_are_consistent_over_bridge(
                    left,
                    right,
                    bridge_T_min,
                    bridge_T_max,
                )
            ):
                continue
            try:
                bridge = make_c1_bridge_segment(
                    PsatEndpoint.from_segment(left.segment, bridge_T_min),
                    PsatEndpoint.from_segment(right.segment, bridge_T_max),
                    source="calculated",
                    method=(
                        f"c1_handoff_{left.segment.method}_to_"
                        f"{right.segment.method}"
                    ),
                    priority=max(left.priority, right.priority),
                    context={
                        "left_priority": left.priority,
                        "right_priority": right.priority,
                        "overlap_T_min": overlap_T_min,
                        "overlap_T_max": overlap_T_max,
                    },
                    metadata={
                        "handoff_bridge": True,
                        "bridge_width_fraction": width_fraction,
                        "smoothing_pressure_limit": pressure_limit,
                        "smoothing_slope_limit": slope_limit,
                    },
                    probe_points=self.policy.bridge_probe_points,
                )
            except PsatCanonicalizationError:
                continue
            if not self._bridge_tracks_source_envelope(
                bridge,
                left.segment,
                right.segment,
            ):
                continue
            return bridge, bridge_T_min, bridge_T_max
        return None

    def _bridge_intervals(
        self,
        overlap: tuple[float, float],
        current_boundary: float,
    ) -> tuple[tuple[float, float, float], ...]:
        overlap_T_min, overlap_T_max = overlap
        overlap_span = overlap_T_max - overlap_T_min
        candidates = set()
        for width_fraction in sorted(set(self.policy.bridge_width_fractions)):
            width = overlap_span * width_fraction
            start_max = overlap_T_max - width
            starts = np.linspace(
                overlap_T_min,
                start_max,
                self.policy.search_sample_count,
            )
            preferred_start = min(
                max(current_boundary - 0.5 * width, overlap_T_min),
                start_max,
            )
            for start in (*starts, preferred_start):
                bridge_T_min = float(start)
                bridge_T_max = bridge_T_min + width
                candidates.add((
                    float(width_fraction),
                    bridge_T_min,
                    bridge_T_max,
                ))
        return tuple(sorted(
            candidates,
            key=lambda item: (
                max(
                    abs(item[1] - current_boundary),
                    abs(item[2] - current_boundary),
                ),
                item[2] - item[1],
                item[1],
            ),
        ))

    def _sources_are_consistent_over_bridge(
        self,
        left: PsatSegmentSlice,
        right: PsatSegmentSlice,
        bridge_T_min: float,
        bridge_T_max: float,
    ) -> bool:
        for temperature in np.linspace(
            bridge_T_min,
            bridge_T_max,
            self.policy.bridge_probe_points,
        ):
            metrics = self._metrics(left, right, float(temperature))
            if metrics is None:
                return False
            pressure_mismatch, slope_mismatch, left_slope, right_slope = metrics
            if (
                pressure_mismatch
                > self.policy.smoothing_max_absolute_relative_pressure_mismatch
                or slope_mismatch
                > self.policy.smoothing_max_absolute_relative_slope_mismatch
            ):
                return False
            if self.policy.direct_junction_policy.require_positive_slopes and (
                left_slope <= 0.0 or right_slope <= 0.0
            ):
                return False
        return True

    def _bridge_tracks_source_envelope(
        self,
        bridge: PsatSegment,
        left: PsatSegment,
        right: PsatSegment,
    ) -> bool:
        slope_relative_margin = (
            self.policy.smoothing_max_absolute_relative_slope_mismatch
        )
        for temperature in np.linspace(
            bridge.T_min,
            bridge.T_max,
            self.policy.bridge_probe_points,
        ):
            temperature = float(temperature)
            try:
                bridge_value = bridge.ln_pressure(temperature)
                left_value = left.ln_pressure(temperature)
                right_value = right.ln_pressure(temperature)
                bridge_slope = bridge.dln_pressure_dT(temperature)
                left_slope = left.dln_pressure_dT(temperature)
                right_slope = right.dln_pressure_dT(temperature)
            except PsatCanonicalizationError:
                return False
            value_tolerance = 1.0e-10 * max(
                abs(left_value),
                abs(right_value),
                1.0,
            )
            if not (
                min(left_value, right_value) - value_tolerance
                <= bridge_value
                <= max(left_value, right_value) + value_tolerance
            ):
                return False
            slope_scale = max(abs(left_slope), abs(right_slope), 1.0e-15)
            slope_margin = slope_relative_margin * slope_scale
            if not (
                min(left_slope, right_slope) - slope_margin
                <= bridge_slope
                <= max(left_slope, right_slope) + slope_margin
            ):
                return False
        return True

    def _assessment(
        self,
        left: PsatSegmentSlice,
        right: PsatSegmentSlice,
        temperature: float,
    ) -> Optional[PsatJunctionAssessment]:
        try:
            left_ln_pressure = left.segment.ln_pressure(temperature)
            right_ln_pressure = right.segment.ln_pressure(temperature)
            left_slope = left.segment.dln_pressure_dT(temperature)
            right_slope = right.segment.dln_pressure_dT(temperature)
        except PsatCanonicalizationError:
            return None
        junction = PsatJunction(
            temperature=temperature,
            left=PsatSegmentSlice(
                left.segment,
                left.segment.T_min,
                left.segment.T_max,
            ),
            right=PsatSegmentSlice(
                right.segment,
                right.segment.T_min,
                right.segment.T_max,
            ),
            delta_ln_pressure=right_ln_pressure - left_ln_pressure,
            delta_dln_pressure_dT=right_slope - left_slope,
        )
        return junction.assess(self.policy.direct_junction_policy)

    def _metrics(
        self,
        left: PsatSegmentSlice,
        right: PsatSegmentSlice,
        temperature: float,
    ) -> Optional[tuple[float, float, float, float]]:
        try:
            delta_ln_pressure = (
                right.segment.ln_pressure(temperature)
                - left.segment.ln_pressure(temperature)
            )
            left_slope = left.segment.dln_pressure_dT(temperature)
            right_slope = right.segment.dln_pressure_dT(temperature)
        except PsatCanonicalizationError:
            return None
        try:
            pressure_mismatch = abs(math.expm1(delta_ln_pressure))
        except OverflowError:
            pressure_mismatch = math.inf
        slope_mismatch = abs(right_slope - left_slope) / max(
            abs(left_slope),
            abs(right_slope),
            1.0e-15,
        )
        return pressure_mismatch, slope_mismatch, left_slope, right_slope

    def _search_temperatures(
        self,
        overlap: tuple[float, float],
        current_boundary: float,
    ) -> tuple[float, ...]:
        values = list(np.linspace(
            overlap[0],
            overlap[1],
            self.policy.search_sample_count,
        ))
        if overlap[0] <= current_boundary <= overlap[1]:
            values.append(float(current_boundary))
        return tuple(sorted(set(float(value) for value in values)))

    @staticmethod
    def _lower_preference_segment(
        left: PsatSegment,
        right: PsatSegment,
        insertion_order: Mapping[int, int],
    ) -> PsatSegment:
        return left if PsatHandoffCoordinator._preference(
            left,
            insertion_order,
        ) < PsatHandoffCoordinator._preference(
            right,
            insertion_order,
        ) else right

    @staticmethod
    def _preference(
        segment: PsatSegment,
        insertion_order: Mapping[int, int],
    ) -> tuple[float, float, int]:
        return (
            float(segment.priority),
            float(segment.quality),
            -insertion_order.get(id(segment), 0),
        )

    def _unsatisfied_handoff_requirement(
        self,
        active: Sequence[PsatSegment],
        assembly: PsatAssembly,
        insertion_order: Mapping[int, int],
    ) -> Optional[PsatSegment]:
        overlap_cache: dict[tuple[int, int], bool] = {}

        def reasonably_overlaps(first: PsatSegment, second: PsatSegment) -> bool:
            key = tuple(sorted((id(first), id(second))))
            if key not in overlap_cache:
                overlap_cache[key] = self._segments_reasonably_overlap(first, second)
            return overlap_cache[key]

        eligible = set()
        for segment in active:
            requirement = segment.handoff_requirement
            if requirement is PsatHandoffRequirement.NONE:
                eligible.add(id(segment))
                continue
            if (
                requirement
                is PsatHandoffRequirement.HIGHER_PREFERENCE_OVERLAP_IF_AVAILABLE
                and not any(
                    other is not segment
                    and other.priority > segment.priority
                    for other in active
                )
            ):
                eligible.add(id(segment))
        while True:
            newly_eligible = {
                id(segment)
                for segment in active
                if (
                    id(segment) not in eligible
                    and any(
                        id(other) in eligible
                        and other.priority > segment.priority
                        and reasonably_overlaps(segment, other)
                        for other in active
                    )
                )
            }
            if not newly_eligible:
                break
            eligible.update(newly_eligible)
        selected = {
            id(item.segment): item.segment
            for item in assembly.slices
            if not item.segment.metadata.get("handoff_bridge")
        }
        unsatisfied = [
            segment
            for segment in selected.values()
            if (
                segment.handoff_requirement is not PsatHandoffRequirement.NONE
                and id(segment) not in eligible
            )
        ]
        if not unsatisfied:
            return None
        return min(
            unsatisfied,
            key=lambda segment: self._preference(segment, insertion_order),
        )

    def _promote_overlap_validated_quality(
        self,
        assembly: PsatAssembly,
        active: Sequence[PsatSegment],
    ) -> PsatAssembly:
        """Promote selected segments corroborated by a stricter source layer."""
        replacements: dict[int, PsatSegment] = {}
        for item in assembly.slices:
            segment = item.segment
            if id(segment) in replacements:
                continue
            target = segment.metadata.get("higher_preference_overlap_quality")
            if (
                target is None
                or segment.metadata.get("quality_basis")
                != "standalone_unvalidated"
            ):
                continue
            try:
                target_quality = float(target)
            except (TypeError, ValueError):
                continue
            corroborators = [
                other
                for other in active
                if (
                    other is not segment
                    and other.priority > segment.priority
                    and self._segments_reasonably_overlap(segment, other)
                )
            ]
            if not corroborators or target_quality <= segment.quality:
                continue
            corroborator = max(
                corroborators,
                key=lambda item: (item.priority, item.quality),
            )
            metadata = dict(segment.metadata)
            metadata.update({
                "quality_basis": "higher_preference_overlap",
                "overlap_validation_original_quality": segment.quality,
                "overlap_validation_source": corroborator.source,
                "overlap_validation_method": corroborator.method,
                "overlap_validation_priority": corroborator.priority,
            })
            replacements[id(segment)] = replace(
                segment,
                quality=min(1.0, target_quality),
                metadata=metadata,
            )
        if not replacements:
            return assembly
        return PsatAssembly(
            target_T_min=assembly.target_T_min,
            target_T_max=assembly.target_T_max,
            slices=tuple(
                replace(
                    item,
                    segment=replacements.get(id(item.segment), item.segment),
                )
                for item in assembly.slices
            ),
        )

    def _segments_reasonably_overlap(
        self,
        first: PsatSegment,
        second: PsatSegment,
    ) -> bool:
        overlap = (
            max(first.T_min, second.T_min),
            min(first.T_max, second.T_max),
        )
        if overlap[1] <= overlap[0] + 1.0e-9:
            return False
        first_slice = PsatSegmentSlice(first, first.T_min, first.T_max)
        second_slice = PsatSegmentSlice(second, second.T_min, second.T_max)
        if self._direct_handoff_temperature(
            first_slice,
            second_slice,
            overlap,
        ) is not None:
            return True
        return any(
            self._sources_are_consistent_over_bridge(
                first_slice,
                second_slice,
                bridge_T_min,
                bridge_T_max,
            )
            for _width_fraction, bridge_T_min, bridge_T_max
            in self._bridge_intervals(overlap, overlap[1])
        )

    @staticmethod
    def _direct_decision(
        left: PsatSegmentSlice,
        right: PsatSegmentSlice,
        temperature: float,
        reason: str,
    ) -> PsatHandoffDecision:
        overlap = PsatHandoffCoordinator._overlap(left, right)
        return PsatHandoffDecision(
            action=PsatHandoffAction.DIRECT,
            left_method=left.segment.method,
            right_method=right.segment.method,
            overlap_T_min=overlap[0] if overlap is not None else None,
            overlap_T_max=overlap[1] if overlap is not None else None,
            handoff_temperature=temperature,
            reason=reason,
        )

    @staticmethod
    def _materialize(
        assembly: PsatAssembly,
    ) -> tuple[tuple[PsatSegment, ...], PsatAssembly]:
        segments = []
        slices = []
        for item in assembly.slices:
            segment = item.segment
            if item.T_min != segment.T_min or item.T_max != segment.T_max:
                segment = segment.clipped(item.T_min, item.T_max)
            segments.append(segment)
            slices.append(PsatSegmentSlice(segment, item.T_min, item.T_max))
        return tuple(segments), PsatAssembly(
            target_T_min=assembly.target_T_min,
            target_T_max=assembly.target_T_max,
            slices=tuple(slices),
        )


@dataclass(frozen=True)
class PsatSegmentProvenance:
    """Serializable description of a selected source interval."""

    source: str
    method: str
    segment_type: str
    priority: int
    quality: float
    T_min: float
    T_max: float
    allow_junction_slope_mismatch: bool = False
    context: Mapping[str, Any] = field(default_factory=dict, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_slice(cls, item: PsatSegmentSlice) -> "PsatSegmentProvenance":
        segment = item.segment
        return cls(
            source=segment.source,
            method=segment.method,
            segment_type=segment.segment_type.value,
            priority=segment.priority,
            quality=segment.quality,
            T_min=item.T_min,
            T_max=item.T_max,
            allow_junction_slope_mismatch=(
                segment.allow_junction_slope_mismatch
            ),
            context=dict(segment.context),
            metadata=dict(segment.metadata),
        )


@dataclass(frozen=True)
class CanonicalPsatFitDiagnostics:
    """Regression and physical-validation results retained with a curve."""

    sample_count: int
    mard_percent: float
    p95_absolute_relative_error_percent: float
    max_absolute_relative_error_percent: float
    max_anchor_log_residual: float
    monotonic: bool
    reduced_condition_number: Optional[float] = None


@dataclass(frozen=True)
class CanonicalPsatSliceFitDiagnostics:
    """Fit errors evaluated separately over one selected source slice."""

    source: str
    method: str
    priority: int
    quality: float
    T_min: float
    T_max: float
    sample_count: int
    mard_percent: float
    p95_absolute_relative_error_percent: float
    max_absolute_relative_error_percent: float


@dataclass(frozen=True)
class PsatSample:
    """One weighted target value supplied to canonical regression."""

    temperature: float
    ln_pressure: float
    weight: float = 1.0
    source: str = ""
    method: str = ""
    context: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise PsatCanonicalizationError("Psat sample temperature must be positive")
        if not math.isfinite(self.ln_pressure):
            raise PsatCanonicalizationError("Psat sample ln(P) must be finite")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise PsatCanonicalizationError("Psat sample weight must be positive")

    @classmethod
    def from_pressure_bar(
        cls,
        temperature: float,
        pressure_bar: float,
        **kwargs: Any,
    ) -> "PsatSample":
        if not math.isfinite(pressure_bar) or pressure_bar <= 0.0:
            raise PsatCanonicalizationError("Psat sample pressure must be positive")
        return cls(
            temperature=temperature,
            ln_pressure=math.log(pressure_bar),
            **kwargs,
        )


@dataclass(frozen=True)
class CanonicalPsatFitPolicy:
    """Numerical and physical acceptance rules for canonical regression."""

    minimum_sample_count: int = 6
    validation_sample_count: int = 1001
    slice_validation_sample_count: int = 201
    require_monotonic: bool = True
    max_anchor_log_residual: float = 1.0e-9
    max_mard_percent: Optional[float] = None
    max_p95_absolute_relative_error_percent: Optional[float] = None
    max_absolute_relative_error_percent: Optional[float] = None
    mard_excess_quality_penalty_factor: float = 7.5
    max_error_excess_quality_penalty_factor: float = 1.5
    quality_middle_pressure_min_bar: float = 0.10
    quality_middle_pressure_max_bar: float = 5.0
    quality_middle_pressure_weight: float = 3.0
    inverse_retry_powers: tuple[int, ...] = (-3, -5, -7)
    pfd_slice_mard_percent: float = 0.25
    pfd_slice_max_error_percent: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum_sample_count < 6:
            raise PsatCanonicalizationError("Canonical fitting requires at least six samples")
        if self.validation_sample_count < 3:
            raise PsatCanonicalizationError("Fit validation requires at least three points")
        if self.slice_validation_sample_count < 3:
            raise PsatCanonicalizationError(
                "Slice fit validation requires at least three points"
            )
        limits = (
            self.max_anchor_log_residual,
            self.max_mard_percent,
            self.max_p95_absolute_relative_error_percent,
            self.max_absolute_relative_error_percent,
        )
        for value in limits:
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise PsatCanonicalizationError("Canonical fit limits must be nonnegative")
        for value in (
            self.pfd_slice_mard_percent,
            self.pfd_slice_max_error_percent,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise PsatCanonicalizationError(
                    "PFD slice fit limits must be finite and nonnegative"
                )
        if any(
            power not in {-3, -5, -7}
            for power in self.inverse_retry_powers
        ):
            raise PsatCanonicalizationError(
                "Canonical inverse retry powers must be selected from -3, -5, and -7"
            )
        if (
            not math.isfinite(self.mard_excess_quality_penalty_factor)
            or self.mard_excess_quality_penalty_factor < 0.0
            or not math.isfinite(self.max_error_excess_quality_penalty_factor)
            or self.max_error_excess_quality_penalty_factor < 0.0
        ):
            raise PsatCanonicalizationError(
                "Canonical fit quality-penalty factors must be nonnegative"
            )
        if not (
            math.isfinite(self.quality_middle_pressure_min_bar)
            and math.isfinite(self.quality_middle_pressure_max_bar)
            and 0.0 < self.quality_middle_pressure_min_bar
            < self.quality_middle_pressure_max_bar
        ):
            raise PsatCanonicalizationError(
                "Canonical quality middle-pressure bounds must satisfy "
                "0 < minimum < maximum"
            )
        if (
            not math.isfinite(self.quality_middle_pressure_weight)
            or self.quality_middle_pressure_weight < 1.0
        ):
            raise PsatCanonicalizationError(
                "Canonical quality middle-pressure weight must be at least one"
            )

    def rejection_reasons(
        self,
        diagnostics: CanonicalPsatFitDiagnostics,
    ) -> tuple[str, ...]:
        reasons = []
        if diagnostics.sample_count < self.minimum_sample_count:
            reasons.append("insufficient canonical fit samples")
        if self.require_monotonic and not diagnostics.monotonic:
            reasons.append("canonical fit is not strictly monotonic")
        if diagnostics.max_anchor_log_residual > self.max_anchor_log_residual:
            reasons.append(
                f"anchor residual {diagnostics.max_anchor_log_residual:.6g} exceeds "
                f"{self.max_anchor_log_residual:.6g}"
            )
        comparisons = (
            (diagnostics.mard_percent, self.max_mard_percent, "MARD"),
            (
                diagnostics.p95_absolute_relative_error_percent,
                self.max_p95_absolute_relative_error_percent,
                "p95 error",
            ),
            (
                diagnostics.max_absolute_relative_error_percent,
                self.max_absolute_relative_error_percent,
                "maximum error",
            ),
        )
        for measured, limit, label in comparisons:
            if limit is not None and measured > limit:
                reasons.append(f"{label} {measured:.6g}% exceeds {limit:.6g}%")
        return tuple(reasons)

    def tolerance_for(
        self,
        *,
        quality: float,
        provenance: Sequence[PsatSegmentProvenance],
        T_min: float,
        T_critical: float,
    ) -> "CanonicalPsatFitTolerance":
        """Derive fit-error thresholds directly from overall curve quality."""
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise PsatCanonicalizationError(
                "Canonical fit tolerance quality must be between zero and one"
            )
        full_source = _provenance_is_one_shot(
            provenance,
            T_min=T_min,
            T_critical=T_critical,
        )
        return CanonicalPsatFitTolerance(
            profile=(
                "quality_scaled_one_shot"
                if full_source
                else "quality_scaled_assembled"
            ),
            full_source=full_source,
            quality_basis=quality,
            mard_threshold_percent=20.0 * (1.0 - quality),
            max_error_threshold_percent=100.0 * (1.0 - quality),
            mard_penalty_factor=self.mard_excess_quality_penalty_factor,
            max_error_penalty_factor=(
                self.max_error_excess_quality_penalty_factor
            ),
        )

    def pfd_slice_rejection_reasons(
        self,
        diagnostics: Sequence[CanonicalPsatSliceFitDiagnostics],
    ) -> tuple[str, ...]:
        reasons = []
        for item in diagnostics:
            if item.priority < int(PsatPriority.PFD_OVERRIDE):
                continue
            if item.mard_percent > self.pfd_slice_mard_percent:
                reasons.append(
                    f"PFD slice {item.method!r} MARD "
                    f"{item.mard_percent:.6g}% exceeds "
                    f"{self.pfd_slice_mard_percent:.6g}%"
                )
            if (
                item.max_absolute_relative_error_percent
                > self.pfd_slice_max_error_percent
            ):
                reasons.append(
                    f"PFD slice {item.method!r} maximum error "
                    f"{item.max_absolute_relative_error_percent:.6g}% exceeds "
                    f"{self.pfd_slice_max_error_percent:.6g}%"
                )
        return tuple(reasons)


CANONICAL_FIT_ERROR_NUMERICAL_TOLERANCE_PERCENT = 1.0e-10


@dataclass(frozen=True)
class CanonicalPsatFitTolerance:
    """Quality-scaled retry thresholds and excess-error quality penalty."""

    profile: str
    full_source: bool
    quality_basis: float
    mard_threshold_percent: float
    max_error_threshold_percent: float
    mard_penalty_factor: float
    max_error_penalty_factor: float

    @property
    def retry_mard_percent(self) -> float:
        return self.mard_threshold_percent

    @property
    def retry_max_error_percent(self) -> float:
        return self.max_error_threshold_percent

    def accepts_simple_form(
        self,
        diagnostics: CanonicalPsatFitDiagnostics,
        *,
        require_monotonic: bool = True,
    ) -> bool:
        return (
            (diagnostics.monotonic or not require_monotonic)
            and diagnostics.mard_percent
            <= (
                self.mard_threshold_percent
                + CANONICAL_FIT_ERROR_NUMERICAL_TOLERANCE_PERCENT
            )
            and diagnostics.max_absolute_relative_error_percent
            <= (
                self.max_error_threshold_percent
                + CANONICAL_FIT_ERROR_NUMERICAL_TOLERANCE_PERCENT
            )
        )

    def excess_fractions(
        self,
        diagnostics: CanonicalPsatFitDiagnostics,
    ) -> tuple[float, float]:
        return (
            max(
                0.0,
                (
                    diagnostics.mard_percent
                    - self.mard_threshold_percent
                    - CANONICAL_FIT_ERROR_NUMERICAL_TOLERANCE_PERCENT
                )
                / 100.0,
            ),
            max(
                0.0,
                (
                    diagnostics.max_absolute_relative_error_percent
                    - self.max_error_threshold_percent
                    - CANONICAL_FIT_ERROR_NUMERICAL_TOLERANCE_PERCENT
                )
                / 100.0,
            ),
        )

    def quality_penalty(
        self,
        diagnostics: CanonicalPsatFitDiagnostics,
    ) -> float:
        mard_excess, max_error_excess = self.excess_fractions(diagnostics)
        return (
            self.mard_penalty_factor * mard_excess
            + self.max_error_penalty_factor * max_error_excess
        )

    def warning(
        self,
        diagnostics: CanonicalPsatFitDiagnostics,
        *,
        original_quality: float,
    ) -> Optional[str]:
        mard_excess, max_error_excess = self.excess_fractions(diagnostics)
        if mard_excess <= 0.0 and max_error_excess <= 0.0:
            return None
        penalty = self.quality_penalty(diagnostics)
        return (
            "Canonical Psat fit exceeded quality-scaled error thresholds: "
            f"overall quality={original_quality:.6g}, "
            f"MARD={diagnostics.mard_percent:.6g}% "
            f"(limit {self.mard_threshold_percent:.6g}%, "
            f"excess {mard_excess:.6g}), "
            f"maximum error="
            f"{diagnostics.max_absolute_relative_error_percent:.6g}% "
            f"(limit {self.max_error_threshold_percent:.6g}%, "
            f"excess {max_error_excess:.6g}); "
            f"quality penalty={penalty:.6g}"
        )


@dataclass(frozen=True)
class CanonicalPsatCurve:
    """Runtime ``ln(P/bar)`` curve in the selected compact representation."""

    A: float
    B: float
    C: float
    D: float
    E: float
    F: float
    T_min: float
    T_critical: float
    P_critical_bar: float
    G: float = 0.0
    H: float = 0.0
    inverse_power: Optional[int] = None
    form: Optional[CanonicalPsatForm] = None
    T_boiling: Optional[float] = None
    quality: float = 1.0
    provenance: tuple[PsatSegmentProvenance, ...] = ()
    diagnostics: Optional[CanonicalPsatFitDiagnostics] = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    supercritical_slope: float = field(init=False)
    lower_continuation_slope: float = field(init=False)

    def __post_init__(self) -> None:
        form = self.form
        if form is None:
            if self.inverse_power is not None or self.H != 0.0:
                form = CanonicalPsatForm.AH
            elif self.G != 0.0:
                form = CanonicalPsatForm.AG
            else:
                form = CanonicalPsatForm.AF
            object.__setattr__(self, "form", form)
        elif not isinstance(form, CanonicalPsatForm):
            try:
                form = CanonicalPsatForm(form)
            except ValueError as error:
                raise PsatCanonicalizationError(
                    f"Unsupported canonical Psat form {self.form!r}"
                ) from error
            object.__setattr__(self, "form", form)
        coefficients = self.coefficients
        if any(not math.isfinite(value) for value in coefficients):
            raise PsatCanonicalizationError("Canonical Psat coefficients must be finite")
        _validate_temperature_range(self.T_min, self.T_critical)
        if not math.isfinite(self.P_critical_bar) or self.P_critical_bar <= 0.0:
            raise PsatCanonicalizationError("Canonical critical pressure must be positive")
        if self.T_boiling is not None and not self.covers_temperature(self.T_boiling):
            raise PsatCanonicalizationError("Canonical boiling point is outside the curve domain")
        if not math.isfinite(self.quality) or not 0.0 <= self.quality <= 1.0:
            raise PsatCanonicalizationError("Canonical Psat quality must be between 0 and 1")
        if self.inverse_power not in {None, -3, -5, -7}:
            raise PsatCanonicalizationError(
                "Canonical inverse power must be None, -3, -5, or -7"
            )
        if form is CanonicalPsatForm.AF and (
            self.G != 0.0 or self.H != 0.0 or self.inverse_power is not None
        ):
            raise PsatCanonicalizationError(
                "Canonical A-F form cannot contain G, H, or inverse_power"
            )
        if form is CanonicalPsatForm.AG and (
            self.H != 0.0 or self.inverse_power is not None
        ):
            raise PsatCanonicalizationError(
                "Canonical A-G form cannot contain H or inverse_power"
            )
        if form is CanonicalPsatForm.AH and self.inverse_power is None:
            raise PsatCanonicalizationError(
                "Canonical A-H form requires an inverse_power"
            )
        supercritical_slope = (
            self.P_critical_bar * self.dln_pressure_dT(self.T_critical)
        )
        if not math.isfinite(supercritical_slope):
            raise PsatCanonicalizationError(
                "Canonical supercritical pressure slope must be finite"
            )
        object.__setattr__(self, "supercritical_slope", supercritical_slope)
        lower_continuation_slope = self.dln_pressure_dT(self.T_min)
        if not math.isfinite(lower_continuation_slope):
            raise PsatCanonicalizationError(
                "Canonical lower continuation slope must be finite"
            )
        object.__setattr__(
            self,
            "lower_continuation_slope",
            lower_continuation_slope,
        )

    @property
    def coefficients(self) -> tuple[float, float, float, float, float, float, float, float]:
        return self.A, self.B, self.C, self.D, self.E, self.F, self.G, self.H

    def covers_temperature(self, T: float, tolerance: float = 1.0e-9) -> bool:
        return self.T_min - tolerance <= T <= self.T_critical + tolerance

    def ln_pressure(self, T: float) -> float:
        self._require_temperature(T)
        value = (
            self.A
            + self.B / T
            + self.C * math.log(T)
            + self.D * T
            + self.E * T**2
            + self.F * T**5
        )
        if self.form in {CanonicalPsatForm.AG, CanonicalPsatForm.AH}:
            value += self.G * T**3
        if self.form is CanonicalPsatForm.AH:
            value += self.H * ((T / self.T_critical) ** self.inverse_power - 1.0)
        if not math.isfinite(value):
            raise PsatCanonicalizationError("Canonical Psat evaluation is non-finite")
        return value

    def pressure_bar(self, T: float) -> float:
        try:
            value = math.exp(self.ln_pressure(T))
        except OverflowError as error:
            raise PsatCanonicalizationError("Canonical Psat evaluation overflowed") from error
        if not math.isfinite(value) or value <= 0.0:
            raise PsatCanonicalizationError("Canonical Psat evaluation is invalid")
        return value

    def dln_pressure_dT(self, T: float) -> float:
        self._require_temperature(T)
        value = (
            -self.B / T**2
            + self.C / T
            + self.D
            + 2.0 * self.E * T
            + 5.0 * self.F * T**4
        )
        if self.form in {CanonicalPsatForm.AG, CanonicalPsatForm.AH}:
            value += 3.0 * self.G * T**2
        if self.form is CanonicalPsatForm.AH:
            value += (
                self.H
                * self.inverse_power
                / self.T_critical
                * (T / self.T_critical) ** (self.inverse_power - 1)
            )
        if not math.isfinite(value):
            raise PsatCanonicalizationError("Canonical Psat derivative is non-finite")
        return value

    @property
    def critical_anchor_log_residual(self) -> float:
        return self.ln_pressure(self.T_critical) - math.log(self.P_critical_bar)

    @property
    def boiling_anchor_log_residual(self) -> Optional[float]:
        if self.T_boiling is None:
            return None
        return self.ln_pressure(self.T_boiling) - math.log(1.01325)

    def _require_temperature(self, T: float) -> None:
        if not math.isfinite(float(T)) or not self.covers_temperature(float(T)):
            raise PsatCanonicalizationError(
                f"Temperature {T!r} K is outside canonical Psat range "
                f"{self.T_min:g}-{self.T_critical:g} K"
            )


def sample_psat_assembly(
    assembly: PsatAssembly,
    temperatures: Iterable[float],
    weight: float = 1.0,
) -> tuple[PsatSample, ...]:
    """Evaluate an assembled target curve into regression samples."""
    samples = []
    for raw_temperature in temperatures:
        temperature = float(raw_temperature)
        item = assembly.slice_at(temperature)
        if item is None:
            raise PsatCanonicalizationError(
                f"Cannot sample uncovered Psat temperature {temperature:g} K"
            )
        segment = item.segment
        samples.append(PsatSample(
            temperature=temperature,
            ln_pressure=item.ln_pressure(temperature),
            weight=weight,
            source=segment.source,
            method=segment.method,
            context=dict(segment.context),
        ))
    return tuple(samples)


def log_pressure_weighted_psat_quality(
    assembly: PsatAssembly,
    *,
    middle_pressure_min_bar: float = 0.10,
    middle_pressure_max_bar: float = 5.0,
    middle_pressure_weight: float = 3.0,
) -> tuple[float, tuple[Mapping[str, Any], ...]]:
    if not (
        0.0 < middle_pressure_min_bar < middle_pressure_max_bar
        and math.isfinite(middle_pressure_weight)
        and middle_pressure_weight >= 1.0
    ):
        raise PsatCanonicalizationError(
            "Invalid log-pressure quality aggregation policy"
        )
    middle_min = math.log(middle_pressure_min_bar)
    middle_max = math.log(middle_pressure_max_bar)
    weighted_quality = 0.0
    total_weight = 0.0
    breakdown = []
    for item in assembly.slices:
        lower = item.ln_pressure(item.T_min)
        upper = item.ln_pressure(item.T_max)
        pressure_min = min(lower, upper)
        pressure_max = max(lower, upper)
        log_span = max(0.0, pressure_max - pressure_min)
        middle_span = max(
            0.0,
            min(pressure_max, middle_max)
            - max(pressure_min, middle_min),
        )
        weighted_span = (
            log_span
            + (middle_pressure_weight - 1.0) * middle_span
        )
        if weighted_span <= 0.0:
            continue
        weighted_quality += item.quality * weighted_span
        total_weight += weighted_span
        breakdown.append({
            "method": item.segment.method,
            "quality": item.quality,
            "log_pressure_span": log_span,
            "middle_log_pressure_span": middle_span,
            "weighted_log_pressure_span": weighted_span,
        })
    quality = (
        weighted_quality / total_weight
        if total_weight > 0.0
        else min(
            (item.quality for item in assembly.slices),
            default=0.0,
        )
    )
    return quality, tuple(breakdown)


class CanonicalPsatFitter:
    """Fit the constrained canonical representation to a completed Psat target."""

    def __init__(self, policy: Optional[CanonicalPsatFitPolicy] = None):
        self.policy = policy or CanonicalPsatFitPolicy()

    def fit_assembly(
        self,
        assembly: PsatAssembly,
        temperatures: Iterable[float],
        *,
        P_critical_bar: float,
        P_min_bar: Optional[float] = None,
        T_boiling: Optional[float] = None,
        junction_policy: Optional[PsatJunctionPolicy] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CanonicalPsatCurve:
        if not assembly.covers_target():
            raise PsatCanonicalizationError(
                "Canonical fitting requires complete target-domain coverage"
            )
        assembly.require_compatible_junctions(junction_policy)
        samples = sample_psat_assembly(assembly, temperatures)
        provenance = tuple(
            PsatSegmentProvenance.from_slice(item)
            for item in assembly.slices
        )
        slice_validation_targets = []
        for item, item_provenance in zip(assembly.slices, provenance):
            slice_temperatures = np.linspace(
                item.T_min,
                item.T_max,
                self.policy.slice_validation_sample_count,
            )
            slice_validation_targets.append((
                item_provenance,
                slice_temperatures,
                np.asarray(
                    [
                        item.ln_pressure(float(temperature))
                        for temperature in slice_temperatures
                    ],
                    dtype=float,
                ),
            ))
        quality, quality_breakdown = log_pressure_weighted_psat_quality(
            assembly,
            middle_pressure_min_bar=(
                self.policy.quality_middle_pressure_min_bar
            ),
            middle_pressure_max_bar=(
                self.policy.quality_middle_pressure_max_bar
            ),
            middle_pressure_weight=(
                self.policy.quality_middle_pressure_weight
            ),
        )
        fit_metadata = dict(metadata or {})
        fit_metadata["quality_aggregation"] = {
            "basis": "log_pressure_weighted_segment_average",
            "middle_pressure_min_bar": (
                self.policy.quality_middle_pressure_min_bar
            ),
            "middle_pressure_max_bar": (
                self.policy.quality_middle_pressure_max_bar
            ),
            "middle_pressure_weight": (
                self.policy.quality_middle_pressure_weight
            ),
            "segment_breakdown": quality_breakdown,
        }
        return self.fit(
            samples,
            T_min=assembly.target_T_min,
            T_critical=assembly.target_T_max,
            P_critical_bar=P_critical_bar,
            P_min_bar=P_min_bar,
            T_boiling=T_boiling,
            quality=quality,
            provenance=provenance,
            slice_validation_targets=tuple(slice_validation_targets),
            metadata=fit_metadata,
        )

    def fit(
        self,
        samples: Sequence[PsatSample],
        *,
        T_min: float,
        T_critical: float,
        P_critical_bar: float,
        P_min_bar: Optional[float] = None,
        T_boiling: Optional[float] = None,
        quality: float = 1.0,
        provenance: Sequence[PsatSegmentProvenance] = (),
        slice_validation_targets: Sequence[
            tuple[PsatSegmentProvenance, np.ndarray, np.ndarray]
        ] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CanonicalPsatCurve:
        _validate_temperature_range(T_min, T_critical)
        if not math.isfinite(P_critical_bar) or P_critical_bar <= 0.0:
            raise PsatCanonicalizationError("Critical pressure must be positive")
        if P_min_bar is not None and (
            not math.isfinite(P_min_bar)
            or P_min_bar <= 0.0
            or P_min_bar >= P_critical_bar
        ):
            raise PsatCanonicalizationError(
                "Canonical minimum pressure must satisfy 0 < P_min < Pc"
            )
        if len(samples) < self.policy.minimum_sample_count:
            raise PsatCanonicalizationError("Insufficient canonical fit samples")
        if T_boiling is not None and not T_min <= T_boiling < T_critical:
            raise PsatCanonicalizationError("Boiling point must lie below Tc in the fit domain")
        for sample in samples:
            if not T_min <= sample.temperature <= T_critical:
                raise PsatCanonicalizationError(
                    "Canonical fit samples must lie inside the target domain"
                )
        distinct_temperatures = {round(sample.temperature, 12) for sample in samples}
        if len(distinct_temperatures) < self.policy.minimum_sample_count:
            raise PsatCanonicalizationError(
                "Canonical fitting requires distinct sample temperatures"
            )

        temperatures = np.asarray([sample.temperature for sample in samples], dtype=float)
        targets = np.asarray([sample.ln_pressure for sample in samples], dtype=float)
        square_root_weights = np.sqrt(
            np.asarray([sample.weight for sample in samples], dtype=float)
        )
        weighted_targets = targets * square_root_weights

        anchor_temperatures = []
        anchor_targets = []
        if P_min_bar is not None:
            anchor_temperatures.append(T_min)
            anchor_targets.append(math.log(P_min_bar))
        if T_boiling is not None:
            anchor_temperatures.append(T_boiling)
            anchor_targets.append(math.log(1.01325))
        anchor_temperatures.append(T_critical)
        anchor_targets.append(math.log(P_critical_bar))
        anchor_temperatures_array = np.asarray(anchor_temperatures, dtype=float)
        constraint_targets = np.asarray(anchor_targets, dtype=float)
        validation_temperatures = np.linspace(
            T_min,
            T_critical,
            self.policy.validation_sample_count,
        )
        tolerance = self.policy.tolerance_for(
            quality=quality,
            provenance=provenance,
            T_min=T_min,
            T_critical=T_critical,
        )

        def fit_candidate(
            form: CanonicalPsatForm,
            inverse_power: Optional[int] = None,
        ):
            matrix = _canonical_design_matrix(
                temperatures,
                T_critical,
                form,
                inverse_power,
            )
            weighted_matrix = matrix * square_root_weights[:, None]
            constraints = _canonical_design_matrix(
                anchor_temperatures_array,
                T_critical,
                form,
                inverse_power,
            )
            scales = np.linalg.norm(weighted_matrix, axis=0)
            if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
                raise PsatCanonicalizationError("Canonical fit matrix has invalid scaling")
            scaled_matrix = weighted_matrix / scales
            scaled_constraints = constraints / scales
            gram = scaled_constraints @ scaled_constraints.T
            try:
                particular = scaled_constraints.T @ np.linalg.solve(
                    gram,
                    constraint_targets,
                )
            except np.linalg.LinAlgError as error:
                raise PsatCanonicalizationError(
                    "Canonical fit constraints are linearly dependent"
                ) from error
            basis = null_space(scaled_constraints)
            reduced_matrix = scaled_matrix @ basis
            reduced_targets = weighted_targets - scaled_matrix @ particular
            reduced_rank = np.linalg.matrix_rank(reduced_matrix)
            if reduced_rank < reduced_matrix.shape[1]:
                raise PsatCanonicalizationError(
                    "Canonical fit sample matrix is rank deficient"
                )
            reduced_coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
                reduced_matrix,
                reduced_targets,
                rcond=1.0e-13,
            )
            scaled_coefficients = particular + basis @ reduced_coefficients
            coefficients = scaled_coefficients / scales
            if np.any(~np.isfinite(coefficients)):
                raise PsatCanonicalizationError(
                    "Canonical fit produced invalid coefficients"
                )

            predicted = matrix @ coefficients
            with np.errstate(over="ignore", invalid="ignore"):
                absolute_relative_errors = np.abs(np.expm1(predicted - targets))
            if np.any(~np.isfinite(absolute_relative_errors)):
                raise PsatCanonicalizationError(
                    "Canonical fit produced non-finite errors"
                )
            constraint_residuals = constraints @ coefficients - constraint_targets
            validation_derivatives = _canonical_derivative(
                coefficients,
                validation_temperatures,
                T_critical,
                form,
                inverse_power,
            )
            diagnostics = CanonicalPsatFitDiagnostics(
                sample_count=len(samples),
                mard_percent=100.0 * float(np.mean(absolute_relative_errors)),
                p95_absolute_relative_error_percent=(
                    100.0 * float(np.percentile(absolute_relative_errors, 95.0))
                ),
                max_absolute_relative_error_percent=(
                    100.0 * float(np.max(absolute_relative_errors))
                ),
                max_anchor_log_residual=float(np.max(np.abs(constraint_residuals))),
                monotonic=bool(
                    np.all(np.isfinite(validation_derivatives))
                    and np.min(validation_derivatives) > 0.0
                ),
                reduced_condition_number=float(np.linalg.cond(reduced_matrix)),
            )
            slice_diagnostics = []
            for (
                item_provenance,
                item_temperatures,
                item_targets,
            ) in slice_validation_targets:
                item_predicted = _canonical_design_matrix(
                    item_temperatures,
                    T_critical,
                    form,
                    inverse_power,
                ) @ coefficients
                with np.errstate(over="ignore", invalid="ignore"):
                    item_errors = np.abs(np.expm1(
                        item_predicted - item_targets
                    ))
                if np.any(~np.isfinite(item_errors)):
                    raise PsatCanonicalizationError(
                        "Canonical fit produced non-finite slice errors"
                    )
                slice_diagnostics.append(CanonicalPsatSliceFitDiagnostics(
                    source=item_provenance.source,
                    method=item_provenance.method,
                    priority=item_provenance.priority,
                    quality=item_provenance.quality,
                    T_min=item_provenance.T_min,
                    T_max=item_provenance.T_max,
                    sample_count=len(item_temperatures),
                    mard_percent=(
                        100.0 * float(np.mean(item_errors))
                    ),
                    p95_absolute_relative_error_percent=(
                        100.0 * float(np.percentile(item_errors, 95.0))
                    ),
                    max_absolute_relative_error_percent=(
                        100.0 * float(np.max(item_errors))
                    ),
                ))
            return (
                coefficients,
                diagnostics,
                form,
                inverse_power,
                tuple(slice_diagnostics),
            )

        def simple_form_is_acceptable(candidate) -> bool:
            diagnostics = candidate[1]
            return (
                tolerance.accepts_simple_form(
                    diagnostics,
                    require_monotonic=self.policy.require_monotonic,
                )
                and not self.policy.rejection_reasons(diagnostics)
                and not self.policy.pfd_slice_rejection_reasons(candidate[4])
            )

        candidates = []
        attempted_forms = []
        af_candidate = fit_candidate(CanonicalPsatForm.AF)
        candidates.append(af_candidate)
        attempted_forms.append(CanonicalPsatForm.AF.value)
        selected = af_candidate if simple_form_is_acceptable(af_candidate) else None

        if selected is None:
            attempted_forms.append(CanonicalPsatForm.AG.value)
            try:
                ag_candidate = fit_candidate(CanonicalPsatForm.AG)
            except PsatCanonicalizationError:
                ag_candidate = None
            if ag_candidate is not None:
                candidates.append(ag_candidate)
                if simple_form_is_acceptable(ag_candidate):
                    selected = ag_candidate

        attempted_inverse_powers = []
        if selected is None and self.policy.inverse_retry_powers:
            attempted_forms.append(CanonicalPsatForm.AH.value)
            ah_candidates = []
            for inverse_power in self.policy.inverse_retry_powers:
                attempted_inverse_powers.append(inverse_power)
                try:
                    candidate = fit_candidate(
                        CanonicalPsatForm.AH,
                        inverse_power,
                    )
                except PsatCanonicalizationError:
                    continue
                candidates.append(candidate)
                ah_candidates.append(candidate)
            if ah_candidates:
                acceptable_ah_candidates = [
                    candidate
                    for candidate in ah_candidates
                    if simple_form_is_acceptable(candidate)
                ]
                if acceptable_ah_candidates:
                    selected = _best_fit_candidate(
                        acceptable_ah_candidates,
                        require_monotonic=self.policy.require_monotonic,
                    )

        if selected is None:
            selected = _best_fit_candidate(
                candidates,
                require_monotonic=self.policy.require_monotonic,
            )
        (
            coefficients,
            diagnostics,
            form,
            inverse_power,
            slice_diagnostics,
        ) = selected
        rejection_reasons = (
            *self.policy.rejection_reasons(diagnostics),
            *self.policy.pfd_slice_rejection_reasons(slice_diagnostics),
        )
        if rejection_reasons:
            raise PsatCanonicalizationError(
                "Canonical fit rejected: " + "; ".join(rejection_reasons)
            )
        quality_penalty = tolerance.quality_penalty(diagnostics)
        penalized_quality = max(0.0, quality - quality_penalty)
        fit_warning = tolerance.warning(
            diagnostics,
            original_quality=quality,
        )
        fit_metadata = dict(metadata or {})
        fit_metadata.setdefault("fit_method", "scaled_constrained_null_space")
        fit_metadata.setdefault("pressure_units", "bar")
        fit_metadata["canonical_form"] = form.value
        fit_metadata["canonical_terms"] = _canonical_form_terms(form)
        fit_metadata["fit_tolerance_profile"] = tolerance.profile
        fit_metadata["fit_tolerance_quality"] = quality
        fit_metadata["fit_original_overall_quality"] = quality
        fit_metadata["fit_quality_penalty"] = quality_penalty
        fit_metadata["fit_penalized_overall_quality"] = penalized_quality
        fit_metadata["fit_mard_threshold_percent"] = (
            tolerance.mard_threshold_percent
        )
        fit_metadata["fit_max_error_threshold_percent"] = (
            tolerance.max_error_threshold_percent
        )
        mard_excess, max_error_excess = tolerance.excess_fractions(
            diagnostics
        )
        fit_metadata["fit_mard_excess_fraction"] = mard_excess
        fit_metadata["fit_max_error_excess_fraction"] = max_error_excess
        fit_warnings = tuple(
            str(item)
            for item in fit_metadata.get("fit_warnings", ())
        )
        if fit_warning is not None:
            fit_warnings += (fit_warning,)
        if fit_warnings:
            fit_metadata["fit_warnings"] = fit_warnings
        canonical_warnings = []
        for warning_key in (
            "input_warnings",
            "junction_warnings",
            "fit_warnings",
        ):
            canonical_warnings.extend(
                str(item)
                for item in fit_metadata.get(warning_key, ())
            )
        if canonical_warnings:
            fit_metadata["canonical_warnings"] = tuple(canonical_warnings)
        fit_metadata["full_source_coverage"] = tolerance.full_source
        fit_metadata["attempted_forms"] = tuple(attempted_forms)
        fit_metadata["attempted_inverse_powers"] = tuple(attempted_inverse_powers)
        fit_metadata["selected_inverse_power"] = inverse_power
        fit_metadata["slice_fit_diagnostics"] = tuple({
            "source": item.source,
            "method": item.method,
            "priority": item.priority,
            "quality": item.quality,
            "T_min": item.T_min,
            "T_max": item.T_max,
            "sample_count": item.sample_count,
            "mard_percent": item.mard_percent,
            "p95_absolute_relative_error_percent": (
                item.p95_absolute_relative_error_percent
            ),
            "max_absolute_relative_error_percent": (
                item.max_absolute_relative_error_percent
            ),
        } for item in slice_diagnostics)
        G = (
            float(coefficients[6])
            if form in {CanonicalPsatForm.AG, CanonicalPsatForm.AH}
            else 0.0
        )
        H = float(coefficients[7]) if form is CanonicalPsatForm.AH else 0.0
        return CanonicalPsatCurve(
            A=float(coefficients[0]),
            B=float(coefficients[1]),
            C=float(coefficients[2]),
            D=float(coefficients[3]),
            E=float(coefficients[4]),
            F=float(coefficients[5]),
            T_min=T_min,
            T_critical=T_critical,
            P_critical_bar=P_critical_bar,
            G=G,
            H=H,
            inverse_power=inverse_power,
            form=form,
            T_boiling=T_boiling,
            quality=penalized_quality,
            provenance=tuple(provenance),
            diagnostics=diagnostics,
            metadata=fit_metadata,
        )


@dataclass(frozen=True)
class PsatCanonicalizationResult:
    """Final curve together with the assembled and completion-stage results."""

    curve: CanonicalPsatCurve
    assembly: Optional[PsatAssembly]
    completion: Optional[PsatCompletionResult]


class PsatCanonicalizer:
    """Complete and fit segments while always exposing fixed Tb/Tc anchors."""

    def __init__(
        self,
        *,
        T_min: float,
        T_critical: float,
        P_critical_bar: float,
        critical_quality: float,
        T_boiling: Optional[float] = None,
        boiling_quality: Optional[float] = None,
        fit_policy: Optional[CanonicalPsatFitPolicy] = None,
        junction_policy: Optional[PsatJunctionPolicy] = None,
        handoff_policy: Optional[PsatHandoffPolicy] = None,
        fit_sample_count: int = 401,
        segment_probe_points: int = 201,
        context: Optional[Mapping[str, Any]] = None,
    ):
        _validate_temperature_range(T_min, T_critical)
        if not math.isfinite(P_critical_bar) or P_critical_bar <= 0.0:
            raise PsatCanonicalizationError("Canonicalizer critical pressure must be positive")
        if T_boiling is not None and not T_min <= T_boiling < T_critical:
            raise PsatCanonicalizationError("Canonicalizer Tb must lie below Tc")
        if fit_sample_count < 6:
            raise PsatCanonicalizationError("Canonicalizer requires at least six fit points")
        self.T_min = float(T_min)
        self.T_critical = float(T_critical)
        self.P_critical_bar = float(P_critical_bar)
        self.T_boiling = float(T_boiling) if T_boiling is not None else None
        self.junction_policy = junction_policy or PsatJunctionPolicy()
        self.handoff_policy = handoff_policy or PsatHandoffPolicy(
            direct_junction_policy=self.junction_policy,
        )
        self.fit_sample_count = fit_sample_count
        self.segment_probe_points = segment_probe_points
        self.context = dict(context or {})
        self.fitter = CanonicalPsatFitter(fit_policy)
        self.anchors = PsatAnchorRegistry.from_tb_tc(
            T_critical=self.T_critical,
            P_critical_bar=self.P_critical_bar,
            critical_quality=critical_quality,
            T_boiling=self.T_boiling,
            boiling_quality=boiling_quality,
            context=self.context,
        )

    def canonicalize(
        self,
        segments: Iterable[PsatSegment],
        relations: Iterable[BoundaryConditionedPsatSegment] = (),
        *,
        temperatures: Optional[Iterable[float]] = None,
        additional_anchors: Iterable[PsatEndpoint] = (),
        canonical_override: Optional[CanonicalPsatCurve] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PsatCanonicalizationResult:
        if canonical_override is not None:
            return PsatCanonicalizationResult(canonical_override, None, None)
        anchors = PsatAnchorRegistry(self.anchors.points)
        for point in additional_anchors:
            anchors.add(point)
        assembler = PsatSegmentAssembler(self.T_min, self.T_critical)
        assembler.extend(segments)
        completion = PsatCompletionCoordinator(
            assembler,
            relations,
            anchors,
            segment_probe_points=self.segment_probe_points,
            handoff_policy=self.handoff_policy,
        ).complete(junction_policy=self.junction_policy)
        if temperatures is None:
            temperatures = np.linspace(
                self.T_min,
                self.T_critical,
                self.fit_sample_count,
            )
        curve = self.fitter.fit_assembly(
            completion.assembly,
            temperatures,
            P_critical_bar=self.P_critical_bar,
            T_boiling=self.T_boiling,
            junction_policy=self.junction_policy,
            metadata={
                **dict(metadata or {}),
                **(
                    {
                        "junction_warnings": completion.junction_warnings,
                    }
                    if completion.junction_warnings
                    else {}
                ),
            },
        )
        return PsatCanonicalizationResult(
            curve=curve,
            assembly=completion.assembly,
            completion=completion,
        )


def _canonical_design_matrix(
    temperatures: np.ndarray,
    T_critical: float,
    form: CanonicalPsatForm,
    inverse_power: Optional[int] = None,
) -> np.ndarray:
    columns = [
        np.ones_like(temperatures),
        1.0 / temperatures,
        np.log(temperatures),
        temperatures,
        temperatures**2,
        temperatures**5,
    ]
    if form in {CanonicalPsatForm.AG, CanonicalPsatForm.AH}:
        columns.append(temperatures**3)
    if form is CanonicalPsatForm.AH:
        columns.append((temperatures / T_critical) ** inverse_power - 1.0)
    return np.column_stack(columns)


def _canonical_derivative(
    coefficients: np.ndarray,
    temperatures: np.ndarray,
    T_critical: float,
    form: CanonicalPsatForm,
    inverse_power: Optional[int] = None,
) -> np.ndarray:
    _A, B, C, D, E, F, *rest = coefficients
    derivative = (
        -B / temperatures**2
        + C / temperatures
        + D
        + 2.0 * E * temperatures
        + 5.0 * F * temperatures**4
    )
    if form in {CanonicalPsatForm.AG, CanonicalPsatForm.AH}:
        derivative += 3.0 * rest[0] * temperatures**2
    if form is CanonicalPsatForm.AH:
        derivative += (
            rest[1]
            * inverse_power
            / T_critical
            * (temperatures / T_critical) ** (inverse_power - 1)
        )
    return derivative


def _canonical_form_terms(form: CanonicalPsatForm) -> str:
    terms = "A+B/T+C*ln(T)+D*T+E*T^2+F*T^5"
    if form in {CanonicalPsatForm.AG, CanonicalPsatForm.AH}:
        terms += "+G*T^3"
    if form is CanonicalPsatForm.AH:
        terms += "+H*((T/Tc)^inverse_power-1)"
    return terms


def _best_fit_candidate(candidates, *, require_monotonic: bool):
    physically_valid = [
        candidate
        for candidate in candidates
        if not require_monotonic or candidate[1].monotonic
    ] or list(candidates)
    if not physically_valid:
        raise PsatCanonicalizationError("No canonical fit candidate could be constructed")
    return min(
        physically_valid,
        key=lambda candidate: (
            candidate[1].max_absolute_relative_error_percent,
            candidate[1].p95_absolute_relative_error_percent,
            candidate[1].mard_percent,
            candidate[1].reduced_condition_number or math.inf,
        ),
    )


def _provenance_is_one_shot(
    provenance: Sequence[PsatSegmentProvenance],
    *,
    T_min: float,
    T_critical: float,
) -> bool:
    if not provenance:
        return True
    if len(provenance) != 1:
        return False
    item = provenance[0]
    tolerance = 1.0e-9 * max(abs(T_min), abs(T_critical), 1.0)
    return (
        item.segment_type
        in {
            PsatSegmentType.CANONICAL_OVERRIDE.value,
            PsatSegmentType.PINNED.value,
        }
        and item.T_min <= T_min + tolerance
        and item.T_max >= T_critical - tolerance
    )


def _validate_temperature_range(T_min: float, T_max: float) -> None:
    if not math.isfinite(float(T_min)) or not math.isfinite(float(T_max)):
        raise PsatCanonicalizationError("Psat temperature limits must be finite")
    if T_min <= 0.0 or T_max <= T_min:
        raise PsatCanonicalizationError(
            "A Psat temperature range must satisfy 0 < T_min < T_max"
        )


def _validate_pressure_range(
    P_min_bar: Optional[float],
    P_max_bar: Optional[float],
) -> None:
    for value in (P_min_bar, P_max_bar):
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise PsatCanonicalizationError("Psat pressure limits must be positive")
    if (
        P_min_bar is not None
        and P_max_bar is not None
        and P_max_bar < P_min_bar
    ):
        raise PsatCanonicalizationError(
            "A Psat pressure range must satisfy P_min <= P_max"
        )


__all__ = [
    "BoundaryConditionedPsatSegment",
    "BoundaryEvaluationFactory",
    "CanonicalPsatForm",
    "CanonicalPsatCurve",
    "CanonicalPsatFitPolicy",
    "CanonicalPsatFitTolerance",
    "CanonicalPsatFitDiagnostics",
    "CanonicalPsatFitter",
    "LnPressureDerivative",
    "LnPressureFunction",
    "PsatAssembly",
    "PsatAnchorRegistry",
    "PsatAnchorRequirement",
    "PsatBoundaryConditions",
    "PsatBoundaryRequirement",
    "PsatCanonicalizationError",
    "PsatCanonicalizationResult",
    "PsatCanonicalizer",
    "PsatCompletionCoordinator",
    "PsatCompletionResult",
    "PsatDerivativeRequirement",
    "PsatEndpoint",
    "PsatGap",
    "PsatGapAssessment",
    "PsatGapConsistency",
    "PsatHandoffAction",
    "PsatHandoffCoordinator",
    "PsatHandoffDecision",
    "PsatHandoffPolicy",
    "PsatHandoffRequirement",
    "PsatHandoffResult",
    "PsatJunction",
    "PsatJunctionAssessment",
    "PsatJunctionPolicy",
    "PsatPriority",
    "PsatSample",
    "PsatSegment",
    "PsatSegmentAssembler",
    "PsatSegmentProbe",
    "PsatSegmentProvenance",
    "PsatSegmentSlice",
    "PsatSegmentType",
    "make_c1_bridge_segment",
    "assess_psat_gap_consistency",
    "probe_psat_segment",
    "sample_psat_assembly",
    "trim_psat_segment_to_validity",
]
