# ruff: noqa: F401
# Legacy mixin modules intentionally inherit shared imports from this module.
"""
Property Resolution System

Implements robust fallback chains for thermodynamic property resolution.
Ensures accurate simulation results by providing multiple methods to obtain
critical properties for vapor pressure, heat capacity, and other parameters.

Property Resolution Order:
1. Local database (chemicals.json)
2. Online databases (PubChem, NIST)
3. Estimation methods (correlations, group contribution)
4. Error with missing property indication

Author: PFD Editor
Version: 1.0
"""

import json

import hashlib

import math

import os

import re

import urllib.request

import urllib.parse

import urllib.error

import html as html_module

from typing import Optional, Dict, Any, Tuple, List, Callable

from dataclasses import dataclass, field

from enum import Enum

from pathlib import Path

from statistics import median

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..physical_constants import R_J_MOL_K
else:
    from physical_constants import R_J_MOL_K

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..pressure_standards import NORMAL_BOILING_PRESSURE_BAR, THERMOCHEMICAL_STANDARD_PRESSURE_BAR
else:
    from pressure_standards import NORMAL_BOILING_PRESSURE_BAR, THERMOCHEMICAL_STANDARD_PRESSURE_BAR

R = R_J_MOL_K  # J/mol-K (gas constant)

ONLINE_ANTOINE_TB_REL_TOL = 0.02

ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K = 1.0


class OnlineAttemptState(str, Enum):
    """Completeness of an online-enabled selected-property resolution."""

    NOT_NEEDED = "not_needed"
    NOT_ATTEMPTED = "not_attempted"
    COMPLETE_WITH_DATA = "complete_with_data"
    COMPLETE_NO_DATA = "complete_no_data"
    TRANSIENT_FAILURE = "transient_failure"


@dataclass
class OnlineAttemptTracker:
    """Aggregate provider outcomes within one selected-result cache build."""

    allow_online: bool
    attempted: bool = False
    received_data: bool = False
    completed_without_data: bool = False
    transient_failure: bool = False
    explicitly_not_attempted: bool = False
    explicitly_not_needed: bool = False

    def record(self, state: OnlineAttemptState | str) -> None:
        state = OnlineAttemptState(state)
        if state is OnlineAttemptState.TRANSIENT_FAILURE:
            self.attempted = True
            self.transient_failure = True
        elif state is OnlineAttemptState.COMPLETE_WITH_DATA:
            self.attempted = True
            self.received_data = True
        elif state is OnlineAttemptState.COMPLETE_NO_DATA:
            self.attempted = True
            self.completed_without_data = True
        elif state is OnlineAttemptState.NOT_ATTEMPTED:
            self.explicitly_not_attempted = True
        elif state is OnlineAttemptState.NOT_NEEDED:
            self.explicitly_not_needed = True

    @property
    def state(self) -> OnlineAttemptState:
        if self.transient_failure:
            return OnlineAttemptState.TRANSIENT_FAILURE
        if self.received_data:
            return OnlineAttemptState.COMPLETE_WITH_DATA
        if self.attempted and self.completed_without_data:
            return OnlineAttemptState.COMPLETE_NO_DATA
        if self.explicitly_not_attempted:
            return OnlineAttemptState.NOT_ATTEMPTED
        if self.explicitly_not_needed:
            return OnlineAttemptState.NOT_NEEDED
        if not self.allow_online:
            return OnlineAttemptState.NOT_ATTEMPTED
        return OnlineAttemptState.NOT_NEEDED

CP_EXTRAPOLATION_LIMIT_K = 20.0

LIQUID_VOLUME_EXTRAPOLATION_LIMIT_K = 20.0

MIN_EOS_PRESSURE_BAR = 1.0e-5

REFERENCE_TEMPERATURE_K = 298.15

WATER_HVAP_298_KJ_PER_MOL = 43.99

SOFT_PROPERTY_QUALITY_THRESHOLD = 0.90

NET_COMBUSTION_PRODUCT_HF_KJ_PER_MOL = {
    'CO2': -393.51,
    'H2O': -241.826,
    'SO2': -296.84,
    'SiO2': -909.4,
}

@dataclass
class PropertyResolutionResult:
    """Result from property resolution attempt"""
    value: Any
    source: str
    method: str
    quality: float = 1.0  # 0-1, lower for weaker/estimated values
    notes: str = ""


@dataclass(frozen=True)
class DensityObservation:
    """One normalized, phase-classified intrinsic mass-density report.

    Density reports are preserved independently from the liquid and solid
    selection routes.  ``phase`` is assigned from explicit source wording or
    from trustworthy phase points at the reported temperature; ambiguous
    reports remain available without being admitted to either phase model.
    """

    mass_density_kg_m3: float
    temperature_K: Optional[float] = None
    phase: str = "ambiguous"
    phase_basis: str = "unclassified"
    material_form: str = "unspecified"
    form_label: str = ""
    polymorph: str = ""
    stereochemistry: str = ""
    is_identity_form: bool = False
    source: str = "unknown"
    method: str = "unknown"
    quality: float = 1.0
    reference: str = ""
    comment: str = ""
    raw: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        density = float(self.mass_density_kg_m3)
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError("Mass density must be positive and finite")
        object.__setattr__(self, "mass_density_kg_m3", density)
        if self.temperature_K is not None:
            temperature = float(self.temperature_K)
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError("Density temperature must be positive and finite")
            object.__setattr__(self, "temperature_K", temperature)
        phase = str(self.phase or "ambiguous").strip().lower()
        if phase not in {"solid", "liquid", "ambiguous"}:
            raise ValueError(f"Unsupported density phase {phase!r}")
        object.__setattr__(self, "phase", phase)
        material_form = str(self.material_form or "unspecified").strip().lower()
        if material_form not in {"anhydrous", "hydrate", "solvate", "unspecified"}:
            raise ValueError(f"Unsupported density material form {material_form!r}")
        object.__setattr__(self, "material_form", material_form)
        quality = float(self.quality)
        if not math.isfinite(quality):
            raise ValueError("Density quality must be finite")
        object.__setattr__(self, "quality", min(max(quality, 0.0), 1.0))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mass_density_kg_m3": self.mass_density_kg_m3,
            "temperature_K": self.temperature_K,
            "phase": self.phase,
            "phase_basis": self.phase_basis,
            "material_form": self.material_form,
            "form_label": self.form_label,
            "polymorph": self.polymorph,
            "stereochemistry": self.stereochemistry,
            "is_identity_form": self.is_identity_form,
            "source": self.source,
            "method": self.method,
            "quality": self.quality,
            "reference": self.reference,
            "comment": self.comment,
            "raw": self.raw,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FusionTransitionRecord:
    """One measured or provided solid-to-liquid transition.

    ``material_form`` is relative to the requested chemical identity:
    ``anhydrous``, ``hydrate``, ``solvate``, or ``unspecified``.  Polymorph
    and stereochemical labels are preserved independently so alternate solid
    forms are never flattened into a context-free Hfus scalar.
    """

    enthalpy_kJ_mol: float
    temperature_K: Optional[float] = None
    material_form: str = "unspecified"
    form_label: str = ""
    polymorph: str = ""
    stereochemistry: str = ""
    is_identity_form: bool = False
    is_source_default: bool = False
    source: str = "unknown"
    method: str = "unknown"
    quality: float = 1.0
    reference: str = ""
    comment: str = ""
    raw: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        enthalpy = float(self.enthalpy_kJ_mol)
        if not math.isfinite(enthalpy) or enthalpy <= 0.0:
            raise ValueError("Fusion enthalpy must be positive and finite")
        object.__setattr__(self, "enthalpy_kJ_mol", enthalpy)
        if self.temperature_K is not None:
            temperature = float(self.temperature_K)
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError("Fusion-transition temperature must be positive and finite")
            object.__setattr__(self, "temperature_K", temperature)
        material_form = str(self.material_form or "unspecified").strip().lower()
        if material_form not in {"anhydrous", "hydrate", "solvate", "unspecified"}:
            raise ValueError(f"Unsupported fusion material form {material_form!r}")
        object.__setattr__(self, "material_form", material_form)
        quality = float(self.quality)
        if not math.isfinite(quality):
            raise ValueError("Fusion-transition quality must be finite")
        object.__setattr__(self, "quality", min(max(quality, 0.0), 1.0))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enthalpy_kJ_mol": self.enthalpy_kJ_mol,
            "temperature_K": self.temperature_K,
            "material_form": self.material_form,
            "form_label": self.form_label,
            "polymorph": self.polymorph,
            "stereochemistry": self.stereochemistry,
            "is_identity_form": self.is_identity_form,
            "is_source_default": self.is_source_default,
            "source": self.source,
            "method": self.method,
            "quality": self.quality,
            "reference": self.reference,
            "comment": self.comment,
            "raw": self.raw,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MeltingTransitionRecord:
    """One form-specific solid melting/freezing observation."""

    temperature_K: float
    enthalpy_kJ_mol: Optional[float] = None
    material_form: str = "unspecified"
    form_label: str = ""
    polymorph: str = ""
    stereochemistry: str = ""
    is_identity_form: bool = False
    is_source_default: bool = False
    source: str = "unknown"
    method: str = "unknown"
    quality: float = 1.0
    reference: str = ""
    comment: str = ""
    raw: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        temperature = float(self.temperature_K)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("Melting-transition temperature must be positive and finite")
        object.__setattr__(self, "temperature_K", temperature)
        if self.enthalpy_kJ_mol is not None:
            enthalpy = float(self.enthalpy_kJ_mol)
            if not math.isfinite(enthalpy) or enthalpy <= 0.0:
                raise ValueError("Fusion enthalpy must be positive and finite")
            object.__setattr__(self, "enthalpy_kJ_mol", enthalpy)
        material_form = str(self.material_form or "unspecified").strip().lower()
        if material_form not in {"anhydrous", "hydrate", "solvate", "unspecified"}:
            raise ValueError(f"Unsupported melting material form {material_form!r}")
        object.__setattr__(self, "material_form", material_form)
        quality = float(self.quality)
        if not math.isfinite(quality):
            raise ValueError("Melting-transition quality must be finite")
        object.__setattr__(self, "quality", min(max(quality, 0.0), 1.0))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature_K": self.temperature_K,
            "enthalpy_kJ_mol": self.enthalpy_kJ_mol,
            "material_form": self.material_form,
            "form_label": self.form_label,
            "polymorph": self.polymorph,
            "stereochemistry": self.stereochemistry,
            "is_identity_form": self.is_identity_form,
            "is_source_default": self.is_source_default,
            "source": self.source,
            "method": self.method,
            "quality": self.quality,
            "reference": self.reference,
            "comment": self.comment,
            "raw": self.raw,
            "metadata": dict(self.metadata),
        }


def classify_fusion_material_form(
    text: str,
    identity_text: str = "",
) -> tuple[str, str, str, str, bool]:
    """Classify hydrate/solvate, polymorph, and stereochemical form labels."""
    normalized = str(text or "").strip()
    lower = normalized.lower().replace("β", "beta").replace("α", "alpha")
    lower = re.sub(r"\bhydrated\s+from\b", "", lower)
    identity_lower = str(identity_text or "").lower()
    hydrate = re.search(
        r"\b(?:(?:mono|di|tri|tetra|penta|hemi)\s*-?)?hydrate(?:d)?\b",
        lower,
    )
    solvate = re.search(r"\b(?:solvate|solvated)\b", lower)
    if hydrate:
        material_form = "hydrate"
    elif solvate:
        material_form = "solvate"
    elif re.search(r"\banhydrous\b", lower):
        material_form = "anhydrous"
    else:
        material_form = "unspecified"
    identity_form = (
        material_form == "hydrate" and "hydrate" in identity_lower
    ) or (
        material_form == "solvate" and "solvate" in identity_lower
    )

    polymorph = ""
    polymorph_match = re.search(
        r"(?:\bform\s+|\bpolymorph\s+)?\b(alpha|beta|gamma|delta)\b",
        lower,
    )
    if polymorph_match:
        polymorph = polymorph_match.group(1)

    stereochemistry = ""
    stereo_match = re.search(r"(?:^|[\s(,-])(dl|d|l)-?(?:[\s),-]|$)", lower)
    if stereo_match:
        stereochemistry = stereo_match.group(1)

    labels = []
    if material_form != "unspecified":
        labels.append(material_form)
    if polymorph:
        labels.append(polymorph)
    if stereochemistry:
        labels.append(stereochemistry)
    return material_form, "/".join(labels), polymorph, stereochemistry, identity_form

@dataclass
class AntoineCoefficients:
    """
    Antoine equation coefficients.

    Form: log10(P) = A - B/(T + C)

    Units are stored as:
    - P in bar
    - T in °C (Celsius)
    """
    A: float
    B: float
    C: float
    T_min: float = 200.0  # K
    T_max: float = 500.0  # K
    source: str = "unknown"
    P_units: str = "bar"  # Original units before conversion

    def vapor_pressure(self, T_K: float) -> float:
        """
        Calculate vapor pressure at temperature T.

        Args:
            T_K: Temperature in Kelvin

        Returns:
            Vapor pressure in bar
        """
        T_C = T_K - 273.15

        try:
            log10_P = self.A - self.B / (self.C + T_C)
            return 10 ** log10_P
        except (ValueError, ZeroDivisionError, OverflowError):
            return None

    def covers_temperature(self, T_K: float, tolerance: float = 1e-9) -> bool:
        if self.T_min is not None and T_K < self.T_min - tolerance:
            return False
        if self.T_max is not None and T_K > self.T_max + tolerance:
            return False
        return True

    def distance_to_range(self, T_K: float) -> float:
        if self.covers_temperature(T_K):
            return 0.0
        if self.T_min is not None and T_K < self.T_min:
            return self.T_min - T_K
        if self.T_max is not None and T_K > self.T_max:
            return T_K - self.T_max
        return 0.0

    @property
    def range_width(self) -> float:
        if self.T_min is None or self.T_max is None:
            return float("inf")
        return max(self.T_max - self.T_min, 1e-12)


@dataclass(frozen=True)
class PsatBoilingPointValidation:
    """Provider-neutral normal-boiling-point reproduction diagnostics."""

    T_boiling: float
    covers_boiling_point: bool
    pressure_bar: Optional[float]
    relative_error: Optional[float]
    accepted: bool


AntoineBoilingPointValidation = PsatBoilingPointValidation


@dataclass(frozen=True)
class DirectPsatConfidenceProfile:
    """Ordered confidence tiers for one gated direct-Psat source class."""

    standalone_quality: float
    higher_priority_overlap_quality: float
    hard_tb_quality: float

    def __post_init__(self) -> None:
        values = (
            self.standalone_quality,
            self.higher_priority_overlap_quality,
            self.hard_tb_quality,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("Direct-Psat confidence qualities must be finite")
        if not (
            0.0 <= self.standalone_quality
            <= self.higher_priority_overlap_quality
            <= self.hard_tb_quality
            <= 1.0
        ):
            raise ValueError(
                "Direct-Psat confidence must satisfy "
                "0 <= standalone <= overlap <= hard Tb <= 1"
            )


STANDARD_DIRECT_TABLE_PROFILE = DirectPsatConfidenceProfile(
    standalone_quality=0.90,
    higher_priority_overlap_quality=0.93,
    hard_tb_quality=0.95,
)

CACHED_NIST_ANTOINE_PROFILE = DirectPsatConfidenceProfile(
    standalone_quality=0.87,
    higher_priority_overlap_quality=0.90,
    hard_tb_quality=0.95,
)

HIGH_QUALITY_ANTOINE_PROFILE = DirectPsatConfidenceProfile(
    standalone_quality=0.95,
    higher_priority_overlap_quality=0.96,
    hard_tb_quality=0.97,
)


@dataclass(frozen=True)
class DirectPsatAdmissionDecision:
    """Provider-neutral admission result for a direct Psat segment."""

    admitted: bool
    validation_status: str
    selected_quality: Optional[float]
    handoff_requirement: Optional[str]
    pressure_bar: Optional[float]
    relative_error: Optional[float]
    warning: Optional[str]

    @property
    def quality_basis(self) -> Optional[str]:
        if not self.admitted:
            return None
        if self.validation_status == "exempt_priority":
            return "exempt_priority"
        if self.validation_status == "hard_tb_validated":
            return "hard_tb_validation"
        return "standalone_unvalidated"

    @property
    def tb_validation_required(self) -> bool:
        return self.validation_status != "exempt_priority"


def validate_psat_boiling_point(
    pressure_at_temperature: Callable[[float], Optional[float]],
    T_min: float,
    T_max: float,
    T_boiling: float,
    *,
    range_tolerance_K: float = ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K,
    relative_tolerance: float = ONLINE_ANTOINE_TB_REL_TOL,
    target_pressure_bar: float = NORMAL_BOILING_PRESSURE_BAR,
) -> PsatBoilingPointValidation:
    """Check whether a bounded direct Psat source reproduces normal pressure."""
    if not callable(pressure_at_temperature):
        raise TypeError("Direct Psat boiling-point validation requires a callable")
    values = (
        T_min,
        T_max,
        T_boiling,
        range_tolerance_K,
        relative_tolerance,
        target_pressure_bar,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Direct Psat boiling-point validation inputs must be finite")
    if (
        T_min <= 0.0
        or T_max <= T_min
        or T_boiling <= 0.0
        or range_tolerance_K < 0.0
        or relative_tolerance < 0.0
    ):
        raise ValueError("Direct Psat boiling-point validation inputs are invalid")
    if target_pressure_bar <= 0.0:
        raise ValueError("Direct Psat boiling-point target pressure must be positive")

    covers_boiling_point = (
        T_min - range_tolerance_K
        <= T_boiling
        <= T_max + range_tolerance_K
    )
    if not covers_boiling_point:
        return PsatBoilingPointValidation(
            T_boiling=float(T_boiling),
            covers_boiling_point=False,
            pressure_bar=None,
            relative_error=None,
            accepted=False,
        )

    try:
        pressure_bar = pressure_at_temperature(float(T_boiling))
    except (ArithmeticError, TypeError, ValueError):
        pressure_bar = None
    try:
        pressure_bar = float(pressure_bar)
    except (TypeError, ValueError):
        pressure_bar = None
    if pressure_bar is None or not math.isfinite(pressure_bar) or pressure_bar <= 0.0:
        return PsatBoilingPointValidation(
            T_boiling=float(T_boiling),
            covers_boiling_point=True,
            pressure_bar=None,
            relative_error=None,
            accepted=False,
        )
    relative_error = abs(pressure_bar - target_pressure_bar) / target_pressure_bar
    return PsatBoilingPointValidation(
        T_boiling=float(T_boiling),
        covers_boiling_point=True,
        pressure_bar=float(pressure_bar),
        relative_error=float(relative_error),
        accepted=relative_error <= relative_tolerance,
    )


def validate_antoine_boiling_point(
    coefficients: AntoineCoefficients,
    T_boiling: float,
    *,
    range_tolerance_K: float = ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K,
    relative_tolerance: float = ONLINE_ANTOINE_TB_REL_TOL,
    target_pressure_bar: float = NORMAL_BOILING_PRESSURE_BAR,
) -> PsatBoilingPointValidation:
    """Compatibility wrapper over provider-neutral direct-Psat validation."""
    if not isinstance(coefficients, AntoineCoefficients):
        raise TypeError("Antoine boiling-point validation requires AntoineCoefficients")
    return validate_psat_boiling_point(
        coefficients.vapor_pressure,
        float(coefficients.T_min),
        float(coefficients.T_max),
        T_boiling,
        range_tolerance_K=range_tolerance_K,
        relative_tolerance=relative_tolerance,
        target_pressure_bar=target_pressure_bar,
    )


def admit_direct_psat_segment(
    *,
    priority: int,
    intrinsic_quality: float,
    confidence_profile: Optional[DirectPsatConfidenceProfile],
    qualified_Tb: Optional[float],
    pressure_at_temperature: Callable[[float], Optional[float]],
    T_min: float,
    T_max: float,
    source_label: str,
    existing_exempt_handoff: str = "none",
    exemption_priority: int = 600,
    range_tolerance_K: float = ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K,
    relative_tolerance: float = ONLINE_ANTOINE_TB_REL_TOL,
) -> DirectPsatAdmissionDecision:
    """Apply the shared priority and hard-Tb gate to one direct segment."""
    priority = int(priority)
    intrinsic_quality = float(intrinsic_quality)
    if not math.isfinite(intrinsic_quality) or not 0.0 <= intrinsic_quality <= 1.0:
        raise ValueError("Direct Psat intrinsic quality must lie in [0, 1]")
    if priority >= int(exemption_priority):
        return DirectPsatAdmissionDecision(
            admitted=True,
            validation_status="exempt_priority",
            selected_quality=intrinsic_quality,
            handoff_requirement=str(existing_exempt_handoff),
            pressure_bar=None,
            relative_error=None,
            warning=None,
        )
    if confidence_profile is None:
        raise ValueError("Gated direct Psat sources require a confidence profile")
    if qualified_Tb is None:
        return DirectPsatAdmissionDecision(
            admitted=True,
            validation_status="validation_unavailable",
            selected_quality=confidence_profile.standalone_quality,
            handoff_requirement="higher_preference_overlap_if_available",
            pressure_bar=None,
            relative_error=None,
            warning=None,
        )
    validation = validate_psat_boiling_point(
        pressure_at_temperature,
        T_min,
        T_max,
        float(qualified_Tb),
        range_tolerance_K=range_tolerance_K,
        relative_tolerance=relative_tolerance,
    )
    if not validation.covers_boiling_point:
        return DirectPsatAdmissionDecision(
            admitted=True,
            validation_status="hard_tb_outside_range",
            selected_quality=confidence_profile.standalone_quality,
            handoff_requirement="higher_preference_overlap_if_available",
            pressure_bar=None,
            relative_error=None,
            warning=None,
        )
    if validation.accepted:
        return DirectPsatAdmissionDecision(
            admitted=True,
            validation_status="hard_tb_validated",
            selected_quality=confidence_profile.hard_tb_quality,
            handoff_requirement="none",
            pressure_bar=validation.pressure_bar,
            relative_error=validation.relative_error,
            warning=None,
        )
    pressure_text = (
        f"Psat(Tb)={validation.pressure_bar:.8g} bar"
        if validation.pressure_bar is not None
        else "Psat(Tb)=invalid"
    )
    error_text = (
        f"relative error={100.0 * validation.relative_error:.6g}%"
        if validation.relative_error is not None
        else "relative error=unavailable"
    )
    return DirectPsatAdmissionDecision(
        admitted=False,
        validation_status="hard_tb_conflict",
        selected_quality=None,
        handoff_requirement=None,
        pressure_bar=validation.pressure_bar,
        relative_error=validation.relative_error,
        warning=(
            f"Discarding {source_label} direct Psat segment: "
            f"Tb={float(qualified_Tb):.8g} K, {pressure_text}, {error_text}, "
            f"allowed tolerance={100.0 * relative_tolerance:g}%"
        ),
    )


@dataclass
class HvapTemperatureFit:
    """Temperature-dependent heat-of-vaporization fit."""

    A: float
    n: float
    Tc: float
    T_min: float
    T_max: float
    mape_percent: float
    kept_points: int
    total_points: int
    source: str = "unknown"
    method: str = "watson_power_fit"

    def value_at(self, T: float) -> Optional[float]:
        try:
            if T >= self.Tc:
                return 0.0
            if T <= 0.0:
                return None
            tau = 1.0 - T / self.Tc
            if tau <= 0.0:
                return None
            return self.A * tau**self.n
        except (ValueError, OverflowError, ZeroDivisionError):
            return None

@dataclass
class HeatCapacityLookup:
    """Temperature-specific online heat-capacity lookup result."""

    value: float
    phase: str
    T: float
    source: str = "NIST WebBook"
    method: str = "nist_tabulated_cp"
    quality: float = 0.88
    notes: str = ""

@dataclass
class HeatCapacityIntegralLookup:
    """Online heat-capacity integral result."""

    value: float
    phase: str
    T1: float
    T2: float
    source: str = "NIST WebBook"
    method: str = "nist_cp_integral"
    quality: float = 0.80
    notes: str = ""

class PropertyResolutionError(Exception):
    """Exception raised when property cannot be resolved"""
    pass
