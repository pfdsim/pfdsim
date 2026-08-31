"""Portable ideal-gas heat-capacity kernels and canonical source lookup.

The classes in this module are deliberately independent of the property
resolver.  A resolved kernel is an immutable, executable correlation which can
be retained by thermodynamic models without carrying any SQLite, JSON, online
lookup, or source-selection work into their numeric loops.
"""

from __future__ import annotations

import json
import hashlib
import math
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .organic_classification import is_strict_organic_formula_counts


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DATABASE_PATH = ROOT / "data" / "ideal_gas_heat_capacity.sqlite"
ATOM_INCREMENT_MODEL_PATH = (
    ROOT / "data" / "ideal_gas_cp_atom_increment_shomate.json"
)
KERNEL_CONTRACT_VERSION = 1
DEFAULT_TMIN_K = 273.15
DEFAULT_TMAX_K = 1500.0
EXTRAPOLATION_WIDTH_K = 10.0
EXTRAPOLATION_QUALITY_PENALTY = 0.02
CLAMP_QUALITY_PENALTY_PER_5K = 0.02
MAX_RANGE_QUALITY_PENALTY = 0.40
ATOM_INCREMENT_MODEL_QUALITY = 0.75
ATOM_INCREMENT_ORGANIC_QUALITY = 0.80
CONVENTIONAL_ORGANIC_CATEGORIES = frozenset({
    "H", "C", "O", "N", "F", "Cl", "P", "S", "Br", "I",
})


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("ideal-gas Cp kernel values must be finite")
    return result


def _polynomial_value(coefficients: Sequence[float], x: float) -> float:
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def _polynomial_difference(coefficients: Sequence[float], x1: float, x2: float) -> float:
    """Evaluate P(x2)-P(x1) without subtracting two large primitives."""
    if len(coefficients) <= 1 or x1 == x2:
        return 0.0
    quotient = float(coefficients[-1])
    value = quotient
    for index in range(len(coefficients) - 2, 0, -1):
        quotient = float(coefficients[index]) + x1 * quotient
        value = quotient + x2 * value
    return (x2 - x1) * value


@dataclass(frozen=True)
class KernelEvaluation:
    value: float
    evaluation_temperature_K: float
    quality: float
    range_penalty: float = 0.0
    range_note: str = ""


@dataclass(frozen=True)
class IdealGasCpKernel:
    """Common range conditioning and provenance for an executable Cp curve."""

    Tmin: float
    Tmax: float
    quality: float
    source: str
    method: str
    notes: str = ""
    fit_mape_percent: float = 0.0
    fit_max_error_percent: float = 0.0
    source_fingerprint: str = ""

    kind = "base"

    def __post_init__(self) -> None:
        Tmin = _finite(self.Tmin)
        Tmax = _finite(self.Tmax)
        quality = _finite(self.quality)
        if Tmin <= 0.0 or Tmax < Tmin:
            raise ValueError("ideal-gas Cp kernel requires 0 < Tmin <= Tmax")
        object.__setattr__(self, "Tmin", Tmin)
        object.__setattr__(self, "Tmax", Tmax)
        object.__setattr__(self, "quality", min(1.0, max(0.0, quality)))

    def covers(self, T: float) -> bool:
        return self.Tmin <= float(T) <= self.Tmax

    def covers_interval(self, T1: float, T2: float) -> bool:
        return self.Tmin <= min(T1, T2) and max(T1, T2) <= self.Tmax

    @property
    def extended_Tmin(self) -> float:
        return max(1.0e-9, self.Tmin - EXTRAPOLATION_WIDTH_K)

    @property
    def extended_Tmax(self) -> float:
        return self.Tmax + EXTRAPOLATION_WIDTH_K

    def _condition_temperature(self, T: float) -> tuple[float, float, str]:
        temperature = _finite(T)
        if temperature <= 0.0:
            raise ValueError("heat-capacity temperature must be positive")
        if self.Tmin <= temperature <= self.Tmax:
            return temperature, 0.0, ""

        if temperature < self.Tmin:
            distance = self.Tmin - temperature
            direction = "below"
            extended = self.extended_Tmin
            evaluation_T = max(temperature, extended)
        else:
            distance = temperature - self.Tmax
            direction = "above"
            extended = self.extended_Tmax
            evaluation_T = min(temperature, extended)

        if distance <= EXTRAPOLATION_WIDTH_K + 1.0e-12:
            return (
                evaluation_T,
                EXTRAPOLATION_QUALITY_PENALTY,
                f"extrapolated {distance:g} K {direction} fitted range",
            )

        clamped_distance = distance - EXTRAPOLATION_WIDTH_K
        penalty = min(
            MAX_RANGE_QUALITY_PENALTY,
            EXTRAPOLATION_QUALITY_PENALTY
            + CLAMP_QUALITY_PENALTY_PER_5K * clamped_distance / 5.0,
        )
        return (
            evaluation_T,
            penalty,
            f"clamped {direction} extended range after {EXTRAPOLATION_WIDTH_K:g} K extrapolation; "
            f"query is {distance:g} K outside fitted range",
        )

    def evaluate(self, T: float) -> KernelEvaluation:
        evaluation_T, penalty, note = self._condition_temperature(T)
        value = float(self._cp_native(evaluation_T))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("ideal-gas Cp kernel produced a nonpositive or nonfinite value")
        return KernelEvaluation(
            value=value,
            evaluation_temperature_K=evaluation_T,
            quality=max(0.0, self.quality - penalty),
            range_penalty=penalty,
            range_note=note,
        )

    def cp(self, T: float) -> float:
        evaluation_T, _, _ = self._condition_temperature(T)
        value = float(self._cp_native(evaluation_T))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("ideal-gas Cp kernel produced a nonpositive or nonfinite value")
        return value

    def quality_at(self, T: float) -> float:
        _, penalty, _ = self._condition_temperature(T)
        return max(0.0, self.quality - penalty)

    def _integrate_conditioned(
        self,
        T1: float,
        T2: float,
        *,
        entropy: bool,
    ) -> float:
        first = _finite(T1)
        second = _finite(T2)
        if first <= 0.0 or second <= 0.0:
            raise ValueError("heat-capacity integration temperatures must be positive")
        if first == second:
            return 0.0
        if second < first:
            return -self._integrate_conditioned(second, first, entropy=entropy)

        lower = self.extended_Tmin
        upper = self.extended_Tmax
        total = 0.0
        cursor = first
        if cursor < lower:
            end = min(second, lower)
            constant = self._cp_native(lower)
            total += constant * (math.log(end / cursor) if entropy else end - cursor)
            cursor = end
        if cursor < second and cursor < upper:
            end = min(second, upper)
            if entropy:
                total += self._delta_s_native(cursor, end)
            else:
                total += self._delta_h_native(cursor, end)
            cursor = end
        if cursor < second:
            constant = self._cp_native(upper)
            total += constant * (math.log(second / cursor) if entropy else second - cursor)
        return float(total)

    def delta_h(self, T1: float, T2: float) -> float:
        """Return integral Cp dT in J/mol using the conditioned curve."""
        return self._integrate_conditioned(T1, T2, entropy=False)

    def delta_s(self, T1: float, T2: float) -> float:
        """Return integral Cp/T dT in J/(mol*K) using the conditioned curve."""
        return self._integrate_conditioned(T1, T2, entropy=True)

    def _cp_native(self, T: float) -> float:
        raise NotImplementedError

    def _validate_curve(self, points: int = 65) -> None:
        """Refuse kernels that fail anywhere on the fitted range or 10 K wings."""
        lower = self.extended_Tmin
        upper = self.extended_Tmax
        if points < 2 or upper <= lower:
            temperatures = (lower, upper)
        else:
            ratio = (upper / lower) ** (1.0 / (points - 1))
            temperatures = tuple(lower * ratio**index for index in range(points))
        for temperature in temperatures:
            value = float(self._cp_native(temperature))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "ideal-gas Cp kernel is nonpositive or nonfinite within its admitted range"
                )

    def _delta_h_native(self, T1: float, T2: float) -> float:
        raise NotImplementedError

    def _delta_s_native(self, T1: float, T2: float) -> float:
        raise NotImplementedError

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract_version": KERNEL_CONTRACT_VERSION,
            "kind": self.kind,
            "Tmin_K": self.Tmin,
            "Tmax_K": self.Tmax,
            "quality": self.quality,
            "source": self.source,
            "method": self.method,
            "notes": self.notes,
            "fit_mape_percent": self.fit_mape_percent,
            "fit_max_error_percent": self.fit_max_error_percent,
            "source_fingerprint": self.source_fingerprint,
        }


@dataclass(frozen=True)
class PolynomialCpKernel(IdealGasCpKernel):
    """Exact Cp polynomial in absolute temperature, coefficients ascending."""

    coefficients: tuple[float, ...] = ()
    kind = "polynomial"

    def __post_init__(self) -> None:
        super().__post_init__()
        coefficients = tuple(_finite(value) for value in self.coefficients)
        if not coefficients:
            raise ValueError("polynomial Cp kernel requires coefficients")
        object.__setattr__(self, "coefficients", coefficients)
        self._validate_curve()

    def _cp_native(self, T: float) -> float:
        return _polynomial_value(self.coefficients, T)

    def _delta_h_native(self, T1: float, T2: float) -> float:
        return sum(
            coefficient * (T2 ** (power + 1) - T1 ** (power + 1)) / (power + 1)
            for power, coefficient in enumerate(self.coefficients)
        )

    def _delta_s_native(self, T1: float, T2: float) -> float:
        total = self.coefficients[0] * math.log(T2 / T1)
        for power, coefficient in enumerate(self.coefficients[1:], start=1):
            total += coefficient * (T2**power - T1**power) / power
        return total

    def to_payload(self) -> dict[str, Any]:
        return {**super().to_payload(), "coefficients": list(self.coefficients)}


@dataclass(frozen=True)
class ShomateCpKernel(IdealGasCpKernel):
    """Standard five-coefficient Shomate heat-capacity equation."""

    coefficients: tuple[float, ...] = ()
    kind = "shomate"

    def __post_init__(self) -> None:
        super().__post_init__()
        coefficients = tuple(_finite(value) for value in self.coefficients)
        if len(coefficients) != 5:
            raise ValueError("Shomate Cp kernel requires five coefficients")
        object.__setattr__(self, "coefficients", coefficients)
        self._validate_curve()

    def _cp_native(self, T: float) -> float:
        t = T / 1000.0
        A, B, C, D, E = self.coefficients
        return A + t * (B + t * (C + t * D)) + E / (t * t)

    def _delta_h_native(self, T1: float, T2: float) -> float:
        t1 = T1 / 1000.0
        t2 = T2 / 1000.0
        A, B, C, D, E = self.coefficients
        return 1000.0 * (
            A * (t2 - t1)
            + B * (t2 * t2 - t1 * t1) / 2.0
            + C * (t2**3 - t1**3) / 3.0
            + D * (t2**4 - t1**4) / 4.0
            - E * (1.0 / t2 - 1.0 / t1)
        )

    def _delta_s_native(self, T1: float, T2: float) -> float:
        t1 = T1 / 1000.0
        t2 = T2 / 1000.0
        A, B, C, D, E = self.coefficients
        return (
            A * math.log(t2 / t1)
            + B * (t2 - t1)
            + C * (t2 * t2 - t1 * t1) / 2.0
            + D * (t2**3 - t1**3) / 3.0
            - E * (1.0 / (t2 * t2) - 1.0 / (t1 * t1)) / 2.0
        )

    def to_payload(self) -> dict[str, Any]:
        return {**super().to_payload(), "coefficients": list(self.coefficients)}


@dataclass(frozen=True)
class AtomIncrementCpModel:
    """Validated empirical atom-increment model loaded from bundled JSON."""

    fingerprint: str
    parameters: tuple[tuple[str, tuple[float, ...]], ...]
    element_routes: tuple[tuple[str, str], ...]
    cv_median_mape_percent: float
    cv_p90_mape_percent: float

    def normalized_counts(
        self, atom_counts: Mapping[str, Any]
    ) -> Optional[tuple[tuple[str, int], ...]]:
        routes = dict(self.element_routes)
        category_counts: dict[str, int] = {}
        for element, raw_count in atom_counts.items():
            try:
                count = int(raw_count)
                numeric_count = float(raw_count)
            except (TypeError, ValueError, OverflowError):
                return None
            if count <= 0 or numeric_count != count:
                return None
            category = routes.get(str(element))
            if category is None:
                return None
            category_counts[category] = category_counts.get(category, 0) + count
        if not category_counts:
            return None
        return tuple(sorted(category_counts.items()))

    def cache_identity(self, atom_counts: Mapping[str, Any]) -> Optional[str]:
        normalized = self.normalized_counts(atom_counts)
        if normalized is None:
            return None
        counts = ";".join(f"{category}:{count}" for category, count in normalized)
        return f"{self.fingerprint}|{counts}"

    def is_conventional_organic(self, atom_counts: Mapping[str, Any]) -> bool:
        normalized = self.normalized_counts(atom_counts)
        if normalized is None:
            return False
        categories = {category for category, _ in normalized}
        return (
            {"C", "H"} <= categories
            and categories <= CONVENTIONAL_ORGANIC_CATEGORIES
            and is_strict_organic_formula_counts(atom_counts)
        )

    def kernel(
        self,
        atom_counts: Mapping[str, Any],
        *,
        identity_note: str = "",
    ) -> Optional[ShomateCpKernel]:
        normalized = self.normalized_counts(atom_counts)
        if normalized is None:
            return None
        parameters = dict(self.parameters)
        coefficients = [0.0] * 5
        for category, count in normalized:
            contribution = parameters.get(category)
            if contribution is None:
                return None
            for index, value in enumerate(contribution):
                coefficients[index] += count * value
        count_note = ", ".join(
            f"{category}={count}" for category, count in normalized
        )
        notes = (
            "Empirical atom-increment Shomate estimate; "
            f"formula-grouped 5-fold CV median range MAPE "
            f"{self.cv_median_mape_percent:.3g}%, P90 "
            f"{self.cv_p90_mape_percent:.3g}%; atom categories: {count_note}"
        )
        if identity_note:
            notes += f"; {identity_note}"
        conventional_organic = self.is_conventional_organic(atom_counts)
        quality = (
            ATOM_INCREMENT_ORGANIC_QUALITY
            if conventional_organic
            else ATOM_INCREMENT_MODEL_QUALITY
        )
        notes += (
            "; conventional-organic validation tier"
            if conventional_organic
            else "; general-species validation tier"
        )
        return ShomateCpKernel(
            Tmin=DEFAULT_TMIN_K,
            Tmax=DEFAULT_TMAX_K,
            quality=quality,
            source="estimated",
            method="atom_increment_shomate_ideal_gas_cp_kernel",
            notes=(
                notes
                + "; coefficients from PFDsim bundled atom-increment model"
            ),
            source_fingerprint=self.fingerprint,
            coefficients=tuple(coefficients),
        )


@dataclass(frozen=True)
class ChebyshevCpKernel(IdealGasCpKernel):
    """Rationally mapped Chebyshev Cp with stored analytic primitives."""

    center: float = 0.0
    scale: float = 0.0
    coefficients: tuple[float, ...] = ()
    h_polynomial: tuple[float, ...] = ()
    h_log_coefficient: float = 0.0
    h_reciprocal_coefficient: float = 0.0
    s_polynomial: tuple[float, ...] = ()
    s_log_minus_coefficient: float = 0.0
    s_log_plus_coefficient: float = 0.0
    kind = "chebyshev"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "center", _finite(self.center))
        object.__setattr__(self, "scale", _finite(self.scale))
        object.__setattr__(self, "coefficients", tuple(_finite(v) for v in self.coefficients))
        object.__setattr__(self, "h_polynomial", tuple(_finite(v) for v in self.h_polynomial))
        object.__setattr__(self, "s_polynomial", tuple(_finite(v) for v in self.s_polynomial))
        if len(self.coefficients) not in {9, 13}:
            raise ValueError("Chebyshev Cp kernel requires degree 8 or 12")
        if not (self.center > 0.0 and 0.0 < self.scale < 1.0):
            raise ValueError("invalid rational Chebyshev mapping")
        self._validate_curve()

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def _map(self, T: float) -> float:
        return (T - self.center) / (self.scale * (T + self.center))

    def _cp_native(self, T: float) -> float:
        x = self._map(T)
        b1 = 0.0
        b2 = 0.0
        for index in range(len(self.coefficients) - 1, 0, -1):
            b0 = 2.0 * x * b1 - b2 + self.coefficients[index]
            b2 = b1
            b1 = b0
        return x * b1 - b2 + self.coefficients[0]

    def _delta_h_native(self, T1: float, T2: float) -> float:
        x1 = self._map(T1)
        x2 = self._map(T2)
        delta_x = x2 - x1
        scale = self.scale
        return (
            _polynomial_difference(self.h_polynomial, x1, x2)
            + self.h_log_coefficient
            * math.log1p((-scale * delta_x) / (1.0 - scale * x1))
            + self.h_reciprocal_coefficient
            * scale
            * delta_x
            / ((1.0 - scale * x2) * (1.0 - scale * x1))
        )

    def _delta_s_native(self, T1: float, T2: float) -> float:
        x1 = self._map(T1)
        x2 = self._map(T2)
        delta_x = x2 - x1
        scale = self.scale
        return (
            _polynomial_difference(self.s_polynomial, x1, x2)
            + self.s_log_minus_coefficient
            * math.log1p((-scale * delta_x) / (1.0 - scale * x1))
            + self.s_log_plus_coefficient
            * math.log1p((scale * delta_x) / (1.0 + scale * x1))
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "center_K": self.center,
            "scale": self.scale,
            "degree": self.degree,
            "coefficients": list(self.coefficients),
            "h_polynomial": list(self.h_polynomial),
            "h_log_coefficient": self.h_log_coefficient,
            "h_reciprocal_coefficient": self.h_reciprocal_coefficient,
            "s_polynomial": list(self.s_polynomial),
            "s_log_minus_coefficient": self.s_log_minus_coefficient,
            "s_log_plus_coefficient": self.s_log_plus_coefficient,
        }


def kernel_from_payload(payload: Mapping[str, Any]) -> IdealGasCpKernel:
    if int(payload.get("contract_version", 0)) != KERNEL_CONTRACT_VERSION:
        raise ValueError("unsupported ideal-gas Cp kernel contract")
    common = dict(
        Tmin=payload["Tmin_K"],
        Tmax=payload["Tmax_K"],
        quality=payload["quality"],
        source=payload["source"],
        method=payload["method"],
        notes=payload.get("notes", ""),
        fit_mape_percent=payload.get("fit_mape_percent", 0.0),
        fit_max_error_percent=payload.get("fit_max_error_percent", 0.0),
        source_fingerprint=payload.get("source_fingerprint", ""),
    )
    kind = str(payload.get("kind", ""))
    if kind == "polynomial":
        return PolynomialCpKernel(**common, coefficients=tuple(payload["coefficients"]))
    if kind == "shomate":
        return ShomateCpKernel(**common, coefficients=tuple(payload["coefficients"]))
    if kind == "chebyshev":
        return ChebyshevCpKernel(
            **common,
            center=payload["center_K"],
            scale=payload["scale"],
            coefficients=tuple(payload["coefficients"]),
            h_polynomial=tuple(payload["h_polynomial"]),
            h_log_coefficient=payload["h_log_coefficient"],
            h_reciprocal_coefficient=payload["h_reciprocal_coefficient"],
            s_polynomial=tuple(payload["s_polynomial"]),
            s_log_minus_coefficient=payload["s_log_minus_coefficient"],
            s_log_plus_coefficient=payload["s_log_plus_coefficient"],
        )
    raise ValueError(f"unsupported ideal-gas Cp kernel kind {kind!r}")


def fit_quality_penalty(max_error_percent: float, mape_percent: float) -> float:
    if max_error_percent < 0.1 and mape_percent < 0.01:
        return 0.0
    return 0.031 * max_error_percent + 0.185 * mape_percent


def _primitive_terms(
    coefficients: Sequence[float], center: float, scale: float
) -> tuple[tuple[float, ...], float, float, tuple[float, ...], float, float]:
    import numpy as np
    from numpy.polynomial import chebyshev as ncheb
    from numpy.polynomial import polynomial as npoly

    power = np.asarray(ncheb.cheb2poly(np.asarray(coefficients, dtype=float)), dtype=float)
    h_numerator = power * (2.0 * center * scale)
    h_quotient, h_remainder = npoly.polydiv(
        h_numerator, np.asarray([1.0, -2.0 * scale, scale * scale])
    )
    h_remainder = np.pad(h_remainder, (0, max(0, 2 - len(h_remainder))))
    h_A = -h_remainder[1] / scale
    h_B = h_remainder[0] - h_A

    s_numerator = power * (2.0 * scale)
    s_quotient, s_remainder = npoly.polydiv(
        s_numerator, np.asarray([1.0, 0.0, -(scale * scale)])
    )
    s_remainder = np.pad(s_remainder, (0, max(0, 2 - len(s_remainder))))
    s_A = 0.5 * (s_remainder[0] + s_remainder[1] / scale)
    s_B = 0.5 * (s_remainder[0] - s_remainder[1] / scale)
    return (
        tuple(float(v) for v in npoly.polyint(h_quotient)),
        float(h_A / -scale),
        float(h_B / scale),
        tuple(float(v) for v in npoly.polyint(s_quotient)),
        float(-s_A / scale),
        float(s_B / scale),
    )


def fit_chebyshev_kernel(
    evaluator: Callable[[Any], Any],
    Tmin: float,
    Tmax: float,
    *,
    quality: float,
    source: str,
    method: str,
    notes: str = "",
    source_fingerprint: str = "",
    degree: int = 8,
) -> ChebyshevCpKernel:
    """Fit one deterministic positive Cp source to a degree-8/12 kernel."""
    import numpy as np
    from numpy.polynomial import chebyshev as ncheb

    Tmin = _finite(Tmin)
    Tmax = _finite(Tmax)
    if degree not in {8, 12} or Tmin <= 0.0 or Tmax <= Tmin:
        raise ValueError("invalid Chebyshev Cp fitting contract")
    center = math.sqrt(Tmin * Tmax)
    scale = (Tmax - center) / (Tmax + center)
    training_T = np.geomspace(Tmin, Tmax, 240)
    values = np.asarray(evaluator(training_T), dtype=float)
    if values.shape != training_T.shape:
        values = np.asarray([evaluator(float(T)) for T in training_T], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("cannot fit nonpositive or nonfinite heat-capacity source")
    x = (training_T - center) / (scale * (training_T + center))
    coefficients = ncheb.chebfit(x, values, degree, w=1.0 / values)

    validation_T = np.geomspace(Tmin, Tmax, 1201)
    reference = np.asarray(evaluator(validation_T), dtype=float)
    if reference.shape != validation_T.shape:
        reference = np.asarray([evaluator(float(T)) for T in validation_T], dtype=float)
    predicted = ncheb.chebval(
        (validation_T - center) / (scale * (validation_T + center)), coefficients
    )
    if np.any(~np.isfinite(reference)) or np.any(reference <= 0.0) or np.any(predicted <= 0.0):
        raise ValueError("invalid heat-capacity fit validation values")
    errors = np.abs(predicted / reference - 1.0) * 100.0
    mape = float(np.mean(errors))
    maximum = float(np.max(errors))
    penalty = fit_quality_penalty(maximum, mape)
    primitives = _primitive_terms(coefficients, center, scale)
    return ChebyshevCpKernel(
        Tmin=Tmin,
        Tmax=Tmax,
        quality=max(0.0, float(quality) - penalty),
        source=source,
        method=method,
        notes=notes,
        fit_mape_percent=mape,
        fit_max_error_percent=maximum,
        source_fingerprint=source_fingerprint,
        center=center,
        scale=scale,
        coefficients=tuple(float(v) for v in coefficients),
        h_polynomial=primitives[0],
        h_log_coefficient=primitives[1],
        h_reciprocal_coefficient=primitives[2],
        s_polynomial=primitives[3],
        s_log_minus_coefficient=primitives[4],
        s_log_plus_coefficient=primitives[5],
    )


def shifted_polynomial_coefficients(
    coefficients: Sequence[float], *, reference: float = 298.15, scale: float = 100.0
) -> tuple[float, ...]:
    """Expand sum(a_n*((T-reference)/scale)^n) into powers of T."""
    result = [0.0] * len(coefficients)
    for power, coefficient in enumerate(coefficients):
        scaled = float(coefficient) / scale**power
        for target_power in range(power + 1):
            result[target_power] += (
                scaled
                * math.comb(power, target_power)
                * (-reference) ** (power - target_power)
            )
    while len(result) > 1 and abs(result[-1]) == 0.0:
        result.pop()
    return tuple(result)


_ATOM_INCREMENT_MODEL_CACHE: dict[Path, Optional[AtomIncrementCpModel]] = {}
_ATOM_INCREMENT_MODEL_LOCK = threading.Lock()


def load_atom_increment_model(
    *, path: Path = ATOM_INCREMENT_MODEL_PATH
) -> Optional[AtomIncrementCpModel]:
    """Load and validate the bundled atom-increment Shomate model once."""
    try:
        normalized_path = path.resolve()
    except OSError:
        normalized_path = path.absolute()
    with _ATOM_INCREMENT_MODEL_LOCK:
        if normalized_path in _ATOM_INCREMENT_MODEL_CACHE:
            return _ATOM_INCREMENT_MODEL_CACHE[normalized_path]

    model = None
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported atom-increment Cp model schema")
        if payload.get("model") != "shomate_atom_increment_ideal_gas_cp_v1":
            raise ValueError("unsupported atom-increment Cp model")

        raw_parameters = payload["parameters"]
        parameters = []
        for category, entry in raw_parameters.items():
            raw = entry["coefficients"]
            coefficients = tuple(_finite(raw[name]) for name in "ABCDE")
            parameters.append((str(category), coefficients))
        if not parameters:
            raise ValueError("atom-increment Cp model has no parameters")

        routing = payload["category_routing"]
        element_routes: dict[str, str] = {}
        for element, category in routing["explicit_elements"].items():
            element_routes[str(element)] = str(category)
        for category, elements in routing["metal_categories"].items():
            for element in elements:
                element_routes[str(element)] = str(category)
        for element in routing["metalloid_elements"]:
            element_routes[str(element)] = "metalloid"
        for element in routing["noble_gas_elements"]:
            element_routes[str(element)] = "noble_gas"

        parameter_names = {category for category, _ in parameters}
        if not set(element_routes.values()) <= parameter_names:
            raise ValueError("atom-increment routing references missing parameters")
        diagnostics = payload["diagnostics"]["cross_validation"]
        if diagnostics.get("grouping") != "formula":
            raise ValueError("atom-increment model lacks formula-grouped validation")
        if int(diagnostics.get("nonpositive_predicted_curves", -1)) != 0:
            raise ValueError("atom-increment model contains nonpositive validation curves")
        range_diagnostics = diagnostics["range_mape_percent"]
        model = AtomIncrementCpModel(
            fingerprint=hashlib.sha256(encoded).hexdigest(),
            parameters=tuple(sorted(parameters)),
            element_routes=tuple(sorted(element_routes.items())),
            cv_median_mape_percent=_finite(range_diagnostics["median"]),
            cv_p90_mape_percent=_finite(range_diagnostics["p90"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        model = None

    with _ATOM_INCREMENT_MODEL_LOCK:
        _ATOM_INCREMENT_MODEL_CACHE[normalized_path] = model
    return model


def clear_atom_increment_model_cache() -> None:
    with _ATOM_INCREMENT_MODEL_LOCK:
        _ATOM_INCREMENT_MODEL_CACHE.clear()


_BUNDLED_CACHE: dict[tuple[Path, str], Optional[IdealGasCpKernel]] = {}
_BUNDLED_LOCK = threading.Lock()


def load_bundled_kernel(
    cas: str, *, path: Path = CANONICAL_DATABASE_PATH
) -> Optional[IdealGasCpKernel]:
    """Load and process-cache one canonical CAS-keyed database record."""
    key = str(cas or "").strip()
    if not key:
        return None
    try:
        normalized_path = path.resolve()
    except OSError:
        normalized_path = path.absolute()
    cache_key = (normalized_path, key)
    with _BUNDLED_LOCK:
        if cache_key in _BUNDLED_CACHE:
            return _BUNDLED_CACHE[cache_key]
    if not path.is_file():
        return None
    try:
        uri = f"file:{path.resolve()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM canonical_ideal_gas_cp WHERE cas = ?", (key,)
            ).fetchone()
    except (OSError, sqlite3.Error):
        return None
    kernel = None
    if row is not None:
        try:
            kernel = ChebyshevCpKernel(
                Tmin=row["Tmin_fit_K"],
                Tmax=row["Tmax_fit_K"],
                quality=row["quality"],
                source=row["source_label"],
                method=f"canonical_{row['source']}_ideal_gas_cp",
                notes=(
                    f"Bundled canonical ideal-gas Cp; source range "
                    f"{row['Tmin_source_K']:g}-{row['Tmax_source_K']:g} K; "
                    f"fit MAPE {row['fit_mape_percent']:.4g}%, "
                    f"max error {row['fit_max_error_percent']:.4g}%"
                ),
                fit_mape_percent=row["fit_mape_percent"],
                fit_max_error_percent=row["fit_max_error_percent"],
                source_fingerprint=row["source_fingerprint"],
                center=row["map_center_K"],
                scale=row["map_scale"],
                coefficients=tuple(json.loads(row["cp_coefficients_json"])),
                h_polynomial=tuple(json.loads(row["h_polynomial_json"])),
                h_log_coefficient=row["h_log_coefficient"],
                h_reciprocal_coefficient=row["h_reciprocal_coefficient"],
                s_polynomial=tuple(json.loads(row["s_polynomial_json"])),
                s_log_minus_coefficient=row["s_log_minus_coefficient"],
                s_log_plus_coefficient=row["s_log_plus_coefficient"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            kernel = None
    with _BUNDLED_LOCK:
        _BUNDLED_CACHE[cache_key] = kernel
    return kernel


def clear_bundled_kernel_cache() -> None:
    with _BUNDLED_LOCK:
        _BUNDLED_CACHE.clear()
