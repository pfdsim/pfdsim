"""Portable ordinary-liquid heat-capacity kernels and bundled source lookup.

The runtime contract intentionally mirrors :mod:`property_resolution.ideal_gas_cp`:
one immutable resolved curve supplies scalar heat capacity plus analytic enthalpy
and entropy increments. Liquid kernels use the same 10 K continuation and
range-quality penalty policy as ideal-gas kernels.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .ideal_gas_cp import (
    CLAMP_QUALITY_PENALTY_PER_5K,
    DEFAULT_TMAX_K,
    DEFAULT_TMIN_K,
    EXTRAPOLATION_QUALITY_PENALTY,
    EXTRAPOLATION_WIDTH_K,
    IdealGasCpKernel,
    MAX_RANGE_QUALITY_PENALTY,
    _finite,
    _polynomial_difference,
    _polynomial_value,
    fit_quality_penalty,
)


if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..physical_constants import R_J_MOL_K
else:
    from physical_constants import R_J_MOL_K

ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DATABASE_PATH = ROOT / "data" / "liquid_heat_capacity.sqlite"
KERNEL_CONTRACT_VERSION = 1
STP_POINT_HALF_WIDTH_K = 5.0
ROWLINSON_BONDI_QUALITY_FACTOR = 0.89
HBD_RATIO_GC_QUALITY_FACTOR = 0.82
MIXED_DONOR_BONDI_QUALITY_FACTOR = 0.70
SCALED_IDEAL_GAS_QUALITY_FACTOR = 0.60
ESTIMATOR_MINIMUM_REDUCED_TEMPERATURE = 0.30
ESTIMATOR_MAXIMUM_REDUCED_TEMPERATURE = 0.95
HBD_RATIO_MINIMUM = 0.15
HBD_RATIO_MAXIMUM = 1.25
MINIMUM_ESTIMATOR_CRITICAL_QUALITY = 0.70
R = R_J_MOL_K


def _chebyshev_value(coefficients: Sequence[float], x: float) -> float:
    b1 = 0.0
    b2 = 0.0
    for index in range(len(coefficients) - 1, 0, -1):
        b0 = 2.0 * x * b1 - b2 + float(coefficients[index])
        b2 = b1
        b1 = b0
    return x * b1 - b2 + float(coefficients[0])


def _chebyshev_difference(
    coefficients: Sequence[float], x1: float, x2: float
) -> float:
    """Evaluate a Chebyshev-series difference without primitive cancellation."""
    if len(coefficients) <= 1 or x1 == x2:
        return 0.0
    delta = x2 - x1
    t_previous = 1.0
    t_current = x2
    q_previous = 0.0
    q_current = 1.0
    total = float(coefficients[1])
    for degree in range(2, len(coefficients)):
        t_next = 2.0 * x2 * t_current - t_previous
        q_next = 2.0 * t_current + 2.0 * x1 * q_current - q_previous
        total += float(coefficients[degree]) * q_next
        t_previous, t_current = t_current, t_next
        q_previous, q_current = q_current, q_next
    return delta * total


class LiquidCpKernel(IdealGasCpKernel):
    """Common liquid range conditioning and executable-curve contract."""

    def _condition_temperature(self, T: float) -> tuple[float, float, str]:
        temperature = _finite(T)
        if temperature <= 0.0:
            raise ValueError("liquid heat-capacity temperature must be positive")
        if self.Tmin <= temperature <= self.Tmax:
            return temperature, 0.0, ""

        if temperature < self.Tmin:
            distance = self.Tmin - temperature
            direction = "below"
            evaluation_T = max(temperature, self.extended_Tmin)
        else:
            distance = temperature - self.Tmax
            direction = "above"
            evaluation_T = min(temperature, self.extended_Tmax)

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
            f"clamped {direction} extended range after "
            f"{EXTRAPOLATION_WIDTH_K:g} K extrapolation; query is "
            f"{distance:g} K outside fitted range",
        )


@dataclass(frozen=True)
class ConstantLiquidCpKernel(LiquidCpKernel):
    value: float = 0.0
    unbounded: bool = False
    kind = "constant_liquid"

    def __post_init__(self) -> None:
        super().__post_init__()
        value = _finite(self.value)
        if value <= 0.0:
            raise ValueError("constant liquid Cp must be positive")
        object.__setattr__(self, "value", value)

    def _condition_temperature(self, T: float) -> tuple[float, float, str]:
        if self.unbounded:
            temperature = _finite(T)
            if temperature <= 0.0:
                raise ValueError("liquid heat-capacity temperature must be positive")
            return temperature, 0.0, ""
        return super()._condition_temperature(T)

    def covers(self, T: float) -> bool:
        return float(T) > 0.0 if self.unbounded else super().covers(T)

    def covers_interval(self, T1: float, T2: float) -> bool:
        if self.unbounded:
            return float(T1) > 0.0 and float(T2) > 0.0
        return super().covers_interval(T1, T2)

    def _integrate_conditioned(self, T1: float, T2: float, *, entropy: bool) -> float:
        if not self.unbounded:
            return super()._integrate_conditioned(T1, T2, entropy=entropy)
        first = _finite(T1)
        second = _finite(T2)
        if first <= 0.0 or second <= 0.0:
            raise ValueError("heat-capacity integration temperatures must be positive")
        return self.value * (
            math.log(second / first) if entropy else second - first
        )

    def _cp_native(self, T: float) -> float:
        return self.value

    def _delta_h_native(self, T1: float, T2: float) -> float:
        return self.value * (T2 - T1)

    def _delta_s_native(self, T1: float, T2: float) -> float:
        return self.value * math.log(T2 / T1)

    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "contract_version": KERNEL_CONTRACT_VERSION,
            "value": self.value,
            "unbounded": self.unbounded,
        }


@dataclass(frozen=True)
class PolynomialLiquidCpKernel(LiquidCpKernel):
    coefficients: tuple[float, ...] = ()
    kind = "polynomial_liquid"

    def __post_init__(self) -> None:
        super().__post_init__()
        coefficients = tuple(_finite(value) for value in self.coefficients)
        if not coefficients:
            raise ValueError("polynomial liquid Cp requires coefficients")
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
        return {
            **super().to_payload(),
            "contract_version": KERNEL_CONTRACT_VERSION,
            "coefficients": list(self.coefficients),
        }


@dataclass(frozen=True)
class ShomateLiquidCpKernel(LiquidCpKernel):
    coefficients: tuple[float, ...] = ()
    kind = "shomate_liquid"

    def __post_init__(self) -> None:
        super().__post_init__()
        coefficients = tuple(_finite(value) for value in self.coefficients)
        if len(coefficients) != 5:
            raise ValueError("Shomate liquid Cp requires five coefficients")
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
        return {
            **super().to_payload(),
            "contract_version": KERNEL_CONTRACT_VERSION,
            "coefficients": list(self.coefficients),
        }


@dataclass(frozen=True)
class LinearChebyshevLiquidCpKernel(LiquidCpKernel):
    center: float = 0.0
    half_width: float = 0.0
    coefficients: tuple[float, ...] = ()
    h_coefficients: tuple[float, ...] = ()
    s_coefficients: tuple[float, ...] = ()
    kind = "linear_chebyshev_liquid"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "center", _finite(self.center))
        object.__setattr__(self, "half_width", _finite(self.half_width))
        object.__setattr__(self, "coefficients", tuple(_finite(v) for v in self.coefficients))
        object.__setattr__(self, "h_coefficients", tuple(_finite(v) for v in self.h_coefficients))
        object.__setattr__(self, "s_coefficients", tuple(_finite(v) for v in self.s_coefficients))
        if len(self.coefficients) not in {9, 13, 17}:
            raise ValueError("linear liquid Chebyshev kernel requires degree 8, 12, or 16")
        if self.half_width <= 0.0:
            raise ValueError("linear liquid Chebyshev mapping requires positive half-width")
        if not self.h_coefficients or not self.s_coefficients:
            raise ValueError("linear liquid Chebyshev kernel requires H and S primitives")
        self._validate_curve()

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def _map(self, T: float) -> float:
        return (T - self.center) / self.half_width

    def _cp_native(self, T: float) -> float:
        return _chebyshev_value(self.coefficients, self._map(T))

    def _delta_h_native(self, T1: float, T2: float) -> float:
        return _chebyshev_difference(
            self.h_coefficients, self._map(T1), self._map(T2)
        )

    def _delta_s_native(self, T1: float, T2: float) -> float:
        return _chebyshev_difference(
            self.s_coefficients, self._map(T1), self._map(T2)
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "contract_version": KERNEL_CONTRACT_VERSION,
            "center_K": self.center,
            "half_width_K": self.half_width,
            "degree": self.degree,
            "coefficients": list(self.coefficients),
            "h_coefficients": list(self.h_coefficients),
            "s_coefficients": list(self.s_coefficients),
        }


@dataclass(frozen=True)
class NativeZabranskyLiquidCpKernel(LiquidCpKernel):
    critical_temperature: float = 0.0
    coefficients: tuple[float, ...] = ()
    kind = "native_zabransky_liquid"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "critical_temperature", _finite(self.critical_temperature))
        object.__setattr__(self, "coefficients", tuple(_finite(v) for v in self.coefficients))
        if len(self.coefficients) != 6:
            raise ValueError("native Zabransky liquid Cp requires six coefficients")
        if self.critical_temperature <= self.Tmax:
            raise ValueError("native Zabransky liquid Cp requires Tc above Tmax")
        self._validate_curve()

    def _cp_native(self, T: float) -> float:
        a1, a2, a3, a4, a5, a6 = self.coefficients
        reduced = T / self.critical_temperature
        gap = 1.0 - reduced
        if gap <= 0.0:
            raise ValueError("native Zabransky liquid Cp is undefined at or above Tc")
        return R * (
            a1 * math.log(gap)
            + a2 / gap
            + a3
            + a4 * reduced
            + a5 * reduced * reduced
            + a6 * reduced**3
        )

    def _delta_h_native(self, T1: float, T2: float) -> float:
        if T1 == T2:
            return 0.0
        if T2 < T1:
            return -self._delta_h_native(T2, T1)
        a1, a2, a3, a4, a5, a6 = self.coefficients
        Tc = self.critical_temperature
        delta = T2 - T1
        log_critical_ratio = math.log1p(-delta / (Tc - T1))
        log_reduced_gap_1 = math.log1p(-T1 / Tc)
        polynomial = _polynomial_difference(
            (
                0.0,
                a3 - a1,
                a4 / (2.0 * Tc),
                a5 / (3.0 * Tc * Tc),
                a6 / (4.0 * Tc**3),
            ),
            T1,
            T2,
        )
        logarithmic = (
            a1 * delta * log_reduced_gap_1
            + (a1 * T2 - Tc * (a1 + a2)) * log_critical_ratio
        )
        return R * (polynomial + logarithmic)

    def _delta_s_native(self, T1: float, T2: float) -> float:
        if T1 == T2:
            return 0.0
        if T2 < T1:
            return -self._delta_s_native(T2, T1)
        from scipy.special import spence

        a1, a2, a3, a4, a5, a6 = self.coefficients
        Tc = self.critical_temperature
        delta = T2 - T1
        log_temperature_ratio = math.log1p(delta / T1)
        log_critical_ratio = math.log1p(-delta / (Tc - T1))
        dilogarithm_difference = float(
            spence(1.0 - T2 / Tc) - spence(1.0 - T1 / Tc)
        )
        polynomial = _polynomial_difference(
            (
                0.0,
                a4 / Tc,
                a5 / (2.0 * Tc * Tc),
                a6 / (3.0 * Tc**3),
            ),
            T1,
            T2,
        )
        return R * (
            (a3 + a2) * log_temperature_ratio
            - a1 * dilogarithm_difference
            - a2 * log_critical_ratio
            + polynomial
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "contract_version": KERNEL_CONTRACT_VERSION,
            "critical_temperature_K": self.critical_temperature,
            "coefficients": list(self.coefficients),
        }


@dataclass(frozen=True)
class ScaledIdealGasLiquidCpKernel(LiquidCpKernel):
    ideal_gas_kernel: Optional[IdealGasCpKernel] = None
    scale_factor: float = 1.3
    kind = "scaled_ideal_gas_liquid"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ideal_gas_kernel is None:
            raise ValueError("scaled liquid Cp requires an ideal-gas kernel")
        factor = _finite(self.scale_factor)
        if factor <= 0.0:
            raise ValueError("scaled liquid Cp factor must be positive")
        object.__setattr__(self, "scale_factor", factor)
        self._validate_curve()

    def _cp_native(self, T: float) -> float:
        return self.scale_factor * self.ideal_gas_kernel._cp_native(T)

    def _delta_h_native(self, T1: float, T2: float) -> float:
        return self.scale_factor * self.ideal_gas_kernel._delta_h_native(T1, T2)

    def _delta_s_native(self, T1: float, T2: float) -> float:
        return self.scale_factor * self.ideal_gas_kernel._delta_s_native(T1, T2)

    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "contract_version": KERNEL_CONTRACT_VERSION,
            "scale_factor": self.scale_factor,
            "ideal_gas_kernel": self.ideal_gas_kernel.to_payload(),
        }


def liquid_kernel_from_payload(payload: Mapping[str, Any]) -> LiquidCpKernel:
    if int(payload.get("contract_version", 0)) != KERNEL_CONTRACT_VERSION:
        raise ValueError("unsupported liquid Cp kernel contract")
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
    if kind == ConstantLiquidCpKernel.kind:
        return ConstantLiquidCpKernel(
            **common, value=payload["value"], unbounded=bool(payload.get("unbounded"))
        )
    if kind == PolynomialLiquidCpKernel.kind:
        return PolynomialLiquidCpKernel(
            **common, coefficients=tuple(payload["coefficients"])
        )
    if kind == ShomateLiquidCpKernel.kind:
        return ShomateLiquidCpKernel(
            **common, coefficients=tuple(payload["coefficients"])
        )
    if kind == LinearChebyshevLiquidCpKernel.kind:
        return LinearChebyshevLiquidCpKernel(
            **common,
            center=payload["center_K"],
            half_width=payload["half_width_K"],
            coefficients=tuple(payload["coefficients"]),
            h_coefficients=tuple(payload["h_coefficients"]),
            s_coefficients=tuple(payload["s_coefficients"]),
        )
    if kind == NativeZabranskyLiquidCpKernel.kind:
        return NativeZabranskyLiquidCpKernel(
            **common,
            critical_temperature=payload["critical_temperature_K"],
            coefficients=tuple(payload["coefficients"]),
        )
    if kind == ScaledIdealGasLiquidCpKernel.kind:
        from .ideal_gas_cp import kernel_from_payload

        return ScaledIdealGasLiquidCpKernel(
            **common,
            scale_factor=payload.get("scale_factor", 1.3),
            ideal_gas_kernel=kernel_from_payload(payload["ideal_gas_kernel"]),
        )
    raise ValueError(f"unsupported liquid Cp kernel kind {kind!r}")


def fit_linear_chebyshev_liquid_kernel(
    evaluator: Callable[[Any], Any],
    Tmin: float,
    Tmax: float,
    *,
    quality: float,
    source: str,
    method: str,
    notes: str = "",
    source_fingerprint: str = "",
) -> LinearChebyshevLiquidCpKernel:
    """Fit a bounded liquid source without sacrificing its low-temperature end."""
    import numpy as np
    from numpy.polynomial import chebyshev as ncheb

    Tmin = _finite(Tmin)
    Tmax = _finite(Tmax)
    if Tmin <= 0.0 or Tmax <= Tmin:
        raise ValueError("invalid liquid Chebyshev fitting range")
    center = 0.5 * (Tmin + Tmax)
    half_width = 0.5 * (Tmax - Tmin)
    training_T = np.linspace(Tmin, Tmax, 257)
    values = np.asarray(evaluator(training_T), dtype=float)
    if values.shape != training_T.shape:
        values = np.asarray([evaluator(float(T)) for T in training_T], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("cannot fit nonpositive or nonfinite liquid heat capacity")

    validation_T = np.linspace(Tmin, Tmax, 1001)
    reference = np.asarray(evaluator(validation_T), dtype=float)
    if reference.shape != validation_T.shape:
        reference = np.asarray([evaluator(float(T)) for T in validation_T], dtype=float)
    if np.any(~np.isfinite(reference)) or np.any(reference <= 0.0):
        raise ValueError("invalid liquid heat-capacity validation source")

    x_training = (training_T - center) / half_width
    x_validation = (validation_T - center) / half_width
    best = None
    for degree in (8, 12, 16):
        vandermonde = ncheb.chebvander(x_training, degree)
        weighted = vandermonde / values[:, None]
        column_scale = np.linalg.norm(weighted, axis=0)
        if np.any(column_scale == 0.0):
            continue
        scaled, *_ = np.linalg.lstsq(
            weighted / column_scale, np.ones(len(values)), rcond=None
        )
        coefficients = scaled / column_scale
        predicted = ncheb.chebval(x_validation, coefficients)
        if np.any(~np.isfinite(predicted)) or np.any(predicted <= 0.0):
            continue
        errors = np.abs(predicted / reference - 1.0) * 100.0
        mape = float(np.mean(errors))
        maximum = float(np.max(errors))

        h_coefficients = ncheb.chebint(coefficients) * half_width
        span_extra_degree = 4 * max(0, math.ceil(math.log2(Tmax / Tmin)))
        entropy_degree = degree + 16 + span_extra_degree
        nodes = np.cos(
            np.pi * (np.arange(entropy_degree + 1) + 0.5) / (entropy_degree + 1)
        )
        temperatures = center + half_width * nodes
        entropy_values = ncheb.chebval(nodes, coefficients) / temperatures
        entropy_integrand = ncheb.chebfit(nodes, entropy_values, entropy_degree)
        s_coefficients = ncheb.chebint(entropy_integrand) * half_width
        candidate = (maximum, mape, degree, coefficients, h_coefficients, s_coefficients)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
        if maximum <= 0.1 + 1.0e-10:
            break
    if best is None:
        raise ValueError("liquid Chebyshev fitting failed")
    maximum, mape, degree, coefficients, h_coefficients, s_coefficients = best
    penalty = fit_quality_penalty(maximum, mape)
    return LinearChebyshevLiquidCpKernel(
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
        half_width=half_width,
        coefficients=tuple(float(value) for value in coefficients),
        h_coefficients=tuple(float(value) for value in h_coefficients),
        s_coefficients=tuple(float(value) for value in s_coefficients),
    )


_BUNDLED_CACHE: dict[tuple[Path, str], Optional[LiquidCpKernel]] = {}
_BUNDLED_LOCK = threading.Lock()
_BUNDLED_IDENTITY_CACHE: dict[tuple[Path, str], Optional[str]] = {}


def load_bundled_liquid_kernel(
    cas: str, *, path: Path = CANONICAL_DATABASE_PATH
) -> Optional[LiquidCpKernel]:
    """Load and process-cache one canonical ordinary-liquid Cp record."""
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
                "SELECT * FROM canonical_liquid_cp WHERE cas = ?", (key,)
            ).fetchone()
    except (OSError, sqlite3.Error):
        return None

    kernel = None
    if row is not None:
        try:
            common = dict(
                Tmin=row["Tmin_fit_K"],
                Tmax=row["Tmax_fit_K"],
                quality=row["quality"],
                source=row["source_label"],
                method=f"canonical_{row['source']}_liquid_cp",
                notes=(
                    "Bundled canonical ordinary-liquid Cp; source range "
                    f"{row['Tmin_source_K']:g}-{row['Tmax_source_K']:g} K; "
                    f"fit MAPE {row['fit_mape_percent']:.4g}%, max error "
                    f"{row['fit_max_error_percent']:.4g}%"
                ),
                fit_mape_percent=row["fit_mape_percent"],
                fit_max_error_percent=row["fit_max_error_percent"],
                source_fingerprint=row["source_fingerprint"],
            )
            if row["model"] == "linear_chebyshev_liquid_cp_v1":
                kernel = LinearChebyshevLiquidCpKernel(
                    **common,
                    center=row["map_center_K"],
                    half_width=row["map_scale"],
                    coefficients=tuple(json.loads(row["cp_coefficients_json"])),
                    h_coefficients=tuple(json.loads(row["h_polynomial_json"])),
                    s_coefficients=tuple(json.loads(row["s_polynomial_json"])),
                )
            elif row["model"] == "native_zabransky_quasipolynomial_v1":
                kernel = NativeZabranskyLiquidCpKernel(
                    **common,
                    critical_temperature=row["critical_temperature_K"],
                    coefficients=tuple(json.loads(row["cp_coefficients_json"])),
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            kernel = None
    with _BUNDLED_LOCK:
        _BUNDLED_CACHE[cache_key] = kernel
    return kernel


def lookup_bundled_liquid_cas(
    identifier: str, *, path: Path = CANONICAL_DATABASE_PATH
) -> Optional[str]:
    """Resolve one unambiguous exact bundled name or formula without Perry."""
    text = str(identifier or '').strip()
    if not text or not path.is_file():
        return None
    try:
        normalized_path = path.resolve()
    except OSError:
        normalized_path = path.absolute()
    cache_key = (normalized_path, text.casefold())
    with _BUNDLED_LOCK:
        if cache_key in _BUNDLED_IDENTITY_CACHE:
            return _BUNDLED_IDENTITY_CACHE[cache_key]
    result = None
    try:
        uri = f"file:{path.resolve()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as connection:
            rows = connection.execute(
                """
                SELECT cas, name, formula
                FROM canonical_liquid_cp
                WHERE name = ? COLLATE NOCASE OR formula = ? COLLATE NOCASE
                """,
                (text, text),
            ).fetchall()
        name_matches = {
            str(row[0]) for row in rows
            if str(row[1] or '').strip().casefold() == text.casefold()
        }
        formula_matches = {
            str(row[0]) for row in rows
            if str(row[2] or '').strip().casefold() == text.casefold()
        }
        if len(name_matches) == 1:
            result = next(iter(name_matches))
        elif len(formula_matches) == 1:
            result = next(iter(formula_matches))
    except (OSError, sqlite3.Error):
        result = None
    with _BUNDLED_LOCK:
        _BUNDLED_IDENTITY_CACHE[cache_key] = result
    return result


def clear_bundled_liquid_kernel_cache() -> None:
    with _BUNDLED_LOCK:
        _BUNDLED_CACHE.clear()
        _BUNDLED_IDENTITY_CACHE.clear()
