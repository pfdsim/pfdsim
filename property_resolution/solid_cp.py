"""Portable solid heat-capacity kernels and canonical source lookup.

Solid heat capacity differs from the gas and ordinary-liquid contracts in one
important respect: a CAS number can have several solid forms and a temperature
curve can cross genuine solid-solid transitions.  This module therefore keeps
native source segments and explicit transition boundaries instead of fitting
one smooth curve across the complete temperature range.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .ideal_gas_cp import (
    CLAMP_QUALITY_PENALTY_PER_5K,
    EXTRAPOLATION_QUALITY_PENALTY,
    EXTRAPOLATION_WIDTH_K,
    MAX_RANGE_QUALITY_PENALTY,
    IdealGasCpKernel,
    KernelEvaluation,
    _finite,
)
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..physical_constants import R_J_MOL_K
else:
    from physical_constants import R_J_MOL_K

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..solid_material_forms import normalize_solid_material_form
else:
    from solid_material_forms import normalize_solid_material_form


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DATABASE_PATH = ROOT / "data" / "solid_heat_capacity.sqlite"
KERNEL_CONTRACT_VERSION = 1
SOLID_MINIMUM_TEMPERATURE_K = 100.0
SOLID_EXTRAPOLATION_WIDTH_K = EXTRAPOLATION_WIDTH_K
SOLID_EXTRAPOLATION_QUALITY_PENALTY = EXTRAPOLATION_QUALITY_PENALTY
SOLID_CLAMP_QUALITY_PENALTY_PER_5K = CLAMP_QUALITY_PENALTY_PER_5K
SOLID_MAX_RANGE_QUALITY_PENALTY = MAX_RANGE_QUALITY_PENALTY
REFERENCE_TEMPERATURE_K = 298.15

MODIFIED_KOPP_CONTRIBUTIONS_J_MOL_K = {
    "C": 10.89, "H": 7.56, "O": 13.42, "N": 18.74, "S": 12.36,
    "F": 26.16, "Cl": 24.69, "Br": 25.36, "I": 25.29,
    "Al": 18.07, "B": 10.10, "Ba": 32.37, "Be": 12.47,
    "Ca": 28.25, "Co": 25.71, "Cu": 26.92, "Fe": 29.08,
    "Hg": 27.87, "K": 28.78, "Li": 23.25, "Mg": 22.69,
    "Mn": 28.06, "Mo": 29.44, "Na": 26.19, "Ni": 25.46,
    "Pb": 31.60, "Si": 17.00, "Sr": 28.41, "Ti": 27.24,
    "V": 29.36, "W": 30.87, "Zr": 26.82,
}
MODIFIED_KOPP_OTHER_J_MOL_K = 26.63


class SolidCpTransitionError(ValueError):
    """Raised when a sensible-heat integral crosses an unresolved transition."""


@dataclass(frozen=True)
class SolidCpKernel(IdealGasCpKernel):
    """Solid-Cp contract with gas-style conditioning and guarded cryogenics."""

    material_form: str = "unspecified"
    polymorph: str = ""
    source_priority: int = 100

    kind = "solid_base"

    def __post_init__(self) -> None:
        super().__post_init__()
        form = normalize_solid_material_form(self.material_form)
        object.__setattr__(self, "material_form", form)
        object.__setattr__(self, "polymorph", str(self.polymorph or "").strip())
        object.__setattr__(self, "source_priority", int(self.source_priority))

    @property
    def transition_temperatures(self) -> tuple[float, ...]:
        return ()

    def active_kernel(self, T: float) -> "SolidCpKernel":
        return self

    def source_at(self, T: float) -> str:
        return self.active_kernel(T).source

    def method_at(self, T: float) -> str:
        return self.active_kernel(T).method

    def notes_at(self, T: float) -> str:
        return self.active_kernel(T).notes

    def _condition_temperature(self, T: float) -> tuple[float, float, str]:
        temperature = _finite(T)
        if (
            temperature < SOLID_MINIMUM_TEMPERATURE_K
            and not self.covers(temperature)
        ):
            raise ValueError(
                f"solid heat-capacity below {SOLID_MINIMUM_TEMPERATURE_K:g} K "
                "requires explicit correlation coverage"
            )
        return super()._condition_temperature(temperature)

    def _integrate_conditioned(
        self,
        T1: float,
        T2: float,
        *,
        entropy: bool,
    ) -> float:
        lower = min(float(T1), float(T2))
        upper = max(float(T1), float(T2))
        cryogenic_upper = min(upper, SOLID_MINIMUM_TEMPERATURE_K)
        if (
            lower < SOLID_MINIMUM_TEMPERATURE_K
            and not (self.Tmin <= lower and cryogenic_upper <= self.Tmax)
        ):
            raise ValueError(
                f"solid heat-capacity integration below "
                f"{SOLID_MINIMUM_TEMPERATURE_K:g} K requires explicit "
                "correlation coverage"
            )
        return super()._integrate_conditioned(T1, T2, entropy=entropy)

    def _validate_curve(self, points: int = 65) -> None:
        if self.Tmax == self.Tmin:
            temperatures = (self.Tmin,)
        else:
            temperatures = tuple(
                self.Tmin + (self.Tmax - self.Tmin) * index / (points - 1)
                for index in range(points)
            )
        for temperature in temperatures:
            value = float(self._cp_native(temperature))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "solid Cp kernel is nonpositive or nonfinite within its source range"
                )

    def _assert_no_transition(self, T1: float, T2: float) -> None:
        low, high = sorted((float(T1), float(T2)))
        crossed = [T for T in self.transition_temperatures if low < T < high]
        if crossed:
            values = ", ".join(f"{T:g} K" for T in crossed)
            raise SolidCpTransitionError(
                "solid sensible-heat integral crosses unresolved solid transition(s) "
                f"at {values}; transition enthalpy is required"
            )

    def delta_h(self, T1: float, T2: float) -> float:
        self._assert_no_transition(T1, T2)
        return super().delta_h(T1, T2)

    def delta_s(self, T1: float, T2: float) -> float:
        self._assert_no_transition(T1, T2)
        return super().delta_s(T1, T2)

    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "contract_version": KERNEL_CONTRACT_VERSION,
            "material_form": self.material_form,
            "polymorph": self.polymorph,
            "source_priority": self.source_priority,
        }


@dataclass(frozen=True)
class ConstantSolidCpKernel(SolidCpKernel):
    value: float = 0.0
    unbounded: bool = False
    kind = "constant_solid"

    def __post_init__(self) -> None:
        super().__post_init__()
        value = _finite(self.value)
        if value <= 0.0:
            raise ValueError("constant solid Cp must be positive")
        object.__setattr__(self, "value", value)

    def _condition_temperature(self, T: float) -> tuple[float, float, str]:
        if self.unbounded:
            temperature = _finite(T)
            if temperature <= 0.0:
                raise ValueError(
                    "solid heat-capacity temperature must be positive"
                )
            return temperature, 0.0, ""
        return super()._condition_temperature(T)

    def covers(self, T: float) -> bool:
        return (
            float(T) > 0.0
            if self.unbounded else super().covers(T)
        )

    def covers_interval(self, T1: float, T2: float) -> bool:
        if self.unbounded:
            return min(float(T1), float(T2)) > 0.0
        return super().covers_interval(T1, T2)

    def _integrate_conditioned(self, T1: float, T2: float, *, entropy: bool) -> float:
        if not self.unbounded:
            return super()._integrate_conditioned(T1, T2, entropy=entropy)
        first, second = _finite(T1), _finite(T2)
        if first <= 0.0 or second <= 0.0:
            raise ValueError("solid heat-capacity integration temperatures must be positive")
        return self.value * (math.log(second / first) if entropy else second - first)

    def _cp_native(self, T: float) -> float:
        return self.value

    def _delta_h_native(self, T1: float, T2: float) -> float:
        return self.value * (T2 - T1)

    def _delta_s_native(self, T1: float, T2: float) -> float:
        return self.value * math.log(T2 / T1)

    def to_payload(self) -> dict[str, Any]:
        return {**super().to_payload(), "value": self.value, "unbounded": self.unbounded}


@dataclass(frozen=True)
class PolynomialSolidCpKernel(SolidCpKernel):
    coefficients: tuple[float, ...] = ()
    kind = "polynomial_solid"

    def __post_init__(self) -> None:
        super().__post_init__()
        coefficients = tuple(_finite(value) for value in self.coefficients)
        if not coefficients:
            raise ValueError("polynomial solid Cp requires coefficients")
        object.__setattr__(self, "coefficients", coefficients)
        self._validate_curve()

    def _cp_native(self, T: float) -> float:
        value = 0.0
        for coefficient in reversed(self.coefficients):
            value = value * T + coefficient
        return value

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
class ShomateSolidCpKernel(SolidCpKernel):
    coefficients: tuple[float, ...] = ()
    kind = "shomate_solid"

    def __post_init__(self) -> None:
        super().__post_init__()
        coefficients = tuple(_finite(value) for value in self.coefficients)
        if len(coefficients) != 5:
            raise ValueError("Shomate solid Cp requires five coefficients")
        object.__setattr__(self, "coefficients", coefficients)
        self._validate_curve()

    def _cp_native(self, T: float) -> float:
        t = T / 1000.0
        A, B, C, D, E = self.coefficients
        return A + t * (B + t * (C + t * D)) + E / (t * t)

    def _delta_h_native(self, T1: float, T2: float) -> float:
        t1, t2 = T1 / 1000.0, T2 / 1000.0
        A, B, C, D, E = self.coefficients
        return 1000.0 * (
            A * (t2 - t1)
            + B * (t2 * t2 - t1 * t1) / 2.0
            + C * (t2**3 - t1**3) / 3.0
            + D * (t2**4 - t1**4) / 4.0
            - E * (1.0 / t2 - 1.0 / t1)
        )

    def _delta_s_native(self, T1: float, T2: float) -> float:
        t1, t2 = T1 / 1000.0, T2 / 1000.0
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
class Perry151SolidCpKernel(SolidCpKernel):
    coefficients: tuple[float, ...] = ()
    kind = "perry_151_solid"

    def __post_init__(self) -> None:
        super().__post_init__()
        coefficients = tuple(_finite(value) for value in self.coefficients)
        if len(coefficients) != 4:
            raise ValueError("Perry 2-151 solid Cp requires four coefficients")
        object.__setattr__(self, "coefficients", coefficients)
        self._validate_curve()

    def _cp_native(self, T: float) -> float:
        a, b, c, d = self.coefficients
        return 4.184 * (a + b * T + c / (T * T) + d * T * T)

    def _delta_h_native(self, T1: float, T2: float) -> float:
        a, b, c, d = self.coefficients
        return 4.184 * (
            a * (T2 - T1)
            + 0.5 * b * (T2 * T2 - T1 * T1)
            - c * (1.0 / T2 - 1.0 / T1)
            + d * (T2**3 - T1**3) / 3.0
        )

    def _delta_s_native(self, T1: float, T2: float) -> float:
        a, b, c, d = self.coefficients
        return 4.184 * (
            a * math.log(T2 / T1)
            + b * (T2 - T1)
            - 0.5 * c * (1.0 / (T2 * T2) - 1.0 / (T1 * T1))
            + 0.5 * d * (T2 * T2 - T1 * T1)
        )

    def to_payload(self) -> dict[str, Any]:
        return {**super().to_payload(), "coefficients": list(self.coefficients)}


@dataclass(frozen=True)
class TabularSolidCpKernel(SolidCpKernel):
    temperatures: tuple[float, ...] = ()
    values: tuple[float, ...] = ()
    kind = "tabular_linear_solid"

    def __post_init__(self) -> None:
        super().__post_init__()
        temperatures = tuple(_finite(value) for value in self.temperatures)
        values = tuple(_finite(value) for value in self.values)
        if len(temperatures) != len(values) or len(values) < 2:
            raise ValueError("tabular solid Cp requires at least two paired points")
        if any(b <= a for a, b in zip(temperatures, temperatures[1:])):
            raise ValueError("tabular solid Cp temperatures must be strictly increasing")
        if abs(temperatures[0] - self.Tmin) > 1.0e-8 or abs(temperatures[-1] - self.Tmax) > 1.0e-8:
            raise ValueError("tabular solid Cp range must match its endpoint temperatures")
        if any(value <= 0.0 for value in values):
            raise ValueError("tabular solid Cp values must be positive")
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "values", values)

    def _interval(self, T: float) -> int:
        index = bisect.bisect_right(self.temperatures, T) - 1
        return max(0, min(index, len(self.temperatures) - 2))

    def _line(self, index: int) -> tuple[float, float]:
        T1, T2 = self.temperatures[index:index + 2]
        C1, C2 = self.values[index:index + 2]
        slope = (C2 - C1) / (T2 - T1)
        return C1 - slope * T1, slope

    def _cp_native(self, T: float) -> float:
        intercept, slope = self._line(self._interval(T))
        return intercept + slope * T

    def _integral(self, T1: float, T2: float, entropy: bool) -> float:
        if T1 == T2:
            return 0.0
        if T2 < T1:
            return -self._integral(T2, T1, entropy)
        cuts = [T1]
        cuts.extend(T for T in self.temperatures[1:-1] if T1 < T < T2)
        cuts.append(T2)
        total = 0.0
        for left, right in zip(cuts, cuts[1:]):
            intercept, slope = self._line(self._interval(0.5 * (left + right)))
            if entropy:
                total += intercept * math.log(right / left) + slope * (right - left)
            else:
                total += intercept * (right - left) + 0.5 * slope * (right**2 - left**2)
        return total

    def _delta_h_native(self, T1: float, T2: float) -> float:
        return self._integral(T1, T2, False)

    def _delta_s_native(self, T1: float, T2: float) -> float:
        return self._integral(T1, T2, True)

    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "temperatures_K": list(self.temperatures),
            "values_J_mol_K": list(self.values),
        }


@dataclass(frozen=True)
class PiecewiseSolidCpKernel(SolidCpKernel):
    segments: tuple[SolidCpKernel, ...] = ()
    transitions: tuple[float, ...] = ()
    kind = "piecewise_solid"

    def __post_init__(self) -> None:
        super().__post_init__()
        segments = tuple(sorted(self.segments, key=lambda item: (item.Tmin, item.Tmax)))
        if not segments:
            raise ValueError("piecewise solid Cp requires at least one segment")
        transitions = tuple(sorted({_finite(value) for value in self.transitions}))
        if abs(segments[0].Tmin - self.Tmin) > 1.0e-8 or abs(segments[-1].Tmax - self.Tmax) > 1.0e-8:
            raise ValueError("piecewise solid Cp range must match its segment envelope")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "transitions", transitions)

    @property
    def transition_temperatures(self) -> tuple[float, ...]:
        nested = {value for segment in self.segments for value in segment.transition_temperatures}
        nested.update(self.transitions)
        return tuple(sorted(nested))

    def active_kernel(self, T: float) -> SolidCpKernel:
        temperature = float(T)
        exact = [segment for segment in self.segments if segment.covers(temperature)]
        if exact:
            return exact[0].active_kernel(temperature)
        if temperature < self.Tmin:
            return self.segments[0].active_kernel(temperature)
        if temperature > self.Tmax:
            return self.segments[-1].active_kernel(temperature)
        raise ValueError(f"no solid Cp segment covers T={temperature:g} K")

    def _cp_native(self, T: float) -> float:
        return self.active_kernel(T)._cp_native(T)

    def _integral(self, T1: float, T2: float, entropy: bool) -> float:
        if T1 == T2:
            return 0.0
        if T2 < T1:
            return -self._integral(T2, T1, entropy)
        self._assert_no_transition(T1, T2)
        cuts = [T1]
        cuts.extend(
            endpoint
            for segment in self.segments
            for endpoint in (segment.Tmin, segment.Tmax)
            if T1 < endpoint < T2
        )
        cuts.append(T2)
        cuts = sorted(set(cuts))
        total = 0.0
        for left, right in zip(cuts, cuts[1:]):
            midpoint = 0.5 * (left + right)
            segment = self.active_kernel(midpoint)
            total += (
                segment.delta_s(left, right)
                if entropy else segment.delta_h(left, right)
            )
        return total

    def _delta_h_native(self, T1: float, T2: float) -> float:
        return self._integral(T1, T2, False)

    def _delta_s_native(self, T1: float, T2: float) -> float:
        return self._integral(T1, T2, True)

    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "segments": [segment.to_payload() for segment in self.segments],
            "transition_temperatures_K": list(self.transitions),
        }


@dataclass(frozen=True)
class LastovkaSolidCpKernel(SolidCpKernel):
    similarity_variable: float = 0.0
    molecular_weight: float = 0.0
    kind = "lastovka_solid"

    def __post_init__(self) -> None:
        super().__post_init__()
        alpha = _finite(self.similarity_variable)
        molecular_weight = _finite(self.molecular_weight)
        if alpha <= 0.0 or molecular_weight <= 0.0:
            raise ValueError("Lastovka solid Cp requires positive alpha and MW")
        object.__setattr__(self, "similarity_variable", alpha)
        object.__setattr__(self, "molecular_weight", molecular_weight)
        self._validate_curve()

    def _cp_native(self, T: float) -> float:
        A1, A2, theta = 0.013183, 0.249381, 151.8675
        C1, C2, D1, D2 = 0.026526, -0.024942, 0.000025, -0.000123
        alpha = self.similarity_variable
        ratio = theta / T
        exponential = math.exp(ratio)
        mass_cp = alpha * (
            3.0 * (A1 + A2 * alpha) * R_J_MOL_K * ratio * ratio
            * exponential / ((exponential - 1.0) ** 2)
            + (C1 + C2 * alpha) * T
            + (D1 + D2 * alpha) * T * T
        )
        return mass_cp * self.molecular_weight

    def _h_primitive(self, T: float) -> float:
        A1, A2, theta = 0.013183, 0.249381, 151.8675
        C1, C2, D1, D2 = 0.026526, -0.024942, 0.000025, -0.000123
        alpha = self.similarity_variable
        value = alpha * (
            T**3 * (D1 + D2 * alpha) / 3.0
            + 0.5 * T * T * (C1 + C2 * alpha)
            + 3.0 * R_J_MOL_K * theta * (A1 + A2 * alpha)
            / math.expm1(theta / T)
        )
        return value * self.molecular_weight

    def _s_primitive(self, T: float) -> float:
        A1, A2, theta = 0.013183, 0.249381, 151.8675
        C1, C2, D1, D2 = 0.026526, -0.024942, 0.000025, -0.000123
        alpha = self.similarity_variable
        exponential = math.exp(theta / T)
        Aterm = A1 + A2 * alpha
        value = alpha * (
            -3.0 * R_J_MOL_K * Aterm * math.log(exponential - 1.0)
            + 0.5 * T * T * (D1 + D2 * alpha)
            + T * (C1 + C2 * alpha)
            + 3.0 * R_J_MOL_K * theta * Aterm
            * (1.0 / (T * exponential - T) + 1.0 / T)
        )
        return value * self.molecular_weight

    def _delta_h_native(self, T1: float, T2: float) -> float:
        return self._h_primitive(T2) - self._h_primitive(T1)

    def _delta_s_native(self, T1: float, T2: float) -> float:
        return self._s_primitive(T2) - self._s_primitive(T1)

    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "similarity_variable_mol_g": self.similarity_variable,
            "molecular_weight_g_mol": self.molecular_weight,
        }


@dataclass(frozen=True)
class ModifiedKoppSolidCpKernel(ConstantSolidCpKernel):
    atom_counts: tuple[tuple[str, float], ...] = ()
    kind = "modified_kopp_solid"

    def to_payload(self) -> dict[str, Any]:
        return {**super().to_payload(), "atom_counts": dict(self.atom_counts)}


@dataclass(frozen=True)
class SolidCpCollectionKernel(SolidCpKernel):
    candidates: tuple[SolidCpKernel, ...] = ()
    kind = "solid_collection"

    def __post_init__(self) -> None:
        super().__post_init__()
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("solid Cp collection requires candidates")
        object.__setattr__(self, "candidates", candidates)

    @property
    def transition_temperatures(self) -> tuple[float, ...]:
        # JANAF tables and their WebBook Shomate fits can report the same
        # transition a fraction of a kelvin apart.  Preserve the value from
        # the highest-ranked candidate and suppress duplicate declarations.
        accepted: list[float] = []
        for item in sorted(self.candidates, key=self._rank):
            for temperature in item.transition_temperatures:
                if not any(abs(temperature - existing) <= 1.0 for existing in accepted):
                    accepted.append(temperature)
        return tuple(sorted(accepted))

    @staticmethod
    def _rank(item: SolidCpKernel) -> tuple[float, int, str]:
        return (-item.quality, item.source_priority, item.method)

    def active_kernel(self, T: float) -> SolidCpKernel:
        temperature = float(T)
        exact = [item for item in self.candidates if item.covers(T)]
        if exact:
            return min(exact, key=self._rank).active_kernel(T)
        extended = [
            item for item in self.candidates
            if item.extended_Tmin <= temperature <= item.extended_Tmax
        ]
        if extended:
            return min(extended, key=self._rank).active_kernel(T)
        if temperature < self.Tmin:
            boundary = min(item.Tmin for item in self.candidates)
            candidates = [item for item in self.candidates if item.Tmin == boundary]
            return min(candidates, key=self._rank).active_kernel(T)
        if temperature > self.Tmax:
            boundary = max(item.Tmax for item in self.candidates)
            candidates = [item for item in self.candidates if item.Tmax == boundary]
            return min(candidates, key=self._rank).active_kernel(T)
        raise ValueError(f"no solid Cp source covers T={temperature:g} K")

    def evaluate(self, T: float) -> KernelEvaluation:
        return self.active_kernel(T).evaluate(T)

    def cp(self, T: float) -> float:
        return self.active_kernel(T).cp(T)

    def quality_at(self, T: float) -> float:
        return self.active_kernel(T).quality_at(T)

    def _cp_native(self, T: float) -> float:
        return self.active_kernel(T)._cp_native(T)

    def _integral(self, T1: float, T2: float, entropy: bool) -> float:
        if T1 == T2:
            return 0.0
        if T2 < T1:
            return -self._integral(T2, T1, entropy)
        self._assert_no_transition(T1, T2)
        direct = [item for item in self.candidates if item.covers_interval(T1, T2)]
        if direct:
            item = min(direct, key=self._rank)
            return item.delta_s(T1, T2) if entropy else item.delta_h(T1, T2)
        cuts = [T1, T2]
        for item in self.candidates:
            cuts.extend(value for value in (item.Tmin, item.Tmax) if T1 < value < T2)
        cuts = sorted(set(cuts))
        total = 0.0
        for left, right in zip(cuts, cuts[1:]):
            eligible = [item for item in self.candidates if item.covers_interval(left, right)]
            if not eligible:
                midpoint = 0.5 * (left + right)
                if right <= self.Tmin or left >= self.Tmax:
                    item = self.active_kernel(midpoint)
                else:
                    raise ValueError(
                        f"solid Cp sources do not continuously cover {left:g}-{right:g} K"
                    )
            else:
                item = min(eligible, key=self._rank)
            total += item.delta_s(left, right) if entropy else item.delta_h(left, right)
        return total

    def _delta_h_native(self, T1: float, T2: float) -> float:
        return self._integral(T1, T2, False)

    def _delta_s_native(self, T1: float, T2: float) -> float:
        return self._integral(T1, T2, True)

    def delta_h(self, T1: float, T2: float) -> float:
        return self._integral(float(T1), float(T2), False)

    def delta_s(self, T1: float, T2: float) -> float:
        return self._integral(float(T1), float(T2), True)

    def to_payload(self) -> dict[str, Any]:
        return {**super().to_payload(), "candidates": [item.to_payload() for item in self.candidates]}


def solid_kernel_from_payload(payload: Mapping[str, Any]) -> SolidCpKernel:
    if int(payload.get("contract_version", 0)) != KERNEL_CONTRACT_VERSION:
        raise ValueError("unsupported solid Cp kernel contract")
    common = dict(
        Tmin=payload["Tmin_K"], Tmax=payload["Tmax_K"],
        quality=payload["quality"], source=payload["source"],
        method=payload["method"], notes=payload.get("notes", ""),
        fit_mape_percent=payload.get("fit_mape_percent", 0.0),
        fit_max_error_percent=payload.get("fit_max_error_percent", 0.0),
        source_fingerprint=payload.get("source_fingerprint", ""),
        material_form=payload.get("material_form", "unspecified"),
        polymorph=payload.get("polymorph", ""),
        source_priority=payload.get("source_priority", 100),
    )
    kind = str(payload.get("kind", ""))
    if kind == ConstantSolidCpKernel.kind:
        return ConstantSolidCpKernel(**common, value=payload["value"], unbounded=bool(payload.get("unbounded")))
    if kind == ModifiedKoppSolidCpKernel.kind:
        return ModifiedKoppSolidCpKernel(
            **common, value=payload["value"], unbounded=bool(payload.get("unbounded")),
            atom_counts=tuple(sorted((str(k), float(v)) for k, v in (payload.get("atom_counts") or {}).items())),
        )
    if kind == PolynomialSolidCpKernel.kind:
        return PolynomialSolidCpKernel(**common, coefficients=tuple(payload["coefficients"]))
    if kind == ShomateSolidCpKernel.kind:
        return ShomateSolidCpKernel(**common, coefficients=tuple(payload["coefficients"]))
    if kind == Perry151SolidCpKernel.kind:
        return Perry151SolidCpKernel(**common, coefficients=tuple(payload["coefficients"]))
    if kind == TabularSolidCpKernel.kind:
        return TabularSolidCpKernel(
            **common,
            temperatures=tuple(payload["temperatures_K"]),
            values=tuple(payload["values_J_mol_K"]),
        )
    if kind == PiecewiseSolidCpKernel.kind:
        return PiecewiseSolidCpKernel(
            **common,
            segments=tuple(solid_kernel_from_payload(item) for item in payload["segments"]),
            transitions=tuple(payload.get("transition_temperatures_K") or ()),
        )
    if kind == LastovkaSolidCpKernel.kind:
        return LastovkaSolidCpKernel(
            **common,
            similarity_variable=payload["similarity_variable_mol_g"],
            molecular_weight=payload["molecular_weight_g_mol"],
        )
    if kind == SolidCpCollectionKernel.kind:
        return SolidCpCollectionKernel(
            **common,
            candidates=tuple(solid_kernel_from_payload(item) for item in payload["candidates"]),
        )
    raise ValueError(f"unsupported solid Cp kernel kind {kind!r}")


_BUNDLED_CACHE: dict[tuple[Path, str, str, str], Optional[SolidCpKernel]] = {}
_BUNDLED_IDENTITY_CACHE: dict[tuple[Path, str], Optional[str]] = {}
_BUNDLED_LOCK = threading.Lock()


def _form_admitted(row: sqlite3.Row, material_form: str, polymorph: str) -> bool:
    row_form = str(row["material_form"] or "unspecified").lower()
    row_polymorph = str(row["polymorph"] or "").casefold()
    requested = normalize_solid_material_form(material_form)
    requested_polymorph = str(polymorph or "").casefold()
    if requested_polymorph:
        return (
            row_polymorph == requested_polymorph
            and (requested == "unspecified" or row_form in {requested, "unspecified"})
        )
    if requested in {"hydrate", "solvate", "glass", "amorphous"}:
        return row_form == requested and not row_polymorph
    if requested == "crystalline":
        return row_form in {"crystalline", "unspecified"} and not row_polymorph
    return bool(row["is_default_form"])


def load_bundled_solid_kernel(
    cas: str,
    *,
    material_form: str = "unspecified",
    polymorph: str = "",
    path: Path = CANONICAL_DATABASE_PATH,
) -> Optional[SolidCpKernel]:
    key = str(cas or "").strip()
    if not key or not path.is_file():
        return None
    normalized_path = path.resolve()
    cache_key = (normalized_path, key, material_form.casefold(), polymorph.casefold())
    with _BUNDLED_LOCK:
        if cache_key in _BUNDLED_CACHE:
            return _BUNDLED_CACHE[cache_key]
    try:
        uri = f"file:{normalized_path}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM canonical_solid_cp WHERE cas = ? ORDER BY quality DESC, source_priority, record_id",
                (key,),
            ).fetchall()
    except (OSError, sqlite3.Error):
        return None
    candidates = []
    for row in rows:
        if not _form_admitted(row, material_form, polymorph):
            continue
        try:
            candidates.append(solid_kernel_from_payload(json.loads(row["kernel_payload_json"])))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
    kernel: Optional[SolidCpKernel]
    if not candidates:
        kernel = None
    elif len(candidates) == 1:
        kernel = candidates[0]
    else:
        aggregate_fingerprint = hashlib.sha256(
            "|".join(sorted(item.source_fingerprint for item in candidates)).encode()
        ).hexdigest()
        kernel = SolidCpCollectionKernel(
            Tmin=min(item.Tmin for item in candidates),
            Tmax=max(item.Tmax for item in candidates),
            quality=max(item.quality for item in candidates),
            source="bundled canonical solid heat-capacity sources",
            method="canonical_solid_cp_collection",
            notes=f"{len(candidates)} form-compatible canonical candidates",
            source_fingerprint=aggregate_fingerprint,
            material_form=material_form,
            polymorph=polymorph,
            candidates=tuple(candidates),
        )
    with _BUNDLED_LOCK:
        _BUNDLED_CACHE[cache_key] = kernel
    return kernel


def lookup_bundled_solid_cas(identifier: str, *, path: Path = CANONICAL_DATABASE_PATH) -> Optional[str]:
    text = str(identifier or "").strip()
    if not text or not path.is_file():
        return None
    normalized_path = path.resolve()
    cache_key = (normalized_path, text.casefold())
    with _BUNDLED_LOCK:
        if cache_key in _BUNDLED_IDENTITY_CACHE:
            return _BUNDLED_IDENTITY_CACHE[cache_key]
    result = None
    try:
        uri = f"file:{normalized_path}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as connection:
            rows = connection.execute(
                "SELECT DISTINCT cas, name, formula FROM canonical_solid_cp "
                "WHERE name = ? COLLATE NOCASE OR formula = ? COLLATE NOCASE",
                (text, text),
            ).fetchall()
        names = {str(row[0]) for row in rows if str(row[1] or "").casefold() == text.casefold()}
        formulas = {str(row[0]) for row in rows if str(row[2] or "").casefold() == text.casefold()}
        if len(names) == 1:
            result = next(iter(names))
        elif len(formulas) == 1:
            result = next(iter(formulas))
    except (OSError, sqlite3.Error):
        result = None
    with _BUNDLED_LOCK:
        _BUNDLED_IDENTITY_CACHE[cache_key] = result
    return result


def clear_bundled_solid_kernel_cache() -> None:
    with _BUNDLED_LOCK:
        _BUNDLED_CACHE.clear()
        _BUNDLED_IDENTITY_CACHE.clear()
