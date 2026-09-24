"""Reusable executable kernels for pure-component dynamic viscosity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from .common import PropertyResolutionError, PropertyResolutionResult


@dataclass(frozen=True)
class ViscosityKernel:
    """Prepared viscosity resolver for one component and phase."""

    phase: str
    evaluator: Callable[
        [float, Optional[float], Optional[float]],
        PropertyResolutionResult,
    ]

    @staticmethod
    def _state_value(name: str, value: Optional[float], units: str) -> Optional[float]:
        if value is None and name != 'T':
            return None
        try:
            normalized = float(value)
        except (TypeError, ValueError) as error:
            raise PropertyResolutionError(
                f"Viscosity state {name} must be a positive finite value in {units}."
            ) from error
        if normalized <= 0.0 or not math.isfinite(normalized):
            raise PropertyResolutionError(
                f"Viscosity state {name} must be a positive finite value in {units}."
            )
        return normalized

    def evaluate(
        self,
        T: float,
        *,
        P: Optional[float] = None,
        rho_molar: Optional[float] = None,
    ) -> PropertyResolutionResult:
        temperature = self._state_value('T', T, 'K')
        pressure = self._state_value('P', P, 'bar')
        density = self._state_value('rho_molar', rho_molar, 'kmol/m^3')
        return self.evaluator(temperature, pressure, density)

    def viscosity(
        self,
        T: float,
        *,
        P: Optional[float] = None,
        rho_molar: Optional[float] = None,
    ) -> float:
        result = self.evaluate(T, P=P, rho_molar=rho_molar)
        value = float(result.value)
        if value <= 0.0 or not math.isfinite(value):
            raise PropertyResolutionError(
                "Viscosity kernel produced a nonpositive or nonfinite value."
            )
        return value
