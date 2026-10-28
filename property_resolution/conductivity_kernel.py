"""Prepared pure-fluid conductivity evaluation with resolver provenance."""

import math
from dataclasses import dataclass
from typing import Callable

from .common import PropertyResolutionError, PropertyResolutionResult


@dataclass(frozen=True)
class ThermalConductivityKernel:
    phase: str
    evaluator: Callable[[float], PropertyResolutionResult]

    @staticmethod
    def temperature(T):
        try:
            temperature = float(T)
        except (TypeError, ValueError, OverflowError) as error:
            raise PropertyResolutionError('Thermal-conductivity temperature must be a positive finite value in K.') from error
        if not math.isfinite(temperature) or temperature <= 0:
            raise PropertyResolutionError('Thermal-conductivity temperature must be a positive finite value in K.')
        return temperature

    def evaluate(self, T: float) -> PropertyResolutionResult:
        return self.evaluator(self.temperature(T))
