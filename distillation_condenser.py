"""Temperature boundary for a rigorous, fully condensed overhead.

Subcooling is measured below incipient-vapor saturation of the *whole*
condensate at condenser pressure, before withdrawal. For VLLE this reference
allows the liquids to re-equilibrate at the reference temperature.
"""

import math
from functools import lru_cache

from scipy.optimize import brentq

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .distillation_specifications import total_condenser_specification
    from .unit_operations_base import UnitOperationError
else:
    from distillation_specifications import total_condenser_specification
    from unit_operations_base import UnitOperationError


class TotalCondenserBoundary:
    """Shared VLE/VLLE boundary; no extra unknowns or nested roots in residuals."""

    def __init__(self, unit, components, pressure, T_min, T_max, *, allow_lle=False):
        self.unit = unit
        self.thermo = unit.thermo
        self.components = tuple(components)
        self.pressure = float(pressure)
        self.T_min, self.T_max = float(T_min), float(T_max)
        self.span = self.T_max-self.T_min
        self.allow_lle = allow_lle
        self._cached_vapor_total = lru_cache(maxsize=512)(self._calculate_vapor_total)
        try:
            self.options = total_condenser_specification(unit.params)
        except ValueError as error:
            raise UnitOperationError(f"{unit.unit_id}: {error}") from error
        if self.options and self.options['temperature_K'] is not None:
            if not self.T_min < self.options['temperature_K'] < self.T_max:
                raise UnitOperationError(
                    f"{unit.unit_id}: condenser_temperature lies outside column temperature bounds "
                    f"({self.T_min:g}, {self.T_max:g}) K"
                )

    def _calculate_vapor_total(self, T, amounts):
        x = dict(zip(self.components, amounts))
        if self.allow_lle:
            split, x1, x2, beta = self.thermo.liquid_liquid_equilibrium(
                x, T, max_iter=100, tol=1e-10,
            )
            if split and 0 < beta < 1:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from .equilibrium_stage_vlle import shared_vlle_vapor_terms
                else:
                    from equilibrium_stage_vlle import shared_vlle_vapor_terms
                terms = shared_vlle_vapor_terms(
                    self.thermo, T, self.pressure, x1, x2, self.components,
                    self.thermo.activity_coefficients(T, x1),
                    self.thermo.activity_coefficients(T, x2),
                )
                return float(sum(terms.values()))
        K = self.thermo.K_values(T, self.pressure, x)
        return float(sum(x[c]*K[c] for c in self.components))

    def vapor_total(self, T, composition):
        amounts = tuple(float(composition[c]) for c in self.components)
        return self._cached_vapor_total(float(T), amounts)

    def residual(self, T, composition, saturated_residual):
        if self.options is None:
            return saturated_residual
        target = self.options['temperature_K']
        if target is not None:
            return (T-target)/self.span
        # Evaluate vapor onset at T + delta, re-equilibrating reference liquids.
        # This avoids a saturation-temperature root solve inside every residual.
        return self.vapor_total(T+self.options['subcooling_K'], composition)-1.

    def saturation_temperature(self, composition, guess):
        def objective(T):
            return self.vapor_total(T, composition)-1.
        width = 2.
        for _ in range(12):
            lo, hi = max(self.T_min, guess-width), min(self.T_max, guess+width)
            if objective(lo)*objective(hi) <= 0:
                return float(brentq(objective, lo, hi, xtol=1e-9, rtol=1e-12))
            if lo == self.T_min and hi == self.T_max:
                break
            width *= 2.
        raise UnitOperationError(
            f"{self.unit.unit_id}: cannot find condensate saturation within column temperature bounds"
        )

    def seed_temperature(self, composition, saturated_guess):
        if self.options is None:
            return saturated_guess
        if self.options['temperature_K'] is not None:
            return self.options['temperature_K']
        target = self.saturation_temperature(composition, saturated_guess)-self.options['subcooling_K']
        if not self.T_min < target < self.T_max:
            raise UnitOperationError(
                f"{self.unit.unit_id}: condenser_subcooling puts the condenser outside column temperature bounds"
            )
        return target

    def diagnostics(self, T, composition, tolerance):
        if self.options is None:
            return {}
        vapor_total = self.vapor_total(T, composition)
        if not math.isfinite(vapor_total) or vapor_total > 1.+max(10*tolerance, 1e-7):
            raise UnitOperationError(
                f"{self.unit.unit_id}: condenser temperature is above condensate saturation; "
                "a total condenser must produce fully liquid condensate"
            )
        delta = self.options['subcooling_K']
        saturation = (T+delta if delta is not None
                      else self.saturation_temperature(composition, T))
        return {'condenser_temperature_K':float(T),
                'condenser_saturation_temperature_K':float(saturation),
                'condenser_subcooling_K':float(saturation-T),
                'condenser_vapor_saturation_ratio':float(vapor_total)}
