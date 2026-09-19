"""Lyngby excess-Gibbs provider for the SRK/MHV2 mixing rule."""

import math
from pathlib import Path

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..unifac import UNIFACModel
    from ..compiled_unifac import CompiledUNIFACBackend
    from ..lyngby_parameters import parameter_table
else:
    from unifac import UNIFACModel
    from compiled_unifac import CompiledUNIFACBackend
    from lyngby_parameters import parameter_table

from .ge_eos import ExcessGibbsState
from .unifac_models import resolve_component_unifac_groups


class MHV2ExcessGibbs:
    """Fixed-component Lyngby model including Dahl's gas/alcohol extension."""

    def __init__(self, components, props, db, component_cas, groups=None):
        self.components = tuple(components)
        self.unifac = UNIFACModel(str(Path(__file__).resolve().parents[1]
                                     / 'data' / 'mhv2_unifac.json'))
        table = parameter_table(True)
        self.component_groups = {}
        for component in components:
            gas = table['gas_components'].get(component_cas[component])
            if groups and component in groups:
                assignment = groups[component]
            elif gas is not None:
                assignment = {gas: 1}
            else:
                assignment = resolve_component_unifac_groups(
                    component, props[component], db, 'UNIFLBY')
            if not assignment:
                raise ValueError(f'RKSMHV2 has no Lyngby groups for {component}')
            resolved = self.unifac._resolve_groups(assignment)
            if 12 in resolved:  # OH identifies alcohol alkyl groups.
                resolved = {key + 58 if key in (1, 2, 3, 4) else key: value
                            for key, value in resolved.items()}
            self.component_groups[component] = resolved
        mains = {self.unifac.subgroups[k].main_group
                 for assignment in self.component_groups.values() for k in assignment}
        for i in mains:
            for j in mains:
                self.unifac.get_interaction(i, j)  # Fail for unavailable data.
        self.backend = CompiledUNIFACBackend.from_model(
            self.unifac, list(components), self.component_groups)

    def _ln_gamma(self, composition, T):
        if self.backend is not None:
            gamma = self.backend.activity_coefficients(composition, T)
        else:
            gamma = self.unifac.activity_coefficients(
                list(self.component_groups.values()), composition, T)
        return tuple(math.log(value) for value in gamma)

    def excess_gibbs_state(self, composition, T, *, temperature_derivative=True):
        values = self._ln_gamma(composition, T)
        if not temperature_derivative:
            return ExcessGibbsState(values, (0.0,) * len(values))
        # Fourth-order central derivative of the same authoritative gamma
        # implementation; no independently maintained caloric model.
        step = T * 1.0e-4
        mm = self._ln_gamma(composition, T - 2.0 * step)
        m = self._ln_gamma(composition, T - step)
        p = self._ln_gamma(composition, T + step)
        pp = self._ln_gamma(composition, T + 2.0 * step)
        derivative = tuple((a - 8.0 * b + 8.0 * c - d) / (12.0 * step)
                           for a, b, c, d in zip(mm, m, p, pp))
        return ExcessGibbsState(values, derivative)
