"""
Shortcut, CMO, and equation-oriented distillation column models.
"""

import math
import time
from typing import Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .physical_constants import R_J_MOL_K
else:
    from physical_constants import R_J_MOL_K

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics import StreamState
else:
    from thermodynamics import StreamState
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .equilibrium_stage_column import EquilibriumStageColumnMixin
else:
    from equilibrium_stage_column import EquilibriumStageColumnMixin
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_base import UnitOperation, UnitOperationError, UnitResult
else:
    from unit_operations_base import UnitOperation, UnitOperationError, UnitResult


_DISTILLATE_MOLAR_NAMES = ('distillate_flow', 'top_flow', 'D', 'D_flow')
_DISTILLATE_MASS_NAMES = (
    'distillate_mass_flow',
    'top_mass_flow',
    'D_mass',
    'D_mass_flow',
    'mass_distillate_flow',
)
_DISTILLATE_MASS_FRACTION_NAMES = (
    'D_mass_to_F_mass',
    'D_mass_fraction',
    'D_to_F_mass',
    'distillate_mass_fraction',
    'distillate_mass_to_feed',
    'top_mass_fraction',
)


def _inlet_mass_flow(unit: UnitOperation, inlet: StreamState) -> float:
    mass_flow = inlet.mass_flow()
    if mass_flow <= 0.0:
        mass_flow = inlet.F * unit.thermo.mixture_MW(inlet.composition)
    return mass_flow


def _mass_param_value_kg_per_h(unit: UnitOperation, name: str) -> float:
    value = float(unit.get_param(name))
    unit_str = (unit.get_param_unit(name) or '').lower()
    return value * 0.45359237 if 'lb' in unit_str else value


def _distillate_flow_spec_from_params(unit: UnitOperation, inlet: StreamState,
                                      default_fraction: Optional[float] = None) -> Optional[dict]:
    D_to_F = unit.get_param('D_to_F')
    if D_to_F is not None:
        return {'kind': 'molar', 'value': inlet.F * float(D_to_F), 'source': 'D_to_F'}

    for name in _DISTILLATE_MASS_FRACTION_NAMES:
        value = unit.get_param(name)
        if value is None:
            continue
        return {
            'kind': 'mass',
            'value': _inlet_mass_flow(unit, inlet) * float(value),
            'source': name,
        }

    for name in _DISTILLATE_MASS_NAMES:
        if unit.get_param(name) is None:
            continue
        return {
            'kind': 'mass',
            'value': _mass_param_value_kg_per_h(unit, name),
            'source': name,
        }

    for name in _DISTILLATE_MOLAR_NAMES:
        value = unit.get_param(name)
        if value is None:
            continue
        unit_str = (unit.get_param_unit(name) or '').lower()
        if any(token in unit_str for token in ('kg', 'lb', 'mass')):
            return {
                'kind': 'mass',
                'value': _mass_param_value_kg_per_h(unit, name),
                'source': name,
            }
        return {'kind': 'molar', 'value': float(value), 'source': name}

    if default_fraction is None:
        return None
    return {
        'kind': 'molar',
        'value': inlet.F * float(default_fraction),
        'source': f'default_{default_fraction:g}F',
    }


class ShortcutDistillation(UnitOperation):
    """Fenske-Underwood-Gilliland shortcut distillation column."""

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if not inlets:
            raise UnitOperationError(f"ShortcutDistillation '{self.unit_id}' has no inlet stream")
        inlet = list(inlets.values())[0]
        if inlet.F < 0.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' requires a nonnegative feed flow"
            )

        comps = list(inlet.composition.keys())
        if len(comps) < 2:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' requires at least 2 components"
            )

        P_top = float(self.get_param('P_condenser', self.get_param('P_top', inlet.P)))
        P_drop = float(self.get_param('P_drop_per_stage', 0.01))
        if P_top <= 0.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' condenser pressure must be positive"
            )
        if P_drop < 0.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' requires nonnegative P_drop_per_stage"
            )
        N_spec = self.get_param('N_stages', self.get_param('stages'))
        N_spec = None if N_spec is None else int(N_spec)
        if N_spec is not None and N_spec < 2:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' requires N_stages >= 2"
            )
        P_bottom = P_top + max((N_spec or 2) - 1, 0) * P_drop
        condenser_type = self._condenser_type()
        if inlet.F <= 1e-14:
            return self._zero_feed_result(
                inlet, N_spec, P_top, P_bottom, condenser_type
            )

        self._validate_distillate_specs()

        key_data = self._key_components(comps, inlet)
        light_key = key_data['light_key']
        heavy_key = key_data['heavy_key']
        alpha = key_data['relative_volatility']
        q = float(self.get_param('q', 1.0 - inlet.vapor_fraction))
        if not math.isfinite(q):
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' requires finite feed quality q"
            )
        RR_param = self.get_param('reflux_ratio', self.get_param('RR'))
        if RR_param is not None and float(RR_param) < 0.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' requires reflux_ratio >= 0"
            )

        recoveries = self._key_recoveries(
            comps, inlet, alpha, light_key, heavy_key, N_spec, RR_param, q
        )
        splits = self._hengstebeck_geddes_split(
            comps, inlet, alpha, light_key, heavy_key,
            recoveries['light_key_distillate'],
            recoveries['heavy_key_distillate'],
        )
        D = splits['D']
        B = splits['B']
        x_D = splits['x_D']
        x_B = splits['x_B']
        if D <= 0.0 or B <= 0.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' produced an empty product stream"
            )

        alpha_lk_hk = alpha[light_key] / max(alpha[heavy_key], 1e-30)
        if alpha_lk_hk <= 1.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' requires light key more volatile than heavy key"
            )
        N_min = math.log(
            max(x_D[light_key], 1e-30) / max(x_B[light_key], 1e-30)
            * max(x_B[heavy_key], 1e-30) / max(x_D[heavy_key], 1e-30)
        ) / math.log(alpha_lk_hk)
        if not math.isfinite(N_min) or N_min <= 0.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' could not compute a positive Fenske N_min"
            )

        theta = self._underwood_root(comps, inlet.composition, alpha, light_key, heavy_key, q)
        R_min = self._underwood_minimum_reflux(comps, x_D, alpha, theta)
        if not math.isfinite(R_min) or R_min < 0.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' could not compute Underwood minimum reflux"
            )

        warnings = []
        if RR_param is None and N_spec is None:
            RR = max(1.3 * R_min, R_min + 0.1)
            stages = self._gilliland_stages(N_min, R_min, RR)
            warnings.append("Using default reflux ratio of max(1.3 * R_min, R_min + 0.1)")
        elif RR_param is None:
            stages = float(N_spec)
            RR = self._gilliland_reflux_for_stages(N_min, R_min, stages)
        else:
            RR = float(RR_param)
            if RR <= R_min:
                raise UnitOperationError(
                    f"ShortcutDistillation '{self.unit_id}' requires reflux_ratio > R_min"
                )
            stages = self._gilliland_stages(N_min, R_min, RR)
            if N_spec is not None and N_spec + 1e-9 < stages:
                warnings.append(
                    f"Specified stages ({N_spec}) are below Gilliland estimate ({stages:.2f})"
                )

        feed_stage = self._kirkbride_feed_stage(
            stages if N_spec is None else float(N_spec),
            D, B, inlet.composition, x_D, x_B, light_key, heavy_key,
        )

        condenser_reference_T, noncondensables = self._condenser_reference(
            x_D, P_top
        )
        if condenser_type == 'total':
            if noncondensables:
                names = ', '.join(noncondensables)
                raise UnitOperationError(
                    f"ShortcutDistillation '{self.unit_id}' detected non-condensable "
                    f"component(s) at the condenser conditions: {names}. Use a "
                    "partial condenser with a vapor distillate/vent, or remove these "
                    "components before a total condenser."
                )
            T_top = self.thermo.bubble_point_T(x_D, P_top, condenser_reference_T)
            reflux_composition = dict(x_D)
            distillate_phase = 'liquid'
            condensed_flow = (RR + 1.0) * D
        else:
            T_top = self.thermo.dew_point_T(x_D, P_top, condenser_reference_T)
            reflux_composition = self._incipient_liquid_composition(x_D, T_top, P_top)
            distillate_phase = 'vapor'
            condensed_flow = RR * D
            if noncondensables:
                warnings.append(
                    "Likely non-condensable component(s) routed through the vapor "
                    f"distillate: {', '.join(noncondensables)}"
                )
        T_bottom = self.thermo.bubble_point_T(x_B, P_bottom)
        distillate = self.thermo.calculate_state(
            T_top, P_top, D, x_D, phase=distillate_phase, flash=False
        )
        bottoms = self.thermo.calculate_state(
            T_bottom, P_bottom, B, x_B, phase='liquid', flash=False
        )

        Q_cond = (
            -condensed_flow
            * self._mixture_hvap(reflux_composition, T_top)
            * 1000.0
        )
        H_in = inlet.F * inlet.H
        H_out = D * distillate.H + B * bottoms.H
        Q_reb = H_out - H_in - Q_cond

        return UnitResult(
            outlet_streams={'distillate': distillate, 'bottoms': bottoms},
            heat_duty=H_out - H_in,
            performance={
                'method': 'FUG',
                'N_stages': N_spec if N_spec is not None else int(math.ceil(stages)),
                'light_key': light_key,
                'heavy_key': heavy_key,
                'light_key_recovery_distillate': recoveries['light_key_distillate'],
                'heavy_key_recovery_distillate': recoveries['heavy_key_distillate'],
                'heavy_key_recovery_bottoms': 1.0 - recoveries['heavy_key_distillate'],
                'distillate_spec_source': recoveries.get('distillate_spec_source'),
                'recovery_spec_closure': recoveries.get('closure'),
                'N_min': N_min,
                'R_min': R_min,
                'underwood_theta': theta,
                'reflux_ratio': RR,
                'specified_stages': N_spec,
                'theoretical_stages': stages,
                'stage_margin': None if N_spec is None else float(N_spec) - stages,
                'feed_stage': feed_stage,
                'distillate_flow': D,
                'bottoms_flow': B,
                'T_top_C': T_top - 273.15,
                'T_bottom_C': T_bottom - 273.15,
                'relative_volatility': alpha_lk_hk,
                'relative_volatilities': alpha,
                'component_distillate_recoveries': splits['recoveries'],
                'condenser_type': condenser_type,
                'reflux_liquid_composition': reflux_composition,
                'condenser_duty_kW': Q_cond / 3600.0,
                'reboiler_duty_kW': Q_reb / 3600.0,
            },
            warnings=warnings,
        )

    def _condenser_type(self) -> str:
        condenser_type = str(self.get_param('condenser_type', 'total')).strip().lower()
        if condenser_type in ('complete', 'liquid', 'total_condenser'):
            condenser_type = 'total'
        if condenser_type in ('vapor', 'partial_vapor', 'partial-condenser', 'partial_condenser'):
            condenser_type = 'partial'
        if condenser_type not in ('total', 'partial'):
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' condenser_type must be total or partial"
            )
        return condenser_type

    def _zero_feed_result(
        self,
        inlet: StreamState,
        N_spec: Optional[int],
        P_top: float,
        P_bottom: float,
        condenser_type: str,
    ) -> UnitResult:
        distillate_phase = 'liquid' if condenser_type == 'total' else 'vapor'
        distillate = self.thermo.calculate_state(
            inlet.T, P_top, 0.0, inlet.composition,
            phase=distillate_phase, flash=False,
            include=('H', 'Cp', 'S', 'rho'),
        )
        bottoms = self.thermo.calculate_state(
            inlet.T, P_bottom, 0.0, inlet.composition,
            phase='liquid', flash=False,
            include=('H', 'Cp', 'S', 'rho'),
        )
        return UnitResult(
            outlet_streams={'distillate': distillate, 'bottoms': bottoms},
            performance={
                'method': 'FUG',
                'N_stages': N_spec,
                'reflux_ratio': self.get_param('reflux_ratio', self.get_param('RR')),
                'T_top_C': inlet.T - 273.15,
                'T_bottom_C': inlet.T - 273.15,
                'condenser_duty_kW': 0.0,
                'reboiler_duty_kW': 0.0,
                'distillate_flow': 0.0,
                'bottoms_flow': 0.0,
                'zero_feed': True,
            },
            warnings=[
                "ShortcutDistillation received zero feed; returning zero product streams"
            ],
        )

    def _validate_distillate_specs(self) -> None:
        groups = []
        if self.get_param('D_to_F') is not None:
            groups.append('D_to_F')
        groups.extend(
            name for name in _DISTILLATE_MASS_FRACTION_NAMES
            if self.get_param(name) is not None
        )
        groups.extend(
            name for name in _DISTILLATE_MASS_NAMES
            if self.get_param(name) is not None
        )
        groups.extend(
            name for name in _DISTILLATE_MOLAR_NAMES
            if self.get_param(name) is not None
        )
        if len(groups) > 1:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' accepts only one distillate flow "
                f"specification; received {', '.join(groups)}"
            )

    def _key_components(self, comps: list[str], inlet: StreamState) -> dict:
        alpha_raw = self._relative_volatilities(comps, inlet)
        light_key = self._component_param(('light_key', 'LK', 'key_light'), comps)
        heavy_key = self._component_param(('heavy_key', 'HK', 'key_heavy'), comps)
        threshold = max(float(self.get_param('key_component_threshold', 1e-8)), 0.0)
        ranked = sorted(
            [
                comp for comp in comps
                if inlet.composition.get(comp, 0.0) > threshold
                or comp in (light_key, heavy_key)
            ],
            key=lambda comp: alpha_raw[comp],
            reverse=True,
        )
        if len(ranked) < 2:
            ranked = sorted(comps, key=lambda comp: alpha_raw[comp], reverse=True)

        if light_key is None and heavy_key is None:
            light_key, heavy_key = self._automatic_key_pair(ranked, inlet)
        elif light_key is None:
            heavy_index = ranked.index(heavy_key)
            if heavy_index == 0:
                raise UnitOperationError(
                    f"ShortcutDistillation '{self.unit_id}' heavy_key has no more-volatile "
                    "adjacent component available as the light key"
                )
            light_key = ranked[heavy_index - 1]
        elif heavy_key is None:
            light_index = ranked.index(light_key)
            if light_index == len(ranked) - 1:
                raise UnitOperationError(
                    f"ShortcutDistillation '{self.unit_id}' light_key has no less-volatile "
                    "adjacent component available as the heavy key"
                )
            heavy_key = ranked[light_index + 1]

        if light_key == heavy_key:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' light_key and heavy_key must differ"
            )
        if alpha_raw[light_key] <= alpha_raw[heavy_key]:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' light_key must be more volatile than heavy_key"
            )
        light_index = ranked.index(light_key)
        heavy_index = ranked.index(heavy_key)
        if heavy_index != light_index + 1:
            between = ', '.join(ranked[light_index + 1:heavy_index])
            detail = f"; intervening component(s): {between}" if between else ''
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' light_key and heavy_key must be "
                f"adjacent in volatility order for a pole-free Underwood root{detail}"
            )

        basis = max(alpha_raw[heavy_key], 1e-30)
        alpha = {comp: max(alpha_raw[comp] / basis, 1e-12) for comp in comps}
        return {
            'light_key': light_key,
            'heavy_key': heavy_key,
            'relative_volatility': alpha,
        }

    def _automatic_key_pair(
        self,
        ranked: list[str],
        inlet: StreamState,
    ) -> tuple[str, str]:
        spec = _distillate_flow_spec_from_params(self, inlet, default_fraction=0.5)
        if spec['kind'] == 'mass':
            available = _inlet_mass_flow(self, inlet)
            raw_weights = {
                comp: (
                    inlet.composition.get(comp, 0.0)
                    * self.thermo.props[comp].MW
                )
                for comp in ranked
            }
        else:
            available = inlet.F
            raw_weights = {
                comp: inlet.composition.get(comp, 0.0)
                for comp in ranked
            }
        weight_total = sum(max(value, 0.0) for value in raw_weights.values())
        if available <= 0.0 or weight_total <= 0.0:
            return ranked[0], ranked[1]

        cut = float(spec['value']) / available
        if not math.isfinite(cut):
            cut = 0.5
        cut = min(max(cut, 0.0), 1.0)
        weights = {
            comp: max(raw_weights[comp], 0.0) / weight_total
            for comp in ranked
        }

        cumulative = 0.0
        crossing_index = len(ranked) - 1
        crossing_fraction = 1.0
        for index, comp in enumerate(ranked):
            weight = weights[comp]
            if cumulative + weight >= cut:
                crossing_index = index
                crossing_fraction = (
                    (cut - cumulative) / weight if weight > 0.0 else 0.5
                )
                break
            cumulative += weight

        if crossing_fraction >= 0.5 and crossing_index < len(ranked) - 1:
            return ranked[crossing_index], ranked[crossing_index + 1]
        if crossing_index > 0:
            return ranked[crossing_index - 1], ranked[crossing_index]
        return ranked[0], ranked[1]

    def _condensable_composition(self, composition: dict[str, float]) -> dict[str, float]:
        filtered = {}
        for comp, fraction in composition.items():
            props = self.thermo.props.get(comp)
            boiling_point = getattr(props, 'Tb', None) if props is not None else None
            if fraction > 0.0 and (boiling_point is None or boiling_point >= 230.0):
                filtered[comp] = fraction
        total = sum(filtered.values())
        if total <= 0.0:
            return composition
        return {comp: fraction / total for comp, fraction in filtered.items()}

    def _condenser_reference(
        self,
        composition: dict[str, float],
        pressure: float,
    ) -> tuple[float, list[str]]:
        condensable = self._condensable_composition(composition)
        reference_T = self.thermo.bubble_point_T(condensable, pressure)
        try:
            K = self.thermo.K_values(reference_T, pressure, condensable)
        except Exception:
            K = {}

        min_fraction = max(
            float(self.get_param('noncondensable_min_distillate_fraction', 1e-8)),
            0.0,
        )
        K_threshold = max(
            float(self.get_param('noncondensable_K_threshold', 100.0)),
            1.0,
        )
        suspects = []
        for comp, fraction in composition.items():
            if fraction <= min_fraction:
                continue
            props = self.thermo.props.get(comp)
            Tc = getattr(props, 'Tc', None) if props is not None else None
            Tb = getattr(props, 'Tb', None) if props is not None else None
            supercritical = Tc is not None and Tc < reference_T + 5.0
            extremely_volatile = (
                K.get(comp, 1.0) > K_threshold
                and (Tb is None or Tb < reference_T - 60.0)
            )
            if supercritical or extremely_volatile:
                suspects.append(comp)
        return reference_T, suspects

    def _incipient_liquid_composition(
        self,
        vapor_composition: dict[str, float],
        temperature: float,
        pressure: float,
    ) -> dict[str, float]:
        comps = list(vapor_composition)
        liquid = dict(vapor_composition)
        for _ in range(100):
            K = self.thermo.K_values(temperature, pressure, liquid)
            raw = {
                comp: vapor_composition.get(comp, 0.0)
                / max(float(K.get(comp, 1.0)), 1e-30)
                for comp in comps
            }
            total = sum(raw.values())
            if total <= 0.0 or not math.isfinite(total):
                raise UnitOperationError(
                    f"ShortcutDistillation '{self.unit_id}' could not determine the "
                    "partial-condenser reflux composition"
                )
            updated = {comp: raw[comp] / total for comp in comps}
            change = max(abs(updated[comp] - liquid.get(comp, 0.0)) for comp in comps)
            liquid = updated
            if change < 1e-10:
                return liquid
        raise UnitOperationError(
            f"ShortcutDistillation '{self.unit_id}' partial-condenser reflux "
            "composition did not converge"
        )

    def _component_param(self, names: tuple[str, ...], comps: list[str]) -> Optional[str]:
        for name in names:
            value = self.get_param(name)
            if value is None:
                continue
            matched = self._match_component(comps, str(value))
            if matched is None:
                raise UnitOperationError(
                    f"ShortcutDistillation '{self.unit_id}' unknown component '{value}'"
                )
            return matched
        return None

    def _match_component(self, comps: list[str], value: str) -> Optional[str]:
        needle = value.strip().lower().replace('_', ' ').replace('-', ' ')
        for comp in comps:
            normalized = comp.lower().replace('_', ' ').replace('-', ' ')
            if normalized == needle:
                return comp
        return None

    def _relative_volatilities(self, comps: list[str], inlet: StreamState) -> dict[str, float]:
        if hasattr(self.thermo, 'K_values'):
            try:
                values = self.thermo.K_values(inlet.T, inlet.P, inlet.composition)
                return {comp: max(float(values.get(comp, 1.0)), 1e-12) for comp in comps}
            except Exception:
                pass
        if hasattr(self.thermo, 'activity_coefficients'):
            try:
                gamma = self.thermo.activity_coefficients(inlet.T, inlet.composition)
                return {
                    comp: max(
                        gamma.get(comp, 1.0) * self.thermo.Psat(comp, inlet.T) / inlet.P,
                        1e-12,
                    )
                    for comp in comps
                }
            except Exception:
                pass
        return {
            comp: max(self.thermo.K_value(comp, inlet.T, inlet.P), 1e-12)
            for comp in comps
        }

    def _key_recoveries(
        self,
        comps: list[str],
        inlet: StreamState,
        alpha: dict[str, float],
        light_key: str,
        heavy_key: str,
        N_spec: Optional[int],
        RR_param,
        q: float,
    ) -> dict:
        lk_names = (
            'light_key_recovery_distillate', 'light_key_recovery',
            'LK_recovery_distillate', 'LK_recovery', 'lk_recovery',
        )
        hk_bottom_names = (
            'heavy_key_recovery_bottoms', 'heavy_key_recovery_bottom',
            'heavy_key_recovery', 'HK_recovery_bottoms', 'HK_recovery',
            'hk_recovery',
        )
        hk_dist_names = (
            'heavy_key_recovery_distillate', 'HK_recovery_distillate',
            'hk_recovery_distillate',
        )

        lk_value, lk_explicit = self._explicit_fraction(lk_names)
        hk_bottom_value, hk_bottom_explicit = self._explicit_fraction(hk_bottom_names)
        hk_dist_value, hk_dist_explicit = self._explicit_fraction(hk_dist_names)
        if hk_bottom_explicit and hk_dist_explicit:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' cannot specify both heavy-key "
                "bottoms and distillate recoveries"
            )

        lk = 0.99 if lk_value is None else lk_value
        if hk_dist_value is not None:
            hk_dist = hk_dist_value
        elif hk_bottom_value is not None:
            hk_dist = 1.0 - hk_bottom_value
        else:
            hk_dist = 0.01

        distillate_spec = _distillate_flow_spec_from_params(self, inlet)
        hk_explicit = hk_bottom_explicit or hk_dist_explicit
        if distillate_spec is not None:
            if lk_explicit and hk_explicit:
                raise UnitOperationError(
                    f"ShortcutDistillation '{self.unit_id}' is over-specified: "
                    "distillate flow cannot be combined with both key recoveries"
                )
            if N_spec is not None and RR_param is not None and not lk_explicit and not hk_explicit:
                lk, hk_dist = self._solve_recoveries_for_stage_reflux_distillate(
                    comps, inlet, alpha, light_key, heavy_key,
                    distillate_spec, int(N_spec), float(RR_param), q,
                )
                closure = 'stages_reflux_distillate_backsolved_key_recoveries'
            elif lk_explicit:
                hk_dist = self._solve_recovery_for_distillate_spec(
                    comps, inlet, alpha, light_key, heavy_key,
                    distillate_spec, fixed_lk=lk, fixed_hk=None,
                )
                closure = 'distillate_spec_replaced_heavy_key_recovery'
            elif hk_explicit:
                lk = self._solve_recovery_for_distillate_spec(
                    comps, inlet, alpha, light_key, heavy_key,
                    distillate_spec, fixed_lk=None, fixed_hk=hk_dist,
                )
                closure = 'distillate_spec_replaced_light_key_recovery'
            else:
                try:
                    hk_dist = self._solve_recovery_for_distillate_spec(
                        comps, inlet, alpha, light_key, heavy_key,
                        distillate_spec, fixed_lk=lk, fixed_hk=None,
                    )
                    closure = 'distillate_spec_replaced_default_heavy_key_recovery'
                except UnitOperationError:
                    lk = self._solve_recovery_for_distillate_spec(
                        comps, inlet, alpha, light_key, heavy_key,
                        distillate_spec, fixed_lk=None, fixed_hk=hk_dist,
                    )
                    closure = 'distillate_spec_replaced_default_light_key_recovery'
        else:
            closure = 'key_recoveries'

        self._validate_key_recoveries(lk, hk_dist)
        return {
            'light_key_distillate': lk,
            'heavy_key_distillate': hk_dist,
            'distillate_spec_source': None if distillate_spec is None else distillate_spec['source'],
            'closure': closure,
        }

    def _validate_key_recoveries(self, lk: float, hk_dist: float) -> None:
        if not 0.0 < lk < 1.0 or not 0.0 < hk_dist < 1.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' key recoveries must be between 0 and 1"
            )
        if lk <= hk_dist:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' light key distillate recovery "
                "must exceed heavy key distillate recovery"
            )

    def _explicit_fraction(self, names: tuple[str, ...]) -> tuple[Optional[float], bool]:
        for name in names:
            value = self.get_param(name)
            if value is not None:
                return float(value), True
        return None, False

    def _solve_recovery_for_distillate_spec(
        self,
        comps: list[str],
        inlet: StreamState,
        alpha: dict[str, float],
        light_key: str,
        heavy_key: str,
        distillate_spec: dict,
        fixed_lk: Optional[float],
        fixed_hk: Optional[float],
    ) -> float:
        from scipy.optimize import brentq

        if (fixed_lk is None) == (fixed_hk is None):
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' needs exactly one fixed key recovery"
            )
        target = float(distillate_spec['value'])
        if distillate_spec['kind'] == 'mass':
            available = _inlet_mass_flow(self, inlet)
        else:
            available = inlet.F
        if not 0.0 < target < available:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' distillate flow is infeasible"
            )

        eps = 1e-10
        if fixed_lk is not None:
            self._validate_key_recoveries(fixed_lk, eps)
            lo = eps
            hi = min(fixed_lk - eps, 1.0 - eps)

            def amount(recovery: float) -> float:
                return self._distillate_amount_from_recoveries(
                    comps, inlet, alpha, light_key, heavy_key,
                    fixed_lk, recovery, distillate_spec['kind'],
                )
        else:
            self._validate_key_recoveries(1.0 - eps, fixed_hk)
            lo = max(fixed_hk + eps, eps)
            hi = 1.0 - eps

            def amount(recovery: float) -> float:
                return self._distillate_amount_from_recoveries(
                    comps, inlet, alpha, light_key, heavy_key,
                    recovery, fixed_hk, distillate_spec['kind'],
                )

        f_lo = amount(lo) - target
        f_hi = amount(hi) - target
        if f_lo == 0.0:
            return lo
        if f_hi == 0.0:
            return hi
        if f_lo * f_hi > 0.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' distillate flow cannot replace "
                "the selected key recovery for this split"
            )
        return float(brentq(lambda value: amount(value) - target, lo, hi, xtol=1e-12, rtol=1e-12))

    def _solve_recoveries_for_stage_reflux_distillate(
        self,
        comps: list[str],
        inlet: StreamState,
        alpha: dict[str, float],
        light_key: str,
        heavy_key: str,
        distillate_spec: dict,
        N_spec: int,
        reflux_ratio: float,
        q: float,
    ) -> tuple[float, float]:
        from scipy.optimize import brentq

        if reflux_ratio <= 0.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' requires reflux_ratio > 0"
            )

        def hk_for_lk(lk_recovery: float) -> float:
            return self._solve_recovery_for_distillate_spec(
                comps, inlet, alpha, light_key, heavy_key,
                distillate_spec, fixed_lk=lk_recovery, fixed_hk=None,
            )

        def stage_difference(lk_recovery: float) -> float:
            hk_recovery = hk_for_lk(lk_recovery)
            stages = self._gilliland_stages_for_recoveries(
                comps, inlet, alpha, light_key, heavy_key,
                lk_recovery, hk_recovery, reflux_ratio, q,
            )
            if not math.isfinite(stages):
                return max(float(N_spec), 1.0) * 1e6
            return stages - float(N_spec)

        eps = 1e-9
        samples = []
        for index in range(201):
            lk = eps + (1.0 - 2.0 * eps) * index / 200.0
            try:
                diff = stage_difference(lk)
            except UnitOperationError:
                continue
            samples.append((lk, diff))

        if not samples:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' could not find feasible key "
                "recoveries for N_stages, reflux_ratio, and distillate flow"
            )

        best_lk, best_diff = min(samples, key=lambda item: abs(item[1]))
        if abs(best_diff) < 1e-8:
            return best_lk, hk_for_lk(best_lk)

        bracket = None
        for (lk_a, diff_a), (lk_b, diff_b) in zip(samples, samples[1:]):
            if diff_a == 0.0:
                bracket = (lk_a, lk_a)
                break
            if diff_a * diff_b <= 0.0:
                bracket = (lk_a, lk_b)
                break
        if bracket is None:
            finite_stages = [
                float(N_spec) + diff
                for _, diff in samples
                if abs(diff) < max(float(N_spec), 1.0) * 1e5
            ]
            range_text = (
                f"{min(finite_stages):.3g} to {max(finite_stages):.3g}"
                if finite_stages else "no finite range"
            )
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' could not match N_stages={N_spec} "
                f"at reflux_ratio={reflux_ratio:g}; feasible stage range is approximately "
                f"{range_text}"
            )

        lo, hi = bracket
        if lo == hi:
            lk = lo
        else:
            lk = float(brentq(stage_difference, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=100))
        return lk, hk_for_lk(lk)

    def _gilliland_stages_for_recoveries(
        self,
        comps: list[str],
        inlet: StreamState,
        alpha: dict[str, float],
        light_key: str,
        heavy_key: str,
        lk_recovery: float,
        hk_recovery: float,
        reflux_ratio: float,
        q: float,
    ) -> float:
        split = self._hengstebeck_geddes_split(
            comps, inlet, alpha, light_key, heavy_key, lk_recovery, hk_recovery
        )
        x_D = split['x_D']
        x_B = split['x_B']
        alpha_lk_hk = alpha[light_key] / max(alpha[heavy_key], 1e-30)
        if alpha_lk_hk <= 1.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' requires light key more volatile than heavy key"
            )
        N_min = math.log(
            max(x_D[light_key], 1e-30) / max(x_B[light_key], 1e-30)
            * max(x_B[heavy_key], 1e-30) / max(x_D[heavy_key], 1e-30)
        ) / math.log(alpha_lk_hk)
        if not math.isfinite(N_min) or N_min <= 0.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' could not compute a positive Fenske N_min"
            )
        theta = self._underwood_root(comps, inlet.composition, alpha, light_key, heavy_key, q)
        R_min = self._underwood_minimum_reflux(comps, x_D, alpha, theta)
        if not math.isfinite(R_min) or R_min < 0.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' could not compute Underwood minimum reflux"
            )
        if reflux_ratio <= R_min:
            return float('inf')
        return self._gilliland_stages(N_min, R_min, reflux_ratio)

    def _distillate_amount_from_recoveries(
        self,
        comps: list[str],
        inlet: StreamState,
        alpha: dict[str, float],
        light_key: str,
        heavy_key: str,
        lk_recovery: float,
        hk_recovery: float,
        kind: str,
    ) -> float:
        split = self._hengstebeck_geddes_split(
            comps, inlet, alpha, light_key, heavy_key, lk_recovery, hk_recovery
        )
        if kind == 'mass':
            return sum(
                split['D'] * split['x_D'].get(comp, 0.0) * self.thermo.props[comp].MW
                for comp in comps
            )
        return split['D']

    def _hengstebeck_geddes_split(
        self,
        comps: list[str],
        inlet: StreamState,
        alpha: dict[str, float],
        light_key: str,
        heavy_key: str,
        lk_recovery: float,
        hk_recovery: float,
    ) -> dict:
        r_lk = lk_recovery / max(1.0 - lk_recovery, 1e-30)
        r_hk = hk_recovery / max(1.0 - hk_recovery, 1e-30)
        log_alpha_lk = math.log(max(alpha[light_key], 1e-30))
        log_alpha_hk = math.log(max(alpha[heavy_key], 1e-30))
        slope = (math.log(r_lk) - math.log(r_hk)) / max(log_alpha_lk - log_alpha_hk, 1e-30)
        intercept = math.log(r_hk) - slope * log_alpha_hk

        dist_moles = {}
        bot_moles = {}
        recoveries = {}
        for comp in comps:
            feed_moles = inlet.F * inlet.composition.get(comp, 0.0)
            log_ratio = intercept + slope * math.log(max(alpha[comp], 1e-30))
            if log_ratio >= 0.0:
                recovery = 1.0 / (1.0 + math.exp(-min(log_ratio, 700.0)))
            else:
                ratio = math.exp(max(log_ratio, -700.0))
                recovery = ratio / (1.0 + ratio)
            recovery = min(max(recovery, 1e-12), 1.0 - 1e-12)
            dist_moles[comp] = feed_moles * recovery
            bot_moles[comp] = feed_moles - dist_moles[comp]
            recoveries[comp] = recovery

        D = sum(dist_moles.values())
        B = sum(bot_moles.values())
        x_D = {comp: dist_moles[comp] / D for comp in comps if dist_moles[comp] > 0.0}
        x_B = {comp: bot_moles[comp] / B for comp in comps if bot_moles[comp] > 0.0}
        return {'D': D, 'B': B, 'x_D': x_D, 'x_B': x_B, 'recoveries': recoveries}

    def _underwood_root(
        self,
        comps: list[str],
        z: dict[str, float],
        alpha: dict[str, float],
        light_key: str,
        heavy_key: str,
        q: float,
    ) -> float:
        from scipy.optimize import brentq

        lo = min(alpha[heavy_key], alpha[light_key])
        hi = max(alpha[heavy_key], alpha[light_key])
        threshold = max(float(self.get_param('key_component_threshold', 1e-8)), 0.0)
        active_comps = [
            comp for comp in comps
            if z.get(comp, 0.0) > threshold or comp in (light_key, heavy_key)
        ]
        interior_poles = [
            comp for comp in active_comps
            if lo < alpha[comp] < hi
        ]
        if interior_poles:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' Underwood key interval contains "
                f"interior volatility pole(s): {', '.join(interior_poles)}"
            )
        eps = max(1e-12, 1e-10 * max(abs(lo), abs(hi), 1.0))
        lower = lo + eps
        upper = hi - eps
        if lower >= upper:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' key volatilities are too close "
                "to bracket an Underwood root"
            )

        def residual(theta: float) -> float:
            return sum(
                alpha[comp] * z.get(comp, 0.0) / (alpha[comp] - theta)
                for comp in active_comps
            ) - (1.0 - q)

        try:
            return float(brentq(residual, lower, upper, xtol=1e-12, rtol=1e-12, maxiter=200))
        except Exception as exc:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' could not solve Underwood root"
            ) from exc

    def _underwood_minimum_reflux(
        self,
        comps: list[str],
        x_D: dict[str, float],
        alpha: dict[str, float],
        theta: float,
    ) -> float:
        return sum(
            alpha[comp] * x_D.get(comp, 0.0) / (alpha[comp] - theta)
            for comp in comps
        ) - 1.0

    def _gilliland_y(self, X: float) -> float:
        X = min(max(float(X), 0.0), 1.0)
        if X <= 1e-12:
            return 1.0
        if X >= 1.0 - 1e-12:
            return 0.0
        exponent = (
            (1.0 + 54.4 * X)
            / (11.0 + 117.2 * X)
            * (X - 1.0)
            / math.sqrt(X)
        )
        return 1.0 - math.exp(exponent)

    def _gilliland_stages(self, N_min: float, R_min: float, R: float) -> float:
        X = (R - R_min) / (R + 1.0)
        Y = self._gilliland_y(X)
        if Y >= 1.0:
            return float('inf')
        return (N_min + Y) / max(1.0 - Y, 1e-30)

    def _gilliland_reflux_for_stages(self, N_min: float, R_min: float, stages: float) -> float:
        from scipy.optimize import brentq

        if stages <= N_min:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' specified stages must exceed N_min"
            )
        target_y = (stages - N_min) / (stages + 1.0)
        if not 0.0 < target_y < 1.0:
            raise UnitOperationError(
                f"ShortcutDistillation '{self.unit_id}' could not invert Gilliland correlation"
            )
        X = float(brentq(
            lambda value: self._gilliland_y(value) - target_y,
            1e-12,
            1.0 - 1e-12,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=200,
        ))
        return (R_min + X) / max(1.0 - X, 1e-30)

    def _kirkbride_feed_stage(
        self,
        stages: float,
        D: float,
        B: float,
        z: dict[str, float],
        x_D: dict[str, float],
        x_B: dict[str, float],
        light_key: str,
        heavy_key: str,
    ) -> int:
        ratio = (
            B / max(D, 1e-30)
            * max(x_B.get(heavy_key, 0.0), 1e-30)
            / max(x_D.get(light_key, 0.0), 1e-30)
            * (
                max(z.get(light_key, 0.0), 1e-30)
                / max(z.get(heavy_key, 0.0), 1e-30)
            ) ** 2
        ) ** 0.206
        rectifying = ratio / (1.0 + ratio) * max(stages, 1.0)
        stage = int(round(rectifying + 1.0))
        return max(1, min(int(math.ceil(max(stages, 1.0))), stage))

    def _mixture_hvap(self, composition: dict[str, float], T: float) -> float:
        hvap = 0.0
        for comp, frac in composition.items():
            props = self.thermo.props.get(comp)
            try:
                value = self.thermo.Hvap_at_T(comp, T)
            except Exception:
                value = props.Hvap if props and props.Hvap else 30.0
            hvap += frac * value
        return hvap


class McCabeThieleDistillation(UnitOperation):
    """
    Binary McCabe-Thiele style distillation column.

    Solves a binary McCabe-Thiele style construction. By default it uses
    straight constant-molar-overflow operating lines. Set
    latent_heat_correction=true to use rational operating curves from constant
    component latent heats.

    Required params:
        N_stages: Number of equilibrium stages
        reflux_ratio: External reflux ratio (L/D)

    Optional params:
        P_condenser: Condenser pressure [bar]
        P_drop_per_stage: Pressure drop per stage [bar]
        condenser_type: 'total' or 'partial'
        D_to_F: Distillate to feed ratio
    """
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        inlet = list(inlets.values())[0]
        
        N = int(self.get_param('N_stages', self.get_param('stages', 10)))
        RR = float(self.get_param('reflux_ratio', self.get_param('RR', 2.0)))
        
        P_top = float(self.get_param('P_condenser', self.get_param('P_top', inlet.P)))
        P_drop = float(self.get_param('P_drop_per_stage', 0.01))
        condenser = self.get_param('condenser_type', 'total')
        q = float(self.get_param('q', 1.0 - inlet.vapor_fraction))
        
        warnings = []
        comps = list(inlet.composition.keys())
        if len(comps) != 2:
            raise UnitOperationError(
                f"McCabeThieleDistillation '{self.unit_id}' supports binary feeds only"
            )
        if N < 1:
            raise UnitOperationError(f"McCabeThieleDistillation '{self.unit_id}' requires N_stages >= 1")
        if RR < 0:
            raise UnitOperationError(f"McCabeThieleDistillation '{self.unit_id}' requires reflux_ratio >= 0")
        
        # Stage pressures
        P = [P_top + (i * P_drop) for i in range(N)]

        light_key, heavy_key, alpha, volatility_data = self._binary_keys(comps, inlet)

        solution = self._solve_binary_column_auto_feed(
            inlet, light_key, heavy_key, N, RR, q, P
        )
        feed_stage = solution['selected_feed_stage']
        polish_window = int(self.get_param(
            'mccabe_thiele_feed_stage_polish_window',
            self.get_param('cmo_feed_stage_polish_window', 2),
        ))
        if polish_window > 0:
            candidates = [(
                solution['stage_error'],
                -(solution['x_D'].get(light_key, 0.0) - solution['x_B'].get(light_key, 0.0)),
                feed_stage,
                solution,
            )]
            for stage in range(
                max(1, feed_stage - polish_window),
                min(N, feed_stage + polish_window) + 1,
            ):
                try:
                    stage_solution = self._solve_binary_column(
                        inlet, light_key, heavy_key, N, stage, RR, q, P
                    )
                except UnitOperationError:
                    continue
                separation = (
                    stage_solution['x_D'].get(light_key, 0.0)
                    - stage_solution['x_B'].get(light_key, 0.0)
                )
                candidates.append((
                    stage_solution['stage_error'],
                    -separation,
                    stage,
                    stage_solution,
                ))
            _, _, feed_stage, solution = min(
                candidates,
                key=lambda candidate: candidate[:3],
            )
        if solution['stage_error'] > 1e-4:
            warnings.append(
                f"Column specification solved with residual {solution['stage_error']:.2e}"
            )

        D = solution['D']
        B = solution['B']
        x_D = solution['x_D']
        x_B = solution['x_B']

        T_top = self.thermo.bubble_point_T(x_D, P[0])
        T_bottom = self.thermo.bubble_point_T(x_B, P[-1])
        T_distillate = T_top
        distillate_phase = 'liquid' if condenser == 'total' else 'vapor'
        distillate = self.thermo.calculate_state(
            T_distillate, P[0], D, x_D, phase=distillate_phase, flash=False
        )
        
        bottoms = self.thermo.calculate_state(T_bottom, P[-1], B, x_B, phase='liquid', flash=False)

        V = (RR + 1.0) * D
        Hvap_avg = self._mixture_hvap(x_D, T_top)
        Q_cond = -V * Hvap_avg * 1000.0
        H_in = inlet.F * inlet.H
        H_out = D * distillate.H + B * bottoms.H
        Q_net = H_out - H_in
        Q_reb = Q_net - Q_cond
        
        return UnitResult(
            outlet_streams={'distillate': distillate, 'bottoms': bottoms},
            heat_duty=Q_net,
            performance={
                'N_stages': N,
                'selected_feed_stage': feed_stage,
                'reflux_ratio': RR,
                'T_top_C': T_top - 273.15, 'T_bottom_C': T_bottom - 273.15,
                'distillate_flow': D, 'bottoms_flow': B,
                'light_key': light_key, 'heavy_key': heavy_key,
                'relative_volatility': alpha,
                'relative_volatility_reference_temperature_C': (
                    volatility_data['reference_temperature'] - 273.15
                ),
                'feed_saturation_temperature_C': (
                    volatility_data['feed_saturation_temperature'] - 273.15
                ),
                'pure_key_saturation_temperatures_C': {
                    comp: temperature - 273.15
                    for comp, temperature in volatility_data[
                        'pure_key_saturation_temperatures'
                    ].items()
                },
                'reference_K_values': volatility_data['K_values'],
                'xD_light_key': solution['x_D'].get(light_key, 0.0),
                'xB_light_key': solution['x_B'].get(light_key, 0.0),
                'stage_error': solution['stage_error'],
                'rectifying_stages': max(0, feed_stage - 1),
                'stripping_stages': N - feed_stage + 1,
                'rectifying_stages_required': solution.get('rectifying_stages_required'),
                'stripping_stages_required': solution.get('stripping_stages_required'),
                'cmo_binary_evaluations': solution.get('cmo_binary_evaluations'),
                'operating_line_model': (
                    'latent_heat_curved'
                    if self._mccabe_thiele_latent_heat_correction()
                    else 'constant_molar_overflow'
                ),
                'condenser_duty_kW': Q_cond / 3600,
                'reboiler_duty_kW': Q_reb / 3600,
            },
            warnings=warnings
        )

    def _binary_keys(
        self,
        comps: list[str],
        inlet: StreamState,
    ) -> tuple[str, str, float, dict]:
        data = self._relative_volatility_data(comps, inlet)
        volatilities = data['K_values']

        sorted_comps = sorted(comps, key=lambda c: volatilities[c], reverse=True)
        light_key, heavy_key = sorted_comps[0], sorted_comps[-1]
        alpha = volatilities[light_key] / max(volatilities[heavy_key], 1e-12)
        if alpha <= 1.0:
            raise UnitOperationError(
                f"McCabeThieleDistillation '{self.unit_id}' cannot identify a volatile light key"
            )
        return light_key, heavy_key, alpha, data

    def _relative_volatilities(self, comps: list[str], inlet: StreamState) -> dict[str, float]:
        return self._relative_volatility_data(comps, inlet)['K_values']

    def _relative_volatility_data(
        self,
        comps: list[str],
        inlet: StreamState,
    ) -> dict:
        cache_key = (
            tuple(comps),
            round(float(inlet.P), 10),
            tuple(round(float(inlet.composition.get(comp, 0.0)), 14) for comp in comps),
        )
        cache = getattr(self, '_cmo_relative_volatility_cache', None)
        if cache is None:
            cache = {}
            self._cmo_relative_volatility_cache = cache
        if cache_key in cache:
            return cache[cache_key]

        try:
            feed_saturation_T = float(
                self.thermo.bubble_point_T(inlet.composition, inlet.P, inlet.T)
            )
        except Exception as exc:
            raise UnitOperationError(
                f"McCabeThieleDistillation '{self.unit_id}' could not determine the feed "
                "saturation temperature for relative-volatility evaluation"
            ) from exc

        def model_K_values(temperature: float) -> dict[str, float]:
            try:
                raw = self.thermo.K_values(
                    temperature, inlet.P, inlet.composition
                )
                values = {
                    comp: max(float(raw.get(comp, 1.0)), 1e-12)
                    for comp in comps
                }
            except Exception as exc:
                raise UnitOperationError(
                    f"McCabeThieleDistillation '{self.unit_id}' could not evaluate model "
                    "K-values for relative volatility"
                ) from exc
            if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
                raise UnitOperationError(
                    f"McCabeThieleDistillation '{self.unit_id}' received invalid model K-values"
                )
            return values

        initial_K = model_K_values(feed_saturation_T)
        ranked = sorted(comps, key=lambda comp: initial_K[comp], reverse=True)
        key_pair = (ranked[0], ranked[-1])
        data = None
        for _ in range(max(len(comps) + 2, 4)):
            pure_temperatures = {}
            for comp in key_pair:
                pure = {name: (1.0 if name == comp else 0.0) for name in comps}
                try:
                    pure_temperatures[comp] = float(
                        self.thermo.bubble_point_T(pure, inlet.P, feed_saturation_T)
                    )
                except Exception as exc:
                    raise UnitOperationError(
                        f"McCabeThieleDistillation '{self.unit_id}' could not determine the pure "
                        f"saturation temperature for {comp} at {inlet.P:g} bar"
                    ) from exc

            reference_T = 0.25 * (
                pure_temperatures[key_pair[0]]
                + pure_temperatures[key_pair[1]]
                + 2.0 * feed_saturation_T
            )
            K_values = model_K_values(reference_T)
            ranked = sorted(comps, key=lambda comp: K_values[comp], reverse=True)
            updated_pair = (ranked[0], ranked[-1])
            data = {
                'K_values': K_values,
                'reference_temperature': reference_T,
                'feed_saturation_temperature': feed_saturation_T,
                'pure_key_saturation_temperatures': pure_temperatures,
            }
            if updated_pair == key_pair:
                cache[cache_key] = data
                return data
            key_pair = updated_pair

        raise UnitOperationError(
            f"McCabeThieleDistillation '{self.unit_id}' relative-volatility key pair did not "
            "stabilize at the weighted saturation reference temperature"
        )

    def _binary_operating_lines(
        self,
        inlet: StreamState,
        light_key: str,
        heavy_key: str,
        zF: float,
        xD_lk: float,
        D: float,
        B: float,
        xB_lk: float,
        reflux_ratio: float,
        q: float,
    ) -> dict:
        if self._mccabe_thiele_latent_heat_correction():
            return self._binary_latent_heat_operating_lines(
                inlet, light_key, heavy_key, zF, xD_lk, D, B, xB_lk,
                reflux_ratio, q,
            )

        rect_slope = reflux_ratio / (reflux_ratio + 1.0)
        rect_intercept = xD_lk / (reflux_ratio + 1.0)

        def rectifying_y(x_lk: float) -> float:
            return rect_slope * x_lk + rect_intercept

        if abs(q - 1.0) < 1e-8:
            x_intersect = zF
            y_intersect = rectifying_y(zF)
        else:
            q_slope = q / (q - 1.0)
            q_intercept = -zF / (q - 1.0)
            denom = rect_slope - q_slope
            x_intersect = zF if abs(denom) < 1e-12 else (q_intercept - rect_intercept) / denom
            y_intersect = rectifying_y(x_intersect)

        strip_slope = (y_intersect - xB_lk) / max(x_intersect - xB_lk, 1e-12)
        strip_intercept = xB_lk - strip_slope * xB_lk

        def stripping_y(x_lk: float) -> float:
            return strip_slope * x_lk + strip_intercept

        return {
            'D': D,
            'B': B,
            'xB_lk': xB_lk,
            'rect_slope': rect_slope,
            'rect_intercept': rect_intercept,
            'strip_slope': strip_slope,
            'strip_intercept': strip_intercept,
            'x_intersect': x_intersect,
            'y_intersect': y_intersect,
            'rectifying_y': rectifying_y,
            'stripping_y': stripping_y,
        }

    def _mccabe_thiele_latent_heat_correction(self) -> bool:
        return self._truthy_param(self.get_param(
            'mccabe_thiele_latent_heat_correction',
            self.get_param('latent_heat_correction', False),
        ))

    def _solve_binary_column(self, inlet: StreamState, light_key: str, heavy_key: str,
                             N: int, feed_stage: int, reflux_ratio: float,
                             q: float, pressures: list[float]) -> dict:
        from scipy.optimize import brentq, minimize_scalar

        zF = inlet.composition.get(light_key, 0.0)
        if not 0.0 < zF < 1.0:
            raise UnitOperationError(
                f"McCabeThieleDistillation '{self.unit_id}' binary feed must contain both keys"
            )

        def material_balance(xD_lk: float):
            D = self._distillate_flow_for_composition(inlet, light_key, xD_lk)
            if D <= 0 or D >= inlet.F:
                return None
            B = inlet.F - D
            xB_lk = (inlet.F * zF - D * xD_lk) / B
            if not 0.0 <= xB_lk < zF < xD_lk <= 1.0:
                return None
            return D, B, xB_lk
        
        table_cache = {}
        equilibrium = {}
        for pressure in pressures:
            table_pressure = round(pressure / 0.05) * 0.05
            if table_pressure not in table_cache:
                table_cache[table_pressure] = self._binary_equilibrium_table(
                    light_key, heavy_key, table_pressure
                )
            equilibrium[pressure] = table_cache[table_pressure]

        def operating_lines(xD_lk: float):
            mb = material_balance(xD_lk)
            if mb is None:
                return None
            D, B, xB_lk = mb
            return self._binary_operating_lines(
                inlet, light_key, heavy_key, zF, xD_lk, D, B, xB_lk,
                reflux_ratio, q,
            )

        def section_stage_requirements(xD_lk: float) -> tuple[float, float]:
            lines = operating_lines(xD_lk)
            if lines is None:
                return float('inf'), float('inf')

            def fractional_steps(y_start: float, x_start: float, x_target: float,
                                 line) -> float:
                y_current = min(max(y_start, 0.0), 1.0)
                x_previous = x_start
                for step in range(1, N + 50):
                    pressure = pressures[min(step - 1, len(pressures) - 1)]
                    x_current = equilibrium[pressure]['x_of_y'](y_current)
                    if x_current <= x_target:
                        fraction = (x_previous - x_target) / max(x_previous - x_current, 1e-12)
                        return (step - 1) + min(max(fraction, 0.0), 1.0)
                    y_current = line(x_current)
                    y_current = min(max(y_current, 0.0), 1.0)
                    x_previous = x_current
                return float('inf')

            rectifying_required = fractional_steps(
                xD_lk,
                xD_lk,
                lines['x_intersect'],
                lines['rectifying_y'],
            )
            stripping_required = fractional_steps(
                lines['y_intersect'],
                lines['x_intersect'],
                lines['xB_lk'],
                lines['stripping_y'],
            )
            return rectifying_required, stripping_required

        rectifying_available = max(0.0, feed_stage - 1.0)
        stripping_available = max(0.0, N - feed_stage + 1.0)

        def section_excess(xD_lk: float) -> float:
            rectifying_required, stripping_required = section_stage_requirements(xD_lk)
            return max(
                rectifying_required - rectifying_available,
                stripping_required - stripping_available,
            )

        def stage_residual(xD_lk: float) -> float:
            lines = operating_lines(xD_lk)
            if lines is None:
                return float('nan')

            y_current = xD_lk
            x_stage = xD_lk
            for stage in range(1, N + 1):
                pressure = pressures[min(stage - 1, len(pressures) - 1)]
                x_stage = equilibrium[pressure]['x_of_y'](y_current)
                if stage < feed_stage:
                    y_current = lines['rectifying_y'](x_stage)
                else:
                    y_current = lines['stripping_y'](x_stage)
                y_current = min(max(y_current, 0.0), 1.0)
            return x_stage - lines['xB_lk']

        lo, hi = self._composition_bounds(inlet, light_key)
        grid = [lo + (hi - lo) * i / 500 for i in range(501)]
        samples = [(x, section_excess(x)) for x in grid]
        samples = [(x, r) for x, r in samples if math.isfinite(r)]

        if not samples:
            raise UnitOperationError(
                f"McCabeThieleDistillation '{self.unit_id}' could not bracket a binary split"
            )

        feasible = [(x, r) for x, r in samples if r <= 1e-9]
        root = feasible[-1][0] if feasible else None
        if root is not None:
            root_index = next(i for i, (x, _) in enumerate(samples) if x >= root)
            if root_index + 1 < len(samples) and samples[root_index + 1][1] > 0:
                x1, r1 = samples[root_index]
                x2, r2 = samples[root_index + 1]
                if r1 <= 0.0 < r2:
                    root = brentq(section_excess, x1, x2, xtol=1e-10, rtol=1e-10, maxiter=100)
        else:
            optimum = minimize_scalar(
                lambda x: max(section_excess(x), 0.0),
                bounds=(lo, hi),
                method='bounded',
                options={'xatol': 1e-10},
            )
            root = optimum.x

        D, B, xB_lk = material_balance(root)
        rectifying_required, stripping_required = section_stage_requirements(root)
        stage_error = max(0.0, section_excess(root))
        return {
            'x_D': {light_key: root, heavy_key: 1.0 - root},
            'x_B': {light_key: xB_lk, heavy_key: 1.0 - xB_lk},
            'D': D,
            'B': B,
            'stage_error': stage_error,
            'rectifying_stages_required': rectifying_required,
            'stripping_stages_required': stripping_required,
        }

    def _solve_binary_column_auto_feed(self, inlet: StreamState, light_key: str, heavy_key: str,
                                       N: int, reflux_ratio: float, q: float,
                                       pressures: list[float]) -> dict:
        """Solve a binary CMO column while selecting feed stage in one pass.

        The stage requirements for a trial distillate composition do not depend
        on a particular integer feed tray. Instead of re-solving the binary
        McCabe construction for every feed stage, solve for the maximum split
        whose rectifying + stripping requirements fit in the available stages,
        then assign the integer feed stage closest to that split.
        """
        from scipy.optimize import brentq, minimize_scalar

        zF = inlet.composition.get(light_key, 0.0)
        if not 0.0 < zF < 1.0:
            raise UnitOperationError(
                f"McCabeThieleDistillation '{self.unit_id}' binary feed must contain both keys"
            )

        def material_balance(xD_lk: float):
            D = self._distillate_flow_for_composition(inlet, light_key, xD_lk)
            if D <= 0 or D >= inlet.F:
                return None
            B = inlet.F - D
            xB_lk = (inlet.F * zF - D * xD_lk) / B
            if not 0.0 <= xB_lk < zF < xD_lk <= 1.0:
                return None
            return D, B, xB_lk

        table_cache = {}
        equilibrium = {}
        for pressure in pressures:
            table_pressure = round(pressure / 0.05) * 0.05
            if table_pressure not in table_cache:
                table_cache[table_pressure] = self._binary_equilibrium_table(
                    light_key, heavy_key, table_pressure
                )
            equilibrium[pressure] = table_cache[table_pressure]

        def operating_lines(xD_lk: float):
            mb = material_balance(xD_lk)
            if mb is None:
                return None
            D, B, xB_lk = mb
            return self._binary_operating_lines(
                inlet, light_key, heavy_key, zF, xD_lk, D, B, xB_lk,
                reflux_ratio, q,
            )

        eval_count = 0

        def section_stage_requirements(xD_lk: float) -> tuple[float, float]:
            nonlocal eval_count
            eval_count += 1
            lines = operating_lines(xD_lk)
            if lines is None:
                return float('inf'), float('inf')

            def fractional_steps(y_start: float, x_start: float, x_target: float,
                                 line) -> float:
                y_current = min(max(y_start, 0.0), 1.0)
                x_previous = x_start
                for step in range(1, N + 50):
                    pressure = pressures[min(step - 1, len(pressures) - 1)]
                    x_current = equilibrium[pressure]['x_of_y'](y_current)
                    if x_current <= x_target:
                        fraction = (x_previous - x_target) / max(x_previous - x_current, 1e-12)
                        return (step - 1) + min(max(fraction, 0.0), 1.0)
                    y_current = min(max(line(x_current), 0.0), 1.0)
                    x_previous = x_current
                return float('inf')

            return (
                fractional_steps(
                    xD_lk, xD_lk, lines['x_intersect'],
                    lines['rectifying_y'],
                ),
                fractional_steps(
                    lines['y_intersect'], lines['x_intersect'], lines['xB_lk'],
                    lines['stripping_y'],
                ),
            )

        def stage_excess(xD_lk: float) -> float:
            rectifying_required, stripping_required = section_stage_requirements(xD_lk)
            return rectifying_required + stripping_required - N

        lo, hi = self._composition_bounds(inlet, light_key)
        grid_points = int(self.get_param('cmo_binary_grid_points', 121))
        grid_points = max(31, min(grid_points, 1001))
        grid = [lo + (hi - lo) * i / (grid_points - 1) for i in range(grid_points)]
        samples = [(x, stage_excess(x)) for x in grid]
        samples = [(x, r) for x, r in samples if math.isfinite(r)]

        if not samples:
            raise UnitOperationError(
                f"McCabeThieleDistillation '{self.unit_id}' could not bracket a binary split"
            )

        feasible = [(x, r) for x, r in samples if r <= 1e-9]
        root = feasible[-1][0] if feasible else None
        if root is not None:
            root_index = next(i for i, (x, _) in enumerate(samples) if x >= root)
            if root_index + 1 < len(samples) and samples[root_index + 1][1] > 0:
                x1, r1 = samples[root_index]
                x2, r2 = samples[root_index + 1]
                if r1 <= 0.0 < r2:
                    root = brentq(stage_excess, x1, x2, xtol=1e-10, rtol=1e-10, maxiter=100)
        else:
            optimum = minimize_scalar(
                lambda x: max(stage_excess(x), 0.0),
                bounds=(lo, hi),
                method='bounded',
                options={'xatol': 1e-10},
            )
            root = optimum.x

        D, B, xB_lk = material_balance(root)
        rectifying_required, stripping_required = section_stage_requirements(root)
        feed_stage_float = rectifying_required + 1.0
        feed_stage = int(round(feed_stage_float))
        feed_stage = max(1, min(N, feed_stage))
        rectifying_available = max(0.0, feed_stage - 1.0)
        stripping_available = max(0.0, N - feed_stage + 1.0)
        stage_error = max(
            0.0,
            rectifying_required - rectifying_available,
            stripping_required - stripping_available,
        )
        return {
            'x_D': {light_key: root, heavy_key: 1.0 - root},
            'x_B': {light_key: xB_lk, heavy_key: 1.0 - xB_lk},
            'D': D,
            'B': B,
            'selected_feed_stage': feed_stage,
            'stage_error': stage_error,
            'rectifying_stages_required': rectifying_required,
            'stripping_stages_required': stripping_required,
            'cmo_binary_evaluations': eval_count,
        }

    def _solve_multicomponent_column(self, inlet: StreamState, comps: list[str],
                                     light_key: str, heavy_key: str, N: int,
                                     feed_stage: int, reflux_ratio: float) -> dict:
        from scipy.optimize import brentq

        feed_moles = {
            comp: inlet.F * inlet.composition.get(comp, 0.0)
            for comp in comps
        }
        feed_mass = {
            comp: moles * self.thermo.props[comp].MW
            for comp, moles in feed_moles.items()
        }

        volatilities = self._relative_volatilities(comps, inlet)

        key_scale = max(volatilities.get(heavy_key, 1.0), 1e-12)
        relative_volatility = {
            comp: max(volatilities[comp] / key_scale, 1e-9)
            for comp in comps
        }

        z_light = inlet.composition.get(light_key, 0.0)
        ideal_feed_stage = 1.0 + (N - 1) * (1.0 - z_light)
        feed_mismatch = abs(feed_stage - ideal_feed_stage) / max(N, 1)
        feed_efficiency = max(0.25, 1.0 - 1.5 * feed_mismatch)
        reflux_efficiency = reflux_ratio / (reflux_ratio + 1.0)
        effective_stages = max(1.0, N * feed_efficiency * max(reflux_efficiency, 0.05))

        scores = {
            comp: relative_volatility[comp] ** effective_stages
            for comp in comps
        }

        def split_fraction(comp: str, theta: float) -> float:
            return min(max(scores[comp] / (scores[comp] + theta), 1e-9), 1.0 - 1e-9)

        spec = self._distillate_flow_raw(inlet)

        def distillate_amount(theta: float) -> float:
            if spec['kind'] == 'mass':
                return sum(
                    feed_mass[comp] * split_fraction(comp, theta)
                    for comp in comps
                )
            return sum(
                feed_moles[comp] * split_fraction(comp, theta)
                for comp in comps
            )

        target = spec['value']
        available = sum(feed_mass.values()) if spec['kind'] == 'mass' else inlet.F
        if not 0.0 < target < available:
            raise UnitOperationError(
                f"McCabeThieleDistillation '{self.unit_id}' distillate flow is infeasible"
            )

        low, high = 1e-18, 1.0
        while distillate_amount(high) > target:
            high *= 10.0
            if high > 1e24:
                raise UnitOperationError(
                    f"McCabeThieleDistillation '{self.unit_id}' could not solve multicomponent split"
                )

        theta = brentq(
            lambda value: distillate_amount(value) - target,
            low,
            high,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=200,
        )

        dist_moles = {
            comp: feed_moles[comp] * split_fraction(comp, theta)
            for comp in comps
        }
        bot_moles = {
            comp: max(feed_moles[comp] - dist_moles[comp], 0.0)
            for comp in comps
        }

        D = sum(dist_moles.values())
        B = sum(bot_moles.values())
        if D <= 0.0 or B <= 0.0:
            raise UnitOperationError(
                f"McCabeThieleDistillation '{self.unit_id}' produced an empty product stream"
            )

        x_D = {comp: value / D for comp, value in dist_moles.items() if value > 0.0}
        x_B = {comp: value / B for comp, value in bot_moles.items() if value > 0.0}

        residual = abs(distillate_amount(theta) - target) / max(target, 1.0)
        return {
            'x_D': x_D,
            'x_B': x_B,
            'D': D,
            'B': B,
            'stage_error': residual,
        }

    def _binary_equilibrium_table(self, light_key: str, heavy_key: str, pressure: float) -> dict:
        import numpy as np

        cache_key = (light_key, heavy_key, float(pressure))
        cache = getattr(self.thermo, '_cmo_binary_equilibrium_cache', None)
        if cache is None:
            cache = {}
            setattr(self.thermo, '_cmo_binary_equilibrium_cache', cache)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        n_points = int(self.get_param('cmo_equilibrium_points', 301))
        n_points = max(151, min(n_points, 2001))
        x_grid = np.linspace(0.0, 1.0, n_points)
        y_values = []
        for x_lk in x_grid:
            x = {light_key: float(x_lk), heavy_key: float(1.0 - x_lk)}
            if x_lk <= 0.0:
                y_values.append(0.0)
                continue
            if x_lk >= 1.0:
                y_values.append(1.0)
                continue
            T = self.thermo.bubble_point_T(x, pressure)
            K = self.thermo.K_values(T, pressure, x)
            y = {c: K.get(c, 1.0) * frac for c, frac in x.items()}
            total = sum(max(0.0, value) for value in y.values())
            y_values.append(max(0.0, min(1.0, y[light_key] / total if total else x_lk)))

        y_grid = np.maximum.accumulate(np.array(y_values, dtype=float))
        y_grid[0] = 0.0
        y_grid[-1] = 1.0
        unique_y, unique_idx = np.unique(y_grid, return_index=True)
        unique_x = x_grid[unique_idx]

        def y_of_x(x_lk: float) -> float:
            return float(np.interp(min(max(x_lk, 0.0), 1.0), x_grid, y_grid))

        def x_of_y(y_lk: float) -> float:
            return float(np.interp(min(max(y_lk, 0.0), 1.0), unique_y, unique_x))

        table = {
            'x_grid': x_grid,
            'y_grid': y_grid,
            'unique_y': unique_y,
            'unique_x': unique_x,
            'y_of_x': y_of_x,
            'x_of_y': x_of_y,
        }
        if len(cache) > 200:
            cache.clear()
        cache[cache_key] = table
        return table

    def _composition_bounds(self, inlet: StreamState, light_key: str) -> tuple[float, float]:
        zF = inlet.composition[light_key]
        eps = 1e-7
        D_spec = self._distillate_flow_raw(inlet)
        if D_spec['kind'] == 'molar':
            D = D_spec['value']
            lo = max(zF + eps, (inlet.F * zF - (inlet.F - D)) / max(D, eps) + eps)
            hi = min(1.0 - eps, inlet.F * zF / max(D, eps) - eps)
        else:
            lo, hi = zF + eps, 1.0 - eps
        if lo >= hi:
            raise UnitOperationError(
                f"McCabeThieleDistillation '{self.unit_id}' distillate flow is infeasible"
            )
        return lo, hi

    def _distillate_flow_raw(self, inlet: StreamState) -> dict:
        return _distillate_flow_spec_from_params(self, inlet, default_fraction=0.5)

    def _distillate_flow_for_composition(self, inlet: StreamState, light_key: str,
                                         xD_lk: float) -> float:
        spec = self._distillate_flow_raw(inlet)
        if spec['kind'] == 'molar':
            return spec['value']

        comps = list(inlet.composition.keys())
        heavy_key = next(comp for comp in comps if comp != light_key)
        mw_lk = self.thermo.props[light_key].MW
        mw_hk = self.thermo.props[heavy_key].MW
        mw_distillate = xD_lk * mw_lk + (1.0 - xD_lk) * mw_hk
        return spec['value'] / mw_distillate

    def _mixture_hvap(self, composition: dict[str, float], T: float) -> float:
        hvap = 0.0
        for comp, frac in composition.items():
            props = self.thermo.props.get(comp)
            try:
                value = self.thermo.Hvap_at_T(comp, T)
            except Exception:
                value = props.Hvap if props and props.Hvap else 30.0
            hvap += frac * value
        return hvap

    def _binary_hvap_values(
        self,
        light_key: str,
        heavy_key: str,
        reference_T: float,
    ) -> tuple[float, float]:
        values = []
        for comp in (light_key, heavy_key):
            props = self.thermo.props.get(comp)
            try:
                value = self.thermo.Hvap_at_T(comp, reference_T)
            except Exception:
                value = props.Hvap if props and props.Hvap else 30.0
            if not math.isfinite(value) or value <= 0.0:
                raise UnitOperationError(
                    f"McCabeThieleDistillation '{self.unit_id}' requires positive Hvap "
                    f"for {comp}"
                )
            values.append(float(value))
        return values[0], values[1]

    @staticmethod
    def _binary_hvap_at_y(y_lk: float, hvap_lk: float, hvap_hk: float) -> float:
        return y_lk * hvap_lk + (1.0 - y_lk) * hvap_hk

    @staticmethod
    def _latent_heat_line(
        x_lk: float,
        anchor_lk: float,
        heat_flow_per_external_mol: float,
        hvap_lk: float,
        hvap_hk: float,
    ) -> float:
        if abs(heat_flow_per_external_mol) < 1e-30:
            return float('nan')
        coefficient = (anchor_lk - x_lk) / heat_flow_per_external_mol
        delta = hvap_lk - hvap_hk
        denominator = 1.0 - coefficient * delta
        if abs(denominator) < 1e-12:
            return float('nan')
        return (x_lk + coefficient * hvap_hk) / denominator

    def _binary_latent_heat_operating_lines(
        self,
        inlet: StreamState,
        light_key: str,
        heavy_key: str,
        zF: float,
        xD_lk: float,
        D: float,
        B: float,
        xB_lk: float,
        reflux_ratio: float,
        q: float,
    ) -> dict:
        from scipy.optimize import brentq

        volatility_data = self._relative_volatility_data(
            [light_key, heavy_key], inlet
        )
        reference_T = volatility_data['reference_temperature']
        hvap_lk, hvap_hk = self._binary_hvap_values(
            light_key, heavy_key, reference_T
        )

        hvap_distillate = self._binary_hvap_at_y(xD_lk, hvap_lk, hvap_hk)
        rectifying_heat_per_D = (reflux_ratio + 1.0) * hvap_distillate

        def rectifying_y(x_lk: float) -> float:
            return self._latent_heat_line(
                x_lk, xD_lk, rectifying_heat_per_D, hvap_lk, hvap_hk
            )

        feed_hvap = self._binary_hvap_at_y(zF, hvap_lk, hvap_hk)
        stripping_heat_flow = (
            D * rectifying_heat_per_D
            - (1.0 - q) * inlet.F * feed_hvap
        )
        strip_heat_per_B = stripping_heat_flow / max(B, 1e-30)
        if not math.isfinite(strip_heat_per_B) or strip_heat_per_B <= 0.0:
            return super()._binary_operating_lines(
                inlet, light_key, heavy_key, zF, xD_lk, D, B, xB_lk,
                reflux_ratio, q,
            )

        def stripping_y(x_lk: float) -> float:
            return self._latent_heat_line(
                x_lk, xB_lk, -strip_heat_per_B, hvap_lk, hvap_hk
            )

        def intersection_residual(x_lk: float) -> float:
            return rectifying_y(x_lk) - stripping_y(x_lk)

        samples = []
        x_low = max(0.0, min(xB_lk, xD_lk))
        x_high = min(1.0, max(xB_lk, xD_lk))
        for index in range(101):
            x = x_low + (x_high - x_low) * index / 100.0
            try:
                value = intersection_residual(x)
            except Exception:
                continue
            if math.isfinite(value):
                samples.append((x, value))
        bracket = None
        for (x1, f1), (x2, f2) in zip(samples, samples[1:]):
            if f1 == 0.0:
                bracket = (x1, x1)
                break
            if f1 * f2 <= 0.0:
                bracket = (x1, x2)
                break
        if bracket is None:
            x_intersect = zF
        elif bracket[0] == bracket[1]:
            x_intersect = bracket[0]
        else:
            x_intersect = float(
                brentq(
                    intersection_residual,
                    bracket[0],
                    bracket[1],
                    xtol=1e-12,
                    rtol=1e-12,
                )
            )
        y_intersect = rectifying_y(x_intersect)
        if not math.isfinite(y_intersect):
            return super()._binary_operating_lines(
                inlet, light_key, heavy_key, zF, xD_lk, D, B, xB_lk,
                reflux_ratio, q,
            )

        return {
            'D': D,
            'B': B,
            'xB_lk': xB_lk,
            'x_intersect': x_intersect,
            'y_intersect': y_intersect,
            'rectifying_y': rectifying_y,
            'stripping_y': stripping_y,
            'operating_line_model': 'latent_heat_curved',
            'hvap_reference_temperature': reference_T,
            'component_hvap': {
                light_key: hvap_lk,
                heavy_key: hvap_hk,
            },
            'rectifying_heat_per_distillate_mol': rectifying_heat_per_D,
            'stripping_heat_per_bottoms_mol': strip_heat_per_B,
        }


class RigorousDistillation(EquilibriumStageColumnMixin, UnitOperation):
    """
    Equation-oriented equilibrium-stage distillation column.

    The converged solution is obtained from a square set of MESH equations
    using a damped sparse Newton method. Initializers range from a cheap smooth
    estimate to coarse rigorous and full-profile CMO solves.
    """

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:

        if not inlets:
            raise UnitOperationError(f"RigorousDistillation '{self.unit_id}' has no inlet stream")
        N = int(self.get_param('N_stages', self.get_param('stages', 10)))
        feed_stage = int(self.get_param('feed_stage', max(1, N // 2)))
        inlet, feed_specs = self._aggregate_feeds(inlets, N, feed_stage)
        if inlet.F <= 0.0:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' requires a positive feed flow"
            )
        comps = self._component_order(inlet)
        nc = len(comps)
        if nc < 2:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' needs >= 2 components"
            )

        RR = float(self.get_param('reflux_ratio', self.get_param('RR', 2.0)))
        condenser = str(self.get_param('condenser_type', 'total')).strip().lower()
        if condenser in ('complete', 'liquid', 'total_condenser'):
            condenser = 'total'
        if condenser in ('vapor', 'partial_vapor', 'partial-condenser', 'partial_condenser'):
            condenser = 'partial'
        if condenser in ('two_phase', 'two-phase', 'mixed_distillate', 'partial_liquid'):
            condenser = 'mixed'
        if condenser in ('decanter', 'heterogeneous', 'heterogeneous_decanter', 'top_decanter'):
            condenser = 'decanter'
        if condenser not in ('total', 'partial', 'mixed', 'decanter'):
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' condenser_type must be total, mixed, partial, or decanter"
            )
        condenser_vapor_fraction = self._condenser_vapor_fraction(condenser)
        decanter_options = self._decanter_options() if condenser == 'decanter' else None
        if N < 2:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' requires N_stages >= 2"
            )
        if not 1 <= feed_stage <= N:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' feed_stage must be between 1 and {N}"
            )
        if RR < 0.0:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' requires reflux_ratio >= 0"
            )

        pressures = self._pressure_profile(N, inlet.P)
        if any(P <= 0.0 for P in pressures):
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' pressure profile must be positive"
            )

        feed_z = self._normalize({comp: inlet.composition.get(comp, 0.0) for comp in comps})
        warnings: list[str] = []
        quality_context = getattr(self.thermo, 'quality_context', None)
        if quality_context is None:
            classification = self._classify_components(comps, feed_z, pressures[0], inlet)
        else:
            with quality_context(
                phase='component_classification',
                affects_result=False,
            ):
                classification = self._classify_components(comps, feed_z, pressures[0], inlet)
        if classification['noncondensables']:
            names = ', '.join(classification['noncondensables'])
            trace_limit = float(self.get_param('noncondensable_trace_limit', 0.02))
            noncondensable_fraction = sum(
                feed_z.get(comp, 0.0) for comp in classification['noncondensables']
            )
            if condenser_vapor_fraction <= 0.0 or noncondensable_fraction > trace_limit:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' detected non-condensable "
                    f"component(s) at the condenser conditions: {names}. Use a partial "
                    "or mixed condenser with a vapor distillate/vent for trace "
                    "non-condensables, or remove these components before a total condenser."
                )
            warnings.append(
                f"Trace non-condensable component(s) routed through the vapor distillate: {names}"
            )

        side_draws = self._parse_side_draws(N)
        distillate_spec = (
            self._decanter_distillate_guess(inlet)
            if condenser == 'decanter'
            else self._distillate_spec(inlet)
        )
        self._validate_external_flow_specs(inlet, distillate_spec, side_draws)
        T_min, T_max = self._temperature_bounds(comps)
        if self.get_param('q') is not None:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' does not support user-specified q; "
                "the feed thermal condition is determined from feed enthalpy"
            )
        q_feed = self._feed_thermal_condition(
            inlet, feed_z, pressures[feed_stage - 1], T_min, T_max
        )

        flow_scale = max(inlet.F, 1.0)
        energy_scale = max(abs(inlet.F * (inlet.H or 0.0)), inlet.F * 50000.0, 1.0)
        component_floor = float(self.get_param('component_scale_floor', 1e-4))
        component_scales = {
            comp: max(inlet.F * feed_z.get(comp, 0.0), flow_scale * component_floor, 1e-12)
            for comp in comps
        }

        stage_phase_model = self._stage_phase_model()
        if stage_phase_model == 'VLLE':
            self._validate_vlle_stage_configuration(condenser, side_draws)
            return self._solve_vlle_mode(
                inlets=inlets,
                inlet=inlet,
                feed_specs=feed_specs,
                comps=comps,
                feed_z=feed_z,
                N=N,
                feed_stage=feed_stage,
                RR=RR,
                pressures=pressures,
                condenser=condenser,
                condenser_vapor_fraction=condenser_vapor_fraction,
                distillate_spec=distillate_spec,
                T_min=T_min,
                T_max=T_max,
                q_feed=q_feed,
                flow_scale=flow_scale,
                energy_scale=energy_scale,
                component_scales=component_scales,
                classification=classification,
                warnings=warnings,
            )

        if quality_context is None:
            initial = self._initial_guess(
                inlet, comps, feed_z, N, feed_stage, RR, q_feed, pressures,
                condenser, condenser_vapor_fraction, distillate_spec, side_draws, T_min, T_max
            )
        else:
            with quality_context(
                phase='initializer',
                affects_result=False,
            ):
                initial = self._initial_guess(
                    inlet, comps, feed_z, N, feed_stage, RR, q_feed, pressures,
                    condenser, condenser_vapor_fraction, distillate_spec, side_draws, T_min, T_max
                )

        model = self._build_mesh_model(
            inlet, feed_specs, comps, feed_z, N, feed_stage - 1, RR, pressures, condenser,
            condenser_vapor_fraction, decanter_options, side_draws, distillate_spec,
            flow_scale, energy_scale, component_scales, T_min, T_max,
        )

        z0 = self._pack_variables(
            initial['T'], initial['x'], initial['L'], initial['V'],
            initial['Q_cond'], initial['Q_reb'], comps, T_min, T_max, energy_scale,
            initial.get('decanter_distillate_x'), initial.get('decanter_beta'),
        )

        solver_options = {
            'mesh_tolerance': float(self.get_param('mesh_tolerance', 2e-6)),
            'max_iterations': int(self.get_param('max_iterations', self.get_param('max_evaluations', 60))),
            'max_jacobian_evaluations': int(self.get_param('max_jacobian_evaluations', 60)),
            'line_search_steps': int(self.get_param('line_search_steps', 16)),
            'finite_difference_rel_step': float(self.get_param('finite_difference_rel_step', 1e-6)),
            'stall_iterations': int(self.get_param('newton_stall_iterations', 0)),
            'stall_relative_tolerance': float(self.get_param(
                'newton_stall_relative_tolerance', 1e-4
            )),
        }
        solver_options['acceptable_mesh_residual'] = float(
            self.get_param(
                'acceptable_mesh_residual',
                max(50.0 * solver_options['mesh_tolerance'], solver_options['mesh_tolerance']),
            )
        )
        if self.get_param('finite_difference_rel_step') is None:
            candidate_steps = [1e-6, 1e-5, 1e-4, 1e-3]
        else:
            candidate_steps = [solver_options['finite_difference_rel_step']]
        solution = None
        attempted_optimized_jacobian = False
        jacobian_fallback = False
        for step in candidate_steps:
            attempt_options = dict(solver_options)
            attempt_options['finite_difference_rel_step'] = step
            attempt = self._sparse_newton_solve(
                model['residual'], model['sparsity'], z0, attempt_options,
                jacobian=model.get('jacobian'),
            )
            attempt['finite_difference_rel_step'] = step
            attempted_optimized_jacobian = attempted_optimized_jacobian or (
                attempt.get('jacobian_method') != 'colored_finite_difference'
            )
            if solution is None or attempt['residual_norm'] < solution['residual_norm']:
                solution = attempt
            if attempt['success']:
                solution = attempt
                break

        if (
            not solution['success']
            and attempted_optimized_jacobian
            and self._truthy_param(self.get_param(
                'colored_jacobian_fallback', True
            ))
        ):
            for step in candidate_steps:
                attempt_options = dict(solver_options)
                attempt_options['finite_difference_rel_step'] = step
                attempt = self._sparse_newton_solve(
                    model['residual'],
                    model['sparsity'],
                    z0,
                    attempt_options,
                    jacobian=None,
                )
                attempt['finite_difference_rel_step'] = step
                if attempt['success'] or attempt['residual_norm'] < solution['residual_norm']:
                    solution = attempt
                if attempt['success']:
                    jacobian_fallback = True
                    warnings.append(
                        "Local/semi-analytic Jacobian did not converge; retried with "
                        "the original colored finite-difference Jacobian."
                    )
                    break

        if not solution['success']:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' MESH solve failed "
                f"(residual {solution['residual_norm']:.2e}): {solution['message']}"
            )
        if solution['residual_norm'] > solver_options['mesh_tolerance']:
            warnings.append(
                f"MESH solver stalled near tolerance and accepted residual "
                f"{solution['residual_norm']:.2e}"
            )

        decoded = model['decode'](solution['x'])
        T = decoded['T']
        x = decoded['x']
        L = decoded['L']
        V = decoded['V']
        Q_cond = decoded['Q_cond']
        Q_reb = decoded['Q_reb']
        stage_props = [
            model['stage_properties'](stage, T[stage], x[stage])
            for stage in range(N)
        ]
        y = [props['y'] for props in stage_props]
        hL = [props['hL'] for props in stage_props]
        hV = [props['hV'] for props in stage_props]
        stage_spinodal = []
        if stage_phase_model == 'VL(L)E':
            spinodal_test = getattr(self.thermo, 'liquid_spinodal_stability', None)
            if not callable(spinodal_test):
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' requires a local "
                    "liquid spinodal test for stage_phase_model=VL(L)E"
                )
            for stage in range(N):
                check = spinodal_test(float(T[stage]), x[stage])
                stage_spinodal.append(check)
            unstable_stages = [
                stage + 1
                for stage, check in enumerate(stage_spinodal)
                if not check['locally_stable']
            ]
            if unstable_stages:
                self._validate_vlle_stage_configuration(condenser, side_draws)
                warnings.append(
                    "VL(L)E stage spinodal check found local liquid instability "
                    f"on stage(s) {unstable_stages}; reran with the VLLE MESH model."
                )
                return self._solve_vlle_mode(
                    inlets=inlets,
                    inlet=inlet,
                    feed_specs=feed_specs,
                    comps=comps,
                    feed_z=feed_z,
                    N=N,
                    feed_stage=feed_stage,
                    RR=RR,
                    pressures=pressures,
                    condenser=condenser,
                    condenser_vapor_fraction=condenser_vapor_fraction,
                    distillate_spec=distillate_spec,
                    T_min=T_min,
                    T_max=T_max,
                    q_feed=q_feed,
                    flow_scale=flow_scale,
                    energy_scale=energy_scale,
                    component_scales=component_scales,
                    classification=classification,
                    warnings=warnings,
                )
        warnings.extend(
            self._post_solve_noncondensable_check(
                comps, feed_z, float(T[0]), float(pressures[0]), x[0], y[0],
                stage_props[0]['K'], condenser_vapor_fraction,
            )
        )

        D = float(V[0])
        decanter_split = None
        decanter_purge = None
        if condenser == 'decanter':
            decanter_split = model['top_decanter_solution'](
                float(T[0]), float(L[0]), float(V[0]), x[0],
                decoded['decanter_distillate_x'], decoded['decanter_beta'],
            )
            if not decanter_split.get('two_phases'):
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' top decanter did not form "
                    "two liquid phases at the converged condenser conditions. Use a "
                    "mixed/partial condenser or adjust entrainer/recycle specifications."
                )
            reflux_selector = (decanter_options or {}).get('reflux_component')
            if reflux_selector:
                reflux_key = self._match_component_name(comps, reflux_selector)
                if (
                    reflux_key is not None
                    and decanter_split['reflux_composition'].get(reflux_key, 0.0)
                    < decanter_split['distillate_composition'].get(reflux_key, 0.0)
                ):
                    raise UnitOperationError(
                        f"RigorousDistillation '{self.unit_id}' converged to a decanter "
                        f"phase assignment where the reflux phase is not richer in {reflux_key}"
                    )
            D = float(decanter_split['distillate_flow'])
            distillate_vapor_flow = 0.0
            distillate_liquid_flow = D
            if decanter_split['purge_flow'] > 0.0:
                decanter_purge = self.thermo.calculate_state(
                    float(T[0]), float(pressures[0]), decanter_split['purge_flow'],
                    decanter_split['reflux_composition'], phase='liquid', flash=False
                )
        else:
            distillate_vapor_flow = condenser_vapor_fraction * D
            distillate_liquid_flow = (1.0 - condenser_vapor_fraction) * D
        B = float(L[-1])
        if condenser == 'decanter':
            distillate_comp = {
                comp: float(decanter_split['distillate_composition'].get(comp, 0.0))
                for comp in comps
            }
            distillate_h = self.thermo.mixture_enthalpy(
                distillate_comp, float(T[0]), vapor_fraction=0.0,
                P=float(pressures[0]),
            )
            distillate = self.thermo.calculate_state(
                float(T[0]), float(pressures[0]), D, distillate_comp,
                phase='liquid', flash=False
            )
        elif condenser_vapor_fraction >= 1.0 - 1e-12:
            distillate_comp = {comp: float(y[0].get(comp, 0.0)) for comp in comps}
            distillate_phase = 'vapor'
            distillate_h = hV[0]
            distillate = self.thermo.calculate_state(
                float(T[0]), float(pressures[0]), D, distillate_comp,
                phase=distillate_phase, flash=False
            )
        elif condenser_vapor_fraction <= 1e-12:
            distillate_comp = {comp: float(x[0].get(comp, 0.0)) for comp in comps}
            distillate_phase = 'liquid'
            distillate_h = hL[0]
            distillate = self.thermo.calculate_state(
                float(T[0]), float(pressures[0]), D, distillate_comp,
                phase=distillate_phase, flash=False
            )
        else:
            x_dist = {comp: float(x[0].get(comp, 0.0)) for comp in comps}
            y_dist = {comp: float(y[0].get(comp, 0.0)) for comp in comps}
            distillate_comp = self._normalize({
                comp: distillate_liquid_flow * x_dist.get(comp, 0.0)
                + distillate_vapor_flow * y_dist.get(comp, 0.0)
                for comp in comps
            })
            distillate_h = (
                (1.0 - condenser_vapor_fraction) * hL[0]
                + condenser_vapor_fraction * hV[0]
            )
            distillate = self._two_phase_stream(
                float(T[0]), float(pressures[0]), D, distillate_comp,
                x_dist, y_dist, condenser_vapor_fraction, distillate_h
            )
        bottoms_comp = {comp: float(x[-1].get(comp, 0.0)) for comp in comps}

        bottoms = self.thermo.calculate_state(
            float(T[-1]), float(pressures[-1]), B, bottoms_comp,
            phase='liquid', flash=False
        )

        outlets = {'distillate': distillate, 'bottoms': bottoms}
        if decanter_purge is not None:
            outlets['decanter_purge'] = decanter_purge
        if condenser != 'decanter' and 1e-12 < condenser_vapor_fraction < 1.0 - 1e-12:
            outlets['distillate_liquid'] = self.thermo.calculate_state(
                float(T[0]), float(pressures[0]), distillate_liquid_flow,
                {comp: float(x[0].get(comp, 0.0)) for comp in comps},
                phase='liquid', flash=False,
            )
            outlets['distillate_vapor'] = self.thermo.calculate_state(
                float(T[0]), float(pressures[0]), distillate_vapor_flow,
                {comp: float(y[0].get(comp, 0.0)) for comp in comps},
                phase='vapor', flash=False,
            )
        side_summaries = []
        for draw in side_draws:
            stage = draw['stage']
            phase = draw['phase']
            draw_flow = model['side_draw_flow'](draw, L, V, x, y)
            if phase == 'vapor':
                comp = {c: float(y[stage].get(c, 0.0)) for c in comps}
                state = self.thermo.calculate_state(
                    float(T[stage]), float(pressures[stage]), draw_flow, comp,
                    phase='vapor', flash=False
                )
            else:
                comp = {c: float(x[stage].get(c, 0.0)) for c in comps}
                state = self.thermo.calculate_state(
                    float(T[stage]), float(pressures[stage]), draw_flow, comp,
                    phase='liquid', flash=False
                )
            outlets[draw['port']] = state
            side_summaries.append({
                'port': draw['port'],
                'stage': stage + 1,
                'phase': phase,
                'flow': float(draw_flow),
                'composition': comp,
            })

        component_balance_error = self._external_component_balance_error_vle(
            comps, inlet, feed_z, distillate, bottoms,
            [outlets[draw['port']] for draw in side_draws]
            + ([decanter_purge] if decanter_purge is not None else []),
            flow_scale,
        )
        if component_balance_error > max(1e-5, 10.0 * solver_options['mesh_tolerance']):
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' failed overall component "
                f"balance after convergence (error {component_balance_error:.2e})"
            )

        if quality_context is None:
            final_residual = model['residual'](solution['x'])
        else:
            with quality_context(phase='residual_diagnostics', affects_result=False):
                final_residual = model['residual'](solution['x'])
        diagnostics = self._rigorous2_residual_diagnostics(
            final_residual, model['residual_labels']
        )
        internal_vapor_to_condenser = float(V[1]) if N > 1 else 0.0
        H_in_external = sum(
            feed['F'] * feed['H']
            for feed in feed_specs
        )
        H_out_external = sum(
            stream.F * stream.H
            for stream in outlets.values()
        )

        result = UnitResult(
            outlet_streams=outlets,
            heat_duty=float(H_out_external - H_in_external),
            performance={
                'N_stages': N,
                'feed_stage': feed_stage,
                'feeds': [
                    {
                        'port': feed['port'],
                        'stage': int(feed['stage_number']),
                        'flow': float(feed['F']),
                        'composition': {
                            comp: float(feed['z'].get(comp, 0.0))
                            for comp in comps
                        },
                        'temperature_C': float(feed['T'] - 273.15),
                        'pressure_bar': float(feed['P']),
                        'vapor_fraction': float(feed['vapor_fraction']),
                    }
                    for feed in feed_specs
                ],
                'feed_thermal_condition_q': float(q_feed),
                'stage_phase_model': stage_phase_model,
                'liquid_phase_routing': 'single',
                'stage_phase_counts': [2] * N,
                'stage_spinodal_minimum_eigenvalues': [
                    float(check['minimum_eigenvalue'])
                    for check in stage_spinodal
                ],
                'condenser_type': condenser,
                'distillate_vapor_fraction': float(condenser_vapor_fraction),
                'reflux_ratio': float(L[0] / max(D, 1e-30)),
                'distillate_flow': D,
                'distillate_liquid_flow': float(distillate_liquid_flow),
                'distillate_vapor_flow': float(distillate_vapor_flow),
                'bottoms_flow': B,
                'mesh_residual': float(solution['residual_norm']),
                'component_balance_error': float(component_balance_error),
                'solver': 'sparse_damped_newton',
                'initializer': str(initial.get('initializer', 'unknown')),
                'jacobian_method': str(solution.get('jacobian_method', 'colored_finite_difference')),
                'jacobian_fallback': bool(jacobian_fallback),
                'jacobian_dense_mb': float(model.get('jacobian_dense_mb', 0.0)),
                'solver_iterations': int(solution['iterations']),
                'function_evaluations': int(solution['function_evaluations']),
                'jacobian_evaluations': int(solution['jacobian_evaluations']),
                'finite_difference_rel_step': float(solution['finite_difference_rel_step']),
                'T_top_C': float(T[0] - 273.15),
                'T_bottom_C': float(T[-1] - 273.15),
                'stage_temperatures_C': [float(value - 273.15) for value in T],
                'stage_pressures_bar': [float(value) for value in pressures],
                'stage_liquid_compositions': [
                    {comp: float(x_stage.get(comp, 0.0)) for comp in comps}
                    for x_stage in x
                ],
                'stage_vapor_compositions': [
                    {comp: float(y_stage.get(comp, 0.0)) for comp in comps}
                    for y_stage in y
                ],
                'liquid_flows': [float(value) for value in L],
                'vapor_flows': [float(value) for value in V],
                'internal_vapor_to_condenser': internal_vapor_to_condenser,
                'condenser_duty_kW': float(Q_cond / 3600),
                'reboiler_duty_kW': float(Q_reb / 3600),
                'residual_diagnostics': diagnostics,
                'component_classification': classification,
                'side_draws': side_summaries,
                'distillate_enthalpy_basis': float(distillate_h),
                'top_decanter': decanter_split,
            },
            warnings=warnings,
        )
        self._store_recycle_profile(result.performance)
        return result


    def _stage_phase_model(self) -> str:
        value = self.get_param('stage_phase_model')
        if value is None:
            value = self.get_param('valid_phases')
        if value is None:
            value = self.get_param('stage_phases')
        if value is None:
            value = getattr(self.thermo, 'fluid_phase_model', 'VLE')
        normalized = str(value).strip().upper().replace('-', '').replace('_', '')
        if normalized in ('VLE', 'VL'):
            return 'VLE'
        if normalized in ('VL(L)E', 'VLL(E)', 'ADAPTIVEVLLE', 'SPINODALVLLE'):
            return 'VL(L)E'
        if normalized in ('VLLE', 'VLL'):
            return 'VLLE'
        raise UnitOperationError(
            f"RigorousDistillation '{self.unit_id}' stage_phase_model must be "
            "VLE, VL(L)E, or VLLE"
        )

    def _validate_vlle_stage_configuration(self, condenser, side_draws) -> None:
        if condenser == 'decanter':
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' cannot combine "
                "stage_phase_model=VLLE with a decanter condenser; use a "
                "total, partial, or mixed condenser until separate decanter "
                "phase routing is implemented."
            )
        if side_draws:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' does not yet support "
                "side draws with stage_phase_model=VLLE because liquid-phase "
                "routing must be specified explicitly."
            )
        if not callable(getattr(self.thermo, 'activity_coefficients', None)):
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' requires an activity-"
                "coefficient thermodynamic method for stage_phase_model=VLLE"
            )
        if not callable(getattr(self.thermo, 'liquid_liquid_equilibrium', None)):
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' thermodynamic method "
                "does not provide liquid-liquid stability calculations"
            )

    def _solve_vlle_mode(
        self,
        *,
        inlets,
        inlet,
        feed_specs,
        comps,
        feed_z,
        N,
        feed_stage,
        RR,
        pressures,
        condenser,
        condenser_vapor_fraction,
        distillate_spec,
        T_min,
        T_max,
        q_feed,
        flow_scale,
        energy_scale,
        component_scales,
        classification,
        warnings,
    ) -> UnitResult:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .equilibrium_stage_vlle import (
                        VLLEProfile,
                        VLLESolveFailure,
                        VLLETopologyCycle,
                        solve_vlle_active_set,
                        three_phase_fugacity_residuals,
                        topology_text,
                    )
        else:
            from equilibrium_stage_vlle import (
                        VLLEProfile,
                        VLLESolveFailure,
                        VLLETopologyCycle,
                        solve_vlle_active_set,
                        three_phase_fugacity_residuals,
                        topology_text,
                    )

        seed_mode = str(self.get_param('vlle_seed', 'auto')).strip().lower()
        if seed_mode in ('azeotrope', 'vlle_azeotropic', 'direct_azeotropic'):
            seed_mode = 'azeotropic'
        if seed_mode not in ('auto', 'cheap', 'homogeneous', 'azeotropic'):
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' vlle_seed must be "
                "auto, cheap, homogeneous, or azeotropic"
            )
        homogeneous_initializer = str(self.get_param(
            'vlle_homogeneous_initializer', 'estimate'
        )).strip().lower().replace('-', '_').replace(' ', '_')
        seed_fallback = False
        seed_failure = None
        azeotropic_candidates = []
        azeotropic_search_seconds = 0.0
        candidate_source = None
        profile_attempts = None
        initializer_attempts = []

        recycle_guess = self._recycle_profile_initial_guess(
            comps, N, T_min, T_max
        )
        initial_active = None
        if recycle_guess is not None and recycle_guess.get('vlle_topology'):
            split_data = []
            for stage in range(N):
                if recycle_guess['vlle_topology'][stage] == 'L':
                    split_data.append((
                        dict(recycle_guess['x1'][stage]),
                        dict(recycle_guess['x2'][stage]),
                        float(recycle_guess['beta'][stage]),
                    ))
                else:
                    split_data.append(None)
            profile = VLLEProfile(
                T=list(recycle_guess['T']),
                aggregate_x=[dict(value) for value in recycle_guess['x']],
                L=list(recycle_guess['L']),
                V=list(recycle_guess['V']),
                Q_cond=float(recycle_guess['Q_cond']),
                Q_reb=float(recycle_guess['Q_reb']),
                split_data=split_data,
            )
            initial_active = [
                value == 'L' for value in recycle_guess['vlle_topology']
            ]
            initializer_label = 'previous_recycle'
        else:
            seed_params = dict(self.params)
            seed_params['stage_phase_model'] = 'VLE'
            seed_params['initializer'] = 'estimate'
            seed_unit = RigorousDistillation(
                f"{self.unit_id}_vlle_seed",
                self.thermo,
                seed_params,
            )
            if seed_mode == 'azeotropic':
                azeotropic_candidates = (
                    self._provided_vlle_azeotrope_candidates(comps)
                )
                candidate_source = 'provided'
                if not azeotropic_candidates:
                    search_started = time.perf_counter()
                    azeotropic_candidates = self._vlle_azeotrope_candidates(
                        comps, pressures[0], feed_z=feed_z
                    )
                    azeotropic_search_seconds = (
                        time.perf_counter() - search_started
                    )
                    candidate_source = 'simultaneous_binary_vlle'
                if not azeotropic_candidates:
                    raise UnitOperationError(
                        f"RigorousDistillation '{self.unit_id}' could not find "
                        "a binary VLLE azeotrope at the condenser pressure; "
                        "provide vlle_azeotrope_composition and "
                        "vlle_azeotrope_temperature for a known higher-order "
                        "azeotrope"
                    )
                seed_unit.params['initializer'] = 'azeotropic'
                requested_profile = str(self.get_param(
                    'vlle_azeotropic_profile', 'auto'
                )).strip().lower().replace('-', '_')
                if requested_profile == 'auto':
                    profile_modes = ('linear', 'log_feed_anchor')
                elif requested_profile in ('linear', 'log_feed_anchor'):
                    profile_modes = (requested_profile,)
                else:
                    raise UnitOperationError(
                        f"RigorousDistillation '{self.unit_id}' "
                        "vlle_azeotropic_profile must be auto, linear, or "
                        "log_feed_anchor"
                    )
                profile_attempts = []
                for profile_mode in profile_modes:
                    profile_attempts.append((
                        f'vlle_azeotropic_{profile_mode}',
                        profile_mode,
                        None,
                    ))
                if self.get_param('vlle_topology_policy') is None:
                    for profile_mode in profile_modes:
                        profile_attempts.append((
                            f'vlle_azeotropic_{profile_mode}_residual_gate',
                            profile_mode,
                            None,
                        ))
                initializer_label = profile_attempts[0][0]
                profile = None
            else:
                cheap = seed_unit._initial_guess(
                    inlet,
                    comps,
                    feed_z,
                    N,
                    feed_stage,
                    RR,
                    q_feed,
                    pressures,
                    condenser,
                    condenser_vapor_fraction,
                    distillate_spec,
                    [],
                    T_min,
                    T_max,
                )
                profile = VLLEProfile(
                    T=[float(value) for value in cheap['T']],
                    aggregate_x=[dict(value) for value in cheap['x']],
                    L=[float(value) for value in cheap['L']],
                    V=[float(value) for value in cheap['V']],
                    Q_cond=float(cheap['Q_cond']),
                    Q_reb=float(cheap['Q_reb']),
                    split_data=[None] * N,
                )
                cheap_has_lle = any(
                    self.thermo.liquid_liquid_equilibrium(
                        x_stage,
                        T_stage,
                        max_iter=100,
                        tol=float(self.get_param('vlle_stability_tolerance', 1e-7)),
                    )[0]
                    for T_stage, x_stage in zip(profile.T, profile.aggregate_x)
                )
                initializer_label = 'vlle_cheap'
            if seed_mode == 'homogeneous' or (
                seed_mode == 'auto' and not cheap_has_lle
            ):
                homogeneous_params = dict(seed_params)
                homogeneous_params['initializer'] = homogeneous_initializer
                if seed_mode == 'auto':
                    requested_iterations = int(self.get_param(
                        'max_iterations', self.get_param('max_evaluations', 60)
                    ))
                    requested_jacobians = int(self.get_param(
                        'max_jacobian_evaluations', 60
                    ))
                    homogeneous_params['max_iterations'] = min(
                        requested_iterations,
                        max(1, int(self.get_param(
                            'vlle_auto_homogeneous_max_iterations', 24
                        ))),
                    )
                    homogeneous_params['max_jacobian_evaluations'] = min(
                        requested_jacobians,
                        max(1, int(self.get_param(
                            'vlle_auto_homogeneous_max_jacobian_evaluations', 24
                        ))),
                    )
                    homogeneous_params['newton_stall_iterations'] = int(
                        self.get_param(
                            'vlle_auto_homogeneous_stall_iterations', 5
                        )
                    )
                    homogeneous_params['newton_stall_relative_tolerance'] = float(
                        self.get_param(
                            'vlle_auto_homogeneous_stall_relative_tolerance',
                            1e-4,
                        )
                    )
                    homogeneous_params['finite_difference_rel_step'] = float(
                        self.get_param(
                            'vlle_auto_homogeneous_finite_difference_rel_step',
                            1e-6,
                        )
                    )
                    homogeneous_params['colored_jacobian_fallback'] = (
                        self._truthy_param(self.get_param(
                            'vlle_auto_homogeneous_colored_fallback', False
                        ))
                    )
                homogeneous_unit = RigorousDistillation(
                    f"{self.unit_id}_vlle_seed",
                    self.thermo,
                    homogeneous_params,
                )
                try:
                    homogeneous = homogeneous_unit.solve(inlets)
                except UnitOperationError as exc:
                    if (
                        seed_mode == 'homogeneous'
                        or 'MESH solve failed' not in str(exc)
                    ):
                        raise
                    seed_fallback = True
                    seed_failure = str(exc)
                    initializer_label = 'vlle_cheap_fallback'
                    warnings.append(
                        "Automatic homogeneous VLLE seed failed; continued "
                        f"with the cheap VLLE seed ({exc})."
                    )
                else:
                    performance = homogeneous.performance
                    profile = VLLEProfile(
                        T=[
                            float(value) + 273.15
                            for value in performance['stage_temperatures_C']
                        ],
                        aggregate_x=[
                            {comp: float(stage.get(comp, 0.0)) for comp in comps}
                            for stage in performance['stage_liquid_compositions']
                        ],
                        L=[float(value) for value in performance['liquid_flows']],
                        V=[float(value) for value in performance['vapor_flows']],
                        Q_cond=float(performance['condenser_duty_kW']) * 3600.0,
                        Q_reb=float(performance['reboiler_duty_kW']) * 3600.0,
                        split_data=[None] * N,
                    )
                    initializer_label = 'vlle_homogeneous'

        solver_options = {
            'mesh_tolerance': float(self.get_param('mesh_tolerance', 2e-6)),
            'acceptable_mesh_residual': float(self.get_param(
                'acceptable_mesh_residual',
                max(50.0 * float(self.get_param('mesh_tolerance', 2e-6)), 2e-6),
            )),
            'max_iterations': int(self.get_param(
                'max_iterations', self.get_param('max_evaluations', 60)
            )),
            'max_jacobian_evaluations': int(self.get_param(
                'max_jacobian_evaluations', 60
            )),
            'line_search_steps': int(self.get_param('line_search_steps', 16)),
            'finite_difference_rel_step': float(self.get_param(
                'finite_difference_rel_step', 1e-6
            )),
        }
        if profile_attempts is None:
            profile_attempts = [(initializer_label, profile, initial_active)]
        solved = None
        last_exception = None
        selected_topology_policy = str(self.get_param(
            'vlle_topology_policy', 'adaptive'
        ))
        for attempt_label, attempt_profile, attempt_active in profile_attempts:
            residual_gate_retry = attempt_label.endswith('_residual_gate')
            if residual_gate_retry and not any(
                item.get('topology_cycle_full', False)
                for item in initializer_attempts
            ):
                continue
            seed_details = {}
            if isinstance(attempt_profile, str):
                direct = seed_unit._initial_guess(
                    inlet,
                    comps,
                    feed_z,
                    N,
                    feed_stage,
                    RR,
                    q_feed,
                    pressures,
                    condenser,
                    condenser_vapor_fraction,
                    distillate_spec,
                    [],
                    T_min,
                    T_max,
                    azeotrope_candidates=azeotropic_candidates,
                    azeotropic_profile=attempt_profile,
                )
                attempt_profile = VLLEProfile(
                    T=[float(value) for value in direct['T']],
                    aggregate_x=[dict(value) for value in direct['x']],
                    L=[float(value) for value in direct['L']],
                    V=[float(value) for value in direct['V']],
                    Q_cond=float(direct['Q_cond']),
                    Q_reb=float(direct['Q_reb']),
                    split_data=[None] * N,
                )
                seed_details = {
                    'profile_top_composition': dict(direct['x'][0]),
                    'profile_bottom_composition': dict(direct['x'][-1]),
                    'azeotropic_endpoints': dict(
                        direct['azeotropic_endpoints']
                    ),
                    'pseudo_components': [
                        dict(item)
                        for item in direct['azeotropic_pseudo_components']
                    ],
                }
            previous_topology_policy = self.params.get('vlle_topology_policy')
            if residual_gate_retry:
                self.params['vlle_topology_policy'] = 'residual_gate'
            try:
                solved = solve_vlle_active_set(
                    self,
                    inlet,
                    feed_specs,
                    comps,
                    pressures,
                    RR,
                    condenser_vapor_fraction,
                    distillate_spec,
                    T_min,
                    T_max,
                    flow_scale,
                    energy_scale,
                    component_scales,
                    attempt_profile,
                    solver_options,
                    initial_active=attempt_active,
                )
            except RuntimeError as exc:
                last_exception = exc
                initializer_attempts.append({
                    'initializer': attempt_label,
                    'success': False,
                    'error': str(exc),
                    'topology_cycle_full': (
                        isinstance(exc, VLLETopologyCycle)
                        and 'L' * N in exc.history
                    ),
                    'work': dict(exc.work) if isinstance(exc, VLLESolveFailure) else {},
                    **seed_details,
                })
                numerical_failure = isinstance(exc, VLLESolveFailure)
                if not numerical_failure:
                    break
                continue
            finally:
                if residual_gate_retry:
                    if previous_topology_policy is None:
                        self.params.pop('vlle_topology_policy', None)
                    else:
                        self.params['vlle_topology_policy'] = (
                            previous_topology_policy
                        )
            initializer_label = attempt_label
            if residual_gate_retry:
                selected_topology_policy = 'residual_gate'
            initializer_attempts.append({
                'initializer': attempt_label,
                'success': True,
                'error': None,
                'initial_topology': solved.topology_history[0],
                'work': dict(solved.work),
                **seed_details,
            })
            break
        if solved is None:
            attempted = '; '.join(
                f"{item['initializer']}: {item['error']}"
                for item in initializer_attempts
            )
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' VLLE MESH solve failed "
                f"after initializer attempt(s): {attempted}"
            ) from last_exception

        # These counters cover all VLLE MESH attempts, including failed
        # profiles and topology/Jacobian fallbacks. Candidate search and the
        # optional homogeneous VLE seed are separate initialization work.
        solver_work = {
            name: sum(attempt['work'][name] for attempt in initializer_attempts)
            for name in solved.work
        }
        decoded = solved.decoded
        stages = decoded['stages']
        stage_props = solved.stage_properties
        T = [float(stage['T']) for stage in stages]
        x = [dict(stage['aggregate_x']) for stage in stages]
        x1 = [dict(stage['x1']) for stage in stages]
        x2 = [dict(stage['x2']) for stage in stages]
        liquid2_fraction = [float(stage['beta']) for stage in stages]
        L = [float(stage['L']) for stage in stages]
        V = [float(stage['V']) for stage in stages]
        y = [dict(item['y']) for item in stage_props]
        hL = [float(item['hL']) for item in stage_props]
        hV = [float(item['hV']) for item in stage_props]
        Q_cond = float(decoded['Q_cond'])
        Q_reb = float(decoded['Q_reb'])
        solution = solved.solver
        solution['finite_difference_rel_step'] = solver_options['finite_difference_rel_step']
        jacobian_fallback = (
            solution.get('jacobian_method') == 'colored_finite_difference'
        )
        if jacobian_fallback:
            warnings.append(
                "VLLE local/semi-analytic Jacobian did not converge; retried "
                "the fixed topology with colored finite differences."
            )
        if solution['residual_norm'] > solver_options['mesh_tolerance']:
            warnings.append(
                f"VLLE MESH solver stalled near tolerance and accepted residual "
                f"{solution['residual_norm']:.2e}"
            )

        fugacity_residuals = []
        for stage, is_active in enumerate(solved.active):
            if not is_active:
                continue
            residuals = three_phase_fugacity_residuals(
                self.thermo,
                T[stage],
                pressures[stage],
                x1[stage],
                x2[stage],
                y[stage],
                comps,
            )
            fugacity_residuals.append((stage + 1, residuals))
        max_liquid_liquid_fugacity_residual = max(
            (item['liquid_liquid'] for _, item in fugacity_residuals),
            default=0.0,
        )
        max_vapor_liquid_fugacity_residual = max(
            (item['vapor_liquid'] for _, item in fugacity_residuals),
            default=0.0,
        )
        max_three_phase_fugacity_residual = max(
            max_liquid_liquid_fugacity_residual,
            max_vapor_liquid_fugacity_residual,
        )
        max_fugacity_residual_stage = next(
            (
                stage for stage, item in fugacity_residuals
                if item['overall'] == max_three_phase_fugacity_residual
            ),
            None,
        )
        if max_three_phase_fugacity_residual > solver_options['acceptable_mesh_residual']:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' failed the post-solve "
                f"three-phase fugacity audit on stage {max_fugacity_residual_stage} "
                f"(log-fugacity residual {max_three_phase_fugacity_residual:.2e})"
            )
        if max_three_phase_fugacity_residual > solver_options['mesh_tolerance']:
            warnings.append(
                f"VLLE three-phase fugacity audit accepted residual "
                f"{max_three_phase_fugacity_residual:.2e} on stage "
                f"{max_fugacity_residual_stage}"
            )

        effective_K_top = {
            comp: y[0].get(comp, 0.0) / max(x[0].get(comp, 0.0), 1e-30)
            for comp in comps
        }
        warnings.extend(self._post_solve_noncondensable_check(
            comps,
            feed_z,
            T[0],
            pressures[0],
            x[0],
            y[0],
            effective_K_top,
            condenser_vapor_fraction,
        ))

        D = V[0]
        B = L[-1]
        distillate_liquid_flow = (1.0 - condenser_vapor_fraction) * D
        distillate_vapor_flow = condenser_vapor_fraction * D

        def liquid_state(stage: int, flow: float, composition: dict[str, float]):
            state = self.thermo.calculate_state(
                T[stage],
                pressures[stage],
                flow,
                composition,
                phase='liquid',
                flash=False,
            )
            state.H = hL[stage]
            try:
                state.Cp = (
                    (1.0 - liquid2_fraction[stage])
                    * self.thermo.mixture_Cp(x1[stage], T[stage], 0.0, P=pressures[stage])
                    + liquid2_fraction[stage]
                    * self.thermo.mixture_Cp(x2[stage], T[stage], 0.0, P=pressures[stage])
                )
            except Exception:
                pass
            return state

        if condenser_vapor_fraction >= 1.0 - 1e-12:
            distillate_comp = dict(y[0])
            distillate_h = hV[0]
            distillate = self.thermo.calculate_state(
                T[0], pressures[0], D, distillate_comp,
                phase='vapor', flash=False,
            )
        elif condenser_vapor_fraction <= 1e-12:
            distillate_comp = dict(x[0])
            distillate_h = hL[0]
            distillate = liquid_state(0, D, distillate_comp)
        else:
            distillate_comp = self._normalize({
                comp: (
                    distillate_liquid_flow * x[0].get(comp, 0.0)
                    + distillate_vapor_flow * y[0].get(comp, 0.0)
                )
                for comp in comps
            })
            distillate_h = (
                (1.0 - condenser_vapor_fraction) * hL[0]
                + condenser_vapor_fraction * hV[0]
            )
            distillate = self._two_phase_stream(
                T[0], pressures[0], D, distillate_comp,
                x[0], y[0], condenser_vapor_fraction, distillate_h,
            )
        bottoms = liquid_state(N - 1, B, x[-1])
        outlets = {'distillate': distillate, 'bottoms': bottoms}
        if 1e-12 < condenser_vapor_fraction < 1.0 - 1e-12:
            outlets['distillate_liquid'] = liquid_state(
                0, distillate_liquid_flow, x[0]
            )
            outlets['distillate_vapor'] = self.thermo.calculate_state(
                T[0], pressures[0], distillate_vapor_flow, y[0],
                phase='vapor', flash=False,
            )

        component_balance_error = self._external_component_balance_error_vle(
            comps,
            inlet,
            feed_z,
            distillate,
            bottoms,
            [],
            flow_scale,
        )
        if component_balance_error > max(
            1e-5, 10.0 * solver_options['mesh_tolerance']
        ):
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' failed overall component "
                f"balance after VLLE convergence (error {component_balance_error:.2e})"
            )

        H_in_external = sum(feed['F'] * feed['H'] for feed in feed_specs)
        # Mixed-condenser phase outlets are views of the combined distillate,
        # not additional external products; count the combined stream once.
        H_out_external = distillate.F * distillate.H + bottoms.F * bottoms.H
        active_stages = [index + 1 for index, active in enumerate(solved.active) if active]
        phase_counts = [3 if active else 2 for active in solved.active]
        performance = {
            'N_stages': N,
            'feed_stage': feed_stage,
            'feeds': [
                {
                    'port': feed['port'],
                    'stage': int(feed['stage_number']),
                    'flow': float(feed['F']),
                    'composition': {
                        comp: float(feed['z'].get(comp, 0.0)) for comp in comps
                    },
                    'temperature_C': float(feed['T'] - 273.15),
                    'pressure_bar': float(feed['P']),
                    'vapor_fraction': float(feed['vapor_fraction']),
                }
                for feed in feed_specs
            ],
            'feed_thermal_condition_q': float(q_feed),
            'condenser_type': condenser,
            'distillate_vapor_fraction': float(condenser_vapor_fraction),
            'reflux_ratio': float(L[0] / max(D, 1e-30)),
            'distillate_flow': float(D),
            'distillate_liquid_flow': float(distillate_liquid_flow),
            'distillate_vapor_flow': float(distillate_vapor_flow),
            'bottoms_flow': float(B),
            'mesh_residual': float(solution['residual_norm']),
            'component_balance_error': float(component_balance_error),
            'solver': 'sparse_damped_newton_vlle_active_set',
            'initializer': initializer_label,
            'vlle_seed_requested': seed_mode,
            'vlle_homogeneous_initializer': homogeneous_initializer,
            'vlle_seed_fallback': bool(seed_fallback),
            'vlle_seed_failure': seed_failure,
            'vlle_initializer_attempts': list(initializer_attempts),
            'vlle_azeotropic_candidates': [
                {
                    'name': str(item['name']),
                    'type': str(item.get('type', 'VLE')),
                    'order': int(item['order']),
                    'temperature_C': float(item['T'] - 273.15),
                    'composition': dict(item['composition']),
                    'fixed_point_residual': float(item.get(
                        'fixed_point_residual', 0.0
                    )),
                    'fugacity_residual': float(item.get(
                        'fugacity_residual', 0.0
                    )),
                    'function_evaluations': int(item.get(
                        'function_evaluations', 0
                    )),
                }
                for item in azeotropic_candidates
            ],
            'vlle_azeotropic_candidate_source': (
                candidate_source if seed_mode == 'azeotropic' else None
            ),
            'vlle_azeotropic_search_seconds': float(
                azeotropic_search_seconds
            ),
            'vlle_azeotropic_profile': str(self.get_param(
                'vlle_azeotropic_profile', 'auto'
            )),
            'vlle_auto_homogeneous_max_iterations': int(self.get_param(
                'vlle_auto_homogeneous_max_iterations', 24
            )),
            'vlle_auto_homogeneous_max_jacobian_evaluations': int(
                self.get_param(
                    'vlle_auto_homogeneous_max_jacobian_evaluations', 24
                )
            ),
            'vlle_auto_homogeneous_stall_iterations': int(self.get_param(
                'vlle_auto_homogeneous_stall_iterations', 5
            )),
            'vlle_auto_homogeneous_stall_relative_tolerance': float(
                self.get_param(
                    'vlle_auto_homogeneous_stall_relative_tolerance', 1e-4
                )
            ),
            'vlle_auto_homogeneous_colored_fallback': self._truthy_param(
                self.get_param(
                    'vlle_auto_homogeneous_colored_fallback', False
                )
            ),
            'jacobian_method': str(solution.get(
                'jacobian_method', 'vlle_semi_analytic_local_thermo'
            )),
            'jacobian_fallback': bool(jacobian_fallback),
            'jacobian_dense_mb': 0.0,
            **solver_work,
            'solver_work_basis': 'all_vlle_mesh_attempts',
            'finite_difference_rel_step': float(solution['finite_difference_rel_step']),
            'T_top_C': float(T[0] - 273.15),
            'T_bottom_C': float(T[-1] - 273.15),
            'stage_temperatures_C': [float(value - 273.15) for value in T],
            'stage_pressures_bar': [float(value) for value in pressures],
            'stage_liquid_compositions': [dict(value) for value in x],
            'stage_vapor_compositions': [dict(value) for value in y],
            'liquid_flows': [float(value) for value in L],
            'vapor_flows': [float(value) for value in V],
            'internal_vapor_to_condenser': float(V[1]) if N > 1 else 0.0,
            'condenser_duty_kW': float(Q_cond / 3600.0),
            'reboiler_duty_kW': float(Q_reb / 3600.0),
            'residual_diagnostics': {
                'max_abs_residual': float(solution['residual_norm']),
            },
            'component_classification': classification,
            'side_draws': [],
            'distillate_enthalpy_basis': float(distillate_h),
            'top_decanter': None,
            'stage_phase_model': 'VLLE',
            'liquid_phase_routing': 'co_routed_equilibrium',
            'stage_phase_counts': phase_counts,
            'stage_liquid1_compositions': x1,
            'stage_liquid2_compositions': x2,
            'stage_liquid2_fractions': liquid2_fraction,
            'stage_liquid1_flows': [
                (1.0 - beta) * flow for beta, flow in zip(liquid2_fraction, L)
            ],
            'stage_liquid2_flows': [
                beta * flow for beta, flow in zip(liquid2_fraction, L)
            ],
            'vlle_active_stages': active_stages,
            'vlle_topology': topology_text(solved.active),
            'vlle_topology_history': list(solved.topology_history),
            'vlle_topology_events': list(solved.topology_events),
            'vlle_topology_policy': selected_topology_policy,
            'vlle_projection_gate_fraction': float(self.get_param(
                'vlle_projection_gate_fraction', 0.15
            )),
            'vlle_projection_enabled': self._truthy_param(self.get_param(
                'vlle_projection_enabled', True
            )),
            'vlle_projection_contraction_ratio': float(self.get_param(
                'vlle_projection_contraction_ratio', 0.5
            )),
            'vlle_projection_candidate_streak': int(self.get_param(
                'vlle_projection_candidate_streak', 2
            )),
            'vlle_projection_phase_fraction_min': float(self.get_param(
                'vlle_phase_fraction_min', 1e-6
            )),
            'vlle_final_topology_projection_checks': int(
                solved.final_topology_projection_checks
            ),
            'vlle_vapor_fugacity_closure': (
                'shared_gamma_phi'
                if getattr(self.thermo, 'vapor_eos', None) is not None
                else 'ideal_vapor_exact'
            ),
            'vlle_max_log_fugacity_residual': float(
                max_three_phase_fugacity_residual
            ),
            'vlle_max_liquid_liquid_log_fugacity_residual': float(
                max_liquid_liquid_fugacity_residual
            ),
            'vlle_max_vapor_liquid_log_fugacity_residual': float(
                max_vapor_liquid_fugacity_residual
            ),
            'vlle_max_fugacity_residual_stage': max_fugacity_residual_stage,
        }
        result = UnitResult(
            outlet_streams=outlets,
            heat_duty=float(H_out_external - H_in_external),
            performance=performance,
            warnings=warnings,
        )
        self._store_recycle_profile(performance)
        return result


    def _distillation_cut_reference_component(
        self,
        inlet: StreamState,
        comps: list[str],
    ) -> Optional[str]:
        if not comps:
            return None
        fallback = max(comps, key=lambda comp: inlet.composition.get(comp, 0.0))
        try:
            feed_z = self._normalize({
                comp: inlet.composition.get(comp, 0.0)
                for comp in comps
            })
            spec = _distillate_flow_spec_from_params(self, inlet, default_fraction=0.5)
            if spec['kind'] == 'mass':
                inlet_mass = _inlet_mass_flow(self, inlet)
                if inlet_mass <= 0.0:
                    return fallback
                cut = float(spec['value']) / inlet_mass
                mixture_mw = sum(
                    feed_z.get(comp, 0.0) * self.thermo.props[comp].MW
                    for comp in comps
                )
                if mixture_mw <= 0.0:
                    return fallback
                weights = {
                    comp: feed_z.get(comp, 0.0) * self.thermo.props[comp].MW / mixture_mw
                    for comp in comps
                }
            else:
                if inlet.F <= 0.0:
                    return fallback
                cut = float(spec['value']) / inlet.F
                weights = dict(feed_z)
            if not math.isfinite(cut) or not 0.0 < cut < 1.0:
                return fallback

            K = self.thermo.K_values(inlet.T, inlet.P, feed_z)
            ranked = sorted(
                comps,
                key=lambda comp: float(K.get(comp, 0.0)),
                reverse=True,
            )
            if not ranked or not all(math.isfinite(float(K.get(comp, 0.0))) for comp in ranked):
                return fallback

            cumulative = 0.0
            for comp in ranked:
                cumulative += max(float(weights.get(comp, 0.0)), 0.0)
                if cumulative >= cut:
                    return comp
            return ranked[-1]
        except Exception:
            return fallback

    def _condenser_vapor_fraction(self, condenser: str) -> float:
        if condenser == 'total':
            return 0.0
        if condenser == 'decanter':
            return 0.0
        if condenser == 'partial':
            return 1.0
        value = self.get_param(
            'distillate_vapor_fraction',
            self.get_param(
                'overhead_vapor_fraction',
                self.get_param('vapor_distillate_fraction', 0.5),
            ),
        )
        fraction = float(value)
        if not 0.0 < fraction < 1.0:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' mixed condenser requires "
                "0 < distillate_vapor_fraction < 1"
            )
        return fraction

    def _decanter_options(self) -> dict:
        reflux_phase = str(self.get_param('decanter_reflux_phase', '') or '').strip().lower()
        distillate_phase = str(self.get_param('decanter_distillate_phase', '') or '').strip().lower()
        reflux_component = self.get_param(
            'decanter_reflux_component',
            self.get_param('reflux_phase_component'),
        )
        distillate_component = self.get_param(
            'decanter_distillate_component',
            self.get_param('distillate_phase_component'),
        )
        purge_fraction = float(self.get_param(
            'decanter_reflux_purge_fraction',
            self.get_param('reflux_purge_fraction', 0.0),
        ))
        if not 0.0 <= purge_fraction < 1.0:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' decanter reflux purge fraction "
                "must be in [0, 1)"
            )
        T_spec = self.get_param(
            'decanter_T',
            self.get_param('condenser_T', self.get_param('T_condenser')),
        )
        return {
            'reflux_phase': reflux_phase,
            'distillate_phase': distillate_phase,
            'reflux_component': str(reflux_component).strip() if reflux_component else None,
            'distillate_component': str(distillate_component).strip() if distillate_component else None,
            'purge_fraction': purge_fraction,
            'T_spec': None if T_spec is None else float(T_spec),
        }

    def _distillate_spec(self, inlet: StreamState) -> dict:
        spec = _distillate_flow_spec_from_params(self, inlet, default_fraction=0.5)
        if spec['kind'] == 'molar':
            if not 0.0 < float(spec['value']) < inlet.F:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' distillate flow is infeasible"
                )
            return spec

        inlet_mass = _inlet_mass_flow(self, inlet)
        if not 0.0 < float(spec['value']) < inlet_mass:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' distillate mass flow is infeasible"
            )
        return spec

    def _decanter_distillate_guess(self, inlet: StreamState) -> dict:
        for name in ('decanter_distillate_guess', 'distillate_flow_guess', 'D_guess'):
            value = self.get_param(name)
            if value is None:
                continue
            guess = float(value)
            if guess <= 0.0:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' {name} must be positive"
                )
            return {'kind': 'decanter', 'value': min(guess, 0.95 * inlet.F), 'source': name}
        D_to_F = self.get_param('D_to_F')
        if D_to_F is not None:
            guess = inlet.F * float(D_to_F)
        else:
            guess = 0.35 * inlet.F
        if not 0.0 < guess < inlet.F:
            guess = 0.35 * inlet.F
        return {'kind': 'decanter', 'value': guess, 'source': 'decanter_guess'}

    def _validate_external_flow_specs(
        self,
        inlet: StreamState,
        distillate_spec: dict,
        side_draws: list[dict],
    ) -> None:
        fixed_side_flow = sum(
            float(draw['flow']) for draw in side_draws if draw.get('flow') is not None
        )
        if fixed_side_flow >= inlet.F:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' fixed side-draw flows "
                "must be less than the feed flow"
            )
        if distillate_spec['kind'] == 'molar':
            external_fixed_flow = float(distillate_spec['value']) + fixed_side_flow
            if external_fixed_flow >= inlet.F:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' fixed distillate and side-draw "
                    "flows must leave a positive bottoms flow"
                )

        fraction_sums: dict[tuple[int, str], float] = {}
        for draw in side_draws:
            if draw.get('fraction') is None:
                continue
            key = (draw['stage'], draw['phase'])
            fraction_sums[key] = fraction_sums.get(key, 0.0) + float(draw['fraction'])
        for (stage, phase), fraction in fraction_sums.items():
            if fraction >= 1.0:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' side draw fractions on "
                    f"stage {stage + 1} {phase} phase must sum to less than 1"
                )

    def _classify_components(
        self,
        comps: list[str],
        feed_z: dict[str, float],
        P_top: float,
        inlet: StreamState,
    ) -> dict:
        condensable_z = {
            comp: frac for comp, frac in feed_z.items()
            if self._is_likely_condensable_candidate(comp)
        }
        if sum(condensable_z.values()) > 1e-10:
            condensable_z = self._normalize(condensable_z)
        else:
            condensable_z = dict(feed_z)

        try:
            T_ref = self._bubble_temperature_from_equation(condensable_z, P_top)
        except Exception:
            T_ref = inlet.T

        noncondensables = []
        volatile = []
        heavy = []
        min_feed_fraction = float(self.get_param('noncondensable_min_feed_fraction', 1e-8))
        for comp in comps:
            props = self.thermo.props.get(comp)
            Tb = getattr(props, 'Tb', None) if props else None
            Tc = getattr(props, 'Tc', None) if props else None
            if Tc is not None and Tc < T_ref + 5.0 and feed_z.get(comp, 0.0) > min_feed_fraction:
                noncondensables.append(comp)
                continue
            if Tb is not None and Tb < T_ref - 50.0:
                volatile.append(comp)
            elif Tb is not None and Tb > T_ref + 80.0:
                heavy.append(comp)

        return {
            'reference_condenser_temperature_K': float(T_ref),
            'reference_condenser_temperature_C': float(T_ref - 273.15),
            'noncondensables': noncondensables,
            'volatile_components': volatile,
            'heavy_components': heavy,
        }

    def _post_solve_noncondensable_check(
        self,
        comps: list[str],
        feed_z: dict[str, float],
        T_top: float,
        P_top: float,
        x_top: dict[str, float],
        y_top: dict[str, float],
        K_top: dict[str, float],
        condenser_vapor_fraction: float,
    ) -> list[str]:
        min_feed_fraction = float(self.get_param('noncondensable_min_feed_fraction', 1e-8))
        K_threshold = float(self.get_param('noncondensable_K_threshold', 100.0))
        suspects = []
        for comp in comps:
            if feed_z.get(comp, 0.0) <= min_feed_fraction:
                continue
            props = self.thermo.props.get(comp)
            Tc = getattr(props, 'Tc', None) if props else None
            Tb = getattr(props, 'Tb', None) if props else None
            K_i = K_top.get(comp, 1.0)
            vapor_enrichment = y_top.get(comp, 0.0) / max(x_top.get(comp, 0.0), 1e-30)
            supercritical = Tc is not None and Tc < T_top + 5.0
            extremely_volatile = (
                K_i > K_threshold
                and vapor_enrichment > 10.0
                and (Tb is None or Tb < T_top - 60.0)
            )
            if supercritical or extremely_volatile:
                suspects.append(comp)

        if not suspects:
            return []

        names = ', '.join(suspects)
        if condenser_vapor_fraction <= 0.0:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' condenser solution indicates "
                f"noncondensables likely present: {names}. Use a partial or mixed "
                "condenser with a vapor distillate/vent, or remove these components "
                "before a total condenser."
            )
        return [
            f"Condenser vapor outlet contains likely non-condensable component(s): {names}"
        ]

    def _initial_guess(
        self,
        inlet: StreamState,
        comps: list[str],
        feed_z: dict[str, float],
        N: int,
        feed_stage: int,
        RR: float,
        q_feed: float,
        pressures: list[float],
        condenser: str,
        condenser_vapor_fraction: float,
        distillate_spec: dict,
        side_draws: list[dict],
        T_min: float,
        T_max: float,
        azeotrope_candidates: Optional[list[dict]] = None,
        azeotropic_profile: str = 'linear',
    ) -> dict:
        recycle_guess = self._recycle_profile_initial_guess(comps, N, T_min, T_max)
        if recycle_guess is not None:
            return recycle_guess

        initializer_value = self.get_param('initializer', self.get_param('initialization'))
        initializer = str(
            initializer_value if initializer_value is not None else 'auto'
        ).strip().lower().replace('-', '_').replace(' ', '_')
        explicit_initializer = (
            initializer_value is not None
            and initializer not in ('default', 'auto')
        )

        if initializer in ('default', 'auto'):
            initializer = (
                'estimate'
                if len(comps) == 2 or condenser == 'decanter'
                else 'coarse_rigorous'
            )

        if initializer in ('coarse', 'coarse_rigorous', 'coarse_grid'):
            coarse_guess = self._coarse_grid_initial_guess(
                inlet, comps, N, feed_stage, pressures, side_draws, T_min, T_max
            )
            if coarse_guess is not None:
                return coarse_guess
            if explicit_initializer:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' could not build a "
                    "coarse_rigorous initializer for this column"
                )
            initializer = '_legacy_cmo_estimate'

        x_top = dict(feed_z)
        x_bottom = dict(feed_z)
        D_guess = self._distillate_molar_guess(distillate_spec, inlet, x_top)
        fixed_side_flow = sum(draw['flow'] or 0.0 for draw in side_draws)
        B_guess = max(inlet.F - D_guess - fixed_side_flow, inlet.F * 1e-6, 1e-9)
        azeotropic_endpoints_active = False
        azeotropic_endpoint_data = None

        if initializer in (
            'azeotropic', 'azeotrope', 'azeotropic_cmo', 'azeotrope_cmo',
            'azeotropic_initializer',
        ):
            try:
                endpoints = self._azeotropic_endpoint_initial_guess(
                    inlet,
                    comps,
                    distillate_spec,
                    N,
                    RR,
                    pressures[0],
                    candidates=azeotrope_candidates,
                )
                if endpoints is not None:
                    x_top = endpoints['x_top']
                    x_bottom = endpoints['x_bottom']
                    D_guess = max(endpoints['D'], 1e-9)
                    B_guess = max(endpoints['B'], 1e-9)
                    azeotropic_endpoints_active = True
                    azeotropic_endpoint_data = endpoints
                elif explicit_initializer:
                    raise UnitOperationError(
                        f"RigorousDistillation '{self.unit_id}' could not build an "
                        "azeotropic initializer for this column"
                    )
            except Exception as exc:
                if explicit_initializer:
                    raise UnitOperationError(
                        f"RigorousDistillation '{self.unit_id}' azeotropic initializer failed: {exc}"
                    ) from exc
        elif initializer in ('cmo', 'cmo_distillation', 'cmo_initializer', 'cmo_hvap'):
            return self._cmo_profile_initial_guess(
                inlet, comps, N, feed_stage, RR, q_feed, pressures, condenser,
                distillate_spec, T_min, T_max,
                latent_heat_correction=(initializer == 'cmo_hvap'),
            )
        elif initializer == '_legacy_cmo_estimate':
            endpoints = self._legacy_cmo_endpoint_initial_guess(
                inlet, comps, N, RR, q_feed, pressures, distillate_spec,
            )
            if endpoints is not None:
                x_top = endpoints['x_top']
                x_bottom = endpoints['x_bottom']
                D_guess = max(endpoints['D'], 1e-9)
                B_guess = max(endpoints['B'], 1e-9)
        elif initializer in ('estimate', 'cheap', 'cheap_estimate', 'smooth', 'smooth_estimate'):
            pass
        else:
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' initializer must be "
                "estimate, coarse_rigorous, cmo, cmo_hvap, or azeotropic"
            )

        T = []
        x = []
        azeotropic_profile = str(azeotropic_profile).strip().lower().replace(
            '-', '_'
        )
        if azeotropic_profile not in ('linear', 'log_feed_anchor'):
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' azeotropic_profile must "
                "be linear or log_feed_anchor"
            )
        feed_index = min(max(feed_stage - 1, 1), N - 2)

        def log_interpolate(left, right, fraction):
            return self._normalize({
                comp: math.exp(
                    (1.0 - fraction)
                    * math.log(max(left.get(comp, 0.0), 1e-12))
                    + fraction
                    * math.log(max(right.get(comp, 0.0), 1e-12))
                )
                for comp in comps
            })

        for stage in range(N):
            frac = stage / max(N - 1, 1)
            if (
                azeotropic_endpoints_active
                and azeotropic_profile == 'log_feed_anchor'
                and N > 2
            ):
                if stage <= feed_index:
                    section_frac = stage / feed_index
                    left, right = x_top, feed_z
                else:
                    section_frac = (stage - feed_index) / (N - 1 - feed_index)
                    left, right = feed_z, x_bottom
                x_stage = log_interpolate(left, right, section_frac)
            else:
                x_stage = {
                    comp: (
                        (1.0 - frac) * x_top.get(comp, 0.0)
                        + frac * x_bottom.get(comp, 0.0)
                    )
                    for comp in comps
                }
            x_stage = self._normalize({comp: max(value, 1e-10) for comp, value in x_stage.items()})

            condensable_stage = {
                comp: x_stage.get(comp, 0.0)
                for comp in comps
                if self._is_likely_condensable_candidate(comp)
            }
            if sum(condensable_stage.values()) > 1e-10:
                bubble_composition = self._normalize(condensable_stage)
            else:
                bubble_composition = x_stage
            try:
                T_stage = self._bubble_temperature_from_equation(
                    bubble_composition, pressures[stage], T_min, T_max
                )
            except Exception:
                T_stage = (1.0 - frac) * inlet.T + frac * max(inlet.T, self.thermo.props[comps[-1]].Tb or inlet.T)
            T_stage = min(max(float(T_stage), T_min + 1e-6), T_max - 1e-6)

            noncond_liquid = {}
            for comp in comps:
                props = self.thermo.props.get(comp)
                Tc = getattr(props, 'Tc', None) if props else None
                if Tc is None or Tc >= T_stage + 5.0:
                    continue
                try:
                    K_guess = max(self.thermo.Psat(comp, T_stage) / pressures[stage], 1.0)
                except Exception:
                    K_guess = 1e3
                noncond_liquid[comp] = min(
                    x_stage.get(comp, 0.0),
                    max(feed_z.get(comp, 0.0) / K_guess, 1e-14),
                )
            noncond_total = min(sum(noncond_liquid.values()), 0.2)
            if noncond_liquid and noncond_total < sum(
                x_stage.get(comp, 0.0) for comp in noncond_liquid
            ):
                condensable_norm = self._normalize({
                    comp: x_stage.get(comp, 0.0)
                    for comp in comps
                    if comp not in noncond_liquid
                })
                x_stage = {
                    comp: condensable_norm.get(comp, 0.0) * (1.0 - noncond_total)
                    + noncond_liquid.get(comp, 0.0)
                    for comp in comps
                }
                x_stage = self._normalize(x_stage)

            x.append(x_stage)
            T.append(T_stage)

        L = []
        V = []
        top_internal_vapor = max((RR + 1.0) * D_guess, inlet.F * 1e-6)
        for stage in range(N):
            if stage == 0:
                L_stage = max(RR * D_guess, inlet.F * 1e-8)
                V_stage = max(D_guess, inlet.F * 1e-8)
            elif stage < feed_stage - 1:
                L_stage = max(RR * D_guess, inlet.F * 1e-8)
                V_stage = top_internal_vapor
            elif stage < N - 1:
                L_stage = max(RR * D_guess + q_feed * inlet.F, inlet.F * 1e-8)
                V_stage = max(top_internal_vapor - (1.0 - q_feed) * inlet.F, inlet.F * 1e-8)
            else:
                L_stage = max(B_guess, inlet.F * 1e-8)
                V_stage = max(top_internal_vapor - (1.0 - q_feed) * inlet.F, inlet.F * 1e-8)
            L.append(L_stage)
            V.append(V_stage)

        if N > 1:
            V[1] = max(L[0] + D_guess, inlet.F * 1e-8)

        h_feed = inlet.H or self.thermo.mixture_enthalpy(
            feed_z, inlet.T, inlet.vapor_fraction, P=inlet.P
        )
        h_top_l = self.thermo.mixture_enthalpy(
            x[0], T[0], vapor_fraction=0.0, P=float(pressures[0])
        )
        h_top_v = self.thermo.mixture_enthalpy(
            x[0], T[0], vapor_fraction=1.0, P=float(pressures[0])
        )
        condensed_overhead = L[0] + (1.0 - condenser_vapor_fraction) * D_guess
        Q_cond = -max(condensed_overhead * abs(h_top_v - h_top_l), inlet.F * 10000.0)
        h_bottom_l = self.thermo.mixture_enthalpy(
            x[-1], T[-1], vapor_fraction=0.0, P=float(pressures[-1])
        )
        Q_reb = max(B_guess * h_bottom_l + D_guess * h_top_l - inlet.F * h_feed - Q_cond, inlet.F * 10000.0)

        initial = {
            'T': T,
            'x': x,
            'L': L,
            'V': V,
            'Q_cond': Q_cond,
            'Q_reb': Q_reb,
            'initializer': initializer,
        }
        if azeotropic_endpoint_data is not None:
            initial['azeotropic_endpoints'] = {
                'x_top': dict(azeotropic_endpoint_data['x_top']),
                'x_bottom': dict(azeotropic_endpoint_data['x_bottom']),
                'D': float(azeotropic_endpoint_data['D']),
                'B': float(azeotropic_endpoint_data['B']),
            }
            initial['azeotropic_pseudo_components'] = [
                dict(item)
                for item in azeotropic_endpoint_data['pseudo_components']
            ]
        if condenser == 'decanter':
            reflux_component = self.get_param('decanter_reflux_component', self.get_param('reflux_phase_component'))
            distillate_guess = dict(x[0])
            if reflux_component:
                reflux_name = str(reflux_component).strip().lower().replace('_', ' ').replace('-', ' ')
                reflux_match = None
                for comp in comps:
                    if comp.lower().replace('_', ' ').replace('-', ' ') == reflux_name:
                        reflux_match = comp
                        break
                if reflux_match is not None:
                    reflux_rich = dict(x[0])
                    reflux_rich[reflux_match] = max(reflux_rich.get(reflux_match, 0.0), 0.65)
                    x[0] = self._normalize(reflux_rich)
                    distillate_guess = dict(feed_z)
                    distillate_guess[reflux_match] = max(distillate_guess.get(reflux_match, 0.0) * 0.3, 1e-8)
                    distillate_guess = self._normalize(distillate_guess)
            raw_reflux = L[0] / max(1.0 - float(self.get_param('decanter_reflux_purge_fraction', self.get_param('reflux_purge_fraction', 0.0))), 1e-8)
            initial['decanter_distillate_x'] = distillate_guess
            initial['decanter_beta'] = D_guess / max(D_guess + raw_reflux, 1e-12)
        return initial

    def _legacy_cmo_endpoint_initial_guess(
        self,
        inlet: StreamState,
        comps: list[str],
        N: int,
        RR: float,
        q_feed: float,
        pressures: list[float],
        distillate_spec: dict,
    ) -> Optional[dict]:
        try:
            params = dict(self.params)
            params['q'] = q_feed
            params['reflux_ratio'] = RR
            if distillate_spec.get('kind') == 'molar':
                params['D'] = float(distillate_spec['value'])
            estimator = McCabeThieleDistillation(
                f"{self.unit_id}_estimate_init", self.thermo, params
            )
            if len(comps) == 2:
                result = estimator.solve({'feed': inlet})
                return {
                    'x_top': self._dense_composition(
                        result.outlet_streams['distillate'].composition, comps
                    ),
                    'x_bottom': self._dense_composition(
                        result.outlet_streams['bottoms'].composition, comps
                    ),
                    'D': result.outlet_streams['distillate'].F,
                    'B': result.outlet_streams['bottoms'].F,
                }

            light_key, heavy_key, _alpha, _data = estimator._binary_keys(comps, inlet)
            candidates = []
            for stage in range(1, N + 1):
                try:
                    stage_solution = estimator._solve_multicomponent_column(
                        inlet, comps, light_key, heavy_key, N, stage, RR
                    )
                except UnitOperationError:
                    continue
                separation = (
                    stage_solution['x_D'].get(light_key, 0.0)
                    - stage_solution['x_B'].get(light_key, 0.0)
                )
                candidates.append((
                    stage_solution['stage_error'],
                    -separation,
                    stage,
                    stage_solution,
                ))
            if not candidates:
                return None
            _error, _separation, _stage, solution = min(
                candidates,
                key=lambda candidate: candidate[:3],
            )
            return {
                'x_top': self._dense_composition(solution['x_D'], comps),
                'x_bottom': self._dense_composition(solution['x_B'], comps),
                'D': solution['D'],
                'B': solution['B'],
            }
        except Exception:
            return None

    def _cmo_profile_initial_guess(
        self,
        inlet: StreamState,
        comps: list[str],
        N: int,
        feed_stage: int,
        RR: float,
        q_feed: float,
        pressures: list[float],
        condenser: str,
        distillate_spec: dict,
        T_min: float,
        T_max: float,
        *,
        latent_heat_correction: bool,
    ) -> dict:
        params = dict(self.params)
        params.update({
            'q': q_feed,
            'feed_stage': feed_stage,
            'reflux_ratio': RR,
            'condenser_type': condenser,
            'latent_heat_correction': latent_heat_correction,
            'cmo_latent_heat_correction': latent_heat_correction,
        })
        params.setdefault('cmo_tolerance', 1e-4)
        if distillate_spec.get('kind') == 'molar':
            params['D'] = float(distillate_spec['value'])
        cmo = CMODistillation(
            f"{self.unit_id}_cmo_init", self.thermo, params
        ).solve({'feed': inlet})
        performance = cmo.performance
        D = cmo.outlet_streams['distillate'].F
        initial = {
            'T': [value + 273.15 for value in performance['stage_temperatures_C']],
            'x': [
                self._dense_composition(stage, comps)
                for stage in performance['stage_liquid_compositions']
            ],
            'L': [max(float(value), 1e-12) for value in performance['liquid_flows']],
            'V': [max(float(value), 1e-12) for value in performance['vapor_flows']],
            'Q_cond': performance['condenser_duty_kW'] * 3600.0,
            'Q_reb': performance['reboiler_duty_kW'] * 3600.0,
            'initializer': 'cmo_hvap' if latent_heat_correction else 'cmo',
        }
        initial['V'][0] = max(D, 1e-12)
        return initial

    def _recycle_profile_initial_guess(
        self,
        comps: list[str],
        N: int,
        T_min: float,
        T_max: float,
    ) -> Optional[dict]:
        context = getattr(self, 'solve_context', {}) or {}
        if context.get('recycle_evaluation') is None:
            return None
        if not self._truthy_param(self.get_param('recycle_warm_start', True)):
            return None
        if self._consume_recycle_warm_start_skip():
            return None
        profile = getattr(self, '_last_recycle_profile', None)
        if not profile:
            return None
        try:
            if profile.get('N') != N:
                return None
            if tuple(profile.get('comps', ())) != tuple(comps):
                return None
            T = [
                min(max(float(value), T_min + 1e-6), T_max - 1e-6)
                for value in profile['T']
            ]
            x = [
                self._normalize({
                    comp: max(float(stage.get(comp, 0.0)), 1e-14)
                    for comp in comps
                })
                for stage in profile['x']
            ]
            L = [max(float(value), 1e-12) for value in profile['L']]
            V = [max(float(value), 1e-12) for value in profile['V']]
            if len(T) != N or len(x) != N or len(L) != N or len(V) != N:
                return None
            return {
                'T': T,
                'x': x,
                'L': L,
                'V': V,
                'Q_cond': float(profile['Q_cond']),
                'Q_reb': float(profile['Q_reb']),
                'initializer': 'previous_recycle',
                **(
                    {
                        'x1': [dict(stage) for stage in profile['x1']],
                        'x2': [dict(stage) for stage in profile['x2']],
                        'beta': [float(value) for value in profile['beta']],
                        'vlle_topology': str(profile['vlle_topology']),
                    }
                    if (
                        len(profile.get('x1', [])) == N
                        and len(profile.get('x2', [])) == N
                        and len(profile.get('beta', [])) == N
                        and len(str(profile.get('vlle_topology', ''))) == N
                    ) else {}
                ),
            }
        except (TypeError, ValueError, KeyError):
            return None

    def _store_recycle_profile(self, performance: dict) -> None:
        context = getattr(self, 'solve_context', {}) or {}
        if context.get('recycle_evaluation') is None:
            return
        if not self._truthy_param(self.get_param('recycle_warm_start', True)):
            return
        try:
            x = performance.get('stage_liquid_compositions')
            T_c = performance.get('stage_temperatures_C')
            L = performance.get('liquid_flows')
            V = performance.get('vapor_flows')
            if not x or not T_c or not L or not V:
                return
            self._last_recycle_profile = {
                'N': int(performance['N_stages']),
                'comps': tuple(x[0].keys()),
                'T': [float(value) + 273.15 for value in T_c],
                'x': [dict(stage) for stage in x],
                'L': [float(value) for value in L],
                'V': [float(value) for value in V],
                'Q_cond': float(performance['condenser_duty_kW']) * 3600.0,
                'Q_reb': float(performance['reboiler_duty_kW']) * 3600.0,
            }
            x1 = performance.get('stage_liquid1_compositions')
            x2 = performance.get('stage_liquid2_compositions')
            beta = performance.get('stage_liquid2_fractions')
            topology = performance.get('vlle_topology')
            if (
                len(x1 or []) == len(x)
                and len(x2 or []) == len(x)
                and len(beta or []) == len(x)
                and len(str(topology or '')) == len(x)
            ):
                self._last_recycle_profile.update({
                    'x1': [dict(stage) for stage in x1],
                    'x2': [dict(stage) for stage in x2],
                    'beta': [float(value) for value in beta],
                    'vlle_topology': str(topology),
                })
        except (TypeError, ValueError, KeyError, IndexError):
            return

    def _coarse_grid_initial_guess(
        self,
        inlet: StreamState,
        comps: list[str],
        N: int,
        feed_stage: int,
        pressures: list[float],
        side_draws: list[dict],
        T_min: float,
        T_max: float,
    ) -> Optional[dict]:
        if self._truthy_param(self.get_param('_skip_coarse_init', False)):
            return None
        condenser_type = str(self.get_param('condenser_type', '')).strip().lower()
        if condenser_type in (
            'decanter', 'heterogeneous', 'heterogeneous_decanter', 'top_decanter',
        ):
            return None
        if condenser_type and condenser_type not in (
            'total', 'complete', 'liquid', 'total_condenser',
        ):
            return None
        recursive_3x = (
            self._truthy_param(self.get_param('_coarse_recursive_3x', False))
            or (
                N > 50
                and self._truthy_param(self.get_param('coarse_recursive_3x', True))
            )
        )
        if not self._truthy_param(self.get_param('coarse_initialization', N >= 16)):
            return None
        if recursive_3x and N <= 12:
            return None
        if (not recursive_3x and N < 16) or side_draws:
            return None
        generated_pressure_profile = self._truthy_param(
            self.get_param('_coarse_generated_stage_pressures', False)
        )
        if (
            self.get_param('stage_pressures', self.get_param('pressure_profile')) is not None
            and not (recursive_3x and generated_pressure_profile)
        ):
            return None

        import numpy as np

        if recursive_3x:
            coarse_N = int(round(N / 3.0))
        else:
            coarse_N = int(self.get_param('coarse_initial_stages', min(12, max(6, N // 2))))
        if coarse_N >= N or coarse_N < 2:
            return None

        coarse_feed_stage = 1 + round((feed_stage - 1) * (coarse_N - 1) / max(N - 1, 1))
        coarse_params = dict(self.params)
        coarse_params['N_stages'] = coarse_N
        coarse_params['feed_stage'] = coarse_feed_stage
        recurse_child = recursive_3x and coarse_N > 12
        coarse_params['_skip_coarse_init'] = not recurse_child
        coarse_params['_coarse_recursive_3x'] = recurse_child
        coarse_params['initializer'] = (
            'coarse_rigorous'
            if recurse_child
            else self.get_param('coarse_initial_initializer', '_legacy_cmo_estimate')
        )
        coarse_params['mesh_tolerance'] = max(float(self.get_param('mesh_tolerance', 2e-6)), 1e-6)
        coarse_params['max_iterations'] = int(self.get_param('coarse_initial_max_iterations', 80))
        coarse_params['max_jacobian_evaluations'] = int(
            self.get_param('coarse_initial_max_jacobian_evaluations', 80)
        )
        coarse_params['T_min'] = T_min
        coarse_params['T_max'] = T_max
        coarse_grid = np.linspace(0.0, 1.0, coarse_N)
        full_grid = np.linspace(0.0, 1.0, N)
        coarse_params['stage_pressures'] = [
            float(np.interp(pos, full_grid, pressures)) for pos in coarse_grid
        ]
        coarse_params['_coarse_generated_stage_pressures'] = True

        try:
            coarse = RigorousDistillation(
                f"{self.unit_id}_coarse_init", self.thermo, coarse_params
            ).solve({'feed': inlet})
        except Exception:
            return None

        perf = coarse.performance
        coarse_x = perf.get('stage_liquid_compositions')
        if not coarse_x:
            return None
        coarse_T = [float(value + 273.15) for value in perf['stage_temperatures_C']]
        coarse_L = [max(float(value), 1e-12) for value in perf['liquid_flows']]
        coarse_V = [max(float(value), 1e-12) for value in perf['vapor_flows']]
        grid_coarse = np.linspace(0.0, 1.0, len(coarse_T))
        grid_full = np.linspace(0.0, 1.0, N)

        T = [
            min(max(float(np.interp(pos, grid_coarse, coarse_T)), T_min + 1e-6), T_max - 1e-6)
            for pos in grid_full
        ]
        L = [max(float(np.interp(pos, grid_coarse, coarse_L)), 1e-12) for pos in grid_full]
        V = [max(float(np.interp(pos, grid_coarse, coarse_V)), 1e-12) for pos in grid_full]
        x = []
        for pos in grid_full:
            stage_comp = {}
            for comp in comps:
                values = [stage.get(comp, 0.0) for stage in coarse_x]
                stage_comp[comp] = max(float(np.interp(pos, grid_coarse, values)), 1e-14)
            x.append(self._normalize(stage_comp))

        return {
            'T': T,
            'x': x,
            'L': L,
            'V': V,
            'Q_cond': float(perf['condenser_duty_kW'] * 3600.0),
            'Q_reb': float(perf['reboiler_duty_kW'] * 3600.0),
            'initializer': 'coarse_rigorous',
        }


    def _azeotropic_endpoint_initial_guess(
        self,
        inlet: StreamState,
        comps: list[str],
        distillate_spec: dict,
        N: int,
        RR: float,
        pressure: float,
        candidates: Optional[list[dict]] = None,
    ) -> Optional[dict]:
        """Build top/bottom endpoint guesses from VLE-only azeotrope pseudos.

        This intentionally uses only endpoint compositions.  The usual initial
        guess path below will fill the internal profile linearly.
        """
        from scipy.optimize import brentq

        if candidates is None:
            candidates = self._vle_azeotrope_candidates(comps, pressure)
        if not candidates:
            return None

        inventory = {
            comp: max(inlet.F * inlet.composition.get(comp, 0.0), 0.0)
            for comp in comps
        }
        inventory_scale = max(sum(inventory.values()), 1e-30)
        pseudos = []

        for candidate in sorted(candidates, key=lambda item: (-item['order'], item['T'])):
            active = [
                comp for comp in comps
                if candidate['composition'].get(comp, 0.0) > 1e-8
            ]
            if not active:
                continue
            amount = min(
                inventory[comp] / max(candidate['composition'][comp], 1e-30)
                for comp in active
            )
            if amount <= inventory_scale * 1e-8:
                continue
            pseudos.append({
                'name': candidate['name'],
                'composition': dict(candidate['composition']),
                'amount': amount,
                'T': candidate['T'],
            })
            for comp in active:
                inventory[comp] = max(
                    inventory[comp] - amount * candidate['composition'][comp],
                    0.0,
                )

        for comp, amount in inventory.items():
            if amount <= inventory_scale * 1e-8:
                continue
            composition = {name: (1.0 if name == comp else 0.0) for name in comps}
            try:
                T_pure = self.thermo.bubble_point_T(composition, pressure)
            except Exception:
                props = self.thermo.props.get(comp)
                T_pure = getattr(props, 'Tb', inlet.T) or inlet.T
            pseudos.append({
                'name': comp,
                'composition': composition,
                'amount': amount,
                'T': float(T_pure),
            })

        if len(pseudos) < 2:
            return None

        pseudo_mw = {}
        for index, pseudo in enumerate(pseudos):
            pseudo_mw[index] = sum(
                pseudo['composition'].get(comp, 0.0) * self.thermo.props[comp].MW
                for comp in comps
            )

        def pseudo_mass(index: int) -> float:
            return pseudos[index]['amount'] * pseudo_mw[index]

        if distillate_spec['kind'] == 'mass':
            target = float(distillate_spec['value'])
            available = sum(pseudo_mass(index) for index in range(len(pseudos)))
            amount_for_theta = lambda theta: sum(
                pseudo_mass(index) * split_fraction(index, theta)
                for index in range(len(pseudos))
            )
        elif distillate_spec['kind'] == 'molar':
            target = float(distillate_spec['value'])
            available = sum(item['amount'] for item in pseudos)
            amount_for_theta = lambda theta: sum(
                item['amount'] * split_fraction(index, theta)
                for index, item in enumerate(pseudos)
            )
        else:
            return None

        if not 0.0 < target < available:
            return None

        reference_temperature = float(self.get_param('azeotropic_reference_temperature', inlet.T))
        hvap = float(self.get_param('azeotropic_hvap_kJ_per_kmol', 35000.0))
        gas_constant = R_J_MOL_K
        volatility = {
            index: math.exp(
                -hvap / gas_constant
                * (1.0 / max(reference_temperature, 1e-12) - 1.0 / max(item['T'], 1e-12))
            )
            for index, item in enumerate(pseudos)
        }
        heavy = min(volatility, key=volatility.get)
        effective_stages = max(1.0, N * RR / max(RR + 1.0, 1e-12))
        scores = {
            name: max(volatility[name] / max(volatility[heavy], 1e-300), 1e-300) ** effective_stages
            for name in volatility
        }

        def split_fraction(index: int, theta: float) -> float:
            score = scores[index]
            return min(max(score / (score + theta), 1e-9), 1.0 - 1e-9)

        low = 1e-300
        high = 1.0
        while amount_for_theta(high) > target:
            high *= 10.0
            if high > 1e300:
                return None
        theta = brentq(
            lambda value: amount_for_theta(value) - target,
            low,
            high,
            xtol=1e-14,
            rtol=1e-12,
            maxiter=200,
        )

        dist_pseudo = {
            index: item['amount'] * split_fraction(index, theta)
            for index, item in enumerate(pseudos)
        }
        bot_pseudo = {
            index: max(item['amount'] - dist_pseudo[index], 0.0)
            for index, item in enumerate(pseudos)
        }

        def expand(amounts: dict[int, float]) -> tuple[dict[str, float], float]:
            moles = {comp: 0.0 for comp in comps}
            for index, amount in amounts.items():
                for comp, fraction in pseudos[index]['composition'].items():
                    moles[comp] += amount * fraction
            total = sum(moles.values())
            if total <= 0.0:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' azeotropic initializer "
                    "produced an empty pseudo-product"
                )
            return ({comp: moles[comp] / total for comp in comps}, total)

        x_top, D = expand(dist_pseudo)
        x_bottom, B = expand(bot_pseudo)
        return {
            'x_top': self._dense_composition(x_top, comps),
            'x_bottom': self._dense_composition(x_bottom, comps),
            'D': D,
            'B': B,
            'pseudo_components': [
                {
                    'name': item['name'],
                    'composition': dict(item['composition']),
                    'amount': float(item['amount']),
                    'temperature_K': float(item['T']),
                    'distillate_fraction': float(split_fraction(
                        index, theta
                    )),
                }
                for index, item in enumerate(pseudos)
            ],
        }

    def _vle_azeotrope_candidates(self, comps: list[str], pressure: float) -> list[dict]:
        cache = getattr(self.thermo, '_distillation_vle_azeotrope_cache', None)
        if cache is None:
            cache = {}
            setattr(self.thermo, '_distillation_vle_azeotrope_cache', cache)

        active_comps = tuple(comps)
        ternary_starts = int(self.get_param('azeotropic_ternary_starts', 4))
        max_ternary = int(self.get_param('azeotropic_max_ternary_combinations', 20))
        cache_key = (
            active_comps,
            round(float(pressure), 10),
            ternary_starts,
            max_ternary,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        import itertools

        candidates = []
        for pair in itertools.combinations(comps, 2):
            root = self._binary_vle_azeotrope(pair, pressure)
            if root is not None:
                composition = {comp: 0.0 for comp in comps}
                composition.update(root['composition'])
                candidates.append({
                    'name': 'az_' + '_'.join(pair),
                    'order': 2,
                    'T': root['T'],
                    'composition': composition,
                })

        if len(comps) >= 3 and max_ternary > 0:
            for index, trio in enumerate(itertools.combinations(comps, 3)):
                if index >= max_ternary:
                    break
                root = self._ternary_vle_azeotrope(trio, pressure, ternary_starts)
                if root is None:
                    continue
                composition = {comp: 0.0 for comp in comps}
                composition.update(root['composition'])
                candidates.append({
                    'name': 'az_' + '_'.join(trio),
                    'order': 3,
                    'T': root['T'],
                    'composition': composition,
                })

        cache[cache_key] = [dict(item) for item in candidates]
        return candidates

    def _ternary_vlle_azeotropes(
        self,
        trio: tuple[str, str, str],
        comps: list[str],
        pressure: float,
        binary_candidates: list[dict],
    ) -> list[dict]:
        """Solve ternary heteroazeotropes without nested flash calculations."""
        import numpy as np
        from scipy.optimize import least_squares

        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .equilibrium_stage_vlle import shared_vlle_vapor_terms
        else:
            from equilibrium_stage_vlle import shared_vlle_vapor_terms

        pure_temperatures = []
        for target in trio:
            composition = {
                comp: (1.0 if comp == target else 0.0) for comp in comps
            }
            try:
                pure_temperatures.append(
                    self.thermo.bubble_point_T(composition, pressure)
                )
            except Exception:
                props = self.thermo.props.get(target)
                pure_temperatures.append(
                    float(getattr(props, 'Tb', 350.0) or 350.0)
                )
        column_T_min, column_T_max = self._temperature_bounds(comps)
        T_low = max(column_T_min, min(pure_temperatures) - 80.0)
        T_high = min(column_T_max, max(pure_temperatures) + 80.0)
        if T_high <= T_low:
            return []

        def softmax(values):
            raw = np.r_[np.asarray(values, dtype=float), 0.0]
            raw -= np.max(raw)
            fractions = np.exp(raw)
            return fractions / np.sum(fractions)

        def logits(values):
            values = np.maximum(np.asarray(values, dtype=float), 1e-12)
            values /= np.sum(values)
            return np.log(values[:-1] / values[-1])

        def temperature(theta):
            return (
                0.5 * (T_low + T_high)
                + 0.5 * (T_high - T_low) * math.tanh(float(theta))
            )

        def temperature_variable(value):
            scaled = (
                (2.0 * float(value) - T_low - T_high)
                / max(T_high - T_low, 1e-12)
            )
            return math.atanh(min(max(scaled, -0.999999), 0.999999))

        def decode(values):
            phase1_values = softmax(values[:2])
            phase2_values = softmax(values[2:4])
            beta_value = min(max(float(values[4]), -40.0), 40.0)
            beta = 1.0 / (1.0 + math.exp(-beta_value))
            T = temperature(values[5])
            liquid1 = {comp: 0.0 for comp in comps}
            liquid2 = {comp: 0.0 for comp in comps}
            for comp, value in zip(trio, phase1_values):
                liquid1[comp] = float(value)
            for comp, value in zip(trio, phase2_values):
                liquid2[comp] = float(value)
            aggregate = {
                comp: (
                    (1.0 - beta) * liquid1.get(comp, 0.0)
                    + beta * liquid2.get(comp, 0.0)
                )
                for comp in comps
            }
            return liquid1, liquid2, beta, T, aggregate

        def equilibrium(values):
            liquid1, liquid2, beta, T, aggregate = decode(values)
            gamma1 = self.thermo.activity_coefficients(T, liquid1)
            gamma2 = self.thermo.activity_coefficients(T, liquid2)
            vapor_terms = shared_vlle_vapor_terms(
                self.thermo,
                T,
                pressure,
                liquid1,
                liquid2,
                comps,
                gamma1,
                gamma2,
            )
            return (
                liquid1,
                liquid2,
                beta,
                T,
                aggregate,
                gamma1,
                gamma2,
                vapor_terms,
            )

        def residual(values):
            try:
                (
                    liquid1,
                    liquid2,
                    _beta,
                    _T,
                    aggregate,
                    gamma1,
                    gamma2,
                    vapor_terms,
                ) = equilibrium(values)
                vapor_total = sum(vapor_terms.values())
                vapor = {
                    comp: vapor_terms[comp] / vapor_total for comp in comps
                }
                return np.array(
                    [
                        math.log(max(
                            liquid1[comp] * gamma1[comp], 1e-300
                        ))
                        - math.log(max(
                            liquid2[comp] * gamma2[comp], 1e-300
                        ))
                        for comp in trio
                    ]
                    + [math.log(max(vapor_total, 1e-300))]
                    + [
                        aggregate[comp] - vapor[comp]
                        for comp in trio[:-1]
                    ],
                    dtype=float,
                )
            except Exception:
                return np.ones(6, dtype=float) * 1e3

        starts = [
            (
                [0.9 if index == first else 0.05 for index in range(3)],
                [0.9 if index == second else 0.05 for index in range(3)],
                0.5,
                min(pure_temperatures) - 10.0,
            )
            for first, second in ((0, 1), (0, 2), (1, 2))
        ]
        trio_set = set(trio)
        for candidate in binary_candidates:
            active = {
                comp
                for comp, value in candidate['composition'].items()
                if value > 1e-8
            }
            if not active.issubset(trio_set) or len(active) != 2:
                continue
            epsilon = 0.02
            phase1 = np.array([
                max(candidate['liquid1'].get(comp, 0.0), epsilon)
                for comp in trio
            ])
            phase2 = np.array([
                max(candidate['liquid2'].get(comp, 0.0), epsilon)
                for comp in trio
            ])
            phase1 /= np.sum(phase1)
            phase2 /= np.sum(phase2)
            starts.append((
                phase1,
                phase2,
                candidate['liquid2_fraction'],
                candidate['T'],
            ))

        tolerance = float(self.get_param(
            'vlle_azeotropic_fugacity_tolerance', 1e-7
        ))
        max_evaluations = max(12, int(self.get_param(
            'vlle_azeotropic_ternary_max_evaluations', 100
        )))
        solutions = []
        for liquid1, liquid2, beta, T_start in starts:
            beta = min(max(float(beta), 1e-8), 1.0 - 1e-8)
            initial = np.r_[
                logits(liquid1),
                logits(liquid2),
                math.log(beta / (1.0 - beta)),
                temperature_variable(T_start),
            ]
            solved = least_squares(
                residual,
                initial,
                method='lm',
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
                max_nfev=max_evaluations,
            )
            norm = float(np.linalg.norm(residual(solved.x), ord=np.inf))
            if not math.isfinite(norm) or norm > tolerance:
                continue
            (
                phase1,
                phase2,
                solved_beta,
                T,
                aggregate,
                _gamma1,
                _gamma2,
                vapor_terms,
            ) = equilibrium(solved.x)
            phase_distance = sum(
                abs(phase1[comp] - phase2[comp]) for comp in trio
            )
            if phase_distance <= float(self.get_param(
                'vlle_phase_distance_min', 1e-3
            )):
                continue
            vapor_total = sum(vapor_terms.values())
            vapor = {
                comp: float(vapor_terms[comp] / vapor_total) for comp in comps
            }
            if any(vapor[comp] <= 1e-5 for comp in trio):
                continue

            def liquid_gibbs(composition):
                gamma = self.thermo.activity_coefficients(T, composition)
                return sum(
                    composition[comp]
                    * math.log(max(
                        composition[comp] * gamma[comp], 1e-300
                    ))
                    for comp in trio
                )

            gibbs_benefit = (
                liquid_gibbs(vapor)
                - (1.0 - solved_beta) * liquid_gibbs(phase1)
                - solved_beta * liquid_gibbs(phase2)
            )
            if gibbs_benefit <= float(self.get_param(
                'vlle_azeotropic_min_gibbs_benefit', 1e-8
            )):
                continue
            candidate = {
                'name': 'vlle_az_' + '_'.join(trio),
                'type': 'VLLE',
                'order': 3,
                'T': float(T),
                'composition': vapor,
                'liquid1': phase1,
                'liquid2': phase2,
                'liquid2_fraction': float(solved_beta),
                'fixed_point_residual': max(
                    abs(aggregate[comp] - vapor[comp]) for comp in trio
                ),
                'fugacity_residual': norm,
                'gibbs_benefit': float(gibbs_benefit),
                'function_evaluations': int(solved.nfev),
            }
            duplicate = any(
                abs(candidate['T'] - item['T']) < 1e-5
                and max(
                    abs(
                        candidate['composition'][comp]
                        - item['composition'][comp]
                    )
                    for comp in comps
                ) < 1e-5
                for item in solutions
            )
            if not duplicate:
                solutions.append(candidate)
        return solutions

    def _provided_vlle_azeotrope_candidates(
        self,
        comps: list[str],
    ) -> list[dict]:
        """Validate explicit VLLE azeotropes without running a phase search."""
        raw_candidates = self.get_param('vlle_azeotropes')
        if raw_candidates is None:
            composition = self.get_param('vlle_azeotrope_composition')
            temperature = self.get_param(
                'vlle_azeotrope_temperature',
                self.get_param('vlle_azeotrope_temperature_K'),
            )
            if composition is None and temperature is None:
                return []
            raw_candidates = [{
                'name': 'provided_vlle_azeotrope',
                'composition': composition,
                'T': temperature,
            }]
        elif isinstance(raw_candidates, dict):
            raw_candidates = [raw_candidates]
        if not isinstance(raw_candidates, (list, tuple)):
            raise UnitOperationError(
                f"RigorousDistillation '{self.unit_id}' vlle_azeotropes must "
                "be a list of mappings"
            )

        candidates = []
        known_components = set(comps)
        for index, raw in enumerate(raw_candidates, 1):
            if not isinstance(raw, dict):
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' VLLE azeotrope "
                    f"candidate {index} must be a mapping"
                )
            composition = raw.get('composition', raw.get('x'))
            temperature = raw.get(
                'T', raw.get('T_K', raw.get('temperature_K'))
            )
            if not isinstance(composition, dict) or temperature is None:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' VLLE azeotrope "
                    f"candidate {index} requires composition and T/T_K"
                )
            unknown = set(composition) - known_components
            if unknown:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' VLLE azeotrope "
                    f"candidate {index} contains unknown component(s): "
                    + ', '.join(sorted(unknown))
                )
            try:
                temperature = float(temperature)
                values = {
                    comp: float(composition.get(comp, 0.0))
                    for comp in comps
                }
            except (TypeError, ValueError) as exc:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' VLLE azeotrope "
                    f"candidate {index} must contain numeric values"
                ) from exc
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' VLLE azeotrope "
                    f"candidate {index} temperature must be positive and finite"
                )
            total = sum(values.values())
            if (
                not math.isfinite(total)
                or any(not math.isfinite(value) or value < 0.0
                       for value in values.values())
            ):
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' VLLE azeotrope "
                    f"candidate {index} composition must be finite and nonnegative"
                )
            if total <= 0.0:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' VLLE azeotrope "
                    f"candidate {index} composition must have a positive total"
                )
            normalized = {
                comp: value / total for comp, value in values.items()
            }
            active = [
                comp for comp, value in normalized.items() if value > 1e-8
            ]
            if len(active) < 2:
                raise UnitOperationError(
                    f"RigorousDistillation '{self.unit_id}' VLLE azeotrope "
                    f"candidate {index} must contain at least two components"
                )
            candidates.append({
                'name': str(raw.get(
                    'name', f'provided_vlle_azeotrope_{index}'
                )),
                'type': 'VLLE',
                'order': len(active),
                'T': temperature,
                'composition': normalized,
                'fixed_point_residual': float(raw.get(
                    'fixed_point_residual', 0.0
                )),
                'fugacity_residual': float(raw.get(
                    'fugacity_residual', 0.0
                )),
            })
        return candidates

    def _binary_vlle_azeotropes(
        self,
        pair: tuple[str, str],
        comps: list[str],
        pressure: float,
    ) -> list[dict]:
        """Solve binary heteroazeotropes as simultaneous VLLE systems."""
        import numpy as np
        from scipy.optimize import least_squares

        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .equilibrium_stage_vlle import shared_vlle_vapor_terms
        else:
            from equilibrium_stage_vlle import shared_vlle_vapor_terms

        comp_a, comp_b = pair
        pure_temperatures = []
        for target in pair:
            composition = {
                comp: (1.0 if comp == target else 0.0) for comp in comps
            }
            try:
                pure_temperatures.append(
                    self.thermo.bubble_point_T(composition, pressure)
                )
            except Exception:
                props = self.thermo.props.get(target)
                pure_temperatures.append(
                    float(getattr(props, 'Tb', 350.0) or 350.0)
                )
        fractions = (
            0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 0.95, 0.98,
        )
        possible_split = False
        for screen_T in (
            min(pure_temperatures) - 10.0,
            min(pure_temperatures),
            min(pure_temperatures) + 20.0,
        ):
            gibbs_values = []
            try:
                for fraction in fractions:
                    composition = {comp: 0.0 for comp in comps}
                    composition[comp_a] = fraction
                    composition[comp_b] = 1.0 - fraction
                    gamma = self.thermo.activity_coefficients(
                        screen_T, composition
                    )
                    gibbs_values.append(
                        fraction * math.log(max(
                            fraction * gamma[comp_a], 1e-300
                        ))
                        + (1.0 - fraction) * math.log(max(
                            (1.0 - fraction) * gamma[comp_b], 1e-300
                        ))
                    )
            except Exception:
                # An unavailable screening model must not suppress a real
                # candidate; let the simultaneous fugacity solve decide.
                possible_split = True
                break
            for index in range(1, len(fractions) - 1):
                left, middle, right = fractions[index - 1:index + 2]
                chord = (
                    gibbs_values[index - 1]
                    + (gibbs_values[index + 1] - gibbs_values[index - 1])
                    * (middle - left) / (right - left)
                )
                if gibbs_values[index] - chord > -1e-3:
                    possible_split = True
                    break
            if possible_split:
                break
        if not possible_split:
            return []
        column_T_min, column_T_max = self._temperature_bounds(comps)
        T_low = max(column_T_min, min(pure_temperatures) - 80.0)
        T_high = min(column_T_max, max(pure_temperatures) + 80.0)
        if T_high <= T_low:
            return []

        def sigmoid(value):
            value = min(max(float(value), -40.0), 40.0)
            return 1.0 / (1.0 + math.exp(-value))

        def logit(value):
            value = min(max(float(value), 1e-12), 1.0 - 1e-12)
            return math.log(value / (1.0 - value))

        def temperature(theta):
            return (
                0.5 * (T_low + T_high)
                + 0.5 * (T_high - T_low) * math.tanh(float(theta))
            )

        def temperature_variable(value):
            scaled = (
                (2.0 * float(value) - T_low - T_high)
                / max(T_high - T_low, 1e-12)
            )
            return math.atanh(min(max(scaled, -0.999999), 0.999999))

        def decode(values):
            x_low = sigmoid(values[0])
            gap_fraction = sigmoid(values[1])
            x_high = x_low + (1.0 - x_low) * gap_fraction
            T = temperature(values[2])
            liquid1 = {comp: 0.0 for comp in comps}
            liquid2 = {comp: 0.0 for comp in comps}
            liquid1[comp_a], liquid1[comp_b] = x_high, 1.0 - x_high
            liquid2[comp_a], liquid2[comp_b] = x_low, 1.0 - x_low
            return liquid1, liquid2, T

        def equilibrium(values):
            liquid1, liquid2, T = decode(values)
            gamma1 = self.thermo.activity_coefficients(T, liquid1)
            gamma2 = self.thermo.activity_coefficients(T, liquid2)
            vapor_terms = shared_vlle_vapor_terms(
                self.thermo,
                T,
                pressure,
                liquid1,
                liquid2,
                comps,
                gamma1,
                gamma2,
            )
            return liquid1, liquid2, T, gamma1, gamma2, vapor_terms

        def residual(values):
            try:
                liquid1, liquid2, _T, gamma1, gamma2, vapor_terms = (
                    equilibrium(values)
                )
                return np.array([
                    math.log(max(
                        liquid1[comp_a] * gamma1[comp_a], 1e-300
                    ))
                    - math.log(max(
                        liquid2[comp_a] * gamma2[comp_a], 1e-300
                    )),
                    math.log(max(
                        liquid1[comp_b] * gamma1[comp_b], 1e-300
                    ))
                    - math.log(max(
                        liquid2[comp_b] * gamma2[comp_b], 1e-300
                    )),
                    math.log(max(sum(vapor_terms.values()), 1e-300)),
                ], dtype=float)
            except Exception:
                return np.ones(3, dtype=float) * 1e3

        max_evaluations = max(8, int(self.get_param(
            'vlle_azeotropic_max_evaluations', 60
        )))
        tolerance = float(self.get_param(
            'vlle_azeotropic_fugacity_tolerance', 1e-7
        ))
        starts = (
            (0.995, 0.005),
            (0.99, 0.15),
            (0.8, 0.05),
            (0.4, 0.02),
        )
        solutions = []
        temperature_starts = {
            min(max(min(pure_temperatures) + offset, T_low + 1e-6), T_high - 1e-6)
            for offset in (-30.0, -10.0, 10.0)
        }
        for x_high, x_low in starts:
            for T_start in temperature_starts:
                initial = np.array([
                    logit(x_low),
                    logit((x_high - x_low) / (1.0 - x_low)),
                    temperature_variable(T_start),
                ])
                solved = least_squares(
                    residual,
                    initial,
                    method='lm',
                    xtol=1e-10,
                    ftol=1e-10,
                    gtol=1e-10,
                    max_nfev=max_evaluations,
                )
                values = residual(solved.x)
                norm = float(np.linalg.norm(values, ord=np.inf))
                if not math.isfinite(norm) or norm > tolerance:
                    continue
                liquid1, liquid2, T, _gamma1, _gamma2, vapor_terms = (
                    equilibrium(solved.x)
                )
                phase_distance = sum(
                    abs(liquid1[comp] - liquid2[comp]) for comp in pair
                )
                if (
                    math.isfinite(norm)
                    and phase_distance > float(self.get_param(
                        'vlle_phase_distance_min', 1e-3
                    ))
                ):
                    solutions.append((
                        norm,
                        liquid1,
                        liquid2,
                        float(T),
                        vapor_terms,
                        int(solved.nfev),
                    ))
        candidates = []
        for (
            norm,
            liquid1,
            liquid2,
            T,
            vapor_terms,
            evaluations,
        ) in sorted(solutions, key=lambda item: item[0]):
            if norm > tolerance:
                continue
            vapor_total = sum(vapor_terms.values())
            vapor = {
                comp: float(vapor_terms[comp] / vapor_total)
                for comp in comps
            }
            denominator = liquid2[comp_a] - liquid1[comp_a]
            if abs(denominator) <= 1e-12:
                continue
            beta = (
                vapor.get(comp_a, 0.0) - liquid1[comp_a]
            ) / denominator
            phase_fraction = min(beta, 1.0 - beta)
            if (
                beta <= 0.0
                or beta >= 1.0
                or phase_fraction <= float(self.get_param(
                    'vlle_phase_fraction_min', 1e-6
                ))
                or any(vapor.get(comp, 0.0) <= 1e-5 for comp in pair)
            ):
                continue

            def liquid_gibbs(composition):
                gamma = self.thermo.activity_coefficients(T, composition)
                return sum(
                    composition[comp]
                    * math.log(max(
                        composition[comp] * gamma[comp], 1e-300
                    ))
                    for comp in pair
                )

            homogeneous_gibbs = liquid_gibbs(vapor)
            split_gibbs = (
                (1.0 - beta) * liquid_gibbs(liquid1)
                + beta * liquid_gibbs(liquid2)
            )
            gibbs_benefit = homogeneous_gibbs - split_gibbs
            if gibbs_benefit <= float(self.get_param(
                'vlle_azeotropic_min_gibbs_benefit', 1e-8
            )):
                continue
            candidate = {
                'name': 'vlle_az_' + '_'.join(pair),
                'type': 'VLLE',
                'order': 2,
                'T': T,
                'composition': vapor,
                'liquid1': liquid1,
                'liquid2': liquid2,
                'liquid2_fraction': float(beta),
                'fixed_point_residual': 0.0,
                'fugacity_residual': float(norm),
                'gibbs_benefit': float(gibbs_benefit),
                'function_evaluations': evaluations,
            }
            duplicate = any(
                abs(T - item['T']) < 1e-5
                and max(
                    abs(vapor[comp] - item['composition'][comp])
                    for comp in comps
                ) < 1e-5
                for item in candidates
            )
            if not duplicate:
                candidates.append(candidate)
        return candidates

    def _vlle_azeotrope_candidates(
        self,
        comps: list[str],
        pressure: float,
        *,
        feed_z: Optional[dict[str, float]] = None,
    ) -> list[dict]:
        """Find binary/ternary VLLE azeotropes with simultaneous solves."""
        import itertools

        configured_max_pairs = self.get_param(
            'vlle_azeotropic_max_binary_pairs'
        )
        max_pairs = (
            None
            if configured_max_pairs is None
            else max(0, int(configured_max_pairs))
        )
        configured_max_ternary = self.get_param(
            'vlle_azeotropic_max_ternary_combinations'
        )
        max_ternary = (
            None
            if configured_max_ternary is None
            else max(0, int(configured_max_ternary))
        )
        exhaustive_ternary = self._truthy_param(self.get_param(
            'vlle_azeotropic_exhaustive_ternary', False
        ))
        ternary_k_min = float(self.get_param(
            'vlle_azeotropic_ternary_k_min', 0.2
        ))
        ternary_feed_min = float(self.get_param(
            'vlle_azeotropic_ternary_feed_fraction_min', 1e-5
        ))
        cache = getattr(self.thermo, '_distillation_vlle_azeotrope_cache', None)
        if cache is None:
            cache = {}
            setattr(self.thermo, '_distillation_vlle_azeotrope_cache', cache)
        cache_key = (
            tuple(comps),
            round(float(pressure), 10),
            -1 if max_pairs is None else max_pairs,
            -1 if max_ternary is None else max_ternary,
            int(self.get_param('vlle_azeotropic_max_evaluations', 60)),
            int(self.get_param(
                'vlle_azeotropic_ternary_max_evaluations', 100
            )),
            float(self.get_param(
                'vlle_azeotropic_fugacity_tolerance', 1e-7
            )),
            float(self.get_param(
                'vlle_azeotropic_min_gibbs_benefit', 1e-8
            )),
            float(self.get_param('vlle_phase_fraction_min', 1e-6)),
            float(self.get_param('vlle_phase_distance_min', 1e-3)),
            self._temperature_bounds(comps),
            exhaustive_ternary,
            ternary_k_min,
            ternary_feed_min,
            (
                None
                if feed_z is None
                else tuple(round(float(feed_z.get(comp, 0.0)), 12) for comp in comps)
            ),
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]
        candidates = []
        for index, pair in enumerate(itertools.combinations(comps, 2)):
            if max_pairs is not None and index >= max_pairs:
                break
            candidates.extend(self._binary_vlle_azeotropes(
                pair, comps, pressure
            ))
        ternary_combinations = set()
        if len(comps) >= 3 and exhaustive_ternary:
            ternary_combinations.update(itertools.combinations(comps, 3))
        elif len(comps) >= 3:
            for candidate in candidates:
                if candidate['order'] != 2:
                    continue
                pair = tuple(
                    comp
                    for comp in comps
                    if candidate['composition'].get(comp, 0.0) > 1e-8
                )
                for third in comps:
                    if third in pair:
                        continue
                    if (
                        feed_z is not None
                        and feed_z.get(third, 0.0) < ternary_feed_min
                    ):
                        continue
                    probe = {
                        comp: (
                            0.999 * candidate['composition'].get(comp, 0.0)
                            + (0.001 if comp == third else 0.0)
                        )
                        for comp in comps
                    }
                    try:
                        K_third = self.thermo.K_values(
                            candidate['T'], pressure, probe
                        ).get(third, 0.0)
                    except Exception:
                        K_third = math.inf
                    if K_third >= ternary_k_min:
                        ternary_combinations.add(tuple(
                            comp for comp in comps
                            if comp in set(pair) | {third}
                        ))
        if ternary_combinations:
            for index, trio in enumerate(sorted(
                ternary_combinations,
                key=lambda values: tuple(comps.index(comp) for comp in values),
            )):
                if max_ternary is not None and index >= max_ternary:
                    break
                candidates.extend(self._ternary_vlle_azeotropes(
                    trio,
                    comps,
                    pressure,
                    candidates,
                ))
        cache[cache_key] = [dict(item) for item in candidates]
        return candidates

    def _binary_vle_azeotrope(self, pair: tuple[str, str], pressure: float) -> Optional[dict]:
        from scipy.optimize import brentq, minimize_scalar

        comp_a, comp_b = pair

        def composition(x_a: float) -> dict[str, float]:
            x_a = min(max(float(x_a), 1e-12), 1.0 - 1e-12)
            return {comp_a: x_a, comp_b: 1.0 - x_a}

        def residual(x_a: float) -> float:
            x = composition(x_a)
            T = self.thermo.bubble_point_T(x, pressure)
            K = self.thermo.K_values(T, pressure, x)
            return math.log(max(K.get(comp_a, 1.0), 1e-300) / max(K.get(comp_b, 1.0), 1e-300))

        root = None
        for eps in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
            try:
                f_low = residual(eps)
                f_high = residual(1.0 - eps)
            except Exception:
                continue
            if f_low * f_high < 0.0:
                root = brentq(residual, eps, 1.0 - eps, xtol=1e-13, rtol=1e-13, maxiter=80)
                break

        if root is None:
            try:
                optimum = minimize_scalar(
                    lambda value: abs(residual(value)),
                    bounds=(1e-6, 1.0 - 1e-6),
                    method='bounded',
                    options={'xatol': 1e-12, 'maxiter': 80},
                )
                if optimum.success and abs(residual(optimum.x)) < 1e-8:
                    root = float(optimum.x)
            except Exception:
                return None

        if root is None or root <= 1e-5 or root >= 1.0 - 1e-5:
            return None

        x = composition(root)
        T = self.thermo.bubble_point_T(x, pressure)
        return {'composition': x, 'T': T}

    def _ternary_vle_azeotrope(
        self,
        trio: tuple[str, str, str],
        pressure: float,
        starts: int,
    ) -> Optional[dict]:
        from scipy.optimize import least_squares
        import numpy as np

        comps = list(trio)
        try:
            pure_T = [
                self.thermo.bubble_point_T(
                    {comp: (1.0 if comp == target else 0.0) for comp in comps},
                    pressure,
                )
                for target in comps
            ]
        except Exception:
            return None
        T_min = max(240.0, min(pure_T) - 60.0)
        T_max = min(700.0, max(pure_T) + 90.0)

        def softmax(values):
            raw = np.array([values[0], values[1], 0.0], dtype=float)
            raw -= np.max(raw)
            exp_values = np.exp(raw)
            return exp_values / np.sum(exp_values)

        def variables_from_x(x_values):
            x_values = np.maximum(np.array(x_values, dtype=float), 1e-14)
            x_values /= np.sum(x_values)
            return np.array([
                math.log(x_values[0] / x_values[2]),
                math.log(x_values[1] / x_values[2]),
            ], dtype=float)

        def bounded_T(theta):
            return 0.5 * (T_min + T_max) + 0.5 * (T_max - T_min) * math.tanh(float(theta))

        def variable_from_T(T):
            scaled = (2.0 * T - T_min - T_max) / max(T_max - T_min, 1e-12)
            scaled = min(max(scaled, -0.999999999), 0.999999999)
            return math.atanh(scaled)

        def residual(values):
            x_values = softmax(values[:2])
            x = dict(zip(comps, x_values))
            T = bounded_T(values[2])
            try:
                K = self.thermo.K_values(T, pressure, x)
                bubble = sum(x[comp] * K.get(comp, 1.0) for comp in comps)
                ref = comps[-1]
                return np.array([
                    math.log(max(K.get(comps[0], 1.0), 1e-300) / max(K.get(ref, 1.0), 1e-300)),
                    math.log(max(K.get(comps[1], 1.0), 1e-300) / max(K.get(ref, 1.0), 1e-300)),
                    math.log(max(bubble, 1e-300)),
                ], dtype=float)
            except Exception:
                return np.ones(3, dtype=float) * 1e3

        x_starts = [
            np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            np.array([0.15, 0.15, 0.70]),
            np.array([0.15, 0.70, 0.15]),
            np.array([0.70, 0.15, 0.15]),
        ][:max(1, int(starts))]
        T_starts = [sum(pure_T) / len(pure_T), min(pure_T), max(pure_T)]

        best = None
        for x_start in x_starts:
            for T_start in T_starts:
                solved = least_squares(
                    residual,
                    np.r_[variables_from_x(x_start), variable_from_T(T_start)],
                    method='trf',
                    xtol=1e-9,
                    ftol=1e-9,
                    gtol=1e-9,
                    max_nfev=100,
                )
                norm = float(np.linalg.norm(residual(solved.x), ord=np.inf))
                x_values = softmax(solved.x[:2])
                T = bounded_T(solved.x[2])
                if (
                    norm < 1e-7
                    and np.all(x_values > 1e-5)
                    and np.all(x_values < 1.0 - 1e-5)
                    and (best is None or norm < best[0])
                ):
                    best = (norm, x_values, T)

        if best is None:
            return None
        _, x_values, T = best
        return {
            'composition': dict(zip(comps, [float(value) for value in x_values])),
            'T': float(T),
        }

    def _distillate_molar_guess(self, spec: dict, inlet: StreamState, composition: dict[str, float]) -> float:
        if spec['kind'] == 'decanter':
            return max(float(spec['value']), 1e-9)
        if spec['kind'] == 'molar':
            return max(float(spec['value']), 1e-9)
        mw = sum(composition.get(comp, 0.0) * self.thermo.props[comp].MW for comp in composition)
        return max(float(spec['value']) / max(mw, 1e-12), 1e-9)

class CMODistillation(RigorousDistillation):
    """Multicomponent CMO MES column with fixed section traffic."""

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        import numpy as np

        if self._stage_phase_model() != 'VLE':
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' supports only "
                "stage_phase_model=VLE; use RigorousDistillation for VLLE stages"
            )

        if not inlets:
            raise UnitOperationError(f"CMODistillation '{self.unit_id}' has no inlet stream")
        positive_inlets = [
            stream for stream in inlets.values()
            if stream.F > 0.0
        ]
        if len(positive_inlets) != 1:
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' currently supports exactly one positive feed"
            )
        inlet = positive_inlets[0]
        if inlet.F <= 0.0:
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' requires a positive feed flow"
            )

        N = int(self.get_param('N_stages', self.get_param('stages', 10)))
        feed_stage = int(self.get_param('feed_stage', max(1, N // 2)))
        RR = float(self.get_param('reflux_ratio', self.get_param('RR', 2.0)))
        condenser = str(self.get_param('condenser_type', 'total')).strip().lower()
        if condenser in ('complete', 'liquid', 'total_condenser'):
            condenser = 'total'
        if condenser in ('vapor', 'partial_vapor', 'partial-condenser', 'partial_condenser'):
            condenser = 'partial'
        if condenser in ('two_phase', 'two-phase', 'mixed_distillate', 'partial_liquid'):
            condenser = 'mixed'
        if condenser in ('decanter', 'heterogeneous', 'heterogeneous_decanter', 'top_decanter'):
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' does not support decanter condensers"
            )
        if condenser not in ('total', 'partial', 'mixed'):
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' condenser_type must be total, mixed, or partial"
            )
        condenser_vapor_fraction = self._condenser_vapor_fraction(condenser)
        if N < 2:
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' requires N_stages >= 2"
            )
        if not 1 <= feed_stage <= N:
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' feed_stage must be between 1 and {N}"
            )
        if RR < 0.0:
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' requires reflux_ratio >= 0"
            )

        comps = self._component_order(inlet)
        nc = len(comps)
        if nc < 2:
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' needs >= 2 components"
            )
        feed_z = self._normalize({
            comp: inlet.composition.get(comp, 0.0)
            for comp in comps
        })
        pressures = self._pressure_profile(N, inlet.P)
        if any(value <= 0.0 for value in pressures):
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' pressure profile must be positive"
            )
        T_min, T_max = self._temperature_bounds(comps)

        distillate_spec = self._distillate_spec(inlet)
        q = float(self.get_param('q', 1.0 - inlet.vapor_fraction))
        if not math.isfinite(q):
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' requires a finite feed quality q"
            )

        volatility_data = McCabeThieleDistillation._relative_volatility_data(self, comps, inlet)
        reference_K = {
            comp: max(float(volatility_data['K_values'].get(comp, 1.0)), 1e-12)
            for comp in comps
        }
        light_key = max(comps, key=lambda comp: reference_K[comp])
        heavy_key = min(comps, key=lambda comp: reference_K[comp])
        relative_volatility = {
            comp: reference_K[comp] / max(reference_K[heavy_key], 1e-30)
            for comp in comps
        }
        latent_heat_correction = self._truthy_param(self.get_param(
            'cmo_latent_heat_correction',
            self.get_param('latent_heat_correction', False),
        ))
        component_hvap = {}
        feed_hvap = None
        if latent_heat_correction:
            reference_T = volatility_data['reference_temperature']
            for comp in comps:
                props = self.thermo.props.get(comp)
                try:
                    value = float(self.thermo.Hvap_at_T(comp, reference_T))
                except Exception:
                    value = float(props.Hvap if props and props.Hvap else 30.0)
                if not math.isfinite(value) or value <= 0.0:
                    raise UnitOperationError(
                        f"CMODistillation '{self.unit_id}' requires positive Hvap "
                        f"for {comp}"
                    )
                component_hvap[comp] = value
            feed_hvap = sum(
                feed_z.get(comp, 0.0) * component_hvap[comp]
                for comp in comps
            )

        flow_scale = max(inlet.F, 1.0)
        component_floor = float(self.get_param('component_scale_floor', 1e-4))
        component_scales = {
            comp: max(inlet.F * feed_z.get(comp, 0.0), flow_scale * component_floor, 1e-12)
            for comp in comps
        }
        D_initial = self._cmo_initial_distillate_flow(inlet, distillate_spec, feed_z)
        x_initial = self._cmo_initial_profile(
            inlet, comps, feed_z, N, RR, q, D_initial
        )
        T_initial = self._cmo_temperature_profile(x_initial, pressures)
        z0 = self._cmo_pack_variables(
            D_initial, T_initial, x_initial, comps, inlet.F, T_min, T_max
        )

        feed_index = feed_stage - 1
        labels: list[tuple] = []
        feed_mw = sum(
            feed_z.get(comp, 0.0) * self.thermo.props[comp].MW
            for comp in comps
        )
        distillate_spec_scale = (
            max(abs(float(distillate_spec['value'])), flow_scale * feed_mw, 1.0)
            if distillate_spec['kind'] == 'mass'
            else flow_scale
        )

        def vapor_hvap(composition: dict[str, float]) -> float:
            total = sum(max(composition.get(comp, 0.0), 0.0) for comp in comps)
            if total <= 1e-30:
                raise ValueError("empty vapor composition")
            return sum(
                max(composition.get(comp, 0.0), 0.0) * component_hvap[comp]
                for comp in comps
            ) / total

        def traffic_for_state(D: float, y: list[dict[str, float]]) -> dict:
            B = inlet.F - D
            L_rect = RR * D
            V_rect = (RR + 1.0) * D
            L_strip = L_rect + q * inlet.F
            V_strip = V_rect - (1.0 - q) * inlet.F
            if min(D, B, V_rect, L_strip, V_strip) <= 1e-12:
                raise ValueError("infeasible fixed CMO traffic")
            if not latent_heat_correction:
                liquid_down = [
                    B if stage == N - 1 else (
                        L_rect if stage < feed_index else L_strip
                    )
                    for stage in range(N)
                ]
                vapor_up = [
                    0.0 if stage == 0 else (
                        V_rect if stage <= feed_index else V_strip
                    )
                    for stage in range(N)
                ]
                return {
                    'D': D,
                    'B': B,
                    'L_down': liquid_down,
                    'V_up': vapor_up,
                    'rectifying_latent_heat_flow': None,
                    'stripping_latent_heat_flow': None,
                }

            if N < 2 or feed_index < 1:
                raise ValueError(
                    "latent-heat correction requires a feed stage below the condenser"
                )
            rectifying_heat = V_rect * vapor_hvap(y[1])
            stripping_heat = (
                rectifying_heat - (1.0 - q) * inlet.F * feed_hvap
            )
            if min(rectifying_heat, stripping_heat) <= 1e-12:
                raise ValueError("infeasible latent-heat flow")

            vapor_up = [0.0] * N
            for stage in range(1, N):
                heat_flow = (
                    rectifying_heat if stage <= feed_index else stripping_heat
                )
                vapor_up[stage] = heat_flow / vapor_hvap(y[stage])

            liquid_down = [0.0] * N
            for stage in range(N - 1):
                if stage < feed_index:
                    liquid_down[stage] = vapor_up[stage + 1] - D
                else:
                    liquid_down[stage] = vapor_up[stage + 1] + B
            liquid_down[-1] = B
            if min(liquid_down) <= 1e-12 or min(vapor_up[1:]) <= 1e-12:
                raise ValueError("infeasible latent-heat-corrected traffic")
            return {
                'D': D,
                'B': B,
                'L_down': liquid_down,
                'V_up': vapor_up,
                'rectifying_latent_heat_flow': rectifying_heat,
                'stripping_latent_heat_flow': stripping_heat,
            }

        def liquid_down_flow(stage: int, traffic: dict) -> float:
            return traffic['L_down'][stage]

        def vapor_up_flow(stage: int, traffic: dict) -> float:
            return traffic['V_up'][stage]

        def residual(vector):
            nonlocal labels
            decoded = self._cmo_decode_variables(
                vector, comps, N, inlet.F, T_min, T_max
            )
            D = decoded['D']
            T = decoded['T']
            x = decoded['x']
            try:
                stage_properties = [
                    self._cmo_explicit_stage_properties(
                        T[stage], stage_x, pressures[stage], comps
                    )
                    for stage, stage_x in enumerate(x)
                ]
            except Exception:
                return np.ones(N * nc + 1, dtype=float) * 1e3
            y = [item['y'] for item in stage_properties]
            try:
                traffic = traffic_for_state(D, y)
            except ValueError:
                return np.ones(N * nc + 1, dtype=float) * 1e3
            D_vapor = condenser_vapor_fraction * D
            D_liquid = (1.0 - condenser_vapor_fraction) * D

            values = []
            current_labels = []
            for stage in range(N):
                liquid_in_flow = 0.0 if stage == 0 else liquid_down_flow(stage - 1, traffic)
                liquid_in_comp = None if stage == 0 else x[stage - 1]
                vapor_in_flow = 0.0 if stage == N - 1 else vapor_up_flow(stage + 1, traffic)
                vapor_in_comp = None if stage == N - 1 else y[stage + 1]
                liquid_out_flow = liquid_down_flow(stage, traffic)
                vapor_out_flow = vapor_up_flow(stage, traffic)

                for comp in comps[:-1]:
                    incoming = (
                        (inlet.F * feed_z.get(comp, 0.0) if stage == feed_index else 0.0)
                    )
                    if liquid_in_comp is not None:
                        incoming += liquid_in_flow * liquid_in_comp.get(comp, 0.0)
                    if vapor_in_comp is not None:
                        incoming += vapor_in_flow * vapor_in_comp.get(comp, 0.0)

                    if stage == 0:
                        outgoing = (
                            (liquid_out_flow + D_liquid) * x[stage].get(comp, 0.0)
                            + D_vapor * y[stage].get(comp, 0.0)
                        )
                    else:
                        outgoing = (
                            liquid_out_flow * x[stage].get(comp, 0.0)
                            + vapor_out_flow * y[stage].get(comp, 0.0)
                        )

                    scale = component_scales[comp]
                    values.append((incoming - outgoing) / scale)
                    current_labels.append(('component', stage + 1, comp, scale))

                values.append(stage_properties[stage]['bubble'])
                current_labels.append(('bubble', stage + 1, None, 1.0))

            if distillate_spec['kind'] == 'mass':
                product_comp = {
                    comp: (
                        (1.0 - condenser_vapor_fraction) * x[0].get(comp, 0.0)
                        + condenser_vapor_fraction * y[0].get(comp, 0.0)
                    )
                    for comp in comps
                }
                product_mw = sum(
                    product_comp.get(comp, 0.0) * self.thermo.props[comp].MW
                    for comp in comps
                )
                values.append(
                    (D * product_mw - float(distillate_spec['value']))
                    / distillate_spec_scale
                )
                current_labels.append(
                    ('distillate_spec', None, None, distillate_spec_scale)
                )
            else:
                values.append(
                    (D - float(distillate_spec['value'])) / distillate_spec_scale
                )
                current_labels.append(
                    ('distillate_spec', None, None, distillate_spec_scale)
                )

            if not labels:
                labels = current_labels
            return np.array(values, dtype=float)

        def cmo_jacobian(vector, _f0, rel_step: float):
            from scipy.sparse import lil_matrix

            decoded = self._cmo_decode_variables(
                vector, comps, N, inlet.F, T_min, T_max
            )
            D = decoded['D']
            T = decoded['T']
            x = decoded['x']
            try:
                stage_properties = [
                    self._cmo_explicit_stage_properties(
                        T[stage], x[stage], pressures[stage], comps
                    )
                    for stage in range(N)
                ]
            except Exception:
                return None
            y = [item['y'] for item in stage_properties]
            try:
                traffic = traffic_for_state(D, y)
            except ValueError:
                return None
            D_vapor = condenser_vapor_fraction * D
            D_liquid = (1.0 - condenser_vapor_fraction) * D

            n_vars = 1 + N * nc
            matrix = lil_matrix((n_vars, n_vars), dtype=float)
            D_step = rel_step * max(abs(float(vector[0])), 1.0)
            D_theta = float(np.clip(vector[0] + D_step, -60.0, 60.0))
            perturbed_D = inlet.F / (1.0 + math.exp(-D_theta))
            try:
                D_traffic = traffic_for_state(perturbed_D, y)
            except ValueError:
                return None
            dD = (perturbed_D - D) / D_step
            dL_dD = [
                (D_traffic['L_down'][stage] - traffic['L_down'][stage]) / D_step
                for stage in range(N)
            ]
            dV_dD = [
                (D_traffic['V_up'][stage] - traffic['V_up'][stage]) / D_step
                for stage in range(N)
            ]

            for stage in range(N):
                for ci, comp in enumerate(comps[:-1]):
                    derivative = 0.0
                    if stage > 0:
                        derivative += (
                            dL_dD[stage - 1]
                            * x[stage - 1].get(comp, 0.0)
                        )
                    if stage < N - 1:
                        derivative += (
                            dV_dD[stage + 1]
                            * y[stage + 1].get(comp, 0.0)
                        )
                    if stage == 0:
                        derivative -= (
                            dL_dD[stage] + (1.0 - condenser_vapor_fraction) * dD
                        ) * x[stage].get(comp, 0.0)
                        derivative -= (
                            condenser_vapor_fraction * dD
                            * y[stage].get(comp, 0.0)
                        )
                    else:
                        derivative -= (
                            dL_dD[stage]
                            * x[stage].get(comp, 0.0)
                            + dV_dD[stage]
                            * y[stage].get(comp, 0.0)
                        )
                    matrix[stage * nc + ci, 0] = derivative / component_scales[comp]

            spec_row = N * nc
            if distillate_spec['kind'] == 'mass':
                product_mw = sum(
                    (
                        (1.0 - condenser_vapor_fraction) * x[0].get(comp, 0.0)
                        + condenser_vapor_fraction * y[0].get(comp, 0.0)
                    ) * self.thermo.props[comp].MW
                    for comp in comps
                )
                matrix[spec_row, 0] = (
                    dD * product_mw / distillate_spec_scale
                )
            else:
                matrix[spec_row, 0] = dD / distillate_spec_scale

            evaluations = 0
            logits_start = 1 + N
            for source_stage in range(N):
                local_columns = [1 + source_stage] + [
                    logits_start + source_stage * (nc - 1) + index
                    for index in range(nc - 1)
                ]
                for local_index, column in enumerate(local_columns):
                    step = rel_step * max(abs(float(vector[column])), 1.0)
                    if local_index == 0:
                        temperature_theta = float(
                            np.clip(vector[column] + step, -60.0, 60.0)
                        )
                        perturbed_T = T_min + (T_max - T_min) / (
                            1.0 + math.exp(-temperature_theta)
                        )
                        perturbed_x = x[source_stage]
                    else:
                        stage_logits_start = (
                            logits_start + source_stage * (nc - 1)
                        )
                        logits = np.array(
                            list(
                                vector[
                                    stage_logits_start:
                                    stage_logits_start + nc - 1
                                ]
                            )
                            + [0.0],
                            dtype=float,
                        )
                        logits[local_index - 1] += step
                        logits = np.clip(logits, -60.0, 60.0)
                        logits -= np.max(logits)
                        exp_values = np.exp(logits)
                        fractions = exp_values / np.sum(exp_values)
                        perturbed_T = T[source_stage]
                        perturbed_x = {
                            comp: float(fractions[index])
                            for index, comp in enumerate(comps)
                        }
                    try:
                        perturbed_properties = self._cmo_explicit_stage_properties(
                            perturbed_T,
                            perturbed_x,
                            pressures[source_stage],
                            comps,
                        )
                    except Exception:
                        return None
                    evaluations += 1

                    if local_index == 0:
                        dx = {comp: 0.0 for comp in comps}
                    else:
                        varied_comp = comps[local_index - 1]
                        varied_fraction = x[source_stage].get(varied_comp, 0.0)
                        dx = {
                            comp: x[source_stage].get(comp, 0.0)
                            * (
                                (1.0 if comp == varied_comp else 0.0)
                                - varied_fraction
                            )
                            for comp in comps
                        }
                    dy = {
                        comp: (
                            perturbed_properties['y'].get(comp, 0.0)
                            - y[source_stage].get(comp, 0.0)
                        ) / step
                        for comp in comps
                    }
                    dbubble = (
                        perturbed_properties['bubble']
                        - stage_properties[source_stage]['bubble']
                    ) / step
                    if latent_heat_correction:
                        perturbed_y = list(y)
                        perturbed_y[source_stage] = perturbed_properties['y']
                        try:
                            perturbed_traffic = traffic_for_state(D, perturbed_y)
                        except ValueError:
                            return None
                        dL = [
                            (
                                perturbed_traffic['L_down'][stage]
                                - traffic['L_down'][stage]
                            ) / step
                            for stage in range(N)
                        ]
                        dV = [
                            (
                                perturbed_traffic['V_up'][stage]
                                - traffic['V_up'][stage]
                            ) / step
                            for stage in range(N)
                        ]
                    else:
                        dL = [0.0] * N
                        dV = [0.0] * N

                    target_stages = (
                        range(N)
                        if latent_heat_correction and source_stage == 1
                        else (
                            stage
                            for stage in (
                                source_stage - 1,
                                source_stage,
                                source_stage + 1,
                            )
                            if 0 <= stage < N
                        )
                    )
                    for target_stage in target_stages:
                        liquid_in_flow = (
                            0.0
                            if target_stage == 0
                            else liquid_down_flow(target_stage - 1, traffic)
                        )
                        vapor_in_flow = (
                            0.0
                            if target_stage == N - 1
                            else vapor_up_flow(target_stage + 1, traffic)
                        )
                        liquid_out_flow = liquid_down_flow(target_stage, traffic)
                        vapor_out_flow = vapor_up_flow(target_stage, traffic)

                        for ci, comp in enumerate(comps[:-1]):
                            derivative = 0.0
                            if target_stage > 0:
                                derivative += (
                                    dL[target_stage - 1]
                                    * x[target_stage - 1].get(comp, 0.0)
                                )
                            if source_stage == target_stage - 1:
                                derivative += liquid_in_flow * dx[comp]
                            if target_stage < N - 1:
                                derivative += (
                                    dV[target_stage + 1]
                                    * y[target_stage + 1].get(comp, 0.0)
                                )
                            if source_stage == target_stage + 1:
                                derivative += vapor_in_flow * dy[comp]
                            if target_stage == 0:
                                derivative -= (
                                    dL[target_stage]
                                    * x[target_stage].get(comp, 0.0)
                                )
                                if source_stage == target_stage:
                                    derivative -= (
                                        liquid_out_flow + D_liquid
                                    ) * dx[comp]
                                    derivative -= D_vapor * dy[comp]
                            else:
                                derivative -= (
                                    dL[target_stage]
                                    * x[target_stage].get(comp, 0.0)
                                    + dV[target_stage]
                                    * y[target_stage].get(comp, 0.0)
                                )
                                if source_stage == target_stage:
                                    derivative -= (
                                        liquid_out_flow * dx[comp]
                                        + vapor_out_flow * dy[comp]
                                    )
                            matrix[target_stage * nc + ci, column] = (
                                derivative / component_scales[comp]
                            )

                    matrix[source_stage * nc + nc - 1, column] = dbubble
                    if (
                        distillate_spec['kind'] == 'mass'
                        and source_stage == 0
                        and local_index > 0
                    ):
                        matrix[spec_row, column] = D * sum(
                            self.thermo.props[comp].MW * (
                                (1.0 - condenser_vapor_fraction) * dx[comp]
                                + condenser_vapor_fraction * dy[comp]
                            )
                            for comp in comps
                        ) / distillate_spec_scale

            return matrix.tocsr(), evaluations

        sparsity = self._cmo_sparsity(N, nc)
        tolerance = float(self.get_param('cmo_tolerance', self.get_param('mesh_tolerance', 1e-7)))
        acceptable = float(
            self.get_param(
                'cmo_acceptable_residual',
                max(1000.0 * tolerance, 1e-4),
            )
        )
        solver_options = {
            'mesh_tolerance': tolerance,
            'acceptable_mesh_residual': acceptable,
            'max_iterations': int(self.get_param('cmo_max_iterations', self.get_param('max_iterations', 60))),
            'max_jacobian_evaluations': int(
                self.get_param(
                    'cmo_max_jacobian_evaluations',
                    self.get_param('max_jacobian_evaluations', 60),
                )
            ),
            'line_search_steps': int(self.get_param('cmo_line_search_steps', self.get_param('line_search_steps', 16))),
            'finite_difference_rel_step': float(
                self.get_param(
                    'cmo_finite_difference_rel_step',
                    self.get_param('finite_difference_rel_step', 1e-6),
                )
            ),
        }
        solution = self._sparse_newton_solve(
            residual,
            sparsity,
            z0,
            solver_options,
            jacobian=cmo_jacobian,
        )
        quality_context = getattr(self.thermo, 'quality_context', None)
        if quality_context is None:
            final_residual = residual(solution['x'])
        else:
            with quality_context(phase='residual_diagnostics', affects_result=False):
                final_residual = residual(solution['x'])
        residual_norm = float(np.linalg.norm(final_residual, ord=np.inf))
        if not solution['success']:
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' MES solve failed "
                f"(residual {residual_norm:.2e}): {solution['message']}"
            )

        decoded = self._cmo_decode_variables(
            solution['x'], comps, N, inlet.F, T_min, T_max
        )
        D = decoded['D']
        B = inlet.F - D
        T = decoded['T']
        x = decoded['x']
        stage_equilibrium = [
            self._cmo_explicit_stage_properties(
                T[stage], stage_x, pressures[stage], comps
            )
            for stage, stage_x in enumerate(x)
        ]
        raw_y = [item['y'] for item in stage_equilibrium]
        traffic = traffic_for_state(D, raw_y)
        y = []
        for stage_y in raw_y:
            total = sum(max(value, 0.0) for value in stage_y.values())
            y.append({
                comp: max(stage_y.get(comp, 0.0), 0.0) / max(total, 1e-30)
                for comp in comps
            })
        stage_K = [item['K'] for item in stage_equilibrium]
        x_D = {comp: float(x[0].get(comp, 0.0)) for comp in comps}
        y_D = {comp: float(y[0].get(comp, 0.0)) for comp in comps}
        x_B = {comp: float(x[-1].get(comp, 0.0)) for comp in comps}

        D_vapor = condenser_vapor_fraction * D
        D_liquid = (1.0 - condenser_vapor_fraction) * D
        if condenser_vapor_fraction >= 1.0 - 1e-12:
            distillate_comp = dict(y_D)
            distillate = self.thermo.calculate_state(
                float(T[0]), float(pressures[0]), D, distillate_comp,
                phase='vapor', flash=False,
            )
        elif condenser_vapor_fraction <= 1e-12:
            distillate_comp = dict(x_D)
            distillate = self.thermo.calculate_state(
                float(T[0]), float(pressures[0]), D, distillate_comp,
                phase='liquid', flash=False,
            )
        else:
            distillate_comp = self._normalize({
                comp: D_liquid * x_D.get(comp, 0.0) + D_vapor * y_D.get(comp, 0.0)
                for comp in comps
            })
            distillate_h = (
                (1.0 - condenser_vapor_fraction)
                * self.thermo.mixture_enthalpy(
                    x_D, float(T[0]), vapor_fraction=0.0, P=float(pressures[0])
                )
                + condenser_vapor_fraction
                * self.thermo.mixture_enthalpy(
                    y_D, float(T[0]), vapor_fraction=1.0, P=float(pressures[0])
                )
            )
            distillate = self._two_phase_stream(
                float(T[0]), float(pressures[0]), D, distillate_comp,
                x_D, y_D, condenser_vapor_fraction, distillate_h,
            )
        bottoms = self.thermo.calculate_state(
            float(T[-1]), float(pressures[-1]), B, x_B,
            phase='liquid', flash=False
        )

        V_cond = traffic['V_up'][1]
        condensed_flow = max(V_cond - D_vapor, 0.0)
        Q_cond = -condensed_flow * McCabeThieleDistillation._mixture_hvap(self, y_D, float(T[0])) * 1000.0
        H_in = inlet.F * inlet.H
        H_out = D * distillate.H + B * bottoms.H
        Q_net = H_out - H_in
        Q_reb = Q_net - Q_cond
        component_balance_error = self._external_component_balance_error_vle(
            comps, inlet, feed_z, distillate, bottoms, [], flow_scale
        )
        warnings = []
        if residual_norm > tolerance:
            warnings.append(
                f"MES solver accepted residual {residual_norm:.2e}"
            )

        outlets = {'distillate': distillate, 'bottoms': bottoms}
        if 1e-12 < condenser_vapor_fraction < 1.0 - 1e-12:
            outlets['distillate_liquid'] = self.thermo.calculate_state(
                float(T[0]), float(pressures[0]), D_liquid, x_D,
                phase='liquid', flash=False,
            )
            outlets['distillate_vapor'] = self.thermo.calculate_state(
                float(T[0]), float(pressures[0]), D_vapor, y_D,
                phase='vapor', flash=False,
            )

        return UnitResult(
            outlet_streams=outlets,
            heat_duty=Q_net,
            performance={
                'method': 'CMO-MES',
                'N_stages': N,
                'feed_stage': feed_stage,
                'reflux_ratio': RR,
                'feed_thermal_condition_q': q,
                'condenser_type': condenser,
                'distillate_vapor_fraction': condenser_vapor_fraction,
                'distillate_flow': D,
                'distillate_liquid_flow': D_liquid,
                'distillate_vapor_flow': D_vapor,
                'bottoms_flow': B,
                'T_top_C': float(T[0] - 273.15),
                'T_bottom_C': float(T[-1] - 273.15),
                'stage_temperatures_C': [float(value - 273.15) for value in T],
                'stage_pressures_bar': [float(value) for value in pressures],
                'stage_liquid_compositions': [
                    {comp: float(stage_x.get(comp, 0.0)) for comp in comps}
                    for stage_x in x
                ],
                'stage_vapor_compositions': [
                    {comp: float(stage_y.get(comp, 0.0)) for comp in comps}
                    for stage_y in y
                ],
                'stage_K_values': [
                    {comp: float(stage_values.get(comp, 0.0)) for comp in comps}
                    for stage_values in stage_K
                ],
                'liquid_flows': [
                    float(liquid_down_flow(stage, traffic))
                    for stage in range(N)
                ],
                'vapor_flows': [
                    float(vapor_up_flow(stage, traffic))
                    for stage in range(N)
                ],
                'light_key': light_key,
                'heavy_key': heavy_key,
                'reference_K_values': reference_K,
                'relative_volatilities': relative_volatility,
                'relative_volatility_reference_temperature_C': (
                    volatility_data['reference_temperature'] - 273.15
                ),
                'temperature_profile_basis': 'explicit_stage_bubble_equations',
                'equilibrium_basis': 'stage_bubble_K_values',
                'latent_heat_correction': latent_heat_correction,
                'constant_component_hvap': (
                    dict(component_hvap) if latent_heat_correction else None
                ),
                'rectifying_latent_heat_flow': traffic[
                    'rectifying_latent_heat_flow'
                ],
                'stripping_latent_heat_flow': traffic[
                    'stripping_latent_heat_flow'
                ],
                'mes_residual': residual_norm,
                'component_balance_error': component_balance_error,
                'solver': 'sparse_damped_newton_cmo_mes',
                'jacobian_method': 'cmo_analytic_local_thermo',
                'solver_iterations': int(solution['iterations']),
                'function_evaluations': int(solution['function_evaluations']),
                'jacobian_evaluations': int(solution['jacobian_evaluations']),
                'finite_difference_rel_step': float(solver_options['finite_difference_rel_step']),
                'solver_message': str(solution['message']),
                'residual_diagnostics': self._rigorous2_residual_diagnostics(
                    final_residual, labels
                ),
                'condenser_duty_kW': Q_cond / 3600.0,
                'reboiler_duty_kW': Q_reb / 3600.0,
            },
            warnings=warnings,
        )

    def _cmo_initial_distillate_flow(
        self,
        inlet: StreamState,
        distillate_spec: dict,
        feed_z: dict[str, float],
    ) -> float:
        if distillate_spec['kind'] == 'molar':
            guess = float(distillate_spec['value'])
        else:
            mw = sum(
                feed_z.get(comp, 0.0) * self.thermo.props[comp].MW
                for comp in feed_z
            )
            guess = float(distillate_spec['value']) / max(mw, 1e-12)
        return min(max(guess, inlet.F * 1e-6), inlet.F * (1.0 - 1e-6))

    def _cmo_initial_profile(
        self,
        inlet: StreamState,
        comps: list[str],
        feed_z: dict[str, float],
        N: int,
        RR: float,
        q: float,
        D_guess: float,
    ) -> list[dict[str, float]]:
        initializer = str(
            self.get_param('cmo_initializer', 'volatility_profile')
        ).strip().lower().replace('-', '_').replace(' ', '_')
        x_top = None
        x_bottom = None
        if initializer in ('cmo', 'legacy_cmo', 'cmo_distillation'):
            try:
                params = dict(self.params)
                params['D_to_F'] = D_guess / max(inlet.F, 1e-30)
                params['reflux_ratio'] = RR
                params['q'] = q
                init = McCabeThieleDistillation(
                    f"{self.unit_id}_estimate_init", self.thermo, params
                ).solve({'feed': inlet})
                x_top = self._dense_composition(
                    init.outlet_streams['distillate'].composition, comps
                )
                x_bottom = self._dense_composition(
                    init.outlet_streams['bottoms'].composition, comps
                )
            except Exception:
                x_top = None
                x_bottom = None

        if x_top is None or x_bottom is None:
            K = McCabeThieleDistillation._relative_volatility_data(self, comps, inlet)['K_values']
            heavy = min(comps, key=lambda comp: K[comp])
            severity = float(self.get_param(
                'cmo_initial_severity',
                1.0,
            ))
            severity = min(max(severity, 0.25), 8.0)
            factors = {
                comp: math.exp(min(max(
                    math.log(max(K[comp] / max(K[heavy], 1e-30), 1e-12))
                    * severity,
                    -60.0,
                ), 60.0))
                for comp in comps
            }
            x_top = self._normalize({
                comp: feed_z.get(comp, 0.0) * factors[comp]
                for comp in comps
            })
            x_bottom = self._normalize({
                comp: feed_z.get(comp, 0.0) / factors[comp]
                for comp in comps
            })

        profile = []
        for stage in range(N):
            frac = stage / max(N - 1, 1)
            profile.append(self._normalize({
                comp: (
                    (1.0 - frac) * max(x_top.get(comp, 0.0), 1e-14)
                    + frac * max(x_bottom.get(comp, 0.0), 1e-14)
                )
                for comp in comps
            }))
        return profile

    def _cmo_pack_variables(
        self,
        D: float,
        T: list[float],
        x: list[dict[str, float]],
        comps: list[str],
        F: float,
        T_min: float,
        T_max: float,
    ):
        import numpy as np

        fraction = min(max(D / max(F, 1e-30), 1e-8), 1.0 - 1e-8)
        values = [math.log(fraction / (1.0 - fraction))]
        T_span = T_max - T_min
        for temperature in T:
            reduced = min(
                max((float(temperature) - T_min) / T_span, 1e-8),
                1.0 - 1e-8,
            )
            values.append(math.log(reduced / (1.0 - reduced)))
        for stage_x in x:
            last = max(stage_x.get(comps[-1], 0.0), 1e-14)
            for comp in comps[:-1]:
                values.append(math.log(max(stage_x.get(comp, 0.0), 1e-14) / last))
        return np.array(values, dtype=float)

    def _cmo_decode_variables(
        self,
        vector,
        comps: list[str],
        N: int,
        F: float,
        T_min: float,
        T_max: float,
    ) -> dict:
        import numpy as np

        theta_D = float(np.clip(vector[0], -60.0, 60.0))
        D = F / (1.0 + math.exp(-theta_D))
        T_theta = np.clip(np.array(vector[1:1 + N], dtype=float), -60.0, 60.0)
        T = T_min + (T_max - T_min) / (1.0 + np.exp(-T_theta))
        x = []
        index = 1 + N
        nc = len(comps)
        for _ in range(N):
            logits = np.array(list(vector[index:index + nc - 1]) + [0.0], dtype=float)
            index += nc - 1
            logits = np.clip(logits, -60.0, 60.0)
            logits -= np.max(logits)
            exp_values = np.exp(logits)
            fractions = exp_values / np.sum(exp_values)
            x.append({
                comp: float(fractions[i])
                for i, comp in enumerate(comps)
            })
        return {'D': float(D), 'T': T, 'x': x}

    def _cmo_sparsity(self, N: int, nc: int):
        from scipy.sparse import lil_matrix

        n_rows = N * nc + 1
        n_cols = 1 + N * nc
        matrix = lil_matrix((n_rows, n_cols), dtype=int)

        def mark_stage(row: int, stage: int) -> None:
            if not 0 <= stage < N:
                return
            matrix[row, 1 + stage] = 1
            start = 1 + N + stage * (nc - 1)
            for col in range(start, start + nc - 1):
                matrix[row, col] = 1

        row = 0
        for stage in range(N):
            for _ in range(nc - 1):
                matrix[row, 0] = 1
                mark_stage(row, stage)
                mark_stage(row, stage - 1)
                mark_stage(row, stage + 1)
                row += 1
            mark_stage(row, stage)
            row += 1
        matrix[row, 0] = 1
        mark_stage(row, 0)
        return matrix.tocsr()

    def _cmo_explicit_stage_properties(
        self,
        T: float,
        x_stage: dict[str, float],
        pressure: float,
        comps: list[str],
    ) -> dict:
        K = None
        if hasattr(self.thermo, 'K_values'):
            try:
                raw_K = self.thermo.K_values(float(T), pressure, x_stage)
                K = {
                    comp: min(max(float(raw_K.get(comp, 1.0)), 1e-12), 1e12)
                    for comp in comps
                }
            except Exception:
                K = None

        if K is None and hasattr(self.thermo, 'activity_coefficients'):
            gamma = self.thermo.activity_coefficients(float(T), x_stage)
            K = {
                comp: min(
                    max(
                        gamma.get(comp, 1.0)
                        * self.thermo.Psat(comp, float(T))
                        / pressure,
                        1e-12,
                    ),
                    1e12,
                )
                for comp in comps
            }
        elif K is None:
            K = {
                comp: min(
                    max(self.thermo.K_value(comp, float(T), pressure), 1e-12),
                    1e12,
                )
                for comp in comps
            }

        kx = {
            comp: max(K[comp] * x_stage.get(comp, 0.0), 0.0)
            for comp in comps
        }
        total = sum(kx.values())
        if total <= 0.0:
            raise UnitOperationError(
                f"CMODistillation '{self.unit_id}' produced an empty vapor composition"
            )
        return {
            'K': K,
            'y': kx,
            'bubble': total - 1.0,
        }

    def _cmo_temperature_profile(
        self,
        x: list[dict[str, float]],
        pressures: list[float],
    ) -> list[float]:
        temperatures = []
        for stage_x, pressure in zip(x, pressures):
            try:
                temperatures.append(float(self.thermo.bubble_point_T(stage_x, pressure)))
            except Exception:
                temperatures.append(float(
                    self._bubble_temperature_from_equation(stage_x, pressure)
                ))
        return temperatures
