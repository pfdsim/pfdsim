"""
Shared base types for unit operation calculations.
"""

import math
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics import StreamState, IdealThermodynamics
else:
    from thermodynamics import StreamState, IdealThermodynamics
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .particle_size_distributions import (
        propagate_particle_size_distributions,
    )
else:
    from particle_size_distributions import (
        propagate_particle_size_distributions,
    )


class UnitOperationError(Exception):
    """Error in unit operation calculation"""
    pass


@dataclass
class UnitResult:
    """Result from a unit operation calculation"""
    outlet_streams: dict[str, StreamState]  # port_name -> state
    heat_duty: float = 0.0  # kJ/h (positive = heat added)
    work: float = 0.0  # kJ/h (positive = work done on system)
    performance: dict = field(default_factory=dict)  # Unit-specific metrics
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'outlets': {k: v.to_dict() for k, v in self.outlet_streams.items()},
            'heat_duty': self.heat_duty,
            'heat_duty_kW': self.heat_duty / 3600,
            'work': self.work,
            'work_kW': self.work / 3600,
            'performance': self.performance,
            'warnings': self.warnings,
        }


class UnitOperation:
    """Base class for unit operations"""

    supports_permanent_solids = False
    particle_size_behavior = 'unsupported'

    def __init_subclass__(cls, **kwargs):
        """Apply the material-capability guard to each concrete solve method."""
        super().__init_subclass__(**kwargs)
        solve_method = cls.__dict__.get('solve')
        if (
            solve_method is None
            or getattr(solve_method, '_permanent_solid_guarded', False)
        ):
            return

        @wraps(solve_method)
        def guarded_solve(self, inlets, *args, **kwargs):
            self.validate_permanent_solid_inlets(inlets)
            result = solve_method(self, inlets, *args, **kwargs)
            if self.particle_size_behavior == 'nonselective':
                for outlet_name, outlet in result.outlet_streams.items():
                    propagate_particle_size_distributions(
                        self.particle_size_sources_for_outlet(
                            outlet_name, inlets, result
                        ),
                        outlet,
                    )
            return result

        guarded_solve._permanent_solid_guarded = True
        cls.solve = guarded_solve

    def __init__(self, unit_id: str, thermo: IdealThermodynamics, params: dict):
        self.unit_id = unit_id
        self.thermo = thermo
        self.params = params
        self.solve_context = {}

    @staticmethod
    def _permanent_solid_components(inlets: dict[str, StreamState]) -> list[str]:
        components = set()
        for stream in inlets.values():
            components.update(
                component
                for component, flow in stream.solid_component_flows.items()
                if float(flow) > 1.0e-15
            )
        return sorted(components)

    def validate_permanent_solid_inlets(
        self,
        inlets: dict[str, StreamState],
    ) -> None:
        """Reject solid-bearing feeds unless the unit declares a routing model."""
        components = self._permanent_solid_components(inlets)
        if components and not self.supports_permanent_solids:
            raise UnitOperationError(
                f"{type(self).__name__} '{self.unit_id}' does not support "
                "permanent-solid-bearing inlet streams; active solid "
                f"component(s): {', '.join(components)}"
            )

    def particle_size_sources_for_outlet(
        self,
        outlet_name: str,
        inlets: dict[str, StreamState],
        result: UnitResult,
    ):
        """Return populations feeding one non-size-selective outlet."""
        if len(inlets) == 1:
            return inlets.values()
        raise UnitOperationError(
            f"{type(self).__name__} '{self.unit_id}' must map inlet particle "
            f"populations to outlet '{outlet_name}'"
        )

    def _consume_recycle_warm_start_skip(self) -> bool:
        remaining = int(getattr(self, '_recycle_warm_start_skip_count', 0) or 0)
        if remaining <= 0:
            return False
        self._recycle_warm_start_skip_count = remaining - 1
        return True

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        """
        Solve the unit operation.

        Args:
            inlets: Dictionary of inlet port name -> StreamState

        Returns:
            UnitResult with outlet streams and duties
        """
        raise NotImplementedError("Subclass must implement solve()")

    def get_param(self, name: str, default=None):
        """Get parameter value with default"""
        # Try exact match first
        if name in self.params:
            return self.params[name]
        # Try case-insensitive
        name_lower = name.lower()
        for k, v in self.params.items():
            if k.lower() == name_lower:
                return v
        return default

    def get_param_unit(self, name: str) -> Optional[str]:
        """Get the original unit string for a parameter, if available."""
        unit_name = f"__unit__{name}"
        if unit_name in self.params:
            return self.params[unit_name]
        unit_name_lower = unit_name.lower()
        for key, value in self.params.items():
            if str(key).lower() == unit_name_lower:
                return value
        return None

    def get_temperature_param(self, name: str, default=None):
        """
        Get a temperature parameter in Kelvin.

        PFD parsing already converts explicit Celsius/Fahrenheit units to
        Kelvin. The legacy no-unit convention treats values below 200 as
        Celsius for hand-written Python unit parameters.
        """
        value = self.get_param(name, default)
        if value is None:
            return None
        value = float(value)
        unit = self.get_param_unit(name)
        if unit:
            return value
        if value < 200.0:
            return value + 273.15
        return value

    def _truthy_param(self, value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in ('0', 'false', 'no', 'off')
        return bool(value)

    def _henry_water_component(self, comps: list[str]) -> str | None:
        explicit = self.get_param('water_component')
        if explicit is not None:
            value = str(explicit).strip()
            if value not in comps:
                raise UnitOperationError(
                    f"{type(self).__name__} '{self.unit_id}' water_component "
                    f"'{value}' is not an active unit component"
                )
            return value
        for comp in comps:
            props = self.thermo.props.get(comp)
            identifiers = {
                str(comp).strip().lower(),
                str(getattr(props, 'symbol', '') or '').strip().lower(),
                str(getattr(props, 'name', '') or '').strip().lower(),
                str(getattr(props, 'formula', '') or '').strip().lower(),
                str(getattr(props, 'CAS', '') or '').strip().lower(),
            }
            if identifiers & {'water', 'h2o', '7732-18-5'}:
                return comp
        return None

    def _henry_noncondensable_candidate(self, comp: str,
                                        reference_T: float) -> bool:
        props = self.thermo.props.get(comp)
        if props is None:
            return False
        Tc = getattr(props, 'Tc', None)
        if Tc is not None:
            return reference_T >= float(Tc)
        Tb = getattr(props, 'Tb', None)
        phase_at_stp = str(getattr(props, 'phase_at_STP', '') or '').lower()
        return (
            phase_at_stp == 'gas'
            and Tb is not None
            and reference_T >= float(Tb) + 25.0
        )

    def _henry_context_from_estimates(
        self,
        comps: list[str],
        *,
        reference_T: float,
        solvent_liquid_moles: dict[str, float],
        potential_liquid_moles: dict[str, float],
        total_available_moles: dict[str, float],
        aqueous_loading_water_moles: float,
    ):
        """Select and freeze a pure-water Henry context before equilibrium."""
        raw_selection = self.get_param('henry_components', 'auto')
        if isinstance(raw_selection, str):
            selection_text = raw_selection.strip()
            selection_key = selection_text.lower().replace('-', '_')
            if selection_key in {'true', 'on', 'enabled'}:
                selection_text = 'auto'
                selection_key = 'auto'
        elif raw_selection is True:
            selection_text = 'auto'
            selection_key = 'auto'
        elif raw_selection is False or raw_selection is None:
            selection_text = 'none'
            selection_key = 'none'
        else:
            selection_text = raw_selection
            selection_key = 'manual'

        water_cutoff = float(self.get_param(
            'henry_water_mole_fraction_min',
            self.get_param('henry_water_cutoff', 0.95),
        ))
        dilute_cutoff = float(self.get_param(
            'henry_dilute_mole_fraction_max',
            self.get_param('henry_dilute_cutoff', 1e-3),
        ))
        if not 0.0 < water_cutoff <= 1.0:
            raise UnitOperationError(
                f"{type(self).__name__} '{self.unit_id}' Henry water cutoff must be in (0, 1]"
            )
        if not 0.0 < dilute_cutoff < 1.0:
            raise UnitOperationError(
                f"{type(self).__name__} '{self.unit_id}' Henry dilute cutoff must be in (0, 1)"
            )

        water = self._henry_water_component(comps)
        info = {
            'selection_mode': 'auto' if selection_key == 'auto' else (
                'none' if selection_key in {'none', 'off', 'false', 'disabled'} else 'manual'
            ),
            'water_component': water,
            'water_mole_fraction_cutoff': water_cutoff,
            'dilute_mole_fraction_cutoff': dilute_cutoff,
            'estimated_water_mole_fraction': 0.0,
            'selection_reference_temperature_C': float(reference_T - 273.15),
            'components': {},
            'enabled': False,
        }
        warnings = []
        if selection_key in {'none', 'off', 'false', 'disabled'}:
            info['disabled_reason'] = 'disabled_by_parameter'
            return None, info, warnings
        if water is None:
            if selection_key == 'auto':
                info['disabled_reason'] = 'no_water_component'
                return None, info, warnings
            raise UnitOperationError(
                f"{type(self).__name__} '{self.unit_id}' cannot apply explicit Henry "
                "components because no water component was identified"
            )

        solvent_total = sum(max(float(value), 0.0)
                            for value in solvent_liquid_moles.values())
        solvent_water = max(float(solvent_liquid_moles.get(water, 0.0)), 0.0)
        potential_total = sum(max(float(value), 0.0)
                              for value in potential_liquid_moles.values())
        potential_water = max(float(potential_liquid_moles.get(water, 0.0)), 0.0)
        solvent_water_fraction = solvent_water / max(solvent_total, 1e-30)
        loaded_water_fraction = potential_water / max(potential_total, 1e-30)
        estimated_water_fraction = min(solvent_water_fraction, loaded_water_fraction)
        info['estimated_water_mole_fraction'] = float(estimated_water_fraction)
        info['solvent_feed_water_mole_fraction'] = float(solvent_water_fraction)
        info['fully_loaded_water_mole_fraction_estimate'] = float(loaded_water_fraction)
        info['aqueous_loading_water_basis_moles'] = float(aqueous_loading_water_moles)

        explicit_components = None
        if selection_key != 'auto':
            if isinstance(selection_text, str):
                explicit_components = [
                    item.strip()
                    for item in selection_text.replace(';', ',').split(',')
                    if item.strip()
                ]
            else:
                explicit_components = [
                    str(item).strip()
                    for item in selection_text
                    if str(item).strip()
                ]

        noncondensable = {
            comp: self._henry_noncondensable_candidate(comp, reference_T)
            for comp in comps
            if comp != water
        }
        selected = []
        reasons = {}
        estimates = {}
        if explicit_components is not None:
            selected = list(dict.fromkeys(explicit_components))
            reasons = {comp: 'manual_override' for comp in selected}
            if estimated_water_fraction < water_cutoff:
                warnings.append(
                    f"{type(self).__name__} '{self.unit_id}' is applying manually selected "
                    f"Henry components at estimated x_{water}={estimated_water_fraction:.4g}, "
                    f"below the pure-water cutoff {water_cutoff:.4g}."
                )
        elif estimated_water_fraction < water_cutoff:
            info['disabled_reason'] = 'estimated_liquid_not_water_rich'
            return None, info, warnings
        else:
            for comp in comps:
                if comp == water:
                    continue
                total_available = max(float(total_available_moles.get(comp, 0.0)), 0.0)
                estimate = total_available / max(
                    aqueous_loading_water_moles + total_available,
                    1e-30,
                )
                estimates[comp] = float(estimate)
                data = self.thermo.henry_component_data(comp)
                if noncondensable.get(comp, False):
                    if data is not None:
                        selected.append(comp)
                        reasons[comp] = 'noncondensable'
                    elif total_available > 0.0:
                        warnings.append(
                            f"{type(self).__name__} '{self.unit_id}' identified "
                            f"{comp} as noncondensable but found no aqueous Henry Hcp; "
                            "the conventional equilibrium path will be used."
                        )
                elif estimate < dilute_cutoff and data is not None:
                    selected.append(comp)
                    reasons[comp] = 'dilute_aqueous_upper_bound'

        if not selected:
            info['disabled_reason'] = 'no_components_selected'
            return None, info, warnings

        try:
            context = self.thermo.create_aqueous_equilibrium_context(
                selected,
                water_component=water,
                pressure_warning_bar=float(self.get_param(
                    'henry_pressure_warning_bar', 20.0
                )),
            )
        except Exception as exc:
            raise UnitOperationError(
                f"{type(self).__name__} '{self.unit_id}' could not create aqueous "
                f"Henry context: {exc}"
            ) from exc

        for comp, data in context.component_data.items():
            info['components'][comp] = {
                'reason': reasons.get(comp, 'manual_override'),
                'estimated_max_aqueous_mole_fraction': estimates.get(comp),
                'Hcp_298_mol_m3_Pa': float(data.hcp_298),
                'B_K': None if data.B is None else float(data.B),
                'quality_h': data.quality_h,
                'quality_b': data.quality_b,
                'quality_score_h': data.quality_score_h,
                'quality_score_b': data.quality_score_b,
                'H_source': data.h_source,
                'B_source': data.b_source,
                'full_quality_temperature_min_C': data.temperature_min_K - 273.15,
                'full_quality_temperature_max_C': data.temperature_max_K - 273.15,
                'temperature_range_source': data.temperature_range_source,
                'richer_temperature_dependence_available_upstream': (
                    data.more_temperature_dependence_available
                ),
                'temperature_model': (
                    data.temperature_correlation.model
                    if data.temperature_correlation is not None
                    else ('van_t_hoff_B' if data.B is not None else 'constant_Hcp')
                ),
                'temperature_correlation_source': (
                    data.temperature_correlation.source
                    if data.temperature_correlation is not None else None
                ),
                'temperature_correlation_DOI': (
                    data.temperature_correlation.doi
                    if data.temperature_correlation is not None else None
                ),
                'temperature_fit_RMS_lnH': (
                    data.temperature_correlation.rms_lnH
                    if data.temperature_correlation is not None else None
                ),
                'temperature_fit_quality_indicator': (
                    data.temperature_correlation.fit_quality
                    if data.temperature_correlation is not None else None
                ),
                'temperature_fit_uncertainty_upper_percent': (
                    data.temperature_correlation.uncertainty_upper_percent
                    if data.temperature_correlation is not None else None
                ),
                'temperature_normalized_to_reference': (
                    data.temperature_correlation.normalize_to_reference
                    if data.temperature_correlation is not None else False
                ),
                'temperature_raw_reference_value_difference_percent': (
                    data.temperature_correlation.raw_reference_value_difference_percent
                    if data.temperature_correlation is not None else None
                ),
                'temperature_raw_reference_slope_difference_K': (
                    data.temperature_correlation.raw_reference_slope_difference_K
                    if data.temperature_correlation is not None else None
                ),
                'temperature_raw_reference_slope_difference_percent': (
                    data.temperature_correlation.raw_reference_slope_difference_percent
                    if data.temperature_correlation is not None else None
                ),
                'Vinf_cm3_per_mol': data.vinf_cm3_per_mol,
                'Vinf_uncertainty_cm3_per_mol': data.vinf_uncertainty_cm3_per_mol,
                'Vinf_estimated_relative_MAE': data.vinf_estimated_relative_mae,
                'Vinf_source': data.vinf_source,
                'Vinf_method': data.vinf_method,
                'Vinf_quality': data.vinf_quality,
                'Vinf_unavailable_reason': data.vinf_unavailable_reason,
                'pressure_correction': (
                    'Krichevsky-Kasarnovsky'
                    if data.vinf_cm3_per_mol is not None else 'unavailable'
                ),
            }
        info['enabled'] = True
        return context, info, warnings

    def _henry_solution_warnings(
        self,
        aqueous_context,
        henry_info: dict,
        liquid_compositions: list[dict[str, float]],
        temperatures: list[float],
        pressures: list[float] | None = None,
    ) -> list[str]:
        """Report departures from the domain used for frozen Henry selection."""
        if aqueous_context is None or not liquid_compositions:
            return []
        warnings = []
        water = aqueous_context.water_component
        minimum_water = min(
            stage_x.get(water, 0.0)
            for stage_x in liquid_compositions
        )
        minimum_temperature = min(float(value) for value in temperatures)
        maximum_temperature = max(float(value) for value in temperatures)
        pressure_values = [float(value) for value in (pressures or [1.0])]
        if len(pressure_values) == 1 and len(temperatures) > 1:
            pressure_values *= len(temperatures)
        if len(pressure_values) != len(temperatures):
            raise UnitOperationError(
                f"{type(self).__name__} '{self.unit_id}' Henry reporting requires "
                "one pressure per solved temperature"
            )
        henry_info['minimum_solved_water_mole_fraction'] = float(minimum_water)
        henry_info['minimum_solved_temperature_C'] = float(
            minimum_temperature - 273.15
        )
        henry_info['maximum_solved_temperature_C'] = float(
            maximum_temperature - 273.15
        )
        henry_info['minimum_solved_pressure_bar'] = min(pressure_values)
        henry_info['maximum_solved_pressure_bar'] = max(pressure_values)
        if minimum_water < henry_info['water_mole_fraction_cutoff']:
            warnings.append(
                f"{type(self).__name__} '{self.unit_id}' selected pure-water Henry "
                f"constants from a pre-solve aqueous estimate, but the solved "
                f"profile reaches x_{water}={minimum_water:.4g} below the "
                f"{henry_info['water_mole_fraction_cutoff']:.4g} cutoff; "
                "Henry components remained frozen for numerical continuity."
            )

        dilute_cutoff = henry_info['dilute_mole_fraction_cutoff']
        for comp, component_info in henry_info['components'].items():
            quality_rows = [
                self.thermo.record_henry_effective_quality(
                    comp, float(T_stage), float(P_stage), aqueous_context
                )
                for T_stage, P_stage in zip(temperatures, pressure_values)
            ]
            effective_rows = [
                row for row in quality_rows if row.get('effective_quality') is not None
            ]
            if effective_rows:
                worst_quality = min(
                    effective_rows,
                    key=lambda row: float(row['effective_quality']),
                )
                component_info.update({
                    'base_correlation_quality': worst_quality['base_quality'],
                    'base_correlation_grade': worst_quality['base_grade'],
                    'minimum_effective_quality': worst_quality['effective_quality'],
                    'minimum_effective_grade': worst_quality['effective_grade'],
                    'maximum_temperature_penalty': max(
                        float(row['temperature_penalty']) for row in effective_rows
                    ),
                    'maximum_pressure_penalty': max(
                        float(row['pressure_penalty']) for row in effective_rows
                    ),
                    'temperature_penalty_per_K': worst_quality['temperature_penalty_per_K'],
                    'pressure_penalty_per_bar_above_10': (
                        worst_quality['pressure_penalty_per_bar_above_10']
                    ),
                    'pressure_quality_method': worst_quality['pressure_method'],
                })
                if worst_quality['effective_grade'] != worst_quality['base_grade']:
                    warnings.append(
                        f"{type(self).__name__} '{self.unit_id}' Henry correlation quality "
                        f"for {comp} falls from {worst_quality['base_grade']} to "
                        f"{worst_quality['effective_grade']} over the solved T/P profile "
                        f"(minimum quality {float(worst_quality['effective_quality']):.3f})."
                    )

            correction_factors = []
            for T_stage, P_stage in zip(temperatures, pressure_values):
                uncorrected = self.thermo.henry_constant_hcp(
                    comp, float(T_stage), aqueous_context
                )
                corrected = self.thermo.henry_constant_hcp(
                    comp, float(T_stage), aqueous_context, P=float(P_stage)
                )
                correction_factors.append(corrected / uncorrected)
            component_info['minimum_Hcp_pressure_correction_factor'] = min(correction_factors)
            component_info['maximum_Hcp_pressure_correction_factor'] = max(correction_factors)

            maximum_fraction = max(
                stage_x.get(comp, 0.0)
                for stage_x in liquid_compositions
            )
            component_info['maximum_solved_aqueous_mole_fraction'] = float(
                maximum_fraction
            )
            if maximum_fraction > dilute_cutoff:
                warnings.append(
                    f"{type(self).__name__} '{self.unit_id}' kept {comp} on its "
                    f"frozen Henry path, but the solved liquid profile reaches "
                    f"x_{comp}={maximum_fraction:.4g} above the dilute cutoff "
                    f"{dilute_cutoff:.4g}; infinite-dilution accuracy may be reduced."
                )

            if component_info.get('reason') != 'noncondensable':
                continue
            props = self.thermo.props.get(comp)
            Tc = getattr(props, 'Tc', None) if props is not None else None
            if Tc is None:
                continue
            temperature_margin = minimum_temperature - float(Tc)
            component_info['minimum_temperature_margin_to_Tc_K'] = float(
                temperature_margin
            )
            if temperature_margin < 0.0:
                warnings.append(
                    f"{type(self).__name__} '{self.unit_id}' selected {comp} as "
                    f"noncondensable at the pre-solve aqueous temperature, but the "
                    f"solved profile falls {abs(temperature_margin):.3g} K below its "
                    "critical temperature; the Henry path remained frozen."
                )
        return warnings
