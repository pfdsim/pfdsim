"""Solid-forming unit operations."""

from __future__ import annotations

import math
from typing import ClassVar

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .crystallizer_specs import (
        CrystallizerSpecificationError,
        finite_crystallizer_number,
        validate_crystallizer_specification,
    )
    from .msmpr_models import (
        MSMPRConvergenceError,
        MSMPRDefinitionError,
        msmpr_rate_definition_from_parameters,
        msmpr_rate_law_from_mapping,
        solve_steady_msmpr,
    )
    from .particle_size_distributions import propagate_particle_size_distributions
    from .thermodynamics_models.activity import ActivityCoefficientThermodynamics
    from .thermodynamics_models.common import ThermodynamicsError
    from .thermodynamics_models.sle import solve_pure_solid_sle
    from .unit_operations_base import UnitOperation, UnitOperationError, UnitResult
else:
    from crystallizer_specs import (
        CrystallizerSpecificationError,
        finite_crystallizer_number,
        validate_crystallizer_specification,
    )
    from msmpr_models import (
        MSMPRConvergenceError,
        MSMPRDefinitionError,
        msmpr_rate_definition_from_parameters,
        msmpr_rate_law_from_mapping,
        solve_steady_msmpr,
    )
    from particle_size_distributions import propagate_particle_size_distributions
    from thermodynamics_models.activity import ActivityCoefficientThermodynamics
    from thermodynamics_models.common import ThermodynamicsError
    from thermodynamics_models.sle import solve_pure_solid_sle
    from unit_operations_base import UnitOperation, UnitOperationError, UnitResult


class Crystallizer(UnitOperation):
    """Equilibrium or steady kinetic MSMPR cooling crystallizer."""

    supports_permanent_solids = True
    particle_size_behavior = 'custom'

    def _number(self, name, default=None, **constraints):
        try:
            return finite_crystallizer_number(
                self.get_param(name, default), name, **constraints
            )
        except CrystallizerSpecificationError as error:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' {error}"
            ) from error

    def _lle_diagnostic(self, slurry):
        """Probe the mother liquor without changing the homogeneous SLE result."""
        mode = getattr(self.thermo, 'fluid_phase_model', 'VLE')
        if mode not in {'VL(L)E', 'VLLE'} or not isinstance(
            self.thermo, ActivityCoefficientThermodynamics
        ):
            return None, []
        liquid_flows = slurry.phase_component_flows()['liquid1']
        if sum(liquid_flows.values()) <= 1e-15:
            return None, []
        composition = self._composition_from_flows(liquid_flows, {})
        try:
            equilibrium = self.thermo._fluid_phase_equilibrium_TP(
                composition, slurry.T, slurry.P
            )
        except ThermodynamicsError as error:
            return {'checked': False, 'fluid_phase_model': mode, 'error': str(error)}, [
                (
                    f"Crystallizer '{self.unit_id}' outlet LLE check failed: {error}. "
                    'The crystallizer assumes one homogeneous liquid mother phase.'
                )
            ]
        detected = (
            equilibrium.liquid1_fraction > 1e-10
            and equilibrium.liquid2_fraction > 1e-10
        )
        diagnostic = {
            'checked': True,
            'fluid_phase_model': mode,
            'lle_detected': detected,
            'stability': equilibrium.stability,
            'liquid1_fraction': equilibrium.liquid1_fraction,
            'liquid2_fraction': equilibrium.liquid2_fraction,
        }
        warnings = []
        if detected:
            warnings.append(
                f"Crystallizer '{self.unit_id}' outlet mother liquor exhibits LLE. "
                'The crystallizer currently does not support crystallization with LLE present; '
                'results assume one homogeneous liquid mother phase.'
            )
        return diagnostic, warnings

    def _finalize(
        self,
        inlet,
        slurry,
        candidates,
        specified_sphericity,
        retention_spec,
        performance,
        phase_details,
        *,
        generated_psds=None,
    ):
        """Finalize either solver's slurry, particles, outlets, duty and diagnostics."""
        candidate_set = set(candidates)
        for component, properties in inlet.solid_particle_properties.items():
            if (
                component not in candidate_set
                and slurry.solid_component_flows.get(component, 0) > 1e-15
            ):
                slurry.solid_particle_properties[component] = dict(properties)
        self._apply_outlet_sphericity(inlet, slurry, candidates, specified_sphericity)
        if 'particle_population_balance' in phase_details:
            component = performance['crystallizing_component']
            sphericity = slurry.solid_particle_properties.get(component, {}).get(
                'sphericity'
            )
            performance['particle_sphericity'] = sphericity
            phase_details['particle_population_balance']['particle_sphericity'] = (
                sphericity
            )
        if generated_psds is None:
            propagate_particle_size_distributions((inlet,), slurry)
        else:
            slurry.solid_particle_size_distributions.update(generated_psds)
        slurry.validate_particle_size_distributions()
        diagnostic, warnings = self._lle_diagnostic(slurry)
        separation = None
        if retention_spec is None:
            outlets = {'out': slurry}
        else:
            cake, liquor, separation = self._split_slurry(slurry, retention_spec)
            outlets = {'cake': cake, 'mother_liquor': liquor}
            for stream in outlets.values():
                propagate_particle_size_distributions((slurry,), stream)
                stream.solid_particle_properties = {
                    c: dict(p)
                    for c, p in slurry.solid_particle_properties.items()
                    if stream.solid_component_flows.get(c, 0) > 1e-15
                }
                stream.validate_particle_size_distributions()
        for stream in outlets.values():
            for name, details in phase_details.items():
                stream.phase_details[name] = dict(details)
            if diagnostic is not None:
                stream.phase_details['crystallizer_lle_check'] = dict(diagnostic)
        if any(
            s.H is None or not math.isfinite(s.H) for s in (inlet, *outlets.values())
        ):
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' requires finite inlet and outlet enthalpy"
            )
        duty = sum(s.F * s.H for s in outlets.values()) - inlet.F * inlet.H
        performance = {
            **performance,
            'T_out_C': slurry.T - 273.15,
            'P_out_bar': slurry.P,
            'outlet_mode': 'slurry' if separation is None else 'cake_split',
            **{
                key: None if separation is None else separation[key]
                for key in (
                    'mother_liquor_retention_basis',
                    'mother_liquor_retention_fraction',
                    'mother_liquor_retention_rate_kg_per_kg_crystals',
                    'retained_mother_liquor_mass_kg_per_h',
                )
            },
            'duty_kW': duty / 3600.0,
        }
        if diagnostic is not None:
            performance['outlet_lle_check'] = diagnostic
        return UnitResult(
            outlet_streams=outlets,
            heat_duty=duty,
            performance=performance,
            warnings=warnings,
        )

    _TIME_FACTORS_H: ClassVar[dict] = {
        'h': 1.0,
        'hr': 1.0,
        'hour': 1.0,
        'hours': 1.0,
        'min': 1.0 / 60.0,
        'minute': 1.0 / 60.0,
        'minutes': 1.0 / 60.0,
        's': 1.0 / 3600.0,
        'sec': 1.0 / 3600.0,
        'second': 1.0 / 3600.0,
        'seconds': 1.0 / 3600.0,
    }
    _VOLUME_FACTORS_M3: ClassVar[dict] = {
        'm3': 1.0,
        'm^3': 1.0,
        'l': 1.0e-3,
        'liter': 1.0e-3,
        'litre': 1.0e-3,
    }
    _DIAMETER_FACTORS_M: ClassVar[dict] = {
        'm': 1.0,
        'mm': 1.0e-3,
        'um': 1.0e-6,
        'µm': 1.0e-6,
    }

    @staticmethod
    def _normalized_unit(unit):
        return str(unit or '').strip().lower().replace(' ', '')

    def _one_dimension(
        self,
        names,
        factors,
        label,
        default_unit,
        *,
        allow_zero=False,
    ):
        found = [
            (name, self.get_param(name), self.get_param_unit(name))
            for name in names
            if self.get_param(name) is not None
        ]
        if len(found) > 1:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' received duplicate aliases for "
                f'{label}: ' + ', '.join(name for name, _value, _unit in found)
            )
        if not found:
            return None
        name, raw_value, raw_unit = found[0]
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' {name} must be numeric"
            ) from error
        unit = self._normalized_unit(raw_unit) or default_unit
        factor = factors.get(unit)
        if factor is None:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' does not recognize {label} "
                f'unit {raw_unit!r}'
            )
        value *= factor
        if not math.isfinite(value) or (value < 0.0 if allow_zero else value <= 0.0):
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' {name} must be "
                f'{"nonnegative" if allow_zero else "positive"} and finite'
            )
        return value

    def _specified_outlet_sphericity(self):
        raw_value = self.get_param('outlet_sphericity')
        if raw_value is None:
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' outlet_sphericity must be numeric"
            ) from error
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' outlet_sphericity must be in (0, 1]"
            )
        return value

    def _apply_outlet_sphericity(
        self,
        inlet,
        outlet,
        components,
        specified_sphericity,
    ):
        for component in components:
            if outlet.solid_component_flows.get(component, 0.0) <= 1.0e-15:
                continue
            sphericity = specified_sphericity
            if (
                sphericity is None
                and inlet.solid_component_flows.get(component, 0.0) > 1.0e-15
            ):
                sphericity = inlet.solid_particle_properties.get(component, {}).get(
                    'sphericity'
                )
                if sphericity is not None:
                    try:
                        sphericity = float(sphericity)
                    except (TypeError, ValueError) as error:
                        raise UnitOperationError(
                            f"Crystallizer '{self.unit_id}' inlet sphericity for "
                            f'{component!r} must be numeric'
                        ) from error
                    if not math.isfinite(sphericity) or not 0.0 < sphericity <= 1.0:
                        raise UnitOperationError(
                            f"Crystallizer '{self.unit_id}' inlet sphericity for "
                            f'{component!r} must be in (0, 1]'
                        )
            if sphericity is None:
                continue
            properties = dict(outlet.solid_particle_properties.get(component, {}))
            properties['sphericity'] = sphericity
            outlet.solid_particle_properties[component] = properties

    def _msmpr_rate_definition(self, kind):
        try:
            return msmpr_rate_definition_from_parameters(self.params, kind)
        except MSMPRDefinitionError as error:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' {error}"
            ) from error

    def _msmpr_geometry(self):
        residence_time = self._one_dimension(
            ('residence_time', 'tau'),
            self._TIME_FACTORS_H,
            'residence time',
            'h',
        )
        volume = self._one_dimension(
            ('volume', 'V'),
            self._VOLUME_FACTORS_M3,
            'volume',
            'm3',
        )
        if (residence_time is None) == (volume is None):
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' MSMPR mode requires exactly one "
                'of residence_time/tau or volume/V'
            )
        return residence_time, volume

    def _mother_liquor_retention_spec(self):
        fraction_names = (
            'mother_liquor_retention',
            'mother_liquor_retention_fraction',
        )
        fraction_specs = [
            (name, self.get_param(name))
            for name in fraction_names
            if self.get_param(name) is not None
        ]
        rate = self.get_param('mother_liquor_retention_rate')
        if len(fraction_specs) > 1 or (fraction_specs and rate is not None):
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' must specify only one mother-"
                'liquor retention basis'
            )
        if fraction_specs:
            name, _value = fraction_specs[0]
            retention = self._number('mother_liquor_retention', minimum=-float('inf'))
            if not math.isfinite(retention) or not 0.0 <= retention <= 1.0:
                raise UnitOperationError(
                    f"Crystallizer '{self.unit_id}' {name} must be a "
                    'fraction between 0 and 1'
                )
            return {
                'basis': 'fraction_of_equilibrium_mother_liquor',
                'value': retention,
            }
        if rate is not None:
            retention_rate = self._number(
                'mother_liquor_retention_rate', inclusive=True
            )
            if not math.isfinite(retention_rate) or retention_rate < 0.0:
                raise UnitOperationError(
                    f"Crystallizer '{self.unit_id}' mother_liquor_retention_rate "
                    'must be a nonnegative mass ratio'
                )
            return {
                'basis': 'kg_liquor_per_kg_crystals',
                'value': retention_rate,
            }
        return None

    @staticmethod
    def _composition_from_flows(component_flows, fallback):
        total = sum(max(0.0, float(value)) for value in component_flows.values())
        if total <= 1.0e-15:
            return dict(fallback)
        return {
            component: max(0.0, float(flow)) / total
            for component, flow in component_flows.items()
            if float(flow) > 1.0e-15
        }

    def _component_mass_flow(self, component_flows):
        return sum(
            float(flow) * float(self.thermo.props[component].MW)
            for component, flow in component_flows.items()
            if float(flow) > 0.0
        )

    def _split_slurry(self, slurry, retention_spec):
        phase_flows = slurry.phase_component_flows()
        liquid_flows = {}
        for phase_name in ('liquid1', 'liquid2'):
            for component, flow in phase_flows[phase_name].items():
                liquid_flows[component] = liquid_flows.get(component, 0.0) + flow
        solid_flows = dict(phase_flows['solid'])
        conventional_solid_set = set(self.thermo.conventional_solid_components)
        crystal_flows = {
            component: flow
            for component, flow in solid_flows.items()
            if component in conventional_solid_set and flow > 1.0e-15
        }
        liquid_mass = self._component_mass_flow(liquid_flows)
        crystal_mass = self._component_mass_flow(crystal_flows)
        if retention_spec['basis'] == 'fraction_of_equilibrium_mother_liquor':
            retention = float(retention_spec['value'])
            retained_liquor_mass = retention * liquid_mass
        else:
            requested_rate = float(retention_spec['value'])
            if requested_rate > 0.0 and crystal_mass <= 1.0e-15:
                raise UnitOperationError(
                    f"Crystallizer '{self.unit_id}' cannot apply a positive "
                    'mother_liquor_retention_rate when no conventional '
                    'crystals form'
                )
            retained_liquor_mass = requested_rate * crystal_mass
            tolerance = max(1.0e-10, 1.0e-12 * liquid_mass)
            if retained_liquor_mass > liquid_mass + tolerance:
                available_rate = (
                    liquid_mass / crystal_mass if crystal_mass > 0.0 else 0.0
                )
                raise UnitOperationError(
                    f"Crystallizer '{self.unit_id}' mother_liquor_retention_rate "
                    f'requires {retained_liquor_mass:g} kg/h mother liquor but '
                    f'only {liquid_mass:g} kg/h is available (maximum rate '
                    f'{available_rate:g} kg/kg crystals)'
                )
            retention = retained_liquor_mass / liquid_mass if liquid_mass > 0.0 else 0.0
        cake_flows = {
            component: solid_flows.get(component, 0.0)
            + retention * liquid_flows.get(component, 0.0)
            for component in set(solid_flows) | set(liquid_flows)
        }
        mother_liquor_flows = {
            component: (1.0 - retention) * flow
            for component, flow in liquid_flows.items()
        }
        cake_total = sum(cake_flows.values())
        liquor_total = sum(mother_liquor_flows.values())
        liquid_fallback = slurry.x1 or slurry.x or slurry.composition
        cake_composition = self._composition_from_flows(cake_flows, liquid_fallback)
        liquor_composition = self._composition_from_flows(
            mother_liquor_flows, liquid_fallback
        )
        cake_conventional_solids = {
            component: flow
            for component, flow in solid_flows.items()
            if component in conventional_solid_set and flow > 1.0e-15
        }
        cake = self.thermo.calculate_state_with_solid_flows(
            slurry.T,
            slurry.P,
            cake_total,
            cake_composition,
            cake_conventional_solids,
            phase='liquid',
        )
        mother_liquor = self.thermo.calculate_state(
            slurry.T,
            slurry.P,
            liquor_total,
            liquor_composition,
            phase='liquid',
        )
        separation = {
            'model': 'ideal_crystal_cake_split',
            'mother_liquor_retention_basis': retention_spec['basis'],
            'mother_liquor_retention_specified_value': retention_spec['value'],
            'mother_liquor_retention_fraction': retention,
            'mother_liquor_retention_rate_kg_per_kg_crystals': (
                retained_liquor_mass / crystal_mass if crystal_mass > 0.0 else 0.0
            ),
            'dry_crystal_mass_kg_per_h': crystal_mass,
            'retained_mother_liquor_mass_kg_per_h': retained_liquor_mass,
        }
        cake.phase_details['crystallizer_separation'] = dict(separation)
        mother_liquor.phase_details['crystallizer_separation'] = dict(separation)
        return cake, mother_liquor, separation

    def _solve_msmpr(
        self,
        inlet,
        temperature,
        pressure,
        candidates,
        mother_liquor_retention,
        max_iterations,
        specified_outlet_sphericity,
    ):
        if len(candidates) != 1:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' MSMPR mode currently requires "
                'exactly one conventional_with_solid component in the feed'
            )
        component = candidates[0]
        requested_component = self.get_param(
            'crystallizing_component', self.get_param('component')
        )
        if (
            requested_component is not None
            and str(requested_component).strip() != component
        ):
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' MSMPR crystallizing_component "
                f'must be {component!r} for this feed'
            )
        active_solids = {
            name: float(flow)
            for name, flow in inlet.solid_component_flows.items()
            if float(flow) > 1.0e-15
        }
        unsupported_solids = sorted(set(active_solids) - {component})
        if unsupported_solids:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' MSMPR mode only accepts seed "
                f'solid {component!r}; unsupported solid component(s): '
                + ', '.join(unsupported_solids)
            )
        seed_flow = active_solids.get(component, 0.0)
        seed_distribution = inlet.solid_particle_size_distributions.get(component)
        residence_time, volume = self._msmpr_geometry()

        try:
            growth_law = msmpr_rate_law_from_mapping(
                self._msmpr_rate_definition('growth'), 'growth'
            )
            nucleation_law = msmpr_rate_law_from_mapping(
                self._msmpr_rate_definition('nucleation'), 'nucleation'
            )
        except MSMPRDefinitionError as error:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' MSMPR kinetics are invalid: {error}"
            ) from error

        nucleus_diameter = self._one_dimension(
            ('nucleus_diameter', 'nucleation_diameter', 'L0'),
            self._DIAMETER_FACTORS_M,
            'nucleus diameter',
            'm',
            allow_zero=True,
        )
        if nucleus_diameter is None:
            nucleus_diameter = 0.0
        classes = self._number('quadrature_classes', 20, integer=True)
        maximum_output_classes = self._number(
            'maximum_output_classes', 200, integer=True
        )
        residual_tolerance = self._number('msmpr_tolerance', 1e-8)
        relative_residual_tolerance = self._number(
            'msmpr_relative_tolerance', 0.0, inclusive=True
        )
        total_component_flows = inlet.component_flows()
        fluid_component_flows = {
            name: total_component_flows.get(name, 0.0)
            for name in self.thermo.components
        }
        try:
            kinetic = solve_steady_msmpr(
                self.thermo,
                component=component,
                total_component_flows_kmol_h=fluid_component_flows,
                T=temperature,
                P=pressure,
                residence_time_h=residence_time,
                volume_m3=volume,
                growth_law=growth_law,
                nucleation_law=nucleation_law,
                seed_solid_flow_kmol_h=seed_flow,
                seed_distribution=seed_distribution,
                nucleus_diameter_m=nucleus_diameter,
                quadrature_classes=classes,
                maximum_output_classes=maximum_output_classes,
                residual_tolerance=residual_tolerance,
                relative_residual_tolerance=relative_residual_tolerance,
                max_iterations=max_iterations,
            )
            solid_flows = (
                {component: kinetic.solid_flow_kmol_h}
                if kinetic.solid_flow_kmol_h > 1.0e-15
                else {}
            )
            outlet = self.thermo.calculate_state_with_solid_flows(
                temperature,
                pressure,
                inlet.F,
                inlet.composition,
                solid_flows,
                phase='liquid',
            )
        except (
            MSMPRConvergenceError,
            MSMPRDefinitionError,
            ThermodynamicsError,
        ) as error:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' MSMPR calculation failed: {error}"
            ) from error

        population_performance = {
            'model': 'steady_ideal_msmpr',
            'saturation_ratio': kinetic.saturation_ratio,
            'relative_supersaturation': kinetic.relative_supersaturation,
            'log_saturation_ratio': kinetic.log_saturation_ratio,
            'saturation_temperature_K': kinetic.saturation_temperature_K,
            'undercooling_K': kinetic.undercooling_K,
            'reduced_undercooling': kinetic.reduced_undercooling,
            'fusion_scaled_undercooling': kinetic.fusion_scaled_undercooling,
            'melting_temperature_K': kinetic.melting_temperature_K,
            'heat_of_fusion_J_mol': kinetic.heat_of_fusion_J_mol,
            'residence_time_h': kinetic.residence_time_h,
            'volume_m3': kinetic.volume_m3,
            'volumetric_flow_m3_h': kinetic.volumetric_flow_m3_h,
            'nucleation_rate_per_m3_h': kinetic.nucleation_rate_per_m3_h,
            'birth_growth_rate_m_h': kinetic.birth_growth_rate_m_h,
            'suspension_density_kg_m3': kinetic.suspension_density_kg_m3,
            'quadrature_classes': classes,
            'maximum_output_classes': maximum_output_classes,
        }
        population_details = {
            **population_performance,
            'component': component,
            'size_coordinate': 'volume_equivalent_diameter',
            'growth_rate_range_m_h': kinetic.growth_rate_range_m_h,
            'quadrature': 'gauss_laguerre_residence_age',
            'assumptions': [
                'steady_state',
                'perfect_mixing',
                'mixed_product_removal',
                'deterministic_growth',
                'no_agglomeration',
                'no_breakage',
                'no_classification',
            ],
        }
        crystallized_flow = max(0.0, kinetic.solid_flow_kmol_h - seed_flow)
        performance = {
            **population_performance,
            'crystallizing_component': component,
            'solid_component_flows_kmol_per_h': {
                component: kinetic.solid_flow_kmol_h,
            },
            'crystallized_component_flows_kmol_per_h': {
                component: crystallized_flow,
            },
            'crystal_yields': {
                component: kinetic.solid_flow_kmol_h
                / max(total_component_flows.get(component, 0.0), 1.0e-300),
            },
            'seed_solid_flow_kmol_per_h': seed_flow,
            'mother_liquor_composition': dict(kinetic.liquid_composition),
            'liquid_activity': kinetic.liquid_activity,
            'saturation_activity': kinetic.saturation_activity,
            'solute_concentration_kmol_m3': kinetic.solute_concentration_kmol_m3,
            'growth_rate_min_m_h': kinetic.growth_rate_range_m_h[0],
            'growth_rate_max_m_h': kinetic.growth_rate_range_m_h[1],
            'nucleated_particle_rate_per_h': kinetic.nucleated_particle_rate_per_h,
            'seed_particle_rate_per_h': kinetic.seed_particle_rate_per_h,
            'total_particle_rate_per_h': kinetic.total_particle_rate_per_h,
            'number_mean_diameter_m': kinetic.number_mean_diameter_m,
            'material_residual_kmol_per_h': kinetic.material_residual_kmol_h,
            'absolute_residual_tolerance_kmol_per_h': kinetic.absolute_residual_tolerance_kmol_h,
            'relative_residual_tolerance': kinetic.relative_residual_tolerance,
            'effective_residual_tolerance_kmol_per_h': kinetic.effective_residual_tolerance_kmol_h,
            'population_balance_evaluations': kinetic.iterations,
            'growth_model': growth_law.model,
            'growth_expression': growth_law.expression.text,
            'growth_rate_unit': growth_law.declared_rate_unit,
            'nucleation_model': nucleation_law.model,
            'nucleation_expression': nucleation_law.expression.text,
            'nucleation_rate_unit': nucleation_law.declared_rate_unit,
        }
        return self._finalize(
            inlet,
            outlet,
            candidates,
            specified_outlet_sphericity,
            mother_liquor_retention,
            performance,
            {'particle_population_balance': population_details},
            generated_psds=(
                {component: kinetic.particle_size_distribution}
                if kinetic.particle_size_distribution is not None
                else {}
            ),
        )

    def solve(self, inlets) -> UnitResult:
        if len(inlets) != 1:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' requires exactly one inlet stream"
            )
        inlet = next(iter(inlets.values()))
        if not math.isfinite(inlet.F) or inlet.F <= 0.0:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' requires positive inlet flow"
            )

        try:
            self.params = validate_crystallizer_specification(
                self.params, self.get_param('__connected_outlet_ports__')
            )
        except CrystallizerSpecificationError as error:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' {error}"
            ) from error
        self._number('t_out', minimum=-float('inf'))
        temperature = self.get_temperature_param('t_out')
        if temperature <= 0:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' requires positive outlet temperature"
            )
        pressure = (
            self._number('p_out')
            if self.get_param('p_out') is not None
            else inlet.P - self._number('p_drop', 0.0, inclusive=True)
        )
        if not math.isfinite(pressure) or pressure <= 0:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' outlet pressure must be positive"
            )
        model = self.get_param('model')
        specified_outlet_sphericity = self._specified_outlet_sphericity()
        candidates = [
            component
            for component in self.thermo.conventional_solid_components
            if inlet.composition.get(component, 0.0) > 1.0e-15
        ]
        if not candidates:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' feed contains no component "
                'declared phase_behavior=conventional_with_solid'
            )

        total_component_flows = inlet.component_flows()
        fluid_component_flows = {
            component: total_component_flows.get(component, 0.0)
            for component in self.thermo.components
        }
        initial_solids = {
            component: inlet.solid_component_flows.get(component, 0.0)
            for component in candidates
        }
        tolerance = self._number('equilibrium_tolerance', 1e-8)
        max_iterations = self._number('max_iterations', 500, integer=True)
        mother_liquor_retention = self._mother_liquor_retention_spec()

        if model == 'msmpr':
            return self._solve_msmpr(
                inlet,
                temperature,
                pressure,
                candidates,
                mother_liquor_retention,
                max_iterations,
                specified_outlet_sphericity,
            )

        try:
            equilibrium = solve_pure_solid_sle(
                self.thermo,
                temperature,
                pressure,
                fluid_component_flows,
                candidates,
                initial_solid_flows=initial_solids,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            outlet = self.thermo.calculate_state_with_solid_flows(
                temperature,
                pressure,
                inlet.F,
                inlet.composition,
                equilibrium.solid_component_flows,
                phase='liquid',
            )
        except ThermodynamicsError as exc:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' SLE calculation failed: {exc}"
            ) from exc

        crystallized = {
            component: max(
                0.0,
                equilibrium.solid_component_flows.get(component, 0.0)
                - initial_solids.get(component, 0.0),
            )
            for component in candidates
        }
        dissolved = {
            component: max(
                0.0,
                initial_solids.get(component, 0.0)
                - equilibrium.solid_component_flows.get(component, 0.0),
            )
            for component in candidates
        }
        crystal_yields = {
            component: (
                equilibrium.solid_component_flows.get(component, 0.0)
                / max(fluid_component_flows.get(component, 0.0), 1.0e-300)
            )
            for component in candidates
        }
        sle_details = {
            'model': 'pure_solids_no_cocrystallization',
            'mother_liquor_composition': dict(equilibrium.liquid_composition),
            'saturation_activities': dict(equilibrium.saturation_activities),
            'liquid_activities': dict(equilibrium.liquid_activities),
            'saturation_residuals': dict(equilibrium.saturation_residuals),
            'iterations': equilibrium.iterations,
            'details': dict(equilibrium.details),
        }
        performance = {
            'model': 'equilibrium_pure_solids',
            'crystallizable_components': candidates,
            'solid_component_flows_kmol_per_h': dict(equilibrium.solid_component_flows),
            'crystallized_component_flows_kmol_per_h': crystallized,
            'dissolved_component_flows_kmol_per_h': dissolved,
            'crystal_yields': crystal_yields,
            'mother_liquor_composition': dict(equilibrium.liquid_composition),
            'saturation_activities': dict(equilibrium.saturation_activities),
            'liquid_activities': dict(equilibrium.liquid_activities),
            'equilibrium_residuals': dict(equilibrium.saturation_residuals),
            'iterations': equilibrium.iterations,
            'equilibrium_details': dict(equilibrium.details),
            'above_melting_candidates': list(
                equilibrium.details.get('above_melting_candidates', ())
            ),
        }
        return self._finalize(
            inlet,
            outlet,
            candidates,
            specified_outlet_sphericity,
            mother_liquor_retention,
            performance,
            {'solid_liquid_equilibrium': sle_details},
        )


__all__ = ['Crystallizer']
