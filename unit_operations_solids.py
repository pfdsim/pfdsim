"""Solid-forming unit operations."""

from __future__ import annotations

import math

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .msmpr_models import (
        MSMPRConvergenceError,
        MSMPRDefinitionError,
        msmpr_rate_definition_from_parameters,
        msmpr_rate_law_from_mapping,
        solve_steady_msmpr,
    )
    from .particle_size_distributions import propagate_particle_size_distributions
    from .thermodynamics_models.common import ThermodynamicsError
    from .thermodynamics_models.sle import solve_pure_solid_sle
    from .unit_operations_base import UnitOperation, UnitOperationError, UnitResult
else:
    from msmpr_models import (
        MSMPRConvergenceError,
        MSMPRDefinitionError,
        msmpr_rate_definition_from_parameters,
        msmpr_rate_law_from_mapping,
        solve_steady_msmpr,
    )
    from particle_size_distributions import propagate_particle_size_distributions
    from thermodynamics_models.common import ThermodynamicsError
    from thermodynamics_models.sle import solve_pure_solid_sle
    from unit_operations_base import UnitOperation, UnitOperationError, UnitResult


class Crystallizer(UnitOperation):
    """Equilibrium or steady kinetic MSMPR cooling crystallizer."""

    supports_permanent_solids = True
    particle_size_behavior = 'custom'

    _TIME_FACTORS_H = {
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
    _VOLUME_FACTORS_M3 = {
        'm3': 1.0,
        'm^3': 1.0,
        'l': 1.0e-3,
        'liter': 1.0e-3,
        'litre': 1.0e-3,
    }
    _DIAMETER_FACTORS_M = {
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
                f"{label}: " + ', '.join(name for name, _value, _unit in found)
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
                f"unit {raw_unit!r}"
            )
        value *= factor
        if not math.isfinite(value) or (
            value < 0.0 if allow_zero else value <= 0.0
        ):
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' {name} must be "
                f"{'nonnegative' if allow_zero else 'positive'} and finite"
            )
        return value

    def _crystallizer_model(self):
        raw_model = self.get_param(
            'model', self.get_param('crystallizer_model', 'equilibrium')
        )
        model = str(raw_model).strip().lower().replace('-', '_')
        aliases = {
            'equilibrium_sle': 'equilibrium',
            'sle': 'equilibrium',
            'kinetic': 'msmpr',
            'steady_msmpr': 'msmpr',
        }
        model = aliases.get(model, model)
        if model not in {'equilibrium', 'msmpr'}:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' model must be equilibrium or MSMPR"
            )
        msmpr_only = []
        for raw_name in self.params:
            name = str(raw_name).lower()
            if name.startswith('__unit__'):
                continue
            if (
                name in {
                    'residence_time', 'tau', 'volume', 'v',
                    'crystallizing_component', 'component',
                    'quadrature_classes', 'particle_classes',
                    'maximum_output_classes', 'max_output_classes',
                    'nucleus_diameter', 'nucleation_diameter', 'l0',
                    'msmpr_tolerance', 'kinetic_tolerance',
                    'msmpr_relative_tolerance', 'kinetic_relative_tolerance',
                }
                or name in {'growth', 'g', 'nucleation', 'b0'}
                or name.startswith(('growth_', 'g_', 'nucleation_', 'b0_'))
            ):
                msmpr_only.append(str(raw_name))
        if model == 'equilibrium' and msmpr_only:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' MSMPR parameter(s) require "
                "model=MSMPR: " + ', '.join(sorted(msmpr_only))
            )
        return model

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
                "of residence_time/tau or volume/V"
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
                "liquor retention basis"
            )
        if fraction_specs:
            name, value = fraction_specs[0]
            retention = float(value)
            if not math.isfinite(retention) or not 0.0 <= retention <= 1.0:
                raise UnitOperationError(
                    f"Crystallizer '{self.unit_id}' {name} must be a "
                    "fraction between 0 and 1"
                )
            return {
                'basis': 'fraction_of_equilibrium_mother_liquor',
                'value': retention,
            }
        if rate is not None:
            retention_rate = float(rate)
            if not math.isfinite(retention_rate) or retention_rate < 0.0:
                raise UnitOperationError(
                    f"Crystallizer '{self.unit_id}' mother_liquor_retention_rate "
                    "must be a nonnegative mass ratio"
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
                    "mother_liquor_retention_rate when no conventional "
                    "crystals form"
                )
            retained_liquor_mass = requested_rate * crystal_mass
            tolerance = max(1.0e-10, 1.0e-12 * liquid_mass)
            if retained_liquor_mass > liquid_mass + tolerance:
                available_rate = (
                    liquid_mass / crystal_mass
                    if crystal_mass > 0.0 else 0.0
                )
                raise UnitOperationError(
                    f"Crystallizer '{self.unit_id}' mother_liquor_retention_rate "
                    f"requires {retained_liquor_mass:g} kg/h mother liquor but "
                    f"only {liquid_mass:g} kg/h is available (maximum rate "
                    f"{available_rate:g} kg/kg crystals)"
                )
            retention = (
                retained_liquor_mass / liquid_mass
                if liquid_mass > 0.0 else 0.0
            )
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
        cake_composition = self._composition_from_flows(
            cake_flows, liquid_fallback
        )
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
                retained_liquor_mass / crystal_mass
                if crystal_mass > 0.0 else 0.0
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
    ):
        if len(candidates) != 1:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' MSMPR mode currently requires "
                "exactly one conventional_with_solid component in the feed"
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
                f"must be {component!r} for this feed"
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
                f"solid {component!r}; unsupported solid component(s): "
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
        raw_classes = self.get_param(
            'quadrature_classes', self.get_param('particle_classes', 20)
        )
        raw_output_classes = self.get_param(
            'maximum_output_classes', self.get_param('max_output_classes', 200)
        )
        try:
            classes = int(raw_classes)
            maximum_output_classes = int(raw_output_classes)
            residual_tolerance = float(self.get_param(
                'msmpr_tolerance', self.get_param('kinetic_tolerance', 1.0e-8)
            ))
            relative_residual_tolerance = float(self.get_param(
                'msmpr_relative_tolerance',
                self.get_param('kinetic_relative_tolerance', 0.0),
            ))
        except (TypeError, ValueError) as error:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' MSMPR numerical controls must "
                "be numeric"
            ) from error
        if (
            float(raw_classes) != classes
            or float(raw_output_classes) != maximum_output_classes
        ):
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' MSMPR class counts must be integers"
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
                if kinetic.solid_flow_kmol_h > 1.0e-15 else {}
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

        if kinetic.particle_size_distribution is not None:
            outlet.solid_particle_size_distributions[component] = (
                kinetic.particle_size_distribution
            )
        outlet.validate_particle_size_distributions()
        population_details = {
            'model': 'steady_ideal_msmpr',
            'component': component,
            'size_coordinate': 'volume_equivalent_diameter',
            'saturation_ratio': kinetic.saturation_ratio,
            'relative_supersaturation': kinetic.relative_supersaturation,
            'log_saturation_ratio': kinetic.log_saturation_ratio,
            'saturation_temperature_K': kinetic.saturation_temperature_K,
            'undercooling_K': kinetic.undercooling_K,
            'reduced_undercooling': kinetic.reduced_undercooling,
            'fusion_scaled_undercooling': (
                kinetic.fusion_scaled_undercooling
            ),
            'melting_temperature_K': kinetic.melting_temperature_K,
            'heat_of_fusion_J_mol': kinetic.heat_of_fusion_J_mol,
            'residence_time_h': kinetic.residence_time_h,
            'volume_m3': kinetic.volume_m3,
            'volumetric_flow_m3_h': kinetic.volumetric_flow_m3_h,
            'nucleation_rate_per_m3_h': kinetic.nucleation_rate_per_m3_h,
            'growth_rate_range_m_h': kinetic.growth_rate_range_m_h,
            'birth_growth_rate_m_h': kinetic.birth_growth_rate_m_h,
            'suspension_density_kg_m3': kinetic.suspension_density_kg_m3,
            'quadrature': 'gauss_laguerre_residence_age',
            'quadrature_classes': classes,
            'maximum_output_classes': maximum_output_classes,
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
        outlet.phase_details['particle_population_balance'] = dict(
            population_details
        )

        separation = None
        if mother_liquor_retention is None:
            outlet_streams = {'out': outlet}
        else:
            cake, mother_liquor, separation = self._split_slurry(
                outlet, mother_liquor_retention
            )
            cake.solid_particle_size_distributions = dict(
                outlet.solid_particle_size_distributions
            )
            cake.phase_details['particle_population_balance'] = dict(
                population_details
            )
            mother_liquor.phase_details['particle_population_balance'] = dict(
                population_details
            )
            outlet_streams = {
                'cake': cake,
                'mother_liquor': mother_liquor,
            }

        if inlet.H is None or any(
            stream.H is None for stream in outlet_streams.values()
        ):
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' requires inlet and outlet enthalpy"
            )
        outlet_enthalpy_flow = sum(
            stream.F * stream.H for stream in outlet_streams.values()
        )
        duty = outlet_enthalpy_flow - inlet.F * inlet.H
        crystallized_flow = max(0.0, kinetic.solid_flow_kmol_h - seed_flow)
        performance = {
            'model': 'steady_ideal_msmpr',
            'T_out_C': temperature - 273.15,
            'P_out_bar': pressure,
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
            'saturation_ratio': kinetic.saturation_ratio,
            'relative_supersaturation': kinetic.relative_supersaturation,
            'log_saturation_ratio': kinetic.log_saturation_ratio,
            'saturation_temperature_K': kinetic.saturation_temperature_K,
            'undercooling_K': kinetic.undercooling_K,
            'reduced_undercooling': kinetic.reduced_undercooling,
            'fusion_scaled_undercooling': (
                kinetic.fusion_scaled_undercooling
            ),
            'melting_temperature_K': kinetic.melting_temperature_K,
            'heat_of_fusion_J_mol': kinetic.heat_of_fusion_J_mol,
            'solute_concentration_kmol_m3': (
                kinetic.solute_concentration_kmol_m3
            ),
            'suspension_density_kg_m3': kinetic.suspension_density_kg_m3,
            'residence_time_h': kinetic.residence_time_h,
            'volume_m3': kinetic.volume_m3,
            'volumetric_flow_m3_h': kinetic.volumetric_flow_m3_h,
            'nucleation_rate_per_m3_h': kinetic.nucleation_rate_per_m3_h,
            'growth_rate_min_m_h': kinetic.growth_rate_range_m_h[0],
            'growth_rate_max_m_h': kinetic.growth_rate_range_m_h[1],
            'birth_growth_rate_m_h': kinetic.birth_growth_rate_m_h,
            'nucleated_particle_rate_per_h': (
                kinetic.nucleated_particle_rate_per_h
            ),
            'seed_particle_rate_per_h': kinetic.seed_particle_rate_per_h,
            'total_particle_rate_per_h': kinetic.total_particle_rate_per_h,
            'number_mean_diameter_m': kinetic.number_mean_diameter_m,
            'material_residual_kmol_per_h': kinetic.material_residual_kmol_h,
            'absolute_residual_tolerance_kmol_per_h': (
                kinetic.absolute_residual_tolerance_kmol_h
            ),
            'relative_residual_tolerance': (
                kinetic.relative_residual_tolerance
            ),
            'effective_residual_tolerance_kmol_per_h': (
                kinetic.effective_residual_tolerance_kmol_h
            ),
            'population_balance_evaluations': kinetic.iterations,
            'quadrature_classes': classes,
            'maximum_output_classes': maximum_output_classes,
            'growth_model': growth_law.model,
            'growth_expression': growth_law.expression.text,
            'growth_rate_unit': growth_law.declared_rate_unit,
            'nucleation_model': nucleation_law.model,
            'nucleation_expression': nucleation_law.expression.text,
            'nucleation_rate_unit': nucleation_law.declared_rate_unit,
            'outlet_mode': (
                'slurry' if mother_liquor_retention is None else 'cake_split'
            ),
            'mother_liquor_retention_basis': (
                None if separation is None
                else separation['mother_liquor_retention_basis']
            ),
            'mother_liquor_retention_fraction': (
                None if separation is None
                else separation['mother_liquor_retention_fraction']
            ),
            'mother_liquor_retention_rate_kg_per_kg_crystals': (
                None if separation is None
                else separation[
                    'mother_liquor_retention_rate_kg_per_kg_crystals'
                ]
            ),
            'retained_mother_liquor_mass_kg_per_h': (
                None if separation is None
                else separation['retained_mother_liquor_mass_kg_per_h']
            ),
            'duty_kW': duty / 3600.0,
        }
        return UnitResult(
            outlet_streams=outlet_streams,
            heat_duty=duty,
            performance=performance,
        )

    def solve(self, inlets) -> UnitResult:
        if len(inlets) != 1:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' requires exactly one inlet stream"
            )
        inlet = next(iter(inlets.values()))
        if inlet.F <= 0.0:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' requires positive inlet flow"
            )

        temperature = self.get_temperature_param('T_out')
        if temperature is None:
            temperature = self.get_temperature_param('T')
        if temperature is None:
            temperature = self.get_temperature_param('temperature')
        if (
            temperature is None
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0.0
        ):
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' requires a positive outlet "
                "temperature T_out/T"
            )
        temperature = float(temperature)

        absolute_pressure = self.get_param(
            'P_out', self.get_param('P', self.get_param('pressure'))
        )
        pressure_drop = self.get_param('P_drop')
        if absolute_pressure is not None and pressure_drop is not None:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' cannot specify both outlet "
                "pressure and P_drop"
            )
        if absolute_pressure is None:
            pressure = inlet.P - float(pressure_drop or 0.0)
        else:
            pressure = float(absolute_pressure)
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' outlet pressure must be positive"
            )

        model = self._crystallizer_model()
        candidates = [
            component
            for component in self.thermo.conventional_solid_components
            if inlet.composition.get(component, 0.0) > 1.0e-15
        ]
        if not candidates:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' feed contains no component "
                "declared phase_behavior=conventional_with_solid"
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
        tolerance = float(self.get_param('equilibrium_tolerance', 1.0e-8))
        max_iterations = int(self.get_param('max_iterations', 500))
        mother_liquor_retention = self._mother_liquor_retention_spec()
        connected_outlets = set(
            self.get_param('__connected_outlet_ports__', ()) or ()
        )
        split_ports = {'cake', 'mother_liquor'}
        if mother_liquor_retention is None and connected_outlets & split_ports:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' cake/mother_liquor outlets "
                "require mother_liquor_retention"
            )
        if mother_liquor_retention is not None and connected_outlets:
            if 'out' in connected_outlets:
                raise UnitOperationError(
                    f"Crystallizer '{self.unit_id}' uses cake and "
                    "mother_liquor outlets when mother_liquor_retention is specified"
                )
            missing_split_ports = sorted(split_ports - connected_outlets)
            if missing_split_ports:
                raise UnitOperationError(
                    f"Crystallizer '{self.unit_id}' cake-split mode is missing "
                    f"connected outlet(s): {', '.join(missing_split_ports)}"
                )
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' equilibrium_tolerance must be positive"
            )
        if max_iterations <= 0:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' max_iterations must be positive"
            )

        if model == 'msmpr':
            return self._solve_msmpr(
                inlet,
                temperature,
                pressure,
                candidates,
                mother_liquor_retention,
                max_iterations,
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

        if inlet.H is None or outlet.H is None:
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' requires inlet and outlet enthalpy"
            )
        if mother_liquor_retention is None:
            outlet_streams = {'out': outlet}
        else:
            cake, mother_liquor, separation = self._split_slurry(
                outlet, mother_liquor_retention
            )
            outlet_streams = {
                'cake': cake,
                'mother_liquor': mother_liquor,
            }
        outlet_enthalpy_flow = sum(
            stream.F * stream.H
            for stream in outlet_streams.values()
            if stream.H is not None
        )
        if any(stream.H is None for stream in outlet_streams.values()):
            raise UnitOperationError(
                f"Crystallizer '{self.unit_id}' requires outlet enthalpy"
            )
        duty = outlet_enthalpy_flow - inlet.F * inlet.H
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
        for stream in outlet_streams.values():
            stream.phase_details['solid_liquid_equilibrium'] = dict(sle_details)
            propagate_particle_size_distributions((inlet,), stream)

        return UnitResult(
            outlet_streams=outlet_streams,
            heat_duty=duty,
            performance={
                'model': 'equilibrium_pure_solids',
                'T_out_C': temperature - 273.15,
                'P_out_bar': pressure,
                'crystallizable_components': candidates,
                'solid_component_flows_kmol_per_h': dict(
                    equilibrium.solid_component_flows
                ),
                'crystallized_component_flows_kmol_per_h': crystallized,
                'dissolved_component_flows_kmol_per_h': dissolved,
                'crystal_yields': crystal_yields,
                'mother_liquor_composition': dict(
                    equilibrium.liquid_composition
                ),
                'saturation_activities': dict(
                    equilibrium.saturation_activities
                ),
                'liquid_activities': dict(equilibrium.liquid_activities),
                'equilibrium_residuals': dict(
                    equilibrium.saturation_residuals
                ),
                'iterations': equilibrium.iterations,
                'equilibrium_details': dict(equilibrium.details),
                'above_melting_candidates': list(
                    equilibrium.details.get('above_melting_candidates', ())
                ),
                'outlet_mode': (
                    'slurry' if mother_liquor_retention is None else 'cake_split'
                ),
                'mother_liquor_retention_basis': (
                    None if mother_liquor_retention is None
                    else separation['mother_liquor_retention_basis']
                ),
                'mother_liquor_retention_fraction': (
                    None if mother_liquor_retention is None
                    else separation['mother_liquor_retention_fraction']
                ),
                'mother_liquor_retention_rate_kg_per_kg_crystals': (
                    None if mother_liquor_retention is None
                    else separation[
                        'mother_liquor_retention_rate_kg_per_kg_crystals'
                    ]
                ),
                'retained_mother_liquor_mass_kg_per_h': (
                    None if mother_liquor_retention is None
                    else separation['retained_mother_liquor_mass_kg_per_h']
                ),
                'duty_kW': duty / 3600.0,
            },
        )


__all__ = ['Crystallizer']
