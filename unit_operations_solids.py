"""Solid-forming unit operations."""

from __future__ import annotations

import math
from typing import ClassVar

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .crystallizer_specs import (
        CrystallizerSpecificationError,
        finite_crystallizer_number,
        validate_layer_crystallizer_specification,
        validate_crystallizer_specification,
    )
    from .empirical_layer_crystallization import (
        EmpiricalLayerDefinitionError,
        empirical_layer_distribution_definitions_from_parameters,
        empirical_layer_distribution_law_from_mapping,
        empirical_layer_growth_definition_from_parameters,
        empirical_layer_growth_law_from_mapping,
        solve_empirical_layer_growth,
    )
    from .msmpr_models import (
        MSMPRConvergenceError,
        MSMPRDefinitionError,
        msmpr_rate_definition_from_parameters,
        msmpr_rate_law_from_mapping,
        solve_steady_msmpr,
    )
    from .particle_size_distributions import propagate_particle_size_distributions
    from .layer_crystallization import solve_layer_growth
    from .layer_sweating import (
        LayerSweatingError, SolidSolutionSolute, allocate_layer_inventory,
        layer_enthalpy, solve_layer_sweating,
    )
    from .thermodynamics_models.activity import ActivityCoefficientThermodynamics
    from .thermodynamics_models.common import ThermodynamicsError
    from .thermodynamics_models.sle import (
        liquid_solution_activities,
        pure_solid_log_saturation_activity,
        solve_pure_solid_sle,
    )
    from .unit_operations_base import UnitOperation, UnitOperationError, UnitResult
else:
    from crystallizer_specs import (
        CrystallizerSpecificationError,
        finite_crystallizer_number,
        validate_layer_crystallizer_specification,
        validate_crystallizer_specification,
    )
    from empirical_layer_crystallization import (
        EmpiricalLayerDefinitionError,
        empirical_layer_distribution_definitions_from_parameters,
        empirical_layer_distribution_law_from_mapping,
        empirical_layer_growth_definition_from_parameters,
        empirical_layer_growth_law_from_mapping,
        solve_empirical_layer_growth,
    )
    from msmpr_models import (
        MSMPRConvergenceError,
        MSMPRDefinitionError,
        msmpr_rate_definition_from_parameters,
        msmpr_rate_law_from_mapping,
        solve_steady_msmpr,
    )
    from particle_size_distributions import propagate_particle_size_distributions
    from layer_crystallization import solve_layer_growth
    from layer_sweating import (
        LayerSweatingError, SolidSolutionSolute, allocate_layer_inventory,
        layer_enthalpy, solve_layer_sweating,
    )
    from thermodynamics_models.activity import ActivityCoefficientThermodynamics
    from thermodynamics_models.common import ThermodynamicsError
    from thermodynamics_models.sle import (
        liquid_solution_activities,
        pure_solid_log_saturation_activity,
        solve_pure_solid_sle,
    )
    from unit_operations_base import UnitOperation, UnitOperationError, UnitResult


class _CrystallizerBase(UnitOperation):
    """Shared execution for suspension and layer crystallizer units."""

    supports_permanent_solids = True
    particle_size_behavior = 'custom'
    crystallization_mode: ClassVar[str]
    specification_validator: ClassVar = None
    unit_type_name: ClassVar[str]

    def _layer_sweating_enabled(self):
        return self.crystallization_mode == 'layer' and self.get_param(
            'sweat_heater_temperature'
        ) is not None

    def _sweat_and_harvest_layer(self, cake, component, retention_spec):
        temperature = cake.T
        heater_temperature = self.get_temperature_param('sweat_heater_temperature')
        harvest_temperature = self.get_temperature_param('harvest_temperature')
        collection_temperature = self.get_temperature_param(
            'sweat_collection_temperature', heater_temperature
        )
        melting = float(self.thermo.props[component].Tm)
        if not all(math.isfinite(t) and t > 0 for t in (
            temperature, heater_temperature, harvest_temperature, collection_temperature
        )):
            raise UnitOperationError('Sweating temperatures must be positive and finite')
        if heater_temperature < cake.T or max(temperature, heater_temperature) >= melting:
            raise UnitOperationError(
                'sweat_heater_temperature must be at least the growth temperature and below host Tm'
            )
        if min(harvest_temperature, collection_temperature) < temperature:
            raise UnitOperationError('Harvest and sweat collection temperatures must be at least the initial layer temperature')
        cycle = self._one_dimension(('cycle_time',), self._TIME_FACTORS_H, 'cycle time', 'h')
        duration = 3600 * self._one_dimension(('sweat_time',), self._TIME_FACTORS_H, 'sweat time', 'h')
        growth_time = self._one_dimension(('growth_time',), self._TIME_FACTORS_H, 'growth time', 'h')
        if growth_time + duration / 3600 > cycle + 1e-12:
            raise UnitOperationError('cycle_time must include growth_time plus sweat_time')
        phases = cake.phase_component_flows()
        if sum(phases['vapor'].values()) > 1e-12:
            raise UnitOperationError('Layer sweating requires a vapor-free layer')
        if any(c != component and n > 0 for c, n in phases['solid'].items()):
            raise UnitOperationError('Layer sweating does not model separate impurity crystals')

        def fields(prefix):
            result = {}
            lookup = {c.casefold(): c for c in self.thermo.components}
            for key in self.params:
                if not key.startswith(prefix):
                    continue
                name = lookup.get(key[len(prefix):].casefold())
                if name is None or name == component:
                    raise UnitOperationError(f'Invalid impurity parameter {key!r}')
                result[name] = key
            return result

        occluded_keys = fields('occluded_fraction_')
        partition_keys = fields('solid_partition_')
        enthalpy_keys = fields('solid_transfer_enthalpy_')
        diffusivity_keys = fields('solid_diffusivity_')
        if set(partition_keys) != set(enthalpy_keys) or set(partition_keys) != set(diffusivity_keys):
            raise UnitOperationError(
                'Each solid-solution impurity needs solid_partition_COMPONENT, '
                'solid_transfer_enthalpy_COMPONENT, and solid_diffusivity_COMPONENT'
            )
        reference = self.get_temperature_param('partition_reference_temperature', 298.15)
        solutes = {}
        for name, key in partition_keys.items():
            solutes[name] = SolidSolutionSolute(
                self._number(key),
                self._one_dimension((enthalpy_keys[name],), {'kj/mol': 1, 'j/mol': .001,
                                    'kj/kmol': .001}, 'solid transfer enthalpy', 'kj/mol',
                                    allow_negative=True),
                self._one_dimension((diffusivity_keys[name],), {'m2/s': 1, 'm^2/s': 1},
                                    'solid diffusivity', 'm2/s', allow_zero=True),
                reference,
            )
        captured = retention_spec.get('trapped_component_flows', {})
        try:
            initial_solid, initial_liquid = allocate_layer_inventory(
                host=component, totals=cake.component_flows(),
                host_solid=phases['solid'].get(component, 0.0), captured=captured,
                occluded_fractions={c: self._number(k, inclusive=True, maximum=1)
                                    for c, k in occluded_keys.items()},
                occluded_host_fraction=self.get_param('occluded_liquid_host_fraction'),
                empirical=self.get_param('model') == 'empirical_layer_growth',
            )
            missing = {c for c, n in initial_solid.items() if c != component and n > 0} - set(solutes)
            if missing:
                raise LayerSweatingError('Missing solid-solution parameters for ' + ', '.join(sorted(missing)))
            initial_solid = {c: n * cycle for c, n in initial_solid.items()}
            initial_liquid = {c: n * cycle for c, n in initial_liquid.items()}
            result = solve_layer_sweating(
                self.thermo, host=component, initial_solid=initial_solid,
                initial_liquid=initial_liquid, solutes=solutes,
                temperature=temperature, pressure=cake.P, duration_s=duration,
                heater_temperature=heater_temperature,
                thermal_conductance_W_K=self._one_dimension(
                    ('sweat_thermal_conductance',), {'w/k': 1, 'kw/k': 1000},
                    'sweating thermal conductance', 'w/k', allow_zero=True),
                opening_coefficient=self._number('sweat_opening_coefficient', inclusive=True),
                host_rate_constant=self._one_dimension(
                    ('sweat_host_rate_constant',), {'1/s': 1, 's^-1': 1, '1/h': 1/3600},
                    'host phase exchange rate', '1/s'),
                diffusion_length=self._one_dimension(
                    ('solid_diffusion_length',), self._DIAMETER_FACTORS_M,
                    'solid diffusion length', 'm') if solutes else 1.0,
                drainage_length=self._one_dimension(
                    ('sweat_drainage_length',), self._DIAMETER_FACTORS_M, 'drainage length', 'm'),
                pore_radius=self._one_dimension(
                    ('sweat_pore_radius',), self._DIAMETER_FACTORS_M, 'pore radius', 'm'),
                tortuosity=self._number('sweat_tortuosity', minimum=1, inclusive=True),
                connected_fraction=self._number('sweat_connected_fraction', inclusive=True, maximum=1),
                residual_saturation=self._number('sweat_residual_saturation', inclusive=True, maximum=1),
                capillary_pressure=self._one_dimension(
                    ('sweat_capillary_pressure',), {'pa': 1, 'kpa': 1000, 'bar': 1e5},
                    'capillary entry pressure', 'pa', allow_zero=True),
                solid_molar_volume=self.thermo._solid_molar_volume(component, temperature),
                relative_tolerance=self._number('sweat_relative_tolerance', 1e-7),
            )
        except (LayerSweatingError, ThermodynamicsError, ValueError) as error:
            raise UnitOperationError(f"LayerCrystallizer '{self.unit_id}' sweating failed: {error}") from error
        product_flows = {c: (result.solid_amounts.get(c, 0) + result.liquid_amounts.get(c, 0)) / cycle
                         for c in cake.composition}
        sweat_flows = {c: n / cycle for c, n in result.sweat_amounts.items()}
        product_total, sweat_total = sum(product_flows.values()), sum(sweat_flows.values())
        if product_total <= 0:
            raise UnitOperationError('Sweating left no harvest product')
        product = self.thermo.calculate_state(
            harvest_temperature, cake.P, product_total,
            self._composition_from_flows(product_flows, cake.composition), phase='liquid', flash=False)
        sweat = self.thermo.calculate_state(
            collection_temperature, cake.P, sweat_total,
            self._composition_from_flows(sweat_flows, cake.composition), phase='liquid', flash=False)

        def check_fluid_phase(label, T, flows):
            total = sum(flows.values())
            if total <= 1e-15:
                return
            composition = self._composition_from_flows(flows, cake.composition)
            fluid = self.thermo.calculate_state(T, cake.P, total, composition, flash=True)
            if fluid.vapor_fraction > 1e-8:
                raise UnitOperationError(f'{label} would vaporize; liquid-only sweating is unsupported')
            if fluid.liquid2_fraction > 1e-8:
                raise UnitOperationError(f'{label} requires a homogeneous liquid phase')
            diagnostic, _ = self._lle_diagnostic(fluid)
            if diagnostic and (not diagnostic.get('checked') or diagnostic.get('lle_detected')):
                raise UnitOperationError(f'{label} requires a homogeneous liquid phase')

        # Check the evolving pore compositions as well as the collected outlets;
        # warming a collection vessel cannot repair an invalid liquid-only hold.
        for point in result.profile:
            for region in ('connected', 'sealed'):
                check_fluid_phase(f'Sweating {region} liquid', point['temperature_K'],
                                  point[f'{region}_liquid_amounts_kmol'])
        # Ideal-solid tangent-plane minimum: -ln(sum_i K_i a_i^L).
        # This checks solid-solution precipitation, including the pure-host limit.
        stability = {}
        for label, stream in (('harvest', product), ('sweat', sweat)):
            if stream.F <= 1e-15:
                continue
            activities = liquid_solution_activities(self.thermo, stream.T, stream.P, stream.composition)
            host_log_k = -pure_solid_log_saturation_activity(
                self.thermo, component, stream.T, stream.P,
                solid_volume_temperature=min(stream.T, melting))
            saturation_sum = math.exp(min(host_log_k, 700)) * activities.get(component, 0)
            saturation_sum += sum(math.exp(min(law.log_partition(stream.T), 700)) * activities.get(c, 0)
                                  for c, law in solutes.items())
            stability[label] = saturation_sum
            if saturation_sum > 1 + 1e-7:
                raise UnitOperationError(
                    f'{label} liquid is unstable to solid-solution precipitation; '
                    f'increase {"harvest_temperature" if label == "harvest" else "sweat_collection_temperature"}'
                )
            check_fluid_phase(label, stream.T, stream.component_flows())
        initial_energy = layer_enthalpy(self.thermo, cake.T, cake.P,
                                       initial_solid, initial_liquid, component, solutes)
        remaining_energy = result.remaining_enthalpy_kJ
        if not all(math.isfinite(h) for h in (initial_energy, remaining_energy)):
            raise UnitOperationError('Sweating requires finite solid-solution and liquid enthalpies')
        initial_impurity = sum(n * self.thermo.props[c].MW for c, n in cake.component_flows().items()
                               if c != component)
        sweat_impurity = sum(n * self.thermo.props[c].MW for c, n in sweat_flows.items() if c != component)
        details = {
            'model': 'finite_rate_solid_solution_and_capillary_darcy_drainage',
            'component': component, 'initial_layer_temperature_K': temperature,
            'sweat_heater_temperature_K': heater_temperature,
            'final_layer_temperature_K': result.temperature,
            'sweat_collection_temperature_K': collection_temperature,
            'harvest_temperature_K': harvest_temperature, 'sweat_time_s': duration,
            'initial_solid_amounts_kmol': initial_solid,
            'initial_occluded_and_retained_liquid_amounts_kmol': initial_liquid,
            'remaining_solid_amounts_kmol': result.solid_amounts,
            'remaining_liquid_amounts_kmol': result.liquid_amounts,
            'remaining_connected_liquid_amounts_kmol': result.connected_liquid_amounts,
            'remaining_sealed_liquid_amounts_kmol': result.sealed_liquid_amounts,
            'supplied_heat_kJ_per_batch': result.supplied_heat_kJ,
            'drained_enthalpy_kJ_per_batch': result.drained_enthalpy_kJ,
            'sweating_duty_kW': result.supplied_heat_kJ / cycle / 3600,
            'collection_conditioning_duty_kW': (sweat.F * sweat.H - result.drained_enthalpy_kJ / cycle) / 3600,
            'energy_balance_residual': result.energy_balance_residual,
            'net_host_melted_kmol': initial_solid[component] - result.solid_amounts[component],
            'sweat_component_flows_kmol_per_h': sweat_flows,
            'harvest_component_flows_kmol_per_h': product_flows,
            'sweat_composition': dict(sweat.composition),
            'harvest_product_composition': dict(product.composition),
            'sweat_mass_kg_per_h': self._component_mass_flow(sweat_flows),
            'harvest_product_mass_kg_per_h': self._component_mass_flow(product_flows),
            'impurity_rejection_to_sweat': sweat_impurity / initial_impurity if initial_impurity > 0 else None,
            'sweating_and_collection_duty_kW': (remaining_energy / cycle + sweat.F * sweat.H
                                              - initial_energy / cycle) / 3600,
            'harvest_duty_kW': (product.F * product.H - remaining_energy / cycle) / 3600,
            'growth_layer_enthalpy_correction_kW': (initial_energy / cycle - cake.F * cake.H) / 3600,
            'liquid_solid_saturation_sums': stability,
            'component_balance_residual': result.component_balance_residual,
            'evaluations': result.evaluations, 'profile': result.profile,
            'assumptions': [
                'common_transient_temperature_with_finite_UA',
                'separate_well_mixed_connected_and_sealed_regions',
                'melting_driven_opening_transfers_solid_and_liquid',
                'ideal_substitutional_solid_solution',
                'constant_effective_solid_molar_volume_equal_to_host',
                'reversible_phase_exchange_with_first_mode_solid_diffusion',
                'capillary_bundle_darcy_drainage_with_cubic_relative_permeability',
                'fixed_envelope_with_free_swelling_no_collapse_or_detachment',
                'constant_heating_medium_temperature_no_equipment_thermal_mass',
            ],
        }
        warnings = []
        if sweat_total <= 1e-15:
            warnings.append('No sweat drained: connectivity, capillary retention, or residual saturation prevents flow.')
        return product, sweat, details, warnings

    def _number(self, name, default=None, **constraints):
        try:
            return finite_crystallizer_number(
                self.get_param(name, default), name, **constraints
            )
        except CrystallizerSpecificationError as error:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' {error}"
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
                    f"{self.unit_type_name} '{self.unit_id}' outlet LLE check failed: {error}. "
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
                f"{self.unit_type_name} '{self.unit_id}' outlet mother liquor exhibits LLE. "
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
        diagnostic_stream = slurry
        sweating_details = None
        sweating_warnings = []
        separation = None
        if retention_spec is None:
            outlets = {'out': slurry}
        else:
            cake, liquor, separation = self._split_slurry(slurry, retention_spec)
            if retention_spec.get('trapped_component_flows'):
                diagnostic_stream = liquor
                if liquor.F <= 1e-15:
                    # Fully retained free liquor still needs its own LLE check;
                    # its composition differs from the historical inclusions.
                    diagnostic_stream = self.thermo.calculate_state(
                        slurry.T, slurry.P, 1.0,
                        performance['mother_liquor_composition'], phase='liquid',
                    )
            if self._layer_sweating_enabled():
                product, sweat, sweating_details, sweating_warnings = (
                    self._sweat_and_harvest_layer(cake, performance.get(
                        'crystallizing_component', candidates[0]), retention_spec)
                )
                phase_details['layer_sweating'] = sweating_details
                outlets = {
                    'product': product,
                    'mother_liquor': liquor,
                    'sweat': sweat,
                }
            else:
                outlets = {'cake': cake, 'mother_liquor': liquor}
            for stream in outlets.values():
                propagate_particle_size_distributions((slurry,), stream)
                stream.solid_particle_properties = {
                    c: dict(p)
                    for c, p in slurry.solid_particle_properties.items()
                    if stream.solid_component_flows.get(c, 0) > 1e-15
                }
                stream.validate_particle_size_distributions()
        diagnostic, warnings = self._lle_diagnostic(diagnostic_stream)
        warnings.extend(sweating_warnings)
        for stream in outlets.values():
            if self.crystallization_mode == 'layer':
                # A deposited layer has no particle diameter or sphericity.
                # State construction and split propagation may attach defaults.
                for component in candidates:
                    stream.solid_particle_size_distributions.pop(component, None)
                    stream.solid_particle_properties.pop(component, None)
            for name, details in phase_details.items():
                stream.phase_details[name] = dict(details)
            if diagnostic is not None:
                stream.phase_details['crystallizer_lle_check'] = dict(diagnostic)
        if any(
            s.H is None or not math.isfinite(s.H) for s in (inlet, *outlets.values())
        ):
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' requires finite inlet and outlet enthalpy"
            )
        duty = sum(s.F * s.H for s in outlets.values()) - inlet.F * inlet.H
        performance = {
            **performance,
            'crystallization_mode': self.crystallization_mode,
            'T_out_C': slurry.T - 273.15,
            'P_out_bar': slurry.P,
            'outlet_mode': (
                'layer_sweating_and_harvest' if sweating_details is not None
                else 'layer_drainage' if self.crystallization_mode == 'layer'
                else 'slurry' if separation is None else 'cake_split'
            ),
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
        if sweating_details is not None:
            performance['layer_sweating'] = sweating_details
            performance['sweat_mass_kg_per_h'] = sweating_details[
                'sweat_mass_kg_per_h'
            ]
            performance['harvest_product_mass_kg_per_h'] = sweating_details[
                'harvest_product_mass_kg_per_h'
            ]
            performance['impurity_rejection_to_sweat'] = sweating_details[
                'impurity_rejection_to_sweat'
            ]
            performance['sweat_composition'] = sweating_details[
                'sweat_composition'
            ]
            performance['harvest_product_composition'] = sweating_details[
                'harvest_product_composition'
            ]
        if diagnostic is not None:
            performance['outlet_lle_check'] = diagnostic
        if 'wall_duty_kW' in performance:
            performance['conditioning_duty_kW'] = duty / 3600 - performance['wall_duty_kW']
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
        allow_negative=False,
    ):
        found = [
            (name, self.get_param(name), self.get_param_unit(name))
            for name in names
            if self.get_param(name) is not None
        ]
        if len(found) > 1:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' received duplicate aliases for "
                f'{label}: ' + ', '.join(name for name, _value, _unit in found)
            )
        if not found:
            return None
        name, raw_value, raw_unit = found[0]
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' {name} must be numeric"
            ) from error
        unit = self._normalized_unit(raw_unit) or default_unit
        factor = factors.get(unit)
        if factor is None:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' does not recognize {label} "
                f'unit {raw_unit!r}'
            )
        value *= factor
        if not math.isfinite(value) or (not allow_negative and (value < 0.0 if allow_zero else value <= 0.0)):
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' {name} must be "
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
                f"{self.unit_type_name} '{self.unit_id}' outlet_sphericity must be numeric"
            ) from error
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' outlet_sphericity must be in (0, 1]"
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
                            f"{self.unit_type_name} '{self.unit_id}' inlet sphericity for "
                            f'{component!r} must be numeric'
                        ) from error
                    if not math.isfinite(sphericity) or not 0.0 < sphericity <= 1.0:
                        raise UnitOperationError(
                            f"{self.unit_type_name} '{self.unit_id}' inlet sphericity for "
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
                f"{self.unit_type_name} '{self.unit_id}' {error}"
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
                f"{self.unit_type_name} '{self.unit_id}' MSMPR mode requires exactly one "
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
                f"{self.unit_type_name} '{self.unit_id}' must specify only one mother-"
                'liquor retention basis'
            )
        if fraction_specs:
            name, _value = fraction_specs[0]
            retention = self._number('mother_liquor_retention', minimum=-float('inf'))
            if not math.isfinite(retention) or not 0.0 <= retention <= 1.0:
                raise UnitOperationError(
                    f"{self.unit_type_name} '{self.unit_id}' {name} must be a "
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
                    f"{self.unit_type_name} '{self.unit_id}' mother_liquor_retention_rate "
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
        trapped_flows = retention_spec.get('trapped_component_flows', {})
        for component, flow in trapped_flows.items():
            available = liquid_flows.get(component, 0.0)
            if not math.isfinite(flow) or flow < 0 or flow > available + max(1e-10, available * 1e-10):
                raise UnitOperationError('Trapped liquid exceeds available component inventory')
            liquid_flows[component] = max(0.0, available - flow)
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
                    f"{self.unit_type_name} '{self.unit_id}' cannot apply a positive "
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
                    f"{self.unit_type_name} '{self.unit_id}' mother_liquor_retention_rate "
                    f'requires {retained_liquor_mass:g} kg/h mother liquor but '
                    f'only {liquid_mass:g} kg/h is available (maximum rate '
                    f'{available_rate:g} kg/kg crystals)'
                )
            retention = retained_liquor_mass / liquid_mass if liquid_mass > 0.0 else 0.0
        cake_flows = {
            component: solid_flows.get(component, 0.0)
            + trapped_flows.get(component, 0.0)
            + retention * liquid_flows.get(component, 0.0)
            for component in set(solid_flows) | set(liquid_flows)
        }
        mother_liquor_flows = {
            component: (1.0 - retention) * flow
            for component, flow in liquid_flows.items()
        }
        cake_total = sum(cake_flows.values())
        liquor_total = sum(mother_liquor_flows.values())
        liquid_fallback = self._composition_from_flows(
            liquid_flows, slurry.x1 or slurry.x or slurry.composition
        )
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
            'trapped_liquid_mass_kg_per_h': self._component_mass_flow(trapped_flows),
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
                f"{self.unit_type_name} '{self.unit_id}' MSMPR mode currently requires "
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
                f"{self.unit_type_name} '{self.unit_id}' MSMPR crystallizing_component "
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
                f"{self.unit_type_name} '{self.unit_id}' MSMPR mode only accepts seed "
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
                f"{self.unit_type_name} '{self.unit_id}' MSMPR kinetics are invalid: {error}"
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
                f"{self.unit_type_name} '{self.unit_id}' MSMPR calculation failed: {error}"
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

    def _solve_layer_growth(self, inlet, temperature, pressure, candidates, retention):
        if inlet.vapor_fraction > 1e-10:
            raise UnitOperationError('Layer growth requires a liquid feed; condensation is not modeled')
        if len(candidates) != 1:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' layer_growth requires one crystallizing component"
            )
        growth_time = self._one_dimension(('growth_time',), self._TIME_FACTORS_H, 'growth time', 'h')
        cycle_time = self._one_dimension(('cycle_time',), self._TIME_FACTORS_H, 'cycle time', 'h')
        if cycle_time < growth_time:
            raise UnitOperationError('Layer cycle_time must be at least growth_time')
        area = self._one_dimension(('cooled_area',), {'m2': 1, 'm^2': 1}, 'cooled area', 'm2')
        film = self._one_dimension(('film_thickness',), self._DIAMETER_FACTORS_M, 'film thickness', 'm')
        thermal_film = self._one_dimension(
            ('thermal_film_thickness',), self._DIAMETER_FACTORS_M, 'thermal film thickness', 'm'
        )
        if thermal_film is None:
            thermal_film = film
        length = self._one_dimension(('plate_length',), self._DIAMETER_FACTORS_M, 'plate length', 'm')
        velocity = self._one_dimension(
            ('liquid_velocity',), {'m/s': 1, 'm/h': 1/3600, 'cm/s': 0.01}, 'liquid velocity', 'm/s'
        )
        diffusivity = self._one_dimension(
            ('binary_diffusivity',), {'m2/s': 1, 'm^2/s': 1, 'cm2/s': 1e-4,
                                      'cm^2/s': 1e-4, 'm2/h': 1/3600},
            'binary diffusivity', 'm2/s',
        )
        self._number('t_wall', minimum=-float('inf'))
        wall = self.get_temperature_param('t_wall')
        component = candidates[0]
        try:
            growth = solve_layer_growth(
                self.thermo, component=component,
                amounts_kmol={c: n * cycle_time for c, n in inlet.component_flows().items()},
                bulk_temperature_K=temperature, pressure_bar=pressure,
                wall_temperature_K=wall, area_m2=area, film_thickness_m=film,
                thermal_film_thickness_m=thermal_film,
                thermal_mode=self.get_param('thermal_mode'), film_model=self.get_param('film_model'),
                plate_length_m=length, liquid_velocity_m_s=velocity,
                inclusion_max_fraction=self._number('inclusion_max_fraction', 0, inclusive=True, maximum=1),
                growth_time_h=growth_time, binary_diffusivity_m2_s=diffusivity,
                relative_tolerance=self._number('layer_relative_tolerance', 1e-6),
                profile_points=self._number('layer_profile_points', 21, minimum=2, inclusive=True, integer=True),
            )
            solid_flow = growth.solid_amount_kmol / cycle_time
            outlet = self.thermo.calculate_state_with_solid_flows(
                growth.bulk_temperature_K, pressure, inlet.F, inlet.composition,
                {component: solid_flow} if solid_flow > 1e-15 else {}, phase='liquid',
            )
        except (ThermodynamicsError, ValueError, TypeError) as error:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' layer growth failed: {error}"
            ) from error
        details = {
            'model': 'planar_quasi_steady_layer_growth',
            'component': component,
            'layer_thickness_m': growth.thickness_m,
            'cooled_area_m2': area,
            'film_thickness_m': growth.profile[-1]['film_thickness_m'],
            'thermal_film_thickness_m': growth.profile[-1]['thermal_film_thickness_m'],
            'solid_conduction_model': 'integrated_temperature_dependent_conductivity',
            'growth_time_h': growth_time,
            'cycle_time_h': cycle_time,
            'T_wall_K': wall,
            'T_bulk_K': growth.bulk_temperature_K,
            'T_initial_K': temperature,
            'thermal_mode': self.get_param('thermal_mode'),
            'film_model': self.get_param('film_model'),
            'plate_length_m': length,
            'liquid_velocity_m_s': velocity,
            'inclusion_max_fraction': self.get_param('inclusion_max_fraction', 0),
            'inclusion_model': 'mechanical_bulk_liquid_capture',
            'mass_transfer_model': 'pseudo_binary_solvent_blend',
            'trapped_component_amounts_kmol_per_batch': growth.trapped_component_amounts_kmol,
            'energy_balance_residual_kJ_per_batch': growth.energy_residual_kJ,
            'temperature_control_energy_kJ_per_batch': growth.profile[-1]['temperature_control_energy_kJ'],
            'solid_amount_kmol_per_batch': growth.solid_amount_kmol,
            'wall_energy_kJ_per_batch': growth.wall_energy_kJ,
            'wall_duty_kW': growth.wall_energy_kJ / (cycle_time * 3600),
            'mass_diffusivity_model': 'specified_maxwell_stefan' if diffusivity is not None else 'wilke_chang_vignes',
            'liquid_conductivity_model': 'Li_1976',
            'evaluations': growth.evaluations,
            'assumptions': [
                'multicomponent_homogeneous_liquid', 'one_pure_solid', 'planar_geometry',
                'fixed_noncrystallizing_component_ratios_in_selective_film',
                self.get_param('thermal_mode') + '_bulk', 'constant_wall_temperature',
                'quasi_steady_thermal_profiles', 'constant_solid_density_at_wall',
                'instantaneous_surface_nucleation', 'no_bulk_nucleation',
                'no_soret_effect', 'growth_submodel_excludes_sweating',
                'harvest_conditioned_to_bulk_temperature',
                'mechanical_inclusions_bypass_selective_film', 'harmonic_porous_conductivity',
                'no_sensible_heat_redistribution_in_existing_layer',
            ],
            'profile': growth.profile,
        }
        performance = {
            **details,
            'crystallizing_component': component,
            'solid_component_flows_kmol_per_h': {component: solid_flow},
            'crystallized_component_flows_kmol_per_h': {component: solid_flow},
            'crystal_yields': {component: solid_flow / inlet.component_flows()[component]},
            'mother_liquor_composition': self._composition_from_flows(growth.liquid_component_amounts_kmol, {}),
        }
        retention = {**retention, 'trapped_component_flows': {
            c: n / cycle_time for c, n in growth.trapped_component_amounts_kmol.items() if n > 0
        }}
        return self._finalize(inlet, outlet, candidates, None, retention, performance,
                              {'layer_crystallization': details})

    def _solve_empirical_layer_growth(
        self, inlet, temperature, pressure, candidates, retention
    ):
        if inlet.vapor_fraction > 1e-10:
            raise UnitOperationError(
                'Empirical layer growth requires a liquid feed; condensation is not modeled'
            )
        requested_component = self.get_param('crystallizing_component')
        if requested_component is None:
            if len(candidates) != 1:
                raise UnitOperationError(
                    f"{self.unit_type_name} '{self.unit_id}' empirical_layer_growth "
                    'requires crystallizing_component when more than one '
                    'conventional_with_solid component is present'
                )
            component = candidates[0]
        else:
            component = next(
                (
                    name for name in candidates
                    if name.casefold() == str(requested_component).strip().casefold()
                ),
                None,
            )
            if component is None:
                raise UnitOperationError(
                    f"{self.unit_type_name} '{self.unit_id}' empirical_layer_growth "
                    f"crystallizing_component {requested_component!r} must name "
                    'a present conventional_with_solid component'
                )
        growth_time = self._one_dimension(
            ('growth_time',), self._TIME_FACTORS_H, 'growth time', 'h'
        )
        cycle_time = self._one_dimension(
            ('cycle_time',), self._TIME_FACTORS_H, 'cycle time', 'h'
        )
        if cycle_time < growth_time:
            raise UnitOperationError('Layer cycle_time must be at least growth_time')
        area = self._one_dimension(
            ('cooled_area',), {'m2': 1, 'm^2': 1}, 'cooled area', 'm2'
        )
        self._number('t_wall', minimum=-float('inf'))
        wall = self.get_temperature_param('t_wall')
        density = self._one_dimension(
            ('layer_solid_density',),
            {'kg/m3': 1, 'kg/m^3': 1, 'g/cm3': 1000, 'g/cm^3': 1000},
            'layer solid density', 'kg/m3',
        )
        try:
            growth_definition = empirical_layer_growth_definition_from_parameters(
                self.params
            )
            growth_law = empirical_layer_growth_law_from_mapping(growth_definition)
            distribution_definitions = (
                empirical_layer_distribution_definitions_from_parameters(
                    self.params, inlet.component_flows()
                )
            )
            if component in distribution_definitions:
                raise EmpiricalLayerDefinitionError(
                    'the crystallizing component cannot have a k_eff law'
                )
            distribution_laws = {
                impurity: empirical_layer_distribution_law_from_mapping(
                    definition, impurity
                )
                for impurity, definition in distribution_definitions.items()
            }
            growth = solve_empirical_layer_growth(
                self.thermo,
                component=component,
                amounts_kmol={
                    name: amount * cycle_time
                    for name, amount in inlet.component_flows().items()
                },
                bulk_temperature_K=temperature,
                pressure_bar=pressure,
                wall_temperature_K=wall,
                area_m2=area,
                growth_time_h=growth_time,
                growth_law=growth_law,
                distribution_laws=distribution_laws,
                solid_density_kg_m3=density,
                relative_tolerance=self._number(
                    'empirical_relative_tolerance', 1e-7
                ),
                profile_points=self._number(
                    'layer_profile_points', 21,
                    minimum=2, inclusive=True, integer=True,
                ),
            )
        except (EmpiricalLayerDefinitionError, ThermodynamicsError, ValueError, TypeError) as error:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' empirical layer growth failed: {error}"
            ) from error
        solid_flow = growth.solid_amount_kmol / cycle_time
        outlet = self.thermo.calculate_state_with_solid_flows(
            temperature, pressure, inlet.F, inlet.composition,
            {component: solid_flow} if solid_flow > 1e-15 else {},
            phase='liquid',
        )
        trapped_flows = {
            name: amount / cycle_time
            for name, amount in growth.trapped_component_amounts_kmol.items()
            if amount > 0
        }
        details = {
            'model': 'empirical_finite_rate_layer_growth',
            'component': component,
            'layer_thickness_m': growth.thickness_m,
            'cooled_area_m2': area,
            'growth_time_h': growth_time,
            'elapsed_growth_time_h': growth.elapsed_growth_time_h,
            'cycle_time_h': cycle_time,
            'T_wall_K': wall,
            'T_bulk_K': temperature,
            'solid_density_kg_m3': growth.solid_density_kg_m3,
            'growth_model': growth_law.model,
            'growth_expression': growth_law.expression.text,
            'growth_rate_unit': growth_law.declared_unit,
            'effective_distribution_basis': (
                'instantaneous_complete_deposited_layer_mole_fraction'
            ),
            'effective_distribution_models': {
                impurity: law.model for impurity, law in distribution_laws.items()
            },
            'effective_distribution_expressions': {
                impurity: law.expression.text
                for impurity, law in distribution_laws.items()
            },
            'unspecified_impurity_behavior': 'complete_rejection',
            'trapped_component_amounts_kmol_per_batch': dict(
                growth.trapped_component_amounts_kmol
            ),
            'stopped_by_inventory': growth.stopped_by_inventory,
            'evaluations': growth.evaluations,
            'assumptions': [
                'one_crystalline_component',
                'empirical_planar_growth_rate',
                'instantaneous_differential_effective_distribution_coefficients',
                'incorporated_impurities_reported_as_trapped_material',
                'constant_bulk_and_wall_temperatures',
                'growth_submodel_excludes_sweating',
                'no_remelting',
                'unspecified_impurities_completely_rejected',
            ],
            'profile': growth.profile,
        }
        performance = {
            **details,
            'crystallizing_component': component,
            'solid_component_flows_kmol_per_h': {component: solid_flow},
            'crystallized_component_flows_kmol_per_h': {component: solid_flow},
            'crystal_yields': {
                component: solid_flow / inlet.component_flows()[component]
            },
            'empirically_incorporated_component_flows_kmol_per_h': trapped_flows,
            'mother_liquor_composition': self._composition_from_flows(
                growth.liquid_component_amounts_kmol, {}
            ),
        }
        retention = {
            **retention,
            'trapped_component_flows': trapped_flows,
        }
        return self._finalize(
            inlet, outlet, [component], None, retention, performance,
            {'layer_crystallization': details},
        )

    def solve(self, inlets) -> UnitResult:
        if len(inlets) != 1:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' requires exactly one inlet stream"
            )
        inlet = next(iter(inlets.values()))
        if not math.isfinite(inlet.F) or inlet.F <= 0.0:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' requires positive inlet flow"
            )

        try:
            self.params = self.specification_validator(
                self.params, self.get_param('__connected_outlet_ports__')
            )
        except CrystallizerSpecificationError as error:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' {error}"
            ) from error
        if self.get_param('model') == 'layer_growth' and self.get_param('thermal_mode') == 'cooling':
            temperature = inlet.T
        else:
            self._number('t_out', minimum=-float('inf'))
            temperature = self.get_temperature_param('t_out')
        if temperature <= 0:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' requires positive outlet temperature"
            )
        pressure = (
            self._number('p_out')
            if self.get_param('p_out') is not None
            else inlet.P - self._number('p_drop', 0.0, inclusive=True)
        )
        if not math.isfinite(pressure) or pressure <= 0:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' outlet pressure must be positive"
            )
        model = self.get_param('model')
        if self.crystallization_mode == 'layer' and any(
            flow > 1e-15 for flow in inlet.solid_component_flows.values()
        ):
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' requires a solid-free "
                'liquid feed; suspended-solid capture is not modeled'
            )
        specified_outlet_sphericity = self._specified_outlet_sphericity()
        candidates = [
            component
            for component in self.thermo.conventional_solid_components
            if inlet.composition.get(component, 0.0) > 1.0e-15
        ]
        if not candidates:
            raise UnitOperationError(
                f"{self.unit_type_name} '{self.unit_id}' feed contains no component "
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

        if model == 'layer_growth':
            return self._solve_layer_growth(
                inlet, temperature, pressure, candidates, mother_liquor_retention,
            )

        if model == 'empirical_layer_growth':
            return self._solve_empirical_layer_growth(
                inlet, temperature, pressure, candidates, mother_liquor_retention,
            )

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
                f"{self.unit_type_name} '{self.unit_id}' SLE calculation failed: {exc}"
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
        phase_details = {'solid_liquid_equilibrium': sle_details}
        if self.crystallization_mode == 'layer':
            performance['model'] = 'equilibrium_pure_solid_layer'
            phase_details['layer_crystallization'] = {
                'model': 'equilibrium_pure_solid_layer',
                'assumptions': [
                    'equilibrium_endpoint',
                    'one_homogeneous_mother_liquor',
                    'pure_solid_deposits',
                    'specified_mother_liquor_retention',
                    'no_growth_or_transport_prediction',
                ],
            }
        return self._finalize(
            inlet,
            outlet,
            candidates,
            specified_outlet_sphericity,
            mother_liquor_retention,
            performance,
            phase_details,
        )


class Crystallizer(_CrystallizerBase):
    """Equilibrium or kinetic MSMPR suspension crystallizer."""

    crystallization_mode = 'suspension'
    specification_validator = staticmethod(validate_crystallizer_specification)
    unit_type_name = 'Crystallizer'


class LayerCrystallizer(_CrystallizerBase):
    """Equilibrium, mechanistic, or empirical layer crystallizer."""

    crystallization_mode = 'layer'
    specification_validator = staticmethod(validate_layer_crystallizer_specification)
    unit_type_name = 'LayerCrystallizer'


__all__ = ['Crystallizer', 'LayerCrystallizer']
