"""Solid-forming unit operations."""

from __future__ import annotations

import math

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics_models.common import ThermodynamicsError
    from .thermodynamics_models.sle import solve_pure_solid_sle
    from .unit_operations_base import UnitOperation, UnitOperationError, UnitResult
else:
    from thermodynamics_models.common import ThermodynamicsError
    from thermodynamics_models.sle import solve_pure_solid_sle
    from unit_operations_base import UnitOperation, UnitOperationError, UnitResult


class Crystallizer(UnitOperation):
    """Equilibrium cooling crystallizer with one retained slurry outlet."""

    supports_permanent_solids = True

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
