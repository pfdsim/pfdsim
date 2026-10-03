"""
Conversion, equilibrium, continuous, batch, and plug-flow reactor operations.
"""

import math

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics import StreamState, ThermodynamicsError
else:
    from thermodynamics import StreamState, ThermodynamicsError
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_base import UnitOperation, UnitOperationError, UnitResult
else:
    from unit_operations_base import UnitOperation, UnitOperationError, UnitResult
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_basic import _ThermoStateSolver
else:
    from unit_operations_basic import _ThermoStateSolver
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .reaction_models import (
        ReactionDefinitionError,
        conversion_reaction_from_mapping,
        equilibrium_reaction_from_mapping,
        solve_conversion_extents,
        solve_homogeneous_equilibrium,
    )
else:
    from reaction_models import (
        ReactionDefinitionError,
        conversion_reaction_from_mapping,
        equilibrium_reaction_from_mapping,
        solve_conversion_extents,
        solve_homogeneous_equilibrium,
    )


class Reactor(UnitOperation):
    """Stoichiometric conversion reactor with inlet-basis extents."""

    supports_reactions = True

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if len(inlets) != 1:
            raise UnitOperationError(
                f"Reactor '{self.unit_id}' requires exactly one inlet stream"
            )
        inlet = next(iter(inlets.values()))
        raw_reactions = self.params.get('reactions', [])
        if not raw_reactions:
            raise UnitOperationError(f"Reactor '{self.unit_id}' has no reactions defined")

        component_symbols = tuple(getattr(
            self.thermo,
            'process_components',
            self.thermo.components,
        ))
        component_metadata = {
            component: self.thermo.props.get(component)
            for component in component_symbols
        }
        try:
            reactions = [
                conversion_reaction_from_mapping(
                    definition,
                    component_symbols,
                    component_metadata,
                )
                for definition in raw_reactions
            ]
            extent_solution = solve_conversion_extents(
                inlet.component_flows(),
                reactions,
            )
        except ReactionDefinitionError as error:
            raise UnitOperationError(
                f"Reactor '{self.unit_id}' reaction specification is invalid: {error}"
            ) from error

        outlet_flows = extent_solution.outlet_component_flows
        total_moles = sum(outlet_flows.values())
        if total_moles <= 0.0:
            raise UnitOperationError(
                f"Reactor '{self.unit_id}' reactions leave no positive outlet flow"
            )
        composition = {
            component: flow / total_moles
            for component, flow in outlet_flows.items()
            if flow > 0.0
        }

        self._validate_formation_enthalpies(
            set(inlet.component_flows()) | set(outlet_flows)
        )
        if inlet.H is None:
            raise UnitOperationError(
                f"Reactor '{self.unit_id}' inlet enthalpy was not calculated"
            )

        P_drop = float(self.get_param('P_drop', 0.0))
        if not math.isfinite(P_drop) or P_drop < 0.0:
            raise UnitOperationError(
                f"Reactor '{self.unit_id}' requires a finite, nonnegative P_drop"
            )
        P_out = inlet.P - P_drop
        if P_out <= 0.0:
            raise UnitOperationError(
                f"Reactor '{self.unit_id}' outlet pressure must be positive"
            )

        duty_spec = self._heat_duty_spec()
        temperature_spec = self.get_temperature_param('T_out')
        if temperature_spec is None:
            temperature_spec = self.get_temperature_param('T')
        raw_mode = self.get_param('mode')
        mode = str(raw_mode or ('duty' if duty_spec is not None else 'isothermal')).strip().lower()
        if mode in {'specified_duty', 'heat_duty'}:
            mode = 'duty'
        if mode not in {'isothermal', 'adiabatic', 'duty'}:
            raise UnitOperationError(
                f"Reactor '{self.unit_id}' mode must be isothermal, adiabatic, or duty"
            )

        inlet_enthalpy_flow = inlet.F * inlet.H
        if mode == 'isothermal':
            if duty_spec is not None:
                raise UnitOperationError(
                    f"Reactor '{self.unit_id}' isothermal mode cannot also specify duty"
                )
            T_out = inlet.T if temperature_spec is None else temperature_spec
            outlet = self.thermo.calculate_state(
                T_out,
                P_out,
                total_moles,
                composition,
                include=('H', 'Cp'),
            )
            Q = total_moles * outlet.H - inlet_enthalpy_flow
        elif mode == 'adiabatic':
            if temperature_spec is not None or duty_spec is not None:
                raise UnitOperationError(
                    f"Reactor '{self.unit_id}' adiabatic mode cannot specify T/T_out or duty"
                )
            Q = 0.0
            outlet = self._state_for_enthalpy(
                composition,
                P_out,
                total_moles,
                inlet_enthalpy_flow / total_moles,
                inlet.T,
            )
        else:
            if duty_spec is None:
                raise UnitOperationError(
                    f"Reactor '{self.unit_id}' duty mode requires Q/duty/heat_duty"
                )
            if temperature_spec is not None:
                raise UnitOperationError(
                    f"Reactor '{self.unit_id}' duty mode cannot also specify T/T_out"
                )
            Q = duty_spec
            outlet = self._state_for_enthalpy(
                composition,
                P_out,
                total_moles,
                (inlet_enthalpy_flow + Q) / total_moles,
                inlet.T,
            )

        energy_residual = total_moles * outlet.H - inlet_enthalpy_flow - Q
        energy_tolerance = max(
            1.0e-5,
            abs(inlet_enthalpy_flow) * 1.0e-9,
            abs(Q) * 1.0e-9,
        )
        if abs(energy_residual) > energy_tolerance:
            raise UnitOperationError(
                f"Reactor '{self.unit_id}' energy balance residual is "
                f"{energy_residual:.6g} kJ/h"
            )

        reaction_results = self._reaction_performance(extent_solution)
        desired_product_result = self._desired_product_performance(
            extent_solution,
            reactions,
            component_symbols,
        )
        performance = {
            'mode': mode,
            'T_in_C': inlet.T - 273.15,
            'T_out_C': outlet.T - 273.15,
            'P_in_bar': inlet.P,
            'P_out_bar': outlet.P,
            'P_drop_bar': P_drop,
            'reactions': reaction_results,
            'component_conversions': extent_solution.component_conversions,
            'component_formation_kmol_h': extent_solution.component_formation_flows,
            'component_consumption_kmol_h': extent_solution.component_consumption_flows,
            'duty_kW': Q / 3600.0,
            'energy_balance_residual_kW': energy_residual / 3600.0,
        }
        if desired_product_result is not None:
            performance['desired_product'] = desired_product_result

        return UnitResult(
            outlet_streams={'out': outlet},
            heat_duty=Q,
            performance=performance,
            warnings=list(extent_solution.warnings),
        )

    def _validate_formation_enthalpies(self, components: set[str]) -> None:
        missing = sorted(
            component
            for component in components
            if (
                self.thermo.props.get(component) is None
                or self.thermo.props[component].Hf is None
            )
        )
        if missing:
            raise UnitOperationError(
                f"{type(self).__name__} '{self.unit_id}' requires standard "
                "formation enthalpy Hf "
                "for every participating or carried component; missing: "
                + ', '.join(missing)
            )

    def _heat_duty_spec(self):
        for name in ('Q', 'duty', 'heat_duty'):
            value = self.get_param(name)
            if value is None:
                continue
            duty = float(value)
            if not math.isfinite(duty):
                raise UnitOperationError(
                    f"Reactor '{self.unit_id}' duty must be finite"
                )
            unit = (self.get_param_unit(name) or '').strip().lower()
            if unit in ('w', 'watt', 'watts'):
                return duty * 3.6
            if unit in ('mw', 'megawatt', 'megawatts'):
                return duty * 3.6e6
            if unit in ('kw', 'kilowatt', 'kilowatts'):
                return duty * 3600.0
            if unit in ('kj/h', 'kj/hr', 'kj per h', 'kj per hour'):
                return duty
            if unit in ('j/h', 'j/hr', 'j per h', 'j per hour'):
                return duty / 1000.0
            return duty * 3600.0 if abs(duty) < 1.0e6 else duty
        return None

    def _state_for_enthalpy(
        self,
        composition: dict[str, float],
        P: float,
        F: float,
        H_target: float,
        T_guess: float,
    ) -> StreamState:
        state, residual = _ThermoStateSolver(
            self.thermo,
            f"Reactor '{self.unit_id}'",
        ).state_at_enthalpy(
            P,
            F,
            composition,
            H_target,
            T_guess,
            include=('H', 'Cp'),
        )
        if abs(residual) > max(1.0e-5, abs(H_target) * 1.0e-8):
            raise UnitOperationError(
                f"Reactor '{self.unit_id}' outlet enthalpy residual is "
                f"{residual:.6g} kJ/kmol"
            )
        return state

    def _reaction_performance(self, extent_solution) -> list[dict]:
        basis_totals: dict[str, float] = {}
        for extent in extent_solution.reaction_extents:
            basis = extent.specification.basis_component
            basis_totals[basis] = basis_totals.get(basis, 0.0) + extent.basis_consumed_kmol_h

        results = []
        for index, extent in enumerate(extent_solution.reaction_extents, start=1):
            specification = extent.specification
            basis = specification.basis_component
            basis_total = basis_totals[basis]
            reaction_yield = (
                extent.basis_consumed_kmol_h / extent.basis_inlet_kmol_h
                if extent.basis_inlet_kmol_h > 0.0 else 0.0
            )
            reaction_selectivity = (
                extent.basis_consumed_kmol_h / basis_total
                if basis_total > 0.0 else 0.0
            )
            standard_heat = sum(
                coefficient * self.thermo.props[component].Hf * 1000.0
                for component, coefficient
                in specification.reaction.stoichiometry.items()
            )
            results.append({
                'index': index,
                'name': specification.name,
                'equation': specification.reaction.equation,
                'reversible_equation': specification.reaction.reversible,
                'stoichiometry': dict(specification.reaction.stoichiometry),
                'basis_component': basis,
                'specified_conversion': specification.specified_conversion,
                'actual_conversion': reaction_yield,
                'yield_on_inlet_basis': reaction_yield,
                'selectivity_on_converted_basis': reaction_selectivity,
                'extent_kmol_h': extent.extent_kmol_h,
                'basis_consumed_kmol_h': extent.basis_consumed_kmol_h,
                'standard_heat_of_reaction_kJ_per_kmol_extent': standard_heat,
            })
        return results

    def _desired_product_performance(
        self,
        extent_solution,
        reactions,
        component_symbols,
    ) -> dict | None:
        desired = self.get_param('desired_product')
        if desired is None:
            return None
        desired = str(desired).strip()
        if desired not in component_symbols:
            raise UnitOperationError(
                f"Reactor '{self.unit_id}' desired_product '{desired}' is not a process component"
            )
        if not any(
            specification.reaction.stoichiometry.get(desired, 0.0) > 0.0
            for specification in reactions
        ):
            raise UnitOperationError(
                f"Reactor '{self.unit_id}' desired_product '{desired}' is not formed by any reaction"
            )

        report_basis = self.get_param('report_basis')
        if report_basis is None:
            bases = {specification.basis_component for specification in reactions}
            if len(bases) != 1:
                raise UnitOperationError(
                    f"Reactor '{self.unit_id}' desired_product reporting requires report_basis "
                    "because the reactions use multiple basis components"
                )
            report_basis = next(iter(bases))
        report_basis = str(report_basis).strip()
        if report_basis not in component_symbols:
            raise UnitOperationError(
                f"Reactor '{self.unit_id}' report_basis '{report_basis}' is not a process component"
            )

        inlet_basis = extent_solution.inlet_component_flows.get(report_basis, 0.0)
        basis_consumed = max(
            0.0,
            inlet_basis - extent_solution.outlet_component_flows.get(report_basis, 0.0),
        )
        desired_formed = max(
            0.0,
            extent_solution.outlet_component_flows.get(desired, 0.0)
            - extent_solution.inlet_component_flows.get(desired, 0.0),
        )
        return {
            'component': desired,
            'report_basis': report_basis,
            'net_formation_kmol_h': desired_formed,
            'yield_mol_per_mol_inlet_basis': (
                desired_formed / inlet_basis if inlet_basis > 0.0 else None
            ),
            'selectivity_mol_per_mol_basis_consumed': (
                desired_formed / basis_consumed if basis_consumed > 0.0 else None
            ),
        }


class EquilibriumReactor(Reactor):
    """Homogeneous stoichiometric chemical-equilibrium reactor."""

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if len(inlets) != 1:
            raise UnitOperationError(
                f"EquilibriumReactor '{self.unit_id}' requires exactly one inlet stream"
            )
        inlet = next(iter(inlets.values()))
        raw_reactions = self.params.get('reactions', [])
        if not raw_reactions:
            raise UnitOperationError(
                f"EquilibriumReactor '{self.unit_id}' has no reactions defined"
            )
        if hasattr(self.thermo, '_initialize_vdm'):
            raise UnitOperationError(
                f"EquilibriumReactor '{self.unit_id}' does not support VDM "
                "thermodynamic methods because vapor association is already an "
                "embedded chemical-equilibrium model"
            )

        phase = str(self.get_param('phase', '')).strip().lower()
        if phase == 'gas':
            phase = 'vapor'
        if phase not in {'vapor', 'liquid'}:
            raise UnitOperationError(
                f"EquilibriumReactor '{self.unit_id}' requires explicit "
                "phase=vapor or phase=liquid"
            )

        component_symbols = tuple(self.thermo.components)
        component_metadata = {
            component: self.thermo.props.get(component)
            for component in component_symbols
        }
        try:
            reactions = tuple(
                equilibrium_reaction_from_mapping(
                    definition,
                    component_symbols,
                    component_metadata,
                )
                for definition in raw_reactions
            )
        except ReactionDefinitionError as error:
            raise UnitOperationError(
                f"EquilibriumReactor '{self.unit_id}' reaction specification "
                f"is invalid: {error}"
            ) from error

        carried = set(inlet.component_flows())
        participating = {
            component
            for specification in reactions
            for component in specification.reaction.components
        }
        self._validate_equilibrium_thermochemistry(
            carried,
            participating,
            reactions,
        )

        P_drop = float(self.get_param('P_drop', 0.0))
        if not math.isfinite(P_drop) or P_drop < 0.0:
            raise UnitOperationError(
                f"EquilibriumReactor '{self.unit_id}' requires a finite, "
                "nonnegative P_drop"
            )
        P_out = inlet.P - P_drop
        if P_out <= 0.0:
            raise UnitOperationError(
                f"EquilibriumReactor '{self.unit_id}' outlet pressure must be positive"
            )

        duty_spec = self._heat_duty_spec()
        temperature_spec = self.get_temperature_param('T_out')
        if temperature_spec is None:
            temperature_spec = self.get_temperature_param('T')
        raw_mode = self.get_param('mode')
        mode = str(
            raw_mode or ('duty' if duty_spec is not None else 'isothermal')
        ).strip().lower()
        if mode in {'specified_duty', 'heat_duty'}:
            mode = 'duty'
        if mode not in {'isothermal', 'adiabatic', 'duty'}:
            raise UnitOperationError(
                f"EquilibriumReactor '{self.unit_id}' mode must be "
                "isothermal, adiabatic, or duty"
            )
        if inlet.H is None:
            raise UnitOperationError(
                f"EquilibriumReactor '{self.unit_id}' inlet enthalpy was not calculated"
            )

        inlet_flows = inlet.component_flows()
        inlet_enthalpy_flow = inlet.F * inlet.H
        cache = {}

        def state_at_temperature(temperature: float):
            key = round(float(temperature), 10)
            if key not in cache:
                cache[key] = self._equilibrium_state_at_temperature(
                    inlet_flows,
                    reactions,
                    float(temperature),
                    P_out,
                    phase,
                )
            return cache[key]

        if mode == 'isothermal':
            if duty_spec is not None:
                raise UnitOperationError(
                    f"EquilibriumReactor '{self.unit_id}' isothermal mode "
                    "cannot also specify duty"
                )
            T_out = inlet.T if temperature_spec is None else temperature_spec
            equilibrium, outlet = state_at_temperature(T_out)
            Q = outlet.F * outlet.H - inlet_enthalpy_flow
        else:
            if temperature_spec is not None:
                raise UnitOperationError(
                    f"EquilibriumReactor '{self.unit_id}' {mode} mode cannot "
                    "also specify T/T_out"
                )
            if mode == 'adiabatic':
                if duty_spec is not None:
                    raise UnitOperationError(
                        f"EquilibriumReactor '{self.unit_id}' adiabatic mode "
                        "cannot specify duty"
                    )
                Q = 0.0
            else:
                if duty_spec is None:
                    raise UnitOperationError(
                        f"EquilibriumReactor '{self.unit_id}' duty mode requires "
                        "Q/duty/heat_duty"
                    )
                Q = duty_spec
            target = inlet_enthalpy_flow + Q
            equilibrium, outlet = self._equilibrium_state_for_enthalpy(
                state_at_temperature,
                target,
                inlet.T,
            )

        energy_residual = outlet.F * outlet.H - inlet_enthalpy_flow - Q
        energy_tolerance = max(
            1.0e-4,
            abs(inlet_enthalpy_flow) * 1.0e-9,
            abs(Q) * 1.0e-9,
        )
        if abs(energy_residual) > energy_tolerance:
            raise UnitOperationError(
                f"EquilibriumReactor '{self.unit_id}' energy balance residual "
                f"is {energy_residual:.6g} kJ/h"
            )

        stability = self._validate_equilibrium_phase_stability(outlet, phase)
        reaction_results = []
        for index, (specification, extent, log_k, log_q, residual, status) in enumerate(zip(
            reactions,
            equilibrium.extents_kmol_h,
            equilibrium.log_equilibrium_constants,
            equilibrium.log_reaction_quotients,
            equilibrium.residuals,
            equilibrium.reaction_statuses,
        ), start=1):
            reaction_result = {
                'index': index,
                'name': specification.name,
                'equation': specification.reaction.equation,
                'stoichiometry': dict(specification.reaction.stoichiometry),
                'extent_kmol_h': extent,
                'ln_equilibrium_constant': log_k,
                'ln_reaction_quotient': log_q,
                'ln_equilibrium_residual': residual,
                'equilibrium_status': status,
                'reaction_quotient_basis': (
                    '1e-14_mole_fraction_floor'
                    if status == 'boundary_limited' else 'interior_composition'
                ),
            }
            reaction_results.append(reaction_result)

        return UnitResult(
            outlet_streams={'out': outlet},
            heat_duty=Q,
            performance={
                'mode': mode,
                'phase': phase,
                'T_in_C': inlet.T - 273.15,
                'T_out_C': outlet.T - 273.15,
                'P_in_bar': inlet.P,
                'P_out_bar': outlet.P,
                'P_drop_bar': P_drop,
                'reactions': reaction_results,
                'equilibrium_iterations': equilibrium.iterations,
                'maximum_interior_ln_equilibrium_residual': (
                    equilibrium.max_interior_residual
                ),
                'maximum_raw_ln_equilibrium_residual': equilibrium.max_residual,
                'equilibrium_objective': equilibrium.objective,
                'component_conversions': self._equilibrium_component_conversions(
                    equilibrium
                ),
                'duty_kW': Q / 3600.0,
                'energy_balance_residual_kW': energy_residual / 3600.0,
                'phase_stability': stability,
            },
            warnings=list(equilibrium.warnings),
        )

    def _validate_equilibrium_thermochemistry(
        self,
        carried: set[str],
        participating: set[str],
        reactions,
    ) -> None:
        _validate_reactor_thermochemistry(
            self.thermo,
            f"EquilibriumReactor '{self.unit_id}'",
            carried,
            participating,
            reactions,
            reactions,
        )

    def _equilibrium_state_at_temperature(
        self,
        inlet_flows,
        reactions,
        T: float,
        P: float,
        phase: str,
    ):
        try:
            equilibrium = solve_homogeneous_equilibrium(
                inlet_flows,
                reactions,
                self.thermo,
                T,
                P,
                phase,
            )
        except Exception as error:
            if isinstance(error, UnitOperationError):
                raise
            raise UnitOperationError(
                f"EquilibriumReactor '{self.unit_id}' equilibrium solve failed "
                f"at {T:.4g} K and {P:.4g} bar: {error}"
            ) from error
        total = sum(equilibrium.outlet_component_flows.values())
        composition = {
            component: flow / total
            for component, flow in equilibrium.outlet_component_flows.items()
        }
        outlet = self.thermo.calculate_state(
            T,
            P,
            total,
            composition,
            phase=phase,
            flash=False,
            include=('H', 'Cp', 'S', 'rho'),
        )
        return equilibrium, outlet

    def _equilibrium_state_for_enthalpy(
        self,
        state_at_temperature,
        target_enthalpy_flow: float,
        T_guess: float,
    ):
        solver = _ThermoStateSolver(
            self.thermo,
            f"EquilibriumReactor '{self.unit_id}'",
        )

        def residual(T: float) -> float:
            _equilibrium, state = state_at_temperature(T)
            return state.F * state.H - target_enthalpy_flow

        temperature = solver._solve_temperature(
            residual,
            solver._temperature_grid(T_guess),
            'reactive outlet enthalpy',
            preferred_T=T_guess,
            residual_tolerance=max(1.0e-4, abs(target_enthalpy_flow) * 1.0e-9),
        )
        return state_at_temperature(temperature)

    def _validate_equilibrium_phase_stability(
        self,
        outlet: StreamState,
        phase: str,
    ) -> dict:
        return _validate_homogeneous_reactor_phase_stability(
            self.thermo,
            outlet,
            phase,
            f"EquilibriumReactor '{self.unit_id}'",
        )

    @staticmethod
    def _equilibrium_component_conversions(equilibrium) -> dict[str, float]:
        return {
            component: (
                (flow - equilibrium.outlet_component_flows.get(component, 0.0))
                / flow
            )
            for component, flow in equilibrium.inlet_component_flows.items()
            if flow > 0.0
        }


def _validate_reactor_thermochemistry(
    thermo,
    unit_label: str,
    carried: set[str],
    equilibrium_participating: set[str],
    all_reactions,
    equilibrium_reactions,
) -> None:
    """Validate enthalpy data and optional thermodynamic reverse-rate data."""
    all_reaction_components = {
        component
        for specification in all_reactions
        for component in specification.reaction.components
    }
    missing_hf = sorted(
        component for component in carried | all_reaction_components
        if thermo.props.get(component) is None
        or thermo.props[component].Hf is None
    )
    missing_gf = sorted(
        component for component in equilibrium_participating
        if thermo.props.get(component) is None
        or thermo.props[component].Gf is None
    )
    missing_s = sorted(
        component for component in equilibrium_participating
        if thermo.props.get(component) is None
        or thermo.props[component].S is None
    )
    messages = []
    if missing_hf:
        messages.append('Hf: ' + ', '.join(missing_hf))
    if missing_gf:
        messages.append('Gf: ' + ', '.join(missing_gf))
    if missing_s:
        messages.append('S: ' + ', '.join(missing_s))
    if messages:
        raise UnitOperationError(
            f"{unit_label} lacks required formation properties ("
            + '; '.join(messages) + ')'
        )

    for component in carried | all_reaction_components:
        thermo.mark_property_source_context_once(
            component,
            'Hf',
            phase='reactor_enthalpy_balance',
            description='Chemical-reactor inlet/outlet enthalpy balance',
        )

    for component in equilibrium_participating:
        try:
            thermo.Cp_ideal_gas(component, 298.15)
        except Exception as error:
            raise UnitOperationError(
                f"{unit_label} requires ideal-gas Cp for {component}: {error}"
            ) from error
        thermo.mark_property_source_context_once(
            component,
            'Gf',
            phase='reaction_gibbs_consistency_check',
            description='298.15 K reaction Gibbs-energy consistency check',
        )

    for specification in equilibrium_reactions:
        stoichiometry = specification.reaction.stoichiometry
        from_hs = thermo.reaction_standard_gibbs(stoichiometry, 298.15)
        from_gf = sum(
            coefficient * thermo.props[component].Gf * 1000.0
            for component, coefficient in stoichiometry.items()
        )
        difference = from_hs - from_gf
        tolerance = max(1000.0, 0.01 * abs(from_gf))
        if abs(difference) > tolerance:
            thermo.add_warning(
                f"Reaction '{specification.reaction.equation}' has inconsistent "
                f"298.15 K standard Gibbs energies: Hf/S route differs from "
                f"tabulated Gf route by {difference / 1000.0:.4g} kJ/mol reaction"
            )


def _validate_homogeneous_reactor_phase_stability(
    thermo,
    outlet: StreamState,
    phase: str,
    unit_label: str,
    *,
    check_global_lle: bool = True,
) -> dict:
    """Reject a forced homogeneous reactor outlet when phase splitting is stable."""
    flashed = thermo.calculate_state(
        outlet.T,
        outlet.P,
        outlet.F,
        outlet.composition,
        include=(),
    )
    vapor_fraction = flashed.fluid_vapor_fraction
    liquid2_fraction = float(flashed.liquid2_fraction or 0.0)
    tolerance = 1.0e-7
    if phase == 'vapor' and (
        vapor_fraction is None or vapor_fraction < 1.0 - tolerance
    ):
        raise UnitOperationError(
            f"{unit_label} homogeneous vapor outlet is phase-unstable; "
            f"unconstrained flash vapor fraction is {vapor_fraction}"
        )
    if phase == 'liquid' and (
        vapor_fraction is None or vapor_fraction > tolerance
    ):
        raise UnitOperationError(
            f"{unit_label} homogeneous liquid outlet is phase-unstable; "
            f"unconstrained flash vapor fraction is {vapor_fraction}"
        )
    if liquid2_fraction > tolerance:
        raise UnitOperationError(
            f"{unit_label} homogeneous outlet is liquid-liquid unstable; "
            f"second-liquid fraction is {liquid2_fraction:.6g}"
        )

    spinodal = None
    global_lle = None
    spinodal_method = getattr(thermo, 'liquid_spinodal_stability', None)
    if phase == 'liquid' and callable(spinodal_method):
        spinodal = spinodal_method(outlet.T, outlet.composition)
        if not spinodal.get('locally_stable', True):
            raise UnitOperationError(
                f"{unit_label} homogeneous liquid outlet is locally "
                "spinodal-unstable; minimum reduced Gibbs eigenvalue is "
                f"{spinodal.get('minimum_eigenvalue')}"
            )
    active_liquid = [
        component for component, fraction in outlet.composition.items()
        if fraction > 1.0e-10
    ]
    lle_method = getattr(thermo, 'liquid_liquid_equilibrium', None)
    if (
        check_global_lle
        and phase == 'liquid'
        and len(active_liquid) > 1
        and callable(lle_method)
    ):
        split, liquid1, liquid2, liquid1_fraction = lle_method(
            outlet.composition,
            outlet.T,
            max_iter=200,
            tol=1.0e-9,
        )
        global_lle = {
            'split': bool(split),
            'liquid1_fraction': float(liquid1_fraction),
            'liquid1_composition': dict(liquid1),
            'liquid2_composition': dict(liquid2),
        }
        if split and tolerance < liquid1_fraction < 1.0 - tolerance:
            raise UnitOperationError(
                f"{unit_label} homogeneous liquid outlet is globally "
                "liquid-liquid unstable"
            )
    return {
        'requested_phase': phase,
        'unconstrained_vapor_fraction': vapor_fraction,
        'unconstrained_liquid2_fraction': liquid2_fraction,
        'flash_status': flashed.phase_status,
        'flash_stability': flashed.phase_stability,
        'liquid_spinodal': spinodal,
        'global_liquid_liquid_stability': global_lle,
        'global_liquid_liquid_stability_checked': bool(
            check_global_lle and phase == 'liquid'
        ),
    }


def _validate_kinetic_rate_output_basis(
    reactions,
    expected_basis: str,
    unit_label: str,
) -> None:
    incompatible = [
        reaction.rate_unit for reaction in reactions
        if reaction.rate_output_basis != expected_basis
    ]
    if incompatible:
        expected = (
            'catalyst-mass'
            if expected_basis == 'catalyst_mass'
            else 'fluid-volume'
        )
        raise UnitOperationError(
            f"{unit_label} requires {expected} rate units; incompatible "
            "rate_unit(s): " + ', '.join(incompatible)
        )


def _find_cstr_thermal_roots(
    residual,
    temperature_min_K: float,
    temperature_max_K: float,
    scan_points: int,
    anchors=(),
) -> list[dict]:
    """Find and classify every sign-bracketed CSTR thermal steady state."""
    import numpy as np
    from scipy.optimize import brentq

    lower = float(temperature_min_K)
    upper = float(temperature_max_K)
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower <= 0.0
        or upper <= lower
    ):
        raise UnitOperationError(
            "CSTR thermal temperature bounds must be positive, finite, and ordered"
        )
    count = int(scan_points)
    if count < 21 or count > 2001:
        raise UnitOperationError(
            "CSTR thermal_scan_points must be between 21 and 2001"
        )
    grid = list(np.linspace(lower, upper, count))
    if lower > 0.0 and upper / lower > 2.0:
        grid.extend(np.geomspace(lower, upper, count))
    for anchor in anchors:
        if anchor is None or not math.isfinite(float(anchor)):
            continue
        value = float(anchor)
        for offset in (0.0, -20.0, -5.0, -1.0, 1.0, 5.0, 20.0):
            grid.append(max(lower, min(upper, value + offset)))
    grid = sorted(set(float(value) for value in grid))

    samples = []
    evaluated = []
    failures = []
    for temperature in grid:
        try:
            value = float(residual(temperature))
            if math.isfinite(value):
                evaluated.append((temperature, value))
                samples.append((temperature, value))
            else:
                samples.append((temperature, None))
        except Exception as error:
            failures.append((temperature, str(error)))
            samples.append((temperature, None))

    roots = []
    exact_tolerance = 1.0e-4
    for temperature, value in evaluated:
        if abs(value) <= exact_tolerance:
            roots.append(temperature)
    for (left_T, left_value), (right_T, right_value) in zip(
        samples,
        samples[1:],
    ):
        if left_value is None or right_value is None:
            continue
        if left_value * right_value >= 0.0:
            continue
        root = brentq(
            residual,
            left_T,
            right_T,
            xtol=1.0e-8,
            rtol=1.0e-10,
            maxiter=200,
        )
        roots.append(float(root))
    unique_roots = []
    for root in sorted(roots):
        if not unique_roots or abs(root - unique_roots[-1]) > 1.0e-5:
            unique_roots.append(root)

    results = []
    for root in unique_roots:
        step = max(1.0e-3, abs(root) * 1.0e-5)
        left = max(lower, root - step)
        right = min(upper, root + step)
        if right > left:
            slope = (residual(right) - residual(left)) / (right - left)
        else:
            slope = float('nan')
        results.append({
            'temperature_K': float(root),
            'temperature_C': float(root - 273.15),
            'energy_residual_kJ_h': float(residual(root)),
            'energy_residual_slope_kJ_h_K': float(slope),
            'thermally_stable': bool(math.isfinite(slope) and slope > 0.0),
        })
    if results:
        return results
    if evaluated:
        best_temperature, best_residual = min(
            evaluated,
            key=lambda item: abs(item[1]),
        )
        raise UnitOperationError(
            "CSTR could not bracket a thermal steady state within "
            f"{lower:g}-{upper:g} K; best residual {best_residual:.6g} kJ/h "
            f"at {best_temperature:.4g} K"
        )
    detail = failures[0][1] if failures else 'no evaluable thermal states'
    raise UnitOperationError(
        "CSTR could not evaluate any thermal steady state: " + detail
    )


class KineticsCSTR(Reactor):
    """Homogeneous CSTR with coupled kinetic and thermal balances."""

    @staticmethod
    def _normalized_unit(unit) -> str:
        return str(unit or '').strip().lower().replace(' ', '')

    def _ua_spec(self) -> float | None:
        ua_entries = [
            (name, self.get_param(name), self.get_param_unit(name))
            for name in ('UA', 'UA_available')
            if self.get_param(name) is not None
        ]
        u_entries = [
            (name, self.get_param(name), self.get_param_unit(name))
            for name in ('U', 'heat_transfer_coefficient')
            if self.get_param(name) is not None
        ]
        area_entries = [
            (name, self.get_param(name), self.get_param_unit(name))
            for name in ('A_heat', 'heat_transfer_area', 'area')
            if self.get_param(name) is not None
        ]
        if len(ua_entries) > 1 or len(u_entries) > 1 or len(area_entries) > 1:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' received duplicate heat-transfer aliases"
            )
        if ua_entries and (u_entries or area_entries):
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' may specify UA or U + A_heat, not both"
            )
        if ua_entries:
            name, raw, unit = ua_entries[0]
            value = float(raw)
            normalized = self._normalized_unit(unit)
            if normalized in {'kw/k', 'kw/c', 'kw/degc'}:
                value *= 1000.0
            elif normalized and normalized not in {'w/k', 'w/c', 'w/degc'}:
                raise UnitOperationError(
                    f"CSTR '{self.unit_id}' does not recognize {name} unit {unit!r}"
                )
            if not math.isfinite(value) or value <= 0.0:
                raise UnitOperationError(f"CSTR '{self.unit_id}' {name} must be positive")
            return value
        if not u_entries and not area_entries:
            return None
        if not u_entries or not area_entries:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' requires both U and A_heat when UA is omitted"
            )
        u_name, u_raw, u_unit = u_entries[0]
        area_name, area_raw, area_unit = area_entries[0]
        U = float(u_raw)
        area = float(area_raw)
        normalized_u = self._normalized_unit(u_unit)
        normalized_area = self._normalized_unit(area_unit)
        if normalized_u in {'kw/m2/k', 'kw/m^2/k', 'kw/m2-k'}:
            U *= 1000.0
        elif normalized_u and normalized_u not in {'w/m2/k', 'w/m^2/k', 'w/m2-k'}:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' does not recognize {u_name} unit {u_unit!r}"
            )
        if normalized_area in {'ft2', 'ft^2'}:
            area *= 0.09290304
        elif normalized_area and normalized_area not in {'m2', 'm^2'}:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' does not recognize {area_name} unit {area_unit!r}"
            )
        if not math.isfinite(U) or U <= 0.0 or not math.isfinite(area) or area <= 0.0:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' U and A_heat must be positive and finite"
            )
        return U * area

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .kinetic_models import (
                        KineticsError,
                        kinetic_reaction_from_mapping,
                        solve_isothermal_cstr,
                    )
        else:
            from kinetic_models import (
                        KineticsError,
                        kinetic_reaction_from_mapping,
                        solve_isothermal_cstr,
                    )

        if len(inlets) != 1:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' requires exactly one inlet stream"
            )
        inlet = next(iter(inlets.values()))
        raw_reactions = self.params.get('reactions', [])
        if not raw_reactions:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' requires at least one kinetic reaction"
            )
        if hasattr(self.thermo, '_initialize_vdm'):
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' does not support VDM thermodynamic methods"
            )
        phase = str(self.get_param('phase', '')).strip().lower()
        if phase == 'gas':
            phase = 'vapor'
        if phase not in {'vapor', 'liquid'}:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' requires explicit phase=vapor or phase=liquid"
            )
        volume = float(self.get_param('volume', self.get_param('V', 0.0)))
        volume_unit = self._normalized_unit(
            self.get_param_unit('volume') or self.get_param_unit('V')
        )
        if volume_unit in {'l', 'liter', 'litre'}:
            volume *= 1.0e-3
        elif volume_unit and volume_unit not in {'m3', 'm^3'}:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' does not recognize volume unit {volume_unit!r}"
            )
        if not math.isfinite(volume) or volume <= 0.0:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' volume must be positive and finite"
            )
        P_drop = float(self.get_param('P_drop', 0.0))
        if not math.isfinite(P_drop) or P_drop < 0.0 or P_drop >= inlet.P:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' P_drop must be nonnegative and below inlet pressure"
            )
        P_out = inlet.P - P_drop

        component_symbols = tuple(self.thermo.components)
        component_metadata = {
            component: self.thermo.props.get(component)
            for component in component_symbols
        }
        try:
            reactions = tuple(
                kinetic_reaction_from_mapping(
                    definition, component_symbols, component_metadata
                )
                for definition in raw_reactions
            )
        except ReactionDefinitionError as error:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' kinetic specification is invalid: {error}"
            ) from error
        _validate_kinetic_rate_output_basis(
            reactions, 'fluid_volume', f"CSTR '{self.unit_id}'"
        )
        reversible_reactions = tuple(
            reaction for reaction in reactions if reaction.reversible
        )
        reversible_components = {
            component
            for reaction in reversible_reactions
            for component in reaction.reaction.components
        }
        _validate_reactor_thermochemistry(
            self.thermo,
            f"CSTR '{self.unit_id}'",
            set(inlet.component_flows()),
            reversible_components,
            reactions,
            reversible_reactions,
        )
        if inlet.H is None:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' inlet enthalpy was not calculated"
            )

        mode = str(self.get_param('mode', 'isothermal')).strip().lower()
        if mode in {'specified_t', 'specified_temperature'}:
            mode = 'isothermal'
        if mode in {'specified_duty', 'heat_duty'}:
            mode = 'duty'
        if mode not in {'isothermal', 'adiabatic', 'duty', 'jacketed'}:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' mode must be isothermal, adiabatic, duty, or jacketed"
            )
        temperature_spec = self.get_temperature_param('T_out')
        if temperature_spec is None:
            temperature_spec = self.get_temperature_param('T')
        duty_spec = self._heat_duty_spec()
        UA = self._ua_spec()
        jacket_temperature = self.get_temperature_param('T_jacket')
        if jacket_temperature is None:
            jacket_temperature = self.get_temperature_param('T_coolant')
        if jacket_temperature is None:
            jacket_temperature = self.get_temperature_param('coolant_T')
        branch = self.get_param('thermal_branch')
        if branch is not None:
            branch = str(branch).strip().lower()
            if branch not in {'lowest', 'nearest_inlet', 'highest'}:
                raise UnitOperationError(
                    f"CSTR '{self.unit_id}' thermal_branch must be lowest, "
                    "nearest_inlet, or highest"
                )

        if mode == 'isothermal':
            if duty_spec is not None or UA is not None or jacket_temperature is not None:
                raise UnitOperationError(
                    f"CSTR '{self.unit_id}' isothermal mode cannot specify duty or jacket data"
                )
            if branch is not None:
                raise UnitOperationError(
                    f"CSTR '{self.unit_id}' isothermal mode does not use thermal_branch"
                )
            if temperature_spec is None:
                temperature_spec = inlet.T
        elif mode == 'adiabatic':
            if temperature_spec is not None or duty_spec is not None or UA is not None or jacket_temperature is not None:
                raise UnitOperationError(
                    f"CSTR '{self.unit_id}' adiabatic mode cannot specify T, duty, or jacket data"
                )
        elif mode == 'duty':
            if duty_spec is None or temperature_spec is not None or UA is not None or jacket_temperature is not None:
                raise UnitOperationError(
                    f"CSTR '{self.unit_id}' duty mode requires duty and cannot specify T or jacket data"
                )
        else:
            if temperature_spec is not None or duty_spec is not None or UA is None or jacket_temperature is None:
                raise UnitOperationError(
                    f"CSTR '{self.unit_id}' jacketed mode requires UA or U+A_heat "
                    "and T_jacket, and cannot specify T or duty"
                )

        cache = {}
        inlet_flows = inlet.component_flows()
        inlet_enthalpy_flow = inlet.F * inlet.H

        def state_at_temperature(temperature):
            key = round(float(temperature), 10)
            if key in cache:
                return cache[key]
            try:
                solution = solve_isothermal_cstr(
                    inlet_flows,
                    reactions,
                    self.thermo,
                    float(temperature),
                    P_out,
                    phase,
                    volume,
                )
            except (KineticsError, ThermodynamicsError) as error:
                raise UnitOperationError(
                    f"CSTR '{self.unit_id}' kinetic solve is invalid: {error}"
                ) from error
            total_out = sum(solution.outlet_component_flows.values())
            if total_out <= 0.0:
                raise UnitOperationError(
                    f"CSTR '{self.unit_id}' kinetic solution leaves no outlet flow"
                )
            composition = {
                component: flow / total_out
                for component, flow in solution.outlet_component_flows.items()
                if flow > 0.0
            }
            outlet = self.thermo.calculate_state(
                float(temperature), P_out, total_out, composition,
                phase=phase, flash=False,
            )
            if outlet.H is None:
                raise UnitOperationError(
                    f"CSTR '{self.unit_id}' outlet enthalpy was not calculated"
                )
            cache[key] = (solution, outlet)
            return cache[key]

        thermal_roots = []
        selected_branch = None
        if mode == 'isothermal':
            solution, outlet = state_at_temperature(temperature_spec)
            heat_duty = outlet.F * outlet.H - inlet_enthalpy_flow
        else:
            def heat_at_temperature(temperature):
                if mode == 'adiabatic':
                    return 0.0
                if mode == 'duty':
                    return duty_spec
                return UA * (jacket_temperature - temperature) * 3.6

            def energy_residual_at_temperature(temperature):
                _solution, trial_outlet = state_at_temperature(temperature)
                return (
                    trial_outlet.F * trial_outlet.H
                    - inlet_enthalpy_flow
                    - heat_at_temperature(temperature)
                )

            temperature_min = self.get_temperature_param('T_min', 200.0)
            temperature_max = self.get_temperature_param('T_max', 2500.0)
            scan_points = int(self.get_param('thermal_scan_points', 161))
            thermal_roots = _find_cstr_thermal_roots(
                energy_residual_at_temperature,
                temperature_min,
                temperature_max,
                scan_points,
                anchors=(inlet.T, jacket_temperature),
            )
            if len(thermal_roots) > 1 and branch is None:
                temperatures = ', '.join(
                    f"{root['temperature_K']:.6g} K"
                    for root in thermal_roots
                )
                raise UnitOperationError(
                    f"CSTR '{self.unit_id}' has multiple thermal steady states "
                    f"({temperatures}); specify thermal_branch=lowest, "
                    "nearest_inlet, or highest"
                )
            selected_branch = branch or 'single'
            if selected_branch == 'lowest' or selected_branch == 'single':
                selected = thermal_roots[0]
            elif selected_branch == 'highest':
                selected = thermal_roots[-1]
            else:
                selected = min(
                    thermal_roots,
                    key=lambda root: abs(root['temperature_K'] - inlet.T),
                )
            solution, outlet = state_at_temperature(selected['temperature_K'])
            heat_duty = heat_at_temperature(outlet.T)

        energy_residual = (
            outlet.F * outlet.H - inlet_enthalpy_flow - heat_duty
        )
        energy_tolerance = max(
            1.0e-4,
            abs(inlet_enthalpy_flow) * 1.0e-9,
            abs(heat_duty) * 1.0e-9,
        )
        if abs(energy_residual) > energy_tolerance:
            raise UnitOperationError(
                f"CSTR '{self.unit_id}' energy residual is {energy_residual:.6g} kJ/h"
            )
        total_out = outlet.F
        volumetric_flow = total_out / solution.rate_state.molar_density_kmol_m3
        residence_time = volume / volumetric_flow
        stability = _validate_homogeneous_reactor_phase_stability(
            self.thermo, outlet, phase, f"CSTR '{self.unit_id}'"
        )

        reactions_performance = []
        for index, (
            specification, extent, rate, residual, driving_residual, status,
        ) in enumerate(zip(
            reactions,
            solution.extents_kmol_h,
            solution.rates_kmol_m3_h,
            solution.residuals_kmol_h,
            solution.driving_residuals,
            solution.reaction_statuses,
        ), start=1):
            reactions_performance.append({
                'index': index,
                'name': specification.name,
                'equation': specification.reaction.equation,
                'stoichiometry': dict(specification.reaction.stoichiometry),
                'kinetic_type': specification.model,
                'reversible': specification.reversible,
                'rate_basis': specification.rate_basis,
                'rate_output_basis': specification.rate_output_basis,
                'concentration_unit': specification.concentration_unit,
                'pressure_unit': specification.pressure_unit,
                'declared_rate_unit': specification.rate_unit,
                'rate_kmol_m3_h': rate,
                'extent_kmol_h': extent,
                'material_rate_residual_kmol_h': residual,
                'thermodynamic_driving_residual': driving_residual,
                'kinetic_status': status,
            })
        warnings = [
            warning
            for specification in reactions
            for warning in specification.validation_warnings
        ]
        return UnitResult(
            outlet_streams={'out': outlet},
            heat_duty=heat_duty,
            performance={
                'mode': mode,
                'phase': phase,
                'volume_m3': volume,
                'residence_time_h': residence_time,
                'T_in_C': inlet.T - 273.15,
                'T_out_C': outlet.T - 273.15,
                'P_in_bar': inlet.P,
                'P_out_bar': outlet.P,
                'P_drop_bar': P_drop,
                'component_conversions': {
                    component: (
                        (flow - solution.outlet_component_flows.get(component, 0.0))
                        / flow
                    )
                    for component, flow in solution.inlet_component_flows.items()
                    if flow > 0.0
                },
                'reactions': reactions_performance,
                'solver_iterations': solution.iterations,
                'solver_objective': solution.objective,
                'maximum_material_rate_residual_kmol_h': solution.maximum_residual_kmol_h,
                'outlet_molar_density_kmol_m3': solution.rate_state.molar_density_kmol_m3,
                'thermal_roots': thermal_roots,
                'selected_thermal_branch': selected_branch,
                'UA_W_per_K': UA,
                'T_jacket_C': (
                    jacket_temperature - 273.15
                    if jacket_temperature is not None else None
                ),
                'duty_kW': heat_duty / 3600.0,
                'energy_balance_residual_kW': energy_residual / 3600.0,
                'phase_stability': stability,
            },
            warnings=list(dict.fromkeys(warnings)),
        )


class KineticsBatch(Reactor):
    """Transient homogeneous batch/semi-batch reactor on a continuous basis."""

    _VOLUME_FACTORS = {
        'm3': 1.0, 'm^3': 1.0, 'l': 1.0e-3,
        'liter': 1.0e-3, 'litre': 1.0e-3,
    }
    _TIME_FACTORS = {
        'h': 1.0, 'hr': 1.0, 'hour': 1.0, 'hours': 1.0,
        'min': 1.0 / 60.0, 'minute': 1.0 / 60.0,
        'minutes': 1.0 / 60.0,
        's': 1.0 / 3600.0, 'sec': 1.0 / 3600.0,
        'second': 1.0 / 3600.0, 'seconds': 1.0 / 3600.0,
    }

    @staticmethod
    def _normalized_unit(unit) -> str:
        return str(unit or '').strip().lower().replace(' ', '')

    def _one_dimension(self, names, factors, label, *, allow_zero=False):
        found = [
            (name, self.get_param(name), self.get_param_unit(name))
            for name in names if self.get_param(name) is not None
        ]
        if len(found) > 1:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' received duplicate aliases for {label}: "
                + ', '.join(name for name, _value, _unit in found)
            )
        if not found:
            return None
        name, raw, unit = found[0]
        value = float(raw)
        normalized = self._normalized_unit(unit)
        if normalized:
            factor = factors.get(normalized)
            if factor is None:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' does not recognize {label} "
                    f"unit {unit!r}"
                )
            value *= factor
        invalid = value < 0.0 if allow_zero else value <= 0.0
        if not math.isfinite(value) or invalid:
            qualifier = 'nonnegative' if allow_zero else 'positive'
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' {name} must be {qualifier} and finite"
            )
        return value

    @staticmethod
    def _name_list(value) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            entries = value
        else:
            entries = str(value).replace(';', ',').split(',')
        return {str(entry).strip() for entry in entries if str(entry).strip()}

    def _time_map(self, names, label) -> dict[str, float]:
        found = [(name, self.get_param(name)) for name in names if self.get_param(name) is not None]
        if len(found) > 1:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' received duplicate aliases for {label}"
            )
        if not found:
            return {}
        _name, raw = found[0]
        entries = raw if isinstance(raw, (list, tuple)) else str(raw).replace(';', ',').split(',')
        result = {}
        for entry in entries:
            text = str(entry).strip()
            if not text:
                continue
            if ':' not in text:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' {label} entries must be port:hours"
                )
            port, value = text.split(':', 1)
            port = port.strip()
            if not port or port in result:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' has an invalid or duplicate {label} port"
                )
            try:
                time_h = float(value)
            except ValueError as error:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' {label} for {port!r} must be hours"
                ) from error
            if not math.isfinite(time_h) or time_h < 0.0:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' {label} for {port!r} must be nonnegative"
                )
            result[port] = time_h
        return result

    def _ua_spec(self) -> float | None:
        ua_entries = [
            (name, self.get_param(name), self.get_param_unit(name))
            for name in ('UA', 'UA_available')
            if self.get_param(name) is not None
        ]
        u_entries = [
            (name, self.get_param(name), self.get_param_unit(name))
            for name in ('U', 'heat_transfer_coefficient')
            if self.get_param(name) is not None
        ]
        area_entries = [
            (name, self.get_param(name), self.get_param_unit(name))
            for name in ('A_heat', 'heat_transfer_area', 'area')
            if self.get_param(name) is not None
        ]
        if len(ua_entries) > 1 or len(u_entries) > 1 or len(area_entries) > 1:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' received duplicate heat-transfer aliases"
            )
        if ua_entries and (u_entries or area_entries):
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' may specify UA or U + A_heat, not both"
            )
        if ua_entries:
            name, raw, raw_unit = ua_entries[0]
            value = float(raw)
            unit = self._normalized_unit(raw_unit)
            if unit in {'kw/k', 'kw/c', 'kw/degc'}:
                value *= 1000.0
            elif unit and unit not in {'w/k', 'w/c', 'w/degc'}:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' does not recognize {name} unit "
                    f"{raw_unit!r}"
                )
            if not math.isfinite(value) or value <= 0.0:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' {name} must be positive and finite"
                )
            return value
        if not u_entries and not area_entries:
            return None
        if not u_entries or not area_entries:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' requires both U and A_heat"
            )
        u_name, u_raw, raw_u_unit = u_entries[0]
        area_name, area_raw, raw_area_unit = area_entries[0]
        coefficient = float(u_raw)
        surface = float(area_raw)
        u_unit = self._normalized_unit(raw_u_unit)
        area_unit = self._normalized_unit(raw_area_unit)
        if u_unit in {'kw/m2/k', 'kw/m^2/k', 'kw/m2-k'}:
            coefficient *= 1000.0
        elif u_unit and u_unit not in {'w/m2/k', 'w/m^2/k', 'w/m2-k'}:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' does not recognize {u_name} unit "
                f"{raw_u_unit!r}"
            )
        if area_unit in {'ft2', 'ft^2'}:
            surface *= 0.09290304
        elif area_unit and area_unit not in {'m2', 'm^2'}:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' does not recognize {area_name} unit "
                f"{raw_area_unit!r}"
            )
        if (
            not math.isfinite(coefficient) or coefficient <= 0.0
            or not math.isfinite(surface) or surface <= 0.0
        ):
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' U and A_heat must be positive and finite"
            )
        return coefficient * surface

    def _batch_energy(self) -> float | None:
        value = self.get_param('Q_batch')
        if value is None:
            value = self.get_param('batch_energy')
            name = 'batch_energy'
        else:
            name = 'Q_batch'
        if value is None:
            return None
        energy = float(value)
        unit = self._normalized_unit(self.get_param_unit(name))
        factors = {
            'kj': 1.0, 'j': 1.0e-3, 'mj': 1.0e3,
            'kwh': 3600.0, 'kw*h': 3600.0,
        }
        if unit:
            factor = factors.get(unit)
            if factor is None:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' does not recognize {name} unit {unit!r}"
                )
            energy *= factor
        if not math.isfinite(energy):
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' {name} must be finite"
            )
        return energy

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .batch_models import (
                        BatchFeed,
                        BatchNumericalControls,
                        BatchThermalModel,
                        solve_adaptive_batch,
                    )
        else:
            from batch_models import (
                        BatchFeed,
                        BatchNumericalControls,
                        BatchThermalModel,
                        solve_adaptive_batch,
                    )
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .kinetic_models import KineticsError, kinetic_reaction_from_mapping
        else:
            from kinetic_models import KineticsError, kinetic_reaction_from_mapping

        if not inlets:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' requires at least one inlet stream"
            )
        if hasattr(self.thermo, '_initialize_vdm'):
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' does not support VDM thermodynamic methods"
            )
        if any(inlet.H is None for inlet in inlets.values()):
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' requires inlet enthalpies"
            )
        raw_reactions = self.params.get('reactions', [])
        if not raw_reactions:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' requires at least one kinetic reaction"
            )
        phase = str(self.get_param('phase', '')).strip().lower()
        if phase == 'gas':
            phase = 'vapor'
        if phase not in {'vapor', 'liquid'}:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' requires phase=vapor or phase=liquid"
            )

        component_symbols = tuple(self.thermo.components)
        component_metadata = {
            component: self.thermo.props.get(component)
            for component in component_symbols
        }
        try:
            reactions = tuple(
                kinetic_reaction_from_mapping(
                    definition, component_symbols, component_metadata
                )
                for definition in raw_reactions
            )
        except ReactionDefinitionError as error:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' kinetic specification is invalid: {error}"
            ) from error
        _validate_kinetic_rate_output_basis(
            reactions, 'fluid_volume', f"BatchReactor '{self.unit_id}'"
        )
        reversible = tuple(reaction for reaction in reactions if reaction.reversible)
        reversible_components = {
            component for reaction in reversible
            for component in reaction.reaction.components
        }
        carried = {
            component for inlet in inlets.values()
            for component in inlet.component_flows()
        }
        _validate_reactor_thermochemistry(
            self.thermo,
            f"BatchReactor '{self.unit_id}'",
            carried,
            reversible_components,
            reactions,
            reversible,
        )

        schedule_aliases = [
            name for name in ('semi_batch_feeds', 'scheduled_feeds')
            if self.get_param(name) is not None
        ]
        if len(schedule_aliases) > 1:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' received duplicate scheduled-feed aliases"
            )
        scheduled_ports = self._name_list(
            self.get_param(schedule_aliases[0]) if schedule_aliases else None
        )
        feed_starts = self._time_map(
            ('feed_start_times', 'feed_starts'), 'feed start times'
        )
        feed_stops = self._time_map(
            ('feed_stop_times', 'feed_stops'), 'feed stop times'
        )
        scheduled_ports.update(feed_starts)
        scheduled_ports.update(feed_stops)
        unknown_ports = scheduled_ports.difference(inlets)
        if unknown_ports:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' schedules unknown inlet port(s): "
                + ', '.join(sorted(unknown_ports))
            )
        if len(scheduled_ports) == len(inlets):
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' requires at least one unscheduled "
                "initial-charge inlet"
            )

        volume = self._one_dimension(
            ('V_batch', 'batch_volume'), self._VOLUME_FACTORS, 'batch volume'
        )
        reaction_time = self._one_dimension(
            ('t_rxn', 't_reaction', 'reaction_time'),
            self._TIME_FACTORS,
            'reaction time',
        )
        fill_time_spec = self._one_dimension(
            ('t_fill', 'fill_time'), self._TIME_FACTORS, 'fill time'
        )
        drain_time_spec = self._one_dimension(
            ('t_drain', 'drain_time'), self._TIME_FACTORS, 'drain time'
        )
        turnaround = self._one_dimension(
            ('t_turnaround', 'turnaround_time'), self._TIME_FACTORS,
            'turnaround time', allow_zero=True,
        )
        turnaround = 0.0 if turnaround is None else turnaround
        vessel_count_entries = [
            (name, self.get_param(name))
            for name in ('N', 'n_vessels', 'n_tanks')
            if self.get_param(name) is not None
        ]
        if len(vessel_count_entries) > 1:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' received duplicate vessel-count aliases"
            )
        vessel_count_value = (
            vessel_count_entries[0][1] if vessel_count_entries else None
        )
        vessel_count = None
        if vessel_count_value is not None:
            raw_count = float(vessel_count_value)
            vessel_count = int(raw_count)
            if (
                not math.isfinite(raw_count) or raw_count < 1.0
                or not math.isclose(raw_count, vessel_count, abs_tol=1.0e-12)
            ):
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' N must be a positive integer"
                )
        if sum(value is not None for value in (volume, reaction_time, vessel_count)) != 2:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' requires exactly two of "
                "V_batch, N, and t_rxn"
            )

        target_pressure = self.get_param('P')
        pressure_mode = str(
            self.get_param('pressure_mode', 'constant_pressure')
        ).strip().lower().replace('-', '_')
        pressure_mode = {
            'constant': 'constant_pressure',
            'isobaric': 'constant_pressure',
        }.get(pressure_mode, pressure_mode)
        if pressure_mode != 'constant_pressure':
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' currently supports only "
                "pressure_mode=constant_pressure"
            )
        pressure = min(inlet.P for inlet in inlets.values()) if target_pressure is None else float(target_pressure)
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' pressure must be positive and finite"
            )
        if any(pressure > inlet.P + 1.0e-9 for inlet in inlets.values()):
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' pressure exceeds an inlet pressure; "
                "compress that feed upstream"
            )

        volumetric_flows = {}
        total_volumetric_flow = 0.0
        for port, inlet in inlets.items():
            composition = inlet.composition
            density = float(self.thermo.mixture_molar_density(
                composition, inlet.T, inlet.P,
                1.0 if phase == 'vapor' else 0.0,
                x=composition if phase == 'liquid' else None,
                y=composition if phase == 'vapor' else None,
            ))
            if not math.isfinite(density) or density <= 0.0:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' could not determine feed density for {port!r}"
                )
            volumetric_flows[port] = inlet.F / density
            total_volumetric_flow += volumetric_flows[port]
        if total_volumetric_flow <= 0.0:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' requires positive inlet flow"
            )

        default_transfer_count = int(fill_time_spec is None) + int(drain_time_spec is None)
        fixed_transfer_time = sum(
            value for value in (fill_time_spec, drain_time_spec) if value is not None
        )
        if volume is not None:
            start_interval = volume / total_volumetric_flow
            fill_time = start_interval if fill_time_spec is None else fill_time_spec
            drain_time = start_interval if drain_time_spec is None else drain_time_spec
            if reaction_time is not None:
                cycle_time = fill_time + reaction_time + drain_time + turnaround
                vessel_count = max(1, int(math.ceil(cycle_time / start_interval - 1.0e-12)))
            else:
                reaction_time = (
                    vessel_count * start_interval - fill_time - drain_time - turnaround
                )
                if reaction_time <= 0.0:
                    raise UnitOperationError(
                        f"BatchReactor '{self.unit_id}' N and V_batch leave no positive reaction time"
                    )
                cycle_time = fill_time + reaction_time + drain_time + turnaround
        else:
            denominator = vessel_count - default_transfer_count
            if denominator <= 0:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' N must exceed the number of "
                    "fill/drain times derived from the batch interval"
                )
            start_interval = (
                fixed_transfer_time + reaction_time + turnaround
            ) / denominator
            volume = total_volumetric_flow * start_interval
            fill_time = start_interval if fill_time_spec is None else fill_time_spec
            drain_time = start_interval if drain_time_spec is None else drain_time_spec
            cycle_time = fill_time + reaction_time + drain_time + turnaround

        batch_frequency = 1.0 / start_interval
        utilization = cycle_time / (vessel_count * start_interval)
        if utilization > 1.0 + 1.0e-10:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' vessel fleet is undersized; "
                f"required utilization is {utilization:.6g}"
            )

        batch_feeds = []
        for port, inlet in inlets.items():
            amounts = {
                component: flow / batch_frequency
                for component, flow in inlet.component_flows().items()
                if flow > 0.0
            }
            batch_feeds.append(BatchFeed(
                port,
                amounts,
                float(inlet.H),
                feed_starts.get(port, 0.0) if port in scheduled_ports else None,
                feed_stops.get(port, reaction_time) if port in scheduled_ports else None,
            ))

        mode = str(self.get_param('mode', 'isothermal')).strip().lower()
        if mode in {'specified_t', 'specified_temperature'}:
            mode = 'isothermal'
        if mode in {'specified_duty', 'heat_duty'}:
            mode = 'duty'
        temperature = self.get_temperature_param('T_out')
        if temperature is None:
            temperature = self.get_temperature_param('T')
        batch_energy = self._batch_energy()
        UA = self._ua_spec()
        jacket_temperature = self.get_temperature_param('T_jacket')
        if jacket_temperature is None:
            jacket_temperature = self.get_temperature_param('T_coolant')
        if jacket_temperature is None:
            jacket_temperature = self.get_temperature_param('coolant_T')
        if mode == 'isothermal':
            if batch_energy is not None or UA is not None or jacket_temperature is not None:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' isothermal mode cannot specify batch energy or jacket data"
                )
            if temperature is None:
                initial_ports = [port for port in inlets if port not in scheduled_ports]
                total_initial_flow = sum(inlets[port].F for port in initial_ports)
                temperature = sum(
                    inlets[port].F * inlets[port].T for port in initial_ports
                ) / total_initial_flow
        elif mode == 'adiabatic':
            if temperature is not None or batch_energy is not None or UA is not None or jacket_temperature is not None:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' adiabatic mode cannot specify T, Q_batch, or jacket data"
                )
        elif mode == 'duty':
            if batch_energy is None or temperature is not None or UA is not None or jacket_temperature is not None:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' duty mode requires Q_batch and cannot specify T or jacket data"
                )
        elif mode == 'jacketed':
            if temperature is not None or batch_energy is not None or UA is None or jacket_temperature is None:
                raise UnitOperationError(
                    f"BatchReactor '{self.unit_id}' jacketed mode requires UA and T_jacket"
                )
        else:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' mode must be isothermal, adiabatic, duty, or jacketed"
            )
        thermal = BatchThermalModel(mode, temperature, batch_energy, UA, jacket_temperature)

        vessel_volume = self._one_dimension(
            ('V_vessel', 'vessel_volume'), self._VOLUME_FACTORS, 'vessel volume'
        )
        fill_fraction = float(self.get_param('fill_fraction', 1.0))
        if not math.isfinite(fill_fraction) or not 0.0 < fill_fraction <= 1.0:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' fill_fraction must lie in (0, 1]"
            )
        usable_vessel_volume = (
            vessel_volume * fill_fraction if vessel_volume is not None else None
        )

        requested_profile_points = int(self.get_param('profile_points', 31))
        context = dict(getattr(self, 'solve_context', {}) or {})
        recycle_iteration = (
            context.get('recycle_evaluation') is not None
            and context.get('recycle_final_pass') is not True
        )
        detailed_diagnostics = (
            not recycle_iteration or context.get('expensive_diagnostics') is True
        )
        controls = BatchNumericalControls(
            str(self.get_param('solver', self.get_param('integration_method', 'auto'))),
            float(self.get_param('relative_tolerance', self.get_param('rtol', 1.0e-7))),
            float(self.get_param('absolute_tolerance', self.get_param('atol', 1.0e-9))),
            self._one_dimension(
                ('maximum_step', 'max_step'), self._TIME_FACTORS,
                'maximum time step'
            ),
            self._one_dimension(
                ('first_step',), self._TIME_FACTORS, 'first time step'
            ),
            requested_profile_points if detailed_diagnostics else 2,
        )
        try:
            solution = solve_adaptive_batch(
                tuple(batch_feeds), reactions, self.thermo, phase, pressure,
                reaction_time, thermal, controls, usable_vessel_volume,
            )
        except (KineticsError, ThermodynamicsError) as error:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' adaptive solve failed: {error}"
            ) from error

        terminal = solution.outlet_state
        outlet_flows = {
            component: amount * batch_frequency
            for component, amount in solution.outlet_component_amounts_kmol.items()
        }
        total_out = sum(outlet_flows.values())
        outlet_composition = {
            component: flow / total_out for component, flow in outlet_flows.items()
        }
        outlet = self.thermo.calculate_state(
            terminal.T, pressure, total_out, outlet_composition,
            phase=phase, flash=False,
        )
        if detailed_diagnostics:
            stability_profile = [
                _validate_homogeneous_reactor_phase_stability(
                    self.thermo, state, phase,
                    f"BatchReactor '{self.unit_id}' profile point {index}",
                    check_global_lle=False,
                )
                for index, state in enumerate(solution.profile_states)
            ]
            global_phase_stability_checks = {}
            global_indices = (
                sorted(set((
                    0,
                    len(solution.profile_states) // 2,
                    len(solution.profile_states) - 1,
                )))
                if phase == 'liquid' else [len(solution.profile_states) - 1]
            )
            for index in global_indices:
                global_phase_stability_checks[index] = (
                    _validate_homogeneous_reactor_phase_stability(
                        self.thermo,
                        solution.profile_states[index],
                        phase,
                        f"BatchReactor '{self.unit_id}' profile point {index}",
                        check_global_lle=True,
                    )
                )
        else:
            stability_profile = []
            global_phase_stability_checks = {}
        terminal_stability = _validate_homogeneous_reactor_phase_stability(
            self.thermo, outlet, phase, f"BatchReactor '{self.unit_id}' outlet"
        )
        global_phase_stability_checks[len(solution.profile_states) - 1] = (
            terminal_stability
        )
        if not stability_profile:
            stability_profile = [terminal_stability]

        maximum_material_residual = max(
            (abs(value) for value in solution.material_balance_residuals_kmol_batch.values()),
            default=0.0,
        )
        batch_input_amount = sum(inlet.F for inlet in inlets.values()) / batch_frequency
        if maximum_material_residual > max(1.0e-10, batch_input_amount * 1.0e-8):
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' material residual is "
                f"{maximum_material_residual:.6g} kmol/batch"
            )
        energy_tolerance = max(
            1.0e-4,
            abs(solution.heat_per_batch_kJ) * 1.0e-9,
            abs(sum(feed.F * feed.H for feed in inlets.values()) / batch_frequency) * 1.0e-9,
        )
        if abs(solution.energy_balance_residual_kJ_batch) > energy_tolerance:
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' energy residual is "
                f"{solution.energy_balance_residual_kJ_batch:.6g} kJ/batch"
            )

        if (
            vessel_volume is not None
            and solution.maximum_working_volume_m3 > vessel_volume * fill_fraction * (1.0 + 1.0e-8)
        ):
            raise UnitOperationError(
                f"BatchReactor '{self.unit_id}' trajectory exceeds usable vessel volume"
            )

        heat_duty = solution.heat_per_batch_kJ * batch_frequency
        input_flows = {
            component: sum(
                inlet.component_flows().get(component, 0.0)
                for inlet in inlets.values()
            )
            for component in component_symbols
        }
        reaction_results = []
        for index, (reaction, extent) in enumerate(zip(
            reactions, solution.reaction_extents_kmol_batch
        ), start=1):
            reaction_results.append({
                'index': index,
                'name': reaction.name,
                'equation': reaction.reaction.equation,
                'stoichiometry': dict(reaction.reaction.stoichiometry),
                'kinetic_type': reaction.model,
                'reversible': reaction.reversible,
                'rate_basis': reaction.rate_basis,
                'rate_output_basis': reaction.rate_output_basis,
                'declared_rate_unit': reaction.rate_unit,
                'extent_kmol_batch': extent,
                'average_extent_kmol_h': extent * batch_frequency,
                'terminal_rate_kmol_m3_h': (
                    solution.profile[-1]['reaction_rates_kmol_m3_h'][index - 1]
                ),
            })
        warnings = [
            warning for reaction in reactions for warning in reaction.validation_warnings
        ]
        discharge_coverage = min(1.0, drain_time / start_interval)
        buffering_required = drain_time + 1.0e-12 < start_interval
        if buffering_required:
            warnings.append(
                "BatchReactor discharge has gaps; the continuous-equivalent outlet "
                "requires a downstream surge buffer"
            )
        performance = {
            'mode': mode,
            'phase': phase,
            'pressure_mode': 'constant_pressure',
            'outlet_basis': 'time_averaged',
            'V_batch_m3': volume,
            'V_vessel_m3': vessel_volume,
            'fill_fraction': fill_fraction,
            'N_vessels': vessel_count,
            'batch_frequency_per_h': batch_frequency,
            'batch_start_interval_h': start_interval,
            'fill_time_h': fill_time,
            'reaction_time_h': reaction_time,
            'drain_time_h': drain_time,
            'turnaround_time_h': turnaround,
            'cycle_time_h': cycle_time,
            'fleet_utilization': utilization,
            'continuous_discharge_coverage': discharge_coverage,
            'surge_buffer_required': buffering_required,
            'semi_batch_feed_ports': sorted(scheduled_ports),
            'maximum_working_volume_m3': solution.maximum_working_volume_m3,
            'initial_charge_T_C': solution.profile[0]['temperature_C'],
            'heat_per_batch_kJ': solution.heat_per_batch_kJ,
            'specified_batch_energy_kJ': batch_energy,
            'UA_W_per_K': UA,
            'T_jacket_C': (
                jacket_temperature - 273.15
                if jacket_temperature is not None else None
            ),
            'average_heat_duty_kW': heat_duty / 3600.0,
            'T_out_C': outlet.T - 273.15,
            'P_out_bar': outlet.P,
            'component_conversions': {
                component: (flow - outlet_flows.get(component, 0.0)) / flow
                for component, flow in input_flows.items() if flow > 0.0
            },
            'reactions': reaction_results,
            'solver_method': solution.solver_method,
            'attempted_solver_methods': list(solution.attempted_methods),
            'solver_evaluations': solution.evaluations,
            'solver_status': solution.status,
            'solver_message': solution.message,
            'profile_points': len(solution.profile),
            'requested_profile_points': requested_profile_points,
            'detailed_diagnostics': detailed_diagnostics,
            'profile': list(solution.profile),
            'maximum_material_balance_residual_kmol_batch': maximum_material_residual,
            'energy_balance_residual_kJ_batch': solution.energy_balance_residual_kJ_batch,
            'phase_stability': terminal_stability,
            'phase_stability_profile': stability_profile,
            'global_phase_stability_profile_checks': global_phase_stability_checks,
        }
        return UnitResult(
            outlet_streams={'out': outlet},
            heat_duty=heat_duty,
            performance=performance,
            warnings=list(dict.fromkeys(warnings)),
        )


class KineticsPFR(Reactor):
    """Adaptive homogeneous plug-flow reactor."""

    _reactor_name = 'PFR'
    _rate_output_basis = 'fluid_volume'

    _LENGTH_FACTORS = {
        'm': 1.0, 'meter': 1.0, 'meters': 1.0,
        'cm': 1.0e-2, 'mm': 1.0e-3, 'km': 1.0e3,
        'ft': 0.3048, 'in': 0.0254,
    }
    _VOLUME_FACTORS = {
        'm3': 1.0, 'm^3': 1.0, 'l': 1.0e-3,
        'liter': 1.0e-3, 'litre': 1.0e-3,
    }

    @staticmethod
    def _normalized_unit(unit) -> str:
        return str(unit or '').strip().lower().replace(' ', '')

    def _dimension(self, names, factors, label, *, allow_zero=False):
        found = []
        for name in names:
            value = self.get_param(name)
            if value is not None:
                found.append((name, value, self.get_param_unit(name)))
        if len(found) > 1:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' received duplicate aliases for {label}: "
                + ', '.join(item[0] for item in found)
            )
        if not found:
            return None
        name, raw, unit = found[0]
        value = float(raw)
        normalized = self._normalized_unit(unit)
        if normalized:
            factor = factors.get(normalized)
            if factor is None:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' does not recognize {label} unit {unit!r}"
                )
            value *= factor
        invalid = value < 0.0 if allow_zero else value <= 0.0
        if not math.isfinite(value) or invalid:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' requires "
                f"{'nonnegative' if allow_zero else 'positive'} finite {name}"
            )
        return value

    def _geometry(self):
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .pfr_models import PFRGeometry
        else:
            from pfr_models import PFRGeometry

        volume = self._dimension(('volume', 'V'), self._VOLUME_FACTORS, 'volume')
        length = self._dimension(('length', 'L'), self._LENGTH_FACTORS, 'length')
        diameter = self._dimension(('diameter', 'D'), self._LENGTH_FACTORS, 'diameter')
        n_tubes = int(self.get_param('n_tubes', 1))
        if n_tubes < 1:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' n_tubes must be positive"
            )
        supplied = sum(value is not None for value in (volume, length, diameter))
        if supplied < 1 or (volume is None and supplied < 2):
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' requires volume or length plus diameter"
            )
        if volume is None:
            volume = n_tubes * math.pi * diameter**2 / 4.0 * length
        elif length is None and diameter is not None:
            length = volume / (n_tubes * math.pi * diameter**2 / 4.0)
        elif diameter is None and length is not None:
            diameter = math.sqrt(4.0 * volume / (n_tubes * math.pi * length))
        elif length is not None and diameter is not None:
            geometric = n_tubes * math.pi * diameter**2 / 4.0 * length
            if abs(geometric / volume - 1.0) > 1.0e-6:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' volume, length, diameter, and n_tubes "
                    f"are inconsistent; geometric volume is {geometric:.8g} m3"
                )
        return PFRGeometry(volume, length, diameter, n_tubes)

    def _heat_transfer_coefficient(self):
        value = self.get_param('U')
        if value is None:
            value = self.get_param('heat_transfer_coefficient')
            name = 'heat_transfer_coefficient'
        else:
            name = 'U'
        if value is None:
            return None
        coefficient = float(value)
        unit = self._normalized_unit(self.get_param_unit(name))
        if unit in {'kw/m2/k', 'kw/m^2/k', 'kw/m2-k'}:
            coefficient *= 1000.0
        elif unit and unit not in {'w/m2/k', 'w/m^2/k', 'w/m2-k'}:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' does not recognize U unit "
                f"{self.get_param_unit(name)!r}"
            )
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' U must be positive"
            )
        return coefficient

    def _adaptive_reaction_configuration(
        self,
        geometry,
        reactions,
        pressure_spec,
        inlet,
        thermal,
        phase,
    ) -> dict:
        return {}

    def _solve_adaptive_model(
        self,
        solver,
        inlet,
        reactions,
        phase,
        geometry,
        thermal,
        pressure_spec,
        controls,
        adaptive_configuration,
    ):
        return solver(
            inlet,
            reactions,
            self.thermo,
            phase,
            geometry,
            thermal,
            pressure_spec,
            controls,
            **adaptive_configuration,
        )

    def _reaction_result_rate_fields(self, solution, index: int) -> dict:
        return {
            'outlet_rate_kmol_m3_h': (
                solution.profile[-1]['reaction_rates_kmol_m3_h'][index - 1]
            ),
        }

    def _extend_performance(self, performance: dict, solution) -> None:
        return None

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .kinetic_models import KineticsError, kinetic_reaction_from_mapping
        else:
            from kinetic_models import KineticsError, kinetic_reaction_from_mapping
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .pfr_models import (
                        PFRNumericalControls,
                        PFRPressureModel,
                        PFRThermalModel,
                        solve_adaptive_pfr,
                    )
        else:
            from pfr_models import (
                        PFRNumericalControls,
                        PFRPressureModel,
                        PFRThermalModel,
                        solve_adaptive_pfr,
                    )

        if len(inlets) != 1:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' requires exactly one inlet stream"
            )
        inlet = next(iter(inlets.values()))
        raw_reactions = self.params.get('reactions', [])
        if not raw_reactions:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' requires at least one kinetic reaction"
            )
        phase = str(self.get_param('phase', '')).strip().lower()
        if phase == 'gas':
            phase = 'vapor'
        if phase not in {'vapor', 'liquid'}:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' requires explicit phase=vapor or phase=liquid"
            )

        component_symbols = tuple(self.thermo.components)
        component_metadata = {
            component: self.thermo.props.get(component)
            for component in component_symbols
        }
        try:
            reactions = tuple(
                kinetic_reaction_from_mapping(
                    definition, component_symbols, component_metadata
                )
                for definition in raw_reactions
            )
        except ReactionDefinitionError as error:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' kinetic specification is invalid: {error}"
            ) from error
        _validate_kinetic_rate_output_basis(
            reactions,
            self._rate_output_basis,
            f"{self._reactor_name} '{self.unit_id}'",
        )

        reversible = tuple(reaction for reaction in reactions if reaction.reversible)
        reversible_components = {
            component
            for reaction in reversible
            for component in reaction.reaction.components
        }
        _validate_reactor_thermochemistry(
            self.thermo,
            f"{self._reactor_name} '{self.unit_id}'",
            set(inlet.component_flows()),
            reversible_components,
            reactions,
            reversible,
        )
        if inlet.H is None:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' inlet enthalpy is missing"
            )

        geometry = self._geometry()
        mode = str(self.get_param('mode', 'isothermal')).strip().lower()
        if mode in {'specified_t', 'specified_temperature'}:
            mode = 'isothermal'
        if mode in {'specified_duty', 'heat_duty'}:
            mode = 'duty'
        duty = self._heat_duty_spec()
        temperature = self.get_temperature_param('T_out')
        if temperature is None:
            temperature = self.get_temperature_param('T')
        if mode == 'isothermal':
            if duty is not None:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' isothermal mode cannot specify duty"
                )
            if temperature is None:
                temperature = inlet.T
        elif mode == 'adiabatic':
            if duty is not None or temperature is not None:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' adiabatic mode cannot specify T or duty"
                )
        elif mode == 'duty':
            if duty is None or temperature is not None:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' duty mode requires Q and cannot specify T"
                )
        elif mode == 'jacketed':
            if duty is not None or temperature is not None:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' jacketed mode cannot specify T or duty"
                )
        else:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' mode must be isothermal, adiabatic, duty, or jacketed"
            )
        jacket_temperature = self.get_temperature_param('T_jacket')
        if jacket_temperature is None:
            jacket_temperature = self.get_temperature_param('T_coolant')
        if jacket_temperature is None:
            jacket_temperature = self.get_temperature_param('coolant_T')
        U = self._heat_transfer_coefficient()
        if mode == 'jacketed' and (U is None or jacket_temperature is None):
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' jacketed mode requires U and T_jacket"
            )
        thermal = PFRThermalModel(mode, temperature, duty, U, jacket_temperature)

        pressure_model = str(self.get_param(
            'pressure_drop_model', self.get_param('P_drop_model', 'none')
        )).strip().lower().replace('-', '_')
        pressure_model = {
            'pipe': 'darcy',
            'unpacked': 'darcy',
            'darcy_weisbach': 'darcy',
            'packed': 'ergun',
            'packed_bed': 'ergun',
            'linear': 'specified',
        }.get(pressure_model, pressure_model)
        P_drop = float(self.get_param('P_drop', 0.0))
        if P_drop < 0.0 or not math.isfinite(P_drop):
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' P_drop must be nonnegative"
            )
        if pressure_model == 'none' and P_drop > 0.0:
            pressure_model = 'specified'
        if pressure_model == 'specified' and P_drop <= 0.0:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' specified pressure drop requires P_drop > 0"
            )
        if P_drop >= inlet.P:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' specified pressure drop exhausts inlet pressure"
            )
        void_fraction = self.get_param('void_fraction')
        particle_diameter = self._dimension(
            ('particle_diameter', 'catalyst_particle_diameter', 'd_p'),
            self._LENGTH_FACTORS,
            'particle diameter',
        )
        if pressure_model == 'ergun':
            if void_fraction is None or particle_diameter is None:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' Ergun pressure drop requires "
                    "void_fraction and particle_diameter"
                )
            void_fraction = float(void_fraction)
            if not 0.0 < void_fraction < 1.0:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' void_fraction must lie between 0 and 1"
                )
        roughness = self._dimension(
            ('roughness', 'roughness_m'), self._LENGTH_FACTORS, 'roughness',
            allow_zero=True,
        )
        pressure_spec = PFRPressureModel(
            pressure_model,
            P_drop,
            float(void_fraction) if void_fraction is not None else None,
            particle_diameter,
            roughness or 0.0,
            str(self.get_param('friction_model', 'churchill')),
        )

        profile_value = self.get_param('profile_points')
        warnings = []
        if profile_value is None and self.get_param('n_segments') is not None:
            profile_value = int(self.get_param('n_segments')) + 1
            warnings.append(
                f"{self._reactor_name} n_segments is deprecated and now controls only profile_points; "
                "adaptive solver accuracy uses relative/absolute tolerances"
            )
        requested_profile_points = int(profile_value or 31)
        solve_context = dict(getattr(self, 'solve_context', {}) or {})
        recycle_iteration = (
            solve_context.get('recycle_evaluation') is not None
            and solve_context.get('recycle_final_pass') is not True
        )
        detailed_diagnostics = (
            not recycle_iteration
            or solve_context.get('expensive_diagnostics') is True
        )
        profile_points = requested_profile_points if detailed_diagnostics else 2
        coordinate_factors = (
            self._LENGTH_FACTORS
            if geometry.uses_length_coordinate else self._VOLUME_FACTORS
        )
        coordinate_label = (
            'maximum axial step' if geometry.uses_length_coordinate
            else 'maximum reactor-volume step'
        )
        maximum_step = self._dimension(
            ('maximum_step', 'max_step'),
            coordinate_factors,
            coordinate_label,
        )
        first_step = self._dimension(
            ('first_step',),
            coordinate_factors,
            'first integration step',
        )
        controls = PFRNumericalControls(
            str(self.get_param('solver', self.get_param('integration_method', 'auto'))),
            float(self.get_param('relative_tolerance', self.get_param('rtol', 1.0e-7))),
            float(self.get_param('absolute_tolerance', self.get_param('atol', 1.0e-9))),
            maximum_step,
            first_step,
            profile_points,
        )
        try:
            adaptive_configuration = self._adaptive_reaction_configuration(
                geometry,
                reactions,
                pressure_spec,
                inlet,
                thermal,
                phase,
            )
            solution = self._solve_adaptive_model(
                solve_adaptive_pfr,
                inlet,
                reactions,
                phase,
                geometry,
                thermal,
                pressure_spec,
                controls,
                adaptive_configuration,
            )
        except (KineticsError, ThermodynamicsError) as error:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' adaptive solve failed: {error}"
            ) from error
        warnings.extend(solution.warnings)

        outlet = solution.outlet_state
        outlet_profile_index = len(solution.profile_states) - 1
        if detailed_diagnostics:
            stability_profile = []
            for profile_index, profile_state in enumerate(solution.profile_states):
                stability_profile.append(
                    _validate_homogeneous_reactor_phase_stability(
                        self.thermo,
                        profile_state,
                        phase,
                        f"{self._reactor_name} '{self.unit_id}' profile point {profile_index}",
                        check_global_lle=False,
                    )
                )
            global_stability_checks = {}
            global_indices = (
                sorted(set((
                    0,
                    len(solution.profile_states) // 2,
                    outlet_profile_index,
                )))
                if phase == 'liquid' else [outlet_profile_index]
            )
            for profile_index in global_indices:
                global_stability_checks[profile_index] = (
                    _validate_homogeneous_reactor_phase_stability(
                        self.thermo,
                        solution.profile_states[profile_index],
                        phase,
                        f"{self._reactor_name} '{self.unit_id}' profile point {profile_index}",
                        check_global_lle=True,
                    )
                )
        else:
            outlet_stability = (
                _validate_homogeneous_reactor_phase_stability(
                    self.thermo,
                    outlet,
                    phase,
                    f"{self._reactor_name} '{self.unit_id}' recycle outlet",
                    check_global_lle=True,
                )
            )
            stability_profile = [outlet_stability]
            global_stability_checks = {
                outlet_profile_index: outlet_stability,
            }
        stability = global_stability_checks[outlet_profile_index]
        maximum_material_residual = max(
            (abs(value) for value in solution.material_balance_residuals_kmol_h.values()),
            default=0.0,
        )
        tolerance = max(1.0e-8, inlet.F * 1.0e-8)
        if maximum_material_residual > tolerance:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' material closure residual is "
                f"{maximum_material_residual:.6g} kmol/h"
            )
        energy_tolerance = max(
            1.0e-4,
            abs(inlet.F * inlet.H) * 1.0e-9,
            abs(solution.heat_duty_kJ_h) * 1.0e-9,
        )
        if abs(solution.energy_balance_residual_kJ_h) > energy_tolerance:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' energy closure residual is "
                f"{solution.energy_balance_residual_kJ_h:.6g} kJ/h"
            )

        reaction_results = []
        for index, (reaction, extent) in enumerate(zip(
            reactions, solution.reaction_extents_kmol_h
        ), start=1):
            reaction_result = {
                'index': index,
                'name': reaction.name,
                'equation': reaction.reaction.equation,
                'stoichiometry': dict(reaction.reaction.stoichiometry),
                'kinetic_type': reaction.model,
                'reversible': reaction.reversible,
                'rate_basis': reaction.rate_basis,
                'rate_output_basis': reaction.rate_output_basis,
                'concentration_unit': reaction.concentration_unit,
                'pressure_unit': reaction.pressure_unit,
                'declared_rate_unit': reaction.rate_unit,
                'extent_kmol_h': extent,
            }
            reaction_result.update(
                self._reaction_result_rate_fields(solution, index)
            )
            reaction_results.append(reaction_result)
            warnings.extend(reaction.validation_warnings)
        performance = {
            'mode': mode,
            'phase': phase,
            'volume_m3': geometry.volume_m3,
            'length_m': geometry.length_m,
            'diameter_m': geometry.diameter_m,
            'n_tubes': geometry.n_tubes,
            'pressure_drop_model': pressure_model,
            'pressure_drop_bar': inlet.P - outlet.P,
            'residence_time_h': solution.residence_time_h,
            'T_in_C': inlet.T - 273.15,
            'T_out_C': outlet.T - 273.15,
            'P_in_bar': inlet.P,
            'P_out_bar': outlet.P,
            'component_conversions': {
                component: (
                    (flow - solution.outlet_component_flows.get(component, 0.0)) / flow
                )
                for component, flow in inlet.component_flows().items()
                if flow > 0.0
            },
            'reactions': reaction_results,
            'solver_method': solution.solver_method,
            'attempted_solver_methods': list(solution.attempted_methods),
            'solver_evaluations': solution.evaluations,
            'solver_status': solution.status,
            'solver_message': solution.message,
            'profile_points': len(solution.profile),
            'requested_profile_points': requested_profile_points,
            'detailed_diagnostics': detailed_diagnostics,
            'profile': list(solution.profile),
            'maximum_material_balance_residual_kmol_h': maximum_material_residual,
            'energy_balance_residual_kW': solution.energy_balance_residual_kJ_h / 3600.0,
            'duty_kW': solution.heat_duty_kJ_h / 3600.0,
            'phase_stability': stability,
            'phase_stability_profile': stability_profile,
            'global_phase_stability_profile_checks': global_stability_checks,
            'component_depletion_events': [
                dict(event) for event in solution.depletion_events
            ],
        }
        inactive_events = [
            event for event in solution.depletion_events
            if event.get('all_reactions_inactive_downstream')
        ]
        if inactive_events:
            first_inactive = min(
                inactive_events,
                key=lambda event: float(event['coordinate']),
            )
            performance['reaction_inactive_from_coordinate'] = float(
                first_inactive['coordinate']
            )
            if 'remaining_catalyst_mass_kg' in first_inactive:
                performance['inactive_remaining_catalyst_mass_kg'] = float(
                    first_inactive['remaining_catalyst_mass_kg']
                )
            if 'remaining_reactor_volume_m3' in first_inactive:
                performance['inactive_remaining_reactor_volume_m3'] = float(
                    first_inactive['remaining_reactor_volume_m3']
                )
        if pressure_model in {'darcy', 'ergun'}:
            performance.update({
                'inlet_velocity_m_s': solution.profile[0].get('velocity_m_s'),
                'outlet_velocity_m_s': solution.profile[-1].get('velocity_m_s'),
                'inlet_reynolds_number': solution.profile[0].get('reynolds_number'),
                'outlet_reynolds_number': solution.profile[-1].get('reynolds_number'),
            })
        self._extend_performance(performance, solution)
        return UnitResult(
            outlet_streams={'out': outlet},
            heat_duty=solution.heat_duty_kJ_h,
            performance=performance,
            warnings=list(dict.fromkeys(warnings)),
        )


class KineticsPackedBed(KineticsPFR):
    """Fixed-catalyst packed-bed reactor with catalyst-mass kinetics."""

    _reactor_name = 'PackedBedReactor'
    _rate_output_basis = 'catalyst_mass'
    _MASS_FACTORS = {
        'kg': 1.0, 'g': 1.0e-3, 'mg': 1.0e-6,
        'lb': 0.45359237, 'lbm': 0.45359237,
        't': 1000.0, 'tonne': 1000.0,
    }
    _BULK_DENSITY_FACTORS = {
        'kg/m3': 1.0, 'kg/m^3': 1.0,
        'g/cm3': 1000.0, 'g/cm^3': 1000.0,
        'kg/l': 1000.0,
        'lb/ft3': 16.01846337, 'lbm/ft3': 16.01846337,
    }
    _DIFFUSIVITY_FACTORS = {
        'm2/s': 1.0, 'm^2/s': 1.0,
        'm2/h': 1.0 / 3600.0, 'm^2/h': 1.0 / 3600.0,
        'cm2/s': 1.0e-4, 'cm^2/s': 1.0e-4,
        'mm2/s': 1.0e-6, 'mm^2/s': 1.0e-6,
    }

    def __init__(self, unit_id: str, thermo, params: dict):
        normalized = dict(params)
        if (
            normalized.get('void_fraction') is not None
            and normalized.get('bed_void_fraction') is not None
        ):
            raise UnitOperationError(
                f"{self._reactor_name} '{unit_id}' received duplicate aliases "
                "for bed void fraction: bed_void_fraction, void_fraction"
            )
        if (
            normalized.get('void_fraction') is None
            and normalized.get('bed_void_fraction') is not None
        ):
            normalized['void_fraction'] = normalized['bed_void_fraction']
        super().__init__(unit_id, thermo, normalized)

    def _geometry(self):
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .pfr_models import PFRGeometry
        else:
            from pfr_models import PFRGeometry

        diameter = self._dimension(
            ('diameter', 'D', 'bed_diameter'),
            self._LENGTH_FACTORS,
            'bed diameter',
        )
        if diameter is None:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' requires bed diameter"
            )
        n_tubes = int(self.get_param('n_tubes', 1))
        if n_tubes < 1:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' n_tubes must be positive"
            )
        area = n_tubes * math.pi * diameter**2 / 4.0
        length = self._dimension(
            ('length', 'L', 'bed_length'),
            self._LENGTH_FACTORS,
            'bed length',
        )
        volume = self._dimension(
            ('bed_volume', 'volume', 'V'),
            self._VOLUME_FACTORS,
            'bed volume',
        )
        catalyst_mass = self._dimension(
            ('catalyst_mass', 'W_cat', 'Wcat'),
            self._MASS_FACTORS,
            'catalyst mass',
        )
        bulk_density = self._dimension(
            ('bulk_catalyst_density', 'catalyst_bulk_density', 'rho_bulk'),
            self._BULK_DENSITY_FACTORS,
            'bulk catalyst density',
        )
        if bulk_density is None:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' requires "
                "bulk_catalyst_density"
            )
        void_fraction = self.get_param(
            'bed_void_fraction', self.get_param('void_fraction')
        )
        if void_fraction is None:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' requires bed_void_fraction"
            )
        void_fraction = float(void_fraction)
        if not math.isfinite(void_fraction) or not 0.0 < void_fraction < 1.0:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' bed_void_fraction must "
                "lie between zero and one"
            )

        if volume is None and length is not None:
            volume = area * length
        elif volume is not None and length is None:
            length = volume / area
        elif volume is not None and length is not None:
            geometric = area * length
            if abs(geometric / volume - 1.0) > 1.0e-6:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' bed_volume, length, "
                    f"diameter, and n_tubes are inconsistent; geometric volume "
                    f"is {geometric:.8g} m3"
                )
        elif catalyst_mass is not None:
            volume = catalyst_mass / bulk_density
            length = volume / area
        else:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' requires length, "
                "bed_volume, or catalyst_mass"
            )

        derived_mass = bulk_density * volume
        if catalyst_mass is None:
            catalyst_mass = derived_mass
        elif abs(derived_mass / catalyst_mass - 1.0) > 1.0e-6:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' catalyst_mass is "
                f"inconsistent with bed volume and bulk density; derived mass "
                f"is {derived_mass:.8g} kg"
            )
        self._packed_bed = {
            'catalyst_mass_kg': catalyst_mass,
            'bulk_catalyst_density_kg_m3_bed': bulk_density,
            'bed_void_fraction': void_fraction,
            'particle_density_kg_m3_particle': bulk_density / (1.0 - void_fraction),
        }
        return PFRGeometry(volume, length, diameter, n_tubes)

    def _diffusion_rate_modifier(self, reactions, pressure_spec):
        model = str(self.get_param('diffusion_model', 'none')).strip().lower()
        model = model.replace('-', '_')
        effective_diffusivity = self._dimension(
            ('effective_diffusivity', 'D_effective', 'D_eff'),
            self._DIFFUSIVITY_FACTORS,
            'effective diffusivity',
        )
        specified_eta = self.get_param('effectiveness_factor')
        requested_limiting = self.get_param(
            'diffusion_limiting_component',
            self.get_param('limiting_component'),
        )
        requested_policy = self.get_param('effectiveness_factor_policy')
        pellet_tolerance_spec = self.get_param('pellet_relative_tolerance')
        pellet_nodes_spec = self.get_param('pellet_maximum_nodes')
        if effective_diffusivity is not None and model == 'none':
            model = 'generalized_power_law_sphere'
        if specified_eta is not None and model == 'none':
            model = 'specified'
        if specified_eta is not None and model != 'specified':
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' effectiveness_factor "
                "cannot be combined with a calculated diffusion model"
            )
        kinetic_basis = str(
            self.get_param('kinetic_basis', 'apparent')
        ).strip().lower()
        if kinetic_basis not in {'apparent', 'intrinsic'}:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' kinetic_basis must be "
                "apparent or intrinsic"
            )
        if kinetic_basis == 'apparent' and model != 'none':
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' apparent kinetics "
                "cannot also apply an effectiveness factor"
            )
        if model == 'none':
            if (
                requested_limiting is not None
                or requested_policy is not None
                or pellet_tolerance_spec is not None
                or pellet_nodes_spec is not None
            ):
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' diffusion limiting "
                    "component, factor policy, and pellet BVP controls require "
                    "a calculated diffusion model"
                )
            self._diffusion_performance = {
                'kinetic_basis': kinetic_basis,
                'diffusion_model': 'none',
            }
            return None
        if model == 'specified':
            if specified_eta is None:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' "
                    "diffusion_model=specified requires effectiveness_factor"
                )
            if effective_diffusivity is not None:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' specified "
                    "effectiveness_factor cannot also specify diffusivity"
                )
            if requested_limiting is not None:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' a specified "
                    "effectiveness_factor does not use diffusion_limiting_component"
                )
            if requested_policy is not None and str(requested_policy).strip().lower() not in {
                'constant', 'inlet', 'constant_inlet', 'constant-inlet',
            }:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' a specified "
                    "effectiveness_factor is constant throughout the bed"
                )
            eta = float(specified_eta)
            if not math.isfinite(eta) or not 0.0 < eta <= 1.0:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' effectiveness_factor "
                    "must lie in (0, 1]"
                )
            self._diffusion_performance = {
                'kinetic_basis': kinetic_basis,
                'diffusion_model': 'specified',
                'specified_effectiveness_factor': eta,
            }
            return lambda _reaction, rate, _state, _coordinate: (eta * rate, eta)
        first_order_only = model in {'first_order_sphere', 'sphere'}
        generalized = model in {
            'generalized_power_law_sphere',
            'generalized_sphere',
            'power_law_sphere',
        }
        rigorous = model in {
            'rigorous_power_law_sphere',
            'rigorous_sphere',
        }
        if not first_order_only and not generalized and not rigorous:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' diffusion_model must be "
                "none, specified, first_order_sphere, or "
                "generalized_power_law_sphere, or rigorous_power_law_sphere"
            )
        if kinetic_basis != 'intrinsic':
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' calculated diffusion "
                "requires kinetic_basis=intrinsic"
            )
        if any(reaction.model != 'power_law' for reaction in reactions):
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' calculated spherical "
                "diffusion requires power-law reactions; custom and custom_net "
                "kinetics require a specified effectiveness_factor"
            )
        reversible = [reaction for reaction in reactions if reaction.reversible]
        if rigorous and reversible:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' rigorous_power_law_sphere "
                "currently supports irreversible reactions; use the generalized "
                "model for one reversible reaction"
            )
        if reversible and len(reactions) != 1:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' a single calculated "
                "effectiveness factor cannot represent multiple reactions when "
                "any reaction is reversible"
            )
        if effective_diffusivity is None:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' calculated spherical diffusion "
                "requires effective_diffusivity"
            )
        radius = self._dimension(
            ('pellet_radius',), self._LENGTH_FACTORS, 'pellet radius'
        )
        particle_diameter = pressure_spec.particle_diameter_m
        if radius is None:
            if particle_diameter is None:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' calculated spherical diffusion "
                    "requires pellet_radius or particle_diameter"
                )
            radius = 0.5 * float(particle_diameter)
        elif particle_diameter is not None and abs(
            2.0 * radius / float(particle_diameter) - 1.0
        ) > 1.0e-6:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' pellet_radius and "
                "particle_diameter are inconsistent"
            )
        candidates = None
        for reaction in reactions:
            consumed_with_positive_order = {
                component
                for component, coefficient in reaction.reaction.stoichiometry.items()
                if coefficient < 0.0
                and float(reaction.reaction_orders.get(component, 0.0)) > 0.0
            }
            candidates = (
                consumed_with_positive_order
                if candidates is None
                else candidates & consumed_with_positive_order
            )
        candidates = candidates or set()
        if requested_limiting is None:
            if len(candidates) != 1:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' cannot infer one "
                    "diffusion-limiting component shared by every reaction; "
                    "specify diffusion_limiting_component"
                )
            limiting_component = next(iter(candidates))
        else:
            limiting_component = str(requested_limiting).strip()
            if limiting_component not in self.thermo.components:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' unknown "
                    f"diffusion_limiting_component {limiting_component!r}"
                )
            if limiting_component not in candidates:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' limiting component "
                    f"{limiting_component!r} must be consumed with positive "
                    "order in every corrected reaction"
                )
        limiting_orders = [
            float(reaction.reaction_orders[limiting_component])
            for reaction in reactions
        ]
        if any(order <= 0.0 or not math.isfinite(order) for order in limiting_orders):
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' limiting-species order "
                "must be positive and finite"
            )
        if not reversible and not rigorous and any(
            abs(order - limiting_orders[0]) > 1.0e-12
            for order in limiting_orders[1:]
        ):
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' reactions can share one "
                "calculated effectiveness factor only when they have the same "
                f"order in {limiting_component}"
            )
        if first_order_only and (
            len(reactions) != 1
            or reactions[0].reversible
            or reactions[0].rate_basis != 'concentration'
            or len([
                order for order in reactions[0].reaction_orders.values()
                if abs(order) > 1.0e-14
            ]) != 1
            or abs(limiting_orders[0] - 1.0) > 1.0e-12
        ):
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' first_order_sphere "
                "requires one irreversible first-order concentration-basis "
                "power-law reaction with one nonzero order"
            )
        particle_density = self._packed_bed['particle_density_kg_m3_particle']
        diffusivity_m2_h = effective_diffusivity * 3600.0
        factor_policy = str(self.get_param(
            'effectiveness_factor_policy', 'constant_inlet'
        )).strip().lower().replace('-', '_')
        factor_policy = {
            'constant': 'constant_inlet',
            'inlet': 'constant_inlet',
            'axial': 'local',
        }.get(factor_policy, factor_policy)
        if factor_policy not in {'constant_inlet', 'local'}:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' "
                "effectiveness_factor_policy must be constant_inlet or local"
            )
        if rigorous and factor_policy != 'constant_inlet':
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' rigorous pellet solving "
                "uses constant_inlet plus optional effectiveness_factor_resolves"
            )
        resolves = int(self.get_param('effectiveness_factor_resolves', 1))
        if resolves < 1 or resolves > 101:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' "
                "effectiveness_factor_resolves must be between 1 and 101"
            )
        if resolves > 1 and not rigorous:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' multiple effectiveness "
                "factor resolves are available only for rigorous_power_law_sphere"
            )
        if not rigorous and (
            pellet_tolerance_spec is not None or pellet_nodes_spec is not None
        ):
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' pellet BVP controls are "
                "available only for rigorous_power_law_sphere"
            )
        pellet_tolerance = float(
            pellet_tolerance_spec if pellet_tolerance_spec is not None else 1.0e-6
        )
        pellet_maximum_nodes = int(
            pellet_nodes_spec if pellet_nodes_spec is not None else 2000
        )
        if rigorous and (
            not math.isfinite(pellet_tolerance) or pellet_tolerance <= 0.0
        ):
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' "
                "pellet_relative_tolerance must be positive and finite"
            )
        if rigorous and pellet_maximum_nodes < 50:
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' "
                "pellet_maximum_nodes must be at least 50"
            )
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .kinetic_models import (
                evaluate_reaction_rate,
                reaction_log_driving_force,
            )
            from .pellet_models import (
                PelletModelError,
                generalized_power_law_spherical_effectiveness,
                rigorous_power_law_network_spherical_effectiveness,
            )
        else:
            from kinetic_models import (
                evaluate_reaction_rate,
                reaction_log_driving_force,
            )
            from pellet_models import (
                PelletModelError,
                generalized_power_law_spherical_effectiveness,
                rigorous_power_law_network_spherical_effectiveness,
            )
        reaction_indices = {
            id(reaction): index for index, reaction in enumerate(reactions)
        }
        cache = {'state': None, 'etas': None}

        def calculated_effectiveness(state, *, force=False):
            if not force and cache['etas'] is not None and (
                factor_policy == 'constant_inlet' or cache['state'] is state
            ):
                return cache['etas']
            concentration = float(
                state.concentrations_kmol_m3.get(limiting_component, 0.0)
            )
            surface_rates = [
                evaluate_reaction_rate(reaction, state, self.thermo)
                for reaction in reactions
            ]
            consumption_terms = [
                -reaction.reaction.stoichiometry[limiting_component] * rate
                for reaction, rate in zip(reactions, surface_rates)
            ]
            consumption = sum(consumption_terms)
            if concentration <= 0.0 or abs(consumption) <= 1.0e-300:
                etas = tuple(1.0 for _ in reactions)
                phi = 0.0
                generalized_phi = 0.0
                apparent_order = limiting_orders[0]
                method = 'zero_surface_source'
                center_ratio = 1.0
                radial_nodes = 0
            elif consumption < 0.0:
                raise UnitOperationError(
                    f"{self._reactor_name} '{self.unit_id}' local net kinetics "
                    f"produce diffusion-limiting component {limiting_component}; "
                    "a positive single effectiveness factor is not applicable"
                )
            else:
                if rigorous:
                    terms = tuple(
                        radius * radius * particle_density * term
                        / (diffusivity_m2_h * concentration)
                        for term in consumption_terms
                    )
                    try:
                        network = (
                            rigorous_power_law_network_spherical_effectiveness(
                                terms,
                                limiting_orders,
                                relative_tolerance=pellet_tolerance,
                                maximum_nodes=pellet_maximum_nodes,
                            )
                        )
                    except PelletModelError as error:
                        raise UnitOperationError(
                            f"{self._reactor_name} '{self.unit_id}' rigorous "
                            f"pellet model failed: {error}"
                        ) from error
                    etas = network.effectiveness_factors
                    phi = math.sqrt(sum(terms))
                    apparent_order = sum(
                        term * order
                        for term, order in zip(terms, limiting_orders)
                    ) / sum(terms)
                    generalized_phi = phi * math.sqrt(
                        (apparent_order + 1.0) / 2.0
                    )
                    method = network.method
                    center_ratio = network.center_concentration_ratio
                    radial_nodes = network.radial_nodes
                    diagnostics = self._diffusion_performance
                    diagnostics['effectiveness_factor_resolves_actual'] = (
                        diagnostics.get('effectiveness_factor_resolves_actual', 0)
                        + 1
                    )
                else:
                    apparent_order = limiting_orders[0]
                    if reversible:
                        reaction = reactions[0]
                        delta = reaction_log_driving_force(
                            reaction, state, self.thermo
                        )
                        if delta >= 0.0:
                            raise UnitOperationError(
                                f"{self._reactor_name} '{self.unit_id}' reversible "
                                "reaction is not consuming the nominated limiting "
                                "species at the local state"
                            )
                        quotient_ratio = math.exp(delta)
                        driving = -math.expm1(delta)
                        stoich = reaction.reaction.stoichiometry[limiting_component]
                        apparent_order = (
                            limiting_orders[0]
                            - quotient_ratio * (limiting_orders[0] + stoich)
                        ) / driving
                        if not math.isfinite(apparent_order) or apparent_order <= 0.0:
                            raise UnitOperationError(
                                f"{self._reactor_name} '{self.unit_id}' reversible "
                                "reaction has nonpositive local apparent order in "
                                f"{limiting_component}"
                            )
                    phi = radius * math.sqrt(
                        particle_density * consumption
                        / (diffusivity_m2_h * concentration)
                    )
                    try:
                        effectiveness = (
                            generalized_power_law_spherical_effectiveness(
                                phi,
                                apparent_order,
                            )
                        )
                    except PelletModelError as error:
                        raise UnitOperationError(
                            f"{self._reactor_name} '{self.unit_id}' pellet model "
                            f"failed: {error}"
                        ) from error
                    etas = tuple(
                        effectiveness.effectiveness_factor for _ in reactions
                    )
                    generalized_phi = effectiveness.generalized_thiele_modulus
                    method = effectiveness.method
                    center_ratio = effectiveness.center_concentration_ratio
                    radial_nodes = effectiveness.radial_nodes
            diagnostics = self._diffusion_performance
            if 'inlet_thiele_modulus' not in diagnostics:
                diagnostics.update({
                    'inlet_thiele_modulus': phi,
                    'inlet_generalized_thiele_modulus': generalized_phi,
                    'inlet_apparent_order': apparent_order,
                    'effectiveness_correlation': method,
                    'inlet_center_concentration_ratio': center_ratio,
                    'pellet_radial_nodes': radial_nodes,
                    'inlet_reaction_effectiveness_factors': list(etas),
                })
            diagnostics['minimum_thiele_modulus'] = min(
                phi, diagnostics.get('minimum_thiele_modulus', phi)
            )
            diagnostics['maximum_thiele_modulus'] = max(
                phi, diagnostics.get('maximum_thiele_modulus', phi)
            )
            diagnostics['minimum_generalized_thiele_modulus'] = min(
                generalized_phi,
                diagnostics.get(
                    'minimum_generalized_thiele_modulus', generalized_phi
                ),
            )
            diagnostics['maximum_generalized_thiele_modulus'] = max(
                generalized_phi,
                diagnostics.get(
                    'maximum_generalized_thiele_modulus', generalized_phi
                ),
            )
            diagnostics['minimum_apparent_order'] = min(
                apparent_order,
                diagnostics.get('minimum_apparent_order', apparent_order),
            )
            diagnostics['maximum_apparent_order'] = max(
                apparent_order,
                diagnostics.get('maximum_apparent_order', apparent_order),
            )
            cache['state'] = state
            cache['etas'] = etas
            return etas

        def modifier(reaction, rate, state, _coordinate):
            eta = calculated_effectiveness(state)[reaction_indices[id(reaction)]]
            return eta * rate, eta

        modifier.calculate_effectiveness = calculated_effectiveness
        modifier.rigorous = rigorous
        modifier.reaction_count = len(reactions)

        self._diffusion_performance = {
            'kinetic_basis': kinetic_basis,
            'diffusion_model': (
                'first_order_sphere'
                if first_order_only else (
                    'rigorous_power_law_sphere'
                    if rigorous else 'generalized_power_law_sphere'
                )
            ),
            'effective_diffusivity_m2_s': effective_diffusivity,
            'pellet_radius_m': radius,
            'diffusion_limiting_component': limiting_component,
            'common_limiting_species_order': (
                None
                if reversible or any(
                    abs(order - limiting_orders[0]) > 1.0e-12
                    for order in limiting_orders[1:]
                )
                else limiting_orders[0]
            ),
            'limiting_species_orders': list(limiting_orders),
            'effectiveness_factor_scope': (
                'rigorous_per_reaction_shared_limiting_species'
                if rigorous else (
                    'single_reversible_reaction_local_apparent_order'
                    if reversible else 'shared_power_law_order'
                )
            ),
            'effectiveness_factor_policy': factor_policy,
            'effectiveness_factor_resolves_requested': resolves,
            'effectiveness_factor_resolves_actual': 0,
            'pellet_relative_tolerance': (
                pellet_tolerance if rigorous else None
            ),
            'pellet_maximum_nodes': (
                pellet_maximum_nodes if rigorous else None
            ),
        }
        return modifier

    def _adaptive_reaction_configuration(
        self,
        geometry,
        reactions,
        pressure_spec,
        inlet,
        thermal,
        phase,
    ) -> dict:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .pfr_models import AxialReactionMeasure
        else:
            from pfr_models import AxialReactionMeasure
        if pressure_spec.model == 'darcy':
            raise UnitOperationError(
                f"{self._reactor_name} '{self.unit_id}' does not support Darcy "
                "pipe pressure drop; use none, specified, or Ergun"
            )
        catalyst_mass = self._packed_bed['catalyst_mass_kg']
        rate_modifier = self._diffusion_rate_modifier(
            reactions, pressure_spec
        )
        if (
            rate_modifier is not None
            and hasattr(rate_modifier, 'calculate_effectiveness')
        ):
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .kinetic_models import HomogeneousRateState
            else:
                from kinetic_models import HomogeneousRateState
            inlet_temperature = (
                thermal.isothermal_temperature_K
                if thermal.mode == 'isothermal'
                else inlet.T
            )
            inlet_rate_state = HomogeneousRateState.from_flows(
                self.thermo,
                inlet.component_flows(),
                inlet_temperature,
                inlet.P,
                phase,
            )
            rate_modifier.calculate_effectiveness(inlet_rate_state)
        return {
            'reaction_measure': AxialReactionMeasure(
                'catalyst_mass',
                catalyst_mass,
                catalyst_mass / geometry.length_m,
            ),
            'rate_modifier': rate_modifier,
            'residence_void_fraction': self._packed_bed['bed_void_fraction'],
        }

    def _reaction_result_rate_fields(self, solution, index: int) -> dict:
        profile = solution.profile[-1]
        return {
            'outlet_rate_kmol_kg_cat_h': (
                profile['reaction_rates_kmol_kg_cat_h'][index - 1]
            ),
            'outlet_intrinsic_rate_kmol_kg_cat_h': (
                profile['intrinsic_reaction_rates_kmol_kg_cat_h'][index - 1]
            ),
            'outlet_effectiveness_factor': (
                profile['effectiveness_factors'][index - 1]
            ),
        }

    def _solve_adaptive_model(
        self,
        _pfr_solver,
        inlet,
        reactions,
        phase,
        geometry,
        thermal,
        pressure_spec,
        controls,
        adaptive_configuration,
    ):
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .pfr_models import solve_adaptive_packed_bed
            from .kinetic_models import HomogeneousRateState
        else:
            from pfr_models import solve_adaptive_packed_bed
            from kinetic_models import HomogeneousRateState
        modifier = adaptive_configuration.get('rate_modifier')
        resolves = int(self.get_param('effectiveness_factor_resolves', 1))
        if (
            resolves <= 1
            or modifier is None
            or not getattr(modifier, 'rigorous', False)
        ):
            return solve_adaptive_packed_bed(
                inlet,
                reactions,
                self.thermo,
                phase,
                geometry,
                thermal,
                pressure_spec,
                controls,
                **adaptive_configuration,
            )

        from dataclasses import replace
        preliminary_controls = replace(controls, profile_points=resolves)
        preliminary = solve_adaptive_packed_bed(
            inlet,
            reactions,
            self.thermo,
            phase,
            geometry,
            thermal,
            pressure_spec,
            preliminary_controls,
            **adaptive_configuration,
        )
        support_positions = [
            float(row['position_m']) for row in preliminary.profile
        ]
        support_factors = []
        for index, state in enumerate(preliminary.profile_states):
            rate_state = HomogeneousRateState.from_flows(
                self.thermo,
                state.component_flows(),
                state.T,
                state.P,
                phase,
            )
            support_factors.append(tuple(
                modifier.calculate_effectiveness(
                    rate_state,
                    force=index > 0,
                )
            ))
        reaction_indices = {
            id(reaction): index for index, reaction in enumerate(reactions)
        }

        def interpolated_modifier(reaction, rate, _state, coordinate):
            reaction_index = reaction_indices[id(reaction)]
            if coordinate <= support_positions[0]:
                eta = support_factors[0][reaction_index]
            elif coordinate >= support_positions[-1]:
                eta = support_factors[-1][reaction_index]
            else:
                from bisect import bisect_right
                upper = bisect_right(support_positions, coordinate)
                lower = upper - 1
                fraction = (
                    (coordinate - support_positions[lower])
                    / (support_positions[upper] - support_positions[lower])
                )
                eta = (
                    support_factors[lower][reaction_index]
                    + fraction * (
                        support_factors[upper][reaction_index]
                        - support_factors[lower][reaction_index]
                    )
                )
            return eta * rate, eta

        final_configuration = dict(adaptive_configuration)
        final_configuration['rate_modifier'] = interpolated_modifier
        self._diffusion_performance.update({
            'effectiveness_factor_policy': 'sampled_interpolation',
            'effectiveness_factor_support_positions_m': support_positions,
            'effectiveness_factor_support_values': [
                list(values) for values in support_factors
            ],
            'effectiveness_factor_preliminary_axial_passes': 1,
        })
        return solve_adaptive_packed_bed(
            inlet,
            reactions,
            self.thermo,
            phase,
            geometry,
            thermal,
            pressure_spec,
            controls,
            **final_configuration,
        )

    def _extend_performance(self, performance: dict, solution) -> None:
        performance.update(self._packed_bed)
        performance.update(self._diffusion_performance)
        performance['fluid_void_volume_m3'] = (
            performance['volume_m3'] * self._packed_bed['bed_void_fraction']
        )
        effectiveness = [
            value
            for row in solution.profile
            for value in row['effectiveness_factors']
        ]
        performance['minimum_effectiveness_factor'] = min(effectiveness)
        performance['maximum_effectiveness_factor'] = max(effectiveness)
