"""
Liquid-liquid, adsorption, absorption, and stripping separation unit operations.
"""

import math

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .equilibrium_stage_column import EquilibriumStageColumnMixin
else:
    from equilibrium_stage_column import EquilibriumStageColumnMixin
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics import StreamState, VLLEFlashResult
else:
    from thermodynamics import StreamState, VLLEFlashResult
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_basic import Flash, _ThermoStateSolver
else:
    from unit_operations_basic import Flash, _ThermoStateSolver
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_base import UnitOperation, UnitOperationError, UnitResult
else:
    from unit_operations_base import UnitOperation, UnitOperationError, UnitResult


class Flash3(Flash):
    """Single-stage vapor-liquid-liquid flash separation."""

    supports_permanent_solids = False

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        self._require_vlle_thermo()

        if len(inlets) != 1:
            raise UnitOperationError(
                f"Flash3 '{self.unit_id}' requires exactly 1 inlet stream"
            )
        inlet = next(iter(inlets.values()))
        if inlet.F <= 0.0:
            raise UnitOperationError(
                f"Flash3 '{self.unit_id}' requires positive inlet flow"
            )

        T = self.get_temperature_param('T')
        if T is None:
            T = self.get_temperature_param('temperature')
        P = self.get_param('P', self.get_param('pressure'))
        VF = self._vapor_fraction_spec()
        Q = self._heat_duty_spec()

        specs = sum(1 for value in (T, P, VF, Q) if value is not None)
        if specs != 2:
            raise UnitOperationError(
                f"Flash3 '{self.unit_id}' requires exactly 2 specifications "
                f"(T, P, VF, Q), got {specs}"
            )

        if T is not None:
            T = float(T)
            if not math.isfinite(T) or T <= 0.0:
                raise UnitOperationError(
                    f"Flash3 '{self.unit_id}' temperature must be positive"
                )
        if P is not None:
            P = float(P)
            if not math.isfinite(P) or P <= 0.0:
                raise UnitOperationError(
                    f"Flash3 '{self.unit_id}' pressure must be positive"
                )
        if VF is not None:
            VF = float(VF)
            if not 0.0 <= VF <= 1.0:
                raise UnitOperationError(
                    f"Flash3 '{self.unit_id}' vapor_fraction must be between 0 and 1"
                )

        composition = self._normalized_composition(inlet.composition)
        reference_T = T if T is not None else inlet.T
        aqueous_context, henry_info, henry_warnings = self._flash_henry_context(
            inlet,
            composition,
            reference_T,
        )
        inlet_H = self._stream_enthalpy(inlet, aqueous_context)
        H_target = inlet_H + Q / inlet.F if Q is not None else None
        max_iter = self._max_iter()
        tol = self._tolerance()
        ph_residual = None

        aqueous_state = None
        if aqueous_context is not None:
            if T is not None and P is not None:
                aqueous_state = self._state_at_TP(
                    composition, T, P, inlet.F, aqueous_context
                )
            elif P is not None and VF is not None:
                aqueous_state = self._state_at_PV(
                    composition, P, VF, inlet.F, inlet.T, aqueous_context
                )
            elif T is not None and VF is not None:
                P = self._find_P_for_VF(
                    composition, T, VF, inlet.P, aqueous_context
                )
                aqueous_state = self._state_at_TP(
                    composition, T, P, inlet.F, aqueous_context
                )
            elif P is not None and Q is not None:
                aqueous_state = self._state_at_PH(
                    composition, P, H_target, inlet.F, inlet.T, aqueous_context
                )
                ph_residual = (aqueous_state.H or 0.0) - H_target
            elif T is not None and Q is not None:
                P = self._find_P_for_H_at_T(
                    composition,
                    T,
                    H_target,
                    inlet.P,
                    inlet.F,
                    aqueous_context,
                )
                aqueous_state = self._state_at_TP(
                    composition, T, P, inlet.F, aqueous_context
                )
            elif VF is not None and Q is not None:
                P = self._find_P_for_VF_and_H(
                    composition,
                    VF,
                    H_target,
                    inlet.P,
                    inlet.F,
                    aqueous_context,
                )
                aqueous_state = self._state_at_PV(
                    composition, P, VF, inlet.F, inlet.T, aqueous_context
                )
            else:
                raise UnitOperationError(
                    f"Flash3 '{self.unit_id}' specification combination not supported"
                )
            T = aqueous_state.T
            P = aqueous_state.P
            result = self._henry_flash3_result(aqueous_state)
            self._reject_henry_liquid_split(
                result, T, max_iter=max_iter, tol=tol
            )
        elif T is not None and P is not None:
            result = self.thermo.flash3_TP(
                composition, T, P, max_iter=max_iter, tol=tol
            )
        elif P is not None and VF is not None:
            T, result = self._flash3_PV(composition, P, VF, inlet.T, max_iter)
        elif T is not None and VF is not None:
            P, result = self._flash3_TV(composition, T, VF, inlet.P, max_iter)
        elif P is not None and Q is not None:
            T, result, ph_residual = self._flash3_PH(
                composition, P, H_target, inlet.T, max_iter
            )
        elif T is not None and Q is not None:
            P = self._find_P_for_H_at_T_flash3(
                composition, T, H_target, inlet.P, max_iter, tol
            )
            result = self.thermo.flash3_TP(
                composition, T, P, max_iter=max_iter, tol=tol
            )
        elif VF is not None and Q is not None:
            P = self._find_P_for_VF_and_H_flash3(
                composition, VF, H_target, inlet.P, max_iter
            )
            T, result = self._flash3_PV(composition, P, VF, inlet.T, max_iter)
        else:
            raise UnitOperationError(
                f"Flash3 '{self.unit_id}' specification combination not supported"
            )

        phase_flows = self._phase_flows(inlet.F, result)
        vapor_out = self._phase_outlet(
            T, P, phase_flows['vapor'], result.y, 'vapor', aqueous_context
        )
        liquid1_out = self._phase_outlet(
            T, P, phase_flows['liquid1'], result.x1, 'liquid', aqueous_context
        )
        liquid2_out = self._phase_outlet(
            T, P, phase_flows['liquid2'], result.x2, 'liquid', aqueous_context
        )
        outlets = {
            'vapor_out': vapor_out,
            'liquid1_out': liquid1_out,
            'liquid2_out': liquid2_out,
        }
        liquid_diagnostics = self._liquid_phase_diagnostics(
            liquid1_out,
            liquid2_out,
            phase_flows['liquid1'],
            phase_flows['liquid2'],
        )

        outlet_H = (
            aqueous_state.H
            if aqueous_state is not None
            else self._flash3_mixture_enthalpy(T, P, result)
        )
        Q_calc = inlet.F * (outlet_H - inlet_H)
        duty_residual = Q_calc - Q if Q is not None else None
        component_residual = self._component_balance_residual(
            composition, inlet.F, list(outlets.values())
        )

        warnings = list(henry_warnings)
        if aqueous_context is not None and result.liquid1_fraction > 1e-12:
            warnings.extend(self._henry_solution_warnings(
                aqueous_context,
                henry_info,
                [result.x1],
                [T],
                [P],
            ))
        if component_residual > 1e-7:
            warnings.append(
                f"Flash3 component balance residual is {component_residual:.3g}"
            )
        if result.extra.get('phase_amounts_underdetermined'):
            warnings.append(
                "Binary invariant VLLE phase amounts are underdetermined by TP "
                "specifications; reported phase amounts use the selected vapor "
                "fraction within the feasible bounds."
            )

        performance = {
            'T_C': T - 273.15,
            'P_bar': P,
            'phase_count': result.phase_count,
            'flash_status': result.status,
            'vapor_fraction': result.vapor_fraction,
            'liquid1_fraction': result.liquid1_fraction,
            'liquid2_fraction': result.liquid2_fraction,
            'vapor_flow_kmol_hr': phase_flows['vapor'],
            'liquid1_flow_kmol_hr': phase_flows['liquid1'],
            'liquid2_flow_kmol_hr': phase_flows['liquid2'],
            'vapor_composition': dict(result.y),
            'liquid1_composition': dict(result.x1),
            'liquid2_composition': dict(result.x2),
            'residual': result.residual,
            'iterations': result.iterations,
            'duty_kW': Q_calc / 3600.0,
            'component_balance_residual': component_residual,
            'henry': henry_info,
        }
        performance.update(liquid_diagnostics)
        if duty_residual is not None:
            performance['duty_residual_kW'] = duty_residual / 3600.0
        if 'vapor_fraction_bounds' in result.extra:
            performance['vapor_fraction_bounds'] = result.extra['vapor_fraction_bounds']

        if ph_residual is not None:
            performance['enthalpy_residual_kJ_kmol'] = ph_residual

        return UnitResult(
            outlet_streams=outlets,
            heat_duty=Q_calc,
            performance=performance,
            warnings=warnings,
        )

    @staticmethod
    def _henry_flash3_result(state: StreamState) -> VLLEFlashResult:
        V = max(0.0, min(1.0, float(state.vapor_fraction)))
        x = dict(state.x or state.composition)
        y = dict(state.y or state.composition)
        if V <= 1e-10:
            phase_count = 1
            status = 'aqueous_henry_liquid'
        elif V >= 1.0 - 1e-10:
            phase_count = 1
            status = 'aqueous_henry_vapor'
        else:
            phase_count = 2
            status = 'aqueous_henry_vle'
        return VLLEFlashResult(
            phase_count=phase_count,
            status=status,
            vapor_fraction=V,
            liquid1_fraction=1.0 - V,
            liquid2_fraction=0.0,
            y=y,
            x1=x,
            x2=x,
            residual=0.0,
            iterations=0,
            extra={'aqueous_henry_context': True},
        )

    def _reject_henry_liquid_split(self, result: VLLEFlashResult, T: float,
                                   max_iter: int, tol: float) -> None:
        if result.liquid1_fraction <= 1e-10:
            return
        try:
            has_lle, _x1, _x2, _beta = self.thermo.liquid_liquid_equilibrium(
                result.x1,
                T,
                max_iter=max_iter,
                tol=max(tol, 1e-8),
            )
        except Exception as exc:
            raise UnitOperationError(
                f"Flash3 '{self.unit_id}' could not verify that the Henry-aware "
                "liquid remains single phase"
            ) from exc
        if has_lle:
            raise UnitOperationError(
                f"Flash3 '{self.unit_id}' found two liquid phases after selecting "
                "a pure-water Henry context. Coupled Henry/VLLE equilibrium is not "
                "supported; disable Henry treatment or use a fitted conventional "
                "activity model for this state."
            )

    def _require_vlle_thermo(self) -> None:
        if not hasattr(self.thermo, 'flash3_TP'):
            raise UnitOperationError(
                f"Flash3 '{self.unit_id}' requires a VLLE-capable activity "
                "coefficient thermodynamic model such as NRTL, UNIQUAC, or UNIFAC"
            )

    def _max_iter(self) -> int:
        max_iter = int(self.get_param('max_iter', self.get_param('flash_max_iter', 100)))
        if max_iter <= 0:
            raise UnitOperationError(f"Flash3 '{self.unit_id}' requires positive max_iter")
        return max_iter

    def _tolerance(self) -> float:
        tol = float(self.get_param('tol', self.get_param('flash_tolerance', 1e-9)))
        if tol <= 0.0:
            raise UnitOperationError(f"Flash3 '{self.unit_id}' requires positive tolerance")
        return tol

    def _flash3_PV(self, composition: dict, P: float, VF: float,
                   T_guess: float, max_iter: int):
        if not hasattr(self.thermo, 'flash3_PV'):
            raise UnitOperationError(
                f"Flash3 '{self.unit_id}' thermodynamic model does not support PV VLLE flash"
            )
        return self.thermo.flash3_PV(
            composition, P, VF, T_guess=T_guess, max_iter=max_iter
        )

    def _flash3_TV(self, composition: dict, T: float, VF: float,
                   P_guess: float, max_iter: int):
        if not hasattr(self.thermo, 'flash3_TV'):
            raise UnitOperationError(
                f"Flash3 '{self.unit_id}' thermodynamic model does not support TV VLLE flash"
            )
        return self.thermo.flash3_TV(
            composition, T, VF, P_guess=P_guess, max_iter=max_iter
        )

    def _flash3_PH(self, composition: dict, P: float, H_target: float,
                   T_guess: float, max_iter: int):
        if not hasattr(self.thermo, 'flash3_PH'):
            raise UnitOperationError(
                f"Flash3 '{self.unit_id}' thermodynamic model does not support PH VLLE flash"
            )
        return self.thermo.flash3_PH(
            composition, P, H_target, T_guess=T_guess, max_iter=max_iter
        )

    def _find_P_for_H_at_T_flash3(self, composition: dict, T: float,
                                  H_target: float, P_guess: float,
                                  max_iter: int, tol: float) -> float:
        def residual(P_value: float) -> float:
            result = self.thermo.flash3_TP(
                composition, T, P_value, max_iter=max_iter, tol=tol
            )
            return self._flash3_mixture_enthalpy(T, P_value, result) - H_target

        return self._solve_pressure(
            residual,
            P_guess,
            f"Flash3 target enthalpy at {T:.2f} K",
            residual_tolerance=max(1e-6, abs(H_target) * 1e-8),
        )

    def _find_P_for_VF_and_H_flash3(self, composition: dict, VF: float,
                                    H_target: float, P_guess: float,
                                    max_iter: int) -> float:
        def residual(P_value: float) -> float:
            T, result = self._flash3_PV(composition, P_value, VF, 350.0, max_iter)
            return self._flash3_mixture_enthalpy(T, P_value, result) - H_target

        return self._solve_pressure(
            residual,
            P_guess,
            f"Flash3 target enthalpy at vapor fraction {VF:g}",
            residual_tolerance=max(1e-6, abs(H_target) * 1e-8),
        )

    def _flash3_mixture_enthalpy(self, T: float, P: float, result) -> float:
        direct = getattr(self.thermo, '_flash3_mixture_enthalpy', None)
        if direct is not None:
            return direct(T, P, result)
        V = max(0.0, float(result.vapor_fraction))
        L1 = max(0.0, float(result.liquid1_fraction))
        L2 = max(0.0, float(result.liquid2_fraction))
        total = V + L1 + L2
        if total <= 0.0:
            return self.thermo.mixture_enthalpy(result.x1, T, 0.0, result.x1, None, P)
        V /= total
        L1 /= total
        L2 /= total
        h_vapor = self.thermo.mixture_enthalpy(result.y, T, 1.0, None, result.y, P)
        h_liquid1 = self.thermo.mixture_enthalpy(result.x1, T, 0.0, result.x1, None, P)
        h_liquid2 = self.thermo.mixture_enthalpy(result.x2, T, 0.0, result.x2, None, P)
        return V * h_vapor + L1 * h_liquid1 + L2 * h_liquid2

    def _phase_flows(self, total_flow: float, result) -> dict[str, float]:
        fractions = {
            'vapor': max(0.0, float(result.vapor_fraction)),
            'liquid1': max(0.0, float(result.liquid1_fraction)),
            'liquid2': max(0.0, float(result.liquid2_fraction)),
        }
        total = sum(fractions.values())
        if total <= 0.0:
            fractions = {'vapor': 0.0, 'liquid1': 1.0, 'liquid2': 0.0}
            total = 1.0
        return {
            name: total_flow * fraction / total
            for name, fraction in fractions.items()
        }

    def _phase_outlet(self, T: float, P: float, F: float,
                      composition: dict, phase: str,
                      aqueous_context=None) -> StreamState:
        composition = self._normalized_composition(composition)
        try:
            state = self.thermo.calculate_state(
                T,
                P,
                max(float(F), 0.0),
                composition,
                phase=phase,
                flash=False,
                include=('H', 'rho', 'S'),
            )
        except Exception:
            state = StreamState(
                T=T,
                P=P,
                F=max(float(F), 0.0),
                composition=dict(composition),
                vapor_fraction=1.0 if phase == 'vapor' else 0.0,
                liquid1_fraction=0.0 if phase == 'vapor' else 1.0,
                x1=None if phase == 'vapor' else dict(composition),
                fluid_phase_model=getattr(
                    self.thermo, 'fluid_phase_model', 'VLE'
                ),
                phase_status=f'fallback_forced_{phase}',
                phase_stability='explicit_phase_constraint',
            )
            state.MW = self.thermo.mixture_MW(composition)
            try:
                state.H = self.thermo.mixture_enthalpy(
                    composition,
                    T,
                    state.vapor_fraction,
                    None if phase == 'vapor' else composition,
                    composition if phase == 'vapor' else None,
                    P,
                )
            except Exception:
                state.H = None
        state.F = max(float(F), 0.0)
        state.composition = dict(composition)
        if phase == 'vapor':
            state.x = None
            state.x1 = None
            state.y = dict(composition)
            state.vapor_fraction = 1.0
            state.liquid1_fraction = 0.0
        else:
            state.x = dict(composition)
            state.x1 = dict(composition)
            state.y = None
            state.vapor_fraction = 0.0
            state.liquid1_fraction = 1.0
            if aqueous_context is not None:
                state.H = self.thermo.aqueous_liquid_enthalpy(
                    composition, T, P, aqueous_context
                )
        return state

    def _liquid_phase_diagnostics(
        self,
        liquid1: StreamState,
        liquid2: StreamState,
        liquid1_flow: float,
        liquid2_flow: float,
    ) -> dict:
        light, heavy, classifier = self._classify_light_heavy(liquid1, liquid2)
        light_label = 'liquid1' if light is liquid1 else 'liquid2'
        heavy_label = 'liquid1' if heavy is liquid1 else 'liquid2'
        light_mass_density = self._mass_density(light)
        heavy_mass_density = self._mass_density(heavy)
        return {
            'liquid1_molar_density_kmol_m3': liquid1.rho,
            'liquid2_molar_density_kmol_m3': liquid2.rho,
            'liquid1_mass_density_kg_m3': self._mass_density(liquid1),
            'liquid2_mass_density_kg_m3': self._mass_density(liquid2),
            'light_mass_density_kg_m3': light_mass_density,
            'heavy_mass_density_kg_m3': heavy_mass_density,
            'light_liquid': light_label,
            'heavy_liquid': heavy_label,
            'light_fraction': light.F / max(liquid1_flow + liquid2_flow, 1e-300),
            'heavy_fraction': heavy.F / max(liquid1_flow + liquid2_flow, 1e-300),
            'light_flow_kmol_hr': light.F,
            'heavy_flow_kmol_hr': heavy.F,
            'light_composition': dict(light.composition),
            'heavy_composition': dict(heavy.composition),
            'phase_classifier': classifier,
        }

    def _classify_light_heavy(self, phase1: StreamState,
                              phase2: StreamState) -> tuple[StreamState, StreamState, str]:
        density1 = self._mass_density(phase1)
        density2 = self._mass_density(phase2)
        if density1 is not None and density2 is not None:
            tolerance = max(1e-9, 1e-8 * max(abs(density1), abs(density2), 1.0))
            if abs(density1 - density2) > tolerance:
                return (
                    (phase1, phase2, 'mass_density')
                    if density1 < density2
                    else (phase2, phase1, 'mass_density')
                )

        selector = self.get_param(
            'heavy_component',
            self.get_param('heavy_key', self.get_param('heavy_phase_component')),
        )
        if selector is not None:
            selector = str(selector)
            if phase1.composition.get(selector, 0.0) != phase2.composition.get(selector, 0.0):
                return (
                    (phase1, phase2, f"heavy_component:{selector}")
                    if phase1.composition.get(selector, 0.0) < phase2.composition.get(selector, 0.0)
                    else (phase2, phase1, f"heavy_component:{selector}")
                )

        return phase1, phase2, 'lle_phase_order_fallback'

    def _mass_density(self, stream: StreamState) -> float | None:
        if stream.rho is None or stream.MW is None:
            return None
        density = stream.rho * stream.MW
        return density if math.isfinite(density) and density > 0.0 else None

    def _component_balance_residual(self, feed_z: dict, feed_F: float,
                                    outlets: list[StreamState]) -> float:
        scale = max(feed_F, 1.0)
        residual = 0.0
        comps = set(feed_z)
        for outlet in outlets:
            comps.update(outlet.composition)
        for comp in comps:
            inlet_moles = feed_F * feed_z.get(comp, 0.0)
            outlet_moles = sum(outlet.F * outlet.composition.get(comp, 0.0) for outlet in outlets)
            residual = max(residual, abs(outlet_moles - inlet_moles) / scale)
        return residual


class Decanter(UnitOperation):
    """
    Liquid-liquid separator using activity-model LLE.
    
    Params:
        T: Optional operating temperature [°C or K]. If omitted, the decanter
           runs adiabatically at the mixed inlet state.
        P: Optional operating pressure [bar]. Defaults to lowest inlet pressure.
        P_drop: Optional pressure drop from the lowest inlet pressure [bar].
    """
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if not hasattr(self.thermo, 'liquid_liquid_equilibrium'):
            raise UnitOperationError(
                f"Decanter '{self.unit_id}' requires an LLE-capable activity "
                "coefficient thermodynamic model such as NRTL, UNIQUAC, or UNIFAC"
            )

        if not inlets:
            raise UnitOperationError(f"Decanter '{self.unit_id}' has no inlet streams")

        self._validate_inlets(inlets)
        P, pressure_policy = self._outlet_pressure(inlets)
        adjusted_inlets = {
            port: self._let_down_to_pressure(stream, P)
            for port, stream in inlets.items()
        }
        self._reject_vapor(adjusted_inlets.values(), "after pressure adjustment")

        total_F, composition, inlet_enthalpy_flow, T_mixed_guess = self._mixed_feed(adjusted_inlets)
        T_spec = self.get_temperature_param('T')
        if T_spec is None:
            T_spec = self.get_temperature_param('temperature')
        Q_spec = self._heat_duty_spec()
        if T_spec is not None and Q_spec is not None:
            raise UnitOperationError(
                f"Decanter '{self.unit_id}' cannot specify both operating temperature and heat duty"
            )

        mode = str(self.get_param('mode', 'adiabatic')).strip().lower()
        if T_spec is not None:
            T = self._validated_temperature(T_spec)
            thermal_mode = 'specified_temperature'
            unit_heat_duty = None
        elif Q_spec is not None:
            T = self._find_temperature_for_duty(
                P, total_F, composition, inlet_enthalpy_flow, Q_spec, T_mixed_guess
            )
            thermal_mode = 'specified_duty'
            unit_heat_duty = Q_spec
        elif mode in ('isothermal', 'fixed_t', 'fixed-temperature', 'fixed_temperature'):
            T = T_mixed_guess
            thermal_mode = 'isothermal'
            unit_heat_duty = None
        elif mode in ('adiabatic', ''):
            mixed_state = self._state_for_enthalpy(
                composition,
                P,
                total_F,
                inlet_enthalpy_flow / total_F,
                T_mixed_guess,
            )
            self._reject_vapor([mixed_state], "in the mixed adiabatic state")
            T = mixed_state.T
            thermal_mode = 'adiabatic'
            unit_heat_duty = 0.0
        else:
            raise UnitOperationError(
                f"Decanter '{self.unit_id}' mode must be adiabatic or isothermal"
            )

        mixed_state = self.thermo.calculate_state(
            T, P, total_F, composition, phase='liquid', flash=False, include=('H', 'rho')
        )
        self._reject_vapor([mixed_state], "at decanter operating conditions")

        lle_tolerance = float(self.get_param('lle_tolerance', self.get_param('tol', 1e-6)))
        if lle_tolerance <= 0.0:
            raise UnitOperationError(
                f"Decanter '{self.unit_id}' requires positive lle_tolerance"
            )
        max_iter = int(self.get_param('max_iter', self.get_param('lle_max_iter', 100)))
        if max_iter <= 0:
            raise UnitOperationError(
                f"Decanter '{self.unit_id}' requires positive max_iter"
            )

        has_lle, x1, x2, beta = self.thermo.liquid_liquid_equilibrium(
            composition, T, max_iter=max_iter, tol=lle_tolerance
        )
        warnings = []
        
        if not has_lle:
            out = self.thermo.calculate_state(
                T, P, total_F, composition, phase='liquid', flash=False, include=('H', 'rho', 'S')
            )
            empty = self.thermo.calculate_state(
                T, P, 0.0, composition, phase='liquid', flash=False, include=('H', 'rho', 'S')
            )
            empty.F = 0.0
            outlet_enthalpy_flow = total_F * self._stream_enthalpy(out)
            enthalpy_residual = outlet_enthalpy_flow - inlet_enthalpy_flow
            if unit_heat_duty is None:
                unit_heat_duty = enthalpy_residual
            warnings.append("No liquid-liquid split at decanter operating conditions")
            return UnitResult(
                outlet_streams={'light': out, 'heavy': empty},
                heat_duty=unit_heat_duty,
                performance={
                    'mode': thermal_mode,
                    'two_phases': False,
                    'phase_count': 1,
                    'no_lle': True,
                    'T_C': T - 273.15,
                    'P_bar': P,
                    'n_inlets': len(inlets),
                    'pressure_policy': pressure_policy,
                    'light_fraction': 1.0,
                    'heavy_fraction': 0.0,
                    'light_flow_kmol_hr': total_F,
                    'heavy_flow_kmol_hr': 0.0,
                    'light_composition': dict(composition),
                    'heavy_composition': dict(composition),
                    'light_mass_density_kg_m3': self._mass_density(out),
                    'heavy_mass_density_kg_m3': None,
                    'duty_kW': unit_heat_duty / 3600.0,
                    'outlet_enthalpy_residual_kW': enthalpy_residual / 3600.0,
                    'component_balance_residual': 0.0,
                },
                warnings=warnings
            )
        
        beta = max(0.0, min(1.0, float(beta)))
        phase1 = self._phase_state(T, P, total_F * (1.0 - beta), x1)
        phase2 = self._phase_state(T, P, total_F * beta, x2)
        light, heavy, classifier = self._classify_light_heavy(phase1, phase2)

        outlet_enthalpy_flow = (
            light.F * self._stream_enthalpy(light)
            + heavy.F * self._stream_enthalpy(heavy)
        )
        enthalpy_residual = outlet_enthalpy_flow - inlet_enthalpy_flow
        if unit_heat_duty is None:
            unit_heat_duty = enthalpy_residual

        component_residual = self._component_balance_residual(composition, total_F, [light, heavy])
        if abs(component_residual) > 1e-7:
            warnings.append(
                f"Decanter component balance residual is {component_residual:.3g}"
            )
        if thermal_mode == 'adiabatic' and abs(enthalpy_residual) > max(1e-3, abs(inlet_enthalpy_flow) * 1e-8):
            warnings.append(
                "Adiabatic decanter outlet enthalpy differs from inlet enthalpy; "
                "activity-model mixture enthalpy is not fully LLE phase-weighted"
            )

        return UnitResult(
            outlet_streams={'light': light, 'heavy': heavy},
            heat_duty=unit_heat_duty,
            performance={
                'mode': thermal_mode,
                'two_phases': True,
                'phase_count': 2,
                'no_lle': False,
                'T_C': T - 273.15,
                'P_bar': P,
                'n_inlets': len(inlets),
                'pressure_policy': pressure_policy,
                'lle_beta_phase2': beta,
                'light_fraction': light.F / total_F,
                'heavy_fraction': heavy.F / total_F,
                'light_flow_kmol_hr': light.F,
                'heavy_flow_kmol_hr': heavy.F,
                'light_composition': dict(light.composition),
                'heavy_composition': dict(heavy.composition),
                'light_molar_density_kmol_m3': light.rho,
                'heavy_molar_density_kmol_m3': heavy.rho,
                'light_mass_density_kg_m3': self._mass_density(light),
                'heavy_mass_density_kg_m3': self._mass_density(heavy),
                'phase_classifier': classifier,
                'duty_kW': unit_heat_duty / 3600.0,
                'outlet_enthalpy_residual_kW': enthalpy_residual / 3600.0,
                'component_balance_residual': component_residual,
            },
            warnings=warnings,
        )

    def _validate_inlets(self, inlets: dict[str, StreamState]) -> None:
        for port, stream in inlets.items():
            if stream.F < 0.0:
                raise UnitOperationError(
                    f"Decanter '{self.unit_id}' inlet '{port}' has negative flow"
                )
            if not math.isfinite(stream.F):
                raise UnitOperationError(
                    f"Decanter '{self.unit_id}' inlet '{port}' flow must be finite"
                )
            if not math.isfinite(stream.T) or stream.T <= 0.0:
                raise UnitOperationError(
                    f"Decanter '{self.unit_id}' inlet '{port}' temperature must be positive"
                )
            if not math.isfinite(stream.P) or stream.P <= 0.0:
                raise UnitOperationError(
                    f"Decanter '{self.unit_id}' inlet '{port}' pressure must be positive"
                )
            if stream.vapor_fraction > 1e-8:
                raise UnitOperationError(
                    f"Decanter '{self.unit_id}' received vapor on inlet '{port}'. "
                    "Use Flash3/ThreePhaseFlash for vapor-liquid-liquid separation."
                )
        if sum(max(stream.F, 0.0) for stream in inlets.values()) <= 0.0:
            raise UnitOperationError(f"Decanter '{self.unit_id}' requires positive total inlet flow")

    def _validated_temperature(self, T: float) -> float:
        T = float(T)
        if not math.isfinite(T) or T <= 0.0:
            raise UnitOperationError(
                f"Decanter '{self.unit_id}' operating temperature must be positive"
            )
        return T

    def _outlet_pressure(self, inlets: dict[str, StreamState]) -> tuple[float, str]:
        P_drop = float(self.get_param('P_drop', 0.0))
        if not math.isfinite(P_drop) or P_drop < 0.0:
            raise UnitOperationError(
                f"Decanter '{self.unit_id}' requires nonnegative P_drop"
            )
        explicit = self.get_param('P', self.get_param('pressure', self.get_param('P_out')))
        min_P = min(stream.P for stream in inlets.values())
        if explicit is None:
            P = min_P - P_drop
            policy = 'auto_lowest_inlet_pressure'
        else:
            P = float(explicit)
            policy = 'specified_pressure'
            if P > min_P + 1e-9:
                raise UnitOperationError(
                    f"Decanter '{self.unit_id}' operating pressure ({P:g} bar) is above "
                    f"the lowest inlet pressure ({min_P:g} bar); add pressure-changing "
                    "equipment upstream"
                )
        if not math.isfinite(P) or P <= 0.0:
            raise UnitOperationError(
                f"Decanter '{self.unit_id}' operating pressure must be positive"
            )
        return P, policy

    def _reject_vapor(self, streams, context: str) -> None:
        max_vf = max((getattr(stream, 'vapor_fraction', 0.0) for stream in streams), default=0.0)
        if max_vf > 1e-8:
            raise UnitOperationError(
                f"Decanter '{self.unit_id}' found vapor {context}. "
                "Use Flash3/ThreePhaseFlash for vapor-liquid-liquid separation."
            )

    def _mixed_feed(self, inlets: dict[str, StreamState]) -> tuple[float, dict, float, float]:
        total_F = 0.0
        total_H = 0.0
        component_moles = {}
        weighted_T = 0.0
        for stream in inlets.values():
            if stream.F <= 0.0:
                continue
            total_F += stream.F
            weighted_T += stream.F * stream.T
            total_H += stream.F * self._stream_enthalpy(stream)
            for comp, frac in stream.composition.items():
                value = max(float(frac), 0.0)
                component_moles[comp] = component_moles.get(comp, 0.0) + stream.F * value

        if total_F <= 0.0:
            raise UnitOperationError(f"Decanter '{self.unit_id}' requires positive total inlet flow")
        comp_total = sum(component_moles.values())
        if comp_total <= 0.0:
            raise UnitOperationError(f"Decanter '{self.unit_id}' feed composition cannot be empty")
        composition = {
            comp: value / comp_total
            for comp, value in component_moles.items()
            if value > 0.0
        }
        return total_F, composition, total_H, weighted_T / total_F

    def _stream_enthalpy(self, stream: StreamState) -> float:
        if stream.H is not None:
            return stream.H
        return self.thermo.mixture_enthalpy(
            stream.composition,
            stream.T,
            stream.vapor_fraction,
            stream.x,
            stream.y,
            stream.P,
        )

    def _let_down_to_pressure(self, stream: StreamState, P: float) -> StreamState:
        if stream.F <= 0.0 or abs(stream.P - P) <= 1e-9:
            return stream.copy()
        if P > stream.P + 1e-9:
            raise UnitOperationError(
                f"Decanter '{self.unit_id}' cannot raise inlet pressure from "
                f"{stream.P:g} to {P:g} bar"
            )
        return self._state_for_enthalpy(
            stream.composition,
            P,
            stream.F,
            self._stream_enthalpy(stream),
            stream.T,
        )

    def _state_for_enthalpy(self, composition: dict, P: float, F: float,
                            H_target: float, T_guess: float) -> StreamState:
        state, residual = _ThermoStateSolver(
            self.thermo,
            f"Decanter '{self.unit_id}'",
        ).state_at_enthalpy(
            P,
            F,
            composition,
            H_target,
            T_guess,
            include=('H', 'rho'),
            force_phase='liquid',
        )
        tolerance = max(1e-6, abs(H_target) * 1e-8)
        if abs(residual) > tolerance:
            raise UnitOperationError(
                f"Decanter '{self.unit_id}' liquid enthalpy solve residual is "
                f"{residual:.4g} kJ/kmol"
            )
        return state

    def _phase_state(self, T: float, P: float, F: float, composition: dict) -> StreamState:
        return self.thermo.calculate_state(
            T,
            P,
            max(float(F), 0.0),
            self._normalize_composition(composition),
            phase='liquid',
            flash=False,
            include=('H', 'rho', 'S'),
        )

    def _classify_light_heavy(self, phase1: StreamState,
                              phase2: StreamState) -> tuple[StreamState, StreamState, str]:
        density1 = self._mass_density(phase1)
        density2 = self._mass_density(phase2)
        if density1 is not None and density2 is not None:
            tolerance = max(1e-9, 1e-8 * max(abs(density1), abs(density2), 1.0))
            if abs(density1 - density2) > tolerance:
                return (
                    (phase1, phase2, 'mass_density')
                    if density1 < density2
                    else (phase2, phase1, 'mass_density')
                )

        selector = self.get_param(
            'heavy_component',
            self.get_param('heavy_key', self.get_param('heavy_phase_component')),
        )
        if selector is not None:
            selector = str(selector)
            if phase1.composition.get(selector, 0.0) != phase2.composition.get(selector, 0.0):
                return (
                    (phase1, phase2, f"heavy_component:{selector}")
                    if phase1.composition.get(selector, 0.0) < phase2.composition.get(selector, 0.0)
                    else (phase2, phase1, f"heavy_component:{selector}")
                )

        return phase1, phase2, 'lle_phase_order_fallback'

    def _mass_density(self, stream: StreamState) -> float | None:
        if stream.rho is None or stream.MW is None:
            return None
        density = stream.rho * stream.MW
        return density if math.isfinite(density) and density > 0.0 else None

    def _component_balance_residual(self, feed_z: dict, feed_F: float,
                                    outlets: list[StreamState]) -> float:
        scale = max(feed_F, 1.0)
        residual = 0.0
        comps = set(feed_z)
        for outlet in outlets:
            comps.update(outlet.composition)
        for comp in comps:
            inlet_moles = feed_F * feed_z.get(comp, 0.0)
            outlet_moles = sum(outlet.F * outlet.composition.get(comp, 0.0) for outlet in outlets)
            residual = max(residual, abs(outlet_moles - inlet_moles) / scale)
        return residual

    @staticmethod
    def _normalize_composition(composition: dict) -> dict:
        total = sum(max(float(value), 0.0) for value in composition.values())
        if total <= 0.0:
            raise UnitOperationError("Decanter feed composition cannot be empty")
        return {
            comp: max(float(value), 0.0) / total
            for comp, value in composition.items()
            if max(float(value), 0.0) > 0.0
        }

    def _heat_duty_spec(self):
        for name in ('Q', 'duty', 'heat_duty'):
            value = self.get_param(name)
            if value is None:
                continue
            duty = float(value)
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
            return duty * 3600.0 if abs(duty) < 1e6 else duty
        return None

    def _find_temperature_for_duty(self, P: float, F: float, composition: dict,
                                   inlet_enthalpy_flow: float, Q: float,
                                   T_guess: float) -> float:
        target = inlet_enthalpy_flow + Q

        def residual(T_value: float) -> float:
            T_value = self._validated_temperature(T_value)
            has_lle, x1, x2, beta = self.thermo.liquid_liquid_equilibrium(composition, T_value)
            if not has_lle:
                state = self.thermo.calculate_state(
                    T_value, P, F, composition, phase='liquid', flash=False, include=('H',)
                )
                return F * self._stream_enthalpy(state) - target
            phase1 = self._phase_state(T_value, P, F * (1.0 - beta), x1)
            phase2 = self._phase_state(T_value, P, F * beta, x2)
            return (
                phase1.F * self._stream_enthalpy(phase1)
                + phase2.F * self._stream_enthalpy(phase2)
                - target
            )

        candidates = sorted(set(
            max(1.0, float(T))
            for T in (
                T_guess,
                T_guess - 100.0,
                T_guess + 100.0,
                250.0,
                298.15,
                350.0,
                450.0,
                600.0,
                900.0,
            )
        ))
        evaluated = []
        for T in candidates:
            try:
                value = residual(T)
                if math.isfinite(value):
                    evaluated.append((T, value))
            except Exception:
                continue
        for T, value in evaluated:
            if abs(value) <= max(1e-3, abs(target) * 1e-8):
                return T
        from scipy.optimize import brentq
        for (T1, f1), (T2, f2) in zip(evaluated, evaluated[1:]):
            if f1 * f2 < 0.0:
                return brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=100)
        raise UnitOperationError(
            f"Decanter '{self.unit_id}' could not bracket an operating temperature "
            "for the specified heat duty"
        )


class MolecularSieveDryer(UnitOperation):
    """
    Selective adsorption dryer for removing trace water from liquid products.

    Inlets: feed
    Outlets: product, adsorbate

    Params:
        water_component: Component name to remove (default: water/H2O if present)
        sieve_type: Molecular sieve type (currently only 3A)
        adsorbent_mass_flow: Fresh/regenerated sieve supplied [kg/h]
        target_water_mole_fraction: Water mole fraction in product
        removal_fraction: Optional direct fraction of inlet water removed
    """

    R = 8.314462618
    WATER_MW_KG_PER_KMOL = 18.01528
    GSTA_3A_WATER = {
        'qmax_kg_per_kg': 0.21,
        'P0_bar': 1.0,
        'dH_J_per_mol': [-46.60e3, -125.0e3, -193.6e3, -272.2e3],
        'dS_J_per_mol_K': [-53.70, -221.1, -356.7, -567.5],
    }
    TARGET_WATER_NAMES = ('target_water_mole_fraction',)
    REMOVAL_FRACTION_NAMES = ('removal_fraction',)
    ADSORBENT_MASS_NAMES = (
        'adsorbent_mass_flow',
        'sieve_mass_flow',
        'molecular_sieve_mass_flow',
        'adsorbent_flow',
        'sieve_flow',
        'adsorbent_mass',
        'sieve_mass',
    )

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        inlet = inlets.get('feed') or inlets.get('in') or list(inlets.values())[0]
        comps = list(inlet.composition.keys())

        water_component = self.get_param('water_component')
        if water_component is None:
            for candidate in ('water', 'H2O'):
                if candidate in inlet.composition:
                    water_component = candidate
                    break
        if water_component not in inlet.composition:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' could not find a water component"
            )

        explicit_specs = []
        if self._any_param_present(self.TARGET_WATER_NAMES):
            explicit_specs.append('target_water_mole_fraction')
        if self._any_param_present(self.REMOVAL_FRACTION_NAMES):
            explicit_specs.append('removal_fraction')
        if self._any_param_present(self.ADSORBENT_MASS_NAMES):
            explicit_specs.append('adsorbent_mass_flow')
        if len(explicit_specs) > 1:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' requires target water fraction, "
                "removal fraction, and adsorbent mass flow to be mutually exclusive "
                f"(specified: {', '.join(explicit_specs)})"
            )

        target = float(self.get_param('target_water_mole_fraction', 1e-4))
        if target < 0.0:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' requires a nonnegative target water fraction"
            )
        if target >= 1.0:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' requires target water mole fraction < 1"
            )
        removal_fraction = self.get_param('removal_fraction')

        inlet_moles = {
            comp: inlet.F * inlet.composition.get(comp, 0.0)
            for comp in comps
        }
        water_in = inlet_moles.get(water_component, 0.0)
        dry_moles = sum(value for comp, value in inlet_moles.items() if comp != water_component)

        adsorbent_rate = self._adsorbent_mass_flow_kg_h()
        if adsorbent_rate is not None:
            return self._solve_3a_equilibrium_dryer(
                inlet,
                comps,
                inlet_moles,
                water_component,
                water_in,
                dry_moles,
                adsorbent_rate,
            )

        if removal_fraction is not None:
            removal_fraction = float(removal_fraction)
            if not 0.0 <= removal_fraction <= 1.0:
                raise UnitOperationError(
                    f"MolecularSieveDryer '{self.unit_id}' requires removal_fraction between 0 and 1"
                )
            desired_water_out = water_in * (1.0 - removal_fraction)
            return self._size_3a_equilibrium_dryer(
                inlet,
                comps,
                inlet_moles,
                water_component,
                water_in,
                dry_moles,
                desired_water_out,
            )
        elif target <= 0.0:
            desired_water_out = 0.0
        else:
            inlet_water_fraction = water_in / max(water_in + dry_moles, 1e-12)
            if target >= inlet_water_fraction - 1e-12:
                desired_water_out = water_in
            elif dry_moles <= 0.0 and water_in > 0.0:
                raise UnitOperationError(
                    f"MolecularSieveDryer '{self.unit_id}' cannot reach a finite target "
                    "water mole fraction with no dry carrier components"
                )
            else:
                desired_water_out = target * dry_moles / max(1.0 - target, 1e-12)

        return self._size_3a_equilibrium_dryer(
            inlet,
            comps,
            inlet_moles,
            water_component,
            water_in,
            dry_moles,
            desired_water_out,
        )

    def _simple_dryer_result(
        self,
        inlet: StreamState,
        comps: list[str],
        inlet_moles: dict[str, float],
        water_component: str,
        water_in: float,
        water_removed: float,
        *,
        product_phase: str,
        adsorbent_rate: float | None = None,
        initial_loading: float | None = None,
        final_loading: float | None = None,
        qeq: float | None = None,
        driver: float | None = None,
        driver_basis: str | None = None,
        reduced_pressure: float | None = None,
    ) -> UnitResult:
        water_removed = max(0.0, min(water_removed, water_in))
        product_moles = dict(inlet_moles)
        product_moles[water_component] = water_in - water_removed
        product_total = sum(product_moles.values())
        if product_total <= 0.0:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' removed the entire stream"
            )
        product_comp = {
            comp: max(value, 0.0) / product_total
            for comp, value in product_moles.items()
            if value > 0.0
        }

        adsorbate_total = water_removed
        adsorbate_comp = {water_component: 1.0}
        if adsorbate_total <= 0.0:
            adsorbate_total = 0.0

        product = self.thermo.calculate_state(
            inlet.T, inlet.P, product_total, product_comp,
            phase=product_phase,
            flash=False,
        )
        adsorbate = self.thermo.calculate_state(
            inlet.T, inlet.P, max(adsorbate_total, 1e-12), adsorbate_comp,
            phase='liquid',
            flash=False,
        )
        adsorbate.F = adsorbate_total

        performance = {
            'water_component': water_component,
            'water_removed_kmol_h': water_removed,
            'water_removal_fraction': water_removed / max(water_in, 1e-12),
            'product_water_mole_fraction': product.composition.get(water_component, 0.0),
        }
        heat_duty = 0.0
        if adsorbent_rate is not None:
            heat_kJ_per_mol = self._gsta_3a_water_heat_release_kJ_per_mol(
                reduced_pressure or 0.0,
                inlet.T,
            )
            heat_release_kJ_h = water_removed * heat_kJ_per_mol * 1000.0
            stream_enthalpy_duty_kJ_h = (
                product.F * (product.H or 0.0)
                + adsorbate.F * (adsorbate.H or 0.0)
                - inlet.F * (inlet.H or 0.0)
            )
            heat_duty = stream_enthalpy_duty_kJ_h
            performance.update({
                'sieve_type': '3A',
                'mode': 'equilibrium_isothermal_pseudo_continuous',
                'adsorbent_mass_flow_kg_h': adsorbent_rate,
                'initial_loading_kg_water_per_kg_sieve': initial_loading,
                'final_loading_kg_water_per_kg_sieve': final_loading,
                'equilibrium_loading_kg_water_per_kg_sieve': qeq,
                'water_removed_kg_h': water_removed * self._component_mw_kg_per_kmol(water_component),
                'water_activity_or_fugacity_driver': driver,
                'adsorption_reduced_pressure': reduced_pressure,
                'driver_basis': driver_basis,
                'heat_of_adsorption_kJ_per_mol_water': heat_kJ_per_mol,
                'heat_release_kJ_h': heat_release_kJ_h,
                'adsorption_heat_release_kJ_h': heat_release_kJ_h,
                'stream_enthalpy_duty_kJ_h': stream_enthalpy_duty_kJ_h,
            })

        return UnitResult(
            outlet_streams={'product': product, 'adsorbate': adsorbate},
            heat_duty=heat_duty,
            performance=performance,
        )

    def _param_present(self, name: str) -> bool:
        if name in self.params:
            return True
        name_lower = name.lower()
        return any(key.lower() == name_lower for key in self.params)

    def _any_param_present(self, names: tuple[str, ...]) -> bool:
        return any(self._param_present(name) for name in names)

    def _adsorbent_mass_flow_kg_h(self):
        for name in self.ADSORBENT_MASS_NAMES:
            value = self.get_param(name)
            if value is not None:
                value = float(value)
                unit = (self.get_param_unit(name) or '').lower()
                if 'lb' in unit:
                    value *= 0.45359237
                if value < 0.0:
                    raise UnitOperationError(
                        f"MolecularSieveDryer '{self.unit_id}' requires nonnegative adsorbent mass flow"
                    )
                return value
        return None

    def _normalized_sieve_type(self) -> str:
        sieve_type = self.get_param('sieve_type', self.get_param('molecular_sieve', '3A'))
        normalized = str(sieve_type).strip().upper().replace('-', '').replace('_', '').replace(' ', '')
        normalized = normalized.replace('Å', 'A').replace('ANGSTROMS', 'A').replace('ANGSTROM', 'A')
        return normalized

    def _solve_3a_equilibrium_dryer(
        self,
        inlet: StreamState,
        comps: list[str],
        inlet_moles: dict[str, float],
        water_component: str,
        water_in: float,
        dry_moles: float,
        adsorbent_rate: float,
    ) -> UnitResult:
        if self._normalized_sieve_type() != '3A':
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' currently supports only 3A molecular sieve"
            )
        if adsorbent_rate <= 0.0:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' requires positive adsorbent mass flow"
            )

        initial_loading = float(self.get_param('initial_loading_kg_per_kg', self.get_param('initial_loading', 0.0)))
        if initial_loading < 0.0:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' requires nonnegative initial loading"
            )
        qmax = self.GSTA_3A_WATER['qmax_kg_per_kg']
        if initial_loading > qmax + 1e-12:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' initial loading exceeds the "
                f"3A maximum loading ({qmax:g} kg water/kg sieve)"
            )

        water_mw = self._component_mw_kg_per_kmol(water_component)

        def composition_for(water_out: float) -> dict[str, float]:
            product_moles = dict(inlet_moles)
            product_moles[water_component] = max(water_out, 0.0)
            total = sum(max(value, 0.0) for value in product_moles.values())
            if total <= 0.0:
                return {water_component: 1.0}
            return {
                comp: max(value, 0.0) / total
                for comp, value in product_moles.items()
                if value > 0.0
            }

        def equilibrium_loading(water_out: float) -> tuple[float, float, str, float]:
            product_comp = composition_for(water_out)
            z, driver, driver_basis = self._water_adsorption_reduced_pressure(
                water_component,
                product_comp,
                inlet.T,
                inlet.P,
                inlet.vapor_fraction,
            )
            return self._gsta_3a_water_loading(z, inlet.T), driver, driver_basis, z

        qeq_in, _, _, _ = equilibrium_loading(water_in)
        if initial_loading > qeq_in + 1e-12:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' supplied 3A sieve is too wet "
                f"for equilibrium adsorption at the inlet conditions "
                f"(initial loading {initial_loading:g} kg/kg > equilibrium "
                f"{qeq_in:g} kg/kg)"
            )
        if water_in <= 0.0 or initial_loading >= qeq_in:
            water_out = water_in
            qeq, driver, driver_basis, reduced_pressure = qeq_in, *equilibrium_loading(water_in)[1:]
        else:
            low = 0.0 if dry_moles > 0.0 else max(water_in * 1e-12, 0.0)
            high = water_in

            def residual(water_out: float) -> float:
                water_removed = water_in - water_out
                q_final = initial_loading + water_removed * water_mw / adsorbent_rate
                qeq, _, _, _ = equilibrium_loading(water_out)
                return q_final - qeq

            if residual(low) <= 0.0:
                water_out = low
            else:
                from scipy.optimize import brentq
                water_out = brentq(residual, low, high, xtol=1e-12, rtol=1e-12, maxiter=100)
            qeq, driver, driver_basis, reduced_pressure = equilibrium_loading(water_out)

        water_removed = max(0.0, min(water_in - water_out, water_in))
        final_loading = initial_loading + water_removed * water_mw / adsorbent_rate
        product_phase = 'liquid' if inlet.vapor_fraction < 0.5 else 'vapor'

        return self._simple_dryer_result(
            inlet,
            comps,
            inlet_moles,
            water_component,
            water_in,
            water_removed,
            product_phase=product_phase,
            adsorbent_rate=adsorbent_rate,
            initial_loading=initial_loading,
            final_loading=final_loading,
            qeq=qeq,
            driver=driver,
            driver_basis=driver_basis,
            reduced_pressure=reduced_pressure,
        )

    def _size_3a_equilibrium_dryer(
        self,
        inlet: StreamState,
        comps: list[str],
        inlet_moles: dict[str, float],
        water_component: str,
        water_in: float,
        dry_moles: float,
        desired_water_out: float,
    ) -> UnitResult:
        if self._normalized_sieve_type() != '3A':
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' currently supports only 3A molecular sieve"
            )

        initial_loading = float(self.get_param('initial_loading_kg_per_kg', self.get_param('initial_loading', 0.0)))
        if initial_loading < 0.0:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' requires nonnegative initial loading"
            )
        qmax = self.GSTA_3A_WATER['qmax_kg_per_kg']
        if initial_loading > qmax + 1e-12:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' initial loading exceeds the "
                f"3A maximum loading ({qmax:g} kg water/kg sieve)"
            )

        if desired_water_out < -1e-12:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' requested negative product water flow"
            )
        if desired_water_out > water_in + 1e-12:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' requested outlet water would require adding water"
            )
        desired_water_out = max(0.0, min(float(desired_water_out), water_in))
        water_removed = water_in - desired_water_out

        water_mw = self._component_mw_kg_per_kmol(water_component)
        product_phase = 'liquid' if inlet.vapor_fraction < 0.5 else 'vapor'

        product_comp = self._composition_for_water_out(inlet_moles, water_component, desired_water_out)
        qeq, driver, driver_basis, reduced_pressure = self._equilibrium_3a_water_loading(
            water_component,
            product_comp,
            inlet.T,
            inlet.P,
            inlet.vapor_fraction,
        )

        if water_removed <= 1e-14:
            return self._simple_dryer_result(
                inlet,
                comps,
                inlet_moles,
                water_component,
                water_in,
                0.0,
                product_phase=product_phase,
                adsorbent_rate=0.0,
                initial_loading=initial_loading,
                final_loading=initial_loading,
                qeq=qeq,
                driver=driver,
                driver_basis=driver_basis,
                reduced_pressure=reduced_pressure,
            )

        working_capacity = qeq - initial_loading
        if working_capacity <= 1e-12:
            raise UnitOperationError(
                f"MolecularSieveDryer '{self.unit_id}' requested outlet water is "
                "thermodynamically impossible for 3A equilibrium adsorption at the "
                "specified temperature, pressure, and initial loading"
            )

        adsorbent_rate = water_removed * water_mw / working_capacity
        return self._simple_dryer_result(
            inlet,
            comps,
            inlet_moles,
            water_component,
            water_in,
            water_removed,
            product_phase=product_phase,
            adsorbent_rate=adsorbent_rate,
            initial_loading=initial_loading,
            final_loading=qeq,
            qeq=qeq,
            driver=driver,
            driver_basis=driver_basis,
            reduced_pressure=reduced_pressure,
        )

    def _composition_for_water_out(
        self,
        inlet_moles: dict[str, float],
        water_component: str,
        water_out: float,
    ) -> dict[str, float]:
        product_moles = dict(inlet_moles)
        product_moles[water_component] = max(water_out, 0.0)
        total = sum(max(value, 0.0) for value in product_moles.values())
        if total <= 0.0:
            return {water_component: 1.0}
        return {
            comp: max(value, 0.0) / total
            for comp, value in product_moles.items()
            if value > 0.0
        }

    def _equilibrium_3a_water_loading(
        self,
        water_component: str,
        composition: dict[str, float],
        T: float,
        P: float,
        vapor_fraction: float,
    ) -> tuple[float, float, str, float]:
        z, driver, driver_basis = self._water_adsorption_reduced_pressure(
            water_component,
            composition,
            T,
            P,
            vapor_fraction,
        )
        return self._gsta_3a_water_loading(z, T), driver, driver_basis, z

    def _component_mw_kg_per_kmol(self, comp: str) -> float:
        props = self.thermo.props.get(comp)
        mw = getattr(props, 'MW', None)
        if mw and mw > 0.0:
            return float(mw)
        return self.WATER_MW_KG_PER_KMOL

    def _water_adsorption_reduced_pressure(
        self,
        water_component: str,
        composition: dict[str, float],
        T: float,
        P: float,
        vapor_fraction: float,
    ) -> tuple[float, float, str]:
        xw = max(float(composition.get(water_component, 0.0)), 0.0)
        p0 = self.GSTA_3A_WATER['P0_bar']
        if vapor_fraction >= 0.5:
            phi = 1.0
            if hasattr(self.thermo, 'fugacity_coefficients'):
                try:
                    phi_values = self.thermo.fugacity_coefficients(T, P, composition, 'vapor')
                    phi = max(float(phi_values.get(water_component, 1.0)), 1e-12)
                except Exception:
                    phi = 1.0
            fugacity_bar = max(phi * xw * P, 0.0)
            return fugacity_bar / p0, fugacity_bar, 'vapor_fugacity_bar'

        gamma = 1.0
        if hasattr(self.thermo, 'activity_coefficients'):
            try:
                gamma_values = self.thermo.activity_coefficients(T, composition)
                gamma = max(float(gamma_values.get(water_component, 1.0)), 0.0)
            except Exception:
                gamma = 1.0
        activity = max(gamma * xw, 0.0)
        try:
            psat_bar = max(float(self.thermo.Psat(water_component, T)), 0.0)
        except Exception:
            psat_bar = 0.03169
        return activity * psat_bar / p0, activity, 'liquid_water_activity'

    def _gsta_3a_water_terms(self, z: float, T: float) -> list[float]:
        data = self.GSTA_3A_WATER
        z = max(float(z), 0.0)
        terms = []
        for index, (dH, dS) in enumerate(zip(data['dH_J_per_mol'], data['dS_J_per_mol_K']), start=1):
            try:
                lnK = -dH / (self.R * T) + dS / self.R
                terms.append(math.exp(max(min(lnK, 700.0), -700.0)) * z**index)
            except (ValueError, OverflowError):
                terms.append(0.0)
        return terms

    def _gsta_3a_water_loading(self, z: float, T: float) -> float:
        data = self.GSTA_3A_WATER
        terms = self._gsta_3a_water_terms(z, T)
        denom = 1.0 + sum(terms)
        if denom <= 0.0:
            return 0.0
        numerator = sum((index + 1) * term for index, term in enumerate(terms))
        return max(0.0, min(data['qmax_kg_per_kg'], data['qmax_kg_per_kg'] / 4.0 * numerator / denom))

    def _gsta_3a_water_heat_release_kJ_per_mol(self, reduced_pressure: float, T: float) -> float:
        data = self.GSTA_3A_WATER
        if reduced_pressure <= 0.0:
            return -data['dH_J_per_mol'][0] / 1000.0
        terms = self._gsta_3a_water_terms(reduced_pressure, T)
        adsorbed_moles_weight = sum((index + 1) * term for index, term in enumerate(terms))
        if adsorbed_moles_weight <= 0.0:
            return -data['dH_J_per_mol'][0] / 1000.0
        enthalpy_weight = sum(term * dH for term, dH in zip(terms, data['dH_J_per_mol']))
        return max(0.0, -enthalpy_weight / adsorbed_moles_weight / 1000.0)


class ShortcutExtractor(UnitOperation):
    """
    Multi-stage counter-current liquid-liquid extractor.
    
    Inlets: feed, solvent
    Outlets: raffinate, extract
    
    Params:
        N_stages: Number of equilibrium stages
        T: Operating temperature [°C or K]
    """
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if not inlets:
            raise UnitOperationError(f"Extractor '{self.unit_id}' needs feed and solvent")
        feed = inlets.get('feed') or inlets.get('in') or list(inlets.values())[0]
        solvent = inlets.get('solvent') or (list(inlets.values())[1] if len(inlets) > 1 else None)
        
        if solvent is None:
            raise UnitOperationError(f"Extractor '{self.unit_id}' needs feed and solvent")
        
        N = int(self.get_param('N_stages', 5))
        T = self.get_param('T', (feed.T + solvent.T) / 2)
        if T is not None:
            T = float(T)
            if T < 200: T += 273.15
        
        P = feed.P
        
        if not hasattr(self.thermo, 'liquid_liquid_equilibrium'):
            raise UnitOperationError(
                f"Extractor '{self.unit_id}' requires an LLE-capable activity model"
            )
        
        comps = list(set(list(feed.composition.keys()) + list(solvent.composition.keys())))
        total_F = feed.F + solvent.F
        z_mix = {c: (feed.F*feed.composition.get(c,0) + solvent.F*solvent.composition.get(c,0))/total_F for c in comps}
        
        has_lle, x1, x2, beta = self.thermo.liquid_liquid_equilibrium(z_mix, T)
        
        if not has_lle:
            out = self.thermo.calculate_state(T, P, total_F, z_mix, phase='liquid', flash=False)
            empty = self.thermo.calculate_state(T, P, 0.0, z_mix, phase='liquid', flash=False)
            empty.F = 0.0
            return UnitResult(
                outlet_streams={'raffinate': out, 'extract': empty},
                warnings=["No LLE at operating conditions"]
            )
        
        solvent_keys = [
            c for c in comps
            if solvent.composition.get(c, 0.0) > feed.composition.get(c, 0.0)
        ]
        score_1 = sum(x1.get(c, 0.0) for c in solvent_keys)
        score_2 = sum(x2.get(c, 0.0) for c in solvent_keys)
        x_ext_eq, x_raf_eq = (x1, x2) if score_1 >= score_2 else (x2, x1)

        # Distribution coefficients relative to the solvent-rich extract phase.
        K_dist = {
            c: (x_ext_eq.get(c, 0.0) / max(x_raf_eq.get(c, 0.0), 1e-10))
            for c in comps
        }
        S_over_F = solvent.F / max(feed.F, 1e-10)
        
        extraction_fractions = {}
        x_raf, x_ext = {}, {}
        for c in comps:
            E = S_over_F * K_dist.get(c, 1.0)
            if abs(E - 1) > 0.01:
                denominator = E**(N+1) - 1.0
                frac_ext = (E**(N+1) - E) / denominator if abs(denominator) > 1e-12 else 0.0
            else:
                frac_ext = N / (N + 1)
            frac_ext = max(0, min(1, frac_ext))
            extraction_fractions[c] = frac_ext
            
            feed_mol = feed.F * feed.composition.get(c, 0)
            solv_mol = solvent.F * solvent.composition.get(c, 0)
            x_ext[c] = feed_mol * frac_ext + solv_mol
            x_raf[c] = feed_mol * (1-frac_ext)
        
        F_raf, F_ext = sum(x_raf.values()), sum(x_ext.values())
        if F_raf > 0: x_raf = {c: v/F_raf for c,v in x_raf.items()}
        if F_ext > 0: x_ext = {c: v/F_ext for c,v in x_ext.items()}
        
        raf = self.thermo.calculate_state(T, P, F_raf, x_raf, phase='liquid', flash=False)
        ext = self.thermo.calculate_state(T, P, F_ext, x_ext, phase='liquid', flash=False)
        
        return UnitResult(
            outlet_streams={'raffinate': raf, 'extract': ext},
            performance={
                'N_stages': N,
                'T_C': T-273.15,
                'two_phases': True,
                'K_dist': K_dist,
                'extraction_fractions': extraction_fractions,
            }
        )


LiquidLiquidExtractor = ShortcutExtractor


class RigorousLiquidLiquidExtractor(UnitOperation):
    """
    Counter-current equilibrium-stage liquid-liquid extractor.

    Inlets: feed, solvent
    Outlets: raffinate, extract

    Each stage solves component material balances and equal liquid-phase
    activities for both exiting liquid phases.
    """

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        if not inlets:
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' needs feed and solvent"
            )
        feed = inlets.get('feed') or inlets.get('in') or list(inlets.values())[0]
        solvent = inlets.get('solvent') or (list(inlets.values())[1] if len(inlets) > 1 else None)
        if solvent is None:
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' needs feed and solvent"
            )
        if not hasattr(self.thermo, 'liquid_liquid_equilibrium'):
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' requires an LLE-capable activity model"
            )

        N = int(self.get_param('N_stages', 5))
        if N < 1:
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' requires N_stages >= 1"
            )
        T = self.get_param('T', (feed.T + solvent.T) / 2)
        if T is not None:
            T = float(T)
            if T < 200:
                T += 273.15
        P = feed.P
        mode = str(self.get_param('mode', 'isothermal')).strip().lower()
        mode_aliases = {
            'fixed_t': 'isothermal',
            'fixed-temperature': 'isothermal',
            'fixed_temperature': 'isothermal',
            'adiabatic-stage': 'adiabatic',
            'adiabatic_stage': 'adiabatic',
        }
        mode = mode_aliases.get(mode, mode)
        if mode not in ('isothermal', 'adiabatic'):
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' unknown mode '{mode}'. "
                "Use isothermal or adiabatic."
            )

        comps = []
        for stream in (feed, solvent):
            for comp in stream.composition:
                if comp not in comps:
                    comps.append(comp)
        nc = len(comps)
        if nc < 2:
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' requires at least 2 components"
            )

        def dense(comp: dict[str, float]) -> dict[str, float]:
            values = {c: max(float(comp.get(c, 0.0)), 0.0) for c in comps}
            total = sum(values.values())
            if total <= 0:
                raise UnitOperationError(
                    f"RigorousLiquidLiquidExtractor '{self.unit_id}' received an empty composition"
                )
            return {c: values[c] / total for c in comps}

        feed_z = dense(feed.composition)
        solvent_z = dense(solvent.composition)
        total_F = feed.F + solvent.F
        mixed_z = {
            c: (feed.F * feed_z.get(c, 0.0) + solvent.F * solvent_z.get(c, 0.0)) / total_F
            for c in comps
        }
        has_lle, x1, x2, _ = self.thermo.liquid_liquid_equilibrium(mixed_z, T)
        if not has_lle:
            out = self.thermo.calculate_state(T, P, total_F, mixed_z, phase='liquid', flash=False)
            empty = self.thermo.calculate_state(T, P, 0.0, mixed_z, phase='liquid', flash=False)
            empty.F = 0.0
            return UnitResult(
                outlet_streams={'raffinate': out, 'extract': empty},
                warnings=["No LLE at operating conditions"],
            )

        solvent_keys = [
            c for c in comps
            if solvent_z.get(c, 0.0) > feed_z.get(c, 0.0)
        ]
        score_1 = sum(x1.get(c, 0.0) for c in solvent_keys)
        score_2 = sum(x2.get(c, 0.0) for c in solvent_keys)
        x_ext_eq, x_raf_eq = (dense(x1), dense(x2)) if score_1 >= score_2 else (dense(x2), dense(x1))

        solver_algorithm = str(
            self.get_param('solver_algorithm', self.get_param('algorithm', 'equation_oriented'))
        ).strip().lower()
        solver_algorithm = {
            'mesh': 'equation_oriented',
            'eo': 'equation_oriented',
            'equation-oriented': 'equation_oriented',
            'equation_based': 'equation_oriented',
            'equation-based': 'equation_oriented',
            'sparse_newton': 'equation_oriented',
            'direct': 'equation_oriented',
            'broyden': 'split_sweep',
            'inverse_broyden': 'split_sweep',
            'split': 'split_sweep',
            'nested_lle': 'split_sweep',
            'stage_sweep': 'split_sweep',
        }.get(solver_algorithm, solver_algorithm)
        if solver_algorithm == 'equation_oriented':
            result = self._solve_equation_oriented(
                feed, solvent, N, T, P, mode, comps, feed_z, solvent_z,
                x_raf_eq, x_ext_eq, solvent_keys,
            )
            self._store_recycle_profile(result.performance)
            return result
        if solver_algorithm != 'split_sweep':
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' unknown solver_algorithm "
                f"'{solver_algorithm}'. Use split_sweep or equation_oriented."
            )

        flow_scale = max(total_F, 1.0)
        lle_tolerance = float(self.get_param('lle_tolerance', 1e-5))
        max_iterations = int(self.get_param('max_iterations', self.get_param('max_evaluations', 200)))
        convergence_tolerance = float(self.get_param('mesh_tolerance', 1e-5))
        damping = min(max(float(self.get_param('damping', 1.0)), 0.05), 1.0)
        acceleration = str(self.get_param('acceleration', 'broyden')).lower()
        fallback_acceleration = 'wegstein' if acceleration in (
            'broyden', 'broyden1', 'secant', 'newton_broyden'
        ) else acceleration
        temperature_scale = max(float(self.get_param('temperature_scale', 1000.0)), 1.0)
        default_t_min = max(250.0, min(feed.T, solvent.T, T) - 100.0)
        default_t_max = min(650.0, max(feed.T, solvent.T, T) + 100.0)
        adiabatic_t_min = float(self.get_param('T_min', self.get_param('adiabatic_T_min', default_t_min)))
        adiabatic_t_max = float(self.get_param('T_max', self.get_param('adiabatic_T_max', default_t_max)))
        if adiabatic_t_min < 200:
            adiabatic_t_min += 273.15
        if adiabatic_t_max < 200:
            adiabatic_t_max += 273.15
        if adiabatic_t_min >= adiabatic_t_max:
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' requires T_min < T_max"
            )

        def normalize_positive(comp: dict[str, float]) -> dict[str, float]:
            values = {c: max(float(comp.get(c, 0.0)), 0.0) for c in comps}
            total = sum(values.values())
            if total <= 0:
                return {c: 1.0 / nc for c in comps}
            return {c: values[c] / total for c in comps}

        shortcut = ShortcutExtractor(
            f"{self.unit_id}_shortcut", self.thermo,
            {'N_stages': N, 'T': T}
        ).solve({'feed': feed, 'solvent': solvent})
        raf_guess = normalize_positive(shortcut.outlet_streams['raffinate'].composition or x_raf_eq)
        ext_guess = normalize_positive(shortcut.outlet_streams['extract'].composition or x_ext_eq)
        R_guess = max(shortcut.outlet_streams['raffinate'].F, flow_scale * 1e-9)
        E_guess = max(shortcut.outlet_streams['extract'].F, flow_scale * 1e-9)

        def mix_streams(flow_a: float, comp_a: dict[str, float],
                        flow_b: float, comp_b: dict[str, float]) -> tuple[float, dict[str, float]]:
            total = flow_a + flow_b
            if total <= 0:
                raise UnitOperationError(
                    f"RigorousLiquidLiquidExtractor '{self.unit_id}' generated an empty stage feed"
                )
            return total, {
                c: (flow_a * comp_a.get(c, 0.0) + flow_b * comp_b.get(c, 0.0)) / total
                for c in comps
            }

        def liquid_state(flow: float, comp: dict[str, float], temperature: float) -> StreamState:
            return self.thermo.calculate_state(
                temperature, P, flow, normalize_positive(comp),
                phase='liquid', flash=False, include=('H', 'Cp')
            )

        def enthalpy_flow(flow: float, comp: dict[str, float], temperature: float) -> float:
            if flow <= 0.0:
                return 0.0
            state = liquid_state(flow, comp, temperature)
            return flow * (state.H or 0.0)

        def stage_input_enthalpy(
            R_in: float,
            x_raf_in: dict[str, float],
            T_raf_in: float,
            E_in: float,
            x_ext_in: dict[str, float],
            T_ext_in: float,
            stage: int,
        ) -> float:
            H_raf_in = feed.F * (feed.H or 0.0) if stage == 0 else enthalpy_flow(R_in, x_raf_in, T_raf_in)
            H_ext_in = solvent.F * (solvent.H or 0.0) if stage == N - 1 else enthalpy_flow(E_in, x_ext_in, T_ext_in)
            return H_raf_in + H_ext_in

        def split_stage_at_T(total: float, z: dict[str, float], temperature: float, stage: int):
            has_stage_lle, stage_x1, stage_x2, beta = self.thermo.liquid_liquid_equilibrium(
                dense(z), temperature, max_iter=200, tol=lle_tolerance
            )
            if not has_stage_lle:
                raise UnitOperationError(
                    f"RigorousLiquidLiquidExtractor '{self.unit_id}' found no LLE on stage {stage + 1}"
                )

            stage_x1 = dense(stage_x1)
            stage_x2 = dense(stage_x2)
            stage_score_1 = sum(stage_x1.get(c, 0.0) for c in solvent_keys)
            stage_score_2 = sum(stage_x2.get(c, 0.0) for c in solvent_keys)
            if stage_score_1 >= stage_score_2:
                x_extract = stage_x1
                x_raffinate = stage_x2
                extract_flow = total * (1.0 - beta)
                raffinate_flow = total * beta
            else:
                x_extract = stage_x2
                x_raffinate = stage_x1
                extract_flow = total * beta
                raffinate_flow = total * (1.0 - beta)

            return (
                max(raffinate_flow, 0.0),
                normalize_positive(x_raffinate),
                max(extract_flow, 0.0),
                normalize_positive(x_extract),
            )

        def outlet_enthalpy(
            R_stage: float,
            x_raf_stage: dict[str, float],
            E_stage: float,
            x_ext_stage: dict[str, float],
            temperature: float,
        ) -> float:
            return (
                enthalpy_flow(R_stage, x_raf_stage, temperature)
                + enthalpy_flow(E_stage, x_ext_stage, temperature)
            )

        def outlet_heat_capacity_flow(
            R_stage: float,
            x_raf_stage: dict[str, float],
            E_stage: float,
            x_ext_stage: dict[str, float],
            temperature: float,
        ) -> float:
            Cp_flow = 0.0
            if R_stage > 0.0:
                raf_state = liquid_state(R_stage, x_raf_stage, temperature)
                Cp_flow += R_stage * (raf_state.Cp or 0.0)
            if E_stage > 0.0:
                ext_state = liquid_state(E_stage, x_ext_stage, temperature)
                Cp_flow += E_stage * (ext_state.Cp or 0.0)
            return max(Cp_flow, 1e-12)

        def solve_adiabatic_stage(
            total: float,
            z: dict[str, float],
            target_enthalpy: float,
            stage: int,
            T_guess: float,
        ):
            from scipy.optimize import brentq

            evaluations = {}

            def evaluate(temperature: float):
                key = round(float(temperature), 8)
                if key in evaluations:
                    return evaluations[key]
                R_stage, x_raf_stage, E_stage, x_ext_stage = split_stage_at_T(
                    total, z, temperature, stage
                )
                residual = outlet_enthalpy(
                    R_stage, x_raf_stage, E_stage, x_ext_stage, temperature
                ) - target_enthalpy
                result = (residual, R_stage, x_raf_stage, E_stage, x_ext_stage)
                evaluations[key] = result
                return result

            current_T = max(adiabatic_t_min, min(adiabatic_t_max, float(T_guess)))
            for _ in range(8):
                try:
                    residual, R_stage, x_raf_stage, E_stage, x_ext_stage = evaluate(current_T)
                except Exception:
                    break
                if abs(residual) < max(1e-4, abs(target_enthalpy) * 1e-10):
                    return R_stage, x_raf_stage, E_stage, x_ext_stage, current_T, residual
                Cp_flow = outlet_heat_capacity_flow(
                    R_stage, x_raf_stage, E_stage, x_ext_stage, current_T
                )
                step = residual / Cp_flow
                if not math.isfinite(step):
                    break
                step = max(-25.0, min(25.0, step))
                next_T = max(adiabatic_t_min, min(adiabatic_t_max, current_T - step))
                if abs(next_T - current_T) < 1e-6:
                    return R_stage, x_raf_stage, E_stage, x_ext_stage, current_T, residual
                current_T = next_T

            candidates = [
                current_T,
                T_guess,
                T,
                0.5 * (adiabatic_t_min + adiabatic_t_max),
                min(max(T_guess - 5.0, adiabatic_t_min), adiabatic_t_max),
                min(max(T_guess + 5.0, adiabatic_t_min), adiabatic_t_max),
                adiabatic_t_min,
                adiabatic_t_max,
            ]
            for delta in (10.0, 20.0, 40.0, 80.0):
                candidates.append(min(max(current_T - delta, adiabatic_t_min), adiabatic_t_max))
                candidates.append(min(max(current_T + delta, adiabatic_t_min), adiabatic_t_max))
            valid = []
            last_error = None
            for candidate in sorted(set(max(adiabatic_t_min, min(adiabatic_t_max, float(Tc))) for Tc in candidates)):
                try:
                    residual, *_ = evaluate(candidate)
                except Exception as exc:
                    last_error = exc
                    continue
                if abs(residual) < 1e-5:
                    _, R_stage, x_raf_stage, E_stage, x_ext_stage = evaluate(candidate)
                    return R_stage, x_raf_stage, E_stage, x_ext_stage, candidate, residual
                valid.append((candidate, residual))

            for (T1, f1), (T2, f2) in zip(valid, valid[1:]):
                if f1 * f2 > 0.0:
                    continue
                root = brentq(
                    lambda temp: evaluate(temp)[0],
                    T1,
                    T2,
                    xtol=1e-6,
                    rtol=1e-9,
                    maxiter=100,
                )
                residual, R_stage, x_raf_stage, E_stage, x_ext_stage = evaluate(root)
                return R_stage, x_raf_stage, E_stage, x_ext_stage, root, residual

            if valid:
                best_T, best_residual = min(valid, key=lambda item: abs(item[1]))
                raise UnitOperationError(
                    f"RigorousLiquidLiquidExtractor '{self.unit_id}' could not close "
                    f"adiabatic energy balance on stage {stage + 1} between "
                    f"{adiabatic_t_min - 273.15:.2f} C and {adiabatic_t_max - 273.15:.2f} C "
                    f"(best residual {best_residual:.4g} kJ/h at {best_T - 273.15:.2f} C)"
                )
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' found no valid LLE "
                f"temperature for adiabatic stage {stage + 1}"
                + (f" ({last_error})" if last_error else "")
            )

        def split_stage(
            total: float,
            z: dict[str, float],
            stage: int,
            target_enthalpy: float,
            T_guess: float,
        ):
            if mode == 'adiabatic':
                return solve_adiabatic_stage(total, z, target_enthalpy, stage, T_guess)
            R_stage, x_raf_stage, E_stage, x_ext_stage = split_stage_at_T(total, z, T, stage)
            energy_residual = outlet_enthalpy(
                R_stage, x_raf_stage, E_stage, x_ext_stage, T
            ) - target_enthalpy
            return R_stage, x_raf_stage, E_stage, x_ext_stage, T, energy_residual

        def stream_difference(R_old, E_old, x_raf_old, x_ext_old, T_old,
                              R_new, E_new, x_raf_new, x_ext_new, T_new) -> float:
            max_diff = 0.0
            for stage in range(N):
                max_diff = max(max_diff, abs(T_new[stage] - T_old[stage]) / temperature_scale)
                for comp in comps:
                    max_diff = max(
                        max_diff,
                        abs(R_new[stage] * x_raf_new[stage].get(comp, 0.0)
                            - R_old[stage] * x_raf_old[stage].get(comp, 0.0)) / flow_scale,
                        abs(E_new[stage] * x_ext_new[stage].get(comp, 0.0)
                            - E_old[stage] * x_ext_old[stage].get(comp, 0.0)) / flow_scale,
                    )
            return max_diff

        def damp_composition(old: dict[str, float], new: dict[str, float]) -> dict[str, float]:
            return normalize_positive({
                c: (1.0 - damping) * old.get(c, 0.0) + damping * new.get(c, 0.0)
                for c in comps
            })

        def flatten_extract(flows: list[float], compositions: list[dict[str, float]]) -> list[float]:
            return [
                max(flows[stage] * compositions[stage].get(comp, 0.0), 1e-16)
                for stage in range(N)
                for comp in comps
            ]

        def unflatten_extract(values: list[float]) -> tuple[list[float], list[dict[str, float]]]:
            flows, compositions = [], []
            for stage in range(N):
                stage_values = {
                    comp: max(float(values[stage * nc + idx]), 1e-16)
                    for idx, comp in enumerate(comps)
                }
                stage_flow = sum(stage_values.values())
                flows.append(stage_flow)
                compositions.append({comp: stage_values[comp] / stage_flow for comp in comps})
            return flows, compositions

        def wegstein_update(current: list[float], mapped: list[float],
                            previous: list[float], previous_mapped: list[float]) -> list[float]:
            updated = []
            for x_now, g_now, x_old, g_old in zip(current, mapped, previous, previous_mapped):
                denominator = x_now - x_old
                if abs(denominator) < 1e-16:
                    updated.append(g_now)
                    continue
                slope = (g_now - g_old) / denominator
                if abs(slope - 1.0) < 1e-12 or not math.isfinite(slope):
                    updated.append(g_now)
                    continue
                q = slope / (slope - 1.0)
                q = min(0.0, max(-5.0, q))
                value = q * x_now + (1.0 - q) * g_now
                if value > 0.0 and math.isfinite(value):
                    updated.append(value)
                else:
                    updated.append(g_now)
            return updated

        def run_sweep(E_current, x_ext_current, T_stage_current):
            R_new, E_new, x_raf_new, x_ext_new, T_stage_new = [], [], [], [], []
            stage_energy_residuals_new = []

            for stage in range(N):
                R_in = feed.F if stage == 0 else R_new[stage - 1]
                x_raf_in = feed_z if stage == 0 else x_raf_new[stage - 1]
                T_raf_in = feed.T if stage == 0 else T_stage_new[stage - 1]
                E_in = solvent.F if stage == N - 1 else E_current[stage + 1]
                x_ext_in = solvent_z if stage == N - 1 else x_ext_current[stage + 1]
                T_ext_in = solvent.T if stage == N - 1 else T_stage_current[stage + 1]

                mixed_flow, mixed_comp = mix_streams(R_in, x_raf_in, E_in, x_ext_in)
                H_in = stage_input_enthalpy(
                    R_in, x_raf_in, T_raf_in, E_in, x_ext_in, T_ext_in, stage
                )
                T_guess = 0.5 * (T_raf_in + T_ext_in)
                R_stage, x_raf_stage, E_stage, x_ext_stage, T_out, energy_residual = split_stage(
                    mixed_flow, mixed_comp, stage, H_in, T_guess
                )
                R_new.append(R_stage)
                E_new.append(E_stage)
                x_raf_new.append(x_raf_stage)
                x_ext_new.append(x_ext_stage)
                T_stage_new.append(T_out)
                stage_energy_residuals_new.append(energy_residual)

            return R_new, E_new, x_raf_new, x_ext_new, T_stage_new, stage_energy_residuals_new

        def pack_solution(E_values, x_ext_values, T_values) -> list[float]:
            values = flatten_extract(E_values, x_ext_values)
            if mode == 'adiabatic':
                values.extend(float(value) for value in T_values)
            return values

        def unpack_solution(values: list[float]):
            extract_values = values[:N * nc]
            E_values, x_ext_values = unflatten_extract(extract_values)
            if mode == 'adiabatic':
                T_values = [
                    max(adiabatic_t_min, min(adiabatic_t_max, float(value)))
                    for value in values[N * nc:N * nc + N]
                ]
            else:
                T_values = [T for _ in range(N)]
            return E_values, x_ext_values, T_values

        def clip_solution_vector(values: list[float]) -> list[float]:
            clipped = [max(float(value), 1e-16) for value in values[:N * nc]]
            if mode == 'adiabatic':
                clipped.extend(
                    max(adiabatic_t_min, min(adiabatic_t_max, float(value)))
                    for value in values[N * nc:N * nc + N]
                )
            return clipped

        def solution_scales() -> list[float]:
            scales = [flow_scale for _ in range(N * nc)]
            if mode == 'adiabatic':
                scales.extend(temperature_scale for _ in range(N))
            return scales

        R = [R_guess for _ in range(N)]
        E = [E_guess for _ in range(N)]
        x_raf = [dict(raf_guess) for _ in range(N)]
        x_ext = [dict(ext_guess) for _ in range(N)]
        T_stage = [T for _ in range(N)]
        residual_norm = float('inf')
        previous_extract = None
        previous_mapped_extract = None
        stage_energy_residuals = [0.0 for _ in range(N)]
        iteration = 0
        solved_with_broyden = False

        if acceleration in ('broyden', 'broyden1', 'secant', 'newton_broyden'):
            import numpy as np

            scales = np.array(solution_scales(), dtype=float)
            last_sweep = {}

            def residual_for_vector(vector):
                vector = clip_solution_vector(list(vector))
                E_eval, x_ext_eval, T_eval = unpack_solution(vector)
                sweep = run_sweep(E_eval, x_ext_eval, T_eval)
                R_map, E_map, x_raf_map, x_ext_map, T_map, energy_map = sweep
                mapped = pack_solution(E_map, x_ext_map, T_map)
                residual = np.array(mapped, dtype=float) - np.array(vector, dtype=float)
                norm = float(np.max(np.abs(residual) / scales))
                balance = self._external_component_balance_error(
                    comps, feed, feed_z, solvent, solvent_z,
                    R_map[-1], x_raf_map[-1], E_map[0], x_ext_map[0],
                    flow_scale,
                )
                last_sweep['value'] = (
                    vector, residual, norm, balance,
                    R_map, E_map, x_raf_map, x_ext_map, T_map, energy_map,
                )
                return residual, norm, balance

            x_vec = np.array(pack_solution(E, x_ext, T_stage), dtype=float)
            residual_vec, residual_norm, outlet_balance_error = residual_for_vector(x_vec)
            inverse_jacobian = -np.eye(len(x_vec))

            for iteration in range(1, max_iterations + 1):
                (
                    _vector, _residual, residual_norm, outlet_balance_error,
                    R_map, E_map, x_raf_map, x_ext_map, T_map, energy_map,
                ) = last_sweep['value']
                if residual_norm < convergence_tolerance and outlet_balance_error < convergence_tolerance:
                    R, E, x_raf, x_ext, T_stage = R_map, E_map, x_raf_map, x_ext_map, T_map
                    stage_energy_residuals = energy_map
                    solved_with_broyden = True
                    break

                step = -inverse_jacobian.dot(residual_vec)
                if not np.all(np.isfinite(step)) or np.max(np.abs(step) / scales) > 5.0:
                    step = residual_vec

                accepted = None
                for factor in (1.0, 0.5, 0.25, 0.1):
                    trial_vec = np.array(clip_solution_vector(list(x_vec + factor * step)), dtype=float)
                    try:
                        trial_residual, trial_norm, trial_balance = residual_for_vector(trial_vec)
                    except UnitOperationError:
                        continue
                    accepted = (trial_vec, trial_residual, trial_norm, trial_balance)
                    if trial_norm <= residual_norm or factor == 0.1:
                        break
                if accepted is None:
                    break

                next_vec, next_residual, next_norm, _next_balance = accepted
                s = next_vec - x_vec
                y = next_residual - residual_vec
                Hy = inverse_jacobian.dot(y)
                denom = float(s.dot(Hy))
                if abs(denom) > 1e-20 and np.all(np.isfinite(Hy)):
                    inverse_jacobian = inverse_jacobian + np.outer((s - Hy), s.dot(inverse_jacobian)) / denom

                x_vec = next_vec
                residual_vec = next_residual

        if not solved_with_broyden:
            for iteration in range(1, max_iterations + 1):
                R_new, E_new, x_raf_new, x_ext_new, T_stage_new, stage_energy_residuals_new = run_sweep(
                    E, x_ext, T_stage
                )

                residual_norm = stream_difference(
                    R, E, x_raf, x_ext, T_stage,
                    R_new, E_new, x_raf_new, x_ext_new, T_stage_new,
                )
                outlet_balance_error = self._external_component_balance_error(
                    comps, feed, feed_z, solvent, solvent_z,
                    R_new[-1], x_raf_new[-1], E_new[0], x_ext_new[0],
                    flow_scale,
                )
                if residual_norm < convergence_tolerance and outlet_balance_error < convergence_tolerance:
                    R, E, x_raf, x_ext, T_stage = R_new, E_new, x_raf_new, x_ext_new, T_stage_new
                    stage_energy_residuals = stage_energy_residuals_new
                    break

                current_extract = flatten_extract(E, x_ext)
                mapped_extract = flatten_extract(E_new, x_ext_new)
                if fallback_acceleration == 'wegstein' and previous_extract is not None and damping >= 0.999:
                    next_extract = wegstein_update(
                        current_extract, mapped_extract,
                        previous_extract, previous_mapped_extract,
                    )
                    E, x_ext = unflatten_extract(next_extract)
                    T_stage = T_stage_new
                else:
                    E = [(1.0 - damping) * E[stage] + damping * E_new[stage] for stage in range(N)]
                    x_ext = [damp_composition(x_ext[stage], x_ext_new[stage]) for stage in range(N)]
                    T_stage = [
                        (1.0 - damping) * T_stage[stage] + damping * T_stage_new[stage]
                        for stage in range(N)
                    ]
                R = [(1.0 - damping) * R[stage] + damping * R_new[stage] for stage in range(N)]
                x_raf = [damp_composition(x_raf[stage], x_raf_new[stage]) for stage in range(N)]
                stage_energy_residuals = stage_energy_residuals_new
                previous_extract = current_extract
                previous_mapped_extract = mapped_extract
            else:
                raise UnitOperationError(
                    f"RigorousLiquidLiquidExtractor '{self.unit_id}' failed to converge "
                    f"after {max_iterations} iterations (residual {residual_norm:.2e})"
                )

        if not solved_with_broyden and residual_norm >= convergence_tolerance:
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' failed to converge "
                f"after {max_iterations} iterations (residual {residual_norm:.2e})"
            )

        raffinate = self.thermo.calculate_state(
            T_stage[-1], P, R[-1], x_raf[-1], phase='liquid', flash=False
        )
        extract = self.thermo.calculate_state(
            T_stage[0], P, E[0], x_ext[0], phase='liquid', flash=False
        )

        component_balance_error = self._external_component_balance_error(
            comps, feed, feed_z, solvent, solvent_z,
            R[-1], x_raf[-1], E[0], x_ext[0], flow_scale,
        )
        if component_balance_error > convergence_tolerance:
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' failed overall mass balance "
                f"(error {component_balance_error:.2e})"
            )
        stage_energy_residuals = self._stage_energy_residuals(
            comps, feed, feed_z, solvent, solvent_z, R, x_raf, E, x_ext, T_stage, P
        )
        total_heat_duty = 0.0
        if mode == 'isothermal':
            total_heat_duty = sum(stage_energy_residuals)
        overall_energy_residual = (
            raffinate.F * (raffinate.H or 0.0)
            + extract.F * (extract.H or 0.0)
            - feed.F * (feed.H or 0.0)
            - solvent.F * (solvent.H or 0.0)
            - total_heat_duty
        )
        energy_scale = max(
            abs(feed.F * (feed.H or 0.0)) + abs(solvent.F * (solvent.H or 0.0)),
            abs(raffinate.F * (raffinate.H or 0.0)) + abs(extract.F * (extract.H or 0.0)),
            1.0,
        )

        result = UnitResult(
            outlet_streams={'raffinate': raffinate, 'extract': extract},
            heat_duty=total_heat_duty,
            performance={
                'N_stages': N,
                'mode': mode,
                'solver': 'deprecated_split_sweep',
                'solver_algorithm': 'split_sweep',
                'acceleration_algorithm': 'inverse_broyden' if solved_with_broyden else fallback_acceleration,
                'T_C': T - 273.15,
                'raffinate_T_C': raffinate.T - 273.15,
                'extract_T_C': extract.T - 273.15,
                'mesh_residual': residual_norm,
                'component_balance_error': component_balance_error,
                'max_stage_energy_residual_kJ_h': max(abs(value) for value in stage_energy_residuals),
                'max_stage_energy_relative_error': max(abs(value) for value in stage_energy_residuals) / energy_scale,
                'overall_energy_residual_kJ_h': overall_energy_residual,
                'overall_energy_relative_error': abs(overall_energy_residual) / energy_scale,
                'solver_iterations': iteration,
                'terminal_raffinate_stage_flow': float(R[-1]),
                'terminal_raffinate_stage_composition': x_raf[-1],
                'raffinate_flows': [float(value) for value in R],
                'extract_flows': [float(value) for value in E],
                'stage_temperatures_C': [float(value - 273.15) for value in T_stage],
                'stage_energy_residuals_kJ_h': [float(value) for value in stage_energy_residuals],
                'stage_raffinate_compositions': x_raf,
                'stage_extract_compositions': x_ext,
            },
            warnings=[
                "RigorousLiquidLiquidExtractor solver_algorithm=split_sweep is deprecated; "
                "use solver_algorithm=equation_oriented."
            ],
        )
        self._store_recycle_profile(result.performance)
        return result

    def _solve_equation_oriented(
        self,
        feed: StreamState,
        solvent: StreamState,
        N: int,
        T: float,
        P: float,
        mode: str,
        comps: list[str],
        feed_z: dict[str, float],
        solvent_z: dict[str, float],
        x_raf_eq: dict[str, float],
        x_ext_eq: dict[str, float],
        solvent_keys: list[str],
    ) -> UnitResult:
        import numpy as np
        from scipy.sparse import lil_matrix

        nc = len(comps)
        total_F = feed.F + solvent.F
        flow_scale = max(total_F, 1.0)
        energy_scale = max(
            abs(feed.F * (feed.H or 0.0)) + abs(solvent.F * (solvent.H or 0.0)),
            total_F * 50000.0,
            1.0,
        )
        component_floor = float(self.get_param('component_scale_floor', 1e-5))
        component_scales = {
            comp: max(
                feed.F * feed_z.get(comp, 0.0) + solvent.F * solvent_z.get(comp, 0.0),
                flow_scale * component_floor,
                1e-12,
            )
            for comp in comps
        }
        activity_scale = max(float(self.get_param('activity_scale', 1.0)), 1e-12)

        default_t_min = max(250.0, min(feed.T, solvent.T, T) - 100.0)
        default_t_max = min(650.0, max(feed.T, solvent.T, T) + 100.0)
        adiabatic_t_min = float(self.get_param('T_min', self.get_param('adiabatic_T_min', default_t_min)))
        adiabatic_t_max = float(self.get_param('T_max', self.get_param('adiabatic_T_max', default_t_max)))
        if adiabatic_t_min < 200:
            adiabatic_t_min += 273.15
        if adiabatic_t_max < 200:
            adiabatic_t_max += 273.15
        if adiabatic_t_min >= adiabatic_t_max:
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' requires T_min < T_max"
            )

        def normalize(comp: dict[str, float]) -> dict[str, float]:
            values = {c: max(float(comp.get(c, 0.0)), 0.0) for c in comps}
            total = sum(values.values())
            if total <= 0.0:
                return {c: 1.0 / nc for c in comps}
            return {c: values[c] / total for c in comps}

        def phase_enthalpy(comp: dict[str, float], temperature: float) -> float:
            return self.thermo.mixture_enthalpy(
                normalize(comp), float(temperature), vapor_fraction=0.0, P=float(P)
            )

        def split_at_temperature(total: float, z: dict[str, float], temperature: float):
            has_lle, x1, x2, beta = self.thermo.liquid_liquid_equilibrium(
                normalize(z), temperature,
                max_iter=200,
                tol=float(self.get_param('lle_tolerance', 1e-5)),
            )
            if not has_lle:
                raise UnitOperationError(
                    f"RigorousLiquidLiquidExtractor '{self.unit_id}' found no LLE "
                    "while initializing equation-oriented solve"
                )
            x1 = normalize(x1)
            x2 = normalize(x2)
            score_1 = sum(x1.get(c, 0.0) for c in solvent_keys)
            score_2 = sum(x2.get(c, 0.0) for c in solvent_keys)
            if score_1 >= score_2:
                return total * beta, x2, total * (1.0 - beta), x1
            return total * (1.0 - beta), x1, total * beta, x2

        def linear_temperature_profile() -> list[float]:
            if mode != 'adiabatic':
                return [float(T) for _ in range(N)]
            return [
                float(feed.T + (solvent.T - feed.T) * stage / max(N - 1, 1))
                for stage in range(N)
            ]

        def shortcut_split_sweep_guess():
            shortcut = ShortcutExtractor(
                f"{self.unit_id}_shortcut", self.thermo,
                {'N_stages': N, 'T': T}
            ).solve({'feed': feed, 'solvent': solvent})
            R_guess = [
                max(shortcut.outlet_streams['raffinate'].F, flow_scale * 1e-12)
                for _ in range(N)
            ]
            E_guess = [
                max(shortcut.outlet_streams['extract'].F, flow_scale * 1e-12)
                for _ in range(N)
            ]
            xR_guess = [
                normalize(shortcut.outlet_streams['raffinate'].composition or x_raf_eq)
                for _ in range(N)
            ]
            xE_guess = [
                normalize(shortcut.outlet_streams['extract'].composition or x_ext_eq)
                for _ in range(N)
            ]
            T_guess = linear_temperature_profile()

            initial_sweeps = int(self.get_param('mesh_initial_sweeps', 2 if mode == 'isothermal' else 1))
            for _ in range(max(initial_sweeps, 0)):
                R_new, E_new, xR_new, xE_new = [], [], [], []
                for stage in range(N):
                    R_in = feed.F if stage == 0 else R_new[stage - 1]
                    xR_in = feed_z if stage == 0 else xR_new[stage - 1]
                    E_in = solvent.F if stage == N - 1 else E_guess[stage + 1]
                    xE_in = solvent_z if stage == N - 1 else xE_guess[stage + 1]
                    mixed_flow = R_in + E_in
                    mixed_z = {
                        comp: (
                            R_in * xR_in.get(comp, 0.0)
                            + E_in * xE_in.get(comp, 0.0)
                        ) / mixed_flow
                        for comp in comps
                    }
                    R_stage, xR_stage, E_stage, xE_stage = split_at_temperature(
                        mixed_flow, mixed_z, T_guess[stage]
                    )
                    R_new.append(max(R_stage, flow_scale * 1e-12))
                    E_new.append(max(E_stage, flow_scale * 1e-12))
                    xR_new.append(xR_stage)
                    xE_new.append(xE_stage)
                R_guess, E_guess, xR_guess, xE_guess = R_new, E_new, xR_new, xE_new

            return {
                'R': R_guess,
                'E': E_guess,
                'x_raf': xR_guess,
                'x_ext': xE_guess,
                'T_stage': T_guess,
                'initializer': 'split_sweep',
                'warnings': [],
            }

        def interpolate_profile(values: list[float], source_N: int) -> list[float]:
            if source_N == N:
                return [float(value) for value in values]
            if source_N <= 1:
                return [float(values[0]) for _ in range(N)]
            coarse_grid = np.linspace(0.0, 1.0, source_N)
            full_grid = np.linspace(0.0, 1.0, N)
            return [float(np.interp(position, coarse_grid, values)) for position in full_grid]

        def interpolate_compositions(values: list[dict[str, float]], source_N: int) -> list[dict[str, float]]:
            if source_N == N:
                return [normalize(stage) for stage in values]
            if source_N <= 1:
                return [normalize(values[0]) for _ in range(N)]
            coarse_grid = np.linspace(0.0, 1.0, source_N)
            full_grid = np.linspace(0.0, 1.0, N)
            interpolated = []
            for position in full_grid:
                stage_comp = {
                    comp: max(
                        float(np.interp(position, coarse_grid, [stage.get(comp, 0.0) for stage in values])),
                        1e-14,
                    )
                    for comp in comps
                }
                interpolated.append(normalize(stage_comp))
            return interpolated

        def initial_guess_from_result(result: UnitResult, source_name: str):
            perf = result.performance
            source_N = len(perf.get('raffinate_flows', []))
            if source_N <= 0:
                return None
            stage_temperatures = [
                float(value + 273.15)
                for value in perf.get('stage_temperatures_C', [])
            ]
            if len(stage_temperatures) != source_N:
                stage_temperatures = [float(T) for _ in range(source_N)]
            if mode == 'adiabatic' and source_name == 'homotopy':
                target_temperatures = linear_temperature_profile()
            elif mode == 'adiabatic':
                target_temperatures = interpolate_profile(stage_temperatures, source_N)
            else:
                target_temperatures = [float(T) for _ in range(N)]

            return {
                'R': [max(value, flow_scale * 1e-12) for value in interpolate_profile(perf['raffinate_flows'], source_N)],
                'E': [max(value, flow_scale * 1e-12) for value in interpolate_profile(perf['extract_flows'], source_N)],
                'x_raf': interpolate_compositions(perf['stage_raffinate_compositions'], source_N),
                'x_ext': interpolate_compositions(perf['stage_extract_compositions'], source_N),
                'T_stage': target_temperatures,
                'initializer': source_name,
                'warnings': list(result.warnings or []),
            }

        def child_params(**updates) -> dict:
            params = dict(self.params)
            params.update(updates)
            params['solver_algorithm'] = 'equation_oriented'
            params['max_iterations'] = int(self.get_param('initial_max_iterations', self.get_param('max_iterations', 80)))
            params['max_jacobian_evaluations'] = int(
                self.get_param('initial_max_jacobian_evaluations', self.get_param('max_jacobian_evaluations', 50))
            )
            return params

        def coarse_grid_guess():
            if self._truthy_param(self.get_param('_skip_coarse_initializer', False)) or N <= 2:
                return None
            requested = self.get_param('coarse_initial_stages', self.get_param('coarse_N'))
            if requested is None:
                coarse_N = max(2, min(N - 1, max(3, min(8, (N + 1) // 2))))
            else:
                coarse_N = int(requested)
            coarse_N = max(2, min(N - 1, coarse_N))
            params = child_params(
                N_stages=coarse_N,
                initializer='split_sweep',
                _skip_coarse_initializer=True,
            )
            try:
                coarse = RigorousLiquidLiquidExtractor(
                    f"{self.unit_id}_coarse_init", self.thermo, params
                ).solve({'feed': feed, 'solvent': solvent})
            except Exception:
                return None
            guess = initial_guess_from_result(coarse, 'coarse_grid')
            if guess is not None:
                guess['coarse_initial_stages'] = coarse_N
            return guess

        def homotopy_guess():
            if self._truthy_param(self.get_param('_skip_homotopy_initializer', False)):
                return None
            if mode == 'adiabatic':
                params = child_params(
                    mode='isothermal',
                    T=T,
                    initializer=self.get_param('homotopy_base_initializer', 'coarse_grid'),
                    _skip_homotopy_initializer=True,
                )
                try:
                    base = RigorousLiquidLiquidExtractor(
                        f"{self.unit_id}_homotopy_init", self.thermo, params
                    ).solve({'feed': feed, 'solvent': solvent})
                except Exception:
                    return None
                return initial_guess_from_result(base, 'homotopy')
            return coarse_grid_guess()

        requested_initializer = str(
            self.get_param('initializer', self.get_param('mesh_initializer', 'auto'))
        ).strip().lower()
        initializer = requested_initializer
        initializer = {
            'default': 'auto',
            'automatic': 'auto',
            'shortcut': 'split_sweep',
            'shortcut_split': 'split_sweep',
            'stage_sweep': 'split_sweep',
            'split': 'split_sweep',
            'coarse': 'coarse_grid',
            'coarse-grid': 'coarse_grid',
            'coarse_grid': 'coarse_grid',
            'continuation': 'homotopy',
            'temperature_homotopy': 'homotopy',
        }.get(initializer, initializer)
        if initializer not in ('auto', 'split_sweep', 'coarse_grid', 'homotopy'):
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' unknown initializer "
                f"'{initializer}'. Use auto, split_sweep, coarse_grid, or homotopy."
            )
        if initializer == 'auto':
            auto_coarse_min_stages = int(self.get_param('auto_coarse_min_stages', 8))
            initializer = 'coarse_grid' if N >= auto_coarse_min_stages else 'split_sweep'

        quality_context = getattr(self.thermo, 'quality_context', None)

        def run_initializer(callback):
            if quality_context is None:
                return callback()
            with quality_context(phase='initializer', affects_result=False):
                return callback()

        initializer_warnings = []
        previous_profile_guess = None
        context = getattr(self, 'solve_context', {}) or {}
        previous_profile = getattr(self, '_last_recycle_profile', None)
        if (
            context.get('recycle_evaluation') is not None
            and self._truthy_param(self.get_param('recycle_warm_start', True))
            and not self._consume_recycle_warm_start_skip()
            and previous_profile is not None
        ):
            try:
                previous_result = UnitResult(
                    outlet_streams={},
                    performance=previous_profile.get('performance', previous_profile),
                    warnings=[],
                )
                previous_profile_guess = initial_guess_from_result(
                    previous_result,
                    'previous_recycle',
                )
            except (TypeError, ValueError, KeyError, IndexError):
                previous_profile_guess = None

        if previous_profile_guess is not None:
            guess = previous_profile_guess
        elif initializer == 'coarse_grid':
            guess = run_initializer(coarse_grid_guess)
            if guess is None:
                initializer_warnings.append(
                    "coarse_grid initializer failed; falling back to split_sweep"
                )
                guess = run_initializer(shortcut_split_sweep_guess)
        elif initializer == 'homotopy':
            guess = run_initializer(homotopy_guess)
            if guess is None:
                initializer_warnings.append(
                    "homotopy initializer failed; falling back to split_sweep"
                )
                guess = run_initializer(shortcut_split_sweep_guess)
        else:
            guess = run_initializer(shortcut_split_sweep_guess)

        initializer_warnings.extend(guess.get('warnings', []))
        R = guess['R']
        E = guess['E']
        x_raf = guess['x_raf']
        x_ext = guess['x_ext']
        T_stage = guess['T_stage']
        initializer_used = guess['initializer']

        stage_var_count = 2 + 2 * (nc - 1) + (1 if mode == 'adiabatic' else 0)
        stage_row_count = 2 * nc + (1 if mode == 'adiabatic' else 0)
        n_vars = N * stage_var_count
        n_rows = N * stage_row_count
        T_span = adiabatic_t_max - adiabatic_t_min

        def offsets(stage: int) -> dict[str, int]:
            base = stage * stage_var_count
            return {
                'R': base,
                'E': base + 1,
                'xR': base + 2,
                'xE': base + 2 + nc - 1,
                'T': base + 2 + 2 * (nc - 1),
            }

        def softmax_logits(values) -> dict[str, float]:
            logits = np.array(list(values) + [0.0], dtype=float)
            logits -= np.max(logits)
            exp_values = np.exp(logits)
            fractions = exp_values / np.sum(exp_values)
            return {comp: float(fractions[i]) for i, comp in enumerate(comps)}

        def encode_temperature(value: float) -> float:
            reduced = min(max((float(value) - adiabatic_t_min) / T_span, 1e-8), 1.0 - 1e-8)
            return math.log(reduced / (1.0 - reduced))

        def decode_temperature(theta: float) -> float:
            theta = min(max(float(theta), -60.0), 60.0)
            return adiabatic_t_min + T_span / (1.0 + math.exp(-theta))

        def composition_logits(comp: dict[str, float]) -> list[float]:
            comp = normalize(comp)
            reference = max(comp.get(comps[-1], 0.0), 1e-14)
            return [
                math.log(max(comp.get(component, 0.0), 1e-14) / reference)
                for component in comps[:-1]
            ]

        def pack() -> np.ndarray:
            values = []
            for stage in range(N):
                values.append(math.log(max(R[stage], flow_scale * 1e-12)))
                values.append(math.log(max(E[stage], flow_scale * 1e-12)))
                values.extend(composition_logits(x_raf[stage]))
                values.extend(composition_logits(x_ext[stage]))
                if mode == 'adiabatic':
                    values.append(encode_temperature(T_stage[stage]))
            return np.array(values, dtype=float)

        def decode(vector):
            R_values, E_values, xR_values, xE_values, T_values = [], [], [], [], []
            for stage in range(N):
                off = offsets(stage)
                R_values.append(float(np.exp(np.clip(vector[off['R']], -40.0, 40.0))))
                E_values.append(float(np.exp(np.clip(vector[off['E']], -40.0, 40.0))))
                xR_values.append(softmax_logits(vector[off['xR']:off['xR'] + nc - 1]))
                xE_values.append(softmax_logits(vector[off['xE']:off['xE'] + nc - 1]))
                if mode == 'adiabatic':
                    T_values.append(decode_temperature(vector[off['T']]))
                else:
                    T_values.append(float(T))
            return R_values, E_values, xR_values, xE_values, T_values

        residual_labels = []

        def residual(vector):
            Rv, Ev, xRv, xEv, Tv = decode(vector)
            hR = [phase_enthalpy(xRv[stage], Tv[stage]) for stage in range(N)]
            hE = [phase_enthalpy(xEv[stage], Tv[stage]) for stage in range(N)]
            gamma_R = [
                self.thermo.activity_coefficients(float(Tv[stage]), xRv[stage])
                for stage in range(N)
            ]
            gamma_E = [
                self.thermo.activity_coefficients(float(Tv[stage]), xEv[stage])
                for stage in range(N)
            ]

            residuals = []
            labels = []
            for stage in range(N):
                for comp in comps:
                    in_comp = (
                        feed.F * feed_z.get(comp, 0.0)
                        if stage == 0
                        else Rv[stage - 1] * xRv[stage - 1].get(comp, 0.0)
                    )
                    in_comp += (
                        solvent.F * solvent_z.get(comp, 0.0)
                        if stage == N - 1
                        else Ev[stage + 1] * xEv[stage + 1].get(comp, 0.0)
                    )
                    out_comp = (
                        Rv[stage] * xRv[stage].get(comp, 0.0)
                        + Ev[stage] * xEv[stage].get(comp, 0.0)
                    )
                    residuals.append((in_comp - out_comp) / component_scales[comp])
                    labels.append(('component', stage + 1, comp, component_scales[comp]))

                for comp in comps:
                    aR = max(xRv[stage].get(comp, 0.0) * gamma_R[stage].get(comp, 1.0), 1e-300)
                    aE = max(xEv[stage].get(comp, 0.0) * gamma_E[stage].get(comp, 1.0), 1e-300)
                    residuals.append((math.log(aR) - math.log(aE)) / activity_scale)
                    labels.append(('activity', stage + 1, comp, activity_scale))

                if mode == 'adiabatic':
                    H_in = (
                        feed.F * (feed.H or 0.0)
                        if stage == 0
                        else Rv[stage - 1] * hR[stage - 1]
                    )
                    H_in += (
                        solvent.F * (solvent.H or 0.0)
                        if stage == N - 1
                        else Ev[stage + 1] * hE[stage + 1]
                    )
                    H_out = Rv[stage] * hR[stage] + Ev[stage] * hE[stage]
                    residuals.append((H_in - H_out) / energy_scale)
                    labels.append(('energy', stage + 1, None, energy_scale))

            if not residual_labels:
                residual_labels.extend(labels)
            return np.array(residuals, dtype=float)

        def sparsity():
            matrix = lil_matrix((n_rows, n_vars), dtype=int)

            def mark_stage(row: int, stage: int):
                if not 0 <= stage < N:
                    return
                off = offsets(stage)
                for col in range(off['R'], off['R'] + stage_var_count):
                    matrix[row, col] = 1

            row = 0
            for stage in range(N):
                local = {stage}
                if stage > 0:
                    local.add(stage - 1)
                if stage < N - 1:
                    local.add(stage + 1)
                for _ in comps:
                    for local_stage in local:
                        mark_stage(row, local_stage)
                    row += 1
                for _ in comps:
                    mark_stage(row, stage)
                    row += 1
                if mode == 'adiabatic':
                    for local_stage in local:
                        mark_stage(row, local_stage)
                    row += 1
            return matrix.tocsr()

        sparsity_matrix = sparsity()

        def semi_analytic_flow_jacobian(vector, f0, rel_step: float):
            if not self._truthy_param(self.get_param('semi_analytic_flow_jacobian', True)):
                return None

            J = lil_matrix(sparsity_matrix.shape, dtype=float)
            column_rows = [
                sparsity_matrix[:, col].nonzero()[0]
                for col in range(sparsity_matrix.shape[1])
            ]
            flow_columns = set()
            for stage in range(N):
                off = offsets(stage)
                flow_columns.add(off['R'])
                flow_columns.add(off['E'])
            nonlinear_columns = [
                col for col in range(n_vars)
                if col not in flow_columns
            ]
            groups = []
            group_rows = []
            for col in nonlinear_columns:
                rows = set(column_rows[col].tolist())
                for index, used_rows in enumerate(group_rows):
                    if rows.isdisjoint(used_rows):
                        groups[index].append(col)
                        used_rows.update(rows)
                        break
                else:
                    groups.append([col])
                    group_rows.append(set(rows))

            evaluations = 0
            for group in groups:
                step = np.zeros_like(vector)
                for col in group:
                    step[col] = rel_step * max(abs(vector[col]), 1.0)
                f_step = residual(vector + step)
                evaluations += 1
                diff = f_step - f0
                for col in group:
                    rows = column_rows[col]
                    if rows.size:
                        J[rows, col] = diff[rows] / step[col]

            Rv, Ev, xRv, xEv, Tv = decode(vector)
            hR = [phase_enthalpy(xRv[stage], Tv[stage]) for stage in range(N)]
            hE = [phase_enthalpy(xEv[stage], Tv[stage]) for stage in range(N)]

            def add(row_index: int, col_index: int, value: float) -> None:
                if value:
                    J[row_index, col_index] = J[row_index, col_index] + value

            for stage in range(N):
                row_base = stage * stage_row_count
                off = offsets(stage)
                for ci, comp in enumerate(comps):
                    row_index = row_base + ci
                    scale = component_scales[comp]
                    if stage > 0:
                        prev = offsets(stage - 1)
                        add(row_index, prev['R'], Rv[stage - 1] * xRv[stage - 1].get(comp, 0.0) / scale)
                    if stage < N - 1:
                        nxt = offsets(stage + 1)
                        add(row_index, nxt['E'], Ev[stage + 1] * xEv[stage + 1].get(comp, 0.0) / scale)
                    add(row_index, off['R'], -Rv[stage] * xRv[stage].get(comp, 0.0) / scale)
                    add(row_index, off['E'], -Ev[stage] * xEv[stage].get(comp, 0.0) / scale)

                if mode == 'adiabatic':
                    energy_row = row_base + 2 * nc
                    if stage > 0:
                        prev = offsets(stage - 1)
                        add(energy_row, prev['R'], Rv[stage - 1] * hR[stage - 1] / energy_scale)
                    if stage < N - 1:
                        nxt = offsets(stage + 1)
                        add(energy_row, nxt['E'], Ev[stage + 1] * hE[stage + 1] / energy_scale)
                    add(energy_row, off['R'], -Rv[stage] * hR[stage] / energy_scale)
                    add(energy_row, off['E'], -Ev[stage] * hE[stage] / energy_scale)

            return J.tocsr(), evaluations

        solver_options = {
            'mesh_tolerance': float(self.get_param('mesh_tolerance', 1e-5)),
            'acceptable_mesh_residual': float(
                self.get_param('acceptable_mesh_residual', max(5e-5, 20.0 * float(self.get_param('mesh_tolerance', 1e-5))))
            ),
            'max_iterations': int(self.get_param('max_iterations', self.get_param('max_evaluations', 80))),
            'max_jacobian_evaluations': int(self.get_param('max_jacobian_evaluations', 50)),
            'line_search_steps': int(self.get_param('line_search_steps', 14)),
            'finite_difference_rel_step': float(self.get_param('finite_difference_rel_step', 1e-6)),
        }
        solution = self._sparse_newton_solve(
            residual, sparsity_matrix, pack(), solver_options,
            jacobian=semi_analytic_flow_jacobian,
        )
        if not solution['success']:
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' equation-oriented MESH "
                f"solve failed (residual {solution['residual_norm']:.2e}): {solution['message']}"
            )

        R, E, x_raf, x_ext, T_stage = decode(solution['x'])
        raffinate = self.thermo.calculate_state(
            T_stage[-1], P, R[-1], x_raf[-1], phase='liquid', flash=False
        )
        extract = self.thermo.calculate_state(
            T_stage[0], P, E[0], x_ext[0], phase='liquid', flash=False
        )

        component_balance_error = self._external_component_balance_error(
            comps, feed, feed_z, solvent, solvent_z,
            R[-1], x_raf[-1], E[0], x_ext[0], flow_scale,
        )
        convergence_tolerance = float(self.get_param('mesh_tolerance', 1e-5))
        if component_balance_error > max(convergence_tolerance, solver_options['acceptable_mesh_residual']):
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' failed overall mass balance "
                f"(error {component_balance_error:.2e})"
            )

        stage_energy_residuals = self._stage_energy_residuals(
            comps, feed, feed_z, solvent, solvent_z, R, x_raf, E, x_ext, T_stage, P
        )
        total_heat_duty = sum(stage_energy_residuals) if mode == 'isothermal' else 0.0
        overall_energy_residual = (
            raffinate.F * (raffinate.H or 0.0)
            + extract.F * (extract.H or 0.0)
            - feed.F * (feed.H or 0.0)
            - solvent.F * (solvent.H or 0.0)
            - total_heat_duty
        )
        outlet_energy_scale = max(
            energy_scale,
            abs(raffinate.F * (raffinate.H or 0.0)) + abs(extract.F * (extract.H or 0.0)),
            1.0,
        )
        if quality_context is None:
            final_residual = residual(solution['x'])
        else:
            with quality_context(phase='residual_diagnostics', affects_result=False):
                final_residual = residual(solution['x'])
        diagnostics = {}
        for value, label in zip(final_residual, residual_labels):
            kind = label[0]
            diagnostics[kind] = max(diagnostics.get(kind, 0.0), abs(float(value)))

        warnings = list(initializer_warnings)
        if solution['residual_norm'] > convergence_tolerance:
            warnings.append(
                f"MESH solver accepted residual {solution['residual_norm']:.2e}"
            )

        return UnitResult(
            outlet_streams={'raffinate': raffinate, 'extract': extract},
            heat_duty=total_heat_duty,
            performance={
                'N_stages': N,
                'mode': mode,
                'solver': 'sparse_damped_newton',
                'solver_algorithm': 'equation_oriented',
                'initializer': initializer_used,
                'requested_initializer': requested_initializer,
                **(
                    {'coarse_initial_stages': int(guess['coarse_initial_stages'])}
                    if 'coarse_initial_stages' in guess else {}
                ),
                'jacobian_method': solution['jacobian_method'],
                'T_C': T - 273.15,
                'raffinate_T_C': raffinate.T - 273.15,
                'extract_T_C': extract.T - 273.15,
                'mesh_residual': solution['residual_norm'],
                'residual_diagnostics': diagnostics,
                'component_balance_error': component_balance_error,
                'max_stage_energy_residual_kJ_h': max(abs(value) for value in stage_energy_residuals),
                'max_stage_energy_relative_error': max(abs(value) for value in stage_energy_residuals) / outlet_energy_scale,
                'overall_energy_residual_kJ_h': overall_energy_residual,
                'overall_energy_relative_error': abs(overall_energy_residual) / outlet_energy_scale,
                'solver_iterations': solution['iterations'],
                'function_evaluations': solution['function_evaluations'],
                'jacobian_evaluations': solution['jacobian_evaluations'],
                'terminal_raffinate_stage_flow': float(R[-1]),
                'terminal_raffinate_stage_composition': x_raf[-1],
                'raffinate_flows': [float(value) for value in R],
                'extract_flows': [float(value) for value in E],
                'stage_temperatures_C': [float(value - 273.15) for value in T_stage],
                'stage_energy_residuals_kJ_h': [float(value) for value in stage_energy_residuals],
                'stage_raffinate_compositions': x_raf,
                'stage_extract_compositions': x_ext,
            },
            warnings=warnings,
        )

    def _store_recycle_profile(self, performance: dict) -> None:
        context = getattr(self, 'solve_context', {}) or {}
        if context.get('recycle_evaluation') is None:
            return
        if not self._truthy_param(self.get_param('recycle_warm_start', True)):
            return
        try:
            if performance.get('raffinate_flows') and performance.get('extract_flows'):
                self._last_recycle_profile = {'performance': dict(performance)}
        except (TypeError, ValueError):
            return

    def _external_component_balance_error(
        self,
        comps: list[str],
        feed: StreamState,
        feed_z: dict[str, float],
        solvent: StreamState,
        solvent_z: dict[str, float],
        raffinate_flow: float,
        raffinate_composition: dict[str, float],
        extract_flow: float,
        extract_composition: dict[str, float],
        flow_scale: float,
    ) -> float:
        return max(
            abs(
                feed.F * feed_z.get(comp, 0.0)
                + solvent.F * solvent_z.get(comp, 0.0)
                - raffinate_flow * raffinate_composition.get(comp, 0.0)
                - extract_flow * extract_composition.get(comp, 0.0)
            ) / flow_scale
            for comp in comps
        )

    def _stage_energy_residuals(
        self,
        comps: list[str],
        feed: StreamState,
        feed_z: dict[str, float],
        solvent: StreamState,
        solvent_z: dict[str, float],
        R: list[float],
        x_raf: list[dict[str, float]],
        E: list[float],
        x_ext: list[dict[str, float]],
        T_stage: list[float],
        P: float,
    ) -> list[float]:
        def state_enthalpy_flow(flow: float, comp: dict[str, float], temperature: float) -> float:
            if flow <= 0.0:
                return 0.0
            values = {c: max(float(comp.get(c, 0.0)), 0.0) for c in comps}
            total = sum(values.values())
            if total <= 0.0:
                return 0.0
            composition = {c: values[c] / total for c in comps}
            state = self.thermo.calculate_state(
                temperature, P, flow, composition,
                phase='liquid', flash=False, include=('H',)
            )
            return flow * (state.H or 0.0)

        residuals = []
        N = len(R)
        for stage in range(N):
            H_raf_in = (
                feed.F * (feed.H or 0.0)
                if stage == 0
                else state_enthalpy_flow(R[stage - 1], x_raf[stage - 1], T_stage[stage - 1])
            )
            H_ext_in = (
                solvent.F * (solvent.H or 0.0)
                if stage == N - 1
                else state_enthalpy_flow(E[stage + 1], x_ext[stage + 1], T_stage[stage + 1])
            )
            H_out = (
                state_enthalpy_flow(R[stage], x_raf[stage], T_stage[stage])
                + state_enthalpy_flow(E[stage], x_ext[stage], T_stage[stage])
            )
            residuals.append(H_out - H_raf_in - H_ext_in)
        return residuals

    def _sparse_newton_solve(self, residual, sparsity, x0, options: dict,
                             jacobian=None) -> dict:
        quality_context = getattr(getattr(self, 'thermo', None), 'quality_context', None)
        if (
            quality_context is not None
            and not getattr(self, '_quality_solver_aux_context_active', False)
        ):
            self._quality_solver_aux_context_active = True
            try:
                with quality_context(phase='solver_iteration', affects_result=False):
                    return self._sparse_newton_solve(
                        residual, sparsity, x0, options, jacobian=jacobian
                    )
            finally:
                self._quality_solver_aux_context_active = False

        import numpy as np
        from scipy.sparse import csc_matrix, eye
        from scipy.sparse.linalg import MatrixRankWarning, spsolve
        import warnings as py_warnings

        x = np.array(x0, dtype=float)
        f = residual(x)
        if not np.all(np.isfinite(f)):
            raise UnitOperationError(
                f"RigorousLiquidLiquidExtractor '{self.unit_id}' generated non-finite initial residuals"
            )

        tolerance = options['mesh_tolerance']
        acceptable_tolerance = max(options.get('acceptable_mesh_residual', tolerance), tolerance)
        max_iterations = options['max_iterations']
        max_jacobians = options['max_jacobian_evaluations']
        line_search_steps = options['line_search_steps']
        rel_step = options['finite_difference_rel_step']

        groups = self._color_jacobian_columns(sparsity)
        function_evaluations = 1
        jacobian_evaluations = 0
        used_model_jacobian = False
        message = "maximum iterations reached"
        last_iteration = 0

        for iteration in range(1, max_iterations + 1):
            last_iteration = iteration
            residual_norm = float(np.linalg.norm(f, ord=np.inf))
            if residual_norm < tolerance:
                return {
                    'success': True,
                    'x': x,
                    'residual_norm': residual_norm,
                    'iterations': iteration - 1,
                    'function_evaluations': function_evaluations,
                    'jacobian_evaluations': jacobian_evaluations,
                    'jacobian_method': (
                        'semi_analytic_flow'
                        if used_model_jacobian else 'colored_finite_difference'
                    ),
                    'message': 'converged',
                }
            if jacobian_evaluations >= max_jacobians:
                message = "maximum Jacobian evaluations reached"
                break

            jacobian_result = jacobian(x, f, rel_step) if jacobian is not None else None
            if jacobian_result is None:
                J, evals = self._finite_difference_jacobian(
                    residual, x, f, sparsity, groups, rel_step
                )
            else:
                J, evals = jacobian_result
                used_model_jacobian = True
            function_evaluations += evals
            jacobian_evaluations += 1

            dx = None
            for shift in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
                try:
                    with py_warnings.catch_warnings():
                        py_warnings.simplefilter('error', MatrixRankWarning)
                        matrix = csc_matrix(J)
                        if shift:
                            matrix = matrix + shift * eye(matrix.shape[0], matrix.shape[1], format='csc')
                        trial_dx = spsolve(matrix, -f)
                    if np.all(np.isfinite(trial_dx)):
                        dx = np.array(trial_dx, dtype=float)
                        break
                except Exception:
                    continue
            if dx is None:
                message = "linear Newton system could not be solved"
                break

            max_abs_step = float(np.max(np.abs(dx))) if dx.size else 0.0
            step_limit = float(self.get_param('newton_step_limit', 8.0))
            if max_abs_step > step_limit:
                dx *= step_limit / max_abs_step

            current_merit = 0.5 * float(np.dot(f, f))
            accepted = False
            best_x = x
            best_f = f
            best_merit = current_merit
            for attempt in range(line_search_steps):
                alpha = 0.5 ** attempt
                x_trial = x + alpha * dx
                f_trial = residual(x_trial)
                function_evaluations += 1
                if not np.all(np.isfinite(f_trial)):
                    continue
                trial_merit = 0.5 * float(np.dot(f_trial, f_trial))
                if trial_merit < best_merit:
                    best_merit = trial_merit
                    best_x = x_trial
                    best_f = f_trial
                if trial_merit <= current_merit * (1.0 - 1e-4 * alpha):
                    x = x_trial
                    f = f_trial
                    accepted = True
                    break
            if not accepted:
                if best_merit < current_merit:
                    x = best_x
                    f = best_f
                else:
                    message = "line search could not reduce the MESH residual"
                    break

        residual_norm = float(np.linalg.norm(f, ord=np.inf))
        return {
            'success': residual_norm < acceptable_tolerance,
            'x': x,
            'residual_norm': residual_norm,
            'iterations': last_iteration,
            'function_evaluations': function_evaluations,
            'jacobian_evaluations': jacobian_evaluations,
            'jacobian_method': (
                'semi_analytic_flow'
                if used_model_jacobian else 'colored_finite_difference'
            ),
            'message': message,
        }

    def _color_jacobian_columns(self, sparsity):
        column_rows = [
            set(sparsity[:, col].nonzero()[0].tolist())
            for col in range(sparsity.shape[1])
        ]
        groups: list[list[int]] = []
        group_rows: list[set[int]] = []
        for col, rows in enumerate(column_rows):
            for index, used_rows in enumerate(group_rows):
                if rows.isdisjoint(used_rows):
                    groups[index].append(col)
                    used_rows.update(rows)
                    break
            else:
                groups.append([col])
                group_rows.append(set(rows))
        return groups

    def _finite_difference_jacobian(self, residual, x, f0, sparsity, groups, rel_step: float):
        import numpy as np
        from scipy.sparse import lil_matrix

        J = lil_matrix(sparsity.shape, dtype=float)
        evaluations = 0
        column_rows = [
            sparsity[:, col].nonzero()[0]
            for col in range(sparsity.shape[1])
        ]
        for group in groups:
            step = np.zeros_like(x)
            for col in group:
                step[col] = rel_step * max(abs(x[col]), 1.0)
            f_step = residual(x + step)
            evaluations += 1
            diff = f_step - f0
            for col in group:
                rows = column_rows[col]
                if rows.size:
                    J[rows, col] = diff[rows] / step[col]
        return J.tocsr(), evaluations


class RigorousAbsorber(EquilibriumStageColumnMixin, UnitOperation):
    """Equation-oriented equilibrium-stage absorber.

    Inlets are gas/vapor and liquid/solvent feeds.  The default arrangement is
    counter-current: liquid feed on the top stage, gas feed on the bottom
    stage, vapor product from the top, and liquid product from the bottom.
    """

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        column_name = type(self).__name__

        if not inlets:
            raise UnitOperationError(f"{column_name} '{self.unit_id}' has no inlet streams")

        N = int(self.get_param('N_stages', self.get_param('stages', 5)))
        if N < 1:
            raise UnitOperationError(
                f"{column_name} '{self.unit_id}' requires N_stages >= 1"
            )
        mode = self._absorber_thermal_mode(column_name)

        positive_inlets = {
            port: stream for port, stream in inlets.items()
            if stream.F > 0.0
        }
        if not positive_inlets:
            raise UnitOperationError(
                f"{column_name} '{self.unit_id}' requires positive inlet flow"
            )
        gas_ports = [
            port for port, stream in positive_inlets.items()
            if self._absorber_feed_phase(port, stream) == 'vapor'
        ]
        liquid_ports = [
            port for port, stream in positive_inlets.items()
            if self._absorber_feed_phase(port, stream) == 'liquid'
        ]
        if not gas_ports or not liquid_ports:
            raise UnitOperationError(
                f"{column_name} '{self.unit_id}' needs at least one gas/vapor "
                "feed and one liquid/solvent feed"
            )

        aggregate, feed_specs = self._absorber_feed_specs(positive_inlets, N)
        comps = self._component_order(aggregate)
        if len(comps) < 2:
            raise UnitOperationError(
                f"{column_name} '{self.unit_id}' needs >= 2 active components"
            )
        feed_z = self._normalize({
            comp: aggregate.composition.get(comp, 0.0)
            for comp in comps
        })
        pressures = self._absorber_pressure_profile(N, aggregate.P)
        if any(P <= 0.0 for P in pressures):
            raise UnitOperationError(
                f"{column_name} '{self.unit_id}' pressure profile must be positive"
            )
        T_min, T_max = self._temperature_bounds(comps)
        quality_context = getattr(self.thermo, 'quality_context', None)
        if quality_context is None:
            aqueous_context, henry_info, henry_warnings = self._absorber_henry_context(
                comps, feed_specs
            )
        else:
            with quality_context(phase='henry_context_selection', affects_result=False):
                aqueous_context, henry_info, henry_warnings = self._absorber_henry_context(
                    comps, feed_specs
                )
        if aqueous_context is not None:
            self._absorber_apply_aqueous_feed_enthalpies(
                feed_specs, aqueous_context
            )
            aggregate.H = (
                sum(feed['F'] * feed['H'] for feed in feed_specs)
                / max(aggregate.F, 1e-30)
            )
        isothermal_T = None
        if mode == 'isothermal':
            isothermal_T = self._absorber_isothermal_temperature_profile(
                N, feed_specs, T_min, T_max
            )

        flow_scale = max(aggregate.F, 1.0)
        energy_scale = max(abs(sum(feed['F'] * feed['H'] for feed in feed_specs)), aggregate.F * 50000.0, 1.0)
        component_floor = float(self.get_param('component_scale_floor', 1e-4))
        component_scales = {
            comp: max(
                sum(feed['F'] * feed['z'].get(comp, 0.0) for feed in feed_specs),
                flow_scale * component_floor,
                1e-12,
            )
            for comp in comps
        }

        model = self._build_absorber_mesh_model(
            feed_specs, comps, N, pressures, flow_scale, energy_scale,
            component_scales, T_min, T_max, mode, isothermal_T,
            aqueous_context,
        )
        solver_options = {
            'mesh_tolerance': float(self.get_param('mesh_tolerance', 2e-6)),
            'acceptable_mesh_residual': float(self.get_param('acceptable_mesh_residual', 1e-4)),
            'max_iterations': int(self.get_param('max_iterations', self.get_param('max_evaluations', 80))),
            'max_jacobian_evaluations': int(self.get_param('max_jacobian_evaluations', 80)),
            'line_search_steps': int(self.get_param('line_search_steps', 16)),
            'finite_difference_rel_step': float(self.get_param('finite_difference_rel_step', 1e-6)),
        }
        initializers = {
            'equilibrium_sweep': self._absorber_initial_guess,
            'loading_profile': self._absorber_loading_initial_guess,
        }
        attempts = []
        solution = None
        selected_initializer = None
        initializer_order = self._absorber_initializer_order()
        trace_initializer_fraction = float(self.get_param(
            'trace_initializer_fraction',
            1.0e-5,
        ))
        if (
            not math.isfinite(trace_initializer_fraction)
            or trace_initializer_fraction < 0.0
        ):
            raise UnitOperationError(
                f"{column_name} '{self.unit_id}' requires a finite, "
                "nonnegative trace_initializer_fraction"
            )
        trace_initializer_components = tuple(
            comp
            for comp in comps
            if 0.0 < sum(
                feed['F'] * feed['z'].get(comp, 0.0)
                for feed in feed_specs
            ) < flow_scale * trace_initializer_fraction
        )
        if trace_initializer_components and 'loading_profile' in initializer_order:
            initializer_order = (
                'loading_profile',
                *(item for item in initializer_order if item != 'loading_profile'),
            )
        if aqueous_context is not None and any(
            item.get('reason') == 'dilute_aqueous_upper_bound'
            for item in henry_info['components'].values()
        ):
            initializer_order = ('loading_profile', 'equilibrium_sweep')
        for initializer_name in initializer_order:
            try:
                if quality_context is None:
                    initial = initializers[initializer_name](
                        comps, feed_specs, N, pressures, T_min, T_max,
                        aqueous_context,
                    )
                else:
                    with quality_context(phase='initializer', affects_result=False):
                        initial = initializers[initializer_name](
                            comps, feed_specs, N, pressures, T_min, T_max,
                            aqueous_context,
                        )
                if isothermal_T is not None:
                    initial['T'] = list(isothermal_T)
                z0 = self._absorber_pack_variables(
                    initial['T'], initial['x'], initial['L'], initial['V'],
                    comps, T_min, T_max,
                )
                candidate = self._sparse_newton_solve(
                    model['residual'], model['sparsity'], z0, solver_options,
                    jacobian=model.get('jacobian'),
                )
                attempts.append({
                    'initializer': initializer_name,
                    'success': bool(candidate['success']),
                    'residual': float(candidate['residual_norm']),
                    'iterations': int(candidate['iterations']),
                })
                if candidate['success']:
                    solution = candidate
                    selected_initializer = initializer_name
                    break
            except (UnitOperationError, ValueError, ArithmeticError, RuntimeError) as exc:
                attempts.append({
                    'initializer': initializer_name,
                    'success': False,
                    'error': str(exc),
                })

        if solution is None:
            last_attempt = attempts[-1] if attempts else {}
            raise UnitOperationError(
                f"{column_name} '{self.unit_id}' MESH solve failed "
                f"after {len(attempts)} initializer attempt(s)"
                + (
                    f" (last residual {last_attempt['residual']:.2e})"
                    if 'residual' in last_attempt else ''
                )
            )

        decoded = model['decode'](solution['x'])
        T = decoded['T']
        x = decoded['x']
        L = decoded['L']
        V = decoded['V']
        stage_props = [
            model['stage_properties'](stage, T[stage], x[stage])
            for stage in range(N)
        ]
        y = [props['y'] for props in stage_props]
        hL = [props['hL'] for props in stage_props]
        hV = [props['hV'] for props in stage_props]

        gas_comp = {comp: float(y[0].get(comp, 0.0)) for comp in comps}
        liquid_comp = {comp: float(x[-1].get(comp, 0.0)) for comp in comps}
        gas_out = self.thermo.calculate_state(
            float(T[0]), float(pressures[0]), float(V[0]), gas_comp,
            phase='vapor', flash=False,
        )
        liquid_out = self.thermo.calculate_state(
            float(T[-1]), float(pressures[-1]), float(L[-1]), liquid_comp,
            phase='liquid', flash=False,
        )
        if aqueous_context is not None:
            liquid_out.H = self.thermo.aqueous_liquid_enthalpy(
                liquid_comp,
                float(T[-1]),
                float(pressures[-1]),
                aqueous_context,
            )

        component_balance_error = self._absorber_component_balance_error(
            comps, feed_specs, [gas_out, liquid_out], flow_scale
        )
        if quality_context is None:
            final_residual = model['residual'](solution['x'])
        else:
            with quality_context(phase='residual_diagnostics', affects_result=False):
                final_residual = model['residual'](solution['x'])
        diagnostics = self._rigorous2_residual_diagnostics(
            final_residual, model['residual_labels']
        )
        H_in = sum(feed['F'] * feed['H'] for feed in feed_specs)
        H_out = gas_out.F * gas_out.H + liquid_out.F * liquid_out.H
        heat_duty = 0.0 if mode == 'adiabatic' else H_out - H_in
        stage_energy_residuals = model['stage_energy_residuals'](T, x, L, V)
        overall_energy_residual = H_out - H_in - heat_duty
        warnings = list(henry_warnings)
        if solution['residual_norm'] > solver_options['mesh_tolerance']:
            warnings.append(
                f"MESH solver accepted residual {solution['residual_norm']:.2e}"
            )
        if aqueous_context is not None:
            warnings.extend(self._absorber_henry_solution_warnings(
                aqueous_context,
                henry_info,
                x,
                T,
                pressures,
            ))

        absorption = {}
        for comp in comps:
            gas_in_comp = sum(
                feed['F'] * feed['z'].get(comp, 0.0)
                for feed in feed_specs
                if feed['phase_hint'] == 'vapor'
            )
            gas_out_comp = gas_out.F * gas_out.composition.get(comp, 0.0)
            if gas_in_comp > 1e-30:
                absorption[comp] = (gas_in_comp - gas_out_comp) / gas_in_comp

        return UnitResult(
            outlet_streams={'gas_out': gas_out, 'liquid_out': liquid_out},
            heat_duty=heat_duty,
            performance={
                'method': 'rigorous_absorber_mesh',
                'mode': mode,
                'N_stages': N,
                'feeds': [
                    {
                        'port': feed['port'],
                        'stage': int(feed['stage_number']),
                        'flow': float(feed['F']),
                        'phase_hint': feed['phase_hint'],
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
                'gas_out_flow': float(gas_out.F),
                'liquid_out_flow': float(liquid_out.F),
                'gas_out_temperature_C': float(gas_out.T - 273.15),
                'liquid_out_temperature_C': float(liquid_out.T - 273.15),
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
                    {comp: float(props['K'].get(comp, 0.0)) for comp in comps}
                    for props in stage_props
                ],
                'liquid_flows': [float(value) for value in L],
                'vapor_flows': [float(value) for value in V],
                'henry': henry_info,
                'component_absorption_fraction': absorption,
                'mesh_residual': float(solution['residual_norm']),
                'component_balance_error': float(component_balance_error),
                'overall_energy_residual_kJ_h': float(overall_energy_residual),
                'overall_energy_relative_error': (
                    abs(overall_energy_residual)
                    / max(abs(H_in), abs(H_out), abs(heat_duty), 1.0)
                ),
                'stage_energy_residuals_kJ_h': [
                    float(value) for value in stage_energy_residuals
                ],
                'stage_heat_duties_kJ_h': [
                    float(-value) for value in stage_energy_residuals
                ],
                'duty_kW': float(heat_duty / 3600.0),
                'solver': 'sparse_damped_newton',
                'initializer': selected_initializer,
                'initializer_order': tuple(initializer_order),
                'initializer_attempts': attempts,
                'trace_initializer_components': trace_initializer_components,
                'jacobian_method': str(solution.get('jacobian_method', 'colored_finite_difference')),
                'solver_iterations': int(solution['iterations']),
                'function_evaluations': int(solution['function_evaluations']),
                'jacobian_evaluations': int(solution['jacobian_evaluations']),
                'finite_difference_rel_step': float(solver_options['finite_difference_rel_step']),
                'residual_diagnostics': diagnostics,
            },
            warnings=warnings,
        )

    def _absorber_feed_phase(self, port: str, stream: StreamState) -> str:
        port_key = str(port).strip().lower().replace('-', '_')
        gas_tokens = ('gas', 'vapor', 'vapour', 'air')
        liquid_tokens = ('liquid', 'solvent', 'lean', 'water')
        if any(token in port_key for token in gas_tokens):
            return 'vapor'
        if any(token in port_key for token in liquid_tokens):
            return 'liquid'
        return 'vapor' if stream.vapor_fraction > 0.5 else 'liquid'

    def _absorber_thermal_mode(self, column_name: str) -> str:
        explicit_T = (
            self.get_param('T') is not None
            or self.get_param('temperature') is not None
            or self.get_param('stage_temperature') is not None
            or self.get_param('stage_temperatures') is not None
            or self.get_param('temperature_profile') is not None
        )
        default_mode = 'isothermal' if explicit_T else 'adiabatic'
        mode = str(self.get_param('mode', default_mode)).strip().lower()
        aliases = {
            '': 'adiabatic',
            'adiabatic': 'adiabatic',
            'mesh': 'adiabatic',
            'equilibrium': 'adiabatic',
            'adiabatic_mesh': 'adiabatic',
            'isothermal': 'isothermal',
            'fixed_t': 'isothermal',
            'fixed-temperature': 'isothermal',
            'fixed_temperature': 'isothermal',
            'specified_temperature': 'isothermal',
        }
        if mode not in aliases:
            raise UnitOperationError(
                f"{column_name} '{self.unit_id}' mode must be adiabatic or isothermal"
            )
        return aliases[mode]

    def _absorber_isothermal_temperature_profile(
        self,
        N: int,
        feed_specs: list[dict],
        T_min: float,
        T_max: float,
    ) -> list[float]:
        profile = self.get_param(
            'stage_temperatures',
            self.get_param('temperature_profile'),
        )
        if profile is not None:
            if isinstance(profile, str):
                values = [
                    item.strip()
                    for item in profile.replace(';', ',').split(',')
                    if item.strip()
                ]
            else:
                values = list(profile)
            if len(values) != N:
                raise UnitOperationError(
                    f"{type(self).__name__} '{self.unit_id}' temperature profile "
                    f"must contain {N} stage values"
                )
            temperatures = [self._absorber_temperature_value(value) for value in values]
        else:
            T_spec = self.get_temperature_param('T')
            if T_spec is None:
                T_spec = self.get_temperature_param('temperature')
            if T_spec is None:
                T_spec = self.get_temperature_param('stage_temperature')
            if T_spec is None:
                total_flow = sum(feed['F'] for feed in feed_specs)
                T_spec = (
                    sum(feed['F'] * feed['T'] for feed in feed_specs)
                    / max(total_flow, 1e-30)
                )
            temperatures = [self._absorber_temperature_value(T_spec) for _ in range(N)]

        for T in temperatures:
            if T <= T_min or T >= T_max:
                raise UnitOperationError(
                    f"{type(self).__name__} '{self.unit_id}' isothermal stage "
                    f"temperature {T:.6g} K is outside solver bounds "
                    f"{T_min:.6g}-{T_max:.6g} K"
                )
        return temperatures

    def _absorber_temperature_value(self, value) -> float:
        T = float(value)
        if T < 200.0:
            T += 273.15
        if not math.isfinite(T) or T <= 0.0:
            raise UnitOperationError(
                f"{type(self).__name__} '{self.unit_id}' temperature must be positive"
            )
        return T

    def _absorber_feed_specs(
        self,
        inlets: dict[str, StreamState],
        N: int,
    ) -> tuple[StreamState, list[dict]]:
        feed_stage_map = self.get_param('feed_stages', {}) or {}
        if isinstance(feed_stage_map, str):
            parsed = {}
            for item in feed_stage_map.replace(';', ',').split(','):
                item = item.strip()
                if not item:
                    continue
                if ':' in item:
                    key, value = item.split(':', 1)
                elif '=' in item:
                    key, value = item.split('=', 1)
                else:
                    continue
                parsed[key.strip()] = value.strip()
            feed_stage_map = parsed

        specs = []
        component_moles: dict[str, float] = {}
        total_flow = 0.0
        enthalpy_flow = 0.0
        vapor_flow = 0.0
        weighted_T = 0.0
        weighted_P = 0.0

        for index, (port, stream) in enumerate(inlets.items()):
            phase_hint = self._absorber_feed_phase(port, stream)
            default_stage = (
                self._gas_feed_default_stage(N)
                if phase_hint == 'vapor'
                else self._liquid_feed_default_stage(N)
            )
            stage_map = dict(feed_stage_map) if isinstance(feed_stage_map, dict) else {}
            stage_map.setdefault(port, default_stage)
            stage = self._feed_stage_for_port(
                port, index, N, default_stage, stage_map
            )
            z = self._normalize(dict(stream.composition))
            h = stream.H
            if h is None:
                h = self.thermo.mixture_enthalpy(
                    z, stream.T, stream.vapor_fraction, P=stream.P
                )
            specs.append({
                'port': port,
                'stage': stage - 1,
                'stage_number': stage,
                'phase_hint': phase_hint,
                'F': float(stream.F),
                'z': z,
                'H': float(h),
                'T': float(stream.T),
                'P': float(stream.P),
                'vapor_fraction': float(stream.vapor_fraction),
            })
            total_flow += stream.F
            enthalpy_flow += stream.F * h
            vapor_flow += stream.F * stream.vapor_fraction
            weighted_T += stream.F * stream.T
            weighted_P += stream.F * stream.P
            for comp, frac in z.items():
                component_moles[comp] = component_moles.get(comp, 0.0) + stream.F * frac

        composition = {
            comp: value / total_flow
            for comp, value in component_moles.items()
            if value > 0.0
        }
        aggregate = self.thermo.calculate_state(
            weighted_T / total_flow,
            weighted_P / total_flow,
            total_flow,
            composition,
            phase='vapor' if vapor_flow / total_flow > 0.5 else 'liquid',
            flash=False,
        )
        aggregate.vapor_fraction = vapor_flow / total_flow
        aggregate.H = enthalpy_flow / total_flow
        return aggregate, specs

    def _gas_feed_default_stage(self, N: int) -> int:
        return int(self.get_param('gas_stage', self.get_param('vapor_stage', N)))

    def _liquid_feed_default_stage(self, N: int) -> int:
        return int(self.get_param('liquid_stage', self.get_param('solvent_stage', 1)))

    def _absorber_pressure_profile(self, N: int, reference_pressure: float) -> list[float]:
        if self.get_param('stage_pressures', self.get_param('pressure_profile')) is not None:
            return self._pressure_profile(N, reference_pressure)

        import numpy as np

        P_top = float(self.get_param('P_top', self.get_param('P', reference_pressure)))
        P_bottom = self.get_param('P_bottom')
        if P_bottom is not None:
            return [float(value) for value in np.linspace(P_top, float(P_bottom), N)]
        P_drop = float(self.get_param('P_drop_per_stage', 0.01))
        return [P_top + stage * P_drop for stage in range(N)]

    def _absorber_water_component(self, comps: list[str]) -> str | None:
        return self._henry_water_component(comps)

    def _absorber_noncondensable_candidate(self, comp: str, reference_T: float) -> bool:
        return self._henry_noncondensable_candidate(comp, reference_T)

    def _absorber_henry_context(
        self,
        comps: list[str],
        feed_specs: list[dict],
    ):
        water = self._henry_water_component(comps)
        liquid_feeds = [
            feed for feed in feed_specs
            if feed['phase_hint'] == 'liquid'
        ]
        solvent_liquid_moles = {
            comp: sum(
                feed['F'] * feed['z'].get(comp, 0.0)
                for feed in liquid_feeds
            )
            for comp in comps
        }
        liquid_water = solvent_liquid_moles.get(water, 0.0)

        mode = self._absorber_thermal_mode(type(self).__name__)
        reference_T = min(feed['T'] for feed in liquid_feeds)
        if mode == 'isothermal':
            profile = self.get_param(
                'stage_temperatures',
                self.get_param('temperature_profile'),
            )
            if profile is not None:
                if isinstance(profile, str):
                    values = [
                        item.strip()
                        for item in profile.replace(';', ',').split(',')
                        if item.strip()
                    ]
                else:
                    values = list(profile)
                if values:
                    reference_T = min(
                        self._absorber_temperature_value(value)
                        for value in values
                    )
            else:
                T_spec = self.get_temperature_param('T')
                if T_spec is None:
                    T_spec = self.get_temperature_param('temperature')
                if T_spec is None:
                    T_spec = self.get_temperature_param('stage_temperature')
                if T_spec is not None:
                    reference_T = float(T_spec)
                else:
                    reference_T = (
                        sum(feed['F'] * feed['T'] for feed in feed_specs)
                        / max(sum(feed['F'] for feed in feed_specs), 1e-30)
                    )

        noncondensable = {
            comp: self._henry_noncondensable_candidate(comp, reference_T)
            for comp in comps
            if comp != water
        }
        potential_liquid_moles = {comp: 0.0 for comp in comps}
        total_available_moles = {comp: 0.0 for comp in comps}
        for feed in feed_specs:
            for comp in comps:
                amount = feed['F'] * feed['z'].get(comp, 0.0)
                total_available_moles[comp] += amount
                if (
                    feed['phase_hint'] == 'liquid'
                    or not noncondensable.get(comp, False)
                ):
                    potential_liquid_moles[comp] += amount

        return self._henry_context_from_estimates(
            comps,
            reference_T=reference_T,
            solvent_liquid_moles=solvent_liquid_moles,
            potential_liquid_moles=potential_liquid_moles,
            total_available_moles=total_available_moles,
            aqueous_loading_water_moles=liquid_water,
        )

    def _absorber_henry_solution_warnings(
        self,
        aqueous_context,
        henry_info: dict,
        liquid_compositions: list[dict[str, float]],
        temperatures: list[float],
        pressures: list[float] | None = None,
    ) -> list[str]:
        return self._henry_solution_warnings(
            aqueous_context,
            henry_info,
            liquid_compositions,
            temperatures,
            pressures,
        )

    def _absorber_apply_aqueous_feed_enthalpies(
        self,
        feed_specs: list[dict],
        aqueous_context,
    ) -> None:
        for feed in feed_specs:
            if feed['phase_hint'] != 'liquid':
                continue
            feed['H'] = float(self.thermo.aqueous_liquid_enthalpy(
                feed['z'],
                feed['T'],
                feed['P'],
                aqueous_context,
            ))

    def _absorber_initializer_order(self) -> tuple[str, ...]:
        return ('loading_profile', 'equilibrium_sweep')

    def _absorber_stage_K_values(
        self,
        comps: list[str],
        T: float,
        P: float,
        x: dict[str, float],
        aqueous_context=None,
    ) -> dict[str, float]:
        if aqueous_context is not None:
            raw = self.thermo.aqueous_K_values(T, P, x, aqueous_context)
            return {
                comp: min(max(float(raw.get(comp, 1.0)), 1e-12), 1e12)
                for comp in comps
            }
        if hasattr(self.thermo, 'K_values'):
            try:
                raw = self.thermo.K_values(T, P, x)
                return {
                    comp: min(max(float(raw.get(comp, 1.0)), 1e-12), 1e12)
                    for comp in comps
                }
            except Exception:
                pass
        if hasattr(self.thermo, 'activity_coefficients'):
            gamma = self.thermo.activity_coefficients(T, x)
            return {
                comp: min(
                    max(
                        gamma.get(comp, 1.0) * self.thermo.Psat(comp, T) / P,
                        1e-12,
                    ),
                    1e12,
                )
                for comp in comps
            }
        return {
            comp: min(max(self.thermo.K_value(comp, T, P), 1e-12), 1e12)
            for comp in comps
        }

    def _absorber_bubble_temperature(
        self,
        comps: list[str],
        x: dict[str, float],
        P: float,
        T_min: float,
        T_max: float,
        T_guess: float,
        aqueous_context=None,
    ) -> float:
        from scipy.optimize import brentq, minimize_scalar

        def residual(T: float) -> float:
            K = self._absorber_stage_K_values(
                comps, float(T), P, x, aqueous_context
            )
            return sum(K[comp] * x.get(comp, 0.0) for comp in comps) - 1.0

        lo = max(float(T_min) + 1e-6, 1.0)
        hi = float(T_max) - 1e-6
        f_lo = residual(lo)
        f_hi = residual(hi)
        if f_lo * f_hi <= 0.0:
            return float(brentq(residual, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=100))

        bounded_guess = min(max(float(T_guess), lo), hi)
        best_T = bounded_guess
        best_error = abs(residual(best_T))
        try:
            minimum = minimize_scalar(
                lambda T: residual(float(T)) ** 2,
                bounds=(lo, hi),
                method='bounded',
                options={'xatol': 1e-7},
            )
            candidate = float(minimum.x)
            candidate_error = abs(residual(candidate))
            if candidate_error < best_error:
                best_T = candidate
        except Exception:
            pass
        return best_T

    def _absorber_initial_guess(
        self,
        comps: list[str],
        feed_specs: list[dict],
        N: int,
        pressures: list[float],
        T_min: float,
        T_max: float,
        aqueous_context=None,
    ) -> dict:
        import numpy as np
        from scipy.optimize import lsq_linear

        liquid_feeds = [feed for feed in feed_specs if feed['phase_hint'] == 'liquid']
        vapor_feeds = [feed for feed in feed_specs if feed['phase_hint'] == 'vapor']
        liquid_flow = sum(feed['F'] for feed in liquid_feeds)
        gas_flow = sum(feed['F'] for feed in vapor_feeds)
        liquid_T = sum(feed['F'] * feed['T'] for feed in liquid_feeds) / max(liquid_flow, 1e-30)
        gas_T = sum(feed['F'] * feed['T'] for feed in vapor_feeds) / max(gas_flow, 1e-30)

        liquid_z = self._normalize({
            comp: sum(feed['F'] * feed['z'].get(comp, 0.0) for feed in liquid_feeds)
            for comp in comps
        })
        vapor_z = self._normalize({
            comp: sum(feed['F'] * feed['z'].get(comp, 0.0) for feed in vapor_feeds)
            for comp in comps
        })

        # Phase-tagged feed contributions give flow profiles that satisfy the
        # total balance on every stage before mass transfer is introduced.
        L = [
            max(sum(
                feed['F']
                for feed in liquid_feeds
                if int(feed['stage']) <= stage
            ), 1e-8)
            for stage in range(N)
        ]
        V = [
            max(sum(
                feed['F']
                for feed in vapor_feeds
                if int(feed['stage']) >= stage
            ), 1e-8)
            for stage in range(N)
        ]

        x = []
        T = []
        for stage in range(N):
            frac = stage / max(N - 1, 1)
            T_stage = (1.0 - frac) * liquid_T + frac * gas_T
            K = self._absorber_stage_K_values(
                comps, T_stage, pressures[stage], liquid_z,
                aqueous_context,
            )
            x_stage = self._normalize({
                comp: (
                    max(liquid_z.get(comp, 0.0), 1e-30)
                    + vapor_z.get(comp, 0.0) / K[comp]
                )
                for comp in comps
            })
            T_stage = self._absorber_bubble_temperature(
                comps, x_stage, pressures[stage], T_min, T_max, T_stage,
                aqueous_context,
            )
            x.append(x_stage)
            T.append(T_stage)

        stage_component_feeds = [
            {
                comp: sum(
                    feed['F'] * feed['z'].get(comp, 0.0)
                    for feed in feed_specs
                    if int(feed['stage']) == stage
                )
                for comp in comps
            }
            for stage in range(N)
        ]
        component_scales = {
            comp: max(
                sum(feed['F'] * feed['z'].get(comp, 0.0) for feed in feed_specs),
                (liquid_flow + gas_flow) * 1e-10,
                1e-12,
            )
            for comp in comps
        }
        relaxation = float(self.get_param('initializer_relaxation', 0.65))
        relaxation = min(max(relaxation, 0.05), 1.0)
        iterations = max(1, int(self.get_param('initializer_iterations', 4)))

        for _ in range(iterations):
            stage_K = [
                self._absorber_stage_K_values(
                    comps, T[stage], pressures[stage], x[stage],
                    aqueous_context,
                )
                for stage in range(N)
            ]
            solved = [{comp: 0.0 for comp in comps} for _ in range(N)]
            for comp in comps:
                matrix = np.zeros((N, N), dtype=float)
                rhs = np.array([
                    stage_component_feeds[stage][comp]
                    for stage in range(N)
                ], dtype=float)
                for stage in range(N):
                    matrix[stage, stage] = L[stage] + V[stage] * stage_K[stage][comp]
                    if stage > 0:
                        matrix[stage, stage - 1] = -L[stage - 1]
                    if stage < N - 1:
                        matrix[stage, stage + 1] = -V[stage + 1] * stage_K[stage + 1][comp]
                try:
                    component_profile = np.linalg.solve(matrix, rhs)
                except np.linalg.LinAlgError:
                    component_profile = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
                for stage in range(N):
                    solved[stage][comp] = max(float(component_profile[stage]), 1e-30)

            for stage in range(N):
                solved_stage = self._normalize(solved[stage])
                blended = self._normalize({
                    comp: (
                        max(x[stage].get(comp, 0.0), 1e-30) ** (1.0 - relaxation)
                        * max(solved_stage.get(comp, 0.0), 1e-30) ** relaxation
                    )
                    for comp in comps
                })
                bubble_T = self._absorber_bubble_temperature(
                    comps, blended, pressures[stage], T_min, T_max, T[stage],
                    aqueous_context,
                )
                x[stage] = blended
                T[stage] = (
                    (1.0 - relaxation) * T[stage]
                    + relaxation * bubble_T
                )

            flow_matrix = np.zeros((N * len(comps), 2 * N), dtype=float)
            flow_rhs = np.zeros(N * len(comps), dtype=float)
            stage_y = []
            for stage in range(N):
                K = self._absorber_stage_K_values(
                    comps, T[stage], pressures[stage], x[stage],
                    aqueous_context,
                )
                kx = {
                    comp: K[comp] * x[stage].get(comp, 0.0)
                    for comp in comps
                }
                total = max(sum(kx.values()), 1e-30)
                stage_y.append({comp: value / total for comp, value in kx.items()})

            for stage in range(N):
                for ci, comp in enumerate(comps):
                    row = stage * len(comps) + ci
                    scale = component_scales[comp]
                    flow_matrix[row, stage] = x[stage].get(comp, 0.0) / scale
                    flow_matrix[row, N + stage] = stage_y[stage].get(comp, 0.0) / scale
                    if stage > 0:
                        flow_matrix[row, stage - 1] -= x[stage - 1].get(comp, 0.0) / scale
                    if stage < N - 1:
                        flow_matrix[row, N + stage + 1] -= stage_y[stage + 1].get(comp, 0.0) / scale
                    flow_rhs[row] = stage_component_feeds[stage][comp] / scale

            try:
                reconciled = lsq_linear(
                    flow_matrix,
                    flow_rhs,
                    bounds=(1e-8, np.inf),
                    lsmr_tol='auto',
                    max_iter=100,
                ).x
                for stage in range(N):
                    L[stage] = (
                        max(L[stage], 1e-8) ** (1.0 - relaxation)
                        * max(float(reconciled[stage]), 1e-8) ** relaxation
                    )
                    V[stage] = (
                        max(V[stage], 1e-8) ** (1.0 - relaxation)
                        * max(float(reconciled[N + stage]), 1e-8) ** relaxation
                    )
            except Exception:
                pass

        for stage in range(N):
            T[stage] = self._absorber_bubble_temperature(
                comps, x[stage], pressures[stage], T_min, T_max, T[stage],
                aqueous_context,
            )
        return {'T': T, 'x': x, 'L': L, 'V': V}

    def _absorber_loading_initial_guess(
        self,
        comps: list[str],
        feed_specs: list[dict],
        N: int,
        pressures: list[float],
        T_min: float,
        T_max: float,
        aqueous_context=None,
    ) -> dict:
        liquid_moles = {comp: 0.0 for comp in comps}
        gas_moles = {comp: 0.0 for comp in comps}
        liquid_flow = 0.0
        gas_flow = 0.0
        liquid_T = 0.0
        gas_T = 0.0
        for feed in feed_specs:
            target = gas_moles if feed['phase_hint'] == 'vapor' else liquid_moles
            if feed['phase_hint'] == 'vapor':
                gas_flow += feed['F']
                gas_T += feed['F'] * feed['T']
            else:
                liquid_flow += feed['F']
                liquid_T += feed['F'] * feed['T']
            for comp in comps:
                target[comp] += feed['F'] * feed['z'].get(comp, 0.0)

        top_liquid = self._normalize({
            comp: liquid_moles.get(comp, 0.0) + 1e-8 * max(gas_moles.get(comp, 0.0), 0.0)
            for comp in comps
        })
        bottom_liquid = self._normalize({
            comp: liquid_moles.get(comp, 0.0) + 0.05 * max(gas_moles.get(comp, 0.0), 0.0)
            for comp in comps
        })
        top_T = liquid_T / max(liquid_flow, 1e-30)
        bottom_T = gas_T / max(gas_flow, 1e-30)

        x = []
        T = []
        for stage in range(N):
            frac = stage / max(N - 1, 1)
            x_stage = self._normalize({
                comp: (
                    (1.0 - frac) * max(top_liquid.get(comp, 0.0), 1e-30)
                    + frac * max(bottom_liquid.get(comp, 0.0), 1e-30)
                )
                for comp in comps
            })
            x.append(x_stage)
            T_guess = (1.0 - frac) * top_T + frac * bottom_T
            try:
                T_bubble = self._absorber_bubble_temperature(
                    comps,
                    x_stage,
                    pressures[stage],
                    T_min,
                    T_max,
                    T_guess,
                    aqueous_context,
                )
                T_guess = 0.5 * T_guess + 0.5 * T_bubble
            except Exception:
                pass
            T.append(min(max(float(T_guess), T_min + 1e-6), T_max - 1e-6))

        return {
            'T': T,
            'x': x,
            'L': [max(liquid_flow, 1e-8) for _ in range(N)],
            'V': [max(gas_flow, 1e-8) for _ in range(N)],
        }

    def _absorber_pack_variables(
        self,
        T: list[float],
        x: list[dict[str, float]],
        L: list[float],
        V: list[float],
        comps: list[str],
        T_min: float,
        T_max: float,
    ):
        import numpy as np

        span = T_max - T_min
        values = []
        for T_stage in T:
            reduced = min(max((T_stage - T_min) / span, 1e-8), 1.0 - 1e-8)
            values.append(math.log(reduced / (1.0 - reduced)))
        for x_stage in x:
            last = max(x_stage.get(comps[-1], 0.0), 1e-30)
            for comp in comps[:-1]:
                values.append(math.log(max(x_stage.get(comp, 0.0), 1e-30) / last))
        values.extend(math.log(max(value, 1e-14)) for value in L)
        values.extend(math.log(max(value, 1e-14)) for value in V)
        return np.array(values, dtype=float)

    def _build_absorber_mesh_model(
        self,
        feed_specs: list[dict],
        comps: list[str],
        N: int,
        pressures: list[float],
        flow_scale: float,
        energy_scale: float,
        component_scales: dict[str, float],
        T_min: float,
        T_max: float,
        mode: str,
        isothermal_T: list[float] | None,
        aqueous_context=None,
    ) -> dict:
        import numpy as np
        from scipy.sparse import csr_matrix, lil_matrix

        nc = len(comps)
        logits_start = N
        L_start = logits_start + N * (nc - 1)
        V_start = L_start + N
        n_vars = V_start + N
        span = T_max - T_min
        stage_feeds = [[] for _ in range(N)]
        for feed in feed_specs:
            stage_feeds[int(feed['stage'])].append(feed)
        if mode == 'isothermal':
            if isothermal_T is None or len(isothermal_T) != N:
                raise UnitOperationError(
                    f"{type(self).__name__} '{self.unit_id}' requires one "
                    "isothermal temperature per stage"
                )
            temperature_targets = [float(value) for value in isothermal_T]
        else:
            temperature_targets = None

        def decode(vector):
            theta = np.clip(np.array(vector[:N], dtype=float), -60.0, 60.0)
            T = T_min + span / (1.0 + np.exp(-theta))
            x = []
            index = logits_start
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
            L = np.exp(np.clip(vector[L_start:L_start + N], -40.0, 40.0))
            V = np.exp(np.clip(vector[V_start:V_start + N], -40.0, 40.0))
            return {'T': T, 'x': x, 'L': L, 'V': V}

        def stage_K_values(T_stage: float, P_stage: float, x_stage: dict[str, float]):
            return self._absorber_stage_K_values(
                comps, T_stage, P_stage, x_stage, aqueous_context
            )

        def stage_properties(stage: int, T_stage: float, x_stage: dict[str, float]):
            K = stage_K_values(float(T_stage), pressures[stage], x_stage)
            kx = {
                comp: max(K[comp] * x_stage.get(comp, 0.0), 0.0)
                for comp in comps
            }
            total = sum(kx.values())
            if total <= 0.0:
                y_stage = dict(x_stage)
            else:
                y_stage = {comp: value / total for comp, value in kx.items()}
            liquid_enthalpy = (
                self.thermo.aqueous_liquid_enthalpy(
                    x_stage,
                    float(T_stage),
                    float(pressures[stage]),
                    aqueous_context,
                )
                if aqueous_context is not None
                else self.thermo.mixture_enthalpy(
                    x_stage, float(T_stage), vapor_fraction=0.0,
                    P=float(pressures[stage]),
                )
            )
            return {
                'K': K,
                'y': y_stage,
                'bubble': sum(K[comp] * x_stage.get(comp, 0.0) for comp in comps) - 1.0,
                'hL': liquid_enthalpy,
                'hV': self.thermo.mixture_enthalpy(
                    y_stage, float(T_stage), vapor_fraction=1.0,
                    P=float(pressures[stage]),
                ),
            }

        residual_labels = []

        def stage_energy_residuals(T, x, L, V):
            props = [stage_properties(stage, T[stage], x[stage]) for stage in range(N)]
            y = [item['y'] for item in props]
            hL = [item['hL'] for item in props]
            hV = [item['hV'] for item in props]
            residuals = []
            for stage in range(N):
                liquid_in_flow = 0.0 if stage == 0 else L[stage - 1]
                liquid_in_h = 0.0 if stage == 0 else hL[stage - 1]
                vapor_in_flow = 0.0 if stage == N - 1 else V[stage + 1]
                vapor_in_h = 0.0 if stage == N - 1 else hV[stage + 1]
                in_energy = (
                    liquid_in_flow * liquid_in_h
                    + vapor_in_flow * vapor_in_h
                    + sum(feed['F'] * feed['H'] for feed in stage_feeds[stage])
                )
                out_energy = L[stage] * hL[stage] + V[stage] * hV[stage]
                residuals.append(float(in_energy - out_energy))
            return residuals

        def residual(vector):
            decoded = decode(vector)
            T = decoded['T']
            x = decoded['x']
            L = decoded['L']
            V = decoded['V']
            props = [stage_properties(stage, T[stage], x[stage]) for stage in range(N)]
            y = [item['y'] for item in props]
            hL = [item['hL'] for item in props]
            hV = [item['hV'] for item in props]
            values = []
            labels = []

            for stage in range(N):
                liquid_in_flow = 0.0 if stage == 0 else L[stage - 1]
                liquid_in_comp = None if stage == 0 else x[stage - 1]
                liquid_in_h = 0.0 if stage == 0 else hL[stage - 1]
                vapor_in_flow = 0.0 if stage == N - 1 else V[stage + 1]
                vapor_in_comp = None if stage == N - 1 else y[stage + 1]
                vapor_in_h = 0.0 if stage == N - 1 else hV[stage + 1]

                for comp in comps:
                    incoming = sum(
                        feed['F'] * feed['z'].get(comp, 0.0)
                        for feed in stage_feeds[stage]
                    )
                    if liquid_in_comp is not None:
                        incoming += liquid_in_flow * liquid_in_comp.get(comp, 0.0)
                    if vapor_in_comp is not None:
                        incoming += vapor_in_flow * vapor_in_comp.get(comp, 0.0)
                    outgoing = (
                        L[stage] * x[stage].get(comp, 0.0)
                        + V[stage] * y[stage].get(comp, 0.0)
                    )
                    scale = component_scales[comp]
                    values.append((incoming - outgoing) / scale)
                    labels.append(('component', stage + 1, comp, scale))

                if mode == 'isothermal':
                    temperature_scale = max(abs(temperature_targets[stage]), 1.0)
                    values.append(
                        (T[stage] - temperature_targets[stage])
                        / temperature_scale
                    )
                    labels.append(
                        ('temperature', stage + 1, None, temperature_scale)
                    )
                else:
                    in_energy = (
                        liquid_in_flow * liquid_in_h
                        + vapor_in_flow * vapor_in_h
                        + sum(feed['F'] * feed['H'] for feed in stage_feeds[stage])
                    )
                    out_energy = L[stage] * hL[stage] + V[stage] * hV[stage]
                    values.append((in_energy - out_energy) / energy_scale)
                    labels.append(('energy', stage + 1, None, energy_scale))

                values.append(props[stage]['bubble'])
                labels.append(('bubble', stage + 1, None, 1.0))

            if not residual_labels:
                residual_labels.extend(labels)
            return np.array(values, dtype=float)

        def sparsity():
            n_rows = N * (nc + 2)
            matrix = lil_matrix((n_rows, n_vars), dtype=int)

            def mark_stage(row: int, stage: int):
                if not 0 <= stage < N:
                    return
                matrix[row, stage] = 1
                start = logits_start + stage * (nc - 1)
                for col in range(start, start + nc - 1):
                    matrix[row, col] = 1
                matrix[row, L_start + stage] = 1
                matrix[row, V_start + stage] = 1

            row = 0
            for stage in range(N):
                local_stages = {stage}
                if stage > 0:
                    local_stages.add(stage - 1)
                if stage < N - 1:
                    local_stages.add(stage + 1)
                for _ in comps:
                    for local in local_stages:
                        mark_stage(row, local)
                    row += 1
                for local in local_stages:
                    mark_stage(row, local)
                row += 1
                mark_stage(row, stage)
                row += 1
            return matrix.tocsr()

        sparsity_matrix = sparsity()

        def semi_analytic_flow_jacobian(vector, f0, rel_step: float):
            if not self._truthy_param(
                self.get_param('semi_analytic_flow_jacobian', True)
            ):
                return None

            decoded = decode(vector)
            T = decoded['T']
            x = decoded['x']
            L = decoded['L']
            V = decoded['V']
            props = [
                stage_properties(stage, T[stage], x[stage])
                for stage in range(N)
            ]
            y = [item['y'] for item in props]
            hL = [item['hL'] for item in props]
            hV = [item['hV'] for item in props]
            J = np.zeros(sparsity_matrix.shape, dtype=float)
            evaluations = 0

            def add(row: int, col: int, value: float) -> None:
                if value:
                    J[row, col] += value

            use_local_thermo = self._truthy_param(
                self.get_param('semi_analytic_local_thermo_jacobian', True)
            )
            if use_local_thermo:
                for stage in range(N):
                    local_columns = [stage]
                    local_columns.extend(range(
                        logits_start + stage * (nc - 1),
                        logits_start + (stage + 1) * (nc - 1),
                    ))
                    for col in local_columns:
                        step = rel_step * max(abs(vector[col]), 1.0)
                        T_perturbed = float(T[stage])
                        x_perturbed = x[stage]
                        if col == stage:
                            theta = min(max(float(vector[col] + step), -60.0), 60.0)
                            T_perturbed = T_min + span / (1.0 + math.exp(-theta))
                        else:
                            start = logits_start + stage * (nc - 1)
                            logits = np.array(
                                list(vector[start:start + nc - 1]) + [0.0],
                                dtype=float,
                            )
                            logits[col - start] += step
                            logits = np.clip(logits, -60.0, 60.0)
                            logits -= np.max(logits)
                            fractions = np.exp(logits)
                            fractions /= np.sum(fractions)
                            x_perturbed = {
                                comp: float(fractions[index])
                                for index, comp in enumerate(comps)
                            }

                        perturbed = stage_properties(
                            stage,
                            T_perturbed,
                            x_perturbed,
                        )
                        dx = {
                            comp: (
                                x_perturbed.get(comp, 0.0)
                                - x[stage].get(comp, 0.0)
                            ) / step
                            for comp in comps
                        }
                        dy = {
                            comp: (
                                perturbed['y'].get(comp, 0.0)
                                - y[stage].get(comp, 0.0)
                            ) / step
                            for comp in comps
                        }
                        dhL = (perturbed['hL'] - hL[stage]) / step
                        dhV = (perturbed['hV'] - hV[stage]) / step
                        dbubble = (
                            perturbed['bubble'] - props[stage]['bubble']
                        ) / step

                        for ci, comp in enumerate(comps):
                            scale = component_scales[comp]
                            add(
                                stage * (nc + 2) + ci,
                                col,
                                -(
                                    L[stage] * dx[comp]
                                    + V[stage] * dy[comp]
                                ) / scale,
                            )
                            if stage < N - 1:
                                add(
                                    (stage + 1) * (nc + 2) + ci,
                                    col,
                                    L[stage] * dx[comp] / scale,
                                )
                            if stage > 0:
                                add(
                                    (stage - 1) * (nc + 2) + ci,
                                    col,
                                    V[stage] * dy[comp] / scale,
                                )

                        local_row = stage * (nc + 2) + nc
                        if mode == 'adiabatic':
                            add(
                                local_row,
                                col,
                                -(
                                    L[stage] * dhL
                                    + V[stage] * dhV
                                ) / energy_scale,
                            )
                            if stage < N - 1:
                                add(
                                    (stage + 1) * (nc + 2) + nc,
                                    col,
                                    L[stage] * dhL / energy_scale,
                                )
                            if stage > 0:
                                add(
                                    (stage - 1) * (nc + 2) + nc,
                                    col,
                                    V[stage] * dhV / energy_scale,
                                )
                        elif col == stage:
                            temperature_scale = max(
                                abs(temperature_targets[stage]),
                                1.0,
                            )
                            add(
                                local_row,
                                col,
                                (T_perturbed - T[stage])
                                / step
                                / temperature_scale,
                            )
                        add(local_row + 1, col, dbubble)
            else:
                column_rows = [
                    sparsity_matrix[:, col].nonzero()[0]
                    for col in range(sparsity_matrix.shape[1])
                ]
                nonlinear_columns = set(range(N))
                nonlinear_columns.update(
                    range(logits_start, logits_start + N * (nc - 1))
                )
                restricted_groups = []
                group_rows = []
                for col in sorted(nonlinear_columns):
                    rows = set(column_rows[col].tolist())
                    for index, used_rows in enumerate(group_rows):
                        if rows.isdisjoint(used_rows):
                            restricted_groups[index].append(col)
                            used_rows.update(rows)
                            break
                    else:
                        restricted_groups.append([col])
                        group_rows.append(set(rows))
                for group in restricted_groups:
                    perturbation = np.zeros_like(vector)
                    for col in group:
                        perturbation[col] = rel_step * max(abs(vector[col]), 1.0)
                    f_step = residual(vector + perturbation)
                    evaluations += 1
                    diff = f_step - f0
                    for col in group:
                        rows = column_rows[col]
                        if rows.size:
                            J[rows, col] = diff[rows] / perturbation[col]

            for stage in range(N):
                for ci, comp in enumerate(comps):
                    row = stage * (nc + 2) + ci
                    scale = component_scales[comp]
                    if stage > 0:
                        add(
                            row,
                            L_start + stage - 1,
                            L[stage - 1] * x[stage - 1].get(comp, 0.0) / scale,
                        )
                    if stage < N - 1:
                        add(
                            row,
                            V_start + stage + 1,
                            V[stage + 1] * y[stage + 1].get(comp, 0.0) / scale,
                        )
                    add(
                        row,
                        L_start + stage,
                        -L[stage] * x[stage].get(comp, 0.0) / scale,
                    )
                    add(
                        row,
                        V_start + stage,
                        -V[stage] * y[stage].get(comp, 0.0) / scale,
                    )

                if mode == 'adiabatic':
                    energy_row = stage * (nc + 2) + nc
                    if stage > 0:
                        add(
                            energy_row,
                            L_start + stage - 1,
                            L[stage - 1] * hL[stage - 1] / energy_scale,
                        )
                    if stage < N - 1:
                        add(
                            energy_row,
                            V_start + stage + 1,
                            V[stage + 1] * hV[stage + 1] / energy_scale,
                        )
                    add(
                        energy_row,
                        L_start + stage,
                        -L[stage] * hL[stage] / energy_scale,
                    )
                    add(
                        energy_row,
                        V_start + stage,
                        -V[stage] * hV[stage] / energy_scale,
                    )

            return (
                csr_matrix(J),
                evaluations,
                (
                    'semi_analytic_local_thermo'
                    if use_local_thermo else 'semi_analytic_flow'
                ),
            )

        return {
            'decode': decode,
            'residual': residual,
            'sparsity': sparsity_matrix,
            'jacobian': semi_analytic_flow_jacobian,
            'stage_properties': stage_properties,
            'stage_energy_residuals': stage_energy_residuals,
            'residual_labels': residual_labels,
        }

    def _absorber_component_balance_error(
        self,
        comps: list[str],
        feed_specs: list[dict],
        outlets: list[StreamState],
        flow_scale: float,
    ) -> float:
        max_error = 0.0
        for comp in comps:
            incoming = sum(
                feed['F'] * feed['z'].get(comp, 0.0)
                for feed in feed_specs
            )
            outgoing = sum(
                stream.F * stream.composition.get(comp, 0.0)
                for stream in outlets
            )
            max_error = max(max_error, abs(incoming - outgoing) / flow_scale)
        return max_error


class RigorousStripper(RigorousAbsorber):
    """Equation-oriented equilibrium-stage gas stripper.

    A liquid feed enters at or near the top and an external stripping gas
    enters at or near the bottom.  Vapor product leaves the top and stripped
    liquid leaves the bottom.
    """

    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        result = super().solve(inlets)
        performance = result.performance
        performance['method'] = 'rigorous_stripper_mesh'
        performance.pop('component_absorption_fraction', None)

        liquid_out = result.outlet_streams['liquid_out']
        stripping = {}
        for comp in liquid_out.composition:
            liquid_in_comp = sum(
                feed['flow'] * feed['composition'].get(comp, 0.0)
                for feed in performance['feeds']
                if feed['phase_hint'] == 'liquid'
            )
            liquid_out_comp = (
                liquid_out.F * liquid_out.composition.get(comp, 0.0)
            )
            if liquid_in_comp > 1e-30:
                stripping[comp] = (
                    liquid_in_comp - liquid_out_comp
                ) / liquid_in_comp
        performance['component_stripping_fraction'] = stripping
        return result

    def _absorber_feed_phase(self, port: str, stream: StreamState) -> str:
        port_key = str(port).strip().lower().replace('-', '_')
        if any(token in port_key for token in ('strip', 'steam', 'gas', 'vapor', 'vapour', 'air')):
            return 'vapor'
        if any(token in port_key for token in ('liquid', 'feed', 'rich', 'solution')):
            return 'liquid'
        return super()._absorber_feed_phase(port, stream)

    def _absorber_initializer_order(self) -> tuple[str, ...]:
        return ('equilibrium_sweep', 'loading_profile')

    def _gas_feed_default_stage(self, N: int) -> int:
        return int(self.get_param(
            'strip_gas_stage',
            self.get_param(
                'stripping_gas_stage',
                self.get_param('gas_stage', self.get_param('vapor_stage', N)),
            ),
        ))

    def _liquid_feed_default_stage(self, N: int) -> int:
        return int(self.get_param(
            'feed_stage',
            self.get_param(
                'liquid_stage',
                self.get_param('rich_liquid_stage', 1),
            ),
        ))


class Absorber(UnitOperation):
    """Gas absorption column. Inlets: gas, liquid. Outlets: gas_out, liquid_out."""
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        gas_in = liquid_in = None
        for name, stream in inlets.items():
            if stream.vapor_fraction > 0.5:
                gas_in = stream
            else:
                liquid_in = stream
        
        if not gas_in: gas_in = inlets.get('gas') or inlets.get('vapor')
        if not liquid_in: liquid_in = inlets.get('liquid') or inlets.get('solvent')
        
        if not gas_in or not liquid_in:
            raise UnitOperationError(f"Absorber '{self.unit_id}' needs gas and liquid inlets")
        
        N = int(self.get_param('N_stages', 5))
        T = float(self.get_param('T', (gas_in.T + liquid_in.T)/2))
        P = float(self.get_param('P', gas_in.P))
        if T < 200: T += 273.15
        
        comps = list(set(list(gas_in.composition.keys()) + list(liquid_in.composition.keys())))
        L, V = liquid_in.F, gas_in.F
        
        K = {c: self.thermo.K_value(c, T, P) if hasattr(self.thermo, 'K_value') else 1.0 for c in comps}
        
        y_out, x_out = {}, {}
        for c in comps:
            A = L / max(V * K.get(c,1), 1e-10)
            frac_abs = (A**(N+1)-A)/max(A**(N+1)-1, 1e-10) if abs(A-1)>0.01 else N/(N+1)
            frac_abs = max(0, min(1, frac_abs))
            
            mol_gas = V * gas_in.composition.get(c, 0)
            mol_liq = L * liquid_in.composition.get(c, 0)
            y_out[c] = mol_gas * (1-frac_abs)
            x_out[c] = mol_gas * frac_abs + mol_liq
        
        V_out, L_out = sum(y_out.values()), sum(x_out.values())
        if V_out > 0: y_out = {c: v/V_out for c,v in y_out.items()}
        if L_out > 0: x_out = {c: v/L_out for c,v in x_out.items()}
        
        gas_out = self.thermo.calculate_state(T, P, V_out, y_out, phase='vapor')
        liq_out = self.thermo.calculate_state(T, P, L_out, x_out, phase='liquid')
        
        return UnitResult(
            outlet_streams={'gas_out': gas_out, 'liquid_out': liq_out},
            performance={'N_stages': N, 'T_C': T-273.15, 'L_over_V': L/max(V,1e-10)}
        )


class Stripper(UnitOperation):
    """Gas stripping column. Inlets: liquid, gas. Outlets: liquid_out, gas_out."""
    
    def solve(self, inlets: dict[str, StreamState]) -> UnitResult:
        liquid_in = gas_in = None
        for name, stream in inlets.items():
            if stream.vapor_fraction < 0.5:
                liquid_in = stream
            else:
                gas_in = stream
        
        if not liquid_in: liquid_in = inlets.get('liquid') or inlets.get('feed')
        if not gas_in: gas_in = inlets.get('gas') or inlets.get('strip_gas')
        
        if not liquid_in:
            raise UnitOperationError(f"Stripper '{self.unit_id}' needs liquid inlet")
        
        N = int(self.get_param('N_stages', 5))
        T = float(self.get_param('T', liquid_in.T))
        P = float(self.get_param('P', liquid_in.P))
        if T < 200: T += 273.15
        
        comps = list(liquid_in.composition.keys())
        if gas_in:
            comps = list(set(comps + list(gas_in.composition.keys())))
        
        L = liquid_in.F
        V = gas_in.F if gas_in else 0.1 * L  # Small strip gas if not specified
        
        K = {c: self.thermo.K_value(c, T, P) if hasattr(self.thermo, 'K_value') else 1.0 for c in comps}
        
        x_out, y_out = {}, {}
        for c in comps:
            S = K.get(c,1) * V / max(L, 1e-10)
            frac_strip = (S**(N+1)-S)/max(S**(N+1)-1, 1e-10) if abs(S-1)>0.01 else N/(N+1)
            frac_strip = max(0, min(1, frac_strip))
            
            mol_liq = L * liquid_in.composition.get(c, 0)
            mol_gas = V * (gas_in.composition.get(c, 0) if gas_in else 0)
            x_out[c] = mol_liq * (1-frac_strip)
            y_out[c] = mol_liq * frac_strip + mol_gas
        
        L_out, V_out = sum(x_out.values()), sum(y_out.values())
        if L_out > 0: x_out = {c: v/L_out for c,v in x_out.items()}
        if V_out > 0: y_out = {c: v/V_out for c,v in y_out.items()}
        
        liq_out = self.thermo.calculate_state(T, P, L_out, x_out, phase='liquid')
        gas_out = self.thermo.calculate_state(T, P, V_out, y_out, phase='vapor')
        
        return UnitResult(
            outlet_streams={'liquid_out': liq_out, 'gas_out': gas_out},
            performance={'N_stages': N, 'T_C': T-273.15, 'V_over_L': V/max(L,1e-10)}
        )
