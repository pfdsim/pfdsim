"""Utility consumption and surface sizing through the shared HeatExchanger."""

import math
import json
from collections.abc import Mapping

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .chemical_properties import ChemicalDatabase
    from .thermodynamics import create_thermodynamics
    from .unit_operations_base import UnitOperationError
    from .unit_conversions import pressure_to_bar
    from .heat_exchanger_transport import transport_parameter, heat_transfer_area
else:
    from chemical_properties import ChemicalDatabase
    from thermodynamics import create_thermodynamics
    from unit_operations_base import UnitOperationError
    from unit_conversions import pressure_to_bar
    from heat_exchanger_transport import transport_parameter, heat_transfer_area


def _utility_composition(value, fluid):
    if value is None:
        if fluid is None:
            raise UnitOperationError('Specify utility_fluid or a JSON utility_composition for this utility')
        return {str(fluid): 1.0}
    if fluid is not None:
        raise UnitOperationError('Specify utility_fluid or utility_composition, not both')
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise UnitOperationError('utility_composition must be a JSON object of mole fractions') from exc
    if not isinstance(value, Mapping) or not value:
        raise UnitOperationError('utility_composition must be a nonempty mole-fraction mapping')
    try:
        composition = {str(key): float(fraction) for key, fraction in value.items()}
    except (TypeError, ValueError) as exc:
        raise UnitOperationError('Utility mole fractions must be numeric') from exc
    if not all(math.isfinite(fraction) and fraction >= 0 for fraction in composition.values()) or not math.isclose(sum(composition.values()), 1.0, abs_tol=1e-8):
        raise UnitOperationError('Utility mole fractions must be finite, nonnegative, and sum to one')
    return {key: fraction for key, fraction in composition.items() if fraction > 0}


def _nonfluid_utility_size(unit, duty, utility):
    if unit.heat_direction <= 0:
        raise UnitOperationError('Electric and fired utility options are heating services')
    if (str(unit.get_param('U_model', 'specified')).lower() != 'specified'
            or unit.get_param('U') is not None or unit.get_param('UA') is not None
            or unit.get_param('UA_available') is not None
            or unit.get_param('LMTD_correction') is not None
            or unit._truthy_param(unit.get_param('estimate_U', False))):
        raise UnitOperationError('Electric/fired surface sizing uses utility_heat_flux; fluid-film U models are not applicable')
    flux = transport_parameter(unit.get_param('utility_heat_flux'), 'utility_heat_flux',
        unit.get_param_unit('utility_heat_flux'), {'': 1, 'w/m2': 1, 'w/m^2': 1, 'kw/m2': 1000, 'kw/m^2': 1000})
    efficiency = transport_parameter(unit.get_param('utility_efficiency', 1.0 if utility == 'electric' else .85),
        'utility_efficiency', unit.get_param_unit('utility_efficiency'), {'': 1, '%': .01, 'percent': .01})
    if efficiency > 1:
        raise UnitOperationError('utility_efficiency must be between zero and one')
    input_kW = duty/3600/efficiency
    metrics = {'utility': utility, 'area_required_m2': duty/3.6/flux,
               'utility_heat_flux_W_m2': flux, 'utility_input_kW': input_kW,
               'utility_loss_kW': input_kW-duty/3600, 'utility_efficiency': efficiency,
               'area_sizing_method': 'specified_surface_heat_flux',
               'utility_efficiency_source': 'specified' if unit.get_param('utility_efficiency') is not None else 'default',
               'utility_area_preliminary': False}
    if utility == 'electric':
        metrics['electric_power_kW'] = input_kW
    else:
        fuel = str(unit.get_param('utility_fuel', unit.get_param('fuel', 'methane')))
        raw = unit.get_param('fuel_heating_value')
        if raw is not None:
            heating_value = transport_parameter(raw, 'fuel_heating_value', unit.get_param_unit('fuel_heating_value'),
                {'': 1, 'kj/kg': 1, 'mj/kg': 1000, 'j/kg': .001})
            source = 'specified net heating value'
        else:
            properties = ChemicalDatabase(enable_online=False).get(fuel)
            if properties is None or properties.Hcomb is None or properties.MW <= 0:
                raise UnitOperationError('Fuel net heat of combustion unavailable; specify fuel_heating_value [kJ/kg]')
            heating_value = abs(properties.Hcomb)*1000/properties.MW
            if not math.isfinite(heating_value) or heating_value <= 0:
                raise UnitOperationError('Fuel heating value must be positive')
            source = 'shared chemical-property net heat of combustion'
        metrics.update(utility_fuel=fuel, fuel_heating_value_kJ_kg=heating_value,
                       fuel_heating_value_source=source, fuel_mass_flow_kg_h=input_kW*3600/heating_value)
    area = unit.get_param('A', unit.get_param('area'))
    if area is not None:
        name = 'A' if unit.get_param('A') is not None else 'area'
        area = heat_transfer_area(area, unit.get_param_unit(name), name=name)
        metrics.update(area_m2=area, area_margin_m2=area-metrics['area_required_m2'],
                       area_utilization=metrics['area_required_m2']/area)
    return metrics, []


class HenryCurveThermodynamics:
    """Preserve the Heater's frozen Henry state construction on utility curves."""

    def __init__(self, base, helper, context):
        self.base, self.helper, self.context = base, helper, context

    def __getattr__(self, name):
        return getattr(self.base, name)

    def calculate_state(self, T, P, F, composition, phase=None, flash=True, include=None):
        if phase is not None:
            return self.base.calculate_state(T, P, F, composition, phase=phase, flash=flash, include=include)
        return self.helper._state_at_TP(composition, T, P, F, self.context)

    def calculate_state_PH(self, P, H, F, composition, include=None, phase=None, T_guess=298.15):
        return self.helper._state_at_PH(composition, P, H, F, T_guess, self.context)

    def calculate_state_PQ(self, P, quality, F, composition, include=None):
        return self.helper._state_at_PV(composition, P, quality, F, 298.15, self.context)


def size_thermal_utility(unit, inlet, outlet, duty, process_drop, *, process_thermo=None):
    """Size fluid or surface-loading service for a specified process duty.

    Steam defaults to saturated supply and saturated-liquid return. Water
    supply/return temperatures are required; no site utility grade is invented.
    Without geometry or explicit U, service-class auto-U gives preliminary area.
    """
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .unit_operations_basic import HeatExchanger
    else:
        from unit_operations_basic import HeatExchanger

    utility = str(unit.get_param('utility')).strip().lower().replace('-', '_').replace(' ', '_')
    if utility in ('electric', 'fired'):
        return _nonfluid_utility_size(unit, duty, utility)
    if utility not in ('steam', 'cooling_water', 'chilled_water', 'hot_water', 'hot_oil', 'thermal_fluid', 'air', 'refrigerant'):
        raise UnitOperationError('Unknown utility; choose steam, cooling_water, chilled_water, hot_water, hot_oil, thermal_fluid, air, refrigerant, electric or fired')
    heating = unit.heat_direction > 0
    if utility == 'steam' and not heating:
        raise UnitOperationError('Steam heating utility requires Heater; use a liquid-water utility with Cooler')
    fluid = unit.get_param('utility_fluid')
    composition_raw = unit.get_param('utility_composition')
    if utility in ('steam', 'cooling_water', 'chilled_water', 'hot_water'):
        if fluid is not None or composition_raw is not None:
            raise UnitOperationError('Water/steam utilities have fixed water identity; use thermal_fluid/hot_oil/refrigerant for a specified fluid')
        composition, default_method = {'water': 1.0}, 'STEAM'
    elif utility == 'air' and fluid is None and composition_raw is None:
        composition, default_method = {'N2': .79, 'O2': .21}, 'IDEAL'
    else:
        composition = _utility_composition(composition_raw, fluid)
        default_method = 'PR' if utility == 'refrigerant' else 'IDEAL'
    method = str(unit.get_param('utility_thermo_method', default_method)).strip().upper()
    if utility == 'refrigerant' and len(composition) != 1:
        raise UnitOperationError('Phase-change refrigerant utilities currently require a pure fluid')
    if method == 'PROCESS':
        if not all(comp in unit.thermo.props for comp in composition):
            raise UnitOperationError('utility_thermo_method=process requires utility components in the process package')
        backend = unit.thermo
    else:
        cache = getattr(unit, '_utility_backends', {})
        cache_key = method, tuple(composition)
        if cache_key not in cache:
            cache[cache_key] = create_thermodynamics(list(composition), method, db=ChemicalDatabase(enable_online=False))
            unit._utility_backends = cache
        backend = cache[cache_key]
    pressure_raw = unit.get_param('utility_P')
    pressure_unit = unit.get_param_unit('utility_P')
    saturation_temperature = unit.get_temperature_param('utility_T')
    if pressure_raw is None:
        if utility in ('steam', 'refrigerant') and saturation_temperature is None:
            raise UnitOperationError('Phase-change utility requires utility_P or saturated utility_T')
        pressure = (backend.bubble_point_P(composition, saturation_temperature)
                    if utility in ('steam', 'refrigerant') else 1.0)
    else:
        pressure = pressure_to_bar(pressure_raw, pressure_unit, strict=True)
    drop = pressure_to_bar(unit.get_param('utility_P_drop', 0.0), unit.get_param_unit('utility_P_drop'), strict=True)
    if not math.isfinite(pressure) or pressure <= 0 or not math.isfinite(drop) or not 0 <= drop < pressure:
        raise UnitOperationError('Utility pressure must be positive and its drop finite, nonnegative and below supply pressure')
    return_pressure = pressure-drop
    supply_temperature = unit.get_temperature_param('utility_T_in')
    return_temperature = unit.get_temperature_param('utility_T_out')
    return_quality_raw = unit.get_param('utility_outlet_vapor_fraction')
    supply_quality_raw = unit.get_param('utility_inlet_vapor_fraction')
    if return_temperature is not None and return_quality_raw is not None:
        raise UnitOperationError('Specify utility_T_out or utility_outlet_vapor_fraction, not both')
    if utility in ('steam', 'refrigerant'):
        state_helper = HeatExchanger(f'{unit.unit_id}:utility', backend, {})
        if supply_temperature is None:
            supply_quality = float(supply_quality_raw if supply_quality_raw is not None else (1.0 if heating else 0.0))
            if not math.isfinite(supply_quality) or not 0 <= supply_quality <= 1:
                raise UnitOperationError('utility_inlet_vapor_fraction must be between zero and one')
            supply = state_helper._state_for_vapor_fraction(composition, pressure, 1.0, supply_quality)
        else:
            if supply_quality_raw is not None:
                raise UnitOperationError('Specify utility_T_in or utility_inlet_vapor_fraction, not both')
            supply = backend.calculate_state(supply_temperature, pressure, 1.0, composition)
            if utility == 'steam' and supply.vapor_fraction < 1-1e-8:
                raise UnitOperationError('Steam utility supply must be vapor; utility_T_in is below saturation')
        if saturation_temperature is not None and abs(backend.bubble_point_T(composition, pressure)-saturation_temperature) > 1e-4:
            raise UnitOperationError('utility_T conflicts with steam supply pressure or temperature')
        if return_temperature is None:
            quality = float(return_quality_raw if return_quality_raw is not None else (0.0 if heating else 1.0))
            if not math.isfinite(quality) or not 0 <= quality <= 1:
                raise UnitOperationError('utility_outlet_vapor_fraction must be between zero and one')
            returning = state_helper._state_for_vapor_fraction(composition, return_pressure, 1.0, quality)
        else:
            returning = backend.calculate_state(return_temperature, return_pressure, 1.0, composition)
    else:
        if supply_temperature is None or return_temperature is None:
            raise UnitOperationError('Sensible utility requires utility_T_in and utility_T_out')
        if return_quality_raw is not None or supply_quality_raw is not None or saturation_temperature is not None:
            raise UnitOperationError('Sensible utilities use supply/return temperatures, not phase-change quality or utility_T')
        supply = backend.calculate_state(supply_temperature, pressure, 1.0, composition)
        returning = backend.calculate_state(return_temperature, return_pressure, 1.0, composition)
        expected_phase = 1.0 if utility == 'air' else 0.0
        if abs(supply.vapor_fraction-expected_phase) > 1e-8 or abs(returning.vapor_fraction-expected_phase) > 1e-8:
            raise UnitOperationError('Sensible utility supply and return must remain gas for air, liquid for water/oil')
    enthalpy_change = returning.H-supply.H
    utility_flow = -duty/enthalpy_change if enthalpy_change else float('inf')
    if not math.isfinite(utility_flow) or utility_flow <= 0:
        raise UnitOperationError('Utility supply/return enthalpies cannot provide the requested heating or cooling duty')
    supply.F = utility_flow
    returning.F = utility_flow
    scope = f'utility:{unit.unit_id}'
    supply.thermo_scope = returning.thermo_scope = scope
    params = {key: value for key, value in unit.params.items()
              if str(key).removeprefix('__unit__').lower() not in (
                  't', 't_out', 'q', 'duty', 'heat_duty', 'vap_frac', 'vapor_frac', 'vapor_fraction', 'vf',
                  't_hot_out', 't_cold_out', 't_tube_out', 't_shell_out')}
    params.update(Q=abs(duty), __unit__Q='kJ/h')
    preliminary = False
    if (str(unit.get_param('U_model', 'specified')).strip().lower() == 'specified'
            and unit.get_param('U') is None and not unit._truthy_param(unit.get_param('estimate_U', False))):
        params['U'] = 'auto'
        preliminary = True
    process_side = str(unit.get_param('process_side', 'tube')).strip().lower()
    if process_side not in ('tube', 'shell'):
        raise UnitOperationError('process_side must be tube or shell')
    utility_side = 'shell' if process_side == 'tube' else 'tube'
    params[f'P_drop_{process_side}'] = process_drop
    params[f'P_drop_{utility_side}'] = drop
    exchanger = HeatExchanger(unit.unit_id, process_thermo or unit.thermo, params,
                             side_thermodynamics={scope: backend})
    exchanger.solve_context = {**(getattr(unit, 'solve_context', {}) or {}), 'expensive_diagnostics': True}
    exchanger_result = exchanger.solve({f'{process_side}_in': inlet, f'{utility_side}_in': supply})
    performance = exchanger_result.performance
    if performance['temperature_cross']:
        raise UnitOperationError('Utility supply/return temperatures create a temperature cross for the requested process target')
    expected_process = exchanger_result.outlet_streams[f'{process_side}_out']
    if abs(expected_process.H-outlet.H) > max(1e-5, abs(outlet.H)*1e-8):
        raise UnitOperationError('Utility exchanger curve does not reproduce the process outlet enthalpy')
    expected_return = exchanger_result.outlet_streams[f'{utility_side}_out']
    if abs(expected_return.H-returning.H) > max(1e-5, abs(returning.H)*1e-8):
        raise UnitOperationError('Utility exchanger curve does not reproduce the utility return enthalpy')
    area = performance.get('area_required_m2', performance.get('auto_U_area_required_m2'))
    metrics = {'utility': utility, 'utility_flow_kmol_h': utility_flow,
               'utility_composition': composition, 'utility_thermo_method': method,
               'utility_mass_flow_kg_h': utility_flow*backend.mixture_MW(composition),
               'utility_T_in_C': supply.T-273.15, 'utility_T_out_C': returning.T-273.15,
               'utility_P_in_bar': pressure, 'utility_P_out_bar': return_pressure,
               'utility_vapor_fraction_in': supply.vapor_fraction,
               'utility_vapor_fraction_out': returning.vapor_fraction,
               'UA_required_W_per_K': performance['UA_required_W_per_K'],
               'area_required_m2': area, 'utility_sizing': performance,
               'utility_area_preliminary': preliminary or performance.get('auto_U_estimation', False)}
    return metrics, exchanger_result.warnings
