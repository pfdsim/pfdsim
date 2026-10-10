"""Material, laminar, phase-change, shell geometry and utility integration."""

import math

import pytest

from chemical_properties import ChemicalDatabase
from heat_exchanger_transport import wall_material_record
from thermodynamics import create_thermodynamics
from transport_correlations import internal_flow_transfer, laminar_annulus_nusselt, shah_boiling_coefficient, TransportCorrelationError
from unit_operations_basic import Heater, Cooler, HeatExchanger
from unit_operations_base import UnitOperationError
from unit_settings import unit_setting_schema
from tests.test_heat_exchanger_transport import ConstantLiquid, parameters


def water_backend():
    return create_thermodynamics(['water'], 'STEAM', db=ChemicalDatabase(enable_online=False))


def shell_parameters(**updates):
    return {'U_model': 'shell_tube', 'tube_inner_diameter': .02,
            'tube_outer_diameter': .024, 'shell_inner_diameter': .24,
            'bundle_diameter': .18, 'tube_count': 12, 'tube_passes': 2,
            'tube_pitch': .03, 'baffle_spacing': .15, 'baffle_cut': .25,
            'baffle_count': 8, 'tube_baffle_clearance': .0004,
            'shell_baffle_clearance': .001, 'wall_material': 'stainless_steel_304',
            'curve_segments': 4, **updates}


def test_material_reference_values_aliases_and_overrides():
    assert wall_material_record('Copper')['conductivity_W_m_K'] == 398
    assert wall_material_record('carbon steel')['conductivity_W_m_K'] == 50
    assert wall_material_record('brass')['conductivity_W_m_K'] == 120
    assert wall_material_record('ss316')['conductivity_W_m_K'] == pytest.approx(9.4*1.730735)
    thermo = ConstantLiquid()
    inlets = {'tube_in': thermo.calculate_state(360, 3, 200, {'water': 1}),
              'shell_in': thermo.calculate_state(300, 3, 300, {'water': 1})}
    kwargs = parameters(Q=25, wall_material='copper')
    kwargs.pop('wall_conductivity')
    material = HeatExchanger('M', thermo, kwargs).solve(inlets)
    explicit = HeatExchanger('E', thermo, {**kwargs, 'wall_conductivity': 16}).solve(inlets)
    assert material.performance['wall_conductivity_W_m_K'] == 398
    assert explicit.performance['wall_conductivity_W_m_K'] == 16
    assert explicit.performance['wall_material']['overridden_by_wall_conductivity']
    assert material.performance['area_required_m2'] < explicit.performance['area_required_m2']
    with pytest.raises(UnitOperationError, match='Unknown wall material'):
        wall_material_record('unknown alloy')


def test_laminar_annulus_matches_published_uniform_flux_reference():
    # NASA TN D-1972: fully developed inner-wall flux, outer wall insulated.
    assert laminar_annulus_nusselt(.5) == pytest.approx(6.181, rel=1e-5)
    assert laminar_annulus_nusselt(.1) == pytest.approx(11.906, rel=3e-5)
    assert internal_flow_transfer(1000, 5, 5)['nusselt'] == pytest.approx(48/11)
    assert internal_flow_transfer(1000, 5, 5, annulus_diameter_ratio=.5)['nusselt'] == pytest.approx(6.181, rel=1e-5)
    with pytest.raises(TransportCorrelationError):
        internal_flow_transfer(5000, 5, 5, annulus_diameter_ratio=.5)


@pytest.mark.parametrize('passes', [1, 2])
def test_shell_tube_coefficients_corrections_and_pass_arrangement(passes):
    thermo = ConstantLiquid()
    # Keep all tube passages turbulent even with one tube pass.
    inlets = {'tube_in': thermo.calculate_state(360, 3, 400, {'water': 1}),
              'shell_in': thermo.calculate_state(300, 3, 600, {'water': 1})}
    result = HeatExchanger('S', thermo, shell_parameters(Q=50, tube_passes=passes)).solve(inlets)
    p = result.performance
    node = p['calculated_U_nodes'][0]
    assert node['shell']['correlation'] == 'zukauskas_bell_corrections'
    assert set(node['shell']['shell_corrections']) == {'Jc', 'Jl', 'Jb', 'Jr', 'Js'}
    assert 0 < node['shell']['shell_corrections']['Jl'] < 1
    assert 0 < node['shell']['shell_corrections']['Jb'] < 1
    assert p['LMTD_correction'] == 1 if passes == 1 else 0 < p['LMTD_correction'] < 1
    tube_out = result.outlet_streams['tube_out']
    shell_out = result.outlet_streams['shell_out']
    assert inlets['tube_in'].F*(inlets['tube_in'].H-tube_out.H) == pytest.approx(50*3600)
    assert inlets['shell_in'].F*(shell_out.H-inlets['shell_in'].H) == pytest.approx(50*3600)


def test_shell_tube_rating_recovers_prescribed_geometry_area():
    thermo = ConstantLiquid()
    inlets = {'tube_in': thermo.calculate_state(360, 3, 400, {'water': 1}),
              'shell_in': thermo.calculate_state(300, 3, 600, {'water': 1})}
    result = HeatExchanger('S', thermo, shell_parameters(length=2)).solve(inlets)
    assert result.performance['area_m2'] == pytest.approx(math.pi*.024*12*2)
    assert result.performance['area_utilization'] == pytest.approx(1, rel=1e-6)


@pytest.mark.parametrize('updates,message', [({'tube_count': 1000}, 'packing capacity'),
                                            ({'tube_pitch': .024}, 'tube_pitch'),
                                            ({'baffle_cut': .8}, 'baffle_cut'),
                                            ({'tube_count': 13}, 'integer tube counts')])
def test_shell_geometry_rejects_impossible_hardware(updates, message):
    thermo = ConstantLiquid()
    inlets = {'tube_in': thermo.calculate_state(360, 3, 400, {'water': 1}),
              'shell_in': thermo.calculate_state(300, 3, 600, {'water': 1})}
    with pytest.raises(UnitOperationError, match=message):
        HeatExchanger('S', thermo, shell_parameters(Q=50, **updates)).solve(inlets)


@pytest.mark.parametrize('tube_condenses', [True, False])
def test_pure_condensation_on_tube_and_annulus(tube_condenses):
    thermo = water_backend()
    steam = thermo.calculate_state_PQ(5, .8, 20, {'water': 1})
    coolant = thermo.calculate_state(350, 3, 200, {'water': 1})
    inlets = {'tube_in': steam, 'shell_in': coolant} if tube_condenses else {'shell_in': steam, 'tube_in': coolant}
    result = HeatExchanger('C', thermo, parameters(Q=20, curve_segments=4)).solve(inlets)
    side = 'tube' if tube_condenses else 'annulus'
    assert all(node[side]['flow_regime'] == 'film_condensation' for node in result.performance['calculated_U_nodes'])
    steam_out = result.outlet_streams['tube_out' if tube_condenses else 'shell_out']
    assert steam.F*(steam.H-steam_out.H) == pytest.approx(20*3600, rel=1e-7)
    assert steam_out.vapor_fraction < .8


def test_saturated_boiling_and_condensation_share_the_wall_solve():
    thermo = water_backend()
    hot = thermo.calculate_state_PQ(5, .8, 20, {'water': 1})
    cold = thermo.calculate_state_PQ(3, .05, 100, {'water': 1})
    result = HeatExchanger('B', thermo, parameters(Q=20, curve_segments=4)).solve({'shell_in': hot, 'tube_in': cold})
    for node in result.performance['calculated_U_nodes']:
        assert node['tube']['flow_regime'] == 'saturated_boiling'
        assert node['annulus']['flow_regime'] == 'film_condensation'
        assert node['wall_temperature_residual_K'] < 1e-5
        assert node['tube']['wall_T_K'] > cold.T
        assert node['annulus']['wall_T_K'] < hot.T
    assert result.outlet_streams['tube_out'].vapor_fraction > .05


def test_boiling_dryout_limit_is_rejected_without_clamping():
    with pytest.raises(TransportCorrelationError, match='0.7'):
        shah_boiling_coefficient(300, .8, .02, 950, 2, .0003, .65, 4200, 2.1e6, 50000)


def test_shell_steam_condensation_uses_horizontal_bundle_film_model():
    thermo = water_backend()
    steam = thermo.calculate_state_PQ(5, .95, 15, {'water': 1})
    process = thermo.calculate_state(320, 3, 400, {'water': 1})
    result = HeatExchanger('S', thermo, shell_parameters(Q=10)).solve({'tube_in': process, 'shell_in': steam})
    assert all(node['shell']['correlation'] == 'nusselt_horizontal_bundle'
               for node in result.performance['calculated_U_nodes'])


def test_shell_saturated_pool_boiling_requires_surface_and_wetting_specification():
    thermo = water_backend()
    hot = thermo.calculate_state(440, 10, 100, {'water': 1})
    boiling = thermo.calculate_state_PQ(3, .02, 50, {'water': 1})
    base = shell_parameters(Q=5)
    with pytest.raises(UnitOperationError, match='shell_pool_boiling'):
        HeatExchanger('S', thermo, base).solve({'tube_in': hot, 'shell_in': boiling})
    result = HeatExchanger('S', thermo, {**base, 'shell_pool_boiling': True, 'boiling_Csf': .013}).solve(
        {'tube_in': hot, 'shell_in': boiling})
    assert all(node['shell']['correlation'] == 'rohsenow' for node in result.performance['calculated_U_nodes'])
    assert result.outlet_streams['shell_out'].vapor_fraction > boiling.vapor_fraction


def test_sensible_to_condensing_to_subcooled_curve_splits_at_both_phase_boundaries():
    thermo = water_backend()
    steam = thermo.calculate_state(450, 5, 1, {'water': 1})
    coolant = thermo.calculate_state(310, 3, 100, {'water': 1})
    result = HeatExchanger('Z', thermo, parameters(T_tube_out=380, curve_segments=4)).solve(
        {'tube_in': steam, 'shell_in': coolant})
    regimes = {node['tube']['flow_regime'] for node in result.performance['calculated_U_nodes']}
    assert 'film_condensation' in regimes
    assert 'laminar' in regimes or 'turbulent' in regimes
    assert any('superheat_treatment' in node['tube'] for node in result.performance['calculated_U_nodes'])
    assert result.outlet_streams['tube_out'].vapor_fraction == 0
    assert steam.F*(steam.H-result.outlet_streams['tube_out'].H) == pytest.approx(
        coolant.F*(result.outlet_streams['shell_out'].H-coolant.H), rel=1e-7)


def test_two_phase_mixture_is_explicitly_rejected_before_curve_reconstruction():
    thermo = create_thermodynamics(['water', 'ethanol'], 'IDEAL', db=ChemicalDatabase(enable_online=False))
    state = thermo.calculate_state(350, 3, 100, {'water': .5, 'ethanol': .5})
    state.vapor_fraction = .5
    state.liquid1_fraction = .5
    cold = thermo.calculate_state(300, 3, 100, {'water': .5, 'ethanol': .5})
    with pytest.raises(UnitOperationError, match='pure fluids'):
        HeatExchanger('M', thermo, parameters(Q=1)).solve({'tube_in': state, 'shell_in': cold})


def test_steam_heater_sizes_utility_and_area_without_adding_process_water():
    thermo = create_thermodynamics(['ethanol'], 'IDEAL', db=ChemicalDatabase(enable_online=False))
    inlet = thermo.calculate_state(300, 3, 100, {'ethanol': 1})
    result = Heater('H', thermo, {'T_out': 330, 'utility': 'steam', 'utility_P': 5,
                                'U': 500, 'curve_segments': 4}).solve({'in': inlet})
    p = result.performance
    assert p['area_required_m2'] > 0
    assert p['utility_mass_flow_kg_h'] > 0
    assert result.outlet_streams['out'].composition == {'ethanol': 1}
    assert set(thermo.components) == {'ethanol'}
    assert p['utility_vapor_fraction_in'] == 1
    assert p['utility_vapor_fraction_out'] == 0
    assert not p['utility_area_preliminary']


def test_steam_heater_uses_calculated_shell_tube_area():
    thermo = water_backend()
    inlet = thermo.calculate_state(300, 3, 200, {'water': 1})
    result = Heater('H', thermo, shell_parameters(T_out=340, utility='steam', utility_P=5)).solve({'in': inlet})
    p = result.performance
    assert p['utility_sizing']['U_model'] == 'shell_tube'
    assert all(node['shell']['correlation'] == 'nusselt_horizontal_bundle'
               for node in p['utility_sizing']['calculated_U_nodes'])
    assert p['area_required_m2'] > 0


@pytest.mark.parametrize('utility', ['cooling_water', 'chilled_water'])
def test_water_cooler_utility_energy_and_area(utility):
    thermo = water_backend()
    inlet = thermo.calculate_state(360, 3, 200, {'water': 1})
    result = Cooler('C', thermo, {'T_out': 330, 'utility': utility,
        'utility_T_in': 290, 'utility_T_out': 305, 'U': 700, 'curve_segments': 4}).solve({'in': inlet})
    p = result.performance
    supply = thermo.calculate_state(290, 1, 1, {'water': 1})
    returning = thermo.calculate_state(305, 1, 1, {'water': 1})
    assert p['utility_flow_kmol_h']*(returning.H-supply.H) == pytest.approx(-result.heat_duty, rel=1e-7)
    assert p['area_required_m2'] > 0


def test_air_utility_and_gas_conductivity_mixing():
    thermo = water_backend()
    inlet = thermo.calculate_state(360, 3, 200, {'water': 1})
    result = Cooler('A', thermo, {'T_out': 340, 'utility': 'air', 'utility_T_in': 300,
        'utility_T_out': 320, 'U': 100, 'curve_segments': 4}).solve({'in': inlet})
    assert result.performance['utility_composition'] == {'N2': .79, 'O2': .21}
    assert result.performance['area_required_m2'] > 0
    air = create_thermodynamics(['N2', 'O2'], 'IDEAL', db=ChemicalDatabase(enable_online=False))
    conductivity = air.transport_mixture_thermal_conductivity({'N2': .79, 'O2': .21}, 300, 1, 1).vapor
    assert .015 < conductivity < .04


def test_geometry_based_air_cooler_uses_gas_mixture_transport():
    thermo = water_backend()
    inlet = thermo.calculate_state(360, 3, 400, {'water': 1})
    result = Cooler('A', thermo, shell_parameters(T_out=340, utility='air', utility_T_in=300,
        utility_T_out=320)).solve({'in': inlet})
    performance = result.performance['utility_sizing']
    assert all(node['shell']['phase'] == 'vapor' for node in performance['calculated_U_nodes'])
    assert all(node['shell']['correlation'] == 'zukauskas_bell_corrections'
               for node in performance['calculated_U_nodes'])
    assert result.performance['area_required_m2'] > 0


def test_default_utility_auto_u_is_marked_preliminary():
    thermo = water_backend()
    inlet = thermo.calculate_state(300, 3, 100, {'water': 1})
    result = Heater('H', thermo, {'T_out': 330, 'utility': 'steam', 'utility_P': 5,
                                'curve_segments': 4}).solve({'in': inlet})
    assert result.performance['utility_area_preliminary']
    assert result.performance['area_required_m2'] > 0
    assert any('preliminary' in warning for warning in result.warnings)


def test_fired_utility_resolves_default_methane_heating_value():
    thermo = water_backend()
    inlet = thermo.calculate_state(300, 3, 100, {'water': 1})
    result = Heater('F', thermo, {'Q': 10, 'utility': 'fired', 'utility_heat_flux': 20000}).solve({'in': inlet})
    performance = result.performance
    assert performance['utility_fuel'] == 'methane'
    assert 40000 < performance['fuel_heating_value_kJ_kg'] < 60000
    assert performance['fuel_mass_flow_kg_h'] > 0


def test_heater_retains_frozen_henry_path_with_a_utility(monkeypatch):
    from unit_operations_basic import Flash
    thermo = water_backend()
    inlet = thermo.calculate_state(300, 3, 100, {'water': 1})
    context = object()
    calls = []
    monkeypatch.setattr(Flash, '_flash_henry_context', lambda self, *args: (context, {'active': True}, []))
    monkeypatch.setattr(Flash, '_stream_enthalpy', lambda self, stream, selected: stream.H)
    def tp(self, composition, T, P, F, selected):
        calls.append(('TP', selected))
        return thermo.calculate_state(T, P, F, composition)
    def ph(self, composition, P, H, F, T, selected):
        calls.append(('PH', selected))
        return thermo.calculate_state_PH(P, H, F, composition)
    monkeypatch.setattr(Flash, '_state_at_TP', tp)
    monkeypatch.setattr(Flash, '_state_at_PH', ph)
    monkeypatch.setattr(Heater, '_henry_solution_warnings', lambda *args: [])
    result = Heater('H', thermo, {'T_out': 330, 'utility': 'steam', 'utility_P': 5,
        'henry_components': 'auto', 'U': 500, 'curve_segments': 4}).solve({'in': inlet})
    assert result.performance['henry']['active']
    assert any(kind == 'PH' for kind, _selected in calls)
    assert all(selected is context for _kind, selected in calls)
    assert result.performance['area_required_m2'] > 0


def test_runnable_examples_and_units_through_flowsheet_api():
    from pathlib import Path
    from simulator import Simulator
    root = Path(__file__).resolve().parents[1]
    for name in ('shell_tube_calculated_u.pfd', 'utility_surface_sizing.pfd'):
        sim = Simulator.from_file(str(root/'examples'/name))
        assert sim.run().converged
        units = sim.get_results_dict()['units']
        assert all(record['performance']['area_required_m2'] > 0 for record in units.values())
        if name == 'utility_surface_sizing.pfd':
            assert units['H-STEAM']['performance']['utility_P_in_bar'] == 5


def test_hot_oil_with_an_explicit_fluid_identity():
    thermo = water_backend()
    inlet = thermo.calculate_state(300, 3, 100, {'water': 1})
    result = Heater('O', thermo, {'T_out': 330, 'utility': 'hot_oil', 'utility_fluid': 'decane',
        'utility_T_in': 430, 'utility_T_out': 400, 'U': 350, 'curve_segments': 4}).solve({'in': inlet})
    assert result.performance['utility_composition'] == {'decane': 1}
    assert result.performance['area_required_m2'] > 0


@pytest.mark.parametrize('utility,efficiency', [('electric', 1), ('fired', .85)])
def test_electric_and_fired_area_power_and_fuel(utility, efficiency):
    thermo = water_backend()
    inlet = thermo.calculate_state(300, 3, 100, {'water': 1})
    result = Heater('E', thermo, {'Q': 50, 'utility': utility, 'utility_heat_flux': 10,
        '__unit__utility_heat_flux': 'kW/m2', 'fuel_heating_value': 50,
        '__unit__fuel_heating_value': 'MJ/kg'}).solve({'in': inlet})
    p = result.performance
    assert p['area_required_m2'] == pytest.approx(5)
    assert p['utility_input_kW'] == pytest.approx(50/efficiency)
    if utility == 'electric':
        assert p['electric_power_kW'] == 50
    else:
        assert p['fuel_mass_flow_kg_h'] == pytest.approx(50/.85*3600/50000)


def test_pure_refrigerant_quality_states_keep_enthalpy_and_phase_inventory():
    backend = create_thermodynamics(['ammonia'], 'PR', db=ChemicalDatabase(enable_online=False))
    helper = HeatExchanger('R', backend, {})
    liquid = helper._state_for_vapor_fraction({'ammonia': 1}, 8, 1, 0)
    vapor = helper._state_for_vapor_fraction({'ammonia': 1}, 8, 1, 1)
    partial = helper._state_for_vapor_fraction({'ammonia': 1}, 8, 1, .3)
    assert partial.H == pytest.approx(.7*liquid.H+.3*vapor.H, rel=1e-9)
    assert partial.fluid_vapor_fraction == pytest.approx(.3)
    assert partial.effective_liquid1_fraction == pytest.approx(.7)
    assert partial.T == pytest.approx(backend.bubble_point_T({'ammonia': 1}, 8), abs=1e-4)


def test_refrigerant_cooling_utility_consumption_and_area():
    thermo = water_backend()
    inlet = thermo.calculate_state(350, 3, 200, {'water': 1})
    result = Cooler('R', thermo, {'T_out': 330, 'utility': 'refrigerant',
        'utility_fluid': 'ammonia', 'utility_P': 8, 'U': 700,
        'curve_segments': 4}).solve({'in': inlet})
    assert result.performance['utility_vapor_fraction_in'] == 0
    assert result.performance['utility_vapor_fraction_out'] == 1
    assert result.performance['utility_mass_flow_kg_h'] > 0
    assert result.performance['area_required_m2'] > 0


@pytest.mark.parametrize('params,message', [
    ({'utility': 'steam'}, 'requires utility_P'),
    ({'utility': 'hot_oil', 'utility_T_in': 430, 'utility_T_out': 400}, 'utility_fluid'),
    ({'utility': 'electric'}, 'utility_heat_flux'),
    ({'utility': 'steam', 'utility_P': 5, 'utility_P_drop': 6}, 'drop'),
    ({'utility': 'hot_water', 'utility_T_in': 320, 'utility_T_out': 340}, 'enthalpies'),
])
def test_missing_or_inconsistent_utility_conditions_fail(params, message):
    thermo = water_backend()
    inlet = thermo.calculate_state(300, 3, 100, {'water': 1})
    with pytest.raises(UnitOperationError, match=message):
        Heater('H', thermo, {'T_out': 330, **params}).solve({'in': inlet})


def test_guided_controls_keep_all_common_utilities_and_geometry():
    heater = {item['name'].lower(): item for item in unit_setting_schema('Heater')}
    cooler = {item['name'].lower(): item for item in unit_setting_schema('Cooler')}
    assert {'steam', 'hot_oil', 'electric', 'fired'} <= set(heater['utility']['values'])
    assert {'cooling_water', 'chilled_water', 'refrigerant', 'air'} <= set(cooler['utility']['values'])
    for fields in (heater, cooler):
        assert 'thermal_fluid' in fields['utility']['values']
        assert fields['utility_p']['unit'] == 'bar'
        assert fields['utility_t_in']['unit'] == 'C'
        assert fields['tube_inner_diameter']['unit'] == 'm'
        assert 'shell_tube' in fields['u_model']['values']
        assert fields['curve_segments']['default'] == 40
        assert 'cocurrent' in fields['flow_pattern']['values']


@pytest.mark.parametrize('gas_ratio', [None, .9])
def test_round_tube_transition_is_continuous_and_interpolates(gas_ratio):
    kwargs = {'gas_temperature_ratio': gas_ratio}
    laminar = internal_flow_transfer(2300-1e-5, 5, 4, **kwargs)['nusselt']
    turbulent = internal_flow_transfer(4000, 5, 4, **kwargs)['nusselt']
    middle = internal_flow_transfer(3150, 5, 4, **kwargs)
    assert middle['flow_regime'] == 'transition'
    assert middle['reynolds'] == 3150
    assert middle['nusselt'] == pytest.approx((laminar+turbulent)/2)
    assert internal_flow_transfer(2300, 5, 4, **kwargs)['nusselt'] == pytest.approx(laminar)
    assert internal_flow_transfer(4000-1e-5, 5, 4, **kwargs)['nusselt'] == pytest.approx(turbulent)


def test_calculated_exchanger_accepts_a_transitional_tube_stream():
    thermo = ConstantLiquid()
    # Re=3150 in the tube; the annulus remains fully turbulent.
    tube_flow = 3150*math.pi*.02*.001/4*3600/18
    hot = thermo.calculate_state(360, 3, tube_flow, {'water': 1})
    cold = thermo.calculate_state(300, 3, 300, {'water': 1})
    result = HeatExchanger('T', thermo, parameters(Q=1, curve_segments=4)).solve(
        {'tube_in': hot, 'shell_in': cold})
    assert all(node['tube']['flow_regime'] == 'transition'
               for node in result.performance['calculated_U_nodes'])
    assert hot.F*(hot.H-result.outlet_streams['tube_out'].H) == pytest.approx(3600)


@pytest.mark.parametrize('model', ['specified', 'double_pipe'])
def test_area_unit_conversion_is_shared_by_design_and_rating(model):
    thermo = ConstantLiquid()
    inlets = {'tube_in': thermo.calculate_state(360, 3, 200, {'water': 1}),
              'shell_in': thermo.calculate_state(300, 3, 300, {'water': 1})}
    common = {'U': 500} if model == 'specified' else parameters(U_model=model)
    metric = HeatExchanger('M', thermo, {**common, 'A': .9290304}).solve(inlets)
    imperial = HeatExchanger('I', thermo, {**common, 'A': 10, '__unit__A': 'ft2'}).solve(inlets)
    assert imperial.performance['area_m2'] == pytest.approx(.9290304)
    assert imperial.heat_duty == pytest.approx(metric.heat_duty)


@pytest.mark.parametrize('params', [
    {'U': float('nan')}, {'U': float('inf')}, {'A': float('nan')},
    {'UA': float('inf')}, {'UA': 10, '__unit__UA': 'unknown'},
    {'U': 500, '__unit__U': 'unknown'}, {'A': 10, '__unit__A': 'unknown'},
    {'P_drop_tube': float('nan')},
])
def test_sizing_rejects_nonfinite_values_and_unknown_units(params):
    thermo = ConstantLiquid()
    inlets = {'tube_in': thermo.calculate_state(360, 3, 200, {'water': 1}),
              'shell_in': thermo.calculate_state(300, 3, 300, {'water': 1})}
    with pytest.raises(UnitOperationError, match='finite|unsupported unit'):
        HeatExchanger('I', thermo, {'Q': 25, **params}).solve(inlets)


@pytest.mark.parametrize('auto', [False, True])
def test_explicit_lmtd_correction_scales_area_and_conductance(auto):
    thermo = water_backend()
    inlets = {'tube_in': thermo.calculate_state(360, 3, 200, {'water': 1}),
              'shell_in': thermo.calculate_state(300, 3, 300, {'water': 1})}
    params = {'Q': 25, 'U': 'auto' if auto else 500, 'curve_segments': 4}
    plain = HeatExchanger('P', thermo, params).solve(inlets).performance
    corrected = HeatExchanger('C', thermo, {**params, 'LMTD_correction': .8}).solve(inlets).performance
    key = 'auto_U_area_required_m2' if auto else 'area_required_m2'
    assert corrected[key] == pytest.approx(plain[key]/.8)
    assert corrected['UA_required_W_per_K'] == pytest.approx(plain['UA_required_W_per_K']/.8)
    assert corrected['LMTD_correction'] == .8
    rated = HeatExchanger('R', thermo, {'U': params['U'], 'A': corrected[key],
        'LMTD_correction': .8, 'curve_segments': 4}).solve(inlets)
    assert rated.performance['duty_kW'] == pytest.approx(25, rel=1e-6)
    if auto:
        assert sum(s['area_m2'] for s in corrected['auto_U_segments']) == pytest.approx(corrected[key])


def test_countercurrent_auto_u_uses_the_locally_paired_phase_regimes():
    thermo = water_backend()
    hot = thermo.calculate_state_PQ(5, .8, 10, {'water': 1})
    cold = thermo.calculate_state(350, 3, 5, {'water': 1})
    hx = HeatExchanger('P', thermo, {'Q': 20, 'U': 'auto', 'curve_segments': 4})
    performance = hx.solve({'tube_in': hot, 'shell_in': cold}).performance
    cold_states = [thermo.calculate_state_PH(3, cold.H+72000*f/5, 5, {'water': 1})
                   for f in (0, .25, .5, .75, 1)]
    expected_area = 0.
    expected_services = []
    for i in range(4):
        a, b = cold_states[3-i:5-i]
        service = 'water_steam_vaporizer' if b.vapor_fraction-a.vapor_fraction > 1e-4 else 'steam_feedwater'
        expected_services.append(service)
        dt1, dt2 = hot.T-a.T, hot.T-b.T
        lmtd = dt1 if abs(dt1-dt2) < 1e-8 else (dt1-dt2)/math.log(dt1/dt2)
        expected_area += 5000/lmtd/hx.SEADER_U_ESTIMATE_TABLE[service]['typical']
    assert len(set(expected_services)) == 2
    assert [s['service'] for s in performance['auto_U_segments']] == expected_services
    # PH reconstruction permits a small enthalpy residual; compare sizing at
    # its numerical accuracy rather than pinning its temperature history.
    assert performance['auto_U_area_required_m2'] == pytest.approx(expected_area, rel=2e-6)


def test_condensation_does_not_require_an_unused_surface_tension(monkeypatch):
    thermo = water_backend()
    def missing(*args, **kwargs):
        raise AssertionError('Condensation does not use surface tension')
    monkeypatch.setattr(thermo, 'transport_mixture_surface_tension', missing)
    steam = thermo.calculate_state_PQ(5, .8, 20, {'water': 1})
    coolant = thermo.calculate_state(350, 3, 200, {'water': 1})
    result = HeatExchanger('C', thermo, parameters(Q=20, curve_segments=4)).solve(
        {'tube_in': steam, 'shell_in': coolant})
    assert result.performance['area_required_m2'] > 0


def test_wetted_shell_pool_does_not_apply_the_flow_boiling_quality_limit():
    thermo = water_backend()
    hot = thermo.calculate_state(440, 10, 100, {'water': 1})
    boiling = thermo.calculate_state_PQ(3, .8, 50, {'water': 1})
    result = HeatExchanger('P', thermo, shell_parameters(
        Q=5, shell_pool_boiling=True, boiling_Csf=.013)).solve({'tube_in': hot, 'shell_in': boiling})
    assert all(node['shell']['correlation'] == 'rohsenow' for node in result.performance['calculated_U_nodes'])


@pytest.mark.parametrize('key', ['UA', 'UA_available', 'LMTD_correction'])
def test_electric_sizing_rejects_fluid_capacity_and_correction_inputs(key):
    thermo = water_backend()
    inlet = thermo.calculate_state(300, 3, 100, {'water': 1})
    with pytest.raises(UnitOperationError, match='fluid-film'):
        Heater('E', thermo, {'Q': 5, 'utility': 'electric', 'utility_heat_flux': 10000,
                            key: 1}).solve({'in': inlet})


def test_physical_side_pressure_drops_are_converted_by_the_flowsheet():
    from pathlib import Path
    from simulator import Simulator
    source = (Path(__file__).resolve().parents[1]/'examples'/'double_pipe_calculated_u.pfd').read_text()
    source = source.rstrip()+'\n        P_drop_tube = 10 [kPa]\n        P_drop_shell = 5 [kPa]\n'
    sim = Simulator.from_string(source)
    assert sim.run().converged
    p = sim.get_results_dict()['units']['HX-100']['performance']
    assert p['P_drop_hot_bar'] == pytest.approx(.1)
    assert p['P_drop_cold_bar'] == pytest.approx(.05)


@pytest.mark.parametrize('operation,supply,returning', [(Heater, 360, 340), (Cooler, 280, 290)])
def test_thermal_fluid_supports_glycol_water_loops(operation, supply, returning):
    thermo = water_backend()
    inlet = thermo.calculate_state(330 if operation is Cooler else 300, 3, 100, {'water': 1})
    composition = {'water': .7, 'ethylene glycol': .3}
    result = operation('G', thermo, {'utility': 'thermal_fluid',
        'utility_composition': composition, 'utility_T_in': supply,
        'utility_T_out': returning, 'T_out': 310, 'U': 500,
        'curve_segments': 4}).solve({'in': inlet})
    backend = create_thermodynamics(list(composition), 'IDEAL', db=ChemicalDatabase(enable_online=False))
    delta_h = (backend.calculate_state(returning, 1, 1, composition).H
               - backend.calculate_state(supply, 1, 1, composition).H)
    assert result.performance['utility_flow_kmol_h']*delta_h == pytest.approx(-result.heat_duty, rel=1e-7)
    assert result.performance['area_required_m2'] > 0
    assert result.performance['utility_composition'] == composition
    assert result.performance['utility_vapor_fraction_in'] == 0
    assert result.performance['utility_vapor_fraction_out'] == 0


@pytest.mark.parametrize('unit,value', [('W/m2-K', 500), ('kW/m2-K', .5),
                                       ('Btu/(h ft2 F)', 500/5.6783)])
def test_specified_u_units_have_the_same_sizing(unit, value):
    thermo = ConstantLiquid()
    inlets = {'tube_in': thermo.calculate_state(360, 3, 200, {'water': 1}),
              'shell_in': thermo.calculate_state(300, 3, 300, {'water': 1})}
    p = HeatExchanger('U', thermo, {'Q': 25, 'U': value, '__unit__U': unit}).solve(inlets).performance
    assert p['U_W_m2_K'] == pytest.approx(500)
    assert p['area_required_m2'] == pytest.approx(.9091911042718753)
