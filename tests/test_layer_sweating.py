"""Thermodynamics, finite mass transfer, and capillary drainage in layer sweating."""
from types import SimpleNamespace
import math

import pytest

import layer_sweating
from chemical_properties import ChemicalDatabase
from crystallizer_specs import CrystallizerSpecificationError, validate_layer_crystallizer_specification
from dof_analyzer import SpecificationStatus, analyze_dof
from layer_sweating import (
    LayerSweatingError, SolidSolutionSolute, allocate_layer_inventory,
    layer_enthalpy, solve_layer_sweating,
)
from pfd_parser import parse_pfd, validate_pfd
from simulator import Simulator
from thermodynamics_models.base import IdealThermodynamics
from unit_operations_base import UnitOperationError
from unit_operations_solids import LayerCrystallizer


class ConstantPropertyLiquid:
    components = ['host', 'impurity']
    props = {'host': SimpleNamespace(MW=18, Tm=400), 'impurity': SimpleNamespace(MW=46)}

    def mixture_liquid_density(self, x, T):
        return 50.0  # kmol/m3

    def mixture_viscosity(self, x, T, P, V):
        return 0.001  # Pa s


    def enthalpy_liquid(self, component, T):
        return .1 * (T - 280)

    def process_solid_enthalpy(self, component, T):
        return .1 * (T - 280)

    def mixture_enthalpy(self, composition, T, V, P):
        return 1000 * sum(x * self.enthalpy_liquid(c, T) for c, x in composition.items())


@pytest.fixture
def kernel(monkeypatch):
    monkeypatch.setattr(layer_sweating, 'pure_solid_log_saturation_activity', lambda *a, **k: math.log(.8))
    def run(**changes):
        values = dict(host='host', initial_solid={'host': 8.0},
                      initial_liquid={'host': 1.0, 'impurity': 1.0}, solutes={},
                      temperature=280., pressure=1., duration_s=200.,
                      heater_temperature=280., thermal_conductance_W_K=1000., opening_coefficient=0.,
                      host_rate_constant=.2, diffusion_length=1e-4,
                      drainage_length=.5, pore_radius=1e-5, tortuosity=2.,
                      connected_fraction=.5, residual_saturation=.1,
                      capillary_pressure=0., solid_molar_volume=.02)
        values.update(changes)
        return solve_layer_sweating(ConstantPropertyLiquid(), **values)
    return run


def test_closed_pores_converge_to_lever_rule_and_conserve(kernel):
    result = kernel(connected_fraction=0, duration_s=1000)
    # x_host^L=.8; one kmol insoluble impurity requires five kmol liquid.
    assert result.solid_amounts['host'] == pytest.approx(5, abs=2e-6)
    assert result.liquid_amounts == pytest.approx({'host': 4, 'impurity': 1}, abs=2e-6)
    assert sum(result.sweat_amounts.values()) == 0
    assert result.component_balance_residual < 1e-10
    assert min(p['phase_transfer_dissipation_kJ_s'] for p in result.profile) >= -1e-12


def test_drainage_depends_on_pore_size_time_and_connectivity(kernel):
    small = kernel(pore_radius=2e-6)
    large = kernel(pore_radius=2e-5)
    longer = kernel(pore_radius=2e-5, duration_s=400)
    blocked = kernel(connected_fraction=0)
    assert 0 < sum(small.sweat_amounts.values()) < sum(large.sweat_amounts.values())
    assert sum(large.sweat_amounts.values()) < sum(longer.sweat_amounts.values())
    assert sum(blocked.sweat_amounts.values()) == 0
    assert sum(longer.liquid_amounts.values()) > 0
    for result in (small, large, longer, blocked):
        assert result.component_balance_residual < 1e-9
        for name, total in {'host': 9, 'impurity': 1}.items():
            assert (result.solid_amounts[name] + result.liquid_amounts[name]
                    + result.sweat_amounts[name]) == pytest.approx(total)


def test_capillary_pressure_stops_drainage(kernel):
    result = kernel(capillary_pressure=1e6)
    assert sum(result.sweat_amounts.values()) == 0
    assert all(p['drainage_m3_s'] == 0 for p in result.profile)


def test_residual_saturation_limits_drainage_without_melting(kernel):
    result = kernel(host_rate_constant=1e-12, pore_radius=1e-3,
                    duration_s=2000, residual_saturation=.4)
    assert result.profile[-1]['liquid_saturation'] >= .4 - 1e-7
    assert 0 < sum(result.sweat_amounts.values()) < 2


def test_bound_impurity_leaves_with_melting_and_by_diffusive_exchange(kernel):
    base = dict(initial_solid={'host': 8., 'impurity': .5},
                initial_liquid={'host': 1., 'impurity': .5})
    frozen = kernel(**base, solutes={'impurity': SolidSolutionSolute(.02, -1., 0)})
    mobile = kernel(**base, solutes={'impurity': SolidSolutionSolute(.02, -1., 1e-10)})
    assert frozen.solid_amounts['impurity'] < .5
    assert frozen.solid_amounts['impurity'] / frozen.solid_amounts['host'] == pytest.approx(.5 / 8)
    assert mobile.solid_amounts['impurity'] < frozen.solid_amounts['impurity']
    assert mobile.sweat_amounts['impurity'] > frozen.sweat_amounts['impurity']
    assert min(p['phase_transfer_dissipation_kJ_s'] for p in mobile.profile) >= -1e-10


def test_closed_solid_solution_reaches_equality_of_chemical_potentials(kernel):
    result = kernel(initial_solid={'host': 8., 'impurity': .1},
                    initial_liquid={'host': 1., 'impurity': .9},
                    solutes={'impurity': SolidSolutionSolute(.1, 0., 1e-9)},
                    connected_fraction=0, duration_s=2000)
    S, L = sum(result.solid_amounts.values()), sum(result.liquid_amounts.values())
    assert result.solid_amounts['host'] / S == pytest.approx(
        result.liquid_amounts['host'] / L / .8, abs=1e-6)
    assert result.solid_amounts['impurity'] / S == pytest.approx(
        .1 * result.liquid_amounts['impurity'] / L, abs=1e-6)


def test_initial_occluded_fraction_changes_finite_time_purification(kernel):
    solutes = {'impurity': SolidSolutionSolute(.02, 0., 1e-14)}
    occluded = kernel(solutes=solutes)
    bound = kernel(solutes=solutes, initial_solid={'host': 8., 'impurity': .8},
                   initial_liquid={'host': 1., 'impurity': .2})
    assert occluded.sweat_amounts['impurity'] > bound.sweat_amounts['impurity']


def test_trace_component_conservation_and_tolerance_convergence(kernel):
    values = dict(initial_solid={'host': 8.},
                  initial_liquid={'host': 1., 'impurity': 1e-10}, duration_s=20.)
    ordinary = kernel(**values)
    refined = kernel(**values, relative_tolerance=1e-9)
    assert ordinary.sweat_amounts == pytest.approx(refined.sweat_amounts, rel=1e-5, abs=1e-18)
    assert ordinary.component_balance_residual < 1e-10


def test_finite_heating_matches_analytical_sensible_heat_response(kernel):
    duration, conductance = 200., 1000.
    result = kernel(heater_temperature=290., thermal_conductance_W_K=conductance,
                    connected_fraction=0., duration_s=duration)
    # Ten kmol with Cp=100 kJ/(kmol K), no latent heat in this fixture.
    capacity = 1000.
    expected = 290 - 10 * math.exp(-conductance / 1000 * duration / capacity)
    assert result.temperature == pytest.approx(expected, abs=2e-6)
    assert result.supplied_heat_kJ == pytest.approx(capacity * (expected - 280), rel=1e-6)
    assert result.energy_balance_residual < 1e-9


def test_sealed_inventory_cannot_drain_without_opening(kernel):
    result = kernel(connected_fraction=.2, opening_coefficient=0.,
                    pore_radius=1e-4, duration_s=1000)
    # Insoluble impurity initially sealed cannot exchange or escape.
    assert result.sealed_liquid_amounts['impurity'] == pytest.approx(.8)
    assert result.sweat_amounts['impurity'] <= .2 + 1e-9
    assert result.connected_liquid_amounts['impurity'] > 0


def test_melting_opens_sealed_regions_and_preserves_every_component(kernel):
    sealed = kernel(connected_fraction=0., opening_coefficient=0.)
    opening = kernel(connected_fraction=0., opening_coefficient=2.)
    assert sum(sealed.sweat_amounts.values()) == 0
    assert opening.sweat_amounts['impurity'] > 0
    assert opening.sealed_liquid_amounts['impurity'] < 1
    assert max(p['opening_rate_per_s'] for p in opening.profile) > 0
    assert opening.component_balance_residual < 1e-9
    assert opening.energy_balance_residual < 1e-9


def test_no_melting_means_no_opening(kernel):
    # Initial liquid is saturated with the pure host.
    result = kernel(initial_liquid={'host': 4., 'impurity': 1.},
                    connected_fraction=0., opening_coefficient=10.)
    assert sum(result.sweat_amounts.values()) == pytest.approx(0., abs=1e-12)
    assert result.sealed_liquid_amounts['impurity'] == pytest.approx(1.)


def test_latent_heat_cools_an_adiabatic_layer_and_heat_supply_increases_melting(monkeypatch):
    class FusionLiquid(ConstantPropertyLiquid):
        props = {'host': SimpleNamespace(MW=18, Tm=300), 'impurity': SimpleNamespace(MW=46)}

        def process_solid_enthalpy(self, component, T):
            return self.enthalpy_liquid(component, T) - 6.

    monkeypatch.setattr(layer_sweating, 'pure_solid_log_saturation_activity',
                        lambda _thermo, _host, T, _P: -6000 / layer_sweating.R * (1/T - 1/300))
    args = dict(host='host', initial_solid={'host': 8.},
                initial_liquid={'host': 1., 'impurity': 1.}, solutes={},
                temperature=280., pressure=1., heater_temperature=295.,
                opening_coefficient=0., duration_s=200., host_rate_constant=.2,
                diffusion_length=1e-4, drainage_length=.5, pore_radius=1e-5,
                tortuosity=2., connected_fraction=0., residual_saturation=.1,
                capillary_pressure=0., solid_molar_volume=.02)
    adiabatic = solve_layer_sweating(FusionLiquid(), **args, thermal_conductance_W_K=0.)
    heated = solve_layer_sweating(FusionLiquid(), **args, thermal_conductance_W_K=1000.)
    assert adiabatic.temperature < 280
    assert adiabatic.supplied_heat_kJ == 0
    assert heated.temperature > adiabatic.temperature
    assert heated.solid_amounts['host'] < adiabatic.solid_amounts['host']
    for result in (adiabatic, heated):
        melted = 8 - result.solid_amounts['host']
        assert 1000 * (result.temperature - 280) + 6000 * melted == pytest.approx(
            result.supplied_heat_kJ, abs=1e-4)
        assert result.energy_balance_residual < 1e-9


def test_partition_temperature_dependence_matches_transfer_enthalpy():
    law = SolidSolutionSolute(.2, -3., 1e-12)
    T, step = 280., .001
    derivative = (law.log_partition(T + step) - law.log_partition(T - step)) / (2 * step)
    assert derivative == pytest.approx(1000 * law.transfer_enthalpy / (layer_sweating.R * T**2))


def test_allocation_accounts_for_occluded_solvent_and_bound_impurity():
    solid, liquid = allocate_layer_inventory(
        host='host', totals={'host': 9., 'impurity': 1.2}, host_solid=8.,
        captured={'impurity': 1.}, occluded_fractions={'impurity': .6},
        occluded_host_fraction=.5, empirical=True)
    assert solid == pytest.approx({'host': 7.4, 'impurity': .4})
    assert liquid == pytest.approx({'host': 1.6, 'impurity': .8})
    # Additional retained mother liquor is not relabeled as co-crystallized.
    assert solid['impurity'] == pytest.approx(.4)


@pytest.mark.parametrize('changes, message', [
    ({'occluded_fractions': {}}, 'Missing occluded_fraction'),
    ({'occluded_host_fraction': .99}, 'exhausts'),
    ({'empirical': False}, 'mechanistic growth'),
])
def test_allocation_rejects_missing_or_impossible_closures(changes, message):
    values = dict(host='host', totals={'host': 9., 'impurity': 1.}, host_solid=8.,
                  captured={'impurity': 1.}, occluded_fractions={'impurity': .5},
                  occluded_host_fraction=.5, empirical=True)
    values.update(changes)
    with pytest.raises(LayerSweatingError, match=message):
        allocate_layer_inventory(**values)


@pytest.fixture
def thermo():
    backend = IdealThermodynamics(['water', 'ethanol'], ChemicalDatabase(enable_online=False))
    backend.configure_permanent_solids(['water', 'ethanol'], [], conventional_solid_components=['water'])
    return backend


def empirical_params(**updates):
    return {
        'model': 'empirical_layer_growth', 'T': 270, 'T_wall': 250,
        'cooled_area': 1, 'growth_time': .01, 'cycle_time': 1,
        'layer_solid_density': 1000, 'growth_rate': 2, '__unit__growth_rate': 'um/s',
        'keff_ethanol': .2, 'occluded_fraction_ethanol': .5,
        'occluded_liquid_host_fraction': .5,
        'solid_partition_ethanol': .02, 'solid_transfer_enthalpy_ethanol': -1,
        'solid_diffusivity_ethanol': 1e-13, 'solid_diffusion_length': 1e-4,
        'sweat_heater_temperature': 272, 'sweat_thermal_conductance': 1.,
        'sweat_opening_coefficient': 1., 'harvest_temperature': 280,
        'sweat_collection_temperature': 280,
        'sweat_time': 1/60, 'sweat_host_rate_constant': .002,
        'sweat_drainage_length': .5, 'sweat_pore_radius': 1e-5,
        'sweat_tortuosity': 2, 'sweat_connected_fraction': .5,
        'sweat_residual_saturation': .1, 'sweat_capillary_pressure': 0,
        '__connected_outlet_ports__': ['product', 'mother_liquor', 'sweat'],
        **updates,
    }


def test_integrated_cycle_conserves_components_and_stage_energy(thermo):
    feed = thermo.calculate_state(270, 1, 10, {'water': .8, 'ethanol': .2}, phase='liquid')
    result = LayerCrystallizer('C', thermo, empirical_params()).solve({'in': feed})
    details = result.performance['layer_sweating']
    assert set(result.outlet_streams) == {'product', 'mother_liquor', 'sweat'}
    assert details['initial_solid_amounts_kmol']['ethanol'] > 0
    assert details['remaining_solid_amounts_kmol']['ethanol'] > 0
    assert details['remaining_liquid_amounts_kmol']['ethanol'] > 0
    assert 0 < details['impurity_rejection_to_sweat'] < 1
    assert details['energy_balance_residual'] < 1e-7
    assert details['remaining_sealed_liquid_amounts_kmol']['ethanol'] > 0
    assert details['sweating_duty_kW'] + details['collection_conditioning_duty_kW'] == pytest.approx(
        details['sweating_and_collection_duty_kW'], abs=1e-8)
    for c, amount in feed.component_flows().items():
        assert sum(s.component_flows().get(c, 0) for s in result.outlet_streams.values()) == pytest.approx(amount)
    assert sum(s.F * s.H for s in result.outlet_streams.values()) == pytest.approx(feed.F * feed.H + result.heat_duty)
    assert all(not s.solid_component_flows for s in result.outlet_streams.values())
    # Independently construct initial physical solid-solution energy.
    law = {'ethanol': SolidSolutionSolute(.02, -1, 1e-13)}
    initial = layer_enthalpy(thermo, 270, 1, details['initial_solid_amounts_kmol'],
                            details['initial_occluded_and_retained_liquid_amounts_kmol'], 'water', law)
    mother = result.outlet_streams['mother_liquor']
    growth_duty = (initial + mother.F * mother.H - feed.F * feed.H) / 3600
    assert (growth_duty + details['sweating_and_collection_duty_kW']
            + details['harvest_duty_kW']) == pytest.approx(result.heat_duty / 3600)


@pytest.mark.parametrize('change, message', [
    ({'sweat_crystal_fraction': .05}, 'predicts temperature'),
    ({'sweat_temperature': 272}, 'use sweat_heater_temperature'),
    ({'sweat_connected_fraction': 1.1}, 'at most'),
    ({'sweat_residual_saturation': 1}, 'must be < 1'),
    ({'model': 'equilibrium'}, 'requires model=layer_growth'),
])
def test_invalid_specifications(change, message):
    with pytest.raises(CrystallizerSpecificationError, match=message):
        validate_layer_crystallizer_specification(empirical_params(**change))


def test_cycle_time_must_cover_growth_and_sweating(thermo):
    feed = thermo.calculate_state(270, 1, 10, {'water': .8, 'ethanol': .2}, phase='liquid')
    with pytest.raises(UnitOperationError, match='cycle_time must include'):
        LayerCrystallizer('C', thermo, empirical_params(sweat_time=1)).solve({'in': feed})


def test_pure_occlusion_with_blocked_drainage_retains_all_impurity(thermo):
    params = empirical_params(occluded_fraction_ethanol=1., sweat_connected_fraction=0.,
                              sweat_opening_coefficient=0.)
    for name in ('solid_partition_ethanol', 'solid_transfer_enthalpy_ethanol',
                 'solid_diffusivity_ethanol', 'solid_diffusion_length'):
        params.pop(name)
    feed = thermo.calculate_state(270, 1, 10, {'water': .8, 'ethanol': .2}, phase='liquid')
    result = LayerCrystallizer('C', thermo, params).solve({'in': feed})
    assert result.outlet_streams['sweat'].F == 0
    assert result.performance['impurity_rejection_to_sweat'] == 0
    assert result.outlet_streams['product'].component_flows()['ethanol'] > 0


def test_harvest_rejects_unstable_solid_solution_liquid(thermo):
    feed = thermo.calculate_state(270, 1, 10, {'water': .8, 'ethanol': .2}, phase='liquid')
    with pytest.raises(UnitOperationError, match='unstable to solid-solution'):
        LayerCrystallizer('C', thermo, empirical_params(
            harvest_temperature=272, solid_partition_ethanol=100,
        )).solve({'in': feed})


PFD = '''
PROCESS: Physical layer sweating
ONLINE_LOOKUP: false
COMPONENTS:
    water | Water | type=conventional_with_solid
    ethanol | Ethanol
STREAM Feed : FEED -> C.in
    T = 270 [K]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.8, ethanol:0.2
STREAM Product : C.product -> PRODUCT
STREAM Mother : C.mother_liquor -> PRODUCT
STREAM Sweat : C.partial_melt -> PRODUCT
UNIT C : LayerCrystallizer
    model = empirical_layer_growth
    T = 270 [K]
    T_wall = 250 [K]
    cooled_area = 1 [m2]
    growth_time = 0.01 [h]
    cycle_time = 1 [h]
    layer_solid_density = 1000 [kg/m3]
    growth_rate = 2 [um/s]
    keff_ethanol = 0.2
    occluded_fraction_ethanol = 0.5
    occluded_liquid_host_fraction = 0.5
    solid_partition_ethanol = 0.02
    solid_transfer_enthalpy_ethanol = -1 [kJ/mol]
    solid_diffusivity_ethanol = 1e-13 [m2/s]
    solid_diffusion_length = 0.1 [mm]
    sweat_heater_temperature = 272 [K]
    sweat_thermal_conductance = 1 [W/K]
    sweat_opening_coefficient = 1
    sweat_collection_temperature = 280 [K]
    harvest_temperature = 280 [K]
    sweat_time = 1 [min]
    sweat_host_rate_constant = 0.002 [1/s]
    sweat_drainage_length = 0.5 [m]
    sweat_pore_radius = 10 [um]
    sweat_tortuosity = 2
    sweat_connected_fraction = 0.5
    sweat_residual_saturation = 0.1
    sweat_capillary_pressure = 0 [Pa]
'''


def test_pfd_ports_dof_units_runtime_and_round_trip():
    pfd = parse_pfd(PFD)
    assert validate_pfd(pfd) == ([], [])
    assert analyze_dof(pfd).unit_results[0].status == SpecificationStatus.OK
    reparsed = parse_pfd(pfd.to_pfd())
    assert validate_pfd(reparsed) == ([], [])
    result = Simulator(reparsed).run()
    assert result.converged, result.errors
    assert result.streams['Product'].T == pytest.approx(280)
    assert 0 < result.units['C'].performance['impurity_rejection_to_sweat'] < 1
