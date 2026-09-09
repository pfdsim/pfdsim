"""Equilibrium layer endpoints, drainage, and shared specification handling."""

import pytest

from chemical_properties import ChemicalDatabase
from crystallizer_specs import (
    CrystallizerSpecificationError,
    validate_crystallizer_specification,
)
from dof_analyzer import SpecificationStatus, analyze_dof
from pfd_parser import parse_pfd, validate_pfd
from simulator import Simulator
from thermodynamics_models.base import IdealThermodynamics
from unit_operations_base import UnitOperationError
from unit_operations_solids import Crystallizer


@pytest.fixture
def thermo():
    backend = IdealThermodynamics(
        ['water', 'ethanol'], ChemicalDatabase(enable_online=False)
    )
    backend.configure_permanent_solids(
        ['water', 'ethanol'], [], conventional_solid_components=['water']
    )
    backend.solid_particle_defaults['water'] = {
        'diameter_m': 1e-4, 'sphericity': 0.8,
    }
    return backend


@pytest.mark.parametrize('retention', [{}, {'mother_liquor_retention': 0.2},
                                    {'mother_liquor_retention_rate': 0.1}])
def test_layer_reuses_equilibrium_and_conserves_components_and_energy(thermo, retention):
    feed = thermo.calculate_state(
        280, 1, 10, {'water': 0.9, 'ethanol': 0.1}, phase='liquid'
    )
    params = {'T': 250, **retention}
    layer = Crystallizer('L', thermo, {
        **params, 'crystallization_mode': 'layer',
    }).solve({'in': feed})
    suspension = Crystallizer('S', thermo, {
        **params, **({} if retention else {'mother_liquor_retention': 0}),
    }).solve({'in': feed})
    assert layer.performance['model'] == 'equilibrium_pure_solid_layer'
    assert layer.performance['outlet_mode'] == 'layer_drainage'
    assert layer.performance['crystallization_mode'] == 'layer'
    assert layer.heat_duty == pytest.approx(suspension.heat_duty)
    assert layer.outlet_streams['cake'].solid_component_flows['water'] > 0
    assert not layer.outlet_streams['mother_liquor'].solid_component_flows
    for component, flow in feed.component_flows().items():
        assert sum(s.component_flows().get(component, 0)
                   for s in layer.outlet_streams.values()) == pytest.approx(flow)
    assert sum(s.F * s.H for s in layer.outlet_streams.values()) == pytest.approx(
        feed.F * feed.H + layer.heat_duty
    )
    assert abs(layer.performance['equilibrium_residuals']['water']) < 1e-8
    for port, stream in layer.outlet_streams.items():
        assert stream.component_flows() == pytest.approx(
            suspension.outlet_streams[port].component_flows()
        )
        assert not stream.solid_particle_size_distributions
        assert not stream.solid_particle_properties
        assert stream.phase_details['layer_crystallization']['model'] == (
            'equilibrium_pure_solid_layer'
        )
    assert suspension.outlet_streams['cake'].solid_particle_size_distributions
    if not retention:
        assert layer.outlet_streams['cake'].solid_fraction == pytest.approx(1)


@pytest.mark.parametrize('spec, ports, message', [
    ({'crystallization_mode': 'invalid'}, None, 'suspension or layer'),
    ({'model': 'MSMPR'}, None, 'MSMPR describes suspension'),
    ({'outlet_sphericity': 0.8}, None, 'only to suspension'),
    ({}, ['out'], 'cake and mother_liquor'),
    ({}, ['cake'], 'requires cake and mother_liquor'),
    ({'mother_liquor_retention': 0, 'mother_liquor_retention_rate': 0},
     None, 'only one mother-liquor retention'),
])
def test_invalid_layer_specifications(spec, ports, message):
    with pytest.raises(CrystallizerSpecificationError, match=message):
        validate_crystallizer_specification(
            {'T': 250, 'crystallization_mode': 'layer', **spec}, ports
        )


def test_layer_rejects_suspended_solids(thermo):
    feed = thermo.calculate_state_with_solid_flows(
        250, 1, 10, {'water': 0.9, 'ethanol': 0.1}, {'water': 1}, phase='liquid'
    )
    with pytest.raises(UnitOperationError, match='solid-free'):
        Crystallizer('L', thermo, {
            'T': 250, 'crystallization_mode': 'layer',
        }).solve({'in': feed})


def test_layer_without_deposition_has_empty_harvest(thermo):
    feed = thermo.calculate_state(
        280, 1, 10, {'water': 0.9, 'ethanol': 0.1}, phase='liquid'
    )
    result = Crystallizer('L', thermo, {
        'T': 280, 'crystallization_mode': 'layer',
    }).solve({'in': feed})
    assert result.outlet_streams['cake'].F == 0
    assert result.outlet_streams['mother_liquor'].F == pytest.approx(feed.F)
    assert result.heat_duty == pytest.approx(0, abs=1e-8)


def test_layer_pfd_ports_dof_and_runtime():
    source = '''
PROCESS: Layer crystallization
ONLINE_LOOKUP: false
COMPONENTS:
    water | Water | type=conventional_with_solid
    ethanol | Ethanol
STREAM Feed : FEED -> C.in
    T = 280 [K]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.9, ethanol:0.1
STREAM Layer : C.layer -> PRODUCT
STREAM Mother : C.mother_liquor -> PRODUCT
UNIT C : Crystallizer
    crystallization_mode = LaYeR
    T = 250 [K]
'''
    pfd = parse_pfd(source)
    assert not validate_pfd(pfd)[0]
    assert analyze_dof(pfd).unit_results[0].status == SpecificationStatus.OK
    result = Simulator(pfd).run()
    assert result.converged, result.errors
    assert result.streams['Layer'].solid_fraction == pytest.approx(1)
    assert result.streams['Mother'].solid_fraction == 0
