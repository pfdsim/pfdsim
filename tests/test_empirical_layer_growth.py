"""Empirical layer-growth expressions, integration, and unit wiring."""

import math

import pytest

from chemical_properties import ChemicalDatabase
from crystallizer_specs import (
    CrystallizerSpecificationError,
    validate_layer_crystallizer_specification,
)
from dof_analyzer import SpecificationStatus, analyze_dof
from empirical_layer_crystallization import (
    EmpiricalLayerDefinitionError,
    empirical_layer_distribution_law_from_mapping,
    empirical_layer_growth_definition_from_parameters,
    empirical_layer_growth_law_from_mapping,
    solve_empirical_layer_growth,
)
from pfd_parser import parse_pfd, validate_pfd
from simulator import Simulator
from thermodynamics_models.base import IdealThermodynamics
from thermodynamics_models.common import ThermodynamicsError
from unit_operations_solids import LayerCrystallizer


@pytest.fixture
def thermo():
    backend = IdealThermodynamics(
        ['water', 'ethanol', 'acetone'], ChemicalDatabase(enable_online=False)
    )
    backend.configure_permanent_solids(
        ['water', 'ethanol', 'acetone'], [],
        conventional_solid_components=['water'],
    )
    return backend


def test_growth_laws_accept_constant_power_and_custom_forms():
    constant = empirical_layer_growth_law_from_mapping({
        'value': 2, 'rate_unit': 'um/s',
    })
    assert constant.evaluate({}) == pytest.approx(0.0072)

    power = empirical_layer_growth_law_from_mapping({
        'model': 'power_law', 'coefficient': 4.68e-7,
        'exponent': 0.89, 'rate_unit': 'm/s',
    })
    assert power.evaluate({'deltaT': 5}) == pytest.approx(
        4.68e-7 * 5**0.89 * 3600
    )

    custom = empirical_layer_growth_law_from_mapping({
        'model': 'custom', 'expression': 'coef*max(Tsat-Twall, 0)**power',
        'rate_unit': 'm/s', 'param_coef': 4.68e-7, 'param_power': 0.89,
    })
    assert custom.evaluate({'Tsat': 280, 'Twall': 275}) == pytest.approx(
        power.evaluate({'deltaT': 5})
    )
    direct_g = empirical_layer_growth_law_from_mapping(
        empirical_layer_growth_definition_from_parameters({
            'g': 2, '__unit__g': 'um/s',
        })
    )
    assert direct_g.evaluate({}) == pytest.approx(0.0072)
    g_alias = empirical_layer_growth_definition_from_parameters({
        'g': 2, '__unit__g': 'um/s',
    })
    assert empirical_layer_growth_law_from_mapping(g_alias).evaluate({}) == pytest.approx(0.0072)
    flattened_constant = empirical_layer_growth_definition_from_parameters({
        'growth_model': 'constant', 'growth_value': 3,
        'growth_unit': 'um/s',
    })
    assert empirical_layer_growth_law_from_mapping(flattened_constant).evaluate({}) == pytest.approx(0.0108)


def test_distribution_expression_uses_impurity_growth_and_composition_context():
    law = empirical_layer_distribution_law_from_mapping({
        'expression': 'alpha*(x_impurity/(1-x_impurity))**beta*exp(c*G_m_s)',
        'param_alpha': 0.99, 'param_beta': 0.28, 'param_c': 59320,
    }, 'ethanol')
    value = law.evaluate({'x_impurity': 0.03, 'G_m_s': 2e-6})
    assert value == pytest.approx(
        0.99 * (0.03 / 0.97)**0.28 * math.exp(59320 * 2e-6)
    )


def test_integrates_multiple_impurities_and_conserves_every_component(thermo):
    growth = empirical_layer_growth_law_from_mapping({
        'value': 1e-5, 'rate_unit': 'm/s',
    })
    distributions = {
        'ethanol': empirical_layer_distribution_law_from_mapping(0.2, 'ethanol'),
        'acetone': empirical_layer_distribution_law_from_mapping({
            'expression': '0.1 + x_impurity + 10*G_m_s',
        }, 'acetone'),
    }
    amounts = {'water': 8, 'ethanol': 1, 'acetone': 1}
    result = solve_empirical_layer_growth(
        thermo, component='water', amounts_kmol=amounts,
        bulk_temperature_K=270, pressure_bar=1, wall_temperature_K=250,
        area_m2=1, growth_time_h=0.01, growth_law=growth,
        distribution_laws=distributions, solid_density_kg_m3=1000,
        profile_points=4,
    )
    assert result.solid_amount_kmol > 0
    assert result.thickness_m == pytest.approx(1e-5 * 3600 * 0.01)
    assert set(result.trapped_component_amounts_kmol) == {'ethanol', 'acetone'}
    for component, initial in amounts.items():
        deposited = (
            result.solid_amount_kmol if component == 'water'
            else result.trapped_component_amounts_kmol.get(component, 0)
        )
        assert result.liquid_component_amounts_kmol[component] + deposited == pytest.approx(initial)
    assert result.profile[-1]['effective_distribution_coefficients']['ethanol'] == pytest.approx(0.2)


@pytest.mark.parametrize('mapping_name', [
    'effective_distributions', 'distribution_coefficients', 'keff', 'k_eff',
])
def test_direct_unit_api_accepts_nested_growth_and_impurity_mappings(thermo, mapping_name):
    feed = thermo.calculate_state(
        270, 1, 10, {'water': 0.8, 'ethanol': 0.1, 'acetone': 0.1},
        phase='liquid',
    )
    result = LayerCrystallizer('C', thermo, {
        'model': 'empirical_layer_growth', 'T': 270, 'T_wall': 250,
        'cooled_area': 1, 'growth_time': 0.01, 'cycle_time': 1,
        'layer_solid_density': 1000,
        'growth': {'value': 2, 'rate_unit': 'um/s'},
        mapping_name: {
            'ethanol': 0.2,
            'acetone': {'expression': 'base + x_impurity', 'param_base': 0.1},
        },
        '__connected_outlet_ports__': ['cake', 'mother_liquor'],
    }).solve({'in': feed})
    assert result.performance['growth_model'] == 'constant'
    assert set(result.performance['effective_distribution_models']) == {
        'ethanol', 'acetone',
    }


def test_invalid_distribution_sum_fails(thermo):
    growth = empirical_layer_growth_law_from_mapping({
        'value': 1e-8, 'rate_unit': 'm/s',
    })
    distributions = {
        'ethanol': empirical_layer_distribution_law_from_mapping(6, 'ethanol'),
        'acetone': empirical_layer_distribution_law_from_mapping(6, 'acetone'),
    }
    with pytest.raises(ThermodynamicsError, match='total impurity mole fraction'):
        solve_empirical_layer_growth(
            thermo, component='water',
            amounts_kmol={'water': 8, 'ethanol': 1, 'acetone': 1},
            bulk_temperature_K=270, pressure_bar=1, wall_temperature_K=250,
            area_m2=1, growth_time_h=0.01, growth_law=growth,
            distribution_laws=distributions, solid_density_kg_m3=1000,
        )


PFD = '''
PROCESS: Empirical layer growth
ONLINE_LOOKUP: false
COMPONENTS:
    water | Water | type=conventional_with_solid
    ethanol | Ethanol
    acetone | Acetone
STREAM Feed : FEED -> C.in
    T = 270 [K]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.8, ethanol:0.1, acetone:0.1
STREAM Layer : C.layer -> PRODUCT
STREAM Mother : C.mother_liquor -> PRODUCT
UNIT C : LayerCrystallizer
    model = empirical_layer_growth
    T = 270 [K]
    T_wall = 250 [K]
    cooled_area = 1 [m2]
    growth_time = 0.01 [h]
    cycle_time = 1 [h]
    layer_solid_density = 1 [g/cm3]
    layer_profile_points = 3
    growth_model = custom
    growth_expression = coef*max(deltaT,0)**power
    growth_param_coef = 4.68e-7
    growth_param_power = 0.89
    growth_rate_unit = m/s
    keff_ethanol_expression = alpha*x_impurity*exp(c*G_m_s)
    keff_ethanol_param_alpha = 0.99
    keff_ethanol_param_c = 59320
    keff_acetone = 0.1
'''


def test_pfd_validates_round_trips_runs_and_reports_empirical_layer():
    pfd = parse_pfd(PFD)
    assert validate_pfd(pfd) == ([], [])
    reparsed = parse_pfd(pfd.to_pfd())
    assert validate_pfd(reparsed) == ([], [])
    assert analyze_dof(pfd).unit_results[0].status == SpecificationStatus.OK
    result = Simulator.from_string(PFD).run()
    assert result.converged, result.errors
    performance = result.units['C'].performance
    assert performance['model'] == 'empirical_finite_rate_layer_growth'
    assert performance['solid_density_kg_m3'] == pytest.approx(1000)
    assert performance['growth_expression'] == 'coef*max(deltaT,0)**power'
    assert set(performance['effective_distribution_expressions']) == {
        'ethanol', 'acetone',
    }
    assert performance['empirically_incorporated_component_flows_kmol_per_h']['ethanol'] > 0
    assert result.streams['Layer'].solid_component_flows['water'] > 0
    for component, flow in result.streams['Feed'].component_flows().items():
        assert sum(
            result.streams[name].component_flows().get(component, 0)
            for name in ('Layer', 'Mother')
        ) == pytest.approx(flow)


def test_explicit_component_selects_layer_when_multiple_solids_are_capable():
    source = PFD.replace(
        '    acetone | Acetone',
        '    acetone | Acetone | type=conventional_with_solid',
    ).replace(
        '    model = empirical_layer_growth',
        '    model = empirical_layer_growth\n    crystallizing_component = water',
    )
    result = Simulator.from_string(source).run()
    assert result.converged, result.errors
    assert result.units['C'].performance['crystallizing_component'] == 'water'
    assert 'acetone' not in result.streams['Layer'].solid_component_flows


def test_direct_dimensioned_growth_rate_runs_with_required_growth_time():
    source = PFD.replace(
        '    growth_model = custom\n'
        '    growth_expression = coef*max(deltaT,0)**power\n'
        '    growth_param_coef = 4.68e-7\n'
        '    growth_param_power = 0.89\n'
        '    growth_rate_unit = m/s\n',
        '    growth_rate = 2 [um/s]\n',
    )
    result = Simulator.from_string(source).run()
    assert result.converged, result.errors
    assert result.units['C'].performance['growth_model'] == 'constant'
    assert result.units['C'].performance['profile'][0][
        'growth_rate_m_s'
    ] == pytest.approx(2e-6)


@pytest.mark.parametrize('change, message', [
    ({'film_thickness': 1e-3}, 'incompatible parameter'),
    ({'crystallization_mode': 'layer'}, 'no longer supported'),
    ({'nucleation_model': 'custom'}, 'does not use nucleation'),
    ({'residence_time': 1}, 'incompatible parameter'),
    ({'layer_relative_tolerance': 1e-6}, 'incompatible parameter'),
])
def test_empirical_spec_rejects_incompatible_inputs(change, message):
    params = {
        'model': 'empirical_layer_growth', 'T': 270, 'T_wall': 250,
        'cooled_area': 1, 'growth_time': 0.1, 'cycle_time': 1,
        'growth_rate': 1e-6, '__unit__growth_rate': 'm/s', **change,
    }
    with pytest.raises(CrystallizerSpecificationError, match=message):
        validate_layer_crystallizer_specification(params)


def test_growth_and_distribution_definition_errors_are_explicit():
    with pytest.raises(EmpiricalLayerDefinitionError, match='explicit rate_unit'):
        empirical_layer_growth_law_from_mapping({'value': 1})
    with pytest.raises(EmpiricalLayerDefinitionError, match='unknown'):
        empirical_layer_growth_law_from_mapping({
            'expression': 'unknown', 'rate_unit': 'm/s',
        })
    with pytest.raises(
        EmpiricalLayerDefinitionError,
        match='Unknown kinetic expression name',
    ):
        empirical_layer_growth_law_from_mapping({
            'expression': 'G + 1', 'rate_unit': 'm/s',
        })
    with pytest.raises(EmpiricalLayerDefinitionError, match='does not accept field'):
        empirical_layer_growth_law_from_mapping({
            'model': 'custom', 'expression': '1', 'coefficient': 2,
            'rate_unit': 'm/s',
        })
    with pytest.raises(EmpiricalLayerDefinitionError, match='duplicate value'):
        empirical_layer_distribution_law_from_mapping({
            'value': 0.1, 'k': 0.2,
        }, 'ethanol')
    with pytest.raises(EmpiricalLayerDefinitionError, match='nonnegative'):
        empirical_layer_distribution_law_from_mapping(-1, 'ethanol')


def test_mechanistic_layer_rejects_empirical_parameters():
    with pytest.raises(CrystallizerSpecificationError, match='empirical layer parameter'):
        validate_layer_crystallizer_specification({
            'model': 'layer_growth', 'T': 270, 'T_wall': 250,
            'cooled_area': 1, 'growth_time': 0.1, 'cycle_time': 1,
            'film_thickness': 1e-3, 'keff_ethanol': 0.2,
        })


def test_flattened_growth_rejects_duplicate_unit_aliases():
    with pytest.raises(EmpiricalLayerDefinitionError, match='duplicate rate-unit'):
        empirical_layer_growth_law_from_mapping(
            empirical_layer_growth_definition_from_parameters({
                'growth_model': 'constant', 'growth_value': 1,
                'growth_unit': 'm/s', 'growth_rate_unit': 'm/h',
            })
        )


def test_uppercase_distribution_parameters_do_not_bypass_model_validation():
    with pytest.raises(CrystallizerSpecificationError, match='empirical distribution'):
        validate_layer_crystallizer_specification({
            'T': 270, 'KEFF_ethanol_PARAM_Alpha': 0.2,
        })


def test_complete_inventory_exhaustion_stops_at_exact_harvest(thermo):
    result = solve_empirical_layer_growth(
        thermo, component='water', amounts_kmol={'water': 1},
        bulk_temperature_K=270, pressure_bar=1, wall_temperature_K=250,
        area_m2=1, growth_time_h=1,
        growth_law=empirical_layer_growth_law_from_mapping({
            'value': 1, 'rate_unit': 'm/h',
        }),
        distribution_laws={}, solid_density_kg_m3=1000,
        profile_points=3,
    )
    assert result.stopped_by_inventory
    assert result.solid_amount_kmol == pytest.approx(1)
    assert result.elapsed_growth_time_h == pytest.approx(thermo.props['water'].MW / 1000)
    assert result.liquid_component_amounts_kmol['water'] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize('expression', ['1', 'sqrt(deltaT)'])
@pytest.mark.parametrize('missing_root', [False, True])
def test_no_growth_without_wall_driving_force(thermo, monkeypatch, expression, missing_root):
    if missing_root:
        monkeypatch.setattr(
            'empirical_layer_crystallization.crystallization_saturation_temperature',
            lambda *args: None,
        )
    result = solve_empirical_layer_growth(
        thermo, component='water', amounts_kmol={'water': 1, 'ethanol': 9},
        bulk_temperature_K=270, pressure_bar=1, wall_temperature_K=260,
        area_m2=1, growth_time_h=0.01,
        growth_law=empirical_layer_growth_law_from_mapping({
            'expression': expression, 'rate_unit': 'm/h',
        }),
        distribution_laws={}, solid_density_kg_m3=1000, profile_points=2,
    )
    assert result.solid_amount_kmol == 0
