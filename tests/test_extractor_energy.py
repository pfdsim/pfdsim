"""Energy conservation when a rigorous extractor produces one liquid phase."""

from unittest.mock import patch

import pytest

from thermodynamics import create_thermodynamics
from unit_operations_base import UnitOperationError, UnitResult
from unit_operations_separation import RigorousLiquidLiquidExtractor


@pytest.fixture
def inlets_and_thermo():
    thermo = create_thermodynamics(['methanol', 'water'], 'NRTL')
    feed = thermo.calculate_state(
        298.15, 1., 100., {'methanol': .5, 'water': .5}, phase='liquid', flash=False,
    )
    solvent = thermo.calculate_state(
        348.15, 1., 100., {'water': 1.}, phase='liquid', flash=False,
    )
    return {'feed': feed, 'solvent': solvent}, thermo


def energy_residual(inlets, result):
    return (
        sum(stream.F * stream.H for stream in result.outlet_streams.values())
        - sum(stream.F * stream.H for stream in inlets.values())
        - result.heat_duty
    )


@pytest.mark.parametrize('algorithm', ('equation_oriented', 'split_sweep'))
@pytest.mark.parametrize('solvent_flow', (100., 250.))
def test_adiabatic_no_lle_conserves_energy(inlets_and_thermo, algorithm, solvent_flow):
    inlets, thermo = inlets_and_thermo
    inlets['solvent'].F = solvent_flow
    unit = RigorousLiquidLiquidExtractor('NO_LLE', thermo, {
        'mode': 'adiabatic', 'solver_algorithm': algorithm,
        'newton_globalization': 'dogleg' if algorithm == 'equation_oriented' else 'line_search',
        'T': 300., 'T_min': 298.15, 'T_max': 348.15,
    })
    result = unit.solve(inlets)
    assert result.warnings == ['No LLE at operating conditions']
    assert result.heat_duty == 0.
    assert abs(energy_residual(inlets, result)) < 1e-3
    assert result.performance['overall_energy_relative_error'] < 1e-8
    raffinate, extract = result.outlet_streams['raffinate'], result.outlet_streams['extract']
    assert extract.F == 0.
    assert extract.T == raffinate.T
    assert 298.15 < raffinate.T < 348.15
    assert raffinate.F == 100. + solvent_flow
    for component in thermo.components:
        incoming = sum(s.F * s.composition.get(component, 0.) for s in inlets.values())
        assert raffinate.F * raffinate.composition.get(component, 0.) == pytest.approx(incoming)


@pytest.mark.parametrize('algorithm', ('equation_oriented', 'split_sweep'))
@pytest.mark.parametrize('temperature', (None, 298.15))
def test_isothermal_no_lle_reports_implied_duty(inlets_and_thermo, algorithm, temperature):
    inlets, thermo = inlets_and_thermo
    params = {'mode': 'isothermal', 'solver_algorithm': algorithm}
    if temperature is not None:
        params['T'] = temperature
    result = RigorousLiquidLiquidExtractor('NO_LLE', thermo, params).solve(inlets)
    expected_temperature = 323.15 if temperature is None else temperature
    assert result.outlet_streams['raffinate'].T == expected_temperature
    assert result.heat_duty < -1000.
    assert abs(energy_residual(inlets, result)) < 1e-6
    assert result.performance['duty_kW'] == result.heat_duty / 3600.


def test_no_lle_calculates_missing_inlet_enthalpy_without_mutating_inputs(inlets_and_thermo):
    inlets, thermo = inlets_and_thermo
    original_enthalpies = {port: stream.H for port, stream in inlets.items()}
    for stream in inlets.values():
        stream.H = None
    result = RigorousLiquidLiquidExtractor('NO_LLE', thermo, {'mode': 'adiabatic'}).solve(inlets)
    incoming = sum(stream.F * original_enthalpies[port] for port, stream in inlets.items())
    outgoing = sum(stream.F * stream.H for stream in result.outlet_streams.values())
    assert abs(outgoing - incoming) < 1e-3
    assert all(stream.H is None for stream in inlets.values())


@pytest.mark.parametrize('bounds', (
    {'T_min': 300., 'T_max': 310.},
    {'adiabatic_T_min': 26.85, 'adiabatic_T_max': 36.85},
))
def test_adiabatic_no_lle_rejects_unreachable_temperature_bounds(inlets_and_thermo, bounds):
    inlets, thermo = inlets_and_thermo
    unit = RigorousLiquidLiquidExtractor('BOUNDS', thermo, dict(bounds, mode='adiabatic'))
    with pytest.raises(UnitOperationError):
        unit.solve(inlets)


def test_adiabatic_no_lle_rechecks_stability_at_solved_temperature(inlets_and_thermo):
    inlets, thermo = inlets_and_thermo
    unit = RigorousLiquidLiquidExtractor('PHASE_CHANGE', thermo, {'mode': 'adiabatic'})
    staged_result = UnitResult(outlet_streams={})
    with patch.object(thermo, 'liquid_liquid_equilibrium', side_effect=[
        (False, {}, {}, 0.),
        (True, {'methanol': .1, 'water': .9}, {'methanol': .8, 'water': .2}, .5),
    ]) as stability, patch.object(unit, '_solve_equation_oriented', return_value=staged_result) as staged:
        result = unit.solve(inlets)
    assert result is staged_result
    assert stability.call_count == 2
    initial_temperature = stability.call_args_list[0].args[1]
    solved_temperature = stability.call_args_list[1].args[1]
    assert abs(solved_temperature - initial_temperature) > 1.
    assert staged.call_args.args[3] == solved_temperature
    assert staged.call_args.args[10] == {'methanol': .1, 'water': .9}


def test_adiabatic_no_lle_rejects_inaccurate_enthalpy_solve(inlets_and_thermo):
    inlets, thermo = inlets_and_thermo
    unit = RigorousLiquidLiquidExtractor('BAD_PH', thermo, {'mode': 'adiabatic'})
    with patch('unit_operations_separation._ThermoStateSolver.state_at_enthalpy',
               return_value=(inlets['feed'], 100.)):
        with pytest.raises(UnitOperationError, match='liquid enthalpy solve residual'):
            unit.solve(inlets)
