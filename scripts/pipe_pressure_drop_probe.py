#!/usr/bin/env python3
"""Probe integrated single-phase pressure drop for representative pipes."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from axial_solver import AxialSolverOptions, integrate_axial
from thermodynamics import create_thermodynamics
from transport_correlations import single_phase_flow


COMMERCIAL_STEEL_ROUGHNESS_M = 4.5e-5


def pipe_drop(component, method, temperature_k, pressure_bar, mass_flow_kg_h,
              diameter_m, length_m, roughness_m, phase):
    thermo = create_thermodynamics([component], method)
    composition = {component: 1.0}
    molecular_weight = thermo.mixture_MW(composition)
    molar_flow = mass_flow_kg_h / molecular_weight
    mass_flow = mass_flow_kg_h / 3600.0
    area = math.pi * diameter_m**2 / 4.0

    def state_at_tp(temperature, pressure):
        state = thermo.calculate_state(
            temperature, pressure, molar_flow, composition,
            phase=phase, flash=False, include=('H',),
        )
        densities = thermo.transport_mixture_density(
            composition, temperature, pressure, state.vapor_fraction,
            state.x, state.y,
        )
        density = densities.vapor if phase == 'vapor' else densities.liquid
        velocity = mass_flow / (density * area)
        return state, density, velocity

    inlet_state, inlet_density, inlet_velocity = state_at_tp(
        temperature_k, pressure_bar
    )
    stagnation_enthalpy = (
        inlet_state.H * 1000.0 / molecular_weight
        + inlet_velocity**2 / 2.0
    )

    def local_state(pressure):
        def energy_residual(temperature):
            state, density, velocity = state_at_tp(temperature, pressure)
            return (
                state.H * 1000.0 / molecular_weight
                + velocity**2 / 2.0
                - stagnation_enthalpy
            )

        temperature = brentq(
            energy_residual,
            max(thermo._tmin + 1e-4, temperature_k - 100.0)
            if method in {'STEAM', 'IF97'} else max(80.0, temperature_k - 150.0),
            temperature_k + 150.0,
        )
        state, density, velocity = state_at_tp(temperature, pressure)
        viscosities = thermo.transport_mixture_viscosity(
            composition, temperature, pressure, state.vapor_fraction,
            state.x, state.y,
        )
        viscosity = viscosities.vapor if phase == 'vapor' else viscosities.liquid
        flow = single_phase_flow(
            mass_flow, density, viscosity, diameter_m,
            roughness_m=roughness_m, elevation_gradient=0.0,
            friction_model='churchill',
        )
        return temperature, density, viscosity, velocity, flow

    def pressure_rhs(_position, values):
        pressure = values[0]
        _, density, _, velocity, flow = local_state(pressure)
        perturbation = max(1e-4, pressure * 1e-4)
        velocity_high = local_state(pressure + perturbation)[3]
        velocity_low = local_state(pressure - perturbation)[3]
        dv_dp_pa = (velocity_high - velocity_low) / (2.0 * perturbation * 1e5)
        acceleration_denominator = 1.0 + density * velocity * dv_dp_pa
        return (-flow.pressure_gradient.friction
                / (1e5 * acceleration_denominator),)

    profile_positions = tuple(
        length_m * fraction for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    solution = integrate_axial(
        pressure_rhs, length_m, (pressure_bar,),
        options=AxialSolverOptions(
            relative_tolerance=1e-8,
            absolute_tolerance=1e-10,
            maximum_step_m=2.0,
        ),
        evaluation_positions_m=profile_positions,
    )
    outlet_pressure = solution.outlet_values[0]
    inlet = local_state(pressure_bar)
    outlet = local_state(outlet_pressure)
    return {
        'method': method,
        'inlet_pressure_bar': pressure_bar,
        'outlet_pressure_bar': outlet_pressure,
        'drop_bar': pressure_bar - outlet_pressure,
        'inlet': inlet,
        'outlet': outlet,
        'profile': [
            (position, pressure, local_state(pressure))
            for position, pressure in zip(solution.position_m, solution.values[0])
        ],
    }


for case in (
    (
        'H2O', 'STEAM', 353.15, 10.0, 5000.0, 0.04, 100.0,
        COMMERCIAL_STEEL_ROUGHNESS_M, 'liquid',
    ),
    (
        'CH4', 'PR', 333.15, 20.0, 5000.0, 0.10, 100.0,
        COMMERCIAL_STEEL_ROUGHNESS_M, 'vapor',
    ),
):
    result = pipe_drop(*case)
    print(result['method'], result['drop_bar'], result['outlet_pressure_bar'])
    for label in ('inlet', 'outlet'):
        temperature, density, viscosity, velocity, flow = result[label]
        print(
            label,
            'T', temperature,
            'rho', density,
            'mu', viscosity,
            'v', velocity,
            'Re', flow.reynolds_number,
            'f', flow.darcy_friction_factor,
        )
    for position, pressure, local in result['profile']:
        temperature, density, viscosity, velocity, flow = local
        print(
            'profile',
            'x', position,
            'P', pressure,
            'T', temperature,
            'Re', flow.reynolds_number,
            'f', flow.darcy_friction_factor,
        )
