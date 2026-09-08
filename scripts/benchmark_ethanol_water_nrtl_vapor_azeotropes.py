#!/usr/bin/env python3
"""Compare NRTL ethanol/water azeotropes across vapor-phase models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from scipy.optimize import brentq


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from thermodynamics_models.factory import create_thermodynamics  # noqa: E402


COMPONENTS = ('ethanol', 'water')
PRESSURES = (
    ('1 atm', 1.01325),
    ('2 bar', 2.0),
    ('5 bar', 5.0),
    ('10 bar', 10.0),
)
METHODS = (
    ('ideal', 'NRTL'),
    ('RK', 'NRTL-RK'),
    ('PR', 'NRTL-PR'),
    ('BV/Tsonopoulos', 'NRTL-BV'),
    ('HOC', 'NRTL-HOC'),
)


def solve_azeotrope(thermo, pressure_bar, initial):
    """Solve the bubble branch for an interior ``K_ethanol/K_water = 1``."""
    states = {}

    def state(ethanol_fraction):
        key = float(ethanol_fraction)
        cached = states.get(key)
        if cached is not None:
            return cached
        liquid = {
            'ethanol': ethanol_fraction,
            'water': 1.0 - ethanol_fraction,
        }
        temperature = thermo.bubble_point_T(
            liquid,
            pressure_bar,
            initial[0],
        )
        K = thermo.K_values(temperature, pressure_bar, liquid)
        result = (temperature, liquid, K)
        states[key] = result
        return result

    def residual(ethanol_fraction):
        _temperature, _liquid, K = state(ethanol_fraction)
        return math.log(
            max(float(K['ethanol']), 1.0e-300)
            / max(float(K['water']), 1.0e-300)
        )

    lower = 0.40
    upper = 1.0 - 1.0e-8
    lower_residual = residual(lower)
    upper_residual = residual(upper)
    if lower_residual * upper_residual >= 0.0:
        samples = sorted({
            *(0.02 + index * 0.96 / 48.0 for index in range(49)),
            0.99,
            0.999,
            0.9999,
            0.99999,
            0.999999,
            upper,
        })
        brackets = []
        previous_x = samples[0]
        previous_residual = residual(previous_x)
        for current_x in samples[1:]:
            current_residual = residual(current_x)
            if previous_residual * current_residual < 0.0:
                brackets.append((previous_x, current_x))
            previous_x = current_x
            previous_residual = current_residual
        if not brackets:
            closest = min(
                samples,
                key=lambda ethanol_fraction: abs(residual(ethanol_fraction)),
            )
            return {
                'status': 'no interior azeotrope',
                'temperature_K': None,
                'temperature_C': None,
                'x_ethanol': None,
                'y_ethanol': None,
                'max_abs_log_K': None,
                'bubble_sum_residual': None,
                'y_minus_x_ethanol': None,
                'function_evaluations': len(states),
                'closest_sample_x_ethanol': closest,
                'closest_sample_log_K_ratio': residual(closest),
            }
        lower, upper = min(
            brackets,
            key=lambda bracket: abs(0.5 * sum(bracket) - initial[1]),
        )
    liquid_ethanol = brentq(
        residual,
        lower,
        upper,
        xtol=1.0e-13,
        rtol=1.0e-13,
        maxiter=100,
    )
    temperature, liquid, K = state(liquid_ethanol)
    unnormalized_vapor = {
        component: liquid[component] * K[component]
        for component in COMPONENTS
    }
    bubble_sum = sum(unnormalized_vapor.values())
    vapor = {
        component: unnormalized_vapor[component] / bubble_sum
        for component in COMPONENTS
    }
    log_K_residual = max(
        abs(math.log(max(float(K[component]), 1.0e-300)))
        for component in COMPONENTS
    )
    if (
        log_K_residual > 1.0e-8
        or abs(bubble_sum - 1.0) > 1.0e-8
        or abs(vapor['ethanol'] - liquid_ethanol) > 1.0e-8
    ):
        raise RuntimeError(
            f"Azeotrope solve failed at {pressure_bar:g} bar: "
            f"max|ln K|={log_K_residual:.3e}, "
            f"bubble residual={bubble_sum - 1.0:.3e}, "
            f"y-x={vapor['ethanol'] - liquid_ethanol:.3e}"
        )
    return {
        'status': 'azeotrope',
        'temperature_K': temperature,
        'temperature_C': temperature - 273.15,
        'x_ethanol': liquid_ethanol,
        'y_ethanol': vapor['ethanol'],
        'max_abs_log_K': log_K_residual,
        'bubble_sum_residual': bubble_sum - 1.0,
        'y_minus_x_ethanol': vapor['ethanol'] - liquid_ethanol,
        'function_evaluations': len(states),
        'closest_sample_x_ethanol': None,
        'closest_sample_log_K_ratio': None,
    }


def benchmark():
    rows = []
    initial_by_pressure = {
        pressure_bar: [351.5 + 7.5 * pressure_bar, 0.88]
        for _label, pressure_bar in PRESSURES
    }
    for vapor_model, method in METHODS:
        thermo = create_thermodynamics(list(COMPONENTS), method)
        previous = None
        for pressure_label, pressure_bar in PRESSURES:
            initial = previous or initial_by_pressure[pressure_bar]
            solved = solve_azeotrope(thermo, pressure_bar, initial)
            if solved['status'] == 'azeotrope':
                previous = [solved['temperature_K'], solved['x_ethanol']]
            rows.append({
                'pressure': pressure_label,
                'pressure_bar': pressure_bar,
                'vapor_model': vapor_model,
                'thermo_method': method,
                **solved,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    rows = benchmark()
    writer = csv.DictWriter(sys.stdout, fieldnames=tuple(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
