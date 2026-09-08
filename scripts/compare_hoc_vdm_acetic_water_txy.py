#!/usr/bin/env python3
"""Compare UNIQUAC-HOC and UNIQUAC-VDM acetic-acid/water Txy at 1 atm."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from thermodynamics_models.factory import create_thermodynamics  # noqa: E402
from thermodynamics_models.common import R  # noqa: E402
from vapor_dimerization import get_dimerization_params  # noqa: E402


COMPONENTS = ('acetic acid', 'water')
PRESSURE_ATM_BAR = 1.01325

# PFDSim-supplied 1 atm Txy table. Columns are temperature [deg C], liquid
# water mole percent, and vapor water mole percent.
REFERENCE_TXY_WATER = (
    (116.5, 2.2, 5.8),
    (114.6, 5.4, 12.3),
    (113.4, 8.6, 16.8),
    (113.5, 9.9, 18.3),
    (113.1, 10.1, 18.8),
    (110.6, 18.9, 29.8),
    (107.8, 30.3, 43.3),
    (106.1, 41.3, 54.5),
    (104.4, 52.2, 64.9),
    (103.1, 62.4, 73.5),
    (102.3, 69.6, 79.2),
    (101.6, 77.8, 85.1),
    (100.8, 87.6, 91.4),
    (100.5, 92.3, 94.4),
    (100.4, 94.5, 96.0),
    (100.1, 98.5, 98.9),
)


def _vapor_composition(thermo, liquid, temperature, pressure):
    K = thermo.K_values(temperature, pressure, liquid)
    unnormalized = {
        component: liquid[component] * K[component]
        for component in COMPONENTS
    }
    total = sum(unnormalized.values())
    return {
        component: unnormalized[component] / total
        for component in COMPONENTS
    }


def _physical_molecule_factor(thermo, temperature, pressure, vapor):
    backend = getattr(thermo, 'vapor_eos', None)
    association_state = getattr(backend, 'association_state', None)
    if callable(association_state):
        return float(
            association_state(temperature, pressure, vapor)[
                'physical_moles_per_nominal'
            ]
        )
    vdm_state = getattr(thermo, '_vdm_vapor_association_state', None)
    if callable(vdm_state):
        state = vdm_state(temperature, pressure, vapor)
        return float(thermo._vdm_physical_moles_per_nominal(state))
    return 1.0


def calculate_curve(method, *, correlation=None, points=21, pressure=PRESSURE_ATM_BAR):
    options = {'correlation': correlation} if correlation else None
    thermo = create_thermodynamics(
        list(COMPONENTS),
        method,
        thermo_options=options,
    )
    rows = []
    guess = 373.15
    for index in range(points):
        acid_fraction = index / (points - 1)
        liquid = {
            'acetic acid': acid_fraction,
            'water': 1.0 - acid_fraction,
        }
        temperature = thermo.bubble_point_T(liquid, pressure, guess)
        guess = temperature
        vapor = _vapor_composition(thermo, liquid, temperature, pressure)
        backend = getattr(thermo, 'vapor_eos', None)
        compressibility = (
            float(backend.compressibility_factor(temperature, pressure, vapor))
            if backend is not None
            else _physical_molecule_factor(thermo, temperature, pressure, vapor)
        )
        rows.append({
            'method': (
                f'{method} | correlation={correlation}'
                if correlation else method
            ),
            'x_acetic_acid': acid_fraction,
            'y_acetic_acid': vapor['acetic acid'],
            'temperature_K': temperature,
            'temperature_C': temperature - 273.15,
            'compressibility_factor': compressibility,
            'physical_moles_per_nominal': _physical_molecule_factor(
                thermo,
                temperature,
                pressure,
                vapor,
            ),
        })
    return rows


def benchmark_reference(
    method,
    *,
    correlation=None,
    pressure=PRESSURE_ATM_BAR,
):
    """Return point residuals and RMSE against the supplied 1 atm table."""
    options = {'correlation': correlation} if correlation else None
    thermo = create_thermodynamics(
        list(COMPONENTS),
        method,
        thermo_options=options,
    )
    rows = []
    temperature_squared_error = 0.0
    vapor_squared_error = 0.0
    guess = 390.0
    for measured_temperature_C, liquid_water_percent, vapor_water_percent in (
        REFERENCE_TXY_WATER
    ):
        liquid_water = liquid_water_percent / 100.0
        liquid = {
            'acetic acid': 1.0 - liquid_water,
            'water': liquid_water,
        }
        calculated_temperature = thermo.bubble_point_T(liquid, pressure, guess)
        guess = calculated_temperature
        vapor = _vapor_composition(
            thermo,
            liquid,
            calculated_temperature,
            pressure,
        )
        calculated_temperature_C = calculated_temperature - 273.15
        measured_vapor_water = vapor_water_percent / 100.0
        temperature_residual = calculated_temperature_C - measured_temperature_C
        vapor_residual = vapor['water'] - measured_vapor_water
        temperature_squared_error += temperature_residual**2
        vapor_squared_error += vapor_residual**2
        rows.append({
            'x_water': liquid_water,
            'temperature_experimental_C': measured_temperature_C,
            'temperature_calculated_C': calculated_temperature_C,
            'temperature_residual_C': temperature_residual,
            'y_water_experimental': measured_vapor_water,
            'y_water_calculated': vapor['water'],
            'y_water_residual': vapor_residual,
        })
    count = len(rows)
    return {
        'method': (
            f'{method} | correlation={correlation}'
            if correlation else method
        ),
        'pressure_bar': pressure,
        'points': count,
        'temperature_rmse_C': math.sqrt(temperature_squared_error / count),
        'vapor_y_rmse_mole_fraction': math.sqrt(vapor_squared_error / count),
        'vapor_y_rmse_mole_percent_points': (
            100.0 * math.sqrt(vapor_squared_error / count)
        ),
        'rows': rows,
    }


def hoc_implied_association_thermodynamics():
    """Return local van't Hoff H/S implied by HOC over the reference range."""
    thermo = create_thermodynamics(
        list(COMPONENTS),
        'UNIQUAC-BV',
        thermo_options={'correlation': 'HOC'},
    )
    provider = thermo.vapor_eos.provider
    acid_index = provider.components.index('acetic acid')
    temperatures_C = sorted({row[0] for row in REFERENCE_TXY_WATER})
    rows = []
    for temperature_C in temperatures_C:
        temperature = temperature_C + 273.15
        equilibrium = provider.association_constant_matrix(temperature)
        derivative = provider.association_constant_matrix(temperature, order=1)
        Kp = float(equilibrium[acid_index][acid_index])
        dKp_dT = float(derivative[acid_index][acid_index])
        delta_H = R * temperature**2 * dKp_dT / Kp
        # Numerically Kp [bar^-1] times the 1 bar standard state is the
        # dimensionless equilibrium constant used inside the logarithm.
        delta_S = R * math.log(Kp) + delta_H / temperature
        rows.append({
            'temperature_C': temperature_C,
            'Kp_bar_inverse': Kp,
            'delta_H_implied_kJ_per_mol': delta_H / 1000.0,
            'delta_S_implied_J_per_mol_K': delta_S,
        })
    inverse_temperatures = [
        1.0 / (row['temperature_C'] + 273.15)
        for row in rows
    ]
    logarithms = [math.log(row['Kp_bar_inverse']) for row in rows]
    mean_inverse_temperature = sum(inverse_temperatures) / len(rows)
    mean_logarithm = sum(logarithms) / len(rows)
    slope = sum(
        (inverse_temperature - mean_inverse_temperature)
        * (logarithm - mean_logarithm)
        for inverse_temperature, logarithm in zip(
            inverse_temperatures,
            logarithms,
        )
    ) / sum(
        (inverse_temperature - mean_inverse_temperature) ** 2
        for inverse_temperature in inverse_temperatures
    )
    intercept = mean_logarithm - slope * mean_inverse_temperature
    return {
        'temperature_range_C': [
            min(row['temperature_C'] for row in rows),
            max(row['temperature_C'] for row in rows),
        ],
        'local_delta_H_range_kJ_per_mol': [
            min(row['delta_H_implied_kJ_per_mol'] for row in rows),
            max(row['delta_H_implied_kJ_per_mol'] for row in rows),
        ],
        'local_delta_S_range_J_per_mol_K': [
            min(row['delta_S_implied_J_per_mol_K'] for row in rows),
            max(row['delta_S_implied_J_per_mol_K'] for row in rows),
        ],
        'vanthoff_fit_delta_H_kJ_per_mol': -R * slope / 1000.0,
        'vanthoff_fit_delta_S_J_per_mol_K': R * intercept,
        'rows': rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points', type=int, default=21)
    parser.add_argument('--pressure-bar', type=float, default=PRESSURE_ATM_BAR)
    parser.add_argument('--json', type=Path)
    parser.add_argument(
        '--reference-summary',
        action='store_true',
        help='benchmark HOC against the embedded PFDSim-supplied 1 atm table',
    )
    args = parser.parse_args()
    if args.points < 2:
        parser.error('--points must be at least 2')
    if args.pressure_bar <= 0.0:
        parser.error('--pressure-bar must be positive')

    if args.reference_summary:
        hoc = benchmark_reference(
            'UNIQUAC-BV',
            correlation='HOC',
            pressure=args.pressure_bar,
        )
        vdm = benchmark_reference(
            'UNIQUAC-VDM',
            pressure=args.pressure_bar,
        )
        result = {
            'HOC': hoc,
            'VDM': vdm,
            'HOC_implied_association_thermodynamics': (
                hoc_implied_association_thermodynamics()
            ),
            'VDM_acetic_acid_parameters': get_dimerization_params(
                'acetic acid'
            ),
        }
        print(json.dumps(result, indent=2))
        if args.json is not None:
            args.json.write_text(
                json.dumps(result, indent=2) + '\n',
                encoding='utf-8',
            )
        return

    rows = calculate_curve(
        'UNIQUAC-BV',
        correlation='HOC',
        points=args.points,
        pressure=args.pressure_bar,
    )
    rows.extend(calculate_curve(
        'UNIQUAC-VDM',
        points=args.points,
        pressure=args.pressure_bar,
    ))

    writer = csv.DictWriter(sys.stdout, fieldnames=tuple(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
