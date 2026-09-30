"""Joint Moreau/Ovejero VLE and excess-enthalpy activity-model fits."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermodynamics import create_thermodynamics
from thermodynamics_models.interaction_estimation import (
    _nrtl_ln_gamma,
    _uniquac_ln_gamma,
)

R = 8.31446261815324
TREF = 313.15
COMPONENTS = ("1-pentanol", "cyclohexane")
R_VALUES = (4.1287, 4.0464)
Q_VALUES = (3.592, 3.24)
PSAT_KPA = (0.908, 24.630)
VL_L_MOL = (0.110, 0.111)
B_L_MOL = ((-3.001, -1.508), (-1.508, -1.925))

MOREAU_PX = [
    (0.0000, 24.569),
    (0.0491, 24.329),
    (0.1037, 24.071),
    (0.1492, 23.788),
    (0.1989, 23.485),
    (0.2492, 23.139),
    (0.2993, 22.764),
    (0.3493, 22.330),
    (0.3991, 21.846),
    (0.4010, 21.832),
    (0.4494, 21.276),
    (0.4511, 21.267),
    (0.4993, 20.614),
    (0.5012, 20.606),
    (0.5492, 19.826),
    (0.5514, 19.931),
    (0.5993, 18.880),
    (0.6016, 18.856),
    (0.6517, 17.716),
    (0.7025, 16.322),
    (0.7517, 14.712),
    (0.8019, 12.754),
    (0.8516, 10.462),
    (0.9015, 7.746),
    (0.9512, 4.560),
    (1.0000, 0.900),
]
HE = {
    298.15: [
        (0.0500, 333.3),
        (0.1001, 440.1),
        (0.1501, 520.5),
        (0.2022, 581.0),
        (0.2522, 622.0),
        (0.3022, 648.7),
        (0.3523, 660.9),
        (0.4023, 658.4),
        (0.4523, 641.4),
        (0.5022, 609.2),
        (0.5522, 573.5),
        (0.6022, 527.4),
        (0.6522, 471.1),
        (0.7022, 411.0),
        (0.7522, 347.2),
        (0.8021, 271.9),
        (0.8501, 206.1),
        (0.9000, 138.1),
        (0.9500, 67.7),
    ],
    313.15: [
        (0.0500, 449.0),
        (0.1000, 581.9),
        (0.1501, 673.3),
        (0.2021, 739.3),
        (0.2522, 785.3),
        (0.3022, 809.4),
        (0.3521, 818.7),
        (0.4022, 809.9),
        (0.4522, 783.5),
        (0.5022, 745.9),
        (0.5522, 709.5),
        (0.6022, 634.2),
        (0.6519, 583.1),
        (0.7019, 494.1),
        (0.7521, 412.1),
        (0.8021, 334.5),
        (0.8501, 250.2),
        (0.9000, 159.6),
        (0.9500, 79.8),
    ],
}
OVEJERO = [
    (0.0059, 0.0057, 353.8, 9.538, 1.003),
    (0.0095, 0.0083, 353.9, 8.535, 1.002),
    (0.0106, 0.0090, 353.9, 8.323, 1.002),
    (0.0157, 0.0125, 354.0, 7.780, 1.001),
    (0.0222, 0.0155, 354.1, 6.804, 1.002),
    (0.0272, 0.0184, 354.3, 6.493, 0.998),
    (0.0335, 0.0204, 354.4, 5.820, 1.000),
    (0.0436, 0.0238, 354.3, 5.254, 1.010),
    (0.0457, 0.0239, 354.5, 4.983, 1.007),
    (0.0496, 0.0250, 354.4, 4.824, 1.013),
    (0.0554, 0.0268, 354.6, 4.590, 1.011),
    (0.0654, 0.0284, 354.7, 4.099, 1.018),
    (0.0680, 0.0292, 354.7, 4.055, 1.020),
]


def unpack(parameters, with_heat_capacity):
    if with_heat_capacity:
        a12, b12, e12, a21, b21, e21 = parameters
    else:
        a12, b12, a21, b21 = parameters
        e12 = e21 = 0.0
    return a12, b12, e12, a21, b21, e21


def interaction_values(model, temperature, parameters, with_heat_capacity):
    a12, b12, e12, a21, b21, e21 = unpack(parameters, with_heat_capacity)
    anchored = (TREF - temperature) / temperature + math.log(temperature / TREF)
    first = a12 + b12 / temperature + e12 * anchored
    second = a21 + b21 / temperature + e21 * anchored
    if model == "UNIQUAC":
        return math.exp(first), math.exp(second)
    return first, second


def ln_gamma(model, x1, temperature, parameters, alpha, with_heat_capacity):
    first, second = interaction_values(
        model, temperature, parameters, with_heat_capacity
    )
    if model == "NRTL":
        return _nrtl_ln_gamma(x1, first, second, alpha)
    return _uniquac_ln_gamma(x1, R_VALUES, Q_VALUES, first, second)


def reduced_ge(model, x1, temperature, parameters, alpha, with_heat_capacity):
    ln1, ln2 = ln_gamma(model, x1, temperature, parameters, alpha, with_heat_capacity)
    return x1 * ln1 + (1.0 - x1) * ln2


def excess_enthalpy(model, x1, temperature, parameters, alpha, with_heat_capacity):
    step = 0.01
    derivative = (
        reduced_ge(model, x1, temperature + step, parameters, alpha, with_heat_capacity)
        - reduced_ge(
            model, x1, temperature - step, parameters, alpha, with_heat_capacity
        )
    ) / (2.0 * step)
    return -R * temperature * temperature * derivative


def source_bubble(model, x1, parameters, alpha, with_heat_capacity):
    ln1, ln2 = ln_gamma(model, x1, TREF, parameters, alpha, with_heat_capacity)
    gamma = (math.exp(ln1), math.exp(ln2))
    x = (x1, 1.0 - x1)
    pressure = sum(x[i] * gamma[i] * PSAT_KPA[i] for i in range(2))
    y = [x[i] * gamma[i] * PSAT_KPA[i] / pressure for i in range(2)]
    for _ in range(100):
        bmix = sum(y[i] * y[j] * B_L_MOL[i][j] for i in range(2) for j in range(2))
        phi = []
        for i in range(2):
            partial = 2.0 * sum(y[j] * B_L_MOL[i][j] for j in range(2)) - bmix
            phi.append(math.exp(partial * pressure / (R * TREF)))
        rhs = []
        for i in range(2):
            phi_sat = math.exp(B_L_MOL[i][i] * PSAT_KPA[i] / (R * TREF))
            poynting = math.exp(VL_L_MOL[i] * (pressure - PSAT_KPA[i]) / (R * TREF))
            rhs.append(x[i] * gamma[i] * PSAT_KPA[i] * phi_sat * poynting / phi[i])
        new_pressure = sum(rhs)
        new_y = [value / new_pressure for value in rhs]
        if (
            abs(new_pressure - pressure) < 1e-12
            and max(abs(new_y[i] - y[i]) for i in range(2)) < 1e-12
        ):
            pressure, y = new_pressure, new_y
            break
        pressure, y = new_pressure, new_y
    return pressure, y[0]


def objective(model, alpha, with_heat_capacity, parameters):
    residuals = []
    for x1, observed in MOREAU_PX[1:-1]:
        calculated, _ = source_bubble(model, x1, parameters, alpha, with_heat_capacity)
        residuals.append((calculated - observed) / 0.005)
    for temperature, rows in HE.items():
        for x1, observed in rows:
            calculated = excess_enthalpy(
                model, x1, temperature, parameters, alpha, with_heat_capacity
            )
            residuals.append((calculated - observed) / (0.005 * observed))
    return np.asarray(residuals)


def initial_guesses(model, with_heat_capacity):
    if model == "NRTL":
        bases = [
            [-0.84, 465.0, -1.46, 1143.0],
            [0.0, 202.3, 0.0, 686.0],
            [0.6461, 0.0, 2.1906, 0.0],
        ]
    else:
        bases = [
            [-0.10, 106.0, 0.35, -353.0],
            [0.0, 75.0, 0.0, -244.0],
            [math.log(1.2705), 0.0, math.log(0.4583), 0.0],
        ]
    if with_heat_capacity:
        bases = [[v[0], v[1], 0.0, v[2], v[3], 0.0] for v in bases]
    return bases


def fit(model, alpha, with_heat_capacity):
    size = 6 if with_heat_capacity else 4
    lower = (
        np.array([-10.0, -10000.0, -500.0, -10.0, -10000.0, -500.0])
        if size == 6
        else np.array([-10.0, -10000.0, -10.0, -10000.0])
    )
    upper = -lower
    results = []
    for initial in initial_guesses(model, with_heat_capacity):
        results.append(
            least_squares(
                lambda p: objective(model, alpha, with_heat_capacity, p),
                initial,
                bounds=(lower, upper),
                max_nfev=5000,
                xtol=1e-11,
                ftol=1e-11,
                gtol=1e-11,
            )
        )
    return min(results, key=lambda result: 2.0 * result.cost)


def metrics(values):
    values = np.asarray(values)
    return {
        "ME": float(values.mean()),
        "MAE": float(abs(values).mean()),
        "RMSE": float(np.sqrt(np.mean(values * values))),
        "MaxAE": float(abs(values).max()),
    }


def override(model, alpha, parameters, with_heat_capacity):
    a12, b12, e12, a21, b21, e21 = unpack(parameters, with_heat_capacity)
    if model == "NRTL":
        return [
            {
                "model": model,
                "component1": COMPONENTS[0],
                "component2": COMPONENTS[1],
                "alpha12": alpha,
                "tau12_c": a12,
                "tau12_d": b12,
                "tau12_e": e12,
                "tau21_c": a21,
                "tau21_d": b21,
                "tau21_e": e21,
                "tau_tref": TREF,
            }
        ]
    return [
        {
            "model": model,
            "component1": COMPONENTS[0],
            "component2": COMPONENTS[1],
            "tau12_a": a12,
            "tau12_b": b12,
            "tau12_c": e12,
            "tau21_a": a21,
            "tau21_b": b21,
            "tau21_c": e21,
            "tau_tref": TREF,
            "use_q_prime": False,
        }
    ]


def score(model, alpha, parameters, with_heat_capacity):
    p_errors = []
    for x1, observed in MOREAU_PX[1:-1]:
        calculated, _ = source_bubble(model, x1, parameters, alpha, with_heat_capacity)
        p_errors.append(calculated - observed)
    he = {}
    for temperature, rows in HE.items():
        he[str(temperature)] = metrics(
            [
                excess_enthalpy(
                    model, x1, temperature, parameters, alpha, with_heat_capacity
                )
                - observed
                for x1, observed in rows
            ]
        )
    thermo = create_thermodynamics(
        list(COMPONENTS),
        f"{model}-HOC",
        interaction_overrides=override(model, alpha, parameters, with_heat_capacity),
    )
    te, ye, pe = [], [], []
    for x1, y1, observed_t, _, _ in OVEJERO:
        composition = {COMPONENTS[0]: x1, COMPONENTS[1]: 1.0 - x1}
        calculated_t = thermo.bubble_point_T(composition, 1.013, observed_t)
        kval = thermo.K_values(calculated_t, 1.013, composition)
        closure = sum(composition[c] * kval[c] for c in composition)
        calculated_y = x1 * kval[COMPONENTS[0]] / closure
        calculated_p = thermo.bubble_point_P(composition, observed_t)
        te.append(calculated_t - observed_t)
        ye.append(calculated_y - y1)
        pe.append(100.0 * (calculated_p / 1.013 - 1.0))
    values_313 = interaction_values(model, TREF, parameters, with_heat_capacity)
    values_354 = interaction_values(model, 354.3, parameters, with_heat_capacity)
    return {
        "Moreau_pressure_kPa": metrics(p_errors),
        "Moreau_HE_J_mol": he,
        "Ovejero": {"T_K": metrics(te), "y1": metrics(ye), "P_percent": metrics(pe)},
        "interaction_values_313_15K": values_313,
        "interaction_values_354_3K": values_354,
    }


def candidate(model, alpha, with_heat_capacity):
    result = fit(model, alpha, with_heat_capacity)
    return {
        "model": model,
        "alpha": alpha,
        "form": "ABH" if with_heat_capacity else "AB",
        "success": bool(result.success),
        "chi_square": float(2.0 * result.cost),
        "parameters": [float(value) for value in result.x],
        "score": score(model, alpha, result.x, with_heat_capacity),
    }


def runtime_interaction(selected):
    model = selected["model"]
    parameters = selected["parameters"]
    a12, b12, e12, a21, b21, e21 = unpack(parameters, True)
    record = {
        "model": model,
        "cas1": "71-41-0",
        "cas2": "110-82-7",
        "component1": COMPONENTS[0],
        "component2": COMPONENTS[1],
        "Tmin_K": 298.15,
        "Tmax_K": 354.7,
        "extrapolation": "inverse_square_cubic",
        "source": "Moreau et al. 2012 and Ovejero et al. 2007",
        "source_dois": [
            "10.1016/j.fluid.2012.01.007",
            "10.1021/je700285j",
        ],
        "source_file": "data/source/activity_fitting/nonwater_binary_parameters_v2.json",
        "fit_status": ("recommended_joint_vle_excess_enthalpy_regularized_interaction"),
        "comment": (
            "1-pentanol + cyclohexane joint Moreau VLE/HE fit validated "
            "against held-out Ovejero dilute VLE; tangent M/T^2+N/T^3 "
            "regularization outside 298.15-354.7 K"
        ),
        "tau_tref": TREF,
        "fit_evidence": selected["score"],
        "regularization": {
            "inside": "declared complete temperature law",
            "outside": "value-and-first-derivative-matched M/T^2+N/T^3",
            "continuity": "parameter, activity coefficient, GE, and HE are continuous",
        },
    }
    if model == "NRTL":
        record.update(
            {
                "alpha12": float(selected["alpha"]),
                "tau12_c": a12,
                "tau12_d": b12,
                "tau12_e": e12,
                "tau21_c": a21,
                "tau21_d": b21,
                "tau21_e": e21,
            }
        )
    else:
        record.update(
            {
                "model_variant": "standard_uniquac",
                "use_q_prime": False,
                "tau12_a": a12,
                "tau12_b": b12,
                "tau12_c": e12,
                "tau21_a": a21,
                "tau21_b": b21,
                "tau21_c": e21,
            }
        )
    return record


def regularized_stress_test(interaction):
    model = interaction["model"]
    thermo = create_thermodynamics(
        list(COMPONENTS),
        model,
        interaction_overrides=[interaction],
    )
    result = {}
    for temperature in (354.7, 553.6, 1000.0):
        dilute_one = thermo.activity_coefficients(
            temperature, {COMPONENTS[0]: 1.0e-10, COMPONENTS[1]: 1.0 - 1.0e-10}
        )[COMPONENTS[0]]
        dilute_two = thermo.activity_coefficients(
            temperature, {COMPONENTS[0]: 1.0 - 1.0e-10, COMPONENTS[1]: 1.0e-10}
        )[COMPONENTS[1]]
        result[str(temperature)] = {
            "gamma1_infinite_dilution": float(dilute_one),
            "gamma2_infinite_dilution": float(dilute_two),
            "equimolar_HE_J_mol": float(
                thermo.excess_enthalpy(
                    {COMPONENTS[0]: 0.5, COMPONENTS[1]: 0.5}, temperature
                )
            ),
        }
    return result


def _stable_numeric_payload(value):
    """Remove optimizer/backend last-bit noise from committed provenance."""
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, (list, tuple)):
        return [_stable_numeric_payload(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _stable_numeric_payload(item)
            for key, item in value.items()
        }
    return value


def build_payload():
    candidates = []
    for alpha in (0.2, 0.3, 0.4, 0.5):
        for with_heat_capacity in (False, True):
            candidates.append(candidate("NRTL", alpha, with_heat_capacity))
    for with_heat_capacity in (False, True):
        candidates.append(candidate("UNIQUAC", None, with_heat_capacity))
    selected_nrtl = next(
        item
        for item in candidates
        if item["model"] == "NRTL" and item["alpha"] == 0.3 and item["form"] == "ABH"
    )
    selected_uniquac = next(
        item
        for item in candidates
        if item["model"] == "UNIQUAC" and item["form"] == "ABH"
    )
    interactions = [
        runtime_interaction(selected_nrtl),
        runtime_interaction(selected_uniquac),
    ]
    payload = {
        "metadata": {
            "description": (
                "Joint activity-model regression of Moreau 2012 total-pressure "
                "VLE and excess enthalpy, independently validated against "
                "Ovejero 2007 dilute Txy data."
            ),
            "script": "scripts/activity_fitting/fit_moreau_ovejero_2012_2007_activity.py",
            "objective": {
                "Moreau_pressure_sigma_kPa": 0.005,
                "Moreau_HE_relative_sigma": 0.005,
                "Ovejero_role": "held-out validation only",
            },
            "selection": (
                "NRTL alpha=0.3 selected from the 0.2/0.3/0.4/0.5 sweep "
                "for conventionality, held-out accuracy, and extrapolation "
                "margin; standard UNIQUAC retained as the companion model."
            ),
            "sources": [
                {
                    "authors": [
                        "A. Moreau",
                        "M. C. Martin",
                        "C. R. Chamorro",
                        "J. J. Segovia",
                    ],
                    "year": 2012,
                    "doi": "10.1016/j.fluid.2012.01.007",
                    "tables": [2, 6],
                },
                {
                    "authors": [
                        "G. Ovejero",
                        "M. D. Romero",
                        "E. Diez",
                        "T. Lopes",
                        "I. Diaz",
                    ],
                    "year": 2007,
                    "doi": "10.1021/je700285j",
                    "table": 5,
                },
            ],
        },
        "raw_data": {
            "moreau_pressure_313_15K": [
                {"x1": x1, "P_kPa": pressure} for x1, pressure in MOREAU_PX
            ],
            "moreau_excess_enthalpy": {
                str(temperature): (
                    [{"x1": 0.0, "HE_J_mol": 0.0}]
                    + [{"x1": x1, "HE_J_mol": value} for x1, value in rows]
                    + [{"x1": 1.0, "HE_J_mol": 0.0}]
                )
                for temperature, rows in HE.items()
            },
            "ovejero_101_3kPa": [
                {
                    "x1": x1,
                    "y1": y1,
                    "T_K": temperature,
                    "reported_gamma1": gamma1,
                    "reported_gamma2": gamma2,
                }
                for x1, y1, temperature, gamma1, gamma2 in OVEJERO
            ],
        },
        "model_selection": candidates,
        "interactions": interactions,
        "regularized_extrapolation_stress_test": {
            interaction["model"]: regularized_stress_test(interaction)
            for interaction in interactions
        },
    }
    return _stable_numeric_payload(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    payload = build_payload()
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.write is None:
        print(encoded, end="")
    else:
        args.write.write_text(encoded)
        print(f"Wrote {args.write}")


if __name__ == "__main__":
    main()
