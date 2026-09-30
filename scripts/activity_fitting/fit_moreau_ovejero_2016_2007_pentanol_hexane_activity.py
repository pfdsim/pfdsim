"""Fit Moreau 2016 pentanol/hexane data and validate against Ovejero 2007."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.activity_fitting import fit_moreau_ovejero_2012_2007_activity as common
from thermodynamics import create_thermodynamics

COMPONENTS = ("1-pentanol", "n-hexane")
R_VALUES = (4.1287, 4.4998)
Q_VALUES = (3.592, 3.856)
PSAT_KPA = (0.905, 37.188)
VL_L_MOL = (0.1101, 0.1345)
B_L_MOL = ((-3.001, -1.973), (-1.973, -1.649))

MOREAU_PXY = [
    (0.0000, 0.0000, 37.188),
    (0.0287, 0.0077, 36.627),
    (0.0813, 0.0120, 36.210),
    (0.1338, 0.0134, 35.774),
    (0.1857, 0.0143, 35.310),
    (0.2380, 0.0152, 34.807),
    (0.2899, 0.0162, 34.271),
    (0.3416, 0.0172, 33.692),
    (0.3931, 0.0182, 33.040),
    (0.4013, 0.0183, 33.041),
    (0.4445, 0.0192, 32.295),
    (0.4516, 0.0193, 32.289),
    (0.4957, 0.0202, 31.431),
    (0.5018, 0.0204, 31.426),
    (0.5468, 0.0215, 30.411),
    (0.5521, 0.0217, 30.404),
    (0.5977, 0.0231, 29.178),
    (0.6029, 0.0233, 29.141),
    (0.6526, 0.0253, 27.617),
    (0.7022, 0.0281, 25.752),
    (0.7526, 0.0320, 23.409),
    (0.8024, 0.0378, 20.537),
    (0.8524, 0.0473, 16.984),
    (0.9018, 0.0659, 12.652),
    (0.9502, 0.1172, 7.474),
    (1.0000, 1.0000, 0.920),
]
MOREAU_PX = [(x1, pressure) for x1, _, pressure in MOREAU_PXY]

HE = {
    298.15: [
        (0.0504, 329.3),
        (0.0999, 413.2),
        (0.1509, 465.6),
        (0.2010, 501.3),
        (0.2524, 526.9),
        (0.3027, 533.6),
        (0.3522, 537.8),
        (0.4030, 524.2),
        (0.4528, 505.8),
        (0.5035, 477.6),
        (0.5535, 441.6),
        (0.6025, 397.1),
        (0.6525, 346.3),
        (0.7015, 290.1),
        (0.7514, 235.6),
        (0.8021, 178.8),
        (0.8518, 126.6),
        (0.9004, 69.6),
        (0.9499, 21.0),
    ],
    313.15: [
        (0.0504, 455.1),
        (0.1000, 569.2),
        (0.1510, 637.7),
        (0.2011, 680.1),
        (0.2525, 705.9),
        (0.3007, 720.9),
        (0.3503, 724.5),
        (0.4032, 711.6),
        (0.4530, 688.6),
        (0.5039, 652.0),
        (0.5539, 605.1),
        (0.6027, 547.3),
        (0.6527, 487.4),
        (0.7016, 415.4),
        (0.7515, 340.1),
        (0.8019, 264.8),
        (0.8517, 200.0),
        (0.9004, 128.2),
        (0.9499, 67.4),
    ],
}

OVEJERO = [
    (0.0106, 0.0072, 341.9),
    (0.0207, 0.0102, 342.0),
    (0.0278, 0.0130, 342.4),
    (0.0288, 0.0131, 342.4),
    (0.0354, 0.0142, 342.2),
    (0.0395, 0.0150, 342.4),
    (0.0467, 0.0159, 342.5),
    (0.0518, 0.0172, 342.4),
    (0.0657, 0.0193, 342.6),
    (0.0694, 0.0197, 342.9),
    (0.0835, 0.0204, 342.7),
    (0.0904, 0.0209, 342.7),
    (0.0958, 0.0214, 342.7),
]

CONFIGURATION_FIELDS = (
    "COMPONENTS",
    "R_VALUES",
    "Q_VALUES",
    "PSAT_KPA",
    "VL_L_MOL",
    "B_L_MOL",
    "MOREAU_PX",
    "HE",
    "OVEJERO",
)


@contextmanager
def configured_common_fit():
    previous = {name: getattr(common, name) for name in CONFIGURATION_FIELDS}
    replacements = {name: globals()[name] for name in CONFIGURATION_FIELDS}
    replacements["OVEJERO"] = [(*row, 0.0, 0.0) for row in OVEJERO]
    try:
        for name, value in replacements.items():
            setattr(common, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(common, name, value)


def _metrics(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    return {
        "ME": float(values.mean()),
        "MAE": float(np.abs(values).mean()),
        "RMSE": float(np.sqrt(np.mean(values * values))),
        "MaxAE": float(np.abs(values).max()),
    }


def _moreau_score(model: str, alpha, parameters, heat_capacity: bool) -> dict:
    pressure = _metrics(
        common.source_bubble(model, x1, parameters, alpha, heat_capacity)[0] - observed
        for x1, observed in MOREAU_PX[1:-1]
    )
    enthalpy = {
        str(temperature): _metrics(
            common.excess_enthalpy(
                model, x1, temperature, parameters, alpha, heat_capacity
            )
            - observed
            for x1, observed in rows
        )
        for temperature, rows in HE.items()
    }
    return {"Moreau_pressure_kPa": pressure, "Moreau_HE_J_mol": enthalpy}


def _fit(model: str, alpha, heat_capacity: bool, starts) -> object:
    size = 6 if heat_capacity else 4
    lower = np.array(
        [-10.0, -10000.0, -500.0, -10.0, -10000.0, -500.0]
        if size == 6
        else [-10.0, -10000.0, -10.0, -10000.0]
    )
    results = [
        least_squares(
            lambda parameters: common.objective(
                model, alpha, heat_capacity, parameters
            ),
            start,
            bounds=(lower, -lower),
            max_nfev=1500,
            xtol=1.0e-10,
            ftol=1.0e-10,
            gtol=1.0e-10,
        )
        for start in starts
    ]
    return min(results, key=lambda result: 2.0 * result.cost)


def _candidate(model: str, alpha, form: str, result) -> dict:
    heat_capacity = form == "ABH"
    return {
        "model": model,
        "alpha": alpha,
        "form": form,
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "chi_square": float(2.0 * result.cost),
        "parameters": [float(value) for value in result.x],
        "score": _moreau_score(model, alpha, result.x, heat_capacity),
    }


def _fit_candidates() -> list[dict]:
    candidates = []
    previous_ab = None
    previous_abh = None
    published = np.array([0.7829, 0.0, 2.0893, 0.0])
    energy = np.array([0.0, 0.7829 * 313.15, 0.0, 2.0893 * 313.15])
    for alpha in (0.2, 0.3, 0.4, 0.5, 0.5397):
        starts = (
            [published, energy] if previous_ab is None else [previous_ab, published]
        )
        result_ab = _fit("NRTL", alpha, False, starts)
        candidates.append(_candidate("NRTL", alpha, "AB", result_ab))
        previous_ab = result_ab.x

        a12, b12, a21, b21 = result_ab.x
        seeded = np.array([a12, b12, 0.0, a21, b21, 0.0])
        curved = np.array([a12, b12, 1.0, a21, b21, -10.0])
        starts = [seeded, curved] if previous_abh is None else [previous_abh, seeded]
        result_abh = _fit("NRTL", alpha, True, starts)
        candidates.append(_candidate("NRTL", alpha, "ABH", result_abh))
        previous_abh = result_abh.x

    published = np.array([math.log(1.7049), 0.0, math.log(0.2106), 0.0])
    energy = np.array([0.0, math.log(1.7049) * 313.15, 0.0, math.log(0.2106) * 313.15])
    result_ab = _fit("UNIQUAC", None, False, [published, energy])
    candidates.append(_candidate("UNIQUAC", None, "AB", result_ab))
    a12, b12, a21, b21 = result_ab.x
    result_abh = _fit(
        "UNIQUAC",
        None,
        True,
        [
            [a12, b12, 0.0, a21, b21, 0.0],
            [a12, b12, -2.0, a21, b21, 5.0],
        ],
    )
    candidates.append(_candidate("UNIQUAC", None, "ABH", result_abh))
    return candidates


def _held_out_score(candidate: dict) -> dict:
    return common.score(
        candidate["model"],
        candidate["alpha"],
        candidate["parameters"],
        candidate["form"] == "ABH",
    )["Ovejero"]


def _published_reproduction() -> dict:
    records = {
        "NRTL": (0.5397, [0.7829, 0.0, 2.0893, 0.0]),
        "UNIQUAC": (
            None,
            [math.log(1.7049), 0.0, math.log(0.2106), 0.0],
        ),
    }
    result = {}
    for model, (alpha, parameters) in records.items():
        result[model] = {
            "alpha12": alpha,
            "parameters": parameters,
            "score": common.score(model, alpha, parameters, False),
            "status": (
                "approximately reproduced on the paper property basis"
                if model == "NRTL"
                else "not reproduced on PFDSim's standard UNIQUAC structural basis"
            ),
        }
    return result


def _runtime_interaction(selected: dict) -> dict:
    model = selected["model"]
    a12, b12, e12, a21, b21, e21 = common.unpack(selected["parameters"], True)
    record = {
        "model": model,
        "cas1": "71-41-0",
        "cas2": "110-54-3",
        "component1": COMPONENTS[0],
        "component2": COMPONENTS[1],
        "Tmin_K": 298.15,
        "Tmax_K": 342.9,
        "extrapolation": "inverse_square_cubic",
        "source": "Moreau et al. 2016 and Ovejero et al. 2007",
        "source_dois": [
            "10.1016/j.fluid.2016.05.031",
            "10.1021/je700285j",
        ],
        "source_file": "data/source/activity_fitting/nonwater_binary_parameters_v2.json",
        "fit_status": "recommended_vle_excess_enthalpy_heldout_validated_interaction",
        "comment": (
            "1-pentanol + n-hexane Moreau total-pressure VLE/excess-enthalpy "
            "fit validated against held-out Ovejero dilute Txy data; tangent "
            "M/T^2+N/T^3 continuation outside 298.15-342.9 K"
        ),
        "tau_tref": 313.15,
        "fit_evidence": selected["score"],
    }
    if model == "NRTL":
        record.update(
            alpha12=float(selected["alpha"]),
            tau12_c=a12,
            tau12_d=b12,
            tau12_e=e12,
            tau21_c=a21,
            tau21_d=b21,
            tau21_e=e21,
        )
    else:
        record.update(
            model_variant="standard_uniquac",
            use_q_prime=False,
            tau12_a=a12,
            tau12_b=b12,
            tau12_c=e12,
            tau21_a=a21,
            tau21_b=b21,
            tau21_c=e21,
        )
    return record


def _stress_test(record: dict) -> dict:
    thermo = create_thermodynamics(
        list(COMPONENTS), record["model"], interaction_overrides=[record]
    )
    result = {}
    for temperature in (342.9, 507.82, 1000.0, 10000.0):
        gamma1 = thermo.activity_coefficients(
            temperature, {COMPONENTS[0]: 1.0e-10, COMPONENTS[1]: 1.0 - 1.0e-10}
        )[COMPONENTS[0]]
        gamma2 = thermo.activity_coefficients(
            temperature, {COMPONENTS[0]: 1.0 - 1.0e-10, COMPONENTS[1]: 1.0e-10}
        )[COMPONENTS[1]]
        result[str(temperature)] = {
            "gamma1_infinite_dilution": float(gamma1),
            "gamma2_infinite_dilution": float(gamma2),
            "equimolar_HE_J_mol": float(
                thermo.excess_enthalpy(
                    {COMPONENTS[0]: 0.5, COMPONENTS[1]: 0.5}, temperature
                )
            ),
        }
    return result


def _stable(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, dict):
        return {key: _stable(item) for key, item in value.items()}
    return value


def build_payload() -> dict:
    with configured_common_fit():
        candidates = _fit_candidates()
        for candidate in candidates:
            if candidate["form"] == "ABH" and (
                candidate["model"] == "UNIQUAC"
                or candidate["alpha"] in (0.3, 0.4, 0.5, 0.5397)
            ):
                candidate["score"]["Ovejero"] = _held_out_score(candidate)
        selected_nrtl = next(
            candidate
            for candidate in candidates
            if candidate["model"] == "NRTL"
            and candidate["alpha"] == 0.5
            and candidate["form"] == "ABH"
        )
        selected_uniquac = next(
            candidate
            for candidate in candidates
            if candidate["model"] == "UNIQUAC" and candidate["form"] == "ABH"
        )
        interactions = [
            _runtime_interaction(selected_nrtl),
            _runtime_interaction(selected_uniquac),
        ]
        payload = {
            "metadata": {
                "description": (
                    "Moreau 2016 1-pentanol + n-hexane VLE/excess-enthalpy "
                    "regression with held-out Ovejero 2007 Txy validation"
                ),
                "script": (
                    "scripts/activity_fitting/fit_moreau_ovejero_2016_2007_pentanol_hexane_activity.py"
                ),
                "objective": {
                    "Moreau_pressure_sigma_kPa": 0.005,
                    "Moreau_HE_relative_sigma": 0.005,
                    "Ovejero_role": "held-out validation only",
                },
                "selection": (
                    "NRTL alpha=0.5 selected for the best uncertainty-weighted "
                    "joint objective and held-out balance; standard UNIQUAC "
                    "retained as a companion raw-data regression."
                ),
                "sources": [
                    {
                        "authors": [
                            "A. Moreau",
                            "J. J. Segovia",
                            "M. D. Bermejo",
                            "M. C. Martin",
                        ],
                        "year": 2016,
                        "doi": "10.1016/j.fluid.2016.05.031",
                        "tables": [2, 4, 5, 6],
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
                        "raw_data_source": "NIST ThermoML",
                    },
                ],
            },
            "source_property_basis": {
                "Psat_kPa_at_313_15K": dict(zip(COMPONENTS, PSAT_KPA)),
                "liquid_molar_volume_L_mol": dict(zip(COMPONENTS, VL_L_MOL)),
                "second_virial_L_mol": B_L_MOL,
            },
            "raw_data": {
                "moreau_pressure_313_15K": [
                    {"x1": x1, "reported_y1_calculated": y1, "P_kPa": pressure}
                    for x1, y1, pressure in MOREAU_PXY
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
                    {"x1": x1, "y1": y1, "T_K": temperature}
                    for x1, y1, temperature in OVEJERO
                ],
            },
            "published_parameter_reproduction": _published_reproduction(),
            "model_selection": candidates,
            "interactions": interactions,
            "regularized_extrapolation_stress_test": {
                interaction["model"]: _stress_test(interaction)
                for interaction in interactions
            },
        }
    return _stable(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    encoded = (
        json.dumps(build_payload(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if args.write is None:
        print(encoded, end="")
    else:
        args.write.write_text(encoded)
        print(f"Wrote {args.write}")


if __name__ == "__main__":
    main()
