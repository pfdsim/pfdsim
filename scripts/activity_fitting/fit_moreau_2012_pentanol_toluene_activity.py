"""Fit Moreau 2012 1-pentanol/toluene VLE and excess enthalpy."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.activity_fitting import compare_activity_extrapolation_regularizations as extrapolation
from scripts.activity_fitting import fit_moreau_ovejero_2012_2007_activity as common

COMPONENTS = ("1-pentanol", "toluene")
R_VALUES = (4.1287, 3.9228)
Q_VALUES = (3.592, 2.968)
PSAT_KPA = (0.908, 7.897)
VL_L_MOL = (0.110, 0.109)
B_L_MOL = ((-3.001, -2.250), (-2.250, -2.043))

MOREAU_PXY = [
    (0.0000, 0.0000, 7.864),
    (0.0493, 0.0283, 7.789),
    (0.0991, 0.0428, 7.698),
    (0.1495, 0.0513, 7.595),
    (0.1997, 0.0571, 7.485),
    (0.2495, 0.0620, 7.365),
    (0.2995, 0.0667, 7.232),
    (0.3497, 0.0716, 7.082),
    (0.3997, 0.0770, 6.918),
    (0.4049, 0.0776, 6.922),
    (0.4497, 0.0829, 6.723),
    (0.4551, 0.0836, 6.725),
    (0.4996, 0.0896, 6.508),
    (0.5052, 0.0904, 6.516),
    (0.5497, 0.0972, 6.268),
    (0.5551, 0.0981, 6.269),
    (0.5997, 0.1062, 5.968),
    (0.6050, 0.1073, 5.966),
    (0.6548, 0.1184, 5.638),
    (0.7046, 0.1326, 5.230),
    (0.7539, 0.1514, 4.760),
    (0.8040, 0.1788, 4.203),
    (0.8532, 0.2211, 3.555),
    (0.9023, 0.2953, 2.799),
    (0.9517, 0.4571, 1.917),
    (1.0000, 1.0000, 0.901),
]
MOREAU_PX = [(x1, pressure) for x1, _, pressure in MOREAU_PXY]
HE = {
    298.15: [
        (0.0511, 539.3),
        (0.1004, 760.3),
        (0.1497, 883.9),
        (0.2012, 972.6),
        (0.2507, 1021.2),
        (0.3003, 1048.1),
        (0.3500, 1062.7),
        (0.4018, 1042.3),
        (0.4516, 1010.7),
        (0.5016, 959.5),
        (0.5517, 893.7),
        (0.6018, 816.0),
        (0.6500, 728.1),
        (0.7003, 632.4),
        (0.7507, 528.9),
        (0.8012, 414.9),
        (0.8497, 312.5),
        (0.9004, 203.1),
        (0.9491, 101.5),
    ],
    313.15: [
        (0.0511, 596.4),
        (0.1004, 883.7),
        (0.1498, 1053.3),
        (0.2013, 1168.8),
        (0.2508, 1231.4),
        (0.3003, 1251.1),
        (0.3500, 1252.4),
        (0.3999, 1253.6),
        (0.4518, 1202.6),
        (0.5016, 1157.0),
        (0.5517, 1089.3),
        (0.6019, 1002.7),
        (0.6501, 904.2),
        (0.7004, 793.7),
        (0.7508, 666.9),
        (0.8013, 540.2),
        (0.8498, 409.2),
        (0.9005, 267.0),
        (0.9492, 132.0),
    ],
}

CONFIGURATION_FIELDS = (
    "COMPONENTS",
    "R_VALUES",
    "Q_VALUES",
    "PSAT_KPA",
    "VL_L_MOL",
    "B_L_MOL",
    "MOREAU_PX",
    "HE",
)


@contextmanager
def configured_common_fit():
    previous = {name: getattr(common, name) for name in CONFIGURATION_FIELDS}
    replacements = {name: globals()[name] for name in CONFIGURATION_FIELDS}
    try:
        for name, value in replacements.items():
            setattr(common, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(common, name, value)


def _metrics(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    return {
        "ME": float(np.mean(array)),
        "MAE": float(np.mean(np.abs(array))),
        "RMSE": float(np.sqrt(np.mean(array**2))),
        "MaxAE": float(np.max(np.abs(array))),
    }


def _score(model: str, alpha, parameters, with_heat_capacity: bool) -> dict:
    pressure_errors = []
    for x1, observed in MOREAU_PX[1:-1]:
        calculated, _ = common.source_bubble(
            model, x1, parameters, alpha, with_heat_capacity
        )
        pressure_errors.append(calculated - observed)
    enthalpy = {
        str(temperature): _metrics(
            common.excess_enthalpy(
                model,
                x1,
                temperature,
                parameters,
                alpha,
                with_heat_capacity,
            )
            - observed
            for x1, observed in rows
        )
        for temperature, rows in HE.items()
    }
    return {
        "Moreau_pressure_kPa": _metrics(pressure_errors),
        "Moreau_HE_J_mol": enthalpy,
        "interaction_values_313_15K": common.interaction_values(
            model, 313.15, parameters, with_heat_capacity
        ),
    }


def _candidate(model: str, alpha, with_heat_capacity: bool) -> dict:
    result = common.fit(model, alpha, with_heat_capacity)
    return {
        "model": model,
        "alpha": alpha,
        "form": "ABH" if with_heat_capacity else "AB",
        "success": bool(result.success),
        "chi_square": float(2.0 * result.cost),
        "parameters": [float(value) for value in result.x],
        "score": _score(model, alpha, result.x, with_heat_capacity),
    }


def _published_reproduction() -> dict:
    nrtl_parameters = [0.5156, 0.0, 1.5908, 0.0]
    uniquac_parameters = [math.log(1.1491), 0.0, math.log(0.6050), 0.0]
    return {
        "NRTL": {
            "alpha12": 0.6024,
            "parameters": nrtl_parameters,
            "score": _score("NRTL", 0.6024, nrtl_parameters, False),
            "status": "reproduced on the paper property basis",
        },
        "UNIQUAC": {
            "parameters": uniquac_parameters,
            "score": _score("UNIQUAC", None, uniquac_parameters, False),
            "status": (
                "not reproduced under PFDSim standard r/q or obvious "
                "direction/sign alternatives"
            ),
        },
    }


def _runtime_interaction(selected: dict) -> dict:
    model = selected["model"]
    parameters = selected["parameters"]
    a12, b12, e12, a21, b21, e21 = common.unpack(parameters, True)
    record = {
        "model": model,
        "cas1": "71-41-0",
        "cas2": "108-88-3",
        "component1": COMPONENTS[0],
        "component2": COMPONENTS[1],
        "Tmin_K": 298.15,
        "Tmax_K": 313.15,
        "extrapolation": "inverse_square_cubic",
        "source": "Moreau et al. 2012",
        "source_doi": "10.1016/j.fluid.2012.01.007",
        "source_file": "data/source/activity_fitting/nonwater_binary_parameters_v2.json",
        "fit_status": "recommended_joint_vle_excess_enthalpy_interaction",
        "comment": (
            "1-pentanol + toluene joint Moreau total-pressure VLE and excess-"
            "enthalpy fit; tangent M/T^2+N/T^3 continuation outside the "
            "298.15-313.15 K evidence range"
        ),
        "tau_tref": 313.15,
        "fit_evidence": selected["score"],
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


def _stress_test(record: dict) -> dict:
    model = record["model"]
    structure = extrapolation._uniquac_structure(record) if model == "UNIQUAC" else None
    result = {}
    for temperature in (313.15, 354.7, 500.0, 591.75, 1000.0, 10000.0):
        first, _ = extrapolation._ln_gamma(
            model,
            record,
            structure,
            1.0e-10,
            temperature,
            "inverse_square_cubic",
        )
        _, second = extrapolation._ln_gamma(
            model,
            record,
            structure,
            1.0 - 1.0e-10,
            temperature,
            "inverse_square_cubic",
        )
        result[str(temperature)] = {
            "gamma1_infinite_dilution": math.exp(first),
            "gamma2_infinite_dilution": math.exp(second),
            "equimolar_HE_J_mol": extrapolation._excess_enthalpy(
                model,
                record,
                structure,
                temperature,
                "inverse_square_cubic",
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
        candidates = [
            _candidate("NRTL", alpha, heat_capacity)
            for alpha in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
            for heat_capacity in (False, True)
        ]
        candidates += [
            _candidate("UNIQUAC", None, heat_capacity)
            for heat_capacity in (False, True)
        ]
        selected_nrtl = next(
            item
            for item in candidates
            if item["model"] == "NRTL"
            and item["alpha"] == 0.5
            and item["form"] == "ABH"
        )
        selected_uniquac = next(
            item
            for item in candidates
            if item["model"] == "UNIQUAC" and item["form"] == "ABH"
        )
        interactions = [
            _runtime_interaction(selected_nrtl),
            _runtime_interaction(selected_uniquac),
        ]
        payload = {
            "metadata": {
                "description": (
                    "PFDSim joint regression of Moreau 2012 1-pentanol + "
                    "toluene total-pressure VLE and excess enthalpy"
                ),
                "script": ("scripts/activity_fitting/fit_moreau_2012_pentanol_toluene_activity.py"),
                "source": {
                    "authors": [
                        "A. Moreau",
                        "M. C. Martin",
                        "C. R. Chamorro",
                        "J. J. Segovia",
                    ],
                    "year": 2012,
                    "doi": "10.1016/j.fluid.2012.01.007",
                    "tables": [3, 5, 6],
                },
                "objective": {
                    "pressure_sigma_kPa": 0.005,
                    "HE_relative_sigma": 0.005,
                    "vapor": "source second-virial property basis",
                },
                "selection": (
                    "NRTL alpha=0.5 selected for the best joint objective and "
                    "pressure/calorimetry balance; standard UNIQUAC retained "
                    "as the companion raw-data refit."
                ),
            },
            "source_property_basis": {
                "Psat_kPa_at_313_15K": dict(zip(COMPONENTS, PSAT_KPA)),
                "liquid_molar_volume_L_mol": dict(zip(COMPONENTS, VL_L_MOL)),
                "second_virial_L_mol": B_L_MOL,
            },
            "raw_data": {
                "pressure_313_15K": [
                    {"x1": x1, "reported_y1_calculated": y1, "P_kPa": pressure}
                    for x1, y1, pressure in MOREAU_PXY
                ],
                "excess_enthalpy": {
                    str(temperature): (
                        [{"x1": 0.0, "HE_J_mol": 0.0}]
                        + [{"x1": x1, "HE_J_mol": value} for x1, value in rows]
                        + [{"x1": 1.0, "HE_J_mol": 0.0}]
                    )
                    for temperature, rows in HE.items()
                },
            },
            "published_parameter_reproduction": _published_reproduction(),
            "model_selection": candidates,
            "interactions": interactions,
            "regularized_extrapolation_stress_test": {
                record["model"]: _stress_test(record) for record in interactions
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
