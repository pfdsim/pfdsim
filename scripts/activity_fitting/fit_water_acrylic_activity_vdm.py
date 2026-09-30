#!/usr/bin/env python3
"""Regress water/acrylic-acid UNIQUAC-VDM and NRTL-VDM parameters.

Measured PTx observations from Tables 1 and 4 form the objective.  Table 2
is derived from Table 1 and is retained as a non-independent Pxy diagnostic.
The paper-calculated vapor compositions in Table 1 are likewise diagnostics.
The curated pfdsim acrylic-acid vapor-dimerization parameters remain fixed.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from thermodynamics import create_thermodynamics
from physical_constants import R_CAL_MOL_K


ACID = "C2H3COOH"
WATER = "H2O"


def table1_data() -> list[dict]:
    """Reuse the carefully transcribed Table 1 blocks without executing its script."""
    source = (ROOT / "scripts/activity_fitting/fit_acrylic_vdm.py").read_text()
    tree = ast.parse(source)
    blocks = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BLOCKS"
            for target in node.targets
        ):
            blocks = ast.literal_eval(node.value)
            break
    if blocks is None:
        raise RuntimeError("Could not locate the Table 1 transcription")
    return [
        {
            "table": "Table 1",
            "subset": f"{pressure_kpa:g} kPa",
            "T_K": temperature_c + 273.15,
            "P_bar": pressure_kpa / 100.0,
            "x_water": x_water,
            "y_water_reference": y_water,
        }
        for pressure_kpa, rows in blocks
        for temperature_c, x_water, y_water in rows
    ]


TABLE4 = {
    "inhibited": [
        (0.00227, 99.983), (0.00495, 99.988), (0.00859, 99.995),
        (0.01382, 100.004), (0.02215, 100.022), (0.02940, 100.039),
        (0.03952, 100.070),
    ],
    "uninhibited": [
        (0.00074, 99.981), (0.00150, 99.983), (0.00561, 99.989),
        (0.01243, 100.000), (0.02090, 100.015), (0.03154, 100.041),
        (0.04263, 100.074), (0.05527, 100.119), (0.06814, 100.172),
    ],
}


TABLE2 = [
    (0.0888, 16.71, 0.4701), (0.1855, 21.89, 0.6318),
    (0.2862, 25.64, 0.7152), (0.3875, 28.65, 0.7685),
    (0.4905, 31.15, 0.8088), (0.5940, 33.56, 0.8425),
    (0.7094, 35.29, 0.8772), (0.8084, 36.63, 0.9086),
    (0.8991, 37.48, 0.9446),
]


def table4_data() -> list[dict]:
    return [
        {
            "table": "Table 4",
            "subset": subset,
            "T_K": temperature_c + 273.15,
            "P_bar": 1.01325,
            "x_water": 1.0 - x_acid,
            "y_water_reference": None,
        }
        for subset, rows in TABLE4.items()
        for x_acid, temperature_c in rows
    ]


TRAINING = table1_data() + table4_data()


def interaction(parameters: np.ndarray | list[float]) -> list[dict]:
    return [{
        "model": "UNIQUAC",
        "component1": WATER,
        "component2": ACID,
        "a12_cal_per_mol": float(parameters[0]),
        "a21_cal_per_mol": float(parameters[1]),
        "use_q_prime": False,
    }]


def temperature_dependent_interaction(parameters: np.ndarray | list[float]) -> list[dict]:
    return [{
        "model": "UNIQUAC",
        "component1": WATER,
        "component2": ACID,
        "tau12_a": float(parameters[0]),
        "tau12_b": float(parameters[1]),
        "tau12_c": 0.0,
        "tau12_d": 0.0,
        "tau12_e": 0.0,
        "tau21_a": float(parameters[2]),
        "tau21_b": float(parameters[3]),
        "tau21_c": 0.0,
        "tau21_d": 0.0,
        "tau21_e": 0.0,
        "tau_tref": 298.15,
        "use_q_prime": False,
    }]


def make_thermo(parameters, parameterization="constant_energy"):
    overrides = (
        interaction(parameters)
        if parameterization == "constant_energy"
        else temperature_dependent_interaction(parameters)
    )
    return create_thermodynamics(
        [WATER, ACID], "UNIQUAC-VDM", interaction_overrides=overrides
    )


def composition(x_water: float) -> dict[str, float]:
    return {WATER: x_water, ACID: 1.0 - x_water}


def closure_residuals(parameters, parameterization="constant_energy") -> np.ndarray:
    thermo = make_thermo(parameters, parameterization)
    values = []
    for row in TRAINING:
        x = composition(row["x_water"])
        k = thermo.K_values(row["T_K"], row["P_bar"], x)
        values.append(math.log(sum(x[c] * k[c] for c in x)))
    return np.asarray(values)


def metrics(values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "ME": float(np.mean(array)),
        "MAE": float(np.mean(np.abs(array))),
        "RMSE": float(np.sqrt(np.mean(array * array))),
        "MaxAE": float(np.max(np.abs(array))),
    }


def score(parameters, parameterization="constant_energy") -> dict:
    thermo = make_thermo(parameters, parameterization)
    scored = []
    for row in TRAINING:
        x = composition(row["x_water"])
        predicted_T = thermo.bubble_point_T(x, row["P_bar"], row["T_K"])
        k = thermo.K_values(predicted_T, row["P_bar"], x)
        denominator = sum(x[c] * k[c] for c in x)
        predicted_y_water = x[WATER] * k[WATER] / denominator
        scored.append({
            **row,
            "T_error_K": predicted_T - row["T_K"],
            "y_error_pp": (
                100.0 * (predicted_y_water - row["y_water_reference"])
                if row["y_water_reference"] is not None else None
            ),
        })

    table2_scored = []
    for x_water, pressure_kpa, y_water in TABLE2:
        x = composition(x_water)
        predicted_P = thermo.bubble_point_P(x, 348.15)
        k = thermo.K_values(348.15, predicted_P, x)
        denominator = sum(x[c] * k[c] for c in x)
        predicted_y = x[WATER] * k[WATER] / denominator
        table2_scored.append({
            "P_error_kPa": 100.0 * predicted_P - pressure_kpa,
            "P_relative_error_percent": 100.0 * (100.0 * predicted_P - pressure_kpa) / pressure_kpa,
            "y_error_pp": 100.0 * (predicted_y - y_water),
        })

    groups = {}
    group_specs = {
        "Tables 1+4 measured overall": scored,
        "Table 1 measured PTx": [row for row in scored if row["table"] == "Table 1"],
        "Table 4 measured dilute PTx": [row for row in scored if row["table"] == "Table 4"],
        "Table 4 inhibited": [row for row in scored if row["subset"] == "inhibited"],
        "Table 4 uninhibited": [row for row in scored if row["subset"] == "uninhibited"],
    }
    for label, rows in group_specs.items():
        groups[label] = {"n": len(rows), "temperature_error_K": metrics(row["T_error_K"] for row in rows)}
        y_errors = [row["y_error_pp"] for row in rows if row["y_error_pp"] is not None]
        if y_errors:
            groups[label]["paper_calculated_y_error_percentage_points"] = metrics(y_errors)

    table1_by_pressure = {}
    for subset in sorted(
        {row["subset"] for row in scored if row["table"] == "Table 1"},
        key=lambda value: float(value.split()[0]),
    ):
        rows = [row for row in scored if row["subset"] == subset]
        table1_by_pressure[subset] = {
            "temperature_error_K": metrics(row["T_error_K"] for row in rows),
            "paper_calculated_y_error_percentage_points": metrics(row["y_error_pp"] for row in rows),
        }

    return {
        "groups": groups,
        "Table 1 by pressure": table1_by_pressure,
        "Table 2 derived Pxy diagnostic": {
            "n": len(table2_scored),
            "pressure_error_kPa": metrics(row["P_error_kPa"] for row in table2_scored),
            "pressure_relative_error_percent": metrics(row["P_relative_error_percent"] for row in table2_scored),
            "paper_calculated_y_error_percentage_points": metrics(row["y_error_pp"] for row in table2_scored),
        },
    }


def fit_uniquac() -> dict:
    starts = [
        (-417.4722, 1189.9025),
        (0.0, 0.0),
        (-500.0, 500.0),
        (500.0, -500.0),
    ]
    solutions = [
        least_squares(
            closure_residuals,
            start,
            bounds=([-5000.0, -5000.0], [5000.0, 5000.0]),
            x_scale="jac",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=1000,
        )
        for start in starts
    ]
    result = min(solutions, key=lambda item: item.cost)
    residual = closure_residuals(result.x)
    energy_to_b = lambda energy: -energy / R_CAL_MOL_K
    temperature_dependent_starts = [
        (0.0, energy_to_b(result.x[0]), 0.0, energy_to_b(result.x[1])),
        (math.log(0.50571), 0.0, math.log(0.51123), 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (1.0, -300.0, -1.0, 300.0),
    ]
    temperature_dependent_solutions = [
        least_squares(
            lambda parameters: closure_residuals(parameters, "A_plus_B_over_T"),
            start,
            bounds=([-20.0, -10000.0, -20.0, -10000.0],
                    [20.0, 10000.0, 20.0, 10000.0]),
            x_scale="jac",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=2000,
        )
        for start in temperature_dependent_starts
    ]
    temperature_dependent = min(
        temperature_dependent_solutions, key=lambda item: item.cost
    )
    temperature_dependent_residual = closure_residuals(
        temperature_dependent.x, "A_plus_B_over_T"
    )
    report = {
        "source": "Olson, Morrison, and Wilson, Ind. Eng. Chem. Res. 2008, 47, 5127-5131",
        "model": "pfdsim UNIQUAC-VDM",
        "fit_basis": {
            "Table 1 measured mixture PTx points": len(table1_data()),
            "Table 4 measured inhibited dilute PTx points": len(TABLE4["inhibited"]),
            "Table 4 measured uninhibited dilute PTx points": len(TABLE4["uninhibited"]),
            "objective": "unweighted ln(sum_i x_i K_i) bubble-closure residual",
            "excluded pure-component baselines": "No binary-interaction information",
            "Table 2": "Non-independent derived interpolation; diagnostic only",
            "Table 3": "Published parameter benchmark; not an observation",
            "Table 1 y": "Paper-calculated; diagnostic only",
        },
        "fixed_VDM": {"delta_H_J_per_mol": -67000.0, "delta_S_J_per_mol_K": -161.4},
        "fitted_UNIQUAC": {
            "component_order": [WATER, ACID],
            "a12_cal_per_mol": float(result.x[0]),
            "a21_cal_per_mol": float(result.x[1]),
            "tau_definition": "tau_ij = exp(-a_ij/(R_cal T))",
        },
        "optimizer": {
            "success": bool(result.success),
            "message": result.message,
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "bubble_closure_ln": metrics(residual),
        },
        "fit_metrics": score(result.x),
        "temperature_dependent_fit": {
            "functional_form": {
                "tau12": "exp(A12 + B12/T)",
                "tau21": "exp(A21 + B21/T)",
                "units": {"A": "dimensionless", "B": "K"},
                "omitted_terms": "tau12_c*T and tau21_c*T fixed to zero",
            },
            "parameters": {
                "A12": float(temperature_dependent.x[0]),
                "B12_K": float(temperature_dependent.x[1]),
                "A21": float(temperature_dependent.x[2]),
                "B21_K": float(temperature_dependent.x[3]),
            },
            "optimizer": {
                "success": bool(temperature_dependent.success),
                "message": temperature_dependent.message,
                "nfev": int(temperature_dependent.nfev),
                "cost": float(temperature_dependent.cost),
                "bubble_closure_ln": metrics(temperature_dependent_residual),
            },
            "fit_metrics": score(
                temperature_dependent.x, "A_plus_B_over_T"
            ),
        },
        "current_catalog_comparison": {
            "parameters": {"a12_cal_per_mol": -417.4722, "a21_cal_per_mol": 1189.9025},
            "metrics": score([-417.4722, 1189.9025]),
        },
    }
    return report


def nrtl_interaction(parameters, alpha: float) -> list[dict]:
    return [{
        "model": "NRTL",
        "component1": WATER,
        "component2": ACID,
        "alpha12": float(alpha),
        "tau12_c": float(parameters[0]),
        "tau12_d": float(parameters[1]),
        "tau12_e": 0.0,
        "tau12_f": 0.0,
        "tau21_c": float(parameters[2]),
        "tau21_d": float(parameters[3]),
        "tau21_e": 0.0,
        "tau21_f": 0.0,
        "tau_tref": 298.15,
    }]


def make_nrtl_thermo(parameters, alpha: float):
    return create_thermodynamics(
        [WATER, ACID],
        "NRTL-VDM",
        interaction_overrides=nrtl_interaction(parameters, alpha),
    )


def nrtl_closure_residuals(parameters, alpha: float) -> np.ndarray:
    thermo = make_nrtl_thermo(parameters, alpha)
    residuals = []
    for row in TRAINING:
        x = composition(row["x_water"])
        k = thermo.K_values(row["T_K"], row["P_bar"], x)
        residuals.append(math.log(sum(x[c] * k[c] for c in x)))
    return np.asarray(residuals)


def score_nrtl(parameters, alpha: float) -> dict:
    thermo = make_nrtl_thermo(parameters, alpha)
    scored = []
    for row in TRAINING:
        x = composition(row["x_water"])
        predicted_T = thermo.bubble_point_T(x, row["P_bar"], row["T_K"])
        k = thermo.K_values(predicted_T, row["P_bar"], x)
        denominator = sum(x[c] * k[c] for c in x)
        predicted_y = x[WATER] * k[WATER] / denominator
        scored.append({
            **row,
            "T_error_K": predicted_T - row["T_K"],
            "y_error_pp": (
                100.0 * (predicted_y - row["y_water_reference"])
                if row["y_water_reference"] is not None else None
            ),
        })

    def group(rows):
        result = {
            "n": len(rows),
            "temperature_error_K": metrics(row["T_error_K"] for row in rows),
        }
        y_errors = [row["y_error_pp"] for row in rows if row["y_error_pp"] is not None]
        if y_errors:
            result["paper_calculated_y_error_percentage_points"] = metrics(y_errors)
        return result

    table2_rows = []
    for x_water, pressure_kpa, y_water in TABLE2:
        x = composition(x_water)
        predicted_P = thermo.bubble_point_P(x, 348.15)
        k = thermo.K_values(348.15, predicted_P, x)
        denominator = sum(x[c] * k[c] for c in x)
        predicted_y = x[WATER] * k[WATER] / denominator
        table2_rows.append({
            "P_error_kPa": 100.0 * predicted_P - pressure_kpa,
            "P_relative_error_percent": (
                100.0 * (100.0 * predicted_P - pressure_kpa) / pressure_kpa
            ),
            "y_error_pp": 100.0 * (predicted_y - y_water),
        })

    return {
        "Tables 1+4 measured overall": group(scored),
        "Table 1 measured PTx": group(
            [row for row in scored if row["table"] == "Table 1"]
        ),
        "Table 4 measured dilute PTx": group(
            [row for row in scored if row["table"] == "Table 4"]
        ),
        "Table 2 derived Pxy diagnostic": {
            "n": len(table2_rows),
            "pressure_error_kPa": metrics(row["P_error_kPa"] for row in table2_rows),
            "pressure_relative_error_percent": metrics(
                row["P_relative_error_percent"] for row in table2_rows
            ),
            "paper_calculated_y_error_percentage_points": metrics(
                row["y_error_pp"] for row in table2_rows
            ),
        },
    }


def fit_nrtl_alpha(alpha: float) -> dict:
    starts = [
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, -1.0, 0.0),
        (0.0, 500.0, 0.0, -500.0),
        (5.0, -1500.0, -2.0, 800.0),
    ]
    solutions = [
        least_squares(
            lambda parameters: nrtl_closure_residuals(parameters, alpha),
            start,
            bounds=([-20.0, -10000.0, -20.0, -10000.0],
                    [20.0, 10000.0, 20.0, 10000.0]),
            x_scale="jac",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=2000,
        )
        for start in starts
    ]
    result = min(solutions, key=lambda item: item.cost)
    residuals = nrtl_closure_residuals(result.x, alpha)
    return {
        "alpha12": alpha,
        "parameters": {
            "tau12_c": float(result.x[0]),
            "tau12_d_K": float(result.x[1]),
            "tau12_e": 0.0,
            "tau12_f_per_K": 0.0,
            "tau21_c": float(result.x[2]),
            "tau21_d_K": float(result.x[3]),
            "tau21_e": 0.0,
            "tau21_f_per_K": 0.0,
            "tau_tref_K": 298.15,
        },
        "optimizer": {
            "success": bool(result.success),
            "message": result.message,
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "bubble_closure_ln": metrics(residuals),
        },
        "fit_metrics": score_nrtl(result.x, alpha),
    }


def fit_nrtl() -> dict:
    candidates = [fit_nrtl_alpha(alpha) for alpha in (0.2, 0.3, 0.4, 0.5)]
    selected = min(
        candidates,
        key=lambda candidate: candidate["fit_metrics"]
        ["Tables 1+4 measured overall"]["temperature_error_K"]["RMSE"],
    )
    return {
        "source": "Olson, Morrison, and Wilson, Ind. Eng. Chem. Res. 2008, 47, 5127-5131",
        "model": "pfdsim NRTL-VDM",
        "component_order": [WATER, ACID],
        "functional_form": "tau_ij = tau_ij_c + tau_ij_d/T",
        "fixed_VDM": {
            "delta_H_J_per_mol": -67000.0,
            "delta_S_J_per_mol_K": -161.4,
        },
        "selection_metric": "Tables 1+4 measured temperature RMSE",
        "alpha_candidate_ceiling": 0.5,
        "selected_alpha12": selected["alpha12"],
        "candidates": candidates,
    }


def main() -> None:
    print(json.dumps({
        "source": "Olson, Morrison, and Wilson, Ind. Eng. Chem. Res. 2008, 47, 5127-5131",
        "fit_basis": "Tables 1 and 4 measured PTx; Table 2 and calculated y diagnostic only",
        "UNIQUAC-VDM": fit_uniquac(),
        "NRTL-VDM": fit_nrtl(),
    }, indent=2))


if __name__ == "__main__":
    main()
