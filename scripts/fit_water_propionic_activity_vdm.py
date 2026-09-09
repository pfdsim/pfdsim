#!/usr/bin/env python3
"""Fit water/propionic-acid activity models to modern salt-free VLE data.

The regression combines Olson et al. (2008) dilute PTx data with the no-salt
columns of isothermal x-y tables at 40, 50, and 60 degrees C. The 1961 broad
Txy dataset is intentionally excluded.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq, least_squares, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thermodynamics import create_thermodynamics
from physical_constants import R_CAL_MOL_K


WATER = "H2O"
ACID = "C2H5COOH"
P_ATM_BAR = 1.01325
PINNED_AZEOTROPE = {
    "P_bar": P_ATM_BAR,
    "T_K": 99.8 + 273.15,
    "x_water": 0.950,
}

# Olson et al. (2008), Table 4: corrected liquid acid mole fraction x2 and T.
DILUTE_PTX_BLOCKS = {
    101.325: [
        (0.00007, 99.978), (0.00026, 99.971), (0.00040, 99.971),
        (0.00053, 99.967), (0.00087, 99.966), (0.00090, 99.961),
        (0.00151, 99.956), (0.00165, 99.955), (0.00248, 99.942),
        (0.00296, 99.947), (0.00428, 99.937), (0.00472, 99.927),
        (0.00600, 99.926), (0.00784, 99.915), (0.00862, 99.902),
        (0.01042, 99.898), (0.01362, 99.879), (0.01428, 99.870),
        (0.02019, 99.846), (0.02186, 99.836), (0.02827, 99.817),
        (0.03579, 99.803), (0.03612, 99.800), (0.04425, 99.790),
        (0.04510, 99.790), (0.04660, 99.790), (0.05374, 99.789),
        (0.06361, 99.797), (0.06422, 99.806), (0.07070, 99.810),
        (0.07952, 99.827), (0.08984, 99.862), (0.10873, 99.921),
    ],
    40.00: [
        (0.00187, 75.938), (0.00452, 75.932), (0.00840, 75.925),
        (0.01138, 75.918), (0.01580, 75.907), (0.01972, 75.898),
        (0.02518, 75.888), (0.02949, 75.881), (0.03412, 75.876),
        (0.04109, 75.871), (0.05058, 75.870), (0.05890, 75.880),
        (0.06539, 75.890), (0.07463, 75.902),
    ],
    13.33: [
        (0.00429, 51.872), (0.00692, 51.881), (0.01050, 51.887),
        (0.01595, 51.897), (0.01978, 51.900), (0.02382, 51.906),
        (0.03362, 51.907), (0.04227, 51.914), (0.05044, 51.922),
        (0.06065, 51.933), (0.07473, 51.957),
    ],
}

# No-salt columns from modern salting studies: x_acid and y_acid. Salted
# columns are intentionally excluded from this non-electrolyte regression.
MODERN_XY_BLOCKS_C = {
    40.0: [
        (0.05, 0.035), (0.10, 0.062), (0.20, 0.091), (0.30, 0.120),
        (0.40, 0.149), (0.50, 0.159), (0.60, 0.181), (0.70, 0.207),
    ],
    50.0: [
        (0.05, 0.031), (0.10, 0.058), (0.20, 0.095), (0.30, 0.125),
        (0.40, 0.133), (0.50, 0.151), (0.60, 0.177), (0.70, 0.217),
    ],
    60.0: [
        (0.020, 0.0143), (0.050, 0.0383), (0.100, 0.0670),
        (0.200, 0.0930), (0.300, 0.1200), (0.400, 0.1410),
        (0.500, 0.1750), (0.600, 0.1900), (0.700, 0.2300),
    ],
}

DILUTE_PTX = [
    {
        "source": "Olson 2008 Table 4", "block": f"{pressure_kpa:g} kPa",
        "P_kPa": pressure_kpa, "P_bar": pressure_kpa / 100.0,
        "T_K": temperature_c + 273.15, "x_water": 1.0 - x_acid,
    }
    for pressure_kpa, rows in DILUTE_PTX_BLOCKS.items()
    for x_acid, temperature_c in rows
]

MODERN_XY = [
    {
        "source": "modern salting study, no-salt column",
        "block": f"{temperature_c:g} C",
        "T_K": temperature_c + 273.15,
        "x_acid": x_acid, "y_acid": y_acid,
    }
    for temperature_c, rows in MODERN_XY_BLOCKS_C.items()
    for x_acid, y_acid in rows
]


def composition(x_water: float) -> dict[str, float]:
    return {WATER: x_water, ACID: 1.0 - x_water}


def metrics(values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "ME": float(np.mean(array)), "MAE": float(np.mean(np.abs(array))),
        "RMSE": float(np.sqrt(np.mean(array * array))),
        "MaxAE": float(np.max(np.abs(array))),
    }


def uniquac_override(parameters) -> list[dict]:
    return [{
        "model": "UNIQUAC", "component1": WATER, "component2": ACID,
        "tau12_a": float(parameters[0]), "tau12_b": float(parameters[1]),
        "tau12_c": 0.0,
        "tau12_d": 0.0, "tau12_e": 0.0,
        "tau21_a": float(parameters[2]), "tau21_b": float(parameters[3]),
        "tau21_c": 0.0, "tau21_d": 0.0, "tau21_e": 0.0,
        "tau_tref": 298.15, "use_q_prime": False,
    }]


def nrtl_override(parameters, alpha: float) -> list[dict]:
    return [{
        "model": "NRTL", "component1": WATER, "component2": ACID,
        "alpha12": float(alpha),
        "tau12_c": float(parameters[0]), "tau12_d": float(parameters[1]),
        "tau12_e": 0.0, "tau12_f": 0.0, "tau12_g": 0.0,
        "tau21_c": float(parameters[2]), "tau21_d": float(parameters[3]),
        "tau21_e": 0.0, "tau21_f": 0.0, "tau21_g": 0.0,
        "tau_tref": 298.15,
    }]


def make_thermo(model: str, parameters=None, alpha: float | None = None):
    overrides = None
    if parameters is not None:
        overrides = (
            uniquac_override(parameters)
            if model == "UNIQUAC-VDM"
            else nrtl_override(parameters, alpha)
        )
    return create_thermodynamics(
        [WATER, ACID], model, interaction_overrides=overrides
    )


def local_temperature_and_y_residuals(thermo, row) -> tuple[float, float | None]:
    """Return locally linearized bubble-T error [K] and y-water error [pp]."""
    x = composition(row["x_water"])
    temperature, pressure = row["T_K"], row["P_bar"]

    def log_closure(at_temperature: float) -> tuple[float, dict[str, float]]:
        k = thermo.K_values(at_temperature, pressure, x)
        closure = sum(x[c] * k[c] for c in x)
        return math.log(max(closure, 1.0e-300)), k

    log_s, k = log_closure(temperature)
    step = 0.05
    log_plus, _ = log_closure(temperature + step)
    log_minus, _ = log_closure(temperature - step)
    slope = (log_plus - log_minus) / (2.0 * step)
    temperature_error = -log_s / slope if abs(slope) > 1.0e-10 else 100.0 * log_s
    if "y_water" not in row:
        return temperature_error, None
    closure = sum(x[c] * k[c] for c in x)
    calculated_y_water = x[WATER] * k[WATER] / max(closure, 1.0e-300)
    return temperature_error, 100.0 * (calculated_y_water - row["y_water"])


def isothermal_y_acid_error(thermo, row) -> float:
    x = composition(1.0 - row["x_acid"])
    pressure = thermo.bubble_point_P(x, row["T_K"])
    k = thermo.K_values(row["T_K"], pressure, x)
    closure = sum(x[c] * k[c] for c in x)
    calculated_y_acid = x[ACID] * k[ACID] / max(closure, 1.0e-300)
    return 100.0 * (calculated_y_acid - row["y_acid"])


def objective(model: str, parameters, alpha: float | None = None) -> np.ndarray:
    thermo = make_thermo(model, parameters, alpha)
    residuals = []
    for block in sorted({row["block"] for row in DILUTE_PTX}):
        rows = [row for row in DILUTE_PTX if row["block"] == block]
        weight = math.sqrt(0.5 / 3.0 / len(rows))
        for row in rows:
            temperature_error, _ = local_temperature_and_y_residuals(thermo, row)
            residuals.append(weight * temperature_error)
    for block in sorted({row["block"] for row in MODERN_XY}):
        rows = [row for row in MODERN_XY if row["block"] == block]
        weight = math.sqrt(0.5 / 3.0 / len(rows))
        residuals.extend(
            weight * isothermal_y_acid_error(thermo, row)
            for row in rows
        )
    return np.asarray(residuals)


def score_rows(thermo, rows, include_y: bool) -> dict:
    scored = []
    for row in rows:
        x = composition(row["x_water"])
        temperature = thermo.bubble_point_T(x, row["P_bar"], row["T_K"])
        k = thermo.K_values(temperature, row["P_bar"], x)
        closure = sum(x[c] * k[c] for c in x)
        item = {**row, "T_error_K": temperature - row["T_K"]}
        if include_y:
            item["y_error_pp"] = 100.0 * (
                x[WATER] * k[WATER] / closure - row["y_water"]
            )
        scored.append(item)
    result = {
        "overall": {
            "n": len(scored),
            "temperature_error_K": metrics(row["T_error_K"] for row in scored),
        },
        "by_pressure": {},
    }
    if include_y:
        result["overall"]["y_water_error_percentage_points"] = metrics(
            row["y_error_pp"] for row in scored
        )
    for block in sorted({row["block"] for row in scored}):
        selected = [row for row in scored if row["block"] == block]
        block_result = {
            "n": len(selected),
            "temperature_error_K": metrics(row["T_error_K"] for row in selected),
        }
        if include_y:
            block_result["y_water_error_percentage_points"] = metrics(
                row["y_error_pp"] for row in selected
            )
        result["by_pressure"][block] = block_result
    return result


def score_modern_xy(thermo) -> dict:
    scored = [
        {**row, "y_error_pp": isothermal_y_acid_error(thermo, row)}
        for row in MODERN_XY
    ]
    return {
        "overall": {
            "n": len(scored),
            "y_acid_error_percentage_points": metrics(
                row["y_error_pp"] for row in scored
            ),
        },
        "by_temperature": {
            block: {
                "n": sum(row["block"] == block for row in scored),
                "y_acid_error_percentage_points": metrics(
                    row["y_error_pp"] for row in scored
                    if row["block"] == block
                ),
            }
            for block in sorted({row["block"] for row in scored})
        },
    }


def azeotropes(thermo, pressure_bar: float = P_ATM_BAR) -> list[dict]:
    def difference(x_water: float) -> float:
        x = composition(x_water)
        temperature = thermo.bubble_point_T(x, pressure_bar, 373.0)
        k = thermo.K_values(temperature, pressure_bar, x)
        closure = sum(x[c] * k[c] for c in x)
        return x[WATER] * k[WATER] / closure - x_water

    grid = np.unique(np.concatenate((
        np.geomspace(1.0e-6, 0.2, 80), np.linspace(0.2, 0.999999, 120),
    )))
    values = [difference(float(value)) for value in grid]
    roots = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if f_left * f_right >= 0.0:
            continue
        root = brentq(difference, float(left), float(right), xtol=1.0e-12)
        if any(abs(root - item["x_water"]) < 1.0e-7 for item in roots):
            continue
        x = composition(root)
        temperature = thermo.bubble_point_T(x, pressure_bar, 373.0)
        k = thermo.K_values(temperature, pressure_bar, x)
        closure = sum(x[c] * k[c] for c in x)
        roots.append({
            "pressure_kPa": 100.0 * pressure_bar,
            "x_water": float(root), "x_acid": float(1.0 - root),
            "y_water": float(x[WATER] * k[WATER] / closure),
            "temperature_K": float(temperature),
            "temperature_C": float(temperature - 273.15),
            "type": "minimum-boiling",
        })
    return roots


def starts_for(model: str) -> list[tuple[float, float, float, float]]:
    if model == "UNIQUAC-VDM":
        return [
            (0.0, -486.4688 / R_CAL_MOL_K,
             0.0, 146.6582 / R_CAL_MOL_K),
            (-6.73772, 2210.71, 5.18562, -1827.73),
            (0.0, 0.0, 0.0, 0.0), (1.0, -300.0, -1.0, 300.0),
        ]
    return [
        (0.0, 0.0, 0.0, 0.0), (1.0, 0.0, -1.0, 0.0),
        (0.0, 500.0, 0.0, -500.0), (5.0, -1500.0, -2.0, 800.0),
    ]


def azeotrope_constraints(
    model: str,
    parameters,
    alpha: float | None = None,
) -> np.ndarray:
    """Hard x=y and bubble-T constraints expressed as ln(K_i)=0."""
    thermo = make_thermo(model, parameters, alpha)
    x = composition(PINNED_AZEOTROPE["x_water"])
    k = thermo.K_values(
        PINNED_AZEOTROPE["T_K"], PINNED_AZEOTROPE["P_bar"], x
    )
    return np.asarray((math.log(k[WATER]), math.log(k[ACID])))


def pinned_azeotrope_diagnostics(thermo) -> dict:
    x = composition(PINNED_AZEOTROPE["x_water"])
    k_at_pin = thermo.K_values(
        PINNED_AZEOTROPE["T_K"], PINNED_AZEOTROPE["P_bar"], x
    )
    calculated_temperature = thermo.bubble_point_T(
        x, PINNED_AZEOTROPE["P_bar"], PINNED_AZEOTROPE["T_K"]
    )
    k = thermo.K_values(calculated_temperature, PINNED_AZEOTROPE["P_bar"], x)
    closure = sum(x[c] * k[c] for c in x)
    return {
        "target": {
            "pressure_kPa": 101.325,
            "temperature_C": 99.8,
            "x_water": 0.950,
            "y_water": 0.950,
        },
        "constraint_ln_K": {
            WATER: float(math.log(k_at_pin[WATER])),
            ACID: float(math.log(k_at_pin[ACID])),
        },
        "calculated": {
            "temperature_C": float(calculated_temperature - 273.15),
            "x_water": 0.950,
            "y_water": float(x[WATER] * k[WATER] / closure),
        },
    }


def fit_unconstrained_start(task):
    model, alpha, start = task
    result = least_squares(
        lambda parameters: objective(model, parameters, alpha),
        start,
        bounds=([-20.0, -10000.0, -20.0, -10000.0],
                [20.0, 10000.0, 20.0, 10000.0]),
        x_scale="jac", xtol=1.0e-11, ftol=1.0e-11, gtol=1.0e-11,
        max_nfev=1500,
    )
    return model, alpha, result


def _scaled_to_parameters(scaled):
    return np.asarray((
        scaled[0], 1000.0 * scaled[1], scaled[2], 1000.0 * scaled[3]
    ))


def _parameters_to_scaled(parameters):
    return np.asarray((
        parameters[0], parameters[1] / 1000.0,
        parameters[2], parameters[3] / 1000.0,
    ))


def _constrained_scaled_cost(scaled, *, model, alpha):
    residuals = objective(model, _scaled_to_parameters(scaled), alpha)
    return 0.5 * float(np.dot(residuals, residuals))


def _scaled_azeotrope_constraints(scaled, *, model, alpha):
    return azeotrope_constraints(
        model,
        _scaled_to_parameters(scaled),
        alpha,
    )


def fit_constrained_start(task, workers):
    model, alpha, start = task

    result = minimize(
        partial(_constrained_scaled_cost, model=model, alpha=alpha),
        _parameters_to_scaled(start),
        method="SLSQP",
        bounds=[(-20.0, 20.0), (-10.0, 10.0),
                (-20.0, 20.0), (-10.0, 10.0)],
        constraints={
            "type": "eq",
            "fun": partial(
                _scaled_azeotrope_constraints,
                model=model,
                alpha=alpha,
            ),
        },
        options={
            "ftol": 1.0e-12,
            "maxiter": 1000,
            "workers": workers,
        },
    )
    return model, alpha, result


def assemble_fit(model, alpha, unconstrained, constrained_solutions) -> dict:
    unconstrained_thermo = make_thermo(model, unconstrained.x, alpha)
    unconstrained_reference = {
        "parameters": [float(value) for value in unconstrained.x],
        "optimizer": {
            "success": bool(unconstrained.success),
            "message": unconstrained.message,
            "cost": float(unconstrained.cost),
            "weighted_objective_residual": metrics(
                objective(model, unconstrained.x, alpha)
            ),
        },
        "azeotrope_constraint_ln_K": {
            WATER: float(azeotrope_constraints(
                model, unconstrained.x, alpha
            )[0]),
            ACID: float(azeotrope_constraints(
                model, unconstrained.x, alpha
            )[1]),
        },
        "modern_salt_free_xy_metrics": score_modern_xy(unconstrained_thermo),
        "dilute_PTx_metrics": score_rows(
            unconstrained_thermo, DILUTE_PTX, include_y=False
        ),
        "azeotropes_at_101_325_kPa": azeotropes(unconstrained_thermo),
    }

    feasible = [
        item for item in constrained_solutions
        if np.max(np.abs(azeotrope_constraints(
            model, _scaled_to_parameters(item.x), alpha
        ))) < 1.0e-7
    ]
    result = min(feasible or constrained_solutions, key=lambda item: item.fun)
    parameters = _scaled_to_parameters(result.x)
    thermo = make_thermo(model, parameters, alpha)
    return {
        "model": model, "alpha12": alpha,
        "parameters": [float(value) for value in parameters],
        "optimizer": {
            "success": bool(result.success), "message": result.message,
            "method": "SLSQP with two exact azeotrope equality constraints",
            "nfev": int(result.nfev), "cost": float(result.fun),
            "weighted_objective_residual": metrics(objective(model, parameters, alpha)),
        },
        "unconstrained_reference": unconstrained_reference,
        "modern_salt_free_xy_metrics": score_modern_xy(thermo),
        "dilute_PTx_metrics": score_rows(thermo, DILUTE_PTX, include_y=False),
        "pinned_azeotrope": pinned_azeotrope_diagnostics(thermo),
        "azeotropes_at_101_325_kPa": azeotropes(thermo),
    }


def old_uniquac_comparison() -> dict:
    thermo = make_thermo("UNIQUAC-VDM")
    return {
        "parameters": {
            "a12_cal_per_mol": 486.4688, "a21_cal_per_mol": -146.6582,
        },
        "source": "Water/Propionicacid p215 1/1a",
        "modern_salt_free_xy_metrics": score_modern_xy(thermo),
        "dilute_PTx_metrics": score_rows(thermo, DILUTE_PTX, include_y=False),
        "azeotropes_at_101_325_kPa": azeotropes(thermo),
    }


def main() -> None:
    tasks = [("UNIQUAC-VDM", None)] + [
        ("NRTL-VDM", alpha) for alpha in (0.2, 0.3, 0.4, 0.5)
    ]
    unconstrained_jobs = [
        (model, alpha, start)
        for model, alpha in tasks
        for start in starts_for(model)
    ]
    max_workers = min(os.cpu_count() or 1, len(unconstrained_jobs))
    print(
        f"Running {len(unconstrained_jobs)} unconstrained starts "
        f"across {max_workers} workers...",
        file=sys.stderr,
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        unconstrained_results = list(executor.map(
            fit_unconstrained_start,
            unconstrained_jobs,
        ))
        unconstrained_by_task = {
            task: min(
                (
                    result for model, alpha, result in unconstrained_results
                    if (model, alpha) == task
                ),
                key=lambda item: item.cost,
            )
            for task in tasks
        }
        constrained_jobs = [
            (model, alpha, start)
            for model, alpha in tasks
            for start in (
                [unconstrained_by_task[(model, alpha)].x]
                + starts_for(model)
            )
        ]
        print(
            f"Running {len(constrained_jobs)} constrained starts with "
            f"{max_workers}-worker numerical Jacobians...",
            file=sys.stderr,
            flush=True,
        )
        constrained_results = []
        for index, job in enumerate(constrained_jobs, start=1):
            model, alpha, _ = job
            label = model if alpha is None else f"{model} alpha={alpha}"
            print(
                f"  constrained start {index}/{len(constrained_jobs)}: "
                f"{label}",
                file=sys.stderr,
                flush=True,
            )
            constrained_results.append(
                fit_constrained_start(job, executor.map)
            )
    print("Scoring fitted models...", file=sys.stderr, flush=True)
    fits = [
        assemble_fit(
            model,
            alpha,
            unconstrained_by_task[(model, alpha)],
            [
                result for result_model, result_alpha, result
                in constrained_results
                if (result_model, result_alpha) == (model, alpha)
            ],
        )
        for model, alpha in tasks
    ]
    uniquac, nrtl_candidates = fits[0], fits[1:]
    selected_nrtl = min(nrtl_candidates, key=lambda item: item["optimizer"]["cost"])
    print(json.dumps({
        "data_basis": {
            "modern_salt_free_xy_points": len(MODERN_XY),
            "dilute_PTx_points": len(DILUTE_PTX),
            "modern_xy_temperature_blocks_C": {
                f"{temperature:g}": len(rows)
                for temperature, rows in MODERN_XY_BLOCKS_C.items()
            },
            "dilute_pressure_blocks_kPa": {
                f"{pressure:g}": len(rows) for pressure, rows in DILUTE_PTX_BLOCKS.items()
            },
            "excluded_data": "1961 broad Txy table and every salted column",
            "regression_data": "Modern dilute PTx plus modern no-salt isothermal x-y",
            "weighting": "Equal source weight; equal block weight within each source; T errors in K and y_acid errors in percentage points",
            "hard_azeotrope_constraint": "At 101.325 kPa and 99.8 C, x_water=y_water=0.950; enforced as K_water=K_acid=1",
        },
        "fixed_VDM": {
            "delta_H_J_per_mol": -63490.0, "delta_S_J_per_mol_K": -152.4,
        },
        "parallelism": {
            "max_workers": max_workers,
            "tasks": ["one UNIQUAC-VDM"] + [
                f"NRTL-VDM alpha={alpha}" for alpha in (0.2, 0.3, 0.4, 0.5)
            ],
        },
        "component_order": [WATER, ACID],
        "UNIQUAC-VDM": uniquac,
        "NRTL-VDM": {
            "selected_alpha12": selected_nrtl["alpha12"],
            "selection_metric": "block-balanced modern PTx plus salt-free x-y objective",
            "candidates": nrtl_candidates, "old_record": None,
        },
        "old_UNIQUAC-VDM": old_uniquac_comparison(),
    }, indent=2))


if __name__ == "__main__":
    main()
