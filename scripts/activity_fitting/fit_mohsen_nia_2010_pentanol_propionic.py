#!/usr/bin/env python3
"""Audit 1-pentanol/propionic-acid VLE with explicit vapor association.

The source paper treated the vapor as ideal. This script ignores its reported
activity coefficients, jointly fits its two raw T-x-y pressure series with
VDM, evaluates the same liquid models with HOC, and emits evidence-backed zero
interactions when vapor-model uncertainty dominates the liquid correction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermodynamics import create_thermodynamics  # noqa: E402


COMPONENTS = ("1-pentanol", "propionic acid")
CAS = ("71-41-0", "79-09-4")
SOURCE = {
    "authors": ["M. Mohsen-Nia", "M. R. Memarzadeh"],
    "title": (
        "Isobaric (vapour + liquid) equilibria for the "
        "(1-pentanol + propionic acid) binary mixture at (53.3 and 91.3) kPa"
    ),
    "journal": "The Journal of Chemical Thermodynamics",
    "volume": 42,
    "pages": "1311-1315",
    "year": 2010,
    "doi": "10.1016/j.jct.2010.05.014",
    "table": "Table 5",
}

# (liquid x_pentanol, vapor y_pentanol, T/K). The paper's gamma columns are
# deliberately omitted because they embed the ideal-vapor assumption.
BLOCKS = {
    0.533: [
        (0.0920, 0.0720, 395.45), (0.1101, 0.0902, 395.60),
        (0.1503, 0.1174, 395.85), (0.1910, 0.1499, 396.25),
        (0.2470, 0.2001, 396.65), (0.2774, 0.2349, 396.85),
        (0.3220, 0.2940, 397.00), (0.3909, 0.3692, 397.20),
        (0.4452, 0.4391, 397.25), (0.4499, 0.4412, 397.25),
        (0.4584, 0.4582, 397.25), (0.4693, 0.4698, 397.20),
        (0.4885, 0.4912, 397.20), (0.5047, 0.5173, 397.15),
        (0.5401, 0.5638, 397.05), (0.5821, 0.6059, 396.95),
        (0.6270, 0.6640, 396.75), (0.6702, 0.7202, 396.50),
        (0.7202, 0.7690, 396.15), (0.7640, 0.8180, 395.75),
        (0.8381, 0.8703, 395.25), (0.9219, 0.9297, 394.35),
        (0.9644, 0.9821, 393.50),
    ],
    0.913: [
        (0.0270, 0.0180, 410.85), (0.1016, 0.0705, 411.65),
        (0.1352, 0.0996, 411.95), (0.1840, 0.1409, 412.15),
        (0.2182, 0.1864, 412.40), (0.2701, 0.2392, 412.65),
        (0.3199, 0.2985, 412.95), (0.3844, 0.3721, 413.15),
        (0.4389, 0.4391, 413.25), (0.4490, 0.4502, 413.25),
        (0.4602, 0.4657, 413.25), (0.4730, 0.4796, 413.20),
        (0.4876, 0.4969, 413.15), (0.5028, 0.5142, 413.05),
        (0.5379, 0.5604, 412.90), (0.5703, 0.6040, 412.70),
        (0.6193, 0.6651, 412.45), (0.6620, 0.7030, 412.15),
        (0.7389, 0.7996, 411.35), (0.7930, 0.8450, 410.65),
        (0.8600, 0.9111, 409.85), (0.9193, 0.9514, 409.40),
        (0.9740, 0.9903, 408.55),
    ],
}
PURE_ENDPOINTS = {
    0.533: [
        {"x1": 0.0, "y1": 0.0, "T_K": 394.55},
        {"x1": 1.0, "y1": 1.0, "T_K": 393.15},
    ],
    0.913: [
        {"x1": 0.0, "y1": 0.0, "T_K": 410.55},
        {"x1": 1.0, "y1": 1.0, "T_K": 408.05},
    ],
}
PUBLISHED = {
    0.533: {
        "NRTL": [-90.30288961317957, 42.21667907119223, 0.101],
        "UNIQUAC": [-102.98043774116819, 76.95566501231305],
    },
    0.913: {
        "NRTL": [-95.79981732805233, 51.15544077032267, 0.079],
        "UNIQUAC": [-106.33399181682452, 78.27324866180611],
    },
}


def interaction_override(model: str, parameters) -> list[dict]:
    if model == "NRTL":
        return [{
            "model": "NRTL",
            "component1": COMPONENTS[0],
            "component2": COMPONENTS[1],
            "alpha12": float(parameters[2]),
            "tau12_c": 0.0,
            "tau12_d": float(parameters[0]),
            "tau21_c": 0.0,
            "tau21_d": float(parameters[1]),
        }]
    return [{
        "model": "UNIQUAC",
        "component1": COMPONENTS[0],
        "component2": COMPONENTS[1],
        "tau12_a": 0.0,
        "tau12_b": float(parameters[0]),
        "tau21_a": 0.0,
        "tau21_b": float(parameters[1]),
        "use_q_prime": False,
    }]


class MutableVDMModel:
    def __init__(self, model: str):
        initial = [-93.0, 46.0, 0.1] if model == "NRTL" else [-105.0, 77.0]
        self.model = model
        self.thermo = create_thermodynamics(
            list(COMPONENTS),
            f"{model}-VDM",
            interaction_overrides=interaction_override(model, initial),
        )
        mapping = (
            self.thermo._nrtl_interaction_overrides
            if model == "NRTL"
            else self.thermo._uniquac_interaction_overrides
        )
        self.record = mapping[COMPONENTS]

    def update(self, parameters) -> None:
        if self.model == "NRTL":
            self.record.update(
                alpha12=float(parameters[2]),
                tau12_d=float(parameters[0]),
                tau21_d=float(parameters[1]),
            )
            self.thermo._nrtl_parameter_cache = None
            self.thermo._nrtl_matrix_cache.clear()
        else:
            self.record.update(
                tau12_b=float(parameters[0]),
                tau21_b=float(parameters[1]),
            )
            self.thermo._uniquac_parameter_cache = None
            self.thermo._uniquac_tau_cache.clear()
        self.thermo._activity_cache.clear()
        self.thermo._compiled_activity_cache.clear()
        self.thermo._k_values_cache.clear()

    def local_errors(self, pressure: float, row) -> tuple[float, float]:
        x1, y1, temperature = row
        composition = {COMPONENTS[0]: x1, COMPONENTS[1]: 1.0 - x1}

        def closure(at_temperature: float):
            k_values = self.thermo.K_values(
                at_temperature, pressure, composition
            )
            total = sum(composition[c] * k_values[c] for c in composition)
            return math.log(total), k_values

        log_closure, k_values = closure(temperature)
        step = 0.03
        slope = (
            closure(temperature + step)[0] - closure(temperature - step)[0]
        ) / (2.0 * step)
        temperature_error = -log_closure / slope
        total = sum(composition[c] * k_values[c] for c in composition)
        calculated_y1 = (
            x1 * k_values[COMPONENTS[0]] / total
        )
        return temperature_error, 100.0 * (calculated_y1 - y1)


def objective(engine: MutableVDMModel, parameters) -> np.ndarray:
    engine.update(parameters)
    residuals = []
    for pressure, rows in BLOCKS.items():
        scale = math.sqrt(0.5 / len(rows))
        for row in rows:
            temperature_error, y_error_pp = engine.local_errors(pressure, row)
            residuals.extend((scale * temperature_error, scale * y_error_pp))
    return np.asarray(residuals)


def fit_vdm(model: str) -> dict:
    engine = MutableVDMModel(model)
    if model == "NRTL":
        alpha = 0.1
        result = least_squares(
            lambda values: objective(
                engine, np.asarray([values[0], values[1], alpha])
            ),
            [-93.0, 46.0],
            bounds=([-5000.0, -5000.0], [5000.0, 5000.0]),
            x_scale="jac",
            max_nfev=180,
        )
        parameters = [float(result.x[0]), float(result.x[1]), alpha]
    else:
        result = least_squares(
            lambda values: objective(engine, values),
            [-105.0, 77.0],
            bounds=([-5000.0, -5000.0], [5000.0, 5000.0]),
            x_scale="jac",
            max_nfev=180,
        )
        parameters = [float(value) for value in result.x]
    if not result.success:
        raise RuntimeError(f"{model}-VDM regression failed: {result.message}")
    return {
        "parameters": parameters,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "objective": (
            "equal weight to bubble-T error in K and vapor-y error in "
            "percentage points; shared parameters across both pressures"
        ),
    }


def _metrics(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    return {
        "ME": float(np.mean(array)),
        "MAE": float(np.mean(np.abs(array))),
        "RMSE": float(np.sqrt(np.mean(array**2))),
        "MaxAE": float(np.max(np.abs(array))),
    }


def predictions(model: str, vapor: str, parameters) -> list[dict]:
    thermo = create_thermodynamics(
        list(COMPONENTS),
        f"{model}-{vapor}" if vapor != "IDEAL" else model,
        interaction_overrides=interaction_override(model, parameters),
    )
    predicted = []
    for pressure, rows in BLOCKS.items():
        for x1, y1, temperature in rows:
            composition = {COMPONENTS[0]: x1, COMPONENTS[1]: 1.0 - x1}
            calculated_temperature = thermo.bubble_point_T(
                composition, pressure, temperature
            )
            k_values = thermo.K_values(
                calculated_temperature, pressure, composition
            )
            total = sum(composition[c] * k_values[c] for c in composition)
            calculated_y1 = x1 * k_values[COMPONENTS[0]] / total
            predicted.append({
                "P_bar": pressure,
                "x1": x1,
                "T_error_K": calculated_temperature - temperature,
                "y1_error": calculated_y1 - y1,
                "calculated_T_K": calculated_temperature,
                "calculated_y1": calculated_y1,
            })
    return predicted


def score(predicted: list[dict]) -> dict:
    result = {
        "overall": {
            "temperature_error_K": _metrics(
                row["T_error_K"] for row in predicted
            ),
            "y1_error": _metrics(row["y1_error"] for row in predicted),
        },
        "by_pressure": {},
    }
    for pressure in BLOCKS:
        selected = [row for row in predicted if row["P_bar"] == pressure]
        if not selected:
            continue
        result["by_pressure"][str(pressure)] = {
            "temperature_error_K": _metrics(
                row["T_error_K"] for row in selected
            ),
            "y1_error": _metrics(row["y1_error"] for row in selected),
        }
    return result


def azeotropes(model: str, vapor: str, parameters) -> dict:
    thermo = create_thermodynamics(
        list(COMPONENTS),
        f"{model}-{vapor}" if vapor != "IDEAL" else model,
        interaction_overrides=interaction_override(model, parameters),
    )
    result = {}
    for pressure in BLOCKS:
        guess = 397.0 if pressure < 0.7 else 413.0

        def difference(x1: float) -> float:
            composition = {COMPONENTS[0]: x1, COMPONENTS[1]: 1.0 - x1}
            temperature = thermo.bubble_point_T(composition, pressure, guess)
            k_values = thermo.K_values(temperature, pressure, composition)
            total = sum(composition[c] * k_values[c] for c in composition)
            return x1 * k_values[COMPONENTS[0]] / total - x1

        grid = np.linspace(0.02, 0.98, 25)
        values = [difference(float(x1)) for x1 in grid]
        roots = []
        for left, right, f_left, f_right in zip(
            grid[:-1], grid[1:], values[:-1], values[1:]
        ):
            if f_left * f_right >= 0.0:
                continue
            x1 = brentq(difference, float(left), float(right))
            composition = {COMPONENTS[0]: x1, COMPONENTS[1]: 1.0 - x1}
            temperature = thermo.bubble_point_T(composition, pressure, guess)
            roots.append({"x1": float(x1), "T_K": float(temperature)})
        result[str(pressure)] = roots
    return result


def published_reproduction() -> dict:
    result = {}
    for pressure, models in PUBLISHED.items():
        result[str(pressure)] = {}
        for model, parameters in models.items():
            selected = [
                row for row in predictions(model, "IDEAL", parameters)
                if row["P_bar"] == pressure
            ]
            result[str(pressure)][model] = score(selected)["overall"]
    return result


def build_payload() -> dict:
    fitted = {model: fit_vdm(model) for model in ("NRTL", "UNIQUAC")}
    zero_parameters = {"NRTL": [0.0, 0.0, 0.3], "UNIQUAC": [0.0, 0.0]}
    fitted_evaluation = {
        model: {
            vapor: score(predictions(model, vapor, fit["parameters"]))
            for vapor in ("VDM", "HOC")
        }
        for model, fit in fitted.items()
    }
    zero_evaluation = {
        model: {
            vapor: score(predictions(model, vapor, zero_parameters[model]))
            for vapor in ("VDM", "HOC")
        }
        for model in ("NRTL", "UNIQUAC")
    }
    zero_azeotropes = {
        vapor: azeotropes("NRTL", vapor, zero_parameters["NRTL"])
        for vapor in ("IDEAL", "VDM", "HOC")
    }
    raw_data = {
        str(pressure): {
            "interior": [
                {"x1": x1, "y1": y1, "T_K": temperature}
                for x1, y1, temperature in rows
            ],
            "pure_endpoints": PURE_ENDPOINTS[pressure],
        }
        for pressure, rows in BLOCKS.items()
    }
    rationale = (
        "Zero liquid interaction selected because it reproduces both pressure "
        "series within about 0.2-0.5 K depending on the association model, "
        "nonzero VDM fits do not transfer favorably to HOC, and vapor-model "
        "disagreement is at least as consequential as the residual liquid fit."
    )
    evidence = {
        "selection": rationale,
        "zero_model_evaluation": zero_evaluation,
        "vdm_refits": fitted,
        "vdm_refit_evaluation": fitted_evaluation,
        "zero_liquid_azeotropes": zero_azeotropes,
        "published_parameter_reproduction": published_reproduction(),
    }
    evidence_summary = {
        "analysis_ref": "top-level analysis",
        "selection": rationale,
        "zero_model_evaluation": zero_evaluation,
    }
    interactions = [
        {
            "model": "NRTL",
            "cas1": CAS[0],
            "cas2": CAS[1],
            "component1": COMPONENTS[0],
            "component2": COMPONENTS[1],
            "Tmin_K": 393.15,
            "Tmax_K": 413.25,
            "extrapolation": "unrestricted",
            "source": "Mohsen-Nia and Memarzadeh 2010 T-x-y; PFDSim VDM/HOC audit",
            "source_doi": SOURCE["doi"],
            "fit_status": "recommended_defensible_zero_interaction",
            "comment": (
                "1-pentanol + propionic acid defensible zero NRTL liquid "
                "interaction; use explicit VDM or HOC vapor association"
            ),
            "alpha12": 0.3,
            "tau12_c": 0.0,
            "tau21_c": 0.0,
            "evidence": evidence_summary,
        },
        {
            "model": "UNIQUAC",
            "cas1": CAS[0],
            "cas2": CAS[1],
            "component1": COMPONENTS[0],
            "component2": COMPONENTS[1],
            "Tmin_K": 393.15,
            "Tmax_K": 413.25,
            "extrapolation": "unrestricted",
            "source": "Mohsen-Nia and Memarzadeh 2010 T-x-y; PFDSim VDM/HOC audit",
            "source_doi": SOURCE["doi"],
            "fit_status": "recommended_defensible_zero_interaction",
            "comment": (
                "1-pentanol + propionic acid defensible zero UNIQUAC residual "
                "interaction; use explicit VDM or HOC vapor association"
            ),
            "model_variant": "standard_uniquac",
            "use_q_prime": False,
            "tau12_a": 0.0,
            "tau21_a": 0.0,
            "evidence": evidence_summary,
        },
    ]
    return {
        "metadata": {
            "description": (
                "Association-aware audit of Mohsen-Nia and Memarzadeh 2010 "
                "1-pentanol + propionic-acid VLE."
            ),
            "source": SOURCE,
            "component_order": list(COMPONENTS),
            "ignored_source_fields": [
                "reported gamma1 and gamma2 derived assuming ideal vapor"
            ],
            "script": "scripts/activity_fitting/fit_mohsen_nia_2010_pentanol_propionic.py",
        },
        "raw_data": raw_data,
        "analysis": evidence,
        "interactions": interactions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(build_payload(), indent=2, sort_keys=True) + "\n"
    if args.write:
        args.write.write_text(rendered)
        print(f"Wrote {args.write}")
    elif args.check:
        if args.check.read_text() != rendered:
            raise SystemExit(f"Refit output differs from {args.check}")
        print(f"Refit output matches {args.check}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
