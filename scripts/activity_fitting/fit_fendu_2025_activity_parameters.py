#!/usr/bin/env python3
"""Refit the Fendu 2025 isobutanol binary p-T-x data for PFDSim.

The paper's objective is retained: unweighted relative pressure residuals.
Candidate forms are deliberately limited to equations represented exactly by
PFDSim. Validation leaves out one complete composition series at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermodynamics_models.interaction_estimation import (  # noqa: E402
    _nrtl_ln_gamma,
    _uniquac_ln_gamma,
)


SOURCE = {
    "authors": ["Elena Mirela Fendu", "Marilena Pricop-Nicolae"],
    "title": (
        "Experimental and Regression VLE Data for Isobutanol + 1-Butanol, "
        "Isobutanol + 2-Ethyl-1-hexanol, and 1-Butanol + "
        "2-Ethyl-1-hexanol Binary Systems"
    ),
    "journal": "Processes",
    "volume": 13,
    "article": 2034,
    "year": 2025,
    "doi": "10.3390/pr13072034",
}

# Each point is (T/K, p/kPa). The reported u(p) is not used because the paper's
# own regression objective is an unweighted relative-pressure sum of squares.
TABLE_3 = {
    0.1023: [
        (326.15, 5.823),
        (330.65, 7.453),
        (335.15, 9.457),
        (339.65, 11.902),
        (344.15, 14.890),
        (348.65, 18.421),
        (353.15, 22.864),
        (357.65, 27.861),
        (362.15, 34.109),
        (366.65, 40.874),
        (371.15, 49.525),
        (375.65, 58.998),
        (380.15, 70.068),
        (384.65, 82.903),
        (389.15, 97.029),
    ],
    0.3007: [
        (324.15, 5.756),
        (328.65, 7.453),
        (333.15, 9.559),
        (337.65, 11.935),
        (342.15, 14.856),
        (346.65, 18.557),
        (351.15, 22.869),
        (355.65, 28.065),
        (360.15, 34.007),
        (364.65, 41.477),
        (369.15, 49.898),
        (373.65, 59.610),
        (378.15, 70.645),
        (382.65, 83.854),
        (387.15, 98.998),
    ],
    0.5011: [
        (322.15, 5.857),
        (326.65, 7.453),
        (331.15, 9.457),
        (335.65, 11.935),
        (340.15, 14.958),
        (344.65, 18.625),
        (349.15, 22.835),
        (353.65, 28.166),
        (358.15, 34.211),
        (362.65, 41.511),
        (367.15, 49.864),
        (371.65, 59.508),
        (376.15, 71.426),
        (380.65, 84.703),
        (385.15, 98.964),
    ],
    0.7006: [
        (320.15, 5.688),
        (324.65, 7.317),
        (329.15, 9.253),
        (333.65, 11.460),
        (338.15, 14.788),
        (342.65, 18.014),
        (347.15, 22.733),
        (351.65, 28.031),
        (356.15, 34.244),
        (360.65, 41.409),
        (365.15, 50.204),
        (369.65, 59.983),
        (374.15, 71.528),
        (378.65, 84.941),
        (383.15, 99.779),
    ],
    0.9027: [
        (318.15, 5.318),
        (322.65, 6.920),
        (327.15, 9.049),
        (331.65, 11.222),
        (336.15, 14.414),
        (340.65, 17.946),
        (345.15, 22.122),
        (349.65, 26.740),
        (354.15, 33.396),
        (358.65, 40.594),
        (363.15, 49.321),
        (367.65, 59.440),
        (372.15, 71.154),
        (376.65, 84.601),
        (381.15, 99.474),
    ],
}

TABLE_4 = {
    0.1085: [
        (349.15, 4.126),
        (354.15, 5.212),
        (359.15, 6.299),
        (364.15, 7.623),
        (369.15, 9.205),
        (374.15, 11.306),
        (379.15, 13.833),
        (384.15, 16.745),
        (389.15, 20.594),
        (394.15, 24.771),
        (399.15, 28.947),
        (404.15, 34.958),
        (409.15, 42.326),
        (414.15, 50.204),
        (419.15, 58.659),
        (424.15, 67.385),
        (429.15, 75.942),
        (434.15, 87.046),
        (439.15, 98.320),
    ],
    0.3014: [
        (333.15, 4.092),
        (338.15, 5.518),
        (343.15, 6.978),
        (348.15, 8.540),
        (353.15, 10.679),
        (358.15, 13.531),
        (363.15, 16.655),
        (368.15, 20.153),
        (373.15, 24.329),
        (378.15, 29.762),
        (383.15, 35.942),
        (388.15, 43.480),
        (393.15, 51.528),
        (398.15, 60.968),
        (403.15, 71.460),
        (408.15, 83.990),
    ],
    0.5017: [
        (328.15, 4.974),
        (333.15, 6.469),
        (338.15, 8.370),
        (343.15, 10.781),
        (348.15, 13.362),
        (353.15, 16.825),
        (358.15, 21.002),
        (363.15, 26.299),
        (368.15, 32.445),
        (373.15, 39.847),
        (378.15, 48.438),
        (383.15, 58.217),
        (388.15, 70.034),
        (393.15, 83.413),
        (398.15, 97.844),
    ],
    0.7041: [
        (319.15, 4.160),
        (324.15, 5.450),
        (329.15, 7.284),
        (334.15, 9.524),
        (339.15, 12.003),
        (344.15, 15.433),
        (349.15, 19.508),
        (354.15, 24.465),
        (359.15, 30.170),
        (364.15, 37.063),
        (369.15, 45.552),
        (374.15, 55.229),
        (379.15, 66.876),
        (384.15, 82.292),
        (389.15, 96.893),
    ],
    0.9006: [
        (315.15, 4.126),
        (319.15, 5.246),
        (323.15, 6.570),
        (327.15, 8.098),
        (331.15, 10.102),
        (335.15, 12.445),
        (339.15, 15.365),
        (343.15, 18.693),
        (347.15, 22.598),
        (351.15, 27.012),
        (355.15, 32.488),
        (359.15, 38.387),
        (363.15, 45.246),
        (367.15, 53.396),
        (371.15, 62.564),
        (375.15, 73.090),
        (379.15, 85.008),
        (383.15, 97.776),
    ],
}

PURE_DATA = {
    "1-butanol": [
        (325.15, 5.174),
        (329.65, 6.658),
        (334.15, 8.495),
        (338.65, 10.750),
        (343.15, 13.499),
        (347.65, 16.823),
        (352.15, 20.816),
        (356.65, 25.580),
        (361.15, 31.228),
        (365.65, 37.881),
        (370.15, 45.674),
        (374.65, 54.749),
        (379.15, 65.259),
        (383.65, 77.369),
        (388.15, 91.252),
    ],
    "isobutanol": [
        (305.15, 2.284),
        (309.65, 3.086),
        (314.15, 4.120),
        (318.65, 5.438),
        (323.15, 7.101),
        (327.65, 9.178),
        (332.15, 11.746),
        (336.65, 14.895),
        (341.15, 18.721),
        (345.65, 23.333),
        (350.15, 28.850),
        (354.65, 35.399),
        (359.15, 43.119),
        (363.65, 52.161),
        (368.15, 62.683),
    ],
    "2-ethyl-1-hexanol": [
        (370.15, 3.635),
        (375.15, 4.699),
        (380.15, 6.011),
        (385.15, 7.613),
        (390.15, 9.553),
        (395.15, 11.880),
        (400.15, 14.651),
        (405.15, 17.924),
        (410.15, 21.762),
        (415.15, 26.234),
        (420.15, 31.409),
        (425.15, 37.362),
        (430.15, 44.171),
        (435.15, 51.918),
        (440.15, 60.686),
        (445.15, 70.563),
        (450.15, 81.640),
        (455.15, 94.008),
    ],
}

SYSTEMS = {
    "isobutanol_1_butanol": {
        "source_index": 19,
        "table": "Table 3",
        "components": ("isobutanol", "1-butanol"),
        "cas": ("78-83-1", "71-36-3"),
        "rq": ((3.4535, 3.4543), (3.048, 3.052)),
        "blocks": TABLE_3,
        "fit_exclusions": [],
        "published_nrtl": {
            "form": "B_over_T_fitted_alpha",
            "parameters": [-2352.3672, 2143.2572, -0.01532],
        },
    },
    "isobutanol_2_ethyl_1_hexanol": {
        "source_index": 20,
        "table": "Table 4",
        "components": ("isobutanol", "2-ethyl-1-hexanol"),
        "cas": ("78-83-1", "104-76-7"),
        "rq": ((3.4535, 6.1511), (3.048, 5.208)),
        "blocks": TABLE_4,
        "fit_exclusions": [
            {
                "x1": 0.9006,
                "T_K": 315.15,
                "p_kPa": 4.126,
                "reason": (
                    "T is 1.264 K below PFDSim's qualified canonical "
                    "2-ethyl-1-hexanol Psat range."
                ),
            }
        ],
        "published_nrtl": {
            "form": "A_plus_B_over_T_plus_C_over_T2_variable_alpha",
            "parameters": {
                "tau12": [-0.31998, -141.92441, 60605.14069],
                "tau21": [0.00301, -75.28924, 62224.75031],
                "alpha": [-0.87252, 0.00574],
            },
        },
    },
}

NRTL_SPECS = {
    "zero": (np.array([]), np.array([]), np.array([])),
    "B_over_T_fixed_alpha": (np.zeros(2), np.full(2, -5000.0), np.full(2, 5000.0)),
    "A_plus_B_over_T_fixed_alpha": (
        np.zeros(4),
        np.array([-10.0, -5000.0, -10.0, -5000.0]),
        np.array([10.0, 5000.0, 10.0, 5000.0]),
    ),
}
UNIQUAC_SPECS = {
    "zero": NRTL_SPECS["zero"],
    "B_over_T": NRTL_SPECS["B_over_T_fixed_alpha"],
    "A_plus_B_over_T": NRTL_SPECS["A_plus_B_over_T_fixed_alpha"],
}


@lru_cache(maxsize=None)
def _pure_antoine(component: str) -> tuple[float, float, float]:
    """Regress log10(P/kPa) = A - B/(T/K + C) to Fendu's pure data."""
    rows = PURE_DATA[component]
    temperatures = np.asarray([row[0] for row in rows])
    log_pressures = np.log10([row[1] for row in rows])
    result = least_squares(
        lambda values: (
            values[0] - values[1] / (temperatures + values[2]) - log_pressures
        ),
        np.asarray([7.0, 1500.0, -50.0]),
        max_nfev=500,
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
    )
    if not result.success:
        raise RuntimeError(
            f"{component} source-local Antoine regression failed: {result.message}"
        )
    return tuple(float(value) for value in result.x)


@lru_cache(maxsize=None)
def psat_kpa(component: str, temperature: float) -> float:
    a, b, c = _pure_antoine(component)
    return 10.0 ** (a - b / (float(temperature) + c))


def _rows(system: dict, omitted_composition: float | None = None) -> list[tuple]:
    excluded = {
        (item["x1"], item["T_K"], item["p_kPa"]) for item in system["fit_exclusions"]
    }
    return [
        (x1, temperature, pressure)
        for x1, points in system["blocks"].items()
        if x1 != omitted_composition
        for temperature, pressure in points
        if (x1, temperature, pressure) not in excluded
    ]


def _nrtl_parameters(form: str, values: np.ndarray, temperature: float) -> tuple:
    if form == "zero":
        return 0.0, 0.0, 0.3
    if form == "B_over_T_fixed_alpha":
        return values[0] / temperature, values[1] / temperature, 0.3
    return (
        values[0] + values[1] / temperature,
        values[2] + values[3] / temperature,
        0.3,
    )


def _uniquac_parameters(form: str, values: np.ndarray, temperature: float) -> tuple:
    if form == "zero":
        return 1.0, 1.0
    if form == "B_over_T":
        return math.exp(values[0] / temperature), math.exp(values[1] / temperature)
    return (
        math.exp(values[0] + values[1] / temperature),
        math.exp(values[2] + values[3] / temperature),
    )


def _pressure_residuals(
    model: str,
    form: str,
    values: np.ndarray,
    rows: list[tuple],
    system: dict,
) -> np.ndarray:
    comp1, comp2 = system["components"]
    residuals = []
    for x1, temperature, pressure in rows:
        if model == "NRTL":
            tau12, tau21, alpha = _nrtl_parameters(form, values, temperature)
            ln_gamma1, ln_gamma2 = _nrtl_ln_gamma(x1, tau12, tau21, alpha)
        else:
            tau12, tau21 = _uniquac_parameters(form, values, temperature)
            ln_gamma1, ln_gamma2 = _uniquac_ln_gamma(
                x1, system["rq"][0], system["rq"][1], tau12, tau21
            )
        calculated = x1 * math.exp(ln_gamma1) * psat_kpa(comp1, temperature) + (
            1.0 - x1
        ) * math.exp(ln_gamma2) * psat_kpa(comp2, temperature)
        residuals.append(calculated / pressure - 1.0)
    return np.asarray(residuals)


def _fit(
    model: str,
    form: str,
    rows: list[tuple],
    system: dict,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    specs = NRTL_SPECS if model == "NRTL" else UNIQUAC_SPECS
    x0, lower, upper = specs[form]
    if not len(x0):
        return x0
    result = least_squares(
        lambda values: _pressure_residuals(model, form, values, rows, system),
        x0 if initial is None else initial,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=750,
    )
    if not result.success:
        raise RuntimeError(f"{model} {form} regression failed: {result.message}")
    return result.x


def _metrics(residuals: np.ndarray, parameter_count: int) -> dict:
    count = len(residuals)
    rss = float(residuals @ residuals)
    aic = count * math.log(max(rss / count, 1.0e-300)) + 2 * parameter_count
    correction = (
        2 * parameter_count * (parameter_count + 1) / (count - parameter_count - 1)
        if parameter_count
        else 0.0
    )
    return {
        "point_count": count,
        "pressure_AARD_percent": 100.0 * float(np.mean(np.abs(residuals))),
        "pressure_RMSE_percent": 100.0 * float(np.sqrt(np.mean(residuals**2))),
        "pressure_max_abs_percent": 100.0 * float(np.max(np.abs(residuals))),
        "pressure_bias_percent": 100.0 * float(np.mean(residuals)),
        "AICc": aic + correction,
    }


def _fit_candidate(model: str, form: str, system: dict) -> dict:
    all_rows = _rows(system)
    values = _fit(model, form, all_rows, system)
    result = {
        "parameters": values.tolist(),
        "fit": _metrics(
            _pressure_residuals(model, form, values, all_rows, system),
            len(values),
        ),
    }
    held_out = []
    for composition in system["blocks"]:
        training = _rows(system, omitted_composition=composition)
        testing = [row for row in all_rows if row[0] == composition]
        if not testing:
            continue
        local_values = _fit(model, form, training, system, values)
        held_out.extend(_pressure_residuals(model, form, local_values, testing, system))
    result["leave_one_composition_out"] = _metrics(np.asarray(held_out), len(values))
    return result


def _published_nrtl_metrics(system: dict) -> dict:
    source = system["published_nrtl"]
    rows = _rows(system)
    residuals = []
    comp1, comp2 = system["components"]
    for x1, temperature, pressure in rows:
        if source["form"] == "B_over_T_fitted_alpha":
            d12, d21, alpha = source["parameters"]
            tau12, tau21 = d12 / temperature, d21 / temperature
        else:
            parameters = source["parameters"]
            a12, b12, c12 = parameters["tau12"]
            a21, b21, c21 = parameters["tau21"]
            alpha_a, alpha_b = parameters["alpha"]
            tau12 = a12 + b12 / temperature + c12 / temperature**2
            tau21 = a21 + b21 / temperature + c21 / temperature**2
            alpha = alpha_a + alpha_b * temperature
        ln_gamma1, ln_gamma2 = _nrtl_ln_gamma(x1, tau12, tau21, alpha)
        calculated = x1 * math.exp(ln_gamma1) * psat_kpa(comp1, temperature) + (
            1 - x1
        ) * math.exp(ln_gamma2) * psat_kpa(comp2, temperature)
        residuals.append(calculated / pressure - 1)
    return _metrics(np.asarray(residuals), 3 if system["source_index"] == 19 else 8)


def _pure_component_metrics(components: tuple[str, str]) -> dict:
    result = {}
    for component in components:
        errors = np.asarray(
            [
                psat_kpa(component, temperature) / pressure - 1.0
                for temperature, pressure in PURE_DATA[component]
            ]
        )
        result[component] = _metrics(errors, 0)
    return result


def _interaction_records(system_name: str, system: dict, fits: dict) -> list[dict]:
    rows = _rows(system)
    temperatures = [row[1] for row in rows]
    nrtl = fits["NRTL"]["A_plus_B_over_T_fixed_alpha"]
    uniquac = fits["UNIQUAC"]["A_plus_B_over_T"]
    nrtl_values = nrtl["parameters"]
    uniquac_values = uniquac["parameters"]
    common = {
        "cas1": system["cas"][0],
        "cas2": system["cas"][1],
        "component1": system["components"][0],
        "component2": system["components"][1],
        "Tmin_K": min(temperatures),
        "Tmax_K": max(temperatures),
        "extrapolation": "inverse_linear_quadratic",
        "source": "Fendu and Pricop-Nicolae 2025 PFDSim p-T-x refit",
        "source_doi": SOURCE["doi"],
        "fit_status": "recommended_ptx_refit",
        "fit_system": system_name,
        "fit_evidence": {
            "selected_model": "A+B/T",
            "objective": "unweighted relative pressure residuals",
            "fit": nrtl["fit"],
            "leave_one_composition_out": nrtl["leave_one_composition_out"],
            "published_nrtl_reproduction": fits["published_nrtl_reproduction"],
        },
    }
    return [
        {
            **common,
            "model": "NRTL",
            "comment": (
                f"{system['components'][0]} + {system['components'][1]} "
                "Fendu 2025 PFDSim NRTL p-T-x refit"
            ),
            "alpha12": 0.3,
            "tau12_c": nrtl_values[0],
            "tau12_d": nrtl_values[1],
            "tau21_c": nrtl_values[2],
            "tau21_d": nrtl_values[3],
        },
        {
            **common,
            "model": "UNIQUAC",
            "comment": (
                f"{system['components'][0]} + {system['components'][1]} "
                "Fendu 2025 PFDSim UNIQUAC p-T-x refit"
            ),
            "model_variant": "standard_uniquac",
            "use_q_prime": False,
            "tau12_a": uniquac_values[0],
            "tau12_b": uniquac_values[1],
            "tau21_a": uniquac_values[2],
            "tau21_b": uniquac_values[3],
            "fit_evidence": {
                "selected_model": "A+B/T",
                "objective": "unweighted relative pressure residuals",
                "fit": uniquac["fit"],
                "leave_one_composition_out": uniquac["leave_one_composition_out"],
                "paper_uniquac_status": (
                    "The paper reports that its UNIQUAC regression had large "
                    "deviations and does not publish those parameters."
                ),
            },
        },
    ]


def build_payload() -> dict:
    analyses = {}
    interactions = []
    for name, system in SYSTEMS.items():
        fits = {
            "NRTL": {form: _fit_candidate("NRTL", form, system) for form in NRTL_SPECS},
            "UNIQUAC": {
                form: _fit_candidate("UNIQUAC", form, system) for form in UNIQUAC_SPECS
            },
            "published_nrtl_reproduction": _published_nrtl_metrics(system),
        }
        analyses[name] = {
            "source_index": system["source_index"],
            "source_table": system["table"],
            "components": list(system["components"]),
            "cas": list(system["cas"]),
            "uniquac_rq": {
                system["components"][0]: {
                    "r": system["rq"][0][0],
                    "q": system["rq"][1][0],
                },
                system["components"][1]: {
                    "r": system["rq"][0][1],
                    "q": system["rq"][1][1],
                },
            },
            "mixed_composition_data": {
                str(composition): [
                    {"T_K": temperature, "p_kPa": pressure}
                    for temperature, pressure in points
                ]
                for composition, points in system["blocks"].items()
            },
            "fit_exclusions": system["fit_exclusions"],
            "pure_component_validation": _pure_component_metrics(system["components"]),
            "candidate_fits": fits,
        }
        interactions.extend(_interaction_records(name, system, fits))
    return {
        "metadata": {
            "description": (
                "PFDSim refits of Fendu 2025 Tables 3 and 4 to compact, "
                "exactly supported NRTL and UNIQUAC forms."
            ),
            "source": SOURCE,
            "objective": "sum((Pcalc/Pexp)-1)^2",
            "validation": "leave one complete liquid-composition series out",
            "pure_component_pressure_basis": {
                "equation": "log10(P/kPa) = A - B/(T/K + C)",
                "source": "Fendu 2025 pure-component pressure tables embedded in this script",
                "coefficients": {
                    component: dict(zip(("A", "B_K", "C_K"), _pure_antoine(component)))
                    for component in PURE_DATA
                },
            },
            "selection": (
                "A+B/T selected over zero and B/T using AICc plus "
                "composition-held-out pressure errors."
            ),
            "script": "scripts/activity_fitting/fit_fendu_2025_activity_parameters.py",
        },
        "analyses": analyses,
        "interactions": interactions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        type=Path,
        help="Write the complete deterministic JSON payload to this path.",
    )
    parser.add_argument(
        "--check",
        type=Path,
        help="Refit and compare with an existing JSON payload.",
    )
    args = parser.parse_args()
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
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
