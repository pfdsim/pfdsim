#!/usr/bin/env python3
"""Reproduce 3-bar azeotropes and compare joint butanol/water activity fits.

The input snapshot is explicit JSON. The script never publishes a fit or writes
runtime interaction tables. Random fitting starts use the request's fixed seed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from thermodynamics_models.factory import create_thermodynamics
from thermodynamics_models.interaction_fitting import FORMS, prepare_fit


NRTL_SEED = {
    "12.constant": 3.07626601, "12.inverse": -489.80683594,
    "12.anchored": -60.05225941, "12.linear": 0.0, "12.quadratic": 0.0,
    "21.constant": 4.36302560, "21.inverse": -241.22842207,
    "21.anchored": -8.71391481, "21.linear": 0.0, "21.quadratic": 0.0,
}
UNIQUAC_SEED = {
    "12.constant": -12.81485434, "12.inverse": 1905.74507135,
    "12.anchored": 0.0, "12.linear": 0.02169653866, "12.quadratic": 0.0,
    "21.constant": 8.68179405, "21.inverse": -1368.13065752,
    "21.anchored": 0.0, "21.linear": -0.01678360690, "21.quadratic": 0.0,
}
EXTRAPOLATION_POLICIES = (
    "unrestricted", "constant_inverse", "inverse_linear_quadratic",
    "inverse_square_cubic",
)


def collapse_vlle_plateau(request):
    """Replace the discontinuity pair between two VLE branches by one invariant."""
    rows = request["observations"]
    vle = [row for row in rows if row["kind"] == "VLE" and not row.get("validation_only")]
    candidates = [
        (abs(right["x1"] - left["x1"]), [left, right])
        for left, right in zip(vle, vle[1:])
        if left["P_bar"] == right["P_bar"]
        and abs(left["T_K"] - right["T_K"]) <= 0.35
        and abs(left["y1"] - right["y1"]) <= 0.012
        and abs(right["x1"] - left["x1"]) >= 0.3
    ]
    if not candidates:
        raise ValueError("No constant-pressure VLLE plateau was identified.")
    _, best = max(candidates)
    removed = {row["id"] for row in best}
    invariant = {
        "id": "reconstructed-vlle",
        "kind": "VLLE",
        "T_K": float(np.median([row["T_K"] for row in best])),
        "P_bar": best[0]["P_bar"],
        "y1": float(np.median([row["y1"] for row in best])),
        "weight": 1.0,
        "source": "Collapsed from plateau samples " + ", ".join(sorted(removed)),
    }
    request["observations"] = [row for row in rows if row["id"] not in removed] + [
        invariant
    ]
    return {
        "removed_ids": sorted(removed),
        "invariant": invariant,
        "liquid_composition_span": [
            min(row["x1"] for row in best), max(row["x1"] for row in best)
        ],
    }


def runtime_thermo(model, parameters, extrapolation):
    record = {
        **parameters, "model": model, "component1": "1-butanol",
        "component2": "water", "extrapolation": extrapolation,
    }
    return create_thermodynamics(
        ["1-butanol", "water"], model, interaction_overrides=[record]
    )


def stable_vle_state(thermo, pressure, x, guess):
    composition = {"1-butanol": x, "water": 1 - x}
    temperature = thermo.bubble_point_T(composition, pressure, T_guess=guess)
    split, _, _, _ = thermo.liquid_liquid_equilibrium(
        composition, temperature, tol=1e-8
    )
    K = thermo.K_values(temperature, pressure, composition)
    vapor = {component: composition[component] * K[component] for component in composition}
    y = vapor["1-butanol"] / sum(vapor.values())
    return float(temperature), float(y), not split


def stable_azeotropes(thermo, pressure, *, points=101):
    grid = np.linspace(1e-4, 1 - 1e-4, points)
    samples, guess = [], 410.0
    for x in grid:
        try:
            temperature, y, stable = stable_vle_state(thermo, pressure, float(x), guess)
            guess = temperature
            samples.append((float(x), temperature, y, stable))
        except Exception as error:
            samples.append((float(x), None, None, False, str(error)))
    roots = []
    for left, right in zip(samples, samples[1:]):
        if len(left) != 4 or len(right) != 4 or not left[3] or not right[3]:
            continue
        f_left, f_right = left[2] - left[0], right[2] - right[0]
        if f_left * f_right >= 0:
            continue

        def difference(x):
            _, y, stable = stable_vle_state(
                thermo, pressure, x, 0.5 * (left[1] + right[1])
            )
            if not stable:
                raise ValueError("Azeotrope bracket entered a two-liquid state")
            return y - x

        try:
            root = brentq(difference, left[0], right[0], xtol=1e-10)
            temperature, y, stable = stable_vle_state(
                thermo, pressure, root, 0.5 * (left[1] + right[1])
            )
            delta = min(0.005, root / 2, (1 - root) / 2)
            lower = stable_vle_state(thermo, pressure, root - delta, temperature)[0]
            upper = stable_vle_state(thermo, pressure, root + delta, temperature)[0]
            kind = (
                "maximum" if temperature > max(lower, upper)
                else "minimum" if temperature < min(lower, upper)
                else "stationary"
            )
            roots.append({
                "x_butanol": float(root), "y_butanol": y, "T_K": temperature,
                "type": kind, "stable": stable,
            })
        except Exception:
            continue
    return roots


def metric(report, kind, quantity, field="RMSE"):
    return report.get("physical_metrics", {}).get(kind, {}).get(quantity, {}).get(field)


def projected_seed(seed, form, temperatures, tref, coordinate_limit=25):
    """Project the reference law into a candidate form in scaled coordinates."""
    temperatures = np.unique(
        np.r_[temperatures, np.linspace(min(temperatures), max(temperatures), 25)]
    )
    u = temperatures / tref
    columns = {
        "constant": np.ones_like(u),
        "inverse": 1 / u,
        "anchored": 1 / u - 1 + np.log(u),
        "linear": u,
        "quadratic": u * u,
    }
    physical_to_basis = {
        "constant": 1.0,
        "inverse": 1 / tref,
        "anchored": 1.0,
        "linear": tref,
        "quadratic": tref * tref,
    }
    result = {}
    for direction in ("12", "21"):
        target = sum(
            seed[f"{direction}.{term}"] * physical_to_basis[term] * columns[term]
            for term in ("constant", "inverse", "anchored", "linear", "quadratic")
        )
        basis = np.column_stack([columns[term] for term in FORMS[form]])
        coordinates = np.linalg.lstsq(basis, target, rcond=1e-10)[0]
        if coordinate_limit is not None:
            coordinates = np.clip(coordinates, -coordinate_limit, coordinate_limit)
        result.update(
            {
                f"{direction}.{term}": float(value / physical_to_basis[term])
                for term, value in zip(FORMS[form], coordinates)
            }
        )
    return result


def case_request(snapshot, model, form, alpha_mode, starts, max_nfev, unbounded=False):
    request = deepcopy(snapshot["request"])
    request.update(
        model=model, form=form, starts=starts, max_nfev=max_nfev, seed=1729,
        bounds={}, cv={"method": "none", "folds": 5}, extrapolation="unrestricted",
    )
    if model == "NRTL":
        request.pop("rq", None)
        requested_alpha = (
            float(alpha_mode.removeprefix("fixed_"))
            if alpha_mode.startswith("fixed_")
            else 0.2
        )
        # normalize_fit_request enforces the standard NRTL range. Negative and
        # zero alpha values are installed only inside this research probe.
        request["alpha"] = min(max(requested_alpha, 0.01), 1.0)
        request["fit_alpha"] = alpha_mode == "fitted"
        seed = NRTL_SEED
    else:
        request["fit_alpha"] = False
        seed = UNIQUAC_SEED
    temperatures = [
        row["T_K"] for row in request["observations"] if not row.get("validation_only")
    ]
    if (
        model == "NRTL"
        and alpha_mode.startswith("fixed_")
        and form == "constant_inverse_anchored"
    ):
        request["initial"] = {
            f"{direction}.{term}": seed[f"{direction}.{term}"]
            for direction in ("12", "21") for term in FORMS[form]
        }
        request["bounds"] = {
            "12.anchored": [-100, 100],
            "21.anchored": [-100, 100],
        }
    else:
        request["initial"] = projected_seed(
            seed, form, temperatures, request.get("T_ref_K", 298.15),
            coordinate_limit=None if unbounded else 25,
        )
    if unbounded:
        request["bounds"] = {
            name: [-1e6, 1e6]
            for name in request["initial"]
            if name.startswith(("12.", "21."))
        }
    if request["fit_alpha"]:
        request["initial"]["alpha12"] = 0.45131325
    return request


def fit_case(arguments):
    snapshot, model, form, alpha_mode, starts, max_nfev, unbounded = arguments
    label = f"{model}:{alpha_mode}:{form}"
    try:
        problem = prepare_fit(case_request(
            snapshot, model, form, alpha_mode, starts, max_nfev, unbounded
        ))
        if model == "NRTL" and alpha_mode.startswith("fixed_"):
            problem.request["alpha"] = float(alpha_mode.removeprefix("fixed_"))
            problem.last = None
        if unbounded:
            for index, name in enumerate(problem.names):
                if name.startswith(("12.", "21.")) or name == "alpha12":
                    problem.lower[index], problem.upper[index] = -np.inf, np.inf
        rows = problem.request["observations"]
        fitted = problem.solve(rows)
        values = fitted.pop("values")
        report = problem.report(values, rows)
        parameters = problem.parameters(values)
        coefficients = dict(
            zip(problem.names, (values * np.asarray(problem.scales)).tolist())
        )
        failed = [
            (point["id"], point["kind"])
            for point in report["points"] if not point["physical"]
        ]
        try:
            invariant = problem.predict_vlle(P=1.0133, T_guess=366)
        except Exception as error:
            invariant = {"error": str(error)}
        extrapolation = {
            policy: stable_azeotropes(runtime_thermo(model, parameters, policy), 3.0)
            for policy in EXTRAPOLATION_POLICIES
        }
        return {
            "case": label, "model": model, "form": form,
            "alpha_mode": alpha_mode, "objective": fitted["objective"],
            "optimizer_success": fitted["success"], "nfev": fitted["nfev"],
            "rank": fitted["rank"], "parameter_count": len(problem.names),
            "failed_observations": failed,
            "bounds": [
                name for index, name in enumerate(problem.names)
                if min(values[index] - problem.lower[index],
                       problem.upper[index] - values[index]) < 1e-5
            ],
            "errors": {
                "LLE_alpha_RMSE": metric(report, "LLE", "x1_alpha"),
                "LLE_beta_RMSE": metric(report, "LLE", "x1_beta"),
                "VLE_T_RMSE_K": metric(report, "VLE", "T_K"),
                "VLE_y_RMSE": metric(report, "VLE", "y1"),
                "VLE_P_RMSE_bar": metric(report, "VLE", "P_bar"),
                "gamma1_relative_RMSE_percent": metric(
                    report, "GAMMA_INF", "gamma1_inf_relative_percent"
                ),
                "gamma2_relative_RMSE_percent": metric(
                    report, "GAMMA_INF", "gamma2_inf_relative_percent"
                ),
                "UCST_T_abs_K": metric(report, "UCST", "T_K", "MAE"),
                "UCST_x_abs": metric(report, "UCST", "x1", "MAE"),
            },
            "one_atm_invariant": invariant,
            "three_bar_azeotropes": extrapolation,
            "coefficients": coefficients, "parameters": parameters,
            "failures": fitted["failures"],
        }
    except Exception as error:
        return {"case": label, "error": f"{type(error).__name__}: {error}"}


def reproduce(snapshot, pressure):
    record = snapshot["published_nrtl"]
    return {
        "pressure_bar": pressure,
        "azeotropes": {
            policy: stable_azeotropes(runtime_thermo("NRTL", record, policy), pressure)
            for policy in EXTRAPOLATION_POLICIES
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--pressure", type=float, default=3.0)
    parser.add_argument("--forms", nargs="+", choices=list(FORMS), default=list(FORMS))
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--collapse-vlle-plateau", action="store_true")
    parser.add_argument("--nrtl-alpha-grid", nargs="+", type=float)
    parser.add_argument("--zero-weight-kinds", nargs="+", default=[])
    parser.add_argument("--unbounded", action="store_true")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    plateau = (
        collapse_vlle_plateau(snapshot["request"])
        if args.collapse_vlle_plateau
        else None
    )
    if plateau:
        print(json.dumps({"plateau_reconstruction": plateau}), flush=True)
    for kind in args.zero_weight_kinds:
        snapshot["request"]["weights"][kind] = 0.0
    if args.reproduce:
        print(json.dumps({"reproduction": reproduce(snapshot, args.pressure)}), flush=True)
    if args.fit:
        cases = []
        for form in args.forms:
            if args.nrtl_alpha_grid is not None:
                cases.extend(
                    (
                        snapshot, "NRTL", form, f"fixed_{alpha:g}",
                        args.starts, args.max_nfev, args.unbounded,
                    )
                    for alpha in args.nrtl_alpha_grid
                )
            else:
                cases.extend([
                    (snapshot, "NRTL", form, "fixed_0.2", args.starts, args.max_nfev, args.unbounded),
                    (snapshot, "NRTL", form, "fitted", args.starts, args.max_nfev, args.unbounded),
                    (snapshot, "UNIQUAC", form, "structural", args.starts, args.max_nfev, args.unbounded),
                ])
        results = []
        with ProcessPoolExecutor(max_workers=min(max(args.workers, 1), 5)) as pool:
            futures = [pool.submit(fit_case, case) for case in cases]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result), flush=True)
        print(
            json.dumps({"results": sorted(results, key=lambda item: item["case"])}),
            flush=True,
        )


if __name__ == "__main__":
    main()
