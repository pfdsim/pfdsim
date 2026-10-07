#!/usr/bin/env python3
"""Probe flexible Redlich–Kister activity laws for the joint butanol dataset.

This is a research probe, not a runtime model. It fits thermodynamically
consistent excess-Gibbs polynomials, checks stable binodals, and audits 3-bar
azeotropes with both unrestricted and clamped temperature extrapolation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from thermodynamics_models.interaction_fitting import prepare_fit
from scripts.activity_fitting.probe_butanol_joint_models import collapse_vlle_plateau


class RedlichKister:
    def __init__(self, problem, composition_degree, temperature_degree, temperatures):
        self.problem = problem
        self.nc = composition_degree + 1
        self.nt = temperature_degree + 1
        self.tmin, self.tmax = min(temperatures), max(temperatures)

    def temperature_basis(self, T, clamp=False):
        T = np.asarray(T, dtype=float)
        z = (2 * T - self.tmin - self.tmax) / (self.tmax - self.tmin)
        if clamp:
            z = np.clip(z, -1, 1)
        values = [np.ones_like(z)]
        if self.nt > 1:
            values.append(z)
        for order in range(2, self.nt):
            values.append(2 * z * values[-1] - values[-2])
        return np.asarray(values)

    def interaction(self, coefficients, T, clamp=False):
        matrix = np.asarray(coefficients[: self.nc * self.nt]).reshape(self.nc, self.nt)
        return matrix @ self.temperature_basis(T, clamp=clamp)

    def log_gamma(self, coefficients, T, x, clamp=False):
        x = np.asarray(x, dtype=float)
        A = self.interaction(coefficients, T, clamp=clamp)
        s = 2 * x - 1
        powers = np.asarray([s**order for order in range(self.nc)])
        polynomial = np.tensordot(A, powers, axes=(0, 0))
        derivative = sum(
            order * A[order] * s ** (order - 1) for order in range(1, self.nc)
        )
        gex = x * (1 - x) * polynomial
        dgdx = (1 - 2 * x) * polynomial + 2 * x * (1 - x) * derivative
        return np.asarray([gex + (1 - x) * dgdx, gex - x * dgdx])

    def chemical_potentials(self, coefficients, T, x, clamp=False):
        x = np.asarray(x, dtype=float)
        return self.log_gamma(coefficients, T, x, clamp=clamp) + np.asarray(
            [np.log(np.maximum(x, 1e-300)), np.log(np.maximum(1 - x, 1e-300))]
        )

    def gibbs(self, coefficients, T, x, clamp=False):
        x = np.asarray(x, dtype=float)
        ideal = x * np.log(np.maximum(x, 1e-300)) + (1 - x) * np.log(
            np.maximum(1 - x, 1e-300)
        )
        gamma = self.log_gamma(coefficients, T, x, clamp=clamp)
        return ideal + x * gamma[0] + (1 - x) * gamma[1]

    def critical_derivatives(self, coefficients, T, x):
        h = min(0.001, x / 4, (1 - x) / 4)
        mu = self.chemical_potentials(
            coefficients, T, x + h * np.arange(-2, 3)
        )
        difference = mu[0] - mu[1]
        curvature = (
            difference[0] - 8 * difference[1] + 8 * difference[3] - difference[4]
        ) / (12 * h)
        third = (
            -difference[0] + 16 * difference[1] - 30 * difference[2]
            + 16 * difference[3] - difference[4]
        ) / (12 * h * h)
        fourth = (
            -difference[0] + 2 * difference[1] - 2 * difference[3] + difference[4]
        ) / (2 * h**3)
        return float(curvature), float(third), float(fourth)

    def stable_binodal(self, coefficients, T):
        grid = np.linspace(1e-5, 1 - 1e-5, 2001)
        values = self.gibbs(coefficients, T, grid)
        hull = []
        for index in range(len(grid)):
            while len(hull) >= 2:
                first, second = hull[-2:]
                left = (values[second] - values[first]) / (grid[second] - grid[first])
                right = (values[index] - values[second]) / (grid[index] - grid[second])
                if left < right:
                    break
                hull.pop()
            hull.append(index)
        gaps = [(right - left, left, right) for left, right in zip(hull, hull[1:]) if right - left > 1]
        if not gaps:
            return None
        _, left, right = max(gaps)

        def residual(pair):
            low, high = sorted(pair)
            mu_low = self.chemical_potentials(coefficients, T, low)
            mu_high = self.chemical_potentials(coefficients, T, high)
            return mu_low - mu_high

        root = least_squares(
            residual, [grid[left], grid[right]],
            bounds=([1e-7, 1e-7], [1 - 1e-7, 1 - 1e-7]),
            max_nfev=100, ftol=1e-12, xtol=1e-12, gtol=1e-10,
        )
        low, high = sorted(root.x)
        if high - low < 1e-4 or max(abs(root.fun)) > 1e-6:
            return None
        mu = 0.5 * (
            self.chemical_potentials(coefficients, T, low)
            + self.chemical_potentials(coefficients, T, high)
        )
        tangent_gap = self.gibbs(coefficients, T, grid) - (
            grid * mu[0] + (1 - grid) * mu[1]
        )
        if min(tangent_gap) < -1e-6:
            return None
        return float(low), float(high)

    def bubble_state(self, coefficients, pressure, x, guess=400, clamp=False):
        def equation(T):
            gamma = np.exp(self.log_gamma(coefficients, T, x, clamp=clamp))
            return (
                x * gamma[0] * self.problem.thermo.Psat(self.problem.components[0], T)
                + (1 - x) * gamma[1] * self.problem.thermo.Psat(self.problem.components[1], T)
                - pressure
            )

        grid = np.linspace(max(280, guess - 100), min(550, guess + 100), 81)
        values = [equation(T) for T in grid]
        bracket = next(
            ((a, b) for a, b, fa, fb in zip(grid, grid[1:], values, values[1:]) if fa * fb <= 0),
            None,
        )
        if bracket is None:
            raise ValueError("No bubble-temperature root")
        T = brentq(equation, *bracket, xtol=1e-9)
        gamma = np.exp(self.log_gamma(coefficients, T, x, clamp=clamp))
        y = x * gamma[0] * self.problem.thermo.Psat(self.problem.components[0], T) / pressure
        mu = self.chemical_potentials(coefficients, T, x, clamp=clamp)
        test = np.linspace(1e-5, 1 - 1e-5, 1001)
        gap = self.gibbs(coefficients, T, test, clamp=clamp) - (
            test * mu[0] + (1 - test) * mu[1]
        )
        return float(T), float(y), float(min(gap) >= -1e-6)


def build_residual(model, request):
    rows = [
        row for row in request["observations"]
        if not row.get("validation_only")
        and (row.get("pin") or row.get("weight", 1) * request["weights"][row["kind"]] > 0)
    ]
    if not rows:
        raise ValueError("The Redlich–Kister probe needs a training observation.")
    if any(row.get("pin") for row in rows):
        raise ValueError("The Redlich–Kister research probe does not enforce hard pins.")
    supported = {"LLE", "VLE", "VLLE", "GAMMA_INF", "UCST", "LCST"}
    if any(row["kind"] not in supported for row in rows):
        raise ValueError("The Redlich–Kister probe supports LLE, VLE, VLLE, GAMMA_INF, UCST and LCST training data.")
    if request.get("vapor", "IDEAL") != "IDEAL" and any(row["kind"] in {"VLE", "VLLE"} for row in rows):
        raise ValueError("The Redlich–Kister probe requires IDEAL vapor for vapor-equilibrium data.")
    scales = request["scales"]
    one_sided = [row for row in rows if row["kind"] == "LLE" and ("x1_alpha" in row) != ("x1_beta" in row)]
    latent = {row["id"]: (model.nc * model.nt + index,) for index, row in enumerate(one_sided)}
    vlle_latent = [
        row for row in rows
        if row["kind"] == "VLLE" and "x1_alpha" not in row and "x1_beta" not in row
    ]
    first_vlle = model.nc * model.nt + len(one_sided)
    for index, row in enumerate(vlle_latent):
        latent[row["id"]] = (first_vlle + 2 * index, first_vlle + 2 * index + 1)

    def endpoints(values, row):
        if "x1_alpha" in row and "x1_beta" in row:
            return row["x1_alpha"], row["x1_beta"]
        gap = expit(values[latent[row["id"]][0]])
        if "x1_alpha" in row:
            return row["x1_alpha"], row["x1_alpha"] + (1 - row["x1_alpha"]) * gap
        return row["x1_beta"] * (1 - gap), row["x1_beta"]

    def residual(values):
        result = []
        for row in rows:
            weight = np.sqrt(row.get("weight", 1) * request["weights"][row["kind"]])
            T = row["T_K"]
            if row["kind"] == "LLE":
                low, high = endpoints(values, row)
                mu_low = model.chemical_potentials(values, T, low)
                mu_high = model.chemical_potentials(values, T, high)
                chemical = (mu_low - mu_high) / scales["log_fugacity"]
                if ("x1_alpha" in row) != ("x1_beta" in row):
                    chemical = chemical / (high - low)
                grid = np.linspace(1e-5, 1 - 1e-5, 43)
                mu = 0.5 * (mu_low + mu_high)
                gap = model.gibbs(values, T, grid) - (grid * mu[0] + (1 - grid) * mu[1])
                stability = np.minimum(gap, 0) / scales["log_fugacity"] / np.sqrt(len(grid))
                result.extend(weight * np.r_[chemical, stability])
            elif row["kind"] == "VLE":
                gamma = model.log_gamma(values, T, row["x1"])
                liquid = np.array([row["x1"], 1 - row["x1"]])
                vapor = np.array([row["y1"], 1 - row["y1"]])
                psat = np.array([
                    model.problem.thermo.Psat(component, T)
                    for component in model.problem.components
                ])
                raw = np.log(liquid) + gamma + np.log(psat) - np.log(vapor) - np.log(row["P_bar"])
                result.extend(weight * raw / scales["log_fugacity"])
            elif row["kind"] == "VLLE":
                if "x1_alpha" in row:
                    low, high = row["x1_alpha"], row["x1_beta"]
                else:
                    low = expit(values[latent[row["id"]][0]])
                    high = low + (1 - low) * expit(values[latent[row["id"]][1]])
                vapor = np.array([row["y1"], 1 - row["y1"]])
                psat = np.array([
                    model.problem.thermo.Psat(component, T)
                    for component in model.problem.components
                ])
                for liquid_x in (low, high):
                    liquid = np.array([liquid_x, 1 - liquid_x])
                    raw = (
                        np.log(liquid) + model.log_gamma(values, T, liquid_x)
                        + np.log(psat) - np.log(vapor) - np.log(row["P_bar"])
                    )
                    result.extend(weight * raw / scales["log_fugacity"])
                mu = 0.5 * (
                    model.chemical_potentials(values, T, low)
                    + model.chemical_potentials(values, T, high)
                )
                grid = np.linspace(1e-5, 1 - 1e-5, 43)
                gap = model.gibbs(values, T, grid) - (
                    grid * mu[0] + (1 - grid) * mu[1]
                )
                result.extend(
                    weight * np.minimum(gap, 0)
                    / scales["log_fugacity"] / np.sqrt(len(grid))
                )
            elif row["kind"] == "GAMMA_INF":
                if "gamma1_inf" in row:
                    prediction = model.log_gamma(values, T, 0.0)[0]
                    result.append(weight * (prediction - np.log(row["gamma1_inf"])) / scales["log_gamma"])
                if "gamma2_inf" in row:
                    prediction = model.log_gamma(values, T, 1.0)[1]
                    result.append(weight * (prediction - np.log(row["gamma2_inf"])) / scales["log_gamma"])
            elif row["kind"] in ("UCST", "LCST"):
                curvature, third, _ = model.critical_derivatives(values, T, row["x1"])
                result.extend(weight * np.array([
                    curvature / scales["curvature"],
                    third / scales["third_derivative"],
                ]))
        return np.asarray(result)

    initial = np.zeros(model.nc * model.nt + len(one_sided) + 2 * len(vlle_latent))
    initial[0] = 2.5
    for row in one_sided:
        initial[latent[row["id"]][0]] = logit(0.95)
    for row in vlle_latent:
        first, gap = latent[row["id"]]
        initial[first], initial[gap] = logit(0.02), logit(0.4)
    local_count = len(initial) - model.nc * model.nt
    lower = np.r_[np.full(model.nc * model.nt, -30.0), np.full(local_count, -14.0)]
    upper = np.r_[np.full(model.nc * model.nt, 30.0), np.full(local_count, 14.0)]
    return residual, initial, lower, upper


def fit_case(snapshot, composition_degree, temperature_degree, max_nfev):
    request = deepcopy(snapshot["request"])
    request["components"] = ["1-butanol", "water"]
    request.pop("initial", None)
    request.pop("bounds", None)
    request.pop("rq", None)
    problem = prepare_fit({**request, "model": "NRTL", "form": "constant", "fit_alpha": False, "initial": {}})
    temperatures = [row["T_K"] for row in problem.request["observations"] if not row["validation_only"]]
    model = RedlichKister(problem, composition_degree, temperature_degree, temperatures)
    residual, initial, lower, upper = build_residual(model, request)
    best = None
    rng = np.random.default_rng(1729 + 10 * composition_degree + temperature_degree)
    for start in (initial, np.clip(initial + rng.normal(0, 0.2, len(initial)), lower, upper)):
        fitted = least_squares(
            residual, start, bounds=(lower, upper), max_nfev=max_nfev,
            diff_step=1e-4, ftol=1e-10, xtol=1e-10, gtol=1e-9,
        )
        if best is None or fitted.cost < best.cost:
            best = fitted
    values = best.x
    errors = {"LLE_alpha": [], "LLE_beta": [], "VLE_T": [], "VLE_y": [], "gamma1_percent": [], "gamma2_percent": []}
    failed = []
    for row in request["observations"]:
        try:
            if row["kind"] == "LLE":
                split = model.stable_binodal(values, row["T_K"])
                if split is None:
                    failed.append((row["id"], "LLE")); continue
                if "x1_alpha" in row: errors["LLE_alpha"].append(split[0] - row["x1_alpha"])
                if "x1_beta" in row: errors["LLE_beta"].append(split[1] - row["x1_beta"])
            elif row["kind"] == "VLE":
                T, y, stable = model.bubble_state(values, row["P_bar"], row["x1"], row["T_K"])
                errors["VLE_T"].append(T - row["T_K"]); errors["VLE_y"].append(y - row["y1"])
                if not stable: failed.append((row["id"], "VLE"))
            elif row["kind"] == "GAMMA_INF":
                if "gamma1_inf" in row:
                    prediction = np.exp(model.log_gamma(values, row["T_K"], 0.0)[0])
                    errors["gamma1_percent"].append(100 * (prediction / row["gamma1_inf"] - 1))
                if "gamma2_inf" in row:
                    prediction = np.exp(model.log_gamma(values, row["T_K"], 1.0)[1])
                    errors["gamma2_percent"].append(100 * (prediction / row["gamma2_inf"] - 1))
        except Exception:
            failed.append((row["id"], row["kind"]))

    def rmse(items):
        return float(np.sqrt(np.mean(np.asarray(items) ** 2))) if items else None

    azeotropes = {}
    for clamp in (False, True):
        samples = []
        for x in np.linspace(1e-4, 1 - 1e-4, 101):
            try:
                T, y, stable = model.bubble_state(values, 3.0, float(x), 415, clamp=clamp)
                samples.append((float(x), T, y, bool(stable)))
            except Exception:
                samples.append((float(x), None, None, False))
        roots = []
        for left, right in zip(samples, samples[1:]):
            if not left[3] or not right[3] or (left[2] - left[0]) * (right[2] - right[0]) >= 0:
                continue
            try:
                root = brentq(
                    lambda x: model.bubble_state(values, 3.0, x, 415, clamp=clamp)[1] - x,
                    left[0], right[0], xtol=1e-9,
                )
                T, y, stable = model.bubble_state(values, 3.0, root, 415, clamp=clamp)
                delta = min(.005, root / 2, (1 - root) / 2)
                side = [model.bubble_state(values, 3.0, root + sign * delta, T, clamp=clamp)[0] for sign in (-1, 1)]
                kind = "maximum" if T > max(side) else "minimum" if T < min(side) else "stationary"
                roots.append({"x_butanol": root, "T_K": T, "type": kind, "stable": bool(stable)})
            except Exception:
                continue
        azeotropes["clamped" if clamp else "unrestricted"] = roots
    try:
        critical = least_squares(
            lambda state: model.critical_derivatives(values, state[0], state[1])[:2],
            [397.85, .108], bounds=([300, .001], [500, .999]), max_nfev=100,
        )
        critical_point = {"T_K":float(critical.x[0]),"x_butanol":float(critical.x[1]),"residual":critical.fun.tolist()}
    except Exception as error:
        critical_point = {"error":str(error)}
    return {
        "case":f"RK:c{composition_degree}:t{temperature_degree}",
        "composition_degree":composition_degree,"temperature_degree":temperature_degree,
        "objective":float(2 * best.cost),"success":bool(best.success),"nfev":best.nfev,
        "parameter_count":len(values),"errors":{key:rmse(value) for key,value in errors.items()},
        "failed_observations":failed,"critical_point":critical_point,
        "three_bar_azeotropes":azeotropes,"coefficients":values[:model.nc*model.nt].reshape(model.nc,model.nt).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--composition-degrees", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--temperature-degrees", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lle-weight", type=float, default=1.0)
    parser.add_argument("--collapse-vlle-plateau", action="store_true")
    parser.add_argument("--zero-weight-kinds", nargs="+", default=[])
    parser.add_argument("--zero-gamma1", action="store_true")
    parser.add_argument("--zero-gamma2", action="store_true")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    if args.collapse_vlle_plateau:
        plateau = collapse_vlle_plateau(snapshot["request"])
        print(json.dumps({"plateau_reconstruction": plateau}), flush=True)
    snapshot["request"]["weights"]["LLE"] = args.lle_weight
    for kind in args.zero_weight_kinds:
        snapshot["request"]["weights"][kind] = 0.0
    for row in snapshot["request"]["observations"]:
        if args.zero_gamma1 and "gamma1_inf" in row:
            row["weight"] = 0.0
        if args.zero_gamma2 and "gamma2_inf" in row:
            row["weight"] = 0.0
    cases = [(snapshot, c, t, args.max_nfev) for c in args.composition_degrees for t in args.temperature_degrees]
    results = []
    with ProcessPoolExecutor(max_workers=min(max(args.workers, 1), 5)) as pool:
        futures = [pool.submit(fit_case, *case) for case in cases]
        for future in as_completed(futures):
            result = future.result(); results.append(result); print(json.dumps(result), flush=True)
    print(json.dumps({"results":sorted(results,key=lambda item:item["case"])}),flush=True)


if __name__ == "__main__":
    main()
