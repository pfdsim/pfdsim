#!/usr/bin/env python3
"""Reproduce or rediscover competing locally stable ternary NRTL splits.

The default verification uses pfdsim's real NRTL implementation and proves
that component ordering can select a higher-Gibbs metastable two-liquid split.
Pass ``--search`` to rerun the deterministic parameter search that originally
found the case.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermodynamics import create_thermodynamics  # noqa: E402


RNG = np.random.default_rng(20260906)
FLOOR = 1e-10
COMPONENTS = ("water", "ethanol", "acetone")
FOUND_TAU = np.asarray([
    [0.0, 6.004253484024018, 3.695557712584656],
    [4.786898609052036, 0.0, 6.702992852097343],
    [6.395561123016729, 5.164717994442046, 0.0],
])
TEMPERATURE = 298.15
FEED = {component: 1.0 / 3.0 for component in COMPONENTS}


def softmax2(values):
    shifted = np.array([values[0], values[1], 0.0], dtype=float)
    shifted -= np.max(shifted)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def logit_phase(x):
    return np.array([math.log(x[0] / x[2]), math.log(x[1] / x[2])])


def sigmoid(value):
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def nrtl_gamma(x, tau, alpha):
    interaction = np.exp(-alpha * tau)
    denominator = x @ interaction
    weighted_tau = (x[:, None] * tau * interaction).sum(axis=0) / denominator
    ln_gamma = np.empty(3)
    for i in range(3):
        first = np.sum(x * tau[:, i] * interaction[:, i]) / denominator[i]
        second = np.sum(
            x * interaction[i, :] / denominator * (tau[i, :] - weighted_tau)
        )
        ln_gamma[i] = first + second
    return np.exp(np.clip(ln_gamma, -50.0, 50.0))


def gibbs(x, tau, alpha):
    gamma = nrtl_gamma(x, tau, alpha)
    return float(np.sum(x * np.log(np.maximum(x * gamma, 1e-300))))


def numerical_hessian(function, point, step=2e-5):
    point = np.asarray(point, dtype=float)
    dimension = len(point)
    result = np.empty((dimension, dimension))
    center = function(point)
    for i in range(dimension):
        ei = np.zeros(dimension)
        ei[i] = step
        result[i, i] = (function(point + ei) - 2.0 * center + function(point - ei)) / step**2
        for j in range(i):
            ej = np.zeros(dimension)
            ej[j] = step
            value = (
                function(point + ei + ej)
                - function(point + ei - ej)
                - function(point - ei + ej)
                + function(point - ei - ej)
            ) / (4.0 * step**2)
            result[i, j] = value
            result[j, i] = value
    return 0.5 * (result + result.T)


def phase_hessian(x, tau, alpha):
    def reduced(values):
        phase = np.array([values[0], values[1], 1.0 - values[0] - values[1]])
        if np.min(phase) <= 0.0:
            return 1e6
        return gibbs(phase, tau, alpha)

    return numerical_hessian(reduced, x[:2])


def split_hessian(z, x1, x2, beta, tau, alpha):
    def constrained(values):
        trial_x1 = np.array([values[0], values[1], 1.0 - values[0] - values[1]])
        trial_beta = values[2]
        if trial_beta <= 0.0 or trial_beta >= 1.0 or np.min(trial_x1) <= 0.0:
            return 1e6
        trial_x2 = (z - (1.0 - trial_beta) * trial_x1) / trial_beta
        if np.min(trial_x2) <= 0.0:
            return 1e6
        return (1.0 - trial_beta) * gibbs(trial_x1, tau, alpha) + trial_beta * gibbs(
            trial_x2, tau, alpha
        )

    distances = np.concatenate((x1, x2, [beta, 1.0 - beta]))
    step = min(2e-5, max(2e-7, float(np.min(distances)) / 20.0))
    return numerical_hessian(constrained, [x1[0], x1[1], beta], step=step)


def solve_splits(z, tau, alpha, starts=70):
    def unpack(values):
        return softmax2(values[:2]), softmax2(values[2:4]), sigmoid(values[4])

    def residual(values):
        x1, x2, beta = unpack(values)
        mu1 = np.log(np.maximum(x1 * nrtl_gamma(x1, tau, alpha), 1e-300))
        mu2 = np.log(np.maximum(x2 * nrtl_gamma(x2, tau, alpha), 1e-300))
        balance = (1.0 - beta) * x1 + beta * x2 - z
        return np.concatenate((mu1 - mu2, 5.0 * balance[:2]))

    seeds = []
    for _ in range(starts):
        phase1 = RNG.dirichlet(np.full(3, 0.45))
        phase2 = RNG.dirichlet(np.full(3, 0.45))
        beta = RNG.uniform(0.08, 0.92)
        seeds.append(np.concatenate((logit_phase(phase1), logit_phase(phase2), [math.log(beta / (1.0 - beta))])))

    solutions = []
    for seed in seeds:
        solved = least_squares(
            residual,
            seed,
            max_nfev=700,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        if np.linalg.norm(residual(solved.x), ord=np.inf) > 2e-7:
            continue
        x1, x2, beta = unpack(solved.x)
        if np.linalg.norm(x1 - x2, ord=1) < 0.04 or not 0.03 < beta < 0.97:
            continue
        if tuple(x2) < tuple(x1):
            x1, x2, beta = x2, x1, 1.0 - beta
        signature = np.concatenate((x1, x2, [beta]))
        if any(np.linalg.norm(signature - old[0], ord=np.inf) < 2e-4 for old in solutions):
            continue
        phase_minimum = min(
            np.linalg.eigvalsh(phase_hessian(x1, tau, alpha))[0],
            np.linalg.eigvalsh(phase_hessian(x2, tau, alpha))[0],
        )
        split_minimum = np.linalg.eigvalsh(split_hessian(z, x1, x2, beta, tau, alpha))[0]
        total_gibbs = (1.0 - beta) * gibbs(x1, tau, alpha) + beta * gibbs(x2, tau, alpha)
        solutions.append((signature, phase_minimum, split_minimum, total_gibbs))
    return sorted(solutions, key=lambda item: item[3])


def parameter_sets():
    patterns = [
        (4.0, 3.5, 3.0),
        (5.0, 4.0, 3.0),
        (5.0, 3.5, 2.5),
        (6.0, 4.0, 2.5),
        (4.5, 4.0, 3.5),
    ]
    for pairs in patterns:
        tau = np.zeros((3, 3))
        for value, (i, j) in zip(pairs, ((0, 1), (0, 2), (1, 2))):
            tau[i, j] = value
            tau[j, i] = value
        yield tau
    for _ in range(180):
        tau = np.zeros((3, 3))
        for i, j in ((0, 1), (0, 2), (1, 2)):
            center = RNG.uniform(2.0, 6.5)
            asymmetry = RNG.uniform(-1.8, 1.8)
            tau[i, j] = center + asymmetry
            tau[j, i] = center - asymmetry
        yield tau


def nrtl_overrides(tau):
    records = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        records.append({
            "model": "NRTL",
            "component1": COMPONENTS[i],
            "component2": COMPONENTS[j],
            "tau12_c": float(tau[i, j]),
            "tau12_d": 0.0,
            "tau21_c": float(tau[j, i]),
            "tau21_d": 0.0,
            "alpha12": 0.30,
        })
    return records


def actual_gibbs(thermo, composition):
    gamma = thermo.activity_coefficients(TEMPERATURE, composition)
    return sum(
        composition[component]
        * math.log(max(composition[component] * gamma[component], 1e-300))
        for component in COMPONENTS
    )


def constrained_split_hessian(thermo, phase1, beta):
    z = np.asarray([FEED[component] for component in COMPONENTS])

    def objective(values):
        x1 = np.asarray((values[0], values[1], 1.0 - values[0] - values[1]))
        trial_beta = float(values[2])
        if trial_beta <= 0.0 or trial_beta >= 1.0 or np.min(x1) <= 0.0:
            return 1e6
        x2 = (z - (1.0 - trial_beta) * x1) / trial_beta
        if np.min(x2) <= 0.0:
            return 1e6
        first = dict(zip(COMPONENTS, x1))
        second = dict(zip(COMPONENTS, x2))
        return (
            (1.0 - trial_beta) * actual_gibbs(thermo, first)
            + trial_beta * actual_gibbs(thermo, second)
        )

    point = [phase1[COMPONENTS[0]], phase1[COMPONENTS[1]], beta]
    return numerical_hessian(objective, point, step=2e-5)


def split_metrics(thermo, result):
    has_lle, phase1, phase2, beta = result
    if not has_lle:
        raise AssertionError("Expected a nontrivial LLE split")
    gamma1 = thermo.activity_coefficients(TEMPERATURE, phase1)
    gamma2 = thermo.activity_coefficients(TEMPERATURE, phase2)
    equilibrium_residual = max(
        abs(
            math.log(max(phase1[component] * gamma1[component], 1e-300))
            - math.log(max(phase2[component] * gamma2[component], 1e-300))
        )
        for component in COMPONENTS
    )
    material_residual = max(
        abs(
            FEED[component]
            - (1.0 - beta) * phase1[component]
            - beta * phase2[component]
        )
        for component in COMPONENTS
    )
    stability1 = thermo.liquid_spinodal_stability(TEMPERATURE, phase1)
    stability2 = thermo.liquid_spinodal_stability(TEMPERATURE, phase2)
    split_minimum = float(np.linalg.eigvalsh(
        constrained_split_hessian(thermo, phase1, beta)
    )[0])
    total_gibbs = (
        (1.0 - beta) * actual_gibbs(thermo, phase1)
        + beta * actual_gibbs(thermo, phase2)
    )
    return {
        "phase1": phase1,
        "phase2": phase2,
        "beta": beta,
        "equilibrium_residual": equilibrium_residual,
        "material_residual": material_residual,
        "phase_eigenvalues": (
            stability1["minimum_eigenvalue"],
            stability2["minimum_eigenvalue"],
        ),
        "phases_locally_stable": (
            stability1["locally_stable"] and stability2["locally_stable"]
        ),
        "split_minimum_eigenvalue": split_minimum,
        "total_gibbs": total_gibbs,
    }


def verify_case():
    thermo = create_thermodynamics(
        list(COMPONENTS),
        "NRTL",
        interaction_overrides=nrtl_overrides(FOUND_TAU),
    )
    homogeneous_gibbs = actual_gibbs(thermo, FEED)
    results = {}
    for order in itertools.permutations(COMPONENTS):
        composition = {component: FEED[component] for component in order}
        split = thermo.liquid_liquid_equilibrium(
            composition,
            TEMPERATURE,
            max_iter=200,
            tol=1e-8,
        )
        results[order] = split_metrics(thermo, split)

    lower = results[("water", "ethanol", "acetone")]
    metastable = results[("acetone", "water", "ethanol")]
    for label, candidate in (("lower", lower), ("metastable", metastable)):
        assert candidate["equilibrium_residual"] < 1e-7, label
        assert candidate["material_residual"] < 1e-10, label
        assert candidate["phases_locally_stable"], label
        assert min(candidate["phase_eigenvalues"]) > 1e-3, label
        assert candidate["split_minimum_eigenvalue"] > 1e-2, label
        assert candidate["total_gibbs"] < homogeneous_gibbs - 0.1, label

    assert lower["total_gibbs"] < metastable["total_gibbs"] - 0.02
    phase_difference = max(
        abs(lower["phase1"][component] - metastable["phase1"][component])
        for component in COMPONENTS
    )
    assert phase_difference > 0.5

    alpha = np.full((3, 3), 0.30)
    np.fill_diagonal(alpha, 0.0)
    for candidate in (lower, metastable):
        for phase_name in ("phase1", "phase2"):
            phase = candidate[phase_name]
            expected = nrtl_gamma(
                np.asarray([phase[component] for component in COMPONENTS]),
                FOUND_TAU,
                alpha,
            )
            actual = thermo.activity_coefficients(TEMPERATURE, phase)
            assert max(
                abs(actual[component] - expected[index])
                for index, component in enumerate(COMPONENTS)
            ) < 1e-10

    print("Competing ternary NRTL LLE case verified")
    print(f"homogeneous Gibbs/RT: {homogeneous_gibbs:.12f}")
    for order, candidate in results.items():
        print(
            f"order={','.join(order):<22} "
            f"G/RT={candidate['total_gibbs']:.12f} "
            f"eq={candidate['equilibrium_residual']:.3e} "
            f"mass={candidate['material_residual']:.3e} "
            f"phase eig={candidate['phase_eigenvalues']} "
            f"split eig={candidate['split_minimum_eigenvalue']:.6g}"
        )
    print(
        "proved ordering defect: acetone-first converges to a locally stable "
        f"split higher by {(metastable['total_gibbs'] - lower['total_gibbs']):.12f} RT"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search",
        action="store_true",
        help="rerun the expensive deterministic parameter search",
    )
    return parser.parse_args()


OPTIONS = parse_args()
if not OPTIONS.search:
    verify_case()
    raise SystemExit(0)


feeds = (
    np.array([1 / 3, 1 / 3, 1 / 3]),
    np.array([0.45, 0.30, 0.25]),
    np.array([0.25, 0.45, 0.30]),
    np.array([0.30, 0.25, 0.45]),
)
alpha = np.full((3, 3), 0.30)
np.fill_diagonal(alpha, 0.0)

for parameter_index, tau in enumerate(parameter_sets()):
    for z in feeds:
        candidates = solve_splits(z, tau, alpha)
        homogeneous = gibbs(z, tau, alpha)
        favorable = [
            candidate
            for candidate in candidates
            if candidate[1] > 2e-3
            and candidate[2] > 2e-3
            and candidate[3] < homogeneous - 2e-5
        ]
        if len(favorable) < 2:
            continue
        if favorable[1][3] - favorable[0][3] < 2e-5:
            continue
        print("FOUND", parameter_index)
        print("z", z.tolist())
        print("tau", tau.tolist())
        print("homogeneous_g", homogeneous)
        for candidate in favorable:
            signature, phase_minimum, split_minimum, total_gibbs = candidate
            print(
                "candidate",
                "x1", signature[:3].tolist(),
                "x2", signature[3:6].tolist(),
                "beta", signature[6],
                "phase_min", phase_minimum,
                "split_min", split_minimum,
                "g", total_gibbs,
            )
        raise SystemExit(0)
    if parameter_index % 10 == 0:
        print("searched", parameter_index, flush=True)

print("NO CASE FOUND")
