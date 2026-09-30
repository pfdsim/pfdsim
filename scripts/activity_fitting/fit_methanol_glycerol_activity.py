#!/usr/bin/env python3
"""Audit joint methanol/glycerol NRTL and UNIQUAC correlations."""

import json
import math
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from thermodynamics_models.interaction_estimation import (
    _nrtl_ln_gamma,
    _uniquac_ln_gamma,
)


def psat_m(T):
    return 100 * 10 ** (5.20277 - 1580.080 / (T + 239.500 - 273.15))


def psat_g(T):
    return 100 * math.exp(10.6190 - 4487.040 / (T - 140.200))


def rec(source, T, x, P, f1=1.0, f2=1.0):
    return {"source": source, "T": T, "x": x, "P": P, "f1": f1, "f2": f2}


sources = {}
# Wiguno 2016 P-x
xs = [0.0979, 0.1954, 0.2989, 0.3987, 0.498, 0.5977, 0.6977, 0.7981, 0.8976]
p313 = [3.31, 6.76, 11.27, 14.19, 16.84, 22.28, 23.47, 27.59, 31.97]
p323 = [6.10, 11.00, 18.04, 23.74, 27.98, 35.68, 39.92, 45.37, 50.54]
wiguno_f = {313.15: 35.55 / psat_m(313.15), 323.15: 55.87 / psat_m(323.15)}
sources["Wiguno_2016"] = [
    rec("Wiguno_2016", T, x, P, wiguno_f[T])
    for x, a, b in zip(xs, p313, p323)
    for T, P in ((313.15, a), (323.15, b))
]
# Soujanya 2010 mole-fraction T-x
p = json.load(
    open(
        ROOT
        / "data/reference/vapor-liquid-equilibria/2010-soujanya-methanol-water-glycerol-thermoml.json"
    )
)
raw = []
for r in p["PureOrMixtureData"][8]["NumValues"]:
    raw.append(
        (
            r["VariableValue"][0]["nVarValue"],
            r["VariableValue"][1]["nVarValue"],
            r["PropertyValue"][0]["nPropValue"],
        )
    )
sf = {
    P: (
        P / psat_m(next(T for x, p, T in raw if p == P and x == 1)),
        P / psat_g(next(T for x, p, T in raw if p == P and x == 0)),
    )
    for P in sorted({p for x, p, T in raw})
}
rows = [rec("Soujanya_2010", T, x, P, *sf[P]) for x, P, T in raw if 0 < x < 1]
sources["Soujanya_2010"] = rows
# Veneral 2013 mass-fraction T-x
p = json.load(
    open(
        ROOT
        / "data/reference/vapor-liquid-equilibria/2013-veneral-methanol-ethanol-water-glycerol-thermoml.json"
    )
)
raw = []
for r in p["PureOrMixtureData"][4]["NumValues"]:
    w = r["VariableValue"][0]["nVarValue"]
    P = r["VariableValue"][1]["nVarValue"]
    T = r["PropertyValue"][0]["nPropValue"]
    x = (w / 32.042) / ((w / 32.042) + (1 - w) / 92.094)
    raw.append((x, P, T))
pure_m = {
    r["PropertyValue"][0]["nPropValue"]: r["VariableValue"][0]["nVarValue"]
    for r in p["PureOrMixtureData"][0]["NumValues"]
}
pure_g = {
    r["PropertyValue"][0]["nPropValue"]: r["VariableValue"][0]["nVarValue"]
    for r in p["PureOrMixtureData"][2]["NumValues"]
}
vf = {
    P: (P / psat_m(pure_m[P]), P / psat_g(pure_g[P]) if P in pure_g else 1.0)
    for P in sorted({p for x, p, T in raw})
}
rows = [rec("Veneral_2013", T, x, P, *vf[P]) for x, P, T in raw if 0 < x < 1]
sources["Veneral_2013"] = rows
# Oliveira 2009 atmospheric T-x
sources["Oliveira_2009"] = [
    rec("Oliveira_2009", T, x, 101.325)
    for x, T in [
        (0.0329, 451.78),
        (0.0487, 435.40),
        (0.0671, 423.42),
        (0.0911, 411.38),
        (0.1144, 399.25),
        (0.159, 390.55),
        (0.1947, 382.21),
        (0.2325, 377.27),
        (0.2637, 373.72),
        (0.2914, 370.57),
        (0.3147, 368.03),
        (0.3426, 365.28),
        (0.3633, 361.28),
        (0.3932, 359.38),
        (0.4218, 358.83),
        (0.4447, 357.24),
        (0.4651, 356.34),
        (0.4848, 354.49),
        (0.5055, 353.24),
        (0.5216, 352.14),
        (0.5434, 351.34),
        (0.5578, 350.54),
        (0.574, 349.84),
    ]
]


def uv(p, T):
    if len(p) == 4:
        return p[0] + p[1] / T, p[2] + p[3] / T
    h = (350 - T) / T + math.log(T / 350)
    return p[0] + p[1] / T + p[2] * h, p[3] + p[4] / T + p[5] * h


def pressure(r, model, alpha, p):
    u, v = uv(p, r["T"])
    x = r["x"]
    if model == "NRTL":
        a, b = _nrtl_ln_gamma(x, u, v, alpha)
    else:
        a, b = _uniquac_ln_gamma(
            x, [1.4311, 4.7957], [1.432, 4.908], math.exp(u), math.exp(v)
        )
    return (
        x * math.exp(a) * psat_m(r["T"]) * r["f1"]
        + (1 - x) * math.exp(b) * psat_g(r["T"]) * r["f2"]
    )


def metrics(rows, model, alpha, p):
    e = np.array([100 * (pressure(r, model, alpha, p) / r["P"] - 1) for r in rows])
    return {
        "ME_percent": float(e.mean()),
        "AARD_percent": float(abs(e).mean()),
        "RMSE_percent": float(np.sqrt(np.mean(e * e))),
        "MaxAE_percent": float(abs(e).max()),
    }


def fit(model, alpha, form, starts):
    def residual(p):
        out = []
        for rows in sources.values():
            out.extend(
                [
                    100
                    * (pressure(r, model, alpha, p) / r["P"] - 1)
                    / math.sqrt(len(rows))
                    for r in rows
                ]
            )
        return out

    n = len(starts[0])
    lo = (
        [-100, -30000, -500, -100, -30000, -500]
        if n == 6
        else [-100, -30000, -100, -30000]
    )
    hi = [-x for x in lo]
    results = [
        least_squares(
            residual,
            s,
            bounds=(lo, hi),
            max_nfev=1200,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        for s in starts
    ]
    best = min(results, key=lambda r: 2 * r.cost)
    result = {
        "model": model,
        "alpha": alpha,
        "form": form,
        "parameters": best.x.tolist(),
        "objective": float(2 * best.cost),
        "nfev": best.nfev,
        "sources": {k: metrics(v, model, alpha, best.x) for k, v in sources.items()},
    }
    print("CANDIDATE " + json.dumps(result), flush=True)
    return result


out = []
for alpha in (0.2, 0.3, 0.5):
    out.append(fit("NRTL", alpha, "AB", [[0, 586.953, 0, -383.897], [0, 0, 0, 0]]))
out.append(fit("UNIQUAC", None, "AB", [[0, 7.974, 0, -119.815], [0, 0, 0, 0]]))
for alpha in (0.2, 0.3, 0.5):
    out.append(
        fit("NRTL", alpha, "ABH", [[0, 586.953, 0, 0, -383.897, 0], [0, 0, 0, 0, 0, 0]])
    )
out.append(
    fit("UNIQUAC", None, "ABH", [[0, 7.974, 0, 0, -119.815, 0], [0, 0, 0, 0, 0, 0]])
)
print(
    "FINAL "
    + json.dumps(
        {"source_counts": {k: len(v) for k, v in sources.items()}, "candidates": out}
    ),
    flush=True,
)
