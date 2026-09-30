#!/usr/bin/env python3
"""Audit joint propanol/glycerol NRTL and UNIQUAC correlations."""

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

R = 8.31446261815324


def psat_alcohol(kind, T):
    c = {
        "1propanol": (4.99991, 1512.940, 205.807),
        "2propanol": (5.24268, 1580.920, 219.610),
    }[kind]
    return 100 * 10 ** (c[0] - c[1] / (T + c[2] - 273.15))


def psat_glycerol(T):
    return 100 * math.exp(10.6190 - 4487.040 / (T - 140.200))


def psat_soujanya(kind, T):
    c = {
        "2propanol": (18.6929, 3640.2, -53.54),
        "glycerol": (17.2392, 4487.04, -140.2),
    }[kind]
    return 0.133 * math.exp(c[0] - c[1] / (T + c[2]))


def record(kind, source, T, x, P):
    return {"kind": kind, "source": source, "T": T, "x": x, "P": P}


sources = {}
# Wibawa 2-propanol isothermal P-x
xs = [0.0986, 0.2015, 0.3017, 0.3999, 0.4998, 0.5998, 0.6995, 0.7993, 0.9024]
p333 = [9.33, 16.13, 19.73, 22.66, 26.66, 28.66, 30.66, 33.33, 34.93]
p343 = [10.67, 18.67, 25.33, 30.40, 35.66, 39.93, 46.66, 50.66, 56.00]
sources["2p_Wibawa"] = [
    record("2propanol", "2p_Wibawa", T, x, P)
    for x, p1, p2 in zip(xs, p333, p343)
    for T, P in ((333.15, p1), (343.15, p2))
]
# Soujanya 2-propanol isobaric T-x
souj = {
    53.33: [
        (0.0839, 341),
        (0.1685, 340.5),
        (0.1928, 340.35),
        (0.3028, 340.1),
        (0.3644, 339.95),
        (0.4015, 339.9),
        (0.4512, 339.8),
        (0.5011, 339.68),
        (0.5512, 339.6),
        (0.6522, 339.55),
        (0.7455, 339.45),
        (0.8441, 339.25),
        (0.8958, 339.35),
        (0.9218, 339.46),
        (0.9478, 339.56),
        (0.9638, 339.52),
        (0.9739, 339.89),
    ],
    66.66: [
        (0.0239, 346.65),
        (0.0719, 345.68),
        (0.0839, 345.36),
        (0.1685, 345.15),
        (0.2538, 345.05),
        (0.3644, 344.9),
        (0.5011, 344.7),
        (0.5512, 344.65),
        (0.6522, 344.6),
        (0.7455, 344.57),
        (0.8441, 344.5),
        (0.8958, 344.62),
        (0.9218, 344.65),
        (0.9478, 344.7),
        (0.9638, 344.75),
        (0.9739, 344.85),
    ],
    79.99: [
        (0.0839, 351.02),
        (0.0948, 349.75),
        (0.1685, 349.45),
        (0.2538, 349.25),
        (0.3644, 349.09),
        (0.4512, 348.97),
        (0.5011, 348.91),
        (0.5512, 348.87),
        (0.6522, 348.83),
        (0.7455, 348.8),
        (0.8441, 348.83),
        (0.8958, 348.93),
        (0.9218, 349.01),
        (0.9478, 349.06),
        (0.9739, 349.15),
    ],
    94.93: [
        (0.0839, 354.84),
        (0.1685, 354.09),
        (0.2538, 353.8),
        (0.3644, 353.65),
        (0.4512, 353.45),
        (0.5011, 353.3),
        (0.5512, 353.15),
        (0.589, 353.05),
        (0.6522, 353.19),
        (0.7031, 353.3),
        (0.7455, 353.35),
        (0.8441, 353.45),
        (0.8958, 353.52),
        (0.9218, 353.52),
        (0.9478, 353.61),
        (0.9739, 353.75),
    ],
}
sources["2p_Soujanya"] = [
    record("2propanol", "2p_Soujanya", T, x, P)
    for P, rows in souj.items()
    for x, T in rows
]
# Oliveira atmospheric curves
sources["2p_Oliveira"] = [
    record("2propanol", "2p_Oliveira", T, x, 101.325)
    for x, T in [
        (0.0064, 468.93),
        (0.0079, 459.08),
        (0.0222, 443.27),
        (0.0293, 425.27),
        (0.0405, 415.17),
        (0.0503, 404.17),
        (0.0777, 395.31),
        (0.098, 389.81),
        (0.1297, 385.01),
        (0.1478, 380.51),
        (0.1821, 377.96),
        (0.2106, 375.56),
        (0.2252, 373.96),
        (0.2492, 373.06),
        (0.2787, 371.56),
        (0.2915, 370.71),
        (0.3121, 370.01),
        (0.3189, 369.81),
        (0.3347, 369.11),
        (0.358, 368.41),
    ]
]
sources["1p_Oliveira"] = [
    record("1propanol", "1p_Oliveira", T, x, 101.325)
    for x, T in [
        (0.0418, 481.69),
        (0.0474, 467.41),
        (0.0559, 452.42),
        (0.0629, 435.93),
        (0.0698, 425.04),
        (0.092, 415.69),
        (0.1111, 407.05),
        (0.138, 402.35),
        (0.1646, 398.25),
        (0.1881, 395.05),
        (0.2088, 392.21),
        (0.2418, 390.31),
        (0.2606, 388.51),
        (0.273, 387.26),
        (0.3001, 386.16),
        (0.3207, 385.36),
        (0.3339, 384.71),
        (0.3518, 383.96),
        (0.3753, 383.41),
        (0.3928, 383.01),
        (0.4044, 382.56),
        (0.4273, 382.26),
        (0.441, 381.96),
    ]
]
# Wiguno 1-propanol isothermal P-x
xs = [0.1223, 0.217, 0.3139, 0.4051, 0.5024, 0.6009, 0.7013, 0.8005, 0.8995]
pp = [
    [3.97, 6.50, 8.35],
    [6.89, 10.61, 17.51],
    [9.41, 16.58, 23.61],
    [13.59, 21.22, 29.97],
    [16.57, 26.19, 39.53],
    [19.63, 31.17, 47.09],
    [22.55, 36.34, 52.93],
    [27.72, 41.92, 61.15],
    [28.91, 47.22, 68.19],
]
sources["1p_Wiguno"] = [
    record("1propanol", "1p_Wiguno", T, x, P)
    for x, vals in zip(xs, pp)
    for T, P in zip((343.15, 353.15, 363.15), vals)
]
# Batutah 1-propanol isobaric Txy (y retained only as provenance; pressure closure fitted)
sources["1p_Batutah"] = [
    record("1propanol", "1p_Batutah", T, x, P)
    for P, rows in {
        16: [
            (0.9066, 330.05),
            (0.7935, 332.05),
            (0.6867, 334.05),
            (0.6064, 335.85),
            (0.4869, 339.25),
            (0.3935, 342.75),
            (0.3133, 348.85),
            (0.2467, 353.35),
            (0.0867, 379.75),
        ],
        101.3: [
            (0.8877, 372.95),
            (0.7616, 374.35),
            (0.6447, 378.65),
            (0.5059, 383.85),
            (0.4076, 388.85),
            (0.2536, 397.25),
            (0.1755, 409.75),
            (0.0841, 424.85),
            (0.0598, 436.65),
        ],
    }.items()
    for x, T in rows
]
rq = {
    "1propanol": ([3.2499, 4.7957], [3.128, 4.908]),
    "2propanol": ([3.2491, 4.7957], [3.124, 4.908]),
}


def predicted(row, model, alpha, p):
    T, x = row["T"], row["x"]
    if len(p) == 4:
        u = p[0] + p[1] / T
        v = p[2] + p[3] / T
    else:
        h = (350.0 - T) / T + math.log(T / 350.0)
        u = p[0] + p[1] / T + p[2] * h
        v = p[3] + p[4] / T + p[5] * h
    if model == "NRTL":
        l1, l2 = _nrtl_ln_gamma(x, u, v, alpha)
    else:
        l1, l2 = _uniquac_ln_gamma(x, *rq[row["kind"]], math.exp(u), math.exp(v))
    if row["source"] == "2p_Soujanya":
        p1 = psat_soujanya("2propanol", T)
        p2 = psat_soujanya("glycerol", T)
    else:
        p1 = psat_alcohol(row["kind"], T)
        p2 = psat_glycerol(T)
    return x * math.exp(l1) * p1 + (1 - x) * math.exp(l2) * p2


def metrics(rows, model, alpha, p):
    e = np.array([100 * (predicted(r, model, alpha, p) / r["P"] - 1) for r in rows])
    return {
        "ME_percent": float(e.mean()),
        "AARD_percent": float(abs(e).mean()),
        "RMSE_percent": float(np.sqrt(np.mean(e * e))),
        "MaxAE_percent": float(abs(e).max()),
    }


def fit(kind, model, alpha, starts):
    selected = {k: v for k, v in sources.items() if v[0]["kind"] == kind}

    def residual(p):
        out = []
        for rows in selected.values():
            out.extend(
                [
                    100
                    * (predicted(r, model, alpha, p) / r["P"] - 1)
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
    fits = [
        least_squares(
            residual,
            s,
            bounds=(lo, hi),
            max_nfev=1000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        for s in starts
    ]
    best = min(fits, key=lambda x: 2 * x.cost)
    return {
        "parameters": best.x.tolist(),
        "nfev": best.nfev,
        "objective": float(2 * best.cost),
        "sources": {k: metrics(v, model, alpha, best.x) for k, v in selected.items()},
    }


out = {}
for kind in ("1propanol", "2propanol"):
    out[kind] = {}
    for alpha in (0.2, 0.3, 0.4, 0.5):
        out[kind][f"NRTL_{alpha}"] = fit(
            kind, "NRTL", alpha, [[0, 0, 0, 0], [-1, 500, 1, -300]]
        )
    out[kind]["UNIQUAC"] = fit(
        kind, "UNIQUAC", None, [[0, 0, 0, 0], [1, -300, -1, 300]]
    )
    for alpha in (0.2, 0.3, 0.4, 0.5):
        out[kind][f"NRTL_ABH_{alpha}"] = fit(
            kind, "NRTL", alpha, [[0, 0, 0, 0, 0, 0], [-1, 500, 0, 1, -300, 0]]
        )
    out[kind]["UNIQUAC_ABH"] = fit(
        kind, "UNIQUAC", None, [[0, 0, 0, 0, 0, 0], [1, -300, 0, -1, 300, 0]]
    )
print(json.dumps(out, indent=2))
