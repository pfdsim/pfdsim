#!/usr/bin/env python3
"""Refit five binary UNIQUAC laws on PFDSim's ordinary structural basis.

All observations are binary measurements. Source vapor-pressure correlations
and vapor treatments are retained, so the regression changes the liquid
structural basis rather than hiding a change in vapor or pure-fluid models.
The output is the reviewed common-basis source collection consumed by the
interaction builder. This script never writes generated runtime parameters.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq, least_squares, minimize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemical_properties import ChemicalDatabase
from chemicals.volume import Yen_Woods_saturation
from interaction_parameters import uniquac_binary_interaction, uniquac_rq_for_component
from physical_constants import R_J_MOL_K
from thermodynamics_models.second_virial import create_second_virial_provider
from thermodynamics_models.factory import create_thermodynamics

REFERENCE = "data/reference/vapor-liquid-equilibria/"
THERMOML = REFERENCE + "2014-zaoui-ethanol-water-glycerol-thermoml.json"
SCRIPT = "scripts/activity_fitting/refit_uniquac_common_basis.py"

# Hartanto 2015 Table 2: x_ethanol, T/K. Vapor compositions in the
# accompanying figure are calculated, not independently measured.
HARTANTO = (
    (0.0000, 429.2),
    (0.0195, 419.0),
    (0.1499, 395.7),
    (0.2045, 391.1),
    (0.3128, 376.5),
    (0.3778, 374.3),
    (0.4647, 368.5),
    (0.5448, 365.0),
    (0.6179, 364.4),
    (0.6855, 359.8),
    (0.7662, 357.9),
    (0.8273, 354.4),
    (0.8649, 352.5),
    (0.9157, 351.7),
    (1.0000, 351.6),
)

# Arce 1995 Table 3, ethanol subsection: x1, y1, T/K, published gamma1,
# gamma2, phi1, phi2. Only x, y, T and P are regression observations;
# the derived gamma/phi columns are retained as source diagnostics.
ARCE = (
    (0.0000, 0.0000, 467.85, None, 1.0000, None, 0.9622),
    (0.0044, 0.1462, 461.91, 1.1316, 1.0000, 1.0044, 0.9605),
    (0.0088, 0.2612, 456.69, 1.1309, 1.0000, 1.0015, 0.9593),
    (0.0137, 0.3666, 452.19, 1.1301, 1.0000, 0.9989, 0.9586),
    (0.0210, 0.4566, 448.10, 1.1290, 1.0000, 0.9962, 0.9586),
    (0.0323, 0.5599, 441.79, 1.1272, 1.0001, 0.9933, 0.9586),
    (0.0434, 0.6557, 435.41, 1.1255, 1.0001, 0.9913, 0.9583),
    (0.0579, 0.7133, 430.80, 1.1231, 1.0002, 0.9898, 0.9587),
    (0.0757, 0.7840, 423.35, 1.1202, 1.0004, 0.9882, 0.9581),
    (0.1032, 0.8564, 413.28, 1.1156, 1.0008, 0.9862, 0.9564),
    (0.1494, 0.9112, 403.52, 1.1077, 1.0019, 0.9842, 0.9545),
    (0.2016, 0.9473, 393.89, 1.0985, 1.0037, 0.9821, 0.9514),
    (0.2747, 0.9688, 384.51, 1.0854, 1.0074, 0.9798, 0.9473),
    (0.3619, 0.9819, 376.14, 1.0698, 1.0143, 0.9775, 0.9427),
    (0.4546, 0.9880, 370.83, 1.0539, 1.0249, 0.9758, 0.9393),
    (0.5258, 0.9920, 367.16, 1.0424, 1.0358, 0.9745, 0.9366),
    (0.5697, 0.9937, 364.98, 1.0357, 1.0438, 0.9737, 0.9349),
    (0.6190, 0.9948, 362.88, 1.0288, 1.0542, 0.9729, 0.9332),
    (0.6961, 0.9959, 360.13, 1.0190, 1.0737, 0.9719, 0.9308),
    (0.7840, 0.9970, 357.31, 1.0100, 1.1012, 0.9707, 0.9282),
    (0.8597, 0.9981, 355.12, 1.0044, 1.1301, 0.9697, 0.9260),
    (0.9156, 0.9986, 353.75, 1.0016, 1.1550, 0.9691, 0.9247),
    (0.9492, 0.9989, 352.85, 1.0006, 1.1716, 0.9686, 0.9237),
    (0.9739, 0.9992, 352.20, 1.0002, 1.1846, 0.9683, 0.9230),
    (1.0000, 1.0000, 351.56, 1.0000, None, 0.9680, None),
)

# Mustain 2022 Tables 2 and 3: alcohol x1 and pressures/kPa at the
# three reported temperatures. Pure alcohol endpoints are not fitted.
MUSTAIN_BUTANOL = (
    (0.0984, 3.38, 4.25, 5.64),
    (0.2011, 4.93, 6.39, 8.26),
    (0.2988, 5.59, 7.19, 9.06),
    (0.4002, 6.13, 7.99, 9.86),
    (0.5000, 6.39, 8.26, 10.39),
    (0.6014, 6.79, 8.61, 10.85),
    (0.7000, 7.06, 9.06, 11.19),
    (0.7996, 7.22, 9.43, 11.67),
    (0.8948, 7.62, 9.70, 12.34),
    (1.0000, 8.05, 10.48, 13.54),
)
MUSTAIN_ISOBUTANOL = (
    (0.0990, 1.99, 2.71, 3.44),
    (0.2029, 2.79, 3.59, 4.87),
    (0.3020, 3.15, 4.12, 5.80),
    (0.3982, 3.33, 4.47, 6.02),
    (0.4976, 3.41, 4.55, 6.09),
    (0.6022, 3.54, 4.71, 6.21),
    (0.7004, 3.61, 4.84, 6.33),
    (0.7955, 3.65, 4.98, 6.45),
    (0.9005, 3.68, 5.05, 6.57),
    (1.0000, 3.83, 5.27, 7.09),
)


def antoine(A, B, C, *, base="log10", pressure_factor=100.0):
    """Coefficients use Celsius in the denominator; output is kPa."""
    return {"A": A, "B": B, "C": C, "base": base, "pressure_factor": pressure_factor}


def psat(correlation, temperature):
    value = correlation["A"] - correlation["B"] / (
        np.asarray(temperature) - 273.15 + correlation["C"]
    )
    return correlation["pressure_factor"] * np.exp(
        value * (math.log(10.0) if correlation["base"] == "log10" else 1.0)
    )


GLYCEROL_MUSTAIN = antoine(10.6190, 4487.040, 132.95, base="ln")


def _thermoml_rows():
    payload = json.loads((ROOT / THERMOML).read_text())
    names = {
        row["RegNum"]["nOrgNum"]: row["sCommonName"][0] for row in payload["Compound"]
    }
    datasets = [
        row
        for row in payload["PureOrMixtureData"]
        if {names[item["RegNum"]["nOrgNum"]] for item in row["Component"]}
        == {"ethanol", "glycerol"}
    ]
    if len(datasets) != 1:
        raise ValueError("Expected one ethanol/glycerol ThermoML dataset")
    dataset = datasets[0]
    temperature_id = next(
        row["nVarNumber"]
        for row in dataset["Variable"]
        if "eTemperature" in row["VariableID"]["VariableType"]
    )
    composition = next(
        row
        for row in dataset["Variable"]
        if "eComponentComposition" in row["VariableID"]["VariableType"]
    )
    if names[composition["VariableID"]["RegNum"]["nOrgNum"]] != "ethanol":
        raise ValueError("ThermoML composition is not ethanol mole fraction")
    return [
        {
            "x1": values[composition["nVarNumber"]],
            "T_K": values[temperature_id],
            "P_kPa": row["PropertyValue"][0]["nPropValue"],
        }
        for row in dataset["NumValues"]
        for values in (
            {item["nVarNumber"]: item["nVarValue"] for item in row["VariableValue"]},
        )
    ]


def cases():
    """Return observations and all fixed source-model conventions."""
    output = {
        "ethanol_hexanol": {
            "components": ["ethanol", "1-hexanol"],
            "cas": ["64-17-5", "111-27-3"],
            "source_doi": "10.13140/RG.2.1.4542.3766",
            "paper": REFERENCE + "2015-hartanto-ethanol-hexanol-vle.pdf",
            "table": "Table 2 (measurements), Tables 3-5 (property basis and fitted parameters)",
            "kind": "isobaric",
            "vapor": "IDEAL",
            "objective": "bubble-temperature error in K",
            "uncertainty_scales": {"x1": 0.001, "T_K": 0.1, "P_kPa": 0.2},
            "rows": [{"x1": x, "T_K": T, "P_kPa": 100.0} for x, T in HARTANTO],
            "psat": [
                antoine(5.33675, 1648.22, 230.918),
                antoine(4.18948, 1295.59, 152.510),
            ],
            "source_r": [2.5755, 5.2731],
            "source_q": [2.588, 4.748],
            "source_q_residual": [2.588, 4.748],
            "source_parameters": [0.0, -11.287, 0.0, -17.367],
            "extrapolation": "unrestricted",
        },
        "ethanol_octanol": {
            "components": ["ethanol", "1-octanol"],
            "cas": ["64-17-5", "111-87-5"],
            "source_doi": "10.1021/je00020a063",
            "paper": REFERENCE + "1995-arce-methanol-ethanol-octanol-vle.pdf",
            "table": "Table 3 ethanol subsection; Table 2 Antoine coefficients; Table 4 published fits",
            "kind": "isobaric_y",
            "vapor": "HOC",
            "objective": "bubble-temperature error in K plus vapor-y error in percentage points",
            "uncertainty_scales": {
                "x1": 0.002,
                "T_K": 0.02,
                "y1": 0.002,
                "P_kPa": 0.01,
            },
            "rows": [
                {
                    "x1": x,
                    "y1": y,
                    "T_K": T,
                    "P_kPa": 101.32,
                    "published_gamma": [g1, g2],
                    "published_phi": [p1, p2],
                }
                for x, y, T, g1, g2, p1, p2 in ARCE
            ],
            "psat": [
                antoine(7.16879, 1552.601, 222.419, pressure_factor=1.0),
                antoine(5.88511, 1264.322, 130.73, pressure_factor=1.0),
            ],
            "source_r": [2.5755, 6.6219],
            "source_q": [2.588, 5.828],
            "source_q_residual": [0.96, 2.71],
            "source_parameters": [0.0, 1005.00 / R_J_MOL_K, 0.0, -2668.64 / R_J_MOL_K],
            "extrapolation": "unrestricted",
            "additional_reference": REFERENCE
            + "1996-arce-water-ethanol-octanol-vle.pdf",
            "source_note": "The 1995 prose specifies ethanol q'=0.96 and octanol q'=2.71. The later 1996 binary GE correlation optimizes octanol q'=5.50. Printed derived gamma/phi columns are not fitted.",
        },
        "ethanol_glycerol": {
            "components": ["ethanol", "glycerol"],
            "cas": ["64-17-5", "56-81-5"],
            "source_doi": "10.1016/j.jct.2013.09.046",
            "paper": REFERENCE + "2016-zaoui-glycerol-binary-vle-modeling-thesis.pdf",
            "table": "Zaoui 2014 Tables 2 and 3, reprinted in thesis; ThermoML dataset 10",
            "kind": "isothermal",
            "vapor": "TSONOPOULOS",
            "objective": "relative pressure residuals",
            "rows": _thermoml_rows(),
            "psat": [
                antoine(8.11220, 1592.864, 226.184, pressure_factor=0.1333223684),
                antoine(8.623560, 2814.5, 201.467, pressure_factor=0.1333223684),
            ],
            "source_r": [2.5755, 4.7957],
            "source_q": [2.588, 4.908],
            "source_q_residual": [2.588, 4.908],
            "source_parameters": [1.54977589, -434.9661965, -1.91357655, 396.45785751],
            "extrapolation": "inverse_square_cubic",
            "source_note": "Previously staged parameters fit the paper's Barker/Redlich-Kister GE surface. New parameters fit the original measured pressures directly; derived vapor compositions are not observations.",
        },
    }
    for key, comp, cas, table, raw, temperatures, pure, published in (
        (
            "butanol_glycerol",
            "1-butanol",
            "71-36-3",
            2,
            MUSTAIN_BUTANOL,
            (333.15, 338.15, 343.15),
            antoine(4.64930, 1395.140, 182.739),
            [-54.3, 1339.4],
        ),
        (
            "isobutanol_glycerol",
            "isobutanol",
            "78-83-1",
            3,
            MUSTAIN_ISOBUTANOL,
            (313.15, 318.15, 323.15),
            antoine(4.34504, 1190.380, 166.670),
            [348.1, 969.4],
        ),
    ):
        output[key] = {
            "components": [comp, "glycerol"],
            "cas": [cas, "56-81-5"],
            "source_doi": "10.1021/acs.jced.1c00937",
            "paper": REFERENCE
            + "2022-mustain-butanol-isobutanol-glycerol-water-vle.pdf",
            "table": f"Table {table}; SI Tables S2-S3",
            "supporting_information": REFERENCE
            + "2022-mustain-butanol-isobutanol-glycerol-water-vle-supporting-information.pdf",
            "kind": "isothermal",
            "vapor": "IDEAL",
            "objective": "absolute pressure residuals in kPa (source objective)",
            "rows": [
                {"x1": row[0], "T_K": T, "P_kPa": pressure}
                for row in raw
                for T, pressure in zip(temperatures, row[1:])
            ],
            "psat": [pure, GLYCEROL_MUSTAIN],
            "source_r": [3.9243 if cas == "71-36-3" else 3.9235, 4.7957],
            "source_q": [3.668 if cas == "71-36-3" else 3.664, 4.908],
            "source_q_residual": [3.668 if cas == "71-36-3" else 3.664, 4.908],
            "source_parameters": [
                0.0,
                -published[0] / R_J_MOL_K,
                0.0,
                -published[1] / R_J_MOL_K,
            ],
            "extrapolation": "unrestricted",
        }
    return output


def ln_gamma(x1, r, q, q_residual, log_tau12, log_tau21):
    """Vectorized binary UNIQUAC, checked against the runtime scalar formula."""
    x = np.column_stack((np.asarray(x1), 1.0 - np.asarray(x1)))
    r, q, qr = np.asarray(r), np.asarray(q), np.asarray(q_residual)
    rx, qx = x @ r, x @ q
    ell = 5.0 * (r - q) - (r - 1.0)
    combinatorial = (
        np.log(r / rx[:, None])
        + 5.0 * q * np.log(q * rx[:, None] / (r * qx[:, None]))
        + ell
        - (r / rx[:, None]) * (x @ ell)[:, None]
    )
    theta = x * qr / (x @ qr)[:, None]
    t12, t21 = np.exp(log_tau12), np.exp(log_tau21)
    col1, col2 = theta[:, 0] + theta[:, 1] * t21, theta[:, 0] * t12 + theta[:, 1]
    residual1 = qr[0] * (
        1.0 - np.log(col1) - theta[:, 0] / col1 - theta[:, 1] * t12 / col2
    )
    residual2 = qr[1] * (
        1.0 - np.log(col2) - theta[:, 0] * t21 / col1 - theta[:, 1] / col2
    )
    return combinatorial + np.column_stack((residual1, residual2))


def physical_parameters(values, form, tref):
    """Use well-scaled optimizer coordinates, then emit a+b/T coefficients."""
    if form == "B_over_T":
        return np.array([0.0, values[0] * tref, 0.0, values[1] * tref])
    return np.array(
        [
            values[0] - values[1],
            values[1] * tref,
            values[2] - values[3],
            values[3] * tref,
        ]
    )


@dataclass
class PreparedCase:
    definition: dict
    r: np.ndarray
    q: np.ndarray
    props: dict
    provider: object

    def fixed_terms(self, temperatures, pressures):
        temperatures, pressures = np.asarray(temperatures), np.asarray(pressures)
        pure = np.column_stack([psat(c, temperatures) for c in self.definition["psat"]])
        matrices = np.zeros((len(temperatures), 2, 2))
        volumes = np.zeros((len(temperatures), 2))
        if self.provider is not None:
            matrices = np.asarray(
                [self.provider.second_virial_matrix(float(T)) for T in temperatures]
            )
            if self.definition["vapor"] == "HOC":
                for j, comp in enumerate(self.definition["components"]):
                    p = self.props[comp]
                    zc = (
                        p.Zc
                        if p.Zc is not None
                        else p.Pc * 1e5 * p.Vc * 1e-6 / (R_J_MOL_K * p.Tc)
                    )
                    volumes[:, j] = [
                        Yen_Woods_saturation(float(T), p.Tc, p.Vc * 1e-6, zc)
                        for T in temperatures
                    ]
        factor = pressures * 1000.0 / (R_J_MOL_K * temperatures)
        log_reference = np.log(pure)
        if self.provider is not None:
            log_reference += (
                np.diagonal(matrices, axis1=1, axis2=2)
                * pure
                * 1000.0
                / (R_J_MOL_K * temperatures[:, None])
            )
            log_reference += (
                volumes
                * (pressures[:, None] - pure)
                * 1000.0
                / (R_J_MOL_K * temperatures[:, None])
            )
        return matrices, factor, log_reference

    def evaluate(self, parameters, rows, *, r=None, q=None, qr=None, fixed=None):
        temperatures = np.asarray([row["T_K"] for row in rows])
        pressures = np.asarray([row["P_kPa"] for row in rows])
        x1 = np.asarray([row["x1"] for row in rows])
        r = self.r if r is None else r
        q = self.q if q is None else q
        qr = q if qr is None else qr
        gamma = ln_gamma(
            x1,
            r,
            q,
            qr,
            parameters[0] + parameters[1] / temperatures,
            parameters[2] + parameters[3] / temperatures,
        )
        matrix, factor, log_reference = (
            self.fixed_terms(temperatures, pressures) if fixed is None else fixed
        )
        targets = np.column_stack((x1, 1.0 - x1)) * np.exp(gamma + log_reference)
        vapor = targets / np.sum(targets, axis=1)[:, None]
        if self.provider is not None:
            for _ in range(80):
                partial = 2.0 * np.einsum("nij,nj->ni", matrix, vapor)
                bmix = np.einsum("ni,ni->n", vapor, partial) / 2.0
                corrected = targets * np.exp(
                    -factor[:, None] * (partial - bmix[:, None])
                )
                total = np.sum(corrected, axis=1)
                new_vapor = corrected / total[:, None]
                if np.max(np.abs(new_vapor - vapor)) < 1e-12:
                    return total, new_vapor[:, 0]
                vapor = new_vapor
            raise RuntimeError("Vapor-composition closure did not converge")
        return np.sum(targets, axis=1), vapor[:, 0]


def prepare(definition, db):
    structures = [uniquac_rq_for_component(c) for c in definition["components"]]
    if any(row is None for row in structures):
        raise ValueError("Ordinary structural parameters are missing")
    props = {}
    provider = None
    if definition["vapor"] != "IDEAL":
        props = {c: db.get(c, fetch_online=False) for c in definition["components"]}
        provider = create_second_virial_provider(
            definition["vapor"],
            definition["components"],
            props,
            allow_online=False,
            chemical_database=db,
        )
    return PreparedCase(
        definition,
        np.asarray([row["r"] for row in structures]),
        np.asarray([row["q"] for row in structures]),
        props,
        provider,
    )


def _metrics(errors):
    values = np.asarray(errors)
    return {
        "MAE": float(np.mean(np.abs(values))),
        "RMSE": float(np.sqrt(np.mean(values**2))),
        "MaxAE": float(np.max(np.abs(values))),
        "bias": float(np.mean(values)),
    }


def observations(definition):
    return [row for row in definition["rows"] if 0.0 < row["x1"] < 1.0]


def objective(case, rows, form, tref):
    temperatures = np.asarray([row["T_K"] for row in rows])
    pressures = np.asarray([row["P_kPa"] for row in rows])
    fixed = case.fixed_terms(temperatures, pressures)
    kind = case.definition["kind"]
    high_fixed = (
        case.fixed_terms(temperatures + 0.02, pressures)
        if kind.startswith("isobaric")
        else None
    )
    low_fixed = (
        case.fixed_terms(temperatures - 0.02, pressures)
        if kind.startswith("isobaric")
        else None
    )

    def residual(values):
        size = 2 if form == "B_over_T" else 4
        params = physical_parameters(values[:size], form, tref)
        if kind.startswith("isobaric"):
            scales = case.definition["uncertainty_scales"]
            adjusted = [
                row | {"x1": row["x1"] + scales["x1"] * offset}
                for row, offset in zip(rows, values[size:])
            ]
        else:
            adjusted = rows
        calculated, y = case.evaluate(params, adjusted, fixed=fixed)
        if kind.startswith("isobaric"):
            high = case.evaluate(
                params,
                [row | {"T_K": row["T_K"] + 0.02} for row in adjusted],
                fixed=high_fixed,
            )[0]
            low = case.evaluate(
                params,
                [row | {"T_K": row["T_K"] - 0.02} for row in adjusted],
                fixed=low_fixed,
            )[0]
            slope = (np.log(high) - np.log(low)) / 0.04
            if np.any(slope <= 1e-8):
                return np.full(len(rows) * (3 if kind == "isobaric_y" else 2), 1e6)
            errors = -np.log(calculated / pressures) / slope
            sigma_T = np.sqrt(
                scales["T_K"] ** 2 + (scales["P_kPa"] / (pressures * slope)) ** 2
            )
            columns = [errors / sigma_T, values[size:]]
            if kind == "isobaric_y":
                columns.append((y - [row["y1"] for row in rows]) / scales["y1"])
            return np.column_stack(columns).ravel()
        errors = calculated - pressures
        return (
            errors / pressures
            if case.definition["objective"] == "relative pressure residuals"
            else errors
        )

    return residual


def stability_grid(case, x_points, T_points):
    definition = case.definition
    temperatures = np.linspace(
        min(row["T_K"] for row in definition["rows"]),
        max(row["T_K"] for row in definition["rows"]),
        T_points,
    )
    x, T = np.meshgrid(np.linspace(1e-4, 1 - 1e-4, x_points), temperatures)
    return x.ravel(), T.ravel()


def stability_curvature(case, parameters, *, x_points=81, T_points=7, points=None):
    """Total g/RT curvature; a homogeneous binary needs nonnegative values.

    Complex-step differentiation avoids subtraction noise near the constraint.
    Pure-component poles are excluded; their ideal-mixing curvature diverges
    positively. The final audit uses a substantially denser grid.
    """
    x, T = stability_grid(case, x_points, T_points) if points is None else points
    gamma = ln_gamma(
        x + 1e-20j,
        case.r,
        case.q,
        case.q,
        parameters[0] + parameters[1] / T,
        parameters[2] + parameters[3] / T,
    )
    return 1 / x + 1 / (1 - x) + np.imag(gamma[:, 0] - gamma[:, 1]) / 1e-20


def minimum_stability(case, parameters, audit_points):
    """Refine sampled minima in continuous composition and temperature."""
    values = stability_curvature(case, parameters, points=audit_points)
    x, temperatures = audit_points
    low, high = float(np.min(temperatures)), float(np.max(temperatures))
    indices = [int(np.argmin(values))]
    for T in (low, (low + high) / 2, high):
        at_temperature = np.flatnonzero(np.isclose(temperatures, T))
        indices.append(int(at_temperature[np.argmin(values[at_temperature])]))
    best = (
        float(np.min(values)),
        (np.array([x[indices[0]]]), np.array([temperatures[indices[0]]])),
    )
    for index in dict.fromkeys(indices):

        def objective(coords):
            points = (np.array([coords[0]]), np.array([low + coords[1] * (high - low)]))
            return float(stability_curvature(case, parameters, points=points)[0])

        start = [x[index], (temperatures[index] - low) / (high - low)]
        result = minimize(
            objective, start, method="L-BFGS-B", bounds=((1e-5, 1 - 1e-5), (0.0, 1.0))
        )
        if float(result.fun) < best[0]:
            best = (
                float(result.fun),
                (np.array([result.x[0]]), np.array([low + result.x[1] * (high - low)])),
            )
    return best


def fit(case, rows, form, *, initial=None):
    tref = float(np.mean([row["T_K"] for row in observations(case.definition)]))
    source = case.definition["source_parameters"]
    guess = (
        np.array([source[1] / tref, source[3] / tref])
        if form == "B_over_T"
        else np.array(
            [
                source[0] + source[1] / tref,
                source[1] / tref,
                source[2] + source[3] / tref,
                source[3] / tref,
            ]
        )
    )
    starts = [guess, np.zeros(len(guess))] if initial is None else [np.asarray(initial)]
    if initial is None:
        starts.append(
            np.array([-0.5, 0.5])
            if form == "B_over_T"
            else np.array([-0.5, 0.0, 0.5, 0.0])
        )
    lower = (
        np.array([-6.0, -6.0])
        if form == "B_over_T"
        else np.array([-6.0, -30.0, -6.0, -30.0])
    )
    size = len(lower)
    upper = -lower
    if case.definition["kind"].startswith("isobaric"):
        sigma = case.definition["uncertainty_scales"]["x1"]
        lower = np.r_[lower, [max(-10.0, (-row["x1"] + 1e-8) / sigma) for row in rows]]
        upper = np.r_[
            upper, [min(10.0, (1.0 - row["x1"] - 1e-8) / sigma) for row in rows]
        ]
        starts = [np.r_[start[:size], np.zeros(len(rows))] for start in starts]
    residual = objective(case, rows, form, tref)
    results = [
        least_squares(
            residual,
            np.clip(start, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            max_nfev=800,
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        for start in starts
    ]
    valid = [
        result
        for result in results
        if result.success and np.all(np.isfinite(result.fun))
    ]
    if not valid:
        raise RuntimeError(f"No converged {form} regression")
    best = min(valid, key=lambda row: float(row.fun @ row.fun))
    unconstrained_cost = float(best.fun @ best.fun)
    audit_points = stability_grid(case, 501, 31)
    unconstrained_curvature = minimum_stability(
        case, physical_parameters(best.x[:size], form, tref), audit_points
    )[0]
    constrained = False
    if unconstrained_curvature < 1e-4:

        def objective_value(values):
            errors = residual(values)
            return float(errors @ errors) / max(1.0, unconstrained_cost)

        constraint_points = stability_grid(case, 81, 7)
        start = best.x
        for refinement in range(6):

            def constraint(values):
                return (
                    stability_curvature(
                        case,
                        physical_parameters(values[:size], form, tref),
                        points=constraint_points,
                    )
                    - 1e-4
                )

            starts = (start, np.zeros(len(lower)))
            solutions = [
                minimize(
                    objective_value,
                    candidate,
                    method="SLSQP",
                    bounds=list(zip(lower, upper)),
                    constraints={"type": "ineq", "fun": constraint},
                    options={"maxiter": 800, "ftol": 1e-9},
                )
                for candidate in starts
            ]
            feasible = [
                candidate
                for candidate in solutions
                if np.isfinite(candidate.fun)
                and np.min(constraint(candidate.x)) >= -1e-6
            ]
            if not feasible:
                raise RuntimeError(
                    f"No stable {form} regression: {[s.message for s in solutions]}"
                )
            solution = min(feasible, key=lambda row: row.fun)
            parameters = physical_parameters(solution.x[:size], form, tref)
            values = stability_curvature(case, parameters, points=audit_points)
            minimum, critical_point = minimum_stability(case, parameters, audit_points)
            if minimum >= 5e-5:
                break
            # Add the worst composition at every audit temperature. This
            # refines narrow minima without thousands of dense constraints.
            indices = np.argmin(values.reshape(31, 501), axis=1) + np.arange(31) * 501
            constraint_points = tuple(
                np.r_[old, audit[indices], critical]
                for old, audit, critical in zip(
                    constraint_points, audit_points, critical_point
                )
            )
            start = solution.x
        else:
            raise RuntimeError(
                "Adaptive stability constraint did not pass the dense audit"
            )
        best.x, best.fun = solution.x, residual(solution.x)
        # Recompute the residual Jacobian at the constrained solution.
        columns = []
        for i in range(len(best.x)):
            shift = np.zeros(len(best.x))
            shift[i] = 1e-5 * max(1.0, abs(best.x[i]))
            columns.append(
                (residual(best.x + shift) - residual(best.x - shift)) / (2 * shift[i])
            )
        best.jac = np.column_stack(columns)
        constrained = True
    jacobian = best.jac[:, :size]
    if len(best.x) > size:
        nuisance = best.jac[:, size:]
        jacobian -= nuisance @ np.linalg.lstsq(nuisance, jacobian, rcond=None)[0]
    singular = np.linalg.svd(jacobian, compute_uv=False)
    parameters = physical_parameters(best.x[:size], form, tref)
    minimum = minimum_stability(case, parameters, audit_points)[0]
    if minimum < -1e-5:
        raise RuntimeError(
            f"Dense-grid stability audit failed: minimum curvature={minimum}"
        )
    return {
        "form": form,
        "optimizer_values": best.x[:size].tolist(),
        "parameters": parameters.tolist(),
        "reference_temperature_K": tref,
        "objective_sum_squares": float(best.fun @ best.fun),
        "unconstrained_nfev": int(best.nfev),
        "unconstrained_optimizer_optimality": float(best.optimality),
        "constrained_optimizer_iterations": int(solution.nit) if constrained else None,
        "constrained_optimizer_success": bool(solution.success)
        if constrained
        else None,
        "constrained_optimizer_message": str(solution.message) if constrained else None,
        "jacobian_singular_values": singular.tolist(),
        "jacobian_condition": float(singular[0] / singular[-1])
        if singular[-1]
        else None,
        "at_parameter_bound": bool(
            np.any(best.x[:size] <= lower[:size] + 1e-4)
            or np.any(best.x[:size] >= upper[:size] - 1e-4)
        ),
        "composition_adjustments_sigma": best.x[size:].tolist(),
        "stability_constraint_active": constrained,
        "unconstrained_objective_sum_squares": unconstrained_cost,
        "unconstrained_minimum_curvature": unconstrained_curvature,
        "minimum_total_G_over_RT_curvature": minimum,
    }


def score(case, rows, parameters, *, source_structure=False, structure=None):
    kwargs = (
        {
            "r": case.definition["source_r"],
            "q": case.definition["source_q"],
            "qr": case.definition["source_q_residual"],
        }
        if source_structure
        else {}
    )
    if structure is not None:
        kwargs = dict(zip(("r", "q", "qr"), structure))
    calculated, vapor = case.evaluate(parameters, rows, **kwargs)
    observed = np.asarray([row["P_kPa"] for row in rows])
    result = {
        "pressure_percent": _metrics(100.0 * (calculated / observed - 1.0)),
        "pressure_kPa": _metrics(calculated - observed),
    }
    if case.definition["kind"].startswith("isobaric"):
        errors, y_errors, root_residuals = [], [], []
        for row in rows:

            def closure(T):
                return math.log(
                    case.evaluate(parameters, [row | {"T_K": float(T)}], **kwargs)[0][0]
                    / row["P_kPa"]
                )

            low, high = row["T_K"] - 20.0, row["T_K"] + 20.0
            temperature = brentq(closure, low, high, xtol=1e-9)
            errors.append(temperature - row["T_K"])
            root_residuals.append(abs(closure(temperature)))
            if "y1" in row:
                predicted_y = case.evaluate(
                    parameters, [row | {"T_K": temperature}], **kwargs
                )[1][0]
                y_errors.append(predicted_y - row["y1"])
        result["temperature_K"] = _metrics(errors)
        result["maximum_bubble_log_pressure_residual"] = max(root_residuals)
        if y_errors:
            result["vapor_y1"] = _metrics(y_errors)
    return result


def validate(case, rows, fitted):
    temperatures = sorted({row["T_K"] for row in rows})
    if case.definition["kind"] == "isothermal":
        folds = [
            [i for i, row in enumerate(rows) if row["T_K"] == T] for T in temperatures
        ]
        mode = "leave one complete temperature out"
    else:
        folds = [list(group) for group in np.array_split(np.arange(len(rows)), 5)]
        mode = "five contiguous liquid-composition blocks"
    residuals = []
    fold_details = []
    for indices in folds:
        testing = [rows[i] for i in indices]
        training = [row for i, row in enumerate(rows) if i not in indices]
        local = fit(case, training, fitted["form"], initial=fitted["optimizer_values"])
        residual = objective(
            case, testing, fitted["form"], fitted["reference_temperature_K"]
        )
        values = np.asarray(local["optimizer_values"])
        if case.definition["kind"].startswith("isobaric"):
            sigma = case.definition["uncertainty_scales"]["x1"]
            lower = [max(-10.0, (-row["x1"] + 1e-8) / sigma) for row in testing]
            upper = [min(10.0, (1 - row["x1"] - 1e-8) / sigma) for row in testing]
            projected = least_squares(
                lambda offsets: residual(np.r_[values, offsets]),
                np.zeros(len(testing)),
                bounds=(lower, upper),
                max_nfev=200,
            )
            if not projected.success:
                raise RuntimeError("Held-out measurement projection did not converge")
            errors = projected.fun
        else:
            errors = residual(values)
        residuals.extend(errors.tolist())
        fold_details.append(
            {
                "count": len(testing),
                "Tmin_K": min(row["T_K"] for row in testing),
                "Tmax_K": max(row["T_K"] for row in testing),
                "parameters": local["parameters"],
            }
        )
    return {
        "method": mode,
        "objective_residual": _metrics(residuals),
        "folds": fold_details,
    }


def build_case(name, definition, db=None, prepared=None):
    case = prepared if prepared is not None else prepare(definition, db)
    rows = observations(definition)
    candidates = []
    rejected_candidates = {}
    for form in ("B_over_T", "A_plus_B_over_T"):
        try:
            fitted = fit(case, rows, form)
            fitted["validation"] = validate(case, rows, fitted)
        except RuntimeError as error:
            rejected_candidates[form] = str(error)
            continue
        fitted["fit"] = score(case, rows, fitted["parameters"])
        if definition["kind"].startswith("isobaric"):
            sigma = definition["uncertainty_scales"]["x1"]
            adjusted = [
                row | {"x1": row["x1"] + sigma * offset}
                for row, offset in zip(rows, fitted["composition_adjustments_sigma"])
            ]
            fitted["fit_at_adjusted_compositions"] = score(
                case, adjusted, fitted["parameters"]
            )
        candidates.append(fitted)
    if not candidates:
        raise RuntimeError(f"No valid candidate: {rejected_candidates}")
    selected = candidates[0]
    if len(candidates) == 2:
        simple, extended = candidates
        improves = (
            extended["validation"]["objective_residual"]["RMSE"]
            < 0.9 * simple["validation"]["objective_residual"]["RMSE"]
        )
        selected = (
            extended if improves and not extended["at_parameter_bound"] else simple
        )
    a12, b12, a21, b21 = selected["parameters"]
    result = {
        "definition": definition,
        "standard_structure": {
            comp: {"r": float(r), "q": float(q)}
            for comp, r, q in zip(definition["components"], case.r, case.q)
        },
        "interior_point_count": len(rows),
        "pure_endpoint_count": len(definition["rows"]) - len(rows),
        "published_fit_on_source_structure": score(
            case, rows, definition["source_parameters"], source_structure=True
        ),
        "unchanged_parameters_on_standard_structure": score(
            case, rows, definition["source_parameters"]
        ),
        "candidates": candidates,
        "rejected_candidates": rejected_candidates,
        "selected_form": selected["form"],
        "selection_rule": "A+B/T requires at least 10% lower held-out objective RMSE and no active coefficient bound; otherwise B/T",
        "provider_warnings": list(getattr(case.provider, "warnings", ())),
        "physical_property_inputs": {
            comp: {
                field: getattr(props, field, None)
                for field in ("CAS", "Tc", "Pc", "Vc", "Zc", "omega", "smiles")
            }
            for comp, props in case.props.items()
        },
        "virial_matrices_at_observation_temperatures_m3_per_mol": virial_snapshot(case),
        "HOC_parameters": [
            asdict(item) for item in getattr(case.provider, "parameters", ())
        ],
        "interaction": {
            "model": "UNIQUAC",
            "cas1": definition["cas"][0],
            "cas2": definition["cas"][1],
            "component1": definition["components"][0],
            "component2": definition["components"][1],
            "model_variant": "standard_uniquac",
            "use_q_prime": False,
            "tau12_a": a12,
            "tau12_b": b12,
            "tau21_a": a21,
            "tau21_b": b21,
            "Tmin_K": min(row["T_K"] for row in definition["rows"]),
            "Tmax_K": max(row["T_K"] for row in definition["rows"]),
            "extrapolation": definition["extrapolation"],
            "source_doi": definition["source_doi"],
            "source_file": "data/source/activity_fitting/uniquac_common_basis_refits.json",
            "fit_status": "recommended_common_basis_refit",
            "comment": f"{name}: ordinary UNIQUAC binary-data refit; vapor={definition['vapor']}",
        },
    }
    return result


def stable(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, (list, tuple)):
        return [stable(item) for item in value]
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items()}
    return value


def reference_hashes(definition):
    references = {definition["paper"]}
    for field in ("additional_reference", "supporting_information"):
        if field in definition:
            references.add(definition[field])
    if definition["source_doi"] == "10.1016/j.jct.2013.09.046":
        references.add(THERMOML)
    output = {}
    for filename in sorted(references):
        with (ROOT / filename).open("rb") as handle:
            output[filename] = hashlib.file_digest(handle, "sha256").hexdigest()
    return output


def virial_snapshot(case):
    return (
        {
            str(T): case.provider.second_virial_matrix(T)
            for T in sorted({row["T_K"] for row in observations(case.definition)})
        }
        if case.provider
        else {}
    )


def audit_result(case, analysis):
    """Attach endpoint, caloric and uncertainty diagnostics without refitting."""
    selected = next(
        item
        for item in analysis["candidates"]
        if item["form"] == analysis["selected_form"]
    )
    parameters = selected["parameters"]
    selected["fit_including_pure_endpoints"] = score(
        case, case.definition["rows"], parameters
    )
    endpoints = [row for row in case.definition["rows"] if row["x1"] in (0.0, 1.0)]
    analysis["pure_pressure_correlation_endpoint_percent"] = _metrics(
        [
            100.0
            * (
                float(
                    psat(
                        case.definition["psat"][0 if row["x1"] == 1.0 else 1],
                        row["T_K"],
                    )
                )
                / row["P_kPa"]
                - 1
            )
            for row in endpoints
        ]
    )
    x, T = stability_grid(case, 101, 11)

    def ge(temperatures):
        gamma = ln_gamma(
            x,
            case.r,
            case.q,
            case.q,
            parameters[0] + parameters[1] / temperatures,
            parameters[2] + parameters[3] / temperatures,
        )
        return x * gamma[:, 0] + (1 - x) * gamma[:, 1]

    step = 0.02
    low, mid, high = ge(T - step), ge(T), ge(T + step)
    derivative = (high - low) / (2 * step)
    he = -R_J_MOL_K * T * T * derivative
    cp = -R_J_MOL_K * (
        2 * T * derivative + T * T * (high - 2 * mid + low) / (step * step)
    )
    analysis["caloric_diagnostics_over_calibration_range"] = {
        "excess_enthalpy_J_mol_range": [float(np.min(he)), float(np.max(he))],
        "excess_Cp_J_mol_K_range": [float(np.min(cp)), float(np.max(cp))],
        "note": "Diagnostics only; no excess-enthalpy or heat-capacity measurements constrain these fits",
    }
    previous_comparison = analysis.get("current_runtime_fit_comparison")
    if previous_comparison and previous_comparison["parameters"] != parameters:
        analysis.setdefault("superseded_runtime_fit_comparison", previous_comparison)
    current = uniquac_binary_interaction(*case.definition["cas"])
    if current is not None:
        structure = [[], [], []]
        for comp in case.definition["components"]:
            values = uniquac_rq_for_component(comp)
            r, q = values["r"], values["q"]
            qr = (
                values.get(
                    "q_prime",
                    values.get("extended_uniquac", {}).get("q_prime", values["q"]),
                )
                if current.get("use_q_prime")
                else q
            )
            for vector, value in zip(structure, (r, q, qr)):
                vector.append(value)
        if "tau12_a" not in current or any(
            current.get(field, 0.0)
            for field in (
                "tau12_c",
                "tau12_d",
                "tau12_e",
                "tau21_c",
                "tau21_d",
                "tau21_e",
            )
        ):
            raise ValueError("Current-fit comparison requires an A+B/T law")
        current_parameters = [
            current["tau12_a"],
            current["tau12_b"],
            current["tau21_a"],
            current["tau21_b"],
        ]
        analysis["current_runtime_fit_comparison"] = {
            "parameters": current_parameters,
            "structure": dict(zip(("r", "q", "q_residual"), structure)),
            "metrics_on_the_same_source_fluid_model": score(
                case,
                observations(case.definition),
                current_parameters,
                structure=structure,
            ),
            "note": "Existing binary parameters and active structural flags; evaluated using the same source Psat/vapor model as the candidates. This is not a validation of canonical runtime Psat.",
        }
    analysis["reference_sha256"] = reference_hashes(case.definition)
    flags = []
    offsets = selected["composition_adjustments_sigma"]
    if offsets and max(abs(value) for value in offsets) > 3:
        flags.append(
            "Some fitted liquid compositions differ by more than three reported uncertainty scales from the observations."
        )
    if offsets and max(abs(value) for value in offsets) > 9.99:
        flags.append(
            "The regression reaches its ten-uncertainty-scale composition-adjustment bound; the data are not reproduced at their quoted precision."
        )
    if selected["minimum_total_G_over_RT_curvature"] < 0.01:
        flags.append(
            "The stable fit lies close to a spinodal boundary; stability outside the calibration range is not established."
        )
    if selected.get("constrained_optimizer_success") is False:
        flags.append(
            "The constrained optimizer did not report success; the candidate is retained on numerical feasibility and residual quality, with continuous stability and exact bubble-root checks."
        )
    if analysis["pure_pressure_correlation_endpoint_percent"]["MaxAE"] > 1.0:
        flags.append(
            "The source pure-fluid correlation differs from at least one measured pure endpoint by more than 1%; excluded endpoints cannot be repaired by liquid interaction parameters."
        )
    analysis["quality_flags"] = flags
    analysis["interaction"]["fit_status"] = "recommended_common_basis_refit"
    analysis["interaction"]["source"] = (
        "PFDSim ordinary-basis refit of binary measurements; DOI "
        + case.definition["source_doi"]
    )
    analysis["interaction"]["fit_quality_flags"] = flags
    analysis["interaction"]["fit_evidence"] = {
        "script": SCRIPT,
        "analysis_key": next(
            name
            for name, definition in cases().items()
            if definition["cas"] == case.definition["cas"]
        ),
        "selected_form": selected["form"],
        "mixture_point_count": analysis["interior_point_count"],
        "fit": selected["fit"],
        "held_out_validation": selected["validation"],
        "minimum_total_G_over_RT_curvature": selected[
            "minimum_total_G_over_RT_curvature"
        ],
        "quality_flags": flags,
        "source_vapor_model": case.definition["vapor"],
    }


def check_active_runtime(definition, interaction, db):
    """Check the selected coefficients against canonical runtime properties.

    Pressure closure is evaluated at measured T, P and x; for gamma-phi
    variants this is a closure residual, not a separately solved bubble P.
    """
    model = (
        "UNIQUAC"
        + {"IDEAL": "", "HOC": "-HOC", "TSONOPOULOS": "-BV"}[definition["vapor"]]
    )
    thermo = create_thermodynamics(definition["components"], model, db=db)
    current = uniquac_binary_interaction(*definition["cas"])
    for field in (
        "tau12_a",
        "tau12_b",
        "tau21_a",
        "tau21_b",
        "use_q_prime",
    ):
        if current[field] != interaction[field]:
            raise ValueError(
                f"Runtime has not activated the reviewed {field} for {definition['components']}"
            )
    errors = []
    first, second = definition["components"]
    for row in observations(definition):
        x = {first: row["x1"], second: 1 - row["x1"]}
        K = thermo.K_values(row["T_K"], row["P_kPa"] / 100.0, x)
        errors.append(100.0 * (sum(x[comp] * K[comp] for comp in x) - 1.0))
    return {
        "method": model,
        "point_count": len(errors),
        "pressure_closure_percent": _metrics(errors),
        "warnings": thermo.warnings,
        "property_basis": "Canonical runtime Psat and the selected runtime vapor backend",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path)
    parser.add_argument("--case", action="append", choices=list(cases()))
    parser.add_argument("--workers", type=int, default=5, choices=range(1, 6))
    parser.add_argument(
        "--runtime-check",
        action="store_true",
        help="Verify selected entries are active and evaluate canonical-property pressure closure",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Reuse completed compatible pairs from a previous report",
    )
    args = parser.parse_args()
    definitions = cases()
    db = ChemicalDatabase(enable_online=False)
    results = {}
    failures = {}
    names = list(args.case or definitions)
    prepared = {name: prepare(definitions[name], db) for name in names}
    if args.resume:
        previous = json.loads(args.resume.read_text())
        if "physical_constraint" not in previous.get("metadata", {}):
            raise ValueError("Cannot resume an unconstrained report")
        for name, analysis in previous.get("analyses", {}).items():
            if name not in names:
                continue
            structural = {
                comp: {"r": float(r), "q": float(q)}
                for comp, r, q in zip(
                    definitions[name]["components"], prepared[name].r, prepared[name].q
                )
            }
            if analysis["definition"] != stable(definitions[name]) or analysis[
                "standard_structure"
            ] != stable(structural):
                raise ValueError(f"Resume inputs changed for {name}")
            if analysis[
                "virial_matrices_at_observation_temperatures_m3_per_mol"
            ] != stable(virial_snapshot(prepared[name])):
                raise ValueError(f"Resume vapor-model inputs changed for {name}")
            if analysis.get("reference_sha256") and analysis[
                "reference_sha256"
            ] != reference_hashes(definitions[name]):
                raise ValueError(f"Resume reference files changed for {name}")
            results[name] = analysis
            audit_result(prepared[name], results[name])
            print(f"Reused completed {name}", flush=True)
    metadata = {
        "script": SCRIPT,
        "description": "Five reviewed ordinary-basis UNIQUAC refits from original binary measurements",
        "activation": "The interaction builder replaces the five corresponding curated UNIQUAC entries with this collection; NRTL and all other UNIQUAC entries are preserved",
        "randomness": "none; fixed deterministic initial guesses",
        "structural_basis": "data/uniquac_rq_cas.json ordinary r/q; residual q equals q",
        "pure_endpoints": "excluded from regression and retained in source definitions",
        "physical_constraint": "Nonnegative total g/RT curvature over the complete calibration range; dense final audit",
        "isobaric_objective": "Profile least squares with latent liquid compositions penalized using published measurement tolerances; local temperature closure is checked by exact bubble roots",
        "held_out_isobaric_metric": "Distance to the fixed prediction curve allowing measurement uncertainty; no interaction coefficients are fitted to the held-out observations",
    }

    def report():
        return stable(
            {
                "metadata": metadata | {"failed_cases": failures},
                "analyses": results,
                "interactions": [
                    results[name]["interaction"] for name in names if name in results
                ],
            }
        )

    with ProcessPoolExecutor(max_workers=min(args.workers, len(names))) as executor:
        futures = {
            executor.submit(
                build_case, name, definitions[name], prepared=prepared[name]
            ): name
            for name in names
            if name not in results
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                audit_result(prepared[name], results[name])
                selected = next(
                    item
                    for item in results[name]["candidates"]
                    if item["form"] == results[name]["selected_form"]
                )
                print(
                    f"{name}: {selected['form']} fit={selected['fit']} held-out={selected['validation']['objective_residual']} minimum curvature={selected['minimum_total_G_over_RT_curvature']}",
                    flush=True,
                )
            except Exception as error:
                failures[name] = str(error)
                print(f"FAILED {name}: {error}", flush=True)
            if args.write:
                # Persist each finished pair, even if another worker later fails.
                args.write.write_text(
                    json.dumps(report(), indent=2, sort_keys=True, allow_nan=False)
                    + "\n"
                )
    if args.runtime_check:
        for name, analysis in results.items():
            analysis["active_runtime_validation"] = check_active_runtime(
                definitions[name], analysis["interaction"], db
            )
    if args.write:
        args.write.write_text(
            json.dumps(report(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(f"Wrote {args.write}", flush=True)
    else:
        print(json.dumps(report(), indent=2, sort_keys=True, allow_nan=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
