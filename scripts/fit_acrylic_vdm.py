"""Fit acrylic-acid VDM parameters to observed T/P/liquid-x bubble data.

The table's calculated vapor composition and relative volatility are retained
only for post-fit diagnostics; they are not included in the objective.
"""

import math
import warnings

import numpy as np
from scipy.optimize import least_squares

from thermodynamics import create_thermodynamics
from vapor_dimerization import VaporDimerizationModel


warnings.simplefilter("ignore", RuntimeWarning)

BLOCKS = [
    (10.00, [(63.26,.0869,.4909),(57.44,.1834,.6619),(54.08,.2845,.7425),(51.98,.3863,.7904),(50.35,.4896,.8252),(48.84,.5934,.8543),(47.95,.7089,.8845),(47.26,.8082,.9132),(46.88,.8990,.9476)]),
    (13.33, [(69.67,.0876,.4763),(63.61,.1841,.6504),(60.09,.2850,.7344),(57.85,.3866,.7847),(56.15,.4898,.8213),(54.58,.5935,.8516),(53.62,.7090,.8827),(52.90,.8082,.9120),(52.50,.8990,.9468)]),
    (26.66, [(86.67,.0894,.4427),(79.86,.1859,.6232),(75.93,.2863,.7151),(73.28,.3875,.7714),(71.33,.4904,.8123),(69.61,.5938,.8456),(68.51,.7092,.8791),(67.67,.8083,.9097),(67.16,.8990,.9451)]),
    (40.00, [(97.66,.0904,.4240),(90.39,.1868,.6075),(86.14,.2870,.7041),(83.20,.3879,.7640),(81.08,.4906,.8075),(79.24,.5940,.8426),(78.03,.7093,.8774),(77.10,.8084,.9086),(76.54,.8991,.9444)]),
    (53.33, [(105.97,.0910,.4109),(98.34,.1875,.5966),(93.87,.2875,.6964),(90.70,.3883,.7589),(88.44,.4908,.8042),(86.48,.5941,.8406),(85.15,.7094,.8764),(84.18,.8084,.9080),(83.59,.8991,.9439)]),
    (66.66, [(112.78,.0915,.4011),(104.84,.1880,.5882),(100.13,.2879,.6905),(96.78,.3885,.7550),(94.40,.4909,.8018),(92.34,.5942,.8392),(90.94,.7094,.8757),(89.92,.8084,.9077),(89.31,.8991,.9436)]),
    (79.99, [(118.57,.0919,.3933),(110.44,.1884,.5813),(105.47,.2882,.6857),(101.94,.3886,.7519),(99.46,.4910,.7999),(97.31,.5942,.8381),(95.86,.7094,.8752),(94.77,.8084,.9074),(94.14,.8991,.9434)]),
    (101.325, [(126.42,.0924,.3833),(117.84,.1889,.5726),(112.64,.2885,.6795),(108.89,.3888,.7479),(106.27,.4911,.7975),(104.00,.5942,.8368),(102.48,.7094,.8746),(101.31,.8084,.9071),(100.64,.8991,.9431)]),
]

DATA = [(t + 273.15, p / 100.0, x, y) for p, rows in BLOCKS for t, x, y in rows]
ACID = "C2H3COOH"


def model(parameters):
    return VaporDimerizationModel(
        ACID,
        f"({ACID})2",
        delta_H=parameters[0] * 1000.0,
        delta_S=parameters[1],
    )


def fit_method(method):
    thermo = create_thermodynamics([ACID, "H2O"], method)
    constants = []
    for temperature, pressure, x_water, y_water in DATA:
        liquid = {ACID: 1.0 - x_water, "H2O": x_water}
        constants.append((
            temperature,
            pressure,
            x_water,
            y_water,
            thermo.activity_coefficients(temperature, liquid),
            {component: thermo.Psat(component, temperature) for component in liquid},
        ))

    def bubble_residuals(parameters):
        association = model(parameters)
        residuals = []
        for temperature, pressure, x_water, y_water, gamma, psat in constants:
            liquid = {ACID: 1.0 - x_water, "H2O": x_water}
            phi_sat_acid = association.fugacity_coefficients(
                temperature, psat[ACID], {ACID: 1.0, "H2O": 0.0}
            )[ACID]
            base_acid = (
                gamma[ACID] * phi_sat_acid * psat[ACID] / pressure
            )
            base_water = gamma["H2O"] * psat["H2O"] / pressure
            acid_x_base = liquid[ACID] * base_acid
            water_x_base = liquid["H2O"] * base_water
            y_acid = acid_x_base / (acid_x_base + water_x_base)
            kappa = association.K_eq(temperature) * pressure
            for _ in range(15):
                alpha = association._alpha_from_equilibrium(kappa, y_acid)
                acid_numerator = acid_x_base / max(1.0 - alpha, 1e-30)
                y_new = acid_numerator / (acid_numerator + water_x_base)
                if abs(y_new - y_acid) < 1e-9:
                    y_acid = y_new
                    break
                y_acid = y_new
            alpha = association._alpha_from_equilibrium(kappa, y_acid)
            denominator = max(1.0 - y_acid * alpha / 2.0, 1e-30)
            k_acid = base_acid * denominator / max(1.0 - alpha, 1e-30)
            k_water = base_water * denominator
            bubble_sum = liquid[ACID] * k_acid + liquid["H2O"] * k_water
            residuals.append(math.log(bubble_sum))
        return np.asarray(residuals)

    best = None
    for start in (
        (-67.0, -161.4),
        (-77.5, -185.2),
        (-55.6, -127.2),
        (-12.3, 0.0),
        (20.0, 100.0),
    ):
        result = least_squares(
            bubble_residuals,
            start,
            bounds=([-300.0, -1000.0], [300.0, 1000.0]),
            x_scale="jac",
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=10000,
        )
        if best is None or result.cost < best.cost:
            best = result

    residuals = bubble_residuals(best.x)
    print(
        method,
        "fit_H_kJ_mol", best.x[0],
        "fit_S_J_mol_K", best.x[1],
        "fit_rms_ln_bubble", math.sqrt(np.mean(residuals**2)),
        "fit_max_abs_ln_bubble", np.max(np.abs(residuals)),
        "cost", best.cost,
    )
    for label, parameters in (
        ("stored", (-67.0, -161.4)),
        ("strong", (-77.5, -185.2)),
        ("Perry_PR", (-55.5569, -127.1714)),
    ):
        trial = bubble_residuals(parameters)
        print(method, label, "rms_ln_bubble", math.sqrt(np.mean(trial**2)))
    return best.x


fits = {}
for method in ("UNIQUAC", "UNIFNIST", "UNIFDMD", "UNIFM2", "UNIFAC2"):
    try:
        fits[method] = fit_method(method)
    except Exception as exc:
        print(method, "FAILED", type(exc).__name__, str(exc))


def score_txy(label, parameters):
    thermo = create_thermodynamics([ACID, "H2O"], "UNIFNIST-VDM")
    thermo._vdm_models[ACID] = model(parameters)
    temperature_errors = []
    vapor_errors = []
    for temperature, pressure, x_water, y_water in DATA:
        liquid = {ACID: 1.0 - x_water, "H2O": x_water}
        calculated_temperature = thermo.bubble_point_T(
            liquid, pressure, temperature
        )
        k_values = thermo.K_values(
            calculated_temperature, pressure, liquid
        )
        denominator = sum(
            liquid[component] * k_values[component]
            for component in liquid
        )
        calculated_y_water = (
            liquid["H2O"] * k_values["H2O"] / denominator
        )
        temperature_errors.append(calculated_temperature - temperature)
        vapor_errors.append(100.0 * (calculated_y_water - y_water))

    def metrics(errors):
        values = np.asarray(errors)
        return (
            float(np.mean(values)),
            float(np.mean(np.abs(values))),
            float(math.sqrt(np.mean(values**2))),
            float(np.max(np.abs(values))),
        )

    print(label, "T_ME_MAE_RMSE_MAX", metrics(temperature_errors))
    print(
        label,
        "calculated_ypp_diagnostic_not_fit_ME_MAE_RMSE_MAX",
        metrics(vapor_errors),
    )


score_txy("UNIFNIST_fit", fits["UNIFNIST"])
score_txy("UNIFNIST_stored", (-67.0, -161.4))
score_txy("UNIFNIST_strong", (-77.5, -185.2))
score_txy("UNIFNIST_Perry_PR", (-55.5569, -127.1714))
