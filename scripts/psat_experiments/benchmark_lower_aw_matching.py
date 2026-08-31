import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from benchmark_perry_aw import (
    Curve,
    load_curves,
    percentile,
    perry_dlnp_dt,
    perry_ln_p,
)
from benchmark_table210_nannoolal_aw import tb_valid_matches
from benchmark_tb_constrained_aw import (
    aw_ln_p_with_omega,
    aw_terms,
    tb_constrained_omega,
)


SWITCH_PRESSURE_BAR = 0.25
BOUNDARY_PRESSURES_BAR = (
    5.0,
    3.0,
    2.0,
    1.5,
    1.25,
    1.01325,
    0.75,
    0.5,
    0.35,
)
TABLE_BOUNDARY_PRESSURES_BAR = (1.01325, 0.5, 0.35)
OUTPUT_PATH = Path("/tmp/lower_aw_matching_benchmark.csv")
SAMPLE_COUNT = 161

BASELINES = (
    "physical",
    "endpoint_constant",
    "endpoint_to_07",
    "endpoint_to_tb",
    "tb_variable",
)
CORRECTIONS = (
    "none",
    "constant_shift",
    "c1_affine_T",
    "c1_affine_invT",
    "c1_relax_T_cubic",
    "c1_relax_T_quintic",
    "c1_relax_invT_cubic",
    "c1_relax_logT_cubic",
)
DYNAMIC_OMEGA_METHODS = (
    "omega_slope_T",
    "omega_slope_invT",
    "omega_slope_logT",
    "omega_hermite_07",
)
TB_LOCALIZED_METHODS = (
    "endpoint_to_tb+c1_relax_tb_T_cubic",
    "endpoint_to_tb+c1_relax_tb_T_quintic",
    "endpoint_to_tb+c1_relax_tb_invT_cubic",
    "endpoint_to_tb+c1_relax_tb_logT_cubic",
)
METHODS = tuple(
    f"{baseline}+{correction}"
    for baseline in BASELINES
    for correction in CORRECTIONS
) + DYNAMIC_OMEGA_METHODS + TB_LOCALIZED_METHODS


@dataclass(frozen=True)
class ReferenceCase:
    dataset: str
    slope_basis: str
    curve: Curve
    boundary_pressure_bar: float
    boundary_temperature_K: float
    boundary_ln_pressure: float
    boundary_slope: float
    switch_temperature_K: float
    tb_K: float
    reference_ln_pressure: object
    reference_temperature: object


@dataclass(frozen=True)
class Model:
    method: str
    ln_pressure: object
    omega: object
    baseline_switch_temperature_K: float
    switch_temperature_K: float
    endpoint_omega: float | None
    endpoint_omega_slope: float | None


def numerical_derivative(function, temperature, scale):
    step = max(1.0e-5 * scale, 1.0e-4)
    return (
        function(temperature + step) - function(temperature - step)
    ) / (2.0 * step)


def pressure_crossing(function, pressure_bar, lower, upper):
    target = math.log(pressure_bar)
    if lower >= upper:
        return None
    sample_temperatures = np.linspace(lower, upper, 201)
    residuals = []
    for temperature in sample_temperatures:
        try:
            residual = function(float(temperature)) - target
        except (ArithmeticError, ValueError):
            residual = math.nan
        residuals.append(residual)
    brackets = []
    for left_index in range(len(sample_temperatures) - 1):
        left_value = residuals[left_index]
        right_value = residuals[left_index + 1]
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            continue
        if left_value == 0.0:
            brackets.append(
                (
                    float(sample_temperatures[left_index]),
                    float(sample_temperatures[left_index]),
                )
            )
        elif left_value * right_value < 0.0:
            brackets.append(
                (
                    float(sample_temperatures[left_index]),
                    float(sample_temperatures[left_index + 1]),
                )
            )
    if not brackets:
        if residuals[-1] == 0.0:
            return upper
        return None
    left, right = brackets[-1]
    if left == right:
        return left
    return brentq(lambda temperature: function(temperature) - target, left, right)


def monotonic_pressure_crossing(function, pressure_bar, lower, upper):
    target = math.log(pressure_bar)
    lower_residual = function(lower) - target
    upper_residual = function(upper) - target
    if lower_residual == 0.0:
        return lower
    if upper_residual == 0.0:
        return upper
    if lower_residual * upper_residual > 0.0:
        return None
    return brentq(
        lambda temperature: function(temperature) - target,
        lower,
        upper,
    )


def endpoint_omega(curve, temperature, ln_pressure):
    value, _roots = tb_constrained_omega(
        curve,
        temperature,
        math.exp(ln_pressure),
    )
    return value


def constant_omega_slope(curve, temperature, omega):
    return numerical_derivative(
        lambda value: aw_ln_p_with_omega(curve, value, omega),
        temperature,
        curve.tc,
    )


def omega_pressure_sensitivity(curve, temperature, omega):
    _f0, f1, f2 = aw_terms(temperature / curve.tc)
    return f1 + 2.0 * omega * f2


def required_omega_slope(case, omega):
    denominator = omega_pressure_sensitivity(
        case.curve,
        case.boundary_temperature_K,
        omega,
    )
    if abs(denominator) < 1.0e-10:
        return None
    constant_slope = constant_omega_slope(
        case.curve,
        case.boundary_temperature_K,
        omega,
    )
    return (case.boundary_slope - constant_slope) / denominator


def linear_omega(first_temperature, first_omega, second_temperature, second_omega):
    if abs(second_temperature - first_temperature) < 1.0e-8:
        return lambda _temperature: first_omega
    slope = (
        second_omega - first_omega
    ) / (second_temperature - first_temperature)
    return lambda temperature: first_omega + slope * (
        temperature - first_temperature
    )


def baseline(case, name):
    curve = case.curve
    boundary_temperature = case.boundary_temperature_K
    omega_endpoint = endpoint_omega(
        curve,
        boundary_temperature,
        case.boundary_ln_pressure,
    )
    if omega_endpoint is None:
        return None
    omega_tb = endpoint_omega(
        curve,
        case.tb_K,
        case.reference_ln_pressure(case.tb_K),
    )
    if omega_tb is None:
        return None
    if name == "physical":
        omega_function = lambda _temperature: curve.omega
    elif name == "endpoint_constant":
        omega_function = lambda _temperature: omega_endpoint
    elif name == "endpoint_to_07":
        omega_function = linear_omega(
            boundary_temperature,
            omega_endpoint,
            0.7 * curve.tc,
            curve.omega,
        )
    elif name == "endpoint_to_tb":
        omega_function = linear_omega(
            boundary_temperature,
            omega_endpoint,
            case.tb_K,
            omega_tb,
        )
    elif name == "tb_variable":
        if case.tb_K >= 0.7 * curve.tc:
            omega_function = lambda _temperature: curve.omega
        else:
            lower_function = linear_omega(
                case.tb_K,
                omega_tb,
                0.7 * curve.tc,
                curve.omega,
            )
            omega_function = lambda temperature: (
                curve.omega
                if temperature >= 0.7 * curve.tc
                else lower_function(temperature)
            )
    else:
        raise ValueError(f"Unsupported baseline: {name}")
    pressure_function = lambda temperature: aw_ln_p_with_omega(
        curve,
        temperature,
        omega_function(temperature),
    )
    lower = max(1.0, 0.15 * curve.tc)
    switch_temperature = pressure_crossing(
        pressure_function,
        SWITCH_PRESSURE_BAR,
        lower,
        boundary_temperature,
    )
    if switch_temperature is None:
        return None
    return pressure_function, omega_function, switch_temperature, omega_endpoint


def relaxed_correction(
    coordinate_name,
    order,
    switch_temperature,
    boundary_temperature,
    delta_value,
    delta_slope,
):
    if coordinate_name == "T":
        denominator = boundary_temperature - switch_temperature
        coordinate = lambda temperature: (
            temperature - switch_temperature
        ) / denominator
        boundary_coordinate_slope = 1.0 / denominator
    elif coordinate_name == "invT":
        denominator = (
            1.0 / switch_temperature - 1.0 / boundary_temperature
        )
        coordinate = lambda temperature: (
            1.0 / switch_temperature - 1.0 / temperature
        ) / denominator
        boundary_coordinate_slope = (
            1.0 / boundary_temperature**2 / denominator
        )
    elif coordinate_name == "logT":
        denominator = math.log(boundary_temperature / switch_temperature)
        coordinate = lambda temperature: math.log(
            temperature / switch_temperature
        ) / denominator
        boundary_coordinate_slope = (
            1.0 / boundary_temperature / denominator
        )
    else:
        raise ValueError(f"Unsupported coordinate: {coordinate_name}")
    endpoint_coordinate_slope = delta_slope / boundary_coordinate_slope
    if order == "cubic":
        return lambda temperature: (
            delta_value
            * (
                -2.0 * coordinate(temperature) ** 3
                + 3.0 * coordinate(temperature) ** 2
            )
            + endpoint_coordinate_slope
            * (
                coordinate(temperature) ** 3
                - coordinate(temperature) ** 2
            )
        )
    matrix = np.array(
        (
            (1.0, 1.0, 1.0),
            (3.0, 4.0, 5.0),
            (6.0, 12.0, 20.0),
        )
    )
    coefficients = np.linalg.solve(
        matrix,
        np.array((delta_value, endpoint_coordinate_slope, 0.0)),
    )
    return lambda temperature: (
        coefficients[0] * coordinate(temperature) ** 3
        + coefficients[1] * coordinate(temperature) ** 4
        + coefficients[2] * coordinate(temperature) ** 5
    )


def corrected_model(
    case,
    baseline_name,
    correction_name,
    base=None,
):
    if base is None:
        base = baseline(case, baseline_name)
    if base is None:
        return None
    (
        baseline_pressure,
        omega_function,
        baseline_switch_temperature,
        omega_endpoint,
    ) = base
    boundary_temperature = case.boundary_temperature_K
    baseline_boundary_slope = numerical_derivative(
        baseline_pressure,
        boundary_temperature,
        case.curve.tc,
    )
    delta_value = (
        case.boundary_ln_pressure
        - baseline_pressure(boundary_temperature)
    )
    delta_slope = case.boundary_slope - baseline_boundary_slope
    if correction_name == "none":
        correction = lambda _temperature: 0.0
    elif correction_name == "constant_shift":
        correction = lambda _temperature: delta_value
    elif correction_name == "c1_affine_T":
        correction = lambda temperature: (
            delta_value
            + delta_slope * (temperature - boundary_temperature)
        )
    elif correction_name == "c1_affine_invT":
        coefficient = -delta_slope * boundary_temperature**2
        correction = lambda temperature: (
            delta_value
            + coefficient
            * (1.0 / temperature - 1.0 / boundary_temperature)
        )
    elif correction_name.startswith("c1_relax_"):
        parts = correction_name.split("_")
        if parts[2] == "tb":
            if boundary_temperature <= case.tb_K + 1.0e-8:
                return None
            _prefix, _relax, _tb, coordinate_name, order = parts
            upper_correction = relaxed_correction(
                coordinate_name,
                order,
                case.tb_K,
                boundary_temperature,
                delta_value,
                delta_slope,
            )
            correction = lambda temperature: (
                0.0
                if temperature <= case.tb_K
                else upper_correction(temperature)
            )
        else:
            _prefix, _relax, coordinate_name, order = parts
            correction = relaxed_correction(
                coordinate_name,
                order,
                baseline_switch_temperature,
                boundary_temperature,
                delta_value,
                delta_slope,
            )
    else:
        raise ValueError(f"Unsupported correction: {correction_name}")
    pressure_function = lambda temperature: (
        baseline_pressure(temperature) + correction(temperature)
    )
    if correction_name.startswith("c1_relax_") or correction_name == "none":
        switch_temperature = baseline_switch_temperature
    else:
        switch_temperature = pressure_crossing(
            pressure_function,
            SWITCH_PRESSURE_BAR,
            max(1.0, 0.15 * case.curve.tc),
            boundary_temperature,
        )
    if switch_temperature is None:
        return None
    return Model(
        method=f"{baseline_name}+{correction_name}",
        ln_pressure=pressure_function,
        omega=omega_function,
        baseline_switch_temperature_K=baseline_switch_temperature,
        switch_temperature_K=switch_temperature,
        endpoint_omega=omega_endpoint,
        endpoint_omega_slope=None,
    )


def dynamic_omega_model(case, method):
    curve = case.curve
    boundary_temperature = case.boundary_temperature_K
    omega_endpoint = endpoint_omega(
        curve,
        boundary_temperature,
        case.boundary_ln_pressure,
    )
    if omega_endpoint is None:
        return None
    omega_slope = required_omega_slope(case, omega_endpoint)
    if omega_slope is None:
        return None
    if method == "omega_slope_T":
        omega_function = lambda temperature: (
            omega_endpoint
            + omega_slope * (temperature - boundary_temperature)
        )
    elif method == "omega_slope_invT":
        coefficient = -omega_slope * boundary_temperature**2
        omega_function = lambda temperature: (
            omega_endpoint
            + coefficient
            * (1.0 / temperature - 1.0 / boundary_temperature)
        )
    elif method == "omega_slope_logT":
        coefficient = omega_slope * boundary_temperature
        omega_function = lambda temperature: (
            omega_endpoint
            + coefficient * math.log(temperature / boundary_temperature)
        )
    elif method == "omega_hermite_07":
        target_temperature = 0.7 * curve.tc
        span = target_temperature - boundary_temperature
        if abs(span) < 1.0e-8:
            return None

        def omega_function(temperature):
            coordinate = (temperature - boundary_temperature) / span
            h00 = 2.0 * coordinate**3 - 3.0 * coordinate**2 + 1.0
            h10 = coordinate**3 - 2.0 * coordinate**2 + coordinate
            h01 = -2.0 * coordinate**3 + 3.0 * coordinate**2
            return (
                h00 * omega_endpoint
                + h10 * span * omega_slope
                + h01 * curve.omega
            )
    else:
        raise ValueError(f"Unsupported dynamic omega method: {method}")
    pressure_function = lambda temperature: aw_ln_p_with_omega(
        curve,
        temperature,
        omega_function(temperature),
    )
    switch_temperature = pressure_crossing(
        pressure_function,
        SWITCH_PRESSURE_BAR,
        max(1.0, 0.15 * curve.tc),
        boundary_temperature,
    )
    if switch_temperature is None:
        return None
    return Model(
        method=method,
        ln_pressure=pressure_function,
        omega=omega_function,
        baseline_switch_temperature_K=switch_temperature,
        switch_temperature_K=switch_temperature,
        endpoint_omega=omega_endpoint,
        endpoint_omega_slope=omega_slope,
    )


def build_model(case, method, baseline_cache=None):
    if method in DYNAMIC_OMEGA_METHODS:
        return dynamic_omega_model(case, method)
    baseline_name, correction_name = method.split("+", 1)
    base = (
        None
        if baseline_cache is None
        else baseline_cache.get(baseline_name)
    )
    return corrected_model(
        case,
        baseline_name,
        correction_name,
        base=base,
    )


def sample_reference_temperatures(case):
    pressures = np.exp(
        np.linspace(
            math.log(SWITCH_PRESSURE_BAR),
            math.log(case.boundary_pressure_bar),
            SAMPLE_COUNT,
        )
    )
    return [
        float(case.reference_temperature(float(pressure)))
        for pressure in pressures
    ]


def evaluate_model(case, method, temperatures, model=None):
    if model is None:
        model = build_model(case, method)
    if model is None:
        return {
            "dataset": case.dataset,
            "slope_basis": case.slope_basis,
            "boundary_pressure_bar": case.boundary_pressure_bar,
            "cas": case.curve.cas,
            "name": case.curve.name,
            "method": method,
            "failed": True,
        }
    signed_errors = []
    diverged = False
    for temperature in temperatures:
        try:
            predicted = model.ln_pressure(temperature)
        except (ArithmeticError, ValueError):
            predicted = math.inf
        reference = case.reference_ln_pressure(temperature)
        log_error = predicted - reference
        if not math.isfinite(log_error) or log_error > 50.0:
            diverged = True
            signed_errors.append(1.0e6)
        elif log_error < -50.0:
            diverged = True
            signed_errors.append(-1.0)
        else:
            signed_errors.append(math.exp(log_error) - 1.0)
    probe_minimum = min(
        case.switch_temperature_K,
        model.switch_temperature_K,
    )
    probe_temperatures = np.linspace(
        probe_minimum,
        case.boundary_temperature_K,
        401,
    )
    predicted = []
    for temperature in probe_temperatures:
        try:
            value = model.ln_pressure(float(temperature))
        except (ArithmeticError, ValueError):
            value = math.inf
        predicted.append(value)
    derivatives = np.diff(predicted) / np.diff(probe_temperatures)
    endpoint_value_error = math.exp(
        model.ln_pressure(case.boundary_temperature_K)
        - case.boundary_ln_pressure
    ) - 1.0
    endpoint_slope = numerical_derivative(
        model.ln_pressure,
        case.boundary_temperature_K,
        case.curve.tc,
    )
    switch_temperature = model.switch_temperature_K
    switch_slope = numerical_derivative(
        model.ln_pressure,
        switch_temperature,
        case.curve.tc,
    )
    reference_switch_slope = numerical_derivative(
        case.reference_ln_pressure,
        switch_temperature,
        case.curve.tc,
    )
    switch_omega = model.omega(switch_temperature)
    return {
        "dataset": case.dataset,
        "slope_basis": case.slope_basis,
        "boundary_pressure_bar": case.boundary_pressure_bar,
        "cas": case.curve.cas,
        "name": case.curve.name,
        "method": method,
        "failed": False,
        "diverged": diverged,
        "boundary_temperature_K": case.boundary_temperature_K,
        "boundary_Tr": case.boundary_temperature_K / case.curve.tc,
        "reference_switch_temperature_K": case.switch_temperature_K,
        "predicted_switch_temperature_K": switch_temperature,
        "switch_temperature_error_K": switch_temperature - case.switch_temperature_K,
        "switch_slope_error": (
            switch_slope / reference_switch_slope - 1.0
        ),
        "mard": mean(abs(error) for error in signed_errors),
        "maximum_error": max(abs(error) for error in signed_errors),
        "signed_switch_pressure_error": signed_errors[0],
        "endpoint_value_error": endpoint_value_error,
        "endpoint_slope_error": endpoint_slope / case.boundary_slope - 1.0,
        "nonmonotone": bool(np.min(derivatives) <= 0.0),
        "endpoint_omega": model.endpoint_omega,
        "endpoint_omega_slope": model.endpoint_omega_slope,
        "switch_omega": switch_omega,
        "switch_omega_change": (
            None
            if model.endpoint_omega is None
            else switch_omega - model.endpoint_omega
        ),
    }


def continuous_cases():
    cases = []
    for curve in load_curves():
        lower = curve.t_min
        upper = min(curve.t_max, curve.tc - 1.0e-7)
        if lower >= upper:
            continue
        reference = lambda temperature, curve=curve: perry_ln_p(
            curve,
            temperature,
        )
        switch_temperature = monotonic_pressure_crossing(
            reference,
            SWITCH_PRESSURE_BAR,
            lower,
            upper,
        )
        tb = monotonic_pressure_crossing(reference, 1.01325, lower, upper)
        if switch_temperature is None or tb is None:
            continue
        inverse = lambda pressure, reference=reference, lower=lower, upper=upper: (
            monotonic_pressure_crossing(reference, pressure, lower, upper)
        )
        for boundary_pressure in BOUNDARY_PRESSURES_BAR:
            boundary_temperature = monotonic_pressure_crossing(
                reference,
                boundary_pressure,
                lower,
                upper,
            )
            if (
                boundary_temperature is None
                or boundary_temperature <= switch_temperature + 1.0e-7
            ):
                continue
            cases.append(
                ReferenceCase(
                    dataset="Perry 2-8 continuous",
                    slope_basis="analytic",
                    curve=curve,
                    boundary_pressure_bar=boundary_pressure,
                    boundary_temperature_K=boundary_temperature,
                    boundary_ln_pressure=math.log(boundary_pressure),
                    boundary_slope=perry_dlnp_dt(
                        curve,
                        boundary_temperature,
                    ),
                    switch_temperature_K=switch_temperature,
                    tb_K=tb,
                    reference_ln_pressure=reference,
                    reference_temperature=inverse,
                )
            )
    return cases


def table_cases():
    cases = []
    for curve, pairs, trusted_tb in tb_valid_matches():
        temperatures = np.array([temperature for temperature, _pressure in pairs])
        ln_pressures = np.log([pressure for _temperature, pressure in pairs])
        reference_interpolator = PchipInterpolator(temperatures, ln_pressures)
        derivative = reference_interpolator.derivative()
        reference = lambda temperature, interpolator=reference_interpolator: float(
            interpolator(temperature)
        )
        lower = float(temperatures[0])
        upper = float(temperatures[-1])
        switch_temperature = monotonic_pressure_crossing(
            reference,
            SWITCH_PRESSURE_BAR,
            lower,
            upper,
        )
        if switch_temperature is None:
            continue
        inverse = lambda pressure, reference=reference, lower=lower, upper=upper: (
            monotonic_pressure_crossing(reference, pressure, lower, upper)
        )
        for boundary_pressure in TABLE_BOUNDARY_PRESSURES_BAR:
            boundary_temperature = monotonic_pressure_crossing(
                reference,
                boundary_pressure,
                lower,
                upper,
            )
            if (
                boundary_temperature is None
                or boundary_temperature <= switch_temperature + 1.0e-7
            ):
                continue
            oracle_slope = float(derivative(boundary_temperature))
            high_pairs = [
                (temperature, pressure)
                for temperature, pressure in pairs
                if temperature > boundary_temperature + 1.0e-8
            ]
            high_temperatures = np.array(
                [boundary_temperature]
                + [temperature for temperature, _pressure in high_pairs]
            )
            high_ln_pressures = np.array(
                [math.log(boundary_pressure)]
                + [math.log(pressure) for _temperature, pressure in high_pairs]
            )
            slope_options = [("full_pchip_oracle", oracle_slope)]
            if len(high_temperatures) >= 3:
                upper_interpolator = PchipInterpolator(
                    high_temperatures,
                    high_ln_pressures,
                )
                slope_options.append(
                    (
                        "upper_only_pchip",
                        float(upper_interpolator.derivative()(boundary_temperature)),
                    )
                )
            for slope_basis, slope in slope_options:
                cases.append(
                    ReferenceCase(
                        dataset="Perry 2-10 PCHIP",
                        slope_basis=slope_basis,
                        curve=curve,
                        boundary_pressure_bar=boundary_pressure,
                        boundary_temperature_K=boundary_temperature,
                        boundary_ln_pressure=math.log(boundary_pressure),
                        boundary_slope=slope,
                        switch_temperature_K=switch_temperature,
                        tb_K=trusted_tb,
                        reference_ln_pressure=reference,
                        reference_temperature=inverse,
                    )
                )
    return cases


def method_summary(rows, method, expected_count):
    selected = [
        row for row in rows
        if row["method"] == method and not row["failed"]
    ]
    if not selected:
        return None
    mards = [row["mard"] for row in selected]
    maxima = [row["maximum_error"] for row in selected]
    switch_errors = [
        abs(row["switch_temperature_error_K"])
        for row in selected
    ]
    return {
        "method": method,
        "count": len(selected),
        "failed": expected_count - len(selected),
        "mean_mard": mean(mards),
        "median_mard": median(mards),
        "p95_mard": percentile(mards, 0.95),
        "maximum_mard": max(mards),
        "median_maximum": median(maxima),
        "p95_maximum": percentile(maxima, 0.95),
        "median_switch_temperature_error": median(switch_errors),
        "p95_switch_temperature_error": percentile(switch_errors, 0.95),
        "nonmonotone": sum(row["nonmonotone"] for row in selected),
        "diverged": sum(row["diverged"] for row in selected),
    }


def report_group(rows, label, expected_count):
    print(f"\n{label}: cases={expected_count}")
    summaries = [
        method_summary(rows, method, expected_count)
        for method in METHODS
    ]
    summaries = [summary for summary in summaries if summary is not None]
    summaries.sort(
        key=lambda summary: (
            summary["p95_mard"],
            summary["median_mard"],
        )
    )
    print(
        "method".ljust(43)
        + " n/fail  medMARD  p95MARD  maxMARD  "
        + "p95|max|  med|dT| p95|dT| nonmono diverge"
    )
    for summary in summaries:
        print(
            summary["method"].ljust(43)
            + f" {summary['count']:3d}/{summary['failed']:<3d}"
            + f" {100 * summary['median_mard']:8.3f}"
            + f" {100 * summary['p95_mard']:8.3f}"
            + f" {100 * summary['maximum_mard']:8.3f}"
            + f" {100 * summary['p95_maximum']:8.3f}"
            + f" {summary['median_switch_temperature_error']:8.3f}"
            + f" {summary['p95_switch_temperature_error']:8.3f}"
            + f" {summary['nonmonotone']:7d}"
            + f" {summary['diverged']:7d}"
        )
    return summaries


def report(rows, cases):
    group_keys = sorted(
        {
            (
                case.dataset,
                case.slope_basis,
                case.boundary_pressure_bar,
            )
            for case in cases
        }
    )
    summaries = {}
    for dataset, slope_basis, boundary_pressure in group_keys:
        selected_cases = [
            case for case in cases
            if (
                case.dataset == dataset
                and case.slope_basis == slope_basis
                and case.boundary_pressure_bar == boundary_pressure
            )
        ]
        selected_rows = [
            row for row in rows
            if (
                row["dataset"] == dataset
                and row["slope_basis"] == slope_basis
                and row["boundary_pressure_bar"] == boundary_pressure
            )
        ]
        label = (
            f"{dataset}; slope={slope_basis}; "
            f"hard endpoint={boundary_pressure:g} bar"
        )
        summaries[(dataset, slope_basis, boundary_pressure)] = report_group(
            selected_rows,
            label,
            len(selected_cases),
        )
    acid_cases = [
        case
        for case in cases
        if (
            case.dataset == "Perry 2-8 continuous"
            and "acid" in case.curve.name.lower()
        )
    ]
    acid_rows = [
        row
        for row in rows
        if (
            row["dataset"] == "Perry 2-8 continuous"
            and "acid" in row["name"].lower()
        )
    ]
    if acid_cases:
        report_group(
            acid_rows,
            "Perry 2-8 continuous acids; all boundary pressures",
            len(acid_cases),
        )
    return summaries


def write_rows(rows):
    keys = sorted({key for row in rows for key in row})
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    cases = continuous_cases() + table_cases()
    print("LOWER AMBROSE-WALTON HARD-ENDPOINT MATCHING")
    print(f"cases={len(cases)} methods={len(METHODS)}")
    rows = []
    for case_index, case in enumerate(cases, 1):
        temperatures = sample_reference_temperatures(case)
        baseline_cache = {
            name: baseline(case, name)
            for name in BASELINES
        }
        for method in METHODS:
            model = build_model(
                case,
                method,
                baseline_cache=baseline_cache,
            )
            rows.append(
                evaluate_model(
                    case,
                    method,
                    temperatures,
                    model=model,
                )
            )
        if case_index % 100 == 0:
            print(f"evaluated {case_index}/{len(cases)} cases")
    write_rows(rows)
    report(rows, cases)
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
