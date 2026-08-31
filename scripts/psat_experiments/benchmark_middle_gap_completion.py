import csv
import math
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from scipy.interpolate import PchipInterpolator

from benchmark_direct_lower_completion import coolprop_cases
from benchmark_perry_aw import (
    load_curves,
    percentile,
    perry_dlnp_dt,
    perry_ln_p,
)
from property_resolution.vapor_pressure_adapter import (
    _ambrose_walton_dln_pressure_dT,
    _ambrose_walton_ln_pressure,
    _ambrose_walton_omega_from_pressure,
    _ambrose_walton_omega_sensitivity,
)


OUTPUT_PATH = Path("/tmp/middle_gap_completion_benchmark.csv")
GRID_POINTS = 241
SAMPLE_POINTS = 101
GAP_LOCATIONS = (
    ("low", 0.25),
    ("middle", 0.50),
    ("high", 0.75),
)
GAP_WIDTH_FRACTIONS = (0.10, 0.25, 0.50)
METHODS = (
    "linear_lnP",
    "hermite_temperature",
    "hermite_inverse_temperature",
    "hermite_log_temperature",
    "physical_omega_aw",
    "physical_omega_aw_c1_correction",
    "linear_endpoint_omega",
    "cubic_endpoint_omega_c1",
    "two_sided_dynamic_omega_blend",
)
_WORKER_REFERENCES = ()


@dataclass(frozen=True)
class ReferenceCase:
    dataset: str
    cas: str
    name: str
    Tc: float
    Pc_bar: float
    omega: float | None
    T_min: float
    T_max: float
    ln_pressure: object
    derivative: object


@dataclass(frozen=True)
class PreparedReference:
    case: ReferenceCase
    T_min: float
    T_max: float
    ln_pressure: object
    derivative: object
    temperature_from_ln_pressure: object
    ln_pressure_min: float
    ln_pressure_max: float


@dataclass(frozen=True)
class Model:
    ln_pressure: object
    derivative: object
    omega_min: float | None = None
    omega_max: float | None = None


def perry_reference_cases():
    return [
        ReferenceCase(
            dataset="Perry 2-8",
            cas=curve.cas,
            name=curve.name,
            Tc=curve.tc,
            Pc_bar=curve.pc_bar,
            omega=curve.omega,
            T_min=curve.t_min,
            T_max=min(curve.t_max, curve.tc),
            ln_pressure=(
                lambda temperature, curve=curve: perry_ln_p(
                    curve,
                    temperature,
                )
            ),
            derivative=(
                lambda temperature, curve=curve: perry_dlnp_dt(
                    curve,
                    temperature,
                )
            ),
        )
        for curve in load_curves()
    ]


def coolprop_reference_cases():
    cases, unavailable = coolprop_cases()
    references = [
        ReferenceCase(
            dataset="CoolProp HEOS",
            cas=case.cas,
            name=case.name,
            Tc=case.Tc,
            Pc_bar=case.Pc_bar,
            omega=case.preferred_omega,
            T_min=case.T_min,
            T_max=case.T_max,
            ln_pressure=case.ln_pressure,
            derivative=case.dln_pressure_dT,
        )
        for case in cases
    ]
    return references, unavailable


def prepare_reference(case):
    lower = max(
        case.T_min + 1.0e-5 * max(case.Tc, 1.0),
        0.35 * case.Tc,
    )
    upper = min(
        case.T_max - 1.0e-5 * max(case.Tc, 1.0),
        0.92 * case.Tc,
    )
    if not 0.0 < lower < upper < case.Tc:
        return None
    temperatures = np.linspace(lower, upper, GRID_POINTS)
    try:
        ln_pressures = np.asarray([
            float(case.ln_pressure(float(temperature)))
            for temperature in temperatures
        ])
    except Exception:
        return None
    if (
        not np.all(np.isfinite(ln_pressures))
        or np.any(np.diff(ln_pressures) <= 0.0)
        or ln_pressures[-1] - ln_pressures[0] < 1.0
    ):
        return None

    interpolator = PchipInterpolator(
        temperatures,
        ln_pressures,
        extrapolate=False,
    )
    interpolator_derivative = interpolator.derivative()
    inverse = PchipInterpolator(
        ln_pressures,
        temperatures,
        extrapolate=False,
    )
    ln_pressure_min = float(ln_pressures[0])
    ln_pressure_max = float(ln_pressures[-1])

    def temperature_from_ln_pressure(value):
        tolerance = 1.0e-12 * max(
            abs(ln_pressure_min),
            abs(ln_pressure_max),
            1.0,
        )
        if value <= ln_pressure_min + tolerance:
            return lower
        if value >= ln_pressure_max - tolerance:
            return upper
        return float(inverse(value))

    if case.dataset.startswith("CoolProp"):
        ln_pressure = lambda T: float(interpolator(T))
        derivative = lambda T: float(interpolator_derivative(T))
    else:
        ln_pressure = case.ln_pressure
        derivative = case.derivative
    return PreparedReference(
        case=case,
        T_min=lower,
        T_max=upper,
        ln_pressure=ln_pressure,
        derivative=derivative,
        temperature_from_ln_pressure=temperature_from_ln_pressure,
        ln_pressure_min=ln_pressure_min,
        ln_pressure_max=ln_pressure_max,
    )


def hermite_value_derivative(
    coordinate,
    left_coordinate,
    right_coordinate,
    left_value,
    right_value,
    left_derivative,
    right_derivative,
):
    span = right_coordinate - left_coordinate
    fraction = (coordinate - left_coordinate) / span
    fraction2 = fraction * fraction
    fraction3 = fraction2 * fraction
    h00 = 2.0 * fraction3 - 3.0 * fraction2 + 1.0
    h10 = fraction3 - 2.0 * fraction2 + fraction
    h01 = -2.0 * fraction3 + 3.0 * fraction2
    h11 = fraction3 - fraction2
    value = (
        h00 * left_value
        + h10 * span * left_derivative
        + h01 * right_value
        + h11 * span * right_derivative
    )
    derivative = (
        (6.0 * fraction2 - 6.0 * fraction) * left_value
        + (3.0 * fraction2 - 4.0 * fraction + 1.0)
        * span
        * left_derivative
        + (-6.0 * fraction2 + 6.0 * fraction) * right_value
        + (3.0 * fraction2 - 2.0 * fraction)
        * span
        * right_derivative
    ) / span
    return value, derivative


def coordinate_hermite_model(
    left_temperature,
    right_temperature,
    left_ln_pressure,
    right_ln_pressure,
    left_slope,
    right_slope,
    coordinate_function,
    coordinate_derivative,
):
    left_coordinate = coordinate_function(left_temperature)
    right_coordinate = coordinate_function(right_temperature)
    left_coordinate_slope = (
        left_slope / coordinate_derivative(left_temperature)
    )
    right_coordinate_slope = (
        right_slope / coordinate_derivative(right_temperature)
    )

    def values(temperature):
        coordinate = coordinate_function(temperature)
        value, derivative = hermite_value_derivative(
            coordinate,
            left_coordinate,
            right_coordinate,
            left_ln_pressure,
            right_ln_pressure,
            left_coordinate_slope,
            right_coordinate_slope,
        )
        return (
            value,
            derivative * coordinate_derivative(temperature),
        )

    return Model(
        ln_pressure=lambda temperature: values(temperature)[0],
        derivative=lambda temperature: values(temperature)[1],
    )


def endpoint_omega(reference, temperature, ln_pressure):
    return _ambrose_walton_omega_from_pressure(
        temperature,
        ln_pressure,
        reference.case.Tc,
        reference.case.Pc_bar,
        preferred_omega=reference.case.omega,
    )


def required_omega_slope(reference, temperature, omega, slope):
    sensitivity = _ambrose_walton_omega_sensitivity(
        temperature,
        reference.case.Tc,
        omega,
    )
    if abs(sensitivity) <= 1.0e-10:
        raise ValueError("AW omega sensitivity is singular")
    return (
        slope
        - _ambrose_walton_dln_pressure_dT(
            temperature,
            reference.case.Tc,
            omega,
        )
    ) / sensitivity


def aw_model(reference, omega_function, omega_derivative):
    def ln_pressure(temperature):
        return _ambrose_walton_ln_pressure(
            temperature,
            reference.case.Tc,
            reference.case.Pc_bar,
            omega_function(temperature),
        )

    def derivative(temperature):
        omega = omega_function(temperature)
        return (
            _ambrose_walton_dln_pressure_dT(
                temperature,
                reference.case.Tc,
                omega,
            )
            + _ambrose_walton_omega_sensitivity(
                temperature,
                reference.case.Tc,
                omega,
            )
            * omega_derivative(temperature)
        )

    return Model(ln_pressure=ln_pressure, derivative=derivative)


def build_model(
    reference,
    method,
    left_temperature,
    right_temperature,
    left_ln_pressure,
    right_ln_pressure,
    left_slope,
    right_slope,
):
    span = right_temperature - left_temperature
    if method == "linear_lnP":
        slope = (right_ln_pressure - left_ln_pressure) / span
        return Model(
            ln_pressure=lambda T: (
                left_ln_pressure + slope * (T - left_temperature)
            ),
            derivative=lambda _T: slope,
        )
    if method == "hermite_temperature":
        return coordinate_hermite_model(
            left_temperature,
            right_temperature,
            left_ln_pressure,
            right_ln_pressure,
            left_slope,
            right_slope,
            lambda T: T,
            lambda _T: 1.0,
        )
    if method == "hermite_inverse_temperature":
        return coordinate_hermite_model(
            left_temperature,
            right_temperature,
            left_ln_pressure,
            right_ln_pressure,
            left_slope,
            right_slope,
            lambda T: 1.0 / T,
            lambda T: -1.0 / T**2,
        )
    if method == "hermite_log_temperature":
        return coordinate_hermite_model(
            left_temperature,
            right_temperature,
            left_ln_pressure,
            right_ln_pressure,
            left_slope,
            right_slope,
            math.log,
            lambda T: 1.0 / T,
        )

    physical_omega = reference.case.omega
    if method in {
        "physical_omega_aw",
        "physical_omega_aw_c1_correction",
    }:
        if physical_omega is None or not math.isfinite(physical_omega):
            raise ValueError("physical omega unavailable")
        baseline = aw_model(
            reference,
            lambda _T: physical_omega,
            lambda _T: 0.0,
        )
        if method == "physical_omega_aw":
            return baseline
        left_value_residual = (
            left_ln_pressure - baseline.ln_pressure(left_temperature)
        )
        right_value_residual = (
            right_ln_pressure - baseline.ln_pressure(right_temperature)
        )
        left_slope_residual = (
            left_slope - baseline.derivative(left_temperature)
        )
        right_slope_residual = (
            right_slope - baseline.derivative(right_temperature)
        )

        def correction_values(temperature):
            return hermite_value_derivative(
                temperature,
                left_temperature,
                right_temperature,
                left_value_residual,
                right_value_residual,
                left_slope_residual,
                right_slope_residual,
            )

        return Model(
            ln_pressure=lambda T: (
                baseline.ln_pressure(T) + correction_values(T)[0]
            ),
            derivative=lambda T: (
                baseline.derivative(T) + correction_values(T)[1]
            ),
        )

    left_omega = endpoint_omega(
        reference,
        left_temperature,
        left_ln_pressure,
    )
    right_omega = endpoint_omega(
        reference,
        right_temperature,
        right_ln_pressure,
    )
    left_omega_slope = required_omega_slope(
        reference,
        left_temperature,
        left_omega,
        left_slope,
    )
    right_omega_slope = required_omega_slope(
        reference,
        right_temperature,
        right_omega,
        right_slope,
    )
    if method == "linear_endpoint_omega":
        omega_slope = (right_omega - left_omega) / span
        return aw_model(
            reference,
            lambda T: left_omega + omega_slope * (T - left_temperature),
            lambda _T: omega_slope,
        )
    if method == "cubic_endpoint_omega_c1":
        def omega_values(temperature):
            return hermite_value_derivative(
                temperature,
                left_temperature,
                right_temperature,
                left_omega,
                right_omega,
                left_omega_slope,
                right_omega_slope,
            )

        model = aw_model(
            reference,
            lambda T: omega_values(T)[0],
            lambda T: omega_values(T)[1],
        )
        omega_samples = [
            omega_values(
                left_temperature + span * index / 100.0
            )[0]
            for index in range(101)
        ]
        return Model(
            ln_pressure=model.ln_pressure,
            derivative=model.derivative,
            omega_min=min(omega_samples),
            omega_max=max(omega_samples),
        )
    if method == "two_sided_dynamic_omega_blend":
        left_model = aw_model(
            reference,
            lambda T: (
                left_omega
                + left_omega_slope * (T - left_temperature)
            ),
            lambda _T: left_omega_slope,
        )
        right_model = aw_model(
            reference,
            lambda T: (
                right_omega
                + right_omega_slope * (T - right_temperature)
            ),
            lambda _T: right_omega_slope,
        )

        def values(temperature):
            fraction = (temperature - left_temperature) / span
            weight = fraction**2 * (3.0 - 2.0 * fraction)
            weight_derivative = (
                6.0 * fraction * (1.0 - fraction) / span
            )
            left_value = left_model.ln_pressure(temperature)
            right_value = right_model.ln_pressure(temperature)
            value = (
                (1.0 - weight) * left_value
                + weight * right_value
            )
            derivative = (
                (1.0 - weight) * left_model.derivative(temperature)
                + weight * right_model.derivative(temperature)
                + weight_derivative * (right_value - left_value)
            )
            return value, derivative

        return Model(
            ln_pressure=lambda T: values(T)[0],
            derivative=lambda T: values(T)[1],
        )
    raise ValueError(f"unknown method {method}")


def evaluate_gap(reference, location, center_fraction, width_fraction):
    full_span = (
        reference.ln_pressure_max - reference.ln_pressure_min
    )
    left_fraction = center_fraction - 0.5 * width_fraction
    right_fraction = center_fraction + 0.5 * width_fraction
    left_target = reference.ln_pressure_min + left_fraction * full_span
    right_target = reference.ln_pressure_min + right_fraction * full_span
    left_temperature = reference.temperature_from_ln_pressure(left_target)
    right_temperature = reference.temperature_from_ln_pressure(right_target)
    left_ln_pressure = reference.ln_pressure(left_temperature)
    right_ln_pressure = reference.ln_pressure(right_temperature)
    left_slope = reference.derivative(left_temperature)
    right_slope = reference.derivative(right_temperature)
    if not (
        reference.T_min <= left_temperature < right_temperature <= reference.T_max
        and left_ln_pressure < right_ln_pressure
        and left_slope > 0.0
        and right_slope > 0.0
    ):
        return []

    reference_ln_pressures = np.linspace(
        left_ln_pressure,
        right_ln_pressure,
        SAMPLE_POINTS,
    )
    temperatures = [
        reference.temperature_from_ln_pressure(float(value))
        for value in reference_ln_pressures
    ]
    rows = []
    for method in METHODS:
        base = {
            "dataset": reference.case.dataset,
            "cas": reference.case.cas,
            "name": reference.case.name,
            "location": location,
            "center_fraction": center_fraction,
            "width_fraction": width_fraction,
            "pressure_span_decades": (
                (right_ln_pressure - left_ln_pressure) / math.log(10.0)
            ),
            "temperature_span_K": right_temperature - left_temperature,
            "left_reduced_temperature": left_temperature / reference.case.Tc,
            "right_reduced_temperature": right_temperature / reference.case.Tc,
            "method": method,
        }
        try:
            model = build_model(
                reference,
                method,
                left_temperature,
                right_temperature,
                left_ln_pressure,
                right_ln_pressure,
                left_slope,
                right_slope,
            )
            predicted = np.asarray([
                float(model.ln_pressure(temperature))
                for temperature in temperatures
            ])
            slopes = np.asarray([
                float(model.derivative(temperature))
                for temperature in temperatures
            ])
            if (
                not np.all(np.isfinite(predicted))
                or not np.all(np.isfinite(slopes))
            ):
                raise ValueError("non-finite model")
            relative_errors = np.abs(
                np.exp(predicted - reference_ln_pressures) - 1.0
            )
            monotonic = bool(
                np.all(np.diff(predicted) > 0.0)
                and np.all(slopes > 0.0)
            )
            left_value_error = abs(
                math.expm1(
                    model.ln_pressure(left_temperature)
                    - left_ln_pressure
                )
            )
            right_value_error = abs(
                math.expm1(
                    model.ln_pressure(right_temperature)
                    - right_ln_pressure
                )
            )
            left_slope_error = abs(
                model.derivative(left_temperature) / left_slope - 1.0
            )
            right_slope_error = abs(
                model.derivative(right_temperature) / right_slope - 1.0
            )
            rows.append({
                **base,
                "status": "ok",
                "mard": float(np.mean(relative_errors)),
                "p95_error": percentile(relative_errors.tolist(), 0.95),
                "maximum_error": float(np.max(relative_errors)),
                "monotonic": monotonic,
                "left_value_error": left_value_error,
                "right_value_error": right_value_error,
                "left_slope_error": left_slope_error,
                "right_slope_error": right_slope_error,
                "omega_min": model.omega_min,
                "omega_max": model.omega_max,
                "failure": "",
            })
        except Exception as error:
            rows.append({
                **base,
                "status": "failed",
                "mard": None,
                "p95_error": None,
                "maximum_error": None,
                "monotonic": False,
                "left_value_error": None,
                "right_value_error": None,
                "left_slope_error": None,
                "right_slope_error": None,
                "omega_min": None,
                "omega_max": None,
                "failure": str(error),
            })
    return rows


def method_summary(rows, method):
    selected = [
        row for row in rows
        if row["method"] == method and row["status"] == "ok"
    ]
    if not selected:
        return None
    mards = [row["mard"] for row in selected]
    maximum_errors = [row["maximum_error"] for row in selected]
    endpoint_slope_errors = [
        max(row["left_slope_error"], row["right_slope_error"])
        for row in selected
    ]
    return {
        "count": len(selected),
        "median_mard": median(mards),
        "p95_mard": percentile(mards, 0.95),
        "maximum_curve_mard": max(mards),
        "maximum_point_error": max(maximum_errors),
        "p95_endpoint_slope_error": percentile(
            endpoint_slope_errors,
            0.95,
        ),
        "nonmonotonic": sum(
            not row["monotonic"] for row in selected
        ),
    }


def report_group(rows, label):
    print(f"\n{label}: cases={len(rows) // len(METHODS)}")
    print(
        "method                                  "
        "median/p95/worst MARD   max point   slope p95   nonmono"
    )
    for method in METHODS:
        summary = method_summary(rows, method)
        if summary is None:
            print(f"{method:40s} unavailable")
            continue
        print(
            f"{method:40s}"
            f"{100 * summary['median_mard']:7.3f}/"
            f"{100 * summary['p95_mard']:7.3f}/"
            f"{100 * summary['maximum_curve_mard']:7.3f}% "
            f"{100 * summary['maximum_point_error']:9.3f}% "
            f"{100 * summary['p95_endpoint_slope_error']:9.3f}% "
            f"{summary['nonmonotonic']:7d}"
        )


def write_rows(rows):
    fieldnames = list(rows[0])
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_reference(index):
    case = _WORKER_REFERENCES[index]
    prepared = prepare_reference(case)
    if prepared is None:
        return [], (case.dataset, case.name)
    rows = []
    for location, center_fraction in GAP_LOCATIONS:
        for width_fraction in GAP_WIDTH_FRACTIONS:
            rows.extend(evaluate_gap(
                prepared,
                location,
                center_fraction,
                width_fraction,
            ))
    return rows, None


def main():
    global _WORKER_REFERENCES
    perry = perry_reference_cases()
    coolprop, coolprop_unavailable = coolprop_reference_cases()
    references = [*perry, *coolprop]
    _WORKER_REFERENCES = tuple(references)
    rows = []
    unavailable = []
    worker_count = min(
        8,
        max(1, int(os.environ.get(
            "PSAT_BENCHMARK_WORKERS",
            multiprocessing.cpu_count(),
        ))),
    )
    context = multiprocessing.get_context("fork")
    with context.Pool(worker_count) as pool:
        results = pool.imap_unordered(
            evaluate_reference,
            range(len(references)),
            chunksize=2,
        )
        for completed, (case_rows, unavailable_case) in enumerate(
            results,
            start=1,
        ):
            rows.extend(case_rows)
            if unavailable_case is not None:
                unavailable.append(unavailable_case)
            if completed % 50 == 0:
                print(
                    f"prepared {completed}/{len(references)} curves",
                    flush=True,
                )

    rows.sort(key=lambda row: (
        row["dataset"],
        row["name"],
        row["location"],
        row["width_fraction"],
        row["method"],
    ))

    print("MIDDLE-GAP PSAT COMPLETION")
    report_group(rows, "All references")
    for dataset in ("Perry 2-8", "CoolProp HEOS"):
        dataset_rows = [
            row for row in rows if row["dataset"] == dataset
        ]
        report_group(dataset_rows, dataset)
        for width_fraction in GAP_WIDTH_FRACTIONS:
            report_group(
                [
                    row for row in dataset_rows
                    if row["width_fraction"] == width_fraction
                ],
                f"{dataset}; width fraction={width_fraction:g}",
            )
        for location, _center_fraction in GAP_LOCATIONS:
            report_group(
                [
                    row for row in dataset_rows
                    if row["location"] == location
                ],
                f"{dataset}; location={location}",
            )
    write_rows(rows)
    print(f"\nunavailable prepared references={len(unavailable)}")
    print(f"CoolProp loader unavailable={len(coolprop_unavailable)}")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
