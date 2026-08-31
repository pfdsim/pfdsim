import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.integrate import solve_ivp

from benchmark_direct_lower_completion import perry_cases
from benchmark_end_to_end_lower_completion import (
    HARD_PRESSURES_BAR,
    MONOCARBOXYLIC_ACID_CAS,
    SWITCH_PRESSURE_BAR,
    TARGET_PRESSURES_BAR,
    dimer_predictions,
    production_lower_aw_boundary,
)
from benchmark_perry_aw import percentile
from property_resolution.common import R


OUTPUT_PATH = Path("/tmp/hard_dimer_parameter_fit_benchmark.csv")
FIT_SPANS_PRESSURE_DECADES = (0.15, 0.30, 0.50)
FIT_SAMPLE_COUNT = 31


@dataclass(frozen=True)
class DimerParameterFit:
    enthalpy_J_mol: float
    entropy_J_mol_K: float
    sample_count: int
    r_squared: float
    log_equilibrium_rmse: float
    minimum_extent: float
    maximum_extent: float


def inferred_extent(case, temperature):
    pressure_bar = math.exp(case.ln_pressure(temperature))
    apparent_enthalpy = (
        R * temperature**2 * case.dln_pressure_dT(temperature)
    )
    monomer_hvap = case.hvap_J_mol(temperature)
    extent = 1.0 - monomer_hvap / apparent_enthalpy
    if (
        not math.isfinite(pressure_bar)
        or not math.isfinite(extent)
        or not 1.0e-4 < extent < 0.4999
    ):
        return None
    pressure_function = (
        extent
        * (1.0 - extent)
        / (1.0 - 2.0 * extent) ** 2
    )
    if pressure_function <= 0.0:
        return None
    return (
        1.0 / temperature,
        math.log(pressure_function / pressure_bar),
        extent,
    )


def fit_dimer_parameters(case, lower_temperature, upper_temperature):
    observations = [
        inferred_extent(case, float(temperature))
        for temperature in np.linspace(
            lower_temperature,
            upper_temperature,
            FIT_SAMPLE_COUNT,
        )
    ]
    observations = [
        observation
        for observation in observations
        if observation is not None
    ]
    if len(observations) < 6:
        return None

    inverse_temperatures = np.asarray([
        observation[0] for observation in observations
    ])
    log_equilibrium_constants = np.asarray([
        observation[1] for observation in observations
    ])
    centered_inverse_temperatures = (
        inverse_temperatures - inverse_temperatures.mean()
    )
    if np.ptp(inverse_temperatures) <= 1.0e-8:
        return None
    design = np.column_stack((
        np.ones(len(observations)),
        centered_inverse_temperatures,
    ))
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
        design,
        log_equilibrium_constants,
        rcond=None,
    )
    if rank < 2:
        return None
    predictions = design @ coefficients
    residual = log_equilibrium_constants - predictions
    residual_sum = float(residual @ residual)
    centered_response = (
        log_equilibrium_constants
        - log_equilibrium_constants.mean()
    )
    total_sum = float(centered_response @ centered_response)
    slope = float(coefficients[1])
    intercept = float(
        coefficients[0] - slope * inverse_temperatures.mean()
    )
    extents = [observation[2] for observation in observations]
    return DimerParameterFit(
        enthalpy_J_mol=-R * slope,
        entropy_J_mol_K=R * intercept,
        sample_count=len(observations),
        r_squared=(
            1.0 - residual_sum / total_sum
            if total_sum > 0.0
            else 1.0
        ),
        log_equilibrium_rmse=math.sqrt(
            residual_sum / len(observations)
        ),
        minimum_extent=min(extents),
        maximum_extent=max(extents),
    )


def dimer_extent(temperature, ln_pressure, fit):
    exponent = (
        fit.entropy_J_mol_K / R
        - fit.enthalpy_J_mol / (R * temperature)
        + ln_pressure
    )
    equilibrium_pressure_product = math.exp(
        min(700.0, max(-700.0, exponent))
    )
    return 0.5 * (
        1.0
        - 1.0 / math.sqrt(
            1.0 + 4.0 * equilibrium_pressure_product
        )
    )


def fitted_dimer_predictions(boundary, targets, fit):
    if not targets:
        return {}
    minimum_temperature = min(
        temperature for _pressure, temperature in targets
    )

    def derivative(temperature, state):
        extent = dimer_extent(temperature, state[0], fit)
        return [
            boundary.case.hvap_J_mol(temperature)
            / (R * temperature**2 * (1.0 - extent))
        ]

    solution = solve_ivp(
        derivative,
        (
            boundary.switch_temperature_K,
            minimum_temperature,
        ),
        [math.log(SWITCH_PRESSURE_BAR)],
        rtol=2.0e-9,
        atol=2.0e-11,
        dense_output=True,
        max_step=max(
            (
                boundary.switch_temperature_K
                - minimum_temperature
            )
            / 25.0,
            0.1,
        ),
    )
    if not solution.success:
        return {}
    return {
        pressure: float(solution.sol(temperature)[0])
        for pressure, temperature in targets
    }


def target_states(boundary):
    targets = []
    for pressure in TARGET_PRESSURES_BAR:
        temperature = boundary.case.temperature_at_pressure(pressure)
        if (
            temperature is not None
            and boundary.case.T_min <= temperature
            and temperature < boundary.switch_temperature_K
        ):
            targets.append((pressure, temperature))
    return targets


def evaluate():
    cases, _unavailable = perry_cases()
    acid_cases = [
        case
        for case in cases
        if case.cas in MONOCARBOXYLIC_ACID_CAS
    ]
    rows = []
    unavailable = []
    for case in acid_cases:
        for hard_pressure in HARD_PRESSURES_BAR:
            boundary = production_lower_aw_boundary(
                case,
                hard_pressure,
            )
            if boundary is None:
                unavailable.append((
                    case.name,
                    hard_pressure,
                    "lower boundary unavailable",
                ))
                continue
            targets = target_states(boundary)
            fixed_predictions = dimer_predictions(
                boundary,
                targets,
            )[0]
            for fit_span in FIT_SPANS_PRESSURE_DECADES:
                fit_upper_pressure = (
                    hard_pressure * 10.0**fit_span
                )
                fit_upper_temperature = (
                    case.temperature_at_pressure(fit_upper_pressure)
                )
                if fit_upper_temperature is None:
                    unavailable.append((
                        case.name,
                        hard_pressure,
                        f"{fit_span:g}-decade fit range unavailable",
                    ))
                    continue
                fit = fit_dimer_parameters(
                    case,
                    boundary.hard_temperature_K,
                    fit_upper_temperature,
                )
                if fit is None:
                    unavailable.append((
                        case.name,
                        hard_pressure,
                        f"{fit_span:g}-decade fit underdetermined",
                    ))
                    continue
                fitted_predictions = fitted_dimer_predictions(
                    boundary,
                    targets,
                    fit,
                )
                fitted_switch_extent = dimer_extent(
                    boundary.switch_temperature_K,
                    math.log(SWITCH_PRESSURE_BAR),
                    fit,
                )
                fitted_switch_slope = (
                    case.hvap_J_mol(
                        boundary.switch_temperature_K
                    )
                    / (
                        R
                        * boundary.switch_temperature_K**2
                        * (1.0 - fitted_switch_extent)
                    )
                )
                for pressure, temperature in targets:
                    reference = case.ln_pressure(temperature)
                    rows.append({
                        "cas": case.cas,
                        "name": case.name,
                        "hard_pressure_bar": hard_pressure,
                        "fit_span_pressure_decades": fit_span,
                        "target_pressure_bar": pressure,
                        "target_temperature_K": temperature,
                        "fitted_enthalpy_J_mol": (
                            fit.enthalpy_J_mol
                        ),
                        "fitted_entropy_J_mol_K": (
                            fit.entropy_J_mol_K
                        ),
                        "fit_sample_count": fit.sample_count,
                        "fit_r_squared": fit.r_squared,
                        "fit_log_equilibrium_rmse": (
                            fit.log_equilibrium_rmse
                        ),
                        "fit_minimum_extent": fit.minimum_extent,
                        "fit_maximum_extent": fit.maximum_extent,
                        "fit_extent_span": (
                            fit.maximum_extent
                            - fit.minimum_extent
                        ),
                        "switch_slope_mismatch": (
                            fitted_switch_slope
                            / boundary.switch_slope
                            - 1.0
                        ),
                        "fitted_relative_error": math.expm1(
                            fitted_predictions[pressure] - reference
                        ),
                        "fixed_relative_error": math.expm1(
                            fixed_predictions[pressure] - reference
                        ),
                    })
    return rows, unavailable


def method_statistics(rows, field_name):
    point_errors = [
        abs(float(row[field_name])) for row in rows
    ]
    case_errors = {}
    for row in rows:
        key = (
            row["cas"],
            row["hard_pressure_bar"],
        )
        case_errors.setdefault(key, []).append(
            abs(float(row[field_name]))
        )
    curve_mards = [
        mean(values) for values in case_errors.values()
    ]
    return (
        median(point_errors),
        percentile(point_errors, 0.95),
        max(point_errors),
        median(curve_mards),
        percentile(curve_mards, 0.95),
        max(curve_mards),
    )


def report(rows):
    for fit_span in FIT_SPANS_PRESSURE_DECADES:
        span_rows = [
            row for row in rows
            if row["fit_span_pressure_decades"] == fit_span
        ]
        fits = {}
        for row in span_rows:
            fits[(row["cas"], row["hard_pressure_bar"])] = row
        enthalpies = [
            row["fitted_enthalpy_J_mol"] for row in fits.values()
        ]
        print(
            f"\nfit span={fit_span:g} pressure decades; "
            f"fits={len(fits)}"
        )
        print(
            "  fitted enthalpy kJ/mol median/p05/p95: "
            f"{median(enthalpies) / 1000.0:.3f}/"
            f"{percentile(enthalpies, 0.05) / 1000.0:.3f}/"
            f"{percentile(enthalpies, 0.95) / 1000.0:.3f}"
        )
        for field_name, label in (
            ("fitted_relative_error", "fitted H/S"),
            ("fixed_relative_error", "fixed H, boundary S"),
        ):
            statistics = method_statistics(span_rows, field_name)
            print(
                f"  {label:19s} "
                f"point={100 * statistics[0]:.3f}/"
                f"{100 * statistics[1]:.3f}/"
                f"{100 * statistics[2]:.3f} "
                f"curve={100 * statistics[3]:.3f}/"
                f"{100 * statistics[4]:.3f}/"
                f"{100 * statistics[5]:.3f}"
            )
        if fit_span == FIT_SPANS_PRESSURE_DECADES[0]:
            for hard_pressure in HARD_PRESSURES_BAR:
                subset = [
                    row for row in span_rows
                    if row["hard_pressure_bar"] == hard_pressure
                ]
                fitted = method_statistics(
                    subset,
                    "fitted_relative_error",
                )
                fixed = method_statistics(
                    subset,
                    "fixed_relative_error",
                )
                print(
                    f"  hard={hard_pressure:g} bar "
                    f"fitted curve={100 * fitted[3]:.3f}/"
                    f"{100 * fitted[4]:.3f}/"
                    f"{100 * fitted[5]:.3f}; "
                    f"fixed={100 * fixed[3]:.3f}/"
                    f"{100 * fixed[4]:.3f}/"
                    f"{100 * fixed[5]:.3f}"
                )


def write_rows(rows):
    with OUTPUT_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows, unavailable = evaluate()
    print(
        "HARD-SEGMENT DIMER H/S FIT\n"
        f"rows={len(rows)} "
        f"curves={len({row['cas'] for row in rows})} "
        f"unavailable={len(unavailable)}"
    )
    report(rows)
    write_rows(rows)
    print(f"\nwrote {OUTPUT_PATH}")
    for item in unavailable[:20]:
        print("unavailable:", item)


if __name__ == "__main__":
    main()
