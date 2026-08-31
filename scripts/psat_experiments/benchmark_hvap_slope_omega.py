import csv
import math
from pathlib import Path
from statistics import mean, median

from scipy.optimize import root

from benchmark_lower_clapeyron import (
    correlation_hvap_kJ_mol,
    hvap_row_at_tb,
)
from benchmark_lower_hybrid import is_banned
from benchmark_lower_psat import PRESSURE_LEVELS_BAR, invert_ln_pressure
from benchmark_perry_aw import load_curves, percentile, perry_ln_p
from benchmark_switch_cutoff import (
    MMHG_PER_BAR,
    linear_omega,
    linear_omega_ln_pressure,
)
from benchmark_table210_nannoolal_aw import tb_valid_matches
from benchmark_tb_constrained_aw import aw_ln_p_with_omega, tb_constrained_omega
from perry_properties import PerryPropertyLibrary
from property_resolution.common import R


TARGET_PRESSURE_BAR = 0.2
ACTIVATION_PRESSURES_MMHG = (760.0, 400.0, 300.0, 200.0)
CORRECTION_ORDERS = (1, 2)
OUTPUT_PATH = Path("/tmp/hvap_slope_omega_benchmark.csv")
HVAP_OUTPUT_PATH = Path("/tmp/hvap_at_0p2bar.csv")


def numerical_slope(function, temperature, tc):
    step = max(1.0e-5 * tc, 1.0e-4)
    return (function(temperature + step) - function(temperature - step)) / (2.0 * step)


def build_model(library, curve, tb):
    if tb >= 0.7 * curve.tc:
        return None
    tb_omega, _roots = tb_constrained_omega(curve, tb, 1.01325)
    if tb_omega is None:
        return None
    row = hvap_row_at_tb(library, curve, tb)
    if row is None:
        return None
    target_temperature = invert_ln_pressure(
        lambda temperature: perry_ln_p(curve, temperature),
        math.log(TARGET_PRESSURE_BAR),
        curve.t_min,
        tb,
    )
    if target_temperature is None or target_temperature < float(row["T_min_K"]):
        return None
    target_hvap = correlation_hvap_kJ_mol(
        library,
        row,
        curve,
        target_temperature,
    )
    if target_hvap is None:
        return None
    return {
        "tb": tb,
        "tb_omega": tb_omega,
        "row": row,
        "reference_target_temperature": target_temperature,
        "reference_target_hvap": target_hvap,
    }


def corrected_ln_pressure(
    curve,
    model,
    activation_temperature,
    coefficient,
    order,
    temperature,
):
    base_omega = linear_omega(
        curve,
        model["tb"],
        model["tb_omega"],
        temperature,
    )
    if temperature >= activation_temperature:
        omega = base_omega
    else:
        coordinate = (temperature - activation_temperature) / curve.tc
        omega = base_omega + coefficient * coordinate**order
    return aw_ln_p_with_omega(curve, temperature, omega)


def solve_correction(library, curve, model, activation_mmHg, order):
    activation_pressure = activation_mmHg / MMHG_PER_BAR
    activation_temperature = invert_ln_pressure(
        lambda temperature: linear_omega_ln_pressure(
            curve,
            model["tb"],
            model["tb_omega"],
            temperature,
        ),
        math.log(activation_pressure),
        max(1.0, 0.15 * curve.tc),
        model["tb"],
    )
    base_target_temperature = invert_ln_pressure(
        lambda temperature: linear_omega_ln_pressure(
            curve,
            model["tb"],
            model["tb_omega"],
            temperature,
        ),
        math.log(TARGET_PRESSURE_BAR),
        max(1.0, 0.15 * curve.tc),
        activation_temperature,
    )
    if activation_temperature is None or base_target_temperature is None:
        return None

    def residual(values):
        coefficient, temperature = values
        if not 0.15 * curve.tc < temperature < activation_temperature:
            return [10.0, 10.0]
        function = lambda value: corrected_ln_pressure(
            curve,
            model,
            activation_temperature,
            coefficient,
            order,
            value,
        )
        hvap = correlation_hvap_kJ_mol(
            library,
            model["row"],
            curve,
            temperature,
        )
        if hvap is None:
            return [10.0, 10.0]
        target_slope = hvap * 1000.0 / (R * temperature**2)
        return [
            function(temperature) - math.log(TARGET_PRESSURE_BAR),
            numerical_slope(function, temperature, curve.tc) / target_slope - 1.0,
        ]

    base_function = lambda value: linear_omega_ln_pressure(
        curve,
        model["tb"],
        model["tb_omega"],
        value,
    )
    base_slope = numerical_slope(base_function, base_target_temperature, curve.tc)
    hvap = correlation_hvap_kJ_mol(
        library,
        model["row"],
        curve,
        base_target_temperature,
    )
    target_slope = hvap * 1000.0 / (R * base_target_temperature**2)
    omega_step = 1.0e-5
    partial_omega = (
        aw_ln_p_with_omega(
            curve,
            base_target_temperature,
            linear_omega(
                curve,
                model["tb"],
                model["tb_omega"],
                base_target_temperature,
            )
            + omega_step,
        )
        - aw_ln_p_with_omega(
            curve,
            base_target_temperature,
            linear_omega(
                curve,
                model["tb"],
                model["tb_omega"],
                base_target_temperature,
            )
            - omega_step,
        )
    ) / (2.0 * omega_step)
    coordinate = (base_target_temperature - activation_temperature) / curve.tc
    derivative_factor = order * coordinate ** (order - 1) / curve.tc
    denominator = partial_omega * derivative_factor
    initial_coefficient = (
        0.0
        if abs(denominator) < 1.0e-12
        else (target_slope - base_slope) / denominator
    )
    solution = root(
        residual,
        [initial_coefficient, base_target_temperature],
        method="hybr",
    )
    if not solution.success:
        return None
    coefficient, target_temperature = map(float, solution.x)
    final_residual = residual((coefficient, target_temperature))
    if max(abs(value) for value in final_residual) > 1.0e-7:
        return None
    target_hvap = correlation_hvap_kJ_mol(
        library,
        model["row"],
        curve,
        target_temperature,
    )
    return {
        "activation_pressure_mmHg": activation_mmHg,
        "activation_temperature_K": activation_temperature,
        "order": order,
        "coefficient": coefficient,
        "target_temperature_K": target_temperature,
        "target_hvap_kJ_mol": target_hvap,
    }


def evaluate_reference_point(curve, model, correction, temperature, pressure_bar):
    prediction = corrected_ln_pressure(
        curve,
        model,
        correction["activation_temperature_K"],
        correction["coefficient"],
        correction["order"],
        temperature,
    )
    return math.exp(prediction) / pressure_bar - 1.0


def prepare_models(library):
    models = {}
    hvap_rows = []
    for curve in load_curves():
        if is_banned(curve):
            continue
        tb_result = library.normal_boiling_point_K(curve.cas)
        if tb_result is None:
            continue
        model = build_model(library, curve, float(tb_result.value))
        if model is None:
            continue
        models[curve.cas] = (curve, model)
        hvap_rows.append(
            {
                "cas": curve.cas,
                "name": curve.name,
                "Tb_K": model["tb"],
                "Tc_K": curve.tc,
                "Pc_bar": curve.pc_bar,
                "omega": curve.omega,
                "temperature_at_0p2bar_K": model["reference_target_temperature"],
                "Hvap_at_0p2bar_kJ_mol": model["reference_target_hvap"],
            }
        )
    return models, hvap_rows


def correlation_points(curve, model):
    points = []
    minimum_pressure = math.exp(perry_ln_p(curve, curve.t_min))
    for pressure_bar in PRESSURE_LEVELS_BAR:
        if pressure_bar < minimum_pressure:
            continue
        temperature = invert_ln_pressure(
            lambda value: perry_ln_p(curve, value),
            math.log(pressure_bar),
            curve.t_min,
            model["tb"],
        )
        if temperature is not None:
            points.append((temperature, pressure_bar))
    return points


def benchmark_records(library, records, reference):
    rows = []
    unavailable = 0
    for curve, model, points in records:
        base_errors = [
            abs(
                math.exp(
                    linear_omega_ln_pressure(
                        curve,
                        model["tb"],
                        model["tb_omega"],
                        temperature,
                    )
                )
                / pressure_bar
                - 1.0
            )
            for temperature, pressure_bar in points
        ]
        for activation_mmHg in ACTIVATION_PRESSURES_MMHG:
            for order in CORRECTION_ORDERS:
                correction = solve_correction(
                    library,
                    curve,
                    model,
                    activation_mmHg,
                    order,
                )
                if correction is None:
                    unavailable += 1
                    continue
                errors = [
                    abs(
                        evaluate_reference_point(
                            curve,
                            model,
                            correction,
                            temperature,
                            pressure_bar,
                        )
                    )
                    for temperature, pressure_bar in points
                ]
                predicted = [
                    corrected_ln_pressure(
                        curve,
                        model,
                        correction["activation_temperature_K"],
                        correction["coefficient"],
                        order,
                        temperature,
                    )
                    for temperature, _pressure in sorted(points)
                ]
                rows.append(
                    {
                        "reference": reference,
                        "cas": curve.cas,
                        "name": curve.name,
                        "activation_pressure_mmHg": activation_mmHg,
                        "order": order,
                        "coefficient": correction["coefficient"],
                        "activation_temperature_K": correction[
                            "activation_temperature_K"
                        ],
                        "target_temperature_K": correction["target_temperature_K"],
                        "target_hvap_kJ_mol": correction["target_hvap_kJ_mol"],
                        "baseline_mard": mean(base_errors),
                        "corrected_mard": mean(errors),
                        "baseline_max_error": max(base_errors),
                        "corrected_max_error": max(errors),
                        "nonmonotone": any(
                            right <= left
                            for left, right in zip(predicted, predicted[1:])
                        ),
                    }
                )
    return rows, unavailable


def report(rows, label):
    print(f"\n{label}")
    baseline = [row["baseline_mard"] for row in rows if row["order"] == 1 and row["activation_pressure_mmHg"] == 760.0]
    print(
        f"  baseline linear-omega AW: n={len(baseline)} "
        f"mean={100*mean(baseline):.2f}% median={100*median(baseline):.2f}% "
        f"p95={100*percentile(baseline, .95):.2f}%"
    )
    for activation_mmHg in ACTIVATION_PRESSURES_MMHG:
        for order in CORRECTION_ORDERS:
            subset = [
                row
                for row in rows
                if row["activation_pressure_mmHg"] == activation_mmHg
                and row["order"] == order
            ]
            values = [row["corrected_mard"] for row in subset]
            print(
                f"  activate={activation_mmHg:.0f} mmHg order={order}: "
                f"n={len(subset)} mean={100*mean(values):.2f}% "
                f"median={100*median(values):.2f}% "
                f"p95={100*percentile(values, .95):.2f}% "
                f"wins={sum(row['corrected_mard'] < row['baseline_mard'] for row in subset)} "
                f"nonmonotone={sum(row['nonmonotone'] for row in subset)}"
            )


def main():
    library = PerryPropertyLibrary()
    library._load()
    models, hvap_rows = prepare_models(library)
    correlation_records = [
        (curve, model, correlation_points(curve, model))
        for curve, model in models.values()
    ]
    table_records = []
    for curve, pairs, _trusted_tb in tb_valid_matches():
        item = models.get(curve.cas)
        if item is None:
            continue
        model_curve, model = item
        table_records.append((model_curve, model, pairs[:-1]))

    correlation_results, correlation_unavailable = benchmark_records(
        library,
        correlation_records,
        "Perry 2-8 correlation",
    )
    table_results, table_unavailable = benchmark_records(
        library,
        table_records,
        "Perry 2-10 table",
    )
    print("Hvap-SLOPE-MATCHED OMEGA CORRECTION AT 0.2 bar; deltaZ=1")
    print(f"Hvap(0.2 bar) components={len(hvap_rows)}")
    hvap_values = [row["Hvap_at_0p2bar_kJ_mol"] for row in hvap_rows]
    print(
        f"Hvap(0.2 bar): median={median(hvap_values):.3f} kJ/mol "
        f"p05={percentile(hvap_values, .05):.3f} "
        f"p95={percentile(hvap_values, .95):.3f}"
    )
    report(table_results, "Perry 2-10 independent table points")
    report(correlation_results, "Perry 2-8 continuous correlations")
    print(
        f"\nFailed correction solves: table={table_unavailable}, "
        f"correlations={correlation_unavailable}"
    )

    with HVAP_OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=hvap_rows[0].keys())
        writer.writeheader()
        writer.writerows(hvap_rows)
    rows = table_results + correlation_results
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {HVAP_OUTPUT_PATH} and {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
