import csv
import math
from pathlib import Path
from statistics import mean, median

from chemicals.identifiers import search_chemical

from benchmark_direct_lower_completion import (
    coolprop_cases,
    dynamic_omega_model,
    perry_cases,
)
from benchmark_perry_aw import percentile
from nannoolal_method import estimate_psat


ANCHOR_PRESSURES_BAR = (1.01325, 0.75, 0.50, 0.25, 0.10, 0.05)
TARGET_PRESSURES_BAR = (0.25, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
OUTPUT_PATH = Path("/tmp/direct_lower_nannoolal_benchmark.csv")


def numerical_derivative(function, temperature, scale):
    step = max(1.0e-5 * scale, 1.0e-4)
    return (
        function(temperature + step)
        - function(temperature - step)
    ) / (2.0 * step)


def nannoolal_ln_pressure(estimate, temperature):
    pressure_kPa = estimate.psat_kPa(temperature)
    if (
        pressure_kPa is None
        or not math.isfinite(pressure_kPa)
        or pressure_kPa <= 0.0
    ):
        raise ValueError("Nannoolal pressure unavailable")
    return math.log(pressure_kPa / 100.0)


def evaluate_case(case, smiles):
    rows = []
    for anchor_pressure in ANCHOR_PRESSURES_BAR:
        try:
            anchor_temperature = case.temperature_at_pressure(
                anchor_pressure
            )
        except (ArithmeticError, ValueError):
            anchor_temperature = None
        if (
            anchor_temperature is None
            or not case.T_min <= anchor_temperature <= case.T_max
            or anchor_temperature > 0.8 * case.Tc
        ):
            continue
        anchor_ln_pressure = math.log(anchor_pressure)
        try:
            anchor_round_trip = case.ln_pressure(anchor_temperature)
            if (
                abs(
                    math.exp(anchor_round_trip) / anchor_pressure - 1.0
                )
                > 1.0e-6
            ):
                continue
            dynamic, _endpoint_omega, _omega_slope = dynamic_omega_model(
                case,
                anchor_temperature,
                anchor_ln_pressure,
            )
            nannoolal_estimate = estimate_psat(
                smiles,
                psat_point=(
                    anchor_temperature,
                    anchor_pressure * 100.0,
                ),
            )
            if (
                nannoolal_estimate.db is None
                or nannoolal_estimate.tb_K is None
            ):
                continue
            nannoolal = lambda temperature: nannoolal_ln_pressure(
                nannoolal_estimate,
                temperature,
            )
            hard_slope = case.dln_pressure_dT(anchor_temperature)
            nannoolal_slope = numerical_derivative(
                nannoolal,
                anchor_temperature,
                case.Tc,
            )
            slope_scale = hard_slope / nannoolal_slope
            slope_relative_error = (
                nannoolal_slope / hard_slope - 1.0
            )
        except (ArithmeticError, TypeError, ValueError):
            continue

        for target_pressure in TARGET_PRESSURES_BAR:
            if target_pressure >= anchor_pressure:
                continue
            try:
                target_temperature = case.temperature_at_pressure(
                    target_pressure
                )
            except (ArithmeticError, ValueError):
                target_temperature = None
            if (
                target_temperature is None
                or not case.T_min <= target_temperature < anchor_temperature
                or target_temperature > 0.8 * case.Tc
            ):
                continue
            try:
                reference_ln_pressure = case.ln_pressure(
                    target_temperature
                )
                if (
                    abs(
                        math.exp(reference_ln_pressure)
                        / target_pressure
                        - 1.0
                    )
                    > 1.0e-6
                ):
                    continue
                dynamic_prediction = dynamic(target_temperature)
                nannoolal_prediction = nannoolal(target_temperature)
                slope_scaled_prediction = (
                    anchor_ln_pressure
                    + slope_scale
                    * (
                        nannoolal_prediction - anchor_ln_pressure
                    )
                )
                target_hard_slope = case.dln_pressure_dT(
                    target_temperature
                )
                target_nannoolal_slope = numerical_derivative(
                    nannoolal,
                    target_temperature,
                    case.Tc,
                )
                target_slope_relative_error = (
                    target_nannoolal_slope / target_hard_slope - 1.0
                )
            except (ArithmeticError, TypeError, ValueError):
                continue

            def relative_error(predicted):
                return math.exp(predicted - reference_ln_pressure) - 1.0

            rows.append({
                "dataset": case.dataset,
                "cas": case.cas,
                "name": case.name,
                "is_acid": "acid" in case.name.lower(),
                "anchor_pressure_bar": anchor_pressure,
                "anchor_temperature_K": anchor_temperature,
                "anchor_reduced_temperature": anchor_temperature / case.Tc,
                "target_pressure_bar": target_pressure,
                "target_temperature_K": target_temperature,
                "target_reduced_temperature": target_temperature / case.Tc,
                "nannoolal_tb_source": nannoolal_estimate.tb_source,
                "nannoolal_warnings": "; ".join(
                    nannoolal_estimate.warnings
                ),
                "nannoolal_slope_relative_error": slope_relative_error,
                "nannoolal_target_slope_relative_error": (
                    target_slope_relative_error
                ),
                "dynamic_omega_relative_error": relative_error(
                    dynamic_prediction
                ),
                "nannoolal_endpoint_relative_error": relative_error(
                    nannoolal_prediction
                ),
                "nannoolal_slope_scaled_relative_error": relative_error(
                    slope_scaled_prediction
                ),
            })
    return rows


def summarize_method(rows, method):
    values = [
        abs(float(row[f"{method}_relative_error"]))
        for row in rows
        if row.get(f"{method}_relative_error") is not None
    ]
    if not values:
        return "n=0"
    return (
        f"n={len(values)} median={100 * median(values):.3f}% "
        f"p95={100 * percentile(values, 0.95):.3f}% "
        f"mean={100 * mean(values):.3f}% "
        f"max={100 * max(values):.3f}%"
    )


def report_group(rows, label):
    print(f"\n{label}: curves={len({row['cas'] for row in rows})} points={len(rows)}")
    for method in (
        "dynamic_omega",
        "nannoolal_endpoint",
        "nannoolal_slope_scaled",
    ):
        print(f"  {method}: {summarize_method(rows, method)}")
    slope_errors_by_case = {}
    for row in rows:
        key = (
            row["cas"],
            row["anchor_pressure_bar"],
        )
        slope_errors_by_case[key] = abs(
            float(row["nannoolal_slope_relative_error"])
        )
    slope_errors = list(slope_errors_by_case.values())
    if slope_errors:
        print(
            f"  Nannoolal handoff slope mismatch: "
            f"median={100 * median(slope_errors):.3f}% "
            f"p95={100 * percentile(slope_errors, 0.95):.3f}% "
            f"max={100 * max(slope_errors):.3f}%"
        )
    for method in (
        "nannoolal_endpoint",
        "nannoolal_slope_scaled",
    ):
        wins = sum(
            abs(float(row[f"{method}_relative_error"]))
            < abs(float(row["dynamic_omega_relative_error"]))
            for row in rows
        )
        print(
            f"  {method} wins versus dynamic omega: "
            f"{wins}/{len(rows)}"
        )


def report(rows):
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        report_group(dataset_rows, dataset)
        for anchor_pressure in ANCHOR_PRESSURES_BAR:
            subset = [
                row for row in dataset_rows
                if row["anchor_pressure_bar"] == anchor_pressure
            ]
            if subset:
                report_group(
                    subset,
                    f"{dataset}; hard endpoint={anchor_pressure:g} bar",
                )
        if dataset.startswith("Perry"):
            acid_rows = [row for row in dataset_rows if row["is_acid"]]
            nonacid_rows = [
                row for row in dataset_rows if not row["is_acid"]
            ]
            if acid_rows:
                report_group(acid_rows, f"{dataset}; acids")
            if nonacid_rows:
                report_group(nonacid_rows, f"{dataset}; non-acids")


def main():
    perry, _perry_unavailable = perry_cases()
    coolprop, _coolprop_unavailable = coolprop_cases()
    cases = perry + coolprop
    rows = []
    unavailable = []
    for index, case in enumerate(cases, 1):
        try:
            smiles = search_chemical(case.cas).smiles
        except Exception as error:
            unavailable.append((
                case.dataset,
                case.name,
                f"identity: {error}",
            ))
            continue
        if not smiles:
            unavailable.append((case.dataset, case.name, "missing SMILES"))
            continue
        case_rows = evaluate_case(case, smiles)
        rows.extend(case_rows)
        if not case_rows:
            unavailable.append((
                case.dataset,
                case.name,
                "Nannoolal unavailable",
            ))
        if index % 50 == 0:
            print(f"evaluated {index}/{len(cases)} cases")
    print("DIRECT LOWER COMPLETION: DYNAMIC OMEGA VS NANNOOLAL")
    print(f"cases={len(cases)} rows={len(rows)} unavailable={len(unavailable)}")
    report(rows)
    if rows:
        keys = sorted({key for row in rows for key in row})
        with OUTPUT_PATH.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
