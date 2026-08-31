import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from scipy.optimize import brentq

from benchmark_direct_lower_completion import coolprop_cases
from benchmark_perry_aw import (
    load_curves,
    percentile,
    perry_ln_p,
)
from perry_properties import PerryPropertyLibrary
from property_resolution.vapor_pressure_adapter import (
    _ambrose_walton_ln_pressure,
    _ambrose_walton_omega_from_pressure,
)


OUTPUT_PATH = Path("/tmp/no_hard_aw_relaxation_benchmark.csv")
TARGET_PRESSURES_BAR = (
    0.50,
    0.25,
    0.10,
    0.05,
    0.0266645,
    0.0133322,
    0.0066661,
    0.002,
    0.001,
)
RELAXATION_SWITCHES_BAR = (0.25, 0.10, 0.05)
RELAXATION_POWERS = (0.5, 1.0)


@dataclass(frozen=True)
class NoHardCase:
    dataset: str
    cas: str
    name: str
    Tc: float
    Pc_bar: float
    omega: float
    Tb: float
    T_min: float
    T_max: float
    ln_pressure: object
    temperature_at_pressure: object


def perry_cases():
    library = PerryPropertyLibrary()
    library._load()
    cases = []
    unavailable = []
    for curve in load_curves():
        result = library.normal_boiling_point_K(curve.cas)
        if result is None:
            unavailable.append((curve.name, "missing Tb"))
            continue
        Tb = float(result.value)
        if not curve.t_min <= Tb < curve.tc:
            unavailable.append((curve.name, "Tb outside Psat range"))
            continue

        def temperature_at_pressure(
            pressure_bar,
            curve=curve,
        ):
            target = math.log(pressure_bar)
            lower_value = perry_ln_p(curve, curve.t_min) - target
            upper_value = perry_ln_p(
                curve,
                min(curve.t_max, curve.tc),
            ) - target
            if lower_value == 0.0:
                return curve.t_min
            if upper_value == 0.0:
                return min(curve.t_max, curve.tc)
            if lower_value * upper_value > 0.0:
                return None
            return brentq(
                lambda T: perry_ln_p(curve, T) - target,
                curve.t_min,
                min(curve.t_max, curve.tc),
            )

        cases.append(NoHardCase(
            dataset="Perry 2-8",
            cas=curve.cas,
            name=curve.name,
            Tc=curve.tc,
            Pc_bar=curve.pc_bar,
            omega=curve.omega,
            Tb=Tb,
            T_min=curve.t_min,
            T_max=min(curve.t_max, curve.tc),
            ln_pressure=(
                lambda T, curve=curve: perry_ln_p(curve, T)
            ),
            temperature_at_pressure=temperature_at_pressure,
        ))
    return cases, unavailable


def coolprop_reference_cases():
    source_cases, unavailable = coolprop_cases()
    cases = []
    for case in source_cases:
        Tb = case.temperature_at_pressure(1.01325)
        if Tb is None or not case.T_min <= Tb < case.Tc:
            unavailable.append((case.name, "missing Tb"))
            continue
        cases.append(NoHardCase(
            dataset="CoolProp HEOS",
            cas=case.cas,
            name=case.name,
            Tc=case.Tc,
            Pc_bar=case.Pc_bar,
            omega=case.preferred_omega,
            Tb=Tb,
            T_min=case.T_min,
            T_max=case.T_max,
            ln_pressure=case.ln_pressure,
            temperature_at_pressure=case.temperature_at_pressure,
        ))
    return cases, unavailable


def model_functions(case):
    if (
        not math.isfinite(case.omega)
        or not -0.5 <= case.omega <= 2.0
    ):
        return {}
    physical = lambda T: _ambrose_walton_ln_pressure(
        T,
        case.Tc,
        case.Pc_bar,
        case.omega,
    )
    models = {"physical_aw": physical}
    if not case.Tb < 0.7 * case.Tc:
        return models
    omega_tb = _ambrose_walton_omega_from_pressure(
        case.Tb,
        math.log(1.01325),
        case.Tc,
        case.Pc_bar,
        preferred_omega=case.omega,
    )
    omega_slope = (
        (case.omega - omega_tb)
        / (0.7 * case.Tc - case.Tb)
    )

    def linear_omega(T):
        return omega_tb + omega_slope * (T - case.Tb)

    def variable_ln_pressure(T):
        return _ambrose_walton_ln_pressure(
            T,
            case.Tc,
            case.Pc_bar,
            linear_omega(T),
        )

    models["tb_variable_omega"] = variable_ln_pressure
    models["clamp_at_tb_omega"] = lambda T: (
        _ambrose_walton_ln_pressure(
            T,
            case.Tc,
            case.Pc_bar,
            omega_tb,
        )
    )
    for switch_pressure in RELAXATION_SWITCHES_BAR:
        switch_label = f"{switch_pressure:g}".replace(".", "p")
        target = math.log(switch_pressure)
        try:
            switch_temperature = brentq(
                lambda T: variable_ln_pressure(T) - target,
                max(case.T_min, 0.15 * case.Tc),
                case.Tb,
            )
        except (ValueError, ArithmeticError):
            continue
        switch_omega = linear_omega(switch_temperature)
        models[f"freeze_omega_at_{switch_label}_bar"] = (
            lambda T, switch_temperature=switch_temperature,
            switch_omega=switch_omega: (
                _ambrose_walton_ln_pressure(
                    T,
                    case.Tc,
                    case.Pc_bar,
                    (
                        linear_omega(T)
                        if T >= switch_temperature
                        else switch_omega
                    ),
                )
            )
        )
        for power in RELAXATION_POWERS:
            power_label = f"{power:g}".replace(".", "p")

            def relaxed_omega(
                T,
                power=power,
                switch_pressure=switch_pressure,
                asymptotic_weight=0.0,
            ):
                variable_pressure = math.exp(variable_ln_pressure(T))
                pressure_ratio = min(
                    1.0,
                    max(0.0, variable_pressure / switch_pressure),
                )
                weight = (
                    asymptotic_weight
                    + (1.0 - asymptotic_weight)
                    * pressure_ratio**power
                )
                return (
                    case.omega
                    + weight * (linear_omega(T) - case.omega)
                )

            models[
                f"relax_to_physical_{switch_label}_n{power_label}"
            ] = (
                lambda T, relaxed_omega=relaxed_omega: (
                    _ambrose_walton_ln_pressure(
                        T,
                        case.Tc,
                        case.Pc_bar,
                        relaxed_omega(T),
                    )
                )
            )
            models[
                f"relax_to_half_{switch_label}_n{power_label}"
            ] = (
                lambda T, relaxed_omega=relaxed_omega: (
                    _ambrose_walton_ln_pressure(
                        T,
                        case.Tc,
                        case.Pc_bar,
                        relaxed_omega(
                            T,
                            asymptotic_weight=0.5,
                        ),
                    )
                )
            )
    return models


def evaluate_case(case):
    models = model_functions(case)
    rows = []
    for pressure in TARGET_PRESSURES_BAR:
        try:
            temperature = case.temperature_at_pressure(pressure)
        except Exception:
            temperature = None
        if (
            temperature is None
            or not case.T_min <= temperature <= case.T_max
            or temperature >= case.Tb
        ):
            continue
        reference = case.ln_pressure(temperature)
        for method, function in models.items():
            try:
                prediction = function(temperature)
                relative_error = math.exp(prediction - reference) - 1.0
                if not math.isfinite(relative_error):
                    raise ValueError("non-finite prediction")
                status = "ok"
                failure = ""
            except Exception as error:
                relative_error = None
                status = "failed"
                failure = str(error)
            rows.append({
                "dataset": case.dataset,
                "cas": case.cas,
                "name": case.name,
                "Tb_K": case.Tb,
                "Tb_over_Tc": case.Tb / case.Tc,
                "pressure_bar": pressure,
                "temperature_K": temperature,
                "reduced_temperature": temperature / case.Tc,
                "method": method,
                "status": status,
                "relative_error": relative_error,
                "failure": failure,
            })
    return rows


def method_statistics(rows, method):
    selected = [
        row for row in rows
        if row["method"] == method and row["status"] == "ok"
    ]
    if not selected:
        return None
    errors = [abs(row["relative_error"]) for row in selected]
    signed = [row["relative_error"] for row in selected]
    by_curve = {}
    for row in selected:
        by_curve.setdefault(row["cas"], []).append(
            abs(row["relative_error"])
        )
    curve_mards = [
        mean(values) for values in by_curve.values()
    ]
    return {
        "points": len(selected),
        "curves": len(by_curve),
        "median": median(errors),
        "p95": percentile(errors, 0.95),
        "maximum": max(errors),
        "curve_median": median(curve_mards),
        "curve_p95": percentile(curve_mards, 0.95),
        "signed_median": median(signed),
        "signed_mean": mean(signed),
    }


def report_group(rows, label):
    methods = sorted({row["method"] for row in rows})
    print(f"\n{label}")
    print(
        "method                                  "
        "point med/p95   curve med/p95   signed med/mean"
    )
    for method in methods:
        stats = method_statistics(rows, method)
        if stats is None:
            continue
        print(
            f"{method:40s}"
            f"{100 * stats['median']:7.3f}/{100 * stats['p95']:7.3f}% "
            f"{100 * stats['curve_median']:7.3f}/"
            f"{100 * stats['curve_p95']:7.3f}% "
            f"{100 * stats['signed_median']:+8.3f}/"
            f"{100 * stats['signed_mean']:+8.3f}%"
        )


def write_rows(rows):
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    perry, perry_unavailable = perry_cases()
    coolprop, coolprop_unavailable = coolprop_reference_cases()
    rows = []
    for index, case in enumerate((*perry, *coolprop), start=1):
        rows.extend(evaluate_case(case))
        if index % 100 == 0:
            print(f"evaluated {index}/{len(perry) + len(coolprop)}")

    print("NO-HARD AW RELAXATION")
    report_group(rows, "All references")
    for dataset in ("Perry 2-8", "CoolProp HEOS"):
        dataset_rows = [
            row for row in rows if row["dataset"] == dataset
        ]
        report_group(dataset_rows, dataset)
        for pressure in TARGET_PRESSURES_BAR:
            report_group(
                [
                    row for row in dataset_rows
                    if row["pressure_bar"] == pressure
                ],
                f"{dataset}; pressure={pressure:g} bar",
            )
    write_rows(rows)
    print(f"\nPerry unavailable={len(perry_unavailable)}")
    print(f"CoolProp unavailable={len(coolprop_unavailable)}")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
