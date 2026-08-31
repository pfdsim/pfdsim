import csv
import math
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
from statistics import mean, median

from chemicals.identifiers import search_chemical
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from benchmark_direct_lower_completion import coolprop_cases, perry_cases
from benchmark_end_to_end_lower_completion import (
    MONOCARBOXYLIC_ACID_CAS,
)
from benchmark_no_hard_aw_relaxation import (
    NoHardCase,
    model_functions,
)
from benchmark_perry_aw import percentile
from nannoolal_method import estimate_psat
from property_resolution.common import R
from property_resolution.vapor_pressure_adapter import (
    _peng_robinson_delta_z_or_ideal,
)


OUTPUT_PATH = Path("/tmp/no_hard_clapeyron_switch_benchmark.csv")
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
SWITCH_PRESSURES_BAR = (
    0.25,
    0.10,
    0.05,
    0.0266645,
    0.0133322,
)
BASE_METHODS = (
    "tb_variable_omega",
    "freeze_omega_at_0p1_bar",
    "relax_to_half_0p25_n0p5",
)
COOLPROP_CACHE_POINTS = 121
_SOURCES = ()


def cached_source(source):
    if not source.dataset.startswith("CoolProp"):
        return source
    temperatures = []
    for pressure in (*TARGET_PRESSURES_BAR, 1.01325):
        try:
            temperature = source.temperature_at_pressure(pressure)
        except Exception:
            temperature = None
        if (
            temperature is not None
            and source.T_min <= temperature <= source.T_max
        ):
            temperatures.append(float(temperature))
    if len(temperatures) < 2:
        return source
    grid = np.linspace(
        min(temperatures),
        max(temperatures),
        COOLPROP_CACHE_POINTS,
    )
    ln_pressure_interpolator = PchipInterpolator(
        grid,
        [source.ln_pressure(float(T)) for T in grid],
        extrapolate=True,
    )
    hvap_interpolator = PchipInterpolator(
        grid,
        [source.hvap_J_mol(float(T)) for T in grid],
        extrapolate=True,
    )
    return replace(
        source,
        ln_pressure=lambda T: float(ln_pressure_interpolator(T)),
        hvap_J_mol=lambda T: float(hvap_interpolator(T)),
    )


def hybrid_case(source):
    Tb = source.temperature_at_pressure(1.01325)
    if Tb is None:
        return None
    reference = NoHardCase(
        dataset=source.dataset,
        cas=source.cas,
        name=source.name,
        Tc=source.Tc,
        Pc_bar=source.Pc_bar,
        omega=source.preferred_omega,
        Tb=Tb,
        T_min=source.T_min,
        T_max=source.T_max,
        ln_pressure=source.ln_pressure,
        temperature_at_pressure=source.temperature_at_pressure,
    )
    return reference, source.hvap_J_mol


def integrate_clapeyron(
    case,
    hvap,
    base_function,
    switch_pressure,
    target_states,
    use_peng_robinson,
):
    try:
        switch_temperature = brentq(
            lambda T: (
                base_function(T) - math.log(switch_pressure)
            ),
            max(case.T_min, 0.15 * case.Tc),
            case.Tb,
        )
    except (ArithmeticError, ValueError):
        return None
    lower_targets = [
        (pressure, temperature)
        for pressure, temperature in target_states
        if temperature < switch_temperature
    ]
    predictions = {
        pressure: base_function(temperature)
        for pressure, temperature in target_states
        if temperature >= switch_temperature
    }
    if not lower_targets:
        return predictions
    minimum_temperature = min(
        temperature for _pressure, temperature in lower_targets
    )

    def derivative(T, state):
        enthalpy = hvap(T)
        if not math.isfinite(enthalpy) or enthalpy <= 0.0:
            raise ValueError("invalid Hvap")
        delta_z = (
            _peng_robinson_delta_z_or_ideal(
                T,
                math.exp(state[0]),
                case.Tc,
                case.Pc_bar,
                case.omega,
            )
            if use_peng_robinson
            else 1.0
        )
        return [enthalpy / (R * T**2 * delta_z)]

    try:
        solution = solve_ivp(
            derivative,
            (switch_temperature, minimum_temperature),
            [math.log(switch_pressure)],
            rtol=2.0e-9,
            atol=2.0e-11,
            dense_output=True,
            max_step=max(
                (switch_temperature - minimum_temperature) / 50.0,
                0.1,
            ),
        )
    except (ArithmeticError, ValueError):
        return None
    if not solution.success:
        return None
    predictions.update({
        pressure: float(solution.sol(temperature)[0])
        for pressure, temperature in lower_targets
    })
    return predictions


def nannoolal_model(case):
    try:
        metadata = search_chemical(case.cas)
        smiles = str(metadata.smiles or "").strip()
        if not smiles:
            return None, None
        estimate = estimate_psat(
            smiles,
            psat_point=(case.Tb, 101.325),
        )
        if estimate.db is None or estimate.tb_K is None:
            return None, smiles
        return (
            lambda T: math.log(estimate.psat_kPa(T) / 100.0),
            smiles,
        )
    except Exception:
        return None, None


def evaluate_case(source):
    source = cached_source(source)
    prepared = hybrid_case(source)
    if prepared is None:
        return []
    case, hvap = prepared
    models = model_functions(case)
    selected_models = {
        method: models[method]
        for method in BASE_METHODS
        if method in models
    }
    if "tb_variable_omega" not in selected_models:
        return []
    target_states = []
    for pressure in TARGET_PRESSURES_BAR:
        try:
            temperature = case.temperature_at_pressure(pressure)
        except Exception:
            temperature = None
        if (
            temperature is not None
            and case.T_min <= temperature < case.Tb
        ):
            target_states.append((pressure, temperature))
    if not target_states:
        return []

    nannoolal, smiles = nannoolal_model(case)
    monoacid = case.cas in MONOCARBOXYLIC_ACID_CAS
    predictions = {
        method: {
            pressure: function(temperature)
            for pressure, temperature in target_states
        }
        for method, function in selected_models.items()
    }
    if nannoolal is not None:
        predictions["nannoolal_tb_anchored"] = {
            pressure: nannoolal(temperature)
            for pressure, temperature in target_states
        }
    for base_method, function in selected_models.items():
        for switch_pressure in SWITCH_PRESSURES_BAR:
            switch_label = f"{switch_pressure:g}".replace(".", "p")
            for use_pr, delta_z_label in (
                (True, "pr"),
                (False, "ideal"),
            ):
                method = (
                    f"{base_method}__{delta_z_label}_clapeyron_"
                    f"at_{switch_label}_bar"
                )
                result = integrate_clapeyron(
                    case,
                    hvap,
                    function,
                    switch_pressure,
                    target_states,
                    use_pr,
                )
                if result is not None:
                    predictions[method] = result

    rows = []
    for method, values in predictions.items():
        for pressure, temperature in target_states:
            if pressure not in values:
                continue
            reference = case.ln_pressure(temperature)
            relative_error = math.exp(values[pressure] - reference) - 1.0
            rows.append({
                "dataset": case.dataset,
                "cas": case.cas,
                "name": case.name,
                "is_monocarboxylic_acid": monoacid,
                "pressure_bar": pressure,
                "temperature_K": temperature,
                "reduced_temperature": temperature / case.Tc,
                "method": method,
                "relative_error": relative_error,
            })
    return rows


def evaluate_source(index):
    return evaluate_case(_SOURCES[index])


def method_statistics(rows, method):
    selected = [row for row in rows if row["method"] == method]
    if not selected:
        return None
    errors = [abs(row["relative_error"]) for row in selected]
    by_curve = {}
    for row in selected:
        by_curve.setdefault(row["cas"], []).append(
            abs(row["relative_error"])
        )
    curve_mards = [mean(values) for values in by_curve.values()]
    return {
        "curves": len(by_curve),
        "points": len(selected),
        "median": median(errors),
        "p95": percentile(errors, 0.95),
        "curve_median": median(curve_mards),
        "curve_p95": percentile(curve_mards, 0.95),
        "maximum": max(errors),
    }


def report_group(rows, label, limit=40):
    methods = sorted({row["method"] for row in rows})
    summaries = []
    for method in methods:
        stats = method_statistics(rows, method)
        if stats is not None:
            summaries.append((stats["curve_median"], method, stats))
    print(f"\n{label}")
    print(
        "method                                                    "
        "point med/p95   curve med/p95   curves"
    )
    for _score, method, stats in sorted(summaries)[:limit]:
        print(
            f"{method:58s}"
            f"{100 * stats['median']:7.3f}/{100 * stats['p95']:7.3f}% "
            f"{100 * stats['curve_median']:7.3f}/"
            f"{100 * stats['curve_p95']:7.3f}% "
            f"{stats['curves']:6d}"
        )


def write_rows(rows):
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    global _SOURCES
    perry, perry_unavailable = perry_cases()
    coolprop, coolprop_unavailable = coolprop_cases()
    rows = []
    sources = [*perry, *coolprop]
    _SOURCES = tuple(sources)
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
            evaluate_source,
            range(len(sources)),
            chunksize=1,
        )
        for completed, case_rows in enumerate(results, start=1):
            rows.extend(case_rows)
            if completed % 50 == 0:
                print(
                    f"evaluated {completed}/{len(sources)}",
                    flush=True,
                )
    rows.sort(key=lambda row: (
        row["dataset"],
        row["name"],
        row["method"],
        row["pressure_bar"],
    ))

    print("NO-HARD CLAPEYRON SWITCH")
    ordinary = [
        row for row in rows
        if not row["is_monocarboxylic_acid"]
    ]
    report_group(ordinary, "All ordinary references")
    for dataset in (
        "Perry 2-8 Psat / 2-69 Hvap",
        "CoolProp HEOS",
    ):
        dataset_rows = [
            row for row in ordinary if row["dataset"] == dataset
        ]
        report_group(dataset_rows, dataset)
        for pressure in TARGET_PRESSURES_BAR:
            report_group(
                [
                    row for row in dataset_rows
                    if row["pressure_bar"] == pressure
                ],
                f"{dataset}; target={pressure:g} bar",
                limit=12,
            )
    write_rows(rows)
    print(f"\nrows={len(rows)}")
    print(f"Perry unavailable={len(perry_unavailable)}")
    print(f"CoolProp unavailable={len(coolprop_unavailable)}")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
