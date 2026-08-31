import csv
import json
import math
from statistics import mean, median
from types import SimpleNamespace

import numpy as np

from benchmark_perry_aw import (
    aw_dlnp_dt,
    aw_ln_p,
    normalize_vapor_pressure_coefficients,
    percentile,
)
from perry_properties import PerryPropertyLibrary
from vapor_pressure_tables import DATA_PATH, get_vapor_pressure_table_library


OUTPUT_PATH = "/tmp/local_tables_perry_benchmark.csv"


def perry_row(library, entry, temperature):
    return library._select_in_range(entry, "vapor_pressure", temperature)


def perry_ln_pressure(row, temperature):
    c1, c2, c3, c4, c5 = normalize_vapor_pressure_coefficients(row)
    return (
        c1
        + c2 / temperature
        + c3 * math.log(temperature)
        + c4 * temperature**c5
        - math.log(100000.0)
    )


def perry_slope(row, temperature):
    _, c2, c3, c4, c5 = normalize_vapor_pressure_coefficients(row)
    return (
        -c2 / temperature**2
        + c3 / temperature
        + c4 * c5 * temperature ** (c5 - 1.0)
    )


def local_endpoint_slope(table, upper, count=5):
    pairs = list(zip(table.temperatures, table.pressures_bar))
    selected = pairs[-count:] if upper else pairs[:count]
    temperatures = np.array([pair[0] for pair in selected])
    ln_pressures = np.log([pair[1] for pair in selected])
    endpoint_temperature = temperatures[-1] if upper else temperatures[0]
    coordinate = (1.0 / temperatures - 1.0 / endpoint_temperature) * endpoint_temperature
    coefficients = np.polynomial.polynomial.polyfit(coordinate, ln_pressures, 2)
    return float(-coefficients[1] / endpoint_temperature)


def relative_percent(value, reference):
    return 100.0 * (value / reference - 1.0)


def overlap_intervals(table, entry):
    intervals = []
    for row in entry.get("vapor_pressure", []):
        low = max(table.T_min, float(row["T_min_K"]))
        high = min(table.T_max, float(row["T_max_K"]))
        if low <= high:
            intervals.append((low, high))
    return intervals


def endpoint_metrics(table, library, entry, upper):
    intervals = overlap_intervals(table, entry)
    if not intervals:
        return None
    temperature = max(high for _low, high in intervals) if upper else min(
        low for low, _high in intervals
    )
    row = perry_row(library, entry, temperature)
    if row is None:
        return None
    table_ln_pressure = float(table._ln_pressure_spline(temperature))
    reference_ln_pressure = perry_ln_pressure(row, temperature)
    reference_slope = perry_slope(row, temperature)
    pchip_slope = float(table._ln_pressure_spline.derivative()(temperature))
    fitted_slope = local_endpoint_slope(table, upper)
    return {
        "temperature_K": temperature,
        "pressure_error_percent": 100.0 * math.expm1(table_ln_pressure - reference_ln_pressure),
        "pchip_slope_error_percent": relative_percent(pchip_slope, reference_slope),
        "five_point_slope_error_percent": relative_percent(fitted_slope, reference_slope),
    }


def dense_overlap(table, library, entry):
    intervals = overlap_intervals(table, entry)

    samples = []
    for low, high in intervals:
        if high == low:
            temperatures = [low]
        else:
            temperatures = np.linspace(low, high, 401)
        for temperature in temperatures:
            temperature = float(temperature)
            row = perry_row(library, entry, temperature)
            if row is None:
                continue
            table_ln_pressure = float(table._ln_pressure_spline(temperature))
            reference_ln_pressure = perry_ln_pressure(row, temperature)
            table_slope = float(table._ln_pressure_spline.derivative()(temperature))
            reference_slope = perry_slope(row, temperature)
            samples.append({
                "temperature_K": temperature,
                "pressure_error_percent": 100.0 * math.expm1(
                    table_ln_pressure - reference_ln_pressure
                ),
                "slope_error_percent": relative_percent(table_slope, reference_slope),
            })
    return samples


def format_endpoint(endpoint):
    if endpoint is None:
        return "not covered by Perry"
    return (
        f"T={endpoint['temperature_K']:.2f} K, "
        f"P={endpoint['pressure_error_percent']:+.3f}%, "
        f"slope PCHIP={endpoint['pchip_slope_error_percent']:+.3f}%, "
        f"slope local-fit={endpoint['five_point_slope_error_percent']:+.3f}%"
    )


def excluded_chemistry(table):
    return "acid" in table.name.lower() or table.key in {"H2O", "NH3"}


def completed_tail_metrics(table, library, entry):
    critical = entry.get("critical_constants") or {}
    required = (critical.get("Tc_K"), critical.get("Pc_MPa"), critical.get("omega"))
    if any(value is None for value in required):
        return None
    tc = float(critical["Tc_K"])
    if table.T_max >= tc - 1.0e-6:
        return None
    endpoint_row = perry_row(library, entry, table.T_max)
    if endpoint_row is None:
        return None

    curve = SimpleNamespace(
        tc=tc,
        pc_bar=float(critical["Pc_MPa"]) * 10.0,
        omega=float(critical["omega"]),
    )
    endpoint_temperature = table.T_max
    endpoint_value = float(table._ln_pressure_spline(endpoint_temperature))
    span = tc - endpoint_temperature
    delta_value = endpoint_value - aw_ln_p(curve, endpoint_temperature)
    slopes = {
        "pchip": float(table._ln_pressure_spline.derivative()(endpoint_temperature)),
        "local_fit": local_endpoint_slope(table, upper=True),
    }
    methods = {"raw_aw": []}
    methods.update({name: [] for name in slopes})
    predictions = {name: [] for name in methods}
    temperatures = np.linspace(endpoint_temperature, tc, 401)
    for temperature in temperatures:
        temperature = float(temperature)
        reference_row = perry_row(library, entry, temperature)
        if reference_row is None:
            continue
        reference = perry_ln_pressure(reference_row, temperature)
        raw_aw = aw_ln_p(curve, temperature)
        predictions["raw_aw"].append(raw_aw)
        methods["raw_aw"].append(100.0 * math.expm1(raw_aw - reference))
        x = (temperature - endpoint_temperature) / span
        for name, endpoint_slope in slopes.items():
            delta_slope = endpoint_slope - aw_dlnp_dt(curve, endpoint_temperature)
            coefficient = span * delta_slope + 2.0 * delta_value
            completed = raw_aw + (1.0 - x) ** 2 * (
                delta_value + coefficient * x
            )
            predictions[name].append(completed)
            methods[name].append(100.0 * math.expm1(completed - reference))

    if not methods["raw_aw"]:
        return None
    result = {"span_K": span}
    for name, errors in methods.items():
        result[name] = {
            "mard_percent": mean(abs(value) for value in errors),
            "p95_abs_percent": percentile([abs(value) for value in errors], 0.95),
            "max_abs_percent": max(abs(value) for value in errors),
            "nonmonotone": sum(
                value2 <= value1
                for value1, value2 in zip(predictions[name], predictions[name][1:])
            ),
        }
    return result


def cutoff_tail_metrics(table, library, entry, cutoff_tr):
    critical = entry.get("critical_constants") or {}
    required = (critical.get("Tc_K"), critical.get("Pc_MPa"), critical.get("omega"))
    if any(value is None for value in required):
        return None
    tc = float(critical["Tc_K"])
    endpoint_temperature = cutoff_tr * tc
    if not table.covers_temperature(endpoint_temperature):
        return None
    if perry_row(library, entry, endpoint_temperature) is None:
        return None
    curve = SimpleNamespace(
        tc=tc,
        pc_bar=float(critical["Pc_MPa"]) * 10.0,
        omega=float(critical["omega"]),
    )
    endpoint_value = float(table._ln_pressure_spline(endpoint_temperature))
    endpoint_slope = float(table._ln_pressure_spline.derivative()(endpoint_temperature))
    span = tc - endpoint_temperature
    delta_value = endpoint_value - aw_ln_p(curve, endpoint_temperature)
    delta_slope = endpoint_slope - aw_dlnp_dt(curve, endpoint_temperature)
    coefficient = span * delta_slope + 2.0 * delta_value
    errors = []
    predictions = []
    for temperature in np.linspace(endpoint_temperature, tc, 401):
        temperature = float(temperature)
        reference_row = perry_row(library, entry, temperature)
        if reference_row is None:
            continue
        x = (temperature - endpoint_temperature) / span
        baseline = aw_ln_p(curve, temperature)
        completed = baseline + (1.0 - x) ** 2 * (
            delta_value + coefficient * x
        )
        reference = perry_ln_pressure(reference_row, temperature)
        predictions.append(completed)
        errors.append(100.0 * math.expm1(completed - reference))
    if not errors:
        return None
    return {
        "mard_percent": mean(abs(value) for value in errors),
        "p95_abs_percent": percentile([abs(value) for value in errors], 0.95),
        "max_abs_percent": max(abs(value) for value in errors),
        "nonmonotone": sum(
            value2 <= value1
            for value1, value2 in zip(predictions, predictions[1:])
        ),
    }


def main():
    table_library = get_vapor_pressure_table_library()
    perry_library = PerryPropertyLibrary()
    perry_library._load()
    raw_tables = json.loads(DATA_PATH.read_text()).get("tables", {})
    reports = []
    detail_rows = []
    unmatched = []

    for table in table_library.tables.values():
        raw_entry = raw_tables.get(table.key, {})
        cas = raw_entry.get("CAS") or raw_entry.get("cas")
        entry = perry_library.get(cas, expand_identity=False) if cas else None
        if entry is None:
            entry = perry_library.get(table.key, expand_identity=False)
        if entry is None:
            entry = perry_library.get(table.name)
        if entry is None:
            unmatched.append((table.name, cas, "no Perry entry"))
            continue
        samples = dense_overlap(table, perry_library, entry)
        if not samples:
            unmatched.append((table.name, cas, "no temperature overlap"))
            continue
        pressure_errors = [sample["pressure_error_percent"] for sample in samples]
        slope_errors = [sample["slope_error_percent"] for sample in samples]
        lower = endpoint_metrics(table, perry_library, entry, upper=False)
        upper = endpoint_metrics(table, perry_library, entry, upper=True)
        report = {
            "key": table.key,
            "name": table.name,
            "source": table.source,
            "banned": excluded_chemistry(table),
            "table_Tmin_K": table.T_min,
            "table_Tmax_K": table.T_max,
            "tc_K": (entry.get("critical_constants") or {}).get("Tc_K"),
            "dense_points": len(samples),
            "pressure_mard_percent": mean(abs(value) for value in pressure_errors),
            "pressure_median_abs_percent": median(abs(value) for value in pressure_errors),
            "pressure_p95_abs_percent": percentile([abs(value) for value in pressure_errors], 0.95),
            "pressure_max_abs_percent": max(abs(value) for value in pressure_errors),
            "slope_median_abs_percent": median(abs(value) for value in slope_errors),
            "slope_p95_abs_percent": percentile([abs(value) for value in slope_errors], 0.95),
            "lower": lower,
            "upper": upper,
            "tail": completed_tail_metrics(table, perry_library, entry),
            "cutoff_tails": {
                cutoff_tr: cutoff_tail_metrics(table, perry_library, entry, cutoff_tr)
                for cutoff_tr in (0.95, 0.98, 0.99)
            },
        }
        reports.append(report)
        for sample in samples:
            detail_rows.append({
                "key": table.key,
                "name": table.name,
                "banned": report["banned"],
                **sample,
            })

    with open(OUTPUT_PATH, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=detail_rows[0].keys())
        writer.writeheader()
        writer.writerows(detail_rows)

    print("LOCAL TABLES VERSUS PERRY 2-8")
    print(f"matched={len(reports)}/{len(table_library.tables)} dense points={len(detail_rows)}")
    for name, cas, reason in unmatched:
        print(f"  unmatched: {name} [{cas or 'no CAS'}]: {reason}")
    for report in reports:
        print(f"\n{report['name']} [{report['key']}] banned={report['banned']}")
        print(
            f"  range={report['table_Tmin_K']:.2f}-{report['table_Tmax_K']:.2f} K "
            + (
                f"Tmax/Tc={report['table_Tmax_K']/float(report['tc_K']):.4f} "
                if report["tc_K"] else ""
            )
            +
            f"pressure MARD={report['pressure_mard_percent']:.3f}% "
            f"median={report['pressure_median_abs_percent']:.3f}% "
            f"p95={report['pressure_p95_abs_percent']:.3f}% "
            f"max={report['pressure_max_abs_percent']:.3f}%"
        )
        print(
            f"  dense slope median={report['slope_median_abs_percent']:.3f}% "
            f"p95={report['slope_p95_abs_percent']:.3f}%"
        )
        print(f"  lower: {format_endpoint(report['lower'])}")
        print(f"  upper: {format_endpoint(report['upper'])}")
        if report["tail"]:
            tail = report["tail"]
            print(f"  C1 tail span={tail['span_K']:.2f} K")
            for method in ("raw_aw", "pchip", "local_fit"):
                metrics = tail[method]
                print(
                    f"    {method}: MARD={metrics['mard_percent']:.3f}% "
                    f"p95={metrics['p95_abs_percent']:.3f}% "
                    f"max={metrics['max_abs_percent']:.3f}% "
                    f"nonmonotone={metrics['nonmonotone']}"
                )
        available_cutoffs = [
            (cutoff_tr, metrics)
            for cutoff_tr, metrics in report["cutoff_tails"].items()
            if metrics is not None
        ]
        if available_cutoffs:
            print("  earlier PCHIP-slope C1 handoffs:")
            for cutoff_tr, metrics in available_cutoffs:
                print(
                    f"    Tr={cutoff_tr:.2f}: MARD={metrics['mard_percent']:.3f}% "
                    f"p95={metrics['p95_abs_percent']:.3f}% "
                    f"max={metrics['max_abs_percent']:.3f}% "
                    f"nonmonotone={metrics['nonmonotone']}"
                )

    for label, selected in (
        ("all", reports),
        ("ordinary", [report for report in reports if not report["banned"]]),
    ):
        pooled = [
            row for row in detail_rows
            if label == "all" or not row["banned"]
        ]
        upper_endpoints = [report["upper"] for report in selected if report["upper"]]
        print(f"\n{label.upper()} SUMMARY")
        print(
            f"  curves={len(selected)} points={len(pooled)} "
            f"pressure MARD={mean(abs(row['pressure_error_percent']) for row in pooled):.3f}% "
            f"median={median(abs(row['pressure_error_percent']) for row in pooled):.3f}% "
            f"p95={percentile([abs(row['pressure_error_percent']) for row in pooled], 0.95):.3f}%"
        )
        print(
            f"  slope median={median(abs(row['slope_error_percent']) for row in pooled):.3f}% "
            f"p95={percentile([abs(row['slope_error_percent']) for row in pooled], 0.95):.3f}%"
        )
        if upper_endpoints:
            print(
                "  upper endpoint medians: "
                f"|P|={median(abs(row['pressure_error_percent']) for row in upper_endpoints):.3f}% "
                f"|PCHIP slope|={median(abs(row['pchip_slope_error_percent']) for row in upper_endpoints):.3f}% "
                f"|local-fit slope|={median(abs(row['five_point_slope_error_percent']) for row in upper_endpoints):.3f}%"
            )
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
