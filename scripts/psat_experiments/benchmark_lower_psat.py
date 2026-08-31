import csv
import math
from pathlib import Path
from statistics import mean, median

from chemicals.identifiers import search_chemical

from benchmark_perry_aw import aw_ln_p, load_curves, percentile, perry_ln_p
from benchmark_table210_nannoolal_aw import tb_valid_matches
from nannoolal_method import estimate_psat
from perry_properties import PerryPropertyLibrary


PRESSURE_LEVELS_BAR = (
    0.5332895,
    0.2666447,
    0.1333224,
    0.0799934,
    0.0533289,
    0.0266645,
    0.0133322,
    0.0066661,
    0.0013332,
)
OUTPUT_PATH = Path("/tmp/lower_psat_benchmark.csv")


def invert_ln_pressure(function, target, lower, upper):
    if not function(lower) <= target <= function(upper):
        return None
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if function(midpoint) < target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def nannoolal_estimate(curve, temperature, pressure_bar):
    metadata = search_chemical(curve.cas)
    estimate = estimate_psat(
        metadata.smiles,
        psat_point=(temperature, pressure_bar * 100.0),
    )
    if estimate.db is None or estimate.tb_K is None:
        return None
    return estimate


def is_ordinary(curve, estimate):
    name = curve.name.lower()
    return (
        not estimate.alcohol_correction
        and "acid" not in name
        and "amine" not in name
        and "glycol" not in name
    )


def evaluate_point(curve, estimate, temperature, pressure_bar, reference):
    aw_pressure = math.exp(aw_ln_p(curve, temperature))
    nannoolal_pressure = estimate.psat_kPa(temperature) / 100.0
    target = math.log(pressure_bar)
    aw_temperature = invert_ln_pressure(
        lambda value: aw_ln_p(curve, value),
        target,
        max(1.0, 0.15 * curve.tc),
        curve.tc,
    )
    nannoolal_temperature = estimate.temperature_K(pressure_bar * 100.0)
    return {
        "reference": reference,
        "cas": curve.cas,
        "name": curve.name,
        "pressure_bar": pressure_bar,
        "pressure_mmHg": pressure_bar * 750.061683,
        "temperature_K": temperature,
        "reduced_temperature": temperature / curve.tc,
        "ordinary": is_ordinary(curve, estimate),
        "alcohol_correction": estimate.alcohol_correction,
        "aw_relative_error": aw_pressure / pressure_bar - 1.0,
        "nannoolal_relative_error": nannoolal_pressure / pressure_bar - 1.0,
        "aw_temperature_error_K": (
            None if aw_temperature is None else aw_temperature - temperature
        ),
        "nannoolal_temperature_error_K": (
            None
            if nannoolal_temperature is None
            else nannoolal_temperature - temperature
        ),
    }


def table_rows():
    rows = []
    unavailable = []
    for curve, pairs, _trusted_tb in tb_valid_matches():
        try:
            estimate = nannoolal_estimate(curve, pairs[-1][0], pairs[-1][1])
        except Exception as error:
            unavailable.append((curve.name, str(error)))
            continue
        if estimate is None:
            unavailable.append((curve.name, "not estimable"))
            continue
        for temperature, pressure_bar in pairs[:-1]:
            rows.append(
                evaluate_point(
                    curve,
                    estimate,
                    temperature,
                    pressure_bar,
                    "Perry 2-10 table",
                )
            )
    return rows, unavailable


def correlation_rows():
    library = PerryPropertyLibrary()
    rows = []
    unavailable = []
    for curve in load_curves():
        tb_result = library.normal_boiling_point_K(curve.cas)
        if tb_result is None:
            continue
        tb = float(tb_result.value)
        try:
            estimate = nannoolal_estimate(curve, tb, 1.01325)
        except Exception as error:
            unavailable.append((curve.name, str(error)))
            continue
        if estimate is None:
            unavailable.append((curve.name, "not estimable"))
            continue
        minimum_pressure = math.exp(perry_ln_p(curve, curve.t_min))
        for pressure_bar in PRESSURE_LEVELS_BAR:
            if pressure_bar < minimum_pressure:
                continue
            temperature = invert_ln_pressure(
                lambda value: perry_ln_p(curve, value),
                math.log(pressure_bar),
                curve.t_min,
                tb,
            )
            if temperature is None:
                continue
            rows.append(
                evaluate_point(
                    curve,
                    estimate,
                    temperature,
                    pressure_bar,
                    "Perry 2-8 correlation",
                )
            )
    return rows, unavailable


def summarize_method(rows, prefix):
    pressure_errors = [abs(row[f"{prefix}_relative_error"]) for row in rows]
    temperature_errors = [
        abs(row[f"{prefix}_temperature_error_K"])
        for row in rows
        if row[f"{prefix}_temperature_error_K"] is not None
    ]
    return (
        f"MARD={100*mean(pressure_errors):.2f}% "
        f"median={100*median(pressure_errors):.2f}% "
        f"p95={100*percentile(pressure_errors, .95):.2f}% "
        f"|dT| median={median(temperature_errors):.2f} K "
        f"p95={percentile(temperature_errors, .95):.2f} K"
    )


def report(rows, label):
    print(f"\n{label}: curves={len({row['cas'] for row in rows})} points={len(rows)}")
    for subset_label, subset in (
        ("all", rows),
        ("ordinary", [row for row in rows if row["ordinary"]]),
    ):
        print(f"  {subset_label}: n={len(subset)}")
        print(f"    AW:        {summarize_method(subset, 'aw')}")
        print(f"    Nannoolal: {summarize_method(subset, 'nannoolal')}")

    print("  by pressure:")
    for pressure_bar in sorted({row["pressure_bar"] for row in rows}, reverse=True):
        subset = [row for row in rows if row["pressure_bar"] == pressure_bar]
        aw_errors = [abs(row["aw_relative_error"]) for row in subset]
        nannoolal_errors = [abs(row["nannoolal_relative_error"]) for row in subset]
        print(
            f"    {pressure_bar * 750.061683:4.0f} mmHg n={len(subset):3d}: "
            f"AW median={100*median(aw_errors):6.2f}% "
            f"p95={100*percentile(aw_errors, .95):7.2f}%; "
            f"Nannoolal median={100*median(nannoolal_errors):6.2f}% "
            f"p95={100*percentile(nannoolal_errors, .95):7.2f}%"
        )

    lowest_pressure = min(row["pressure_bar"] for row in rows)
    lowest = [row for row in rows if row["pressure_bar"] == lowest_pressure]
    print(f"  worst at {lowest_pressure * 750.061683:.0f} mmHg:")
    for row in sorted(
        lowest,
        key=lambda item: max(
            abs(item["aw_relative_error"]),
            abs(item["nannoolal_relative_error"]),
        ),
        reverse=True,
    )[:15]:
        print(
            f"    {row['name']} [{row['cas']}]: Tr={row['reduced_temperature']:.3f}, "
            f"AW={100*row['aw_relative_error']:+.1f}%, "
            f"Nannoolal={100*row['nannoolal_relative_error']:+.1f}%"
        )


def main():
    table, table_unavailable = table_rows()
    correlations, correlation_unavailable = correlation_rows()
    print("LOWER-RANGE Psat BENCHMARK")
    report(table, "Perry 2-10 independent table points")
    report(correlations, "Perry 2-8 continuous correlations")
    print(
        f"\nNannoolal unavailable: table={len(table_unavailable)}, "
        f"correlations={len(correlation_unavailable)}"
    )

    rows = table + correlations
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
