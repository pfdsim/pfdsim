import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from antoine_properties import get_antoine_table
from benchmark_perry_aw import (
    Curve,
    aw_dlnp_dt,
    aw_ln_p,
    completed_ln_p,
    normalize_vapor_pressure_coefficients,
    percentile,
    perry_dlnp_dt,
    perry_ln_p,
)
from perry_properties import PerryPropertyLibrary


def antoine_ln_p(entry, temperature):
    temperature_c = temperature - 273.15
    return math.log(10.0) * (entry.A - entry.B / (entry.C + temperature_c))


def antoine_dlnp_dt(entry, temperature):
    temperature_c = temperature - 273.15
    return math.log(10.0) * entry.B / (entry.C + temperature_c) ** 2


def c1_antoine_aw_ln_p(curve, entry, temperature):
    t0 = entry.T_max
    span = curve.tc - t0
    x = (temperature - t0) / span
    delta_value = antoine_ln_p(entry, t0) - aw_ln_p(curve, t0)
    delta_slope = antoine_dlnp_dt(entry, t0) - aw_dlnp_dt(curve, t0)
    coefficient = span * delta_slope + 2.0 * delta_value
    return aw_ln_p(curve, temperature) + (1.0 - x) ** 2 * (delta_value + coefficient * x)


def endpoint_bin(reduced_temperature):
    if reduced_temperature < 0.60:
        return "<0.60"
    if reduced_temperature < 0.65:
        return "0.60-0.65"
    if reduced_temperature < 0.70:
        return "0.65-0.70"
    if reduced_temperature < 0.75:
        return "0.70-0.75"
    if reduced_temperature < 0.80:
        return "0.75-0.80"
    if reduced_temperature < 0.85:
        return "0.80-0.85"
    return ">=0.85"


def load_matches():
    antoine = get_antoine_table()
    perry = PerryPropertyLibrary()
    perry._load()
    seen_rows = set()
    by_cas = defaultdict(list)
    for rows in antoine._by_name.values():
        for entry in rows:
            row_key = (
                entry.record_id, entry.name, entry.A, entry.B, entry.C,
                entry.T_min, entry.T_max,
            )
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            perry_entry = perry.get(entry.name)
            if not perry_entry:
                continue
            vapor_rows = perry_entry.get("vapor_pressure") or []
            critical = perry_entry.get("critical_constants") or {}
            if len(vapor_rows) != 1:
                continue
            if any(critical.get(key) is None for key in ("Tc_K", "Pc_MPa", "omega")):
                continue
            tc = float(critical["Tc_K"])
            vapor_row = vapor_rows[0]
            if not (
                float(vapor_row["T_min_K"]) <= entry.T_max
                and entry.T_min < entry.T_max < tc
                and float(vapor_row["T_max_K"]) >= tc - 0.01
            ):
                continue
            curve = Curve(
                cas=perry_entry["cas"],
                name=perry_entry.get("name") or perry_entry["cas"],
                tc=tc,
                pc_bar=float(critical["Pc_MPa"]) * 10.0,
                omega=float(critical["omega"]),
                t_min=float(vapor_row["T_min_K"]),
                t_max=float(vapor_row["T_max_K"]),
                coefficients=normalize_vapor_pressure_coefficients(vapor_row),
            )
            by_cas[curve.cas].append((entry, curve))

    matches = []
    for candidates in by_cas.values():
        matches.append(max(candidates, key=lambda item: item[0].T_max))
    return sorted(matches, key=lambda item: item[0].T_max / item[1].tc)


def curve_metrics(entry, curve):
    cutoff_tr = entry.T_max / curve.tc
    temperatures = [
        entry.T_max + (curve.tc - entry.T_max) * index / 400.0
        for index in range(401)
    ]
    reference = [perry_ln_p(curve, temperature) for temperature in temperatures]
    completed = [c1_antoine_aw_ln_p(curve, entry, temperature) for temperature in temperatures]
    oracle = [completed_ln_p(curve, temperature, cutoff_tr, 2) for temperature in temperatures]
    extrapolated = [antoine_ln_p(entry, temperature) for temperature in temperatures]

    completed_errors = [math.exp(predicted - truth) - 1.0 for predicted, truth in zip(completed, reference)]
    oracle_errors = [math.exp(predicted - truth) - 1.0 for predicted, truth in zip(oracle, reference)]
    extrapolated_errors = [math.exp(predicted - truth) - 1.0 for predicted, truth in zip(extrapolated, reference)]
    handoff_error = math.exp(antoine_ln_p(entry, entry.T_max) - perry_ln_p(curve, entry.T_max)) - 1.0
    handoff_slope_error = antoine_dlnp_dt(entry, entry.T_max) / perry_dlnp_dt(curve, entry.T_max) - 1.0
    derivatives = [
        (completed[index + 1] - completed[index - 1]) / (temperatures[index + 1] - temperatures[index - 1])
        for index in range(1, len(temperatures) - 1)
    ]
    return {
        "cas": curve.cas,
        "name": curve.name,
        "antoine_name": entry.name,
        "antoine_Tmin_K": entry.T_min,
        "antoine_Tmax_K": entry.T_max,
        "Tc_K": curve.tc,
        "endpoint_Tr": cutoff_tr,
        "endpoint_bin": endpoint_bin(cutoff_tr),
        "handoff_relative_error": handoff_error,
        "handoff_slope_relative_error": handoff_slope_error,
        "c1_mard": mean(abs(error) for error in completed_errors),
        "c1_max_abs_relative_error": max(abs(error) for error in completed_errors),
        "oracle_mard": mean(abs(error) for error in oracle_errors),
        "antoine_extrapolation_mard": mean(abs(error) for error in extrapolated_errors),
        "antoine_critical_relative_error": math.exp(extrapolated[-1]) / curve.pc_bar - 1.0,
        "perry_critical_relative_error": math.exp(reference[-1]) / curve.pc_bar - 1.0,
        "nonmonotone": min(derivatives) <= 0.0,
        "overshoots_pc": max(completed[:-1]) > math.log(curve.pc_bar) + 1.0e-10,
    }


def describe(rows, label):
    c1 = [row["c1_mard"] for row in rows]
    oracle = [row["oracle_mard"] for row in rows]
    handoff = [abs(row["handoff_relative_error"]) for row in rows]
    slope = [abs(row["handoff_slope_relative_error"]) for row in rows]
    maxima = [row["c1_max_abs_relative_error"] for row in rows]
    extrapolated = [row["antoine_extrapolation_mard"] for row in rows]
    print(f"{label}: n={len(rows)}")
    print(f"  Antoine/Perry handoff |P error|: median={100*median(handoff):.3f}% p90={100*percentile(handoff,.90):.3f}% p95={100*percentile(handoff,.95):.3f}%")
    print(f"  Antoine/Perry handoff |slope error|: median={100*median(slope):.3f}% p90={100*percentile(slope,.90):.3f}% p95={100*percentile(slope,.95):.3f}%")
    print(f"  C1 AW completion MARD: mean={100*mean(c1):.3f}% median={100*median(c1):.3f}% p90={100*percentile(c1,.90):.3f}% p95={100*percentile(c1,.95):.3f}% max={100*max(c1):.3f}%")
    print(f"  C1 AW per-curve max error: median={100*median(maxima):.3f}% p95={100*percentile(maxima,.95):.3f}% max={100*max(maxima):.3f}%")
    print(f"  Oracle Perry-handoff C1 MARD: mean={100*mean(oracle):.3f}% median={100*median(oracle):.3f}%")
    print(f"  Direct Antoine extrapolation MARD: mean={100*mean(extrapolated):.3f}% median={100*median(extrapolated):.3f}%")
    print(f"  nonmonotone={sum(row['nonmonotone'] for row in rows)}, overshoots Pc={sum(row['overshoots_pc'] for row in rows)}")


def main():
    matches = load_matches()
    rows = [curve_metrics(entry, curve) for entry, curve in matches]
    print(f"MATCHED COMPOUNDS: {len(rows)}")
    describe(rows, "All")
    print("\nBY ANTOINE ENDPOINT Tr")
    order = ("<0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80", "0.80-0.85", ">=0.85")
    for label in order:
        subset = [row for row in rows if row["endpoint_bin"] == label]
        if subset:
            describe(subset, label)

    print("\nHANDOFF-CONSISTENT SUBSETS")
    for threshold in (0.01, 0.02, 0.05):
        subset = [row for row in rows if abs(row["handoff_relative_error"]) <= threshold]
        describe(subset, f"|Antoine/Perry endpoint mismatch| <= {100*threshold:g}%")

    print("\nWORST 20 C1 COMPLETIONS")
    for row in sorted(rows, key=lambda item: item["c1_mard"], reverse=True)[:20]:
        print(
            f"  {row['name']} [{row['cas']}]: Tr0={row['endpoint_Tr']:.3f}, "
            f"handoff P={100*row['handoff_relative_error']:+.2f}%, "
            f"slope={100*row['handoff_slope_relative_error']:+.2f}%, "
            f"MARD={100*row['c1_mard']:.2f}%, max={100*row['c1_max_abs_relative_error']:.2f}%"
        )

    output = Path("/tmp/antoine_aw_tail_benchmark.csv")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
