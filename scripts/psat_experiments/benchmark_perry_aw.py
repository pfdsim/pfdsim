import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from perry_properties import PerryPropertyLibrary


@dataclass
class Curve:
    cas: str
    name: str
    tc: float
    pc_bar: float
    omega: float
    t_min: float
    t_max: float
    coefficients: tuple[float, ...]


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return math.nan
    position = fraction * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def perry_ln_p(curve, temperature):
    c1, c2, c3, c4, c5 = curve.coefficients[:5]
    return c1 + c2 / temperature + c3 * math.log(temperature) + c4 * temperature**c5 - math.log(100000.0)


def perry_dlnp_dt(curve, temperature):
    _, c2, c3, c4, c5 = curve.coefficients[:5]
    return -c2 / temperature**2 + c3 / temperature + c4 * c5 * temperature ** (c5 - 1.0)


def aw_ln_p(curve, temperature):
    tr = temperature / curve.tc
    if tr >= 1.0:
        return math.log(curve.pc_bar)
    tau = 1.0 - tr
    f0 = (-5.97616 * tau + 1.29874 * tau**1.5 - 0.60394 * tau**2.5 - 1.06841 * tau**5) / tr
    f1 = (-5.03365 * tau + 1.11505 * tau**1.5 - 5.41217 * tau**2.5 - 7.46628 * tau**5) / tr
    f2 = (-0.64771 * tau + 2.41539 * tau**1.5 - 4.26979 * tau**2.5 + 3.25259 * tau**5) / tr
    return math.log(curve.pc_bar) + f0 + curve.omega * f1 + curve.omega**2 * f2


def aw_dlnp_dt(curve, temperature):
    step = max(1.0e-5 * curve.tc, 1.0e-4)
    return (aw_ln_p(curve, temperature + step) - aw_ln_p(curve, temperature - step)) / (2.0 * step)


def normalize_vapor_pressure_coefficients(row):
    raw = [float(value) for value in row["coefficients"]]
    if len(raw) == 5:
        return tuple(raw)
    if len(raw) == 3:
        return tuple(raw + [0.0, 1.0])

    candidates = []
    if len(raw) > 5:
        for omitted in range(len(raw)):
            candidate = raw[:omitted] + raw[omitted + 1:]
            if len(candidate) == 5:
                candidates.append(candidate)
    if not candidates:
        raise ValueError(f"Unsupported Perry Psat coefficient count: {len(raw)}")

    def endpoint_error(coefficients):
        c1, c2, c3, c4, c5 = coefficients
        total = 0.0
        for temperature_key, pressure_key in (("T_min_K", "P_at_T_min_Pa"), ("T_max_K", "P_at_T_max_Pa")):
            temperature = float(row[temperature_key])
            reference = float(row[pressure_key])
            try:
                predicted_ln = c1 + c2 / temperature + c3 * math.log(temperature) + c4 * temperature**c5
                total += abs(predicted_ln - math.log(reference))
            except (ValueError, OverflowError):
                return math.inf
        return total

    return tuple(min(candidates, key=endpoint_error))


def completed_ln_p(curve, temperature, cutoff_tr, order):
    t0 = cutoff_tr * curve.tc
    span = curve.tc - t0
    x = (temperature - t0) / span
    baseline = aw_ln_p(curve, temperature)
    delta_value = perry_ln_p(curve, t0) - aw_ln_p(curve, t0)
    if order == 0:
        return baseline
    if order == 1:
        return baseline + (1.0 - x) ** 2 * delta_value
    delta_slope = perry_dlnp_dt(curve, t0) - aw_dlnp_dt(curve, t0)
    coefficient = span * delta_slope + 2.0 * delta_value
    return baseline + (1.0 - x) ** 2 * (delta_value + coefficient * x)


def load_curves():
    library = PerryPropertyLibrary()
    library._load()
    curves = []
    for cas, entry in library.chemicals.items():
        rows = entry.get("vapor_pressure") or []
        critical = entry.get("critical_constants") or {}
        if len(rows) != 1:
            continue
        row = rows[0]
        required = (
            critical.get("Tc_K"), critical.get("Pc_MPa"), critical.get("omega"),
            row.get("T_min_K"), row.get("T_max_K"), row.get("coefficients"),
        )
        if any(value is None for value in required):
            continue
        curves.append(Curve(
            cas=cas,
            name=entry.get("name") or cas,
            tc=float(critical["Tc_K"]),
            pc_bar=float(critical["Pc_MPa"]) * 10.0,
            omega=float(critical["omega"]),
            t_min=float(row["T_min_K"]),
            t_max=float(row["T_max_K"]),
            coefficients=normalize_vapor_pressure_coefficients(row),
        ))
    return curves


def endpoint_report(curves):
    rows = []
    for curve in curves:
        p_tc = math.exp(perry_ln_p(curve, curve.tc))
        relative_error = p_tc / curve.pc_bar - 1.0
        rows.append({
            "cas": curve.cas,
            "name": curve.name,
            "tc_K": curve.tc,
            "tmax_K": curve.t_max,
            "tmax_minus_tc_K": curve.t_max - curve.tc,
            "pc_bar": curve.pc_bar,
            "perry_p_at_tc_bar": p_tc,
            "relative_error": relative_error,
            "absolute_relative_error": abs(relative_error),
        })
    errors = [row["absolute_relative_error"] for row in rows]
    print("ENDPOINT CONSISTENCY")
    print(f"curves={len(rows)}")
    print("Perry P(Tc) vs Perry Pc absolute relative error:")
    print(f"  mean={100*mean(errors):.6f}% median={100*median(errors):.6f}% p90={100*percentile(errors,0.90):.6f}% p95={100*percentile(errors,0.95):.6f}% p99={100*percentile(errors,0.99):.6f}% max={100*max(errors):.6f}%")
    for threshold in (0.001, 0.005, 0.01, 0.02, 0.05):
        print(f"  <= {100*threshold:g}%: {sum(error <= threshold for error in errors)}/{len(errors)}")
    print("  worst 15:")
    for row in sorted(rows, key=lambda item: item["absolute_relative_error"], reverse=True)[:15]:
        print(f"    {row['name']} [{row['cas']}]: Tc={row['tc_K']:.5g} K, Tmax-Tc={row['tmax_minus_tc_K']:+.5g} K, Pc={row['pc_bar']:.6g} bar, Pfit(Tc)={row['perry_p_at_tc_bar']:.6g} bar, error={100*row['relative_error']:+.4f}%")
    return rows


def tail_report(curves, cutoff_tr):
    eligible = [curve for curve in curves if curve.t_min <= cutoff_tr * curve.tc and curve.t_max >= curve.tc - 0.01]
    print(f"\nTAIL COMPLETION FROM Tr={cutoff_tr:.1f}")
    print(f"eligible={len(eligible)}/{len(curves)}")
    methods = {0: "raw_aw", 1: "value_matched", 2: "value_slope_matched"}
    summaries = {order: [] for order in methods}
    pooled = {order: [] for order in methods}
    diagnostics = {order: {"nonmonotone": [], "overshoot": []} for order in methods}
    detail_rows = []

    for curve in eligible:
        temperatures = [curve.tc * (cutoff_tr + (1.0 - cutoff_tr) * index / 400.0) for index in range(401)]
        reference = [perry_ln_p(curve, temperature) for temperature in temperatures]
        for order, method in methods.items():
            predicted = [completed_ln_p(curve, temperature, cutoff_tr, order) for temperature in temperatures]
            relative_errors = [math.exp(pred - ref) - 1.0 for pred, ref in zip(predicted, reference)]
            absolute_errors = [abs(value) for value in relative_errors]
            mard = mean(absolute_errors)
            maximum = max(absolute_errors)
            summaries[order].append((mard, maximum, curve, relative_errors[-1]))
            pooled[order].extend(absolute_errors)
            derivatives = [
                (predicted[index + 1] - predicted[index - 1]) / (temperatures[index + 1] - temperatures[index - 1])
                for index in range(1, len(temperatures) - 1)
            ]
            if min(derivatives) <= 0.0:
                diagnostics[order]["nonmonotone"].append(curve)
            if max(predicted[:-1]) > math.log(curve.pc_bar) + 1.0e-10:
                diagnostics[order]["overshoot"].append(curve)
            detail_rows.append({
                "cutoff_tr": cutoff_tr,
                "method": method,
                "cas": curve.cas,
                "name": curve.name,
                "mard": mard,
                "max_abs_relative_error": maximum,
                "critical_relative_error_vs_perry": relative_errors[-1],
            })

    for order, method in methods.items():
        curve_mards = [item[0] for item in summaries[order]]
        curve_maxima = [item[1] for item in summaries[order]]
        print(f"  {method}:")
        print(f"    pooled MARD={100*mean(pooled[order]):.4f}%")
        print(f"    per-curve MARD median={100*median(curve_mards):.4f}% p90={100*percentile(curve_mards,0.90):.4f}% p95={100*percentile(curve_mards,0.95):.4f}% p99={100*percentile(curve_mards,0.99):.4f}% max={100*max(curve_mards):.4f}%")
        print(f"    per-curve max error median={100*median(curve_maxima):.4f}% p95={100*percentile(curve_maxima,0.95):.4f}% max={100*max(curve_maxima):.4f}%")
        print(f"    nonmonotone={len(diagnostics[order]['nonmonotone'])}, overshoots Pc={len(diagnostics[order]['overshoot'])}")
        print("    worst 12 by curve MARD:")
        for mard, maximum, curve, endpoint_error in sorted(summaries[order], key=lambda item: item[0], reverse=True)[:12]:
            print(f"      {curve.name} [{curve.cas}]: MARD={100*mard:.3f}%, max={100*maximum:.3f}%, endpoint-vs-Perry={100*endpoint_error:+.3f}%")

    sample_trs = [cutoff_tr + (1.0-cutoff_tr)*fraction for fraction in (0.125,0.25,0.5,0.75,0.9,0.95,0.99)]
    print("  value+slope matched pooled errors by Tr:")
    for tr in sample_trs:
        errors=[]
        for curve in eligible:
            temperature=tr*curve.tc
            errors.append(abs(math.exp(completed_ln_p(curve,temperature,cutoff_tr,2)-perry_ln_p(curve,temperature))-1.0))
        print(f"    Tr={tr:.5f}: MARD={100*mean(errors):.4f}% median={100*median(errors):.4f}% p95={100*percentile(errors,0.95):.4f}%")
    return detail_rows


def main():
    curves = load_curves()
    endpoint_rows = endpoint_report(curves)
    detail_rows = []
    for cutoff in (0.8, 0.9):
        detail_rows.extend(tail_report(curves, cutoff))
    endpoint_path = Path("/tmp/perry_endpoint_consistency.csv")
    detail_path = Path("/tmp/perry_aw_tail_benchmark.csv")
    with endpoint_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=endpoint_rows[0].keys())
        writer.writeheader(); writer.writerows(endpoint_rows)
    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_rows[0].keys())
        writer.writeheader(); writer.writerows(detail_rows)
    print(f"\nWrote {endpoint_path} and {detail_path}")


if __name__ == "__main__":
    main()
