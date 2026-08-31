import csv
import math
from pathlib import Path
from statistics import mean, median

from benchmark_perry_aw import aw_ln_p, load_curves, percentile, perry_ln_p
from benchmark_tb_nannoolal_aw import direct_tb_candidates, preferred_candidates
from perry_properties import PerryPropertyLibrary
from pressure_standards import NORMAL_BOILING_PRESSURE_BAR


OUTPUT_PATH = Path("/tmp/tb_constrained_aw_benchmark.csv")


def aw_terms(reduced_temperature):
    tau = max(0.0, 1.0 - reduced_temperature)
    f0 = (
        -5.97616 * tau
        + 1.29874 * tau**1.5
        - 0.60394 * tau**2.5
        - 1.06841 * tau**5
    ) / reduced_temperature
    f1 = (
        -5.03365 * tau
        + 1.11505 * tau**1.5
        - 5.41217 * tau**2.5
        - 7.46628 * tau**5
    ) / reduced_temperature
    f2 = (
        -0.64771 * tau
        + 2.41539 * tau**1.5
        - 4.26979 * tau**2.5
        + 3.25259 * tau**5
    ) / reduced_temperature
    return f0, f1, f2


def tb_constrained_omega(curve, tb, pressure_bar):
    f0, f1, f2 = aw_terms(tb / curve.tc)
    target = math.log(pressure_bar / curve.pc_bar)
    if abs(f2) < 1.0e-12:
        return (target - f0) / f1, [(target - f0) / f1]
    discriminant = f1**2 - 4.0 * f2 * (f0 - target)
    if discriminant < 0.0:
        return None, []
    root = math.sqrt(discriminant)
    roots = [(-f1 + root) / (2.0 * f2), (-f1 - root) / (2.0 * f2)]
    plausible = [value for value in roots if -0.5 <= value <= 2.0]
    if len(plausible) == 1:
        return plausible[0], roots
    candidates = plausible or roots
    return min(candidates, key=lambda value: abs(value - curve.omega)), roots


def aw_ln_p_with_omega(curve, temperature, omega):
    if temperature >= curve.tc:
        return math.log(curve.pc_bar)
    f0, f1, f2 = aw_terms(temperature / curve.tc)
    return math.log(curve.pc_bar) + f0 + omega * f1 + omega**2 * f2


def evaluate(curve, tb, pressure_bar, tb_source):
    effective_omega, roots = tb_constrained_omega(curve, tb, pressure_bar)
    if effective_omega is None:
        return None
    temperatures = [tb + (curve.tc - tb) * index / 400.0 for index in range(401)]
    reference = [perry_ln_p(curve, temperature) for temperature in temperatures]
    plain = [aw_ln_p(curve, temperature) for temperature in temperatures]
    constrained = [
        aw_ln_p_with_omega(curve, temperature, effective_omega)
        for temperature in temperatures
    ]
    plain_errors = [
        abs(math.exp(predicted - truth) - 1.0)
        for predicted, truth in zip(plain, reference)
    ]
    constrained_errors = [
        abs(math.exp(predicted - truth) - 1.0)
        for predicted, truth in zip(constrained, reference)
    ]
    return {
        "cas": curve.cas,
        "name": curve.name,
        "tb_source": tb_source,
        "tb_K": tb,
        "tb_Tr": tb / curve.tc,
        "source_omega": curve.omega,
        "effective_omega": effective_omega,
        "omega_change": effective_omega - curve.omega,
        "other_root": max(roots, key=lambda value: abs(value - effective_omega)),
        "plain_aw_mard": mean(plain_errors),
        "constrained_aw_mard": mean(constrained_errors),
        "plain_aw_max_error": max(plain_errors),
        "constrained_aw_max_error": max(constrained_errors),
        "plain_endpoint_error": math.exp(plain[0]) / pressure_bar - 1.0,
        "constrained_endpoint_error": math.exp(constrained[0]) / pressure_bar - 1.0,
        "constrained_nonmonotone": any(
            right <= left for left, right in zip(constrained, constrained[1:])
        ),
        "constrained_overshoots_pc": (
            max(constrained[:-1]) > math.log(curve.pc_bar) + 1.0e-10
        ),
    }


def summarize(rows, label):
    plain = [row["plain_aw_mard"] for row in rows]
    constrained = [row["constrained_aw_mard"] for row in rows]
    print(f"{label}: n={len(rows)}")
    print(
        f"  plain AW: mean={100*mean(plain):.3f}% median={100*median(plain):.3f}% "
        f"p95={100*percentile(plain, .95):.3f}%"
    )
    print(
        f"  Tb-constrained AW: mean={100*mean(constrained):.3f}% "
        f"median={100*median(constrained):.3f}% "
        f"p95={100*percentile(constrained, .95):.3f}%"
    )
    print(
        f"  wins: constrained="
        f"{sum(row['constrained_aw_mard'] < row['plain_aw_mard'] for row in rows)}, "
        f"plain={sum(row['plain_aw_mard'] < row['constrained_aw_mard'] for row in rows)}"
    )
    print(
        f"  nonmonotone={sum(row['constrained_nonmonotone'] for row in rows)} "
        f"overshoots={sum(row['constrained_overshoots_pc'] for row in rows)}"
    )


def main():
    curves = load_curves()
    preferred = preferred_candidates(direct_tb_candidates(curves))
    direct_rows = []
    rejected = []
    for curve, tb, source in preferred:
        if not curve.t_min <= tb < curve.tc:
            rejected.append((curve.name, tb, curve.t_min, curve.tc))
            continue
        row = evaluate(curve, tb, NORMAL_BOILING_PRESSURE_BAR, source)
        if row is not None:
            direct_rows.append(row)

    library = PerryPropertyLibrary()
    exact_rows = []
    for curve in curves:
        tb_result = library.normal_boiling_point_K(curve.cas)
        if tb_result is None:
            continue
        row = evaluate(
            curve,
            float(tb_result.value),
            NORMAL_BOILING_PRESSURE_BAR,
            "Perry curve 1-atm crossing",
        )
        if row is not None:
            exact_rows.append(row)

    print("Tb-CONSTRAINED AMBROSE-WALTON")
    summarize(direct_rows, "independent direct Tb")
    summarize(exact_rows, "Perry-exact Tb diagnostic")
    omega_changes = [abs(row["omega_change"]) for row in direct_rows]
    print(
        f"direct |effective omega - source omega|: "
        f"median={median(omega_changes):.6f} "
        f"p95={percentile(omega_changes, .95):.6f} max={max(omega_changes):.6f}"
    )
    print(f"rejected direct Tbs outside Perry range={len(rejected)}")

    print("\nWORST 20 DIRECT Tb-CONSTRAINED")
    for row in sorted(
        direct_rows,
        key=lambda item: item["constrained_aw_mard"],
        reverse=True,
    )[:20]:
        print(
            f"  {row['name']} [{row['cas']}]: Trb={row['tb_Tr']:.3f}, "
            f"omega={row['source_omega']:.4f}->{row['effective_omega']:.4f}, "
            f"plain={100*row['plain_aw_mard']:.2f}%, "
            f"constrained={100*row['constrained_aw_mard']:.2f}%"
        )

    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=direct_rows[0].keys())
        writer.writeheader()
        writer.writerows(direct_rows)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
