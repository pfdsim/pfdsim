import math
from statistics import mean, median

from chemicals.identifiers import search_chemical
from scipy.interpolate import PchipInterpolator

from benchmark_perry_aw import percentile
from benchmark_table210_aw import (
    evaluate,
    inverse_temperature_quadratic_slope,
    load_matches,
)
from nannoolal_method import estimate_psat
from perry_properties import PerryPropertyLibrary
from pressure_standards import NORMAL_BOILING_PRESSURE_BAR
from property_resolution.common import (
    ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K,
    ONLINE_ANTOINE_TB_REL_TOL,
    R,
)


def tb_valid_matches():
    library = PerryPropertyLibrary()
    result = []
    for curve, pairs in load_matches():
        tb = library.normal_boiling_point_K(curve.cas)
        if tb is None:
            continue
        trusted_tb = float(tb.value)
        temperatures = [pair[0] for pair in pairs]
        ln_pressures = [math.log(pair[1]) for pair in pairs]
        if not (
            temperatures[0] - ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K
            <= trusted_tb
            <= temperatures[-1] + ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K
        ):
            continue
        pressure = math.exp(float(PchipInterpolator(temperatures, ln_pressures, extrapolate=True)(trusted_tb)))
        if abs(pressure / NORMAL_BOILING_PRESSURE_BAR - 1.0) > ONLINE_ANTOINE_TB_REL_TOL:
            continue
        result.append((curve, pairs, trusted_tb))
    return result


def nannoolal_slope(smiles, temperature, pressure_bar):
    estimate = estimate_psat(
        smiles,
        psat_point=(temperature, pressure_bar * 100.0),
    )
    if estimate.db is None or estimate.tb_K is None:
        return None, estimate
    anchored_pressure = estimate.psat_kPa(temperature)
    dhvap = estimate.dhvap_J_mol(temperature, dz_vap=1.0)
    if anchored_pressure is None or dhvap is None or anchored_pressure <= 0.0 or dhvap <= 0.0:
        return None, estimate
    slope = dhvap / (R * temperature**2)
    return slope, estimate


def main():
    matches = tb_valid_matches()
    rows = []
    unavailable = []
    anchor_errors = []
    for curve, pairs, trusted_tb in matches:
        try:
            metadata = search_chemical(curve.cas)
            smiles = metadata.smiles
        except Exception as error:
            unavailable.append((curve.name, f"identity: {error}"))
            continue
        if not smiles:
            unavailable.append((curve.name, "missing SMILES"))
            continue
        try:
            slope, estimate = nannoolal_slope(smiles, pairs[-1][0], pairs[-1][1])
        except Exception as error:
            unavailable.append((curve.name, f"Nannoolal: {error}"))
            continue
        if slope is None:
            unavailable.append((curve.name, "; ".join(estimate.warnings) or "not estimable"))
            continue
        anchored_pressure_bar = estimate.psat_kPa(pairs[-1][0]) / 100.0
        anchor_errors.append(abs(anchored_pressure_bar / pairs[-1][1] - 1.0))
        row = evaluate(
            curve,
            pairs,
            "nannoolal_single_point",
            lambda _pairs, slope=slope: slope,
        )
        row["smiles"] = smiles
        row["trusted_Tb_K"] = trusted_tb
        row["nannoolal_tb_K"] = estimate.tb_K
        row["nannoolal_alcohol_correction"] = estimate.alcohol_correction
        row["nannoolal_warnings"] = "; ".join(estimate.warnings)
        table_row = evaluate(
            curve,
            pairs,
            "inverse_T_quadratic_last5",
            lambda values: inverse_temperature_quadratic_slope(values, 5),
        )
        rows.append((row, table_row))

    print("NANNOOLAL SINGLE-POINT DERIVATIVE")
    print(f"Tb-validated tables={len(matches)}")
    print(f"Nannoolal slopes available={len(rows)}")
    print(f"unavailable={len(unavailable)}")
    if unavailable:
        print("unavailable examples:")
        for name, reason in unavailable[:20]:
            print(f"  {name}: {reason}")
    print(f"maximum anchor pressure residual={100*max(anchor_errors):.12g}%")

    nannoolal_rows = [row for row, _ in rows]
    table_rows = [row for _, row in rows]
    slope_errors = [abs(row["slope_relative_error"]) for row in nannoolal_rows]
    mards = [row["c1_mard"] for row in nannoolal_rows]
    maxima = [row["c1_max_abs_relative_error"] for row in nannoolal_rows]
    oracle = [row["oracle_mard"] for row in nannoolal_rows]
    print(
        f"slope |relative error|: mean={100*mean(slope_errors):.3f}% "
        f"median={100*median(slope_errors):.3f}% "
        f"p90={100*percentile(slope_errors,.90):.3f}% "
        f"p95={100*percentile(slope_errors,.95):.3f}% "
        f"max={100*max(slope_errors):.3f}%"
    )
    print(
        f"completion MARD: mean={100*mean(mards):.3f}% "
        f"median={100*median(mards):.3f}% "
        f"p90={100*percentile(mards,.90):.3f}% "
        f"p95={100*percentile(mards,.95):.3f}% "
        f"max={100*max(mards):.3f}%"
    )
    print(
        f"per-curve max error: median={100*median(maxima):.3f}% "
        f"p95={100*percentile(maxima,.95):.3f}% "
        f"max={100*max(maxima):.3f}%"
    )
    print(
        f"oracle MARD at same endpoints: mean={100*mean(oracle):.3f}% "
        f"median={100*median(oracle):.3f}% "
        f"p95={100*percentile(oracle,.95):.3f}%"
    )
    print(
        f"nonmonotone={sum(row['nonmonotone'] for row in nannoolal_rows)}, "
        f"overshoots={sum(row['overshoots_pc'] for row in nannoolal_rows)}"
    )

    table_mards = [row["c1_mard"] for row in table_rows]
    print(
        f"paired five-point table MARD: mean={100*mean(table_mards):.3f}% "
        f"median={100*median(table_mards):.3f}% "
        f"p90={100*percentile(table_mards,.90):.3f}% "
        f"p95={100*percentile(table_mards,.95):.3f}% "
        f"max={100*max(table_mards):.3f}%"
    )
    print(
        f"paired wins: Nannoolal="
        f"{sum(nannoolal['c1_mard'] < table['c1_mard'] for nannoolal, table in rows)}, "
        f"five-point table="
        f"{sum(table['c1_mard'] < nannoolal['c1_mard'] for nannoolal, table in rows)}"
    )

    ordinary = [
        (nannoolal, table)
        for nannoolal, table in rows
        if not nannoolal["nannoolal_alcohol_correction"]
        and "acid" not in nannoolal["name"].lower()
        and "amine" not in nannoolal["name"].lower()
        and "glycol" not in nannoolal["name"].lower()
    ]
    for label, index in (("Nannoolal ordinary", 0), ("five-point table ordinary", 1)):
        values = [pair[index]["c1_mard"] for pair in ordinary]
        print(
            f"{label}: n={len(values)}, mean={100*mean(values):.3f}%, "
            f"median={100*median(values):.3f}%, "
            f"p95={100*percentile(values,.95):.3f}%, max={100*max(values):.3f}%"
        )

    print("\nBY ENDPOINT Tr")
    for label in ("<0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80"):
        subset = [row for row in nannoolal_rows if row["endpoint_bin"] == label]
        if not subset:
            continue
        values = [row["c1_mard"] for row in subset]
        print(
            f"{label}: n={len(subset)}, mean={100*mean(values):.3f}%, "
            f"median={100*median(values):.3f}%, "
            f"p95={100*percentile(values,.95):.3f}%, max={100*max(values):.3f}%"
        )

    print("\nWORST 20")
    for row in sorted(nannoolal_rows, key=lambda item: item["c1_mard"], reverse=True)[:20]:
        print(
            f"  {row['name']} [{row['cas']}]: Tr0={row['endpoint_Tr']:.3f}, "
            f"P={100*row['perry_endpoint_relative_error']:+.2f}%, "
            f"slope={100*row['slope_relative_error']:+.2f}%, "
            f"MARD={100*row['c1_mard']:.2f}%, max={100*row['c1_max_abs_relative_error']:.2f}%"
        )


if __name__ == "__main__":
    main()
