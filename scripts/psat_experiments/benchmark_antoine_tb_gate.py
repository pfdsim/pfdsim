import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

from antoine_properties import get_antoine_table
from benchmark_antoine_aw import curve_metrics, describe, endpoint_bin
from benchmark_perry_aw import Curve, normalize_vapor_pressure_coefficients, percentile
from perry_properties import PerryPropertyLibrary
from pressure_standards import NORMAL_BOILING_PRESSURE_BAR
from property_resolution.common import (
    ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K,
    ONLINE_ANTOINE_TB_REL_TOL,
)


def unique_antoine_rows():
    table = get_antoine_table()
    seen = set()
    for rows in table._by_name.values():
        for entry in rows:
            key = (
                entry.record_id, entry.name, entry.A, entry.B, entry.C,
                entry.T_min, entry.T_max,
            )
            if key in seen:
                continue
            seen.add(key)
            yield entry


def build_curve(perry_entry):
    rows = perry_entry.get("vapor_pressure") or []
    critical = perry_entry.get("critical_constants") or {}
    if len(rows) != 1 or any(critical.get(key) is None for key in ("Tc_K", "Pc_MPa", "omega")):
        return None
    row = rows[0]
    return Curve(
        cas=perry_entry["cas"],
        name=perry_entry.get("name") or perry_entry["cas"],
        tc=float(critical["Tc_K"]),
        pc_bar=float(critical["Pc_MPa"]) * 10.0,
        omega=float(critical["omega"]),
        t_min=float(row["T_min_K"]),
        t_max=float(row["T_max_K"]),
        coefficients=normalize_vapor_pressure_coefficients(row),
    )


def load_validated_matches():
    perry = PerryPropertyLibrary()
    perry._load()
    all_by_cas = defaultdict(list)
    valid_by_cas = defaultdict(list)
    validation_rows = []

    for antoine in unique_antoine_rows():
        perry_entry = perry.get(antoine.name)
        if not perry_entry:
            continue
        curve = build_curve(perry_entry)
        if curve is None:
            continue
        tb_result = perry.normal_boiling_point_K(perry_entry["cas"])
        if tb_result is None:
            continue
        tb = float(tb_result.value)
        pressure_at_tb = antoine.vapor_pressure(tb)
        range_ok = (
            antoine.T_min - ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K
            <= tb
            <= antoine.T_max + ONLINE_ANTOINE_TB_RANGE_TOLERANCE_K
        )
        pressure_error = abs(pressure_at_tb / NORMAL_BOILING_PRESSURE_BAR - 1.0)
        pressure_ok = pressure_error <= ONLINE_ANTOINE_TB_REL_TOL
        benchmarkable = (
            curve.t_min <= antoine.T_max
            and antoine.T_min < antoine.T_max < curve.tc
            and curve.t_max >= curve.tc - 0.01
        )
        record = {
            "cas": curve.cas,
            "name": curve.name,
            "record_id": antoine.record_id,
            "Tb_K": tb,
            "Tmin_K": antoine.T_min,
            "Tmax_K": antoine.T_max,
            "endpoint_Tr": antoine.T_max / curve.tc,
            "pressure_at_Tb_bar": pressure_at_tb,
            "Tb_pressure_relative_error": pressure_at_tb / NORMAL_BOILING_PRESSURE_BAR - 1.0,
            "range_ok": range_ok,
            "pressure_ok": pressure_ok,
            "benchmarkable": benchmarkable,
        }
        validation_rows.append(record)
        all_by_cas[curve.cas].append((antoine, curve, record))
        if range_ok and pressure_ok and benchmarkable:
            valid_by_cas[curve.cas].append((antoine, curve, record))

    selected = [
        max(candidates, key=lambda item: item[0].T_max)
        for candidates in valid_by_cas.values()
    ]
    selected.sort(key=lambda item: item[0].T_max / item[1].tc)
    return selected, validation_rows, all_by_cas


def summarize_bins(rows):
    print("\nBY VALIDATED ANTOINE ENDPOINT Tr")
    for label in ("<0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80", "0.80-0.85", ">=0.85"):
        subset = [row for row in rows if endpoint_bin(row["endpoint_Tr"]) == label]
        if subset:
            describe(subset, label)


def main():
    selected, validation_rows, all_by_cas = load_validated_matches()
    rows = []
    for antoine, curve, validation in selected:
        result = curve_metrics(antoine, curve)
        result["Tb_K"] = validation["Tb_K"]
        result["Tb_pressure_relative_error"] = validation["Tb_pressure_relative_error"]
        rows.append(result)

    matched_compounds = len(all_by_cas)
    containing_tb = {
        row["cas"] for row in validation_rows if row["range_ok"] and row["benchmarkable"]
    }
    reproducing_tb = {
        row["cas"] for row in validation_rows
        if row["range_ok"] and row["pressure_ok"] and row["benchmarkable"]
    }
    print("TB VALIDATION GATE")
    print(f"matched compounds={matched_compounds}")
    print(f"with a benchmarkable Antoine row containing Tb={len(containing_tb)}")
    print(f"also reproducing 1 atm within {100*ONLINE_ANTOINE_TB_REL_TOL:g}%={len(reproducing_tb)}")
    print(f"selected validated curves={len(rows)}")
    tb_errors = [abs(row["Tb_pressure_relative_error"]) for row in rows]
    print(
        f"selected Tb pressure error: median={100*median(tb_errors):.4f}% "
        f"p90={100*percentile(tb_errors,.90):.4f}% "
        f"max={100*max(tb_errors):.4f}%"
    )
    print()
    describe(rows, "All Tb-validated")
    summarize_bins(rows)

    print("\nWORST 15 TB-VALIDATED COMPLETIONS")
    for row in sorted(rows, key=lambda item: item["c1_mard"], reverse=True)[:15]:
        print(
            f"  {row['name']} [{row['cas']}]: Tr0={row['endpoint_Tr']:.3f}, "
            f"Tb check={100*row['Tb_pressure_relative_error']:+.2f}%, "
            f"handoff P={100*row['handoff_relative_error']:+.2f}%, "
            f"slope={100*row['handoff_slope_relative_error']:+.2f}%, "
            f"MARD={100*row['c1_mard']:.2f}%"
        )

    output = Path("/tmp/antoine_tb_validated_aw_benchmark.csv")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
