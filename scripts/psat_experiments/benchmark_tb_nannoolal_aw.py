import csv
import json
import math
from pathlib import Path
from statistics import mean, median

from chemicals.identifiers import search_chemical

from benchmark_perry_aw import load_curves, percentile
from benchmark_table210_aw import evaluate
from benchmark_table210_nannoolal_aw import nannoolal_slope, tb_valid_matches
from perry_properties import PerryPropertyLibrary
from pressure_standards import NORMAL_BOILING_PRESSURE_BAR
from textbook_properties import get_textbook_property_library


CHEMICALS_PATH = Path(__file__).parents[2] / "data" / "chemicals.json"
OUTPUT_PATH = Path("/tmp/direct_tb_nannoolal_aw_benchmark.csv")


def direct_tb_candidates(curves):
    textbook = get_textbook_property_library()
    chemicals_payload = json.loads(CHEMICALS_PATH.read_text())
    chemicals_by_cas = {
        str(entry["CAS"]): entry
        for entry in chemicals_payload.get("chemicals", {}).values()
        if entry.get("CAS") and entry.get("Tb") is not None
    }

    candidates = []
    for curve in curves:
        textbook_entry = textbook.get(curve.name)
        if textbook_entry and textbook_entry.get("Tb") is not None:
            candidates.append((curve, float(textbook_entry["Tb"]), "textbook"))
        chemicals_entry = chemicals_by_cas.get(curve.cas)
        if chemicals_entry:
            candidates.append((curve, float(chemicals_entry["Tb"]), "chemicals.json"))
    return candidates


def preferred_candidates(candidates):
    source_priority = {"textbook": 0, "chemicals.json": 1}
    preferred = {}
    for candidate in candidates:
        curve, _, source = candidate
        current = preferred.get(curve.cas)
        if current is None or source_priority[source] < source_priority[current[2]]:
            preferred[curve.cas] = candidate
    return list(preferred.values())


def evaluate_candidates(candidates):
    perry = PerryPropertyLibrary()
    rows = []
    unavailable = []
    rejected = []
    anchor_errors = []

    for curve, tb, source in candidates:
        if not curve.t_min <= tb < curve.tc:
            rejected.append((curve.name, source, tb, curve.t_min, curve.tc))
            continue
        reference_tb_result = perry.normal_boiling_point_K(curve.cas)
        if reference_tb_result is None:
            rejected.append((curve.name, source, tb, curve.t_min, curve.tc))
            continue
        try:
            metadata = search_chemical(curve.cas)
            smiles = metadata.smiles
        except Exception as error:
            unavailable.append((curve.name, source, f"identity: {error}"))
            continue
        if not smiles:
            unavailable.append((curve.name, source, "missing SMILES"))
            continue
        try:
            slope, estimate = nannoolal_slope(
                smiles,
                tb,
                NORMAL_BOILING_PRESSURE_BAR,
            )
        except Exception as error:
            unavailable.append((curve.name, source, f"Nannoolal: {error}"))
            continue
        if slope is None:
            unavailable.append(
                (curve.name, source, "; ".join(estimate.warnings) or "not estimable")
            )
            continue

        anchored_pressure_bar = estimate.psat_kPa(tb) / 100.0
        anchor_errors.append(
            abs(anchored_pressure_bar / NORMAL_BOILING_PRESSURE_BAR - 1.0)
        )
        row = evaluate(
            curve,
            [(tb, NORMAL_BOILING_PRESSURE_BAR)],
            "direct_tb_nannoolal",
            lambda _pairs, slope=slope: slope,
        )
        row["tb_source"] = source
        row["reference_Tb_K"] = float(reference_tb_result.value)
        row["tb_error_K"] = tb - float(reference_tb_result.value)
        row["smiles"] = smiles
        row["nannoolal_alcohol_correction"] = estimate.alcohol_correction
        row["nannoolal_warnings"] = "; ".join(estimate.warnings)
        rows.append(row)

    return rows, unavailable, rejected, anchor_errors


def table210_comparison_rows(cas_values):
    rows = {}
    for curve, pairs, _trusted_tb in tb_valid_matches():
        if curve.cas not in cas_values:
            continue
        try:
            smiles = search_chemical(curve.cas).smiles
            slope, _estimate = nannoolal_slope(smiles, pairs[-1][0], pairs[-1][1])
        except Exception:
            continue
        if slope is None:
            continue
        rows[curve.cas] = evaluate(
            curve,
            pairs,
            "table210_nannoolal",
            lambda _pairs, slope=slope: slope,
        )
    return rows


def summarize(rows, label):
    if not rows:
        print(f"{label}: n=0")
        return
    pressure_errors = [abs(row["perry_endpoint_relative_error"]) for row in rows]
    slope_errors = [abs(row["slope_relative_error"]) for row in rows]
    mards = [row["c1_mard"] for row in rows]
    maxima = [row["c1_max_abs_relative_error"] for row in rows]
    oracle = [row["oracle_mard"] for row in rows]
    print(f"{label}: n={len(rows)}")
    if all("tb_error_K" in row for row in rows):
        tb_errors = [abs(row["tb_error_K"]) for row in rows]
        print(
            f"  |Tb - Perry Tb| median={median(tb_errors):.3f} K "
            f"p95={percentile(tb_errors, .95):.3f} K max={max(tb_errors):.3f} K"
        )
    print(
        f"  endpoint |P mismatch| median={100*median(pressure_errors):.3f}% "
        f"p95={100*percentile(pressure_errors, .95):.3f}%"
    )
    print(
        f"  endpoint |slope error| median={100*median(slope_errors):.3f}% "
        f"p90={100*percentile(slope_errors, .90):.3f}% "
        f"p95={100*percentile(slope_errors, .95):.3f}%"
    )
    print(
        f"  completion MARD mean={100*mean(mards):.3f}% "
        f"median={100*median(mards):.3f}% p90={100*percentile(mards, .90):.3f}% "
        f"p95={100*percentile(mards, .95):.3f}% max={100*max(mards):.3f}%"
    )
    print(
        f"  max-error median={100*median(maxima):.3f}% "
        f"p95={100*percentile(maxima, .95):.3f}% "
        f"oracle median={100*median(oracle):.3f}% "
        f"nonmonotone={sum(row['nonmonotone'] for row in rows)} "
        f"overshoots={sum(row['overshoots_pc'] for row in rows)}"
    )


def is_ordinary(row):
    name = row["name"].lower()
    return (
        not row["nannoolal_alcohol_correction"]
        and "acid" not in name
        and "amine" not in name
        and "glycol" not in name
    )


def main():
    curves = load_curves()
    candidates = direct_tb_candidates(curves)
    preferred = preferred_candidates(candidates)
    rows, unavailable, rejected, anchor_errors = evaluate_candidates(candidates)
    preferred_cas = {curve.cas for curve, _, _ in preferred}
    preferred_source = {curve.cas: source for curve, _, source in preferred}
    preferred_rows = [
        row
        for row in rows
        if row["cas"] in preferred_cas
        and row["tb_source"] == preferred_source[row["cas"]]
    ]
    table210_rows = table210_comparison_rows({row["cas"] for row in preferred_rows})
    paired = [
        (row, table210_rows[row["cas"]])
        for row in preferred_rows
        if row["cas"] in table210_rows
    ]

    print("DIRECT Tb + NANNOOLAL DERIVATIVE + C1 AMBROSE-WALTON")
    print(f"Perry reference curves={len(curves)}")
    print(f"direct Tb candidates={len(candidates)}")
    print(f"preferred unique compounds={len(preferred)}")
    print(f"evaluated source rows={len(rows)}")
    print(f"rejected outside Perry range or missing reference Tb={len(rejected)}")
    print(f"Nannoolal unavailable={len(unavailable)}")
    if anchor_errors:
        print(f"maximum Nannoolal anchor residual={100*max(anchor_errors):.12g}%")

    print("\nBY DIRECT Tb SOURCE")
    for source in ("textbook", "chemicals.json"):
        summarize([row for row in rows if row["tb_source"] == source], source)

    print("\nPREFERRED UNION (textbook before chemicals.json)")
    summarize(preferred_rows, "all preferred")
    summarize([row for row in preferred_rows if is_ordinary(row)], "ordinary preferred")

    print("\nPAIRED AGAINST TABLE 2-10 ENDPOINT")
    summarize([direct for direct, _table in paired], "direct Tb paired")
    summarize([table for _direct, table in paired], "table 2-10 paired")
    print(
        f"paired wins: direct Tb="
        f"{sum(direct['c1_mard'] < table['c1_mard'] for direct, table in paired)}, "
        f"table 2-10="
        f"{sum(table['c1_mard'] < direct['c1_mard'] for direct, table in paired)}"
    )

    print("\nBY Tb REDUCED TEMPERATURE")
    bins = (
        ("<0.60", 0.0, 0.60),
        ("0.60-0.65", 0.60, 0.65),
        ("0.65-0.70", 0.65, 0.70),
        ("0.70-0.75", 0.70, 0.75),
        (">=0.75", 0.75, math.inf),
    )
    for label, lower, upper in bins:
        summarize(
            [row for row in preferred_rows if lower <= row["endpoint_Tr"] < upper],
            label,
        )

    print("\nUNAVAILABLE")
    for name, source, reason in unavailable:
        print(f"  {name} [{source}]: {reason}")
    print("\nREJECTED")
    for name, source, tb, t_min, tc in rejected:
        print(f"  {name} [{source}]: Tb={tb:.3f} K, Perry Tmin={t_min:.3f} K, Tc={tc:.3f} K")

    print("\nWORST 20 PREFERRED")
    for row in sorted(preferred_rows, key=lambda item: item["c1_mard"], reverse=True)[:20]:
        print(
            f"  {row['name']} [{row['cas']}]: source={row['tb_source']}, "
            f"Trb={row['endpoint_Tr']:.3f}, dTb={row['tb_error_K']:+.2f} K, "
            f"P={100*row['perry_endpoint_relative_error']:+.2f}%, "
            f"slope={100*row['slope_relative_error']:+.2f}%, "
            f"MARD={100*row['c1_mard']:.2f}%, "
            f"max={100*row['c1_max_abs_relative_error']:.2f}%"
        )

    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
