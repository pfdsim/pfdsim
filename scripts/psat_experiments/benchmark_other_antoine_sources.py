import math
import re
from statistics import median

import numpy as np

from antoine_properties import get_antoine_table
from benchmark_perry_aw import percentile
from compound_identity import get_compound_identity_resolver
from perry_properties import PerryPropertyLibrary
from pressure_standards import NORMAL_BOILING_PRESSURE_BAR
from property_resolution.coolprop import (
    coolprop_props_si,
    coolprop_reference_from_candidates,
    coolprop_saturation_pressure,
    coolprop_saturation_temperature,
)
from property_resolution.runtime_cache import (
    RUNTIME_PROPERTY_CACHE_PATH,
    SQLiteJSONCache,
)
from textbook_properties import get_textbook_property_library


DIRECT_PRESSURE_LIMIT = 0.02
DIRECT_SLOPE_LIMIT = 0.10
SMOOTH_PRESSURE_LIMIT = 0.025
SMOOTH_SLOPE_LIMIT = 0.15


class ReferenceCurve:
    def __init__(self, source, T_min, T_max, ln_pressure, derivative, T_boiling):
        self.source = source
        self.T_min = T_min
        self.T_max = T_max
        self.ln_pressure = ln_pressure
        self.derivative = derivative
        self.T_boiling = T_boiling


def normalized_cas(identifier):
    text = str(identifier or "").strip()
    if re.fullmatch(r"\d{2,7}-\d{2}-\d", text):
        return text
    return None


def resolve_cas(identifier, perry, identity):
    cas = normalized_cas(identifier)
    if cas:
        return cas
    entry = perry.get(str(identifier))
    if entry and entry.get("cas"):
        return str(entry["cas"])
    try:
        return identity.resolve_cas(str(identifier), allow_formula=False)
    except Exception:
        return None


def coolprop_reference_curve(cas):
    reference = coolprop_reference_from_candidates((cas,))
    if reference is None:
        return None
    T_min = coolprop_props_si("Ttriple", reference)
    T_max = coolprop_props_si("Tcrit", reference)
    if T_min is None or T_max is None or T_max <= T_min:
        return None
    T_max = math.nextafter(T_max, 0.0)

    def ln_pressure(T):
        pressure_pa = coolprop_saturation_pressure(reference, T)
        if pressure_pa is None:
            raise ValueError("CoolProp saturation pressure unavailable")
        return math.log(pressure_pa / 100000.0)

    def derivative(T):
        step = min(max(1.0e-5 * T, 1.0e-3), (T_max - T_min) / 1000.0)
        if T - step >= T_min and T + step <= T_max:
            return (ln_pressure(T + step) - ln_pressure(T - step)) / (2.0 * step)
        if T + 2.0 * step <= T_max:
            return (
                -3.0 * ln_pressure(T)
                + 4.0 * ln_pressure(T + step)
                - ln_pressure(T + 2.0 * step)
            ) / (2.0 * step)
        return (
            3.0 * ln_pressure(T)
            - 4.0 * ln_pressure(T - step)
            + ln_pressure(T - 2.0 * step)
        ) / (2.0 * step)

    T_boiling = coolprop_saturation_temperature(
        reference,
        NORMAL_BOILING_PRESSURE_BAR * 100000.0,
    )
    return ReferenceCurve(
        f"CoolProp {reference.backend}",
        T_min,
        T_max,
        ln_pressure,
        derivative,
        T_boiling,
    )


def perry_reference_curve(cas, perry):
    entry = perry.get(cas, expand_identity=False)
    if not entry:
        return None
    rows = entry.get("vapor_pressure") or ()
    if len(rows) != 1:
        return None
    row = rows[0]
    coefficients = perry.vapor_pressure_coefficients(row)
    if coefficients is None:
        return None
    try:
        T_min = float(row["T_min_K"])
        T_max = float(row["T_max_K"])
    except (KeyError, TypeError, ValueError):
        return None
    C1, C2, C3, C4, C5 = coefficients

    def ln_pressure(T):
        return (
            C1
            + C2 / T
            + C3 * math.log(T)
            + C4 * T**C5
            - math.log(100000.0)
        )

    def derivative(T):
        return -C2 / T**2 + C3 / T + C4 * C5 * T ** (C5 - 1.0)

    boiling = perry.normal_boiling_point_K(cas)
    return ReferenceCurve(
        "Perry 2-8",
        T_min,
        T_max,
        ln_pressure,
        derivative,
        boiling.value if boiling else None,
    )


def reference_curve(cas, perry):
    return coolprop_reference_curve(cas) or perry_reference_curve(cas, perry)


def textbook_rows():
    library = get_textbook_property_library()
    library._load()
    for name, entry in library._chemicals.items():
        if any(entry.get(key) is None for key in (
            "antoine_A",
            "antoine_B",
            "antoine_C",
            "antoine_Tmin",
            "antoine_Tmax",
        )):
            continue
        yield {
            "source": "Smith Appendix B",
            "identifier": name,
            "A": float(entry["antoine_A"]),
            "B": float(entry["antoine_B"]),
            "C": float(entry["antoine_C"]),
            "T_min": float(entry["antoine_Tmin"]),
            "T_max": float(entry["antoine_Tmax"]),
        }


def antoine_txt_rows():
    table = get_antoine_table()
    for entries in table._by_name.values():
        for entry in entries:
            yield {
                "source": "data/antoine.txt",
                "identifier": entry.name,
                "A": entry.A,
                "B": entry.B,
                "C": entry.C,
                "T_min": entry.T_min,
                "T_max": entry.T_max,
            }


def cache_identifier(cache_key):
    identifier = str(cache_key)[len("antoine_"):]
    return re.sub(r"_\d+\.\d{2}K$", "", identifier)


def nist_cache_rows():
    cache = SQLiteJSONCache(RUNTIME_PROPERTY_CACHE_PATH, 'property_resolver')
    for cache_key, payload in cache.items(prefix='antoine_'):
        if payload.get("source") != "NIST WebBook":
            continue
        if any(payload.get(key) is None for key in ("A", "B", "C", "T_min", "T_max")):
            continue
        yield {
            "source": "cached NIST WebBook",
            "identifier": cache_identifier(cache_key),
            "A": float(payload["A"]),
            "B": float(payload["B"]),
            "C": float(payload["C"]),
            "T_min": float(payload["T_min"]),
            "T_max": float(payload["T_max"]),
        }


def evaluate_row(row, cas, reference):
    overlap_min = max(row["T_min"], reference.T_min)
    overlap_max = min(row["T_max"], reference.T_max)
    if overlap_max <= overlap_min:
        return None
    denominator_min = row["C"] + overlap_min - 273.15
    denominator_max = row["C"] + overlap_max - 273.15
    if (
        row["B"] <= 0.0
        or denominator_min == 0.0
        or denominator_max == 0.0
        or min(denominator_min, denominator_max) < 0.0 < max(
            denominator_min,
            denominator_max,
        )
    ):
        return None
    ln10 = math.log(10.0)

    def ln_pressure(T):
        return ln10 * (row["A"] - row["B"] / (row["C"] + T - 273.15))

    def derivative(T):
        return ln10 * row["B"] / (row["C"] + T - 273.15) ** 2

    temperatures = np.linspace(overlap_min, overlap_max, 201)
    pressure_errors = []
    slope_errors = []
    for temperature in temperatures:
        try:
            pressure_errors.append(abs(math.expm1(
                ln_pressure(float(temperature))
                - reference.ln_pressure(float(temperature))
            )))
            source_slope = derivative(float(temperature))
            reference_slope = reference.derivative(float(temperature))
        except (ArithmeticError, TypeError, ValueError):
            return None
        slope_errors.append(
            abs(source_slope - reference_slope)
            / max(abs(source_slope), abs(reference_slope), 1.0e-15)
        )
    pressure_errors = np.asarray(pressure_errors)
    slope_errors = np.asarray(slope_errors)
    direct = (pressure_errors <= DIRECT_PRESSURE_LIMIT) & (
        slope_errors <= DIRECT_SLOPE_LIMIT
    )
    smooth = (pressure_errors <= SMOOTH_PRESSURE_LIMIT) & (
        slope_errors <= SMOOTH_SLOPE_LIMIT
    )
    boiling_error = None
    covers_boiling = False
    if reference.T_boiling is not None and (
        row["T_min"] - 1.0 <= reference.T_boiling <= row["T_max"] + 1.0
    ):
        covers_boiling = True
        boiling_error = abs(
            math.exp(ln_pressure(reference.T_boiling))
            / NORMAL_BOILING_PRESSURE_BAR
            - 1.0
        )
    return {
        **row,
        "cas": cas,
        "reference": reference.source,
        "overlap_T_min": overlap_min,
        "overlap_T_max": overlap_max,
        "pressure_mard": float(np.mean(pressure_errors)),
        "pressure_p95": float(np.quantile(pressure_errors, 0.95)),
        "pressure_max": float(np.max(pressure_errors)),
        "slope_median": float(np.median(slope_errors)),
        "slope_p95": float(np.quantile(slope_errors, 0.95)),
        "minimum_pressure_error": float(np.min(pressure_errors)),
        "any_direct_handoff": bool(np.any(direct)),
        "any_smooth_handoff": bool(np.any(smooth)),
        "covers_boiling": covers_boiling,
        "boiling_error": boiling_error,
        "passes_boiling": covers_boiling and boiling_error <= 0.02,
    }


def collect(source_rows):
    perry = PerryPropertyLibrary()
    identity = get_compound_identity_resolver()
    results = []
    seen = set()
    for row in source_rows:
        cas = resolve_cas(row["identifier"], perry, identity)
        if not cas:
            continue
        key = (
            cas,
            row["A"],
            row["B"],
            row["C"],
            row["T_min"],
            row["T_max"],
        )
        if key in seen:
            continue
        seen.add(key)
        reference = reference_curve(cas, perry)
        if reference is None:
            continue
        result = evaluate_row(row, cas, reference)
        if result is not None:
            results.append(result)
    return results


def summarize(rows, label):
    print(f"{label}: matched rows={len(rows)} components={len({row['cas'] for row in rows})}")
    if not rows:
        return
    print(
        f"  references: CoolProp={sum(row['reference'].startswith('CoolProp') for row in rows)} "
        f"Perry2-8={sum(row['reference']=='Perry 2-8' for row in rows)}"
    )
    pressure_mards = [row["pressure_mard"] for row in rows]
    pressure_p95 = [row["pressure_p95"] for row in rows]
    slope_medians = [row["slope_median"] for row in rows]
    slope_p95 = [row["slope_p95"] for row in rows]
    print(
        f"  per-row pressure MARD median={100*median(pressure_mards):.3f}% "
        f"p90={100*percentile(pressure_mards, .90):.3f}% "
        f"p95={100*percentile(pressure_mards, .95):.3f}%"
    )
    print(
        f"  per-row pressure-p95 median={100*median(pressure_p95):.3f}% "
        f"p95={100*percentile(pressure_p95, .95):.3f}%"
    )
    print(
        f"  slope median-error median={100*median(slope_medians):.3f}% "
        f"p95={100*percentile(slope_medians, .95):.3f}%; "
        f"slope-p95 median={100*median(slope_p95):.3f}%"
    )
    print(
        f"  any direct handoff={sum(row['any_direct_handoff'] for row in rows)}/{len(rows)} "
        f"any smoothing-envelope handoff={sum(row['any_smooth_handoff'] for row in rows)}/{len(rows)}"
    )
    boiling_rows = [row for row in rows if row["covers_boiling"]]
    accepted_rows = [
        row for row in rows
        if row["passes_boiling"]
        or (not row["covers_boiling"] and row["any_smooth_handoff"])
    ]
    print(
        f"  covers reference Tb={len(boiling_rows)}; "
        f"passes 2% Tb gate={sum(row['passes_boiling'] for row in boiling_rows)}/{len(boiling_rows)}"
    )
    print(
        f"  accepted by Tb-or-overlap policy={len(accepted_rows)}/{len(rows)}; "
        f"accepted pressure-MARD median="
        f"{100*median([row['pressure_mard'] for row in accepted_rows]):.3f}%"
    )


def main():
    sources = (
        ("Smith Appendix B", textbook_rows()),
        ("data/antoine.txt", antoine_txt_rows()),
        ("cached NIST WebBook", nist_cache_rows()),
    )
    for label, source_rows in sources:
        rows = collect(source_rows)
        summarize(rows, label)
        print("  worst pressure MARD:")
        for row in sorted(rows, key=lambda item: item["pressure_mard"], reverse=True)[:5]:
            print(
                f"    {row['identifier']} [{row['cas']}] vs {row['reference']}: "
                f"MARD={100*row['pressure_mard']:.2f}% "
                f"slope-p95={100*row['slope_p95']:.2f}%"
            )
        print()


if __name__ == "__main__":
    main()
