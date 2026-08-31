import csv
import math
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.interpolate import PchipInterpolator

from benchmark_lower_hybrid import SMALL_POLAR_INORGANICS
from benchmark_perry_aw import percentile
from perry_properties import PerryPropertyLibrary
from property_resolution.common import R


OUTPUT_PATH = Path("/tmp/perry_table210_hvap_derivative.csv")


def is_banned(name):
    normalized = name.strip().lower()
    return "acid" in normalized or normalized in SMALL_POLAR_INORGANICS


def virial_rackett_delta_z(T, pressure_bar, Tc, Pc_bar, omega):
    Tr = T / Tc
    if not 0.0 < Tr < 0.995:
        return None
    b0 = (
        0.1445
        - 0.330 / Tr
        - 0.1385 / Tr**2
        - 0.0121 / Tr**3
        - 0.000607 / Tr**8
    )
    b1 = 0.0637 + 0.331 / Tr**2 - 0.423 / Tr**3 - 0.008 / Tr**8
    pressure_pa = pressure_bar * 100000.0
    Pc_pa = Pc_bar * 100000.0
    second_virial = R * Tc / Pc_pa * (b0 + omega * b1)
    vapor_z = 1.0 + second_virial * pressure_pa / (R * T)
    rackett_z = 0.29056 - 0.08775 * omega
    saturated_liquid_volume = (
        R
        * Tc
        / Pc_pa
        * rackett_z ** (1.0 + (1.0 - Tr) ** (2.0 / 7.0))
    )
    delta_z = vapor_z - pressure_pa * saturated_liquid_volume / (R * T)
    return delta_z if math.isfinite(delta_z) and delta_z > 0.05 else None


def patel_teja_valderrama_delta_z(
    T,
    pressure_bar,
    Tc,
    Pc_bar,
    Vc_m3_per_kmol,
    omega,
):
    pressure_pa = pressure_bar * 100000.0
    Pc_pa = Pc_bar * 100000.0
    critical_volume = Vc_m3_per_kmol / 1000.0
    critical_z = Pc_pa * critical_volume / (R * Tc)
    omega_a = 0.66121 - 0.76105 * critical_z
    omega_b = 0.02207 + 0.20868 * critical_z
    omega_c = 0.57765 - 1.87080 * critical_z
    factor = (
        0.46283
        + 3.58230 * omega * critical_z
        + 8.19417 * (omega * critical_z) ** 2
    )
    alpha = (1.0 + factor * (1.0 - math.sqrt(T / Tc))) ** 2
    a = omega_a * R**2 * Tc**2 / Pc_pa * alpha
    b = omega_b * R * Tc / Pc_pa
    c = omega_c * R * Tc / Pc_pa
    roots = np.roots([
        pressure_pa,
        pressure_pa * c - R * T,
        a - pressure_pa * (b**2 + 2.0 * b * c) - R * T * (b + c),
        pressure_pa * b**2 * c + R * T * b * c - a * b,
    ])
    volumes = sorted(
        root.real
        for root in roots
        if abs(root.imag) < 1.0e-9 and root.real > b
    )
    if len(volumes) < 2:
        return None
    delta_z = pressure_pa * (volumes[-1] - volumes[0]) / (R * T)
    return delta_z if math.isfinite(delta_z) and delta_z > 0.05 else None


def selected_window(point_count, index, count):
    count = min(count, point_count)
    start = max(0, min(index - count // 2, point_count - count))
    return slice(start, start + count)


def temperature_pchip_slopes(temperatures, ln_pressures):
    interpolator = PchipInterpolator(temperatures, ln_pressures)
    return np.asarray(interpolator.derivative()(temperatures), dtype=float)


def inverse_temperature_pchip_slopes(temperatures, ln_pressures):
    coordinate = -1.0 / temperatures
    interpolator = PchipInterpolator(coordinate, ln_pressures)
    return np.asarray(
        interpolator.derivative()(coordinate) / temperatures**2,
        dtype=float,
    )


def local_inverse_temperature_slopes(temperatures, ln_pressures, degree, count):
    slopes = []
    for index, temperature in enumerate(temperatures):
        window = selected_window(len(temperatures), index, count)
        selected_temperatures = temperatures[window]
        selected_pressures = ln_pressures[window]
        coordinate = (
            1.0 / selected_temperatures - 1.0 / temperature
        ) * temperature
        coefficients = np.polynomial.polynomial.polyfit(
            coordinate,
            selected_pressures,
            degree,
        )
        slopes.append(float(-coefficients[1] / temperature))
    return np.asarray(slopes, dtype=float)


def pchip_quadratic_blend_slopes(temperatures, ln_pressures):
    pchip = temperature_pchip_slopes(temperatures, ln_pressures)
    quadratic = local_inverse_temperature_slopes(
        temperatures,
        ln_pressures,
        2,
        5,
    )
    return 0.5 * (pchip + quadratic)


def robust_consensus_slopes(temperatures, ln_pressures):
    candidates = np.vstack((
        temperature_pchip_slopes(temperatures, ln_pressures),
        inverse_temperature_pchip_slopes(temperatures, ln_pressures),
        local_inverse_temperature_slopes(
            temperatures,
            ln_pressures,
            2,
            4,
        ),
        local_inverse_temperature_slopes(
            temperatures,
            ln_pressures,
            2,
            5,
        ),
    ))
    return np.median(candidates, axis=0)


DERIVATIVE_METHODS = {
    "temperature_log_pchip": temperature_pchip_slopes,
    "inverse_T_log_pchip": inverse_temperature_pchip_slopes,
    "inverse_T_linear_local3": (
        lambda temperatures, pressures: local_inverse_temperature_slopes(
            temperatures,
            pressures,
            1,
            3,
        )
    ),
    "inverse_T_linear_local5": (
        lambda temperatures, pressures: local_inverse_temperature_slopes(
            temperatures,
            pressures,
            1,
            5,
        )
    ),
    "inverse_T_quadratic_local4": (
        lambda temperatures, pressures: local_inverse_temperature_slopes(
            temperatures,
            pressures,
            2,
            4,
        )
    ),
    "inverse_T_quadratic_local5": (
        lambda temperatures, pressures: local_inverse_temperature_slopes(
            temperatures,
            pressures,
            2,
            5,
        )
    ),
    "pchip_quadratic_blend": pchip_quadratic_blend_slopes,
    "robust_consensus": robust_consensus_slopes,
}


def pressure_band(pressure_bar):
    if pressure_bar <= 0.01:
        return "<=0.01 bar"
    if pressure_bar <= 0.1:
        return "0.01-0.1 bar"
    if pressure_bar <= 0.5:
        return "0.1-0.5 bar"
    return ">0.5 bar"


def load_rows():
    library = PerryPropertyLibrary()
    library._load()
    library._load_table_2_10()
    rows = []
    paired_components = set()
    for cas, table_entry in library.table_2_10_chemicals.items():
        entry = library.get(cas, expand_identity=False)
        if not entry:
            continue
        pairs = sorted(
            (float(point["T_K"]), float(point["P_bar"]))
            for point in table_entry.get("vapor_pressure", ())
            if point.get("T_K") is not None and point.get("P_bar") is not None
        )
        if len(pairs) < 4:
            continue
        temperatures = np.asarray([pair[0] for pair in pairs], dtype=float)
        pressures = np.asarray([pair[1] for pair in pairs], dtype=float)
        if (
            np.any(~np.isfinite(temperatures))
            or np.any(~np.isfinite(pressures))
            or np.any(np.diff(temperatures) <= 0.0)
            or np.any(pressures <= 0.0)
        ):
            continue
        ln_pressures = np.log(pressures)
        method_slopes = {
            method: function(temperatures, ln_pressures)
            for method, function in DERIVATIVE_METHODS.items()
        }
        name = entry.get("name") or table_entry.get("name") or cas
        allowed = not is_banned(name)
        critical = entry.get("critical_constants") or {}
        critical_values = tuple(
            critical.get(key)
            for key in ("Tc_K", "Pc_MPa", "Vc_m3_per_kmol", "omega")
        )
        component_has_point = False
        for index, (temperature, pressure_bar) in enumerate(pairs):
            hvap = library.heat_of_vaporization_value_from_entry(
                entry,
                temperature,
            )
            if hvap is None:
                continue
            reference_hvap, hvap_row, _method = hvap
            component_has_point = True
            virial_delta_z = None
            ptv_delta_z = None
            if critical_values[0] is not None and critical_values[1] is not None and critical_values[3] is not None:
                virial_delta_z = virial_rackett_delta_z(
                    temperature,
                    pressure_bar,
                    float(critical_values[0]),
                    float(critical_values[1]) * 10.0,
                    float(critical_values[3]),
                )
            if all(value is not None for value in critical_values):
                ptv_delta_z = patel_teja_valderrama_delta_z(
                    temperature,
                    pressure_bar,
                    float(critical_values[0]),
                    float(critical_values[1]) * 10.0,
                    float(critical_values[2]),
                    float(critical_values[3]),
                )
            for method, slopes in method_slopes.items():
                slope = float(slopes[index])
                inferred_hvap = R * temperature**2 * slope / 1000.0
                relative_error = inferred_hvap / reference_hvap - 1.0
                rows.append({
                    "method": method,
                    "cas": cas,
                    "name": name,
                    "allowed": allowed,
                    "point_index": index,
                    "point_count": len(pairs),
                    "is_highest_point": index == len(pairs) - 1,
                    "temperature_K": temperature,
                    "pressure_bar": pressure_bar,
                    "pressure_band": pressure_band(pressure_bar),
                    "slope_dlnP_dT": slope,
                    "inferred_hvap_kJ_mol": inferred_hvap,
                    "perry_hvap_kJ_mol": reference_hvap,
                    "relative_error": relative_error,
                    "virial_delta_z": virial_delta_z,
                    "virial_relative_error": (
                        inferred_hvap * virial_delta_z / reference_hvap - 1.0
                        if virial_delta_z is not None
                        else math.nan
                    ),
                    "ptv_delta_z": ptv_delta_z,
                    "ptv_relative_error": (
                        inferred_hvap * ptv_delta_z / reference_hvap - 1.0
                        if ptv_delta_z is not None
                        else math.nan
                    ),
                    "implied_delta_z": (
                        reference_hvap / inferred_hvap
                        if inferred_hvap > 0.0
                        else math.nan
                    ),
                    "hvap_T_min_K": hvap_row.get("T_min_K"),
                    "hvap_T_max_K": hvap_row.get("T_max_K"),
                })
        if component_has_point:
            paired_components.add(cas)
    return rows, paired_components


def load_perry_2_8_overlap_rows():
    library = PerryPropertyLibrary()
    library._load()
    library._load_table_2_10()
    rows = []
    for cas, table_entry in library.table_2_10_chemicals.items():
        entry = library.get(cas, expand_identity=False)
        if not entry:
            continue
        vapor_rows = entry.get("vapor_pressure") or ()
        if len(vapor_rows) != 1:
            continue
        coefficients = library.vapor_pressure_coefficients(vapor_rows[0])
        if coefficients is None:
            continue
        pairs = sorted(
            (float(point["T_K"]), float(point["P_bar"]))
            for point in table_entry.get("vapor_pressure", ())
            if point.get("T_K") is not None and point.get("P_bar") is not None
        )
        if len(pairs) < 4:
            continue
        temperatures = np.asarray([pair[0] for pair in pairs], dtype=float)
        ln_pressures = np.log([pair[1] for pair in pairs])
        table_spline = PchipInterpolator(temperatures, ln_pressures)
        row = vapor_rows[0]
        overlap_min = max(temperatures[0], float(row["T_min_K"]))
        overlap_max = min(temperatures[-1], float(row["T_max_K"]))
        if overlap_max <= overlap_min:
            continue
        C1, C2, C3, C4, C5 = coefficients
        sample_temperatures = np.linspace(overlap_min, overlap_max, 401)
        table_ln_pressure = table_spline(sample_temperatures)
        table_slopes = table_spline.derivative()(sample_temperatures)
        perry_ln_pressure = (
            C1
            + C2 / sample_temperatures
            + C3 * np.log(sample_temperatures)
            + C4 * sample_temperatures**C5
            - math.log(100000.0)
        )
        perry_slopes = (
            -C2 / sample_temperatures**2
            + C3 / sample_temperatures
            + C4 * C5 * sample_temperatures ** (C5 - 1.0)
        )
        pressure_mismatch = np.abs(np.expm1(table_ln_pressure - perry_ln_pressure))
        slope_mismatch = np.abs(table_slopes - perry_slopes) / np.maximum(
            np.maximum(np.abs(table_slopes), np.abs(perry_slopes)),
            1.0e-15,
        )
        endpoint_pressure_mismatch = float(pressure_mismatch[-1])
        endpoint_slope_mismatch = float(slope_mismatch[-1])
        direct_mask = (pressure_mismatch <= 0.02) & (slope_mismatch <= 0.10)
        smooth_mask = (pressure_mismatch <= 0.025) & (slope_mismatch <= 0.15)
        name = entry.get("name") or table_entry.get("name") or cas
        rows.append({
            "cas": cas,
            "name": name,
            "allowed": not is_banned(name),
            "endpoint_pressure_mismatch": endpoint_pressure_mismatch,
            "endpoint_slope_mismatch": endpoint_slope_mismatch,
            "minimum_pressure_mismatch": float(np.min(pressure_mismatch)),
            "endpoint_pressure_direct": endpoint_pressure_mismatch <= 0.02,
            "endpoint_pressure_smooth": endpoint_pressure_mismatch <= 0.025,
            "any_pressure_direct": bool(np.any(pressure_mismatch <= 0.02)),
            "any_direct_handoff": bool(np.any(direct_mask)),
            "any_smooth_handoff": bool(np.any(smooth_mask)),
        })
    return rows


def summarize_perry_overlap(rows, label):
    print(f"{label}: n={len(rows)}")
    if not rows:
        return
    endpoint_errors = [row["endpoint_pressure_mismatch"] for row in rows]
    minimum_errors = [row["minimum_pressure_mismatch"] for row in rows]
    print(
        f"  endpoint pressure <=2%: "
        f"{sum(row['endpoint_pressure_direct'] for row in rows)}/{len(rows)}; "
        f"<=2.5%: {sum(row['endpoint_pressure_smooth'] for row in rows)}/{len(rows)}"
    )
    print(
        f"  endpoint |P mismatch| median={100*median(endpoint_errors):.3f}% "
        f"p90={100*percentile(endpoint_errors, .90):.3f}% "
        f"p95={100*percentile(endpoint_errors, .95):.3f}%"
    )
    print(
        f"  any overlap pressure <=2%: "
        f"{sum(row['any_pressure_direct'] for row in rows)}/{len(rows)}; "
        f"direct value+slope handoff: "
        f"{sum(row['any_direct_handoff'] for row in rows)}/{len(rows)}; "
        f"within smoothing envelope: "
        f"{sum(row['any_smooth_handoff'] for row in rows)}/{len(rows)}"
    )
    print(
        f"  minimum overlap |P mismatch| median={100*median(minimum_errors):.3f}% "
        f"p95={100*percentile(minimum_errors, .95):.3f}%"
    )


def summarize(rows, label):
    print(f"{label}: n={len(rows)}")
    if not rows:
        return
    errors = [row["relative_error"] for row in rows]
    absolute_errors = [abs(error) for error in errors]
    implied_delta_z = [
        row["implied_delta_z"]
        for row in rows
        if math.isfinite(row["implied_delta_z"])
    ]
    print(
        f"  Hvap |error| mean={100*mean(absolute_errors):.3f}% "
        f"median={100*median(absolute_errors):.3f}% "
        f"p90={100*percentile(absolute_errors, .90):.3f}% "
        f"p95={100*percentile(absolute_errors, .95):.3f}% "
        f"max={100*max(absolute_errors):.3f}%"
    )
    print(
        f"  signed median={100*median(errors):+.3f}% "
        f"within 5%={sum(error <= .05 for error in absolute_errors)}/{len(rows)} "
        f"within 10%={sum(error <= .10 for error in absolute_errors)}/{len(rows)}"
    )
    if implied_delta_z:
        print(
            f"  implied deltaZ median={median(implied_delta_z):.5f} "
            f"p05={percentile(implied_delta_z, .05):.5f} "
            f"p95={percentile(implied_delta_z, .95):.5f}"
        )


def summarize_corrected(rows, label, error_key, delta_z_key):
    usable = [
        row for row in rows
        if math.isfinite(row[error_key]) and math.isfinite(row[delta_z_key])
    ]
    print(f"{label}: n={len(usable)}")
    if not usable:
        return
    errors = [row[error_key] for row in usable]
    absolute_errors = [abs(error) for error in errors]
    delta_z = [row[delta_z_key] for row in usable]
    print(
        f"  corrected Hvap |error| mean={100*mean(absolute_errors):.3f}% "
        f"median={100*median(absolute_errors):.3f}% "
        f"p90={100*percentile(absolute_errors, .90):.3f}% "
        f"p95={100*percentile(absolute_errors, .95):.3f}%"
    )
    print(
        f"  signed median={100*median(errors):+.3f}% "
        f"within 5%={sum(error <= .05 for error in absolute_errors)}/{len(usable)} "
        f"within 10%={sum(error <= .10 for error in absolute_errors)}/{len(usable)} "
        f"deltaZ median={median(delta_z):.5f}"
    )


def main():
    rows, paired_components = load_rows()
    overlap_rows = load_perry_2_8_overlap_rows()
    print(f"PAIRED COMPONENTS: {len(paired_components)}")
    print(f"PAIRED METHOD-POINT ROWS: {len(rows)}")
    print("\nTABLE 2-10 VERSUS PERRY 2-8 OVERLAP")
    summarize_perry_overlap(overlap_rows, "ALL CHEMISTRY")
    summarize_perry_overlap(
        [row for row in overlap_rows if row["allowed"]],
        "ALLOWED CHEMISTRY",
    )
    for chemistry_label, chemistry_filter in (
        ("ALL CHEMISTRY", lambda row: True),
        ("ALLOWED CHEMISTRY", lambda row: row["allowed"]),
    ):
        print(f"\n{chemistry_label}")
        for method in DERIVATIVE_METHODS:
            subset = [
                row for row in rows
                if row["method"] == method and chemistry_filter(row)
            ]
            summarize(subset, method)
        print("\nHIGHEST TABLE POINT")
        for method in DERIVATIVE_METHODS:
            subset = [
                row for row in rows
                if (
                    row["method"] == method
                    and row["is_highest_point"]
                    and chemistry_filter(row)
                )
            ]
            summarize(subset, method)

    allowed_rows = [row for row in rows if row["allowed"]]
    print("\nALLOWED CHEMISTRY WITH THERMODYNAMIC deltaZ CORRECTION")
    for correction_label, error_key, delta_z_key in (
        ("virial/Rackett", "virial_relative_error", "virial_delta_z"),
        ("Patel-Teja-Valderrama", "ptv_relative_error", "ptv_delta_z"),
    ):
        print(f"\n{correction_label}: ALL PAIRED POINTS")
        for method in DERIVATIVE_METHODS:
            summarize_corrected(
                [row for row in allowed_rows if row["method"] == method],
                method,
                error_key,
                delta_z_key,
            )
        print(f"\n{correction_label}: HIGHEST TABLE POINT")
        for method in DERIVATIVE_METHODS:
            summarize_corrected(
                [
                    row for row in allowed_rows
                    if row["method"] == method and row["is_highest_point"]
                ],
                method,
                error_key,
                delta_z_key,
            )

    print("\nALLOWED CHEMISTRY BY PRESSURE BAND")
    for band in ("<=0.01 bar", "0.01-0.1 bar", "0.1-0.5 bar", ">0.5 bar"):
        print(f"\n{band}")
        for method in DERIVATIVE_METHODS:
            summarize(
                [
                    row for row in allowed_rows
                    if row["method"] == method and row["pressure_band"] == band
                ],
                method,
            )

    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
