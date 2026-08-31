#!/usr/bin/env python3
"""Compare Rowlinson forms with canonical ordinary-liquid heat capacities."""

import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from property_resolution.ideal_gas_cp import load_bundled_kernel
from property_resolution.liquid_cp import load_bundled_liquid_kernel


R = 8.31446261815324


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


liquid_db = sqlite3.connect(ROOT / "data" / "liquid_heat_capacity.sqlite")
liquid_db.row_factory = sqlite3.Row
critical_db = sqlite3.connect(ROOT / "data" / "effective_criticals.sqlite")
critical_db.row_factory = sqlite3.Row
criticals = {
    row["CAS"]: row for row in critical_db.execute("SELECT * FROM effective_criticals")
}
liquid_rows = liquid_db.execute(
    "SELECT cas, name, formula, Tmin_fit_K, Tmax_fit_K "
    "FROM canonical_liquid_cp ORDER BY cas"
).fetchall()

names = ("Poling", "Bondi", "1.3x")
point_errors = {name: [] for name in names}
case_errors = {name: [] for name in names}
bin_names = ("0.30-0.50", "0.50-0.70", "0.70-0.85", "0.85-0.95")
bin_errors = {name: {label: [] for label in bin_names} for name in names}
case_metadata = {}

for row in liquid_rows:
    cas = row["cas"]
    critical = criticals.get(cas)
    gas = load_bundled_kernel(cas)
    if critical is None or gas is None:
        continue
    Tc = float(critical["Tc"])
    omega = float(critical["omega"])
    Tmin = max(float(row["Tmin_fit_K"]), gas.Tmin, 0.30 * Tc)
    Tmax = min(float(row["Tmax_fit_K"]), gas.Tmax, 0.95 * Tc)
    if Tmax <= Tmin:
        continue
    liquid = load_bundled_liquid_kernel(cas)
    errors = {name: [] for name in names}
    for index in range(25):
        T = Tmin + (Tmax - Tmin) * index / 24.0
        Tr = T / Tc
        target = liquid.cp(T)
        Cpig = gas.cp(T)
        one_minus_Tr = 1.0 - Tr
        predictions = {
            "Poling": Cpig + R * (
                1.586
                + 0.49 / one_minus_Tr
                + omega * (
                    4.2775
                    + 6.3 * one_minus_Tr ** (1.0 / 3.0) / Tr
                    + 0.4355 / one_minus_Tr
                )
            ),
            "Bondi": Cpig + R * (
                1.45
                + 0.45 / one_minus_Tr
                + 0.25 * omega * (
                    17.11
                    + 25.2 * one_minus_Tr ** (1.0 / 3.0) / Tr
                    + 1.742 / one_minus_Tr
                )
            ),
            "1.3x": 1.3 * Cpig,
        }
        if Tr < 0.50:
            label = "0.30-0.50"
        elif Tr < 0.70:
            label = "0.50-0.70"
        elif Tr < 0.85:
            label = "0.70-0.85"
        else:
            label = "0.85-0.95"
        for name, prediction in predictions.items():
            error = abs(prediction / target - 1.0) * 100.0
            point_errors[name].append(error)
            errors[name].append(error)
            bin_errors[name][label].append(error)
    for name in names:
        case_errors[name].append(statistics.fmean(errors[name]))
    case_metadata[cas] = (
        row["name"], row["formula"], Tmin / Tc, Tmax / Tc,
        {name: statistics.fmean(errors[name]) for name in names},
    )

print(f"Eligible compounds: {len(case_metadata)}; sampled points: {len(point_errors['Poling'])}")
for name in names:
    points = point_errors[name]
    cases = case_errors[name]
    print(
        f"{name}: point median={statistics.median(points):.3f}%, "
        f"mean={statistics.fmean(points):.3f}%, p95={percentile(points, .95):.3f}%; "
        f"case MARD median={statistics.median(cases):.3f}%, "
        f"mean={statistics.fmean(cases):.3f}%, p95={percentile(cases, .95):.3f}%"
    )

for label in bin_names:
    print(f"\nTr {label}; n={len(bin_errors['Poling'][label])}")
    for name in names:
        values = bin_errors[name][label]
        if values:
            print(
                f"  {name}: median={statistics.median(values):.3f}%, "
                f"mean={statistics.fmean(values):.3f}%, p95={percentile(values, .95):.3f}%"
            )

print("\nWorst Poling case MARD:")
ordered = sorted(
    case_metadata.items(), key=lambda item: item[1][4]["Poling"], reverse=True
)
for cas, (name, formula, Trmin, Trmax, errors) in ordered[:20]:
    print(
        f"{cas} {name[:28]:28} {formula:16} Tr={Trmin:.2f}-{Trmax:.2f} "
        f"Poling={errors['Poling']:.1f}% Bondi={errors['Bondi']:.1f}% "
        f"1.3x={errors['1.3x']:.1f}%"
    )
