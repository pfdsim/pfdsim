#!/usr/bin/env python3
"""Refit the two differing Rowlinson constants with grouped validation."""

import hashlib
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from chemicals.identifiers import search_chemical
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.optimize import minimize

from property_resolution.ideal_gas_cp import load_bundled_kernel
from property_resolution.liquid_cp import load_bundled_liquid_kernel


from physical_constants import R_J_MOL_K

R = R_J_MOL_K
HALOGENS = {9, 17, 35, 53}
STRICT_INORGANIC_CAS = {"74-90-8"}  # hydrogen cyanide
STRICT_ORGANIC_CAS = {"144-62-7"}  # oxalic acid


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


def structure(cas):
    try:
        smiles = str(search_chemical(cas).smiles or "").strip()
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
    except Exception:
        return None
    return molecule


def project_organic(molecule):
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        if atom.GetTotalNumHs() > 0:
            return True
        if any(neighbor.GetAtomicNum() in HALOGENS for neighbor in atom.GetNeighbors()):
            return True
    return False


def strict_class(cas, molecule):
    admitted = project_organic(molecule)
    if admitted is None:
        return "unresolved", None
    hbd = int(rdMolDescriptors.CalcNumHBD(molecule))
    organic = (admitted or cas in STRICT_ORGANIC_CAS) and cas not in STRICT_INORGANIC_CAS
    if not organic:
        return "inorganic", hbd
    return ("organic_hbd" if hbd > 0 else "organic_no_hbd"), hbd


def metrics(errors, case_ids):
    values = np.abs(np.asarray(errors, dtype=float)) * 100.0
    by_case = {}
    for value, cas in zip(values, case_ids):
        by_case.setdefault(cas, []).append(float(value))
    case_mards = [statistics.fmean(group) for group in by_case.values()]
    return {
        "point_median": float(np.median(values)),
        "point_mean": float(np.mean(values)),
        "point_p95": percentile(values.tolist(), 0.95),
        "case_median": statistics.median(case_mards),
        "case_mean": statistics.fmean(case_mards),
        "case_p95": percentile(case_mards, 0.95),
        "lt5": sum(value < 5.0 for value in case_mards) / len(case_mards),
        "lt10": sum(value < 10.0 for value in case_mards) / len(case_mards),
        "bias": float(np.mean(np.asarray(errors))) * 100.0,
    }


def print_metrics(label, errors, case_ids):
    result = metrics(errors, case_ids)
    print(
        f"{label:24} point median={result['point_median']:6.3f}% "
        f"mean={result['point_mean']:6.3f}% p95={result['point_p95']:6.3f}% | "
        f"case MARD median={result['case_median']:6.3f}% "
        f"mean={result['case_mean']:6.3f}% p95={result['case_p95']:6.3f}% | "
        f"bias={result['bias']:+6.3f}% <5%={result['lt5']:5.1%} <10%={result['lt10']:5.1%}"
    )


def fit_parameters(X, y, kind):
    if kind == "absolute_ls":
        return np.linalg.lstsq(X, y, rcond=None)[0]
    X_relative = X / target_train[:, None]
    y_relative = y / target_train
    initial = np.linalg.lstsq(X_relative, y_relative, rcond=None)[0]
    if kind == "relative_ls":
        return initial
    result = minimize(
        lambda parameters: float(np.mean(np.abs(X_relative @ parameters - y_relative))),
        initial,
        method="Nelder-Mead",
        options={"xatol": 1e-11, "fatol": 1e-12, "maxiter": 5000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


liquid_db = sqlite3.connect(ROOT / "data" / "liquid_heat_capacity.sqlite")
liquid_db.row_factory = sqlite3.Row
critical_db = sqlite3.connect(ROOT / "data" / "effective_criticals.sqlite")
critical_db.row_factory = sqlite3.Row
criticals = {row["CAS"]: row for row in critical_db.execute("SELECT * FROM effective_criticals")}
rows = liquid_db.execute(
    "SELECT cas, name, Tmin_fit_K, Tmax_fit_K FROM canonical_liquid_cp ORDER BY cas"
).fetchall()

records = []
for row in rows:
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
    molecule = structure(cas)
    category, hbd = strict_class(cas, molecule)
    liquid = load_bundled_liquid_kernel(cas)
    points = []
    for index in range(25):
        T = Tmin + (Tmax - Tmin) * index / 24.0
        Tr = T / Tc
        one_minus_Tr = 1.0 - Tr
        Cpig = gas.cp(T)
        target = liquid.cp(T)
        omega_term = omega * (
            4.2775
            + 6.3 * one_minus_Tr ** (1.0 / 3.0) / Tr
            + 0.4355 / one_minus_Tr
        )
        base = Cpig + R * omega_term
        points.append((R, R / one_minus_Tr, base, target))
    records.append({"cas": cas, "name": row["name"], "category": category, "points": points})

for category in ("organic_no_hbd", "organic_hbd", "inorganic", "unresolved"):
    subset = [record for record in records if record["category"] == category]
    print(f"{category}: compounds={len(subset)}, points={sum(len(r['points']) for r in subset)}")

eligible = [record for record in records if record["category"] == "organic_no_hbd"]
case_ids = np.asarray([record["cas"] for record in eligible for _ in record["points"]])
point_rows = [point for record in eligible for point in record["points"]]
X_all = np.asarray([[point[0], point[1]] for point in point_rows])
base_all = np.asarray([point[2] for point in point_rows])
target_all = np.asarray([point[3] for point in point_rows])
y_all = target_all - base_all

print("\nFixed forms on strict-organic, no-HBD population:")
for label, parameters in (("Poling (1.586, 0.49)", np.array([1.586, 0.49])), ("Bondi (1.45, 0.45)", np.array([1.45, 0.45]))):
    relative_errors = (base_all + X_all @ parameters) / target_all - 1.0
    print_metrics(label, relative_errors, case_ids)

print("\nFull-data fitted coefficients (descriptive only):")
full_coefficients = {}
for kind in ("absolute_ls", "relative_ls", "relative_lad"):
    target_train = target_all
    parameters = fit_parameters(X_all, y_all, kind)
    full_coefficients[kind] = parameters
    relative_errors = (base_all + X_all @ parameters) / target_all - 1.0
    print(f"{kind:24} a={parameters[0]:.9f} b={parameters[1]:.9f}")
    print_metrics("  in-sample", relative_errors, case_ids)

print("\n10-fold CAS-grouped out-of-fold results:")
fold_by_cas = {
    record["cas"]: int(hashlib.sha256(record["cas"].encode()).hexdigest()[:8], 16) % 10
    for record in eligible
}
for kind in ("absolute_ls", "relative_ls", "relative_lad"):
    out_of_fold = np.empty(len(target_all))
    fold_coefficients = []
    for fold in range(10):
        test = np.asarray([fold_by_cas[cas] == fold for cas in case_ids])
        train = ~test
        X_train = X_all[train]
        y_train = y_all[train]
        target_train = target_all[train]
        parameters = fit_parameters(X_train, y_train, kind)
        fold_coefficients.append(parameters)
        out_of_fold[test] = (base_all[test] + X_all[test] @ parameters) / target_all[test] - 1.0
    coefficients = np.asarray(fold_coefficients)
    print(
        f"{kind:24} a mean={np.mean(coefficients[:, 0]):.6f} "
        f"range={np.min(coefficients[:, 0]):.6f}..{np.max(coefficients[:, 0]):.6f}; "
        f"b mean={np.mean(coefficients[:, 1]):.6f} "
        f"range={np.min(coefficients[:, 1]):.6f}..{np.max(coefficients[:, 1]):.6f}"
    )
    print_metrics("  out-of-fold", out_of_fold, case_ids)

print("\nFixed forms by strict category (all inorganics combined):")
for category in ("organic_hbd", "inorganic", "unresolved"):
    subset = [record for record in records if record["category"] == category]
    ids = np.asarray([record["cas"] for record in subset for _ in record["points"]])
    points = [point for record in subset for point in record["points"]]
    X = np.asarray([[point[0], point[1]] for point in points])
    base = np.asarray([point[2] for point in points])
    target = np.asarray([point[3] for point in points])
    print(f"  {category}:")
    for label, parameters in (("Poling", np.array([1.586, 0.49])), ("Bondi", np.array([1.45, 0.45]))):
        print_metrics("    " + label, (base + X @ parameters) / target - 1.0, ids)
