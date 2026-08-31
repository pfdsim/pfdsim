#!/usr/bin/env python3
"""Analyze Rowlinson HBD residuals by donor class and molecular size."""

import os
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from chemicals.identifiers import search_chemical
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, Descriptors, rdMolDescriptors
from scipy.stats import spearmanr

from property_resolution.ideal_gas_cp import load_bundled_kernel
from property_resolution.liquid_cp import load_bundled_liquid_kernel


R = 8.31446261815324
HALOGENS = {9, 17, 35, 53}
STRICT_INORGANIC_CAS = {"74-90-8"}
STRICT_ORGANIC_CAS = {"144-62-7"}
FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)


def structure(cas):
    try:
        smiles = str(search_chemical(cas).smiles or "").strip()
        return smiles, Chem.MolFromSmiles(smiles) if smiles else None
    except Exception:
        return "", None


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


def donor_atom_class(atom):
    atomic_number = atom.GetAtomicNum()
    if atomic_number == 8:
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIsAromatic():
                return "phenol"
            if neighbor.GetAtomicNum() == 6 and any(
                bond.GetBondTypeAsDouble() >= 1.9
                and bond.GetOtherAtom(neighbor).GetAtomicNum() in (8, 16)
                for bond in neighbor.GetBonds()
                if bond.GetOtherAtom(neighbor).GetIdx() != atom.GetIdx()
            ):
                return "carboxylic_acid"
        return "alcohol_OH"
    if atomic_number == 7:
        if atom.GetIsAromatic():
            return "aromatic_NH"
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() in (6, 15, 16) and any(
                bond.GetBondTypeAsDouble() >= 1.9
                and bond.GetOtherAtom(neighbor).GetAtomicNum() in (7, 8, 16)
                for bond in neighbor.GetBonds()
                if bond.GetOtherAtom(neighbor).GetIdx() != atom.GetIdx()
            ):
                return "amide_like_NH"
        return "amine_NH"
    if atomic_number == 16:
        return "thiol_SH"
    return f"other_{atom.GetSymbol()}H"


def donor_signature(molecule):
    donor_atom_ids = []
    for feature in FEATURE_FACTORY.GetFeaturesForMol(molecule):
        if feature.GetFamily() == "Donor":
            donor_atom_ids.extend(feature.GetAtomIds())
    classes = [donor_atom_class(molecule.GetAtomWithIdx(index)) for index in sorted(set(donor_atom_ids))]
    unique = sorted(set(classes))
    return "+".join(unique), classes


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


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
    smiles, molecule = structure(cas)
    if molecule is None:
        continue
    admitted = project_organic(molecule)
    organic = (admitted or cas in STRICT_ORGANIC_CAS) and cas not in STRICT_INORGANIC_CAS
    hbd = int(rdMolDescriptors.CalcNumHBD(molecule))
    if not organic or hbd <= 0:
        continue
    hba = int(rdMolDescriptors.CalcNumHBA(molecule))
    heavy = int(molecule.GetNumHeavyAtoms())
    hetero = sum(atom.GetAtomicNum() not in (1, 6) for atom in molecule.GetAtoms())
    carbons = sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
    signature, donor_classes = donor_signature(molecule)
    liquid = load_bundled_liquid_kernel(cas)
    signed = {"Poling": [], "Bondi": []}
    reduced_temperatures = []
    cp_ratios = []
    for index in range(25):
        T = Tmin + (Tmax - Tmin) * index / 24.0
        Tr = T / Tc
        Cpig = gas.cp(T)
        target = liquid.cp(T)
        cp_ratios.append(Cpig / target)
        u = 1.0 - Tr
        omega_term = omega * (4.2775 + 6.3 * u ** (1.0 / 3.0) / Tr + 0.4355 / u)
        for model, a, b in (("Poling", 1.586, 0.49), ("Bondi", 1.45, 0.45)):
            prediction = Cpig + R * (a + b / u + omega_term)
            signed[model].append((prediction / target - 1.0) * 100.0)
        reduced_temperatures.append(Tr)
    record = {
        "cas": cas,
        "name": row["name"],
        "smiles": smiles,
        "class": signature,
        "hbd": hbd,
        "hba": hba,
        "heavy": heavy,
        "hetero": hetero,
        "carbons": carbons,
        "mw": float(Descriptors.MolWt(molecule)),
        "mean_Tr": statistics.fmean(reduced_temperatures),
        "ratio_mean": statistics.fmean(cp_ratios),
        "ratio_median": statistics.median(cp_ratios),
        "ratio_slope": float(np.polyfit(reduced_temperatures, cp_ratios, 1)[0]),
    }
    for model in signed:
        record[model + "_bias"] = statistics.fmean(signed[model])
        record[model + "_mard"] = statistics.fmean(abs(value) for value in signed[model])
        record[model + "_slope"] = float(np.polyfit(reduced_temperatures, signed[model], 1)[0])
    records.append(record)

print(f"Strict-organic HBD records: {len(records)}")
print("\nDonor-class compound-level results:")
classes = sorted({record["class"] for record in records})
for donor_class in classes:
    subset = [record for record in records if record["class"] == donor_class]
    if len(subset) < 2:
        continue
    print(f"{donor_class:34} n={len(subset):2}")
    for model in ("Poling", "Bondi"):
        biases = [record[model + "_bias"] for record in subset]
        mards = [record[model + "_mard"] for record in subset]
        print(
            f"  {model:7} bias mean={statistics.fmean(biases):+7.2f}% "
            f"median={statistics.median(biases):+7.2f}% "
            f"MARD mean={statistics.fmean(mards):6.2f}% "
            f"median={statistics.median(mards):6.2f}% "
            f"underpredict={sum(value < 0 for value in biases) / len(biases):5.1%}"
        )

descriptors = {
    "HBD count": lambda r: r["hbd"],
    "heavy atoms": lambda r: r["heavy"],
    "heavy atoms/HBD": lambda r: r["heavy"] / r["hbd"],
    "HBD/heavy atoms": lambda r: r["hbd"] / r["heavy"],
    "MW/HBD": lambda r: r["mw"] / r["hbd"],
    "hetero/heavy atoms": lambda r: r["hetero"] / r["heavy"],
    "HBA/HBD": lambda r: r["hba"] / r["hbd"],
    "mean Tr": lambda r: r["mean_Tr"],
}
print("\nSpearman correlations across all HBD organics:")
for descriptor, accessor in descriptors.items():
    x = [accessor(record) for record in records]
    if len(set(x)) < 2:
        print(f"{descriptor:20} skipped (constant descriptor)")
        continue
    fields = []
    for model in ("Poling", "Bondi"):
        for measure in ("bias", "mard"):
            y = [record[f"{model}_{measure}"] for record in records]
            rho, pvalue = spearmanr(x, y)
            fields.append(f"{model[0]}-{measure[0]} rho={rho:+.3f} p={pvalue:.3g}")
    print(f"{descriptor:20} " + "; ".join(fields))

print("\nWithin-class size correlations (Bondi bias vs heavy atoms/HBD):")
for donor_class in classes:
    subset = [record for record in records if record["class"] == donor_class]
    if len(subset) < 6:
        continue
    x = [record["heavy"] / record["hbd"] for record in subset]
    y = [record["Bondi_bias"] for record in subset]
    rho, pvalue = spearmanr(x, y)
    print(f"{donor_class:34} n={len(subset):2} rho={rho:+.3f} p={pvalue:.3g}")

# Explain variance in case-mean signed error using donor class, molecular scale,
# donor density, and mean reduced temperature. Report leave-one-out predictions
# to avoid presenting the descriptive R2 as predictive performance.
common_classes = [name for name in classes if sum(r["class"] == name for r in records) >= 3]
reference_class = max(common_classes, key=lambda name: sum(r["class"] == name for r in records))
def design(record):
    values = [
        1.0,
        np.log(record["heavy"] / record["hbd"]),
        record["hbd"] / record["heavy"],
        record["mean_Tr"],
    ]
    values.extend(1.0 if record["class"] == name else 0.0 for name in common_classes if name != reference_class)
    return values
X = np.asarray([design(record) for record in records])
print(f"\nLinear class/size model; reference donor class={reference_class}:")
for model in ("Poling", "Bondi"):
    y = np.asarray([record[model + "_bias"] for record in records])
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ coefficients
    r2 = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - np.mean(y)) ** 2)
    loo = np.empty(len(records))
    for index in range(len(records)):
        train = np.arange(len(records)) != index
        beta = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
        loo[index] = X[index] @ beta
    loo_r2 = 1.0 - np.sum((y - loo) ** 2) / np.sum((y - np.mean(y)) ** 2)
    print(
        f"  {model}: descriptive R2={r2:.3f}; leave-one-compound-out R2={loo_r2:.3f}; "
        f"MAE={np.mean(np.abs(y-loo)):.2f}%"
    )

print("\nMost negative and positive Bondi case biases:")
for label, ordered in (
    ("underprediction", sorted(records, key=lambda r: r["Bondi_bias"])),
    ("overprediction", sorted(records, key=lambda r: r["Bondi_bias"], reverse=True)),
):
    print(label + ":")
    for record in ordered[:15]:
        print(
            f"  {record['cas']:12} {record['name'][:27]:27} {record['class']:24} "
            f"HBD/heavy={record['hbd']}/{record['heavy']} "
            f"bias={record['Bondi_bias']:+7.2f}% MARD={record['Bondi_mard']:6.2f}% "
            f"Tr={record['mean_Tr']:.3f}"
        )

print("\nCpig/Cpl ratio by donor class:")
for donor_class in classes:
    subset = [record for record in records if record["class"] == donor_class]
    if len(subset) < 2:
        continue
    ratios = [record["ratio_mean"] for record in subset]
    slopes = [record["ratio_slope"] for record in subset]
    print(
        f"{donor_class:34} n={len(subset):2} "
        f"ratio mean={statistics.fmean(ratios):.4f} median={statistics.median(ratios):.4f} "
        f"range={min(ratios):.4f}..{max(ratios):.4f}; "
        f"d(ratio)/dTr mean={statistics.fmean(slopes):+.4f} "
        f"median={statistics.median(slopes):+.4f}"
    )

ratio_descriptors = {
    "HBD count": lambda r: r["hbd"],
    "heavy atoms": lambda r: r["heavy"],
    "heavy atoms/HBD": lambda r: r["heavy"] / r["hbd"],
    "HBD/heavy atoms": lambda r: r["hbd"] / r["heavy"],
    "MW/HBD": lambda r: r["mw"] / r["hbd"],
    "hetero/heavy atoms": lambda r: r["hetero"] / r["heavy"],
    "mean Tr": lambda r: r["mean_Tr"],
    "Bondi bias": lambda r: r["Bondi_bias"],
    "Bondi MARD": lambda r: r["Bondi_mard"],
}
print("\nSpearman correlations with compound-mean Cpig/Cpl:")
for label, accessor in ratio_descriptors.items():
    rho, pvalue = spearmanr(
        [accessor(record) for record in records],
        [record["ratio_mean"] for record in records],
    )
    print(f"{label:20} rho={rho:+.3f} p={pvalue:.3g}")

print("\nWithin-class Cpig/Cpl vs heavy atoms/HBD:")
for donor_class in classes:
    subset = [record for record in records if record["class"] == donor_class]
    if len(subset) < 6:
        continue
    rho, pvalue = spearmanr(
        [record["heavy"] / record["hbd"] for record in subset],
        [record["ratio_mean"] for record in subset],
    )
    print(f"{donor_class:34} n={len(subset):2} rho={rho:+.3f} p={pvalue:.3g}")

print("\nLowest and highest compound-mean Cpig/Cpl:")
for label, ordered in (
    ("lowest", sorted(records, key=lambda r: r["ratio_mean"])),
    ("highest", sorted(records, key=lambda r: r["ratio_mean"], reverse=True)),
):
    print(label + ":")
    for record in ordered[:12]:
        print(
            f"  {record['cas']:12} {record['name'][:27]:27} {record['class']:24} "
            f"HBD/heavy={record['hbd']}/{record['heavy']} ratio={record['ratio_mean']:.4f} "
            f"slope={record['ratio_slope']:+.4f} bias={record['Bondi_bias']:+7.2f}%"
        )
