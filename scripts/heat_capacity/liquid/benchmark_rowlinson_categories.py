#!/usr/bin/env python3
"""Partition Rowlinson errors by organic, inorganic, and HBD classes."""

import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemicals.identifiers import search_chemical
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from property_resolution.ideal_gas_cp import load_bundled_kernel
from property_resolution.liquid_cp import load_bundled_liquid_kernel


R = 8.31446261815324
HALOGENS = {9, 17, 35, 53}


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


def structure(cas):
    try:
        smiles = str(search_chemical(cas).smiles or "").strip()
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
    except Exception:
        return "", None
    return smiles, molecule


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


def summarize(records, label):
    print(f"\n{label}: compounds={len(records)}, points={sum(len(r['errors']['Poling']) for r in records)}")
    if not records:
        return
    for model in ("Poling", "Bondi", "1.3x"):
        points = [error for record in records for error in record["errors"][model]]
        signed = [error for record in records for error in record["signed"][model]]
        cases = [statistics.fmean(record["errors"][model]) for record in records]
        print(
            f"  {model:7}: median={statistics.median(points):6.3f}% "
            f"mean={statistics.fmean(points):6.3f}% p95={percentile(points, .95):6.3f}% "
            f"case-MARD median={statistics.median(cases):6.3f}% "
            f"mean={statistics.fmean(cases):6.3f}% p95={percentile(cases, .95):6.3f}% "
            f"bias={statistics.fmean(signed):+6.3f}% "
            f"cases<5%={sum(x < 5 for x in cases) / len(cases):5.1%} "
            f"cases<10%={sum(x < 10 for x in cases) / len(cases):5.1%}"
        )


liquid_db = sqlite3.connect(ROOT / "data" / "liquid_heat_capacity.sqlite")
liquid_db.row_factory = sqlite3.Row
critical_db = sqlite3.connect(ROOT / "data" / "effective_criticals.sqlite")
critical_db.row_factory = sqlite3.Row
criticals = {
    row["CAS"]: row for row in critical_db.execute("SELECT * FROM effective_criticals")
}
rows = liquid_db.execute(
    "SELECT cas, name, formula, Tmin_fit_K, Tmax_fit_K "
    "FROM canonical_liquid_cp ORDER BY cas"
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
        hbd = None
        organic = None
    else:
        hbd = int(rdMolDescriptors.CalcNumHBD(molecule))
        organic = project_organic(molecule)
    liquid = load_bundled_liquid_kernel(cas)
    errors = {name: [] for name in ("Poling", "Bondi", "1.3x")}
    signed = {name: [] for name in errors}
    for index in range(25):
        T = Tmin + (Tmax - Tmin) * index / 24.0
        Tr = T / Tc
        target = liquid.cp(T)
        Cpig = gas.cp(T)
        one_minus_Tr = 1.0 - Tr
        predictions = {
            "Poling": Cpig + R * (
                1.586 + 0.49 / one_minus_Tr
                + omega * (4.2775 + 6.3 * one_minus_Tr ** (1.0 / 3.0) / Tr + 0.4355 / one_minus_Tr)
            ),
            "Bondi": Cpig + R * (
                1.45 + 0.45 / one_minus_Tr
                + 0.25 * omega * (17.11 + 25.2 * one_minus_Tr ** (1.0 / 3.0) / Tr + 1.742 / one_minus_Tr)
            ),
            "1.3x": 1.3 * Cpig,
        }
        for model, prediction in predictions.items():
            relative = (prediction / target - 1.0) * 100.0
            signed[model].append(relative)
            errors[model].append(abs(relative))
    records.append({
        "cas": cas,
        "name": row["name"],
        "formula": row["formula"],
        "smiles": smiles,
        "hbd": hbd,
        "organic": organic,
        "errors": errors,
        "signed": signed,
    })

print(f"Eligible overlap: {len(records)} compounds")
summarize([r for r in records if r["organic"] and r["hbd"] == 0], "Project-organic, no HBD")
summarize([r for r in records if r["organic"] and r["hbd"] and r["hbd"] > 0], "Project-organic, HBD")
summarize([r for r in records if r["organic"] is False and r["hbd"] == 0], "Project-inorganic, no HBD")
summarize([r for r in records if r["organic"] is False and r["hbd"] and r["hbd"] > 0], "Project-inorganic, HBD")
summarize([r for r in records if r["organic"] is None or r["hbd"] is None], "Unresolved structure")

# Conventional chemistry taxonomy differs from pfdsim's deliberately narrow
# C-H/C-halogen admission rule in two identities in this exact overlap.
strict_inorganic_cas = {"74-90-8"}  # hydrogen cyanide
strict_organic_cas = {"144-62-7"}  # oxalic acid
strict_organic = [
    r for r in records
    if (r["organic"] or r["cas"] in strict_organic_cas)
    and r["cas"] not in strict_inorganic_cas
]
strict_inorganic = [
    r for r in records
    if (r["organic"] is False and r["cas"] not in strict_organic_cas)
    or r["cas"] in strict_inorganic_cas
]
summarize([r for r in strict_organic if r["hbd"] == 0], "Strict-organic, no HBD")
summarize([r for r in strict_organic if r["hbd"] and r["hbd"] > 0], "Strict-organic, HBD")
summarize([r for r in strict_inorganic if r["hbd"] == 0], "Strict-inorganic, no HBD")
summarize([r for r in strict_inorganic if r["hbd"] and r["hbd"] > 0], "Strict-inorganic, HBD")

for label, subset in (
    ("Worst project-organic no-HBD", [r for r in records if r["organic"] and r["hbd"] == 0]),
    ("Worst project-organic HBD", [r for r in records if r["organic"] and r["hbd"] and r["hbd"] > 0]),
    ("Worst project-inorganic", [r for r in records if r["organic"] is False]),
):
    print(f"\n{label} Poling case MARD:")
    ordered = sorted(subset, key=lambda r: statistics.fmean(r["errors"]["Poling"]), reverse=True)
    for record in ordered[:15]:
        print(
            f"  {record['cas']:12} {record['name'][:30]:30} HBD={record['hbd']} "
            f"Poling={statistics.fmean(record['errors']['Poling']):6.2f}% "
            f"Bondi={statistics.fmean(record['errors']['Bondi']):6.2f}%"
        )

print("\nProject-organic candidates for strict inorganic reclassification (<=2 carbons, no C-C bond):")
for record in records:
    molecule = Chem.MolFromSmiles(record["smiles"]) if record["smiles"] else None
    if not record["organic"] or molecule is None:
        continue
    carbons = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6]
    carbon_carbon = any(
        bond.GetBeginAtom().GetAtomicNum() == 6 and bond.GetEndAtom().GetAtomicNum() == 6
        for bond in molecule.GetBonds()
    )
    if len(carbons) <= 2 and not carbon_carbon:
        mard = statistics.fmean(record["errors"]["Poling"])
        print(
            f"  {record['cas']:12} {record['name'][:30]:30} {record['formula']:12} "
            f"HBD={record['hbd']} MARD={mard:6.2f}% smiles={record['smiles']}"
        )

print("\nAll project-inorganic identities:")
for record in records:
    if record["organic"] is False:
        print(
            f"  {record['cas']:12} {record['name'][:30]:30} {record['formula']:12} "
            f"HBD={record['hbd']} smiles={record['smiles']}"
        )
