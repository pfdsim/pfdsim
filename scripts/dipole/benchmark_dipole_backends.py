#!/usr/bin/env python3
"""Benchmark GFN2-xTB and PySCF DFT dipoles against chemicals/CCCBDB."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.units import Bohr
from chemicals.identifiers import search_chemical
import chemicals.dipole as chemicals_dipole
from pyscf import dft, gto, lib
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tblite.ase import TBLite
from tblite.interface import Calculator


AU_DIPOLE_TO_DEBYE = 2.541746473
ALLOWED_ELEMENTS = {"H", "C", "N", "O", "F", "Si", "P", "S", "Cl"}
FIELDNAMES = [
    "cas",
    "name",
    "smiles",
    "heavy_atoms",
    "experimental_D",
    "gfn2_xtb_D",
    "pbe0_aug_cc_pvdz_D",
    "gfn2_error_D",
    "pbe0_error_D",
    "xtb_opt_converged",
    "pyscf_converged",
    "xtb_seconds",
    "pyscf_seconds",
    "error",
]


def eligible_records() -> list[dict[str, object]]:
    chemicals_dipole._load_dipole_data()
    source = chemicals_dipole.dipole_sources["CCCBDB"]
    records = []
    for cas, row in source.iterrows():
        try:
            metadata = search_chemical(cas)
            molecule = Chem.MolFromSmiles(metadata.smiles)
        except Exception:
            continue
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            continue
        if Chem.GetFormalCharge(molecule) != 0:
            continue
        if not any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()):
            continue
        if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
            continue
        elements = {atom.GetSymbol() for atom in molecule.GetAtoms()}
        if not elements <= ALLOWED_ELEMENTS or molecule.GetNumHeavyAtoms() > 8:
            continue
        records.append(
            {
                "cas": cas,
                "name": row["Chemical"],
                "smiles": metadata.smiles,
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "experimental_D": float(row["dipole_moment"]),
            }
        )
    return records


def select_records(records: list[dict[str, object]], limit: int, seed: int) -> list[dict[str, object]]:
    """Sample across experimental-dipole ranges instead of favoring one range."""
    bins = [[], [], [], [], []]
    for record in records:
        value = float(record["experimental_D"])
        index = 0 if value == 0 else 1 if value < 0.75 else 2 if value < 1.75 else 3 if value < 2.75 else 4
        bins[index].append(record)

    rng = random.Random(seed)
    selected = []
    per_bin = limit // len(bins)
    remainder = limit % len(bins)
    for index, candidates in enumerate(bins):
        count = min(len(candidates), per_bin + (index < remainder))
        selected.extend(rng.sample(candidates, count))

    if len(selected) < limit:
        chosen = {str(record["cas"]) for record in selected}
        remainder_pool = [record for record in records if str(record["cas"]) not in chosen]
        selected.extend(rng.sample(remainder_pool, min(limit - len(selected), len(remainder_pool))))
    return sorted(selected, key=lambda record: str(record["cas"]))


def initial_geometry(smiles: str, cas: str, conformers: int) -> Atoms:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(hashlib.sha256(cas.encode()).hexdigest()[:7], 16)
    params.pruneRmsThresh = 0.25
    conformer_ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=conformers, params=params))
    if not conformer_ids:
        raise RuntimeError("RDKit could not embed a conformer")

    energies: list[tuple[float, int]] = []
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        properties = AllChem.MMFFGetMoleculeProperties(molecule)
        for conformer_id in conformer_ids:
            forcefield = AllChem.MMFFGetMoleculeForceField(molecule, properties, confId=conformer_id)
            forcefield.Minimize(maxIts=500)
            energies.append((forcefield.CalcEnergy(), conformer_id))
    else:
        for conformer_id in conformer_ids:
            forcefield = AllChem.UFFGetMoleculeForceField(molecule, confId=conformer_id)
            forcefield.Minimize(maxIts=500)
            energies.append((forcefield.CalcEnergy(), conformer_id))

    conformer = molecule.GetConformer(min(energies)[1])
    return Atoms(
        [atom.GetSymbol() for atom in molecule.GetAtoms()],
        positions=np.asarray(conformer.GetPositions()),
    )


def calculate(record: dict[str, object], conformers: int) -> dict[str, object]:
    result = dict(record)
    result.update({field: "" for field in FIELDNAMES if field not in result})
    try:
        atoms = initial_geometry(str(record["smiles"]), str(record["cas"]), conformers)

        started = time.perf_counter()
        atoms.calc = TBLite(method="GFN2-xTB", charge=0, multiplicity=1, verbosity=0)
        result["xtb_opt_converged"] = bool(BFGS(atoms, logfile=None).run(fmax=0.01, steps=300))
        raw = Calculator("GFN2-xTB", atoms.numbers, atoms.positions / Bohr).singlepoint()
        xtb_D = float(np.linalg.norm(raw.get("dipole")) * AU_DIPOLE_TO_DEBYE)
        result["xtb_seconds"] = time.perf_counter() - started
        result["gfn2_xtb_D"] = xtb_D
        result["gfn2_error_D"] = xtb_D - float(record["experimental_D"])

        started = time.perf_counter()
        xyz = ";".join(
            f"{symbol} {x:.12f} {y:.12f} {z:.12f}"
            for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions, strict=True)
        )
        molecule = gto.M(
            atom=xyz,
            unit="Angstrom",
            charge=0,
            spin=0,
            basis="aug-cc-pvdz",
            verbose=0,
        )
        mean_field = dft.RKS(molecule, xc="pbe0").density_fit()
        mean_field.grids.level = 3
        mean_field.conv_tol = 1e-10
        mean_field.max_cycle = 100
        mean_field.kernel()
        result["pyscf_converged"] = bool(mean_field.converged)
        if not mean_field.converged:
            raise RuntimeError("PySCF SCF did not converge")
        dft_D = float(np.linalg.norm(mean_field.dip_moment(unit="Debye", verbose=0)))
        result["pyscf_seconds"] = time.perf_counter() - started
        result["pbe0_aug_cc_pvdz_D"] = dft_D
        result["pbe0_error_D"] = dft_D - float(record["experimental_D"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def metrics(
    rows: list[dict[str, object]],
    prediction: str,
) -> dict[str, float | int | None]:
    pairs = [
        (float(row["experimental_D"]), float(row[prediction]))
        for row in rows
        if row[prediction] != ""
    ]
    if not pairs:
        return {
            "n": 0,
            "mae_D": None,
            "rmse_D": None,
            "bias_D": None,
            "median_ae_D": None,
            "max_ae_D": None,
            "r2": None,
        }

    experimental = np.asarray([pair[0] for pair in pairs])
    predicted = np.asarray([pair[1] for pair in pairs])
    errors = predicted - experimental
    return {
        "n": len(pairs),
        "mae_D": float(np.mean(np.abs(errors))),
        "rmse_D": float(np.sqrt(np.mean(errors**2))),
        "bias_D": float(np.mean(errors)),
        "median_ae_D": float(np.median(np.abs(errors))),
        "max_ae_D": float(np.max(np.abs(errors))),
        "r2": (
            float(np.corrcoef(experimental, predicted)[0, 1] ** 2)
            if len(pairs) >= 2
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--conformers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("dipole_benchmark_results.csv"))
    parser.add_argument("--summary", type=Path, default=Path("dipole_benchmark_summary.json"))
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    lib.num_threads(args.threads)
    records = select_records(eligible_records(), args.limit, args.seed)
    rows = []
    overall_started = time.perf_counter()
    for index, record in enumerate(records, 1):
        row = calculate(record, args.conformers)
        rows.append(row)
        write_results(args.output, rows)
        print(
            f"[{index:02d}/{len(records):02d}] {row['cas']} {row['name']}: "
            f"exp={row['experimental_D']} xTB={row['gfn2_xtb_D']} "
            f"DFT={row['pbe0_aug_cc_pvdz_D']} error={row['error']}",
            flush=True,
        )

    summary = {
        "eligible_population": len(eligible_records()),
        "sample_size": len(records),
        "seed": args.seed,
        "conformers_requested": args.conformers,
        "geometry": "lowest MMFF/UFF conformer, then GFN2-xTB optimization",
        "gfn2_xtb": metrics(rows, "gfn2_xtb_D"),
        "pyscf_pbe0_aug_cc_pvdz": metrics(rows, "pbe0_aug_cc_pvdz_D"),
        "failures": sum(bool(row["error"]) for row in rows),
        "elapsed_seconds": time.perf_counter() - overall_started,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
