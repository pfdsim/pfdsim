#!/usr/bin/env python3
"""Benchmark PBE0/aug-cc-pVDZ RRHO ideal-gas heat capacities.

Practicality note: an observed run completed only about 10 of 100 molecules
after one hour.  At roughly ten hours for this small benchmark, full
PBE0/aug-cc-pVDZ geometry optimization and Hessian evaluation is too expensive
for routine PFDsim property resolution; this script is retained only as an
offline research benchmark.

The deterministic sample and empirical reference curves are supplied by
``benchmark_xtb_rrho_cp.py``.  Each molecule starts from that benchmark's
RDKit conformer search and GFN2-xTB optimization, is reoptimized at
PBE0/aug-cc-pVDZ, and receives an analytic PBE0 Hessian.  PySCF's harmonic
analysis and rigid-rotor/harmonic-oscillator thermochemistry then provide Cp.

Rows are checkpointed after every molecule.  Re-running with the same output
path resumes completed CAS records instead of recomputing them.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS
from pyscf import dft, gto, lib
from pyscf.data import nist
from pyscf.hessian import thermo
from rdkit import RDLogger


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from property_resolution.dipole_moment import DipoleMomentMixin  # noqa: E402
from scripts.heat_capacity.gas.benchmark_xtb_rrho_cp import (  # noqa: E402
    DATABASE,
    TEMPERATURES,
    load_candidates,
    select_candidates,
)


DEFAULT_CSV = ROOT / "outputs" / "pbe0_rrho_cp_benchmark.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "pbe0_rrho_cp_benchmark_summary.json"
DEFAULT_LOG = ROOT / "outputs" / "pbe0_rrho_cp_benchmark.log"
HARTREE_PER_K_TO_J_MOL_K = nist.HARTREE2J * nist.AVOGADRO
FIELDNAMES = [
    "cas",
    "name",
    "formula",
    "source",
    "smiles",
    "heavy_atoms",
    "atoms",
    "linear",
    "imaginary_modes",
    "lowest_frequency_cm_1",
    "pbe0_optimization_seconds",
    "pbe0_hessian_seconds",
    *[
        field
        for temperature in TEMPERATURES
        for field in (
            f"reference_{temperature:g}_K",
            f"pbe0_rrho_{temperature:g}_K",
            f"error_{temperature:g}_K",
            f"ape_{temperature:g}_K",
        )
    ],
    "status",
    "error",
]


def build_molecule(atoms, charge: int, multiplicity: int):
    xyz = ";".join(
        f"{symbol} {x:.12f} {y:.12f} {z:.12f}"
        for symbol, (x, y, z) in zip(
            atoms.get_chemical_symbols(), atoms.positions, strict=True
        )
    )
    return gto.M(
        atom=xyz,
        unit="Angstrom",
        charge=charge,
        spin=multiplicity - 1,
        basis="aug-cc-pvdz",
        verbose=0,
    )


def run_pbe0(molecule, density_matrix=None):
    mean_field = (
        dft.RKS(molecule, xc="pbe0")
        if molecule.multiplicity == 1
        else dft.UKS(molecule, xc="pbe0")
    ).density_fit()
    mean_field.grids.level = 3
    mean_field.conv_tol = 1.0e-10
    mean_field.max_cycle = 100
    mean_field.kernel(dm0=density_matrix)
    if not mean_field.converged:
        raise RuntimeError("PySCF PBE0 SCF did not converge")
    return mean_field


class PySCFPBE0Calculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, charge: int, multiplicity: int):
        super().__init__()
        self.charge = charge
        self.multiplicity = multiplicity
        self.density_matrix = None
        self.mean_field = None

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        molecule = build_molecule(self.atoms, self.charge, self.multiplicity)
        mean_field = run_pbe0(molecule, self.density_matrix)
        gradient = np.asarray(mean_field.nuc_grad_method().kernel(), dtype=float)
        self.density_matrix = mean_field.make_rdm1()
        self.mean_field = mean_field
        self.results = {
            "energy": float(mean_field.e_tot * units.Hartree),
            "forces": -gradient * units.Hartree / units.Bohr,
        }


def selected_frequencies(mean_field) -> tuple[np.ndarray, np.ndarray, int]:
    hessian = mean_field.Hessian().kernel()
    analysis = thermo.harmonic_analysis(mean_field.mol, hessian)
    frequencies = np.asarray(analysis["freq_au"], dtype=complex)
    wavenumbers = np.asarray(analysis["freq_wavenumber"], dtype=complex)
    imaginary = int(np.count_nonzero(np.abs(frequencies.imag) > 0.0))
    return frequencies, wavenumbers, imaginary


def calculate(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            key: candidate[key]
            for key in ("cas", "name", "formula", "source", "smiles", "heavy_atoms")
        }
    )
    try:
        atoms, charge, multiplicity = DipoleMomentMixin._xtb_optimized_geometry(
            str(candidate["smiles"])
        )
        calculator = PySCFPBE0Calculator(charge, multiplicity)
        atoms.calc = calculator
        started = time.perf_counter()
        converged = BFGS(atoms, logfile=None).run(
            fmax=args.force_threshold,
            steps=args.optimization_steps,
        )
        row["pbe0_optimization_seconds"] = time.perf_counter() - started
        if not converged:
            raise RuntimeError("PBE0 geometry optimization did not converge")

        # Ensure the stored wavefunction corresponds exactly to ASE's final geometry.
        atoms.get_potential_energy()
        mean_field = calculator.mean_field
        started = time.perf_counter()
        frequencies, wavenumbers, imaginary_modes = selected_frequencies(mean_field)
        row["pbe0_hessian_seconds"] = time.perf_counter() - started
        significant_imaginary = np.abs(wavenumbers.imag) > args.imaginary_cutoff_cm_1
        if np.any(significant_imaginary):
            raise RuntimeError(
                f"optimized structure has {int(np.count_nonzero(significant_imaginary))} "
                f"imaginary mode(s) above {args.imaginary_cutoff_cm_1:g} cm^-1"
            )
        frequencies = np.abs(frequencies)
        row["atoms"] = len(atoms)
        row["linear"] = bool(thermo._get_rotor_type(  # noqa: SLF001
            thermo.rotation_const(
                mean_field.mol.atom_mass_list(isotope_avg=True),
                mean_field.mol.atom_coords()
                - np.average(
                    mean_field.mol.atom_coords(),
                    axis=0,
                    weights=mean_field.mol.atom_mass_list(isotope_avg=True),
                ),
                "GHz",
            )
        ) == "LINEAR")
        row["imaginary_modes"] = imaginary_modes
        row["lowest_frequency_cm_1"] = (
            float(np.min(np.abs(wavenumbers)))
            if len(frequencies)
            else ""
        )
        for temperature, reference in candidate["references"].items():
            rrho = thermo.thermo(
                mean_field,
                frequencies,
                temperature=temperature,
                pressure=100000.0,
            )
            prediction = float(rrho["Cp_tot"][0] * HARTREE_PER_K_TO_J_MOL_K)
            error = prediction - reference
            row[f"reference_{temperature:g}_K"] = reference
            row[f"pbe0_rrho_{temperature:g}_K"] = prediction
            row[f"error_{temperature:g}_K"] = error
            row[f"ape_{temperature:g}_K"] = 100.0 * abs(error / reference)
        row["status"] = "matched"
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def read_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(rows: Sequence[dict[str, Any]], temperature: float) -> dict[str, Any]:
    usable = [row for row in rows if row[f"ape_{temperature:g}_K"] != ""]
    if not usable:
        return {"n": 0}
    errors = np.asarray([float(row[f"error_{temperature:g}_K"]) for row in usable])
    percentages = np.asarray([float(row[f"ape_{temperature:g}_K"]) for row in usable])
    return {
        "n": len(usable),
        "mae_J_mol_K": float(np.mean(np.abs(errors))),
        "rmse_J_mol_K": float(np.sqrt(np.mean(errors**2))),
        "bias_J_mol_K": float(np.mean(errors)),
        "mape_percent": float(np.mean(percentages)),
        "median_ape_percent": float(np.median(percentages)),
        "maximum_ape_percent": float(np.max(percentages)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-heavy-atoms", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--force-threshold", type=float, default=0.01)
    parser.add_argument("--optimization-steps", type=int, default=100)
    parser.add_argument("--imaginary-cutoff-cm-1", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    lib.num_threads(args.threads)
    candidates = load_candidates(
        args.database,
        args.max_heavy_atoms,
        paired_psi4=False,
    )
    selected = select_candidates(candidates, args.limit, args.seed)
    print(
        f"eligible={len(candidates)} selected={len(selected)} seed={args.seed} "
        f"max_heavy_atoms={args.max_heavy_atoms} threads={args.threads}",
        flush=True,
    )
    if args.list_only:
        for row in selected:
            print(f"{row['cas']} | {row['name']} | {row['smiles']} | heavy={row['heavy_atoms']}")
        return

    rows = read_checkpoint(args.output)
    completed = {str(row["cas"]) for row in rows}
    overall_started = time.perf_counter()
    for index, candidate in enumerate(selected, 1):
        if candidate["cas"] in completed:
            print(f"[{index:03d}/{len(selected):03d}] {candidate['cas']}: checkpointed", flush=True)
            continue
        row = calculate(candidate, args)
        rows.append(row)
        write_csv(args.output, rows)
        print(
            f"[{index:03d}/{len(selected):03d}] {row['cas']} {row['name']}: "
            f"status={row['status']} Cp298={row['pbe0_rrho_298.15_K']} "
            f"APE298={row['ape_298.15_K']} error={row['error']}",
            flush=True,
        )

    selected_cas = {str(row["cas"]) for row in selected}
    selected_rows = [row for row in rows if str(row["cas"]) in selected_cas]
    summary = {
        "method": "PBE0/aug-cc-pVDZ optimized geometry, analytic Hessian, plain RRHO",
        "reference_scope": "canonical empirical ideal-gas Cp records",
        "eligible_population": len(candidates),
        "sample_size": len(selected),
        "seed": args.seed,
        "max_heavy_atoms": args.max_heavy_atoms,
        "threads": args.threads,
        "successful_molecules": sum(row["status"] == "matched" for row in selected_rows),
        "failed_molecules": sum(row["status"] != "matched" for row in selected_rows),
        "metrics": {
            f"{temperature:g}_K": metric_summary(selected_rows, temperature)
            for temperature in TEMPERATURES
        },
        "elapsed_seconds_this_invocation": time.perf_counter() - overall_started,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
