#!/usr/bin/env python3
"""Probe GFN2-xTB harmonic-RRHO ideal-gas heat capacities.

References come from the canonical ideal-gas heat-capacity database used by
the established fallback benchmarks.  Computational ``psi4_adjusted`` rows
are excluded, matching the empirical-reference policy of
``benchmark_all_local_cp_298k.py``.

The probe selects a deterministic sample across molecular-size buckets,
optimizes the lowest of ten RDKit MMFF/UFF conformers with GFN2-xTB, obtains a
numerical Hessian from central finite differences of tblite forces, and
compares plain harmonic-RRHO Cp at 298.15, 400, 500, 600, 800, 1000, and
1500 K.  This is an expensive probe: the default sample requires about 6N
xTB force evaluations per molecule after optimization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import units
from ase.vibrations import Vibrations
from chemicals.elements import simple_formula_parser
from chemicals.identifiers import search_chemical
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physical_constants import R_J_MOL_K  # noqa: E402
from property_resolution.dipole_moment import DipoleMomentMixin  # noqa: E402


DATABASE = ROOT / "data" / "ideal_gas_heat_capacity.sqlite"
DEFAULT_CSV = ROOT / "outputs" / "xtb_rrho_cp_probe.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "xtb_rrho_cp_probe_summary.json"
TEMPERATURES = (298.15, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0)
NONEMPIRICAL_SOURCES = {"psi4_adjusted"}
ALLOWED_ELEMENTS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
FIELDNAMES = [
    "cas",
    "name",
    "formula",
    "source",
    "smiles",
    "heavy_atoms",
    "atoms",
    "geometry",
    "imaginary_modes",
    "lowest_frequency_cm_1",
    "optimization_and_hessian_seconds",
    *[
        field
        for temperature in TEMPERATURES
        for field in (
            f"reference_{temperature:g}_K",
            f"xtb_rrho_{temperature:g}_K",
            f"error_{temperature:g}_K",
            f"ape_{temperature:g}_K",
        )
    ],
    "status",
    "error",
]


def chebyshev_value(row: sqlite3.Row, temperature: float) -> float:
    coefficients = json.loads(row["cp_coefficients_json"])
    center = float(row["map_center_K"])
    scale = float(row["map_scale"])
    mapped = (temperature - center) / (scale * (temperature + center))
    return float(np.polynomial.chebyshev.chebval(mapped, coefficients))


def resolve_smiles(cas: str, name: str) -> str | None:
    for identifier in (cas, name):
        try:
            metadata = search_chemical(identifier)
        except Exception:
            continue
        smiles = str(getattr(metadata, "smiles", "") or "").strip()
        if smiles:
            return smiles
    return None


def normalized_formula(formula: str) -> dict[str, float] | None:
    try:
        return {
            element: float(count)
            for element, count in simple_formula_parser(formula).items()
        }
    except Exception:
        return None


def load_candidates(
    database: Path,
    max_heavy_atoms: int,
    *,
    paired_psi4: bool,
) -> list[dict[str, Any]]:
    psi4_records: dict[str, Any] = {}
    if paired_psi4:
        import chemicals.heat_capacity as heat_capacity

        psi4_path = (
            Path(heat_capacity.folder)
            / "psi4_adjusted_characteristic_temperatures.json"
        )
        psi4_records = json.loads(psi4_path.read_text(encoding="utf-8"))

    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM canonical_ideal_gas_cp
            WHERE source != 'psi4_adjusted'
              AND Tmin_fit_K <= 298.15 AND Tmax_fit_K >= 298.15
            ORDER BY cas
            """
        ).fetchall()

    candidates = []
    for row in rows:
        cas = str(row["cas"])
        if paired_psi4:
            if cas not in psi4_records:
                continue
            try:
                metadata = search_chemical(cas)
            except Exception:
                continue
            # Reject alternate, deprecated, or otherwise mis-keyed identities.
            # This removes the suspicious duplicate allene and acetylene rows.
            if str(metadata.CASs) != cas:
                continue
            if normalized_formula(str(metadata.formula)) != normalized_formula(
                str(row["formula"])
            ):
                continue
        smiles = resolve_smiles(cas, str(row["name"]))
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            continue
        if Chem.GetFormalCharge(molecule) != 0:
            continue
        if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
            continue
        if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
            continue
        if any(atom.GetSymbol() not in ALLOWED_ELEMENTS for atom in molecule.GetAtoms()):
            continue
        heavy_atoms = int(molecule.GetNumHeavyAtoms())
        if heavy_atoms > max_heavy_atoms:
            continue
        references = {
            temperature: chebyshev_value(row, temperature)
            for temperature in TEMPERATURES
            if float(row["Tmin_fit_K"]) <= temperature <= float(row["Tmax_fit_K"])
        }
        candidates.append(
            {
                "cas": cas,
                "name": str(row["name"]),
                "formula": str(row["formula"]),
                "source": str(row["source"]),
                "smiles": smiles,
                "heavy_atoms": heavy_atoms,
                "references": references,
            }
        )
    return candidates


def size_bucket(heavy_atoms: int) -> str:
    if heavy_atoms <= 2:
        return "1-2"
    if heavy_atoms <= 4:
        return "3-4"
    if heavy_atoms <= 6:
        return "5-6"
    return "7+"


def select_candidates(
    candidates: Sequence[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(candidates):
        return list(candidates)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        buckets[size_bucket(int(candidate["heavy_atoms"]))].append(candidate)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['cas']}".encode()
            ).hexdigest()
        )
    selected = []
    ordered_buckets = [buckets[key] for key in ("1-2", "3-4", "5-6", "7+") if buckets[key]]
    while len(selected) < limit and any(ordered_buckets):
        for bucket in ordered_buckets:
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
    return sorted(selected, key=lambda row: str(row["cas"]))


def excluded_cas_from_csv(paths: Sequence[Path]) -> set[str]:
    excluded = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            excluded.update(str(row["cas"]) for row in csv.DictReader(handle))
    return excluded


def molecular_geometry(atoms) -> str:
    if len(atoms) == 1:
        return "monatomic"
    if len(atoms) == 2:
        return "linear"
    moments = np.sort(np.asarray(atoms.get_moments_of_inertia(), dtype=float))
    return "linear" if moments[0] <= 1.0e-5 * moments[-1] else "nonlinear"


def select_vibrational_energies(
    energies: Sequence[complex], geometry: str, natoms: int
) -> list[complex]:
    count = 0 if geometry == "monatomic" else 3 * natoms - (5 if geometry == "linear" else 6)
    ordered = sorted((complex(value) for value in energies), key=abs)
    return ordered[-count:] if count else []


def harmonic_rrho_cp(
    temperature: float, energies_eV: Sequence[float], geometry: str
) -> float:
    rotational = 0.0 if geometry == "monatomic" else 1.0 if geometry == "linear" else 1.5
    cp_over_r = 2.5 + rotational
    for energy in energies_eV:
        reduced = energy / (units.kB * temperature)
        if reduced < 1.0e-5:
            cp_over_r += 1.0
        else:
            exp_negative = math.exp(-reduced)
            cp_over_r += reduced**2 * exp_negative / (1.0 - exp_negative) ** 2
    return R_J_MOL_K * cp_over_r


def calculate(candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    row = {field: "" for field in FIELDNAMES}
    row.update({key: candidate[key] for key in ("cas", "name", "formula", "source", "smiles", "heavy_atoms")})
    started = time.perf_counter()
    try:
        atoms, _charge, _multiplicity = DipoleMomentMixin._xtb_optimized_geometry(
            str(candidate["smiles"])
        )
        geometry = molecular_geometry(atoms)
        with tempfile.TemporaryDirectory(prefix="pfdsim-xtb-vib-") as directory:
            vibrations = Vibrations(
                atoms,
                name=str(Path(directory) / "vib"),
                delta=args.displacement,
                nfree=2,
            )
            vibrations.run()
            selected = select_vibrational_energies(
                vibrations.get_energies(), geometry, len(atoms)
            )

        cutoff_eV = args.imaginary_cutoff_cm_1 * units.invcm
        significant_imaginary = [
            energy for energy in selected if abs(energy.imag) > cutoff_eV
        ]
        if significant_imaginary:
            raise RuntimeError(
                f"optimized structure has {len(significant_imaginary)} imaginary "
                f"mode(s) above {args.imaginary_cutoff_cm_1:g} cm^-1"
            )
        real_energies = [
            abs(energy) if energy.imag else float(energy.real) for energy in selected
        ]
        row["atoms"] = len(atoms)
        row["geometry"] = geometry
        row["imaginary_modes"] = sum(bool(energy.imag) for energy in selected)
        row["lowest_frequency_cm_1"] = (
            min(real_energies) / units.invcm if real_energies else ""
        )
        for temperature, reference in candidate["references"].items():
            prediction = harmonic_rrho_cp(temperature, real_energies, geometry)
            error = prediction - reference
            row[f"reference_{temperature:g}_K"] = reference
            row[f"xtb_rrho_{temperature:g}_K"] = prediction
            row[f"error_{temperature:g}_K"] = error
            row[f"ape_{temperature:g}_K"] = 100.0 * abs(error / reference)
        row["status"] = "matched"
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["optimization_and_hessian_seconds"] = time.perf_counter() - started
    return row


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
    errors = [float(row[f"error_{temperature:g}_K"]) for row in usable]
    absolute = [abs(value) for value in errors]
    percentages = [float(row[f"ape_{temperature:g}_K"]) for row in usable]
    return {
        "n": len(usable),
        "mae_J_mol_K": statistics.fmean(absolute),
        "rmse_J_mol_K": math.sqrt(statistics.fmean(value * value for value in errors)),
        "bias_J_mol_K": statistics.fmean(errors),
        "mape_percent": statistics.fmean(percentages),
        "median_ape_percent": statistics.median(percentages),
        "maximum_ape_percent": max(percentages),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--limit", type=int, default=100, help="Use 0 for every eligible record.")
    parser.add_argument("--max-heavy-atoms", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--displacement", type=float, default=0.01, help="Finite-difference displacement in angstrom.")
    parser.add_argument("--imaginary-cutoff-cm-1", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--paired-psi4",
        action="store_true",
        help=(
            "Require adjusted-Psi4 coverage and an exact chemicals CAS/formula "
            "identity match, excluding suspicious alias-keyed references."
        ),
    )
    parser.add_argument(
        "--exclude-csv",
        action="append",
        type=Path,
        default=[],
        help="Exclude every CAS already listed in this result CSV; repeatable.",
    )
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    candidates = load_candidates(
        args.database,
        args.max_heavy_atoms,
        paired_psi4=args.paired_psi4,
    )
    excluded_cas = excluded_cas_from_csv(args.exclude_csv)
    candidates = [row for row in candidates if row["cas"] not in excluded_cas]
    selected = select_candidates(candidates, args.limit, args.seed)
    print(
        f"eligible={len(candidates)} selected={len(selected)} seed={args.seed} "
        f"max_heavy_atoms={args.max_heavy_atoms} paired_psi4={args.paired_psi4} "
        f"excluded_prior_cas={len(excluded_cas)}",
        flush=True,
    )
    if args.list_only:
        for row in selected:
            print(f"{row['cas']} | {row['name']} | {row['smiles']} | heavy={row['heavy_atoms']}")
        return

    rows = []
    overall_started = time.perf_counter()
    for index, candidate in enumerate(selected, 1):
        row = calculate(candidate, args)
        rows.append(row)
        write_csv(args.output, rows)
        print(
            f"[{index:02d}/{len(selected):02d}] {row['cas']} {row['name']}: "
            f"status={row['status']} Cp298={row['xtb_rrho_298.15_K']} "
            f"APE298={row['ape_298.15_K']} error={row['error']}",
            flush=True,
        )

    summary = {
        "method": "GFN2-xTB numerical Hessian; plain harmonic RRHO",
        "reference_scope": "canonical empirical ideal-gas Cp records",
        "excluded_sources": sorted(NONEMPIRICAL_SOURCES),
        "eligible_population": len(candidates),
        "sample_size": len(selected),
        "seed": args.seed,
        "max_heavy_atoms": args.max_heavy_atoms,
        "paired_psi4": args.paired_psi4,
        "excluded_prior_cas": len(excluded_cas),
        "successful_molecules": sum(row["status"] == "matched" for row in rows),
        "failed_molecules": sum(row["status"] != "matched" for row in rows),
        "metrics": {
            f"{temperature:g}_K": metric_summary(rows, temperature)
            for temperature in TEMPERATURES
        },
        "elapsed_seconds": time.perf_counter() - overall_started,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
