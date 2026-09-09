#!/usr/bin/env python3
"""Compare old and corrected mode selection on identical xTB Hessians.

Use a small seeded empirical cohort plus a synthetic saddle spectrum that
exposes imaginary-mode loss. No previous benchmark outputs are overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from ase import units
from ase.vibrations import Vibrations

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.heat_capacity.gas.benchmark_xtb_rrho_cp import (  # noqa: E402
    DATABASE,
    TEMPERATURES,
    DipoleMomentMixin,
    harmonic_rrho_cp,
    load_candidates,
    molecular_geometry,
    select_candidates,
    select_vibrational_energies,
)


def compare(energies, geometry, natoms):
    count = 0 if geometry == "monatomic" else 3 * natoms - (5 if geometry == "linear" else 6)
    old = sorted(energies, key=lambda value: (value**2).real)[-count:] if count else []
    new = select_vibrational_energies(energies, geometry, natoms)
    result = {}
    for label, modes in (("old", old), ("corrected", new)):
        rejected = any(abs(value.imag) > 20.0 * units.invcm for value in modes)
        result[label] = {
            "status": "rejected" if rejected else "accepted",
            "selected_cm_1": [[value.real / units.invcm, value.imag / units.invcm] for value in modes],
        }
    if all(item["status"] == "accepted" for item in result.values()):
        result["maximum_cp_change_J_mol_K"] = max(
            abs(harmonic_rrho_cp(T, [abs(v) for v in old], geometry)
                - harmonic_rrho_cp(T, [abs(v) for v in new], geometry))
            for T in TEMPERATURES
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    # Six near-zero external modes and three internal modes, one imaginary.
    saddle = [1j, 2, 3, 4, 5, 6, 100j, 500, 1000]
    print(json.dumps({"synthetic_saddle": compare(
        [complex(v) * units.invcm for v in saddle], "nonlinear", 3
    )}), flush=True)
    candidates = select_candidates(
        load_candidates(DATABASE, 2, paired_psi4=True), args.limit, args.seed
    )
    for candidate in candidates:
        result = {"cas": candidate["cas"], "name": candidate["name"]}
        try:
            atoms, _, _ = DipoleMomentMixin._xtb_optimized_geometry(candidate["smiles"])
            with tempfile.TemporaryDirectory(prefix="pfdsim-mode-comparison-") as directory:
                vibrations = Vibrations(atoms, name=str(Path(directory) / "vib"), delta=0.01, nfree=2)
                vibrations.run()
                energies = [complex(v) for v in vibrations.get_energies()]
            result.update(compare(energies, molecular_geometry(atoms), len(atoms)))
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
