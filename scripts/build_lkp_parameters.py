#!/usr/bin/env python3
"""Build LKP's CAS-keyed table from pinned DWSIM data, with audited errata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source/lkp_ip.dat.txt"
OUTPUT = ROOT / "data/lkp_binary_interactions_cas.json"
COMMIT = "1abf72d1b6b41d3e9a8cc770d3cc4e8fc76e5766"
SOURCE_URL = (
    f"https://github.com/DanWBR/dwsim/blob/{COMMIT}/"
    "DWSIM.Thermodynamics/Assets/lkp_ip.dat"
)
PAPER = "https://doi.org/10.1021/i260067a020"

# Verified visually against Table I, p. 327 of Plocker, Knapp & Prausnitz
# (1978), stored in data/reference/thermodynamic-models. Keys are original
# (component1, component2, value) triples so changed upstream input fails.
CORRECTIONS = {
    ("Methane", "N-pentane", "1.2401"): ("Methane", "N-pentane", "1.240"),
    ("Propane", "N-pentane", "1.008"): ("Propane", "N-pentane", "1.006"),
    ("N-pentane", "N-hexane", "0.998"): ("N-pentane", "N-hexane", "0.996"),
    ("N-octane", "N-decane", "0.991"): ("N-nonane", "N-decane", "0.991"),
    ("Nitrogen", "Nitrogen", "0.977"): ("Nitrogen", "Methane", "0.977"),
    ("Carbon dioxide", "Benzene", "1018"): ("Carbon dioxide", "Benzene", "1.018"),
    ("Hydrogen", "Methane", "1.604"): ("Hydrogen", "Ethane", "1.604"),
    ("Hydrogen", "Ethylene", "1408"): ("Hydrogen", "Ethylene", "1.498"),
    ("Hydrogen", "Propane", "1828"): ("Hydrogen", "Propane", "1.826"),
    ("Hydrogen", "itbutane", "2093"): ("Hydrogen", "N-butane", "2.093"),
    ("Hydrogen", "N-hexane", "2.466"): ("Hydrogen", "N-hexane", "2.456"),
    ("Hydrogen", "N-heptane", "2.834"): ("Hydrogen", "N-heptane", "2.634"),
    ("Carbon monoxide", "Carbon Methane", "0.974"): ("Carbon monoxide", "Methane", "0.974"),
}

CAS = {
    "Acetylene": "74-86-2", "Ammonia": "7664-41-7", "Argon": "7440-37-1",
    "Benzene": "71-43-2", "Carbon dioxide": "124-38-9",
    "Carbon monoxide": "630-08-0", "Cyclohexane": "110-82-7",
    "Ethane": "74-84-0", "Ethylene": "74-85-1", "Hydrogen": "1333-74-0",
    "Hydrogen sulfide": "7783-06-4", "Isobutane": "75-28-5",
    "Isobutene": "115-11-7", "Isooctane": "540-84-1", "Isopentane": "78-78-4",
    "Krypton": "7439-90-9", "Methane": "74-82-8", "Methanol": "67-56-1",
    "N-butane": "106-97-8", "N-decane": "124-18-5", "N-heptane": "142-82-5",
    "N-hexane": "110-54-3", "N-nonane": "111-84-2", "N-octane": "111-65-9",
    "N-pentane": "109-66-0", "Nitrogen": "7727-37-9", "Nitrous oxide": "10024-97-2",
    "Oxygen": "7782-44-7", "Propane": "74-98-6", "Propylene": "115-07-1",
    "R12": "75-71-8", "Water": "7732-18-5",
}


def build() -> dict:
    raw = SOURCE.read_bytes()
    records, corrections, seen = [], [], set()
    for line_number, line in enumerate(raw.decode("cp1252").splitlines()[1:], 2):
        original = tuple(line.split("\t"))
        first, second, value = corrected = CORRECTIONS.get(original, original)
        pair = tuple(sorted((CAS[first], CAS[second])))
        if pair[0] == pair[1] or pair in seen:
            raise ValueError(f"Invalid or duplicate LKP pair at line {line_number}")
        seen.add(pair)
        records.append({"cas1": pair[0], "cas2": pair[1], "kij": float(value)})
        if corrected != original:
            corrections.append({"line": line_number, "original": original, "corrected": corrected})
    if len(corrections) != len(CORRECTIONS):
        raise ValueError("The pinned DWSIM input no longer matches the errata manifest")
    return {
        "source": SOURCE_URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "errata_source": PAPER,
        "convention": "Tc_ij = kij * sqrt(Tc_i * Tc_j); neutral kij = 1",
        "corrections": corrections,
        "records": sorted(records, key=lambda r: (r["cas1"], r["cas2"])),
    }


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"Built {OUTPUT.relative_to(ROOT)}")
