#!/usr/bin/env python3
"""Convert ChemSep interaction .dat.txt files into compact JSON tables."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_DATA = DATA / "source"
ACTIVITY_SOURCE_DATA = SOURCE_DATA / "activity_fitting"
ARCHIVED_DATA = DATA / "archived"


def parse_component_ids() -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    path = SOURCE_DATA / "csid.dat.txt"
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("CHEMSEP"):
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        entry = {"id": parts[0], "name": parts[1]}
        if len(parts) > 2 and parts[2]:
            entry["dwsim_name"] = parts[2]
        entries[parts[0]] = entry
    return entries


def parse_eos_interactions(filename: str, model: str) -> list[dict]:
    records: list[dict] = []
    for line in (SOURCE_DATA / filename).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("==="):
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) < 3:
            continue
        try:
            id1, id2 = parts[0], parts[1]
            kij = float(parts[2])
        except ValueError:
            continue
        records.append(
            {
                "model": model,
                "id1": id1,
                "id2": id2,
                "kij": kij,
                "comment": parts[3] if len(parts) > 3 else "",
            }
        )
    return records


def parse_nrtl() -> list[dict]:
    records: list[dict] = []
    for line in (ACTIVITY_SOURCE_DATA / "nrtl.dat.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("==="):
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) < 5:
            continue
        try:
            id1, id2 = parts[0], parts[1]
            a12 = float(parts[2])
            a21 = float(parts[3])
            alpha12 = float(parts[4])
        except ValueError:
            continue
        records.append(
            {
                "id1": id1,
                "id2": id2,
                "a12_cal_per_mol": a12,
                "a21_cal_per_mol": a21,
                "alpha12": alpha12,
                "comment": parts[5] if len(parts) > 5 else "",
            }
        )
    return records


def parse_uniquac() -> list[dict]:
    records: list[dict] = []
    for line in (ACTIVITY_SOURCE_DATA / "uniquac.dat.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("==="):
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) < 4:
            continue
        try:
            id1, id2 = parts[0], parts[1]
            a12 = float(parts[2])
            a21 = float(parts[3])
        except ValueError:
            continue
        records.append(
            {
                "id1": id1,
                "id2": id2,
                "a12_cal_per_mol": a12,
                "a21_cal_per_mol": a21,
                "comment": parts[4] if len(parts) > 4 else "",
            }
        )
    return records


def main() -> None:
    ARCHIVED_DATA.mkdir(exist_ok=True)
    components = parse_component_ids()
    eos_records = (
        parse_eos_interactions("srk_ip.dat.txt", "SRK")
        + parse_eos_interactions("pr_ip.dat.txt", "PR")
    )
    nrtl_records = parse_nrtl()
    uniquac_records = parse_uniquac()

    (SOURCE_DATA / "chemsep_components.json").write_text(
        json.dumps({"components": components}, indent=2, sort_keys=True) + "\n"
    )
    (ARCHIVED_DATA / "eos_binary_interactions.json").write_text(
        json.dumps({"interactions": eos_records}, indent=2, sort_keys=True) + "\n"
    )
    (ARCHIVED_DATA / "nrtl_binary_interactions.json").write_text(
        json.dumps(
            {
                "units": "cal/mol",
                "source": "ChemSep DECHEMA NRTL data at 1 atm",
                "interactions": nrtl_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (ARCHIVED_DATA / "uniquac_binary_interactions.json").write_text(
        json.dumps(
            {
                "units": "cal/mol",
                "source": "ChemSep DECHEMA UNIQUAC data at 1 atm",
                "interactions": uniquac_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
