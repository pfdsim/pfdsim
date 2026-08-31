#!/usr/bin/env python3
"""Forward unaffected critical-cache rows across organic-classifier versions."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemicals.identifiers import search_chemical  # noqa: E402
from compound_identity import parse_formula_counts  # noqa: E402
from property_resolution.organic_classification import (  # noqa: E402
    classify_strict_molecular_organic,
)


DEFAULT_CACHE = ROOT / "data" / "runtime" / "saturation_properties_cache.sqlite"


def identity_formula(props: dict) -> str:
    formula = str(props.get("formula") or props.get("Formula") or "").strip()
    if formula:
        return formula
    for key in ("CAS", "cas", "name"):
        value = props.get(key)
        if not value:
            continue
        try:
            formula = str(search_chemical(str(value)).formula or "").strip()
        except Exception:
            continue
        if formula:
            return formula
    return ""


def old_is_organic(props: dict, formula: str) -> bool:
    smiles = props.get("smiles")
    if smiles:
        try:
            from rdkit import Chem

            molecule = Chem.MolFromSmiles(str(smiles))
        except Exception:
            molecule = None
        if molecule is not None:
            for atom in molecule.GetAtoms():
                if atom.GetAtomicNum() != 6:
                    continue
                if atom.GetTotalNumHs() > 0:
                    return True
                if any(
                    neighbor.GetAtomicNum() in {9, 17, 35, 53}
                    for neighbor in atom.GetNeighbors()
                ):
                    return True
            return False
    counts = parse_formula_counts(formula) if formula else None
    return bool(
        counts
        and counts.get("C", 0) > 0
        and (
            counts.get("H", 0) > 0
            or any(counts.get(element, 0) > 0 for element in ("F", "Cl", "Br", "I"))
        )
    )


def classification_changed(row: sqlite3.Row) -> tuple[bool, str, bool, bool]:
    metadata = json.loads(row["input_metadata_json"])
    props = metadata.get("properties") or {}
    formula = identity_formula(props)
    old = old_is_organic(props, formula)
    new = classify_strict_molecular_organic(
        cas=props.get("CAS") or props.get("cas") or row["cas"],
        formula=formula,
        smiles=props.get("smiles"),
    ).is_organic
    return old != new, formula, old, new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--from-version", type=int, default=12)
    parser.add_argument("--to-version", type=int, default=13)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    connection = sqlite3.connect(args.cache, timeout=30.0)
    connection.row_factory = sqlite3.Row
    columns = [
        str(row[1])
        for row in connection.execute(
            "PRAGMA table_info(resolved_critical_properties_cache)"
        )
    ]
    rows = connection.execute(
        "SELECT * FROM resolved_critical_properties_cache WHERE cache_version = ?",
        (args.from_version,),
    ).fetchall()
    affected = []
    unaffected = []
    for row in rows:
        changed, formula, old, new = classification_changed(row)
        if changed:
            affected.append((row, formula, old, new))
        else:
            unaffected.append(row)

    print(
        f"from={args.from_version} rows={len(rows)} unaffected={len(unaffected)} "
        f"affected={len(affected)} apply={args.apply}"
    )
    for row, formula, old, new in affected:
        print(
            "affected\t"
            f"{row['component_key']}\t{row['symbol']}\t{row['cas']}\t"
            f"{row['component_name']}\t{formula}\t{old}->{new}"
        )

    if args.apply:
        placeholders = ",".join("?" for _ in columns)
        statement = (
            f"INSERT OR IGNORE INTO resolved_critical_properties_cache "
            f"({','.join(columns)}) VALUES ({placeholders})"
        )
        before = connection.total_changes
        connection.execute("BEGIN IMMEDIATE")
        for row in unaffected:
            values = [row[column] for column in columns]
            values[columns.index("cache_version")] = args.to_version
            connection.execute(statement, values)
        connection.execute(f"PRAGMA user_version = {int(args.to_version)}")
        connection.commit()
        inserted = connection.total_changes - before
        target_count = connection.execute(
            "SELECT COUNT(*) FROM resolved_critical_properties_cache "
            "WHERE cache_version = ?",
            (args.to_version,),
        ).fetchone()[0]
        print(f"inserted={inserted} target_version_rows={target_count}")
    connection.close()


if __name__ == "__main__":
    main()
