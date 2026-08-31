#!/usr/bin/env python3
"""Build CAS-keyed PRSV alpha parameters from the DWSIM source table."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_DATA = DATA / "source"
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compound_identity import compact_identifier, get_compound_identity_resolver


PRSV_SOURCE = SOURCE_DATA / "prsv2.dat"
OUTPUT = DATA / "prsv_parameters_cas.json"

# DWSIM source names that are either abbreviated or absent from the central
# resolver but are unambiguous in the PRSV table context.
SOURCE_NAME_OVERRIDES = {
    "Propene": "Propylene",
    "Butanone": "2-Butanone",
    "Indane": "496-11-7",
    "Naphtalene": "Naphthalene",
    "1-Methyl-naphtalene": "1-methylnaphthalene",
    "2-Methyl-naphtalene": "2-methylnaphthalene",
    "9,10-Dihydrophenanthrene": "776-35-2",
    "Methylbutanone": "3-methyl-2-butanone",
    "Dimethylbutanone": "75-97-8",
    "5-Nonanone": "502-56-7",
    "Dimethyl Ether": "Dimethyl ether",
    "Methyl Ethyl Ether": "Methylethyl ether",
    "Methyl n-Propyl Ether": "Methylpropyl ether",
    "Methyl i-Propyl Ether": "Methylisopropyl ether",
    "Methyl n-Butyl Ether": "Methylbutyl ether",
    "Methyl t-Butyl Ether": "Methyl tert-butyl ether",
    "Ethyl n-Propyl Ether": "Ethylpropyl ether",
    "Di-n-Propyl Ether": "Dipropyl ether",
    "Di-i-Propyl Ether": "Diisopropyl ether",
    "Methyl Phenyl Ether": "Anisole",
    "Dimethylformamide": "N,N-Dimethyl formamide",
    "1-Propylamine": "n-Propylamine",
    "2-Propylamine": "Isopropyl amine",
    "2-Methoxypropionitrile": "78-82-0",
    "2-Methyl-2-Propylamine": "tert-butylamine",
    "Thianaphthene": "95-15-8",
}

RESOLVED_SOURCE_NAMES = {
    "2-Methoxypropionitrile": "2-methylpropanenitrile",
}

EXCLUDED_SOURCE_NAMES = {
    "Nitrotoluene": (
        "Generic Nitrotoluene row best matches 4-nitrotoluene by PRSV2 saturation "
        "checks, but is excluded because predicted Tsat errors are about 2.4 C at "
        "1 atm and 3.6 C at 0.2666 bar."
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_alias(value: Any) -> str:
    return compact_identifier(str(value or ""))


def valid_cas(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text if CAS_RE.match(text) else None


def add_alias(
    aliases: dict[str, tuple[str, str]],
    alias: Any,
    cas: str,
    source: str,
) -> None:
    key = normalize_alias(alias)
    if key and valid_cas(cas) and key not in aliases:
        aliases[key] = (cas, source)


def add_component_aliases(
    aliases: dict[str, tuple[str, str]],
    cas: str,
    entry: dict[str, Any],
    source: str,
) -> None:
    for value in (
        entry.get("name"),
        entry.get("formula"),
        entry.get("resolved_query"),
        entry.get("table_name"),
        entry.get("raw_table_name"),
        *(entry.get("aliases") or []),
    ):
        add_alias(aliases, value, cas, source)


def add_output_alias(aliases: dict[str, str], alias: Any, cas: str) -> None:
    key = normalize_alias(alias)
    if key and valid_cas(cas):
        aliases.setdefault(key, cas)


def local_aliases() -> dict[str, tuple[str, str]]:
    aliases: dict[str, tuple[str, str]] = {}

    uniquac_path = DATA / "uniquac_rq_cas.json"
    if uniquac_path.exists():
        payload = load_json(uniquac_path)
        for alias, cas in payload.get("aliases", {}).items():
            add_alias(aliases, alias, cas, "uniquac_rq_cas.json")
        for cas, entry in payload.get("components", {}).items():
            add_component_aliases(aliases, cas, entry, "uniquac_rq_cas.json")

    interaction_index_path = DATA / "interaction_component_cas_index.json"
    if interaction_index_path.exists():
        payload = load_json(interaction_index_path)
        for cas, entry in payload.get("components", {}).items():
            add_component_aliases(aliases, cas, entry, "interaction_component_cas_index.json")

    for filename in ("perry_properties.json", "perry_table_2_10_vapor_pressure.json"):
        path = DATA / filename
        if not path.exists():
            continue
        payload = load_json(path)
        for cas, entry in payload.get("chemicals", {}).items():
            add_component_aliases(aliases, cas, entry, filename)

    return aliases


def resolve_cas(
    name: str,
    resolver: Any,
    aliases: dict[str, tuple[str, str]],
) -> tuple[Optional[str], str]:
    cas = resolver.resolve_cas(name, allow_formula=False)
    if cas:
        return cas, "compound_identity"

    key = normalize_alias(name)
    if key in aliases:
        return aliases[key]

    override = SOURCE_NAME_OVERRIDES.get(name)
    if not override:
        return None, ""
    direct_cas = valid_cas(override)
    if direct_cas:
        return direct_cas, "manual_prsv_identity_override"

    cas = resolver.resolve_cas(override, allow_formula=False)
    if cas:
        return cas, f"manual_prsv_identity_override:{override}:compound_identity"

    key = normalize_alias(override)
    if key in aliases:
        cas, source = aliases[key]
        return cas, f"manual_prsv_identity_override:{override}:{source}"
    return None, ""


def prsv_records() -> list[dict[str, Any]]:
    with open(PRSV_SOURCE, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    resolver = get_compound_identity_resolver()
    aliases = local_aliases()
    components: dict[str, dict[str, Any]] = {}
    output_aliases: dict[str, str] = {}
    unresolved = []

    for row in prsv_records():
        source_name = str(row.get("CompoundName") or "").strip()
        kappa1 = parse_float(row.get("Kappa1"))
        kappa2 = parse_float(row.get("Kappa2"))
        kappa3 = parse_float(row.get("Kappa3"))
        if not source_name or kappa1 is None or kappa2 is None or kappa3 is None:
            unresolved.append({"source_name": source_name, "reason": "invalid_source_row"})
            continue
        if source_name in EXCLUDED_SOURCE_NAMES:
            unresolved.append({
                "source_name": source_name,
                "reason": "excluded_after_saturation_check",
                "notes": EXCLUDED_SOURCE_NAMES[source_name],
            })
            continue

        cas, identity_source = resolve_cas(source_name, resolver, aliases)
        if not cas:
            unresolved.append({"source_name": source_name, "reason": "cas_not_resolved"})
            continue

        entry = {
            "cas": cas,
            "name": RESOLVED_SOURCE_NAMES.get(source_name, source_name),
            "source_name": source_name,
            "prsv1": {
                "kappa1": kappa1,
            },
            "prsv2": {
                "kappa1": kappa1,
                "kappa2": kappa2,
                "kappa3": kappa3,
            },
            "source": "DWSIM PRSV2 parameter table",
            "source_file": str(PRSV_SOURCE.relative_to(ROOT)),
            "identity_source": identity_source,
        }
        components[cas] = entry
        add_output_alias(output_aliases, source_name, cas)

    payload = {
        "metadata": {
            "key_basis": "CAS",
            "description": "CAS-keyed PRSV1/PRSV2 pure-component alpha parameters from DWSIM.",
            "source_files": [str(PRSV_SOURCE.relative_to(ROOT))],
            "component_count": len(components),
            "source_row_count": len(prsv_records()),
            "unresolved_component_count": len(unresolved),
            "unresolved_components": unresolved,
            "notes": [
                "PRSV1 uses kappa1.",
                "PRSV2 uses kappa1, kappa2, and kappa3.",
                "CAS resolution uses compound_identity first, then existing local CAS-keyed indexes, then audited PRSV source-name overrides.",
            ],
        },
        "components": dict(sorted(components.items())),
        "aliases": dict(sorted(output_aliases.items())),
    }
    write_json(OUTPUT, payload)
    print(
        f"Wrote {len(components)} PRSV CAS entries from {len(prsv_records())} source rows; "
        f"unresolved={len(unresolved)}"
    )
    for item in unresolved:
        print(f"  unresolved: {item['source_name']} ({item['reason']})")


if __name__ == "__main__":
    main()
