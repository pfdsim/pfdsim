#!/usr/bin/env python3
"""Build CAS-keyed interaction parameter tables from ChemSep-ID sources."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from statistics import median
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE_DATA = DATA / "source"
ARCHIVED_DATA = DATA / "archived"
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")

ASSORTED_ALCOHOL_ETHER_CAS = {
    "1-butanol": "71-36-3",
    "1-propanol": "71-23-8",
    "2-methoxyethanol": "109-86-4",
    "2-propanol": "67-63-0",
    "acetonitrile": "75-05-8",
    "anisole": "100-66-3",
    "butylamine": "109-73-9",
    "cyclopentanone": "120-92-3",
    "dibutyl ether": "142-96-1",
    "diethyl ether": "60-29-7",
    "diisopropyl ether": "108-20-3",
    "dipropyl ether": "111-43-3",
    "ethanol": "64-17-5",
    "isopropyl acetate": "108-21-4",
    "methanol": "67-56-1",
    "n-heptane": "142-82-5",
    "n-hexane": "110-54-3",
    "n-octane": "111-65-9",
    "propanone": "67-64-1",
    "propylamine": "107-10-8",
    "tetrahydrofuran": "109-99-9",
    "water": "7732-18-5",
}

SOURCE_INTERACTION_FILES = (
    "eos_binary_interactions.json",
    "nrtl_binary_interactions.json",
    "uniquac_binary_interactions.json",
)

EOS_IPD_FILES = {
    "SRK": "srk.ipd",
    "PR": "pr.ipd",
}

ACTIVITY_IPD_FILES = {
    "NRTL": "nrtl.ipd",
    "UNIQUAC": "uniquac.ipd",
}

CURATED_ACTIVITY_DISABLED_RECORDS = {
    (
        "NRTL",
        ("56-23-5", "67-56-1"),
        "Tetrachloromethane/Methanol p279 1/2c",
    ): (
        "Rejected by curated duplicate check: UNIFNIST gamma-infinity and "
        "1 atm Txy diagnostics both favor Methanol/Tetrachloromethane p18 1/2c."
    ),
    (
        "UNIQUAC",
        ("56-23-5", "67-56-1"),
        "Tetrachloromethane/Methanol p279 1/2c",
    ): (
        "Rejected by curated duplicate check: UNIFNIST gamma-infinity and "
        "1 atm Txy diagnostics both favor Methanol/Tetrachloromethane p18 1/2c."
    ),
    (
        "UNIQUAC",
        ("67-56-1", "7732-18-5"),
        "Methanol/Water | Water/Methanol",
    ): (
        "Rejected by curated duplicate check: 1 atm Txy diagnostics favor "
        "Methanol/Water (Kojima+Kato)."
    ),
    (
        "NRTL",
        ("7732-18-5", "78-93-3"),
        "Water/2-Butanone p277 1/1a",
    ): "Rejected by curated duplicate check: known azeotrope match favors 2-Butanone/Water p279 1/1a.",
    (
        "UNIQUAC",
        ("7732-18-5", "78-93-3"),
        "Water/2-Butanone p277 1/1a",
    ): "Rejected by curated duplicate check: known azeotrope match favors 2-Butanone/Water p279 1/1a.",
    (
        "NRTL",
        ("64-17-5", "78-93-3"),
        "Ethanol/2-Butanone p327 1/2c",
    ): "Rejected by curated duplicate check: known azeotrope match favors Ethanol/2-Butanone p342 1/2a.",
    (
        "UNIQUAC",
        ("64-17-5", "78-93-3"),
        "Ethanol/2-Butanone p327 1/2c",
    ): "Rejected by curated duplicate check: known azeotrope match favors Ethanol/2-Butanone p342 1/2a.",
    (
        "NRTL",
        ("110-82-7", "67-56-1"),
        "Methanol/CycloHexane p243 1/2a",
    ): "Rejected by curated duplicate check: known azeotrope match favors Methanol/Cyclohexane p211 1/2c.",
    (
        "UNIQUAC",
        ("110-82-7", "67-56-1"),
        "Methanol/CycloHexane p243 1/2a",
    ): "Rejected by curated duplicate check: known azeotrope match favors Methanol/Cyclohexane p211 1/2c.",
    (
        "NRTL",
        ("142-82-5", "67-56-1"),
        "Methanol/1-Heptane p241 1/2c",
    ): "Rejected by curated duplicate check: known azeotrope match favors Methanol/n-Heptane.",
    (
        "NRTL",
        ("142-82-5", "67-56-1"),
        "Methanol/Heptane p243 1/2c",
    ): "Rejected by curated duplicate check: known azeotrope match favors Methanol/n-Heptane.",
    (
        "UNIQUAC",
        ("142-82-5", "67-56-1"),
        "Methanol/1-Heptane p241 1/2c",
    ): "Rejected by curated duplicate check: known azeotrope match favors Methanol/n-Heptane.",
    (
        "UNIQUAC",
        ("142-82-5", "67-56-1"),
        "Methanol/Heptane p243 1/2c",
    ): "Rejected by curated duplicate check: known azeotrope match favors Methanol/n-Heptane.",
    (
        "NRTL",
        ("110-82-7", "64-17-5"),
        "Ethanol/Cyclohexane p419 1/2c",
    ): "Rejected by curated duplicate check: known azeotrope match favors Ethanol/CycloHexane p441 1/2a.",
    (
        "UNIQUAC",
        ("110-82-7", "64-17-5"),
        "Ethanol/Cyclohexane p419 1/2c",
    ): "Rejected by curated duplicate check: known azeotrope match favors Ethanol/CycloHexane p441 1/2a.",
    (
        "NRTL",
        ("64-17-5", "67-64-1"),
        "Acetone/Ethanol p312 1/2c",
    ): "Rejected by curated duplicate check: experimental VLE residuals favor Ethanol/Acetone p323 1/2a.",
    (
        "UNIQUAC",
        ("64-17-5", "67-64-1"),
        "Acetone/Ethanol p312 1/2c",
    ): "Rejected by curated duplicate check: experimental VLE residuals favor Ethanol/Acetone p323 1/2a.",
    (
        "NRTL",
        ("64-17-5", "67-56-1"),
        "Methanol/Ethanol p55 1/2a",
    ): "Rejected by curated duplicate check: experimental VLE residuals favor Methanol/Ethanol p60 1/2c.",
    (
        "UNIQUAC",
        ("64-17-5", "67-56-1"),
        "Methanol/Ethanol p55 1/2a",
    ): "Rejected by curated duplicate check: experimental VLE residuals favor Methanol/Ethanol p60 1/2c.",
}


def normalize_name(value: str) -> str:
    text = value.lower().strip()
    replacements = {
        "n,n-": "",
        "n.n-": "",
        "n-": "",
        "normal ": "",
        "tert.": "tert-",
        "tert ": "tert-",
        "t-": "tert-",
        "sec.": "sec-",
        "sec ": "sec-",
        "iso-": "iso",
        "i-": "iso",
        "p-": "p",
        "m-": "m",
        "o-": "o",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("flouro", "fluoro")
    text = text.replace("propnaol", "propanol")
    text = text.replace("pyridien", "pyridine")
    text = text.replace("pyridinr", "pyridine")
    return re.sub(r"[^a-z0-9]+", "", text)


NAME_ALIASES = {
    "h2o": "water",
    "c2h5oh": "ethanol",
    "ch3oh": "methanol",
    "nh3": "ammonia",
    "co": "carbonmonoxide",
    "co2": "carbondioxide",
    "n2": "nitrogen",
    "o2": "oxygen",
    "ar": "argon",
    "h2": "hydrogen",
    "c6h6": "benzene",
    "c6h14": "hexane",
    "c7h16": "heptane",
    "etoh": "ethanol",
    "meoh": "methanol",
    "ethylalcohol": "ethanol",
    "methylalcohol": "methanol",
    "ethylbenzol": "ethylbenzene",
    "ethylbezene": "ethylbenzene",
    "ethylacetate": "ethylacetate",
    "ethylacetat": "ethylacetate",
    "ethylether": "diethylether",
    "propanoicacid": "propionicacid",
    "isopentane": "2methylbutane",
    "neopentane": "22dimethylpropane",
    "isobutene": "2methylpropene",
    "methylal": "dimethoxymethane",
    "methylisopropylketone": "3methyl2butanone",
    "ipentanol": "3methyl1butanol",
    "monochlorobenzene": "chlorobenzene",
    "butylchloride": "1chlorobutane",
    "isopentylacetate": "isoamylacetate",
    "diacetonealcohol": "4hydroxy4methyl2pentanone",
    "nmethylpyrrolidone": "1methyl2pyrrolidinone",
}

COMMENT_ALIASES = {
    "C1": "methane",
    "C2": "ethane",
    "C3": "propane",
    "nC3": "propane",
    "nC4": "butane",
    "nC5": "pentane",
    "nC6": "hexane",
    "nC7": "heptane",
    "nC8": "octane",
    "nC9": "nonane",
    "nC10": "decane",
    "CarbonDioxide": "carbon dioxide",
    "CarbonMonoxide": "carbon monoxide",
    "Helium-4": "helium",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_record_identity_overrides() -> dict[str, dict[str, dict]]:
    path = SOURCE_DATA / "interaction_record_identity_overrides.json"
    if not path.exists():
        return {}
    return load_json(path).get("records", {})


def add_name_alias(mapping: dict[str, tuple[str, str, str]], name: Optional[str], cas: Optional[str], source: str) -> None:
    if not name or not cas or not CAS_RE.match(cas):
        return
    key = normalize_name(str(name))
    if not key:
        return
    mapping.setdefault(key, (cas, source, str(name)))
    alias = NAME_ALIASES.get(key)
    if alias:
        mapping.setdefault(alias, (cas, source, str(name)))


def local_name_to_cas() -> dict[str, tuple[str, str, str]]:
    mapping: dict[str, tuple[str, str, str]] = {}

    chemicals_path = DATA / "chemicals.json"
    if chemicals_path.exists():
        for key, entry in load_json(chemicals_path).get("chemicals", {}).items():
            cas = entry.get("CAS") or entry.get("cas")
            for name in (key, entry.get("name"), entry.get("symbol"), entry.get("formula")):
                add_name_alias(mapping, name, cas, "chemicals.json")

    perry_path = DATA / "perry_properties.json"
    if perry_path.exists():
        for cas, entry in load_json(perry_path).get("chemicals", {}).items():
            names = [entry.get("name"), entry.get("table_name"), entry.get("formula")]
            names.extend(entry.get("names") or [])
            names.extend(entry.get("formulas") or [])
            for name in names:
                add_name_alias(mapping, name, cas, "perry_properties.json")

    vapor_path = DATA / "perry_table_2_10_vapor_pressure.json"
    if vapor_path.exists():
        for cas, entry in load_json(vapor_path).get("chemicals", {}).items():
            names = [entry.get("name"), entry.get("table_name"), entry.get("formula"), entry.get("query")]
            names.extend(entry.get("aliases") or [])
            for name in names:
                add_name_alias(mapping, name, cas, "perry_table_2_10_vapor_pressure.json")

    return mapping


def clean_comment_side(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^\d+\s+", "", text)
    text = re.sub(r"\s+p\d+.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\d+\s+1/.*$", "", text)
    text = re.sub(r"\s+T=.*$", "", text)
    return text.strip()


def collect_component_candidates() -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = {}
    components = load_json(SOURCE_DATA / "chemsep_components.json")["components"]
    for comp_id, entry in components.items():
        candidates[comp_id] = [
            value for value in (entry.get("name"), entry.get("dwsim_name"))
            if value
        ]

    for filename in SOURCE_INTERACTION_FILES:
        for record in load_json(ARCHIVED_DATA / filename)["interactions"]:
            comment = record.get("comment", "")
            if "/" not in comment:
                continue
            left, right = comment.split("/", 1)
            id1, id2 = str(record["id1"]), str(record["id2"])
            if not candidates.get(id1):
                candidates.setdefault(id1, []).append(clean_comment_side(left))
            if not candidates.get(id2):
                candidates.setdefault(id2, []).append(clean_comment_side(right))

    for comp_id, values in list(candidates.items()):
        for value in list(values):
            alias = COMMENT_ALIASES.get(value)
            if alias:
                values.append(alias)
    return candidates


def interaction_source_ids() -> set[str]:
    ids: set[str] = set()
    for filename in SOURCE_INTERACTION_FILES:
        for record in load_json(ARCHIVED_DATA / filename)["interactions"]:
            ids.add(str(record["id1"]))
            ids.add(str(record["id2"]))
    return ids


def load_chemicals_resolver(path: Optional[str]):
    if not path:
        return None
    sys.path.insert(0, path)
    try:
        from chemicals.identifiers import search_chemical
    except Exception as exc:
        raise RuntimeError(f"Could not import chemicals.identifiers from {path!r}") from exc
    return search_chemical


def resolve_component_ids(chemicals_path: Optional[str] = None) -> tuple[dict[str, dict], list[dict]]:
    name_map = local_name_to_cas()
    candidates = collect_component_candidates()
    source_ids = interaction_source_ids()
    overrides = load_json(SOURCE_DATA / "interaction_component_cas_overrides.json").get("components", {})
    search_chemical = load_chemicals_resolver(chemicals_path)
    resolved: dict[str, dict] = {}
    unresolved: list[dict] = []

    for comp_id in sorted(source_ids, key=lambda value: int(value) if value.isdigit() else value):
        values = candidates.get(comp_id, [])
        override = overrides.get(comp_id)
        if override:
            override_alias_keys = {
                normalize_name(str(value))
                for value in (
                    [override.get("name"), override.get("query")]
                    + list(override.get("aliases", []))
                )
                if value
            }
            candidate_aliases = {
                value for value in values
                if normalize_name(value) in override_alias_keys
            }
            resolved[comp_id] = {
                "cas": override["cas"],
                "name": override.get("name", ""),
                "formula": override.get("formula", ""),
                "source": override.get("source", "manual_identity_override"),
                "query": override.get("query", ""),
                "aliases": sorted(
                    {value for value in override.get("aliases", []) if value}
                    | {value for value in candidate_aliases if value}
                ),
            }
            continue

        hit = None
        for value in values:
            key = normalize_name(value)
            for lookup_key in (key, NAME_ALIASES.get(key)):
                if lookup_key and lookup_key in name_map:
                    cas, source, matched_name = name_map[lookup_key]
                    hit = {
                        "cas": cas,
                        "name": matched_name,
                        "formula": "",
                        "source": source,
                        "query": value,
                        "aliases": sorted({item for item in values if item}),
                    }
                    break
            if hit:
                break

        if hit is None and search_chemical is not None:
            for value in values:
                if not value or value == "HC":
                    continue
                try:
                    chemical = search_chemical(value)
                except Exception:
                    continue
                hit = {
                    "cas": chemical.CASs,
                    "name": chemical.common_name,
                    "formula": chemical.formula,
                    "source": "chemicals package",
                    "query": value,
                    "aliases": sorted({item for item in values if item}),
                }
                break

        if hit and CAS_RE.match(hit["cas"]):
            resolved[comp_id] = hit
        else:
            unresolved.append({
                "source_component_id": comp_id,
                "candidates": sorted({item for item in values if item}),
            })

    return resolved, unresolved


def temperature_range(comment: str) -> tuple[float, float] | None:
    match = re.search(
        r"T=([0-9]+(?:\.[0-9]+)?)(?:-([0-9]+(?:\.[0-9]+)?))?K",
        comment,
        re.IGNORECASE,
    )
    if not match:
        return None
    low = float(match.group(1))
    high = float(match.group(2)) if match.group(2) is not None else low
    return min(low, high), max(low, high)


def component_from_record_override(entry: dict, side: str) -> Optional[dict]:
    override = entry.get(side)
    if not override:
        return None
    cas = override.get("cas", "")
    if not CAS_RE.match(cas):
        raise ValueError(f"Invalid CAS {cas!r} in record identity override for {side}")
    return {
        "cas": cas,
        "name": override.get("name", ""),
        "formula": override.get("formula", ""),
        "source": override.get("source", "manual_record_identity_override"),
        "query": override.get("query", override.get("name", "")),
        "aliases": list(override.get("aliases", [])),
    }


def convert_records(filename: str, resolved: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    converted: list[dict] = []
    skipped: list[dict] = []
    record_overrides = load_record_identity_overrides().get(filename, {})
    for index, record in enumerate(load_json(ARCHIVED_DATA / filename)["interactions"]):
        id1, id2 = str(record["id1"]), str(record["id2"])
        override = record_overrides.get(str(index), {})
        comp1 = component_from_record_override(override, "component1") or resolved.get(id1)
        comp2 = component_from_record_override(override, "component2") or resolved.get(id2)
        if comp1 is None or comp2 is None:
            skipped.append({
                "index": index,
                "source_file": filename,
                "source_id1": id1,
                "source_id2": id2,
                "comment": record.get("comment", ""),
                "reason": "missing unambiguous CAS for one or both source component IDs",
            })
            continue

        new_record = {
            key: value
            for key, value in record.items()
            if key not in {"id1", "id2"}
        }
        new_record.update({
            "cas1": comp1["cas"],
            "cas2": comp2["cas"],
            "component1": comp1.get("name", ""),
            "component2": comp2.get("name", ""),
            "source_id1": id1,
            "source_id2": id2,
        })
        for key in ("disabled", "disabled_reason"):
            if key in override:
                new_record[key] = override[key]
        if "kij" in new_record:
            new_record["kij"] = float(new_record["kij"])
            ranged = temperature_range(record.get("comment", ""))
            if ranged is not None:
                new_record["Tmin_K"], new_record["Tmax_K"] = ranged
        converted.append(new_record)
    return converted, skipped


def record_override_components() -> dict[str, dict]:
    components: dict[str, dict] = {}
    for filename, records in load_record_identity_overrides().items():
        for index, override in records.items():
            for side in ("component1", "component2"):
                component = component_from_record_override(override, side)
                if component is None:
                    continue
                components[f"{filename}:{index}:{side}"] = component
    return components


def resolve_matrix_component_cas(name: str) -> str:
    name_map = local_name_to_cas()
    key = normalize_name(name)
    for lookup_key in (key, NAME_ALIASES.get(key)):
        if lookup_key and lookup_key in name_map:
            return name_map[lookup_key][0]
    raise ValueError(f"Could not resolve CAS for supplemental NRTL component {name!r}")


def supplemental_nrtl_matrix_records(existing: list[dict]) -> tuple[list[dict], int]:
    path = SOURCE_DATA / "nrtl_temperature_matrix_interactions.json"
    if not path.exists():
        return [], 0

    payload = load_json(path)
    components = payload["components"]
    matrix = payload["nrtl_parameters"]
    disabled_pairs = {
        tuple(sorted(item.get("components", []))): item.get("reason", "")
        for item in payload.get("disabled_pairs", [])
    }
    cas_by_name = {
        name: resolve_matrix_component_cas(name)
        for name in components
    }
    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }

    records: list[dict] = []
    new_pairs = 0
    for i, comp_i in enumerate(components):
        for comp_j in components[i + 1:]:
            forward = matrix[comp_i][comp_j]
            reverse = matrix[comp_j][comp_i]
            cas_i = cas_by_name[comp_i]
            cas_j = cas_by_name[comp_j]
            pair = tuple(sorted((cas_i, cas_j)))
            if pair not in existing_pairs:
                new_pairs += 1
            record = {
                "cas1": cas_i,
                "cas2": cas_j,
                "component1": comp_i,
                "component2": comp_j,
                "tau12_c": float(forward["a"]),
                "tau12_d": float(forward["b"]),
                "tau12_e": 0.0,
                "tau12_f": 0.0,
                "tau21_c": float(reverse["a"]),
                "tau21_d": float(reverse["b"]),
                "tau21_e": 0.0,
                "tau21_f": 0.0,
                "alpha12": max(float(forward.get("c", 0.0)), float(reverse.get("c", 0.0))),
                "comment": f"{comp_i}/{comp_j} supplemental NRTL matrix; tau_ij = a_ij + b_ij/T",
                "source": payload.get("source", "supplemental NRTL matrix"),
            }
            disabled_reason = disabled_pairs.get(tuple(sorted((comp_i, comp_j))))
            if disabled_reason:
                record["disabled"] = True
                record["disabled_reason"] = disabled_reason
            records.append(record)
    return records, new_pairs


def supplemental_water_aromatic_regression_records(
    existing: list[dict],
    model: str,
) -> tuple[list[dict], int]:
    path = SOURCE_DATA / "water_aromatic_solubility_regression.json"
    if not path.exists():
        return [], 0

    payload = load_json(path)
    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    records: list[dict] = []
    new_pairs = 0
    model_key = model.upper()
    for system in payload.get("systems", []):
        params = system.get("fitted_parameters", {}).get(model_key)
        if not params:
            continue
        component1, component2 = system["components"]
        cas1, cas2 = system["cas"]
        pair = tuple(sorted((cas1, cas2)))
        if pair not in existing_pairs:
            new_pairs += 1
        record = {
            "cas1": cas1,
            "cas2": cas2,
            "component1": component1,
            "component2": component2,
            "Tmin_K": float(system["applicable_temperature_range_K"][0]),
            "Tmax_K": float(system["applicable_temperature_range_K"][1]),
            "source": payload["source"],
            "comment": (
                f"{component1}/{component2} water/aromatic solubility regression "
                f"from {payload['source']}"
            ),
        }
        if model_key == "NRTL":
            record.update({
                "alpha12": float(params["alpha12"]),
                "tau12_c": float(params["tau12_c"]),
                "tau12_d": float(params["tau12_d"]),
                "tau12_e": float(params.get("tau12_e", 0.0)),
                "tau12_f": float(params.get("tau12_f", 0.0)),
                "tau21_c": float(params["tau21_c"]),
                "tau21_d": float(params["tau21_d"]),
                "tau21_e": float(params.get("tau21_e", 0.0)),
                "tau21_f": float(params.get("tau21_f", 0.0)),
                "tau_tref": 298.15,
            })
        elif model_key == "UNIQUAC":
            record.update({
                "model_variant": "standard_uniquac",
                "use_q_prime": False,
                "tau12_a": float(params["tau12_a"]),
                "tau12_b": float(params["tau12_b"]),
                "tau12_c": float(params.get("tau12_c", 0.0)),
                "tau21_a": float(params["tau21_a"]),
                "tau21_b": float(params["tau21_b"]),
                "tau21_c": float(params.get("tau21_c", 0.0)),
            })
        else:
            continue
        records.append(record)
    return records, new_pairs


def supplemental_water_organic_binary_fit_records(
    existing: list[dict],
    model: str,
) -> tuple[list[dict], int, set[tuple[str, str]]]:
    """Build the curated water/organic interaction overlay for one model."""
    path = SOURCE_DATA / "water_organic_binary_fits.json"
    if not path.exists():
        return [], 0, set()

    payload = load_json(path)
    model_key = model.upper()
    source_key = model_key.lower()
    if model_key not in {"NRTL", "UNIQUAC"}:
        return [], 0, set()

    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    records: list[dict] = []
    covered_pairs: set[tuple[str, str]] = set()
    for system in payload.get("pairs", []):
        fit = system.get(source_key)
        if not fit or not fit.get("recommended", False):
            continue
        cas1, cas2 = system["cas1"], system["cas2"]
        pair = tuple(sorted((cas1, cas2)))
        if pair in covered_pairs:
            raise ValueError(
                f"Duplicate {model_key} water/organic fit for CAS pair {pair}"
            )
        covered_pairs.add(pair)
        temperature_range = system["temperature_range_K"]
        params = fit["parameters"]
        record = {
            "cas1": cas1,
            "cas2": cas2,
            "component1": system["component1"],
            "component2": system["component2"],
            "Tmin_K": float(temperature_range["Tmin"]),
            "Tmax_K": float(temperature_range["Tmax"]),
            "source": "water_organic_binary_fits.json",
            "source_file": "data/source/water_organic_binary_fits.json",
            "fit_status": system["fit_status"],
            "comment": (
                f"{system['component1']}/{system['component2']} curated "
                f"water/organic {model_key} regression"
            ),
        }
        if model_key == "NRTL":
            record.update({
                "alpha12": float(fit["alpha12"]),
                "tau12_c": float(params["tau12_c"]),
                "tau12_d": float(params["tau12_d"]),
                "tau12_e": float(params.get("tau12_e", 0.0)),
                "tau12_f": float(params.get("tau12_f", 0.0)),
                "tau21_c": float(params["tau21_c"]),
                "tau21_d": float(params["tau21_d"]),
                "tau21_e": float(params.get("tau21_e", 0.0)),
                "tau21_f": float(params.get("tau21_f", 0.0)),
                "tau_tref": float(fit.get("tau_tref_K", 298.15)),
            })
        else:
            record.update({
                "model_variant": fit.get("model_variant", "standard_uniquac"),
                "use_q_prime": bool(fit.get("use_q_prime", False)),
                "tau12_a": float(params["tau12_a"]),
                "tau12_b": float(params["tau12_b"]),
                "tau12_c": float(params.get("tau12_c", 0.0)),
                "tau21_a": float(params["tau21_a"]),
                "tau21_b": float(params["tau21_b"]),
                "tau21_c": float(params.get("tau21_c", 0.0)),
            })
        records.append(record)

    new_pairs = len(covered_pairs - existing_pairs)
    return records, new_pairs, covered_pairs


def supplemental_literature_vle_activity_records(
    existing: list[dict],
    model: str,
) -> tuple[list[dict], int, set[tuple[str, str]]]:
    """Build the final curated overlay from the literature VLE fit payloads."""
    model_key = model.upper()
    if model_key not in {"NRTL", "UNIQUAC"}:
        return [], 0, set()

    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    records: list[dict] = []
    covered_pairs: set[tuple[str, str]] = set()

    def add_record(
        *,
        cas1: str,
        cas2: str,
        component1: str,
        component2: str,
        source: str,
        parameters: dict,
        comment: str,
        temperature_range: tuple[float, float] | None = None,
        fit_status: str = "recommended_literature_interaction",
    ) -> None:
        pair = tuple(sorted((cas1, cas2)))
        if pair in covered_pairs:
            raise ValueError(
                f"Duplicate curated {model_key} literature fit for CAS pair {pair}"
            )
        covered_pairs.add(pair)
        record = {
            "cas1": cas1,
            "cas2": cas2,
            "component1": component1,
            "component2": component2,
            "source": source,
            "source_file": f"data/source/{source}",
            "fit_status": fit_status,
            "comment": comment,
        }
        if temperature_range is not None:
            record.update({
                "Tmin_K": float(temperature_range[0]),
                "Tmax_K": float(temperature_range[1]),
            })
        if model_key == "NRTL":
            record.update({
                "alpha12": float(parameters["alpha12"]),
                "tau12_c": float(parameters["tau12_c"]),
                "tau12_d": float(parameters.get("tau12_d", 0.0)),
                "tau12_e": float(parameters.get("tau12_e", 0.0)),
                "tau12_f": float(parameters.get("tau12_f", 0.0)),
                "tau21_c": float(parameters["tau21_c"]),
                "tau21_d": float(parameters.get("tau21_d", 0.0)),
                "tau21_e": float(parameters.get("tau21_e", 0.0)),
                "tau21_f": float(parameters.get("tau21_f", 0.0)),
                "tau_tref": float(parameters.get("tau_tref", 298.15)),
            })
        else:
            record.update({
                "model_variant": parameters.get(
                    "model_variant", "standard_uniquac"
                ),
                "use_q_prime": bool(parameters.get("use_q_prime", False)),
                "tau12_a": float(parameters["tau12_a"]),
                "tau12_b": float(parameters.get("tau12_b", 0.0)),
                "tau12_c": float(parameters.get("tau12_c", 0.0)),
                "tau21_a": float(parameters["tau21_a"]),
                "tau21_b": float(parameters.get("tau21_b", 0.0)),
                "tau21_c": float(parameters.get("tau21_c", 0.0)),
            })
        records.append(record)

    acetaldehyde_file = "acetaldehyde_water_vle.json"
    acetaldehyde = load_json(SOURCE_DATA / acetaldehyde_file)
    acetaldehyde_range = tuple(
        acetaldehyde["system"]["recommended_temperature_range_K"]
    )
    if model_key == "UNIQUAC":
        fit = acetaldehyde["preferred_fit"]
        add_record(
            cas1="75-07-0",
            cas2="7732-18-5",
            component1="Acetaldehyde",
            component2="Water",
            source=acetaldehyde_file,
            parameters={
                "tau12_a": fit["A_12"],
                "tau12_b": fit["B_12_K"],
                "tau21_a": fit["A_21"],
                "tau21_b": fit["B_21_K"],
            },
            comment=(
                "Acetaldehyde/Water source-balanced temperature-dependent "
                "UNIQUAC literature regression"
            ),
            temperature_range=acetaldehyde_range,
            fit_status="recommended_effective_pseudobinary_regression",
        )
    else:
        nrtl_payload = acetaldehyde["recommended_NRTL_fit"]
        fit = nrtl_payload["preferred_fit"]
        add_record(
            cas1="75-07-0",
            cas2="7732-18-5",
            component1="Acetaldehyde",
            component2="Water",
            source=acetaldehyde_file,
            parameters={
                "alpha12": fit["alpha_12"],
                "tau12_c": fit["A_12"],
                "tau12_d": fit["B_12_K"],
                "tau21_c": fit["A_21"],
                "tau21_d": fit["B_21_K"],
            },
            comment=(
                "Acetaldehyde/Water source-balanced temperature-dependent "
                "NRTL literature regression"
            ),
            temperature_range=tuple(
                nrtl_payload["system"]["recommended_temperature_range_K"]
            ),
            fit_status="recommended_effective_pseudobinary_regression",
        )

    acid_file = "three_acids_vle.json"
    acid_payload = load_json(SOURCE_DATA / acid_file)
    acid_cas = {
        "Acetic acid": "64-19-7",
        "Propionic acid": "79-09-4",
        "Acrylic acid": "79-10-7",
    }
    for system in acid_payload["systems"]:
        component1 = system["component_1"]["name"]
        component2 = system["component_2"]["name"]
        fit = system["models"][model_key]
        raw = fit["parameters"]
        if model_key == "NRTL":
            parameters = {
                "alpha12": fit["alpha"],
                "tau12_c": 0.0,
                "tau12_d": raw["A_12_K"],
                "tau21_c": 0.0,
                "tau21_d": raw["A_21_K"],
            }
        else:
            parameters = {
                "tau12_a": 0.0,
                "tau12_b": -float(raw["A_12_K"]),
                "tau21_a": 0.0,
                "tau21_b": -float(raw["A_21_K"]),
            }

        fit_source = system.get("fit_source")
        temperature_range = None
        if fit_source and "temperature_K" in fit_source:
            temperature = float(fit_source["temperature_K"])
            temperature_range = (temperature, temperature)
        elif fit_source and fit_source.get("points"):
            temperatures = [
                float(point[2]) + 273.15 for point in fit_source["points"]
            ]
            temperature_range = (min(temperatures), max(temperatures))

        is_defensible_zero = (
            component1 == "Propionic acid" and component2 == "Acrylic acid"
        )
        add_record(
            cas1=acid_cas[component1],
            cas2=acid_cas[component2],
            component1=component1,
            component2=component2,
            source=acid_file,
            parameters=parameters,
            comment=(
                f"{component1}/{component2} defensible zero residual {model_key} "
                "interaction after statistical vapor cross-association"
                if is_defensible_zero
                else f"{component1}/{component2} fitted residual {model_key} "
                "interaction with statistical vapor cross-association"
            ),
            temperature_range=temperature_range,
            fit_status=(
                "recommended_defensible_zero_interaction"
                if is_defensible_zero
                else "recommended_literature_interaction"
            ),
        )
        if is_defensible_zero:
            records[-1]["comment"] += (
                "; expected zero-interaction error is well below the "
                "experimental error margin (about 1-2% activity-coefficient "
                "nonideality or less)"
            )

    water_acid_file = "water_c3_acid_vle_interactions.json"
    water_acid_payload = load_json(SOURCE_DATA / water_acid_file)
    water_acid_cas = {
        "water_acrylic_acid": "79-10-7",
        "water_propionic_acid": "79-09-4",
    }
    water_acid_names = {
        "water_acrylic_acid": "Acrylic acid",
        "water_propionic_acid": "Propionic acid",
    }
    water_acid_ranges = {
        "water_acrylic_acid": tuple(
            water_acid_payload["systems"]["water_acrylic_acid"]["metadata"]
            ["fit_basis"]["temperature_range_K"]
        ),
        "water_propionic_acid": (313.15, 373.15),
    }
    for system_key, system in water_acid_payload["systems"].items():
        fit_key = f"{model_key}-VDM"
        fit = system["fitted_parameters"][fit_key]
        raw = fit["parameters"]
        if model_key == "NRTL":
            parameters = {
                "alpha12": raw["alpha12"],
                "tau12_c": raw["tau12_c"],
                "tau12_d": raw["tau12_d"],
                "tau12_e": raw.get("tau12_e", 0.0),
                "tau12_f": raw.get("tau12_f", 0.0),
                "tau21_c": raw["tau21_c"],
                "tau21_d": raw["tau21_d"],
                "tau21_e": raw.get("tau21_e", 0.0),
                "tau21_f": raw.get("tau21_f", 0.0),
                "tau_tref": raw.get("tau_tref", 298.15),
            }
        else:
            parameters = {
                "model_variant": fit.get("model_variant", "standard_uniquac"),
                "use_q_prime": fit.get("use_q_prime", False),
                "tau12_a": raw["tau12_a"],
                "tau12_b": raw["tau12_b"],
                "tau12_c": raw.get("tau12_c", 0.0),
                "tau21_a": raw["tau21_a"],
                "tau21_b": raw["tau21_b"],
                "tau21_c": raw.get("tau21_c", 0.0),
            }
        acid_name = water_acid_names[system_key]
        add_record(
            cas1="7732-18-5",
            cas2=water_acid_cas[system_key],
            component1="Water",
            component2=acid_name,
            source=water_acid_file,
            parameters=parameters,
            comment=(
                f"Water/{acid_name} temperature-dependent {model_key} "
                "literature regression with fixed VDM"
            ),
            temperature_range=water_acid_ranges[system_key],
            fit_status=(
                "recommended_literature_interaction"
                if fit["recommended"]
                else "retained_literature_alternative"
            ),
        )

    return records, len(covered_pairs - existing_pairs), covered_pairs


def supplemental_assorted_alcohol_ether_records(
    existing: list[dict],
    model: str,
) -> tuple[list[dict], int, set[tuple[str, str]]]:
    """Build recommended alcohol/ether fits and evidence-backed zeroes."""
    path = SOURCE_DATA / "assorted_alcohols_ethers.json"
    if not path.exists():
        return [], 0, set()
    model_key = model.upper()
    source_key = model_key.lower()
    if model_key not in {"NRTL", "UNIQUAC"}:
        return [], 0, set()

    payload = load_json(path)
    excluded_pair = frozenset(("2-propanol", "water"))
    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    records: list[dict] = []
    covered_pairs: set[tuple[str, str]] = set()

    def pair_identity(components: list[str]) -> tuple[str, str, tuple[str, str]]:
        component1, component2 = components
        try:
            cas1 = ASSORTED_ALCOHOL_ETHER_CAS[component1]
            cas2 = ASSORTED_ALCOHOL_ETHER_CAS[component2]
        except KeyError as error:
            raise ValueError(
                f"Missing CAS mapping for assorted alcohol/ether component {error.args[0]!r}"
            ) from error
        return cas1, cas2, tuple(sorted((cas1, cas2)))

    for entry in payload["entries"]:
        components = entry["components"]
        if frozenset(components) == excluded_pair:
            continue
        parameters = entry.get(source_key)
        if not (
            entry.get("recommended", False)
            and parameters
            and parameters.get("recommended", False)
        ):
            continue
        cas1, cas2, pair = pair_identity(components)
        if pair in covered_pairs:
            raise ValueError(
                f"Duplicate recommended {model_key} assorted fit for CAS pair {pair}"
            )
        covered_pairs.add(pair)
        validity = entry["validity"]
        source_ids = list(
            entry.get("source_ids")
            or entry.get("sources_used_for_refit")
            or entry.get("sources")
            or []
        )
        record = {
            "cas1": cas1,
            "cas2": cas2,
            "component1": components[0],
            "component2": components[1],
            "Tmin_K": float(validity["T_K"][0]),
            "Tmax_K": float(validity["T_K"][1]),
            "source": (
                "assorted_alcohols_ethers.json"
                + (f"; {', '.join(source_ids)}" if source_ids else "")
            ),
            "source_file": "data/source/assorted_alcohols_ethers.json",
            "fit_status": "recommended_literature_interaction",
            "fit_vapor_treatment": {
                "type": "likely_HOC_or_equivalent_association_correction",
                "certainty": "inferred_not_confirmed_for_every_source",
                "note": (
                    "The underlying alcohol/ether VLE reductions likely used "
                    "Hayden-O'Connell or an equivalent association-aware vapor "
                    "treatment, but this is not confirmed uniformly across all "
                    "contributing sources."
                ),
            },
            "comment": (
                f"{components[0]}/{components[1]} recommended temperature-dependent "
                f"{model_key} literature interaction"
            ),
        }
        if model_key == "NRTL":
            record.update({
                "alpha12": float(parameters["alpha"]),
                "tau12_c": float(parameters["a12"]),
                "tau12_d": float(parameters["b12_K"]),
                "tau12_e": 0.0,
                "tau12_f": float(parameters.get("c12_per_K", 0.0)),
                "tau21_c": float(parameters["a21"]),
                "tau21_d": float(parameters["b21_K"]),
                "tau21_e": 0.0,
                "tau21_f": float(parameters.get("c21_per_K", 0.0)),
                "tau_tref": 298.15,
            })
        else:
            if parameters.get("model_variant") != "standard_uniquac":
                raise ValueError(
                    f"Recommended nonstandard UNIQUAC entry was not excluded: {components}"
                )
            record.update({
                "model_variant": "standard_uniquac",
                "use_q_prime": False,
                "tau12_a": float(parameters["a12"]),
                "tau12_b": float(parameters["b12_K"]),
                "tau12_c": float(parameters.get("c12_per_K", 0.0)),
                "tau21_a": float(parameters["a21"]),
                "tau21_b": float(parameters["b21_K"]),
                "tau21_c": float(parameters.get("c21_per_K", 0.0)),
            })
        records.append(record)

    for entry in payload["retired_or_not_promoted"]:
        if "use_zero_interaction" not in entry.get("status", ""):
            continue
        components = entry["components"]
        cas1, cas2, pair = pair_identity(components)
        if pair in covered_pairs:
            raise ValueError(
                f"Defensible zero duplicates an active {model_key} assorted fit for {pair}"
            )
        covered_pairs.add(pair)
        source_ids = list(entry.get("source_ids", []))
        record = {
            "cas1": cas1,
            "cas2": cas2,
            "component1": components[0],
            "component2": components[1],
            "source": (
                "assorted_alcohols_ethers.json"
                + (f"; {', '.join(source_ids)}" if source_ids else "")
            ),
            "source_file": "data/source/assorted_alcohols_ethers.json",
            "fit_status": "recommended_defensible_zero_interaction",
            "fit_vapor_treatment": {
                "type": "likely_HOC_or_equivalent_association_correction",
                "certainty": "inferred_not_confirmed_for_every_source",
                "note": (
                    "The underlying VLE evidence likely used Hayden-O'Connell "
                    "or an equivalent association-aware vapor treatment, but "
                    "this is not confirmed uniformly across all sources."
                ),
            },
            "comment": (
                f"{components[0]}/{components[1]} defensible zero {model_key} "
                f"interaction; {entry['reason']}"
            ),
        }
        if model_key == "NRTL":
            record.update({
                "alpha12": 0.3,
                "tau12_c": 0.0,
                "tau12_d": 0.0,
                "tau12_e": 0.0,
                "tau12_f": 0.0,
                "tau21_c": 0.0,
                "tau21_d": 0.0,
                "tau21_e": 0.0,
                "tau21_f": 0.0,
                "tau_tref": 298.15,
            })
        else:
            record.update({
                "model_variant": "standard_uniquac",
                "use_q_prime": False,
                "tau12_a": 0.0,
                "tau12_b": 0.0,
                "tau12_c": 0.0,
                "tau21_a": 0.0,
                "tau21_b": 0.0,
                "tau21_c": 0.0,
            })
        records.append(record)

    return records, len(covered_pairs - existing_pairs), covered_pairs


def supplemental_isopropanol_water_records(
    existing: list[dict],
    model: str,
) -> tuple[list[dict], int, set[tuple[str, str]]]:
    """Build the curated broad-range 2-propanol/water interaction fit."""
    path = SOURCE_DATA / "isopropanol_water_interactions.json"
    if not path.exists():
        return [], 0, set()
    model_key = model.upper()
    if model_key not in {"NRTL", "UNIQUAC"}:
        return [], 0, set()

    payload = load_json(path)
    component_order = payload["system"]["component_order"]
    component1 = component_order["1"]
    component2 = component_order["2"]
    cas1, cas2 = component1["cas"], component2["cas"]
    pair = tuple(sorted((cas1, cas2)))
    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    model_data = payload["models"][model_key]
    if not model_data.get("selected", False):
        return [], 0, set()

    validity = payload["recommended_validity"]
    fit_sources = [
        source["source"] for source in payload["fit_provenance"]["common_dataset"]
    ]
    record = {
        "cas1": cas1,
        "cas2": cas2,
        "component1": component1["name"],
        "component2": component2["name"],
        "Tmin_K": float(validity["temperature_min_K"]),
        "Tmax_K": float(validity["temperature_max_K"]),
        "source": (
            "isopropanol_water_interactions.json; " + ", ".join(fit_sources)
        ),
        "source_file": "data/source/isopropanol_water_interactions.json",
        "fit_status": "recommended_broad_range_literature_interaction",
        "fit_vapor_treatment": payload["vapor_phase_treatment"],
        "comment": (
            f"2-propanol/Water broad-range temperature-dependent {model_key} "
            "fit with truncated-second-virial vapor correction "
            "(B12=-220 cm^3/mol)"
        ),
    }
    parameters = model_data["parameters"]
    forward = parameters["1_to_2"]
    reverse = parameters["2_to_1"]
    if model_key == "NRTL":
        alpha12 = float(parameters["alpha_12"])
        alpha21 = float(parameters["alpha_21"])
        if not math.isclose(alpha12, alpha21):
            raise ValueError(
                "PFDSim's NRTL runtime requires symmetric alpha for the "
                "2-propanol/water fit"
            )
        record.update({
            "alpha12": alpha12,
            "tau12_c": float(forward["a"]),
            "tau12_d": float(forward["b"]),
            "tau12_e": 0.0,
            "tau12_f": float(forward["c"]),
            "tau21_c": float(reverse["a"]),
            "tau21_d": float(reverse["b"]),
            "tau21_e": 0.0,
            "tau21_f": float(reverse["c"]),
            "tau_tref": 298.15,
        })
    else:
        record.update({
            "model_variant": "standard_uniquac",
            "use_q_prime": False,
            "tau12_a": float(forward["a"]),
            "tau12_b": float(forward["b"]),
            "tau12_c": float(forward["c"]),
            "tau21_a": float(reverse["a"]),
            "tau21_b": float(reverse["b"]),
            "tau21_c": float(reverse["c"]),
        })
    return [record], 0 if pair in existing_pairs else 1, {pair}


def supplemental_phenolic_temperature_records(
    existing: list[dict],
    model: str,
) -> tuple[list[dict], int]:
    path = SOURCE_DATA / "phenolic_temperature_interactions.json"
    if not path.exists():
        return [], 0

    payload = load_json(path)
    tref = float(payload["metadata"]["temperature_reference_K"])
    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    records: list[dict] = []
    new_pairs = 0
    model_key = model.upper()
    for system in payload.get("interactions", []):
        params = system.get("models", {}).get(model_key)
        if not params:
            continue
        component1, component2 = system["components"]
        cas1, cas2 = system["cas"]
        pair = tuple(sorted((cas1, cas2)))
        if pair not in existing_pairs:
            new_pairs += 1
        c12c = float(params["C12C_K"])
        c21c = float(params["C21C_K"])
        c12t = float(params["C12T"])
        c21t = float(params["C21T"])
        record = {
            "cas1": cas1,
            "cas2": cas2,
            "component1": component1,
            "component2": component2,
            "source": payload["metadata"]["source"],
            "comment": (
                f"{component1}/{component2} phenolic temperature-dependent "
                f"{model_key} Table 5; Cij = CijC + CijT*(T - 273.15 K)"
            ),
        }
        if model_key == "NRTL":
            record.update({
                "alpha12": float(params["alpha12"]),
                "tau12_c": c12t,
                "tau12_d": c12c - tref * c12t,
                "tau12_e": 0.0,
                "tau12_f": 0.0,
                "tau21_c": c21t,
                "tau21_d": c21c - tref * c21t,
                "tau21_e": 0.0,
                "tau21_f": 0.0,
                "tau_tref": tref,
            })
        elif model_key == "UNIQUAC":
            record.update({
                "model_variant": "standard_uniquac",
                "use_q_prime": False,
                "tau12_a": -c12t,
                "tau12_b": -c12c + tref * c12t,
                "tau12_c": 0.0,
                "tau21_a": -c21t,
                "tau21_b": -c21c + tref * c21t,
                "tau21_c": 0.0,
            })
        else:
            continue
        records.append(record)
    return records, new_pairs


def supplemental_cesari_phenolic_nrtl_records(existing: list[dict]) -> tuple[list[dict], int]:
    path = SOURCE_DATA / "cesari_phenolic_nrtl_interactions.json"
    if not path.exists():
        return [], 0

    payload = load_json(path)
    r_j_per_mol_k = 8.31446261815324
    alpha = float(payload["metadata"]["alpha12"])
    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    records: list[dict] = []
    new_pairs = 0
    for system in payload.get("interactions", []):
        if system.get("emit_runtime") is False:
            continue
        component1, component2 = system["components"]
        cas1, cas2 = system["cas"]
        pair = tuple(sorted((cas1, cas2)))
        if pair not in existing_pairs:
            new_pairs += 1
        records.append({
            "cas1": cas1,
            "cas2": cas2,
            "component1": component1,
            "component2": component2,
            "source": payload["metadata"]["source"],
            "alpha12": alpha,
            "tau12_c": float(system["b12_J_per_mol_K"]) / r_j_per_mol_k,
            "tau12_d": float(system["a12_J_per_mol"]) / r_j_per_mol_k,
            "tau12_e": 0.0,
            "tau12_f": 0.0,
            "tau21_c": float(system["b21_J_per_mol_K"]) / r_j_per_mol_k,
            "tau21_d": float(system["a21_J_per_mol"]) / r_j_per_mol_k,
            "tau21_e": 0.0,
            "tau21_f": 0.0,
            "tau_tref": 298.15,
            "comment": (
                f"{component1}/{component2} Cesari phenolic NRTL Table {system['table']}; "
                "Delta_g_ij = a_ij + b_ij*T"
            ),
        })
    return records, new_pairs


def supplemental_diethyl_ether_water_records(
    existing: list[dict],
    model: str,
) -> tuple[list[dict], int]:
    path = SOURCE_DATA / "diethyl_ether_water_35c_fit.json"
    if not path.exists():
        return [], 0

    payload = load_json(path)
    model_key = model.upper()
    params = payload.get("fitted_parameters", {}).get(model_key)
    if not params:
        return [], 0

    cas1, cas2 = payload["metadata"]["cas"]
    component1, component2 = payload["metadata"]["components"]
    pair = tuple(sorted((cas1, cas2)))
    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    record = {
        "cas1": cas1,
        "cas2": cas2,
        "component1": component1,
        "component2": component2,
        "source": payload["metadata"]["source"],
        "comment": (
            f"{component1}/{component2} 35 C VLLE + HE fitted {model_key}; "
            f"{payload['metadata']['source']}"
        ),
    }
    if model_key == "NRTL":
        record.update({
            "alpha12": float(params["alpha12"]),
            "tau12_c": float(params["tau12_c"]),
            "tau12_d": float(params["tau12_d"]),
            "tau12_e": float(params.get("tau12_e", 0.0)),
            "tau12_f": float(params.get("tau12_f", 0.0)),
            "tau21_c": float(params["tau21_c"]),
            "tau21_d": float(params["tau21_d"]),
            "tau21_e": float(params.get("tau21_e", 0.0)),
            "tau21_f": float(params.get("tau21_f", 0.0)),
            "tau_tref": float(params.get("tau_tref", payload["metadata"]["temperature_K"])),
        })
    elif model_key == "UNIQUAC":
        record.update({
            "model_variant": "standard_uniquac",
            "use_q_prime": False,
            "tau12_a": float(params["tau12_a"]),
            "tau12_b": float(params["tau12_b"]),
            "tau12_c": float(params.get("tau12_c", 0.0)),
            "tau21_a": float(params["tau21_a"]),
            "tau21_b": float(params["tau21_b"]),
            "tau21_c": float(params.get("tau21_c", 0.0)),
        })
    else:
        return [], 0
    return [record], 0 if pair in existing_pairs else 1


def supplemental_1_butanol_water_records(
    existing: list[dict],
    model: str,
) -> tuple[list[dict], int]:
    path = SOURCE_DATA / "1_butanol_water_interactions.json"
    if not path.exists():
        return [], 0

    payload = load_json(path)
    model_key = model.upper()
    params = payload.get("parameters", {}).get(model_key)
    if not params:
        return [], 0

    metadata = payload["metadata"]
    cas1, cas2 = metadata["cas"]
    component1, component2 = metadata["components"]
    pair = tuple(sorted((cas1, cas2)))
    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    record = {
        "cas1": cas1,
        "cas2": cas2,
        "component1": component1,
        "component2": component2,
        "source": metadata["source"],
        "comment": f"{component1}/{component2} temperature-dependent {model_key}",
    }
    if model_key == "NRTL":
        record.update({
            "alpha12": float(params["alpha12"]),
            "tau12_c": float(params["tau12_c"]),
            "tau12_d": float(params["tau12_d"]),
            "tau12_e": float(params["tau12_e"]),
            "tau12_f": float(params.get("tau12_f", 0.0)),
            "tau21_c": float(params["tau21_c"]),
            "tau21_d": float(params["tau21_d"]),
            "tau21_e": float(params["tau21_e"]),
            "tau21_f": float(params.get("tau21_f", 0.0)),
            "tau_tref": float(params["tau_tref"]),
        })
    elif model_key == "UNIQUAC":
        record.update({
            "model_variant": "standard_uniquac",
            "use_q_prime": False,
            "tau12_a": float(params["tau12_a"]),
            "tau12_b": float(params["tau12_b"]),
            "tau12_c": float(params["tau12_c"]),
            "tau21_a": float(params["tau21_a"]),
            "tau21_b": float(params["tau21_b"]),
            "tau21_c": float(params["tau21_c"]),
        })
    else:
        return [], 0
    return [record], 0 if pair in existing_pairs else 1


def supplemental_acetic_acid_vle_records(
    existing: list[dict],
    model: str,
) -> tuple[list[dict], int]:
    path = SOURCE_DATA / "acetic_acid_vle_interactions.json"
    if not path.exists():
        return [], 0

    payload = load_json(path)
    metadata = payload["metadata"]
    model_key = model.upper()
    component1 = metadata["component1"]
    cas1 = metadata["cas1"]
    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    records: list[dict] = []
    new_pairs = 0
    for system in payload["systems"]:
        params = system.get("parameters", {}).get(model_key)
        if not params:
            continue
        component2 = system["component2"]
        cas2 = system["cas2"]
        pair = tuple(sorted((cas1, cas2)))
        if pair not in existing_pairs:
            new_pairs += 1
        record = {
            "cas1": cas1,
            "cas2": cas2,
            "component1": component1,
            "component2": component2,
            "Tmin_K": float(system["temperature_range_K"][0]),
            "Tmax_K": float(system["temperature_range_K"][1]),
            "source": system["source"],
            "comment": (
                f"{component1}/{component2} fitted {model_key} with fixed VDM; "
                f"{system['data_basis']}; {system['source']}"
            ),
            "a12_cal_per_mol": float(params["a12_cal_per_mol"]),
            "a21_cal_per_mol": float(params["a21_cal_per_mol"]),
        }
        if model_key == "NRTL":
            record["alpha12"] = float(params["alpha12"])
        elif model_key == "UNIQUAC":
            record["model_variant"] = params.get(
                "model_variant", "standard_uniquac"
            )
            record["use_q_prime"] = bool(params.get("use_q_prime", False))
        else:
            continue
        records.append(record)
        existing_pairs.add(pair)
    return records, new_pairs


def supplemental_ester_alcohol_fit_records(
    existing: list[dict],
    model: str,
) -> tuple[list[dict], int]:
    path = SOURCE_DATA / "ester_alcohol_nrtl_uniquac_recommended_fits.json"
    if not path.exists():
        return [], 0

    payload = load_json(path)
    metadata = payload["metadata"]
    component_cas = metadata["component_cas"]
    model_key = model.upper()
    if model_key not in {"NRTL", "UNIQUAC"}:
        return [], 0
    direct_rq = load_json(DATA / "uniquac_rq_cas.json").get("components", {})
    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    records: list[dict] = []
    new_pairs: set[tuple[str, str]] = set()
    for fit in payload.get("recommended_fits", []):
        if fit["model"].upper() != model_key:
            continue
        if fit["form"] not in {"B/T", "A+B/T"}:
            raise ValueError(
                f"Unsupported ester interaction form {fit['form']!r} for "
                f"{fit['pair']} {model_key}"
            )
        unsupported = (
            float(fit["C12"]),
            float(fit["D12_per_K"]),
            float(fit["C21"]),
            float(fit["D21_per_K"]),
        )
        if any(value != 0.0 for value in unsupported):
            raise ValueError(
                f"Ester interaction {fit['pair']} {model_key} uses unsupported "
                "nonzero C*ln(T) or D*T terms"
            )
        component1 = fit["component_1"]
        component2 = fit["component_2"]
        try:
            cas1 = component_cas[component1]
            cas2 = component_cas[component2]
        except KeyError as error:
            raise ValueError(
                f"Missing ester interaction CAS for {error.args[0]!r}"
            ) from error
        if model_key == "UNIQUAC":
            missing_rq = [cas for cas in (cas1, cas2) if cas not in direct_rq]
            if missing_rq:
                raise ValueError(
                    f"Missing direct UNIQUAC r/q for ester interaction CAS: "
                    + ", ".join(missing_rq)
                )
        pair = tuple(sorted((cas1, cas2)))
        if pair not in existing_pairs:
            new_pairs.add(pair)
        record = {
            "cas1": cas1,
            "cas2": cas2,
            "component1": component1,
            "component2": component2,
            "Tmin_K": float(fit["Tmin_K"]),
            "Tmax_K": float(fit["Tmax_K"]),
            "Pmin_kPa": float(fit["Pmin_kPa"]),
            "Pmax_kPa": float(fit["Pmax_kPa"]),
            "fit_form": fit["form"],
            "fit_vapor_treatment": fit["vapor_treatment"],
            "fit_points": int(fit["n_points"]),
            "fit_sources": int(fit["n_sources"]),
            "fit_pressure_AARD_percent": float(fit["P_AARD_percent"]),
            "fit_vapor_composition_AAD": float(fit["y1_AAD"]),
            "selection_reason": fit["selection_reason"],
            "source": "ester_alcohol_nrtl_uniquac_fit_report.txt",
            "comment": (
                f"{fit['pair']} recommended {model_key} {fit['form']} fit; "
                f"vapor={fit['vapor_treatment']}; n={fit['n_points']} from "
                f"{fit['n_sources']} source(s)"
            ),
        }
        if model_key == "NRTL":
            record.update({
                "alpha12": float(fit["alpha"]),
                "tau12_c": float(fit["A12"]),
                "tau12_d": float(fit["B12_K"]),
                "tau12_e": 0.0,
                "tau12_f": 0.0,
                "tau21_c": float(fit["A21"]),
                "tau21_d": float(fit["B21_K"]),
                "tau21_e": 0.0,
                "tau21_f": 0.0,
                "tau_tref": 298.15,
            })
        else:
            record.update({
                "model_variant": "standard_uniquac",
                "use_q_prime": False,
                "tau12_a": -float(fit["A12"]),
                "tau12_b": -float(fit["B12_K"]),
                "tau12_c": 0.0,
                "tau21_a": -float(fit["A21"]),
                "tau21_b": -float(fit["B21_K"]),
                "tau21_c": 0.0,
            })
        records.append(record)
    return records, len(new_pairs)


def supplemental_water_ethylene_oxide_records(
    existing: list[dict],
    models: tuple[str, ...],
) -> tuple[list[dict], int]:
    path = SOURCE_DATA / "water_ethylene_oxide_interactions.json"
    if not path.exists():
        return [], 0

    payload = load_json(path)
    metadata = payload["metadata"]
    component1, component2 = metadata["components"]
    cas1, cas2 = metadata["cas"]
    pair = tuple(sorted((cas1, cas2)))
    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    parameters = payload["fitted_parameters"]
    records: list[dict] = []
    for model in models:
        model_key = model.upper()
        params = parameters.get(model_key)
        if not params:
            continue
        record = {
            "cas1": cas1,
            "cas2": cas2,
            "component1": component1,
            "component2": component2,
            "source": metadata["source"],
            "comment": (
                f"{component1}/{component2} fitted {model_key}; "
                f"{metadata['source']}"
            ),
        }
        if model_key == "NRTL":
            record.update({
                "Tmin_K": float(metadata["activity_temperature_range_K"][0]),
                "Tmax_K": float(metadata["activity_temperature_range_K"][1]),
                "alpha12": float(params["alpha12"]),
                "tau12_c": float(params["tau12_c"]),
                "tau12_d": float(params["tau12_d"]),
                "tau12_e": float(params.get("tau12_e", 0.0)),
                "tau12_f": float(params.get("tau12_f", 0.0)),
                "tau21_c": float(params["tau21_c"]),
                "tau21_d": float(params["tau21_d"]),
                "tau21_e": float(params.get("tau21_e", 0.0)),
                "tau21_f": float(params.get("tau21_f", 0.0)),
                "tau_tref": float(params.get("tau_tref", 298.15)),
            })
        elif model_key == "UNIQUAC":
            record.update({
                "Tmin_K": float(metadata["activity_temperature_range_K"][0]),
                "Tmax_K": float(metadata["activity_temperature_range_K"][1]),
                "model_variant": params.get("model_variant", "standard_uniquac"),
                "use_q_prime": bool(params.get("use_q_prime", False)),
                "tau12_a": float(params["tau12_a"]),
                "tau12_b": float(params["tau12_b"]),
                "tau12_c": float(params.get("tau12_c", 0.0)),
                "tau21_a": float(params["tau21_a"]),
                "tau21_b": float(params["tau21_b"]),
                "tau21_c": float(params.get("tau21_c", 0.0)),
            })
        elif model_key in {"PR", "SRK"}:
            record.update({
                "model": model_key,
                "kij": float(params["kij"]),
                "Tmin_K": float(metadata["eos_temperature_range_K"][0]),
                "Tmax_K": float(metadata["eos_temperature_range_K"][1]),
            })
        else:
            continue
        records.append(record)
    return records, 0 if pair in existing_pairs else 1


def supplemental_extended_uniquac_records(existing: list[dict]) -> tuple[list[dict], int]:
    path = SOURCE_DATA / "nagata_gmehling_extended_uniquac_interactions.json"
    if not path.exists():
        return [], 0

    payload = load_json(path)
    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }

    records: list[dict] = []
    new_pairs = 0
    for source_record in payload["interactions"]:
        comp_i = source_record["component1"]
        comp_j = source_record["component2"]
        cas_i = resolve_matrix_component_cas(comp_i)
        cas_j = resolve_matrix_component_cas(comp_j)
        pair = tuple(sorted((cas_i, cas_j)))
        if pair not in existing_pairs:
            new_pairs += 1
        records.append({
            "cas1": cas_i,
            "cas2": cas_j,
            "component1": comp_i,
            "component2": comp_j,
            "disabled": True,
            "disabled_reason": (
                "Extended UNIQUAC parameters from this paper do not reproduce "
                "known azeotropes when used in the standard UNIQUAC runtime path."
            ),
            "model_variant": "extended_uniquac",
            "tau12_a": -float(source_record["B12"]),
            "tau12_b": -float(source_record["A12"]),
            "tau12_c": -float(source_record["C12"]),
            "tau21_a": -float(source_record["B21"]),
            "tau21_b": -float(source_record["A21"]),
            "tau21_c": -float(source_record["C21"]),
            "use_q_prime": True,
            "tau_expression": payload["metadata"]["tau_expression"],
            "source": payload["metadata"]["source"],
            "comment": f"{comp_i}/{comp_j} extended UNIQUAC Table 2; tau_ij = exp(-B_ij - A_ij/T - C_ij*T)",
        })
    return records, new_pairs


def ipd_component_names(comment: str) -> tuple[str, str]:
    if "/" not in comment:
        return "", ""
    left, right = comment.split("/", 1)
    return clean_comment_side(left), clean_comment_side(right)


def parse_ipd_records(model: str, filename: str) -> list[dict]:
    path = SOURCE_DATA / filename
    if not path.exists():
        return []

    records: list[dict] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "[", "-")) or "=" in stripped[:20]:
            continue
        parts = stripped.split()
        if len(parts) < 3 or not (CAS_RE.match(parts[0]) and CAS_RE.match(parts[1])):
            continue

        cas1, cas2 = parts[0], parts[1]
        try:
            if model in {"SRK", "PR"}:
                values = {"kij": float(parts[2]), "model": model}
                comment = " ".join(parts[3:])
            elif model == "NRTL":
                values = {
                    "a12_cal_per_mol": float(parts[2]),
                    "a21_cal_per_mol": float(parts[3]),
                    "alpha12": float(parts[4]),
                }
                comment = " ".join(parts[5:])
            elif model == "UNIQUAC":
                values = {
                    "a12_cal_per_mol": float(parts[2]),
                    "a21_cal_per_mol": float(parts[3]),
                }
                comment = " ".join(parts[4:])
            else:
                continue
        except (IndexError, ValueError):
            continue

        component1, component2 = ipd_component_names(comment)
        record = {
            "cas1": cas1,
            "cas2": cas2,
            "component1": component1,
            "component2": component2,
            "comment": comment,
            "source": f"DWSIM ChemSepIPD {filename}",
            "source_file": filename,
            "source_line": line_number,
            **values,
        }
        if model in {"SRK", "PR"}:
            ranged = temperature_range(comment)
            if ranged is not None:
                record["Tmin_K"], record["Tmax_K"] = ranged
        records.append(record)
    return records


def eos_record_hydration_key(record: dict) -> tuple[str, tuple[str, str], tuple[float | None, float | None] | None]:
    temperature_key = None
    if "Tmin_K" in record and "Tmax_K" in record:
        temperature_key = (float(record["Tmin_K"]), float(record["Tmax_K"]))
    return (
        str(record["model"]).upper(),
        tuple(sorted((record["cas1"], record["cas2"]))),
        temperature_key,
    )


def collapse_eos_duplicate_records(records: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    groups: dict[
        tuple[str, tuple[str, str], tuple[float | None, float | None] | None],
        list[int],
    ] = {}
    for index, record in enumerate(records):
        groups.setdefault(eos_record_hydration_key(record), []).append(index)

    duplicate_groups = [indexes for indexes in groups.values() if len(indexes) > 1]
    duplicate_indexes = {
        index
        for indexes in duplicate_groups
        for index in indexes
    }
    collapsed_by_first_index = {}
    for indexes in duplicate_groups:
        originals = [records[index] for index in indexes]
        kij = float(median(float(record["kij"]) for record in originals))
        collapsed = dict(originals[0])
        collapsed["kij"] = kij
        collapsed["comment"] = " | ".join(
            record.get("comment", "")
            for record in originals
            if record.get("comment")
        )
        collapsed["source"] = "median of duplicate EOS records"
        collapsed["duplicate_policy"] = "median_kij_for_same_model_pair_temperature_range"
        collapsed["duplicate_records"] = [
            {
                "kij": float(record["kij"]),
                "comment": record.get("comment", ""),
                "source": record.get("source", ""),
                "source_file": record.get("source_file", ""),
                "source_line": record.get("source_line"),
                "source_id1": record.get("source_id1", ""),
                "source_id2": record.get("source_id2", ""),
            }
            for record in originals
        ]
        collapsed_by_first_index[indexes[0]] = collapsed

    collapsed_records: list[dict] = []
    for index, record in enumerate(records):
        if index in collapsed_by_first_index:
            collapsed_records.append(collapsed_by_first_index[index])
        elif index not in duplicate_indexes:
            collapsed_records.append(record)

    return collapsed_records, {
        "eos_duplicate_policy": "Exact same model, unordered CAS pair, and temperature range are collapsed with median k_ij.",
        "eos_collapsed_duplicate_groups": len(duplicate_groups),
        "eos_collapsed_duplicate_records": len(duplicate_indexes) - len(duplicate_groups),
    }


def reverse_activity_record(record: dict) -> bool:
    return (record["cas1"], record["cas2"]) != tuple(sorted((record["cas1"], record["cas2"])))


def activity_record_signature(record: dict, model: str) -> tuple:
    reverse = reverse_activity_record(record)
    model_key = model.upper()
    if model_key == "NRTL":
        alpha = float(record["alpha12"])
        if "tau12_c" in record and "tau21_c" in record:
            forward = (
                float(record["tau12_c"]),
                float(record.get("tau12_d", 0.0)),
                float(record.get("tau12_e", 0.0)),
                float(record.get("tau12_f", 0.0)),
            )
            backward = (
                float(record["tau21_c"]),
                float(record.get("tau21_d", 0.0)),
                float(record.get("tau21_e", 0.0)),
                float(record.get("tau21_f", 0.0)),
            )
            return ("tau", backward, forward, alpha) if reverse else ("tau", forward, backward, alpha)
        forward = float(record["a12_cal_per_mol"])
        backward = float(record["a21_cal_per_mol"])
        return ("energy", backward, forward, alpha) if reverse else ("energy", forward, backward, alpha)

    if "tau12_a" in record and "tau21_a" in record:
        forward = (
            float(record["tau12_a"]),
            float(record.get("tau12_b", 0.0)),
            float(record.get("tau12_c", 0.0)),
        )
        backward = (
            float(record["tau21_a"]),
            float(record.get("tau21_b", 0.0)),
            float(record.get("tau21_c", 0.0)),
        )
        return (
            "tau",
            backward if reverse else forward,
            forward if reverse else backward,
            bool(record.get("use_q_prime", False)),
            record.get("model_variant", "standard_uniquac"),
        )
    forward = float(record["a12_cal_per_mol"])
    backward = float(record["a21_cal_per_mol"])
    return ("energy", backward, forward) if reverse else ("energy", forward, backward)


def activity_signatures_equivalent(first: Any, second: Any) -> bool:
    if type(first) is not type(second):
        return False
    if isinstance(first, tuple):
        return (
            len(first) == len(second)
            and all(activity_signatures_equivalent(a, b) for a, b in zip(first, second))
        )
    if isinstance(first, (bool, str)):
        return first == second
    return math.isclose(float(first), float(second), rel_tol=1e-4, abs_tol=1e-3)


def duplicate_source_summary(record: dict) -> dict[str, Any]:
    return {
        "comment": record.get("comment", ""),
        "source": record.get("source", ""),
        "source_file": record.get("source_file", ""),
        "source_line": record.get("source_line"),
        "source_id1": record.get("source_id1", ""),
        "source_id2": record.get("source_id2", ""),
    }


def collapse_activity_equivalent_duplicates(
    records: list[dict],
    model: str,
) -> tuple[list[dict], dict[str, Any]]:
    groups: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(tuple(sorted((record["cas1"], record["cas2"]))), []).append(index)

    duplicate_indexes: set[int] = set()
    collapsed_by_first_index: dict[int, dict] = {}
    collapsed_groups = 0
    for indexes in groups.values():
        clusters: list[tuple[tuple, list[int]]] = []
        for index in indexes:
            signature = activity_record_signature(records[index], model)
            for cluster_signature, cluster_indexes in clusters:
                if activity_signatures_equivalent(signature, cluster_signature):
                    cluster_indexes.append(index)
                    break
            else:
                clusters.append((signature, [index]))

        for _signature, cluster_indexes in clusters:
            if len(cluster_indexes) < 2:
                continue
            collapsed_groups += 1
            duplicate_indexes.update(cluster_indexes)
            originals = [records[index] for index in cluster_indexes]
            collapsed = dict(originals[0])
            collapsed["comment"] = " | ".join(
                record.get("comment", "")
                for record in originals
                if record.get("comment")
            )
            collapsed["duplicate_policy"] = (
                "first_equivalent_activity_record_after_canonical_orientation"
            )
            collapsed["duplicate_records"] = [
                {
                    **duplicate_source_summary(record),
                    "cas1": record["cas1"],
                    "cas2": record["cas2"],
                }
                for record in originals
            ]
            collapsed_by_first_index[cluster_indexes[0]] = collapsed

    collapsed_records: list[dict] = []
    for index, record in enumerate(records):
        if index in collapsed_by_first_index:
            collapsed_records.append(collapsed_by_first_index[index])
        elif index not in duplicate_indexes:
            collapsed_records.append(record)

    return collapsed_records, {
        "activity_duplicate_policy": (
            "Only equivalent records after canonical CAS orientation are collapsed; "
            "competing same-form fits remain separate."
        ),
        "activity_collapsed_equivalent_duplicate_groups": collapsed_groups,
        "activity_collapsed_equivalent_duplicate_records": (
            len(duplicate_indexes) - collapsed_groups
        ),
        "activity_duplicate_equivalence_tolerance": "rel_tol=1e-4, abs_tol=1e-3",
    }


def apply_curated_activity_disables(records: list[dict], model: str) -> int:
    disabled = 0
    for record in records:
        key = (
            model.upper(),
            tuple(sorted((record["cas1"], record["cas2"]))),
            record.get("comment", ""),
        )
        reason = CURATED_ACTIVITY_DISABLED_RECORDS.get(key)
        if reason is None:
            continue
        if not record.get("disabled"):
            disabled += 1
        record["disabled"] = True
        record["disabled_reason"] = reason
    return disabled


def ipd_hydration_records(existing: list[dict], source_name: str) -> tuple[list[dict], int, dict[str, int]]:
    records: list[dict] = []
    by_source: dict[str, int] = {}

    if source_name == "eos_binary_interactions.json":
        existing_keys = {eos_record_hydration_key(record) for record in existing}
        existing_model_pairs = {
            (
                str(record["model"]).upper(),
                tuple(sorted((record["cas1"], record["cas2"]))),
            )
            for record in existing
        }
        new_pairs = 0
        for model, filename in EOS_IPD_FILES.items():
            for record in parse_ipd_records(model, filename):
                key = eos_record_hydration_key(record)
                if key in existing_keys:
                    continue
                model_pair = (key[0], key[1])
                if model_pair not in existing_model_pairs:
                    new_pairs += 1
                    existing_model_pairs.add(model_pair)
                existing_keys.add(key)
                records.append(record)
                by_source[filename] = by_source.get(filename, 0) + 1
        return records, new_pairs, by_source

    if source_name == "nrtl_binary_interactions.json":
        model, filename = "NRTL", ACTIVITY_IPD_FILES["NRTL"]
    elif source_name == "uniquac_binary_interactions.json":
        model, filename = "UNIQUAC", ACTIVITY_IPD_FILES["UNIQUAC"]
    else:
        return [], 0, {}

    existing_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in existing
    }
    new_pairs = 0
    for record in parse_ipd_records(model, filename):
        pair = tuple(sorted((record["cas1"], record["cas2"])))
        if pair in existing_pairs:
            continue
        existing_pairs.add(pair)
        new_pairs += 1
        records.append(record)
        by_source[filename] = by_source.get(filename, 0) + 1
    return records, new_pairs, by_source


def component_index(resolved: dict[str, dict]) -> dict[str, Any]:
    by_cas: dict[str, dict] = {}
    aliases: dict[str, str] = {}
    all_components = dict(resolved)
    all_components.update(record_override_components())
    for source_id, entry in all_components.items():
        cas = entry["cas"]
        component = by_cas.setdefault(
            cas,
            {
                "cas": cas,
                "name": entry.get("name", ""),
                "formula": entry.get("formula", ""),
                "sources": sorted({entry.get("source", "")}),
                "source_component_ids": [],
                "aliases": [],
            },
        )
        component["source_component_ids"].append(source_id)
        component["sources"] = sorted(set(component["sources"]) | {entry.get("source", "")})
        component["aliases"] = sorted(
            set(component["aliases"])
            | {entry.get("name", ""), entry.get("query", "")}
            | set(entry.get("aliases", []))
        )

    for cas, component in by_cas.items():
        for alias in component["aliases"]:
            key = normalize_name(alias)
            if key:
                aliases.setdefault(key, cas)
        aliases.setdefault(normalize_name(cas), cas)
        component["source_component_ids"] = sorted(
            set(component["source_component_ids"]),
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )
        component["aliases"] = [alias for alias in component["aliases"] if alias]

    return {
        "metadata": {
            "key_basis": "CAS",
            "description": "Component identity index generated from ChemSep IDs for CAS-keyed interaction parameters.",
        },
        "components": dict(sorted(by_cas.items())),
        "aliases": dict(sorted(aliases.items())),
    }


def build_interaction_payload(
    source_name: str,
    resolved: dict[str, dict],
    unresolved: list[dict],
) -> dict:
    records, skipped = convert_records(source_name, resolved)
    base_pairs = {
        tuple(sorted((record["cas1"], record["cas2"])))
        for record in records
    }
    supplemental_records: list[dict] = []
    supplemental_new_pairs = 0
    water_organic_overlay_records = 0
    water_organic_overlay_replaced_records = 0
    water_organic_overlay_replaced_pairs = 0
    literature_vle_overlay_records = 0
    literature_vle_overlay_replaced_records = 0
    literature_vle_overlay_replaced_pairs = 0
    assorted_overlay_records = 0
    assorted_overlay_zero_records = 0
    assorted_overlay_replaced_records = 0
    assorted_overlay_replaced_pairs = 0
    isopropanol_water_overlay_records = 0
    isopropanol_water_overlay_replaced_records = 0
    isopropanol_water_overlay_replaced_pairs = 0
    water_eo_models = {
        "eos_binary_interactions.json": ("PR", "SRK"),
        "nrtl_binary_interactions.json": ("NRTL",),
        "uniquac_binary_interactions.json": ("UNIQUAC",),
    }.get(source_name, ())
    water_eo_records, water_eo_new_pairs = supplemental_water_ethylene_oxide_records(
        records,
        water_eo_models,
    )
    records.extend(water_eo_records)
    supplemental_records.extend(water_eo_records)
    supplemental_new_pairs += water_eo_new_pairs
    acid_model = {
        "nrtl_binary_interactions.json": "NRTL",
        "uniquac_binary_interactions.json": "UNIQUAC",
    }.get(source_name)
    if acid_model is not None:
        acid_records, acid_new_pairs = supplemental_acetic_acid_vle_records(
            records,
            acid_model,
        )
        records[0:0] = acid_records
        supplemental_records += acid_records
        supplemental_new_pairs += acid_new_pairs
        ester_records, ester_new_pairs = supplemental_ester_alcohol_fit_records(
            records,
            acid_model,
        )
        records[0:0] = ester_records
        supplemental_records += ester_records
        supplemental_new_pairs += ester_new_pairs
    if source_name == "nrtl_binary_interactions.json":
        matrix_records, matrix_new_pairs = supplemental_nrtl_matrix_records(records)
        records.extend(matrix_records)
        regression_records, regression_new_pairs = supplemental_water_aromatic_regression_records(
            records,
            "NRTL",
        )
        supplemental_records += matrix_records + regression_records
        supplemental_new_pairs += matrix_new_pairs + regression_new_pairs
        records.extend(regression_records)
        phenolic_records, phenolic_new_pairs = supplemental_phenolic_temperature_records(
            records,
            "NRTL",
        )
        supplemental_records += phenolic_records
        supplemental_new_pairs += phenolic_new_pairs
        records.extend(phenolic_records)
        cesari_records, cesari_new_pairs = supplemental_cesari_phenolic_nrtl_records(
            records,
        )
        supplemental_records += cesari_records
        supplemental_new_pairs += cesari_new_pairs
        records.extend(cesari_records)
        ether_records, ether_new_pairs = supplemental_diethyl_ether_water_records(
            records,
            "NRTL",
        )
        supplemental_records += ether_records
        supplemental_new_pairs += ether_new_pairs
        records.extend(ether_records)
        butanol_records, butanol_new_pairs = supplemental_1_butanol_water_records(
            records,
            "NRTL",
        )
        supplemental_records += butanol_records
        supplemental_new_pairs += butanol_new_pairs
        records.extend(butanol_records)
    elif source_name == "uniquac_binary_interactions.json":
        extended_records, extended_new_pairs = supplemental_extended_uniquac_records(records)
        records.extend(extended_records)
        regression_records, regression_new_pairs = supplemental_water_aromatic_regression_records(
            records,
            "UNIQUAC",
        )
        supplemental_records += extended_records + regression_records
        supplemental_new_pairs += extended_new_pairs + regression_new_pairs
        records.extend(regression_records)
        phenolic_records, phenolic_new_pairs = supplemental_phenolic_temperature_records(
            records,
            "UNIQUAC",
        )
        supplemental_records += phenolic_records
        supplemental_new_pairs += phenolic_new_pairs
        records.extend(phenolic_records)
        ether_records, ether_new_pairs = supplemental_diethyl_ether_water_records(
            records,
            "UNIQUAC",
        )
        supplemental_records += ether_records
        supplemental_new_pairs += ether_new_pairs
        records.extend(ether_records)
        butanol_records, butanol_new_pairs = supplemental_1_butanol_water_records(
            records,
            "UNIQUAC",
        )
        supplemental_records += butanol_records
        supplemental_new_pairs += butanol_new_pairs
        records.extend(butanol_records)

    if acid_model is not None:
        overlay_records, _overlay_new_pairs, overlay_pairs = (
            supplemental_water_organic_binary_fit_records(records, acid_model)
        )
        superseded = [
            record for record in records
            if tuple(sorted((record["cas1"], record["cas2"]))) in overlay_pairs
        ]
        superseded_ids = {id(record) for record in superseded}
        water_organic_overlay_replaced_records = len(superseded)
        water_organic_overlay_replaced_pairs = len({
            tuple(sorted((record["cas1"], record["cas2"])))
            for record in superseded
        })
        records = [record for record in records if id(record) not in superseded_ids]
        supplemental_records = [
            record for record in supplemental_records
            if id(record) not in superseded_ids
        ]
        records[0:0] = overlay_records
        supplemental_records += overlay_records
        water_organic_overlay_records = len(overlay_records)
        supplemental_new_pairs = len({
            tuple(sorted((record["cas1"], record["cas2"])))
            for record in supplemental_records
        } - base_pairs)

        literature_records, _literature_new_pairs, literature_pairs = (
            supplemental_literature_vle_activity_records(records, acid_model)
        )
        superseded = [
            record for record in records
            if tuple(sorted((record["cas1"], record["cas2"])))
            in literature_pairs
        ]
        superseded_ids = {id(record) for record in superseded}
        literature_vle_overlay_replaced_records = len(superseded)
        literature_vle_overlay_replaced_pairs = len({
            tuple(sorted((record["cas1"], record["cas2"])))
            for record in superseded
        })
        records = [
            record for record in records if id(record) not in superseded_ids
        ]
        supplemental_records = [
            record for record in supplemental_records
            if id(record) not in superseded_ids
        ]
        records[0:0] = literature_records
        supplemental_records += literature_records
        literature_vle_overlay_records = len(literature_records)
        supplemental_new_pairs = len({
            tuple(sorted((record["cas1"], record["cas2"])))
            for record in supplemental_records
        } - base_pairs)

        assorted_records, _assorted_new_pairs, assorted_pairs = (
            supplemental_assorted_alcohol_ether_records(records, acid_model)
        )
        superseded = [
            record for record in records
            if tuple(sorted((record["cas1"], record["cas2"]))) in assorted_pairs
        ]
        supplemental_ids = {id(record) for record in supplemental_records}
        curated_conflicts = [
            record for record in superseded if id(record) in supplemental_ids
        ]
        if curated_conflicts:
            conflict_labels = [
                record.get("comment", str((record["cas1"], record["cas2"])))
                for record in curated_conflicts
            ]
            raise ValueError(
                "assorted_alcohols_ethers.json would replace newer curated "
                f"records; review required: {conflict_labels}"
            )
        superseded_ids = {id(record) for record in superseded}
        assorted_overlay_replaced_records = len(superseded)
        assorted_overlay_replaced_pairs = len({
            tuple(sorted((record["cas1"], record["cas2"])))
            for record in superseded
        })
        records = [
            record for record in records if id(record) not in superseded_ids
        ]
        records[0:0] = assorted_records
        supplemental_records += assorted_records
        assorted_overlay_records = len(assorted_records)
        assorted_overlay_zero_records = sum(
            record.get("fit_status") == "recommended_defensible_zero_interaction"
            for record in assorted_records
        )
        supplemental_new_pairs = len({
            tuple(sorted((record["cas1"], record["cas2"])))
            for record in supplemental_records
        } - base_pairs)

        ipa_records, _ipa_new_pairs, ipa_pairs = (
            supplemental_isopropanol_water_records(records, acid_model)
        )
        superseded = [
            record for record in records
            if tuple(sorted((record["cas1"], record["cas2"]))) in ipa_pairs
        ]
        supplemental_ids = {id(record) for record in supplemental_records}
        curated_conflicts = [
            record for record in superseded if id(record) in supplemental_ids
        ]
        if curated_conflicts:
            conflict_labels = [
                record.get("comment", str((record["cas1"], record["cas2"])))
                for record in curated_conflicts
            ]
            raise ValueError(
                "isopropanol_water_interactions.json would replace newer "
                f"curated records; review required: {conflict_labels}"
            )
        superseded_ids = {id(record) for record in superseded}
        isopropanol_water_overlay_replaced_records = len(superseded)
        isopropanol_water_overlay_replaced_pairs = len({
            tuple(sorted((record["cas1"], record["cas2"])))
            for record in superseded
        })
        records = [
            record for record in records if id(record) not in superseded_ids
        ]
        records[0:0] = ipa_records
        supplemental_records += ipa_records
        isopropanol_water_overlay_records = len(ipa_records)
        supplemental_new_pairs = len({
            tuple(sorted((record["cas1"], record["cas2"])))
            for record in supplemental_records
        } - base_pairs)

    hydration_records, hydration_new_pairs, hydration_by_source = ipd_hydration_records(
        records,
        source_name,
    )
    records.extend(hydration_records)
    dedupe_metadata: dict[str, Any] = {}
    if source_name == "eos_binary_interactions.json":
        records, dedupe_metadata = collapse_eos_duplicate_records(records)
    elif source_name == "nrtl_binary_interactions.json":
        records, dedupe_metadata = collapse_activity_equivalent_duplicates(records, "NRTL")
        dedupe_metadata["activity_curated_disabled_records"] = apply_curated_activity_disables(
            records,
            "NRTL",
        )
    elif source_name == "uniquac_binary_interactions.json":
        records, dedupe_metadata = collapse_activity_equivalent_duplicates(records, "UNIQUAC")
        dedupe_metadata["activity_curated_disabled_records"] = apply_curated_activity_disables(
            records,
            "UNIQUAC",
        )

    if source_name == "nrtl_binary_interactions.json":
        # tau_f is an optional linear-T extension. Preserve real nonzero
        # source values, but keep the historical runtime payload compact by
        # representing the documented zero default through omission.
        for record in records:
            for field in ("tau12_f", "tau21_f"):
                if field in record and abs(float(record[field])) <= 0.0:
                    record.pop(field)

    payload = {
        "metadata": {
            "key_basis": "CAS",
            "source_file": source_name,
            "converted_records": len(records),
            "base_converted_records": len(records) - len(supplemental_records) - len(hydration_records),
            "supplemental_records": len(supplemental_records),
            "supplemental_new_pairs": supplemental_new_pairs,
            "ipd_hydration_records": len(hydration_records),
            "ipd_hydration_new_pairs": hydration_new_pairs,
            "ipd_hydration_by_source": hydration_by_source,
            "disabled_records": sum(1 for record in records if record.get("disabled")),
            "skipped_records": len(skipped),
            "unresolved_components": unresolved,
            "skipped": skipped,
            **dedupe_metadata,
        },
        "interactions": records,
    }
    if acid_model is not None:
        payload["metadata"].update({
            "water_organic_overlay_records": water_organic_overlay_records,
            "water_organic_overlay_replaced_records": water_organic_overlay_replaced_records,
            "water_organic_overlay_replaced_pairs": water_organic_overlay_replaced_pairs,
            "literature_vle_overlay_records": literature_vle_overlay_records,
            "literature_vle_overlay_replaced_records": literature_vle_overlay_replaced_records,
            "literature_vle_overlay_replaced_pairs": literature_vle_overlay_replaced_pairs,
            "assorted_overlay_records": assorted_overlay_records,
            "assorted_overlay_zero_records": assorted_overlay_zero_records,
            "assorted_overlay_replaced_records": assorted_overlay_replaced_records,
            "assorted_overlay_replaced_pairs": assorted_overlay_replaced_pairs,
            "isopropanol_water_overlay_records": isopropanol_water_overlay_records,
            "isopropanol_water_overlay_replaced_records": isopropanol_water_overlay_replaced_records,
            "isopropanol_water_overlay_replaced_pairs": isopropanol_water_overlay_replaced_pairs,
        })
    source_payload = load_json(ARCHIVED_DATA / source_name)
    for key in ("source", "units"):
        if key in source_payload:
            payload["metadata"][key] = source_payload[key]
    return payload


def write_interaction_file(
    output_name: str,
    source_name: str,
    resolved: dict[str, dict],
    unresolved: list[dict],
) -> None:
    payload = build_interaction_payload(source_name, resolved, unresolved)
    write_json(DATA / output_name, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chemicals-path",
        help="Optional temporary path containing the chemicals package, used only for identity resolution gaps.",
    )
    args = parser.parse_args()

    resolved, unresolved = resolve_component_ids(args.chemicals_path)
    write_json(DATA / "interaction_component_cas_index.json", component_index(resolved))
    write_interaction_file("eos_binary_interactions_cas.json", "eos_binary_interactions.json", resolved, unresolved)
    write_interaction_file("nrtl_binary_interactions_cas.json", "nrtl_binary_interactions.json", resolved, unresolved)
    write_interaction_file("uniquac_binary_interactions_cas.json", "uniquac_binary_interactions.json", resolved, unresolved)

    unresolved_ids = {item["source_component_id"] for item in unresolved}
    skipped_total = 0
    for output_name in (
        "eos_binary_interactions_cas.json",
        "nrtl_binary_interactions_cas.json",
        "uniquac_binary_interactions_cas.json",
    ):
        skipped_total += load_json(DATA / output_name)["metadata"]["skipped_records"]
    print(
        f"Resolved {len(resolved)} source component IDs; "
        f"unresolved={len(unresolved_ids)}; skipped_records={skipped_total}"
    )


if __name__ == "__main__":
    main()
