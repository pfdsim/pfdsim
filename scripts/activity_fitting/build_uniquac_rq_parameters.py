#!/usr/bin/env python3
"""Build CAS-keyed UNIQUAC pure-component r/q parameters."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SOURCE_DATA = DATA / "source" / "activity_fitting"
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


def normalize_name(value: str) -> str:
    text = value.lower().strip()
    text = text.replace("n,n-", "")
    text = text.replace("n.n-", "")
    text = text.replace("n-", "")
    text = text.replace("normal ", "")
    text = text.replace("iso-", "iso")
    text = text.replace("p-", "p")
    text = text.replace("m-", "m")
    text = text.replace("o-", "o")
    return re.sub(r"[^a-z0-9]+", "", text)


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


def valid_cas(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text if CAS_RE.match(text) else None


def source_priority(source: str) -> tuple[int, str]:
    if "thermochimica_acta_1995_rq.json" in source:
        return (-4, source)
    if "water_nonwater_binary_parameters_curated" in source:
        return (-3, source)
    if "ethanol_water_interactions" in source:
        return (-2, source)
    if "in this work" in source or "ester_uniquac_combinatorial" in source:
        return (-1, source)
    if "DWSIM.Thermodynamics" in source:
        return (0, source)
    if "Nagata and Gmehling" in source:
        return (2, source)
    if "water_organic_binary_fits" in source:
        return (3, source)
    return (1, source)


def add_alias(
    aliases: dict[str, Optional[str]], alias: Optional[str], cas: str
) -> None:
    if not alias:
        return
    for key in {normalize_name(str(alias)), normalize_name(cas)}:
        if not key:
            continue
        existing = aliases.get(key)
        if existing is None and key in aliases:
            continue
        if existing is not None and existing != cas:
            aliases[key] = None
        else:
            aliases[key] = cas


def add_record(
    components: dict[str, dict], aliases: dict[str, Optional[str]], record: dict
) -> None:
    cas = valid_cas(record.get("cas"))
    r = parse_float(record.get("r"))
    q = parse_float(record.get("q"))
    if not cas or r is None or q is None or r <= 0.0 or q <= 0.0:
        return
    source = str(record.get("source", "") or "unknown")
    name = str(record.get("name", "") or "")
    entry = components.setdefault(
        cas,
        {
            "cas": cas,
            "name": name,
            "formula": str(record.get("formula", "") or ""),
            "r": r,
            "q": q,
            "source": source,
            "records": [],
            "aliases": [],
        },
    )
    source_record = {
        "name": name,
        "formula": str(record.get("formula", "") or ""),
        "r": r,
        "q": q,
        "source": source,
    }
    if record.get("dwsim_id"):
        source_record["dwsim_id"] = str(record["dwsim_id"])
    entry["records"].append(source_record)
    if source_priority(source) < source_priority(entry["source"]):
        entry.update(
            {
                "name": name,
                "formula": str(record.get("formula", "") or ""),
                "r": r,
                "q": q,
                "source": source,
            }
        )
    q_prime = parse_float(record.get("q_prime"))
    if record.get("q_prime") is not None:
        if q_prime is None or q_prime <= 0.0:
            raise ValueError(f"Invalid q_prime metadata for CAS {cas}")
        entry["q_prime"] = q_prime
    entry["aliases"] = sorted(
        set(entry.get("aliases", []))
        | {
            value
            for value in (name, record.get("formula"), record.get("dwsim_id"))
            if value
        }
    )
    for alias in (name, cas, record.get("dwsim_id")):
        add_alias(aliases, alias, cas)


def csv_records() -> list[dict]:
    path = SOURCE_DATA / "dwsim_uniquac_combinatorial_parameters.csv"
    records = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                {
                    "cas": row.get("cas"),
                    "name": row.get("name"),
                    "formula": row.get("formula"),
                    "dwsim_id": row.get("dwsim_id"),
                    "r": row.get("UNIQUAC_r"),
                    "q": row.get("UNIQUAC_q"),
                    "source": row.get("source") or "DWSIM UNIQUAC parameters",
                }
            )
    return records


def nagata_records() -> list[dict]:
    payload = load_json(SOURCE_DATA / "nagata_gmehling_extended_uniquac_rq.json")
    source = payload["metadata"]["source"]
    records = []
    for item in payload["components"]:
        q = float(item["q"])
        q_prime = item.get("q_prime")
        q_prime_expression = item.get("q_prime_expression")
        if q_prime is None and q_prime_expression == "q^0.1":
            q_prime = q**0.1
        record = {
            "cas": item["cas"],
            "name": item["name"],
            "formula": item.get("formula", ""),
            "r": item["r"],
            "q": q,
            "source": source,
            "q_prime": q_prime,
            "q_prime_expression": q_prime_expression,
            "antoine": {
                "A": item["antoine_A"],
                "B": item["antoine_B"],
                "C": item["antoine_C"],
                "pressure_units": "mmHg",
                "temperature_units": "degC",
            },
        }
        records.append(record)
    return records


def water_ethylene_oxide_records() -> list[dict]:
    path = SOURCE_DATA / "water_ethylene_oxide_interactions.json"
    if not path.exists():
        return []
    payload = load_json(path)
    metadata = payload["metadata"]
    parameters = payload["fitted_parameters"]["UNIQUAC"]
    return [
        {
            "cas": metadata["cas"][1],
            "name": metadata["components"][1],
            "formula": "C2H4O",
            "r": parameters["component2_r"],
            "q": parameters["component2_q"],
            "source": metadata["source"],
        }
    ]


def water_organic_binary_fit_records() -> list[dict]:
    path = SOURCE_DATA / "water_organic_binary_fits.json"
    if not path.exists():
        return []
    payload = load_json(path)
    records_by_cas: dict[str, dict] = {}
    for pair in payload.get("pairs", []):
        structural = pair.get("uniquac", {}).get("structural_parameters", {})
        for index, component_key in ((1, "component1"), (2, "component2")):
            name = pair[component_key]
            keys = {
                normalize_name(name),
                normalize_name(name.replace("-", "_")),
            }
            parameters = next(
                (
                    values
                    for structural_name, values in structural.items()
                    if normalize_name(structural_name) in keys
                ),
                None,
            )
            if parameters is None:
                raise ValueError(
                    f"Missing UNIQUAC structural parameters for {name!r} in "
                    f"{pair['pair_id']}"
                )
            cas = pair[f"cas{index}"]
            record = {
                "cas": cas,
                "name": name,
                "r": parameters["r"],
                "q": parameters["q"],
                "source": "water_organic_binary_fits.json; curated UNIQUAC database structural parameters",
            }
            previous = records_by_cas.get(cas)
            if previous is not None and (
                float(previous["r"]) != float(record["r"])
                or float(previous["q"]) != float(record["q"])
            ):
                raise ValueError(
                    f"Conflicting UNIQUAC structural parameters for CAS {cas}"
                )
            records_by_cas[cas] = record
    return list(records_by_cas.values())


def mibk_vle_rq_records() -> list[dict]:
    path = SOURCE_DATA / "mibk_water_acids_vle.json"
    if not path.exists():
        return []
    payload = load_json(path)
    parameters = payload["H2O_MIBK"]["recommended_parameters"]["UNIQUAC"][
        "structural_parameters_used_in_fit"
    ]["MIBK"]
    return [
        {
            "cas": payload["H2O_MIBK"]["system"]["cas"]["MIBK"],
            "name": "methyl isobutyl ketone",
            "formula": "C6H12O",
            "r": parameters["r"],
            "q": parameters["q"],
            "source": (
                "mibk_water_acids_vle.json; structural parameters used in the "
                "recommended Water/MIBK UNIQUAC regression"
            ),
        }
    ]


def ethanol_water_rq_records() -> list[dict]:
    path = SOURCE_DATA / "ethanol_water_interactions.json"
    if not path.exists():
        return []
    payload = load_json(path)
    parameters = payload["parameters"]["activity_models"]["UNIQUAC"][
        "structural_parameters_reconstruction_recommended"
    ]["ethanol"]
    return [
        {
            "cas": "64-17-5",
            "name": "Ethanol",
            "formula": "C2H6O",
            "r": parameters["R"],
            "q": parameters["Q"],
            "source": (
                "ethanol_water_interactions.json; Voutsas et al. (2011) "
                "UNIQUAC structural basis"
            ),
        }
    ]


def ester_combinatorial_records() -> list[dict]:
    path = SOURCE_DATA / "ester_uniquac_combinatorial_parameters.json"
    if not path.exists():
        return []
    payload = load_json(path)
    records = []
    for item in payload.get("components", []):
        record = dict(item)
        aliases = list(record.pop("aliases", []))
        record["source"] = (
            f"ester_uniquac_combinatorial_parameters.json; {record['source']}"
        )
        records.append(record)
        for alias in aliases:
            alias_record = dict(record)
            alias_record["name"] = alias
            records.append(alias_record)
    return records


def assorted_alcohol_ether_rq_records(existing_cas: set[str]) -> list[dict]:
    """Return only new CAS-keyed r/q entries needed by recommended fits."""
    path = SOURCE_DATA / "assorted_alcohols_ethers.json"
    if not path.exists():
        return []
    payload = load_json(path)
    excluded_pair = frozenset(("2-propanol", "water"))
    records_by_cas: dict[str, dict] = {}

    def add_component(name: str, parameters: dict, provenance: str) -> None:
        try:
            cas = ASSORTED_ALCOHOL_ETHER_CAS[name]
        except KeyError as error:
            raise ValueError(
                f"Missing CAS mapping for assorted alcohol/ether component {name!r}"
            ) from error
        if cas in existing_cas:
            return
        record = {
            "cas": cas,
            "name": name,
            "r": float(parameters["r"]),
            "q": float(parameters["q"]),
            "source": (
                "assorted_alcohols_ethers.json; recommended UNIQUAC structural "
                f"basis; {provenance}"
            ),
        }
        previous = records_by_cas.get(cas)
        if previous is not None and (
            not math.isclose(float(previous["r"]), record["r"])
            or not math.isclose(float(previous["q"]), record["q"])
        ):
            raise ValueError(
                f"Conflicting recommended assorted UNIQUAC r/q values for CAS {cas}"
            )
        records_by_cas[cas] = record

    for entry in payload["entries"]:
        if frozenset(entry["components"]) == excluded_pair:
            continue
        uniquac = entry.get("uniquac")
        if not (
            entry.get("recommended", False)
            and uniquac
            and uniquac.get("recommended", False)
        ):
            continue
        for name, parameters in uniquac["r_q_use"]["components"].items():
            add_component(
                name,
                parameters,
                str(uniquac.get("r_q_provenance_status", "documented basis")),
            )

    registry = payload["uniquac_structural_parameter_registry"]
    for entry in payload["retired_or_not_promoted"]:
        if "use_zero_interaction" not in entry.get("status", ""):
            continue
        for name in entry["components"]:
            candidates = [
                item
                for item in registry.values()
                if item["component"] == name
                and item.get("active_database_standard", True)
            ]
            if candidates:
                add_component(name, candidates[0], "defensible-zero structural basis")
        for name, parameters in entry.get("source_uniquac_r_q", {}).items():
            if isinstance(parameters, dict) and "r" in parameters and "q" in parameters:
                add_component(name, parameters, "defensible-zero source basis")

    return list(records_by_cas.values())


def curated_water_nonwater_rq_records() -> list[dict]:
    records = []
    for filename in (
        "water_nonwater_binary_parameters_curated.json",
        "eg_glycerol_activity_parameters.json",
    ):
        path = SOURCE_DATA / filename
        if not path.exists():
            continue
        for item in load_json(path).get("uniquac_components", []):
            record = dict(item)
            record["source"] = f"{filename}; " + str(
                record.get("source", "reviewed staged collection")
            )
            records.append(record)
    return records


def thermochimica_1995_rq_records() -> list[dict]:
    filename = "thermochimica_acta_1995_rq.json"
    payload = load_json(SOURCE_DATA / filename)
    source = f"{filename}; {payload['metadata']['source']}"
    return [
        {**item, "source": source}
        for item in payload["components"]
    ]


def apply_extended_metadata(
    components: dict[str, dict], aliases: dict[str, Optional[str]], records: list[dict]
) -> None:
    for record in records:
        add_record(components, aliases, record)
        cas = valid_cas(record.get("cas"))
        if not cas or cas not in components:
            continue
        extended = {
            "r": float(record["r"]),
            "q": float(record["q"]),
            "q_prime": float(record["q_prime"]),
            "source": record["source"],
            "antoine": record["antoine"],
        }
        if record.get("q_prime_expression"):
            extended["q_prime_expression"] = record["q_prime_expression"]
        components[cas]["extended_uniquac"] = extended


def build_uniquac_rq_payload() -> dict[str, Any]:
    components: dict[str, dict] = {}
    aliases: dict[str, Optional[str]] = {}

    for record in csv_records():
        add_record(components, aliases, record)
    for record in water_ethylene_oxide_records():
        add_record(components, aliases, record)
    for record in water_organic_binary_fit_records():
        add_record(components, aliases, record)
    for record in mibk_vle_rq_records():
        add_record(components, aliases, record)
    for record in ethanol_water_rq_records():
        add_record(components, aliases, record)
    for record in ester_combinatorial_records():
        add_record(components, aliases, record)
    for record in curated_water_nonwater_rq_records():
        add_record(components, aliases, record)
    apply_extended_metadata(components, aliases, nagata_records())
    assorted_records = assorted_alcohol_ether_rq_records(set(components))
    for record in assorted_records:
        add_record(components, aliases, record)
    for record in thermochimica_1995_rq_records():
        add_record(components, aliases, record)

    ambiguous_aliases = sorted(key for key, cas in aliases.items() if cas is None)
    payload = {
        "metadata": {
            "key_basis": "CAS",
            "description": "CAS-keyed UNIQUAC pure-component r/q parameters.",
            "source_files": [
                "data/source/activity_fitting/dwsim_uniquac_combinatorial_parameters.csv",
                "data/source/activity_fitting/water_ethylene_oxide_interactions.json",
                "data/source/activity_fitting/water_organic_binary_fits.json",
                "data/source/activity_fitting/mibk_water_acids_vle.json",
                "data/source/activity_fitting/ethanol_water_interactions.json",
                "data/source/activity_fitting/nagata_gmehling_extended_uniquac_rq.json",
                "data/source/activity_fitting/ester_uniquac_combinatorial_parameters.json",
                "data/source/activity_fitting/assorted_alcohols_ethers.json",
                "data/source/activity_fitting/water_nonwater_binary_parameters_curated.json",
                "data/source/activity_fitting/eg_glycerol_activity_parameters.json",
                "data/source/activity_fitting/thermochimica_acta_1995_rq.json",
            ],
            "component_count": len(components),
            "ambiguous_alias_count": len(ambiguous_aliases),
            "ambiguous_aliases": ambiguous_aliases,
        },
        "components": dict(sorted(components.items())),
        "aliases": dict(
            sorted((key, cas) for key, cas in aliases.items() if cas is not None)
        ),
    }
    return payload


def main() -> None:
    payload = build_uniquac_rq_payload()
    write_json(DATA / "uniquac_rq_cas.json", payload)
    print(
        f"Wrote {len(payload['components'])} UNIQUAC r/q CAS entries; "
        f"ambiguous_aliases={payload['metadata']['ambiguous_alias_count']}"
    )


if __name__ == "__main__":
    main()
