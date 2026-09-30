"""Local CAS-keyed interaction parameter lookup for cubic EOS and activity models."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Optional

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from .chemical_properties import ChemicalProperties
else:
    from chemical_properties import ChemicalProperties


DATA_DIR = Path(__file__).parent / "data"
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K = 10.0


def canonical_eos_model_key(model: str) -> str:
    """Return the CAS-table EOS family key for a cubic model name."""
    return (
        "SRK"
        if str(model).upper().replace("_", "-") in ("SRK", "RKS", "RKS-BM", "SRK-BM")
        else "PR"
    )


def _normalize_name(value: str) -> str:
    text = value.lower().strip()
    text = text.replace("n,n-", "")
    text = text.replace("n-", "")
    text = text.replace("normal ", "")
    text = text.replace("iso-", "iso")
    text = text.replace("p-", "p")
    text = text.replace("m-", "m")
    text = text.replace("o-", "o")
    text = text.replace("flouro", "fluoro")
    text = text.replace("propnaol", "propanol")
    text = text.replace("pyridien", "pyridine")
    text = text.replace("pyridinr", "pyridine")
    text = re.sub(r"[^a-z0-9]+", "", text)
    synonyms = {
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
    }
    return synonyms.get(text, text)


def _canonical_cas(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    return text if CAS_RE.match(text) else None


def cas_for_component(
    name: str, props: Optional[ChemicalProperties] = None
) -> Optional[str]:
    """Return a CAS number for interaction-parameter lookup."""
    try:
        if __package__ and __package__.split(".", 1)[0] == "pfdsim":
            from .compound_identity import get_compound_identity_resolver
        else:
            from compound_identity import get_compound_identity_resolver
        resolver = get_compound_identity_resolver()
    except Exception:
        resolver = None

    candidates = [name]
    if props is not None:
        direct = _canonical_cas(props.CAS)
        if direct:
            return direct
        candidates.extend([props.symbol, props.name])

    for candidate in candidates:
        direct = _canonical_cas(candidate)
        if direct:
            return direct
        if candidate and resolver is not None:
            cas = resolver.resolve_cas(str(candidate), allow_formula=False)
            if cas:
                return cas
    return None


@lru_cache(maxsize=1)
def _eos_interactions() -> dict[str, dict[tuple[str, str], list[dict]]]:
    path = DATA_DIR / "eos_binary_interactions_cas.json"
    data = json.loads(path.read_text())["interactions"]
    grouped: dict[str, dict[tuple[str, str], list[dict]]] = {}
    for record in data:
        model = record["model"].upper()
        key = tuple(sorted((record["cas1"], record["cas2"])))
        normalized = dict(record)
        normalized["kij"] = float(record["kij"])
        if "Tmin_K" in record and "Tmax_K" in record:
            normalized["temperature_range"] = (
                float(record["Tmin_K"]),
                float(record["Tmax_K"]),
            )
        else:
            normalized["temperature_range"] = _temperature_range(
                record.get("comment", "")
            )
        grouped.setdefault(model, {}).setdefault(key, []).append(normalized)

    return grouped


def _temperature_range(comment: str) -> tuple[float, float] | None:
    match = re.search(
        r"T=([0-9]+(?:\.[0-9]+)?)(?:-([0-9]+(?:\.[0-9]+)?))?K",
        comment,
        re.IGNORECASE,
    )
    if not match:
        return None
    low = float(match.group(1))
    high = float(match.group(2)) if match.group(2) is not None else low
    return (min(low, high), max(low, high))


def _effective_temperature_range(low: float, high: float) -> tuple[float, float]:
    if low == high:
        return (
            low - EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K,
            high + EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K,
        )
    return low, high


def _temperature_score(record: dict, T: float) -> tuple[bool, float, float]:
    temperature_range = record.get("temperature_range")
    if temperature_range is None:
        return (True, float("inf"), float("inf"))
    low, high = temperature_range
    effective_low, effective_high = _effective_temperature_range(low, high)
    if effective_low <= T <= effective_high:
        return (False, 0.0, effective_high - effective_low)
    return (
        True,
        min(abs(T - effective_low), abs(T - effective_high)),
        effective_high - effective_low,
    )


def _median_kij(records: list[dict], T: Optional[float] = None) -> float:
    return median(eos_record_kij(record, T) for record in records)


def _median_kij_record(records: list[dict], T: Optional[float] = None) -> dict:
    kij = _median_kij(records, T)
    return min(records, key=lambda record: abs(eos_record_kij(record, T) - kij))


def eos_record_kij(record: dict, T: Optional[float] = None) -> float:
    """Evaluate a constant or simple temperature-dependent k_ij record."""
    if any(key in record for key in ("kij_a", "kij_b", "kij_c")):
        T_eval = float(
            T if T is not None else record.get("T_ref_K", record.get("Tref_K", 298.15))
        )
        return (
            float(record.get("kij_a", 0.0))
            + float(record.get("kij_b", 0.0)) / T_eval
            + float(record.get("kij_c", 0.0)) * T_eval
        )
    return float(record["kij"])


def _select_eos_record(records: list[dict], T: Optional[float] = None) -> dict:
    kij = _select_eos_kij(records, T)
    return min(records, key=lambda record: abs(eos_record_kij(record, T) - kij))


def _select_eos_kij(records: list[dict], T: Optional[float] = None) -> float:
    static_records = [
        record for record in records if record.get("temperature_range") is None
    ]
    ranged_records = [
        record for record in records if record.get("temperature_range") is not None
    ]
    if T is None:
        candidates = static_records or records
        return float(eos_record_kij(_median_kij_record(candidates, T), T))
    in_range = [
        record for record in ranged_records if _temperature_score(record, T)[0] is False
    ]
    if in_range:
        return float(
            eos_record_kij(
                min(in_range, key=lambda record: _temperature_score(record, T)), T
            )
        )
    nearest_ranged = (
        min(ranged_records, key=lambda record: _temperature_score(record, T))
        if ranged_records
        else None
    )
    if static_records:
        static_kij = _median_kij(static_records, T)
        if nearest_ranged is not None:
            return 0.5 * (static_kij + eos_record_kij(nearest_ranged, T))
        return float(eos_record_kij(_median_kij_record(static_records, T), T))
    if nearest_ranged is not None:
        return float(eos_record_kij(nearest_ranged, T))
    return float(eos_record_kij(_median_kij_record(records, T), T))


def _eos_model_key(model: str) -> str:
    return canonical_eos_model_key(model)


def eos_binary_interaction_records(
    model: str, cas1: Optional[str], cas2: Optional[str]
) -> list[dict]:
    if not cas1 or not cas2 or cas1 == cas2:
        return []
    model_key = _eos_model_key(model)
    return _eos_interactions().get(model_key, {}).get(tuple(sorted((cas1, cas2))), [])


def eos_binary_interaction(
    model: str,
    cas1: Optional[str],
    cas2: Optional[str],
    T: Optional[float] = None,
) -> float:
    if not cas1 or not cas2 or cas1 == cas2:
        return 0.0
    records = eos_binary_interaction_records(model, cas1, cas2)
    if not records:
        return 0.0
    return _select_eos_kij(records, T)


@lru_cache(maxsize=1)
def _nrtl_interactions() -> dict[tuple[str, str], dict]:
    path = DATA_DIR / "nrtl_binary_interactions_cas.json"
    data = json.loads(path.read_text())["interactions"]
    interactions: dict[tuple[str, str], dict] = {}
    for record in data:
        if record.get("disabled"):
            continue
        cas1, cas2 = record["cas1"], record["cas2"]
        if cas1 == cas2:
            continue
        interaction = {
            "cas1": cas1,
            "cas2": cas2,
            "alpha12": float(record["alpha12"]),
            "comment": record.get("comment", ""),
        }
        for field in ("extrapolation", "Tmin_K", "Tmax_K"):
            if field in record:
                interaction[field] = record[field]
        if "tau12_c" in record and "tau21_c" in record:
            interaction.update(
                {
                    "tau12_c": float(record["tau12_c"]),
                    "tau12_d": float(record.get("tau12_d", 0.0)),
                    "tau12_e": float(record.get("tau12_e", 0.0)),
                    "tau12_f": float(record.get("tau12_f", 0.0)),
                    "tau12_g": float(record.get("tau12_g", 0.0)),
                    "tau21_c": float(record["tau21_c"]),
                    "tau21_d": float(record.get("tau21_d", 0.0)),
                    "tau21_e": float(record.get("tau21_e", 0.0)),
                    "tau21_f": float(record.get("tau21_f", 0.0)),
                    "tau21_g": float(record.get("tau21_g", 0.0)),
                    "tau_tref": float(record.get("tau_tref", 298.15)),
                }
            )
        else:
            interaction.update(
                {
                    "a12_cal_per_mol": float(record["a12_cal_per_mol"]),
                    "a21_cal_per_mol": float(record["a21_cal_per_mol"]),
                }
            )
        key = tuple(sorted((cas1, cas2)))
        existing = interactions.get(key)
        if existing is not None:
            if "tau12_c" in interaction and "tau12_c" not in existing:
                interactions[key] = interaction
            continue
        interactions[key] = interaction
    return interactions


def _nrtl_oriented_record(data: dict, reverse: bool = False) -> dict:
    result = {
        "alpha12": data["alpha12"],
        "comment": data.get("comment", ""),
    }
    for field in ("extrapolation", "Tmin_K", "Tmax_K"):
        if field in data:
            result[field] = data[field]
    if "tau12_c" in data:
        if not reverse:
            result.update(
                {
                    "tau12_c": data["tau12_c"],
                    "tau12_d": data["tau12_d"],
                    "tau12_e": data.get("tau12_e", 0.0),
                    "tau12_f": data.get("tau12_f", 0.0),
                    "tau12_g": data.get("tau12_g", 0.0),
                    "tau21_c": data["tau21_c"],
                    "tau21_d": data["tau21_d"],
                    "tau21_e": data.get("tau21_e", 0.0),
                    "tau21_f": data.get("tau21_f", 0.0),
                    "tau21_g": data.get("tau21_g", 0.0),
                    "tau_tref": data.get("tau_tref", 298.15),
                }
            )
        else:
            result.update(
                {
                    "tau12_c": data["tau21_c"],
                    "tau12_d": data["tau21_d"],
                    "tau12_e": data.get("tau21_e", 0.0),
                    "tau12_f": data.get("tau21_f", 0.0),
                    "tau12_g": data.get("tau21_g", 0.0),
                    "tau21_c": data["tau12_c"],
                    "tau21_d": data["tau12_d"],
                    "tau21_e": data.get("tau12_e", 0.0),
                    "tau21_f": data.get("tau12_f", 0.0),
                    "tau21_g": data.get("tau12_g", 0.0),
                    "tau_tref": data.get("tau_tref", 298.15),
                }
            )
        return result

    if not reverse:
        result.update(
            {
                "a12_cal_per_mol": data["a12_cal_per_mol"],
                "a21_cal_per_mol": data["a21_cal_per_mol"],
            }
        )
    else:
        result.update(
            {
                "a12_cal_per_mol": data["a21_cal_per_mol"],
                "a21_cal_per_mol": data["a12_cal_per_mol"],
            }
        )
    return result


def orient_nrtl_interaction(data: dict, reverse: bool = False) -> dict:
    return _nrtl_oriented_record(data, reverse=reverse)


def nrtl_binary_interaction(cas1: Optional[str], cas2: Optional[str]) -> Optional[dict]:
    if not cas1 or not cas2 or cas1 == cas2:
        return None
    data = _nrtl_interactions().get(tuple(sorted((cas1, cas2))))
    if data is None:
        return None
    if data["cas1"] == cas1 and data["cas2"] == cas2:
        return _nrtl_oriented_record(data)
    return _nrtl_oriented_record(data, reverse=True)


@lru_cache(maxsize=1)
def _uniquac_interactions() -> dict[tuple[str, str], dict]:
    path = DATA_DIR / "uniquac_binary_interactions_cas.json"
    data = json.loads(path.read_text())["interactions"]
    interactions: dict[tuple[str, str], dict] = {}
    for record in data:
        if record.get("disabled"):
            continue
        cas1, cas2 = record["cas1"], record["cas2"]
        if cas1 == cas2:
            continue
        interaction = {
            "cas1": cas1,
            "cas2": cas2,
            "comment": record.get("comment", ""),
            "model_variant": record.get("model_variant", "standard_uniquac"),
            "use_q_prime": bool(record.get("use_q_prime", False)),
        }
        for field in ("extrapolation", "Tmin_K", "Tmax_K"):
            if field in record:
                interaction[field] = record[field]
        if "tau12_a" in record and "tau21_a" in record:
            interaction.update(
                {
                    "tau12_a": float(record["tau12_a"]),
                    "tau12_b": float(record.get("tau12_b", 0.0)),
                    "tau12_c": float(record.get("tau12_c", 0.0)),
                    "tau12_d": float(record.get("tau12_d", 0.0)),
                    "tau12_e": float(record.get("tau12_e", 0.0)),
                    "tau21_a": float(record["tau21_a"]),
                    "tau21_b": float(record.get("tau21_b", 0.0)),
                    "tau21_c": float(record.get("tau21_c", 0.0)),
                    "tau21_d": float(record.get("tau21_d", 0.0)),
                    "tau21_e": float(record.get("tau21_e", 0.0)),
                    "tau_tref": float(record.get("tau_tref", 298.15)),
                }
            )
        else:
            interaction.update(
                {
                    "a12_cal_per_mol": float(record["a12_cal_per_mol"]),
                    "a21_cal_per_mol": float(record["a21_cal_per_mol"]),
                }
            )
        key = tuple(sorted((cas1, cas2)))
        existing = interactions.get(key)
        if existing is not None:
            if "tau12_a" in interaction and "tau12_a" not in existing:
                interactions[key] = interaction
            continue
        interactions[key] = interaction
    return interactions


def _uniquac_oriented_record(data: dict, reverse: bool = False) -> dict:
    result = {
        "comment": data.get("comment", ""),
        "model_variant": data.get("model_variant", "standard_uniquac"),
        "use_q_prime": bool(data.get("use_q_prime", False)),
    }
    for field in ("extrapolation", "Tmin_K", "Tmax_K"):
        if field in data:
            result[field] = data[field]
    if "tau12_a" in data:
        if not reverse:
            result.update(
                {
                    "tau12_a": data["tau12_a"],
                    "tau12_b": data["tau12_b"],
                    "tau12_c": data.get("tau12_c", 0.0),
                    "tau12_d": data.get("tau12_d", 0.0),
                    "tau12_e": data.get("tau12_e", 0.0),
                    "tau21_a": data["tau21_a"],
                    "tau21_b": data["tau21_b"],
                    "tau21_c": data.get("tau21_c", 0.0),
                    "tau21_d": data.get("tau21_d", 0.0),
                    "tau21_e": data.get("tau21_e", 0.0),
                    "tau_tref": data.get("tau_tref", 298.15),
                }
            )
        else:
            result.update(
                {
                    "tau12_a": data["tau21_a"],
                    "tau12_b": data["tau21_b"],
                    "tau12_c": data.get("tau21_c", 0.0),
                    "tau12_d": data.get("tau21_d", 0.0),
                    "tau12_e": data.get("tau21_e", 0.0),
                    "tau21_a": data["tau12_a"],
                    "tau21_b": data["tau12_b"],
                    "tau21_c": data.get("tau12_c", 0.0),
                    "tau21_d": data.get("tau12_d", 0.0),
                    "tau21_e": data.get("tau12_e", 0.0),
                    "tau_tref": data.get("tau_tref", 298.15),
                }
            )
        return result

    if not reverse:
        result.update(
            {
                "a12_cal_per_mol": data["a12_cal_per_mol"],
                "a21_cal_per_mol": data["a21_cal_per_mol"],
            }
        )
    else:
        result.update(
            {
                "a12_cal_per_mol": data["a21_cal_per_mol"],
                "a21_cal_per_mol": data["a12_cal_per_mol"],
            }
        )
    return result


def orient_uniquac_interaction(data: dict, reverse: bool = False) -> dict:
    return _uniquac_oriented_record(data, reverse=reverse)


def uniquac_binary_interaction(
    cas1: Optional[str], cas2: Optional[str]
) -> Optional[dict]:
    if not cas1 or not cas2 or cas1 == cas2:
        return None
    data = _uniquac_interactions().get(tuple(sorted((cas1, cas2))))
    if data is None:
        return None
    if data["cas1"] == cas1 and data["cas2"] == cas2:
        return _uniquac_oriented_record(data)
    return _uniquac_oriented_record(data, reverse=True)


@lru_cache(maxsize=1)
def _uniquac_rq_payload() -> dict:
    path = DATA_DIR / "uniquac_rq_cas.json"
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def _uniquac_rq_components() -> dict[str, dict]:
    return _uniquac_rq_payload().get("components", {})


@lru_cache(maxsize=1)
def _uniquac_rq_aliases() -> dict[str, str]:
    return _uniquac_rq_payload().get("aliases", {})


def _public_uniquac_rq_entry(entry: dict) -> dict:
    result = {
        "r": float(entry["r"]),
        "q": float(entry["q"]),
        "source": entry.get("source", ""),
        "name": entry.get("name", ""),
        "cas": entry.get("cas", ""),
    }
    if "extended_uniquac" in entry:
        result["extended_uniquac"] = dict(entry["extended_uniquac"])
    if "q_prime" in entry:
        result["q_prime"] = float(entry["q_prime"])
    return result


def uniquac_rq_for_component(
    name: str,
    props: Optional[ChemicalProperties] = None,
) -> Optional[dict]:
    if props is not None:
        provided_r = getattr(props, "uniquac_r", None)
        provided_q = getattr(props, "uniquac_q", None)
        if provided_r is not None and provided_q is not None:
            return {
                "r": float(provided_r),
                "q": float(provided_q),
                "source": "provided",
                "name": getattr(props, "name", "") or name,
                "cas": getattr(props, "CAS", "") or "",
            }

    components = _uniquac_rq_components()
    aliases = _uniquac_rq_aliases()

    def add_candidate(values: list[str], value: Optional[str]) -> None:
        if value and value not in values:
            values.append(value)

    def looks_like_formula_candidate(value: Optional[str]) -> bool:
        if not value:
            return False
        try:
            if __package__ and __package__.split(".", 1)[0] == "pfdsim":
                from .compound_identity import looks_like_formula
            else:
                from compound_identity import looks_like_formula
            return looks_like_formula(str(value))
        except Exception:
            return False

    candidates = []
    add_candidate(candidates, name)

    if props is not None:
        add_candidate(candidates, props.CAS)
        add_candidate(candidates, props.name)
        add_candidate(candidates, props.symbol)

    try:
        if __package__ and __package__.split(".", 1)[0] == "pfdsim":
            from .compound_identity import get_compound_identity_resolver
        else:
            from compound_identity import get_compound_identity_resolver
        resolver = get_compound_identity_resolver()
        for identifier in candidates:
            if not identifier or looks_like_formula_candidate(identifier):
                continue
            cas = resolver.resolve_cas(str(identifier), allow_formula=False)
            if cas and cas in components:
                return _public_uniquac_rq_entry(components[cas])
        for identifier in list(candidates):
            if looks_like_formula_candidate(identifier):
                continue
            identity = resolver.resolve(str(identifier), allow_formula=False)
            if not identity:
                continue
            for value in identity.identifiers():
                if value == identity.formula or looks_like_formula_candidate(value):
                    continue
                add_candidate(candidates, value)
    except Exception:
        pass

    for candidate in candidates:
        if not candidate:
            continue
        if looks_like_formula_candidate(candidate):
            continue
        direct = _canonical_cas(candidate)
        cas = direct or aliases.get(_normalize_name(str(candidate)))
        if cas and cas in components:
            return _public_uniquac_rq_entry(components[cas])
    return None
