"""Shared access to the published 2005 PSRK pure-component table."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


PSRK_PURE_COMPONENT_PATH = (
    Path(__file__).resolve().parent / "data" / "psrk" / "pure_components.json"
)


@lru_cache(maxsize=1)
def psrk_component_records() -> dict[str, dict[str, Any]]:
    """Return the PSRK pure-component table keyed by canonical CAS number."""
    try:
        payload = json.loads(PSRK_PURE_COMPONENT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    records = payload.get("components") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        cas = str(record.get("CAS") or "").strip()
        if cas and cas not in result:
            result[cas] = record
    return result


def psrk_component_record(cas: Optional[str]) -> Optional[dict[str, Any]]:
    if not cas:
        return None
    return psrk_component_records().get(str(cas).strip())


def psrk_unstarred_critical_values(cas: Optional[str]) -> dict[str, float]:
    """Return independently unstarred PSRK Tc [K] and Pc [bar]."""
    record = psrk_component_record(cas)
    if not record:
        return {}
    result: dict[str, float] = {}
    if not bool(record.get("Tc_estimated")):
        try:
            value = float(record["Tc_K"])
            if math.isfinite(value) and value > 0.0:
                result["Tc"] = value
        except (KeyError, TypeError, ValueError):
            pass
    if not bool(record.get("Pc_estimated")):
        try:
            value = float(record["Pc_kPa"]) / 100.0
            if math.isfinite(value) and value > 0.0:
                result["Pc"] = value
        except (KeyError, TypeError, ValueError):
            pass
    return result


def compatible_psrk_mathias_copeman(
    cas: Optional[str],
    Tc_K: object,
    Pc_bar: object,
    *,
    relative_tolerance: float = 0.01,
) -> Optional[dict[str, Any]]:
    """Return PSRK Mathias-Copeman data when its critical bundle is compatible."""
    record = psrk_component_record(cas)
    if not record or bool(record.get("Tc_estimated")) or bool(record.get("Pc_estimated")):
        return None
    try:
        selected_Tc = float(Tc_K)
        selected_Pc = float(Pc_bar)
        psrk_Tc = float(record["Tc_K"])
        psrk_Pc = float(record["Pc_kPa"]) / 100.0
    except (KeyError, TypeError, ValueError):
        return None
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (selected_Tc, selected_Pc, psrk_Tc, psrk_Pc)
    ):
        return None
    if (
        abs(selected_Tc / psrk_Tc - 1.0) > relative_tolerance
        or abs(selected_Pc / psrk_Pc - 1.0) > relative_tolerance
    ):
        return None
    parameters = record.get("mathias_copeman")
    if not isinstance(parameters, dict):
        return None
    try:
        c1 = float(parameters["c1"])
        c2 = float(parameters["c2"])
        c3 = float(parameters["c3"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (c1, c2, c3)):
        return None
    return {
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "Tmin_K": parameters.get("Tmin_K"),
        "Tmax_K": parameters.get("Tmax_K"),
        "source": "PSRK 2005 supplementary pure-component table",
        "psrk_Tc_K": psrk_Tc,
        "psrk_Pc_bar": psrk_Pc,
    }
