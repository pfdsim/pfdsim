"""Structured experimental phase-point parsing and consensus selection."""

from __future__ import annotations

import json
import math
import re
from statistics import median
from typing import Any, Iterable, Mapping, Optional


_NUMBER = r"(?<![A-Za-z0-9])([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
_OPTIONAL_UNIT = r"(?:(?:°\s*)?([CFK])\b)?"
_REQUIRED_UNIT = r"(?:°\s*)?([CFK])\b"
_RANGE_RE = re.compile(
    _NUMBER
    + r"\s*"
    + _OPTIONAL_UNIT
    + r"\s*(?:to|[-–—])\s*"
    + _NUMBER
    + r"\s*"
    + _OPTIONAL_UNIT,
    re.IGNORECASE,
)
_SCALAR_RE = re.compile(
    _NUMBER + r"\s*" + _REQUIRED_UNIT,
    re.IGNORECASE,
)


def _temperature_K(value: float, unit: str) -> float:
    unit = str(unit).upper()
    if unit == "K":
        return float(value)
    if unit == "C":
        return float(value) + 273.15
    if unit == "F":
        return (float(value) - 32.0) * 5.0 / 9.0 + 273.15
    raise ValueError(f"Unsupported temperature unit {unit!r}")


def _text_qualifiers(text: str, start: int) -> dict[str, Any]:
    lower = text.lower()
    prefix = lower[max(0, start - 32):start]
    polymorph_match = re.search(
        r"(?:^|[^a-z])"
        r"(alpha|beta|gamma|delta|form\s+[ivx0-9]+|phase\s+[ivx0-9]+)"
        r"\s*[-:]?\s*$",
        prefix,
    )
    return {
        "polymorph": polymorph_match.group(1) if polymorph_match else "",
        "anhydrous": "anhydrous" in lower,
        "hydrate": bool(re.search(r"\b(?:mono|di|tri)?hydrate\b", lower)),
        "decomposition": bool(re.search(r"\bdecomp(?:oses|osition|\.)?\b", lower)),
        "bound": bool(re.search(r"(?:[<>]=?|\babove\b|\bbelow\b)\s*$", prefix)),
    }


def _candidate(
    *,
    value_K: float,
    low_K: float,
    high_K: float,
    uncertainty_K: Optional[float],
    source: str,
    method: str,
    reference: str,
    comment: str,
    raw: str,
    qualifiers: Mapping[str, Any],
    sample_count: Optional[int] = None,
) -> dict[str, Any]:
    return {
        "value_K": float(value_K),
        "low_K": float(low_K),
        "high_K": float(high_K),
        "uncertainty_K": (
            None if uncertainty_K is None else float(abs(uncertainty_K))
        ),
        "source": str(source),
        "method": str(method),
        "reference": str(reference or ""),
        "comment": str(comment or ""),
        "raw": str(raw),
        "qualifiers": dict(qualifiers),
        "sample_count": None if sample_count is None else int(sample_count),
    }


def parse_reported_temperature_candidates(
    text: str,
    *,
    source: str,
    method: str,
    reference: str = "",
    comment: str = "",
) -> list[dict[str, Any]]:
    """Parse all unit-qualified scalar/range temperatures in one report."""
    raw = str(text or "").replace("−", "-").replace("Â", "")
    lower = raw.lower()
    if not raw.strip():
        return []
    if "decomposition temp" in lower or "decomposition point" in lower:
        return []
    if "sublim" in lower and not re.search(r"\b(?:melt|fus|freez)", lower):
        return []

    candidates: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for match in _RANGE_RE.finditer(raw):
        first, first_unit, second, second_unit = match.groups()
        unit1 = first_unit or second_unit
        unit2 = second_unit or first_unit
        if unit1 is None or unit2 is None:
            continue
        try:
            value1 = _temperature_K(float(first), unit1)
            value2 = _temperature_K(float(second), unit2)
        except (TypeError, ValueError):
            continue
        low, high = sorted((value1, value2))
        if low <= 0.0 or high >= 5000.0:
            continue
        qualifiers = _text_qualifiers(raw, match.start())
        candidates.append(_candidate(
            value_K=0.5 * (low + high),
            low_K=low,
            high_K=high,
            uncertainty_K=0.5 * (high - low),
            source=source,
            method=f"{method}_range",
            reference=reference,
            comment=comment,
            raw=raw,
            qualifiers=qualifiers,
        ))
        occupied.append(match.span())

    def inside_range(span: tuple[int, int]) -> bool:
        return any(start <= span[0] and span[1] <= stop for start, stop in occupied)

    for match in _SCALAR_RE.finditer(raw):
        if inside_range(match.span()):
            continue
        value_text, unit = match.groups()
        try:
            value = _temperature_K(float(value_text), unit)
        except (TypeError, ValueError):
            continue
        if value <= 0.0 or value >= 5000.0:
            continue
        candidates.append(_candidate(
            value_K=value,
            low_K=value,
            high_K=value,
            uncertainty_K=None,
            source=source,
            method=f"{method}_scalar",
            reference=reference,
            comment=comment,
            raw=raw,
            qualifiers=_text_qualifiers(raw, match.start()),
        ))
    return candidates


def nist_temperature_candidate(
    value_text: str,
    unit: str,
    *,
    method: str,
    reference: str,
    comment: str,
) -> Optional[dict[str, Any]]:
    """Parse one structured NIST Tfus/Ttriple row."""
    raw_value = str(value_text or "").replace("−", "-")
    normalized_unit = str(unit or "").strip().lower()
    if normalized_unit not in {"k", "kelvin", "c", "°c", "f", "°f"}:
        return None
    unit_letter = "K" if normalized_unit in {"k", "kelvin"} else normalized_unit[-1].upper()
    central_match = re.search(r"[-+]?\d+(?:\.\d*)?", raw_value)
    if central_match is None:
        return None
    try:
        central_value = _temperature_K(float(central_match.group(0)), unit_letter)
    except ValueError:
        return None
    if not math.isfinite(central_value) or central_value <= 0.0:
        return None
    candidate = _candidate(
        value_K=central_value,
        low_K=central_value,
        high_K=central_value,
        uncertainty_K=None,
        source="nist",
        method=(
            "nist_avg"
            if str(method).strip().lower() in {"avg", "average"}
            else "nist_reported"
        ),
        reference=reference,
        comment=comment,
        raw=raw_value,
        qualifiers=_text_qualifiers(raw_value, central_match.start()),
    )
    uncertainty_match = re.search(
        r"(?:±|\+/-|&plusmn;)\s*(\d+(?:\.\d*)?)",
        raw_value,
        re.IGNORECASE,
    )
    if uncertainty_match is None:
        uncertainty_match = re.search(
            r"uncertainty[^0-9]{0,40}(\d+(?:\.\d*)?)\s*([CFK])?",
            str(comment or ""),
            re.IGNORECASE,
        )
    if uncertainty_match:
        uncertainty = float(uncertainty_match.group(1))
        uncertainty_unit = (
            uncertainty_match.group(2).upper()
            if uncertainty_match.lastindex and uncertainty_match.lastindex >= 2
            and uncertainty_match.group(2)
            else unit_letter
        )
        if uncertainty_unit == "F":
            uncertainty *= 5.0 / 9.0
        candidate["uncertainty_K"] = uncertainty
        candidate["low_K"] = candidate["value_K"] - uncertainty
        candidate["high_K"] = candidate["value_K"] + uncertainty
    sample_match = re.search(
        r"average\s+of\s+(\d+)(?:\s+out\s+of\s+(\d+))?",
        str(comment or ""),
        re.IGNORECASE,
    )
    if sample_match:
        candidate["sample_count"] = int(sample_match.group(1))
        if sample_match.group(2):
            candidate["reported_count"] = int(sample_match.group(2))
    candidate["method"] = (
        "nist_avg" if str(method).strip().lower() in {"avg", "average"}
        else "nist_reported"
    )
    candidate["raw"] = raw_value
    return candidate


def legacy_temperature_candidate(
    value: Any,
    *,
    source: str,
    method: str,
    notes: str = "",
) -> Optional[dict[str, Any]]:
    try:
        value_K = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_K) or value_K <= 0.0:
        return None
    return _candidate(
        value_K=value_K,
        low_K=value_K,
        high_K=value_K,
        uncertainty_K=None,
        source=source,
        method=method,
        reference="",
        comment=notes,
        raw=f"legacy selected value {value_K:g} K",
        qualifiers={},
    )


def _effective_uncertainty(candidate: Mapping[str, Any]) -> float:
    uncertainty = candidate.get("uncertainty_K")
    try:
        uncertainty = float(uncertainty)
    except (TypeError, ValueError):
        uncertainty = 0.0
    if uncertainty > 0.0:
        return max(0.25, uncertainty)
    if float(candidate.get("high_K", 0.0)) > float(candidate.get("low_K", 0.0)):
        return max(
            0.5,
            0.5 * (
                float(candidate["high_K"])
                - float(candidate["low_K"])
            ),
        )
    return 1.0 if candidate.get("source") == "pubchem" else 1.5


def _candidate_evidence(candidate: Mapping[str, Any]) -> float:
    source = str(candidate.get("source") or "")
    method = str(candidate.get("method") or "")
    evidence = 1.0 if source == "pubchem" else 1.1
    if method == "nist_avg":
        evidence = 1.4
        count = candidate.get("sample_count")
        if isinstance(count, int) and count > 1:
            evidence *= min(1.5, 1.0 + math.log10(count) / 5.0)
    if method.endswith("_range"):
        evidence *= 0.85
    qualifiers = candidate.get("qualifiers") or {}
    if qualifiers.get("decomposition"):
        evidence *= 0.45
    if qualifiers.get("bound"):
        evidence *= 0.20
    return evidence


def _clusters(candidates: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    remaining = set(range(len(candidates)))
    clusters = []
    while remaining:
        connected = {remaining.pop()}
        changed = True
        while changed:
            changed = False
            for index in list(remaining):
                candidate = candidates[index]
                if any(
                    (
                        bool((candidate.get('qualifiers') or {}).get('hydrate'))
                        == bool((candidates[other].get('qualifiers') or {}).get('hydrate'))
                    )
                    and (
                        abs(float(candidate["value_K"]) - float(candidates[other]["value_K"]))
                        <= max(
                            3.0,
                            1.25 * (
                                _effective_uncertainty(candidate)
                                + _effective_uncertainty(candidates[other])
                            ),
                        )
                    )
                    for other in connected
                ):
                    remaining.remove(index)
                    connected.add(index)
                    changed = True
        clusters.append([candidates[index] for index in sorted(connected)])
    return clusters


def select_temperature_consensus(
    property_name: str,
    candidates: Iterable[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    """Select an uncertainty-aware consensus from NIST and PubChem reports."""
    valid = []
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        try:
            value = float(candidate["value_K"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and 0.0 < value < 5000.0:
            valid.append(candidate)
    if not valid:
        return None

    exact = [
        item for item in valid
        if not (item.get("qualifiers") or {}).get("bound")
    ]
    if exact:
        valid = exact
    precise = [item for item in valid if _effective_uncertainty(item) < 10.0]
    broad = []
    if len(precise) >= 2:
        broad = [item for item in valid if _effective_uncertainty(item) >= 10.0]
        valid = precise

    clusters = _clusters(valid)
    ranked = []
    for cluster in clusters:
        score = sum(_candidate_evidence(item) for item in cluster)
        sources = {str(item.get("source") or "") for item in cluster}
        if len(sources) > 1:
            score *= 1.15
        spread = max(float(item["value_K"]) for item in cluster) - min(
            float(item["value_K"]) for item in cluster
        )
        ranked.append((score, -spread, len(cluster), cluster))
    ranked.sort(key=lambda item: item[:3], reverse=True)
    selected_score, _negative_spread, _count, selected = ranked[0]

    weights = [
        _candidate_evidence(item) / (_effective_uncertainty(item) ** 2)
        for item in selected
    ]
    weight_sum = sum(weights)
    value = sum(
        weight * float(item["value_K"])
        for weight, item in zip(weights, selected)
    ) / weight_sum
    spread = max(float(item["value_K"]) for item in selected) - min(
        float(item["value_K"]) for item in selected
    )
    sources = {str(item.get("source") or "") for item in selected}
    nist_avg = any(item.get("method") == "nist_avg" for item in selected)

    if len(sources) > 1:
        quality = 0.985
        method = f"nist_pubchem_{property_name}_consensus"
    elif sources == {"pubchem"} and len(selected) >= 3:
        quality = 0.970
        method = f"pubchem_{property_name}_consensus"
    elif sources == {"pubchem"} and len(selected) == 2:
        quality = 0.945
        method = f"pubchem_{property_name}_consensus"
    elif sources == {"nist"} and nist_avg:
        uncertainty = min(_effective_uncertainty(item) for item in selected)
        quality = 0.965 if uncertainty <= 2.0 else 0.920
        method = f"nist_avg_{property_name}"
    elif sources == {"nist"}:
        if len(selected) == 1:
            try:
                uncertainty = float(selected[0].get('uncertainty_K'))
            except (TypeError, ValueError):
                uncertainty = _effective_uncertainty(selected[0])
            quality = (
                0.960 if uncertainty <= 0.10
                else 0.940 if uncertainty <= 1.0
                else 0.915
            )
        else:
            quality = 0.945
        method = f"nist_{property_name}_consensus"
    else:
        quality = 0.880
        method = f"pubchem_reported_{property_name}"

    if spread > 5.0:
        quality -= 0.04
    elif spread > 2.0:
        quality -= 0.015
    if selected and all(
        (item.get("qualifiers") or {}).get("decomposition")
        for item in selected
    ):
        quality = min(quality, 0.78)
    if len(selected) == 1:
        uncertainty = _effective_uncertainty(selected[0])
        if uncertainty > 20.0:
            quality = min(quality, 0.65)
        elif uncertainty > 10.0:
            quality = min(quality, 0.72)
        elif uncertainty > 5.0:
            quality = min(quality, 0.82)
        elif uncertainty > 2.0:
            quality = min(quality, 0.90)
    if selected and all(
        (item.get('qualifiers') or {}).get('bound')
        for item in selected
    ):
        quality = min(quality, 0.62)

    alternatives = ranked[1:]
    material_conflict = bool(
        alternatives and alternatives[0][0] >= 0.35 * selected_score
    )
    if material_conflict:
        quality -= 0.08
    quality = max(0.0, min(0.995, quality))

    source_names = "+".join(sorted(sources))
    notes = (
        f"Selected {property_name}={value:.8g} K from {len(selected)} "
        f"consistent {source_names} report(s); selected spread={spread:.6g} K"
    )
    if broad:
        notes += (
            f"; ignored {len(broad)} broad-uncertainty report(s): "
            + ", ".join(
                f"{item.get('source')} {float(item['value_K']):.6g}±"
                f"{_effective_uncertainty(item):.6g} K"
                for item in broad
            )
        )
    if alternatives:
        notes += f"; rejected {sum(len(item[3]) for item in alternatives)} outlier report(s)"
    if material_conflict:
        notes += "; material alternate cluster reduced quality"

    return {
        "value": value,
        "quality": quality,
        "method": method,
        "notes": notes,
        "selected_candidates": selected,
        "broad_candidates": broad,
        "outlier_candidates": [item for ranked_item in alternatives for item in ranked_item[3]],
    }


def select_pressure_consensus(
    property_name: str,
    candidates: Iterable[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    valid = []
    for raw in candidates:
        candidate = dict(raw)
        try:
            value = float(candidate["value_bar"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            valid.append(candidate)
    if not valid:
        return None
    values = sorted(float(item["value_bar"]) for item in valid)
    selected = median(values)
    relative_spread = 0.0 if len(values) == 1 else (values[-1] - values[0]) / selected
    if relative_spread > 0.20:
        return None
    sources = {str(item.get("source") or "") for item in valid}
    quality = 0.97 if len(sources) > 1 else (0.94 if len(valid) > 1 else 0.88)
    return {
        "value": selected,
        "quality": quality,
        "method": (
            f"nist_pubchem_{property_name}_consensus"
            if len(sources) > 1
            else f"{next(iter(sources), 'online')}_{property_name}_consensus"
        ),
        "notes": (
            f"Selected {property_name}={selected:.8g} bar from {len(valid)} "
            f"online report(s); relative spread={relative_spread:.3%}"
        ),
        "selected_candidates": valid,
    }


def candidate_json(candidate: Mapping[str, Any]) -> str:
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))
