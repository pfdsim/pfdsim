"""Fetch, parse, and normalize PubChem experimental viscosity annotations.

This module deliberately has no resolver integration.  PubChem PUG-View
publishes viscosity as attributed, free-form text, so acquisition, parsing,
and anchor-eligibility normalization remain separate and auditable here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Callable, Iterable, Mapping, Optional
import urllib.error
import urllib.parse
import urllib.request


PUBCHEM_PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
PUBCHEM_USER_AGENT = "PFD-Editor/1.0"

_NUMBER = r"[+-]?(?:\d+(?:[.,]\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TEMPERATURE_UNIT = r"(?:(?:°|deg(?:rees?)?)\s*)?(?P<tunit>K|C|F)\b"
_DYNAMIC_UNIT = (
    r"(?P<vunit>"
    r"milli\s*pascal\s*[-·. ]?\s*seconds?|"
    r"micro\s*pascal\s*[-·. ]?\s*seconds?|"
    r"(?:m|u|µ|μ)\s*pa\s*[-·.* ]?\s*s|"
    r"pa\s*[-·.* ]?\s*s|"
    r"kg\s*/\s*\(?\s*m\s*[-·.* ]?\s*s\s*\)?|"
    r"n\s*[-·.* ]?\s*s\s*/\s*(?:m\s*(?:2|²|\^2)|sq\.?\s*m)|"
    r"cent[ai]?\s*poise|centipoises?|c\s*p(?:s)?\b|"
    r"millipoises?|m\s*p\b|"
    r"poises?|p\b"
    r")"
)
_VALUE_FIRST = re.compile(
    rf"(?P<value>{_NUMBER})\s*{_DYNAMIC_UNIT}"
    rf"(?:\s*\([^)]{{0,24}}\))?\s*(?:at|@|,|:|/)?\s*"
    rf"(?P<temperature>{_NUMBER})\s*{_TEMPERATURE_UNIT}",
    re.IGNORECASE,
)
_TEMPERATURE_FIRST = re.compile(
    rf"(?P<temperature>{_NUMBER})\s*{_TEMPERATURE_UNIT}"
    rf"\s*(?:at|@|,|:|=|-)?\s*"
    rf"(?P<value>{_NUMBER})\s*{_DYNAMIC_UNIT}",
    re.IGNORECASE,
)
_SHARED_UNIT_PAIR = re.compile(
    rf"(?P<value>{_NUMBER})\s*(?:at|@)\s*"
    rf"(?P<temperature>{_NUMBER})\s*{_TEMPERATURE_UNIT}",
    re.IGNORECASE,
)
_SHARED_DYNAMIC_UNIT = re.compile(
    rf"\ball\s+(?:values?\s+)?(?:are\s+)?(?:in|as)\s+{_DYNAMIC_UNIT}",
    re.IGNORECASE,
)
_PRESSURE = re.compile(
    rf"(?P<pressure>{_NUMBER})\s*"
    r"(?P<punit>bar|kpa|mpa|pa|atm(?:osphere)?s?|mm\s*hg|torr)\b",
    re.IGNORECASE,
)
_KINEMATIC_UNIT = re.compile(
    r"(?:mm|cm|m|ft|in)\s*(?:2|²|\^2)\s*/\s*s|"
    r"\bsq\.?\s*(?:mm|cm|m|ft|in)\s*/\s*(?:s|sec(?:ond)?s?)\b|"
    r"\b(?:centi)?stokes?\b|\bcst\b|"
    r"\b(?:saybolt|redwood|engler|ssu|sus)\b",
    re.IGNORECASE,
)
_NON_DYNAMIC_KIND = re.compile(
    r"\b(?:kinematic|intrinsic|reduced|relative|specific)\s+viscosity\b|"
    r"\bviscosity\s+(?:index|number)\b",
    re.IGNORECASE,
)
_NON_PURE_BASIS = re.compile(
    r"\b(?:aqueous|solution|mixture|blend|formulation|suspension|emulsion|"
    r"slurry|dispersion|serum|plasma|urine|blood|broth|extract|"
    r"isomers|reaction\s+mass|"
    r"commercial\s+(?:grade|product)|saturated\s+(?:aqueous\s+)?solution)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:wt|vol|mol|mass)\s*%|"
    r"\b(?:weight|volume|mole)\s+percent\b",
    re.IGNORECASE,
)
_VAPOR_BASIS = re.compile(
    r"\b(?:gas(?:eous)?|vapou?r)(?:\s+phase)?\s+viscosity\b|"
    r"\bviscosity\s+of\s+(?:the\s+)?(?:gas|vapou?r)\b",
    re.IGNORECASE,
)
_PREDICTIVE = re.compile(
    r"\b(?:estimated|estimate|predicted|prediction|calculated|modeled|"
    r"modelled|qspr|q-spr|epi\s*suite)\b",
    re.IGNORECASE,
)
_INEQUALITY_OR_APPROXIMATION = re.compile(
    r"(?:<=|>=|<|>|≤|≥|~|≈|±|\+\s*/\s*-)|"
    r"\b(?:about|approx(?:\.|imately)?|less\s+than|more\s+than|"
    r"not\s+(?:less|more)\s+than|plus\s+or\s+minus|"
    r"maximum|minimum|max\.|min\.)\b",
    re.IGNORECASE,
)
_VALUE_RANGE = re.compile(
    rf"{_NUMBER}\s*(?:-|–|—|\bto\b)\s*{_NUMBER}\s*{_DYNAMIC_UNIT}",
    re.IGNORECASE,
)
_TEMPERATURE_RANGE = re.compile(
    rf"(?:at|@)\s*{_NUMBER}\s*(?:-|–|—|\bto\b)\s*"
    rf"{_NUMBER}\s*(?:°\s*)?[KCF]\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PubChemViscosityAnnotation:
    """One attributed free-form value from a PUG-View Viscosity section."""

    text: str
    record_title: str = ""
    reference_number: Optional[int] = None
    references: tuple[str, ...] = ()
    source_name: str = ""
    source_url: str = ""
    comment: str = ""
    peer_reviewed: bool = False
    predictive: bool = False


@dataclass(frozen=True)
class ParsedViscosityObservation:
    """Syntactically parsed dynamic-viscosity observation before SI checks."""

    value: float
    unit: str
    temperature: float
    temperature_unit: str
    pressure: Optional[float]
    pressure_unit: Optional[str]
    raw: str
    annotation: PubChemViscosityAnnotation


@dataclass(frozen=True)
class NormalizedViscosityPoint:
    """Normalized dynamic-viscosity point in SI units with phase metadata."""

    viscosity_Pa_s: float
    temperature_K: float
    pressure_bar: Optional[float]
    phase_basis: str
    source: str
    method: str
    record_title: str
    reference_number: Optional[int]
    reference: str
    source_name: str
    source_url: str
    comment: str
    peer_reviewed: bool
    raw: str


@dataclass(frozen=True)
class RejectedViscosityObservation:
    """A PubChem annotation that was not safe to use as a liquid anchor."""

    raw: str
    reason: str
    reference: str = ""
    source_name: str = ""


@dataclass(frozen=True)
class ViscosityNormalizationResult:
    points: tuple[NormalizedViscosityPoint, ...]
    rejected: tuple[RejectedViscosityObservation, ...]


@dataclass(frozen=True)
class PubChemViscosityResult:
    identifier: str
    cid: int
    annotations: tuple[PubChemViscosityAnnotation, ...]
    points: tuple[NormalizedViscosityPoint, ...]
    rejected: tuple[RejectedViscosityObservation, ...]


class PubChemViscosityFetchError(LookupError):
    """A transient or malformed PubChem viscosity response."""


def _annotation_reference(annotation: PubChemViscosityAnnotation) -> str:
    return "; ".join(item for item in annotation.references if item)


def _clean_text(text: Any) -> str:
    value = str(text or "")
    for broken, repaired in (
        ("Â", ""),
        ("â€“", "–"),
        ("â€”", "—"),
        ("â‰¤", "≤"),
        ("â‰¥", "≥"),
        ("Î¼", "µ"),
        ("μ", "µ"),
    ):
        value = value.replace(broken, repaired)
    return re.sub(r"\s+", " ", value).strip()


def _number_value(text: str) -> float:
    value = str(text).strip()
    if "," in value:
        if "." in value or re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", value):
            value = value.replace(",", "")
        else:
            value = value.replace(",", ".")
    return float(value)


def _measurement_text(text: str) -> str:
    """Normalize harmless typography without inventing missing conditions."""
    cleaned = _clean_text(text)
    return re.sub(
        rf"\(\s*(?P<temperature>{_NUMBER})\s*"
        r"(?:(?:°|deg(?:rees?)?)\s*)?(?P<tunit>[KCF])\s*\)",
        r" at \g<temperature> °\g<tunit>",
        cleaned,
        flags=re.IGNORECASE,
    )


def _information_texts(information: Mapping[str, Any]) -> list[str]:
    value = information.get("Value")
    if not isinstance(value, Mapping):
        return []
    texts = [
        _clean_text(marked.get("String"))
        for marked in value.get("StringWithMarkup", ()) or ()
        if isinstance(marked, Mapping) and marked.get("String")
    ]
    unit = _clean_text(value.get("Unit"))
    texts.extend(
        _clean_text(f"{number} {unit}")
        for number in value.get("Number", ()) or ()
    )
    return [text for text in texts if text]


def extract_pubchem_viscosity_annotations(
    payload: Mapping[str, Any],
) -> tuple[PubChemViscosityAnnotation, ...]:
    """Extract attributed raw strings from every PUG-View Viscosity section."""
    root = payload.get("Record", payload)
    if not isinstance(root, Mapping):
        return ()
    reference_map: dict[int, Mapping[str, Any]] = {}
    record_title = _clean_text(root.get("RecordTitle"))
    for reference in root.get("Reference", ()) or ():
        if not isinstance(reference, Mapping):
            continue
        try:
            number = int(reference.get("ReferenceNumber"))
        except (TypeError, ValueError):
            continue
        reference_map[number] = reference

    annotations: list[PubChemViscosityAnnotation] = []

    def walk(node: Mapping[str, Any], inside_viscosity: bool = False) -> None:
        heading = _clean_text(node.get("TOCHeading"))
        heading_is_viscosity = bool(re.search(r"\bviscosit", heading, re.I))
        active = inside_viscosity or heading_is_viscosity
        if active:
            for information in node.get("Information", ()) or ():
                if not isinstance(information, Mapping):
                    continue
                try:
                    reference_number = int(information.get("ReferenceNumber"))
                except (TypeError, ValueError):
                    reference_number = None
                source = reference_map.get(reference_number, {})
                inline_references = information.get("Reference", ()) or ()
                if isinstance(inline_references, str):
                    inline_references = (inline_references,)
                references = tuple(
                    _clean_text(item) for item in inline_references if item
                )
                if not references:
                    references = tuple(
                        _clean_text(item.get("Citation"))
                        for item in information.get("ExtendedReference", ()) or ()
                        if isinstance(item, Mapping) and item.get("Citation")
                    )
                if not references and source.get("Name"):
                    references = (_clean_text(source.get("Name")),)
                comment = _clean_text(
                    " ".join(
                        str(item)
                        for item in (
                            information.get("Name"),
                            information.get("Description"),
                        )
                        if item
                    )
                )
                source_name = _clean_text(source.get("SourceName"))
                source_url = _clean_text(source.get("URL"))
                peer_reviewed = "peer reviewed" in comment.lower()
                for text in _information_texts(information):
                    predictive_text = f"{comment} {text}"
                    annotations.append(PubChemViscosityAnnotation(
                        text=text,
                        record_title=record_title,
                        reference_number=reference_number,
                        references=references,
                        source_name=source_name,
                        source_url=source_url,
                        comment=comment,
                        peer_reviewed=peer_reviewed,
                        predictive=bool(_PREDICTIVE.search(predictive_text)),
                    ))
        for child in node.get("Section", ()) or ():
            if isinstance(child, Mapping):
                walk(child, active)

    walk(root)
    unique: dict[tuple[Any, ...], PubChemViscosityAnnotation] = {}
    for annotation in annotations:
        key = (
            annotation.text.casefold(),
            annotation.record_title.casefold(),
            annotation.reference_number,
            annotation.references,
            annotation.source_name.casefold(),
            annotation.comment.casefold(),
        )
        unique.setdefault(key, annotation)
    return tuple(unique.values())


def _dynamic_factor(unit: str) -> Optional[float]:
    normalized = re.sub(r"[\s·.\-/*()^]", "", _clean_text(unit).lower())
    if normalized in {"pas", "pascalsecond", "pascalseconds"}:
        return 1.0
    if normalized in {
        "mpas", "millipascalsecond", "millipascalseconds",
        "cp", "centipoise", "centipoises", "centapoise",
        "centpoise", "cps",
    }:
        return 1.0e-3
    if normalized in {"millipoise", "millipoises", "mp"}:
        return 1.0e-4
    if normalized in {
        "upas", "µpas", "micropascalsecond", "micropascalseconds",
    }:
        return 1.0e-6
    if normalized in {"p", "poise", "poises"}:
        return 0.1
    if normalized in {
        "kgms", "nsm2", "nsm²", "nssqm",
    }:
        return 1.0
    return None


def _temperature_K(value: float, unit: str) -> Optional[float]:
    unit = unit.upper()
    if unit == "K":
        temperature = value
    elif unit == "C":
        temperature = value + 273.15
    elif unit == "F":
        temperature = (value - 32.0) * 5.0 / 9.0 + 273.15
    else:
        return None
    return temperature if temperature > 0.0 and math.isfinite(temperature) else None


def _pressure_bar(value: float, unit: str) -> Optional[float]:
    normalized = re.sub(r"\s+", "", unit.lower())
    factors = {
        "bar": 1.0,
        "kpa": 0.01,
        "mpa": 10.0,
        "pa": 1.0e-5,
        "atm": 1.01325,
        "atmosphere": 1.01325,
        "atmospheres": 1.01325,
        "mmhg": 1.333223684e-3,
        "torr": 1.333223684e-3,
    }
    factor = factors.get(normalized)
    if factor is None:
        return None
    pressure = value * factor
    return pressure if pressure > 0.0 and math.isfinite(pressure) else None


def _rejection(
    annotation: PubChemViscosityAnnotation,
    reason: str,
    raw: Optional[str] = None,
) -> RejectedViscosityObservation:
    return RejectedViscosityObservation(
        raw=raw or annotation.text,
        reason=reason,
        reference=_annotation_reference(annotation),
        source_name=annotation.source_name,
    )


def parse_pubchem_viscosity_annotation(
    annotation: PubChemViscosityAnnotation,
) -> tuple[tuple[ParsedViscosityObservation, ...], tuple[RejectedViscosityObservation, ...]]:
    """Parse exact dynamic-viscosity/temperature pairs from one annotation."""
    text = _measurement_text(annotation.text)
    context = _clean_text(
        f"{annotation.record_title} {annotation.comment} {text}"
    )
    if annotation.predictive or _PREDICTIVE.search(context):
        return (), (_rejection(annotation, "predictive or calculated value"),)
    if _NON_PURE_BASIS.search(context):
        return (), (_rejection(annotation, "solution, mixture, or product value"),)
    if _NON_DYNAMIC_KIND.search(context):
        return (), (_rejection(annotation, "not ordinary dynamic viscosity"),)
    if _VAPOR_BASIS.search(context):
        return (), (_rejection(annotation, "explicit gas or vapor viscosity"),)
    if _INEQUALITY_OR_APPROXIMATION.search(text):
        return (), (_rejection(annotation, "limit, inequality, or approximate value"),)
    if _VALUE_RANGE.search(text) or _TEMPERATURE_RANGE.search(text):
        return (), (_rejection(annotation, "value or temperature range"),)

    shared_match = _SHARED_DYNAMIC_UNIT.search(text)
    shared_unit = shared_match.group("vunit") if shared_match else None
    parsed: list[ParsedViscosityObservation] = []
    rejected: list[RejectedViscosityObservation] = []
    clauses = [
        clause.strip(" ,")
        for clause in re.split(r"[;\n•]+", text)
        if clause.strip(" ,")
    ]
    for clause in clauses:
        if _KINEMATIC_UNIT.search(clause):
            rejected.append(_rejection(
                annotation,
                "kinematic viscosity requires a compatible density",
                clause,
            ))
            continue
        matches = list(_VALUE_FIRST.finditer(clause))
        matches.extend(_TEMPERATURE_FIRST.finditer(clause))
        if not matches and shared_unit:
            matches = list(_SHARED_UNIT_PAIR.finditer(clause))
        accepted_matches = []
        for match in sorted(matches, key=lambda item: item.start()):
            if accepted_matches and match.start() < accepted_matches[-1].end():
                continue
            accepted_matches.append(match)
        for index, match in enumerate(accepted_matches):
            unit = match.groupdict().get("vunit") or shared_unit
            if unit is None:
                continue
            try:
                value = _number_value(match.group("value"))
                temperature = _number_value(match.group("temperature"))
            except (TypeError, ValueError):
                continue
            pressure = None
            pressure_unit = None
            next_start = (
                accepted_matches[index + 1].start()
                if index + 1 < len(accepted_matches)
                else len(clause)
            )
            pressure_match = _PRESSURE.search(clause, match.end(), next_start)
            if pressure_match is not None:
                pressure = _number_value(pressure_match.group("pressure"))
                pressure_unit = pressure_match.group("punit")
            parsed.append(ParsedViscosityObservation(
                value=value,
                unit=unit,
                temperature=temperature,
                temperature_unit=match.group("tunit"),
                pressure=pressure,
                pressure_unit=pressure_unit,
                raw=clause,
                annotation=annotation,
            ))
        if matches:
            continue
        if _KINEMATIC_UNIT.search(text):
            reason = "kinematic viscosity requires a compatible density"
        elif re.search(_DYNAMIC_UNIT, clause, re.IGNORECASE):
            reason = "dynamic viscosity lacks an explicit numerical temperature"
        else:
            reason = "no supported dynamic-viscosity point"
        rejected.append(_rejection(annotation, reason, clause))
    return tuple(parsed), tuple(rejected)


def normalize_pubchem_viscosity_annotations(
    annotations: Iterable[PubChemViscosityAnnotation],
) -> ViscosityNormalizationResult:
    """Return exact SI dynamic-viscosity points with unsafe bases rejected."""
    points: list[NormalizedViscosityPoint] = []
    rejected: list[RejectedViscosityObservation] = []
    for annotation in annotations:
        annotation_context = _clean_text(
            f"{annotation.record_title} {annotation.comment} {annotation.text}"
        )
        parsed, parse_rejections = parse_pubchem_viscosity_annotation(annotation)
        rejected.extend(parse_rejections)
        for observation in parsed:
            factor = _dynamic_factor(observation.unit)
            temperature = _temperature_K(
                observation.temperature,
                observation.temperature_unit,
            )
            if factor is None:
                rejected.append(_rejection(
                    annotation,
                    f"unsupported dynamic-viscosity unit {observation.unit!r}",
                    observation.raw,
                ))
                continue
            viscosity = observation.value * factor
            if viscosity <= 0.0 or not math.isfinite(viscosity):
                rejected.append(_rejection(
                    annotation,
                    "nonpositive or nonfinite viscosity",
                    observation.raw,
                ))
                continue
            if temperature is None:
                rejected.append(_rejection(
                    annotation,
                    "nonphysical temperature",
                    observation.raw,
                ))
                continue
            pressure = None
            if observation.pressure is not None and observation.pressure_unit:
                pressure = _pressure_bar(
                    observation.pressure,
                    observation.pressure_unit,
                )
                if pressure is None:
                    rejected.append(_rejection(
                        annotation,
                        "nonphysical pressure",
                        observation.raw,
                    ))
                    continue
            points.append(NormalizedViscosityPoint(
                viscosity_Pa_s=viscosity,
                temperature_K=temperature,
                pressure_bar=pressure,
                phase_basis=(
                    "liquid"
                    if re.search(
                        r"\b(?:liquid|molten|melt)\b",
                        annotation_context,
                        re.I,
                    )
                    else "unspecified"
                ),
                source="pubchem",
                method="pubchem_reported_dynamic_viscosity",
                record_title=annotation.record_title,
                reference_number=annotation.reference_number,
                reference=_annotation_reference(annotation),
                source_name=annotation.source_name,
                source_url=annotation.source_url,
                comment=annotation.comment,
                peer_reviewed=annotation.peer_reviewed,
                raw=observation.raw,
            ))

    unique: dict[tuple[Any, ...], NormalizedViscosityPoint] = {}
    for point in points:
        key = (
            round(point.viscosity_Pa_s, 15),
            round(point.temperature_K, 10),
            None if point.pressure_bar is None else round(point.pressure_bar, 10),
            point.reference.casefold(),
            point.raw.casefold(),
        )
        unique.setdefault(key, point)
    return ViscosityNormalizationResult(tuple(unique.values()), tuple(rejected))


class PubChemViscosityFetcher:
    """Small unintegrated client for PubChem PUG-View viscosity annotations."""

    def __init__(
        self,
        opener: Optional[Callable[..., Any]] = None,
        *,
        timeout: float = 15.0,
        user_agent: str = PUBCHEM_USER_AGENT,
    ) -> None:
        self._opener = opener or urllib.request.urlopen
        self.timeout = float(timeout)
        self.user_agent = str(user_agent)

    def _json(self, url: str) -> Mapping[str, Any]:
        request = urllib.request.Request(url)
        request.add_header("User-Agent", self.user_agent)
        try:
            with self._opener(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {}
            raise PubChemViscosityFetchError(
                f"PubChem request failed with HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PubChemViscosityFetchError(
                "Transient PubChem viscosity request failure"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise PubChemViscosityFetchError(
                "Malformed PubChem viscosity response"
            ) from exc
        return payload if isinstance(payload, Mapping) else {}

    def resolve_cid(self, identifier: str | int) -> Optional[int]:
        if isinstance(identifier, int) or str(identifier).strip().isdigit():
            cid = int(identifier)
            return cid if cid > 0 else None
        quoted = urllib.parse.quote(str(identifier).strip(), safe="")
        payload = self._json(
            f"{PUBCHEM_PUG_REST}/compound/name/{quoted}/cids/JSON"
        )
        cids = payload.get("IdentifierList", {}).get("CID", ())
        try:
            cid = int(cids[0])
        except (IndexError, KeyError, TypeError, ValueError):
            return None
        return cid if cid > 0 else None

    def fetch(self, identifier: str | int) -> Optional[PubChemViscosityResult]:
        cid = self.resolve_cid(identifier)
        if cid is None:
            return None
        query = urllib.parse.urlencode({"heading": "Viscosity"})
        payload = self._json(
            f"{PUBCHEM_PUG_VIEW}/data/compound/{cid}/JSON?{query}"
        )
        if not payload:
            return None
        annotations = extract_pubchem_viscosity_annotations(payload)
        normalized = normalize_pubchem_viscosity_annotations(annotations)
        return PubChemViscosityResult(
            identifier=str(identifier),
            cid=cid,
            annotations=annotations,
            points=normalized.points,
            rejected=normalized.rejected,
        )


__all__ = [
    "NormalizedViscosityPoint",
    "ParsedViscosityObservation",
    "PubChemViscosityAnnotation",
    "PubChemViscosityFetchError",
    "PubChemViscosityFetcher",
    "PubChemViscosityResult",
    "RejectedViscosityObservation",
    "ViscosityNormalizationResult",
    "extract_pubchem_viscosity_annotations",
    "normalize_pubchem_viscosity_annotations",
    "parse_pubchem_viscosity_annotation",
]
