"""Liquid mixture viscosity mixing rules and parameter loaders."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Mapping, Optional


DATA_DIR = Path(__file__).resolve().parent / "data"
LIQUID_VISCOSITY_INTERACTIONS_PATH = DATA_DIR / "liquid_viscosity_interactions.json"
LIQUID_VISCOSITY_UTM_PATH = DATA_DIR / "liquid_viscosity_utm.json"
CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
JOUYBAN_ACREE_TEMPERATURE_WARNING_K = 10.0
UNIFAC_VISCO_TMIN_K = 273.15 + 10.0
UNIFAC_VISCO_TMAX_K = 273.15 + 90.0
UTM_GROUP_RQ = {
    "std:2": (0.6744, 0.54),
    "std:1": (0.9011, 0.848),
    "do:78": (0.6744, 0.54),
    "std:9": (0.5313, 0.40),
    "std_main:9": (0.7713, 0.64),
    "std:77": (1.0020, 0.88),
    "std:14": (1.000, 1.20),
    "std:15": (1.4311, 1.432),
    "std:17": (0.8952, 0.68),
    "std:42": (1.3013, 1.224),
    "visco:FCH2O": (0.9183, 1.10),
    "std:72": (3.0856, 2.736),
    "std:29": (1.3692, 1.236),
    "std:32": (1.207, 0.936),
    "std:40": (1.8701, 1.724),
    "std:54": (2.0086, 1.868),
    "std:57": (1.4199, 1.104),
    "std:37": (2.9993, 2.113),
    "std:67": (2.8266, 2.472),
    "std:118": (2.6869, 2.120),
    "visco:(CH2O)3PO": (3.7249, 3.216),
}


class LiquidMixtureViscosityError(ValueError):
    """Raised when a liquid-mixture viscosity calculation is ill-defined."""


@dataclass(frozen=True)
class JouybanAcreeParameters:
    """Binary Jouyban-Acree viscosity interaction coefficients."""

    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class LiquidMixtureViscosityResult:
    """Automatic liquid-mixture viscosity result."""

    value: float
    method: str
    warnings: tuple[str, ...]
    pair_methods: dict[tuple[str, str], str]
    excess_g_over_rt: float


def normalized_mole_fractions(composition: Mapping[str, float]) -> dict[str, float]:
    """Return positive normalized mole fractions."""
    values = {}
    for comp, value in composition.items():
        try:
            amount = float(value)
        except (TypeError, ValueError) as exc:
            raise LiquidMixtureViscosityError(
                f"Mole fraction for {comp!r} must be numeric."
            ) from exc
        if not math.isfinite(amount):
            raise LiquidMixtureViscosityError(
                f"Mole fraction for {comp!r} must be finite."
            )
        if amount < 0.0:
            raise LiquidMixtureViscosityError(
                f"Mole fraction for {comp!r} cannot be negative."
            )
        if amount > 0.0:
            values[str(comp)] = amount
    total = sum(values.values())
    if total <= 0.0:
        raise LiquidMixtureViscosityError("Composition must contain a positive total.")
    return {comp: value / total for comp, value in values.items()}


def _positive_lookup(
    values: Mapping[str, float],
    comp: str,
    label: str,
) -> float:
    try:
        value = float(values[comp])
    except KeyError as exc:
        raise LiquidMixtureViscosityError(f"Missing {label} for {comp!r}.") from exc
    except (TypeError, ValueError) as exc:
        raise LiquidMixtureViscosityError(
            f"{label} for {comp!r} must be numeric."
        ) from exc
    if value <= 0.0 or not math.isfinite(value):
        raise LiquidMixtureViscosityError(
            f"{label} for {comp!r} must be positive and finite."
        )
    return value


def _ln_pure_viscosity_sum(
    x: Mapping[str, float],
    pure_viscosities: Mapping[str, float],
) -> float:
    return sum(
        xi * math.log(_positive_lookup(pure_viscosities, comp, "pure viscosity"))
        for comp, xi in x.items()
    )


def _interaction_value(
    interactions: Optional[Mapping[object, float]],
    comp_i: str,
    comp_j: str,
) -> float:
    if not interactions:
        return 0.0
    for key in ((comp_i, comp_j), (comp_j, comp_i), frozenset((comp_i, comp_j))):
        if key not in interactions:
            continue
        try:
            value = float(interactions[key])
        except (TypeError, ValueError) as exc:
            raise LiquidMixtureViscosityError(
                f"Interaction parameter for {comp_i!r}/{comp_j!r} must be numeric."
            ) from exc
        if not math.isfinite(value):
            raise LiquidMixtureViscosityError(
                f"Interaction parameter for {comp_i!r}/{comp_j!r} must be finite."
            )
        return value
    return 0.0


def grunberg_nissan_viscosity(
    composition: Mapping[str, float],
    pure_viscosities: Mapping[str, float],
    interaction_parameters: Optional[Mapping[object, float]] = None,
) -> float:
    """Dynamic liquid viscosity [Pa*s] from Grunberg-Nissan.

    ``interaction_parameters`` are dimensionless binary ``G_ij`` values keyed by
    ``(comp_i, comp_j)`` or ``frozenset({comp_i, comp_j})``. Missing pairs use
    ``G_ij = 0``, reducing the equation to logarithmic ideal mixing.
    """
    x = normalized_mole_fractions(composition)
    ln_mu = _ln_pure_viscosity_sum(x, pure_viscosities)
    components = list(x)
    for index, comp_i in enumerate(components):
        for comp_j in components[index + 1 :]:
            ln_mu += (
                x[comp_i]
                * x[comp_j]
                * _interaction_value(interaction_parameters, comp_i, comp_j)
            )
    value = math.exp(ln_mu)
    if value <= 0.0 or not math.isfinite(value):
        raise LiquidMixtureViscosityError(
            "Grunberg-Nissan viscosity calculation produced an invalid value."
        )
    return value


def unifac_visco_viscosity(
    composition: Mapping[str, float],
    pure_viscosities: Mapping[str, float],
    excess_g_over_rt: float = 0.0,
) -> float:
    """Dynamic liquid viscosity [Pa*s] from the simplified UNIFAC-VISCO form.

    This is the updated no-volume-terms form:

    ``ln(mu) = sum_i x_i ln(mu_i) + G_visco^E / RT``.

    The caller supplies the dimensionless excess-viscosity Gibbs term. With
    ``excess_g_over_rt = 0`` the result is ideal logarithmic mixing.
    """
    x = normalized_mole_fractions(composition)
    try:
        excess = float(excess_g_over_rt)
    except (TypeError, ValueError) as exc:
        raise LiquidMixtureViscosityError(
            "UNIFAC-VISCO excess_g_over_rt must be numeric."
        ) from exc
    if not math.isfinite(excess):
        raise LiquidMixtureViscosityError(
            "UNIFAC-VISCO excess_g_over_rt must be finite."
        )
    value = math.exp(_ln_pure_viscosity_sum(x, pure_viscosities) + excess)
    if value <= 0.0 or not math.isfinite(value):
        raise LiquidMixtureViscosityError(
            "UNIFAC-VISCO viscosity calculation produced an invalid value."
        )
    return value


def _finite_parameter(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LiquidMixtureViscosityError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise LiquidMixtureViscosityError(f"{label} must be finite.")
    return number


def _jouyban_acree_contribution(
    xi: float,
    xj: float,
    T: float,
    parameters: JouybanAcreeParameters,
    label: str,
) -> float:
    coefficients = tuple(
        _finite_parameter(value, f"{label} coefficient {index}")
        for index, value in enumerate(parameters.coefficients)
    )
    if not coefficients:
        return 0.0
    composition_delta = xi - xj
    polynomial = sum(
        coefficient * composition_delta**index
        for index, coefficient in enumerate(coefficients)
    )
    return xi * xj * polynomial / T


def jouyban_acree_viscosity(
    composition: Mapping[str, float],
    pure_viscosities: Mapping[str, float],
    T: float,
    interaction_parameters: Mapping[object, JouybanAcreeParameters],
) -> float:
    """Dynamic liquid viscosity [Pa*s] from the Jouyban-Acree model.

    Pair coefficients are supplied as ``JouybanAcreeParameters`` keyed by
    ``(comp_i, comp_j)``, ``(comp_j, comp_i)``, or
    ``frozenset({comp_i, comp_j})``. Missing pairs contribute zero, so partially
    covered parameter sets degrade to the log-mixing baseline for those pairs.

    The implemented multicomponent form is:

    ``ln(mu) = sum_i x_i ln(mu_i)
             + sum_i sum_j>i x_i x_j / T * sum_k A_k,ij (x_i - x_j)^k``.
    """
    temperature = _positive_lookup({'T': T}, 'T', "temperature")
    x = normalized_mole_fractions(composition)
    ln_mu = _ln_pure_viscosity_sum(x, pure_viscosities)
    components = list(x)
    for index, comp_i in enumerate(components):
        for comp_j in components[index + 1 :]:
            params = None
            sign = 1.0
            if interaction_parameters:
                for key, delta_sign in (
                    ((comp_i, comp_j), 1.0),
                    ((comp_j, comp_i), -1.0),
                    (frozenset((comp_i, comp_j)), 1.0),
                ):
                    if key in interaction_parameters:
                        params = interaction_parameters[key]
                        sign = delta_sign
                        break
            if params is None:
                continue
            ln_mu += _jouyban_acree_contribution(
                x[comp_i],
                x[comp_j],
                temperature,
                params,
                f"{comp_i!r}/{comp_j!r}",
            ) if sign > 0.0 else _jouyban_acree_contribution(
                x[comp_j],
                x[comp_i],
                temperature,
                params,
                f"{comp_j!r}/{comp_i!r}",
            )
    value = math.exp(ln_mu)
    if value <= 0.0 or not math.isfinite(value):
        raise LiquidMixtureViscosityError(
            "Jouyban-Acree viscosity calculation produced an invalid value."
        )
    return value


def _liquid_viscosity_override_map(
    interaction_overrides: Optional[list[Mapping[str, object]]],
) -> dict[frozenset[str], list[Mapping[str, object]]]:
    records = {}
    for record in interaction_overrides or []:
        if str(record.get("model", "")).upper() != "LIQUID_VISCOSITY":
            continue
        comp1 = record.get("component1")
        comp2 = record.get("component2")
        if not comp1 or not comp2 or comp1 == comp2:
            continue
        records.setdefault(frozenset((str(comp1), str(comp2))), []).append(record)
    return records


def _liquid_viscosity_override_for_temperature(
    records: list[Mapping[str, object]],
    T: float,
) -> Optional[Mapping[str, object]]:
    fallback = None
    for record in records:
        Tmin = record.get("Tmin_K")
        Tmax = record.get("Tmax_K")
        if Tmin is None and Tmax is None:
            fallback = record
            continue
        low = -math.inf if Tmin is None else float(Tmin)
        high = math.inf if Tmax is None else float(Tmax)
        if low <= T <= high:
            return record
    return fallback


def _liquid_viscosity_override_excess(
    record: Mapping[str, object],
    xi: float,
    xj: float,
    T: float,
    label: str,
) -> float:
    form = str(record.get("viscosity_form") or "grunberg_nissan").lower()
    if form in {"grunberg_nissan", "gn", "constant", "constant_excess"}:
        return xi * xj * _finite_parameter(
            record.get("G", record.get("excess_g_over_rt", 0.0)),
            f"{label} liquid-viscosity G",
        )

    if form in {"jouyban_acree", "jouyban", "ja"}:
        params = JouybanAcreeParameters((
            _finite_parameter(record.get("A0_K", record.get("A0")), f"{label} A0_K"),
            _finite_parameter(record.get("A1_K", record.get("A1", 0.0)), f"{label} A1_K"),
            _finite_parameter(record.get("A2_K", record.get("A2", 0.0)), f"{label} A2_K"),
        ))
        return _jouyban_acree_contribution(xi, xj, T, params, label)

    if form in {"excess_poly", "poly", "polynomial"}:
        delta = xi - xj
        theta = (T - _finite_parameter(record.get("T_ref_K", 298.15), f"{label} T_ref_K")) / 100.0
        polynomial = (
            _finite_parameter(record.get("A", 0.0), f"{label} A")
            + _finite_parameter(record.get("B", 0.0), f"{label} B") * delta
            + _finite_parameter(record.get("C", 0.0), f"{label} C") * delta * delta
            + _finite_parameter(record.get("D", 0.0), f"{label} D") * theta
            + _finite_parameter(record.get("E", 0.0), f"{label} E") * theta * delta
            + _finite_parameter(record.get("F", 0.0), f"{label} F") * theta * theta
        )
        return xi * xj * polynomial

    raise LiquidMixtureViscosityError(
        f"Unsupported liquid-viscosity override form {form!r} for {label}."
    )


def _normalize_name(value: object) -> str:
    text = str(value or "").lower().strip()
    text = text.replace("n,n-", "")
    text = text.replace("n-", "")
    text = text.replace("normal ", "")
    text = text.replace("iso-", "iso")
    text = text.replace("p-", "p")
    text = text.replace("m-", "m")
    text = text.replace("o-", "o")
    text = re.sub(r"[^a-z0-9]+", "", text)
    synonyms = {
        "h2o": "water",
        "etoh": "ethanol",
        "meoh": "methanol",
        "acn": "acetonitrile",
        "ace": "acetone",
        "etac": "ethylacetate",
        "ethylacetat": "ethylacetate",
        "ethylalcohol": "ethanol",
        "methylalcohol": "methanol",
    }
    return synonyms.get(text, text)


def _canonical_cas(value: object) -> Optional[str]:
    text = str(value or "").strip()
    return text if CAS_RE.match(text) else None


@lru_cache(maxsize=1)
def _liquid_viscosity_interactions_payload() -> dict:
    return json.loads(LIQUID_VISCOSITY_INTERACTIONS_PATH.read_text())


@lru_cache(maxsize=1)
def _liquid_viscosity_utm_payload() -> dict:
    return json.loads(LIQUID_VISCOSITY_UTM_PATH.read_text())


@lru_cache(maxsize=1)
def _liquid_viscosity_aliases() -> dict[str, str]:
    payload = _liquid_viscosity_interactions_payload()
    aliases: dict[str, str] = {}
    for cas, component in payload.get("components", {}).items():
        aliases[_normalize_name(cas)] = cas
        aliases[_normalize_name(cas.replace("-", ""))] = cas
        aliases[_normalize_name(component.get("name"))] = cas
        for label in component.get("labels", []) or []:
            aliases[_normalize_name(label)] = cas
    for record in payload.get("binary_interactions", {}).values():
        for component in record.get("components", []) or []:
            cas = component.get("cas")
            if not cas:
                continue
            aliases[_normalize_name(component.get("name"))] = cas
            aliases[_normalize_name(component.get("label"))] = cas
    return {key: value for key, value in aliases.items() if key}


def _props_value(props: object, attr: str) -> Optional[object]:
    if props is None:
        return None
    if isinstance(props, Mapping):
        return props.get(attr)
    return getattr(props, attr, None)


def _component_cas(
    comp: str,
    *,
    component_cas: Optional[Mapping[str, str]] = None,
    component_props: Optional[Mapping[str, object]] = None,
) -> Optional[str]:
    if component_cas and comp in component_cas:
        direct = _canonical_cas(component_cas[comp])
        if direct:
            return direct
    direct = _canonical_cas(comp)
    if direct:
        return direct

    props = component_props.get(comp) if component_props else None
    props_cas = _canonical_cas(_props_value(props, "CAS"))
    if props_cas:
        return props_cas

    try:
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .interaction_parameters import cas_for_component
        else:
            from interaction_parameters import cas_for_component

        resolved = cas_for_component(comp, props)
        if resolved:
            return resolved
    except Exception:
        pass

    aliases = _liquid_viscosity_aliases()
    for candidate in (
        comp,
        _props_value(props, "name"),
        _props_value(props, "symbol"),
        _props_value(props, "formula"),
    ):
        key = _normalize_name(candidate)
        if key in aliases:
            return aliases[key]
    return None


def _closest_jouyban_entry(record: dict, T: float) -> dict:
    entries = record.get("entries", []) or []
    if not entries:
        raise LiquidMixtureViscosityError("Jouyban-Acree record has no entries.")

    def score(entry: dict) -> tuple[float, float]:
        distance = abs(float(entry["temperature_K"]) - T)
        ard = float(entry.get("jouyban_acree", {}).get("ARD_pct", math.inf))
        return distance, ard

    return min(entries, key=score)


def _jouyban_acree_pair_record(
    cas_i: Optional[str],
    cas_j: Optional[str],
    T: float,
) -> Optional[tuple[dict, dict, bool]]:
    if not cas_i or not cas_j or cas_i == cas_j:
        return None
    records = _liquid_viscosity_interactions_payload().get("binary_interactions", {})
    key = f"{cas_i}|{cas_j}"
    reverse_key = f"{cas_j}|{cas_i}"
    if key in records:
        record = records[key]
        return record, _closest_jouyban_entry(record, T), False
    if reverse_key in records:
        record = records[reverse_key]
        return record, _closest_jouyban_entry(record, T), True
    return None


def _jouyban_parameters_from_entry(entry: dict) -> JouybanAcreeParameters:
    params = entry.get("jouyban_acree", {})
    return JouybanAcreeParameters(
        (
            float(params["A0_K"]),
            float(params["A1_K"]),
            float(params["A2_K"]),
        )
    )


_UTM_SHARED_NUMERIC_SUBGROUPS = frozenset({
    1, 2, 9, 14, 15, 17, 18, 19, 21, 22, 23,
    29, 32, 40, 42, 43, 54, 57, 67, 72, 77,
})


def _utm_group_key_from_unifac_group(
    group: object,
    *,
    variant: Optional[str] = None,
) -> Optional[str]:
    if isinstance(group, str) and group.startswith(("std:", "do:", "std_main:", "visco:")):
        return group
    text = str(group).strip()
    upper = text.upper()
    if text.isdigit():
        subgroup = int(text)
        normalized_variant = str(variant or '').upper()
        if normalized_variant in {'UNIFDMD', 'UNIFM2', 'UNIFNIST'}:
            if subgroup == 78:
                candidate = 'do:78'
                if candidate in _liquid_viscosity_utm_payload().get('groups', {}):
                    return candidate
            if subgroup not in _UTM_SHARED_NUMERIC_SUBGROUPS:
                return None
        elif not normalized_variant and subgroup not in _UTM_SHARED_NUMERIC_SUBGROUPS:
            # Bare integer IDs above the shared core are ambiguous (for
            # example 37 and 118). Callers can pass a model variant or an
            # explicit ``std:``/``do:`` UTM key to disambiguate them.
            return None
        candidate = f"std:{subgroup}"
        if candidate in _liquid_viscosity_utm_payload().get("groups", {}):
            return candidate
    name_map = {
        "CH3": "std:1",
        "CH2": "std:2",
        "CY-CH2": "do:78",
        "ACH": "std:9",
        "CH3CO": "std_main:9",
        "CH2CO": "std_main:9",
        "CO": "std_main:9",
        "CH3COO": "std:77",
        "CH2COO": "std:77",
        "HCOO": "std:77",
        "COO": "std:77",
        "OH": "std:14",
        "CH3OH": "std:15",
        "ACOH": "std:17",
        "COOH": "std:42",
        "HCOOH": "std:42",
        "DMF": "std:72",
        "CH2NH2": "std:29",
        "CH2NH": "std:32",
        "CH3CN": "std:40",
        "CH3NO2": "std:54",
        "ACNO2": "std:57",
        "C5H5N": "std:37",
        "DMSO": "std:67",
        "CH2SO2CH2": "std:118",
        "(CH2)2SU": "std:118",
        "FCH2O": "visco:FCH2O",
        "(CH2O)3PO": "visco:(CH2O)3PO",
    }
    return name_map.get(upper)


def _utm_group_contributions_from_unifac_group(
    group: object,
    *,
    variant: Optional[str] = None,
) -> dict[str, float]:
    text = str(group).strip()
    upper = text.upper()
    split_map = {
        "CH3COO": {"std:1": 1.0, "std:77": 1.0},
        "21": {"std:1": 1.0, "std:77": 1.0},
        "STD:21": {"std:1": 1.0, "std:77": 1.0},
        "CH2COO": {"std:2": 1.0, "std:77": 1.0},
        "22": {"std:2": 1.0, "std:77": 1.0},
        "STD:22": {"std:2": 1.0, "std:77": 1.0},
        "HCOO": {"std:77": 1.0},
        "HCOOH": {"std:42": 1.0},
        "43": {"std:42": 1.0},
        "STD:43": {"std:42": 1.0},
        "CH3CO": {"std:1": 1.0, "std_main:9": 1.0},
        "18": {"std:1": 1.0, "std_main:9": 1.0},
        "STD:18": {"std:1": 1.0, "std_main:9": 1.0},
        "CH2CO": {"std:2": 1.0, "std_main:9": 1.0},
        "19": {"std:2": 1.0, "std_main:9": 1.0},
        "STD:19": {"std:2": 1.0, "std_main:9": 1.0},
    }
    if upper in split_map:
        return dict(split_map[upper])
    key = _utm_group_key_from_unifac_group(group, variant=variant)
    if key is None:
        return {}
    return {key: 1.0}


def _utm_groups_from_unifac_groups(
    groups: Mapping[object, float],
    *,
    variant: Optional[str] = None,
) -> dict[str, float]:
    result: dict[str, float] = {}
    missing = []
    for group, count in groups.items():
        count_value = float(count)
        if count_value == 0.0:
            continue
        contributions = _utm_group_contributions_from_unifac_group(
            group,
            variant=variant,
        )
        if not contributions:
            missing.append(str(group))
            continue
        for key, multiplier in contributions.items():
            result[key] = result.get(key, 0.0) + count_value * multiplier
    if missing:
        raise LiquidMixtureViscosityError(
            "UNIFAC-VISCO group mapping unavailable for " + ", ".join(sorted(missing))
        )
    if not result:
        raise LiquidMixtureViscosityError("UNIFAC-VISCO component has no usable groups.")
    return result


def _component_unifac_visco_groups(
    comp: str,
    *,
    component_groups: Optional[Mapping[str, Mapping[object, float]]] = None,
    component_group_variant: Optional[str] = None,
    component_smiles: Optional[Mapping[str, str]] = None,
    component_props: Optional[Mapping[str, object]] = None,
) -> dict[str, float]:
    """Resolve temperature-independent groups once per structural definition.

    Keys contain values rather than object identities so edited properties,
    explicit group overrides, and UNIFAC variants cannot reuse stale groups.
    Failed structural assignments are cached too: the pair fallback should
    not rebuild the chemical database at every mixture temperature.
    """
    props = component_props.get(comp) if component_props else None
    properties = tuple(
        (key, str(value) if value is not None else None)
        for key in ('name', 'symbol', 'formula', 'smiles', 'MW', 'CAS', 'cas')
        for value in (_props_value(props, key),)
    )
    groups = tuple(sorted(
        (component_groups.get(comp) or {}).items(), key=lambda item: str(item[0])
    )) if component_groups else ()
    entries, error = _cached_component_unifac_visco_groups(
        comp, groups, component_group_variant,
        component_smiles.get(comp) if component_smiles else None, properties,
    )
    if error is not None:
        raise LiquidMixtureViscosityError(error)
    return dict(entries)


@lru_cache(maxsize=1024)
def _cached_component_unifac_visco_groups(comp, groups, variant, smiles, properties):
    try:
        result = _resolve_component_unifac_visco_groups(
            comp, component_groups={comp: dict(groups)} if groups else None,
            component_group_variant=variant,
            component_smiles={comp: smiles} if smiles else None,
            component_props={comp: dict(properties)},
        )
    except LiquidMixtureViscosityError as error:
        return (), str(error)
    return tuple(result.items()), None


def _resolve_component_unifac_visco_groups(
    comp: str,
    *,
    component_groups: Optional[Mapping[str, Mapping[object, float]]] = None,
    component_group_variant: Optional[str] = None,
    component_smiles: Optional[Mapping[str, str]] = None,
    component_props: Optional[Mapping[str, object]] = None,
) -> dict[str, float]:
    last_error = None
    if component_groups and comp in component_groups:
        try:
            return _utm_groups_from_unifac_groups(
                component_groups[comp],
                variant=component_group_variant,
            )
        except Exception as exc:
            # The active thermodynamic family may contain subgroup IDs whose
            # meaning is incompatible with the standard/Dortmund UTM tables.
            # Resolve the structure again below using the first compatible
            # UNIFAC family instead of reinterpreting a colliding integer ID.
            last_error = exc

    props = component_props.get(comp) if component_props else None
    identifiers = [
        comp,
        _props_value(props, "name"),
        _props_value(props, "symbol"),
        _props_value(props, "formula"),
    ]
    smiles = component_smiles.get(comp) if component_smiles else None
    if not smiles:
        smiles = _props_value(props, "smiles")
    if not smiles:
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .chemical_properties import ChemicalDatabase
            else:
                from chemical_properties import ChemicalDatabase
            db = ChemicalDatabase(enable_online=True)
            for identifier in identifiers:
                if not identifier:
                    continue
                result = db.resolve_smiles_info(
                    str(identifier),
                    fetch_online=True,
                    props=props,
                )
                smiles = result.smiles if result else None
                if smiles:
                    break
        except Exception:
            smiles = None

    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .unifac import get_unifac_groups
    else:
        from unifac import get_unifac_groups

    # Prefer modified variants because they preserve cyclic CH2 groups; standard
    # UNIFAC collapses cycloalkanes to ordinary CH2, which selects wrong UTM
    # interactions.
    try:
        expected_mw = float(_props_value(props, "MW"))
    except (TypeError, ValueError):
        expected_mw = None
    for variant in ("UNIFDMD", "UNIFNIST", "UNIFAC"):
        for identifier in identifiers:
            if not identifier:
                continue
            try:
                groups = get_unifac_groups(
                    str(identifier),
                    variant=variant,
                    expected_mw=expected_mw,
                    smiles=str(smiles) if smiles else None,
                )
                return _utm_groups_from_unifac_groups(groups, variant=variant)
            except Exception as exc:
                last_error = exc
        if smiles:
            try:
                label = str(_props_value(props, "name") or comp)
                groups = get_unifac_groups(label, smiles=str(smiles), variant=variant)
                return _utm_groups_from_unifac_groups(groups, variant=variant)
            except Exception as exc:
                last_error = exc
    raise LiquidMixtureViscosityError(
        f"Cannot determine UNIFAC-VISCO groups for {comp!r}: {last_error}"
    )


def _utm_beta(source: str, target: str) -> float:
    if source == target:
        return 0.0
    matrix = _liquid_viscosity_utm_payload().get("beta_nm_K", {})
    try:
        return float(matrix[source][target])
    except KeyError as exc:
        raise LiquidMixtureViscosityError(
            f"UNIFAC-VISCO beta missing for {source}->{target}."
        ) from exc


def _group_fractions(component_groups: list[dict[str, float]], x: list[float]) -> dict[str, float]:
    totals: dict[str, float] = {}
    denominator = 0.0
    for xi, groups in zip(x, component_groups):
        group_total = sum(groups.values())
        denominator += xi * group_total
        for group, count in groups.items():
            totals[group] = totals.get(group, 0.0) + xi * count
    if denominator <= 0.0:
        raise LiquidMixtureViscosityError("UNIFAC-VISCO group denominator is zero.")
    return {group: value / denominator for group, value in totals.items() if value > 0.0}


def _area_group_fractions(component_groups: list[dict[str, float]], x: list[float]) -> dict[str, float]:
    group_fractions = _group_fractions(component_groups, x)
    denominator = sum(
        UTM_GROUP_RQ[group][1] * fraction
        for group, fraction in group_fractions.items()
    )
    if denominator <= 0.0:
        raise LiquidMixtureViscosityError("UNIFAC-VISCO area-fraction denominator is zero.")
    return {
        group: UTM_GROUP_RQ[group][1] * fraction / denominator
        for group, fraction in group_fractions.items()
    }


def _utm_component_rq(groups: Mapping[str, float]) -> tuple[float, float]:
    r_value = 0.0
    q_value = 0.0
    missing = []
    for group, count in groups.items():
        if group not in UTM_GROUP_RQ:
            missing.append(group)
            continue
        r_group, q_group = UTM_GROUP_RQ[group]
        r_value += float(count) * r_group
        q_value += float(count) * q_group
    if missing:
        raise LiquidMixtureViscosityError(
            "UNIFAC-VISCO R/Q unavailable for " + ", ".join(sorted(missing))
        )
    if r_value <= 0.0 or q_value <= 0.0:
        raise LiquidMixtureViscosityError(
            "UNIFAC-VISCO component R/Q calculation produced nonpositive values."
        )
    return r_value, q_value


def _unifac_visco_combinatorial_excess_g_over_rt(
    component_groups: list[dict[str, float]],
    x: list[float],
) -> float:
    r = []
    q = []
    for groups in component_groups:
        r_i, q_i = _utm_component_rq(groups)
        r.append(r_i)
        q.append(q_i)

    sum_xr = sum(xi * ri for xi, ri in zip(x, r)) or 1e-30
    sum_xq = sum(xi * qi for xi, qi in zip(x, q)) or 1e-30
    z = 10.0
    l_values = [
        0.5 * z * (r_i - q_i) - (r_i - 1.0)
        for r_i, q_i in zip(r, q)
    ]
    sum_xl = sum(xi * li for xi, li in zip(x, l_values))

    excess = 0.0
    for xi, r_i, q_i, l_i in zip(x, r, q, l_values):
        if xi <= 0.0:
            continue
        phi_over_x = max(r_i / sum_xr, 1e-30)
        theta_over_phi = max(q_i * sum_xr / (r_i * sum_xq), 1e-30)
        ln_gamma_c = (
            math.log(phi_over_x)
            + 0.5 * z * q_i * math.log(theta_over_phi)
            + l_i
            - phi_over_x * sum_xl
        )
        excess += xi * ln_gamma_c
    return excess


def _utm_ln_group_activity(group_fractions: Mapping[str, float], T: float) -> dict[str, float]:
    groups = list(group_fractions)
    denominator_by_target = {}
    for target in groups:
        denominator_by_target[target] = sum(
            group_fractions[source] * math.exp(-_utm_beta(source, target) / T)
            for source in groups
        )
        if denominator_by_target[target] <= 0.0:
            raise LiquidMixtureViscosityError(
                f"UNIFAC-VISCO denominator is zero for group {target}."
            )

    result = {}
    for group in groups:
        second = sum(
            group_fractions[target]
            * math.exp(-_utm_beta(group, target) / T)
            / denominator_by_target[target]
            for target in groups
        )
        result[group] = UTM_GROUP_RQ[group][1] * (
            1.0 - math.log(denominator_by_target[group]) - second
        )
    return result


def _unifac_visco_excess_g_over_rt(
    component_groups: list[dict[str, float]],
    x: list[float],
    T: float,
) -> float:
    combinatorial = _unifac_visco_combinatorial_excess_g_over_rt(component_groups, x)
    mix_area_fractions = _area_group_fractions(component_groups, x)
    mix_ln_gamma = _utm_ln_group_activity(mix_area_fractions, T)
    residual = 0.0
    for xi, groups in zip(x, component_groups):
        pure_area_fractions = _area_group_fractions([groups], [1.0])
        pure_ln_gamma = _utm_ln_group_activity(pure_area_fractions, T)
        component_residual = sum(
            count * (mix_ln_gamma[group] - pure_ln_gamma[group])
            for group, count in groups.items()
        )
        residual += xi * component_residual
    return combinatorial - residual


def _unifac_visco_pair_excess_for_actual_composition(
    xi: float,
    xj: float,
    groups_i: Mapping[str, float],
    groups_j: Mapping[str, float],
    T: float,
) -> float:
    pair_total = xi + xj
    if pair_total <= 0.0:
        return 0.0
    zi = xi / pair_total
    zj = xj / pair_total
    binary_excess = _unifac_visco_excess_g_over_rt(
        [dict(groups_i), dict(groups_j)],
        [zi, zj],
        T,
    )
    return pair_total * pair_total * binary_excess


def estimate_liquid_mixture_viscosity(
    composition: Mapping[str, float],
    pure_viscosities: Mapping[str, float],
    T: float,
    *,
    component_cas: Optional[Mapping[str, str]] = None,
    component_smiles: Optional[Mapping[str, str]] = None,
    component_groups: Optional[Mapping[str, Mapping[object, float]]] = None,
    component_group_variant: Optional[str] = None,
    component_props: Optional[Mapping[str, object]] = None,
    interaction_overrides: Optional[list[Mapping[str, object]]] = None,
) -> LiquidMixtureViscosityResult:
    """Estimate liquid mixture viscosity with the configured model hierarchy.

    The selector uses Jouyban-Acree for each binary pair when a CAS-keyed
    interaction is available. Missing pairs fall back to UNIFAC-VISCO using the
    UTM group-parameter file. If both sources are unavailable, that pair
    contributes zero excess viscosity and a warning is returned.
    """
    temperature = _positive_lookup({"T": T}, "T", "temperature")
    x = normalized_mole_fractions(composition)
    warnings: list[str] = []
    pair_methods: dict[tuple[str, str], str] = {}
    component_cas_map = {
        comp: _component_cas(
            comp,
            component_cas=component_cas,
            component_props=component_props,
        )
        for comp in x
    }
    component_utm_groups: dict[str, dict[str, float]] = {}
    override_map = _liquid_viscosity_override_map(interaction_overrides)
    total_excess = 0.0
    components = list(x)

    for index, comp_i in enumerate(components):
        for comp_j in components[index + 1 :]:
            pair_key = (comp_i, comp_j)
            xi, xj = x[comp_i], x[comp_j]
            override_records = override_map.get(frozenset((comp_i, comp_j)))
            override = (
                _liquid_viscosity_override_for_temperature(
                    override_records,
                    temperature,
                )
                if override_records else None
            )
            if override is not None:
                reversed_record = override.get("component1") != comp_i
                total_excess += _liquid_viscosity_override_excess(
                    override,
                    xj if reversed_record else xi,
                    xi if reversed_record else xj,
                    temperature,
                    f"{comp_j!r}/{comp_i!r}" if reversed_record else f"{comp_i!r}/{comp_j!r}",
                )
                pair_methods[pair_key] = f"override_{override['viscosity_form']}"
                continue
            ja_record = _jouyban_acree_pair_record(
                component_cas_map.get(comp_i),
                component_cas_map.get(comp_j),
                temperature,
            )
            if ja_record is not None:
                record, entry, reversed_record = ja_record
                params = _jouyban_parameters_from_entry(entry)
                if reversed_record:
                    total_excess += _jouyban_acree_contribution(
                        xj, xi, temperature, params, f"{comp_j!r}/{comp_i!r}"
                    )
                else:
                    total_excess += _jouyban_acree_contribution(
                        xi, xj, temperature, params, f"{comp_i!r}/{comp_j!r}"
                    )
                entry_temperature = float(entry["temperature_K"])
                if abs(entry_temperature - temperature) > JOUYBAN_ACREE_TEMPERATURE_WARNING_K:
                    warnings.append(
                        f"Jouyban-Acree pair {comp_i}/{comp_j} used "
                        f"{entry_temperature:.2f} K entry for {temperature:.2f} K."
                    )
                pair_methods[pair_key] = "jouyban_acree"
                continue

            try:
                if comp_i not in component_utm_groups:
                    component_utm_groups[comp_i] = _component_unifac_visco_groups(
                        comp_i,
                        component_groups=component_groups,
                        component_group_variant=component_group_variant,
                        component_smiles=component_smiles,
                        component_props=component_props,
                    )
                if comp_j not in component_utm_groups:
                    component_utm_groups[comp_j] = _component_unifac_visco_groups(
                        comp_j,
                        component_groups=component_groups,
                        component_group_variant=component_group_variant,
                        component_smiles=component_smiles,
                        component_props=component_props,
                    )
                total_excess += _unifac_visco_pair_excess_for_actual_composition(
                    xi,
                    xj,
                    component_utm_groups[comp_i],
                    component_utm_groups[comp_j],
                    temperature,
                )
                pair_methods[pair_key] = "unifac_visco"
                if not (UNIFAC_VISCO_TMIN_K <= temperature <= UNIFAC_VISCO_TMAX_K):
                    warnings.append(
                        f"UNIFAC-VISCO pair {comp_i}/{comp_j} used at "
                        f"{temperature - 273.15:.2f} C, outside 10-90 C range."
                    )
            except Exception as exc:
                total_excess += 0.0
                pair_methods[pair_key] = "zero"
                warnings.append(
                    f"No liquid-viscosity interaction for {comp_i}/{comp_j}; "
                    f"using zero excess contribution ({exc})."
                )

    value = unifac_visco_viscosity(x, pure_viscosities, total_excess)
    methods = set(pair_methods.values())
    if not pair_methods:
        method = "pure_component"
    elif methods == {"jouyban_acree"}:
        method = "jouyban_acree"
    elif methods == {"unifac_visco"}:
        method = "unifac_visco"
    elif "zero" in methods:
        method = "jouyban_acree_unifac_visco_zero_fallback"
    elif methods <= {"jouyban_acree", "unifac_visco"}:
        method = "jouyban_acree_unifac_visco"
    else:
        method = "liquid_mixture_viscosity"
    return LiquidMixtureViscosityResult(
        value=value,
        method=method,
        warnings=tuple(warnings),
        pair_methods=pair_methods,
        excess_g_over_rt=total_excess,
    )
