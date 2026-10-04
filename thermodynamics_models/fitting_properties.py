"""Fitting property policy: estimated auxiliaries, qualified regional Psat."""

import math

if __package__.split(".", 1)[0] == "pfdsim":
    from ..property_resolver import get_property_resolver
else:
    from property_resolver import get_property_resolver

COMPONENT_FIELDS = {
    "MW": "molecular_weight",
    "Tc_K": "Tc",
    "Pc_bar": "Pc",
    "Tb_K": "Tb",
    "omega": "omega",
    "Vc_cm3_mol": "Vc",
    "Zc": "Zc",
    "dipole_D": "dipole_moment",
    "hoc_eta": "hoc_eta",
    "Rprime_A": "modified_radius_of_gyration",
    "smiles": "smiles",
}
VAPOR_REQUIREMENTS = {
    "IDEAL": {
        "fields": [],
        "description": "Qualified Psat; no vapor correction constants.",
    },
    "RK": {
        "fields": ["Tc_K", "Pc_bar"],
        "description": "Tc and Pc for the RK vapor EOS.",
    },
    "PR": {
        "fields": ["Tc_K", "Pc_bar", "omega"],
        "description": "Tc, Pc and acentric factor for PR.",
    },
    "VDM": {
        "fields": [],
        "description": "Psat and associator parameters from PFDSim or component VDM definitions; generic association estimates are warned.",
    },
    "TSONOPOULOS": {
        "fields": ["Tc_K", "Pc_bar", "Vc_cm3_mol", "omega", "dipole_D"],
        "description": "Tc/Pc/Vc/omega; dipole is needed only for the applicable polar classes.",
    },
    "PITZER-CURL": {
        "fields": ["Tc_K", "Pc_bar", "Vc_cm3_mol", "omega"],
        "description": "Tc/Pc/Vc/omega; no dipole or association eta.",
    },
    "ABBOTT": {
        "fields": ["Tc_K", "Pc_bar", "Vc_cm3_mol", "omega"],
        "description": "Tc/Pc/Vc/omega; no dipole or association eta.",
    },
    "HOC": {
        "fields": ["Tc_K", "Pc_bar", "dipole_D", "Rprime_A", "hoc_eta"],
        "description": "Tc/Pc, dipole, modified radius and pure eta. Enter eta to override the group estimate or warned zero default; cross eta is under vapor parameters.",
    },
}
_ESTIMATED = ("estim", "nannoolal", "joback", "functional", "group", "xtb", "pvdz")
_PSAT_ESTIMATED = (
    "estim",
    "ambrose",
    "nannoolal",
    "clapeyron",
    "hermite",
    "completion",
    "fallback",
)


def _estimated(result):
    return any(
        word
        in (str(result.get("method", "")) + " " + str(result.get("source", ""))).lower()
        for word in _ESTIMATED
    )


def _record(component, field, result):
    return {
        "component": component,
        "property": field,
        "value": float(result.value),
        "quality": result.quality,
        "source": result.source,
        "method": result.method,
        "notes": result.notes,
    }


def prepare_auxiliary_properties(thermo, definition, request):
    """Resolve only needed auxiliary inputs through the shared property resolver."""
    resolver = get_property_resolver()
    records, warnings = [], []
    vapor = request["vapor"]
    for component in thermo.components:
        props, known = thermo.props[component], thermo._resolver_known_props[component]
        identifier = props.CAS or props.name
        required = {"Tc", "Pc"} | {
            COMPONENT_FIELDS[field]
            for field in VAPOR_REQUIREMENTS[vapor]["fields"]
            if field in ("Tc_K", "Pc_bar", "Vc_cm3_mol", "omega")
        }
        missing = [field for field in required if getattr(props, field, None) is None]
        resolved = (
            resolver.resolve_critical_properties(
                identifier,
                known,
                allow_online=request["online_lookup"],
                allow_estimation=request["estimate_properties"],
            )
            if missing
            else {}
        )
        for field in sorted(required):
            value = getattr(props, field, None)
            if value is None:
                item = resolved.get(field)
                if item is None:
                    raise ValueError(
                        f"Cannot resolve {field} for {props.name}; enter the property or provide an identifiable structure. Psat is not estimated."
                    )
                value = float(item.value)
                record = _record(component, field, item)
                setattr(props, field, value)
                known[field] = value
                known.setdefault("property_sources", {})[field] = {
                    key: record[key] for key in ("source", "method", "quality", "notes")
                }
                props.property_sources[field] = dict(known["property_sources"][field])
                props.property_sources[field] = dict(known["property_sources"][field])
                setattr(definition.get_component(component), field, value)
            else:
                provenance = (known.get("property_sources") or {}).get(field) or {}
                record = {
                    "component": component,
                    "property": field,
                    "value": float(value),
                    "quality": provenance.get("quality", 1),
                    "source": provenance.get("source", "PFDSim"),
                    "method": provenance.get("method", "known"),
                    "notes": provenance.get("notes", ""),
                }
            if (
                not math.isfinite(float(value))
                or field != "omega"
                and float(value) <= 0
            ):
                raise ValueError(f"Invalid {field} for {props.name}.")
            if _estimated(record):
                if not request["estimate_properties"]:
                    raise ValueError(
                        f"{field} for {props.name} is estimated; enter it or enable supporting-property estimates."
                    )
                warnings.append(
                    f"{props.name}: {field} uses {record['method']} ({record['source']}, quality={record['quality']})."
                )
            records.append(record)
        needs_dipole = vapor == "HOC"
        if vapor == "TSONOPOULOS":
            from .second_virial import (
                _tsonopoulos_species_type,
                _TSONOPOULOS_DIPOLE_SPECIES,
            )

            needs_dipole = (
                _tsonopoulos_species_type(component, props)
                in _TSONOPOULOS_DIPOLE_SPECIES
            )
        if needs_dipole:
            item = resolver.resolve_dipole_moment(
                identifier,
                known,
                allow_online=request["online_lookup"],
                allow_estimation=request["estimate_properties"],
            )
            record = _record(component, "dipole_moment", item)
            if _estimated(record) and not request["estimate_properties"]:
                raise ValueError(
                    f"Dipole for {props.name} is estimated; enter a value or enable supporting-property estimates."
                )
            props.dipole_moment = float(item.value)
            known["dipole_moment"] = float(item.value)
            props.property_sources["dipole_moment"] = {
                key: record[key] for key in ("source", "method", "quality", "notes")
            }
            props.property_sources["dipole_moment"] = {
                key: record[key] for key in ("source", "method", "quality", "notes")
            }
            definition.get_component(component).dipole_moment = float(item.value)
            records.append(record)
            if _estimated(record) or item.quality is None or item.quality < 0.9:
                warnings.append(
                    f"{props.name}: dipole {item.value:g} D uses {item.method}, quality={item.quality}. Enter a dipole to replace this estimate/low-quality value."
                )
        if vapor == "HOC":
            if (
                not request["estimate_properties"]
                and getattr(props, "modified_radius_of_gyration", None) is None
            ):
                raise ValueError(
                    f"HOC requires Rprime_A for {props.name}; enter it or enable supporting-property estimation."
                )
            item = resolver.resolve_modified_radius_of_gyration(
                identifier,
                known,
                allow_online=request["online_lookup"],
                allow_estimation=request["estimate_properties"],
            )
            record = _record(component, "modified_radius_of_gyration", item)
            if _estimated(record) and not request["estimate_properties"]:
                raise ValueError(
                    f"HOC requires a modified radius for {props.name}; enter Rprime_A or enable estimation."
                )
            props.modified_radius_of_gyration = float(item.value)
            known["modified_radius_of_gyration"] = float(item.value)
            props.property_sources["modified_radius_of_gyration"] = {
                key: record[key] for key in ("source", "method", "quality", "notes")
            }
            props.property_sources["modified_radius_of_gyration"] = {
                key: record[key] for key in ("source", "method", "quality", "notes")
            }
            definition.get_component(component).modified_radius_of_gyration = float(
                item.value
            )
            records.append(record)
            if _estimated(record) or item.quality is None or item.quality < 0.9:
                warnings.append(
                    f"{props.name}: HOC modified radius uses {item.method}, quality={item.quality}."
                )
            if props.hoc_eta is None:
                from .hayden_oconnell import (
                    hoc_association_group,
                    HOC_GROUP_ASSOCIATION_PARAMETERS,
                )

                classification = hoc_association_group(
                    component,
                    props,
                    chemical_database=thermo.db,
                    allow_online=request["online_lookup"],
                    resolver_properties=known,
                    include_features=True,
                )
                group = classification["group"]
                if props.smiles:
                    known["smiles"] = props.smiles
                eta = HOC_GROUP_ASSOCIATION_PARAMETERS.get(group, 0.0)
                if classification["has_carboxylic_acid"] and eta == 0:
                    raise ValueError(
                        f"HOC η cannot default to zero for acid {props.name}; its multifunctional group is not covered by the recorded organic-acid value. Enter hoc_eta."
                    )
                if eta == 0 and not request["allow_hoc_eta_default"]:
                    raise ValueError(
                        f"HOC eta is unavailable for {props.name}; enter hoc_eta or allow the zero default."
                    )
                props.hoc_eta = eta
                known["hoc_eta"] = eta
                definition.get_component(component).hoc_eta = eta
                records.append(
                    {
                        "component": component,
                        "property": "hoc_eta",
                        "value": eta,
                        "source": "HOC group correlation" if eta else "zero default",
                        "method": "tabulated group default" if eta else "default",
                        "quality": None,
                        "notes": group,
                    }
                )
                warnings.append(
                    f"{props.name}: HOC eta={eta:g} is {'the recorded ' + group + ' group default' if eta else 'the zero default'}. Enter hoc_eta to override it."
                )
            else:
                source = (
                    (known.get("property_sources") or {}).get("hoc_eta")
                    or props.property_sources.get("hoc_eta")
                    or {}
                )
                records.append(
                    {
                        "component": component,
                        "property": "hoc_eta",
                        "value": float(props.hoc_eta),
                        "source": source.get("source", "provided"),
                        "method": source.get("method", "provided"),
                        "quality": source.get("quality", 1),
                        "notes": source.get("notes", ""),
                    }
                )
    return records, warnings


def qualify_psat(thermo, temperatures, request):
    """Check endpoints and every source-region boundary/midpoint in the target span."""
    resolver = get_property_resolver()
    low, high = min(temperatures), max(temperatures)
    checks = []
    for component in thermo.components:
        props, known = thermo.props[component], thermo._resolver_known_props[component]
        identifier = props.CAS or props.name
        samples = resolver.vapor_pressure_quality_samples(
            identifier, low, high, known, allow_online=request["online_lookup"]
        )
        for temperature, item in samples:
            tag = (str(item.method) + " " + str(item.source)).lower()
            if (
                item.quality is None
                or not math.isfinite(float(item.quality))
                or item.quality < 0.9
                or any(word in tag for word in _PSAT_ESTIMATED)
            ):
                raise ValueError(
                    f"Psat for {props.name} at {temperature:g} K has quality {item.quality!r} ({item.method}; {item.source}). Vapor-equilibrium fitting requires non-estimated Psat quality ≥0.9 throughout the target region; provide your own Psat correlation."
                )
            checks.append({**_record(component, "Psat", item), "T_K": temperature})
    return checks
