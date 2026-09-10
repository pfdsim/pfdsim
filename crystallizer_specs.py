"""Shared, thermodynamics-independent crystallizer specification handling."""

from __future__ import annotations

import math


class CrystallizerSpecificationError(ValueError):
    def __init__(self, message, dof=-1):
        super().__init__(message)
        self.dof = dof


_ALIASES = {
    "tout": "t_out",
    "t": "t_out",
    "temperature": "t_out",
    "pout": "p_out",
    "p": "p_out",
    "pressure": "p_out",
    "crystallizer_model": "model",
    "tau": "residence_time",
    "v": "volume",
    "component": "crystallizing_component",
    "particle_classes": "quadrature_classes",
    "max_output_classes": "maximum_output_classes",
    "nucleation_diameter": "nucleus_diameter",
    "l0": "nucleus_diameter",
    "kinetic_tolerance": "msmpr_tolerance",
    "kinetic_relative_tolerance": "msmpr_relative_tolerance",
    "mother_liquor_retention_fraction": "mother_liquor_retention",
    "effective_diffusivity": "binary_diffusivity",
    "coolant_temperature": "t_wall",
    "t_coolant": "t_wall",
    "empirical_solid_density": "layer_solid_density",
}
_MODELS = {
    "equilibrium_sle": "equilibrium",
    "sle": "equilibrium",
    "kinetic": "msmpr",
    "steady_msmpr": "msmpr",
    "empirical_layer": "empirical_layer_growth",
    "layer_empirical": "empirical_layer_growth",
}
_MSMPR_ONLY = {
    "residence_time",
    "volume",
    "crystallizing_component",
    "quadrature_classes",
    "maximum_output_classes",
    "nucleus_diameter",
    "msmpr_tolerance",
    "msmpr_relative_tolerance",
}
_LAYER_GROWTH_ONLY = {
    "thermal_mode", "film_model", "plate_length", "liquid_velocity", "inclusion_max_fraction",
    "thermal_film_thickness",
    "growth_time", "cycle_time", "cooled_area", "film_thickness", "t_wall",
    "binary_diffusivity", "layer_relative_tolerance", "layer_profile_points",
}
_EMPIRICAL_LAYER_ONLY = {
    "growth_rate", "empirical_relative_tolerance", "layer_solid_density",
    "effective_distributions", "distribution_coefficients", "keff", "k_eff",
}
_DISTRIBUTION_PREFIXES = (
    "keff_", "k_eff_", "distribution_", "effective_distribution_",
)


def canonical_crystallizer_parameter(name):
    name = str(name).strip().lower()
    return _ALIASES.get(name, name)


def normalize_crystallizer_parameters(parameters):
    """Canonicalize aliases, preserving original unit metadata and rate keys."""
    result = {}
    for raw_name, value in parameters.items():
        name = str(raw_name).strip().lower()
        if name.startswith("__unit__"):
            name = "__unit__" + canonical_crystallizer_parameter(name[8:])
        elif not name.startswith("__"):
            name = canonical_crystallizer_parameter(name)
            for prefix in (
                "growth_param_",
                "g_param_",
                "nucleation_param_",
                "b0_param_",
            ):
                if name.startswith(prefix):
                    # Expression parameter symbols are case-sensitive, even
                    # though the unit's specification names are not.
                    name = prefix + str(raw_name).strip()[len(prefix) :]
                    break
            if any(name.startswith(prefix) for prefix in _DISTRIBUTION_PREFIXES) and "_param_" in name:
                # Preserve expression-parameter symbol case after ``param_``.
                end = name.index("_param_") + len("_param_")
                name = name[:end] + str(raw_name).strip()[end:]
        if name in result:
            raise CrystallizerSpecificationError(
                f"must specify only one {name} (duplicate aliases)"
            )
        result[name] = value
    model = str(result.get("model", "equilibrium")).strip().lower().replace("-", "_")
    result["model"] = _MODELS.get(model, model)
    result["crystallization_mode"] = str(
        result.get(
            "crystallization_mode",
            "layer" if result["model"] in {"layer_growth", "empirical_layer_growth"} else "suspension",
        )
    ).strip().lower()
    return result


def validate_crystallizer_specification(parameters, outlet_ports=None, *, require_temperature=True):
    """Share rules while allowing the parser to validate incomplete flowsheets."""
    values = normalize_crystallizer_parameters(parameters)
    model = values["model"]
    if model not in {"equilibrium", "msmpr", "layer_growth", "empirical_layer_growth"}:
        raise CrystallizerSpecificationError(
            "model must be equilibrium, MSMPR, layer_growth, or empirical_layer_growth"
        )
    mode = values["crystallization_mode"]
    if mode not in {"suspension", "layer"}:
        raise CrystallizerSpecificationError(
            "crystallization_mode must be suspension or layer"
        )
    if mode == "layer":
        if model == "msmpr":
            raise CrystallizerSpecificationError(
                "layer crystallization requires equilibrium or layer_growth; MSMPR describes suspension"
            )
        if values.get("outlet_sphericity") is not None:
            raise CrystallizerSpecificationError(
                "outlet_sphericity applies only to suspension crystallization"
            )
        if all(values.get(key) is None for key in (
            "mother_liquor_retention", "mother_liquor_retention_rate"
        )):
            values["mother_liquor_retention"] = 0.0
    if model in {"layer_growth", "empirical_layer_growth"} and mode != "layer":
        raise CrystallizerSpecificationError(
            f"model={model} requires crystallization_mode=layer"
        )
    if model == "layer_growth":
        empirical_parameters = set(values) & _EMPIRICAL_LAYER_ONLY
        empirical_parameters |= {
            key for key in values if key.startswith(_DISTRIBUTION_PREFIXES)
        }
        if empirical_parameters:
            raise CrystallizerSpecificationError(
                "empirical layer parameter(s) conflict with layer_growth: "
                + ", ".join(sorted(empirical_parameters))
            )
        for name, default, allowed in (
            ("thermal_mode", "isothermal", {"isothermal", "cooling"}),
            ("film_model", "specified", {"specified", "flat_plate"}),
        ):
            values[name] = str(values.get(name, default)).strip().lower()
            if values[name] not in allowed:
                raise CrystallizerSpecificationError(f"{name} must be " + " or ".join(sorted(allowed)))
        if values["thermal_mode"] == "cooling" and values.get("t_out") is not None:
            raise CrystallizerSpecificationError("cooling predicts outlet temperature; omit T/T_out and specify inlet temperature")
        film_keys = ("film_thickness",) if values["film_model"] == "specified" else ("plate_length", "liquid_velocity")
        incompatible = (("plate_length", "liquid_velocity") if values["film_model"] == "specified"
                        else ("film_thickness", "thermal_film_thickness"))
        if any(values.get(key) is not None for key in incompatible):
            raise CrystallizerSpecificationError("film_model conflicts with supplied film/geometry specifications")
        missing = [key for key in ("growth_time", "cycle_time", "cooled_area", "t_wall", *film_keys)
                   if values.get(key) is None]
        if missing:
            raise CrystallizerSpecificationError(
                "layer_growth requires " + ", ".join(missing), len(missing)
            )
    elif model == "empirical_layer_growth":
        incompatible = set(values) & {
            "thermal_mode", "film_model", "plate_length", "liquid_velocity",
            "inclusion_max_fraction", "thermal_film_thickness",
            "film_thickness", "binary_diffusivity", "layer_relative_tolerance",
        }
        incompatible |= set(values) & (_MSMPR_ONLY - {"crystallizing_component"})
        if incompatible:
            raise CrystallizerSpecificationError(
                "incompatible parameter(s) conflict with empirical_layer_growth: "
                + ", ".join(sorted(incompatible))
            )
        missing = [
            key for key in ("growth_time", "cycle_time", "cooled_area", "t_wall")
            if values.get(key) is None
        ]
        has_growth = any(values.get(key) is not None for key in ("growth_rate", "growth", "g")) or any(
            key.startswith("growth_") and key not in {"growth_time"}
            for key in values
        )
        if not has_growth:
            missing.append("growth kinetics")
        if missing:
            raise CrystallizerSpecificationError(
                "empirical_layer_growth requires " + ", ".join(missing), len(missing)
            )
    elif set(values) & (_LAYER_GROWTH_ONLY | _EMPIRICAL_LAYER_ONLY):
        raise CrystallizerSpecificationError(
            "finite-rate layer parameters require model=layer_growth or "
            "empirical_layer_growth"
        )
    if model != "empirical_layer_growth" and any(
        key.startswith(_DISTRIBUTION_PREFIXES) for key in values
    ):
        raise CrystallizerSpecificationError(
            "empirical distribution parameters require model=empirical_layer_growth"
        )
    if require_temperature and values.get("t_out") is None and not (
        model == "layer_growth" and values["thermal_mode"] == "cooling"
    ):
        raise CrystallizerSpecificationError("requires outlet temperature T_out/T", 1)
    kinetic_keys = [
        key
        for key in values
        if key not in _LAYER_GROWTH_ONLY and (key in _MSMPR_ONLY
        or key in {"growth", "g", "nucleation", "b0"}
        or key.startswith(("growth_", "g_", "nucleation_", "b0_")))
    ]
    if model not in {"msmpr", "empirical_layer_growth"} and kinetic_keys:
        raise CrystallizerSpecificationError(
            "MSMPR parameter(s) require model=MSMPR: " + ", ".join(sorted(kinetic_keys))
        )
    if model == "msmpr":
        dimensions = sum(
            values.get(key) is not None for key in ("residence_time", "volume")
        )
        if dimensions != 1:
            raise CrystallizerSpecificationError(
                "MSMPR mode requires exactly one of residence_time/tau or volume/V",
                1 if dimensions == 0 else -1,
            )
        missing = [
            kind
            for kind, aliases in (
                ("growth", ("growth", "g")),
                ("nucleation", ("nucleation", "b0")),
            )
            if not any(
                key == alias or key.startswith(alias + "_")
                for key in values
                for alias in aliases
            )
        ]
        if missing:
            raise CrystallizerSpecificationError(
                "MSMPR mode requires " + " and ".join(missing) + " kinetics",
                len(missing),
            )
    if model == "empirical_layer_growth":
        forbidden = [
            key for key in values
            if key in {"nucleation", "b0"}
            or key.startswith(("nucleation_", "b0_"))
        ]
        if forbidden:
            raise CrystallizerSpecificationError(
                "empirical layer growth does not use nucleation kinetics: "
                + ", ".join(sorted(forbidden))
            )
    if values.get("p_out") is not None and values.get("p_drop") is not None:
        raise CrystallizerSpecificationError(
            "cannot specify both outlet pressure and P_drop"
        )
    retention = values.get("mother_liquor_retention") is not None
    rate = values.get("mother_liquor_retention_rate") is not None
    if retention and rate:
        raise CrystallizerSpecificationError(
            "must specify only one mother-liquor retention basis"
        )
    if outlet_ports:
        ports = set(outlet_ports)
        split = {"cake", "mother_liquor"}
        if retention or rate:
            if "out" in ports:
                raise CrystallizerSpecificationError(
                    "uses cake and mother_liquor outlets when mother_liquor_retention is specified"
                )
            if not split <= ports:
                raise CrystallizerSpecificationError(
                    "cake-split mode requires cake and mother_liquor outlets", 1
                )
        elif ports & split:
            raise CrystallizerSpecificationError(
                "cake/mother_liquor outlets require mother_liquor_retention", 1
            )
    return values


def finite_crystallizer_number(
    value, name, *, minimum=0.0, inclusive=False, integer=False, maximum=None
):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise CrystallizerSpecificationError(f"{name} must be numeric") from error
    if not math.isfinite(number) or (
        number < minimum if inclusive else number <= minimum
    ):
        relation = "at least" if inclusive else "greater than"
        raise CrystallizerSpecificationError(
            f"{name} must be finite and {relation} {minimum:g}"
        )
    if maximum is not None and number > maximum:
        raise CrystallizerSpecificationError(f"{name} must be at most {maximum:g}")
    if integer and number != int(number):
        raise CrystallizerSpecificationError(f"{name} must be an integer")
    return int(number) if integer else number
