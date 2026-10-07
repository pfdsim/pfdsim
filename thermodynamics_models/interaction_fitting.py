"""Experimental binary NRTL/UNIQUAC regression using PFDSim thermodynamics.

All temperatures, pressures and enthalpies in normalized input are K, bar and
J/mol (numerically identical to the runtime's kJ/kmol). Weights multiply sums
of squared, uncertainty-scaled residuals. Pins are bounded constraints, never
large artificial weights. This module also owns input and PFD export contracts.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math

import numpy as np
from scipy.optimize import brentq, least_squares, minimize

from .factory import create_thermodynamics
from .common import ThermodynamicsError
from .fitting_data import (
    FIELD_LABELS,
    KINDS,
    MISSING_TOKENS,
    interpret_paste,
    is_missing_cell,
    tabular_matrix,
)
from .fitting_psat import PSAT_FORMS, apply_psat, normalize_psat, install_fitting_psat
from .fitting_properties import (
    COMPONENT_FIELDS,
    VAPOR_REQUIREMENTS,
    prepare_auxiliary_properties,
    qualify_psat,
)
from .fitting_diagnostics import physical_metrics, build_objective_plots
from .fitting_optimizer import solve_start

if __package__.split(".", 1)[0] == "pfdsim":
    from ..pfd_parser import (
        Component,
        InteractionParameter,
        ProcessFlowDiagram,
        ParseError,
        parse_pfd,
    )
    from ..simulator import Simulator, SimulationError
    from ..cubic_eos import CubicEOSError
    from ..rk_eos import RKError
else:
    from pfd_parser import (
        Component,
        InteractionParameter,
        ProcessFlowDiagram,
        ParseError,
        parse_pfd,
    )
    from simulator import Simulator, SimulationError
    from cubic_eos import CubicEOSError
    from rk_eos import RKError

_MODEL_ERRORS = (
    ThermodynamicsError,
    SimulationError,
    ParseError,
    CubicEOSError,
    RKError,
)


VAPOR_OBJECTIVES = frozenset({"VLE", "AZEOTROPE", "VLLE"})
FORMS = {
    "constant": ("constant",),
    "inverse": ("inverse",),
    "constant_inverse": ("constant", "inverse"),
    "constant_inverse_anchored": ("constant", "inverse", "anchored"),
    "constant_inverse_linear": ("constant", "inverse", "linear"),
    "constant_inverse_anchored_linear": ("constant", "inverse", "anchored", "linear"),
    "full": ("constant", "inverse", "anchored", "linear", "quadratic"),
}
VAPORS = ("IDEAL", "RK", "PR", "VDM", "TSONOPOULOS", "PITZER-CURL", "ABBOTT", "HOC")
SCALES = {
    "log_fugacity": 0.01,
    "log_gamma": 0.01,
    "HE_J_mol": 100.0,
    "curvature": 0.1,
    "third_derivative": 0.1,
}
_ALIASES = {
    "HETEROAZEOTROPE": "VLLE",
    "HETERO_AZEOTROPE": "VLLE",
    "H^E": "HE",
    "H_E": "HE",
    "EXCESS_ENTHALPY": "HE",
    "GAMMA_INFINITY": "GAMMA_INF",
    "Γ_INF": "GAMMA_INF",
}


def _number(value, name, *, low=-math.inf, high=math.inf):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be finite and between {low:g} and {high:g}.")
    return result


def _boolean(value, name):
    if value is True or isinstance(value, str) and value in ("true", "True"):
        return True
    if value is False or isinstance(value, str) and value in ("false", "False", ""):
        return False
    raise ValueError(f"{name} must be true or false.")


def inspect_observations(value, *, import_options=None, components=None):
    """Propose a table interpretation, leaving missing choices for the caller."""
    proposal = interpret_paste(value, options=import_options, components=components)
    raw = proposal.pop("raw_observations", None)
    proposal["observations"] = None
    if proposal["ready"]:
        try:
            proposal["observations"] = parse_observations(raw)
        except ValueError as error:
            if proposal["structured"]:
                raise
            proposal["ready"] = False
            proposal["needs_review"] = True
            proposal["issues"].append(str(error))
    return proposal


def parse_observations(value, *, import_options=None, components=None):
    """Normalize explicit observations or a resolved human table proposal."""
    if isinstance(value, str) or tabular_matrix(value) is not None:
        proposal = inspect_observations(
            value, import_options=import_options, components=components
        )
        if not proposal["ready"]:
            raise ValueError(
                "Table interpretation needs input: "
                + " ".join(proposal["issues"])
                + " Use the import preview or supply import_options; the table does not need reformatting."
            )
        return proposal["observations"]
    if isinstance(value, dict):
        if "datasets" in value:
            datasets = value["datasets"]
            if not isinstance(datasets, list):
                raise ValueError("datasets must be an array.")
            rows = []
            for dataset in datasets:
                if not isinstance(dataset, dict) or not isinstance(
                    dataset.get("rows"), list
                ):
                    raise ValueError("Each dataset needs a rows array.")
                defaults = {
                    key: dataset[key]
                    for key in ("kind", "weight", "source", "group", "sigma")
                    if key in dataset
                }
                for row in dataset["rows"]:
                    if not isinstance(row, dict):
                        raise ValueError("Each observation must be an object.")
                    rows.append({**defaults, **row})
            value = rows
        else:
            value = value.get("observations", value.get("rows"))
    if not isinstance(value, list) or not 1 <= len(value) <= 2000:
        raise ValueError("Provide an array of 1–2000 observations.")
    allowed = {
        "kind",
        "type",
        "T_K",
        "T_C",
        "P_bar",
        "P_kPa",
        "P_atm",
        "x1",
        "y1",
        "x1_alpha",
        "x1_beta",
        "HE_J_mol",
        "HE_kJ_mol",
        "gamma1_inf",
        "gamma2_inf",
        "weight",
        "pin",
        "validation_only",
        "pin_tolerance",
        "sigma",
        "source",
        "group",
        "id",
    }
    rows = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"Observation {index + 1} must be an object.")
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"Observation {index + 1}: unknown fields {', '.join(sorted(unknown))}."
            )
        row = dict(raw)
        for key in allowed - {"kind", "type", "pin", "validation_only", "source", "group", "id"}:
            if key in row and is_missing_cell(row[key]):
                del row[key]
        if (
            "kind" in row
            and "type" in row
            and str(row["kind"]).upper() != str(row["type"]).upper()
        ):
            raise ValueError(f"Observation {index + 1}: kind and type disagree.")
        kind = str(row.pop("type", row.get("kind", "VLE"))).upper()
        row["kind"] = _ALIASES.get(kind, kind)
        if row["kind"] not in KINDS:
            raise ValueError(
                f"Observation {index + 1}: kind must be one of {', '.join(KINDS)}."
            )
        for target, choices in (
            ("T_K", {"T_K": (1, 0), "T_C": (1, 273.15)}),
            ("P_bar", {"P_bar": (1, 0), "P_kPa": (0.01, 0), "P_atm": (1.01325, 0)}),
            ("HE_J_mol", {"HE_J_mol": (1, 0), "HE_kJ_mol": (1000, 0)}),
        ):
            present = [key for key in choices if key in row]
            if len(present) > 1:
                raise ValueError(
                    f"Observation {index + 1}: specify {target} in only one unit."
                )
            if present:
                key = present[0]
                factor, offset = choices[key]
                row[target] = _number(row.pop(key), key) * factor + offset
        if "T_K" not in row or row["T_K"] <= 0:
            raise ValueError(
                f"Observation {index + 1}: positive T_K or T_C above absolute zero is required."
            )
        for key in ("x1", "y1", "x1_alpha", "x1_beta"):
            if key in row:
                row[key] = _number(
                    row[key],
                    key,
                    low=0.000001 if kind in ("UCST", "LCST") else 0,
                    high=0.999999 if kind in ("UCST", "LCST") else 1,
                )
        if "P_bar" in row:
            row["P_bar"] = _number(row["P_bar"], "P_bar", low=1e-9, high=10000)
        required = {
            "VLE": ("x1", "P_bar"),
            "LLE": (),
            "VLLE": ("P_bar",),
            "HE": ("x1", "HE_J_mol"),
            "GAMMA_INF": (),
            "AZEOTROPE": ("x1", "P_bar"),
            "UCST": (),
            "LCST": (),
        }[row["kind"]]
        missing = set(required) - row.keys()
        if missing:
            raise ValueError(
                f"Observation {index + 1}: missing {', '.join(sorted(missing))}."
            )
        measurements = {
            "x1",
            "y1",
            "P_bar",
            "x1_alpha",
            "x1_beta",
            "HE_J_mol",
            "gamma1_inf",
            "gamma2_inf",
        }
        applicable = {
            "VLE": {"x1", "y1", "P_bar"},
            "LLE": {"x1_alpha", "x1_beta", "P_bar"},
            "VLLE": {"x1_alpha", "x1_beta", "P_bar", "y1"},
            "HE": {"x1", "HE_J_mol"},
            "GAMMA_INF": {"gamma1_inf", "gamma2_inf"},
            "AZEOTROPE": {"x1", "y1", "P_bar"},
            "UCST": {"x1"},
            "LCST": {"x1"},
        }
        unused = row.keys() & measurements - applicable[row["kind"]]
        if unused:
            raise ValueError(
                f"Observation {index + 1}: {', '.join(sorted(unused))} does not apply to {row['kind']}."
            )
        if row["kind"] == "VLLE" and ("x1_alpha" in row) != ("x1_beta" in row):
            raise ValueError(
                "VLLE needs both liquid endpoints, or neither so the phases can be inferred."
            )
        if row["kind"] == "VLLE" and "y1" in row and not 0 < row["y1"] < 1:
            raise ValueError(
                "VLLE vapor composition must be strictly between zero and one."
            )
        if row["kind"] == "LLE":
            endpoints = row.keys() & {"x1_alpha", "x1_beta"}
            if not endpoints:
                raise ValueError("LLE needs x1_alpha and/or x1_beta.")
            if any(not 0 < row[key] < 1 for key in endpoints):
                raise ValueError("LLE endpoints must be strictly between zero and one.")
        if row["kind"] in ("LLE", "VLLE") and {"x1_alpha", "x1_beta"} <= row.keys():
            a, b = sorted((row["x1_alpha"], row["x1_beta"]))
            if not 0 < a < b < 1:
                raise ValueError(
                    "LLE endpoints must be distinct and strictly between zero and one."
                )
            row["x1_alpha"], row["x1_beta"] = a, b
        if row["kind"] in ("VLE", "AZEOTROPE"):
            if not 0 < row["x1"] < 1 or ("y1" in row and not 0 < row["y1"] < 1):
                raise ValueError(
                    "VLE/azeotrope compositions must be strictly between zero and one."
                )
            if row["kind"] == "AZEOTROPE" and "y1" in row and row["y1"] != row["x1"]:
                raise ValueError(
                    "An azeotrope requires y1=x1; omit y1 or use the same value."
                )
        if row["kind"] == "GAMMA_INF" and not any(
            key in row for key in ("gamma1_inf", "gamma2_inf")
        ):
            raise ValueError("GAMMA_INF needs gamma1_inf and/or gamma2_inf.")
        for key in ("gamma1_inf", "gamma2_inf"):
            if key in row:
                row[key] = _number(row[key], key, low=1e-30)
        row["weight"] = _number(row.get("weight", 1), "weight", low=0, high=1e12)
        row["pin"] = _boolean(row.get("pin", False), "pin")
        row["validation_only"] = _boolean(
            row.get("validation_only", False), "validation_only"
        )
        if row["pin"] and row["validation_only"]:
            raise ValueError("A validation-only point cannot be a hard pin.")
        row["pin_tolerance"] = _number(
            row.get("pin_tolerance", 1e-5), "pin_tolerance", low=1e-9, high=1
        )
        sigma = row.get("sigma", {})
        if isinstance(sigma, str):
            sigma = json.loads(sigma) if sigma.startswith("{") else float(sigma)
        if isinstance(sigma, dict):
            if set(sigma) - SCALES.keys():
                raise ValueError(f"sigma keys must be {', '.join(SCALES)}.")
            sigma = {
                key: _number(val, f"sigma.{key}", low=1e-12)
                for key, val in sigma.items()
            }
        else:
            sigma = _number(sigma, "sigma", low=1e-12)
        row["sigma"] = sigma
        row["id"] = str(row.get("id", index + 1))
        row["source"] = str(row.get("source", ""))
        row["group"] = str(row.get("group", row["source"] or row["id"]))
        rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Observation ids must be unique.")
    return rows


def normalize_fit_request(request):
    """Validate controls without resolving properties or running calculations."""
    if not isinstance(request, dict):
        raise ValueError("A fit request must be an object.")
    allowed = {
        "components",
        "model",
        "vapor",
        "form",
        "alpha",
        "fit_alpha",
        "rq",
        "observations",
        "weights",
        "scales",
        "cv",
        "starts",
        "max_nfev",
        "seed",
        "T_ref_K",
        "initial",
        "bounds",
        "pfd_text",
        "scope",
        "online_lookup",
        "source",
        "vapor_parameters",
        "extrapolation",
        "import_options",
        "import_report",
        "psat",
        "component_properties",
        "estimate_properties",
        "allow_hoc_eta_default",
    }
    unknown = request.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown fit settings: {', '.join(sorted(unknown))}.")
    result = deepcopy(request)
    components = result.get("components")
    if (
        not isinstance(components, list)
        or len(components) != 2
        or any(not isinstance(c, str) or not c.strip() for c in components)
    ):
        raise ValueError("Choose two distinct component identifiers.")
    result["components"] = [c.strip() for c in components]
    if components[0] == components[1]:
        raise ValueError("Choose two distinct components.")
    result["model"] = str(result.get("model", "NRTL")).upper()
    result["vapor"] = str(result.get("vapor", "IDEAL")).upper()
    result["form"] = str(result.get("form", "constant_inverse"))
    if (
        result["model"] not in ("NRTL", "UNIQUAC")
        or result["vapor"] not in VAPORS
        or result["form"] not in FORMS
    ):
        raise ValueError(
            "Choose a supported model, vapor treatment and temperature form."
        )
    result["fit_alpha"] = _boolean(result.get("fit_alpha", False), "fit_alpha")
    if result["fit_alpha"] and result["model"] != "NRTL":
        raise ValueError("Fitting alpha is available only for NRTL.")
    result["alpha"] = _number(result.get("alpha", 0.3), "alpha", low=0.01, high=1)
    result["T_ref_K"] = _number(result.get("T_ref_K", 298.15), "T_ref_K", low=1)
    result["online_lookup"] = _boolean(
        result.get("online_lookup", False), "online_lookup"
    )
    result["estimate_properties"] = _boolean(
        result.get("estimate_properties", True), "estimate_properties"
    )
    result["allow_hoc_eta_default"] = _boolean(
        result.get("allow_hoc_eta_default", True), "allow_hoc_eta_default"
    )
    if "psat" in result:
        if not isinstance(result["psat"], list) or len(result["psat"]) != 2:
            raise ValueError(
                "psat must contain one definition (or null) for each of the two components."
            )
        result["psat"] = [normalize_psat(spec) for spec in result["psat"]]
    if "component_properties" in result:
        properties = result["component_properties"]
        if (
            not isinstance(properties, list)
            or len(properties) != 2
            or any(
                not isinstance(item, dict)
                or item.keys()
                - {
                    "MW",
                    "Tc_K",
                    "Pc_bar",
                    "Tb_K",
                    "omega",
                    "Vc_cm3_mol",
                    "Zc",
                    "dipole_D",
                    "hoc_eta",
                    "Rprime_A",
                    "smiles",
                }
                for item in properties
            )
        ):
            raise ValueError(
                "component_properties must be two objects using MW, Tc_K, Pc_bar, Tb_K and omega."
            )
        result["component_properties"] = [
            {
                key: str(val).strip()
                if key == "smiles"
                else _number(
                    val,
                    f"component_properties.{key}",
                    low=-1
                    if key == "omega"
                    else 0
                    if key in ("dipole_D", "hoc_eta", "Rprime_A")
                    else 1e-9,
                )
                for key, val in item.items()
            }
            for item in properties
        ]
    for key, default, high in (
        ("starts", 3, 10),
        ("max_nfev", 500, 5000),
        ("seed", 1729, 2**32 - 1),
    ):
        val = _number(
            result.get(key, default), key, low=1 if key != "seed" else 0, high=high
        )
        if not val.is_integer():
            raise ValueError(f"{key} must be an integer.")
        result[key] = int(val)
    for field, defaults in (("weights", dict.fromkeys(KINDS, 1.0)), ("scales", SCALES)):
        values = result.get(field, {})
        if not isinstance(values, dict) or values.keys() - defaults.keys():
            raise ValueError(f"{field} keys must be {', '.join(defaults)}.")
        result[field] = {
            **defaults,
            **{
                key: _number(
                    val,
                    f"{field}.{key}",
                    low=0 if field == "weights" else 1e-12,
                    high=1e12,
                )
                for key, val in values.items()
            },
        }
    if (
        isinstance(result.get("observations"), str)
        or tabular_matrix(result.get("observations")) is not None
    ):
        proposal = inspect_observations(
            result["observations"],
            import_options=result.get("import_options"),
            components=result["components"],
        )
        if not proposal["ready"]:
            raise ValueError(
                "Table interpretation needs input: "
                + " ".join(proposal["issues"])
                + " Supply import_options or use the browser import preview."
            )
        result["observations"] = proposal["observations"]
        result["import_report"] = [
            {key: value for key, value in proposal.items() if key != "observations"}
        ]
    else:
        result["observations"] = parse_observations(result.get("observations"))
    result["observations"] = [
        row
        for row in result["observations"]
        if row["pin"]
        or row["validation_only"]
        or row["weight"] * result["weights"][row["kind"]] > 0
    ]
    if not result["observations"]:
        raise ValueError("Enable at least one objective or pinned observation.")
    if not any(not row["validation_only"] for row in result["observations"]):
        raise ValueError(
            "At least one training observation is required; all points are validation-only."
        )
    rq = result.get("rq")
    if rq is not None:
        if result["model"] != "UNIQUAC" or not isinstance(rq, list) or len(rq) != 2:
            raise ValueError("UNIQUAC rq must be two objects containing r and q.")
        if any(not isinstance(item, dict) or set(item) != {"r", "q"} for item in rq):
            raise ValueError("Each UNIQUAC rq entry must contain exactly r and q.")
        result["rq"] = [
            {
                key: _number(item.get(key), f"rq[{i}].{key}", low=1e-6, high=1000)
                for key in ("r", "q")
            }
            for i, item in enumerate(rq)
        ]
    cv = result.get("cv") or {"method": "none"}
    if not isinstance(cv, dict) or cv.keys() - {"method", "folds"}:
        raise ValueError("cv accepts method and folds.")
    cv["method"] = cv.get("method", "none")
    if cv["method"] not in (
        "none",
        "kfold",
        "leave_temperature_out",
        "leave_group_out",
    ):
        raise ValueError(
            "Choose none, kfold, leave_temperature_out or leave_group_out cross-validation."
        )
    folds = _number(cv.get("folds", 5), "cv.folds", low=2, high=20)
    if not folds.is_integer():
        raise ValueError("cv.folds must be an integer.")
    cv["folds"] = int(folds)
    result["cv"] = cv
    if not isinstance(result.get("scope", "global"), str) or not isinstance(
        result.get("pfd_text", ""), str
    ):
        raise ValueError("scope and pfd_text must be text.")
    policy = result.get("extrapolation", "unrestricted")
    if policy not in (
        "unrestricted",
        "constant_inverse",
        "inverse_linear_quadratic",
        "inverse_square_cubic",
    ):
        raise ValueError("Choose a supported extrapolation policy.")
    result["extrapolation"] = policy
    temperatures = [
        row["T_K"] for row in result["observations"] if not row["validation_only"]
    ]
    if min(temperatures) == max(temperatures):
        training_kinds = {
            row["kind"] for row in result["observations"] if not row["validation_only"]
        }
        has_value_and_derivative = "HE" in training_kinds and bool(training_kinds - {"HE"})
        term_count = len(FORMS[result["form"]])
        if term_count > (2 if has_value_and_derivative else 1):
            raise ValueError(
                "A single-temperature fit permits one term, or two terms when "
                "training data include both equilibrium/activity values and HE "
                "(the temperature derivative). More terms require training "
                "measurements at multiple temperatures."
            )
    if result["form"] == "constant" and any(
        row["kind"] == "HE" and not row["validation_only"] and abs(row["HE_J_mol"]) > 0
        for row in result["observations"]
    ):
        raise ValueError(
            "A constant activity law gives HE=0. Choose a temperature-dependent form to fit nonzero HE."
        )
    return result


def _method(request):
    vapor = (
        request["vapor"]
        if any(row["kind"] in VAPOR_OBJECTIVES for row in request["observations"])
        else "IDEAL"
    )
    suffix = (
        ""
        if vapor == "IDEAL"
        else "-" + vapor
        if vapor in ("RK", "PR", "VDM")
        else "-BV"
    )
    return request["model"] + suffix


def prepare_fit(request):
    """Resolve component definitions through the existing PFD initialization path."""
    request = normalize_fit_request(request)
    pfd = (
        parse_pfd(request["pfd_text"])
        if request.get("pfd_text")
        else ProcessFlowDiagram()
    )
    scope = request.get("scope", "global")
    selected = pfd.get_thermo_scope(scope) if scope != "global" else None
    if scope != "global" and selected is None:
        raise ValueError(f"Thermodynamic scope {scope!r} does not exist.")
    symbols = []
    for i, identifier in enumerate(request["components"]):
        component = next(
            (c for c in pfd.components if identifier in (c.symbol, c.name)), None
        )
        if component is None:
            if request.get("pfd_text"):
                raise ValueError(
                    f"Component {identifier!r} is not defined in this PFD."
                )
            component = Component(symbol=f"Fit_{i + 1}", name=identifier)
            pfd.components.append(component)
        if component.phase_behavior == "permanent_solid":
            raise ValueError("Activity fitting requires fluid components.")
        for name, value in request.get("component_properties", [{}, {}])[i].items():
            setattr(component, COMPONENT_FIELDS[name], value)
        if request.get("psat"):
            apply_psat(component, request["psat"][i])
        if request.get("rq"):
            component.uniquac_r = request["rq"][i]["r"]
            component.uniquac_q = request["rq"][i]["q"]
        symbols.append(component.symbol)
    if len(set(symbols)) != 2:
        raise ValueError("Choose two distinct components.")
    # Fit a binary package. Preserve its physical definitions and vapor cross
    # corrections, but do not initialize or solve the surrounding equipment.
    binary = ProcessFlowDiagram()
    binary.components = [deepcopy(pfd.get_component(c)) for c in symbols]
    binary.metadata.thermo_method = request["model"]
    binary.metadata.thermo_options = {}
    binary.metadata.online_lookup = request["online_lookup"]
    binary.metadata.psat_minimum_pressure_bar = pfd.metadata.psat_minimum_pressure_bar
    binary.metadata.allow_computation = True
    binary.metadata.fluid_phase_model = "VLE"
    candidates = [
        {
            "component1": item.component1,
            "component2": item.component2,
            "model": item.model,
            "scope": item.scope,
            "parameters": item.parameters,
        }
        for item in pfd.interaction_parameters
        if {item.component1, item.component2} == set(symbols)
        and item.model.upper() not in ("NRTL", "UNIQUAC")
    ]
    binary.interaction_parameters = [
        InteractionParameter(
            item["component1"],
            item["component2"],
            item["model"],
            parameters=deepcopy(item["parameters"]),
        )
        for item in pfd.effective_scoped_records(candidates, scope)
    ]
    # Explicitly override database activity coefficients, including at zero.
    prefix = "c" if request["model"] == "NRTL" else "a"
    parameters = {
        f"tau12_{prefix}": 0.0,
        f"tau21_{prefix}": 0.0,
        "tau_tref": request["T_ref_K"],
    }
    if request["model"] == "NRTL":
        parameters["alpha12"] = request["alpha"]
    binary.interaction_parameters.append(
        InteractionParameter(*symbols, request["model"], parameters=parameters)
    )
    try:
        sim = Simulator.from_string(binary.to_pfd())
        intended = (
            request["model"] + "-HOC" if request["vapor"] == "HOC" else _method(request)
        )
        sim.initialize(
            property_methods=[intended]
            if any(row["kind"] in VAPOR_OBJECTIVES for row in request["observations"])
            else []
        )
    except _MODEL_ERRORS as error:
        raise ValueError(f"Cannot initialize the fitting model: {error}") from error
    thermo = sim.thermo_packages["global"]
    property_records, property_warnings = [], list(thermo.warnings)
    if any(row["kind"] in VAPOR_OBJECTIVES for row in request["observations"]):
        property_records, warnings = prepare_auxiliary_properties(
            thermo, binary, request
        )
        property_warnings.extend(warnings)
        base = thermo
        # This database belongs to this fitting context. Model constructors
        # deepcopy its templates, so pass resolved component values through
        # those templates before constructing the intended vapor provider.
        for component in symbols:
            base.db.chemicals[component]=deepcopy(base.props[component])
        binary.metadata.thermo_method = _method(request)
        binary.metadata.thermo_options = (
            {"correlation": request["vapor"]}
            if request["vapor"] not in ("IDEAL", "RK", "PR", "VDM")
            else {}
        )
        thermo = create_thermodynamics(
            symbols,
            _method(request),
            db=base.db,
            interaction_overrides=base.interaction_overrides,
            thermo_options=binary.metadata.thermo_options,
        )
        thermo._resolver_known_props = deepcopy(base._resolver_known_props)
        install_fitting_psat(thermo,binary)
        property_warnings.extend(thermo.warnings)
        property_records.extend(
            qualify_psat(
                thermo,
                [
                    row["T_K"]
                    for row in request["observations"]
                    if row["kind"] in VAPOR_OBJECTIVES
                ],
                request,
            )
        )
    elif request["vapor"] != "IDEAL":
        property_warnings.append(
            f"No vapor-equilibrium objective is present; {request['vapor']} is not needed and the liquid model is fitted independently."
        )
    if not any(row["kind"] in VAPOR_OBJECTIVES for row in request["observations"]):
        install_fitting_psat(thermo,binary)
    identities = [thermo.props[c].CAS or thermo.props[c].name.lower() for c in symbols]
    if len(set(identities)) != 2:
        raise ValueError("These identifiers resolve to the same chemical.")
    problem = FitProblem(request, thermo, binary)
    problem.property_records, problem.property_warnings = (
        property_records,
        list(dict.fromkeys(property_warnings)),
    )
    return problem


class FitProblem:
    """Mutable fitting context; never mutates a caller's thermodynamic package."""

    numerical_errors = (ValueError, OverflowError, FloatingPointError, *_MODEL_ERRORS)

    def __init__(self, request, thermo, definition):
        self.request, self.thermo, self.definition = request, thermo, definition
        self.components = tuple(thermo.components)
        self.model = request["model"]
        self.terms = FORMS[request["form"]]
        self.names = [
            f"{direction}.{term}" for direction in ("12", "21") for term in self.terms
        ]
        self.scales = [
            self._coefficient_scale(term) for _ in range(2) for term in self.terms
        ]
        if request["fit_alpha"]:
            self.names.append("alpha12")
            self.scales.append(1.0)
        self.critical_names = {}
        for row in request["observations"]:
            if (
                row["kind"] in ("UCST", "LCST")
                and "x1" not in row
                and not row["validation_only"]
            ):
                name = f"critical_x1.{row['id']}"
                self.critical_names[row["id"]] = len(self.names)
                self.names.append(name)
                self.scales.append(1.0)
        self.vlle_names = {}
        self.lle_names = {}
        for row in request["observations"]:
            if (
                row["kind"] == "LLE"
                and ("x1_alpha" in row) != ("x1_beta" in row)
                and not row["validation_only"]
            ):
                self.lle_names[row["id"]] = len(self.names)
                self.names.append(f"lle_gap.{row['id']}")
                self.scales.append(1.0)
            if (
                row["kind"] == "VLLE"
                and "x1_alpha" not in row
                and not row["validation_only"]
            ):
                self.vlle_names[row["id"]] = (len(self.names), len(self.names) + 1)
                self.names.extend((f"vlle_xa.{row['id']}", f"vlle_gap.{row['id']}"))
                self.scales.extend((1.0, 1.0))
        self.vapor_specs = deepcopy(request.get("vapor_parameters", []))
        if not any(
            row["kind"] in VAPOR_OBJECTIVES and not row["validation_only"]
            for row in request["observations"]
        ) and any(spec.get("fit") for spec in self.vapor_specs):
            raise ValueError(
                "Fitting vapor parameters requires vapor-equilibrium training data; validation-only points cannot determine them."
            )
        if not any(row["kind"] in VAPOR_OBJECTIVES for row in request["observations"]):
            if any(spec.get("fit") for spec in self.vapor_specs):
                raise ValueError(
                    "Fitting vapor parameters requires vapor-equilibrium training data."
                )
            self.vapor_specs = []
        if not isinstance(self.vapor_specs, list):
            raise ValueError("vapor_parameters must be an array.")
        for spec in self.vapor_specs:
            if not isinstance(spec, dict) or spec.keys() - {
                "model",
                "field",
                "value",
                "fit",
                "lower",
                "upper",
                "scale",
            }:
                raise ValueError(
                    "Vapor parameter fields: model, field, value, fit, lower, upper, scale."
                )
            allowed = {
                "VDM": {"delta_H_residual_J_per_mol", "delta_S_residual_J_per_mol_K"},
                "HOC": {"eta"},
                "TSONOPOULOS": {"kij"},
                "PR": {"kij"},
                "SRK": {"kij"},
            }
            expected = {
                "VDM": "VDM",
                "HOC": "HOC",
                "TSONOPOULOS": "TSONOPOULOS",
                "PR": "PR",
            }.get(request["vapor"])
            if spec.get("model") != expected or spec.get("field") not in allowed.get(
                expected, set()
            ):
                raise ValueError(
                    "Vapor parameters must match the selected vapor treatment."
                )
            spec["value"] = _number(spec.get("value", 0), "vapor parameter value")
            spec["fit"] = _boolean(spec.get("fit", False), "vapor parameter fit")
            if spec["fit"]:
                spec["index"] = len(self.names)
                self.names.append(f"vapor.{spec['model']}.{spec['field']}")
                default_scale = (
                    10
                    if spec["field"].endswith("_J_per_mol_K")
                    else 1000
                    if "_J_per_mol" in spec["field"]
                    else 1
                )
                self.scales.append(
                    _number(spec.get("scale", default_scale), "vapor scale", low=1e-12)
                )
        if len({(s["model"], s["field"]) for s in self.vapor_specs}) != len(
            self.vapor_specs
        ):
            raise ValueError("Vapor parameters must be unique.")
        if (
            any(spec["model"] == "VDM" for spec in self.vapor_specs)
            and len(thermo._vdm_models) < 2
        ):
            raise ValueError(
                "VDM cross-association parameters require two associating components; supply their component VDM definitions in a PFD if needed."
            )
        self.lower, self.upper = (
            np.full(len(self.names), -30.0),
            np.full(len(self.names), 30.0),
        )
        self.initial = np.zeros(len(self.names))
        for i, name in enumerate(self.names):
            if name == "alpha12":
                self.lower[i], self.upper[i], self.initial[i] = (
                    0.01,
                    1,
                    request["alpha"],
                )
            elif name.startswith("critical_x1"):
                self.lower[i], self.upper[i], self.initial[i] = 0.005, 0.995, 0.5
            elif name.startswith("vlle_"):
                self.lower[i], self.upper[i], self.initial[i] = (
                    (1e-6, 0.999, 0.05)
                    if name.startswith("vlle_xa")
                    else (0.001, 0.999, 0.95)
                )
            elif name.startswith("lle_gap."):
                self.lower[i], self.upper[i], self.initial[i] = 1e-6, 1 - 1e-6, 0.95
        for spec in self.vapor_specs:
            if spec["fit"]:
                i = spec["index"]
                self.initial[i] = spec["value"] / self.scales[i]
                self.lower[i] = (
                    _number(spec.get("lower", -10 * self.scales[i]), "vapor lower")
                    / self.scales[i]
                )
                self.upper[i] = (
                    _number(spec.get("upper", 10 * self.scales[i]), "vapor upper")
                    / self.scales[i]
                )
        for field in ("initial", "bounds"):
            values = request.get(field, {})
            if not isinstance(values, dict) or values.keys() - set(self.names):
                raise ValueError(
                    f"{field} keys must match parameter names: {', '.join(self.names)}."
                )
            for name, value in values.items():
                i = self.names.index(name)
                if field == "initial":
                    self.initial[i] = _number(value, name) / self.scales[i]
                else:
                    if not isinstance(value, list) or len(value) != 2:
                        raise ValueError(f"bounds.{name} must be [lower, upper].")
                    self.lower[i], self.upper[i] = [
                        _number(v, name) / self.scales[i] for v in value
                    ]
        if (
            np.any(self.lower >= self.upper)
            or np.any(self.initial < self.lower)
            or np.any(self.initial > self.upper)
        ):
            raise ValueError("Bounds must increase and contain initial values.")
        if request["fit_alpha"]:
            i = self.names.index("alpha12")
            if self.lower[i] < 0.01 or self.upper[i] > 1:
                raise ValueError("NRTL alpha bounds must remain within [0.01, 1].")
        for i in self.critical_names.values():
            if self.lower[i] < 0.001 or self.upper[i] > 0.999:
                raise ValueError(
                    "Critical composition bounds must remain within [0.001, 0.999]."
                )
        for first, gap in self.vlle_names.values():
            if (
                self.lower[first] <= 0
                or self.upper[first] >= 1
                or self.lower[gap] <= 0
                or self.upper[gap] >= 1
            ):
                raise ValueError(
                    "VLLE latent phase coordinates must be bounded strictly inside (0,1)."
                )
        for index in self.lle_names.values():
            if self.lower[index] <= 0 or self.upper[index] >= 1:
                raise ValueError(
                    "LLE latent phase coordinates must be bounded strictly inside (0,1)."
                )
        self.last = None
        self.last_vapor = None
        self.model_indices = np.array([
            i for i, name in enumerate(self.names)
            if name.startswith(("12.", "21.", "vapor.")) or name == "alpha12"
        ])

    def _coefficient_scale(self, term):
        tref = self.request["T_ref_K"]
        return {
            "constant": 1,
            "inverse": tref,
            "anchored": 1,
            "linear": 1 / tref,
            "quadratic": 1 / tref**2,
        }[term]

    def parameters(self, values):
        fields = (
            ("c", "d", "e", "f", "g")
            if self.model == "NRTL"
            else ("a", "b", "c", "d", "e")
        )
        mapping = dict(
            zip(("constant", "inverse", "anchored", "linear", "quadratic"), fields)
        )
        result = {
            f"tau{direction}_{field}": 0.0
            for direction in ("12", "21")
            for field in fields
        }
        for i, name in enumerate(self.names):
            if name[:2] in ("12", "21"):
                direction, term = name.split(".")
                result[f"tau{direction}_{mapping[term]}"] = float(
                    values[i] * self.scales[i]
                )
        result["tau_tref"] = self.request["T_ref_K"]
        temperatures = [
            row["T_K"]
            for row in self.request["observations"]
            if not row["validation_only"]
        ]
        result.update(
            Tmin_K=min(temperatures),
            Tmax_K=max(temperatures),
            extrapolation=self.request["extrapolation"],
        )
        if self.model == "NRTL":
            result["alpha12"] = (
                float(values[self.names.index("alpha12")])
                if self.request["fit_alpha"]
                else self.request["alpha"]
            )
        else:
            result.update(model_variant="standard_uniquac", use_q_prime=False)
        return result

    def vapor_records(self, values):
        records = {}
        for spec in self.vapor_specs:
            record = records.setdefault(
                spec["model"],
                {
                    "model": spec["model"],
                    "component1": self.components[0],
                    "component2": self.components[1],
                },
            )
            record[spec["field"]] = (
                float(values[spec["index"]] * self.scales[spec["index"]])
                if spec["fit"]
                else spec["value"]
            )
        if "VDM" in records:
            records["VDM"].setdefault("delta_H_residual_J_per_mol", 0.0)
            records["VDM"].setdefault("delta_S_residual_J_per_mol_K", 0.0)
        return list(records.values())

    def install(self, values):
        if self.last is not None and np.array_equal(values, self.last):
            return
        if self.last is not None and np.array_equal(
            np.asarray(values)[self.model_indices], self.last[self.model_indices]
        ):
            self.last = np.array(values, copy=True)
            return
        vapor = self.vapor_records(values)
        if vapor and vapor != self.last_vapor:
            # Reuse resolved properties; rebuild the authoritative vapor provider
            # when its parameters change, including association-model caches.
            overrides = [
                {
                    "model": item.model,
                    "component1": item.component1,
                    "component2": item.component2,
                    **item.parameters,
                }
                for item in self.definition.interaction_parameters
                if item.model != self.model
                and item.model not in {r["model"] for r in vapor}
            ]
            overrides += vapor + [
                {
                    "model": self.model,
                    "component1": self.components[0],
                    "component2": self.components[1],
                    **self.parameters(values),
                }
            ]
            previous = self.thermo
            self.thermo = create_thermodynamics(
                list(self.components),
                _method(self.request),
                db=previous.db,
                interaction_overrides=overrides,
                thermo_options=self.definition.metadata.thermo_options,
            )
            self.thermo._resolver_known_props = deepcopy(previous._resolver_known_props)
            install_fitting_psat(self.thermo,self.definition)
            self.last_vapor = deepcopy(vapor)
        else:
            mapping = (
                self.thermo._nrtl_interaction_overrides
                if self.model == "NRTL"
                else self.thermo._uniquac_interaction_overrides
            )
            mapping[tuple(sorted(self.components))] = {
                "model": self.model,
                "component1": self.components[0],
                "component2": self.components[1],
                **self.parameters(values),
            }
            for name in (
                "_activity_cache",
                "_nrtl_matrix_cache",
                "_uniquac_tau_cache",
                "_compiled_activity_cache",
                "_compiled_lle_cache",
                "_k_values_cache",
            ):
                cache = getattr(self.thermo, name, None)
                if cache is not None:
                    cache.clear()
            if self.model == "NRTL":
                self.thermo._nrtl_parameter_cache = None
            else:
                self.thermo._uniquac_parameter_cache = None
            self.thermo._compiled_vlle = None
            self.thermo._compiled_vlle_initialized = False
        self.last = np.array(values, copy=True)

    def composition(self, x):
        return {self.components[0]: x, self.components[1]: 1 - x}

    def checked_gamma(self, T, x):
        gamma = self.thermo.activity_coefficients(T, self.composition(x))
        self._check_gamma_limits(T, np.array([gamma[c] for c in self.components]))
        return gamma

    def checked_gamma_many(self, T, x):
        """Reject runtime clipping, which destroys derivative consistency."""
        x = np.asarray(x, dtype=float)
        compositions = np.column_stack((x, 1 - x))
        temperatures = np.broadcast_to(np.asarray(T, dtype=float), x.shape)
        backend = self.thermo._compiled_activity_backend()
        for temperature in np.unique(temperatures):
            self.thermo._warn_activity_interaction_extrapolation(float(temperature))
        if backend is None:
            gamma = np.array([
                [values[c] for c in self.components]
                for temperature, composition in zip(temperatures, compositions)
                for values in [self.thermo.activity_coefficients(float(temperature), dict(zip(self.components, composition))) ]
            ], dtype=float).reshape(-1, len(self.components))
        else:
            gamma = backend.activity_coefficients_many(compositions, temperatures)
        self._check_gamma_limits(temperatures, gamma)
        return gamma

    def _check_gamma_limits(self, temperatures, gamma):
        if np.any(~np.isfinite(gamma) | (gamma <= 1e-12) | (gamma >= math.exp(50) * (1 - 1e-14))):
            raise ValueError(
                "Runtime activity-coefficient limit reached; narrow the coefficient bounds or choose a less flexible temperature law."
            )
        if self.model == "UNIQUAC":
            for temperature in np.unique(temperatures):
                tau = np.asarray(self.thermo._uniquac_tau_matrix(float(temperature)))
                if np.any((tau <= math.exp(-50) * (1 + 1e-14)) | (tau >= math.exp(50) * (1 - 1e-14))):
                    raise ValueError(
                        "Runtime UNIQUAC interaction-exponent limit reached; narrow the coefficient bounds."
                    )

    def chemical_potentials(self, T, x):
        gamma = self.checked_gamma(T, x)
        return np.log(np.maximum([
            x * gamma[self.components[0]], (1 - x) * gamma[self.components[1]],
        ], 1e-300))

    def chemical_potentials_many(self, T, x):
        x = np.asarray(x, dtype=float)
        gamma = self.checked_gamma_many(T, x)
        return np.log(np.maximum(np.column_stack((x, 1 - x)) * gamma, 1e-300))

    def gibbs(self, T, x):
        mu = self.chemical_potentials(T, x)
        return x * mu[0] + (1 - x) * mu[1]

    def critical_derivatives(self, T, x):
        h = min(0.001, x / 4, (1 - x) / 4)
        mu = self.chemical_potentials_many(T, x + h * np.arange(-2, 3))
        m = mu[:, 0] - mu[:, 1]
        curvature = (m[0] - 8 * m[1] + 8 * m[3] - m[4]) / (12 * h)
        third = (-m[0] + 16 * m[1] - 30 * m[2] + 16 * m[3] - m[4]) / (12 * h * h)
        fourth = (-m[0] + 2 * m[1] - 2 * m[3] + m[4]) / (2 * h**3)
        return float(curvature), float(third), float(fourth)

    def _sigma(self, row, name):
        sigma = row["sigma"]
        return (
            sigma.get(name, self.request["scales"][name])
            if isinstance(sigma, dict)
            else sigma
        )

    def _check_psat_temperature(self, T):
        if not hasattr(self, "_qualified_temperatures"):
            self._qualified_temperatures = set()
        if float(T) not in self._qualified_temperatures:
            qualify_psat(self.thermo, [float(T)], self.request)
            self._qualified_temperatures.add(float(T))

    def predict_vle(self, x, *, T=None, P=None, T_guess=350):
        composition = self.composition(x)
        if T is None:
            T = self.thermo.bubble_point_T(composition, P, T_guess=T_guess)
        else:
            P = self.thermo.bubble_point_P(composition, T)
        self._check_psat_temperature(T)
        self.checked_gamma(T, x)
        K = self.thermo.K_values(T, P, composition)
        vapor = {c: composition[c] * K[c] for c in self.components}
        total = sum(vapor.values())
        if abs(total - 1) > 2e-4:
            raise ValueError("Bubble prediction did not converge.")
        vapor = {c: v / total for c, v in vapor.items()}
        self.thermo.vapor_fugacity_coefficients(T, P, vapor)
        self.thermo._check_diagram_vle(composition, vapor, T, P)
        return {"T_K": float(T), "P_bar": float(P), "y1": vapor[self.components[0]]}

    def predict_lle(self, T, *, feed_x1=None):
        if feed_x1 is None:
            # This feed-independent search supplies only a feed hint. Its
            # stationary pair is not the final coexistence prediction.
            phases = self.thermo.binary_liquid_coexistence(T, tol=1e-8)
            if phases is None:
                raise ValueError("No stable two-liquid split is predicted.")
            feed_x1 = sum(phase[self.components[0]] for phase in phases) / 2
        for attempt in range(2):
            split, first, second, fraction = self.thermo.liquid_liquid_equilibrium(
                self.composition(feed_x1), T, tol=1e-8
            )
            if not split:
                raise ValueError("No stable two-liquid split is predicted.")
            a, b = sorted((first[self.components[0]], second[self.components[0]]))
            _, raw, stability = self._lle_errors(T, a, b, 1)
            if max(abs(value) for value in raw) <= 1e-6 and stability["minimum_tangent_gap"] >= -1e-6:
                return {
                    "x1_alpha": a, "x1_beta": b, "split": True,
                    "liquid_fraction": float(fraction),
                }
            if attempt == 0 and stability["minimum_tangent_gap"] < -1e-6:
                # The ordinary interpreted flash can inherit a collapsed
                # stationary hint. Move the feed toward the lower-Gibbs state
                # identified by the existing tangent audit and flash again.
                feed_x1 = ((a + b) / 2 + stability["minimum_tangent_gap_composition"]) / 2
                continue
            break
        raise ValueError("The predicted liquid coexistence fails chemical-potential or tangent stability checks.")

    def predict_vlle(self, *, T=None, P=None, T_guess=350):
        try:
            liquid = self.predict_lle(T if T is not None else T_guess)
        except ValueError:
            if T is not None:
                raise
            # An isobaric solve can cross a critical temperature: absence of
            # coexistence at its initial guess does not rule out the root.
            composition = self.composition(0.5)
        else:
            composition = self.composition(0.5 * (liquid["x1_alpha"] + liquid["x1_beta"]))
        state = (
            self.thermo._binary_invariant_at_temperature(composition, T, 100)
            if T is not None
            else self.thermo._binary_invariant_at_pressure(composition, P, T_guess, 100)
        )
        if state is None:
            raise ValueError("No stable binary three-phase invariant is predicted.")
        independent, phases = state
        if T is None:
            T = independent
        else:
            P = independent
        self._check_psat_temperature(T)
        a, b = sorted((phases.x1[self.components[0]], phases.x2[self.components[0]]))
        for liquid in (phases.x1, phases.x2):
            self.thermo._check_diagram_vle(liquid, phases.y, T, P)
        return {
            "T_K": float(T),
            "P_bar": float(P),
            "x1_alpha": a,
            "x1_beta": b,
            "y1": phases.y[self.components[0]],
            "phase_count": phases.phase_count,
        }

    def _lle_errors(self, T, a, b, sigma):
        grid = np.r_[np.linspace(0.00001, 0.99999, 41), a, b]
        potentials = self.chemical_potentials_many(T, grid)
        mua, mub = potentials[-2:]
        raw = (mua - mub).tolist()
        mu = (mua + mub) / 2
        # Keep the residual dimension fixed when a fitted endpoint crosses a
        # grid node. Repeated evaluation coordinates are harmless here.
        gaps = grid * (potentials[:, 0] - mu[0]) + (1 - grid) * (potentials[:, 1] - mu[1])
        scaled = np.r_[
            np.array(raw) / sigma, np.minimum(gaps, 0) / sigma / math.sqrt(len(grid))
        ]
        return (
            scaled,
            raw,
            {
                "log_activity_residuals": raw, "minimum_tangent_gap": float(min(gaps)),
                "minimum_tangent_gap_composition": float(grid[int(np.argmin(gaps))]),
            },
        )

    def _liquid_endpoints(self, values, row):
        if {"x1_alpha", "x1_beta"} <= row.keys():
            return row["x1_alpha"], row["x1_beta"]
        if row["id"] in self.lle_names:
            gap = values[self.lle_names[row["id"]]]
            if "x1_alpha" in row:
                a = row["x1_alpha"]
                return a, a + (1 - a) * gap
            b = row["x1_beta"]
            return b * (1 - gap), b
        if row["id"] in self.vlle_names:
            first, gap = self.vlle_names[row["id"]]
            a = values[first]
            return a, a + (1 - a) * values[gap]
        state = self.predict_lle(row["T_K"])
        return (
            row.get("x1_alpha", state["x1_alpha"]),
            row.get("x1_beta", state["x1_beta"]),
        )

    def _vle(self, row):
        T, P, x = row["T_K"], row["P_bar"], row["x1"]
        self.checked_gamma(T, x)
        liquid = self.thermo._phase_log_fugacities(T, P, self.composition(x), "liquid")
        y = row.get("y1", x if row["kind"] == "AZEOTROPE" else None)

        def errors(y1):
            if self.request["vapor"] != "IDEAL":
                # The runtime phase-audit helper permits an ideal-vapor fallback
                # after an EOS failure. Regression must report that failure.
                phi = self.thermo.vapor_fugacity_coefficients(
                    T, P, self.composition(y1)
                )
                if any(
                    not math.isfinite(value) or value <= 0 for value in phi.values()
                ):
                    raise ValueError(
                        "The selected vapor treatment returned invalid fugacity coefficients."
                    )
            vapor = self.thermo._phase_log_fugacities(
                T, P, self.composition(y1), "vapor"
            )
            return np.array([liquid[c] - vapor[c] for c in self.components])

        if y is None:
            if self.request["vapor"] == "IDEAL":
                a = np.exp([liquid[c] for c in self.components])
                y = float(a[0] / sum(a))
            else:
                y = brentq(
                    lambda z: float(np.diff(errors(z))[0]), 1e-12, 1 - 1e-12, xtol=1e-12
                )
            residual = [float(np.mean(errors(y)))]
        else:
            residual = errors(y).tolist()
        return residual, {
            "evaluation_y1": y,
            "log_fugacity_residuals": errors(y).tolist(),
        }

    def row_errors(self, values, row):
        self.install(values)
        kind, T = row["kind"], row["T_K"]
        if kind in ("VLE", "AZEOTROPE"):
            raw, prediction = self._vle(row)
            scaled = np.array(raw) / self._sigma(row, "log_fugacity")
        elif kind == "HE":
            delta = max(1e-3, T * 1e-4)
            for temperature in (max(1, T - delta), T, T + delta):
                self.checked_gamma(temperature, row["x1"])
            predicted = self.thermo.excess_enthalpy(self.composition(row["x1"]), T)
            raw = [predicted - row["HE_J_mol"]]
            scaled = np.array(raw) / self._sigma(row, "HE_J_mol")
            prediction = {"HE_J_mol": predicted}
        elif kind == "GAMMA_INF":
            raw, prediction = [], {}
            for index, key in enumerate(("gamma1_inf", "gamma2_inf")):
                if key in row:
                    # Runtime formula evaluates the exact zero-concentration limit.
                    gamma = self.checked_gamma(T, float(index))[self.components[index]]
                    prediction[key] = gamma
                    raw.append(math.log(gamma / row[key]))
            scaled = np.array(raw) / self._sigma(row, "log_gamma")
        elif kind == "LLE":
            a, b = self._liquid_endpoints(values, row)
            if not 0 < a < b < 1:
                raise ValueError(
                    "No distinct ordered LLE endpoints are available for this observation."
                )
            scaled, raw, prediction = self._lle_errors(
                T, a, b, self._sigma(row, "log_fugacity")
            )
            if ("x1_alpha" in row) != ("x1_beta" in row):
                # Equal-state chemical potentials are a trivial solution. Do
                # not let an unmeasured endpoint reduce the loss by collapsing
                # onto the measured endpoint instead of finding coexistence.
                scaled[:2] /= b - a
            prediction.update(
                evaluation_x1_alpha=float(a), evaluation_x1_beta=float(b)
            )
        elif kind == "VLLE":
            a, b = self._liquid_endpoints(values, row)
            first = {**row, "kind": "VLE", "x1": a}
            _, vapor = self._vle(first)
            y = vapor["evaluation_y1"]
            _, second = self._vle({**row, "kind": "VLE", "x1": b, "y1": y})
            raw = vapor["log_fugacity_residuals"] + second["log_fugacity_residuals"]
            tangent, _, stability = self._lle_errors(
                T, a, b, self._sigma(row, "log_fugacity")
            )
            scaled = np.r_[
                np.array(raw) / self._sigma(row, "log_fugacity"), tangent[2:]
            ]
            prediction = {
                **stability,
                "evaluation_x1_alpha": float(a),
                "evaluation_x1_beta": float(b),
                "evaluation_y1": y,
                "log_fugacity_residuals": raw,
            }
        else:
            x = row.get(
                "x1",
                values[self.critical_names[row["id"]]]
                if row["id"] in self.critical_names
                else 0.5,
            )
            if "x1" not in row and row["id"] not in self.critical_names:
                location = minimize(
                    lambda z: self.critical_derivatives(T, float(z[0]))[0],
                    [0.5],
                    bounds=[(0.005, 0.995)],
                    method="L-BFGS-B",
                )
                x = float(location.x[0])
            curvature, third, fourth = self.critical_derivatives(T, x)
            delta = max(0.01, T * 1e-4)
            below = self.critical_derivatives(T - delta, x)[0]
            above = self.critical_derivatives(T + delta, x)[0]
            slope = (above - below) / (2 * delta)
            sign = 1 if kind == "UCST" else -1
            raw = [curvature, third]
            scaled = np.array(
                [
                    curvature / self._sigma(row, "curvature"),
                    third / self._sigma(row, "third_derivative"),
                    max(0, 1e-4 - fourth),
                    max(0, 1e-6 - sign * slope) * T,
                ]
            )
            prediction = {
                "x1": float(x),
                "curvature": curvature,
                "third_derivative": third,
                "fourth_derivative": fourth,
                "temperature_slope": slope,
            }
        if not np.all(np.isfinite(scaled)):
            raise ValueError(f"Nonfinite prediction for observation {row['id']}.")
        return scaled, raw, prediction

    def residuals(self, values, rows, *, pins=False):
        output = []
        for row in rows:
            if row["validation_only"]:
                continue
            if pins != row["pin"]:
                continue
            errors, _, _ = self.row_errors(values, row)
            if pins:
                output.extend(errors)
            else:
                output.extend(
                    errors
                    * math.sqrt(row["weight"] * self.request["weights"][row["kind"]])
                )
        return np.asarray(output)

    def solve(self, rows, progress=None):
        rows = [row for row in rows if not row["validation_only"]]
        if not rows:
            raise ValueError("At least one training observation is required.")
        rng = np.random.default_rng(self.request["seed"])
        starts = [self.initial.copy()]
        for index in range(1, self.request["starts"]):
            start = self.initial.copy()
            start[: 2 * len(self.terms)] = rng.normal(0, 1, 2 * len(self.terms))
            # Positive NRTL interactions give a useful demixing initial guess.
            if (
                index == 1
                and self.model == "NRTL"
                and any(row["kind"] in ("LLE", "VLLE", "UCST", "LCST") for row in rows)
            ):
                start[0] = start[len(self.terms)] = 2.5
            starts.append(np.clip(start, self.lower + 1e-10, self.upper - 1e-10))
        candidates, failures = [], []
        for index, start in enumerate(starts):
            if progress:
                progress(
                    f"Fitting {len(rows)} observations · start {index + 1}/{len(starts)}"
                )
            try:
                if self.lle_names:
                    # Seed unmeasured branches from this start's actual
                    # coexistence. A fixed far-end guess can miss narrow or
                    # strongly asymmetric gaps, especially for UNIQUAC.
                    self.install(start)
                    coexistence = {}
                    for row in rows:
                        coordinate = self.lle_names.get(row["id"])
                        if coordinate is None or f"lle_gap.{row['id']}" in self.request.get(
                            "initial", {}
                        ):
                            continue
                        T = row["T_K"]
                        if T not in coexistence:
                            try:
                                coexistence[T] = self.predict_lle(T)
                            except (ValueError, OverflowError, *_MODEL_ERRORS):
                                coexistence[T] = None
                        phases = coexistence[T]
                        if phases is None:
                            continue
                        if "x1_alpha" in row:
                            a = row["x1_alpha"]
                            gap = (phases["x1_beta"] - a) / (1 - a)
                        else:
                            b = row["x1_beta"]
                            gap = (b - phases["x1_alpha"]) / b
                        if 0 < gap < 1:
                            start[coordinate] = np.clip(
                                gap, self.lower[coordinate], self.upper[coordinate]
                            )
                fitted = solve_start(self, rows, start)
                candidates.append(fitted)
            except (
                ValueError,
                OverflowError,
                FloatingPointError,
                *_MODEL_ERRORS,
            ) as error:
                failures.append(f"Start {index + 1}: {error}")
        if not candidates:
            raise ValueError(
                "No feasible fit was found. Check the model, bounds and pins. "
                + " ".join(failures)
            )
        candidates.sort(key=lambda item: (not item["success"], item["objective"]))
        return {**candidates[0], "failures": failures}

    def report(self, values, rows, *, audit=True):
        points, objectives = [], {}
        for row in rows:
            try:
                scaled, raw, prediction = self.row_errors(values, row)
            except (ValueError, OverflowError, *_MODEL_ERRORS) as error:
                if not row["validation_only"]:
                    raise
                points.append(
                    {
                        "id": row["id"],
                        "kind": row["kind"],
                        "observed": row,
                        "predicted": {"equilibrium_error": str(error)},
                        "raw_residuals": [],
                        "scaled_residuals": [],
                        "weighted_sum_squares": 0.0,
                        "physical": False,
                        "pin_satisfied": None,
                        "role": "validation",
                    }
                )
                continue
            physical = True
            if audit and row["kind"] in ("VLE", "AZEOTROPE"):
                x = self.composition(row["x1"])
                try:
                    at_temperature = self.predict_vle(row["x1"], T=row["T_K"])
                    split, _, _, _ = self.thermo.liquid_liquid_equilibrium(
                        x, row["T_K"], tol=1e-8
                    )
                    prediction.update(
                        P_bar=at_temperature["P_bar"],
                        y1=at_temperature["y1"],
                        y1_at_T=at_temperature["y1"],
                        P_error_bar=at_temperature["P_bar"] - row["P_bar"],
                        liquid_stable=not split,
                    )
                    isobaric = (
                        sum(
                            other["kind"] == row["kind"]
                            and math.isclose(
                                other.get("P_bar", -1), row["P_bar"], rel_tol=1e-7
                            )
                            for other in self.request["observations"]
                        )
                        >= 2
                    )
                    try:
                        at_pressure = self.predict_vle(
                            row["x1"], P=row["P_bar"], T_guess=row["T_K"]
                        )
                        prediction.update(
                            T_K=at_pressure["T_K"],
                            T_error_K=at_pressure["T_K"] - row["T_K"],
                            y1_at_P=at_pressure["y1"],
                        )
                        if isobaric:
                            prediction["y1"] = at_pressure["y1"]
                    except (
                        ValueError,
                        RuntimeError,
                        OverflowError,
                        *_MODEL_ERRORS,
                    ) as error:
                        prediction["temperature_error"] = str(error)
                        if isobaric:
                            prediction["y1"] = None
                    if "y1" in row or row["kind"] == "AZEOTROPE":
                        prediction["y1_error"] = (
                            prediction["y1"] - row.get("y1", row["x1"])
                            if prediction["y1"] is not None
                            else None
                        )
                    physical = not split
                except (
                    ValueError,
                    RuntimeError,
                    OverflowError,
                    *_MODEL_ERRORS,
                ) as error:
                    prediction["equilibrium_error"] = str(error)
                    physical = False
            elif audit and row["kind"] == "LLE":
                try:
                    a, b = self._liquid_endpoints(values, row)
                    prediction.update(self.predict_lle(row["T_K"], feed_x1=(a + b) / 2))
                except (ValueError, RuntimeError, OverflowError, *_MODEL_ERRORS) as error:
                    prediction["equilibrium_error"] = str(error)
                    physical = False
            elif audit and row["kind"] == "VLLE":
                try:
                    at_temperature = self.predict_vlle(T=row["T_K"])
                    prediction.update(at_temperature)
                    prediction["evaluated_T_K"] = prediction.pop("T_K")
                    try:
                        at_pressure = self.predict_vlle(
                            P=row["P_bar"], T_guess=row["T_K"]
                        )
                        prediction["T_K"] = at_pressure["T_K"]
                    except (
                        ValueError,
                        RuntimeError,
                        OverflowError,
                        *_MODEL_ERRORS,
                    ) as error:
                        prediction["temperature_error"] = str(error)
                    physical = at_temperature["phase_count"] == 3
                except (
                    ValueError,
                    RuntimeError,
                    OverflowError,
                    *_MODEL_ERRORS,
                ) as error:
                    prediction["equilibrium_error"] = str(error)
                    physical = False
            elif row["kind"] in ("UCST", "LCST"):
                sign = 1 if row["kind"] == "UCST" else -1
                physical = (
                    prediction["fourth_derivative"] > 0
                    and sign * prediction["temperature_slope"] > 0
                )
                if audit:
                    x = prediction["x1"]
                    mu = self.chemical_potentials(row["T_K"], x)
                    gap = min(
                        self.gibbs(row["T_K"], z) - (z * mu[0] + (1 - z) * mu[1])
                        for z in np.linspace(1e-5, 1 - 1e-5, 201)
                    )
                    prediction["minimum_tangent_gap"] = gap
                    physical = physical and gap >= -1e-5
                    try:
                        root = least_squares(
                            lambda z: np.array(
                                self.critical_derivatives(float(z[0]), float(z[1]))[:2]
                            ),
                            [row["T_K"], prediction["x1"]],
                            bounds=(
                                [max(1, row["T_K"] * 0.5), 0.005],
                                [row["T_K"] * 1.5, 0.995],
                            ),
                            diff_step=1e-4,
                            max_nfev=100,
                        )
                        if max(abs(root.fun)) > 1e-4:
                            raise ValueError(
                                "A model critical point could not be localized."
                            )
                        prediction.update(T_K=float(root.x[0]), x1=float(root.x[1]))
                    except (ValueError, OverflowError) as error:
                        prediction["critical_error"] = str(error)
            weighted = (
                float(scaled @ scaled)
                * row["weight"]
                * self.request["weights"][row["kind"]]
            )
            objective = objectives.setdefault(
                row["kind"], {"points": 0, "residuals": [], "weighted_sum_squares": 0.0}
            )
            objective["points"] += 1
            objective["residuals"].extend(scaled.tolist())
            if not row["pin"] and not row["validation_only"]:
                objective["weighted_sum_squares"] += weighted
            points.append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "observed": row,
                    "predicted": prediction,
                    "raw_residuals": raw,
                    "scaled_residuals": scaled.tolist(),
                    "weighted_sum_squares": weighted,
                    "physical": bool(physical),
                    "pin_satisfied": bool(
                        np.max(np.abs(scaled)) <= row["pin_tolerance"] + 1e-7
                    )
                    if row["pin"]
                    else None,
                    "role": "validation" if row["validation_only"] else "training",
                }
            )
        for objective in objectives.values():
            errors = np.asarray(objective.pop("residuals"))
            objective.update(
                scaled_RMSE=float(np.sqrt(np.mean(errors**2))),
                scaled_max_abs=float(max(abs(errors))),
            )
        return {
            "points": points,
            "objectives": objectives,
            "physical_metrics": physical_metrics(points),
        }


def _cv_folds(request):
    rows = request["observations"]
    eligible = [
        i for i, row in enumerate(rows) if not row["pin"] and not row["validation_only"]
    ]
    method = request["cv"]["method"]
    if method == "none":
        return []
    if len(eligible) < 2:
        raise ValueError("Cross-validation needs at least two unpinned observations.")
    if method == "kfold":
        rng = np.random.default_rng(request["seed"])
        count = min(request["cv"]["folds"], len(eligible))
        return [
            indices.tolist()
            for indices in np.array_split(rng.permutation(eligible), count)
        ]
    field = "T_K" if method == "leave_temperature_out" else "group"
    groups = sorted({rows[i][field] for i in eligible})
    if not 2 <= len(groups) <= 20:
        raise ValueError(
            "Grouped cross-validation needs 2–20 distinct unpinned groups."
        )
    return [[i for i in eligible if rows[i][field] == group] for group in groups]


def fit_interactions(request, *, progress=None):
    """Fit a binary system and optionally refit independent validation folds."""
    problem = prepare_fit(request)
    request = problem.request
    folds = _cv_folds(request)
    if progress:
        progress(
            f"Initialized {_method(request)} · {len(problem.names)} free parameters"
        )
    training_rows = [
        row for row in request["observations"] if not row["validation_only"]
    ]
    validation_rows = [row for row in request["observations"] if row["validation_only"]]
    optimum = problem.solve(training_rows, progress)
    values = optimum.pop("values")
    report = problem.report(values, training_rows)
    validation_only = problem.report(values, validation_rows)
    warnings = list(problem.property_warnings) + list(optimum["failures"])
    if any(not point["physical"] for point in validation_only["points"]):
        warnings.append(
            "Some validation-only observations could not be predicted or failed phase checks; they were not used in fitting."
        )
    if (
        len({row["T_K"] for row in request["observations"]}) == 1
        and len(problem.terms) > 1
        and not any(row["kind"] == "HE" for row in request["observations"])
    ):
        warnings.append(
            "Only one measurement temperature is present. A multi-term temperature law is not uniquely determined without additional caloric data."
        )
    if optimum["rank"] < len(problem.names):
        warnings.append(
            "The residual Jacobian is rank deficient; these data do not uniquely identify every fitted parameter."
        )
    if not optimum["success"]:
        warnings.append(
            "The optimizer stopped without convergence. Review the residuals before using this fit."
        )
    if any(not point["physical"] for point in report["points"]):
        warnings.append(
            "One or more equilibrium observations failed the final equilibrium/phase-stability audit."
        )
    at_bound = [
        name
        for i, name in enumerate(problem.names)
        if min(values[i] - problem.lower[i], problem.upper[i] - values[i]) < 1e-5
    ]
    if at_bound:
        warnings.append("Parameters at bounds: " + ", ".join(at_bound))
    cv = {
        "method": request["cv"]["method"],
        "folds": [],
        "pinned_points_policy": "Pins remain in every training fold and are never counted as held-out observations.",
    }
    for index, held in enumerate(folds):
        if progress:
            progress(f"Cross-validation fold {index + 1}/{len(folds)}")
        train = [
            row
            for i, row in enumerate(request["observations"])
            if i not in held and not row["validation_only"]
        ]
        test = [{**request["observations"][i], "validation_only": True} for i in held]
        # Held-out observations retain the required property/vapor context,
        # but never enter the objective or introduce latent fitted coordinates.
        # No all-data coefficients enter a validation fold.
        local_request = deepcopy(request)
        local_request["observations"] = train + test
        local_request["cv"] = {"method": "none"}
        training_ids = {row["id"] for row in train}
        local_request["initial"] = {
            k: v
            for k, v in request.get("initial", {}).items()
            if not k.startswith(("critical_x1.", "vlle_", "lle_"))
            or k.partition(".")[2] in training_ids
        }
        local_request["bounds"] = {
            k: v
            for k, v in request.get("bounds", {}).items()
            if not k.startswith(("critical_x1.", "vlle_", "lle_"))
            or k.partition(".")[2] in training_ids
        }
        try:
            local = prepare_fit(local_request)
            fitted = local.solve(
                [row for row in local.request["observations"] if not row["validation_only"]],
                progress,
            )
            # For an unobserved critical composition, locate the curvature
            # minimum using only the fitted thermodynamic surface.
            for row in test:
                if row["kind"] in ("UCST", "LCST") and "x1" not in row:
                    local.install(fitted["values"])
                    located = minimize(
                        lambda z: local.critical_derivatives(row["T_K"], float(z[0]))[
                            0
                        ],
                        [0.5],
                        bounds=[(0.005, 0.995)],
                        method="L-BFGS-B",
                    )
                    row = row.copy()
                    row["x1"] = float(located.x[0])
                    test = [row if item["id"] == row["id"] else item for item in test]
            validation = local.report(fitted["values"], test)
            cv["folds"].append(
                {
                    "fold": index + 1,
                    "success": fitted["success"],
                    "rank": fitted["rank"],
                    "parameter_count": len(local.names),
                    "training_ids": [row["id"] for row in train],
                    "held_out_ids": [row["id"] for row in test],
                    **validation,
                }
            )
        except (ValueError, OverflowError, *_MODEL_ERRORS) as error:
            cv["folds"].append(
                {
                    "fold": index + 1,
                    "success": False,
                    "held_out_ids": [row["id"] for row in test],
                    "error": str(error),
                }
            )
    failed_folds = [
        fold
        for fold in cv["folds"]
        if not fold["success"]
        or any(not point["physical"] for point in fold.get("points", []))
    ]
    if failed_folds:
        warnings.append(
            f"{len(failed_folds)} cross-validation fold(s) failed fitting or held-out phase checks; inspect the validation report."
        )
    deficient_folds = [
        fold
        for fold in cv["folds"]
        if "rank" in fold and fold["rank"] < fold["parameter_count"]
    ]
    if deficient_folds:
        warnings.append(
            f"{len(deficient_folds)} cross-validation fold(s) have rank-deficient training fits."
        )
    problem.install(values)
    parameters = problem.parameters(values)
    result = {
        "success": optimum["success"]
        and all(point["physical"] for point in report["points"]),
        "schema_version": 1,
        "model": request["model"],
        "method": _method(request),
        "components": list(problem.components),
        "component_names": [problem.thermo.props[c].name for c in problem.components],
        "component_cas": [
            problem.thermo.props[c].CAS or None for c in problem.components
        ],
        "parameters": parameters,
        "vapor_parameters": problem.vapor_records(values),
        "rq": [
            {"r": problem.thermo.r[c], "q": problem.thermo.q[c]}
            for c in problem.components
        ]
        if request["model"] == "UNIQUAC"
        else None,
        "coefficients": dict(
            zip(problem.names, (values * np.asarray(problem.scales)).tolist())
        ),
        "optimizer": optimum,
        "warnings": warnings,
        "property_provenance": problem.property_records,
        "validation_only": validation_only,
        **report,
        "cross_validation": cv,
        "request": request,
        "definition_pfd": problem.definition.to_pfd(),
    }
    exported = export_fit(result)
    result.update(exported)
    all_points = {
        point["id"]: point for point in report["points"] + validation_only["points"]
    }
    result["points"] = [all_points[row["id"]] for row in request["observations"]]
    result["plots"] = build_objective_plots(
        problem, values, request["observations"], result["points"], progress
    )
    return result


def _psat_physical_basis(correlation):
    return {key:value for key,value in correlation.items() if key not in ("source","quality_note")} if correlation is not None else None


def export_fit(result, *, pfd_text=None, scope="global", component_map=None):
    """Export entries or apply them to an existing PFD without losing its model."""
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ValueError("Provide a PFDSim fit result with schema_version=1.")
    pfd = parse_pfd(pfd_text) if pfd_text else parse_pfd(result["definition_pfd"])
    selected = pfd.get_thermo_scope(scope) if scope != "global" else None
    if scope != "global" and selected is None:
        raise ValueError(f"Thermodynamic scope {scope!r} does not exist.")
    fitted_definitions = parse_pfd(result["definition_pfd"])
    mapping = {}
    for fitted in fitted_definitions.components:
        target_id = (component_map or {}).get(fitted.symbol)
        target = (
            pfd.get_component(target_id)
            if target_id
            else next(
                (
                    c
                    for c in pfd.components
                    if c.symbol == fitted.symbol and c.name == fitted.name
                ),
                None,
            )
        )
        if target is None and not target_id:
            target = next(
                (
                    c
                    for c in pfd.components
                    if c.name.casefold() == fitted.name.casefold()
                ),
                None,
            )
        if target is None:
            raise ValueError(
                f"Map fitted component {fitted.name!r} to an existing PFD component."
            )
        # Verify identity through an existing explicit definition, preventing
        # accidental application to an unrelated component with the same symbol.
        if target_id and target.name.casefold() != fitted.name.casefold():
            from .factory import create_thermodynamics

            identity = create_thermodynamics([target.name, fitted.name], "IDEAL")
            first, second = identity.props[target.name], identity.props[fitted.name]
            if not first.CAS or first.CAS != second.CAS:
                raise ValueError(f"{target.symbol} does not identify {fitted.name}.")
        fitted_index = result["components"].index(fitted.symbol)
        for name, value in (
            result["request"]
            .get("component_properties", [{}, {}])[fitted_index]
            .items()
        ):
            setattr(target, COMPONENT_FIELDS[name], value)
        if result["request"].get("psat"):
            apply_psat(target, result["request"]["psat"][fitted_index])
        for record in result.get("property_provenance", []):
            field = record.get("property")
            if (
                record.get("component") == fitted.symbol
                and field in set(COMPONENT_FIELDS.values())
                and getattr(fitted, field, None) is not None
            ):
                setattr(target, field, getattr(fitted, field))
        if any(
            row["kind"] in VAPOR_OBJECTIVES for row in result["request"]["observations"]
        ):
            physical_fields = (
                "molecular_weight",
                "Tc",
                "Pc",
                "Vc",
                "Zc",
                "omega",
                "Tb",
                "Hvap",
                "antoine_A",
                "antoine_B",
                "antoine_C",
                "antoine_Tmin",
                "antoine_Tmax",
                "rho",
                "rho_T",
                "dipole_moment",
                "radius_of_gyration",
                "modified_radius_of_gyration",
                "hoc_eta",
                "critical_properties_unavailable",
                "henry_Hcp",
                "henry_B",
                "henry_Tmin",
                "henry_Tmax",
                "henry_Vinf",
            )
            different = [
                name
                for name in physical_fields
                if getattr(target, name) != getattr(fitted, name)
            ]
            if _psat_physical_basis(target.property_correlations.get("Psat")) != _psat_physical_basis(fitted.property_correlations.get("Psat")):
                different.append("Psat property correlation")
            if (
                pfd.metadata.psat_minimum_pressure_bar
                != fitted_definitions.metadata.psat_minimum_pressure_bar
            ):
                different.append("psat_minimum_pressure_bar")
            if different:
                raise ValueError(
                    f"The destination's physical-property basis differs for {target.symbol}: {', '.join(different)}. Fit using this PFD's definitions or reconcile them before applying these VLE parameters."
                )
        mapping[fitted.symbol] = target.symbol
    if len(set(mapping.values())) != 2:
        raise ValueError(
            "Export component mapping must identify two distinct components."
        )
    if scope == "global":
        pfd.metadata.thermo_method = result["method"]
        pfd.metadata.thermo_options = fitted_definitions.metadata.thermo_options
    else:
        if fitted_definitions.metadata.thermo_options:
            # Scoped BV correlation travels through its existing method alias.
            if result["request"]["vapor"] not in ("HOC", "TSONOPOULOS"):
                raise ValueError(
                    "This second-virial correlation is supported only in the global PFD scope."
                )
            selected.method = result["model"] + (
                "-HOC" if result["request"]["vapor"] == "HOC" else "-BV"
            )
        else:
            selected.method = result["method"]
    c1, c2 = [mapping[c] for c in result["components"]]
    for i, symbol in enumerate((c1, c2)):
        target, fitted = (
            pfd.get_component(symbol),
            fitted_definitions.get_component(result["components"][i]),
        )
        if result["rq"]:
            target.uniquac_r, target.uniquac_q = (
                result["rq"][i]["r"],
                result["rq"][i]["q"],
            )
        if result["request"]["vapor"] == "VDM":
            target.vapor_dimerization = deepcopy(fitted.vapor_dimerization)
    parameters = {key.lower(): value for key, value in result["parameters"].items()}
    entries = [
        InteractionParameter(
            c1,
            c2,
            result["model"],
            scope=None if scope == "global" else scope,
            parameters=parameters,
        )
    ]
    # Preserve fixed imported vapor corrections as well as fitted corrections.
    vapor_records = [
        {"model": item.model, **item.parameters}
        for item in fitted_definitions.interaction_parameters
        if item.model != result["model"]
    ]
    vapor_records += result.get("vapor_parameters", [])
    for record in vapor_records:
        params = {
            key.lower(): value
            for key, value in record.items()
            if key not in ("model", "component1", "component2")
        }
        entries = [entry for entry in entries if entry.model != record["model"]]
        entries.append(
            InteractionParameter(
                c1,
                c2,
                record["model"],
                scope=None if scope == "global" else scope,
                parameters=params,
            )
        )
    models = {entry.model for entry in entries}
    pfd.interaction_parameters = [
        item
        for item in pfd.interaction_parameters
        if not (
            {item.component1, item.component2} == {c1, c2}
            and item.model in models
            and (item.scope or "global") == scope
        )
    ] + entries
    # Existing estimation rules remain; explicit interaction overrides take
    # precedence without changing estimation for other pairs.
    text = pfd.to_pfd()
    parse_pfd(text)  # Export must satisfy the actual parser contract.
    entry = "INTERACTION_PARAMETERS:\n" + "\n".join(item.to_pfd() for item in entries)
    return {"entry": entry, "pfd_text": text, "pfd": pfd.to_dict()}


def validate_runtime_inclusion(result):
    """Check that publishing coefficients preserves the fitted liquid model.

    Psat and vapor definitions describe how measurements were reduced, not the
    activity law being published. Audit measurements on their original basis and
    verify activities and excess enthalpy independently of the shared vapor basis.
    """
    if result.get("success") is not True:
        raise ValueError("Only a converged fit passing phase checks can be published.")
    original = prepare_fit(result["request"])
    if result.get("model") != original.request["model"] or result.get(
        "method"
    ) != _method(original.request):
        raise ValueError(
            "The saved model/vapor treatment disagrees with the fitting request."
        )
    cas = [original.thermo.props[c].CAS or None for c in original.components]
    if any(not identity for identity in cas) or cas != result.get("component_cas"):
        raise ValueError(
            "Shared runtime fits require verified CAS identities; custom components remain available through PFD export."
        )
    values = np.array(
        [
            result["coefficients"][name] / original.scales[i]
            for i, name in enumerate(original.names)
        ]
    )
    original.install(values)
    expected_parameters = original.parameters(values)
    if result["parameters"].keys() != expected_parameters.keys():
        raise ValueError(
            "The saved request and exported runtime parameter fields disagree; re-export the fit report before publication."
        )
    for key, expected in expected_parameters.items():
        actual = result["parameters"].get(key)
        if isinstance(expected, bool):
            if actual is not expected:
                raise ValueError(
                    "The saved model variant and exported runtime parameters disagree."
                )
        elif isinstance(expected, (int, float)):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
            ):
                raise ValueError(
                    "The saved coefficients and exported runtime parameters disagree; re-export the fit report before publication."
                )
        elif actual != expected:
            raise ValueError(
                "The saved model variant and exported runtime parameters disagree."
            )
    canonical_request = deepcopy(original.request)
    canonical_request["components"] = cas
    for key in (
        "pfd_text",
        "scope",
        "psat",
        "component_properties",
        "rq",
        "vapor_parameters",
        "initial",
        "bounds",
        "weights",
        "import_options",
    ):
        canonical_request.pop(key, None)
    temperatures = sorted({row["T_K"] for row in original.request["observations"]})
    # Reuse the liquid-only initialization path. These rows configure temperature
    # coverage and retain the original availability of caloric constraints; no
    # regression or measurement audit uses their placeholder values. In
    # particular, do not qualify shared Psat at these temperatures:
    # a liquid mixture can exist below a pure component's freezing point.
    canonical_request["vapor"] = "IDEAL"
    canonical_request["observations"] = [
        {"kind": "GAMMA_INF", "T_K": T, "gamma1_inf": 1.0}
        for T in temperatures
    ]
    if any(row["kind"] == "HE" and not row["validation_only"] for row in original.request["observations"]):
        canonical_request["observations"].append(
            {"kind": "HE", "T_K": temperatures[0], "x1": 0.5, "HE_J_mol": 0.0}
        )
    standard = prepare_fit(canonical_request)
    if original.model == "UNIQUAC":
        for i in range(2):
            if not math.isclose(
                original.thermo.r[original.components[i]],
                standard.thermo.r[standard.components[i]],
                rel_tol=1e-9,
            ) or not math.isclose(
                original.thermo.q[original.components[i]],
                standard.thermo.q[standard.components[i]],
                rel_tol=1e-9,
            ):
                raise ValueError(
                    "This fit uses custom UNIQUAC R/Q that differs from the shared structural basis. Keep it as a PFD fit or curate the structural parameters before runtime publication."
                )
    params = {
        "model": result["model"],
        "component1": standard.components[0],
        "component2": standard.components[1],
        **result["parameters"],
    }
    runtime = create_thermodynamics(
        list(standard.components),
        result["model"],
        db=standard.thermo.db,
        interaction_overrides=[params],
    )
    runtime._resolver_known_props = deepcopy(standard.thermo._resolver_known_props)
    standard.thermo = runtime
    for T in temperatures:
        for x in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
            fitted_gamma = original.checked_gamma(T, x)
            runtime_gamma = standard.checked_gamma(T, x)
            if not np.allclose(
                [math.log(fitted_gamma[c]) for c in original.components],
                [math.log(runtime_gamma[c]) for c in standard.components],
                rtol=0,
                atol=1e-6,
            ) or not math.isclose(
                original.thermo.excess_enthalpy(original.composition(x), T),
                runtime.excess_enthalpy(standard.composition(x), T),
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    "The published liquid activity model differs from the fitted activities or excess enthalpy. Use PFD export or reconcile the liquid model before publication."
                )
    audit = original.report(values, original.request["observations"])
    if any(
        not point["physical"] or point["pin_satisfied"] is False
        for point in audit["points"]
        if point["role"] == "training"
    ):
        raise ValueError(
            "The submission fails recomputed physical checks or hard-pin constraints."
        )
    return {"component_cas": cas, "property_basis": "shared_liquid_activity_verified"}


def fitting_catalog():
    return {
        "kinds": list(KINDS),
        "forms": {name: list(terms) for name, terms in FORMS.items()},
        "vapors": list(VAPORS),
        "scales": SCALES,
        "import_fields": FIELD_LABELS,
        "missing_tokens": sorted(MISSING_TOKENS),
        "psat_forms": PSAT_FORMS,
        "vapor_requirements": VAPOR_REQUIREMENTS,
        "example": [
            {"kind": "VLE", "T_K": 350, "P_bar": 1, "x1": 0.3, "y1": 0.6},
            {"kind": "HE", "T_K": 300, "x1": 0.5, "HE_J_mol": 500},
            {"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 3},
        ],
    }
