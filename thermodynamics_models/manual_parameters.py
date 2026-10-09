"""Direct, sourced activity parameters using the runtime's own model and export."""

from copy import deepcopy
import math

from .interaction_fitting import _number, _MODEL_ERRORS, export_fit, Simulator, ProcessFlowDiagram, Component, InteractionParameter
from .activity import ActivityCoefficientThermodynamics


def normalize_manual_parameters(data):
    if not isinstance(data, dict) or data.keys() - {
        "components", "model", "basis", "unit", "values", "alpha", "T_ref_K",
        "Tmin_K", "Tmax_K", "extrapolation",
        "fit_method", "statistics",
    }:
        raise ValueError("Provide components, model, parameter basis and values.")
    components = data.get("components")
    if (not isinstance(components, list) or len(components) != 2
            or any(not isinstance(c, str) or not c.strip() for c in components)):
        raise ValueError("Choose two distinct component identifiers.")
    components = [c.strip() for c in components]
    if components[0].casefold() == components[1].casefold():
        raise ValueError("Choose two distinct components.")
    model, basis = data.get("model"), data.get("basis")
    if model not in ("NRTL", "UNIQUAC") or basis not in ("energy", "tau", "law"):
        raise ValueError("Choose NRTL/UNIQUAC and energy, tau or temperature-law coefficients.")
    fields = list("cdefg" if model == "NRTL" else "abcde")
    keys = {f"{direction}.{field}" for direction in ("12", "21") for field in fields} if basis == "law" else {"12", "21"}
    values = data.get("values")
    if not isinstance(values, dict) or values.keys() != keys:
        raise ValueError("Supply both directions and every displayed coefficient (zero is allowed).")
    values = {key: _number(value, key) for key, value in values.items()}
    if basis == "tau" and model == "UNIQUAC" and any(v <= 0 for v in values.values()):
        raise ValueError("UNIQUAC tau values must be positive; temperature-law coefficients describe ln(tau).")
    unit = data.get("unit", "J/mol")
    if unit not in ("J/mol", "kJ/mol", "cal/mol", "kcal/mol"):
        raise ValueError("Choose J/mol, kJ/mol, cal/mol or kcal/mol.")
    low, high = data.get("Tmin_K"), data.get("Tmax_K")
    if (low is None) != (high is None):
        raise ValueError("Supply both temperature limits or leave both blank.")
    if low is not None:
        low, high = _number(low, "Tmin_K", low=1), _number(high, "Tmax_K", low=1)
        if low > high:
            raise ValueError("Tmin_K must not exceed Tmax_K.")
    extrapolation = data.get("extrapolation", "unrestricted")
    if extrapolation not in ActivityCoefficientThermodynamics.ACTIVITY_EXTRAPOLATION_MODES:
        raise ValueError("Choose a supported activity-parameter extrapolation policy.")
    if extrapolation != "unrestricted" and low is None:
        raise ValueError("A bounded extrapolation policy needs temperature limits.")
    reported = {}
    for field in ("fit_method", "statistics"):
        value = data.get(field, "")
        if not isinstance(value, str) or len(value) > 10000:
            raise ValueError("Reported fitting method and statistics must be text of at most 10000 characters.")
        reported[field] = value.strip()
    return {
        "components": components, "model": model, "basis": basis, "unit": unit,
        "values": values, "alpha": _number(data.get("alpha", .3), "alpha", low=0),
        "T_ref_K": _number(data.get("T_ref_K", 298.15), "T_ref_K", low=1),
        "Tmin_K": low, "Tmax_K": high, "extrapolation": extrapolation,
        **reported,
    }


def prepare_manual_parameters(data):
    """Resolve and install exactly the submitted law, without regression data."""
    data = normalize_manual_parameters(data)
    model, basis = data["model"], data["basis"]
    parameters = {"extrapolation": data["extrapolation"]}
    if data["Tmin_K"] is not None:
        parameters.update(Tmin_K=data["Tmin_K"], Tmax_K=data["Tmax_K"])
    if model == "NRTL":
        parameters["alpha12"] = data["alpha"]
    if basis == "energy":
        factor = {"J/mol": 1 / 4.184, "kJ/mol": 1000 / 4.184, "cal/mol": 1, "kcal/mol": 1000}[data["unit"]]
        parameters.update({f"a{direction}_cal_per_mol": value * factor for direction, value in data["values"].items()})
    else:
        fields = "cdefg" if model == "NRTL" else "abcde"
        for direction in ("12", "21"):
            for field in fields:
                value = data["values"].get(f"{direction}.{field}", 0.0)
                if basis == "tau" and field == fields[0]:
                    value = data["values"][direction]
                    if model == "UNIQUAC":
                        value = math.log(value)
                parameters[f"tau{direction}_{field}"] = value
        parameters["tau_tref"] = data["T_ref_K"]
    if model == "UNIQUAC":
        parameters.update(model_variant="standard_uniquac", use_q_prime=False)
    if any(not math.isfinite(v) for v in parameters.values() if isinstance(v, (int, float))):
        raise ValueError("Converted parameter values must be finite.")
    pfd = ProcessFlowDiagram()
    pfd.metadata.thermo_method = model
    pfd.metadata.online_lookup = False
    pfd.components = [Component(symbol=f"Manual_{i + 1}", name=name) for i, name in enumerate(data["components"])]
    symbols = [c.symbol for c in pfd.components]
    pfd.interaction_parameters = [InteractionParameter(*symbols, model, parameters=parameters)]
    try:
        sim = Simulator.from_string(pfd.to_pfd())
        sim.initialize(property_methods=[])
    except _MODEL_ERRORS as error:
        raise ValueError(f"Cannot initialize the parameter model: {error}") from error
    thermo = sim.thermo_packages["global"]
    cas = [thermo.props[c].CAS for c in symbols]
    if not all(cas) or len(set(cas)) != 2:
        raise ValueError("Shared parameters require two distinct resolved CAS identities.")
    result = {
        "schema_version": 1, "submission_origin": "manual_parameters",
        "success": True, "manual_input": data, "model": model, "method": model,
        "components": symbols, "component_cas": cas,
        "component_names": [thermo.props[c].name for c in symbols],
        "parameters": parameters, "rq": None, "objectives": {},
        "reported_fit": {"method": data["fit_method"], "statistics": data["statistics"]},
        "request": {"components": data["components"], "model": model, "vapor": "IDEAL", "observations": [], "weights": {}},
        "definition_pfd": pfd.to_pfd(),
        "warnings": list(thermo.warnings),
    }
    result.update(export_fit(result))
    return result, thermo


def validate_manual_inclusion(result):
    canonical, thermo = prepare_manual_parameters(result.get("manual_input"))
    for key in ("schema_version", "model", "method", "components", "component_cas", "component_names", "rq", "parameters", "request", "definition_pfd", "reported_fit"):
        if result.get(key) != canonical[key]:
            raise ValueError("The saved manual input and runtime parameters disagree.")
    data = canonical["manual_input"]
    temperatures = sorted(set([data["T_ref_K"]] if data["Tmin_K"] is None else [data["Tmin_K"], (data["Tmin_K"] + data["Tmax_K"]) / 2, data["Tmax_K"]]))
    for T in temperatures:
        for x in (0, .05, .25, .5, .75, .95, 1):
            try:
                gamma = thermo.activity_coefficients(T, dict(zip(canonical["components"], (x, 1 - x))))
            except (OverflowError, *_MODEL_ERRORS) as error:
                raise ValueError(f"The parameter law cannot produce valid activities: {error}") from error
            if any(not math.isfinite(v) or v <= 0 for v in gamma.values()):
                raise ValueError("The parameter law must produce finite positive activities.")
    return {"component_cas": canonical["component_cas"], "property_basis": "shared_liquid_activity_verified", "warnings": canonical["warnings"], "origin": "manual_parameters"}


def preview_manual_parameters(payload, progress):
    result, thermo = prepare_manual_parameters(payload["parameters"])
    c1, c2 = result["components"]
    T, count, kind = payload["T_K"], payload["n_points"], payload["kind"]
    progress(f"Previewing {kind} at {T:g} K")
    x = [i / count for i in range(count + 1)]
    errors, series = [], []
    def curve(name, coordinates, values):
        return {"name": name, "x": coordinates, "y": values, "mode": "line"}
    if kind == "VLE":
        chart = thermo.generate_Pxy_data(c1, c2, T, count)
        series = [curve("Bubble · liquid x₁", chart["x"], chart["P_bubble"]), curve("Dew · vapor y₁", chart["y"], chart["P_dew"])]
        errors = chart["errors"]
        y_label = "Pressure (bar)"
        note = "P–x–y at fixed temperature with an ideal vapor. Liquid stability is shown separately by LLE preview."
    else:
        first, second = [], []
        for coordinate in x:
            composition = {c1: coordinate, c2: 1 - coordinate}
            try:
                if kind == "GAMMA":
                    gamma = thermo.activity_coefficients(T, composition)
                    a, b = gamma[c1], gamma[c2]
                    if any(not math.isfinite(v) or v <= 0 for v in (a, b)):
                        raise ValueError("Nonfinite or nonpositive activity coefficient.")
                else:
                    split, liquid1, liquid2, beta = thermo.liquid_liquid_equilibrium(composition, T)
                    if split:
                        thermo._check_diagram_split(composition, [(1 - beta, liquid1), (beta, liquid2)], T)
                        a, b = sorted((liquid1[c1], liquid2[c1]))
                    else:
                        a = b = coordinate
                first.append(a)
                second.append(b)
            except (ValueError, RuntimeError, OverflowError, *_MODEL_ERRORS) as error:
                first.append(None)
                second.append(None)
                errors.append({"x": coordinate, "error": str(error)})
        series = [curve("γ1" if kind == "GAMMA" else "Liquid α", x, first), curve("γ2" if kind == "GAMMA" else "Liquid β", x, second)]
        y_label = "Activity coefficient γ" if kind == "GAMMA" else "Equilibrium liquid x₁"
        note = "Homogeneous liquid activity coefficients, including metastable compositions." if kind == "GAMMA" else "Separated branches indicate two liquid phases; coincident diagonal branches indicate one liquid phase."
    result["warnings"] = list(thermo.warnings)
    return {"success": True, "result": deepcopy(result), "note": note, "plots": [{"kind": kind, "title": f"{kind} · {T:g} K · {' / '.join(result['component_names'])}", "x_label": "Mole fraction of component 1", "y_label": y_label, "series": series, "errors": errors}]}
