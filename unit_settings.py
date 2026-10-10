"""Discover editor controls from registered unit implementations and DOF metadata.

This is presentation metadata, not another specification validator. Runtime
get_param calls supply names, aliases, and literal defaults; existing rules add
descriptions, units, choices and which settings belong in the primary panel.
"""

import ast
from functools import lru_cache
import inspect
import re
import textwrap

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from .unit_operations import UNIT_CLASSES
    from .dof_analyzer import get_unit_info
else:
    from unit_operations import UNIT_CLASSES
    from dof_analyzer import get_unit_info

LABELS = {
    "newton_globalization": "Newton step strategy",
    "n_stages": "Number of stages",
    "feed_stage": "Feed stage",
    "feed_stages": "Feed stage assignments",
    "reflux_ratio": "Reflux ratio (L/D)",
    "rr": "Reflux ratio (L/D)",
    "d_to_f": "Distillate / feed ratio",
    "d_rate": "Distillate molar flow",
    "d_mass": "Distillate mass flow",
    "p_condenser": "Condenser pressure",
    "p_top": "Top pressure",
    "p_bottom": "Bottom pressure",
    "p": "Operating pressure",
    "p_out": "Outlet pressure",
    "t": "Operating temperature",
    "t_out": "Outlet temperature",
    "q": "Heat duty",
    "ua": "Heat transfer capacity (UA)",
    "u": "Heat transfer coefficient",
    "a": "Heat transfer area",
    "v": "Reactor volume",
    "vf": "Vapor fraction",
    "vap_frac": "Vapor fraction",
    "eta": "Hydraulic efficiency",
    "eta_isen": "Isentropic efficiency",
    "eta_mech": "Mechanical efficiency",
    "v_batch": "Volume per batch vessel",
    "n": "Number of batch vessels",
    "t_rxn": "Reaction time",
    "capture_cut_size": "Particle capture cut size",
    "p_drop": "Pressure drop",
    "delta_p": "Pressure change",
}
UNITS = {
    "t": "C",
    "t_out": "C",
    "p": "bar",
    "p_out": "bar",
    "p_top": "bar",
    "p_bottom": "bar",
    "p_condenser": "bar",
    "q": "kW",
    "volume": "m3",
    "v": "m3",
    "length": "m",
    "diameter": "m",
    "roughness": "m",
    "velocity": "m/s",
    "d_rate": "kmol/h",
    "d_mass": "kg/h",
    "t_hot_out": "C",
    "t_cold_out": "C",
    "t_tube_out": "C",
    "t_shell_out": "C",
    "t_min": "K",
    "t_max": "K",
    "t_guess": "K",
    "p_drop": "bar",
    "delta_p": "bar",
    "p_drop_per_stage": "bar",
}


def _call_name(node):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"get_param", "get_temperature_param", "get_param_unit"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def unit_supports_reactions(unit_type):
    """Expose the registered runtime model's reaction capability."""
    return UNIT_CLASSES[unit_type].supports_reactions


@lru_cache(maxsize=None)
def unit_setting_schema(unit_type):
    cls = UNIT_CLASSES[unit_type]
    info = get_unit_info(unit_type)
    schema, aliases = {}, {}
    pending = [cls.solve, cls.__init__]
    if unit_type in ('Heater', 'Cooler'):
        exchanger = UNIT_CLASSES['HeatExchanger']
        pending.extend((exchanger._calculated_u_spec, exchanger._shell_transport_geometry,
                        exchanger._curve_segments, exchanger._flow_pattern,
                        exchanger._ua_spec))
    seen = set()
    while pending:
        function = pending.pop()
        if function in seen:
            continue
        seen.add(function)
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        except (OSError, TypeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                candidate = None
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                ):
                    candidate = getattr(cls, node.func.attr, None)
                elif (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Call)
                    and isinstance(node.func.value.func, ast.Name)
                    and node.func.value.func.id == "super"
                    and not node.func.value.args
                ):
                    # Overrides such as RigorousStripper.solve delegate their
                    # numerical settings to a parent implementation.
                    owner = next((base for base in cls.__mro__
                                  if function.__name__ in base.__dict__
                                  and inspect.unwrap(base.__dict__[function.__name__])
                                  is inspect.unwrap(function)), None)
                    if owner is not None:
                        parents = cls.__mro__[cls.__mro__.index(owner) + 1:]
                        candidate = next((getattr(base, node.func.attr) for base in parents
                                          if inspect.isfunction(getattr(base, node.func.attr, None))), None)
                elif isinstance(node.func, ast.Name):
                    candidate = getattr(function, "__globals__", {}).get(node.func.id)
                if inspect.isfunction(candidate) and candidate not in seen:
                    pending.append(candidate)
            name = _call_name(node)
            if (
                name is None
                or name.startswith("_")
                or node.func.attr == "get_param_unit"
            ):
                continue
            item = schema.setdefault(name.lower(), {"name": name, "default": None})
            if node.func.attr == "get_temperature_param":
                item["unit"] = "C"
            if len(node.args) > 1:
                fallback = node.args[1]
                nested = _call_name(fallback)
                if nested:
                    aliases[nested.lower()] = name.lower()
                while _call_name(fallback) and len(fallback.args) > 1:
                    fallback = fallback.args[1]
                try:
                    item["default"] = ast.literal_eval(fallback)
                except (ValueError, TypeError):
                    pass
    declared_primary = {name.lower() for name in info.get("primary_specs", [])}
    metadata_specs = dict(info.get('optional_specs', {}))
    if unit_type in ('Heater', 'Cooler'):
        exchanger_specs = get_unit_info('HeatExchanger')['optional_specs']
        for name in ('U_model', 'wall_material', 'tube_side', 'tube_inner_diameter',
                     'tube_outer_diameter', 'shell_inner_diameter', 'wall_conductivity',
                     'length', 'orientation', 'bundle_diameter', 'tube_pitch', 'tube_count',
                     'tube_passes', 'shell_passes', 'tube_layout_angle', 'baffle_spacing',
                     'baffle_cut', 'baffle_count', 'tube_baffle_clearance',
                     'shell_baffle_clearance', 'sealing_strip_pairs', 'boiling_Csf',
                     'boiling_n', 'shell_pool_boiling', 'fouling_tube', 'fouling_shell',
                     'type', 'flow_pattern', 'curve_segments', 'A', 'UA', 'UA_available',
                     'estimate_U'):
            metadata_specs[name] = exchanger_specs[name]
    for name, metadata in metadata_specs.items():
        if name.lower() not in schema and name.lower() not in declared_primary:
            continue
        item = schema.setdefault(name.lower(), {"name": name, "default": None})
        item.update(metadata)
    primary = {name.lower() for name in info.get("primary_specs", [])}
    primary.update({"t", "p", "t_out", "p_out", "q", "mode", "phase", "thermo_scope"})
    primary.update(
        part.lower()
        for spec in info.get("required_specs", [])
        for part in re.split(r"[|,+]", spec)
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", part)
    )
    for name in info.get("primary_specs", []):
        schema.setdefault(name.lower(), {"name": name, "default": None})
    for key, item in schema.items():
        target = key
        visited = set()
        while target in aliases and target not in visited:
            visited.add(target)
            target = aliases[target]
        item["canonical"] = target
        item["label"] = (
            LABELS.get(target)
            or item.get("description")
            or item["name"].replace("_", " ").capitalize()
        )
        item["unit"] = item.get("unit") or UNITS.get(target)
        if target == "phase":
            item["values"] = ["vapor", "liquid"]
        default = item.get("default")
        item["type"] = (
            "boolean"
            if isinstance(default, bool)
            else "number"
            if isinstance(default, (float, int))
            else "array"
            if isinstance(default, (list, tuple))
            else "object"
            if isinstance(default, dict)
            else "text"
        )
        if target in primary or key in primary:
            item["section"] = "operation"
        elif any(
            word in key
            for word in (
                "solver",
                "jacobian",
                "tolerance",
                "max_iter",
                "max_eval",
                "initializer",
                "line_search",
                "damping",
                "acceleration",
                "mesh_",
                "residual",
                "newton",
                "step_limit",
            )
        ):
            item["section"] = "solver"
        else:
            item["section"] = "equipment"
    return list(schema.values())
