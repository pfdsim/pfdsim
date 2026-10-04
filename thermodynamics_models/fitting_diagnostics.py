"""Physical errors and objective curves from the authoritative fitted package."""

import math
import numpy as np


def physical_metrics(points):
    groups = {}
    for point in points:
        kind, observed, predicted = point["kind"], point["observed"], point["predicted"]
        quantities = {
            "HE": {"HE_J_mol": "J/mol"},
            "GAMMA_INF": {"gamma1_inf": "dimensionless", "gamma2_inf": "dimensionless"},
            "LLE": {"x1_alpha": "mole fraction", "x1_beta": "mole fraction"},
            "VLLE": {
                "T_K": "K",
                "P_bar": "bar",
                "y1": "mole fraction",
                "x1_alpha": "mole fraction",
                "x1_beta": "mole fraction",
            },
            "VLE": {"T_K": "K", "P_bar": "bar", "y1": "mole fraction"},
            "AZEOTROPE": {"T_K": "K", "P_bar": "bar", "y1": "mole fraction"},
            "UCST": {"T_K": "K", "x1": "mole fraction"},
            "LCST": {"T_K": "K", "x1": "mole fraction"},
        }[kind]
        for quantity, unit in quantities.items():
            actual = observed.get(
                quantity,
                observed.get("x1")
                if kind == "AZEOTROPE" and quantity == "y1"
                else None,
            )
            calculated = predicted.get(quantity)
            record = groups.setdefault(kind, {}).setdefault(
                quantity, {"unit": unit, "errors": [], "unavailable": 0}
            )
            if actual is None:
                continue
            if calculated is None or not math.isfinite(float(calculated)):
                record["unavailable"] += 1
                continue
            record["errors"].append(float(calculated - actual))
            if kind == "GAMMA_INF":
                relative = groups[kind].setdefault(
                    quantity + "_relative_percent",
                    {"unit": "%", "errors": [], "unavailable": 0},
                )
                relative["errors"].append(100 * (calculated / actual - 1))
            if kind in ("LLE", "VLLE") and quantity in ("x1_alpha", "x1_beta"):
                complementary = groups[kind].setdefault(
                    quantity.replace("x1", "x2"),
                    {"unit": "mole fraction", "errors": [], "unavailable": 0},
                )
                complementary["errors"].append(float(actual - calculated))
    for quantities in groups.values():
        for record in quantities.values():
            values = np.asarray(record.pop("errors"))
            record.update(
                n=len(values),
                MAE=float(np.mean(abs(values))) if len(values) else None,
                RMSE=float(np.sqrt(np.mean(values**2))) if len(values) else None,
                bias=float(np.mean(values)) if len(values) else None,
                max_abs=float(max(abs(values))) if len(values) else None,
            )
    return groups


def _observations(rows, x, y, name):
    result = []
    for validation in (False, True):
        selected = [
            row
            for row in rows
            if row.get("validation_only", False) == validation
            and x(row) is not None
            and y(row) is not None
        ]
        if selected:
            result.append(
                {
                    "name": name
                    + (" · validation-only" if validation else " · training"),
                    "mode": "markers",
                    "role": "validation" if validation else "training",
                    "x": [x(row) for row in selected],
                    "y": [y(row) for row in selected],
                }
            )
    return result


def build_objective_plots(problem, values, rows, points, progress=None):
    """Sample actual model curves; unavailable states stay explicit gaps."""
    problem.install(values)
    plots = []
    count = 21

    def sample(coordinates, evaluate, coordinate_name):
        predictions, errors = [], []
        for coordinate in coordinates:
            try:
                value = float(evaluate(float(coordinate)))
                if not math.isfinite(value):
                    raise ValueError("The model prediction is not finite.")
                predictions.append(value)
            except Exception as error:
                predictions.append(None)
                errors.append({coordinate_name: float(coordinate), "error": str(error)})
        return predictions, errors

    for kind in sorted({row["kind"] for row in rows}):
        selected = [row for row in rows if row["kind"] == kind]
        if progress:
            progress(f"Computing {kind} predicted/data curves")
        if kind in ("VLE", "AZEOTROPE"):
            groups = []
            for field in ("P_bar", "T_K"):
                for fixed in sorted({row[field] for row in selected}):
                    data = [
                        row
                        for row in selected
                        if math.isclose(row[field], fixed, rel_tol=1e-7)
                    ]
                    if len(data) >= 2:
                        groups.append((field, fixed, data))
            for field, fixed, data in groups:
                isobaric = field == "P_bar"
                x = np.linspace(
                    min(row["x1"] for row in data),
                    max(row["x1"] for row in data),
                    count,
                )
                model_y, dependent, errors = [], [], []
                guess = float(np.mean([row["T_K"] for row in data]))
                for composition in x:
                    try:
                        state = (
                            problem.predict_vle(
                                float(composition), P=fixed, T_guess=guess
                            )
                            if isobaric
                            else problem.predict_vle(float(composition), T=fixed)
                        )
                        model_y.append(state["y1"])
                        dependent.append(
                            state["T_K"] - 273.15 if isobaric else state["P_bar"]
                        )
                        guess = state["T_K"]
                    except Exception as error:
                        model_y.append(None)
                        dependent.append(None)
                        errors.append({"x1": float(composition), "error": str(error)})
                axis = "T / °C" if isobaric else "P / bar"
                data_y = (
                    (lambda row: row["T_K"] - 273.15)
                    if isobaric
                    else (lambda row: row["P_bar"])
                )
                series = [
                    {
                        "name": "Predicted liquid curve",
                        "mode": "line",
                        "x": x.tolist(),
                        "y": dependent,
                    },
                    {
                        "name": "Predicted vapor curve",
                        "mode": "line",
                        "x": model_y,
                        "y": dependent,
                    },
                ]
                series += _observations(
                    data, lambda row: row["x1"], data_y, "Liquid data"
                )
                series += _observations(
                    data,
                    lambda row: row.get(
                        "y1", row["x1"] if kind == "AZEOTROPE" else None
                    ),
                    data_y,
                    "Vapor data",
                )
                plots.append(
                    {
                        "kind": kind,
                        "title": f"{kind} · {fixed:g} {'bar' if isobaric else 'K'}",
                        "x_label": "Component 1 mole fraction",
                        "y_label": axis,
                        "series": series,
                        "errors": errors,
                    }
                )
        elif kind == "HE":
            for temperature in sorted({row["T_K"] for row in selected}):
                data = [row for row in selected if row["T_K"] == temperature]
                x = np.linspace(
                    min(row["x1"] for row in data),
                    max(row["x1"] for row in data),
                    count,
                )
                def enthalpy(z):
                    delta = max(1e-3, temperature * 1e-4)
                    for T in (max(1, temperature - delta), temperature, temperature + delta):
                        problem.checked_gamma(T, z)
                    return problem.thermo.excess_enthalpy(problem.composition(z), temperature)

                y, errors = sample(x, enthalpy, "x1")
                plots.append(
                    {
                        "kind": kind,
                        "title": f"Hᴱ · {temperature:g} K",
                        "x_label": "Component 1 mole fraction",
                        "y_label": "Hᴱ / J mol⁻¹",
                        "series": [
                            {
                                "name": "Predicted Hᴱ",
                                "mode": "line",
                                "x": x.tolist(),
                                "y": y,
                            }
                        ]
                        + _observations(
                            data,
                            lambda row: row["x1"],
                            lambda row: row["HE_J_mol"],
                            "Calorimetry",
                        ),
                        "errors": errors,
                    }
                )
        elif kind in ("LLE", "VLLE"):
            temperature = np.linspace(
                min(row["T_K"] for row in selected),
                max(row["T_K"] for row in selected),
                count,
            )
            a, b, vapor, pressure, errors = [], [], [], [], []
            for T in temperature:
                try:
                    if kind == "VLLE":
                        state = problem.predict_vlle(T=float(T))
                        a.append(state["x1_alpha"])
                        b.append(state["x1_beta"])
                        vapor.append(state["y1"])
                        pressure.append(state["P_bar"])
                    else:
                        state = problem.predict_lle(float(T))
                        a.append(state["x1_alpha"])
                        b.append(state["x1_beta"])
                except Exception as error:
                    a.append(None)
                    b.append(None)
                    if kind == "VLLE":
                        vapor.append(None)
                        pressure.append(None)
                    errors.append({"T_K": float(T), "error": str(error)})
            plots.append(
                {
                    "kind": kind,
                    "title": f"{kind} liquid branches",
                    "x_label": "Component 1 mole fraction",
                    "y_label": "T / °C",
                    "series": [
                        {
                            "name": "Predicted liquid α",
                            "mode": "line",
                            "x": a,
                            "y": (temperature - 273.15).tolist(),
                        },
                        {
                            "name": "Predicted liquid β",
                            "mode": "line",
                            "x": b,
                            "y": (temperature - 273.15).tolist(),
                        },
                    ]
                    + _observations(
                        selected,
                        lambda row: row.get("x1_alpha"),
                        lambda row: row["T_K"] - 273.15,
                        "Liquid α data",
                    )
                    + _observations(
                        selected,
                        lambda row: row.get("x1_beta"),
                        lambda row: row["T_K"] - 273.15,
                        "Liquid β data",
                    ),
                    "errors": errors,
                }
            )
            if kind == "VLLE":
                for name, y, field, label in (
                    ("Vapor composition", vapor, "y1", "y₁ mole fraction"),
                    ("Pressure", pressure, "P_bar", "P / bar"),
                ):
                    plots.append(
                        {
                            "kind": kind,
                            "title": f"VLLE {name}",
                            "x_label": "T / °C",
                            "y_label": label,
                            "series": [
                                {
                                    "name": "Predicted " + name,
                                    "mode": "line",
                                    "x": (temperature - 273.15).tolist(),
                                    "y": y,
                                }
                            ]
                            + _observations(
                                selected,
                                lambda row: row["T_K"] - 273.15,
                                lambda row: row.get(field),
                                name + " data",
                            ),
                            "errors": errors,
                        }
                    )
        elif kind == "GAMMA_INF":
            T = np.linspace(
                min(row["T_K"] for row in selected),
                max(row["T_K"] for row in selected),
                count,
            )
            series, errors = [], []
            for index, key in enumerate(("gamma1_inf", "gamma2_inf")):
                if not any(key in row for row in selected):
                    continue
                predicted, unavailable = sample(
                    T,
                    lambda t: problem.checked_gamma(t, float(index))[problem.components[index]],
                    "T_K",
                )
                errors.extend({**item, "property": key} for item in unavailable)
                series.append(
                    {
                        "name": f"Predicted γ{index + 1}∞",
                        "mode": "line",
                        "x": T.tolist(),
                        "y": predicted,
                    }
                )
                series += _observations(
                    selected,
                    lambda row: row["T_K"],
                    lambda row: row.get(key),
                    f"γ{index + 1}∞ data",
                )
            plots.append(
                {
                    "kind": kind,
                    "title": "Infinite-dilution activity coefficients",
                    "x_label": "T / K",
                    "y_label": "γ∞",
                    "series": series,
                    "errors": errors,
                }
            )
        # A parity plot remains useful for sparse or non-isothermal/isobaric
        # data, including critical targets whose model roots may be unavailable.
        for quantity, label in (
            ("T_K", "T / K"),
            ("P_bar", "P / bar"),
            ("y1", "y₁"),
            ("x1", "Critical x₁"),
        ):
            pairs = [
                point
                for point in points
                if point["kind"] == kind
                and point["observed"].get(quantity) is not None
                and point["predicted"].get(quantity) is not None
            ]
            if not pairs or quantity == "x1" and kind not in ("UCST", "LCST"):
                continue
            values_x = [point["observed"][quantity] for point in pairs]
            low, high = min(values_x), max(values_x)
            series = [
                {
                    "name": "Perfect agreement",
                    "mode": "line",
                    "x": [low, high],
                    "y": [low, high],
                }
            ]
            for role in ("training", "validation"):
                group = [point for point in pairs if point["role"] == role]
                if group:
                    series.append(
                        {
                            "name": role + " data/model",
                            "mode": "markers",
                            "role": role,
                            "x": [point["observed"][quantity] for point in group],
                            "y": [point["predicted"][quantity] for point in group],
                        }
                    )
            plots.append(
                {
                    "kind": kind,
                    "title": f"{kind} · {label} parity",
                    "x_label": "Observed " + label,
                    "y_label": "Predicted " + label,
                    "series": series,
                    "errors": [],
                }
            )
    return plots
