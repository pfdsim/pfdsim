#!/usr/bin/env python3
"""Benchmark weak ideal-gas Cp fallbacks against the canonical database.

The fitted models are validated out of sample.  By default, every molecular
formula is kept wholly in one fold, so isomers with identical atom-count
vectors cannot be split between training and validation.

This script is read-only: it does not modify the canonical database or any
runtime cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from chemicals.elements import molecular_weight, simple_formula_parser
from numpy.polynomial import chebyshev as ncheb


ROOT = Path(__file__).resolve().parents[3]
DATABASE = ROOT / "data" / "ideal_gas_heat_capacity.sqlite"
R = 8.31446261815324
REFERENCE_TEMPERATURES = (298.15, 500.0, 1000.0)

CATEGORIES = (
    "H",
    "C",
    "O",
    "N",
    "F",
    "Cl",
    "P",
    "S",
    "Si",
    "B",
    "Br",
    "I",
    "light_metal",
    "heavy_metal",
    "metalloid",
    "noble_gas",
)
EXPLICIT_ELEMENTS = frozenset(CATEGORIES[:12])
LEGACY_LIGHT_METALS = frozenset(
    {"Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "Rb", "Sr", "Cs", "Ba", "Fr", "Ra"}
)
METALLOIDS = frozenset({"Ge", "As", "Sb", "Te", "Po"})
NOBLE_GASES = frozenset({"He", "Ne", "Ar", "Kr", "Xe", "Rn"})
ALKALI_METALS = frozenset({"Li", "Na", "K", "Rb", "Cs", "Fr"})
ALKALINE_EARTH_METALS = frozenset({"Be", "Mg", "Ca", "Sr", "Ba", "Ra"})
EARLY_TRANSITION_METALS = frozenset(
    {
        "Sc", "Ti", "V", "Cr", "Mn", "Y", "Zr", "Nb", "Mo", "Tc",
        "Hf", "Ta", "W", "Re", "Rf", "Db", "Sg", "Bh",
        # The database has too little f-block coverage for another identifiable
        # category, so inner-transition elements share the early-transition fit.
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu", "Ac", "Th", "Pa", "U", "Np",
        "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    }
)
LATE_TRANSITION_METALS = frozenset(
    {
        "Fe", "Co", "Ni", "Cu", "Zn", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Os", "Ir", "Pt", "Au", "Hg", "Hs", "Mt", "Ds", "Rg", "Cn",
    }
)
POST_TRANSITION_METALS = frozenset(
    {"Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Nh", "Fl", "Mc", "Lv"}
)
ALL_METALS = (
    ALKALI_METALS
    | ALKALINE_EARTH_METALS
    | EARLY_TRANSITION_METALS
    | LATE_TRANSITION_METALS
    | POST_TRANSITION_METALS
)

# The local chemicals metadata identity for this CAS is unrelated.  NIST
# identifies it as magnesium monohydroxide, HMgO, which is consistent with the
# stored WebBook Cp curve.
FORMULA_OVERRIDES = {"12141-11-6": "HMgO"}
SHOMATE_TERMS = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class Component:
    cas: str
    name: str
    formula: str
    source: str
    molecular_weight: float
    counts: np.ndarray
    Tmin: float
    Tmax: float
    center: float
    scale: float
    coefficients: np.ndarray
    fold: int

    def cp(self, temperatures: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(temperatures, dtype=float)
        mapped = (values - self.center) / (self.scale * (values + self.center))
        return ncheb.chebval(mapped, self.coefficients)


def metal_categories_for(metal_routing: str) -> tuple[str, ...]:
    if metal_routing == "light_heavy":
        return ("light_metal", "heavy_metal")
    if metal_routing == "blocks":
        return ("s_block_metal", "transition_metal", "post_transition_metal")
    if metal_routing == "families":
        return (
            "alkali_metal",
            "alkaline_earth_metal",
            "transition_metal",
            "post_transition_metal",
        )
    if metal_routing == "transition_split":
        return (
            "alkali_metal",
            "alkaline_earth_metal",
            "early_transition_metal",
            "late_transition_metal",
            "post_transition_metal",
        )
    raise ValueError(f"Unsupported metal routing: {metal_routing}")


def categories_for(
    selenium_routing: str, metal_routing: str
) -> tuple[str, ...]:
    selenium = ("Se",) if selenium_routing == "separate" else ()
    return (
        CATEGORIES[:12]
        + selenium
        + metal_categories_for(metal_routing)
        + ("metalloid", "noble_gas")
    )


def metal_category(element: str, *, metal_routing: str) -> str:
    if metal_routing == "light_heavy":
        return "light_metal" if element in LEGACY_LIGHT_METALS else "heavy_metal"
    if metal_routing == "blocks":
        if element in ALKALI_METALS or element in ALKALINE_EARTH_METALS:
            return "s_block_metal"
        if element in POST_TRANSITION_METALS:
            return "post_transition_metal"
        return "transition_metal"
    if element in ALKALI_METALS:
        return "alkali_metal"
    if element in ALKALINE_EARTH_METALS:
        return "alkaline_earth_metal"
    if element in POST_TRANSITION_METALS:
        return "post_transition_metal"
    if metal_routing == "transition_split":
        if element in LATE_TRANSITION_METALS:
            return "late_transition_metal"
        return "early_transition_metal"
    if metal_routing == "families":
        return "transition_metal"
    raise ValueError(f"Unsupported metal routing: {metal_routing}")


def metal_category_elements(metal_routing: str) -> dict[str, list[str]]:
    result = {category: [] for category in metal_categories_for(metal_routing)}
    for element in sorted(ALL_METALS):
        result[metal_category(element, metal_routing=metal_routing)].append(element)
    return result


def element_category(
    element: str, *, selenium_routing: str, metal_routing: str
) -> str:
    if element in {"D", "T"}:
        return "H"
    if element == "Se":
        return "Se" if selenium_routing == "separate" else selenium_routing
    if element in EXPLICIT_ELEMENTS:
        return element
    if element in METALLOIDS:
        return "metalloid"
    if element in NOBLE_GASES:
        return "noble_gas"
    return metal_category(element, metal_routing=metal_routing)


def temperature_basis(kind: str, temperatures: Sequence[float] | np.ndarray) -> np.ndarray:
    reduced = np.asarray(temperatures, dtype=float) / 1000.0
    if kind == "constant":
        return np.column_stack([np.ones_like(reduced)])
    if kind == "affine":
        return np.column_stack([np.ones_like(reduced), reduced])
    if kind == "cubic":
        return np.column_stack(
            [np.ones_like(reduced), reduced, reduced**2, reduced**3]
        )
    if kind == "shomate":
        return np.column_stack(
            [np.ones_like(reduced), reduced, reduced**2, reduced**3, reduced**-2]
        )
    raise ValueError(f"Unsupported temperature basis: {kind}")


def atom_design(component: Component, temperatures: np.ndarray, kind: str) -> np.ndarray:
    basis = temperature_basis(kind, temperatures)
    return np.einsum("ij,k->ikj", basis, component.counts).reshape(len(basis), -1)


def relative_least_squares(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    relative_design = design / values[:, None]
    scales = np.sqrt(np.mean(relative_design * relative_design, axis=0))
    scales = np.where(scales > 1e-14, scales, 1.0)
    scaled_coefficients = np.linalg.lstsq(
        relative_design / scales,
        np.ones(len(values)),
        rcond=1e-10,
    )[0]
    return scaled_coefficients / scales


def fit_atom_model(
    components: Iterable[Component],
    kind: str,
    *,
    points: int,
    at_298: bool = False,
) -> np.ndarray:
    designs = []
    values = []
    for component in components:
        if at_298:
            if not component.Tmin <= 298.15 <= component.Tmax:
                continue
            temperatures = np.asarray([298.15])
        else:
            temperatures = np.linspace(component.Tmin, component.Tmax, points)
        designs.append(atom_design(component, temperatures, kind))
        values.append(component.cp(temperatures))
    return relative_least_squares(np.vstack(designs), np.concatenate(values))


def fit_power_model(
    components: Iterable[Component],
    *,
    points: int,
    at_298: bool = False,
) -> tuple[float, float]:
    log_weights = []
    log_capacities = []
    for component in components:
        if at_298:
            if not component.Tmin <= 298.15 <= component.Tmax:
                continue
            capacities = component.cp([298.15])
        else:
            capacities = component.cp(
                np.linspace(component.Tmin, component.Tmax, points)
            )
        log_weights.extend([math.log(component.molecular_weight)] * len(capacities))
        log_capacities.extend(np.log(capacities).tolist())
    design = np.column_stack([np.ones(len(log_weights)), log_weights])
    intercept, exponent = np.linalg.lstsq(
        design, np.asarray(log_capacities), rcond=None
    )[0]
    return math.exp(intercept), float(exponent)


def model_values(
    model: str,
    parameters,
    component: Component,
    temperatures: np.ndarray,
) -> np.ndarray:
    if model == "current":
        value = 2.0 * R * (component.molecular_weight / 30.0)
        return np.full_like(temperatures, value)
    if model.startswith("power"):
        factor, exponent = parameters
        return np.full_like(
            temperatures, factor * component.molecular_weight**exponent
        )
    kind = model.removeprefix("atom_").removesuffix("298")
    return atom_design(component, temperatures, kind) @ parameters


def load_components(
    path: Path,
    *,
    folds: int,
    grouping: str,
    selenium_routing: str,
    metal_routing: str,
) -> tuple[list[Component], int, tuple[str, ...]]:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM canonical_ideal_gas_cp ORDER BY cas"
        ).fetchall()

    categories = categories_for(selenium_routing, metal_routing)
    category_index = {name: index for index, name in enumerate(categories)}
    components = []
    excluded = 0
    for row in rows:
        formula = FORMULA_OVERRIDES.get(
            row["cas"],
            str(row["formula"] or "").replace("_{", "").replace("}", "").strip(),
        )
        if not formula or formula.lower() == "mixture":
            excluded += 1
            continue
        try:
            atoms = simple_formula_parser(formula)
            weight = float(molecular_weight(atoms))
        except Exception:
            excluded += 1
            continue
        counts = np.zeros(len(categories))
        for element, count in atoms.items():
            counts[
                category_index[
                    element_category(
                        element,
                        selenium_routing=selenium_routing,
                        metal_routing=metal_routing,
                    )
                ]
            ] += float(count)
        fold_key = formula if grouping == "formula" else row["cas"]
        fold = int(hashlib.sha256(fold_key.encode()).hexdigest()[:8], 16) % folds
        components.append(
            Component(
                cas=row["cas"],
                name=row["name"],
                formula=formula,
                source=row["source"],
                molecular_weight=weight,
                counts=counts,
                Tmin=float(row["Tmin_fit_K"]),
                Tmax=float(row["Tmax_fit_K"]),
                center=float(row["map_center_K"]),
                scale=float(row["map_scale"]),
                coefficients=np.asarray(
                    json.loads(row["cp_coefficients_json"]), dtype=float
                ),
                fold=fold,
            )
        )
    return components, excluded, categories


def percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.percentile(values, quantile))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_summary(result: dict) -> dict:
    range_errors = np.asarray(result["range"], dtype=float)
    return {
        "components": int(len(range_errors)),
        "range_mape_percent": {
            "mean": float(np.mean(range_errors)),
            "median": float(np.median(range_errors)),
            "p90": percentile(range_errors, 90),
            "p95": percentile(range_errors, 95),
            "maximum": float(np.max(range_errors)),
        },
        "fraction_within_10_percent": float(np.mean(range_errors <= 10.0)),
        "fraction_within_20_percent": float(np.mean(range_errors <= 20.0)),
        "point_median_absolute_error_percent": {
            f"{temperature:g}_K": float(np.median(result["points"][temperature]))
            for temperature in REFERENCE_TEMPERATURES
        },
        "delta_h_298_15_to_1000_K_median_absolute_error_percent": float(
            np.median(result["enthalpy"])
        ),
        "delta_s_298_15_to_1000_K_median_absolute_error_percent": float(
            np.median(result["entropy"])
        ),
        "nonpositive_predicted_curves": int(result["negative"]),
    }


def full_fit_summary(
    components: Sequence[Component],
    coefficients: np.ndarray,
    *,
    test_points: int,
) -> dict:
    result = {
        "range": [],
        "points": defaultdict(list),
        "enthalpy": [],
        "entropy": [],
        "negative": 0,
    }
    for component in components:
        temperatures = np.linspace(component.Tmin, component.Tmax, test_points)
        actual = component.cp(temperatures)
        predicted = model_values(
            "atom_shomate", coefficients, component, temperatures
        )
        result["range"].append(
            float(np.mean(100.0 * np.abs((predicted - actual) / actual)))
        )
        result["negative"] += int(np.any(predicted <= 0.0))
        for temperature in REFERENCE_TEMPERATURES:
            if component.Tmin <= temperature <= component.Tmax:
                actual_point = float(component.cp([temperature])[0])
                predicted_point = float(
                    model_values(
                        "atom_shomate",
                        coefficients,
                        component,
                        np.asarray([temperature]),
                    )[0]
                )
                result["points"][temperature].append(
                    100.0 * abs(predicted_point / actual_point - 1.0)
                )
        if component.Tmin <= 298.15 and component.Tmax >= 1000.0:
            integral_temperatures = np.linspace(298.15, 1000.0, 501)
            actual_curve = component.cp(integral_temperatures)
            predicted_curve = model_values(
                "atom_shomate",
                coefficients,
                component,
                integral_temperatures,
            )
            actual_h = float(np.trapezoid(actual_curve, integral_temperatures))
            predicted_h = float(
                np.trapezoid(predicted_curve, integral_temperatures)
            )
            actual_s = float(
                np.trapezoid(
                    actual_curve / integral_temperatures, integral_temperatures
                )
            )
            predicted_s = float(
                np.trapezoid(
                    predicted_curve / integral_temperatures,
                    integral_temperatures,
                )
            )
            result["enthalpy"].append(100.0 * abs(predicted_h / actual_h - 1.0))
            result["entropy"].append(100.0 * abs(predicted_s / actual_s - 1.0))
    return metric_summary(result)


def export_shomate_model(
    path: Path,
    *,
    args: argparse.Namespace,
    components: Sequence[Component],
    categories: Sequence[str],
    excluded: int,
    coefficients: np.ndarray,
    cross_validation: dict,
) -> None:
    width = len(SHOMATE_TERMS)
    category_parameters = {}
    for index, category in enumerate(categories):
        block = coefficients[index * width : (index + 1) * width]
        category_parameters[category] = {
            "occurrence_count": int(
                sum(component.counts[index] > 0 for component in components)
            ),
            "coefficients": {
                term: float(value) for term, value in zip(SHOMATE_TERMS, block)
            },
            "contribution_J_per_mol_atom_K": {
                f"{temperature:g}_K": float(
                    temperature_basis("shomate", [temperature])[0] @ block
                )
                for temperature in REFERENCE_TEMPERATURES
            },
        }

    explicit_routes = {element: element for element in CATEGORIES[:12]}
    explicit_routes.update({"D": "H", "T": "H"})
    explicit_routes["Se"] = (
        "Se" if args.selenium_routing == "separate" else args.selenium_routing
    )
    metalloid_elements = set(METALLOIDS)
    noble_gas_elements = set(NOBLE_GASES)
    if args.selenium_routing == "metalloid":
        metalloid_elements.add("Se")
    elif args.selenium_routing == "noble_gas":
        noble_gas_elements.add("Se")
    payload = {
        "schema_version": 1,
        "model": "shomate_atom_increment_ideal_gas_cp_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "equation": (
            "Cp(T) = sum_g n_g*(A_g + B_g*t + C_g*t^2 + D_g*t^3 "
            "+ E_g/t^2), t=T/1000"
        ),
        "units": {
            "temperature": "K",
            "heat_capacity": "J/(mol*K)",
            "atom_increment": "J/(mol-atom*K)",
        },
        "source_database": {
            "path": str(args.database.resolve().relative_to(ROOT)),
            "sha256": sha256_path(args.database),
            "canonical_records": int(len(components) + excluded),
            "formula_usable_records": int(len(components)),
            "excluded_records": int(excluded),
        },
        "training": {
            "fit_points_per_component": int(args.fit_points),
            "fit_temperature_sampling": (
                "equally spaced over each canonical component's fitted range"
            ),
            "objective": (
                "ordinary least squares on (Cp_predicted-Cp_canonical)/Cp_canonical; "
                "each component has the same number of sampled temperatures"
            ),
            "minimum_source_temperature_K": float(
                min(component.Tmin for component in components)
            ),
            "maximum_source_temperature_K": float(
                max(component.Tmax for component in components)
            ),
            "formula_overrides": dict(FORMULA_OVERRIDES),
            "selenium_routing": args.selenium_routing,
            "metal_routing": args.metal_routing,
        },
        "category_routing": {
            "explicit_elements": explicit_routes,
            "metal_categories": metal_category_elements(args.metal_routing),
            "metalloid_elements": sorted(metalloid_elements),
            "noble_gas_elements": sorted(noble_gas_elements),
        },
        "parameters": category_parameters,
        "diagnostics": {
            "full_fit": full_fit_summary(
                components, coefficients, test_points=args.test_points
            ),
            "cross_validation": {
                "folds": int(args.folds),
                "grouping": args.grouping,
                "test_points_per_component": int(args.test_points),
                **metric_summary(cross_validation),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"WROTE_MODEL {path} sha256={sha256_path(path)}")


def benchmark(args: argparse.Namespace) -> None:
    components, excluded, categories = load_components(
        args.database,
        folds=args.folds,
        grouping=args.grouping,
        selenium_routing=args.selenium_routing,
        metal_routing=args.metal_routing,
    )
    total = len(components) + excluded
    print(
        f"DATA total={total} formula_usable={len(components)} excluded={excluded} "
        f"coverage={100.0 * len(components) / total:.2f}% grouping={args.grouping} "
        f"selenium_routing={args.selenium_routing} metal_routing={args.metal_routing}"
    )
    print(
        "CATEGORY_OCCURRENCE "
        + " ".join(
            f"{name}={sum(component.counts[index] > 0 for component in components)}"
            for index, name in enumerate(categories)
        )
    )

    models = (
        "current",
        "power_range",
        "power_298",
        "atom_constant",
        "atom_constant298",
        "atom_affine",
        "atom_cubic",
        "atom_shomate",
    )
    fitted = {model: {} for model in models if model != "current"}
    for fold in range(args.folds):
        training = [component for component in components if component.fold != fold]
        fitted["power_range"][fold] = fit_power_model(
            training, points=args.fit_points
        )
        fitted["power_298"][fold] = fit_power_model(
            training, points=args.fit_points, at_298=True
        )
        fitted["atom_constant"][fold] = fit_atom_model(
            training, "constant", points=args.fit_points
        )
        fitted["atom_constant298"][fold] = fit_atom_model(
            training, "constant", points=args.fit_points, at_298=True
        )
        for kind in ("affine", "cubic", "shomate"):
            fitted[f"atom_{kind}"][fold] = fit_atom_model(
                training, kind, points=args.fit_points
            )

    metrics = {
        model: {
            "range": [],
            "points": defaultdict(list),
            "enthalpy": [],
            "entropy": [],
            "negative": 0,
        }
        for model in models
    }
    for component in components:
        temperatures = np.linspace(component.Tmin, component.Tmax, args.test_points)
        actual = component.cp(temperatures)
        for model in models:
            parameters = None if model == "current" else fitted[model][component.fold]
            predicted = model_values(model, parameters, component, temperatures)
            relative_error = 100.0 * np.abs((predicted - actual) / actual)
            metrics[model]["range"].append(float(np.mean(relative_error)))
            metrics[model]["negative"] += int(np.any(predicted <= 0.0))
            for temperature in REFERENCE_TEMPERATURES:
                if component.Tmin <= temperature <= component.Tmax:
                    actual_point = float(component.cp([temperature])[0])
                    predicted_point = float(
                        model_values(
                            model,
                            parameters,
                            component,
                            np.asarray([temperature]),
                        )[0]
                    )
                    metrics[model]["points"][temperature].append(
                        100.0 * abs(predicted_point / actual_point - 1.0)
                    )
            if component.Tmin <= 298.15 and component.Tmax >= 1000.0:
                integral_temperatures = np.linspace(298.15, 1000.0, 501)
                actual_curve = component.cp(integral_temperatures)
                predicted_curve = model_values(
                    model, parameters, component, integral_temperatures
                )
                actual_h = float(np.trapezoid(actual_curve, integral_temperatures))
                predicted_h = float(
                    np.trapezoid(predicted_curve, integral_temperatures)
                )
                actual_s = float(
                    np.trapezoid(
                        actual_curve / integral_temperatures, integral_temperatures
                    )
                )
                predicted_s = float(
                    np.trapezoid(
                        predicted_curve / integral_temperatures,
                        integral_temperatures,
                    )
                )
                metrics[model]["enthalpy"].append(
                    100.0 * abs(predicted_h / actual_h - 1.0)
                )
                metrics[model]["entropy"].append(
                    100.0 * abs(predicted_s / actual_s - 1.0)
                )

    print(
        "OUT_OF_SAMPLE "
        "model|range_median|range_mean|range_p90|range_p95|within10|within20|"
        "298med|500med|1000med|Hmed|Smed|negative_curves"
    )
    for model in models:
        result = metrics[model]
        range_errors = np.asarray(result["range"])
        print(
            f"{model}|{np.median(range_errors):.2f}%|{np.mean(range_errors):.2f}%|"
            f"{percentile(range_errors, 90):.2f}%|{percentile(range_errors, 95):.2f}%|"
            f"{100.0 * np.mean(range_errors <= 10.0):.1f}%|"
            f"{100.0 * np.mean(range_errors <= 20.0):.1f}%|"
            f"{np.median(result['points'][298.15]):.2f}%|"
            f"{np.median(result['points'][500.0]):.2f}%|"
            f"{np.median(result['points'][1000.0]):.2f}%|"
            f"{np.median(result['enthalpy']):.2f}%|"
            f"{np.median(result['entropy']):.2f}%|"
            f"{result['negative']}"
        )

    power_range = fit_power_model(components, points=args.fit_points)
    power_298 = fit_power_model(components, points=args.fit_points, at_298=True)
    print(
        f"POWER_FULL range_A={power_range[0]:.9g} range_B={power_range[1]:.9g} "
        f"at298_A={power_298[0]:.9g} at298_B={power_298[1]:.9g}"
    )

    full_atom_coefficients = {}
    for kind in ("constant", "affine", "cubic", "shomate"):
        coefficients = fit_atom_model(components, kind, points=args.fit_points)
        full_atom_coefficients[kind] = coefficients
        width = temperature_basis(kind, [298.15]).shape[1]
        increments = []
        for index, category in enumerate(categories):
            block = coefficients[index * width : (index + 1) * width]
            contributions = [
                float(temperature_basis(kind, [temperature])[0] @ block)
                for temperature in REFERENCE_TEMPERATURES
            ]
            increments.append(
                f"{category}:{contributions[0]:+.2f}/"
                f"{contributions[1]:+.2f}/{contributions[2]:+.2f}"
            )
        print(
            f"INCREMENTS_atom_{kind} at298/500/1000_J_per_mol_atom_K "
            + " ".join(increments)
        )

    if args.output_json is not None:
        export_shomate_model(
            args.output_json,
            args=args,
            components=components,
            categories=categories,
            excluded=excluded,
            coefficients=full_atom_coefficients["shomate"],
            cross_validation=metrics["atom_shomate"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--grouping",
        choices=("formula", "cas"),
        default="formula",
        help="Keep identical formulas or only identical CAS records within a fold.",
    )
    parser.add_argument(
        "--selenium-routing",
        choices=("noble_gas", "metalloid", "separate"),
        default="separate",
        help="Choose whether selenium shares a category or gets its own increment.",
    )
    parser.add_argument(
        "--metal-routing",
        choices=("light_heavy", "blocks", "families", "transition_split"),
        default="light_heavy",
        help="Choose the level of chemical-family detail for metal increments.",
    )
    parser.add_argument("--fit-points", type=int, default=41)
    parser.add_argument("--test-points", type=int, default=101)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write the full-data Shomate atom-increment model to this path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
