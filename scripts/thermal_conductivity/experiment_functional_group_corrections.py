#!/usr/bin/env python3
"""Cross-validate functional-group corrections to gas conductivity estimates.

Entire compounds are held out together, and ridge strength is selected by
inner compound-held-out cross-validation. Candidate forms include constant
log multipliers, group-count multipliers, direct R/Cv additions to the
dimensionless conductivity factor, and reduced-temperature terms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from chemicals.identifiers import search_chemical
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perry_properties import PerryPropertyLibrary  # noqa: E402
from physical_constants import R_J_MOL_K  # noqa: E402
from property_resolution.ideal_gas_cp import CANONICAL_DATABASE_PATH  # noqa: E402
from property_resolution.organic_classification import (  # noqa: E402
    classify_strict_molecular_organic,
)
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    ACENTRIC_METHOD,
    DEFAULT_CONDUCTIVITY_DATABASE,
    DEFAULT_PERRY_DATABASE,
    METHODS,
    evaluate_compound,
    percentile,
)


SMARTS = {
    "carboxylic_acid": "[CX3](=[OX1])[OX2H1]",
    "alcohol": "[CX4][OX2H1]",
    "phenol": "[c][OX2H1]",
    "ether": "[OD2]([#6;!$(C=O)])[#6;!$(C=O)]",
    "ester": "[CX3](=O)[OX2][#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3;H1,H2](=O)",
    "amide": "[CX3](=O)[NX3]",
    "amine": "[NX3;H0,H1,H2;!$(N[C,S,P]=O);!$([N+])]",
    "nitrile": "[CX2]#N",
    "nitro": "[N+](=O)[O-]",
    "thiol": "[SX2H1]",
    "sulfide": "[#6][SX2][#6]",
    "sulfoxide": "[SX3](=O)",
    "sulfone": "[SX4](=O)(=O)",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
}
PATTERNS = {name: Chem.MolFromSmarts(value) for name, value in SMARTS.items()}
ELEMENT_FEATURES = {
    "fluorine": "F",
    "chlorine": "Cl",
    "bromine": "Br",
    "iodine": "I",
}
RIDGE_ALPHAS = (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
DEFAULT_EXCLUDED_CAS = ("144-62-7",)


@dataclass(frozen=True)
class Observation:
    cas: str
    reference: float
    baseline: float
    temperature_K: float
    reduced_temperature: float
    r_over_cv: float
    baseline_factor: float


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target_space: str
    group_values: str
    temperature_degree: int
    multiply_r_over_cv: bool
    targeted_subclasses: str = ""
    prune_low_improvement: bool = False
    split_hydrocarbon: bool = False


MODEL_SPECS = (
    ModelSpec("log_constant_presence", "log", "presence", 0, False),
    ModelSpec("log_constant_count", "log", "count", 0, False),
    ModelSpec("factor_r_over_cv_presence", "factor", "presence", 0, True),
    ModelSpec("factor_r_over_cv_count", "factor", "count", 0, True),
    ModelSpec("log_linear_Tr_presence", "log", "presence", 1, False),
    ModelSpec("log_inverse_T_presence", "log", "presence", 1, False),
    ModelSpec("log_inverse_Tr_presence", "log", "presence", 1, False),
    ModelSpec("log_quadratic_Tr_presence", "log", "presence", 2, False),
    ModelSpec("log_constant_targeted_presence", "log", "presence", 0, False, "both"),
    ModelSpec("log_linear_Tr_targeted_presence", "log", "presence", 1, False, "both"),
    ModelSpec("log_inverse_T_targeted_presence", "log", "presence", 1, False, "both"),
    ModelSpec(
        "log_inverse_T_acid_targeted_presence",
        "log",
        "presence",
        1,
        False,
        "acid",
    ),
    ModelSpec(
        "log_inverse_T_acid_detailed_presence",
        "log",
        "presence",
        1,
        False,
        "acid_detailed",
    ),
    ModelSpec(
        "log_constant_acid_detailed_presence",
        "log",
        "presence",
        0,
        False,
        "acid_detailed",
    ),
    ModelSpec(
        "log_linear_Tr_acid_detailed_presence",
        "log",
        "presence",
        1,
        False,
        "acid_detailed",
    ),
    ModelSpec(
        "log_inverse_Tr_acid_detailed_presence",
        "log",
        "presence",
        1,
        False,
        "acid_detailed",
    ),
    ModelSpec(
        "log_inverse_T_acid_detailed_pruned_presence",
        "log",
        "presence",
        1,
        False,
        "acid_detailed",
        True,
    ),
    ModelSpec(
        "log_inverse_T_acid_detailed_pruned_hc_split_presence",
        "log",
        "presence",
        1,
        False,
        "acid_detailed",
        True,
        True,
    ),
    ModelSpec(
        "log_inverse_T_fluorine_targeted_presence",
        "log",
        "presence",
        1,
        False,
        "fluorine",
    ),
    ModelSpec("factor_r_over_cv_linear_Tr_presence", "factor", "presence", 1, True),
    ModelSpec("factor_r_over_cv_quadratic_Tr_presence", "factor", "presence", 2, True),
    ModelSpec("factor_r_over_cv_linear_Tr_count", "factor", "count", 1, True),
    ModelSpec("factor_r_over_cv_quadratic_Tr_count", "factor", "count", 2, True),
)

TARGETED_FEATURES = {
    "acid_straight_saturated_monocarboxylic",
    "acid_branched_saturated_monocarboxylic",
    "acid_unsaturated_or_aromatic_monocarboxylic",
    "acid_dicarboxylic",
    "acid_straight_short_C1_C3",
    "acid_straight_medium_C4_C6",
    "acid_straight_long_C7_plus",
    "fluoro_organic_monofluoro",
    "fluoro_organic_polyfluoro",
    "fluoro_inorganic",
    "hc_acyclic_alkane",
    "hc_saturated_cyclic",
    "hc_alkene",
    "hc_alkyne",
    "hc_aromatic",
}
ACID_TARGETED_FEATURES = {
    "acid_straight_saturated_monocarboxylic",
    "acid_branched_saturated_monocarboxylic",
    "acid_unsaturated_or_aromatic_monocarboxylic",
    "acid_dicarboxylic",
}
ACID_DETAILED_FEATURES = {
    "acid_straight_short_C1_C3",
    "acid_straight_medium_C4_C6",
    "acid_straight_long_C7_plus",
    "acid_branched_saturated_monocarboxylic",
    "acid_unsaturated_or_aromatic_monocarboxylic",
    "acid_dicarboxylic",
}
FLUORINE_TARGETED_FEATURES = {
    "fluoro_organic_monofluoro",
    "fluoro_organic_polyfluoro",
    "fluoro_inorganic",
}
HYDROCARBON_SPLIT_FEATURES = {
    "hc_acyclic_alkane",
    "hc_saturated_cyclic",
    "hc_alkene",
    "hc_alkyne",
    "hc_aromatic",
}
LOW_IMPROVEMENT_FEATURES = {
    "acid_branched_saturated_monocarboxylic",
    "acid_dicarboxylic",
    "amine",
    "bromine",
    "ester",
    "ether",
    "fluorine",
}


def functional_group_counts(cas: str, formula: str = "") -> dict[str, float]:
    """Return local structure-derived group counts for one CAS number."""
    try:
        metadata = search_chemical(cas)
        smiles = str(metadata.smiles or "").strip()
    except Exception:
        return {}
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return {}
    identity = classify_strict_molecular_organic(
        cas=cas,
        formula=formula or getattr(metadata, "formula", ""),
        smiles=smiles,
    )
    if not identity.is_organic:
        return {}

    counts = {
        name: float(len(molecule.GetSubstructMatches(pattern)))
        for name, pattern in PATTERNS.items()
    }
    atom_symbols = Counter(atom.GetSymbol() for atom in molecule.GetAtoms())
    counts.update(
        {
            name: float(atom_symbols[element])
            for name, element in ELEMENT_FEATURES.items()
        }
    )
    counts["aromatic_ring"] = float(rdMolDescriptors.CalcNumAromaticRings(molecule))
    counts["aliphatic_ring"] = float(rdMolDescriptors.CalcNumAliphaticRings(molecule))
    counts["heteroaromatic_N"] = float(
        sum(
            atom.GetSymbol() == "N" and atom.GetIsAromatic()
            for atom in molecule.GetAtoms()
        )
    )
    counts["heteroaromatic_O"] = float(
        sum(
            atom.GetSymbol() == "O" and atom.GetIsAromatic()
            for atom in molecule.GetAtoms()
        )
    )
    heavy_elements = {symbol for symbol in atom_symbols if symbol != "H"}
    counts["hydrocarbon_only"] = float(heavy_elements == {"C"})
    if heavy_elements == {"C"}:
        if any(atom.GetIsAromatic() for atom in molecule.GetAtoms()):
            hydrocarbon_class = "hc_aromatic"
        elif any(bond.GetBondTypeAsDouble() >= 2.9 for bond in molecule.GetBonds()):
            hydrocarbon_class = "hc_alkyne"
        elif any(bond.GetBondTypeAsDouble() >= 1.9 for bond in molecule.GetBonds()):
            hydrocarbon_class = "hc_alkene"
        elif molecule.GetRingInfo().NumRings() > 0:
            hydrocarbon_class = "hc_saturated_cyclic"
        else:
            hydrocarbon_class = "hc_acyclic_alkane"
        counts[hydrocarbon_class] = 1.0
    acid_count = int(counts["carboxylic_acid"])
    carbon_atoms = [atom for atom in molecule.GetAtoms() if atom.GetSymbol() == "C"]
    if acid_count >= 2:
        counts["acid_dicarboxylic"] = 1.0
    elif acid_count == 1:
        carbon_branch = any(
            sum(neighbor.GetSymbol() == "C" for neighbor in atom.GetNeighbors()) >= 3
            for atom in carbon_atoms
        )
        unsaturated_or_aromatic = any(
            atom.GetIsAromatic()
            or any(
                bond.GetBondTypeAsDouble() > 1.0
                and not (
                    {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
                    == {"C", "O"}
                )
                for bond in atom.GetBonds()
            )
            for atom in carbon_atoms
        )
        if unsaturated_or_aromatic:
            counts["acid_unsaturated_or_aromatic_monocarboxylic"] = 1.0
        elif carbon_branch:
            counts["acid_branched_saturated_monocarboxylic"] = 1.0
        else:
            counts["acid_straight_saturated_monocarboxylic"] = 1.0
            carbon_count = len(carbon_atoms)
            if carbon_count <= 3:
                counts["acid_straight_short_C1_C3"] = 1.0
            elif carbon_count <= 6:
                counts["acid_straight_medium_C4_C6"] = 1.0
            else:
                counts["acid_straight_long_C7_plus"] = 1.0

    fluorine_count = int(atom_symbols["F"])
    if fluorine_count:
        if atom_symbols["C"]:
            suffix = "monofluoro" if fluorine_count == 1 else "polyfluoro"
            counts[f"fluoro_organic_{suffix}"] = 1.0
        else:
            counts["fluoro_inorganic"] = 1.0
    return {name: value for name, value in counts.items() if value > 0.0}


def load_observations(
    args: argparse.Namespace,
) -> tuple[list[Observation], dict, Counter]:
    thermal = json.loads(args.conductivity_database.read_text(encoding="utf-8"))[
        "chemicals"
    ]
    perry = PerryPropertyLibrary(path=args.perry_database)
    observations = []
    groups = {}
    exclusions: Counter = Counter()
    for cas, thermal_entry in sorted(thermal.items()):
        if not thermal_entry.get("vapor_thermal_conductivity"):
            continue
        if cas in args.exclude_cas:
            exclusions[f"explicitly excluded CAS {cas}"] += 1
            continue
        try:
            perry_entry = perry.get(cas, expand_identity=False)
            if perry_entry is None:
                raise ValueError("Perry property record unavailable")
            _, points = evaluate_compound(
                cas,
                thermal_entry,
                perry_entry,
                sample_points=args.points,
                method=args.baseline,
            )
        except Exception as exc:
            exclusions[f"{type(exc).__name__}: {exc}"] += 1
            continue
        groups[cas] = functional_group_counts(
            cas,
            str(thermal_entry.get("formula") or ""),
        )
        observations.extend(
            Observation(
                cas=cas,
                reference=point.reference_W_per_m_K,
                baseline=point.predicted_W_per_m_K,
                temperature_K=point.temperature_K,
                reduced_temperature=point.reduced_temperature,
                r_over_cv=R_J_MOL_K / point.cv_J_per_mol_K,
                baseline_factor=point.dimensionless_factor,
            )
            for point in points
        )
    return observations, groups, exclusions


def retained_features(groups: dict, minimum_compounds: int) -> list[str]:
    support: Counter = Counter()
    for counts in groups.values():
        support.update(name for name, value in counts.items() if value > 0.0)
    return sorted(
        name
        for name, count in support.items()
        if count >= minimum_compounds and name not in TARGETED_FEATURES
    )


def targeted_features(groups: dict, minimum_compounds: int) -> list[str]:
    support: Counter = Counter()
    for counts in groups.values():
        support.update(name for name, value in counts.items() if value > 0.0)
    return sorted(
        name for name in TARGETED_FEATURES if support[name] >= minimum_compounds
    )


def group_matrix(
    observations: Sequence[Observation],
    groups: dict,
    feature_names: Sequence[str],
    mode: str,
) -> np.ndarray:
    matrix = np.asarray(
        [
            [groups[observation.cas].get(name, 0.0) for name in feature_names]
            for observation in observations
        ],
        dtype=float,
    )
    return (matrix > 0.0).astype(float) if mode == "presence" else matrix


def design_matrix(
    observations: Sequence[Observation],
    group_values: np.ndarray,
    spec: ModelSpec,
) -> np.ndarray:
    if "inverse_Tr" in spec.name:
        reduced = np.asarray(
            [
                1.0 / observation.reduced_temperature - 1.0
                for observation in observations
            ]
        )
    elif "inverse_T" in spec.name:
        reduced = np.asarray(
            [500.0 / observation.temperature_K - 1.0 for observation in observations]
        )
    else:
        reduced = np.asarray(
            [observation.reduced_temperature - 1.0 for observation in observations]
        )
    r_over_cv = np.asarray([observation.r_over_cv for observation in observations])
    terms = []
    for degree in range(spec.temperature_degree + 1):
        block = group_values * reduced[:, None] ** degree
        if spec.multiply_r_over_cv:
            block *= r_over_cv[:, None]
        terms.append(block)
    return np.concatenate(terms, axis=1)


def balanced_folds(cases: Sequence[str], folds: int, salt: str) -> dict[str, int]:
    ordered = sorted(
        set(cases),
        key=lambda cas: hashlib.sha256(f"{salt}:{cas}".encode()).hexdigest(),
    )
    return {cas: index % folds for index, cas in enumerate(ordered)}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(ROOT))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": sha256_path(resolved)}


def compound_equal_weights(cases: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts = Counter(cases[mask])
    weights = np.zeros(len(cases), dtype=float)
    weights[mask] = [1.0 / counts[cas] for cas in cases[mask]]
    return weights


def fit_ridge(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    active = weights > 0.0
    x = matrix[active]
    y = target[active]
    w = weights[active]
    scale = np.sqrt(np.sum(w[:, None] * x * x, axis=0) / np.sum(w))
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    standardized = x / scale
    gram = standardized.T @ (w[:, None] * standardized)
    rhs = standardized.T @ (w * y)
    coefficients = np.linalg.solve(
        gram + max(alpha, 1.0e-12) * np.identity(gram.shape[0]),
        rhs,
    )
    return coefficients / scale


def target_values(observations: Sequence[Observation], space: str) -> np.ndarray:
    reference = np.asarray([observation.reference for observation in observations])
    baseline = np.asarray([observation.baseline for observation in observations])
    if space == "log":
        return np.log(reference / baseline)
    factors = np.asarray([observation.baseline_factor for observation in observations])
    return factors * (reference / baseline - 1.0)


def corrected_predictions(
    observations: Sequence[Observation],
    matrix: np.ndarray,
    coefficients: np.ndarray,
    space: str,
) -> np.ndarray:
    baseline = np.asarray([observation.baseline for observation in observations])
    correction = matrix @ coefficients
    if space == "log":
        return baseline * np.exp(np.clip(correction, -50.0, 50.0))
    factors = np.asarray([observation.baseline_factor for observation in observations])
    return baseline * np.maximum(factors + correction, 1.0e-12) / factors


def compound_equal_mape(
    observations: Sequence[Observation],
    predictions: np.ndarray,
    mask: np.ndarray,
) -> float:
    cases = np.asarray([observation.cas for observation in observations])
    reference = np.asarray([observation.reference for observation in observations])
    weights = compound_equal_weights(cases, mask)
    errors = np.abs(predictions / reference - 1.0)
    return float(np.sum(weights * errors) / np.sum(weights))


def select_alpha(
    observations: Sequence[Observation],
    matrix: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    spec: ModelSpec,
    inner_folds: int,
    salt: str,
) -> float:
    cases = np.asarray([observation.cas for observation in observations])
    assignments = balanced_folds(cases[training], inner_folds, salt)
    scores = []
    for alpha in RIDGE_ALPHAS:
        fold_scores = []
        for fold in range(inner_folds):
            validation = np.asarray(
                [
                    training[index] and assignments.get(cas) == fold
                    for index, cas in enumerate(cases)
                ]
            )
            inner_training = training & ~validation
            coefficients = fit_ridge(
                matrix,
                target,
                compound_equal_weights(cases, inner_training),
                alpha,
            )
            predictions = corrected_predictions(
                observations, matrix, coefficients, spec.target_space
            )
            fold_scores.append(
                compound_equal_mape(observations, predictions, validation)
            )
        scores.append((sum(fold_scores) / len(fold_scores), alpha))
    return min(scores)[1]


def cross_validated_predictions(
    observations: Sequence[Observation],
    groups: dict,
    feature_names: Sequence[str],
    spec: ModelSpec,
    outer_folds: int,
    inner_folds: int,
) -> tuple[np.ndarray, list[float]]:
    cases = np.asarray([observation.cas for observation in observations])
    values = group_matrix(observations, groups, feature_names, spec.group_values)
    matrix = design_matrix(observations, values, spec)
    target = target_values(observations, spec.target_space)
    assignments = balanced_folds(
        cases, outer_folds, "outer:functional_group_corrections"
    )
    predictions = np.full(len(observations), np.nan)
    selected_alphas = []
    for fold in range(outer_folds):
        test = np.asarray([assignments[cas] == fold for cas in cases])
        training = ~test
        alpha = select_alpha(
            observations,
            matrix,
            target,
            training,
            spec,
            inner_folds,
            f"inner:functional_group_corrections:{fold}",
        )
        selected_alphas.append(alpha)
        coefficients = fit_ridge(
            matrix,
            target,
            compound_equal_weights(cases, training),
            alpha,
        )
        all_predictions = corrected_predictions(
            observations, matrix, coefficients, spec.target_space
        )
        predictions[test] = all_predictions[test]
    if not np.all(np.isfinite(predictions)) or np.any(predictions <= 0.0):
        raise RuntimeError(f"{spec.name} produced invalid held-out predictions")
    return predictions, selected_alphas


def metrics(observations: Sequence[Observation], predictions: np.ndarray) -> dict:
    reference = np.asarray([observation.reference for observation in observations])
    cases = np.asarray([observation.cas for observation in observations])
    signed = 100.0 * (predictions / reference - 1.0)
    absolute = np.abs(signed)
    curve_mapes = [float(np.mean(absolute[cases == cas])) for cas in sorted(set(cases))]
    return {
        "points": len(observations),
        "compounds": len(set(cases)),
        "mape_percent": float(np.mean(absolute)),
        "median_ape_percent": float(np.median(absolute)),
        "p90_ape_percent": percentile(absolute, 0.90),
        "p95_ape_percent": percentile(absolute, 0.95),
        "maximum_ape_percent": float(np.max(absolute)),
        "mean_signed_error_percent": float(np.mean(signed)),
        "compound_equal_mean_curve_mape_percent": float(np.mean(curve_mapes)),
        "compound_equal_median_curve_mape_percent": float(np.median(curve_mapes)),
    }


def group_diagnostics(
    observations: Sequence[Observation],
    groups: dict,
    feature_names: Sequence[str],
    baseline: np.ndarray,
    corrected: np.ndarray,
) -> dict:
    cases = np.asarray([observation.cas for observation in observations])
    reference = np.asarray([observation.reference for observation in observations])
    output = {}
    for feature in feature_names:
        mask = np.asarray([groups[cas].get(feature, 0.0) > 0.0 for cas in cases])
        before = 100.0 * (baseline[mask] / reference[mask] - 1.0)
        after = 100.0 * (corrected[mask] / reference[mask] - 1.0)
        output[feature] = {
            "compounds": len(set(cases[mask])),
            "baseline_mape_percent": float(np.mean(np.abs(before))),
            "corrected_mape_percent": float(np.mean(np.abs(after))),
            "baseline_bias_percent": float(np.mean(before)),
            "corrected_bias_percent": float(np.mean(after)),
        }
    return output


def final_fit(
    observations: Sequence[Observation],
    groups: dict,
    feature_names: Sequence[str],
    spec: ModelSpec,
    folds: int,
) -> dict:
    """Fit the selected form to all compounds and return portable coefficients."""
    cases = np.asarray([observation.cas for observation in observations])
    all_rows = np.ones(len(observations), dtype=bool)
    values = group_matrix(observations, groups, feature_names, spec.group_values)
    matrix = design_matrix(observations, values, spec)
    target = target_values(observations, spec.target_space)
    alpha = select_alpha(
        observations,
        matrix,
        target,
        all_rows,
        spec,
        folds,
        f"final:{spec.name}",
    )
    coefficients = fit_ridge(
        matrix,
        target,
        compound_equal_weights(cases, all_rows),
        alpha,
    ).reshape(spec.temperature_degree + 1, len(feature_names))
    predictions = corrected_predictions(
        observations,
        matrix,
        coefficients.ravel(),
        spec.target_space,
    )
    by_feature = {
        feature: {
            "basis_coefficients": [
                float(coefficients[degree, index])
                for degree in range(spec.temperature_degree + 1)
            ]
        }
        for index, feature in enumerate(feature_names)
    }
    if "inverse_T" in spec.name and "inverse_Tr" not in spec.name:
        basis = "x = 500 K/T - 1"
        for record in by_feature.values():
            a, b = record["basis_coefficients"]
            record["A"] = a - b
            record["B_K"] = 500.0 * b
        direct_form = "ln(k/k0) = sum_g I_g*(A_g + B_g/T), T in K"
    elif "inverse_Tr" in spec.name:
        basis = "x = 1/Tr - 1"
        for record in by_feature.values():
            a, b = record["basis_coefficients"]
            record["A"] = a - b
            record["B"] = b
        direct_form = "ln(k/k0) = sum_g I_g*(A_g + B_g/Tr)"
    elif "linear_Tr" in spec.name:
        basis = "x = Tr - 1"
        direct_form = "ln(k/k0) = sum_g I_g*(a_g + b_g*(Tr - 1))"
    else:
        basis = "x = 1"
        direct_form = f"{spec.target_space} correction with constant group terms"
    return {
        "model": spec.name,
        "ridge_alpha": alpha,
        "alpha_selection_folds": folds,
        "training_weighting": "equal total weight per CAS compound",
        "basis": basis,
        "direct_form": direct_form,
        "feature_order": list(feature_names),
        "coefficients": by_feature,
        "resubstitution_metrics_not_for_validation": metrics(observations, predictions),
    }


def format_model(name: str, result: dict, baseline_mape: float) -> str:
    removed = baseline_mape - result["mape_percent"]
    relative = 100.0 * removed / baseline_mape
    return (
        f"  {name:42s} MAPE={result['mape_percent']:6.2f}% "
        f"removed={removed:+6.2f} points ({relative:+5.1f}%) "
        f"P95={result['p95_ape_percent']:6.2f}% "
        f"bias={result['mean_signed_error_percent']:+6.2f}%"
    )


def run_experiment(args: argparse.Namespace) -> tuple[dict, str]:
    observations, groups, exclusions = load_observations(args)
    if not observations:
        raise RuntimeError("No benchmark observations are available")
    features = retained_features(groups, args.minimum_group_compounds)
    specific_features = targeted_features(groups, args.minimum_targeted_group_compounds)
    if not features:
        raise RuntimeError("No functional groups meet the support threshold")

    baseline_predictions = np.asarray(
        [observation.baseline for observation in observations]
    )
    baseline_metrics = metrics(observations, baseline_predictions)
    model_results = {}
    predictions_by_model = {}
    features_by_model = {}
    selected_specs = [
        spec for spec in MODEL_SPECS if not args.models or spec.name in args.models
    ]
    for spec in selected_specs:
        model_features = features
        if spec.targeted_subclasses:
            replaced = set()
            selected_specific = []
            if spec.targeted_subclasses in {"acid", "acid_detailed", "both"}:
                replaced.add("carboxylic_acid")
                acid_features = (
                    ACID_DETAILED_FEATURES
                    if spec.targeted_subclasses == "acid_detailed"
                    else ACID_TARGETED_FEATURES
                )
                selected_specific.extend(
                    feature for feature in specific_features if feature in acid_features
                )
            if spec.targeted_subclasses in {"fluorine", "both"}:
                replaced.add("fluorine")
                selected_specific.extend(
                    feature
                    for feature in specific_features
                    if feature in FLUORINE_TARGETED_FEATURES
                )
            model_features = [
                feature for feature in features if feature not in replaced
            ] + selected_specific
        if spec.prune_low_improvement:
            model_features = [
                feature
                for feature in model_features
                if feature not in LOW_IMPROVEMENT_FEATURES
            ]
        if spec.split_hydrocarbon:
            model_features = [
                feature for feature in model_features if feature != "hydrocarbon_only"
            ] + [
                feature
                for feature in specific_features
                if feature in HYDROCARBON_SPLIT_FEATURES
            ]
        predictions, alphas = cross_validated_predictions(
            observations,
            groups,
            model_features,
            spec,
            args.outer_folds,
            args.inner_folds,
        )
        result = metrics(observations, predictions)
        result["selected_ridge_alphas"] = alphas
        result["mape_points_removed"] = (
            baseline_metrics["mape_percent"] - result["mape_percent"]
        )
        model_results[spec.name] = result
        predictions_by_model[spec.name] = predictions
        features_by_model[spec.name] = model_features

    lowest_mape_name = min(
        model_results,
        key=lambda name: model_results[name]["mape_percent"],
    )
    selected_name = args.selected_model or lowest_mape_name
    if selected_name not in model_results:
        raise ValueError(
            f"Selected model {selected_name!r} was not included by --model"
        )
    selected_spec = next(spec for spec in selected_specs if spec.name == selected_name)
    diagnostics = group_diagnostics(
        observations,
        groups,
        features_by_model[selected_name],
        baseline_predictions,
        predictions_by_model[selected_name],
    )
    fitted = final_fit(
        observations,
        groups,
        features_by_model[selected_name],
        selected_spec,
        args.final_fit_folds,
    )
    payload = {
        "schema_version": 2,
        "experiment": "compound_held_out_functional_group_conductivity_corrections",
        "baseline_method": args.baseline,
        "cross_validation": {
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "ridge_alphas": RIDGE_ALPHAS,
            "fold_unit": "CAS compound",
            "training_weighting": "equal total weight per compound",
        },
        "reproducibility": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "rdkit", "chemicals")
            },
            "argv": sys.argv,
            "inputs": {
                "thermal_conductivity": artifact_record(args.conductivity_database),
                "perry_properties": artifact_record(args.perry_database),
                "ideal_gas_heat_capacity": artifact_record(CANONICAL_DATABASE_PATH),
                "uv_lock": artifact_record(ROOT / "uv.lock"),
            },
            "scripts": {
                "experiment": artifact_record(Path(__file__)),
                "baseline_benchmark": artifact_record(
                    Path(__file__).with_name("benchmark_gas_viscosity_cv_relation.py")
                ),
                "reproduce": artifact_record(Path(__file__).with_name("reproduce.sh")),
            },
        },
        "coverage": {
            "compounds": baseline_metrics["compounds"],
            "sampled_states": baseline_metrics["points"],
            "points_per_curve": args.points,
            "explicitly_excluded_cas": sorted(set(args.exclude_cas)),
            "exclusions": dict(exclusions),
        },
        "features": features,
        "targeted_features": specific_features,
        "features_by_model": features_by_model,
        "minimum_group_compounds": args.minimum_group_compounds,
        "minimum_targeted_group_compounds": args.minimum_targeted_group_compounds,
        "zeroed_low_improvement_features": sorted(LOW_IMPROVEMENT_FEATURES),
        "pruning_rule": (
            "fixed zero after less than 15% relative class-MAPE improvement in the "
            "preceding compound-held-out experiment"
        ),
        "baseline": baseline_metrics,
        "models": model_results,
        "lowest_mape_model": lowest_mape_name,
        "selected_model": selected_name,
        "selected_model_group_diagnostics": diagnostics,
        "final_full_data_fit": fitted,
    }

    lines = [
        "Compound-held-out functional-group correction experiment",
        f"baseline: {args.baseline}",
        (
            f"coverage: {baseline_metrics['compounds']} compounds, "
            f"{baseline_metrics['points']} sampled states"
        ),
        (
            f"validation: {args.outer_folds}-fold outer / {args.inner_folds}-fold "
            "inner by whole CAS compound"
        ),
        (
            f"features: {len(features)} groups with at least "
            f"{args.minimum_group_compounds} compounds"
        ),
        (
            f"targeted acid/fluorine features: {len(specific_features)} with at least "
            f"{args.minimum_targeted_group_compounds} compounds"
        ),
        "",
        "HELD-OUT PERFORMANCE",
        format_model(
            "uncorrected baseline",
            baseline_metrics,
            baseline_metrics["mape_percent"],
        ),
    ]
    for name, result in sorted(
        model_results.items(), key=lambda item: item[1]["mape_percent"]
    ):
        lines.append(format_model(name, result, baseline_metrics["mape_percent"]))
    lines.extend(("", f"SELECTED MODEL GROUP EFFECTS: {selected_name}"))
    for name, result in sorted(
        diagnostics.items(),
        key=lambda item: (
            item[1]["corrected_mape_percent"] - item[1]["baseline_mape_percent"]
        ),
    ):
        lines.append(
            f"  {name:20s} n={result['compounds']:3d} "
            f"MAPE {result['baseline_mape_percent']:6.2f}% -> "
            f"{result['corrected_mape_percent']:6.2f}%  "
            f"bias {result['baseline_bias_percent']:+6.2f}% -> "
            f"{result['corrected_bias_percent']:+6.2f}%"
        )
    if exclusions:
        lines.extend(("", "EXCLUSIONS"))
        for reason, count in exclusions.most_common():
            lines.append(f"  {count:4d} {reason}")
    return payload, "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=METHODS, default=ACENTRIC_METHOD)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=[spec.name for spec in MODEL_SPECS],
        help="Run only this model form; repeat to select multiple forms.",
    )
    parser.add_argument(
        "--selected-model",
        choices=[spec.name for spec in MODEL_SPECS],
        help="Model to fit and report as selected; default is minimum held-out MAPE.",
    )
    parser.add_argument(
        "--conductivity-database",
        type=Path,
        default=DEFAULT_CONDUCTIVITY_DATABASE,
    )
    parser.add_argument("--perry-database", type=Path, default=DEFAULT_PERRY_DATABASE)
    parser.add_argument(
        "--exclude-cas",
        action="append",
        default=list(DEFAULT_EXCLUDED_CAS),
        help="CAS to exclude; repeatable (default excludes oxalic acid 144-62-7).",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=51,
        help="Uniform points per Perry conductivity curve (default: 51).",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument(
        "--final-fit-folds",
        type=int,
        default=5,
        help="Compound folds used to select the final full-data ridge alpha (default: 5).",
    )
    parser.add_argument(
        "--minimum-group-compounds",
        type=int,
        default=8,
        help="Minimum compound support for a retained group term (default: 8).",
    )
    parser.add_argument(
        "--minimum-targeted-group-compounds",
        type=int,
        default=3,
        help="Minimum support for targeted acid/fluorine subclasses (default: 3).",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points must be at least 2")
    if args.outer_folds < 2 or args.inner_folds < 2:
        parser.error("outer and inner folds must each be at least 2")
    if args.final_fit_folds < 2:
        parser.error("--final-fit-folds must be at least 2")
    if args.minimum_group_compounds < 2:
        parser.error("--minimum-group-compounds must be at least 2")
    if args.minimum_targeted_group_compounds < 2:
        parser.error("--minimum-targeted-group-compounds must be at least 2")
    return args


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    args = parse_args()
    payload, report = run_experiment(args)
    print(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
