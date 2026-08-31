#!/usr/bin/env python3
"""Benchmark and fit the HBD-organic Cpig/Cpl ratio group contribution."""

import hashlib
import os
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from chemicals.identifiers import search_chemical
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, rdMolDescriptors
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from property_resolution.ideal_gas_cp import load_bundled_kernel
from property_resolution.liquid_cp import load_bundled_liquid_kernel
from chemical_properties import ChemicalDatabase
from property_resolver import PropertyResolver


R = 8.31446261815324
HALOGENS = {9, 17, 35, 53}
STRICT_INORGANIC_CAS = {"74-90-8"}
STRICT_ORGANIC_CAS = {"144-62-7"}
DONOR_CLASSES = (
    "alcohol_OH",
    "amide_like_NH",
    "amine_NH",
    "aromatic_NH",
    "carboxylic_acid",
    "phenol",
    "thiol_SH",
    "other",
)
FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)
ALPHAS = (1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
GLYCEROL_CAS = "56-81-5"
GLYCEROL_SHOMATE = (12.44057022, 386.95602492, -219.74920838, 48.63893620, 1.21146806)


class GlycerolIdealGasCp:
    Tmin = 250.0
    Tmax = 1500.0

    @staticmethod
    def cp(T):
        A, B, C, D, E = GLYCEROL_SHOMATE
        t = T / 1000.0
        return A + B*t + C*t*t + D*t*t*t + E/(t*t)


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


def structure(cas):
    try:
        smiles = str(search_chemical(cas).smiles or "").strip()
        return Chem.MolFromSmiles(smiles) if smiles else None
    except Exception:
        return None


def project_organic(molecule):
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        if atom.GetTotalNumHs() > 0:
            return True
        if any(neighbor.GetAtomicNum() in HALOGENS for neighbor in atom.GetNeighbors()):
            return True
    return False


def donor_atom_class(atom):
    atomic_number = atom.GetAtomicNum()
    if atomic_number == 8:
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIsAromatic():
                return "phenol"
            if neighbor.GetAtomicNum() == 6 and any(
                bond.GetBondTypeAsDouble() >= 1.9
                and bond.GetOtherAtom(neighbor).GetAtomicNum() in (8, 16)
                for bond in neighbor.GetBonds()
                if bond.GetOtherAtom(neighbor).GetIdx() != atom.GetIdx()
            ):
                return "carboxylic_acid"
        return "alcohol_OH"
    if atomic_number == 7:
        if atom.GetIsAromatic():
            return "aromatic_NH"
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() in (6, 15, 16) and any(
                bond.GetBondTypeAsDouble() >= 1.9
                and bond.GetOtherAtom(neighbor).GetAtomicNum() in (7, 8, 16)
                for bond in neighbor.GetBonds()
                if bond.GetOtherAtom(neighbor).GetIdx() != atom.GetIdx()
            ):
                return "amide_like_NH"
        return "amine_NH"
    if atomic_number == 16:
        return "thiol_SH"
    return "other"


def donor_counts(molecule):
    counts = {name: 0 for name in DONOR_CLASSES}
    atom_ids = set()
    for feature in FEATURE_FACTORY.GetFeaturesForMol(molecule):
        if feature.GetFamily() == "Donor":
            atom_ids.update(feature.GetAtomIds())
    for index in atom_ids:
        counts[donor_atom_class(molecule.GetAtomWithIdx(index))] += 1
    return counts


def stable_fold(cas, folds, salt):
    digest = hashlib.sha256((salt + cas).encode()).hexdigest()
    return int(digest[:8], 16) % folds


def metrics(errors, case_ids):
    values = np.abs(np.asarray(errors)) * 100.0
    by_case = {}
    for value, cas in zip(values, case_ids):
        by_case.setdefault(cas, []).append(float(value))
    case_mards = [statistics.fmean(group) for group in by_case.values()]
    return {
        "point_median": float(np.median(values)),
        "point_mean": float(np.mean(values)),
        "point_p95": percentile(values.tolist(), 0.95),
        "case_median": statistics.median(case_mards),
        "case_mean": statistics.fmean(case_mards),
        "case_p95": percentile(case_mards, 0.95),
        "lt5": sum(value < 5.0 for value in case_mards) / len(case_mards),
        "lt10": sum(value < 10.0 for value in case_mards) / len(case_mards),
        "bias": float(np.mean(errors)) * 100.0,
    }


def print_metrics(label, errors, case_ids):
    result = metrics(errors, case_ids)
    print(
        f"{label:32} median={result['point_median']:6.3f}% "
        f"mean={result['point_mean']:6.3f}% p95={result['point_p95']:6.3f}% | "
        f"case median={result['case_median']:6.3f}% mean={result['case_mean']:6.3f}% "
        f"p95={result['case_p95']:6.3f}% | bias={result['bias']:+6.3f}% "
        f"<5%={result['lt5']:5.1%} <10%={result['lt10']:5.1%}"
    )
    return result


def feature_vector(point, specification):
    proportions = point["donor_proportions"]
    if specification["size"] == "heavy":
        size = np.log(point["heavy"] / point["hbd"])
    elif specification["size"] == "cpig":
        size = np.log(point["Cpig"] / (R * point["hbd"]))
    else:
        raise ValueError(specification["size"])
    values = [
        size,
        np.log(float(point["hbd"])),
        point["hetero"] / point["heavy"],
        point["Tr"],
        point["Tr"] ** 2,
    ]
    values.extend(proportions)
    if specification["interactions"]:
        values.extend(value * size for value in proportions)
        values.extend(value * point["Tr"] for value in proportions)
        values.extend(value * point["Tr"] ** 2 for value in proportions)
    if specification["omega"]:
        values.extend((point["omega"], point["omega"] * point["Tr"]))
    return values


def fit_predict(X_train, y_train, X_test, alpha, log_target):
    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(X_train)
    transformed_test = scaler.transform(X_test)
    target = np.log(y_train) if log_target else y_train
    model = Ridge(alpha=alpha)
    model.fit(transformed_train, target)
    prediction = model.predict(transformed_test)
    return np.exp(prediction) if log_target else prediction


def select_alpha(points, X, ratio, train_mask, log_target, inner_salt="inner"):
    train_cases = sorted(set(points[index]["cas"] for index in np.flatnonzero(train_mask)))
    scores = []
    for alpha in ALPHAS:
        fold_errors = []
        for fold in range(5):
            validation_cases = {cas for cas in train_cases if stable_fold(cas, 5, inner_salt) == fold}
            validation = train_mask & np.asarray([point["cas"] in validation_cases for point in points])
            inner_train = train_mask & ~validation
            if not np.any(validation) or not np.any(inner_train):
                continue
            predicted_ratio = fit_predict(
                X[inner_train], ratio[inner_train], X[validation], alpha, log_target
            )
            predicted_ratio = np.clip(predicted_ratio, 0.15, 1.25)
            # Cpl_pred/Cpl_true = ratio_true/ratio_predicted.
            errors = ratio[validation] / predicted_ratio - 1.0
            fold_errors.extend(np.abs(errors))
        scores.append((statistics.fmean(fold_errors), alpha))
    return min(scores)[1]


liquid_db = sqlite3.connect(ROOT / "data" / "liquid_heat_capacity.sqlite")
liquid_db.row_factory = sqlite3.Row
critical_db = sqlite3.connect(ROOT / "data" / "effective_criticals.sqlite")
critical_db.row_factory = sqlite3.Row
criticals = {row["CAS"]: row for row in critical_db.execute("SELECT * FROM effective_criticals")}
rows = liquid_db.execute(
    "SELECT cas, name, Tmin_fit_K, Tmax_fit_K FROM canonical_liquid_cp ORDER BY cas"
).fetchall()
resolver_database = ChemicalDatabase(enable_online=False)
resolver = PropertyResolver()

points = []
compound_names = {}
primary_classes = {}
for row in rows:
    cas = row["cas"]
    critical = criticals.get(cas)
    gas = load_bundled_kernel(cas)
    if gas is None and cas == "56-81-5":
        glycerol_props = resolver_database.get(cas, fetch_online=False).to_dict()
        gas = resolver.resolve_ideal_gas_cp_kernel(
            cas,
            glycerol_props,
            allow_online=False,
            allow_estimation=False,
        )
    if cas == GLYCEROL_CAS and gas is None:
        gas = GlycerolIdealGasCp()
    if critical is None or gas is None:
        continue
    molecule = structure(cas)
    if molecule is None:
        continue
    admitted = project_organic(molecule)
    organic = (admitted or cas in STRICT_ORGANIC_CAS) and cas not in STRICT_INORGANIC_CAS
    hbd = int(rdMolDescriptors.CalcNumHBD(molecule))
    if not organic or hbd <= 0:
        continue
    counts = donor_counts(molecule)
    if sum(counts.values()) != hbd:
        raise RuntimeError(f"donor feature mismatch for {cas}")
    Tc = float(critical["Tc"])
    omega = float(critical["omega"])
    Tmin = max(float(row["Tmin_fit_K"]), gas.Tmin, 0.30 * Tc)
    Tmax = min(float(row["Tmax_fit_K"]), gas.Tmax, 0.95 * Tc)
    if Tmax <= Tmin:
        continue
    heavy = int(molecule.GetNumHeavyAtoms())
    hetero = sum(atom.GetAtomicNum() not in (1, 6) for atom in molecule.GetAtoms())
    proportions = tuple(counts[name] / hbd for name in DONOR_CLASSES)
    primary_classes[cas] = max(counts, key=counts.get) if sum(value > 0 for value in counts.values()) == 1 else "mixed"
    compound_names[cas] = row["name"]
    liquid = load_bundled_liquid_kernel(cas)
    for index in range(25):
        T = Tmin + (Tmax - Tmin) * index / 24.0
        Tr = T / Tc
        Cpig = gas.cp(T)
        Cpl = liquid.cp(T)
        u = 1.0 - Tr
        omega_term = omega * (4.2775 + 6.3 * u ** (1.0 / 3.0) / Tr + 0.4355 / u)
        point = {
            "cas": cas,
            "Tr": Tr,
            "omega": omega,
            "Cpig": Cpig,
            "Cpl": Cpl,
            "ratio": Cpig / Cpl,
            "hbd": hbd,
            "heavy": heavy,
            "hetero": hetero,
            "donor_proportions": proportions,
            "Poling_Cpl": Cpig + R * (1.586 + 0.49 / u + omega_term),
            "Bondi_Cpl": Cpig + R * (1.45 + 0.45 / u + omega_term),
        }
        points.append(point)

case_ids = np.asarray([point["cas"] for point in points])
ratio = np.asarray([point["ratio"] for point in points])
true_cpl = np.asarray([point["Cpl"] for point in points])
print(f"Dataset: {len(set(case_ids))} HBD organic compounds; {len(points)} temperature points")
print("Donor classes: " + ", ".join(
    f"{name}={sum(value == name for value in primary_classes.values())}"
    for name in sorted(set(primary_classes.values()))
))

print("\nPublished baselines:")
poling_errors = np.asarray([point["Poling_Cpl"] for point in points]) / true_cpl - 1.0
bondi_errors = np.asarray([point["Bondi_Cpl"] for point in points]) / true_cpl - 1.0
print_metrics("Rowlinson-Poling", poling_errors, case_ids)
print_metrics("Rowlinson-Bondi", bondi_errors, case_ids)

specifications = []
for size in ("heavy", "cpig"):
    for interactions in (False, True):
        for omega in (False, True):
            for log_target in (False, True):
                specifications.append({
                    "size": size,
                    "interactions": interactions,
                    "omega": omega,
                    "log_target": log_target,
                })

results = []
print("\nNested 10-fold CAS-grouped ratio-GC results:")
for specification in specifications:
    X = np.asarray([feature_vector(point, specification) for point in points])
    out_of_fold_ratio = np.empty(len(points))
    selected_alphas = []
    for fold in range(10):
        test_cases = {cas for cas in set(case_ids) if stable_fold(cas, 10, "outer") == fold}
        test = np.asarray([cas in test_cases for cas in case_ids])
        train = ~test
        alpha = select_alpha(points, X, ratio, train, specification["log_target"])
        selected_alphas.append(alpha)
        prediction = fit_predict(
            X[train], ratio[train], X[test], alpha, specification["log_target"]
        )
        out_of_fold_ratio[test] = np.clip(prediction, 0.15, 1.25)
    errors = ratio / out_of_fold_ratio - 1.0
    label = (
        f"{specification['size']} "
        f"{'interact' if specification['interactions'] else 'additive'} "
        f"{'omega' if specification['omega'] else 'no-omega'} "
        f"{'log-ratio' if specification['log_target'] else 'raw-ratio'}"
    )
    result = print_metrics(label, errors, case_ids)
    result.update({"label": label, "errors": errors, "alphas": selected_alphas})
    results.append(result)

best = min(results, key=lambda result: result["case_mean"])
print("\nBest by out-of-fold case-mean MARD:")
print(best["label"])
print(f"Selected alphas by outer fold: {best['alphas']}")
print_metrics("best ratio GC", best["errors"], case_ids)

def grouped_donor_fractions(point, grouping):
    values = point["donor_proportions"]
    alcohol, amide, amine, aromatic_n, acid, phenol, thiol, other = values
    if grouping == "full":
        return (alcohol, amide, amine + aromatic_n, acid, phenol, thiol + other)
    if grouping == "merge_n":
        return (alcohol, amide + amine + aromatic_n, acid, phenol, thiol + other)
    if grouping == "merge_oh":
        return (alcohol + phenol, amide, amine + aromatic_n, acid, thiol + other)
    if grouping == "merge_both":
        return (alcohol + phenol, amide + amine + aromatic_n, acid, thiol + other)
    if grouping == "element":
        return (alcohol + phenol + acid, amide + amine + aromatic_n, thiol + other)
    raise ValueError(grouping)


def reduced_feature_vector(point, specification):
    size = np.log(point["Cpig"] / (R * point["hbd"]))
    Tr = point["Tr"]
    fractions = grouped_donor_fractions(point, specification["grouping"])
    nonreference = fractions[1:]
    values = [size, Tr]
    if specification.get("Tr2"):
        values.append(Tr * Tr)
    if specification.get("log_hbd", specification.get("auxiliary", False)):
        values.append(np.log(float(point["hbd"])))
    if specification.get("hetero_fraction", specification.get("auxiliary", False)):
        values.append(point["hetero"] / point["heavy"])
    values.extend(nonreference)
    if specification.get("size_interactions"):
        values.extend(value * size for value in nonreference)
    if specification.get("Tr_interactions"):
        values.extend(value * Tr for value in nonreference)
    if specification.get("Tr2_interactions"):
        values.extend(value * Tr * Tr for value in nonreference)
    return values


reduction_specs = [
    ("independent-full", dict(grouping="full", Tr2=True, auxiliary=True, size_interactions=True, Tr_interactions=True, Tr2_interactions=True)),
    ("no-auxiliary", dict(grouping="full", Tr2=True, auxiliary=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=True)),
    ("generic-quadratic", dict(grouping="full", Tr2=True, auxiliary=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("linear-class-interactions", dict(grouping="full", Tr2=False, auxiliary=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("size-interactions-only", dict(grouping="full", Tr2=False, auxiliary=False, size_interactions=True, Tr_interactions=False, Tr2_interactions=False)),
    ("temperature-interactions-only", dict(grouping="full", Tr2=False, auxiliary=False, size_interactions=False, Tr_interactions=True, Tr2_interactions=False)),
    ("additive", dict(grouping="full", Tr2=False, auxiliary=False, size_interactions=False, Tr_interactions=False, Tr2_interactions=False)),
    ("merge-N-linear", dict(grouping="merge_n", Tr2=False, auxiliary=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("merge-OH-linear", dict(grouping="merge_oh", Tr2=False, auxiliary=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("merge-both-linear", dict(grouping="merge_both", Tr2=False, auxiliary=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("merge-both-quadratic", dict(grouping="merge_both", Tr2=True, auxiliary=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=True)),
    ("element-linear", dict(grouping="element", Tr2=False, auxiliary=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("full-size-generic-Tr2", dict(grouping="full", Tr2=True, auxiliary=False, size_interactions=True, Tr_interactions=False, Tr2_interactions=False)),
    ("drop-log-HBD", dict(grouping="full", Tr2=True, log_hbd=False, hetero_fraction=True, size_interactions=True, Tr_interactions=True, Tr2_interactions=True)),
    ("drop-hetero-fraction", dict(grouping="full", Tr2=True, log_hbd=True, hetero_fraction=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=True)),
    ("aux-no-class-Tr2", dict(grouping="full", Tr2=True, auxiliary=True, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("aux-no-class-Tr", dict(grouping="full", Tr2=True, auxiliary=True, size_interactions=True, Tr_interactions=False, Tr2_interactions=True)),
    ("aux-no-class-size", dict(grouping="full", Tr2=True, auxiliary=True, size_interactions=False, Tr_interactions=True, Tr2_interactions=True)),
    ("aux-linear", dict(grouping="full", Tr2=False, auxiliary=True, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("merge-N-full", dict(grouping="merge_n", Tr2=True, auxiliary=True, size_interactions=True, Tr_interactions=True, Tr2_interactions=True)),
    ("merge-N-no-class-Tr2", dict(grouping="merge_n", Tr2=True, auxiliary=True, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("merge-OH-full", dict(grouping="merge_oh", Tr2=True, auxiliary=True, size_interactions=True, Tr_interactions=True, Tr2_interactions=True)),
    ("merge-both-full", dict(grouping="merge_both", Tr2=True, auxiliary=True, size_interactions=True, Tr_interactions=True, Tr2_interactions=True)),
    ("logHBD-linear", dict(grouping="full", Tr2=False, log_hbd=True, hetero_fraction=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("logHBD-generic-quadratic", dict(grouping="full", Tr2=True, log_hbd=True, hetero_fraction=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("logHBD-no-class-Tr", dict(grouping="full", Tr2=True, log_hbd=True, hetero_fraction=False, size_interactions=True, Tr_interactions=False, Tr2_interactions=True)),
    ("logHBD-no-class-size", dict(grouping="full", Tr2=True, log_hbd=True, hetero_fraction=False, size_interactions=False, Tr_interactions=True, Tr2_interactions=True)),
    ("merge-N-logHBD-linear", dict(grouping="merge_n", Tr2=False, log_hbd=True, hetero_fraction=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("merge-N-logHBD-quadratic", dict(grouping="merge_n", Tr2=True, log_hbd=True, hetero_fraction=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
    ("merge-N-logHBD-size-only", dict(grouping="merge_n", Tr2=False, log_hbd=True, hetero_fraction=False, size_interactions=True, Tr_interactions=False, Tr2_interactions=False)),
    ("merge-N-logHBD-temperature-only", dict(grouping="merge_n", Tr2=False, log_hbd=True, hetero_fraction=False, size_interactions=False, Tr_interactions=True, Tr2_interactions=False)),
    ("merge-N-logHBD-additive", dict(grouping="merge_n", Tr2=False, log_hbd=True, hetero_fraction=False, size_interactions=False, Tr_interactions=False, Tr2_interactions=False)),
    ("merge-N-linear-no-logHBD", dict(grouping="merge_n", Tr2=False, log_hbd=False, hetero_fraction=False, size_interactions=True, Tr_interactions=True, Tr2_interactions=False)),
]

print("\nStructured feature-count reduction; nested 10-fold CAS-grouped:")
reduction_results = []
for label, specification in reduction_specs:
    X = np.asarray([reduced_feature_vector(point, specification) for point in points])
    predictions = np.empty(len(points))
    selected_alphas = []
    for fold in range(10):
        test_cases = {cas for cas in set(case_ids) if stable_fold(cas, 10, "outer") == fold}
        test = np.asarray([cas in test_cases for cas in case_ids])
        train = ~test
        alpha = select_alpha(points, X, ratio, train, True)
        selected_alphas.append(alpha)
        predictions[test] = np.clip(
            fit_predict(X[train], ratio[train], X[test], alpha, True),
            0.15,
            1.25,
        )
    errors = ratio / predictions - 1.0
    result = print_metrics(f"{label} [{X.shape[1]}]", errors, case_ids)
    result.update({"label": label, "features": X.shape[1], "errors": errors, "alphas": selected_alphas, "specification": specification})
    reduction_results.append(result)

print("\nRepeated grouped-CV stability for compact candidates:")
for compact_label in ("merge-N-logHBD-linear", "merge-N-logHBD-size-only"):
    compact_result = next(result for result in reduction_results if result["label"] == compact_label)
    specification = compact_result["specification"]
    X = np.asarray([reduced_feature_vector(point, specification) for point in points])
    repetitions = []
    for repetition in range(5):
        predictions = np.empty(len(points))
        for fold in range(10):
            test_cases = {
                cas for cas in set(case_ids)
                if stable_fold(cas, 10, f"repeat-outer-{repetition}") == fold
            }
            test = np.asarray([cas in test_cases for cas in case_ids])
            train = ~test
            alpha = select_alpha(
                points, X, ratio, train, True,
                inner_salt=f"repeat-inner-{repetition}",
            )
            predictions[test] = np.clip(
                fit_predict(X[train], ratio[train], X[test], alpha, True),
                0.15,
                1.25,
            )
        result = metrics(ratio / predictions - 1.0, case_ids)
        repetitions.append(result)
        print(
            f"  {compact_label} [{X.shape[1]}] rep={repetition} "
            f"mean={result['case_mean']:.3f}% median={result['case_median']:.3f}% "
            f"p95={result['case_p95']:.3f}% <10%={result['lt10']:.1%}"
        )
    print(
        f"    averages: mean={statistics.fmean(r['case_mean'] for r in repetitions):.3f}% "
        f"median={statistics.fmean(r['case_median'] for r in repetitions):.3f}% "
        f"p95={statistics.fmean(r['case_p95'] for r in repetitions):.3f}% "
        f"<10%={statistics.fmean(r['lt10'] for r in repetitions):.1%}"
    )

print("\nThiol-exclusion hybrid test:")
non_thiol_compounds = {
    cas for cas, donor_class in primary_classes.items() if donor_class != "thiol_SH"
}
non_thiol_mask = np.asarray([cas in non_thiol_compounds for cas in case_ids])
thiol_mask = ~non_thiol_mask
non_thiol_points = [point for point, keep in zip(points, non_thiol_mask) if keep]
non_thiol_ratio = ratio[non_thiol_mask]
non_thiol_case_ids = case_ids[non_thiol_mask]
thiol_case_ids = case_ids[thiol_mask]

def non_thiol_feature_vector(point):
    size = np.log(point["Cpig"] / (R * point["hbd"]))
    Tr = point["Tr"]
    alcohol, amide, amine, aromatic_n, acid, phenol, _, _ = point["donor_proportions"]
    fractions = (alcohol, amide + amine + aromatic_n, acid, phenol)
    nonreference = fractions[1:]
    return [
        size,
        Tr,
        np.log(float(point["hbd"])),
        *nonreference,
        *(value * size for value in nonreference),
        *(value * Tr for value in nonreference),
    ]

X_non_thiol = np.asarray([non_thiol_feature_vector(point) for point in non_thiol_points])
hybrid_repetitions = []
non_thiol_repetitions = []
for repetition in range(5):
    predictions = np.empty(len(non_thiol_points))
    for fold in range(10):
        test_cases = {
            cas for cas in set(non_thiol_case_ids)
            if stable_fold(cas, 10, f"repeat-outer-{repetition}") == fold
        }
        test = np.asarray([cas in test_cases for cas in non_thiol_case_ids])
        train = ~test
        # select_alpha only needs CAS identities and supports an arbitrary point subset.
        alpha = select_alpha(
            non_thiol_points,
            X_non_thiol,
            non_thiol_ratio,
            train,
            True,
            inner_salt=f"repeat-inner-{repetition}",
        )
        predictions[test] = np.clip(
            fit_predict(
                X_non_thiol[train], non_thiol_ratio[train],
                X_non_thiol[test], alpha, True,
            ),
            0.15,
            1.25,
        )
    non_thiol_errors = non_thiol_ratio / predictions - 1.0
    hybrid_errors = np.empty(len(points))
    hybrid_errors[non_thiol_mask] = non_thiol_errors
    hybrid_errors[thiol_mask] = bondi_errors[thiol_mask]
    non_thiol_result = metrics(non_thiol_errors, non_thiol_case_ids)
    hybrid_result = metrics(hybrid_errors, case_ids)
    non_thiol_repetitions.append(non_thiol_result)
    hybrid_repetitions.append(hybrid_result)
    print(
        f"  rep={repetition} O/N-GC[12] mean={non_thiol_result['case_mean']:.3f}% "
        f"median={non_thiol_result['case_median']:.3f}% p95={non_thiol_result['case_p95']:.3f}%; "
        f"hybrid-all mean={hybrid_result['case_mean']:.3f}% "
        f"median={hybrid_result['case_median']:.3f}% p95={hybrid_result['case_p95']:.3f}%"
    )
for label, repeated in (("O/N-GC[12]", non_thiol_repetitions), ("hybrid-all", hybrid_repetitions)):
    print(
        f"    {label}: average mean={statistics.fmean(r['case_mean'] for r in repeated):.3f}% "
        f"median={statistics.fmean(r['case_median'] for r in repeated):.3f}% "
        f"p95={statistics.fmean(r['case_p95'] for r in repeated):.3f}% "
        f"<10%={statistics.fmean(r['lt10'] for r in repeated):.1%}"
    )

print("\nExplicit-alcohol and glycol-class tests (O/N donor population):")

def alcohol_variant_fractions(point, *, explicit_amide=False, glycol=False):
    alcohol, amide, amine, aromatic_n, acid, phenol, _, _ = point["donor_proportions"]
    glycol_fraction = 0.0
    # A glycol/polyol class is admitted only when every donor is alcohol-OH
    # and there are at least two such donors. This excludes diethanolamine.
    if glycol and point["hbd"] >= 2 and abs(alcohol - 1.0) < 1e-12:
        glycol_fraction = alcohol
        alcohol = 0.0
    if explicit_amide:
        values = [alcohol, glycol_fraction, amide, amine + aromatic_n, acid, phenol]
    else:
        values = [alcohol, glycol_fraction, amide + amine + aromatic_n, acid, phenol]
    if not glycol:
        values.pop(1)
    return tuple(values)

def alcohol_variant_features(point, *, coding, explicit_amide=False, glycol=False):
    size = np.log(point["Cpig"] / (R * point["hbd"]))
    Tr = point["Tr"]
    fractions = alcohol_variant_fractions(
        point, explicit_amide=explicit_amide, glycol=glycol
    )
    if coding == "reference-fraction":
        class_values = fractions[1:]
    elif coding == "explicit-fraction":
        class_values = fractions
    elif coding == "explicit-count":
        class_values = tuple(value * point["hbd"] for value in fractions)
    else:
        raise ValueError(coding)
    return [
        size,
        Tr,
        np.log(float(point["hbd"])),
        *class_values,
        *(value * size for value in class_values),
        *(value * Tr for value in class_values),
    ]

alcohol_variants = [
    ("reference alcohol", dict(coding="reference-fraction", explicit_amide=False, glycol=False)),
    ("explicit alcohol fraction", dict(coding="explicit-fraction", explicit_amide=False, glycol=False)),
    ("explicit alcohol count", dict(coding="explicit-count", explicit_amide=False, glycol=False)),
    ("glycol separate", dict(coding="reference-fraction", explicit_amide=False, glycol=True)),
    ("explicit alcohol + glycol", dict(coding="explicit-fraction", explicit_amide=False, glycol=True)),
    ("explicit count + glycol", dict(coding="explicit-count", explicit_amide=False, glycol=True)),
    ("amide separate control", dict(coding="reference-fraction", explicit_amide=True, glycol=False)),
    ("amide + glycol separate", dict(coding="reference-fraction", explicit_amide=True, glycol=True)),
]
final_repeated_errors = []
for variant_label, variant in alcohol_variants:
    X_variant = np.asarray([
        alcohol_variant_features(point, **variant) for point in non_thiol_points
    ])
    repeated = []
    glycerol_mards = []
    polyol_mards = []
    for repetition in range(5):
        predictions = np.empty(len(non_thiol_points))
        for fold in range(10):
            test_cases = {
                cas for cas in set(non_thiol_case_ids)
                if stable_fold(cas, 10, f"repeat-outer-{repetition}") == fold
            }
            test = np.asarray([cas in test_cases for cas in non_thiol_case_ids])
            train = ~test
            alpha = select_alpha(
                non_thiol_points,
                X_variant,
                non_thiol_ratio,
                train,
                True,
                inner_salt=f"repeat-inner-{repetition}",
            )
            predictions[test] = np.clip(
                fit_predict(
                    X_variant[train], non_thiol_ratio[train],
                    X_variant[test], alpha, True,
                ),
                0.15,
                1.25,
            )
        errors = non_thiol_ratio / predictions - 1.0
        repeated.append(metrics(errors, non_thiol_case_ids))
        if variant_label == "glycol separate":
            final_repeated_errors.append(errors.copy())
        glycerol = non_thiol_case_ids == GLYCEROL_CAS
        glycerol_mards.append(float(np.mean(np.abs(errors[glycerol]))) * 100.0)
        polyol_cases = {
            cas for cas in set(non_thiol_case_ids)
            if primary_classes[cas] == "alcohol_OH"
            and next(point["hbd"] for point in non_thiol_points if point["cas"] == cas) >= 2
            and abs(next(point["donor_proportions"][0] for point in non_thiol_points if point["cas"] == cas) - 1.0) < 1e-12
        }
        polyol = np.asarray([cas in polyol_cases for cas in non_thiol_case_ids])
        polyol_mards.append(metrics(errors[polyol], non_thiol_case_ids[polyol])["case_mean"])
    print(
        f"  {variant_label:27} [{X_variant.shape[1]:2}] "
        f"mean={statistics.fmean(r['case_mean'] for r in repeated):.3f}% "
        f"median={statistics.fmean(r['case_median'] for r in repeated):.3f}% "
        f"p95={statistics.fmean(r['case_p95'] for r in repeated):.3f}% "
        f"<10%={statistics.fmean(r['lt10'] for r in repeated):.1%} "
        f"glycerol={statistics.fmean(glycerol_mards):.3f}% "
        f"polyols={statistics.fmean(polyol_mards):.3f}%"
    )

print("\nFinal 15-predictor polyol-GC fit:")
final_variant = dict(coding="reference-fraction", explicit_amide=False, glycol=True)
X_final = np.asarray([
    alcohol_variant_features(point, **final_variant) for point in non_thiol_points
])
final_feature_names = [
    "s",
    "Tr",
    "ln_N_ONH",
    "f_polyol",
    "f_nitrogen",
    "f_acid",
    "f_phenol",
    "f_polyol_times_s",
    "f_nitrogen_times_s",
    "f_acid_times_s",
    "f_phenol_times_s",
    "f_polyol_times_Tr",
    "f_nitrogen_times_Tr",
    "f_acid_times_Tr",
    "f_phenol_times_Tr",
]
if X_final.shape[1] != len(final_feature_names):
    raise RuntimeError("final feature-name mismatch")

# Select one deployment regularization strength from repeated grouped folds on
# the complete training set. The unbiased error report remains the nested
# outer-CV result captured above.
alpha_scores = []
for alpha in ALPHAS:
    alpha_errors = []
    for repetition in range(10):
        for fold in range(5):
            validation_cases = {
                cas for cas in set(non_thiol_case_ids)
                if stable_fold(cas, 5, f"final-alpha-{repetition}") == fold
            }
            validation = np.asarray([
                cas in validation_cases for cas in non_thiol_case_ids
            ])
            train = ~validation
            predicted = np.clip(
                fit_predict(
                    X_final[train], non_thiol_ratio[train],
                    X_final[validation], alpha, True,
                ),
                0.15,
                1.25,
            )
            alpha_errors.extend(np.abs(non_thiol_ratio[validation] / predicted - 1.0))
    alpha_scores.append((statistics.fmean(alpha_errors), alpha))
selected_final_alpha = min(alpha_scores)[1]
print("alpha scores (mean absolute Cpl relative error):")
for score, alpha in alpha_scores:
    print(f"  alpha={alpha:g}: {100.0 * score:.6f}%")
print(f"selected alpha={selected_final_alpha:g}")

final_scaler = StandardScaler()
X_final_standardized = final_scaler.fit_transform(X_final)
final_model = Ridge(alpha=selected_final_alpha)
final_model.fit(X_final_standardized, np.log(non_thiol_ratio))
raw_coefficients = final_model.coef_ / final_scaler.scale_
raw_intercept = float(
    final_model.intercept_ - np.dot(raw_coefficients, final_scaler.mean_)
)
print(f"intercept={raw_intercept:.15g}")
for name, coefficient in zip(final_feature_names, raw_coefficients):
    print(f"{name}={coefficient:.15g}")
full_predictions = np.exp(raw_intercept + X_final @ raw_coefficients)
print_metrics(
    "full-fit training residual",
    non_thiol_ratio / full_predictions - 1.0,
    non_thiol_case_ids,
)

first_point_by_cas = {}
for point in points:
    first_point_by_cas.setdefault(point["cas"], point)

def final_error_class(cas):
    primary = primary_classes[cas]
    point = first_point_by_cas[cas]
    if primary == "thiol_SH":
        return "thiol-only (Bondi)"
    alcohol = point["donor_proportions"][0]
    if point["hbd"] >= 2 and abs(alcohol - 1.0) < 1e-12:
        return "polyol"
    if primary == "alcohol_OH":
        return "monohydric alcohol"
    if primary in {"amine_NH", "amide_like_NH", "aromatic_NH"}:
        return "nitrogen donor"
    if primary == "carboxylic_acid":
        return "carboxylic acid"
    if primary == "phenol":
        return "phenol"
    return "mixed O/N donors"

hybrid_repeated_errors = []
for gc_errors in final_repeated_errors:
    hybrid_errors = np.empty(len(points))
    hybrid_errors[non_thiol_mask] = gc_errors
    hybrid_errors[thiol_mask] = bondi_errors[thiol_mask]
    hybrid_repeated_errors.append(hybrid_errors)

print("held-out repeated-CV errors by final class:")
error_classes = [
    "monohydric alcohol",
    "polyol",
    "nitrogen donor",
    "carboxylic acid",
    "phenol",
    "mixed O/N donors",
    "thiol-only (Bondi)",
]
for error_class in ["TOTAL HYBRID", "O/N GC"] + error_classes:
    repeated_metrics = []
    for errors in hybrid_repeated_errors:
        if error_class == "TOTAL HYBRID":
            mask = np.ones(len(points), dtype=bool)
        elif error_class == "O/N GC":
            mask = non_thiol_mask
        else:
            mask = np.asarray([
                final_error_class(cas) == error_class for cas in case_ids
            ])
        repeated_metrics.append(metrics(errors[mask], case_ids[mask]))
    case_count = len(set(case_ids[mask]))
    print(
        f"  {error_class:23} n={case_count:2} "
        f"mean={statistics.fmean(r['case_mean'] for r in repeated_metrics):.6f}% "
        f"median={statistics.fmean(r['case_median'] for r in repeated_metrics):.6f}% "
        f"p95={statistics.fmean(r['case_p95'] for r in repeated_metrics):.6f}% "
        f"bias={statistics.fmean(r['bias'] for r in repeated_metrics):+.6f}% "
        f"lt5={statistics.fmean(r['lt5'] for r in repeated_metrics):.6f} "
        f"lt10={statistics.fmean(r['lt10'] for r in repeated_metrics):.6f}"
    )

print("\nRepeated nested grouped-CV stability for the two leading log-ratio models:")
for use_omega in (False, True):
    specification = {
        "size": "cpig",
        "interactions": True,
        "omega": use_omega,
        "log_target": True,
    }
    X = np.asarray([feature_vector(point, specification) for point in points])
    repeated_results = []
    for repetition in range(5):
        predictions = np.empty(len(points))
        alphas = []
        for fold in range(10):
            test_cases = {
                cas for cas in set(case_ids)
                if stable_fold(cas, 10, f"repeat-outer-{repetition}") == fold
            }
            test = np.asarray([cas in test_cases for cas in case_ids])
            train = ~test
            alpha = select_alpha(
                points,
                X,
                ratio,
                train,
                True,
                inner_salt=f"repeat-inner-{repetition}",
            )
            alphas.append(alpha)
            predictions[test] = np.clip(
                fit_predict(X[train], ratio[train], X[test], alpha, True),
                0.15,
                1.25,
            )
        errors = ratio / predictions - 1.0
        result = metrics(errors, case_ids)
        repeated_results.append(result)
        print(
            f"  {'omega' if use_omega else 'no-omega':8} repetition={repetition} "
            f"mean={result['case_mean']:.3f}% median={result['case_median']:.3f}% "
            f"p95={result['case_p95']:.3f}% <10%={result['lt10']:.1%} "
            f"alphas={alphas}"
        )
    for metric in ("case_mean", "case_median", "case_p95", "lt10"):
        values = [result[metric] for result in repeated_results]
        scale = 100.0 if metric == "lt10" else 1.0
        suffix = "%" if metric != "lt10" else " percentage-points"
        print(
            f"    {metric}: mean={statistics.fmean(values) * scale:.3f}{suffix}; "
            f"range={min(values) * scale:.3f}..{max(values) * scale:.3f}{suffix}"
        )

print("\nBest-model case MARD by primary donor class:")
for donor_class in sorted(set(primary_classes.values())):
    mask = np.asarray([primary_classes[cas] == donor_class for cas in case_ids])
    print(f"  {donor_class} n={sum(value == donor_class for value in primary_classes.values())}")
    print_metrics("    ratio GC", best["errors"][mask], case_ids[mask])
    print_metrics("    Bondi", bondi_errors[mask], case_ids[mask])

case_comparison = []
for cas in sorted(set(case_ids)):
    mask = case_ids == cas
    gc_mard = float(np.mean(np.abs(best["errors"][mask]))) * 100.0
    bondi_mard = float(np.mean(np.abs(bondi_errors[mask]))) * 100.0
    case_comparison.append((gc_mard - bondi_mard, cas, gc_mard, bondi_mard))
print("\nLargest case improvements and regressions versus Bondi:")
for label, ordered in (
    ("improvements", sorted(case_comparison)),
    ("regressions", sorted(case_comparison, reverse=True)),
):
    print(label + ":")
    for difference, cas, gc_mard, bondi_mard in ordered[:15]:
        print(
            f"  {cas:12} {compound_names[cas][:28]:28} {primary_classes[cas]:18} "
            f"GC={gc_mard:6.2f}% Bondi={bondi_mard:6.2f}% delta={difference:+6.2f} pp"
        )
