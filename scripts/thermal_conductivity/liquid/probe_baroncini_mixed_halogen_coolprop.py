#!/usr/bin/env python3
"""Probe Baroncini refrigerant parameters on CoolProp mixed halocarbons.

Mixed halocarbons contain carbon and at least two distinct elements among F,
Cl, Br, and I, with no other heavy elements.  CoolProp HEOS saturated-liquid
conductivity at Q=0 is sampled from the supported triple-point boundary to
Tr=0.95.  The immediate critical region is excluded because CoolProp includes
critical enhancement while the Baroncini equation tends to zero at Tc.  The
authors' A=0.562 correction is applied to R2x (C1H1X3) fluids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import CoolProp
import CoolProp.CoolProp as CP
import numpy as np
from rdkit import Chem, RDLogger


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.thermal_conductivity.liquid.benchmark_baroncini_perry import (  # noqa: E402
    baroncini_conductivity_W_m_K,
)
from scripts.thermal_conductivity.liquid.benchmark_govender_perry import (  # noqa: E402
    local_smiles,
)
from scripts.thermal_conductivity.liquid.probe_baroncini_refrigerants import (  # noqa: E402
    CORRECTED_A,
    uses_authors_A_correction,
)


RDLogger.DisableLog("rdApp.*")

PARAMETERS = {"A": 0.494, "a": 0.0, "b": 0.5, "c": -0.167}
HALOGENS = {9: "F", 17: "Cl", 35: "Br", 53: "I"}
MAXIMUM_REDUCED_TEMPERATURE = 0.95
SAMPLE_POINTS = 101
MINIMUM_VALID_POINTS = 80
DEFAULT_OUTPUT = (
    ROOT
    / "scripts"
    / "thermal_conductivity"
    / "liquid"
    / "results"
    / "baroncini_mixed_halogen_coolprop_probe.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mixed_halocarbon_details(smiles: str) -> dict | None:
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return None
    atomic_numbers = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
    halogen_numbers = atomic_numbers & set(HALOGENS)
    if 6 not in atomic_numbers or len(halogen_numbers) < 2:
        return None
    if atomic_numbers - ({1, 6} | set(HALOGENS)):
        return None
    halogen_counts = Counter(
        HALOGENS[atom.GetAtomicNum()]
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() in HALOGENS
    )
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
    aromatic = any(atom.GetIsAromatic() for atom in molecule.GetAtoms())
    rings = len(molecule.GetRingInfo().AtomRings())
    carbon_multiple_bonds = sum(
        bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        and bond.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
        for bond in molecule.GetBonds()
    )
    if aromatic:
        topology = "aromatic"
    elif rings:
        topology = "cyclic"
    elif carbon_multiple_bonds:
        topology = "unsaturated_acyclic"
    else:
        topology = "saturated_acyclic"
    return {
        "carbon_atoms": carbon_atoms,
        "hydrogen_atoms": sum(
            atom.GetAtomicNum() == 1 for atom in Chem.AddHs(molecule).GetAtoms()
        ),
        "halogen_counts": dict(sorted(halogen_counts.items())),
        "halogen_combination": "+".join(sorted(halogen_counts)),
        "total_halogens": sum(halogen_counts.values()),
        "topology": topology,
        "rings": rings,
        "carbon_multiple_bonds": carbon_multiple_bonds,
    }


def metrics(errors: list[float]) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=float)
    absolute = np.abs(values)
    return {
        "points": len(values),
        "mape_percent": float(np.mean(absolute)),
        "median_ape_percent": float(np.median(absolute)),
        "p95_ape_percent": float(np.percentile(absolute, 95.0)),
        "maximum_ape_percent": float(np.max(absolute)),
        "mean_signed_error_percent": float(np.mean(values)),
        "within_20_percent": float(100.0 * np.mean(absolute <= 20.0)),
    }


def coolprop_identity(fluid: str) -> tuple[str, str, str]:
    cas = CP.get_fluid_param_string(fluid, "CAS").strip()
    formula = CP.get_fluid_param_string(fluid, "formula").strip()
    return cas, formula, local_smiles(cas)


def saturation_band(fluid: str) -> tuple[float, float, float, float]:
    state = CP.AbstractState("HEOS", fluid)
    critical_temperature = float(state.T_critical())
    lower = max(float(state.Tmin()), float(state.Ttriple()))
    upper = min(float(state.Tmax()), MAXIMUM_REDUCED_TEMPERATURE * critical_temperature)
    lower += max(1.0e-6, abs(lower) * 1.0e-9)
    if not all(math.isfinite(value) for value in (lower, upper, critical_temperature)):
        raise ValueError("nonfinite CoolProp temperature boundary")
    if not lower < upper:
        raise ValueError("empty CoolProp saturated-liquid temperature range")
    molecular_weight = float(CP.PropsSI("MOLAR_MASS", fluid)) * 1000.0
    if not math.isfinite(molecular_weight) or molecular_weight <= 0.0:
        raise ValueError("invalid CoolProp molecular weight")
    return lower, upper, critical_temperature, molecular_weight


def run_probe() -> dict:
    exclusions: Counter[str] = Counter()
    excluded_records = []
    all_errors = []
    all_baseline_errors = []
    carbon_errors: defaultdict[str, list[float]] = defaultdict(list)
    combination_errors: defaultdict[str, list[float]] = defaultdict(list)
    compounds = []
    seen_cas = set()
    for fluid in sorted(CP.get_global_param_string("FluidsList").split(",")):
        cas = ""
        formula = ""
        try:
            if CP.get_fluid_param_string(fluid, "pure").strip().lower() != "true":
                continue
            cas, formula, smiles = coolprop_identity(fluid)
            if not cas or cas in seen_cas:
                continue
            details = mixed_halocarbon_details(smiles)
            if details is None:
                continue
            seen_cas.add(cas)
            lower, upper, critical_temperature, molecular_weight = saturation_band(
                fluid
            )
        except (ValueError, RuntimeError) as exc:
            reason = f"identity or constants: {exc}"
            exclusions[reason] += 1
            if cas:
                excluded_records.append(
                    {"fluid": fluid, "cas": cas, "formula": formula, "reason": reason}
                )
            continue

        errors = []
        baseline_errors = []
        correction_applies = uses_authors_A_correction(details)
        corrected_parameters = {
            **PARAMETERS,
            "A": CORRECTED_A if correction_applies else PARAMETERS["A"],
        }
        valid_temperatures = []
        for temperature in np.linspace(lower, upper, SAMPLE_POINTS):
            temperature = float(temperature)
            try:
                reference = float(
                    CP.PropsSI("CONDUCTIVITY", "T", temperature, "Q", 0.0, fluid)
                )
                baseline = baroncini_conductivity_W_m_K(
                    temperature,
                    normal_boiling_temperature_K=1.0,
                    critical_temperature_K=critical_temperature,
                    molecular_weight_g_mol=molecular_weight,
                    **PARAMETERS,
                )
                predicted = baroncini_conductivity_W_m_K(
                    temperature,
                    normal_boiling_temperature_K=1.0,
                    critical_temperature_K=critical_temperature,
                    molecular_weight_g_mol=molecular_weight,
                    **corrected_parameters,
                )
            except (ValueError, RuntimeError, OverflowError):
                continue
            if not all(math.isfinite(value) and value > 0.0 for value in (reference, predicted)):
                continue
            baseline_errors.append(100.0 * (baseline / reference - 1.0))
            errors.append(100.0 * (predicted / reference - 1.0))
            valid_temperatures.append(temperature)
        if len(errors) < MINIMUM_VALID_POINTS:
            reason = "insufficient valid CoolProp conductivity states"
            exclusions[reason] += 1
            excluded_records.append(
                {
                    "fluid": fluid,
                    "cas": cas,
                    "formula": formula,
                    "reason": reason,
                    "valid_states": len(errors),
                }
            )
            continue

        all_errors.extend(errors)
        all_baseline_errors.extend(baseline_errors)
        carbon_bin = "C1" if details["carbon_atoms"] == 1 else "C2_plus"
        carbon_errors[carbon_bin].extend(errors)
        combination_errors[str(details["halogen_combination"])].extend(errors)
        compounds.append(
            {
                "fluid": fluid,
                "cas": cas,
                "formula": formula,
                "smiles": smiles,
                "details": details,
                "Tmin_K": min(valid_temperatures),
                "Tmax_K": max(valid_temperatures),
                "Tc_K": critical_temperature,
                "molecular_weight_g_mol": molecular_weight,
                "authors_A_correction": correction_applies,
                "A_used": corrected_parameters["A"],
                "baseline_errors": metrics(baseline_errors),
                "errors": metrics(errors),
            }
        )
    compounds.sort(key=lambda item: item["errors"]["mape_percent"], reverse=True)
    return {
        "source": {
            "library": "CoolProp",
            "version": str(CoolProp.__version__),
            "backend": "HEOS",
            "property": "saturated-liquid CONDUCTIVITY at Q=0",
            "maximum_reduced_temperature": MAXIMUM_REDUCED_TEMPERATURE,
        },
        "parameters": {
            "general": PARAMETERS,
            "corrected_A": CORRECTED_A,
            "correction_domain": "C1H1X3 (R2x)",
        },
        "coverage": {
            "compounds": len(compounds),
            "sampled_states": len(all_errors),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "excluded_records": excluded_records,
        "baseline_overall": metrics(all_baseline_errors),
        "overall": metrics(all_errors),
        "by_carbon_count": {
            carbon_bin: {
                "compounds": sum(
                    ("C1" if item["details"]["carbon_atoms"] == 1 else "C2_plus")
                    == carbon_bin
                    for item in compounds
                ),
                **metrics(errors),
            }
            for carbon_bin, errors in sorted(carbon_errors.items())
        },
        "by_halogen_combination": {
            combination: {
                "compounds": sum(
                    item["details"]["halogen_combination"] == combination
                    for item in compounds
                ),
                **metrics(errors),
            }
            for combination, errors in sorted(combination_errors.items())
        },
        "compounds": compounds,
    }


def _line(values: dict) -> str:
    return (
        f"MAPE={values['mape_percent']:6.2f}% "
        f"MdAPE={values['median_ape_percent']:6.2f}% "
        f"P95={values['p95_ape_percent']:6.2f}% "
        f"bias={values['mean_signed_error_percent']:+6.2f}%"
    )


def report(result: dict) -> str:
    coverage = result["coverage"]
    lines = [
        "Baroncini mixed-halogen refrigerants vs CoolProp saturated-liquid conductivity",
        f"coverage: {coverage['compounds']} compounds, {coverage['sampled_states']} states",
        "baseline A=0.494: " + _line(result["baseline_overall"]),
        "authors' corrected A: " + _line(result["overall"]),
        "",
        "BY CARBON COUNT",
    ]
    for carbon_bin, values in result["by_carbon_count"].items():
        lines.append(
            f"  {carbon_bin:<8} n={values['compounds']:3d} {_line(values)}"
        )
    lines.extend(("", "BY HALOGEN COMBINATION"))
    for combination, values in result["by_halogen_combination"].items():
        lines.append(
            f"  {combination:<8} n={values['compounds']:3d} {_line(values)}"
        )
    lines.extend(("", "COMPOUNDS, WORST FIRST"))
    for index, compound in enumerate(result["compounds"], 1):
        values = compound["errors"]
        baseline = compound["baseline_errors"]
        lines.append(
            f"  {index:2d}. {compound['fluid']:<12} {compound['cas']:<12} "
            f"{compound['formula']:<18} "
            f"{str(compound['details']['halogen_counts']):<20} "
            f"MAPE={baseline['mape_percent']:6.2f}% -> {values['mape_percent']:6.2f}% "
            f"A={compound['A_used']:.3f}"
        )
    if coverage["exclusions"]:
        lines.extend(("", "EXCLUSIONS"))
        for reason, count in coverage["exclusions"].items():
            lines.append(f"  {count:3d} {reason}")
        for item in result["excluded_records"]:
            lines.append(
                f"      {item['fluid']:<12} {item['cas']:<12} "
                f"{item['formula']:<18} {item['reason']}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_probe()
    script = Path(__file__).resolve()
    base_script = ROOT / "scripts" / "thermal_conductivity" / "liquid" / "benchmark_baroncini_perry.py"
    refrigerant_script = (
        ROOT
        / "scripts"
        / "thermal_conductivity"
        / "liquid"
        / "probe_baroncini_refrigerants.py"
    )
    result["reproducibility"] = {
        "python": platform.python_version(),
        "script": str(script.relative_to(ROOT)),
        "script_sha256": _sha256(script),
        "base_script_sha256": _sha256(base_script),
        "refrigerant_script_sha256": _sha256(refrigerant_script),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(report(result))


if __name__ == "__main__":
    main()
