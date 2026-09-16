#!/usr/bin/env python3
"""Evaluate CoolProp non-mixed halocarbons absent from the Perry probe.

Eligible fluids contain carbon and exactly one halogen element type, contain no
other heavy elements, and have a CAS number absent from the maintained Perry
Baroncini refrigerant artifact.  CoolProp HEOS saturated-liquid conductivity
is sampled at Q=0 through Tr=0.95.  Both the general A=0.494 prediction and the
authors' A=0.562 correction for non-mixed C1 refrigerants are retained.
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
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.thermal_conductivity.liquid.benchmark_baroncini_perry import (  # noqa: E402
    baroncini_conductivity_W_m_K,
)
from scripts.thermal_conductivity.liquid.benchmark_govender_perry import (  # noqa: E402
    local_smiles,
)
from scripts.thermal_conductivity.liquid.probe_baroncini_mixed_halogen_coolprop import (  # noqa: E402
    MINIMUM_VALID_POINTS,
    SAMPLE_POINTS,
    metrics,
    saturation_band,
)
from scripts.thermal_conductivity.liquid.probe_baroncini_refrigerants import (  # noqa: E402
    CORRECTED_A,
    PARAMETERS,
    classify,
    uses_authors_A_correction,
)


DEFAULT_PERRY_ARTIFACT = (
    ROOT
    / "scripts"
    / "thermal_conductivity"
    / "liquid"
    / "results"
    / "baroncini_refrigerant_probe.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "scripts"
    / "thermal_conductivity"
    / "liquid"
    / "results"
    / "baroncini_nonmixed_halogen_coolprop_probe.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_probe(perry_artifact: Path) -> dict:
    perry_data = json.loads(perry_artifact.read_text())
    perry_cas = {str(item["cas"]) for item in perry_data["compounds"]}
    exclusions: Counter[str] = Counter()
    excluded_records = []
    all_errors = []
    all_baseline_errors = []
    family_errors: defaultdict[str, list[float]] = defaultdict(list)
    carbon_errors: defaultdict[str, list[float]] = defaultdict(list)
    compounds = []
    seen_cas = set()
    for fluid in sorted(CP.get_global_param_string("FluidsList").split(",")):
        cas = ""
        formula = ""
        try:
            if CP.get_fluid_param_string(fluid, "pure").strip().lower() != "true":
                continue
            cas = CP.get_fluid_param_string(fluid, "CAS").strip()
            formula = CP.get_fluid_param_string(fluid, "formula").strip()
            if not cas or cas in seen_cas or cas in perry_cas:
                continue
            smiles = local_smiles(cas)
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
                continue
            details = classify(molecule)
            if details is None or len(details["halogen_counts"]) != 1:
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

        correction_applies = uses_authors_A_correction(details)
        corrected_parameters = {
            **PARAMETERS,
            "A": CORRECTED_A if correction_applies else PARAMETERS["A"],
        }
        errors = []
        baseline_errors = []
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
            if not all(
                math.isfinite(value) and value > 0.0
                for value in (reference, baseline, predicted)
            ):
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
        family = str(details["halogen_family"])
        carbon_bin = "C1" if details["carbon_atoms"] == 1 else "C2_plus"
        family_errors[family].extend(errors)
        carbon_errors[carbon_bin].extend(errors)
        compounds.append(
            {
                "fluid": fluid,
                "cas": cas,
                "formula": formula,
                "smiles": smiles,
                "classification": details,
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
            "maximum_reduced_temperature": 0.95,
        },
        "parameters": {
            "general": PARAMETERS,
            "corrected_A": CORRECTED_A,
            "correction_domain": "non-mixed C1 with one halogen element type",
        },
        "coverage": {
            "compounds": len(compounds),
            "sampled_states": len(all_errors),
            "perry_compounds_excluded": len(perry_cas),
            "exclusions": dict(sorted(exclusions.items())),
        },
        "excluded_records": excluded_records,
        "baseline_overall": metrics(all_baseline_errors),
        "overall": metrics(all_errors),
        "by_halogen_family": {
            family: {
                "compounds": sum(
                    item["classification"]["halogen_family"] == family
                    for item in compounds
                ),
                **metrics(errors),
            }
            for family, errors in sorted(family_errors.items())
        },
        "by_carbon_count": {
            carbon_bin: {
                "compounds": sum(
                    (
                        "C1"
                        if item["classification"]["carbon_atoms"] == 1
                        else "C2_plus"
                    )
                    == carbon_bin
                    for item in compounds
                ),
                **metrics(errors),
            }
            for carbon_bin, errors in sorted(carbon_errors.items())
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
        "CoolProp non-mixed halocarbons absent from the Perry Baroncini probe",
        f"coverage: {coverage['compounds']} compounds, {coverage['sampled_states']} states",
        "baseline A=0.494: " + _line(result["baseline_overall"]),
        "authors' corrected A: " + _line(result["overall"]),
        "",
        "BY HALOGEN FAMILY",
    ]
    for family, values in result["by_halogen_family"].items():
        lines.append(f"  {family:<8} n={values['compounds']:3d} {_line(values)}")
    lines.extend(("", "BY CARBON COUNT"))
    for carbon_bin, values in result["by_carbon_count"].items():
        lines.append(
            f"  {carbon_bin:<8} n={values['compounds']:3d} {_line(values)}"
        )
    lines.extend(("", "COMPOUNDS, WORST FIRST"))
    for index, compound in enumerate(result["compounds"], 1):
        baseline = compound["baseline_errors"]
        values = compound["errors"]
        lines.append(
            f"  {index:2d}. {compound['fluid']:<12} {compound['cas']:<12} "
            f"{compound['formula']:<20} "
            f"MAPE={baseline['mape_percent']:6.2f}% -> {values['mape_percent']:6.2f}% "
            f"A={compound['A_used']:.3f}"
        )
    if result["excluded_records"]:
        lines.extend(("", "ELIGIBLE FLUIDS WITHOUT USABLE CONDUCTIVITY"))
        for item in result["excluded_records"]:
            lines.append(
                f"  {item['fluid']:<12} {item['cas']:<12} "
                f"{item['formula']:<20} {item['reason']}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--perry-artifact", type=Path, default=DEFAULT_PERRY_ARTIFACT
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_probe(args.perry_artifact)
    script = Path(__file__).resolve()
    dependencies = (
        ROOT
        / "scripts"
        / "thermal_conductivity"
        / "liquid"
        / "probe_baroncini_refrigerants.py",
        ROOT
        / "scripts"
        / "thermal_conductivity"
        / "liquid"
        / "probe_baroncini_mixed_halogen_coolprop.py",
    )
    result["reproducibility"] = {
        "python": platform.python_version(),
        "script": str(script.relative_to(ROOT)),
        "script_sha256": _sha256(script),
        "dependency_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in dependencies
        },
        "perry_artifact_sha256": _sha256(args.perry_artifact),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(report(result))


if __name__ == "__main__":
    main()
