#!/usr/bin/env python3
"""Compare corrected Baroncini and revised Govender on refrigerants.

The population combines the maintained Perry halocarbon probe with the mixed
and Perry-absent non-mixed CoolProp probes.  Govender uses a Perry normal
boiling point for Perry compounds and a CoolProp 101325 Pa saturation
temperature for CoolProp compounds.  Accuracy comparisons use only compounds
and temperature states supported by both methods; Baroncini's full coverage is
reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import CoolProp
import CoolProp.CoolProp as CP
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import govender_method  # noqa: E402
from perry_properties import PerryPropertyLibrary  # noqa: E402
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    conductivity_reference,
)
from scripts.thermal_conductivity.liquid.benchmark_baroncini_perry import (  # noqa: E402
    SAMPLE_POINTS,
    baroncini_conductivity_W_m_K,
)
from scripts.thermal_conductivity.liquid.benchmark_govender_perry import (  # noqa: E402
    boiling_point,
)
from scripts.thermal_conductivity.liquid.probe_baroncini_refrigerants import (  # noqa: E402
    metrics,
)


NORMAL_BOILING_PRESSURE_PA = 101325.0
DEFAULT_CONDUCTIVITY_DATABASE = ROOT / "data" / "perry_thermal_conductivity.json"
RESULTS = ROOT / "scripts" / "thermal_conductivity" / "liquid" / "results"
DEFAULT_PERRY_ARTIFACT = RESULTS / "baroncini_refrigerant_probe.json"
DEFAULT_MIXED_COOLPROP_ARTIFACT = (
    RESULTS / "baroncini_mixed_halogen_coolprop8_probe.json"
)
DEFAULT_NONMIXED_COOLPROP_ARTIFACT = (
    RESULTS / "baroncini_nonmixed_halogen_coolprop8_probe.json"
)
DEFAULT_OUTPUT = RESULTS / "baroncini_vs_govender_refrigerants.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_errors(
    temperatures_and_references: list[tuple[float, float]],
    *,
    baroncini_inputs: dict,
    govender: govender_method.GovenderResult,
) -> tuple[list[float], list[float]]:
    baroncini_errors = []
    govender_errors = []
    for temperature, reference in temperatures_and_references:
        baroncini = baroncini_conductivity_W_m_K(
            temperature,
            normal_boiling_temperature_K=1.0,
            **baroncini_inputs,
        )
        govender_value = govender.conductivity_W_m_K(temperature)
        baroncini_errors.append(100.0 * (baroncini / reference - 1.0))
        govender_errors.append(100.0 * (govender_value / reference - 1.0))
    return baroncini_errors, govender_errors


def evaluate_perry(
    artifact: dict,
    conductivity_database: dict,
    perry: PerryPropertyLibrary,
) -> tuple[list[dict], list[dict]]:
    successful = []
    excluded = []
    for item in artifact["compounds"]:
        cas = str(item["cas"])
        entry = conductivity_database[cas]
        states = []
        for curve in entry["liquid_thermal_conductivity"]:
            for temperature in np.linspace(
                float(curve["T_min_K"]),
                float(curve["T_max_K"]),
                SAMPLE_POINTS,
            ):
                temperature = float(temperature)
                if temperature > float(item["Tc_K"]) * (1.0 + 1.0e-10):
                    continue
                states.append((temperature, conductivity_reference(curve, temperature)))
        try:
            tb_K, tb_method = boiling_point(perry, cas)
            estimate = govender_method.estimate(
                item["smiles"], tb=tb_K, local_refits=True
            )
            baroncini_errors, govender_errors = method_errors(
                states,
                baroncini_inputs={
                    "critical_temperature_K": float(item["Tc_K"]),
                    "molecular_weight_g_mol": float(item["molecular_weight_g_mol"]),
                    "A": float(item["A_used"]),
                    "a": 0.0,
                    "b": 0.5,
                    "c": -0.167,
                },
                govender=estimate,
            )
        except (ValueError, TypeError, OverflowError, govender_method.GovenderError) as exc:
            excluded.append(
                {"source": "perry", "cas": cas, "name": item["name"], "reason": str(exc)}
            )
            continue
        successful.append(
            {
                "source": "perry",
                "cas": cas,
                "name": item["name"],
                "formula": item["formula"],
                "smiles": item["smiles"],
                "Tb_K": tb_K,
                "Tb_method": tb_method,
                "baroncini": metrics(baroncini_errors),
                "govender": metrics(govender_errors),
                "baroncini_errors": baroncini_errors,
                "govender_errors": govender_errors,
            }
        )
    return successful, excluded


def evaluate_coolprop(artifact: dict, source: str) -> tuple[list[dict], list[dict]]:
    runtime_version = str(CoolProp.__version__)
    artifact_version = str(artifact["source"]["version"])
    if runtime_version != artifact_version:
        raise ValueError(
            f"CoolProp artifact version {artifact_version} does not match runtime {runtime_version}"
        )
    successful = []
    excluded = []
    for item in artifact["compounds"]:
        fluid = str(item["fluid"])
        states = []
        for temperature in np.linspace(
            float(item["Tmin_K"]), float(item["Tmax_K"]), SAMPLE_POINTS
        ):
            temperature = float(temperature)
            reference = float(
                CP.PropsSI("CONDUCTIVITY", "T", temperature, "Q", 0.0, fluid)
            )
            states.append((temperature, reference))
        try:
            tb_K = float(
                CP.PropsSI(
                    "T", "P", NORMAL_BOILING_PRESSURE_PA, "Q", 0.0, fluid
                )
            )
            estimate = govender_method.estimate(
                item["smiles"], tb=tb_K, local_refits=True
            )
            baroncini_errors, govender_errors = method_errors(
                states,
                baroncini_inputs={
                    "critical_temperature_K": float(item["Tc_K"]),
                    "molecular_weight_g_mol": float(item["molecular_weight_g_mol"]),
                    "A": float(item["A_used"]),
                    "a": 0.0,
                    "b": 0.5,
                    "c": -0.167,
                },
                govender=estimate,
            )
        except (ValueError, TypeError, OverflowError, RuntimeError, govender_method.GovenderError) as exc:
            excluded.append(
                {"source": source, "cas": item["cas"], "name": fluid, "reason": str(exc)}
            )
            continue
        successful.append(
            {
                "source": source,
                "cas": item["cas"],
                "name": fluid,
                "formula": item["formula"],
                "smiles": item["smiles"],
                "Tb_K": tb_K,
                "Tb_method": "coolprop_HEOS_normal_boiling_point",
                "baroncini": metrics(baroncini_errors),
                "govender": metrics(govender_errors),
                "baroncini_errors": baroncini_errors,
                "govender_errors": govender_errors,
            }
        )
    return successful, excluded


def summarized_population(records: list[dict]) -> dict:
    baroncini_errors = [
        error for record in records for error in record["baroncini_errors"]
    ]
    govender_errors = [
        error for record in records for error in record["govender_errors"]
    ]
    return {
        "compounds": len(records),
        "baroncini": metrics(baroncini_errors),
        "govender": metrics(govender_errors),
        "baroncini_better_compounds": sum(
            record["baroncini"]["mape_percent"]
            < record["govender"]["mape_percent"]
            for record in records
        ),
        "govender_better_compounds": sum(
            record["govender"]["mape_percent"]
            < record["baroncini"]["mape_percent"]
            for record in records
        ),
    }


def run_comparison(
    conductivity_path: Path,
    perry_artifact_path: Path,
    mixed_artifact_path: Path,
    nonmixed_artifact_path: Path,
) -> dict:
    conductivity_database = json.loads(conductivity_path.read_text())["chemicals"]
    perry_artifact = json.loads(perry_artifact_path.read_text())
    mixed_artifact = json.loads(mixed_artifact_path.read_text())
    nonmixed_artifact = json.loads(nonmixed_artifact_path.read_text())
    records = []
    exclusions = []
    source_records, source_exclusions = evaluate_perry(
        perry_artifact, conductivity_database, PerryPropertyLibrary()
    )
    records.extend(source_records)
    exclusions.extend(source_exclusions)
    for artifact, source in (
        (mixed_artifact, "coolprop_mixed"),
        (nonmixed_artifact, "coolprop_nonmixed_perry_absent"),
    ):
        source_records, source_exclusions = evaluate_coolprop(artifact, source)
        records.extend(source_records)
        exclusions.extend(source_exclusions)

    seen = set()
    duplicates = []
    unique_records = []
    for record in records:
        if record["cas"] in seen:
            duplicates.append(record)
            continue
        seen.add(record["cas"])
        unique_records.append(record)
    source_vectors = defaultdict(list)
    for record in unique_records:
        source_vectors[record["source"]].append(record)
    overall_summary = summarized_population(unique_records)
    source_summaries = {
        source: summarized_population(values)
        for source, values in source_vectors.items()
    }
    serialized_records = []
    for record in unique_records:
        serialized = {
            key: value
            for key, value in record.items()
            if key not in {"baroncini_errors", "govender_errors"}
        }
        serialized["mape_difference_govender_minus_baroncini"] = (
            record["govender"]["mape_percent"]
            - record["baroncini"]["mape_percent"]
        )
        serialized_records.append(serialized)
    serialized_records.sort(
        key=lambda item: item["mape_difference_govender_minus_baroncini"],
        reverse=True,
    )
    return {
        "coverage": {
            "candidate_compounds": (
                len(perry_artifact["compounds"])
                + len(mixed_artifact["compounds"])
                + len(nonmixed_artifact["compounds"])
            ),
            "common_compounds": len(unique_records),
            "govender_exclusions": len(exclusions),
            "duplicate_cas": len(duplicates),
        },
        "overall_common_set": overall_summary,
        "by_source": source_summaries,
        "exclusions": exclusions,
        "compounds": serialized_records,
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
    overall = result["overall_common_set"]
    lines = [
        "Corrected Baroncini vs revised Govender for refrigerants",
        (
            f"coverage: {coverage['common_compounds']}/{coverage['candidate_compounds']} "
            f"common compounds; Govender exclusions={coverage['govender_exclusions']}"
        ),
        "Baroncini: " + _line(overall["baroncini"]),
        "Govender:  " + _line(overall["govender"]),
        (
            f"compound wins: Baroncini={overall['baroncini_better_compounds']}, "
            f"Govender={overall['govender_better_compounds']}"
        ),
        "",
        "BY SOURCE",
    ]
    for source, values in result["by_source"].items():
        lines.extend(
            (
                f"  {source}: n={values['compounds']}",
                "    Baroncini: " + _line(values["baroncini"]),
                "    Govender:  " + _line(values["govender"]),
            )
        )
    if result["exclusions"]:
        lines.extend(("", "GOVENDER EXCLUSIONS"))
        for item in result["exclusions"]:
            lines.append(
                f"  {item['source']:<32} {item['name']:<24} "
                f"{item['cas']:<12} {item['reason']}"
            )
    lines.extend(("", "LARGEST METHOD DIFFERENCES"))
    for item in result["compounds"][:15]:
        lines.append(
            f"  {item['name']:<28} {item['source']:<30} "
            f"Baroncini={item['baroncini']['mape_percent']:6.2f}% "
            f"Govender={item['govender']['mape_percent']:6.2f}%"
        )
    lines.extend(("", "LARGEST GOVENDER ADVANTAGES"))
    for item in reversed(result["compounds"][-10:]):
        lines.append(
            f"  {item['name']:<28} {item['source']:<30} "
            f"Baroncini={item['baroncini']['mape_percent']:6.2f}% "
            f"Govender={item['govender']['mape_percent']:6.2f}%"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conductivity-database", type=Path, default=DEFAULT_CONDUCTIVITY_DATABASE
    )
    parser.add_argument("--perry-artifact", type=Path, default=DEFAULT_PERRY_ARTIFACT)
    parser.add_argument(
        "--mixed-coolprop-artifact",
        type=Path,
        default=DEFAULT_MIXED_COOLPROP_ARTIFACT,
    )
    parser.add_argument(
        "--nonmixed-coolprop-artifact",
        type=Path,
        default=DEFAULT_NONMIXED_COOLPROP_ARTIFACT,
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_comparison(
        args.conductivity_database,
        args.perry_artifact,
        args.mixed_coolprop_artifact,
        args.nonmixed_coolprop_artifact,
    )
    script = Path(__file__).resolve()
    dependencies = (
        args.perry_artifact,
        args.mixed_coolprop_artifact,
        args.nonmixed_coolprop_artifact,
        ROOT / "govender_method.py",
    )
    result["reproducibility"] = {
        "python": platform.python_version(),
        "coolprop_version": str(CoolProp.__version__),
        "script": str(script.relative_to(ROOT)),
        "script_sha256": _sha256(script),
        "dependency_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in dependencies
        },
        "conductivity_database_sha256": _sha256(args.conductivity_database),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(report(result))


if __name__ == "__main__":
    main()
