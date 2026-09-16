#!/usr/bin/env python3
"""Probe hydrocarbon conductivity sensitivity to Nannoolal critical temperature.

Perry V20 is held fixed.  Perry Tc is compared with Nannoolal Tc using either
Perry's normal boiling point as an anchor or Nannoolal's own estimated boiling
point.  Metrics use only compounds and temperature states supported by all
three cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nannoolal_method  # noqa: E402
from perry_properties import PerryPropertyLibrary  # noqa: E402
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    conductivity_reference,
)
from scripts.thermal_conductivity.liquid.benchmark_govender_perry import (  # noqa: E402
    boiling_point,
)
from scripts.thermal_conductivity.liquid.benchmark_hydrocarbon_model_perry import (  # noqa: E402
    SAMPLE_POINTS,
    conductivity,
)


DEFAULT_BENCHMARK = (
    ROOT
    / "scripts"
    / "thermal_conductivity"
    / "liquid"
    / "results"
    / "hydrocarbon_model_perry_benchmark.json"
)
DEFAULT_CONDUCTIVITY_DATABASE = ROOT / "data" / "perry_thermal_conductivity.json"
DEFAULT_OUTPUT = (
    ROOT
    / "scripts"
    / "thermal_conductivity"
    / "liquid"
    / "results"
    / "hydrocarbon_tc_sensitivity_probe.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(errors: list[float]) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=float)
    absolute = np.abs(values)
    return {
        "points": len(values),
        "mape_percent": float(np.mean(absolute)),
        "median_ape_percent": float(np.median(absolute)),
        "p95_ape_percent": float(np.percentile(absolute, 95.0)),
        "mean_signed_error_percent": float(np.mean(values)),
    }


def carbon_count(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return 0
    return sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())


def evaluate_population(
    rows: list[dict],
    conductivity_database: dict,
    *,
    minimum_carbons: int,
) -> dict:
    selected = [row for row in rows if row["carbon_count"] >= minimum_carbons]
    conductivity_errors = {name: [] for name in ("perry", "anchored", "structure")}
    tc_errors = {name: [] for name in ("anchored", "structure")}
    tb_errors = []
    temperature_bins = {
        "Tr_below_0.6": {name: [] for name in conductivity_errors},
        "Tr_0.6_to_0.8": {name: [] for name in conductivity_errors},
        "Tr_at_least_0.8": {name: [] for name in conductivity_errors},
    }

    for row in selected:
        tc_values = {
            "perry": row["perry_Tc_K"],
            "anchored": row["anchored_Tc_K"],
            "structure": row["structure_Tc_K"],
        }
        common_maximum_temperature = min(tc_values.values())
        tc_errors["anchored"].append(
            100.0 * (row["anchored_Tc_K"] / row["perry_Tc_K"] - 1.0)
        )
        tc_errors["structure"].append(
            100.0 * (row["structure_Tc_K"] / row["perry_Tc_K"] - 1.0)
        )
        tb_errors.append(
            100.0 * (row["structure_Tb_K"] / row["perry_Tb_K"] - 1.0)
        )
        for curve in conductivity_database[row["cas"]][
            "liquid_thermal_conductivity"
        ]:
            for temperature in np.linspace(
                float(curve["T_min_K"]),
                float(curve["T_max_K"]),
                SAMPLE_POINTS,
            ):
                temperature = float(temperature)
                if temperature > common_maximum_temperature * (1.0 + 1.0e-10):
                    continue
                reduced_temperature = temperature / row["perry_Tc_K"]
                if reduced_temperature < 0.6:
                    bin_name = "Tr_below_0.6"
                elif reduced_temperature < 0.8:
                    bin_name = "Tr_0.6_to_0.8"
                else:
                    bin_name = "Tr_at_least_0.8"
                reference = conductivity_reference(curve, temperature)
                for name, critical_temperature in tc_values.items():
                    predicted = conductivity(
                        temperature,
                        molecular_weight_g_mol=row["molecular_weight_g_mol"],
                        critical_temperature_K=critical_temperature,
                        molar_volume_20C_cm3_mol=row["V20_cm3_mol"],
                        straight_chain=row["straight_chain"],
                    )
                    error = 100.0 * (predicted / reference - 1.0)
                    conductivity_errors[name].append(error)
                    temperature_bins[bin_name][name].append(error)

    return {
        "minimum_carbons": minimum_carbons,
        "compounds": len(selected),
        "boiling_temperature": {
            "structure_vs_perry": metrics(tb_errors),
        },
        "critical_temperature": {
            "anchored_vs_perry": metrics(tc_errors["anchored"]),
            "structure_vs_perry": metrics(tc_errors["structure"]),
        },
        "conductivity": {
            name: metrics(errors) for name, errors in conductivity_errors.items()
        },
        "conductivity_by_reduced_temperature": {
            bin_name: {
                name: metrics(errors) for name, errors in values.items()
            }
            for bin_name, values in temperature_bins.items()
        },
    }


def run_probe(benchmark_path: Path, conductivity_path: Path) -> dict:
    benchmark = json.loads(benchmark_path.read_text())
    conductivity_database = json.loads(conductivity_path.read_text())["chemicals"]
    perry = PerryPropertyLibrary()
    rows = []
    exclusions = []
    for compound in benchmark["compounds"]:
        try:
            perry_tb, perry_tb_method = boiling_point(perry, compound["cas"])
            structure = nannoolal_method.estimate(compound["smiles"])
            anchored = nannoolal_method.estimate(compound["smiles"], tb=perry_tb)
            if structure.tb_K is None or structure.tc_K is None or anchored.tc_K is None:
                raise ValueError("Nannoolal Tb or Tc unavailable")
        except (ValueError, nannoolal_method.NannoolalError) as exc:
            exclusions.append(
                {"cas": compound["cas"], "name": compound["name"], "reason": str(exc)}
            )
            continue
        rows.append(
            {
                **compound,
                "carbon_count": carbon_count(compound["smiles"]),
                "perry_Tb_K": perry_tb,
                "perry_Tb_method": perry_tb_method,
                "structure_Tb_K": float(structure.tb_K),
                "perry_Tc_K": compound["Tc_K"],
                "anchored_Tc_K": float(anchored.tc_K),
                "structure_Tc_K": float(structure.tc_K),
            }
        )
    return {
        "all_supported": evaluate_population(
            rows, conductivity_database, minimum_carbons=1
        ),
        "C_at_least_3": evaluate_population(
            rows, conductivity_database, minimum_carbons=3
        ),
        "exclusions": exclusions,
    }


def _line(values: dict) -> str:
    return (
        f"MAPE={values['mape_percent']:.2f}% "
        f"MdAPE={values['median_ape_percent']:.2f}% "
        f"P95={values['p95_ape_percent']:.2f}% "
        f"bias={values['mean_signed_error_percent']:+.2f}%"
    )


def report(result: dict) -> str:
    lines = ["Hydrocarbon conductivity sensitivity to Nannoolal Tc"]
    for key in ("all_supported", "C_at_least_3"):
        population = result[key]
        lines.extend(
            (
                "",
                f"{key}: {population['compounds']} compounds, "
                f"{population['conductivity']['perry']['points']} common states",
                "  Tc anchored by Perry Tb: "
                + _line(population["critical_temperature"]["anchored_vs_perry"]),
                "  Tc fully structure-based: "
                + _line(population["critical_temperature"]["structure_vs_perry"]),
                "  k with Perry Tc:          "
                + _line(population["conductivity"]["perry"]),
                "  k with anchored Tc:        "
                + _line(population["conductivity"]["anchored"]),
                "  k with structure Tc:       "
                + _line(population["conductivity"]["structure"]),
            )
        )
    if result["exclusions"]:
        lines.extend(("", "EXCLUSIONS"))
        for item in result["exclusions"]:
            lines.append(f"  {item['name']} ({item['cas']}): {item['reason']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--conductivity-database",
        type=Path,
        default=DEFAULT_CONDUCTIVITY_DATABASE,
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_probe(args.benchmark, args.conductivity_database)
    script = Path(__file__).resolve()
    result["reproducibility"] = {
        "python": platform.python_version(),
        "script": str(script.relative_to(ROOT)),
        "script_sha256": _sha256(script),
        "benchmark_sha256": _sha256(args.benchmark),
        "conductivity_database_sha256": _sha256(args.conductivity_database),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(report(result))


if __name__ == "__main__":
    main()
