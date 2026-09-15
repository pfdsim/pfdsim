#!/usr/bin/env python3
"""Probe Govender aliphatic-COOH group 53 against Perry liquid curves.

This is a diagnostic, not a refit proposal.  It compares the published
coefficients with two fragmentation reinterpretations, a one-digit table-error
hypothesis, reuse of the aromatic-COOH coefficients, and an unconstrained
two-parameter fit.  It consumes the maintained Perry-Tb benchmark artifact so
the population is exactly the same as the primary benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import govender_method as gm  # noqa: E402
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    DEFAULT_CONDUCTIVITY_DATABASE,
    artifact_record,
    conductivity_reference,
)


DEFAULT_BENCHMARK = (
    Path(__file__).with_name("results") / "govender_perry_tb_benchmark.json"
)
ACID_PATTERN = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")
ALPHA_E_GROUP = {1: 2, 4: 7, 5: 8, 6: 9}


def observations(benchmark_path: Path, conductivity_path: Path) -> list[dict]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    thermal = json.loads(conductivity_path.read_text(encoding="utf-8"))["chemicals"]
    result = []
    for compound in benchmark["compounds"]:
        groups = {int(gid): int(count) for gid, count in compound["groups"].items()}
        molecule = Chem.MolFromSmiles(compound["smiles"])
        n_heavy = molecule.GetNumHeavyAtoms()
        n_carbon = sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
        sum_A = sum(gm.CONTRIBUTIONS[gid][0] * count
                    for gid, count in groups.items())
        sum_B = sum(gm.CONTRIBUTIONS[gid][1] * count
                    for gid, count in groups.items())
        alpha_A_delta = alpha_B_delta = 0.0
        for match in molecule.GetSubstructMatches(ACID_PATTERN):
            carbonyl = molecule.GetAtomWithIdx(match[0])
            alpha = next(
                (atom for atom in carbonyl.GetNeighbors()
                 if atom.GetAtomicNum() == 6),
                None,
            )
            if alpha is None:
                continue
            old_gid = gm._carbon_group(alpha)
            new_gid = ALPHA_E_GROUP.get(old_gid)
            if new_gid is not None:
                alpha_A_delta += (
                    gm.CONTRIBUTIONS[new_gid][0] - gm.CONTRIBUTIONS[old_gid][0]
                )
                alpha_B_delta += (
                    gm.CONTRIBUTIONS[new_gid][1] - gm.CONTRIBUTIONS[old_gid][1]
                )
        row = thermal[compound["cas"]]["liquid_thermal_conductivity"][0]
        for temperature in np.linspace(
            float(row["T_min_K"]), float(row["T_max_K"]),
            benchmark["method"]["sampling_points_per_curve"],
        ):
            result.append({
                "cas": compound["cas"],
                "name": compound["name"],
                "n_heavy": n_heavy,
                "n_carbon": n_carbon,
                "frequency": groups.get(53, 0),
                "tb": float(compound["tb_K"]),
                "sum_A": sum_A,
                "sum_B": sum_B,
                "alpha_A_delta": alpha_A_delta,
                "alpha_B_delta": alpha_B_delta,
                "temperature": float(temperature),
                "reference": conductivity_reference(row, float(temperature)),
            })
    return result


def errors(rows: list[dict], scenario: str,
           fitted: tuple[float, float] | None = None) -> np.ndarray:
    values = []
    published_A, published_B = gm.CONTRIBUTIONS[53]
    aromatic_A, aromatic_B = gm.CONTRIBUTIONS[48]
    for row in rows:
        n = row["n_heavy"]
        frequency = row["frequency"]
        sum_A, sum_B = row["sum_A"], row["sum_B"]
        if scenario == "alpha_carbon_as_e":
            sum_A += row["alpha_A_delta"]
            sum_B += row["alpha_B_delta"]
        elif scenario == "carbon_atoms_for_n":
            n = row["n_carbon"]
        elif scenario == "double_count_group_53":
            sum_A += frequency * published_A
            sum_B += frequency * published_B
        elif scenario == "candidate_B_minus_3.1203":
            sum_B += frequency * (-3.1203 - published_B)
        elif scenario == "aromatic_COOH_coefficients":
            sum_A += frequency * (aromatic_A - published_A)
            sum_B += frequency * (aromatic_B - published_B)
        elif scenario == "least_squares_refit":
            fit_A, fit_B = fitted
            sum_A += frequency * (fit_A - published_A)
            sum_B += frequency * (fit_B - published_B)
        elif scenario != "published":
            raise ValueError(f"unknown scenario {scenario}")
        predicted = (
            math.exp(math.log(n) * sum_B / n)
            + sum_A / row["tb"] * (1.0 - row["temperature"] / row["tb"])
        )
        values.append(100.0 * (predicted / row["reference"] - 1.0))
    return np.asarray(values)


def metrics(values: np.ndarray) -> dict[str, float | int]:
    absolute = np.abs(values)
    return {
        "points": len(values),
        "mape_percent": float(np.mean(absolute)),
        "median_ape_percent": float(np.median(absolute)),
        "p95_ape_percent": float(np.percentile(absolute, 95)),
        "mean_signed_error_percent": float(np.mean(values)),
    }


def fit_group_53(rows: list[dict]) -> tuple[float, float]:
    published_A, published_B = gm.CONTRIBUTIONS[53]

    def residual(parameters):
        return errors(rows, "least_squares_refit", tuple(parameters)) / 100.0

    fit = least_squares(residual, (published_A, published_B))
    return float(fit.x[0]), float(fit.x[1])


def build_report(payload: dict) -> str:
    lines = [
        "Govender group 53 (aliphatic COOH) discrepancy probe",
        f"population: {payload['acid_compounds']} compounds, {payload['acid_points']} points",
        f"published coefficients: A={payload['published']['A']}, B={payload['published']['B']}",
        "",
        "ACID-ONLY RESULTS",
    ]
    for name, result in payload["acid_scenarios"].items():
        lines.append(
            f"  {name:30s} MAPE={result['mape_percent']:7.2f}% "
            f"MdAPE={result['median_ape_percent']:7.2f}% "
            f"P95={result['p95_ape_percent']:7.2f}% "
            f"bias={result['mean_signed_error_percent']:+7.2f}%"
        )
    fit = payload["least_squares_coefficients"]
    fit_cv = payload["least_squares_leave_one_compound_out"]
    lines.extend((
        "",
        f"diagnostic least-squares coefficients: A={fit['A']:.6f}, B={fit['B']:.6f}",
        f"least-squares leave-one-compound-out MAPE: {fit_cv['mape_percent']:.2f}%",
        "",
        "EFFECT ON COMPLETE 261-COMPOUND BENCHMARK",
    ))
    for name, result in payload["overall_scenarios"].items():
        lines.append(
            f"  {name:30s} MAPE={result['mape_percent']:7.2f}% "
            f"P95={result['p95_ape_percent']:7.2f}% "
            f"bias={result['mean_signed_error_percent']:+7.2f}%"
        )
    return "\n".join(lines)


def probe(args: argparse.Namespace) -> tuple[dict, str]:
    all_rows = observations(args.benchmark, args.conductivity_database)
    acid_rows = [row for row in all_rows if row["frequency"]]
    fit = fit_group_53(acid_rows)
    held_out = []
    cases = sorted({row["cas"] for row in acid_rows})
    for cas in cases:
        training = [row for row in acid_rows if row["cas"] != cas]
        testing = [row for row in acid_rows if row["cas"] == cas]
        fold_fit = fit_group_53(training)
        held_out.extend(errors(testing, "least_squares_refit", fold_fit))
    scenarios = (
        "published",
        "alpha_carbon_as_e",
        "carbon_atoms_for_n",
        "double_count_group_53",
        "candidate_B_minus_3.1203",
        "aromatic_COOH_coefficients",
        "least_squares_refit",
    )
    payload = {
        "schema_version": 1,
        "probe": "govender_aliphatic_cooh_group_53",
        "acid_compounds": len({row["cas"] for row in acid_rows}),
        "acid_points": len(acid_rows),
        "published": {"A": gm.CONTRIBUTIONS[53][0], "B": gm.CONTRIBUTIONS[53][1]},
        "least_squares_coefficients": {"A": fit[0], "B": fit[1]},
        "least_squares_leave_one_compound_out": metrics(np.asarray(held_out)),
        "acid_scenarios": {
            name: metrics(errors(acid_rows, name, fit)) for name in scenarios
        },
        "overall_scenarios": {
            name: metrics(errors(all_rows, name, fit))
            for name in (
                "published", "double_count_group_53",
                "candidate_B_minus_3.1203", "aromatic_COOH_coefficients",
            )
        },
        "inputs": {
            "benchmark": artifact_record(args.benchmark),
            "thermal_conductivity": artifact_record(args.conductivity_database),
            "script": artifact_record(Path(__file__)),
            "govender_method": artifact_record(ROOT / "govender_method.py"),
        },
    }
    return payload, build_report(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--conductivity-database", type=Path, default=DEFAULT_CONDUCTIVITY_DATABASE
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, report = probe(args)
    print(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
