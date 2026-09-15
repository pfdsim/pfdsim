#!/usr/bin/env python3
"""Probe nonaromatic three-, four-, and five-member ring corrections.

Corrections contribute directly to Govender's structural sums A and B once per
SSSR ring.  Three- and five-member parameters are fitted with equal compound
weight.  Four-member parameters are their arithmetic midpoint and cyclobutane
is therefore a genuine interpolation holdout.  NMP is retained as an external
five-member-ring validation point and is never fitted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
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
)
from scripts.thermal_conductivity.liquid.probe_govender_group_transcriptions import (  # noqa: E402
    DEFAULT_BENCHMARK,
    fit_group,
    load_observations,
)
from scripts.thermal_conductivity.liquid.validate_govender_sparse_candidates import (  # noqa: E402
    DEFAULT_DATA as DEFAULT_LITERATURE_DATA,
    conductivity_W_m_K,
    temperature_K,
)


DEFAULT_GROUP_PROBE = (
    Path(__file__).with_name("results") / "govender_group_transcription_probe.json"
)


def ring_counts(smiles: str) -> Counter[int]:
    molecule = Chem.MolFromSmiles(smiles)
    return Counter(
        len(ring) for ring in molecule.GetRingInfo().AtomRings()
        if not all(molecule.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
    )


def attach_ring_counts(rows: list[dict], benchmark_path: Path) -> None:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    counts = {
        compound["cas"]: ring_counts(compound["smiles"])
        for compound in benchmark["compounds"]
    }
    for row in rows:
        row["ring_counts"] = counts[row["cas"]]


def corrected_errors(
    rows: list[dict], size: int, correction: tuple[float, float]
) -> np.ndarray:
    correction_A, correction_B = correction
    result = []
    for row in rows:
        frequency = row["ring_counts"].get(size, 0)
        sum_A = row["sum_A"] + frequency * correction_A
        sum_B = row["sum_B"] + frequency * correction_B
        predicted = (
            math.exp(math.log(row["n"]) * sum_B / row["n"])
            + sum_A / row["tb"] * (1.0 - row["temperature"] / row["tb"])
        )
        result.append(100.0 * (predicted / row["reference"] - 1.0))
    return np.asarray(result)


def summarize(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "mape_percent": float(np.mean(absolute)),
        "median_ape_percent": float(np.median(absolute)),
        "p95_ape_percent": float(np.percentile(absolute, 95)),
        "mean_signed_error_percent": float(np.mean(values)),
    }


def weighted_residual(
    rows: list[dict], size: int, correction: tuple[float, float]
) -> np.ndarray:
    frequencies = Counter(row["cas"] for row in rows)
    return np.asarray([
        error / math.sqrt(frequencies[row["cas"]])
        for error, row in zip(corrected_errors(rows, size, correction), rows)
    ]) / 100.0


def fit_correction(rows: list[dict], size: int) -> tuple[float, float]:
    result = least_squares(
        lambda values: weighted_residual(
            rows, size, (float(values[0]), float(values[1]))
        ),
        (0.0, 0.0),
    )
    return float(result.x[0]), float(result.x[1])


def correction_probe(
    rows: list[dict], size: int, excluded_names: frozenset[str] = frozenset()
) -> tuple[dict, dict[str, tuple]]:
    all_selected = [row for row in rows if row["ring_counts"].get(size, 0)]
    selected = [row for row in all_selected if row["name"] not in excluded_names]
    cases = sorted({row["cas"] for row in selected})
    fitted = fit_correction(selected, size)
    held_out = []
    held_out_parameters = {}
    for cas in cases:
        training = [row for row in selected if row["cas"] != cas]
        testing = [row for row in selected if row["cas"] == cas]
        correction = fit_correction(training, size)
        held_out_parameters[cas] = correction
        held_out.extend(corrected_errors(testing, size, correction))
    baseline = corrected_errors(selected, size, (0.0, 0.0))
    corrected = corrected_errors(selected, size, fitted)
    compounds = []
    for cas in cases:
        testing = [row for row in selected if row["cas"] == cas]
        compounds.append({
            "cas": cas,
            "name": testing[0]["name"],
            "baseline_mape_percent": float(np.mean(np.abs(
                corrected_errors(testing, size, (0.0, 0.0))
            ))),
            "corrected_mape_percent": float(np.mean(np.abs(
                corrected_errors(testing, size, fitted)
            ))),
        })
    excluded_results = []
    for name in sorted(excluded_names):
        testing = [row for row in all_selected if row["name"] == name]
        if testing:
            excluded_results.append({
                "name": name,
                "baseline": summarize(corrected_errors(
                    testing, size, (0.0, 0.0)
                )),
                "corrected": summarize(corrected_errors(testing, size, fitted)),
            })
    return ({
        "ring_size": size,
        "compounds": len(cases),
        "correction_A": fitted[0],
        "correction_B": fitted[1],
        "baseline": summarize(baseline),
        "fitted": summarize(corrected),
        "leave_one_compound_out": summarize(np.asarray(held_out)),
        "compound_results": compounds,
        "excluded_validation": excluded_results,
    }, held_out_parameters)


def base_parameters(rows: list[dict], cross_validated: bool) -> dict:
    subclasses = {
        53: [row for row in rows if 53 in row["groups"]],
        58: [row for row in rows
             if 58 in row["groups"] and row["name"] != "Quinone"],
        100: [row for row in rows
              if 100 in row["groups"] and row["name"] != "Tetrahydrothiophene"],
    }
    parameters = {}
    for gid, selected in subclasses.items():
        cases = {row["cas"] for row in selected}
        if cross_validated:
            parameters[gid] = {
                cas: fit_group(gid, [row for row in selected if row["cas"] != cas])
                for cas in cases
            }
        else:
            fitted = fit_group(gid, selected)
            parameters[gid] = {cas: fitted for cas in cases}
    return parameters


def overall_errors(
    rows: list[dict],
    ring_parameters: dict[int, tuple[float, float] | dict[str, tuple]],
    *,
    defensible_base: bool,
    cross_validated_base: bool = False,
) -> np.ndarray:
    base = base_parameters(rows, cross_validated_base) if defensible_base else {}
    result = []
    for row in rows:
        sum_A, sum_B = row["sum_A"], row["sum_B"]
        if defensible_base:
            for gid, by_cas in base.items():
                parameters = by_cas.get(row["cas"])
                if parameters is None:
                    continue
                frequency = row["groups"][gid]
                sum_A += frequency * (parameters[0] - gm.CONTRIBUTIONS[gid][0])
                sum_B += frequency * (parameters[1] - gm.CONTRIBUTIONS[gid][1])
        for size, parameters in ring_parameters.items():
            if isinstance(parameters, dict):
                correction = parameters.get(row["cas"])
                if correction is None:
                    continue
            else:
                correction = parameters
            frequency = row["ring_counts"].get(size, 0)
            sum_A += frequency * correction[0]
            sum_B += frequency * correction[1]
        predicted = (
            math.exp(math.log(row["n"]) * sum_B / row["n"])
            + sum_A / row["tb"] * (1.0 - row["temperature"] / row["tb"])
        )
        result.append(100.0 * (predicted / row["reference"] - 1.0))
    return np.asarray(result)


def compound_ranking(rows: list[dict], errors: np.ndarray) -> list[dict]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row, error in zip(rows, errors):
        grouped.setdefault((row["cas"], row["name"]), []).append(float(error))
    result = []
    for (cas, name), values in grouped.items():
        absolute = [abs(value) for value in values]
        result.append({
            "cas": cas,
            "name": name,
            "mape_percent": sum(absolute) / len(absolute),
            "mean_signed_error_percent": sum(values) / len(values),
            "maximum_ape_percent": max(absolute),
        })
    return sorted(result, key=lambda item: item["mape_percent"], reverse=True)


def external_nmp_validation(
    data_path: Path, correction: tuple[float, float]
) -> dict:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    compound = next(
        item for item in data["compounds"]
        if item["name"] == "N-Methyl-2-pyrrolidone"
    )
    estimate = gm.estimate(
        compound["smiles"], tb=float(compound["normal_boiling_point_K"]),
        local_refits=False,
    )
    molecule = Chem.MolFromSmiles(compound["smiles"])
    sum_A = sum(gm.CONTRIBUTIONS[gid][0] * frequency
                for gid, frequency in estimate.groups.items())
    sum_B = sum(gm.CONTRIBUTIONS[gid][1] * frequency
                for gid, frequency in estimate.groups.items())
    baseline_errors = []
    corrected_values = []
    for raw_T, raw_k in compound["observations"]:
        T = temperature_K(float(raw_T), compound["temperature_unit"])
        reference = conductivity_W_m_K(
            float(raw_k), compound["conductivity_unit"]
        )
        baseline = estimate.conductivity_W_m_K(T)
        predicted = (
            math.exp(math.log(molecule.GetNumHeavyAtoms())
                     * (sum_B + correction[1]) / molecule.GetNumHeavyAtoms())
            + (sum_A + correction[0]) / estimate.tb_K * (1.0 - T / estimate.tb_K)
        )
        baseline_errors.append(100.0 * (baseline / reference - 1.0))
        corrected_values.append(100.0 * (predicted / reference - 1.0))
    return {
        "compound": compound["name"],
        "points": len(baseline_errors),
        "baseline": summarize(np.asarray(baseline_errors)),
        "corrected": summarize(np.asarray(corrected_values)),
    }


def build_report(payload: dict) -> str:
    lines = [
        "Govender nonaromatic small-ring correction probe",
        "Corrections add to sum(A) and sum(B) once per SSSR ring.",
        "",
    ]
    for size in (3, 5):
        item = payload["fitted_corrections"][str(size)]
        lines.append(
            f"{size}-member: n={item['compounds']} A={item['correction_A']:.6f} "
            f"B={item['correction_B']:.6f}; MAPE "
            f"{item['baseline']['mape_percent']:.2f}% -> "
            f"{item['fitted']['mape_percent']:.2f}%, "
            f"LOOCV={item['leave_one_compound_out']['mape_percent']:.2f}%"
        )
        for excluded in item["excluded_validation"]:
            lines.append(
                f"  held-out {excluded['name']}: "
                f"{excluded['baseline']['mape_percent']:.2f}% -> "
                f"{excluded['corrected']['mape_percent']:.2f}%"
            )
    item = payload["interpolated_four_member"]
    lines.append(
        f"4-member midpoint: A={item['correction_A']:.6f} "
        f"B={item['correction_B']:.6f}; cyclobutane MAPE "
        f"{item['baseline']['mape_percent']:.2f}% -> "
        f"{item['corrected']['mape_percent']:.2f}%"
    )
    nmp = payload["external_validation"]
    lines.append(
        f"external NMP: {nmp['baseline']['mape_percent']:.2f}% -> "
        f"{nmp['corrected']['mape_percent']:.2f}%"
    )
    lines.extend(("", "COMPLETE PERRY BENCHMARK"))
    for name, result in payload["overall"].items():
        lines.append(
            f"  {name:39s} MAPE={result['mape_percent']:.2f}% "
            f"P95={result['p95_ape_percent']:.2f}% "
            f"bias={result['mean_signed_error_percent']:+.2f}%"
        )
    lines.extend(("", "WORST 20 COMPOUNDS AFTER FIXED REFITS AND RING CORRECTIONS"))
    for rank, item in enumerate(payload["worst_compounds"][:20], 1):
        lines.append(
            f"  {rank:2d}. {item['name'][:34]:34s} {item['cas']:12s} "
            f"MAPE={item['mape_percent']:7.2f}% "
            f"bias={item['mean_signed_error_percent']:+7.2f}%"
        )
    return "\n".join(lines)


def probe(args: argparse.Namespace) -> tuple[dict, str]:
    rows = load_observations(args.benchmark, args.conductivity_database)
    attach_ring_counts(rows, args.benchmark)
    three, three_cv = correction_probe(rows, 3)
    five, five_cv = correction_probe(
        rows, 5, frozenset({"Tetrahydrothiophene"})
    )
    correction_3 = (three["correction_A"], three["correction_B"])
    correction_5 = (five["correction_A"], five["correction_B"])
    correction_4 = tuple(
        (left + right) / 2.0 for left, right in zip(correction_3, correction_5)
    )
    four_rows = [row for row in rows if row["ring_counts"].get(4, 0)]
    fixed_rings = {3: correction_3, 4: correction_4, 5: correction_5}
    five_cv["110-01-0"] = correction_5
    cv_rings = {3: three_cv, 4: correction_4, 5: five_cv}
    fixed_combined_errors = overall_errors(
        rows, fixed_rings, defensible_base=True
    )
    payload = {
        "schema_version": 1,
        "probe": "govender_nonaromatic_small_ring_second_order_corrections",
        "fitted_corrections": {"3": three, "5": five},
        "interpolated_four_member": {
            "compounds": len({row["cas"] for row in four_rows}),
            "correction_A": correction_4[0],
            "correction_B": correction_4[1],
            "baseline": summarize(corrected_errors(four_rows, 4, (0.0, 0.0))),
            "corrected": summarize(corrected_errors(four_rows, 4, correction_4)),
        },
        "external_validation": external_nmp_validation(
            args.literature_data, correction_5
        ),
        "overall": {
            "published": summarize(overall_errors(
                rows, {}, defensible_base=False
            )),
            "published_plus_fixed_rings": summarize(overall_errors(
                rows, fixed_rings, defensible_base=False
            )),
            "published_plus_ring_LOOCV": summarize(overall_errors(
                rows, cv_rings, defensible_base=False
            )),
            "defensible_refits": summarize(overall_errors(
                rows, {}, defensible_base=True
            )),
            "defensible_refits_plus_fixed_rings": summarize(overall_errors(
                rows, fixed_rings, defensible_base=True
            )),
            "defensible_refits_plus_ring_LOOCV": summarize(overall_errors(
                rows, cv_rings, defensible_base=True, cross_validated_base=True
            )),
        },
        "worst_compounds": compound_ranking(rows, fixed_combined_errors),
        "inputs": {
            "benchmark": artifact_record(args.benchmark),
            "conductivity_database": artifact_record(args.conductivity_database),
            "literature_data": artifact_record(args.literature_data),
            "group_probe": artifact_record(args.group_probe),
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
    parser.add_argument("--literature-data", type=Path, default=DEFAULT_LITERATURE_DATA)
    parser.add_argument("--group-probe", type=Path, default=DEFAULT_GROUP_PROBE)
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
