#!/usr/bin/env python3
"""Probe tertiary-amine and amine/alcohol corrections for Govender.

The population combines Perry trimethylamine/triethylamine curves with the
literature observations in ``data/tertiary_alcohol_amines.json``. Fits weight
each compound equally. The primary and secondary acyclic amine coefficients
remain fixed. Three nested models are compared by leave-one-compound-out
validation: group 84 alone, group 84 plus one amine/alcohol correction per
molecule, and group 84 plus one correction per amine-OH pair.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
from collections import Counter
from pathlib import Path

import chemicals
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
    load_observations,
)
from scripts.thermal_conductivity.liquid.probe_govender_ring_corrections import (  # noqa: E402
    attach_ring_counts,
    base_parameters,
    compound_ranking,
    summarize,
)


DEFAULT_LITERATURE_DATA = (
    Path(__file__).with_name("data") / "tertiary_alcohol_amines.json"
)
DEFAULT_RING_PROBE = (
    Path(__file__).with_name("results") / "govender_ring_correction_probe.json"
)
IDENTITIES = {
    "tri-n-propylamine": ("102-69-2", "CCCN(CCC)CCC"),
    "tri-n-butylamine": ("102-82-9", "CCCCN(CCCC)CCCC"),
    "monoethanolamine": ("141-43-5", "NCCO"),
    "N,N-dimethylethanolamine": ("108-01-0", "CN(C)CCO"),
    "N,N-diethylethanolamine": ("100-37-8", "CCN(CC)CCO"),
    "diethanolamine": ("111-42-2", "OCCNCCO"),
    "N-methyldiethanolamine": ("105-59-9", "CN(CCO)CCO"),
    "N-ethyldiethanolamine": ("139-87-7", "CCN(CCO)CCO"),
    "triethanolamine": ("102-71-6", "N(CCO)(CCO)CCO"),
}


def triethanolamine_groups() -> dict[int, int]:
    """Govender's ordinary counts before its polyol-domain refusal."""
    return {7: 6, 49: 18, 84: 1, 175: 3}


def load_population(
    benchmark_path: Path,
    conductivity_path: Path,
    literature_path: Path,
) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    for row in load_observations(benchmark_path, conductivity_path):
        if 84 not in row["groups"]:
            continue
        item = dict(row)
        item.update({"alcohol_groups": 0, "source": "Perry Table 2-147"})
        rows.append(item)

    literature = json.loads(literature_path.read_text(encoding="utf-8"))
    boiling_points = {}
    for observation in literature:
        name = observation["chemical"]
        cas, smiles = IDENTITIES[name]
        tb = chemicals.Tb(cas)
        if tb is None:
            raise ValueError(f"normal boiling point unavailable for {name}")
        boiling_points[cas] = {
            "chemical": name,
            "tb_K": float(tb),
            "available_methods": chemicals.Tb_methods(cas),
        }
        molecule = Chem.MolFromSmiles(smiles)
        if name == "triethanolamine":
            groups = triethanolamine_groups()
        else:
            groups = gm.estimate(
                smiles, tb=float(tb), local_refits=False
            ).groups
        sum_A = sum(
            gm.CONTRIBUTIONS[gid][0] * frequency
            for gid, frequency in groups.items()
        )
        sum_B = sum(
            gm.CONTRIBUTIONS[gid][1] * frequency
            for gid, frequency in groups.items()
        )
        rows.append({
            "cas": cas,
            "name": name,
            "groups": groups,
            "n": molecule.GetNumHeavyAtoms(),
            "tb": float(tb),
            "sum_A": sum_A,
            "sum_B": sum_B,
            "temperature": float(observation["temperature_K"]),
            "reference": float(observation["k_W_m_K"]),
            "alcohol_groups": groups.get(175, 0),
            "source": observation["source"],
        })
    return rows, boiling_points


def interaction_frequency(row: dict, rule: str) -> int:
    count = int(row["alcohol_groups"])
    if rule == "none":
        return 0
    if rule == "molecule":
        return int(count > 0)
    if rule == "pair":
        return count
    raise ValueError(f"unknown interaction rule {rule}")


def errors(rows: list[dict], parameters: tuple[float, ...], rule: str) -> np.ndarray:
    A84, B84 = parameters[:2]
    interaction_A, interaction_B = parameters[2:] if len(parameters) == 4 else (0.0, 0.0)
    published_A, published_B = gm.CONTRIBUTIONS[84]
    result = []
    for row in rows:
        group_frequency = row["groups"].get(84, 0)
        interaction = interaction_frequency(row, rule)
        sum_A = (
            row["sum_A"] + group_frequency * (A84 - published_A)
            + interaction * interaction_A
        )
        sum_B = (
            row["sum_B"] + group_frequency * (B84 - published_B)
            + interaction * interaction_B
        )
        predicted = (
            math.exp(math.log(row["n"]) * sum_B / row["n"])
            + sum_A / row["tb"] * (1.0 - row["temperature"] / row["tb"])
        )
        result.append(100.0 * (predicted / row["reference"] - 1.0))
    return np.asarray(result)


def weighted_residual(
    rows: list[dict], parameters: tuple[float, ...], rule: str
) -> np.ndarray:
    point_counts = Counter(row["cas"] for row in rows)
    return np.asarray([
        error / math.sqrt(point_counts[row["cas"]])
        for error, row in zip(errors(rows, parameters, rule), rows)
    ]) / 100.0


def fit(rows: list[dict], rule: str) -> tuple[float, ...]:
    initial = gm.CONTRIBUTIONS[84] if rule == "none" else (*gm.CONTRIBUTIONS[84], 0.0, 0.0)
    result = least_squares(
        lambda values: weighted_residual(rows, tuple(values), rule), initial
    )
    return tuple(map(float, result.x))


def compound_summaries(
    rows: list[dict], values: np.ndarray
) -> list[dict]:
    result = []
    for cas in sorted({row["cas"] for row in rows}):
        selected = [
            (row, error) for row, error in zip(rows, values) if row["cas"] == cas
        ]
        errors_for_compound = [error for _, error in selected]
        result.append({
            "cas": cas,
            "name": selected[0][0]["name"],
            "points": len(selected),
            "mape_percent": float(np.mean(np.abs(errors_for_compound))),
            "mean_signed_error_percent": float(np.mean(errors_for_compound)),
        })
    return result


def summarize_compounds(compounds: list[dict]) -> dict[str, float | int]:
    return {
        "compounds": len(compounds),
        "points": sum(item["points"] for item in compounds),
        "compound_equal_mape_percent": float(np.mean([
            item["mape_percent"] for item in compounds
        ])),
        "compound_equal_bias_percent": float(np.mean([
            item["mean_signed_error_percent"] for item in compounds
        ])),
    }


def cross_validate(rows: list[dict], rule: str) -> tuple[dict, list[dict]]:
    held_out_rows = []
    held_out_errors = []
    for cas in sorted({row["cas"] for row in rows}):
        training = [row for row in rows if row["cas"] != cas]
        testing = [row for row in rows if row["cas"] == cas]
        parameters = fit(training, rule)
        held_out_rows.extend(testing)
        held_out_errors.extend(errors(testing, parameters, rule))
    compounds = compound_summaries(held_out_rows, np.asarray(held_out_errors))
    return summarize_compounds(compounds), compounds


def evaluate_model(rows: list[dict], rule: str) -> dict:
    parameters = fit(rows, rule)
    fitted_compounds = compound_summaries(rows, errors(rows, parameters, rule))
    cv_summary, cv_compounds = cross_validate(rows, rule)
    result = {
        "rule": rule,
        "parameters": {
            "group_84_A": parameters[0],
            "group_84_B": parameters[1],
        },
        "fitted": summarize_compounds(fitted_compounds),
        "leave_one_compound_out": cv_summary,
        "leave_one_compound_out_compounds": cv_compounds,
    }
    if len(parameters) == 4:
        result["parameters"].update({
            "interaction_A": parameters[2],
            "interaction_B": parameters[3],
        })
    return result


def published_summary(rows: list[dict]) -> dict:
    parameters = gm.CONTRIBUTIONS[84]
    compounds = compound_summaries(rows, errors(rows, parameters, "none"))
    return {
        "parameters": {"group_84_A": parameters[0], "group_84_B": parameters[1]},
        "summary": summarize_compounds(compounds),
        "compounds": compounds,
    }


def complete_benchmark_errors(
    rows: list[dict],
    ring_probe_path: Path,
    parameters: tuple[float, ...] | None,
    interaction: bool,
) -> np.ndarray:
    base = base_parameters(rows, False)
    ring_probe = json.loads(ring_probe_path.read_text(encoding="utf-8"))
    three = ring_probe["fitted_corrections"]["3"]
    five = ring_probe["fitted_corrections"]["5"]
    four = ring_probe["interpolated_four_member"]
    rings = {
        3: (three["correction_A"], three["correction_B"]),
        4: (four["correction_A"], four["correction_B"]),
        5: (five["correction_A"], five["correction_B"]),
    }
    result = []
    for row in rows:
        sum_A, sum_B = row["sum_A"], row["sum_B"]
        for gid, by_cas in base.items():
            fitted = by_cas.get(row["cas"])
            if fitted is None:
                continue
            frequency = row["groups"][gid]
            sum_A += frequency * (fitted[0] - gm.CONTRIBUTIONS[gid][0])
            sum_B += frequency * (fitted[1] - gm.CONTRIBUTIONS[gid][1])
        for size, correction in rings.items():
            frequency = row["ring_counts"].get(size, 0)
            sum_A += frequency * correction[0]
            sum_B += frequency * correction[1]
        if parameters is not None:
            frequency = row["groups"].get(84, 0)
            sum_A += frequency * (parameters[0] - gm.CONTRIBUTIONS[84][0])
            sum_B += frequency * (parameters[1] - gm.CONTRIBUTIONS[84][1])
            has_amine = bool(set(row["groups"]) & {80, 81, 82, 84, 93})
            has_alcohol = bool(set(row["groups"]) & {49, 153})
            if interaction and has_amine and has_alcohol:
                sum_A += parameters[2]
                sum_B += parameters[3]
        predicted = (
            math.exp(math.log(row["n"]) * sum_B / row["n"])
            + sum_A / row["tb"] * (1.0 - row["temperature"] / row["tb"])
        )
        result.append(100.0 * (predicted / row["reference"] - 1.0))
    return np.asarray(result)


def build_report(payload: dict) -> str:
    lines = [
        "Govender tertiary-amine and amine/alcohol refit probe",
        "All fits and validation metrics weight each compound equally.",
        "Primary and secondary acyclic amine coefficients remain published.",
        "",
        "PURE TERTIARY AMINES",
    ]
    pure = payload["pure_tertiary_amines"]
    lines.append(
        f"  published: n={pure['published']['summary']['compounds']} "
        f"MAPE={pure['published']['summary']['compound_equal_mape_percent']:.2f}%"
    )
    fitted = pure["refit"]
    lines.append(
        f"  refit A={fitted['parameters']['group_84_A']:.6f} "
        f"B={fitted['parameters']['group_84_B']:.6f}: "
        f"fit MAPE={fitted['fitted']['compound_equal_mape_percent']:.2f}%, "
        f"LOOCV={fitted['leave_one_compound_out']['compound_equal_mape_percent']:.2f}%"
    )
    lines.extend(("", "ALL 11 COMPOUNDS"))
    lines.append(
        f"  published MAPE={payload['all_compounds']['published']['summary']['compound_equal_mape_percent']:.2f}%"
    )
    for name in ("group_84_only", "interaction_once", "interaction_per_oh"):
        result = payload["all_compounds"][name]
        parameters = result["parameters"]
        detail = (
            f"; interaction A={parameters['interaction_A']:.6f} "
            f"B={parameters['interaction_B']:.6f}"
            if "interaction_A" in parameters else ""
        )
        lines.append(
            f"  {name:20s} A84={parameters['group_84_A']:.6f} "
            f"B84={parameters['group_84_B']:.6f}{detail}; "
            f"fit={result['fitted']['compound_equal_mape_percent']:.2f}%, "
            f"LOOCV={result['leave_one_compound_out']['compound_equal_mape_percent']:.2f}%"
        )
    lines.extend(("", "SELECTED INTERACTION-ONCE LOOCV BY COMPOUND"))
    selected = payload["all_compounds"]["interaction_once"]
    for compound in sorted(
        selected["leave_one_compound_out_compounds"],
        key=lambda item: item["mape_percent"], reverse=True,
    ):
        lines.append(
            f"  {compound['name'][:30]:30s} n={compound['points']:3d} "
            f"MAPE={compound['mape_percent']:6.2f}%"
        )
    lines.extend((
        "",
        "AMINE-AMINE INTERACTION",
        "  unavailable: ethylenediamine is the only independent compound",
    ))
    lines.extend(("", "EFFECT ON COMPLETE REVISED PERRY BENCHMARK"))
    for name, result in payload["complete_benchmark"].items():
        if name == "worst_compounds_with_interaction":
            continue
        lines.append(
            f"  {name:24s} MAPE={result['mape_percent']:.2f}% "
            f"P95={result['p95_ape_percent']:.2f}% "
            f"bias={result['mean_signed_error_percent']:+.2f}%"
        )
    lines.extend(("", "WORST 20 WITH GROUP 84 AND INTERACTION REFIT"))
    for rank, compound in enumerate(
        payload["complete_benchmark"]["worst_compounds_with_interaction"][:20], 1
    ):
        lines.append(
            f"  {rank:2d}. {compound['name'][:34]:34s} "
            f"MAPE={compound['mape_percent']:7.2f}%"
        )
    return "\n".join(lines)


def probe(args: argparse.Namespace) -> tuple[dict, str]:
    rows, boiling_points = load_population(
        args.benchmark, args.conductivity_database, args.literature_data
    )
    pure = [row for row in rows if not row["alcohol_groups"]]
    pure_refit = evaluate_model(pure, "none")
    all_group_only = evaluate_model(rows, "none")
    all_interaction = evaluate_model(rows, "molecule")
    all_pair = evaluate_model(rows, "pair")
    complete_rows = load_observations(args.benchmark, args.conductivity_database)
    attach_ring_counts(complete_rows, args.benchmark)
    pure_parameters = (
        pure_refit["parameters"]["group_84_A"],
        pure_refit["parameters"]["group_84_B"],
    )
    interaction_parameters = (
        all_interaction["parameters"]["group_84_A"],
        all_interaction["parameters"]["group_84_B"],
        all_interaction["parameters"]["interaction_A"],
        all_interaction["parameters"]["interaction_B"],
    )
    existing_errors = complete_benchmark_errors(
        complete_rows, args.ring_probe, None, False
    )
    pure_errors = complete_benchmark_errors(
        complete_rows, args.ring_probe, pure_parameters, False
    )
    interaction_errors = complete_benchmark_errors(
        complete_rows, args.ring_probe, interaction_parameters, True
    )
    payload = {
        "schema_version": 1,
        "probe": "govender_tertiary_amine_and_amine_alcohol_refits",
        "weighting": "equal least-squares weight per compound",
        "pure_tertiary_amines": {
            "published": published_summary(pure),
            "refit": pure_refit,
        },
        "all_compounds": {
            "published": published_summary(rows),
            "group_84_only": all_group_only,
            "interaction_once": all_interaction,
            "interaction_per_oh": all_pair,
        },
        "complete_benchmark": {
            "before_amine_refits": summarize(existing_errors),
            "pure_group_84_refit": summarize(pure_errors),
            "group_84_and_interaction": summarize(interaction_errors),
            "worst_compounds_with_interaction": compound_ranking(
                complete_rows, interaction_errors
            ),
        },
        "boiling_points": boiling_points,
        "inputs": {
            "benchmark": artifact_record(args.benchmark),
            "conductivity_database": artifact_record(args.conductivity_database),
            "literature_data": artifact_record(args.literature_data),
            "ring_probe": artifact_record(args.ring_probe),
            "script": artifact_record(Path(__file__)),
            "govender_method": artifact_record(ROOT / "govender_method.py"),
            "uv_lock": artifact_record(ROOT / "uv.lock"),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("chemicals", "numpy", "rdkit", "scipy")
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
    parser.add_argument("--ring-probe", type=Path, default=DEFAULT_RING_PROBE)
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
