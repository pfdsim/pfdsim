#!/usr/bin/env python3
"""Probe suspect Govender coefficients for simple transcription errors.

For each selected nonacid group, compare the published pair against simple
one-digit, sign, and decimal-place variants and against an unconstrained
two-parameter fit.  These are diagnostics only: overlapping groups and sparse
compound support prevent the fitted values from being interpreted as corrected
coefficients.
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
GROUPS = {
    10: ("ring CH2", "4.67", "-1.218"),
    21: ("ring C=C carbon", "10.37", "-1.217"),
    24: ("ring CH2 attached to electronegative atom", "7.65", "-1.860"),
    33: ("conjugated chain diene", "-8.434", "0.2742"),
    35: ("F on nonaromatic carbon", "28.679", "-1.4986"),
    45: ("Br on nonaromatic carbon", "6.135", "-2.7263"),
    58: ("ketone", "0.546", "-0.2229"),
    70: ("primary amide", "26.620", "-3.0101"),
    72: ("tertiary amide", "11.442", "0.8726"),
    84: ("tertiary amine", "-12.576", "1.1656"),
    93: ("cyclic secondary amine", "9.092", "0.5287"),
    100: ("thioether", "11.579", "0.0759"),
}
PAPER_CLASS_SUPPORT = {
    10: ("cyclic alkanes", 6, False),
    21: ("cyclic alkenes", 5, False),
    24: ("not disclosed", None, False),
    33: ("noncyclic alkenes", 12, False),
    35: ("fluorinated compounds", 20, False),
    45: ("brominated compounds", 11, False),
    58: ("ketones", 7, True),
    70: ("amides", 3, False),
    72: ("amides", 3, False),
    84: ("tertiary amines", 4, True),
    93: ("secondary amines", 5, False),
    100: ("sulfur compounds", 2, True),
}


def candidate_values(text: str) -> list[tuple[float, str]]:
    result: dict[float, str] = {}
    for index, character in enumerate(text):
        if not character.isdigit():
            continue
        for replacement in "0123456789":
            if replacement == character:
                continue
            candidate = text[:index] + replacement + text[index + 1:]
            try:
                value = float(candidate)
            except ValueError:
                continue
            result.setdefault(
                value, f"digit {character}->{replacement} in {text}"
            )
    value = float(text)
    result.setdefault(-value, f"sign flip of {text}")
    result.setdefault(value * 10.0, f"decimal x10 from {text}")
    result.setdefault(value / 10.0, f"decimal /10 from {text}")
    return sorted(result.items())


def load_observations(benchmark_path: Path, conductivity_path: Path) -> list[dict]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    thermal = json.loads(conductivity_path.read_text(encoding="utf-8"))["chemicals"]
    points = benchmark["method"]["sampling_points_per_curve"]
    observations = []
    for compound in benchmark["compounds"]:
        groups = {int(gid): int(count) for gid, count in compound["groups"].items()}
        molecule = Chem.MolFromSmiles(compound["smiles"])
        n = molecule.GetNumHeavyAtoms()
        sum_A = sum(gm.CONTRIBUTIONS[gid][0] * count
                    for gid, count in groups.items())
        sum_B = sum(gm.CONTRIBUTIONS[gid][1] * count
                    for gid, count in groups.items())
        row = thermal[compound["cas"]]["liquid_thermal_conductivity"][0]
        for temperature in np.linspace(
            float(row["T_min_K"]), float(row["T_max_K"]), points
        ):
            observations.append({
                "cas": compound["cas"],
                "name": compound["name"],
                "groups": groups,
                "n": n,
                "tb": float(compound["tb_K"]),
                "sum_A": sum_A,
                "sum_B": sum_B,
                "temperature": float(temperature),
                "reference": conductivity_reference(row, float(temperature)),
            })
    return observations


def errors(rows: list[dict], gid: int, A: float, B: float) -> np.ndarray:
    published_A, published_B = gm.CONTRIBUTIONS[gid]
    result = []
    for row in rows:
        frequency = row["groups"].get(gid, 0)
        sum_A = row["sum_A"] + frequency * (A - published_A)
        sum_B = row["sum_B"] + frequency * (B - published_B)
        try:
            predicted = (
                math.exp(math.log(row["n"]) * sum_B / row["n"])
                + sum_A / row["tb"]
                * (1.0 - row["temperature"] / row["tb"])
            )
        except OverflowError:
            predicted = math.inf
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


def fit_group(gid: int, rows: list[dict]) -> tuple[float, float]:
    published_A, published_B = gm.CONTRIBUTIONS[gid]
    result = least_squares(
        lambda parameters: errors(
            rows, gid, float(parameters[0]), float(parameters[1])
        ) / 100.0,
        (published_A, published_B),
    )
    return float(result.x[0]), float(result.x[1])


def probe_group(gid: int, rows: list[dict]) -> dict:
    name, A_text, B_text = GROUPS[gid]
    selected = [row for row in rows if gid in row["groups"]]
    cases = sorted({row["cas"] for row in selected})
    published_A, published_B = gm.CONTRIBUTIONS[gid]
    published_errors = errors(selected, gid, published_A, published_B)

    candidates = [
        (A, published_B, f"A: {description}")
        for A, description in candidate_values(A_text)
    ] + [
        (published_A, B, f"B: {description}")
        for B, description in candidate_values(B_text)
    ]
    finite_candidates = []
    for A, B, description in candidates:
        values = errors(selected, gid, A, B)
        if np.all(np.isfinite(values)):
            finite_candidates.append((float(np.mean(np.abs(values))), A, B, description))
    _, candidate_A, candidate_B, description = min(finite_candidates)
    candidate_errors = errors(selected, gid, candidate_A, candidate_B)

    def residual(parameters):
        return errors(selected, gid, float(parameters[0]), float(parameters[1])) / 100.0

    fitted = least_squares(residual, (published_A, published_B))
    fitted_A, fitted_B = map(float, fitted.x)
    per_compound = []
    for cas in cases:
        mask = np.asarray([row["cas"] == cas for row in selected])
        per_compound.append({
            "cas": cas,
            "name": next(row["name"] for row in selected if row["cas"] == cas),
            "mape_percent": float(np.mean(np.abs(published_errors[mask]))),
        })
    worst = max(per_compound, key=lambda item: item["mape_percent"])
    without_worst = np.asarray([
        value for value, row in zip(published_errors, selected)
        if row["cas"] != worst["cas"]
    ])
    transcription_cv = refit_cv = None
    if len(cases) >= 2:
        transcription_held_out = []
        refit_held_out = []
        for held_out in cases:
            training = [row for row in selected if row["cas"] != held_out]
            testing = [row for row in selected if row["cas"] == held_out]

            training_candidates = []
            for A, B, candidate_description in candidates:
                values = errors(training, gid, A, B)
                if np.all(np.isfinite(values)):
                    training_candidates.append((
                        float(np.mean(np.abs(values))), A, B,
                        candidate_description,
                    ))
            _, cv_A, cv_B, _ = min(training_candidates)
            transcription_held_out.extend(errors(testing, gid, cv_A, cv_B))

            def training_residual(parameters):
                return errors(
                    training, gid, float(parameters[0]), float(parameters[1])
                ) / 100.0

            cv_fit = least_squares(
                training_residual, (published_A, published_B)
            )
            refit_held_out.extend(
                errors(testing, gid, float(cv_fit.x[0]), float(cv_fit.x[1]))
            )
        transcription_cv = summarize(np.asarray(transcription_held_out))
        refit_cv = summarize(np.asarray(refit_held_out))
    support = (
        "underdetermined" if len(cases) == 1 else
        "very weak" if len(cases) == 2 else
        "weak" if len(cases) <= 4 else
        "limited" if len(cases) < 10 else
        "useful"
    )
    paper_class, paper_compounds, exact_class = PAPER_CLASS_SUPPORT[gid]
    return {
        "group": gid,
        "name": name,
        "compounds": len(cases),
        "refit_support": support,
        "perry_sampled_states": len(selected),
        "paper_support": {
            "class": paper_class,
            "compounds": paper_compounds,
            "is_exact_group_class": exact_class,
            "class_specific_data_points": None,
        },
        "published": {
            "A": published_A, "B": published_B,
            **summarize(published_errors),
        },
        "best_simple_candidate": {
            "A": candidate_A, "B": candidate_B,
            "description": description,
            **summarize(candidate_errors),
        },
        "least_squares_fit": {
            "A": fitted_A, "B": fitted_B,
            **summarize(errors(selected, gid, fitted_A, fitted_B)),
        },
        "leave_one_compound_out": {
            "simple_candidate": transcription_cv,
            "full_refit": refit_cv,
        },
        "worst_compound": worst,
        "published_without_worst_mape_percent": (
            float(np.mean(np.abs(without_worst))) if len(without_worst) else None
        ),
    }


def refit_subclass(gid: int, rows: list[dict], excluded_names: set[str]) -> dict:
    selected = [
        row for row in rows
        if gid in row["groups"] and row["name"] not in excluded_names
    ]
    cases = sorted({row["cas"] for row in selected})
    fitted_A, fitted_B = fit_group(gid, selected)
    held_out = []
    for cas in cases:
        training = [row for row in selected if row["cas"] != cas]
        testing = [row for row in selected if row["cas"] == cas]
        cv_A, cv_B = fit_group(gid, training)
        held_out.extend(errors(testing, gid, cv_A, cv_B))
    return {
        "group": gid,
        "excluded_compounds": sorted(excluded_names),
        "compounds": len(cases),
        "A": fitted_A,
        "B": fitted_B,
        "training": summarize(errors(selected, gid, fitted_A, fitted_B)),
        "leave_one_compound_out": summarize(np.asarray(held_out)),
    }


def combined_correction_errors(rows: list[dict], *, cross_validated: bool) -> np.ndarray:
    subclasses = {
        53: [row for row in rows if 53 in row["groups"]],
        58: [row for row in rows
             if 58 in row["groups"] and row["name"] != "Quinone"],
        100: [row for row in rows
              if 100 in row["groups"] and row["name"] != "Tetrahydrothiophene"],
    }
    parameters: dict[int, dict[str, tuple[float, float]]] = {}
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

    result = []
    for row in rows:
        sum_A, sum_B = row["sum_A"], row["sum_B"]
        for gid in subclasses:
            fitted = parameters[gid].get(row["cas"])
            if fitted is None:
                continue
            frequency = row["groups"][gid]
            sum_A += frequency * (fitted[0] - gm.CONTRIBUTIONS[gid][0])
            sum_B += frequency * (fitted[1] - gm.CONTRIBUTIONS[gid][1])
        predicted = (
            math.exp(math.log(row["n"]) * sum_B / row["n"])
            + sum_A / row["tb"]
            * (1.0 - row["temperature"] / row["tb"])
        )
        result.append(100.0 * (predicted / row["reference"] - 1.0))
    return np.asarray(result)


def build_report(payload: dict) -> str:
    lines = [
        "Govender nonacid group transcription probe",
        "Population excludes every compound containing COOH group 48 or 53.",
        "Simple candidates change one digit, flip a sign, or move a decimal place.",
        "",
    ]
    for item in payload["groups"]:
        published = item["published"]
        candidate = item["best_simple_candidate"]
        fitted = item["least_squares_fit"]
        cv = item["leave_one_compound_out"]
        lines.extend((
            f"GROUP {item['group']} {item['name']} (n={item['compounds']}; "
            f"refit support={item['refit_support']})",
            f"  support: Perry={item['compounds']} curves/"
            f"{item['perry_sampled_states']} sampled states; paper "
            f"{item['paper_support']['class']}="
            f"{item['paper_support']['compounds'] if item['paper_support']['compounds'] is not None else 'not reported'} compounds"
            f"{' (exact class)' if item['paper_support']['is_exact_group_class'] else ' (broader class)'}",
            f"  published A={published['A']:.6g} B={published['B']:.6g}: "
            f"MAPE={published['mape_percent']:.2f}%",
            f"  candidate A={candidate['A']:.6g} B={candidate['B']:.6g}: "
            f"MAPE={candidate['mape_percent']:.2f}% ({candidate['description']})",
            f"  free fit  A={fitted['A']:.6g} B={fitted['B']:.6g}: "
            f"MAPE={fitted['mape_percent']:.2f}%",
            (
                f"  leave-one-compound-out: candidate="
                f"{cv['simple_candidate']['mape_percent']:.2f}%, "
                f"full refit={cv['full_refit']['mape_percent']:.2f}%"
                if cv["full_refit"] is not None
                else "  leave-one-compound-out: insufficient compounds"
            ),
            f"  worst: {item['worst_compound']['name']} "
            f"{item['worst_compound']['mape_percent']:.2f}%; "
            f"published without worst="
            f"{item['published_without_worst_mape_percent'] if item['published_without_worst_mape_percent'] is not None else 'n/a'}",
            "",
        ))
    lines.append("TARGETED SUBCLASS REFITS")
    for name, result in payload["subclass_refits"].items():
        lines.append(
            f"  {name}: n={result['compounds']} A={result['A']:.6g} "
            f"B={result['B']:.6g}; training MAPE="
            f"{result['training']['mape_percent']:.2f}%, LOOCV MAPE="
            f"{result['leave_one_compound_out']['mape_percent']:.2f}%"
        )
    lines.append("")
    lines.append("COMBINED EFFECT ON COMPLETE BENCHMARK")
    for name, result in payload["combined_benchmark"].items():
        lines.append(
            f"  {name:34s} MAPE={result['mape_percent']:.2f}% "
            f"P95={result['p95_ape_percent']:.2f}% "
            f"bias={result['mean_signed_error_percent']:+.2f}%"
        )
    return "\n".join(lines).rstrip()


def probe(args: argparse.Namespace) -> tuple[dict, str]:
    all_rows = load_observations(args.benchmark, args.conductivity_database)
    rows = [row for row in all_rows if not set(row["groups"]) & {48, 53}]
    published_overall = np.asarray([
        100.0 * (
            (
                math.exp(math.log(row["n"]) * row["sum_B"] / row["n"])
                + row["sum_A"] / row["tb"]
                * (1.0 - row["temperature"] / row["tb"])
            ) / row["reference"] - 1.0
        )
        for row in all_rows
    ])
    payload = {
        "schema_version": 1,
        "probe": "govender_nonacid_group_simple_transcriptions",
        "groups": [probe_group(gid, rows) for gid in GROUPS],
        "subclass_refits": {
            "ordinary_ketones_excluding_quinone": refit_subclass(
                58, rows, {"Quinone"}
            ),
            "acyclic_thioethers_excluding_tetrahydrothiophene": refit_subclass(
                100, rows, {"Tetrahydrothiophene"}
            ),
        },
        "combined_benchmark": {
            "published": summarize(published_overall),
            "fixed_subclass_refits": summarize(
                combined_correction_errors(all_rows, cross_validated=False)
            ),
            "subclass_refits_LOOCV": summarize(
                combined_correction_errors(all_rows, cross_validated=True)
            ),
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
