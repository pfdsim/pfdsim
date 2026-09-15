#!/usr/bin/env python3
"""Validate Perry-derived Govender typo candidates on external literature data.

The literature observations are never fitted.  Each candidate was selected by
the earlier Perry-only transcription probe and is evaluated here unchanged:
group 33 flips the sign of B, group 72 changes one digit of B, and group 93
flips the sign of B.  Results are averaged equally by compound as well as by
reported observation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import govender_method as gm  # noqa: E402
from scripts.thermal_conductivity.benchmark_gas_viscosity_cv_relation import (  # noqa: E402
    artifact_record,
)


DEFAULT_DATA = (
    Path(__file__).with_name("data") / "govender_sparse_group_literature.json"
)
PERRY_DERIVED_CANDIDATES = {
    33: {"A": -8.434, "B": -0.2742, "change": "B sign flip"},
    72: {"A": 11.442, "B": 0.0726, "change": "B digit 8->0"},
    93: {"A": 9.092, "B": -0.5287, "change": "B sign flip"},
}
BTU_IN_PER_HR_FT2_F_TO_W_PER_M_K = (
    1055.05585262 * 0.0254 / (3600.0 * 0.3048**2 * (5.0 / 9.0))
)


def temperature_K(value: float, unit: str) -> float:
    if unit == "K":
        return value
    if unit == "degC":
        return value + 273.15
    if unit == "degF":
        return (value - 32.0) * 5.0 / 9.0 + 273.15
    raise ValueError(f"unsupported temperature unit {unit}")


def conductivity_W_m_K(value: float, unit: str) -> float:
    if unit == "mW/(m K)":
        return value / 1000.0
    if unit == "BTU in/(hr ft^2 degF)":
        return value * BTU_IN_PER_HR_FT2_F_TO_W_PER_M_K
    raise ValueError(f"unsupported conductivity unit {unit}")


def prediction(
    groups: dict[int, int], n: int, tb: float, temperature: float,
    target_gid: int, A: float, B: float,
) -> float:
    sum_A = sum(gm.CONTRIBUTIONS[gid][0] * frequency
                for gid, frequency in groups.items())
    sum_B = sum(gm.CONTRIBUTIONS[gid][1] * frequency
                for gid, frequency in groups.items())
    frequency = groups[target_gid]
    sum_A += frequency * (A - gm.CONTRIBUTIONS[target_gid][0])
    sum_B += frequency * (B - gm.CONTRIBUTIONS[target_gid][1])
    return (
        math.exp(math.log(n) * sum_B / n)
        + sum_A / tb * (1.0 - temperature / tb)
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def validation(data_path: Path) -> tuple[dict, str]:
    source = json.loads(data_path.read_text(encoding="utf-8"))
    observations = []
    for compound in source["compounds"]:
        gid = int(compound["target_group"])
        candidate = PERRY_DERIVED_CANDIDATES[gid]
        molecule = Chem.MolFromSmiles(compound["smiles"])
        estimate = gm.estimate_from_mol(
            molecule,
            tb=float(compound["normal_boiling_point_K"]),
            smiles=compound["smiles"],
            local_refits=False,
        )
        groups = estimate.groups
        if groups.get(gid, 0) != 1:
            raise ValueError(
                f"{compound['name']} expected one instance of group {gid}; "
                f"fragmentation was {groups}"
            )
        for original_temperature, original_conductivity in compound["observations"]:
            T = temperature_K(
                float(original_temperature), compound["temperature_unit"]
            )
            reference = conductivity_W_m_K(
                float(original_conductivity), compound["conductivity_unit"]
            )
            published = estimate.conductivity_W_m_K(T)
            proposed = prediction(
                groups,
                molecule.GetNumHeavyAtoms(),
                estimate.tb_K,
                T,
                gid,
                float(candidate["A"]),
                float(candidate["B"]),
            )
            observations.append({
                "cas": compound["cas"],
                "name": compound["name"],
                "group": gid,
                "temperature_K": T,
                "reference_W_per_m_K": reference,
                "published_W_per_m_K": published,
                "candidate_W_per_m_K": proposed,
                "published_signed_error_percent": 100.0 * (published / reference - 1.0),
                "candidate_signed_error_percent": 100.0 * (proposed / reference - 1.0),
            })

    by_group: dict[int, list[dict]] = defaultdict(list)
    for observation in observations:
        by_group[observation["group"]].append(observation)
    group_results = {}
    for gid, rows in sorted(by_group.items()):
        cases = sorted({row["cas"] for row in rows})
        compound_results = []
        for cas in cases:
            selected = [row for row in rows if row["cas"] == cas]
            compound_results.append({
                "cas": cas,
                "name": selected[0]["name"],
                "points": len(selected),
                "published_mape_percent": mean([
                    abs(row["published_signed_error_percent"]) for row in selected
                ]),
                "candidate_mape_percent": mean([
                    abs(row["candidate_signed_error_percent"]) for row in selected
                ]),
                "published_bias_percent": mean([
                    row["published_signed_error_percent"] for row in selected
                ]),
                "candidate_bias_percent": mean([
                    row["candidate_signed_error_percent"] for row in selected
                ]),
            })
        group_results[str(gid)] = {
            "candidate": PERRY_DERIVED_CANDIDATES[gid],
            "compounds": len(cases),
            "points": len(rows),
            "compound_equal_published_mape_percent": mean([
                row["published_mape_percent"] for row in compound_results
            ]),
            "compound_equal_candidate_mape_percent": mean([
                row["candidate_mape_percent"] for row in compound_results
            ]),
            "compound_results": compound_results,
        }

    payload = {
        "schema_version": 1,
        "validation": "govender_perry_derived_sparse_candidates_external_literature",
        "method": "no fitting; candidates fixed before literature evaluation",
        "conversion": {
            "BTU_in_per_hr_ft2_F_to_W_per_m_K": BTU_IN_PER_HR_FT2_F_TO_W_PER_M_K
        },
        "groups": group_results,
        "observations": observations,
        "inputs": {
            "literature_data": artifact_record(data_path),
            "script": artifact_record(Path(__file__)),
            "govender_method": artifact_record(ROOT / "govender_method.py"),
        },
    }
    lines = [
        "Govender sparse-group external validation",
        "Candidates selected from Perry and held fixed; literature data were not fitted.",
        "",
    ]
    for gid, result in group_results.items():
        candidate = result["candidate"]
        lines.append(
            f"GROUP {gid}: n={result['compounds']} compounds/{result['points']} points; "
            f"published MAPE={result['compound_equal_published_mape_percent']:.2f}%, "
            f"candidate MAPE={result['compound_equal_candidate_mape_percent']:.2f}% "
            f"({candidate['change']}, A={candidate['A']}, B={candidate['B']})"
        )
        for compound in result["compound_results"]:
            lines.append(
                f"  {compound['name']}: {compound['points']} points, "
                f"{compound['published_mape_percent']:.2f}% -> "
                f"{compound['candidate_mape_percent']:.2f}%"
            )
    return payload, "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, report = validation(args.data)
    print(report)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
