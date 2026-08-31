#!/usr/bin/env python3
"""Convert the 2005 PSRK supplementary Word tables to JSON.

The source is the legacy ``.doc`` supplement for Horstmann et al.,
Fluid Phase Equilibria 227 (2005) 157-164.  ``antiword`` is used only as
the document reader; all table-boundary checks, type conversion, validation,
and JSON serialization live here so the generated data remain reproducible.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "source" / "1-s2.0-S0378381204005072-mmc1.doc"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "psrk"
GROUP_OUTPUT_NAME = "group_parameters.json"
PURE_OUTPUT_NAME = "pure_components.json"

SOURCE = {
    "paper": (
        "S. Horstmann, A. Jabloniec, J. Krafczyk, K. Fischer, and J. Gmehling, "
        "PSRK group contribution equation of state: comprehensive revision and "
        "extension IV, including critical constants and alpha-function parameters "
        "for 1000 components, Fluid Phase Equilibria 227 (2005) 157-164"
    ),
    "doi": "10.1016/j.fluid.2004.11.002",
    "supplement": DEFAULT_SOURCE.name,
}

PROVENANCE = {
    "a": "Original UNIFAC parameters, source reference [7].",
    "b": "Previously published PSRK parameters, source references [2,5,8-9].",
    "c": "New or revised PSRK parameters from the 2005 work and source reference [10].",
}

CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")
INTERACTION_INDEX_PATTERN = re.compile(r"^(\d+)([abc])$")
ASSIGNMENT_PATTERN = re.compile(
    r"^(?:\d+\s*×\s*\d+)(?:\s*;\s*\d+\s*×\s*\d+)*$"
)

# A small set of Table 3 cells has a consistent source-formatting defect:
# ``1×10N`` is rendered as ``11×N``.  These corrections are constrained by
# the molecular formula and by the PSRK subgroup table.  R142b has a separate
# missing subgroup digit, likewise recoverable as CH3 (subgroup 1).
SUBGROUP_ASSIGNMENT_CORRECTIONS = {
    "624-89-5": ("1×1; 1×2; 11×2", "1×1; 1×2; 1×102"),
    "352-93-2": ("2×1; 1×2; 11×3", "2×1; 1×2; 1×103"),
    "287-27-4": ("2×2; 11×3", "2×2; 1×103"),
    "1613-51-0": ("4×2; 11×3", "4×2; 1×103"),
    "1795-09-1": ("1×1; 2×2; 1×3; 11×3", "1×1; 2×2; 1×3; 1×103"),
    "4740-00-5": ("1×1; 2×2; 1×3; 11×3", "1×1; 2×2; 1×3; 1×103"),
    "5258-50-4": ("1×1; 3×2; 1×3; 11×3", "1×1; 3×2; 1×3; 1×103"),
    "5161-16-0": ("1×1; 3×2; 1×3; 11×3", "1×1; 3×2; 1×3; 1×103"),
    "5161-17-1": ("1×1; 3×2; 1×3; 11×3", "1×1; 3×2; 1×3; 1×103"),
    "1795-01-3": ("1×1; 1×2; 11×7", "1×1; 1×2; 1×107"),
    "7133-36-0": ("4×2; 1×3; 11×2", "4×2; 1×3; 1×102"),
    "100-68-5": ("5×9; 1×10; 11×2", "5×9; 1×10; 1×102"),
    "2690-08-6": ("2×1; 13×2;11×3", "2×1; 13×2; 1×103"),
    "872-55-9": ("1×1; 1×2; 11×7", "1×1; 1×2; 1×107"),
    "107-98-2": ("2×1; 11×1", "2×1; 1×101"),
    "628-29-5": ("1×1; 3×2; 11×2", "1×1; 3×2; 1×102"),
    "75-68-3": ("1×; 1×90", "1×1; 1×90"),
}


class ConversionError(RuntimeError):
    """Raised when the source does not have the expected table structure."""


def _antiword_text(path: Path) -> str:
    executable = shutil.which("antiword")
    if executable is None:
        raise ConversionError(
            "antiword is required to read the legacy PSRK .doc supplement"
        )
    try:
        process = subprocess.run(
            [executable, "-w", "0", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        raise ConversionError(f"antiword failed for {path}: {detail}") from error
    return process.stdout


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _line_starting(lines: Iterable[str], prefix: str) -> str:
    matches = [line for line in lines if line.startswith(prefix)]
    if len(matches) != 1:
        raise ConversionError(
            f"Expected one extracted line beginning with {prefix!r}, found {len(matches)}"
        )
    return matches[0]


def _pipe_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.split("|")]


def _fixed_rows(cells: Sequence[str], width: int, context: str) -> list[list[str]]:
    if len(cells) % width:
        raise ConversionError(
            f"{context} has {len(cells)} cells, not a multiple of row width {width}"
        )
    rows = [list(cells[index:index + width]) for index in range(0, len(cells), width)]
    for number, row in enumerate(rows, start=1):
        if row[-1] != "":
            raise ConversionError(f"{context} row {number} has no empty row separator")
    return rows


def _float(value: str, context: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ConversionError(f"Invalid number {value!r} in {context}") from error


def _optional_float(value: str, context: str) -> float | None:
    if not value or value == "n.a.":
        return None
    return _float(value, context)


def _starred_float(value: str, context: str) -> tuple[float, bool]:
    estimated = value.endswith("*")
    raw_value = value[:-1] if estimated else value
    return _float(raw_value, context), estimated


def _parse_subgroup_assignment(
    value: str,
    *,
    component: str,
    malformed: list[dict],
) -> list[dict] | None:
    if value == "n.a.":
        return None
    if not ASSIGNMENT_PATTERN.fullmatch(value):
        malformed.append({
            "component": component,
            "source_value": value,
            "reason": "incomplete or malformed subgroup assignment in source",
        })
        return None

    counts: dict[int, int] = {}
    for count_text, subgroup_text in re.findall(r"(\d+)\s*×\s*(\d+)", value):
        subgroup = int(subgroup_text)
        count = int(count_text)
        if subgroup in counts:
            raise ConversionError(
                f"Duplicate subgroup {subgroup} in assignment for {component}"
            )
        counts[subgroup] = count
    return [
        {"subgroup_id": subgroup, "count": counts[subgroup]}
        for subgroup in sorted(counts)
    ]


def parse_group_table(line: str) -> list[dict]:
    cells = _pipe_cells(line)
    if cells[-1] != "":
        raise ConversionError("Table 1 extraction does not end in an empty cell")
    cells.pop()  # antiword adds an extra cell for the final pipe.
    rows = _fixed_rows(cells, 9, "Table 1")

    subgroups: list[dict] = []
    for number, row in enumerate(rows, start=1):
        main_group, main_name, subgroup, subgroup_name, r_value, q_value, example, increments, _ = row
        try:
            record = {
                "main_group_id": int(main_group),
                "main_group_name": main_name,
                "subgroup_id": int(subgroup),
                "subgroup_name": subgroup_name,
                "R": _float(r_value, f"Table 1 row {number} R"),
                "Q": _float(q_value, f"Table 1 row {number} Q"),
                "example_component": example,
                "example_increments": increments,
            }
        except ValueError as error:
            raise ConversionError(f"Invalid group index in Table 1 row {number}") from error
        subgroups.append(record)

    subgroup_ids = [record["subgroup_id"] for record in subgroups]
    if len(subgroup_ids) != len(set(subgroup_ids)):
        raise ConversionError("Table 1 contains duplicate subgroup IDs")
    return subgroups


def _interaction_coefficient(
    value: str,
    *,
    context: str,
    normalizations: list[dict],
) -> float:
    if not value:
        return 0.0
    if value == "308.9.":
        normalizations.append({
            "context": context,
            "source_value": value,
            "normalized_value": 308.9,
            "reason": "source value has a second trailing decimal point",
        })
        return 308.9
    return _float(value, context)


def parse_interaction_table(line: str) -> tuple[list[dict], list[dict]]:
    cells = _pipe_cells(line)
    expected_header = [
        "i", "j", "aij / K", "bij", "cij / K-1",
        "aji / K", "bji", "cji / K-1", "",
    ]
    if cells[:9] != expected_header:
        raise ConversionError(f"Unexpected Table 2 header: {cells[:9]!r}")
    cells = cells[9:]
    if cells[-1] != "":
        raise ConversionError("Table 2 extraction does not end in an empty cell")
    cells.pop()  # antiword adds an extra cell for the final pipe.
    rows = _fixed_rows(cells, 9, "Table 2")

    interactions: list[dict] = []
    normalizations: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()
    for number, row in enumerate(rows, start=1):
        index, j_text, aij, bij, cij, aji, bji, cji, _ = row
        match = INTERACTION_INDEX_PATTERN.fullmatch(index)
        if match is None:
            raise ConversionError(f"Invalid Table 2 i/provenance value {index!r}")
        i = int(match.group(1))
        j = int(j_text)
        provenance_code = match.group(2)
        pair = (i, j)
        if i >= j:
            raise ConversionError(f"Table 2 pair is not ordered i < j: {pair}")
        if pair in seen_pairs:
            raise ConversionError(f"Duplicate Table 2 interaction pair {pair}")
        seen_pairs.add(pair)

        interactions.append({
            "main_group_i": i,
            "main_group_j": j,
            "provenance_code": provenance_code,
            "i_to_j": {
                "a_K": _interaction_coefficient(
                    aij,
                    context=f"Table 2 row {number} aij",
                    normalizations=normalizations,
                ),
                "b": _interaction_coefficient(
                    bij,
                    context=f"Table 2 row {number} bij",
                    normalizations=normalizations,
                ),
                "c_per_K": _interaction_coefficient(
                    cij,
                    context=f"Table 2 row {number} cij",
                    normalizations=normalizations,
                ),
            },
            "j_to_i": {
                "a_K": _interaction_coefficient(
                    aji,
                    context=f"Table 2 row {number} aji",
                    normalizations=normalizations,
                ),
                "b": _interaction_coefficient(
                    bji,
                    context=f"Table 2 row {number} bji",
                    normalizations=normalizations,
                ),
                "c_per_K": _interaction_coefficient(
                    cji,
                    context=f"Table 2 row {number} cji",
                    normalizations=normalizations,
                ),
            },
        })
    return interactions, normalizations


def parse_component_table(line: str) -> tuple[list[dict], list[dict]]:
    cells = _pipe_cells(line)
    expected_header = [
        "english name", "formula", "CAS-nr.", "Tc,i / K", "Pc,i / kPa",
        "vc,i / cm3 mol-1", "ωi", "c1,i", "c2,i", "c3,i", "Tmin / K",
        "Tmax / K", "increments [counter × sub group number]", "",
    ]
    if cells[:14] != expected_header:
        raise ConversionError(f"Unexpected Table 3 header: {cells[:14]!r}")
    if not cells[-1].startswith("* predicted value"):
        raise ConversionError("Table 3 predicted-value footnote was not found")
    footnote = cells.pop()
    rows = _fixed_rows(cells[14:], 14, "Table 3")

    components: list[dict] = []
    malformed_assignments: list[dict] = []
    assignment_normalizations: list[dict] = []
    seen_cas: set[str] = set()
    for number, row in enumerate(rows, start=1):
        (
            name, formula, cas, tc_text, pc_text, vc_text, omega_text,
            c1_text, c2_text, c3_text, tmin_text, tmax_text,
            assignment_text, _,
        ) = row
        if not CAS_PATTERN.fullmatch(cas):
            raise ConversionError(f"Invalid CAS number {cas!r} in Table 3 row {number}")
        if cas in seen_cas:
            raise ConversionError(f"Duplicate CAS number {cas} in Table 3")
        seen_cas.add(cas)

        source_assignment_text = assignment_text
        correction = SUBGROUP_ASSIGNMENT_CORRECTIONS.get(cas)
        if correction is not None:
            expected_source, corrected_assignment = correction
            if assignment_text != expected_source:
                raise ConversionError(
                    f"Expected corrected subgroup source value {expected_source!r} "
                    f"for {cas}, found {assignment_text!r}"
                )
            assignment_text = corrected_assignment
            assignment_normalizations.append({
                "component": name,
                "CAS": cas,
                "source_value": source_assignment_text,
                "normalized_value": corrected_assignment,
                "reason": (
                    "Restored a displaced or missing subgroup digit using the "
                    "molecular formula and the published PSRK subgroup table."
                ),
            })

        tc, tc_estimated = _starred_float(tc_text, f"Table 3 row {number} Tc")
        pc, pc_estimated = _starred_float(pc_text, f"Table 3 row {number} Pc")
        components.append({
            "name": name,
            "formula": formula,
            "CAS": cas,
            "Tc_K": tc,
            "Tc_estimated": tc_estimated,
            "Pc_kPa": pc,
            "Pc_estimated": pc_estimated,
            "Vc_cm3_per_mol": _optional_float(vc_text, f"Table 3 row {number} Vc"),
            "omega": _float(omega_text, f"Table 3 row {number} omega"),
            "mathias_copeman": {
                "c1": _float(c1_text, f"Table 3 row {number} c1"),
                "c2": _float(c2_text, f"Table 3 row {number} c2"),
                "c3": _float(c3_text, f"Table 3 row {number} c3"),
                "Tmin_K": _optional_float(tmin_text, f"Table 3 row {number} Tmin"),
                "Tmax_K": _optional_float(tmax_text, f"Table 3 row {number} Tmax"),
            },
            "subgroups": _parse_subgroup_assignment(
                assignment_text,
                component=name,
                malformed=malformed_assignments,
            ),
            **(
                {"subgroup_assignment_source": source_assignment_text}
                if source_assignment_text != assignment_text
                or (
                    source_assignment_text != "n.a."
                    and not ASSIGNMENT_PATTERN.fullmatch(source_assignment_text)
                )
                else {}
            ),
        })

    if footnote != "* predicted value [20,21]":
        raise ConversionError(f"Unexpected Table 3 footnote: {footnote!r}")
    return components, malformed_assignments + assignment_normalizations


def convert(path: Path) -> tuple[dict, dict]:
    lines = _nonempty_lines(_antiword_text(path))
    subgroups = parse_group_table(_line_starting(lines, "1 |CH2"))
    interactions, normalizations = parse_interaction_table(
        _line_starting(lines, "i |j |aij")
    )
    components, assignment_source_notes = parse_component_table(
        _line_starting(lines, "english name |formula")
    )

    main_groups = {record["main_group_id"] for record in subgroups}
    subgroup_ids = {record["subgroup_id"] for record in subgroups}
    interaction_groups = {
        group
        for record in interactions
        for group in (record["main_group_i"], record["main_group_j"])
    }
    unknown_interaction_groups = sorted(interaction_groups - main_groups)
    if unknown_interaction_groups:
        raise ConversionError(
            f"Interactions reference unknown main groups: {unknown_interaction_groups}"
        )

    unknown_component_subgroups = sorted({
        assignment["subgroup_id"]
        for component in components
        for assignment in (component["subgroups"] or [])
        if assignment["subgroup_id"] not in subgroup_ids
    })
    if unknown_component_subgroups:
        raise ConversionError(
            f"Components reference unknown subgroups: {unknown_component_subgroups}"
        )

    provenance_counts = Counter(
        record["provenance_code"] for record in interactions
    )
    group_payload = {
        "metadata": {
            "schema_version": 1,
            "model": "PSRK/UNIFAC (published 2005 parameterization)",
            "source": {**SOURCE, "supplement": path.name},
            "equation": "psi_ij(T) = exp(-(a_ij + b_ij*T + c_ij*T^2)/T)",
            "units": {"a": "K", "b": "dimensionless", "c": "K^-1", "T": "K"},
            "blank_interaction_coefficients": "Converted to numeric zero.",
            "interaction_provenance": PROVENANCE,
            "counts": {
                "main_groups": len(main_groups),
                "subgroups": len(subgroups),
                "unordered_interaction_pairs": len(interactions),
                "directed_interactions": 2 * len(interactions),
                "interaction_pairs_by_provenance": dict(sorted(provenance_counts.items())),
            },
            "source_normalizations": normalizations,
        },
        "subgroups": subgroups,
        "interactions": interactions,
    }

    pure_payload = {
        "metadata": {
            "schema_version": 1,
            "model": "PSRK pure-component parameters (published 2005 parameterization)",
            "source": {**SOURCE, "supplement": path.name},
            "estimated_value_marker": (
                "Tc_K and Pc_kPa flags preserve source asterisks; the supplement "
                "defines an asterisk as a predicted value from references [20,21]."
            ),
            "subgroup_assignment_format": (
                "Each item stores a PSRK/UNIFAC subgroup ID and its count; null means "
                "the source says n.a. or contains an incomplete assignment."
            ),
            "counts": {
                "components": len(components),
                "estimated_Tc": sum(record["Tc_estimated"] for record in components),
                "estimated_Pc": sum(record["Pc_estimated"] for record in components),
                "missing_Vc": sum(record["Vc_cm3_per_mol"] is None for record in components),
                "without_subgroup_assignment": sum(record["subgroups"] is None for record in components),
                "subgroup_assignment_source_notes": len(assignment_source_notes),
            },
            "subgroup_assignment_source_notes": assignment_source_notes,
        },
        "components": components,
    }
    return group_payload, pure_payload


def _serialized(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _write_outputs(output_dir: Path, group_payload: dict, pure_payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / GROUP_OUTPUT_NAME).write_text(
        _serialized(group_payload), encoding="utf-8"
    )
    (output_dir / PURE_OUTPUT_NAME).write_text(
        _serialized(pure_payload), encoding="utf-8"
    )


def _check_outputs(output_dir: Path, group_payload: dict, pure_payload: dict) -> None:
    expected = {
        GROUP_OUTPUT_NAME: _serialized(group_payload),
        PURE_OUTPUT_NAME: _serialized(pure_payload),
    }
    mismatches = []
    for filename, content in expected.items():
        path = output_dir / filename
        if not path.exists():
            mismatches.append(f"missing {path}")
        elif path.read_text(encoding="utf-8") != content:
            mismatches.append(f"stale {path}")
    if mismatches:
        raise ConversionError("Generated outputs do not match: " + ", ".join(mismatches))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that existing JSON files exactly match regenerated content.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    group_payload, pure_payload = convert(args.source)
    if args.check:
        _check_outputs(args.output_dir, group_payload, pure_payload)
        action = "Verified"
    else:
        _write_outputs(args.output_dir, group_payload, pure_payload)
        action = "Wrote"
    print(
        f"{action} {len(group_payload['subgroups'])} subgroups, "
        f"{len(group_payload['interactions'])} interaction pairs, and "
        f"{len(pure_payload['components'])} pure components in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
