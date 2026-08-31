#!/usr/bin/env python3
"""Build the NIST-modified UNIFAC runtime parameter JSON.

The supplement attached to Kang, Diky, and Frenkel (FPE 388, 2015)
contains two Word tables: subgroup parameters and directed main-group
interaction parameters.  The 2017 corrigendum fixes several labels and
examples; those corrections are applied here so the generated JSON can be
used as a clean data source. Later source tables are then applied as explicit,
auditable overlays; currently this includes the 2023 Dantas-Ceriani lactone
interaction reparametrization.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DOCX = ROOT / "data" / "source" / "1-s2.0-S0378381214007353-mmc1.docx"
LACTONE_OVERLAY_JSON = (
    ROOT / "data" / "source" / "nist_unifac_lactone_interactions_table1.json"
)
OUTPUT_JSON = ROOT / "data" / "nist_modified_unifac_params.json"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


CORRECTED_MAIN_GROUP_NAMES = {
    83: "ACCO",
    86: "C2Cl4",
}

CORRECTED_SUBGROUP_EXAMPLES = {
    39: ("2,6-dimethylpyridine", "1 AC2N, 3 ACH, 2 CH3"),
    160: ("2,3-benzofuran", "5 ACH, 1 AC, 1 AC2HO"),
    162: ("2,6-lupetidine", "1 c-CH-NH, 2 CH3, 3 c-CH2, 1 c-CH"),
    163: ("2,2,6,6-tetramethylpiperidine", "1 c-C-NH, 4 CH3, 3 c-CH2, 1 c-C"),
    188: ("piperazine", "2 c-CH2-NH, 2 c-CH2"),
    198: ("2,5-dimethylpyrrole", "1 AC2NH, 2 ACH, 2 CH3"),
    305: ("2-methyl-2-nitropropane", "3 CH3, 1 CNO2"),
    306: ("diphenylamine", "10 ACH, 1 AC, 1 ACNH"),
    309: ("dimethoxymethane", "2 CH3, 1 CH2(O)2"),
}


def _cell_text(cell: ET.Element) -> str:
    parts: list[str] = []
    for paragraph in cell.findall("./w:p", NS):
        text = "".join(
            text_node.text or ""
            for text_node in paragraph.findall(".//w:t", NS)
        ).strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _table_rows(table: ET.Element) -> Iterable[list[str]]:
    for row in table.findall("./w:tr", NS):
        yield [_cell_text(cell) for cell in row.findall("./w:tc", NS)]


def _parse_main_group(label: str) -> tuple[int, str] | None:
    match = re.match(r"^\((\d+)\)\s*(.+?)\s*$", label)
    if match is None:
        return None
    number = int(match.group(1))
    name = match.group(2).strip()
    name = CORRECTED_MAIN_GROUP_NAMES.get(number, name)
    return number, name


def _float(value: str) -> float:
    value = value.strip()
    return 0.0 if not value else float(value)


def _load_interaction_overlay(path: Path) -> dict:
    """Load and validate one bidirectional interaction overlay source."""
    payload = json.loads(path.read_text())
    fixed_group = payload.get("fixed_main_group") or {}
    fixed_number = int(fixed_group.get("number"))
    rows = payload.get("interactions")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Interaction overlay {path} has no interaction rows")

    seen_pairs: set[tuple[int, int]] = set()
    for index, row in enumerate(rows):
        m = int(row["m"])
        n = int(row["n"])
        if n != fixed_number:
            raise ValueError(
                f"Interaction overlay {path} row {index} uses n={n}, "
                f"expected fixed main group {fixed_number}"
            )
        pair = tuple(sorted((m, n)))
        if m == n or pair in seen_pairs:
            raise ValueError(
                f"Interaction overlay {path} has duplicate/invalid pair {pair}"
            )
        seen_pairs.add(pair)
        for direction in ("m_to_n", "n_to_m"):
            coefficients = row.get(direction)
            if not isinstance(coefficients, dict):
                raise ValueError(
                    f"Interaction overlay {path} row {index} lacks {direction}"
                )
            for coefficient in ("a1", "a2", "a3"):
                float(coefficients[coefficient])
        float(row["Tmin_K"])
        if row.get("Tmax_K") is not None:
            float(row["Tmax_K"])
    return payload


def _apply_interaction_overlay(
    interactions: list[dict],
    overlay_path: Path,
) -> tuple[list[dict], dict]:
    """Replace/add all directed parameter records from one source overlay."""
    payload = _load_interaction_overlay(overlay_path)
    source_citation = payload["metadata"]["source_citation"]
    overlay_rows = payload["interactions"]
    original_pairs = {(record["i"], record["j"]) for record in interactions}
    overlay_directed_pairs = {
        directed
        for row in overlay_rows
        for directed in ((int(row["m"]), int(row["n"])),
                         (int(row["n"]), int(row["m"])))
    }
    interactions = [
        record for record in interactions
        if (record["i"], record["j"]) not in overlay_directed_pairs
    ]

    replaced: list[list[int]] = []
    added: list[list[int]] = []
    for row in overlay_rows:
        m = int(row["m"])
        n = int(row["n"])
        action = (
            "replaced"
            if (m, n) in original_pairs or (n, m) in original_pairs
            else "added"
        )
        (replaced if action == "replaced" else added).append([m, n])
        Tmin = float(row["Tmin_K"])
        source_Tmax = row.get("Tmax_K")
        Tmax = Tmin if source_Tmax is None else float(source_Tmax)
        for i, j, direction in (
            (m, n, "m_to_n"),
            (n, m, "n_to_m"),
        ):
            coefficients = row[direction]
            a3 = float(coefficients["a3"])
            interactions.append({
                "i": i,
                "j": j,
                "a1": float(coefficients["a1"]),
                "a2": float(coefficients["a2"]),
                "a3": a3,
                "a3_times_1000": 1000.0 * a3,
                "Tmin": Tmin,
                "Tmax": Tmax,
                "temperature_basis": row["temperature_basis"],
                "source_overlay": {
                    "file": overlay_path.name,
                    "table": payload["metadata"]["table"],
                    "doi": source_citation["doi"],
                    "direction": direction,
                    "markers": list(coefficients.get("markers") or []),
                    "action": action,
                    "source_Tmax_K": source_Tmax,
                },
            })

    diagnostics = {
        "file": overlay_path.name,
        "doi": source_citation["doi"],
        "fixed_main_group": int(payload["fixed_main_group"]["number"]),
        "unordered_pair_count": len(overlay_rows),
        "directed_parameter_count": 2 * len(overlay_rows),
        "replaced_unordered_pairs": replaced,
        "added_unordered_pairs": added,
    }
    return interactions, diagnostics


def parse_docx(
    path: Path,
    interaction_overlay_path: Path | None = LACTONE_OVERLAY_JSON,
) -> tuple[list[dict], list[dict], dict]:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))

    tables = root.findall(".//w:tbl", NS)
    if len(tables) < 2:
        raise RuntimeError(f"Expected at least two Word tables in {path}")

    subgroups: list[dict] = []
    rejected_subgroups: list[dict] = []
    current_main_group: int | None = None
    current_main_group_name = ""
    seen_subgroup_numbers: dict[int, dict] = {}

    for cells in _table_rows(tables[0]):
        if len(cells) == 1:
            parsed_main = _parse_main_group(cells[0])
            if parsed_main is not None:
                current_main_group, current_main_group_name = parsed_main
            continue
        if len(cells) < 4:
            continue
        try:
            number = int(cells[0])
            name = cells[1].strip()
            # Corrigendum: Supplement Table 1 transposed the Rk/Qk headers.
            r_value = _float(cells[2])
            q_value = _float(cells[3])
        except (TypeError, ValueError):
            continue

        if current_main_group is None:
            rejected_subgroups.append({
                "number": number,
                "name": name,
                "reason": "subgroup row appeared before a main group header",
            })
            continue

        main_group = current_main_group
        main_group_name = current_main_group_name

        # Corrigendum: main group 86 was erroneously defined as 84.
        if number == 179 and name == "C2Cl4":
            main_group = 86
            main_group_name = "C2Cl4"

        example = cells[4].strip() if len(cells) > 4 else ""
        example_groups = cells[5].strip() if len(cells) > 5 else ""
        corrections: list[str] = []

        # The supplement uses subgroup number 309 for both CH=NOH and the
        # corrigendum-added CH2(O)2 acetal.  Subgroup numbers are internal
        # identifiers, so preserve the oxime under a collision-free number
        # instead of deleting main group 82 from the parameter set.
        if number == 309 and name == "CH=NOH":
            number = 1309
            corrections.append("internal_id_collision_309_to_1309")

        corrected_example = CORRECTED_SUBGROUP_EXAMPLES.get(number)
        if number == 309 and name != "CH2(O)2":
            corrected_example = None
        if corrected_example is not None:
            example, example_groups = corrected_example
            corrections.append("corrigendum_example")

        if main_group_name == "ACCOO" and main_group == 83:
            main_group_name = "ACCO"
            corrections.append("corrigendum_main_group_83_name")

        record = {
            "number": number,
            "name": name,
            "main_group": main_group,
            "main_group_name": main_group_name,
            "R": r_value,
            "Q": q_value,
            "example": example,
            "example_groups": example_groups,
        }
        if corrections:
            record["corrections"] = corrections

        existing = seen_subgroup_numbers.get(number)
        if existing is not None:
            # Any remaining collision is an unrecognized source-data defect.
            keep_new = number == 309 and name == "CH2(O)2"
            if keep_new:
                rejected_subgroups.append({
                    **existing,
                    "reason": "duplicate subgroup number superseded by corrigendum row",
                })
                subgroups.remove(existing)
                subgroups.append(record)
                seen_subgroup_numbers[number] = record
            else:
                rejected_subgroups.append({
                    **record,
                    "reason": "duplicate subgroup number; canonical row already present",
                })
            continue

        subgroups.append(record)
        seen_subgroup_numbers[number] = record

    interactions: list[dict] = []
    for cells in _table_rows(tables[1]):
        if len(cells) < 7:
            continue
        try:
            main_group_i = int(cells[0])
            main_group_j = int(cells[1])
            a1 = _float(cells[2])
            a2 = _float(cells[3])
            a3_times_1000 = _float(cells[4])
            t_min = _float(cells[5])
            t_max = _float(cells[6])
        except (TypeError, ValueError):
            continue
        interactions.append({
            "i": main_group_i,
            "j": main_group_j,
            "a1": a1,
            "a2": a2,
            "a3": a3_times_1000 / 1000.0,
            "a3_times_1000": a3_times_1000,
            "Tmin": t_min,
            "Tmax": t_max,
        })

    # A directed UNIFAC pair is usable only when both directions were fitted.
    # The supplement contains one orphan 59->13 record; excluding that orphan
    # restores the publication's stated 984 group-group interaction pairs.
    parsed_directed_pairs = {(record["i"], record["j"]) for record in interactions}
    rejected_interactions = [
        record for record in interactions
        if (record["j"], record["i"]) not in parsed_directed_pairs
    ]
    interactions = [
        record for record in interactions
        if (record["j"], record["i"]) in parsed_directed_pairs
    ]

    overlay_diagnostics = None
    if interaction_overlay_path is not None:
        interactions, overlay_diagnostics = _apply_interaction_overlay(
            interactions,
            interaction_overlay_path,
        )

    main_groups = {
        record["main_group"]: record["main_group_name"]
        for record in subgroups
    }
    directed_pairs = {(record["i"], record["j"]) for record in interactions}
    missing_reverse = sorted(
        [list(pair) for pair in directed_pairs if (pair[1], pair[0]) not in directed_pairs]
    )
    interaction_main_groups = sorted({
        group
        for record in interactions
        for group in (record["i"], record["j"])
    })

    diagnostics = {
        "subgroup_count": len(subgroups),
        "main_group_count": len(main_groups),
        "directed_interaction_count": len(interactions),
        "unordered_interaction_pair_count": len({
            tuple(sorted((record["i"], record["j"])))
            for record in interactions
        }),
        "interaction_main_groups_without_subgroups": [
            group for group in interaction_main_groups if group not in main_groups
        ],
        "main_groups_without_interactions": [
            group for group in sorted(main_groups) if group not in interaction_main_groups
        ],
        "directed_pairs_missing_reverse": missing_reverse,
        "rejected_subgroups": rejected_subgroups,
        "rejected_interactions": rejected_interactions,
    }
    if overlay_diagnostics is not None:
        diagnostics["interaction_overlay"] = overlay_diagnostics
    return subgroups, interactions, diagnostics


def main() -> None:
    subgroups, interactions, diagnostics = parse_docx(SOURCE_DOCX)
    payload = {
        "metadata": {
            "model": "NIST-modified UNIFAC",
            "source": {
                "paper": "Kang, Diky, and Frenkel, Fluid Phase Equilibria 388 (2015) 128-141",
                "doi": "10.1016/j.fluid.2014.12.042",
                "supplement": SOURCE_DOCX.name,
                "corrigendum": "Kang, Diky, and Frenkel, Fluid Phase Equilibria 440 (2017) 122-123",
                "corrigendum_doi": "10.1016/j.fluid.2017.02.014",
                "interaction_overlays": [
                    {
                        "paper": "Dantas and Ceriani, Fluid Phase Equilibria 565 (2023) 113673",
                        "doi": "10.1016/j.fluid.2022.113673",
                        "table": "Table 1",
                        "file": LACTONE_OVERLAY_JSON.name,
                    }
                ],
            },
            "equations": {
                "interaction_energy": "delta_u_ij = a1 + a2*T + a3*T^2",
                "temperature_unit": "K",
                "a3_note": "The source table reports 1000*a3; JSON a3 is already divided by 1000.",
                "combinatorial": "Modified UNIFAC combinatorial term using phi and phi_prime with r_i^(3/4).",
            },
            "corrections_applied": [
                "Supplement Table 1 Rk/Qk labels transposed; cells 3 and 4 are stored as R and Q respectively.",
                "Main group 86 C2Cl4 restored from supplement typo that labeled it as 84.",
                "Main group 83 spelling corrected to ACCO.",
                "Corrigendum example corrections applied for subgroups 39, 160, 162, 163, 188, 198, 305, 306, and 309.",
                "Conflicting subgroup 309 CH=NOH preserved under collision-free internal subgroup number 1309; corrigendum subgroup 309 CH2(O)2 kept canonical.",
                "Orphan directed interaction 59->13 rejected because no reverse parameter is published.",
                "Dantas-Ceriani 2023 Table 1 lactone interactions overlayed onto main group 63, replacing original pairs and adding newly regressed pairs.",
            ],
            "implementation_notes": [
                "The paper recommends use inside each parameter's specified temperature range.",
                "The paper states this matrix was fitted with emphasis on VLE, not specifically LLE.",
            ],
        },
        "subgroups": subgroups,
        "interactions": interactions,
        "diagnostics": diagnostics,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"Wrote {OUTPUT_JSON}")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
