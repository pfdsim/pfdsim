#!/usr/bin/env python3
"""Compile local solid heat-capacity sources into a portable SQLite artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from chemicals import heat_capacity as source_cp
from chemicals.heat_capacity import PiecewiseHeatCapacity
from chemicals.identifiers import check_CAS, search_chemical


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "data" / "solid_heat_capacity.sqlite"
SCHEMA_VERSION = 1
CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")
CRC_GRAPHITE_BAD_CAS = "74-82-8"
GRAPHITE_CAS = "7782-42-5"
GENERIC_CARBON_CAS = "7440-44-0"
DIAMOND_CAS = "7782-40-3"
JANAF_PSEUDO_IDENTIFIERS = {"2099576000-00-0", "2099592000-00-0"}

SOURCE_SPECS = {
    "janaf_1998": (1, 0.985, "NIST-JANAF", "NIST-JANAF 1998 solid table"),
    "webbook_shomate": (2, 0.975, "NIST-JANAF", "NIST WebBook solid Shomate"),
    "crc_temperature_table": (3, 0.970, "CRC Handbook", "CRC solid Cp temperature table"),
    "crc_standard_point": (4, 0.960, "CRC Handbook", "CRC standard solid Cp at 298.15 K"),
    "perry_151": (5, None, "Perry 2-151", "Perry 2-151 solid correlation"),
    "perry_151_point": (6, 0.900, "Perry 2-151", "Perry 2-151 point from unbounded row"),
}


@dataclass
class Candidate:
    cas: str
    name: str
    formula: str
    source: str
    material_form: str
    polymorph: str
    is_default_form: bool
    kernel: Any
    identity_status: str = "validated"
    identity_notes: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def source_priority(self) -> int:
        return SOURCE_SPECS[self.source][0]

    @property
    def lineage(self) -> str:
        return SOURCE_SPECS[self.source][2]

    @property
    def source_label(self) -> str:
        return SOURCE_SPECS[self.source][3]

    @property
    def fingerprint(self) -> str:
        payload = {
            "cas": self.cas, "source": self.source,
            "material_form": self.material_form, "polymorph": self.polymorph,
            "kernel": self.kernel.to_payload(), "details": self.details,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass
class Quarantine:
    identifier: str
    source: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


def _kernel_classes():
    from property_resolution.solid_cp import (
        ConstantSolidCpKernel, Perry151SolidCpKernel, PiecewiseSolidCpKernel,
        ShomateSolidCpKernel, TabularSolidCpKernel,
    )
    return (
        ConstantSolidCpKernel, Perry151SolidCpKernel, PiecewiseSolidCpKernel,
        ShomateSolidCpKernel, TabularSolidCpKernel,
    )


def valid_cas(value: str) -> bool:
    text = str(value or "").strip()
    return bool(CAS_PATTERN.fullmatch(text)) and check_CAS(text)


def metadata_for_cas(cas: str) -> tuple[str, str, str]:
    try:
        metadata = search_chemical(cas)
    except Exception:
        return "", "", "unavailable"
    if str(metadata.CASs) != cas:
        return "", "", f"CAS lookup redirected to {metadata.CASs}"
    return str(metadata.common_name or ""), str(metadata.formula or ""), "validated"


def non_water_elements(formula: str) -> set[str]:
    return {
        token for token in re.findall(r"[A-Z][a-z]?", str(formula or ""))
        if token not in {"H", "O"}
    }


def inferred_form(name: str, *, default: str = "crystalline") -> tuple[str, str]:
    text = str(name or "").strip()
    lowered = text.casefold()
    if "hydrate" in lowered:
        return "hydrate", ""
    if "solvate" in lowered:
        return "solvate", ""
    if "glass" in lowered or "vitreous" in lowered or "amorphous" in lowered:
        return "glass", ""
    for token in ("diamond", "graphite", "alpha", "beta", "gamma", "delta"):
        if token in lowered:
            return default, token
    return default, ""


def common(cas: str, source: str, Tmin: float, Tmax: float, *, quality: Optional[float] = None,
           form: str = "crystalline", polymorph: str = "", notes: str = "") -> dict[str, Any]:
    rank, base_quality, _, label = SOURCE_SPECS[source]
    return dict(
        Tmin=Tmin, Tmax=Tmax,
        quality=base_quality if quality is None else quality,
        source=label, method=f"canonical_{source}_solid_cp",
        notes=notes, source_priority=rank,
        material_form=form, polymorph=polymorph,
    )


def split_janaf_table(temperatures, values) -> tuple[list[tuple[list[float], list[float]]], list[float]]:
    segments: list[tuple[list[float], list[float]]] = []
    transitions: list[float] = []
    current_T: list[float] = []
    current_Cp: list[float] = []
    for temperature, cp in zip(temperatures, values):
        T, value = float(temperature), float(cp)
        if T <= 0.0 or value <= 0.0:
            continue
        if current_T and abs(T - current_T[-1]) <= 1.0e-12:
            if len(current_T) >= 2:
                segments.append((current_T, current_Cp))
            transitions.append(T)
            current_T, current_Cp = [T], [value]
        else:
            current_T.append(T)
            current_Cp.append(value)
    if len(current_T) >= 2:
        segments.append((current_T, current_Cp))
    return segments, transitions


def collect_janaf(candidates: list[Candidate], quarantines: list[Quarantine]) -> None:
    _, _, PiecewiseKernel, _, TabularKernel = _kernel_classes()
    for raw_cas, table in source_cp.Cp_dict_JANAF_solid.items():
        cas = str(raw_cas)
        if cas in JANAF_PSEUDO_IDENTIFIERS or not valid_cas(cas):
            quarantines.append(Quarantine(cas, "janaf_1998", "identifier is not a usable CAS"))
            continue
        segments, transitions = split_janaf_table(*table)
        if not segments:
            quarantines.append(Quarantine(cas, "janaf_1998", "no positive two-point segment"))
            continue
        name, formula, identity_status = metadata_for_cas(cas)
        children = []
        for Ts, Cps in segments:
            children.append(TabularKernel(
                **common(cas, "janaf_1998", Ts[0], Ts[-1]),
                temperatures=tuple(Ts), values=tuple(Cps),
            ))
        if len(children) == 1:
            kernel = children[0]
        else:
            kernel = PiecewiseKernel(
                **common(cas, "janaf_1998", children[0].Tmin, children[-1].Tmax),
                segments=tuple(children), transitions=tuple(transitions),
            )
        candidates.append(Candidate(
            cas, name or cas, formula, "janaf_1998", "crystalline", "", True, kernel,
            identity_status=identity_status,
            details={"knots": len(table[0]), "segments": len(children), "transitions": transitions},
        ))


def collect_webbook(candidates: list[Candidate], quarantines: list[Quarantine]) -> None:
    _, _, PiecewiseKernel, ShomateKernel, _ = _kernel_classes()
    for cas, model in source_cp.WebBook_Shomate_solids.items():
        if not valid_cas(cas):
            quarantines.append(Quarantine(cas, "webbook_shomate", "invalid CAS"))
            continue
        models = model.models if isinstance(model, PiecewiseHeatCapacity) else [model]
        children = []
        invalid = None
        for item in models:
            try:
                A, B_si, C_si, D_si, E_si = (float(value) for value in item.coeffs)
                # ``chemicals`` stores Shomate in direct SI-temperature form;
                # the portable kernel retains the public NIST t=T/1000 form.
                coefficients = (
                    A,
                    B_si * 1.0e3,
                    C_si * 1.0e6,
                    D_si * 1.0e9,
                    E_si / 1.0e6,
                )
                child = ShomateKernel(
                    **common(cas, "webbook_shomate", item.Tmin, item.Tmax),
                    coefficients=coefficients,
                )
            except ValueError as exc:
                invalid = str(exc)
                break
            children.append(child)
        if invalid:
            quarantines.append(Quarantine(cas, "webbook_shomate", invalid))
            continue
        transitions = []
        janaf_T = source_cp.Cp_dict_JANAF_solid.get(cas, [[], []])[0]
        duplicate_T = [janaf_T[i] for i in range(len(janaf_T) - 1) if janaf_T[i + 1] == janaf_T[i]]
        for first, second in zip(children, children[1:]):
            boundary = second.Tmin
            first_cp, second_cp = first._cp_native(first.Tmax), second._cp_native(second.Tmin)
            jump = abs(first_cp - second_cp) / max((abs(first_cp) + abs(second_cp)) / 2.0, 1.0e-30)
            if any(abs(boundary - value) <= 1.0 for value in duplicate_T) or jump > 0.01:
                transitions.append(boundary)
        if len(children) == 1:
            kernel = children[0]
        else:
            kernel = PiecewiseKernel(
                **common(cas, "webbook_shomate", children[0].Tmin, children[-1].Tmax),
                segments=tuple(children), transitions=tuple(transitions),
            )
        name, formula, identity_status = metadata_for_cas(cas)
        candidates.append(Candidate(
            cas, name or cas, formula, "webbook_shomate", "crystalline", "", True, kernel,
            identity_status=identity_status,
            details={"segments": len(children), "transitions": transitions},
        ))


def perry_quality(error: Any) -> float:
    text = str(error or "").strip().casefold()
    approximate = text.endswith("a")
    try:
        value = float(text.rstrip("a"))
    except ValueError:
        return 0.92
    if value <= 1.0:
        quality = 0.98
    elif value <= 3.0:
        quality = 0.96
    elif value <= 5.0:
        quality = 0.94
    elif value <= 10.0:
        quality = 0.90
    else:
        quality = 0.86
    return quality - (0.01 if approximate else 0.0)


def collect_perry(candidates: list[Candidate], quarantines: list[Quarantine]) -> None:
    ConstantKernel, PerryKernel, _, _, _ = _kernel_classes()
    for original_cas, phases in source_cp.Cp_dict_PerryI.items():
        for phase in ("c", "gls"):
            row = phases.get(phase)
            if row is None:
                continue
            cas = str(original_cas)
            form = "glass" if phase == "gls" else "crystalline"
            polymorph = str(row.get("Subphase") or "").strip()
            if cas == GENERIC_CARBON_CAS and polymorph.casefold() == "diamond":
                cas = DIAMOND_CAS
            if not valid_cas(cas):
                quarantines.append(Quarantine(cas, "perry_151", "invalid or absent CAS", dict(row)))
                continue
            source_formula = str(row.get("Formula") or "").strip()
            name, resolved_formula, identity_status = metadata_for_cas(cas)
            if resolved_formula:
                source_elements = non_water_elements(source_formula)
                resolved_elements = non_water_elements(resolved_formula)
                if source_elements and resolved_elements and source_elements != resolved_elements:
                    quarantines.append(Quarantine(
                        cas, "perry_151", "source formula conflicts with CAS identity",
                        {"source_formula": source_formula, "resolved_formula": resolved_formula, "resolved_name": name},
                    ))
                    continue
            coefficients = tuple(float(row.get(key) or 0.0) for key in ("Const", "Lin", "Quadinv", "Quad"))
            quality = perry_quality(row.get("Error"))
            try:
                Tmin = float(row.get("Tmin"))
                Tmax = float(row.get("Tmax"))
                kernel = PerryKernel(
                    **common(cas, "perry_151", Tmin, Tmax, quality=quality, form=form, polymorph=polymorph),
                    coefficients=coefficients,
                )
                source = "perry_151"
            except (TypeError, ValueError):
                Tmin = 293.15
                Tmax = 303.15
                try:
                    a, b, c, d = coefficients
                    value = 4.184 * (a + b * 298.15 + c / 298.15**2 + d * 298.15**2)
                    kernel = ConstantKernel(
                        **common(cas, "perry_151_point", Tmin, Tmax, quality=min(quality, 0.90), form=form, polymorph=polymorph),
                        value=value,
                    )
                    source = "perry_151_point"
                except ValueError as exc:
                    quarantines.append(Quarantine(cas, "perry_151", str(exc), dict(row)))
                    continue
            is_default = not polymorph and form == "crystalline"
            if cas == DIAMOND_CAS:
                is_default = True
            candidates.append(Candidate(
                cas, name or cas, source_formula or resolved_formula, source,
                form, polymorph, is_default, kernel,
                identity_status=identity_status,
                details={"phase": phase, "error": row.get("Error"), "original_cas": original_cas},
            ))


def collect_crc_curve(candidates: list[Candidate], quarantines: list[Quarantine]) -> None:
    _, _, _, _, TabularKernel = _kernel_classes()
    path = Path(source_cp.__file__).parent / "Heat Capacity" / "CRCHeatCapacitySolids.tsv"
    table = pd.read_csv(path, sep="\t", dtype={"CASRN": str})
    for _, row in table.iterrows():
        original_cas = str(row["CASRN"])
        name = str(row["NAME/T(K) J/mol/K"]).strip()
        cas = GRAPHITE_CAS if original_cas == CRC_GRAPHITE_BAD_CAS and name.casefold() == "graphite" else original_cas
        if not valid_cas(cas):
            quarantines.append(Quarantine(cas, "crc_temperature_table", "invalid CAS"))
            continue
        temperatures, values = [], []
        for column in table.columns[2:]:
            value = row[column]
            if pd.notna(value):
                temperatures.append(float(column))
                values.append(float(value))
        if len(values) < 2 or any(value <= 0.0 for value in values):
            quarantines.append(Quarantine(cas, "crc_temperature_table", "invalid tabular values"))
            continue
        form, polymorph = inferred_form(name)
        kernel = TabularKernel(
            **common(cas, "crc_temperature_table", temperatures[0], temperatures[-1], form=form, polymorph=polymorph),
            temperatures=tuple(temperatures), values=tuple(values),
        )
        resolved_name, formula, identity_status = metadata_for_cas(cas)
        candidates.append(Candidate(
            cas, name or resolved_name or cas, formula, "crc_temperature_table",
            form, polymorph, True, kernel,
            identity_status=identity_status,
            identity_notes=(f"remapped from erroneous {original_cas}" if cas != original_cas else ""),
            details={"points": len(values), "original_cas": original_cas},
        ))


def collect_crc_scalar(candidates: list[Candidate]) -> None:
    ConstantKernel, _, _, _, _ = _kernel_classes()
    rows = source_cp.CRC_standard_data[source_cp.CRC_standard_data.Cps.notna()]
    for cas, row in rows.iterrows():
        cas = str(cas)
        if not valid_cas(cas):
            continue
        name = str(row.Chemical or cas)
        form, polymorph = inferred_form(name)
        resolved_name, formula, identity_status = metadata_for_cas(cas)
        kernel = ConstantKernel(
            **common(cas, "crc_standard_point", 293.15, 303.15, form=form, polymorph=polymorph),
            value=float(row.Cps),
        )
        candidates.append(Candidate(
            cas, name or resolved_name or cas, formula, "crc_standard_point",
            form, polymorph, True, kernel,
            identity_status=identity_status,
            details={"reference_temperature_K": 298.15},
        ))


def crosschecks(candidates: list[Candidate]) -> list[dict[str, Any]]:
    result = []
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.cas, candidate.material_form)].append(candidate)
    for (cas, form), items in grouped.items():
        for index, first in enumerate(items):
            for second in items[index + 1:]:
                low = max(first.kernel.Tmin, second.kernel.Tmin)
                high = min(first.kernel.Tmax, second.kernel.Tmax)
                if high < low:
                    continue
                temperatures = np.linspace(low, high, 41) if high > low else np.asarray([low])
                errors = []
                for temperature in temperatures:
                    try:
                        a, b = first.kernel.cp(float(temperature)), second.kernel.cp(float(temperature))
                    except (ValueError, OverflowError):
                        continue
                    errors.append(abs(a - b) / max((abs(a) + abs(b)) / 2.0, 1.0e-30) * 100.0)
                if errors:
                    result.append({
                        "cas": cas, "material_form": form,
                        "first_source": first.source, "second_source": second.source,
                        "same_lineage": first.lineage == second.lineage,
                        "Tmin_K": low, "Tmax_K": high,
                        "points": len(errors), "median_percent": float(np.median(errors)),
                        "p90_percent": float(np.quantile(errors, 0.9)),
                        "maximum_percent": max(errors),
                    })
    return result


def payload_with_fingerprint(payload: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    """Stamp one canonical-source fingerprint through nested phase segments."""
    result = dict(payload)
    result["source_fingerprint"] = fingerprint
    if isinstance(result.get("segments"), list):
        result["segments"] = [
            payload_with_fingerprint(item, fingerprint) for item in result["segments"]
        ]
    return result


def build() -> tuple[list[Candidate], list[Quarantine], list[dict[str, Any]]]:
    candidates: list[Candidate] = []
    quarantines: list[Quarantine] = []
    collect_janaf(candidates, quarantines)
    collect_webbook(candidates, quarantines)
    collect_crc_curve(candidates, quarantines)
    collect_crc_scalar(candidates)
    collect_perry(candidates, quarantines)
    candidates.sort(key=lambda item: (item.cas, item.source_priority, item.material_form, item.polymorph))
    return candidates, quarantines, crosschecks(candidates)


def write_database(path: Path, candidates: list[Candidate], quarantines: list[Quarantine], checks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="solid-cp-", suffix=".sqlite", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with sqlite3.connect(temporary) as connection:
            connection.executescript("""
                PRAGMA journal_mode=DELETE;
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
                CREATE TABLE canonical_solid_cp (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cas TEXT NOT NULL, name TEXT, formula TEXT,
                    material_form TEXT NOT NULL, polymorph TEXT NOT NULL,
                    is_default_form INTEGER NOT NULL,
                    source TEXT NOT NULL, source_label TEXT NOT NULL,
                    source_lineage TEXT NOT NULL, source_priority INTEGER NOT NULL,
                    quality REAL NOT NULL, Tmin_K REAL NOT NULL, Tmax_K REAL NOT NULL,
                    identity_status TEXT NOT NULL, identity_notes TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    kernel_payload_json TEXT NOT NULL, details_json TEXT NOT NULL
                );
                CREATE INDEX idx_solid_cp_cas ON canonical_solid_cp(cas);
                CREATE TABLE source_candidate_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cas TEXT NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL,
                    material_form TEXT, polymorph TEXT, reason TEXT,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE source_identity_audit (
                    identity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cas TEXT NOT NULL, source TEXT NOT NULL,
                    identity_status TEXT NOT NULL, notes TEXT NOT NULL,
                    name TEXT, formula TEXT
                );
                CREATE TABLE source_transition_boundary (
                    boundary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cas TEXT NOT NULL, source TEXT NOT NULL,
                    material_form TEXT NOT NULL, transition_temperature_K REAL NOT NULL,
                    transition_enthalpy_available INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE source_crosscheck (
                    check_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cas TEXT NOT NULL, material_form TEXT NOT NULL,
                    first_source TEXT NOT NULL, second_source TEXT NOT NULL,
                    same_lineage INTEGER NOT NULL, Tmin_K REAL, Tmax_K REAL,
                    points INTEGER NOT NULL, median_percent REAL,
                    p90_percent REAL, maximum_percent REAL
                );
                CREATE TABLE source_quarantine (
                    quarantine_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier TEXT NOT NULL, source TEXT NOT NULL,
                    reason TEXT NOT NULL, details_json TEXT NOT NULL
                );
            """)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_count": len(candidates) + len(quarantines),
                "admitted_count": len(candidates),
                "unique_cas": len({item.cas for item in candidates}),
                "candidate_sources": dict(Counter(item.source for item in candidates)),
                "quarantine_count": len(quarantines),
                "crosscheck_count": len(checks),
                "transition_count": sum(len(item.kernel.transition_temperatures) for item in candidates),
                "selection_policy": "runtime quality/range/form arbitration over admitted native candidates",
            }
            connection.executemany(
                "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
                [(key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()],
            )
            for item in candidates:
                fingerprint = item.fingerprint
                payload = payload_with_fingerprint(item.kernel.to_payload(), fingerprint)
                connection.execute("""
                    INSERT INTO canonical_solid_cp(
                        cas,name,formula,material_form,polymorph,is_default_form,
                        source,source_label,source_lineage,source_priority,quality,Tmin_K,Tmax_K,
                        identity_status,identity_notes,source_fingerprint,kernel_payload_json,details_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    item.cas, item.name, item.formula, item.material_form, item.polymorph,
                    int(item.is_default_form), item.source, item.source_label, item.lineage,
                    item.source_priority, item.kernel.quality, item.kernel.Tmin, item.kernel.Tmax,
                    item.identity_status, item.identity_notes, fingerprint,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    json.dumps(item.details, sort_keys=True, default=str),
                ))
                connection.execute(
                    "INSERT INTO source_candidate_audit(cas,source,status,material_form,polymorph,reason,details_json) VALUES (?,?,?,?,?,?,?)",
                    (item.cas, item.source, "admitted", item.material_form, item.polymorph, "", json.dumps(item.details, sort_keys=True, default=str)),
                )
                connection.execute(
                    "INSERT INTO source_identity_audit(cas,source,identity_status,notes,name,formula) VALUES (?,?,?,?,?,?)",
                    (item.cas, item.source, item.identity_status, item.identity_notes, item.name, item.formula),
                )
                for temperature in item.kernel.transition_temperatures:
                    connection.execute(
                        "INSERT INTO source_transition_boundary(cas,source,material_form,transition_temperature_K) VALUES (?,?,?,?)",
                        (item.cas, item.source, item.material_form, temperature),
                    )
            connection.executemany(
                "INSERT INTO source_crosscheck(cas,material_form,first_source,second_source,same_lineage,Tmin_K,Tmax_K,points,median_percent,p90_percent,maximum_percent) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(x["cas"], x["material_form"], x["first_source"], x["second_source"], int(x["same_lineage"]), x["Tmin_K"], x["Tmax_K"], x["points"], x["median_percent"], x["p90_percent"], x["maximum_percent"]) for x in checks],
            )
            connection.executemany(
                "INSERT INTO source_quarantine(identifier,source,reason,details_json) VALUES (?,?,?,?)",
                [(x.identifier, x.source, x.reason, json.dumps(x.details, sort_keys=True, default=str)) for x in quarantines],
            )
            connection.executemany(
                "INSERT INTO source_candidate_audit(cas,source,status,material_form,polymorph,reason,details_json) VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        item.identifier, item.source, "quarantined", "", "",
                        item.reason, json.dumps(item.details, sort_keys=True, default=str),
                    )
                    for item in quarantines
                ],
            )
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        temporary.replace(path)
        path.chmod(0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    candidates, quarantines, checks = build()
    write_database(args.output, candidates, quarantines, checks)
    print(json.dumps({
        "output": str(args.output), "records": len(candidates),
        "unique_cas": len({item.cas for item in candidates}),
        "sources": dict(Counter(item.source for item in candidates)),
        "quarantines": len(quarantines), "crosschecks": len(checks),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
