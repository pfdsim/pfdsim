"""Lookup support for the local Antoine coefficient table."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


MMHG_TO_BAR = 0.001333223684


@dataclass(frozen=True)
class AntoineTableEntry:
    """One Antoine row converted to project units."""

    record_id: str
    formula: str
    name: str
    A: float
    B: float
    C: float
    T_min: float
    T_max: float
    source: str

    def vapor_pressure(self, T_K: float) -> float:
        T_C = T_K - 273.15
        return 10 ** (self.A - self.B / (self.C + T_C))


def normalize_identifier(identifier: str) -> str:
    """Normalize a compound name/formula for table matching."""
    return re.sub(r"[^a-z0-9]+", "", str(identifier).lower())


def normalize_formula(formula: str) -> str:
    """Normalize chemical formula text without losing case semantics."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(formula)).upper()


class AntoineTable:
    """Parsed Antoine table with conservative name/formula lookup."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or Path(__file__).parent / "data" / "antoine.txt"
        self._by_name: dict[str, list[AntoineTableEntry]] = {}
        self._by_formula: dict[str, list[AntoineTableEntry]] = {}
        self._load()

    def get(self, identifier: str, T: Optional[float] = None) -> Optional[AntoineTableEntry]:
        """Return the best Antoine row for an identifier and optional temperature."""
        entries = self.entries(identifier)
        if not entries:
            return None
        return self._select_entry(list(entries), T)

    def entries(self, identifier: str) -> tuple[AntoineTableEntry, ...]:
        """Return every conservatively matched Antoine row for an identifier."""
        name_key = normalize_identifier(identifier)
        entries = self._by_name.get(name_key)
        if entries:
            return tuple(entries)

        formula_key = normalize_formula(identifier)
        entries = self._by_formula.get(formula_key)
        if not entries:
            return ()

        # Molecular formula alone can identify structural isomers. Use it only
        # when all matching rows are aliases/ranges for the same named compound.
        compound_names = {normalize_identifier(entry.name) for entry in entries}
        if len(compound_names) > 1:
            return ()
        return tuple(dict.fromkeys(entries))

    def _load(self):
        if not self.path.exists():
            return

        with open(self.path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    entry = AntoineTableEntry(
                        record_id=(row.get("ID") or "").strip(),
                        formula=(row.get("Formula") or "").strip(),
                        name=(row.get("Compound Name") or "").strip(),
                        A=float(row["A"]) + math.log10(MMHG_TO_BAR),
                        B=float(row["B"]),
                        C=float(row["C"]),
                        T_min=float(row["TMIN"]) + 273.15,
                        T_max=float(row["TMAX"]) + 273.15,
                        source="data/antoine.txt",
                    )
                except (KeyError, TypeError, ValueError):
                    continue

                if not entry.name or entry.T_max <= entry.T_min or entry.B <= 0:
                    continue

                self._by_name.setdefault(normalize_identifier(entry.name), []).append(entry)
                formula = normalize_formula(entry.formula)
                if formula:
                    self._by_formula.setdefault(formula, []).append(entry)
                    # The table has a few OCR-like formula quirks, e.g. H20 for
                    # water. Keep the correction local to formula matching.
                    corrected = formula.replace("0", "O")
                    if corrected != formula:
                        self._by_formula.setdefault(corrected, []).append(entry)

    @staticmethod
    def _select_entry(
        entries: list[AntoineTableEntry],
        T: Optional[float],
    ) -> AntoineTableEntry:
        if T is not None:
            in_range = [entry for entry in entries if entry.T_min <= T <= entry.T_max]
            if in_range:
                return min(
                    in_range,
                    key=lambda entry: (
                        entry.T_max - entry.T_min,
                        abs((entry.T_min + entry.T_max) / 2.0 - T),
                    ),
                )

        reference_T = T if T is not None else 298.15
        ambient = [entry for entry in entries if entry.T_min <= reference_T <= entry.T_max]
        if ambient:
            return min(ambient, key=lambda entry: entry.T_max - entry.T_min)

        return min(
            entries,
            key=lambda entry: min(abs(entry.T_min - reference_T), abs(entry.T_max - reference_T)),
        )


@lru_cache(maxsize=1)
def get_antoine_table() -> AntoineTable:
    return AntoineTable()
