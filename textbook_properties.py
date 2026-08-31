"""Offline textbook property lookup.

The data file is generated from Appendix B of the bundled Smith/Van Ness
thermodynamics textbook. Runtime code reads the JSON directly so property lookup
does not repeatedly parse the PDF.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


class TextbookPropertyLibrary:
    """Lookup helper for extracted textbook pure-species properties."""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else Path(__file__).parent / "data" / "textbook_properties.json"
        self._loaded = False
        self._chemicals: dict[str, dict[str, Any]] = {}
        self._aliases: dict[str, str] = {}

    @staticmethod
    def _normalize(identifier: str) -> str:
        identifier = identifier.strip().lower()
        identifier = identifier.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-")
        identifier = re.sub(r"\s+", " ", identifier)
        return identifier

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if not self.path.exists():
            return

        with open(self.path, "r") as f:
            payload = json.load(f)

        self._chemicals = payload.get("chemicals", {})
        formula_to_names: dict[str, list[str]] = {}

        for name, entry in self._chemicals.items():
            self._aliases[self._normalize(name)] = name

            formula = entry.get("formula")
            if formula:
                formula_to_names.setdefault(self._normalize(formula), []).append(name)

            normalized = self._normalize(name)
            if normalized.startswith("n-"):
                self._aliases.setdefault(normalized[2:], name)
            if normalized.startswith("iso-"):
                self._aliases.setdefault("isobutane" if normalized == "iso-butane" else normalized.replace("iso-", "iso"), name)

        # Formula aliases are only safe when the formula maps to one species in
        # the extracted table; otherwise isomers would be silently confused.
        for formula, names in formula_to_names.items():
            if len(names) == 1:
                self._aliases.setdefault(formula, names[0])

    def get(self, identifier: str) -> Optional[dict[str, Any]]:
        self._load()
        canonical = self._aliases.get(self._normalize(identifier))
        if not canonical:
            return None
        entry = dict(self._chemicals[canonical])
        entry["canonical_name"] = canonical
        return entry


_library: Optional[TextbookPropertyLibrary] = None


def get_textbook_property_library() -> TextbookPropertyLibrary:
    global _library
    if _library is None:
        _library = TextbookPropertyLibrary()
    return _library
