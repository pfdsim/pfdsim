"""Shared solid material-form vocabulary for parser and property resolvers."""

from __future__ import annotations


SOLID_MATERIAL_FORMS = frozenset({
    "unspecified", "crystalline", "glass", "hydrate", "solvate",
})

_ALIASES = {
    "": "unspecified",
    "unknown": "unspecified",
    "default": "unspecified",
    "solid": "unspecified",
    "crystal": "crystalline",
    "crystalline solid": "crystalline",
    "amorphous": "glass",
    "vitreous": "glass",
    "glassy": "glass",
}


def normalize_solid_material_form(value) -> str:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    normalized = _ALIASES.get(text, text.replace(" ", "_"))
    if normalized not in SOLID_MATERIAL_FORMS:
        allowed = ", ".join(sorted(SOLID_MATERIAL_FORMS))
        raise ValueError(f"Unsupported solid material form {value!r}; expected one of {allowed}")
    return normalized

