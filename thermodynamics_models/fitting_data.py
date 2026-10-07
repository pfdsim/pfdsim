"""Extract and interpret experimental tables without guessing missing conditions.

This module produces an import proposal. The fitting module remains the single
authority for validating normalized observations. Uncertain mappings and units
are visible in the proposal; the browser can resolve them without rewriting data.
"""

from __future__ import annotations

import csv
from html import unescape
from html.parser import HTMLParser
import json
import io
import math
import re
import unicodedata

import numpy as np

if __package__.split(".", 1)[0] == "pfdsim":
    from ..unit_conversions import pressure_to_bar, temperature_to_kelvin
else:
    from unit_conversions import pressure_to_bar, temperature_to_kelvin


FIELD_LABELS = {
    "temperature": "Temperature",
    "pressure": "Pressure",
    "x1": "Liquid composition",
    "y1": "Vapor composition",
    "x1_alpha": "Liquid α composition",
    "x1_beta": "Liquid β composition",
    "enthalpy": "Excess enthalpy",
    "gamma1_inf": "Component 1 γ∞",
    "gamma2_inf": "Component 2 γ∞",
    "kind": "Observation kind",
    "weight": "Row weight",
    "pin": "Hard pin",
    "validation_only": "Validation-only point",
    "pin_tolerance": "Pin tolerance",
    "sigma": "Residual uncertainty",
    "source": "Source",
    "group": "Validation group",
    "id": "Point ID",
    "ignore": "Ignore column",
}
_CANONICAL = {
    "T_K": "temperature",
    "T_C": "temperature",
    "P_bar": "pressure",
    "P_kPa": "pressure",
    "P_atm": "pressure",
    "HE_J_mol": "enthalpy",
    "HE_kJ_mol": "enthalpy",
    "type": "kind",
    **{
        key: key
        for key in FIELD_LABELS
        if key not in ("temperature", "pressure", "enthalpy", "ignore")
    },
}
_NUM = r"[+−–-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+−–-]?\d+)?"
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
ENTHALPY_UNITS = {"J/mol": 1.0, "kJ/mol": 1000.0, "cal/mol": 4.184, "kcal/mol": 4184.0}
KINDS = ("VLE", "LLE", "HE", "GAMMA_INF", "AZEOTROPE", "VLLE", "UCST", "LCST")
MISSING_TOKENS = frozenset(
    (
        "",
        "?",
        "??",
        "-",
        "--",
        "—",
        "–",
        "−",
        "..",
        "...",
        "…",
        "none",
        "null",
        "nil",
        "na",
        "n/a",
        "n.a.",
        "nan",
        "<na>",
        "#n/a",
        "nd",
        "n.d.",
        "missing",
        "unknown",
        "unavailable",
        "not available",
        "not measured",
        "not reported",
    )
)


def is_missing_cell(cell):
    """Recognize explicit absence without treating arbitrary bad data as missing."""
    return (
        cell is None
        or isinstance(cell, str)
        and " ".join(cell.split()).casefold() in MISSING_TOKENS
    )


def cell_number(cell):
    """Parse nominal values, scientific notation and common PDF footnotes."""
    value = unescape(str(cell)).strip().replace("−", "-").replace("–", "-")
    value = re.sub(r"^([+-])\s+(?=\d|\.\d)", r"\1", value)
    value = re.sub(r"\[\d+\]|[†‡*]+$", "", value).strip()
    value = re.sub(r"(?<=\d)\s*\(\d+\)$", "", value)
    value = re.split(r"\s*(?:±|\+/-)\s*", value)[0]
    value = re.sub(
        r"\s*[×·]\s*10\s*\^?\s*([+\-⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹\d]+)$",
        lambda match: "e" + match[1].translate(_SUPERSCRIPTS),
        value,
    )
    if not re.fullmatch(_NUM.replace("−–", ""), value):
        return None
    try:
        result = float(value.replace(",", "."))
    except ValueError:
        return None
    return result if math.isfinite(result) else None


class _HTMLTables(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.table, self.row, self.cell = [], None, None, None
        self.spans, self.column, self.cell_span = {}, 0, (1, 1)
        self.caption = None
        self.context = []

    def handle_starttag(self, tag, attrs):
        if tag == "table" and self.table is None:
            context = " ".join("".join(self.context).split())
            self.table, self.spans = ([[context]] if context else []), {}
            self.context = []
        elif tag == "caption" and self.table is not None:
            self.caption = []
        elif tag == "tr" and self.table is not None:
            self.row, self.column = [], 0
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
            attrs = dict(attrs)
            try:
                self.cell_span = (
                    max(1, min(30, int(attrs.get("colspan", 1)))),
                    max(1, min(100, int(attrs.get("rowspan", 1)))),
                )
            except ValueError:
                self.cell_span = (1, 1)
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def _carry_spans(self):
        while self.column in self.spans:
            text, remaining = self.spans[self.column]
            self.row.append(text)
            if remaining == 1:
                del self.spans[self.column]
            else:
                self.spans[self.column] = (text, remaining - 1)
            self.column += 1

    def handle_endtag(self, tag):
        if self.table is None and tag in ("p", "div", "h1", "h2", "h3", "h4", "br"):
            self.context.append("\n")
        if tag == "caption" and self.caption is not None:
            self.table.append([" ".join("".join(self.caption).split())])
            self.caption = None
        elif tag in ("td", "th") and self.cell is not None:
            self._carry_spans()
            text = " ".join("".join(self.cell).split())
            colspan, rowspan = self.cell_span
            for _ in range(colspan):
                self.row.append(text)
                if rowspan > 1:
                    self.spans[self.column] = (text, rowspan - 1)
                self.column += 1
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self._carry_spans()
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table, self.row, self.cell = None, None, None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        elif self.caption is not None:
            self.caption.append(data)
        elif self.table is None:
            self.context.append(data)


def _split_line(line):
    line = line.strip(" \r\n")
    if not line:
        return []
    if "|" in line:
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        return [cell.strip() for cell in line.split("|")]
    if "\t" in line:
        return [cell.strip() for cell in line.split("\t")]
    if ";" in line:
        return next(csv.reader([line], delimiter=";"))
    # Keep standalone signs until the expected table width is known.
    numeric_line = re.sub(r"(?<=\d)\s+\.\s+(?=\d)", ".", line)
    words = re.findall(
        r"\bnot\s+(?:available|measured|reported)\b|\S+", numeric_line, re.I
    )
    if len(words) > 1 and all(
        cell_number(word) is not None or is_missing_cell(word) for word in words
    ):
        return words
    if "," in line and not re.fullmatch(_NUM, line):
        return next(csv.reader([line]))
    if len(words) > 1 and (
        sum(cell_number(word) is not None for word in words) >= 2
        or cell_number(words[0]) is not None
        and all(
            cell_number(word) is not None or is_missing_cell(word) for word in words[1:]
        )
    ):
        return words
    if 1 < len(words) <= 30 and all(word in _CANONICAL for word in words):
        return words
    # Compact paper headers often use one space, including within each unit.
    symbol = r"(?:[TtPp]|u|U|σ|[xyw](?:_?[12lI₁₂])?(?:[′″'\"]|[αβ])?|[Hh]\s*(?:_?m\s*)?(?:\^?\s*[Eeᴱ])|(?:(?:ln|log10)\s*)?(?:γ|gamma))[12₁₂]?(?:∞|\^?inf)?"
    starts = list(re.finditer(rf"(?<!\S){symbol}(?=\s|[/([]|$)", line))
    if len(starts) >= 2 and not line[: starts[0].start()].strip():
        cells = [
            line[start.start() : end].strip()
            for start, end in zip(
                starts, [item.start() for item in starts[1:]] + [len(line)]
            )
        ]
        if all(
            _field(cell)[0] != "ignore"
            or _field(cell)[1] in ("calculated column", "uncertainty column")
            for cell in cells
        ):
            return cells
    if len(starts) == 1 and starts[0].start() == 0 and _field(line)[0] != "ignore":
        return [line]
    return re.split(r"\s{2,}|\s+(?=[+−–-]?(?:\d|\.))", line)


def _is_data(cells, headers=()):
    numbers = sum(cell_number(cell) is not None for cell in cells)
    for header in reversed(headers):
        if len(header) == len(cells) and all(label in _CANONICAL for label in header):
            roles = [_CANONICAL[label] for label in header]
            numeric_roles = {
                "temperature",
                "pressure",
                "x1",
                "y1",
                "x1_alpha",
                "x1_beta",
                "enthalpy",
                "gamma1_inf",
                "gamma2_inf",
            }
            if any(
                role in numeric_roles
                and (cell_number(cell) is not None or is_missing_cell(cell))
                for role, cell in zip(roles, cells)
            ):
                return True
    numeric_or_missing = (numbers > 0 or bool(headers)) and all(
        cell_number(cell) is not None
        or is_missing_cell(cell)
        or str(cell).strip().upper() in KINDS
        for cell in cells
    )
    return numbers >= min(2, len(cells)) and numbers > 0 or numeric_or_missing


def _temperature_unit_label(unit):
    return unit if unit == "K" else "°" + unit


def _temperature_header_conditions(row):
    text = " ".join(row)
    matches = list(re.finditer(rf"{_NUM}\s*(?:K|°\s*[CF])\b", text, re.I))
    if 2 <= len(matches) <= 29:
        return [_series_header_hint(match.group()) for match in matches]
    if row and _field(row[0])[0] == "temperature":
        unit = _units(text).get("temperature_unit")
        values = re.findall(rf"(?<![A-Za-z0-9_]){_NUM}(?![A-Za-z0-9_])", text)
        if unit and 2 <= len(values) <= 29:
            return [
                _series_header_hint(f"{cell} {_temperature_unit_label(unit)}")
                for cell in values
            ]
    return []


def _repeated_header_layout(headers):
    """Expand a shared measurement heading followed by explicit conditions."""
    explicit = next(
        (
            row
            for row in headers
            if 3 <= len(row) <= 30
            and _field(row[0])[0] == "x1"
            and all(_field(cell)[0] == "enthalpy" for cell in row[1:])
        ),
        None,
    )
    heading = next(
        (
            row
            for row in headers
            if sum(_field(cell)[0] == "x1" for cell in row) == 1
            and len({cell for cell in row if _field(cell)[0] == "enthalpy"}) == 1
        ),
        None,
    )
    if heading is None:
        compositions = {
            cell for row in headers for cell in row if _field(cell)[0] == "x1"
        }
        measurements = {
            cell for row in headers for cell in row if _field(cell)[0] == "enthalpy"
        }
        if len(compositions) != 1 or len(measurements) != 1:
            return explicit
        heading = [compositions.pop(), measurements.pop()]
    for row in headers:
        if row is heading:
            continue
        conditions = _temperature_header_conditions(row)
        if not conditions:
            continue
        composition = next(cell for cell in heading if _field(cell)[0] == "x1")
        measurement = next(cell for cell in heading if _field(cell)[0] == "enthalpy")
        labels = [composition] + [
            f"{measurement} at {item['temperature']:g} {_temperature_unit_label(item['temperature_unit'])}"
            for item in conditions
        ]
        return labels
    return explicit


def _header_width(headers):
    expanded = _repeated_header_layout(headers)
    return (
        len(expanded)
        if expanded
        else next(
            (
                len(header)
                for header in reversed(headers)
                if sum(_field(cell)[0] != "ignore" for cell in header) >= 2
            ),
            None,
        )
    )


def _align_pdf_signs(cells, width):
    """Join detached negatives only when column count determines every sign."""
    candidates = {
        index
        for index, cell in enumerate(cells[:-1])
        if cell in ("-", "−", "–")
        and cell_number(cells[index + 1]) is not None
        and not cells[index + 1].startswith(("-", "−", "–", "+"))
    }
    extra = len(cells) - width
    if extra <= 0 or not candidates:
        return cells, None
    if extra > len(candidates):
        return cells, None
    if extra < len(candidates):
        return (
            cells,
            "Standalone dashes could be missing cells or detached negative signs; confirm the column alignment.",
        )
    aligned = []
    for index, cell in enumerate(cells):
        if index in candidates:
            continue
        aligned.append("-" + cell if index - 1 in candidates else cell)
    return aligned, None


def _table_blocks(lines, *, expected_width=None, line_numbers=None):
    """Keep rectangular numeric blocks and nearby headings, not surrounding prose."""
    blocks, pending, active = [], [], None
    extracted = [_split_line(line) if isinstance(line, str) else line for line in lines]
    flattened_column = all(
        len(cells) == 1
        and (cell_number(cells[0]) is not None or is_missing_cell(cells[0]))
        for cells in extracted
        if cells
    ) and any(cells and cell_number(cells[0]) is not None for cells in extracted)
    clean_widths = {
        len(cells)
        for cells in extracted
        if len(cells) > 1 and all(cell_number(cell) is not None for cell in cells)
    }
    fallback_width = next(iter(clean_widths)) if len(clean_widths) == 1 else None
    for index, cells in enumerate(extracted):
        line_number = line_numbers[index] if line_numbers else index + 1
        if cells and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells):
            continue
        alignment_issue = None
        if (
            (len(lines) > 1 or pending)
            and isinstance(lines[index], str)
            and not any(delimiter in lines[index] for delimiter in ("\t", "|", ";"))
            and (
                "," not in lines[index]
                or all(
                    cell_number(word) is not None or is_missing_cell(word)
                    for word in lines[index].split()
                )
            )
        ):
            headers = active["headers"] if active else pending
            width = (
                expected_width
                or _header_width(headers)
                or (len(active["rows"][0]) if active else fallback_width)
            )
            if width:
                cells, alignment_issue = _align_pdf_signs(cells, width)
                if alignment_issue:
                    unassigned = list(cells)
                    cells = cells[:width] + [""] * max(0, width - len(cells))
        if (
            cells
            and _field(cells[0])[0] == "temperature"
            and _temperature_header_conditions(cells)
        ):
            if active:
                blocks.append(active)
                active = None
            pending.append(cells)
            continue
        if cells and (
            _is_data(cells, active["headers"] if active else pending)
            or len(cells) == 1
            and is_missing_cell(cells[0])
            and (flattened_column or active and len(active["rows"][0]) == 1)
        ):
            if (
                active is None
                or not active.get("column_headers")
                and len(cells) != len(active["rows"][0])
            ):
                inherited = active["headers"] if active is not None else []
                if active:
                    blocks.append(active)
                active = {
                    "headers": pending[-8:] or inherited,
                    "rows": [],
                    "line_numbers": [],
                }
                expanded = _repeated_header_layout(active["headers"])
                if expanded:
                    active["column_headers"] = expanded
                    active["ambiguous_rows"] = []
                pending = []
            if alignment_issue:
                active.setdefault("ambiguous_rows", []).append(
                    {
                        "row": len(active["rows"]),
                        "line": line_number,
                        "values": unassigned,
                        "reason": alignment_issue,
                    }
                )
            if active.get("column_headers"):
                width = len(active["column_headers"])
                if len(cells) != width:
                    active["ambiguous_rows"].append(
                        {
                            "row": len(active["rows"]),
                            "line": line_number,
                            "values": cells[1:],
                            "reason": f"Extracted {len(cells) - 1} HE values for {width - 1} temperature columns; their column positions are ambiguous.",
                        }
                    )
                cells = cells[:width] + [""] * max(0, width - len(cells))
            active["rows"].append(cells)
            active["line_numbers"].append(line_number)
        else:
            if active:
                blocks.append(active)
                active = None
            if cells and all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in cells):
                continue
            if any(str(cell).strip() for cell in cells):
                pending.append(cells)
    if active:
        blocks.append(active)
    # A single observed point is valid when accompanied by a recognizable header.
    return [
        block
        for block in blocks
        if len(block["rows"]) >= 2
        or len(block["rows"][0]) > 1
        or any(
            _field(label)[0] != "ignore"
            for header in block["headers"]
            for label in header
        )
    ]


def _clean_label(label):
    return (
        unicodedata.normalize("NFKC", unescape(label))
        .lower()
        .replace("\u00ad", "")
        .replace("\u200b", "")
        .replace("◦", "°")
        .replace("vapour", "vapor")
        .replace("γ", "gamma")
        .replace("∞", "inf")
        .replace("α", "alpha")
        .replace("β", "beta")
        .replace("₁", "1")
        .replace("₂", "2")
    )


def _assigned_conditions(text):
    """Extract constant conditions, including units written before the value."""
    annotations = []
    for field, label, units in (
        ("pressure", "pressure|p", "mmhg|torr|kpa|mpa|atm|bar|psi|pa"),
        ("temperature", "temperature|temp|t", "kelvin|celsius|fahrenheit|°?\\s*[ckf]"),
    ):
        pattern = (
            rf"\b(?:{label})\s*"
            rf"(?:(?:[_/,(\[:]\s*)*({units})(?![a-z])\s*[)\]]*\s*)?"
            rf"(?:=|:|≈|~|\bat\b)\s*({_NUM})\s*(?:[([]\s*({units})\s*[)\]]|({units}))?(?![a-z])"
        )
        for match in re.finditer(pattern, text, re.I):
            before, number, bracketed, after = match.groups()
            declared = {
                unit.replace(" ", "").lower()
                for unit in (before, bracketed, after)
                if unit
            }
            for unit in declared or {None}:
                annotations.append((field, number, unit, match.span()))
    return annotations


def _field(label):
    """Map descriptive headers to roles; never relabel calculated data as measured."""
    label = label.strip()
    if label in _CANONICAL:
        return _CANONICAL[label], "explicit"
    lower = _clean_label(label)
    compact = re.sub(r"[^a-z0-9]", "", lower)
    annotations = lower.replace("_", " ")
    if re.search(
        r"\b(calc(?:ulated)?|predicted|fit(?:ted)?|theoretical|model)\b|\bcal\b(?!\s*(?:/|mol\b))",
        annotations,
    ) or compact.endswith(
        ("calc", "calculated", "fit", "fitted", "predicted", "model")
    ):
        return "ignore", "calculated column"
    if re.search(
        r"\b(?:uncertaint(?:y|ies)|std\.?|standard deviation|error)\b|^(?:u|sigma|σ)\s*[(\[]|^δ\s*[tpxy]",
        annotations,
    ):
        return "ignore", "uncertainty column"
    conditions = _assigned_conditions(lower)
    for start, end in sorted({item[3] for item in conditions}, reverse=True):
        lower = lower[:start] + " " + lower[end:]
    compact = re.sub(r"[^a-z0-9]", "", lower)
    if (
        "enthalpy" in lower
        or re.search(r"h\s*(?:_?m\s*)?(?:\^\s*e|e\b)", lower)
        or compact.startswith("he")
    ):
        return "enthalpy", "header"
    if re.search(r"\b(temp(?:erature)?|t)\b", lower) or re.fullmatch(
        r"(?:temperature|temp|t)[ckf]", compact
    ):
        return "temperature", "header"
    if re.search(r"\b(pressure|p)\b", lower) or compact in (
        "pbar",
        "pkpa",
        "patm",
        "pmmhg",
    ):
        return "pressure", "header"
    if "gamma" in lower and ("inf" in lower or "infinite" in lower):
        return "gamma2_inf" if re.search(
            r"gamma\s*2", lower
        ) else "gamma1_inf", "header"
    if "alpha" in lower or compact in ("xa", "x1a", "x1alpha"):
        return "x1_alpha", "header"
    if "beta" in lower or compact in ("xb", "x1b", "x1beta"):
        return "x1_beta", "header"
    if re.match(r"x(?:_|\s*)[li](?:\s|[/([]|$)", lower):
        return "x1", "OCR composition header"
    if re.match(r"x[12]?\s*(?:″|′′|''|\^?\(?ii\)?)(?![a-z])", lower):
        return "x1_beta", "header"
    if re.match(r"x[12]?\s*(?:′|'|\^?\(?i\)?)(?![a-z])", lower):
        return "x1_alpha", "header"
    if (
        "vapor" in lower
        or compact in ("y", "y1", "y2", "y1mol", "y2mol")
        or re.match(r"y[12]?(?:\s|[/([]|$)", lower)
    ):
        return "y1", "header"
    if (
        "liquid" in lower
        or compact in ("x", "x1", "x2", "x1mol", "x2mol")
        or re.match(r"x[12]?(?:\s|[/([]|$)", lower)
    ):
        return "x1", "header"
    if re.search(r"\b(mole|mass|mol|weight)\s+fraction\b", lower):
        return "x1", "header"
    fields = {item[0] for item in conditions}
    return (
        (fields.pop(), "condition heading")
        if len(fields) == 1
        else ("ignore", "unrecognized header")
    )


def _descriptive_gamma_role(label, components):
    """Resolve named solute-in-solvent gamma-infinity headers to component order."""
    if not components or len(components) != 2:
        return None
    cleaned = _clean_label(label)
    if "gamma" not in cleaned or not ("inf" in cleaned or "infinite" in cleaned):
        return None
    compact = re.sub(
        r"(?:mathrm|text|operatorname|gamma|infinite|inf|log|ln)", "", cleaned
    )
    compact = re.sub(r"[^a-z0-9]", "", compact)
    names = [re.sub(r"[^a-z0-9]", "", _clean_label(name)) for name in components]
    positions = [compact.find(name) if name else -1 for name in names]
    if min(positions) < 0 or positions[0] == positions[1]:
        return None
    first = 0 if positions[0] < positions[1] else 1
    start = positions[first] + len(names[first])
    if "in" not in compact[start : positions[1 - first]]:
        return None
    return "gamma1_inf" if first == 0 else "gamma2_inf"


def _units(text):
    lower = _clean_label(text)
    result = {}
    temperature_label = r"\b(?:temperature|temp|t)[^a-z0-9]*"
    if re.search(rf"°\s*c|celsius|{temperature_label}c\b", lower):
        result["temperature_unit"] = "C"
    elif re.search(rf"°\s*f|fahrenheit|{temperature_label}f\b", lower):
        result["temperature_unit"] = "F"
    elif re.search(rf"°\s*k|kelvin|{temperature_label}k\b", lower):
        result["temperature_unit"] = "K"
    for unit in ("mmhg", "torr", "kpa", "mpa", "atm", "bar", "psi", "pa"):
        if re.search(rf"(?<![a-z]){unit}(?![a-z])", lower) or f"p_{unit}" in lower:
            result["pressure_unit"] = unit
            break
    energy = re.search(
        r"(?<![a-z])(kcal|cal|kj|j)\s*(?:/\s*mol|[·.]?\s*mol\s*\^?\s*[-−]\s*1)(?!\d)",
        lower,
    )
    if energy:
        result["enthalpy_unit"] = {
            "j": "J/mol",
            "kj": "kJ/mol",
            "cal": "cal/mol",
            "kcal": "kcal/mol",
        }[energy[1]]
    elif _field(text)[0] == "enthalpy" and re.search(r"(?<![a-z])j\s*mol\s+1\b", lower):
        result["enthalpy_unit"] = "J/mol"
    elif "he_kj_mol" in lower:
        result["enthalpy_unit"] = "kJ/mol"
    elif "he_j_mol" in lower:
        result["enthalpy_unit"] = "J/mol"
    percent = "%" in lower or "percent" in lower
    mass = bool(re.search(r"\b(mass|weight|wt)\b", lower))
    mole = bool(re.search(r"\b(mole|molar|mol)\b", lower))
    composition_percent = bool(
        re.search(
            r"\b(?:mole|molar|mol|mass|weight|wt|liquid|vapor|composition|solubility|[xyw][12]?)\b\s*(?:fraction\s*)?(?:percent|%)|(?:%|percent)\s*(?:by\s*)?(?:mole|mol|mass|weight)\b",
            lower,
        )
    )
    if percent and composition_percent or (mass or mole) and "fraction" in lower:
        result["composition_basis"] = ("mass" if mass else "mole") + (
            "_percent" if percent else "_fraction"
        )
    return result


def _context_defaults(text):
    result = {}
    text = _clean_label(text)
    assigned = _assigned_conditions(text)
    for field, pattern in (
        (
            "pressure",
            rf"(?:\bat\s+|\bp\s*=\s*)?({_NUM})\s*[([]?\s*(bar|kpa|mpa|pa|atm|mmhg|torr|psi)\b",
        ),
        (
            "temperature",
            rf"({_NUM})\s*[([]?\s*(°?\s*k|°\s*[cf]|kelvin|celsius|fahrenheit)\b",
        ),
    ):
        matches = re.findall(pattern, text, re.I) + [
            (number, unit) for found, number, unit, _ in assigned if found == field
        ]
        values = {
            (
                cell_number(number),
                unit.replace(" ", "").replace("°", "").lower() if unit else None,
            )
            for number, unit in matches
        }
        units = {
            unit for number, unit in values if number is not None and unit is not None
        }
        if any(unit is None for number, unit in values):
            result[field + "_unit"] = None
        if len(units) == 1:
            unit = units.pop()
            result[field + "_unit"] = (
                {"kelvin": "K", "celsius": "C", "fahrenheit": "F"}.get(
                    unit, unit.replace("°", "").upper()
                )
                if field == "temperature"
                else unit
            )
        if len(values) == 1:
            value, _ = values.pop()
            if value is not None:
                result[field] = value
    if (
        re.search(
            r"atmospheric pressure|normal pressure|standard atmosphere", text, re.I
        )
        and "pressure" not in result
    ):
        result.update(pressure=1.0, pressure_unit="atm")
    return result


def _reference_component(text):
    match = re.search(
        r"(?:%\s*(?:by\s*)?(?:mole|mass|weight)|(?:mole|mass)\s+(?:fraction|percent)(?:\s+of)?)\s+([A-Za-z][A-Za-z0-9,()\- ]*?)(?=\n|\t|\||$)",
        text,
        re.I,
    )
    if not match:
        return None
    name = match[1].strip()
    return (
        name
        if name.lower() not in ("liquid", "vapor", "vapour", "x", "y")
        and len(name) <= 80
        else None
    )


def _rank_temperature_pair(rows, mapping=None, excluded=()):
    numbers = np.array(
        [[cell_number(cell) for cell in row] for row in rows], dtype=object
    )
    candidates = []
    for temp in range(numbers.shape[1]):
        if (
            temp in excluded
            or mapping
            and mapping[temp] not in ("ignore", "temperature")
        ):
            continue
        if any(value is None for value in numbers[:, temp]):
            continue
        temperature = np.array(numbers[:, temp], dtype=float)
        if (
            not np.all(temperature > -273.15)
            or np.max(temperature) <= 1
            or np.ptp(temperature) == 0
        ):
            continue
        fractions = [
            index
            for index in range(numbers.shape[1])
            if index != temp
            and index not in excluded
            and (not mapping or mapping[index] in ("ignore", "x1", "y1"))
            and all(
                value is not None and 0 <= value <= 100 for value in numbers[:, index]
            )
        ]
        for first in fractions:
            for second in fractions:
                if first >= second:
                    continue
                a, b = (
                    np.array(numbers[:, first], dtype=float),
                    np.array(numbers[:, second], dtype=float),
                )
                if min(np.ptp(a), np.ptp(b)) <= 0:
                    continue
                if np.max(a) <= 1 and np.max(b) <= 1:
                    score = 8.0
                else:
                    score = 2.0
                    # Shared pure endpoints are particularly strong evidence.
                    score += 3 * int(any((a == 0) & (b == 0)))
                    score += 3 * int(any((a == 100) & (b == 100)))
                correlation = float(np.corrcoef(a, b)[0, 1])
                if not math.isfinite(correlation) or correlation < 0.3:
                    continue
                score += correlation
                typical = float(np.median(temperature))
                # Magnitude is weak evidence: pressure may exceed temperature.
                score += 0.75 * int(170 <= typical <= 1000)
                score += 0.5 * int(
                    all(
                        typical
                        > float(np.median(np.asarray(numbers[:, index], dtype=float)))
                        for index in range(numbers.shape[1])
                        if index != temp
                        and index not in excluded
                        and all(value is not None for value in numbers[:, index])
                    )
                )
                # Smooth, ordered composition curves are common in copied Txy.
                score += int(np.all(np.diff(a) >= 0) or np.all(np.diff(a) <= 0))
                score += int(np.all(np.diff(b) >= 0) or np.all(np.diff(b) <= 0))
                candidates.append((score, temp, first, second))
    return sorted(candidates, reverse=True)


def _table_dimension(value, name, maximum):
    if value is None:
        return None
    try:
        number = float(value)
        if (
            isinstance(value, bool)
            or not math.isfinite(number)
            or not number.is_integer()
            or not 1 <= number <= maximum
        ):
            raise ValueError
        return int(number)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{name} must be an integer between 1 and {maximum}."
        ) from error


def _flattened(block, options):
    single_line = len(block["rows"]) == 1 and len(block["rows"][0]) > 1
    values = (
        list(block["rows"][0]) if single_line else [row[0] for row in block["rows"]]
    )
    count = _table_dimension(options.get("column_count"), "column_count", 30)
    row_count = _table_dimension(options.get("row_count"), "row_count", 2000)

    def pending(issue):
        return {
            **block,
            "rows": [[" ".join(values)]],
            "flattened": True,
            "dimensions_needed": True,
            "row_count": row_count,
            "column_count": count,
            "value_count": len(values),
        }, [issue]

    if single_line and (count is None or row_count is None):
        return pending(
            "This paste is a table flattened onto one line. Enter its row and column counts and choose whether values run across rows or down columns."
        )
    if row_count is not None and count is not None:
        values, alignment_issue = _align_pdf_signs(values, row_count * count)
        if alignment_issue:
            return {
                **block,
                "rows": [[""] * count for _ in range(row_count)],
                "flattened": True,
                "row_count": row_count,
                "column_count": count,
                "value_count": len(values),
                "layout": options.get("layout", "rows"),
                "ambiguous_rows": [
                    {
                        "row": index,
                        "line": 1,
                        "values": values,
                        "reason": alignment_issue,
                    }
                    for index in range(row_count)
                ],
            }, []
    if count is None:
        context = " ".join(" ".join(row) for row in block["headers"])
        count = (
            1
            if options.get("kind") in ("UCST", "LCST")
            or re.search(r"\b(?:ucst|lcst)\b|critical solution", context, re.I)
            else 3
            if re.search(r"liquid|vapor|vapour|txy", context, re.I)
            else 2
            if re.search(r"enthalpy|gamma|γ", context, re.I)
            else 3
        )
    if row_count is not None and len(values) != row_count * count:
        return pending(
            f"The {len(values)} values do not fill {row_count} rows × {count} columns. Correct the dimensions or the source values."
        )
    if len(values) % count:
        return pending(
            f"{len(values)} one-cell lines do not divide into {count} columns. Set the column count or exclude incomplete rows."
        )
    row_layout = [values[i : i + count] for i in range(0, len(values), count)]
    col_layout = np.array(values, dtype=object).reshape(count, -1).T.tolist()
    layout = options.get("layout")
    if layout not in (None, "rows", "columns"):
        raise ValueError("layout must be rows or columns.")
    if layout is None:
        row_score = _rank_temperature_pair(row_layout)
        col_score = _rank_temperature_pair(col_layout)
        layout = (
            "columns"
            if col_score and (not row_score or col_score[0][0] > row_score[0][0])
            else "rows"
        )
    return {
        **block,
        "rows": col_layout if layout == "columns" else row_layout,
        "line_numbers": list(range(1, len(row_layout) + 1)),
        "flattened": True,
        "layout": layout,
        "row_count": len(row_layout),
        "column_count": count,
        "value_count": len(values),
    }, []


def _headers(block, width):
    if block.get("column_headers"):
        return list(block["column_headers"])
    combined = ["" for _ in range(width)]
    for row in block["headers"]:
        if all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in row):
            continue
        if len(row) == width:
            if not any(
                _field(cell)[0] != "ignore" or _units(cell) or _context_defaults(cell)
                for cell in row
            ):
                continue
            for index, cell in enumerate(row):
                combined[index] += " " + cell
        elif len(row) == 2 and width == 3:
            first_role = _field(row[0])[0]
            if first_role == "temperature" or _units(row[0]).get("temperature_unit"):
                combined[0] += " " + row[0]
                combined[1] += " " + row[1]
                combined[2] += " " + row[1]
            elif all(
                _field(cell)[0] in ("x1", "y1", "x1_alpha", "x1_beta") for cell in row
            ):
                combined[1] += " " + row[0]
                combined[2] += " " + row[1]
        elif len(row) == 1 and width == 1:
            combined[0] += " " + row[0]
    return [text.strip() for text in combined]


def _column_field_info(block, headers):
    """The specific measurement heading takes precedence over its group caption."""
    if block.get("column_headers"):
        return [_field(header) for header in headers]
    fields = []
    for index, header in enumerate(headers):
        field = _field(header)
        if field[1] in ("calculated column", "uncertainty column"):
            fields.append(field)
            continue
        if field[1] == "condition heading":
            field = ("ignore", "condition heading")
        for row in reversed(block["headers"]):
            if len(row) != len(headers):
                continue
            candidate = _field(row[index])
            if (
                candidate[0] != "ignore"
                and candidate[1] != "condition heading"
                or candidate[1] in ("calculated column", "uncertainty column")
            ):
                field = candidate
                break
        fields.append(field)
    return fields


def _repeated_vle_series(mapping, headers):
    """Recognize complete measured groups while leaving intervening junk ignored."""
    active = [(index, role) for index, role in enumerate(mapping) if role != "ignore"]
    axes = {role for _, role in active} & {"temperature", "pressure"}
    if len(axes) != 1 or any(role not in (*axes, "x1", "y1") for _, role in active):
        return None
    axis = axes.pop()
    groups, columns, roles = [], [], set()
    for index, role in active:
        if role in roles:
            groups.append((columns, roles))
            columns, roles = [], set()
        columns.append(index)
        roles.add(role)
    groups.append((columns, roles))
    if len(groups) < 2 or any(not {axis, "x1"} <= fields for _, fields in groups):
        return None
    condition = "pressure" if axis == "temperature" else "temperature"
    series = []
    for number, (columns, _) in enumerate(groups):
        hints = [_series_header_hint(headers[index]) for index in columns]
        spec = {"name": f"Series {number + 1}", "columns": columns}
        for key in (condition, condition + "_unit"):
            values = {hint[key] for hint in hints if key in hint}
            if len(values) == 1:
                spec[key] = values.pop()
        series.append(spec)
    return {
        "series": series,
        "mapping": mapping,
        "shared_columns": [],
        "layout": "TxyGroups" if axis == "temperature" else "PxyGroups",
    }


def _column_settings(label, role):
    """Only a column's own quantity can supply its unit or component index."""
    keys = {
        "temperature": {"temperature_unit"},
        "pressure": {"pressure_unit"},
        "enthalpy": {"enthalpy_unit"},
        "x1": {"composition_basis"},
        "y1": {"composition_basis"},
        "x1_alpha": {"composition_basis"},
        "x1_beta": {"composition_basis"},
    }.get(role, set())
    settings = {key: value for key, value in _units(label).items() if key in keys}
    if role in ("x1", "y1", "x1_alpha", "x1_beta"):
        indexed = list(
            re.finditer(
                r"(?<![a-z0-9])[xyw]\s*[_,(\[]?\s*([12])(?=\b|[′″'\"]|alpha|beta)",
                _clean_label(label),
            )
        )
        if indexed:
            settings["composition_component"] = int(indexed[-1][1])
    if role in ("gamma1_inf", "gamma2_inf"):
        if re.match(r"ln\s*[(]?\s*gamma", _clean_label(label)):
            settings["logarithm"] = "natural"
        elif re.match(r"log\s*(?:_?10|₁₀)\s*[(]?\s*gamma", _clean_label(label)):
            settings["logarithm"] = "decimal"
        elif re.match(r"log\s*[(]?\s*gamma", _clean_label(label)):
            settings["logarithm"] = "unspecified"
    return settings


def tabular_matrix(value):
    rows = value.get("rows") if isinstance(value, dict) else value
    return (
        rows
        if isinstance(rows, list)
        and rows
        and all(isinstance(row, list) for row in rows)
        else None
    )


def _quoted_delimited_rows(text):
    if not re.search(r'(?m)(?:^|[,\t;])[ \t]*"', text):
        return None
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",\t;")
    except csv.Error:
        return None
    reader = csv.reader(io.StringIO(text), dialect, strict=True)
    rows, line_numbers = [], []
    try:
        while True:
            line = reader.line_num + 1
            row = next(reader, None)
            if row is None:
                break
            rows.append([cell.strip() for cell in row])
            line_numbers.append(line)
    except csv.Error:
        return None
    return (rows, line_numbers) if any(len(row) > 1 for row in rows) else None


def _default_kind(value, kind):
    if not kind:
        return value
    from copy import deepcopy

    value = deepcopy(value)
    if isinstance(value, list):
        for row in value:
            if isinstance(row, dict) and not ("kind" in row or "type" in row):
                row["kind"] = kind
    elif isinstance(value, dict):
        if "datasets" in value:
            for dataset in value["datasets"]:
                if isinstance(dataset, dict):
                    dataset.setdefault("kind", kind)
        else:
            key = "rows" if "rows" in value else "observations"
            if key in value:
                value[key] = _default_kind(value[key], kind)
    return value


def interpret_paste(value, *, options=None, components=None):
    """Return an auditable table proposal, with issues instead of invented units."""
    options = options or {}
    allowed = {
        "table",
        "mapping",
        "kind",
        "temperature",
        "temperature_unit",
        "pressure",
        "pressure_unit",
        "enthalpy_unit",
        "composition_basis",
        "composition_component",
        "molecular_weights",
        "exclude_rows",
        "column_count",
        "row_count",
        "layout",
        "series",
        "shared_columns",
        "cell_edits",
        "infer_series",
    }
    if not isinstance(options, dict) or options.keys() - allowed:
        raise ValueError(
            "Unknown import options; use the column/units controls in the import preview."
        )
    if "infer_series" in options and not isinstance(options["infer_series"], bool):
        raise ValueError("infer_series must be true or false.")
    if not isinstance(value, str):
        matrix = tabular_matrix(value)
        if matrix is not None:
            headers = (
                value.get("headers", value.get("columns"))
                if isinstance(value, dict)
                else None
            )
            if headers is not None and (
                not isinstance(headers, list)
                or any(not isinstance(cell, str) for cell in headers)
            ):
                raise ValueError("Matrix headers must be an array of column names.")
            lines = ([headers] if headers else []) + matrix
            proposal = interpret_paste(
                "\n".join(
                    "\t".join("" if cell is None else str(cell) for cell in row)
                    for row in lines
                ),
                options=options,
                components=components,
            )
            proposal["original_text"] = json.dumps(value, ensure_ascii=False)
            return proposal
        return {
            "structured": True,
            "ready": True,
            "needs_review": False,
            "raw_observations": _default_kind(value, options.get("kind")),
            "notes": [],
            "issues": [],
            "excluded": [],
        }
    text = value.strip(" \r\n")
    if not text.strip() or len(text) > 1_000_000:
        raise ValueError("Paste a table of at most one million characters.")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("Close the fenced input block with ```.")
        text = "\n".join(lines[1:-1])
    if text.startswith(("[", "{")) and not re.match(r"^\[\d+\]\s+[A-Za-z]", text):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid observation JSON: {error}") from error
        return interpret_paste(decoded, options=options, components=components)
    if re.search(r"<table\b", text, re.I):
        parser = _HTMLTables()
        parser.feed(text)
        blocks = [block for table in parser.tables for block in _table_blocks(table)]
    else:
        quoted_csv = _quoted_delimited_rows(text)
        blocks = _table_blocks(
            quoted_csv[0] if quoted_csv else text.splitlines(),
            expected_width=len(options["mapping"])
            if isinstance(options.get("mapping"), list)
            else None,
            line_numbers=quoted_csv[1] if quoted_csv else None,
        )
    if not blocks:
        raise ValueError(
            "No numeric table found. Paste the table, including any headings or captions; columns may also be separated by spaces or newlines."
        )
    table_index = options.get(
        "table", max(range(len(blocks)), key=lambda index: len(blocks[index]["rows"]))
    )
    if (
        isinstance(table_index, bool)
        or not isinstance(table_index, int)
        or not 0 <= table_index < len(blocks)
    ):
        raise ValueError("Choose a table from this paste.")
    return _interpret_table(
        blocks[table_index],
        options=options,
        components=components,
        value=value,
        table_index=table_index,
        blocks=blocks,
    )


def _leading_series_condition_row(rows):
    return bool(
        rows
        and cell_number(rows[0][0]) is None
        and (
            _field(rows[0][0])[0] in ("x1", "temperature", "pressure")
            or not str(rows[0][0]).strip()
        )
        and all(cell_number(cell) is not None for cell in rows[0][1:])
    )


def _interpret_table(block, *, options, components, value, table_index, blocks):
    flattened_issues = []
    header_width = _header_width(block["headers"])
    if (len(block["rows"][0]) == 1 and len(block["rows"]) > 1) or (
        len(block["rows"]) == 1
        and len(block["rows"][0]) > 1
        and not block.get("layout")
        and not block.get("ambiguous_rows")
        and ("series" not in options or "row_count" in options)
        and (
            "row_count" in options
            or not any(
                _field(cell)[0] != "ignore"
                for header in block["headers"]
                for cell in header
            )
            or header_width
            and len(block["rows"][0]) >= 2 * header_width
        )
    ):
        block, flattened_issues = _flattened(block, options)
    rows = block["rows"]
    # Preview row numbers always address data rows. Remove a condition heading
    # before edits, exclusions and series projection use those same numbers.
    condition_heading = _leading_series_condition_row(rows) and (
        "series" in options or bool(str(rows[0][0]).strip())
    )
    if condition_heading:
        if len(rows) == 1:
            raise ValueError("No measurement rows remain after the condition heading.")
        headers = _headers(block, len(rows[0]))
        block = {
            **block,
            "column_headers": [
                str(cell).strip() or headers[index]
                for index, cell in enumerate(rows[0])
            ],
            "rows": rows[1:],
            "line_numbers": block["line_numbers"][1:],
            "ambiguous_rows": [
                {**item, "row": item["row"] - 1}
                for item in block.get("ambiguous_rows", [])
                if item["row"] > 0
            ],
        }
        rows = block["rows"]
    if len(rows) > 2000 or len(rows[0]) > 30:
        raise ValueError("Import at most 2000 rows and 30 columns at a time.")
    edits = options.get("cell_edits", [])
    if not isinstance(edits, list) or len(edits) > 60000:
        raise ValueError(
            "cell_edits must be an array of at most 60000 cell corrections."
        )
    if edits:
        rows = [list(row) for row in rows]
        for edit in edits:
            if (
                not isinstance(edit, dict)
                or set(edit) != {"row", "column", "value"}
                or any(
                    isinstance(edit[key], bool) or not isinstance(edit[key], int)
                    for key in ("row", "column")
                )
                or not 0 <= edit["row"] < len(rows)
                or not 0 <= edit["column"] < len(rows[edit["row"]])
                or not isinstance(edit["value"], str)
                or len(edit["value"]) > 1000
            ):
                raise ValueError(
                    "Each cell edit needs valid zero-based row/column indices and a text value of at most 1000 characters."
                )
            rows[edit["row"]][edit["column"]] = edit["value"]
        block = {**block, "rows": rows}
        corrected = {(edit["row"], edit["column"]) for edit in edits}
        block["ambiguous_rows"] = [
            item
            for item in block.get("ambiguous_rows", [])
            if not all(
                (item["row"], column) in corrected
                for column in range(len(rows[item["row"]]))
            )
        ]
    width = len(rows[0])
    headers = _headers(block, width)
    context = "\n".join("\t".join(row) for row in block["headers"])
    field_info = _column_field_info(block, headers)
    mapping = [field[0] for field in field_info]
    gamma_reassignments = []
    for index, role in enumerate(mapping):
        if role not in ("gamma1_inf", "gamma2_inf"):
            continue
        described = _descriptive_gamma_role(headers[index], components)
        if described and described != role:
            mapping[index] = described
            gamma_reassignments.append((headers[index], described))
    header_series = _repeated_vle_series(mapping, headers)
    notes, issues = [], list(flattened_issues)
    for header, role in gamma_reassignments:
        notes.append(
            f"{header} was assigned to {role} from the named solute and selected component order."
        )
    issues.extend(
        _alignment_issues(
            block.get("ambiguous_rows", []), options.get("exclude_rows", [])
        )
    )
    if condition_heading:
        notes.append(
            "The leading axis/condition row was treated as a heading; it remains in the original paste."
        )
    elif block.get("column_headers"):
        notes.append(
            "Repeated HE columns and their temperatures were recognized from the shared heading and condition row. Short rows retain their unassigned measurements for alignment review."
        )
    explicit_header = any(mapping[index] != "ignore" for index in range(width))
    canonical = (
        bool(block["headers"])
        and all(label in _CANONICAL for label in block["headers"][-1])
        and len(block["headers"][-1]) == width
    )
    suggested = _context_defaults(context)
    suggested.update(_units(context))
    column_settings = [
        _column_settings(header, role) for header, role in zip(headers, mapping)
    ]
    for key in (
        "temperature_unit",
        "pressure_unit",
        "enthalpy_unit",
        "composition_basis",
        "composition_component",
    ):
        values = {item[key] for item in column_settings if key in item}
        if len(values) == 1:
            suggested[key] = values.pop()
        elif len(values) > 1:
            notes.append(
                f"Columns use different {key.replace('_', ' ')} values; each labeled column retains its own definition unless overridden."
            )
    reference = _reference_component(context)
    if reference and components:
        matching = [
            i + 1
            for i, component in enumerate(components)
            if reference.casefold() == component.strip().casefold()
        ]
        suggested["composition_component"] = matching[0] if len(matching) == 1 else None
    else:
        suggested.setdefault("composition_component", 1)
    if header_series:
        kind = "VLE"
    elif re.search(r"vlle|hetero.?azeotrop", context, re.I) or {
        "x1_alpha",
        "x1_beta",
        "y1",
        "pressure",
    } <= set(mapping):
        kind = "VLLE"
    elif "enthalpy" in mapping or any(
        _field(cell)[0] == "enthalpy" for row in block["headers"] for cell in row
    ):
        kind = "HE"
    elif "gamma1_inf" in mapping or "gamma2_inf" in mapping:
        kind = "GAMMA_INF"
    elif (
        "x1_alpha" in mapping
        or "x1_beta" in mapping
        or re.search(r"\b(lle|mutual solubilit|tie.line)\b", context, re.I)
    ):
        kind = "LLE"
    elif re.search(r"\bucst\b|upper critical solution", context, re.I):
        kind = "UCST"
    elif re.search(r"\blcst\b|lower critical solution", context, re.I):
        kind = "LCST"
    elif re.search(r"azeotrop", context, re.I):
        kind = "AZEOTROPE"
    else:
        kind = "VLE"
    if "kind" in mapping:
        stated = {str(row[mapping.index("kind")]).strip().upper() for row in rows}
        if len(stated) == 1 and stated <= set(KINDS):
            kind = stated.pop()
    if kind == "LLE" and "x1_beta" in mapping:
        for index, header in enumerate(headers):
            if (
                re.match(r"x(?:_|\s*)i(?:\s|[/([]|$)", _clean_label(header))
                and mapping[index] == "x1"
            ):
                mapping[index] = "x1_alpha"
    excluded_columns = {
        index
        for index, header in enumerate(headers)
        if field_info[index][1] in ("calculated column", "uncertainty column")
    }
    candidates = _rank_temperature_pair(rows, mapping, excluded_columns)
    if (
        not header_series
        and kind in ("VLE", "AZEOTROPE")
        and candidates
        and (
            not explicit_header
            or "temperature" not in mapping
            or "x1" not in mapping
            or "y1" not in mapping
        )
    ):
        _, temp, first, second = candidates[0]
        mapping[temp] = "temperature"
        if mapping[first] == "y1" or mapping[second] == "x1":
            first, second = second, first
        fixed_compositions = mapping[first] == "x1" or mapping[second] == "y1"
        mapping[first], mapping[second] = "x1", "y1"
        numbers = np.array(
            [[cell_number(cell) for cell in row] for row in rows], dtype=object
        )
        # At pure endpoints, lower boiling T indicates the more volatile
        # reference component. This provides a liquid/vapor order suggestion.
        a, b = numbers[:, first].astype(float), numbers[:, second].astype(float)
        scale = 1 if max(max(a), max(b)) <= 1 else 100
        low = np.flatnonzero((a == 0) & (b == 0))
        high = np.flatnonzero((a == scale) & (b == scale))
        if len(low) and len(high):
            more_volatile = numbers[high[0], temp] < numbers[low[0], temp]
            if not fixed_compositions and bool(np.median(b - a) > 0) != more_volatile:
                mapping[first], mapping[second] = "y1", "x1"
            notes.append(
                "Liquid/vapor order is suggested by the pure-endpoint boiling-temperature trend; review the mapping, especially for azeotropes."
            )
        else:
            notes.append(
                "The two composition curves suggest Txy data. Liquid/vapor order is provisional because the pure endpoints are absent."
            )
        notes.append(
            "Column roles were inferred from numeric ranges, relative column magnitudes and co-varying composition curves; review OCR-damaged headings."
        )
        if "pressure" not in mapping:
            remaining = [
                index
                for index, role in enumerate(mapping)
                if role == "ignore" and index not in excluded_columns
            ]
            if len(remaining) == 1 and all(
                cell_number(row[remaining[0]]) is not None
                and cell_number(row[remaining[0]]) > 0
                for row in rows
            ):
                mapping[remaining[0]] = "pressure"
                if "pressure_unit" not in suggested and "pressure_unit" not in options:
                    suggested["pressure_unit"] = None
                notes.append(
                    "The remaining positive-valued column is suggested as pressure; its unit must be stated or confirmed."
                )
    elif not explicit_header:
        # Common two-column isothermal HE and infinite-dilution tables require
        # context or a user-selected kind; no third curve is invented.
        if width == 2 and kind == "HE":
            mapping = ["x1", "enthalpy"]
        elif width == 2 and kind == "GAMMA_INF":
            mapping = ["temperature", "gamma1_inf"]
        elif width == 3 and kind == "HE":
            fractions = [
                index
                for index in range(width)
                if all(
                    cell_number(row[index]) is not None
                    and 0 <= cell_number(row[index]) <= 1
                    for row in rows
                )
            ]
            temperatures = [
                index
                for index in range(width)
                if all(
                    cell_number(row[index]) is not None and cell_number(row[index]) > 1
                    for row in rows
                )
            ]
            if len(fractions) == len(temperatures) == 1:
                enthalpy = next(
                    index
                    for index in range(width)
                    if index not in (fractions[0], temperatures[0])
                )
                mapping[fractions[0]], mapping[temperatures[0]], mapping[enthalpy] = (
                    "x1",
                    "temperature",
                    "enthalpy",
                )
                notes.append(
                    "Calorimetry column roles were suggested from the excess-enthalpy caption and numeric ranges; confirm the units."
                )
        elif width == 3 and kind == "LLE":
            temperatures = [
                index
                for index in range(width)
                if all(
                    cell_number(row[index]) is not None and cell_number(row[index]) > 1
                    for row in rows
                )
            ]
            if len(temperatures) == 1:
                temp = temperatures[0]
                endpoints = [index for index in range(width) if index != temp]
                if all(
                    cell_number(row[index]) is not None
                    and 0 < cell_number(row[index]) < 1
                    for index in endpoints
                    for row in rows
                ):
                    mapping[temp] = "temperature"
                    mapping[endpoints[0]], mapping[endpoints[1]] = "x1_alpha", "x1_beta"
                    notes.append(
                        "LLE endpoint columns were suggested from the mutual-solubility caption and fractional ranges; confirm their component and basis."
                    )
        else:
            notes.append(
                "The column roles are ambiguous. Choose them below; the original cells are preserved."
            )
    else:
        # In descriptive headers a shared group heading can contain '%' but no
        # temperature label. Numeric evidence fills only the still-unmapped T.
        if "temperature" not in mapping and candidates:
            temp = candidates[0][1]
            if mapping[temp] == "ignore":
                mapping[temp] = "temperature"
                notes.append("Temperature column suggested from numeric ranges.")
    settings = {
        "kind": kind,
        "temperature_unit": None,
        "pressure_unit": "bar",
        "enthalpy_unit": None,
        "composition_basis": None,
        "composition_component": suggested.get("composition_component", 1),
        **suggested,
    }
    if "temperature" in mapping and not settings["temperature_unit"]:
        values = [cell_number(row[mapping.index("temperature")]) for row in rows]
        known = [value for value in values if value is not None]
        settings["temperature_unit"] = "C" if known and max(known) < 170 else "K"
        notes.append(
            f"Temperature units are not stated. {settings['temperature_unit']} is a magnitude-based suggestion, not a known unit."
        )
    compositions = [
        i
        for i, role in enumerate(mapping)
        if role in ("x1", "y1", "x1_alpha", "x1_beta")
    ]
    if compositions and not settings["composition_basis"]:
        maximum = max(
            (cell_number(row[i]) or 0 for row in rows for i in compositions), default=0
        )
        settings["composition_basis"] = (
            "mole_fraction" if canonical or maximum <= 1 else "mole_percent"
        )
        if not canonical:
            notes.append(
                "Composition scale was suggested from the values; mole versus mass basis cannot be determined from unlabeled numbers alone."
            )
    settings.update(
        {
            key: value
            for key, value in options.items()
            if key
            not in (
                "table",
                "mapping",
                "exclude_rows",
                "column_count",
                "row_count",
                "layout",
                "series",
                "shared_columns",
                "cell_edits",
                "infer_series",
            )
        }
    )
    if "mapping" in options and not block.get("dimensions_needed"):
        mapping = options["mapping"]
        if (
            not isinstance(mapping, list)
            or len(mapping) != width
            or any(role not in FIELD_LABELS for role in mapping)
        ):
            raise ValueError("Provide one supported column role per extracted column.")
    column_settings = [
        _column_settings(header, role) for header, role in zip(headers, mapping)
    ]
    detected_series = _repeated_vle_series(mapping, headers)
    if detected_series and settings["kind"] == "VLE":
        condition = (
            "pressure" if detected_series["layout"] == "TxyGroups" else "temperature"
        )
        unit_key = condition + "_unit"
        for spec in detected_series["series"]:
            if spec.get(unit_key) is None and options.get(unit_key):
                spec[unit_key] = options[unit_key]
        if unit_key not in options and any(
            condition in spec and spec.get(unit_key) is None
            for spec in detected_series["series"]
        ):
            settings[unit_key] = None
    exclude_rows = options.get("exclude_rows", [])
    if not isinstance(exclude_rows, list) or any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(rows)
        for index in exclude_rows
    ):
        raise ValueError("exclude_rows must contain valid zero-based row numbers.")
    if "series" in options:
        return _interpret_repeated(
            block,
            options=options,
            components=components,
            value=value,
            table_index=table_index,
            blocks=blocks,
            headers=headers,
            mapping=mapping,
            settings=settings,
            notes=notes,
            suggestion=header_series,
        )
    if (
        detected_series
        and settings["kind"] == "VLE"
        and options.get("infer_series", True)
    ):
        return _interpret_repeated(
            block,
            options={
                **options,
                "series": detected_series["series"],
                "shared_columns": [],
            },
            components=components,
            value=value,
            table_index=table_index,
            blocks=blocks,
            headers=headers,
            mapping=mapping,
            settings=settings,
            notes=notes,
            suggestion=header_series,
            inferred=True,
        )
    compositions = [
        i
        for i, role in enumerate(mapping)
        if role in ("x1", "y1", "x1_alpha", "x1_beta")
    ]
    if compositions and not settings["composition_basis"]:
        maximum = max(
            (cell_number(row[i]) or 0 for row in rows for i in compositions), default=0
        )
        settings["composition_basis"] = (
            "mole_fraction" if maximum <= 1 else "mole_percent"
        )
    if len([role for role in mapping if role != "ignore"]) != len(
        set(role for role in mapping if role != "ignore")
    ):
        issues.append(
            "Each column role can be used only once in single-series mode. Choose Repeated series for repeated measurements, or ignore calculated/reference columns."
        )
    if all(role == "ignore" for role in mapping):
        issues.append("Choose the column meanings.")
    required = {
        "VLE": {"temperature", "pressure", "x1"},
        "LLE": {"temperature"},
        "VLLE": {"temperature", "pressure"},
        "HE": {"temperature", "x1", "enthalpy"},
        "GAMMA_INF": {"temperature"},
        "AZEOTROPE": {"temperature", "pressure", "x1"},
        "UCST": {"temperature"},
        "LCST": {"temperature"},
    }
    if settings["kind"] not in required:
        raise ValueError("Choose a supported observation kind.")
    row_kinds = (
        {str(row[mapping.index("kind")]).strip().upper() for row in rows}
        if "kind" in mapping
        else {settings["kind"]}
    )
    required_fields = (
        set.intersection(*(required[item] for item in row_kinds))
        if row_kinds <= required.keys()
        else required[settings["kind"]]
    )
    missing = required_fields - set(mapping)
    for field in ("temperature", "pressure"):
        if (
            field in missing
            and settings.get(field) is not None
            and str(settings[field]).strip()
        ):
            missing.remove(field)
    if missing:
        issues.append("Supply or map: " + ", ".join(sorted(missing)) + ".")
    if "LLE" in row_kinds and not {"x1_alpha", "x1_beta"} & set(mapping):
        issues.append("Map at least one LLE liquid endpoint: x1_alpha or x1_beta.")
    if settings["kind"] == "GAMMA_INF" and not {"gamma1_inf", "gamma2_inf"} & set(
        mapping
    ):
        issues.append("Map at least one infinite-dilution activity coefficient.")
    if compositions and any(
        {**settings, **column_settings[index], **options}.get("composition_component")
        not in (1, 2)
        for index in compositions
    ):
        issues.append(
            f"Choose which component the compositions describe{': ' + reference if reference else ''}, or update the mixture's components."
        )
    if (
        any(
            {**settings, **column_settings[index], **options}.get("temperature_unit")
            not in ("C", "K", "F", "°C", "°F", "°K")
            for index, role in enumerate(mapping)
            if role == "temperature"
        )
        or "temperature" not in mapping
        and settings.get("temperature_unit") not in ("C", "K", "F", "°C", "°F", "°K")
    ):
        issues.append("Choose the temperature unit.")
    if "enthalpy" in mapping and any(
        {**settings, **column_settings[index], **options}.get("enthalpy_unit")
        not in ENTHALPY_UNITS
        for index, role in enumerate(mapping)
        if role == "enthalpy"
    ):
        issues.append("Choose the excess-enthalpy unit.")
    if (
        any(
            not {**settings, **column_settings[index], **options}.get("pressure_unit")
            for index, role in enumerate(mapping)
            if role == "pressure"
        )
        or settings["kind"] in ("VLE", "AZEOTROPE", "VLLE", "LLE")
        and settings.get("pressure") is not None
        and not settings.get("pressure_unit")
    ):
        issues.append(
            "Choose the pressure unit; the magnitude alone does not identify it."
        )
    if any(item.get("logarithm") == "unspecified" for item in column_settings):
        issues.append(
            "The activity-coefficient header says log without a base; specify ln or log10 in the heading."
        )
    if any(item.get("logarithm") in ("natural", "decimal") for item in column_settings):
        notes.append(
            "Logarithmic infinite-dilution activity coefficients are converted to dimensionless gamma values using the stated base."
        )
    composition_bases = [
        {**settings, **column_settings[index], **options}.get("composition_basis")
        for index in compositions
    ]
    if any(
        local not in ("mole_fraction", "mole_percent", "mass_fraction", "mass_percent")
        for local in composition_bases
    ):
        issues.append("Choose the composition basis.")
    molecular_weights = settings.get("molecular_weights")
    if any(local in ("mass_fraction", "mass_percent") for local in composition_bases):
        if (
            not isinstance(molecular_weights, list)
            or len(molecular_weights) != 2
            or any(
                cell_number(weight) is None or cell_number(weight) <= 0
                for weight in molecular_weights
            )
        ):
            issues.append(
                "Mass compositions need positive molecular weights for components 1 and 2."
            )
    raw_observations = []
    excluded = [
        {"row": index, "reason": "Excluded in the import preview."}
        for index in exclude_rows
    ]
    if any(
        "±" in cell or "+/-" in cell or re.search(r"\d\(\d+\)", cell)
        for row in rows
        for cell in row
    ):
        notes.append(
            "Nominal values were extracted from uncertainty annotations. Original annotations are preserved; set appropriate residual scales/weights separately."
        )
    if not issues:
        for index, cells in enumerate(rows):
            if index in exclude_rows:
                continue
            observation = {"kind": settings["kind"]}
            try:
                for column, role in enumerate(mapping):
                    local_settings = {**settings, **column_settings[column], **options}
                    cell = str(cells[column]).strip()
                    if role == "ignore" or not cell:
                        continue
                    if role in (
                        "kind",
                        "source",
                        "group",
                        "id",
                        "pin",
                        "sigma",
                        "validation_only",
                    ):
                        observation[role] = cell
                        continue
                    if is_missing_cell(cell):
                        continue
                    number = cell_number(cell)
                    if number is None:
                        raise ValueError(f"Column {column + 1} is not a numeric value.")
                    if role == "temperature":
                        observation["T_K"] = temperature_to_kelvin(
                            number, local_settings["temperature_unit"]
                        )
                    elif role == "pressure":
                        observation["P_bar"] = pressure_to_bar(
                            number, local_settings["pressure_unit"], strict=True
                        )
                    elif role == "enthalpy":
                        observation["HE_J_mol"] = (
                            number * ENTHALPY_UNITS[local_settings["enthalpy_unit"]]
                        )
                    elif role in ("x1", "y1", "x1_alpha", "x1_beta"):
                        local_basis = local_settings["composition_basis"]
                        fraction = number / (
                            100 if local_basis.endswith("_percent") else 1
                        )
                        if not 0 <= fraction <= 1:
                            raise ValueError(
                                "Composition lies outside the chosen fraction/percentage scale."
                            )
                        if local_settings.get("composition_component", 1) == 2:
                            fraction = 1 - fraction
                        if local_basis.startswith("mass"):
                            mw1, mw2 = [float(weight) for weight in molecular_weights]
                            fraction = (fraction / mw1) / (
                                fraction / mw1 + (1 - fraction) / mw2
                            )
                        observation[role] = fraction
                    elif role in ("gamma1_inf", "gamma2_inf") and local_settings.get(
                        "logarithm"
                    ):
                        observation[role] = (
                            math.exp(number)
                            if local_settings["logarithm"] == "natural"
                            else 10**number
                        )
                    else:
                        observation[role] = number
                if "T_K" not in observation and settings.get("temperature") is not None:
                    observation["T_K"] = temperature_to_kelvin(
                        float(settings["temperature"]), settings["temperature_unit"]
                    )
                if (
                    "P_bar" not in observation
                    and settings.get("pressure") is not None
                    and observation["kind"].upper()
                    in ("VLE", "AZEOTROPE", "VLLE", "LLE")
                ):
                    observation["P_bar"] = pressure_to_bar(
                        float(settings["pressure"]),
                        settings["pressure_unit"],
                        strict=True,
                    )
                if observation["kind"].upper() in (
                    "VLE",
                    "AZEOTROPE",
                ) and observation.get("x1") in (0, 1):
                    if "y1" in observation and observation["y1"] != observation["x1"]:
                        raise ValueError(
                            "A pure-liquid endpoint must have the same pure vapor composition."
                        )
                    excluded.append(
                        {
                            "row": index,
                            "reason": "Pure-component endpoint: no binary interaction information.",
                        }
                    )
                    continue
                raw_observations.append(observation)
            except (ValueError, TypeError, ZeroDivisionError, OverflowError) as error:
                issues.append(f"Row {index + 1}: {error}")
    if excluded:
        notes.append(
            f"{len(excluded)} rows are excluded from binary regression and retained in this preview."
        )
    if not issues and not raw_observations:
        issues.append(
            "No binary observations remain after endpoint/exclusion handling."
        )
    return {
        "structured": False,
        "ready": not issues,
        "needs_review": not canonical or bool(issues or excluded or len(blocks) > 1),
        "columns": [
            {
                "index": index,
                "label": headers[index] or f"Column {index + 1}",
                "role": mapping[index],
                "settings": column_settings[index],
                "reason": field_info[index][1],
            }
            for index in range(width)
        ],
        "raw_rows": rows,
        "ambiguous_rows": block.get("ambiguous_rows", []),
        "line_numbers": block["line_numbers"],
        "settings": settings,
        "table": table_index,
        "tables": [
            {"index": i, "rows": len(item["rows"]), "columns": len(item["rows"][0])}
            for i, item in enumerate(blocks)
        ],
        "flattened": block.get("flattened", False),
        "dimensions_needed": block.get("dimensions_needed", False),
        "row_count": block.get("row_count"),
        "column_count": block.get("column_count"),
        "value_count": block.get("value_count"),
        "layout": block.get("layout", "rows"),
        "reference_component": reference,
        "notes": notes,
        "issues": issues,
        "excluded": excluded,
        "raw_observations": raw_observations if not issues else None,
        "original_text": value,
        "cell_edits": edits,
        "series_available": len(rows[0]) > 2,
        "series_header_hints": _series_header_hints(headers, rows),
        "series_suggestion": header_series,
    }


def _series_header_hint(label):
    hint = {**_context_defaults(label), **_units(label)}
    if cell_number(label) is not None:
        hint["condition_value"] = cell_number(label)
    return hint


def _alignment_issues(ambiguous_rows, excluded):
    if not isinstance(excluded, list):
        raise ValueError("exclude_rows must contain valid zero-based row numbers.")
    return [
        f"Row {item['row'] + 1}: {item['reason']} Preserve empty column positions or exclude this row before import."
        for item in ambiguous_rows
        if item["row"] not in excluded
    ]


def _series_header_hints(headers, rows):
    labels = headers
    if _leading_series_condition_row(rows):
        labels = [
            str(rows[0][index]).strip() or headers[index]
            for index in range(len(headers))
        ]
    return [_series_header_hint(label) for label in labels]


def _interpret_repeated(
    block,
    *,
    options,
    components,
    value,
    table_index,
    blocks,
    headers,
    mapping,
    settings,
    notes,
    suggestion=None,
    inferred=False,
):
    """Project explicit series into the existing single-series conversion path."""
    series = options["series"]
    if not isinstance(series, list) or not 1 <= len(series) <= 30:
        raise ValueError(
            "Define 1–30 repeated series, each with its own columns and conditions."
        )
    width = len(mapping)

    def column_indices(values, label):
        if (
            not isinstance(values, list)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < width
                for index in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError(
                f"{label} must contain unique zero-based column indices from this table."
            )
        return values

    shared = column_indices(options.get("shared_columns", []), "shared_columns")
    shared_fields = [mapping[index] for index in shared if mapping[index] != "ignore"]
    if len(shared_fields) != len(set(shared_fields)):
        raise ValueError("Shared columns must have distinct meanings.")
    conditions = {
        "kind",
        "temperature",
        "temperature_unit",
        "pressure",
        "pressure_unit",
        "enthalpy_unit",
        "composition_basis",
        "composition_component",
        "molecular_weights",
    }
    permitted = {"name", "columns", "weight", "sigma", "validation_only"} | conditions
    source_rows = block["rows"]
    line_numbers = block["line_numbers"]
    ambiguous_rows = block.get("ambiguous_rows", [])
    notes = list(notes)
    raw, issues, reports, lineage, excluded = [], [], [], [], []
    issues.extend(_alignment_issues(ambiguous_rows, options.get("exclude_rows", [])))
    used = set()
    required = {
        "HE": {"x1", "enthalpy"},
        "VLE": {"x1"},
        "LLE": set(),
        "VLLE": set(),
        "AZEOTROPE": {"x1"},
        "GAMMA_INF": set(),
        "UCST": set(),
        "LCST": set(),
    }
    for number, spec in enumerate(series):
        if not isinstance(spec, dict) or spec.keys() - permitted:
            raise ValueError(
                "Series accept name, columns, conditions, weight, sigma and validation_only."
            )
        columns = column_indices(spec.get("columns"), f"Series {number + 1} columns")
        if not columns or set(columns) & set(shared) or set(columns) & used:
            raise ValueError(
                "Each series needs its own columns; shared columns must be assigned separately."
            )
        used.update(columns)
        selected = sorted(shared + columns)
        roles = [mapping[index] for index in selected]
        child_settings = {
            **settings,
            **{key: spec[key] for key in conditions if key in spec},
        }
        if (
            child_settings["kind"] in ("HE", "GAMMA_INF", "UCST", "LCST")
            and "pressure" not in roles
        ):
            child_settings.pop("pressure", None)
        name = str(spec.get("name") or f"Series {number + 1}")
        projected, row_indices, omitted = [], [], []
        for row_index, row in enumerate(source_rows):
            if row_index in options.get("exclude_rows", []):
                omitted.append(
                    {"row": row_index, "reason": "Excluded in the import preview."}
                )
                continue
            alignment = next(
                (item for item in ambiguous_rows if item["row"] == row_index), None
            )
            if alignment:
                omitted.append({"row": row_index, "reason": alignment["reason"]})
                continue
            cells = [str(row[index]).strip() for index in selected]
            absent = {
                role
                for role, cell in zip(roles, cells)
                if role != "ignore" and is_missing_cell(cell)
            }
            mandatory = (
                required.get(child_settings["kind"], set())
                | (
                    {"temperature"}
                    if "temperature" in roles
                    and child_settings.get("temperature") is None
                    else set()
                )
                | (
                    {"pressure"}
                    if "pressure" in roles and child_settings.get("pressure") is None
                    else set()
                )
            )
            if (
                absent & mandatory
                or child_settings["kind"] == "GAMMA_INF"
                and {role for role in roles if role.startswith("gamma")} <= absent
                or child_settings["kind"] == "LLE"
                and {role for role in roles if role in ("x1_alpha", "x1_beta")} <= absent
            ):
                omitted.append(
                    {
                        "row": row_index,
                        "reason": "Missing measured value for this series.",
                    }
                )
                continue
            projected.append(cells)
            row_indices.append(row_index)
        child_options = {
            key: val for key, val in child_settings.items() if key in conditions
        }
        for key in (
            "temperature_unit",
            "pressure_unit",
            "enthalpy_unit",
            "composition_basis",
            "composition_component",
        ):
            if (
                key not in options
                and key not in spec
                and any(
                    key in _column_settings(headers[index], mapping[index])
                    for index in selected
                )
            ):
                child_options.pop(key, None)
        child_options.update(mapping=roles, column_count=len(selected), layout="rows")
        if not projected:
            issues.append(
                f"{name}: no observations remain; ignore this series or supply measured values."
            )
            reports.append(
                {
                    "index": number,
                    "name": name,
                    "columns": columns,
                    "ready": False,
                    "observations": 0,
                    "excluded": omitted,
                    "observation_indices": [],
                }
            )
            continue
        # Already extracted rows are not rediscovered as separate numeric blocks:
        # source strings and missing optional cells must not truncate a series.
        child_block = {
            "headers": [[headers[index] for index in selected]],
            "rows": projected,
            "line_numbers": [line_numbers[index] for index in row_indices],
            "layout": "rows",
        }
        child = _interpret_table(
            child_block,
            options=child_options,
            components=components,
            value=value,
            table_index=0,
            blocks=[child_block],
        )
        for problem in child["issues"]:
            issues.append(f"{name}: {problem}")
        if not child["ready"]:
            reports.append(
                {
                    "index": number,
                    "name": name,
                    "columns": columns,
                    "ready": False,
                    "observations": 0,
                    "issues": child["issues"],
                    "excluded": omitted,
                    "observation_indices": [],
                }
            )
            continue
        child_excluded = {item["row"] for item in child["excluded"]}
        included = [
            index
            for position, index in enumerate(row_indices)
            if position not in child_excluded
        ]
        indices = []
        for observation, source_row in zip(
            child["raw_observations"], included, strict=True
        ):
            observation = dict(observation)
            if "id" in observation:
                observation["id"] = f"{observation['id']}.s{number + 1}"
            group_generated = "group" not in observation
            observation.setdefault("group", name)
            for key in ("weight", "sigma", "validation_only"):
                if key in spec:
                    observation[key] = spec[key]
            indices.append(len(raw))
            raw.append(observation)
            lineage.append(
                {
                    "series": number,
                    "row": source_row,
                    "columns": selected,
                    "group_generated": group_generated,
                }
            )
        omitted += [
            {"row": row_indices[item["row"]], "reason": item["reason"]}
            for item in child["excluded"]
        ]
        excluded.extend({**item, "series": number} for item in omitted)
        reports.append(
            {
                "index": number,
                "name": name,
                "columns": columns,
                "ready": True,
                "observations": len(indices),
                "settings": child["settings"],
                "excluded": omitted,
                "observation_indices": indices,
            }
        )
    if len(raw) > 2000:
        issues.append(
            f"The table expands to {len(raw)} observations; select fewer series or import at most 2000 observations at once."
        )
    ignored = [
        index for index in range(width) if index not in used and index not in shared
    ]
    notes = list(notes) + [
        f"Recognized {len(series)} VLE series from the measured column headings; review their conditions."
        if inferred
        else f"Repeated-series mode is explicit: {len(series)} series share {len(shared)} columns."
    ]
    if excluded:
        notes.append(
            f"{len(excluded)} series-observations are excluded; omissions and reasons are retained per series. Other measured series on those source rows remain available."
        )
    if ignored:
        notes.append(
            "Unassigned columns are ignored: "
            + ", ".join(str(index + 1) for index in ignored)
            + "."
        )
    field_info = _column_field_info(block, headers)
    return {
        "structured": False,
        "ready": not issues,
        "needs_review": True,
        "columns": [
            {
                "index": index,
                "label": headers[index] or f"Column {index + 1}",
                "role": mapping[index],
                "settings": _column_settings(headers[index], mapping[index]),
                "reason": field_info[index][1],
            }
            for index in range(width)
        ],
        "raw_rows": source_rows,
        "ambiguous_rows": ambiguous_rows,
        "line_numbers": line_numbers,
        "settings": settings,
        "table": table_index,
        "tables": [
            {"index": index, "rows": len(item["rows"]), "columns": len(item["rows"][0])}
            for index, item in enumerate(blocks)
        ],
        "flattened": block.get("flattened", False),
        "layout": block.get("layout", "rows"),
        "reference_component": None,
        "notes": notes,
        "issues": issues,
        "excluded": excluded,
        "raw_observations": raw if not issues else None,
        "original_text": value,
        "cell_edits": options.get("cell_edits", []),
        "series_available": True,
        "series": series,
        "shared_columns": shared,
        "series_inferred": inferred,
        "series_suggestion": suggestion,
        "series_reports": reports,
        "observation_sources": lineage,
        "series_header_hints": [_series_header_hint(header) for header in headers],
    }
