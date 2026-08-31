"""Extract Appendix B pure-species properties from the bundled Smith textbook.

This is intentionally an offline build step. Runtime lookup uses the generated
``data/textbook_properties.json`` file and never opens the PDF.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOURCE_DATA_DIR = DATA_DIR / "source"
OUTPUT = DATA_DIR / "textbook_properties.json"
PDF_PATTERN = "J.M. Smith*.pdf"


def parse_float(text: str) -> float | None:
    text = text.strip().replace("\u2212", "-")
    if not text or text.startswith("."):
        return None
    if text.endswith("."):
        text = text[:-1]
    return float(text)


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("*", "")).strip()


def merge_key(name: str) -> str:
    key = normalize_name(name).lower()
    key = key.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-")
    key = key.replace("iso-", "iso")
    return key


def parse_temperature_range(text: str) -> tuple[float, float] | None:
    text = text.replace("\u2212", "-")
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*[—-]\s*(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1)) + 273.15, float(match.group(2)) + 273.15


def formula_from_name(name: str) -> str | None:
    match = re.search(r"\(([A-Z][A-Za-z0-9]*)\)", name)
    return match.group(1) if match else None


def extract_pages(reader: PdfReader, start_page: int, end_page: int) -> list[str]:
    """Extract one-indexed page range, inclusive."""
    return [
        reader.pages[i - 1].extract_text() or ""
        for i in range(start_page, end_page + 1)
    ]


def parse_characteristic_table(pages: list[str]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    number_pattern = re.compile(r"[-\u2212]?\d+(?:\.\d*)?")

    for page in pages:
        for raw_line in page.splitlines():
            line = normalize_name(raw_line)
            if not line or line.startswith(("Table", "Molar", "Final", "smi", "†")):
                continue

            matches = list(number_pattern.finditer(line))
            if len(matches) < 6:
                continue

            n_values = 7 if len(matches) >= 7 else 6
            data_matches = matches[-n_values:]
            name = normalize_name(line[: data_matches[0].start()])

            if not name or any(header in name for header in ("Appendix", "Properties")):
                continue

            values = [parse_float(match.group(0)) for match in data_matches]
            if n_values == 6 and ". ." in line:
                # Sulfuric acid has no omega entry in the source table; the
                # remaining six values are MW, Tc, Pc, Zc, Vc, and Tn.
                mw, Tc, Pc, Zc, Vc, Tb = values
                omega = None
            elif n_values == 6:
                mw, omega, Tc, Pc, Zc, Vc = values
                Tb = None
            else:
                mw, omega, Tc, Pc, Zc, Vc, Tb = values

            if mw is None or Tc is None or Pc is None:
                continue

            entries[name] = {
                "name": name,
                "formula": formula_from_name(name),
                "MW": mw,
                "omega": omega,
                "Tc": Tc,
                "Pc": Pc,
                "Zc": Zc,
                "Vc": Vc,
                "Tb": Tb,
                "source_table": "Smith8 Appendix B Table B.1",
            }

    return entries


def parse_formula_table(pages: list[str]) -> dict[str, str]:
    """Extract name-to-formula hints from Appendix C tables."""
    formulas: dict[str, str] = {}
    formula_pattern = re.compile(r"^(?P<name>.+?)\s+(?P<formula>(?:[A-Z][a-z]?\d*)+)\s+(?:\(|[-\u2212]?\d)")

    for page in pages:
        for raw_line in page.splitlines():
            line = normalize_name(raw_line)
            if not line or line.startswith(("Table", "Chemical species", "Alkanes", "Miscellaneous", "Final", "smi")):
                continue

            match = formula_pattern.match(line)
            if not match:
                continue

            name = normalize_name(match.group("name"))
            formula = match.group("formula")
            if name and formula:
                formulas.setdefault(merge_key(name), formula)

    return formulas


def parse_antoine_table(pages: list[str]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    row_pattern = re.compile(
        r"^(?P<name>.+?)\s+"
        r"(?P<formula>[A-Z][A-Za-z0-9]*)\s+"
        r"(?P<A>\d+(?:\.\d+)?)\s+"
        r"(?P<B>\d+(?:\.\d+)?)\s+"
        r"(?P<C>\d+(?:\.\d+)?)\s+"
        r"(?P<Tmin>[-\u2212]?\d+(?:\.\d+)?)\s*[—-]\s*"
        r"(?P<Tmax>[-\u2212]?\d+(?:\.\d+)?)\s+"
        r"(?P<Hvap>\d+(?:\.\d+)?)\s+"
        r"(?P<Tb>[-\u2212]?\d+(?:\.\d+)?)$"
    )

    for page in pages:
        for raw_line in page.splitlines():
            line = normalize_name(raw_line)
            match = row_pattern.match(line)
            if not match:
                continue

            A_ln_kpa = float(match.group("A"))
            B_ln_kpa = float(match.group("B"))
            C_celsius = float(match.group("C"))
            Tmin_C = float(match.group("Tmin").replace("\u2212", "-"))
            Tmax_C = float(match.group("Tmax").replace("\u2212", "-"))
            Tb_C = float(match.group("Tb").replace("\u2212", "-"))

            # Textbook form: ln(Psat/kPa) = A - B/(t_C + C).
            # Runtime form: log10(Psat/bar) = A - B/(t_C + C).
            A_log10_bar = A_ln_kpa / math.log(10.0) - 2.0
            B_log10_bar = B_ln_kpa / math.log(10.0)

            name = normalize_name(match.group("name"))
            entries[name] = {
                "name": name,
                "formula": match.group("formula"),
                "antoine_A": A_log10_bar,
                "antoine_B": B_log10_bar,
                "antoine_C": C_celsius,
                "antoine_Tmin": Tmin_C + 273.15,
                "antoine_Tmax": Tmax_C + 273.15,
                "Hvap": float(match.group("Hvap")),
                "Tb": Tb_C + 273.15,
                "textbook_A_ln_kPa": A_ln_kpa,
                "textbook_B_ln_kPa": B_ln_kpa,
                "textbook_C_C": C_celsius,
                "source_table": "Smith8 Appendix B Table B.2",
            }

    return entries


def merge_tables(
    characteristic: dict[str, dict[str, Any]],
    antoine: dict[str, dict[str, Any]],
    formulas: dict[str, str],
) -> dict[str, Any]:
    chemicals_by_key = {merge_key(name): dict(entry) for name, entry in characteristic.items()}

    for name, entry in antoine.items():
        key = merge_key(name)
        combined = chemicals_by_key.setdefault(
            key,
            {
                "name": name,
                "formula": entry["formula"],
                "source_table": "Smith8 Appendix B Table B.2",
            },
        )
        combined["formula"] = combined.get("formula") or entry["formula"]
        for key, value in entry.items():
            if key in {"name", "source_table"}:
                continue
            if value is not None:
                combined[key] = value
        sources = [combined.get("source_table"), entry["source_table"]]
        combined["source_table"] = "; ".join(dict.fromkeys(filter(None, sources)))

    for key, formula in formulas.items():
        if key in chemicals_by_key and not chemicals_by_key[key].get("formula"):
            chemicals_by_key[key]["formula"] = formula

    chemicals = {
        entry["name"]: entry
        for entry in sorted(chemicals_by_key.values(), key=lambda item: item["name"])
    }

    return {
        "metadata": {
            "source": "J.M. Smith, H.C. Van Ness, M.M. Abbott, M.T. Swihart, Introduction to Chemical Engineering Thermodynamics, 8th ed., Appendix B",
            "extracted_pages": [669, 670, 671, 672, 673, 675, 676, 677, 678, 679, 680, 681],
            "units": {
                "MW": "g/mol",
                "Tc": "K",
                "Pc": "bar",
                "Vc": "cm3/mol",
                "Tb": "K",
                "Hvap": "kJ/mol at normal boiling point",
                "antoine": "log10(P_bar) = A - B/(T_C + C)",
                "antoine_Tmin_Tmax": "K",
            },
        },
        "chemicals": dict(sorted(chemicals.items())),
    }


def main() -> None:
    pdfs = sorted(SOURCE_DATA_DIR.glob(PDF_PATTERN))
    if not pdfs:
        raise FileNotFoundError(f"No textbook PDF matching {PDF_PATTERN!r} in {SOURCE_DATA_DIR}")

    reader = PdfReader(str(pdfs[0]))
    characteristic = parse_characteristic_table(extract_pages(reader, 669, 671))
    antoine = parse_antoine_table(extract_pages(reader, 672, 673))
    formulas = parse_formula_table(extract_pages(reader, 675, 681))
    payload = merge_tables(characteristic, antoine, formulas)

    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(payload['chemicals'])} textbook property records to {OUTPUT}")


if __name__ == "__main__":
    main()
