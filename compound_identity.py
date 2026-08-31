"""Canonical compound identity lookup helpers.

This module keeps name/formula/CAS/SMILES resolution in one place.  It is
deliberately conservative with molecular formulas because formulas can refer
to multiple structural isomers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional


DATA_DIR = Path(__file__).parent / "data"

ELEMENT_SYMBOLS = frozenset({
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
})

_SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_name(identifier: str) -> str:
    value = str(identifier).strip().lower()
    value = value.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-")
    value = value.replace("flouro", "fluoro")
    value = value.replace("propnaol", "propanol")
    value = value.replace("pyridien", "pyridine")
    value = value.replace("pyridinr", "pyridine")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def compact_identifier(identifier: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(identifier).lower())


def normalize_formula(identifier: str) -> str:
    return re.sub(r"[^A-Za-z0-9()]+", "", str(identifier)).upper()


def looks_like_formula(identifier: str) -> bool:
    value = str(identifier).strip()
    if not value or " " in value:
        return False
    return bool(re.fullmatch(r"(?:[A-Z][a-z]?\d*|\(|\)|\d+)+", value))


def classify_identifier(identifier: str) -> str:
    """Classify an external identifier before invoking identity providers."""
    value = str(identifier or '').strip()
    if not value:
        return 'empty'
    if value.lower().startswith('inchi='):
        return 'inchi'
    if re.fullmatch(r'[A-Z]{14}-[A-Z]{10}-[A-Z]', value.upper()):
        return 'inchikey'
    if re.fullmatch(r'\d{2,7}-\d{2}-\d', value):
        return 'cas'
    if looks_like_formula(value):
        return 'formula'
    if any(char.isspace() for char in value) or re.search(r'[,;]', value):
        return 'name'
    if re.fullmatch(r'[A-Za-z][A-Za-z-]*', value) and not value.isupper():
        return 'name'
    if re.search(r'[\[\]()=#@+\\/]', value) or re.search(r'[bcnops]\d', value):
        return 'smiles'
    return 'name'


def parse_formula_counts(formula: str) -> Optional[dict[str, int]]:
    """Parse a molecular formula into element counts.

    Supports ordinary condensed formulas such as ``CH3COOH``, grouped formulas
    such as ``(C2H5)2O`` and ``Fe2(SO4)3``, and hydrate/dot notation such as
    ``CuSO4·5H2O``. Returns ``None`` when the input is malformed or contains an
    unknown element symbol.
    """
    value = str(formula or "").strip().translate(_SUBSCRIPT_TRANSLATION)
    value = re.sub(r"\s+", "", value)
    value = value.replace("*", ".").replace("·", ".").replace("•", ".")
    if not value:
        return None

    total: dict[str, int] = {}
    for segment in value.split("."):
        if not segment:
            return None
        multiplier, pos = _parse_formula_number(segment, 0)
        if pos == 0:
            multiplier = 1
        if multiplier <= 0:
            return None
        counts, pos = _parse_formula_group(segment, pos, None)
        if counts is None or pos != len(segment):
            return None
        for element, count in counts.items():
            total[element] = total.get(element, 0) + multiplier * count

    return total or None


def _parse_formula_group(
    formula: str,
    pos: int,
    terminator: Optional[str],
) -> tuple[Optional[dict[str, int]], int]:
    counts: dict[str, int] = {}
    closing = {")": "(", "]": "["}

    while pos < len(formula):
        char = formula[pos]
        if terminator and char == terminator:
            return counts, pos + 1
        if char in closing:
            return None, pos
        if char in "([":
            group_terminator = ")" if char == "(" else "]"
            group_counts, pos = _parse_formula_group(formula, pos + 1, group_terminator)
            if group_counts is None:
                return None, pos
            multiplier, pos = _parse_formula_number(formula, pos)
            if multiplier <= 0:
                return None, pos
            for element, count in group_counts.items():
                counts[element] = counts.get(element, 0) + multiplier * count
            continue
        if not char.isupper():
            return None, pos

        element_end = pos + 1
        if element_end < len(formula) and formula[element_end].islower():
            element_end += 1
        element = formula[pos:element_end]
        if element not in ELEMENT_SYMBOLS:
            return None, pos
        multiplier, pos = _parse_formula_number(formula, element_end)
        if multiplier <= 0:
            return None, pos
        counts[element] = counts.get(element, 0) + multiplier

    if terminator:
        return None, pos
    return counts, pos


def _parse_formula_number(formula: str, pos: int) -> tuple[int, int]:
    start = pos
    while pos < len(formula) and formula[pos].isdigit():
        pos += 1
    if pos == start:
        return 1, pos
    return int(formula[start:pos]), pos


def straight_chain_alkane_formula_symbol(identifier: str) -> Optional[str]:
    """Return the normal-alkane formula symbol for bare CnH2n+2 input."""
    formula = normalize_formula(identifier)
    match = re.fullmatch(r"C(\d*)H(\d+)", formula)
    if not match:
        return None
    carbon_count = int(match.group(1) or "1")
    hydrogen_count = int(match.group(2))
    if carbon_count < 1 or hydrogen_count != 2 * carbon_count + 2:
        return None
    return f"C{carbon_count if carbon_count > 1 else ''}H{hydrogen_count}"


@dataclass
class CompoundIdentity:
    """Canonical local identity for a compound."""

    symbol: str
    name: str = ""
    formula: str = ""
    cas: str = ""
    smiles: str = ""
    aliases: list[str] = field(default_factory=list)

    def identifiers(self) -> list[str]:
        values = [self.symbol, self.name, self.formula, self.cas, *self.aliases]
        result = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result


class CompoundIdentityResolver:
    """Resolve common identifiers to local canonical symbols."""

    MANUAL_ALIASES = {
        # Alcohol synonyms that appear in property tables but not chemicals.json.
        "ethyl alcohol": "C2H5OH",
        "ethyl-alcohol": "C2H5OH",
        "grain alcohol": "C2H5OH",
        "methyl alcohol": "CH3OH",
        "wood alcohol": "CH3OH",
        # Common spelling variants.
        "carbon tet": "CCl4",
        "ethyl ether": "(C2H5)2O",
        "ether": "(C2H5)2O",
        "tfa": "CF3COOH",
        "acrylic-acid": "C2H3COOH",
        "propenoic acid": "C2H3COOH",
        "2 propenoic acid": "C2H3COOH",
        "2-propenoic acid": "C2H3COOH",
        "prop 2 enoic acid": "C2H3COOH",
        "prop-2-enoic acid": "C2H3COOH",
    }

    SMILES_ALIASES = {
        "O": "H2O",
        "CO": "CH3OH",
        "CCO": "C2H5OH",
        "CCCO": "C3H8O",
        "CC(C)O": "C3H7OH",
        "C": "CH4",
        "CC": "C2H6",
        "C=C": "C2H4",
        "CCC": "C3H8",
        "CC=C": "C3H6",
        "CCCC": "C4H10",
        "CCCCC": "C5H12",
        "CCCCCC": "C6H14",
        "CCCCCCC": "C7H16",
        "c1ccccc1": "C6H6",
        "Cc1ccccc1": "C7H8",
        "CC(=O)C": "CH3COCH3",
        "CC(=O)O": "CH3COOH",
        "C=CC(=O)O": "C2H3COOH",
        "CC=O": "CH3CHO",
        "C=O": "CH2O",
        "O=C=O": "CO2",
        "[C-]#[O+]": "CO",
        "N": "NH3",
        "N#N": "N2",
        "O=O": "O2",
        "[H][H]": "H2",
        "C1CCCCC1": "C6H12",
        "CCOCC": "(C2H5)2O",
        "CC(=O)OCC": "C4H8O2",
        "ClC(Cl)Cl": "CHCl3",
        "ClCCl": "CH2Cl2",
        "ClC(Cl)(Cl)Cl": "CCl4",
    }

    def __init__(self, chemicals_path: Optional[Path] = None):
        self.chemicals_path = chemicals_path or DATA_DIR / "chemicals.json"
        self.identities: dict[str, CompoundIdentity] = {}
        self._name_aliases: dict[str, str] = {}
        self._compact_aliases: dict[str, str] = {}
        self._cas_aliases: dict[str, str] = {}
        self._external_cas_aliases: dict[str, str] = {}
        self._formula_aliases: dict[str, str] = {}
        self._ambiguous_formulas: dict[str, set[str]] = {}
        self._canonical_smiles_aliases: dict[str, str] = {}
        self._inchi_aliases: dict[str, str] = {}
        self._inchikey_aliases: dict[str, str] = {}
        self._structure_aliases_ready = False
        self._load()
        self._load_interaction_cas_index()

    def _add_alias(self, alias: str, symbol: str):
        if not alias or symbol not in self.identities:
            return
        key = normalize_name(alias)
        if key:
            self._name_aliases.setdefault(key, symbol)
        compact = compact_identifier(alias)
        if compact:
            self._compact_aliases.setdefault(compact, symbol)

    def _load(self):
        if not self.chemicals_path.exists():
            return
        payload = json.loads(self.chemicals_path.read_text())
        formula_to_symbols: dict[str, set[str]] = {}

        for symbol, data in payload.get("chemicals", {}).items():
            identity = CompoundIdentity(
                symbol=symbol,
                name=data.get("name", symbol) or symbol,
                formula=data.get("formula", symbol) or symbol,
                cas=data.get("CAS", "") or "",
                smiles=data.get("smiles", "") or "",
            )
            self.identities[symbol] = identity
            formula_to_symbols.setdefault(normalize_formula(identity.formula), set()).add(symbol)

        for symbol, identity in self.identities.items():
            for alias in (symbol, identity.name):
                self._add_alias(alias, symbol)
            if identity.cas:
                self._cas_aliases[compact_identifier(identity.cas)] = symbol

        for formula, symbols in formula_to_symbols.items():
            if len(symbols) == 1:
                self._formula_aliases[formula] = next(iter(symbols))
            else:
                self._ambiguous_formulas[formula] = symbols

        for alias, symbol in self.MANUAL_ALIASES.items():
            self._add_alias(alias, symbol)
            if symbol in self.identities:
                self.identities[symbol].aliases.append(alias)

    def _add_external_cas_alias(self, alias: str, cas: str):
        if not alias or not cas:
            return
        compact_cas = compact_identifier(cas)
        if not compact_cas:
            return
        self._external_cas_aliases.setdefault(compact_cas, cas)
        if looks_like_formula(alias):
            return
        name_key = normalize_name(alias)
        if name_key:
            self._external_cas_aliases.setdefault(name_key, cas)
        compact = compact_identifier(alias)
        if compact:
            self._external_cas_aliases.setdefault(compact, cas)

    def _load_interaction_cas_index(self):
        path = DATA_DIR / "interaction_component_cas_index.json"
        if not path.exists():
            return
        payload = json.loads(path.read_text())
        for cas, entry in payload.get("components", {}).items():
            aliases = [cas, entry.get("name", "")]
            aliases.extend(entry.get("aliases") or [])
            for alias in aliases:
                self._add_external_cas_alias(str(alias), cas)

    def resolve_symbol(self, identifier: str, allow_formula: bool = True) -> Optional[str]:
        """Return a canonical local symbol, or None when unknown/ambiguous."""
        if not identifier:
            return None
        if identifier in self.identities:
            return identifier

        cas_key = compact_identifier(identifier)
        if cas_key in self._cas_aliases:
            return self._cas_aliases[cas_key]

        name_key = normalize_name(identifier)
        if name_key in self._name_aliases:
            return self._name_aliases[name_key]

        compact = compact_identifier(identifier)
        if compact in self._compact_aliases:
            return self._compact_aliases[compact]

        identifier_kind = classify_identifier(identifier)
        if identifier_kind in {'smiles', 'inchi', 'inchikey'}:
            if identifier in self.SMILES_ALIASES:
                symbol = self.SMILES_ALIASES[identifier]
                if symbol in self.identities:
                    return symbol
            structure_symbol = self._resolve_structure_symbol(identifier)
            if structure_symbol:
                return structure_symbol

        if allow_formula and identifier_kind == 'formula':
            formula = normalize_formula(identifier)
            return self._formula_aliases.get(formula)

        return None

    @staticmethod
    def is_inchi(identifier: str) -> bool:
        return str(identifier or '').strip().lower().startswith('inchi=')

    @staticmethod
    def is_inchikey(identifier: str) -> bool:
        return bool(re.fullmatch(
            r'[A-Z]{14}-[A-Z]{10}-[A-Z]',
            str(identifier or '').strip().upper(),
        ))

    def _ensure_structure_aliases(self) -> None:
        if self._structure_aliases_ready:
            return
        self._structure_aliases_ready = True
        try:
            from rdkit import Chem, rdBase
        except ImportError:
            return

        candidates = [
            (smiles, symbol)
            for smiles, symbol in self.SMILES_ALIASES.items()
            if symbol in self.identities
        ]
        candidates.extend(
            (identity.smiles, symbol)
            for symbol, identity in self.identities.items()
            if identity.smiles
        )
        for smiles, symbol in candidates:
            try:
                with rdBase.BlockLogs():
                    molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    continue
                canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
                inchi = Chem.MolToInchi(molecule)
                inchikey = Chem.MolToInchiKey(molecule)
            except Exception:
                continue
            if canonical:
                self._canonical_smiles_aliases.setdefault(canonical, symbol)
            if inchi:
                self._inchi_aliases.setdefault(inchi, symbol)
            if inchikey:
                self._inchikey_aliases.setdefault(inchikey.upper(), symbol)

    def _resolve_structure_symbol(self, identifier: str) -> Optional[str]:
        text = str(identifier or '').strip()
        if not text:
            return None
        self._ensure_structure_aliases()
        if self.is_inchi(text):
            symbol = self._inchi_aliases.get(text)
            if symbol:
                return symbol
            try:
                from rdkit import Chem, rdBase
                with rdBase.BlockLogs():
                    molecule = Chem.MolFromInchi(text)
                canonical = (
                    Chem.MolToSmiles(molecule, isomericSmiles=True)
                    if molecule is not None else ''
                )
            except Exception:
                canonical = ''
            return self._canonical_smiles_aliases.get(canonical)
        if self.is_inchikey(text):
            return self._inchikey_aliases.get(text.upper())
        try:
            from rdkit import Chem, rdBase
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(text)
            canonical = (
                Chem.MolToSmiles(molecule, isomericSmiles=True)
                if molecule is not None else ''
            )
        except Exception:
            canonical = ''
        return self._canonical_smiles_aliases.get(canonical)

    def is_structure_identifier(self, identifier: str) -> bool:
        """Return whether text is a parseable explicit structure identifier."""
        text = str(identifier or '').strip()
        identifier_kind = classify_identifier(text)
        if identifier_kind in {'inchi', 'inchikey'}:
            return True
        if identifier_kind != 'smiles':
            return False
        try:
            from rdkit import Chem, rdBase
            with rdBase.BlockLogs():
                return Chem.MolFromSmiles(text) is not None
        except Exception:
            return False

    def resolve_structure_symbol(self, smiles: str) -> Optional[str]:
        """Match an already resolved molecular graph to a local identity."""
        return self._resolve_structure_symbol(smiles)

    def resolve(self, identifier: str, allow_formula: bool = True) -> Optional[CompoundIdentity]:
        symbol = self.resolve_symbol(identifier, allow_formula=allow_formula)
        return self.identities.get(symbol) if symbol else None

    def resolve_cas(self, identifier: str, allow_formula: bool = False) -> Optional[str]:
        """Return a CAS number for a known local or external indexed identity."""
        if not identifier:
            return None
        compact = compact_identifier(identifier)
        if re.fullmatch(r"\d{2,7}\d{2}\d", compact):
            match = re.fullmatch(r"(\d{2,7})(\d{2})(\d)", compact)
            if match:
                return "-".join(match.groups())

        identity = self.resolve(identifier, allow_formula=allow_formula)
        if identity and identity.cas:
            return identity.cas

        if not allow_formula and looks_like_formula(identifier):
            return None

        name_key = normalize_name(identifier)
        for key in (name_key, compact):
            cas = self._external_cas_aliases.get(key)
            if cas:
                return cas
        return None

    def is_ambiguous_formula(self, identifier: str) -> bool:
        if not looks_like_formula(identifier):
            return False
        return normalize_formula(identifier) in self._ambiguous_formulas

    def straight_chain_alkane_symbol(self, identifier: str) -> Optional[str]:
        symbol = straight_chain_alkane_formula_symbol(identifier)
        if symbol and symbol in self.identities:
            return symbol
        return None

    def candidate_identifiers(
        self,
        identifier: str,
        extra: Optional[list[str]] = None,
        allow_formula: bool = True,
    ) -> list[str]:
        """Return ordered aliases suitable for property table lookup."""
        result = []

        def add(value: Optional[str]):
            if value and value not in result:
                result.append(value)

        add(identifier)
        identity = self.resolve(identifier, allow_formula=allow_formula)
        if identity:
            for value in identity.identifiers():
                if not allow_formula and value == identity.formula:
                    continue
                add(value)

        for value in extra or []:
            add(value)
            extra_identity = self.resolve(value, allow_formula=allow_formula)
            if extra_identity:
                for candidate in extra_identity.identifiers():
                    if not allow_formula and candidate == extra_identity.formula:
                        continue
                    add(candidate)

        return result


@lru_cache(maxsize=1)
def get_compound_identity_resolver() -> CompoundIdentityResolver:
    return CompoundIdentityResolver()
