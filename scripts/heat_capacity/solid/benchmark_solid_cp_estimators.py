#!/usr/bin/env python3
"""Reproduce the retained 298.15 K solid-Cp estimator benchmarks."""

from __future__ import annotations

import sys
import re
from pathlib import Path

import numpy as np
from chemicals import heat_capacity as source_cp
from chemicals.elements import periodic_table, simple_formula_parser
from chemicals.heat_capacity import Lastovka_solid
from chemicals.identifiers import search_chemical


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from property_resolution.organic_classification import is_strict_organic_formula_counts
from property_resolution.solid_cp import (
    MODIFIED_KOPP_CONTRIBUTIONS_J_MOL_K,
    MODIFIED_KOPP_OTHER_J_MOL_K,
)
from compound_identity import parse_formula_counts


LASTOVKA_ELEMENTS = {'C', 'H', 'N', 'O', 'S'}
LASTOVKA_LIMITS = {
    'C': (0.613, 0.952), 'H': (0.0373, 0.152),
    'N': (0.0, 0.154), 'O': (0.0, 0.188), 'S': (0.0, 0.296),
}


def metrics(errors: list[float]) -> str:
    values = np.asarray(errors) * 100.0
    return (
        f'n={len(values)} mean={values.mean():.3f}% median={np.median(values):.3f}% '
        f'P90={np.quantile(values, 0.9):.3f}% max={values.max():.3f}% '
        f'within10={(values < 10.0).mean() * 100.0:.1f}%'
    )


def main() -> None:
    kopp_organic = []
    lastovka = []
    rows = source_cp.CRC_standard_data[source_cp.CRC_standard_data.Cps.notna()]
    for cas, row in rows.iterrows():
        try:
            metadata = search_chemical(cas)
            if str(metadata.CASs) != str(cas):
                continue
            atoms = simple_formula_parser(metadata.formula)
            molecular_weight = float(metadata.MW)
        except Exception:
            continue
        reference = float(row.Cps)
        kopp = sum(
            count * MODIFIED_KOPP_CONTRIBUTIONS_J_MOL_K.get(
                element, MODIFIED_KOPP_OTHER_J_MOL_K,
            )
            for element, count in atoms.items()
        )
        if is_strict_organic_formula_counts(atoms):
            kopp_organic.append(abs(kopp - reference) / reference)
        if (
            12.24 <= molecular_weight <= 402.4
            and set(atoms) <= LASTOVKA_ELEMENTS
            and atoms.get('C', 0) > 0
            and atoms.get('H', 0) > 0
        ):
            fractions = {
                element: atoms.get(element, 0) * periodic_table[element].MW / molecular_weight
                for element in LASTOVKA_ELEMENTS
            }
            if all(
                low <= fractions[element] <= high
                for element, (low, high) in LASTOVKA_LIMITS.items()
            ):
                alpha = sum(atoms.values()) / molecular_weight
                prediction = Lastovka_solid(298.15, alpha, MW=molecular_weight)
                lastovka.append(abs(prediction - reference) / reference)

    kopp_broad = []
    for cas, phases in source_cp.Cp_dict_PerryI.items():
        if cas not in rows.index:
            continue
        reference = rows.loc[cas, 'Cps']
        if reference != reference:
            continue
        row = phases.get('c') or phases.get('gls')
        formula = str((row or {}).get('Formula') or '').strip()
        if re.match(r'^\d+\s', formula):
            continue
        atoms = parse_formula_counts(formula)
        if not atoms:
            continue
        try:
            metadata = search_chemical(cas)
        except Exception:
            metadata = None
        if metadata is not None and str(metadata.CASs) == str(cas):
            resolved = simple_formula_parser(metadata.formula)
            source_elements = {element for element in atoms if element not in {'H', 'O'}}
            resolved_elements = {element for element in resolved if element not in {'H', 'O'}}
            if source_elements != resolved_elements:
                continue
        prediction = sum(
            count * MODIFIED_KOPP_CONTRIBUTIONS_J_MOL_K.get(
                element, MODIFIED_KOPP_OTHER_J_MOL_K,
            )
            for element, count in atoms.items()
        )
        kopp_broad.append(abs(prediction - float(reference)) / float(reference))

    print('Modified Kopp, strict organic CRC:', metrics(kopp_organic))
    print('Lastovka, published organic domain:', metrics(lastovka))
    print('Modified Kopp, vetted Perry-formula/CRC subset:', metrics(kopp_broad))


if __name__ == '__main__':
    main()
