import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from compound_identity import get_compound_identity_resolver, parse_formula_counts


class CompoundIdentityFormulaTests(unittest.TestCase):
    def test_parse_formula_counts_handles_common_condensed_formulas(self):
        self.assertEqual(parse_formula_counts('H2O'), {'H': 2, 'O': 1})
        self.assertEqual(parse_formula_counts('CO2'), {'C': 1, 'O': 2})
        self.assertEqual(parse_formula_counts('CH3COOH'), {'C': 2, 'H': 4, 'O': 2})
        self.assertEqual(parse_formula_counts('CF3COOH'), {'C': 2, 'F': 3, 'O': 2, 'H': 1})

    def test_parse_formula_counts_handles_groups_and_hydrates(self):
        self.assertEqual(parse_formula_counts('(C2H5)2O'), {'C': 4, 'H': 10, 'O': 1})
        self.assertEqual(parse_formula_counts('Fe2(SO4)3'), {'Fe': 2, 'S': 3, 'O': 12})
        self.assertEqual(parse_formula_counts('CuSO4·5H2O'), {'Cu': 1, 'S': 1, 'O': 9, 'H': 10})

    def test_parse_formula_counts_preserves_element_case(self):
        self.assertEqual(parse_formula_counts('Co'), {'Co': 1})
        self.assertEqual(parse_formula_counts('CO'), {'C': 1, 'O': 1})

    def test_parse_formula_counts_accepts_subscripts(self):
        self.assertEqual(parse_formula_counts('H₂SO₄'), {'H': 2, 'S': 1, 'O': 4})

    def test_parse_formula_counts_rejects_malformed_input(self):
        for formula in ('', 'Xx2', 'C0H4', 'Fe2(SO4', 'Fe2)SO4(', 'CH3-COOH'):
            self.assertIsNone(parse_formula_counts(formula))

    def test_resolve_cas_uses_local_and_interaction_aliases(self):
        resolver = get_compound_identity_resolver()

        self.assertEqual(resolver.resolve_cas('ethanol'), '64-17-5')
        self.assertEqual(resolver.resolve_cas('3-Methylpyridien'), '108-99-6')
        self.assertEqual(resolver.resolve_cas('N-methylpyrrolidone'), '872-50-4')
        self.assertIsNone(resolver.resolve_cas('C3H6O'))


if __name__ == '__main__':
    unittest.main()
