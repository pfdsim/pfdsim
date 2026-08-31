import unittest

from property_resolution.organic_classification import (
    classify_strict_molecular_organic,
    hydrogen_bond_donor_profile,
    is_strict_organic_formula_counts,
)


class StrictOrganicClassificationTests(unittest.TestCase):
    def test_known_inorganic_carbon_families_are_rejected(self):
        cases = (
            ('74-90-8', 'CHN', 'C#N'),
            ('630-08-0', 'CO', '[C-]#[O+]'),
            ('124-38-9', 'CO2', 'O=C=O'),
            ('75-15-0', 'CS2', 'S=C=S'),
            ('463-58-1', 'COS', 'O=C=S'),
            ('460-19-5', 'C2N2', 'N#CC#N'),
            ('504-64-3', 'C3O2', 'O=C=C=C=O'),
            ('463-79-6', 'CH2O3', 'OC(=O)O'),
            ('75-44-5', 'COCl2', 'O=C(Cl)Cl'),
            ('353-50-4', 'COF2', 'O=C(F)F'),
        )
        for cas, formula, smiles in cases:
            with self.subTest(cas=cas):
                result = classify_strict_molecular_organic(
                    cas=cas,
                    formula=formula,
                    smiles=smiles,
                )
                self.assertFalse(result.is_organic)

    def test_strict_classifier_retains_intended_organic_families(self):
        cases = (
            ('C2H3N', 'CC#N'),               # organic nitrile
            ('CH2O2', 'OC=O'),                # formic acid
            ('C2H2O4', 'O=C(O)C(=O)O'),       # oxalic acid
            ('CH4N2O', 'NC(=O)N'),             # urea
            ('CCl4', 'ClC(Cl)(Cl)Cl'),         # carbon tetrachloride
            ('C4H12Si', 'C[Si](C)(C)C'),       # ordinary organosilicon
        )
        for formula, smiles in cases:
            with self.subTest(formula=formula):
                result = classify_strict_molecular_organic(
                    formula=formula,
                    smiles=smiles,
                )
                self.assertTrue(result.is_organic)

    def test_formula_only_classification_uses_same_strict_policy(self):
        self.assertTrue(is_strict_organic_formula_counts({'C': 2, 'H': 2, 'O': 4}))
        self.assertFalse(is_strict_organic_formula_counts({'C': 1, 'H': 1, 'N': 1}))
        self.assertFalse(is_strict_organic_formula_counts({'Na': 2, 'C': 1, 'O': 3}))
        self.assertFalse(is_strict_organic_formula_counts({'C': 1, 'H': 2, 'O': 3}))

    def test_charged_salts_and_disconnected_structures_are_rejected(self):
        charged = classify_strict_molecular_organic(
            formula='C2H3O2',
            smiles='CC(=O)[O-]',
        )
        salt = classify_strict_molecular_organic(
            formula='CH5NO3',
            smiles='[NH4+].[O-]C(=O)O',
        )
        self.assertFalse(charged.is_organic)
        self.assertFalse(salt.is_organic)


class HydrogenBondDonorProfileTests(unittest.TestCase):
    def test_profiles_match_final_liquid_cp_classes(self):
        ethanol = hydrogen_bond_donor_profile('CCO')
        self.assertEqual(ethanol.alcohol_oh, 1)
        self.assertEqual(ethanol.onh_classes, ('alcohol',))
        self.assertFalse(ethanol.is_polyol)

        glycerol = hydrogen_bond_donor_profile('OCC(O)CO')
        self.assertEqual(glycerol.alcohol_oh, 3)
        self.assertTrue(glycerol.is_polyol)
        self.assertEqual(glycerol.gc_fractions()['polyol'], 1.0)

        oxalic = hydrogen_bond_donor_profile('O=C(O)C(=O)O')
        self.assertEqual(oxalic.carboxylic_acid_oh, 2)
        self.assertEqual(oxalic.onh_classes, ('acid',))

        amide = hydrogen_bond_donor_profile('CNC=O')
        self.assertEqual(amide.nitrogen_nh, 1)
        self.assertEqual(amide.onh_classes, ('nitrogen',))

        thiol = hydrogen_bond_donor_profile('CS')
        self.assertEqual(thiol.thiol_sh, 1)
        self.assertEqual(thiol.all_classes, ('thiol',))

        mixed = hydrogen_bond_donor_profile('OCCNCCO')
        self.assertEqual(mixed.alcohol_oh, 2)
        self.assertEqual(mixed.nitrogen_nh, 1)
        self.assertEqual(mixed.onh_classes, ('alcohol', 'nitrogen'))


if __name__ == '__main__':
    unittest.main()
