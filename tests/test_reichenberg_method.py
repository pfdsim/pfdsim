import math
import unittest
from collections import Counter

import reichenberg_method as rm


class ReichenbergMethodTests(unittest.TestCase):
    def assertGroups(self, smiles, expected):
        result = rm.fragment(smiles)
        self.assertEqual(result.groups, Counter(expected))
        return result

    def test_fragments_perry_ethyl_acetate_example(self):
        result = self.assertGroups(
            "CCOC(=O)C",
            {"ch3": 2, "ch2": 1, "ester": 1},
        )
        self.assertAlmostEqual(result.contribution_sum, 37.96)

    def test_fragments_representative_functional_groups(self):
        cases = {
            "CO": {"ch3": 1, "oh_alcohol": 1},
            "CCO": {"ch3": 1, "ch2": 1, "oh_alcohol": 1},
            "CC(=O)C": {"ch3": 2, "carbonyl": 1},
            "CC#N": {"ch3": 1, "nitrile": 1},
            "CC=O": {"ch3": 1, "aldehyde": 1},
            "CC(=O)O": {"ch3": 1, "carboxylic_acid": 1},
            "CNC": {"ch3": 2, "amine_nh": 1},
        }
        for smiles, expected in cases.items():
            with self.subTest(smiles=smiles):
                self.assertGroups(smiles, expected)

    def test_fragments_ring_and_aromatic_groups(self):
        self.assertGroups("c1ccccc1", {"ring_alkene_ch": 6})
        self.assertGroups("C1CCCCC1", {"ring_ch2": 6})
        self.assertGroups(
            "n1ccccc1",
            {"ring_imine_n": 1, "ring_alkene_ch": 5},
        )

    def test_structure_profile_counts_heavy_atoms_and_carbon_hydrogen_bonds(self):
        ethanol = rm.structure_profile("CCO")
        self.assertEqual(ethanol.heavy_atoms, 3)
        self.assertEqual(ethanol.carbon_atoms, 2)
        self.assertEqual(ethanol.heteroatoms, 1)
        self.assertTrue(ethanol.has_carbon_hydrogen_bond)

        carbon_dioxide = rm.structure_profile("O=C=O")
        self.assertEqual(carbon_dioxide.heavy_atoms, 3)
        self.assertFalse(carbon_dioxide.has_carbon_hydrogen_bond)

    def test_rejects_structures_not_represented_in_table(self):
        for smiles in (
            "N#C", "N#CC#N", "CN(C)C", "CCI", "CSC", "F", "BrBr", "C1CN1"
        ):
            with self.subTest(smiles=smiles):
                with self.assertRaises(rm.ReichenbergFragmentationError):
                    rm.fragment(smiles)

    def test_reproduces_perry_ethyl_acetate_example(self):
        result = rm.viscosity_Pa_s(
            401.25,
            88.1051,
            523.3,
            38.8,
            dipole_D=1.78,
            fragmentation=rm.fragment("CCOC(=O)C"),
        )
        self.assertTrue(math.isclose(result, 1.003e-5, rel_tol=5.0e-4))


if __name__ == "__main__":
    unittest.main()
