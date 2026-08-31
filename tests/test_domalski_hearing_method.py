"""Paper-table validation for the standalone Domalski--Hearing method."""

import os
import sys
import unittest
import hashlib
import json
from collections import Counter

from rdkit import Chem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import domalski_hearing_method as dh
from tests.domalski_hearing_paper_examples import (
    RARE_GROUP_EXAMPLES,
    TABLE_EXAMPLES,
)


class PaperValidationTests(unittest.TestCase):
    """Calculated columns in Domalski & Hearing (1993), Tables 4--54."""

    def assert_phase(self, smiles, phase, h, cp, entropy, places=2):
        value = getattr(dh.estimate(smiles), phase)
        self.assertAlmostEqual(value.enthalpy_formation_kJ_mol, h, places=places)
        self.assertAlmostEqual(value.heat_capacity_J_mol_K, cp, places=places)
        self.assertAlmostEqual(value.entropy_J_mol_K, entropy, places=places)

    def test_methane_table_4(self):
        result = dh.estimate("C")
        self.assertEqual(result.groups, {"C-(H)4": 1})
        self.assertEqual(result.symmetry_number, 12)
        self.assert_phase("C", "gas", -74.48, 35.73, 186.26)

    def test_propane_table_4(self):
        result = dh.estimate("CCC")
        self.assertEqual(result.groups, {"C-(H)2(C)2": 1, "C-(H)3(C)": 2})
        self.assertEqual(result.symmetry_number, 18)
        self.assert_phase("CCC", "gas", -105.15, 74.35, 269.77)

    def test_isobutane_table_5_branch_corrections(self):
        result = dh.estimate("CC(C)C")
        self.assertEqual(result.corrections, {"CH3-tertiary": 3})
        self.assertEqual(result.symmetry_number, 81)
        self.assert_phase("CC(C)C", "gas", -134.73, 97.27, 291.82)

    def test_total_symmetry_for_nested_branching_tables_5_and_6(self):
        cases = {
            "CCC(C)C": 27,              # 2-methylbutane
            "CCCC(C)CCC": 54,           # 4-methylheptane
            "CC(C)C(C)C": 162,          # 2,3-dimethylbutane
            "CC(C)(C)C": 972,           # neopentane
        }
        for smiles, expected in cases.items():
            with self.subTest(smiles=smiles):
                self.assertEqual(dh.estimate(smiles).symmetry_number, expected)

    def test_ethylene_table_7(self):
        result = dh.estimate("C=C")
        self.assertEqual(result.groups, {"Cd-(H)2": 2})
        self.assertEqual(result.symmetry_number, 4)
        self.assert_phase("C=C", "gas", 52.64, 42.76, 219.51)

    def test_cis_3_penten_1_yne_table_9_group_identity(self):
        result = dh.estimate("C#C/C=C\\C")
        self.assertEqual(result.groups["Cd-(H)(Cd)"], 1)
        self.assertEqual(result.groups["Cd-(H)(C)"], 1)
        self.assertAlmostEqual(
            result.gas.enthalpy_formation_kJ_mol, 262.11, places=2)
        self.assertAlmostEqual(
            result.gas.heat_capacity_J_mol_K, 88.24, places=2)
        self.assertAlmostEqual(
            result.liquid.enthalpy_formation_kJ_mol, 230.13, places=2)

    def test_benzene_table_10_all_phases(self):
        result = dh.estimate("c1ccccc1")
        self.assertEqual(result.symmetry_number, 12)
        self.assert_phase("c1ccccc1", "gas", 82.86, 81.66, 269.20)
        self.assert_phase("c1ccccc1", "liquid", 48.96, 136.08, 173.22)
        self.assert_phase("c1ccccc1", "solid", 39.18, 120.78, 136.50)

    def test_toluene_table_10(self):
        result = dh.estimate("Cc1ccccc1")
        self.assertEqual(result.symmetry_number, 6)
        self.assert_phase("Cc1ccccc1", "gas", 50.43, 103.53, 318.36)
        self.assert_phase("Cc1ccccc1", "liquid", 12.35, 159.98, 208.15)

    def test_xylene_positional_corrections_and_symmetry_table_10(self):
        cases = {
            "Cc1ccccc1C": ({"ortho-hydrocarbon": 1}, 18, 350.13),
            "Cc1cccc(C)c1": ({"meta-hydrocarbon": 1}, 18, 352.63),
            "Cc1ccc(C)cc1": ({}, 18, 352.63),
        }
        for smiles, (corrections, symmetry, entropy) in cases.items():
            with self.subTest(smiles=smiles):
                result = dh.estimate(smiles)
                self.assertEqual(result.corrections, corrections)
                self.assertEqual(result.symmetry_number, symmetry)
                self.assertAlmostEqual(
                    result.gas.entropy_J_mol_K, entropy, places=2)

    def test_methanol_and_ethanol_table_15(self):
        self.assert_phase("CO", "gas", -201.59, 43.89, 239.69)
        self.assert_phase("CO", "liquid", -239.11, 81.12, 127.19)
        self.assert_phase("CCO", "gas", -234.49, 64.22, 283.12)
        self.assert_phase("CCO", "liquid", -274.91, 114.76, 159.78)

    def test_isopropanol_table_15_corrections_and_symmetry(self):
        result = dh.estimate("CC(C)O")
        self.assertEqual(result.corrections, {"CH3-tertiary": 2})
        self.assertEqual(result.symmetry_number, 18)
        self.assert_phase("CC(C)O", "gas", -274.47, 89.58, 309.06)
        self.assert_phase("CC(C)O", "liquid", -318.68, 167.43, 180.66)

    def test_dimethyl_ether_table_16(self):
        result = dh.estimate("COC")
        self.assertEqual(result.symmetry_number, 18)
        self.assert_phase("COC", "gas", -185.94, 70.00, 259.94)

    def test_acetone_table_18(self):
        self.assert_phase("CC(=O)C", "gas", -217.19, 74.89, 294.92)
        self.assert_phase("CC(=O)C", "liquid", -247.98, 125.93, 200.41)

    def test_acetic_acid_table_19(self):
        self.assert_phase("CC(=O)O", "gas", -433.80, 66.52, 282.49)
        self.assert_phase("CC(=O)O", "liquid", -482.62, 119.28, 154.30)

    def test_methyl_acetate_table_21(self):
        gas = dh.estimate("CC(=O)OC").gas
        self.assertAlmostEqual(gas.enthalpy_formation_kJ_mol, -410.63, places=2)
        self.assertAlmostEqual(gas.heat_capacity_J_mol_K, 87.82, places=2)
        self.assert_phase("CC(=O)OC", "liquid", -440.61, 137.52, 237.60)

    def test_methylamine_table_26(self):
        result = dh.estimate("CN")
        self.assertEqual(result.symmetry_number, 3)
        self.assert_phase("CN", "gas", -23.01, 50.08, 242.59)
        self.assert_phase("CN", "liquid", -47.28, 99.07, 155.01)

    def test_trimethylamine_table_26_quaternary_correction(self):
        result = dh.estimate("CN(C)C")
        self.assertEqual(result.corrections, {"CH3-quaternary": 3})
        self.assertEqual(result.symmetry_number, 81)
        self.assert_phase("CN(C)C", "gas", -23.96, 92.29, 283.71)

    def test_aniline_table_26(self):
        self.assert_phase("Nc1ccccc1", "gas", 87.00, 108.47, 319.16)
        self.assert_phase("Nc1ccccc1", "liquid", 31.30, 191.01, 191.63)

    def test_phenylenediamine_positional_corrections_table_26(self):
        cases = {
            "Nc1ccccc1N": ("NH2-NH2-ortho", -0.58),
            "Nc1cccc(N)c1": ("NH2-NH2-meta", -7.58),
            "Nc1ccc(N)cc1": (None, 2.42),
        }
        for smiles, (correction, solid_h) in cases.items():
            with self.subTest(smiles=smiles):
                result = dh.estimate(smiles)
                if correction is not None:
                    self.assertEqual(result.corrections[correction], 1)
                self.assertAlmostEqual(
                    result.solid.enthalpy_formation_kJ_mol, solid_h, places=2)

    def test_pyridine_table_32(self):
        self.assert_phase("n1ccccc1", "gas", 138.05, 78.12, 282.80)
        self.assert_phase("n1ccccc1", "liquid", 95.30, 133.15, 180.75)

    def test_naphthalene_table_11_fused_aromatic_groups(self):
        result = dh.estimate("c1ccc2ccccc2c1")
        self.assertEqual(
            result.groups,
            {"CB-(H)(CB)2": 8, "CBF-(CBF)(CB)2": 2})
        self.assertEqual(result.corrections, {"naphthalene-unsub": 2})
        self.assert_phase("c1ccc2ccccc2c1", "gas", 150.68, 132.54, 335.63)

    def test_unsubstituted_ring_symmetry_tables_12_and_13(self):
        cases = {
            "C1CC1": (6, 237.44),
            "C1CCC1": (8, 265.39),
            "C1CCCC1": (10, 292.88),
            "C1CCCCC1": (6, 298.24),
            "C1CCCCCC1": (2, 342.33),
            "C1CCCCCCC1": (8, 366.77),
        }
        for smiles, (symmetry, entropy) in cases.items():
            with self.subTest(smiles=smiles):
                result = dh.estimate(smiles)
                self.assertEqual(result.symmetry_number, symmetry)
                self.assertAlmostEqual(
                    result.gas.entropy_J_mol_K, entropy, places=2)

    def test_named_cage_and_cyclophane_calculated_columns_tables_13_14(self):
        cases = {
            "C1C2CC3CC1CC(C2)C3": (-134.60, -197.20),  # adamantane
            "C1CC2=CC(=CC=C2)CCC3=CC=CC1=C3": (170.50, 78.50),
            "C1CC2=CC(=CC=C2)CCC3=CC=C1C=C3": (218.40, 130.90),
            "C1CC2=CC=C(CCC3=CC=C1C=C3)C=C2": (244.77, 146.70),
            "C1CC2=CC=C(CCCC3=CC=C(C1)C=C3)C=C2": (129.37, 26.15),
        }
        for smiles, (gas_h, solid_h) in cases.items():
            with self.subTest(smiles=smiles):
                result = dh.estimate(smiles)
                self.assertAlmostEqual(
                    result.gas.enthalpy_formation_kJ_mol, gas_h, places=2)
                self.assertAlmostEqual(
                    result.solid.enthalpy_formation_kJ_mol, solid_h, places=2)

    def test_rigid_cage_rotational_symmetry_tables_12_and_14(self):
        spiropentane = dh.estimate("C1CC12CC2")
        self.assertEqual(spiropentane.symmetry_number, 4)
        self.assert_phase("C1CC12CC2", "gas", 185.18, 88.12, 282.21)

        cubane = dh.estimate("C12C3C4C1C5C2C3C45")
        self.assertEqual(cubane.symmetry_number, 24)

    def test_nitromethane_table_37(self):
        result = dh.estimate("C[N+](=O)[O-]")
        self.assertEqual(result.groups, {"C-(H)3(NO2)": 1})
        self.assert_phase("C[N+](=O)[O-]", "gas", -74.86, 57.32, 275.01)

    def test_single_source_named_nitrogen_skeletons_tables_32_35(self):
        pyrrolizidine = dh.estimate("CC1CCC2CCC(C)N12")
        self.assertAlmostEqual(
            pyrrolizidine.gas.enthalpy_formation_kJ_mol, -66.70, places=2)
        self.assertAlmostEqual(
            pyrrolizidine.liquid.enthalpy_formation_kJ_mol, -114.40, places=2)

        cyanurate = dh.estimate("CN1C(=O)N(C)C(=O)N(C)C1=O")
        self.assertAlmostEqual(
            cyanurate.gas.enthalpy_formation_kJ_mol, -589.70, places=2)
        self.assertAlmostEqual(
            cyanurate.solid.enthalpy_formation_kJ_mol, -677.92, places=2)

    def test_methanethiol_table_42(self):
        self.assert_phase("CS", "gas", -23.62, 51.49, 255.86)

    def test_dimethyl_sulfide_table_43(self):
        self.assert_phase("CSC", "gas", -37.53, 74.10, 285.80)
        self.assert_phase("CSC", "liquid", -65.40, 118.11, 196.40)

    def test_chloroethane_table_51(self):
        self.assert_phase("CCCl", "gas", -111.71, 63.26, 277.43)
        self.assert_phase("CCCl", "liquid", -134.51, 100.24, 187.57)

    def test_chlorobenzene_table_51(self):
        self.assert_phase("Clc1ccccc1", "gas", 52.02, 97.38, 312.87)
        self.assert_phase("Clc1ccccc1", "liquid", 8.60, 148.67, 199.82)


class ApiAndDomainTests(unittest.TestCase):
    def test_derived_formation_values_follow_paper_equations(self):
        gas = dh.estimate("CCC").gas
        self.assertAlmostEqual(gas.entropy_formation_J_mol_K, -269.74, places=2)
        self.assertAlmostEqual(gas.gibbs_formation_kJ_mol, -24.73, places=2)
        self.assertAlmostEqual(gas.ln_formation_equilibrium_constant, 9.98, places=2)

    def test_optical_isomer_term_from_smiles_structure(self):
        result = dh.estimate("CCC(C)O")
        self.assertEqual(result.optical_isomers, 2)
        self.assertEqual(result.symmetry_number, 9)

    def test_optical_isomer_count_distinguishes_meso_and_chiral_forms(self):
        meso = dh.estimate("C[C@H](O)[C@H](O)C")
        chiral = dh.estimate("C[C@H](O)[C@@H](O)C")
        self.assertEqual(meso.optical_isomers, 1)
        self.assertEqual(chiral.optical_isomers, 2)
        ambiguous = dh.estimate("CC(O)C(C)O")
        self.assertIsNone(ambiguous.optical_isomers)
        self.assertIsNone(ambiguous.gas.entropy_J_mol_K)
        self.assertTrue(any(
            "specify stereochemistry" in warning
            for warning in ambiguous.warnings))

    def test_blank_table_cells_remain_unavailable(self):
        result = dh.estimate("C")
        self.assertIsNone(result.liquid.enthalpy_formation_kJ_mol)
        self.assertIsNone(result.solid.entropy_J_mol_K)
        self.assertTrue(result.warnings)

    def test_groups_with_entirely_blank_published_rows_are_not_invented(self):
        examples = {
            "C-(H)(C)2(SO)": "CC(C)S(=O)C",
            "C-(H)(C)(Br)2": "CC(Br)Br",
            "C-(C)2(Br)2": "CC(Br)(Br)C",
            "C-(C)(I)3": "IC(I)(I)C",
            "C-(C)2(I)2": "CC(I)(I)C",
            "C-(C)(Br)2(F)": "BrC(Br)(F)C",
            "C-(Br)(Cl)(F)": "FC(Cl)Br",
            "Ct-(F)": "C#CF",
            "Cd-(I)2": "C=C(I)I",
        }
        blank_rows = {
            name for name, row in dh.GROUP_VALUES.items()
            if all(value is None for value in vars(row).values())
        }
        self.assertEqual(set(examples), blank_rows)
        for group, smiles in examples.items():
            with self.subTest(group=group, smiles=smiles):
                result = dh.estimate(smiles)
                self.assertIn(group, result.groups)
                for phase in (result.gas, result.liquid, result.solid):
                    self.assertIsNone(phase.enthalpy_formation_kJ_mol)
                    self.assertIsNone(phase.heat_capacity_J_mol_K)
                    self.assertIsNone(phase.entropy_J_mol_K)

    def test_four_previously_omitted_table_2_rows(self):
        expected = {
            "C-(C)2(CB)2": (None, None, None, None, None, None,
                              52.81, None, None),
            "C-(C)(CB)3": (None, None, None, None, None, None,
                             116.25, 39.83, None),
            "O-(H)(Cd)": (None, None, None, None, 37.78, None,
                            None, None, None),
            "Cd-(I)2": (None,) * 9,
        }
        for name, cells in expected.items():
            with self.subTest(name=name):
                self.assertEqual(tuple(vars(dh.GROUP_VALUES[name]).values()), cells)

        hexaphenylethane = (
            "C(c1ccccc1)(c2ccccc2)(c3ccccc3)"
            "C(c4ccccc4)(c5ccccc5)c6ccccc6")
        self.assertAlmostEqual(
            dh.estimate(hexaphenylethane).solid.enthalpy_formation_kJ_mol,
            511.80, places=2)
        self.assertIn(
            "C-(C)2(CB)2",
            dh.fragment("CC(C)(c1ccccc1)c1ccccc1").groups)
        self.assertAlmostEqual(
            dh.estimate("C=CO").liquid.heat_capacity_J_mol_K,
            90.75, places=2)

    def test_invalid_and_out_of_domain_structures_are_rejected(self):
        with self.assertRaises(dh.DomalskiHearingError):
            dh.estimate("not smiles")
        with self.assertRaises(dh.FragmentationError):
            dh.estimate("C.C")
        with self.assertRaises(dh.FragmentationError):
            dh.estimate("C[SiH3]")
        with self.assertRaises(dh.FragmentationError):
            dh.estimate("C[NH3+]")

    def test_hf_applicability_keeps_corrected_small_rings(self):
        propane = dh.assess_hf_applicability(dh.estimate("CCC"))
        cyclopropane = dh.assess_hf_applicability(dh.estimate("C1CC1"))

        self.assertTrue(propane.applicable)
        self.assertEqual(propane.reasons, ())
        self.assertTrue(cyclopropane.applicable)
        self.assertEqual(cyclopropane.reasons, ())

    def test_hf_applicability_rejects_complex_rings_and_ss_bonds(self):
        cubane = dh.assess_hf_applicability(
            dh.estimate("C12C3C4C1C5C2C3C45")
        )
        disulfide = dh.assess_hf_applicability(dh.estimate("CSSC"))

        self.assertFalse(cubane.applicable)
        self.assertIn("four or more rings", cubane.reasons)
        self.assertFalse(disulfide.applicable)
        self.assertEqual(disulfide.reasons, ("sulfur-sulfur bond",))

    def test_published_nonring_second_order_corrections(self):
        cases = {
            "Cl/C=C\\Cl": "cis-Cl-Cl",
            "I/C=C\\I": "cis-I-I",
            "C/C=C\\Br": "cis-CH3-Br",
            "C/C=C\\I": "cis-CH3-I",
            "Cc1ncccc1": "Nr-CH3-ortho",
            "n1ncccc1": "Nr-Nr-ortho",
            "O=[N+]([O-])CC([N+](=O)[O-])":
                "NO2-NO2-aliphatic-adjacent",
            "O=[N+]([O-])OCCO[N+](=O)[O-]":
                "ONO2-ONO2-aliphatic-adjacent",
            "Fc1ccccc1-c1ccccc1F": "ortho-F-F-prime",
            "Clc1ccccc1-c1ccccc1Cl": "ortho-Cl-Cl-prime",
            "O=C(Cl)c1ccccc1C(=O)Cl": "ortho-COCl-COCl",
            "O=C(Cl)c1cccc(C(=O)Cl)c1": "meta-COCl-COCl",
        }
        for smiles, correction in cases.items():
            with self.subTest(smiles=smiles, correction=correction):
                self.assertEqual(dh.fragment(smiles).corrections[correction], 1)


class ExhaustivePublishedRowCoverageTests(unittest.TestCase):
    """Every Table-2 group/correction row is exercised at least once.

    The hashes make the expected complete inventories independent of the
    production dictionaries: adding, removing, renaming, or changing any cell
    requires an intentional test update.  The second test then passes every
    single row through the same summation path used by ``estimate``.
    """

    EXPECTED_GROUP_COUNT = 347
    EXPECTED_CORRECTION_COUNT = 205
    EXPECTED_AMBIGUOUS_CORRECTION_COUNT = 2
    EXPECTED_GROUP_SHA256 = (
        "d467d9230364b510c2af3ce97baf87bde7dd16c5d32598b512de6a05afcd920e"
    )
    EXPECTED_CORRECTION_SHA256 = (
        "925416925752cc5a3623a26d331fc070054228ae9b3a4919b681e6b3811ae36b"
    )
    EXPECTED_AMBIGUOUS_CORRECTION_SHA256 = (
        "67438abdeddb943330acf2b2a93e02d8e7cfeca9d7929a1f38a0b91dc8902043"
    )
    # Appendix 1 explicitly names the source compound for group rows fitted
    # from only one compound.  These are the most important rows to protect
    # with an independent molecular example because no homologous regression
    # can cross-check them.
    APPENDIX_1_SOURCE_EXAMPLES = {
        "C-(H)(C)2(Ct)": "CC(C)C#C",                 # 3-methyl-1-butyne
        "C-(C)2(Ct)2": "CC(C)(C#C)C#C",             # 3,3-dimethylpenta-1,4-diyne
        "Cd-(C)(CB)": "CC(=C)c1ccccc1",              # alpha-methylstyrene
        "C-(H)2(Cd)(CB)": "C=CCc1ccccc1",            # 2-propenylbenzene
        "C-(H)(C)(Cd)(CB)": "CC(C=C)c1ccccc1",
        "C-(O)3(C)": "CC(OC)(OC)OC",                 # 1,1,1-trimethoxyethane
        "CO-(H)(CO)": "O=CC=O",                      # glyoxal
        "CO-(H)(Cd)": "C/C=C/C=O",                   # trans-2-butenal
        "Cd-(H)(CO)": "C/C=C/C=O",
        "CO-(H)(CB)": "O=Cc1ccccc1",                 # benzaldehyde
        "CO-(CO)(CB)": "O=C(c1ccccc1)C(=O)c1ccccc1", # benzil
        "CO-(C)(CO)": "CC(=O)C(C)=O",                # biacetyl
        "C-(C)2(CN)2": "CC(C)(C#N)C#N",
        "C-(C)3(CN)": "CC(C)(C)C#N",
        "C-(CB)3(N3)": "[N-]=[N+]=NC(c1ccccc1)(c1ccccc1)c1ccccc1",
        "C-(H)(C)2(NA)": "CC(C)N=NC(C)C",
        "CB-(CNO)(CB)2": "N#Cc1ccc(C#[N+][O-])cc1",
        "C-(H)2(CB)(NO2)": "[O-][N+](=O)Cc1ccccc1",
        "S-(CB)(H)": "Sc1ccccc1",
        "C-(H)2(CB)(S)": "SCc1ccccc1",
        "S-(CB)2": "c1ccc(Sc2ccccc2)cc1",
        "S-(CB)(S)": "c1ccc(SSc2ccccc2)cc1",
        "C-(C)3(SO)": "CCS(=O)C(C)(C)C",
        "SO2-(Cd)2": "C=CS(=O)(=O)C=C",
        "SO2-(CB)2": "O=S(=O)(c1ccccc1)c1ccccc1",
        "SO2-(SO2)(CB)": "O=S(=O)(c1ccccc1)S(=O)(=O)c1ccccc1",
        "CO-(C)(F)": "CC(=O)F",
        "Ct-(Cl)": "CC#CCl",
        "C-(H)2(CB)(Cl)": "ClCc1ccccc1",
        "CO-(C)(Cl)": "CC(=O)Cl",
        "CO-(CB)(Cl)": "O=C(Cl)c1ccccc1",
        "Ct-(Br)": "CC#CBr",
        "C-(H)2(CB)(Br)": "BrCc1ccccc1",
        "CO-(C)(Br)": "CC(=O)Br",
        "C-(C)3(I)": "CC(C)(C)I",
        "Ct-(I)": "CC#CI",
        "C-(H)2(CB)(I)": "ICc1ccccc1",
        "CO-(C)(I)": "CC(=O)I",
        "C-(H)(C)(Cl)(F)": "CC(F)Cl",
        "C-(H)(C)(Br)(Cl)": "BrC(Cl)C(Cl)Br",
        "C-(C)(Br)(F)2": "BrC(F)(F)C(F)(F)Br",
        "Cd-(Cl)(F)": "FC(F)=C(F)Cl",
    }

    @staticmethod
    def table_digest(table):
        payload = json.dumps(
            {name: list(vars(row).values()) for name, row in sorted(table.items())},
            separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def test_complete_published_parameter_inventory(self):
        self.assertEqual(len(dh.GROUP_VALUES), self.EXPECTED_GROUP_COUNT)
        self.assertEqual(
            self.table_digest(dh.GROUP_VALUES), self.EXPECTED_GROUP_SHA256)
        self.assertEqual(
            len(dh.CORRECTION_VALUES), self.EXPECTED_CORRECTION_COUNT)
        self.assertEqual(
            self.table_digest(dh.CORRECTION_VALUES),
            self.EXPECTED_CORRECTION_SHA256)
        self.assertEqual(
            len(dh.AMBIGUOUS_PAPER_CORRECTION_VALUES),
            self.EXPECTED_AMBIGUOUS_CORRECTION_COUNT)
        self.assertEqual(
            self.table_digest(dh.AMBIGUOUS_PAPER_CORRECTION_VALUES),
            self.EXPECTED_AMBIGUOUS_CORRECTION_SHA256)

    def test_conflicting_p826_nh2_rows_are_preserved_but_not_operational(self):
        self.assertEqual(
            set(dh.AMBIGUOUS_PAPER_CORRECTION_VALUES),
            {"p826-ortho-NH2-NH2", "p826-meta-NH2-NH2"})
        self.assertNotIn("ortho-NH2-NH2", dh.CORRECTION_VALUES)
        self.assertNotIn("meta-NH2-NH2", dh.CORRECTION_VALUES)
        self.assertEqual(
            dh.fragment("Nc1ccccc1N").corrections,
            Counter({"NH2-NH2-ortho": 1}))

    def test_every_group_and_correction_runs_through_property_summation(self):
        mol = Chem.MolFromSmiles("C")
        phases = ("gas", "liquid", "solid")
        for name, row in dh.GROUP_VALUES.items():
            with self.subTest(kind="group", name=name):
                frag = dh.Fragmentation(
                    smiles="C", mol=mol, groups=Counter({name: 1}),
                    symmetry_number=1, optical_isomers=1)
                for phase in phases:
                    for cell, expected in enumerate(row.phase(phase)):
                        self.assertEqual(
                            dh._sum_property(frag, phase, cell), expected)
        for name, row in dh.CORRECTION_VALUES.items():
            with self.subTest(kind="correction", name=name):
                frag = dh.Fragmentation(
                    smiles="C", mol=mol, corrections=Counter({name: 1}),
                    symmetry_number=1, optical_isomers=1)
                for phase in phases:
                    for cell, expected in enumerate(row.phase(phase)):
                        self.assertEqual(
                            dh._sum_property(frag, phase, cell), expected)

    def test_every_named_skeleton_fixture_selects_its_published_correction(self):
        exercised = set()
        for smiles, correction in dh._NAMED_CORRECTION_SMILES.items():
            with self.subTest(smiles=smiles, correction=correction):
                fragmentation = dh.fragment(smiles)
                self.assertGreater(fragmentation.corrections[correction], 0)
                exercised.add(correction)
        self.assertEqual(exercised, set(dh._NAMED_STRUCTURE_CORRECTIONS.values()))

    def test_appendix_1_single_compound_groups_use_the_named_source_structure(self):
        for group, smiles in self.APPENDIX_1_SOURCE_EXAMPLES.items():
            with self.subTest(group=group, smiles=smiles):
                self.assertIn(group, dh.fragment(smiles).groups)

    def test_every_fragment_group_has_a_paper_named_representative(self):
        covered = set()
        for smiles, name, page in TABLE_EXAMPLES:
            with self.subTest(name=name, page=page):
                covered.update(dh.fragment(smiles).groups)
        for group, smiles in self.APPENDIX_1_SOURCE_EXAMPLES.items():
            covered.update(dh.fragment(smiles).groups)
        for smiles in dh._NAMED_CORRECTION_SMILES:
            covered.update(dh.fragment(smiles).groups)
        for group, smiles in RARE_GROUP_EXAMPLES.items():
            with self.subTest(group=group, smiles=smiles):
                fragmentation = dh.fragment(smiles)
                self.assertIn(group, fragmentation.groups)
                covered.update(fragmentation.groups)
        self.assertEqual(covered, set(dh.GROUP_VALUES))


if __name__ == "__main__":
    unittest.main()
