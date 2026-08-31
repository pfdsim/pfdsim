import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nannoolal_method import (
    FragmentationError,
    acentric_factor,
    estimate,
    estimate_psat,
    estimate_viscosity,
    fragment,
)
from rdkit import Chem


class WorkedExampleTests(unittest.TestCase):
    """The eight numeric worked examples published in the two papers.

    Part 1 (Fluid Phase Equilib. 226 (2004) 45-63), Tables 17a-d and
    Part 2 (Fluid Phase Equilib. 252 (2007) 1-27), Tables 42a-d.
    These must reproduce essentially exactly: they validate both the
    fragmentation and the transcribed contribution tables.
    """

    def test_tb_tetramethylhexane_table_17a(self):
        r = estimate('CCC(C)(C)C(C)(C)CC')
        self.assertAlmostEqual(r.tb_K, 429.5, delta=0.2)
        self.assertEqual(r.groups, {1: 6, 4: 2, 6: 2})
        self.assertEqual(r.corrections, {133: 1})

    def test_tb_diisopropanolamine_table_17b(self):
        r = estimate('CC(O)CNCC(C)O')
        self.assertAlmostEqual(r.tb_K, 509.3, delta=0.2)
        self.assertEqual(r.groups, {1: 2, 7: 4, 34: 2, 42: 1})
        self.assertEqual(sorted(r.interaction_classes), ['A', 'A', 'N'])

    def test_tb_perfluoroacetone_table_17c(self):
        r = estimate('FC(F)(F)C(=O)C(F)(F)F')
        self.assertAlmostEqual(r.tb_K, 246.3, delta=0.2)
        self.assertEqual(r.groups, {7: 2, 21: 6, 51: 1})
        self.assertEqual(r.corrections, {120: 1, 121: 2, 123: 1})

    def test_tb_methyl_m_toluate_table_17d(self):
        r = estimate('COC(=O)c1cccc(C)c1')
        self.assertAlmostEqual(r.tb_K, 490.6, delta=0.2)
        self.assertEqual(r.groups, {2: 1, 3: 1, 15: 4, 16: 2, 45: 1})
        self.assertEqual(r.corrections, {128: 1, 134: 1})

    def test_tc_tetramethylbutane_table_42a(self):
        r = estimate('CC(C)(C)C(C)(C)C', tb=379.6)
        self.assertAlmostEqual(r.tc_K, 566.8, delta=0.3)

    def test_pc_diethylene_glycol_monomethyl_ether_table_42b(self):
        r = estimate('COCCOCCO')
        self.assertAlmostEqual(r.pc_kPa, 3689.9, delta=2.0)
        self.assertEqual(r.groups[35], 1)   # primary OH, 5 C across the ethers
        self.assertEqual(r.groups[38], 2)   # two ethers

    def test_vc_trichlorosilane_table_42c(self):
        r = estimate('Cl[SiH](Cl)Cl')
        self.assertAlmostEqual(r.vc_cm3_mol, 262.9, delta=0.3)
        self.assertEqual(r.groups, {27: 3})           # Si handled separately
        self.assertEqual(r.corrections, {124: 1, 217: 1})

    def test_tc_perfluoroacetone_table_42d(self):
        r = estimate('FC(F)(F)C(=O)C(F)(F)F', tb=245.9)
        self.assertAlmostEqual(r.tc_K, 358.0, delta=0.3)


class FragmentationTests(unittest.TestCase):
    def _groups(self, smiles):
        return dict(fragment(Chem.MolFromSmiles(smiles)).groups)

    def test_paper_documented_small_molecule_deviation(self):
        # The paper reports ethanol at about -18.4 K; reproducing that
        # deviation confirms group assignments 1/7/36.
        r = estimate('CCO')
        self.assertEqual(r.groups, {1: 1, 7: 1, 36: 1})
        self.assertAlmostEqual(r.tb_K - 351.4, -18.4, delta=1.0)

    def test_alcohol_differentiation(self):
        self.assertIn(36, self._groups('CO'))            # methanol: short chain
        self.assertIn(34, self._groups('CC(O)C'))        # secondary
        self.assertIn(33, self._groups('CC(C)(C)O'))     # tertiary
        self.assertIn(37, self._groups('Oc1ccccc1'))     # phenol
        self.assertIn(35, self._groups('CCCCCO'))        # primary long chain

    def test_aromatics_and_rings(self):
        self.assertEqual(self._groups('c1ccccc1'), {15: 6})
        self.assertEqual(self._groups('c1ccc2ccccc2c1'), {15: 8, 18: 2})
        self.assertEqual(self._groups('c1ccc(-c2ccccc2)cc1'), {15: 10, 214: 2})
        self.assertEqual(self._groups('C1CCCCC1'), {9: 6})

    def test_ortho_meta_para_exclusive_rule(self):
        frag = fragment(Chem.MolFromSmiles('Cc1ccccc1C'))    # o-xylene
        self.assertEqual(frag.corrections.get(127), 1)
        frag = fragment(Chem.MolFromSmiles('Cc1ccc(C)cc1'))  # p-xylene
        self.assertEqual(frag.corrections.get(129), 1)
        # 1,2,3-trimethylbenzene has ortho and meta pairs -> no correction
        frag = fragment(Chem.MolFromSmiles('Cc1cccc(C)c1C'))
        for cid in (127, 128, 129):
            self.assertNotIn(cid, frag.corrections)

    def test_aromatic_carbonyl_ring_demotion(self):
        # Coumarin: RDKit calls the pyranone ring aromatic; the fragmenter
        # must still find the lactone (group 47).
        groups = self._groups('O=c1ccc2ccccc2o1')
        self.assertIn(47, groups)
        self.assertIn(62, groups)

    def test_cyclic_sp2_anhydride_claims_ring(self):
        self.assertEqual(self._groups('O=C1OC(=O)C=C1'), {96: 1})
        r = estimate('O=C1OC(=O)C=C1')
        self.assertAlmostEqual(r.tb_K, 475.1, delta=8.0)  # maleic anhydride

    def test_saturated_cyclic_anhydride_extension_group(self):
        # Succinic-type anhydrides use the local extension group 219 (fitted
        # in this work) and carry a warning; acetic anhydride stays group 76.
        self.assertEqual(self._groups('O=C1CCC(=O)O1'), {219: 1, 9: 2})
        r = estimate('O=C1CCC(=O)O1')
        self.assertAlmostEqual(r.tb_K, 534.1, delta=5.0)   # succinic anhydride
        self.assertTrue(any('group 219' in w for w in r.warnings))
        self.assertIsNone(r.tc_K)
        self.assertIn(76, self._groups('CC(=O)OC(C)=O'))
        r = estimate('CC(=O)OC(C)=O')                      # acetic anhydride
        self.assertFalse(any('group 219' in w for w in r.warnings))

    def test_nitro_groups(self):
        groups = self._groups('O=[N+]([O-])c1ccccc1')
        self.assertIn(69, groups)
        r = estimate('O=[N+]([O-])c1ccccc1', tb=483.85)
        self.assertAlmostEqual(r.tb_K, 483.9, delta=8.0)
        self.assertAlmostEqual(r.tc_K, 718.0, delta=5.0)
        self.assertAlmostEqual(r.pc_kPa, 4160.0, delta=4160.0 * 0.04)
        self.assertAlmostEqual(r.vc_cm3_mol, 335.0, delta=335.0 * 0.04)
        self.assertTrue(any('locally fitted extension' in w for w in r.warnings))

        # Faithful published mode keeps group 69 criticals unavailable.
        published = estimate(
            'O=[N+]([O-])c1ccccc1', tb=483.85, local_refits=False,
        )
        self.assertIsNone(published.tc_K)
        self.assertIsNone(published.pc_kPa)
        self.assertIsNone(published.vc_cm3_mol)

    def test_aromatic_nitro_amine_and_hydroxyl_criticals_are_refused(self):
        cases = {
            'aromatic primary amine (group 41-Q)': 'Nc1ccccc1[N+](=O)[O-]',
            'phenolic hydroxyl (group 37-Q)': 'Oc1ccccc1[N+](=O)[O-]',
        }
        for expected_warning, smiles in cases.items():
            with self.subTest(expected_warning=expected_warning):
                result = estimate(smiles)
                self.assertIsNotNone(result.tb_K)
                self.assertIsNone(result.tc_K)
                self.assertIsNone(result.pc_kPa)
                self.assertIsNone(result.vc_cm3_mol)
                self.assertTrue(any(
                    expected_warning in warning for warning in result.warnings
                ))
                # Critical refusal must not discard the separately published
                # group-69 vapor-pressure model.
                psat = estimate_psat(smiles)
                self.assertIsNotNone(psat.db)
                self.assertIsNotNone(psat.tb_K)

        # Remote/aliphatic NH2 and OH environments are not evidence for the
        # strongly conjugated nitroaniline/nitrophenol refusal.
        for smiles in (
            'NCCc1ccc([N+](=O)[O-])cc1',
            'OCCc1ccc([N+](=O)[O-])cc1',
        ):
            with self.subTest(remote_function=smiles):
                result = estimate(smiles)
                self.assertIsNotNone(result.tc_K)
                self.assertIsNotNone(result.pc_kPa)
                self.assertIsNotNone(result.vc_cm3_mol)

    def test_diacid_interaction_extension(self):
        # COOH-COOH Tc/Pc fitted to pulse-heating diacid data (local
        # extension; data/reference/IMG_3694.jpeg).  Adipic acid:
        # Tb 610.5 K, Tc 841 K, Pc 3.85 MPa.
        r = estimate('OC(=O)CCCCC(=O)O', tb=610.5)
        self.assertEqual(sorted(r.interaction_classes), ['C', 'C'])
        self.assertAlmostEqual(r.tc_K, 841.0, delta=20.0)
        self.assertAlmostEqual(r.pc_kPa, 3850.0, delta=3850 * 0.04)
        self.assertTrue(any('COOH-COOH' in w for w in r.warnings))
        # mono-acids must not be touched by the extension
        r = estimate('CCCC(=O)O')
        self.assertFalse(any('COOH-COOH' in w for w in r.warnings))

    def test_diol_subtype_split(self):
        # OH-OH Tc split (local extension, keyed on OH group subtype after
        # the 35/36 chain-rule fix): 36+36 pairs use the PUBLISHED constant
        # (MEG/1,3-PD/1,4-BD all within +-7 K); 34-involved pairs use -252
        # (fit: 1,2-PD + 1,3-BD); 35-involved pairs are additive (C5-C10
        # terminal diols).
        r = estimate('OCCO', tb=470.5)                    # ethylene glycol
        self.assertAlmostEqual(r.tc_K, 720.0, delta=10.0)
        self.assertTrue(any('subtype' in w for w in r.warnings))
        r = estimate('OCCCO', tb=487.6)                   # 1,3-propanediol
        self.assertAlmostEqual(r.tc_K, 720.0, delta=10.0)
        r = estimate('OCCCCCCO', tb=523.2)                # 1,6-hexanediol
        self.assertAlmostEqual(r.tc_K, 740.0, delta=10.0)
        r = estimate('CC(O)CO', tb=460.8)                 # 1,2-propanediol
        self.assertAlmostEqual(r.tc_K, 676.4, delta=5.0)
        # 1,2-BD holdout: -20 K vs the (shakiest-source) 694 K datum; the
        # documented weak point of the split
        r = estimate('CCC(O)CO', tb=469.2)
        self.assertAlmostEqual(r.tc_K, 694.0, delta=25.0)
        # glycerol (36+36, 34+36 x2): handled per pair; no experimental Tc
        # exists (decomposes), so only a sanity window is asserted
        r = estimate('OCC(O)CO', tb=563.2)
        self.assertAlmostEqual(r.tc_K, 810.0, delta=50.0)
        # faithful mode: published single constant for all pairs
        r_pub = estimate('OCCCCCCO', tb=523.2, local_refits=False)
        self.assertGreater(r_pub.tc_K, 760.0)             # ~+29 K bias here

    def test_glycol_ethers_pairwise(self):
        # Glycol ethers under the published pairwise A-D / D-D interactions
        # (DEG's ATN Table-1 misprint replaced by Nikitin's 753 +- 8 K, see
        # scripts/extract_acs_jced_5b00571_table1.py).  Post chain-rule fix,
        # most of the family is reproduced well; DEG (-27 K) and TEG (-22 K)
        # are known misses of the published parameter set (documented in the
        # module, deliberately not patched locally).
        r = estimate('COCCO', tb=397.5)                   # 2-methoxyethanol
        self.assertAlmostEqual(r.tc_K, 579.6, delta=15.0)
        r = estimate('COCC(C)O', tb=393.2)                # 1-methoxy-2-propanol
        self.assertAlmostEqual(r.tc_K, 579.8, delta=10.0)
        r = estimate('CCOCCOCCO', tb=475.2)               # DEGEE
        self.assertAlmostEqual(r.tc_K, 670.0, delta=10.0)
        r = estimate('OCCOCCOCCOCCO', tb=600.6)           # tetraEG
        self.assertAlmostEqual(r.tc_K, 800.0, delta=10.0)
        r = estimate('OCCOCCO', tb=518.0)                 # DEG: known -27 K
        self.assertAlmostEqual(r.tc_K, 726.0, delta=10.0)
        r = estimate('OCCOCCOCCO', tb=559.0)              # TEG: known -22 K
        self.assertAlmostEqual(r.tc_K, 775.0, delta=10.0)

    def test_local_refits_layer(self):
        # Refitted values (groups 44/53 Pc, 37 Vc) are on by default, warned
        # about, and fully reversible via local_refits=False.
        r = estimate('CCCCCCCC(=O)O')                    # octanoic acid
        self.assertAlmostEqual(r.pc_kPa, 2870.0, delta=2870 * 0.05)
        self.assertTrue(any('refitted' in w for w in r.warnings))
        r_pub = estimate('CCCCCCCC(=O)O', local_refits=False)
        self.assertLess(r_pub.pc_kPa, r.pc_kPa)          # published bias ~-6 %
        self.assertFalse(any('refitted' in w for w in r_pub.warnings))

        r = estimate('CCCCCCCCCCCCS')                    # 1-dodecanethiol (ATN-12)
        self.assertAlmostEqual(r.pc_kPa, 1810.0, delta=1810 * 0.06)

        r = estimate('Oc1ccccc1')                        # phenol Vc
        self.assertAlmostEqual(r.vc_cm3_mol, 229.0, delta=229 * 0.04)
        r_pub = estimate('Oc1ccccc1', local_refits=False)
        self.assertGreater(r_pub.vc_cm3_mol, 275.0)      # published +25 % bias

        # published Tb path is never touched by the refits
        self.assertAlmostEqual(r.tb_K, estimate('Oc1ccccc1',
                                                local_refits=False).tb_K)

    def test_do_not_estimate_cases(self):
        r = estimate('CC(=O)C(C)=O')          # 1,2-diketone (group 118)
        self.assertIsNone(r.tb_K)
        self.assertTrue(r.warnings)
        r = estimate('NCCC(=O)O')             # amino acid: zwitterion (218)
        self.assertIsNone(r.tb_K)

    def test_out_of_scope_molecules_raise(self):
        for smiles in ('O', 'C', 'O=C=O', 'NN', 'C=O', 'OC=O'):
            with self.assertRaises(FragmentationError):
                estimate(smiles)


class LiteratureToleranceTests(unittest.TestCase):
    """Coarse checks against experimental data (DIPPR/CRC rounded values).

    Tolerances reflect the method's published accuracy (Tb AAD 6.5 K,
    Tc AAD 4.3 K with experimental Tb, Pc ~2-4 %, Vc ~2-4 %).
    """

    CASES = [
        # smiles, exp Tb K, exp Tc K, exp Pc kPa, exp Vc cm3/mol
        ('CCCCCCCCCC', 447.3, 617.7, 2110, 617),      # n-decane
        ('Cc1ccccc1', 383.8, 591.8, 4108, 316),       # toluene
        ('CCCCCCCCO', 468.3, 652.5, 2777, 490),       # 1-octanol
        ('CCOC(C)=O', 350.2, 523.3, 3880, 286),       # ethyl acetate
        ('CCC(C)=O', 352.7, 535.5, 4150, 267),        # 2-butanone
        ('Clc1ccccc1', 404.9, 632.4, 4520, 308),      # chlorobenzene
        ('c1ccncc1', 388.4, 620.0, 5670, 243),        # pyridine
        ('C1CCOC1', 339.1, 540.2, 5190, 224),         # THF
    ]

    def test_common_compounds(self):
        for smiles, tb, tc, pc, vc in self.CASES:
            with self.subTest(smiles=smiles):
                r = estimate(smiles, tb=tb)
                self.assertAlmostEqual(r.tb_K, tb, delta=15.0)
                self.assertAlmostEqual(r.tc_K, tc, delta=15.0)
                self.assertLess(abs(r.pc_kPa - pc) / pc, 0.12)
                self.assertLess(abs(r.vc_cm3_mol - vc) / vc, 0.11)


class PsatWorkedExampleTests(unittest.TestCase):
    """The eight numeric worked examples of Part 3 (Tables 15a-15h).

    Part 3 (Fluid Phase Equilib. 269 (2008) 117-133).  dB is checked to the
    printed precision; pressures to the printed kPa values.  These pin both
    the dB tables and the (shared) fragmentation -- 15e/15g are the examples
    that fix the 35/36 alcohol chain rule to carbon counting.
    """

    # smiles, expected groups, corrections, dB, Tb [K], T [K], Ps calc [kPa]
    CASES = [
        ('CC1=CCC2CC1C2(C)C',                        # 15a alpha-pinene
         {1: 3, 9: 2, 10: 2, 11: 1, 62: 1}, {125: 1, 132: 2},
         0.1078596, 429.00, 388.15, 31.03),
        ('OCCO',                                     # 15b 1,2-ethanediol
         {7: 2, 36: 2}, {},
         1.1310491, 470.50, 410.65, 13.05),
        ('O=C(C(F)(F)F)C(F)(F)F',                    # 15c perfluoro-2-propanone
         {7: 2, 21: 6, 51: 1}, {120: 1, 121: 2, 123: 1},
         0.1179392, 245.90, 210.16, 14.63),
        ('C=CC(=O)O',                                # 15d acrylic acid
         {44: 1, 61: 1}, {134: 1},
         0.9163297, 413.60, 344.15, 6.52),
        ('CC(=O)OCCO',                               # 15e glycol monoacetate
         {1: 1, 7: 2, 36: 1, 45: 1}, {},
         0.5111422, 458.65, 352.65, 2.24),
        ('ClC(Cl)C(F)(F)Cl',                         # 15h R122
         {7: 2, 21: 2, 26: 2, 27: 1}, {121: 1, 124: 1},
         0.0447374, 344.25, 297.46, 17.5093),
    ]

    def test_worked_examples(self):
        for smiles, groups, corrections, db, tb, t, ps in self.CASES:
            with self.subTest(smiles=smiles):
                r = estimate_psat(smiles, tb=tb)
                self.assertEqual(r.groups, groups)
                self.assertEqual(r.corrections, corrections)
                self.assertAlmostEqual(r.db, db, delta=5e-7)
                self.assertAlmostEqual(r.psat_kPa(t), ps, delta=0.011)
                # inversion round-trips through the same closed form
                self.assertAlmostEqual(r.temperature_K(r.psat_kPa(t)), t,
                                       delta=1e-6)

    def test_dipropyl_succinate_table_15f(self):
        # dB and Tb back-calculation from a single low-pressure point
        # (0.4 kPa at 374.65 K -> 520.85 K; experimental Tb 523.95 K).
        r = estimate_psat('CCCOC(=O)CCC(=O)OCCC', psat_point=(374.65, 0.4))
        self.assertAlmostEqual(r.db, 0.9878298, delta=5e-7)
        self.assertAlmostEqual(r.tb_K, 520.85, delta=0.02)
        self.assertEqual(r.tb_source, 'from psat point')

    def test_diethanolamine_table_15g(self):
        # The paper's printed dB (1.7493490) contradicts its own eq. (8):
        # it ADDS the (negative) interaction sum.  Consistent application
        # -- as in examples 15b/15e/15f -- gives 1.6131919, which also lies
        # closer to the experimental 0.410 kPa (0.404 vs 0.354 calculated).
        # This module follows the equation, not the misprint.
        r = estimate_psat('OCCNCCO', tb=541.15)
        self.assertEqual(r.groups, {7: 4, 36: 2, 42: 1})
        self.assertAlmostEqual(r.db, 1.6131919, delta=5e-7)
        self.assertAlmostEqual(r.psat_kPa(401.13), 0.404, delta=0.001)


class PsatBehaviorTests(unittest.TestCase):

    def test_monofunctional_alcohol_correction(self):
        # eq. (9) of Part 3 is misprinted; the real form is eq. (8-6) of the
        # 2006 thesis (f(T) = a[2/(1+exp((201/(T-91))^7)) - 1], a = 0.37704),
        # applied to mono-functional alcohols only.
        r = estimate_psat('CCO', tb=351.44)                # ethanol
        self.assertTrue(r.alcohol_correction)
        self.assertTrue(any('eq. (9)' in w for w in r.warnings))
        # 298.15 K: 7.87 kPa experimental; uncorrected model gives 10.3
        self.assertAlmostEqual(r.psat_kPa(298.15), 7.87, delta=0.6)
        # numeric inversion must round-trip through the corrected curve
        p = r.psat_kPa(298.15)
        self.assertAlmostEqual(r.temperature_K(p), 298.15, delta=1e-5)
        # multi-functional compounds must not carry the correction
        r = estimate_psat('OCCO', tb=470.5)                # ethanediol
        self.assertFalse(r.alcohol_correction)
        self.assertFalse(any('eq. (9)' in w for w in r.warnings))

    def test_group_without_db_value(self):
        # group 109 (thioester) has no dB in Table 4 of Part 3
        r = estimate_psat('CC(=O)SC', tb=369.8)
        self.assertIsNone(r.db)
        self.assertTrue(any('group 109' in w for w in r.warnings))

    def test_dhvap_sanity(self):
        # n-heptane at Tb: 31.77 kJ/mol experimental; dZvap ~0.93 there
        r = estimate_psat('CCCCCCC', tb=371.57)
        dh = r.dhvap_J_mol(371.57, dz_vap=0.93)
        self.assertAlmostEqual(dh, 31770.0, delta=1500.0)

    def test_estimated_tb_anchor_warns(self):
        r = estimate_psat('CCCCCC')
        self.assertEqual(r.tb_source, 'estimated')
        self.assertTrue(any('anchored at the internally estimated'
                            in w for w in r.warnings))

    def test_acentric_factor(self):
        # omega = -log10(Ps(0.7 Tc)/Pc) - 1; with experimental anchors the
        # normal fluids land within ~+-0.006 of Perry 2-106
        for smi, tb, tc, pc, w_exp, tol in [
                ('CCCCCCC', 371.6, 540.2, 2740, 0.3495, 0.01),   # n-heptane
                ('CCCCCCCCCC', 447.3, 617.7, 2110, 0.4923, 0.01),
                ('CCOC(C)=O', 350.2, 523.3, 3880, 0.3664, 0.01),
                ('CC(=O)O', 391.0, 591.9, 5786, 0.4665, 0.05)]:  # dimerizes
            w = acentric_factor(smi, tb=tb, tc=tc, pc_kPa=pc)
            self.assertAlmostEqual(w, w_exp, delta=tol)
        # fully predictive mode still returns something sane
        w = acentric_factor('Cc1ccccc1')
        self.assertAlmostEqual(w, 0.264, delta=0.05)


class ViscosityWorkedExampleTests(unittest.TestCase):
    """Tables 28a-28c of Part 4 (FPE 281 (2009) 97-119).

    The examples' own arithmetic uses the truncated constant 3.777 in 28a
    and 28b but the full published 3.7777 in 28c; the module follows the
    published constant, so 28a/28b dBv values sit exactly 0.0007 above the
    printed ones.  28c's printed GI line also carries a sign typo (the
    subtraction shown cannot reproduce its own printed result; S + GI does,
    to 7 digits)."""

    def test_28a_diethylamine(self):
        r = estimate_viscosity('CCNCC', tb=329.0)
        self.assertAlmostEqual(r.dbv, 4.7713112 + 0.0007, places=6)
        self.assertAlmostEqual(r.tv_K, 210.38, delta=0.02)
        self.assertAlmostEqual(r.viscosity_mPa_s(308.15), 0.2674, delta=0.001)

    def test_28b_glycol_ether_interaction(self):
        r = estimate_viscosity('CCCOCCO', tb=424.5)
        self.assertAlmostEqual(r.dbv, 6.0699243 + 0.0007, places=6)
        self.assertAlmostEqual(r.tv_K, 301.197, delta=0.02)
        self.assertAlmostEqual(r.viscosity_mPa_s(318.15), 0.9218, delta=0.001)
        self.assertEqual(sorted(r.interaction_classes), ['A', 'D'])

    def test_28c_mea_interaction(self):
        r = estimate_viscosity('NCCO', tb=443.45)
        self.assertAlmostEqual(r.dbv, 12.3879997, places=5)
        self.assertAlmostEqual(r.tv_K, 382.276, delta=0.02)
        self.assertAlmostEqual(r.viscosity_mPa_s(363.15), 2.6137, delta=0.002)


class ViscosityBehaviorTests(unittest.TestCase):
    def test_anchored_mode_round_trip(self):
        r = estimate_viscosity('CCNCC', visc_point=(308.15, 0.2740))
        self.assertEqual(r.tv_source, 'from viscosity point')
        self.assertAlmostEqual(r.viscosity_mPa_s(308.15), 0.2740, places=6)
        # Tv is by definition the temperature where eta = 1.3 mPa s
        self.assertAlmostEqual(r.temperature_K(1.3), r.tv_K, places=6)

    def test_internal_tb_mode_warns(self):
        r = estimate_viscosity('CCCCCC')
        self.assertEqual(r.tv_source, 'estimated (internal Tb)')
        self.assertTrue(any('errors compound' in w for w in r.warnings))
        self.assertIsNotNone(r.viscosity_mPa_s(298.15))

    def test_zwitterion_refused(self):
        r = estimate_viscosity('NCC(=O)O', tb=500.0)   # glycine
        self.assertIsNone(r.dbv)
        self.assertIsNone(r.viscosity_mPa_s(400.0))

    def test_ketone_ketone_flagged_questionable(self):
        r = estimate_viscosity('CC(=O)CC(C)=O', tb=413.55)
        self.assertIsNotNone(r.dbv)
        self.assertTrue(any('questionable' in w for w in r.warnings))

    def test_small_acid_refused_but_psat_unaffected(self):
        # dimerization: [4] removed small acids from its own regression
        r = estimate_viscosity('CC(=O)O', tb=391.05)      # acetic acid
        self.assertIsNone(r.dbv)
        self.assertTrue(any('dimerization' in w for w in r.warnings))
        self.assertIsNotNone(estimate_psat('CC(=O)O', tb=391.05).db)
        # heptanoic acid (C7) is past the documented failure range
        r7 = estimate_viscosity('CCCCCCC(=O)O', tb=496.15)
        self.assertIsNotNone(r7.dbv)
        self.assertIsNotNone(r7.viscosity_mPa_s(350.0))

    def test_ortho_chelate_refused_even_anchored(self):
        r = estimate_viscosity('COC(=O)c1ccccc1O',        # methyl salicylate
                               visc_point=(298.15, 3.0))
        self.assertIsNone(r.dbv)
        self.assertIsNone(r.viscosity_mPa_s(298.15))
        self.assertTrue(any('chelating' in w for w in r.warnings))

    def test_heavy_perhalomethane_refused(self):
        r = estimate_viscosity('BrC(Br)(Br)Br', tb=462.65)   # CBr4: 642%
        self.assertIsNone(r.dbv)


if __name__ == '__main__':
    unittest.main()
