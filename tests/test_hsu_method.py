"""Tests for hsu_method: fragmentation, worked example, corrections, guards,
value regressions, and the property_resolution/viscosity.py integration."""
import math
import os
import sys
import unittest
import warnings
from unittest.mock import patch

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import hsu_method as hm


def frag_dict(smiles):
    return dict(hm.fragment(smiles).groups)


class FragmentationTests(unittest.TestCase):
    CASES = [
        ('CCCCCC', {'ch3': 2, 'ch2': 4}),
        ('Cc1ccccc1', {'ch3': 1, 'aromatic_ch': 5, 'aromatic_c_simple': 1}),
        ('ClCCl', {'cl2': 1, 'ch2': 1}),                     # carbon counted
        ('ClC(Cl)Cl', {'cl3': 1}),                           # carbon subsumed
        ('ClC(Cl)(Cl)Cl', {'cl4': 1}),
        ('C#CCCCC', {'alkyne_terminal': 1, 'ch2': 3, 'ch3': 1}),
        ('CC#CC', {'alkyne_internal': 1, 'ch3': 2}),
        ('COc1ccccc1', {'ch3': 1, 'aromatic_o': 1, 'aromatic_ch': 5,
                        'aromatic_c_simple': 1}),
        ('C(=C)Cl', {'vinyl_chcl': 1, 'alkene_ch2': 1}),
        ('CS(C)=O', {'sulfoxide': 1, 'ch3': 2}),
        ('FC(F)(F)c1ccccc1', {'f3': 1, 'c': 1, 'aromatic_ch': 5,
                              'aromatic_c_simple': 1}),
        ('CNc1ccccc1', {'aromatic_amine_secondary': 1, 'ch3': 1,
                        'aromatic_ch': 5, 'aromatic_c_simple': 1}),
        ('CC(=O)Cl', {'acid_chloride': 1, 'ch3': 1}),
        ('COC(=O)OC', {'carbonate': 1, 'ch3': 2}),
        ('CCOC(C)=O', {'ester_lt8': 1, 'ch3': 2, 'ch2': 1}),
        ('NC=O', {'formamide_molecule': 1}),
        ('Ic1ccccc1', {'aromatic_i': 1, 'aromatic_ch': 5,
                       'aromatic_c_simple': 1}),
        ('[O-][N+](=O)c1ccccc1', {'aromatic_nitro': 1, 'aromatic_ch': 5,
                                  'aromatic_c_simple': 1}),
        ('C1CCC2CCCCC2C1', {'ring_ch2': 8, 'ring_ch': 2}),
        ('c1ccc2ccccc2c1', {'aromatic_ch': 8, 'aromatic_c_naphthalene': 2}),
        ('c1ccc(-c2ccccc2)cc1', {'aromatic_ch': 10, 'aromatic_c_biphenyl': 2}),
        ('C1=CCCCC1', {'ring_alkene_ch': 2, 'ring_ch2': 4}),
        ('OC1CCCCC1', {'oh_ring': 1, 'ring_ch2': 5, 'ring_ch': 1}),
    ]

    def test_fragmentations(self):
        for smiles, want in self.CASES:
            with self.subTest(smiles=smiles):
                self.assertEqual(frag_dict(smiles), want)

    REJECTS = [
        ('OCCO', 'polyhydric'), ('C1CCOC1', 'ring ether'),
        ('CN(C)C=O', 'amide'), ('O=C1CCCCC1', 'ring ketone'),
        ('c1ccc(Oc2ccccc2)cc1', 'diaryl ether'),
        ('c1ccc(Nc2ccccc2)cc1', 'diaryl amine'),
        ('CC(C)Br', 'secondary bromide'), ('c1ccncc1', 'heterocycle'),
        ('C1CCc2ccccc2C1', 'tetralin'), ('ClC(F)Cl', 'mixed halogen'),
        ('OC(=O)CCC(O)=O', 'polycarboxylic'), ('C=CC=C', 'polyene'),
        ('C#CC#C', 'C#C'), ('O1CCOCC1', 'ring ether'),
        ('CCOO', 'oxygen environment'), ('CSSC', 'disulfide'),
        # 2026-07-12 VDI-block gates
        ('NNc1ccccc1', 'N-N bond (phenylhydrazine +293%)'),
        ('CC(Cl)(Cl)Cl', 'substituted CCl3 (+81%)'),
        ('ClC(Cl)C(Cl)Cl', 'adjacent polyhalogenated carbons (+69%)'),
        ('COC(=O)c1ccccc1O', 'ortho-chelate (methyl salicylate +1207%)'),
        ('Oc1ccccc1[N+](=O)[O-]', 'ortho-chelate (o-nitrophenol)'),
        ('OC(=O)CCl', 'halogenated acid (chloroacetic +32%)'),
    ]

    def test_rejections(self):
        for smiles, why in self.REJECTS:
            with self.subTest(smiles=smiles):
                with self.assertRaises(hm.HsuFragmentationError):
                    hm.fragment(smiles)

    def test_phenolic_tr_cap(self):
        frag = hm.fragment('Oc1ccccc1')
        self.assertAlmostEqual(frag.tr_max, 0.65)
        self.assertAlmostEqual(hm.fragment('CCCCCC').tr_max, 0.75)


class WorkedExampleTests(unittest.TestCase):
    def test_benzotrifluoride_paper_example(self):
        """Hsu example 2: T=313.15 K, Pc=35.59 bar -> eta = 0.4925 mPa.s."""
        res = hm.estimate_viscosity('FC(F)(F)c1ccccc1', pc_kPa=3559.0)
        self.assertAlmostEqual(res.viscosity_mPa_s(313.15), 0.4925, places=4)

    def test_chloroform_table6(self):
        """Table 6 replay: chloroform, cl3 subsumes the carbon."""
        res = hm.estimate_viscosity('ClC(Cl)Cl', pc_kPa=5470.0)
        self.assertAlmostEqual(res.viscosity_mPa_s(273.15), 0.708, delta=0.01)

    def test_bromobenzene_PFDSim_correction(self):
        """Q88 a/10: ~0.985 mPa.s at 298 K (Perry ~1.03; -99.9% as printed)."""
        res = hm.estimate_viscosity('Brc1ccccc1', pc_kPa=4520.0)
        v = res.viscosity_mPa_s(298.15)
        self.assertGreater(v, 0.7)
        self.assertLess(v, 1.4)

    def test_alkene_sign_fix(self):
        """Q7 sign flip: 1-butene ~0.79 mPa.s at 163 K (Table 6: 0.790)."""
        res = hm.estimate_viscosity('C=CCC', pc_kPa=4020.0)
        self.assertAlmostEqual(res.viscosity_mPa_s(163.0), 0.79, delta=0.12)


class QualityGuardTests(unittest.TestCase):
    def test_dsum_penalty(self):
        real = hm.estimate_viscosity('C1CCCCC1', pc_kPa=4080, pc_quality=1.0)
        est = hm.estimate_viscosity('C1CCCCC1', pc_kPa=4110, pc_quality=0.75)
        self.assertAlmostEqual(real.quality, 0.75, places=2)
        self.assertLessEqual(est.quality, 0.3)     # hard guard fired
        self.assertGreaterEqual(est.quality, 0.1)  # but stays usable
        hexane = hm.estimate_viscosity('CCCCCC', pc_kPa=3100, pc_quality=0.75)
        self.assertAlmostEqual(hexane.quality, 0.75, places=2)

    def test_method_factor_caps(self):
        self.assertAlmostEqual(
            hm.fragment('CC(=O)Cl').method_factor, 0.55, places=2)   # first member
        self.assertAlmostEqual(
            hm.fragment('CCC(=O)Cl').method_factor, 0.65, places=2)
        self.assertAlmostEqual(
            hm.fragment('C1CCC2CCCCC2C1').method_factor, 0.50, places=2)

    def test_tr_validity_returns_none(self):
        res = hm.estimate_viscosity('CCCCCC', pc_kPa=3025.0, tc_K=507.6)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertIsNone(res.viscosity_mPa_s(0.80 * 507.6))
        self.assertIsNotNone(res.viscosity_mPa_s(0.70 * 507.6))


class RefitUnitTests(unittest.TestCase):
    def test_alkyne_units_pc_robust(self):
        """Refit alkyne units carry d=0: whole-molecule Sd stays small."""
        for smi in ('C#CCCCC', 'CC#CC'):
            frag = hm.fragment(smi)
            self.assertLess(abs(frag.d_sum), 1.0)

    def test_internal_alkyne_sanity(self):
        """2-butyne ~0.19 mPa.s at 300 K (Perry: ~0.20)."""
        res = hm.estimate_viscosity('CC#CC', pc_kPa=4870.0)
        self.assertAlmostEqual(res.viscosity_mPa_s(300.0), 0.20, delta=0.05)


class ValueRegressionTests(unittest.TestCase):
    """Pin module-level values at 298.15 K (transferred from the old
    resolver-level whitelisted/cautious family tests, recomputed for the
    native engine).  eta in mPa.s; Pc in kPa."""

    CASES = (
        # smiles, pc_kPa, eta_mPa_s, method_factor
        ('CCCCCC', 3025.0, 0.29923305515, 0.75),          # hexane
        ('CCOCC', 3640.0, 0.22490374181, 0.75),           # diethyl ether
        ('CC(=O)O', 5786.0, 1.0712365344, 0.75),          # acetic acid
        ('CC#N', 4850.0, 0.35059615456, 0.75),            # acetonitrile
        ('CCCl', 5270.0, 0.25613067439, 0.75),            # chloroethane
        ('CC#CC', 4870.0, 0.19536522854, 0.65),           # 2-butyne (refit)
        ('C1=CCCCC1', 4350.0, 0.62133306459, 0.70),       # cyclohexene
        ('CC(=O)OC(=O)C', 4000.0, 1.0899685620, 0.70),    # acetic anhydride
        ('CC(=O)Cl', 5870.0, 0.36813835559, 0.55),        # acetyl chloride
        ('CCNCC', 3710.0, 0.26824281248, 0.70),           # diethylamine (Q56 hers)
    )

    def test_values_and_factors(self):
        for smiles, pc_kpa, eta, factor in self.CASES:
            with self.subTest(smiles=smiles):
                res = hm.estimate_viscosity(smiles, pc_kPa=pc_kpa)
                self.assertAlmostEqual(
                    res.viscosity_mPa_s(298.15) / eta, 1.0, places=6)
                self.assertAlmostEqual(
                    res.fragmentation.method_factor, factor, places=2)


class ResolverIntegrationTests(unittest.TestCase):
    """The hsu_liquid_viscosity rung of property_resolution/viscosity.py."""

    @classmethod
    def setUpClass(cls):
        from property_resolver import PropertyResolver
        from property_resolution.base import (PropertyResolutionResult,
                                              PropertyResolutionError)
        cls.PropertyResolver = PropertyResolver
        cls.Result = PropertyResolutionResult
        cls.Error = PropertyResolutionError

    def resolve(self, symbol, smiles, tc, pc_bar, T=298.15, pc_quality=1.0,
                patch_yoon=False):
        resolver = self.PropertyResolver()
        critical = {
            'Tc': self.Result(tc, 'provided', 'direct', 1.0, ''),
            'Pc': self.Result(pc_bar, 'provided', 'direct', pc_quality, ''),
        }
        patches = [
            patch.object(resolver, '_get_perry_evaluation', return_value=None),
            patch.object(resolver, '_coolprop_viscosity', return_value=None),
            patch.object(resolver, 'resolve_critical_properties',
                         return_value=critical),
        ]
        if patch_yoon:
            patches.append(patch.object(resolver, '_yoon_thodos_viscosity',
                                        return_value=None))
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return resolver.resolve_viscosity(
                symbol, T, phase='liquid', props={'smiles': smiles})

    def test_paper_worked_example_through_resolver(self):
        viscosity = self.resolve('benzotrifluoride', 'FC(F)(F)c1ccccc1',
                                 565.0, 35.59, T=313.15)
        self.assertEqual(viscosity.method, 'hsu_liquid_viscosity')
        self.assertEqual(viscosity.source, 'calculated')
        self.assertAlmostEqual(viscosity.value * 1e3, 0.4925, places=4)
        self.assertAlmostEqual(viscosity.quality, 0.70, places=3)
        self.assertIn('groups:', viscosity.notes)
        self.assertIn('1 f3', viscosity.notes)
        self.assertIn('Sum(d)', viscosity.notes)

    def test_pc_quality_guard_scales_with_sum_d(self):
        # cyclohexane Sum(d) = -9.1: estimated Pc collapses quality to the
        # last-resort floor; hexane Sum(d) = -0.05 is untouched.
        cyclo = self.resolve('cyclohexane-like', 'C1CCCCC1', 553.8, 40.8,
                             pc_quality=0.75)
        hexane = self.resolve('hexane-like', 'CCCCCC', 507.6, 30.25,
                              pc_quality=0.75)
        self.assertAlmostEqual(cyclo.quality, 0.15, places=3)
        self.assertAlmostEqual(hexane.quality, 0.75, places=3)

    def test_mild_pc_uncertainty_penalizes_proportionally(self):
        # ethanol Sum(d) = -1.13: only the excess beyond |Sum d| = 1 is
        # penalized -> 0.75 * (1 - 0.13*0.2) at Pc quality 0.80.
        viscosity = self.resolve('ethanol-like', 'CCO', 514.0, 61.4,
                                 pc_quality=0.80)
        self.assertAlmostEqual(viscosity.value, 0.0010798814984, places=9)
        self.assertAlmostEqual(viscosity.quality, 0.7303, places=3)

    def test_rejections_fall_through_the_ladder(self):
        for symbol, smiles, tc, pc in (
            ('meg-like', 'OCCO', 720.0, 82.0),          # polyhydric
            ('thf-like', 'C1CCOC1', 540.1, 51.9),       # ring ether
            ('dmf-like', 'CN(C)C=O', 649.6, 44.2),      # disubstituted amide
        ):
            with self.subTest(symbol=symbol):
                with self.assertRaises(self.Error):
                    self.resolve(symbol, smiles, tc, pc)

    def test_tr_validity_window_enforced(self):
        # ether Tr floor 0.45 (low-T divergence) and the global 0.75 cap
        for symbol, smiles, tc, pc, temperature in (
            ('ether-like', 'CCOCC', 466.7, 36.4, 200.0),     # Tr 0.43
            ('ethanol-like', 'CCO', 400.0, 61.4, 350.0),     # Tr 0.875
        ):
            with self.subTest(symbol=symbol):
                with self.assertRaises(self.Error):
                    self.resolve(symbol, smiles, tc, pc,
                                 T=temperature, patch_yoon=True)

    def test_fragmentation_cache_reuses_results_and_rejections(self):
        resolver = self.PropertyResolver()
        critical = {
            'Tc': self.Result(507.6, 'provided', 'direct', 1.0, ''),
            'Pc': self.Result(30.25, 'provided', 'direct', 1.0, ''),
        }
        with patch.object(resolver, '_get_perry_evaluation', return_value=None), \
             patch.object(resolver, '_coolprop_viscosity', return_value=None), \
             patch.object(resolver, 'resolve_critical_properties',
                          return_value=critical):
            first = resolver.resolve_viscosity(
                'hexane-like', 298.15, phase='liquid',
                props={'smiles': 'CCCCCC'})
            with patch.object(hm, 'fragment',
                              side_effect=AssertionError('cache miss')):
                second = resolver.resolve_viscosity(
                    'hexane-like', 310.0, phase='liquid',
                    props={'smiles': 'CCCCCC'})
            with self.assertRaises(self.Error):
                resolver.resolve_viscosity(
                    'meg-like', 298.15, phase='liquid',
                    props={'smiles': 'OCCO'})
            self.assertIsInstance(
                resolver._hsu_fragmentation_cache['OCCO'],
                hm.HsuFragmentationError)
        self.assertGreater(first.value, second.value)   # eta falls with T


if __name__ == '__main__':
    unittest.main()
