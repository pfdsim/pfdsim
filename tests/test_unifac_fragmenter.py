import json
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from unifac import UNIFACModel, parse_smiles_to_unifac
from unifac_fragmenter import UNIFACFragmentationError, fragment, fragment_safe


# Every subgroup ID present in NIST modified UNIFAC but absent from the
# Dortmund table.  A case may contain other subgroups as well; the assertion
# below verifies that the NIST-specific structural environment is selected.
NIST_ONLY_SUBGROUP_CASES = {
    116: 'N#Cc1ccccc1',
    117: 'CN=C=O',
    118: 'CCN=C=O',
    120: 'O=C=Nc1ccccc1',
    121: 'CC(=O)OC(=O)C',
    125: 'O=C1CCN1',
    126: 'O=C1CCO1',
    127: 'CC(=O)Oc1ccccc1',
    128: 'CCC(=O)Oc1ccccc1',
    129: 'CC(C)C(=O)Oc1ccccc1',
    130: 'CC(C)(C)C(=O)Oc1ccccc1',
    131: 'CCOCCO',
    132: 'CCOC(C)CO',
    133: 'COCC(C)O',
    134: 'CSC',
    135: 'CCSCC',
    136: 'CC(C)SC(C)C',
    137: 'CSC(C)(C)C',
    138: 'COOC',
    139: 'CCOOCC',
    140: 'CC(C)OOC(C)C',
    141: 'CC(C)(C)OOC(C)(C)C',
    142: 'COOc1ccccc1',
    143: 'CC(F)C',
    144: 'CC(F)(Cl)C',
    145: 'CC(F)(Cl)Cl',
    146: 'CC(F)F',
    147: 'FC(F)Cl',
    148: 'FC(F)(Cl)Cl',
    149: 'FC(F)F',
    150: 'FC(F)(F)Cl',
    151: 'FC(F)(F)F',
    152: 'CC(C)(OCC)OCC',
    154: 'CCN(C)c1ccccc1',
    155: 'CCN(CC)c1ccccc1',
    156: 'CNc1ccccc1',
    157: 'CCNc1ccccc1',
    158: 'CC(C)Nc1ccccc1',
    159: 'c1ccoc1',
    160: 'c1ccc2occc2c1',
    161: 'c1ccc2oc3ccccc3c2c1',
    162: 'CC1CCCC(C)N1',
    163: 'CC1(CC)CCCC(C)(C)N1',
    164: 'CN1C(C)CCCC1',
    165: 'CCN1C(C)CCCC1',
    166: 'CC(C)N1C(C)CCCC1',
    170: 'C[SiH3]',
    171: 'CC[SiH2]CC',
    172: 'C[Si](C)(C)O[SiH](C)O[Si](C)(C)C',
    173: 'C[Si](C)(C)O[Si](C)(C)C',
    174: 'C[SiH2]O[SiH2]C',
    175: 'C[SiH](C)O[SiH](C)C',
    176: 'C[Si]1(C)O[Si](C)(C)O[Si](C)(C)O[Si](C)(C)O1',
    177: 'CC(=NO)C',
    180: 'NC1CCCCC1',
    186: 'CC(OCC)OCC',
    187: 'CSc1ccccc1',
    188: 'C1CNCCN1',
    190: 'CCN1CCCCC1',
    191: 'CC(C)N1CCCCC1',
    192: 'CC(S)C',
    193: 'CC(C)(C)S',
    194: 'Sc1ccccc1',
    198: '[nH]1c(C)ccc1C',
    199: 'CCOC(=O)Oc1ccccc1',
    200: 'O=C(Oc1ccccc1)Oc1ccccc1',
    202: 'CC1=CCCCC1',
    203: 'CC1=C(C)CCCC1',
    204: 'OCC(O)CO',
    205: 'CC(O)CO',
    206: 'CC(O)C(O)C',
    207: 'CC(C)(O)CO',
    208: 'CC(O)C(O)(C)C',
    301: 'CC(C)C(=O)C(C)C',
    302: 'CC(=O)C(C)(C)C',
    303: 'CC(C)C#N',
    304: 'CC(C)(C)C#N',
    305: 'CC(C)(C)[N+](=O)[O-]',
    306: 'c1ccc(cc1)Nc1ccccc1',
    307: 'CC(C)(N)c1ccccc1',
    308: 'C=O',
    309: 'COCOC',
    1309: 'CCC=NO',
}


CLASSIC_SUBGROUP_CASES = {
    1: 'CC', 2: 'CCC', 3: 'CC(C)C', 4: 'CC(C)(C)C',
    5: 'C=CC', 6: 'CC=CC', 7: 'C=C(C)C', 8: 'CC=C(C)C',
    9: 'c1ccccc1', 10: 'c1ccc(-c2ccccc2)cc1',
    11: 'Cc1ccccc1', 12: 'CCc1ccccc1', 13: 'CC(C)c1ccccc1',
    14: 'CCO', 15: 'CO', 16: 'O', 17: 'Oc1ccccc1',
    18: 'CC(=O)C', 19: 'CCC(=O)CC', 20: 'CC=O',
    21: 'CC(=O)OC', 22: 'CCC(=O)OC', 23: 'COC=O',
    24: 'COC', 25: 'CCOCC', 26: 'CC(C)OC(C)C', 27: 'C1CCOC1',
    28: 'CN', 29: 'CCN', 30: 'CC(C)N',
    31: 'CNC', 32: 'CCNCC', 33: 'CC(C)NC(C)C',
    34: 'CN(C)C', 35: 'CCN(CC)CC', 36: 'Nc1ccccc1',
    37: 'n1ccccc1', 38: 'Cc1ccccn1', 39: 'Cc1cccc(C)n1',
    40: 'CC#N', 41: 'CCC#N', 42: 'CC(=O)O', 43: 'O=CO',
    44: 'CCCl', 45: 'CC(Cl)C', 46: 'CC(Cl)(C)C',
    47: 'ClCCl', 48: 'CC(Cl)Cl', 49: 'CC(Cl)(Cl)C',
    50: 'ClC(Cl)Cl', 51: 'CC(Cl)(Cl)Cl', 52: 'ClC(Cl)(Cl)Cl',
    53: 'Clc1ccccc1', 54: 'C[N+](=O)[O-]',
    55: 'CC[N+](=O)[O-]', 56: 'CC(C)[N+](=O)[O-]',
    57: 'O=[N+]([O-])c1ccccc1', 58: 'S=C=S',
    59: 'CS', 60: 'CCS', 61: 'O=Cc1ccco1', 62: 'OCCO',
    63: 'CCI', 64: 'CCBr', 65: 'C#CCC', 66: 'CC#CC',
    67: 'CS(C)=O', 68: 'C=CC#N', 69: 'ClC=C(Cl)Cl',
    70: 'CC(C)=C(C)C', 71: 'Fc1ccccc1', 72: 'CN(C)C=O',
    73: 'CCN(CC)C=O', 74: 'CC(F)(F)F', 75: 'CC(F)(F)C',
    76: 'CC(F)(C)C', 77: 'COC(=O)C(C)(C)C',
    78: 'C[SiH3]', 79: 'CC[SiH2]CC', 80: 'C[SiH](C)C',
    81: 'C[Si](C)(C)C', 82: 'C[SiH2]O[SiH2]C',
    83: 'C[SiH](C)O[SiH](C)C',
    84: 'C[Si](C)(C)O[Si](C)(C)C',
    85: 'CN1CCCC(=O)1', 86: 'FC(Cl)(Cl)Cl',
    87: 'FC(Cl)(Cl)C', 88: 'FC(Cl)(Cl)', 89: 'FC(Cl)C',
    90: 'FC(F)(Cl)C', 91: 'FC(F)(Cl)', 92: 'FC(F)(F)Cl',
    93: 'FC(F)(Cl)Cl', 94: 'CC(=O)N', 95: 'CC(=O)NC',
    96: 'CC(=O)NCC', 97: 'CC(=O)N(C)C',
    98: 'CC(=O)N(C)CC', 99: 'CC(=O)N(CC)CC',
    100: 'CCOCCO', 101: 'COC(C)CO',
    102: 'CSC', 103: 'CCSCC', 104: 'CC(C)SC(C)C',
    105: 'C1COCCN1', 106: 'c1ccsc1', 107: 'Cc1ccsc1',
    108: 'Cc1cc(C)sc1', 109: 'CN=C=O',
}


DORTMUND_SUBGROUP_CASES = {
    1: 'CC', 2: 'CCC', 3: 'CC(C)C', 4: 'CC(C)(C)C',
    5: 'C=CC', 6: 'CC=CC', 7: 'C=C(C)C', 8: 'CC=C(C)C',
    9: 'c1ccccc1', 10: 'c1ccc(-c2ccccc2)cc1',
    11: 'Cc1ccccc1', 12: 'CCc1ccccc1', 13: 'CC(C)c1ccccc1',
    14: 'CCO', 15: 'CO', 16: 'O', 17: 'Oc1ccccc1',
    18: 'CC(=O)C', 19: 'CCC(=O)CC', 20: 'CC=O',
    21: 'CC(=O)OC', 22: 'CCC(=O)OC', 23: 'COC=O',
    24: 'COC', 25: 'CCOCC', 26: 'CC(C)OC(C)C', 27: 'C1CCOC1',
    28: 'CN', 29: 'CCN', 30: 'CC(C)N',
    31: 'CNC', 32: 'CCNCC', 33: 'CC(C)NC(C)C',
    34: 'CN(C)C', 35: 'CCN(CC)CC', 36: 'Nc1ccccc1',
    37: 'n1ccccc1', 38: 'Cc1ccccn1', 39: 'Cc1cccc(C)n1',
    40: 'CC#N', 41: 'CCC#N', 42: 'CC(=O)O', 43: 'O=CO',
    44: 'CCCl', 45: 'CC(Cl)C', 46: 'CC(Cl)(C)C',
    47: 'ClCCl', 48: 'CC(Cl)Cl', 49: 'CC(Cl)(Cl)C',
    50: 'ClC(Cl)Cl', 51: 'CC(Cl)(Cl)Cl', 52: 'ClC(Cl)(Cl)Cl',
    53: 'Clc1ccccc1', 54: 'C[N+](=O)[O-]',
    55: 'CC[N+](=O)[O-]', 56: 'CC(C)[N+](=O)[O-]',
    57: 'O=[N+]([O-])c1ccccc1', 58: 'S=C=S',
    59: 'CS', 60: 'CCS', 61: 'O=Cc1ccco1', 62: 'OCCO',
    63: 'CCI', 64: 'CCBr', 65: 'C#CCC', 66: 'CC#CC',
    67: 'CS(C)=O', 68: 'C=CC#N', 69: 'ClC=C(Cl)Cl',
    70: 'CC(C)=C(C)C', 71: 'Fc1ccccc1', 72: 'CN(C)C=O',
    73: 'CCN(CC)C=O', 74: 'CC(F)(F)F', 75: 'CC(F)(F)C',
    76: 'CC(F)(C)C', 77: 'COC(=O)C(C)(C)C',
    78: 'C1CCCCC1', 79: 'CC1CCCCC1', 80: 'CC1(C)CCCCC1',
    81: 'CC(O)C', 82: 'CC(C)(C)O', 83: 'C1COCOC1',
    84: 'C1OCOCO1', 85: 'CC(C)(C)N', 86: 'CN1CCCC(=O)1',
    87: 'CCN1CCCC(=O)1', 88: 'CC(C)N1CCCC(=O)1',
    89: 'CC(C)(C)N1CCCC(=O)1', 91: 'CC(=O)N',
    92: 'CC(=O)NC', 93: 'CNC=O', 94: 'CCNC=O',
    100: 'CC(=O)NCC', 101: 'CC(=O)N(C)C',
    102: 'CC(=O)N(C)CC', 103: 'CC(=O)N(CC)CC',
    104: 'c1ccsc1', 105: 'Cc1cccs1', 106: 'Cc1ccc(C)s1',
    107: 'CC1CO1', 108: 'CC1(C)OC1C', 109: 'CC1OC1C',
    110: 'O=S1(=O)CCCC1', 111: 'O=S1(=O)C(C)CCC1',
    112: 'COC(=O)OC', 113: 'CCOC(=O)OCC', 114: 'COC(=O)OCC',
    119: 'C1CO1', 122: 'CSC', 123: 'CCSCC',
    124: 'CC(C)SC(C)C', 153: 'CC1(C)OC1',
    178: 'C[n+]1cc(C)n(C)c1',
    179: 'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F',
    184: 'C[n+]1ccn(C)c1', 189: '[NH2+]1CCCC1',
    195: '[B-](F)(F)(F)F', 196: 'c1cc[nH+]cc1',
    197: 'O=S(=O)([O-])C(F)(F)F', 201: 'CSSC',
    209: 'O=S(=O)([O-])[O-]', 210: 'O=S(=O)(O)[O-]',
    211: '[P-](F)(F)(F)(F)(F)F', 220: 'Cc1cccc[nH+]1',
}


NIST_COLLIDING_ID_CASES = {
    # These IDs exist in Dortmund too, but mean different chemistry in NIST.
    119: 'CC(C)N=C=O',
    122: 'CS(=O)(=O)c1ccccc1',
    123: 'O=Cc1ccccc1',
    124: 'O=C(O)c1ccccc1',
    153: 'CN(C)c1ccccc1',
    178: 'CC(=O)c1ccccc1',
    179: 'ClC(Cl)=C(Cl)Cl',
    189: 'CN1CCCCC1',
    195: 'CC(C)(C)c1ccccc1',
    196: 'c1cc[nH]c1',
    197: 'Cc1ccc[nH]1',
    201: 'C1=CCCCC1',
    209: 'CC(C)(O)C(O)(C)C',
}


class NativeUNIFACFragmenterTests(unittest.TestCase):
    def test_flask_api_routes_explicit_smiles_to_native_fragmenter(self):
        from app import app

        client = app.test_client()
        cases = (
            ({'smiles': 'OCC(O)CO', 'variant': 'UNIFNIST'}, {'204': 1}),
            ({'smiles': 'C1CCCCC1', 'variant': 'UNIFDMD'}, {'78': 6}),
            ({'smiles': 'n1ccccc1', 'variant': 'UNIFAC'}, {'37': 1}),
        )
        for payload, expected in cases:
            with self.subTest(variant=payload['variant']):
                response = client.post('/api/unifac-groups', json=payload)
                body = response.get_json()
                self.assertEqual(response.status_code, 200, body)
                self.assertEqual(body['groups'], expected)
                self.assertEqual(body['source'], 'native_smiles')

    def test_simulator_routes_pfd_smiles_using_selected_variant(self):
        from simulator import Simulator

        cases = (
            ('UNIFNIST', 'glycerol', 'Glycerol', 92.09382,
             'OCC(O)CO', {204: 1}),
            ('UNIFDMD-RK', 'cyclohexane', 'Cyclohexane', 84.15948,
             'C1CCCCC1', {78: 6}),
            ('UNIFAC-PR', 'pyridine', 'Pyridine', 79.0999,
             'n1ccccc1', {37: 1}),
        )
        for method, symbol, name, molecular_weight, smiles, expected in cases:
            with self.subTest(method=method):
                simulator = Simulator.from_string(
                    'PROCESS: Native fragmenter integration\n'
                    'ONLINE_LOOKUP: false\n'
                    f'THERMO_METHOD: {method}\n'
                    'COMPONENTS:\n'
                    f'    {symbol} | {name} | MW={molecular_weight}, SMILES={smiles}\n'
                    '    water | Water | MW=18.01528, SMILES=O\n'
                )
                simulator.initialize()
                self.assertEqual(
                    simulator.thermo.component_groups[symbol],
                    expected,
                )

    def test_variant_aliases_and_strict_failure_paths(self):
        self.assertEqual(fragment('C1CCCCC1', 'UNIFAC2'), {2: 6})
        self.assertEqual(fragment('C1CCCCC1', 'UNIFM2'), {78: 6})

        with self.assertRaisesRegex(UNIFACFragmentationError, 'unsupported UNIFAC variant'):
            fragment('CC', 'NOT-A-MODEL')
        with self.assertRaisesRegex(UNIFACFragmentationError, 'could not parse SMILES'):
            fragment('not a smiles', 'UNIFAC')
        with self.assertRaisesRegex(UNIFACFragmentationError, 'element P is unsupported'):
            fragment('CP', 'UNIFAC')
        with self.assertRaisesRegex(UNIFACFragmentationError, 'charged and radical'):
            fragment('C[NH3+]', 'UNIFNIST')
        with self.assertRaisesRegex(UNIFACFragmentationError, 'silicon groups are absent'):
            fragment('C[SiH3]', 'UNIFDMD')
        with self.assertRaisesRegex(UNIFACFragmentationError, 'epoxide substitution'):
            fragment('C1CO1', 'UNIFNIST')

        self.assertIsNone(fragment_safe('not a smiles', 'UNIFAC'))
        self.assertEqual(fragment_safe('COC', 'UNIFNIST'), {1: 1, 24: 1})

    def test_same_numeric_ids_do_not_share_cross_variant_meanings(self):
        collision_cases = [
            ('C1CO1', 'UNIFDMD', {119: 1}),
            ('CC(C)N=C=O', 'UNIFNIST', {1: 2, 119: 1}),
            ('CSC', 'UNIFDMD', {1: 1, 122: 1}),
            ('CS(=O)(=O)c1ccccc1', 'UNIFNIST', {1: 1, 9: 5, 122: 1}),
            ('CCSCC', 'UNIFDMD', {1: 2, 2: 1, 123: 1}),
            ('O=Cc1ccccc1', 'UNIFNIST', {9: 5, 123: 1}),
            ('CC(C)SC(C)C', 'UNIFDMD', {1: 4, 3: 1, 124: 1}),
            ('O=C(O)c1ccccc1', 'UNIFNIST', {9: 5, 124: 1}),
            ('CC1(C)OC1', 'UNIFDMD', {1: 2, 153: 1}),
            ('CN(C)c1ccccc1', 'UNIFNIST', {9: 5, 153: 1}),
            ('CC(=O)c1ccccc1', 'UNIFNIST', {1: 1, 9: 5, 178: 1}),
            ('ClC(Cl)=C(Cl)Cl', 'UNIFNIST', {179: 1}),
            ('CN1CCCCC1', 'UNIFNIST', {78: 4, 189: 1}),
            ('CC(C)(C)c1ccccc1', 'UNIFNIST', {1: 3, 9: 5, 195: 1}),
            ('c1cc[nH]c1', 'UNIFNIST', {9: 2, 196: 1}),
            ('C1=CCCCC1', 'UNIFNIST', {78: 4, 201: 1}),
            ('CC(C)(O)C(O)(C)C', 'UNIFNIST', {1: 4, 209: 1}),
        ]
        for smiles, variant, expected in collision_cases:
            with self.subTest(smiles=smiles, variant=variant):
                self.assertEqual(fragment(smiles, variant), expected)

    def test_variant_level_ring_and_heteroaromatic_differences(self):
        expected = {
            ('C1CCCCC1', 'UNIFAC'): {2: 6},
            ('C1CCCCC1', 'UNIFDMD'): {78: 6},
            ('C1CCCCC1', 'UNIFNIST'): {78: 6},
            ('C1CCOC1', 'UNIFAC'): {2: 3, 27: 1},
            ('C1CCOC1', 'UNIFDMD'): {27: 1, 78: 2},
            ('C1CCOC1', 'UNIFNIST'): {27: 1, 78: 2},
            ('n1ccccc1', 'UNIFAC'): {37: 1},
            ('n1ccccc1', 'UNIFDMD'): {9: 3, 37: 1},
            ('n1ccccc1', 'UNIFNIST'): {9: 3, 37: 1},
            ('c1ccsc1', 'UNIFAC'): {106: 1},
            ('c1ccsc1', 'UNIFDMD'): {9: 2, 104: 1},
            ('c1ccsc1', 'UNIFNIST'): {9: 2, 104: 1},
            ('C1=CCCCC1', 'UNIFAC'): {2: 4, 6: 1},
            ('C1=CCCCC1', 'UNIFDMD'): {6: 1, 78: 4},
            ('C1=CCCCC1', 'UNIFNIST'): {78: 4, 201: 1},
        }
        for (smiles, variant), groups in expected.items():
            with self.subTest(smiles=smiles, variant=variant):
                self.assertEqual(fragment(smiles, variant), groups)

    def test_cho_names_and_numeric_assignments_are_unambiguous(self):
        for filename in ('unifac_params.json', 'nist_modified_unifac_params.json'):
            model = UNIFACModel(os.path.join(ROOT, 'data', filename))
            with self.subTest(filename=filename):
                self.assertEqual(model._resolve_groups({'CHO': 1}), {20: 1})
                self.assertEqual(model._resolve_groups({'CH-O': 1}), {26: 1})

        dortmund = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_dmd.txt'))
        self.assertEqual(dortmund._resolve_groups({'CHO': 1}), {20: 1})
        self.assertEqual(dortmund._resolve_groups({'CH-O': 1}), {26: 1})

        for variant in ('UNIFAC', 'UNIFDMD', 'UNIFNIST'):
            with self.subTest(variant=variant, functional_group='aldehyde'):
                self.assertEqual(fragment('CC=O', variant), {1: 1, 20: 1})
            with self.subTest(variant=variant, functional_group='ether_methine'):
                self.assertEqual(
                    fragment('CC(C)OC(C)C', variant),
                    {1: 4, 3: 1, 26: 1},
                )

    def test_every_nist_subgroup_has_a_representative_structure(self):
        model = UNIFACModel(os.path.join(ROOT, 'data', 'nist_modified_unifac_params.json'))
        cases = {
            subgroup: NIST_ONLY_SUBGROUP_CASES.get(
                subgroup,
                NIST_COLLIDING_ID_CASES.get(
                    subgroup,
                    DORTMUND_SUBGROUP_CASES.get(
                        subgroup,
                        CLASSIC_SUBGROUP_CASES.get(subgroup),
                    ),
                ),
            )
            for subgroup in model.subgroups
        }

        self.assertEqual(set(cases), set(model.subgroups))
        self.assertFalse({subgroup for subgroup, smiles in cases.items()
                          if smiles is None})
        for subgroup, smiles in cases.items():
            with self.subTest(subgroup=subgroup, smiles=smiles):
                assigned = fragment(smiles, 'UNIFNIST')
                self.assertIn(subgroup, assigned)
                self.assertGreater(assigned[subgroup], 0)

    def test_every_dortmund_subgroup_has_a_representative_structure(self):
        model = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_dmd.txt'))

        self.assertEqual(set(DORTMUND_SUBGROUP_CASES), set(model.subgroups))
        for subgroup, smiles in DORTMUND_SUBGROUP_CASES.items():
            with self.subTest(subgroup=subgroup, smiles=smiles):
                assigned = fragment(smiles, 'UNIFDMD')
                self.assertIn(subgroup, assigned)
                self.assertGreater(assigned[subgroup], 0)

    def test_every_classic_subgroup_has_a_representative_structure(self):
        model = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_params.json'))

        self.assertEqual(set(CLASSIC_SUBGROUP_CASES), set(model.subgroups))
        for subgroup, smiles in CLASSIC_SUBGROUP_CASES.items():
            with self.subTest(subgroup=subgroup, smiles=smiles):
                assigned = fragment(smiles, 'UNIFAC')
                self.assertIn(subgroup, assigned)
                self.assertGreater(assigned[subgroup], 0)

    def test_every_nist_only_subgroup_has_a_representative_structure(self):
        model = UNIFACModel(os.path.join(ROOT, 'data', 'nist_modified_unifac_params.json'))
        dortmund = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_dmd.txt'))
        expected_ids = set(model.subgroups) - set(dortmund.subgroups)

        self.assertEqual(set(NIST_ONLY_SUBGROUP_CASES), expected_ids)
        for subgroup, smiles in NIST_ONLY_SUBGROUP_CASES.items():
            with self.subTest(subgroup=subgroup, smiles=smiles):
                assigned = fragment(smiles, 'UNIFNIST')
                self.assertIn(subgroup, assigned)
                self.assertGreater(assigned[subgroup], 0)

    def test_common_groups_are_consistent_across_parameter_families(self):
        cases = {
            'CCO': {1: 1, 2: 1, 14: 1},
            'CC(=O)C': {1: 1, 18: 1},
            'CC(=O)OCC': {1: 1, 2: 1, 21: 1},
            'COC': {1: 1, 24: 1},
            'CCOCC': {1: 2, 2: 1, 25: 1},
            'Cc1ccccc1': {9: 5, 11: 1},
            'CC#N': {40: 1},
            'C[N+](=O)[O-]': {54: 1},
        }
        for variant in ('UNIFAC', 'UNIFDMD', 'UNIFNIST'):
            for smiles, expected in cases.items():
                with self.subTest(variant=variant, smiles=smiles):
                    self.assertEqual(fragment(smiles, variant), expected)

        self.assertEqual(fragment('c1ccncc1', 'UNIFAC'), {37: 1})
        self.assertEqual(fragment('c1ccncc1', 'UNIFDMD'), {9: 3, 37: 1})
        self.assertEqual(fragment('c1ccncc1', 'UNIFNIST'), {9: 3, 37: 1})
        self.assertEqual(fragment('c1ccsc1', 'UNIFAC'), {106: 1})
        self.assertEqual(fragment('c1ccsc1', 'UNIFDMD'), {9: 2, 104: 1})

    def test_modified_variants_distinguish_alcohol_and_cyclic_groups(self):
        self.assertEqual(fragment('CC(O)C', 'UNIFAC'), {1: 2, 3: 1, 14: 1})
        self.assertEqual(fragment('CC(O)C', 'UNIFDMD'), {1: 2, 3: 1, 81: 1})
        self.assertEqual(fragment('C1CCCCC1', 'UNIFAC'), {2: 6})
        self.assertEqual(fragment('C1CCCCC1', 'UNIFNIST'), {78: 6})
        self.assertEqual(fragment('CC1CO1', 'UNIFDMD'), {1: 1, 107: 1})
        self.assertEqual(fragment('C1CO1', 'UNIFDMD'), {119: 1})

    def test_nist_uses_dedicated_polyol_anhydride_and_acetal_groups(self):
        self.assertEqual(fragment('OCC(O)CO', 'UNIFNIST'), {204: 1})
        self.assertEqual(fragment('OCCO', 'UNIFNIST'), {62: 1})
        self.assertEqual(fragment('CC(=O)OC(=O)C', 'UNIFNIST'), {1: 2, 121: 1})
        self.assertEqual(fragment('COCOC', 'UNIFNIST'), {1: 2, 309: 1})

    def test_nist_preserves_both_source_subgroups_numbered_309(self):
        model = UNIFACModel(os.path.join(ROOT, 'data', 'nist_modified_unifac_params.json'))

        self.assertEqual(model.subgroups[309].name, 'CH2(O)2')
        self.assertEqual(model.subgroups[1309].name, 'CH=NOH')
        self.assertEqual(fragment('CCC=NO', 'UNIFNIST'), {1: 1, 2: 1, 1309: 1})
        self.assertEqual(fragment('CC(=NO)C', 'UNIFNIST'), {1: 2, 177: 1})

    def test_dme_alias_and_structure_have_the_same_atom_balanced_assignment(self):
        from unifac import get_unifac_groups

        self.assertEqual(get_unifac_groups('DME', variant='UNIFNIST'),
                         {'CH3': 1, 'CH3O': 1})
        self.assertEqual(parse_smiles_to_unifac('COC', 'UNIFNIST'), {1: 1, 24: 1})

    def test_unsupported_structures_fail_instead_of_cross_variant_fallback(self):
        with self.assertRaisesRegex(UNIFACFragmentationError, 'anhydrides require NIST'):
            fragment('CC(=O)OC(=O)C', 'UNIFDMD')
        with self.assertRaisesRegex(UNIFACFragmentationError, 'epoxides require'):
            fragment('CC1CO1', 'UNIFAC')

    def test_generated_nist_dataset_has_complete_bidirectional_pairs(self):
        with open(os.path.join(ROOT, 'data', 'nist_modified_unifac_params.json')) as handle:
            data = json.load(handle)

        diagnostics = data['diagnostics']
        self.assertEqual(diagnostics['directed_interaction_count'], 1982)
        self.assertEqual(diagnostics['unordered_interaction_pair_count'], 991)
        self.assertEqual(diagnostics['directed_pairs_missing_reverse'], [])
        self.assertEqual(diagnostics['rejected_interactions'][0]['i'], 59)
        self.assertEqual(diagnostics['rejected_interactions'][0]['j'], 13)
        self.assertEqual(
            diagnostics['interaction_overlay']['directed_parameter_count'],
            24,
        )
        by_pair = {
            (row['i'], row['j']): row
            for row in data['interactions']
        }
        self.assertEqual(
            (by_pair[(1, 63)]['a1'], by_pair[(1, 63)]['a2'], by_pair[(1, 63)]['a3']),
            (2500.0, -4.54957, 0.0005866),
        )
        self.assertEqual(
            (by_pair[(63, 3)]['a1'], by_pair[(63, 3)]['a2'], by_pair[(63, 3)]['a3']),
            (-1009.548, 2.17462, 0.0),
        )
        self.assertEqual(
            by_pair[(42, 63)]['source_overlay']['markers'],
            ['d'],
        )
        self.assertEqual(
            data['metadata']['source']['interaction_overlays'][0]['doi'],
            '10.1016/j.fluid.2022.113673',
        )

        model = UNIFACModel(
            os.path.join(ROOT, 'data', 'nist_modified_unifac_params.json')
        )
        self.assertEqual(
            model.interaction_coefficients[(1, 63)],
            (2500.0, -4.54957, 0.0005866),
        )
        self.assertEqual(
            model.interaction_coefficients[(63, 1)],
            (1690.251, -8.83803, 0.009),
        )

    def test_nist_lactone_builder_overlay_replaces_and_adds_pairs(self):
        from scripts.convert_nist_modified_unifac_docx import (
            LACTONE_OVERLAY_JSON,
            SOURCE_DOCX,
            parse_docx,
        )

        _subgroups, original, original_diagnostics = parse_docx(
            SOURCE_DOCX,
            interaction_overlay_path=None,
        )
        _subgroups, overlaid, overlaid_diagnostics = parse_docx(
            SOURCE_DOCX,
            interaction_overlay_path=LACTONE_OVERLAY_JSON,
        )
        original_by_pair = {(row['i'], row['j']): row for row in original}
        overlaid_by_pair = {(row['i'], row['j']): row for row in overlaid}

        self.assertEqual(original_diagnostics['directed_interaction_count'], 1968)
        self.assertEqual(original_diagnostics['unordered_interaction_pair_count'], 984)
        self.assertNotIn('interaction_overlay', original_diagnostics)
        self.assertNotIn((1, 63), original_by_pair)
        self.assertAlmostEqual(original_by_pair[(3, 63)]['a1'], -210.75)

        self.assertEqual(overlaid_diagnostics['directed_interaction_count'], 1982)
        self.assertEqual(overlaid_diagnostics['unordered_interaction_pair_count'], 991)
        self.assertEqual(overlaid_by_pair[(1, 63)]['a1'], 2500.0)
        self.assertEqual(overlaid_by_pair[(63, 1)]['a1'], 1690.251)
        self.assertEqual(overlaid_by_pair[(3, 63)]['a1'], 1174.903)
        self.assertEqual(overlaid_by_pair[(63, 3)]['a1'], -1009.548)
        self.assertEqual(overlaid_by_pair[(3, 63)]['source_overlay']['markers'], ['c'])
        self.assertEqual(overlaid_by_pair[(42, 63)]['source_overlay']['markers'], ['d'])
        self.assertEqual(overlaid_by_pair[(42, 63)]['a1'], -691.82)
        self.assertEqual(overlaid_by_pair[(42, 63)]['a2'], 7.72858)
        self.assertEqual(overlaid_by_pair[(10, 63)]['Tmax'], 298.15)
        self.assertIsNone(
            overlaid_by_pair[(10, 63)]['source_overlay']['source_Tmax_K']
        )

        overlay = overlaid_diagnostics['interaction_overlay']
        self.assertEqual(
            overlay['replaced_unordered_pairs'],
            [[3, 63], [4, 63], [7, 63], [42, 63], [43, 63]],
        )
        self.assertEqual(
            overlay['added_unordered_pairs'],
            [[1, 63], [5, 63], [9, 63], [10, 63], [11, 63], [13, 63], [20, 63]],
        )


if __name__ == '__main__':
    unittest.main()
