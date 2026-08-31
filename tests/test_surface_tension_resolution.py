import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from rdkit import Chem

from pfd_parser import PFDParser
from property_resolver import (
    PropertyResolutionError,
    PropertyResolutionResult,
    PropertyResolver,
)
from simulator import Simulator


ROOT = Path(__file__).resolve().parents[1]


class SurfaceTensionResolutionTests(unittest.TestCase):
    def assertClose(self, actual, expected, *, rel=1e-8, abs_tol=1e-12):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol),
            f'{actual!r} != {expected!r}',
        )

    def test_generated_correlation_database_has_expected_method_counts(self):
        payload = json.loads(
            (ROOT / 'data' / 'surface_tension_correlations_cas.json').read_text()
        )

        self.assertEqual(payload['metadata']['compound_count'], 690)
        self.assertEqual(payload['metadata']['correlation_count'], 1037)
        self.assertEqual(payload['metadata']['source_counts'], {
            'jasper_lange': 522,
            'mulero_cachadina_refprop': 115,
            'somayajulu': 64,
            'somayajulu_revised': 64,
            'vdi_ppds_11_dippr_eq106': 272,
        })
        self.assertIn('56-81-5', payload['chemicals'])

    def test_local_correlation_priority_prefers_refprop_then_vdi(self):
        resolver = PropertyResolver()

        water = resolver.resolve_surface_tension('water', 298.15)
        glycerol = resolver.resolve_surface_tension('glycerol', 298.15)

        self.assertEqual(water.method, 'mulero_cachadina_refprop_surface_tension')
        self.assertClose(water.value, 0.07205503890847453)
        self.assertClose(water.quality, 0.98)
        self.assertIn('7732-18-5', water.notes)

        self.assertEqual(glycerol.method, 'vdi_ppds_11_dippr_eq106_surface_tension')
        self.assertClose(glycerol.value, 0.0636450900639547)
        self.assertClose(glycerol.quality, 0.95)
        self.assertIn('56-81-5', glycerol.notes)

    def test_supported_surface_tension_equation_forms(self):
        resolver = PropertyResolver()
        correlations = {
            'jasper': {
                'equation': 'Jasper',
                'coefficients': {'a': 24.05, 'b': 0.0832},
            },
            'somayajulu': {
                'equation': 'Somayajulu',
                'coefficients': {
                    'Tc': 513.92,
                    'A': 111.4452,
                    'B': -146.0229,
                    'C': 89.57030,
                },
            },
            'refprop': {
                'equation': 'REFPROP_sigma',
                'coefficients': {
                    'Tc': 513.9,
                    'sigma0': 0.05,
                    'n0': 0.952,
                    'sigma1': 0.0,
                    'n1': 0.0,
                    'sigma2': 0.0,
                    'n2': 0.0,
                },
            },
            'dippr': {
                'equation': 'DIPPR_EQ106',
                'coefficients': {
                    'Tc': 513.9,
                    'A': 0.0786269,
                    'B': 1.28646,
                    'C': -0.112304,
                    'D': 0.0,
                    'E': 0.0,
                },
            },
        }

        values = {
            name: resolver._evaluate_surface_tension_correlation(correlation, 298.15)
            for name, correlation in correlations.items()
        }

        self.assertClose(values['jasper'], 0.02197)
        self.assertGreater(values['somayajulu'], 0.0)
        self.assertGreater(values['refprop'], 0.0)
        self.assertGreater(values['dippr'], 0.0)

    def test_provided_poly_x_and_constant_overrides_precede_local_data(self):
        resolver = PropertyResolver()
        poly_props = {
            'CAS': '64-17-5',
            'name': 'ethanol',
            'property_correlations': {
                'sigma': {
                    'equation': 'poly_x',
                    'coefficients': {'A': 0.02, 'B': 0.005},
                    'Tmin_K': 159.15,
                    'Tmax_K': 351.15,
                    'source': 'test override',
                },
            },
        }
        constant_props = {
            'CAS': '64-17-5',
            'surface_tension': 0.04,
        }

        polynomial = resolver.resolve_surface_tension('ethanol', 308.15, poly_props)
        constant = resolver.resolve_surface_tension('ethanol', 298.15, constant_props)

        self.assertEqual(polynomial.method, 'provided_surface_tension_fit')
        self.assertClose(polynomial.value, 0.0205)
        self.assertIn('range 159.15-351.15 K', polynomial.notes)
        self.assertEqual(constant.method, 'constant_surface_tension')
        self.assertClose(constant.value, 0.04)

    def test_pfd_dippr_override_round_trips_and_feeds_resolver(self):
        source = (
            'PROCESS: Surface Tension Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            '\n'
            'COMPONENTS:\n'
            '    XCOR | Surface tension test | MW=50.0, Tc=600.0, Pc=30.0, Tb=350.0\n'
            '\n'
            'PROPERTY_CORRELATIONS:\n'
            '    XCOR.sigma | equation=DIPPR_EQ106, Tmin_K=250.0, Tmax_K=500.0, Tc_K=600.0, A=0.08, B=1.2, C=0.1\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = XCOR:1.0\n'
        )
        parsed = PFDParser().parse(source)
        correlation = parsed.components[0].property_correlations['sigma']
        reparsed = PFDParser().parse(parsed.to_pfd())

        self.assertEqual(correlation['equation'], 'DIPPR_EQ106')
        self.assertEqual(correlation['Tc_K'], 600.0)
        self.assertEqual(correlation['coefficients'], {'A': 0.08, 'B': 1.2, 'C': 0.1})
        self.assertEqual(
            reparsed.components[0].property_correlations['sigma'],
            correlation,
        )

        with patch('chemical_properties.OnlinePropertyFetcher.fetch_from_pubchem', return_value=None):
            simulation = Simulator.from_string(source)
            result = simulation.run()
        self.assertTrue(result.converged)
        known = simulation.thermo._resolver_known_props['XCOR']
        surface_tension = PropertyResolver().resolve_surface_tension('XCOR', 300.0, known)
        Tr = 300.0 / 600.0
        expected = 0.08 * (1.0 - Tr) ** (1.2 + 0.1 * Tr)

        self.assertEqual(surface_tension.method, 'provided_surface_tension_fit')
        self.assertClose(surface_tension.value, expected)
        self.assertEqual(surface_tension.quality, 1.0)

    def test_knotts_table_contains_all_perry_entries(self):
        payload = json.loads(
            (ROOT / 'data' / 'perry_surface_tension_parachor_groups.json').read_text()
        )
        groups = payload['groups']

        self.assertEqual(len(groups), 100)
        self.assertEqual(groups['nonring_ch3'], 55.25)
        self.assertEqual(groups['secondary_secondary_adjacency'], -2.73)
        self.assertEqual(groups['secondary_tertiary_adjacency'], -3.61)
        self.assertEqual(groups['tertiary_tertiary_adjacency'], -6.10)
        self.assertEqual(groups['aromatic_para_correction'], 3.40)

    def test_knotts_fragmentation_detects_carbon_ring_and_aromatic_corrections(self):
        resolver = PropertyResolver()
        cases = {
            'CCC#C': {
                'nonring_ch3': 1,
                'nonring_ch2_1_11': 1,
                'alkyne_c': 1,
                'alkyne_ch': 1,
            },
            'CC(C)C': {
                'nonring_ch3': 3,
                'nonring_ch': 1,
                'branch': 1,
            },
            'C1CCCCC1': {
                'ring_ch2': 6,
                'ring_6_correction': 1,
            },
            'Cc1ccc(C)cc1': {
                'aromatic_ch': 4,
                'aromatic_c': 2,
                'nonring_ch3': 2,
                'aromatic_para_correction': 1,
            },
            'Cc1cccc2ccccc12': {
                'aromatic_ch': 7,
                'aromatic_c': 1,
                'fused_aromatic_aromatic_c': 2,
                'nonring_ch3': 1,
                'substituted_naphthalene_correction': 1,
            },
        }

        for smiles, expected in cases.items():
            with self.subTest(smiles=smiles):
                fragmented = resolver._knotts_fragment_molecule(Chem.MolFromSmiles(smiles))
                self.assertIsNotNone(fragmented)
                groups, _note = fragmented
                self.assertEqual(dict(groups), expected)

    def test_knotts_fragmentation_detects_polyfunctional_and_heteroatom_groups(self):
        resolver = PropertyResolver()
        cases = {
            'OCC(O)CO': {
                'alcohol_primary_oh': 2,
                'alcohol_secondary_oh': 1,
            },
            'CCOC(=O)C': {'ester': 1},
            'CC(=O)N': {'amide_primary': 1},
            'C[N+](=O)[O-]': {'nitro': 1},
            'CS(C)=O': {'sulfoxide_nonring': 1},
            'Clc1ccccc1': {'aromatic_chlorine': 1},
        }

        for smiles, expected_subset in cases.items():
            with self.subTest(smiles=smiles):
                fragmented = resolver._knotts_fragment_molecule(Chem.MolFromSmiles(smiles))
                self.assertIsNotNone(fragmented)
                groups, _note = fragmented
                for group, count in expected_subset.items():
                    self.assertEqual(groups[group], count)

    def test_knotts_fragmentation_requires_a_carbon_skeleton(self):
        resolver = PropertyResolver()

        for smiles in ('[N]=O', '[O]N=O', 'N#[N+][O-]', 'N(=O)F', 'N#N'):
            with self.subTest(smiles=smiles):
                self.assertIsNone(
                    resolver._knotts_fragment_molecule(Chem.MolFromSmiles(smiles))
                )

        organic_nitro = resolver._knotts_fragment_molecule(
            Chem.MolFromSmiles('C[N+](=O)[O-]')
        )
        self.assertIsNotNone(organic_nitro)
        self.assertEqual(organic_nitro[0]['nitro'], 1)

    def test_knotts_fragmentation_rejects_multiple_halogens_on_one_atom(self):
        resolver = PropertyResolver()

        for smiles in ('C(F)F', 'CC(Cl)Cl', 'C(Br)(Br)(Br)Br', 'FC(F)(Cl)Br'):
            with self.subTest(smiles=smiles):
                self.assertIsNone(
                    resolver._knotts_fragment_molecule(Chem.MolFromSmiles(smiles))
                )

    def test_knotts_fragmentation_uses_aromatic_fluorine_for_fluorobenzene(self):
        resolver = PropertyResolver()

        fragmented = resolver._knotts_fragment_molecule(
            Chem.MolFromSmiles('Fc1ccccc1')
        )

        self.assertIsNotNone(fragmented)
        groups, _note = fragmented
        self.assertEqual(groups['aromatic_fluorine'], 1)
        self.assertEqual(groups['fluorine'], 0)

    def test_knotts_branch_adjacency_corrections_are_counted_once_per_bond(self):
        resolver = PropertyResolver()
        cases = {
            'CC(C)C(C)C': {
                'branch': 2,
                'secondary_secondary_adjacency': 1,
            },
            'CC(C)C(C)(C)C': {
                'branch': 3,
                'secondary_tertiary_adjacency': 1,
            },
            'CC(C)(C)C(C)(C)C': {
                'branch': 4,
                'tertiary_tertiary_adjacency': 1,
            },
            'CC(C)C(C)C(C)C': {
                'branch': 3,
                'secondary_secondary_adjacency': 2,
            },
            'CC(C)(C)C(C)C(C)(C)C': {
                'branch': 5,
                'secondary_tertiary_adjacency': 2,
            },
        }

        for smiles, expected in cases.items():
            with self.subTest(smiles=smiles):
                groups, note = resolver._knotts_fragment_molecule(
                    Chem.MolFromSmiles(smiles)
                )
                for group, count in expected.items():
                    self.assertEqual(groups[group], count)
                self.assertIn('branch-adjacency', note)

    def test_perry_ethylacetylene_parachor_is_reproduced(self):
        resolver = PropertyResolver()
        groups, _note = resolver._knotts_fragment_molecule(
            Chem.MolFromSmiles('CCC#C')
        )
        contributions = resolver._knotts_group_contributions()
        parachor = sum(
            count * contributions[group]
            for group, count in groups.items()
        )

        self.assertClose(parachor, 167.45)
        sigma = (parachor * 13.2573 / 1000.0) ** 4 / 1000.0
        self.assertClose(sigma, 0.02428627698001289)

    def test_knotts_fragmentation_cache_preserves_groups_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            first = PropertyResolver()
            second = PropertyResolver()
            with patch.object(first, 'CACHE_DIR', Path(directory)), \
                 patch.object(second, 'CACHE_DIR', Path(directory)):
                original = first._knotts_fragmentation('ethanol', {'smiles': 'CCO'})
                cached = second._knotts_fragmentation('ethanol', {'smiles': 'CCO'})

        self.assertIsNotNone(original)
        self.assertIsNotNone(cached)
        self.assertEqual(original[0], cached[0])
        self.assertEqual(original[2], cached[2])

    def test_knotts_vapor_density_uses_ptv_only_with_reliable_zc(self):
        resolver = PropertyResolver()
        base = {
            'Tc': PropertyResolutionResult(513.9, 'local', 'test', 0.98),
            'Pc': PropertyResolutionResult(61.48, 'local', 'test', 0.98),
            'omega': PropertyResolutionResult(0.644, 'local', 'test', 0.98),
        }
        psat = PropertyResolutionResult(0.0787, 'local', 'test', 0.98)
        hard = {
            **base,
            'Zc': PropertyResolutionResult(0.241, 'local', 'test', 0.95),
        }
        soft = {
            **base,
            'Zc': PropertyResolutionResult(0.241, 'estimated', 'test', 0.95),
        }

        with patch.object(resolver, 'resolve_vapor_pressure', return_value=psat):
            ptv = resolver._knotts_saturated_vapor_density('ethanol', {}, 298.15, hard)
            pr = resolver._knotts_saturated_vapor_density('ethanol', {}, 298.15, soft)

        self.assertEqual(ptv.method, 'ptv_eos_saturated_vapor_density')
        self.assertEqual(pr.method, 'pr_eos_saturated_vapor_density')
        self.assertGreater(ptv.value, 0.0)
        self.assertGreater(pr.value, 0.0)

    def test_knotts_vapor_density_quality_gates_eos_inputs(self):
        resolver = PropertyResolver()
        psat = PropertyResolutionResult(0.0787, 'local', 'test', 0.98)
        threshold_inputs = {
            'Tc': PropertyResolutionResult(513.9, 'local', 'test', 0.80),
            'Pc': PropertyResolutionResult(61.48, 'local', 'test', 0.80),
            'omega': PropertyResolutionResult(0.644, 'local', 'test', 0.70),
        }

        with patch.object(resolver, 'resolve_vapor_pressure', return_value=psat):
            at_threshold = resolver._knotts_saturated_vapor_density(
                'ethanol', {}, 298.15, threshold_inputs
            )
        self.assertEqual(at_threshold.method, 'pr_eos_saturated_vapor_density')

        low_quality_cases = {
            'Tc': (0.79, 0.80, 0.70),
            'Pc': (0.80, 0.79, 0.70),
            'omega': (0.80, 0.80, 0.69),
        }
        for input_name, qualities in low_quality_cases.items():
            with self.subTest(input=input_name):
                critical = {
                    'Tc': PropertyResolutionResult(513.9, 'local', 'test', qualities[0]),
                    'Pc': PropertyResolutionResult(61.48, 'local', 'test', qualities[1]),
                    'omega': PropertyResolutionResult(0.644, 'local', 'test', qualities[2]),
                }
                result = resolver._knotts_saturated_vapor_density(
                    'ethanol', {}, 298.15, critical
                )
                self.assertEqual(result.value, 0.0)
                self.assertEqual(
                    result.method,
                    'zero_saturated_vapor_density_fallback',
                )
                self.assertClose(result.quality, 0.60)

    def test_knotts_vapor_density_zero_is_final_eos_fallback(self):
        resolver = PropertyResolver()
        critical = {
            'Tc': PropertyResolutionResult(513.9, 'local', 'test', 0.98),
            'Pc': PropertyResolutionResult(61.48, 'local', 'test', 0.98),
            'omega': PropertyResolutionResult(0.644, 'local', 'test', 0.98),
        }
        psat = PropertyResolutionResult(0.0787, 'local', 'test', 0.98)

        with patch.object(resolver, 'resolve_vapor_pressure', return_value=psat), \
             patch.object(resolver, '_knotts_pr_vapor_density', return_value=None):
            result = resolver._knotts_saturated_vapor_density(
                'ethanol', {}, 298.15, critical
            )

        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.method, 'zero_saturated_vapor_density_fallback')
        self.assertIn('PR vapor roots were unavailable', result.notes)

    def test_knotts_parachor_works_without_critical_properties(self):
        resolver = PropertyResolver()
        liquid_density = PropertyResolutionResult(10.0, 'local', 'test', 0.98)
        structure = PropertyResolutionResult('C', 'provided', 'test', 1.0)

        with patch.object(resolver, 'resolve_critical_properties', return_value={}), \
             patch.object(resolver, 'resolve_liquid_molar_density', return_value=liquid_density), \
             patch.object(
                 resolver,
                 '_knotts_fragmentation',
                 return_value=(Counter({'test_group': 1}), structure, 'test mapping'),
             ), \
             patch.object(resolver, '_knotts_group_contributions', return_value={'test_group': 200.0}):
            result = resolver._knotts_parachor_surface_tension('test', {}, 300.0)

        expected = (200.0 * 10.0 / 1000.0) ** 4 / 1000.0
        self.assertClose(result.value, expected)
        self.assertClose(result.quality, 0.50)
        self.assertIn('Tr unavailable', result.notes)
        self.assertIn('rhoV=0', result.notes)
        self.assertIn('zero_saturated_vapor_density_fallback', result.notes)

    def test_knotts_parachor_quality_uses_density_and_vapor_method_penalties(self):
        resolver = PropertyResolver()
        tc = PropertyResolutionResult(500.0, 'local', 'test', 0.95)

        cases = {
            'ptv_high_rhol': (0.98, 'ptv_eos_saturated_vapor_density', tc, 0.80),
            'ptv_low_rhol': (0.85, 'ptv_eos_saturated_vapor_density', tc, 0.73),
            'pr_high_rhol': (0.98, 'pr_eos_saturated_vapor_density', tc, 0.78),
            'pr_low_rhol': (0.85, 'pr_eos_saturated_vapor_density', tc, 0.71),
            'zero_high_rhol': (
                0.98,
                'zero_saturated_vapor_density_fallback',
                tc,
                0.70,
            ),
            'missing_tc': (
                0.40,
                'zero_saturated_vapor_density_fallback',
                None,
                0.50,
            ),
        }

        for name, (rho_quality, vapor_method, tc_result, expected) in cases.items():
            with self.subTest(name=name):
                liquid_density = PropertyResolutionResult(
                    10.0, 'calculated', 'test', rho_quality
                )
                vapor_density = PropertyResolutionResult(
                    0.1, 'calculated', vapor_method, 0.01
                )
                quality = resolver._knotts_parachor_quality(
                    liquid_density,
                    vapor_density,
                    tc_result,
                )
                self.assertClose(quality, expected)

    def test_knotts_parachor_uses_liquid_minus_vapor_density(self):
        resolver = PropertyResolver()
        critical = {
            'Tc': PropertyResolutionResult(500.0, 'local', 'test', 0.98),
        }
        liquid_density = PropertyResolutionResult(10.0, 'local', 'test', 0.98)
        vapor_density = PropertyResolutionResult(
            1.0,
            'calculated',
            'ptv_eos_saturated_vapor_density',
            0.90,
        )
        structure = PropertyResolutionResult('C', 'provided', 'test', 1.0)

        with patch.object(resolver, 'resolve_critical_properties', return_value=critical), \
             patch.object(resolver, 'resolve_liquid_molar_density', return_value=liquid_density), \
             patch.object(resolver, '_knotts_saturated_vapor_density', return_value=vapor_density), \
             patch.object(
                 resolver,
                 '_knotts_fragmentation',
                 return_value=(Counter({'test_group': 1}), structure, 'test mapping'),
             ), \
             patch.object(resolver, '_knotts_group_contributions', return_value={'test_group': 200.0}):
            result = resolver._knotts_parachor_surface_tension('test', {}, 300.0)

        expected = (200.0 * (10.0 - 1.0) / 1000.0) ** 4 / 1000.0
        self.assertClose(result.value, expected)
        self.assertClose(result.quality, 0.80)
        self.assertIn('rhoL=10', result.notes)
        self.assertIn('rhoV=1', result.notes)

    def test_knotts_parachor_is_fallback_after_fitted_correlations(self):
        resolver = PropertyResolver()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(resolver, 'CACHE_DIR', Path(directory)), \
             patch.object(resolver, '_surface_tension_from_local_correlations', return_value=None):
            surface_tension = resolver.resolve_surface_tension('ethanol', 298.15)

        self.assertEqual(surface_tension.method, 'knotts_parachor_group_contribution')
        # Liquid density feeding the parachor now comes from the CoolProp
        # reference EOS for ethanol rather than the Perry fit.
        self.assertClose(
            surface_tension.value,
            0.021639288944768516,
            rel=1.0e-6,
        )
        self.assertClose(surface_tension.quality, 0.80)
        self.assertIn('ptv_eos_saturated_vapor_density', surface_tension.notes)
        self.assertIn('1 alcohol_primary_oh', surface_tension.notes)

    def test_knotts_parachor_rejects_near_critical_state(self):
        resolver = PropertyResolver()
        props = {'Tc': 500.0, 'Pc': 40.0, 'omega': 0.2, 'Zc': 0.25}

        result = resolver._knotts_parachor_surface_tension('test', props, 450.0)

        self.assertIsNone(result)

    def test_surface_tension_rejects_invalid_temperature(self):
        resolver = PropertyResolver()

        for temperature in (0.0, -1.0, math.inf, math.nan, 'warm'):
            with self.subTest(temperature=temperature):
                with self.assertRaises(PropertyResolutionError):
                    resolver.resolve_surface_tension('water', temperature)


if __name__ == '__main__':
    unittest.main()
