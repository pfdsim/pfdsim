import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms

from pfd_parser import parse_pfd
from property_resolution.common import PropertyResolutionResult
from property_resolver import PropertyResolver
from simulator import SimulationError, Simulator


class RadiusOfGyrationResolutionTests(unittest.TestCase):
    def setUp(self):
        self.resolver = PropertyResolver()

    def test_linear_principal_moment_definition(self):
        atoms = Atoms('H2', positions=[[-0.37, 0.0, 0.0], [0.37, 0.0, 0.0]])
        conventional, modified, linear = self.resolver._radii_from_atoms(atoms)
        self.assertTrue(linear)
        self.assertAlmostEqual(conventional, 0.37)
        self.assertAlmostEqual(modified, 0.37)

    def test_nonlinear_radii_are_translation_invariant(self):
        atoms = Atoms(
            'OH2',
            positions=[
                [0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.2399872, 0.927297, 0.0],
            ],
        )
        original = self.resolver._radii_from_atoms(atoms)
        translated = atoms.copy()
        translated.positions += np.array([8.0, -3.0, 4.5])
        shifted = self.resolver._radii_from_atoms(translated)
        self.assertFalse(original[2])
        self.assertGreater(original[0], 0.0)
        self.assertGreater(original[1], 0.0)
        self.assertAlmostEqual(original[0], shifted[0])
        self.assertAlmostEqual(original[1], shifted[1])

    def test_both_resolvers_share_one_cached_geometry_evaluation(self):
        smiles_result = PropertyResolutionResult(
            value='O',
            source='test',
            method='test_smiles',
            quality=1.0,
        )
        atoms = Atoms(
            'OH2',
            positions=[
                [0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.2399872, 0.927297, 0.0],
            ],
        )
        with (
            patch.object(
                self.resolver,
                '_resolve_smiles_result',
                return_value=smiles_result,
            ),
            patch.object(
                self.resolver,
                '_load_dipole_artifact',
                return_value=None,
            ),
            patch.object(
                self.resolver,
                '_backend_is_available',
                return_value=True,
            ),
            patch.object(
                self.resolver,
                '_resolve_xtb_geometry',
                return_value=(atoms, 0, 1),
            ) as geometry,
        ):
            results = self.resolver.resolve_radii_of_gyration(
                'water test',
                {'name': 'water test', 'smiles': 'O'},
                allow_online=False,
            )

        self.assertEqual(geometry.call_count, 1)
        self.assertGreater(results['radius_of_gyration'].value, 0.0)
        self.assertGreater(results['modified_radius_of_gyration'].value, 0.0)
        self.assertEqual(results['radius_of_gyration'].quality, 0.85)
        self.assertEqual(results['modified_radius_of_gyration'].quality, 0.85)
        self.assertIn('4% MAE', results['modified_radius_of_gyration'].notes)

    def test_pfd_overrides_round_trip_with_canonical_names(self):
        pfd = parse_pfd(
            'COMPONENTS:\n'
            '    X | example | R=1.25, R_HOC=2.5\n'
        )
        component = pfd.components[0]
        self.assertEqual(component.radius_of_gyration, 1.25)
        self.assertEqual(component.modified_radius_of_gyration, 2.5)
        rendered = pfd.to_pfd()
        self.assertIn('R=1.25', rendered)
        self.assertIn('R_prime=2.5', rendered)

    def test_invalid_pfd_radius_is_rejected(self):
        with self.assertRaisesRegex(SimulationError, 'finite nonnegative value'):
            Simulator.from_string(
                'COMPONENTS:\n'
                '    X | example | modified_radius_of_gyration=-1\n'
            )

    def test_explicit_override_has_provenance_and_units(self):
        result = self.resolver.resolve_modified_radius_of_gyration(
            'example',
            {
                'modified_radius_of_gyration': 2.4,
                'property_sources': {
                    'modified_radius_of_gyration': {
                        'source': 'provided',
                        'method': 'pfd_component_override',
                        'quality': 1.0,
                    },
                },
            },
            allow_online=False,
        )
        self.assertEqual(result.value, 2.4)
        self.assertEqual(result.method, 'pfd_component_override')
        self.assertIn('angstrom', result.notes)

    def test_pfd_overrides_reach_thermodynamic_property_snapshot(self):
        simulator = Simulator.from_string(
            'PROCESS: Gyration Override Integration\n'
            'ONLINE_LOOKUP: false\n'
            'COMPONENTS:\n'
            '    X | methane | Rg=1.1, R_prime=1.3\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 300 [K]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = X:1\n'
        ).initialize()
        props = simulator.thermo.props['X']
        known = simulator.thermo._resolver_known_props['X']
        self.assertEqual(props.radius_of_gyration, 1.1)
        self.assertEqual(props.modified_radius_of_gyration, 1.3)
        self.assertEqual(known['radius_of_gyration'], 1.1)
        self.assertEqual(known['modified_radius_of_gyration'], 1.3)
        self.assertEqual(
            known['property_sources']['modified_radius_of_gyration']['method'],
            'pfd_component_override',
        )


if __name__ == '__main__':
    unittest.main()
