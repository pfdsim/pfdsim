import json
import math
import sqlite3
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from chemicals.heat_capacity import (
    Lastovka_solid, Perry_151,
)
from chemicals import heat_capacity as source_heat_capacity

from chemical_properties import ChemicalDatabase, ChemicalProperties
from pfd_parser import parse_pfd, validate_pfd
from property_resolver import PropertyResolver
from property_resolution import PropertyResolutionError, PropertyResolutionResult
from property_resolution.solid_cp import (
    ConstantSolidCpKernel, LastovkaSolidCpKernel,
    ModifiedKoppSolidCpKernel, Perry151SolidCpKernel,
    PiecewiseSolidCpKernel, PolynomialSolidCpKernel,
    ShomateSolidCpKernel, SolidCpCollectionKernel,
    SolidCpTransitionError, TabularSolidCpKernel,
    clear_bundled_solid_kernel_cache, load_bundled_solid_kernel,
    solid_kernel_from_payload,
)
from property_resolution.solid_volume import (
    clear_bundled_solid_volume_cache, load_bundled_solid_volume,
)
from thermodynamics import create_thermodynamics
from simulator import Simulator


ROOT = Path(__file__).resolve().parents[1]
CP_DATABASE = ROOT / 'data' / 'solid_heat_capacity.sqlite'
VOLUME_DATABASE = ROOT / 'data' / 'solid_volume.sqlite'


class SolidCpKernelTests(unittest.TestCase):
    def assertClose(self, actual, expected, rel=1e-9, abs_tol=1e-9):
        self.assertLessEqual(abs(actual - expected), max(abs_tol, rel * max(abs(expected), 1.0)))

    def common(self, **overrides):
        values = dict(
            Tmin=200.0, Tmax=500.0, quality=0.95,
            source='fixture', method='fixture_solid_cp',
            material_form='crystalline', polymorph='',
        )
        values.update(overrides)
        return values

    def assertPrimitiveDerivatives(self, kernel, T, rel=2e-6):
        step = max(1e-3, 1e-5 * T)
        cp = kernel.cp(T)
        dH = kernel.delta_h(T - step, T + step) / (2.0 * step)
        dS = kernel.delta_s(T - step, T + step) / (2.0 * step)
        self.assertClose(dH, cp, rel=rel)
        self.assertClose(dS, cp / T, rel=rel)

    def test_constant_polynomial_shomate_and_perry_primitives(self):
        kernels = [
            ConstantSolidCpKernel(**self.common(), value=80.0),
            PolynomialSolidCpKernel(**self.common(), coefficients=(20.0, 0.2, 1e-4)),
            ShomateSolidCpKernel(
                **self.common(), coefficients=(30.0, 20.0, 3.0, -1.0, 0.2),
            ),
            Perry151SolidCpKernel(
                **self.common(), coefficients=(4.8, 0.00322, -1000.0, 1e-7),
            ),
        ]
        for kernel in kernels:
            with self.subTest(kind=kernel.kind):
                self.assertPrimitiveDerivatives(kernel, 300.0)
                restored = solid_kernel_from_payload(kernel.to_payload())
                self.assertClose(restored.cp(300.0), kernel.cp(300.0))
        self.assertClose(
            kernels[-1].cp(300.0),
            Perry_151(300.0, 4.8, 0.00322, -1000.0, 1e-7),
        )

    def test_tabular_kernel_has_exact_linear_primitives(self):
        kernel = TabularSolidCpKernel(
            **self.common(Tmin=200.0, Tmax=400.0),
            temperatures=(200.0, 300.0, 400.0),
            values=(50.0, 80.0, 130.0),
        )
        self.assertClose(kernel.cp(250.0), 65.0)
        self.assertClose(kernel.delta_h(200.0, 400.0), 17000.0)
        self.assertPrimitiveDerivatives(kernel, 350.0)
        restored = solid_kernel_from_payload(kernel.to_payload())
        self.assertClose(restored.delta_s(225.0, 375.0), kernel.delta_s(225.0, 375.0))

    def test_piecewise_continuous_integrates_and_transition_refuses(self):
        first = PolynomialSolidCpKernel(
            **self.common(Tmin=200.0, Tmax=300.0), coefficients=(20.0, 0.1),
        )
        second = PolynomialSolidCpKernel(
            **self.common(Tmin=300.0, Tmax=400.0), coefficients=(35.0, 0.05),
        )
        continuous = PiecewiseSolidCpKernel(
            **self.common(Tmin=200.0, Tmax=400.0),
            segments=(first, second), transitions=(),
        )
        expected = first.delta_h(250.0, 300.0) + second.delta_h(300.0, 350.0)
        self.assertClose(continuous.delta_h(250.0, 350.0), expected)

        transitioned = PiecewiseSolidCpKernel(
            **self.common(Tmin=200.0, Tmax=400.0),
            segments=(first, second), transitions=(300.0,),
        )
        with self.assertRaisesRegex(SolidCpTransitionError, 'transition enthalpy'):
            transitioned.delta_h(250.0, 350.0)
        with self.assertRaises(SolidCpTransitionError):
            transitioned.delta_s(250.0, 350.0)

    def test_collection_selects_quality_and_preserves_physical_transition(self):
        weak = ConstantSolidCpKernel(
            **self.common(Tmin=200.0, Tmax=400.0, quality=0.80, source_priority=9),
            value=90.0,
        )
        low = ConstantSolidCpKernel(
            **self.common(Tmin=200.0, Tmax=300.0, quality=0.98, source_priority=1),
            value=80.0,
        )
        high = ConstantSolidCpKernel(
            **self.common(Tmin=300.0, Tmax=400.0, quality=0.98, source_priority=1),
            value=100.0,
        )
        primary = PiecewiseSolidCpKernel(
            **self.common(Tmin=200.0, Tmax=400.0, quality=0.98, source_priority=1),
            segments=(low, high), transitions=(300.0,),
        )
        collection = SolidCpCollectionKernel(
            **self.common(Tmin=200.0, Tmax=400.0, quality=0.98),
            candidates=(weak, primary),
        )
        self.assertEqual(collection.cp(250.0), 80.0)
        with self.assertRaises(SolidCpTransitionError):
            collection.delta_h(250.0, 350.0)

    def test_lastovka_matches_reference_equation_and_primitives(self):
        alpha = 22.0 / 154.21
        kernel = LastovkaSolidCpKernel(
            **self.common(Tmin=100.0, Tmax=350.0, quality=0.70),
            similarity_variable=alpha, molecular_weight=154.21,
        )
        self.assertClose(kernel.cp(300.0), Lastovka_solid(300.0, alpha, MW=154.21))
        self.assertPrimitiveDerivatives(kernel, 300.0, rel=1e-5)

    def test_modified_kopp_reproduces_perry_dibenzothiophene_example(self):
        counts = (('C', 12.0), ('H', 8.0), ('S', 1.0))
        value = 12 * 10.89 + 8 * 7.56 + 12.36
        kernel = ModifiedKoppSolidCpKernel(
            **self.common(Tmin=293.15, Tmax=303.15, quality=0.72),
            value=value, atom_counts=counts,
        )
        self.assertClose(kernel.cp(298.15), 203.52)
        self.assertEqual(solid_kernel_from_payload(kernel.to_payload()).atom_counts, counts)

    def test_solid_range_is_strict_not_far_clamped(self):
        kernel = ConstantSolidCpKernel(**self.common(Tmin=293.15, Tmax=303.15), value=100.0)
        self.assertEqual(kernel.cp(308.15), 100.0)
        self.assertLess(kernel.quality_at(308.15), kernel.quality)
        with self.assertRaisesRegex(ValueError, 'source range'):
            kernel.cp(309.0)


class SolidDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clear_bundled_solid_kernel_cache()
        clear_bundled_solid_volume_cache()

    def test_database_integrity_counts_sources_and_quarantines(self):
        with sqlite3.connect(CP_DATABASE) as connection:
            self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(connection.execute('SELECT count(*) FROM canonical_solid_cp').fetchone()[0], 1508)
            self.assertEqual(connection.execute('SELECT count(DISTINCT cas) FROM canonical_solid_cp').fetchone()[0], 837)
            self.assertEqual(connection.execute('SELECT count(*) FROM source_quarantine').fetchone()[0], 23)
            self.assertEqual(dict(connection.execute(
                'SELECT status,count(*) FROM source_candidate_audit GROUP BY status'
            )), {'admitted': 1508, 'quarantined': 23})
            sources = dict(connection.execute(
                'SELECT source,count(*) FROM canonical_solid_cp GROUP BY source'
            ))
            self.assertEqual(sources, {
                'crc_standard_point': 529,
                'crc_temperature_table': 42,
                'janaf_1998': 340,
                'perry_151': 261,
                'perry_151_point': 6,
                'webbook_shomate': 330,
            })
            self.assertEqual(connection.execute(
                "SELECT count(*) FROM source_quarantine WHERE reason='source formula conflicts with CAS identity'"
            ).fetchone()[0], 19)
            self.assertGreater(connection.execute(
                'SELECT min(Tmin_K) FROM canonical_solid_cp'
            ).fetchone()[0], 0.0)
        with sqlite3.connect(VOLUME_DATABASE) as connection:
            self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(connection.execute('SELECT count(*) FROM canonical_solid_volume').fetchone()[0], 1872)
            self.assertEqual(connection.execute(
                "SELECT count(*) FROM canonical_solid_volume WHERE material_form='hydrate'"
            ).fetchone()[0], 310)

    def test_graphite_identity_is_corrected_and_diamond_is_separate(self):
        with sqlite3.connect(CP_DATABASE) as connection:
            self.assertEqual(connection.execute(
                "SELECT count(*) FROM canonical_solid_cp WHERE cas='74-82-8' AND source='crc_temperature_table'"
            ).fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT count(*) FROM canonical_solid_cp WHERE cas='7782-42-5' AND source='crc_temperature_table'"
            ).fetchone()[0], 1)
        graphite = load_bundled_solid_kernel('7782-42-5')
        diamond = load_bundled_solid_kernel('7782-40-3')
        self.assertAlmostEqual(graphite.cp(298.15), 8.517, places=3)
        self.assertAlmostEqual(diamond.cp(298.15), 6.1, places=3)

    def test_crystalline_and_glass_records_are_selectable(self):
        crystalline = load_bundled_solid_kernel('7631-86-9', material_form='crystalline')
        glass = load_bundled_solid_kernel('7631-86-9', material_form='glass')
        self.assertIsNotNone(crystalline)
        self.assertIsNotNone(glass)
        self.assertNotEqual(crystalline.method_at(300.0), glass.method_at(300.0))
        self.assertNotAlmostEqual(crystalline.cp(300.0), glass.cp(300.0), places=3)

    def test_polymorph_only_source_is_not_guessed(self):
        self.assertIsNone(load_bundled_solid_kernel('12141-45-6'))
        selected = load_bundled_solid_kernel(
            '12141-45-6', material_form='crystalline', polymorph='sillimanite',
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.polymorph, 'sillimanite')

    def test_hydrate_volume_is_identity_specific(self):
        hydrate = load_bundled_solid_volume('7758-99-8')
        self.assertIsNotNone(hydrate)
        self.assertEqual(hydrate.material_form, 'hydrate')
        self.assertIsNone(
            load_bundled_solid_volume('7758-99-8', material_form='crystalline')
        )

    def test_webbook_unit_conversion_matches_native_source(self):
        with sqlite3.connect(CP_DATABASE) as connection:
            row = connection.execute(
                "SELECT kernel_payload_json FROM canonical_solid_cp "
                "WHERE cas='7647-14-5' AND source='webbook_shomate'"
            ).fetchone()
        kernel = solid_kernel_from_payload(json.loads(row[0]))
        native = source_heat_capacity.WebBook_Shomate_solids['7647-14-5']
        for temperature in (298.15, 500.0, 800.0):
            self.assertAlmostEqual(kernel.cp(temperature), native.calculate(temperature), places=9)

    def test_transition_metadata_blocks_crossing_integral(self):
        barium = load_bundled_solid_kernel('7440-39-3')
        self.assertIn(582.53, barium.transition_temperatures)
        self.assertNotIn(583.0, barium.transition_temperatures)
        with self.assertRaises(SolidCpTransitionError):
            barium.delta_h(580.0, 585.0)

    def test_representative_canonical_values_and_volume(self):
        nacl = load_bundled_solid_kernel('7647-14-5')
        naoh = load_bundled_solid_kernel('1310-73-2')
        phenol = load_bundled_solid_kernel('108-95-2')
        self.assertAlmostEqual(nacl.cp(298.15), 50.509, places=3)
        self.assertTrue(nacl.source_fingerprint)
        self.assertAlmostEqual(naoh.cp(298.15), 59.53, places=2)
        self.assertAlmostEqual(phenol.cp(298.15), 127.4, places=2)
        self.assertAlmostEqual(
            load_bundled_solid_volume('7647-14-5').molar_volume_m3_per_kmol,
            0.0269322580645,
        )

    def test_every_canonical_payload_loads_and_has_consistent_primitives(self):
        failures = []
        with sqlite3.connect(CP_DATABASE) as connection:
            rows = connection.execute(
                'SELECT record_id,kernel_payload_json FROM canonical_solid_cp ORDER BY record_id'
            ).fetchall()
        for record_id, payload_json in rows:
            try:
                kernel = solid_kernel_from_payload(json.loads(payload_json))
                self.assertTrue(kernel.source_fingerprint)
                active = kernel.segments[0] if isinstance(kernel, PiecewiseSolidCpKernel) else kernel
                self.assertTrue(active.source_fingerprint)
                T = 0.5 * (active.Tmin + active.Tmax)
                cp = active.cp(T)
                step = min(0.01, 0.1 * (active.Tmax - active.Tmin))
                if step > 0.0:
                    derivative = active.delta_h(T - step, T + step) / (2.0 * step)
                    if abs(derivative / cp - 1.0) > 2e-5:
                        raise AssertionError(f'primitive mismatch {derivative} versus {cp}')
            except Exception as error:
                failures.append((record_id, str(error)))
        self.assertEqual(failures, [])


class SolidResolverTests(unittest.TestCase):
    def setUp(self):
        self.resolver = PropertyResolver()

    def test_canonical_solid_cp_and_density_resolve_offline(self):
        props = {'CAS': '7647-14-5', 'formula': 'NaCl', 'MW': 58.44277}
        cp = self.resolver.resolve_heat_capacity('NaCl', 298.15, 'solid', props, allow_online=False)
        density = self.resolver.resolve_solid_mass_density('NaCl', 298.15, props, allow_online=False)
        volume = self.resolver.resolve_solid_molar_volume('NaCl', 298.15, props, allow_online=False)
        self.assertEqual(cp.method, 'canonical_janaf_1998_solid_cp')
        self.assertAlmostEqual(cp.value, 50.509, places=3)
        self.assertEqual(volume.method, 'crc_solid_constant_molar_volume')
        self.assertEqual(density.method, 'solid_mass_density_from_molar_volume')
        self.assertAlmostEqual(density.value, 2169.99146, places=3)

    def test_provided_cps_correlation_precedes_scalar_and_canonical(self):
        props = {
            'CAS': '7647-14-5', 'formula': 'NaCl', 'MW': 58.44277,
            'Cp_solid': 999.0,
            'property_sources': {'Cp_solid': {'method': 'pfd_component_override'}},
            'property_correlations': {
                'Cps': {
                    'equation': 'perry_151', 'Tmin_K': 200.0, 'Tmax_K': 400.0,
                    'coefficients': {'A': 10.0, 'B': 0.01, 'C': 0.0, 'D': 0.0},
                    '_pfd_override': True,
                },
            },
        }
        kernel = self.resolver.resolve_solid_cp_kernel('NaCl', props, allow_online=False)
        self.assertEqual(kernel.method, 'provided_perry_151_solid_cp_kernel')
        self.assertAlmostEqual(kernel.cp(300.0), 4.184 * 13.0)

    def test_provided_solid_cp_equation_families(self):
        fixtures = {
            'poly_x': ({'A': 100.0, 'B': 10.0}, 100.185),
            'shomate': ({'A': 20.0, 'B': 10.0, 'C': 0.0, 'D': 0.0, 'E': 0.0}, 23.0),
        }
        for equation, (coefficients, expected) in fixtures.items():
            with self.subTest(equation=equation):
                props = {
                    'property_correlations': {
                        'Cps': {
                            'equation': equation, 'Tmin_K': 200.0, 'Tmax_K': 400.0,
                            'coefficients': coefficients, '_pfd_override': True,
                        },
                    },
                }
                kernel = self.resolver.resolve_solid_cp_kernel(
                    f'fixture {equation}', props, allow_online=False,
                )
                self.assertAlmostEqual(kernel.cp(300.0), expected, places=8)

    def test_explicit_constant_solid_cp_is_authoritative_and_unbounded(self):
        props = {
            'CAS': '7647-14-5', 'Cp_solid': 88.0,
            'property_sources': {'Cp_solid': {'method': 'pfd_component_override', 'source': 'PFD'}},
        }
        kernel = self.resolver.resolve_solid_cp_kernel('NaCl', props, allow_online=False)
        self.assertEqual(kernel.cp(50.0), 88.0)
        self.assertEqual(kernel.cp(1000.0), 88.0)
        self.assertEqual(kernel.quality, 1.0)

    def test_non_pfd_scalar_does_not_shadow_canonical_database(self):
        props = {
            'CAS': '7647-14-5', 'Cp_solid': 999.0,
            'property_sources': {'Cp_solid': {'source': 'dataset', 'quality': 0.95}},
        }
        kernel = self.resolver.resolve_solid_cp_kernel('NaCl', props, allow_online=False)
        self.assertAlmostEqual(kernel.cp(298.15), 50.509, places=3)

    def test_lastovka_remains_one_consistent_temperature_dependent_kernel(self):
        props = {'formula': 'C12H10', 'MW': 154.21, 'Tm': 350.0}
        with patch('property_resolution.heat_capacity.lookup_bundled_solid_cas', return_value=None):
            kernel = self.resolver.resolve_solid_cp_kernel(
                'synthetic hydrocarbon', props, allow_online=False,
            )
        self.assertIsInstance(kernel, LastovkaSolidCpKernel)
        self.assertEqual(kernel.method_at(298.15), 'lastovka_solid_cp_kernel')
        self.assertEqual(kernel.method_at(320.0), 'lastovka_solid_cp_kernel')
        self.assertPrimitiveIdentity(kernel, 298.15)

    def assertPrimitiveIdentity(self, kernel, temperature):
        step = 0.01
        derivative = kernel.delta_h(temperature - step, temperature + step) / (2.0 * step)
        self.assertAlmostEqual(derivative, kernel.cp(temperature), places=5)

    def test_modified_kopp_is_organic_point_fallback_without_lastovka_anchor(self):
        props = {'formula': 'C12H10', 'MW': 154.21}
        with patch('property_resolution.heat_capacity.lookup_bundled_solid_cas', return_value=None):
            kernel = self.resolver.resolve_solid_cp_kernel(
                'synthetic hydrocarbon', props, allow_online=False,
            )
        self.assertIsInstance(kernel, ModifiedKoppSolidCpKernel)
        self.assertAlmostEqual(kernel.cp(298.15), 12 * 10.89 + 10 * 7.56)

    def test_modified_kopp_supports_inorganic_formula_at_lower_quality(self):
        props = {'formula': 'Ca3P2', 'MW': 182.18}
        with patch('property_resolution.heat_capacity.lookup_bundled_solid_cas', return_value=None):
            kernel = self.resolver.resolve_solid_cp_kernel('fixture salt', props, allow_online=False)
        self.assertIsInstance(kernel, ModifiedKoppSolidCpKernel)
        self.assertAlmostEqual(kernel.cp(298.15), 3 * 28.25 + 2 * 26.63)
        self.assertEqual(kernel.quality, 0.64)
        with self.assertRaises(ValueError):
            kernel.cp(350.0)

    def test_modified_kopp_rejects_net_charged_formula(self):
        self.assertIsNone(self.resolver._modified_kopp_solid_cp_kernel(
            'sodium ion', {'formula': 'Na+', 'MW': 22.99},
        ))

    def test_lastovka_refuses_missing_transition_anchor(self):
        props = {'formula': 'C12H10', 'MW': 154.21}
        self.assertIsNone(self.resolver._lastovka_solid_cp_kernel('fixture', props))

    def test_explicit_density_and_volume_are_checked_for_consistency(self):
        props = {'MW': 100.0, 'rho_solid': 2000.0, 'Vm_solid': 0.05}
        density = self.resolver.resolve_solid_mass_density('fixture', 298.15, props, allow_online=False)
        volume = self.resolver.resolve_solid_molar_volume('fixture', 298.15, props, allow_online=False)
        self.assertEqual(density.value, 2000.0)
        self.assertEqual(volume.value, 0.05)
        with self.assertRaisesRegex(PropertyResolutionError, 'inconsistent'):
            self.resolver.resolve_solid_mass_density(
                'fixture', 298.15, {**props, 'Vm_solid': 0.10}, allow_online=False,
            )

    def test_provided_solid_density_correlation_has_priority(self):
        props = {
            'CAS': '7647-14-5', 'MW': 58.44277,
            'property_correlations': {
                'rhos': {
                    'equation': 'poly_x', 'Tmin_K': 250.0, 'Tmax_K': 350.0,
                    'coefficients': {'A': 2000.0, 'B': -10.0},
                    '_pfd_override': True,
                },
            },
        }
        result = self.resolver.resolve_solid_mass_density('NaCl', 298.15, props, allow_online=False)
        self.assertEqual(result.method, 'provided_solid_density_fit')
        self.assertAlmostEqual(result.value, 2000.0)

    def test_crc_solid_volume_uses_zero_expansion_with_quality_decay(self):
        props = {'CAS': '7647-14-5', 'MW': 58.44277}
        ambient = self.resolver.resolve_solid_molar_volume('NaCl', 298.15, props, allow_online=False)
        hot = self.resolver.resolve_solid_molar_volume('NaCl', 548.15, props, allow_online=False)
        self.assertEqual(hot.value, ambient.value)
        self.assertAlmostEqual(ambient.quality, 0.94)
        self.assertAlmostEqual(hot.quality, 0.84)
        self.assertIn('zero inorganic thermal expansion', hot.notes)

    def test_canonical_kernel_does_not_invent_absolute_zero_continuation(self):
        kernel = self.resolver.resolve_solid_cp_kernel(
            'NaCl', {'CAS': '7647-14-5', 'formula': 'NaCl', 'MW': 58.44277},
            allow_online=False,
        )
        with self.assertRaises(ValueError):
            kernel.cp(90.0)

    def test_solid_nist_cache_contract_is_phase_preserving_v3(self):
        key = self.resolver._nist_cp_cache_key('fixture')
        self.assertTrue(key.startswith('cp_nist_v3_'))

    def test_online_or_organic_density_path_remains_fallback(self):
        fallback = PropertyResolutionResult(1234.0, 'online', 'fixture_density', 0.80)
        with (
            patch.object(self.resolver, '_bundled_solid_volume', return_value=None),
            patch.object(
                self.resolver,
                '_resolve_observed_or_estimated_solid_mass_density',
                return_value=fallback,
            ) as resolver_fallback,
        ):
            result = self.resolver.resolve_solid_mass_density(
                'fixture', 298.15, {'MW': 100.0}, allow_online=True,
            )
        self.assertIs(result, fallback)
        resolver_fallback.assert_called_once()

    def test_nist_parser_preserves_solid_transition_and_shomate(self):
        html = '''
        <table class="data" aria-label="Constant pressure heat capacity of solid">
          <tr><th>C p,solid (J/mol*K)</th><th>Temperature (K)</th></tr>
          <tr><td>50</td><td>200</td></tr><tr><td>70</td><td>300</td></tr>
          <tr><td>90</td><td>300</td></tr><tr><td>110</td><td>400</td></tr>
        </table>
        <table class="data" aria-label="Solid Phase Heat Capacity (Shomate Equation)">
          <tr><th>Temperature (K)</th><th>400. to 800.</th></tr>
          <tr><td>A</td><td>20</td></tr><tr><td>B</td><td>10</td></tr>
          <tr><td>C</td><td>0</td></tr><tr><td>D</td><td>0</td></tr>
          <tr><td>E</td><td>0</td></tr>
        </table>'''
        tables = self.resolver._parse_nist_cp_tables(html)
        self.assertEqual(tables['solid'][1:3], [[300.0, 70.0], [300.0, 90.0]])
        self.assertIn('solid_shomate', tables)
        kernel = self.resolver._kernel_from_nist_solid_source(tables)
        self.assertAlmostEqual(kernel.cp(500.0), 25.0)

    def test_nist_tabular_solid_transition_is_not_median_merged(self):
        tables = {
            'solid': [
                [200.0, 50.0], [300.0, 70.0],
                [300.0, 90.0], [400.0, 110.0],
            ],
        }
        kernel = self.resolver._kernel_from_nist_solid_source(tables)
        self.assertIsInstance(kernel, PiecewiseSolidCpKernel)
        self.assertEqual(kernel.transition_temperatures, (300.0,))
        with self.assertRaises(SolidCpTransitionError):
            kernel.delta_h(250.0, 350.0)

    def test_pfd_solid_properties_and_correlations_round_trip(self):
        text = '''
PFD_VERSION: 1.0
PROCESS: Solid properties
COMPONENTS:
    X | Test solid | formula=C2H4O, MW=44.05, Cp_solid=88, rho_solid=1250, Vm_solid=0.03524, solid_material_form=amorphous, solid_polymorph=beta
PROPERTY_CORRELATIONS:
    X.Cps | equation=perry_151, A=10, B=0.01, C=0, D=0, Tmin=200, Tmax=400
    X.rhos | equation=poly_x, A=1200, B=-2, Tmin=200, Tmax=400
'''
        pfd = parse_pfd(text)
        component = pfd.components[0]
        self.assertEqual(component.solid_material_form, 'glass')
        self.assertEqual(component.solid_polymorph, 'beta')
        self.assertEqual(set(component.property_correlations), {'Cps', 'rhos'})
        restored = parse_pfd(pfd.to_pfd()).components[0]
        self.assertEqual(restored.Cp_solid, 88.0)
        self.assertEqual(restored.rho_solid, 1250.0)
        self.assertEqual(restored.Vm_solid, 0.03524)
        self.assertEqual(restored.solid_material_form, 'glass')

    def test_invalid_solid_material_form_is_rejected(self):
        with self.assertRaisesRegex(Exception, 'Unsupported solid material form'):
            parse_pfd('''PFD_VERSION: 1.0\nPROCESS: bad\nCOMPONENTS:\n    X | X | solid_material_form=plasma\n''')

    def test_pfd_validation_rejects_inconsistent_solid_density_pair(self):
        pfd = parse_pfd('''
PFD_VERSION: 1.0
PROCESS: bad density
COMPONENTS:
    X | X | MW=100, rho_solid=1000, Vm_solid=0.05
''')
        errors, _ = validate_pfd(pfd)
        self.assertTrue(any('Inconsistent solid density' in error for error in errors))

    def test_pfd_validation_rejects_nonpositive_solid_scalars(self):
        pfd = parse_pfd('''
PFD_VERSION: 1.0
PROCESS: bad solid scalar
COMPONENTS:
    X | X | Cp_solid=-1, rho_solid=0, Vm_solid=-2
''')
        errors, _ = validate_pfd(pfd)
        self.assertEqual(sum('Invalid' in error for error in errors), 3)

    def test_simulator_hydrates_pfd_solid_overrides_into_thermodynamics(self):
        text = '''
PFD_VERSION: 1.0
PROCESS: Solid property integration
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
COMPONENTS:
    X | Test solid | formula=C2H4O, MW=44.05, Cp_solid=88, rho_solid=1250, Vm_solid=0.03524, solid_material_form=glass
'''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            simulator = Simulator.from_string(text).initialize()
        props = simulator.thermo.props['X']
        self.assertEqual(props.solid_material_form, 'glass')
        self.assertEqual(simulator.thermo.Cp_solid('X', 500.0), 88.0)
        self.assertEqual(props.solid_mass_density(298.15), 1250.0)


class SolidPublicApiTests(unittest.TestCase):
    def test_chemical_properties_uses_resolved_kernel(self):
        props = ChemicalProperties(
            symbol='NaCl', name='Sodium chloride', formula='NaCl',
            CAS='7647-14-5', MW=58.44277,
        )
        self.assertAlmostEqual(props.solid_heat_capacity(298.15), 50.509, places=3)
        self.assertAlmostEqual(
            props.solid_delta_H(295.0, 300.0),
            props.solid_cp_kernel().delta_h(295.0, 300.0),
        )

    def test_thermodynamics_exposes_solid_cp_enthalpy_and_entropy(self):
        thermo = create_thermodynamics(['NaCl'], 'IDEAL')
        self.assertAlmostEqual(thermo.Cp_solid('NaCl', 298.15), 50.509, places=3)
        h1 = thermo.enthalpy_solid('NaCl', 298.15)
        h2 = thermo.enthalpy_solid('NaCl', 300.0)
        self.assertGreater(h2, h1)
        s1 = thermo.entropy_solid('NaCl', 298.15)
        s2 = thermo.entropy_solid('NaCl', 300.0)
        self.assertGreater(s2, s1)

    def test_local_database_solid_density_coverage_improves(self):
        database = ChemicalDatabase(enable_online=False)
        nacl = database.get('NaCl', fetch_online=False)
        naoh = database.get('NaOH', fetch_online=False)
        self.assertAlmostEqual(
            nacl.solid_mass_density(298.15),
            nacl.MW / nacl.solid_molar_volume(298.15),
            places=9,
        )
        self.assertAlmostEqual(
            naoh.solid_mass_density(298.15),
            naoh.MW / naoh.solid_molar_volume(298.15),
            places=9,
        )


if __name__ == '__main__':
    unittest.main()
