import math
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from perry_properties import get_perry_property_library
from property_resolver import PropertyResolver


class PerryPropertyLookupTests(unittest.TestCase):
    def setUp(self):
        self.library = get_perry_property_library()
        self.resolver = PropertyResolver()

    def assertClose(self, actual, expected, *, rel=1e-8, abs_tol=1e-12):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol),
            f'{actual!r} != {expected!r}',
        )

    def test_library_evaluates_verified_perry_correlations_with_expected_units(self):
        acetic_vp = self.library.vapor_pressure_bar('acetic acid', 289.81)
        self.assertIsNotNone(acetic_vp)
        self.assertEqual(acetic_vp.units, 'bar')
        self.assertEqual(acetic_vp.method, 'perry_vapor_pressure')
        self.assertClose(acetic_vp.value, 0.012769038092402922)

        benzaldehyde_table_vp = self.library.table_2_10_vapor_pressure_bar('benzaldehyde', 400.0)
        self.assertIsNotNone(benzaldehyde_table_vp)
        self.assertEqual(benzaldehyde_table_vp.units, 'bar')
        self.assertEqual(benzaldehyde_table_vp.method, 'perry_table_2_10_vapor_pressure')
        self.assertClose(benzaldehyde_table_vp.value, 0.22583453556519711)

        benzaldehyde_normal_vp = self.library.table_2_10_vapor_pressure_bar('benzaldehyde', 452.15)
        self.assertClose(benzaldehyde_normal_vp.value, 1.01325)
        self.assertIsNone(self.library.table_2_10_vapor_pressure_bar('benzaldehyde', 250.0))

        benzaldehyde_tm = self.library.normal_melting_point_K('benzaldehyde')
        self.assertIsNotNone(benzaldehyde_tm)
        self.assertEqual(benzaldehyde_tm.method, 'perry_table_2_10_normal_melting_point')
        self.assertClose(benzaldehyde_tm.value, 247.15)

        acetic_cp = self.library.heat_capacity_J_per_mol_K('acetic acid', 289.81, 'liquid')
        self.assertIsNotNone(acetic_cp)
        self.assertEqual(acetic_cp.units, 'J/(mol*K)')
        self.assertEqual(acetic_cp.method, 'perry_liquid_cp_eq100')
        self.assertClose(acetic_cp.value, 122.13381973584998)

        acetic_density = self.library.liquid_molar_density_mol_per_dm3('acetic acid', 289.81)
        self.assertIsNotNone(acetic_density)
        self.assertEqual(acetic_density.units, 'mol/dm^3')
        self.assertEqual(acetic_density.method, 'perry_density_eq105')
        self.assertClose(acetic_density.value, 17.491747483847327)

        acetic_volume = self.library.liquid_molar_volume_m3_per_kmol('acetic acid', 289.81)
        self.assertIsNotNone(acetic_volume)
        self.assertEqual(acetic_volume.units, 'm^3/kmol')
        self.assertClose(acetic_volume.value, 1.0 / acetic_density.value)

        liquid_viscosity = self.library.viscosity_Pa_s('acetic acid', 289.81, 'liquid')
        vapor_viscosity = self.library.viscosity_Pa_s('acetic acid', 289.81, 'vapor')
        self.assertClose(liquid_viscosity.value, 0.0012653517240117334)
        self.assertClose(vapor_viscosity.value, 7.053342259026438e-06)

        ideal_cp = self.library.heat_capacity_J_per_mol_K('acetic acid', 50.0, 'ideal_gas')
        self.assertIsNotNone(ideal_cp)
        self.assertEqual(ideal_cp.method, 'perry_ideal_gas_cp_hyperbolic')
        self.assertClose(ideal_cp.value, 40.20000461311089)

    def test_vapor_pressure_coefficient_quirks_are_normalized(self):
        sulfur_hexafluoride = self.library.vapor_pressure_bar(
            '2551-62-4',
            223.15,
        )
        acetamide = self.library.vapor_pressure_bar('60-35-5', 353.33)

        self.assertIsNotNone(sulfur_hexafluoride)
        self.assertIsNotNone(acetamide)
        self.assertClose(sulfur_hexafluoride.value, 2.3, rel=0.002)
        self.assertClose(acetamide.value, 0.00336, rel=0.01)
        sf6_row = self.library.vapor_pressure_correlations('2551-62-4')[0]
        acetamide_row = self.library.vapor_pressure_correlations('60-35-5')[0]
        self.assertEqual(
            self.library.vapor_pressure_coefficients(sf6_row),
            (29.16, -2383.6, -1.1342, 0.0, 1.0),
        )
        self.assertEqual(
            self.library.vapor_pressure_coefficients(acetamide_row),
            (125.81, -12376.0, -14.589, 5.0824e-06, 2.0),
        )

    def test_library_respects_ranges_and_ambiguous_formula_aliases(self):
        self.assertIsNone(self.library.vapor_pressure_bar('acetic acid', 700.0))
        self.assertIsNone(self.library.heat_capacity_J_per_mol_K('acetic acid', 450.0, 'liquid'))
        self.assertIsNone(self.library.get('C2H6O'))
        self.assertIsNotNone(self.library.get('64-17-5'))
        self.assertEqual(self.library.get('64-17-5')['name'], 'Ethanol')

    def test_resolver_canonical_vapor_pressure_uses_hydrated_database_props(self):
        ethanol = self.resolver.resolve_vapor_pressure('ethanol', 413.0)
        self.assertEqual(ethanol.method, 'coolprop_HEOS_psat')
        self.assertIn('canonical', ethanol.notes.lower())

        acetic = self.resolver.resolve_vapor_pressure('acetic acid', 300.0)
        self.assertEqual(acetic.method, 'perry_2_8_vapor_pressure')
        self.assertEqual(acetic.source, 'Perry 9th Table 2-8')
        self.assertClose(acetic.value, 0.023071582299057314, rel=0.01)

        benzaldehyde = self.resolver.resolve_vapor_pressure('benzaldehyde', 400.0, {})
        self.assertEqual(benzaldehyde.method, 'perry_2_10_vapor_pressure')
        self.assertEqual(benzaldehyde.source, 'Perry 9th Table 2-10')
        self.assertClose(benzaldehyde.quality, 0.95)
        self.assertClose(benzaldehyde.value, 0.22583453556519711, rel=0.01)
        self.assertIn('canonical', benzaldehyde.notes.lower())

        carbon_disulfide = self.resolver.resolve_vapor_pressure('carbon disulfide', 319.0)
        self.assertEqual(carbon_disulfide.method, 'perry_2_8_vapor_pressure')
        self.assertEqual(carbon_disulfide.source, 'Perry 9th Table 2-8')

    def test_table_2_10_vapor_pressure_preempts_online_hydration(self):
        resolver = PropertyResolver()
        with patch(
            'chemical_properties.OnlinePropertyFetcher.fetch_from_pubchem',
            side_effect=AssertionError('online PubChem lookup should not run before local Table 2-10 Psat'),
        ):
            result = resolver.resolve_vapor_pressure('benzaldehyde', 400.0)

        self.assertEqual(result.method, 'perry_2_10_vapor_pressure')
        self.assertClose(result.value, 0.22583453556519711, rel=0.01)

    def test_resolver_uses_hydrated_database_critical_properties(self):
        # CoolProp outranks database hydration for its reference fluids, so
        # silence that rung and isolate the persistent result cache to exercise
        # the hydration path itself.
        resolver = PropertyResolver()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            PropertyResolver,
            'SATURATION_PROPERTIES_CACHE_PATH',
            Path(directory, 'critical_properties.sqlite'),
        ), patch(
            'property_resolver._resolver',
            resolver,
        ), patch.object(
            PropertyResolver,
            '_coolprop_critical_properties',
            return_value=None,
        ):
            critical = resolver.resolve_critical_properties('ethanol')

        self.assertEqual(critical['Tc'].method, 'chemicals_json')
        self.assertEqual(critical['Pc'].method, 'chemicals_json')
        self.assertEqual(critical['Vc'].method, 'chemicals_json')
        self.assertEqual(critical['Zc'].source, 'local')
        self.assertEqual(critical['Zc'].method, 'acs_jced_5b00571_table1')
        self.assertClose(critical['Zc'].value, 0.241)
        self.assertClose(critical['Tc'].value, 513.9)
        self.assertClose(critical['Pc'].value, 61.4)
        self.assertClose(critical['Vc'].value, 167.0)
        self.assertClose(critical['omega'].value, 0.644)

    def test_resolver_heat_capacity_prefers_phase_specific_sources(self):
        provided_liquid = self.resolver.resolve_heat_capacity(
            'ethanol',
            298.15,
            phase='liquid',
            props={
                'Cp_liquid': 123.4,
                'property_sources': {
                    'Cp_liquid': {
                        'source': 'provided',
                        'method': 'pfd_component_override',
                        'quality': 1.0,
                    },
                },
            },
        )
        self.assertEqual(provided_liquid.source, 'provided')
        self.assertClose(provided_liquid.value, 123.4)

        perry_liquid = self.resolver.resolve_heat_capacity('ethanol', 298.15, phase='liquid')
        self.assertEqual(perry_liquid.method, 'canonical_zabransky_p_spline_liquid_cp')
        self.assertClose(perry_liquid.value, 112.08848590533385)

        provided_gas = self.resolver.resolve_heat_capacity(
            'ethanol',
            298.15,
            phase='ideal_gas',
            props={'Cp_coeffs': [1.0, 0.0, 0.0, 0.0]},
        )
        self.assertEqual(provided_gas.source, 'provided')
        self.assertClose(provided_gas.value, 1.0)

        perry_gas = self.resolver.resolve_heat_capacity('ethanol', 298.15, phase='ideal_gas')
        self.assertEqual(perry_gas.method, 'canonical_perry_9e_ideal_gas_cp')
        self.assertClose(perry_gas.value, 65.10418051539888)

    def test_resolver_bounded_extrapolates_heat_capacity(self):
        provided_props = {
            'property_correlations': {
                'Cpl': {
                    'equation': 'poly_x',
                    'Tmin_K': 290.0,
                    'Tmax_K': 300.0,
                    'coefficients': {'A': 100.0, 'B': 20.0},
                    'source': 'unit test',
                },
                'Cpg': {
                    'equation': 'poly_x',
                    'Tmin_K': 300.0,
                    'Tmax_K': 400.0,
                    'coefficients': {'A': 30.0, 'B': 10.0},
                    'source': 'unit test',
                },
            },
        }

        liquid = self.resolver.resolve_heat_capacity(
            'madeupium',
            320.0,
            phase='liquid',
            props=provided_props,
        )
        clamped_liquid = self.resolver.resolve_heat_capacity(
            'madeupium',
            350.0,
            phase='liquid',
            props=provided_props,
        )
        self.assertEqual(liquid.method, 'provided_heat_capacity_fit')
        self.assertClose(liquid.value, 100.0 + 20.0 * ((310.0 - 298.15) / 100.0))
        self.assertClose(clamped_liquid.value, liquid.value)
        self.assertIn('clamped above extended range', liquid.notes)

        with_scalar = dict(provided_props, Cp_liquid=123.4)
        scalar = self.resolver.resolve_heat_capacity(
            'madeupium',
            320.0,
            phase='liquid',
            props=with_scalar,
        )
        self.assertEqual(scalar.method, 'provided_heat_capacity_fit')
        self.assertClose(scalar.value, liquid.value)

        gas = self.resolver.resolve_heat_capacity(
            'madeupium',
            250.0,
            phase='ideal_gas',
            props=provided_props,
        )
        self.assertEqual(gas.method, 'provided_heat_capacity_fit')
        self.assertClose(gas.value, 30.0 + 10.0 * ((290.0 - 298.15) / 100.0))
        self.assertIn('clamped below extended range', gas.notes)

        perry_liquid = self.resolver.resolve_heat_capacity('ethanol', 450.0, phase='liquid')
        bundled_liquid = self.resolver.resolve_liquid_cp_kernel('ethanol')
        self.assertEqual(perry_liquid.method, bundled_liquid.method)
        self.assertClose(perry_liquid.value, bundled_liquid.cp(450.0))

        perry_gas = self.resolver.resolve_heat_capacity('ethanol', 40.0, phase='ideal_gas')
        self.assertEqual(perry_gas.method, 'canonical_perry_9e_ideal_gas_cp')
        self.assertClose(perry_gas.value, 42.94063779644761)
        self.assertIn('clamped below extended range', perry_gas.notes)

    def test_resolver_bounded_extrapolates_perry_viscosity_before_estimation(self):
        acetone_entry = self.library.get('acetone')
        acetone_row = acetone_entry['liquid_viscosity'][0]

        near = self.resolver.resolve_viscosity('acetone', 333.15, phase='liquid')
        farther = self.resolver.resolve_viscosity('acetone', 343.15, phase='liquid')

        self.assertEqual(near.method, 'perry_liquid_viscosity_eq101_bounded_extrapolation')
        self.assertClose(
            near.value,
            self.library._eval_liquid_viscosity_Pa_s(acetone_row, 333.15),
        )
        self.assertClose(near.quality, 0.94)
        self.assertIn('quality penalty 0.03', near.notes)

        self.assertEqual(farther.method, 'perry_liquid_viscosity_eq101_bounded_extrapolation')
        self.assertClose(
            farther.value,
            self.library._eval_liquid_viscosity_Pa_s(acetone_row, 343.15),
        )
        self.assertClose(farther.quality, 0.87)
        self.assertIn('quality penalty 0.10', farther.notes)

    def test_resolver_exposes_perry_scalar_phase_change_properties(self):
        tb = self.resolver.resolve_boiling_point(
            'methyl isobutyl ketone',
            {},
            allow_online=False,
            allow_estimation=False,
        )
        tm = self.resolver.resolve_melting_point('acetic acid', {}, allow_online=False)
        hvap = self.resolver.resolve_hvap(
            'methyl isobutyl ketone',
            {},
            allow_online=False,
            allow_estimation=False,
        )
        hfus = self.resolver.resolve_hfus('acetic acid', {}, allow_online=False)

        self.assertEqual(tb.method, 'perry_vapor_pressure_normal_boiling_point')
        self.assertClose(tb.value, 389.2946349170139)
        self.assertEqual(tm.method, 'perry_heat_of_fusion_melting_point')
        self.assertClose(tm.value, 289.85)
        self.assertEqual(hvap.method, 'perry_heat_of_vaporization')
        self.assertClose(hvap.value, 34.692667885531584)
        self.assertEqual(hfus.method, 'perry_heat_of_fusion')
        self.assertClose(hfus.value, 11.7286954618752)

    def test_perry_scalar_hfus_prefers_a_row_with_its_paired_tm(self):
        hfus = self.library.heat_of_fusion_kJ_per_mol('112-05-0')
        tm = self.library.normal_melting_point_K('112-05-0')

        self.assertIsNotNone(hfus)
        self.assertIsNotNone(tm)
        self.assertEqual(hfus.correlation['table_name'], 'Pelargonic acid (n-) (α-)')
        self.assertEqual(tm.correlation['table_name'], 'Pelargonic acid (n-) (α-)')
        self.assertClose(hfus.value, 20.2791390320784)
        self.assertClose(tm.value, 285.5)

    def test_perry_fusion_routes_preserve_polymorphs_and_blank_tm(self):
        fusion = self.resolver.resolve_fusion_transitions(
            '79-11-8', {}, allow_online=False,
        )
        melting = self.resolver.resolve_melting_transitions(
            '79-11-8', {}, allow_online=False,
        )
        scalar_hfus = self.resolver.resolve_hfus(
            '79-11-8', {}, allow_online=False,
        )
        scalar_tm = self.resolver.resolve_melting_point(
            '79-11-8', {}, allow_online=False,
        )

        self.assertEqual(len(fusion), 2)
        self.assertEqual(
            {(record.polymorph, record.temperature_K) for record in fusion},
            {('alpha', 334.34999999999997), ('beta', 329.15)},
        )
        self.assertTrue(all(record.material_form == 'unspecified' for record in fusion))
        self.assertTrue(any(
            record.enthalpy_kJ_mol is not None and record.polymorph == 'alpha'
            for record in melting
        ))
        self.assertEqual(len(melting), 2)
        self.assertClose(scalar_tm.value, 334.34999999999997)
        self.assertClose(scalar_hfus.value, 12.280364013980801)

        nonanoic = self.resolver.resolve_fusion_transitions(
            '112-05-0', {}, allow_online=False,
        )
        self.assertEqual(len(nonanoic), 2)
        beta = next(record for record in nonanoic if record.polymorph == 'beta')
        alpha = next(record for record in nonanoic if record.polymorph == 'alpha')
        self.assertIsNone(beta.temperature_K)
        self.assertEqual(beta.stereochemistry, '')
        self.assertFalse(beta.is_source_default)
        self.assertClose(alpha.temperature_K, 285.5)
        self.assertClose(
            self.resolver.resolve_hfus('112-05-0', {}, allow_online=False).value,
            alpha.enthalpy_kJ_mol,
        )

        beta_naphthol = self.resolver.resolve_fusion_transitions(
            '135-19-3', {}, allow_online=False,
        )
        self.assertEqual(len(beta_naphthol), 1)
        self.assertEqual(beta_naphthol[0].polymorph, '')
        self.assertEqual(beta_naphthol[0].form_label, '')
        self.assertTrue(beta_naphthol[0].is_source_default)
        self.assertEqual(
            beta_naphthol[0].metadata['identity_form_qualifiers'],
            ['beta'],
        )

    def test_resolver_exposes_perry_volume_viscosity_and_formation_properties(self):
        volume = self.resolver.resolve_liquid_molar_volume('ethyl acetate', 298.15)
        density = self.resolver.resolve_liquid_molar_density('ethyl acetate', 298.15)
        self.assertEqual(volume.method, 'perry_molar_volume_eq105')
        self.assertClose(volume.value, 1.0 / density.value)

        viscosity = self.resolver.resolve_viscosity('ethyl acetate', 298.15, 'liquid')
        self.assertEqual(viscosity.method, 'perry_liquid_viscosity_eq101')
        self.assertClose(viscosity.value, 0.00043030529317067255)

        formation = self.resolver.resolve_formation_properties('acetic acid')
        self.assertClose(formation['Hf'].value, -432.8)
        self.assertClose(formation['Gf'].value, -374.5)
        self.assertClose(formation['S'].value, 282.5)
        self.assertClose(formation['Hcomb'].value, -786.6)

    def test_liquid_molar_volume_uses_small_extrapolation_then_fitted_rackett_cache(self):
        # Methanol is a CoolProp fluid; silence that top rung so the Perry
        # extrapolation and fitted-Rackett machinery under test still runs.
        with patch.object(
            PropertyResolver,
            '_coolprop_saturated_liquid_volume',
            return_value=None,
        ):
            endpoint = self.resolver.resolve_liquid_molar_volume('methanol', 175.47)
            near = self.resolver.resolve_liquid_molar_volume('methanol', 160.0)
            self.assertEqual(near.method, 'perry_molar_volume_eq105_extrapolated')
            self.assertLess(near.quality, endpoint.quality)

            with tempfile.TemporaryDirectory() as tmpdir:
                cache_path = Path(tmpdir) / 'liquid_volume_zra_cache.json'
                resolver = PropertyResolver()
                with patch.object(PropertyResolver, 'LIQUID_VOLUME_ZRA_CACHE_PATH', cache_path):
                    far = resolver.resolve_liquid_molar_volume_nearest('methanol', 150.0)
                    self.assertEqual(far.method, 'rackett_fitted_zra')
                    self.assertLess(far.quality, endpoint.quality)
                    self.assertIn('Z_RA fitted once', far.notes)
                    sqlite_path = cache_path.with_suffix('.sqlite')
                    self.assertTrue(sqlite_path.exists())
                    with sqlite3.connect(sqlite_path) as connection:
                        self.assertGreater(
                            connection.execute(
                                """
                                SELECT count(*) FROM runtime_json_cache
                                WHERE namespace = 'liquid_volume_zra_v1'
                                """
                            ).fetchone()[0],
                            0,
                        )


if __name__ == '__main__':
    unittest.main()
