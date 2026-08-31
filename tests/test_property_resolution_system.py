import math
import os
import sqlite3
import sys
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chemical_properties import (
    ChemicalDatabase,
    ChemicalProperties,
    OnlinePropertyFetcher,
    _inferred_phase_at_stp,
)
import domalski_hearing_method as dh
from perry_properties import get_perry_property_library
from property_resolver import AntoineCoefficients, FusionTransitionRecord, HeatCapacityLookup, HvapTemperatureFit, PropertyResolutionError, PropertyResolutionResult, PropertyResolver
from thermodynamics import IdealThermodynamics, create_thermodynamics


class PropertyResolutionSystemTests(unittest.TestCase):
    def assertClose(self, actual, expected, *, rel=1e-8, abs_tol=1e-12):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol),
            f'{actual!r} != {expected!r}',
        )

    def write_effective_criticals_db(self, directory, rows):
        path = Path(directory, 'effective_criticals.sqlite')
        with sqlite3.connect(path) as connection:
            connection.execute(
                """
                CREATE TABLE effective_criticals (
                    CAS TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    MW REAL NOT NULL,
                    Tc REAL NOT NULL,
                    Pc REAL NOT NULL,
                    Vc REAL NOT NULL,
                    Zc REAL NOT NULL,
                    omega REAL NOT NULL,
                    Tc_quality REAL NOT NULL,
                    Pc_quality REAL NOT NULL,
                    Vc_quality REAL NOT NULL,
                    Zc_quality REAL NOT NULL,
                    omega_quality REAL NOT NULL
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO effective_criticals (
                    CAS, name, MW, Tc, Pc, Vc, Zc, omega,
                    Tc_quality, Pc_quality, Vc_quality, Zc_quality, omega_quality
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return path

    def test_chemicals_json_no_longer_shadows_perry_cp_or_psat(self):
        payload = json.loads(Path(ROOT, 'data', 'chemicals.json').read_text())
        removed_keys = {
            'antoine_A',
            'antoine_B',
            'antoine_C',
            'antoine_Tmin',
            'antoine_Tmax',
            'Cp_coeffs',
            'Cp_liquid',
        }
        self.assertTrue(removed_keys.isdisjoint(payload['_metadata']['units']))
        for symbol, entry in payload['chemicals'].items():
            if symbol in {'3-methylpentane', 'H2SO4'}:
                continue
            self.assertTrue(removed_keys.isdisjoint(entry))

    def test_ambiguous_hexane_formula_does_not_shadow_perry_isomer(self):
        database = ChemicalDatabase(enable_online=False)

        self.assertIsNone(database.get('C6H14', fetch_online=False))
        self.assertEqual(database.get_user_component('C6H14', fetch_online=False).name, 'n-Hexane')
        self.assertEqual(database.get_user_component('C4H10', fetch_online=False).name, 'n-Butane')

        isohexane = database.get('2-methylpentane', fetch_online=False)
        self.assertIsNotNone(isohexane)
        self.assertEqual(isohexane.symbol, '2-methylpentane')
        self.assertEqual(isohexane.CAS, '107-83-5')
        self.assertEqual(isohexane.source, 'perry')

        n_hexane = database.get('hexane', fetch_online=False)
        self.assertIsNotNone(n_hexane)
        self.assertEqual(n_hexane.symbol, 'C6H14')
        self.assertEqual(n_hexane.CAS, '110-54-3')
        self.assertEqual(database.chemicals['C6H14'].name, 'n-Hexane')

    def test_coolprop_globally_replaces_supported_critical_bundles(self):
        database = ChemicalDatabase(enable_online=False)

        water = database.get('water', fetch_online=False)
        self.assertIsNotNone(water)

        self.assertClose(water.Tc, 647.096)
        self.assertClose(water.Pc, 220.64)
        self.assertClose(water.Vc, 55.94803726708075)
        self.assertClose(water.Zc, 0.22943845208521263)
        self.assertEqual(water.property_sources['Tc']['method'], 'coolprop_IF97_critical')
        self.assertEqual(water.property_sources['Zc']['method'], 'coolprop_critical_identity')
        self.assertClose(
            water.Zc,
            water.Pc * 100000.0 * water.Vc * 1.0e-6 / (8.314462618 * water.Tc),
        )

        cyclohexane = database.get('cyclohexane', fetch_online=False)
        self.assertClose(cyclohexane.Vc, 310.17389460325523)
        self.assertClose(cyclohexane.Zc, 0.2749736625910168)
        self.assertEqual(cyclohexane.property_sources['Vc']['method'], 'coolprop_HEOS_critical')
        self.assertEqual(cyclohexane.property_sources['Zc']['method'], 'coolprop_critical_identity')

    def test_unlabeled_chemicals_json_criticals_have_curated_quality(self):
        database = ChemicalDatabase(enable_online=False)
        props = database.get('1-propanol', fetch_online=False)

        for key in ('Tc', 'Pc', 'omega'):
            with self.subTest(property=key):
                source = props.property_sources[key]
                self.assertEqual(source['source'], 'local')
                self.assertEqual(source['method'], 'chemicals_json')
                self.assertClose(source['quality'], 0.995)

    def test_database_can_resolve_perry_only_compound_before_online_lookup(self):
        database = ChemicalDatabase(enable_online=True)

        with patch.object(database.online_fetcher, '_estimate_missing_properties', side_effect=lambda props: props), \
             patch.object(database.online_fetcher, 'fetch_from_pubchem') as pubchem:
            props = database.get('methyl isobutyl ketone')

        self.assertIsNotNone(props)
        self.assertEqual(props.source, 'perry')
        self.assertEqual(props.CAS, '108-10-1')
        self.assertClose(props.MW, 100.15888)
        self.assertClose(props.Tc, 574.6)
        self.assertClose(props.Pc, 32.7)
        self.assertClose(props.Zc, 0.253)
        pubchem.assert_not_called()

    def test_ambiguous_formula_does_not_hydrate_wrong_perry_isomer(self):
        database = ChemicalDatabase(enable_online=False)
        resolver = PropertyResolver()

        acetone = database.get('acetone', fetch_online=False)
        self.assertEqual(acetone.name, 'Acetone')
        acetone_cp = resolver.resolve_heat_capacity(
            acetone.symbol,
            298.15,
            phase='liquid',
            props=database._resolver_props_dict(acetone),
        )
        self.assertEqual(acetone_cp.method, 'canonical_perry_9e_liquid_cp')

        propylene_oxide_props = {
            'symbol': 'propylene oxide',
            'name': '2-methyloxirane',
            'formula': 'C3H6O',
            'MW': 58.08,
            'Tb': 307.37777777777774,
            'antoine_A': 3.779503020064361,
            'antoine_B': 915.31,
            'antoine_C': 208.29,
            'antoine_Tmin': 225.15,
            'antoine_Tmax': 340.15,
        }

        with patch(
            'property_resolution.critical.EFFECTIVE_CRITICALS_DB_PATH',
            Path(tempfile.gettempdir(), 'missing_effective_criticals.sqlite'),
        ):
            critical = resolver.resolve_critical_properties(
                'propylene oxide',
                propylene_oxide_props,
                allow_online=False,
        )
        self.assertEqual(critical['Vc'].source, 'estimated')
        self.assertEqual(critical['Vc'].method, 'nannoolal_vc')
        self.assertClose(critical['Vc'].value, 194.00321998396652)
        self.assertClose(critical['Vc'].quality, 0.85)
        self.assertEqual(critical['Zc'].source, 'calculated')
        self.assertEqual(critical['Zc'].method, 'critical_volume_identity')

        with tempfile.TemporaryDirectory() as directory:
            cp_resolver = PropertyResolver()
            cp_resolver.CACHE_DIR = Path(directory)
            with (
                patch.object(cp_resolver, '_get_derived_liquid_cp_kernel', return_value=None),
                patch.object(cp_resolver, '_fetch_nist_cp_source', return_value=None),
            ):
                cp = cp_resolver.resolve_heat_capacity(
                    'propylene oxide',
                    298.15,
                    phase='liquid',
                    props=propylene_oxide_props,
                    allow_online=False,
                )
        self.assertEqual(cp.source, 'estimated')

        with patch(
            'property_resolution.critical.EFFECTIVE_CRITICALS_DB_PATH',
            Path(tempfile.gettempdir(), 'missing_effective_criticals.sqlite'),
        ), patch.object(
            resolver, '_fetch_liquid_density_pubchem', return_value=None
        ):
            density = resolver.resolve_liquid_molar_density(
                'propylene oxide',
                298.15,
                propylene_oxide_props,
            )
        self.assertEqual(density.source, 'calculated')
        self.assertEqual(density.method, 'rackett_yamada_gunn')
        self.assertNotEqual(density.method, 'provided_liquid_density_fit')
        self.assertClose(density.quality, 0.7254)
        self.assertGreater(density.value, 0.0)

    def test_completed_common_chemicals_use_local_overrides_and_portable_fits(self):
        database = ChemicalDatabase(enable_online=False)
        resolver = PropertyResolver()

        expected = {
            'glycerol': ('C3H8O3', 0.718734),
            'aniline': ('C6H7N', 0.37663),
            'pyridine': ('C5H5N', 0.24133),
            'propylene oxide': ('C3H6O_PO', 0.249),
            'acetophenone': ('C8H8O', 0.39332),
        }

        for name, (symbol, omega) in expected.items():
            props = database.get(name, fetch_online=False)
            self.assertIsNotNone(props, name)
            self.assertEqual(props.symbol, symbol)
            self.assertClose(props.omega, omega)

            resolver_props = database._resolver_props_dict(props)
            psat = resolver.resolve_vapor_pressure(
                props.symbol,
                props.Tb,
                resolver_props,
                allow_online=False,
            )
            cp_liquid = resolver.resolve_heat_capacity(
                props.symbol,
                298.15,
                phase='liquid',
                props=resolver_props,
            )
            rho_liquid = resolver.resolve_liquid_molar_density(
                props.symbol,
                298.15,
                resolver_props,
            )

            self.assertGreater(psat.value, 0.0)
            self.assertIn('Canonical', psat.notes)
            if name == 'propylene oxide':
                self.assertEqual(cp_liquid.method, 'provided_heat_capacity_fit')
            else:
                self.assertTrue(cp_liquid.method.startswith('canonical_zabransky_p_'))
                self.assertNotIn('Cpl', resolver_props['property_correlations'])
                self.assertEqual(
                    resolver_props['property_sources']['Cpl']['method'],
                    'canonical_liquid_cp_database',
                )
            self.assertEqual(rho_liquid.method, 'provided_liquid_density_fit')
            if name == 'glycerol':
                for field in ('Tc', 'Pc', 'Vc', 'Zc', 'omega'):
                    self.assertIs(
                        resolver_props['property_sources'][field]['replaceable'],
                        False,
                    )
                critical = resolver.resolve_critical_properties(
                    props.symbol,
                    resolver_props,
                    allow_online=False,
                    allow_estimation=True,
                )
                self.assertEqual(critical['Tc'].method, 'effective_critical')
                self.assertEqual(critical['Pc'].method, 'effective_critical')
                self.assertEqual(critical['omega'].method, 'effective_critical')
                self.assertClose(critical['omega'].value, 0.718734)

    def test_propylene_oxide_fit_and_critical_quality_labels(self):
        database = ChemicalDatabase(enable_online=False)
        props = database.get('propylene oxide', fetch_online=False).to_dict()

        correlations = props['property_correlations']
        self.assertEqual(
            set(correlations),
            {'Psat', 'Hvap', 'Cpl', 'Cpg', 'rhol', 'mul', 'mug'},
        )
        for name, correlation in correlations.items():
            with self.subTest(correlation=name):
                self.assertIn(correlation['selected_model'], {'HEOS_FIT', 'REFPROP_FIT'})
                self.assertEqual(correlation['quality'], 0.99)

        for name in ('Tc', 'Pc', 'Vc', 'Zc'):
            with self.subTest(critical=name):
                self.assertEqual(props['property_sources'][name]['quality'], 0.995)

    def test_resolver_exposes_full_critical_set_and_keeps_provided_priority(self):
        resolver = PropertyResolver()

        critical = resolver.resolve_critical_properties('ethanol', allow_online=False)
        self.assertEqual(set(critical), {'Tc', 'Pc', 'Vc', 'Zc', 'omega'})
        self.assertEqual(critical['Tc'].method, 'coolprop_HEOS_critical')
        self.assertEqual(critical['Zc'].source, 'calculated')
        self.assertEqual(critical['Zc'].method, 'coolprop_critical_identity')
        self.assertClose(critical['Zc'].value, 0.2469573323737847)

        provided = resolver.resolve_critical_properties(
            'madeupium',
            {'Tc': 500.0, 'Pc': 50.0, 'Vc': 200.0, 'omega': 0.1},
            allow_online=False,
        )
        self.assertEqual(provided['Tc'].source, 'provided')
        self.assertEqual(provided['Pc'].source, 'provided')
        self.assertEqual(provided['Vc'].source, 'provided')
        self.assertEqual(provided['omega'].source, 'provided')
        self.assertEqual(provided['Zc'].source, 'exact')
        self.assertEqual(provided['Zc'].method, 'critical_volume_identity')
        self.assertClose(provided['Zc'].value, 50.0 * 100000.0 * 200.0e-6 / (8.314462618 * 500.0))

    def test_coolprop_critical_source_precedes_acs_and_keeps_partial_fill(self):
        resolver = PropertyResolver()

        cyclohexane = resolver.resolve_critical_properties(
            'cyclohexane',
            {'CAS': '110-82-7'},
            allow_online=False,
        )
        self.assertEqual(cyclohexane['Tc'].method, 'coolprop_HEOS_critical')
        self.assertClose(cyclohexane['Tc'].value, 553.6000188557726)
        self.assertClose(cyclohexane['Tc'].quality, 0.995)
        self.assertEqual(cyclohexane['Pc'].method, 'coolprop_HEOS_critical')
        self.assertClose(cyclohexane['Pc'].value, 40.805258791621355)
        self.assertEqual(cyclohexane['Vc'].method, 'coolprop_HEOS_critical')
        self.assertClose(cyclohexane['Vc'].value, 310.17389460325523)

        with tempfile.TemporaryDirectory() as directory:
            acs_path = Path(directory, 'acs_partial.json')
            acs_path.write_text(json.dumps({
                'chemicals': {
                    '123-45-6': {
                        'critical_properties': {
                            'Tc_K': 610.0,
                            'Tc_uncertainty_K': 2.0,
                            'Pc_MPa': 4.2,
                            'Pc_uncertainty_MPa': 0.1,
                        },
                    },
                },
            }))
            partial_resolver = PropertyResolver()
            perry_vc = {
                'Vc': SimpleNamespace(
                    value=250.0,
                    source='local',
                    method='perry_critical',
                    units='cm^3/mol',
                ),
            }

            with patch('property_resolution.critical.ACS_JCED_5B00571_TABLE1_PATH', acs_path), \
                 patch.object(partial_resolver, '_get_perry_critical_properties', return_value=perry_vc):
                partial = partial_resolver.resolve_critical_properties(
                    'Partial ACS',
                    {'CAS': '123-45-6'},
                    allow_online=False,
                    allow_estimation=False,
                )

        self.assertEqual(partial['Tc'].method, 'acs_jced_5b00571_table1')
        self.assertClose(partial['Tc'].value, 610.0)
        self.assertEqual(partial['Pc'].method, 'acs_jced_5b00571_table1')
        self.assertClose(partial['Pc'].value, 42.0)
        self.assertEqual(partial['Vc'].method, 'perry_critical')
        self.assertClose(partial['Vc'].value, 250.0)

    def test_critical_unavailable_flag_blocks_hydration_and_resolution(self):
        database = ChemicalDatabase(enable_online=False)
        resolver = PropertyResolver()

        salt = database.get('NaCl', fetch_online=False)
        self.assertIsNotNone(salt)
        self.assertClose(salt.Tb, 1738.0)
        self.assertIsNone(salt.Tc)
        self.assertIsNone(salt.Pc)
        self.assertIsNone(salt.Vc)
        self.assertTrue(salt.critical_properties_unavailable)

        critical = resolver.resolve_critical_properties(
            'NaCl',
            database._resolver_props_dict(salt),
            allow_online=True,
            allow_estimation=True,
        )
        for result in critical.values():
            self.assertIsNone(result.value)
            self.assertEqual(result.method, 'critical_properties_unavailable')

    def test_effective_criticals_win_over_online_when_hard(self):
        resolver = PropertyResolver()
        with tempfile.TemporaryDirectory() as directory:
            db_path = self.write_effective_criticals_db(
                directory,
                [(
                    '123-45-6', 'Hard Effective', 100.0,
                    510.0, 42.0, 220.0, 0.22, 0.31,
                    0.94, 0.94, 0.94, 0.94, 0.94,
                )],
            )
            with patch('property_resolution.critical.EFFECTIVE_CRITICALS_DB_PATH', db_path), \
                 patch.object(resolver, '_fetch_phase_change_online') as phase:
                critical = resolver.resolve_critical_properties(
                    'Hard Effective',
                    {'CAS': '123-45-6'},
                    allow_online=True,
                )

        self.assertEqual(critical['Tc'].source, 'local')
        self.assertEqual(critical['Tc'].method, 'effective_critical')
        self.assertClose(critical['Tc'].value, 510.0)
        self.assertClose(critical['Tc'].quality, 0.94)
        phase.assert_not_called()

    def test_online_replaces_soft_effective_criticals_and_uses_uncertain_quality(self):
        resolver = PropertyResolver()
        with tempfile.TemporaryDirectory() as directory:
            db_path = self.write_effective_criticals_db(
                directory,
                [(
                    '234-56-7', 'Soft Effective', 100.0,
                    500.0, 40.0, 200.0, 0.20, 0.30,
                    0.86, 0.86, 0.86, 0.86, 0.86,
                )],
            )
            with patch('property_resolution.critical.EFFECTIVE_CRITICALS_DB_PATH', db_path), \
                 patch.object(
                     resolver,
                     '_fetch_phase_change_online',
                     return_value={
                         'Tc': 520.0,
                         'Pc': 44.0,
                         '_sources': {'Tc': 'pubchem', 'Pc': 'nist_phase_change'},
                         '_qualities': {'Tc': 0.88, 'Pc': 0.95},
                     },
                 ):
                critical = resolver.resolve_critical_properties(
                    'Soft Effective',
                    {'CAS': '234-56-7'},
                    allow_online=True,
                )

        self.assertEqual(critical['Tc'].source, 'online')
        self.assertEqual(critical['Tc'].method, 'pubchem')
        self.assertClose(critical['Tc'].value, 520.0)
        self.assertClose(critical['Tc'].quality, 0.88)
        self.assertEqual(critical['Pc'].source, 'online')
        self.assertEqual(critical['Pc'].method, 'nist_phase_change')
        self.assertClose(critical['Pc'].quality, 0.95)
        self.assertEqual(critical['Vc'].method, 'effective_critical')
        self.assertClose(critical['Vc'].quality, 0.86)

    def test_weak_effective_critical_can_be_replaced_by_downstream_estimation(self):
        resolver = PropertyResolver()
        with tempfile.TemporaryDirectory() as directory:
            db_path = self.write_effective_criticals_db(
                directory,
                [(
                    '345-67-8', 'Weak Effective', 100.0,
                    500.0, 40.0, 200.0, 0.20, 0.75,
                    0.94, 0.94, 0.94, 0.94, 0.75,
                )],
            )
            with patch('property_resolution.critical.EFFECTIVE_CRITICALS_DB_PATH', db_path), \
                 patch.object(resolver, '_fetch_phase_change_online', return_value=None):
                critical = resolver.resolve_critical_properties(
                    'Weak Effective',
                    {'CAS': '345-67-8', 'Tb': 360.0},
                    allow_online=True,
                    allow_estimation=True,
                )

        self.assertEqual(critical['omega'].source, 'calculated')
        self.assertEqual(critical['omega'].method, 'lee_kesler')
        self.assertNotEqual(critical['omega'].value, 0.75)
        self.assertClose(critical['omega'].quality, 0.94 * 0.80)

    def test_scalar_resolvers_cover_common_and_missing_properties(self):
        resolver = PropertyResolver()

        direct_tb = resolver.resolve_boiling_point(
            'ethanol',
            {'Tb': 123.4},
            allow_online=False,
        )
        self.assertEqual(direct_tb.source, 'provided')
        self.assertClose(direct_tb.value, 123.4)
        self.assertClose(
            resolver.resolve_boiling_point('ethanol', allow_online=False).value,
            351.57040446751455,
        )
        self.assertEqual(
            resolver.resolve_boiling_point('ethanol', allow_online=False).method,
            'coolprop_HEOS_boiling_point',
        )
        table_tb = resolver.resolve_boiling_point(
            'benzaldehyde',
            allow_online=False,
            allow_estimation=False,
        )
        self.assertEqual(table_tb.method, 'perry_table_2_10_normal_boiling_point')
        self.assertClose(table_tb.value, 452.15)

        hvap = resolver.resolve_hvap('ethanol', allow_online=False, allow_estimation=False)
        self.assertEqual(hvap.method, 'perry_heat_of_vaporization')
        self.assertClose(hvap.value, 39.18343971979801)
        self.assertEqual(
            resolver.resolve_hfus('ethanol', {'Hfus': 5.02}, allow_online=False).source,
            'provided',
        )

    def test_temperature_specific_hvap_uses_watson_before_scalar_return(self):
        resolver = PropertyResolver()
        props = {
            'Hvap': 40.0,
            'Tb': 350.0,
            'Tc': 600.0,
            'property_sources': {
                'Hvap': {'source': 'local', 'method': 'test_hvap', 'quality': 0.96},
                'Tb': {'source': 'textbook', 'method': 'test_tb', 'quality': 0.95},
                'Tc': {'source': 'online', 'method': 'test_tc', 'quality': 0.97},
            },
        }

        hvap = resolver.resolve_hvap(
            'madeupium',
            props,
            T=300.0,
            allow_online=False,
            allow_estimation=False,
        )
        expected = 40.0 * ((1.0 - 300.0 / 600.0) / (1.0 - 350.0 / 600.0)) ** 0.38

        self.assertEqual(hvap.method, 'watson_hvap')
        self.assertClose(hvap.value, expected)
        self.assertClose(hvap.quality, 0.95 * 0.88)
        self.assertIn('local/test_hvap', hvap.notes)

        scalar = resolver.resolve_hvap(
            'madeupium',
            props,
            allow_online=False,
            allow_estimation=False,
        )
        self.assertEqual(scalar.method, 'test_hvap')
        self.assertClose(scalar.value, 40.0)

    def test_hvap_watson_reference_prefers_temperature_dependent_tb_value(self):
        resolver = PropertyResolver()
        props = {
            'Hvap': 999.0,
            'Tb': 350.0,
            'Tc': 600.0,
            'property_correlations': {
                'Hvap': {
                    'equation': 'poly_x',
                    'Tmin_K': 340.0,
                    'Tmax_K': 360.0,
                    'coefficients': {'A': 40.0},
                    'source': 'unit test',
                },
            },
        }

        hvap = resolver.resolve_hvap(
            'madeupium',
            props,
            T=320.0,
            allow_online=False,
            allow_estimation=False,
        )
        expected = 40.0 * ((1.0 - 320.0 / 600.0) / (1.0 - 350.0 / 600.0)) ** 0.38

        self.assertEqual(hvap.method, 'watson_hvap')
        self.assertClose(hvap.value, expected)
        self.assertIn('provided/provided_hvap_fit', hvap.notes)

    def test_hvap_watson_can_seed_from_smith_textbook_scalar(self):
        resolver = PropertyResolver()
        props = {
            'Tb': 350.0,
            'Tc': 600.0,
        }

        with patch.object(resolver, '_get_textbook_entry', return_value={'Hvap': 40.0}):
            hvap = resolver.resolve_hvap(
                'madeupium',
                props,
                T=300.0,
                allow_online=False,
                allow_estimation=False,
            )

        expected = 40.0 * ((1.0 - 300.0 / 600.0) / (1.0 - 350.0 / 600.0)) ** 0.38
        self.assertEqual(hvap.method, 'watson_hvap')
        self.assertClose(hvap.value, expected)
        self.assertIn('textbook/Smith8 Appendix B', hvap.notes)

    def test_perry_only_temperature_dependent_properties_resolve_without_database_entry(self):
        resolver = PropertyResolver()

        vp = resolver.resolve_vapor_pressure('methyl isobutyl ketone', 350.0)
        cp = resolver.resolve_heat_capacity('methyl isobutyl ketone', 298.15, phase='ideal_gas')
        viscosity = resolver.resolve_viscosity('methyl isobutyl ketone', 298.15, phase='liquid')
        formation = resolver.resolve_formation_properties('methyl isobutyl ketone')

        self.assertEqual(vp.method, 'perry_2_8_vapor_pressure')
        self.assertEqual(cp.method, 'canonical_perry_9e_ideal_gas_cp')
        self.assertEqual(viscosity.method, 'perry_liquid_viscosity_eq101')
        self.assertClose(formation['Hf'].value, -286.4)
        self.assertClose(formation['S'].value, 412.9)

    def test_coolprop_viscosity_fills_perry_gap(self):
        resolver = PropertyResolver()
        viscosity = resolver.resolve_viscosity(
            'R11',
            298.15,
            phase='liquid',
            props={'CAS': '75-69-4', 'name': 'R11'},
        )

        self.assertEqual(viscosity.method, 'coolprop_HEOS_viscosity')
        self.assertEqual(viscosity.source, 'local')
        self.assertClose(viscosity.quality, 0.99)
        self.assertClose(viscosity.value, 0.0004331861787669167, rel=1e-9)
        self.assertIn('CoolProp HEOS transport model for R11', viscosity.notes)
        self.assertIn('max(1 atm, Psat=', viscosity.notes)

    def test_coolprop_alias_preflight_avoids_import_for_unknown_fluids(self):
        resolver = PropertyResolver()
        with patch('property_resolution.coolprop._COOLPROP_MODULE', None):
            self.assertIsNone(
                resolver._coolprop_reference(
                    'definitely not a coolprop fluid',
                    {'name': 'notacoolpropfluid', 'symbol': 'Xx2'},
                )
            )
            from property_resolution import coolprop as coolprop_module
            self.assertIsNone(coolprop_module._COOLPROP_MODULE)

    def test_coolprop_requires_resolved_cas_for_identity(self):
        resolver = PropertyResolver()
        components = (
            ('NA', {'symbol': 'NA', 'name': 'sodium', 'formula': 'Na'}),
            ('C2H6O', {'symbol': 'C2H6O', 'name': 'unknown'}),
            ('Co', {'symbol': 'Co', 'name': 'cobalt'}),
            ('R11', {'symbol': 'R11', 'name': 'custom solvent'}),
            ('ethanol', {'symbol': 'ETOH', 'name': 'ethanol'}),
        )

        for symbol, props in components:
            with self.subTest(symbol=symbol):
                self.assertIsNone(resolver._coolprop_reference(symbol, props))

        triple = resolver.resolve_triple_point(
            'NA',
            components[0][1],
            allow_online=False,
        )
        self.assertEqual(triple['Tt'].source, 'missing')
        self.assertEqual(triple['Pt'].source, 'missing')

    def test_database_hydration_resolves_cas_before_coolprop_matching(self):
        resolver = PropertyResolver()
        props = resolver._coerce_props('ethanol', None, allow_online=False)

        self.assertEqual(props['CAS'], '64-17-5')
        reference = resolver._coolprop_reference('ethanol', props)
        self.assertIsNotNone(reference)
        self.assertEqual(reference.qualified_name, 'HEOS::Ethanol')

    def test_unique_perry_formula_identity_reaches_coolprop(self):
        resolver = PropertyResolver()
        expected = {
            'Ne': ('7440-01-9', 'HEOS::Neon'),
            'N2O': ('10024-97-2', 'HEOS::NitrousOxide'),
            'CH3Cl': ('74-87-3', 'HEOS::R40'),
        }

        for symbol, (cas, qualified_name) in expected.items():
            with self.subTest(symbol=symbol):
                props = resolver._coerce_props(symbol, None, allow_online=False)
                self.assertEqual(props['CAS'], cas)
                reference = resolver._coolprop_reference(symbol, props)
                self.assertIsNotNone(reference)
                self.assertEqual(reference.qualified_name, qualified_name)

                critical = resolver.resolve_critical_properties(
                    symbol,
                    allow_online=False,
                    allow_estimation=False,
                )
                for item in critical.values():
                    self.assertIsNotNone(item.value)
                    self.assertClose(item.quality, 0.995)
                self.assertEqual(critical['Tc'].method, 'coolprop_HEOS_critical')
                self.assertEqual(critical['Zc'].method, 'coolprop_critical_identity')

    def test_perry_identity_without_coolprop_fluid_keeps_perry_criticals(self):
        resolver = PropertyResolver()
        critical = resolver.resolve_critical_properties(
            'C2H5Cl',
            allow_online=False,
            allow_estimation=False,
        )

        for item in critical.values():
            self.assertEqual(item.method, 'perry_critical')
            self.assertClose(item.quality, 0.98)

    def test_chloroform_criticals_are_not_direct_local_values(self):
        resolver = PropertyResolver()
        critical = resolver.resolve_critical_properties(
            'CHCl3',
            allow_online=False,
            allow_estimation=False,
        )

        for item in critical.values():
            self.assertEqual(item.method, 'perry_critical')
            self.assertClose(item.quality, 0.98)

    def test_coolprop_resolves_primary_phase_change_anchors(self):
        resolver = PropertyResolver()
        props = {'CAS': '64-17-5', 'name': 'ethanol'}

        tb = resolver.resolve_boiling_point(
            'ethanol',
            props,
            allow_online=False,
            allow_estimation=False,
        )
        triple = resolver.resolve_triple_point(
            'ethanol',
            props,
            allow_online=False,
        )

        self.assertEqual(tb.method, 'coolprop_HEOS_boiling_point')
        self.assertAlmostEqual(tb.value, 351.57040446751455)
        self.assertAlmostEqual(tb.quality, 0.995)
        self.assertEqual(triple['Tt'].method, 'coolprop_HEOS_triple_point')
        self.assertAlmostEqual(triple['Tt'].value, 159.1)
        self.assertAlmostEqual(triple['Pt'].value, 7.353928225172699e-09)

    def test_hydration_preserves_independent_tm_for_triple_confirmation(self):
        oxygen = ChemicalDatabase(enable_online=False).get(
            'oxygen',
            fetch_online=False,
        )
        self.assertIsNotNone(oxygen)
        melting_source = oxygen.property_sources['Tm']
        validator = melting_source.get('independent_validator')
        self.assertIsInstance(validator, dict)
        self.assertAlmostEqual(validator['value'], 54.4)
        self.assertEqual(validator['method'], 'direct')
        self.assertEqual(oxygen.property_sources['Tt']['quality'], 0.995)
        self.assertIn(
            'independent provided/direct Tm=54.4 K',
            oxygen.property_sources['Tt']['notes'],
        )

    def test_hydration_resolves_phase_anchors_and_tb_before_criticals(self):
        calls = []
        critical_seen = {}
        missing = lambda name: PropertyResolutionResult(
            None, 'missing', 'none', 0.0, f'{name} unavailable'
        )

        class ResolverFixture:
            def resolve_melting_point(self, _symbol, _props, **_options):
                calls.append('Tm')
                return missing('Tm')

            def resolve_triple_point(self, _symbol, _props, **_options):
                calls.append('Tt/Pt')
                return {'Tt': missing('Tt'), 'Pt': missing('Pt')}

            def resolve_boiling_point(self, _symbol, _props, **_options):
                calls.append('Tb')
                return PropertyResolutionResult(
                    410.0,
                    'local',
                    'fixture_source_backed_tb',
                    0.97,
                    'fixture normal boiling point',
                )

            def resolve_critical_properties(self, _symbol, props, **_options):
                calls.append('criticals')
                critical_seen.update(props)
                return {}

            def resolve_hvap(self, _symbol, _props, **_options):
                calls.append('Hvap')
                return missing('Hvap')

            def resolve_hfus(self, _symbol, _props, **_options):
                calls.append('Hfus')
                return missing('Hfus')

            def resolve_formation_properties(self, _symbol, _props, **_options):
                calls.append('formation')
                return {}

        component = ChemicalProperties(
            symbol='XORDER',
            name='ordering fixture',
            formula='C8H10',
            MW=106.17,
        )
        database = ChemicalDatabase(enable_online=False)
        with patch(
            'property_resolver.get_property_resolver',
            return_value=ResolverFixture(),
        ):
            database._hydrate_properties(component, allow_online=False)

        self.assertEqual(
            calls,
            ['Tm', 'Tt/Pt', 'Tb', 'criticals', 'Hvap', 'Hfus', 'formation'],
        )
        self.assertEqual(critical_seen['Tb'], 410.0)
        self.assertEqual(
            critical_seen['property_sources']['Tb']['method'],
            'fixture_source_backed_tb',
        )

    def test_soft_triple_pressure_does_not_control_stp_phase(self):
        provisional = ChemicalProperties(
            symbol='XSOFT',
            name='soft triple fixture',
            formula='X',
            MW=10.0,
            Tt=250.0,
            Pt=2.0,
            property_sources={
                key: {
                    'source': 'local',
                    'method': 'provisional_triple_fixture',
                    'quality': 0.89,
                }
                for key in ('Tt', 'Pt')
            },
        )
        self.assertIsNone(_inferred_phase_at_stp(provisional))

        confirmed = ChemicalProperties(
            **{
                **provisional.to_dict(),
                'property_sources': {
                    key: {
                        'source': 'local',
                        'method': 'confirmed_triple_fixture',
                        'quality': 0.995,
                    }
                    for key in ('Tt', 'Pt')
                },
            }
        )
        self.assertEqual(_inferred_phase_at_stp(confirmed), 'gas')

    def test_triple_pressure_rejects_nonexistent_normal_boiling_point(self):
        resolver = PropertyResolver()
        props = {
            'CAS': '124-38-9',
            'name': 'carbon dioxide',
            'Tb': 194.7,
            'Tt': 216.592,
            'Pt': 5.179643434477257,
            'property_sources': {
                key: {
                    'source': 'local',
                    'method': f'fixture_{key}',
                    'quality': 0.995,
                }
                for key in ('Tb', 'Tt', 'Pt')
            },
        }

        result = resolver.resolve_boiling_point(
            'carbon dioxide',
            props,
            allow_online=False,
            allow_estimation=False,
        )

        self.assertIsNone(result.value)
        self.assertEqual(result.method, 'no_normal_boiling_point_at_1atm')
        self.assertIn('5.17964 bar', result.notes)

    def test_hard_triple_topology_rejects_even_higher_quality_tb(self):
        resolver = PropertyResolver()
        props = {
            'name': 'hard triple topology fixture',
            'Tb': 300.0,
            'Tt': 250.0,
            'Pt': 2.0,
            'property_sources': {
                'Tb': {
                    'source': 'local',
                    'method': 'higher_quality_tb_fixture',
                    'quality': 0.99,
                },
                'Tt': {
                    'source': 'local',
                    'method': 'hard_triple_fixture',
                    'quality': 0.95,
                },
                'Pt': {
                    'source': 'local',
                    'method': 'hard_triple_fixture',
                    'quality': 0.95,
                },
            },
        }
        result = resolver.resolve_boiling_point(
            'hard triple topology fixture',
            props,
            allow_online=False,
            allow_estimation=False,
        )
        self.assertIsNone(result.value)
        self.assertEqual(result.method, 'no_normal_boiling_point_at_1atm')

    def test_provisional_triple_rejects_only_lower_quality_tb(self):
        resolver = PropertyResolver()
        base = {
            'name': 'provisional triple quality fixture',
            'Tb': 300.0,
            'Tt': 250.0,
            'Pt': 2.0,
            'property_sources': {
                'Tt': {
                    'source': 'local',
                    'method': 'provisional_triple_fixture',
                    'quality': 0.89,
                },
                'Pt': {
                    'source': 'local',
                    'method': 'provisional_triple_fixture',
                    'quality': 0.89,
                },
            },
        }
        cases = (
            (0.88, None, 'no_normal_boiling_point_at_1atm'),
            (0.89, 300.0, 'tb_fixture'),
            (0.95, 300.0, 'tb_fixture'),
        )
        for quality, expected_value, expected_method in cases:
            with self.subTest(Tb_quality=quality):
                props = {
                    **base,
                    'property_sources': {
                        **base['property_sources'],
                        'Tb': {
                            'source': 'local',
                            'method': 'tb_fixture',
                            'quality': quality,
                        },
                    },
                }
                result = resolver.resolve_boiling_point(
                    f'provisional triple fixture {quality}',
                    props,
                    allow_online=False,
                    allow_estimation=False,
                )
                self.assertEqual(result.value, expected_value)
                self.assertEqual(result.method, expected_method)
                if expected_value is None:
                    self.assertIn(
                        f'Tb quality={quality:g}',
                        result.notes,
                    )

    def test_provisional_ttriple_quality_arbitrates_subtriple_tb(self):
        resolver = PropertyResolver()
        base = {
            'name': 'provisional Tt quality fixture',
            'Tb': 240.0,
            'Tt': 250.0,
            'property_sources': {
                'Tt': {
                    'source': 'local',
                    'method': 'provisional_tt_fixture',
                    'quality': 0.89,
                },
            },
        }
        for quality, rejected in ((0.88, True), (0.89, False), (0.95, False)):
            with self.subTest(Tb_quality=quality):
                props = {
                    **base,
                    'property_sources': {
                        **base['property_sources'],
                        'Tb': {
                            'source': 'local',
                            'method': 'tb_fixture',
                            'quality': quality,
                        },
                    },
                }
                result = resolver.resolve_boiling_point(
                    f'provisional Tt fixture {quality}',
                    props,
                    allow_online=False,
                    allow_estimation=False,
                )
                if rejected:
                    self.assertIsNone(result.value)
                    self.assertEqual(
                        result.method,
                        'invalid_normal_boiling_point_below_triple_point',
                    )
                else:
                    self.assertEqual(result.value, 240.0)
                    self.assertEqual(result.method, 'tb_fixture')

    def test_pfd_boiling_override_remains_authoritative_above_triple_pressure(self):
        resolver = PropertyResolver()
        props = {
            'CAS': '124-38-9',
            'name': 'custom carbon dioxide',
            'Tb': 190.0,
            'Tt': 216.0,
            'Pt': 5.0,
            'property_sources': {
                'Tb': {
                    'source': 'provided',
                    'method': 'pfd_component_override',
                    'quality': 1.0,
                },
                'Tt': {
                    'source': 'provided',
                    'method': 'pfd_component_override',
                    'quality': 1.0,
                },
                'Pt': {
                    'source': 'provided',
                    'method': 'pfd_component_override',
                    'quality': 1.0,
                },
            },
        }

        result = resolver.resolve_boiling_point(
            'custom carbon dioxide',
            props,
            allow_online=False,
            allow_estimation=False,
        )

        self.assertEqual(result.value, 190.0)
        self.assertEqual(result.method, 'pfd_component_override')

    def test_coolprop_resolves_primary_critical_properties(self):
        resolver = PropertyResolver()
        critical = resolver.resolve_critical_properties(
            'ethanol',
            {'CAS': '64-17-5', 'name': 'ethanol'},
            allow_online=False,
            allow_estimation=False,
        )

        self.assertEqual(critical['Tc'].method, 'coolprop_HEOS_critical')
        self.assertAlmostEqual(critical['Tc'].value, 514.7092848812961)
        self.assertAlmostEqual(critical['Pc'].value, 62.679145827020946)
        self.assertAlmostEqual(critical['Vc'].value, 168.6145483266295)
        self.assertEqual(critical['Zc'].method, 'coolprop_critical_identity')
        self.assertAlmostEqual(critical['omega'].value, 0.644)
        self.assertAlmostEqual(critical['Tc'].quality, 0.995)

    def test_coolprop_water_uses_if97_anchors(self):
        resolver = PropertyResolver()
        props = {'CAS': '7732-18-5', 'name': 'water'}

        tb = resolver.resolve_boiling_point(
            'water',
            props,
            allow_online=False,
            allow_estimation=False,
        )
        triple = resolver.resolve_triple_point('water', props, allow_online=False)
        critical = resolver.resolve_critical_properties(
            'water',
            props,
            allow_online=False,
            allow_estimation=False,
        )

        self.assertEqual(tb.method, 'coolprop_IF97_boiling_point')
        self.assertEqual(triple['Tt'].method, 'coolprop_IF97_triple_point')
        self.assertEqual(critical['Tc'].method, 'coolprop_IF97_critical')
        self.assertAlmostEqual(tb.value, 373.12430000048056)
        self.assertAlmostEqual(triple['Tt'].value, 273.16)
        self.assertAlmostEqual(critical['Pc'].value, 220.64000000320607)

    def test_pfd_critical_and_triple_overrides_beat_coolprop(self):
        resolver = PropertyResolver()
        props = {
            'CAS': '64-17-5',
            'name': 'ethanol',
            'Tc': 500.0,
            'Pc': 50.0,
            'Tt': 150.0,
            'Pt': 0.001,
            'Tb': 340.0,
            'Tm': 150.0,
            'property_sources': {
                key: {
                    'source': 'provided',
                    'method': 'pfd_component_override',
                    'quality': 1.0,
                }
                for key in ('Tc', 'Pc', 'Tt', 'Pt', 'Tb', 'Tm')
            },
        }

        critical = resolver.resolve_critical_properties(
            'ethanol',
            props,
            allow_online=False,
            allow_estimation=False,
        )
        triple = resolver.resolve_triple_point('ethanol', props, allow_online=False)
        tb = resolver.resolve_boiling_point(
            'ethanol',
            props,
            allow_online=False,
            allow_estimation=False,
        )
        tm = resolver.resolve_melting_point(
            'ethanol',
            props,
            allow_online=False,
        )

        self.assertEqual(critical['Tc'].method, 'pfd_component_override')
        self.assertEqual(critical['Tc'].value, 500.0)
        self.assertEqual(triple['Tt'].method, 'pfd_component_override')
        self.assertEqual(triple['Pt'].value, 0.001)
        self.assertEqual(tb.method, 'pfd_component_override')
        self.assertEqual(tb.value, 340.0)
        self.assertEqual(tm.method, 'pfd_component_override')
        self.assertEqual(tm.value, 150.0)

    def test_partial_pfd_critical_override_recomputes_zc_from_selected_bundle(self):
        resolver = PropertyResolver()
        props = {
            'CAS': '64-17-5',
            'name': 'ethanol',
            'Tc': 500.0,
            'Pc': 50.0,
            'property_sources': {
                key: {
                    'source': 'provided',
                    'method': 'pfd_component_override',
                    'quality': 1.0,
                }
                for key in ('Tc', 'Pc')
            },
        }

        critical = resolver.resolve_critical_properties(
            'ethanol',
            props,
            allow_online=False,
            allow_estimation=False,
        )

        expected_zc = (
            50.0 * 100000.0 * critical['Vc'].value * 1.0e-6
            / (8.314462618 * 500.0)
        )
        self.assertEqual(critical['Tc'].method, 'pfd_component_override')
        self.assertEqual(critical['Pc'].method, 'pfd_component_override')
        self.assertEqual(critical['Vc'].method, 'coolprop_HEOS_critical')
        self.assertEqual(critical['Zc'].method, 'critical_volume_identity')
        self.assertClose(critical['Zc'].value, expected_zc)
        self.assertClose(critical['Zc'].quality, 0.995)

    def test_partial_pfd_triple_override_does_not_mix_provider_pair(self):
        resolver = PropertyResolver()
        props = {
            'CAS': '64-17-5',
            'name': 'ethanol',
            'Tt': 150.0,
            'Pt': 7.353928225172699e-09,
            'property_sources': {
                'Tt': {
                    'source': 'provided',
                    'method': 'pfd_component_override',
                    'quality': 1.0,
                },
                'Pt': {
                    'source': 'local',
                    'method': 'coolprop_HEOS_triple_point',
                    'quality': 0.995,
                },
            },
        }

        triple = resolver.resolve_triple_point(
            'ethanol',
            props,
            allow_online=False,
        )

        self.assertEqual(triple['Tt'].method, 'pfd_component_override')
        self.assertEqual(triple['Tt'].value, 150.0)
        self.assertEqual(triple['Pt'].source, 'missing')
        self.assertIsNone(triple['Pt'].value)

    def test_hydration_discards_stale_coupled_values_after_partial_pfd_override(self):
        database = ChemicalDatabase(enable_online=False)
        ethanol = database.get('ethanol', fetch_online=False)
        ethanol.Tc = 500.0
        ethanol.Pc = 50.0
        ethanol.Tt = 150.0
        for attr in ('Tc', 'Pc', 'Tt'):
            ethanol.property_sources[attr] = {
                'source': 'provided',
                'method': 'pfd_component_override',
                'quality': 1.0,
            }

        database._hydrate_properties(ethanol, allow_online=False)

        expected_zc = (
            ethanol.Pc * 100000.0 * ethanol.Vc * 1.0e-6
            / (8.314462618 * ethanol.Tc)
        )
        self.assertClose(ethanol.Zc, expected_zc)
        self.assertEqual(
            ethanol.property_sources['Zc']['method'],
            'critical_volume_identity',
        )
        self.assertIsNone(ethanol.Pt)
        self.assertNotIn('Pt', ethanol.property_sources)

    def test_coolprop_viscosity_uses_supplied_pressure(self):
        resolver = PropertyResolver()
        viscosity = resolver.resolve_viscosity(
            'R11',
            298.15,
            phase='liquid',
            props={'CAS': '75-69-4', 'name': 'R11'},
            P=10.0,
            rho_molar=12.0,
        )

        self.assertEqual(viscosity.method, 'coolprop_HEOS_viscosity')
        self.assertClose(viscosity.value, 0.0004374222811141793)
        self.assertIn('liquid at P=10 bar', viscosity.notes)

    def test_coolprop_gas_viscosity_uses_supplied_pressure(self):
        resolver = PropertyResolver()
        props = {'CAS': '7732-18-5', 'name': 'water'}

        with patch.object(resolver, '_get_perry_evaluation', return_value=None):
            at_one_atm = resolver.resolve_viscosity(
                'water',
                473.15,
                phase='vapor',
                props=props,
                P=1.01325,
            )
            at_ten_bar = resolver.resolve_viscosity(
                'water',
                473.15,
                phase='vapor',
                props=props,
                P=10.0,
            )

        self.assertEqual(at_one_atm.method, 'coolprop_IF97_viscosity')
        self.assertClose(at_one_atm.value, 1.6203512281568972e-05)
        self.assertClose(at_ten_bar.value, 1.5876012565946373e-05)
        self.assertIn('vapor at P=10 bar', at_ten_bar.notes)

    def test_coolprop_unspecified_pressure_uses_phase_consistent_atmospheric_bound(self):
        resolver = PropertyResolver()
        props = {'CAS': '7732-18-5', 'name': 'water'}

        with patch.object(resolver, '_get_perry_evaluation', return_value=None):
            ordinary_liquid = resolver.resolve_viscosity(
                'water',
                300.0,
                phase='liquid',
                props=props,
            )
            hot_vapor = resolver.resolve_viscosity(
                'water',
                473.15,
                phase='vapor',
                props=props,
            )
            cold_vapor = resolver.resolve_viscosity(
                'water',
                300.0,
                phase='vapor',
                props=props,
            )

        self.assertClose(ordinary_liquid.value, 0.0008537422562299644)
        self.assertIn('max(1 atm, Psat=', ordinary_liquid.notes)
        self.assertClose(hot_vapor.value, 1.6203512281568972e-05)
        self.assertIn('min(1 atm, Psat=', hot_vapor.notes)
        self.assertClose(cold_vapor.value, 9.759577935514269e-06)
        self.assertIn('saturated vapor (min(1 atm, Psat=0.0353659 bar)', cold_vapor.notes)

    def test_coolprop_wins_over_perry_overlap_with_explicit_pressure(self):
        resolver = PropertyResolver()
        viscosity = resolver.resolve_viscosity(
            'water',
            300.0,
            phase='liquid',
            props={'CAS': '7732-18-5', 'name': 'water'},
            P=100.0,
        )

        self.assertEqual(viscosity.method, 'coolprop_IF97_viscosity')
        self.assertClose(viscosity.quality, 0.99)
        self.assertClose(viscosity.value, 0.0008529928996950898)
        self.assertIn('liquid at P=100 bar', viscosity.notes)

    def test_lucas_backcalculates_saturation_viscosity_from_one_atm_baseline(self):
        resolver = PropertyResolver()
        baseline = PropertyResolutionResult(
            value=0.00068,
            source='local',
            method='perry_liquid_viscosity_eq101',
            quality=0.97,
            notes='mock Perry baseline',
        )
        baseline.units = 'Pa*s'
        critical = {
            'Tc': PropertyResolutionResult(572.2, 'provided', 'direct', 1.0, ''),
            'Pc': PropertyResolutionResult(34.7, 'provided', 'direct', 1.0, ''),
            'omega': PropertyResolutionResult(0.236, 'provided', 'direct', 1.0, ''),
        }
        psat = PropertyResolutionResult(0.0, 'provided', 'direct', 1.0, '')

        with patch.object(resolver, '_coolprop_viscosity', return_value=None), \
             patch.object(resolver, '_get_perry_evaluation', return_value=baseline), \
             patch.object(resolver, 'resolve_critical_properties', return_value=critical), \
             patch.object(resolver, 'resolve_vapor_pressure', return_value=psat):
            viscosity = resolver.resolve_viscosity(
                'methylcyclohexane-like',
                300.0,
                phase='liquid',
                props={},
                P=500.0,
            )

        self.assertEqual(viscosity.method, 'lucas_compressed_liquid_viscosity')
        self.assertClose(viscosity.value, 0.001066652756481425)
        self.assertClose(viscosity.quality, 0.679)
        self.assertIn('back-calculated saturation viscosity', viscosity.notes)
        self.assertIn('F(P)/F(1 atm)=1.56861', viscosity.notes)
        self.assertIn('pressure-quality multiplier 0.70', viscosity.notes)

    def test_lucas_pressure_quality_bins_follow_coolprop_benchmark(self):
        resolver = PropertyResolver()
        expected = {
            5.0: 1.00,
            10.0: 0.99,
            20.0: 0.97,
            50.0: 0.95,
            100.0: 0.89,
            250.0: 0.85,
            500.0: 0.70,
            1000.0: 0.55,
            3000.0: 0.55,
            3001.0: 0.55,
        }
        for pressure, factor in expected.items():
            with self.subTest(pressure=pressure):
                self.assertClose(
                    resolver._lucas_pressure_quality_factor(pressure),
                    factor,
                )

    def test_lucas_uncorrected_pressure_quality_bins(self):
        resolver = PropertyResolver()
        expected = {
            2.0: 1.00,
            10.0: 0.96,
            20.0: 0.91,
            50.0: 0.85,
            100.0: 0.73,
            250.0: 0.55,
            500.0: 0.45,
            1000.0: 0.45,
        }
        for pressure, factor in expected.items():
            with self.subTest(pressure=pressure):
                self.assertClose(
                    resolver._lucas_uncorrected_pressure_quality_factor(pressure),
                    factor,
                )

    def test_lucas_corrected_quality_uses_critical_uncertainty_formula(self):
        resolver = PropertyResolver()
        baseline = PropertyResolutionResult(
            value=0.00068,
            source='local',
            method='perry_liquid_viscosity_eq101',
            quality=0.97,
            notes='mock Perry baseline',
        )
        inputs = tuple(
            PropertyResolutionResult(1.0, 'provided', 'direct', quality, '')
            for quality in (0.95, 0.90, 0.92, 0.94)
        )

        quality, uncorrected, applied_factor, fallback_factor, critical_quality = (
            resolver._lucas_corrected_quality(baseline, 100.0, inputs)
        )

        self.assertClose(
            quality,
            0.97 * 0.89 * (1.0 - 11.5 * 0.10**1.8 * (1.0 - 0.73)),
        )
        self.assertClose(uncorrected, 0.97 * 0.73)
        self.assertClose(applied_factor, 0.89)
        self.assertClose(fallback_factor, 0.73)
        self.assertClose(critical_quality, 0.90)

        high_pressure = resolver._lucas_corrected_quality(
            baseline,
            1000.0,
            inputs,
        )
        self.assertClose(
            high_pressure[0],
            0.97 * 0.55 * (1.0 - 11.5 * 0.10**1.8 * (1.0 - 0.45)),
        )
        self.assertClose(high_pressure[1], 0.97 * 0.45)
        self.assertGreater(high_pressure[0], high_pressure[1])

    def test_lucas_skips_correction_when_corrected_quality_is_worse(self):
        resolver = PropertyResolver()
        baseline = PropertyResolutionResult(
            value=0.00068,
            source='local',
            method='perry_liquid_viscosity_eq101',
            quality=0.97,
            notes='mock Perry baseline',
        )
        baseline.units = 'Pa*s'
        critical = {
            'Tc': PropertyResolutionResult(572.2, 'calculated', 'estimate', 0.50, ''),
            'Pc': PropertyResolutionResult(34.7, 'calculated', 'estimate', 0.50, ''),
            'omega': PropertyResolutionResult(0.236, 'calculated', 'estimate', 0.50, ''),
        }
        psat = PropertyResolutionResult(0.0, 'calculated', 'estimate', 0.50, '')

        with patch.object(resolver, '_coolprop_viscosity', return_value=None), \
             patch.object(resolver, '_get_perry_evaluation', return_value=baseline), \
             patch.object(resolver, 'resolve_critical_properties', return_value=critical), \
             patch.object(resolver, 'resolve_vapor_pressure', return_value=psat):
            viscosity = resolver.resolve_viscosity(
                'methylcyclohexane-like',
                300.0,
                phase='liquid',
                props={},
                P=100.0,
            )

        self.assertEqual(viscosity.method, baseline.method)
        self.assertClose(viscosity.value, baseline.value)
        self.assertClose(viscosity.quality, 0.97 * 0.73)
        self.assertIn('corrected quality', viscosity.notes)
        self.assertIn('uncorrected pressure-quality multiplier 0.73', viscosity.notes)

    def test_lucas_preserves_one_atm_baseline_without_resolving_inputs(self):
        resolver = PropertyResolver()
        baseline = PropertyResolutionResult(
            value=0.00068,
            source='local',
            method='perry_liquid_viscosity_eq101',
            quality=0.97,
            notes='mock Perry baseline',
        )
        baseline.units = 'Pa*s'

        with patch.object(resolver, '_coolprop_viscosity', return_value=None), \
             patch.object(resolver, '_get_perry_evaluation', return_value=baseline), \
             patch.object(
                 resolver,
                 'resolve_critical_properties',
                 side_effect=AssertionError('Lucas inputs should not resolve at 1 atm'),
             ):
            viscosity = resolver.resolve_viscosity(
                'methylcyclohexane-like',
                300.0,
                phase='liquid',
                props={},
                P=1.01325,
            )

        self.assertEqual(viscosity.method, 'perry_liquid_viscosity_eq101')
        self.assertClose(viscosity.value, baseline.value)

    def test_lucas_keeps_baseline_when_saturation_pressure_is_unavailable(self):
        resolver = PropertyResolver()
        baseline = PropertyResolutionResult(
            value=0.00068,
            source='local',
            method='perry_liquid_viscosity_eq101',
            quality=0.97,
            notes='mock Perry baseline',
        )
        baseline.units = 'Pa*s'
        critical = {
            'Tc': PropertyResolutionResult(572.2, 'provided', 'direct', 1.0, ''),
            'Pc': PropertyResolutionResult(34.7, 'provided', 'direct', 1.0, ''),
            'omega': PropertyResolutionResult(0.236, 'provided', 'direct', 1.0, ''),
        }

        with patch.object(resolver, '_coolprop_viscosity', return_value=None), \
             patch.object(resolver, '_get_perry_evaluation', return_value=baseline), \
             patch.object(resolver, 'resolve_critical_properties', return_value=critical), \
             patch.object(
                 resolver,
                 'resolve_vapor_pressure',
                 side_effect=PropertyResolutionError('missing'),
             ):
            viscosity = resolver.resolve_viscosity(
                'methylcyclohexane-like',
                300.0,
                phase='liquid',
                props={},
                P=500.0,
            )

        self.assertEqual(viscosity.method, 'perry_liquid_viscosity_eq101')
        self.assertClose(viscosity.value, baseline.value)
        self.assertClose(viscosity.quality, 0.97 * 0.45)
        self.assertIn('Lucas compressed-liquid correction skipped: Psat unavailable', viscosity.notes)
        self.assertIn('uncorrected pressure-quality multiplier 0.45', viscosity.notes)

    def test_coolprop_wins_over_perry_without_pressure(self):
        resolver = PropertyResolver()
        viscosity = resolver.resolve_viscosity(
            'water',
            300.0,
            phase='liquid',
            props={'CAS': '7732-18-5', 'name': 'water'},
        )

        self.assertEqual(viscosity.method, 'coolprop_IF97_viscosity')
        self.assertClose(viscosity.quality, 0.99)
        self.assertClose(viscosity.value, 0.0008537422562299644)
        self.assertIn('max(1 atm, Psat=', viscosity.notes)

    def test_provided_pressure_viscosity_fit_uses_reduced_density(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C3H8',
            'Tc': 400.0,
            'Pc': 50.0,
            'Zc': 0.25,
            'property_correlations': {
                'mug': {
                    'equation': 'viscosity_exp_rhor',
                    'quality': 0.98,
                    'coefficients': {
                        'A': math.log(10.0),
                        'B': 0.0,
                        'C': 0.0,
                        'D': 0.0,
                        'E': math.log(2.0),
                        'F': 0.0,
                        'x': 0.5,
                        'y': 0.25,
                    },
                },
            },
        }
        rho_c = 50.0 / (0.25 * (8.314462618 / 100.0) * 400.0)

        with patch.object(resolver, '_coolprop_viscosity', return_value=None):
            viscosity = resolver.resolve_viscosity(
                'provided-propane',
                400.0,
                phase='vapor',
                props=props,
                P=50.0,
                rho_molar=0.5 * rho_c,
            )

        self.assertEqual(viscosity.method, 'provided_pressure_viscosity_fit')
        self.assertClose(viscosity.value, 11.3125e-6)
        self.assertClose(viscosity.quality, 0.98)
        self.assertIn('rho_r=0.5', viscosity.notes)
        self.assertIn('critical density from Zc=0.25', viscosity.notes)

    def test_provided_pressure_viscosity_fit_uses_dilute_term_without_pressure(self):
        resolver = PropertyResolver()
        props = {
            'property_correlations': {
                'mug': {
                    'equation': 'viscosity_exp_rhor',
                    'coefficients': {'A': math.log(10.0)},
                },
            },
        }

        viscosity = resolver.resolve_viscosity(
            'provided-gas',
            400.0,
            phase='vapor',
            props=props,
        )

        self.assertEqual(viscosity.method, 'provided_pressure_viscosity_fit')
        self.assertClose(viscosity.value, 10.0e-6)
        self.assertClose(viscosity.quality, 0.98)
        self.assertIn('dilute-density term with P unspecified', viscosity.notes)

    def test_pressure_fit_without_density_falls_through_to_coolprop(self):
        resolver = PropertyResolver()
        props = {
            'property_correlations': {
                'mug': {
                    'equation': 'viscosity_exp_rhor',
                    'coefficients': {'A': math.log(10.0)},
                },
            },
        }
        coolprop = PropertyResolutionResult(
            value=2.0e-5,
            source='local',
            method='coolprop_viscosity',
            quality=0.99,
            notes='mock actual-pressure value',
        )

        with patch.object(resolver, '_coolprop_viscosity', return_value=coolprop):
            viscosity = resolver.resolve_viscosity(
                'provided-gas',
                400.0,
                phase='vapor',
                props=props,
                P=50.0,
            )

        self.assertIs(viscosity, coolprop)

    def test_viscosity_rejects_invalid_optional_state(self):
        resolver = PropertyResolver()

        for state in ({'P': 0.0}, {'P': float('nan')}, {'rho_molar': -1.0}):
            with self.subTest(state=state):
                with self.assertRaises(PropertyResolutionError):
                    resolver.resolve_viscosity(
                        'water',
                        300.0,
                        phase='liquid',
                        props={'CAS': '7732-18-5', 'name': 'water'},
                        **state,
                    )

    def test_jossi_stiel_thodos_reproduces_perry_co2_example(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'CO2',
            'MW': 44.01,
            'Tc': 304.21,
            'Pc': 73.83,
            'Zc': 0.274,
        }
        critical = resolver.resolve_critical_properties(
            'CO2',
            props,
            allow_online=False,
        )
        rho_c = critical['Pc'].value / (
            critical['Zc'].value
            * (8.314462618 / 100.0)
            * critical['Tc'].value
        )
        baseline = PropertyResolutionResult(
            value=1.74e-5,
            source='local',
            method='perry_vapor_viscosity_eq102',
            quality=0.97,
            notes='Perry low-pressure baseline',
        )
        baseline.units = 'Pa*s'

        with patch.object(resolver, '_coolprop_viscosity', return_value=None), \
             patch.object(resolver, '_get_perry_evaluation', return_value=baseline):
            corrected = resolver.resolve_viscosity(
                'CO2',
                350.0,
                phase='vapor',
                props=props,
                P=200.0,
                rho_molar=1.295 * rho_c,
            )

        self.assertEqual(corrected.method, 'jossi_stiel_thodos_nonpolar')
        self.assertClose(corrected.value, 4.8867620256836254e-05)
        self.assertClose(corrected.quality, 0.7081)
        self.assertIn('rho_r=1.295', corrected.notes)
        self.assertIn('pressure-quality multiplier 0.73', corrected.notes)
        self.assertIn('symmetric molecule override', corrected.notes)

    def test_jossi_requires_both_explicit_pressure_and_density(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'CO2', 'MW': 44.01, 'Tc': 304.21, 'Pc': 73.83, 'Zc': 0.274,
        }
        baseline = PropertyResolutionResult(
            value=1.74e-5,
            source='local',
            method='perry_vapor_viscosity_eq102',
            quality=0.97,
            notes='Perry low-pressure baseline',
        )
        baseline.units = 'Pa*s'

        with patch.object(resolver, '_coolprop_viscosity', return_value=None), \
             patch.object(resolver, '_get_perry_evaluation', return_value=baseline):
            density_only = resolver.resolve_viscosity(
                'CO2', 350.0, phase='vapor', props=props, rho_molar=13.8,
            )
            pressure_only = resolver.resolve_viscosity(
                'CO2', 350.0, phase='vapor', props=props, P=200.0,
            )

        self.assertEqual(density_only.method, 'perry_vapor_viscosity_eq102')
        self.assertEqual(pressure_only.method, 'perry_vapor_viscosity_eq102')
        self.assertClose(density_only.value, 1.74e-5)
        self.assertClose(pressure_only.value, 1.74e-5)

    def test_jossi_polarity_uses_unifac_defaults_and_association_guards(self):
        resolver = PropertyResolver()
        cases = (
            ('propane', {'formula': 'C3H8', 'smiles': 'CCC'}, 'nonpolar'),
            ('acetone', {'formula': 'C3H6O', 'smiles': 'CC(=O)C'}, 'polar'),
            ('acetic acid', {'formula': 'C2H4O2', 'smiles': 'CC(=O)O'}, None),
            ('methanol', {'formula': 'CH4O', 'smiles': 'CO'}, None),
            ('1-butanol', {'formula': 'C4H10O', 'smiles': 'CCCCO'}, 'polar'),
            ('hexafluoroethane', {'formula': 'C2F6', 'smiles': 'FC(F)(F)C(F)(F)F'}, 'nonpolar'),
        )

        for symbol, props, expected in cases:
            with self.subTest(symbol=symbol):
                branch, _, _ = resolver._jossi_polarity(symbol, props)
                self.assertEqual(branch, expected)

        with patch.object(resolver, '_jossi_unifac_groups', return_value=None):
            branch, factor, note = resolver._jossi_polarity(
                'unresolved chloride',
                {'formula': 'C5H9Cl'},
            )
        self.assertEqual(branch, 'polar')
        self.assertClose(factor, 0.75)
        self.assertIn('contains heteroatoms', note)

    def test_jossi_polarity_interprets_native_dortmund_numeric_groups(self):
        resolver = PropertyResolver()
        cases = (
            ('native methanol', {'formula': 'CH4O', 'smiles': 'CO'}, None, {'15': 1}),
            ('native acid', {'formula': 'C2H4O2', 'smiles': 'CC(=O)O'}, None,
             {'1': 1, '42': 1}),
            ('native amine', {'formula': 'C2H7N', 'smiles': 'CCN'}, None,
             {'1': 1, '29': 1}),
            ('native amide', {'formula': 'C2H5NO', 'smiles': 'CC(=O)N'}, None,
             {'1': 1, '91': 1}),
            ('native butanol', {'formula': 'C4H10O', 'smiles': 'CCCCO'}, 'polar',
             {'1': 1, '2': 3, '14': 1}),
        )

        for symbol, props, expected_branch, expected_groups in cases:
            with self.subTest(symbol=symbol):
                groups = resolver._jossi_unifac_groups(symbol, props)
                self.assertEqual(dict(groups), expected_groups)
                branch, _, _ = resolver._jossi_polarity(symbol, props)
                self.assertEqual(branch, expected_branch)

    def test_jossi_associating_gas_keeps_low_pressure_baseline(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'CH4O',
            'smiles': 'CO',
            'MW': 32.04,
            'Tc': 512.6,
            'Pc': 80.9,
            'Zc': 0.224,
        }
        baseline = PropertyResolutionResult(
            value=1.0e-5,
            source='local',
            method='perry_vapor_viscosity_eq102',
            quality=0.97,
            notes='Perry low-pressure baseline',
        )
        baseline.units = 'Pa*s'

        with patch.object(resolver, '_coolprop_viscosity', return_value=None), \
             patch.object(resolver, '_get_perry_evaluation', return_value=baseline):
            result = resolver.resolve_viscosity(
                'methanol',
                450.0,
                phase='vapor',
                props=props,
                P=100.0,
                rho_molar=5.0,
            )

        self.assertEqual(result.method, 'perry_vapor_viscosity_eq102')
        self.assertClose(result.value, baseline.value)
        self.assertIn('Jossi pressure correction skipped', result.notes)
        self.assertIn('small alcohol or amine', result.notes)

    def test_jossi_applies_polar_branch_from_unifac_classification(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C3H6O',
            'smiles': 'CC(=O)C',
            'MW': 58.08,
            'Tc': 508.1,
            'Pc': 47.0,
            'Zc': 0.233,
        }
        rho_c = 47.0 / (0.233 * (8.314462618 / 100.0) * 508.1)
        baseline = PropertyResolutionResult(
            value=1.0e-5,
            source='local',
            method='perry_vapor_viscosity_eq102',
            quality=0.97,
            notes='Perry low-pressure baseline',
        )
        baseline.units = 'Pa*s'

        with patch.object(resolver, '_coolprop_viscosity', return_value=None), \
             patch.object(resolver, '_get_perry_evaluation', return_value=baseline):
            result = resolver.resolve_viscosity(
                'acetone',
                450.0,
                phase='vapor',
                props=props,
                P=100.0,
                rho_molar=rho_c,
            )

        self.assertEqual(result.method, 'jossi_stiel_thodos_polar')
        self.assertGreater(result.value, baseline.value)
        self.assertClose(result.quality, 0.484709)
        self.assertIn('Dortmund groups classified as polar', result.notes)

    def test_jossi_pressure_quality_bins_follow_coolprop_benchmark(self):
        resolver = PropertyResolver()
        expected = (
            ('nonpolar', 0.10, 0.940),
            ('nonpolar', 0.20, 0.915),
            ('nonpolar', 0.50, 0.842),
            ('nonpolar', 0.62, 0.842),
            ('nonpolar', 0.75, 0.785),
            ('nonpolar', 1.295, 0.730),
            ('nonpolar', 2.60, 0.50),
            ('polar', 0.10, 0.957),
            ('polar', 0.20, 0.921),
            ('polar', 0.50, 0.838),
            ('polar', 0.75, 0.688),
            ('polar', 1.0, 0.526),
            ('polar', 1.25, 0.50),
            ('polar', 2.60, 0.50),
        )
        for branch, rho_r, factor in expected:
            with self.subTest(branch=branch, rho_r=rho_r):
                self.assertClose(
                    resolver._jossi_pressure_quality_factor(branch, rho_r),
                    factor,
                )

    def test_jossi_piecewise_polar_equations_cover_all_density_bands(self):
        resolver = PropertyResolver()
        expected = {
            0.05: 0.05937669320153712,
            0.5: 1.0504804971528359,
            1.0: 3.2016928248854377,
            2.4: 31.633378324506364,
        }

        for rho_r, increment in expected.items():
            with self.subTest(rho_r=rho_r):
                self.assertClose(
                    resolver._jossi_dimensionless_increment('polar', rho_r),
                    increment,
                )

    def test_coolprop_viscosity_skips_missing_transport_model_and_falls_through_to_hsu(self):
        resolver = PropertyResolver()

        with patch.object(resolver, '_get_perry_evaluation', return_value=None):
            viscosity = resolver.resolve_viscosity(
                'acetone',
                298.15,
                phase='liquid',
                props={'CAS': '67-64-1', 'name': 'Acetone'},
        )

        self.assertEqual(viscosity.method, 'hsu_liquid_viscosity')
        self.assertClose(viscosity.value, 0.0002906827264208932, rel=1e-6)  # native hsu_method engine

    def test_yoon_thodos_estimates_gas_viscosity_after_local_sources(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C3H8',
            'MW': 44.0956,
            'Tc': 369.83,
            'Pc': 42.48,
        }

        with patch.object(resolver, '_get_perry_evaluation', return_value=None), \
             patch.object(resolver, '_coolprop_viscosity', return_value=None):
            viscosity = resolver.resolve_viscosity('propane', 353.0, phase='vapor', props=props)

        self.assertEqual(viscosity.method, 'yoon_thodos_gas_viscosity')
        self.assertEqual(viscosity.source, 'calculated')
        self.assertClose(viscosity.value, 9.842741087150416e-06)
        self.assertClose(viscosity.quality, 0.88)
        self.assertIn('hydrocarbon multiplier 0.88', viscosity.notes)

    def test_yoon_thodos_quality_uses_sparse_heteroatom_multiplier(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C4H10O',
            'MW': 74.12,
            'Tc': 563.0,
            'Pc': 44.0,
            'property_sources': {
                'Tc': {'quality': 0.93},
                'Pc': {'quality': 0.90},
            },
        }

        with patch.object(resolver, '_get_perry_evaluation', return_value=None), \
             patch.object(resolver, '_coolprop_viscosity', return_value=None), \
             patch.object(resolver, '_reichenberg_structure', return_value=(None, None)):
            viscosity = resolver.resolve_viscosity('butanol-ish', 350.0, phase='gas', props=props)

        self.assertEqual(viscosity.method, 'yoon_thodos_gas_viscosity')
        self.assertClose(viscosity.quality, 0.792)
        self.assertIn('slightly polar organic fallback multiplier 0.88', viscosity.notes)

    def test_yoon_thodos_quality_uses_heteroatom_rich_multiplier(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C2H6O',
            'MW': 46.07,
            'Tc': 514.0,
            'Pc': 61.4,
        }

        with patch.object(resolver, '_get_perry_evaluation', return_value=None), \
             patch.object(resolver, '_coolprop_viscosity', return_value=None), \
             patch.object(resolver, '_reichenberg_structure', return_value=(None, None)):
            viscosity = resolver.resolve_viscosity('ethanol-ish', 350.0, phase='vapor', props=props)

        self.assertEqual(viscosity.method, 'yoon_thodos_gas_viscosity')
        self.assertClose(viscosity.quality, 0.75)
        self.assertIn('polar fallback multiplier 0.75', viscosity.notes)

    def test_reichenberg_estimates_slightly_polar_organic_vapor_as_soft(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C4H10O',
            'smiles': 'CCCCO',
            'MW': 74.1216,
            'Tc': 563.0,
            'Pc': 44.0,
        }

        with patch.object(resolver, '_get_perry_evaluation', return_value=None), \
             patch.object(resolver, '_coolprop_viscosity', return_value=None):
            viscosity = resolver.resolve_viscosity(
                '1-butanol fixture', 350.0, phase='vapor', props=props,
            )

        self.assertEqual(
            viscosity.method,
            'reichenberg_zero_dipole_organic_gas_viscosity',
        )
        self.assertEqual(viscosity.source, 'estimated')
        self.assertClose(viscosity.quality, 0.91)
        self.assertTrue(resolver._result_is_soft(viscosity))
        self.assertIn('slightly polar organic', viscosity.notes)
        self.assertIn('5 heavy atoms', viscosity.notes)

    def test_reichenberg_uses_lower_quality_for_polar_organic_vapor(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C2H6O',
            'smiles': 'CCO',
            'MW': 46.0684,
            'Tc': 514.0,
            'Pc': 61.4,
        }

        with patch.object(resolver, '_get_perry_evaluation', return_value=None), \
             patch.object(resolver, '_coolprop_viscosity', return_value=None):
            viscosity = resolver.resolve_viscosity(
                'ethanol fixture', 350.0, phase='vapor', props=props,
            )

        self.assertEqual(
            viscosity.method,
            'reichenberg_zero_dipole_organic_gas_viscosity',
        )
        self.assertEqual(viscosity.source, 'estimated')
        self.assertClose(viscosity.quality, 0.86)
        self.assertIn('polar organic', viscosity.notes)

    def test_reichenberg_uses_inorganic_branch_for_three_heavy_atoms(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'SO2',
            'MW': 64.066,
            'Tc': 430.64,
            'Pc': 78.84,
        }

        with patch.object(resolver, '_get_perry_evaluation', return_value=None), \
             patch.object(resolver, '_coolprop_viscosity', return_value=None):
            viscosity = resolver.resolve_viscosity(
                'sulfur dioxide fixture', 350.0, phase='vapor', props=props,
            )

        self.assertEqual(
            viscosity.method,
            'reichenberg_zero_dipole_inorganic_gas_viscosity',
        )
        self.assertEqual(viscosity.source, 'estimated')
        self.assertClose(viscosity.quality, 0.80)
        self.assertIn('3 heavy atoms', viscosity.notes)

    def test_reichenberg_uses_inorganic_branch_when_carbon_has_no_hydrogen(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'CO2',
            'smiles': 'O=C=O',
            'MW': 44.0095,
            'Tc': 304.1282,
            'Pc': 73.773,
        }

        with patch.object(resolver, '_get_perry_evaluation', return_value=None), \
             patch.object(resolver, '_coolprop_viscosity', return_value=None):
            viscosity = resolver.resolve_viscosity(
                'carbon dioxide fixture', 350.0, phase='vapor', props=props,
            )

        self.assertEqual(
            viscosity.method,
            'reichenberg_zero_dipole_inorganic_gas_viscosity',
        )
        self.assertClose(viscosity.quality, 0.80)

    def test_reichenberg_quality_uses_weakest_required_input(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C4H10O',
            'smiles': 'CCCCO',
            'MW': 74.1216,
            'Tc': 563.0,
            'Pc': 44.0,
            'property_sources': {
                'MW': {'quality': 0.97},
                'Tc': {'quality': 0.94},
                'Pc': {'quality': 0.90},
                'smiles': {'quality': 0.96},
            },
        }

        with patch.object(resolver, '_get_perry_evaluation', return_value=None), \
             patch.object(resolver, '_coolprop_viscosity', return_value=None):
            viscosity = resolver.resolve_viscosity(
                '1-butanol quality fixture', 350.0, phase='vapor', props=props,
            )

        self.assertEqual(viscosity.source, 'estimated')
        self.assertClose(viscosity.quality, 0.90 * 0.91)

    def test_small_nonhydrocarbon_falls_back_to_low_quality_yoon_thodos(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'CH4O',
            'smiles': 'CO',
            'MW': 32.042,
            'Tc': 512.6,
            'Pc': 80.9,
        }

        with patch.object(resolver, '_get_perry_evaluation', return_value=None), \
             patch.object(resolver, '_coolprop_viscosity', return_value=None):
            viscosity = resolver.resolve_viscosity(
                'methanol fixture', 350.0, phase='vapor', props=props,
            )

        self.assertEqual(viscosity.method, 'yoon_thodos_gas_viscosity')
        self.assertClose(viscosity.quality, 0.50)
        self.assertIn('2 heavy atoms', viscosity.notes)

    def test_nonfragmentable_organics_fall_back_to_class_specific_yoon(self):
        resolver = PropertyResolver()
        cases = (
            (
                'triethylamine fixture',
                {
                    'formula': 'C6H15N', 'smiles': 'CCN(CC)CC',
                    'MW': 101.193, 'Tc': 535.0, 'Pc': 30.4,
                },
                0.88,
                'slightly polar',
            ),
            (
                'dimethyl sulfide fixture',
                {
                    'formula': 'C2H6S', 'smiles': 'CSC',
                    'MW': 62.134, 'Tc': 503.0, 'Pc': 55.3,
                },
                0.75,
                'polar',
            ),
        )

        for name, props, expected_quality, note in cases:
            with self.subTest(name=name), \
                 patch.object(resolver, '_get_perry_evaluation', return_value=None), \
                 patch.object(resolver, '_coolprop_viscosity', return_value=None):
                viscosity = resolver.resolve_viscosity(
                    name, 350.0, phase='vapor', props=props,
                )
            self.assertEqual(viscosity.method, 'yoon_thodos_gas_viscosity')
            self.assertClose(viscosity.quality, expected_quality)
            self.assertIn(note, viscosity.notes)

    def test_yoon_thodos_does_not_apply_to_liquid_viscosity(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C3H8',
            'MW': 44.0956,
            'Tc': 369.83,
            'Pc': 42.48,
        }

        with patch.object(resolver, '_get_perry_evaluation', return_value=None), \
             patch.object(resolver, '_coolprop_viscosity', return_value=None):
            with self.assertRaises(PropertyResolutionError):
                resolver.resolve_viscosity('propane', 300.0, phase='liquid', props=props)

    def test_nist_formation_parser_extracts_gas_thermochemistry(self):
        resolver = PropertyResolver()
        html = '''
            <table class="data" aria-label="One dimensional data">
                <tr><th>Quantity</th><th>Value</th><th>Units</th><th>Method</th><th>Reference</th><th>Comment</th></tr>
                <tr><td>&#916;<sub>f</sub>H&deg;<sub>gas</sub></td><td>-74.87</td><td>kJ/mol</td><td>Review</td><td>Chase, 1998</td><td></td></tr>
                <tr><td>&#916;<sub>f</sub>H&deg;<sub>gas</sub></td><td>-74.5 &plusmn; 0.4</td><td>kJ/mol</td><td>Ccb</td><td>ref</td><td></td></tr>
                <tr><td>&#916;<sub>c</sub>H&deg;<sub>gas</sub></td><td>-891.0</td><td>kJ/mol</td><td>Ccb</td><td>ref</td><td></td></tr>
                <tr><td>&#916;<sub>c</sub>H&deg;<sub>gas</sub></td><td>-889.0</td><td>kJ/mol</td><td>Ccb</td><td>ref</td><td></td></tr>
                <tr><td>S&deg;<sub>gas</sub></td><td>188.66</td><td>J/mol*K</td><td>N/A</td><td>ref</td><td></td></tr>
                <tr><td>S&deg;<sub>gas,1 bar</sub></td><td>186.25</td><td>J/mol*K</td><td>Review</td><td>Chase, 1998</td><td></td></tr>
                <tr><td>&#916;<sub>f</sub>H&deg;<sub>liquid</sub></td><td>-100.0</td><td>kJ/mol</td><td>Review</td><td>ref</td><td></td></tr>
                <tr><td>&#916;<sub>f</sub>G&deg;<sub>liquid</sub></td><td>-90.0</td><td>kJ/mol</td><td>Review</td><td>ref</td><td></td></tr>
                <tr><td>S&deg;<sub>liquid</sub></td><td>70.0</td><td>J/mol*K</td><td>Review</td><td>ref</td><td></td></tr>
            </table>
        '''

        parsed = resolver._parse_nist_formation_properties(html)

        self.assertClose(parsed['Hf'], -74.87)
        self.assertClose(parsed['Hf_liquid'], -100.0)
        self.assertClose(parsed['Gf_liquid'], -90.0)
        self.assertClose(parsed['Hcomb'], -890.0)
        self.assertClose(parsed['S'], 186.25)
        self.assertClose(parsed['S_liquid'], 70.0)
        self.assertEqual(parsed['_sources']['Hf'], 'nist_gas_thermochemistry')
        self.assertEqual(parsed['_sources']['Hf_liquid'], 'nist_liquid_thermochemistry')
        self.assertEqual(parsed['_sources']['Gf_liquid'], 'nist_liquid_thermochemistry')
        self.assertEqual(parsed['_sources']['S_liquid'], 'nist_liquid_thermochemistry')
        self.assertEqual(parsed['_sources']['Hcomb'], 'nist_gas_thermochemistry_median')

    def test_nist_formation_fetch_requests_gas_and_liquid_mask(self):
        resolver = PropertyResolver()
        html = '''
            <table class="data" aria-label="One dimensional data">
                <tr><th>Quantity</th><th>Value</th><th>Units</th><th>Method</th><th>Reference</th><th>Comment</th></tr>
                <tr><td>&#916;<sub>f</sub>H&deg;<sub>liquid</sub></td><td>-100.0</td><td>kJ/mol</td><td>Review</td><td>ref</td><td></td></tr>
            </table>
        '''
        urls = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return html.encode()

        def fake_urlopen(request, timeout):
            urls.append(request.full_url)
            return Response()

        with patch.object(resolver, '_get_cache', return_value=None), \
             patch.object(resolver, '_set_cache'), \
             patch('urllib.request.urlopen', side_effect=fake_urlopen):
            parsed = resolver._fetch_formation_nist('madeupium')

        self.assertClose(parsed['Hf_liquid'], -100.0)
        self.assertTrue(urls)
        self.assertIn('Mask=3', urls[0])

    def test_online_formation_resolution_uses_nist_for_missing_fields(self):
        resolver = PropertyResolver()
        nist = {
            'Hf': -74.87,
            'S': 186.25,
            'Hcomb': -890.0,
            '_sources': {
                'Hf': 'nist_gas_thermochemistry',
                'S': 'nist_gas_thermochemistry',
                'Hcomb': 'nist_gas_thermochemistry_median',
            },
            '_notes': {
                'Hf': 'NIST WebBook gas thermochemistry Review row',
                'S': 'NIST WebBook gas thermochemistry Review row',
                'Hcomb': 'NIST WebBook gas thermochemistry median of 2 direct row(s)',
            },
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None), \
             patch.object(resolver, '_fetch_formation_online', return_value=nist):
            formation = resolver.resolve_formation_properties('madeupium', {}, allow_online=True)

        self.assertClose(formation['Hf'].value, -74.87)
        self.assertEqual(formation['Hf'].source, 'online')
        self.assertEqual(formation['Hf'].method, 'nist_gas_thermochemistry')
        self.assertClose(formation['Hcomb'].value, -890.0)
        self.assertEqual(formation['Gf'].source, 'missing')

    def test_offline_formation_resolution_does_not_fetch_nist(self):
        resolver = PropertyResolver()

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None), \
             patch.object(resolver, '_fetch_formation_online') as fetch:
            formation = resolver.resolve_formation_properties('madeupium', {}, allow_online=False)

        fetch.assert_not_called()
        self.assertEqual(formation['Hf'].source, 'missing')

    def test_formation_uses_domalski_hearing_as_final_hf_s_rung(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C3H8',
            'smiles': 'CCC',
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            formation = resolver.resolve_formation_properties(
                'madeupium', props, allow_online=False,
            )

        expected = dh.estimate('CCC')
        correction = dh.R_J_MOL_K * math.log(dh.PRESSURE_PA / 100000.0)
        self.assertClose(
            formation['Hf'].value,
            expected.gas.enthalpy_formation_kJ_mol,
        )
        self.assertEqual(formation['Hf'].source, 'estimated')
        self.assertEqual(formation['Hf'].method, 'domalski_hearing_gas_hf')
        self.assertClose(formation['Hf'].quality, 0.85)
        self.assertClose(
            formation['S'].value,
            expected.gas.entropy_J_mol_K + correction,
        )
        self.assertEqual(formation['S'].source, 'estimated')
        self.assertEqual(
            formation['S'].method,
            'domalski_hearing_gas_entropy_1bar',
        )
        self.assertClose(formation['S'].quality, 0.89)
        self.assertNotIn('quality reduced for an S-S bonded compound', formation['S'].notes)

    def test_formation_domalski_hf_exclusion_does_not_exclude_s(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C2H6S2',
            'smiles': 'CSSC',
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            formation = resolver.resolve_formation_properties(
                'madeupium', props, allow_online=False,
            )

        self.assertIsNone(formation['Hf'].value)
        self.assertEqual(formation['Hf'].source, 'missing')
        self.assertEqual(
            formation['Hf'].method,
            'domalski_hearing_hf_not_applicable',
        )
        self.assertIn('sulfur-sulfur bond', formation['Hf'].notes)
        self.assertIsNotNone(formation['S'].value)
        self.assertEqual(
            formation['S'].method,
            'domalski_hearing_gas_entropy_1bar',
        )
        self.assertClose(formation['S'].quality, 0.75)
        self.assertIn('quality reduced for an S-S bonded compound', formation['S'].notes)

    def test_formation_domalski_empty_fragmentation_is_not_an_estimate(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C2N2',
            'smiles': 'N#CC#N',
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            formation = resolver.resolve_formation_properties(
                'madeupium', props, allow_online=False,
            )

        self.assertIsNone(formation['Hf'].value)
        self.assertEqual(formation['Hf'].source, 'missing')
        self.assertIsNone(formation['S'].value)
        self.assertEqual(formation['S'].source, 'missing')

    def test_formation_direct_hf_s_precede_domalski_hearing(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C3H8',
            'smiles': 'CCC',
            'Hf': -123.0,
            'S': 234.0,
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None), \
             patch.object(resolver, '_resolve_domalski_hearing_formation') as fallback:
            formation = resolver.resolve_formation_properties(
                'madeupium', props, allow_online=False,
            )

        fallback.assert_not_called()
        self.assertClose(formation['Hf'].value, -123.0)
        self.assertEqual(formation['Hf'].method, 'direct')
        self.assertClose(formation['S'].value, 234.0)
        self.assertEqual(formation['S'].method, 'direct')

    def test_formation_hf_derives_gas_from_local_liquid_hf_and_hvap(self):
        resolver = PropertyResolver()
        props = {
            'Hf_liquid': -100.0,
            'Hvap': 30.0,
            'Tb': 350.0,
            'Tc': 600.0,
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            formation = resolver.resolve_formation_properties('madeupium', props, allow_online=False)

        hvap_298 = 30.0 * ((1.0 - 298.15 / 600.0) / (1.0 - 350.0 / 600.0)) ** 0.38
        self.assertEqual(formation['Hf'].method, 'liquid_hf_plus_hvap')
        self.assertClose(formation['Hf'].value, -100.0 + hvap_298)
        self.assertIn('provided/direct', formation['Hf'].notes)

    def test_formation_hf_derives_gas_from_online_liquid_hf_when_needed(self):
        resolver = PropertyResolver()
        online = {
            'Hf_liquid': -100.0,
            '_sources': {'Hf_liquid': 'nist_liquid_thermochemistry'},
            '_notes': {'Hf_liquid': 'NIST liquid row'},
        }
        props = {
            'Hvap': 30.0,
            'Tb': 350.0,
            'Tc': 600.0,
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None), \
             patch.object(resolver, '_fetch_formation_online', return_value=online):
            formation = resolver.resolve_formation_properties('madeupium', props, allow_online=True)

        hvap_298 = 30.0 * ((1.0 - 298.15 / 600.0) / (1.0 - 350.0 / 600.0)) ** 0.38
        self.assertEqual(formation['Hf'].method, 'liquid_hf_plus_hvap')
        self.assertClose(formation['Hf'].value, -100.0 + hvap_298)
        self.assertIn('online/nist_liquid_thermochemistry', formation['Hf'].notes)

    def test_formation_gf_derives_from_hf_and_absolute_gas_entropy(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'H2O',
            'Hf': -241.83,
            'S': 188.84,
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            formation = resolver.resolve_formation_properties('madeupium', props, allow_online=False)

        element_s = 130.7 + 0.5 * 205.2
        expected = -241.83 - 298.15 * (188.84 - element_s) / 1000.0
        self.assertEqual(formation['Gf'].method, 'hf_absolute_entropy_to_gf')
        self.assertClose(formation['Gf'].value, expected)

    def test_formation_absolute_gas_entropy_derives_from_hf_and_gf(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'H2O',
            'Hf': -241.83,
            'Gf': -228.60,
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            formation = resolver.resolve_formation_properties('madeupium', props, allow_online=False)

        element_s = 130.7 + 0.5 * 205.2
        expected = (-241.83 - (-228.60)) * 1000.0 / 298.15 + element_s
        self.assertEqual(formation['S'].method, 'hf_gf_to_absolute_entropy')
        self.assertClose(formation['S'].value, expected)

    def test_formation_gf_and_s_derive_from_liquid_gf(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'H2O',
            'Hf': -241.83,
            'Gf_liquid': -237.10,
        }
        psat = PropertyResolutionResult(
            value=0.0317,
            source='provided',
            method='test_psat',
            quality=0.99,
        )

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None), \
             patch.object(resolver, 'resolve_vapor_pressure', return_value=psat):
            formation = resolver.resolve_formation_properties('madeupium', props, allow_online=False)

        element_s = 130.7 + 0.5 * 205.2
        expected_gf = -237.10 - 8.314462618 * 298.15 * math.log(0.0317) / 1000.0
        expected_s = (-241.83 - expected_gf) * 1000.0 / 298.15 + element_s
        self.assertEqual(formation['Gf'].method, 'liquid_gf_plus_standard_vaporization_gibbs')
        self.assertClose(formation['Gf'].value, expected_gf)
        self.assertClose(formation['Gf'].quality, 0.99 * 0.88)
        self.assertEqual(formation['S'].method, 'hf_gf_to_absolute_entropy')
        self.assertClose(formation['S'].value, expected_s)
        self.assertEqual(formation['S'].source, 'calculated')
        self.assertClose(formation['S'].quality, 0.99 * 0.88)

    def test_formation_gf_and_s_derive_from_liquid_entropy(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'H2O',
            'Hf': -241.83,
            'S_liquid': 69.91,
        }
        hvap = PropertyResolutionResult(
            value=43.99,
            source='provided',
            method='test_hvap',
            quality=0.99,
        )
        psat = PropertyResolutionResult(
            value=0.0317,
            source='provided',
            method='test_psat',
            quality=0.99,
        )

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None), \
             patch.object(resolver, 'resolve_hvap', return_value=hvap), \
             patch.object(resolver, 'resolve_vapor_pressure', return_value=psat):
            formation = resolver.resolve_formation_properties('madeupium', props, allow_online=False)

        element_s = 130.7 + 0.5 * 205.2
        hf_liquid = -241.83 - 43.99
        gf_liquid = hf_liquid - 298.15 * (69.91 - element_s) / 1000.0
        expected_gf = gf_liquid - 8.314462618 * 298.15 * math.log(0.0317) / 1000.0
        expected_s = (-241.83 - expected_gf) * 1000.0 / 298.15 + element_s
        self.assertEqual(formation['Gf'].method, 'liquid_gf_plus_standard_vaporization_gibbs')
        self.assertClose(formation['Gf'].value, expected_gf)
        self.assertEqual(formation['S'].method, 'hf_gf_to_absolute_entropy')
        self.assertClose(formation['S'].value, expected_s)

    def test_formation_hcomb_converts_gross_to_net(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'CH4',
            'Hcomb_gross': -890.33,
            'property_sources': {
                'Hcomb_gross': {'source': 'local', 'method': 'test_gross_hcomb', 'quality': 0.94},
            },
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            formation = resolver.resolve_formation_properties('madeupium', props, allow_online=False)

        self.assertEqual(formation['Hcomb'].method, 'gross_to_net_hcomb')
        self.assertClose(formation['Hcomb'].value, -890.33 + 2.0 * 43.99)
        self.assertEqual(formation['Hcomb'].source, 'exact')
        self.assertClose(formation['Hcomb'].quality, 0.94)

    def test_formation_hcomb_calculates_organic_halide_net_combustion(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'CH3Cl',
            'phase_at_STP': 'gas',
            'Hf': -85.7,
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            formation = resolver.resolve_formation_properties('madeupium', props, allow_online=False)

        expected = -393.51 + 1.5 * -241.826 - (-85.7)
        self.assertEqual(formation['Hcomb'].method, 'formula_net_hcomb')
        self.assertClose(formation['Hcomb'].value, expected)

    def test_formation_hcomb_uses_liquid_reactant_state(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'CS2',
            'phase_at_STP': 'liquid',
            'Hf': 116.9,
            'Hf_liquid': 89.7,
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            formation = resolver.resolve_formation_properties('madeupium', props, allow_online=False)

        expected = -393.51 + 2.0 * -296.84 - 89.7
        self.assertEqual(formation['Hcomb'].method, 'formula_net_hcomb')
        self.assertClose(formation['Hcomb'].value, expected)

    def test_formation_hcomb_uses_solid_hf_but_does_not_derive_it(self):
        resolver = PropertyResolver()

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            graphite = resolver.resolve_formation_properties(
                'madeupium',
                {
                    'formula': 'C',
                    'phase_at_STP': 'solid',
                    'Hf_solid': 0.0,
                },
                allow_online=False,
            )
            missing = resolver.resolve_formation_properties(
                'madeupium',
                {
                    'formula': 'C',
                    'phase_at_STP': 'solid',
                    'Hf': 716.7,
                },
                allow_online=False,
            )

        self.assertEqual(graphite['Hcomb'].method, 'formula_net_hcomb')
        self.assertClose(graphite['Hcomb'].value, -393.51)
        self.assertEqual(missing['Hcomb'].source, 'missing')

    def test_formation_hcomb_leaves_metals_unsupported(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'NaCl',
            'phase_at_STP': 'solid',
            'Hf_solid': -411.2,
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            formation = resolver.resolve_formation_properties('madeupium', props, allow_online=False)

        self.assertEqual(formation['Hcomb'].source, 'missing')

    def test_formation_hcomb_calculates_silicon_to_silica(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'SiF4',
            'phase_at_STP': 'gas',
            'Hf': -1614.94,
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            formation = resolver.resolve_formation_properties('madeupium', props, allow_online=False)

        expected = -909.4 - (-1614.94)
        self.assertEqual(formation['Hcomb'].method, 'formula_net_hcomb')
        self.assertClose(formation['Hcomb'].value, expected)

    def test_formation_hcomb_leaves_phosphorus_unsupported(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'PH3',
            'phase_at_STP': 'gas',
            'Hf': 5.4,
        }

        with patch.object(resolver, '_get_perry_formation_properties', return_value=None):
            formation = resolver.resolve_formation_properties('madeupium', props, allow_online=False)

        self.assertEqual(formation['Hcomb'].source, 'missing')

    def test_nist_cp_parser_keeps_liquid_and_gas_tables_separate(self):
        resolver = PropertyResolver()
        html = '''
            <table class="data" aria-label="Constant pressure heat capacity of gas">
                <tr><th>C p,gas (J/mol*K)</th><th>Temperature (K)</th><th>Reference</th><th>Comment</th></tr>
                <tr><td>50.0</td><td>250.0</td><td>ref</td><td></td></tr>
                <tr><td>60.0</td><td>300.0</td><td>ref</td><td></td></tr>
                <tr><td>70.0</td><td>350.0</td><td>ref</td><td></td></tr>
            </table>
            <table class="data" aria-label="Constant pressure heat capacity of liquid">
                <tr><th>C p,liquid (J/mol*K)</th><th>Temperature (K)</th><th>Reference</th><th>Comment</th></tr>
                <tr><td>100.0</td><td>290.0</td><td>ref</td><td></td></tr>
                <tr><td>110.0</td><td>298.15</td><td>ref</td><td></td></tr>
                <tr><td>111.0</td><td>298.15</td><td>ref</td><td></td></tr>
                <tr><td>112.0</td><td>310.0</td><td>ref</td><td></td></tr>
            </table>
            <table class="data" aria-label="Gas Phase Heat Capacity (Shomate Equation)">
                <tr><th>Temperature (K)</th><th>298. to 1000.</th></tr>
                <tr><td>A</td><td>1.0</td></tr>
            </table>
        '''

        parsed = resolver._parse_nist_cp_tables(html)
        gas = resolver._evaluate_nist_cp_tables(parsed, 325.0, 'ideal_gas')
        liquid = resolver._evaluate_nist_cp_tables(parsed, 298.15, 'liquid')

        self.assertEqual(set(parsed), {'gas', 'liquid'})
        self.assertClose(gas.value, 65.0)
        self.assertEqual(gas.phase, 'gas')
        self.assertEqual(gas.method, 'nist_tabulated_gas_cp')
        liquid_kernel = resolver._kernel_from_nist_liquid_source(parsed)
        self.assertClose(liquid.value, liquid_kernel.cp(298.15))
        self.assertEqual(liquid.phase, 'liquid')
        self.assertEqual(liquid.method, 'nist_linear_liquid_cp_kernel')

    def test_nist_small_gas_cp_table_clamps_outside_range(self):
        resolver = PropertyResolver()
        tables = {'gas': [[300.0, 40.0], [350.0, 50.0], [400.0, 60.0]]}

        low = resolver._evaluate_nist_cp_tables(tables, 250.0, 'ideal_gas')
        high = resolver._evaluate_nist_cp_tables(tables, 450.0, 'ideal_gas')

        self.assertClose(low.value, 40.0)
        self.assertIn('clamped below', low.notes)
        self.assertClose(high.value, 60.0)
        self.assertIn('clamped above', high.notes)

    def test_nist_mid_sized_gas_cp_table_uses_linear_fit_with_extrapolation(self):
        resolver = PropertyResolver()
        tables = {
            'gas': [
                [300.0, 130.0],
                [350.0, 140.0],
                [400.0, 150.0],
                [450.0, 160.0],
            ]
        }

        cp = resolver._evaluate_nist_cp_tables(tables, 500.0, 'ideal_gas')

        self.assertEqual(cp.method, 'nist_linear_gas_cp_fit')
        self.assertClose(cp.value, 170.0)
        self.assertIn('extrapolated above', cp.notes)

    def test_nist_large_gas_cp_table_uses_shomate_style_fit(self):
        resolver = PropertyResolver()
        rows = []
        for T in range(300, 1301, 100):
            t = T / 1000.0
            Cp = 25.0 + 40.0 * t + 5.0 * t * t - 1.0 * t * t * t + 0.2 / (t * t)
            rows.append([float(T), Cp])
        tables = {'gas': rows}

        cp = resolver._evaluate_nist_cp_tables(tables, 750.0, 'ideal_gas')
        t = 0.75
        expected = 25.0 + 40.0 * t + 5.0 * t * t - t * t * t + 0.2 / (t * t)

        self.assertEqual(cp.method, 'nist_shomate_gas_cp_fit')
        self.assertClose(cp.value, expected, rel=1e-6)

    def test_nist_cp_integrals_cover_constant_linear_and_shomate_forms(self):
        resolver = PropertyResolver()

        liquid = resolver._integrate_nist_cp_tables(
            {'liquid': [[290.0, 100.0], [300.0, 120.0]]},
            300.0,
            350.0,
            'liquid',
        )
        self.assertEqual(liquid.method, 'nist_linear_liquid_cp_kernel_integral')
        # The normalized liquid kernel follows the two-point slope through its
        # 10 K continuation, then holds the extended endpoint constant.
        self.assertClose(liquid.value, (0.5 * (120.0 + 140.0) * 10.0 + 140.0 * 40.0) / 1000.0)

        linear = resolver._integrate_nist_cp_tables(
            {
                'gas': [
                    [300.0, 130.0],
                    [350.0, 140.0],
                    [400.0, 150.0],
                    [450.0, 160.0],
                ]
            },
            300.0,
            500.0,
            'ideal_gas',
        )
        self.assertEqual(linear.method, 'nist_linear_gas_cp_integral')
        self.assertClose(linear.value, (70.0 * 200.0 + 0.2 * (500.0**2 - 300.0**2) / 2.0) / 1000.0)

        rows = []
        for T in range(300, 1301, 100):
            t = T / 1000.0
            Cp = 25.0 + 40.0 * t + 5.0 * t * t - t * t * t + 0.2 / (t * t)
            rows.append([float(T), Cp])
        shomate = resolver._integrate_nist_cp_tables({'gas': rows}, 400.0, 900.0, 'ideal_gas')
        t1 = 0.4
        t2 = 0.9
        expected = (
            25.0 * (t2 - t1)
            + 40.0 * (t2**2 - t1**2) / 2.0
            + 5.0 * (t2**3 - t1**3) / 3.0
            - (t2**4 - t1**4) / 4.0
            - 0.2 * (1.0 / t2 - 1.0 / t1)
        )
        self.assertEqual(shomate.method, 'nist_shomate_gas_cp_integral')
        self.assertClose(shomate.value, expected, rel=1e-6)

    def test_online_cp_resolution_uses_nist_tabulated_result(self):
        resolver = PropertyResolver()
        source = {'liquid': [[298.15, 123.4]], '_source': 'synthetic NIST table'}

        with (
            patch.object(resolver, '_get_derived_liquid_cp_kernel', return_value=None),
            patch.object(resolver, '_set_derived_liquid_cp_kernel'),
            patch.object(resolver, '_fetch_nist_cp_source', return_value=source),
        ):
            cp = resolver.resolve_heat_capacity('madeupium', 298.15, phase='liquid', props={})

        self.assertClose(cp.value, 123.4)
        self.assertEqual(cp.source, 'NIST Chemistry WebBook')
        self.assertEqual(cp.method, 'nist_constant_liquid_cp_kernel')
        self.assertIn('Single tabulated liquid Cp point', cp.notes)

    def test_liquid_cp_tries_online_before_scaled_gas_polynomial_fallback(self):
        source = {'liquid': [[298.15, 111.1]], '_source': 'synthetic liquid table'}
        props = {'Cp_coeffs': [50.0, 0.0, 0.0, 0.0]}

        resolver = PropertyResolver()
        with (
            patch.object(resolver, '_get_derived_liquid_cp_kernel', return_value=None),
            patch.object(resolver, '_set_derived_liquid_cp_kernel'),
            patch.object(resolver, '_fetch_nist_cp_source', return_value=source),
        ):
                cp = resolver.resolve_heat_capacity(
                    'madeupium', 298.15, phase='liquid', props=props
                )

        self.assertClose(cp.value, 111.1)
        self.assertEqual(cp.source, 'NIST Chemistry WebBook')

        fallback_resolver = PropertyResolver()
        with (
            patch.object(fallback_resolver, '_get_derived_liquid_cp_kernel', return_value=None),
            patch.object(fallback_resolver, '_fetch_nist_cp_source', return_value=None),
        ):
            fallback = fallback_resolver.resolve_heat_capacity(
                'fallbackium', 298.15, phase='liquid', props=props
            )

        self.assertClose(fallback.value, 65.0)
        self.assertEqual(fallback.method, 'scaled_ideal_gas_liquid_cp_kernel')

    def test_nist_phase_change_parser_extracts_scalars_and_hvap_fit(self):
        resolver = PropertyResolver()
        Tc = 514.0
        A = 60.0
        n = 0.38
        standard_hvap = A * (1.0 - 298.15 / Tc) ** n
        hvap_rows = []
        for T in (280.0, 320.0, 360.0, 400.0, 460.0):
            hvap = A * (1.0 - T / Tc) ** n
            hvap_rows.append(
                f'<tr><td>{hvap:.6f}</td><td>{T:.1f}</td><td>N/A</td><td>ref</td><td></td></tr>'
            )
        html = f'''
            <table class="data">
                <tr><th>Quantity</th><th>Value</th><th>Units</th><th>Method</th><th>Reference</th><th>Comment</th></tr>
                <tr><td>T<sub>boil</sub></td><td>351.5 &plusmn; 0.2</td><td>K</td><td>AVG</td><td>N/A</td><td></td></tr>
                <tr><td>T<sub>fus</sub></td><td>159.</td><td>K</td><td>AVG</td><td>N/A</td><td></td></tr>
                <tr><td>T<sub>c</sub></td><td>{Tc}</td><td>K</td><td>AVG</td><td>N/A</td><td></td></tr>
                <tr><td>P<sub>c</sub></td><td>63.0</td><td>bar</td><td>AVG</td><td>N/A</td><td></td></tr>
                <tr><td>V<sub>c</sub></td><td>0.168</td><td>l/mol</td><td>N/A</td><td>ref</td><td></td></tr>
                <tr><td>&Delta;<sub>vap</sub>H&deg;</td><td>{standard_hvap:.6f}</td><td>kJ/mol</td><td>AVG</td><td>N/A</td><td></td></tr>
            </table>
            <table class="data" aria-label="Enthalpy of vaporization">
                <tr><th>&Delta;<sub>vap</sub>H (kJ/mol)</th><th>Temperature (K)</th><th>Method</th><th>Reference</th><th>Comment</th></tr>
                {''.join(hvap_rows)}
            </table>
            <table class="data" aria-label="Enthalpy of fusion">
                <tr><th>&Delta;<sub>fus</sub>H (kJ/mol)</th><th>Temperature (K)</th><th>Reference</th><th>Comment</th></tr>
                <tr><td>4.9</td><td>158.5</td><td>ref</td><td></td></tr>
                <tr><td>5.1</td><td>159.0</td><td>ref</td><td></td></tr>
            </table>
        '''

        parsed = resolver._parse_nist_phase_change(html)

        self.assertClose(parsed['Tb'], 351.5)
        self.assertClose(parsed['Tm'], 159.0)
        self.assertClose(parsed['Tc'], Tc)
        self.assertClose(parsed['Pc'], 63.0)
        self.assertClose(parsed['Vc'], 168.0)
        self.assertClose(parsed['_qualities']['Tc'], 0.96)
        self.assertClose(parsed['_qualities']['Pc'], 0.96)
        self.assertClose(parsed['_qualities']['Vc'], 0.94)
        self.assertClose(parsed['Hfus'], 5.0)
        self.assertEqual(len(parsed['Hfus_records']), 2)
        self.assertEqual(
            [record['temperature_K'] for record in parsed['Hfus_records']],
            [158.5, 159.0],
        )
        self.assertEqual(
            [record['enthalpy_kJ_mol'] for record in parsed['Hfus_records']],
            [4.9, 5.1],
        )
        self.assertNotIn('Hvap', parsed)
        self.assertNotIn('Hvap_fit', parsed)
        self.assertEqual(len(parsed['Hvap_records']), 6)
        standard = next(
            record for record in parsed['Hvap_records']
            if record['basis'] == 'standard_298'
        )
        self.assertClose(standard['T_ref'], 298.15)

        resolver._finalize_online_hvap(parsed)

        self.assertIsInstance(parsed['Hvap_fit'], HvapTemperatureFit)
        self.assertClose(parsed['Hvap_fit'].value_at(351.5), A * (1.0 - 351.5 / Tc) ** n, rel=5e-5)
        self.assertClose(parsed['Hvap'], A * (1.0 - 351.5 / Tc) ** n, rel=5e-5)

    def test_fusion_and_melting_routes_exclude_other_material_forms(self):
        resolver = PropertyResolver()
        props = {
            'name': 'test parent',
            'Tm': 300.0,
            'fusion_transitions': [
                {
                    'enthalpy_kJ_mol': 10.0,
                    'temperature_K': 300.0,
                    'material_form': 'anhydrous',
                    'source': 'provided',
                    'method': 'anhydrous_measurement',
                    'quality': 0.96,
                },
                {
                    'enthalpy_kJ_mol': 20.0,
                    'temperature_K': 350.0,
                    'material_form': 'hydrate',
                    'form_label': 'monohydrate',
                    'source': 'provided',
                    'method': 'hydrate_measurement',
                    'quality': 0.99,
                },
            ],
            'melting_transitions': [
                {
                    'temperature_K': 350.0,
                    'material_form': 'hydrate',
                    'form_label': 'monohydrate',
                    'source': 'provided',
                    'method': 'hydrate_melting',
                    'quality': 0.99,
                },
            ],
        }

        all_fusion = resolver.resolve_fusion_transitions(
            'test parent', props, allow_online=False,
        )
        hydrate_fusion = resolver.resolve_fusion_transitions(
            'test parent', props, allow_online=False, material_form='hydrate',
        )
        all_melting = resolver.resolve_melting_transitions(
            'test parent', props, allow_online=False,
        )
        bare_hfus = resolver.resolve_hfus(
            'test parent', props, allow_online=False,
        )
        bare_tm = resolver.resolve_melting_point(
            'test parent', props, allow_online=False,
        )

        self.assertEqual(len(all_fusion), 2)
        self.assertEqual(len(hydrate_fusion), 1)
        self.assertTrue(any(
            record.material_form == 'hydrate' and record.temperature_K == 350.0
            for record in all_melting
        ))
        self.assertClose(bare_hfus.value, 10.0)
        self.assertClose(bare_tm.value, 300.0)

        hydrate_props = dict(props)
        hydrate_props['name'] = 'test parent monohydrate'
        hydrate_props['Tm'] = 350.0
        hydrate_hfus = resolver.resolve_hfus(
            'test parent monohydrate', hydrate_props, allow_online=False,
        )
        self.assertClose(hydrate_hfus.value, 20.0)

        structured_only = dict(props)
        structured_only.pop('Tm')
        structured_only['melting_transitions'].insert(0, {
            'temperature_K': 300.0,
            'material_form': 'anhydrous',
            'source': 'provided',
            'method': 'anhydrous_melting',
            'quality': 0.96,
        })
        selected_tm = resolver.resolve_melting_point(
            'test parent', structured_only, allow_online=False,
        )
        self.assertClose(selected_tm.value, 300.0)
        self.assertEqual(selected_tm.method, 'anhydrous_melting')

    def test_online_melting_consensus_keeps_hydrates_out_of_bare_tm(self):
        resolver = PropertyResolver()
        payload = {
            '_phase_candidates': {
                'Tm': [
                    {
                        'value_K': 300.0,
                        'low_K': 300.0,
                        'high_K': 300.0,
                        'uncertainty_K': None,
                        'source': 'pubchem',
                        'method': 'reported_anhydrous',
                        'reference': 'a',
                        'comment': 'anhydrous',
                        'raw': '300 K',
                        'qualifiers': {'anhydrous': True},
                        'sample_count': None,
                        'material_form': 'anhydrous',
                        'is_identity_form': False,
                    },
                    {
                        'value_K': 350.0,
                        'low_K': 350.0,
                        'high_K': 350.0,
                        'uncertainty_K': None,
                        'source': 'pubchem',
                        'method': 'reported_hydrate',
                        'reference': 'h',
                        'comment': 'monohydrate',
                        'raw': '350 K',
                        'qualifiers': {'hydrate': True},
                        'sample_count': None,
                        'material_form': 'hydrate',
                        'form_label': 'hydrate',
                        'is_identity_form': False,
                    },
                ],
            },
        }
        resolver._finalize_online_phase_point_candidates(payload)
        self.assertClose(payload['Tm'], 300.0)
        self.assertEqual(len(payload['_phase_candidates']['Tm']), 2)

    def test_unlabeled_tm_keeps_near_form_corroboration_not_distant_polymorph(self):
        resolver = PropertyResolver()

        def candidate(value, polymorph=''):
            return {
                'value_K': value,
                'low_K': value,
                'high_K': value,
                'uncertainty_K': None,
                'source': 'pubchem',
                'method': 'reported_melting',
                'reference': str(value),
                'comment': polymorph,
                'raw': f'{value:g} K {polymorph}',
                'qualifiers': {},
                'sample_count': None,
                'material_form': 'unspecified',
                'form_label': polymorph,
                'polymorph': polymorph,
                'is_identity_form': False,
            }

        payload = {'_phase_candidates': {'Tm': [
            candidate(300.0),
            candidate(300.5, 'alpha'),
            candidate(350.0, 'beta'),
        ]}}
        resolver._finalize_online_phase_point_candidates(payload)
        self.assertClose(payload['Tm'], 300.25)
        self.assertEqual(
            payload['_sources']['Tm'],
            'pubchem_melting_point_consensus',
        )
        selected = payload['_selected_phase_candidates']['Tm'][
            'selected_candidates'
        ]
        self.assertEqual({item['value_K'] for item in selected}, {300.0, 300.5})

    def test_pubchem_fusion_records_preserve_hydrate_and_anhydrous_forms(self):
        resolver = PropertyResolver()
        node = {
            'TOCHeading': 'Heat of Fusion',
            'Information': [
                {
                    'ReferenceNumber': 1,
                    'Value': {'StringWithMarkup': [{
                        'String': '10.0 kJ/mol at 300 K, anhydrous',
                    }]},
                },
                {
                    'ReferenceNumber': 2,
                    'Description': 'monohydrate measurement',
                    'Value': {'StringWithMarkup': [{
                        'String': '20.0 kJ/mol at 350 K',
                    }]},
                },
            ],
        }
        parent = {}
        resolver._extract_phase_change_from_pubchem_node(
            node, parent, identity_text='test parent',
        )
        resolver._finalize_online_fusion_records(parent)
        self.assertEqual(len(parent['Hfus_records']), 2)
        self.assertClose(parent['Hfus'], 10.0)
        self.assertEqual(
            {record['material_form'] for record in parent['Hfus_records']},
            {'anhydrous', 'hydrate'},
        )

        hydrate = {}
        resolver._extract_phase_change_from_pubchem_node(
            node, hydrate, identity_text='test parent monohydrate',
        )
        resolver._finalize_online_fusion_records(hydrate)
        self.assertClose(hydrate['Hfus'], 20.0)
        selected = next(
            record for record in hydrate['Hfus_records']
            if record['material_form'] == 'hydrate'
        )
        self.assertTrue(selected['is_identity_form'])

    def test_multiple_polymorphs_require_a_bare_tm_or_default_form(self):
        resolver = PropertyResolver()
        records = tuple(
            FusionTransitionRecord(
                enthalpy_kJ_mol=value,
                temperature_K=temperature,
                polymorph=form,
                form_label=form,
                source='online',
                method='reported_fusion',
                quality=0.95,
            )
            for form, temperature, value in (
                ('alpha', 300.0, 10.0),
                ('beta', 350.0, 20.0),
            )
        )
        self.assertEqual(
            resolver._select_bare_fusion_records(records, None),
            (),
        )
        selected = resolver._select_bare_fusion_records(records, 350.0)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].polymorph, 'beta')

    def test_resolver_owned_flattened_hfus_does_not_bypass_transition_records(self):
        resolver = PropertyResolver()
        props = {
            'Tm': 300.0,
            'Hfus': 99.0,
            'property_sources': {
                'Hfus': {
                    'source': 'online',
                    'method': 'N/A',
                    'quality': 0.90,
                    'notes': 'legacy flattened online scalar',
                },
            },
            'fusion_transitions': [{
                'enthalpy_kJ_mol': 10.0,
                'temperature_K': 300.0,
                'material_form': 'anhydrous',
                'source': 'nist_phase_change',
                'method': 'N/A',
                'quality': 0.94,
            }],
        }
        result = resolver.resolve_hfus(
            'flattened fixture', props, allow_online=False,
        )
        self.assertClose(result.value, 10.0)
        self.assertEqual(result.source, 'nist_phase_change')

    def test_nist_critical_parser_prefers_avg_and_rejects_conflicting_rows(self):
        resolver = PropertyResolver()
        preferred_html = '''
            <table class="data">
                <tr><th>Quantity</th><th>Value</th><th>Units</th><th>Method</th><th>Reference</th></tr>
                <tr><td>T<sub>c</sub></td><td>500</td><td>K</td><td>EXP</td><td>older</td></tr>
                <tr><td>T<sub>c</sub></td><td>510</td><td>K</td><td>AVG</td><td>recommended</td></tr>
                <tr><td>P<sub>c</sub></td><td>40</td><td>bar</td><td>EXP</td><td>one</td></tr>
                <tr><td>P<sub>c</sub></td><td>41</td><td>bar</td><td>EXP</td><td>two</td></tr>
            </table>
        '''
        conflicting_html = '''
            <table class="data">
                <tr><th>Quantity</th><th>Value</th><th>Units</th><th>Method</th><th>Reference</th></tr>
                <tr><td>T<sub>c</sub></td><td>500</td><td>K</td><td>EXP</td><td>one</td></tr>
                <tr><td>T<sub>c</sub></td><td>550</td><td>K</td><td>EXP</td><td>two</td></tr>
            </table>
        '''

        preferred = resolver._parse_nist_phase_change(preferred_html)
        conflicting = resolver._parse_nist_phase_change(conflicting_html)

        self.assertClose(preferred['Tc'], 510.0)
        self.assertClose(preferred['_qualities']['Tc'], 0.96)
        self.assertClose(preferred['Pc'], 40.5)
        self.assertClose(preferred['_qualities']['Pc'], 0.95)
        self.assertNotIn('Tc', conflicting)

    def test_pubchem_critical_parser_requires_reported_units_and_consensus(self):
        resolver = PropertyResolver()
        root = {
            'Section': [
                {
                    'TOCHeading': 'Critical Temperature',
                    'Information': [
                        {
                            'Name': 'QSPR predicted value',
                            'Value': {'StringWithMarkup': [{'String': '600 K'}]},
                        },
                        {
                            'Reference': 'reported handbook value',
                            'Value': {'StringWithMarkup': [{'String': '500 K'}]},
                        },
                    ],
                },
                {
                    'TOCHeading': 'Critical Pressure',
                    'Information': [
                        {'Value': {'StringWithMarkup': [{'String': '50 bar'}]}},
                        {'Value': {'StringWithMarkup': [{'String': '5.05 MPa'}]}},
                    ],
                },
                {
                    'TOCHeading': 'Critical Volume',
                    'Information': [{'Value': {'Number': [200]}}],
                },
            ],
        }
        result = {}

        resolver._extract_phase_change_from_pubchem_node(root, result)
        resolver._finalize_pubchem_critical_candidates(result)

        self.assertClose(result['Tc'], 500.0)
        self.assertClose(result['_qualities']['Tc'], 0.88)
        self.assertClose(result['Pc'], 50.25)
        self.assertClose(result['_qualities']['Pc'], 0.89)
        self.assertNotIn('Vc', result)

    def test_pubchem_critical_parser_rejects_specific_gravity_temperature_below_tb(self):
        resolver = PropertyResolver()
        root = {
            'Section': [
                {
                    'TOCHeading': 'Boiling Point',
                    'Information': [{
                        'Value': {'StringWithMarkup': [{'String': '109.9 °C'}]},
                    }],
                },
                {
                    'TOCHeading': 'Critical Temperature and Pressure',
                    'Information': [{
                        'Description': 'PEER REVIEWED',
                        'Value': {'StringWithMarkup': [{
                            'String': '0.955-0.959 (15/4 °C)',
                        }]},
                    }],
                },
            ],
        }
        result = {}

        resolver._extract_phase_change_from_pubchem_node(root, result)
        resolver._finalize_pubchem_critical_candidates(result)

        self.assertClose(result['Tb'], 383.05)
        self.assertNotIn('Tc', result)
        self.assertTrue(any(
            'Tc=277.15 K does not exceed reported Tb=383.05 K' in rejection
            for rejection in result['_rejections']
        ))

        valid = {'Tb': 383.05, '_critical_candidates': {'Tc': [574.6]}}
        resolver._finalize_pubchem_critical_candidates(valid)
        self.assertClose(valid['Tc'], 574.6)
        self.assertClose(valid['_qualities']['Tc'], 0.88)

    def test_pubchem_critical_parser_accepts_written_temperature_units(self):
        resolver = PropertyResolver()
        expected = (
            ('751 deg K', 751.0),
            ('751 degrees Kelvin', 751.0),
            ('25 deg C', 298.15),
            ('25 degrees Celsius', 298.15),
            ('77 deg F', 298.15),
            ('77 degrees Fahrenheit', 298.15),
        )
        for text, temperature in expected:
            with self.subTest(text):
                self.assertClose(resolver._parse_temperature_K(text), temperature)

        result = {}
        resolver._extract_phase_change_from_pubchem_node(
            {
                'TOCHeading': 'Critical Temperature and Pressure',
                'Information': [{
                    'Value': {'StringWithMarkup': [{
                        'String': (
                            'Critical temperature = 751 deg K; '
                            'Critical pressure = 398X10+6 Pa.'
                        ),
                    }]},
                }],
            },
            result,
        )
        resolver._finalize_pubchem_critical_candidates(result)
        self.assertClose(result['Tc'], 751.0)
        self.assertClose(result['Pc'], 3980.0)

        guarded = {}
        resolver._extract_phase_change_from_pubchem_node(
            {
                'Section': [
                    {
                        'TOCHeading': 'Molecular Formula',
                        'Information': [{
                            'Value': {'StringWithMarkup': [{'String': 'C6H4ClNO2'}]},
                        }],
                    },
                    {
                        'TOCHeading': 'Critical Temperature and Pressure',
                        'Information': [{
                            'Value': {'StringWithMarkup': [{
                                'String': (
                                    'Critical temperature = 751 degrees Kelvin; '
                                    'Critical pressure = 398X10+6 Pa.'
                                ),
                            }]},
                        }],
                    },
                ],
            },
            guarded,
        )
        resolver._finalize_pubchem_critical_candidates(guarded)
        self.assertClose(guarded['Tc'], 751.0)
        self.assertNotIn('Pc', guarded)
        self.assertTrue(any('500' not in warning and 'organic' in warning for warning in guarded['_rejections']))

    def test_pubchem_critical_parser_rejects_conflicting_reported_values(self):
        resolver = PropertyResolver()
        root = {
            'TOCHeading': 'Critical Pressure',
            'Information': [
                {'Value': {'StringWithMarkup': [{'String': '40 bar'}]}},
                {'Value': {'StringWithMarkup': [{'String': '60 bar'}]}},
            ],
        }
        result = {}

        resolver._extract_phase_change_from_pubchem_node(root, result)
        resolver._finalize_pubchem_critical_candidates(result)

        self.assertNotIn('Pc', result)

    def test_pubchem_hvap_records_preserve_temperature_and_reject_predictions(self):
        resolver = PropertyResolver()
        root = {
            'TOCHeading': 'Heat of Vaporization',
            'Information': [
                {
                    'Description': 'PEER REVIEWED',
                    'Reference': ['handbook'],
                    'Value': {'StringWithMarkup': [{'String': '42.32 kJ/mol at 25 °C'}]},
                },
                {
                    'Description': 'estimated model value',
                    'Value': {'StringWithMarkup': [{'String': '99 kJ/mol at 25 °C'}]},
                },
                {
                    'Description': 'calculated from experimental vapor-pressure data',
                    'Value': {'StringWithMarkup': [{'String': '41.8 kJ/mol at 30 °C'}]},
                },
                {
                    'Description': 'PEER REVIEWED',
                    'Reference': ['constant-pressure source'],
                    'Value': {'StringWithMarkup': [{'String': '13.3 kcal/mole @ constant pressure'}]},
                },
            ],
        }
        result = {}

        resolver._extract_phase_change_from_pubchem_node(root, result)

        self.assertEqual(len(result['Hvap_records']), 3)
        explicit = result['Hvap_records'][0]
        derived_experimental = result['Hvap_records'][1]
        unknown = result['Hvap_records'][2]
        self.assertClose(explicit['value'], 42.32)
        self.assertClose(explicit['T_ref'], 298.15)
        self.assertEqual(explicit['basis'], 'explicit_temperature')
        self.assertClose(derived_experimental['value'], 41.8)
        self.assertClose(derived_experimental['T_ref'], 303.15)
        self.assertClose(unknown['value'], 13.3 * 4.184)
        self.assertIsNone(unknown['T_ref'])
        self.assertEqual(unknown['basis'], 'unspecified')

        result.update({'Tb': 351.5, 'Tc': 514.0})
        resolver._finalize_online_hvap(result)

        expected = resolver._watson_hvap_value(41.8, 303.15, 351.5, 514.0)
        self.assertClose(result['Hvap'], expected)
        self.assertEqual(result['_sources']['Hvap'], 'pubchem_hvap_at_tb')

    def test_nist_standard_hvap_is_converted_from_298_not_assumed_at_tb(self):
        resolver = PropertyResolver()
        html = '''
            <table class="data">
                <tr><th>Quantity</th><th>Value</th><th>Units</th><th>Method</th><th>Reference</th></tr>
                <tr><td>T<sub>boil</sub></td><td>487.0</td><td>K</td><td>AVG</td><td>N/A</td></tr>
                <tr><td>T<sub>c</sub></td><td>729.0</td><td>K</td><td>AVG</td><td>N/A</td></tr>
                <tr><td>&Delta;<sub>vap</sub>H&deg;</td><td>62.0</td><td>kJ/mol</td><td>EXP</td><td>one</td></tr>
                <tr><td>&Delta;<sub>vap</sub>H&deg;</td><td>63.5</td><td>kJ/mol</td><td>AVG</td><td>recommended</td></tr>
                <tr><td>&Delta;<sub>vap</sub>H&deg;</td><td>65.0</td><td>kJ/mol</td><td>EXP</td><td>two</td></tr>
            </table>
        '''
        parsed = resolver._parse_nist_phase_change(html)
        resolver._finalize_online_hvap(parsed)

        expected_tb = resolver._watson_hvap_value(63.5, 298.15, 487.0, 729.0)
        self.assertClose(parsed['Hvap'], expected_tb)
        self.assertLess(parsed['Hvap'], 63.5)
        self.assertIn('Watson-scaled from 298.15 K', parsed['_notes']['Hvap'])

        with patch.object(resolver, '_fetch_phase_change_pubchem', return_value=None), \
             patch.object(resolver, '_fetch_phase_change_nist', return_value=parsed):
            at_standard = resolver.resolve_hvap(
                'madeupium',
                {},
                T=298.15,
                allow_online=True,
                allow_estimation=False,
            )

        self.assertClose(at_standard.value, 63.5)

    def test_online_hvap_ranking_prefers_near_tb_pubchem_over_distant_nist(self):
        resolver = PropertyResolver()
        result = {
            'Tb': 487.0,
            'Tc': 729.0,
            'Hvap_records': [
                {
                    'value': 45.86,
                    'T_ref': 487.15,
                    'basis': 'explicit_temperature',
                    'source': 'pubchem',
                    'method': 'PEER REVIEWED',
                    'quality': 0.88,
                    'reference': 'near-Tb source',
                },
                {
                    'value': 53.1,
                    'T_ref': 332.0,
                    'basis': 'saturation',
                    'source': 'nist_phase_change',
                    'method': 'N/A',
                    'quality': 0.94,
                    'reference': 'distant source',
                },
            ],
        }

        resolver._finalize_online_hvap(result)

        expected = resolver._watson_hvap_value(45.86, 487.15, 487.0, 729.0)
        self.assertClose(result['Hvap'], expected)
        self.assertEqual(result['_sources']['Hvap'], 'pubchem_hvap_at_tb')

    def test_nist_hvap_fit_can_use_merged_pubchem_critical_temperature(self):
        resolver = PropertyResolver()
        pubchem = {
            'Tb': 350.0,
            'Tc': 520.0,
            '_sources': {'Tb': 'pubchem', 'Tc': 'pubchem'},
        }
        nist_records = [
            {
                'value': 62.0 * (1.0 - T / 520.0) ** 0.39,
                'T_ref': T,
                'basis': 'saturation',
                'source': 'nist_phase_change',
                'method': 'N/A',
                'quality': 0.94,
                'reference': 'test',
            }
            for T in (280.0, 310.0, 340.0, 380.0, 430.0)
        ]
        nist = {'Hvap_records': nist_records, '_sources': {}}

        with patch.object(resolver, '_fetch_phase_change_pubchem', return_value=pubchem), \
             patch.object(resolver, '_fetch_phase_change_nist', return_value=nist):
            merged = resolver._fetch_phase_change_online('madeupium', {})

        self.assertIsInstance(merged['Hvap_fit'], HvapTemperatureFit)
        self.assertClose(merged['Hvap_fit'].Tc, 520.0)
        self.assertEqual(merged['_sources']['Hvap_fit'], 'nist_phase_change')

    def test_legacy_hvap_cache_is_not_used_as_hvap_at_tb(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.CACHE_DIR = Path(directory)
            resolver._set_cache('hvap_isoamyl alcohol', {'hvap': 55.6472, 'source': 'pubchem'})

            with patch.object(resolver, '_fetch_phase_change_online', return_value=None):
                value = resolver.get_hvap_online('isoamyl alcohol')

        self.assertIsNone(value)

    def test_online_phase_change_prefers_nist_criticals_over_pubchem(self):
        resolver = PropertyResolver()
        pubchem = {
            'Tc': 500.0,
            'Pc': 40.0,
            'Tb': 350.0,
            '_sources': {'Tc': 'pubchem', 'Pc': 'pubchem', 'Tb': 'pubchem'},
            '_qualities': {'Tc': 0.88, 'Pc': 0.88},
        }
        nist = {
            'Tc': 510.0,
            'Pc': 42.0,
            'Tb': 351.0,
            '_sources': {
                'Tc': 'nist_phase_change',
                'Pc': 'nist_phase_change',
                'Tb': 'nist_phase_change',
            },
            '_qualities': {'Tc': 0.96, 'Pc': 0.95},
        }

        with patch.object(resolver, '_fetch_phase_change_pubchem', return_value=pubchem), \
             patch.object(resolver, '_fetch_phase_change_nist', return_value=nist):
            merged = resolver._fetch_phase_change_online('madeupium', {})

        self.assertClose(merged['Tc'], 510.0)
        self.assertClose(merged['Pc'], 42.0)
        self.assertClose(merged['Tb'], 350.0)
        self.assertEqual(merged['_sources']['Tc'], 'nist_phase_change')
        self.assertClose(merged['_qualities']['Tc'], 0.96)

    def test_online_phase_change_cache_roundtrip_preserves_metadata_and_hvap_records(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            writer = PropertyResolver()
            writer.CACHE_DIR = cache_dir
            pubchem = {
                'Tc': 500.0,
                '_sources': {'Tc': 'pubchem'},
                '_qualities': {'Tc': 0.88},
                '_notes': {'Tc': 'strict PubChem value'},
            }
            hvap_records = [{
                'value': 42.3,
                'T_ref': 298.15,
                'basis': 'standard_298',
                'source': 'nist_phase_change',
                'method': 'AVG',
                'quality': 0.93,
                'reference': 'NIST test',
                'comment': '',
                'raw': '',
            }]
            nist = {
                'Tc': 514.0,
                'Hvap_records': hvap_records,
                '_sources': {
                    'Tc': 'nist_phase_change',
                },
                '_qualities': {'Tc': 0.96},
                '_notes': {'Tc': 'NIST AVG row'},
            }

            version = writer.ONLINE_PHASE_CHANGE_CACHE_VERSION
            writer._set_cache(f'phase_pubchem_v{version}_A/B', pubchem)
            writer._set_cache(f'phase_nist_v{version}_NIST test', nist)

            reader = PropertyResolver()
            reader.CACHE_DIR = cache_dir
            loaded_pubchem = reader._fetch_phase_change_pubchem('A/B')
            loaded_nist = reader._fetch_phase_change_nist('NIST test')

            self.assertEqual(loaded_pubchem, pubchem)
            self.assertEqual(loaded_nist['Hvap_records'], hvap_records)
            self.assertClose(loaded_nist['_qualities']['Tc'], 0.96)
            self.assertEqual(loaded_nist['_notes']['Tc'], 'NIST AVG row')
            cache_path = cache_dir / 'property_cache.sqlite'
            self.assertTrue(cache_path.is_file())
            with sqlite3.connect(cache_path) as connection:
                rows = connection.execute(
                    """
                    SELECT cache_key, provider, contract_version
                    FROM runtime_json_cache
                    WHERE namespace = 'property_resolver'
                    ORDER BY cache_key
                    """
                ).fetchall()
            self.assertEqual(rows, [
                (f'phase_nist_v{version}_NIST test', 'nist', version),
                (f'phase_pubchem_v{version}_A/B', 'pubchem', version),
            ])

    def test_online_phase_change_priority_and_hvap_fit_resolution(self):
        resolver = PropertyResolver()
        pubchem = {
            'Tb': 300.0,
            'Tc': 514.0,
            'Hvap_records': [{
                'value': 99.0,
                'T_ref': None,
                'basis': 'unspecified',
                'source': 'pubchem',
                'method': 'reported',
                'quality': 0.76,
                'reference': '',
                'comment': '',
                'raw': '99 kJ/mol',
            }],
            '_sources': {'Tb': 'pubchem', 'Tc': 'pubchem'},
        }
        nist_records = []
        for T in (280.0, 320.0, 360.0, 400.0, 460.0):
            nist_records.append({
                'value': 60.0 * (1.0 - T / 514.0) ** 0.38,
                'T_ref': T,
                'basis': 'saturation',
                'source': 'nist_phase_change',
                'method': 'N/A',
                'quality': 0.94,
                'reference': 'test',
                'comment': '',
                'raw': '',
            })
        nist = {
            'Tb': 310.0,
            'Tm': 150.0,
            'Hfus': 5.0,
            'Hvap_records': nist_records,
            '_sources': {
                'Tb': 'nist_phase_change',
                'Tm': 'nist_phase_change',
                'Hfus': 'nist_phase_change',
            },
        }

        with patch.object(resolver, '_fetch_phase_change_pubchem', return_value=pubchem), \
             patch.object(resolver, '_fetch_phase_change_nist', return_value=nist):
            tb = resolver.resolve_boiling_point('madeupium', {}, allow_online=True)
            tm = resolver.resolve_melting_point('madeupium', {}, allow_online=True)
            hvap = resolver.resolve_hvap(
                'madeupium',
                {'Tb': 300.0},
                T=350.0,
                allow_online=True,
                allow_estimation=False,
            )
            hfus = resolver.resolve_hfus('madeupium', {}, allow_online=True)

        self.assertEqual(tb.method, 'pubchem')
        self.assertClose(tb.value, 300.0)
        self.assertEqual(tm.method, 'nist_melting_point_consensus')
        self.assertClose(tm.value, 150.0)
        self.assertEqual(hvap.method, 'nist_hvap_watson_fit')
        self.assertClose(hvap.value, 60.0 * (1.0 - 350.0 / 514.0) ** 0.38, rel=1e-5)
        self.assertEqual(hfus.method, 'nist_phase_change')
        self.assertClose(hfus.value, 5.0)

    def test_nist_hvap_fit_beats_scalar_hvap_when_no_local_fit_exists(self):
        resolver = PropertyResolver()
        fit = HvapTemperatureFit(
            A=55.0,
            n=0.4,
            Tc=600.0,
            T_min=280.0,
            T_max=420.0,
            mape_percent=1.0,
            kept_points=6,
            total_points=6,
            source='NIST WebBook',
        )
        nist_records = []
        for T in (280.0, 310.0, 340.0, 390.0, 420.0):
            nist_records.append({
                'value': fit.value_at(T),
                'T_ref': T,
                'basis': 'saturation',
                'source': 'nist_phase_change',
                'method': 'N/A',
                'quality': 0.94,
                'reference': 'test',
                'comment': '',
                'raw': '',
            })
        nist = {
            'Tb': 360.0,
            'Tc': 600.0,
            'Hvap_records': nist_records,
            '_sources': {
                'Tb': 'nist_phase_change',
                'Tc': 'nist_phase_change',
            },
        }

        with patch.object(resolver, '_fetch_phase_change_pubchem', return_value=None), \
             patch.object(resolver, '_fetch_phase_change_nist', return_value=nist):
            hvap = resolver.resolve_hvap(
                'madeupium',
                {'Tb': 360.0, 'Hvap': 99.0},
                allow_online=True,
                allow_estimation=False,
            )

        self.assertEqual(hvap.method, 'nist_hvap_watson_fit')
        self.assertClose(hvap.value, fit.value_at(360.0))

    def test_critical_online_lookup_preserves_per_property_sources_and_quality(self):
        resolver = PropertyResolver()

        with patch.object(
                 resolver,
                 '_fetch_phase_change_online',
                 return_value={
                     'Tc': 500.0,
                     'Pc': 40.0,
                     'Vc': 200.0,
                     '_sources': {
                         'Tc': 'pubchem',
                         'Pc': 'nist_phase_change',
                         'Vc': 'nist_phase_change',
                     },
                     '_qualities': {'Tc': 0.88, 'Pc': 0.96, 'Vc': 0.95},
                 },
             ):
            critical = resolver.resolve_critical_properties(
                'madeupium',
                {},
                allow_online=True,
                allow_estimation=False,
            )

        self.assertEqual(critical['Tc'].method, 'pubchem')
        self.assertClose(critical['Tc'].value, 500.0)
        self.assertClose(critical['Tc'].quality, 0.88)
        self.assertEqual(critical['Pc'].method, 'nist_phase_change')
        self.assertClose(critical['Pc'].value, 40.0)
        self.assertClose(critical['Pc'].quality, 0.96)
        self.assertEqual(critical['Vc'].method, 'nist_phase_change')
        self.assertClose(critical['Vc'].value, 200.0)
        self.assertClose(critical['Vc'].quality, 0.95)

    def test_critical_pressure_caps_and_zc_range_reject_impossible_values(self):
        resolver = PropertyResolver()

        self.assertTrue(resolver._critical_pressure_is_admissible(
            {'formula': 'C6H4ClNO2'}, 500.0,
        ))
        self.assertFalse(resolver._critical_pressure_is_admissible(
            {'formula': 'C6H4ClNO2'}, 500.0001,
        ))
        self.assertTrue(resolver._critical_formula_is_organic(
            {'formula': 'CCl4', 'smiles': 'ClC(Cl)(Cl)Cl'},
        ))
        self.assertFalse(resolver._critical_formula_is_organic(
            {'formula': 'CSi', 'smiles': '[C-]#[Si+]'},
        ))
        self.assertFalse(resolver._critical_formula_is_organic(
            {'formula': 'CO2', 'smiles': 'O=C=O'},
        ))
        self.assertTrue(resolver._critical_formula_is_metal_or_salt(
            {'formula': 'MoCl5'},
        ))
        self.assertTrue(resolver._critical_formula_is_metal_or_salt(
            {'formula': 'NH4Cl'},
        ))
        self.assertFalse(resolver._critical_formula_is_metal_or_salt(
            {'formula': 'S4'},
        ))
        self.assertFalse(resolver._critical_formula_is_metal_or_salt(
            {'formula': 'C32H66'},
        ))
        self.assertFalse(resolver._critical_formula_is_metal_or_salt(
            {
                'formula': 'C10H10Fe',
                'smiles': '[Fe].c1cccc1.c1cccc1',
            },
        ))
        self.assertTrue(resolver._critical_formula_is_metal_or_salt(
            {'CAS': '7439-98-7', 'name': 'Molybdenum'},
        ))
        self.assertTrue(resolver._critical_formula_is_metal_or_salt(
            {'CAS': '12125-02-9', 'name': 'Ammonium chloride'},
        ))
        self.assertFalse(resolver._critical_formula_is_metal_or_salt(
            {'CAS': '544-85-4', 'name': 'Dotriacontane'},
        ))
        self.assertTrue(resolver._critical_pressure_is_admissible(
            {'formula': 'Hg'}, 2000.0,
        ))
        self.assertFalse(resolver._critical_pressure_is_admissible(
            {'formula': 'Hg'}, 2000.0001,
        ))
        for value, admitted in ((0.1, True), (0.6, True), (0.0999, False), (0.6001, False)):
            with self.subTest(Zc=value):
                self.assertEqual(
                    resolver._critical_compressibility_is_admissible(value),
                    admitted,
                )

        with patch.object(
            resolver,
            '_fetch_phase_change_online',
            return_value={
                'Tc': 751.0,
                'Pc': 3980.0,
                '_sources': {'Tc': 'pubchem', 'Pc': 'pubchem'},
                '_qualities': {'Tc': 0.88, 'Pc': 0.88},
            },
        ):
            critical = resolver.resolve_critical_properties(
                '100-00-5',
                {
                    'CAS': '100-00-5',
                    'formula': 'C6H4ClNO2',
                    'smiles': 'O=[N+]([O-])c1ccc(Cl)cc1',
                    'Tb': 515.3722222222223,
                    'property_sources': {
                        'Tb': {'source': 'online', 'method': 'pubchem', 'quality': 0.95},
                        'smiles': {'source': 'online', 'method': 'pubchem', 'quality': 0.97},
                    },
                },
                allow_online=True,
                allow_estimation=True,
            )

        self.assertEqual(critical['Tc'].method, 'pubchem')
        self.assertClose(critical['Tc'].value, 751.0)
        self.assertNotEqual(critical['Pc'].method, 'pubchem')
        self.assertLessEqual(critical['Pc'].value, 500.0)
        if critical['Zc'].value is not None:
            self.assertGreaterEqual(critical['Zc'].value, 0.1)
            self.assertLessEqual(critical['Zc'].value, 0.6)

        effective = resolver.resolve_critical_properties(
            'metal pseudo-critical test',
            {
                'formula': 'MoCl5',
                'Tc': 848.0,
                'Pc': 3000.0,
                'Vc': 371.0,
                'Zc': 0.05,
                'omega': 0.2,
                'property_sources': {
                    key: {
                        'source': 'eos effective',
                        'method': 'effective_critical',
                        'quality': 0.8,
                    }
                    for key in ('Tc', 'Pc', 'Vc', 'Zc', 'omega')
                },
            },
            allow_online=False,
            allow_estimation=False,
        )
        self.assertClose(effective['Pc'].value, 3000.0)
        self.assertClose(effective['Zc'].value, 0.05)

    def test_zc_identity_does_not_backcalculate_soft_pc(self):
        resolver = PropertyResolver()

        critical = resolver.resolve_critical_properties(
            'madeupium',
            {
                'Tc': 605.4,
                'Pc': 91.30154687229381,
                'Vc': 434.0,
                'Tb': 426.15,
                'property_sources': {
                    'Tc': {'source': 'online', 'method': 'nist_phase_change', 'quality': 0.98},
                    'Pc': {'source': 'estimated', 'method': 'lydersen_style', 'quality': 0.60},
                    'Vc': {'source': 'online', 'method': 'nist_phase_change', 'quality': 0.96},
                    'Tb': {'source': 'textbook', 'method': 'test_tb', 'quality': 0.95},
                    'Zc': {'source': 'calculated', 'method': 'critical_volume_identity'},
                    'omega': {'source': 'calculated', 'method': 'lee_kesler'},
                },
            },
            allow_online=False,
            allow_estimation=True,
        )

        self.assertEqual(critical['Pc'].method, 'lydersen_style')
        self.assertClose(critical['Pc'].value, 91.30154687229381)
        self.assertClose(critical['Pc'].quality, 0.60)
        self.assertIsNone(critical['Zc'].value)
        self.assertEqual(critical['Zc'].method, 'none')
        self.assertEqual(critical['omega'].method, 'lee_kesler')
        self.assertClose(critical['omega'].quality, 0.60 * 0.80)

    def test_soft_zc_does_not_backcalculate_weaker_pc(self):
        resolver = PropertyResolver()

        critical = resolver.resolve_critical_properties(
            'madeupium',
            {
                'Tc': 600.0,
                'Pc': 80.0,
                'Vc': 250.0,
                'Zc': 0.25,
                'property_sources': {
                    'Tc': {'source': 'local', 'method': 'test_tc', 'quality': 0.96},
                    'Pc': {'source': 'estimated', 'method': 'lydersen_style', 'quality': 0.60},
                    'Vc': {'source': 'local', 'method': 'test_vc', 'quality': 0.95},
                    'Zc': {'source': 'estimated', 'method': 'test_zc_estimate', 'quality': 0.86},
                },
            },
            allow_online=False,
            allow_estimation=True,
        )

        self.assertEqual(critical['Pc'].method, 'lydersen_style')
        self.assertClose(critical['Pc'].value, 80.0)
        self.assertEqual(critical['Zc'].method, 'test_zc_estimate')
        self.assertClose(critical['Zc'].value, 0.25)

    def test_nannoolal_vc_fallback_enables_zc_identity_from_tc_pc(self):
        resolver = PropertyResolver()

        critical = resolver.resolve_critical_properties(
            'madeupium',
            {
                'formula': 'C6H6',
                'smiles': 'c1ccccc1',
                'Tc': 562.2,
                'Pc': 48.9,
                'property_sources': {
                    'Tc': {'source': 'local', 'method': 'test_tc', 'quality': 0.98},
                    'Pc': {'source': 'local', 'method': 'test_pc', 'quality': 0.97},
                    'formula': {'source': 'provided', 'method': 'direct', 'quality': 1.0},
                    'smiles': {'source': 'provided', 'method': 'provided_smiles', 'quality': 1.0},
                },
            },
            allow_online=False,
            allow_estimation=True,
        )

        expected_vc = 260.86754483941434
        expected_zc = 48.9 * 100000.0 * expected_vc * 1.0e-6 / (8.314462618 * 562.2)
        self.assertEqual(critical['Vc'].method, 'nannoolal_vc')
        self.assertClose(critical['Vc'].value, expected_vc)
        self.assertClose(critical['Vc'].quality, 0.85)
        self.assertEqual(critical['Zc'].method, 'critical_volume_identity')
        self.assertClose(critical['Zc'].value, expected_zc)
        self.assertClose(critical['Zc'].quality, 0.85)

    def test_formula_vc_quality_penalizes_missing_ring_structure(self):
        resolver = PropertyResolver()

        critical = resolver.resolve_critical_properties(
            'madeupium',
            {
                'formula': 'C3H6O',
                'property_sources': {
                    'formula': {'source': 'provided', 'method': 'direct', 'quality': 1.0},
                },
            },
            allow_online=False,
            allow_estimation=True,
        )

        self.assertEqual(critical['Vc'].method, 'atom_count_large_ring_vc')
        self.assertClose(critical['Vc'].quality, 0.75)
        self.assertIn('assumed acyclic', critical['Vc'].notes)

    def test_exact_critical_identity_preserves_minimum_real_input_quality(self):
        resolver = PropertyResolver()

        critical = resolver.resolve_critical_properties(
            'madeupium',
            {
                'Tc': 500.0,
                'Pc': 50.0,
                'Vc': 200.0,
                'property_sources': {
                    'Tc': {'source': 'local', 'method': 'test', 'quality': 0.98},
                    'Pc': {'source': 'textbook', 'method': 'test', 'quality': 0.96},
                    'Vc': {'source': 'online', 'method': 'test', 'quality': 0.93},
                },
            },
            allow_online=False,
            allow_estimation=False,
        )

        self.assertEqual(critical['Zc'].source, 'exact')
        self.assertEqual(critical['Zc'].method, 'critical_volume_identity')
        self.assertClose(critical['Zc'].quality, 0.93)

    def test_low_quality_exact_metadata_is_still_soft(self):
        weak_exact = PropertyResolutionResult(
            value=10.0,
            source='exact',
            method='critical_volume_identity',
            quality=0.85,
        )
        strong_exact = PropertyResolutionResult(
            value=10.0,
            source='exact',
            method='critical_volume_identity',
            quality=0.95,
        )

        self.assertTrue(PropertyResolver._result_is_soft(weak_exact))
        self.assertFalse(PropertyResolver._result_is_soft(strong_exact))
        self.assertTrue(PropertyResolver._source_meta_is_soft({'source': 'exact', 'quality': 0.85}))
        self.assertFalse(PropertyResolver._source_meta_is_soft({'source': 'exact', 'quality': 0.95}))

    def test_liquid_volume_ptv_requires_hard_zc_before_yamada_gunn(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C4H10',
            'MW': 58.12,
            'Tc': 425.2,
            'Pc': 38.0,
            'omega': 0.2,
            'Zc': 0.25,
            'Tb': 272.7,
            'property_sources': {
                'Tc': {'source': 'local', 'method': 'test_tc', 'quality': 0.98},
                'Pc': {'source': 'local', 'method': 'test_pc', 'quality': 0.98},
                'omega': {'source': 'local', 'method': 'test_omega', 'quality': 0.98},
                'Tb': {'source': 'local', 'method': 'test_tb', 'quality': 0.98},
                'Zc': {'source': 'local', 'method': 'test_zc', 'quality': 0.92},
            },
        }

        with patch.object(PropertyResolver, '_fetch_liquid_density_online', return_value=None):
            hard = resolver.resolve_liquid_molar_volume('madeupium', 300.0, props)
        self.assertEqual(hard.method, 'ptv_eos_liquid_volume')
        self.assertClose(hard.quality, 0.92 * 0.84)

        soft_props = dict(props)
        soft_props['property_sources'] = dict(props['property_sources'])
        soft_props['property_sources']['Zc'] = {
            'source': 'estimated',
            'method': 'test_zc_estimate',
            'quality': 0.70,
        }
        with patch.object(PropertyResolver, '_fetch_liquid_density_online', return_value=None):
            soft = resolver.resolve_liquid_molar_volume('madeupium', 300.0, soft_props)
        self.assertEqual(soft.method, 'rackett_yamada_gunn')
        self.assertClose(soft.quality, 0.98 * 0.78)

    def test_liquid_volume_unsupported_formula_elements_fall_back_to_mw(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'NaOH',
            'MW': 40.0,
        }

        with patch.object(PropertyResolver, '_fetch_liquid_density_online', return_value=None):
            result = resolver.resolve_liquid_molar_volume('NaOH', 298.15, props)

        self.assertEqual(result.method, 'mw_liquid_volume_heuristic')
        self.assertEqual(result.source, 'estimated')
        self.assertGreater(result.value, 0.0)

    def test_bundled_sulfuric_acid_uses_perry_liquid_density_fit(self):
        database = ChemicalDatabase(enable_online=False)
        chemical = database.get('H2SO4', fetch_online=False)
        props = chemical.to_dict()
        resolver = PropertyResolver()
        correlation = props['property_correlations']['rhol']

        self.assertEqual(correlation['source'], 'Perry Table 2-55')
        self.assertClose(correlation['quality'], 0.98)
        self.assertClose(correlation['Tmin_K'], 273.15)
        self.assertClose(correlation['Tmax_K'], 373.15)

        for celsius in (0.0, 60.0, 100.0):
            with self.subTest(celsius=celsius):
                temperature = celsius + 273.15
                density = resolver.resolve_liquid_molar_density(
                    'H2SO4', temperature, props
                )
                volume = resolver.resolve_liquid_molar_volume(
                    'H2SO4', temperature, props
                )
                expected_mass_density = 1000.0 * (
                    1.8517 - 1.09e-3 * celsius + 1.64e-6 * celsius**2
                )

                self.assertEqual(
                    density.method,
                    'provided_liquid_density_fit',
                )
                self.assertEqual(density.source, 'provided')
                self.assertClose(density.quality, 0.98)
                self.assertEqual(
                    volume.method,
                    'provided_liquid_molar_volume_fit',
                )
                self.assertClose(
                    density.value * chemical.MW,
                    expected_mass_density,
                )
                self.assertClose(volume.value, 1.0 / density.value)

    def test_bundled_sulfuric_acid_uses_nist_janaf_liquid_cp_fit(self):
        database = ChemicalDatabase(enable_online=False)
        chemical = database.get('H2SO4', fetch_online=False)
        props = chemical.to_dict()
        resolver = PropertyResolver()
        correlation = props['property_correlations']['Cpl']

        self.assertIsNone(chemical.Cp_liquid)
        self.assertEqual(correlation['source'], 'NIST JANAF')
        self.assertClose(correlation['quality'], 0.97)
        self.assertClose(correlation['Tmin_K'], 283.15)
        self.assertClose(correlation['Tmax_K'], 573.15)
        kernel = resolver.resolve_liquid_cp_kernel(
            'H2SO4', props, allow_online=False
        )
        self.assertEqual(kernel.method, 'provided_poly_x_liquid_cp_kernel')
        self.assertEqual(kernel.source, 'NIST JANAF')
        self.assertClose(kernel.quality, 0.97)

        for celsius in (10.0, 60.0, 300.0):
            with self.subTest(celsius=celsius):
                result = resolver.resolve_heat_capacity(
                    'H2SO4',
                    celsius + 273.15,
                    'liquid',
                    props,
                    allow_online=False,
                )
                expected = 133.695 + 0.1939145 * celsius

                self.assertEqual(result.method, 'provided_heat_capacity_fit')
                self.assertEqual(result.source, 'NIST JANAF')
                self.assertClose(result.quality, 0.97)
                self.assertClose(result.value, expected)

    def test_liquid_volume_provided_rhol_extrapolates_then_uses_cached_fitted_rackett(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C5H12',
            'MW': 72.15,
            '_allow_online_lookup': False,
            'Tc': 469.7,
            'Pc': 33.7,
            'omega': 0.25,
            'property_correlations': {
                'rhol': {
                    'equation': 'poly_x',
                    'coefficients': {'A': 620.0, 'B': -80.0},
                    'Tmin_K': 250.0,
                    'Tmax_K': 350.0,
                    'quality': 0.98,
                },
            },
            'property_sources': {
                'Tc': {'source': 'local', 'method': 'test_tc', 'quality': 0.97},
                'Pc': {'source': 'local', 'method': 'test_pc', 'quality': 0.96},
                'omega': {'source': 'local', 'method': 'test_omega', 'quality': 0.95},
            },
        }

        near = resolver.resolve_liquid_molar_volume('madeupium', 360.0, props)
        self.assertEqual(near.method, 'provided_liquid_molar_volume_fit_extrapolated')
        self.assertClose(near.quality, 0.98 * 0.96)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'liquid_volume_zra_cache.json'
            resolver = PropertyResolver()
            with patch.object(PropertyResolver, 'LIQUID_VOLUME_ZRA_CACHE_PATH', cache_path):
                far = resolver.resolve_liquid_molar_volume('madeupium', 400.0, props)
                self.assertEqual(far.method, 'rackett_fitted_zra')
                self.assertClose(far.quality, 0.96 * 0.94)
                with sqlite3.connect(cache_path.with_suffix('.sqlite')) as connection:
                    row = connection.execute(
                        """
                        SELECT payload_json FROM runtime_json_cache
                        WHERE namespace = 'liquid_volume_zra_v1'
                        """
                    ).fetchone()
                self.assertIn('provided_rhol', row[0])

                cached = resolver.resolve_liquid_molar_volume('madeupium', 410.0, props)
                self.assertEqual(cached.method, 'rackett_fitted_zra')
                self.assertClose(cached.quality, far.quality)

    def test_all_thermo_methods_use_resolver_liquid_density_for_liquid_states(self):
        resolver = PropertyResolver()

        for method in ('IDEAL', 'RK', 'PR', 'UNIFAC'):
            with self.subTest(method=method):
                thermo = create_thermodynamics(['methanol'], method)
                state = thermo.calculate_state(
                    298.15,
                    50.0,
                    1.0,
                    {'methanol': 1.0},
                    phase='liquid',
                    flash=False,
                )
                volume = resolver.resolve_liquid_molar_volume(
                    'methanol',
                    298.15,
                    thermo._resolver_known_props['methanol'],
                ).value

                self.assertClose(state.rho, 1.0 / volume, rel=1e-10)

    def test_pubchem_density_parser_handles_messy_reporting_formats(self):
        resolver = PropertyResolver()
        info = {
            'Description': 'PEER REVIEWED',
            'Reference': ['Example handbook reference'],
        }

        points = []
        liquid_props = {'phase_at_STP': 'liquid'}
        for text in (
            '0.7893 g/cu cm at 20 °C',
            'Relative density (water = 1): 0.79',
            '1.040-1.047',
            '1050 kg/cu m at 25 °C',
            'Density of saturated vapor-air mixture at 760 mm Hg (air = 1): 1.22 at 26 °C',
            'Density of aqueous solutions at 20 °C/4 °C: 0.9939 (1%)',
            'Bulk density: 6.5 lb/gal',
            '1.05 at 1515 °F (USCG, 1999) - Denser than water',
        ):
            points.extend(
                resolver._parse_pubchem_liquid_density_text(
                    text,
                    info,
                    liquid_props,
                )
            )

        values = sorted(round(point['rho_g_cm3'], 4) for point in points)
        self.assertEqual(values, [0.7893, 0.79, 1.0435, 1.05])
        relative = [point for point in points if point['kind'] == 'relative'][0]
        self.assertClose(relative['rho_g_cm3'], 0.79)
        self.assertLess(relative['quality'], 0.82)
        self.assertTrue(relative['temperature_assumed'])
        self.assertEqual(relative['T_K'], 293.15)
        self.assertTrue(all(120.0 <= point['T_K'] <= 650.0 for point in points))

    def test_pubchem_density_cluster_rejects_inconsistent_values(self):
        resolver = PropertyResolver()
        points = []
        liquid_props = {'phase_at_STP': 'liquid'}
        for text in (
            '0.9629 g at 20 °C',
            '0.969 (USCG, 1999) - Less dense than water; will float',
            'Relative density (water = 1): 0.97',
            '1.413 @25 °C',
        ):
            points.extend(
                resolver._parse_pubchem_liquid_density_text(
                    text,
                    {},
                    liquid_props,
                )
            )

        selected = resolver._select_pubchem_density_cluster(points)
        selected_values = {round(point['rho_g_cm3'], 4) for point in selected}
        self.assertIn(0.9629, selected_values)
        self.assertIn(0.969, selected_values)
        self.assertIn(0.97, selected_values)
        self.assertNotIn(1.413, selected_values)

    def test_pubchem_density_fetch_caches_successful_cluster(self):
        resolver = PropertyResolver()
        fake_payload = {
            'Record': {
                'Section': [
                    {
                        'TOCHeading': 'Density',
                        'Information': [
                            {
                                'Description': 'PEER REVIEWED',
                                'Reference': ['CRC Handbook'],
                                'Value': {
                                    'StringWithMarkup': [
                                        {'String': '1.1498 @ 20 °C/4 °C'},
                                    ],
                                },
                            },
                            {
                                'Value': {
                                    'StringWithMarkup': [
                                        {'String': 'Relative density (water = 1): 1.15'},
                                    ],
                                },
                            },
                        ],
                    },
                ],
            },
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(fake_payload).encode('utf-8')

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PropertyResolver, 'CACHE_DIR', Path(tmpdir)):
                resolver = PropertyResolver()
                with patch.object(PropertyResolver, '_get_pubchem_cid', return_value=7751) as cid_mock:
                    with patch('urllib.request.urlopen', return_value=FakeResponse()) as urlopen_mock:
                        liquid_props = {
                            'name': 'ethyl chloroacetate',
                            'phase_at_STP': 'liquid',
                        }
                        first = resolver._fetch_liquid_density_pubchem(
                            'ethyl chloroacetate', liquid_props,
                        )
                        second = resolver._fetch_liquid_density_pubchem(
                            'ethyl chloroacetate', liquid_props,
                        )

        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertEqual(cid_mock.call_count, 1)
        self.assertEqual(urlopen_mock.call_count, 1)
        self.assertEqual(first['cid'], 7751)
        self.assertEqual(len(first['points']), 2)
        self.assertGreaterEqual(first['quality'], 0.85)
        self.assertLess(first['quality'], 0.88)
        self.assertTrue(any(point['temperature_assumed'] for point in first['points']))

    def test_pubchem_density_uses_rackett_when_critical_pair_is_hard(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C4H7ClO2',
            'MW': 122.55,
            'Tc': 590.0,
            'Pc': 41.0,
            'omega': 0.25,
            'Zc': 0.26,
            'property_sources': {
                'Tc': {'source': 'local', 'method': 'test_tc', 'quality': 0.96},
                'Pc': {'source': 'local', 'method': 'test_pc', 'quality': 0.95},
                'omega': {'source': 'local', 'method': 'test_omega', 'quality': 0.94},
                'Zc': {'source': 'local', 'method': 'test_zc', 'quality': 0.93},
            },
        }
        online = {
            'cid': 7751,
            'parsed_points': 3,
            'points': [
                {'rho_g_cm3': 1.15, 'T_K': 298.15, 'quality': 0.88, 'kind': 'explicit', 'text': '1.15 g/cm³'},
                {'rho_g_cm3': 1.1498, 'T_K': 293.15, 'quality': 0.88, 'kind': 'bare', 'text': '1.1498 @ 20 °C/4 °C'},
            ],
            'quality': 0.90,
        }

        with patch.object(PropertyResolver, '_fetch_liquid_density_online', return_value=online):
            result = resolver.resolve_liquid_molar_volume('ethyl chloroacetate', 330.0, props)

        self.assertEqual(result.source, 'online')
        self.assertEqual(result.method, 'pubchem_rackett_fitted_zra_liquid_volume')
        self.assertClose(result.quality, 0.90 * 0.98)
        self.assertIn('fitted from PubChem density', result.notes)

    def test_pubchem_density_thermal_expansion_terminates_before_predictive_methods(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'C8H14O',
            'MW': 126.20,
            'Tc': 600.0,
            'Pc': 30.0,
            'omega': 0.3,
            'Zc': 0.26,
            'property_sources': {
                'Tc': {'source': 'estimated', 'method': 'test_tc', 'quality': 0.70},
                'Pc': {'source': 'estimated', 'method': 'test_pc', 'quality': 0.70},
                'omega': {'source': 'estimated', 'method': 'test_omega', 'quality': 0.70},
                'Zc': {'source': 'estimated', 'method': 'test_zc', 'quality': 0.70},
            },
        }
        online = {
            'cid': 61346,
            'parsed_points': 1,
            'points': [
                {'rho_g_cm3': 0.816, 'T_K': 298.15, 'quality': 0.80, 'kind': 'range', 'text': '0.813-0.819'},
            ],
            'quality': 0.80,
        }

        with patch.object(PropertyResolver, '_fetch_liquid_density_online', return_value=online):
            result = resolver.resolve_liquid_molar_volume('1-octen-3-one', 298.15, props)

        self.assertEqual(result.source, 'online')
        self.assertEqual(result.method, 'pubchem_liquid_density_thermal_expansion')
        self.assertClose(result.value, 126.20 / 816.0)
        self.assertClose(result.quality, 0.80)

    def test_liquid_volume_atom_fallback_applies_ring_compactness_correction(self):
        resolver = PropertyResolver()

        def density(name, formula, MW, smiles=None):
            props = {'formula': formula, 'MW': MW}
            if smiles is not None:
                props['smiles'] = smiles
            result = resolver._formula_liquid_volume_estimate(name, 298.15, props)
            return MW / result.value / 1000.0, result

        hexene_rho, hexene = density('hexene', 'C6H12', 84.161, 'C=CCCCC')
        cyclohexane_rho, cyclohexane = density('cyclohexane', 'C6H12', 84.161, 'C1CCCCC1')
        benzene_raw_rho, _ = density('benzene', 'C6H6', 78.112, 'C=CC=CC=C')
        benzene_rho, _benzene = density('benzene', 'C6H6', 78.112, 'c1ccccc1')
        thf_raw_rho, _ = density('THF', 'C4H8O', 72.107, 'CCCCO')
        thf_rho, thf = density('THF', 'C4H8O', 72.107, 'C1CCOC1')
        pyrrole_raw_rho, _ = density('pyrrole', 'C4H5N', 67.091, 'C=CC=CN')
        pyrrole_rho, pyrrole = density('pyrrole', 'C4H5N', 67.091, '[nH]1cccc1')
        furan_raw_rho, _ = density('furan', 'C4H4O', 68.075, 'C=CC=CO')
        named_furan_rho, named_furan = density('furan', 'C4H4O', 68.075)
        smiles_wins_rho, smiles_wins = density('benzene', 'C6H14', 86.18, 'CCCCCC')
        smiles_wins_raw_rho, _ = density('hexane', 'C6H14', 86.18, 'CCCCCC')

        self.assertClose(cyclohexane_rho / hexene_rho, 1.35 / 1.15)
        self.assertClose(benzene_rho / benzene_raw_rho, 1.35 / 1.15)
        self.assertClose(thf_rho / thf_raw_rho, 1.25 / 1.05)
        self.assertClose(pyrrole_rho / pyrrole_raw_rho, 1.25 / 1.05)
        self.assertClose(named_furan_rho / furan_raw_rho, 1.25 / 1.05)
        self.assertClose(smiles_wins_rho, smiles_wins_raw_rho)
        self.assertIn('ring compactness density multiplier 1.35', cyclohexane.notes)
        self.assertIn('ring compactness density multiplier 1.25', thf.notes)
        self.assertNotIn('ring compactness', hexene.notes)
        self.assertNotIn('ring compactness', smiles_wins.notes)

    def test_liquid_volume_atom_fallback_applies_chain_acid_and_aromatic_name_corrections(self):
        resolver = PropertyResolver()

        def multiplier(name, counts, identifiers=None):
            props = {'name': name, 'identifiers': identifiers or []}
            return resolver._liquid_volume_atom_density_multiplier(name, props, counts)

        self.assertEqual(multiplier('pentane', {'C': 5}), (1.05, 'carbon-count chain correction'))
        self.assertEqual(multiplier('octane', {'C': 8}), (1.15, 'carbon-count chain correction'))
        self.assertEqual(multiplier('dodecane', {'C': 12}), (1.25, 'carbon-count chain correction'))
        self.assertEqual(multiplier('tridecane', {'C': 13}), (1.30, 'carbon-count chain correction'))
        self.assertEqual(
            multiplier('hexanoic acid', {'C': 6, 'O': 2}, identifiers=['hexanoic acid']),
            (1.265, 'carbon-count chain correction + carboxylic-acid correction'),
        )
        self.assertEqual(
            multiplier('nonanoic acid', {'C': 9, 'O': 2}, identifiers=['nonanoic acid']),
            (1.30, 'carbon-count chain correction + carboxylic-acid correction'),
        )

        aromatic_names = (
            'styrene',
            'm-cresol',
            'anisole',
            'cumene',
            '1,2,3,4-tetrahydronaphthalene',
        )
        for name in aromatic_names:
            self.assertEqual(
                resolver._liquid_volume_ring_compactness_factor(name, {'name': name}),
                1.35,
            )

    def test_liquid_volume_pr_fallback_uses_local_pressure_floor(self):
        resolver = PropertyResolver()
        props = {
            'formula': 'Xe',
            'MW': 131.3,
            'Tc': 500.0,
            'Pc': 40.0,
            'omega': 3.0,
            'Tb': 300.0,
            'property_sources': {
                'Tc': {'source': 'local', 'method': 'test_tc', 'quality': 0.98},
                'Pc': {'source': 'local', 'method': 'test_pc', 'quality': 0.98},
                'omega': {'source': 'local', 'method': 'test_omega', 'quality': 0.98},
                'Tb': {'source': 'local', 'method': 'test_tb', 'quality': 0.98},
            },
        }
        low_psat = PropertyResolutionResult(
            value=1.0e-9,
            source='calculated',
            method='test_low_psat',
            quality=0.97,
        )

        with patch.object(PropertyResolver, '_fetch_liquid_density_online', return_value=None):
            with patch.object(resolver, 'resolve_vapor_pressure', return_value=low_psat):
                volume = resolver.resolve_liquid_molar_volume('madeupium', 300.0, props)

        self.assertEqual(volume.method, 'pr_eos_liquid_volume')
        self.assertClose(volume.quality, 0.98 * 0.75)
        self.assertIn('P=1e-05 bar', volume.notes)
        self.assertIn('minimum 1 Pa', volume.notes)

    def test_liquid_volume_fallback_accuracy_against_perry_density_sample(self):
        resolver = PropertyResolver()
        library = get_perry_property_library()
        library._load()
        errors = {
            'fitted_zra': [],
            'ptv': [],
            'yamada_gunn': [],
        }

        def interior_temperatures(Tmin, Tmax, count=5):
            return [
                Tmin + (Tmax - Tmin) * (index + 1) / (count + 1)
                for index in range(count)
            ]

        def percentile(values, percent):
            ordered = sorted(values)
            position = (len(ordered) - 1) * percent / 100.0
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return ordered[lower]
            return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)

        for _cas, entry in library.chemicals.items():
            critical = entry.get('critical_constants') or {}
            rows = entry.get('liquid_density') or []
            if not rows:
                continue
            try:
                Tc = float(critical['Tc_K'])
                Pc = float(critical['Pc_MPa']) * 10.0
                omega = float(critical['omega'])
                Zc = float(critical['Zc'])
            except (KeyError, TypeError, ValueError):
                continue

            for row in rows:
                if row.get('equation_id') not in {100, 105}:
                    continue
                Tmin = float(row['T_min_K'])
                Tmax = float(row['T_max_K'])
                fit_samples = []
                for sample_T in resolver._sample_temperatures(Tmin, Tmax):
                    rho = library._eval_liquid_density_mol_per_dm3(row, sample_T)
                    if rho and rho > 0.0:
                        fit_samples.append((sample_T, 1.0 / rho))
                zra_fit = resolver._fit_zra_from_samples(fit_samples, Tc, Pc)
                zra_yamada_gunn = resolver._yamada_gunn_zra(omega)
                self.assertIsNotNone(zra_fit)
                self.assertIsNotNone(zra_yamada_gunn)

                for T in interior_temperatures(Tmin, Tmax):
                    rho_ref = library._eval_liquid_density_mol_per_dm3(row, T)
                    if rho_ref is None or rho_ref <= 0.0:
                        continue

                    fitted_volume = resolver._rackett_volume_m3_per_kmol(T, Tc, Pc, zra_fit)
                    yg_volume = resolver._rackett_volume_m3_per_kmol(T, Tc, Pc, zra_yamada_gunn)
                    ptv_volume = resolver._pure_ptv_liquid_volume(T, 1.01325, Tc, Pc, omega, Zc)
                    self.assertIsNotNone(fitted_volume)
                    self.assertIsNotNone(yg_volume)

                    errors['fitted_zra'].append(abs(1.0 / fitted_volume - rho_ref) / rho_ref * 100.0)
                    errors['yamada_gunn'].append(abs(1.0 / yg_volume - rho_ref) / rho_ref * 100.0)
                    if ptv_volume is not None:
                        errors['ptv'].append(abs(1.0 / ptv_volume - rho_ref) / rho_ref * 100.0)

        self.assertGreater(len(errors['fitted_zra']), 1500)
        self.assertGreater(len(errors['ptv']), 1500)
        self.assertLess(percentile(errors['fitted_zra'], 50), 0.5)
        self.assertLess(percentile(errors['fitted_zra'], 90), 2.0)
        self.assertGreater(sum(err < 5.0 for err in errors['fitted_zra']) / len(errors['fitted_zra']), 0.98)
        self.assertLess(percentile(errors['ptv'], 50), 4.0)
        self.assertLess(percentile(errors['ptv'], 90), 12.5)
        self.assertGreater(sum(err < 5.0 for err in errors['ptv']) / len(errors['ptv']), 0.63)
        self.assertLess(percentile(errors['yamada_gunn'], 50), 5.0)
        self.assertLess(percentile(errors['yamada_gunn'], 90), 18.0)
        self.assertGreater(sum(err < 5.0 for err in errors['yamada_gunn']) / len(errors['yamada_gunn']), 0.55)

    def test_critical_identity_with_estimated_input_is_calculated_and_quality_weighted(self):
        resolver = PropertyResolver()

        critical = resolver.resolve_critical_properties(
            'madeupium',
            {
                'Tc': 500.0,
                'Pc': 50.0,
                'Vc': 200.0,
                'property_sources': {
                    'Tc': {'source': 'local', 'method': 'test', 'quality': 0.98},
                    'Pc': {'source': 'estimated', 'method': 'lydersen_style', 'quality': 0.60},
                    'Vc': {'source': 'online', 'method': 'test', 'quality': 0.96},
                },
            },
            allow_online=False,
            allow_estimation=False,
        )

        self.assertEqual(critical['Zc'].source, 'calculated')
        self.assertEqual(critical['Zc'].method, 'critical_volume_identity')
        self.assertClose(critical['Zc'].quality, 0.60)

    def test_lee_kesler_quality_uses_minimum_input_quality_times_method_factor(self):
        resolver = PropertyResolver()

        critical = resolver.resolve_critical_properties(
            'madeupium',
            {
                'Tc': 500.0,
                'Pc': 50.0,
                'Tb': 350.0,
                'property_sources': {
                    'Tc': {'source': 'local', 'method': 'test', 'quality': 0.98},
                    'Pc': {'source': 'estimated', 'method': 'lydersen_style', 'quality': 0.60},
                    'Tb': {'source': 'textbook', 'method': 'test', 'quality': 0.96},
                },
            },
            allow_online=False,
            allow_estimation=True,
        )

        self.assertEqual(critical['omega'].method, 'lee_kesler')
        self.assertClose(critical['omega'].quality, 0.60 * 0.80)
        self.assertIn('HBA/HBD unavailable', critical['omega'].notes)

    def test_lee_kesler_quality_uses_hbond_descriptor_penalty(self):
        resolver = PropertyResolver()
        base = {
            'Tc': 500.0,
            'Pc': 50.0,
            'Tb': 350.0,
            'property_sources': {
                'Tc': {'source': 'local', 'method': 'test', 'quality': 0.98},
                'Pc': {'source': 'local', 'method': 'test', 'quality': 0.97},
                'Tb': {'source': 'textbook', 'method': 'test', 'quality': 0.96},
            },
        }

        normal = resolver.resolve_critical_properties(
            'madeupium',
            {**base, 'smiles': 'COCCO'},
            allow_online=False,
            allow_estimation=True,
        )
        multiple_donors = resolver.resolve_critical_properties(
            'madeupium',
            {**base, 'smiles': 'OCCO'},
            allow_online=False,
            allow_estimation=True,
        )
        multiple_acceptors = resolver.resolve_critical_properties(
            'madeupium',
            {**base, 'smiles': 'COC(=O)OC'},
            allow_online=False,
            allow_estimation=True,
        )

        self.assertClose(normal['omega'].quality, 0.96 * 0.95)
        self.assertClose(multiple_donors['omega'].quality, 0.96 * 0.80)
        self.assertClose(multiple_acceptors['omega'].quality, 0.96 * 0.80)
        self.assertIn('HBA=2, HBD=1', normal['omega'].notes)
        self.assertIn('HBA=2, HBD=2', multiple_donors['omega'].notes)
        self.assertIn('HBA=3, HBD=0', multiple_acceptors['omega'].notes)

    def test_exact_formation_derivations_preserve_minimum_real_input_quality(self):
        resolver = PropertyResolver()

        formation = resolver.resolve_formation_properties(
            'madeupium',
            {
                'formula': 'H2O',
                'Hf': -241.83,
                'S': 188.84,
                'property_sources': {
                    'Hf': {'source': 'local', 'method': 'test', 'quality': 0.96},
                    'S': {'source': 'textbook', 'method': 'test', 'quality': 0.95},
                },
            },
            allow_online=False,
        )

        self.assertEqual(formation['Gf'].source, 'exact')
        self.assertEqual(formation['Gf'].method, 'hf_absolute_entropy_to_gf')
        self.assertClose(formation['Gf'].quality, 0.95)

    def test_hf_from_liquid_and_hvap_is_exact_when_inputs_are_real(self):
        resolver = PropertyResolver()

        formation = resolver.resolve_formation_properties(
            'madeupium',
            {
                'Hf_liquid': -285.82,
                'Hvap': 43.99,
                'property_sources': {
                    'Hf_liquid': {'source': 'local', 'method': 'test', 'quality': 0.96},
                    'Hvap': {'source': 'textbook', 'method': 'test', 'quality': 0.95},
                },
            },
            allow_online=False,
        )

        self.assertEqual(formation['Hf'].source, 'exact')
        self.assertEqual(formation['Hf'].method, 'liquid_hf_plus_hvap')
        self.assertClose(formation['Hf'].value, -241.83)
        self.assertClose(formation['Hf'].quality, 0.95)

    def test_estimator_quality_uses_input_quality_and_method_factor(self):
        resolver = PropertyResolver()

        boiling = resolver.resolve_boiling_point(
            'madeupium',
            {
                'MW': 100.0,
                'property_sources': {
                    'MW': {'source': 'online', 'method': 'test_mw', 'quality': 0.80},
                },
            },
            allow_online=False,
            allow_estimation=True,
        )
        self.assertEqual(boiling.method, 'mw_correlation')
        self.assertClose(boiling.value, 64.29202662 * 100.0 ** 0.39694617)
        self.assertClose(boiling.quality, 0.80 * 0.30)

        formula_boiling = resolver.resolve_boiling_point(
            'madeupium',
            {
                'MW': 100.0,
                'formula': 'C6H6',
                'property_sources': {
                    'MW': {'source': 'online', 'method': 'test_mw', 'quality': 0.80},
                },
            },
            allow_online=False,
            allow_estimation=True,
        )
        expected_formula_tb = (
            166.312
            - 2.95305 * 6
            + 67.7668 * 6 ** 0.685653
        )
        self.assertEqual(formula_boiling.method, 'formula_no_hbd_boiling_point')
        self.assertClose(formula_boiling.value, expected_formula_tb)
        self.assertClose(formula_boiling.quality, 0.45)

        with patch.object(resolver, '_estimate_nannoolal_boiling_point', return_value=None):
            hbd_boiling = resolver.resolve_boiling_point(
                'madeupium',
                {'formula': 'C2H6O', 'MW': 46.07, 'smiles': 'CCO'},
                allow_online=False,
                allow_estimation=True,
            )
        expected_hbd_tb = (
            166.312
            - 2.95305 * 6
            + 67.7668 * 2 ** 0.685653
            + 23.4180
            + 46.7294
        )
        self.assertEqual(hbd_boiling.method, 'formula_hbd_boiling_point')
        self.assertClose(hbd_boiling.value, expected_hbd_tb)
        self.assertClose(hbd_boiling.quality, 0.55)

        critical = resolver.resolve_critical_properties(
            'madeupium',
            {
                'MW': 100.0,
                'formula': 'C6H6',
                'property_sources': {
                    'MW': {'source': 'online', 'method': 'test_mw', 'quality': 0.80},
                },
            },
            allow_online=False,
            allow_estimation=True,
        )
        estimated_tb = formula_boiling.value
        expected_pc = 205.0 / 6.0 * (estimated_tb / 373.15) ** 0.8
        self.assertEqual(critical['Tc'].method, 'guldberg_rule')
        self.assertClose(critical['Tc'].value, 1.5 * estimated_tb)
        self.assertClose(critical['Tc'].quality, 0.45 * 0.80)
        self.assertEqual(critical['Pc'].method, 'atom_count_ring_tb_pc')
        self.assertClose(critical['Pc'].value, expected_pc)
        self.assertClose(critical['Pc'].quality, 0.45)

        hvap = resolver.resolve_hvap(
            'madeupium',
            {
                'Tb': 350.0,
                'property_sources': {
                    'Tb': {'source': 'online', 'method': 'test_tb', 'quality': 0.90},
                },
            },
            allow_online=False,
            allow_estimation=True,
        )
        self.assertEqual(hvap.method, 'trouton')
        self.assertClose(hvap.quality, 0.90 * 0.55)

        with tempfile.TemporaryDirectory() as directory:
            cp_resolver = PropertyResolver()
            cp_resolver.CACHE_DIR = Path(directory)
            cp = cp_resolver.resolve_heat_capacity(
                'qualityium',
                300.0,
                phase='liquid',
                props={
                    'MW': 90.0,
                    'formula': 'C6H14',
                    'property_sources': {
                        'MW': {'source': 'online', 'method': 'test_mw', 'quality': 0.82},
                    },
                },
                allow_online=False,
            )
        self.assertEqual(cp.method, 'scaled_ideal_gas_liquid_cp_kernel')
        # The estimated Tc/omega are below the 0.70 predictive-admission floor,
        # so the resolver continues to the terminal 1.3x multiplier.
        self.assertClose(cp.quality, 0.60 * 0.80)

    def test_nannoolal_estimator_quality_tiers(self):
        resolver = PropertyResolver()
        props = {'formula': 'C2H6O', 'smiles': 'CCO'}

        boiling = resolver.resolve_boiling_point(
            'madeupium',
            props,
            allow_online=False,
            allow_estimation=True,
        )
        self.assertEqual(boiling.method, 'nannoolal_tb')
        self.assertClose(boiling.quality, 0.80)

        critical = resolver.resolve_critical_properties(
            'madeupium',
            props,
            allow_online=False,
            allow_estimation=True,
        )
        self.assertEqual(critical['Tc'].method, 'nannoolal_tc')
        self.assertClose(critical['Tc'].quality, 0.75)
        self.assertEqual(critical['Pc'].method, 'nannoolal_pc')
        self.assertClose(critical['Pc'].quality, 0.80)
        self.assertEqual(critical['Vc'].method, 'nannoolal_vc')
        self.assertClose(critical['Vc'].quality, 0.85)

        real_tb_critical = resolver.resolve_critical_properties(
            'madeupium',
            {'formula': 'C2H6O', 'smiles': 'CCO', 'Tb': 351.5},
            allow_online=False,
            allow_estimation=True,
        )
        self.assertEqual(real_tb_critical['Tc'].method, 'nannoolal_tc')
        self.assertClose(real_tb_critical['Tc'].quality, 0.85)

    def test_nannoolal_aromatic_nitro_extension_and_conjugated_refusal(self):
        resolver = PropertyResolver()
        nitrobenzene = resolver.resolve_critical_properties(
            'group69 extension test',
            {
                'formula': 'C6H5NO2',
                'smiles': 'O=[N+]([O-])c1ccccc1',
                'Tb': 483.85,
                'property_sources': {
                    'Tb': {
                        'source': 'reference',
                        'method': 'experimental_tb',
                        'quality': 0.95,
                    },
                    'smiles': {
                        'source': 'reference',
                        'method': 'exact_structure',
                        'quality': 0.99,
                    },
                },
            },
            allow_online=False,
            allow_estimation=True,
        )
        self.assertEqual(nitrobenzene['Tc'].method, 'nannoolal_tc')
        self.assertEqual(nitrobenzene['Pc'].method, 'nannoolal_pc')
        self.assertEqual(nitrobenzene['Vc'].method, 'nannoolal_vc')
        self.assertClose(nitrobenzene['Tc'].value, 718.0, rel=0.01)
        self.assertClose(nitrobenzene['Pc'].value, 41.6, rel=0.04)
        self.assertClose(nitrobenzene['Vc'].value, 335.0, rel=0.04)

        for cas, formula, smiles in (
            ('88-74-4', 'C6H6N2O2', 'Nc1ccccc1[N+](=O)[O-]'),
            ('88-75-5', 'C6H5NO3', 'Oc1ccccc1[N+](=O)[O-]'),
        ):
            with self.subTest(cas=cas):
                refused = resolver.resolve_critical_properties(
                    cas,
                    {'CAS': cas, 'formula': formula, 'smiles': smiles},
                    allow_online=False,
                    allow_estimation=True,
                )
                for property_name in ('Tc', 'Pc', 'Vc'):
                    self.assertFalse(
                        refused[property_name].method.startswith('nannoolal_')
                    )

    def test_opsin_smiles_to_nannoolal_end_to_end_for_unlisted_ester(self):
        name = 'Isobutyl 2,3,3-trifluorocyclopentane-1-carboxylate'
        expected_smiles = 'FC1C(CCC1(F)F)C(=O)OCC(C)C'
        cache_path = Path(ROOT, 'data', 'runtime', 'smiles_cache.sqlite')
        with sqlite3.connect(cache_path) as connection:
            connection.execute(
                """
                DELETE FROM smiles_cache
                WHERE identifier = ? OR smiles = ?
                """,
                (' '.join(name.lower().split()), expected_smiles),
            )

        resolver = PropertyResolver()
        props = {'name': name, 'symbol': name}

        mw = resolver.resolve_molecular_weight(name, props, allow_online=False)
        boiling = resolver.resolve_boiling_point(
            name,
            props,
            allow_online=False,
            allow_estimation=True,
        )
        critical = resolver.resolve_critical_properties(
            name,
            props,
            allow_online=False,
            allow_estimation=True,
        )

        self.assertEqual(mw.method, 'rdkit_molwt_from_smiles')
        self.assertIn('py2opsin', mw.notes)
        self.assertClose(mw.value, 224.222, rel=1e-12)

        self.assertEqual(boiling.method, 'nannoolal_tb')
        self.assertIn('opsin_smiles_cache', boiling.notes)
        self.assertClose(boiling.value, 462.9431034614878)

        self.assertEqual(critical['Tc'].method, 'nannoolal_tc')
        self.assertEqual(critical['Pc'].method, 'nannoolal_pc')
        self.assertEqual(critical['Vc'].method, 'nannoolal_vc')
        self.assertClose(critical['Tc'].value, 638.4378903702496)
        self.assertClose(critical['Pc'].value, 23.009775835318642)
        self.assertClose(critical['Vc'].value, 637.7339088935)

        with sqlite3.connect(cache_path) as connection:
            row = connection.execute(
                """
                SELECT smiles, source, quality
                FROM smiles_cache
                WHERE identifier = ?
                """,
                (' '.join(name.lower().split()),),
            ).fetchone()
        self.assertEqual(row, (expected_smiles, 'opsin', 0.97))

    def test_pc_fallback_name_ring_score_covers_common_ring_stems(self):
        resolver = PropertyResolver()
        cases = {
            'p-Xylene': 2,
            'Diphenyl ether': 4,
            'Ethylene oxide': 1,
            '1,2,3,4-Tetrahydronaphthalene': 3,
            'Phenanthrene': 6,
            'Tetrahydrothiophene': 1,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    resolver._pc_fallback_name_ring_score(name),
                    expected,
                )

    def test_charged_species_are_banned_from_estimation(self):
        resolver = PropertyResolver()
        props = {
            'symbol': 'OX',
            'name': 'oxalate',
            'formula': 'C2O4-2',
            'Tb': 419.2,
            'MW': 88.02,
        }

        self.assertIsNone(resolver._pc_fallback_non_hydrogen_atoms(props['formula']))
        self.assertIsNone(resolver._estimate_pc_from_formula_tb(props))
        self.assertIsNone(resolver._estimate_vc_from_formula_structure(props))
        self.assertIsNone(resolver._estimate_boiling_point_from_formula(props))
        # The MW power law is formula-blind, so the ladder entry itself
        # must refuse before reaching it.
        tb_props = {key: value for key, value in props.items() if key != 'Tb'}
        self.assertIsNone(resolver._estimate_boiling_point_fallback(tb_props))

        # The neutral parent stays estimable through the same paths.
        self.assertEqual(resolver._pc_fallback_non_hydrogen_atoms('C2H2O4'), 6)

    def test_online_negative_cache_still_records_missing_pubchem_lookups_once(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch.object(OnlinePropertyFetcher, '_load_lookup_rules', return_value={'reference_temperature_K': 298.15}):
                fetcher = OnlinePropertyFetcher(cache_dir=cache_dir)

            with patch.object(fetcher, '_get_cached', return_value=None), \
                 patch.object(fetcher, '_save_missing_cache') as save_missing, \
                 patch('urllib.request.urlopen', side_effect=ValueError('not found')):
                self.assertIsNone(fetcher.fetch_from_pubchem('not-a-real-perry-or-pubchem-thing'))

        save_missing.assert_called_once()

    def test_psat_and_cp_use_same_path_from_resolver_props_and_thermo(self):
        database = ChemicalDatabase(enable_online=False)
        props = database.get('ethanol', fetch_online=False)
        thermo = IdealThermodynamics(['C2H5OH'], db=database)
        resolver = PropertyResolver()

        T = 350.0
        resolver_psat = resolver.resolve_vapor_pressure('ethanol', T).value
        props_psat = props.Psat(T)
        thermo_psat = thermo.Psat('C2H5OH', T)

        self.assertClose(props_psat, resolver_psat)
        self.assertClose(thermo_psat, resolver_psat)

        resolver_cp = resolver.resolve_heat_capacity('ethanol', T, phase='ideal_gas').value
        props_cp = props.Cp(T)
        thermo_cp = thermo.Cp_ideal_gas('C2H5OH', T)

        self.assertClose(props_cp, resolver_cp)
        self.assertClose(thermo_cp, resolver_cp)



if __name__ == '__main__':
    unittest.main()
