import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from property_resolution import (
    DensityObservation,
    PropertyResolutionError,
    PropertyResolutionResult,
)
from property_resolution.runtime_cache import cache_key_metadata
from property_resolver import PropertyResolver


class DensityPhaseResolutionTests(unittest.TestCase):
    @staticmethod
    def solid_props(**updates):
        props = {
            'name': 'naphthalene',
            'CAS': '91-20-3',
            'formula': 'C10H8',
            'MW': 128.17,
            'Tm': 353.35,
            'Tb': 491.0,
            'phase_at_STP': 'solid',
            'property_sources': {
                'Tm': {'source': 'local', 'method': 'direct', 'quality': 0.98},
                'Tb': {'source': 'local', 'method': 'direct', 'quality': 0.98},
                'MW': {'source': 'local', 'method': 'direct', 'quality': 0.99},
            },
        }
        props.update(updates)
        return props

    @staticmethod
    def liquid_props(**updates):
        props = {
            'name': 'example liquid',
            'CAS': '100-00-0',
            'formula': 'C5H10',
            'MW': 70.0,
            'Tm': 180.0,
            'Tb': 400.0,
            'phase_at_STP': 'liquid',
            'property_sources': {
                'Tm': {'source': 'local', 'method': 'direct', 'quality': 0.98},
                'Tb': {'source': 'local', 'method': 'direct', 'quality': 0.98},
            },
        }
        props.update(updates)
        return props

    def observation(self, **updates):
        values = {
            'mass_density_kg_m3': 1160.0,
            'temperature_K': 293.15,
            'phase': 'solid',
            'phase_basis': 'explicit_solid_wording',
            'source': 'pubchem',
            'method': 'pubchem_reported_density',
            'quality': 0.90,
            'raw': '1.16 g/cm3 at 20 C',
        }
        values.update(updates)
        return DensityObservation(**values)

    def test_parser_handles_common_and_exotic_density_units(self):
        resolver = PropertyResolver()
        cases = (
            ('1.162 at 20 °C/4 °C', 1162.0, 293.15),
            ('Solid density: 2170 kg/cu m at 25 °C', 2170.0, 298.15),
            ('Crystal density 1.20-1.24 g cm-3 at 293 K', 1220.0, 293.0),
            ('Liquid density 6.5 lb/US gal at 68 °F', 778.8717755, 293.15),
            ('Solid density 80 lb/ft3 at 20 deg C', 1281.47704, 293.15),
            ('Density 0.789 kg/L at 20 C', 789.0, 293.15),
            ('Density 789 g/L at 20 C', 789.0, 293.15),
            ('Specific gravity d20/4 1.25', 1250.0, 293.15),
            ('Crystal density 19.3 g/cm3 at 20 C', 19300.0, 293.15),
            ('Density 1.15E3 kg/m3 at 298 K', 1150.0, 298.0),
            ('Crystal density 1.234 ± 0.005 g·cm⁻³ at 20-22 °C', 1234.0, 294.15),
            ('Crystal density 1.234(2) g cm-3 at 293.15(5) K', 1234.0, 293.15),
            ('Density 1.15 × 10^3 kg m⁻³ at 25 +/- 2 °C', 1150.0, 298.15),
        )
        for text, expected_density, expected_temperature in cases:
            with self.subTest(text=text):
                records = resolver._parse_pubchem_density_text(text, {})
                self.assertEqual(len(records), 1)
                self.assertAlmostEqual(
                    records[0]['mass_density_kg_m3'], expected_density, places=5,
                )
                self.assertAlmostEqual(
                    records[0]['temperature_K'], expected_temperature, places=5,
                )

    def test_parser_rejects_nonintrinsic_and_predictive_density_rows(self):
        resolver = PropertyResolver()
        rejected = (
            'Bulk density: 6.5 lb/gal',
            'Tapped density 0.72 g/cm3',
            'Powder apparent density 0.55 g/cm3',
            'Relative vapor density (air = 1): 4.42',
            'Density of aqueous solution: 1.2 g/cm3',
            'Density of 20 wt% commercial formulation: 1.08 g/cm3',
            'Predicted density: 1.3 g/cm3',
            'Specific volume: 0.9 cm3/g',
        )
        for text in rejected:
            with self.subTest(text=text):
                self.assertEqual(resolver._parse_pubchem_density_text(text, {}), [])

    def test_parser_preserves_calculated_crystal_and_material_form_metadata(self):
        resolver = PropertyResolver()
        records = resolver._parse_pubchem_density_text(
            'Calculated crystal density of the monohydrate alpha form: 1.44 g/cm3',
            {'Reference': 'Example crystallographic article'},
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record['method'], 'pubchem_calculated_crystal_density')
        self.assertEqual(record['phase_hint'], 'solid')
        self.assertEqual(record['material_form'], 'hydrate')
        self.assertEqual(record['polymorph'], 'alpha')
        self.assertLess(record['quality'], 0.90)

    def test_density_specific_crystal_forms_preserve_sulfur_allotropes(self):
        resolver = PropertyResolver()
        for text, expected in (
            ('2.07 g/cu cm (Sulfur (rhombic)/', 'rhombic'),
            ('2.00 (Sulfur (monoclinic)/', 'monoclinic'),
            ('Density 1.96 /beta-Sulfur/', 'beta'),
        ):
            with self.subTest(text=text):
                record = resolver._parse_pubchem_density_text(text, {})[0]
                self.assertEqual(record['polymorph'], expected)
                self.assertIn(expected, record['form_label'])

    def test_bare_hydrate_cid_density_inherits_identity_form(self):
        resolver = PropertyResolver()
        raw = resolver._parse_pubchem_density_text('1.85 @ 20°C', {})[0]
        observation = resolver._density_observation_from_record(
            raw,
            {
                'name': 'calcium chloride dihydrate',
                'phase_at_STP': 'solid',
            },
            'calcium chloride dihydrate CaCl2.2H2O',
        )
        self.assertEqual(observation.material_form, 'hydrate')
        self.assertTrue(observation.is_identity_form)
        self.assertTrue(
            observation.metadata['material_form_inherited_from_identity']
        )

    def test_phase_classification_uses_wording_then_hard_boundaries_then_stp(self):
        resolver = PropertyResolver()
        solid_props = self.solid_props()
        liquid_props = self.liquid_props()
        cases = (
            ({'phase_hint': 'solid', 'temperature_K': 500.0}, solid_props, 'solid', 'explicit_solid'),
            ({'phase_hint': 'liquid', 'temperature_K': 200.0}, solid_props, 'liquid', 'explicit_liquid'),
            ({'phase_hint': 'ambiguous', 'temperature_K': 293.15}, solid_props, 'solid', 'below hard Tm'),
            ({'phase_hint': 'ambiguous', 'temperature_K': 300.0}, liquid_props, 'liquid', 'between hard Tm'),
            ({'phase_hint': 'ambiguous', 'temperature_K': None}, solid_props, 'solid', 'phase_at_STP=solid'),
            ({'phase_hint': 'ambiguous', 'temperature_K': None}, liquid_props, 'liquid', 'phase_at_STP=liquid'),
            ({'phase_hint': 'ambiguous', 'temperature_K': 352.0}, solid_props, 'ambiguous', 'within'),
            ({'phase_hint': 'ambiguous', 'temperature_K': 500.0}, solid_props, 'ambiguous', 'not safely'),
        )
        for record, props, expected_phase, basis_text in cases:
            with self.subTest(record=record, expected_phase=expected_phase):
                phase, basis, _ = resolver._classify_density_phase(record, props)
                self.assertEqual(phase, expected_phase)
                self.assertIn(basis_text, basis)

    def test_soft_tm_does_not_classify_a_temperature_specific_density(self):
        resolver = PropertyResolver()
        props = self.solid_props(
            phase_at_STP='unknown',
            property_sources={
                'Tm': {'source': 'estimated', 'method': 'estimated_tm', 'quality': 0.95},
            },
        )
        phase, basis, _ = resolver._classify_density_phase(
            {'phase_hint': 'ambiguous', 'temperature_K': 293.15},
            props,
        )
        self.assertEqual(phase, 'ambiguous')
        self.assertIn('no explicit phase', basis)

    def test_tm_alone_does_not_prove_a_high_temperature_report_is_liquid(self):
        resolver = PropertyResolver()
        props = self.solid_props(Tb=None, phase_at_STP='solid')
        phase, basis, _ = resolver._classify_density_phase(
            {'phase_hint': 'ambiguous', 'temperature_K': 400.0},
            props,
        )
        self.assertEqual(phase, 'ambiguous')
        self.assertIn('no trustworthy Tb', basis)

    def test_tb_at_quality_080_is_sufficient_only_as_a_phase_upper_bound(self):
        resolver = PropertyResolver()
        record = {'phase_hint': 'ambiguous', 'temperature_K': 400.0}
        for quality, expected in ((0.80, 'liquid'), (0.79, 'ambiguous')):
            with self.subTest(quality=quality):
                props = self.solid_props(
                    Tb=450.0,
                    property_sources={
                        'Tm': {'source': 'local', 'method': 'direct', 'quality': 0.98},
                        'Tb': {
                            'source': 'estimated',
                            'method': 'nannoolal_tb',
                            'quality': quality,
                        },
                    },
                )
                phase, _, _ = resolver._classify_density_phase(record, props)
                self.assertEqual(phase, expected)

    def test_temperatureless_reports_use_20c_only_as_penalized_evaluation_anchor(self):
        resolver = PropertyResolver()
        raw = resolver._parse_pubchem_density_text('Density: 0.79', {})[0]
        observation = resolver._density_observation_from_record(
            raw,
            self.liquid_props(),
            'example liquid',
        )
        self.assertIsNone(observation.temperature_K)
        point = resolver._density_observation_to_liquid_point(observation)
        self.assertEqual(point['T_K'], 293.15)
        self.assertTrue(point['temperature_assumed'])
        self.assertLess(point['quality'], observation.quality)

    def test_pubchem_source_cache_preserves_solid_and_liquid_records_once(self):
        resolver = PropertyResolver()
        fake_payload = {
            'Record': {
                'Section': [{
                    'TOCHeading': 'Density',
                    'Information': [
                        {'Value': {'StringWithMarkup': [{'String': '2.17 at 25 °C/4 °C'}]}},
                        {'Value': {'StringWithMarkup': [{
                            'String': 'Density of molten material at 850 °C: 1.549 g/cu cm',
                        }]}},
                        {'Value': {'StringWithMarkup': [{
                            'String': 'Density of aqueous solution: 1.2 g/cm3',
                        }]}},
                    ],
                }],
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
                with patch.object(resolver, '_get_pubchem_cid', return_value=5234) as cid:
                    with patch('urllib.request.urlopen', return_value=FakeResponse()) as request:
                        first = resolver._fetch_density_pubchem('sodium chloride')
                        second = resolver._fetch_density_pubchem('sodium chloride')

        self.assertEqual(first, second)
        self.assertEqual(cid.call_count, 1)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(first['raw_rows'], 3)
        self.assertEqual(first['parsed_records'], 2)
        self.assertEqual(len(first['records']), 2)
        self.assertEqual(
            {record['phase_hint'] for record in first['records']},
            {'ambiguous', 'liquid'},
        )

    def test_density_cache_key_has_specific_online_provenance(self):
        metadata = cache_key_metadata(
            'density_pubchem_v2_naphthalene',
            {'records': []},
        )
        self.assertEqual(metadata['cache_family'], 'density_pubchem')
        self.assertEqual(metadata['provider'], 'pubchem')
        self.assertEqual(metadata['contract_version'], 2)
        self.assertEqual(metadata['identifier_key'], 'naphthalene')

    def test_structured_route_classifies_and_filters_without_recaching_selection(self):
        resolver = PropertyResolver()
        source = {
            'cid': 931,
            'records': [
                resolver._parse_pubchem_density_text('1.162 at 20 °C/4 °C', {})[0],
                resolver._parse_pubchem_density_text(
                    'Molten liquid density 0.98 g/cm3 at 90 °C', {},
                )[0],
            ],
        }
        with patch.object(resolver, '_fetch_density_online', return_value=source):
            all_records = resolver.resolve_density_observations(
                'naphthalene', self.solid_props(),
            )
            solid_records = resolver.resolve_density_observations(
                'naphthalene', self.solid_props(), phase='solid',
            )
        self.assertEqual({item.phase for item in all_records}, {'solid', 'liquid'})
        self.assertEqual(len(solid_records), 1)
        self.assertEqual(solid_records[0].mass_density_kg_m3, 1162.0)

    def test_same_cached_source_record_reclassifies_when_phase_points_change(self):
        resolver = PropertyResolver()
        source = {
            'cid': 931,
            'records': [
                resolver._parse_pubchem_density_text('1.162 at 20 °C/4 °C', {})[0],
            ],
        }
        with patch.object(resolver, '_fetch_density_online', return_value=source):
            solid = resolver.resolve_density_observations(
                'compound', self.solid_props(),
            )[0]
            liquid = resolver.resolve_density_observations(
                'compound', self.liquid_props(),
            )[0]
        self.assertEqual(solid.phase, 'solid')
        self.assertEqual(liquid.phase, 'liquid')

    def test_offline_structured_route_does_not_probe_online_source(self):
        resolver = PropertyResolver()
        with patch.object(resolver, '_fetch_density_online') as fetch:
            result = resolver.resolve_density_observations(
                'compound', self.solid_props(), allow_online=False,
            )
        self.assertEqual(result, ())
        fetch.assert_not_called()

    def test_solid_only_data_never_enters_liquid_density_payload(self):
        resolver = PropertyResolver()
        source = {
            'cid': 931,
            'raw_rows': 1,
            'records': [
                resolver._parse_pubchem_density_text('1.162 at 20 °C/4 °C', {})[0],
            ],
        }
        with patch.object(resolver, '_fetch_density_online', return_value=source):
            liquid = resolver._fetch_liquid_density_online(
                'naphthalene', self.solid_props(),
            )
        self.assertIsNone(liquid)

    def test_solid_only_data_makes_liquid_resolution_use_ptv_not_pubchem_fit(self):
        resolver = PropertyResolver()
        props = self.solid_props(
            name='madeupium',
            CAS='999-99-9',
            formula='C7H12O2',
            smiles='CCCCCC(=O)O',
            Tc=650.0,
            Pc=35.0,
            omega=0.30,
            Zc=0.26,
            property_sources={
                **self.solid_props()['property_sources'],
                'Tc': {'source': 'local', 'method': 'direct', 'quality': 0.94},
                'Pc': {'source': 'local', 'method': 'direct', 'quality': 0.94},
                'omega': {'source': 'local', 'method': 'direct', 'quality': 0.94},
                'Zc': {'source': 'local', 'method': 'direct', 'quality': 0.94},
            },
        )
        source = {
            'cid': 931,
            'raw_rows': 1,
            'records': [
                resolver._parse_pubchem_density_text('1.162 at 20 °C/4 °C', {})[0],
            ],
        }
        with patch.object(resolver, '_fetch_density_online', return_value=source):
            result = resolver.resolve_liquid_molar_volume(
                'madeupium', 370.0, props,
            )
        self.assertNotIn('pubchem', result.method)
        self.assertIn('ptv', result.method)

    def test_bare_liquid_and_explicit_molten_data_reach_liquid_payload(self):
        resolver = PropertyResolver()
        source = {
            'cid': 1,
            'raw_rows': 2,
            'records': [
                resolver._parse_pubchem_density_text('Density: 0.79', {})[0],
                resolver._parse_pubchem_density_text(
                    'Molten liquid density 0.75 g/cm3 at 80 °C', {},
                )[0],
            ],
        }
        with patch.object(resolver, '_fetch_density_online', return_value=source):
            liquid = resolver._fetch_liquid_density_online(
                'example liquid', self.liquid_props(),
            )
        self.assertIsNotNone(liquid)
        self.assertEqual(len(liquid['points']), 2)
        self.assertTrue(any(point['temperature_assumed'] for point in liquid['points']))

    def test_solid_density_and_molar_routes_use_reported_cluster(self):
        resolver = PropertyResolver()
        observations = (
            self.observation(mass_density_kg_m3=1150.0, temperature_K=None),
            self.observation(mass_density_kg_m3=1162.0, temperature_K=293.15),
            self.observation(
                mass_density_kg_m3=1250.0,
                temperature_K=293.15,
                polymorph='beta',
                form_label='beta',
            ),
            self.observation(
                mass_density_kg_m3=1400.0,
                material_form='hydrate',
                form_label='hydrate',
                raw='hydrate density 1.4 g/cm3',
            ),
        )
        with patch.object(resolver, 'resolve_density_observations', return_value=observations):
            density = resolver.resolve_solid_mass_density(
                'naphthalene', 298.15, self.solid_props(),
            )
            volume = resolver.resolve_solid_molar_volume(
                'naphthalene', 298.15, self.solid_props(),
            )
            molar_density = resolver.resolve_solid_molar_density(
                'naphthalene', 298.15, self.solid_props(),
            )
        self.assertEqual(density.method, 'pubchem_solid_density_cluster')
        self.assertAlmostEqual(density.value, 1156.0 / (1.0 + 1.7e-4 * 5.0))
        self.assertIn('assumed 293.15 K', density.notes)
        self.assertIn('alpha_v=0.00017 1/K', density.notes)
        self.assertAlmostEqual(volume.value, 128.17 / density.value)
        self.assertAlmostEqual(molar_density.value, density.value / 128.17)

    def test_alternate_material_forms_and_polymorph_only_data_are_not_bare(self):
        resolver = PropertyResolver()
        hydrate = self.observation(
            material_form='hydrate',
            form_label='hydrate',
            raw='monohydrate solid density 1.4 g/cm3',
        )
        polymorphs = (
            self.observation(polymorph='alpha', form_label='alpha'),
            self.observation(mass_density_kg_m3=1180.0, polymorph='beta', form_label='beta'),
        )
        for observations in ((hydrate,), polymorphs):
            with self.subTest(observations=observations):
                with patch.object(
                    resolver, 'resolve_density_observations', return_value=observations,
                ):
                    with self.assertRaises(PropertyResolutionError):
                        resolver.resolve_solid_mass_density(
                            'parent', 298.15, self.solid_props(),
                        )

    def test_identity_hydrate_can_use_its_own_solid_density(self):
        resolver = PropertyResolver()
        hydrate = self.observation(
            mass_density_kg_m3=1400.0,
            material_form='hydrate',
            form_label='hydrate',
            is_identity_form=True,
            raw='monohydrate solid density 1.4 g/cm3',
        )
        with patch.object(
            resolver, 'resolve_density_observations', return_value=(hydrate,),
        ):
            result = resolver.resolve_solid_mass_density(
                'parent monohydrate', 298.15, self.solid_props(),
            )
        self.assertEqual(result.value, 1400.0)

    def test_identity_hydrate_does_not_fall_back_to_anhydrous_density(self):
        resolver = PropertyResolver()
        anhydrous = self.observation(
            material_form='anhydrous',
            form_label='anhydrous',
            raw='anhydrous solid density 1.16 g/cm3',
        )
        props = self.solid_props(name='parent monohydrate')
        with patch.object(
            resolver, 'resolve_density_observations', return_value=(anhydrous,),
        ):
            with self.assertRaises(PropertyResolutionError):
                resolver.resolve_solid_mass_density(
                    'parent monohydrate', 298.15, props,
                )

    def test_organic_fallback_domain_rejects_only_high_graph_symmetry_and_nonorganics(self):
        resolver = PropertyResolver()

        def check(smiles, expected, reason=''):
            structure = PropertyResolutionResult(
                value=smiles,
                source='local',
                method='fixture_smiles',
                quality=0.99,
            )
            with patch.object(resolver, '_resolve_smiles_result', return_value=structure):
                result, note = resolver._solid_density_heuristic_structure(
                    'fixture',
                    {'name': 'fixture', 'smiles': smiles},
                    allow_online=False,
                )
            self.assertEqual(result is not None, expected)
            if reason:
                self.assertIn(reason, note)

        cases = (
            ('CCCCO', True, '1 heavy-atom graph automorphism'),
            ('CCCCCC', True, '2 heavy-atom graph automorphism'),
            ('C=CCCCC', True, '1 heavy-atom graph automorphism'),
            ('CO', True, '1 heavy-atom graph automorphism'),
            ('CCO', True, '1 heavy-atom graph automorphism'),
            ('C1CCC1', True, '8 heavy-atom graph automorphism'),
            ('c1ccccc1', False, 'at least 10'),
            ('C1CCCCC1', False, 'at least 10'),
            ('C12C3C4C1C5C2C3C45', False, 'at least 10'),
            ('ClC(Cl)(Cl)Cl', False, 'at least 10'),
            ('O=S=O', False, 'not a molecular organic'),
            ('C[N+](C)(C)C', False, 'formally charged'),
            ('C[Mg]C', False, 'metal-containing'),
            ('CCO.CCO', False, 'multi-fragment'),
        )
        for smiles, expected, reason in cases:
            with self.subTest(smiles=smiles):
                check(smiles, expected, reason)

    def test_organic_fallback_prefers_ttriple_then_hard_tm(self):
        resolver = PropertyResolver()
        tt = PropertyResolutionResult(310.0, 'local', 'fixture_tt', 0.95)
        tm = PropertyResolutionResult(315.0, 'local', 'fixture_tm', 0.96)
        with patch.object(
            resolver,
            'resolve_triple_point',
            return_value={'Tt': tt},
        ), patch.object(
            resolver,
            'resolve_melting_point',
            return_value=tm,
        ) as melting:
            selected, kind = resolver._solid_density_transition_temperature(
                'fixture', {}, allow_online=False,
            )
        self.assertIs(selected, tt)
        self.assertEqual(kind, 'Tt')
        melting.assert_not_called()

        soft_tt = PropertyResolutionResult(310.0, 'estimated', 'fixture_tt', 0.99)
        with patch.object(
            resolver,
            'resolve_triple_point',
            return_value={'Tt': soft_tt},
        ), patch.object(
            resolver,
            'resolve_melting_point',
            return_value=tm,
        ):
            selected, kind = resolver._solid_density_transition_temperature(
                'fixture', {}, allow_online=False,
            )
        self.assertIs(selected, tm)
        self.assertEqual(kind, 'Tm')

        with patch.object(
            resolver,
            'resolve_triple_point',
            return_value={'Tt': tt},
        ), patch.object(
            resolver,
            'resolve_melting_point',
            return_value=tm,
        ):
            selected, kind = resolver._solid_density_transition_temperature(
                'fixture', {}, allow_online=False, requested_temperature=312.0,
            )
        self.assertIs(selected, tm)
        self.assertEqual(kind, 'Tm')

    def test_organic_fallback_uses_requested_temperature_directly(self):
        resolver = PropertyResolver()
        structure = PropertyResolutionResult(
            'CCCCO', 'local', 'fixture_smiles', 0.99,
        )
        transition = PropertyResolutionResult(
            300.0, 'local', 'fixture_tt', 0.95,
        )
        liquid_molar_density = PropertyResolutionResult(
            10.0, 'calculated', 'fixture_liquid_density', 0.80,
        )
        props = {'MW': 100.0, 'smiles': 'CCCCO'}
        with patch.object(
            resolver,
            '_solid_density_heuristic_structure',
            return_value=(structure, 'eligible fixture'),
        ), patch.object(
            resolver,
            '_solid_density_transition_temperature',
            return_value=(transition, 'Tt'),
        ), patch.object(
            resolver,
            'resolve_liquid_molar_density',
            return_value=liquid_molar_density,
        ):
            at_transition, reason = resolver._solid_density_organic_fallback(
                'fixture', 300.0, props, allow_online=False,
            )
            colder, _ = resolver._solid_density_organic_fallback(
                'fixture', 150.0, props, allow_online=False,
            )
        self.assertEqual(reason, '')
        self.assertAlmostEqual(at_transition.value, 1.12 * 1000.0)
        self.assertAlmostEqual(colder.value, 1.20 * 1000.0)
        self.assertAlmostEqual(at_transition.quality, 0.70 * 0.80)
        self.assertAlmostEqual(colder.quality, 0.70 * 0.80)
        self.assertEqual(
            at_transition.method,
            'organic_volume_of_fusion_solid_density',
        )
        self.assertIn('5.6%', at_transition.notes)
        self.assertIn('J. Chem. Eng. Data (2004) 49 (6): 1512–1514', at_transition.notes)
        self.assertIn('rho_s(T)=(1.28-0.16*T/Tt)', at_transition.notes)

    def test_intervening_transition_comes_from_solid_cp_kernel(self):
        resolver = PropertyResolver()

        class Kernel:
            transition_temperatures = (125.0, 225.0, 300.0)

        with patch.object(
            resolver,
            'resolve_solid_cp_kernel',
            return_value=Kernel(),
        ):
            transition = resolver._solid_density_intervening_transition(
                'fixture', 150.0, 300.0, {}, allow_online=False,
            )
            boundary_only = resolver._solid_density_intervening_transition(
                'fixture', 225.0, 300.0, {}, allow_online=False,
            )
        self.assertEqual(transition, 225.0)
        self.assertIsNone(boundary_only)

    def test_solid_expansion_policy_applies_only_to_neutral_organic_crystals(self):
        resolver = PropertyResolver()

        def policy(smiles, props=None):
            structure = PropertyResolutionResult(
                smiles, 'local', 'fixture_smiles', 0.99,
            )
            with patch.object(resolver, '_resolve_smiles_result', return_value=structure):
                return resolver._solid_density_expansion_policy(
                    'fixture',
                    {'smiles': smiles, **(props or {})},
                    allow_online=False,
                )

        for smiles in ('CCO', 'c1ccccc1', 'CCCCCC'):
            with self.subTest(smiles=smiles):
                alpha, penalty, note = policy(smiles)
                self.assertEqual(alpha, 1.7e-4)
                self.assertEqual(penalty, 0.01)
                self.assertIn('neutral molecular organic', note)

        for smiles in ('[Na+].[Cl-]', 'C[Mg]C', 'O=S=O', 'C[N+](C)(C)C'):
            with self.subTest(smiles=smiles):
                alpha, penalty, note = policy(smiles)
                self.assertEqual(alpha, 0.0)
                self.assertEqual(penalty, 0.02)
                self.assertIn('constant-density policy', note)

        alpha, penalty, note = policy('CCO', {'name': 'ethanol monohydrate'})
        self.assertEqual(alpha, 0.0)
        self.assertEqual(penalty, 0.02)
        self.assertIn('hydrate identity', note)

    def test_solid_expansion_quality_steps_round_down_at_complete_5k_intervals(self):
        resolver = PropertyResolver()
        self.assertEqual(resolver._solid_density_temperature_steps(345.15, 293.15), 10)
        self.assertEqual(resolver._solid_density_temperature_steps(298.14, 293.15), 0)
        self.assertEqual(resolver._solid_density_temperature_steps(298.15, 293.15), 1)

        organic = self.observation(temperature_K=293.15, quality=0.90)
        with patch.object(
            resolver,
            'resolve_density_observations',
            return_value=(organic,),
        ), patch.object(
            resolver,
            '_solid_density_expansion_policy',
            return_value=(1.7e-4, 0.01, 'organic fixture'),
        ):
            result = resolver.resolve_solid_mass_density(
                'fixture', 345.15, self.solid_props(),
            )
        self.assertAlmostEqual(
            result.value,
            1160.0 / (1.0 + 1.7e-4 * 52.0),
        )
        self.assertAlmostEqual(result.quality, 0.80)
        self.assertIn('quality penalty=0.1', result.notes)

        with patch.object(
            resolver,
            'resolve_density_observations',
            return_value=(organic,),
        ), patch.object(
            resolver,
            '_solid_density_expansion_policy',
            return_value=(0.0, 0.02, 'inorganic fixture'),
        ):
            inorganic = resolver.resolve_solid_mass_density(
                'fixture', 345.15, self.solid_props(),
            )
        self.assertEqual(inorganic.value, 1160.0)
        self.assertAlmostEqual(inorganic.quality, 0.70)
        self.assertIn('quality penalty=0.2', inorganic.notes)

    def test_organic_fallback_accepts_low_transition_and_enforces_reduced_temperature(self):
        resolver = PropertyResolver()
        structure = PropertyResolutionResult(
            'CCCCO', 'local', 'fixture_smiles', 0.99,
        )
        transition = PropertyResolutionResult(
            300.0, 'local', 'fixture_tm', 0.95,
        )
        props = {'MW': 100.0, 'smiles': 'CCCCO'}
        with patch.object(
            resolver,
            '_solid_density_heuristic_structure',
            return_value=(structure, 'eligible fixture'),
        ), patch.object(
            resolver,
            '_solid_density_transition_temperature',
            return_value=(transition, 'Tm'),
        ), patch.object(
            resolver,
            'resolve_liquid_molar_density',
            return_value=PropertyResolutionResult(
                10.0, 'calculated', 'fixture_liquid_density', 0.80,
            ),
        ):
            result, reason = resolver._solid_density_organic_fallback(
                'fixture', 250.0, props, allow_online=False,
            )
        self.assertIsNotNone(result)

        low_transition = PropertyResolutionResult(
            290.0, 'local', 'fixture_tm', 0.95,
        )
        with patch.object(
            resolver,
            '_solid_density_heuristic_structure',
            return_value=(structure, 'eligible fixture'),
        ), patch.object(
            resolver,
            '_solid_density_transition_temperature',
            return_value=(low_transition, 'Tm'),
        ), patch.object(
            resolver,
            'resolve_liquid_molar_density',
            return_value=PropertyResolutionResult(
                10.0, 'calculated', 'fixture_liquid_density', 0.80,
            ),
        ):
            result, reason = resolver._solid_density_organic_fallback(
                'fixture', 250.0, props, allow_online=False,
            )
        self.assertIsNotNone(result)
        self.assertEqual(reason, '')
        self.assertAlmostEqual(result.value, (1.28 - 0.16 * 250.0 / 290.0) * 1000.0)

        with patch.object(
            resolver,
            '_solid_density_heuristic_structure',
            return_value=(structure, 'eligible fixture'),
        ), patch.object(
            resolver,
            '_solid_density_transition_temperature',
            return_value=(low_transition, 'Tm'),
        ):
            result, reason = resolver._solid_density_organic_fallback(
                'fixture', 86.0, props, allow_online=False,
            )
        self.assertIsNone(result)
        self.assertIn('below 0.3*Tm=87 K', reason)

        weak_liquid = PropertyResolutionResult(
            10.0, 'estimated', 'weak_liquid_density', 0.54,
        )
        with patch.object(
            resolver,
            '_solid_density_heuristic_structure',
            return_value=(structure, 'eligible fixture'),
        ), patch.object(
            resolver,
            '_solid_density_transition_temperature',
            return_value=(transition, 'Tm'),
        ), patch.object(
            resolver,
            'resolve_liquid_molar_density',
            return_value=weak_liquid,
        ):
            result, reason = resolver._solid_density_organic_fallback(
                'fixture', 298.0, props, allow_online=False,
            )
        self.assertIsNone(result)
        self.assertIn('below quality 0.55', reason)

    def test_organic_fallback_rejects_intervening_solid_transition(self):
        resolver = PropertyResolver()
        structure = PropertyResolutionResult(
            'CCCCO', 'local', 'fixture_smiles', 0.99,
        )
        transition = PropertyResolutionResult(
            300.0, 'local', 'fixture_tm', 0.95,
        )
        with patch.object(
            resolver,
            '_solid_density_heuristic_structure',
            return_value=(structure, 'eligible fixture'),
        ), patch.object(
            resolver,
            '_solid_density_transition_temperature',
            return_value=(transition, 'Tm'),
        ), patch.object(
            resolver,
            '_solid_density_intervening_transition',
            return_value=225.0,
        ):
            result, reason = resolver._solid_density_organic_fallback(
                'fixture', 150.0, {'MW': 100.0}, allow_online=False,
            )
        self.assertIsNone(result)
        self.assertIn('solid transition at 225 K', reason)

    def test_reported_solid_density_remains_ahead_of_organic_fallback(self):
        resolver = PropertyResolver()
        reported = self.observation()
        with patch.object(
            resolver,
            'resolve_density_observations',
            return_value=(reported,),
        ), patch.object(
            resolver,
            '_solid_density_organic_fallback',
        ) as fallback:
            result = resolver.resolve_solid_mass_density(
                'naphthalene', 298.15, self.solid_props(),
            )
        self.assertEqual(result.method, 'pubchem_solid_density_cluster')
        fallback.assert_not_called()


if __name__ == '__main__':
    unittest.main()
