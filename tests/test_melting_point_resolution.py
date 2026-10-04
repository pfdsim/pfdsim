import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chemical_properties import ChemicalDatabase
from perry_properties import get_perry_property_library
from property_resolution.phase_point_candidates import (
    nist_temperature_candidate,
    parse_reported_temperature_candidates,
    select_temperature_consensus,
)
from property_resolution.resolver import PropertyResolver
from tests.live_provider import run_optional_live_provider
from tests.cache_isolation import empty_runtime_cache


class MeltingPointFormatTests(unittest.TestCase):
    def assertClose(self, actual, expected, *, rel=1e-8, abs_tol=1e-12):
        self.assertTrue(
            abs(actual - expected) <= max(abs_tol, rel * max(abs(actual), abs(expected))),
            f'{actual!r} != {expected!r}',
        )

    @staticmethod
    def pubchem_candidates(*reports):
        candidates = []
        for report in reports:
            candidates.extend(parse_reported_temperature_candidates(
                report,
                source='pubchem',
                method='pubchem_reported_melting',
            ))
        return candidates

    def test_range_parser_handles_all_observed_unit_layouts(self):
        cases = {
            '156 to 160 °F': (342.0388888888889, 344.26111111111106),
            '174-179 °C': (447.15, 452.15),
            '132 °C to 135 °C': (405.15, 408.15),
            '132.7-135 °C': (405.85, 408.15),
            '94.00 to 95.00 °C. @ 760.00 mm Hg': (367.15, 368.15),
        }
        for report, (low, high) in cases.items():
            with self.subTest(report=report):
                parsed = self.pubchem_candidates(report)
                self.assertEqual(len(parsed), 1)
                self.assertClose(parsed[0]['low_K'], low)
                self.assertClose(parsed[0]['high_K'], high)
                self.assertClose(parsed[0]['value_K'], 0.5 * (low + high))

    def test_negative_temperature_and_hyphenated_polymorphs_are_distinct(self):
        negative = self.pubchem_candidates('-37.3 °C')
        polymorphs = self.pubchem_candidates(
            'alpha-118.6 °F; beta-79 °F; gamma-117 °F'
        )

        self.assertClose(negative[0]['value_K'], 235.85)
        self.assertEqual(len(polymorphs), 3)
        self.assertEqual(
            [item['qualifiers']['polymorph'] for item in polymorphs],
            ['alpha', 'beta', 'gamma'],
        )
        self.assertClose(polymorphs[0]['value_K'], 321.26111111111106)
        self.assertClose(polymorphs[1]['value_K'], 299.26111111111106)
        self.assertClose(polymorphs[2]['value_K'], 320.3722222222222)

    def test_form_decomposition_and_irrelevant_sublimation_qualifiers(self):
        anhydrous = self.pubchem_candidates('307 °F (anhydrous)')
        decomposes = self.pubchem_candidates('153 °C (decomposes)')
        irrelevant = self.pubchem_candidates(
            'Latent heat at melting point; Decomposition temp = 135 °C'
        )
        sublimation = self.pubchem_candidates('Sublimes at 178 °C')

        self.assertTrue(anhydrous[0]['qualifiers']['anhydrous'])
        self.assertTrue(decomposes[0]['qualifiers']['decomposition'])
        self.assertEqual(irrelevant, [])
        self.assertEqual(sublimation, [])

    def test_pubchem_tree_filters_predictions_and_keeps_structured_numbers(self):
        resolver = PropertyResolver()
        root = {
            'TOCHeading': 'Melting Point',
            'Information': [
                {
                    'Name': 'QSPR predicted value',
                    'Value': {'StringWithMarkup': [{'String': '600 K'}]},
                },
                {
                    'ReferenceNumber': 41,
                    'Value': {'Number': [47.8], 'Unit': '°C'},
                },
                {
                    'ReferenceNumber': 57,
                    'Value': {'StringWithMarkup': [
                        {'String': 'Melting point: 47 °C (gamma-Benzophenone)'},
                    ]},
                },
            ],
        }
        result = {}

        resolver._extract_phase_change_from_pubchem_node(root, result)
        resolver._finalize_online_phase_point_candidates(result)

        candidates = result['_phase_candidates']['Tm']
        self.assertEqual(len(candidates), 2)
        self.assertEqual({item['reference'] for item in candidates}, {'41', '57'})
        self.assertClose(result['Tm'], 320.4, rel=2e-3)
        self.assertEqual(
            result['_sources']['Tm'],
            'pubchem_melting_point_consensus',
        )

    def test_hydrate_reports_do_not_merge_with_anhydrous_consensus(self):
        candidates = self.pubchem_candidates(
            '153 °C (anhydrous)',
            '154 °C',
            '100 °C (monohydrate)',
            '101 °C (monohydrate)',
        )
        selected = select_temperature_consensus('melting_point', candidates)
        self.assertClose(selected['value'], 426.65)
        self.assertTrue(all(
            not item['qualifiers']['hydrate']
            for item in selected['selected_candidates']
        ))

    def test_decomposition_only_consensus_has_capped_quality(self):
        candidates = self.pubchem_candidates(
            '153 °C (decomposes)',
            '154 °C (decomposes)',
            '153.5 °C (decomposes)',
        )
        selected = select_temperature_consensus('melting_point', candidates)
        self.assertLessEqual(selected['quality'], 0.78)

    def test_anisole_prefers_tight_pubchem_cluster_over_broad_nist_average(self):
        candidates = self.pubchem_candidates(
            '-37.3 °C', '-37.5 °C', '-37 °C', '-37.3 °C'
        )
        candidates.append(nist_temperature_candidate(
            '250. ± 40.',
            'K',
            method='AVG',
            reference='N/A',
            comment='Average of 9 values',
        ))

        selected = select_temperature_consensus('melting_point', candidates)

        self.assertEqual(selected['method'], 'pubchem_melting_point_consensus')
        self.assertClose(selected['value'], 235.875)
        self.assertGreaterEqual(selected['quality'], 0.97)
        self.assertEqual(len(selected['broad_candidates']), 1)
        self.assertIn('ignored 1 broad-uncertainty', selected['notes'])

        nist_only = select_temperature_consensus(
            'melting_point',
            [candidates[-1]],
        )
        self.assertLessEqual(nist_only['quality'], 0.65)

    def test_benzophenone_selects_alpha_gamma_consensus_and_rejects_beta(self):
        candidates = self.pubchem_candidates(
            'alpha-118.6 °F; beta-79 °F; gamma-117 °F',
            '47.8 °C',
            '48.5 °C',
            'Melting point: 47 °C (gamma-Benzophenone)',
            '26 °C',
            '48.5 °C',
            '119 °F',
            '49 °C',
        )
        candidates.append(nist_temperature_candidate(
            '321.2 ± 0.7',
            'K',
            method='AVG',
            reference='N/A',
            comment='Average of 23 out of 24 values',
        ))

        selected = select_temperature_consensus('melting_point', candidates)

        self.assertEqual(selected['method'], 'nist_pubchem_melting_point_consensus')
        self.assertClose(selected['value'], 321.2, rel=3e-3)
        self.assertGreaterEqual(selected['quality'], 0.96)
        outliers = [item['value_K'] for item in selected['outlier_candidates']]
        self.assertTrue(any(abs(value - 299.26) < 0.1 for value in outliers))

    def test_nist_parser_preserves_avg_uncertainty_count_and_triple_temperature(self):
        resolver = PropertyResolver()
        html = '''
            <table class="data" aria-label="One dimensional data">
              <tr><th>Quantity</th><th>Value</th><th>Units</th><th>Method</th><th>Reference</th><th>Comment</th></tr>
              <tr><td>T<sub>fus</sub></td><td>343. ± 1.</td><td>K</td><td>AVG</td><td>N/A</td><td>Average of 285 out of 294 values</td></tr>
              <tr><td>T<sub>triple</sub></td><td>342.090</td><td>K</td><td>N/A</td><td>TRC reference</td><td>Uncertainty assigned by TRC = 0.01 K</td></tr>
            </table>
        '''

        parsed = resolver._parse_nist_phase_change(html)

        self.assertClose(parsed['Tm'], 343.0)
        self.assertClose(parsed['Tt'], 342.09)
        candidate = parsed['_phase_candidates']['Tm'][0]
        self.assertClose(candidate['uncertainty_K'], 1.0)
        self.assertEqual(candidate['sample_count'], 285)
        self.assertEqual(candidate['reported_count'], 294)
        self.assertIn('TRC reference', json.dumps(parsed['_phase_candidates']['Tt']))
        self.assertClose(
            parsed['_phase_candidates']['Tt'][0]['uncertainty_K'],
            0.01,
        )
        self.assertClose(parsed['_qualities']['Tt'], 0.96)

    def test_nist_multiple_fusion_rows_are_order_independent(self):
        resolver = PropertyResolver()
        rows = [
            '<tr><td>T<sub>fus</sub></td><td>449.</td><td>K</td><td>N/A</td><td>Mjojo</td><td>Crystal phase 1; Uncertainty assigned by TRC = 1 K</td></tr>',
            '<tr><td>T<sub>fus</sub></td><td>451.5</td><td>K</td><td>N/A</td><td>Frandsen</td><td>Uncertainty assigned by TRC = 0.1 K</td></tr>',
        ]

        def payload(ordered):
            return '''
                <table class="data" aria-label="One dimensional data">
                  <tr><th>Quantity</th><th>Value</th><th>Units</th><th>Method</th><th>Reference</th><th>Comment</th></tr>
                  %s
                </table>
            ''' % ''.join(ordered)

        forward = resolver._parse_nist_phase_change(payload(rows))
        reverse = resolver._parse_nist_phase_change(payload(list(reversed(rows))))

        self.assertClose(forward['Tm'], reverse['Tm'])
        self.assertGreater(forward['Tm'], 451.3)
        self.assertEqual(len(forward['_phase_candidates']['Tm']), 2)
        self.assertEqual(
            forward['_sources']['Tm'],
            'nist_melting_point_consensus',
        )

    def test_cross_source_merge_assigns_consensus_quality_and_provenance(self):
        resolver = PropertyResolver()
        pubchem_candidates = self.pubchem_candidates('69 °C', '71 °C', '70 °C')
        nist_candidate = nist_temperature_candidate(
            '343. ± 1.', 'K', method='AVG', reference='N/A',
            comment='Average of 285 out of 294 values',
        )
        pubchem = {
            '_phase_candidates': {'Tm': pubchem_candidates},
            '_sources': {}, '_qualities': {}, '_notes': {},
        }
        nist = {
            '_phase_candidates': {'Tm': [nist_candidate]},
            '_sources': {}, '_qualities': {}, '_notes': {},
        }

        with patch.object(resolver, '_fetch_phase_change_pubchem', return_value=pubchem), \
             patch.object(resolver, '_fetch_phase_change_nist', return_value=nist):
            merged = resolver._fetch_phase_change_online('biphenyl', {})

        self.assertEqual(
            merged['_sources']['Tm'],
            'nist_pubchem_melting_point_consensus',
        )
        self.assertGreaterEqual(merged['_qualities']['Tm'], 0.98)
        self.assertClose(merged['Tm'], 343.15, rel=3e-3)
        self.assertEqual(len(merged['_phase_candidates']['Tm']), 4)

    def test_online_triple_temperature_and_pressure_reach_triple_resolver(self):
        resolver = PropertyResolver()
        online = {
            'Tt': 368.2,
            'Pt': 0.0012,
            '_sources': {
                'Tt': 'nist_pubchem_triple_temperature_consensus',
                'Pt': 'pubchem_triple_pressure_consensus',
            },
            '_qualities': {'Tt': 0.985, 'Pt': 0.94},
            '_notes': {'Tt': 'consistent reports', 'Pt': 'reported pressure'},
        }
        props = {
            'name': 'madeupium',
            'Tm': 368.0,
            'property_sources': {
                'Tm': {
                    'source': 'local', 'method': 'fixture_tm', 'quality': 0.98,
                },
            },
        }

        with patch.object(resolver, '_coolprop_phase_reference', return_value=None), \
             patch.object(resolver, '_fetch_phase_change_online', return_value=online):
            triple = resolver.resolve_triple_point(
                'madeupium', props, allow_online=True
            )

        self.assertClose(triple['Tt'].value, 368.2)
        self.assertEqual(
            triple['Tt'].method,
            'nist_pubchem_triple_temperature_consensus',
        )
        self.assertClose(triple['Tt'].quality, 0.985)
        self.assertClose(triple['Pt'].value, 0.0012)

    def test_pubchem_triple_section_collects_temperature_and_pressure(self):
        resolver = PropertyResolver()
        root = {
            'TOCHeading': 'Triple Point',
            'Information': [
                {
                    'ReferenceNumber': 12,
                    'Value': {'StringWithMarkup': [
                        {'String': '342.10 K at 0.0050 bar'},
                    ]},
                },
                {
                    'Name': 'predicted model value',
                    'Value': {'StringWithMarkup': [
                        {'String': '400 K at 1 bar'},
                    ]},
                },
            ],
        }
        result = {}

        resolver._extract_phase_change_from_pubchem_node(root, result)
        resolver._finalize_online_phase_point_candidates(result)

        self.assertClose(result['Tt'], 342.10)
        self.assertClose(result['Pt'], 0.0050)
        self.assertEqual(len(result['_phase_candidates']['Tt']), 1)
        self.assertEqual(len(result['_phase_candidates']['Pt']), 1)


@empty_runtime_cache
class MeltingPointResolutionTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def assertClose(self, actual, expected, *, rel=1e-8, abs_tol=1e-12):
        self.assertTrue(
            abs(actual - expected) <= max(abs_tol, rel * max(abs(actual), abs(expected))),
            f'{actual!r} != {expected!r}',
        )

    def test_direct_coolprop_and_perry_melting_resolution(self):
        resolver = PropertyResolver()
        direct = resolver.resolve_melting_point(
            'ethanol', {'Tm': 123.4}, allow_online=False
        )
        ethanol = resolver.resolve_melting_point('ethanol', allow_online=False)
        benzaldehyde = resolver.resolve_melting_point(
            'benzaldehyde', allow_online=False
        )

        self.assertEqual(direct.source, 'provided')
        self.assertEqual(ethanol.method, 'coolprop_HEOS_melting_point')
        self.assertClose(ethanol.value, 158.38389460326692)
        self.assertClose(ethanol.quality, 0.995)
        self.assertEqual(
            benzaldehyde.method,
            'perry_table_2_10_normal_melting_point',
        )
        self.assertClose(benzaldehyde.value, 247.15)

    def test_curated_tm_overrides_precede_perry_without_coolprop_fusion_line(self):
        resolver = PropertyResolver()
        library = get_perry_property_library()
        expected = {
            '2,2,3,3-tetramethylbutane': 373.8,
            'ethyl acetate': 189.6,
            '1,2,3,5-tetramethylbenzene': 249.5,
            '2-butyne': 240.7,
            'butyric acid': 268.0,
            '1-butene': 87.8,
            '3-methyl-3-ethylpentane': 182.5,
            'dichloroacetic acid': 282.9,
            '1-butanol': 183.3,
        }
        for identifier, value in expected.items():
            result = resolver.resolve_melting_point(identifier, allow_online=False)
            self.assertEqual(result.method, 'curated_tm_override', identifier)
            self.assertClose(result.quality, 0.98)
            self.assertClose(result.value, value)
            self.assertTrue(
                library.normal_melting_point_K(identifier)
                or library.table_2_10_normal_melting_point_K(identifier)
            )

        isobutane = resolver.resolve_melting_point('isobutane', allow_online=False)
        self.assertEqual(isobutane.method, 'coolprop_HEOS_melting_point')
        self.assertClose(isobutane.value, 113.77396654097308)

    def test_curated_tm_overrides_are_declared_in_chemicals_json(self):
        payload = json.loads((self.ROOT / 'data' / 'chemicals.json').read_text())
        expected = {
            '594-82-1': ('2,2,3,3-Tetramethylbutane', 373.8),
            'C4H8O2': ('Ethyl Acetate', 189.6),
            '527-53-7': ('1,2,3,5-Tetramethylbenzene', 249.5),
            '503-17-3': ('2-Butyne', 240.7),
            '107-92-6': ('Butyric acid', 268.0),
            '106-98-9': ('1-Butene', 87.8),
            '1067-08-9': ('3-Methyl-3-ethylpentane', 182.5),
            'iC4H10': ('Isobutane', 113.73),
            '79-43-6': ('Dichloroacetic acid', 282.9),
            'C4H9OH': ('1-Butanol', 183.3),
        }
        for key, (name, tm) in expected.items():
            entry = payload['chemicals'][key]
            source = entry.get('property_sources', {}).get('Tm', {})
            self.assertEqual(entry['name'], name)
            self.assertClose(entry['Tm'], tm)
            self.assertEqual(source.get('method'), 'curated_tm_override')
            self.assertClose(source.get('quality'), 0.98)

    def test_coolprop_precedes_consistent_non_pfd_value(self):
        resolver = PropertyResolver()
        result = resolver.resolve_melting_point(
            'ethanol',
            {
                'CAS': '64-17-5', 'name': 'ethanol', 'Tm': 160.0,
                'property_sources': {
                    'Tm': {
                        'source': 'local', 'method': 'fixture_tm', 'quality': 0.98,
                    },
                },
            },
            allow_online=False,
        )
        self.assertEqual(result.method, 'coolprop_HEOS_melting_point')
        self.assertClose(result.value, 158.38389460326692)

    def test_inconsistent_coolprop_curve_falls_back_to_hard_tm(self):
        resolver = PropertyResolver()
        props = {
            'CAS': '1333-74-0', 'name': 'hydrogen', 'Tm': 13.99,
            'property_sources': {
                'Tm': {
                    'source': 'local',
                    'method': 'experimental_hydrogen_tm',
                    'quality': 0.98,
                },
            },
        }
        melting = resolver.resolve_melting_point('hydrogen', props, allow_online=False)
        triple = resolver.resolve_triple_point('hydrogen', props, allow_online=False)
        self.assertEqual(melting.value, 13.99)
        self.assertEqual(melting.method, 'experimental_hydrogen_tm')
        self.assertIn(
            'outside the declared domain 236.062-239143 bar',
            melting.notes,
        )
        self.assertIn('extrapolation rejected', melting.notes)
        self.assertEqual(triple['Tt'].method, 'coolprop_HEOS_triple_point')

    def test_co2_fusion_uses_triple_pressure(self):
        result = PropertyResolver().resolve_melting_point(
            'carbon dioxide',
            {'CAS': '124-38-9', 'name': 'carbon dioxide'},
            allow_online=False,
        )
        self.assertEqual(
            result.method,
            'coolprop_HEOS_fusion_point_at_triple_pressure',
        )
        self.assertClose(result.value, 216.59200306719677)
        self.assertIn('no stable fusion equilibrium exists at 1 atm', result.notes)

    def test_out_of_domain_coolprop_fusion_uses_fallback_or_missing(self):
        resolver = PropertyResolver()
        carbon_monoxide = resolver.resolve_melting_point(
            'carbon monoxide',
            {'CAS': '630-08-0', 'name': 'carbon monoxide'},
            allow_online=False,
        )
        self.assertEqual(
            carbon_monoxide.method,
            'perry_table_2_10_normal_melting_point',
        )
        self.assertIn('outside the declared domain', carbon_monoxide.notes)
        self.assertIn('extrapolation rejected', carbon_monoxide.notes)

        helium = resolver.resolve_melting_point(
            'helium',
            {'CAS': '7440-59-7', 'name': 'helium'},
            allow_online=False,
        )
        self.assertIsNone(helium.value)
        self.assertEqual(
            helium.method,
            'coolprop_melting_point_outside_validity',
        )
        self.assertIn('22.1436-455494 bar', helium.notes)

    def test_in_domain_uncorroborated_tm_and_internal_triple_closure(self):
        resolver = PropertyResolver()
        props = {'CAS': '7782-41-4', 'name': 'fluorine'}
        melting = resolver.resolve_melting_point(
            'fluorine', props, allow_online=False
        )
        triple = resolver.resolve_triple_point(
            'fluorine', props, allow_online=False
        )
        self.assertEqual(melting.method, 'coolprop_HEOS_melting_point')
        self.assertClose(melting.quality, 0.98)
        self.assertIn('declared pressure domain', melting.notes)
        self.assertClose(triple['Tt'].quality, 0.995)
        self.assertClose(triple['Pt'].quality, 0.995)
        self.assertIn('solid-liquid/VLE pressure closure', triple['Tt'].notes)

    def test_unverified_coolprop_triple_pair_is_provisional(self):
        resolver = PropertyResolver()
        props = {'CAS': '7440-59-7', 'name': 'helium'}
        melting = resolver.resolve_melting_point(
            'helium', props, allow_online=False
        )
        triple = resolver.resolve_triple_point(
            'helium', props, allow_online=False
        )
        self.assertIsNone(melting.value)
        self.assertClose(triple['Tt'].value, 2.1768)
        self.assertClose(triple['Pt'].value, 0.05039330380576782)
        self.assertClose(triple['Tt'].quality, 0.89)
        self.assertClose(triple['Pt'].quality, 0.89)
        self.assertIn('provisional', triple['Tt'].notes)

    def test_provisional_coolprop_topology_rejects_only_its_own_tsub_as_tb(self):
        props = {'CAS': '2551-62-4', 'name': 'sulfur hexafluoride', 'formula': 'F6S'}
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = (
                Path(directory) / 'saturation.sqlite'
            )
            triple = resolver.resolve_triple_point(
                'sulfur hexafluoride',
                props,
                allow_online=False,
            )
            props['Tt'] = triple['Tt'].value
            props['Pt'] = triple['Pt'].value
            props['property_sources'] = {
                key: {
                    'source': item.source,
                    'method': item.method,
                    'quality': item.quality,
                    'notes': item.notes,
                }
                for key, item in triple.items()
            }
            boiling = resolver.resolve_boiling_point(
                'sulfur hexafluoride',
                props,
                allow_online=False,
                allow_estimation=True,
            )
        self.assertEqual(triple['Pt'].quality, 0.89)
        self.assertIsNone(boiling.value)
        self.assertEqual(boiling.method, 'no_normal_boiling_point_at_1atm')
        self.assertIn('triple topology quality=0.89', boiling.notes)

    def test_online_agreement_confirms_and_keeps_coolprop_pair(self):
        props = {'CAS': '7440-59-7', 'name': 'helium'}
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = (
                Path(directory) / 'saturation.sqlite'
            )
            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                return_value={
                    'Tt': 2.18,
                    'Pt': 0.06,
                    '_sources': {'Tt': 'online_fixture', 'Pt': 'online_fixture'},
                    '_qualities': {'Tt': 0.97, 'Pt': 0.97},
                },
            ):
                triple = resolver.resolve_triple_point(
                    'helium', props, allow_online=True
                )
        self.assertClose(triple['Tt'].value, 2.1768)
        self.assertClose(triple['Pt'].value, 0.05039330380576782)
        self.assertEqual(triple['Tt'].method, 'coolprop_HEOS_triple_point')
        self.assertClose(triple['Tt'].quality, 0.995)
        self.assertIn('corroborating online/online_fixture', triple['Tt'].notes)

    def test_online_tm_alone_can_confirm_provisional_coolprop_pair(self):
        props = {'CAS': '7440-59-7', 'name': 'helium'}
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = (
                Path(directory) / 'saturation.sqlite'
            )
            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                return_value={
                    'Tm': 2.18,
                    '_sources': {'Tm': 'online_tm_fixture'},
                    '_qualities': {'Tm': 0.96},
                },
            ):
                triple = resolver.resolve_triple_point(
                    'helium', props, allow_online=True
                )
        self.assertClose(triple['Tt'].value, 2.1768)
        self.assertClose(triple['Pt'].value, 0.05039330380576782)
        self.assertClose(triple['Tt'].quality, 0.995)
        self.assertIn('online/online_tm_fixture Tm=2.18 K', triple['Tt'].notes)

    def test_conflicting_online_tm_rejects_unverified_coolprop_pair(self):
        props = {'CAS': '7440-59-7', 'name': 'helium'}
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = (
                Path(directory) / 'saturation.sqlite'
            )
            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                return_value={
                    'Tm': 3.0,
                    '_sources': {'Tm': 'online_tm_fixture'},
                    '_qualities': {'Tm': 0.96},
                },
            ):
                triple = resolver.resolve_triple_point(
                    'helium', props, allow_online=True
                )
        for result in triple.values():
            self.assertIsNone(result.value)
            self.assertEqual(
                result.method,
                'coolprop_triple_point_inconsistent_with_melting_point',
            )
            self.assertIn('hard Tm=3 K', result.notes)

    def test_online_tm_breaks_conflicting_ttriple_tie(self):
        props = {'CAS': '7440-59-7', 'name': 'helium'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saturation.sqlite'
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = path
            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                return_value={
                    'Tm': 2.18,
                    'Tt': 3.0,
                    'Pt': 0.08,
                    '_sources': {
                        'Tm': 'online_tm_fixture',
                        'Tt': 'online_tt_fixture',
                        'Pt': 'online_tt_fixture',
                    },
                    '_qualities': {'Tm': 0.96, 'Tt': 0.96, 'Pt': 0.94},
                },
            ):
                coolprop_supported = resolver.resolve_triple_point(
                    'helium', props, allow_online=True
                )
            self.assertClose(coolprop_supported['Tt'].value, 2.1768)
            self.assertEqual(
                coolprop_supported['Tt'].method,
                'coolprop_HEOS_triple_point',
            )
            self.assertClose(coolprop_supported['Tt'].quality, 0.995)

            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = Path(
                directory,
                'other.sqlite',
            )
            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                return_value={
                    'Tm': 3.01,
                    'Tt': 3.0,
                    'Pt': 0.08,
                    '_sources': {
                        'Tm': 'online_tm_fixture',
                        'Tt': 'online_tt_fixture',
                        'Pt': 'online_tt_fixture',
                    },
                    '_qualities': {'Tm': 0.96, 'Tt': 0.96, 'Pt': 0.94},
                },
            ):
                online_supported = resolver.resolve_triple_point(
                    'helium', props, allow_online=True
                )
            self.assertClose(online_supported['Tt'].value, 3.0)
            self.assertEqual(online_supported['Tt'].method, 'online_tt_fixture')

    def test_online_conflict_replaces_entire_provisional_coolprop_pair(self):
        props = {'CAS': '7440-59-7', 'name': 'helium'}
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = (
                Path(directory) / 'saturation.sqlite'
            )
            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                return_value={
                    'Tt': 3.0,
                    'Pt': 0.08,
                    '_sources': {'Tt': 'online_fixture', 'Pt': 'online_fixture'},
                    '_qualities': {'Tt': 0.96, 'Pt': 0.94},
                },
            ):
                triple = resolver.resolve_triple_point(
                    'helium', props, allow_online=True
                )
        self.assertClose(triple['Tt'].value, 3.0)
        self.assertClose(triple['Pt'].value, 0.08)
        self.assertEqual(triple['Tt'].method, 'online_fixture')
        self.assertEqual(triple['Pt'].method, 'online_fixture')
        self.assertIn('external result selected', triple['Tt'].notes)

    def test_partial_online_conflict_never_keeps_coolprop_pressure(self):
        props = {'CAS': '7440-59-7', 'name': 'helium'}
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.SATURATION_PROPERTIES_CACHE_PATH = (
                Path(directory) / 'saturation.sqlite'
            )
            with patch.object(
                resolver,
                '_fetch_phase_change_online',
                return_value={
                    'Tt': 3.0,
                    '_sources': {'Tt': 'online_fixture'},
                    '_qualities': {'Tt': 0.96},
                },
            ):
                triple = resolver.resolve_triple_point(
                    'helium', props, allow_online=True
                )
        self.assertClose(triple['Tt'].value, 3.0)
        self.assertEqual(triple['Tt'].method, 'online_fixture')
        self.assertIsNone(triple['Pt'].value)
        self.assertEqual(triple['Pt'].method, 'none')
        self.assertIn('external result selected', triple['Pt'].notes)

    def test_independent_tm_confirms_coolprop_without_online_probe(self):
        resolver = PropertyResolver()
        props = {
            'CAS': '1333-74-0',
            'name': 'hydrogen',
            'Tm': 13.99,
            'property_sources': {
                'Tm': {
                    'source': 'local',
                    'method': 'experimental_hydrogen_tm',
                    'quality': 0.98,
                },
            },
        }
        with patch.object(
            resolver,
            '_fetch_phase_change_online',
            side_effect=AssertionError('confirmed CoolProp pair probed online'),
        ):
            triple = resolver.resolve_triple_point(
                'hydrogen', props, allow_online=True
            )
        self.assertEqual(triple['Tt'].method, 'coolprop_HEOS_triple_point')
        self.assertClose(triple['Tt'].quality, 0.995)
        self.assertIn('experimental_hydrogen_tm', triple['Tt'].notes)

    def test_hard_tm_rejects_coolprop_and_selects_supported_online_pair(self):
        resolver = PropertyResolver()
        props = {
            'CAS': '74-99-7',
            'name': 'methyl acetylene',
            'Tm': 170.45,
            'property_sources': {
                'Tm': {
                    'source': 'local',
                    'method': 'perry_heat_of_fusion_melting_point',
                    'quality': 0.95,
                },
            },
        }
        with patch.object(
            resolver,
            '_fetch_phase_change_online',
            return_value={
                'Tt': 170.5,
                'Pt': 0.01,
                '_sources': {'Tt': 'online_fixture', 'Pt': 'online_fixture'},
                '_qualities': {'Tt': 0.96, 'Pt': 0.94},
            },
        ):
            triple = resolver.resolve_triple_point(
                'methyl acetylene', props, allow_online=True
            )
        self.assertClose(triple['Tt'].value, 170.5)
        self.assertClose(triple['Pt'].value, 0.01)
        self.assertEqual(triple['Tt'].method, 'online_fixture')
        self.assertIn('CoolProp Tt=273 K differs', triple['Tt'].notes)

    def test_live_oxygen_tm_corroborates_coolprop_triple_pair(self):
        props = {'CAS': '7782-44-7', 'name': 'oxygen'}
        with tempfile.TemporaryDirectory() as directory:
            resolver = PropertyResolver()
            resolver.CACHE_DIR = Path(directory, 'source_cache')
            resolver.SATURATION_PROPERTIES_CACHE_PATH = Path(
                directory,
                'saturation.sqlite',
            )
            live = run_optional_live_provider(
                resolver,
                lambda: resolver.resolve_triple_point(
                    'oxygen',
                    props,
                    allow_online=True,
                ),
                label='live oxygen fusion corroboration',
            )
            if not live.completed:
                return
            triple = live.value
        self.assertEqual(triple['Tt'].method, 'coolprop_HEOS_triple_point')
        self.assertClose(triple['Tt'].value, 54.361)
        self.assertClose(triple['Tt'].quality, 0.995)
        self.assertIn('confirmed by corroborating online/', triple['Tt'].notes)

    def test_coolprop_triple_placeholder_is_rejected_against_hard_tm(self):
        resolver = PropertyResolver()
        props = {
            'CAS': '74-99-7', 'name': 'methyl acetylene', 'Tm': 170.45,
            'property_sources': {
                'Tm': {
                    'source': 'local',
                    'method': 'perry_heat_of_fusion_melting_point',
                    'quality': 0.95,
                },
            },
        }
        triple = resolver.resolve_triple_point(
            'methyl acetylene', props, allow_online=False
        )
        for result in triple.values():
            self.assertIsNone(result.value)
            self.assertEqual(
                result.method,
                'coolprop_triple_point_inconsistent_with_melting_point',
            )

    def test_hydration_clears_sublimator_tb_and_uses_tm_before_triple(self):
        database = ChemicalDatabase(enable_online=False)
        co2 = database.get('carbon dioxide', fetch_online=False)
        naphthalene = database.get('naphthalene', fetch_online=False)
        propyne = database.get('methyl acetylene', fetch_online=False)
        cyclopropane = database.get('cyclopropane', fetch_online=False)

        self.assertIsNone(co2.Tb)
        self.assertEqual(co2.phase_at_STP, 'gas')
        self.assertIsNotNone(naphthalene)
        self.assertEqual(naphthalene.phase_at_STP, 'solid')
        self.assertIsNotNone(propyne.Tb)
        self.assertIsNone(propyne.Tt)
        self.assertIsNotNone(cyclopropane.Tb)
        self.assertIsNone(cyclopropane.Tt)


if __name__ == '__main__':
    unittest.main()
