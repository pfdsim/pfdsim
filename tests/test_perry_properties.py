import json
import math
import os
import sys
import unittest
from pathlib import Path


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class PerryPropertyExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(Path(ROOT, 'data', 'perry_properties.json').read_text())
        cls.chemicals = cls.payload['chemicals']
        cls.fusion_payload = json.loads(Path(ROOT, 'data', 'perry_heat_of_fusion.json').read_text())
        cls.fusion_chemicals = cls.fusion_payload['chemicals']

    def assertClose(self, actual, expected, *, rel=1e-9, abs_tol=1e-12):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol),
            f'{actual!r} != {expected!r}',
        )

    def test_table_counts_and_metadata_match_expected_extraction(self):
        tables = self.payload['metadata']['tables']

        self.assertEqual(len(self.chemicals), 345)
        self.assertEqual(tables['vapor_pressure']['rows_extracted'], 345)
        self.assertEqual(tables['critical_constants']['rows_extracted'], 345)
        self.assertEqual(tables['liquid_density']['rows_extracted'], 347)
        self.assertEqual(tables['liquid_heat_capacity']['rows_extracted'], 347)
        self.assertEqual(tables['ideal_gas_heat_capacity_polynomial']['rows_extracted'], 61)
        self.assertEqual(tables['ideal_gas_heat_capacity_hyperbolic']['rows_extracted'], 341)
        self.assertEqual(tables['vapor_viscosity']['rows_extracted'], 345)
        self.assertEqual(tables['liquid_viscosity']['rows_extracted'], 340)
        self.assertEqual(tables['formation_properties']['rows_extracted'], 338)
        self.assertEqual(tables['heat_of_vaporization']['rows_extracted'], 345)
        self.assertNotIn('heat_of_fusion', tables)
        self.assertEqual(tables['liquid_density']['units']['density'], 'mol/dm^3')
        self.assertEqual(tables['liquid_heat_capacity']['units']['heat_capacity'], 'J/(kmol*K)')
        self.assertTrue(all('heat_of_fusion' not in entry for entry in self.chemicals.values()))

    def test_main_perry_tables_are_cas_keyed_and_row_counts_are_conserved(self):
        tables = self.payload['metadata']['tables']
        disallowed_casless_tables = {'2-10', '2-68'}
        self.assertTrue(disallowed_casless_tables.isdisjoint(
            {metadata['table'] for metadata in tables.values()}
        ))

        for key, metadata in tables.items():
            actual_rows = 0
            for entry in self.chemicals.values():
                if key not in entry:
                    continue
                value = entry[key]
                actual_rows += len(value) if isinstance(value, list) else 1
            self.assertEqual(
                actual_rows,
                metadata['rows_extracted'],
                f'{key} rows were parsed but not retained',
            )

    def test_water_vapor_pressure_critical_density_and_cp_are_extracted(self):
        water = self.chemicals['7732-18-5']
        self.assertEqual(water['name'], 'Water')
        self.assertEqual(water['formula'], 'H2O')

        vp = water['vapor_pressure'][0]
        self.assertEqual(vp['source_table'], '2-8')
        self.assertEqual(vp['coefficients'], [73.649, -7258.2, -7.3037, 4.1653e-06, 2.0])
        self.assertClose(vp['T_min_K'], 273.16)
        self.assertClose(vp['P_at_T_min_Pa'], 611.0)
        self.assertClose(vp['T_max_K'], 647.1)
        self.assertClose(vp['P_at_T_max_Pa'], 21930000.0)
        calculated = math.exp(
            vp['coefficients'][0]
            + vp['coefficients'][1] / vp['T_min_K']
            + vp['coefficients'][2] * math.log(vp['T_min_K'])
            + vp['coefficients'][3] * vp['T_min_K'] ** vp['coefficients'][4]
        )
        self.assertAlmostEqual(calculated, vp['P_at_T_min_Pa'], delta=2.0)

        critical = water['critical_constants']
        self.assertClose(critical['Tc_K'], 647.096)
        self.assertClose(critical['Pc_MPa'], 22.064)
        self.assertClose(critical['Vc_m3_per_kmol'], 0.0559472)
        self.assertClose(critical['omega'], 0.344861)

        densities = water['liquid_density']
        self.assertEqual([row['equation_id'] for row in densities], [100, 119])
        self.assertClose(densities[0]['value_at_T_min'], 55.497)
        self.assertClose(densities[0]['molar_volume_at_T_min_dm3_per_mol'], 1.0 / 55.497)
        self.assertClose(densities[1]['value_at_T_max'], 17.874)

        liquid_cp = water['liquid_heat_capacity'][0]
        self.assertEqual(liquid_cp['equation_id'], 100)
        self.assertClose(liquid_cp['printed_value_at_T_min'], 0.7615)
        self.assertClose(liquid_cp['value_at_T_min'], 76150.0)
        self.assertClose(liquid_cp['value_at_T_max'], 89394.0)

    def test_acetamide_split_decimal_is_corrected_by_the_extractor(self):
        from scripts.extract_perry_properties import TABLES, parse_property

        vapor_spec = next(spec for spec in TABLES if spec.key == 'vapor_pressure')
        identity = {'cas': '60-35-5', 'compound_number': 2}
        record = parse_property(
            vapor_spec,
            identity,
            [
                125.8,
                1.0,
                -12376.0,
                -14.589,
                5.0824e-06,
                2.0,
                353.33,
                336.0,
                761.0,
                6569000.0,
            ],
        )

        expected = [125.81, -12376.0, -14.589, 5.0824e-06, 2.0]
        self.assertEqual(record['coefficients'], expected)
        self.assertEqual(
            self.chemicals['60-35-5']['vapor_pressure'][0]['coefficients'],
            expected,
        )

    def test_fusion_identity_matching_is_elemental_and_not_substring_based(self):
        from scripts.extract_perry_properties import names_compatible, normalize_formula

        self.assertEqual(normalize_formula('C6H5NH2'), normalize_formula('C6H7N'))
        self.assertFalse(names_compatible('propylbenzene', 'isopropylbenzene'))
        self.assertFalse(names_compatible('durene', 'isodurene'))
        self.assertFalse(names_compatible('3-methylpentane', '3-ethyl-3-methylpentane'))
        self.assertTrue(names_compatible('Methylbenzene (Toluene)', 'Toluene (Methylbenzene)'))

    def test_ethanol_preserves_low_and_high_temperature_gas_cp_and_formation_values(self):
        ethanol = self.chemicals['64-17-5']
        self.assertEqual(ethanol['name'], 'Ethanol')
        self.assertEqual(ethanol['compound_numbers'], [126])

        low_cp = ethanol['ideal_gas_heat_capacity_polynomial'][0]
        self.assertEqual(low_cp['source_table'], '2-74')
        self.assertEqual(low_cp['coefficients'], [32585.0, 87.4, 0.05])
        self.assertClose(low_cp['T_min_K'], 50.0)
        self.assertClose(low_cp['value_at_T_max'], 52070.0)

        high_cp = ethanol['ideal_gas_heat_capacity_hyperbolic'][0]
        self.assertEqual(high_cp['source_table'], '2-75')
        self.assertEqual(high_cp['printed_coefficients'], [0.492, 1.4577, 1.6628, 0.939, 744.7])
        self.assertEqual(high_cp['coefficient_multipliers'], [100000.0, 100000.0, 1000.0, 100000.0, 1.0])
        self.assertEqual(high_cp['coefficients'], [49200.0, 145770.0, 1662.8, 93900.0, 744.7])
        self.assertClose(high_cp['value_at_T_min'], 61172.0)
        self.assertClose(high_cp['value_at_T_max'], 165760.0)

        formation = ethanol['formation_properties']
        self.assertClose(formation['Hf_ideal_gas_J_per_kmol'], -234950000.0)
        self.assertClose(formation['Gf_ideal_gas_J_per_kmol'], -167850000.0)
        self.assertClose(formation['S_ideal_gas_J_per_kmol_K'], 280640.0)
        self.assertClose(formation['net_Hcomb_J_per_kmol'], -1235000000.0)

        self.assertNotIn('normal_points', ethanol)

        hvap = ethanol['heat_of_vaporization'][0]
        self.assertEqual(hvap['source_table'], '2-69')
        self.assertEqual(hvap['coefficient_multipliers'], [10000000.0, 1.0, 1.0, 1.0])
        self.assertClose(hvap['coefficients'][0], 65831000.0)

    def test_acids_and_ester_keep_transport_and_critical_data(self):
        acetic = self.chemicals['64-19-7']
        self.assertClose(acetic['liquid_heat_capacity'][0]['coefficients'][0], 139640.0)
        self.assertClose(acetic['liquid_heat_capacity'][0]['value_at_T_min'], 122130.0)
        self.assertClose(acetic['liquid_viscosity'][0]['value_at_T_min'], 0.001265)
        self.assertClose(acetic['formation_properties']['Hf_ideal_gas_J_per_kmol'], -432800000.0)
        self.assertNotIn('normal_points', acetic)

        acrylic = self.chemicals['79-10-7']
        self.assertClose(acrylic['critical_constants']['Tc_K'], 615.0)
        self.assertClose(acrylic['critical_constants']['omega'], 0.538324)
        self.assertClose(acrylic['vapor_viscosity'][0]['value_at_T_min'], 7.679e-06)
        self.assertClose(acrylic['liquid_viscosity'][0]['value_at_T_max'], 0.0002086)

        ethyl_acetate = self.chemicals['141-78-6']
        self.assertClose(ethyl_acetate['critical_constants']['Pc_MPa'], 3.88)
        self.assertClose(ethyl_acetate['liquid_density'][0]['molar_volume_at_T_max_dm3_per_mol'], 1.0 / 3.4793)
        self.assertClose(ethyl_acetate['liquid_heat_capacity'][0]['value_at_T_max'], 187960.0)
        self.assertClose(ethyl_acetate['liquid_viscosity'][0]['value_at_T_min'], 0.001132)

    def test_duplicate_correlations_are_preserved_for_benzene(self):
        benzene = self.chemicals['71-43-2']
        liquid_cp = benzene['liquid_heat_capacity']

        self.assertEqual(len(liquid_cp), 2)
        self.assertEqual([row['source_table'] for row in liquid_cp], ['2-72', '2-72'])
        self.assertEqual([row['T_max_K'] for row in liquid_cp], [353.24, 500.0])
        self.assertEqual([row['value_at_T_max'] for row in liquid_cp], [150400.0, 204380.0])

    def test_missing_rows_remain_missing_instead_of_fabricated(self):
        water = self.chemicals['7732-18-5']
        self.assertNotIn('formation_properties', water)

        argon = self.chemicals['7440-37-1']
        self.assertNotIn('ideal_gas_heat_capacity_hyperbolic', argon)
        self.assertIn('ideal_gas_heat_capacity_polynomial', argon)

    def test_normal_points_are_not_embedded_and_boiling_points_are_curve_derived(self):
        from perry_properties import get_perry_property_library

        library = get_perry_property_library()

        self.assertTrue(all('normal_points' not in entry for entry in self.chemicals.values()))

        ethanol_tb = library.normal_boiling_point_K('64-17-5')
        self.assertIsNotNone(ethanol_tb)
        self.assertEqual(ethanol_tb.method, 'perry_vapor_pressure_normal_boiling_point')
        self.assertClose(ethanol_tb.value, 351.46033248849017)

        acetic_tb = library.normal_boiling_point_K('64-19-7')
        self.assertIsNotNone(acetic_tb)
        self.assertEqual(acetic_tb.method, 'perry_vapor_pressure_normal_boiling_point')
        self.assertClose(acetic_tb.value, 391.157980877604)

        methyl_methacrylate_tb = library.normal_boiling_point_K('80-62-6')
        self.assertIsNotNone(methyl_methacrylate_tb)
        self.assertEqual(methyl_methacrylate_tb.method, 'perry_vapor_pressure_normal_boiling_point')
        self.assertClose(methyl_methacrylate_tb.value, 373.19636455545015)

    def test_split_identity_preserves_spaced_inorganic_formulas(self):
        expected = {
            '10024-97-2': ('Nitrous oxide', 'N2O'),
            '2551-62-4': ('Sulfur hexafluoride', 'F6S'),
            '7446-09-5': ('Sulfur dioxide', 'O2S'),
            '7446-11-9': ('Sulfur trioxide', 'O3S'),
            '7783-54-2': ('Nitrogen trifluoride', 'F3N'),
        }

        for cas, (name, formula) in expected.items():
            entry = self.chemicals[cas]
            self.assertEqual(entry['name'], name)
            self.assertEqual(entry['formula'], formula)
            self.assertEqual(entry['names'], [name])
            self.assertEqual(entry['formulas'], [formula])

    def test_curated_amine_critical_volume_corrections_survive_regeneration(self):
        expected = {
            '107-10-8': (0.23, 0.2639, 0.26, 0.298),
            '107-15-3': (0.204, 0.2603, 0.264, 0.337),
            '108-18-9': (0.407, 0.2995, 0.418, 0.308),
            '74-89-5': (0.141, 0.2942, 0.154, 0.321),
            '75-04-7': (0.18, 0.2667, 0.207, 0.307),
        }
        for cas, (vc, zc, original_vc, original_zc) in expected.items():
            critical = self.chemicals[cas]['critical_constants']
            self.assertClose(critical['Vc_m3_per_kmol'], vc)
            self.assertClose(critical['Zc'], zc)
            self.assertClose(critical['_vc_patch']['original_Vc_m3_per_kmol'], original_vc)
            self.assertClose(critical['_vc_patch']['original_Zc'], original_zc)

    def test_heat_of_fusion_is_separate_and_resolved_without_formula_only_merges(self):
        metadata = self.fusion_payload['metadata']

        self.assertEqual(metadata['source_table'], '2-68')
        self.assertEqual(metadata['rows_extracted'], 246)
        self.assertEqual(metadata['rows_resolved'], 246)
        self.assertEqual(metadata['unique_cas_resolved'], 243)
        self.assertEqual(metadata['rows_unresolved'], 0)
        self.assertEqual(
            metadata['resolver'],
            'layout_context+elemental_formula+chemicals+'
            'strict_perry_table_2_10+audited_overrides',
        )
        self.assertEqual(
            metadata['resolution_sources'],
            [
                'Perry Table 2-10 CAS map',
                'audited Perry Table 2-68 identity override',
                'chemicals.identifiers',
            ],
        )
        self.assertEqual(self.fusion_payload['unresolved'], [])

        acetic = self.fusion_chemicals['64-19-7']['heat_of_fusion'][0]
        self.assertEqual(acetic['table_name'], 'Acetic acid')
        self.assertClose(acetic['Tm_K'], 289.85)
        self.assertClose(acetic['Hfus_kJ_per_mol'], 11.7286954618752)

        phenanthrene = self.fusion_chemicals['85-01-8']['heat_of_fusion'][0]
        self.assertEqual(phenanthrene['table_name'], 'Phenanthrene')
        self.assertClose(phenanthrene['Tm_K'], 369.45)
        self.assertNotIn('119-64-2', self.fusion_chemicals)
        self.assertNotIn('93-58-3', self.fusion_chemicals)

        trimethylpentane = self.fusion_chemicals['564-02-3']['heat_of_fusion'][0]
        self.assertEqual(trimethylpentane['resolution_source'], 'chemicals.identifiers')
        self.assertEqual(trimethylpentane['table_name'], '2,2,3-Trimethylpentane')
        self.assertClose(trimethylpentane['Tm_K'], 160.88)

    def test_heat_of_fusion_identity_collisions_are_split_by_exact_cas(self):
        expected = {
            '103-65-1': ('n-Propylbenzene', 173.65, 8.533900255118398),
            '98-82-8': ('Isopropylbenzene', 177.122, 9.6653837892384),
            '1067-08-9': ('3-Methyl-3-ethylpentane', 182.28, 10.82850821684576),
            '589-81-1': ('3-Methylheptane', 152.65, 11.372394978145602),
            '1640-89-7': ('Ethylcyclopentane', 134.715, 4.559996272944),
            '1638-26-2': ('1,1-Dimethylcyclopentane', 203.42, 1.3803231961344),
            '1192-18-3': ('cis-1,2-Dimethylcyclopentane', 219.3, 1.5898365384047999),
            '822-50-4': ('trans-1,2-Dimethylcyclopentane', 155.58, 6.4415082486271995),
            '1759-58-6': ('trans-1,3-Dimethylcyclopentane', 139.47, 7.3658318174672),
            '527-53-7': ('Isodurene', 249.15, 12.91608197312),
            '95-93-2': ('Durene', 352.45, 21.002672425855998),
        }
        for cas, (name, tm, hfus) in expected.items():
            rows = self.fusion_chemicals[cas]['heat_of_fusion']
            self.assertEqual(len(rows), 1, cas)
            self.assertEqual(rows[0]['table_name'], name)
            self.assertClose(rows[0]['Tm_K'], tm)
            self.assertClose(rows[0]['Hfus_kJ_per_mol'], hfus)

        typo = self.fusion_chemicals['589-81-1']['heat_of_fusion'][0]
        self.assertEqual(typo['source_table_name'], '3-Methylpentane')
        self.assertIn('corrected to 3-Methylheptane', typo['source_correction'])

    def test_heat_of_fusion_layout_context_and_formula_errata_are_preserved(self):
        meta_aminobenzoic = self.fusion_chemicals['99-05-8']['heat_of_fusion'][0]
        self.assertEqual(meta_aminobenzoic['table_name'], 'Aminobenzoic acid (m-)')
        self.assertEqual(meta_aminobenzoic['source_name_fragment'], '(m-)')

        diisopropyl_ether = self.fusion_chemicals['108-20-3']['heat_of_fusion'][0]
        self.assertEqual(diisopropyl_ether['table_name'], 'Isopropyl ether')
        self.assertEqual(diisopropyl_ether['source_name_fragment'], 'ether')

        dimethyl_fumarate = self.fusion_chemicals['624-49-7']['heat_of_fusion'][0]
        self.assertEqual(dimethyl_fumarate['table_name'], 'Methyl fumarate')
        self.assertEqual(dimethyl_fumarate['source_name_fragment'], 'fumarate')

        aniline = self.fusion_chemicals['62-53-3']['heat_of_fusion'][0]
        self.assertEqual(aniline['formula'], 'C6H5NH2')
        self.assertClose(aniline['molecular_weight_estimate'], 93.12648)

        formula_errata = {
            '79-92-5': ('C10H12', 'C10H16'),
            '65-85-0': ('C7H8O2', 'C7H6O2'),
            '57-11-4': ('C18H30O2', 'C18H36O2'),
            '110-94-1': ('C6H8O4', 'C5H8O4'),
        }
        for cas, (source_formula, corrected_formula) in formula_errata.items():
            row = self.fusion_chemicals[cas]['heat_of_fusion'][0]
            self.assertEqual(row['source_formula'], source_formula)
            self.assertEqual(row['formula'], corrected_formula)
            self.assertIn('corrected', row['source_correction'])

    def test_heat_of_fusion_retains_form_rows_without_fabricating_tm(self):
        expected = {
            '79-11-8': 2,
            '112-05-0': 2,
            '112-37-8': 2,
        }
        for cas, count in expected.items():
            self.assertEqual(len(self.fusion_chemicals[cas]['heat_of_fusion']), count)

        pelargic = self.fusion_chemicals['112-05-0']['heat_of_fusion']
        beta = next(row for row in pelargic if 'β-' in row['table_name'])
        self.assertIsNone(beta['Tm_K'])
        self.assertIsNone(beta['Tm_C'])
        self.assertClose(beta['Hfus_cal_per_g'], 39.04)

        undecylic = self.fusion_chemicals['112-37-8']['heat_of_fusion']
        beta = next(row for row in undecylic if 'beta-' in row['table_name'])
        self.assertIsNone(beta['Tm_K'])
        self.assertClose(beta['Hfus_cal_per_g'], 42.91)


class PerryTable210ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(Path(ROOT, 'data', 'perry_table_2_10_vapor_pressure.json').read_text())
        cls.chemicals = cls.payload['chemicals']

    def assertClose(self, actual, expected, *, rel=1e-9, abs_tol=1e-12):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol),
            f'{actual!r} != {expected!r}',
        )

    def test_table_2_10_extraction_and_chemicals_resolution_counts_are_pinned(self):
        metadata = self.payload['metadata']

        self.assertEqual(metadata['source_table'], '2-10')
        self.assertEqual(metadata['rows_extracted'], 1090)
        self.assertEqual(metadata['rows_resolved'], 923)
        self.assertEqual(metadata['unique_cas_resolved'], 921)
        self.assertEqual(metadata['rows_unresolved'], 167)
        self.assertEqual(metadata['resolver'], 'chemicals+preserved')
        self.assertEqual(len(self.chemicals), 921)
        self.assertEqual(metadata['merge']['preserved_current_only_cas'], 16)
        self.assertEqual(metadata['merge']['removed_unresolved_rows_covered_by_preserved_cas'], 15)
        self.assertEqual(metadata['merge']['formula_conflicts'], 0)
        self.assertEqual(metadata['merge']['tb_conflicts'], 0)

        reasons = {}
        for row in self.payload['unresolved']:
            reasons[row['reason']] = reasons.get(row['reason'], 0) + 1
        self.assertEqual(reasons, {'formula_mismatch': 17, 'not_found': 150})

    def test_table_2_10_keeps_formula_isomers_separate_by_cas(self):
        methyl_methacrylate = self.chemicals['80-62-6']
        tiglic_acid = self.chemicals['80-59-1']
        self.assertEqual(methyl_methacrylate['table_name'], 'Methyl methacrylate')
        self.assertEqual(tiglic_acid['table_name'], 'Tiglic acid')
        self.assertEqual(methyl_methacrylate['formula'], tiglic_acid['formula'])
        self.assertClose(methyl_methacrylate['Tb_K'], 374.15)
        self.assertClose(tiglic_acid['Tb_K'], 471.65)

        pentane = self.chemicals['109-66-0']
        neopentane = self.chemicals['463-82-1']
        self.assertEqual(pentane['table_name'], 'n-Pentane')
        self.assertEqual(neopentane['table_name'], 'neo-Pentane (2,2-dimethylpropane)')
        self.assertEqual(pentane['formula'], neopentane['formula'])
        self.assertClose(pentane['Tb_K'], 309.25)
        self.assertClose(neopentane['Tb_K'], 282.65)

    def test_table_2_10_writes_tb_only_when_760_mmhg_column_exists(self):
        for entry in self.chemicals.values():
            pressures = {point['P_mmHg'] for point in entry['vapor_pressure']}
            if 760.0 in pressures:
                self.assertIn('Tb_K', entry)
                tb_point = next(point for point in entry['vapor_pressure'] if point['P_mmHg'] == 760.0)
                self.assertClose(entry['Tb_K'], tb_point['T_K'])
            else:
                self.assertNotIn('Tb_K', entry)

        thiodiglycol = self.chemicals['111-48-8']
        self.assertEqual(thiodiglycol['table_name'], 'Thiodiglycol (2,2′-thiodiethanol)')
        self.assertNotIn('Tb_K', thiodiglycol)
        self.assertEqual(
            [point['P_mmHg'] for point in thiodiglycol['vapor_pressure']],
            [1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0],
        )

    def test_table_2_10_preserves_melting_points_when_present(self):
        ethanol = self.chemicals['64-17-5']
        acetic_acid = self.chemicals['64-19-7']

        self.assertClose(ethanol['Tm_K'], 161.15)
        self.assertClose(acetic_acid['Tm_K'], 289.85)

        entries_with_tm = sum('Tm_K' in entry for entry in self.chemicals.values())
        self.assertEqual(entries_with_tm, 596)

    def test_table_2_10_source_sign_corrections_are_applied(self):
        diethyl_ether = self.chemicals['60-29-7']
        ethyl_formate = self.chemicals['109-94-4']

        diethyl_ether_40 = next(
            point for point in diethyl_ether['vapor_pressure']
            if point['P_mmHg'] == 40.0
        )
        self.assertClose(diethyl_ether_40['T_C'], -27.7)
        self.assertClose(diethyl_ether_40['T_K'], 245.45)

        ethyl_formate_100 = next(
            point for point in ethyl_formate['vapor_pressure']
            if point['P_mmHg'] == 100.0
        )
        self.assertClose(ethyl_formate_100['T_C'], 5.4)
        self.assertClose(ethyl_formate_100['T_K'], 278.55)

    def test_table_2_10_payloads_are_numeric_and_monotonic(self):
        expected_pressures = {1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0, 200.0, 400.0, 760.0}

        for cas, entry in self.chemicals.items():
            points = entry['vapor_pressure']
            self.assertGreaterEqual(len(points), 4, cas)
            last_pressure = None
            last_temperature = None
            seen_pressures = set()
            for point in points:
                self.assertIn(point['P_mmHg'], expected_pressures, cas)
                for key in ('P_mmHg', 'P_bar', 'T_C', 'T_K'):
                    self.assertIsInstance(point[key], (int, float), (cas, key))
                    self.assertTrue(math.isfinite(point[key]), (cas, key))
                self.assertClose(point['T_K'], point['T_C'] + 273.15)
                self.assertClose(point['P_bar'], 1.01325 * point['P_mmHg'] / 760.0)
                self.assertNotIn(point['P_mmHg'], seen_pressures, cas)
                seen_pressures.add(point['P_mmHg'])
                if last_pressure is not None:
                    self.assertGreater(point['P_mmHg'], last_pressure, cas)
                    self.assertGreater(point['T_K'], last_temperature, cas)
                last_pressure = point['P_mmHg']
                last_temperature = point['T_K']

            if 760.0 in seen_pressures:
                self.assertIn('Tb_K', entry, cas)
            else:
                self.assertNotIn('Tb_K', entry, cas)

    def test_table_2_10_spline_reproduces_all_points_and_stays_monotone(self):
        from perry_properties import get_perry_property_library

        library = get_perry_property_library()

        for cas, entry in self.chemicals.items():
            for point in entry['vapor_pressure']:
                result = library.table_2_10_vapor_pressure_bar(cas, point['T_K'])
                self.assertIsNotNone(result, cas)
                self.assertEqual(result.method, 'perry_table_2_10_vapor_pressure')
                self.assertClose(result.value, point['P_bar'], rel=5e-12, abs_tol=1e-14)

            temperatures = [point['T_K'] for point in entry['vapor_pressure']]
            last_pressure = None
            for idx in range(11):
                T = temperatures[0] + (temperatures[-1] - temperatures[0]) * idx / 10.0
                result = library.table_2_10_vapor_pressure_bar(cas, T)
                self.assertIsNotNone(result, (cas, T))
                self.assertGreater(result.value, 0.0, (cas, T))
                if last_pressure is not None:
                    self.assertGreaterEqual(result.value, last_pressure, (cas, T))
                last_pressure = result.value

    def test_table_2_10_malformed_multiline_names_are_repaired(self):
        self.assertEqual(
            self.chemicals['110-52-1']['table_name'],
            'Tetramethylene dibromide (1,4-dibromobutane)',
        )
        self.assertEqual(
            self.chemicals['108-45-2']['table_name'],
            'm-Phenylene diamine (1,3-phenylenediamine)',
        )
        self.assertEqual(
            self.chemicals['111-96-6']['table_name'],
            'Diethylene glycol dimethyl ether Di(2-methoxyethyl) ether',
        )
        self.assertNotIn(
            'Styrene benzene]',
            {row['table_name'] for row in self.payload['unresolved']},
        )


if __name__ == '__main__':
    unittest.main()
