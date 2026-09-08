import math
import os
import sys
import unittest
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chemical_properties import ChemicalDatabase, ChemicalProperties
from simulator import Simulator
from thermodynamics import (
    R,
    ThermodynamicsError,
    create_thermodynamics,
    get_henry_constant_database,
)
from unit_operations_basic import Flash
from unit_operations_base import UnitOperationError
from unit_operations_separation import Flash3


class HenryThermodynamicsTests(unittest.TestCase):
    @staticmethod
    def _hot_wet_ethylene_case():
        components = ['ethylene', 'ethanol', 'diethyl ether', 'water']
        composition = {
            'ethylene': 0.03382250709059489,
            'ethanol': 0.0961008504050113,
            'diethyl ether': 0.0005688800886797305,
            'water': 0.8695077624157141,
        }
        flow = 47.43226368669802
        thermo = create_thermodynamics(components, 'UNIFNIST-RK')
        inlet = thermo.calculate_state(
            423.15,
            1.01325,
            flow,
            composition,
        )
        return thermo, inlet

    def test_bundled_database_has_expected_hcp_units_and_quality(self):
        database = get_henry_constant_database()
        ethylene = database.get('74-85-1')
        oxygen = database.get('7782-44-7')

        self.assertEqual(len(database), 6794)
        self.assertEqual(database.metadata['source_version'], '5.0.0')
        self.assertEqual(database.metadata['schema_version'], '6')
        self.assertEqual(database.metadata['excluded_seawater_rows'], '98')
        self.assertEqual(database.metadata['Vinf_source_doi'], '10.1021/je000215o')
        self.assertEqual(database.metadata['temperature_source_correlations'], '238')
        self.assertEqual(database.metadata['resolved_temperature_correlations'], '362')
        self.assertEqual(database.metadata['rejected_temperature_correlations'], '18')
        self.assertAlmostEqual(ethylene.hcp_298, 5.03382276967697e-5, places=12)
        self.assertAlmostEqual(ethylene.B, 2084.72222222222, places=8)
        self.assertEqual((ethylene.quality_h, ethylene.quality_b), ('A+', 'A'))
        self.assertAlmostEqual(ethylene.quality_score_h, 0.974553927363178, places=12)
        self.assertAlmostEqual(ethylene.vinf_298_cm3_per_mol, 45.4, places=12)
        self.assertAlmostEqual(ethylene.vinf_uncertainty_cm3_per_mol, 1.3, places=12)
        self.assertAlmostEqual(oxygen.hcp_298, 1.26327092635363e-5, places=12)

    def test_ideal_thermo_hydrates_henry_data_and_temperature_dependence(self):
        thermo = create_thermodynamics(['water', 'ethylene'], 'IDEAL')
        props = thermo.props['ethylene']
        data = thermo.henry_component_data('ethylene')
        context = thermo.create_aqueous_equilibrium_context(['ethylene'], 'water')

        self.assertAlmostEqual(props.henry_Hcp, 5.03382276967697e-5, places=12)
        self.assertAlmostEqual(props.henry_B, 2084.72222222222, places=8)
        self.assertEqual(props.property_sources['henry_Hcp']['grade'], 'A+')
        self.assertAlmostEqual(
            props.property_sources['henry_Hcp']['quality'],
            0.974553927363178,
            places=12,
        )
        self.assertEqual(data.h_source, 'database')
        self.assertEqual(data.vinf_method, 'henry_vinf_measured_298K')
        self.assertAlmostEqual(data.vinf_cm3_per_mol, 45.4, places=12)
        self.assertTrue(data.more_temperature_dependence_available)
        self.assertEqual(data.temperature_correlation.model, 'exp_a_b_over_t_c_log_t')
        correlation = data.temperature_correlation
        expected = math.exp(
            correlation.A + correlation.B / 310.0 + correlation.C * math.log(310.0)
        )
        self.assertAlmostEqual(
            thermo.henry_constant_hcp('ethylene', 310.0, context),
            expected,
            delta=expected * 1e-12,
        )

    def test_common_gases_outrank_review_only_entries(self):
        database = get_henry_constant_database()
        common_gases = (
            '124-38-9',   # carbon dioxide
            '1333-74-0',  # hydrogen
            '7727-37-9',  # nitrogen
            '7782-44-7',  # oxygen
            '7440-37-1',  # argon
            '74-82-8',    # methane
            '74-85-1',    # ethylene
            '74-98-6',    # propane
        )
        for cas in common_gases:
            with self.subTest(cas=cas):
                self.assertEqual(database.get(cas).quality_h, 'A+')

        self.assertEqual(database.get('124-04-9').quality_h, 'B+')  # adipic acid
        self.assertEqual(database.get('584-03-2').quality_h, 'B+')  # 1,2-butanediol

    def test_iapws_common_gas_correlations_match_check_values_and_anchors(self):
        checks = {
            'oxygen': 1.5024,
            'nitrogen': 2.1716,
            'hydrogen': 1.9702,
            'methane': 1.4034,
            'carbon dioxide': -1.7508,
        }
        for component, expected_ln_kh_gpa in checks.items():
            with self.subTest(component=component):
                thermo = create_thermodynamics(['water', component], 'IDEAL')
                context = thermo.create_aqueous_equilibrium_context([component], 'water')
                data = context.component_data[component]
                raw_hcp = thermo._raw_iapws_hcp(300.0, context, data)
                concentration = thermo.aqueous_solvent_molar_concentration(300.0, context)
                ln_kh_gpa = math.log((concentration / raw_hcp) / 1.0e9)

                self.assertEqual(data.temperature_correlation.model, 'iapws_g7_04')
                self.assertAlmostEqual(ln_kh_gpa, expected_ln_kh_gpa, delta=8e-5)
                self.assertAlmostEqual(
                    thermo.henry_constant_hcp(component, 298.15, context, P=1.0),
                    data.hcp_298,
                    delta=data.hcp_298 * 1e-12,
                )
                self.assertAlmostEqual(
                    thermo.henry_dln_hcp_dinvT(component, 298.15, 1.0, context),
                    data.B,
                    delta=1e-5,
                )
                self.assertEqual(
                    thermo.henry_effective_quality(component, 400.0, 1.0, context)[
                        'temperature_penalty'
                    ],
                    0.0,
                )

    def test_brockbank_correlation_uses_reported_range_and_preserves_anchor(self):
        thermo = create_thermodynamics(['water', 'acetone'], 'IDEAL')
        context = thermo.create_aqueous_equilibrium_context(['acetone'], 'water')
        data = context.component_data['acetone']

        self.assertEqual(data.temperature_correlation.model, 'brockbank_dippr101')
        self.assertTrue(data.temperature_correlation.normalize_to_reference)
        self.assertEqual((data.temperature_min_K, data.temperature_max_K), (273.0, 373.15))
        self.assertEqual(data.temperature_correlation.uncertainty_upper_percent, 10.0)
        self.assertAlmostEqual(
            thermo.henry_constant_hcp('acetone', 298.15, context, P=1.0),
            data.hcp_298,
            delta=data.hcp_298 * 1e-12,
        )
        self.assertAlmostEqual(
            thermo.henry_dln_hcp_dinvT('acetone', 298.15, 1.0, context),
            data.B,
            delta=1e-5,
        )

    def test_iapws_oxygen_wide_range_check_values(self):
        thermo = create_thermodynamics(['water', 'oxygen'], 'IDEAL')
        context = thermo.create_aqueous_equilibrium_context(['oxygen'], 'water')
        data = context.component_data['oxygen']
        for temperature, expected_ln_kh_gpa in (
            (300.0, 1.5024),
            (400.0, 1.8832),
            (500.0, 1.1630),
            (600.0, -0.0276),
        ):
            with self.subTest(temperature=temperature):
                raw_hcp = thermo._raw_iapws_hcp(temperature, context, data)
                concentration = thermo.aqueous_solvent_molar_concentration(
                    temperature, context
                )
                actual = math.log((concentration / raw_hcp) / 1.0e9)
                self.assertAlmostEqual(actual, expected_ln_kh_gpa, delta=8e-5)

    def test_chapoy_propane_correlation_uses_measured_range_and_anchor(self):
        thermo = create_thermodynamics(['water', 'propane'], 'IDEAL')
        context = thermo.create_aqueous_equilibrium_context(['propane'], 'water')
        data = context.component_data['propane']

        self.assertEqual(data.temperature_correlation.model, 'chapoy_dippr101')
        self.assertTrue(data.temperature_correlation.normalize_to_reference)
        self.assertEqual((data.temperature_min_K, data.temperature_max_K), (277.62, 368.16))
        self.assertEqual(data.temperature_correlation.fit_quality, 0.985)
        self.assertEqual(
            data.temperature_correlation.doi,
            '10.1016/j.fluid.2004.08.040',
        )
        self.assertAlmostEqual(
            thermo.henry_constant_hcp('propane', 298.15, context, P=1.0),
            data.hcp_298,
            delta=data.hcp_298 * 1e-12,
        )
        self.assertAlmostEqual(
            thermo.henry_dln_hcp_dinvT('propane', 298.15, 1.0, context),
            data.B,
            delta=1e-5,
        )

    def test_low_quality_anchor_keeps_raw_curve_without_normalization(self):
        thermo = create_thermodynamics(['water', 'phenol'], 'IDEAL')
        context = thermo.create_aqueous_equilibrium_context(['phenol'], 'water')
        data = context.component_data['phenol']
        correlation = data.temperature_correlation

        self.assertEqual(correlation.model, 'brockbank_dippr101')
        self.assertFalse(correlation.normalize_to_reference)
        raw = thermo._raw_brockbank_hcp(298.15, context, data)
        actual = thermo.henry_constant_hcp('phenol', 298.15, context, P=1.0)
        self.assertAlmostEqual(actual, raw, delta=raw * 1e-12)
        self.assertGreater(abs(actual / data.hcp_298 - 1.0), 0.25)
        quality = thermo.henry_effective_quality('phenol', 298.15, 1.0, context)
        self.assertEqual(quality['base_quality'], correlation.fit_quality)
        self.assertFalse(quality['temperature_normalized_to_reference'])

    def test_incompatible_a_range_curve_is_rejected(self):
        database = ChemicalDatabase(enable_online=False)
        database.chemicals['ANTH'] = ChemicalProperties(
            symbol='ANTH', name='Anthracene', formula='C14H10',
            CAS='120-12-7', MW=178.23,
        )
        thermo = create_thermodynamics(['water', 'ANTH'], 'IDEAL', db=database)
        context = thermo.create_aqueous_equilibrium_context(['ANTH'], 'water')
        data = context.component_data['ANTH']
        self.assertIsNone(data.temperature_correlation)

    def test_ideal_aqueous_k_bypasses_psat_for_henry_component(self):
        thermo = create_thermodynamics(['water', 'ethylene'], 'IDEAL')
        context = thermo.create_aqueous_equilibrium_context(['ethylene'], 'water')
        original_psat = thermo.Psat

        def guarded_psat(comp, T):
            if comp == 'ethylene':
                raise AssertionError('Henry component must not request Psat')
            return original_psat(comp, T)

        with patch.object(thermo, 'Psat', side_effect=guarded_psat):
            K = thermo.aqueous_K_values(
                298.15,
                1.01325,
                {'water': 0.99999, 'ethylene': 1e-5},
                context,
            )

        expected = (
            thermo.aqueous_solvent_molar_concentration(298.15, context)
            / (
                thermo.henry_constant_hcp('ethylene', 298.15, context, P=1.01325)
                * 1.01325e5
            )
        )
        self.assertAlmostEqual(K['ethylene'], expected, delta=expected * 1e-12)
        self.assertGreater(K['ethylene'], 1e4)

    def test_context_selection_is_immutable_and_not_composition_switched(self):
        thermo = create_thermodynamics(['water', 'ethylene', 'acetone'], 'IDEAL')
        context = thermo.create_aqueous_equilibrium_context(['ethylene'], 'water')

        with self.assertRaises(TypeError):
            context.component_data['acetone'] = thermo.henry_component_data('acetone')
        first = thermo.aqueous_K_values(
            298.15,
            1.01325,
            {'water': 0.999989, 'ethylene': 1e-5, 'acetone': 1e-6},
            context,
        )
        second = thermo.aqueous_K_values(
            298.15,
            1.01325,
            {'water': 0.99899, 'ethylene': 1e-5, 'acetone': 0.001},
            context,
        )

        self.assertEqual(context.henry_components, frozenset({'ethylene'}))
        self.assertAlmostEqual(first['ethylene'], second['ethylene'], places=10)
        self.assertNotAlmostEqual(first['acetone'], first['ethylene'], places=3)

    def test_ordinary_k_values_remain_independent_of_aqueous_context(self):
        thermo = create_thermodynamics(['water', 'acetone'], 'UNIFNIST')
        composition = {'water': 0.999, 'acetone': 0.001}
        ordinary_before = thermo.K_values(298.15, 1.01325, composition)
        context = thermo.create_aqueous_equilibrium_context(['acetone'], 'water')
        aqueous = thermo.aqueous_K_values(298.15, 1.01325, composition, context)
        ordinary_after = thermo.K_values(298.15, 1.01325, composition)

        self.assertEqual(ordinary_before, ordinary_after)
        self.assertNotAlmostEqual(ordinary_before['acetone'], aqueous['acetone'], places=5)

    def test_uniquac_excludes_deferred_gases_from_liquid_submixture(self):
        components = ['water', 'ethanol', 'carbon monoxide']
        thermo = create_thermodynamics(components, 'UNIQUAC-VDM')
        reference = create_thermodynamics(['water', 'ethanol'], 'UNIQUAC-VDM')
        thermo.initialize()

        self.assertNotIn('carbon monoxide', thermo.r)
        self.assertIn('carbon monoxide', thermo._deferred_uniquac_rq_errors)
        vapor = thermo.calculate_state(
            633.15,
            1.0,
            1.0,
            {'water': 0.85, 'ethanol': 0.10, 'carbon monoxide': 0.05},
            phase='vapor',
            flash=False,
        )
        self.assertTrue(math.isfinite(vapor.H))

        composition = {
            'water': 0.8991,
            'ethanol': 0.0999,
            'carbon monoxide': 0.001,
        }
        gamma = thermo.activity_coefficients(298.15, composition)
        reference_gamma = reference.activity_coefficients(
            298.15,
            {'water': 0.9, 'ethanol': 0.1},
        )
        self.assertEqual(gamma['carbon monoxide'], 1.0)
        self.assertAlmostEqual(gamma['water'], reference_gamma['water'], places=11)
        self.assertAlmostEqual(gamma['ethanol'], reference_gamma['ethanol'], places=11)

        context = thermo.create_aqueous_equilibrium_context(
            ['carbon monoxide'],
            'water',
        )
        aqueous = thermo.aqueous_K_values(298.15, 1.01325, composition, context)
        self.assertGreater(aqueous['carbon monoxide'], 1.0)
        self.assertTrue(all(math.isfinite(value) for value in aqueous.values()))
        self.assertTrue(math.isfinite(
            thermo.aqueous_liquid_enthalpy(composition, 298.15, 1.01325, context)
        ))

    def test_deferred_uniquac_compiled_submixture_reports_interaction_clamp(self):
        thermo = create_thermodynamics(
            ['water', 'ethanol', 'ethylene'],
            'UNIQUAC',
            interaction_overrides=[{
                'component1': 'water',
                'component2': 'ethanol',
                'model': 'UNIQUAC',
                'tau12_a': 0.1,
                'tau12_b': 0.0,
                'tau21_a': -0.2,
                'tau21_b': 0.0,
                'do_not_extrapolate': True,
                'Tmin_K': 300.0,
                'Tmax_K': 350.0,
            }],
        )
        self.assertIn('ethylene', thermo._deferred_uniquac_rq_errors)
        if thermo._compiled_activity_backend() is None:
            self.skipTest('Compiled UNIQUAC backend is unavailable')

        thermo.activity_coefficients(
            400.0,
            {'water': 0.5, 'ethanol': 0.5, 'ethylene': 0.0},
        )
        warnings = [
            warning for warning in thermo.warnings
            if 'do_not_extrapolate=true' in warning
        ]
        self.assertEqual(len(warnings), 1)
        self.assertIn('350 K instead of 400 K', warnings[0])

    def test_uniquac_excludes_formula_symbol_carbon_monoxide_from_liquid(self):
        components = ['water', 'acrylic acid', 'propionic acid', 'CO']
        thermo = create_thermodynamics(components, 'UNIQUAC-VDM')
        composition = {
            'water': 0.85,
            'acrylic acid': 0.10,
            'propionic acid': 0.047,
            'CO': 0.003,
        }

        self.assertNotIn('CO', thermo.r)
        self.assertIn('CO', thermo._deferred_uniquac_rq_errors)
        gamma = thermo.activity_coefficients(626.45, composition)
        self.assertEqual(gamma['CO'], 1.0)
        self.assertGreater(thermo.K_values(626.45, 1.0, composition)['CO'], 1.0)

        flashed = thermo.calculate_state(
            626.45,
            1.0,
            1.0,
            composition,
            include=(),
        )
        self.assertEqual(flashed.fluid_vapor_fraction, 1.0)

    def test_uniquac_henry_context_uses_normalized_non_henry_submixture(self):
        full = create_thermodynamics(
            ['water', 'ethanol', 'carbon monoxide'],
            'UNIQUAC-VDM',
        )
        reference = create_thermodynamics(['water', 'ethanol'], 'UNIQUAC-VDM')
        context = full.create_aqueous_equilibrium_context(
            ['carbon monoxide'],
            'water',
        )
        liquid = {
            'water': 0.891,
            'ethanol': 0.099,
            'carbon monoxide': 0.010,
        }
        bulk = {'water': 0.9, 'ethanol': 0.1}
        full_gamma = full.activity_coefficients(
            320.0,
            full._aqueous_bulk_activity_composition(liquid, context),
        )
        reference_gamma = reference.activity_coefficients(320.0, bulk)

        for component in bulk:
            self.assertAlmostEqual(
                full_gamma[component],
                reference_gamma[component],
                places=11,
            )

    def test_flash_uses_frozen_henry_context_for_all_specification_pairs(self):
        thermo, inlet = self._hot_wet_ethylene_case()
        common = {'henry_components': 'ethylene'}
        reference = Flash(
            'F-TP',
            thermo,
            {**common, 'T': 348.15, 'P': 1.01325},
        ).solve({'in': inlet})
        vapor_fraction = reference.performance['vapor_fraction']
        duty_kW = reference.heat_duty / 3600.0

        self.assertAlmostEqual(vapor_fraction, 0.08843135859633866, places=9)
        self.assertAlmostEqual(
            reference.outlet_streams['vapor_out'].F
            * reference.outlet_streams['vapor_out'].composition['ethylene']
            / (inlet.F * inlet.composition['ethylene']),
            0.9995344815833964,
            places=7,
        )
        self.assertEqual(
            set(reference.performance['henry']['components']),
            {'ethylene'},
        )
        henry_report = reference.performance['henry']['components']['ethylene']
        self.assertEqual(henry_report['base_correlation_grade'], 'A')
        self.assertEqual(henry_report['minimum_effective_grade'], 'B+')
        self.assertAlmostEqual(henry_report['maximum_temperature_penalty'], 0.125)
        self.assertEqual(henry_report['Vinf_method'], 'henry_vinf_measured_298K')
        self.assertTrue(henry_report['richer_temperature_dependence_available_upstream'])
        self.assertLess(henry_report['maximum_Hcp_pressure_correction_factor'], 1.0)

        cases = {
            'PV': {'P': 1.01325, 'VF': vapor_fraction},
            'TV': {'T': 348.15, 'VF': vapor_fraction},
            'PQ': {'P': 1.01325, 'Q': duty_kW},
            'TQ': {'T': 348.15, 'Q': duty_kW},
            'VQ': {'VF': vapor_fraction, 'Q': duty_kW},
        }
        for label, params in cases.items():
            with self.subTest(specifications=label):
                result = Flash(
                    f'F-{label}',
                    thermo,
                    {**common, **params},
                ).solve({'in': inlet})
                self.assertAlmostEqual(result.performance['T_C'], 75.0, places=6)
                self.assertAlmostEqual(result.performance['P_bar'], 1.01325, places=7)
                self.assertAlmostEqual(
                    result.performance['vapor_fraction'],
                    vapor_fraction,
                    places=8,
                )
                self.assertAlmostEqual(result.heat_duty / 3600.0, duty_kW, places=5)

    def test_flash_henry_auto_selection_and_water_rich_gate(self):
        thermo = create_thermodynamics(['water', 'ethylene'], 'UNIFNIST')
        inlet = thermo.calculate_state(
            400.0,
            1.01325,
            1.0,
            {'water': 0.999, 'ethylene': 0.001},
            phase='vapor',
            flash=False,
        )
        selected = Flash('F-AUTO', thermo, {'T': 348.15, 'P': 1.01325}).solve(
            {'in': inlet}
        )
        henry = selected.performance['henry']
        self.assertTrue(henry['enabled'])
        self.assertEqual(henry['components']['ethylene']['reason'], 'noncondensable')

        conventional_thermo = create_thermodynamics(['water', 'ethanol'], 'UNIFAC')
        conventional_inlet = conventional_thermo.calculate_state(
            370.0,
            1.01325,
            1.0,
            {'water': 0.4, 'ethanol': 0.6},
        )
        auto = Flash(
            'F-GATED', conventional_thermo, {'T': 350.0, 'P': 1.01325}
        ).solve({'in': conventional_inlet})
        disabled = Flash(
            'F-OFF',
            conventional_thermo,
            {'T': 350.0, 'P': 1.01325, 'henry_components': 'none'},
        ).solve({'in': conventional_inlet})
        self.assertFalse(auto.performance['henry']['enabled'])
        self.assertEqual(
            auto.performance['henry']['disabled_reason'],
            'estimated_liquid_not_water_rich',
        )
        self.assertAlmostEqual(
            auto.performance['vapor_fraction'],
            disabled.performance['vapor_fraction'],
            places=12,
        )

    def test_flash3_uses_henry_for_vle_and_rejects_coupled_henry_vlle(self):
        thermo, inlet = self._hot_wet_ethylene_case()
        result = Flash3(
            'F3-HENRY',
            thermo,
            {'T': 348.15, 'P': 1.01325, 'henry_components': 'ethylene'},
        ).solve({'in': inlet})
        self.assertEqual(result.performance['flash_status'], 'aqueous_henry_vle')
        self.assertEqual(result.performance['phase_count'], 2)
        self.assertEqual(
            set(result.performance['henry']['components']),
            {'ethylene'},
        )

        lle_thermo = create_thermodynamics(['water', 'benzene'], 'UNIFNIST')
        lle_inlet = lle_thermo.calculate_state(
            313.15,
            1.01325,
            1.0,
            {'water': 0.5, 'benzene': 0.5},
            phase='liquid',
            flash=False,
        )
        with self.assertRaisesRegex(UnitOperationError, 'Coupled Henry/VLLE'):
            Flash3(
                'F3-VLLE',
                lle_thermo,
                {'T': 313.15, 'P': 1.01325, 'henry_components': 'benzene'},
            ).solve({'in': lle_inlet})

    def test_aqueous_k_paths_cover_activity_gamma_phi_and_eos_models(self):
        composition = {'water': 0.99999, 'ethylene': 1e-5}
        values = {}
        for method in (
            'IDEAL', 'UNIFNIST', 'UNIFNIST-RK', 'UNIFNIST-PR',
            'UNIFNIST-BV', 'RK', 'PR',
        ):
            with self.subTest(method=method):
                thermo = create_thermodynamics(['water', 'ethylene'], method)
                context = thermo.create_aqueous_equilibrium_context(['ethylene'], 'water')
                K = thermo.aqueous_K_values(298.15, 1.01325, composition, context)
                self.assertTrue(all(math.isfinite(value) and value > 0.0 for value in K.values()))
                self.assertGreater(K['ethylene'], 1e4)
                values[method] = K['ethylene']

        self.assertAlmostEqual(values['UNIFNIST'], values['IDEAL'])
        self.assertNotAlmostEqual(values['UNIFNIST-RK'], values['UNIFNIST'], places=4)
        self.assertNotAlmostEqual(values['UNIFNIST-BV'], values['UNIFNIST'], places=4)
        self.assertNotAlmostEqual(values['PR'], values['UNIFNIST'], places=4)

    def test_vdm_aqueous_path_preserves_vapor_association(self):
        thermo = create_thermodynamics(
            ['water', 'acetic acid', 'ethylene'],
            'UNIFNIST-VDM',
        )
        context = thermo.create_aqueous_equilibrium_context(['ethylene'], 'water')
        K = thermo.aqueous_K_values(
            330.0,
            1.01325,
            {'water': 0.98999, 'acetic acid': 0.01, 'ethylene': 1e-5},
            context,
        )

        self.assertGreater(K['ethylene'], 1e4)
        self.assertTrue(all(math.isfinite(value) and value > 0.0 for value in K.values()))

    def test_henry_k_uses_converged_vapor_fugacity_coefficient(self):
        cases = [
            ('UNIFNIST-RK', ['water', 'ethylene'], {'water': 0.99999, 'ethylene': 1e-5}),
            ('UNIFNIST-BV', ['water', 'ethylene'], {'water': 0.99999, 'ethylene': 1e-5}),
            ('PR', ['water', 'ethylene'], {'water': 0.99999, 'ethylene': 1e-5}),
            (
                'UNIFNIST-VDM',
                ['water', 'acetic acid', 'ethylene'],
                {'water': 0.98999, 'acetic acid': 0.01, 'ethylene': 1e-5},
            ),
        ]
        for method, components, composition in cases:
            with self.subTest(method=method):
                thermo = create_thermodynamics(components, method)
                context = thermo.create_aqueous_equilibrium_context(['ethylene'], 'water')
                K = thermo.aqueous_K_values(330.0, 2.0, composition, context)
                y_unnormalized = {
                    comp: composition.get(comp, 0.0) * K[comp]
                    for comp in components
                }
                y_total = sum(y_unnormalized.values())
                y = {comp: value / y_total for comp, value in y_unnormalized.items()}
                phi_v = thermo.fugacity_coefficients(330.0, 2.0, y, 'vapor')
                ideal_henry_K = thermo._henry_ideal_vapor_K_value(
                    'ethylene', 330.0, 2.0, context
                )

                self.assertAlmostEqual(
                    K['ethylene'],
                    ideal_henry_K / phi_v['ethylene'],
                    delta=K['ethylene'] * 2e-8,
                )

    def test_aqueous_enthalpy_replaces_normal_henry_liquid_contribution(self):
        thermo = create_thermodynamics(['water', 'ethylene'], 'IDEAL')
        context = thermo.create_aqueous_equilibrium_context(['ethylene'], 'water')
        x = {'water': 0.999, 'ethylene': 0.001}
        T = 305.0
        density_derivative = thermo._aqueous_solvent_dln_concentration_dinvT(T, context)
        B_x = (
            thermo.henry_dln_hcp_dinvT('ethylene', T, 1.01325, context)
            - density_derivative
        )
        bulk_h = thermo.mixture_enthalpy({'water': 1.0}, T, 0.0, P=1.01325)
        dissolved_h = 1000.0 * thermo.enthalpy_ideal_gas('ethylene', T) - R * B_x
        expected = 0.999 * bulk_h + 0.001 * dissolved_h

        with patch.object(thermo, 'mixture_enthalpy', wraps=thermo.mixture_enthalpy) as wrapped:
            actual = thermo.aqueous_liquid_enthalpy(x, T, 1.01325, context)
        bulk_composition = wrapped.call_args.args[0]

        self.assertAlmostEqual(actual, expected, places=8)
        self.assertNotIn('ethylene', bulk_composition)

    def test_activity_and_eos_enthalpy_exclude_henry_solute_from_bulk_model(self):
        x = {'water': 0.9999, 'benzene': 0.0001}
        for method in ('UNIFNIST', 'PR'):
            with self.subTest(method=method):
                thermo = create_thermodynamics(['water', 'benzene'], method)
                context = thermo.create_aqueous_equilibrium_context(['benzene'], 'water')
                with patch.object(
                    thermo,
                    'mixture_enthalpy',
                    wraps=thermo.mixture_enthalpy,
                ) as wrapped:
                    value = thermo.aqueous_liquid_enthalpy(x, 298.15, 1.01325, context)

                self.assertTrue(math.isfinite(value))
                bulk_composition = wrapped.call_args.args[0]
                self.assertNotIn('benzene', bulk_composition)

    def test_prebound_water_volume_avoids_resolver_in_temperature_loop(self):
        thermo = create_thermodynamics(['water', 'ethylene'], 'IDEAL')
        context = thermo.create_aqueous_equilibrium_context(['ethylene'], 'water')

        with patch.object(
            thermo,
            '_liquid_molar_volume_for_poynting',
            side_effect=AssertionError('hot loop must not call the resolver fallback'),
        ):
            concentrations = [
                thermo.aqueous_solvent_molar_concentration(T, context)
                for T in (285.0, 298.15, 320.0, 350.0)
            ]

        self.assertTrue(all(value > 5e4 for value in concentrations))
        self.assertNotEqual(concentrations[0], concentrations[-1])

    def test_missing_b_and_low_quality_are_warned_once(self):
        database = ChemicalDatabase(enable_online=False)
        database.chemicals['XH'] = ChemicalProperties(
            symbol='XH',
            name='p-anisoyl chloride',
            formula='C8H7ClO2',
            CAS='100-07-2',
            MW=170.59,
        )
        thermo = create_thermodynamics(['water', 'XH'], 'IDEAL', db=database)
        context = thermo.create_aqueous_equilibrium_context(['XH'], 'water')
        thermo.create_aqueous_equilibrium_context(['XH'], 'water')

        self.assertIsNone(context.component_data['XH'].B)
        self.assertEqual(context.component_data['XH'].quality_h, 'D+')
        self.assertEqual(
            sum('temperature coefficient B is unavailable' in warning for warning in thermo.warnings),
            1,
        )
        self.assertEqual(
            sum('low database quality grade D+' in warning for warning in thermo.warnings),
            1,
        )

        h_low = thermo.henry_constant_hcp('XH', 280.0, context)
        h_high = thermo.henry_constant_hcp('XH', 340.0, context)
        self.assertEqual(h_low, h_high)
        x = {'water': 0.999, 'XH': 0.001}
        actual = thermo.aqueous_liquid_enthalpy(x, 310.0, 1.01325, context)
        density_derivative = thermo._aqueous_solvent_dln_concentration_dinvT(310.0, context)
        bulk_h = thermo.mixture_enthalpy({'water': 1.0}, 310.0, 0.0, P=1.01325)
        dissolved_h = 1000.0 * thermo.enthalpy_ideal_gas('XH', 310.0) + R * density_derivative
        self.assertAlmostEqual(actual, 0.999 * bulk_h + 0.001 * dissolved_h, places=8)

    def test_measured_vinf_applies_krichevsky_kasarnovsky_correction(self):
        thermo = create_thermodynamics(['water', 'ethylene'], 'IDEAL')
        context = thermo.create_aqueous_equilibrium_context(['ethylene'], 'water')
        data = context.component_data['ethylene']
        uncorrected = thermo.henry_constant_hcp('ethylene', 298.15, context)
        corrected = thermo.henry_constant_hcp('ethylene', 298.15, context, P=100.0)
        expected_factor = math.exp(
            -data.vinf_cm3_per_mol * 1e-6 * (100.0 - 1.0) / (8.314e-5 * 298.15)
        )

        self.assertAlmostEqual(corrected / uncorrected, expected_factor, places=12)
        quality = thermo.henry_effective_quality('ethylene', 298.15, 100.0, context)
        self.assertEqual(quality['pressure_method'], 'measured_or_provided_Vinf')
        self.assertAlmostEqual(quality['pressure_penalty'], 0.018, places=12)
        self.assertEqual(
            thermo.henry_effective_quality('ethylene', 298.15, 10.0, context)[
                'pressure_penalty'
            ],
            0.0,
        )
        self.assertAlmostEqual(
            thermo.henry_effective_quality('ethylene', 298.15, 11.0, context)[
                'pressure_penalty'
            ],
            0.0002,
        )

    def test_uncorrected_high_pressure_warning_is_deduplicated(self):
        database = ChemicalDatabase(enable_online=False)
        database.chemicals['XH'] = ChemicalProperties(
            symbol='XH', name='No volume gas', formula='XH', CAS='', MW=40.0,
            henry_Hcp=1e-4, henry_B=1500.0,
            property_sources={
                'henry_Hcp': {'quality': 0.9, 'method': 'test'},
                'henry_B': {'quality': 0.9, 'method': 'test'},
            },
        )
        thermo = create_thermodynamics(['water', 'XH'], 'IDEAL', db=database)
        context = thermo.create_aqueous_equilibrium_context(['XH'], 'water')
        composition = {'water': 0.99999, 'XH': 1e-5}

        thermo.aqueous_K_values(298.15, 30.0, composition, context)
        thermo.aqueous_K_values(310.0, 30.0, composition, context)

        self.assertEqual(
            sum('no Krichevsky-Kasarnovsky' in warning for warning in thermo.warnings),
            1,
        )
        quality = thermo.henry_effective_quality('XH', 298.15, 30.0, context)
        self.assertEqual(quality['pressure_method'], 'uncorrected')
        self.assertAlmostEqual(quality['pressure_penalty'], 0.06, places=12)

    def test_temperature_quality_uses_default_and_explicit_ranges(self):
        database = ChemicalDatabase(enable_online=False)
        database.chemicals['HB'] = ChemicalProperties(
            symbol='HB', name='H and B', formula='HB', CAS='', MW=40.0,
            henry_Hcp=1e-4, henry_B=1500.0,
            property_sources={
                'henry_Hcp': {'quality': 0.9, 'method': 'test'},
                'henry_B': {'quality': 0.8, 'method': 'test'},
            },
        )
        database.chemicals['HONLY'] = ChemicalProperties(
            symbol='HONLY', name='H only', formula='HO', CAS='', MW=41.0,
            henry_Hcp=2e-4,
            property_sources={'henry_Hcp': {'quality': 0.9, 'method': 'test'}},
        )
        database.chemicals['RANGED'] = ChemicalProperties(
            symbol='RANGED', name='Ranged', formula='R', CAS='', MW=42.0,
            henry_Hcp=3e-4, henry_B=1200.0,
            henry_Tmin=290.0, henry_Tmax=310.0,
            property_sources={
                'henry_Hcp': {'quality': 0.9, 'method': 'test'},
                'henry_B': {'quality': 0.8, 'method': 'test'},
            },
        )
        thermo = create_thermodynamics(
            ['water', 'HB', 'HONLY', 'RANGED'], 'IDEAL', db=database
        )
        context = thermo.create_aqueous_equilibrium_context(
            ['HB', 'HONLY', 'RANGED'], 'water'
        )

        self.assertAlmostEqual(
            thermo.henry_effective_quality('HB', 323.15, 1.0, context)['effective_quality'],
            0.8,
        )
        self.assertAlmostEqual(
            thermo.henry_effective_quality('HB', 333.15, 1.0, context)['effective_quality'],
            0.75,
        )
        self.assertAlmostEqual(
            thermo.henry_effective_quality('HONLY', 308.15, 1.0, context)['effective_quality'],
            0.8,
        )
        self.assertAlmostEqual(
            thermo.henry_effective_quality('RANGED', 315.0, 1.0, context)['effective_quality'],
            0.775,
        )

    def test_vc_fallback_requires_quality_at_least_point_eight(self):
        database = ChemicalDatabase(enable_online=False)
        for symbol, quality in (('GOODVC', 0.8), ('BADVC', 0.79)):
            database.chemicals[symbol] = ChemicalProperties(
                symbol=symbol, name=symbol, formula=symbol, CAS='', MW=40.0,
                Vc=100.0, henry_Hcp=1e-4,
                property_sources={
                    'Vc': {'quality': quality, 'method': 'test'},
                    'henry_Hcp': {'quality': 0.9, 'method': 'test'},
                },
            )
        thermo = create_thermodynamics(['water', 'GOODVC', 'BADVC'], 'IDEAL', db=database)
        good = thermo.henry_component_data('GOODVC')
        bad = thermo.henry_component_data('BADVC')

        self.assertAlmostEqual(good.vinf_cm3_per_mol, 10.74 + 0.2683 * 100.0)
        self.assertEqual(good.vinf_method, 'henry_vinf_from_critical_volume')
        self.assertIsNone(good.vinf_uncertainty_cm3_per_mol)
        self.assertEqual(good.vinf_estimated_relative_mae, 0.09)
        context = thermo.create_aqueous_equilibrium_context(['GOODVC', 'BADVC'], 'water')
        good_quality = thermo.henry_effective_quality('GOODVC', 298.15, 20.0, context)
        bad_quality = thermo.henry_effective_quality('BADVC', 298.15, 20.0, context)
        self.assertAlmostEqual(good_quality['pressure_penalty'], 0.01)
        self.assertIsNone(bad.vinf_cm3_per_mol)
        self.assertEqual(bad.vinf_unavailable_reason, 'Vc_quality_below_0.8')
        self.assertAlmostEqual(bad_quality['pressure_penalty'], 0.03)

    def test_context_rejects_water_missing_h_and_invalid_override(self):
        thermo = create_thermodynamics(['water', 'ethylene'], 'IDEAL')
        with self.assertRaisesRegex(ThermodynamicsError, 'cannot use its own'):
            thermo.create_aqueous_equilibrium_context(['water'], 'water')

        database = ChemicalDatabase(enable_online=False)
        database.chemicals['NOH'] = ChemicalProperties(
            symbol='NOH', name='No Henry data', formula='X', MW=50.0,
        )
        missing = create_thermodynamics(['water', 'NOH'], 'IDEAL', db=database)
        with self.assertRaisesRegex(ThermodynamicsError, 'No Henry Hcp'):
            missing.create_aqueous_equilibrium_context(['NOH'], 'water')

        database.chemicals['BADH'] = ChemicalProperties(
            symbol='BADH', name='Bad Henry data', formula='Y', MW=60.0,
            henry_Hcp=-1.0,
        )
        invalid = create_thermodynamics(['water', 'BADH'], 'IDEAL', db=database)
        with self.assertRaisesRegex(ThermodynamicsError, 'positive and finite'):
            invalid.create_aqueous_equilibrium_context(['BADH'], 'water')

        database.chemicals['BADRANGE'] = ChemicalProperties(
            symbol='BADRANGE', name='Bad range', formula='BR', MW=61.0,
            henry_Hcp=1e-4, henry_B=1000.0,
            henry_Tmin=330.0, henry_Tmax=300.0,
        )
        invalid_range = create_thermodynamics(
            ['water', 'BADRANGE'], 'IDEAL', db=database
        )
        with self.assertRaisesRegex(ThermodynamicsError, 'positive, finite, and ordered'):
            invalid_range.create_aqueous_equilibrium_context(['BADRANGE'], 'water')

        database.chemicals['BADVINF'] = ChemicalProperties(
            symbol='BADVINF', name='Bad Vinf', formula='BV', MW=62.0,
            henry_Hcp=1e-4, henry_Vinf=-1.0,
        )
        invalid_vinf = create_thermodynamics(
            ['water', 'BADVINF'], 'IDEAL', db=database
        )
        with self.assertRaisesRegex(ThermodynamicsError, 'Vinf.*positive and finite'):
            invalid_vinf.create_aqueous_equilibrium_context(['BADVINF'], 'water')

    def test_pfd_field_overrides_merge_independently_with_database(self):
        pfd = (
            'PROCESS: Henry Override\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: IDEAL\n'
            '\n'
            'COMPONENTS:\n'
            '    water | Water | CAS=7732-18-5\n'
            '    ethylene | Ethylene | CAS=74-85-1, henry_Hcp=9e-5\n'
            '    oxygen | Oxygen | CAS=7782-44-7, henry_B=2222\n'
            '    nitrogen | Nitrogen | CAS=7727-37-9, henry_Hcp=7e-6, henry_B=1333, '
            'henry_Tmin=285, henry_Tmax=315, henry_Vinf=40, '
            'henry_Vinf_uncertainty=2\n'
            '\n'
            'STREAM Feed : FEED -> M-1.in\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = water:1.0\n'
            'STREAM Product : M-1.out -> PRODUCT\n'
            '\n'
            'UNIT M-1\n'
            '    TYPE: Mixer\n'
            '    PORTS:\n'
            '        in : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '        mode = adiabatic\n'
        )

        simulator = Simulator.from_string(pfd)
        result = simulator.run()
        ethylene = simulator.thermo.props['ethylene']
        ethylene_data = simulator.thermo.henry_component_data('ethylene')
        oxygen = simulator.thermo.henry_component_data('oxygen')
        nitrogen = simulator.thermo.henry_component_data('nitrogen')

        self.assertTrue(result.converged)
        self.assertAlmostEqual(ethylene.henry_Hcp, 9e-5, places=12)
        self.assertAlmostEqual(ethylene.henry_B, 2084.72222222222, places=8)
        self.assertEqual((ethylene_data.h_source, ethylene_data.b_source), ('pfd', 'database'))
        self.assertAlmostEqual(oxygen.hcp_298, 1.26327092635363e-5, places=12)
        self.assertAlmostEqual(oxygen.B, 2222.0, places=8)
        self.assertEqual((oxygen.h_source, oxygen.b_source), ('database', 'pfd'))
        self.assertAlmostEqual(nitrogen.hcp_298, 7e-6, places=12)
        self.assertAlmostEqual(nitrogen.B, 1333.0, places=8)
        self.assertEqual((nitrogen.h_source, nitrogen.b_source), ('pfd', 'pfd'))
        self.assertEqual((nitrogen.temperature_min_K, nitrogen.temperature_max_K), (285, 315))
        self.assertEqual(nitrogen.temperature_range_source, 'provided')
        self.assertEqual(nitrogen.vinf_cm3_per_mol, 40)
        self.assertEqual(nitrogen.vinf_method, 'pfd_component_override')
        self.assertIn('henry_Hcp=9e-05', simulator.pfd.components[1].to_pfd())

    def test_non_pfd_component_property_is_not_mislabeled_as_database_data(self):
        database = ChemicalDatabase(enable_online=False)
        database.chemicals['XH'] = ChemicalProperties(
            symbol='XH',
            name='Custom ethylene datum',
            formula='C2H4',
            CAS='74-85-1',
            MW=28.05,
            henry_Hcp=9e-5,
            property_sources={
                'henry_Hcp': {
                    'source': 'laboratory measurement',
                    'method': 'custom_lab',
                    'quality': 0.7,
                },
            },
        )
        thermo = create_thermodynamics(['water', 'XH'], 'IDEAL', db=database)
        data = thermo.henry_component_data('XH')

        self.assertEqual(data.hcp_298, 9e-5)
        self.assertEqual(data.h_source, 'component_property')
        self.assertIsNone(data.quality_h)
        self.assertEqual(data.b_source, 'database')


if __name__ == '__main__':
    unittest.main()
