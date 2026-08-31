import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chemical_properties import ChemicalDatabase, ChemicalProperties
from property_resolution import PropertyResolutionResult
from simulator import Simulator
from thermodynamics import create_thermodynamics
from unit_operations_basic import Flash


class ContextualQualityReportTests(unittest.TestCase):
    @staticmethod
    def _thermo():
        db = ChemicalDatabase(enable_online=False)
        db.chemicals['X'] = ChemicalProperties(
            symbol='X',
            name='Context quality component',
            formula='X',
            MW=50.0,
            Tb=350.0,
            Tc=500.0,
            Pc=40.0,
            omega=0.2,
            Hvap=30.0,
            Cp_coeffs=[33.0, 0.0, 0.0, 0.0],
            phase_at_STP='liquid',
        )
        return create_thermodynamics(['X'], 'IDEAL', db)

    @staticmethod
    def _sim_for_thermo(thermo):
        sim = Simulator.__new__(Simulator)
        sim.thermo = thermo
        sim.thermo_method = 'IDEAL'
        sim.pfd = SimpleNamespace(
            metadata=SimpleNamespace(process_name='Context Quality', version='1.0'),
            streams=[],
            units=[],
        )
        sim.result = SimpleNamespace(
            recycle_info={},
            converged=True,
            iterations=0,
            mass_balance_error=0.0,
            energy_balance_error=0.0,
            streams={},
            units={},
            errors=[],
            warnings=[],
        )
        return sim

    @staticmethod
    def _resolver():
        class FakeResolver:
            def resolve_surface_tension(self, symbol, T, props=None):
                if T < 320.0:
                    return PropertyResolutionResult(
                        value=0.08,
                        source='calculated',
                        method='test_surface_tension_fit',
                        quality=0.40,
                        notes='auxiliary broad bracketing probe',
                    )
                if T < 335.0:
                    return PropertyResolutionResult(
                        value=0.12,
                        source='calculated',
                        method='test_surface_tension_fit',
                        quality=0.82,
                        notes='result-affecting surface-tension lookup',
                    )
                return PropertyResolutionResult(
                    value=0.20,
                    source='provided',
                    method='high_quality_surface_tension_fit',
                    quality=0.98,
                    notes='high-quality provided surface-tension fit',
                )

        return FakeResolver()

    def test_auxiliary_contexts_are_retained_but_not_surfaced(self):
        thermo = self._thermo()
        with patch('property_resolver.get_property_resolver', return_value=self._resolver()):
            with thermo.quality_context(
                kind='unit',
                unit_id='COL-1',
                unit_type='RigorousDistillation',
                phase='component_classification',
                affects_result=False,
            ):
                thermo._pure_surface_tension('X', 300.0)

            with thermo.quality_context(
                kind='unit',
                unit_id='COL-1',
                unit_type='RigorousDistillation',
                phase='solve',
                affects_result=True,
            ):
                thermo._pure_surface_tension('X', 329.54)

        sim = self._sim_for_thermo(thermo)
        surfaced = sim._property_quality_report()
        retained = sim._property_quality_report(include_suppressed=True)

        self.assertEqual(len(surfaced), 1)
        row = surfaced[0]
        self.assertEqual(row['component'], 'X')
        self.assertEqual(row['property'], 'surface_tension(T)')
        self.assertEqual(row['method'], 'test_surface_tension_fit')
        self.assertEqual(row['severity'], 'medium')
        self.assertAlmostEqual(row['quality'], 0.82)
        self.assertEqual(row['count'], 1)
        self.assertEqual(row['total_count'], 2)
        self.assertEqual(row['contexts'][0]['unit_id'], 'COL-1')
        self.assertEqual(row['contexts'][0]['phase'], 'solve')

        auxiliary_rows = [
            item for item in retained
            if item['component'] == 'X'
            and item['property'] == 'surface_tension(T)'
            and item['method'] == 'test_surface_tension_fit'
        ]
        self.assertEqual(len(auxiliary_rows), 1)
        contexts = auxiliary_rows[0]['contexts']
        self.assertTrue(any(context['phase'] == 'component_classification' for context in contexts))
        self.assertTrue(any(context['phase'] == 'solve' for context in contexts))

    def test_high_quality_result_entries_are_retained_but_not_surfaced(self):
        thermo = self._thermo()
        with patch('property_resolver.get_property_resolver', return_value=self._resolver()):
            with thermo.quality_context(
                kind='stream',
                stream_id='Feed',
                phase='feed_spec',
                affects_result=True,
            ):
                thermo._pure_surface_tension('X', 340.0)

        sim = self._sim_for_thermo(thermo)
        self.assertEqual(sim._property_quality_report(), [])

        retained = sim._property_quality_report(include_suppressed=True)
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]['suppressed_reason'], 'high_quality')
        self.assertEqual(retained[0]['count'], 1)
        self.assertEqual(retained[0]['contexts'][0]['stream_id'], 'Feed')

    def test_recycle_trial_contexts_are_excluded_but_final_pass_counts(self):
        thermo = self._thermo()
        with patch('property_resolver.get_property_resolver', return_value=self._resolver()):
            with thermo.quality_context(
                kind='unit',
                unit_id='RECYCLE-UNIT',
                unit_type='Heater',
                phase='solve',
                affects_result=False,
            ):
                thermo._pure_surface_tension('X', 300.0)

            with thermo.quality_context(
                kind='unit',
                unit_id='RECYCLE-UNIT',
                unit_type='Heater',
                phase='solve',
                affects_result=True,
            ):
                thermo._pure_surface_tension('X', 329.54)

        sim = self._sim_for_thermo(thermo)
        report = sim._property_quality_report()
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]['count'], 1)
        self.assertEqual(report[0]['total_count'], 2)
        self.assertAlmostEqual(report[0]['quality'], 0.82)

    def test_solver_iteration_contexts_are_excluded_from_surface_report(self):
        thermo = self._thermo()
        with patch('property_resolver.get_property_resolver', return_value=self._resolver()):
            with thermo.quality_context(
                kind='unit',
                unit_id='COL-1',
                unit_type='RigorousDistillation',
                phase='solver_iteration',
                affects_result=False,
            ):
                thermo._pure_surface_tension('X', 300.0)

        sim = self._sim_for_thermo(thermo)
        self.assertEqual(sim._property_quality_report(), [])

        retained = sim._property_quality_report(include_suppressed=True)
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]['suppressed_reason'], 'no_result_context')
        self.assertEqual(retained[0]['contexts'][0]['phase'], 'solver_iteration')

    def test_pfr_quality_report_prints_contexts(self):
        thermo = self._thermo()
        with patch('property_resolver.get_property_resolver', return_value=self._resolver()):
            with thermo.quality_context(
                kind='unit',
                unit_id='COL-1',
                unit_type='RigorousDistillation',
                phase='solve',
                affects_result=True,
            ):
                thermo._pure_surface_tension('X', 329.54)

        sim = self._sim_for_thermo(thermo)
        pfr = sim._generate_pfr()

        self.assertIn('medium: X.surface_tension(T) quality=0.820', pfr)
        self.assertIn('resolver_calls = 1, T = 329.54 K', pfr)
        self.assertIn('contexts = unit COL-1 (RigorousDistillation) solve: 1 call(s)', pfr)

    def test_static_model_parameter_context_can_surface_result_quality(self):
        thermo = self._thermo()
        thermo.props['X'].property_sources['Tc'] = {
            'source': 'estimated',
            'method': 'guldberg_rule',
            'quality': 0.55,
            'notes': 'critical temperature used by EOS',
        }
        thermo.mark_property_source_context(
            'X',
            'Tc',
            kind='thermo_model',
            phase='pr_eos_parameters',
            affects_result=True,
        )

        sim = self._sim_for_thermo(thermo)
        report = sim._property_quality_report()

        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]['component'], 'X')
        self.assertEqual(report[0]['property'], 'Tc')
        self.assertEqual(report[0]['severity'], 'estimated')
        self.assertEqual(report[0]['contexts'][0]['kind'], 'thermo_model')
        self.assertEqual(report[0]['contexts'][0]['phase'], 'pr_eos_parameters')

    def test_static_thermochemical_sources_surface_when_state_uses_them(self):
        thermo = self._thermo()
        props = thermo.props['X']
        props.Hf = -100.0
        props.S = 200.0
        props.Hvap = 25.0
        props.property_sources['Hf'] = {
            'source': 'estimated',
            'method': 'group_contribution_hf',
            'quality': 0.55,
            'notes': 'estimated enthalpy of formation',
        }
        props.property_sources['S'] = {
            'source': 'provided',
            'method': 'entropy_table',
            'quality': 0.92,
            'notes': 'entropy table source',
        }
        props.property_sources['Hvap'] = {
            'source': 'provided',
            'method': 'normal_hvap',
            'quality': 0.96,
            'notes': 'high quality normal Hvap',
        }

        with thermo.quality_context(
            kind='stream',
            stream_id='Feed',
            phase='feed_spec',
            affects_result=True,
        ):
            thermo.calculate_state(
                298.15,
                1.0,
                1.0,
                {'X': 1.0},
                phase='liquid',
                flash=False,
            )

        sim = self._sim_for_thermo(thermo)
        report = sim._property_quality_report()
        keyed = {(item['component'], item['property']): item for item in report}

        self.assertIn(('X', 'Hf'), keyed)
        self.assertEqual(keyed[('X', 'Hf')]['severity'], 'estimated')
        self.assertIn(('X', 'S'), keyed)
        self.assertEqual(keyed[('X', 'S')]['severity'], 'watch')
        self.assertNotIn(('X', 'Hvap'), keyed)

        retained = sim._property_quality_report(include_suppressed=True)
        retained_keyed = {(item['component'], item['property']): item for item in retained}
        self.assertEqual(retained_keyed[('X', 'Hvap')]['suppressed_reason'], 'high_quality')

    def test_henry_static_sources_surface_when_aqueous_k_values_use_them(self):
        db = ChemicalDatabase(enable_online=False)
        db.chemicals['H2O'] = ChemicalProperties(
            symbol='H2O',
            name='Water',
            formula='H2O',
            MW=18.015,
            Tb=373.15,
            Tc=647.096,
            Pc=220.64,
            omega=0.344,
            phase_at_STP='liquid',
        )
        db.chemicals['XH'] = ChemicalProperties(
            symbol='XH',
            name='Henry test gas',
            formula='XH',
            MW=40.0,
            Tb=250.0,
            Tc=400.0,
            Pc=35.0,
            omega=0.1,
            henry_Hcp=1e-4,
            henry_B=1500.0,
            phase_at_STP='gas',
        )
        thermo = create_thermodynamics(['H2O', 'XH'], 'IDEAL', db)
        thermo.props['XH'].property_sources['henry_Hcp'] = {
            'source': 'test',
            'method': 'provided_henry_hcp',
            'quality': 0.92,
            'notes': 'Henry Hcp test source',
        }
        thermo.props['XH'].property_sources['henry_B'] = {
            'source': 'test',
            'method': 'provided_henry_b',
            'quality': 0.50,
            'notes': 'low-quality Henry B test source',
        }
        context = thermo.create_aqueous_equilibrium_context(
            ['XH'],
            water_component='H2O',
        )
        with thermo.quality_context(
            kind='unit',
            unit_id='ABS-1',
            unit_type='RigorousAbsorber',
            phase='solve',
            affects_result=True,
        ):
            thermo.aqueous_K_values(298.15, 1.0, {'H2O': 0.999, 'XH': 0.001}, context)

        sim = self._sim_for_thermo(thermo)
        report = sim._property_quality_report()
        keyed = {(item['component'], item['property']): item for item in report}

        self.assertIn(('XH', 'henry_Hcp'), keyed)
        self.assertEqual(keyed[('XH', 'henry_Hcp')]['severity'], 'watch')
        self.assertIn(('XH', 'henry_B'), keyed)
        self.assertEqual(keyed[('XH', 'henry_B')]['severity'], 'very_low')
        self.assertEqual(keyed[('XH', 'henry_B')]['contexts'][0]['phase'], 'aqueous_henry_equilibrium')

    def test_henry_solved_temperature_quality_is_reported(self):
        thermo = create_thermodynamics(['water', 'ethylene'], 'IDEAL')
        inlet = thermo.calculate_state(
            348.15,
            1.01325,
            1.0,
            {'water': 0.999, 'ethylene': 0.001},
        )
        Flash(
            'F-HQ',
            thermo,
            {'T': 348.15, 'P': 1.01325, 'henry_components': 'ethylene'},
        ).solve({'in': inlet})

        report = self._sim_for_thermo(thermo)._property_quality_report()
        effective = next(
            row for row in report
            if row['component'] == 'ethylene'
            and row['property'] == 'henry_Hcp_effective(T)'
        )
        self.assertEqual(effective['method'], 'henry_effective_TP_quality')
        self.assertEqual(effective['severity'], 'medium')
        self.assertAlmostEqual(effective['quality'], 0.824999, places=6)
        self.assertEqual((effective['T_min'], effective['T_max']), (348.15, 348.15))

    def test_unused_henry_data_does_not_surface_detailed_quality(self):
        thermo = create_thermodynamics(['water', 'acetone'], 'IDEAL')
        inlet = thermo.calculate_state(
            298.15,
            1.01325,
            1.0,
            {'water': 0.999, 'acetone': 0.001},
        )
        result = Flash(
            'F-NO-H',
            thermo,
            {'T': 298.15, 'P': 1.01325, 'henry_components': 'none'},
        ).solve({'in': inlet})

        self.assertFalse(result.performance['henry']['enabled'])
        self.assertEqual(result.performance['henry']['disabled_reason'], 'disabled_by_parameter')
        report = self._sim_for_thermo(thermo)._property_quality_report()
        self.assertFalse(any('henry' in row['property'].lower() for row in report))

    def test_thermo_property_contexts_are_isolated_from_shared_database_records(self):
        first = create_thermodynamics(['water', 'acetone'], 'UNIFNIST')
        context = first.create_aqueous_equilibrium_context(['acetone'], 'water')
        first.aqueous_K_values(
            298.15,
            1.01325,
            {'water': 0.999, 'acetone': 0.001},
            context,
        )

        second = create_thermodynamics(['water', 'acetone'], 'IDEAL')
        self.assertIsNot(first.props['acetone'], second.props['acetone'])
        self.assertIsNot(
            first.props['acetone'].property_sources,
            second.props['acetone'].property_sources,
        )
        self.assertTrue(any(
            source.get('contexts')
            for name, source in first.props['acetone'].property_sources.items()
            if name.startswith('henry_') and isinstance(source, dict)
        ))
        self.assertFalse(any(
            source.get('contexts')
            for name, source in second.props['acetone'].property_sources.items()
            if name.startswith('henry_') and isinstance(source, dict)
        ))

    def test_generated_methylpentane_report_hides_auxiliary_quality_noise(self):
        sim = Simulator.from_file(os.path.join(ROOT, '..', 'TESTS', 'goodluckseparatingme.pfd'))
        result = sim.run()
        self.assertTrue(result.converged, result.errors)
        pfr = sim._generate_pfr()

        self.assertNotIn('henry_Hcp', pfr)
        self.assertNotIn('henry_B', pfr)

        retained = sim._property_quality_report(include_suppressed=True)
        suppressed_methods = {
            (item['property'], item['method'], item.get('suppressed_reason'))
            for item in retained
        }
        self.assertIn(('henry_B', 'henry_database_temperature_coefficient', 'no_result_context'), suppressed_methods)

    def test_manual_dataset_sources_are_expanded_to_per_property_sources(self):
        db = ChemicalDatabase(enable_online=False)
        props = db.get('3-methylpentane', fetch_online=False)

        self.assertIsNotNone(props)
        self.assertNotIn('dataset', props.property_sources)
        for key in ('Hf', 'S', 'Hvap', 'Cp_coeffs', 'Cp_liquid', 'Antoine'):
            with self.subTest(key=key):
                self.assertIn(key, props.property_sources)
                self.assertEqual(
                    props.property_sources[key]['method'],
                    'Manual 3-methylpentane property card',
                )
        self.assertEqual(props.property_sources['Hf']['quality'], 0.98)
        self.assertEqual(props.property_sources['S']['quality'], 0.98)
        self.assertEqual(props.property_sources['Hvap']['quality'], 0.98)
        self.assertEqual(props.property_sources['Antoine']['quality'], 0.98)
        self.assertEqual(props.property_sources['Cp_coeffs']['quality'], 0.95)
        self.assertEqual(props.property_sources['Cp_liquid']['quality'], 0.95)

    def test_manual_dataset_properties_are_retained_high_quality_when_used(self):
        sim = Simulator.from_file(os.path.join(ROOT, '..', 'TESTS', 'goodluckseparatingme.pfd'))
        result = sim.run()
        self.assertTrue(result.converged, result.errors)

        report = sim._property_quality_report()
        liquid_cp = next(
            item for item in report
            if item['component'] == '3-mp' and item['property'] == 'Cp_liquid(T)'
        )
        self.assertEqual(liquid_cp['method'], 'stp_point_constant_liquid_cp_kernel')
        self.assertLess(liquid_cp['quality'], 0.95)

        retained = sim._property_quality_report(include_suppressed=True)
        keyed = {(item['component'], item['property']): item for item in retained}
        for key in ('Hf', 'S', 'Hvap'):
            with self.subTest(key=key):
                self.assertIn(('3-mp', key), keyed)
                self.assertEqual(keyed[('3-mp', key)]['suppressed_reason'], 'high_quality')
                self.assertTrue(keyed[('3-mp', key)]['affects_result'])
        self.assertEqual(keyed[('3-mp', 'Cp_liquid')]['suppressed_reason'], 'no_result_context')
        self.assertEqual(keyed[('3-mp', 'Cp_liquid(T)')]['quality'], liquid_cp['quality'])


if __name__ == '__main__':
    unittest.main()
