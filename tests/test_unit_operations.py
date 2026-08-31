import math
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from thermodynamics import StreamState, create_thermodynamics
from simulator import Simulator
from unifac import get_unifac_groups
import unit_operations_basic as basic_ops
from unit_operations import (
    Absorber,
    CMODistillation,
    Compressor,
    Cooler,
    Decanter,
    ShortcutDistillation,
    EquilibriumReactor,
    Expander,
    Flash,
    Flash3,
    HeatExchanger,
    Heater,
    KineticsCSTR,
    KineticsBatch,
    KineticsPFR,
    KineticsPackedBed,
    LiquidLiquidExtractor,
    McCabeThieleDistillation,
    Mixer,
    MolecularSieveDryer,
    Pipe,
    Pump,
    Reactor,
    RigorousAbsorber,
    RigorousDistillation,
    RigorousLiquidLiquidExtractor,
    RigorousStripper,
    ShortcutExtractor,
    Splitter,
    Stripper,
    UNIT_CLASSES,
    UnitOperationError,
    Valve,
)


def component_moles(streams):
    totals = {}
    for stream in streams:
        for comp, frac in stream.composition.items():
            totals[comp] = totals.get(comp, 0.0) + stream.F * frac
    return totals


def relative_component_balance(inlets, outlets):
    inlet_totals = component_moles(inlets)
    outlet_totals = component_moles(outlets)
    keys = set(inlet_totals) | set(outlet_totals)
    denom = sum(abs(value) for value in inlet_totals.values()) or 1.0
    return sum(abs(outlet_totals.get(k, 0.0) - inlet_totals.get(k, 0.0)) for k in keys) / denom


class UnitOperationSmokeTests(unittest.TestCase):
    def setUp(self):
        self.ideal = create_thermodynamics(['H2O', 'C2H5OH', 'N2', 'O2'], 'IDEAL')
        self.unifac = create_thermodynamics(['water', 'butanol'], 'UNIFAC')
        self.reactive = create_thermodynamics(['CO', 'H2', 'CH3OH'], 'IDEAL')

        self.liquid = self.ideal.calculate_state(
            298.15, 1.0, 100.0, {'H2O': 0.8, 'C2H5OH': 0.2}, phase='liquid'
        )
        self.liquid2 = self.ideal.calculate_state(
            320.0, 1.2, 50.0, {'H2O': 0.3, 'C2H5OH': 0.7}, phase='liquid'
        )
        self.gas = self.ideal.calculate_state(
            350.0, 1.0, 100.0, {'N2': 0.79, 'O2': 0.21}, phase='vapor'
        )
        self.hot = self.ideal.calculate_state(
            420.0, 2.0, 100.0, {'H2O': 1.0}, phase='liquid'
        )
        self.cold = self.ideal.calculate_state(
            300.0, 2.0, 100.0, {'C2H5OH': 1.0}, phase='liquid'
        )
        self.lle_feed = self.unifac.calculate_state(
            298.15, 1.0, 100.0, {'water': 0.5, 'butanol': 0.5}, phase='liquid'
        )
        self.solvent = self.unifac.calculate_state(
            298.15, 1.0, 50.0, {'water': 1.0}, phase='liquid'
        )
        self.reactive_feed = self.reactive.calculate_state(
            523.15, 80.0, 100.0, {'CO': 0.333, 'H2': 0.667}, phase='vapor'
        )

    def test_every_registered_unit_class_has_a_smoke_case(self):
        vlle = create_thermodynamics(['water', 'methanol', 'benzene'], 'NRTL')
        vlle_feed = StreamState(
            T=333.0,
            P=1.01325,
            F=100.0,
            composition={'water': 0.20, 'methanol': 0.30, 'benzene': 0.50},
            vapor_fraction=0.0,
        )
        cases = [
            (Mixer('U', self.ideal, {'mode': 'adiabatic'}), {'in1': self.liquid, 'in2': self.liquid2}, True),
            (Splitter('U', self.ideal, {'split_frac': 0.4}), {'in': self.liquid}, True),
            (Pump('U', self.ideal, {'P_out': 5}), {'in': self.liquid}, True),
            (Compressor('U', self.ideal, {'P_out': 5}), {'in': self.gas}, True),
            (Expander('U', self.ideal, {'P_out': 0.5}), {'in': self.gas}, True),
            (Valve('U', self.ideal, {'P_out': 0.5}), {'in': self.liquid}, True),
            (Pipe('U', self.ideal, {'length': 1.0, 'diameter': 0.1, 'max_step': 1.0, 'profile_points': 2}), {'in': self.liquid}, True),
            (Heater('U', self.ideal, {'T_out': 350}), {'in': self.liquid}, True),
            (Cooler('U', self.ideal, {'T_out': 290}), {'in': self.liquid}, True),
            (HeatExchanger('U', self.ideal, {'T_hot_out': 360}), {'hot_in': self.hot, 'cold_in': self.cold}, True),
            (Flash('U', self.ideal, {'T': 351.15, 'P': 1}), {'in': self.liquid}, True),
            (Flash3('U', vlle, {'T': 333.0, 'P': 1.01325, 'max_iter': 200}), {'in': vlle_feed}, True),
            (ShortcutDistillation('U', self.ideal, {'N_stages': 10, 'reflux_ratio': 2, 'D_to_F': 0.4}), {'feed': self.liquid}, True),
            (McCabeThieleDistillation('U', self.ideal, {'N_stages': 10, 'reflux_ratio': 2, 'D_to_F': 0.4}), {'feed': self.liquid}, True),
            (CMODistillation('U', self.ideal, {'N_stages': 6, 'feed_stage': 3, 'reflux_ratio': 2, 'D_to_F': 0.4, 'P_drop_per_stage': 0.0, 'mesh_tolerance': 1e-5}), {'feed': self.liquid}, True),
            (RigorousDistillation('U', self.ideal, {'N_stages': 4, 'feed_stage': 2, 'reflux_ratio': 2, 'D_to_F': 0.4, 'P_drop_per_stage': 0.0, 'mesh_tolerance': 1e-5}), {'feed': self.liquid}, True),
            (Decanter('U', self.unifac, {'T': 25, 'P': 1}), {'in': self.lle_feed}, True),
            (MolecularSieveDryer('U', self.ideal, {'target_water_mole_fraction': 0.01}), {'feed': self.liquid}, True),
            (ShortcutExtractor('U', self.unifac, {'N_stages': 3, 'T': 25}), {'feed': self.lle_feed, 'solvent': self.solvent}, True),
            (RigorousLiquidLiquidExtractor('U', self.unifac, {'N_stages': 1, 'T': 25}), {'feed': self.lle_feed, 'solvent': self.solvent}, True),
            (Absorber('U', self.ideal, {'N_stages': 3, 'T': 25, 'P': 1}), {'gas': self.gas, 'liquid': self.liquid}, True),
            (RigorousAbsorber('U', self.ideal, {'N_stages': 1, 'mode': 'isothermal', 'T': 25, 'P_drop_per_stage': 0.0, 'mesh_tolerance': 1e-5}), {'gas': self.gas, 'liquid': self.liquid}, True),
            (Stripper('U', self.ideal, {'N_stages': 3, 'T': 80, 'P': 1}), {'liquid': self.liquid, 'gas': self.gas}, True),
            (RigorousStripper('U', self.ideal, {'N_stages': 1, 'mode': 'isothermal', 'T': 25, 'P_drop_per_stage': 0.0, 'mesh_tolerance': 1e-5}), {'liquid': self.liquid, 'gas': self.gas}, True),
            (Reactor('U', self.reactive, {'T': 250, 'reactions': [{'equation': 'CO + 2 H2 -> CH3OH', 'conversion': '0.25'}]}), {'in': self.reactive_feed}, False),
            (EquilibriumReactor('U', self.reactive, {'T': 523.15, 'phase': 'vapor', 'reactions': [{'equation': 'CO + 2 H2 <=> CH3OH'}]}), {'in': self.reactive_feed}, False),
            (KineticsCSTR('U', self.reactive, {'volume': 5, 'T': 523.15, 'phase': 'vapor', 'reactions': [{'equation': 'CO + 2 H2 -> CH3OH', 'A': '10', 'Ea': '0', 'Ea_unit': 'J/mol', 'rate_basis': 'concentration', 'concentration_unit': 'kmol/m3', 'pressure_unit': 'bar', 'rate_unit': 'kmol/m3/h'}]}), {'in': self.reactive_feed}, False),
            (KineticsBatch('U', self.reactive, {'V_batch': 5, 't_reaction': 0.01, 'T': 523.15, 'phase': 'vapor', 'profile_points': 3, 'reactions': [{'equation': 'CO + 2 H2 -> CH3OH', 'A': '10', 'Ea': '0', 'Ea_unit': 'J/mol', 'rate_basis': 'concentration', 'concentration_unit': 'kmol/m3', 'pressure_unit': 'bar', 'rate_unit': 'kmol/m3/h'}]}), {'in': self.reactive_feed}, False),
            (KineticsPFR('U', self.reactive, {'volume': 5, 'T': 523.15, 'phase': 'vapor', 'profile_points': 6, 'reactions': [{'equation': 'CO + 2 H2 -> CH3OH', 'A': '10', 'Ea': '0', 'Ea_unit': 'J/mol', 'rate_basis': 'concentration', 'concentration_unit': 'kmol/m3', 'pressure_unit': 'bar', 'rate_unit': 'kmol/m3/h'}]}), {'in': self.reactive_feed}, False),
            (KineticsPackedBed('U', self.reactive, {'catalyst_mass': 5, 'bulk_catalyst_density': 500, 'bed_void_fraction': 0.4, 'diameter': 0.5, 'particle_diameter': 0.005, 'T': 523.15, 'phase': 'vapor', 'profile_points': 3, 'reactions': [{'equation': 'CO + 2 H2 -> CH3OH', 'A': '0.01', 'Ea': '0', 'Ea_unit': 'J/mol', 'rate_basis': 'concentration', 'concentration_unit': 'kmol/m3', 'pressure_unit': 'bar', 'rate_unit': 'kmol/kg_cat/h'}]}), {'in': self.reactive_feed}, False),
        ]

        tested_classes = {type(unit) for unit, _, _ in cases}
        self.assertEqual(set(UNIT_CLASSES.values()), tested_classes)

        for unit, inlets, should_balance in cases:
            with self.subTest(unit=type(unit).__name__):
                result = unit.solve(inlets)
                self.assertGreaterEqual(len(result.outlet_streams), 1)
                for outlet in result.outlet_streams.values():
                    self.assertGreaterEqual(outlet.F, 0.0)
                    self.assertAlmostEqual(sum(outlet.composition.values()), 1.0, places=6)
                    self.assertGreater(outlet.T, 0.0)
                    self.assertGreater(outlet.P, 0.0)

                if should_balance:
                    balance_error = relative_component_balance(
                        list(inlets.values()),
                        list(result.outlet_streams.values()),
                    )
                    self.assertLess(balance_error, 1e-6)

    def test_distillation_unit_names_are_registered_without_legacy_classes(self):
        self.assertIs(
            UNIT_CLASSES['McCabeThieleDistillation'],
            McCabeThieleDistillation,
        )
        self.assertIs(UNIT_CLASSES['CMODistillation'], CMODistillation)
        self.assertNotIn('CMODistillation2', UNIT_CLASSES)
        self.assertNotIn('CMODistillation3', UNIT_CLASSES)

    def test_rigorous_absorber_and_stripper_unit_names_are_registered(self):
        self.assertIs(UNIT_CLASSES['RigorousAbsorber'], RigorousAbsorber)
        self.assertIs(
            UNIT_CLASSES['RigorousAbsorptionColumn'],
            RigorousAbsorber,
        )
        self.assertIs(UNIT_CLASSES['RigorousStripper'], RigorousStripper)
        self.assertIs(
            UNIT_CLASSES['RigorousStrippingColumn'],
            RigorousStripper,
        )

    def test_rigorous_absorber_isothermal_mode_reports_heat_duty(self):
        result = RigorousAbsorber(
            'A',
            self.ideal,
            {
                'N_stages': 1,
                'mode': 'isothermal',
                'T': 25,
                'P_drop_per_stage': 0.0,
                'mesh_tolerance': 1e-5,
            },
        ).solve({'gas': self.gas, 'liquid': self.liquid})

        self.assertEqual(result.performance['mode'], 'isothermal')
        self.assertAlmostEqual(
            result.performance['stage_temperatures_C'][0],
            25.0,
            places=4,
        )
        self.assertAlmostEqual(
            result.heat_duty / 3600.0,
            result.performance['duty_kW'],
            places=10,
        )
        self.assertAlmostEqual(
            result.performance['overall_energy_relative_error'],
            0.0,
            places=10,
        )

    def test_rigorous_absorber_temperature_spec_implies_isothermal_mode(self):
        result = RigorousStripper(
            'S',
            self.ideal,
            {
                'N_stages': 1,
                'T': 25,
                'P_drop_per_stage': 0.0,
                'mesh_tolerance': 1e-5,
            },
        ).solve({'liquid': self.liquid, 'gas': self.gas})

        self.assertEqual(result.performance['mode'], 'isothermal')
        self.assertAlmostEqual(
            result.performance['stage_temperatures_C'][0],
            25.0,
            places=4,
        )

    def test_rigorous_absorber_and_stripper_report_initializer_paths(self):
        absorber = RigorousAbsorber(
            'A',
            self.ideal,
            {
                'N_stages': 1,
                'mode': 'isothermal',
                'T': 25,
                'P_drop_per_stage': 0.0,
                'mesh_tolerance': 1e-5,
            },
        ).solve({'gas': self.gas, 'liquid': self.liquid})
        stripper = RigorousStripper(
            'S',
            self.ideal,
            {
                'N_stages': 1,
                'mode': 'isothermal',
                'T': 25,
                'P_drop_per_stage': 0.0,
                'mesh_tolerance': 1e-5,
            },
        ).solve({'liquid': self.liquid, 'gas': self.gas})

        self.assertEqual(absorber.performance['initializer'], 'loading_profile')
        self.assertEqual(
            absorber.performance['initializer_attempts'][0]['initializer'],
            'loading_profile',
        )
        self.assertTrue(absorber.performance['initializer_attempts'][0]['success'])
        self.assertEqual(stripper.performance['initializer'], 'equilibrium_sweep')
        self.assertEqual(
            stripper.performance['initializer_attempts'][0]['initializer'],
            'equilibrium_sweep',
        )
        self.assertTrue(stripper.performance['initializer_attempts'][0]['success'])

    def test_rigorous_absorber_adiabatic_mode_closes_energy_balance(self):
        result = RigorousAbsorber(
            'A',
            self.ideal,
            {
                'N_stages': 1,
                'mode': 'adiabatic',
                'P_drop_per_stage': 0.0,
                'mesh_tolerance': 1e-5,
            },
        ).solve({'gas': self.gas, 'liquid': self.liquid})

        self.assertEqual(result.performance['mode'], 'adiabatic')
        self.assertAlmostEqual(result.heat_duty, 0.0, places=8)
        self.assertLess(result.performance['overall_energy_relative_error'], 1e-6)
        self.assertLess(result.performance['component_balance_error'], 1e-6)

    def test_rigorous_absorber_rejects_bad_isothermal_profile_length(self):
        with self.assertRaisesRegex(UnitOperationError, 'temperature profile'):
            RigorousAbsorber(
                'A',
                self.ideal,
                {
                    'N_stages': 2,
                    'mode': 'isothermal',
                    'stage_temperatures': '25',
                    'P_drop_per_stage': 0.0,
                },
            ).solve({'gas': self.gas, 'liquid': self.liquid})

    def test_absorber_henry_auto_requires_water_rich_liquid(self):
        unit = RigorousAbsorber('A', self.ideal, {'N_stages': 3})
        _, feeds = unit._absorber_feed_specs(
            {'gas': self.gas, 'liquid': self.liquid},
            3,
        )
        context, info, warnings = unit._absorber_henry_context(
            list(self.ideal.components),
            feeds,
        )

        self.assertIsNone(context)
        self.assertFalse(info['enabled'])
        self.assertEqual(info['disabled_reason'], 'estimated_liquid_not_water_rich')
        self.assertAlmostEqual(info['estimated_water_mole_fraction'], 0.8)
        self.assertEqual(warnings, [])

    def test_absorber_henry_manual_override_and_disable_controls(self):
        manual = RigorousAbsorber(
            'A',
            self.ideal,
            {'N_stages': 3, 'henry_components': 'N2'},
        )
        _, feeds = manual._absorber_feed_specs(
            {'gas': self.gas, 'liquid': self.liquid},
            3,
        )
        context, info, warnings = manual._absorber_henry_context(
            list(self.ideal.components),
            feeds,
        )

        self.assertEqual(context.henry_components, frozenset({'N2'}))
        self.assertEqual(info['components']['N2']['reason'], 'manual_override')
        self.assertTrue(any('below the pure-water cutoff' in warning for warning in warnings))

        disabled = RigorousAbsorber(
            'A',
            self.ideal,
            {'N_stages': 3, 'henry_components': 'none'},
        )
        context, info, warnings = disabled._absorber_henry_context(
            list(self.ideal.components),
            feeds,
        )
        self.assertIsNone(context)
        self.assertEqual(info['disabled_reason'], 'disabled_by_parameter')
        self.assertEqual(warnings, [])

    def test_absorber_henry_keeps_concentrated_condensable_conventional(self):
        thermo = create_thermodynamics(
            ['water', 'acetaldehyde', 'nitrogen', 'oxygen'],
            'UNIQUAC',
        )
        liquid = thermo.calculate_state(
            323.15,
            1.01325,
            200.0,
            {'water': 0.95, 'acetaldehyde': 0.05},
            phase='liquid',
            flash=False,
        )
        air = thermo.calculate_state(
            353.15,
            1.01325,
            100.0,
            {'nitrogen': 0.79, 'oxygen': 0.21},
            phase='vapor',
            flash=False,
        )
        unit = RigorousStripper('S', thermo, {'N_stages': 5})
        _, feeds = unit._absorber_feed_specs({'liquid': liquid, 'air': air}, 5)
        context, info, _ = unit._absorber_henry_context(
            list(thermo.components),
            feeds,
        )

        self.assertTrue(context is not None)
        self.assertNotIn('acetaldehyde', context.henry_components)
        self.assertEqual(
            context.henry_components,
            frozenset({'nitrogen', 'oxygen'}),
        )
        self.assertAlmostEqual(info['estimated_water_mole_fraction'], 0.95)

    def test_absorber_henry_noncondensable_check_uses_aqueous_temperature(self):
        thermo = create_thermodynamics(['water', 'propane'], 'IDEAL')
        water = thermo.calculate_state(
            298.15,
            1.01325,
            2000.0,
            {'water': 1.0},
            phase='liquid',
            flash=False,
        )
        hot_propane = thermo.calculate_state(
            423.15,
            1.01325,
            100.0,
            {'propane': 1.0},
            phase='vapor',
            flash=False,
        )
        unit = RigorousAbsorber('A', thermo, {'N_stages': 3})
        _, feeds = unit._absorber_feed_specs(
            {'gas': hot_propane, 'water': water},
            3,
        )
        context, info, _ = unit._absorber_henry_context(
            list(thermo.components),
            feeds,
        )

        self.assertIsNone(context)
        self.assertEqual(info['disabled_reason'], 'no_components_selected')
        self.assertAlmostEqual(info['selection_reference_temperature_C'], 25.0)
        self.assertGreater(thermo.props['propane'].Tc, 298.15)

    def test_absorber_henry_dilution_bound_uses_liquid_feed_water(self):
        thermo = create_thermodynamics(['water', 'acetone'], 'IDEAL')
        water = thermo.calculate_state(
            298.15,
            1.01325,
            100.0,
            {'water': 1.0},
            phase='liquid',
            flash=False,
        )
        wet_gas = thermo.calculate_state(
            423.15,
            1.01325,
            900.5,
            {'water': 900.0 / 900.5, 'acetone': 0.5 / 900.5},
            phase='vapor',
            flash=False,
        )
        unit = RigorousAbsorber('A', thermo, {'N_stages': 3})
        _, feeds = unit._absorber_feed_specs(
            {'gas': wet_gas, 'water': water},
            3,
        )
        context, info, _ = unit._absorber_henry_context(
            list(thermo.components),
            feeds,
        )

        self.assertIsNone(context)
        self.assertEqual(info['disabled_reason'], 'no_components_selected')
        self.assertAlmostEqual(info['aqueous_loading_water_basis_moles'], 100.0)
        self.assertGreater(info['estimated_water_mole_fraction'], 0.99)

    def test_absorber_henry_reports_solved_domain_departures(self):
        thermo = create_thermodynamics(['water', 'ethylene'], 'IDEAL')
        water = thermo.calculate_state(
            298.15,
            1.01325,
            1000.0,
            {'water': 1.0},
            phase='liquid',
            flash=False,
        )
        ethylene = thermo.calculate_state(
            298.15,
            1.01325,
            0.1,
            {'ethylene': 1.0},
            phase='vapor',
            flash=False,
        )
        unit = RigorousAbsorber('A', thermo, {'N_stages': 2})
        _, feeds = unit._absorber_feed_specs(
            {'gas': ethylene, 'water': water},
            2,
        )
        context, info, _ = unit._absorber_henry_context(
            list(thermo.components),
            feeds,
        )
        warnings = unit._absorber_henry_solution_warnings(
            context,
            info,
            [
                {'water': 0.9999, 'ethylene': 0.0001},
                {'water': 0.998, 'ethylene': 0.002},
            ],
            [298.15, 270.0],
        )

        self.assertTrue(any('above the dilute cutoff' in warning for warning in warnings))
        self.assertTrue(any('below its critical temperature' in warning for warning in warnings))
        self.assertAlmostEqual(
            info['components']['ethylene']['maximum_solved_aqueous_mole_fraction'],
            0.002,
        )
        self.assertLess(
            info['components']['ethylene']['minimum_temperature_margin_to_Tc_K'],
            0.0,
        )
        report = info['components']['ethylene']
        self.assertAlmostEqual(report['maximum_temperature_penalty'], 0.04075)
        self.assertEqual(report['pressure_quality_method'], 'measured_or_provided_Vinf')
        self.assertIn('minimum_Hcp_pressure_correction_factor', report)
        self.assertAlmostEqual(info['minimum_solved_temperature_C'], -3.15)
        self.assertAlmostEqual(info['maximum_solved_temperature_C'], 25.0)

    def test_rigorous_stripper_trace_unifnist_case_reports_stripping(self):
        components = [
            'water',
            'acetaldehyde',
            'acetone',
            'benzene',
            'diethyl ether',
            'nitrogen',
            'oxygen',
        ]
        groups = {
            comp: get_unifac_groups(comp, variant='UNIFNIST')
            for comp in components[:5]
        }
        thermo = create_thermodynamics(
            components,
            'UNIFNIST',
            unifac_groups=groups,
        )
        masses = {
            'water': 999.98,
            'acetaldehyde': 0.005,
            'acetone': 0.005,
            'benzene': 0.005,
            'diethyl ether': 0.005,
        }
        moles = {
            comp: mass / thermo.props[comp].MW
            for comp, mass in masses.items()
        }
        liquid_F = sum(moles.values())
        liquid_z = {
            comp: amount / liquid_F
            for comp, amount in moles.items()
        }
        air_MW = (
            0.79 * thermo.props['nitrogen'].MW
            + 0.21 * thermo.props['oxygen'].MW
        )
        liquid = thermo.calculate_state(
            298.15,
            1.01325,
            liquid_F,
            liquid_z,
            phase='liquid',
            flash=False,
        )
        air = thermo.calculate_state(
            298.15,
            1.01325,
            1000.0 / air_MW,
            {'nitrogen': 0.79, 'oxygen': 0.21},
            phase='vapor',
            flash=False,
        )

        result = RigorousStripper(
            'S',
            thermo,
            {
                'N_stages': 5,
                'P_drop_per_stage': 0.0,
                'component_solve_threshold': 1e-12,
                'component_scale_floor': 1e-10,
                'T_min': 250.0,
                'T_max': 400.0,
            },
        ).solve({'liquid': liquid, 'air': air})
        stripping = result.performance['component_stripping_fraction']
        henry = result.performance['henry']
        conventional = RigorousStripper(
            'S-NO-HENRY',
            thermo,
            {
                'N_stages': 5,
                'P_drop_per_stage': 0.0,
                'component_solve_threshold': 1e-12,
                'component_scale_floor': 1e-10,
                'T_min': 250.0,
                'T_max': 400.0,
                'henry_components': 'none',
            },
        ).solve({'liquid': liquid, 'air': air})
        conventional_stripping = conventional.performance[
            'component_stripping_fraction'
        ]

        self.assertEqual(
            conventional.performance['initializer'],
            'loading_profile',
        )
        self.assertEqual(
            conventional.performance['initializer_attempts'][0]['initializer'],
            'loading_profile',
        )
        self.assertIn(
            'acetaldehyde',
            conventional.performance['trace_initializer_components'],
        )
        self.assertLess(result.performance['mesh_residual'], 2e-6)
        self.assertLess(result.performance['component_balance_error'], 1e-10)
        self.assertAlmostEqual(stripping['acetaldehyde'], 0.97911, delta=2e-3)
        self.assertAlmostEqual(stripping['acetone'], 0.86773, delta=2e-3)
        self.assertGreater(stripping['benzene'], 0.99999)
        self.assertGreater(stripping['diethyl ether'], 0.9999)
        self.assertLess(stripping['acetone'], stripping['acetaldehyde'])
        for comp in ('acetaldehyde', 'acetone', 'diethyl ether'):
            self.assertLess(stripping[comp], conventional_stripping[comp])
        self.assertTrue(henry['enabled'])
        self.assertFalse(conventional.performance['henry']['enabled'])
        self.assertEqual(
            henry['components']['acetaldehyde']['reason'],
            'dilute_aqueous_upper_bound',
        )
        self.assertEqual(
            henry['components']['nitrogen']['reason'],
            'noncondensable',
        )

    def test_rigorous_absorber_concentrated_uniquac_acetaldehyde_case(self):
        thermo = create_thermodynamics(
            ['water', 'acetaldehyde', 'nitrogen', 'oxygen'],
            'UNIQUAC',
        )
        gas = thermo.calculate_state(
            298.15,
            1.01325,
            100.0,
            {'acetaldehyde': 0.1, 'nitrogen': 0.711, 'oxygen': 0.189},
            phase='vapor',
            flash=False,
        )
        water = thermo.calculate_state(
            298.15,
            1.01325,
            400.0,
            {'water': 1.0},
            phase='liquid',
            flash=False,
        )

        result = RigorousAbsorber(
            'A',
            thermo,
            {'N_stages': 5, 'P_drop_per_stage': 0.0},
        ).solve({'gas': gas, 'water': water})
        performance = result.performance

        self.assertEqual(performance['mode'], 'adiabatic')
        self.assertEqual(performance['initializer'], 'loading_profile')
        self.assertLess(performance['mesh_residual'], 1e-5)
        self.assertAlmostEqual(
            performance['component_absorption_fraction']['acetaldehyde'],
            0.55653331,
            delta=2e-5,
        )
        self.assertAlmostEqual(
            performance['gas_out_flow'],
            98.19742256,
            delta=5e-4,
        )
        self.assertAlmostEqual(
            performance['liquid_out_flow'],
            401.80257744,
            delta=5e-4,
        )
        self.assertAlmostEqual(
            performance['gas_out_temperature_C'],
            28.56140799,
            delta=2e-3,
        )
        self.assertAlmostEqual(
            performance['liquid_out_temperature_C'],
            25.70795438,
            delta=2e-3,
        )

        finite = RigorousAbsorber(
            'A-FINITE',
            thermo,
            {
                'N_stages': 5,
                'P_drop_per_stage': 0.0,
                'semi_analytic_flow_jacobian': False,
            },
        ).solve({'gas': gas, 'water': water})
        finite_performance = finite.performance
        global_thermo = RigorousAbsorber(
            'A-GLOBAL-THERMO',
            thermo,
            {
                'N_stages': 5,
                'P_drop_per_stage': 0.0,
                'semi_analytic_local_thermo_jacobian': False,
            },
        ).solve({'gas': gas, 'water': water})
        global_performance = global_thermo.performance
        self.assertEqual(
            performance['jacobian_method'],
            'semi_analytic_local_thermo',
        )
        self.assertEqual(
            global_performance['jacobian_method'],
            'semi_analytic_flow',
        )
        self.assertEqual(
            finite_performance['jacobian_method'],
            'colored_finite_difference',
        )
        self.assertLess(
            performance['function_evaluations'],
            global_performance['function_evaluations'],
        )
        self.assertLess(
            global_performance['function_evaluations'],
            finite_performance['function_evaluations'],
        )
        self.assertAlmostEqual(
            performance['gas_out_flow'],
            finite_performance['gas_out_flow'],
            delta=2e-6,
        )
        self.assertAlmostEqual(
            performance['liquid_out_temperature_C'],
            finite_performance['liquid_out_temperature_C'],
            delta=2e-5,
        )
        self.assertAlmostEqual(
            performance['gas_out_flow'],
            global_performance['gas_out_flow'],
            delta=2e-6,
        )

    def test_rigorous_absorber_hot_wet_feed_uses_selective_henry_components(self):
        components = ['ethylene', 'ethanol', 'diethyl ether', 'water']
        thermo = create_thermodynamics(components, 'UNIFNIST-RK')
        feed_masses = {
            'ethylene': 45.0,
            'ethanol': 210.0,
            'diethyl ether': 2.0,
            'water': 743.0,
        }
        feed_moles = {
            comp: mass / thermo.props[comp].MW
            for comp, mass in feed_masses.items()
        }
        feed_F = sum(feed_moles.values())
        gas = thermo.calculate_state(
            423.15,
            1.01325,
            feed_F,
            {comp: value / feed_F for comp, value in feed_moles.items()},
            phase='vapor',
            flash=False,
        )
        water = thermo.calculate_state(
            298.15,
            1.01325,
            1000.0 / thermo.props['water'].MW,
            {'water': 1.0},
            phase='liquid',
            flash=False,
        )

        result = RigorousAbsorber(
            'A',
            thermo,
            {
                'N_stages': 6,
                'P_drop_per_stage': 0.0,
                'T_min': 250.0,
                'T_max': 500.0,
                'max_iterations': 100,
            },
        ).solve({'gas': gas, 'water': water})
        performance = result.performance
        henry = performance['henry']

        self.assertEqual(
            set(henry['components']),
            {'ethylene', 'diethyl ether'},
        )
        self.assertEqual(henry['components']['ethylene']['reason'], 'noncondensable')
        self.assertEqual(
            henry['components']['diethyl ether']['reason'],
            'dilute_aqueous_upper_bound',
        )
        self.assertNotIn('ethanol', henry['components'])
        self.assertLess(performance['mesh_residual'], 1e-6)
        self.assertLess(performance['component_balance_error'], 1e-9)
        self.assertLess(
            performance['component_absorption_fraction']['ethylene'],
            1e-3,
        )
        self.assertGreater(
            performance['component_absorption_fraction']['ethanol'],
            0.1,
        )
        self.assertGreater(
            result.outlet_streams['liquid_out'].composition['water'],
            0.99,
        )

    def test_rigorous_stripper_concentrated_uniquac_acetaldehyde_case(self):
        thermo = create_thermodynamics(
            ['water', 'acetaldehyde', 'nitrogen', 'oxygen'],
            'UNIQUAC',
        )
        liquid = thermo.calculate_state(
            323.15,
            1.01325,
            200.0,
            {'water': 0.95, 'acetaldehyde': 0.05},
            phase='liquid',
            flash=False,
        )
        air = thermo.calculate_state(
            353.15,
            1.01325,
            100.0,
            {'nitrogen': 0.79, 'oxygen': 0.21},
            phase='vapor',
            flash=False,
        )

        result = RigorousStripper(
            'S',
            thermo,
            {'N_stages': 5, 'P_drop_per_stage': 0.0},
        ).solve({'liquid': liquid, 'air': air})
        performance = result.performance

        self.assertEqual(performance['mode'], 'adiabatic')
        self.assertEqual(performance['initializer'], 'equilibrium_sweep')
        self.assertLess(performance['mesh_residual'], 2e-6)
        self.assertAlmostEqual(
            performance['component_stripping_fraction']['acetaldehyde'],
            0.99904329,
            delta=2e-5,
        )
        self.assertAlmostEqual(
            performance['gas_out_temperature_C'],
            29.94899520,
            delta=2e-3,
        )
        self.assertAlmostEqual(
            performance['liquid_out_temperature_C'],
            25.59130950,
            delta=2e-3,
        )

    def test_flash_validates_basic_specs(self):
        with self.assertRaises(UnitOperationError):
            Flash('F', self.ideal, {'T': 300.0, 'P': 1.0}).solve({})
        with self.assertRaises(UnitOperationError):
            Flash('F', self.ideal, {'T': 300.0, 'P': 1.0}).solve({
                'a': self.liquid,
                'b': self.liquid2,
            })
        with self.assertRaises(UnitOperationError):
            Flash('F', self.ideal, {'P': -1.0, 'vapor_fraction': 0.5}).solve({'in': self.liquid})
        with self.assertRaises(UnitOperationError):
            Flash('F', self.ideal, {'P': 1.0, 'vapor_fraction': 1.2}).solve({'in': self.liquid})
        with self.assertRaises(UnitOperationError):
            Flash('F', self.ideal, {'P': 1.0}).solve({'in': self.liquid})
        with self.assertRaises(UnitOperationError):
            Flash('F', self.ideal, {'T': 300.0, 'P': 1.0, 'Q': 0.0}).solve({'in': self.liquid})

    def test_flash_supports_all_two_spec_combinations(self):
        thermo = create_thermodynamics(['C3H8', 'C4H10'], 'PR')
        feed = thermo.calculate_state(
            323.15,
            10.0,
            100.0,
            {'C3H8': 0.5, 'C4H10': 0.5},
        )
        target = Flash('FTP', thermo, {'T': 300.0, 'P': 5.0}).solve({'in': feed})
        target_T = target.performance['T_C'] + 273.15
        target_P = target.performance['P_bar']
        target_VF = target.performance['vapor_fraction']
        target_Q = target.performance['duty_kW']

        cases = [
            {'P': target_P, 'vapor_fraction': target_VF},
            {'T': target_T, 'vapor_fraction': target_VF},
            {'P': target_P, 'Q': target_Q},
            {'T': target_T, 'Q': target_Q},
            {'vapor_fraction': target_VF, 'Q': target_Q},
        ]
        for params in cases:
            with self.subTest(params=params):
                result = Flash('F', thermo, params).solve({'in': feed})
                self.assertAlmostEqual(result.performance['T_C'] + 273.15, target_T, places=4)
                self.assertAlmostEqual(result.performance['P_bar'], target_P, places=4)
                self.assertAlmostEqual(result.performance['vapor_fraction'], target_VF, places=5)
                self.assertAlmostEqual(result.performance['duty_kW'], target_Q, places=3)
                self.assertAlmostEqual(result.outlet_streams['vapor_out'].F, feed.F * target_VF, places=5)
                self.assertAlmostEqual(result.outlet_streams['liquid_out'].F, feed.F * (1.0 - target_VF), places=5)

    def test_flash_temperature_only_inherits_upstream_pressure(self):
        result = Flash('F', self.ideal, {'T': 313.15}).solve({
            'in': self.liquid2,
        })

        self.assertAlmostEqual(result.performance['P_bar'], self.liquid2.P)
        self.assertEqual(result.performance['pressure_source'], 'upstream')
        for outlet in result.outlet_streams.values():
            self.assertAlmostEqual(outlet.P, self.liquid2.P)

    def test_henry_aware_cooler_and_adiabatic_flash_preserve_tp_result(self):
        thermo = create_thermodynamics(
            ['water', 'acetaldehyde', 'carbon monoxide'],
            'UNIQUAC',
        )
        feed = thermo.calculate_state(
            450.0,
            1.2,
            100.0,
            {'water': 0.85, 'acetaldehyde': 0.05, 'carbon monoxide': 0.10},
            phase='vapor',
            flash=False,
        )
        direct = Flash('DIRECT', thermo, {
            'T': 313.15,
            'P': feed.P,
            'henry_components': 'carbon monoxide',
        }).solve({'in': feed})
        cooled = Cooler('COOL', thermo, {
            'T': 313.15,
            'henry_components': 'carbon monoxide',
        }).solve({'in': feed})
        separated = Flash('SEP', thermo, {
            'Q': 0.0,
            'henry_components': 'carbon monoxide',
        }).solve({'in': cooled.outlet_streams['out']})

        self.assertAlmostEqual(cooled.heat_duty, direct.heat_duty, places=5)
        self.assertAlmostEqual(
            cooled.outlet_streams['out'].vapor_fraction,
            direct.performance['vapor_fraction'],
            places=10,
        )
        self.assertAlmostEqual(separated.performance['P_bar'], feed.P)
        self.assertEqual(separated.performance['pressure_source'], 'upstream')
        self.assertAlmostEqual(separated.performance['T_C'], 40.0, places=7)
        self.assertAlmostEqual(separated.performance['duty_kW'], 0.0, places=7)
        self.assertAlmostEqual(
            separated.performance['vapor_fraction'],
            direct.performance['vapor_fraction'],
            places=9,
        )

    def test_flash_pressure_duty_uses_robust_ph_for_wet_steam(self):
        thermo = create_thermodynamics(['H2O'], 'STEAM')
        feed = thermo.calculate_state(
            300.0 + 273.15,
            86.0,
            100.0,
            {'H2O': 1.0},
            include=('H', 'S', 'Cp', 'rho'),
        )

        result = Flash('FS', thermo, {'P': 0.1, 'Q': 0.0}).solve({'in': feed})

        self.assertAlmostEqual(result.performance['duty_kW'], 0.0, places=8)
        self.assertAlmostEqual(result.performance['T_C'], 45.807548, places=5)
        self.assertGreater(result.performance['vapor_fraction'], 0.45)
        self.assertLess(result.performance['vapor_fraction'], 0.52)
        self.assertAlmostEqual(result.performance['duty_residual_kW'], 0.0, places=8)

    def test_flash3_tp_splits_ternary_vlle_into_three_outlets(self):
        thermo = create_thermodynamics(['water', 'methanol', 'benzene'], 'NRTL')
        z = {'water': 0.20, 'methanol': 0.30, 'benzene': 0.50}
        feed = StreamState(
            T=333.0,
            P=1.01325,
            F=100.0,
            composition=z,
            vapor_fraction=0.0,
        )

        result = Flash3('F3', thermo, {'T': 333.0, 'P': 1.01325, 'max_iter': 200}).solve({'in': feed})

        self.assertEqual(result.performance['flash_status'], 'structured_vlle_feed_lle_seed')
        self.assertEqual(result.performance['phase_count'], 3)
        self.assertEqual(
            set(result.outlet_streams),
            {'vapor_out', 'liquid1_out', 'liquid2_out'},
        )
        self.assertGreater(result.outlet_streams['vapor_out'].F, 0.0)
        self.assertGreater(result.outlet_streams['liquid1_out'].F, 0.0)
        self.assertGreater(result.outlet_streams['liquid2_out'].F, 0.0)
        self.assertAlmostEqual(
            sum(stream.F for stream in result.outlet_streams.values()),
            feed.F,
            places=8,
        )
        self.assertLess(
            relative_component_balance([feed], list(result.outlet_streams.values())),
            1e-8,
        )
        self.assertLess(result.performance['residual'], 1e-7)
        self.assertIn(result.performance['light_liquid'], ('liquid1', 'liquid2'))
        self.assertIn(result.performance['heavy_liquid'], ('liquid1', 'liquid2'))
        self.assertNotEqual(
            result.performance['light_liquid'],
            result.performance['heavy_liquid'],
        )
        self.assertEqual(result.performance['phase_classifier'], 'mass_density')
        self.assertGreater(
            result.performance['heavy_mass_density_kg_m3'],
            result.performance['light_mass_density_kg_m3'],
        )
        self.assertAlmostEqual(
            result.performance['light_fraction'] + result.performance['heavy_fraction'],
            1.0,
            places=12,
        )
        self.assertEqual(
            result.performance['light_composition'],
            result.outlet_streams[f"{result.performance['light_liquid']}_out"].composition,
        )
        self.assertEqual(
            result.performance['heavy_composition'],
            result.outlet_streams[f"{result.performance['heavy_liquid']}_out"].composition,
        )

    def test_flash3_light_heavy_reporting_uses_heavy_component_fallback(self):
        flash = Flash3('F3', self.ideal, {'heavy_component': 'B'})
        phase1 = StreamState(
            T=300.0,
            P=1.0,
            F=40.0,
            composition={'A': 0.9, 'B': 0.1},
            vapor_fraction=0.0,
        )
        phase2 = StreamState(
            T=300.0,
            P=1.0,
            F=60.0,
            composition={'A': 0.2, 'B': 0.8},
            vapor_fraction=0.0,
        )

        diagnostics = flash._liquid_phase_diagnostics(phase1, phase2, phase1.F, phase2.F)

        self.assertEqual(diagnostics['phase_classifier'], 'heavy_component:B')
        self.assertEqual(diagnostics['light_liquid'], 'liquid1')
        self.assertEqual(diagnostics['heavy_liquid'], 'liquid2')
        self.assertEqual(diagnostics['light_composition'], phase1.composition)
        self.assertEqual(diagnostics['heavy_composition'], phase2.composition)

    def test_decanter_heavy_component_fallback_marks_richer_phase_heavy(self):
        decanter = Decanter('D', self.ideal, {'heavy_component': 'B'})
        phase1 = StreamState(
            T=300.0,
            P=1.0,
            F=40.0,
            composition={'A': 0.9, 'B': 0.1},
            vapor_fraction=0.0,
        )
        phase2 = StreamState(
            T=300.0,
            P=1.0,
            F=60.0,
            composition={'A': 0.2, 'B': 0.8},
            vapor_fraction=0.0,
        )

        light, heavy, classifier = decanter._classify_light_heavy(phase1, phase2)

        self.assertIs(light, phase1)
        self.assertIs(heavy, phase2)
        self.assertEqual(classifier, 'heavy_component:B')

    def test_flash3_pv_and_ph_specs_recover_reference_vlle_state(self):
        thermo = create_thermodynamics(['water', 'ethanol', 'cyclohexane'], 'UNIFNIST')
        z = {'water': 0.30, 'ethanol': 0.20, 'cyclohexane': 0.50}
        P = 1.01325
        T_reference = 337.0
        reference = thermo.flash3_TP(z, T_reference, P, max_iter=200)

        pv_feed = StreamState(
            T=T_reference + 5.0,
            P=P,
            F=100.0,
            composition=z,
            vapor_fraction=0.0,
        )
        pv = Flash3(
            'F3PV',
            thermo,
            {'P': P, 'vapor_fraction': reference.vapor_fraction, 'max_iter': 200},
        ).solve({'in': pv_feed})

        self.assertAlmostEqual(pv.performance['T_C'] + 273.15, T_reference, delta=1e-3)
        self.assertAlmostEqual(
            pv.performance['vapor_fraction'],
            reference.vapor_fraction,
            places=6,
        )
        self.assertEqual(pv.performance['phase_count'], 3)

        H_reference = thermo._flash3_mixture_enthalpy(T_reference, P, reference)
        ph_feed = StreamState(
            T=T_reference + 8.0,
            P=P,
            F=1.0,
            composition=z,
            vapor_fraction=0.0,
            H=H_reference,
        )
        ph = Flash3(
            'F3PH',
            thermo,
            {'P': P, 'Q': 0.0, 'max_iter': 200},
        ).solve({'in': ph_feed})

        self.assertAlmostEqual(ph.performance['T_C'] + 273.15, T_reference, delta=1e-3)
        self.assertEqual(ph.performance['phase_count'], 3)
        self.assertLess(abs(ph.performance['duty_kW']), 0.01)
        self.assertLess(abs(ph.performance['duty_residual_kW']), 0.01)
        self.assertLess(abs(ph.performance['enthalpy_residual_kJ_kmol']), 20.0)

    def test_flash3_vapor_fraction_duty_recovers_three_phase_branch(self):
        thermo = create_thermodynamics(['water', 'ethanol', 'cyclohexane'], 'UNIFNIST')
        z = {'water': 0.30, 'ethanol': 0.20, 'cyclohexane': 0.50}
        P = 1.01325
        F = 100.0
        T_reference = 337.0
        reference = thermo.flash3_TP(z, T_reference, P, max_iter=200)
        H_reference = thermo._flash3_mixture_enthalpy(T_reference, P, reference)

        self.assertEqual(reference.phase_count, 3)
        for feed_T in (330.0, 345.0, 360.0):
            with self.subTest(feed_T=feed_T):
                feed = StreamState(
                    T=feed_T,
                    P=P,
                    F=F,
                    composition=z,
                    vapor_fraction=0.0,
                )
                H_feed = thermo.mixture_enthalpy(
                    feed.composition,
                    feed.T,
                    feed.vapor_fraction,
                    feed.x,
                    feed.y,
                    feed.P,
                )
                Q_kW = F * (H_reference - H_feed) / 3600.0

                result = Flash3(
                    'F3VH',
                    thermo,
                    {
                        'vapor_fraction': reference.vapor_fraction,
                        'Q': Q_kW,
                        'max_iter': 200,
                    },
                ).solve({'in': feed})

                self.assertAlmostEqual(
                    result.performance['T_C'] + 273.15,
                    T_reference,
                    delta=1e-3,
                )
                self.assertAlmostEqual(result.performance['P_bar'], P, places=6)
                self.assertAlmostEqual(
                    result.performance['vapor_fraction'],
                    reference.vapor_fraction,
                    places=6,
                )
                self.assertEqual(result.performance['phase_count'], 3)
                self.assertEqual(result.performance['flash_status'], reference.status)
                self.assertLess(abs(result.performance['duty_residual_kW']), 1e-3)

    def test_flash3_aliases_and_rejects_non_vlle_thermo(self):
        self.assertIs(UNIT_CLASSES['Flash3'], Flash3)
        self.assertIs(UNIT_CLASSES['ThreePhaseFlash'], Flash3)
        self.assertIs(UNIT_CLASSES['VLLEFlash'], Flash3)

        with self.assertRaisesRegex(UnitOperationError, 'VLLE-capable'):
            Flash3('F3', self.ideal, {'T': 300.0, 'P': 1.0}).solve({'in': self.liquid})

    def test_flash3_reports_binary_invariant_phase_amount_warning(self):
        thermo = create_thermodynamics(['water', 'chloroform'], 'UNIFNIST')
        z = {'water': 0.5, 'chloroform': 0.5}
        T = 329.1264566618235
        P = 1.01325
        feed = StreamState(T=T, P=P, F=100.0, composition=z, vapor_fraction=0.0)

        result = Flash3('F3', thermo, {'T': T, 'P': P, 'max_iter': 200}).solve({'in': feed})

        self.assertEqual(result.performance['flash_status'], 'binary_invariant_vlle')
        self.assertEqual(result.performance['phase_count'], 3)
        self.assertIn('vapor_fraction_bounds', result.performance)
        self.assertTrue(any('underdetermined' in warning for warning in result.warnings))

    def test_decanter_validates_specs_and_rejects_vapor(self):
        thermo = create_thermodynamics(['water', 'hexane'], 'UNIFAC')
        liquid = thermo.calculate_state(
            298.15, 1.0, 100.0, {'water': 0.5, 'hexane': 0.5}, phase='liquid'
        )
        vapor = thermo.calculate_state(
            330.0, 1.0, 100.0, {'water': 0.5, 'hexane': 0.5}, phase='vapor'
        )

        with self.assertRaisesRegex(UnitOperationError, 'no inlet'):
            Decanter('D', thermo, {}).solve({})
        with self.assertRaisesRegex(UnitOperationError, 'pressure must be positive'):
            Decanter('D', thermo, {'P': -1.0}).solve({'in': liquid})
        with self.assertRaisesRegex(UnitOperationError, 'operating temperature must be positive'):
            Decanter('D', thermo, {'T': -300.0}).solve({'in': liquid})
        with self.assertRaisesRegex(UnitOperationError, 'cannot specify both'):
            Decanter('D', thermo, {'T': 300.0, 'Q': 1.0}).solve({'in': liquid})
        with self.assertRaisesRegex(UnitOperationError, 'Use Flash3/ThreePhaseFlash'):
            Decanter('D', thermo, {}).solve({'in': vapor})

        high_pressure = thermo.calculate_state(
            298.15, 2.0, 100.0, {'water': 0.5, 'hexane': 0.5}, phase='liquid'
        )
        with self.assertRaisesRegex(UnitOperationError, 'above the lowest inlet pressure'):
            Decanter('D', thermo, {'P': 3.0}).solve({'in': high_pressure})

    def test_decanter_mixes_multiple_inlets_and_reports_lle_diagnostics(self):
        thermo = create_thermodynamics(['water', 'hexane'], 'UNIFAC')
        aqueous = thermo.calculate_state(
            298.15, 1.0, 60.0, {'water': 0.8, 'hexane': 0.2}, phase='liquid'
        )
        organic = thermo.calculate_state(
            298.15, 1.0, 40.0, {'water': 0.05, 'hexane': 0.95}, phase='liquid'
        )

        result = Decanter('D', thermo, {}).solve({'aqueous': aqueous, 'organic': organic})
        light = result.outlet_streams['light']
        heavy = result.outlet_streams['heavy']

        self.assertTrue(result.performance['two_phases'])
        self.assertEqual(result.performance['n_inlets'], 2)
        self.assertEqual(result.performance['phase_classifier'], 'mass_density')
        self.assertGreater(light.composition['hexane'], heavy.composition['hexane'])
        self.assertGreater(
            result.performance['heavy_mass_density_kg_m3'],
            result.performance['light_mass_density_kg_m3'],
        )
        self.assertAlmostEqual(light.F + heavy.F, 100.0, places=8)
        self.assertLess(result.performance['component_balance_residual'], 1e-9)
        self.assertAlmostEqual(result.heat_duty, 0.0, places=8)
        self.assertIn('outlet_enthalpy_residual_kW', result.performance)

    def test_decanter_accepts_activity_models_and_reports_no_lle(self):
        self.assertIs(UNIT_CLASSES['FlashLLE'], Decanter)

        for method in ('NRTL', 'UNIQUAC'):
            with self.subTest(method=method):
                thermo = create_thermodynamics(['water', 'butanol'], method)
                feed = thermo.calculate_state(
                    298.15,
                    1.0,
                    100.0,
                    {'water': 0.5, 'butanol': 0.5},
                    phase='liquid',
                )
                result = Decanter('D', thermo, {}).solve({'in': feed})
                self.assertIn('two_phases', result.performance)
                self.assertIn('light_composition', result.performance)
                self.assertIn('heavy_composition', result.performance)
                self.assertIn('duty_kW', result.performance)
                self.assertGreaterEqual(result.outlet_streams['light'].F, 0.0)
                self.assertGreaterEqual(result.outlet_streams['heavy'].F, 0.0)

        no_lle = Decanter('D2', create_thermodynamics(['water', 'butanol'], 'NRTL'), {})
        feed = no_lle.thermo.calculate_state(
            298.15, 1.0, 100.0, {'water': 0.5, 'butanol': 0.5}, phase='liquid'
        )
        result = no_lle.solve({'in': feed})
        self.assertFalse(result.performance['two_phases'])
        self.assertTrue(result.performance['no_lle'])
        self.assertEqual(result.performance['phase_count'], 1)
        self.assertEqual(result.outlet_streams['heavy'].F, 0.0)
        self.assertIn('No liquid-liquid split', result.warnings[0])

    def test_shortcut_extractor_aliases(self):
        self.assertIs(UNIT_CLASSES['ShortcutExtractor'], ShortcutExtractor)
        self.assertIs(UNIT_CLASSES['Extractor'], ShortcutExtractor)
        self.assertIs(UNIT_CLASSES['LiquidLiquidExtractor'], ShortcutExtractor)
        self.assertIs(UNIT_CLASSES['LLE'], ShortcutExtractor)
        self.assertIs(LiquidLiquidExtractor, ShortcutExtractor)

    def test_decanter_uses_density_not_water_content_for_heavy_phase(self):
        class FakeLLEThermo:
            def liquid_liquid_equilibrium(self, composition, T, max_iter=100, tol=1e-6):
                return (
                    True,
                    {'water': 0.8, 'dense_solvent': 0.2},
                    {'water': 0.2, 'dense_solvent': 0.8},
                    0.5,
                )

            def mixture_MW(self, composition):
                return (
                    18.0 * composition.get('water', 0.0)
                    + 150.0 * composition.get('dense_solvent', 0.0)
                )

            def mixture_enthalpy(self, composition, T, vapor_fraction=0.0,
                                 x=None, y=None, P=1.0):
                return 1000.0 * T * (
                    1.0 + 0.2 * composition.get('dense_solvent', 0.0)
                )

            def calculate_state(self, T, P, F, composition, phase=None,
                                flash=False, include=None):
                state = StreamState(
                    T=T,
                    P=P,
                    F=F,
                    composition=dict(composition),
                    vapor_fraction=0.0,
                    x=dict(composition),
                    y=None,
                )
                state.MW = self.mixture_MW(composition)
                state.H = self.mixture_enthalpy(composition, T, 0.0, composition, None, P)
                mass_density = (
                    850.0
                    if composition.get('water', 0.0) > composition.get('dense_solvent', 0.0)
                    else 1300.0
                )
                state.rho = mass_density / state.MW
                return state

        feed = StreamState(
            T=300.0,
            P=1.0,
            F=100.0,
            composition={'water': 0.5, 'dense_solvent': 0.5},
            vapor_fraction=0.0,
            x={'water': 0.5, 'dense_solvent': 0.5},
            y=None,
        )
        feed.MW = 84.0
        feed.H = 330000.0
        feed.rho = 1000.0 / feed.MW

        result = Decanter('D', FakeLLEThermo(), {'T': 300.0}).solve({'in': feed})

        self.assertEqual(result.performance['phase_classifier'], 'mass_density')
        self.assertGreater(
            result.outlet_streams['heavy'].composition['dense_solvent'],
            result.outlet_streams['light'].composition['dense_solvent'],
        )
        self.assertGreater(
            result.performance['heavy_mass_density_kg_m3'],
            result.performance['light_mass_density_kg_m3'],
        )

    def test_decanter_pins_dcm_rich_phase_as_heavy(self):
        thermo = create_thermodynamics(
            ['water', 'acrylic acid', 'acetic acid', 'dichloromethane'],
            'UNIFAC',
        )
        composition = {
            'water': 0.4,
            'acrylic acid': 0.1,
            'acetic acid': 0.1,
            'dichloromethane': 0.4,
        }
        feed = thermo.calculate_state(
            298.15, 1.0, 100.0, composition, phase='liquid', flash=False
        )

        result = Decanter('D', thermo, {'T': 25.0, 'P': 1.0}).solve({'in': feed})
        light = result.outlet_streams['light']
        heavy = result.outlet_streams['heavy']

        self.assertTrue(result.performance['two_phases'])
        self.assertEqual(result.performance['phase_classifier'], 'mass_density')
        self.assertGreater(light.composition['water'], 0.80)
        self.assertLess(light.composition['dichloromethane'], 0.05)
        self.assertGreater(heavy.composition['dichloromethane'], 0.50)
        self.assertGreater(heavy.composition['acrylic acid'], 0.10)
        self.assertGreater(
            result.performance['heavy_mass_density_kg_m3'],
            result.performance['light_mass_density_kg_m3'],
        )
        self.assertAlmostEqual(result.performance['light_fraction'], 0.337, delta=0.05)
        self.assertAlmostEqual(result.performance['heavy_fraction'], 0.663, delta=0.05)

    def test_decanter_temperature_spec_reports_heat_duty(self):
        thermo = create_thermodynamics(['water', 'hexane'], 'UNIFAC')
        feed = thermo.calculate_state(
            298.15, 1.0, 100.0, {'water': 0.5, 'hexane': 0.5}, phase='liquid'
        )

        result = Decanter('D', thermo, {'T': 35.0}).solve({'in': feed})

        self.assertEqual(result.performance['mode'], 'specified_temperature')
        self.assertAlmostEqual(result.performance['T_C'], 35.0, places=8)
        self.assertAlmostEqual(result.heat_duty / 3600.0, result.performance['duty_kW'], places=10)
        self.assertNotAlmostEqual(result.heat_duty, 0.0, delta=1e-6)
        self.assertAlmostEqual(
            result.heat_duty / 3600.0,
            result.performance['outlet_enthalpy_residual_kW'],
            places=8,
        )

    def test_mixer_supports_explicit_temperature_pressure_and_duty_specs(self):
        high_pressure = self.ideal.calculate_state(
            330.0, 3.0, 40.0, {'H2O': 0.2, 'C2H5OH': 0.8}, phase='liquid'
        )
        low_pressure = self.ideal.calculate_state(
            300.0, 1.0, 60.0, {'H2O': 0.9, 'C2H5OH': 0.1}, phase='liquid'
        )

        adiabatic = Mixer('M1', self.ideal, {}).solve({
            'high': high_pressure,
            'low': low_pressure,
        })
        mixed = adiabatic.outlet_streams['out']

        self.assertAlmostEqual(mixed.P, 1.0, places=8)
        self.assertEqual(
            adiabatic.performance['pressure_policy'],
            'auto_valve_to_lowest_inlet_pressure',
        )
        self.assertAlmostEqual(adiabatic.heat_duty, 0.0, places=8)
        inlet_enthalpy = (
            high_pressure.F * high_pressure.H
            + low_pressure.F * low_pressure.H
        )
        self.assertAlmostEqual(mixed.F * mixed.H, inlet_enthalpy, delta=1e-4)

        specified = Mixer(
            'M2',
            self.ideal,
            {'T_out': 315.0, 'P_out': 0.8},
        ).solve({'high': high_pressure, 'low': low_pressure})
        outlet = specified.outlet_streams['out']

        self.assertAlmostEqual(outlet.T, 315.0, places=8)
        self.assertAlmostEqual(outlet.P, 0.8, places=8)
        self.assertAlmostEqual(
            specified.heat_duty,
            outlet.F * outlet.H - inlet_enthalpy,
            delta=1e-4,
        )

        duty = Mixer(
            'M3',
            self.ideal,
            {'Q': 25.0, 'P_out': 0.9},
        ).solve({'high': high_pressure, 'low': low_pressure})
        duty_out = duty.outlet_streams['out']
        self.assertAlmostEqual(duty.heat_duty, 25.0 * 3600.0, places=8)
        self.assertAlmostEqual(
            duty_out.F * duty_out.H,
            inlet_enthalpy + duty.heat_duty,
            delta=1e-4,
        )

    def test_mixer_rejects_pressure_boost_or_overspecified_energy(self):
        with self.assertRaisesRegex(UnitOperationError, 'above the lowest inlet pressure'):
            Mixer('M', self.ideal, {'P_out': 1.1}).solve({
                'low': self.liquid,
                'high': self.liquid2,
            })

        with self.assertRaisesRegex(UnitOperationError, 'both outlet temperature and heat duty'):
            Mixer('M', self.ideal, {'T_out': 310.0, 'Q': 10.0}).solve({
                'low': self.liquid,
                'high': self.liquid2,
            })

        with self.assertRaisesRegex(UnitOperationError, 'could not satisfy target enthalpy'):
            Mixer('M', self.ideal, {'Q': 1.0e12}).solve({
                'low': self.liquid,
                'high': self.liquid2,
            })

    def test_mixer_recomputes_missing_inlet_enthalpy_with_nonideal_thermo(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIFAC')
        ethanol_rich = thermo.calculate_state(
            298.15, 1.0, 10.0, {'ethanol': 0.8, 'water': 0.2},
            phase='liquid', flash=False,
        )
        water_rich = thermo.calculate_state(
            330.0, 1.0, 20.0, {'ethanol': 0.2, 'water': 0.8},
            phase='liquid', flash=False,
        )
        expected_enthalpy_flow = (
            ethanol_rich.F * ethanol_rich.H
            + water_rich.F * water_rich.H
        )
        ethanol_rich = ethanol_rich.copy()
        water_rich = water_rich.copy()
        ethanol_rich.H = None
        water_rich.H = None

        result = Mixer('M', thermo, {}).solve({
            'ethanol_rich': ethanol_rich,
            'water_rich': water_rich,
        })
        outlet = result.outlet_streams['out']

        self.assertAlmostEqual(result.heat_duty, 0.0, places=8)
        self.assertAlmostEqual(outlet.F * outlet.H, expected_enthalpy_flow, delta=1e-3)
        self.assertAlmostEqual(outlet.P, 1.0, places=8)

    def test_heat_exchanger_supports_tube_shell_q_spec_and_pressure_drops(self):
        hot = self.ideal.calculate_state(
            500.0, 5.0, 100.0, {'N2': 0.79, 'O2': 0.21}, phase='vapor'
        )
        cold = self.ideal.calculate_state(
            300.0, 4.0, 100.0, {'N2': 0.79, 'O2': 0.21}, phase='vapor'
        )
        result = HeatExchanger(
            'HX',
            self.ideal,
            {
                'Q': 20.0,
                'P_drop_tube': 0.2,
                'P_drop_shell': 0.05,
            },
        ).solve({
            'tube_in': hot,
            'shell_in': cold,
        })

        tube_out = result.outlet_streams['tube_out']
        shell_out = result.outlet_streams['shell_out']
        Q = 20.0 * 3600.0

        self.assertAlmostEqual(tube_out.P, hot.P - 0.2, places=8)
        self.assertAlmostEqual(shell_out.P, cold.P - 0.05, places=8)
        self.assertLess(tube_out.T, hot.T)
        self.assertGreater(shell_out.T, cold.T)
        self.assertAlmostEqual(
            hot.F * (hot.H - tube_out.H),
            Q,
            delta=1e-3,
        )
        self.assertAlmostEqual(
            cold.F * (shell_out.H - cold.H),
            Q,
            delta=1e-3,
        )
        self.assertAlmostEqual(result.performance['duty_kW'], 20.0, places=8)
        self.assertAlmostEqual(result.performance['hot_side_duty_kW'], -20.0, places=8)
        self.assertAlmostEqual(result.performance['cold_side_duty_kW'], 20.0, places=8)
        self.assertEqual(result.performance['hot_in_port'], 'tube_in')
        self.assertEqual(result.performance['cold_in_port'], 'shell_in')
        self.assertEqual(result.heat_duty, 0.0)
        self.assertGreater(result.performance['UA_required_W_per_K'], 0.0)
        self.assertNotIn('area_required_m2', result.performance)
        self.assertFalse(result.performance['auto_U_estimation'])
        self.assertEqual(len(result.performance['profile_heat_fraction']), 41)
        self.assertEqual(len(result.performance['profile_hot_T_C']), 41)
        self.assertEqual(len(result.performance['profile_cold_T_C']), 41)
        self.assertEqual(len(result.performance['profile_delta_T_K']), 41)
        self.assertAlmostEqual(result.performance['profile_heat_fraction'][0], 0.0)
        self.assertAlmostEqual(result.performance['profile_heat_fraction'][-1], 1.0)
        self.assertAlmostEqual(result.performance['profile_hot_T_C'][0], hot.T - 273.15)
        self.assertAlmostEqual(result.performance['profile_hot_T_C'][-1], tube_out.T - 273.15)
        self.assertAlmostEqual(result.performance['profile_cold_T_C'][0], shell_out.T - 273.15)
        self.assertAlmostEqual(result.performance['profile_cold_T_C'][-1], cold.T - 273.15)

    def test_heat_exchanger_forces_each_physical_side_phase_without_flashing(self):
        hot = self.ideal.calculate_state(
            400.0,
            2.0,
            100.0,
            {'N2': 0.79, 'O2': 0.21},
            phase='vapor',
            flash=False,
        )
        cold = self.ideal.calculate_state(
            300.0,
            2.0,
            100.0,
            {'C2H5OH': 1.0},
            phase='liquid',
            flash=False,
        )

        with patch.object(
            self.ideal,
            'flash_TP',
            side_effect=AssertionError('forced exchanger path invoked TP flash'),
        ):
            result = HeatExchanger(
                'HX-PHASE',
                self.ideal,
                {
                    'Q': 1.0,
                    'tube_phase': 'vapor',
                    'shell_phase': 'liquid',
                    'curve_segments': 8,
                },
            ).solve({
                'tube_in': hot,
                'shell_in': cold,
            })

        self.assertEqual(
            result.outlet_streams['tube_out'].vapor_fraction,
            1.0,
        )
        self.assertEqual(
            result.outlet_streams['shell_out'].vapor_fraction,
            0.0,
        )
        self.assertEqual(result.performance['hot_phase_mode'], 'vapor')
        self.assertEqual(result.performance['cold_phase_mode'], 'liquid')
        self.assertEqual(len(result.performance['profile_hot_T_C']), 9)
        self.assertEqual(len(result.performance['profile_cold_T_C']), 9)

    def test_heat_exchanger_rejects_invalid_forced_phase_specs(self):
        with self.assertRaisesRegex(UnitOperationError, 'must be auto, vapor, or liquid'):
            HeatExchanger(
                'HX-BAD-PHASE',
                self.ideal,
                {'Q': 1.0, 'tube_phase': 'solid'},
            ).solve({'tube_in': self.gas, 'shell_in': self.cold})

        with self.assertRaisesRegex(UnitOperationError, 'incompatible with forced vapor'):
            HeatExchanger(
                'HX-BAD-INLET',
                self.ideal,
                {'Q': 1.0, 'hot_phase': 'vapor'},
            ).solve({'hot_in': self.hot, 'cold_in': self.cold})

        with self.assertRaisesRegex(UnitOperationError, 'has no tube inlet port'):
            HeatExchanger(
                'HX-NO-TUBE',
                self.ideal,
                {'Q': 1.0, 'tube_phase': 'vapor'},
            ).solve({'hot_in': self.gas, 'cold_in': self.cold})

    def test_heat_exchanger_curve_rating_matches_design_ua(self):
        hot = self.ideal.calculate_state(
            500.0, 5.0, 100.0, {'N2': 0.79, 'O2': 0.21}, phase='vapor'
        )
        cold = self.ideal.calculate_state(
            300.0, 4.0, 100.0, {'N2': 0.79, 'O2': 0.21}, phase='vapor'
        )

        design = HeatExchanger(
            'HXD',
            self.ideal,
            {'Q': 20.0, 'curve_segments': 20},
        ).solve({'hot_in': hot, 'cold_in': cold})
        UA = design.performance['UA_required_W_per_K']

        counter = HeatExchanger(
            'HXR',
            self.ideal,
            {'UA': UA, 'curve_segments': 20},
        ).solve({'hot_in': hot, 'cold_in': cold})
        cocurrent = HeatExchanger(
            'HXC',
            self.ideal,
            {'UA': UA, 'curve_segments': 20, 'flow_pattern': 'cocurrent'},
        ).solve({'hot_in': hot, 'cold_in': cold})

        self.assertAlmostEqual(counter.performance['duty_kW'], 20.0, places=6)
        self.assertAlmostEqual(
            counter.performance['UA_required_W_per_K'],
            UA,
            delta=1e-5,
        )
        self.assertEqual(counter.performance['flow_pattern'], 'countercurrent')
        self.assertEqual(cocurrent.performance['flow_pattern'], 'cocurrent')
        self.assertLess(cocurrent.performance['duty_kW'], counter.performance['duty_kW'])

    def test_heat_exchanger_design_sizing_reports_from_optional_u_and_area(self):
        hot = self.ideal.calculate_state(
            500.0, 5.0, 100.0, {'N2': 0.79, 'O2': 0.21}, phase='vapor'
        )
        cold = self.ideal.calculate_state(
            300.0, 4.0, 100.0, {'N2': 0.79, 'O2': 0.21}, phase='vapor'
        )

        with_u = HeatExchanger(
            'HXU',
            self.ideal,
            {'Q': 20.0, 'U': 100.0, 'curve_segments': 20},
        ).solve({'hot_in': hot, 'cold_in': cold})
        UA = with_u.performance['UA_required_W_per_K']

        self.assertAlmostEqual(with_u.performance['area_required_m2'], UA / 100.0, places=8)

        with_area = HeatExchanger(
            'HXA',
            self.ideal,
            {'Q': 20.0, 'A': 2.0, 'curve_segments': 20},
        ).solve({'hot_in': hot, 'cold_in': cold})
        self.assertAlmostEqual(with_area.performance['U_required_W_m2_K'], UA / 2.0, places=8)

        with_both = HeatExchanger(
            'HXB',
            self.ideal,
            {'Q': 20.0, 'U': 100.0, 'A': 2.0, 'curve_segments': 20},
        ).solve({'hot_in': hot, 'cold_in': cold})
        self.assertAlmostEqual(with_both.performance['UA_available_W_per_K'], 200.0, places=8)
        self.assertAlmostEqual(
            with_both.performance['UA_margin_W_per_K'],
            200.0 - UA,
            places=8,
        )

    def test_heat_exchanger_can_defer_curve_metrics_during_recycle_iterations(self):
        hot = self.ideal.calculate_state(
            500.0, 5.0, 100.0, {'N2': 0.79, 'O2': 0.21}, phase='vapor'
        )
        cold = self.ideal.calculate_state(
            300.0, 4.0, 100.0, {'N2': 0.79, 'O2': 0.21}, phase='vapor'
        )

        deferred = HeatExchanger(
            'HXD',
            self.ideal,
            {'Q': 20.0, 'U': 100.0, 'A': 2.0, 'curve_segments': 20},
        )
        deferred.solve_context = {
            'recycle_evaluation': 3,
            'expensive_diagnostics': False,
        }
        deferred_result = deferred.solve({'hot_in': hot, 'cold_in': cold})

        self.assertTrue(deferred_result.performance['curve_metrics_delayed'])
        self.assertIsNone(deferred_result.performance['UA_required_W_per_K'])
        self.assertNotIn('area_required_m2', deferred_result.performance)
        self.assertNotIn('UA_margin_W_per_K', deferred_result.performance)
        self.assertNotIn('profile_hot_T_C', deferred_result.performance)
        self.assertAlmostEqual(deferred_result.performance['duty_kW'], 20.0, places=8)

        final = HeatExchanger(
            'HXF',
            self.ideal,
            {'Q': 20.0, 'U': 100.0, 'A': 2.0, 'curve_segments': 20},
        )
        final.solve_context = {
            'recycle_final_pass': True,
            'expensive_diagnostics': True,
        }
        final_result = final.solve({'hot_in': hot, 'cold_in': cold})

        self.assertFalse(final_result.performance['curve_metrics_delayed'])
        self.assertGreater(final_result.performance['UA_required_W_per_K'], 0.0)
        self.assertIn('area_required_m2', final_result.performance)
        self.assertIn('UA_margin_W_per_K', final_result.performance)
        self.assertEqual(len(final_result.performance['profile_hot_T_C']), 21)

    def test_heat_exchanger_auto_u_is_explicit_preliminary_estimate(self):
        hot = self.ideal.calculate_state(
            500.0, 5.0, 100.0, {'N2': 0.79, 'O2': 0.21}, phase='vapor'
        )
        cold = self.ideal.calculate_state(
            300.0, 4.0, 100.0, {'N2': 0.79, 'O2': 0.21}, phase='vapor'
        )

        no_auto = HeatExchanger(
            'HXN',
            self.ideal,
            {'Q': 20.0, 'curve_segments': 8},
        ).solve({'hot_in': hot, 'cold_in': cold})
        self.assertFalse(no_auto.performance['auto_U_estimation'])
        self.assertNotIn('auto_U_area_required_m2', no_auto.performance)

        auto = HeatExchanger(
            'HXA',
            self.ideal,
            {'Q': 20.0, 'U': 'auto', 'curve_segments': 8},
        ).solve({'hot_in': hot, 'cold_in': cold})
        self.assertTrue(auto.performance['auto_U_estimation'])
        self.assertGreater(auto.performance['auto_U_area_required_m2'], 0.0)
        self.assertEqual(auto.performance['auto_U_service_counts'], {'gas_gas': 8})
        self.assertTrue(any('auto-U estimates are preliminary' in warning for warning in auto.warnings))

        rating = HeatExchanger(
            'HXR',
            self.ideal,
            {'U': 'auto', 'A': auto.performance['auto_U_area_required_m2'], 'curve_segments': 8},
        ).solve({'hot_in': hot, 'cold_in': cold})
        self.assertAlmostEqual(rating.performance['duty_kW'], 20.0, places=5)

    def test_heat_exchanger_auto_u_uses_seader_liquid_service_classes(self):
        hot_water = self.ideal.calculate_state(
            360.0, 1.0, 1000.0, {'H2O': 1.0}, phase='liquid', flash=False
        )
        cold_water = self.ideal.calculate_state(
            300.0, 1.0, 1000.0, {'H2O': 1.0}, phase='liquid', flash=False
        )
        water = HeatExchanger(
            'HXW',
            self.ideal,
            {'Q': 10.0, 'U': 'auto', 'curve_segments': 8},
        ).solve({'hot_in': hot_water, 'cold_in': cold_water})
        self.assertEqual(water.performance['auto_U_service_counts'], {'demin_water_water': 8})
        self.assertAlmostEqual(
            water.performance['auto_U_segments'][0]['U_W_m2_K'],
            400.0 * 5.6783,
            places=6,
        )

        hot_ethanol = self.ideal.calculate_state(
            330.0, 1.0, 100.0, {'C2H5OH': 1.0}, phase='liquid', flash=False
        )
        solvent = HeatExchanger(
            'HXS',
            self.ideal,
            {'Q': 2.0, 'U': 'auto', 'curve_segments': 8},
        ).solve({'hot_in': hot_ethanol, 'cold_in': cold_water})
        self.assertEqual(solvent.performance['auto_U_service_counts'], {'organic_solvent_water': 8})
        self.assertAlmostEqual(
            solvent.performance['auto_U_segments'][0]['U_W_m2_K'],
            100.0 * 5.6783,
            places=6,
        )
        self.assertEqual(solvent.performance['auto_U_segments'][0]['source'], 'Seader Table 12.5')

    def test_heat_exchanger_auto_u_recognizes_dowtherm_mixture(self):
        thermo = create_thermodynamics(['biphenyl', 'diphenyl ether'], 'IDEAL')
        composition = {'biphenyl': 0.25, 'diphenyl ether': 0.75}
        unit = HeatExchanger('HXD', thermo, {})
        vapor = thermo.calculate_state(
            530.0, 1.0, 100.0, composition, phase='vapor', flash=False
        )
        partially_condensed = thermo.calculate_state(
            520.0, 1.0, 100.0, composition, phase='vapor', flash=False
        )
        partially_condensed.vapor_fraction = 0.8
        liquid = thermo.calculate_state(
            430.0, 1.0, 100.0, composition, phase='liquid', flash=False
        )
        warmer_liquid = thermo.calculate_state(
            450.0, 1.0, 100.0, composition, phase='liquid', flash=False
        )

        vapor_family = unit._stream_family(vapor, partially_condensed)
        self.assertEqual(vapor_family['family'], 'dowtherm')
        self.assertIn('diphenyl-ether fraction', vapor_family['basis'])

        service = unit._segment_service_info(
            vapor,
            partially_condensed,
            liquid,
            warmer_liquid,
        )
        self.assertEqual(service['service'], 'dowtherm_vapor_dowtherm_liquid')
        self.assertEqual(service['fallback_service'], 'condensing_vapor')

    def test_heat_exchanger_unifac_amine_detection_is_cached(self):
        unit = HeatExchanger('HXU', SimpleNamespace(props={}), {})
        props = SimpleNamespace(
            name='Grouped organic',
            symbol='grouped_organic',
            formula='C3H9N',
            smiles='CCCN',
        )

        with patch('unifac.get_unifac_groups', return_value={'CH2NH2': 1}) as groups:
            self.assertTrue(unit._component_has_amine_marker('', props))
            self.assertTrue(unit._component_has_amine_marker('', props))

        groups.assert_called_once_with('Grouped organic', smiles='CCCN')

    def test_heat_exchanger_native_numeric_unifac_markers_are_recognized(self):
        unit = HeatExchanger('HXU', SimpleNamespace(props={}), {})
        cases = (
            (
                SimpleNamespace(
                    name='Native numeric amine', symbol='NUM_AMINE',
                    formula='C2H7N', smiles='CCN',
                ),
                'amine',
                {1: 1, 29: 1},
            ),
            (
                SimpleNamespace(
                    name='Native numeric alcohol', symbol='NUM_ALCOHOL',
                    formula='C3H8O', smiles='CCCO',
                ),
                'alcohol',
                {1: 1, 2: 2, 14: 1},
            ),
            (
                SimpleNamespace(
                    name='Native numeric aromatic', symbol='NUM_AROMATIC',
                    formula='C6H6', smiles='c1ccccc1',
                ),
                'aromatic',
                {9: 6},
            ),
        )

        for props, marker, expected_groups in cases:
            with self.subTest(marker=marker):
                self.assertEqual(
                    unit._unifac_groups_for_component(props),
                    expected_groups,
                )
                self.assertIn(marker, unit._component_functional_markers(props))

    def test_heat_exchanger_seader_service_classifier_matrix(self):
        props = {
            'water': SimpleNamespace(name='Water', symbol='H2O', formula='H2O', MW=18.015, Tb=373.15),
            'ethanol': SimpleNamespace(name='Ethanol', symbol='C2H5OH', formula='C2H6O', MW=46.07, Tb=351.5),
            'grouped_alcohol': SimpleNamespace(name='Grouped oxygenate', symbol='grouped_alcohol', formula='C3H8O', MW=60.10, Tb=370.0, unifac_groups={'CH3': 1, 'CH2': 2, 'OH': 1}),
            'ethyl_acetate': SimpleNamespace(name='Ethyl acetate', symbol='C4H8O2', formula='C4H8O2', MW=88.11, Tb=350.2),
            'sodium_chloride': SimpleNamespace(name='Sodium chloride', symbol='NaCl', formula='NaCl', MW=58.44, Tb=None),
            'sodium_hydroxide': SimpleNamespace(name='Sodium hydroxide', symbol='NaOH', formula='NaOH', MW=40.0, Tb=None),
            'potassium_chloride': SimpleNamespace(name='Potassium chloride', symbol='KCl', formula='KCl', MW=74.55, Tb=None),
            'mea': SimpleNamespace(name='Monoethanolamine', symbol='MEA', formula='C2H7NO', MW=61.08, Tb=443.0),
            'amine_from_groups': SimpleNamespace(name='Grouped organic', symbol='grouped_amine', formula='C3H9N', MW=59.11, Tb=320.0, unifac_groups={'CH2NH2': 1, 'CH3': 1}),
            'nitrogen': SimpleNamespace(name='Nitrogen', symbol='N2', formula='N2', MW=28.01, Tb=77.4),
            'oxygen': SimpleNamespace(name='Oxygen', symbol='O2', formula='O2', MW=32.0, Tb=90.2),
            'hydrogen': SimpleNamespace(name='Hydrogen', symbol='H2', formula='H2', MW=2.016, Tb=20.3),
            'methane': SimpleNamespace(name='Methane', symbol='CH4', formula='CH4', MW=16.04, Tb=111.7),
            'propane': SimpleNamespace(name='Propane', symbol='C3H8', formula='C3H8', MW=44.1, Tb=231.0),
            'benzene': SimpleNamespace(name='Benzene', symbol='benzene', formula='C6H6', MW=78.11, Tb=353.2, unifac_groups={'ACH': 6}),
            'heptane': SimpleNamespace(name='Heptane', symbol='C7H16', formula='C7H16', MW=100.2, Tb=371.6),
            'nonane': SimpleNamespace(name='Nonane', symbol='C9H20', formula='C9H20', MW=128.3, Tb=423.9),
            'dodecane': SimpleNamespace(name='Dodecane', symbol='C12H26', formula='C12H26', MW=170.3, Tb=489.5),
            'eicosane': SimpleNamespace(name='Eicosane', symbol='C20H42', formula='C20H42', MW=282.6, Tb=616.9),
            'triacontane': SimpleNamespace(name='Triacontane', symbol='C30H62', formula='C30H62', MW=422.8, Tb=722.0),
            'fuel_oil': SimpleNamespace(name='No. 2 fuel oil', symbol='fuel_oil', formula='C14H30', MW=198.4, Tb=520.0),
            'lube_oil': SimpleNamespace(name='High viscosity lube oil', symbol='lube_oil', formula='C24H50', MW=338.7, Tb=680.0),
            'ammonia': SimpleNamespace(name='Ammonia', symbol='NH3', formula='NH3', MW=17.03, Tb=239.8),
            'chlorine': SimpleNamespace(name='Chlorine', symbol='Cl2', formula='Cl2', MW=70.9, Tb=239.1),
            'sulfur_dioxide': SimpleNamespace(name='Sulfur dioxide', symbol='SO2', formula='SO2', MW=64.1, Tb=263.1),
            'biphenyl': SimpleNamespace(name='Biphenyl', symbol='biphenyl', formula='C12H10', MW=154.2, Tb=529.0),
            'diphenyl_ether': SimpleNamespace(name='Diphenyl ether', symbol='diphenyl_ether', formula='C12H10O', MW=170.2, Tb=532.0),
        }
        unit = HeatExchanger('HXM', SimpleNamespace(props=props), {})

        def state(comp, vf=0.0, T=330.0, P=1.0):
            return StreamState(T=T, P=P, F=100.0, composition=comp, vapor_fraction=vf)

        liquid_cases = [
            ({'water': 0.9, 'sodium_hydroxide': 0.1}, {'water': 1.0}, 'water_caustic'),
            ({'water': 0.8, 'mea': 0.2}, {'water': 1.0}, 'amine_water'),
            ({'water': 0.8, 'amine_from_groups': 0.2}, {'water': 1.0}, 'amine_water'),
            ({'ethyl_acetate': 1.0}, {'water': 0.9, 'sodium_chloride': 0.1}, 'organic_solvent_brine'),
            ({'ethyl_acetate': 1.0}, {'water': 0.9, 'potassium_chloride': 0.1}, 'organic_solvent_brine'),
            ({'ethyl_acetate': 1.0}, {'ethanol': 1.0}, 'organic_solvent_organic'),
            ({'heptane': 1.0}, {'water': 1.0}, 'gasoline_water'),
            ({'nonane': 1.0}, {'water': 1.0}, 'naphtha_water'),
            ({'dodecane': 1.0}, {'water': 1.0}, 'kerosene_gas_oil_water'),
            # C30 melts near 339 K; keep the hot side above melting until
            # solid-phase treatment exists.
            ({'triacontane': 1.0}, {'water': 1.0}, 'heavy_oil_water', 353.0),
            ({'fuel_oil': 1.0}, {'water': 1.0}, 'fuel_oil_water'),
            ({'lube_oil': 1.0}, {'water': 1.0}, 'lube_oil_high_water'),
        ]
        for case in liquid_cases:
            hot_comp, cold_comp, expected = case[:3]
            hot_T_in = case[3] if len(case) > 3 else 330.0
            with self.subTest(service=expected):
                service = unit._segment_service_info(
                    state(hot_comp, vf=0.0, T=hot_T_in),
                    state(hot_comp, vf=0.0, T=hot_T_in - 5.0),
                    state(cold_comp, vf=0.0, T=300.0),
                    state(cold_comp, vf=0.0, T=305.0),
                )
                self.assertEqual(service['service'], expected)

        condensing_cases = [
            ({'water': 1.0}, {'water': 1.0}, 'steam_feedwater'),
            ({'water': 1.0}, {'fuel_oil': 1.0}, 'steam_no2_fuel_oil'),
            ({'ethanol': 1.0}, {'water': 1.0}, 'alcohol_vapor_water'),
            ({'grouped_alcohol': 1.0}, {'water': 1.0}, 'alcohol_vapor_water'),
            ({'benzene': 0.5, 'water': 0.5}, {'water': 1.0}, 'aromatic_vapor_water_azeotrope'),
            ({'propane': 1.0}, {'water': 1.0}, 'low_boiling_hydrocarbon_water'),
            ({'eicosane': 1.0}, {'water': 1.0}, 'high_boiling_hydrocarbon_water'),
            ({'sulfur_dioxide': 1.0}, {'water': 1.0}, 'sulfur_dioxide_water'),
            ({'biphenyl': 0.25, 'diphenyl_ether': 0.75}, {'biphenyl': 0.25, 'diphenyl_ether': 0.75}, 'dowtherm_vapor_dowtherm_liquid'),
        ]
        for hot_comp, cold_comp, expected in condensing_cases:
            with self.subTest(service=expected):
                service = unit._segment_service_info(
                    state(hot_comp, vf=1.0, T=530.0),
                    state(hot_comp, vf=0.7, T=520.0),
                    state(cold_comp, vf=0.0, T=300.0),
                    state(cold_comp, vf=0.0, T=320.0),
                )
                self.assertEqual(service['service'], expected)

        gas_liquid_cases = [
            ({'nitrogen': 0.79, 'oxygen': 0.21}, {'water': 1.0}, 5.0, 'compressed_air_water'),
            ({'nitrogen': 0.79, 'oxygen': 0.21}, {'water': 1.0}, 1.0, 'atmospheric_air_water'),
            ({'hydrogen': 0.7, 'methane': 0.3}, {'water': 1.0}, 5.0, 'water_hydrogen_natural_gas'),
        ]
        for gas_comp, liquid_comp, pressure, expected in gas_liquid_cases:
            with self.subTest(service=expected):
                service = unit._segment_service_info(
                    state(gas_comp, vf=1.0, T=360.0, P=pressure),
                    state(gas_comp, vf=1.0, T=350.0, P=pressure),
                    state(liquid_comp, vf=0.0, T=300.0),
                    state(liquid_comp, vf=0.0, T=310.0),
                )
                self.assertEqual(service['service'], expected)

        vaporizer_cases = [
            ({'ammonia': 1.0}, 'ammonia_steam_vaporizer'),
            ({'chlorine': 1.0}, 'chlorine_steam_vaporizer'),
            ({'propane': 1.0}, 'propane_butane_steam_vaporizer'),
            ({'water': 1.0}, 'water_steam_vaporizer'),
        ]
        for cold_comp, expected in vaporizer_cases:
            with self.subTest(service=expected):
                service = unit._segment_service_info(
                    state({'water': 1.0}, vf=1.0, T=450.0),
                    state({'water': 1.0}, vf=0.7, T=440.0),
                    state(cold_comp, vf=0.0, T=250.0),
                    state(cold_comp, vf=0.5, T=260.0),
                )
                self.assertEqual(service['service'], expected)

    def test_heat_exchanger_nonideal_phase_change_uses_curve_enthalpy(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'NRTL-RK')
        composition = {'ethanol': 0.5, 'water': 0.5}
        hot = thermo.calculate_state(
            390.0,
            1.01325,
            100.0,
            composition,
            phase='vapor',
            flash=False,
        )
        cold = thermo.calculate_state(
            300.0,
            1.01325,
            1000.0,
            composition,
            phase='liquid',
            flash=False,
        )

        result = HeatExchanger(
            'HXN',
            thermo,
            {'hot_vapor_fraction': 0.5, 'curve_segments': 8},
        ).solve({'hot_in': hot, 'cold_in': cold})
        hot_out = result.outlet_streams['hot_out']
        cold_out = result.outlet_streams['cold_out']

        self.assertGreater(hot.vapor_fraction, hot_out.vapor_fraction)
        self.assertAlmostEqual(hot_out.vapor_fraction, 0.5, places=8)
        self.assertGreater(result.performance['UA_required_W_per_K'], 0.0)
        self.assertAlmostEqual(
            hot.F * (hot.H - hot_out.H),
            cold.F * (cold_out.H - cold.H),
            delta=1e-3,
        )

    def test_heat_exchanger_temperature_specs_close_energy_balance(self):
        hot_spec = HeatExchanger(
            'HXH',
            self.ideal,
            {'T_hot_out': 360.0},
        ).solve({
            'hot_in': self.hot,
            'cold_in': self.cold,
        })
        hot_out = hot_spec.outlet_streams['hot_out']
        cold_out = hot_spec.outlet_streams['cold_out']
        Q_hot = self.hot.F * (self.hot.H - hot_out.H)
        Q_cold = self.cold.F * (cold_out.H - self.cold.H)

        self.assertAlmostEqual(hot_out.T, 360.0, places=8)
        self.assertAlmostEqual(hot_out.P, self.hot.P, places=8)
        self.assertAlmostEqual(cold_out.P, self.cold.P, places=8)
        self.assertAlmostEqual(Q_hot, Q_cold, delta=1e-3)

        cold_spec = HeatExchanger(
            'HXC',
            self.ideal,
            {'T_cold_out': 330.0},
        ).solve({
            'hot_in': self.hot,
            'cold_in': self.cold,
        })
        hot_out = cold_spec.outlet_streams['hot_out']
        cold_out = cold_spec.outlet_streams['cold_out']
        Q_hot = self.hot.F * (self.hot.H - hot_out.H)
        Q_cold = self.cold.F * (cold_out.H - self.cold.H)

        self.assertAlmostEqual(cold_out.T, 330.0, places=8)
        self.assertAlmostEqual(Q_hot, Q_cold, delta=1e-3)

    def test_heat_exchanger_supports_physical_side_temperature_specs(self):
        result = HeatExchanger(
            'HXT',
            self.ideal,
            {'T_tube_out': 330.0},
        ).solve({
            'tube_in': self.cold,
            'shell_in': self.hot,
        })

        tube_out = result.outlet_streams['tube_out']
        shell_out = result.outlet_streams['shell_out']
        Q_shell = self.hot.F * (self.hot.H - shell_out.H)
        Q_tube = self.cold.F * (tube_out.H - self.cold.H)

        self.assertAlmostEqual(tube_out.T, 330.0, places=8)
        self.assertLess(shell_out.T, self.hot.T)
        self.assertAlmostEqual(Q_shell, Q_tube, delta=1e-3)
        self.assertEqual(result.performance['hot_in_port'], 'shell_in')
        self.assertEqual(result.performance['cold_in_port'], 'tube_in')

    def test_heat_exchanger_rejects_bad_specs_and_pressure_drops(self):
        with self.assertRaisesRegex(UnitOperationError, 'exactly 2 inlet streams'):
            HeatExchanger('HX', self.ideal, {'Q': 1.0}).solve({'hot_in': self.hot})

        with self.assertRaisesRegex(UnitOperationError, 'rating mode requires'):
            HeatExchanger('HX', self.ideal, {}).solve({
                'hot_in': self.hot,
                'cold_in': self.cold,
            })

        with self.assertRaisesRegex(UnitOperationError, 'exactly one'):
            HeatExchanger('HX', self.ideal, {'T_hot_out': 360.0, 'Q': 1.0}).solve({
                'hot_in': self.hot,
                'cold_in': self.cold,
            })

        with self.assertRaisesRegex(UnitOperationError, 'rating mode requires'):
            HeatExchanger('HX', self.ideal, {'U': 500.0}).solve({
                'hot_in': self.hot,
                'cold_in': self.cold,
            })

        with self.assertRaisesRegex(UnitOperationError, 'nonnegative P_drop_tube'):
            HeatExchanger('HX', self.ideal, {'Q': 1.0, 'P_drop_tube': -0.1}).solve({
                'tube_in': self.hot,
                'shell_in': self.cold,
            })

        with self.assertRaisesRegex(UnitOperationError, 'outlet pressure'):
            HeatExchanger('HX', self.ideal, {'Q': 1.0, 'P_drop_hot': 3.0}).solve({
                'hot_in': self.hot,
                'cold_in': self.cold,
            })

    def test_heat_exchanger_rejects_wrong_heat_transfer_direction(self):
        with self.assertRaisesRegex(UnitOperationError, 'must transfer heat'):
            HeatExchanger('HX', self.ideal, {'T_hot_out': 440.0}).solve({
                'hot_in': self.hot,
                'cold_in': self.cold,
            })

        with self.assertRaisesRegex(UnitOperationError, 'must transfer heat'):
            HeatExchanger('HX', self.ideal, {'T_cold_out': 290.0}).solve({
                'hot_in': self.hot,
                'cold_in': self.cold,
            })

        with self.assertRaisesRegex(UnitOperationError, 'must transfer heat'):
            HeatExchanger('HX', self.ideal, {'Q': -5.0}).solve({
                'hot_in': self.hot,
                'cold_in': self.cold,
            })

    def test_pump_supports_pressure_specs_and_separates_work_accounting(self):
        result = Pump(
            'P',
            self.ideal,
            {'delta_P': 4.0, 'eta': 0.8, 'eta_mech': 0.9},
        ).solve({'in': self.liquid})
        outlet = result.outlet_streams['out']

        hydraulic_work = (1.0 / self.liquid.rho) * 4.0 * 100.0
        fluid_work = hydraulic_work / 0.8
        shaft_work = fluid_work / 0.9

        self.assertAlmostEqual(outlet.P, 5.0, places=8)
        self.assertEqual(result.performance['pressure_spec'], 'delta_P')
        self.assertEqual(result.performance['liquid_volume_basis'], 'inlet_stream_density')
        self.assertAlmostEqual(
            result.performance['hydraulic_work_kJ_per_kmol'],
            hydraulic_work,
            places=8,
        )
        self.assertAlmostEqual(
            result.performance['fluid_work_kJ_per_kmol'],
            fluid_work,
            places=8,
        )
        self.assertAlmostEqual(
            result.performance['shaft_work_kJ_per_kmol'],
            shaft_work,
            places=8,
        )
        self.assertAlmostEqual(result.work, self.liquid.F * fluid_work, places=8)
        self.assertAlmostEqual(
            result.performance['shaft_power_kW'],
            self.liquid.F * shaft_work / 3600.0,
            places=8,
        )
        self.assertAlmostEqual(
            outlet.H - self.liquid.H,
            fluid_work,
            delta=1e-5,
        )

        ratio = Pump('P2', self.ideal, {'pressure_ratio': 3.0}).solve({'in': self.liquid})
        self.assertAlmostEqual(ratio.outlet_streams['out'].P, 3.0, places=8)
        self.assertEqual(ratio.performance['pressure_spec'], 'pressure_ratio')

    def test_pump_rejects_invalid_pressure_efficiency_or_phase_specs(self):
        with self.assertRaisesRegex(UnitOperationError, 'only one pressure target'):
            Pump('P', self.ideal, {'P_out': 5.0, 'delta_P': 2.0}).solve({'in': self.liquid})

        with self.assertRaisesRegex(UnitOperationError, 'greater than inlet pressure'):
            Pump('P', self.ideal, {'P_out': 0.5}).solve({'in': self.liquid})

        with self.assertRaisesRegex(UnitOperationError, 'pressure_ratio > 1'):
            Pump('P', self.ideal, {'pressure_ratio': 1.0}).solve({'in': self.liquid})

        with self.assertRaisesRegex(UnitOperationError, 'hydraulic efficiency'):
            Pump('P', self.ideal, {'P_out': 5.0, 'eta': 0.0}).solve({'in': self.liquid})

        wet_feed = self.liquid.copy()
        wet_feed.vapor_fraction = 0.02
        with self.assertRaisesRegex(UnitOperationError, 'pumps require liquid feed'):
            Pump('P', self.ideal, {'P_out': 5.0}).solve({'in': wet_feed})

    def test_compressor_uses_entropy_based_isentropic_efficiency(self):
        components = ['N2', 'O2', 'NH3', 'CH4', 'H2O', 'AR', 'H2']
        flows = {
            'N2': 1347.537,
            'O2': 3.220,
            'NH3': 2.401,
            'CH4': 45.018,
            'H2O': 0.0,
            'AR': 40.996,
            'H2': 4184.463,
        }
        total_flow = 5623.635
        composition = {comp: flows[comp] / total_flow for comp in components}
        thermo = create_thermodynamics(components, 'RKS-BM')
        inlet = thermo.calculate_state(
            64.8 + 273.15,
            10.0,
            total_flow,
            composition,
            phase='vapor',
            flash=False,
        )

        result = Compressor(
            'C',
            thermo,
            {'P_out': 100.0, 'eta_isen': 0.8, 'eta_mech': 0.95},
        ).solve({'in': inlet})
        outlet = result.outlet_streams['out']

        expected_fluid_power_kW = 18167.0
        expected_shaft_power_kW = expected_fluid_power_kW / 0.95
        power_tolerance = 0.0002

        self.assertAlmostEqual(outlet.T - 273.15, 452.5, delta=0.1)
        self.assertAlmostEqual(
            result.work / 3600.0,
            expected_fluid_power_kW,
            delta=expected_fluid_power_kW * power_tolerance,
        )
        self.assertAlmostEqual(
            result.performance['shaft_power_kW'],
            expected_shaft_power_kW,
            delta=expected_shaft_power_kW * power_tolerance,
        )
        self.assertAlmostEqual(result.performance['T_isentropic_C'], 374.87, delta=0.05)
        self.assertGreater(result.performance['entropy_generation_kJ_per_kmol_K'], 0.0)

    def test_compressor_and_expander_pressure_specs_and_work_accounting(self):
        compressed = Compressor(
            'C',
            self.ideal,
            {'pressure_ratio': 3.0, 'eta_isen': 0.8, 'eta_mech': 0.9},
        ).solve({'in': self.gas})
        self.assertAlmostEqual(compressed.outlet_streams['out'].P, 3.0, places=8)
        self.assertEqual(compressed.performance['pressure_spec'], 'pressure_ratio')
        self.assertAlmostEqual(
            compressed.performance['shaft_power_kW'],
            compressed.work / 3600.0 / 0.9,
            places=8,
        )

        expanded = Expander(
            'E',
            self.ideal,
            {'pressure_ratio': 2.0, 'eta_isen': 1.0, 'eta_mech': 0.9},
        ).solve({'in': self.gas})
        outlet = expanded.outlet_streams['out']
        self.assertAlmostEqual(outlet.P, 0.5, places=8)
        self.assertEqual(expanded.performance['pressure_spec'], 'pressure_ratio')
        self.assertLess(expanded.work, 0.0)
        self.assertAlmostEqual(outlet.S, self.gas.S, places=6)
        self.assertAlmostEqual(
            expanded.performance['power_recovered_kW'],
            -expanded.work / 3600.0 * 0.9,
            places=8,
        )

    def test_expander_handles_pure_component_wet_outlet(self):
        steam_flow = 59.02 * 3600.0 / 18.01528

        for method in ('IDEAL', 'PR'):
            thermo = create_thermodynamics(['H2O'], method)
            for inlet_T_C in (300.0, 400.0, 500.0):
                with self.subTest(method=method, inlet_T_C=inlet_T_C):
                    inlet = thermo.calculate_state(
                        inlet_T_C + 273.15,
                        86.0,
                        steam_flow,
                        {'H2O': 1.0},
                        phase='vapor',
                        flash=False,
                    )
                    result = Expander(
                        'E',
                        thermo,
                        {'P_out': 0.1, 'eta_isen': 0.75, 'eta_mech': 1.0},
                    ).solve({'in': inlet})
                    outlet = result.outlet_streams['out']

                    try:
                        k_residual = thermo.K_value('H2O', outlet.T, outlet.P, outlet.composition) - 1.0
                    except TypeError:
                        k_residual = thermo.K_value('H2O', outlet.T, outlet.P) - 1.0
                    saturated_liquid = thermo.calculate_state(
                        outlet.T,
                        outlet.P,
                        steam_flow,
                        {'H2O': 1.0},
                        phase='liquid',
                        flash=False,
                    )
                    saturated_vapor = thermo.calculate_state(
                        outlet.T,
                        outlet.P,
                        steam_flow,
                        {'H2O': 1.0},
                        phase='vapor',
                        flash=False,
                    )
                    H_low = min(saturated_liquid.H, saturated_vapor.H)
                    H_high = max(saturated_liquid.H, saturated_vapor.H)

                    self.assertAlmostEqual(k_residual, 0.0, delta=1e-5)
                    self.assertGreater(outlet.vapor_fraction, 0.0)
                    self.assertLess(outlet.vapor_fraction, 1.0)
                    self.assertGreaterEqual(outlet.H, H_low - 1e-6)
                    self.assertLessEqual(outlet.H, H_high + 1e-6)
                    self.assertGreaterEqual(
                        result.performance['entropy_generation_kJ_per_kmol_K'],
                        -1e-8,
                    )
                    self.assertAlmostEqual(
                        result.performance['enthalpy_residual_kJ_per_kmol'],
                        0.0,
                        delta=1e-8,
                    )
                    self.assertLess(result.work, 0.0)
                    self.assertGreater(result.performance['power_recovered_kW'], 0.0)

    def test_expander_keeps_superheated_ethylene_vapor(self):
        for method, expected_T_C, expected_power_kW in (
            ('IDEAL', 96.7471755, 337.5478),
            ('PR', 90.7494462, 329.4862),
        ):
            with self.subTest(method=method):
                thermo = create_thermodynamics(['C2H4'], method)
                inlet = thermo.calculate_state(
                    300.0 + 273.15,
                    45.0,
                    100.0,
                    {'C2H4': 1.0},
                    phase='vapor',
                    flash=False,
                )
                result = Expander(
                    'E',
                    thermo,
                    {'P_out': 2.0, 'eta_isen': 1.0, 'eta_mech': 1.0},
                ).solve({'in': inlet})
                outlet = result.outlet_streams['out']

                self.assertAlmostEqual(outlet.T - 273.15, expected_T_C, places=3)
                self.assertAlmostEqual(outlet.vapor_fraction, 1.0, places=8)
                self.assertAlmostEqual(
                    result.performance['power_recovered_kW'],
                    expected_power_kW,
                    places=3,
                )
                self.assertAlmostEqual(outlet.S, inlet.S, places=6)

    def test_expander_handles_multicomponent_wet_outlet_with_activity_eos(self):
        mw_ethanol = 46.06844
        mw_water = 18.01528
        n_ethanol = 0.5 / mw_ethanol
        n_water = 0.5 / mw_water
        composition = {
            'ethanol': n_ethanol / (n_ethanol + n_water),
            'water': n_water / (n_ethanol + n_water),
        }
        thermo = create_thermodynamics(['ethanol', 'water'], 'NRTL-RK')
        inlet = thermo.calculate_state(
            180.0 + 273.15,
            10.0,
            100.0,
            composition,
        )
        result = Expander(
            'E',
            thermo,
            {'P_out': 1.01325, 'eta_isen': 0.8, 'eta_mech': 1.0},
        ).solve({'in': inlet})
        outlet = result.outlet_streams['out']
        T_bubble = thermo.bubble_point_T(composition, 1.01325, 350.0)
        T_dew = thermo.dew_point_T(composition, 1.01325, 350.0)

        self.assertAlmostEqual(inlet.vapor_fraction, 1.0, places=8)
        self.assertGreater(outlet.vapor_fraction, 0.0)
        self.assertLess(outlet.vapor_fraction, 1.0)
        self.assertGreaterEqual(outlet.T, min(T_bubble, T_dew) - 1e-6)
        self.assertLessEqual(outlet.T, max(T_bubble, T_dew) + 1e-6)
        self.assertNotAlmostEqual(outlet.x['ethanol'], outlet.y['ethanol'], places=3)
        self.assertGreater(
            result.performance['entropy_generation_kJ_per_kmol_K'],
            0.0,
        )
        self.assertAlmostEqual(
            result.performance['enthalpy_residual_kJ_per_kmol'],
            0.0,
            delta=5e-4,
        )
        self.assertLess(result.work, 0.0)

    def test_compressor_and_expander_reject_invalid_specs(self):
        with self.assertRaisesRegex(UnitOperationError, 'only one pressure target'):
            Compressor('C', self.ideal, {'P_out': 5.0, 'pressure_ratio': 2.0}).solve({'in': self.gas})

        with self.assertRaisesRegex(UnitOperationError, 'greater than inlet pressure'):
            Compressor('C', self.ideal, {'P_out': 0.8}).solve({'in': self.gas})

        with self.assertRaisesRegex(UnitOperationError, 'less than inlet pressure'):
            Expander('E', self.ideal, {'P_out': 2.0}).solve({'in': self.gas})

        with self.assertRaisesRegex(UnitOperationError, 'polytropic efficiency'):
            Compressor('C', self.ideal, {'P_out': 5.0, 'eta_poly': 0.75}).solve({'in': self.gas})

    def test_pfr_reports_pump_utility_and_process_work_separately(self):
        simulator = Simulator.from_string(
            'PROCESS: Pump Work Report\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            '\n'
            'COMPONENTS:\n'
            '    H2O | Water | MW=18.015\n'
            '\n'
            'STREAM Feed : FEED -> P-1.in\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 100 [kmol/h]\n'
            '    x = H2O:1.0\n'
            '\n'
            'STREAM Product : P-1.out -> PRODUCT\n'
            '\n'
            'UNIT P-1\n'
            '    TYPE: Pump\n'
            '    PORTS:\n'
            '        in : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '        delta_P = 4 [bar]\n'
            '        eta = 0.8\n'
            '        eta_mech = 0.5\n'
        )
        simulator.run()
        pfr = simulator._generate_pfr()
        process_work = simulator.result.units['P-1'].work / 3600.0
        utility_work = simulator.result.units['P-1'].performance['shaft_power_kW']

        self.assertIn(f'total_work = {utility_work:.2f} [kW]', pfr)
        self.assertIn(f'total_process_work = {process_work:.2f} [kW]', pfr)
        self.assertIn(f'    work = {utility_work:.4f} [kW]', pfr)
        self.assertIn(f'    process_work = {process_work:.4f} [kW]', pfr)
        self.assertGreater(utility_work, process_work)

    def test_splitter_supports_named_outlet_flow_specs(self):
        result = Splitter(
            'S',
            self.ideal,
            {
                'outlets': 'product, purge, recycle',
                'product_flow': 30.0,
                'purge_split_frac': 0.2,
                'P_product': 0.9,
                'P_drop_purge': 10.0,
                '__unit__P_drop_purge': 'kPa',
            },
        ).solve({'in': self.liquid})

        product = result.outlet_streams['product']
        purge = result.outlet_streams['purge']
        recycle = result.outlet_streams['recycle']

        self.assertAlmostEqual(product.F, 30.0, places=8)
        self.assertAlmostEqual(purge.F, 20.0, places=8)
        self.assertAlmostEqual(recycle.F, 50.0, places=8)
        self.assertAlmostEqual(product.P, 0.9, places=8)
        self.assertAlmostEqual(purge.P, 0.9, places=8)
        self.assertLess(relative_component_balance([self.liquid], [product, purge, recycle]), 1e-12)
        self.assertEqual(result.performance['mode'], 'bulk_split')

    def test_splitter_supports_component_split_placeholder_separator(self):
        result = Splitter(
            'SEP',
            self.ideal,
            {
                'outlets': 'top, bottom',
                'component_splits': 'C2H5OH:top=0.9; H2O:top=0.1',
            },
        ).solve({'in': self.liquid})

        top = result.outlet_streams['top']
        bottom = result.outlet_streams['bottom']

        self.assertAlmostEqual(top.F, 26.0, places=8)
        self.assertAlmostEqual(bottom.F, 74.0, places=8)
        self.assertAlmostEqual(top.composition['C2H5OH'], 18.0 / 26.0, places=8)
        self.assertAlmostEqual(bottom.composition['C2H5OH'], 2.0 / 74.0, places=8)
        self.assertLess(relative_component_balance([self.liquid], [top, bottom]), 1e-12)
        self.assertEqual(result.performance['mode'], 'component_split')

    def test_splitter_supports_mass_flow_and_outlet_state_specs(self):
        target_mass_flow = self.liquid.mass_flow() * 0.25
        result = Splitter(
            'S',
            self.ideal,
            {
                'outlets': 'product, reject',
                'mass_flow_product': target_mass_flow,
                '__unit__mass_flow_product': 'kg/h',
                'T_product': 35.0,
                'P_product': 1.0,
                '__unit__P_product': 'atm',
                'phase_product': 'liquid',
            },
        ).solve({'in': self.liquid})

        product = result.outlet_streams['product']
        reject = result.outlet_streams['reject']

        self.assertAlmostEqual(product.F, self.liquid.F * 0.25, places=8)
        self.assertAlmostEqual(reject.F, self.liquid.F * 0.75, places=8)
        self.assertAlmostEqual(product.mass_flow(), target_mass_flow, places=8)
        self.assertAlmostEqual(product.T, 308.15, places=8)
        self.assertAlmostEqual(product.P, 1.01325, places=8)
        self.assertAlmostEqual(product.vapor_fraction, 0.0, places=8)
        expected_duty = (
            product.F * product.H
            + reject.F * reject.H
            - self.liquid.F * self.liquid.H
        )
        self.assertAlmostEqual(result.heat_duty, expected_duty, places=8)
        self.assertAlmostEqual(
            result.performance['implicit_state_change_duty_kW'],
            expected_duty / 3600,
            places=8,
        )
        self.assertLess(relative_component_balance([self.liquid], [product, reject]), 1e-12)

    def test_splitter_rejects_invalid_component_split_specs(self):
        with self.assertRaisesRegex(UnitOperationError, 'unknown component'):
            Splitter(
                'S',
                self.ideal,
                {
                    'outlets': 'top, bottom',
                    'component_splits': 'ethanol:top=0.9',
                },
            ).solve({'in': self.liquid})

        with self.assertRaisesRegex(UnitOperationError, 'unknown outlet'):
            Splitter(
                'S',
                self.ideal,
                {
                    'outlets': 'top, bottom',
                    'component_splits': 'C2H5OH:vent=0.5',
                },
            ).solve({'in': self.liquid})

        with self.assertRaisesRegex(UnitOperationError, 'exceed 1'):
            Splitter(
                'S',
                self.ideal,
                {
                    'outlets': 'top, bottom',
                    'component_splits': 'C2H5OH:top=0.7,bottom=0.4',
                },
            ).solve({'in': self.liquid})

    def test_splitter_requires_exactly_one_inlet(self):
        with self.assertRaisesRegex(UnitOperationError, 'no inlet stream'):
            Splitter('S', self.ideal, {}).solve({})

        with self.assertRaisesRegex(UnitOperationError, 'exactly one inlet stream'):
            Splitter('S', self.ideal, {}).solve({
                'in1': self.liquid,
                'in2': self.liquid2,
            })

    def test_valve_throttles_hot_ethanol_isenthalpically(self):
        thermo = create_thermodynamics(['C2H5OH'], 'UNIFAC-RK')
        feed = thermo.calculate_state(
            413.15, 10.0, 100.0, {'C2H5OH': 1.0}, phase='liquid'
        )
        result = Valve('V', thermo, {'P_out': 1.0}).solve({'in': feed})
        outlet = result.outlet_streams['out']

        self.assertAlmostEqual(outlet.P, 1.0, places=8)
        self.assertLess(outlet.T, feed.T - 10.0)
        self.assertGreater(outlet.vapor_fraction, 0.01)
        self.assertLess(outlet.vapor_fraction, 0.99)
        self.assertLess(abs(outlet.H - feed.H), 1e-6)
        self.assertEqual(result.warnings, [])

    def test_valve_handles_multicomponent_wet_isenthalpic_flash(self):
        mw_ethanol = 46.06844
        mw_water = 18.01528
        n_ethanol = 0.5 / mw_ethanol
        n_water = 0.5 / mw_water
        composition = {
            'ethanol': n_ethanol / (n_ethanol + n_water),
            'water': n_water / (n_ethanol + n_water),
        }
        thermo = create_thermodynamics(['ethanol', 'water'], 'NRTL-RK')
        feed = thermo.calculate_state(
            120.0 + 273.15,
            10.0,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )

        result = Valve('V', thermo, {'P_out': 1.01325}).solve({'in': feed})
        outlet = result.outlet_streams['out']
        T_bubble = thermo.bubble_point_T(composition, 1.01325, 350.0)
        T_dew = thermo.dew_point_T(composition, 1.01325, T_bubble)

        self.assertGreater(outlet.vapor_fraction, 0.0)
        self.assertLess(outlet.vapor_fraction, 1.0)
        self.assertGreaterEqual(outlet.T, min(T_bubble, T_dew) - 1e-6)
        self.assertLessEqual(outlet.T, max(T_bubble, T_dew) + 1e-6)
        self.assertNotAlmostEqual(outlet.x['ethanol'], outlet.y['ethanol'], places=3)
        self.assertAlmostEqual(outlet.H, feed.H, delta=1e-4)
        self.assertAlmostEqual(
            result.performance['enthalpy_error_kJ_per_kmol'],
            0.0,
            delta=1e-4,
        )
        self.assertAlmostEqual(
            outlet.rho,
            thermo.mixture_molar_density(
                outlet.composition,
                outlet.T,
                outlet.P,
                outlet.vapor_fraction,
                outlet.x,
                outlet.y,
            ),
            places=10,
        )
        self.assertEqual(result.warnings, [])

    def test_shortcut_distillation_handles_zero_feed_recycle_startup(self):
        thermo = create_thermodynamics(['C2H4', 'N2'], 'IDEAL')
        feed = thermo.calculate_state(
            313.15,
            19.0,
            0.0,
            {'C2H4': 0.99, 'N2': 0.01},
            phase='liquid',
            flash=False,
            include=('H', 'Cp', 'S'),
        )

        result = ShortcutDistillation(
            'D',
            thermo,
            {'N_stages': 25, 'reflux_ratio': 3.5, 'P_top': 2.0},
        ).solve({'feed': feed})

        self.assertTrue(result.performance['zero_feed'])
        self.assertEqual(result.outlet_streams['distillate'].F, 0.0)
        self.assertEqual(result.outlet_streams['bottoms'].F, 0.0)
        self.assertEqual(result.heat_duty, 0.0)
        self.assertIn('zero feed', result.warnings[0])

    def test_cooler_respects_parsed_cryogenic_celsius_temperature(self):
        thermo = create_thermodynamics(['N2', 'O2', 'Ar'], 'RKS-BM')
        feed = thermo.calculate_state(
            303.15,
            1.01325,
            407.2,
            {'N2': 0.78, 'O2': 0.21, 'Ar': 0.01},
            phase='vapor',
            flash=False,
        )
        result = Cooler(
            'E',
            thermo,
            {'T_out': 83.15, '__unit__T_out': 'C', 'P_drop': 0.01325},
        ).solve({'in': feed})
        outlet = result.outlet_streams['out']

        self.assertAlmostEqual(outlet.T, 83.15, places=8)
        self.assertAlmostEqual(outlet.T - 273.15, -190.0, places=8)
        self.assertLess(result.heat_duty, 0.0)

    def test_heater_and_cooler_reject_overspecified_heat_specs(self):
        with self.assertRaisesRegex(UnitOperationError, 'exactly one'):
            Heater('H', self.ideal, {'T_out': 350.0, 'Q': 10.0}).solve({
                'in': self.liquid,
            })
        with self.assertRaisesRegex(UnitOperationError, 'exactly one'):
            Cooler('C', self.ideal, {'T_out': 290.0, 'vap_frac': 0.0}).solve({
                'in': self.liquid,
            })

    def test_heater_and_cooler_enforce_heat_direction(self):
        with self.assertRaisesRegex(UnitOperationError, 'must add heat'):
            Heater('H', self.ideal, {'T_out': 290.0}).solve({'in': self.liquid})
        with self.assertRaisesRegex(UnitOperationError, 'must remove heat'):
            Cooler('C', self.ideal, {'T_out': 350.0}).solve({'in': self.liquid})
        with self.assertRaisesRegex(UnitOperationError, 'must add heat'):
            Heater('H', self.ideal, {'Q': -5.0}).solve({'in': self.liquid})
        with self.assertRaisesRegex(UnitOperationError, 'must remove heat'):
            Cooler('C', self.ideal, {'Q': 5.0}).solve({'in': self.liquid})

    def test_heater_and_cooler_validate_pressure_drop(self):
        with self.assertRaisesRegex(UnitOperationError, 'nonnegative P_drop'):
            Heater('H', self.ideal, {'T_out': 350.0, 'P_drop': -0.1}).solve({
                'in': self.liquid,
            })
        with self.assertRaisesRegex(UnitOperationError, 'outlet pressure must be positive'):
            Cooler('C', self.ideal, {'T_out': 290.0, 'P_drop': 2.0}).solve({
                'in': self.liquid,
            })

    def test_heater_vapor_fraction_alias_uses_direct_pq_for_steam(self):
        thermo = create_thermodynamics(['H2O'], 'STEAM')
        feed = thermo.calculate_state_PQ(0.1, 0.0, 100.0, {'H2O': 1.0})
        calls = {'PQ': 0}
        original_pq = thermo.calculate_state_PQ

        def counted_pq(*args, **kwargs):
            calls['PQ'] += 1
            return original_pq(*args, **kwargs)

        thermo.calculate_state_PQ = counted_pq

        result = Heater('H', thermo, {'vapor_fraction': 1.0}).solve({'in': feed})
        outlet = result.outlet_streams['out']

        self.assertEqual(calls['PQ'], 1)
        self.assertEqual(outlet.vapor_fraction, 1.0)
        self.assertGreater(result.heat_duty, 0.0)

    def test_heater_duty_uses_direct_ph_for_steam(self):
        thermo = create_thermodynamics(['H2O'], 'STEAM')
        feed = thermo.calculate_state_PQ(1.0, 0.0, 100.0, {'H2O': 1.0})
        calls = {'PH': 0}
        original_ph = thermo.calculate_state_PH

        def counted_ph(*args, **kwargs):
            calls['PH'] += 1
            return original_ph(*args, **kwargs)

        thermo.calculate_state_PH = counted_ph

        result = Heater('H', thermo, {'Q': 1.0, '__unit__Q': 'MW'}).solve({'in': feed})
        outlet = result.outlet_streams['out']

        self.assertEqual(calls['PH'], 1)
        self.assertAlmostEqual(result.heat_duty, 3.6e6, places=6)
        self.assertAlmostEqual(
            outlet.H,
            feed.H + result.heat_duty / feed.F,
            places=6,
        )
        self.assertAlmostEqual(outlet.T, feed.T, places=8)
        self.assertGreater(outlet.vapor_fraction, 0.0)
        self.assertLess(outlet.vapor_fraction, 1.0)

    def test_heater_duty_uses_seeded_newton_before_bracketing(self):
        thermo = create_thermodynamics(['N2', 'O2'], 'IDEAL')
        composition = {'N2': 0.79, 'O2': 0.21}
        feed = thermo.calculate_state(
            330.0, 5.0, 100.0, composition, phase='vapor'
        )
        target = thermo.calculate_state(
            450.0, 5.0, 100.0, composition, phase='vapor'
        )
        duty = feed.F * (target.H - feed.H)

        original_brentq = basic_ops.brentq

        def fail_brentq(*_args, **_kwargs):
            raise AssertionError("PH inversion should use seeded Newton before bracketing")

        basic_ops.brentq = fail_brentq
        try:
            result = Heater(
                'H',
                thermo,
                {'Q': duty, '__unit__Q': 'kJ/h'},
            ).solve({'in': feed})
        finally:
            basic_ops.brentq = original_brentq

        outlet = result.outlet_streams['out']
        self.assertAlmostEqual(outlet.T, target.T, places=8)
        self.assertAlmostEqual(outlet.H, target.H, places=7)

    def test_methanol_water_shortcut_and_cmo_distillation_respect_specs(self):
        thermo = create_thermodynamics(['CH3OH', 'H2O'], 'UNIFAC')
        methanol_moles = 300.0 / thermo.props['CH3OH'].MW
        water_moles = 700.0 / thermo.props['H2O'].MW
        total_moles = methanol_moles + water_moles
        composition = {
            'CH3OH': methanol_moles / total_moles,
            'H2O': water_moles / total_moles,
        }
        T_feed = thermo.bubble_point_T(composition, 1.0)
        feed = thermo.calculate_state(
            T_feed, 1.0, total_moles, composition, phase='liquid'
        )

        cmo = McCabeThieleDistillation(
            'U', thermo,
            {
                'N_stages': 15,
                'reflux_ratio': 1.5,
                'distillate_flow': 300.0,
                '__unit__distillate_flow': 'kg/h',
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
            },
        ).solve({'feed': feed})

        cmo_distillate = cmo.outlet_streams['distillate']
        cmo_bottoms = cmo.outlet_streams['bottoms']

        self.assertAlmostEqual(cmo_distillate.mass_flow(), 300.0, delta=1.0)
        self.assertGreater(cmo_distillate.composition['CH3OH'], feed.composition['CH3OH'])
        self.assertLess(cmo_bottoms.composition['CH3OH'], feed.composition['CH3OH'])
        self.assertGreater(cmo_distillate.T - 273.15, 64.0)
        self.assertLess(cmo_distillate.T - 273.15, 66.5)
        self.assertGreater(cmo_bottoms.T - 273.15, 98.0)
        self.assertLess(cmo_bottoms.T - 273.15, 101.0)
        self.assertNotIn('feed_stage_auto', cmo.performance)
        self.assertGreaterEqual(cmo.performance['selected_feed_stage'], 1)
        self.assertLessEqual(cmo.performance['selected_feed_stage'], 15)
        self.assertLess(
            relative_component_balance(
                [feed],
                [cmo_distillate, cmo_bottoms],
            ),
            1e-8,
        )

        shortcut = ShortcutDistillation(
            'U', thermo,
            {
                'N_stages': 15,
                'reflux_ratio': 1.5,
                'distillate_flow': 300.0,
                '__unit__distillate_flow': 'kg/h',
                'P_condenser': 1.0,
            },
        ).solve({'feed': feed})
        shortcut_distillate = shortcut.outlet_streams['distillate']
        shortcut_bottoms = shortcut.outlet_streams['bottoms']

        self.assertAlmostEqual(shortcut_distillate.mass_flow(), 300.0, delta=1.0)
        self.assertGreater(shortcut_distillate.composition['CH3OH'], feed.composition['CH3OH'])
        self.assertLess(shortcut_bottoms.composition['CH3OH'], feed.composition['CH3OH'])
        self.assertEqual(shortcut.performance['method'], 'FUG')
        self.assertAlmostEqual(shortcut.performance['theoretical_stages'], 15.0, places=5)
        self.assertLess(abs(shortcut.performance['stage_margin']), 1e-4)
        self.assertGreaterEqual(shortcut.performance['feed_stage'], 1)
        self.assertLessEqual(shortcut.performance['feed_stage'], 15)

        low_reflux = ShortcutDistillation(
            'U', thermo,
            {
                'N_stages': 15,
                'reflux_ratio': 0.5,
                'distillate_flow': 300.0,
                '__unit__distillate_flow': 'kg/h',
                'P_condenser': 1.0,
            },
        ).solve({'feed': feed})
        high_reflux = ShortcutDistillation(
            'U', thermo,
            {
                'N_stages': 15,
                'reflux_ratio': 5.0,
                'distillate_flow': 300.0,
                '__unit__distillate_flow': 'kg/h',
                'P_condenser': 1.0,
            },
        ).solve({'feed': feed})

        self.assertGreater(
            high_reflux.outlet_streams['distillate'].composition['CH3OH'],
            low_reflux.outlet_streams['distillate'].composition['CH3OH'],
        )

    def test_shortcut_distillation_fug_recovery_mode_satisfies_formula_identities(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        composition = {'methanol': 0.3, 'water': 0.7}
        T_feed = thermo.bubble_point_T(composition, 1.0)
        feed = thermo.calculate_state(
            T_feed, 1.0, 100.0, composition, phase='liquid', flash=False
        )
        result = ShortcutDistillation(
            'S', thermo,
            {
                'light_key': 'methanol',
                'heavy_key': 'water',
                'light_key_recovery_distillate': 0.95,
                'heavy_key_recovery_bottoms': 0.95,
                'reflux_ratio': 2.0,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
            },
        ).solve({'feed': feed})

        perf = result.performance
        distillate = result.outlet_streams['distillate']
        bottoms = result.outlet_streams['bottoms']
        alpha = perf['relative_volatilities']
        theta = perf['underwood_theta']
        fenske = math.log(
            distillate.composition['methanol'] / bottoms.composition['methanol']
            * bottoms.composition['water'] / distillate.composition['water']
        ) / math.log(alpha['methanol'] / alpha['water'])
        underwood_feed = sum(
            alpha[comp] * composition[comp] / (alpha[comp] - theta)
            for comp in composition
        )
        underwood_reflux = sum(
            alpha[comp] * distillate.composition[comp] / (alpha[comp] - theta)
            for comp in composition
        ) - 1.0

        self.assertAlmostEqual(perf['N_min'], fenske, places=10)
        self.assertAlmostEqual(underwood_feed, 0.0, places=10)
        self.assertAlmostEqual(perf['R_min'], underwood_reflux, places=10)
        self.assertEqual(perf['recovery_spec_closure'], 'key_recoveries')
        self.assertAlmostEqual(perf['light_key_recovery_distillate'], 0.95, places=10)
        self.assertAlmostEqual(perf['heavy_key_recovery_bottoms'], 0.95, places=10)
        self.assertLess(relative_component_balance([feed], [distillate, bottoms]), 1e-10)

    def test_shortcut_distillation_backsolves_recoveries_from_stages_reflux_and_cut(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        composition = {'methanol': 0.3, 'water': 0.7}
        feed = thermo.calculate_state(
            thermo.bubble_point_T(composition, 1.0),
            1.0,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        result = ShortcutDistillation(
            'S', thermo,
            {
                'N_stages': 20,
                'reflux_ratio': 1.0,
                'D_to_F': 0.3,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
            },
        ).solve({'feed': feed})

        self.assertAlmostEqual(result.outlet_streams['distillate'].F, 30.0, places=8)
        self.assertAlmostEqual(result.performance['theoretical_stages'], 20.0, places=4)
        self.assertLess(abs(result.performance['stage_margin']), 1e-4)
        self.assertEqual(
            result.performance['recovery_spec_closure'],
            'stages_reflux_distillate_backsolved_key_recoveries',
        )
        self.assertGreater(
            result.performance['light_key_recovery_distillate'],
            result.performance['heavy_key_recovery_distillate'],
        )
        self.assertLess(
            relative_component_balance(
                [feed],
                [result.outlet_streams['distillate'], result.outlet_streams['bottoms']],
            ),
            1e-10,
        )

    def test_shortcut_distillation_backsolve_handles_minimum_reflux_asymptote(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        composition = {'methanol': 0.3, 'water': 0.7}
        feed = thermo.calculate_state(
            thermo.bubble_point_T(composition, 1.0),
            1.0,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        result = ShortcutDistillation(
            'S', thermo,
            {
                'N_stages': 30,
                'reflux_ratio': 8.0,
                'D_to_F': 0.3,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
            },
        ).solve({'feed': feed})

        self.assertAlmostEqual(result.performance['theoretical_stages'], 30.0, delta=5e-4)
        self.assertLess(result.performance['R_min'], 8.0)
        self.assertAlmostEqual(result.outlet_streams['distillate'].F, 30.0, places=7)

    def test_shortcut_distillation_cut_replaces_one_key_recovery(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        composition = {'methanol': 0.3, 'water': 0.7}
        feed = thermo.calculate_state(
            thermo.bubble_point_T(composition, 1.0),
            1.0,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        fixed_lk = ShortcutDistillation(
            'LK', thermo,
            {
                'D_to_F': 0.3,
                'light_key_recovery_distillate': 0.98,
                'reflux_ratio': 2.0,
            },
        ).solve({'feed': feed})
        fixed_hk = ShortcutDistillation(
            'HK', thermo,
            {
                'D_to_F': 0.3,
                'heavy_key_recovery_bottoms': 0.99,
                'reflux_ratio': 2.0,
            },
        ).solve({'feed': feed})

        self.assertAlmostEqual(fixed_lk.outlet_streams['distillate'].F, 30.0, places=8)
        self.assertAlmostEqual(fixed_hk.outlet_streams['distillate'].F, 30.0, places=8)
        self.assertEqual(
            fixed_lk.performance['recovery_spec_closure'],
            'distillate_spec_replaced_heavy_key_recovery',
        )
        self.assertEqual(
            fixed_hk.performance['recovery_spec_closure'],
            'distillate_spec_replaced_light_key_recovery',
        )

        with self.assertRaisesRegex(UnitOperationError, 'over-specified'):
            ShortcutDistillation(
                'OVER', thermo,
                {
                    'D_to_F': 0.3,
                    'light_key_recovery_distillate': 0.98,
                    'heavy_key_recovery_bottoms': 0.99,
                    'reflux_ratio': 2.0,
                },
            ).solve({'feed': feed})

    def test_shortcut_distillation_stage_only_and_reflux_only_design_modes(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        composition = {'methanol': 0.3, 'water': 0.7}
        feed = thermo.calculate_state(
            thermo.bubble_point_T(composition, 1.0),
            1.0,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        recovery_specs = {
            'light_key_recovery_distillate': 0.95,
            'heavy_key_recovery_bottoms': 0.95,
        }
        stage_only = ShortcutDistillation(
            'N', thermo, {**recovery_specs, 'N_stages': 20}
        ).solve({'feed': feed})
        reflux_only = ShortcutDistillation(
            'R', thermo, {**recovery_specs, 'reflux_ratio': 2.0}
        ).solve({'feed': feed})

        self.assertAlmostEqual(stage_only.performance['theoretical_stages'], 20.0)
        self.assertGreater(stage_only.performance['reflux_ratio'], stage_only.performance['R_min'])
        self.assertAlmostEqual(stage_only.performance['stage_margin'], 0.0)
        self.assertGreater(reflux_only.performance['theoretical_stages'], reflux_only.performance['N_min'])
        self.assertIsNone(reflux_only.performance['specified_stages'])

    def test_shortcut_distillation_uses_partial_condenser_for_noncondensables(self):
        thermo = create_thermodynamics(
            ['C2H4', 'N2', 'O2', 'C2H4O', 'CO2', 'H2O'], 'IDEAL'
        )
        composition = {
            'C2H4': 0.0192,
            'N2': 0.0198,
            'O2': 0.0023,
            'C2H4O': 0.8892,
            'CO2': 0.0012,
            'H2O': 0.0683,
        }
        feed = thermo.calculate_state(
            313.15, 19.0, 25.0, composition, phase='liquid', flash=False
        )
        params = {'N_stages': 25, 'reflux_ratio': 3.5, 'P_top': 2.0}
        with self.assertRaisesRegex(UnitOperationError, 'non-condensable'):
            ShortcutDistillation('EO-TOTAL', thermo, params).solve({'feed': feed})

        result = ShortcutDistillation(
            'EO', thermo,
            {**params, 'condenser_type': 'partial'},
        ).solve({'feed': feed})

        self.assertEqual(result.performance['light_key'], 'C2H4O')
        self.assertEqual(result.performance['heavy_key'], 'H2O')
        self.assertEqual(result.outlet_streams['distillate'].vapor_fraction, 1.0)
        self.assertIn('N2', result.outlet_streams['distillate'].composition)
        self.assertGreater(result.outlet_streams['distillate'].T, 250.0)
        self.assertIn('non-condensable', ' '.join(result.warnings))
        reflux_x = result.performance['reflux_liquid_composition']
        self.assertGreater(reflux_x['C2H4O'], 0.9)
        self.assertGreater(reflux_x['H2O'], result.outlet_streams['distillate'].composition['H2O'])
        self.assertLess(
            relative_component_balance(
                [feed],
                [result.outlet_streams['distillate'], result.outlet_streams['bottoms']],
            ),
            1e-10,
        )

    def test_shortcut_distillation_selects_adjacent_keys_at_cut_and_honors_overrides(self):
        thermo = create_thermodynamics(['methanol', 'ethanol', 'water'], 'IDEAL')
        composition = {'methanol': 0.2, 'ethanol': 0.5, 'water': 0.3}
        feed = thermo.calculate_state(
            350.0, 1.0, 100.0, composition, phase='liquid', flash=False
        )
        comps = list(composition)

        low_cut = ShortcutDistillation('LOW', thermo, {'D_to_F': 0.3})
        low_keys = low_cut._key_components(comps, feed)
        self.assertEqual((low_keys['light_key'], low_keys['heavy_key']), ('methanol', 'ethanol'))

        high_cut = ShortcutDistillation('HIGH', thermo, {'D_to_F': 0.6})
        high_keys = high_cut._key_components(comps, feed)
        self.assertEqual((high_keys['light_key'], high_keys['heavy_key']), ('ethanol', 'water'))

        light_override = ShortcutDistillation('LK', thermo, {'light_key': 'ethanol'})
        light_keys = light_override._key_components(comps, feed)
        self.assertEqual((light_keys['light_key'], light_keys['heavy_key']), ('ethanol', 'water'))

        heavy_override = ShortcutDistillation('HK', thermo, {'heavy_key': 'ethanol'})
        heavy_keys = heavy_override._key_components(comps, feed)
        self.assertEqual((heavy_keys['light_key'], heavy_keys['heavy_key']), ('methanol', 'ethanol'))

        with self.assertRaisesRegex(UnitOperationError, 'adjacent in volatility order'):
            ShortcutDistillation(
                'NONADJ', thermo,
                {'light_key': 'methanol', 'heavy_key': 'water'},
            )._key_components(comps, feed)

    def test_shortcut_distillation_underwood_rejects_an_interior_pole(self):
        thermo = create_thermodynamics(['methanol', 'ethanol', 'water'], 'IDEAL')
        unit = ShortcutDistillation('POLE', thermo, {})
        with self.assertRaisesRegex(UnitOperationError, 'interior volatility pole'):
            unit._underwood_root(
                ['methanol', 'ethanol', 'water'],
                {'methanol': 0.2, 'ethanol': 0.5, 'water': 0.3},
                {'methanol': 4.0, 'ethanol': 2.0, 'water': 1.0},
                'methanol',
                'water',
                1.0,
            )

    def test_shortcut_distillation_validates_specs_and_partial_condenser(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        composition = {'methanol': 0.3, 'water': 0.7}
        feed = thermo.calculate_state(
            thermo.bubble_point_T(composition, 1.0),
            1.0,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        partial = ShortcutDistillation(
            'P', thermo,
            {
                'light_key_recovery_distillate': 0.95,
                'heavy_key_recovery_bottoms': 0.95,
                'reflux_ratio': 2.0,
                'condenser_type': 'partial',
            },
        ).solve({'feed': feed})
        distillate = partial.outlet_streams['distillate']
        reflux_x = partial.performance['reflux_liquid_composition']
        self.assertEqual(distillate.vapor_fraction, 1.0)
        self.assertAlmostEqual(
            distillate.T,
            thermo.dew_point_T(distillate.composition, distillate.P, distillate.T),
            places=6,
        )
        K = thermo.K_values(distillate.T, distillate.P, reflux_x)
        y_raw = {comp: reflux_x[comp] * K[comp] for comp in reflux_x}
        y_total = sum(y_raw.values())
        for comp in distillate.composition:
            self.assertAlmostEqual(
                distillate.composition[comp],
                y_raw[comp] / y_total,
                places=6,
            )

        invalid_cases = [
            ({'N_stages': 1}, 'N_stages'),
            ({'P_condenser': 0.0}, 'pressure'),
            ({'P_drop_per_stage': -0.1}, 'nonnegative'),
            ({'D_to_F': 0.3, 'D': 30.0}, 'only one distillate'),
            ({'light_key': 'missing'}, 'unknown component'),
            ({'light_key': 'water', 'heavy_key': 'methanol'}, 'more volatile'),
        ]
        for params, message in invalid_cases:
            with self.subTest(params=params):
                with self.assertRaisesRegex(UnitOperationError, message):
                    ShortcutDistillation('BAD', thermo, params).solve({'feed': feed})

        with self.assertRaisesRegex(UnitOperationError, 'no inlet'):
            ShortcutDistillation('EMPTY', thermo, {}).solve({})

    def test_molecular_sieve_3a_liquid_equilibrium_dryer(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIQUAC')
        ethanol_moles = 95.0 / 46.06844
        water_moles = 5.0 / 18.01528
        total_moles = ethanol_moles + water_moles
        composition = {
            'ethanol': ethanol_moles / total_moles,
            'water': water_moles / total_moles,
        }
        feed = thermo.calculate_state(
            298.15,
            1.0,
            total_moles,
            composition,
            phase='liquid',
            flash=False,
        )

        result = MolecularSieveDryer(
            'MS-3A',
            thermo,
            {'sieve_type': '3A', 'adsorbent_mass_flow': 25.0},
        ).solve({'feed': feed})

        product = result.outlet_streams['product']
        adsorbate = result.outlet_streams['adsorbate']
        performance = result.performance
        self.assertEqual(performance['mode'], 'equilibrium_isothermal_pseudo_continuous')
        self.assertEqual(performance['driver_basis'], 'liquid_water_activity')
        self.assertLess(product.composition['water'], feed.composition['water'])
        self.assertGreater(adsorbate.F, 0.0)
        self.assertAlmostEqual(
            performance['final_loading_kg_water_per_kg_sieve'],
            performance['equilibrium_loading_kg_water_per_kg_sieve'],
            places=9,
        )
        self.assertLess(result.heat_duty, 0.0)
        self.assertGreater(performance['adsorption_heat_release_kJ_h'], 0.0)
        stream_enthalpy_duty = (
            product.F * product.H
            + adsorbate.F * adsorbate.H
            - feed.F * feed.H
        )
        self.assertAlmostEqual(result.heat_duty, stream_enthalpy_duty, places=8)
        self.assertAlmostEqual(
            performance['stream_enthalpy_duty_kJ_h'],
            stream_enthalpy_duty,
            places=8,
        )
        self.assertLess(relative_component_balance([feed], [product, adsorbate]), 1e-10)

    def test_molecular_sieve_3a_vapor_equilibrium_dryer_uses_fugacity_basis(self):
        thermo = create_thermodynamics(['water', 'N2'], 'RK')
        feed = thermo.calculate_state(
            350.0,
            1.0,
            1.0,
            {'water': 0.05, 'N2': 0.95},
            phase='vapor',
            flash=False,
        )

        result = MolecularSieveDryer(
            'MS-3A-GAS',
            thermo,
            {'sieve_type': '3A', 'adsorbent_mass_flow': 2.0},
        ).solve({'feed': feed})

        product = result.outlet_streams['product']
        adsorbate = result.outlet_streams['adsorbate']
        performance = result.performance
        self.assertEqual(performance['driver_basis'], 'vapor_fugacity_bar')
        self.assertLess(product.composition['water'], feed.composition['water'])
        self.assertGreater(adsorbate.F, 0.0)
        self.assertAlmostEqual(
            performance['final_loading_kg_water_per_kg_sieve'],
            performance['equilibrium_loading_kg_water_per_kg_sieve'],
            places=9,
        )
        self.assertLess(result.heat_duty, 0.0)
        self.assertLess(relative_component_balance([feed], [product, adsorbate]), 1e-10)

    def test_molecular_sieve_specs_are_mutually_exclusive(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIQUAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            1.0,
            {'ethanol': 0.95, 'water': 0.05},
            phase='liquid',
            flash=False,
        )

        with self.assertRaisesRegex(UnitOperationError, 'mutually exclusive'):
            MolecularSieveDryer(
                'MS-3A',
                thermo,
                {
                    'target_water_mole_fraction': 0.01,
                    'adsorbent_mass_flow': 25.0,
                },
            ).solve({'feed': feed})

        with self.assertRaisesRegex(UnitOperationError, 'mutually exclusive'):
            MolecularSieveDryer(
                'MS-3A',
                thermo,
                {
                    'removal_fraction': 0.5,
                    'adsorbent_mass_flow': 25.0,
                },
            ).solve({'feed': feed})

        with self.assertRaisesRegex(UnitOperationError, 'mutually exclusive'):
            MolecularSieveDryer(
                'MS-3A',
                thermo,
                {
                    'target_water_mole_fraction': 0.01,
                    'removal_fraction': 0.5,
                },
            ).solve({'feed': feed})

    def test_molecular_sieve_rejects_impossible_target_or_removal_specs(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIQUAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            1.0,
            {'ethanol': 0.95, 'water': 0.05},
            phase='liquid',
            flash=False,
        )

        with self.assertRaisesRegex(UnitOperationError, 'between 0 and 1'):
            MolecularSieveDryer(
                'MS-REMOVAL',
                thermo,
                {'removal_fraction': 1.1},
            ).solve({'feed': feed})

        with self.assertRaisesRegex(UnitOperationError, 'thermodynamically impossible'):
            MolecularSieveDryer(
                'MS-DRY',
                thermo,
                {'target_water_mole_fraction': 0.0},
            ).solve({'feed': feed})

        with self.assertRaisesRegex(UnitOperationError, 'thermodynamically impossible'):
            MolecularSieveDryer(
                'MS-FULL-REMOVAL',
                thermo,
                {'removal_fraction': 1.0},
            ).solve({'feed': feed})

    def test_molecular_sieve_target_above_feed_water_needs_no_sieve(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIQUAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            1.0,
            {'ethanol': 0.95, 'water': 0.05},
            phase='liquid',
            flash=False,
        )

        result = MolecularSieveDryer(
            'MS-ALREADY-DRY',
            thermo,
            {'target_water_mole_fraction': 0.10},
        ).solve({'feed': feed})

        self.assertAlmostEqual(
            result.outlet_streams['product'].composition['water'],
            feed.composition['water'],
        )
        self.assertAlmostEqual(result.performance['adsorbent_mass_flow_kg_h'], 0.0)
        self.assertAlmostEqual(result.performance['water_removed_kmol_h'], 0.0)
        self.assertAlmostEqual(result.heat_duty, 0.0)

    def test_molecular_sieve_target_and_removal_specs_size_3a_flow(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIQUAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            1.0,
            {'ethanol': 0.95, 'water': 0.05},
            phase='liquid',
            flash=False,
        )

        target_result = MolecularSieveDryer(
            'MS-TARGET',
            thermo,
            {'target_water_mole_fraction': 0.01},
        ).solve({'feed': feed})
        target_performance = target_result.performance
        self.assertAlmostEqual(
            target_result.outlet_streams['product'].composition['water'],
            0.01,
            places=12,
        )
        self.assertGreater(target_performance['adsorbent_mass_flow_kg_h'], 0.0)
        self.assertAlmostEqual(
            target_performance['final_loading_kg_water_per_kg_sieve'],
            target_performance['equilibrium_loading_kg_water_per_kg_sieve'],
            places=12,
        )
        self.assertLess(target_result.heat_duty, 0.0)

        removal_result = MolecularSieveDryer(
            'MS-REMOVAL',
            thermo,
            {'removal_fraction': 0.5},
        ).solve({'feed': feed})
        removal_performance = removal_result.performance
        self.assertAlmostEqual(removal_performance['water_removal_fraction'], 0.5)
        self.assertGreater(removal_performance['adsorbent_mass_flow_kg_h'], 0.0)
        self.assertAlmostEqual(
            removal_performance['final_loading_kg_water_per_kg_sieve'],
            removal_performance['equilibrium_loading_kg_water_per_kg_sieve'],
            places=12,
        )
        self.assertLess(removal_result.heat_duty, 0.0)

    def test_molecular_sieve_rejects_initial_loading_above_3a_maximum(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIQUAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            1.0,
            {'ethanol': 0.95, 'water': 0.05},
            phase='liquid',
            flash=False,
        )

        with self.assertRaisesRegex(UnitOperationError, 'maximum loading'):
            MolecularSieveDryer(
                'MS-LOADED',
                thermo,
                {
                    'sieve_type': '3A',
                    'adsorbent_mass_flow': 25.0,
                    'initial_loading': 0.3,
                },
            ).solve({'feed': feed})

    def test_molecular_sieve_rejects_initial_loading_above_inlet_equilibrium(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIQUAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            1.0,
            {'ethanol': 0.999, 'water': 0.001},
            phase='liquid',
            flash=False,
        )

        with self.assertRaisesRegex(UnitOperationError, 'too wet'):
            MolecularSieveDryer(
                'MS-WET',
                thermo,
                {
                    'sieve_type': '3A',
                    'adsorbent_mass_flow': 10.0,
                    'initial_loading': 0.2,
                },
            ).solve({'feed': feed})

    def test_mccabe_thiele_binary_distillation_selects_internal_feed_stage(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIFAC')
        feed = thermo.calculate_state(
            298.15, 1.0, 100.0, {'ethanol': 0.5, 'water': 0.5}, phase='liquid'
        )

        result = McCabeThieleDistillation(
            'U', thermo,
            {
                'N_stages': 40,
                'reflux_ratio': 5,
                'D_to_F': 0.5,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
            },
        ).solve({'feed': feed})

        distillate = result.outlet_streams['distillate']
        bottoms = result.outlet_streams['bottoms']
        self.assertGreater(distillate.composition['ethanol'], feed.composition['ethanol'])
        self.assertLess(bottoms.composition['ethanol'], feed.composition['ethanol'])
        self.assertGreaterEqual(result.performance['selected_feed_stage'], 1)
        self.assertLessEqual(result.performance['selected_feed_stage'], 40)
        self.assertLess(result.performance['stage_error'], 1e-4)
        self.assertLess(relative_component_balance([feed], [distillate, bottoms]), 1e-8)

    def test_mccabe_thiele_hexane_heptane_auto_stage_separates_cleanly(self):
        thermo = create_thermodynamics(['hexane', 'heptane'], 'UNIFAC')
        feed = thermo.calculate_state(
            330.0, 1.0, 100.0, {'hexane': 0.5, 'heptane': 0.5}, phase='liquid'
        )

        result = McCabeThieleDistillation(
            'U', thermo,
            {
                'N_stages': 12,
                'reflux_ratio': 3,
                'D_to_F': 0.5,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
            },
        ).solve({'feed': feed})

        distillate = result.outlet_streams['distillate']
        bottoms = result.outlet_streams['bottoms']
        self.assertGreater(distillate.composition['hexane'], 0.9)
        self.assertLess(bottoms.composition['hexane'], 0.1)
        self.assertGreaterEqual(result.performance['selected_feed_stage'], 1)
        self.assertLessEqual(result.performance['selected_feed_stage'], 12)

    def test_mccabe_thiele_uses_model_k_values_at_weighted_saturation_temperature(self):
        thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')
        composition = {'CH3COOH': 0.01554377, 'H2O': 0.98445623}
        pressure = 1.01325
        feed = thermo.calculate_state(
            298.15,
            pressure,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        result = McCabeThieleDistillation(
            'VINEGAR', thermo,
            {
                'N_stages': 30,
                'reflux_ratio': 2.0,
                'D_to_F': 0.975,
                'P_condenser': pressure,
                'P_drop_per_stage': 0.0,
            },
        ).solve({'feed': feed})

        feed_saturation = thermo.bubble_point_T(composition, pressure, feed.T)
        pure_water = thermo.bubble_point_T(
            {'CH3COOH': 0.0, 'H2O': 1.0}, pressure, feed_saturation
        )
        pure_acid = thermo.bubble_point_T(
            {'CH3COOH': 1.0, 'H2O': 0.0}, pressure, feed_saturation
        )
        expected_reference = 0.25 * (
            pure_water + pure_acid + 2.0 * feed_saturation
        )
        expected_K = thermo.K_values(expected_reference, pressure, composition)

        self.assertEqual(result.performance['light_key'], 'H2O')
        self.assertEqual(result.performance['heavy_key'], 'CH3COOH')
        self.assertAlmostEqual(
            result.performance['relative_volatility_reference_temperature_C'],
            expected_reference - 273.15,
            places=8,
        )
        for comp in composition:
            self.assertAlmostEqual(
                result.performance['reference_K_values'][comp],
                expected_K[comp],
                places=10,
            )
        self.assertLess(
            result.outlet_streams['distillate'].composition['CH3COOH'],
            composition['CH3COOH'],
        )
        self.assertGreater(
            result.outlet_streams['bottoms'].composition['CH3COOH'],
            0.60,
        )
        self.assertEqual(result.performance['selected_feed_stage'], 17)

    def test_mccabe_thiele_uses_latent_heat_curved_operating_lines(self):

        cases = []
        thermo = create_thermodynamics(['methanol', 'water'], 'NRTL')
        composition = {'methanol': 0.3, 'water': 0.7}
        feed = thermo.calculate_state(
            thermo.bubble_point_T(composition, 1.0),
            1.0,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        cases.append((
            thermo,
            feed,
            {
                'N_stages': 20,
                'reflux_ratio': 1.0,
                'D_to_F': 0.3,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
            },
            15,
            'methanol',
        ))

        thermo = create_thermodynamics(['propane', 'n-butane'], 'PR-TWU')
        composition = {'propane': 0.4, 'n-butane': 0.6}
        feed = thermo.calculate_state(
            thermo.bubble_point_T(composition, 15.0),
            15.0,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        cases.append((
            thermo,
            feed,
            {
                'N_stages': 12,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 15.0,
                'P_drop_per_stage': 0.0,
            },
            8,
            'propane',
        ))

        for thermo, feed, params, feed_stage, component in cases:
            with self.subTest(component=component):
                cmo = McCabeThieleDistillation('MT', thermo, params).solve({'feed': feed})
                cmo_hvap = McCabeThieleDistillation(
                    'MT-HVAP',
                    thermo,
                    {**params, 'latent_heat_correction': True},
                ).solve({'feed': feed})
                rigorous = RigorousDistillation(
                    'RIG',
                    thermo,
                    {
                        **params,
                        'feed_stage': feed_stage,
                        'initializer': 'cheap_estimate',
                        'mesh_tolerance': 1e-6,
                        'max_iterations': 400,
                        'max_jacobian_evaluations': 400,
                    },
                ).solve({'feed': feed})

                self.assertEqual(
                    cmo_hvap.performance['operating_line_model'],
                    'latent_heat_curved',
                )
                cmo_error = abs(
                    cmo.outlet_streams['distillate'].composition[component]
                    - rigorous.outlet_streams['distillate'].composition[component]
                )
                cmo_hvap_error = abs(
                    cmo_hvap.outlet_streams['distillate'].composition[component]
                    - rigorous.outlet_streams['distillate'].composition[component]
                )
                self.assertLess(cmo_hvap_error, cmo_error)
                self.assertLess(
                    relative_component_balance(
                        [feed],
                        [cmo_hvap.outlet_streams['distillate'], cmo_hvap.outlet_streams['bottoms']],
                    ),
                    1e-8,
                )

    def test_methanol_water_hvap_variants_improve_mccabe_and_cmo_compositions(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'NRTL')
        composition = {'methanol': 0.3, 'water': 0.7}
        pressure = 1.0
        feed = thermo.calculate_state(
            thermo.bubble_point_T(composition, pressure),
            pressure,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        common = {
            'N_stages': 20,
            'reflux_ratio': 1.0,
            'D_to_F': 0.3,
            'P_condenser': pressure,
            'P_drop_per_stage': 0.0,
        }
        mccabe = McCabeThieleDistillation(
            'MT', thermo, common
        ).solve({'feed': feed})
        mccabe_hvap = McCabeThieleDistillation(
            'MT-HVAP',
            thermo,
            {**common, 'latent_heat_correction': True},
        ).solve({'feed': feed})

        cmo_common = {
            **common,
            'feed_stage': 15,
            'mesh_tolerance': 1e-7,
            'max_iterations': 160,
            'max_jacobian_evaluations': 160,
        }
        cmo = CMODistillation(
            'CMO', thermo, cmo_common
        ).solve({'feed': feed})
        cmo_hvap = CMODistillation(
            'CMO-HVAP',
            thermo,
            {**cmo_common, 'latent_heat_correction': True},
        ).solve({'feed': feed})
        rigorous = RigorousDistillation(
            'RIGOROUS',
            thermo,
            {
                **cmo_common,
                'initializer': 'cheap_estimate',
            },
        ).solve({'feed': feed})

        for name, baseline, corrected in (
            ('McCabe-Thiele', mccabe, mccabe_hvap),
            ('CMO', cmo, cmo_hvap),
        ):
            for product in ('distillate', 'bottoms'):
                with self.subTest(method=name, product=product):
                    rigorous_methanol = rigorous.outlet_streams[
                        product
                    ].composition['methanol']
                    baseline_error = abs(
                        baseline.outlet_streams[product].composition['methanol']
                        - rigorous_methanol
                    )
                    corrected_error = abs(
                        corrected.outlet_streams[product].composition['methanol']
                        - rigorous_methanol
                    )
                    self.assertLess(corrected_error, baseline_error)

    def test_multicomponent_rigorous_distillation_balances_and_separates(self):
        thermo = create_thermodynamics(['CH3OH', 'C2H5OH', 'H2O'], 'IDEAL')
        feed = thermo.calculate_state(
            360.0,
            1.0,
            100.0,
            {'CH3OH': 0.2, 'C2H5OH': 0.3, 'H2O': 0.5},
            phase='liquid',
        )
        with self.assertRaisesRegex(UnitOperationError, 'binary feeds only'):
            McCabeThieleDistillation(
                'MT',
                thermo,
                {
                    'N_stages': 20,
                    'reflux_ratio': 2.0,
                    'D_to_F': 0.4,
                },
            ).solve({'feed': feed})

        unit = CMODistillation(
            'U', thermo,
            {
                'N_stages': 20,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
            },
        )

        result = unit.solve({'feed': feed})
        distillate = result.outlet_streams['distillate']
        bottoms = result.outlet_streams['bottoms']

        self.assertLess(relative_component_balance([feed], [distillate, bottoms]), 1e-7)
        self.assertAlmostEqual(distillate.F, 40.0, places=6)
        self.assertGreater(distillate.composition['CH3OH'], feed.composition['CH3OH'])
        self.assertGreater(distillate.composition['C2H5OH'], feed.composition['C2H5OH'])
        self.assertGreater(bottoms.composition['H2O'], feed.composition['H2O'])

    def test_distillation_pressure_drop_profiles(self):
        thermo = create_thermodynamics(['H2O', 'C2H5OH'], 'IDEAL')
        feed = thermo.calculate_state(
            298.15, 1.0, 100.0, {'H2O': 0.8, 'C2H5OH': 0.2}, phase='liquid'
        )
        common = {
            'N_stages': 5,
            'reflux_ratio': 2.0,
            'D_to_F': 0.4,
            'P_condenser': 1.0,
            'P_drop_per_stage': 0.1,
        }

        shortcut = ShortcutDistillation('S', thermo, common).solve({'feed': feed})
        self.assertAlmostEqual(shortcut.outlet_streams['distillate'].P, 1.0)
        self.assertAlmostEqual(shortcut.outlet_streams['bottoms'].P, 1.4)

        cmo = McCabeThieleDistillation('C', thermo, common).solve({'feed': feed})
        self.assertAlmostEqual(cmo.outlet_streams['distillate'].P, 1.0)
        self.assertAlmostEqual(cmo.outlet_streams['bottoms'].P, 1.4)

        rigorous = RigorousDistillation(
            'R',
            thermo,
            {
                **common,
                'feed_stage': 2,
                'mesh_tolerance': 1e-5,
            },
        ).solve({'feed': feed})
        self.assertEqual(
            [round(value, 6) for value in rigorous.performance['stage_pressures_bar']],
            [1.0, 1.1, 1.2, 1.3, 1.4],
        )
        self.assertAlmostEqual(rigorous.outlet_streams['distillate'].P, 1.0)
        self.assertAlmostEqual(rigorous.outlet_streams['bottoms'].P, 1.4)

        bottom_spec = RigorousDistillation(
            'RB',
            thermo,
            {
                **common,
                'feed_stage': 2,
                'P_bottom': 1.8,
                'mesh_tolerance': 1e-5,
            },
        ).solve({'feed': feed})
        self.assertEqual(
            [round(value, 6) for value in bottom_spec.performance['stage_pressures_bar']],
            [1.0, 1.2, 1.4, 1.6, 1.8],
        )

    def test_cmo_distillation_supports_partial_and_mixed_condensers(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        composition = {'methanol': 0.3, 'water': 0.7}
        feed = thermo.calculate_state(
            thermo.bubble_point_T(composition, 1.0),
            1.0,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        common = {
            'N_stages': 12,
            'feed_stage': 6,
            'reflux_ratio': 2.0,
            'D_to_F': 0.3,
            'P_condenser': 1.0,
            'P_drop_per_stage': 0.0,
            'mesh_tolerance': 1e-6,
            'max_iterations': 120,
        }

        partial = CMODistillation(
            'PARTIAL',
            thermo,
            {**common, 'condenser_type': 'partial'},
        ).solve({'feed': feed})
        self.assertEqual(partial.outlet_streams['distillate'].vapor_fraction, 1.0)
        self.assertEqual(partial.performance['distillate_vapor_fraction'], 1.0)
        self.assertLess(partial.performance['mes_residual'], 1e-6)

        mixed = CMODistillation(
            'MIXED',
            thermo,
            {
                **common,
                'condenser_type': 'mixed',
                'distillate_vapor_fraction': 0.4,
            },
        ).solve({'feed': feed})
        self.assertAlmostEqual(mixed.outlet_streams['distillate'].vapor_fraction, 0.4)
        self.assertIn('distillate_liquid', mixed.outlet_streams)
        self.assertIn('distillate_vapor', mixed.outlet_streams)
        self.assertLess(mixed.performance['mes_residual'], 1e-6)
        self.assertLess(
            relative_component_balance(
                [feed],
                [mixed.outlet_streams['distillate'], mixed.outlet_streams['bottoms']],
            ),
            1e-8,
        )

        corrected = CMODistillation(
            'HVAP',
            thermo,
            {**common, 'latent_heat_correction': True},
        ).solve({'feed': feed})
        self.assertTrue(corrected.performance['latent_heat_correction'])
        self.assertIsNotNone(corrected.performance['constant_component_hvap'])
        self.assertLess(corrected.performance['mes_residual'], 1e-6)

    def test_rigorous_distillation_initializer_routing(self):
        binary = create_thermodynamics(['methanol', 'water'], 'NRTL')
        binary_z = {'methanol': 0.3, 'water': 0.7}
        binary_feed = binary.calculate_state(
            binary.bubble_point_T(binary_z, 1.0),
            1.0,
            100.0,
            binary_z,
            phase='liquid',
            flash=False,
        )
        binary_common = {
            'N_stages': 12,
            'feed_stage': 8,
            'reflux_ratio': 2.0,
            'D_to_F': 0.3,
            'P_condenser': 1.0,
            'P_drop_per_stage': 0.0,
            'mesh_tolerance': 1e-6,
            'max_iterations': 160,
            'max_jacobian_evaluations': 160,
        }
        binary_results = {}
        for requested, expected in (
            (None, 'estimate'),
            ('auto', 'estimate'),
            ('estimate', 'estimate'),
            ('cheap_estimate', 'cheap_estimate'),
            ('cmo', 'cmo'),
            ('cmo-hvap', 'cmo_hvap'),
        ):
            with self.subTest(initializer=requested):
                params = dict(binary_common)
                if requested is not None:
                    params['initializer'] = requested
                result = RigorousDistillation(
                    f'R-{requested}', binary, params
                ).solve({'feed': binary_feed})
                self.assertEqual(result.performance['initializer'], expected)
                self.assertLess(result.performance['mesh_residual'], 1e-6)
                binary_results[requested] = result

        self.assertEqual(
            binary_results['estimate'].performance['solver_iterations'],
            binary_results['cheap_estimate'].performance['solver_iterations'],
        )
        for comp in binary_z:
            self.assertAlmostEqual(
                binary_results['estimate'].outlet_streams['distillate'].composition[comp],
                binary_results['cheap_estimate'].outlet_streams['distillate'].composition[comp],
                places=12,
            )

        cheap_params = {**binary_common, 'initializer': 'cheap_estimate'}
        forbidden = AssertionError('cheap_estimate invoked a specialized initializer')
        with (
            patch.object(
                RigorousDistillation,
                '_legacy_cmo_endpoint_initial_guess',
                side_effect=forbidden,
            ),
            patch.object(
                RigorousDistillation,
                '_coarse_grid_initial_guess',
                side_effect=forbidden,
            ),
            patch.object(
                RigorousDistillation,
                '_cmo_profile_initial_guess',
                side_effect=forbidden,
            ),
            patch.object(
                RigorousDistillation,
                '_azeotropic_endpoint_initial_guess',
                side_effect=forbidden,
            ),
        ):
            cheap_result = RigorousDistillation(
                'R-CHEAP-PIN', binary, cheap_params
            ).solve({'feed': binary_feed})
        self.assertEqual(cheap_result.performance['initializer'], 'cheap_estimate')
        self.assertLess(cheap_result.performance['mesh_residual'], 1e-6)

        multicomponent = create_thermodynamics(
            ['methanol', 'ethanol', 'water'], 'IDEAL'
        )
        multicomponent_feed = multicomponent.calculate_state(
            360.0,
            1.0,
            100.0,
            {'methanol': 0.2, 'ethanol': 0.3, 'water': 0.5},
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'R-MULTI',
            multicomponent,
            {
                'N_stages': 18,
                'feed_stage': 9,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'mesh_tolerance': 1e-5,
                'max_iterations': 120,
                'max_jacobian_evaluations': 120,
            },
        ).solve({'feed': multicomponent_feed})
        self.assertEqual(result.performance['initializer'], 'coarse_rigorous')
        self.assertLess(result.performance['mesh_residual'], 1e-5)

    def test_coarse_rigorous_uses_legacy_cmo_estimate_seed(self):
        thermo = create_thermodynamics(
            ['methanol', 'ethanol', 'water'], 'IDEAL'
        )
        feed = thermo.calculate_state(
            360.0,
            1.0,
            100.0,
            {'methanol': 0.2, 'ethanol': 0.3, 'water': 0.5},
            phase='liquid',
            flash=False,
        )
        calls = []
        original = RigorousDistillation._legacy_cmo_endpoint_initial_guess

        def tracking_estimate(unit, *args, **kwargs):
            calls.append(unit.unit_id)
            return original(unit, *args, **kwargs)

        with patch.object(
            RigorousDistillation,
            '_legacy_cmo_endpoint_initial_guess',
            tracking_estimate,
        ):
            result = RigorousDistillation(
                'R-COARSE',
                thermo,
                {
                    'N_stages': 18,
                    'feed_stage': 9,
                    'reflux_ratio': 2.0,
                    'D_to_F': 0.4,
                    'P_condenser': 1.0,
                    'P_drop_per_stage': 0.0,
                    'initializer': 'coarse_rigorous',
                    'mesh_tolerance': 1e-5,
                    'max_iterations': 120,
                    'max_jacobian_evaluations': 120,
                },
            ).solve({'feed': feed})

        self.assertTrue(calls)
        self.assertTrue(any('coarse_init' in unit_id for unit_id in calls))
        self.assertEqual(result.performance['initializer'], 'coarse_rigorous')

    def test_cmo_rigorous_initializer_defaults_to_relaxed_tolerance_and_respects_override(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'NRTL')
        composition = {'methanol': 0.3, 'water': 0.7}
        feed = thermo.calculate_state(
            thermo.bubble_point_T(composition, 1.0),
            1.0,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        common = {
            'N_stages': 12,
            'feed_stage': 8,
            'reflux_ratio': 2.0,
            'D_to_F': 0.3,
            'P_condenser': 1.0,
            'P_drop_per_stage': 0.0,
            'initializer': 'cmo',
            'mesh_tolerance': 1e-6,
            'max_iterations': 120,
            'max_jacobian_evaluations': 120,
        }
        observed = []
        original = CMODistillation.solve

        def capture_tolerance(unit, inlets):
            observed.append(float(unit.get_param('cmo_tolerance')))
            return original(unit, inlets)

        with patch.object(CMODistillation, 'solve', capture_tolerance):
            default_result = RigorousDistillation(
                'R-CMO-DEFAULT', thermo, common
            ).solve({'feed': feed})
            override_result = RigorousDistillation(
                'R-CMO-OVERRIDE',
                thermo,
                {**common, 'cmo_tolerance': 2e-5},
            ).solve({'feed': feed})

        self.assertEqual(observed, [1e-4, 2e-5])
        self.assertLess(default_result.performance['mesh_residual'], 1e-6)
        self.assertLess(override_result.performance['mesh_residual'], 1e-6)

    def test_explicit_rigorous_initializers_fail_hard(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'IDEAL')
        feed = thermo.calculate_state(
            350.0,
            1.0,
            100.0,
            {'methanol': 0.3, 'water': 0.7},
            phase='liquid',
            flash=False,
        )
        common = {
            'N_stages': 8,
            'feed_stage': 4,
            'reflux_ratio': 2.0,
            'D_to_F': 0.3,
            'P_condenser': 1.0,
            'P_drop_per_stage': 0.0,
        }

        with patch.object(
            RigorousDistillation,
            '_cmo_profile_initial_guess',
            side_effect=UnitOperationError('cmo failed'),
        ):
            for initializer in ('cmo', 'cmo-hvap'):
                with self.subTest(initializer=initializer):
                    with self.assertRaisesRegex(UnitOperationError, 'cmo failed'):
                        RigorousDistillation(
                            'R-CMO-FAIL',
                            thermo,
                            {**common, 'initializer': initializer},
                        ).solve({'feed': feed})

        with patch.object(
            RigorousDistillation,
            '_azeotropic_endpoint_initial_guess',
            return_value=None,
        ):
            with self.assertRaisesRegex(UnitOperationError, 'azeotropic initializer'):
                RigorousDistillation(
                    'R-AZ-FAIL',
                    thermo,
                    {**common, 'initializer': 'azeotropic'},
                ).solve({'feed': feed})

        with patch.object(
            RigorousDistillation,
            '_coarse_grid_initial_guess',
            return_value=None,
        ):
            with self.assertRaisesRegex(UnitOperationError, 'coarse_rigorous initializer'):
                RigorousDistillation(
                    'R-COARSE-FAIL',
                    thermo,
                    {**common, 'initializer': 'coarse_rigorous'},
                ).solve({'feed': feed})

    def test_rigorous_cmo_initializers_preserve_condenser_and_pressure_profile(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'NRTL')
        composition = {'methanol': 0.3, 'water': 0.7}
        feed = thermo.calculate_state(
            thermo.bubble_point_T(composition, 1.0),
            1.0,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        pressures = [1.0 + 0.01 * stage for stage in range(8)]
        original_solve = CMODistillation.solve

        for initializer, condenser, vapor_fraction, corrected in (
            ('cmo', 'partial', None, False),
            ('cmo-hvap', 'mixed', 0.35, True),
        ):
            captured = []

            def tracking_solve(unit, inlets):
                inlet = next(iter(inlets.values()))
                captured.append({
                    'condenser': unit.get_param('condenser_type'),
                    'vapor_fraction': unit.get_param('distillate_vapor_fraction'),
                    'pressures': unit._pressure_profile(8, inlet.P),
                    'corrected': unit._truthy_param(
                        unit.get_param('cmo_latent_heat_correction', False)
                    ),
                })
                return original_solve(unit, inlets)

            params = {
                'N_stages': 8,
                'feed_stage': 4,
                'reflux_ratio': 2.0,
                'D_to_F': 0.3,
                'stage_pressures': pressures,
                'condenser_type': condenser,
                'initializer': initializer,
                'mesh_tolerance': 1e-6,
                'max_iterations': 120,
                'max_jacobian_evaluations': 120,
            }
            if vapor_fraction is not None:
                params['distillate_vapor_fraction'] = vapor_fraction

            with self.subTest(initializer=initializer, condenser=condenser):
                with patch.object(CMODistillation, 'solve', tracking_solve):
                    result = RigorousDistillation(
                        f'R-{initializer}', thermo, params
                    ).solve({'feed': feed})

                self.assertEqual(len(captured), 1)
                self.assertEqual(captured[0]['condenser'], condenser)
                self.assertEqual(captured[0]['pressures'], pressures)
                self.assertEqual(captured[0]['corrected'], corrected)
                self.assertEqual(result.performance['condenser_type'], condenser)
                if vapor_fraction is not None:
                    self.assertAlmostEqual(
                        captured[0]['vapor_fraction'], vapor_fraction
                    )
                    self.assertAlmostEqual(
                        result.performance['distillate_vapor_fraction'],
                        vapor_fraction,
                    )

    def test_nitrile_column_azeotropic_initializer_converges_within_eight_iterations(self):
        groups = {
            'acrylonitrile': {68: 1},
            'acetonitrile': {40: 1},
            'water': {16: 1},
        }
        thermo = create_thermodynamics(
            ['acrylonitrile', 'acetonitrile', 'water'],
            'UNIFNIST',
            None,
            groups,
        )
        composition = {
            'acrylonitrile': 0.6,
            'acetonitrile': 0.1,
            'water': 0.3,
        }
        pressure = 1.01325
        feed = thermo.calculate_state(
            thermo.bubble_point_T(composition, pressure),
            pressure,
            100.0,
            composition,
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'NITRILE-COLUMN',
            thermo,
            {
                'N_stages': 80,
                'feed_stage': 70,
                'reflux_ratio': 10.0,
                'D_to_F': 0.861,
                'P_condenser': pressure,
                'P_drop_per_stage': 0.0,
                'condenser_type': 'total',
                'initializer': 'azeotropic',
                'mesh_tolerance': 1e-6,
                'max_iterations': 400,
                'max_jacobian_evaluations': 400,
            },
        ).solve({'feed': feed})

        self.assertEqual(result.performance['initializer'], 'azeotropic')
        self.assertLessEqual(result.performance['solver_iterations'], 8)
        self.assertLess(result.performance['mesh_residual'], 1e-6)
        self.assertLess(
            relative_component_balance(
                [feed],
                [result.outlet_streams['distillate'], result.outlet_streams['bottoms']],
            ),
            1e-6,
        )
        distillate = result.outlet_streams['distillate']
        bottoms = result.outlet_streams['bottoms']
        self.assertGreater(distillate.composition['acrylonitrile'], 0.69)
        self.assertLess(distillate.composition['acetonitrile'], 0.001)
        self.assertLess(bottoms.composition['acrylonitrile'], 0.005)
        self.assertGreater(bottoms.composition['acetonitrile'], 0.70)

        acrylonitrile_recovery = (
            distillate.F * distillate.composition['acrylonitrile']
            / (feed.F * feed.composition['acrylonitrile'])
        )
        acetonitrile_recovery = (
            bottoms.F * bottoms.composition['acetonitrile']
            / (feed.F * feed.composition['acetonitrile'])
        )
        self.assertGreater(acrylonitrile_recovery, 0.999)
        self.assertGreater(acetonitrile_recovery, 0.99)

    def test_rigorous_distillation_solves_mesh_balances(self):
        thermo = create_thermodynamics(['H2O', 'C2H5OH'], 'IDEAL')
        feed = thermo.calculate_state(
            298.15, 1.0, 100.0, {'H2O': 0.8, 'C2H5OH': 0.2}, phase='liquid'
        )
        unit = RigorousDistillation(
            'U', thermo,
            {
                'N_stages': 4,
                'feed_stage': 2,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
            },
        )

        result = unit.solve({'feed': feed})
        distillate = result.outlet_streams['distillate']
        bottoms = result.outlet_streams['bottoms']

        self.assertLess(result.performance['mesh_residual'], 1e-6)
        self.assertLess(relative_component_balance([feed], [distillate, bottoms]), 1e-8)
        self.assertGreater(distillate.composition['C2H5OH'], feed.composition['C2H5OH'])
        self.assertGreater(bottoms.composition['H2O'], feed.composition['H2O'])
        self.assertGreater(
            max(result.performance['liquid_flows']) - min(result.performance['liquid_flows']),
            1.0,
        )

    def test_binary_rigorous_distillation_uses_robust_square_solver(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'methanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'U', thermo,
            {
                'N_stages': 20,
                'feed_stage': 15,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'mesh_tolerance': 1e-7,
            },
        ).solve({'feed': feed})

        distillate = result.outlet_streams['distillate']
        self.assertEqual(result.performance['solver'], 'sparse_damped_newton')
        self.assertLess(result.performance['mesh_residual'], 1e-7)
        self.assertLess(result.performance['component_balance_error'], 1e-7)
        self.assertIn('component', result.performance['residual_diagnostics'])
        self.assertIn('energy', result.performance['residual_diagnostics'])
        self.assertIn('max_scaled_abs', result.performance['residual_diagnostics']['component'])
        self.assertGreater(distillate.composition['methanol'], feed.composition['methanol'])

    def test_rigorous_distillation_semi_analytic_jacobian_handles_partial_mass_spec(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'methanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )
        common = {
            'N_stages': 12,
            'feed_stage': 7,
            'reflux_ratio': 2.0,
            'D_mass_to_F_mass': 0.35,
            'P_condenser': 1.0,
            'P_drop_per_stage': 0.0,
            'condenser_type': 'partial',
            'mesh_tolerance': 1e-6,
            'max_iterations': 100,
            'initializer': 'cheap_estimate',
        }
        finite = RigorousDistillation(
            'U_fd', thermo,
            {**common, 'semi_analytic_flow_jacobian': False},
        ).solve({'feed': feed})
        flow_only = RigorousDistillation(
            'U_flow', thermo,
            {**common, 'semi_analytic_local_thermo_jacobian': False},
        ).solve({'feed': feed})
        memory_limited = RigorousDistillation(
            'U_memory', thermo,
            {**common, 'semi_analytic_dense_limit_mb': 1e-9},
        ).solve({'feed': feed})
        semi = RigorousDistillation('U_semi', thermo, common).solve({'feed': feed})

        self.assertEqual(
            finite.performance['jacobian_method'],
            'colored_finite_difference',
        )
        self.assertEqual(
            flow_only.performance['jacobian_method'],
            'semi_analytic_flow',
        )
        self.assertEqual(
            memory_limited.performance['jacobian_method'],
            'semi_analytic_flow',
        )
        self.assertEqual(
            semi.performance['jacobian_method'],
            'semi_analytic_local_thermo',
        )
        self.assertLess(semi.performance['mesh_residual'], 1e-6)
        self.assertLess(
            semi.performance['function_evaluations'],
            flow_only.performance['function_evaluations'],
        )
        self.assertLess(
            flow_only.performance['function_evaluations'],
            finite.performance['function_evaluations'],
        )
        self.assertAlmostEqual(
            semi.outlet_streams['distillate'].composition['methanol'],
            finite.outlet_streams['distillate'].composition['methanol'],
            places=8,
        )

    def test_rigorous_distillation_local_jacobian_matches_all_condenser_and_spec_types(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'methanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )
        for condenser in ('total', 'partial', 'mixed'):
            for mass_spec in (False, True):
                with self.subTest(condenser=condenser, mass_spec=mass_spec):
                    common = {
                        'N_stages': 10,
                        'feed_stage': 6,
                        'reflux_ratio': 2.0,
                        'P_condenser': 1.0,
                        'P_drop_per_stage': 0.0,
                        'condenser_type': condenser,
                        'mesh_tolerance': 1e-6,
                        'max_iterations': 100,
                        'initializer': 'cheap_estimate',
                    }
                    if condenser == 'mixed':
                        common['distillate_vapor_fraction'] = 0.3
                    if mass_spec:
                        common['D_mass_to_F_mass'] = 0.35
                    else:
                        common['D_to_F'] = 0.35

                    local = RigorousDistillation(
                        'LOCAL', thermo, common,
                    ).solve({'feed': feed})
                    original = RigorousDistillation(
                        'ORIGINAL',
                        thermo,
                        {**common, 'semi_analytic_flow_jacobian': False},
                    ).solve({'feed': feed})

                    self.assertEqual(
                        local.performance['jacobian_method'],
                        'semi_analytic_local_thermo',
                    )
                    self.assertEqual(
                        original.performance['jacobian_method'],
                        'colored_finite_difference',
                    )
                    self.assertFalse(local.performance['jacobian_fallback'])
                    self.assertLess(local.performance['jacobian_dense_mb'], 200.0)
                    self.assertLess(
                        local.performance['function_evaluations'],
                        original.performance['function_evaluations'],
                    )
                    self.assertAlmostEqual(
                        local.outlet_streams['distillate'].F,
                        original.outlet_streams['distillate'].F,
                        delta=1e-6,
                    )
                    self.assertAlmostEqual(
                        local.performance['T_top_C'],
                        original.performance['T_top_C'],
                        delta=2e-5,
                    )
                    for comp in ('methanol', 'water'):
                        self.assertAlmostEqual(
                            local.outlet_streams['distillate'].composition[comp],
                            original.outlet_streams['distillate'].composition[comp],
                            delta=2e-7,
                        )
                        self.assertAlmostEqual(
                            local.outlet_streams['bottoms'].composition[comp],
                            original.outlet_streams['bottoms'].composition[comp],
                            delta=2e-7,
                        )

    def test_rigorous_distillation_local_jacobian_failure_retries_original(self):
        import numpy as np
        from scipy.sparse import csr_matrix

        class BrokenLocalJacobianDistillation(RigorousDistillation):
            def _build_mesh_model(self, *args, **kwargs):
                model = super()._build_mesh_model(*args, **kwargs)
                shape = model['sparsity'].shape

                def broken_jacobian(vector, f0, rel_step):
                    matrix = csr_matrix(
                        ([np.nan], ([0], [0])),
                        shape=shape,
                    )
                    return matrix, 0, 'semi_analytic_local_thermo'

                model['jacobian'] = broken_jacobian
                return model

        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'methanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )
        result = BrokenLocalJacobianDistillation(
            'FALLBACK',
            thermo,
            {
                'N_stages': 8,
                'feed_stage': 5,
                'reflux_ratio': 2.0,
                'D_to_F': 0.35,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'mesh_tolerance': 1e-6,
                'max_iterations': 100,
                'initializer': 'cheap_estimate',
            },
        ).solve({'feed': feed})

        self.assertEqual(
            result.performance['jacobian_method'],
            'colored_finite_difference',
        )
        self.assertTrue(result.performance['jacobian_fallback'])
        self.assertTrue(any(
            'retried with the original colored finite-difference Jacobian'
            in warning
            for warning in result.warnings
        ))
        self.assertLess(result.performance['mesh_residual'], 1e-6)

    def test_rigorous_distillation_local_jacobian_matches_thermo_families(self):
        cases = [
            (
                'ideal',
                ['methanol', 'ethanol'],
                'IDEAL',
                {'methanol': 0.3, 'ethanol': 0.7},
                1.0,
            ),
            (
                'gamma_phi',
                ['methanol', 'ethanol', 'water', 'acetone'],
                'UNIQUAC-RK',
                {
                    'methanol': 0.291,
                    'ethanol': 0.665,
                    'water': 0.035,
                    'acetone': 0.009,
                },
                1.0,
            ),
            (
                'eos',
                ['n-hexane', 'toluene'],
                'PR',
                {'n-hexane': 0.4, 'toluene': 0.6},
                1.0,
            ),
        ]
        for name, components, method, composition, pressure in cases:
            with self.subTest(name=name, method=method):
                thermo = create_thermodynamics(components, method)
                feed = thermo.calculate_state(
                    thermo.bubble_point_T(composition, pressure),
                    pressure,
                    100.0,
                    composition,
                    phase='liquid',
                    flash=False,
                )
                common = {
                    'N_stages': 12,
                    'feed_stage': 7,
                    'reflux_ratio': 2.0,
                    'D_to_F': 0.35,
                    'P_condenser': pressure,
                    'P_drop_per_stage': 0.0,
                    'mesh_tolerance': 1e-6,
                    'max_iterations': 150,
                    'initializer': 'cheap_estimate',
                }
                local = RigorousDistillation(
                    'LOCAL', thermo, common,
                ).solve({'feed': feed})
                original = RigorousDistillation(
                    'ORIGINAL',
                    thermo,
                    {**common, 'semi_analytic_flow_jacobian': False},
                ).solve({'feed': feed})

                self.assertEqual(
                    local.performance['jacobian_method'],
                    'semi_analytic_local_thermo',
                )
                self.assertLess(
                    local.performance['function_evaluations'],
                    original.performance['function_evaluations'],
                )
                self.assertAlmostEqual(
                    local.performance['T_top_C'],
                    original.performance['T_top_C'],
                    delta=2e-5,
                )
                for comp in components:
                    self.assertAlmostEqual(
                        local.outlet_streams['distillate'].composition[comp],
                        original.outlet_streams['distillate'].composition[comp],
                        delta=2e-7,
                    )
                    self.assertAlmostEqual(
                        local.outlet_streams['bottoms'].composition[comp],
                        original.outlet_streams['bottoms'].composition[comp],
                        delta=2e-7,
                    )

    def test_rigorous_distillation_sparse_newton_solves_feed_stage_curve(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'methanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )
        purities = []
        for feed_stage in (6, 10, 13, 17):
            result = RigorousDistillation(
                'U', thermo,
                {
                    'N_stages': 20,
                    'feed_stage': feed_stage,
                    'reflux_ratio': 2.0,
                    'D_to_F': 0.4,
                    'P_condenser': 1.0,
                    'P_drop_per_stage': 0.0,
                    'mesh_tolerance': 1e-5,
                },
            ).solve({'feed': feed})
            distillate = result.outlet_streams['distillate']
            bottoms = result.outlet_streams['bottoms']
            self.assertEqual(result.performance['solver'], 'sparse_damped_newton')
            self.assertLess(result.performance['mesh_residual'], 1e-5)
            self.assertLess(relative_component_balance([feed], [distillate, bottoms]), 1e-6)
            purities.append(distillate.composition['methanol'])

        self.assertGreater(purities[1], purities[0])
        self.assertGreater(purities[2], purities[1])
        self.assertGreater(purities[2], purities[3])

    def test_rigorous_distillation_partial_condenser_and_side_draws(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'methanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'U', thermo,
            {
                'N_stages': 16,
                'feed_stage': 8,
                'reflux_ratio': 2.0,
                'D_to_F': 0.35,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'condenser_type': 'partial',
                'side_draws': [
                    {'stage': 7, 'phase': 'liquid', 'flow': 5.0, 'port': 'side_liq'}
                ],
                'mesh_tolerance': 1e-5,
            },
        ).solve({'feed': feed})

        self.assertEqual(result.outlet_streams['distillate'].vapor_fraction, 1.0)
        self.assertEqual(
            result.performance['jacobian_method'],
            'colored_finite_difference',
        )
        self.assertIn('side_liq', result.outlet_streams)
        self.assertAlmostEqual(result.outlet_streams['side_liq'].F, 5.0, places=6)
        self.assertLess(
            relative_component_balance(
                [feed],
                [
                    result.outlet_streams['distillate'],
                    result.outlet_streams['bottoms'],
                    result.outlet_streams['side_liq'],
                ],
            ),
            1e-7,
        )
        self.assertGreater(
            result.outlet_streams['distillate'].composition['methanol'],
            feed.composition['methanol'],
        )

    def test_rigorous_distillation_accepts_multiple_feed_stages(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        methanol_rich = thermo.calculate_state(
            298.15,
            1.0,
            60.0,
            {'methanol': 0.55, 'water': 0.45},
            phase='liquid',
            flash=False,
        )
        water_rich = thermo.calculate_state(
            330.0,
            1.0,
            40.0,
            {'methanol': 0.15, 'water': 0.85},
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'U', thermo,
            {
                'N_stages': 12,
                'feed_stage': 5,
                'water_stage_from_bottom': 3,
                'reflux_ratio': 2.0,
                'D_to_F': 0.35,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'mesh_tolerance': 1e-6,
                'max_iterations': 80,
            },
        ).solve({'feed': methanol_rich, 'water': water_rich})

        self.assertEqual(
            [(feed['port'], feed['stage']) for feed in result.performance['feeds']],
            [('feed', 5), ('water', 10)],
        )
        self.assertLess(result.performance['mesh_residual'], 1e-5)
        self.assertLess(
            relative_component_balance(
                [methanol_rich, water_rich],
                [result.outlet_streams['distillate'], result.outlet_streams['bottoms']],
            ),
            1e-6,
        )
        self.assertGreater(
            result.outlet_streams['distillate'].composition['methanol'],
            result.outlet_streams['bottoms'].composition['methanol'],
        )

    def test_rigorous_distillation_mixed_condenser_has_two_distillate_phases(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'methanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'U', thermo,
            {
                'N_stages': 16,
                'feed_stage': 8,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'condenser_type': 'mixed',
                'distillate_vapor_fraction': 0.25,
                'mesh_tolerance': 1e-5,
            },
        ).solve({'feed': feed})

        self.assertAlmostEqual(result.outlet_streams['distillate'].vapor_fraction, 0.25)
        self.assertIn('distillate_liquid', result.outlet_streams)
        self.assertIn('distillate_vapor', result.outlet_streams)
        self.assertAlmostEqual(
            result.outlet_streams['distillate_liquid'].F
            + result.outlet_streams['distillate_vapor'].F,
            result.outlet_streams['distillate'].F,
            places=6,
        )
        self.assertLess(
            relative_component_balance(
                [feed],
                [result.outlet_streams['distillate'], result.outlet_streams['bottoms']],
            ),
            1e-5,
        )

    def test_rigorous_distillation_top_decanter_selects_reflux_phase_and_purge(self):
        thermo = create_thermodynamics(['ethanol', 'water', 'benzene'], 'UNIFAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            20.0,
            {
                'ethanol': 0.35,
                'water': 0.25,
                'benzene': 0.40,
            },
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'AZD', thermo,
            {
                'N_stages': 16,
                'feed_stage': 8,
                'condenser_type': 'decanter',
                'decanter_reflux_component': 'benzene',
                'decanter_reflux_purge_fraction': 0.02,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'mesh_tolerance': 1e-5,
                'max_iterations': 80,
                'max_jacobian_evaluations': 80,
            },
        ).solve({'feed': feed})

        decanter = result.performance['top_decanter']
        self.assertEqual(result.performance['initializer'], 'estimate')
        self.assertEqual(
            result.performance['jacobian_method'],
            'colored_finite_difference',
        )
        self.assertTrue(decanter['two_phases'])
        self.assertIn('decanter_purge', result.outlet_streams)
        self.assertAlmostEqual(
            result.outlet_streams['distillate'].F,
            0.4 * feed.F,
            places=5,
        )
        self.assertGreater(
            decanter['reflux_composition']['benzene'],
            decanter['distillate_composition']['benzene'],
        )
        self.assertAlmostEqual(
            result.outlet_streams['decanter_purge'].F,
            0.02 * decanter['reflux_raw_flow'],
            places=6,
        )
        self.assertLess(
            relative_component_balance(
                [feed],
                [
                    result.outlet_streams['distillate'],
                    result.outlet_streams['bottoms'],
                    result.outlet_streams['decanter_purge'],
                ],
            ),
            1e-5,
        )

    def test_rigorous_distillation_rejects_infeasible_external_flow_specs(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'methanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )

        with self.assertRaisesRegex(UnitOperationError, 'distillate mass flow is infeasible'):
            RigorousDistillation(
                'U', thermo,
                {
                    'N_stages': 8,
                    'feed_stage': 4,
                    'reflux_ratio': 2.0,
                    'top_flow': feed.mass_flow() * 1.01,
                    '__unit__top_flow': 'kg/h',
                },
            ).solve({'feed': feed})

        with self.assertRaisesRegex(UnitOperationError, 'no inlet stream'):
            RigorousDistillation(
                'U', thermo,
                {'N_stages': 8, 'feed_stage': 4, 'reflux_ratio': 2.0, 'D_to_F': 0.4},
            ).solve({})

        with self.assertRaisesRegex(UnitOperationError, 'leave a positive bottoms flow'):
            RigorousDistillation(
                'U', thermo,
                {
                    'N_stages': 8,
                    'feed_stage': 4,
                    'reflux_ratio': 2.0,
                    'D_to_F': 0.6,
                    'side_draws': [{'stage': 5, 'phase': 'liquid', 'flow': 45.0}],
                },
            ).solve({'feed': feed})

        with self.assertRaisesRegex(UnitOperationError, 'fractions .* must sum to less than 1'):
            RigorousDistillation(
                'U', thermo,
                {
                    'N_stages': 8,
                    'feed_stage': 4,
                    'reflux_ratio': 2.0,
                    'D_to_F': 0.4,
                    'side_draws': [
                        {'stage': 5, 'phase': 'liquid', 'fraction': 0.6},
                        {'stage': 5, 'phase': 'liquid', 'fraction': 0.5},
                    ],
                },
            ).solve({'feed': feed})

    def test_rigorous_distillation_fails_fast_for_noncondensables(self):
        thermo = create_thermodynamics(['methanol', 'water', 'CO', 'H2'], 'UNIFAC')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'methanol': 0.4, 'water': 0.598, 'CO': 0.001, 'H2': 0.001},
            phase='liquid',
            flash=False,
        )
        with self.assertRaisesRegex(UnitOperationError, 'non-condensable'):
            RigorousDistillation(
                'U', thermo,
                {
                    'N_stages': 20,
                    'feed_stage': 10,
                    'reflux_ratio': 2.0,
                    'D_to_F': 0.4,
                    'P_condenser': 1.0,
                },
            ).solve({'feed': feed})

        partial = RigorousDistillation(
            'U', thermo,
            {
                'N_stages': 12,
                'feed_stage': 6,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'condenser_type': 'partial',
                'initializer': 'estimate',
                'mesh_tolerance': 1e-5,
            },
        ).solve({'feed': feed})
        self.assertIn('likely non-condensable', ' '.join(partial.warnings))
        self.assertGreater(partial.outlet_streams['distillate'].composition['CO'], 0.0)

        tiny_feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'methanol': 0.4, 'water': 0.599999999, 'CO': 1e-9, 'H2': 0.0},
            phase='liquid',
            flash=False,
        )
        tiny = RigorousDistillation(
            'U', thermo,
            {
                'N_stages': 8,
                'feed_stage': 4,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'mesh_tolerance': 1e-5,
            },
        ).solve({'feed': tiny_feed})
        self.assertLess(tiny.outlet_streams['distillate'].composition.get('CO', 0.0), 1e-7)

    def test_rigorous_distillation_methanol_trace_contaminants(self):
        cases = [
            (
                ['methanol', 'water', 'ethanol'],
                {'methanol': 0.4, 'water': 0.595, 'ethanol': 0.005},
                13,
                'ethanol',
                {},
            ),
            (
                ['methanol', 'water', 'acetaldehyde'],
                {'methanol': 0.4, 'water': 0.595, 'acetaldehyde': 0.005},
                13,
                'acetaldehyde',
                {
                    'condenser_type': 'mixed',
                    'condenser_vapor_fraction': 0.01,
                    'initializer': 'cheap_estimate',
                },
            ),
        ]
        for components, composition, feed_stage, light_trace, column_options in cases:
            with self.subTest(trace=light_trace, components=components):
                thermo = create_thermodynamics(components, 'UNIFAC')
                feed = thermo.calculate_state(
                    298.15,
                    1.0,
                    100.0,
                    composition,
                    phase='liquid',
                    flash=False,
                )
                params = {
                    'N_stages': 20,
                    'feed_stage': feed_stage,
                    'reflux_ratio': 2.0,
                    'D_to_F': 0.4,
                    'P_condenser': 1.0,
                    'P_drop_per_stage': 0.0,
                    'mesh_tolerance': 1e-5,
                    'max_iterations': 120,
                }
                params.update(column_options)
                result = RigorousDistillation('U', thermo, params).solve({'feed': feed})

                self.assertLess(result.performance['mesh_residual'], 5e-4)
                self.assertLess(
                    relative_component_balance(
                        [feed],
                        [result.outlet_streams['distillate'], result.outlet_streams['bottoms']],
                    ),
                    1e-4,
                )
                self.assertGreater(
                    result.outlet_streams['distillate'].composition['methanol'],
                    feed.composition['methanol'],
                )
                self.assertGreater(
                    result.outlet_streams['distillate'].composition[light_trace],
                    feed.composition[light_trace],
                )

    def test_rigorous_distillation_accepts_mass_feed_and_distillate_specs(self):
        MW = {'methanol': 32.04, 'water': 18.015}
        w_methanol = 0.30
        mole_methanol = (w_methanol / MW['methanol']) / (
            w_methanol / MW['methanol'] + (1.0 - w_methanol) / MW['water']
        )
        feed_mass = 100.0 * (
            mole_methanol * MW['methanol'] + (1.0 - mole_methanol) * MW['water']
        )

        cases = [
            ('mass_fraction', 'D_mass_to_F_mass = 0.30', 0.30 * feed_mass),
            (
                'absolute_mass',
                f'distillate_mass_flow = {0.30 * feed_mass:.12g} [kg/h]',
                0.30 * feed_mass,
            ),
        ]
        for name, distillate_spec, target_mass in cases:
            with self.subTest(spec=name):
                result = Simulator.from_string(
                    'PROCESS: Methanol Water Mass Distillate Test\n'
                    'VERSION: 1.0\n'
                    'THERMO_METHOD: NRTL\n'
                    '\n'
                    'COMPONENTS:\n'
                    '    methanol | Methanol | MW=32.04\n'
                    '    water | Water | MW=18.015\n'
                    '\n'
                    'STREAM Feed : FEED -> COL.feed\n'
                    '    T = 25 [C]\n'
                    '    P = 1 [bar]\n'
                    f'    F_mass = {feed_mass:.12g} [kg/h]\n'
                    '    w = methanol:0.30, water:0.70\n'
                    '\n'
                    'STREAM Distillate : COL.distillate -> PRODUCT\n'
                    'STREAM Bottoms : COL.bottoms -> PRODUCT\n'
                    '\n'
                    'UNIT COL\n'
                    '    TYPE: RigorousDistillation\n'
                    '    PORTS:\n'
                    '        feed : inlet\n'
                    '        distillate : outlet\n'
                    '        bottoms : outlet\n'
                    '    PARAMS:\n'
                    '        N_stages = 20\n'
                    '        feed_stage = 10\n'
                    '        reflux_ratio = 2.0\n'
                    f'        {distillate_spec}\n'
                    '        P_condenser = 1 [bar]\n'
                    '        P_drop_per_stage = 0.0\n'
                    '        condenser_type = total\n'
                    '        initializer = auto\n'
                    '        mesh_tolerance = 1e-6\n'
                    '        max_iterations = 160\n'
                ).run()

                self.assertTrue(result.converged, result.errors)
                feed = result.streams['Feed']
                distillate = result.streams['Distillate']
                bottoms = result.streams['Bottoms']
                self.assertAlmostEqual(feed.F, 100.0, places=5)
                self.assertAlmostEqual(feed.composition['methanol'], mole_methanol, places=7)
                self.assertAlmostEqual(distillate.mass_flow(), target_mass, places=4)
                self.assertAlmostEqual(distillate.mass_flow() / feed.mass_flow(), 0.30, places=7)
                self.assertGreater(distillate.composition['methanol'], feed.composition['methanol'])
                self.assertLess(bottoms.composition['methanol'], feed.composition['methanol'])

    def test_rigorous_distillation_ethanol_benzene_azeotropic_case(self):
        binary = create_thermodynamics(['ethanol', 'water'], 'UNIFAC')
        wet_feed = binary.calculate_state(
            298.15, 1.0, 100.0, {'ethanol': 0.1, 'water': 0.9}, phase='liquid'
        )
        beer_column = RigorousDistillation(
            'ETH', binary,
            {
                'N_stages': 30,
                'feed_stage': 21,
                'reflux_ratio': 8.0,
                'D': 11.0,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'mesh_tolerance': 1e-5,
            },
        ).solve({'feed': wet_feed})
        azeotrope_product = beer_column.outlet_streams['distillate']
        self.assertAlmostEqual(azeotrope_product.composition['ethanol'], 0.883, delta=0.02)

        ternary = create_thermodynamics(['ethanol', 'water', 'benzene'], 'UNIFAC')
        ternary_ethanol = ternary.calculate_state(
            azeotrope_product.T,
            azeotrope_product.P,
            azeotrope_product.F,
            azeotrope_product.composition,
            phase='liquid',
            flash=False,
        )
        benzene = ternary.calculate_state(
            298.15, 1.0, 6.0, {'benzene': 1.0}, phase='liquid', flash=False
        )
        mixed = Mixer('M', ternary, {'mode': 'adiabatic'}).solve(
            {'ethanol_feed': ternary_ethanol, 'benzene_feed': benzene}
        ).outlet_streams['out']

        entrainer_column = RigorousDistillation(
            'AZE', ternary,
            {
                'N_stages': 16,
                'feed_stage': 8,
                'reflux_ratio': 4.0,
                'D': 10.8,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
                'stage_phase_model': 'VLLE',
                'mesh_tolerance': 1e-5,
            },
        ).solve({'feed': mixed})
        distillate = entrainer_column.outlet_streams['distillate']
        bottoms = entrainer_column.outlet_streams['bottoms']

        self.assertLess(entrainer_column.performance['mesh_residual'], 1e-5)
        self.assertLess(relative_component_balance([mixed], [distillate, bottoms]), 1e-7)
        self.assertGreater(distillate.composition['benzene'], mixed.composition['benzene'])
        self.assertGreater(distillate.composition['water'], mixed.composition['water'])
        self.assertLess(bottoms.composition['water'], 0.001)

    def test_trace_impurities_keep_effectively_binary_balance_rows(self):
        for method in ('UNIFAC', 'UNIQUAC'):
            with self.subTest(method=method):
                thermo = create_thermodynamics(
                    ['methanol', 'water', 'ethanol', 'isopropanol', 'CO', 'H2'],
                    method,
                )
                feed = thermo.calculate_state(
                    298.15,
                    1.0,
                    100.0,
                    {
                        'methanol': 0.400,
                        'water': 0.591,
                        'ethanol': 0.005,
                        'isopropanol': 0.002,
                        'CO': 0.001,
                        'H2': 0.001,
                    },
                    phase='liquid',
                    flash=False,
                )
                result = RigorousDistillation(
                    'U', thermo,
                    {
                        'N_stages': 6,
                        'feed_stage': 3,
                        'reflux_ratio': 2.0,
                        'D_to_F': 0.4,
                        'P_condenser': 1.0,
                        'P_drop_per_stage': 0.0,
                        'max_evaluations': 20,
                        'mesh_tolerance': 5.0,
                        'condenser_type': 'partial',
                    },
                ).solve({'feed': feed})

                if method == 'UNIFAC':
                    self.assertEqual(
                        sorted(thermo.unifac_excluded_components),
                        ['CO', 'H2'],
                    )
                else:
                    self.assertEqual(
                        sorted(thermo._deferred_uniquac_rq_errors),
                        ['CO', 'H2'],
                    )
                self.assertIn('likely non-condensable', ' '.join(result.warnings))
                self.assertIn('component', result.performance['residual_diagnostics'])
                self.assertIn('energy', result.performance['residual_diagnostics'])

    def test_rigorous_distillation_solves_multicomponent_mesh_balances(self):
        thermo = create_thermodynamics(['CH3OH', 'C2H5OH', 'H2O'], 'IDEAL')
        feed = thermo.calculate_state(
            330.0,
            1.0,
            100.0,
            {'CH3OH': 0.2, 'C2H5OH': 0.3, 'H2O': 0.5},
            phase='liquid',
        )
        unit = RigorousDistillation(
            'U', thermo,
            {
                'N_stages': 5,
                'feed_stage': 3,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
            },
        )

        result = unit.solve({'feed': feed})
        distillate = result.outlet_streams['distillate']
        bottoms = result.outlet_streams['bottoms']

        self.assertLess(result.performance['mesh_residual'], 1e-6)
        self.assertLess(relative_component_balance([feed], [distillate, bottoms]), 1e-7)
        self.assertEqual(set(distillate.composition), {'CH3OH', 'C2H5OH', 'H2O'})
        self.assertEqual(set(bottoms.composition), {'CH3OH', 'C2H5OH', 'H2O'})
        self.assertGreater(distillate.composition['CH3OH'], feed.composition['CH3OH'])
        self.assertGreater(distillate.composition['C2H5OH'], feed.composition['C2H5OH'])
        self.assertGreater(bottoms.composition['H2O'], feed.composition['H2O'])

    def test_ethanol_water_azeotrope_and_benzene_entrainer(self):
        binary = create_thermodynamics(['ethanol', 'water'], 'UNIFAC')
        azeotrope = binary.generate_Txy_data('ethanol', 'water', 1.0, n_points=100)['azeotrope']
        self.assertIsNotNone(azeotrope)
        self.assertAlmostEqual(azeotrope['x'], 0.883, delta=0.02)
        self.assertAlmostEqual(azeotrope['T'], 77.7, delta=1.0)

        wet_feed = binary.calculate_state(
            298.15, 1.0, 100.0, {'ethanol': 0.1, 'water': 0.9}, phase='liquid'
        )
        azeotrope_column = McCabeThieleDistillation(
            'U', binary,
            {
                'N_stages': 40,
                'reflux_ratio': 8.0,
                'D': 11.3,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
            },
        ).solve({'feed': wet_feed})
        azeotrope_product = azeotrope_column.outlet_streams['distillate']
        self.assertAlmostEqual(azeotrope_product.composition['ethanol'], azeotrope['x'], delta=0.02)
        self.assertLess(azeotrope_product.composition['ethanol'], 0.91)

        ternary = create_thermodynamics(['ethanol', 'water', 'benzene'], 'UNIFAC')
        azeotrope_moles = 100.0
        benzene_moles = 50.0
        ternary_feed_comp = {
            'ethanol': azeotrope_product.composition['ethanol'] * azeotrope_moles,
            'water': azeotrope_product.composition['water'] * azeotrope_moles,
            'benzene': benzene_moles,
        }
        total = sum(ternary_feed_comp.values())
        ternary_feed_comp = {comp: value / total for comp, value in ternary_feed_comp.items()}
        ternary_feed_T = ternary.bubble_point_T(ternary_feed_comp, 1.0)
        ternary_feed = ternary.calculate_state(
            ternary_feed_T, 1.0, total, ternary_feed_comp, phase='liquid'
        )

        entrainer_column = CMODistillation(
            'U', ternary,
            {
                'N_stages': 30,
                'reflux_ratio': 4.0,
                'D_to_F': 0.5,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
            },
        ).solve({'feed': ternary_feed})
        distillate = entrainer_column.outlet_streams['distillate']
        bottoms = entrainer_column.outlet_streams['bottoms']

        self.assertLess(relative_component_balance([ternary_feed], [distillate, bottoms]), 1e-8)
        self.assertGreater(distillate.composition['benzene'], ternary_feed.composition['benzene'])
        self.assertGreater(distillate.composition['water'], ternary_feed.composition['water'])
        self.assertGreater(bottoms.composition['ethanol'], ternary_feed.composition['ethanol'])
        self.assertLess(bottoms.composition['water'], 0.001)

    def test_extractor_uses_solvent_rich_phase_and_subunit_extraction_factor(self):
        components = ['diethyl ether', 'n-hexane', 'acrylic acid', 'water']
        thermo = create_thermodynamics(components, 'UNIFAC')

        def stream(masses_kg_h):
            molar = {
                comp: masses_kg_h.get(comp, 0.0) * 1000.0 / thermo.props[comp].MW
                for comp in components
            }
            total = sum(molar.values())
            composition = {comp: value / total for comp, value in molar.items()}
            return thermo.calculate_state(
                298.15, 1.0, total, composition, phase='liquid', flash=False
            )

        feed = stream({'acrylic acid': 200.0, 'water': 1500.0})
        solvent = stream({'diethyl ether': 250.0, 'n-hexane': 250.0})

        result = ShortcutExtractor(
            'U', thermo, {'N_stages': 20, 'T': 25}
        ).solve({'feed': feed, 'solvent': solvent})

        extract = result.outlet_streams['extract']
        raffinate = result.outlet_streams['raffinate']
        acid_extracted = (
            extract.F
            * extract.composition.get('acrylic acid', 0.0)
            * thermo.props['acrylic acid'].MW
            / 1000.0
        )

        self.assertGreater(result.performance['K_dist']['acrylic acid'], 1.0)
        self.assertGreater(acid_extracted, 40.0)
        self.assertLess(acid_extracted, 60.0)
        self.assertGreater(extract.composition['diethyl ether'], raffinate.composition['diethyl ether'])
        self.assertGreater(raffinate.composition['water'], extract.composition['water'])

    def test_extractors_report_clean_input_and_lle_capability_errors(self):
        with self.assertRaisesRegex(UnitOperationError, 'needs feed and solvent'):
            ShortcutExtractor('U', self.ideal, {}).solve({})
        with self.assertRaisesRegex(UnitOperationError, 'needs feed and solvent'):
            RigorousLiquidLiquidExtractor('U', self.ideal, {}).solve({})

        for cls in (ShortcutExtractor, RigorousLiquidLiquidExtractor):
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(UnitOperationError, 'LLE-capable activity model') as raised:
                    cls('U', self.ideal, {}).solve({
                        'feed': self.liquid,
                        'solvent': self.liquid2,
                    })
                self.assertNotIn('UNIFAC thermo', str(raised.exception))

    def test_rigorous_extractor_closes_countercurrent_lle_stages(self):
        components = ['diethyl ether', 'n-hexane', 'acrylic acid', 'water']
        thermo = create_thermodynamics(components, 'UNIFAC')

        def stream(masses_kg_h):
            molar = {
                comp: masses_kg_h.get(comp, 0.0) * 1000.0 / thermo.props[comp].MW
                for comp in components
            }
            total = sum(molar.values())
            composition = {comp: value / total for comp, value in molar.items() if value > 0.0}
            return thermo.calculate_state(
                298.15, 1.0, total, composition, phase='liquid', flash=False
            )

        feed = stream({'acrylic acid': 200.0, 'water': 1500.0})
        solvent = stream({'diethyl ether': 250.0, 'n-hexane': 250.0})
        result = RigorousLiquidLiquidExtractor(
            'U', thermo, {'N_stages': 3, 'T': 25, 'max_iterations': 80}
        ).solve({'feed': feed, 'solvent': solvent})

        extract = result.outlet_streams['extract']
        acid_extracted = (
            extract.F
            * extract.composition.get('acrylic acid', 0.0)
            * thermo.props['acrylic acid'].MW
            / 1000.0
        )

        self.assertEqual(result.performance['solver'], 'sparse_damped_newton')
        self.assertEqual(result.performance['solver_algorithm'], 'equation_oriented')
        self.assertEqual(result.performance['requested_initializer'], 'auto')
        self.assertEqual(result.performance['initializer'], 'split_sweep')
        self.assertLess(result.performance['mesh_residual'], 1e-5)
        self.assertLess(result.performance['component_balance_error'], 1e-5)
        self.assertGreater(acid_extracted, 55.0)
        self.assertLess(acid_extracted, 65.0)

    def test_equation_oriented_rigorous_extractor_matches_split_sweep(self):
        components = ['diethyl ether', 'n-hexane', 'acrylic acid', 'water']
        thermo = create_thermodynamics(components, 'UNIFAC')

        def stream(masses_kg_h):
            molar = {
                comp: masses_kg_h.get(comp, 0.0) * 1000.0 / thermo.props[comp].MW
                for comp in components
            }
            total = sum(molar.values())
            composition = {comp: value / total for comp, value in molar.items() if value > 0.0}
            return thermo.calculate_state(
                298.15, 1.0, total, composition, phase='liquid', flash=False
            )

        feed = stream({'acrylic acid': 200.0, 'water': 1500.0})
        solvent = stream({'diethyl ether': 250.0, 'n-hexane': 250.0})
        common = {'N_stages': 3, 'T': 25, 'max_iterations': 80}
        split = RigorousLiquidLiquidExtractor(
            'U_split', thermo, {**common, 'solver_algorithm': 'split_sweep'}
        ).solve({'feed': feed, 'solvent': solvent})
        mesh = RigorousLiquidLiquidExtractor(
            'U_mesh', thermo, {**common, 'solver_algorithm': 'equation_oriented'}
        ).solve({'feed': feed, 'solvent': solvent})

        self.assertEqual(mesh.performance['solver'], 'sparse_damped_newton')
        self.assertEqual(mesh.performance['solver_algorithm'], 'equation_oriented')
        self.assertIn('deprecated', split.warnings[0])
        self.assertLess(mesh.performance['mesh_residual'], 1e-6)
        self.assertLess(mesh.performance['component_balance_error'], 1e-7)
        self.assertLess(
            mesh.performance['function_evaluations'],
            split.performance['solver_iterations'] * 10,
        )
        for stream_name in ('extract', 'raffinate'):
            split_stream = split.outlet_streams[stream_name]
            mesh_stream = mesh.outlet_streams[stream_name]
            self.assertLess(
                abs(mesh_stream.F - split_stream.F) / max(split_stream.F, 1.0),
                2e-5,
            )
            for comp in components:
                self.assertAlmostEqual(
                    mesh_stream.composition.get(comp, 0.0),
                    split_stream.composition.get(comp, 0.0),
                    delta=2e-5,
                )

    def test_equation_oriented_extractor_initializers_converge(self):
        components = ['pyridine', 'water', 'diethyl ether']
        thermo = create_thermodynamics(components, 'UNIFNIST')

        def stream(masses_kg_h, T):
            molar = {
                comp: masses_kg_h.get(comp, 0.0) * 1000.0 / thermo.props[comp].MW
                for comp in components
            }
            total = sum(molar.values())
            composition = {comp: value / total for comp, value in molar.items() if value > 0.0}
            return thermo.calculate_state(
                T, 1.0, total, composition, phase='liquid', flash=False
            )

        feed = stream({'pyridine': 100.0, 'water': 900.0}, 303.15)
        solvent = stream({'diethyl ether': 500.0}, 293.15)
        recoveries = []
        for initializer in ('auto', 'split_sweep', 'coarse_grid', 'homotopy'):
            with self.subTest(initializer=initializer):
                result = RigorousLiquidLiquidExtractor(
                    'U', thermo,
                    {
                        'N_stages': 4,
                        'mode': 'adiabatic',
                        'T': 298.15,
                        'initializer': initializer,
                        'max_iterations': 80,
                        'max_jacobian_evaluations': 60,
                    },
                ).solve({'feed': feed, 'solvent': solvent})

                performance = result.performance
                extract = result.outlet_streams['extract']
                recovery = (
                    extract.F
                    * extract.composition.get('pyridine', 0.0)
                    / (feed.F * feed.composition.get('pyridine', 1.0))
                )
                recoveries.append(recovery)
                self.assertEqual(performance['requested_initializer'], initializer)
                expected_initializer = 'split_sweep' if initializer == 'auto' else initializer
                self.assertEqual(performance['initializer'], expected_initializer)
                self.assertLess(performance['mesh_residual'], 1e-6)
                self.assertLess(performance['overall_energy_relative_error'], 1e-8)

        self.assertLess(max(recoveries) - min(recoveries), 1e-8)

    def test_auto_initializer_uses_coarse_grid_for_larger_extractors(self):
        components = ['pyridine', 'water', 'diethyl ether']
        thermo = create_thermodynamics(components, 'UNIFNIST')

        def stream(masses_kg_h, T):
            molar = {
                comp: masses_kg_h.get(comp, 0.0) * 1000.0 / thermo.props[comp].MW
                for comp in components
            }
            total = sum(molar.values())
            composition = {comp: value / total for comp, value in molar.items() if value > 0.0}
            return thermo.calculate_state(
                T, 1.0, total, composition, phase='liquid', flash=False
            )

        feed = stream({'pyridine': 100.0, 'water': 900.0}, 303.15)
        solvent = stream({'diethyl ether': 500.0}, 293.15)
        result = RigorousLiquidLiquidExtractor(
            'U', thermo,
            {
                'N_stages': 20,
                'mode': 'adiabatic',
                'T': 298.15,
                'max_iterations': 80,
                'max_jacobian_evaluations': 60,
            },
        ).solve({'feed': feed, 'solvent': solvent})

        performance = result.performance
        self.assertEqual(performance['requested_initializer'], 'auto')
        self.assertEqual(performance['initializer'], 'coarse_grid')
        self.assertEqual(performance['coarse_initial_stages'], 8)
        self.assertLess(performance['mesh_residual'], 1e-6)
        self.assertLess(performance['overall_energy_relative_error'], 1e-8)

    def test_adiabatic_rigorous_extractor_closes_stage_energy_balances(self):
        components = ['pyridine', 'water', 'diethyl ether']
        thermo = create_thermodynamics(components, 'UNIFNIST')

        def stream(masses_kg_h, T):
            molar = {
                comp: masses_kg_h.get(comp, 0.0) * 1000.0 / thermo.props[comp].MW
                for comp in components
            }
            total = sum(molar.values())
            composition = {comp: value / total for comp, value in molar.items() if value > 0.0}
            return thermo.calculate_state(
                T, 1.0, total, composition, phase='liquid', flash=False
            )

        feed = stream({'pyridine': 100.0, 'water': 900.0}, 303.15)
        solvent = stream({'diethyl ether': 500.0}, 293.15)
        result = RigorousLiquidLiquidExtractor(
            'U', thermo,
            {'N_stages': 3, 'mode': 'adiabatic', 'T': 298.15, 'max_iterations': 80}
        ).solve({'feed': feed, 'solvent': solvent})

        performance = result.performance

        self.assertEqual(performance['mode'], 'adiabatic')
        self.assertEqual(result.heat_duty, 0.0)
        self.assertLess(performance['component_balance_error'], 1e-5)
        self.assertLess(performance['max_stage_energy_relative_error'], 1e-5)
        self.assertLess(performance['overall_energy_relative_error'], 1e-5)
        self.assertGreater(performance['extract_T_C'], performance['raffinate_T_C'])

if __name__ == '__main__':
    unittest.main()
