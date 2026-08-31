import math
import unittest

from pfd_parser import ParseError, ProcessFlowDiagram, parse_pfd
from simulator import SimulationError, Simulator
from thermodynamics import (
    ActivityCoefficientThermodynamics,
    StreamState,
    VLLEFlashResult,
    create_thermodynamics,
)
from unit_operations_basic import Flash, Heater
from unit_operations_distillation import RigorousDistillation


class FluidPhaseMetadataTests(unittest.TestCase):
    def test_metadata_defaults_to_vle_and_round_trips_all_public_modes(self):
        default = parse_pfd('PROCESS: default\n')
        self.assertEqual(default.metadata.fluid_phase_model, 'VLE')
        self.assertNotIn('FLUID_PHASE_MODEL:', default.metadata.to_pfd())

        for declared, canonical in (
            ('VLE', 'VLE'),
            ('VL(L)E', 'VL(L)E'),
            ('adaptive_vlle', 'VL(L)E'),
            ('VLLE', 'VLLE'),
        ):
            with self.subTest(declared=declared):
                pfd = parse_pfd(
                    f'PROCESS: phase model\nFLUID_PHASE_MODEL: {declared}\n'
                )
                self.assertEqual(pfd.metadata.fluid_phase_model, canonical)
                restored = ProcessFlowDiagram.from_dict(pfd.to_dict())
                self.assertEqual(restored.metadata.fluid_phase_model, canonical)
                if canonical == 'VLE':
                    self.assertNotIn('FLUID_PHASE_MODEL:', pfd.metadata.to_pfd())
                else:
                    self.assertIn(
                        f'FLUID_PHASE_MODEL: {canonical}',
                        pfd.metadata.to_pfd(),
                    )

    def test_invalid_fluid_phase_model_is_rejected(self):
        solid_requests = (
            'SLE',
            'SVLE',
            'S/L/L/E',
            'solid liquid equilibrium',
            'crystallization',
            'precipitation',
            'dissolution',
            'solubility',
            'freezing',
            'sublimation',
        )
        for model in solid_requests:
            with self.subTest(model=model):
                with self.assertRaises(ParseError) as caught:
                    parse_pfd(
                        f'PROCESS: invalid\nFLUID_PHASE_MODEL: {model}\n'
                    )
                message = str(caught.exception)
                self.assertIn(
                    f"Solid-equilibrium phase model '{model}' is not supported",
                    message,
                )
                self.assertIn('phase_behavior=permanent_solid', message)
                self.assertIn('solid_material_form=crystalline', message)
                self.assertIn('phase_behavior=soluble_solid', message)
                self.assertIn('implemented in the future', message)
                self.assertNotIn('Did you mean', message)

        with self.assertRaisesRegex(
            ParseError,
            r"Unknown fluid phase model 'VLLEE'\. Did you mean 'VLLE'\?",
        ):
            parse_pfd('PROCESS: typo\nFLUID_PHASE_MODEL: VLLEE\n')

    def test_non_lle_thermodynamics_rejects_lle_aware_global_mode(self):
        text = (
            'PROCESS: invalid phase model\n'
            'THERMO_METHOD: IDEAL\n'
            'FLUID_PHASE_MODEL: VLLE\n'
            'ONLINE_LOOKUP: false\n'
            'COMPONENTS:\n'
            '    H2O | Water\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = H2O:1\n'
        )
        with self.assertRaisesRegex(SimulationError, 'requires an LLE-capable'):
            Simulator.from_string(text)

    def test_simulator_applies_declared_mode_to_lle_thermodynamics(self):
        text = (
            'PROCESS: adaptive phase model\n'
            'THERMO_METHOD: NRTL\n'
            'FLUID_PHASE_MODEL: VL(L)E\n'
            'ONLINE_LOOKUP: false\n'
            'COMPONENTS:\n'
            '    H2O | Water\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = H2O:1\n'
        )
        simulator = Simulator.from_string(text).initialize()
        self.assertEqual(simulator.pfd.metadata.fluid_phase_model, 'VL(L)E')
        self.assertEqual(simulator.thermo.fluid_phase_model, 'VL(L)E')
        adaptive_thermo = simulator.thermo
        simulator.pfd.metadata.fluid_phase_model = 'VLLE'
        simulator.initialize()
        self.assertIsNot(simulator.thermo, adaptive_thermo)
        self.assertEqual(simulator.thermo.fluid_phase_model, 'VLLE')
        result = simulator.run()
        self.assertTrue(result.converged, result.errors)
        payload = simulator.get_results_dict()
        self.assertEqual(payload['metadata']['fluid_phase_model'], 'VLLE')
        self.assertNotIn('solid_component_flows', payload['streams']['Feed'])
        report = simulator._generate_pfr()
        self.assertIn('FLUID_PHASE_MODEL: VLLE', report)
        self.assertNotIn('solid_fraction = 0.0000', report)


class StreamPhaseInventoryTests(unittest.TestCase):
    def test_legacy_state_has_explicit_empty_liquid2_and_solid_views(self):
        state = StreamState(
            T=300.0,
            P=1.0,
            F=10.0,
            composition={'a': 0.4, 'b': 0.6},
            vapor_fraction=0.25,
            x={'a': 0.5, 'b': 0.5},
            y={'a': 0.1, 'b': 0.9},
        )
        self.assertEqual(
            state.phase_fractions(),
            {'vapor': 0.25, 'liquid1': 0.75, 'liquid2': 0.0, 'solid': 0.0},
        )
        self.assertEqual(state.solid_component_flows, {})
        serialized = state.to_dict()
        self.assertNotIn('x1', serialized)
        self.assertNotIn('x2', serialized)
        self.assertNotIn('liquid2_fraction', serialized)
        self.assertNotIn('solid_component_flows', serialized)
        self.assertNotIn('solid_fraction', serialized)
        self.assertAlmostEqual(
            serialized['phase_component_flows']['vapor']['a'],
            0.25,
        )

        copied = state.copy()
        copied.solid_component_flows['future-solid'] = 1.0
        self.assertEqual(state.solid_component_flows, {})

    def test_multifluid_state_preserves_pooled_and_distinct_liquids(self):
        state = StreamState(
            T=300.0,
            P=1.0,
            F=10.0,
            composition={'a': 0.5, 'b': 0.5},
            vapor_fraction=0.2,
            liquid1_fraction=0.3,
            liquid2_fraction=0.5,
            x={'a': 0.5, 'b': 0.5},
            x1={'a': 0.9, 'b': 0.1},
            x2={'a': 0.26, 'b': 0.74},
            y={'a': 0.1, 'b': 0.9},
            fluid_phase_model='VLLE',
            phase_status='three_phase',
        )
        self.assertAlmostEqual(sum(state.phase_fractions().values()), 1.0)
        copied = state.copy()
        self.assertEqual(copied.x1, state.x1)
        self.assertEqual(copied.x2, state.x2)
        self.assertEqual(copied.fluid_phase_model, 'VLLE')
        phase_flows = state.phase_component_flows()
        self.assertAlmostEqual(sum(phase_flows['vapor'].values()), 2.0)
        self.assertAlmostEqual(sum(phase_flows['liquid1'].values()), 3.0)
        self.assertAlmostEqual(sum(phase_flows['liquid2'].values()), 5.0)


class _RegularSolutionThermodynamics(ActivityCoefficientThermodynamics):
    def __init__(self, interaction_parameter: float):
        # This small analytic fixture exercises phase-policy dispatch without
        # loading the component/property databases.
        self.components = ['a', 'b']
        self.interaction_parameter = float(interaction_parameter)
        self.activity_calls = 0
        self.flash3_calls = 0
        self.fluid_phase_model = 'VLE'
        self._compiled_vlle_initialized = True
        self._compiled_vlle = None

    def activity_coefficients(self, T, composition):
        self.activity_calls += 1
        x_a = float(composition.get('a', 0.0))
        x_b = float(composition.get('b', 0.0))
        A = self.interaction_parameter
        return {
            'a': math.exp(A * x_b * x_b),
            'b': math.exp(A * x_a * x_a),
        }

    def flash_TP(self, composition, T, P):
        z = self._normalize_phase_composition(composition)
        return 0.0, z, z

    def flash3_TP(self, composition, T, P, max_iter=100, tol=1e-9):
        self.flash3_calls += 1
        z = self._normalize_phase_composition(composition)
        return VLLEFlashResult(
            phase_count=2,
            status='lle_only',
            vapor_fraction=0.0,
            liquid1_fraction=0.5,
            liquid2_fraction=0.5,
            y=z,
            x1={'a': 0.9, 'b': 0.1},
            x2={'a': 0.1, 'b': 0.9},
            residual=0.0,
            iterations=1,
        )


class FluidPhasePolicyTests(unittest.TestCase):
    def test_spinodal_hessian_is_cheap_and_detects_regular_solution(self):
        stable = _RegularSolutionThermodynamics(1.0)
        result = stable.liquid_spinodal_stability(300.0, {'a': 0.5, 'b': 0.5})
        self.assertTrue(result['locally_stable'])
        self.assertGreater(result['minimum_eigenvalue'], 0.0)
        self.assertEqual(stable.activity_calls, 2)

        unstable = _RegularSolutionThermodynamics(3.0)
        result = unstable.liquid_spinodal_stability(300.0, {'a': 0.5, 'b': 0.5})
        self.assertFalse(result['locally_stable'])
        self.assertLess(result['minimum_eigenvalue'], 0.0)
        self.assertEqual(unstable.activity_calls, 2)

    def test_vle_adaptive_and_global_dispatch_have_distinct_contracts(self):
        thermo = _RegularSolutionThermodynamics(3.0)
        z = {'a': 0.5, 'b': 0.5}

        thermo.set_fluid_phase_model('VLE')
        constrained = thermo._fluid_phase_equilibrium_TP(z, 300.0, 1.0)
        self.assertEqual(constrained.liquid2_fraction, 0.0)
        self.assertEqual(constrained.stability, 'vle_constrained')
        self.assertEqual(thermo.flash3_calls, 0)

        thermo.set_fluid_phase_model('VL(L)E')
        adaptive = thermo._fluid_phase_equilibrium_TP(z, 300.0, 1.0)
        self.assertEqual(adaptive.liquid2_fraction, 0.5)
        self.assertEqual(adaptive.stability, 'spinodal_unstable_vlle_selected')
        self.assertEqual(thermo.flash3_calls, 1)

        thermo.set_fluid_phase_model('VLLE')
        global_result = thermo._fluid_phase_equilibrium_TP(z, 300.0, 1.0)
        self.assertEqual(global_result.liquid2_fraction, 0.5)
        self.assertEqual(global_result.stability, 'global_vlle_search')
        self.assertEqual(thermo.flash3_calls, 2)

    def test_vlle_aware_modes_require_activity_thermodynamics(self):
        thermo = create_thermodynamics(['water'], 'IDEAL')
        thermo.set_fluid_phase_model('VLE')
        with self.assertRaisesRegex(Exception, 'requires an LLE-capable'):
            thermo.set_fluid_phase_model('VLLE')

    def test_rigorous_distillation_inherits_or_overrides_global_stage_policy(self):
        thermo = _RegularSolutionThermodynamics(1.0)
        for mode in ('VLE', 'VL(L)E', 'VLLE'):
            with self.subTest(mode=mode):
                thermo.set_fluid_phase_model(mode)
                unit = RigorousDistillation('D', thermo, {})
                self.assertEqual(unit._stage_phase_model(), mode)

        thermo.set_fluid_phase_model('VLLE')
        constrained = RigorousDistillation(
            'D-VLE', thermo, {'stage_phase_model': 'VLE'}
        )
        self.assertEqual(constrained._stage_phase_model(), 'VLE')

    def test_adaptive_rigorous_distillation_checks_every_vle_stage(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIFAC')
        thermo.set_fluid_phase_model('VL(L)E')
        feed = thermo.calculate_state(
            298.15,
            1.0,
            100.0,
            {'methanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
        )
        result = RigorousDistillation(
            'D-ADAPTIVE',
            thermo,
            {
                'N_stages': 4,
                'feed_stage': 2,
                'reflux_ratio': 2.0,
                'D_to_F': 0.4,
                'P_condenser': 1.0,
                'P_drop_per_stage': 0.0,
            },
        ).solve({'feed': feed})
        self.assertEqual(result.performance['stage_phase_model'], 'VL(L)E')
        eigenvalues = result.performance['stage_spinodal_minimum_eigenvalues']
        self.assertEqual(len(eigenvalues), 4)
        self.assertTrue(all(value >= -1.0e-6 for value in eigenvalues))

    def test_henry_standard_state_fails_in_lle_aware_global_modes(self):
        thermo = create_thermodynamics(['water', 'nitrogen'], 'NRTL')
        thermo.set_fluid_phase_model('VL(L)E')
        with self.assertRaisesRegex(
            Exception,
            'Henry-aware liquid-liquid stability is not implemented',
        ):
            thermo.create_aqueous_equilibrium_context(
                ['nitrogen'],
                water_component='water',
            )

    def test_real_nrtl_global_state_retains_three_phases_and_bulk_balance(self):
        components = ['water', 'methanol', 'benzene']
        z = {'water': 0.20, 'methanol': 0.30, 'benzene': 0.50}
        thermo = create_thermodynamics(components, 'NRTL')

        thermo.set_fluid_phase_model('VLE')
        constrained = thermo.calculate_state(333.0, 1.01325, 100.0, z, include=())
        reference_V, reference_x, reference_y = thermo.flash_TP(
            z, 333.0, 1.01325
        )
        self.assertAlmostEqual(constrained.vapor_fraction, reference_V, places=12)
        for component in components:
            self.assertAlmostEqual(
                constrained.x[component], reference_x[component], places=12
            )
            self.assertAlmostEqual(
                constrained.y[component], reference_y[component], places=12
            )
        self.assertEqual(constrained.liquid2_fraction, 0.0)
        self.assertEqual(constrained.phase_stability, 'vle_constrained')

        thermo.set_fluid_phase_model('VL(L)E')
        local = thermo.calculate_state(333.0, 1.01325, 100.0, z, include=())
        self.assertEqual(local.liquid2_fraction, 0.0)
        self.assertEqual(local.phase_stability, 'locally_stable_spinodal_only')

        thermo.set_fluid_phase_model('VLLE')
        state = thermo.calculate_state(
            333.0,
            1.01325,
            100.0,
            z,
            include=('H', 'S', 'Cp', 'rho'),
        )
        fractions = state.phase_fractions()
        self.assertGreater(fractions['vapor'], 0.0)
        self.assertGreater(fractions['liquid1'], 0.0)
        self.assertGreater(fractions['liquid2'], 0.0)
        self.assertAlmostEqual(sum(fractions.values()), 1.0, places=10)
        for component in components:
            reconstructed = (
                fractions['vapor'] * state.y[component]
                + fractions['liquid1'] * state.x1[component]
                + fractions['liquid2'] * state.x2[component]
            )
            self.assertAlmostEqual(reconstructed, z[component], delta=2e-6)
            pooled = (
                fractions['liquid1'] * state.x1[component]
                + fractions['liquid2'] * state.x2[component]
            ) / (fractions['liquid1'] + fractions['liquid2'])
            self.assertAlmostEqual(state.x[component], pooled, places=10)
        self.assertTrue(math.isfinite(state.H))
        self.assertTrue(math.isfinite(state.S))
        self.assertGreater(state.Cp, 0.0)
        self.assertGreater(state.rho, 0.0)

        recovered = thermo.calculate_state_PQ(
            1.01325,
            state.vapor_fraction,
            100.0,
            z,
            include=(),
        )
        self.assertAlmostEqual(recovered.T, 333.0, delta=2.0e-4)
        self.assertGreater(recovered.liquid2_fraction, 0.0)
        self.assertEqual(recovered.phase_stability, 'global_vlle_search')

        heated = Heater(
            'H-VLLE',
            thermo,
            {'T_out': 335.0},
        ).solve({'in': state}).outlet_streams['out']
        direct_heated = thermo.calculate_state(
            335.0,
            state.P,
            state.F,
            state.composition,
            include=(),
        )
        self.assertEqual(heated.phase_stability, 'global_vlle_search')
        self.assertAlmostEqual(
            heated.vapor_fraction,
            direct_heated.vapor_fraction,
            places=10,
        )
        self.assertAlmostEqual(
            heated.liquid2_fraction,
            direct_heated.liquid2_fraction,
            places=10,
        )

        flash = Flash('F-VLLE', thermo, {'T': 333.0, 'P': 1.01325})
        separated = flash.solve({'in': state})
        liquid_out = separated.outlet_streams['liquid_out']
        self.assertEqual(liquid_out.vapor_fraction, 0.0)
        self.assertGreater(liquid_out.effective_liquid1_fraction, 0.0)
        self.assertGreater(liquid_out.liquid2_fraction, 0.0)
        self.assertAlmostEqual(
            liquid_out.effective_liquid1_fraction + liquid_out.liquid2_fraction,
            1.0,
            places=10,
        )
        self.assertEqual(liquid_out.x1, state.x1)
        self.assertEqual(liquid_out.x2, state.x2)
        vapor_out = separated.outlet_streams['vapor_out']
        outlet_enthalpy_flow = (
            vapor_out.F * vapor_out.H + liquid_out.F * liquid_out.H
        )
        self.assertAlmostEqual(
            outlet_enthalpy_flow,
            state.F * state.H,
            delta=max(1.0e-5, abs(state.F * state.H) * 1.0e-9),
        )

        explicitly_liquid = thermo.calculate_state(
            333.0, 1.01325, 100.0, z, phase='liquid', flash=False, include=()
        )
        self.assertEqual(explicitly_liquid.liquid2_fraction, 0.0)
        self.assertEqual(
            explicitly_liquid.phase_stability,
            'explicit_phase_constraint',
        )


if __name__ == '__main__':
    unittest.main()
