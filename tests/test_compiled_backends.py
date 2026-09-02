import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from thermodynamics import ActivityCoefficientThermodynamics, create_thermodynamics
from simulator import Simulator


class CompiledBackendTests(unittest.TestCase):
    def test_simulator_initialization_compiles_only_reachable_backends(self):
        ordinary = Simulator.from_file(
            os.path.join(ROOT, 'examples', 'unifac_flash.pfd')
        )
        ordinary.initialize()
        self.assertTrue(
            ordinary.thermo._compiled_unifac.compilation_complete
        )
        self.assertFalse(
            ordinary.thermo._compiled_lle.compilation_complete
        )
        self.assertFalse(ordinary.thermo._compiled_vlle_initialized)
        ordinary.run()
        self.assertFalse(
            ordinary.thermo._compiled_lle.compilation_complete
        )
        self.assertFalse(ordinary.thermo._compiled_vlle_initialized)

        lle = Simulator.from_file(
            os.path.join(ROOT, 'examples', 'butanol_water_lle.pfd')
        )
        lle.initialize()
        self.assertTrue(lle.thermo._compiled_lle.compilation_complete)
        self.assertFalse(lle.thermo._compiled_vlle_initialized)

        vlle = Simulator.from_file(
            os.path.join(ROOT, 'examples', 'global_vlle_water_methanol_benzene.pfd')
        )
        vlle.initialize()
        self.assertTrue(
            vlle.thermo._compiled_lle_backend(298.15).compilation_complete
        )
        self.assertTrue(vlle.thermo._compiled_vlle_initialized)
        self.assertTrue(vlle.thermo._compiled_vlle.compilation_complete)

    def test_compiled_eos_signatures_do_not_change_during_run(self):
        from compiled_cubic_eos import (
            _cubic_departure_enthalpy_numba,
            _cubic_departure_entropy_numba,
            _cubic_fugacity_numba,
            _cubic_mixture_numba,
            _cubic_phi_phi_k_numba,
            _cubic_roots_state_numba,
        )

        simulator = Simulator.from_file(
            os.path.join(ROOT, 'examples', 'air_3a_molecular_sieve_drying.pfd')
        )
        simulator.initialize()
        backend = simulator.thermo.cubic
        self.assertTrue(backend._compiled_backend.compilation_complete)
        dispatchers = (
            _cubic_mixture_numba,
            _cubic_roots_state_numba,
            _cubic_fugacity_numba,
            _cubic_departure_enthalpy_numba,
            _cubic_departure_entropy_numba,
            _cubic_phi_phi_k_numba,
        )
        before = tuple(tuple(dispatcher.signatures) for dispatcher in dispatchers)
        self.assertTrue(all(before))

        result = simulator.run()

        after = tuple(tuple(dispatcher.signatures) for dispatcher in dispatchers)
        self.assertTrue(result.converged, result.errors)
        self.assertEqual(after, before)

    def test_compiled_constrained_vle_candidate_matches_reference_at_endpoints(self):
        cases = (
            (
                'NRTL',
                ['water', 'methanol', 'benzene'],
                {'water': 0.20, 'methanol': 0.30, 'benzene': 0.50},
                333.0,
            ),
            (
                'UNIQUAC',
                ['water', 'ethanol', 'benzene'],
                {'water': 0.30, 'ethanol': 0.10, 'benzene': 0.60},
                339.0,
            ),
            (
                'UNIFNIST',
                ['water', 'ethanol', 'cyclohexane'],
                {'water': 0.30, 'ethanol': 0.20, 'cyclohexane': 0.50},
                337.0,
            ),
        )
        for method, components, composition, temperature in cases:
            with self.subTest(method=method):
                thermo = create_thermodynamics(components, method)
                backend = thermo.compiled_vlle_backend()
                if backend is None:
                    self.skipTest(f"Compiled {method} VLLE backend is unavailable")
                reference_V, reference_x, reference_y = thermo.flash_TP(
                    composition,
                    temperature,
                    1.01325,
                )
                compiled_V, compiled_x, compiled_y = backend.flash_VLE_TP(
                    composition,
                    temperature,
                    1.01325,
                    max_iter=200,
                )
                self.assertAlmostEqual(compiled_V, reference_V, delta=2.0e-6)
                for index, component in enumerate(components):
                    self.assertAlmostEqual(
                        compiled_x[index], reference_x[component], delta=2.0e-6
                    )
                    self.assertAlmostEqual(
                        compiled_y[index], reference_y[component], delta=2.0e-6
                    )

    def test_compiled_unifac_preserves_infinite_dilution_activity(self):
        thermo = create_thermodynamics(['water', 'benzene'], 'UNIFNIST')
        if thermo._compiled_unifac is None:
            self.skipTest("Compiled UNIFNIST backend is unavailable")

        gamma_above = thermo.activity_coefficients(
            298.15,
            {'water': 1.0 - 1e-8, 'benzene': 1e-8},
        )['benzene']
        gamma_below = thermo.activity_coefficients(
            298.15,
            {'water': 1.0 - 1e-12, 'benzene': 1e-12},
        )['benzene']
        gamma_zero = thermo.activity_coefficients(
            298.15,
            {'water': 1.0, 'benzene': 0.0},
        )['benzene']

        compiled_backend = thermo._compiled_unifac
        thermo._compiled_unifac = None
        thermo._activity_cache.clear()
        gamma_reference = thermo.activity_coefficients(
            298.15,
            {'water': 1.0, 'benzene': 0.0},
        )['benzene']
        thermo._compiled_unifac = compiled_backend

        self.assertGreater(gamma_zero, 10.0)
        self.assertAlmostEqual(gamma_below, gamma_zero, delta=gamma_zero * 1e-8)
        self.assertAlmostEqual(gamma_above, gamma_zero, delta=gamma_zero * 1e-4)
        self.assertAlmostEqual(gamma_reference, gamma_zero, delta=gamma_zero * 1e-10)

    def test_compiled_unifac_matches_reference_backend(self):
        components = ['diethyl ether', 'n-hexane', 'acrylic acid', 'water']
        composition = {
            'diethyl ether': 0.45,
            'n-hexane': 0.45,
            'acrylic acid': 0.08,
            'water': 0.02,
        }
        x = [composition[comp] for comp in components]

        for method in ('UNIFAC', 'UNIFAC2', 'UNIFDMD', 'UNIFM2', 'UNIFNIST'):
            with self.subTest(method=method):
                thermo = create_thermodynamics(components, method)
                if thermo._compiled_unifac is None:
                    self.skipTest(f"Compiled {method} backend is unavailable")
                groups = [thermo.component_groups[comp] for comp in components]

                reference = thermo.unifac.activity_coefficients(groups, x, 298.15)
                compiled = thermo._compiled_unifac.activity_coefficients(x, 298.15)

                for ref_value, compiled_value in zip(reference, compiled):
                    self.assertAlmostEqual(
                        ref_value,
                        compiled_value,
                        delta=abs(ref_value) * 1e-10,
                    )

    def test_compiled_lle_matches_reference_splitter(self):
        components = ['diethyl ether', 'n-hexane', 'acrylic acid', 'water']
        # Use the component order generated inside the rigorous extractor:
        # aqueous feed components first, then solvent components.
        composition = {
            'acrylic acid': 0.024974946990,
            'water': 0.748981289912,
            'diethyl ether': 0.121392237235,
            'n-hexane': 0.104651525863,
        }

        for method in ('UNIFAC', 'UNIFDMD', 'UNIFNIST'):
            with self.subTest(method=method):
                thermo = create_thermodynamics(components, method)
                if thermo._compiled_lle is None:
                    self.skipTest(f"Compiled {method} LLE backend is unavailable")

                reference = ActivityCoefficientThermodynamics.liquid_liquid_equilibrium(
                    thermo, composition, 298.15, max_iter=200, tol=1e-5
                )
                compiled = thermo.liquid_liquid_equilibrium(
                    composition, 298.15, max_iter=200, tol=1e-5
                )

                self.assertEqual(reference[0], compiled[0])
                self.assertAlmostEqual(reference[3], compiled[3], delta=2e-8)
                for phase_index in (1, 2):
                    for comp in composition:
                        self.assertAlmostEqual(
                            reference[phase_index][comp],
                            compiled[phase_index][comp],
                            delta=2e-8,
                        )

    def test_compiled_unifac_binary_lle_matches_reference_splitter(self):
        cases = [
            (
                ['water', 'ethyl acetate'],
                {'water': 0.5, 'ethyl acetate': 0.5},
                5e-6,
            ),
            (
                ['water', 'toluene'],
                {'water': 0.5, 'toluene': 0.5},
                5e-8,
            ),
            (
                ['methanol', 'heptane'],
                {'methanol': 0.5, 'heptane': 0.5},
                5e-6,
            ),
            (
                ['hexane', 'water'],
                {'hexane': 0.001, 'water': 0.999},
                5e-6,
            ),
            (
                ['hexane', 'water'],
                {'hexane': 0.999, 'water': 0.001},
                5e-6,
            ),
        ]

        for components, composition, tolerance in cases:
            with self.subTest(components=components):
                thermo = create_thermodynamics(components, 'UNIFAC')
                if thermo._compiled_lle is None:
                    self.skipTest("Compiled UNIFAC LLE backend is unavailable")

                reference = ActivityCoefficientThermodynamics.liquid_liquid_equilibrium(
                    thermo, composition, 298.15, max_iter=200, tol=1e-6
                )
                compiled = thermo._compiled_lle.split(
                    composition, 298.15, max_iter=200, tol=1e-6
                )

                self.assertEqual(reference[0], compiled[0])
                self.assertAlmostEqual(reference[3], compiled[3], delta=tolerance)
                for phase_index in (1, 2):
                    for comp in composition:
                        self.assertAlmostEqual(
                            reference[phase_index][comp],
                            compiled[phase_index][comp],
                            delta=tolerance,
                        )

    def test_compiled_unifac_binary_lle_rejects_endpoint_split(self):
        thermo = create_thermodynamics(['water', 'butanol'], 'UNIFAC')
        if thermo._compiled_lle is None:
            self.skipTest("Compiled UNIFAC LLE backend is unavailable")

        compiled = thermo._compiled_lle.split(
            {'water': 0.5, 'butanol': 0.5}, 298.15, max_iter=200, tol=1e-6
        )

        self.assertFalse(compiled[0])

    def test_compiled_nrtl_and_uniquac_match_reference_activity(self):
        components = ['H2O', 'CH3OH', 'methyl acetate', '(C2H5)2O']
        composition = {
            'H2O': 0.10,
            'CH3OH': 0.01,
            'methyl acetate': 0.44,
            '(C2H5)2O': 0.45,
        }

        for method in ('NRTL', 'UNIQUAC'):
            with self.subTest(method=method):
                reference_thermo = create_thermodynamics(components, method)
                reference_thermo._compiled_activity_backend = lambda _T: None

                thermo = create_thermodynamics(components, method)
                compiled = thermo._compiled_activity_backend(298.15)
                if compiled is None:
                    self.skipTest(f"Compiled {method} activity backend is unavailable")

                x = [composition[comp] for comp in components]
                for T in (298.15, 350.0):
                    reference = reference_thermo.activity_coefficients(
                        T,
                        composition,
                    )
                    compiled_values = compiled.activity_coefficients(x, T)

                    for index, comp in enumerate(components):
                        self.assertAlmostEqual(
                            reference[comp],
                            compiled_values[index],
                            delta=abs(reference[comp]) * 1e-10,
                        )

    def test_compiled_nrtl_and_uniquac_preserve_infinite_dilution_activity(self):
        for method in ('NRTL', 'UNIQUAC'):
            with self.subTest(method=method):
                thermo = create_thermodynamics(['water', 'benzene'], method)
                compiled = thermo._compiled_activity_backend(298.15)
                if compiled is None:
                    self.skipTest(f"Compiled {method} activity backend is unavailable")

                gamma_above = thermo.activity_coefficients(
                    298.15,
                    {'water': 1.0 - 1e-8, 'benzene': 1e-8},
                )['benzene']
                gamma_below = thermo.activity_coefficients(
                    298.15,
                    {'water': 1.0 - 1e-12, 'benzene': 1e-12},
                )['benzene']
                gamma_zero = thermo.activity_coefficients(
                    298.15,
                    {'water': 1.0, 'benzene': 0.0},
                )['benzene']

                reference_thermo = create_thermodynamics(['water', 'benzene'], method)
                reference_thermo._compiled_activity_backend = lambda _T: None
                gamma_reference = reference_thermo.activity_coefficients(
                    298.15,
                    {'water': 1.0, 'benzene': 0.0},
                )['benzene']

                self.assertGreater(gamma_zero, 10.0)
                self.assertAlmostEqual(gamma_below, gamma_zero, delta=gamma_zero * 1e-8)
                self.assertAlmostEqual(gamma_above, gamma_zero, delta=gamma_zero * 1e-4)
                self.assertAlmostEqual(gamma_reference, gamma_zero, delta=gamma_zero * 1e-10)

    def test_compiled_nrtl_and_uniquac_lle_matches_reference_splitter(self):
        components = ['H2O', 'CH3OH', 'methyl acetate', '(C2H5)2O']
        composition = {
            'H2O': 0.45,
            'CH3OH': 0.01,
            'methyl acetate': 0.25,
            '(C2H5)2O': 0.29,
        }

        for method in ('NRTL', 'UNIQUAC'):
            with self.subTest(method=method):
                reference_thermo = create_thermodynamics(components, method)
                reference_thermo._compiled_activity_backend = lambda _T: None
                reference_thermo._compiled_lle_backend = lambda _T: None
                reference = ActivityCoefficientThermodynamics.liquid_liquid_equilibrium(
                    reference_thermo, composition, 298.15, max_iter=200, tol=1e-5
                )

                thermo = create_thermodynamics(components, method)
                if thermo._compiled_lle_backend(298.15) is None:
                    self.skipTest(f"Compiled {method} LLE backend is unavailable")

                compiled = thermo.liquid_liquid_equilibrium(
                    composition, 298.15, max_iter=200, tol=1e-5
                )

                self.assertEqual(reference[0], compiled[0])
                self.assertAlmostEqual(reference[3], compiled[3], delta=5e-7)
                for phase_index in (1, 2):
                    for comp in composition:
                        self.assertAlmostEqual(
                            reference[phase_index][comp],
                            compiled[phase_index][comp],
                            delta=5e-7,
                        )

    def test_compiled_uniquac_lle_accepts_component_subset(self):
        components = ['H2O', 'CH3OH', 'methyl acetate', '(C2H5)2O']
        composition = {
            'H2O': 0.46,
            'methyl acetate': 0.25,
            '(C2H5)2O': 0.29,
        }
        thermo = create_thermodynamics(components, 'UNIQUAC')
        backend = thermo._compiled_lle_backend(298.15)
        if backend is None:
            self.skipTest("Compiled UNIQUAC LLE backend is unavailable")

        reference = ActivityCoefficientThermodynamics.liquid_liquid_equilibrium(
            thermo,
            composition,
            298.15,
            max_iter=200,
            tol=1e-5,
        )
        compiled = backend.split(
            composition,
            298.15,
            max_iter=200,
            tol=1e-5,
        )

        self.assertIsNotNone(compiled)
        self.assertEqual(reference[0], compiled[0])
        self.assertAlmostEqual(reference[3], compiled[3], delta=5e-7)
        for phase_index in (1, 2):
            for comp in composition:
                self.assertAlmostEqual(
                    reference[phase_index][comp],
                    compiled[phase_index][comp],
                    delta=5e-7,
                )

    def test_nrtl_and_uniquac_binary_lle_use_compiled_backend(self):
        compiled_split = (
            True,
            {'water': 0.9, 'acetonitrile': 0.1},
            {'water': 0.2, 'acetonitrile': 0.8},
            0.4,
        )

        for method in ('NRTL', 'UNIQUAC'):
            with self.subTest(method=method):
                thermo = create_thermodynamics(['water', 'acetonitrile'], method)
                calls = []

                def split(composition, T, max_iter=100, tol=1e-6):
                    calls.append((composition, T, max_iter, tol))
                    return compiled_split

                thermo._compiled_lle_backend = lambda _T: SimpleNamespace(split=split)

                result = thermo.liquid_liquid_equilibrium(
                    {'water': 0.5, 'acetonitrile': 0.5},
                    298.15,
                    max_iter=123,
                    tol=1e-7,
                )

                self.assertEqual(result, compiled_split)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][2], 123)
                self.assertEqual(calls[0][3], 1e-7)

    def test_compiled_unifac_vlle_matches_reference_ternary_tp(self):
        components = ['water', 'ethanol', 'cyclohexane']
        z = {'water': 0.30, 'ethanol': 0.20, 'cyclohexane': 0.50}
        thermo = create_thermodynamics(
            components,
            'UNIFNIST',
            activity_interaction_max_temperature_K=320.0,
        )
        backend = thermo.compiled_vlle_backend()
        if backend is None:
            self.skipTest("Compiled UNIFNIST VLLE backend is unavailable")

        reference = thermo.flash3_TP(z, 337.0, 1.01325, max_iter=200)
        compiled = backend.flash_TP(z, 337.0, 1.01325, max_iter=200)

        self.assertEqual(compiled.status, reference.status)
        self.assertEqual(compiled.phase_count, reference.phase_count)
        self.assertLess(compiled.iterations, 25)
        self.assertAlmostEqual(compiled.vapor_fraction, reference.vapor_fraction, delta=2e-3)
        self.assertAlmostEqual(compiled.liquid1_fraction, reference.liquid1_fraction, delta=2e-3)
        self.assertAlmostEqual(compiled.liquid2_fraction, reference.liquid2_fraction, delta=2e-3)
        for index, comp in enumerate(components):
            self.assertAlmostEqual(compiled.y[index], reference.y[comp], delta=1e-3)
            self.assertAlmostEqual(compiled.x1[index], reference.x1[comp], delta=1e-3)
            self.assertAlmostEqual(compiled.x2[index], reference.x2[comp], delta=1e-3)

    def test_compiled_unifac_vlle_matches_reference_binary_invariant(self):
        components = ['water', 'chloroform']
        z = {'water': 0.5, 'chloroform': 0.5}
        T = 329.1264566618235
        P = 1.01325
        thermo = create_thermodynamics(components, 'UNIFNIST')
        backend = thermo.compiled_vlle_backend()
        if backend is None:
            self.skipTest("Compiled UNIFNIST VLLE backend is unavailable")

        reference = thermo.flash3_TP(z, T, P, max_iter=200)
        compiled = backend.flash_TP(z, T, P, max_iter=200)

        self.assertEqual(reference.status, 'binary_invariant_vlle')
        self.assertEqual(compiled.status, 'binary_invariant_vlle')
        self.assertEqual(compiled.phase_count, 3)
        self.assertAlmostEqual(compiled.vapor_fraction, reference.vapor_fraction, delta=2e-5)
        for index, comp in enumerate(components):
            self.assertAlmostEqual(compiled.x1[index], reference.x1[comp], delta=1e-7)
            self.assertAlmostEqual(compiled.x2[index], reference.x2[comp], delta=1e-7)
            self.assertAlmostEqual(compiled.y[index], reference.y[comp], delta=2e-5)

    def test_compiled_unifac_vlle_pv_uses_compiled_tp(self):
        components = ['water', 'ethanol', 'cyclohexane']
        z = {'water': 0.30, 'ethanol': 0.20, 'cyclohexane': 0.50}
        thermo = create_thermodynamics(components, 'UNIFNIST')
        backend = thermo.compiled_vlle_backend()
        if backend is None:
            self.skipTest("Compiled UNIFNIST VLLE backend is unavailable")

        reference = thermo.flash3_TP(z, 337.0, 1.01325, max_iter=200)
        compiled_reference = backend.flash_TP(z, 337.0, 1.01325, max_iter=200)
        T, compiled = backend.flash_PV(
            z,
            1.01325,
            compiled_reference.vapor_fraction,
            T_guess=342.0,
            max_iter=200,
        )
        reference_bubble = thermo.bubble_point_T_vlle(z, 1.01325, T_guess=337.0, max_iter=200)
        bubble_T, bubble_result = backend.flash_PV(
            z,
            1.01325,
            0.0,
            T_guess=337.0,
            max_iter=200,
        )
        reference_dew = thermo.dew_point_T_vlle(reference.y, 1.01325, T_guess=337.0, max_iter=200)
        dew_T, dew_result = backend.flash_PV(
            reference.y,
            1.01325,
            1.0,
            T_guess=337.0,
            max_iter=200,
        )

        self.assertFalse(hasattr(backend, 'bubble_point_T_vlle'))
        self.assertFalse(hasattr(backend, 'dew_point_T_vlle'))
        self.assertEqual(compiled.status, 'structured_vlle_feed_lle_seed')
        self.assertAlmostEqual(T, 337.0, delta=0.02)
        self.assertAlmostEqual(compiled.vapor_fraction, compiled_reference.vapor_fraction, delta=2e-5)
        self.assertAlmostEqual(bubble_T, reference_bubble, delta=0.02)
        self.assertEqual(
            bubble_result.status,
            backend.flash_TP(z, bubble_T, 1.01325, max_iter=200).status,
        )
        self.assertAlmostEqual(dew_T, reference_dew, delta=0.02)
        self.assertEqual(
            dew_result.status,
            backend.flash_TP(reference.y, dew_T, 1.01325, max_iter=200).status,
        )

    def test_vlle_tv_uses_compiled_tp_for_ternary_activity_models(self):
        components = ['water', 'ethanol', 'cyclohexane']
        z = {'water': 0.30, 'ethanol': 0.20, 'cyclohexane': 0.50}
        T = 337.0
        P = 1.01325
        thermo = create_thermodynamics(components, 'UNIFNIST')
        backend = thermo.compiled_vlle_backend()
        if backend is None:
            self.skipTest("Compiled UNIFNIST VLLE backend is unavailable")

        reference = thermo.flash3_TP(z, T, P, max_iter=200)
        calls = []
        original = backend.flash_TP

        def counted_flash_TP(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        backend.flash_TP = counted_flash_TP

        P_tv, tv = thermo.flash3_TV(
            z,
            T,
            reference.vapor_fraction,
            P_guess=1.2 * P,
            max_iter=200,
        )

        self.assertGreater(len(calls), 0)
        self.assertAlmostEqual(P_tv, P, delta=5e-5)
        self.assertEqual(tv.phase_count, 3)
        self.assertAlmostEqual(tv.vapor_fraction, reference.vapor_fraction, delta=2e-3)

    def test_vlle_tv_skips_compiled_tp_for_binary_invariant_case(self):
        components = ['water', 'chloroform']
        z = {'water': 0.5, 'chloroform': 0.5}
        T = 329.1264566618235
        P = 1.01325
        thermo = create_thermodynamics(components, 'UNIFNIST')
        backend = thermo.compiled_vlle_backend()
        if backend is None:
            self.skipTest("Compiled UNIFNIST VLLE backend is unavailable")

        reference = thermo.flash3_TP(z, T, P, max_iter=200)
        calls = []
        original = backend.flash_TP

        def counted_flash_TP(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        backend.flash_TP = counted_flash_TP

        P_tv, tv = thermo.flash3_TV(
            z,
            T,
            reference.vapor_fraction,
            P_guess=1.2 * P,
            max_iter=200,
        )

        self.assertEqual(calls, [])
        self.assertAlmostEqual(P_tv, P, delta=5e-5)
        self.assertEqual(tv.phase_count, 3)

    def test_vlle_ph_uses_compiled_tp_for_ternary_vlle_branch_probe(self):
        components = ['water', 'ethanol', 'cyclohexane']
        z = {'water': 0.30, 'ethanol': 0.20, 'cyclohexane': 0.50}
        T = 337.0
        P = 1.01325
        thermo = create_thermodynamics(components, 'UNIFNIST')
        backend = thermo.compiled_vlle_backend()
        if backend is None:
            self.skipTest("Compiled UNIFNIST VLLE backend is unavailable")

        reference = thermo.flash3_TP(z, T, P, max_iter=200)
        H_reference = thermo._flash3_mixture_enthalpy(T, P, reference)
        calls = []
        original = backend.flash_TP

        def counted_flash_TP(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        backend.flash_TP = counted_flash_TP

        T_ph, ph, residual = thermo.flash3_PH(
            z,
            P,
            H_reference,
            T_guess=T + 8.0,
            max_iter=200,
        )

        self.assertGreater(len(calls), 0)
        self.assertAlmostEqual(T_ph, T, delta=1e-3)
        self.assertEqual(ph.phase_count, 3)
        self.assertLess(abs(residual), 20.0)

    def test_vlle_ph_skips_compiled_tp_for_non_vlle_fallback(self):
        components = ['water', 'ethanol', 'cyclohexane']
        z = {'water': 0.20, 'ethanol': 0.60, 'cyclohexane': 0.20}
        T = 337.0
        P = 1.01325
        thermo = create_thermodynamics(components, 'UNIFNIST')
        backend = thermo.compiled_vlle_backend()
        if backend is None:
            self.skipTest("Compiled UNIFNIST VLLE backend is unavailable")

        reference = thermo.flash3_TP(z, T, P, max_iter=200)
        H_reference = thermo._flash3_mixture_enthalpy(T, P, reference)
        calls = []
        original = backend.flash_TP

        def counted_flash_TP(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        backend.flash_TP = counted_flash_TP

        T_ph, ph, residual = thermo.flash3_PH(
            z,
            P,
            H_reference,
            T_guess=T + 8.0,
            max_iter=200,
        )

        self.assertEqual(calls, [])
        self.assertAlmostEqual(T_ph, T, delta=1e-6)
        self.assertEqual(ph.phase_count, 2)
        self.assertLess(abs(residual), 1e-5)

    def test_vlle_ps_uses_compiled_tp_for_ternary_vlle_branch_probe(self):
        components = ['water', 'ethanol', 'cyclohexane']
        z = {'water': 0.30, 'ethanol': 0.20, 'cyclohexane': 0.50}
        T = 337.0
        P = 1.01325
        thermo = create_thermodynamics(components, 'UNIFNIST')
        backend = thermo.compiled_vlle_backend()
        if backend is None:
            self.skipTest("Compiled UNIFNIST VLLE backend is unavailable")

        reference = thermo.flash3_TP(z, T, P, max_iter=200)
        S_reference = thermo._flash3_mixture_entropy(T, P, reference)
        calls = []
        original = backend.flash_TP

        def counted_flash_TP(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        backend.flash_TP = counted_flash_TP

        T_ps, ps, residual = thermo.flash3_PS(
            z,
            P,
            S_reference,
            T_guess=T + 40.0,
            max_iter=200,
        )

        self.assertGreater(len(calls), 0)
        self.assertAlmostEqual(T_ps, T, delta=1e-3)
        self.assertEqual(ps.phase_count, 3)
        self.assertLess(abs(residual), 0.1)

    def test_vlle_ps_skips_compiled_tp_for_non_vlle_fallback(self):
        components = ['water', 'ethanol', 'cyclohexane']
        z = {'water': 0.20, 'ethanol': 0.60, 'cyclohexane': 0.20}
        T = 337.0
        P = 1.01325
        thermo = create_thermodynamics(components, 'UNIFNIST')
        backend = thermo.compiled_vlle_backend()
        if backend is None:
            self.skipTest("Compiled UNIFNIST VLLE backend is unavailable")

        reference = thermo.flash3_TP(z, T, P, max_iter=200)
        S_reference = thermo._flash3_mixture_entropy(T, P, reference)
        calls = []
        original = backend.flash_TP

        def counted_flash_TP(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        backend.flash_TP = counted_flash_TP

        T_ps, ps, residual = thermo.flash3_PS(
            z,
            P,
            S_reference,
            T_guess=T + 40.0,
            max_iter=200,
        )

        self.assertEqual(calls, [])
        self.assertAlmostEqual(T_ps, T, delta=1e-6)
        self.assertEqual(ps.phase_count, 2)
        self.assertLess(abs(residual), 1e-6)

    def test_vlle_ps_skips_compiled_tp_for_binary_invariant_case(self):
        components = ['water', 'chloroform']
        z = {'water': 0.5, 'chloroform': 0.5}
        T = 329.1264566618235
        P = 1.01325
        thermo = create_thermodynamics(components, 'UNIFNIST')
        backend = thermo.compiled_vlle_backend()
        if backend is None:
            self.skipTest("Compiled UNIFNIST VLLE backend is unavailable")

        _T_pv, reference = thermo.flash3_PV(
            z,
            P,
            0.2,
            T_guess=T,
            max_iter=200,
        )
        S_reference = thermo._flash3_mixture_entropy(T, P, reference)
        calls = []
        original = backend.flash_TP

        def counted_flash_TP(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        backend.flash_TP = counted_flash_TP

        T_ps, ps, residual = thermo.flash3_PS(
            z,
            P,
            S_reference,
            T_guess=T + 40.0,
            max_iter=200,
        )

        self.assertEqual(calls, [])
        self.assertAlmostEqual(T_ps, T, delta=1e-6)
        self.assertEqual(ps.status, 'binary_invariant_vlle')
        self.assertAlmostEqual(ps.vapor_fraction, 0.2, delta=1e-8)
        self.assertLess(abs(residual), 1e-7)

    def test_compiled_nrtl_and_uniquac_vlle_match_reference_tp(self):
        cases = [
            (
                'NRTL',
                ['water', 'methanol', 'benzene'],
                {'water': 0.20, 'methanol': 0.30, 'benzene': 0.50},
                333.0,
            ),
            (
                'UNIQUAC',
                ['water', 'ethanol', 'benzene'],
                {'water': 0.30, 'ethanol': 0.10, 'benzene': 0.60},
                339.0,
            ),
        ]

        for method, components, z, T in cases:
            with self.subTest(method=method):
                thermo = create_thermodynamics(
                    components,
                    method,
                    activity_interaction_max_temperature_K=T - 15.0,
                )
                backend = thermo.compiled_vlle_backend()
                if backend is None:
                    self.skipTest(f"Compiled {method} VLLE backend is unavailable")

                reference = thermo.flash3_TP(z, T, 1.01325, max_iter=200)
                compiled = backend.flash_TP(z, T, 1.01325, max_iter=200)

                self.assertEqual(compiled.status, reference.status)
                self.assertEqual(compiled.phase_count, reference.phase_count)
                self.assertLess(compiled.iterations, 25)
                self.assertAlmostEqual(compiled.vapor_fraction, reference.vapor_fraction, delta=2e-3)
                for index, comp in enumerate(components):
                    self.assertAlmostEqual(compiled.y[index], reference.y[comp], delta=1e-3)
                    self.assertAlmostEqual(compiled.x1[index], reference.x1[comp], delta=1e-3)
                    self.assertAlmostEqual(compiled.x2[index], reference.x2[comp], delta=1e-3)

    def test_compiled_vlle_tpd_reseeds_missed_vle_branch(self):
        components = ['water', 'benzene', 'toluene']
        z = {
            'water': 0.5398628992077538,
            'benzene': 0.2490280908487376,
            'toluene': 0.2111090099435087,
        }

        for method in ('UNIFAC', 'NRTL', 'UNIQUAC'):
            with self.subTest(method=method):
                thermo = create_thermodynamics(components, method)
                backend = thermo.compiled_vlle_backend()
                if backend is None:
                    self.skipTest(f"Compiled {method} VLLE backend is unavailable")

                for T in (348.0, 352.0, 354.0, 356.0):
                    with self.subTest(method=method, T=T):
                        reference = thermo.flash3_TP(z, T, 1.0, max_iter=200)
                        compiled = backend.flash_TP(z, T, 1.0, max_iter=200)

                        self.assertEqual(reference.status, 'ordinary_vle')
                        self.assertEqual(compiled.status, reference.status)
                        self.assertEqual(compiled.phase_count, 2)
                        self.assertAlmostEqual(
                            compiled.vapor_fraction,
                            reference.vapor_fraction,
                            delta=1e-4,
                        )
                        for index, component in enumerate(components):
                            self.assertAlmostEqual(
                                compiled.x1[index],
                                reference.x1[component],
                                delta=1e-6,
                            )
                            self.assertAlmostEqual(
                                compiled.y[index],
                                reference.y[component],
                                delta=1e-4,
                            )

                vapor_reference = thermo.flash3_TP(z, 357.0, 1.0, max_iter=200)
                vapor_compiled = backend.flash_TP(z, 357.0, 1.0, max_iter=200)
                self.assertEqual(vapor_reference.status, 'single_vapor')
                self.assertEqual(vapor_compiled.status, 'single_vapor')

                target = thermo.flash3_TP(z, 354.0, 1.0, max_iter=200)
                T_pv, pv = backend.flash_PV(
                    z,
                    1.0,
                    target.vapor_fraction,
                    T_guess=370.0,
                    max_iter=200,
                )
                self.assertAlmostEqual(T_pv, 354.0, delta=5e-4)
                self.assertEqual(pv.status, 'ordinary_vle')
                self.assertAlmostEqual(
                    pv.vapor_fraction,
                    target.vapor_fraction,
                    delta=1e-7,
                )

    def test_gamma_phi_models_do_not_attach_ideal_vapor_compiled_vlle(self):
        for method in (
            'NRTL-RK', 'UNIQUAC-PR', 'UNIFNIST-RK',
            'NRTL-VDM',
            'UNIQUAC-VDM', 'UNIFNIST-VDM',
        ):
            with self.subTest(method=method):
                thermo = create_thermodynamics(
                    ['water', 'methanol', 'benzene'],
                    method,
                )
                self.assertIsNone(thermo.compiled_vlle_backend())






if __name__ == '__main__':
    unittest.main()
