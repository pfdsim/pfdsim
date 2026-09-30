"""Broad regression tests for the thermodynamic-model audit fixes.

Sweep-style by design: the consistency tests iterate SUPPORTED_METHODS, so
any method added to the factory registry is covered automatically without
touching this file.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from thermodynamics_models import ThermodynamicsError, create_thermodynamics
from thermodynamics_models.activity import ActivityCoefficientThermodynamics
from thermodynamics_models.factory import SUPPORTED_METHODS
from thermodynamics_models.nrtl_uniquac import UNIQUACThermodynamics

WATER_ETHANOL = {'water': 0.4, 'ethanol': 0.6}
ACID_WATER = {'CH3COOH': 0.5, 'H2O': 0.5}


def _components_for(method: str) -> tuple[list[str], dict[str, float]]:
    if method == 'STEAM':
        return ['H2O'], {'H2O': 1.0}
    if 'VDM' in method:
        return ['CH3COOH', 'H2O'], dict(ACID_WATER)
    return ['water', 'ethanol'], dict(WATER_ETHANOL)


class _MargulesBinary(ActivityCoefficientThermodynamics):
    """Two-parameter Margules binary with directly controlled gammas."""

    A12 = 0.0
    A21 = 0.0

    def activity_coefficients(self, T, composition):
        c = self.components
        x1 = max(composition.get(c[0], 0.0), 0.0)
        x2 = max(composition.get(c[1], 0.0), 0.0)
        total = x1 + x2
        if total <= 0.0:
            return {k: 1.0 for k in c}
        x1, x2 = x1 / total, x2 / total
        ln1 = x2 ** 2 * (self.A12 + 2.0 * (self.A21 - self.A12) * x1)
        ln2 = x1 ** 2 * (self.A21 + 2.0 * (self.A12 - self.A21) * x2)
        return {c[0]: math.exp(ln1), c[1]: math.exp(ln2)}


def _margules_for_binodal(xa: float, xb: float) -> type[_MargulesBinary]:
    """Fit A12/A21 so the isoactivity binodal is exactly (xa, xb)."""

    def coefficients(x1: float):
        x2 = 1.0 - x1
        return (
            (x2 ** 2 - 2.0 * x2 ** 2 * x1, 2.0 * x2 ** 2 * x1),
            (2.0 * x1 ** 2 * x2, x1 ** 2 - 2.0 * x1 ** 2 * x2),
        )

    (c1a, c2a), (c1b, c2b) = coefficients(xa), coefficients(xb)
    matrix = np.array([
        [c1a[0] - c1b[0], c1a[1] - c1b[1]],
        [c2a[0] - c2b[0], c2a[1] - c2b[1]],
    ])
    rhs = np.array([math.log(xb / xa), math.log((1.0 - xb) / (1.0 - xa))])
    a12, a21 = np.linalg.solve(matrix, rhs)
    return type('FittedMargules', (_MargulesBinary,), {'A12': a12, 'A21': a21})


class MethodSweepTests(unittest.TestCase):
    """Invariants that every supported thermo method must satisfy."""

    _cache: dict[str, object] = {}

    @classmethod
    def _thermo(cls, method: str):
        thermo = cls._cache.get(method)
        if thermo is None:
            comps, _ = _components_for(method)
            thermo = create_thermodynamics(comps, method)
            cls._cache[method] = thermo
        return thermo

    def test_cp_equals_dHdT_for_every_method_and_phase(self):
        """Cp must be the T-derivative of the model's own enthalpy."""
        d = 0.5
        for method in SUPPORTED_METHODS:
            _, z = _components_for(method)
            thermo = self._thermo(method)
            if method == 'STEAM':
                cases = ((450.0, 1.0, 1.0), (350.0, 0.0, 1.0))
            else:
                cases = ((420.0, 1.0, 2.0), (320.0, 0.0, 2.0))
            for T, vf, P in cases:
                phase = 'vapor' if vf > 0.5 else 'liquid'
                with self.subTest(method=method, phase=phase):
                    cp = thermo.mixture_Cp(z, T, vf, P)
                    h_high = thermo.mixture_enthalpy(z, T + d, vf, P=P)
                    h_low = thermo.mixture_enthalpy(z, T - d, vf, P=P)
                    dHdT = (h_high - h_low) / (2.0 * d)
                    self.assertTrue(
                        math.isclose(cp, dHdT, rel_tol=5e-3, abs_tol=0.05),
                        f'{method} {phase}: Cp={cp!r} vs dH/dT={dHdT!r}',
                    )

    def test_cp_reported_in_kJ_per_kmol_K_for_every_method(self):
        """Magnitude window catches mol-basis (/1000) and J-basis (*1000)."""
        for method in SUPPORTED_METHODS:
            _, z = _components_for(method)
            thermo = self._thermo(method)
            T = 450.0 if method == 'STEAM' else 420.0
            with self.subTest(method=method):
                cp = thermo.mixture_Cp(z, T, 1.0, 1.0)
                self.assertGreater(cp, 20.0)
                self.assertLess(cp, 400.0)

    def test_flash_false_without_phase_raises_for_every_method(self):
        for method in SUPPORTED_METHODS:
            if method == 'STEAM':
                continue  # IF97 states are native; flash is not used there
            _, z = _components_for(method)
            thermo = self._thermo(method)
            with self.subTest(method=method):
                with self.assertRaises(ThermodynamicsError):
                    thermo.calculate_state(320.0, 1.0, 1.0, z, flash=False)

    def test_bubble_and_dew_pressures_for_every_activity_method(self):
        """sum(x K) = 1 at the bubble pressure; dew <= bubble; both positive."""
        T = 350.0
        for method in SUPPORTED_METHODS:
            _, z = _components_for(method)
            thermo = self._thermo(method)
            if not isinstance(thermo, ActivityCoefficientThermodynamics):
                continue
            with self.subTest(method=method):
                P_bub = thermo.bubble_point_P(z, T)
                self.assertGreater(P_bub, 0.0)
                K = thermo.K_values(T, P_bub, z)
                residual = sum(z[c] * K.get(c, 1.0) for c in z) - 1.0
                self.assertLess(abs(residual), 1e-6)
                P_dew = thermo.dew_point_P(z, T)
                self.assertGreater(P_dew, 0.0)
                self.assertLessEqual(P_dew, P_bub * (1.0 + 1e-9))

    def test_dew_point_pressure_is_deterministic(self):
        for method in ('UNIFAC', 'UNIFAC-PR', 'UNIFNIST-VDM'):
            _, z = _components_for(method)
            thermo = self._thermo(method)
            with self.subTest(method=method):
                first = thermo.dew_point_P(z, 360.0)
                second = thermo.dew_point_P(z, 360.0)
                self.assertGreater(first, 0.0)
                self.assertTrue(math.isclose(first, second, rel_tol=1e-12))

    def test_vapor_eos_and_legacy_rk_alias(self):
        gamma_phi = self._thermo('NRTL-PR')
        self.assertIsNotNone(gamma_phi.vapor_eos)
        self.assertIs(gamma_phi.rk, gamma_phi.vapor_eos)
        plain = self._thermo('UNIFAC')
        self.assertIsNone(plain.vapor_eos)

    def test_steam_and_ideal_cp_share_a_basis(self):
        ideal = self._thermo('IDEAL')
        steam = self._thermo('STEAM')
        cp_ideal = ideal.mixture_Cp({'water': 1.0}, 450.0, 1.0)
        cp_steam = steam.mixture_Cp({'H2O': 1.0}, 450.0, 1.0)
        self.assertTrue(0.9 < cp_steam / cp_ideal < 1.2)


class SteamForcedPhaseTests(unittest.TestCase):
    """Forced-phase steam properties follow the saturation continuation."""

    def setUp(self):
        self.steam = create_thermodynamics(['H2O'], 'STEAM')

    def test_hot_liquid_properties_are_liquid(self):
        volume = self.steam.mixture_liquid_molar_volume({'H2O': 1.0}, 400.0)
        self.assertTrue(0.018 < volume < 0.021, volume)
        cp = self.steam.mixture_Cp({'H2O': 1.0}, 400.0, 0.0)
        self.assertTrue(72.0 < cp < 82.0, cp)

    def test_cool_vapor_properties_are_vapor(self):
        cp = self.steam.mixture_Cp({'H2O': 1.0}, 350.0, 1.0)
        self.assertTrue(30.0 < cp < 40.0, cp)

    def test_stable_phase_paths_unchanged(self):
        cp = self.steam.mixture_Cp({'H2O': 1.0}, 298.15, 0.0)
        self.assertTrue(74.0 < cp < 77.0, cp)

    def test_known_pressure_beats_the_standard_state(self):
        cp = self.steam.mixture_Cp({'H2O': 1.0}, 400.0, 0.0, 5.0)
        self.assertTrue(74.0 < cp < 80.0, cp)


class FlashFallbackTests(unittest.TestCase):
    """#8: a failed TP flash degrades to K-values, then warns heuristically."""

    @staticmethod
    def _boom(*_args, **_kwargs):
        raise RuntimeError('flash exploded')

    def test_k_value_fallback_picks_the_right_phase(self):
        for method in ('IDEAL', 'UNIFAC', 'PR'):
            comps, z = _components_for(method)
            thermo = create_thermodynamics(comps, method)
            thermo.flash_TP = self._boom
            with self.subTest(method=method):
                vapor = thermo.calculate_state(420.0, 2.0, 1.0, z)
                self.assertEqual(vapor.vapor_fraction, 1.0)
                liquid = thermo.calculate_state(300.0, 2.0, 1.0, z)
                self.assertEqual(liquid.vapor_fraction, 0.0)
                self.assertTrue(
                    any('K-value evaluation' in w for w in thermo.warnings)
                )

    def test_last_resort_heuristic_warns(self):
        thermo = create_thermodynamics(['water', 'ethanol'], 'IDEAL')
        thermo.flash_TP = self._boom
        thermo.K_values = self._boom
        state = thermo.calculate_state(500.0, 1.0, 1.0, dict(WATER_ETHANOL))
        self.assertEqual(state.vapor_fraction, 1.0)
        self.assertTrue(any('both failed' in w for w in thermo.warnings))


class BinaryLLETests(unittest.TestCase):
    """#1: the reference binary splitter handles arbitrary gap locations."""

    def _split(self, thermo_cls, feed_x1: float):
        thermo = thermo_cls(['water', 'ethanol'])
        return thermo.liquid_liquid_equilibrium(
            {'water': feed_x1, 'ethanol': 1.0 - feed_x1}, 298.15
        )

    def _assert_finds_gap(self, xa: float, xb: float, feeds):
        thermo_cls = _margules_for_binodal(xa, xb)
        for feed in feeds:
            with self.subTest(binodal=(xa, xb), feed=feed):
                has, x1p, x2p, beta = self._split(thermo_cls, feed)
                self.assertTrue(has)
                found = sorted((x1p['water'], x2p['water']))
                self.assertAlmostEqual(found[0], min(xa, xb), places=3)
                self.assertAlmostEqual(found[1], max(xa, xb), places=3)
                # Lever rule: z = beta*x2 + (1-beta)*x1
                recon = beta * x2p['water'] + (1.0 - beta) * x1p['water']
                self.assertAlmostEqual(recon, feed, places=6)

    def test_gap_below_half(self):
        self._assert_finds_gap(0.0164, 0.330, (0.08, 0.15, 0.25))

    def test_gap_above_half(self):
        self._assert_finds_gap(0.670, 0.9836, (0.75, 0.85, 0.95))

    def test_gap_straddling_half(self):
        thermo_cls = type('Sym', (_MargulesBinary,), {'A12': 2.8, 'A21': 2.8})
        thermo = thermo_cls(['water', 'ethanol'])
        has, x1p, x2p, _beta = thermo.liquid_liquid_equilibrium(
            {'water': 0.5, 'ethanol': 0.5}, 298.15
        )
        self.assertTrue(has)
        self.assertAlmostEqual(min(x1p['water'], x2p['water']), 0.0927, places=3)
        self.assertAlmostEqual(max(x1p['water'], x2p['water']), 0.9073, places=3)

    def test_feed_outside_gap_does_not_split(self):
        thermo_cls = _margules_for_binodal(0.0164, 0.330)
        has, *_ = self._split(thermo_cls, 0.005)
        self.assertFalse(has)
        has, *_ = self._split(thermo_cls, 0.60)
        self.assertFalse(has)

    def test_miscible_system_does_not_split(self):
        thermo_cls = type('Weak', (_MargulesBinary,), {'A12': 1.0, 'A21': 1.0})
        has, *_ = self._split(thermo_cls, 0.5)
        self.assertFalse(has)


class UniquacQPrimeTests(unittest.TestCase):
    """#4: extended UNIQUAC uses q'-based area fractions in the residual."""

    def _with_q_prime(self):
        thermo = UNIQUACThermodynamics(['water', 'ethanol'])
        thermo.q_prime['water'] = 1.0
        thermo.q_prime['ethanol'] = 0.92
        original = thermo._uniquac_interaction_for_components

        def with_flag(comp_i, comp_j):
            data = original(comp_i, comp_j)
            if data is not None:
                data = dict(data)
                data['use_q_prime'] = True
            return data

        thermo._uniquac_interaction_for_components = with_flag
        thermo._uniquac_parameter_cache = None
        thermo._uniquac_tau_cache.clear()
        thermo._compiled_activity_cache.clear()
        return thermo

    def _gammas(self, thermo, T, x, compiled: bool):
        thermo._activity_cache.clear()
        if compiled:
            return thermo.activity_coefficients(T, x)
        from unittest.mock import patch
        with patch.object(thermo, '_compiled_activity_backend', return_value=None):
            return thermo.activity_coefficients(T, x)

    def test_q_prime_changes_gammas_and_keeps_pure_compiled_parity(self):
        T, x = 350.0, {'water': 0.4, 'ethanol': 0.6}
        baseline = UNIQUACThermodynamics(['water', 'ethanol'])
        base_gamma = self._gammas(baseline, T, x, compiled=False)
        thermo = self._with_q_prime()
        pure = self._gammas(thermo, T, x, compiled=False)
        self.assertTrue(any(abs(pure[c] - base_gamma[c]) > 1e-6 for c in pure))
        if thermo._compiled_activity_backend(T) is not None:
            compiled = self._gammas(thermo, T, x, compiled=True)
            for comp in pure:
                self.assertAlmostEqual(compiled[comp], pure[comp], places=10)


    def test_submixture_retains_package_structure_across_scalar_and_compiled_paths(self):
        from unittest.mock import patch
        from compiled_activity import CompiledUNIQUACBackend

        thermo = UNIQUACThermodynamics(['ethanol', 'water', '1-octanol'])
        composition = {'ethanol': 0.4, 'water': 0.6}
        components = ['ethanol', 'water']
        with patch.object(thermo, '_compiled_activity_backend', return_value=None):
            full = thermo.activity_coefficients(350.0, composition)
            subset = thermo._activity_coefficients_for_components(350.0, composition, components)
        for component in components:
            self.assertAlmostEqual(subset[component], full[component], places=12)

        backend = CompiledUNIQUACBackend.from_thermo(thermo, components)
        if backend is None:
            self.skipTest('Compiled UNIQUAC backend is unavailable')
        values = backend.activity_coefficients([0.4, 0.6], 350.0)
        for index, component in enumerate(components):
            self.assertAlmostEqual(values[index], subset[component], places=12)


    def test_q_prime_model_satisfies_gibbs_duhem(self):
        thermo = self._with_q_prime()
        T, h = 350.0, 1e-6
        ga = self._gammas(thermo, T, {'water': 0.4 + h, 'ethanol': 0.6 - h}, False)
        gb = self._gammas(thermo, T, {'water': 0.4 - h, 'ethanol': 0.6 + h}, False)
        residual = (
            0.4 * (math.log(ga['water']) - math.log(gb['water']))
            + 0.6 * (math.log(ga['ethanol']) - math.log(gb['ethanol']))
        ) / (2.0 * h)
        self.assertLess(abs(residual), 1e-6)


class Flash3RoundtripTests(unittest.TestCase):
    """Structural: the shared PH/PS solver reproduces known flash states."""

    def test_binary_lle_system_ph_and_ps_roundtrip(self):
        thermo = create_thermodynamics(['water', 'butanol'], 'UNIFAC')
        z, P, T0 = {'water': 0.55, 'butanol': 0.45}, 1.0, 355.0
        reference = thermo.flash3_TP(z, T0, P)
        H0 = thermo._flash3_mixture_enthalpy(T0, P, reference)
        S0 = thermo._flash3_mixture_entropy(T0, P, reference)
        T_h, _, res_h = thermo.flash3_PH(z, P, H0, T_guess=340.0)
        T_s, _, res_s = thermo.flash3_PS(z, P, S0, T_guess=340.0)
        self.assertAlmostEqual(T_h, T0, places=3)
        self.assertAlmostEqual(T_s, T0, places=3)
        self.assertLess(abs(res_h), 1.0)
        self.assertLess(abs(res_s), 1e-2)


class ChartConsistencyTests(unittest.TestCase):
    """#9: all diagram generators share the model's own K-values."""

    def test_pxy_charts_are_internally_consistent(self):
        for method in ('UNIFAC', 'UNIFAC-PR', 'UNIFNIST-VDM'):
            comps, _ = _components_for(method)
            thermo = create_thermodynamics(comps, method)
            with self.subTest(method=method):
                data = thermo.generate_Pxy_data(comps[0], comps[1], 350.0, n_points=8)
                self.assertTrue(all(0.0 <= y <= 1.0 for y in data['y']))
                for bubble, dew in zip(data['P_bubble'], data['P_dew']):
                    self.assertGreaterEqual(bubble, dew - 1e-9)


class ImportHygieneTests(unittest.TestCase):
    """Structural: common.py no longer leaks third-party names."""

    def test_common_all_is_curated(self):
        import thermodynamics_models.common as common
        for name in ('np', 'math', 're', 'dataclass', 'field', 'Optional',
                      'brentq', 'least_squares', 'ChemicalDatabase'):
            self.assertNotIn(name, common.__all__)
        self.assertIn('ThermodynamicsError', common.__all__)
        self.assertIn('R', common.__all__)


if __name__ == '__main__':
    unittest.main()
