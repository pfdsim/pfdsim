import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chemical_properties import ChemicalDatabase, ChemicalProperties, get_database
import property_resolver as property_resolver_module
from property_resolver import HeatCapacityIntegralLookup, PropertyResolver, get_property_resolver
from thermodynamics import ActivityCoefficientThermodynamics, T_REF, create_thermodynamics
from physical_constants import R_BAR_M3_MOL_K


class ConstantExcessActivityThermo(ActivityCoefficientThermodynamics):
    """Small test model with a known liquid excess enthalpy."""

    def activity_coefficients(self, T, composition):
        return {comp: 1.0 for comp in self.components}

    def excess_enthalpy(self, composition, T):
        return 123.0


class FakeRK:
    def __init__(self, departure=0.0, phi=None):
        self.departure = departure
        self.phi = phi or {}
        self.fugacity_calls = []

    def departure_enthalpy(self, T, P, composition, phase='vapor'):
        if isinstance(self.departure, Exception):
            raise self.departure
        return self.departure

    def fugacity_coefficients(self, T, P, composition, phase='vapor'):
        self.fugacity_calls.append((T, P, dict(composition), phase))
        return {
            comp: self.phi.get(comp, 1.0)
            for comp in composition
        }


class ConstantResidualActivityThermo(ActivityCoefficientThermodynamics):
    """Small gamma-phi-like test model with a known RK vapor departure."""

    def __init__(self, components, db, departure):
        super().__init__(components, db)
        self.rk = FakeRK(departure)

    def activity_coefficients(self, T, composition):
        return {comp: 1.0 for comp in self.components}

    def excess_enthalpy(self, composition, T):
        return 0.0


class EnthalpyResolutionTests(unittest.TestCase):
    def _custom_db(self, hvap=40.0):
        db = ChemicalDatabase(enable_online=False)
        db.chemicals = {
            'X': ChemicalProperties(
                symbol='X',
                name='Test liquid',
                formula='X',
                MW=50.0,
                Cp_coeffs=[50.0, 0.0, 0.0, 0.0],
                Cp_liquid=100.0,
                Hvap=hvap,
                Tb=350.0,
                Tc=600.0,
                Pc=30.0,
                omega=0.2,
                phase_at_STP='liquid',
            )
        }
        db._build_aliases()
        return db

    def test_liquid_enthalpy_integrates_liquid_cp(self):
        thermo = create_thermodynamics(['X'], 'IDEAL', self._custom_db())

        delta_h = thermo.enthalpy_liquid('X', T_REF + 10.0) - thermo.enthalpy_liquid('X', T_REF)

        self.assertAlmostEqual(delta_h, 1.0, places=8)

    def test_enthalpy_uses_analytic_provided_cp_integrals(self):
        db = self._custom_db()
        props = db.chemicals['X']
        props.Cp_coeffs = [10.0, 2.0, 0.5, 0.1]
        props.Cp_liquid = 123.0
        thermo = create_thermodynamics(['X'], 'IDEAL', db)
        T = T_REF + 12.0

        expected_ideal = (
            10.0 * (T - T_REF)
            + 2.0 * (T**2 - T_REF**2) / 2.0
            + 0.5 * (T**3 - T_REF**3) / 3.0
            + 0.1 * (T**4 - T_REF**4) / 4.0
        ) / 1000.0
        expected_liquid_delta = 123.0 * (T - T_REF) / 1000.0

        with patch.object(thermo, 'Cp_ideal_gas', side_effect=AssertionError('Cp sampled')):
            self.assertAlmostEqual(thermo.enthalpy_ideal_gas('X', T), expected_ideal, places=10)
        with patch.object(thermo, 'Cp_liquid', side_effect=AssertionError('Cp sampled')):
            delta_h = thermo.enthalpy_liquid('X', T) - thermo.enthalpy_liquid('X', T_REF)
            self.assertAlmostEqual(delta_h, expected_liquid_delta, places=10)

    def test_analytic_perry_cp_integrals_match_quadrature(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'IDEAL')

        def simpson(func, T1, T2, steps=400):
            if steps % 2:
                steps += 1
            h = (T2 - T1) / steps
            total = func(T1) + func(T2)
            for index in range(1, steps):
                total += (4 if index % 2 else 2) * func(T1 + index * h)
            return total * h / 3.0 / 1000.0

        liquid = thermo._integrate_cp_analytic('methanol', T_REF, 360.0, 'liquid')
        liquid_quad = simpson(lambda T: thermo.Cp_liquid('methanol', T), T_REF, 360.0)
        vapor = thermo._integrate_cp_analytic('water', T_REF, 450.0, 'ideal_gas')
        vapor_quad = simpson(lambda T: thermo.Cp_ideal_gas('water', T), T_REF, 450.0)

        self.assertIsNotNone(liquid)
        self.assertIsNotNone(vapor)
        self.assertAlmostEqual(liquid, liquid_quad, delta=1e-8)
        self.assertAlmostEqual(vapor, vapor_quad, delta=1e-8)

    def test_analytic_portable_poly_x_cp_integral(self):
        db = ChemicalDatabase(enable_online=False)
        db.chemicals = {
            'X': ChemicalProperties(
                symbol='X',
                name='Portable Cp test',
                formula='X',
                MW=50.0,
                Hvap=40.0,
                Tb=350.0,
                Tc=600.0,
                Pc=30.0,
                phase_at_STP='liquid',
                property_correlations={
                    'Cpg': {
                        'equation': 'poly_x',
                        'coefficients': {'A': 20.0, 'B': 5.0, 'C': 2.0},
                        'Tmin_K': 250.0,
                        'Tmax_K': 500.0,
                    },
                    'Cpl': {
                        'equation': 'poly_x',
                        'coefficients': {'A': 80.0, 'B': 4.0},
                        'Tmin_K': 250.0,
                        'Tmax_K': 500.0,
                    },
                },
            )
        }
        db._build_aliases()
        thermo = create_thermodynamics(['X'], 'IDEAL', db)
        T1, T2 = 300.0, 360.0
        x1 = (T1 - 298.15) / 100.0
        x2 = (T2 - 298.15) / 100.0
        expected_gas = 100.0 * (
            20.0 * (x2 - x1)
            + 5.0 * (x2**2 - x1**2) / 2.0
            + 2.0 * (x2**3 - x1**3) / 3.0
        ) / 1000.0
        expected_liquid = 100.0 * (
            80.0 * (x2 - x1)
            + 4.0 * (x2**2 - x1**2) / 2.0
        ) / 1000.0

        self.assertAlmostEqual(
            thermo._integrate_cp_analytic('X', T1, T2, 'ideal_gas'),
            expected_gas,
            places=12,
        )
        self.assertAlmostEqual(
            thermo._integrate_cp_analytic('X', T1, T2, 'liquid'),
            expected_liquid,
            places=12,
        )

    def test_enthalpy_uses_analytic_online_cp_integral_hook(self):
        db = ChemicalDatabase(enable_online=False)
        db.chemicals = {
            'X': ChemicalProperties(
                symbol='X',
                name='Online Cp integral test',
                formula='X',
                MW=50.0,
                Hvap=40.0,
                Tb=350.0,
                Tc=600.0,
                Pc=30.0,
            )
        }
        db._build_aliases()
        thermo = create_thermodynamics(['X'], 'IDEAL', db)
        T = T_REF + 10.0

        class FakeResolver:
            def integrate_cp_online(self, symbol, T1, T2, phase, props=None):
                return HeatCapacityIntegralLookup(
                    value=1.23,
                    phase=phase,
                    T1=T1,
                    T2=T2,
                    method='test_online_cp_integral',
                )

        with patch('property_resolver.get_property_resolver', return_value=FakeResolver()), \
             patch.object(thermo, 'Cp_ideal_gas', side_effect=AssertionError('Cp sampled')):
            self.assertAlmostEqual(thermo.enthalpy_ideal_gas('X', T), 1.23, places=12)

    def test_liquid_enthalpy_reference_connects_to_temperature_dependent_hvap(self):
        thermo = create_thermodynamics(['X'], 'IDEAL', self._custom_db())

        latent_offset = thermo.enthalpy_ideal_gas('X', T_REF) - thermo.enthalpy_liquid('X', T_REF)

        self.assertAlmostEqual(latent_offset, thermo.Hvap_at_T('X', T_REF), places=8)

    def test_scalar_hvap_fallback_uses_watson_temperature_dependence(self):
        thermo = create_thermodynamics(['X'], 'IDEAL', self._custom_db())

        hvap_cold = thermo.Hvap_at_T('X', 300.0)
        hvap_mid = thermo.Hvap_at_T('X', 450.0)
        hvap_hot = thermo.Hvap_at_T('X', 590.0)

        self.assertGreater(hvap_cold, hvap_mid)
        self.assertGreater(hvap_mid, hvap_hot)
        self.assertGreater(hvap_hot, 0.0)

    def test_perry_hvap_takes_priority_over_chemicals_json_scalar(self):
        thermo = create_thermodynamics(['C2H5OH'], 'IDEAL')
        props = thermo.props['C2H5OH']
        props.Hvap = 999.0
        thermo._resolver_known_props['C2H5OH']['Hvap'] = 999.0
        thermo._hvap_cache.clear()
        thermo._hvap_T_cache.clear()

        hvap = thermo.Hvap_at_T('C2H5OH', 350.0)
        perry = thermo._fast_perry_hvap('C2H5OH', 350.0)

        self.assertIsNotNone(perry)
        self.assertAlmostEqual(hvap, perry, places=10)
        self.assertLess(hvap, 100.0)

    def test_activity_model_liquid_enthalpy_uses_same_liquid_cp_path(self):
        thermo = create_thermodynamics(
            ['X'],
            'UNIFAC',
            self._custom_db(),
            unifac_groups={'X': {'CH3': 1}},
        )

        delta_h = thermo.enthalpy_liquid('X', T_REF + 10.0) - thermo.enthalpy_liquid('X', T_REF)

        self.assertAlmostEqual(delta_h, 1.0, places=8)

    def test_activity_mixture_enthalpy_includes_liquid_excess_enthalpy(self):
        db = self._custom_db()
        thermo = ConstantExcessActivityThermo(['X'], db)
        base = create_thermodynamics(['X'], 'IDEAL', db)
        composition = {'X': 1.0}

        liquid_h = thermo.mixture_enthalpy(composition, 320.0, vapor_fraction=0.0)
        base_liquid_h = base.mixture_enthalpy(composition, 320.0, vapor_fraction=0.0)

        two_phase_h = thermo.mixture_enthalpy(
            composition,
            320.0,
            vapor_fraction=0.25,
            x=composition,
            y=composition,
        )
        base_two_phase_h = base.mixture_enthalpy(
            composition,
            320.0,
            vapor_fraction=0.25,
            x=composition,
            y=composition,
        )

        self.assertAlmostEqual(liquid_h - base_liquid_h, 123.0, places=8)
        self.assertAlmostEqual(two_phase_h - base_two_phase_h, 0.75 * 123.0, places=8)

    def test_gamma_phi_mixture_enthalpy_includes_vapor_residual_enthalpy(self):
        db = self._custom_db()
        thermo = ConstantResidualActivityThermo(['X'], db, departure=-45.0)
        base = create_thermodynamics(['X'], 'IDEAL', db)
        composition = {'X': 1.0}

        vapor_h = thermo.mixture_enthalpy(composition, 450.0, vapor_fraction=1.0, P=5.0)
        base_vapor_h = base.mixture_enthalpy(composition, 450.0, vapor_fraction=1.0, P=5.0)

        two_phase_h = thermo.mixture_enthalpy(
            composition,
            450.0,
            vapor_fraction=0.4,
            x=composition,
            y=composition,
            P=5.0,
        )
        base_two_phase_h = base.mixture_enthalpy(
            composition,
            450.0,
            vapor_fraction=0.4,
            x=composition,
            y=composition,
            P=5.0,
        )

        self.assertAlmostEqual(vapor_h - base_vapor_h, -45.0, places=8)
        self.assertAlmostEqual(two_phase_h - base_two_phase_h, 0.4 * -45.0, places=8)

    def test_gamma_phi_vapor_residual_enthalpy_fallback_warns_once(self):
        thermo = ConstantResidualActivityThermo(
            ['X'],
            self._custom_db(),
            departure=RuntimeError('departure failed'),
        )
        composition = {'X': 1.0}

        first = thermo._vapor_residual_enthalpy(composition, 450.0, 5.0)
        second = thermo._vapor_residual_enthalpy(composition, 450.0, 5.0)

        self.assertEqual(first, 0.0)
        self.assertEqual(second, 0.0)
        warnings = [
            warning for warning in thermo.warnings
            if 'vapor residual enthalpy with the selected backend' in warning
        ]
        self.assertEqual(len(warnings), 1)

    def test_eos_enthalpy_does_not_depend_on_scalar_hvap(self):
        base = get_database().get('H2O', fetch_online=False).to_dict()

        db_low = ChemicalDatabase(enable_online=False)
        db_high = ChemicalDatabase(enable_online=False)
        low = dict(base)
        high = dict(base)
        low['Hvap'] = 1.0
        high['Hvap'] = 999.0
        db_low.chemicals = {'H2O': ChemicalProperties(**low)}
        db_high.chemicals = {'H2O': ChemicalProperties(**high)}
        db_low._build_aliases()
        db_high._build_aliases()

        h_low = create_thermodynamics(['H2O'], 'PR', db_low).calculate_state(
            350.0, 1.0, 1.0, {'H2O': 1.0}, phase='liquid', flash=False
        ).H
        h_high = create_thermodynamics(['H2O'], 'PR', db_high).calculate_state(
            350.0, 1.0, 1.0, {'H2O': 1.0}, phase='liquid', flash=False
        ).H

        self.assertAlmostEqual(h_low, h_high, places=8)


class GammaPhiReferenceStateTests(unittest.TestCase):
    def _custom_db(self, property_correlations=None):
        db = ChemicalDatabase(enable_online=False)
        db.chemicals = {
            'X': ChemicalProperties(
                symbol='X',
                name='Test liquid',
                formula='X',
                MW=50.0,
                Cp_coeffs=[50.0, 0.0, 0.0, 0.0],
                Cp_liquid=100.0,
                Hvap=40.0,
                Tb=350.0,
                Tc=600.0,
                Pc=30.0,
                omega=0.2,
                phase_at_STP='liquid',
                property_correlations=property_correlations or {},
            )
        }
        db._build_aliases()
        return db

    def _thermo(self, *, phi=0.8, property_correlations=None):
        thermo = ConstantResidualActivityThermo(
            ['X'],
            self._custom_db(property_correlations),
            departure=0.0,
        )
        thermo.rk = FakeRK(phi={'X': phi})
        return thermo

    def test_phi_sat_uses_pure_rk_fugacity_at_saturation(self):
        thermo = self._thermo(phi=0.82)

        phi_sat = thermo._gamma_phi_phi_sat('X', 320.0, 3.0)
        cached = thermo._gamma_phi_phi_sat('X', 320.0, 3.0)

        self.assertAlmostEqual(phi_sat, 0.82)
        self.assertAlmostEqual(cached, 0.82)
        self.assertEqual(len(thermo.rk.fugacity_calls), 1)
        self.assertEqual(
            thermo.rk.fugacity_calls[0],
            (320.0, 3.0, {'X': 1.0}, 'vapor'),
        )

    def test_poynting_uses_resolved_liquid_molar_volume(self):
        thermo = self._thermo(
            property_correlations={
                'rhol': {
                    'equation': 'poly_x',
                    'coefficients': {'A': 1000.0},
                    'Tmin_K': 250.0,
                    'Tmax_K': 350.0,
                    'quality': 0.96,
                }
            }
        )

        poynting = thermo._gamma_phi_poynting_factor('X', 300.0, 20.0, 2.0)
        expected = math.exp((50.0 / 1000.0 / 1000.0) * (20.0 - 2.0) / (R_BAR_M3_MOL_K * 300.0))

        self.assertAlmostEqual(poynting, expected, places=12)
        self.assertEqual(thermo.warnings, [])

    def test_poynting_uses_prebound_liquid_volume_without_resolver(self):
        thermo = self._thermo(
            property_correlations={
                'rhol': {
                    'equation': 'poly_x',
                    'coefficients': {'A': 1000.0},
                    'Tmin_K': 250.0,
                    'Tmax_K': 350.0,
                    'quality': 0.96,
                }
            }
        )

        with patch('property_resolver.get_property_resolver') as get_resolver:
            first = thermo._gamma_phi_poynting_factor('X', 300.0, 20.0, 2.0)
            second = thermo._gamma_phi_poynting_factor('X', 300.0, 20.0, 2.0)

        self.assertAlmostEqual(first, second, places=15)
        get_resolver.assert_not_called()
        self.assertIn(('X', 300.0), thermo._liquid_molar_volume_cache)

    def test_poynting_uses_resolver_volume_chain_outside_prebound_extrapolation(self):
        thermo = self._thermo(
            property_correlations={
                'rhol': {
                    'equation': 'poly_x',
                    'coefficients': {'A': 1000.0, 'B': 100.0},
                    'Tmin_K': 250.0,
                    'Tmax_K': 350.0,
                    'quality': 0.96,
                }
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'liquid_volume_zra_cache.json'
            temp_resolver = PropertyResolver()
            with patch.object(PropertyResolver, 'LIQUID_VOLUME_ZRA_CACHE_PATH', cache_path), \
                 patch.object(property_resolver_module, '_resolver', temp_resolver):
                poynting = thermo._gamma_phi_poynting_factor('X', 400.0, 20.0, 2.0)
                resolver_volume = get_property_resolver().resolve_liquid_molar_volume(
                    'X',
                    400.0,
                    thermo._resolver_known_props['X'],
                )

        expected = math.exp((resolver_volume.value / 1000.0) * (20.0 - 2.0) / (R_BAR_M3_MOL_K * 400.0))
        self.assertAlmostEqual(poynting, expected, places=12)
        self.assertEqual(resolver_volume.method, 'rackett_fitted_zra')
        self.assertFalse(any('nearest supported temperature' in warning for warning in thermo.warnings))

    def test_reference_factor_includes_phi_sat_psat_and_poynting(self):
        thermo = self._thermo(
            phi=0.75,
            property_correlations={
                'rhol': {
                    'equation': 'poly_x',
                    'coefficients': {'A': 1000.0},
                    'Tmin_K': 250.0,
                    'Tmax_K': 350.0,
                    'quality': 0.96,
                }
            },
        )
        thermo.Psat = lambda comp, T: 2.0

        reference = thermo._gamma_phi_reference_factors(300.0, 20.0)
        poynting = math.exp((50.0 / 1000.0 / 1000.0) * (20.0 - 2.0) / (R_BAR_M3_MOL_K * 300.0))

        self.assertAlmostEqual(reference['X'], 0.75 * 2.0 * poynting, places=12)


class VDMResidualEnthalpyTests(unittest.TestCase):
    def test_vdm_liquid_enthalpy_uses_apparent_hvap_reference(self):
        thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')
        comp = 'CH3COOH'
        composition = {comp: 1.0, 'H2O': 0.0}
        pressure = thermo.Psat(comp, T_REF)

        apparent_hvap = thermo.Hvap_at_T(comp, T_REF)
        liquid_h = thermo.mixture_enthalpy(
            composition, T_REF, vapor_fraction=0.0, P=pressure
        )
        vapor_h = thermo.mixture_enthalpy(
            composition, T_REF, vapor_fraction=1.0, P=pressure
        )

        self.assertAlmostEqual(
            vapor_h - liquid_h,
            1000.0 * apparent_hvap,
            places=8,
        )

    def test_vdm_pure_saturated_acid_shifts_vapor_and_liquid_equally(self):
        thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')
        base = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC')
        comp = 'CH3COOH'
        composition = {comp: 1.0, 'H2O': 0.0}
        T = thermo.props[comp].Tb
        P = thermo.Psat(comp, T)

        vapor_h = thermo.mixture_enthalpy(composition, T, vapor_fraction=1.0, P=P)
        liquid_h = thermo.mixture_enthalpy(composition, T, vapor_fraction=0.0, P=P)
        base_vapor_h = base.mixture_enthalpy(composition, T, vapor_fraction=1.0, P=P)
        base_liquid_h = base.mixture_enthalpy(composition, T, vapor_fraction=0.0, P=P)
        residual = thermo._vapor_residual_enthalpy(composition, T, P)
        reference = thermo._vdm_pure_saturated_association_enthalpy(comp, T)

        self.assertAlmostEqual(residual, reference, places=8)
        self.assertAlmostEqual(vapor_h - base_vapor_h, reference, places=8)
        self.assertAlmostEqual(liquid_h - base_liquid_h, reference, places=8)
        self.assertAlmostEqual(vapor_h - liquid_h, base_vapor_h - base_liquid_h, places=8)

    def test_vdm_saturated_association_reference_keeps_physical_calculation(self):
        thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')
        comp = 'CH3COOH'
        T = 391.0
        pressure = thermo.Psat(comp, T)
        model = thermo._vdm_models[comp]
        state = model.association_state(
            T,
            pressure,
            {comp: 1.0},
            rk_model=None,
        )

        self.assertAlmostEqual(
            thermo._vdm_pure_saturated_association_enthalpy(comp, T),
            state['association_enthalpy'],
            places=10,
        )

    def test_vdm_pure_saturated_acid_has_no_high_temperature_reference_drift(self):
        thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')
        base = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC')
        comp = 'CH3COOH'
        composition = {comp: 1.0, 'H2O': 0.0}
        T = 450.0
        P = thermo.Psat(comp, T)

        vapor_h = thermo.mixture_enthalpy(composition, T, vapor_fraction=1.0, P=P)
        liquid_h = thermo.mixture_enthalpy(composition, T, vapor_fraction=0.0, P=P)
        base_vapor_h = base.mixture_enthalpy(composition, T, vapor_fraction=1.0, P=P)
        base_liquid_h = base.mixture_enthalpy(composition, T, vapor_fraction=0.0, P=P)
        residual = thermo._vapor_residual_enthalpy(composition, T, P)
        reference = thermo._vdm_pure_saturated_association_enthalpy(comp, T)

        self.assertAlmostEqual(residual, reference, places=8)
        self.assertAlmostEqual(vapor_h - base_vapor_h, reference, places=8)
        self.assertAlmostEqual(liquid_h - base_liquid_h, reference, places=8)
        self.assertAlmostEqual(vapor_h - liquid_h, base_vapor_h - base_liquid_h, places=8)

    def test_vdm_vapor_residual_enthalpy_uses_actual_single_acid_extent(self):
        thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')
        base = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC')
        composition = {'CH3COOH': 0.5, 'H2O': 0.5}
        T = 391.0
        P = 1.01325

        model = thermo._vdm_models['CH3COOH']
        mix_state = model.association_state(T, P, composition, rk_model=None)
        expected = mix_state['association_enthalpy']

        residual = thermo._vapor_residual_enthalpy(composition, T, P)
        vdm_h = thermo.mixture_enthalpy(composition, T, vapor_fraction=1.0, P=P)
        base_h = base.mixture_enthalpy(composition, T, vapor_fraction=1.0, P=P)

        self.assertLess(residual, 0.0)
        self.assertAlmostEqual(residual, expected, places=8)
        self.assertAlmostEqual(vdm_h - base_h, expected, places=8)

    def test_vdm_fugacity_and_residual_enthalpy_share_single_acid_state(self):
        thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')
        composition = {'CH3COOH': 0.5, 'H2O': 0.5}
        model = thermo._vdm_models['CH3COOH']
        thermo._vdm_pure_saturated_association_enthalpy('CH3COOH', 391.0)
        original = model._equilibrium_state
        calls = {'count': 0}

        def counted(*args, **kwargs):
            calls['count'] += 1
            return original(*args, **kwargs)

        model._equilibrium_state = counted
        thermo.fugacity_coefficients(391.0, 1.01325, composition)
        thermo._vapor_residual_enthalpy(composition, 391.0, 1.01325)

        self.assertEqual(calls['count'], 0)

    def test_vdm_vapor_residual_enthalpy_uses_actual_multi_acid_extents(self):
        thermo = create_thermodynamics(['CH3COOH', 'C2H5COOH', 'H2O'], 'UNIFAC-VDM')
        composition = {'CH3COOH': 0.2, 'C2H5COOH': 0.15, 'H2O': 0.65}
        T = 391.0
        P = 1.01325

        _active, model = thermo._active_vdm_model(composition)
        mix_state = model.association_state(T, P, composition, rk_model=None)
        expected = mix_state['association_enthalpy']

        residual = thermo._vapor_residual_enthalpy(composition, T, P)

        self.assertLess(residual, 0.0)
        self.assertAlmostEqual(residual, expected, places=8)

    def test_vdm_actual_vapor_association_tends_to_zero_with_pressure(self):
        thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')
        composition = {'CH3COOH': 0.5, 'H2O': 0.5}

        atmospheric = thermo._vapor_residual_enthalpy(
            composition, 450.0, 1.01325
        )
        dilute = thermo._vapor_residual_enthalpy(
            composition, 450.0, 1.0e-6
        )

        self.assertLess(atmospheric, 0.0)
        self.assertLess(abs(dilute), abs(atmospheric) * 2.0e-6)

    def test_vdm_liquid_and_vapor_cp_match_enthalpy_derivatives(self):
        thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')
        composition = {'CH3COOH': 1.0, 'H2O': 0.0}
        T = 391.0
        P = thermo.Psat('CH3COOH', T)
        step = 0.01

        for vapor_fraction in (0.0, 1.0):
            with self.subTest(vapor_fraction=vapor_fraction):
                derivative = (
                    thermo.mixture_enthalpy(
                        composition, T + step, vapor_fraction, P=P
                    )
                    - thermo.mixture_enthalpy(
                        composition, T - step, vapor_fraction, P=P
                    )
                ) / (2.0 * step)
                cp = thermo.mixture_Cp(
                    composition,
                    T,
                    vapor_fraction,
                    P=P,
                )
                self.assertAlmostEqual(derivative, cp, places=4)

    def test_vdm_fugacity_and_residual_enthalpy_share_multi_acid_state(self):
        thermo = create_thermodynamics(['CH3COOH', 'C2H5COOH', 'H2O'], 'UNIFAC-VDM')
        composition = {'CH3COOH': 0.2, 'C2H5COOH': 0.15, 'H2O': 0.65}
        _active, model = thermo._active_vdm_model(composition)
        original = model._solve_true_moles
        calls = {'count': 0}

        def counted(*args, **kwargs):
            calls['count'] += 1
            return original(*args, **kwargs)

        model._solve_true_moles = counted
        thermo.fugacity_coefficients(391.0, 1.01325, composition)
        thermo._vapor_residual_enthalpy(composition, 391.0, 1.01325)

        self.assertEqual(calls['count'], 1)

    def test_vdm_pure_reference_enthalpy_fallback_warns_once(self):
        thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')
        model = thermo._vdm_models['CH3COOH']
        original = model.association_state

        def failing(*args, **kwargs):
            raise RuntimeError('reference failed')

        model.association_state = failing
        try:
            first = thermo._vdm_pure_saturated_association_enthalpy('CH3COOH', 391.0)
            second = thermo._vdm_pure_saturated_association_enthalpy('CH3COOH', 391.0)
        finally:
            model.association_state = original

        self.assertEqual(first, 0.0)
        self.assertEqual(second, 0.0)
        warnings = [
            warning for warning in thermo.warnings
            if 'pure saturated VDM association enthalpy reference' in warning
        ]
        self.assertEqual(len(warnings), 1)

    def test_vdm_residual_enthalpy_fallback_warns_once(self):
        thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')
        composition = {'CH3COOH': 0.5, 'H2O': 0.5}
        _active, model = thermo._active_vdm_model(composition)
        original = model.association_state

        def failing(*args, **kwargs):
            raise RuntimeError('association failed')

        model.association_state = failing
        try:
            first = thermo._vapor_residual_enthalpy(composition, 391.0, 1.01325)
            second = thermo._vapor_residual_enthalpy(composition, 391.0, 1.01325)
        finally:
            model.association_state = original

        self.assertEqual(first, 0.0)
        self.assertEqual(second, 0.0)
        warnings = [
            warning for warning in thermo.warnings
            if 'VDM vapor association residual enthalpy' in warning
        ]
        self.assertEqual(len(warnings), 1)


if __name__ == '__main__':
    unittest.main()
