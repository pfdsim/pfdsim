import json
import math
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
from scipy.optimize import brentq, least_squares

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pfd_parser import parse_pfd
from simulator import Simulator
from thermodynamics import ThermodynamicsError, create_thermodynamics
from chemical_properties import ChemicalDatabase, ChemicalProperties
from cubic_eos import CubicEOS
from rk_eos import RedlichKwong
from unifac import (
    DORTMUND_UNIFAC_KNOWN_OVERRIDES,
    UNIFACModel,
    get_unifac_groups,
)
from interaction_parameters import (
    _select_eos_kij,
    cas_for_component,
    eos_binary_interaction,
    nrtl_binary_interaction,
    uniquac_binary_interaction,
    uniquac_rq_for_component,
)
from property_resolution.coolprop import (
    coolprop_reference_for,
    coolprop_saturation_pressure,
)
from scripts.build_cas_interaction_parameters import (
    build_interaction_payload,
    resolve_component_ids,
    supplemental_acetic_acid_vle_records,
    supplemental_ester_alcohol_fit_records,
    supplemental_literature_vle_activity_records,
    supplemental_water_organic_binary_fit_records,
)
from scripts.build_uniquac_rq_parameters import build_uniquac_rq_payload


class ThermodynamicMethodTests(unittest.TestCase):
    def _coolprop_methanol_psat_at_80c(self):
        reference = coolprop_reference_for(
            'methanol',
            {'CAS': '67-56-1'},
        )
        if reference is None:
            self.skipTest('CoolProp methanol reference is unavailable')
        pressure_pa = coolprop_saturation_pressure(reference, 353.15)
        if pressure_pa is None:
            self.skipTest('CoolProp methanol saturation pressure is unavailable')
        pressure_bar = pressure_pa / 1.0e5
        self.assertAlmostEqual(pressure_bar, 1.8111262316, places=8)
        return pressure_bar

    def test_thermo_psat_uses_linear_pressure_continuation_above_tc(self):
        thermo = create_thermodynamics(['CO2'], 'IDEAL')
        coefficients = thermo.get_Psat_coefficients('CO2')
        Tc = coefficients[8]
        supercritical_slope = coefficients[10]
        critical_pressure = thermo.Psat('CO2', Tc)
        delta_T = 25.0

        self.assertGreater(supercritical_slope, 0.0)
        self.assertAlmostEqual(
            thermo.Psat('CO2', Tc + delta_T),
            critical_pressure + supercritical_slope * delta_T,
            places=10,
        )

    def test_thermo_psat_uses_inverse_temperature_continuation_below_tmin(self):
        thermo = create_thermodynamics(['C3H8O3'], 'IDEAL')
        coefficients = thermo.get_Psat_coefficients('C3H8O3')
        T_min = coefficients[11]
        lower_slope = coefficients[12]
        pressure_at_min = thermo.Psat('C3H8O3', T_min)
        temperature = 0.95 * T_min
        expected = pressure_at_min * math.exp(
            -T_min**2 * lower_slope * (1.0 / temperature - 1.0 / T_min)
        )

        self.assertGreater(lower_slope, 0.0)
        self.assertLess(temperature, T_min)
        self.assertAlmostEqual(
            thermo.Psat('C3H8O3', temperature),
            expected,
            places=16,
        )
        self.assertLess(thermo.Psat('C3H8O3', 211.1924260923589), 1.0e-10)

    def test_transport_helpers_return_separate_two_phase_properties(self):
        thermo = create_thermodynamics(['C3H8', 'C4H10'], 'PR')
        composition = {'C3H8': 0.5, 'C4H10': 0.5}
        T = 323.15
        P = 10.0
        vapor_fraction, liquid_x, vapor_y = thermo.flash_TP(composition, T, P)

        viscosities = thermo.transport_mixture_viscosity(
            composition, T, P, vapor_fraction, liquid_x, vapor_y
        )
        densities = thermo.transport_mixture_density(
            composition, T, P, vapor_fraction, liquid_x, vapor_y
        )

        expected_liquid_mu = thermo.mixture_viscosity(
            liquid_x, T, P, 0.0, x=liquid_x
        )
        expected_vapor_mu = thermo.mixture_viscosity(
            vapor_y, T, P, 1.0, y=vapor_y
        )
        expected_liquid_rho = (
            thermo.mixture_molar_density(liquid_x, T, P, 0.0, x=liquid_x)
            * thermo.mixture_MW(liquid_x)
        )
        expected_vapor_rho = (
            thermo.mixture_molar_density(vapor_y, T, P, 1.0, y=vapor_y)
            * thermo.mixture_MW(vapor_y)
        )
        self.assertAlmostEqual(viscosities.liquid, expected_liquid_mu, places=16)
        self.assertAlmostEqual(viscosities.vapor, expected_vapor_mu, places=16)
        self.assertAlmostEqual(densities.liquid, expected_liquid_rho, places=12)
        self.assertAlmostEqual(densities.vapor, expected_vapor_rho, places=12)
        self.assertGreater(densities.liquid, densities.vapor)

    def test_formula_like_database_symbols_remain_specific_for_transport_resolution(self):
        thermo = create_thermodynamics(['CH3COCH3'], 'IDEAL')
        viscosity = thermo.transport_mixture_viscosity(
            {'CH3COCH3': 1.0},
            333.15,
            1.0,
            vapor_fraction=0.0,
        )

        self.assertAlmostEqual(viscosity.liquid, 0.00022858366418377558)

    def test_transport_helpers_only_resolve_active_single_phase(self):
        thermo = create_thermodynamics(['C3H8'], 'IDEAL')
        composition = {'C3H8': 1.0}

        vapor_mu = thermo.transport_mixture_viscosity(
            composition, 350.0, 5.0, vapor_fraction=1.0
        )
        vapor_rho = thermo.transport_mixture_density(
            composition, 350.0, 5.0, vapor_fraction=1.0
        )
        liquid_mu = thermo.transport_mixture_viscosity(
            composition, 250.0, 5.0, vapor_fraction=0.0
        )
        liquid_rho = thermo.transport_mixture_density(
            composition, 250.0, 5.0, vapor_fraction=0.0
        )

        self.assertIsNone(vapor_mu.liquid)
        self.assertIsNotNone(vapor_mu.vapor)
        self.assertIsNone(vapor_rho.liquid)
        self.assertIsNotNone(vapor_rho.vapor)
        self.assertIsNotNone(liquid_mu.liquid)
        self.assertIsNone(liquid_mu.vapor)
        self.assertIsNotNone(liquid_rho.liquid)
        self.assertIsNone(liquid_rho.vapor)

    def test_calculate_state_can_include_vapor_mixture_viscosity(self):
        thermo = create_thermodynamics(['C3H8', 'C4H10'], 'IDEAL')
        composition = {'C3H8': 0.4, 'C4H10': 0.6}

        state = thermo.calculate_state(
            350.0,
            5.0,
            1.0,
            composition,
            phase='vapor',
            flash=False,
            include=('mu',),
        )

        self.assertAlmostEqual(state.mu, 9.049330002871137e-06, places=14)
        self.assertEqual(state.to_dict()['mu'], state.mu)
        self.assertEqual(state.copy().mu, state.mu)
        self.assertIsNone(state.H)
        self.assertIsNone(state.Cp)
        self.assertTrue(any(
            row['property'] == 'vapor_viscosity(T)'
            for row in thermo.lazy_property_quality_sources()
        ))

    def test_calculate_state_can_include_liquid_mixture_viscosity(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'IDEAL')

        state = thermo.calculate_state(
            298.15,
            1.0,
            1.0,
            {'ethanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
            include=('viscosity',),
        )

        self.assertAlmostEqual(state.mu, 0.0021640087685553034, places=16)
        self.assertIsNone(thermo.calculate_state(
            298.15,
            1.0,
            1.0,
            {'ethanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
            include=('rho',),
        ).mu)

    def test_activity_calculate_state_can_include_mixture_viscosity(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIFAC')

        state = thermo.calculate_state(
            298.15,
            1.0,
            1.0,
            {'ethanol': 0.4, 'water': 0.6},
            phase='liquid',
            flash=False,
            include=('mu',),
        )

        self.assertAlmostEqual(state.mu, 0.0021640087685553034, places=16)

    def test_pr_flash_iterates_composition_dependent_k_values(self):
        thermo = create_thermodynamics(['C3H8', 'C4H10'], 'PR')
        self.assertEqual(thermo.props['C4H10'].name, 'n-Butane')

        vapor_fraction, liquid_x, vapor_y = thermo.flash_TP(
            {'C3H8': 0.5, 'C4H10': 0.5},
            323.15,
            10.0,
        )

        self.assertAlmostEqual(vapor_fraction, 0.227121267, places=8)
        self.assertAlmostEqual(liquid_x['C3H8'], 0.446382688, places=8)
        self.assertAlmostEqual(vapor_y['C3H8'], 0.682456186, places=8)

    def test_unifac_rk_gamma_phi_changes_high_pressure_vle(self):
        composition = {'ethanol': 0.4, 'water': 0.6}
        unifac = create_thermodynamics(['ethanol', 'water'], 'UNIFAC')
        gamma_phi = create_thermodynamics(['ethanol', 'water'], 'UNIFAC-RK')

        T_unifac = unifac.bubble_point_T(composition, 10.0)
        T_gamma_phi = gamma_phi.bubble_point_T(composition, 10.0)
        T_gamma_phi_1bar = gamma_phi.bubble_point_T(composition, 1.0)

        self.assertLess(T_gamma_phi, T_unifac)
        self.assertGreater(T_gamma_phi, T_gamma_phi_1bar + 40.0)
        K = gamma_phi.K_values(420.0, 10.0, composition)
        self.assertGreater(K['ethanol'], K['water'])

    def test_unifac_rk_10bar_ethanol_water_azeotrope_regression(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIFAC-RK')

        def y_minus_x(x_ethanol):
            composition = {'ethanol': x_ethanol, 'water': 1.0 - x_ethanol}
            T = thermo.bubble_point_T(composition, 10.0)
            K = thermo.K_values(T, 10.0, composition)
            y_ethanol = composition['ethanol'] * K['ethanol']
            y_water = composition['water'] * K['water']
            y_ethanol /= y_ethanol + y_water
            return y_ethanol - x_ethanol

        x_azeotrope = brentq(y_minus_x, 0.75, 0.95)
        composition = {'ethanol': x_azeotrope, 'water': 1.0 - x_azeotrope}
        T_azeotrope = thermo.bubble_point_T(composition, 10.0)

        self.assertAlmostEqual(x_azeotrope, 0.8112, delta=0.003)
        self.assertAlmostEqual(T_azeotrope - 273.15, 149.37, delta=0.3)

    def test_uniquac_rk_11atm_methanol_water_reference(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIQUAC-RK')
        pressure_bar = 11.2 * 1.01325
        reference = [
            (0.038, 0.164, 177.9),
            (0.067, 0.241, 174.7),
            (0.104, 0.338, 170.1),
            (0.171, 0.434, 165.6),
            (0.240, 0.512, 161.7),
            (0.329, 0.583, 158.2),
            (0.443, 0.664, 154.3),
            (0.511, 0.710, 152.1),
            (0.670, 0.804, 148.0),
            (0.717, 0.825, 146.9),
            (0.845, 0.908, 144.5),
        ]

        temperature_errors = []
        y_errors = []
        for x_methanol, y_methanol_ref, T_c_ref in reference:
            composition = {'methanol': x_methanol, 'water': 1.0 - x_methanol}
            T = thermo.bubble_point_T(composition, pressure_bar)
            K = thermo.K_values(T, pressure_bar, composition)
            y_methanol = composition['methanol'] * K['methanol']
            y_water = composition['water'] * K['water']
            y_methanol /= y_methanol + y_water
            temperature_errors.append(abs((T - 273.15) - T_c_ref))
            y_errors.append(abs(y_methanol - y_methanol_ref))

        self.assertLess(sum(temperature_errors) / len(temperature_errors), 0.4)
        self.assertLess(sum(y_errors) / len(y_errors), 0.015)

    def test_gamma_phi_bubble_scan_ignores_low_temperature_reference_artifact(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIQUAC-PR')
        self.assertEqual(getattr(thermo.rk, 'model', None), 'PR')

        composition = {'methanol': 0.443, 'water': 0.557}
        T = thermo.bubble_point_T(composition, 11.2 * 1.01325)

        self.assertGreater(T, 400.0)
        self.assertLess(T, 450.0)
        self.assertAlmostEqual(T - 273.15, 153.82, delta=0.5)

    def test_gamma_phi_bubble_scan_uses_pressure_saturation_window(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIQUAC-RK')
        pressure = 30.0

        for composition in [
            {'methanol': 1.0, 'water': 0.0},
            {'methanol': 0.443, 'water': 0.557},
            {'methanol': 0.0, 'water': 1.0},
        ]:
            with self.subTest(composition=composition):
                T = thermo.bubble_point_T(composition, pressure)
                K = thermo.K_values(T, pressure, composition)
                self.assertAlmostEqual(
                    sum(composition[comp] * K[comp] for comp in composition),
                    1.0,
                    delta=2e-6,
                )

    def test_bubble_scan_warns_when_pressure_exceeds_pure_critical_pressure(self):
        thermo = create_thermodynamics(['methanol', 'water'], 'UNIQUAC-RK')
        composition = {'methanol': 0.443, 'water': 0.557}

        T = thermo.bubble_point_T(composition, 90.0)
        K = thermo.K_values(T, 90.0, composition)

        self.assertAlmostEqual(
            sum(composition[comp] * K[comp] for comp in composition),
            1.0,
            delta=2e-6,
        )
        self.assertTrue(
            any('above the pure critical pressure for methanol' in warning for warning in thermo.warnings)
        )

    def test_added_property_methods_construct_and_return_directional_k_values(self):
        composition = {'ethanol': 0.4, 'water': 0.6}
        for method in [
            'SRK', 'PR', 'PSRK', 'RKS-BM', 'PR-BM', 'SRK-MC', 'PR-MC', 'PRSV1', 'PRSV2',
            'SRK-TWU', 'PR-TWU',
            'UNIFAC2', 'UNIFDMD', 'UNIFM2', 'UNIFNIST',
            'UNIFDMD-RK', 'UNIFNIST-RK',
            'UNIFDMD-PR', 'UNIFNIST-PR', 'UNIFAC-PR',
            'UNIFAC-VDM', 'UNIFDMD-VDM', 'UNIFNIST-VDM', 'UNIQUAC-VDM',
            'NRTL', 'NRTL-VDM', 'NRTL-RK', 'NRTL-PR',
            'UNIQUAC', 'UNIQUAC-RK', 'UNIQUAC-PR',
        ]:
            with self.subTest(method=method):
                thermo = create_thermodynamics(['ethanol', 'water'], method)
                T = thermo.bubble_point_T(composition, 10.0)
                K = thermo.K_values(T, 10.0, composition)
                self.assertGreater(T, 350.0)
                self.assertGreater(K['ethanol'], K['water'])
                self.assertAlmostEqual(
                    composition['ethanol'] * K['ethanol']
                    + composition['water'] * K['water'],
                    1.0,
                    delta=2e-4,
                )
                if method in ('UNIFNIST-RK', 'NRTL-RK', 'UNIQUAC-RK', 'UNIFNIST-PR', 'NRTL-PR', 'UNIQUAC-PR'):
                    self.assertAlmostEqual(
                        thermo.K_value('ethanol', T, 10.0, composition),
                        K['ethanol'],
                        places=10,
                    )

    def test_compiled_cubic_eos_matches_python_reference_for_all_variants(self):
        methods = (
            'SRK', 'PR', 'RKS-BM', 'PR-BM', 'SRK-MC', 'PR-MC',
            'PRSV1', 'PRSV2', 'SRK-TWU', 'PR-TWU',
        )
        states = (
            (350.0, 5.0, {'ethanol': 0.4, 'water': 0.6}),
            (700.0, 40.0, {'ethanol': 0.2, 'water': 0.8}),
        )
        database = ChemicalDatabase(enable_online=False)

        for method in methods:
            for temperature, pressure, composition in states:
                with self.subTest(method=method, temperature=temperature):
                    compiled = CubicEOS(
                        ['ethanol', 'water'], method, database
                    )
                    if compiled._compiled_backend is None:
                        self.skipTest(
                            'Numba compiled cubic EOS backend is unavailable'
                        )
                    self.assertTrue(
                        compiled._compiled_backend.compilation_complete
                    )
                    reference = CubicEOS(
                        ['ethanol', 'water'], method, database
                    )
                    reference._compiled_backend = None

                    compiled_mixture = compiled.mixture_params(
                        temperature, composition
                    )
                    reference_mixture = reference.mixture_params(
                        temperature, composition
                    )
                    self.assertAlmostEqual(
                        compiled_mixture[0], reference_mixture[0], places=10
                    )
                    self.assertAlmostEqual(
                        compiled_mixture[1], reference_mixture[1], places=12
                    )
                    self.assertAlmostEqual(
                        compiled.mixture_da_dT(temperature, composition),
                        reference.mixture_da_dT(temperature, composition),
                        places=9,
                    )

                    compiled_roots = compiled.compressibility_roots(
                        temperature, pressure, composition
                    )
                    reference_roots = reference.compressibility_roots(
                        temperature, pressure, composition
                    )
                    self.assertEqual(len(compiled_roots), len(reference_roots))
                    for compiled_value, reference_value in zip(
                        compiled_roots, reference_roots
                    ):
                        self.assertAlmostEqual(
                            compiled_value, reference_value, places=11
                        )

                    for phase in ('liquid', 'vapor'):
                        compiled_phi = compiled.fugacity_coefficients(
                            temperature, pressure, composition, phase
                        )
                        reference_phi = reference.fugacity_coefficients(
                            temperature, pressure, composition, phase
                        )
                        for component in composition:
                            self.assertAlmostEqual(
                                compiled_phi[component],
                                reference_phi[component],
                                places=10,
                            )
                        self.assertAlmostEqual(
                            compiled.departure_enthalpy(
                                temperature, pressure, composition, phase
                            ),
                            reference.departure_enthalpy(
                                temperature, pressure, composition, phase
                            ),
                            places=8,
                        )
                        self.assertAlmostEqual(
                            compiled.departure_entropy(
                                temperature, pressure, composition, phase
                            ),
                            reference.departure_entropy(
                                temperature, pressure, composition, phase
                            ),
                            places=10,
                        )

                    compiled_K = compiled.phi_phi_K_values(
                        temperature, pressure, composition
                    )
                    reference_K = reference.phi_phi_K_values(
                        temperature, pressure, composition
                    )
                    for component in composition:
                        self.assertAlmostEqual(
                            compiled_K[component],
                            reference_K[component],
                            places=9,
                        )

    def test_compiled_redlich_kwong_matches_python_reference(self):
        database = ChemicalDatabase(enable_online=False)
        states = (
            (180.0, 20.0, {'CH4': 0.7, 'C2H6': 0.3}),
            (350.0, 50.0, {'CH4': 0.4, 'C2H6': 0.6}),
        )
        for temperature, pressure, composition in states:
            with self.subTest(temperature=temperature):
                compiled = RedlichKwong(['CH4', 'C2H6'], database)
                if compiled._compiled_backend is None:
                    self.skipTest(
                        'Numba compiled cubic EOS backend is unavailable'
                    )
                self.assertTrue(
                    compiled._compiled_backend.compilation_complete
                )
                reference = RedlichKwong(['CH4', 'C2H6'], database)
                reference._compiled_backend = None

                compiled_roots = compiled.compressibility_cubic(
                    temperature, pressure, composition
                )
                reference_roots = reference.compressibility_cubic(
                    temperature, pressure, composition
                )
                self.assertEqual(len(compiled_roots), len(reference_roots))
                for compiled_value, reference_value in zip(
                    compiled_roots, reference_roots
                ):
                    self.assertAlmostEqual(
                        compiled_value, reference_value, places=11
                    )

                for phase in ('liquid', 'vapor'):
                    compiled_phi = compiled.fugacity_coefficients(
                        temperature, pressure, composition, phase
                    )
                    reference_phi = reference.fugacity_coefficients(
                        temperature, pressure, composition, phase
                    )
                    for component in composition:
                        self.assertAlmostEqual(
                            compiled_phi[component],
                            reference_phi[component],
                            places=10,
                        )
                    self.assertAlmostEqual(
                        compiled.departure_enthalpy(
                            temperature, pressure, composition, phase
                        ),
                        reference.departure_enthalpy(
                            temperature, pressure, composition, phase
                        ),
                        places=8,
                    )
                    self.assertAlmostEqual(
                        compiled.departure_entropy(
                            temperature, pressure, composition, phase
                        ),
                        reference.departure_entropy(
                            temperature, pressure, composition, phase
                        ),
                        places=10,
                    )

                compiled_K = compiled.phi_phi_K_values(
                    temperature, pressure, composition
                )
                reference_K = reference.phi_phi_K_values(
                    temperature, pressure, composition
                )
                for component in composition:
                    self.assertAlmostEqual(
                        compiled_K[component],
                        reference_K[component],
                        places=9,
                    )

    def test_srk_mc_loads_psrk_parameters_and_uses_mc_c1_for_supercritical_bm_branch(self):
        eos = CubicEOS(['CH4'], 'SRK-MC', ChemicalDatabase(enable_online=False))
        params = eos.params['CH4']

        self.assertAlmostEqual(params.mc_c1, 0.49258)
        self.assertAlmostEqual(params.mc_c2, 0.0)
        self.assertAlmostEqual(params.mc_c3, 0.0)
        self.assertEqual(
            params.mc_source,
            'PSRK 2005 supplementary pure-component table',
        )

        subcritical_T = 0.8 * params.Tc
        theta = 1.0 - math.sqrt(subcritical_T / params.Tc)
        expected_subcritical = (
            1.0
            + params.mc_c1 * theta
            + params.mc_c2 * theta**2
            + params.mc_c3 * theta**3
        ) ** 2
        self.assertAlmostEqual(eos._alpha(params, subcritical_T), expected_subcritical)

        supercritical_T = 1.2 * params.Tc
        Tr = supercritical_T / params.Tc
        c = 1.0 - 2.0 / (2.0 + params.mc_c1)
        power = 1.0 + params.mc_c1 / 2.0
        expected_supercritical = math.exp(2.0 * c * (1.0 - Tr**power))
        self.assertAlmostEqual(eos._alpha(params, supercritical_T), expected_supercritical)

        soave_m = eos._m_soave(params.omega)
        soave_c = 1.0 - 2.0 / (2.0 + soave_m)
        soave_power = 1.0 + soave_m / 2.0
        soave_supercritical = math.exp(2.0 * soave_c * (1.0 - Tr**soave_power))
        self.assertNotAlmostEqual(eos._alpha(params, supercritical_T), soave_supercritical)

    def test_srk_mc_derivative_matches_finite_difference(self):
        eos = CubicEOS(['CH4'], 'SRK-MC', ChemicalDatabase(enable_online=False))
        composition = {'CH4': 1.0}

        for T in (0.85 * eos.params['CH4'].Tc, 1.2 * eos.params['CH4'].Tc):
            with self.subTest(T=T):
                h = 1e-3
                finite_difference = (
                    eos.mixture_params(T + h, composition)[0]
                    - eos.mixture_params(T - h, composition)[0]
                ) / (2.0 * h)

                self.assertAlmostEqual(
                    eos.mixture_da_dT(T, composition),
                    finite_difference,
                    delta=max(abs(finite_difference) * 1e-6, 1e-8),
                )

    def test_srk_mc_improves_methanol_pure_eos_psat_at_80c(self):
        db = ChemicalDatabase(enable_online=False)
        if db.get('methanol', fetch_online=False) is None:
            self.skipTest('methanol is not available in the local database')

        comp = 'methanol'
        T = 80.0 + 273.15
        reference_psat = self._coolprop_methanol_psat_at_80c()

        def pure_eos_psat(model: str) -> float:
            eos = CubicEOS([comp], model, db)
            composition = {comp: 1.0}

            def residual(P_bar: float) -> float:
                phi_l = eos.fugacity_coefficients(T, P_bar, composition, 'liquid')[comp]
                phi_v = eos.fugacity_coefficients(T, P_bar, composition, 'vapor')[comp]
                return math.log(phi_l) - math.log(phi_v)

            return brentq(
                residual,
                0.2 * reference_psat,
                3.0 * reference_psat,
                xtol=1e-10,
                rtol=1e-10,
            )

        srk_psat = pure_eos_psat('SRK')
        srk_mc_psat = pure_eos_psat('SRK-MC')
        srk_error = abs(srk_psat - reference_psat) / reference_psat
        srk_mc_error = abs(srk_mc_psat - reference_psat) / reference_psat

        self.assertAlmostEqual(srk_psat, 1.7851763036, places=8)
        self.assertAlmostEqual(srk_mc_psat, 1.7869471810, places=8)
        self.assertLess(srk_mc_error, srk_error)

    def test_prsv1_loads_table_parameters_and_uses_smooth_taper(self):
        eos = CubicEOS(['CH4'], 'PRSV1', ChemicalDatabase(enable_online=False))
        params = eos.params['CH4']

        self.assertAlmostEqual(params.kappa1, -0.00159)
        self.assertAlmostEqual(params.kappa2, 0.0)
        self.assertAlmostEqual(params.kappa3, 0.0)

        for Tr in (0.6, 0.85, 1.2):
            with self.subTest(Tr=Tr):
                sqrt_Tr = math.sqrt(Tr)
                taper, _ = eos._prsv1_taper(Tr)
                kappa0 = eos._prsv_kappa0(params.omega)
                expected_kappa = (
                    kappa0
                    + params.kappa1 * taper * (1.0 + sqrt_Tr) * (0.7 - Tr)
                )
                expected_alpha = (1.0 + expected_kappa * (1.0 - sqrt_Tr)) ** 2

                self.assertAlmostEqual(eos._alpha(params, Tr * params.Tc), expected_alpha)

        self.assertEqual(eos._prsv1_taper(1.2), (0.0, 0.0))

    def test_prsv1_tapered_derivative_matches_finite_difference_at_boundaries(self):
        eos = CubicEOS(['CH4'], 'PRSV1', ChemicalDatabase(enable_online=False))
        params = eos.params['CH4']

        for T in (0.7 * params.Tc, 0.85 * params.Tc, params.Tc, 1.2 * params.Tc):
            with self.subTest(T=T):
                h = 1e-3
                finite_difference = (
                    eos.pure_a('CH4', T + h)
                    - eos.pure_a('CH4', T - h)
                ) / (2.0 * h)

                self.assertAlmostEqual(
                    eos.pure_da_dT('CH4', T),
                    finite_difference,
                    delta=max(abs(finite_difference) * 1e-6, 1e-8),
                )

    def test_prsv2_loads_table_parameters_without_prsv1_taper(self):
        eos = CubicEOS(['CH4'], 'PRSV2', ChemicalDatabase(enable_online=False))
        params = eos.params['CH4']
        Tr = 1.2
        sqrt_Tr = math.sqrt(Tr)

        self.assertAlmostEqual(params.kappa1, -0.00159)
        self.assertAlmostEqual(params.kappa2, 0.1521)
        self.assertAlmostEqual(params.kappa3, 0.517)

        kappa0 = eos._prsv_kappa0(params.omega)
        h = params.kappa1 + params.kappa2 * (params.kappa3 - Tr) * (1.0 - sqrt_Tr)
        expected_kappa = kappa0 + h * (1.0 + sqrt_Tr) * (0.7 - Tr)
        expected_alpha = (1.0 + expected_kappa * (1.0 - sqrt_Tr)) ** 2

        prsv1 = CubicEOS(['CH4'], 'PRSV1', ChemicalDatabase(enable_online=False))

        self.assertAlmostEqual(eos._alpha(params, Tr * params.Tc), expected_alpha)
        self.assertNotAlmostEqual(
            eos._alpha(params, Tr * params.Tc),
            prsv1._alpha(prsv1.params['CH4'], Tr * params.Tc),
        )

    def test_prsv_methods_improve_methanol_pure_eos_psat_at_80c(self):
        db = ChemicalDatabase(enable_online=False)
        if db.get('methanol', fetch_online=False) is None:
            self.skipTest('methanol is not available in the local database')

        comp = 'methanol'
        T = 80.0 + 273.15
        reference_psat = self._coolprop_methanol_psat_at_80c()

        def pure_eos_psat(model: str) -> float:
            eos = CubicEOS([comp], model, db)
            composition = {comp: 1.0}

            def residual(P_bar: float) -> float:
                phi_l = eos.fugacity_coefficients(T, P_bar, composition, 'liquid')[comp]
                phi_v = eos.fugacity_coefficients(T, P_bar, composition, 'vapor')[comp]
                return math.log(phi_l) - math.log(phi_v)

            return brentq(
                residual,
                0.2 * reference_psat,
                3.0 * reference_psat,
                xtol=1e-10,
                rtol=1e-10,
            )

        pr_psat = pure_eos_psat('PR')
        prsv1_psat = pure_eos_psat('PRSV1')
        prsv2_psat = pure_eos_psat('PRSV2')
        pr_error = abs(pr_psat - reference_psat) / reference_psat

        self.assertAlmostEqual(pr_psat, 1.8223444316, places=8)
        self.assertAlmostEqual(prsv1_psat, 1.8024153944, places=8)
        self.assertAlmostEqual(prsv2_psat, 1.8007036430, places=8)
        self.assertLess(abs(prsv1_psat - reference_psat) / reference_psat, pr_error)
        self.assertLess(abs(prsv2_psat - reference_psat) / reference_psat, pr_error)

    def test_twu_methods_load_family_tables_and_apply_twu_alpha(self):
        cases = [
            ('SRK-TWU', 0.217, 0.9082, 1.8172),
            ('PR-TWU', 0.1474, 0.9075, 1.8241),
        ]

        for model, twu_l, twu_m, twu_n in cases:
            with self.subTest(model=model):
                eos = CubicEOS(['CH4'], model, ChemicalDatabase(enable_online=False))
                params = eos.params['CH4']
                Tr = 0.8

                self.assertAlmostEqual(params.twu_l, twu_l)
                self.assertAlmostEqual(params.twu_m, twu_m)
                self.assertAlmostEqual(params.twu_n, twu_n)

                expected_alpha = Tr ** (twu_n * (twu_m - 1.0)) * math.exp(
                    twu_l * (1.0 - Tr ** (twu_n * twu_m))
                )
                self.assertAlmostEqual(eos._alpha(params, Tr * params.Tc), expected_alpha)

    def test_twu_derivative_matches_finite_difference(self):
        for model in ('SRK-TWU', 'PR-TWU'):
            eos = CubicEOS(['CH4'], model, ChemicalDatabase(enable_online=False))
            params = eos.params['CH4']
            composition = {'CH4': 1.0}

            for T in (0.8 * params.Tc, 1.2 * params.Tc):
                with self.subTest(model=model, T=T):
                    h = 1e-3
                    finite_difference = (
                        eos.mixture_params(T + h, composition)[0]
                        - eos.mixture_params(T - h, composition)[0]
                    ) / (2.0 * h)

                    self.assertAlmostEqual(
                        eos.mixture_da_dT(T, composition),
                        finite_difference,
                        delta=max(abs(finite_difference) * 1e-6, 1e-8),
                    )

    def test_twu_methods_methanol_pure_eos_psat_at_80c(self):
        db = ChemicalDatabase(enable_online=False)
        if db.get('methanol', fetch_online=False) is None:
            self.skipTest('methanol is not available in the local database')

        comp = 'methanol'
        T = 80.0 + 273.15
        reference_psat = self._coolprop_methanol_psat_at_80c()

        def pure_eos_psat(model: str) -> float:
            eos = CubicEOS([comp], model, db)
            composition = {comp: 1.0}

            def residual(P_bar: float) -> float:
                phi_l = eos.fugacity_coefficients(T, P_bar, composition, 'liquid')[comp]
                phi_v = eos.fugacity_coefficients(T, P_bar, composition, 'vapor')[comp]
                return math.log(phi_l) - math.log(phi_v)

            return brentq(
                residual,
                0.2 * reference_psat,
                3.0 * reference_psat,
                xtol=1e-10,
                rtol=1e-10,
            )

        srk_psat = pure_eos_psat('SRK')
        srk_twu_psat = pure_eos_psat('SRK-TWU')
        pr_psat = pure_eos_psat('PR')
        pr_twu_psat = pure_eos_psat('PR-TWU')

        self.assertAlmostEqual(srk_psat, 1.7851763036, places=8)
        self.assertAlmostEqual(srk_twu_psat, 1.8257897007, places=8)
        self.assertAlmostEqual(pr_psat, 1.8223444316, places=8)
        self.assertAlmostEqual(pr_twu_psat, 1.8113989473, places=8)
        self.assertLess(
            abs(pr_twu_psat - reference_psat) / reference_psat,
            abs(pr_psat - reference_psat) / reference_psat,
        )
        # With the current methanol critical inputs the generalized Twu
        # alpha now slightly improves on plain SRK as well (1.06% vs 1.19%
        # error); it previously degraded it.
        self.assertLess(
            abs(srk_twu_psat - reference_psat) / reference_psat,
            abs(srk_psat - reference_psat) / reference_psat,
        )

    def test_pfd_mc_parameters_override_table_values(self):
        pfd = (
            'PROCESS: MC Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: SRK-MC\n'
            '\n'
            'COMPONENTS:\n'
            '    CH4 | Methane | MW=16.043, mc_c1=0.8, mc_c2=-0.1, mc_c3=0.02\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 120 [K]\n'
            '    P = 5 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = CH4:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        sim.run()
        params = sim.thermo.cubic.params['CH4']

        self.assertAlmostEqual(params.mc_c1, 0.8)
        self.assertAlmostEqual(params.mc_c2, -0.1)
        self.assertAlmostEqual(params.mc_c3, 0.02)

    def test_pfd_mc_c1_c2_override_table_and_default_c3_to_zero(self):
        pfd = (
            'PROCESS: MC Partial Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: SRK-MC\n'
            '\n'
            'COMPONENTS:\n'
            '    methanol | Methanol | mc_c1=0.8, mc_c2=-0.1\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 350 [K]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = methanol:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        params = sim.thermo.cubic.params['methanol']

        self.assertTrue(result.converged)
        self.assertAlmostEqual(params.mc_c1, 0.8)
        self.assertAlmostEqual(params.mc_c2, -0.1)
        self.assertAlmostEqual(params.mc_c3, 0.0)
        self.assertFalse(any('Mathias-Copeman alpha parameters unavailable' in warning for warning in result.warnings))

    def test_pr_mc_accepts_user_mc_c1_without_fallback_warning(self):
        pfd = (
            'PROCESS: PR MC C1 Only\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: PR-MC\n'
            '\n'
            'COMPONENTS:\n'
            '    X | Imaginary | MW=40.0, Tc=400.0, Pc=40.0, omega=0.2, '
            'mc_c1=0.8, Cp_coeffs=[30.0,0.0,0.0,0.0]\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 450 [K]\n'
            '    P = 5 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = X:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        params = sim.thermo.cubic.params['X']

        self.assertTrue(result.converged)
        self.assertAlmostEqual(params.mc_c1, 0.8)
        self.assertAlmostEqual(params.mc_c2, 0.0)
        self.assertAlmostEqual(params.mc_c3, 0.0)
        self.assertFalse(any('Mathias-Copeman alpha parameters unavailable' in warning for warning in result.warnings))

    def test_mc_parameters_are_ignored_with_warning_for_non_mc_methods(self):
        pfd = (
            'PROCESS: Non-MC Ignores MC Parameters\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: SRK\n'
            '\n'
            'COMPONENTS:\n'
            '    CH4 | Methane | MW=16.043, mc_c1=0.8, mc_c2=-0.1\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 120 [K]\n'
            '    P = 5 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = CH4:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        report = sim._generate_pfr()
        params = sim.thermo.cubic.params['CH4']

        self.assertTrue(result.converged)
        self.assertIsNone(params.mc_c1)
        self.assertIsNone(params.mc_c2)
        self.assertIsNone(params.mc_c3)
        self.assertTrue(any('Ignoring Mathias-Copeman alpha parameters for CH4' in warning for warning in result.warnings))
        self.assertIn('Ignoring Mathias-Copeman alpha parameters for CH4', report)

    def test_pfd_prsv1_kappa1_overrides_table_and_ignores_prsv2_parameters(self):
        pfd = (
            'PROCESS: PRSV1 Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PRSV1\n'
            '\n'
            'COMPONENTS:\n'
            '    methanol | Methanol | kappa1=0.4, kappa2=2.0, kappa3=3.0\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 350 [K]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = methanol:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        report = sim._generate_pfr()
        params = sim.thermo.cubic.params['methanol']

        self.assertTrue(result.converged)
        self.assertAlmostEqual(params.kappa1, 0.4)
        self.assertAlmostEqual(params.kappa2, 0.0)
        self.assertAlmostEqual(params.kappa3, 0.0)
        self.assertTrue(any('Ignoring PRSV2 alpha parameters for methanol' in warning for warning in result.warnings))
        self.assertIn('Ignoring PRSV2 alpha parameters for methanol', report)
        self.assertFalse(any('PRSV alpha parameters unavailable' in warning for warning in result.warnings))

    def test_pfd_prsv2_accepts_kappa1_only_and_overrides_table(self):
        pfd = (
            'PROCESS: PRSV2 Kappa1 Only\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PRSV2\n'
            '\n'
            'COMPONENTS:\n'
            '    methanol | Methanol | kappa1=0.4\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 350 [K]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = methanol:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        params = sim.thermo.cubic.params['methanol']

        self.assertTrue(result.converged)
        self.assertAlmostEqual(params.kappa1, 0.4)
        self.assertAlmostEqual(params.kappa2, 0.0)
        self.assertAlmostEqual(params.kappa3, 0.0)
        self.assertFalse(any('PRSV alpha parameters unavailable' in warning for warning in result.warnings))

    def test_prsv_parameters_are_ignored_with_warning_for_non_prsv_methods(self):
        pfd = (
            'PROCESS: Non-PRSV Ignores Kappa Parameters\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PR\n'
            '\n'
            'COMPONENTS:\n'
            '    CH4 | Methane | MW=16.043, kappa1=0.4, kappa2=2.0\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 120 [K]\n'
            '    P = 5 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = CH4:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        report = sim._generate_pfr()
        params = sim.thermo.cubic.params['CH4']

        self.assertTrue(result.converged)
        self.assertIsNone(params.kappa1)
        self.assertIsNone(params.kappa2)
        self.assertIsNone(params.kappa3)
        self.assertTrue(any('Ignoring PRSV alpha parameters for CH4' in warning for warning in result.warnings))
        self.assertIn('Ignoring PRSV alpha parameters for CH4', report)

    def test_prsv_missing_parameters_use_kappa0_only_with_pfr_warning(self):
        for method in ('PRSV1', 'PRSV2'):
            with self.subTest(method=method):
                pfd = (
                    'PROCESS: PRSV Fallback\n'
                    'VERSION: 1.0\n'
                    'ONLINE_LOOKUP: false\n'
                    f'THERMO_METHOD: {method}\n'
                    '\n'
                    'COMPONENTS:\n'
                    '    X | Imaginary | MW=40.0, Tc=400.0, Pc=40.0, omega=0.2, '
                    'Cp_coeffs=[30.0,0.0,0.0,0.0]\n'
                    '\n'
                    'STREAM Feed : FEED -> PRODUCT\n'
                    '    T = 450 [K]\n'
                    '    P = 5 [bar]\n'
                    '    F = 1 [kmol/h]\n'
                    '    x = X:1.0\n'
                )

                sim = Simulator.from_string(pfd)
                result = sim.run()
                report = sim._generate_pfr()
                prsv_params = sim.thermo.cubic.params['X']
                pr = CubicEOS(['X'], 'PR', sim.thermo.db)
                pr_params = pr.params['X']
                Tr = 450.0 / prsv_params.Tc
                kappa0 = sim.thermo.cubic._prsv_kappa0(prsv_params.omega)
                expected_kappa0_alpha = (1.0 + kappa0 * (1.0 - math.sqrt(Tr))) ** 2

                self.assertTrue(result.converged)
                self.assertAlmostEqual(prsv_params.kappa1, 0.0)
                self.assertAlmostEqual(prsv_params.kappa2, 0.0)
                self.assertAlmostEqual(prsv_params.kappa3, 0.0)
                self.assertAlmostEqual(
                    sim.thermo.cubic._alpha(prsv_params, 450.0),
                    expected_kappa0_alpha,
                )
                self.assertNotAlmostEqual(
                    sim.thermo.cubic._alpha(prsv_params, 450.0),
                    pr._alpha(pr_params, 450.0),
                )
                self.assertTrue(any('PRSV alpha parameters unavailable' in warning for warning in result.warnings))
                self.assertTrue(any('kappa0-only alpha' in warning for warning in result.warnings))
                self.assertIn('PRSV alpha parameters unavailable', report)

    def test_pfd_twu_parameters_override_table_values_and_ignore_vt_c(self):
        pfd = (
            'PROCESS: Twu Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PR-TWU\n'
            '\n'
            'COMPONENTS:\n'
            '    methanol | Methanol | twu_l=0.8, twu_m=0.9, twu_n=1.7, twu_c=1e-5\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 350 [K]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = methanol:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        report = sim._generate_pfr()
        params = sim.thermo.cubic.params['methanol']
        props = sim.thermo.cubic.props['methanol']

        self.assertTrue(result.converged)
        self.assertAlmostEqual(params.twu_l, 0.8)
        self.assertAlmostEqual(params.twu_m, 0.9)
        self.assertAlmostEqual(params.twu_n, 1.7)
        self.assertIsNone(props.twu_c)
        self.assertTrue(any('Ignoring Twu volume-translation parameters for methanol' in warning for warning in result.warnings))
        self.assertIn('Ignoring Twu volume-translation parameters for methanol', report)
        self.assertFalse(any('Twu alpha parameters unavailable' in warning for warning in result.warnings))

    def test_pfd_incomplete_twu_parameters_do_not_mix_with_table_values(self):
        pfd = (
            'PROCESS: Twu Partial Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PR-TWU\n'
            '\n'
            'COMPONENTS:\n'
            '    methanol | Methanol | twu_l=0.8, twu_m=0.9\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 350 [K]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = methanol:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        report = sim._generate_pfr()
        params = sim.thermo.cubic.params['methanol']
        pr = CubicEOS(['methanol'], 'PR', sim.thermo.db)

        self.assertTrue(result.converged)
        self.assertIsNone(params.twu_l)
        self.assertIsNone(params.twu_m)
        self.assertIsNone(params.twu_n)
        self.assertAlmostEqual(
            sim.thermo.cubic._alpha(params, 350.0),
            pr._alpha(pr.params['methanol'], 350.0),
        )
        self.assertTrue(any('Twu alpha parameters incomplete for methanol' in warning for warning in result.warnings))
        self.assertIn('Twu alpha parameters incomplete for methanol', report)

    def test_twu_parameters_are_ignored_with_warning_for_non_twu_methods(self):
        pfd = (
            'PROCESS: Non-Twu Ignores Twu Parameters\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PR\n'
            '\n'
            'COMPONENTS:\n'
            '    CH4 | Methane | MW=16.043, twu_l=0.8, twu_m=0.9, twu_n=1.7, twu_c=1e-5\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 120 [K]\n'
            '    P = 5 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = CH4:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        report = sim._generate_pfr()
        params = sim.thermo.cubic.params['CH4']

        self.assertTrue(result.converged)
        self.assertIsNone(params.twu_l)
        self.assertIsNone(params.twu_m)
        self.assertIsNone(params.twu_n)
        self.assertTrue(any('Ignoring Twu alpha parameters for CH4' in warning for warning in result.warnings))
        self.assertTrue(any('Ignoring Twu volume-translation parameters for CH4' in warning for warning in result.warnings))
        self.assertIn('Ignoring Twu alpha parameters for CH4', report)
        self.assertIn('Ignoring Twu volume-translation parameters for CH4', report)

    def test_twu_missing_parameters_fall_back_to_base_alpha_with_pfr_warning(self):
        for method, parent in (('SRK-TWU', 'SRK'), ('PR-TWU', 'PR')):
            with self.subTest(method=method):
                pfd = (
                    'PROCESS: Twu Fallback\n'
                    'VERSION: 1.0\n'
                    'ONLINE_LOOKUP: false\n'
                    f'THERMO_METHOD: {method}\n'
                    '\n'
                    'COMPONENTS:\n'
                    '    X | Imaginary | MW=40.0, Tc=400.0, Pc=40.0, omega=0.2, '
                    'Cp_coeffs=[30.0,0.0,0.0,0.0]\n'
                    '\n'
                    'STREAM Feed : FEED -> PRODUCT\n'
                    '    T = 450 [K]\n'
                    '    P = 5 [bar]\n'
                    '    F = 1 [kmol/h]\n'
                    '    x = X:1.0\n'
                )

                sim = Simulator.from_string(pfd)
                result = sim.run()
                report = sim._generate_pfr()
                twu_params = sim.thermo.cubic.params['X']
                base = CubicEOS(['X'], parent, sim.thermo.db)

                self.assertTrue(result.converged)
                self.assertIsNone(twu_params.twu_l)
                self.assertAlmostEqual(
                    sim.thermo.cubic._alpha(twu_params, 450.0),
                    base._alpha(base.params['X'], 450.0),
                )
                self.assertTrue(any('Twu alpha parameters unavailable for X' in warning for warning in result.warnings))
                self.assertIn('Twu alpha parameters unavailable for X', report)

    def test_pr_mc_missing_parameters_falls_back_to_parent_bm_with_pfr_warning(self):
        pfd = (
            'PROCESS: PR MC Fallback\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: PR-MC\n'
            '\n'
            'COMPONENTS:\n'
            '    X | Imaginary | MW=40.0, Tc=400.0, Pc=40.0, omega=0.2, '
            'Cp_coeffs=[30.0,0.0,0.0,0.0]\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 450 [K]\n'
            '    P = 5 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = X:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        report = sim._generate_pfr()
        mc_params = sim.thermo.cubic.params['X']
        bm = CubicEOS(['X'], 'PR-BM', sim.thermo.db)
        bm_params = bm.params['X']

        self.assertTrue(result.converged)
        self.assertIsNone(mc_params.mc_c1)
        self.assertAlmostEqual(
            sim.thermo.cubic._alpha(mc_params, 450.0),
            bm._alpha(bm_params, 450.0),
        )
        self.assertTrue(any('Mathias-Copeman alpha parameters unavailable' in warning for warning in result.warnings))
        self.assertIn('Mathias-Copeman alpha parameters unavailable', report)

    def test_nrtl_rk_flash_iterates_composition_dependent_k_values_near_dew(self):
        mw_ethanol = 46.06844
        mw_water = 18.01528
        n_ethanol = 0.5 / mw_ethanol
        n_water = 0.5 / mw_water
        composition = {
            'ethanol': n_ethanol / (n_ethanol + n_water),
            'water': n_water / (n_ethanol + n_water),
        }
        thermo = create_thermodynamics(['ethanol', 'water'], 'NRTL-RK')
        T = 89.7136 + 273.15
        P = 1.01325
        T_bubble = thermo.bubble_point_T(composition, P, T)
        T_dew = thermo.dew_point_T(composition, P, T)

        state = thermo.calculate_state(T, P, 100.0, composition)

        self.assertGreater(T, min(T_bubble, T_dew))
        self.assertLess(T, max(T_bubble, T_dew))
        self.assertGreater(state.vapor_fraction, 0.0)
        self.assertLess(state.vapor_fraction, 1.0)
        self.assertNotAlmostEqual(state.x['ethanol'], state.y['ethanol'], places=3)

    def test_eos_dew_point_iterates_composition_dependent_k_values(self):
        composition = {'CH4': 0.5, 'C2H6': 0.5}
        P = 20.0
        for method in ('RK', 'PR'):
            with self.subTest(method=method):
                thermo = create_thermodynamics(['CH4', 'C2H6'], method)
                T_bubble = thermo.bubble_point_T(composition, P, 300.0)
                T_dew = thermo.dew_point_T(composition, P, T_bubble)
                T_mid = 0.5 * (T_bubble + T_dew)

                state = thermo.calculate_state(T_mid, P, 100.0, composition)

                self.assertGreater(T_dew, T_bubble)
                self.assertGreater(state.vapor_fraction, 0.0)
                self.assertLess(state.vapor_fraction, 1.0)
                self.assertNotAlmostEqual(state.x['CH4'], state.y['CH4'], places=3)

    def test_vdm_activity_methods_correct_acetic_acid_vapor_fugacity(self):
        composition = {'CH3COOH': 0.05, 'H2O': 0.95}
        for base_method, vdm_method in [
            ('UNIFAC', 'UNIFAC-VDM'),
            ('UNIFDMD', 'UNIFDMD-VDM'),
            ('UNIFNIST', 'UNIFNIST-VDM'),
            ('NRTL', 'NRTL-VDM'),
            ('UNIQUAC', 'UNIQUAC-VDM'),
        ]:
            with self.subTest(method=vdm_method):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    base = create_thermodynamics(['CH3COOH', 'H2O'], base_method)
                    vdm = create_thermodynamics(['CH3COOH', 'H2O'], vdm_method)

                T = vdm.bubble_point_T(composition, 1.01325)
                K_base = base.K_values(T, 1.01325, composition)
                K_vdm = vdm.K_values(T, 1.01325, composition)
                y = {
                    comp: composition[comp] * K_vdm[comp]
                    for comp in composition
                }
                y_total = sum(y.values())
                y = {comp: value / y_total for comp, value in y.items()}
                phi = vdm.fugacity_coefficients(T, 1.01325, y)

                self.assertGreater(T, 369.0)
                self.assertLess(T, 374.0)
                self.assertLess(K_vdm['CH3COOH'], K_base['CH3COOH'])
                self.assertLess(phi['CH3COOH'], 1.0)
                self.assertAlmostEqual(
                    sum(composition[comp] * K_vdm[comp] for comp in composition),
                    1.0,
                    delta=1e-6,
                )

        self.assertEqual(
            parse_pfd(
                'PROCESS: NRTL VDM Alias\n'
                'THERMO_METHOD: NRTL_VDM\n'
                'COMPONENTS:\n'
                '    CH3COOH | Acetic Acid | MW=60.052\n'
                '    H2O | Water | MW=18.015\n'
            ).metadata.thermo_method,
            'NRTL-VDM',
        )
        self.assertEqual(
            parse_pfd(
                'PROCESS: VDM Alias\n'
                'THERMO_METHOD: UNIFAC_VDM\n'
                'COMPONENTS:\n'
                '    CH3COOH | Acetic Acid | MW=60.052\n'
                '    H2O | Water | MW=18.015\n'
            ).metadata.thermo_method,
            'UNIFAC-VDM',
        )
        self.assertEqual(
            parse_pfd(
                'PROCESS: Dortmund VDM Alias\n'
                'THERMO_METHOD: DORTMUND-UNIFAC-VDM\n'
                'COMPONENTS:\n'
                '    CH3COOH | Acetic Acid | MW=60.052\n'
                '    H2O | Water | MW=18.015\n'
            ).metadata.thermo_method,
            'UNIFDMD-VDM',
        )
        self.assertEqual(
            parse_pfd(
                'PROCESS: NIST VDM Alias\n'
                'THERMO_METHOD: NIST-UNIFAC-VDM\n'
                'COMPONENTS:\n'
                '    CH3COOH | Acetic Acid | MW=60.052\n'
                '    H2O | Water | MW=18.015\n'
            ).metadata.thermo_method,
            'UNIFNIST-VDM',
        )

    def test_uniquac_vdm_matches_acetic_acid_water_txy_without_azeotrope(self):
        experimental = [
            (116.5, 0.022, 0.058),
            (114.6, 0.054, 0.123),
            (113.4, 0.086, 0.168),
            (113.5, 0.099, 0.183),
            (113.1, 0.101, 0.188),
            (110.6, 0.189, 0.298),
            (107.8, 0.303, 0.433),
            (106.1, 0.413, 0.545),
            (104.4, 0.522, 0.649),
            (103.1, 0.624, 0.735),
            (102.3, 0.696, 0.792),
            (101.6, 0.778, 0.851),
            (100.8, 0.876, 0.914),
            (100.5, 0.923, 0.944),
            (100.4, 0.945, 0.960),
            (100.1, 0.985, 0.989),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')

        t_errors = []
        for T_exp_C, x_water, _y_water in experimental:
            composition = {'CH3COOH': 1.0 - x_water, 'H2O': x_water}
            T_calc = thermo.bubble_point_T(composition, 1.01325)
            t_errors.append(T_calc - 273.15 - T_exp_C)

        t_rms = math.sqrt(sum(error * error for error in t_errors) / len(t_errors))
        self.assertLess(t_rms, 0.4)

        previous = None
        min_diff = float('inf')
        for index in range(1, 1000):
            x_water = index / 1000.0
            composition = {'CH3COOH': 1.0 - x_water, 'H2O': x_water}
            T_calc = thermo.bubble_point_T(composition, 1.01325)
            K = thermo.K_values(T_calc, 1.01325, composition)
            y_water = composition['H2O'] * K['H2O'] / sum(
                composition[comp] * K[comp]
                for comp in composition
            )
            diff = y_water - x_water
            min_diff = min(min_diff, diff)
            if previous is not None:
                self.assertGreaterEqual(previous * diff, 0.0)
            previous = diff
        self.assertGreater(min_diff, 0.0)

    def test_unifdmd_uses_dortmund_parameters_and_parser_aliases(self):
        classic = create_thermodynamics(['ethanol', 'water'], 'UNIFAC')
        dortmund = create_thermodynamics(['ethanol', 'water'], 'UNIFDMD')
        composition = {'ethanol': 0.4, 'water': 0.6}

        self.assertEqual(dortmund.unifac.variant, 'UNIFDMD')
        self.assertAlmostEqual(dortmund.r['ethanol'], 2.4952, places=4)
        self.assertNotAlmostEqual(dortmund.r['ethanol'], classic.r['ethanol'], places=3)

        gamma = dortmund.activity_coefficients(298.15, composition)
        self.assertGreater(gamma['ethanol'], 1.0)
        self.assertGreater(gamma['water'], 1.0)
        self.assertNotAlmostEqual(
            gamma['ethanol'],
            classic.activity_coefficients(298.15, composition)['ethanol'],
            places=3,
        )

        pfd = parse_pfd(
            'PROCESS: Alias Test\n'
            'THERMO_METHOD: DORTMUND-UNIFAC\n'
            'COMPONENTS:\n'
            '    ethanol | Ethanol | MW=46.07\n'
            '    water | Water | MW=18.015\n'
        )
        self.assertEqual(pfd.metadata.thermo_method, 'UNIFDMD')

    def test_unifnist_loads_nist_modified_parameters(self):
        data_path = os.path.join(ROOT, 'data', 'nist_modified_unifac_params.json')
        model = UNIFACModel(data_path)

        self.assertEqual(model.variant, 'UNIFNIST')
        self.assertAlmostEqual(model.subgroups[1].R, 0.6325, places=4)
        self.assertAlmostEqual(model.subgroups[16].Q, 2.4561, places=4)
        self.assertAlmostEqual(model.subgroups[42].Q, 0.9215, places=4)
        self.assertEqual(model.subgroups[42].main_group, 20)
        self.assertEqual(model.subgroup_by_name['HCOOH'].main_group, 44)

        self.assertEqual(
            model.interaction_coefficients[(7, 20)],
            (82.33, 1.0692, -0.0015475),
        )
        self.assertEqual(
            model.interaction_coefficients[(20, 7)],
            (195.87, -1.9941, 0.0024708),
        )

        thermo = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIFNIST')
        self.assertEqual(thermo.unifac.variant, 'UNIFNIST')
        self.assertIsNotNone(thermo._compiled_unifac)
        if thermo._compiled_unifac is not None:
            self.assertEqual(thermo._compiled_unifac.variant_id, 1)

    def test_unifac2_loads_completed_classic_interaction_matrix(self):
        classic = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_params.json'))
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIFAC2')
        model = thermo.unifac

        self.assertEqual(model.variant, 'UNIFAC2')
        self.assertEqual(len(model.subgroups), 113)
        self.assertEqual(model.calculate_r_q({'CH3': 1, 'CH2': 1, 'OH': 1}),
                         classic.calculate_r_q({'CH3': 1, 'CH2': 1, 'OH': 1}))
        self.assertAlmostEqual(model.interactions[(1, 2)], 28.47801, places=6)
        self.assertAlmostEqual(model.interactions[(2, 1)], 16.999664, places=6)

        main_groups = {subgroup.main_group for subgroup in model.subgroups.values()}
        self.assertTrue(all(
            (main_i, main_j) in model.interactions
            for main_i in main_groups
            for main_j in main_groups
        ))
        self.assertFalse(any('interaction parameters missing' in warning
                             for warning in thermo.warnings))
        self.assertIsNotNone(thermo._compiled_unifac)
        self.assertEqual(thermo._compiled_unifac.variant_id, 0)
        self.assertEqual(
            model._resolve_groups(get_unifac_groups('cyclohexane', variant='UNIFAC2')),
            {2: 6},
        )

    def test_unifm2_loads_completed_dortmund_interaction_matrices(self):
        dortmund = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_dmd.txt'))
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIFM2')
        model = thermo.unifac

        self.assertEqual(model.variant, 'UNIFM2')
        self.assertEqual(len(model.subgroups), 125)
        self.assertEqual(model.calculate_r_q({1: 1, 2: 1, 14: 1}),
                         dortmund.calculate_r_q({1: 1, 2: 1, 14: 1}))
        self.assertAlmostEqual(model.interaction_coefficients[(1, 2)][0],
                               63.270508, places=6)
        self.assertAlmostEqual(model.interaction_coefficients[(1, 2)][1],
                               -0.086422, places=6)
        self.assertEqual(model.interaction_coefficients[(1, 2)][2], 0.0)

        main_groups = {subgroup.main_group for subgroup in model.subgroups.values()}
        self.assertTrue(all(
            (main_i, main_j) in model.interaction_coefficients
            for main_i in main_groups
            for main_j in main_groups
        ))
        self.assertFalse(any('interaction parameters missing' in warning
                             for warning in thermo.warnings))
        self.assertIsNotNone(thermo._compiled_unifac)
        self.assertEqual(thermo._compiled_unifac.variant_id, 1)
        self.assertEqual(
            model._resolve_groups(get_unifac_groups('cyclohexane', variant='UNIFM2')),
            {78: 6},
        )

    def test_unifac2_and_unifm2_are_supported_pfd_methods(self):
        for method in ('UNIFAC2', 'UNIFM2'):
            with self.subTest(method=method):
                pfd = parse_pfd(
                    'PROCESS: Completed UNIFAC matrix\n'
                    f'THERMO_METHOD: {method}\n'
                    'COMPONENTS:\n'
                    '    ethanol | Ethanol | MW=46.07\n'
                    '    water | Water | MW=18.015\n'
                )
                self.assertEqual(pfd.metadata.thermo_method, method)

    def test_activity_model_stream_cp_uses_phase_and_consistent_units(self):
        thermo = create_thermodynamics(['water', 'ethanol'], 'UNIFAC')
        composition = {'water': 0.5, 'ethanol': 0.5}
        T = 298.15
        P = 1.01325

        liquid = thermo.calculate_state(T, P, 1.0, composition, phase='liquid', flash=False)
        vapor = thermo.calculate_state(T, P, 1.0, composition, phase='vapor', flash=False)

        self.assertAlmostEqual(liquid.Cp, thermo.mixture_Cp(liquid.composition, T, 0.0))
        self.assertAlmostEqual(vapor.Cp, thermo.mixture_Cp(vapor.composition, T, 1.0))
        self.assertGreater(liquid.Cp, vapor.Cp)
        # kJ/kmol-K basis: catches both a mol-basis (/1000) and a J-basis
        # (*1000) regression for water/ethanol (~90 liquid, ~50 vapor).
        self.assertGreater(vapor.Cp, 20.0)
        self.assertLess(liquid.Cp, 200.0)
        self.assertLess(vapor.Cp, 200.0)

        x = {'water': 0.9, 'ethanol': 0.1}
        y = {'water': 0.2, 'ethanol': 0.8}
        thermo.flash = lambda _T, _P, _z: (0.25, x, y)

        two_phase = thermo.calculate_state(T, P, 1.0, composition)
        expected = (
            0.75 * thermo.mixture_Cp(x, T, 0.0)
            + 0.25 * thermo.mixture_Cp(y, T, 1.0)
        )
        self.assertAlmostEqual(two_phase.Cp, expected)

    def test_unifnist_vdm_predicts_formic_acid_water_azeotrope(self):
        thermo = create_thermodynamics(['HCOOH', 'H2O'], 'UNIFNIST-VDM')

        def y_minus_x_water(x_water):
            composition = {'HCOOH': 1.0 - x_water, 'H2O': x_water}
            T = thermo.bubble_point_T(composition, 1.01325, 380.0)
            K = thermo.K_values(T, 1.01325, composition)
            y_water = composition['H2O'] * K['H2O'] / sum(
                composition[comp] * K[comp]
                for comp in composition
            )
            return y_water - x_water

        x_water = brentq(y_minus_x_water, 0.38, 0.46)
        composition = {'HCOOH': 1.0 - x_water, 'H2O': x_water}
        T = thermo.bubble_point_T(composition, 1.01325, 380.0)

        mw_acid = thermo.props['HCOOH'].MW
        mw_water = thermo.props['H2O'].MW
        wt_acid = (
            composition['HCOOH'] * mw_acid
            / (composition['HCOOH'] * mw_acid + composition['H2O'] * mw_water)
        )

        self.assertAlmostEqual(T - 273.15, 107.3, delta=0.2)
        self.assertAlmostEqual(100.0 * wt_acid, 77.5, delta=0.3)

    def test_uniquac_matches_chemsep_ethanol_water_reference(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'UNIQUAC')
        gamma = thermo.activity_coefficients(343.15, {'ethanol': 0.252, 'water': 0.748})

        self.assertAlmostEqual(gamma['ethanol'], 1.977454, places=5)
        self.assertAlmostEqual(gamma['water'], 1.1397696, places=5)
        self.assertEqual(
            parse_pfd(
                'PROCESS: UNIQUAC Alias\n'
                'THERMO_METHOD: UNIQUAC_RK\n'
                'COMPONENTS:\n'
                '    ethanol | Ethanol | MW=46.07\n'
                '    water | Water | MW=18.015\n'
            ).metadata.thermo_method,
            'UNIQUAC-RK',
        )
        self.assertEqual(
            parse_pfd(
                'PROCESS: UNIQUAC Alias\n'
                'THERMO_METHOD: UNIQUAC_PENG_ROBINSON\n'
                'COMPONENTS:\n'
                '    ethanol | Ethanol | MW=46.07\n'
                '    water | Water | MW=18.015\n'
            ).metadata.thermo_method,
            'UNIQUAC-PR',
        )

    def test_uniquac_prefers_first_chemsep_duplicate_and_uses_direct_rq_when_available(self):
        interaction = uniquac_binary_interaction('67-56-1', '7732-18-5')
        self.assertIsNotNone(interaction)
        self.assertAlmostEqual(interaction['a12_cal_per_mol'], -337.1298, places=4)
        self.assertAlmostEqual(interaction['a21_cal_per_mol'], 549.2958, places=4)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            thermo = create_thermodynamics(['1-propanol', 'water'], 'UNIQUAC')

        self.assertAlmostEqual(thermo.r['1-propanol'], 2.7800, places=8)
        self.assertAlmostEqual(thermo.q['1-propanol'], 2.5130, places=8)
        self.assertFalse(any('estimated from UNIFAC' in warning for warning in thermo.warnings))
        self.assertFalse(
            any('estimated from UNIFAC' in str(warning.message) for warning in caught)
        )

    def test_fitted_acetic_acid_nrtl_and_uniquac_pairs_are_available(self):
        built_nrtl, nrtl_new_pairs = supplemental_acetic_acid_vle_records(
            [], 'NRTL'
        )
        built_uniquac, uniquac_new_pairs = supplemental_acetic_acid_vle_records(
            [], 'UNIQUAC'
        )
        self.assertEqual(nrtl_new_pairs, 2)
        self.assertEqual(uniquac_new_pairs, 1)
        self.assertEqual(
            {(record['cas1'], record['cas2']) for record in built_nrtl},
            {('64-19-7', '7732-18-5'), ('64-19-7', '141-78-6')},
        )
        self.assertEqual(
            {(record['cas1'], record['cas2']) for record in built_uniquac},
            {('64-19-7', '141-78-6')},
        )

        acid_water = nrtl_binary_interaction('64-19-7', '7732-18-5')
        water_acid = nrtl_binary_interaction('7732-18-5', '64-19-7')
        self.assertAlmostEqual(
            acid_water['a12_cal_per_mol'], -6.1320111905598615, places=12
        )
        self.assertAlmostEqual(
            acid_water['a21_cal_per_mol'], 590.4171250260812, places=12
        )
        self.assertAlmostEqual(acid_water['alpha12'], 0.3, places=12)
        self.assertAlmostEqual(
            water_acid['a12_cal_per_mol'], acid_water['a21_cal_per_mol'], places=12
        )
        self.assertAlmostEqual(
            water_acid['a21_cal_per_mol'], acid_water['a12_cal_per_mol'], places=12
        )
        self.assertIn('1952, 44(8), 1864-1872', acid_water['comment'])

        acid_ester_nrtl = nrtl_binary_interaction('64-19-7', '141-78-6')
        ester_acid_nrtl = nrtl_binary_interaction('141-78-6', '64-19-7')
        self.assertAlmostEqual(
            acid_ester_nrtl['a12_cal_per_mol'], -223.13760955083086, places=12
        )
        self.assertAlmostEqual(
            acid_ester_nrtl['a21_cal_per_mol'], 525.2700041929397, places=12
        )
        self.assertAlmostEqual(acid_ester_nrtl['alpha12'], 0.3, places=12)
        self.assertAlmostEqual(
            ester_acid_nrtl['a12_cal_per_mol'],
            acid_ester_nrtl['a21_cal_per_mol'],
            places=12,
        )
        self.assertIn('1937, 29(6), 709-710', acid_ester_nrtl['comment'])

        acid_ester_uniquac = uniquac_binary_interaction('64-19-7', '141-78-6')
        ester_acid_uniquac = uniquac_binary_interaction('141-78-6', '64-19-7')
        self.assertAlmostEqual(
            acid_ester_uniquac['a12_cal_per_mol'],
            -257.42432034225703,
            places=12,
        )
        self.assertAlmostEqual(
            acid_ester_uniquac['a21_cal_per_mol'],
            465.43722267062594,
            places=12,
        )
        self.assertAlmostEqual(
            ester_acid_uniquac['a12_cal_per_mol'],
            acid_ester_uniquac['a21_cal_per_mol'],
            places=12,
        )
        self.assertIn('1937, 29(6), 709-710', acid_ester_uniquac['comment'])

        legacy_water_uniquac = uniquac_binary_interaction('64-19-7', '7732-18-5')
        self.assertAlmostEqual(
            legacy_water_uniquac['a12_cal_per_mol'], 407.0073, places=4
        )
        self.assertAlmostEqual(
            legacy_water_uniquac['a21_cal_per_mol'], -251.6868, places=4
        )

        nrtl_water = create_thermodynamics(['acetic acid', 'water'], 'NRTL')
        nrtl_ester = create_thermodynamics(
            ['acetic acid', 'ethyl acetate'], 'NRTL'
        )
        self.assertFalse(any('missing' in warning for warning in nrtl_water.warnings))
        self.assertFalse(any('missing' in warning for warning in nrtl_ester.warnings))

    def test_uniquac_estimates_missing_rq_from_unifac(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            thermo = create_thermodynamics(['formaldehyde', 'water'], 'UNIQUAC')

        r_expected, q_expected = UNIFACModel().calculate_r_q(
            get_unifac_groups('formaldehyde')
        )
        self.assertAlmostEqual(thermo.r['formaldehyde'], r_expected, places=8)
        self.assertAlmostEqual(thermo.q['formaldehyde'], q_expected, places=8)
        self.assertTrue(any('estimated from UNIFAC' in warning for warning in thermo.warnings))
        self.assertFalse(
            any('estimated from UNIFAC' in str(warning.message) for warning in caught)
        )

    def test_uniquac_rq_fallback_uses_native_numeric_smiles_groups(self):
        db = ChemicalDatabase(enable_online=False)
        db.chemicals['NATIVE_ETHER'] = ChemicalProperties(
            symbol='NATIVE_ETHER',
            name='Native ether probe',
            formula=None,
            MW=102.18,
            Tb=340.0,
            smiles='CC(C)OC(C)C',
        )

        thermo = create_thermodynamics(
            ['NATIVE_ETHER', 'water'],
            'UNIQUAC',
            db,
        )
        expected_r, expected_q = UNIFACModel().calculate_r_q(
            {1: 4, 3: 1, 26: 1}
        )

        self.assertAlmostEqual(thermo.r['NATIVE_ETHER'], expected_r)
        self.assertAlmostEqual(thermo.q['NATIVE_ETHER'], expected_q)
        self.assertTrue(any(
            "UNIQUAC r/q parameters missing for 'NATIVE_ETHER'" in warning
            for warning in thermo.warnings
        ))

    def test_uniquac_supports_temperature_dependent_direct_tau_parameters(self):
        forward = uniquac_binary_interaction('7697-37-2', '7732-18-5')
        reverse = uniquac_binary_interaction('7732-18-5', '7697-37-2')

        self.assertAlmostEqual(forward['tau12_a'], 0.23936, places=6)
        self.assertAlmostEqual(forward['tau12_b'], -85.163, places=6)
        self.assertAlmostEqual(reverse['tau12_a'], -0.59832, places=6)
        self.assertAlmostEqual(reverse['tau12_b'], 626.111, places=6)

        thermo = create_thermodynamics(['HNO3', 'H2O'], 'UNIQUAC')
        tau = thermo._uniquac_tau_matrix(298.15)

        self.assertAlmostEqual(tau[0][1], math.exp(0.23936 - 85.163 / 298.15))
        self.assertAlmostEqual(tau[1][0], math.exp(-0.59832 + 626.111 / 298.15))

        rk_thermo = create_thermodynamics(['HNO3', 'H2O'], 'UNIQUAC-RK')
        self.assertEqual(rk_thermo.r['HNO3'], thermo.r['HNO3'])
        self.assertEqual(rk_thermo.q['HNO3'], thermo.q['HNO3'])

        pr_thermo = create_thermodynamics(['HNO3', 'H2O'], 'UNIQUAC-PR')
        self.assertEqual(pr_thermo.r['HNO3'], thermo.r['HNO3'])
        self.assertEqual(pr_thermo.q['HNO3'], thermo.q['HNO3'])

    def test_phenolic_temperature_interactions_use_paper_cij_form(self):
        T = 333.15
        tref = 273.15

        nrtl = nrtl_binary_interaction('108-88-3', '108-95-2')
        self.assertIn('phenolic temperature-dependent NRTL', nrtl['comment'])
        self.assertAlmostEqual(nrtl['alpha12'], 0.2)
        self.assertAlmostEqual(nrtl['tau12_c'], -4.3775, places=6)
        self.assertAlmostEqual(
            nrtl['tau12_d'],
            857.14 - tref * -4.3775,
            places=6,
        )
        self.assertAlmostEqual(nrtl['tau21_c'], 2.8430, places=6)
        self.assertAlmostEqual(
            nrtl['tau21_d'],
            -308.41 - tref * 2.8430,
            places=6,
        )

        nrtl_thermo = create_thermodynamics(['toluene', 'phenol'], 'NRTL')
        tau, alpha = nrtl_thermo._nrtl_matrices(T)
        self.assertAlmostEqual(
            tau[0][1],
            (857.14 + -4.3775 * (T - tref)) / T,
            places=8,
        )
        self.assertAlmostEqual(
            tau[1][0],
            (-308.41 + 2.8430 * (T - tref)) / T,
            places=8,
        )
        self.assertAlmostEqual(alpha[0][1], 0.2, places=8)
        self.assertAlmostEqual(alpha[1][0], 0.2, places=8)

        uniquac = uniquac_binary_interaction('108-88-3', '108-95-2')
        self.assertIn('phenolic temperature-dependent UNIQUAC', uniquac['comment'])
        self.assertAlmostEqual(uniquac['tau12_a'], 1.6642, places=6)
        self.assertAlmostEqual(
            uniquac['tau12_b'],
            -369.57 + tref * -1.6642,
            places=6,
        )
        self.assertAlmostEqual(uniquac['tau21_a'], -0.9573, places=6)
        self.assertAlmostEqual(
            uniquac['tau21_b'],
            146.32 + tref * 0.9573,
            places=6,
        )

        uniquac_thermo = create_thermodynamics(['toluene', 'phenol'], 'UNIQUAC')
        tau = uniquac_thermo._uniquac_tau_matrix(T)
        self.assertAlmostEqual(
            tau[0][1],
            math.exp(-(369.57 + -1.6642 * (T - tref)) / T),
            places=8,
        )
        self.assertAlmostEqual(
            tau[1][0],
            math.exp(-(-146.32 + 0.9573 * (T - tref)) / T),
            places=8,
        )

    def test_cesari_phenolic_nrtl_interactions_use_energy_over_rt_form(self):
        R = 8.31446261815324
        T = 323.15

        water_guaiacol = nrtl_binary_interaction('7732-18-5', '90-05-1')
        self.assertIn('Water/Guaiacol Cesari phenolic NRTL', water_guaiacol['comment'])
        self.assertAlmostEqual(water_guaiacol['alpha12'], 0.3)
        self.assertAlmostEqual(water_guaiacol['tau12_c'], 50.56 / R, places=8)
        self.assertAlmostEqual(water_guaiacol['tau12_d'], -2904.47 / R, places=8)
        self.assertAlmostEqual(water_guaiacol['tau21_c'], -41.82 / R, places=8)
        self.assertAlmostEqual(water_guaiacol['tau21_d'], 14631.14 / R, places=8)

        thermo = create_thermodynamics(['ethanol', 'phenol'], 'NRTL')
        tau, alpha = thermo._nrtl_matrices(T)
        self.assertAlmostEqual(
            tau[0][1],
            (-0.04 + 0.01 * T) / (R * T),
            places=8,
        )
        self.assertAlmostEqual(
            tau[1][0],
            (-10.99 + 5.47 * T) / (R * T),
            places=8,
        )
        self.assertAlmostEqual(alpha[0][1], 0.3, places=8)
        self.assertAlmostEqual(alpha[1][0], 0.3, places=8)

        ethanol_phenol = nrtl_binary_interaction('64-17-5', '108-95-2')
        self.assertIn('Ethanol/Phenol Cesari phenolic NRTL', ethanol_phenol['comment'])
        self.assertAlmostEqual(ethanol_phenol['tau12_c'], 0.01 / R, places=8)
        self.assertAlmostEqual(ethanol_phenol['tau21_c'], 5.47 / R, places=8)

        water_cresol = nrtl_binary_interaction('7732-18-5', '95-48-7')
        self.assertIn('Water/2-Cresol phenolic temperature-dependent NRTL', water_cresol['comment'])
        self.assertNotIn('Cesari', water_cresol['comment'])

    def test_diethyl_ether_water_broad_fit_preserves_35c_vlle_anchor(self):
        T = 308.15
        psat_ether = 103.264
        psat_water = 5.633
        x_ether = 0.011948

        nrtl = nrtl_binary_interaction('60-29-7', '7732-18-5')
        self.assertIn('curated water/organic NRTL regression', nrtl['comment'])
        self.assertAlmostEqual(nrtl['alpha12'], 0.3)
        self.assertAlmostEqual(nrtl['tau12_c'], 0.099330046, places=8)
        self.assertAlmostEqual(nrtl['tau12_d'], 635.321936037, places=8)
        self.assertAlmostEqual(nrtl['tau21_c'], 11.754188386, places=8)
        self.assertAlmostEqual(nrtl['tau21_d'], -2538.603239639, places=8)

        thermo = create_thermodynamics(['diethyl ether', 'water'], 'NRTL')
        gamma = thermo.activity_coefficients(
            T,
            {'diethyl ether': x_ether, 'water': 1.0 - x_ether},
        )
        pressure = (
            x_ether * gamma['diethyl ether'] * psat_ether
            + (1.0 - x_ether) * gamma['water'] * psat_water
        )
        y_ether = x_ether * gamma['diethyl ether'] * psat_ether / pressure
        self.assertAlmostEqual(pressure, 104.57, delta=0.1)
        self.assertAlmostEqual(y_ether, 0.9467, delta=0.001)

        uniquac = uniquac_binary_interaction('60-29-7', '7732-18-5')
        self.assertIn('curated water/organic UNIQUAC regression', uniquac['comment'])
        self.assertAlmostEqual(uniquac['tau12_a'], 11.059254211, places=8)
        self.assertAlmostEqual(uniquac['tau12_b'], -2688.791776972, places=8)
        self.assertAlmostEqual(uniquac['tau21_a'], -12.439879602, places=8)
        self.assertAlmostEqual(uniquac['tau21_b'], 2315.818669335, places=8)

    def test_1_butanol_water_temperature_interactions_override_legacy_records(self):
        nrtl = nrtl_binary_interaction('71-36-3', '7732-18-5')
        self.assertIn('temperature-dependent NRTL', nrtl['comment'])
        self.assertAlmostEqual(nrtl['tau12_c'], 3.07626601, places=8)
        self.assertAlmostEqual(nrtl['tau12_d'], -489.80683594, places=8)
        self.assertAlmostEqual(nrtl['tau12_e'], -60.05225941, places=8)
        self.assertAlmostEqual(nrtl['tau21_c'], 4.36302560, places=8)
        self.assertAlmostEqual(nrtl['tau21_d'], -241.22842207, places=8)
        self.assertAlmostEqual(nrtl['tau21_e'], -8.71391481, places=8)
        self.assertAlmostEqual(nrtl['tau_tref'], 298.15, places=8)
        self.assertAlmostEqual(nrtl['alpha12'], 0.45131325, places=8)

        uniquac = uniquac_binary_interaction('71-36-3', '7732-18-5')
        self.assertIn('temperature-dependent UNIQUAC', uniquac['comment'])
        self.assertAlmostEqual(uniquac['tau12_a'], -12.81485434, places=8)
        self.assertAlmostEqual(uniquac['tau12_b'], 1905.74507135, places=8)
        self.assertAlmostEqual(uniquac['tau12_c'], 0.02169653866, places=10)
        self.assertAlmostEqual(uniquac['tau21_a'], 8.68179405, places=8)
        self.assertAlmostEqual(uniquac['tau21_b'], -1368.13065752, places=8)
        self.assertAlmostEqual(uniquac['tau21_c'], -0.01678360690, places=10)

    def test_extended_uniquac_interactions_are_retained_but_disabled(self):
        with open(
            os.path.join(ROOT, 'data', 'uniquac_binary_interactions_cas.json'),
            encoding='utf-8',
        ) as handle:
            payload = json.load(handle)
        self.assertEqual(payload['metadata']['disabled_records'], 39)
        self.assertEqual(payload['metadata']['activity_curated_disabled_records'], 10)
        records = payload['interactions']
        extended = [
            record for record in records
            if record.get('comment', '').startswith('Ethanol/Benzene extended UNIQUAC')
        ]
        self.assertEqual(len(extended), 1)
        self.assertTrue(extended[0]['disabled'])
        self.assertEqual(extended[0]['model_variant'], 'extended_uniquac')
        self.assertTrue(extended[0]['use_q_prime'])
        self.assertAlmostEqual(extended[0]['tau12_a'], -2.5229, places=6)
        self.assertAlmostEqual(extended[0]['tau12_b'], 216.07, places=6)
        self.assertAlmostEqual(extended[0]['tau12_c'], -0.0041, places=6)

        interaction = uniquac_binary_interaction('64-17-5', '71-43-2')
        self.assertIsNotNone(interaction)
        self.assertEqual(interaction['model_variant'], 'standard_uniquac')
        self.assertFalse(interaction['use_q_prime'])
        self.assertNotIn('tau12_a', interaction)
        self.assertAlmostEqual(interaction['a12_cal_per_mol'], -127.9893, places=4)
        self.assertAlmostEqual(interaction['a21_cal_per_mol'], 744.8826, places=4)

        methanol_water = uniquac_binary_interaction('67-56-1', '7732-18-5')
        self.assertEqual(methanol_water['comment'], 'Methanol/Water (Kojima+Kato)')
        self.assertAlmostEqual(methanol_water['a12_cal_per_mol'], -337.1298, places=4)

        methanol_carbon_tet = uniquac_binary_interaction('67-56-1', '56-23-5')
        self.assertEqual(methanol_carbon_tet['comment'], 'Methanol/Tetrachloromethane p18 1/2c')
        self.assertAlmostEqual(methanol_carbon_tet['a12_cal_per_mol'], -95.2921, places=4)

        selected_uniquac = [
            ('7732-18-5', '78-93-3', '2-Butanone/Water p279 1/1a'),
            ('64-17-5', '78-93-3', 'Ethanol/2-Butanone p342 1/2a'),
            ('110-82-7', '67-56-1', 'Methanol/Cyclohexane p211 1/2c'),
            ('142-82-5', '67-56-1', 'Methanol/n-Heptane'),
            ('110-82-7', '64-17-5', 'Ethanol/CycloHexane p441 1/2a'),
            ('64-17-5', '67-64-1', 'Ethanol/Acetone p323 1/2a'),
            ('67-56-1', '64-17-5', 'Methanol/Ethanol p60 1/2c'),
        ]
        for cas1, cas2, comment in selected_uniquac:
            with self.subTest(model='UNIQUAC', pair=(cas1, cas2)):
                self.assertEqual(uniquac_binary_interaction(cas1, cas2)['comment'], comment)

        thermo = create_thermodynamics(['ethanol', 'benzene'], 'UNIQUAC')
        params = thermo._uniquac_parameter_matrices()
        self.assertAlmostEqual(params['q_residual'][0], thermo.q['ethanol'], places=8)
        self.assertAlmostEqual(params['q_residual'][1], thermo.q['benzene'], places=8)

        def y_minus_x(x_ethanol):
            composition = {'ethanol': x_ethanol, 'benzene': 1.0 - x_ethanol}
            T = thermo.bubble_point_T(composition, 1.01325)
            K = thermo.K_values(T, 1.01325, composition)
            y_ethanol = x_ethanol * K['ethanol']
            y_benzene = (1.0 - x_ethanol) * K['benzene']
            return y_ethanol / (y_ethanol + y_benzene) - x_ethanol

        x_azeotrope = brentq(y_minus_x, 0.3, 0.6)
        composition = {'ethanol': x_azeotrope, 'benzene': 1.0 - x_azeotrope}
        T_azeotrope = thermo.bubble_point_T(composition, 1.01325)
        mw_ethanol = thermo.props['ethanol'].MW
        mw_benzene = thermo.props['benzene'].MW
        w_ethanol = (
            x_azeotrope * mw_ethanol
            / (x_azeotrope * mw_ethanol + (1.0 - x_azeotrope) * mw_benzene)
        )

        self.assertAlmostEqual(w_ethanol, 0.324, delta=0.01)
        self.assertAlmostEqual(T_azeotrope - 273.15, 68.2, delta=0.5)

    def test_nrtl_prefers_temperature_dependent_tau_duplicate(self):
        forward = nrtl_binary_interaction('67-64-1', '67-56-1')
        reverse = nrtl_binary_interaction('67-56-1', '67-64-1')

        self.assertNotIn('a12_cal_per_mol', forward)
        self.assertAlmostEqual(forward['tau12_c'], 0.0, places=6)
        self.assertAlmostEqual(forward['tau12_d'], 101.9, places=6)
        self.assertAlmostEqual(forward['tau21_c'], 0.0, places=6)
        self.assertAlmostEqual(forward['tau21_d'], 114.1, places=6)
        self.assertAlmostEqual(reverse['tau12_d'], 114.1, places=6)
        self.assertAlmostEqual(reverse['tau21_d'], 101.9, places=6)

    def test_nrtl_disabled_records_do_not_override_better_legacy_data(self):
        with open(
            os.path.join(ROOT, 'data', 'nrtl_binary_interactions_cas.json'),
            encoding='utf-8',
        ) as handle:
            payload = json.load(handle)
        self.assertEqual(payload['metadata']['disabled_records'], 15)
        self.assertEqual(payload['metadata']['activity_curated_disabled_records'], 9)

        disabled = [
            record for record in payload['interactions']
            if record.get('disabled')
        ]
        self.assertEqual(
            {record['comment'] for record in disabled},
            {
                '1 Chloroform/Benzene p72 1/7',
                'Acetone/Ethanol p312 1/2c',
                'Benzene/Chloroform supplemental NRTL matrix; tau_ij = a_ij + b_ij/T',
                'Benzene/Water supplemental NRTL matrix; tau_ij = a_ij + b_ij/T',
                'Ethanol/2-Butanone p327 1/2c',
                'Ethanol/Cyclohexane p419 1/2c',
                'Methanol/1-Heptane p241 1/2c',
                'Methanol/CycloHexane p243 1/2a',
                'Methanol/Ethanol p55 1/2a',
                'Methanol/Heptane p243 1/2c',
                'p-Xylene/Chloroform supplemental NRTL matrix; tau_ij = a_ij + b_ij/T',
                'p-Xylene/Water supplemental NRTL matrix; tau_ij = a_ij + b_ij/T',
                'Tetrachloromethane/Methanol p279 1/2c',
                'Water/2-Butanone p277 1/1a',
                'Toluene/Water supplemental NRTL matrix; tau_ij = a_ij + b_ij/T',
            },
        )

        interaction = nrtl_binary_interaction('71-43-2', '67-66-3')
        self.assertEqual(interaction['comment'], 'Benzene/Chloroform 1/7')
        self.assertNotIn('tau12_c', interaction)
        self.assertAlmostEqual(interaction['a12_cal_per_mol'], -227.3671, places=4)
        self.assertAlmostEqual(interaction['a21_cal_per_mol'], -86.1025, places=4)
        self.assertIsNone(nrtl_binary_interaction('74-87-3', '71-43-2'))
        methanol_carbon_tet = nrtl_binary_interaction('67-56-1', '56-23-5')
        self.assertEqual(methanol_carbon_tet['comment'], 'Methanol/Tetrachloromethane p18 1/2c')
        self.assertAlmostEqual(methanol_carbon_tet['a12_cal_per_mol'], 378.8254, places=4)

        selected_nrtl = [
            ('7732-18-5', '78-93-3', '2-Butanone/Water p279 1/1a'),
            ('64-17-5', '78-93-3', 'Ethanol/2-Butanone p342 1/2a'),
            ('110-82-7', '67-56-1', 'Methanol/Cyclohexane p211 1/2c'),
            ('142-82-5', '67-56-1', 'Methanol/n-Heptane'),
            ('110-82-7', '64-17-5', 'Ethanol/CycloHexane p441 1/2a'),
        ]
        for cas1, cas2, comment in selected_nrtl:
            with self.subTest(model='NRTL', pair=(cas1, cas2)):
                self.assertEqual(nrtl_binary_interaction(cas1, cas2)['comment'], comment)

    def test_nrtl_supports_temperature_dependent_tau_parameters(self):
        forward = nrtl_binary_interaction('7697-37-2', '7732-18-5')
        reverse = nrtl_binary_interaction('7732-18-5', '7697-37-2')

        self.assertAlmostEqual(forward['tau12_c'], 1.533, places=6)
        self.assertAlmostEqual(forward['tau12_d'], 84.6, places=6)
        self.assertAlmostEqual(reverse['tau12_c'], 0.178, places=6)
        self.assertAlmostEqual(reverse['tau12_d'], -865.8, places=6)

        thermo = create_thermodynamics(['HNO3', 'H2O'], 'NRTL')
        tau, alpha = thermo._nrtl_matrices(298.15)

        self.assertAlmostEqual(tau[0][1], 1.817, places=3)
        self.assertAlmostEqual(tau[1][0], -2.726, places=3)
        self.assertAlmostEqual(alpha[0][1], 0.3, places=6)
        self.assertAlmostEqual(alpha[1][0], 0.3, places=6)

    def test_interaction_parameter_tables_are_cas_keyed(self):
        def load_data(filename):
            with open(os.path.join(ROOT, 'data', filename), encoding='utf-8') as handle:
                return json.load(handle)

        for filename in (
            'eos_binary_interactions_cas.json',
            'nrtl_binary_interactions_cas.json',
            'uniquac_binary_interactions_cas.json',
        ):
            with self.subTest(filename=filename):
                payload = load_data(filename)
                self.assertEqual(payload['metadata']['key_basis'], 'CAS')
                for record in payload['interactions']:
                    self.assertIn('cas1', record)
                    self.assertIn('cas2', record)
                    self.assertNotIn('id1', record)
                    self.assertNotIn('id2', record)

        eos_payload = load_data('eos_binary_interactions_cas.json')
        self.assertEqual(
            {record['comment'] for record in eos_payload['metadata']['skipped']},
            {'Water/HC'},
        )
        self.assertEqual(eos_payload['metadata']['converted_records'], 621)
        self.assertEqual(eos_payload['metadata']['base_converted_records'], 353)
        self.assertEqual(eos_payload['metadata']['supplemental_records'], 2)
        self.assertEqual(eos_payload['metadata']['supplemental_new_pairs'], 1)
        self.assertEqual(eos_payload['metadata']['eos_collapsed_duplicate_groups'], 2)
        self.assertEqual(eos_payload['metadata']['eos_collapsed_duplicate_records'], 2)
        self.assertEqual(eos_payload['metadata']['ipd_hydration_records'], 266)
        self.assertEqual(eos_payload['metadata']['ipd_hydration_new_pairs'], 141)
        self.assertEqual(
            eos_payload['metadata']['ipd_hydration_by_source'],
            {'pr.ipd': 6, 'srk.ipd': 260},
        )
        eos_groups = {}
        for record in eos_payload['interactions']:
            key = (
                record['model'],
                tuple(sorted((record['cas1'], record['cas2']))),
                (record.get('Tmin_K'), record.get('Tmax_K')),
            )
            eos_groups[key] = eos_groups.get(key, 0) + 1
        self.assertTrue(all(count == 1 for count in eos_groups.values()))
        self.assertAlmostEqual(
            eos_binary_interaction('PR', '7727-37-9', '124-38-9', 273.15),
            -0.01445,
            places=6,
        )
        self.assertAlmostEqual(
            eos_binary_interaction('PR', '7783-06-4', '74-98-6'),
            0.02055,
            places=6,
        )
        self.assertAlmostEqual(
            eos_binary_interaction('SRK', '1333-74-0', '7727-37-9', 100.0),
            0.0563,
            places=6,
        )
        self.assertAlmostEqual(
            eos_binary_interaction('SRK', '1333-74-0', '7727-37-9', 90.0),
            0.1141,
            places=6,
        )
        nrtl_payload = load_data('nrtl_binary_interactions_cas.json')
        self.assertEqual(nrtl_payload['metadata']['skipped_records'], 0)
        self.assertEqual(nrtl_payload['metadata']['converted_records'], 434)
        self.assertEqual(nrtl_payload['metadata']['base_converted_records'], 348)
        self.assertEqual(nrtl_payload['metadata']['supplemental_records'], 81)
        self.assertEqual(nrtl_payload['metadata']['supplemental_new_pairs'], 48)
        self.assertEqual(nrtl_payload['metadata']['water_organic_overlay_records'], 7)
        self.assertEqual(nrtl_payload['metadata']['water_organic_overlay_replaced_records'], 4)
        self.assertEqual(nrtl_payload['metadata']['water_organic_overlay_replaced_pairs'], 3)
        self.assertEqual(nrtl_payload['metadata']['literature_vle_overlay_records'], 6)
        self.assertEqual(nrtl_payload['metadata']['literature_vle_overlay_replaced_records'], 0)
        self.assertEqual(nrtl_payload['metadata']['literature_vle_overlay_replaced_pairs'], 0)
        self.assertEqual(nrtl_payload['metadata']['ipd_hydration_records'], 5)
        self.assertEqual(
            nrtl_payload['metadata']['activity_collapsed_equivalent_duplicate_groups'],
            3,
        )
        self.assertEqual(
            nrtl_payload['metadata']['activity_collapsed_equivalent_duplicate_records'],
            3,
        )
        nrtl_pxylene = [
            record for record in nrtl_payload['interactions']
            if record.get('comment') == 'Benzene/pXylene p310 1/7 | 6 Benzene/P-Xylene p310 1/7'
        ]
        self.assertEqual(len(nrtl_pxylene), 1)
        self.assertAlmostEqual(nrtl_pxylene[0]['a12_cal_per_mol'], -50.2635)
        self.assertEqual(len(nrtl_pxylene[0]['duplicate_records']), 2)
        self.assertIsNotNone(nrtl_binary_interaction('106-99-0', '67-56-1'))
        uniquac_payload = load_data('uniquac_binary_interactions_cas.json')
        self.assertEqual(uniquac_payload['metadata']['skipped_records'], 0)
        self.assertEqual(uniquac_payload['metadata']['converted_records'], 420)
        self.assertEqual(uniquac_payload['metadata']['base_converted_records'], 327)
        self.assertEqual(uniquac_payload['metadata']['supplemental_records'], 75)
        self.assertEqual(uniquac_payload['metadata']['supplemental_new_pairs'], 47)
        self.assertEqual(uniquac_payload['metadata']['water_organic_overlay_records'], 7)
        self.assertEqual(uniquac_payload['metadata']['water_organic_overlay_replaced_records'], 3)
        self.assertEqual(uniquac_payload['metadata']['water_organic_overlay_replaced_pairs'], 2)
        self.assertEqual(uniquac_payload['metadata']['literature_vle_overlay_records'], 6)
        self.assertEqual(uniquac_payload['metadata']['literature_vle_overlay_replaced_records'], 3)
        self.assertEqual(uniquac_payload['metadata']['literature_vle_overlay_replaced_pairs'], 3)
        self.assertEqual(uniquac_payload['metadata']['ipd_hydration_records'], 18)
        self.assertEqual(
            uniquac_payload['metadata']['activity_collapsed_equivalent_duplicate_groups'],
            45,
        )
        self.assertEqual(
            uniquac_payload['metadata']['activity_collapsed_equivalent_duplicate_records'],
            45,
        )
        uniquac_methane_water = [
            record for record in uniquac_payload['interactions']
            if record.get('comment') == 'Methane/Water | Water/Methane'
        ]
        self.assertEqual(len(uniquac_methane_water), 1)
        self.assertEqual(len(uniquac_methane_water[0]['duplicate_records']), 2)
        self.assertIsNotNone(uniquac_binary_interaction('616-38-6', '107-21-1'))
        self.assertIsNotNone(nrtl_binary_interaction('67-56-1', '75-25-2'))
        self.assertEqual(cas_for_component('3-Methylpyridien'), '108-99-6')
        self.assertEqual(cas_for_component('p-Xylene'), '106-42-3')

    def test_binary_interaction_builders_match_runtime_json(self):
        resolved, unresolved = resolve_component_ids()
        cases = (
            (
                'eos_binary_interactions.json',
                'eos_binary_interactions_cas.json',
            ),
            (
                'nrtl_binary_interactions.json',
                'nrtl_binary_interactions_cas.json',
            ),
            (
                'uniquac_binary_interactions.json',
                'uniquac_binary_interactions_cas.json',
            ),
        )
        for source_name, runtime_name in cases:
            with self.subTest(runtime_name=runtime_name):
                built = build_interaction_payload(
                    source_name,
                    resolved,
                    unresolved,
                )
                with open(
                    os.path.join(ROOT, 'data', runtime_name),
                    encoding='utf-8',
                ) as handle:
                    runtime_text = handle.read()
                built_text = json.dumps(built, indent=2, sort_keys=True) + '\n'
                self.assertEqual(built_text, runtime_text)

    def test_uniquac_rq_builder_matches_runtime_json(self):
        built = build_uniquac_rq_payload()
        with open(
            os.path.join(ROOT, 'data', 'uniquac_rq_cas.json'),
            encoding='utf-8',
        ) as handle:
            runtime_text = handle.read()
        built_text = json.dumps(built, indent=2, sort_keys=True) + '\n'
        self.assertEqual(built_text, runtime_text)
        self.assertEqual(built['metadata']['component_count'], 66)
        expected = {
            '2-butanol': (3.45, 3.04),
            'n-propyl acetate': (4.153, 3.656),
            '1-pentanol': (4.129, 3.592),
            '2-pentanol': (4.283, 3.556),
            '3-methyl-1-butanol': (4.273, 3.478),
            'butyric acid': (3.5514, 3.15242),
            '2-octanol': (6.15128, 5.20828),
            'ethyl propanoate': (4.1535, 3.6559),
            'ethyl butanoate': (4.8279, 4.19632),
            'diisopropyl ether': (4.7421, 4.088),
        }
        for component, (expected_r, expected_q) in expected.items():
            with self.subTest(component=component):
                record = uniquac_rq_for_component(component)
                self.assertIsNotNone(record)
                self.assertAlmostEqual(record['r'], expected_r, places=8)
                self.assertAlmostEqual(record['q'], expected_q, places=8)
                self.assertNotIn('estimated', record['source'].lower())

    def test_water_organic_binary_fits_replace_existing_runtime_pairs(self):
        source_path = os.path.join(
            ROOT,
            'data',
            'source',
            'water_organic_binary_fits.json',
        )
        with open(source_path, encoding='utf-8') as handle:
            source = json.load(handle)

        resolved, unresolved = resolve_component_ids()
        for model in ('NRTL', 'UNIQUAC'):
            records, new_pairs, covered_pairs = (
                supplemental_water_organic_binary_fit_records([], model)
            )
            self.assertEqual(len(records), 7)
            self.assertEqual(new_pairs, 7)
            self.assertEqual(len(covered_pairs), 7)
            built = build_interaction_payload(
                f'{model.lower()}_binary_interactions.json',
                resolved,
                unresolved,
            )
            self.assertEqual(built['metadata']['water_organic_overlay_records'], 7)
            by_pair = {
                tuple(sorted((record['cas1'], record['cas2']))): record
                for record in records
            }
            for fit in source['pairs']:
                with self.subTest(model=model, pair=fit['pair_id']):
                    pair = tuple(sorted((fit['cas1'], fit['cas2'])))
                    matches = [
                        record for record in built['interactions']
                        if tuple(sorted((record['cas1'], record['cas2']))) == pair
                    ]
                    self.assertEqual(len(matches), 1)
                    record = matches[0]
                    expected_record = dict(by_pair[pair])
                    if model == 'NRTL':
                        record = {
                            **record,
                            'tau12_f': record.get('tau12_f', 0.0),
                            'tau21_f': record.get('tau21_f', 0.0),
                        }
                        expected_record = {
                            **expected_record,
                            'tau12_f': expected_record.get('tau12_f', 0.0),
                            'tau21_f': expected_record.get('tau21_f', 0.0),
                        }
                    self.assertEqual(record, expected_record)
                    self.assertEqual(record['source_pair_id'], fit['pair_id'])
                    self.assertEqual(record['Tmin_K'], fit['temperature_range_K']['Tmin'])
                    self.assertEqual(record['Tmax_K'], fit['temperature_range_K']['Tmax'])
                    if model == 'NRTL':
                        runtime = nrtl_binary_interaction(fit['cas1'], fit['cas2'])
                        reverse = nrtl_binary_interaction(fit['cas2'], fit['cas1'])
                        self.assertAlmostEqual(
                            runtime['tau12_c'],
                            fit['nrtl']['parameters']['tau12_c'],
                        )
                        self.assertAlmostEqual(
                            reverse['tau12_c'],
                            fit['nrtl']['parameters']['tau21_c'],
                        )
                        self.assertAlmostEqual(
                            runtime['tau_tref'], fit['nrtl']['tau_tref_K']
                        )
                    else:
                        runtime = uniquac_binary_interaction(fit['cas1'], fit['cas2'])
                        reverse = uniquac_binary_interaction(fit['cas2'], fit['cas1'])
                        self.assertAlmostEqual(
                            runtime['tau12_a'],
                            fit['uniquac']['parameters']['tau12_a'],
                        )
                        self.assertAlmostEqual(
                            reverse['tau12_a'],
                            fit['uniquac']['parameters']['tau21_a'],
                        )

        dipe_rq = uniquac_rq_for_component('diisopropyl ether')
        self.assertEqual(dipe_rq['source'].split(';', 1)[0], 'water_organic_binary_fits.json')
        self.assertAlmostEqual(dipe_rq['r'], 4.7421)
        self.assertAlmostEqual(dipe_rq['q'], 4.088)
        dipe_thermo = create_thermodynamics(
            ['water', 'diisopropyl ether'],
            'UNIQUAC',
        )
        self.assertFalse(any('r/q parameters missing' in warning for warning in dipe_thermo.warnings))

    def test_literature_vle_activity_overlay_retains_both_models(self):
        expected_pairs = {
            ('75-07-0', '7732-18-5'),
            ('64-19-7', '79-09-4'),
            ('64-19-7', '79-10-7'),
            ('79-09-4', '79-10-7'),
            ('79-10-7', '7732-18-5'),
            ('79-09-4', '7732-18-5'),
        }
        zero_pair = tuple(sorted(('79-09-4', '79-10-7')))

        for model in ('NRTL', 'UNIQUAC'):
            with self.subTest(model=model):
                records, new_pairs, covered_pairs = (
                    supplemental_literature_vle_activity_records([], model)
                )
                self.assertEqual(len(records), 6)
                self.assertEqual(new_pairs, 6)
                self.assertEqual(covered_pairs, {
                    tuple(sorted(pair)) for pair in expected_pairs
                })
                by_pair = {
                    tuple(sorted((record['cas1'], record['cas2']))): record
                    for record in records
                }
                zero = by_pair[zero_pair]
                self.assertEqual(
                    zero['fit_status'],
                    'recommended_defensible_zero_interaction',
                )
                self.assertIn(
                    'well below the experimental error margin',
                    zero['zero_interaction_basis'],
                )
                if model == 'NRTL':
                    self.assertEqual(zero['tau12_c'], 0.0)
                    self.assertEqual(zero['tau12_d'], 0.0)
                    self.assertEqual(zero['tau21_c'], 0.0)
                    self.assertEqual(zero['tau21_d'], 0.0)
                    acetaldehyde = by_pair[
                        tuple(sorted(('75-07-0', '7732-18-5')))
                    ]
                    self.assertAlmostEqual(acetaldehyde['tau12_c'], 5.25487)
                    self.assertAlmostEqual(acetaldehyde['tau12_d'], -1310.86)
                else:
                    self.assertEqual(zero['tau12_a'], 0.0)
                    self.assertEqual(zero['tau12_b'], 0.0)
                    self.assertEqual(zero['tau21_a'], 0.0)
                    self.assertEqual(zero['tau21_b'], 0.0)
                    acetaldehyde = by_pair[
                        tuple(sorted(('75-07-0', '7732-18-5')))
                    ]
                    self.assertAlmostEqual(acetaldehyde['tau12_a'], 3.71393)
                    self.assertAlmostEqual(acetaldehyde['tau12_b'], -1316.43)

                resolved, unresolved = resolve_component_ids()
                built = build_interaction_payload(
                    f'{model.lower()}_binary_interactions.json',
                    resolved,
                    unresolved,
                )
                self.assertEqual(
                    built['metadata']['literature_vle_overlay_records'], 6
                )
                for pair in covered_pairs:
                    matches = [
                        record for record in built['interactions']
                        if tuple(sorted((record['cas1'], record['cas2']))) == pair
                    ]
                    self.assertEqual(len(matches), 1)

    def test_all_recommended_ester_interactions_are_built_and_used(self):
        source_path = os.path.join(
            ROOT,
            'data',
            'source',
            'ester_alcohol_nrtl_uniquac_recommended_fits.json',
        )
        with open(source_path, encoding='utf-8') as handle:
            source = json.load(handle)
        for model in ('NRTL', 'UNIQUAC'):
            built, new_pairs = supplemental_ester_alcohol_fit_records([], model)
            self.assertEqual(len(built), 17)
            self.assertEqual(new_pairs, 17)
            by_pair = {record['comment'].split(' recommended ', 1)[0]: record for record in built}
            fits = [
                fit for fit in source['recommended_fits']
                if fit['model'] == model
            ]
            self.assertEqual(len(fits), 17)
            for fit in fits:
                with self.subTest(model=model, pair=fit['pair']):
                    record = by_pair[fit['pair']]
                    self.assertEqual(record['Tmin_K'], fit['Tmin_K'])
                    self.assertEqual(record['Tmax_K'], fit['Tmax_K'])
                    self.assertEqual(
                        record['fit_vapor_treatment'], fit['vapor_treatment']
                    )
                    self.assertEqual(record['fit_points'], fit['n_points'])
                    self.assertEqual(record['fit_sources'], fit['n_sources'])
                    if model == 'NRTL':
                        self.assertAlmostEqual(record['tau12_c'], fit['A12'])
                        self.assertAlmostEqual(record['tau12_d'], fit['B12_K'])
                        self.assertAlmostEqual(record['tau21_c'], fit['A21'])
                        self.assertAlmostEqual(record['tau21_d'], fit['B21_K'])
                        runtime = nrtl_binary_interaction(
                            record['cas1'], record['cas2']
                        )
                        self.assertAlmostEqual(runtime['tau12_c'], fit['A12'])
                    else:
                        self.assertAlmostEqual(record['tau12_a'], -fit['A12'])
                        self.assertAlmostEqual(record['tau12_b'], -fit['B12_K'])
                        self.assertAlmostEqual(record['tau21_a'], -fit['A21'])
                        self.assertAlmostEqual(record['tau21_b'], -fit['B21_K'])
                        runtime = uniquac_binary_interaction(
                            record['cas1'], record['cas2']
                        )
                        self.assertAlmostEqual(runtime['tau12_a'], -fit['A12'])

    def test_ethyl_acetate_water_25c_nrtl_solubilities(self):
        thermo = create_thermodynamics(['ethyl acetate', 'water'], 'NRTL')
        overall = {
            'ethyl acetate': 0.728342018634107,
            'water': 0.271657981365893,
        }
        split, ester_rich, aqueous, ester_rich_fraction = (
            thermo.liquid_liquid_equilibrium(
                overall,
                298.15,
                max_iter=500,
                tol=1.0e-10,
            )
        )

        self.assertTrue(split)
        self.assertAlmostEqual(
            ester_rich['water'], 0.13887931039065995, places=8
        )
        self.assertAlmostEqual(
            aqueous['ethyl acetate'], 0.016026170013681607, places=8
        )
        self.assertAlmostEqual(
            ester_rich['ethyl acetate'], 0.8611206896093401, places=8
        )
        self.assertAlmostEqual(
            aqueous['water'], 0.9839738299863184, places=8
        )
        self.assertAlmostEqual(ester_rich_fraction, 0.15711694715374797, places=8)

    def test_ethyl_formate_methanol_nrtl_azeotrope_matches_literature(self):
        thermo = create_thermodynamics(['ethyl formate', 'methanol'], 'NRTL')

        def decode(values):
            ethyl_formate = 1.0 / (1.0 + math.exp(-values[0]))
            return (
                {
                    'ethyl formate': ethyl_formate,
                    'methanol': 1.0 - ethyl_formate,
                },
                values[1],
            )

        def residual(values):
            composition, temperature = decode(values)
            k_values = thermo.K_values(temperature, 1.0, composition)
            return (
                math.log(k_values['ethyl formate']),
                math.log(k_values['methanol']),
            )

        solutions = []
        for initial_fraction in (0.2, 0.5, 0.8):
            solution = least_squares(
                residual,
                (
                    math.log(initial_fraction / (1.0 - initial_fraction)),
                    324.0,
                ),
                bounds=((-12.0, 280.0), (12.0, 380.0)),
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
            )
            composition, temperature = decode(solution.x)
            solutions.append((composition, temperature))

        reference_composition, reference_temperature = solutions[0]
        for composition, temperature in solutions[1:]:
            self.assertAlmostEqual(temperature, reference_temperature, places=8)
            self.assertAlmostEqual(
                composition['ethyl formate'],
                reference_composition['ethyl formate'],
                places=8,
            )

        x_ethyl_formate = reference_composition['ethyl formate']
        mw_ethyl_formate = thermo.props['ethyl formate'].MW
        mw_methanol = thermo.props['methanol'].MW
        mass_fraction = (
            x_ethyl_formate * mw_ethyl_formate
            / (
                x_ethyl_formate * mw_ethyl_formate
                + (1.0 - x_ethyl_formate) * mw_methanol
            )
        )

        self.assertEqual(round(reference_temperature - 273.15), 51)
        self.assertEqual(round(100.0 * mass_fraction), 84)

    def test_water_ethylene_oxide_fitted_interactions_are_built_and_used(self):
        water_cas = '7732-18-5'
        eo_cas = '75-21-8'

        with open(
            os.path.join(ROOT, 'data', 'source', 'water_ethylene_oxide_interactions.json'),
            encoding='utf-8',
        ) as handle:
            source = json.load(handle)
        self.assertEqual(source['metadata']['cas'], [water_cas, eo_cas])
        self.assertEqual(source['metadata']['activity_temperature_range_K'], [280.0, 420.0])
        self.assertEqual(source['metadata']['eos_temperature_range_K'], [330.0, 400.0])
        self.assertEqual(source['fitted_parameters']['UNIQUAC']['component2_r'], 1.59)
        self.assertEqual(source['fitted_parameters']['UNIQUAC']['component2_q'], 1.64)

        nrtl = nrtl_binary_interaction(water_cas, eo_cas)
        self.assertAlmostEqual(nrtl['tau12_c'], 5.85370)
        self.assertAlmostEqual(nrtl['tau12_d'], -1547.70)
        self.assertAlmostEqual(nrtl['tau21_c'], -4.70938)
        self.assertAlmostEqual(nrtl['tau21_d'], 1926.66)
        self.assertAlmostEqual(nrtl['alpha12'], 0.20)
        reversed_nrtl = nrtl_binary_interaction(eo_cas, water_cas)
        self.assertAlmostEqual(reversed_nrtl['tau12_c'], nrtl['tau21_c'])
        self.assertAlmostEqual(reversed_nrtl['tau12_d'], nrtl['tau21_d'])
        self.assertAlmostEqual(reversed_nrtl['tau21_c'], nrtl['tau12_c'])
        self.assertAlmostEqual(reversed_nrtl['tau21_d'], nrtl['tau12_d'])

        uniquac = uniquac_binary_interaction(water_cas, eo_cas)
        self.assertAlmostEqual(uniquac['tau12_a'], -2.31209)
        self.assertAlmostEqual(uniquac['tau12_b'], 673.029)
        self.assertAlmostEqual(uniquac['tau21_a'], 3.13674)
        self.assertAlmostEqual(uniquac['tau21_b'], -1397.36)
        reversed_uniquac = uniquac_binary_interaction(eo_cas, water_cas)
        self.assertAlmostEqual(reversed_uniquac['tau12_a'], uniquac['tau21_a'])
        self.assertAlmostEqual(reversed_uniquac['tau12_b'], uniquac['tau21_b'])

        self.assertAlmostEqual(
            eos_binary_interaction('PR', water_cas, eo_cas, 360.0),
            -0.10,
        )
        self.assertAlmostEqual(
            eos_binary_interaction('SRK', water_cas, eo_cas, 360.0),
            -0.13,
        )

        with open(
            os.path.join(ROOT, 'data', 'eos_binary_interactions_cas.json'),
            encoding='utf-8',
        ) as handle:
            eos_payload = json.load(handle)
        eos_records = [
            record for record in eos_payload['interactions']
            if {record['cas1'], record['cas2']} == {water_cas, eo_cas}
        ]
        self.assertEqual({record['model'] for record in eos_records}, {'PR', 'SRK'})
        for record in eos_records:
            self.assertEqual(record['Tmin_K'], 330.0)
            self.assertEqual(record['Tmax_K'], 400.0)

        nrtl_pr = create_thermodynamics(['H2O', 'C2H4O'], 'NRTL-PR')
        gamma = nrtl_pr.activity_coefficients(
            298.15,
            {'H2O': 1.0 - 1.0e-10, 'C2H4O': 1.0e-10},
        )
        self.assertAlmostEqual(gamma['C2H4O'], 6.666631833463465, places=8)
        self.assertFalse(any(
            'interaction parameters missing' in warning
            for warning in nrtl_pr.warnings
        ))

        bottoms = {
            'H2O': 0.9923309487336059,
            'C2H4O': 0.007669051266394127,
        }
        nrtl_temperature = nrtl_pr.bubble_point_T(bottoms, 2.24, 380.0)
        self.assertAlmostEqual(nrtl_temperature, 382.21945717059197, places=6)

        uniquac_pr = create_thermodynamics(['H2O', 'C2H4O'], 'UNIQUAC-PR')
        self.assertAlmostEqual(uniquac_pr.r['C2H4O'], 1.59)
        self.assertAlmostEqual(uniquac_pr.q['C2H4O'], 1.64)
        uniquac_gamma = uniquac_pr.activity_coefficients(
            298.15,
            {'H2O': 1.0 - 1.0e-10, 'C2H4O': 1.0e-10},
        )
        self.assertAlmostEqual(
            uniquac_gamma['C2H4O'],
            6.749482481162802,
            places=8,
        )
        self.assertFalse(any(
            "UNIQUAC r/q parameters missing for 'C2H4O'" in warning
            for warning in uniquac_pr.warnings
        ))
        uniquac_temperature = uniquac_pr.bubble_point_T(bottoms, 2.24, 380.0)
        self.assertAlmostEqual(uniquac_temperature, 382.4763417742024, places=6)

    def test_interaction_conversion_corrects_row_level_identity_mismatches(self):
        def records(filename, comment):
            with open(os.path.join(ROOT, 'data', filename), encoding='utf-8') as handle:
                payload = json.load(handle)
            return [
                record for record in payload['interactions']
                if record.get('comment') == comment
            ]

        for filename in (
            'nrtl_binary_interactions_cas.json',
            'uniquac_binary_interactions_cas.json',
        ):
            with self.subTest(filename=filename, comment='Ethanol/2-Butanol'):
                record = records(filename, 'Ethanol/2-Butanol p346 1/2c')[0]
                self.assertEqual(record['cas2'], '78-92-2')
                self.assertEqual(record['component2'], '2-butanol')

            with self.subTest(filename=filename, comment='Ethanol/2.6-Dimethylpyridine'):
                record = records(filename, 'Ethanol/2.6-Dimethylpyridine p447 1/2c')[0]
                self.assertEqual(record['cas2'], '108-48-5')
                self.assertEqual(record['component2'], '2,6-dimethylpyridine')

            with self.subTest(filename=filename, comment='Propylamine/1-Propanol'):
                record = records(filename, 'Propylamine/1-Propanol p492 1/2c')[0]
                self.assertEqual(record['cas1'], '107-10-8')
                self.assertEqual(record['cas2'], '71-23-8')

            with self.subTest(filename=filename, comment='Methanol/Formamide'):
                record = records(filename, 'Methanol/Formamide p30 1/2c')[0]
                self.assertEqual(record['cas1'], '67-56-1')
                self.assertEqual(record['cas2'], '75-12-7')

    def test_eos_temperature_specific_kij_selection_blends_static_fallback(self):
        self.assertAlmostEqual(
            eos_binary_interaction('PR', '124-38-9', '7783-06-4', 300.0),
            0.0967,
            places=6,
        )
        self.assertAlmostEqual(
            eos_binary_interaction('PR', '124-38-9', '7783-06-4', 400.0),
            0.09835,
            places=6,
        )

        eos = CubicEOS(['CO2', 'H2S'], 'PR')
        self.assertAlmostEqual(eos._kij('CO2', 'H2S', 300.0), 0.0967, places=6)
        self.assertAlmostEqual(eos._kij('CO2', 'H2S', 400.0), 0.09835, places=6)

    def test_eos_point_temperature_records_get_ten_kelvin_window(self):
        records = [
            {'kij': 0.10},
            {'kij': 0.20, 'temperature_range': (190.0, 220.0)},
            {'kij': 0.30, 'temperature_range': (202.0, 202.0)},
        ]

        self.assertAlmostEqual(_select_eos_kij(records, 200.0), 0.30)
        self.assertAlmostEqual(_select_eos_kij(records, 202.0), 0.30)
        self.assertAlmostEqual(_select_eos_kij(records, 215.0), 0.20)
        self.assertAlmostEqual(_select_eos_kij(records, 230.0), 0.15)

        eos = object.__new__(CubicEOS)
        eos._temperature_kij = {
            ('A', 'B'): [
                (0.20, 190.0, 220.0),
                (0.30, 202.0, 202.0),
            ],
        }
        eos._static_kij = {('A', 'B'): 0.10}
        eos._kij_cache = {}

        self.assertAlmostEqual(eos._kij('A', 'B', 200.0), 0.30)
        self.assertAlmostEqual(eos._kij('A', 'B', 215.0), 0.20)
        self.assertAlmostEqual(eos._kij('A', 'B', 230.0), 0.15)

    def test_temperature_dependent_pfd_kij_contributes_to_da_mix_dT(self):
        eos = CubicEOS(
            ['CO2', 'H2S'],
            'PR',
            ChemicalDatabase(enable_online=False),
            interaction_overrides=[{
                'model': 'PR',
                'component1': 'CO2',
                'component2': 'H2S',
                'kij_a': 0.1,
                'kij_b': 30.0,
                'kij_c': 0.001,
            }],
        )
        composition = {'CO2': 0.4, 'H2S': 0.6}
        T = 325.0
        h = 1e-3
        finite_difference = (
            eos.mixture_params(T + h, composition)[0]
            - eos.mixture_params(T - h, composition)[0]
        ) / (2.0 * h)

        self.assertAlmostEqual(eos.mixture_da_dT(T, composition), finite_difference, delta=abs(finite_difference) * 1e-5)

    def test_nrtl_supplemental_tau_matrix_predicts_ethanol_water_azeotrope(self):
        interaction = nrtl_binary_interaction('64-17-5', '7732-18-5')
        self.assertIsNotNone(interaction)
        self.assertAlmostEqual(interaction['tau12_c'], -0.801)
        self.assertAlmostEqual(interaction['tau12_d'], 246.2)
        self.assertAlmostEqual(interaction['tau21_c'], 3.458)
        self.assertAlmostEqual(interaction['tau21_d'], -586.1)
        self.assertAlmostEqual(interaction['alpha12'], 0.3)

        thermo = create_thermodynamics(['ethanol', 'water'], 'NRTL')
        azeotrope = thermo.generate_Txy_data(
            'ethanol',
            'water',
            1.01325,
            n_points=101,
        )['azeotrope']

        self.assertIsNotNone(azeotrope)
        self.assertAlmostEqual(azeotrope['x'], 0.898, delta=0.01)
        self.assertAlmostEqual(azeotrope['T'], 78.1, delta=0.3)

    def test_water_aromatic_regressions_roughly_reproduce_solubility_data(self):
        cases = [
            ('benzene', 343.15, 104.0, 11.3e-3, 6.10e-4),
            ('toluene', 348.15, 71.0, 12.2e-3, 1.89e-4),
            ('p-Xylene', 348.15, 52.0, 12.4e-3, 6.38e-5),
        ]

        for method in ('NRTL', 'UNIQUAC'):
            for organic, T, P_kPa, xw_organic_rich, xo_water_rich in cases:
                with self.subTest(method=method, organic=organic):
                    thermo = create_thermodynamics(['water', organic], method)
                    has_lle, phase1, phase2, _beta = thermo.liquid_liquid_equilibrium(
                        {'water': 0.5, organic: 0.5},
                        T,
                    )
                    self.assertTrue(has_lle)
                    organic_rich, water_rich = sorted(
                        (phase1, phase2),
                        key=lambda phase: phase[organic],
                        reverse=True,
                    )
                    self.assertLess(
                        abs(math.log(organic_rich['water'] / xw_organic_rich)),
                        0.12,
                    )
                    self.assertLess(
                        abs(math.log(water_rich[organic] / xo_water_rich)),
                        0.12,
                    )
                    pressure_bar = P_kPa / 100.0
                    average_bubble_pressure = 0.5 * (
                        thermo.bubble_point_P(organic_rich, T)
                        + thermo.bubble_point_P(water_rich, T)
                    )
                    self.assertLess(
                        abs(average_bubble_pressure / pressure_bar - 1.0),
                        0.08,
                    )

    def test_unifac_and_unifdmd_distinguish_aldehyde_and_ether_ch(self):
        classic = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_params.json'))
        dortmund = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_dmd.txt'))

        for model in (classic, dortmund):
            self.assertIn('CHO', model.subgroup_by_name)
            self.assertIn('CH-O', model.subgroup_by_name)
            self.assertEqual(model.subgroup_by_name['CHO'].number, 20)
            self.assertEqual(model.subgroup_by_name['CH-O'].number, 26)
            self.assertIn(20, model._resolve_groups({'CHO': 1}))
            self.assertIn(26, model._resolve_groups({'CH-O': 1}))

        self.assertEqual(len(dortmund.subgroups), 125)
        self.assertIn(20, dortmund._resolve_groups(get_unifac_groups('acetaldehyde', variant='UNIFDMD')))

        with self.assertRaisesRegex(ValueError, 'primary/secondary/tertiary OH subgroup'):
            get_unifac_groups('butanol', variant='UNIFDMD')

        self.assertGreater(dortmund.calculate_r_q({'CH-O': 1})[0], 0.0)

    def test_modified_unifac_uses_cyclic_alkane_groups_for_known_cycloalkanes(self):
        classic = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_params.json'))
        dortmund = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_dmd.txt'))
        nist = UNIFACModel(os.path.join(ROOT, 'data', 'nist_modified_unifac_params.json'))

        self.assertEqual(classic._resolve_groups(get_unifac_groups('cyclohexane')), {2: 6})
        self.assertEqual(dortmund._resolve_groups(get_unifac_groups('cyclohexane', variant='UNIFDMD')), {78: 6})
        self.assertEqual(nist._resolve_groups(get_unifac_groups('cyclohexane', variant='UNIFNIST')), {78: 6})
        self.assertEqual(dortmund._resolve_groups(get_unifac_groups('cyclopentane', variant='UNIFDMD')), {78: 5})
        self.assertEqual(nist._resolve_groups(get_unifac_groups('cyclopentane', variant='UNIFNIST')), {78: 5})

    def test_dortmund_uses_dedicated_epoxide_group_for_ethylene_oxide(self):
        dortmund = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_dmd.txt'))

        self.assertNotIn('C2H4O', DORTMUND_UNIFAC_KNOWN_OVERRIDES)
        self.assertNotIn('C3H6O', DORTMUND_UNIFAC_KNOWN_OVERRIDES)

        for identifier in ('75-21-8', 'ethylene oxide', 'oxirane'):
            with self.subTest(identifier=identifier):
                groups = get_unifac_groups(identifier, variant='UNIFDMD')
                self.assertEqual(groups, {119: 1})
                self.assertEqual(dortmund._resolve_groups(groups), {119: 1})

        subgroup = dortmund.subgroups[119]
        self.assertEqual(subgroup.name, 'H2COCH2')
        self.assertEqual(subgroup.main_group, 53)

        for identifier in (
            'C3H6O_PO',
            '75-56-9',
            'propylene oxide',
            '1,2-propylene oxide',
            '1,2-epoxypropane',
            'methyloxirane',
        ):
            with self.subTest(identifier=identifier):
                groups = get_unifac_groups(identifier, variant='UNIFDMD')
                self.assertEqual(groups, {1: 1, 107: 1})
                self.assertEqual(
                    dortmund._resolve_groups(groups),
                    {1: 1, 107: 1},
                )

        self.assertEqual(dortmund.subgroups[107].name, 'H2COCH')
        self.assertEqual(dortmund.subgroups[107].main_group, 53)

    def test_native_fragmentation_assigns_variant_specific_unifac_groups(self):
        classic = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_params.json'))
        dortmund = UNIFACModel(os.path.join(ROOT, 'data', 'unifac_dmd.txt'))
        nist = UNIFACModel(os.path.join(ROOT, 'data', 'nist_modified_unifac_params.json'))

        aldehyde = get_unifac_groups('acetaldehyde-smiles', smiles='CC=O')
        ether = get_unifac_groups(
            'diisopropyl-ether-smiles',
            smiles='CC(C)OC(C)C',
        )
        self.assertIn(20, classic._resolve_groups(aldehyde))
        self.assertIn(26, classic._resolve_groups(ether))

        aniline = get_unifac_groups(
            'aniline-smiles',
            smiles='Nc1ccccc1',
            variant='UNIFNIST',
        )
        pyridine = get_unifac_groups(
            'pyridine-smiles',
            smiles='c1ccncc1',
            variant='UNIFNIST',
        )
        self.assertEqual(nist._resolve_groups(aniline), {9: 5, 36: 1})
        self.assertEqual(nist._resolve_groups(pyridine), {9: 3, 37: 1})

        glycerol_dmd = get_unifac_groups(
            'glycerol-smiles',
            smiles='C(C(CO)O)O',
            variant='UNIFDMD',
        )
        glycerol_nist = get_unifac_groups(
            'glycerol-smiles',
            smiles='C(C(CO)O)O',
            variant='UNIFNIST',
        )
        self.assertEqual(dortmund._resolve_groups(glycerol_dmd), {2: 2, 3: 1, 14: 2, 81: 1})
        self.assertEqual(nist._resolve_groups(glycerol_nist), {204: 1})

        decalin_nist = get_unifac_groups(
            'decalin-smiles',
            smiles='C1CCC2CCCCC2C1',
            variant='UNIFNIST',
        )
        self.assertEqual(nist._resolve_groups(decalin_nist), {78: 8, 79: 2})

        for smiles in ['CC1CO1', 'C(=O)(C(F)(F)F)O', 'CC(=O)c1ccccc1']:
            groups = get_unifac_groups(
                'native-fragmented',
                smiles=smiles,
                variant='UNIFDMD',
            )
            self.assertGreater(dortmund.calculate_r_q(groups)[0], 0.0)

        self.assertEqual(
            nist._resolve_groups(get_unifac_groups('pyridine', variant='UNIFNIST')),
            {9: 3, 37: 1},
        )

    def test_unifac_thermo_resolves_smiles_before_native_fragmentation(self):
        db = ChemicalDatabase(enable_online=True)
        db._smiles_cache_path = os.path.join(tempfile.mkdtemp(), 'smiles.sqlite')
        db.chemicals['ANI'] = ChemicalProperties(
            symbol='ANI',
            name='Resolver Aniline',
            formula=None,
            MW=93.13,
            Tb=457.28,
        )
        db.chemicals['PYR'] = ChemicalProperties(
            symbol='PYR',
            name='Resolver Pyridine',
            formula=None,
            MW=79.10,
            Tb=388.35,
        )

        structures = {
            'Resolver Aniline': {'smiles': 'Nc1ccccc1', 'source': 'test_pubchem'},
            'Resolver Pyridine': {'smiles': 'c1ccncc1', 'source': 'test_pubchem'},
        }

        with patch.object(
            db,
            '_smiles_from_pubchem',
        ) as mocked_fetch:
            from chemical_properties import SmilesResolution
            mocked_fetch.side_effect = lambda identifier: (
                SmilesResolution(
                    smiles=structures[identifier]['smiles'],
                    source=structures[identifier]['source'],
                    method='pubchem_structure',
                    quality=0.97,
                    identifier=identifier,
                )
                if identifier in structures else None
            )
            thermo = create_thermodynamics(['ANI', 'PYR'], 'UNIFNIST', db)

        self.assertEqual(thermo.component_groups['ANI'], {9: 5, 36: 1})
        self.assertEqual(thermo.component_groups['PYR'], {9: 3, 37: 1})
        self.assertEqual(db.chemicals['ANI'].smiles, 'Nc1ccccc1')
        self.assertEqual(db.chemicals['PYR'].smiles, 'c1ccncc1')
        self.assertIn('Resolver Aniline', [call.args[0] for call in mocked_fetch.call_args_list])
        self.assertIn('Resolver Pyridine', [call.args[0] for call in mocked_fetch.call_args_list])

    def test_hexane_water_lle_detects_tiny_second_phase(self):
        thermo = create_thermodynamics(['hexane', 'water'], 'UNIFAC')

        for z_hexane in (0.001, 0.5, 0.999):
            with self.subTest(z_hexane=z_hexane):
                has_lle, x1, x2, beta = thermo.liquid_liquid_equilibrium(
                    {'hexane': z_hexane, 'water': 1.0 - z_hexane},
                    298.15,
                )
                self.assertTrue(has_lle)
                self.assertGreater(x1['hexane'], 0.99)
                self.assertLess(x1['water'], 0.01)
                self.assertLess(x2['hexane'], 0.001)
                self.assertGreater(x2['water'], 0.999)
                self.assertGreater(beta, 0.0)
                self.assertLess(beta, 1.0)

        miscible = create_thermodynamics(['ethanol', 'water'], 'UNIFAC')
        has_lle, _, _, _ = miscible.liquid_liquid_equilibrium(
            {'ethanol': 0.5, 'water': 0.5},
            298.15,
        )
        self.assertFalse(has_lle)

    def test_binary_lle_fallback_does_not_invent_endpoint_split(self):
        thermo = create_thermodynamics(['ethanol', 'hexane'], 'UNIFNIST')

        has_lle, x1, x2, beta = thermo.liquid_liquid_equilibrium(
            {'ethanol': 0.5, 'hexane': 0.5},
            298.15,
        )

        self.assertFalse(has_lle)
        self.assertEqual(x1, {'ethanol': 0.5, 'hexane': 0.5})
        self.assertEqual(x2, {'ethanol': 0.5, 'hexane': 0.5})
        self.assertEqual(beta, 0.0)

    def test_reference_flash3_tp_reports_lle_only_and_single_liquid(self):
        lle = create_thermodynamics(['hexane', 'water'], 'UNIFAC')
        lle_result = lle.flash3_TP(
            {'hexane': 0.5, 'water': 0.5},
            298.15,
            1.01325,
        )

        self.assertEqual(lle_result.status, 'lle_only')
        self.assertEqual(lle_result.phase_count, 2)
        self.assertAlmostEqual(lle_result.vapor_fraction, 0.0)
        self.assertGreater(lle_result.x1['hexane'], 0.99)
        self.assertGreater(lle_result.x2['water'], 0.99)
        self.assertLess(lle_result.residual, 1e-6)

        miscible = create_thermodynamics(['ethanol', 'water'], 'UNIFAC')
        single = miscible.flash3_TP(
            {'ethanol': 0.5, 'water': 0.5},
            298.15,
            1.01325,
        )

        self.assertEqual(single.status, 'single_liquid')
        self.assertEqual(single.phase_count, 1)
        self.assertEqual(single.x1, {'ethanol': 0.5, 'water': 0.5})
        self.assertEqual(single.x2, {'ethanol': 0.5, 'water': 0.5})

    def test_reference_flash3_tp_solves_ternary_vlle_with_low_residual(self):
        thermo = create_thermodynamics(['water', 'ethanol', 'cyclohexane'], 'UNIFNIST')
        z = {'water': 0.30, 'ethanol': 0.20, 'cyclohexane': 0.50}

        result = thermo.flash3_TP(z, 337.0, 1.01325, max_iter=200)

        self.assertEqual(result.status, 'structured_vlle_feed_lle_seed')
        self.assertEqual(result.phase_count, 3)
        self.assertGreater(result.vapor_fraction, 0.0)
        self.assertGreater(result.liquid1_fraction, 0.0)
        self.assertGreater(result.liquid2_fraction, 0.0)
        self.assertAlmostEqual(
            result.vapor_fraction + result.liquid1_fraction + result.liquid2_fraction,
            1.0,
            places=12,
        )
        for comp, expected in z.items():
            actual = (
                result.vapor_fraction * result.y[comp]
                + result.liquid1_fraction * result.x1[comp]
                + result.liquid2_fraction * result.x2[comp]
            )
            self.assertAlmostEqual(actual, expected, places=10)
        self.assertLess(result.residual, 1e-7)
        self.assertGreater(result.x1['water'], 0.7)
        self.assertGreater(result.x2['cyclohexane'], 0.9)
        self.assertGreater(result.extra['acceleration_attempts'], 0)
        self.assertGreater(result.extra['acceleration_accepts'], 0)
        self.assertLess(result.iterations, 30)

    def test_reference_flash3_tp_broyden_accelerates_gamma_phi_vlle(self):
        cases = [
            (
                'NRTL-RK',
                ['water', 'methanol', 'benzene'],
                {'water': 0.20, 'methanol': 0.30, 'benzene': 0.50},
                333.0,
            ),
            (
                'UNIQUAC-PR',
                ['water', 'ethanol', 'benzene'],
                {'water': 0.30, 'ethanol': 0.10, 'benzene': 0.60},
                339.0,
            ),
        ]

        for method, components, z, T in cases:
            with self.subTest(method=method):
                thermo = create_thermodynamics(components, method)

                result = thermo.flash3_TP(z, T, 1.01325, max_iter=200)

                self.assertEqual(result.status, 'structured_vlle_feed_lle_seed')
                self.assertEqual(result.phase_count, 3)
                self.assertGreater(result.extra['acceleration_attempts'], 0)
                self.assertGreater(result.extra['acceleration_accepts'], 0)
                self.assertLess(result.iterations, 20)
                self.assertLess(result.residual, 1e-7)
                for comp, expected in z.items():
                    actual = (
                        result.vapor_fraction * result.y[comp]
                        + result.liquid1_fraction * result.x1[comp]
                        + result.liquid2_fraction * result.x2[comp]
                    )
                    self.assertAlmostEqual(actual, expected, places=10)

    def test_reference_flash3_tp_binary_invariant_reports_underdetermined_amounts(self):
        thermo = create_thermodynamics(['water', 'chloroform'], 'UNIFNIST')
        z = {'water': 0.5, 'chloroform': 0.5}
        P = 1.01325

        def pressure_residual(T):
            has_lle, x1, _x2, _beta = thermo.liquid_liquid_equilibrium(z, T)
            if not has_lle:
                return float('nan')
            return thermo.bubble_point_P(x1, T) - P

        samples = []
        for T in range(300, 391):
            try:
                value = pressure_residual(float(T))
            except Exception:
                continue
            if math.isfinite(value):
                samples.append((float(T), value))
        bracket = None
        for left, right in zip(samples, samples[1:]):
            if left[1] * right[1] <= 0.0:
                bracket = (left[0], right[0])
                break
        self.assertIsNotNone(bracket)
        T_vlle = brentq(pressure_residual, bracket[0], bracket[1])

        result = thermo.flash3_TP(z, T_vlle, P)

        self.assertEqual(result.status, 'binary_invariant_vlle')
        self.assertEqual(result.phase_count, 3)
        self.assertTrue(result.extra['phase_amounts_underdetermined'])
        self.assertLess(result.residual, 1e-6)
        self.assertAlmostEqual(T_vlle - 273.15, 55.95, delta=0.05)
        low, high = result.extra['vapor_fraction_bounds']
        self.assertLessEqual(low, result.vapor_fraction)
        self.assertGreaterEqual(high, result.vapor_fraction)
        self.assertGreater(result.x1['water'], 0.99)
        self.assertGreater(result.x2['chloroform'], 0.99)

    def test_legacy_vlle_flash_delegates_to_reference_flash3(self):
        thermo = create_thermodynamics(['water', 'ethanol', 'cyclohexane'], 'UNIFNIST')
        z = {'water': 0.30, 'ethanol': 0.20, 'cyclohexane': 0.50}

        reference = thermo.flash3_TP(z, 337.0, 1.01325, max_iter=200)
        vapor_fraction, liquid2_fraction, y, x1, x2 = thermo.vlle_flash(
            z,
            337.0,
            1.01325,
            max_iter=200,
        )

        self.assertAlmostEqual(vapor_fraction, reference.vapor_fraction)
        self.assertAlmostEqual(liquid2_fraction, reference.liquid2_fraction)
        self.assertEqual(y, reference.y)
        self.assertEqual(x1, reference.x1)
        self.assertEqual(x2, reference.x2)

    def test_vlle_bubble_point_detects_binary_heteroazeotrope(self):
        thermo = create_thermodynamics(['water', 'chloroform'], 'UNIFNIST')
        z = {'water': 0.5, 'chloroform': 0.5}
        P = 1.01325

        T_bubble = thermo.bubble_point_T_vlle(z, P, T_guess=330.0)
        result = thermo.flash3_TP(z, T_bubble, P)

        self.assertAlmostEqual(T_bubble - 273.15, 55.976457, places=4)
        self.assertEqual(result.status, 'binary_invariant_vlle')
        self.assertTrue(result.extra['phase_amounts_underdetermined'])
        self.assertLess(result.residual, 1e-6)

    def test_binary_invariant_vlle_spec_flashes_select_phase_amounts(self):
        cases = [
            (
                'NRTL',
                ['water', 'toluene'],
                {'water': 0.5, 'toluene': 0.5},
                84.493668,
                0.88916,
            ),
            (
                'UNIFNIST',
                ['water', 'chloroform'],
                {'water': 0.5, 'chloroform': 0.5},
                55.976457,
                0.59576,
            ),
        ]
        P = 1.01325
        target_vapor_fraction = 0.2

        for method, components, z, expected_T_C, minimum_high_bound in cases:
            with self.subTest(method=method, components=components):
                thermo = create_thermodynamics(components, method)

                T_pv, pv = thermo.flash3_PV(
                    z,
                    P,
                    target_vapor_fraction,
                    T_guess=350.0,
                    max_iter=200,
                )

                self.assertEqual(pv.status, 'binary_invariant_vlle')
                self.assertEqual(pv.phase_count, 3)
                self.assertAlmostEqual(T_pv - 273.15, expected_T_C, places=4)
                self.assertAlmostEqual(pv.vapor_fraction, target_vapor_fraction, places=12)
                self.assertAlmostEqual(
                    pv.vapor_fraction + pv.liquid1_fraction + pv.liquid2_fraction,
                    1.0,
                    places=12,
                )
                low, high = pv.extra['vapor_fraction_bounds']
                self.assertLessEqual(low, target_vapor_fraction)
                self.assertGreater(high, minimum_high_bound)
                self.assertEqual(pv.extra['selected_vapor_fraction'], target_vapor_fraction)

                P_tv, tv = thermo.flash3_TV(
                    z,
                    T_pv,
                    target_vapor_fraction,
                    P_guess=1.2 * P,
                    max_iter=200,
                )

                self.assertAlmostEqual(P_tv, P, delta=5e-9)
                self.assertEqual(tv.status, 'binary_invariant_vlle')
                self.assertAlmostEqual(tv.vapor_fraction, target_vapor_fraction, places=12)
                self.assertAlmostEqual(tv.liquid1_fraction, pv.liquid1_fraction, places=12)
                self.assertAlmostEqual(tv.liquid2_fraction, pv.liquid2_fraction, places=12)

                H_target = thermo._flash3_mixture_enthalpy(T_pv, P, pv)
                T_ph, ph, residual = thermo.flash3_PH(
                    z,
                    P,
                    H_target,
                    T_guess=350.0,
                    max_iter=200,
                )

                self.assertAlmostEqual(T_ph, T_pv, places=8)
                self.assertEqual(ph.status, 'binary_invariant_vlle')
                self.assertAlmostEqual(ph.vapor_fraction, target_vapor_fraction, places=12)
                self.assertLess(abs(residual), 1e-6)

                S_target = thermo._flash3_mixture_entropy(T_pv, P, pv)
                T_ps, ps, entropy_residual = thermo.flash3_PS(
                    z,
                    P,
                    S_target,
                    T_guess=T_pv + 40.0,
                    max_iter=200,
                )

                self.assertAlmostEqual(T_ps, T_pv, places=8)
                self.assertEqual(ps.status, 'binary_invariant_vlle')
                self.assertAlmostEqual(ps.vapor_fraction, target_vapor_fraction, places=12)
                self.assertLess(abs(entropy_residual), 1e-7)

    def test_activity_scalar_solver_does_not_return_best_non_root_sample(self):
        thermo = create_thermodynamics(['water', 'ethanol'], 'NRTL')

        with self.assertRaisesRegex(ThermodynamicsError, 'Could not bracket impossible target'):
            thermo._solve_grid_scalar_residual(
                lambda value: 1.0 + value,
                [1.0, 2.0, 3.0],
                preferred=2.0,
                label='impossible target',
            )

    def test_pr_bubble_point_rejects_eo_water_nonroot_sample(self):
        components = ['C2H4', 'O2', 'C2H4O', 'CO2', 'H2O', 'N2']
        composition = {
            'C2H4O': 0.11700312431386949,
            'H2O': 0.8829968756861305,
        }
        pressure = 2.24
        thermo = create_thermodynamics(
            components,
            'PR',
            interaction_overrides=[{
                'model': 'PR',
                'component1': 'C2H4O',
                'component2': 'H2O',
                'kij': 0.0,
            }],
        )

        with self.assertRaisesRegex(
            ThermodynamicsError,
            r'best residual 0\.210707 at T=245\.545 K',
        ):
            thermo.bubble_point_T(composition, pressure, 350.0)

        fitted = create_thermodynamics(
            components,
            'PR',
            interaction_overrides=[{
                'model': 'PR',
                'component1': 'C2H4O',
                'component2': 'H2O',
                'kij': -0.1,
            }],
        )
        temperature = fitted.bubble_point_T(
            composition,
            pressure,
            350.0,
        )
        K_values = fitted.K_values(
            temperature,
            pressure,
            composition,
        )
        residual = sum(
            composition[component] * K_values[component]
            for component in composition
        ) - 1.0

        self.assertAlmostEqual(temperature, 316.2191566, places=6)
        self.assertLess(abs(residual), 1.0e-8)

    def test_vlle_bubble_dew_and_spec_flashes_recover_ternary_reference_state(self):
        thermo = create_thermodynamics(['water', 'ethanol', 'cyclohexane'], 'UNIFNIST')
        z = {'water': 0.30, 'ethanol': 0.20, 'cyclohexane': 0.50}
        P = 1.01325
        T_reference = 337.0

        reference = thermo.flash3_TP(z, T_reference, P, max_iter=200)
        H_reference = thermo._flash3_mixture_enthalpy(T_reference, P, reference)
        S_reference = thermo._flash3_mixture_entropy(T_reference, P, reference)

        T_bubble = thermo.bubble_point_T_vlle(z, P, T_guess=T_reference, max_iter=200)
        bubble = thermo.flash3_TP(z, T_bubble, P, max_iter=200)
        self.assertEqual(bubble.phase_count, 3)
        self.assertLess(bubble.vapor_fraction, 1e-5)
        self.assertLess(abs(T_bubble - 336.468179), 1e-4)

        T_dew = thermo.dew_point_T_vlle(reference.y, P, T_guess=T_reference, max_iter=200)
        self.assertAlmostEqual(T_dew, T_reference, places=6)

        T_pv, pv = thermo.flash3_PV(
            z,
            P,
            reference.vapor_fraction,
            T_guess=T_reference + 5.0,
            max_iter=200,
        )
        self.assertAlmostEqual(T_pv, T_reference, places=5)
        self.assertAlmostEqual(pv.vapor_fraction, reference.vapor_fraction, places=6)
        self.assertLess(pv.residual, 1e-7)

        P_tv, tv = thermo.flash3_TV(
            z,
            T_reference,
            reference.vapor_fraction,
            P_guess=1.15 * P,
            max_iter=200,
        )
        self.assertAlmostEqual(P_tv, P, delta=5e-5)
        self.assertEqual(tv.phase_count, 3)
        self.assertAlmostEqual(tv.vapor_fraction, reference.vapor_fraction, places=8)

        T_ph, ph, residual = thermo.flash3_PH(
            z,
            P,
            H_reference,
            T_guess=T_reference,
            max_iter=200,
        )
        self.assertAlmostEqual(T_ph, T_reference, delta=1e-3)
        self.assertEqual(ph.phase_count, 3)
        self.assertLess(abs(residual), 20.0)

        T_ph_offset, ph_offset, residual_offset = thermo.flash3_PH(
            z,
            P,
            H_reference,
            T_guess=T_reference + 8.0,
            max_iter=200,
        )
        self.assertAlmostEqual(T_ph_offset, T_reference, delta=1e-3)
        self.assertEqual(ph_offset.phase_count, 3)
        self.assertLess(abs(residual_offset), 20.0)

        T_ps, ps, entropy_residual = thermo.flash3_PS(
            z,
            P,
            S_reference,
            T_guess=T_reference - 30.0,
            max_iter=200,
        )
        self.assertAlmostEqual(T_ps, T_reference, delta=1e-3)
        self.assertEqual(ps.phase_count, 3)
        self.assertLess(abs(entropy_residual), 0.1)

        T_ps_offset, ps_offset, entropy_residual_offset = thermo.flash3_PS(
            z,
            P,
            S_reference,
            T_guess=T_reference + 40.0,
            max_iter=200,
        )
        self.assertAlmostEqual(T_ps_offset, T_reference, delta=1e-3)
        self.assertEqual(ps_offset.phase_count, 3)
        self.assertLess(abs(entropy_residual_offset), 0.1)

    def test_vlle_ph_prefers_three_phase_enthalpy_root_with_offset_guess(self):
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

        for method, components, z, T_reference in cases:
            with self.subTest(method=method):
                thermo = create_thermodynamics(components, method)
                P = 1.01325
                reference = thermo.flash3_TP(z, T_reference, P, max_iter=200)
                H_reference = thermo._flash3_mixture_enthalpy(T_reference, P, reference)

                T_ph, ph, residual = thermo.flash3_PH(
                    z,
                    P,
                    H_reference,
                    T_guess=T_reference + 8.0,
                    max_iter=200,
                )

                self.assertEqual(reference.phase_count, 3)
                self.assertAlmostEqual(T_ph, T_reference, delta=1e-3)
                self.assertEqual(ph.phase_count, 3)
                self.assertLess(abs(residual), 20.0)

    def test_vlle_pv_prefers_three_phase_temperature_branch_with_bad_guesses(self):
        thermo = create_thermodynamics(['water', 'ethanol', 'cyclohexane'], 'UNIFNIST')
        z = {'water': 0.30, 'ethanol': 0.20, 'cyclohexane': 0.50}
        P = 1.01325
        T_reference = 337.0
        reference = thermo.flash3_TP(z, T_reference, P, max_iter=200)

        self.assertEqual(reference.phase_count, 3)
        for T_guess in (330.0, 345.0, 360.0):
            with self.subTest(T_guess=T_guess):
                T_pv, pv = thermo.flash3_PV(
                    z,
                    P,
                    reference.vapor_fraction,
                    T_guess=T_guess,
                    max_iter=200,
                )

                self.assertAlmostEqual(T_pv, T_reference, delta=1e-3)
                self.assertEqual(pv.phase_count, 3)
                self.assertEqual(pv.status, reference.status)

    def test_vlle_ps_prefers_three_phase_entropy_root_with_bad_guesses(self):
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

        for method, components, z, T_reference in cases:
            with self.subTest(method=method):
                thermo = create_thermodynamics(components, method)
                P = 1.01325
                reference = thermo.flash3_TP(z, T_reference, P, max_iter=200)
                S_reference = thermo._flash3_mixture_entropy(T_reference, P, reference)

                for T_guess in (T_reference - 30.0, T_reference + 40.0):
                    with self.subTest(T_guess=T_guess):
                        T_ps, ps, residual = thermo.flash3_PS(
                            z,
                            P,
                            S_reference,
                            T_guess=T_guess,
                            max_iter=200,
                        )

                        self.assertEqual(reference.phase_count, 3)
                        self.assertAlmostEqual(T_ps, T_reference, delta=1e-3)
                        self.assertEqual(ps.phase_count, 3)
                        self.assertLess(abs(residual), 0.1)

    def test_vlle_tv_prefers_narrow_three_phase_pressure_branch(self):
        thermo = create_thermodynamics(['water', 'methanol', 'benzene'], 'UNIFNIST')
        z = {'water': 0.30, 'methanol': 0.40, 'benzene': 0.30}
        T = 330.0

        has_lle, x1, x2, _beta = thermo.liquid_liquid_equilibrium(
            thermo._normalize_phase_composition(z),
            T,
            max_iter=200,
            tol=1e-8,
        )
        self.assertTrue(has_lle)
        p_top = 0.5 * (
            thermo.bubble_point_P(thermo._normalize_phase_composition(x1), T)
            + thermo.bubble_point_P(thermo._normalize_phase_composition(x2), T)
        )
        P_reference = 0.992 * p_top
        reference = thermo.flash3_TP(z, T, P_reference, max_iter=200)
        self.assertEqual(reference.phase_count, 3)

        P_tv, tv = thermo.flash3_TV(
            z,
            T,
            reference.vapor_fraction,
            P_guess=1.15 * P_reference,
            max_iter=200,
        )

        self.assertAlmostEqual(P_tv, P_reference, delta=5e-5)
        self.assertEqual(tv.phase_count, 3)
        self.assertAlmostEqual(
            tv.vapor_fraction,
            reference.vapor_fraction,
            delta=5e-7,
        )

    def test_vlle_spec_flashes_recover_high_alcohol_non_vlle_state(self):
        thermo = create_thermodynamics(['water', 'ethanol', 'cyclohexane'], 'UNIFNIST')
        z = {'water': 0.20, 'ethanol': 0.60, 'cyclohexane': 0.20}
        T_reference = 337.0
        P_reference = 1.01325

        reference = thermo.flash3_TP(z, T_reference, P_reference, max_iter=200)
        self.assertEqual(reference.phase_count, 2)
        self.assertEqual(reference.status, 'ordinary_vle')

        T_pv, pv = thermo.flash3_PV(
            z,
            P_reference,
            reference.vapor_fraction,
            T_guess=T_reference + 5.0,
            max_iter=200,
        )
        self.assertAlmostEqual(T_pv, T_reference, places=6)
        self.assertEqual(pv.phase_count, 2)
        self.assertAlmostEqual(pv.vapor_fraction, reference.vapor_fraction, places=7)

        P_tv, tv = thermo.flash3_TV(
            z,
            T_reference,
            reference.vapor_fraction,
            P_guess=1.15 * P_reference,
            max_iter=200,
        )
        self.assertAlmostEqual(P_tv, P_reference, delta=5e-5)
        self.assertEqual(tv.phase_count, 2)
        self.assertAlmostEqual(tv.vapor_fraction, reference.vapor_fraction, places=7)

        H_reference = thermo._flash3_mixture_enthalpy(T_reference, P_reference, reference)
        T_ph, ph, residual = thermo.flash3_PH(
            z,
            P_reference,
            H_reference,
            T_guess=T_reference + 8.0,
            max_iter=200,
        )
        self.assertAlmostEqual(T_ph, T_reference, places=6)
        self.assertEqual(ph.phase_count, 2)
        self.assertAlmostEqual(ph.vapor_fraction, reference.vapor_fraction, places=7)
        self.assertLess(abs(residual), 1e-5)

        S_reference = thermo._flash3_mixture_entropy(T_reference, P_reference, reference)
        for T_guess in (T_reference - 30.0, T_reference + 40.0):
            T_ps, ps, entropy_residual = thermo.flash3_PS(
                z,
                P_reference,
                S_reference,
                T_guess=T_guess,
                max_iter=200,
            )
            self.assertAlmostEqual(T_ps, T_reference, places=6)
            self.assertEqual(ps.phase_count, 2)
            self.assertAlmostEqual(ps.vapor_fraction, reference.vapor_fraction, places=7)
            self.assertLess(abs(entropy_residual), 1e-6)

    def test_flash3_tpd_reseeds_missed_water_rich_vle_branch(self):
        thermo = create_thermodynamics(
            ['water', 'benzene', 'toluene'],
            'NRTL-PR',
        )
        z = {
            'water': 0.5398628992077538,
            'benzene': 0.2490280908487376,
            'toluene': 0.2111090099435087,
        }
        expected_vapor_fractions = {
            352.2: 0.8482717473,
            353.0: 0.8725868675,
            354.0: 0.9060517107,
            355.0: 0.9434663744,
            356.0: 0.9855375146,
        }

        previous_vapor_fraction = 0.0
        for T, expected in expected_vapor_fractions.items():
            with self.subTest(T=T):
                result = thermo.flash3_TP(z, T, 1.0, max_iter=200)

                self.assertEqual(result.status, 'ordinary_vle')
                self.assertEqual(result.phase_count, 2)
                self.assertAlmostEqual(result.vapor_fraction, expected, places=8)
                self.assertGreater(result.vapor_fraction, previous_vapor_fraction)
                self.assertGreater(result.x1['water'], 0.999)
                if T == 352.2:
                    self.assertNotIn('stability_seed', result.extra)
                else:
                    self.assertEqual(
                        result.extra['stability_seed'],
                        'feed_lle_tpd',
                    )
                    self.assertLess(result.extra['liquid_tpd'], 0.0)
                    self.assertLess(
                        result.extra['reduced_gibbs_delta'],
                        0.0,
                    )
                previous_vapor_fraction = result.vapor_fraction

        vapor = thermo.flash3_TP(z, 357.0, 1.0, max_iter=200)
        self.assertEqual(vapor.status, 'single_vapor')
        self.assertEqual(vapor.phase_count, 1)
        self.assertEqual(vapor.vapor_fraction, 1.0)
        self.assertNotIn('stability_seed', vapor.extra)

    def test_flash3_ph_recovers_efficient_expansion_across_seeded_vle_branch(self):
        thermo = create_thermodynamics(
            ['water', 'benzene', 'toluene'],
            'NRTL-PR',
        )
        z = {
            'water': 0.5398628992077538,
            'benzene': 0.2490280908487376,
            'toluene': 0.2111090099435087,
        }
        inlet_T = 573.15
        inlet_P = 90.0
        outlet_P = 1.0
        efficiency = 0.8

        inlet = thermo.flash3_TP(z, inlet_T, inlet_P, max_iter=200)
        inlet_H = thermo._flash3_mixture_enthalpy(inlet_T, inlet_P, inlet)
        inlet_S = thermo._flash3_mixture_entropy(inlet_T, inlet_P, inlet)
        isentropic_T, isentropic, entropy_residual = thermo.flash3_PS(
            z,
            outlet_P,
            inlet_S,
            T_guess=350.0,
            max_iter=200,
        )
        isentropic_H = thermo._flash3_mixture_enthalpy(
            isentropic_T, outlet_P, isentropic
        )
        target_H = inlet_H - efficiency * (inlet_H - isentropic_H)

        outlet_T, outlet, enthalpy_residual = thermo.flash3_PH(
            z,
            outlet_P,
            target_H,
            T_guess=isentropic_T + 20.0,
            max_iter=200,
        )
        outlet_H = thermo._flash3_mixture_enthalpy(outlet_T, outlet_P, outlet)
        recovered_efficiency = (inlet_H - outlet_H) / (inlet_H - isentropic_H)

        self.assertEqual(inlet.status, 'single_vapor')
        self.assertAlmostEqual(isentropic_T, 352.0901885, places=5)
        self.assertAlmostEqual(isentropic.vapor_fraction, 0.8450888582, places=8)
        self.assertLess(abs(entropy_residual), 1e-6)
        self.assertEqual(outlet.status, 'ordinary_vle')
        self.assertAlmostEqual(outlet_T, 353.6090869, places=5)
        self.assertAlmostEqual(outlet.vapor_fraction, 0.8925337283, places=8)
        self.assertGreater(outlet.x1['water'], 0.999)
        self.assertLess(abs(enthalpy_residual), 1e-4)
        self.assertAlmostEqual(recovered_efficiency, efficiency, places=10)



if __name__ == '__main__':
    unittest.main()
