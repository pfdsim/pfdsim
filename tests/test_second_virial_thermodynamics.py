import math
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from chemicals.virial import (
    BVirial_Abbott,
    BVirial_Pitzer_Curl,
    BVirial_Tsonopoulos_extended,
)

from physical_constants import R_J_MOL_K
from pfd_parser import ProcessFlowDiagram, parse_pfd
from simulator import Simulator
from thermodynamics_models.factory import create_thermodynamics
from thermodynamics_models.gamma_phi import (
    NRTLBVThermodynamics,
    UNIFACBVThermodynamics,
    UNIFDMDBVThermodynamics,
    UNIFNISTBVThermodynamics,
    UNIQUACBVThermodynamics,
)
from thermodynamics_models.hayden_oconnell import (
    HOCComponentParameters,
    HaydenOConnellSecondVirialProvider,
    hoc_association_group,
)
from thermodynamics_models.second_virial import (
    AbbottSecondVirialProvider,
    ChemicalAssociationSecondVirialVaporBackend,
    PitzerCurlSecondVirialProvider,
    SecondVirialVaporBackend,
    TsonopoulosSecondVirialProvider,
    create_second_virial_vapor_backend,
)
from thermodynamics_models.tsonopoulos_kij import (
    exact_tsonopoulos_kij_records,
)


def component_props(**overrides):
    values = {
        'Tc': 500.0,
        'Pc': 50.0,
        'Vc': 200.0,
        'omega': 0.2,
        'CAS': '999-99-9',
        'formula': 'C3H6O',
        'smiles': 'CC(=O)C',
        'dipole_moment': 2.7,
        'modified_radius_of_gyration': 2.1,
        'property_sources': {
            'dipole_moment': {
                'source': 'provided',
                'method': 'pfd_component_override',
                'quality': 1.0,
            },
            'modified_radius_of_gyration': {
                'source': 'provided',
                'method': 'pfd_component_override',
                'quality': 1.0,
            },
        },
    }
    values.update(overrides)

    def to_dict():
        return dict(values)

    return SimpleNamespace(**values, to_dict=to_dict)


class FixedSecondVirialProvider:
    name = 'FIXED-TEST'

    derivative_matrices_at_400 = {
        0: ((-1.0e-4, -1.5e-4), (-1.5e-4, -2.0e-4)),
        1: ((2.0e-7, 3.0e-7), (3.0e-7, 4.0e-7)),
        2: ((-5.0e-10, -7.0e-10), (-7.0e-10, -9.0e-10)),
    }

    def second_virial_matrix(self, T, order=0):
        delta_T = float(T) - 400.0
        base = self.derivative_matrices_at_400[0]
        first = self.derivative_matrices_at_400[1]
        second = self.derivative_matrices_at_400[2]
        if order == 0:
            return tuple(tuple(
                base[i][j]
                + first[i][j] * delta_T
                + 0.5 * second[i][j] * delta_T**2
                for j in range(2)
            ) for i in range(2))
        if order == 1:
            return tuple(tuple(
                first[i][j] + second[i][j] * delta_T
                for j in range(2)
            ) for i in range(2))
        return second


class SecondVirialBackendTests(unittest.TestCase):
    def test_fugacity_and_residual_properties_follow_truncated_virial_equations(self):
        backend = SecondVirialVaporBackend(['A', 'B'], FixedSecondVirialProvider())
        composition = {'A': 0.25, 'B': 0.75}
        T = 400.0
        P = 8.0
        fractions = (0.25, 0.75)

        def mixture(order):
            matrix = FixedSecondVirialProvider.derivative_matrices_at_400[order]
            return sum(
                fractions[i] * fractions[j] * matrix[i][j]
                for i in range(2)
                for j in range(2)
            )

        phi = backend.fugacity_coefficients(T, P, composition)
        B_mix = mixture(0)
        factor = P * 1.0e5 / (R_J_MOL_K * T)
        for i, component in enumerate(('A', 'B')):
            partial = 2.0 * sum(
                fractions[j]
                * FixedSecondVirialProvider.derivative_matrices_at_400[0][i][j]
                for j in range(2)
            ) - B_mix
            self.assertAlmostEqual(phi[component], math.exp(factor * partial))

        self.assertAlmostEqual(
            backend.departure_gibbs(T, P, composition),
            P * 1.0e5 * B_mix,
        )
        self.assertAlmostEqual(
            backend.departure_enthalpy(T, P, composition),
            P * 1.0e5 * (B_mix - T * mixture(1)),
        )
        self.assertAlmostEqual(
            backend.departure_entropy(T, P, composition),
            -P * 1.0e5 * mixture(1),
        )
        self.assertAlmostEqual(
            backend.departure_heat_capacity(T, P, composition),
            -P * 1.0e5 * T * mixture(2),
        )
        expected_Z = 1.0 + B_mix * P * 1.0e5 / (R_J_MOL_K * T)
        self.assertAlmostEqual(
            backend.compressibility_factor(T, P, composition),
            expected_Z,
        )
        self.assertAlmostEqual(
            backend.molar_volume(T, P, composition),
            expected_Z * R_J_MOL_K * T / (P * 1.0e5) * 1.0e6,
        )

    def test_custom_provider_is_independent_of_activity_model(self):
        for method in ('NRTL-BV', 'UNIQUAC-BV'):
            with self.subTest(method=method):
                thermo = create_thermodynamics(
                    ['ethanol', 'water'],
                    method,
                    second_virial_provider=FixedSecondVirialProvider(),
                )
                self.assertIs(thermo.vapor_eos.provider.__class__, FixedSecondVirialProvider)
                self.assertEqual(thermo.vapor_eos.correlation, 'FIXED-TEST')

    def test_activity_model_vapor_properties_include_virial_departures(self):
        composition = {'ethanol': 0.25, 'water': 0.75}
        corrected = create_thermodynamics(
            ['ethanol', 'water'],
            'NRTL-BV',
            second_virial_provider=FixedSecondVirialProvider(),
        )
        ideal_vapor = create_thermodynamics(['ethanol', 'water'], 'NRTL')
        T = 400.0
        P = 8.0
        backend = corrected.vapor_eos

        self.assertAlmostEqual(
            corrected.mixture_enthalpy(composition, T, 1.0, P=P)
            - ideal_vapor.mixture_enthalpy(composition, T, 1.0, P=P),
            backend.departure_enthalpy(T, P, composition),
        )
        self.assertAlmostEqual(
            corrected.mixture_entropy(composition, T, 1.0, P=P)
            - ideal_vapor.mixture_entropy(composition, T, 1.0, P=P),
            backend.departure_entropy(T, P, composition),
        )
        self.assertAlmostEqual(
            corrected.mixture_Cp(composition, T, 1.0, P=P)
            - ideal_vapor.mixture_Cp(composition, T, 1.0, P=P),
            backend.departure_heat_capacity(T, P, composition),
            places=6,
        )
        expected_volume = backend.molar_volume(T, P, composition) / 1000.0
        self.assertAlmostEqual(
            corrected.vapor_molar_volume_for_density(T, P, composition),
            expected_volume,
        )
        self.assertAlmostEqual(
            corrected.mixture_molar_density(composition, T, P),
            1.0 / expected_volume,
        )


class TsonopoulosProviderTests(unittest.TestCase):
    def test_published_table_contains_latest_water_assessment(self):
        records = exact_tsonopoulos_kij_records()
        water = '7732-18-5'
        expected = {
            '74-82-8': (0.319, 0.014),      # methane
            '110-54-3': (0.494, 0.017),     # n-hexane
            '67-56-1': (0.012, 0.008),      # methanol
            '64-17-5': (0.05, 0.03),        # ethanol
            '7664-41-7': (-0.171, 0.013),   # ammonia
            '67-64-1': (-0.020, 0.030),     # acetone
        }
        for other, (value, uncertainty) in expected.items():
            with self.subTest(other=other):
                record = records[frozenset((water, other))]
                self.assertEqual(record.value, value)
                self.assertEqual(record.uncertainty, uncertainty)
                self.assertIn('Plyasunov and Shock (2003)', record.source)

    def test_factory_resolves_dipole_for_supported_polar_family(self):
        thermo = create_thermodynamics(['ethanol', 'water'], 'NRTL-BV')
        provider = thermo.vapor_eos.provider
        self.assertEqual(provider.species_types['ethanol'], 'alkanol')
        self.assertEqual(provider.species_types['water'], 'water')
        self.assertEqual(
            provider.dipole_results['ethanol'].method,
            'cccbdb_experimental_dipole',
        )
        self.assertAlmostEqual(thermo.props['ethanol'].dipole_moment, 1.44)

    def test_pure_polar_coefficient_uses_resolved_dipole_correction(self):
        props = component_props()
        provider = TsonopoulosSecondVirialProvider(['acetone'], {'acetone': props})
        T = 430.0
        matrix = provider.second_virial_matrix(T)
        tc, pc_pa, _vc, omega, a_value, b_value = provider._parameters[0]
        expected = BVirial_Tsonopoulos_extended(
            T,
            tc,
            pc_pa,
            omega,
            a=a_value,
            b=b_value,
        )
        uncorrected = BVirial_Tsonopoulos_extended(T, tc, pc_pa, omega)
        self.assertEqual(provider.species_types['acetone'], 'ketone')
        self.assertAlmostEqual(matrix[0][0], expected)
        self.assertNotAlmostEqual(matrix[0][0], uncorrected, places=10)

    def test_published_cross_rules_and_polar_pair_rules(self):
        props = {
            'ketone': component_props(Tc=510.0, Pc=48.0, Vc=210.0, omega=0.25),
            'alkane': component_props(
                Tc=425.0,
                Pc=36.0,
                Vc=255.0,
                omega=0.19,
                formula='C4H10',
                smiles='CCCC',
                dipole_moment=0.0,
            ),
            'ether': component_props(
                Tc=470.0,
                Pc=44.0,
                Vc=225.0,
                omega=0.23,
                formula='C4H10O',
                smiles='CCOCC',
                dipole_moment=1.2,
            ),
        }
        provider = TsonopoulosSecondVirialProvider(
            ['ketone', 'alkane', 'ether'],
            props,
        )
        tc_ij, pc_ij, omega_ij = provider._cross_parameters(0, 1)
        tc_i, pc_i, vc_i, omega_i, _a_i, _b_i = provider._parameters[0]
        tc_j, pc_j, vc_j, omega_j, _a_j, _b_j = provider._parameters[1]
        record = provider.binary_kij_records[frozenset(('ketone', 'alkane'))]
        self.assertEqual(record.value, 0.13)
        self.assertEqual(record.method, 'class default')
        self.assertAlmostEqual(tc_ij, math.sqrt(tc_i * tc_j) * 0.87)
        self.assertAlmostEqual(
            pc_ij,
            4.0 * tc_ij * (pc_i * vc_i / tc_i + pc_j * vc_j / tc_j)
            / (vc_i ** (1.0 / 3.0) + vc_j ** (1.0 / 3.0)) ** 3,
        )
        self.assertAlmostEqual(omega_ij, 0.5 * (omega_i + omega_j))
        self.assertEqual(provider._cross_polar_parameters(0, 1), (0.0, 0.0))

        a_ketone = provider._parameters[0][4]
        b_ketone = provider._parameters[0][5]
        a_ether = provider._parameters[2][4]
        b_ether = provider._parameters[2][5]
        self.assertEqual(
            provider._cross_polar_parameters(0, 2),
            (0.5 * (a_ketone + a_ether), 0.5 * (b_ketone + b_ether)),
        )

    def test_latest_fitted_water_binary_value_has_provenance(self):
        props = {
            'ethanol': component_props(
                Tc=514.71,
                Pc=62.68,
                Vc=168.6,
                omega=0.644,
                CAS='64-17-5',
                formula='C2H6O',
                smiles='CCO',
                dipole_moment=1.44,
            ),
            'water': component_props(
                Tc=647.096,
                Pc=220.64,
                Vc=55.95,
                omega=0.344,
                CAS='7732-18-5',
                formula='H2O',
                smiles='O',
                dipole_moment=1.85,
            ),
        }
        provider = TsonopoulosSecondVirialProvider(
            ['ethanol', 'water'],
            props,
            allow_online=False,
        )
        record = provider.binary_kij_records[
            frozenset(('ethanol', 'water'))
        ]
        self.assertEqual(record.value, 0.05)
        self.assertEqual(record.uncertainty, 0.03)
        self.assertEqual(record.temperature_range_K, (400.0, 525.0))
        self.assertIn('Plyasunov and Shock (2003)', record.source)
        polar = provider.polar_parameter_records['ethanol']
        self.assertEqual((polar.a, polar.b), (0.0878, 0.0578))
        self.assertIn('Tsonopoulos and Dymond (1997)', polar.source)
        tc_ij, _pc_ij, _omega_ij = provider._cross_parameters(0, 1)
        self.assertAlmostEqual(tc_ij, math.sqrt(514.71 * 647.096) * 0.95)

    def test_explicit_kij_override_precedes_published_binary(self):
        props = {
            'ethanol': component_props(
                CAS='64-17-5', formula='C2H6O', smiles='CCO'
            ),
            'water': component_props(
                CAS='7732-18-5', formula='H2O', smiles='O'
            ),
        }
        provider = TsonopoulosSecondVirialProvider(
            ['ethanol', 'water'],
            props,
            binary_kijs={('ethanol', 'water'): 0.08},
            allow_online=False,
        )
        record = provider.binary_kij_records[
            frozenset(('ethanol', 'water'))
        ]
        self.assertEqual(record.value, 0.08)
        self.assertEqual(record.method, 'explicit override')

    def test_pfd_kij_override_reaches_tsonopoulos_provider(self):
        simulator = Simulator.from_string(
            'THERMO_METHOD: NRTL-BV\n'
            'ONLINE_LOOKUP: false\n'
            'COMPONENTS:\n'
            '    EtOH | ethanol\n'
            '    H2O | water\n'
            'INTERACTION_PARAMETERS:\n'
            '    EtOH/H2O | model=TSONOPOULOS-1974, k_ij=0.08\n'
        ).initialize()
        provider = simulator.thermo.vapor_eos.provider
        record = provider.binary_kij_records[frozenset(('EtOH', 'H2O'))]
        self.assertEqual(record.value, 0.08)
        self.assertEqual(record.method, 'explicit override')
        self.assertIn(
            'model=TSONOPOULOS-1974, k_ij=0.08',
            simulator.pfd.to_pfd(),
        )

    def test_inactive_provider_override_is_ignored_with_warning(self):
        simulator = Simulator.from_string(
            'THERMO_METHOD: NRTL-BV\n'
            'ONLINE_LOOKUP: false\n'
            'COMPONENTS:\n'
            '    EtOH | ethanol\n'
            '    H2O | water\n'
            'INTERACTION_PARAMETERS:\n'
            '    EtOH/H2O | model=HOC, eta=1.23\n'
        ).initialize()
        self.assertTrue(any(
            'Ignoring HOC vapor interaction parameters' in warning
            for warning in simulator.thermo.warnings
        ))

    def test_predictive_volume_rules_cover_unlisted_alkanes(self):
        first = component_props(
            CAS='', formula='C9H20', smiles='CCCCCCCCC', Vc=400.0
        )
        second = component_props(
            CAS='', formula='C12H26', smiles='CCCCCCCCCCCC', Vc=500.0
        )
        provider = TsonopoulosSecondVirialProvider(
            ['nonane', 'dodecane'],
            {'nonane': first, 'dodecane': second},
            allow_online=False,
        )
        expected = 1.0 - (
            2.0 * (400.0 * 500.0) ** (1.0 / 6.0)
            / (400.0 ** (1.0 / 3.0) + 500.0 ** (1.0 / 3.0))
        ) ** 3
        record = provider.binary_kij_records[
            frozenset(('nonane', 'dodecane'))
        ]
        self.assertAlmostEqual(record.value, expected)
        self.assertEqual(record.method, 'critical-volume correlation')

        water = component_props(
            CAS='7732-18-5', formula='H2O', smiles='O', Vc=55.95
        )
        aqueous = TsonopoulosSecondVirialProvider(
            ['water', 'nonane'],
            {'water': water, 'nonane': first},
            allow_online=False,
        )
        water_record = aqueous.binary_kij_records[
            frozenset(('water', 'nonane'))
        ]
        self.assertAlmostEqual(
            water_record.value,
            0.6114 - 2.7135 / math.sqrt(400.0),
        )


class NonpolarCorrespondingStatesProviderTests(unittest.TestCase):
    def test_pitzer_curl_and_abbott_match_chemicals_with_derivatives(self):
        props = component_props(
            Tc=425.2,
            Pc=38.0,
            Vc=255.0,
            omega=0.193,
            formula='C4H10',
            smiles='CC(C)C',
            dipole_moment=0.0,
        )
        cases = (
            (PitzerCurlSecondVirialProvider, BVirial_Pitzer_Curl),
            (AbbottSecondVirialProvider, BVirial_Abbott),
        )
        for provider_type, correlation in cases:
            with self.subTest(provider=provider_type.name):
                provider = provider_type(['isobutane'], {'isobutane': props})
                for order in (0, 1, 2):
                    self.assertAlmostEqual(
                        provider.second_virial_matrix(510.0, order)[0][0],
                        correlation(
                            510.0,
                            425.2,
                            38.0e5,
                            0.193,
                            order=order,
                        ),
                    )

    def test_nonpolar_providers_do_not_resolve_dipoles(self):
        props = component_props(smiles='CC(=O)C', dipole_moment=2.7)
        for provider_type in (
            PitzerCurlSecondVirialProvider,
            AbbottSecondVirialProvider,
        ):
            with self.subTest(provider=provider_type.name):
                provider = provider_type(['acetone'], {'acetone': props})
                self.assertEqual(provider.species_types, {'acetone': 'normal'})
                self.assertEqual(provider.dipole_results, {})


class HaydenOConnellProviderTests(unittest.TestCase):
    def test_pfd_pure_and_binary_eta_overrides_reach_hoc_provider(self):
        simulator = Simulator.from_string(
            'THERMO_METHOD: NRTL-HOC\n'
            'ONLINE_LOOKUP: false\n'
            'COMPONENTS:\n'
            '    EtOH | ethanol | HOC_eta=1.7, R_prime=1.7\n'
            '    H2O | water | R_prime=0.8\n'
            'INTERACTION_PARAMETERS:\n'
            '    EtOH/H2O | model=HAYDEN-OCONNELL, eta_ij=1.23\n'
        ).initialize()
        provider = simulator.thermo.vapor_eos.provider
        self.assertEqual(provider.pure_eta_overrides, {'EtOH': 1.7})
        self.assertEqual(provider.parameters[0].eta, 1.7)
        self.assertEqual(
            simulator.thermo.props['EtOH'].property_sources['hoc_eta']['method'],
            'pfd_component_override',
        )
        self.assertEqual(
            provider.binary_eta_overrides[frozenset(('EtOH', 'H2O'))],
            1.23,
        )
        epsilon, _sigma, _omega, _dipole = provider._cross_parameters(
            provider.parameters[0],
            provider.parameters[1],
        )
        self.assertEqual(provider._eta_for_indices(0, 1, epsilon), 1.23)

    def test_group_classifier_uses_documented_single_groups(self):
        cases = {
            'CCO': 'hydroxyl',
            'CC(=O)C': 'ketone',
            'CC=O': 'aldehyde',
            'COC=O': 'formate',
            'CCOC(=O)C': 'ester',
            'CC#N': 'nitrile',
            'C[N+](=O)[O-]': 'nitro',
            'c1ccccc1O': 'phenyl hydroxyl',
            'NCC(=O)O': 'multifunctional',
        }
        for smiles, expected in cases.items():
            with self.subTest(smiles=smiles):
                props = component_props(CAS='', smiles=smiles)
                self.assertEqual(hoc_association_group('test', props), expected)

    def test_multifunctional_eta_fallback_is_auditable(self):
        props = component_props(
            CAS='',
            formula='C2H5NO2',
            smiles='NCC(=O)O',
        )
        provider = HaydenOConnellSecondVirialProvider(
            ['glycine'],
            {'glycine': props},
        )
        self.assertEqual(provider.parameters[0].association_group, 'multifunctional')
        self.assertEqual(provider.parameters[0].eta, 0.0)
        self.assertTrue(any('no additive multifunctional rule' in item for item in provider.warnings))

    def test_water_and_alcohol_share_table_iv_hydroxyl_group(self):
        water = component_props(
            CAS='7732-18-5',
            formula='H2O',
            smiles='O',
        )
        ethanol = component_props(
            CAS='64-17-5',
            formula='C2H6O',
            smiles='CCO',
        )
        self.assertEqual(hoc_association_group('water', water), 'hydroxyl')
        self.assertEqual(hoc_association_group('ethanol', ethanol), 'hydroxyl')
        first = HOCComponentParameters(
            300.0, 100.0, 0.1, 1.8, 1.55, 'hydroxyl', '7732-18-5'
        )
        second = HOCComponentParameters(
            350.0, 120.0, 0.2, 1.4, 1.55, 'hydroxyl', '64-17-5'
        )
        self.assertEqual(
            HaydenOConnellSecondVirialProvider._eta_for_pair(
                first,
                second,
                320.0,
            ),
            1.55,
        )

    def test_nonpolar_pure_coefficient_matches_paper_equations(self):
        props = component_props(
            Tc=425.2,
            Pc=38.0 * 1.01325,
            CAS='',
            formula='C4H10',
            smiles='CC(C)C',
            dipole_moment=0.0,
            modified_radius_of_gyration=2.0,
        )
        provider = HaydenOConnellSecondVirialProvider(
            ['isobutane'],
            {'isobutane': props},
        )
        parameter = provider.parameters[0]
        T = 510.0
        inverse_reduced = parameter.epsilon_K / T - 1.6 * parameter.omega_prime
        b0 = 1.2618 * parameter.sigma3_A3
        expected = b0 * (
            0.94
            - 1.47 * inverse_reduced
            - 0.85 * inverse_reduced**2
            + 1.015 * inverse_reduced**3
            - 0.3 * math.exp(1.99 * parameter.epsilon_K / T)
        ) * 1.0e-6
        self.assertEqual(parameter.association_group, 'hydrocarbon')
        self.assertAlmostEqual(provider.second_virial_matrix(T)[0][0], expected)

    def test_temperature_derivatives_are_exact_derivatives_of_same_B(self):
        props = component_props(
            CAS='',
            smiles='CC(=O)C',
            modified_radius_of_gyration=2.1,
        )
        provider = HaydenOConnellSecondVirialProvider(
            ['acetone'],
            {'acetone': props},
        )
        T = 430.0
        step = 0.05
        lower = provider.second_virial_matrix(T - step)[0][0]
        center = provider.second_virial_matrix(T)[0][0]
        upper = provider.second_virial_matrix(T + step)[0][0]
        numerical_first = (upper - lower) / (2.0 * step)
        numerical_second = (upper - 2.0 * center + lower) / step**2
        analytic_first = provider.second_virial_matrix(T, 1)[0][0]
        analytic_second = provider.second_virial_matrix(T, 2)[0][0]
        self.assertAlmostEqual(analytic_first, numerical_first, delta=abs(analytic_first) * 2e-7)
        self.assertAlmostEqual(analytic_second, numerical_second, delta=abs(analytic_second) * 2e-5)

    def test_fortran_physical_and_chemical_terms_are_reproduced(self):
        T = 360.0
        epsilon = 420.0
        sigma3 = 115.0
        omega_prime = 0.12
        reduced_dipole = 1.4
        eta = 0.9
        inverse_reduced = epsilon / T - 1.6 * omega_prime
        reduced_prime = reduced_dipole - 0.25
        b0 = 1.2618 * sigma3
        free = b0 * (
            0.94
            - 1.47 * inverse_reduced
            - 0.85 * inverse_reduced**2
            + 1.015 * inverse_reduced**3
            - reduced_prime * (
                0.75
                - 3.0 * inverse_reduced
                + 2.1 * inverse_reduced**2
                + 2.1 * inverse_reduced**3
            )
        )
        physical_bound = (
            b0
            * (-0.3 - 0.05 * reduced_dipole)
            * math.exp((1.99 + 0.2 * reduced_dipole**2) * epsilon / T)
        )
        chemical = (
            b0
            * math.exp(eta * (650.0 / (epsilon + 300.0) - 4.27))
            * (1.0 - math.exp(eta * 1500.0 / T))
        )
        expected = (free + physical_bound + chemical) * 1.0e-6
        actual = HaydenOConnellSecondVirialProvider._coefficient_jet(
            T,
            epsilon,
            sigma3,
            omega_prime,
            reduced_dipole,
            eta,
        ).value
        self.assertAlmostEqual(actual, expected)

    def test_representative_group_solvation_parameter_is_selected(self):
        aromatic = HOCComponentParameters(
            300.0, 100.0, 0.1, 0.0, 0.0, 'benzene', '71-43-2'
        )
        ketone = HOCComponentParameters(350.0, 120.0, 0.2, 2.5, 0.9, 'ketone')
        self.assertEqual(
            HaydenOConnellSecondVirialProvider._eta_for_pair(
                aromatic,
                ketone,
                320.0,
            ),
            0.50,
        )

    def test_table_iv_labels_transfer_but_unlabelled_compounds_do_not(self):
        benzene = HOCComponentParameters(
            300.0, 100.0, 0.1, 0.0, 0.0, 'benzene', '71-43-2'
        )
        generic_ketone = HOCComponentParameters(
            350.0, 120.0, 0.2, 2.5, 0.9, 'ketone', '999-99-9'
        )
        generic_ether = HOCComponentParameters(
            350.0, 120.0, 0.2, 1.2, 0.0, 'ether', '999-99-8'
        )
        self.assertEqual(
            HaydenOConnellSecondVirialProvider._eta_for_pair(
                benzene,
                generic_ketone,
                320.0,
            ),
            0.50,
        )
        self.assertEqual(
            HaydenOConnellSecondVirialProvider._eta_for_pair(
                benzene,
                generic_ether,
                320.0,
            ),
            0.0,
        )

    def test_corrected_dichloromethane_ketone_table_v_row(self):
        dichloromethane = HOCComponentParameters(
            300.0, 100.0, 0.1, 1.6, 0.0,
            'dichloromethane', '75-09-2',
        )
        ketone = HOCComponentParameters(
            350.0, 120.0, 0.2, 2.5, 0.9, 'ketone', '999-99-9'
        )
        self.assertEqual(
            HaydenOConnellSecondVirialProvider._eta_for_pair(
                dichloromethane,
                ketone,
                320.0,
            ),
            0.92,
        )

    def test_acid_chemical_theory_separates_total_B_from_physical_B_and_Kp(self):
        props = component_props(
            CAS='64-19-7',
            formula='C2H4O2',
            smiles='CC(=O)O',
            dipole_moment=1.7,
            modified_radius_of_gyration=2.5,
        )
        provider = HaydenOConnellSecondVirialProvider(
            ['acetic acid'],
            {'acetic acid': props},
        )
        T = 391.1
        total = provider.second_virial_matrix(T)[0][0]
        physical = provider.physical_second_virial_matrix(T)[0][0]
        equilibrium = provider.association_constant_matrix(T)[0][0]

        self.assertEqual(provider.associating_components, ('acetic acid',))
        self.assertTrue(provider.has_chemical_theory)
        self.assertLess(total, physical)
        self.assertAlmostEqual(
            equilibrium,
            -(total - physical) * 1.0e5 / (R_J_MOL_K * T),
        )

    def test_multiple_acids_use_same_group_eta_and_cross_dimer_symmetry(self):
        props = {
            'acetic acid': component_props(
                Tc=592.0,
                Pc=57.9,
                CAS='64-19-7',
                formula='C2H4O2',
                smiles='CC(=O)O',
                dipole_moment=1.7,
                modified_radius_of_gyration=2.5,
            ),
            'formic acid': component_props(
                Tc=588.0,
                Pc=58.1,
                CAS='64-18-6',
                formula='CH2O2',
                smiles='O=CO',
                dipole_moment=1.4,
                modified_radius_of_gyration=1.7,
            ),
        }
        provider = HaydenOConnellSecondVirialProvider(
            ['acetic acid', 'formic acid'],
            props,
        )
        epsilon, _sigma, _omega, _dipole = provider._cross_parameters(
            provider.parameters[0],
            provider.parameters[1],
        )
        self.assertEqual(
            provider._eta_for_pair(
                provider.parameters[0],
                provider.parameters[1],
                epsilon,
            ),
            4.5,
        )

        T = 390.0
        _free, association = provider._pair_terms(T, 0, 1)
        equilibrium = provider.association_constant_matrix(T)[0][1]
        self.assertAlmostEqual(
            equilibrium,
            -2.0 * association.value * 1.0e5 / (R_J_MOL_K * T),
        )


class HOCChemicalTheoryBackendTests(unittest.TestCase):
    def test_identical_acids_preserve_fugacity_through_infinite_dilution(self):
        provider = SimpleNamespace(
            name='symmetric acids',
            associating_components=('A', 'B'),
            physical_second_virial_matrix=lambda T, order=0: ((0., 0.), (0., 0.)),
            association_constant_matrix=lambda T, order=0: (
                ((10., 20.), (20., 10.)) if order == 0
                else ((0., 0.), (0., 0.))
            ),
        )
        expected_phi = 2.0 / (1.0 + math.sqrt(41.0))
        expected_total = 0.5 * (1.0 + 1.0 / math.sqrt(41.0))
        for compiled in (True, False):
            backend = ChemicalAssociationSecondVirialVaporBackend(('A', 'B'), provider)
            context = nullcontext() if compiled else patch(
                'compiled_vdm.solve_n_acid_true_moles', return_value=None,
            )
            with context:
                for fraction in (0., 1e-100, 1e-15, 1e-14, 1e-12, 0.3, 1.):
                    with self.subTest(compiled=compiled, fraction=fraction):
                        state = backend.association_state(
                            400., 1., {'A': 1. - fraction, 'B': fraction},
                        )
                        for phi in state['chemical_phi'].values():
                            self.assertAlmostEqual(phi, expected_phi, delta=1e-9)
                        self.assertAlmostEqual(
                            state['physical_moles_per_nominal'], expected_total,
                            delta=1e-9,
                        )

    def test_fallback_spans_zero_weak_and_strong_association(self):
        solve = ChemicalAssociationSecondVirialVaporBackend._solve_association_fallback
        for kappa in (0., 1e-12, 1e-9, 1e-6, 1., 10., 1e6, 1e12):
            with self.subTest(kappa=kappa):
                monomers, extents, total = solve([1.], 0., [0], [0], [kappa])
                expected_monomer = 1.0 / math.sqrt(1.0 + 4.0 * kappa)
                self.assertAlmostEqual(monomers[0] / expected_monomer, 1., delta=1e-8)
                self.assertAlmostEqual(monomers[0] + 2. * extents[0], 1., delta=1e-8)
                self.assertAlmostEqual(total, monomers[0] + extents[0], delta=1e-8)

    def test_fallback_preserves_relative_balance_of_strongly_bound_trace_acid(self):
        solve = ChemicalAssociationSecondVirialVaporBackend._solve_association_fallback
        nominal = [0.7, 1e-30]
        kappa = [10., 1e12, 1e12]
        monomers, extents, total = solve(nominal, 0.3, [0, 0, 1], [0, 1, 1], kappa)
        recovered = [
            monomers[0] + 2. * extents[0] + extents[1],
            monomers[1] + 2. * extents[2] + extents[1],
        ]
        for actual, expected in zip(recovered, nominal):
            self.assertAlmostEqual(actual / expected, 1., delta=1e-8)
        self.assertAlmostEqual(total, 0.3 + sum(monomers) + sum(extents), delta=1e-8)

    @staticmethod
    def _acid_provider(components=('acetic acid',)):
        available = {
            'acetic acid': component_props(
                Tc=592.0,
                Pc=57.9,
                CAS='64-19-7',
                formula='C2H4O2',
                smiles='CC(=O)O',
                dipole_moment=1.7,
                modified_radius_of_gyration=2.5,
            ),
            'formic acid': component_props(
                Tc=588.0,
                Pc=58.1,
                CAS='64-18-6',
                formula='CH2O2',
                smiles='O=CO',
                dipole_moment=1.4,
                modified_radius_of_gyration=1.7,
            ),
            'inert': component_props(
                Tc=425.0,
                Pc=38.0,
                CAS='',
                formula='C4H10',
                smiles='CCCC',
                dipole_moment=0.0,
                modified_radius_of_gyration=2.0,
            ),
        }
        provider = HaydenOConnellSecondVirialProvider(
            list(components),
            {component: available[component] for component in components},
        )
        return provider

    def test_backend_factory_selects_chemical_theory_only_for_acids(self):
        acid = self._acid_provider()
        acid_backend = create_second_virial_vapor_backend(
            acid.components,
            acid,
        )
        self.assertIsInstance(
            acid_backend,
            ChemicalAssociationSecondVirialVaporBackend,
        )

        nonacid = self._acid_provider(('inert',))
        nonacid_backend = create_second_virial_vapor_backend(
            nonacid.components,
            nonacid,
        )
        self.assertIs(type(nonacid_backend), SecondVirialVaporBackend)

    def test_chemical_theory_recovers_total_B_at_low_pressure(self):
        provider = self._acid_provider(('acetic acid', 'formic acid', 'inert'))
        backend = create_second_virial_vapor_backend(provider.components, provider)
        composition = {
            'acetic acid': 0.35,
            'formic acid': 0.25,
            'inert': 0.40,
        }
        T = 390.0
        P = 1.0e-7
        Z = backend.compressibility_factor(T, P, composition)
        recovered = (Z - 1.0) * R_J_MOL_K * T / (P * 1.0e5)
        fractions = tuple(composition[component] for component in provider.components)
        matrix = provider.second_virial_matrix(T)
        expected = sum(
            fractions[i] * fractions[j] * matrix[i][j]
            for i in range(len(fractions))
            for j in range(len(fractions))
        )
        self.assertAlmostEqual(recovered, expected, delta=abs(expected) * 2.0e-5)

    def test_multiple_acid_state_contains_all_homo_and_heterodimers(self):
        provider = self._acid_provider(('acetic acid', 'formic acid', 'inert'))
        backend = create_second_virial_vapor_backend(provider.components, provider)
        composition = {
            'acetic acid': 0.35,
            'formic acid': 0.25,
            'inert': 0.40,
        }
        state = backend.association_state(390.0, 1.01325, composition)
        self.assertEqual(
            set(state['extents']),
            {
                ('acetic acid', 'acetic acid'),
                ('acetic acid', 'formic acid'),
                ('formic acid', 'formic acid'),
            },
        )
        self.assertTrue(all(value > 0.0 for value in state['extents'].values()))
        self.assertGreater(state['physical_moles_per_nominal'], 0.5)
        self.assertLess(state['physical_moles_per_nominal'], 1.0)

    def test_fugacity_departures_and_volume_use_one_association_state(self):
        provider = self._acid_provider(('acetic acid', 'inert'))
        backend = create_second_virial_vapor_backend(provider.components, provider)
        composition = {'acetic acid': 0.7, 'inert': 0.3}
        T = 391.1
        P = 1.01325
        phi = backend.fugacity_coefficients(T, P, composition)
        expected_gibbs = R_J_MOL_K * T * sum(
            composition[component] * math.log(phi[component])
            for component in composition
        )
        self.assertAlmostEqual(
            backend.departure_gibbs(T, P, composition),
            expected_gibbs,
        )
        self.assertTrue(all(value > 0.0 for value in phi.values()))
        self.assertGreater(phi['inert'], 1.0)
        compressibility = backend.compressibility_factor(T, P, composition)
        self.assertGreater(compressibility, 0.0)

        pressure_step = 1.0e-5
        def dimensionless_gibbs_at_pressure(pressure):
            values = backend.fugacity_coefficients(
                T,
                pressure,
                composition,
            )
            return sum(
                composition[component] * math.log(values[component])
                for component in composition
            )

        compressibility_from_gibbs = 1.0 + P * (
            dimensionless_gibbs_at_pressure(P + pressure_step)
            - dimensionless_gibbs_at_pressure(P - pressure_step)
        ) / (2.0 * pressure_step)
        self.assertAlmostEqual(
            compressibility,
            compressibility_from_gibbs,
            places=8,
        )

        step = 0.01
        def dimensionless_gibbs(temperature):
            return sum(
                composition[component]
                * math.log(
                    backend.fugacity_coefficients(
                        temperature,
                        P,
                        composition,
                    )[component]
                )
                for component in composition
            )

        numerical_enthalpy = -R_J_MOL_K * T**2 * (
            dimensionless_gibbs(T + step)
            - dimensionless_gibbs(T - step)
        ) / (2.0 * step)
        self.assertAlmostEqual(
            backend.departure_enthalpy(T, P, composition),
            numerical_enthalpy,
            delta=abs(numerical_enthalpy) * 2.0e-6,
        )


class SecondVirialActivityVariantTests(unittest.TestCase):
    def test_all_activity_bv_variants_are_available(self):
        expected = {
            'NRTL-BV': NRTLBVThermodynamics,
            'UNIQUAC-BV': UNIQUACBVThermodynamics,
            'UNIFAC-BV': UNIFACBVThermodynamics,
            'UNIFDMD-BV': UNIFDMDBVThermodynamics,
            'UNIFNIST-BV': UNIFNISTBVThermodynamics,
        }
        provider = FixedSecondVirialProvider()
        for method, expected_type in expected.items():
            with self.subTest(method=method):
                thermo = create_thermodynamics(
                    ['ethanol', 'water'],
                    method,
                    second_virial_provider=provider,
                )
                self.assertIsInstance(thermo, expected_type)
                phi = thermo.fugacity_coefficients(
                    350.0,
                    5.0,
                    {'ethanol': 0.5, 'water': 0.5},
                )
                self.assertTrue(all(value > 0.0 for value in phi.values()))

    def test_pfd_correlation_option_round_trips_and_reaches_backend(self):
        source = (
            'PROCESS: BV syntax\n'
            'THERMO_METHOD: NRTL-BV | correlation=Tsonopoulos-1974\n'
            'COMPONENTS:\n'
            '    methanol | Methanol\n'
        )
        pfd = parse_pfd(source)
        self.assertEqual(pfd.metadata.thermo_method, 'NRTL-BV')
        self.assertEqual(
            pfd.metadata.thermo_options,
            {'correlation': 'TSONOPOULOS'},
        )
        restored = ProcessFlowDiagram.from_dict(pfd.to_dict())
        self.assertEqual(restored.metadata.thermo_options, pfd.metadata.thermo_options)
        self.assertIn(
            'THERMO_METHOD: NRTL-BV | correlation=TSONOPOULOS',
            restored.to_pfd(),
        )

        simulator = Simulator(restored).initialize()
        self.assertEqual(simulator.thermo.vapor_eos.correlation, 'TSONOPOULOS')
        self.assertEqual(simulator.thermo_options, {'correlation': 'TSONOPOULOS'})

    def test_selectable_correlations_reach_their_providers(self):
        cases = {
            'Pitzer_Curl': ('PITZER-CURL', PitzerCurlSecondVirialProvider),
            'Abbott-Lee-Kesler': ('ABBOTT', AbbottSecondVirialProvider),
            'HOC': ('HOC', HaydenOConnellSecondVirialProvider),
        }
        for spelling, (canonical, provider_type) in cases.items():
            with self.subTest(correlation=spelling):
                pfd = parse_pfd(
                    f'THERMO_METHOD: NRTL-BV | correlation={spelling}\n'
                    'COMPONENTS:\n'
                    '    isobutane | Isobutane | dipole=0, R_prime=1.8\n'
                )
                self.assertEqual(
                    pfd.metadata.thermo_options,
                    {'correlation': canonical},
                )
                simulator = Simulator(pfd).initialize()
                self.assertIsInstance(
                    simulator.thermo.vapor_eos.provider,
                    provider_type,
                )
                self.assertEqual(
                    simulator.thermo.vapor_eos.correlation,
                    canonical,
                )

    def test_unknown_bv_correlation_is_rejected(self):
        with self.assertRaisesRegex(Exception, 'Unsupported second-virial correlation'):
            parse_pfd('THERMO_METHOD: NRTL-BV | correlation=NOT-A-CORRELATION\n')

    def test_direct_hoc_suffixes_convert_to_bv_hoc(self):
        aliases = {
            'NRTL-HOC': NRTLBVThermodynamics,
            'UNIQUAC-HOC': UNIQUACBVThermodynamics,
            'UNIFAC-HOC': UNIFACBVThermodynamics,
            'UNIFDMD-HOC': UNIFDMDBVThermodynamics,
            'UNIFNIST-HOC': UNIFNISTBVThermodynamics,
        }
        props = component_props(
            Tc=425.2,
            Pc=38.0 * 1.01325,
            CAS='',
            formula='C4H10',
            smiles='CC(C)C',
            dipole_moment=0.0,
            modified_radius_of_gyration=2.0,
        )
        for method, expected_type in aliases.items():
            with self.subTest(method=method):
                provider = HaydenOConnellSecondVirialProvider(
                    ['isobutane'],
                    {'isobutane': props},
                )
                thermo = create_thermodynamics(
                    ['isobutane'],
                    method,
                    second_virial_provider=provider,
                )
                self.assertIsInstance(thermo, expected_type)
                self.assertEqual(thermo.vapor_eos.correlation, 'HOC')

    def test_direct_hoc_suffix_canonicalizes_pfd_and_rejects_conflicts(self):
        for method, canonical in {
            'NRTL_HOC': 'NRTL-BV',
            'UNIQUAC-HOC': 'UNIQUAC-BV',
            'UNIFAC-HOC': 'UNIFAC-BV',
            'UNIFDMD-HOC': 'UNIFDMD-BV',
            'UNIFNIST-HOC': 'UNIFNIST-BV',
            'UNIF-HOC': 'UNIFAC-BV',
        }.items():
            with self.subTest(method=method):
                pfd = parse_pfd(f'THERMO_METHOD: {method}\n')
                self.assertEqual(pfd.metadata.thermo_method, canonical)
                self.assertEqual(
                    pfd.metadata.thermo_options,
                    {'correlation': 'HOC'},
                )
                self.assertIn(
                    f'THERMO_METHOD: {canonical} | correlation=HOC',
                    pfd.to_pfd(),
                )

        with self.assertRaisesRegex(
            Exception,
            'implies correlation=HOC.*cannot use correlation=ABBOTT',
        ):
            parse_pfd(
                'THERMO_METHOD: NRTL-HOC | correlation=ABBOTT\n'
            )


if __name__ == '__main__':
    unittest.main()
