import math
import os
import sys
import unittest
import warnings
from unittest.mock import patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chemical_properties import get_database
from compound_identity import get_compound_identity_resolver
from simulator import Simulator, SimulationError
from thermodynamics import create_thermodynamics
from vapor_dimerization import (
    CROSS_DIMER_STATISTICAL_DELTA_S_J_MOL_K,
    GENERIC_MONOCARBOXYLIC_DELTA_H_J_MOL,
    GENERIC_MONOCARBOXYLIC_DELTA_S_J_MOL_K,
    MultiVaporDimerizationModel,
    P_STD,
    VaporDimerizationModel,
    estimate_dimer_properties,
    get_cross_dimerization_residual,
    get_dimerization_params,
    is_monocarboxylic_acid,
    list_dimerizing_acids,
)


class VaporDimerizationTests(unittest.TestCase):
    def test_common_acid_names_use_database_rows(self):
        exact = get_dimerization_params('CH3COOH')
        named = get_dimerization_params('acetic acid')

        self.assertIsNotNone(exact)
        self.assertEqual(named, exact)
        self.assertEqual(named['source'], 'database')

    def test_oxygenated_acids_use_curated_dimerization_parameters(self):
        expected = {
            'lactic acid': (-153.0, -66300),
            'L-lactic acid': (-153.0, -66300),
            '(S)-Lactic acid': (-153.0, -66300),
            '79-33-4': (-153.0, -66300),
            'pyruvic acid': (-151.0, -32600),
            '2-oxopropanoic acid': (-151.0, -32600),
            '127-17-3': (-151.0, -32600),
        }

        for identifier, (delta_S, delta_H) in expected.items():
            with self.subTest(identifier=identifier):
                params = get_dimerization_params(identifier)
                self.assertIsNotNone(params)
                self.assertEqual(params['delta_S_J_per_mol_K'], delta_S)
                self.assertEqual(params['delta_H_J_per_mol'], delta_H)
                self.assertEqual(params['source'], 'database')

    def test_structurally_ambiguous_formula_does_not_imply_acid(self):
        self.assertIsNone(get_dimerization_params('C3H6O2'))
        self.assertIsNone(get_dimerization_params('CH3COOC2H5'))
        self.assertFalse(is_monocarboxylic_acid('C3H6O2'))

    def test_structural_monoacid_recognition_is_shared(self):
        for identifier in (
            '111-14-8',
            'heptanoic acid',
            'CCCCCC(=O)O',
        ):
            with self.subTest(identifier=identifier):
                self.assertTrue(is_monocarboxylic_acid(identifier))

        self.assertFalse(is_monocarboxylic_acid('benzene'))
        self.assertFalse(is_monocarboxylic_acid('succinic acid'))
        self.assertIsNotNone(get_dimerization_params('heptanoic acid'))

    def test_unlisted_monoacid_uses_calibrated_generic_parameters(self):
        params = get_dimerization_params('heptanoic acid')

        self.assertIsNotNone(params)
        self.assertEqual(
            params['delta_H_J_per_mol'],
            GENERIC_MONOCARBOXYLIC_DELTA_H_J_MOL,
        )
        self.assertEqual(
            params['delta_S_J_per_mol_K'],
            GENERIC_MONOCARBOXYLIC_DELTA_S_J_MOL_K,
        )
        self.assertEqual(params['source'], 'estimated (monocarboxylic acid default)')

    def test_acrylic_acid_resolves_locally_for_vdm_use(self):
        resolver = get_compound_identity_resolver()

        for identifier in ['C2H3COOH', 'acrylic acid', 'prop-2-enoic acid', 'C3H4O2']:
            with self.subTest(identifier=identifier):
                identity = resolver.resolve(identifier)
                self.assertIsNotNone(identity)
                self.assertEqual(identity.symbol, 'C2H3COOH')

        props = get_database().get('C2H3COOH', fetch_online=False)
        self.assertIsNotNone(props)
        self.assertEqual(props.CAS, '79-10-7')
        self.assertAlmostEqual(props.MW, 72.06)

    def test_list_dimerizing_acids_uses_json_units(self):
        acids = list_dimerizing_acids()

        self.assertGreater(len(acids), 0)
        self.assertIn('delta_S_J_per_mol_K', acids[0])
        self.assertIn('delta_H_J_per_mol', acids[0])

    def test_dimerization_solution_satisfies_equilibrium(self):
        model = VaporDimerizationModel('A', 'A2', delta_S=-155.5, delta_H=-65500.0)
        T = 391.0
        P = 1.01325
        y_true = model.solve_dimerization(
            T,
            P,
            {'A': 0.5, 'B': 0.5},
            {'A': 1.0, 'A2': 1.0, 'B': 1.0},
        )

        self.assertAlmostEqual(sum(y_true.values()), 1.0, places=12)
        lhs = (y_true['A2'] * P / P_STD) / ((y_true['A'] * P / P_STD) ** 2)
        self.assertAlmostEqual(lhs, model.K_eq(T), places=12)

    def test_pfd_vdm_overrides_enable_nonacids_and_cross_residuals(self):
        source = (
            'PROCESS: Arbitrary VDM Overrides\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: UNIQUAC-VDM\n'
            'ONLINE_LOOKUP: false\n\n'
            'COMPONENTS:\n'
            '    E | Ethanol | CAS=64-17-5, uniquac_r=2.1055, '
            'uniquac_q=1.972, VDM={delta_H:-50000,delta_S:-120}\n'
            '    A | Acetone | CAS=67-64-1, uniquac_r=2.5735, '
            'uniquac_q=2.336, VDM={delta_H_J_per_mol:-40000,'
            'delta_S_J_per_mol_K:-100}\n\n'
            'INTERACTION_PARAMETERS:\n'
            '    E/A | model=VDM, delta_H_residual=1000, '
            'delta_S_residual_J_per_mol_K=-2.5\n\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = E:0.5, A:0.5\n'
        )

        simulator = Simulator.from_string(source)
        result = simulator.run()

        self.assertTrue(result.converged)
        ethanol = simulator.thermo._vdm_models['E']
        acetone = simulator.thermo._vdm_models['A']
        self.assertEqual(ethanol.association_enthalpy(), -50000.0)
        self.assertEqual(ethanol.association_entropy(), -120.0)
        self.assertEqual(acetone.association_enthalpy(), -40000.0)
        self.assertEqual(acetone.association_entropy(), -100.0)
        self.assertEqual(
            simulator.thermo.props['A'].property_sources[
                'vapor_dimerization'
            ]['method'],
            'pfd_component_override',
        )

        _single, multi = simulator.thermo._active_vdm_model({'E': 0.5, 'A': 0.5})
        pair = next(
            key for key in multi._pair_delta_H
            if set(key) == {'E', 'A'}
        )
        self.assertEqual(multi._pair_delta_H[pair], -44000.0)
        self.assertAlmostEqual(
            multi._pair_delta_S[pair],
            -112.5 + CROSS_DIMER_STATISTICAL_DELTA_S_J_MOL_K,
        )

    def test_pfd_vdm_component_override_wins_over_curated_acid_data(self):
        source = (
            'PROCESS: Curated VDM Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: UNIQUAC-VDM\n'
            'ONLINE_LOOKUP: false\n\n'
            'COMPONENTS:\n'
            '    AcOH | Acetic acid | CAS=64-19-7, uniquac_r=2.2024, '
            'uniquac_q=2.072, VDM={delta_H:-50000,delta_S:-120}\n\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 100 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = AcOH:1\n'
        )

        simulator = Simulator.from_string(source)
        result = simulator.run()
        model = simulator.thermo._vdm_models['AcOH']

        self.assertTrue(result.converged)
        self.assertEqual(model.association_enthalpy(), -50000.0)
        self.assertEqual(model.association_entropy(), -120.0)

    def test_pfd_vdm_cross_override_requires_two_active_homodimers(self):
        source = (
            'PROCESS: Incomplete Cross VDM\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: UNIQUAC-VDM\n'
            'ONLINE_LOOKUP: false\n\n'
            'COMPONENTS:\n'
            '    E | Ethanol | CAS=64-17-5, uniquac_r=2.1055, '
            'uniquac_q=1.972, VDM={delta_H:-50000,delta_S:-120}\n'
            '    A | Acetone | CAS=67-64-1, uniquac_r=2.5735, '
            'uniquac_q=2.336\n\n'
            'INTERACTION_PARAMETERS:\n'
            '    E/A | model=VDM, delta_H_residual=1000, '
            'delta_S_residual=-2.5\n\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = E:0.5, A:0.5\n'
        )

        simulator = Simulator.from_string(source)
        with self.assertRaisesRegex(
            SimulationError,
            r'VDM cross override for A/E requires active homodimer parameters.*missing A',
        ):
            simulator.run()

    def test_no_rk_association_state_uses_single_analytic_solve(self):
        model = VaporDimerizationModel('A', 'A2', delta_S=-155.5, delta_H=-65500.0)
        T = 391.0
        P = 1.01325
        nominal = {'A': 0.5, 'B': 0.5}

        with patch.object(model, 'solve_dimerization', wraps=model.solve_dimerization) as solve:
            state = model.association_state(T, P, nominal, rk_model=None)

        self.assertEqual(solve.call_count, 1)
        y_true, phi_physical, phi_total, rk_failure = model._equilibrium_state(
            T, P, nominal, rk_model=None
        )
        self.assertIsNone(rk_failure)
        for comp, value in y_true.items():
            self.assertAlmostEqual(state['y_true'][comp], value, places=12)
        for comp, value in phi_physical.items():
            self.assertAlmostEqual(state['phi_physical'][comp], value, places=12)
        for comp, value in phi_total.items():
            self.assertAlmostEqual(state['phi_total'][comp], value, places=12)

    def test_vdm_vapor_density_uses_physical_molecules_per_nominal_mole(self):
        cases = (
            (
                ['CH3COOH', 'H2O'],
                {'CH3COOH': 0.5, 'H2O': 0.5},
            ),
            (
                ['CH3COOH', 'C2H5COOH', 'H2O'],
                {'CH3COOH': 0.25, 'C2H5COOH': 0.20, 'H2O': 0.55},
            ),
        )
        for components, composition in cases:
            with self.subTest(components=components):
                thermo = create_thermodynamics(components, 'UNIQUAC-VDM')
                ideal = create_thermodynamics(components, 'IDEAL')
                _acid, model = thermo._active_vdm_model(composition)
                state = model.association_state(
                    390.0,
                    1.01325,
                    composition,
                    rk_model=None,
                )
                physical_factor = 1.0 - sum(state['extents'].values())
                self.assertGreater(physical_factor, 0.0)
                self.assertLess(physical_factor, 1.0)
                ideal_density = ideal.mixture_molar_density(
                    composition,
                    390.0,
                    1.01325,
                    1.0,
                    y=composition,
                )
                self.assertAlmostEqual(
                    thermo.mixture_molar_density(
                        composition,
                        390.0,
                        1.01325,
                        1.0,
                        y=composition,
                    ),
                    ideal_density / physical_factor,
                    places=12,
                )

    def test_single_acid_vdm_k_values_match_full_fixed_point(self):
        T = 378.15
        P = 1.01325
        composition = {'CH3COOH': 0.2, 'H2O': 0.8}
        thermo_fast = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')
        thermo_ref = create_thermodynamics(['CH3COOH', 'H2O'], 'UNIQUAC-VDM')

        fast = thermo_fast.K_values(T, P, composition)
        x = {comp: max(float(composition.get(comp, 0.0)), 0.0) for comp in thermo_ref.components}
        total = sum(x.values())
        x = {comp: value / total for comp, value in x.items()}
        _, model = thermo_ref._active_vdm_model(x)
        gamma = thermo_ref.activity_coefficients(T, x)
        phi_sat = {comp: thermo_ref._vdm_phi_sat(comp, T) for comp in thermo_ref.components}
        reference = {
            comp: float(max(
                1e-6,
                min(1e6, gamma.get(comp, 1.0) * phi_sat[comp] * thermo_ref.Psat(comp, T) / P),
            ))
            for comp in thermo_ref.components
        }
        y_sum = sum(x[comp] * reference[comp] for comp in thermo_ref.components)
        y = {comp: x[comp] * reference[comp] / y_sum for comp in thermo_ref.components}

        for _ in range(15):
            phi_v = model.fugacity_coefficients(T, P, y, rk_model=None)
            K_new = {}
            for comp in thermo_ref.components:
                phi = max(phi_v.get(comp, 1.0), 1e-12)
                value = gamma.get(comp, 1.0) * phi_sat[comp] * thermo_ref.Psat(comp, T) / (phi * P)
                K_new[comp] = float(max(1e-6, min(1e6, value)))
            y_new = {comp: x[comp] * K_new[comp] for comp in thermo_ref.components}
            y_sum = sum(max(value, 0.0) for value in y_new.values())
            y_new = {comp: max(value, 0.0) / y_sum for comp, value in y_new.items()}
            reference = K_new
            if max(abs(y_new[comp] - y.get(comp, 0.0)) for comp in thermo_ref.components) < 1e-9:
                break
            y = y_new

        for comp in thermo_fast.components:
            self.assertAlmostEqual(fast[comp], reference[comp], delta=abs(reference[comp]) * 2e-8)

    def test_cross_dimer_uses_statistical_factor_plus_residual(self):
        acetic = VaporDimerizationModel('A', 'A2', delta_S=-150.0, delta_H=-60000.0)
        propionic = VaporDimerizationModel('B', 'B2', delta_S=-160.0, delta_H=-70000.0)
        model = MultiVaporDimerizationModel(
            {'A': acetic, 'B': propionic},
            {
                ('A', 'B'): {
                    'delta_H_residual_J_per_mol': -5000.0,
                    'delta_S_residual_J_per_mol_K': 10.0,
                }
            },
        )

        delta_H, delta_S = model._cross_delta('A', 'B')
        self.assertAlmostEqual(delta_H, -70000.0)
        self.assertAlmostEqual(
            delta_S,
            -145.0 + CROSS_DIMER_STATISTICAL_DELTA_S_J_MOL_K,
        )

        T = 390.0
        P = 1.01325
        y_true = model.solve_dimerization(
            T,
            P,
            {'A': 0.25, 'B': 0.25, 'water': 0.5},
        )
        self.assertAlmostEqual(sum(y_true.values()), 1.0, places=12)
        self.assertIn('(A)(B)', y_true)
        lhs = (
            y_true['(A)(B)'] * P / P_STD
            / ((y_true['A'] * P / P_STD) * (y_true['B'] * P / P_STD))
        )
        rhs = math.exp(delta_S / 8.314 - delta_H / (8.314 * T))
        self.assertAlmostEqual(lhs, rhs, places=7)

    def test_cross_dimer_statistical_factor_is_exactly_two(self):
        model_a = VaporDimerizationModel(
            'A', 'A2', delta_S=-150.0, delta_H=-60000.0,
        )
        model_b = VaporDimerizationModel(
            'B', 'B2', delta_S=-160.0, delta_H=-70000.0,
        )
        model = MultiVaporDimerizationModel({'A': model_a, 'B': model_b})
        T = 390.0

        delta_H_ab = model._pair_delta_H[('A', 'B')]
        delta_S_ab = model._pair_delta_S[('A', 'B')]
        K_ab = math.exp(delta_S_ab / 8.314 - delta_H_ab / (8.314 * T))

        self.assertAlmostEqual(
            K_ab,
            2.0 * math.sqrt(model_a.K_eq(T) * model_b.K_eq(T)),
            places=12,
        )

    def test_compiled_two_acid_solver_satisfies_balances(self):
        from compiled_vdm import solve_two_acid_true_moles

        n_a = 0.25
        n_b = 0.20
        inert_total = 0.55
        k_aa = 8.0
        k_ab = 3.5
        k_bb = 6.0
        solution = solve_two_acid_true_moles(n_a, n_b, inert_total, k_aa, k_ab, k_bb)
        if solution is None:
            self.skipTest('compiled VDM backend is unavailable')

        a, b, e_aa, e_ab, e_bb, total = solution
        self.assertAlmostEqual(a + 2.0 * e_aa + e_ab, n_a, places=10)
        self.assertAlmostEqual(b + 2.0 * e_bb + e_ab, n_b, places=10)
        self.assertAlmostEqual(total, inert_total + a + b + e_aa + e_ab + e_bb, places=10)
        self.assertAlmostEqual(e_aa, k_aa * a * a / total, places=10)
        self.assertAlmostEqual(e_ab, k_ab * a * b / total, places=10)
        self.assertAlmostEqual(e_bb, k_bb * b * b / total, places=10)

    def test_three_acid_solver_satisfies_all_association_equilibria(self):
        models = {
            'A': VaporDimerizationModel('A', 'A2', delta_S=-150.0, delta_H=-60000.0),
            'B': VaporDimerizationModel('B', 'B2', delta_S=-160.0, delta_H=-70000.0),
            'C': VaporDimerizationModel('C', 'C2', delta_S=-155.0, delta_H=-65000.0),
        }
        model = MultiVaporDimerizationModel(models)
        T = 390.0
        P = 1.01325
        y_true = model.solve_dimerization(
            T,
            P,
            {'A': 0.15, 'B': 0.12, 'C': 0.10, 'water': 0.63},
        )

        self.assertAlmostEqual(sum(y_true.values()), 1.0, places=12)
        for acid_i in ['A', 'B', 'C']:
            for acid_j in ['A', 'B', 'C']:
                if acid_j < acid_i:
                    continue
                dimer = model._dimer_symbol(acid_i, acid_j)
                lhs = (
                    y_true[dimer] * P / P_STD
                    / ((y_true[acid_i] * P / P_STD) * (y_true[acid_j] * P / P_STD))
                )
                if acid_i == acid_j:
                    rhs = models[acid_i].K_eq(T)
                else:
                    delta_H, delta_S = model._cross_delta(acid_i, acid_j)
                    rhs = math.exp(delta_S / 8.314 - delta_H / (8.314 * T))
                self.assertAlmostEqual(lhs, rhs, places=7)

    def test_compiled_n_acid_solver_satisfies_balances(self):
        from compiled_vdm import solve_n_acid_true_moles

        nominal = [0.15, 0.12, 0.10]
        inert_total = 0.63
        pair_i = [0, 0, 0, 1, 1, 2]
        pair_j = [0, 1, 2, 1, 2, 2]
        pair_kappa = [8.0, 3.5, 4.0, 6.0, 2.5, 7.0]
        solution = solve_n_acid_true_moles(
            nominal,
            inert_total,
            pair_i,
            pair_j,
            pair_kappa,
        )
        if solution is None:
            self.skipTest('compiled VDM backend is unavailable')

        monomers, extents, total = solution
        balances = list(monomers)
        for i, j, extent in zip(pair_i, pair_j, extents):
            if i == j:
                balances[i] += 2.0 * extent
            else:
                balances[i] += extent
                balances[j] += extent
        for actual, expected in zip(balances, nominal):
            self.assertAlmostEqual(actual, expected, places=10)
        self.assertAlmostEqual(total, inert_total + sum(monomers) + sum(extents), places=10)
        for i, j, kappa, extent in zip(pair_i, pair_j, pair_kappa, extents):
            self.assertAlmostEqual(extent, kappa * monomers[i] * monomers[j] / total, places=10)

    def test_cross_dimer_residuals_can_be_read_from_json_shape(self):
        fake_data = {
            'cross_dimers': [
                {
                    'symbols': ['CH3COOH', 'C2H5COOH'],
                    'delta_H_residual': -1234.0,
                    'delta_S_residual': 5.5,
                    'source': 'unit test',
                }
            ],
            'dimers': [],
        }
        with patch('vapor_dimerization._dimerization_data', return_value=fake_data):
            residual = get_cross_dimerization_residual('acetic acid', 'propionic acid')

        self.assertAlmostEqual(residual['delta_H_residual_J_per_mol'], -1234.0)
        self.assertAlmostEqual(residual['delta_S_residual_J_per_mol_K'], 5.5)
        self.assertEqual(residual['source'], 'unit test')

    def test_three_acid_pairs_have_explicit_zero_cross_dimer_residuals(self):
        pairs = (
            ('acetic acid', 'C2H5COOH'),
            ('C2H3COOH', 'CH3COOH'),
            ('propionic acid', 'acrylic acid'),
        )
        for first, second in pairs:
            with self.subTest(pair=(first, second)):
                residual = get_cross_dimerization_residual(first, second)
                reverse = get_cross_dimerization_residual(second, first)
                for record in (residual, reverse):
                    self.assertEqual(
                        record['delta_H_residual_J_per_mol'], 0.0
                    )
                    self.assertEqual(
                        record['delta_S_residual_J_per_mol_K'], 0.0
                    )
                    self.assertNotEqual(record['source'], 'ideal combining rule')
                    self.assertIn('three_acids_vle.json', record['source'])

    def test_rk_failure_warns_before_ideal_fallback(self):
        class BrokenRK:
            def fugacity_coefficients(self, *args, **kwargs):
                raise RuntimeError('bad RK')

        model = VaporDimerizationModel('A', 'A2', delta_S=-155.5, delta_H=-65500.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            phi = model.fugacity_coefficients(
                391.0,
                1.01325,
                {'A': 0.5, 'B': 0.5},
                rk_model=BrokenRK(),
            )

        self.assertIn('A', phi)
        self.assertTrue(any('bad RK' in str(w.message) for w in caught))

    def test_estimated_dimer_hf_uses_kj_units_and_exothermic_sign(self):
        db = get_database()
        monomer = db.get('CH3COOH', fetch_online=False)
        if monomer is None or monomer.Hf is None:
            self.skipTest('local acetic acid heat of formation is unavailable')

        props = estimate_dimer_properties('CH3COOH', '(CH3COOH)2', db=db)
        expected = 2.0 * monomer.Hf - 65.5

        self.assertTrue(math.isfinite(props['Hf']))
        self.assertAlmostEqual(props['Hf'], expected, places=8)


if __name__ == '__main__':
    unittest.main()
