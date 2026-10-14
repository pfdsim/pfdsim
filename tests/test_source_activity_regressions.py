"""Scientific source-regression checks, separate from approved data identity."""

from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import unittest

import numpy as np
import pytest

from scripts.activity_fitting.binary_regression import (
    central_jacobian, complex_jacobian, fit_regression,
    log_activity, projected_gradient_cosine,
)
from scripts.build_cas_interaction_parameters import (
    activity_record_signature, supplemental_curated_water_nonwater_records,
)
from tests.cache_isolation import isolated_runtime_caches
from tests.fit_source_comparison import assert_fit_source_equal, assert_source_inputs_equal
from thermodynamics_models.interaction_estimation import _nrtl_ln_gamma, _uniquac_ln_gamma

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'fendu': ('fit_fendu_2025_activity_parameters', 'fendu_2025_activity_refits'),
    'mohsen': ('fit_mohsen_nia_2010_pentanol_propionic', 'mohsen_nia_2010_pentanol_propionic_association'),
    'cyclohexane': ('fit_moreau_ovejero_2012_2007_activity', 'moreau_ovejero_pentanol_cyclohexane_joint_fit'),
    'toluene': ('fit_moreau_2012_pentanol_toluene_activity', 'moreau_2012_pentanol_toluene_joint_fit'),
    'hexane': ('fit_moreau_ovejero_2016_2007_pentanol_hexane_activity', 'moreau_ovejero_2016_2007_pentanol_hexane_joint_fit'),
}
CASE = unittest.TestCase()


def json_value(value):
    return json.loads(json.dumps(value))


def load_source(name):
    return json.loads((ROOT / 'data/source/activity_fitting' / (SOURCES[name][1] + '.json')).read_text())


def model_context(module):
    return module.configured_common_fit() if hasattr(module, 'configured_common_fit') else nullcontext()


@pytest.fixture(scope='module', params=list(SOURCES))
def source(request):
    name = request.param
    module = importlib.import_module('scripts.activity_fitting.' + SOURCES[name][0])
    with isolated_runtime_caches():
        generated = json_value(module.build_payload())
    return name, module, load_source(name), generated


def test_regeneration_preserves_source_data_and_model_policy(source):
    name, _, approved, generated = source
    assert_source_inputs_equal(CASE, generated, approved, refitted_antoine=name == 'fendu')
    if name == 'mohsen':
        assert generated['analysis']['selection'] == approved['analysis']['selection']
        # The publication selects ZERO interactions, irrespective of diagnostic fits.
        for actual, expected in zip(generated['interactions'], approved['interactions']):
            assert {k: v for k, v in actual.items() if k.startswith('tau')} == {
                k: v for k, v in expected.items() if k.startswith('tau')}


def test_approved_coefficients_are_exported_exactly():
    # Exercise the authoritative publication builder without activating new fits.
    for model in ('NRTL', 'UNIQUAC'):
        records, _, _ = supplemental_curated_water_nonwater_records([], model)
        for name in SOURCES:
            for approved in load_source(name)['interactions']:
                if approved['model'] != model:
                    continue
                pair = {approved['cas1'], approved['cas2']}
                exported = [r for r in records if {r['cas1'], r['cas2']} == pair]
                assert len(exported) == 1
                assert activity_record_signature(exported[0], model) == activity_record_signature(approved, model)
                assert exported[0]['source_file'] == 'data/source/activity_fitting/' + SOURCES[name][1] + '.json'


def test_regression_quality_and_stationarity(source):
    name, module, approved, generated = source
    if name == 'fendu':
        for system_name, system in module.SYSTEMS.items():
            rows = module._rows(system)
            old_fits = approved['analyses'][system_name]['candidate_fits']
            new_fits = generated['analyses'][system_name]['candidate_fits']
            for model, forms in new_fits.items():
                if model not in ('NRTL', 'UNIQUAC'):
                    continue
                specs = module.NRTL_SPECS if model == 'NRTL' else module.UNIQUAC_SPECS
                for form, fit in forms.items():
                    function = lambda p: module._pressure_residuals(model, form, p, rows, system)
                    parameters = np.asarray(fit['parameters'])
                    old_parameters = np.asarray(old_fits[model][form]['parameters'])
                    residual, old_residual = function(parameters), function(old_parameters)
                    assert residual @ residual <= old_residual @ old_residual + 1e-9 * max(1., old_residual @ old_residual)
                    assert fit['fit']['pressure_RMSE_percent'] == pytest.approx(
                        100 * np.sqrt(np.mean(residual**2)), rel=1e-10)
                    # The whole held-out composition series remains excluded.
                    assert fit['leave_one_composition_out']['point_count'] == old_fits[model][form]['leave_one_composition_out']['point_count']
                    assert fit['leave_one_composition_out']['pressure_RMSE_percent'] <= (
                        old_fits[model][form]['leave_one_composition_out']['pressure_RMSE_percent'] + 1e-4)
                    if len(parameters):
                        lower, upper = specs[form][1:]
                        assert np.all((parameters >= lower) & (parameters <= upper))
                        assert projected_gradient_cosine(residual, complex_jacobian(function, parameters),
                                                         parameters, lower, upper) < 1e-7
    elif name == 'mohsen':
        for model, fit in generated['analysis']['vdm_refits'].items():
            engine = module.MutableVDMModel(model)
            def function(p):
                return module.objective(engine, np.r_[p, .1] if model == 'NRTL' else p)
            parameters = np.asarray(fit['parameters'][:2])
            previous = np.asarray(approved['analysis']['vdm_refits'][model]['parameters'][:2])
            residual, old_residual = function(parameters), function(previous)
            assert fit['objective'] == approved['analysis']['vdm_refits'][model]['objective']
            assert 2 * fit['cost'] == pytest.approx(residual @ residual, rel=1e-8)
            assert residual @ residual <= old_residual @ old_residual + 1e-7
            assert projected_gradient_cosine(residual, central_jacobian(function, parameters),
                                             parameters, -5000., 5000.) < 1e-7
    else:
        common = module.common if hasattr(module, 'common') else module
        with model_context(module):
            assert [(c['model'], c['alpha'], c['form']) for c in generated['model_selection']] == [
                (c['model'], c['alpha'], c['form']) for c in approved['model_selection']]
            for candidate, previous in zip(generated['model_selection'], approved['model_selection']):
                model, alpha = candidate['model'], candidate['alpha']
                curved = candidate['form'] == 'ABH'
                function = lambda p: common.objective(model, alpha, curved, p)
                parameters = np.asarray(candidate['parameters'])
                residual = function(parameters)
                old_residual = function(np.asarray(previous['parameters']))
                lower = np.array([-10., -10000., -500., -10., -10000., -500.]
                                 if curved else [-10., -10000., -10., -10000.])
                assert candidate['success']
                assert np.isfinite(parameters).all()
                assert np.all((parameters >= lower) & (parameters <= -lower))
                assert candidate['chi_square'] == pytest.approx(residual @ residual, rel=1e-9)
                assert residual @ residual <= old_residual @ old_residual + 1e-8 * max(1., old_residual @ old_residual)
                assert projected_gradient_cosine(residual, complex_jacobian(function, parameters),
                                                 parameters, lower, -lower) < 1e-7


def test_fixed_approved_parameter_reports_remain_consistent(source):
    name, module, approved, generated = source
    if name == 'fendu':
        for component, previous in approved['metadata']['pure_component_pressure_basis']['coefficients'].items():
            temperatures = np.array([row[0] for row in module.PURE_DATA[component]])
            reference = 10**(previous['A'] - previous['B_K'] / (temperatures + previous['C_K']))
            assert np.max(abs(module.psat_kpa(component, temperatures) / reference - 1)) < 1e-7
        for system in approved['analyses']:
            assert_fit_source_equal(
                CASE,
                {'published_nrtl_reproduction': generated['analyses'][system]['candidate_fits']['published_nrtl_reproduction']},
                {'published_nrtl_reproduction': approved['analyses'][system]['candidate_fits']['published_nrtl_reproduction']},
            )
    elif name == 'mohsen':
        # These reports use published/zero coefficients, not newly refitted ones.
        for key in ('published_parameter_reproduction', 'zero_model_evaluation', 'zero_liquid_azeotropes'):
            assert_fit_source_equal(CASE, {'analysis': {key: generated['analysis'][key]}},
                                    {'analysis': {key: approved['analysis'][key]}})
    else:
        if name in ('toluene', 'hexane'):
            assert_fit_source_equal(
                CASE,
                {'published_parameter_reproduction': generated['published_parameter_reproduction']},
                {'published_parameter_reproduction': approved['published_parameter_reproduction']},
            )
        common = module.common if hasattr(module, 'common') else module
        with model_context(module):
            for candidate in approved['model_selection']:
                model, alpha = candidate['model'], candidate['alpha']
                values, curved = candidate['parameters'], candidate['form'] == 'ABH'
                calculated = (common.score(model, alpha, values, curved) if name == 'cyclohexane'
                              else module._score(model, alpha, values, curved) if name == 'toluene'
                              else module._moreau_score(model, alpha, values, curved))
                expected = candidate['score']
                # Hexane computes held-out diagnostics only for selected forms.
                if name == 'hexane':
                    expected = {k: v for k, v in expected.items() if k != 'Ovejero'}
                assert_fit_source_equal(CASE, {'score': json_value(calculated)}, {'score': expected})
            for record in approved['interactions']:
                calculated = (common.regularized_stress_test(record) if name == 'cyclohexane'
                              else module._stress_test(record))
                assert_fit_source_equal(
                    CASE, {'stress': json_value(calculated)},
                    {'stress': approved['regularized_extrapolation_stress_test'][record['model']]},
                )


def test_selected_regressions_keep_heldout_and_stability_checks(source):
    name, module, approved, generated = source
    if name in ('fendu', 'mohsen'):
        return
    common = module.common if hasattr(module, 'common') else module
    compositions = np.unique(np.r_[np.linspace(.001, .999, 501),
                                    np.geomspace(1e-7, .001, 31),
                                    1 - np.geomspace(1e-7, .001, 31)])
    with model_context(module):
        for record in generated['interactions']:
            model = record['model']
            alpha = record.get('alpha12') if model == 'NRTL' else None
            candidate = next(c for c in generated['model_selection'] if
                             (c['model'], c['alpha'], c['form']) == (model, alpha, 'ABH'))
            previous = next(c for c in approved['model_selection'] if
                            (c['model'], c['alpha'], c['form']) == (model, alpha, 'ABH'))
            for T in np.linspace(record['Tmin_K'], record['Tmax_K'], 31):
                first, second = common.ln_gamma(model, compositions + 1j * 1e-25, T,
                                                candidate['parameters'], alpha, True)
                curvature = 1 / (compositions * (1 - compositions)) + np.imag(first - second) / 1e-25
                assert np.min(curvature) > 0
            if name in ('cyclohexane', 'hexane'):
                heldout = candidate['score']['Ovejero']
                reference = previous['score']['Ovejero']
                # Budgets below the source's reported 0.1 K / 0.0001 y precision.
                assert heldout['T_K']['RMSE'] <= reference['T_K']['RMSE'] + .01
                assert heldout['y1']['RMSE'] <= reference['y1']['RMSE'] + 5e-5
            # Preserve continuation checks: finite, positive gamma and decay at
            # large T, rather than pinning a new fit to old extrapolated digits.
            stress = generated['regularized_extrapolation_stress_test'][model]
            for values in stress.values():
                assert np.isfinite(list(values.values())).all()
                for key in ('gamma1_infinite_dilution', 'gamma2_infinite_dilution'):
                    assert 1e-12 < values[key] < np.exp(50)
            first, last = stress[min(stress, key=float)], stress[max(stress, key=float)]
            for key in ('gamma1_infinite_dilution', 'gamma2_infinite_dilution'):
                assert abs(np.log(last[key])) < abs(np.log(first[key]))


@pytest.mark.parametrize('model', ['NRTL', 'UNIQUAC'])
def test_activity_derivatives_against_independent_equations(model):
    from scripts.activity_fitting import fit_moreau_ovejero_2012_2007_activity as common
    rng = np.random.default_rng(4182)
    for _ in range(12):
        x, T = float(rng.uniform(.02, .98)), float(rng.uniform(298.15, 354.7))
        p = np.array([.2, 200., -2., -.3, -150., 3.]) + rng.normal(size=6)
        alpha = .3 if model == 'NRTL' else None
        tau12, tau21 = common.interaction_values(model, T, p, True)
        expected = (_nrtl_ln_gamma(x, tau12, tau21, alpha) if model == 'NRTL' else
                    _uniquac_ln_gamma(x, common.R_VALUES, common.Q_VALUES, tau12, tau21))
        np.testing.assert_allclose(common.ln_gamma(model, x, T, p, alpha, True), expected, atol=1e-12, rtol=0)
        reference = -common.R * T*T * np.imag(common.reduced_ge(model, x, T+1j*1e-25, p, alpha, True)) / 1e-25
        assert common.excess_enthalpy(model, x, T, p, alpha, True) == pytest.approx(reference, abs=1e-8)
        function = lambda values: common.objective(model, alpha, True, values)
        np.testing.assert_allclose(complex_jacobian(function, p), central_jacobian(function, p, 1e-5), rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize('model', ['NRTL', 'UNIQUAC'])
def test_source_pressure_closes_virial_fugacity_equations(model):
    from scripts.activity_fitting import fit_moreau_ovejero_2012_2007_activity as common
    x = np.array([.01, .2, .6, .99])
    p, alpha = np.array([.2, 200., -2., -.3, -150., 3.]), .3 if model == 'NRTL' else None
    pressure, y1 = common.source_bubble(model, x, p, alpha, True)
    y = np.column_stack((y1, 1-y1))
    b, pure, volume = np.asarray(common.B_L_MOL), np.asarray(common.PSAT_KPA), np.asarray(common.VL_L_MOL)
    bmix = np.einsum('ni,ij,nj->n', y, b, y)
    phi = np.exp((2*y@b.T-bmix[:, None])*pressure[:, None]/(common.R*common.TREF))
    tau12, tau21 = common.interaction_values(model, common.TREF, p, True)
    gamma = np.exp(log_activity(model, x, tau12, tau21, alpha=alpha, r=common.R_VALUES, q=common.Q_VALUES))
    liquid = (np.column_stack((x, 1-x))*gamma*pure*np.exp(np.diag(b)*pure/(common.R*common.TREF))
              *np.exp(volume*(pressure[:, None]-pure)/(common.R*common.TREF)))
    np.testing.assert_allclose(y*pressure[:, None]*phi, liquid, rtol=1e-12, atol=1e-12)


def test_fit_quality_rejects_unconverged_results_and_handles_active_bounds():
    with pytest.raises(RuntimeError, match='No converged stationary'):
        fit_regression(lambda p: np.array([np.exp(p[0])-20]), [[0.]], ([-10.], [10.]), max_nfev=1)
    result = fit_regression(lambda p: np.array([p[0]-2]), [[0.]], ([-1.], [1.]))
    assert result.x[0] == pytest.approx(1., abs=1e-8)
    assert projected_gradient_cosine(result.fun, result.jac, result.x, [-1.], [1.]) == 0.
