"""Identifiable solvent/logP transfers, held-out exclusion and singular designs."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_logp_transfer import accumulate_logp, fit_logp, predict_logp, slope_support
from benchmark_solvent_transfer_multipliers import contributions


def synthetic(logp):
    a, b = np.array([0., .4, -.8]), np.array([0., .6, -.1])
    training = [(i, np.arange(3), a + b*p + 100*i) for i, p in enumerate(logp)]
    return training, a, b


def test_known_solvent_offsets_slopes_and_unknown_solvent():
    logp = np.array([-2., 0., 2., 4.])
    training, a, b = synthetic(logp)
    matrices, vectors = accumulate_logp(contributions(training), logp, np.zeros(4, int), 4, 1)
    model = fit_logp(matrices[0], vectors[0], logp.mean(), logp.std(), 0)
    reference, target, p = np.array([0, 1, 2, 0]), np.array([1, 2, 0, 3]), np.array([3., 3., 3., 3.])
    prediction = predict_logp(model, reference, target, p)
    assert np.allclose(prediction[:3], a[target[:3]]-a[reference[:3]] + 3*(b[target[:3]]-b[reference[:3]]), atol=1e-10)
    assert np.isnan(prediction[3])


def test_rank_deficient_slopes_only_allow_identifiable_logp_contrasts():
    logp = np.array([2., 2., 2.])
    training, a, b = synthetic(logp)
    matrices, vectors = accumulate_logp(contributions(training), logp, np.zeros(3, int), 3, 1)
    model = fit_logp(matrices[0], vectors[0], 2., 1., 0)
    prediction = predict_logp(model, np.array([0, 0]), np.array([1, 1]), np.array([2., 3.]))
    assert np.isclose(prediction[0], a[1] + 2*b[1], atol=1e-10)
    assert np.isnan(prediction[1])


def test_held_out_response_cannot_change_fitted_transfer():
    logp = np.array([-2., 0., 2., 4.])
    training, _, _ = synthetic(logp)
    folds = np.array([0, 1, 1, 1])
    def trained(groups):
        matrices, vectors = accumulate_logp(contributions(groups), logp, folds, 3, 2)
        return fit_logp(matrices[1], vectors[1], logp[1:].mean(), logp[1:].std(), 0)
    original = trained(training)
    changed = training.copy()
    changed[0] = (0, np.arange(3), np.array([1000., -1000., 9000.]))
    new = trained(changed)
    assert np.allclose(original['beta'], new['beta'], atol=1e-12)


def test_sparse_slopes_are_zero_and_repeated_groups_do_not_inflate_support():
    logp = np.array([-2., 0., 2., 4.])
    training, _, _ = synthetic(logp)
    # Extra observations cannot supply more independent training molecules.
    counted = training + [training[0]]*100 + [training[1]]*100
    folds = np.array([0, 1, 1, 1])
    counts, enabled = slope_support(counted, logp, folds, 0, 3, minimum=4)
    assert counts.tolist() == [3, 3, 3]
    assert not np.any(enabled)
    matrices, vectors = accumulate_logp(contributions(training), logp, folds, 3, 2)
    model = fit_logp(matrices[1], vectors[1], logp[1:].mean(), logp[1:].std(), 0, enabled)
    assert np.all(model['beta'][1] == 0)
    prediction = predict_logp(model, np.array([0, 0]), np.array([1, 1]), np.array([-100., 100.]))
    assert np.all(np.isfinite(prediction))
    assert np.isclose(prediction[0], prediction[1])
