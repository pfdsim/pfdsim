"""Distinguish predictive R² from correlation and preserve CV loss semantics."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_nannoolal_metrics import headline_metrics


def test_perfect_correlation_does_not_imply_good_predictive_r2():
    target = np.array([-3., -2., -1.])
    prediction = np.repeat((target + 2)[:, None], 2, axis=1)
    metrics = headline_metrics(prediction, target, np.arange(3))['metrics']
    assert np.isclose(metrics['Pearson_r'], 1)
    assert metrics['R2'] < 0
    assert np.isclose(metrics['MAE_log10'], 2)


def test_repeated_predictions_are_scored_before_averaging_and_invalids_retained():
    target = np.array([-3., -2., -1., 0.])
    prediction = target[:, None] + np.array([-1., 1.])[None, :]
    prediction[-1, 0] = 4
    result = headline_metrics(prediction, target, np.arange(4))
    assert result['valid_all_repeats'] == 3
    assert result['invalid_supported_predictions'] == 1
    assert np.isclose(result['metrics']['MAE_log10'], 1)
    assert np.isclose(result['metrics']['RMSE_log10'], 1)
