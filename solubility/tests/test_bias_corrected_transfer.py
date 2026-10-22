"""Verify calibration identity, held-out exclusion and uncertainty families."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark_bias_corrected_transfer import self_check
from benchmark_solvent_transfer_multipliers import difference_bootstrap


def test_calibration_does_not_create_information_or_use_held_out_errors():
    result = self_check()
    assert result['solvent_only_UNIFAC_reduces_to_matched_baseline']
    assert result['held_out_error_perturbation_does_not_change_fit']
    assert result['known_bias_recovered']
    assert result['unseen_solvent_unavailable']


def test_requested_multiplicity_preserves_existing_statistics():
    delta = np.random.default_rng(42).normal(size=(35,10))
    original = difference_bootstrap(delta,123,2000)
    extended = difference_bootstrap(delta,123,2000,requested_family_size=60)
    for key,value in original.items():
        assert extended[key]==value
    lower,upper = extended['paired_requested_family_95pct_CI']
    old_lower,old_upper = original['paired_familywise_95pct_CI_six_variants']
    assert lower<=old_lower<=old_upper<=upper
