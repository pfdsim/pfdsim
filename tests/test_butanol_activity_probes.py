"""Observation isolation in the maintained butanol research probes."""

import numpy as np
import pytest

from scripts.activity_fitting.probe_butanol_redlich_kister import RedlichKister, build_residual


def test_redlich_kister_residual_excludes_validation_and_disabled_observations():
    model = RedlichKister(None, 1, 1, [300, 330])
    request = {
        "weights": {"GAMMA_INF": 1, "LLE": 0},
        "scales": {"log_gamma": .01, "log_fugacity": .01},
        "observations": [
            {"id": "training", "kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 2},
        ],
    }
    baseline, initial, lower, upper = build_residual(model, request)
    request["observations"] += [
        {"id": "validation", "kind": "LLE", "T_K": 330, "x1_alpha": .1,
         "validation_only": True, "pin": True},
        {"id": "disabled-kind", "kind": "LLE", "T_K": 330, "x1_beta": .9},
        {"id": "disabled-row", "kind": "GAMMA_INF", "T_K": 330,
         "gamma1_inf": 100, "weight": 0},
    ]
    residual, actual, actual_lower, actual_upper = build_residual(model, request)
    np.testing.assert_array_equal(actual, initial)
    np.testing.assert_array_equal(actual_lower, lower)
    np.testing.assert_array_equal(actual_upper, upper)
    np.testing.assert_allclose(residual(actual), baseline(initial), atol=0)


@pytest.mark.parametrize("row,vapor,error", [
    ({"kind": "GAMMA_INF", "gamma1_inf": 2, "pin": True}, "IDEAL", "hard pins"),
    ({"kind": "HE", "HE_J_mol": 0, "x1": .3}, "IDEAL", "supports"),
    ({"kind": "VLE", "P_bar": 1, "x1": .3, "y1": .5}, "PR", "IDEAL vapor"),
])
def test_redlich_kister_probe_rejects_unsupported_training_contracts(row, vapor, error):
    model = RedlichKister(None, 1, 1, [300, 330])
    request = {
        "vapor": vapor, "weights": {row["kind"]: 1}, "scales": {},
        "observations": [{"id": "training", "T_K": 300, **row}],
    }
    with pytest.raises(ValueError, match=error):
        build_residual(model, request)
