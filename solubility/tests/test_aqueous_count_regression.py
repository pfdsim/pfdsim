"""Numerical checks for count and size regressions without running the study."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark_aqueous_count_regression import ridge_grid,self_check,size_basis


def test_dependent_counts_and_nonlinear_size_recoveries():
    result = self_check()
    assert result['rank_deficient_OLS_prediction']
    assert result['nonlinear_size_recovery']
    assert result['nested_training_only_size_selection']


def test_unseen_element_is_flagged_without_test_set_scaling():
    train = np.array([[0.,1.],[0.,2.],[0.,3.]])
    test = np.array([[7.,4.]])
    prediction,coefficients,_,unseen = ridge_grid(train,np.array([2.,4.,6.]),test,np.array([0.]))
    assert np.allclose(prediction,[[8.]])
    assert coefficients[0,0]==0
    assert unseen.tolist()==[True]


def test_size_bases_handle_zero_heavy_atoms():
    for form in ('linear','log','surface','quadratic','flexible'):
        assert np.all(np.isfinite(size_basis(np.array([0.,1.,10.]),form)))
