"""Custom Psat uses shared property paths and survives complete PFD export."""

import math

import numpy as np
import pytest

from pfd_parser import parse_pfd
from simulator import Simulator
from thermodynamics_models.interaction_fitting import (
    prepare_fit,
    fit_interactions,
    export_fit,
)
from thermodynamics_models.fitting_psat import normalize_psat
from property_resolution.log_correlations import (
    dippr101_log_value,
    dippr101_log_derivative,
)
from property_resolution.vapor_pressure_adapter import _psat_correlation_functions
from .test_interaction_fitting import request


@pytest.mark.parametrize(
    "form",
    [
        "antoine",
        "dippr101",
        "canonical_psat_af",
        "canonical_psat_ag",
        "canonical_psat_ah",
    ],
)
def test_custom_psat_obscure_components_fit_and_export(form):
    # All five inputs represent ln(P/bar)=5-1000/T. Its Tc/Pc anchor is exact.
    coefficients = {
        key: 0.0
        for key in "ABCDEFGH"[
            : {
                "antoine": 3,
                "dippr101": 5,
                "canonical_psat_af": 6,
                "canonical_psat_ag": 7,
                "canonical_psat_ah": 8,
            }[form]
        ]
    }
    coefficients.update(A=5, B=-1000)
    spec = {
        "form": form,
        "coefficients": coefficients,
        "Tmin_K": 280,
        "Tmax_K": 500,
        "source": 'Synthetic curve for fitting test\nThermochimica "source", original metadata retained',
    }
    if form == "antoine":
        spec.update(temperature_unit="K", pressure_unit="bar")
        spec["coefficients"] = {"A": 5 / math.log(10), "B": 1000 / math.log(10), "C": 0}
    if form == "dippr101":
        spec.update(pressure_unit="pa")
        spec["coefficients"].update(A=5 + math.log(1e5), E=1)
    if form == "canonical_psat_ah":
        spec["inverse_power"] = -3
    data = request(
        components=["Obscure Fit Substance A", "Obscure Fit Substance B"],
        form="inverse",
        component_properties=[
            {
                "MW": 50,
                "Tc_K": 500,
                "Pc_bar": math.exp(3),
                "Tb_K": 1000 / (5 - math.log(1.01325)),
                "omega": 0.1,
            },
            {
                "MW": 70,
                "Tc_K": 500,
                "Pc_bar": math.exp(3),
                "Tb_K": 1000 / (5 - math.log(1.01325)),
                "omega": 0.1,
            },
        ],
        psat=[spec, spec],
    )
    # A full canonical override avoids unnecessary completion fitting in this test.
    spec["Tmin_K"] = 100
    problem = prepare_fit(data)
    for T in (300, 350):
        assert problem.thermo.Psat(problem.components[0], T) == pytest.approx(
            math.exp(5 - 1000 / T), rel=1e-13
        )
    truth = np.array([1.2, 0.6])
    problem.install(truth)
    rows = []
    for T in (300, 350):
        rows.append(
            {
                "kind": "GAMMA_INF",
                "T_K": T,
                "gamma1_inf": problem.checked_gamma(T, 0)[problem.components[0]],
                "gamma2_inf": problem.checked_gamma(T, 1)[problem.components[1]],
            }
        )
    data["observations"] = rows
    fitted = fit_interactions(data)
    assert fitted["success"]
    restored = Simulator.from_string(fitted["pfd_text"])
    restored.initialize()
    # Export retains its normal runtime canonicalization; only the fitter uses
    # direct supplied Psat. The original exported definition stays portable.
    assert restored.thermo_packages["global"].Psat(problem.components[0],325)>0
    target = parse_pfd(fitted["definition_pfd"])
    target.components[0].property_correlations.clear()
    target.components[0].antoine_A = target.components[0].antoine_B = target.components[
        0
    ].antoine_C = None
    target.components[0].antoine_Tmin = target.components[0].antoine_Tmax = None
    merged = export_fit(fitted, pfd_text=target.to_pfd())
    assert (
        parse_pfd(merged["pfd_text"]).components[0]
        == parse_pfd(fitted["pfd_text"]).components[0]
    )


def test_dippr_parser_and_analytic_derivative_use_shared_formula():
    coefficients = {"A": 15, "B": -2500, "C": -1.5, "D": 1e-6, "E": 2.2}
    pfd = parse_pfd(
        "COMPONENTS:\n    X | Custom | MW=50, Tc=500, Pc=30\nPROPERTY_CORRELATIONS:\n    X.Psat | equation=dippr_eq101, Tmin_K=250, Tmax_K=450, A=15, B=-2500, C=-1.5, D=0.000001, E=2.2\n"
    )
    restored = parse_pfd(pfd.to_pfd())
    assert (
        restored.components[0].property_correlations
        == pfd.components[0].property_correlations
    )
    evaluate, derivative = _psat_correlation_functions(
        "dippr_eq101", coefficients, {}, {}
    )
    assert evaluate(325) == dippr101_log_value(325, coefficients)
    assert derivative(325) == dippr101_log_derivative(325, coefficients)
    assert derivative(325) == pytest.approx(
        (evaluate(325.001) - evaluate(324.999)) / 0.002, rel=1e-7
    )


def test_custom_psat_rejects_ambiguous_units_and_singular_antoines():
    with pytest.raises(ValueError, match="singular"):
        normalize_psat(
            {
                "form": "antoine",
                "coefficients": {"A": 5, "B": 1000, "C": -50},
                "Tmin_K": 300,
                "Tmax_K": 400,
            }
        )
    with pytest.raises(ValueError, match="Kelvin"):
        normalize_psat(
            {
                "form": "dippr101",
                "coefficients": dict.fromkeys("ABCDE", 0),
                "Tmin_K": 300,
                "Tmax_K": 400,
                "temperature_unit": "C",
            }
        )
