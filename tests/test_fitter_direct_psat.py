"""Fitter-only supplied Psat evaluation, with no shared canonical refitting."""

import math

import numpy as np
import pytest

from pfd_parser import parse_pfd
from property_resolution.vapor_pressure import VaporPressureMixin
from thermodynamics_models.common import ThermodynamicsError
from thermodynamics_models.interaction_fitting import prepare_fit, fit_interactions
from .test_interaction_fitting import request


def specification(form):
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
        "Tmin_K": 290,
        "Tmax_K": 380,
        "source": "Original supplied correlation",
    }
    if form == "antoine":
        spec.update(temperature_unit="K", pressure_unit="kpa")
        spec["coefficients"] = {
            "A": 5 / math.log(10) + 2,
            "B": 1000 / math.log(10),
            "C": 0,
        }
    elif form == "dippr101":
        spec.update(pressure_unit="pa")
        spec["coefficients"].update(A=5 + math.log(1e5), D=1e-7, E=2.4)
    elif form == "canonical_psat_ag":
        spec["coefficients"]["G"] = 2e-9
    elif form == "canonical_psat_ah":
        spec.update(inverse_power=-5)
        spec["coefficients"].update(G=2e-9, H=0.08)
    return spec


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
def test_all_native_forms_are_exact_and_never_refitted(form, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Supplied fitting Psat must not enter shared canonicalization")

    monkeypatch.setattr(
        VaporPressureMixin, "_canonical_vapor_pressure_runtime", forbidden
    )
    spec = specification(form)
    data = request(
        form="constant",
        psat=[spec, spec],
        component_properties=[{"Tc_K": 500}, {"Tc_K": 500}],
        observations=[{"kind": "VLE", "T_K": 330, "P_bar": 1, "x1": 0.5}],
    )
    problem = prepare_fit(data)
    for T in (290, 315, 330, 360, 380):
        value = 5 - 1000 / T
        if form == "dippr101":
            value += 1e-7 * T**2.4
        if form in ("canonical_psat_ag", "canonical_psat_ah"):
            value += 2e-9 * T**3
        if form == "canonical_psat_ah":
            value += 0.08 * ((T / 500) ** -5 - 1)
        assert problem.thermo.Psat(problem.components[0], T) == pytest.approx(
            math.exp(value), rel=1e-13
        )
    assert all(
        record["method"].startswith("fitter_direct_")
        for record in problem.property_records
        if record["property"] == "Psat"
    )
    for T in (math.nextafter(290, -math.inf), math.nextafter(380, math.inf)):
        with pytest.raises(ThermodynamicsError, match="outside its declared range"):
            problem.thermo.Psat(problem.components[0], T)


def test_out_of_range_observation_fails_before_optimization():
    spec = specification("dippr101")
    data = request(
        form="constant",
        psat=[spec, spec],
        observations=[{"kind": "VLE", "T_K": 385, "P_bar": 1, "x1": 0.5}],
    )
    with pytest.raises(ThermodynamicsError, match="385 K.*290.*380"):
        prepare_fit(data)


def test_fitted_correlations_and_vapor_reference_use_direct_psat(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No supplied fitting curve may be canonically refitted")

    monkeypatch.setattr(
        VaporPressureMixin, "_canonical_vapor_pressure_runtime", forbidden
    )
    first = specification("antoine")
    second = specification("dippr101")
    data = request(
        form="constant",
        starts=1,
        psat=[first, second],
        observations=[{"kind": "VLE", "T_K": 330, "P_bar": 1, "x1": 0.5}],
    )
    problem = prepare_fit(data)
    truth = np.array([0.5, 1.1])
    problem.install(truth)
    rows = []
    for T in (310, 330, 350):
        for x in (0.2, 0.5, 0.8):
            gamma = problem.checked_gamma(T, x)
            first_term = (
                x
                * gamma[problem.components[0]]
                * problem.thermo.Psat(problem.components[0], T)
            )
            second_term = (
                (1 - x)
                * gamma[problem.components[1]]
                * problem.thermo.Psat(problem.components[1], T)
            )
            P = first_term + second_term
            rows.append(
                {"kind": "VLE", "T_K": T, "P_bar": P, "x1": x, "y1": first_term / P}
            )
    data["observations"] = rows
    result = fit_interactions(data)
    assert result["success"], result["warnings"]
    assert result["optimizer"]["objective"] < 1e-8
    assert result["coefficients"]["12.constant"] == pytest.approx(0.5, abs=1e-5)
    assert result["coefficients"]["21.constant"] == pytest.approx(1.1, abs=1e-5)
    assert parse_pfd(result["pfd_text"]).components[0].antoine_A is not None


def test_direct_psat_survives_fitted_vapor_parameter_reconstruction():
    spec = specification("antoine")
    data = request(
        form="constant",
        vapor="PR",
        psat=[spec, spec],
        vapor_parameters=[
            {
                "model": "PR",
                "field": "kij",
                "fit": True,
                "value": 0,
                "lower": -0.5,
                "upper": 0.5,
            }
        ],
        observations=[{"kind": "VLE", "T_K": 330, "P_bar": 1, "x1": 0.5}],
    )
    problem = prepare_fit(data)
    for value in (0, 0.07, 0.12):
        parameters = problem.initial.copy()
        parameters[-1] = value
        problem.install(parameters)
        assert problem.thermo.Psat(problem.components[0], 330) == pytest.approx(
            math.exp(5 - 1000 / 330), rel=1e-13
        )
        _, _, prediction = problem.row_errors(
            parameters, problem.request["observations"][0]
        )
        assert math.isfinite(prediction["evaluation_y1"])
        with pytest.raises(ThermodynamicsError, match="outside its declared range"):
            problem.thermo.Psat(problem.components[0], 381)


def test_imported_pfd_correlation_is_also_evaluated_directly(monkeypatch):
    monkeypatch.setattr(
        VaporPressureMixin,
        "_canonical_vapor_pressure_runtime",
        lambda *args, **kwargs: pytest.fail("Import must not refit supplied Psat"),
    )
    pfd = """ONLINE_LOOKUP: false
COMPONENTS:
    A | acetone
    B | water
PROPERTY_CORRELATIONS:
    A.Psat | equation=dippr_eq101, Tmin_K=300, Tmax_K=360, A=5, B=-1000, C=0, D=0.0000001, E=2.4
    B.Psat | equation=canonical_psat_af, Tmin_K=300, Tmax_K=360, A=5, B=-1000, C=0, D=0, E=0, F=0
"""
    problem = prepare_fit(
        request(
            components=["A", "B"],
            pfd_text=pfd,
            form="constant",
            observations=[{"kind": "VLE", "T_K": 330, "P_bar": 1, "x1": 0.5}],
        )
    )
    assert problem.thermo.Psat("A", 330) == pytest.approx(
        math.exp(5 - 1000 / 330 + 1e-7 * 330**2.4), rel=1e-13
    )
    assert problem.thermo.Psat("B", 330) == pytest.approx(
        math.exp(5 - 1000 / 330), rel=1e-13
    )


def test_unsupplied_component_still_uses_normal_resolver_and_no_shared_method_is_replaced():
    from thermodynamics_models.nrtl_uniquac import NRTLThermodynamics

    original = NRTLThermodynamics.Psat
    spec = specification("antoine")
    problem = prepare_fit(
        request(
            form="constant",
            psat=[spec, None],
            observations=[{"kind": "VLE", "T_K": 330, "P_bar": 1, "x1": 0.5}],
        )
    )
    assert NRTLThermodynamics.Psat is original
    assert problem.thermo.Psat(problem.components[0], 330) == pytest.approx(
        math.exp(5 - 1000 / 330), rel=1e-13
    )
    record = next(
        record
        for record in problem.property_records
        if record["component"] == problem.components[1] and record["property"] == "Psat"
    )
    assert not record["method"].startswith("fitter_direct_")


@pytest.mark.parametrize(
    "vapor", ["IDEAL", "RK", "PR", "VDM", "TSONOPOULOS", "PITZER-CURL", "ABBOTT", "HOC"]
)
def test_every_vapor_reference_reads_fitter_local_supplied_pressure(vapor, monkeypatch):
    from thermodynamics_models.base import IdealThermodynamics

    monkeypatch.setattr(
        IdealThermodynamics,
        "get_Psat_coefficients",
        lambda *args, **kwargs: pytest.fail(
            "Fitting must not request canonical coefficients for supplied Psat"
        ),
    )
    spec = specification("antoine")
    data = request(
        vapor=vapor,
        form="constant",
        psat=[spec, spec],
        observations=[{"kind": "VLE", "T_K": 330, "P_bar": 1, "x1": 0.5}],
    )
    problem = prepare_fit(data)
    _, _, prediction = problem.row_errors(
        problem.initial, problem.request["observations"][0]
    )
    assert math.isfinite(prediction["evaluation_y1"])
    assert problem.thermo.Psat(problem.components[0], 330) == pytest.approx(
        math.exp(5 - 1000 / 330), rel=1e-13
    )


def test_supplied_psat_does_not_force_unused_critical_anchors(monkeypatch):
    from property_resolution.resolver import PropertyResolver

    spec = specification("dippr101")
    problem = prepare_fit(
        request(
            components=["Own component A", "Own component B"],
            component_properties=[{"MW": 50}, {"MW": 70}],
            form="constant",
            psat=[spec, spec],
            observations=[{"kind": "GAMMA_INF", "T_K": 330, "gamma1_inf": 2}],
        )
    )
    for props in problem.thermo.props.values():
        props.Tc = props.Pc = None
    from thermodynamics_models.fitting_properties import prepare_auxiliary_properties

    monkeypatch.setattr(
        PropertyResolver,
        "resolve_critical_properties",
        lambda *args, **kwargs: pytest.fail(
            "Direct IDEAL supplied Psat does not need critical anchors"
        ),
    )
    records, _ = prepare_auxiliary_properties(
        problem.thermo, problem.definition, {**problem.request, "vapor": "IDEAL"}
    )
    assert not any(record["property"] in ("Tc", "Pc") for record in records)
