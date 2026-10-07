"""Regional Psat policy, physical diagnostics, VLLE and held-out observations."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from pfd_parser import Component, ProcessFlowDiagram
from property_resolution.common import PropertyResolutionResult
from thermodynamics_models import fitting_properties as properties
from thermodynamics_models.interaction_fitting import (
    FitProblem,
    fit_interactions,
    prepare_fit,
    normalize_fit_request,
    parse_observations,
)
from .test_interaction_fitting import request, synthetic


def test_unavailable_validation_gamma_keeps_fit_and_plot_gaps(monkeypatch):
    checked_gamma = FitProblem.checked_gamma

    def unavailable_at_high_temperature(self, T, x):
        if T >= 400:
            raise ValueError("Activity law is unavailable at this temperature")
        return checked_gamma(self, T, x)

    monkeypatch.setattr(FitProblem, "checked_gamma", unavailable_at_high_temperature)
    data = request(form="constant")
    data["observations"].append({
        "kind": "GAMMA_INF", "T_K": 400, "gamma1_inf": 1,
        "validation_only": True,
    })
    result = fit_interactions(data)
    assert result["success"]
    assert not result["validation_only"]["points"][0]["physical"]
    plot = next(plot for plot in result["plots"] if plot["kind"] == "GAMMA_INF")
    assert plot["series"][0]["y"][-1] is None
    assert "unavailable" in plot["errors"][-1]["error"]


def test_unavailable_validation_lle_audit_does_not_abort_training_fit(monkeypatch):
    from thermodynamics_models.nrtl_uniquac import NRTLThermodynamics
    from thermodynamics_models.common import ThermodynamicsError

    def unavailable(*args, **kwargs):
        raise ThermodynamicsError("Unavailable LLE phase audit")

    monkeypatch.setattr(NRTLThermodynamics, "liquid_liquid_equilibrium", unavailable)
    data = request(form="constant")
    data["observations"].append({
        "kind": "LLE", "T_K": 400,
        "x1_alpha": 0.1, "x1_beta": 0.9,
        "validation_only": True,
    })
    result = fit_interactions(data)
    assert result["success"]
    point = result["validation_only"]["points"][0]
    assert not point["physical"]
    assert point["predicted"]["equilibrium_error"] == "Unavailable LLE phase audit"
    metric = result["validation_only"]["physical_metrics"]["LLE"]["x1_alpha"]
    assert metric["n"] == 0 and metric["unavailable"] == 1


def test_cross_validation_preserves_nonideal_vapor_for_liquid_only_training():
    data = request(vapor="PR", cv={"method": "leave_group_out"})
    for row in data["observations"]:
        row["group"] = "liquid"
    row = {"kind": "VLE", "T_K": 350, "P_bar": 1, "x1": 0.4, "group": "vapor"}
    data["observations"].append(row)
    problem = prepare_fit(data)
    problem.install(np.zeros(4))
    prediction = problem.predict_vle(0.4, T=350)
    row.update(P_bar=prediction["P_bar"], y1=prediction["y1"])
    result = fit_interactions(data)
    fold = next(fold for fold in result["cross_validation"]["folds"] if fold["held_out_ids"] == ["3"])
    assert fold["success"], fold
    point = fold["points"][0]
    assert point["physical"], point
    assert max(abs(value) for value in point["scaled_residuals"]) < 1e-6


def test_lle_and_vlle_predictions_find_asymmetric_miscibility_gaps():
    from thermodynamics_models.factory import create_thermodynamics

    reference = create_thermodynamics(["butanol", "water"], "NRTL")
    T = 350
    tau, alpha, _ = reference._nrtl_cached_matrices(T)
    problem = prepare_fit(request(
        components=["butanol", "water"], form="constant", alpha=alpha[0][1],
    ))
    problem.install(np.array([tau[0][1], tau[1][0]]))
    split, first, second, _ = problem.thermo.liquid_liquid_equilibrium(
        problem.composition(0.1), T, tol=1e-8,
    )
    assert split
    endpoints = sorted((first[problem.components[0]], second[problem.components[0]]))
    assert 0 < endpoints[0] < endpoints[1] < 0.5
    lle = problem.predict_lle(T)
    assert [lle["x1_alpha"], lle["x1_beta"]] == pytest.approx(endpoints, abs=1e-6)
    vlle = problem.predict_vlle(T=T)
    assert vlle["phase_count"] == 3
    assert [vlle["x1_alpha"], vlle["x1_beta"]] == pytest.approx(endpoints, abs=1e-6)


@pytest.mark.parametrize("compiled_flash", [True, False])
def test_fitter_flashes_stationary_hints_into_stable_butanol_coexistence(compiled_flash, monkeypatch):
    from thermodynamics_models.fitting_diagnostics import build_objective_plots

    # Captured regression model: the feed-independent search returned nearly
    # collapsed stationary pairs at these two curve sampling temperatures.
    coefficients = np.array([
        -17.738987178093353, 513.6959740292948, 30.0, .10062049304041591,
        -.00016958498076724676, -22.902466499087573, 2150.628123486273,
        -14.610389286060201, .10062049304041591, -.00010320165407430895,
    ])
    expected = [
        (385.275, .03598380013755666, .2586304864303386),
        (391.5125, .04794195630619725, .214131459245179),
    ]
    problem = prepare_fit(request(
        components=["1-butanol", "water"], form="full", alpha=.2,
        observations=[{"kind":"LLE", "T_K":T, "x1_alpha":a, "x1_beta":b} for T,a,b in expected],
    ))
    values = coefficients / problem.scales
    problem.install(values)
    if not compiled_flash:
        monkeypatch.setattr(problem.thermo, "_compiled_lle_backend", lambda T=None: None)
    for T,a,b in expected:
        state = problem.predict_lle(T)
        assert [state["x1_alpha"], state["x1_beta"]] == pytest.approx([a,b], abs=1e-6)
        assert 0 < state["liquid_fraction"] < 1
        _, raw, stability = problem._lle_errors(T,state["x1_alpha"],state["x1_beta"],1)
        assert max(abs(value) for value in raw) < 1e-6
        assert stability["minimum_tangent_gap"] >= -1e-6
    report = problem.report(values, problem.request["observations"])
    assert all(point["physical"] for point in report["points"])
    if compiled_flash:
        plots = build_objective_plots(problem,values,problem.request["observations"],report["points"])
        assert not plots[0]["errors"]
        assert all(value is not None for series in plots[0]["series"][:2] for value in series["x"])


def test_isobaric_vle_plot_uses_stable_vlle_plateau():
    from thermodynamics_models.factory import create_thermodynamics
    from thermodynamics_models.fitting_diagnostics import build_objective_plots

    reference = create_thermodynamics(["butanol", "water"], "NRTL")
    T = 350
    tau, alpha, _ = reference._nrtl_cached_matrices(T)
    problem = prepare_fit(request(
        components=["butanol", "water"], form="constant", alpha=alpha[0][1],
    ))
    values = np.array([tau[0][1], tau[1][0]])
    problem.install(values)
    rows = parse_observations([
        {"kind":"VLE", "T_K":365, "P_bar":1, "x1":.01},
        {"kind":"VLE", "T_K":370, "P_bar":1, "x1":.4},
    ])
    plot = build_objective_plots(problem, values, rows, [])[0]
    assert not plot["errors"], plot["errors"]
    liquid, vapor = plot["series"][:2]
    endpoints = next(
        series for series in plot["series"]
        if series["name"] == "Predicted VLLE liquid endpoints"
    )
    invariant_vapor = next(
        series for series in plot["series"]
        if series["name"] == "Predicted VLLE vapor"
    )
    lean, rich = endpoints["x"]
    plateau = [
        temperature for x, temperature in zip(liquid["x"], liquid["y"])
        if lean <= x <= rich
    ]
    assert len(plateau) >= 2
    assert max(plateau) - min(plateau) < 1e-8
    assert invariant_vapor["y"] == pytest.approx([plateau[0]])
    for liquid_x, temperature in zip(liquid["x"], vapor["y"]):
        if lean <= liquid_x <= rich and temperature is not None:
            assert temperature == pytest.approx(plateau[0], abs=1e-8)


@pytest.mark.parametrize("interval", [(.411, .419), (.413, .413)])
@pytest.mark.parametrize("available", [True, False])
def test_isobaric_objective_samples_only_the_observation_interval(monkeypatch, interval, available):
    from thermodynamics_models.fitting_diagnostics import build_objective_plots

    problem = prepare_fit(request(form="constant"))
    thermo = problem.thermo

    def no_invariant(**kwargs):
        raise ValueError("No VLLE invariant")

    def bubble(composition, pressure):
        assert interval[0] <= composition[problem.components[0]] <= interval[1]
        if not available:
            raise ValueError("Unavailable bubble state")
        return 350

    monkeypatch.setattr(problem, "predict_vlle", no_invariant)
    monkeypatch.setattr(thermo, "bubble_point_T", bubble)
    monkeypatch.setattr(thermo, "K_values", lambda *args: dict.fromkeys(problem.components, 1))
    monkeypatch.setattr(thermo, "_check_diagram_vle", lambda *args: None)
    monkeypatch.setattr(thermo, "dew_point_T", lambda *args: pytest.fail("Unused dew solve"))
    rows = parse_observations([
        {"kind": "VLE", "T_K": T, "P_bar": 1, "x1": x}
        for T, x in zip((349, 351), interval)
    ])
    plot = build_objective_plots(problem, problem.initial, rows, [])[0]
    liquid, vapor = plot["series"][:2]
    assert liquid["x"][0] == interval[0]
    assert liquid["x"][-1] == interval[1]
    assert len(liquid["x"]) == (21 if interval[0] != interval[1] else 1)
    if available:
        assert not plot["errors"]
        assert liquid["y"] == pytest.approx([76.85] * len(liquid["x"]))
        assert vapor["x"] == pytest.approx(liquid["x"])
    else:
        assert len(plot["errors"]) == len(liquid["x"])
        assert all(T is None for T in liquid["y"])
        assert all("Unavailable bubble state" in error["error"] for error in plot["errors"])


def test_calorimetry_plot_rejects_clipped_interior_states(monkeypatch):
    from thermodynamics_models.fitting_diagnostics import build_objective_plots

    problem = prepare_fit(request(form="constant"))
    checked_gamma = problem.checked_gamma

    def clipped_at_middle(T, x):
        if x == 0.5:
            raise ValueError("Runtime activity-coefficient limit reached")
        return checked_gamma(T, x)

    monkeypatch.setattr(problem, "checked_gamma", clipped_at_middle)
    rows = parse_observations([
        {"kind": "HE", "T_K": 300, "x1": x, "HE_J_mol": 0} for x in (0.2, 0.8)
    ])
    plot = build_objective_plots(problem, np.zeros(2), rows, [])[0]
    assert plot["series"][0]["y"][10] is None
    assert plot["errors"][0]["x1"] == 0.5


def test_validation_only_does_not_change_coefficients_and_has_separate_metrics():
    data, _, _ = synthetic(kinds=("HE", "GAMMA_INF"))
    baseline = fit_interactions(data)
    extra = deepcopy(data)
    extra["observations"].append(
        {
            "kind": "HE",
            "T_K": 325,
            "x1": 0.5,
            "HE_J_mol": 10000,
            "validation_only": True,
        }
    )
    extra["cv"] = {"method": "kfold", "folds": 2}
    result = fit_interactions(extra)
    assert result["coefficients"] == pytest.approx(baseline["coefficients"], abs=1e-6)
    assert result["validation_only"]["physical_metrics"]["HE"]["HE_J_mol"]["MAE"] > 9000
    assert result["physical_metrics"]["HE"]["HE_J_mol"]["RMSE"] < 1e-5
    held_id = result["points"][-1]["id"]
    for fold in result["cross_validation"]["folds"]:
        assert (
            held_id not in fold.get("training_ids", [])
            and held_id not in fold["held_out_ids"]
        )
    assert any(
        series.get("role") == "validation"
        for plot in result["plots"]
        for series in plot["series"]
    )
    assert result["points"][-1]["role"] == "validation"


def test_validation_pin_conflict_and_no_training_are_explicit_errors():
    with pytest.raises(ValueError, match="cannot be a hard pin"):
        parse_observations(
            [
                {
                    "kind": "HE",
                    "T_K": 300,
                    "x1": 0.5,
                    "HE_J_mol": 100,
                    "validation_only": True,
                    "pin": True,
                }
            ]
        )
    with pytest.raises(ValueError, match="training observation"):
        normalize_fit_request(
            request(
                observations=[
                    {
                        "kind": "GAMMA_INF",
                        "T_K": 300,
                        "gamma1_inf": 2,
                        "validation_only": True,
                    }
                ]
            )
        )


def test_validation_only_flags_survive_table_import():
    parsed = parse_observations(
        "kind,T_K,x1,HE_J_mol,validation_only\nHE,300,0.5,100,true"
    )
    assert parsed[0]["validation_only"] is True


def test_quality_gate_checks_interior_source_regions_and_includes_vlle(monkeypatch):
    from property_resolution.resolver import PropertyResolver

    monkeypatch.setattr(
        PropertyResolver,
        "vapor_pressure_quality_samples",
        lambda *args, **kwargs: [
            (300, PropertyResolutionResult(1, "measured", "experiment", 0.995)),
            (320, PropertyResolutionResult(1, "estimated", "completion", 0.8)),
            (350, PropertyResolutionResult(1, "measured", "experiment", 0.995)),
        ],
    )
    for kind in ("VLE", "VLLE"):
        row = {"kind": kind, "T_K": 350, "P_bar": 1}
        if kind == "VLE":
            row["x1"] = 0.5
        with pytest.raises(ValueError, match="320.*quality 0.8"):
            prepare_fit(request(form="constant", observations=[row]))


def test_physical_vle_errors_and_real_curves_use_runtime_predictions():
    data, _, _ = synthetic()
    result = fit_interactions(data)
    metrics = result["physical_metrics"]
    assert metrics["VLE"]["P_bar"]["RMSE"] < 1e-7
    assert metrics["VLE"]["T_K"]["MAE"] < 1e-4
    assert metrics["VLE"]["y1"]["MAE"] < 1e-6
    assert metrics["HE"]["HE_J_mol"]["MAE"] < 1e-4
    assert any(
        plot["kind"] == "VLE"
        and any(
            series["mode"] == "line" and len(series["x"]) == 21
            for series in plot["series"]
        )
        for plot in result["plots"]
    )
    assert any(plot["kind"] == "HE" for plot in result["plots"])


def test_low_quality_psat_fails_only_for_vapor_objectives(monkeypatch):
    from property_resolution.resolver import PropertyResolver

    def low_quality(*args, **kwargs):
        return [(350, PropertyResolutionResult(1, "estimated", "nannoolal", 0.85))]

    monkeypatch.setattr(PropertyResolver, "vapor_pressure_quality_samples", low_quality)
    with pytest.raises(ValueError, match="quality.*0.85"):
        prepare_fit(
            request(
                form="constant",
                observations=[{"kind": "VLE", "T_K": 350, "P_bar": 1, "x1": 0.5}],
            )
        )
    liquid = prepare_fit(
        request(
            vapor="HOC",
            form="constant",
            components=["Unknown QA", "Unknown QB"],
            component_properties=[{"MW": 50}, {"MW": 60}],
            observations=[{"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 2}],
        )
    )
    assert liquid.thermo.vapor_eos is None
    assert not any(item["property"] == "Psat" for item in liquid.property_records)


@pytest.mark.parametrize(
    "vapor", ["IDEAL", "RK", "PR", "VDM", "TSONOPOULOS", "PITZER-CURL", "ABBOTT", "HOC"]
)
def test_all_vapor_models_have_audited_requirements_and_qualified_psat(vapor):
    problem = prepare_fit(
        request(
            vapor=vapor,
            form="constant",
            observations=[{"kind": "VLE", "T_K": 350, "P_bar": 1, "x1": 0.4}],
        )
    )
    assert any(
        item["property"] == "Psat" and item["quality"] >= 0.9
        for item in problem.property_records
    )
    if vapor in ("PITZER-CURL", "ABBOTT"):
        assert not any(
            item["property"] == "dipole_moment" for item in problem.property_records
        )
    if vapor == "HOC":
        assert any(item["property"] == "hoc_eta" for item in problem.property_records)


def test_dipole_estimate_below_point_nine_is_allowed_with_warning(monkeypatch):
    resolver = SimpleNamespace(
        resolve_dipole_moment=lambda *args, **kwargs: PropertyResolutionResult(
            1.6,
            "functional-class estimate",
            "functional_estimate",
            0.4,
            "Estimated dipole",
        )
    )
    monkeypatch.setattr(properties, "get_property_resolver", lambda: resolver)
    props = SimpleNamespace(
        CAS="67-64-1",
        name="acetone",
        formula="C3H6O",
        smiles="CC(=O)C",
        Tc=508.1,
        Pc=47,
        Vc=209,
        omega=0.3,
        dipole_moment=None,
        property_sources={},
    )
    thermo = SimpleNamespace(
        components=["A"],
        props={"A": props},
        _resolver_known_props={"A": {"Tc": 508.1, "Pc": 47, "Vc": 209, "omega": 0.3}},
    )
    definition = ProcessFlowDiagram(components=[Component("A", "acetone")])
    records, warnings = properties.prepare_auxiliary_properties(
        thermo,
        definition,
        {
            "vapor": "TSONOPOULOS",
            "online_lookup": False,
            "estimate_properties": True,
            "allow_hoc_eta_default": True,
        },
    )
    assert any(
        item["property"] == "dipole_moment" and item["quality"] == 0.4
        for item in records
    )
    assert any("dipole" in warning for warning in warnings)
    assert definition.components[0].dipole_moment == 1.6


def test_selected_data_type_fills_missing_json_kind_without_overwriting_explicit_kind():
    from thermodynamics_models.interaction_fitting import inspect_observations

    data = [{"T_K": 300, "x1": 0.5, "HE_J_mol": 100}]
    assert (
        inspect_observations(data, import_options={"kind": "HE"})["observations"][0][
            "kind"
        ]
        == "HE"
    )
    assert "kind" not in data[0]


def test_vlle_known_and_inferred_endpoints_use_a_shared_vapor_state():
    data = request(
        form="constant",
        starts=2,
        observations=[
            {
                "kind": "VLLE",
                "T_K": 330,
                "P_bar": 1,
                "x1_alpha": 0.05,
                "x1_beta": 0.95,
                "y1": 0.5,
            }
        ],
    )
    problem = prepare_fit(data)
    truth = np.array([2.5, 2.5])
    problem.install(truth)
    state = problem.predict_vlle(T=330)
    assert state["phase_count"] == 3
    data["observations"] = [
        {
            "kind": "VLLE",
            "T_K": 330,
            "P_bar": state["P_bar"],
            "y1": state["y1"],
            "x1_alpha": state["x1_alpha"],
            "x1_beta": state["x1_beta"],
        }
    ]
    fitted = fit_interactions(data)
    assert fitted["success"], fitted["warnings"]
    assert fitted["physical_metrics"]["VLLE"]["x1_alpha"]["MAE"] < 1e-4
    assert fitted["physical_metrics"]["VLLE"]["P_bar"]["MAE"] < 1e-5
    inferred = deepcopy(data)
    for key in ("x1_alpha", "x1_beta"):
        inferred["observations"][0].pop(key)
    inferred["initial"] = {
        "12.constant": 2.5,
        "21.constant": 2.5,
        "vlle_xa.1": state["x1_alpha"],
        "vlle_gap.1": (state["x1_beta"] - state["x1_alpha"]) / (1 - state["x1_alpha"]),
    }
    fitted = fit_interactions(inferred)
    assert fitted["success"], fitted["warnings"]
    assert fitted["points"][0]["predicted"]["phase_count"] == 3
