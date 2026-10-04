"""Experimental regression, physical constraints and lossless PFD application."""

from copy import deepcopy
import json
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from cli import main as cli_main
from pfd_parser import parse_pfd
from simulator import Simulator
from thermodynamics_models.interaction_fitting import (
    export_fit,
    fit_interactions,
    normalize_fit_request,
    parse_observations,
    prepare_fit,
)


def request(**settings):
    return {
        "components": ["ethanol", "water"],
        "model": "NRTL",
        "starts": 1,
        "form": "constant_inverse",
        "observations": [
            {"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 1, "gamma2_inf": 1},
            {"kind": "GAMMA_INF", "T_K": 350, "gamma1_inf": 1, "gamma2_inf": 1},
        ],
        **settings,
    }


def synthetic(settings=None, values=None, kinds=("VLE", "HE", "GAMMA_INF")):
    settings = request(**(settings or {}))
    problem = prepare_fit(settings)
    if values is None:
        values = [0.4, 1.5, -0.2, 0.8] + ([0.42] if settings.get("fit_alpha") else [])
    values = np.array(values)
    rows = []
    for T in (300, 325, 350):
        problem.install(values)
        if "GAMMA_INF" in kinds:
            rows.append(
                {
                    "kind": "GAMMA_INF",
                    "T_K": T,
                    "gamma1_inf": problem.thermo.activity_coefficients(
                        T, problem.composition(0)
                    )[problem.components[0]],
                    "gamma2_inf": problem.thermo.activity_coefficients(
                        T, problem.composition(1)
                    )[problem.components[1]],
                }
            )
        for x in (0.2, 0.5, 0.8):
            if "HE" in kinds:
                rows.append(
                    {
                        "kind": "HE",
                        "T_K": T,
                        "x1": x,
                        "HE_J_mol": problem.thermo.excess_enthalpy(
                            problem.composition(x), T
                        ),
                    }
                )
            if "VLE" in kinds:
                liquid = problem.thermo._phase_log_fugacities(
                    T, 1, problem.composition(x), "liquid"
                )
                pressure = sum(math.exp(liquid[c]) for c in problem.components)
                y = math.exp(liquid[problem.components[0]]) / pressure
                rows.append(
                    {"kind": "VLE", "T_K": T, "x1": x, "P_bar": pressure, "y1": y}
                )
    settings["observations"] = rows
    return settings, problem, values


def test_markdown_json_datasets_and_units():
    rows = parse_observations("""```markdown
| kind | T_C | P_kPa | x1 | weight | pin |
| :--- | ---: | --- | --- | --- | --- |
| VLE | 25 | 100 | 0.4 | 3 | true |
```""")
    assert rows[0]["T_K"] == 298.15
    assert rows[0]["P_bar"] == 1
    assert rows[0]["pin"] is True
    datasets = {
        "datasets": [
            {
                "kind": "HE",
                "weight": 2,
                "source": "paper",
                "rows": [{"T_K": 300, "x1": 0.4, "HE_kJ_mol": 0.2}],
            }
        ]
    }
    assert parse_observations(datasets)[0]["HE_J_mol"] == 200
    assert parse_observations(datasets)[0]["weight"] == 2
    assert (
        parse_observations("kind,T_K,gamma1_inf\nGAMMA_INF,300,2")[0]["gamma1_inf"] == 2
    )


@pytest.mark.parametrize(
    "rows",
    [
        [{"kind": "VLE", "T_K": 300, "T_C": 25, "P_bar": 1, "x1": 0.5}],
        [{"kind": "LLE", "T_K": 300, "x1_alpha": 0.2, "x1_beta": 0.2}],
        [{"kind": "HE", "T_K": 300, "x1": 0.5, "HE_J_mol": 20, "weigth": 3}],
        [{"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 0}],
        [{"kind": "AZEOTROPE", "T_K": 300, "x1": 0.4, "y1": 0.6, "P_bar": 1}],
    ],
)
def test_bad_data_are_rejected(rows):
    with pytest.raises(ValueError):
        parse_observations(rows)


@pytest.mark.parametrize("model", ["NRTL", "UNIQUAC"])
def test_joint_fit_reproduces_runtime_and_export(model):
    settings = {"model": model}
    if model == "UNIQUAC":
        settings["rq"] = [{"r": 2.2, "q": 1.8}, {"r": 0.92, "q": 1.4}]
    data, reference, truth = synthetic(settings)
    result = fit_interactions(data)
    assert result["success"], result["warnings"]
    assert result["optimizer"]["objective"] < 1e-8
    sim = Simulator.from_string(result["pfd_text"])
    sim.initialize()
    thermo = sim.thermo_packages["global"]
    reference.install(truth)
    for T in (310, 340):
        x = reference.composition(0.4)
        assert thermo.activity_coefficients(T, x) == pytest.approx(
            reference.thermo.activity_coefficients(T, x), rel=1e-6
        )
        assert thermo.excess_enthalpy(x, T) == pytest.approx(
            reference.thermo.excess_enthalpy(x, T), abs=1e-4
        )
    if model == "UNIQUAC":
        assert result["rq"] == settings["rq"]
        assert thermo.r[reference.components[0]] == 2.2


def test_missing_vapor_compositions_and_fitted_alpha():
    data, _, _ = synthetic({"fit_alpha": True})
    for row in data["observations"]:
        if row["kind"] == "VLE":
            row.pop("y1")
    result = fit_interactions(data)
    assert result["optimizer"]["objective"] < 1e-6
    assert result["coefficients"]["alpha12"] == pytest.approx(0.42, abs=1e-4)


def test_weights_pins_and_infeasible_pins():
    data, _, _ = synthetic(kinds=("GAMMA_INF",))
    data["observations"][0]["pin"] = True
    data["weights"] = {"GAMMA_INF": 3}
    data["observations"][1]["weight"] = 7
    result = fit_interactions(data)
    assert result["success"]
    assert result["points"][0]["pin_satisfied"]
    total = sum(
        point["weighted_sum_squares"]
        for point in result["points"]
        if not point["observed"]["pin"]
    )
    assert total == pytest.approx(
        result["objectives"]["GAMMA_INF"]["weighted_sum_squares"]
    )
    bad = deepcopy(data)
    conflict = deepcopy(bad["observations"][0])
    conflict["gamma1_inf"] *= 2
    bad["observations"].append(conflict)
    with pytest.raises(ValueError, match="No feasible fit"):
        fit_interactions(bad)


def test_cv_is_independent_and_pins_are_training_only():
    data, _, _ = synthetic(kinds=("GAMMA_INF", "HE"))
    data["observations"][0]["pin"] = True
    data["cv"] = {"method": "kfold", "folds": 3}
    result = fit_interactions(data)
    folds = result["cross_validation"]["folds"]
    assert len(folds) == 3
    for fold in folds:
        assert fold["success"], fold
        assert "1" in fold["training_ids"]
        assert "1" not in fold["held_out_ids"]
        assert not set(fold["held_out_ids"]) & set(fold["training_ids"])


def test_lle_uses_actual_stable_split():
    data = request(form="constant", starts=2)
    problem = prepare_fit(data)
    truth = np.array([2.5, 2.5])
    problem.install(truth)
    split, xa, xb, _ = problem.thermo.liquid_liquid_equilibrium(
        problem.composition(0.5), 300, tol=1e-8
    )
    assert split
    a, b = sorted((xa[problem.components[0]], xb[problem.components[0]]))
    data["observations"] = [{"kind": "LLE", "T_K": 300, "x1_alpha": a, "x1_beta": b}]
    result = fit_interactions(data)
    assert result["success"], result["warnings"]
    assert result["optimizer"]["objective"] < 1e-7
    assert result["points"][0]["predicted"]["x1_alpha"] == pytest.approx(a, abs=1e-5)
    assert result["points"][0]["predicted"]["x1_beta"] == pytest.approx(b, abs=1e-5)


def test_azeotrope_uses_equal_vapor_and_liquid_composition():
    data, problem, truth = synthetic(kinds=("GAMMA_INF",))
    T = 350
    problem.install(truth)

    def difference(x):
        gamma = problem.thermo.activity_coefficients(T, problem.composition(x))
        return math.log(
            gamma[problem.components[0]]
            * problem.thermo.Psat(problem.components[0], T)
            / (
                gamma[problem.components[1]]
                * problem.thermo.Psat(problem.components[1], T)
            )
        )

    x = brentq(difference, 1e-5, 1 - 1e-5)
    gamma = problem.thermo.activity_coefficients(T, problem.composition(x))
    P = gamma[problem.components[0]] * problem.thermo.Psat(problem.components[0], T)
    data["observations"].append(
        {"kind": "AZEOTROPE", "T_K": T, "P_bar": P, "x1": x, "weight": 10}
    )
    result = fit_interactions(data)
    assert result["optimizer"]["objective"] < 1e-8
    assert result["points"][-1]["predicted"]["y1"] == x


@pytest.mark.parametrize("kind", ["UCST", "LCST"])
def test_critical_points_require_correct_temperature_direction(kind):
    data = request(form="inverse")
    problem = prepare_fit(data)
    T = 330

    def curvature(tau):
        problem.install(np.array([tau * T / 298.15, tau * T / 298.15]))
        return problem.critical_derivatives(T, 0.5)[0]

    critical_tau = brentq(curvature, 0, 3)
    data = request(form="constant_inverse")
    problem = prepare_fit(data)
    constant = 0 if kind == "UCST" else 4
    inverse = (critical_tau - constant) * T / 298.15
    values = np.array([constant, inverse, constant, inverse])
    rows = []
    for temp in (310, 350):
        problem.install(values)
        rows.append(
            {
                "kind": "GAMMA_INF",
                "T_K": temp,
                "gamma1_inf": problem.thermo.activity_coefficients(
                    temp, problem.composition(0)
                )[problem.components[0]],
                "gamma2_inf": problem.thermo.activity_coefficients(
                    temp, problem.composition(1)
                )[problem.components[1]],
            }
        )
    rows.append({"kind": kind, "T_K": T, "x1": 0.5, "weight": 5})
    data["observations"] = rows
    data["initial"] = dict(zip(problem.names, values * np.array(problem.scales)))
    result = fit_interactions(data)
    assert result["success"], result["warnings"]
    critical = result["points"][-1]["predicted"]
    assert abs(critical["curvature"]) < 1e-4
    assert abs(critical["third_derivative"]) < 1e-4
    assert critical["fourth_derivative"] > 0


def test_apply_existing_pfd_preserves_other_definitions():
    data, _, _ = synthetic()
    result = fit_interactions(data)
    original = parse_pfd("""PROCESS: Existing
THERMO_METHOD: IDEAL
ONLINE_LOOKUP: false
COMPONENTS:
    A | ethanol
    B | water
    C | methanol
INTERACTION_PARAMETERS:
    A/B | model=NRTL, a12=10, a21=20
    A/C | model=NRTL, a12=30, a21=40
UNIT M
    TYPE: Mixer
    PORTS:
        in1 : inlet
        out : outlet
""")
    exported = export_fit(
        result, pfd_text=original.to_pfd(), component_map={"Fit_1": "A", "Fit_2": "B"}
    )
    updated = parse_pfd(exported["pfd_text"])
    assert updated.units == original.units
    assert updated.components[2] == original.components[2]
    assert len(updated.interaction_parameters) == 2
    assert (
        next(i for i in updated.interaction_parameters if i.component2 == "C")
        == original.interaction_parameters[1]
    )
    with pytest.raises(ValueError, match="does not identify"):
        export_fit(
            result,
            pfd_text=original.to_pfd(),
            component_map={"Fit_1": "C", "Fit_2": "B"},
        )


def test_cli_writes_entry_and_pfd_without_replacing_inputs(tmp_path):
    data, _, _ = synthetic(kinds=("GAMMA_INF",))
    source = tmp_path / "data.json"
    source.write_text(json.dumps(data))
    report, entry, pfd = [
        tmp_path / name for name in ("result.json", "entry.pfd", "mixture.pfd")
    ]
    assert (
        cli_main(
            [
                "fit",
                str(source),
                "-o",
                str(report),
                "--entry",
                str(entry),
                "--pfd-output",
                str(pfd),
            ]
        )
        == 0
    )
    assert json.loads(report.read_text())["success"]
    assert "INTERACTION_PARAMETERS:" in entry.read_text()
    assert parse_pfd(pfd.read_text()).metadata.thermo_method == "NRTL"
    with pytest.raises(SystemExit):
        cli_main(["fit", str(source), "-o", str(source)])


def test_zero_objective_weight_does_not_disable_pin():
    data = request(weights={"GAMMA_INF": 0})
    data["observations"][0]["pin"] = True
    normalized = normalize_fit_request(data)
    assert len(normalized["observations"]) == 1


def test_joint_vapor_parameter_fit_round_trips():
    data = request(
        vapor="PR",
        vapor_parameters=[
            {
                "model": "PR",
                "field": "kij",
                "value": 0,
                "fit": True,
                "lower": -0.5,
                "upper": 0.5,
            }
        ],
    )
    data["observations"] = [{"kind": "VLE", "T_K": 320, "P_bar": 1, "x1": 0.5}]
    problem = prepare_fit(data)
    truth = np.array([0.4, 1.5, -0.2, 0.8, 0.08])
    rows = []
    for T in (320, 350):
        problem.install(truth)
        rows.append(
            {
                "kind": "GAMMA_INF",
                "T_K": T,
                "gamma1_inf": problem.thermo.activity_coefficients(
                    T, problem.composition(0)
                )[problem.components[0]],
                "gamma2_inf": problem.thermo.activity_coefficients(
                    T, problem.composition(1)
                )[problem.components[1]],
            }
        )
        for x in (0.2, 0.6):
            row = {"kind": "VLE", "T_K": T, "x1": x, "P_bar": 1}

            def closure(P):
                row["P_bar"] = P
                return problem._vle(row)[0][0]

            P = brentq(closure, 0.001, 3)
            row["P_bar"] = P
            row["y1"] = problem._vle(row)[1]["evaluation_y1"]
            rows.append(row.copy())
    data["observations"] = rows
    data["cv"] = {"method": "kfold", "folds": 2}
    result = fit_interactions(data)
    assert result["success"], result["warnings"]
    assert result["optimizer"]["objective"] < 1e-8
    assert result["method"] == "NRTL-PR"
    assert all("error" not in fold for fold in result["cross_validation"]["folds"])
    assert result["vapor_parameters"][0]["kij"] == pytest.approx(0.08, abs=1e-3)
    restored = Simulator.from_string(result["pfd_text"])
    restored.initialize()
    thermo = restored.thermo_packages["global"]
    problem.install(truth)
    composition = problem.composition(0.4)
    assert thermo.vapor_fugacity_coefficients(340, 1, composition) == pytest.approx(
        problem.thermo.vapor_fugacity_coefficients(340, 1, composition), rel=1e-4
    )


def test_scope_inheritance_is_shared_and_export_is_isolated():
    text = """ONLINE_LOOKUP: false
COMPONENTS:
    A | ethanol
    B | water
THERMO_SCOPES:
    Parent | method=NRTL-PR, inherit=global
    Child | method=NRTL-PR, inherit=Parent
    Isolated | method=NRTL-PR
INTERACTION_PARAMETERS:
    A/B | model=PR, kij=0.1
    A/B | model=PR, scope=Parent, kij=0.2
"""
    pfd = parse_pfd(text)
    assert pfd.thermo_scope_lineage("Child") == ["global", "Parent", "Child"]
    settings = request(components=["A", "B"], pfd_text=text, scope="Child", vapor="PR")
    inherited = prepare_fit(settings)
    assert inherited.definition.interaction_parameters[0].parameters["kij"] == 0.2
    isolated = prepare_fit({**settings, "scope": "Isolated"})
    assert len(isolated.definition.interaction_parameters) == 1  # activity only
    data, _, _ = synthetic(kinds=("GAMMA_INF",))
    fitted = fit_interactions(data)
    updated = parse_pfd(
        export_fit(
            fitted,
            pfd_text=text,
            scope="Child",
            component_map={"Fit_1": "A", "Fit_2": "B"},
        )["pfd_text"]
    )
    assert updated.metadata == pfd.metadata
    assert updated.get_thermo_scope("Child").method == "NRTL"
    assert updated.get_thermo_scope("Parent") == pfd.get_thermo_scope("Parent")
    assert updated.interaction_parameters[-1].scope == "Child"


def test_submit_saved_report_does_not_refit(tmp_path, monkeypatch):
    import fit_cli

    data, _, _ = synthetic(kinds=("GAMMA_INF",))
    result = fit_interactions(data)
    report = tmp_path / "report.json"
    report.write_text(json.dumps(result))

    def submit(server, uploaded, source):
        assert server == "http://localhost:5000"
        assert uploaded["parameters"] == result["parameters"]
        assert source["citation"] == "Archived source"
        return {"submission": {"id": "review-id"}}

    monkeypatch.setattr(fit_cli, "_submit", submit)
    monkeypatch.setattr(
        fit_cli,
        "fit_interactions",
        lambda *args, **kwargs: pytest.fail("Submission must never rerun fitting"),
    )
    assert (
        cli_main(
            [
                "fit",
                "submit",
                str(report),
                "--server",
                "http://localhost:5000",
                "--source",
                "Archived source",
            ]
        )
        == 0
    )


@pytest.mark.parametrize("model", ["NRTL", "UNIQUAC"])
def test_clipped_runtime_regime_cannot_be_used_for_regression(model):
    problem = prepare_fit(request(model=model, form="constant"))
    problem.install(np.array([200.0, 200.0]))
    with pytest.raises(ValueError, match="limit reached"):
        problem.checked_gamma(300, 0)


def test_export_rejects_a_different_vle_property_basis():
    data, _, _ = synthetic()
    result = fit_interactions(data)
    destination = parse_pfd(result["definition_pfd"])
    destination.components[0].antoine_A = 5
    destination.components[0].antoine_B = 1500
    destination.components[0].antoine_C = 200
    destination.components[0].antoine_Tmin = 250
    destination.components[0].antoine_Tmax = 400
    with pytest.raises(ValueError, match="physical-property basis differs"):
        export_fit(result, pfd_text=destination.to_pfd())


def test_multiline_publication_citation_cannot_invalidate_or_change_a_fit():
    data, _, _ = synthetic(kinds=("GAMMA_INF",))
    baseline = fit_interactions(data)
    citation = 'Authors, "Excess enthalpies", table 2\nThermochimica Acta, 2026; C:\\references\\paper'
    cited = fit_interactions({**data, "source": citation})
    assert cited["success"]
    assert cited["coefficients"] == pytest.approx(baseline["coefficients"], abs=1e-9)
    assert cited["optimizer"]["objective"] == pytest.approx(
        baseline["optimizer"]["objective"], abs=1e-12
    )
    assert "comment" not in cited["parameters"]
    assert (
        cited["request"]["source"] == citation
    )  # Legacy/CLI provenance remains available.
    assert cited["pfd_text"] == baseline["pfd_text"]
    assert parse_pfd(cited["pfd_text"]).interaction_parameters


def test_smart_pfd_metadata_is_single_line_and_quote_safe():
    from pfd_parser import Component, InteractionParameter, ProcessFlowDiagram

    comment = 'Authors, "A paper"\r\nThermochimica Acta\nDOI: example | path C:\\notes\\citation'
    pfd = ProcessFlowDiagram(
        components=[Component("A", "acetone"), Component("B", "water")],
        interaction_parameters=[
            InteractionParameter(
                "A",
                "B",
                "NRTL",
                parameters={
                    "tau12_c": 1.0,
                    "tau21_c": 2.0,
                    "alpha12": 0.3,
                    "comment": comment,
                },
            )
        ],
    )
    text = pfd.to_pfd()
    assert len([line for line in text.splitlines() if "Thermochimica" in line]) == 1
    assert not any(line.startswith("Thermochimica") for line in text.splitlines())
    parsed = parse_pfd(text)
    assert parsed.interaction_parameters[0].parameters["comment"] == " ".join(
        comment.splitlines()
    )
    assert pfd.interaction_parameters[0].parameters["comment"] == comment
    pfd.interaction_parameters[0].parameters["comment"] = "can't"
    assert (
        parse_pfd(pfd.to_pfd()).interaction_parameters[0].parameters["comment"]
        == "can't"
    )
