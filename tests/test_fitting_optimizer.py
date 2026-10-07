"""Conditioned coordinates, runtime batching, sparse derivatives and ABCD laws."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from simulator import Simulator
from thermodynamics_models.interaction_fitting import prepare_fit, fit_interactions
from thermodynamics_models.fitting_optimizer import (
    _EvaluationLimit,
    temperature_transform,
    ConditionedObjective,
    solve_start,
)


def settings(**extra):
    return {
        "components": ["ethanol", "water"],
        "form": "constant_inverse_anchored_linear",
        "starts": 1,
        "observations": [
            {"kind": "GAMMA_INF", "T_K": T, "gamma1_inf": 1, "gamma2_inf": 1}
            for T in (280, 300, 330, 360, 395)
        ],
        **extra,
    }


@pytest.mark.parametrize("form", ["constant_inverse_anchored_linear", "full"])
def test_temperature_coordinates_condition_the_same_law_without_rotating_bounds(form):
    problem = prepare_fit(settings(form=form))
    rows = problem.request["observations"]
    transform = temperature_transform(problem, rows)
    temperatures = np.unique(
        np.r_[[row["T_K"] for row in rows], np.linspace(280, 395, 5)]
    )
    u = temperatures / problem.request["T_ref_K"]
    basis = np.column_stack(
        [np.ones_like(u), 1 / u, 1 / u - 1 + np.log(u), u]
        + ([u * u] if form == "full" else [])
    )
    count = len(problem.terms)
    conditioned = basis @ transform[:count, :count]
    assert np.linalg.cond(conditioned) <= max(1, np.linalg.cond(basis) / 1000) * (
        1 + 1e-7
    )
    physical = np.linspace(-0.2, 0.3, len(problem.names))
    coordinates = np.linalg.solve(transform, physical)
    np.testing.assert_allclose(transform @ coordinates, physical, atol=1e-10)
    np.testing.assert_allclose(
        conditioned @ coordinates[:count], basis @ physical[:count], atol=1e-10
    )


@pytest.mark.parametrize("model", ["NRTL", "UNIQUAC"])
@pytest.mark.parametrize("compiled", [True, False])
def test_batched_activities_and_lle_objective_match_the_runtime(
    model, compiled, monkeypatch
):
    problem = prepare_fit(settings(model=model))
    values = np.array([0.4, 1.2, -0.1, 0.03, -0.2, 0.7, 0.2, -0.04])
    problem.install(values)
    if not compiled:
        monkeypatch.setattr(
            problem.thermo, "_compiled_activity_backend", lambda T=None: None
        )
    x = np.array([0, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0])
    temperatures = np.linspace(280, 395, len(x))
    actual = problem.checked_gamma_many(temperatures, x)
    expected = np.array(
        [
            [
                problem.thermo.activity_coefficients(T, problem.composition(z))[c]
                for c in problem.components
            ]
            for T, z in zip(temperatures, x)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    T, a, b, sigma = 320, 0.1, 0.85, 0.01
    grid = np.r_[np.linspace(0.00001, 0.99999, 41), a, b]
    mu = np.array(
        [
            np.log(
                np.maximum(
                    [
                        z * gamma[problem.components[0]],
                        (1 - z) * gamma[problem.components[1]],
                    ],
                    1e-300,
                )
            )
            for z in grid
            for gamma in [
                problem.thermo.activity_coefficients(T, problem.composition(z))
            ]
        ]
    )
    average = (mu[-2] + mu[-1]) / 2
    gaps = (
        grid * mu[:, 0]
        + (1 - grid) * mu[:, 1]
        - (grid * average[0] + (1 - grid) * average[1])
    )
    expected = np.r_[
        (mu[-2] - mu[-1]) / sigma, np.minimum(gaps, 0) / sigma / np.sqrt(len(grid))
    ]
    np.testing.assert_allclose(
        problem._lle_errors(T, a, b, sigma)[0], expected, atol=1e-12
    )
    json.dumps(problem.critical_derivatives(T, 0.3))


def test_local_endpoint_derivatives_do_not_recalculate_other_observations():
    problem = prepare_fit(
        settings(
            form="constant",
            observations=[
                {"kind": "LLE", "T_K": 300, "x1_alpha": 0.1},
                {"kind": "LLE", "T_K": 300, "x1_beta": 0.9},
                {"kind": "GAMMA_INF", "T_K": 330, "gamma1_inf": 2},
            ],
        )
    )
    rows = problem.request["observations"]
    objective = ConditionedObjective(problem, rows, np.eye(len(problem.names)))
    values = problem.initial.copy()
    values[:2] = [2, 2]
    coordinates = objective.encode(values)
    objective.evaluate(coordinates)
    jacobian = objective.jac(coordinates)
    # 3 base rows + two global columns * 2 probes * 3 rows + 2 local columns * 2 probes.
    assert objective.row_calls == 19
    for index, row in enumerate(rows):
        for coordinate, identifier in objective.local.items():
            if identifier != row["id"]:
                assert np.all(jacobian[objective.slices[index], coordinate] == 0)


def test_local_composition_changes_reuse_the_physical_activity_backend():
    problem = prepare_fit(
        settings(
            form="constant", observations=[{"kind": "LLE", "T_K": 300, "x1_alpha": 0.1}]
        )
    )
    values = problem.initial.copy()
    problem.install(values)
    backend = problem.thermo._compiled_activity_backend()
    values[problem.lle_names["1"]] = 0.8
    problem.install(values)
    assert problem.thermo._compiled_activity_backend() is backend


def test_fraction_coordinates_preserve_physical_values_and_bounds_near_endpoints():
    problem = prepare_fit(settings(form="constant", observations=[
        {"kind":"LLE", "T_K":300, "x1_alpha":.1},
        {"kind":"LLE", "T_K":300, "x1_beta":.9},
    ]))
    objective = ConditionedObjective(problem,problem.request["observations"],np.eye(len(problem.names)))
    values = np.array([2.,2.,.00002,.99998])
    coordinates = objective.encode(values)
    np.testing.assert_allclose(objective.decode(coordinates),values,atol=1e-15)
    assert np.all(coordinates >= objective.coordinate_bounds(problem.lower))
    assert np.all(coordinates <= objective.coordinate_bounds(problem.upper))
    derivative = objective.decoding_jacobian(values)
    for index in objective.local:
        shifted = coordinates.copy()
        shifted[index] += 1e-5
        numerical = (objective.decode(shifted)-objective.decode(coordinates))/1e-5
        np.testing.assert_allclose(numerical,derivative[:,index],atol=1e-9)


def test_residual_evaluation_budget_excludes_finite_difference_probes():
    problem = prepare_fit(
        settings(
            form="constant",
            max_nfev=1,
            observations=[
                {"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 2, "gamma2_inf": 3},
            ],
        )
    )
    fitted = solve_start(problem, problem.request["observations"], problem.initial)
    assert not fitted["success"]
    assert fitted["nfev"] == 1
    assert fitted["row_evaluations"] > fitted["nfev"]
    np.testing.assert_array_equal(fitted["values"], problem.initial)


def test_derivative_probe_at_a_model_domain_boundary_uses_the_valid_side():
    row = {"id": "boundary", "kind": "HE", "T_K": 300, "pin": False, "weight": 1}

    def errors(values, row):
        if values[0] < 0:
            raise ValueError("Outside model domain")
        return np.array([values.sum() - 1]), [], {}

    problem = SimpleNamespace(
        row_errors=errors,
        request={"weights": {"HE": 1}},
        numerical_errors=(ValueError,),
        lower=np.array([0.0, -1.0]),
        upper=np.ones(2),
        lle_names={},
        critical_names={},
        vlle_names={},
        install=lambda values: None,
    )
    objective = ConditionedObjective(problem, [row], np.eye(2))
    np.testing.assert_allclose(
        objective.jac(np.array([0.0, 0.1])), [[1.0, 1.0]], atol=1e-6
    )


def test_invalid_trial_attempts_consume_the_residual_budget():
    row = {"id": "trial", "kind": "HE", "pin": False, "weight": 1}

    def errors(values, row):
        if values[0] < 0:
            raise ValueError("Outside model domain")
        return values, [], {}

    problem = SimpleNamespace(
        row_errors=errors, request={"weights": {"HE": 1}},
        lle_names={}, critical_names={}, vlle_names={},
    )
    objective = ConditionedObjective(problem, [row], np.eye(1))
    objective.evaluate(np.array([0.1]))
    objective.limit = 2
    with pytest.raises(ValueError, match="Outside model domain"):
        objective.evaluate(np.array([-0.1]))
    assert objective.calls == 2
    with pytest.raises(_EvaluationLimit):
        objective.evaluate(np.array([0.2]))


@pytest.mark.parametrize("pinned", [False, True])
def test_solver_excludes_validation_observations_even_when_all_rows_are_supplied(monkeypatch, pinned):
    problem = prepare_fit(settings(form="constant", observations=[
        {"kind": "GAMMA_INF", "T_K": 300, "gamma1_inf": 1, "gamma2_inf": 1, "pin": pinned},
        {"kind": "LLE", "T_K": 400, "x1_alpha": .1,
         "validation_only": True},
    ]))
    original = problem.row_errors

    def training_only(values, row):
        assert not row["validation_only"], "Validation observations entered optimization"
        return original(values, row)

    monkeypatch.setattr(problem, "row_errors", training_only)
    fitted = problem.solve(problem.request["observations"])
    assert fitted["success"]
    assert fitted["objective"] < 1e-12
    np.testing.assert_allclose(fitted["values"], problem.initial, atol=1e-12)
    with pytest.raises(ValueError, match="training observation"):
        problem.solve([problem.request["observations"][1]])


@pytest.mark.parametrize("compiled", [True, False])
def test_empty_activity_batch_keeps_its_component_axis(compiled, monkeypatch):
    problem = prepare_fit(settings())
    problem.install(problem.initial)
    if not compiled:
        monkeypatch.setattr(problem.thermo, "_compiled_activity_backend", lambda T=None: None)
    assert problem.checked_gamma_many(300, np.array([])).shape == (0, 2)


@pytest.mark.parametrize("model", ["NRTL", "UNIQUAC"])
def test_abcd_fits_and_exports_zero_quadratic_coefficients(model):
    request = settings(model=model)
    reference = prepare_fit(request)
    truth = np.array([0.4, 1.2, -0.1, 0.03, -0.2, 0.7, 0.2, -0.04])
    reference.install(truth)
    for row in request["observations"]:
        row["gamma1_inf"] = reference.checked_gamma(row["T_K"], 0)[
            reference.components[0]
        ]
        row["gamma2_inf"] = reference.checked_gamma(row["T_K"], 1)[
            reference.components[1]
        ]
    request["initial"] = dict(zip(reference.names, truth * reference.scales))
    result = fit_interactions(request)
    assert result["optimizer"]["objective"] < 1e-8
    quadratic = "g" if model == "NRTL" else "e"
    assert result["parameters"][f"tau12_{quadratic}"] == 0
    assert result["parameters"][f"tau21_{quadratic}"] == 0
    restored = Simulator.from_string(result["pfd_text"]).initialize().thermo
    for T in (290, 345, 390):
        assert restored.activity_coefficients(
            T, reference.composition(0.4)
        ) == pytest.approx(reference.checked_gamma(T, 0.4), rel=1e-8)


def test_conditioned_solver_enforces_original_coefficient_box_at_active_bounds():
    terms = ("constant", "inverse", "anchored", "linear", "quadratic")
    T = np.linspace(280, 395, 10)
    u = T / 298.15
    basis = np.column_stack([np.ones_like(u), 1 / u, 1 / u - 1 + np.log(u), u, u * u])
    rows = [
        {"id": "curve", "T_K": float(t), "kind": "HE", "pin": False, "weight": 1}
        for t in T
    ]
    problem = SimpleNamespace(
        terms=terms,
        names=[f"{direction}.{term}" for direction in ("12", "21") for term in terms],
        lower=np.full(10, -1.0),
        upper=np.full(10, 1.0),
        lle_names={},
        critical_names={},
        vlle_names={},
        request={"T_ref_K": 298.15, "max_nfev": 100, "weights": {"HE": 1}},
        install=lambda values: None,
        numerical_errors=(ValueError, OverflowError, FloatingPointError),
    )

    def errors(values, row):
        index = next(i for i, item in enumerate(rows) if item is row)
        return (
            np.array(
                [basis[index] @ (values[:5] - 10), basis[index] @ (values[5:] - 10)]
            ),
            [],
            {},
        )

    problem.row_errors = errors
    result = solve_start(problem, rows, np.zeros(10))
    assert np.all(result["values"] >= -1)
    assert np.all(result["values"] <= 1)
    np.testing.assert_allclose(result["values"], 1, atol=1e-4)
