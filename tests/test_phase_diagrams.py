"""Diagram correctness: gaps, phase balance, fugacity and model coverage."""

import numpy as np
import pytest

from thermodynamics import create_thermodynamics
from thermodynamics_models.phase_diagrams import PhaseDiagramMixin
from thermodynamics_models.common import ThermodynamicsError


@pytest.mark.parametrize("method", ["IDEAL", "PR", "SRK", "NRTL", "UNIFAC", "UNIQUAC"])
def test_binary_charts_use_the_selected_model(method):
    thermo = create_thermodynamics(["ethanol", "water"], method)
    chart = thermo.generate_Pxy_data("ethanol", "water", 323.15, n_points=4)
    assert len(chart["x"]) == 5
    assert not chart["errors"]
    for x, y, pressure in zip(chart["x"], chart["y"], chart["P_bubble"]):
        composition = {"ethanol": x, "water": 1 - x}
        K = thermo.K_values(323.15, pressure, composition)
        assert abs(x * K["ethanol"] + (1 - x) * K["water"] - 1) < 2e-4
        assert abs(y - x * K["ethanol"]) < 2e-4


def test_failed_dew_points_are_gaps_instead_of_invented_equilibria():
    thermo = create_thermodynamics(["ethanol", "water"], "IDEAL")
    thermo.dew_point_T = lambda *args: (_ for _ in ()).throw(
        ValueError("deliberate failure")
    )
    chart = thermo.generate_Txy_data("ethanol", "water", 1, n_points=4)
    assert all(value is None for value in chart["T_dew"])
    assert all(value is not None for value in chart["T_bubble"])
    assert len(chart["errors"]) == 5


def test_unconverged_bubble_temperatures_are_rejected():
    thermo = create_thermodynamics(["ethanol", "water"], "IDEAL")
    # No root exists here. An exhausted temperature solve must raise instead
    # of returning its last iterate as a calculated equilibrium point.
    with pytest.raises(ValueError, match="No bubble points converged"):
        thermo.generate_Txy_data("ethanol", "water", 10000, n_points=4)


@pytest.mark.parametrize("method", ["PR", "SRK"])
def test_supercritical_eos_extrapolations_are_not_phase_boundaries(method):
    thermo = create_thermodynamics(["ethanol", "water"], method)
    with pytest.raises(ValueError, match="No bubble points converged"):
        thermo.generate_Pxy_data("ethanol", "water", 700, n_points=4)


def test_diagram_guard_rejects_a_solver_returning_an_unconverged_iterate():
    thermo = create_thermodynamics(["ethanol", "water"], "IDEAL")
    thermo.bubble_point_T = lambda *args: 300.0
    with pytest.raises(ValueError, match="Bubble point did not converge"):
        thermo.generate_Txy_data("ethanol", "water", 1, n_points=4)


def test_ideal_temperature_boundaries_satisfy_equilibrium():
    thermo = create_thermodynamics(["ethanol", "water"], "IDEAL")
    composition = {"ethanol": 0.5, "water": 0.5}
    bubble = thermo.bubble_point_T(composition, 1)
    K = thermo.K_values(bubble, 1, composition)
    assert abs(sum(composition[c] * K[c] for c in composition) - 1) < 1e-7
    vapor = {c: composition[c] * K[c] for c in composition}
    dew = thermo.dew_point_T(vapor, 1)
    assert abs(dew - bubble) < 1e-4


def test_dew_temperature_rejects_a_scan_without_a_root():
    from thermodynamics_models.common import ThermodynamicsError

    thermo = create_thermodynamics(["ethanol", "water"], "IDEAL")
    thermo.K_values = lambda *args: {"ethanol": 0.1, "water": 0.1}
    with pytest.raises(
        ThermodynamicsError, match="Could not bracket dew point temperature"
    ):
        thermo.dew_point_T({"ethanol": 0.5, "water": 0.5}, 1)


@pytest.mark.parametrize("boundary", ["pressure", "temperature"])
def test_dew_boundaries_reject_unconverged_liquid_compositions(boundary):
    thermo = create_thermodynamics(["ethanol", "water"], "IDEAL")
    vapor = {"ethanol": 0.2, "water": 0.8}

    def oscillating_K(T, P, x):
        # sum(y/K) has an exact root, but the normalized incipient liquid
        # alternates between (0.2, 0.8) and (0.8, 0.2). At that scalar root
        # the individual component fugacity ratios are 4 and 1/4.
        factor = 350 / T / P
        return {
            "ethanol": factor * vapor["ethanol"] / (1 - x["ethanol"]),
            "water": factor * vapor["water"] / x["ethanol"],
        }

    thermo.K_values = oscillating_K
    with pytest.raises((ValueError, ThermodynamicsError), match="dew|Dew"):
        if boundary == "pressure":
            thermo.dew_point_P(vapor, 350)
        else:
            thermo.dew_point_T(vapor, 1)


def test_xy_does_not_compute_unused_dew_points():
    thermo = create_thermodynamics(["ethanol", "water"], "IDEAL")
    thermo.dew_point_P = lambda *args: pytest.fail("An xy curve needs no dew solve")
    chart = thermo.generate_xy_data("ethanol", "water", T=323.15, n_points=4)
    assert len(chart["y"]) == 5
    assert not chart["errors"]


def test_activity_and_eos_share_one_binary_chart_implementation():
    for method in ["IDEAL", "PR", "NRTL", "UNIFAC"]:
        thermo = create_thermodynamics(["ethanol", "water"], method)
        assert thermo.generate_Txy_data.__func__ is PhaseDiagramMixin.generate_Txy_data


def test_ternary_lle_tie_lines_meet_phase_equilibrium_and_balance():
    components = ["water", "methanol", "benzene"]
    thermo = create_thermodynamics(components, "NRTL")
    diagram = thermo.generate_ternary_lle_data(components, 298.15, n_points=6)
    assert len(diagram["samples"]) == 28
    splits = [point for point in diagram["samples"] if point["status"] == "lle"]
    assert splits
    for point in splits:
        beta = point["liquid2_fraction"]
        g1, g2 = (
            thermo.activity_coefficients(298.15, point[key]) for key in ("x1", "x2")
        )
        for component in components:
            assert (
                abs(
                    (1 - beta) * point["x1"][component]
                    + beta * point["x2"][component]
                    - point["z"][component]
                )
                < 1e-6
            )
            if point["z"][component] > 1e-12:
                np.testing.assert_allclose(
                    point["x1"][component] * g1[component],
                    point["x2"][component] * g2[component],
                    rtol=1e-4,
                    atol=1e-12,
                )


def test_ternary_vlle_map_includes_real_three_phase_equilibrium_and_restores_mode():
    components = ["water", "methanol", "benzene"]
    thermo = create_thermodynamics(components, "NRTL")
    thermo.set_fluid_phase_model("VLE")
    diagram = thermo.generate_vlle_data(components, 333.0, 1.01325, n_points=10)
    point = next(
        p
        for p in diagram["samples"]
        if p["z"] == {"water": 0.2, "methanol": 0.3, "benzene": 0.5}
    )
    assert point["status"] != "failed", point
    assert all(
        point[key] > 1e-6
        for key in ("vapor_fraction", "liquid1_fraction", "liquid2_fraction")
    )
    assert point["material_balance_residual"] < 1e-6
    assert point["equilibrium_log_residual"] < 1e-4
    assert thermo.fluid_phase_model == "VLE"


def test_binary_vlle_map_has_conserved_phase_fractions():
    thermo = create_thermodynamics(["butanol", "water"], "NRTL")
    diagram = thermo.generate_vlle_data(["butanol", "water"], 298.15, 1, n_points=4)
    valid = [p for p in diagram["samples"] if p["status"] != "failed"]
    assert valid
    for point in valid:
        assert (
            abs(
                sum(
                    point[k]
                    for k in ("vapor_fraction", "liquid1_fraction", "liquid2_fraction")
                )
                - 1
            )
            < 1e-7
        )


@pytest.mark.parametrize("method", ["NRTL", "UNIFAC", "UNIQUAC"])
def test_butanol_water_constant_pressure_vlle_envelope_and_binodal(method):
    thermo = create_thermodynamics(["butanol", "water"], method)
    chart = thermo.generate_binary_vlle_data("butanol", "water", 1, n_points=10)
    assert not chart["errors"], chart["errors"]
    azeotrope = chart["heteroazeotrope"]
    assert azeotrope is not None
    temperature = azeotrope["temperature_C"] + 273.15
    lean, rich = azeotrope["x1"]["butanol"], azeotrope["x2"]["butanol"]
    assert 0 < lean < azeotrope["y"]["butanol"] < rich < 0.5
    assert chart["x"] == sorted(chart["x"])
    assert sum(0 < x < lean for x in chart["x"]) >= 4
    assert sum(rich < x < 1 for x in chart["x"]) >= 4
    plateau = [
        (x, T) for x, T in zip(chart["x"], chart["T_bubble"]) if lean <= x <= rich
    ]
    assert len(plateau) >= 3
    for x, T in plateau:
        assert abs(T - azeotrope["temperature_C"]) < 1e-6
    for liquid in ("x1", "x2"):
        assert abs(thermo.bubble_point_P(azeotrope[liquid], temperature) - 1) < 1e-6
        assert (
            thermo._check_diagram_vle(azeotrope[liquid], azeotrope["y"], temperature, 1)
            < 1e-4
        )
    binodal = chart["binodal"]
    assert len(binodal["samples"]) == 11
    assert binodal["T"] == sorted(binodal["T"])
    assert abs(binodal["T"][-1] - azeotrope["temperature_C"]) < 1e-6
    assert abs(binodal["x1"][-1] - lean) < 1e-6
    assert abs(binodal["x2"][-1] - rich) < 1e-6
    for sample in binodal["samples"]:
        assert sample["status"] == "lle", sample
        beta = sample["liquid2_fraction"]
        for c in sample["z"]:
            assert (
                abs(
                    (1 - beta) * sample["x1"][c]
                    + beta * sample["x2"][c]
                    - sample["z"][c]
                )
                < 1e-6
            )
        assert sample["material_balance_residual"] < 1e-6
        assert sample["activity_log_residual"] < 1e-4


def test_miscible_binary_vlle_envelope_has_no_invented_binodal():
    thermo = create_thermodynamics(["ethanol", "water"], "NRTL")
    chart = thermo.generate_binary_vlle_data("ethanol", "water", 1, n_points=4)
    assert not chart["errors"], chart["errors"]
    assert chart["heteroazeotrope"] is None
    assert chart["binodal"]["T"] == []
    assert all(T is not None for T in chart["T_bubble"])


def test_non_lle_models_reject_requested_liquid_split_diagrams():
    thermo = create_thermodynamics(["water", "ethanol", "methanol"], "IDEAL")
    with pytest.raises(ValueError, match="LLE-capable"):
        thermo.generate_ternary_lle_data(
            ["water", "ethanol", "methanol"], 298.15, n_points=4
        )
