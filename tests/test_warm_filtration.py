from pathlib import Path

import pytest

from chemical_properties import ChemicalDatabase
from simulator import Simulator
from thermodynamics_models.base import IdealThermodynamics
from unit_operations import Filter
from unit_operations_base import UnitOperationError
from unit_operations_solids import Crystallizer


@pytest.fixture(scope="module")
def thermo():
    thermo = IdealThermodynamics(
        ["water", "ethanol"], ChemicalDatabase(enable_online=False)
    )
    thermo.configure_permanent_solids(
        ["water", "ethanol", "NaCl"],
        ["NaCl"],
        {
            "water": {"diameter_m": 1e-4, "sphericity": 0.9},
            "NaCl": {"diameter_m": 1e-4, "sphericity": 0.9},
        },
        conventional_solid_components=["water"],
    )
    # Illustrative subcooled-liquid transport fits; the test exercises the
    # actual resolver and temperature/composition-dependent viscosity path.
    for c in ("water", "ethanol"):
        thermo._resolver_known_props[c].setdefault("property_correlations", {})[
            "mul"
        ] = {
            "equation": "exp_poly_x",
            "coefficients": {"A": -6.90775527898, "B": -2},
            "Tmin_K": 220,
            "Tmax_K": 360,
            "_pfd_override": True,
        }
    return thermo


PARAMS = {
    "washing_model": "equilibrium",
    "cycle_time": 600,
    "P_drop": 2,
    "porosity": 0.4,
    "capture_cut_size": 0,
    "wash_cells": 2,
    "wash_steps": 8,
    "medium_resistance": 1e9,
    "equilibrium_nucleus_diameter": 1e-5,
}


def states(thermo):
    feed = thermo.calculate_state(
        275, 3, 100, {"water": 0.9, "ethanol": 0.1}, phase="liquid"
    )
    slurry = (
        Crystallizer("C", thermo, {"T": 250}).solve({"in": feed}).outlet_streams["out"]
    )
    wash = thermo.calculate_state(290, 3, 5, {"water": 1}, phase="liquid")
    return {"in": slurry, "wash": wash}


def balances(inlets, result):
    for c in set().union(*(s.composition for s in inlets.values())):
        assert sum(
            s.component_flows().get(c, 0) for s in result.outlet_streams.values()
        ) == pytest.approx(
            sum(s.component_flows().get(c, 0) for s in inlets.values()), abs=1e-8
        )
    incoming = sum(s.F * s.H for s in inlets.values())
    outgoing = sum(s.F * s.H for s in result.outlet_streams.values())
    assert outgoing == pytest.approx(incoming + result.heat_duty, rel=1e-7, abs=1e-4)


def test_real_warm_melt_wash_balances_and_hydraulic_change(thermo):
    inlets = states(thermo)
    result = Filter("F", thermo, PARAMS).solve(inlets)
    balances(inlets, result)
    p = result.performance
    assert p["cycle_feasible"]
    assert p["required_cycle_time_s"] == pytest.approx(600)
    assert p["maximum_equilibrium_residual"] < 1e-8
    assert any(t > 250.001 for t in p["cell_temperatures_K"])
    assert abs(p["net_solid_change_kmol_h"]["water"]) > 1e-4
    assert p["final_cake_resistance_per_m"] != pytest.approx(
        p["initial_cake_resistance_per_m"], rel=1e-4
    )
    cake = result.outlet_streams["cake"]
    assert cake.solid_particle_size_distributions[
        "water"
    ].total_molar_flow == pytest.approx(cake.solid_component_flows["water"])


def test_warm_wash_heats_inert_cake_without_melting_it(thermo):
    feed = thermo.calculate_state(298.15, 3, 101, {"water": 100 / 101, "NaCl": 1 / 101})
    wash = thermo.calculate_state(320, 3, 2, {"water": 1})
    inlets = {"in": feed, "wash": wash}
    result = Filter("F", thermo, PARAMS).solve(inlets)
    balances(inlets, result)
    assert result.outlet_streams["cake"].solid_component_flows["NaCl"] == pytest.approx(
        1
    )
    assert 298.15 < result.outlet_streams["cake"].T < 320
    assert result.performance["net_solid_change_kmol_h"]["NaCl"] == pytest.approx(0)


def test_equilibrium_mode_requires_wash(thermo):
    with pytest.raises(UnitOperationError, match="wash inlet"):
        Filter("F", thermo, PARAMS).solve({"in": states(thermo)["in"]})


def test_legacy_model_still_rejects_unequal_temperatures(thermo):
    params = {
        k: v
        for k, v in PARAMS.items()
        if k not in ("washing_model", "wash_steps", "equilibrium_nucleus_diameter")
    }
    with pytest.raises(UnitOperationError, match="same temperature"):
        Filter("F", thermo, params).solve(states(thermo))


def test_equilibrium_washing_with_deliquoring(thermo):
    feed = thermo.calculate_state(298.15, 3, 101, {"water": 100 / 101, "NaCl": 1 / 101})
    wash = thermo.calculate_state(320, 3, 2, {"water": 1})
    inlets = {"in": feed, "wash": wash}
    params = PARAMS | {
        "deliquoring_time": 30,
        "entry_pressure": 0.1,
        "residual_saturation": 0.1,
        "pore_index": 0.5,
    }
    result = Filter("F", thermo, params).solve(inlets)
    balances(inlets, result)
    assert result.performance["cake_saturation"] < 1


def test_pressure_rating_sizing_roundtrip(thermo):
    feed = thermo.calculate_state(298.15, 3, 101, {"water": 100 / 101, "NaCl": 1 / 101})
    wash = thermo.calculate_state(320, 3, 2, {"water": 1})
    inlets = {"in": feed, "wash": wash}
    sized = Filter("F", thermo, PARAMS).solve(inlets)
    params = {k: v for k, v in PARAMS.items() if k != "P_drop"}
    solved = Filter("F", thermo, params | {"area": sized.performance["area_m2"]}).solve(
        inlets
    )
    assert solved.performance["pressure_drop_bar"] == pytest.approx(2, rel=1e-6)
    rated = Filter(
        "F", thermo, PARAMS | {"area": sized.performance["area_m2"] / 2}
    ).solve(inlets)
    assert not rated.performance["cycle_feasible"]
    assert rated.warnings
    balances(inlets, solved)


def test_pfd_example_and_report(tmp_path):
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "equilibrium_warm_melt_washing.pfd"
    )
    sim = Simulator.from_file(str(path))
    result = sim.run()
    assert result.converged, result.errors
    assert abs(result.mass_balance_error) < 1e-8
    assert abs(result.energy_balance_error) < 1e-6
    p = result.units["F"].performance
    assert p["washing_model"] == "equilibrium"
    assert p["maximum_equilibrium_residual"] < 1e-8
    assert p["pressure_drop_bar"] == pytest.approx(2)
    sim.write_results(str(tmp_path / "warm.pfr"))
    assert "cell_temperatures_K" in (tmp_path / "warm.pfr").read_text()


def test_pressure_search_can_start_above_a_small_feasible_drop(thermo):
    feed = thermo.calculate_state(298.15, 0.035, 101, {'water': 100/101, 'NaCl': 1/101})
    wash = thermo.calculate_state(298.5, 0.035, 2, {'water': 1})
    inlets = {'in': feed, 'wash': wash}
    params = PARAMS | {'P_drop': 0.0001, 'wash_steps': 4}
    sized = Filter('F', thermo, params).solve(inlets)
    params.pop('P_drop')
    result = Filter('F', thermo, params | {'area': sized.performance['area_m2']}).solve(inlets)
    assert result.performance['pressure_drop_bar'] == pytest.approx(0.0001, abs=1e-8)
    balances(inlets, result)
