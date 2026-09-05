import math
from pathlib import Path

import numpy as np
import pytest

from chemical_properties import ChemicalDatabase
from dof_analyzer import analyze_dof
from filtration_models import cycle_coefficients, deliquor, required_area, wash_profile
from particle_size_distributions import ParticleSizeDistribution
from pfd_parser import parse_pfd, validate_pfd
from simulator import Simulator
from thermodynamics_models.base import IdealThermodynamics
from unit_operations import Filter, create_unit
from unit_operations_base import UnitOperationError


def test_washing_analytic_mixed_and_displacement_limits():
    for ratio in (0.0, 0.01, 0.5, 1.0, 3.0, 100.0):
        assert wash_profile(ratio, 1)[0] == pytest.approx(math.exp(-ratio))
    assert np.all(wash_profile(0, 20) == 1)
    assert np.mean(wash_profile(0.4, 200)) == pytest.approx(0.6, abs=1e-8)
    assert np.mean(wash_profile(2, 200)) < 1e-20


def test_darcy_formation_washing_and_area_against_hand_calculation():
    formation, washing = cycle_coefficients(
        dry_mass=10,
        filtrate_volume=0.5,
        pore_volume=0.01,
        wash_volume=0.03,
        alpha=1e10,
        medium_resistance=2e9,
        pressure_drop=2e5,
        feed_viscosity=1e-3,
        wash_viscosity=1e-3,
        wash_cells=10,
    )
    # A=2 m²: t_f = mu/dP * (alpha*M*V/(2*A²) + Rm*V/A).
    assert formation[0] / 4 + formation[1] / 2 == pytest.approx(33.75)
    assert washing[0] / 4 + washing[1] / 2 == pytest.approx(3.9)
    assert required_area(
        formation[0] + washing[0], formation[1] + washing[1], 37.65
    ) == pytest.approx(2)


def test_viscosity_transient_integral_matches_single_cell_solution():
    _, washing = cycle_coefficients(
        dry_mass=1,
        filtrate_volume=1,
        pore_volume=1,
        wash_volume=2,
        alpha=1,
        medium_resistance=1,
        pressure_drop=1,
        feed_viscosity=2,
        wash_viscosity=1,
        wash_cells=1,
    )
    assert washing == pytest.approx((3 - math.exp(-2), 3 - math.exp(-2)))


def drain(**overrides):
    params = {
        "duration": 10,
        "pore_volume": 0.01,
        "area": 1,
        "cake_resistance": 1e10,
        "medium_resistance": 1e9,
        "viscosity": 0.001,
        "pressure_drop": 2e5,
        "entry_pressure": 2e4,
        "residual_saturation": 0.1,
        "pore_index": 0.5,
        "relative_permeability_exponent": 7,
    }
    return deliquor(**(params | overrides))


def test_deliquoring_threshold_time_and_capillary_equilibrium():
    assert drain(duration=0)[0] == 1
    assert drain(pressure_drop=1e4) == (1, 1)
    assert drain(pressure_drop=2e4) == (1, 1)
    short, equilibrium = drain()
    long, _ = drain(duration=1e5)
    assert equilibrium == pytest.approx(0.1 + 0.9 * math.sqrt(0.1))
    assert equilibrium <= long < short < 1
    assert drain(area=2)[0] < short
    assert drain(viscosity=0.002)[0] > short


@pytest.fixture(scope="module")
def thermo():
    model = IdealThermodynamics(
        ["water", "ethanol"], ChemicalDatabase(enable_online=False)
    )
    model.configure_permanent_solids(
        ["water", "ethanol", "NaCl"],
        ["NaCl"],
        {"NaCl": {"diameter_m": 1e-4, "sphericity": 0.9}},
    )
    return model


@pytest.fixture
def feeds(thermo):
    feed = thermo.calculate_state(
        298.15, 3, 101, {"water": 90 / 101, "ethanol": 10 / 101, "NaCl": 1 / 101}
    )
    feed.solid_particle_size_distributions["NaCl"] = ParticleSizeDistribution(
        (1e-5, 1e-4), (0.2, 0.8)
    )
    wash = thermo.calculate_state(298.15, 3, 1, {"water": 1})
    return {"in": feed, "wash": wash}


PARAMS = {
    "cycle_time": 600,
    "P_drop": 2,
    "porosity": 0.4,
    "capture_cut_size": 0,
    "specific_cake_resistance": 1e10,
    "medium_resistance": 1e9,
    "liquid_viscosity": 0.001,
}
DRAIN = {
    "deliquoring_time": 60,
    "entry_pressure": 0.2,
    "residual_saturation": 0.1,
    "pore_index": 0.5,
}


def assert_balances(inlets, result):
    components = set().union(*(s.composition for s in inlets.values()))
    for c in components:
        incoming = sum(s.component_flows().get(c, 0) for s in inlets.values())
        outgoing = sum(
            s.component_flows().get(c, 0) for s in result.outlet_streams.values()
        )
        assert outgoing == pytest.approx(incoming, abs=1e-10)
    h_in = sum(s.F * s.H for s in inlets.values())
    h_out = sum(s.F * s.H for s in result.outlet_streams.values())
    assert h_in + result.heat_duty == pytest.approx(h_out, abs=1e-8)


@pytest.mark.parametrize(
    "washing,draining", [(False, False), (True, False), (False, True), (True, True)]
)
def test_unit_component_energy_solids_and_population_balances(
    thermo, feeds, washing, draining
):
    inlets = feeds if washing else {"in": feeds["in"]}
    result = Filter("F", thermo, PARAMS | (DRAIN if draining else {})).solve(inlets)
    assert_balances(inlets, result)
    cake, filtrate = result.outlet_streams["cake"], result.outlet_streams["filtrate"]
    assert cake.solid_component_flows == pytest.approx({"NaCl": 1})
    assert not filtrate.solid_component_flows
    actual = cake.solid_particle_size_distributions["NaCl"]
    expected = feeds["in"].solid_particle_size_distributions["NaCl"]
    assert actual.diameters_m == expected.diameters_m
    assert actual.molar_flows_kmol_per_h == pytest.approx(
        expected.molar_flows_kmol_per_h
    )
    assert not filtrate.solid_particle_size_distributions
    assert cake.solid_particle_properties["NaCl"]["sphericity"] == 0.9
    assert result.performance["cycle_feasible"]
    assert result.performance["required_cycle_time_s"] == pytest.approx(600)
    assert (
        result.performance["cake_saturation"] < 1
        if draining
        else result.performance["cake_saturation"] == 1
    )


def test_washing_removes_impurity_and_drainage_reduces_liquid(thermo, feeds):
    raw = Filter("F", thermo, PARAMS).solve({"in": feeds["in"]})
    washed = Filter("F", thermo, PARAMS).solve(feeds)
    dried = Filter("F", thermo, PARAMS | DRAIN).solve(feeds)
    assert (
        washed.outlet_streams["cake"].component_flows()["ethanol"]
        < raw.outlet_streams["cake"].component_flows()["ethanol"]
    )
    assert (
        dried.performance["retained_liquid_kg_per_h"]
        < washed.performance["retained_liquid_kg_per_h"]
    )


def test_rating_capacity_and_compressible_cake(thermo, feeds):
    size = Filter("F", thermo, PARAMS).solve(feeds).performance["area_m2"]
    rating = Filter("F", thermo, PARAMS | {"area": size / 2}).solve(feeds)
    assert not rating.performance["cycle_feasible"]
    assert rating.performance["capacity_ratio"] == pytest.approx(0.5)
    assert rating.warnings
    assert_balances(feeds, rating)
    compressed = Filter("F", thermo, PARAMS | {"compressibility": 0.5}).solve(feeds)
    assert compressed.performance["specific_cake_resistance_m_per_kg"] == pytest.approx(
        1e10 * math.sqrt(2)
    )
    assert compressed.performance["area_m2"] > size


@pytest.mark.parametrize(
    "params,match",
    [
        ({"P_drop": 3}, "positive filtrate pressure"),
        ({"porosity": 1}, "porosity"),
        ({"porosity": 0}, "porosity"),
        ({"specific_cake_resistance": -1}, "specific_cake_resistance"),
        ({"cycle_time": math.nan}, "cycle_time"),
        ({"area": 0}, "area"),
        ({"compressibility": 1}, "compressibility"),
        ({"wash_cells": 1.5}, "wash_cells"),
        ({"deliquoring_time": 600}, "cycle_time"),
        ({"entry_pressure": 0.1}, "positive deliquoring_time"),
        ({"deliquoring_time": 1}, "entry_pressure"),
        ({"cake_moisture": 0.2}, "unknown parameter"),
        ({"__unit__cycle_time": "fortnight"}, "invalid unit"),
        ({"__connected_outlet_ports__": ["cake"]}, "connected cake and filtrate"),
    ],
)
def test_invalid_specs(thermo, feeds, params, match):
    with pytest.raises(UnitOperationError, match=match):
        Filter("F", thermo, PARAMS | params).solve(feeds)


def test_invalid_feed_and_wash_conditions(thermo, feeds):
    with pytest.raises(UnitOperationError, match="suspended solids"):
        Filter("F", thermo, PARAMS).solve({"in": feeds["wash"]})
    with pytest.raises(UnitOperationError, match="solids-free"):
        Filter("F", thermo, PARAMS).solve({"in": feeds["in"], "wash": feeds["in"]})
    cold = feeds["wash"].copy()
    cold.T -= 1
    with pytest.raises(UnitOperationError, match="same temperature"):
        Filter("F", thermo, PARAMS).solve({"in": feeds["in"], "wash": cold})
    dense = thermo.calculate_state(298.15, 3, 100, {"water": 0.001, "NaCl": 0.999})
    with pytest.raises(UnitOperationError, match="insufficient liquid"):
        Filter("F", thermo, PARAMS).solve({"in": dense})


def test_conventional_solid_is_preserved_without_reequilibrating():
    model = IdealThermodynamics(["water"], ChemicalDatabase(enable_online=False))
    model.configure_permanent_solids(
        ["water"], [], conventional_solid_components=["water"]
    )
    feed = model.calculate_state_with_solid_flows(
        270, 3, 10, {"water": 1}, {"water": 1}, phase="liquid"
    )
    result = Filter("F", model, PARAMS).solve({"in": feed})
    assert result.outlet_streams["cake"].solid_component_flows == pytest.approx(
        {"water": 1}
    )
    assert_balances({"in": feed}, result)


def test_example_parser_dof_roundtrip_and_simulation(tmp_path):
    path = (
        Path(__file__).resolve().parents[1] / "examples" / "cake_filtration_washing.pfd"
    )
    pfd = parse_pfd(path.read_text())
    errors, _ = validate_pfd(pfd)
    assert not errors
    assert not analyze_dof(pfd).errors
    reparsed = parse_pfd(pfd.to_pfd())
    assert not analyze_dof(reparsed).errors
    sim = Simulator(reparsed)
    result = sim.run()
    assert result.converged
    assert abs(result.mass_balance_error) < 1e-8
    assert result.units['F'].performance['pressure_drop_bar'] == pytest.approx(2)
    assert result.units['F'].performance['cycle_time_s'] == 600
    report_path = tmp_path / 'filter.pfr'
    sim.write_results(str(report_path))
    report = report_path.read_text()
    assert 'solid_capture_mass_fraction' in report
    assert 'pressure_drop_bar' in report
    assert 'SOLID_PARTICLE_SIZE_DISTRIBUTIONS:' in report
    assert create_unit("Filter", "F", None, {}).__class__ is Filter


def test_class_selective_capture_and_shape_effect_on_slip(thermo, feeds):
    feed = feeds["in"]
    feed.solid_particle_properties["NaCl"]["sphericity"] = 1
    params = PARAMS | {"capture_cut_size": 1e-5, "capture_sharpness": 2}
    result = Filter("F", thermo, params | DRAIN).solve(feeds)
    cake = result.outlet_streams["cake"].solid_particle_size_distributions["NaCl"]
    filtrate = result.outlet_streams["filtrate"].solid_particle_size_distributions[
        "NaCl"
    ]
    assert cake.molar_flows_kmol_per_h == pytest.approx((0.1, 0.8 * 100 / 101))
    assert np.add(
        cake.molar_flows_kmol_per_h, filtrate.molar_flows_kmol_per_h
    ) == pytest.approx((0.2, 0.8))
    assert_balances(feeds, result)
    assert filtrate.molar_fractions[0] > 0.9
    feed.solid_particle_properties["NaCl"]["sphericity"] = 0.5
    less_spherical = Filter("F", thermo, params | DRAIN).solve(feeds)
    assert (
        less_spherical.performance["solid_capture_mass_fraction"]
        > result.performance["solid_capture_mass_fraction"]
    )
    assert_balances(feeds, less_spherical)


def test_kozeny_carman_uses_retained_psd_surface_and_sphericity(thermo, feeds):
    params = {k: v for k, v in PARAMS.items() if k != "specific_cake_resistance"}
    result = Filter("F", thermo, params).solve({"in": feeds["in"]})
    solid_vm = thermo._solid_molar_volume("NaCl", feeds["in"].T)
    density = thermo.props["NaCl"].MW / solid_vm
    surface = 6 / 0.9 * (0.2 / 1e-5 + 0.8 / 1e-4)
    expected_alpha = 5 * surface**2 * 0.6 / (density * 0.4**3)
    assert result.performance["specific_cake_resistance_m_per_kg"] == pytest.approx(
        expected_alpha
    )
    feeds["in"].solid_particle_properties["NaCl"]["sphericity"] = 0.45
    shaped = Filter("F", thermo, params).solve({"in": feeds["in"]})
    assert shaped.performance["specific_cake_resistance_m_per_kg"] == pytest.approx(
        4 * expected_alpha
    )
    assert shaped.performance["area_m2"] > result.performance["area_m2"]
    selective = Filter("F", thermo, params | {"capture_cut_size": 2e-5}).solve(
        {"in": feeds["in"]}
    )
    assert (
        selective.performance["specific_cake_resistance_m_per_kg"]
        < shaped.performance["specific_cake_resistance_m_per_kg"]
    )


def test_pressure_solution_and_particle_size_scaling(thermo, feeds):
    params = {k: v for k, v in PARAMS.items() if k != "specific_cake_resistance"}
    params["medium_resistance"] = 0
    sized = Filter("F", thermo, params).solve(feeds)
    pressure_params = {k: v for k, v in params.items() if k != "P_drop"}
    pressure_params["area"] = sized.performance["area_m2"]
    solved = Filter("F", thermo, pressure_params).solve(feeds)
    assert solved.performance["mode"] == "pressure"
    assert solved.performance["pressure_drop_bar"] == pytest.approx(2)
    assert_balances(feeds, solved)
    original = feeds["in"].solid_particle_size_distributions["NaCl"]
    feeds["in"].solid_particle_size_distributions["NaCl"] = ParticleSizeDistribution(
        tuple(2 * d for d in original.diameters_m),
        original.molar_flows_kmol_per_h,
    )
    coarser = Filter("F", thermo, pressure_params).solve(feeds)
    assert coarser.performance["pressure_drop_bar"] == pytest.approx(0.5)
    with pytest.raises(UnitOperationError, match="available feed pressure"):
        Filter(
            "F", thermo, pressure_params | {"area": pressure_params["area"] / 10}
        ).solve(feeds)


def test_selective_conventional_solid_preserves_dissolved_inventory():
    model = IdealThermodynamics(["water"], ChemicalDatabase(enable_online=False))
    model.configure_permanent_solids(
        ["water"], [], conventional_solid_components=["water"]
    )
    feed = model.calculate_state_with_solid_flows(
        270, 3, 10, {"water": 1}, {"water": 1}, phase="liquid"
    )
    feed.solid_particle_size_distributions["water"] = ParticleSizeDistribution(
        (1e-5,), (1,)
    )
    feed.solid_particle_properties["water"] = {"sphericity": 1}
    result = Filter("F", model, PARAMS | {"capture_cut_size": 1e-5}).solve({"in": feed})
    assert result.outlet_streams["cake"].solid_component_flows[
        "water"
    ] == pytest.approx(0.5)
    assert result.outlet_streams["filtrate"].solid_component_flows[
        "water"
    ] == pytest.approx(0.5)
    assert sum(
        s.phase_component_flows()["liquid1"]["water"]
        for s in result.outlet_streams.values()
    ) == pytest.approx(9)
    assert_balances({"in": feed}, result)


def test_missing_particle_data_fails_for_predictive_resistance(thermo, feeds):
    feed = feeds["in"]
    feed.solid_particle_size_distributions = {}
    feed.solid_particle_properties = {}
    params = {k: v for k, v in PARAMS.items() if k != "specific_cake_resistance"}
    with pytest.raises(UnitOperationError, match="PSD"):
        Filter("F", thermo, params).solve({"in": feed})


def test_filter_dof_missing_specs_and_numeric_ports():
    path = (
        Path(__file__).resolve().parents[1] / "examples" / "cake_filtration_washing.pfd"
    )
    text = path.read_text()
    numeric = (
        text.replace("F.in", "F.1")
        .replace("F.wash", "F.2")
        .replace("F.filtrate", "F.3")
        .replace("F.cake", "F.4")
    )
    pfd = parse_pfd(numeric)
    assert not validate_pfd(pfd)[0]
    pfd.units[0].params = [
        p for p in pfd.units[0].params if p.name != "capture_cut_size"
    ]
    assert any("capture_cut_size" in error for error in analyze_dof(pfd).errors)
