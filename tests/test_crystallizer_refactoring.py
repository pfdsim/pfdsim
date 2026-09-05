from unittest.mock import Mock

import pytest

from chemical_properties import ChemicalDatabase
from crystallizer_specs import normalize_crystallizer_parameters
from dof_analyzer import SpecificationStatus, analyze_dof
from particle_size_distributions import ParticleSizeDistribution
from pfd_parser import parse_pfd, validate_pfd
from simulator import Simulator
from thermodynamics_models.base import FluidPhaseEquilibrium, IdealThermodynamics
from thermodynamics_models.common import ThermodynamicsError
from thermodynamics_models.factory import create_thermodynamics
from unit_operations_base import UnitOperationError
from unit_operations_solids import Crystallizer


@pytest.fixture
def activity():
    thermo = create_thermodynamics(
        ["water", "ethanol"], "NRTL", ChemicalDatabase(enable_online=False)
    )
    thermo.configure_permanent_solids(
        ["water", "ethanol"], [], conventional_solid_components=["water"]
    )
    return thermo


def feed_for(thermo):
    return thermo.calculate_state(
        280, 1, 10, {"water": 0.9, "ethanol": 0.1}, phase="liquid"
    )


def params_for(model, split):
    params = {"T": 250, "outlet_sphericity": 0.74}
    if model == "msmpr":
        params.update(
            {
                "model": "MSMPR",
                "tau": 1,
                "quadrature_classes": 4,
                "growth": {"expression": "1e-5", "rate_unit": "m/h"},
                "nucleation": {"expression": "1e9", "rate_unit": "1/m3/h"},
            }
        )
    if split:
        params["mother_liquor_retention"] = 0.1
    return params


def equilibrium_result(second_liquid):
    return FluidPhaseEquilibrium(
        vapor_fraction=0,
        liquid1_fraction=1 - second_liquid,
        liquid2_fraction=second_liquid,
        y={},
        x1={"water": 0.99, "ethanol": 0.01},
        x2={"water": 0.1, "ethanol": 0.9} if second_liquid else {},
        status="lle_only" if second_liquid else "single_liquid",
        stability="test_equilibrium",
    )


@pytest.mark.parametrize("mode", ["VL(L)E", "VLLE"])
@pytest.mark.parametrize("model", ["equilibrium", "msmpr"])
@pytest.mark.parametrize("split", [False, True])
def test_lle_warning_probes_outlet_liquid_once_without_changing_balances(
    activity, monkeypatch, mode, model, split
):
    feed = feed_for(activity)
    params = params_for(model, split)
    baseline = Crystallizer("C", activity, params).solve({"in": feed})
    probe = Mock(return_value=equilibrium_result(0.4))
    monkeypatch.setattr(activity, "_fluid_phase_equilibrium_TP", probe)
    activity.set_fluid_phase_model(mode)
    result = Crystallizer("C", activity, params).solve({"in": feed})
    probe.assert_called_once()
    composition, temperature, pressure = probe.call_args.args
    assert composition == pytest.approx(result.performance["mother_liquor_composition"])
    assert temperature == 250
    assert pressure == 1
    assert len(result.warnings) == 1
    assert "does not support crystallization with LLE present" in result.warnings[0]
    assert result.performance["outlet_lle_check"]["lle_detected"]
    assert result.heat_duty == pytest.approx(baseline.heat_duty)
    for port, stream in result.outlet_streams.items():
        assert stream.component_flows() == pytest.approx(
            baseline.outlet_streams[port].component_flows()
        )
        assert stream.solid_component_flows == pytest.approx(
            baseline.outlet_streams[port].solid_component_flows
        )
        assert stream.phase_fractions()["liquid2"] == 0
        stream.validate_particle_size_distributions()
        assert stream.phase_details["crystallizer_lle_check"]["lle_detected"]
        if model == "msmpr":
            assert (
                stream.phase_details["particle_population_balance"][
                    "particle_sphericity"
                ]
                == 0.74
            )
    if model == "msmpr":
        assert result.performance["particle_sphericity"] == 0.74


@pytest.mark.parametrize("second_liquid", [0, 1e-12])
def test_stable_outlet_does_not_warn(activity, monkeypatch, second_liquid):
    feed = feed_for(activity)
    activity.set_fluid_phase_model("VLLE")
    monkeypatch.setattr(
        activity,
        "_fluid_phase_equilibrium_TP",
        Mock(return_value=equilibrium_result(second_liquid)),
    )
    result = Crystallizer("C", activity, {"T": 250}).solve({"in": feed})
    assert not result.warnings
    assert not result.performance["outlet_lle_check"]["lle_detected"]


def test_vle_mode_does_not_probe(activity, monkeypatch):
    feed = feed_for(activity)
    probe = Mock(side_effect=AssertionError("unexpected phase probe"))
    monkeypatch.setattr(activity, "_fluid_phase_equilibrium_TP", probe)
    result = Crystallizer("C", activity, {"T": 250}).solve({"in": feed})
    probe.assert_not_called()
    assert not result.warnings
    assert "outlet_lle_check" not in result.performance


def test_nonactivity_model_does_not_probe(monkeypatch):
    thermo = IdealThermodynamics(
        ["water", "ethanol"], ChemicalDatabase(enable_online=False)
    )
    thermo.configure_permanent_solids(
        ["water", "ethanol"], [], conventional_solid_components=["water"]
    )
    feed = feed_for(thermo)
    # Even a manually modified phase-policy flag must not enable an unsupported probe.
    thermo.fluid_phase_model = "VLLE"
    probe = Mock(side_effect=AssertionError("unexpected phase probe"))
    monkeypatch.setattr(thermo, "_fluid_phase_equilibrium_TP", probe)
    result = Crystallizer("C", thermo, {"T": 250}).solve({"in": feed})
    probe.assert_not_called()
    assert not result.warnings


def test_failed_lle_check_is_reported_as_unchecked(activity, monkeypatch):
    feed = feed_for(activity)
    activity.set_fluid_phase_model("VLLE")
    monkeypatch.setattr(
        activity,
        "_fluid_phase_equilibrium_TP",
        Mock(side_effect=ThermodynamicsError("probe failed")),
    )
    result = Crystallizer("C", activity, {"T": 250}).solve({"in": feed})
    assert "outlet LLE check failed" in result.warnings[0]
    assert result.performance["outlet_lle_check"]["checked"] is False


@pytest.mark.parametrize("split", [False, True])
def test_inert_particles_preserve_feed_shape_and_psd(split):
    thermo = IdealThermodynamics(
        ["water", "ethanol"], ChemicalDatabase(enable_online=False)
    )
    thermo.configure_permanent_solids(
        ["water", "ethanol", "NaCl"],
        ["NaCl"],
        {"NaCl": {"diameter_m": 1e-4, "sphericity": 0.9}},
        conventional_solid_components=["water"],
    )
    feed = thermo.calculate_state(
        280, 1, 10, {"water": 0.8, "ethanol": 0.1, "NaCl": 0.1}, phase="liquid"
    )
    feed.solid_particle_properties["NaCl"] = {"diameter_m": 2e-4, "sphericity": 0.55}
    distribution = ParticleSizeDistribution((1e-5, 2e-4), (0.4, 0.6))
    feed.solid_particle_size_distributions["NaCl"] = distribution
    result = Crystallizer("C", thermo, params_for("equilibrium", split)).solve(
        {"in": feed}
    )
    solid_out = result.outlet_streams["cake" if split else "out"]
    assert (
        solid_out.solid_particle_properties["NaCl"]
        == feed.solid_particle_properties["NaCl"]
    )
    assert solid_out.solid_particle_properties["water"]["sphericity"] == 0.74
    psd = solid_out.solid_particle_size_distributions["NaCl"]
    assert psd.diameters_m == distribution.diameters_m
    assert psd.molar_flows_kmol_per_h == pytest.approx(
        distribution.molar_flows_kmol_per_h
    )
    solid_out.solid_particle_properties["NaCl"]["sphericity"] = 0.8
    assert feed.solid_particle_properties["NaCl"]["sphericity"] == 0.55


@pytest.mark.parametrize("temperature", ["T_out", "Tout", "T", "temperature"])
@pytest.mark.parametrize("pressure", ["P_out", "Pout", "P", "pressure"])
def test_aliases_use_same_units_in_pfd_dof_and_runtime(temperature, pressure):
    source = f"""
PROCESS: aliases
ONLINE_LOOKUP: false
COMPONENTS:
    water | Water | type=conventional_with_solid
    ethanol | Ethanol
STREAM Feed : FEED -> C.in
    T = 280 [K]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.9, ethanol:0.1
STREAM Product : C.out -> PRODUCT
UNIT C : Crystallizer
    {temperature} = -23.15 [C]
    {pressure} = 150 [kPa]
"""
    pfd = parse_pfd(source)
    assert not validate_pfd(pfd)[0]
    assert analyze_dof(pfd).unit_results[0].status == SpecificationStatus.OK
    result = Simulator(pfd).run()
    assert result.converged, result.errors
    assert result.streams["Product"].T == pytest.approx(250)
    assert result.streams["Product"].P == pytest.approx(1.5)


@pytest.mark.parametrize(
    "spec",
    [
        {"max_iterations": 2.5},
        {"max_iterations": float("inf")},
        {"equilibrium_tolerance": "bad"},
    ],
)
def test_invalid_numerics_raise_unit_errors(activity, spec):
    with pytest.raises(UnitOperationError):
        Crystallizer("C", activity, {"T": 250, **spec}).solve(
            {"in": feed_for(activity)}
        )


def test_normalization_preserves_rate_expressions_and_units():
    result = normalize_crystallizer_parameters(
        {
            "Tout": 250,
            "__unit__Tout": "K",
            "crystallizer_model": "steady-MSMPR",
            "TAU": 1,
            "__unit__TAU": "h",
            "growth_expression": "k * L",
        }
    )
    assert result["t_out"] == 250 and result["__unit__t_out"] == "K"
    assert result["model"] == "msmpr"
    assert result["residence_time"] == 1 and result["__unit__residence_time"] == "h"
    assert result["growth_expression"] == "k * L"


def test_case_sensitive_custom_kinetic_parameters_survive_normalization(activity):
    params = {
        "T": 250,
        "model": "MSMPR",
        "tau": 1,
        "quadrature_classes": 4,
        "growth_expression": "kG",
        "growth_param_kG": 1e-5,
        "growth_rate_unit": "m/h",
        "nucleation_expression": "kB",
        "nucleation_param_kB": 1e9,
        "nucleation_rate_unit": "1/m3/h",
    }
    unit = Crystallizer("C", activity, params)
    result = unit.solve({"in": feed_for(activity)})
    assert result.performance["growth_expression"] == "kG"
    assert result.performance["nucleation_expression"] == "kB"
    repeated = unit.solve({"in": feed_for(activity)})
    assert repeated.performance["solid_component_flows_kmol_per_h"] == pytest.approx(
        result.performance["solid_component_flows_kmol_per_h"]
    )


@pytest.mark.parametrize("mode", ["VL(L)E", "VLLE"])
def test_real_outlet_lle_warning_reaches_flowsheet_report(mode, tmp_path):
    source = f"""
PROCESS: Crystallizer LLE diagnostic
ONLINE_LOOKUP: false
THERMO_METHOD: UNIFNIST
FLUID_PHASE_MODEL: {mode}
COMPONENTS:
    water | Water | type=conventional_with_solid
    toluene | Toluene
STREAM Feed : FEED -> C.in
    T = 298.15 [K]
    P = 1 [bar]
    F = 10 [kmol/h]
    x = water:0.5, toluene:0.5
STREAM Product : C.out -> PRODUCT
UNIT C : Crystallizer
    T = 298.15 [K]
"""
    simulator = Simulator.from_string(source)
    result = simulator.run()
    assert result.converged, result.errors
    assert result.units["C"].performance["outlet_lle_check"]["lle_detected"]
    assert any(
        "does not support crystallization with LLE present" in w
        for w in result.warnings
    )
    report = tmp_path / "crystallizer.pfr"
    simulator.write_results(str(report))
    assert "does not support crystallization with LLE present" in report.read_text()
