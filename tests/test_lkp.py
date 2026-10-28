"""LKP data provenance, EOS thermodynamic identities, and simulator integration."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from chemical_properties import ChemicalDatabase, ChemicalProperties
from pfd_parser import ParseError, parse_pfd
from scripts.build_lkp_parameters import build
from simulator import Simulator, SimulationError
from thermodynamics import (
    LKPThermodynamics, ThermodynamicsError, ThermodynamicsValidityError,
    create_thermodynamics,
)
from thermodynamics_models.common import R
from thermodynamics_models.factory import SUPPORTED_METHODS
from thermodynamics_models.lkp import (
    _fluid_state, _pressure_and_slope, _coefficients,
    _spinodals, estimate_kij, reduced_roots,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def binary():
    return create_thermodynamics(["CH4", "C2H6"], "LKP", ChemicalDatabase(enable_online=False))


def test_parameter_table_is_reproducible_and_has_no_duplicate_pairs():
    payload = json.loads((ROOT / "data/lkp_binary_interactions_cas.json").read_text())
    assert payload == json.loads(json.dumps(build()))
    assert len(payload["records"]) == 142
    assert len(payload["corrections"]) == 13
    pairs = {(r["cas1"], r["cas2"]): r["kij"] for r in payload["records"]}
    assert len(pairs) == 142
    assert all(a != b and 0.5 < value < 4 for (a, b), value in pairs.items())
    for pair, expected in (
        (("124-38-9", "71-43-2"), 1.018),
        (("1333-74-0", "74-84-0"), 1.604),
        (("1333-74-0", "74-85-1"), 1.498),
        (("106-97-8", "1333-74-0"), 2.093),
        (("111-84-2", "124-18-5"), 0.991),
        (("74-82-8", "7727-37-9"), 0.977),
    ):
        assert pairs[tuple(sorted(pair))] == expected


def species(cas, Tc, Vc, omega=0.2, formula="C12H26"):
    return SimpleNamespace(CAS=cas, Tc=Tc, Vc=Vc, omega=omega, formula=formula)


def acentric_model(omega):
    props = ChemicalProperties(symbol="test", name="test", formula="C12H26", MW=170.,
                               Tc=662.927608670662, Pc=18.,
                               Vc=750.0617371029749, omega=omega)
    return LKPThermodynamics(["test"], SimpleNamespace(get_user_component=lambda c: props))


@pytest.mark.parametrize("gas,kind", [
    (species("1333-74-0", 33, 65), "hydrogen"),
    (species("74-82-8", 190, 99, formula="CH4"), "asymmetric"),
    (species("7727-37-9", 126, 90, formula="N2"), "asymmetric"),
    (species("630-08-0", 133, 93, formula="CO"), "asymmetric"),
    (species("124-38-9", 304.21, 94, formula="CO2"), "CO2/H2S"),
    (species("7783-06-4", 373, 98, formula="H2S"), "CO2/H2S"),
])
def test_missing_pair_correlations_and_symmetry(gas, kind):
    heavy = species("112-40-3", 658.1, 754, 0.574)
    value, source = estimate_kij(gas, heavy)
    assert kind in source
    assert estimate_kij(heavy, gas) == (value, source)
    if kind == "hydrogen":
        expected = .2036 + .002246 * 658.1 + .8364 * .574 + 1.2111 * .754**(1/3)
    elif kind == "asymmetric":
        expected = 1 + .2802 * math.log(658.1/gas.Tc) * math.log(754/gas.Vc) - .0561 * math.log(658.1/gas.Tc)
    else:
        X = math.log(658.1/304.21) * math.log(.754/.094)
        expected = .9276 + .10719 * X + .04505 * X*X
        if gas.CAS == "7783-06-4":
            expected -= .027
    assert value == pytest.approx(expected, rel=1e-14)


def test_correlation_domains_exclude_polar_compounds_and_similar_criticals():
    polar = species("", 650, 150, formula="C2H6O")
    light = species("", 190, 99, formula="CH4")
    assert estimate_kij(light, polar)[0] == 1
    assert estimate_kij(species("", 500, 300), species("", 550, 400))[0] == 1
    co2 = species("124-38-9", 304.21, 94, formula="CO2")
    assert estimate_kij(co2, light)[0] == 1


@pytest.mark.parametrize("fluid", [0, 1])
@pytest.mark.parametrize("Tr,Pr", [(0.65, .01), (.9, .4), (.999, .98), (1.2, 2), (3., 25.), (2., 1e-8)])
def test_stable_roots_satisfy_eos(fluid, Tr, Pr):
    roots = reduced_roots(Tr, Pr, fluid)
    assert roots == tuple(sorted(roots))
    for v in roots:
        pressure, slope = _pressure_and_slope(v, Tr, _coefficients(Tr, fluid))
        assert pressure == pytest.approx(Pr, rel=2e-9, abs=1e-12)
        assert slope <= 0


@pytest.mark.parametrize("fluid", [0, 1])
def test_root_bracketing_finds_nearly_coincident_spinodal_roots(fluid):
    Tr = .99
    extrema = _spinodals(Tr, fluid)
    assert len(extrema) >= 2
    Pr = _pressure_and_slope(extrema[-1], Tr, _coefficients(Tr, fluid))[0] * (1 - 1e-9)
    roots = reduced_roots(Tr, Pr, fluid)
    assert len(roots) == 2
    assert roots[-1] > extrema[-1]


@pytest.mark.parametrize("fluid", [0, 1])
def test_both_roots_survive_very_close_to_critical_temperature(fluid):
    # Both fluids have Tc within 3e-7 of reduced unity. The unstable volume
    # interval here is much narrower than a reasonable direct root grid.
    Tr = .9999995
    extrema = _spinodals(Tr, fluid)
    assert len(extrema) == 2
    pressures = [_pressure_and_slope(v, Tr, _coefficients(Tr, fluid))[0] for v in extrema]
    Pr = sum(pressures) / 2
    roots = reduced_roots(Tr, Pr, fluid)
    assert len(roots) == 2
    assert roots[0] < extrema[0] < extrema[1] < roots[1]


@pytest.mark.parametrize("fluid,Tr,vc", [
    (0, .9999997157370651, .2905008596182494),
    (1, .9999999244985385, .2559988654084895),
])
def test_critical_limiting_root_is_not_rejected_by_slope_roundoff(fluid, Tr, vc):
    coefficients = _coefficients(Tr, fluid)
    Pr = _pressure_and_slope(vc, Tr, coefficients)[0]
    roots = reduced_roots(Tr, Pr, fluid)
    assert roots
    for v in roots:
        assert _pressure_and_slope(v, Tr, coefficients)[0] == pytest.approx(Pr, abs=1e-12)


@pytest.mark.parametrize("fluid", [0, 1])
def test_simple_fluid_gibbs_matches_independent_pressure_integration(fluid):
    Tr, Pr = 1.5, 3.0
    def integrand(p):
        return (_fluid_state(Tr, p, fluid, "vapor")[0] - 1) / p
    integral = quad(integrand, 1e-8, Pr, epsabs=1e-10)[0]
    g = _fluid_state(Tr, Pr, fluid, "vapor")[1]
    assert g == pytest.approx(integral, abs=2e-9)


def test_parameter_priority_and_cas_identity(binary):
    assert binary.kij[0, 1] == 1.052
    assert binary.kij[1, 0] == 1.052
    assert np.diag(binary.kij).tolist() == [1, 1]
    override = create_thermodynamics(["CH4", "C2H6"], "LKP", ChemicalDatabase(enable_online=False),
        interaction_overrides=[{"model": "LKP", "component1": "C2H6", "component2": "CH4", "kij": 1.1}])
    assert override.kij[0, 1] == 1.1
    assert override.interaction_parameter_sources[("C2H6", "CH4")]["source"] == "PFD interaction override"
    assert "LKP" in SUPPORTED_METHODS


def test_correlated_pairs_have_provenance_and_do_not_replace_table_values():
    model = create_thermodynamics(["H2", "CH4", "n-dodecane"], "LKP", ChemicalDatabase(enable_online=False))
    tabulated = model.interaction_parameter_sources[("CH4", "H2")]
    estimated = model.interaction_parameter_sources[("H2", "n-dodecane")]
    assert tabulated["kij"] == 1.216
    assert "table" in tabulated["source"]
    assert "hydrogen correlation" in estimated["source"]
    assert any("H2/n-dodecane" in warning for warning in model.warnings)


def test_missing_critical_volume_is_rejected(binary):
    from copy import deepcopy
    props = deepcopy(binary.props["CH4"])
    props.Vc = None
    db = SimpleNamespace(get_user_component=lambda c: props)
    with pytest.raises(ThermodynamicsError, match="LKP requires.*Vc"):
        LKPThermodynamics(["CH4"], db)


def test_multicomponent_permutation_and_pure_component_limit():
    components = ["CH4", "C2H6", "H2"]
    db = ChemicalDatabase(enable_online=False)
    forward = create_thermodynamics(components, "LKP", db)
    reverse = create_thermodynamics(list(reversed(components)), "LKP", db)
    x = {"CH4": .2, "C2H6": .7, "H2": .1}
    assert forward.fugacity_coefficients(300, 30, x) == pytest.approx(reverse.fugacity_coefficients(300, 30, x), rel=1e-12)
    pure = create_thermodynamics(["CH4"], "LKP", db)
    x = {"CH4": 1.}
    assert forward.compressibility_factor(180, 5, x) == pytest.approx(pure.compressibility_factor(180, 5, x), abs=1e-12)
    assert forward.departure_enthalpy(180, 5, x) == pytest.approx(pure.departure_enthalpy(180, 5, x), abs=1e-10)


@pytest.mark.parametrize("components,T,P,phase", [
    (["CH4", "C2H6"], 200., 10., "liquid"),
    (["CH4", "C2H6"], 300., 50., "vapor"),
    (["H2", "CH4"], 250., 30., "vapor"),
    (["H2", "C2H6"], 180., 10., "liquid"),
])
def test_gibbs_helmholtz_pressure_and_component_derivatives(components, T, P, phase):
    model = create_thermodynamics(components, "LKP", ChemicalDatabase(enable_online=False))
    a, b = components
    x = {a: .3, b: .7}
    dt, dp, dx = 1e-3, 1e-3, 1e-5
    g = model.departure_gibbs(T, P, x, phase)
    h = model.departure_enthalpy(T, P, x, phase)
    s = model.departure_entropy(T, P, x, phase)
    derivative = (model.departure_gibbs(T+dt, P, x, phase)/(T+dt)
                  - model.departure_gibbs(T-dt, P, x, phase)/(T-dt))/(2*dt)
    assert h == pytest.approx(-T*T*derivative, rel=2e-7, abs=1e-5)
    assert g == pytest.approx(h - T*s, abs=1e-10)
    derivative_p = (model.departure_gibbs(T, P+dp, x, phase)
                    - model.departure_gibbs(T, P-dp, x, phase))/(2*dp)
    volume_dep = model.molar_volume(T, P, x, phase) - (R/100)*T/P
    assert derivative_p == pytest.approx(100*volume_dep, rel=2e-7, abs=1e-5)
    phi = model.fugacity_coefficients(T, P, x, phase)
    assert sum(x[c]*math.log(phi[c]) for c in x) == pytest.approx(g/(R*T), abs=1e-12)
    xp, xm = {a: .3+dx, b: .7-dx}, {a: .3-dx, b: .7+dx}
    dgdx = (model.departure_gibbs(T, P, xp, phase)-model.departure_gibbs(T, P, xm, phase))/(2*dx)
    assert R*T*math.log(phi[a]/phi[b]) == pytest.approx(dgdx, rel=2e-7, abs=1e-5)
    phip, phim = model.fugacity_coefficients(T, P, xp, phase), model.fugacity_coefficients(T, P, xm, phase)
    gibbs_duhem = sum(x[c]*(math.log(phip[c])-math.log(phim[c]))/(2*dx) for c in x)
    assert gibbs_duhem == pytest.approx(0, abs=2e-7)


def test_low_pressure_limit_zero_fraction_and_volume_units(binary):
    x = {"CH4": 1., "C2H6": 0.}
    T, P = 300., 1e-7
    assert binary.compressibility_factor(T, P, x) == pytest.approx(1, abs=1e-8)
    assert binary.molar_volume(T, P, x) == pytest.approx((R/100)*T/P, rel=1e-8)
    assert binary.departure_enthalpy(T, P, x) == pytest.approx(0, abs=1e-4)
    assert list(binary.fugacity_coefficients(T, P, x).values()) == pytest.approx([1., 1.], abs=1e-8)
    assert binary.compressibility_factor(300, 10, {"CH4": 2}) == binary.compressibility_factor(300, 10, x)


def test_property_normalization_handles_large_finite_component_amounts(binary):
    fractions = {"CH4": .5, "C2H6": .5}
    amounts = {"CH4": 1e308, "C2H6": 1e308}
    assert binary.fugacity_coefficients(300., 10., amounts) == pytest.approx(
        binary.fugacity_coefficients(300., 10., fractions), rel=1e-12)
    assert binary.departure_enthalpy(300., 10., amounts) == pytest.approx(
        binary.departure_enthalpy(300., 10., fractions), rel=1e-12)


def test_tp_flash_closes_mass_and_fugacity_balances(binary):
    overall = {"CH4": .5, "C2H6": .5}
    beta, liquid, vapor = binary.flash_TP(overall, 200., 10.)
    assert 0 < beta < 1
    pl = binary.fugacity_coefficients(200., 10., liquid, "liquid")
    pv = binary.fugacity_coefficients(200., 10., vapor, "vapor")
    for c in overall:
        assert (1-beta)*liquid[c]+beta*vapor[c] == pytest.approx(overall[c], abs=1e-9)
        assert liquid[c]*pl[c] == pytest.approx(vapor[c]*pv[c], abs=2e-8)


@pytest.mark.parametrize('T,P,methane_fraction', [
    tuple(map(float, row)) for row in np.random.default_rng(4182).uniform(
        [150., 1., .05], [290., 65., .95], size=(80, 3))
])
def test_seeded_flash_range_preserves_equilibrium_and_material_balance(binary, T, P, methane_fraction):
    z = {'CH4': methane_fraction, 'C2H6': 1 - methane_fraction}
    beta, x, y = binary.flash_TP(z, T, P)
    assert 0.0 <= beta <= 1.0
    for c in z:
        assert (1-beta)*x[c] + beta*y[c] == pytest.approx(z[c], abs=1e-9)
    if 0.0 < beta < 1.0:
        pl = binary.fugacity_coefficients(T, P, x, 'liquid')
        pv = binary.fugacity_coefficients(T, P, y, 'vapor')
        assert max(abs(math.log(x[c]*pl[c]/(y[c]*pv[c]))) for c in z) < 1e-8


@pytest.mark.parametrize("T,P,methane_fraction,phase", [
    (209., 56.4, .79, "liquid"),
    (267.4, 37.25, .405, "vapor"),
    (245.65, 50.72, .664, "two_phase"),
])
def test_wilson_seed_does_not_determine_final_flash(binary, T, P, methane_fraction, phase):
    overall = {"CH4": methane_fraction, "C2H6": 1 - methane_fraction}
    beta, liquid, vapor = binary.flash_TP(overall, T, P)
    if phase == "liquid":
        assert beta == 0.0
    elif phase == "vapor":
        assert beta == 1.0
    else:
        assert 0.0 < beta < 1.0
        pl = binary.fugacity_coefficients(T, P, liquid, "liquid")
        pv = binary.fugacity_coefficients(T, P, vapor, "vapor")
        assert max(abs(math.log(liquid[c] * pl[c] / (vapor[c] * pv[c])))
                   for c in overall) < 1e-8
    for c in overall:
        assert (1 - beta) * liquid[c] + beta * vapor[c] == pytest.approx(overall[c], abs=1e-9)
    state = binary.calculate_state(T, P, 1., overall, include=())
    assert state.vapor_fraction == pytest.approx(beta, abs=1e-9)
    assert state.phase_status != "fallback_vle"


def test_unconverged_flash_is_not_replaced_by_a_guessed_state(binary, monkeypatch):
    overall = {"CH4": .664, "C2H6": .336}
    # A previous converged result must not bypass a different iteration budget.
    binary.flash_TP(overall, 245.65, 50.72)
    # One iteration cannot establish equilibrium for this split. Exercise the
    # real iteration budget and then the ordinary calculate_state error path.
    with pytest.raises(ThermodynamicsValidityError, match="did not converge"):
        binary._iterative_K_flash_TP(overall, 245.65, 50.72, max_iter=1)
    flash = binary._iterative_K_flash_TP
    monkeypatch.setattr(binary, "flash_TP", lambda z, T, P: flash(z, T, P, max_iter=1))
    with pytest.raises(ThermodynamicsValidityError, match="did not converge"):
        binary.calculate_state(245.65, 50.72, 1., overall, include=())


@pytest.mark.parametrize("omega,phase,Pr", [
    (.5373675653066226, "vapor", .8),
    (-.2, "vapor", .75),
])
def test_acentric_extrapolation_rejects_negative_isothermal_compressibility(omega, phase, Pr):
    model = acentric_model(omega)
    x = {"test": 1.}
    T = .95 * model.props["test"].Tc
    mixture = model._mixing(T, np.array([1.]))
    P = Pr * mixture.Pc
    # Independently differentiate the unguarded corresponding-states volumes
    # to verify this exercises a positive dV/dP, rather than an invalid Z.
    def unguarded_volume(p):
        simple = _fluid_state(.95, p / mixture.Pc, 0, phase)[0]
        reference = _fluid_state(.95, p / mixture.Pc, 1, phase)[0]
        return (simple + omega / .3978 * (reference - simple)) * R / 100 * T / p
    assert unguarded_volume(P) > 0
    assert unguarded_volume(P + 1e-5) > unguarded_volume(P - 1e-5)
    for operation in (
        lambda: model.molar_volume(T, P, x, phase),
        lambda: model.fugacity_coefficients(T, P, x, phase),
        lambda: model.departure_gibbs(T, P, x, phase),
        lambda: model.departure_enthalpy(T, P, x, phase),
        lambda: model.mixture_Cp(x, T, float(phase == "vapor"), P),
        lambda: model.calculate_state(T, P, 1., x, phase=phase, include=("H",)),
    ):
        with pytest.raises(ThermodynamicsValidityError, match="mechanically unstable"):
            operation()
    # The explicitly requested unstable vapor is rejected, while an ordinary
    # flash can select the other, mechanically stable liquid branch.
    state = model.calculate_state(T, P, 1., x, include=())
    assert state.phase_status == 'single_liquid'


def test_invalid_vapor_trial_does_not_abort_valid_decane_liquid():
    model = create_thermodynamics(['n-decane'], 'LKP', ChemicalDatabase(enable_online=False))
    z = {'n-decane': 1.}
    T = .95 * model.props['n-decane'].Tc
    P = .8 * model._mixing(T, np.array([1.])).Pc
    dp = 1e-4
    assert model.molar_volume(T, P + dp, z, 'liquid') < model.molar_volume(T, P - dp, z, 'liquid')
    with pytest.raises(ThermodynamicsValidityError, match='mechanically unstable'):
        model.molar_volume(T, P, z, 'vapor')
    state = model.calculate_state(T, P, 1., z, include=('H',))
    assert state.phase_status == 'single_liquid'
    assert state.H == pytest.approx(model.mixture_enthalpy(z, T, 0., P=P))


@pytest.mark.parametrize("omega", [-.2, 0., .3978, .5373675653066226])
@pytest.mark.parametrize("phase", ["liquid", "vapor"])
def test_stable_acentric_extrapolation_preserves_volume_and_gibbs_derivative(omega, phase):
    model = acentric_model(omega)
    x = {"test": 1.}
    T = .65 * model.props["test"].Tc
    P = .01 * model._mixing(T, np.array([1.])).Pc
    dp = 1e-5
    v = model.molar_volume(T, P, x, phase)
    assert v > 0
    assert model.molar_volume(T, P + dp, x, phase) < model.molar_volume(T, P - dp, x, phase)
    dg_dp = (model.departure_gibbs(T, P + dp, x, phase)
             - model.departure_gibbs(T, P - dp, x, phase)) / (2 * dp)
    assert dg_dp == pytest.approx(100 * (v - R / 100 * T / P), rel=2e-7, abs=1e-5)


def test_bubble_and_dew_temperatures_close_phase_boundary_residuals(binary):
    x = {"CH4": .5, "C2H6": .5}
    bubble = binary.bubble_point_T(x, 10., 190.)
    dew = binary.dew_point_T(x, 10., 230.)
    assert bubble < dew
    assert sum(x[c]*k for c, k in binary.K_values(bubble, 10., x).items()) == pytest.approx(1, abs=1e-6)
    # The dew solver iterates the incipient liquid composition internally.
    beta, _, _ = binary.flash_TP(x, dew * (1 + 1e-6), 10.)
    assert beta > .999


@pytest.mark.parametrize("phase", ["liquid", "vapor"])
def test_heat_capacity_is_total_enthalpy_derivative(binary, phase):
    x = {"CH4": .5, "C2H6": .5}
    T, P, dt = 200., 10., .05
    vf = float(phase == "vapor")
    cp = binary.mixture_Cp(x, T, vf, P)
    derivative = (binary.mixture_enthalpy(x, T+dt, vf, P=P)
                  - binary.mixture_enthalpy(x, T-dt, vf, P=P))/(2*dt)
    assert cp == pytest.approx(derivative, rel=2e-6)


@pytest.mark.parametrize("T,P,x,phase", [
    (0., 1., {"CH4": 1}, "vapor"), (300., -1., {"CH4": 1}, "vapor"),
    (300., 1., {"CH4": -1}, "vapor"), (300., 1., {}, "vapor"),
    (300., 1., {"CH4": math.nan}, "vapor"), (300., 1., {"unknown": 1}, "vapor"),
    (300., 1., {"CH4": 1}, "solid"),
])
def test_invalid_states_fail_descriptively(binary, T, P, x, phase):
    with pytest.raises(ThermodynamicsError, match="LKP"):
        binary.compressibility_factor(T, P, x, phase)


PFD = """PROCESS: LKP integration
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: LKP
COMPONENTS:
    CH4 | Methane | Tc=190.56, Pc=45.99, Vc=98.6, omega=0.011
    C2H6 | Ethane
THERMO_SCOPES:
    inherited | method=LKP, inherit=global
    local | method=LKP
INTERACTION_PARAMETERS:
    CH4/C2H6 | model=LKP, kij=1.06
    C2H6/CH4 | model=LKP, kij=1.07, scope=local
STREAM Feed : FEED -> PRODUCT
    T = 300 [K]
    P = 10 [bar]
    F = 1 [kmol/h]
    x = CH4:0.5, C2H6:0.5
"""


def test_pfd_simulator_scopes_reporting_and_property_overrides():
    pfd = parse_pfd(PFD)
    assert parse_pfd(pfd.to_pfd()).metadata.thermo_method == "LKP"
    assert parse_pfd(PFD.replace("THERMO_METHOD: LKP", "THERMO_METHOD: LEE-KESLER-PLOCKER")).metadata.thermo_method == "LKP"
    sim = Simulator.from_string(PFD)
    result = sim.run()
    assert result.converged
    assert isinstance(sim.thermo, LKPThermodynamics)
    assert sim.thermo.props["CH4"].Vc == 98.6
    assert sim.thermo.kij[0, 1] == 1.06
    assert sim.thermo_packages["inherited"].kij[0, 1] == 1.06
    assert sim.thermo_packages["local"].kij[0, 1] == 1.07
    assert "THERMO_METHOD: LKP" in sim._generate_pfr()


@pytest.mark.parametrize("parameter", ["kij=0", "kij=-1", "kij=1, k_ij=1", "comment=missing"])
def test_invalid_pfd_interactions_fail(parameter):
    text = PFD.replace("kij=1.06", parameter)
    with pytest.raises((SimulationError, ThermodynamicsError), match="LKP"):
        Simulator.from_string(text).initialize(property_methods=[])


def test_unknown_lkp_parameter_law_is_rejected():
    with pytest.raises(ParseError):
        parse_pfd(PFD.replace("kij=1.06", "kij_a=1.06"))


def test_web_configuration_offers_lkp():
    pytest.importorskip("flask")
    import app as web
    with web.app.test_client() as client:
        config = client.get("/api/config").get_json()
    assert "LKP" in config["thermo_methods"]
    assert "LKP" in config["chart_methods"]
