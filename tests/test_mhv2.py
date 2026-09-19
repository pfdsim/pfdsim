"""MHV2 thermodynamic identities and end-to-end model integration."""

import math

import numpy as np
import pytest

from cubic_eos import CubicEOS, R_CM3
from chemical_properties import ChemicalDatabase
from lyngby_parameters import parameter_table
from thermodynamics import create_thermodynamics
from thermodynamics_models.ge_eos import ExcessGibbsState, ModifiedHuronVidalSecondOrderMixingRule


@pytest.fixture(scope='module')
def model():
    return CubicEOS(['ethanol', 'water'], 'RKSMHV2', ChemicalDatabase(enable_online=False))


def test_mhv2_quadratic_and_partial_molar_derivatives():
    rule = ModifiedHuronVidalSecondOrderMixingRule()
    x, b, d, dd = (.3, .7), (30., 60.), (7., 14.), (-.02, -.03)
    excess = ExcessGibbsState((0., 0.), (0., 0.))
    state = rule.mix(x, b, d, dd, excess)
    rhs = sum(xi*math.log(state.b/bi) for xi, bi in zip(x, b))
    lhs = rule.q1*(state.D-np.dot(x, d)) + rule.q2*(state.D**2-np.dot(x, np.square(d)))
    assert lhs == pytest.approx(rhs, abs=2e-14)
    assert np.dot(x, state.composition_derivatives) == pytest.approx(state.D)
    for i in range(2):
        h = 1e-6
        n = np.array(x)
        n[i] += h
        plus = n.sum()*rule.mix(tuple(n/n.sum()), b, d, dd, excess).D
        n[i] -= 2*h
        minus = n.sum()*rule.mix(tuple(n/n.sum()), b, d, dd, excess).D
        assert (plus-minus)/(2*h) == pytest.approx(state.composition_derivatives[i], rel=1e-8)


@pytest.mark.parametrize('phase', ['liquid', 'vapor'])
def test_departure_and_fugacity_identities(model, phase):
    T, P = 350., 1.
    x = {'ethanol': .4, 'water': .6}
    phi = model.fugacity_coefficients(T, P, x, phase)
    g = model.departure_gibbs(T, P, x, phase)
    assert g == pytest.approx(.1*R_CM3*T*sum(x[c]*math.log(phi[c]) for c in x), rel=1e-9)
    h = .01
    gp = model.departure_gibbs(T+h, P, x, phase)/(T+h)
    gm = model.departure_gibbs(T-h, P, x, phase)/(T-h)
    assert model.departure_enthalpy(T, P, x, phase) == pytest.approx(-T*T*(gp-gm)/(2*h), rel=1e-7)
    for c in x:
        step = 1e-5
        xp = dict(x)
        xp[c] += step
        xm = dict(x)
        xm[c] -= step
        partial = ((1+step)*model.departure_gibbs(T, P, xp, phase)
                   -(1-step)*model.departure_gibbs(T, P, xm, phase))/(2*step)
        assert partial == pytest.approx(.1*R_CM3*T*math.log(phi[c]), abs=2e-5)


def test_pure_limit_and_alpha(model):
    for T in (300., 700.):
        for c in model.components:
            x = {comp: float(comp == c) for comp in model.components}
            mixed = model.mixture_params(T, x)
            assert mixed[0] == pytest.approx(model.pure_a(c, T), rel=1e-12)
            step = T*1e-5
            derivative = (model.pure_a(c, T+step)-model.pure_a(c, T-step))/(2*step)
            assert model.pure_da_dT(c, T) == pytest.approx(derivative, rel=1e-8)
    assert model.params['ethanol'].mc_c1 == 1.4252


def test_gas_extension_and_alcohol_footnote():
    model = CubicEOS(['H2', 'N2', 'ethanol', 'hexane'], 'RKSMHV2', ChemicalDatabase(enable_online=False))
    provider = model._ge_provider
    assert provider.component_groups['ethanol'] == {59: 1, 60: 1, 12: 1}
    assert provider.component_groups['hexane'] == {1: 2, 2: 4}
    assert provider.unifac.interaction_coefficients[(35, 22)] == (986., -2.133, 0.)
    assert provider.unifac.interaction_coefficients[(22, 24)] == (0., 0., 0.)
    assert (6, 29) not in provider.unifac.interactions
    table = parameter_table(True)
    assert len(table['gas_components']) == 13
    assert len(table['alpha']) == 51


def test_integrated_flash_material_and_fugacity_closure():
    model = create_thermodynamics(['ethanol', 'water'], 'RKSMHV2')
    z = {'ethanol': .5, 'water': .5}
    T, P = 355., 1.
    vf, x, y = model.flash_TP(z, T, P)
    assert 0.0 < vf < 1.0
    pl = model.fugacity_coefficients(T, P, x, 'liquid')
    pv = model.fugacity_coefficients(T, P, y, 'vapor')
    for c in z:
        assert (1-vf)*x[c]+vf*y[c] == pytest.approx(z[c], abs=1e-8)
        assert math.log(x[c]*pl[c]/(y[c]*pv[c])) == pytest.approx(0., abs=1e-7)


def test_parser_simulator_override_and_report():
    from simulator import Simulator
    source = '''PROCESS: MHV2 override
ONLINE_LOOKUP: false
THERMO_METHOD: RKS-MHV2
COMPONENTS:
    Gas | Methane | Tc=200, Pc=50, omega=0.2, mc_c1=0.8, mc_c2=-0.1
STREAM Feed : FEED -> PRODUCT
    T = 180 [K]
    P = 5 [bar]
    F = 1 [kmol/h]
    x = Gas:1
'''
    simulator = Simulator.from_string(source)
    result = simulator.run()
    assert result.converged
    params = simulator.thermo.cubic.params['Gas']
    assert params.Tc == 200.
    assert params.mc_c1 == .8
    assert params.mc_c2 == -.1
    assert params.mc_c3 == 0.
    assert 'THERMO_METHOD: RKSMHV2' in simulator._generate_pfr()
    simulator.initialize(thermo_method='SRK_MHV2')
    assert simulator.thermo_method == 'RKSMHV2'
    assert simulator.thermo.cubic.params['Gas'].mc_c1 == .8


def test_missing_gas_parameters_are_not_assumed_zero():
    from cubic_eos import CubicEOSError
    with pytest.raises(CubicEOSError, match='unavailable'):
        CubicEOS(['acetylene', 'water'], 'RKSMHV2', ChemicalDatabase(enable_online=False))


def test_builder_reproduces_runtime_tables():
    from scripts.build_lyngby_parameters import build
    assert build() == parameter_table()
    assert build(gas_extension=True) == parameter_table(True)


def test_executable_flash_example():
    from pathlib import Path
    from simulator import Simulator
    path = Path(__file__).resolve().parents[1] / 'examples' / 'ethanol_water_mhv2.pfd'
    result = Simulator.from_file(str(path)).run()
    assert result.converged
    assert result.mass_balance_error < 1e-6
    assert result.streams['Vapor'].F > 0.
    assert result.streams['Liquid'].F > 0.
