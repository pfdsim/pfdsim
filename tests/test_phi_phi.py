"""Numerical convergence of the shared phase-equilibrium iteration."""

import math
from types import SimpleNamespace

import pytest

import phi_phi
from thermodynamics_models.base import IdealThermodynamics
from thermodynamics_models.common import R


class ConvexRegularSolution:
    """Ideal-gas volumes plus g_res/(RT) = A*x_A*x_B, with A < 0.

    Its Gibbs curvature is 1/x_A + 1/x_B - 2*A > 0 everywhere, so the
    equilibrium is one homogeneous phase. Nevertheless its undamped TPD
    substitution has eigenvalue A/2 at the equimolar solution: for A < -2,
    full steps diverge while suitably damped steps converge.
    """

    components = ['A', 'B']
    props = {c: SimpleNamespace(Tc=100., Pc=10., omega=0.) for c in components}
    _rachford_rice_bounded = IdealThermodynamics._rachford_rice_bounded
    _flash_phase_compositions = IdealThermodynamics._flash_phase_compositions
    _set_limited_cache = staticmethod(IdealThermodynamics._set_limited_cache)
    _check_diagram_vle = IdealThermodynamics._check_diagram_vle
    component_activities = IdealThermodynamics.component_activities
    add_warning = IdealThermodynamics.add_warning

    def __init__(self, interaction):
        self.interaction = interaction
        self.warnings, self._warning_keys = [], set()

    def _eos_root_kind(self, *args):
        return 'supercritical'

    def fugacity_coefficients(self, T, P, composition, phase):
        return {c: math.exp(self.interaction*(1-composition[c])**2) for c in self.components}

    def molar_volume(self, T, P, composition, phase):
        return R*T/(100*P)


@pytest.mark.parametrize('interaction', [-4., -4.5])
def test_stable_convex_mixture_converges_despite_oscillatory_substitution(interaction):
    thermo = ConvexRegularSolution(interaction)
    z = {'A': .5, 'B': .5}
    beta, x, y = phi_phi.solve_phi_phi_flash(thermo, z, 300., 1., 100, 1e-9)
    assert beta == 1.
    assert x == y == z


def test_unavailable_numba_does_not_load_native_backend_modules(monkeypatch):
    import builtins
    original = builtins.__import__
    def import_without_native(name, *args, **kwargs):
        if name.endswith('compiled_phi_phi'):
            raise AssertionError('Native modules must not load without Numba')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(phi_phi.numerics, 'njit', None)
    monkeypatch.setattr(builtins, '__import__', import_without_native)
    thermo = ConvexRegularSolution(-4.)
    z = {'A': .5, 'B': .5}
    beta, x, y = phi_phi.solve_phi_phi_flash(thermo, z, 300., 1., 100, 1e-9)
    assert beta == 1.
    assert x == y == z


@pytest.mark.parametrize('method', ['PR', 'PSRK', 'RKSMHV2'])
def test_instability_search_retains_declared_zero_fraction_components(method):
    from chemical_properties import ChemicalDatabase
    from thermodynamics import create_thermodynamics
    thermo = create_thermodynamics(['CH4', 'C2H6', 'H2'], method, ChemicalDatabase(enable_online=False))
    z = {'CH4': .664, 'C2H6': .336, 'H2': 0.}
    T, P = 245.65, 50.72
    beta, x, y = thermo.flash_TP(z, T, P)
    assert 0. < beta < 1.
    assert x['H2'] == y['H2'] == 0.
    pl, pv = thermo.fugacity_coefficients(T, P, x, 'liquid'), thermo.fugacity_coefficients(T, P, y, 'vapor')
    for c in ('CH4', 'C2H6'):
        assert (1-beta)*x[c]+beta*y[c] == pytest.approx(z[c], abs=1e-9)
        assert abs(math.log(x[c]*pl[c]/(y[c]*pv[c]))) < 1e-8


def test_custom_fugacity_method_controls_the_equilibrium_solution(monkeypatch):
    from chemical_properties import ChemicalDatabase
    from thermodynamics import create_thermodynamics
    thermo = create_thermodynamics(['CH4', 'C2H6'], 'PR', ChemicalDatabase(enable_online=False))
    original = thermo.fugacity_coefficients
    def custom(T, P, composition, phase='vapor'):
        correction = math.exp(.05) if phase == 'vapor' else 1.
        return {c: value*correction for c, value in original(T, P, composition, phase).items()}
    monkeypatch.setattr(thermo, 'fugacity_coefficients', custom)
    z = {'CH4': .5, 'C2H6': .5}
    beta, x, y = thermo.flash_TP(z, 200., 10.)
    assert 0. < beta < 1.
    pl, pv = custom(200., 10., x, 'liquid'), custom(200., 10., y, 'vapor')
    assert max(abs(math.log(x[c]*pl[c]/(y[c]*pv[c]))) for c in z) < 1e-8


@pytest.mark.parametrize('compiled', [True, False])
def test_boundary_jacobian_includes_temperature_dependent_interactions(compiled):
    import numpy as np
    from chemical_properties import ChemicalDatabase
    from thermodynamics import create_thermodynamics
    thermo = create_thermodynamics(['CH4', 'C2H6'], 'PR', ChemicalDatabase(enable_online=False),
        interaction_overrides=[{'model': 'PR', 'component1': 'CH4', 'component2': 'C2H6',
                                'kij_a': .01, 'kij_b': 30., 'kij_c': .001}])
    if not compiled:
        thermo.cubic._compiled_backend = None
    z = np.array([.4, .6])
    values = np.array([math.log(200.), .7])
    def state(T, P, composition, phase, data):
        fractions = dict(zip(thermo.components, composition))
        phi = thermo.fugacity_coefficients(T, P, fractions, 'liquid' if phase == 0 else 'vapor')
        return 0, np.log([phi[c] for c in thermo.components])
    def equations(point):
        temperature = math.exp(point[0])
        first = 1/(1+math.exp(-point[1]))
        y = np.array([first, 1-first])
        _, pl = state(temperature, 10., z, 0, None)
        _, pv = state(temperature, 10., y, 1, None)
        return np.log(z/y)+pl-pv
    bounds = (np.array([math.log(1.), -1400.]), np.array([math.log(2000.), 1400.]))
    jac = phi_phi._boundary_jacobian(thermo, values, z, np.arange(2), 10., True, True,
                                     bounds, equations, state)
    reference = np.zeros((2, 2))
    step = 1e-5
    for column in range(2):
        direction = np.zeros(2)
        direction[column] = step
        reference[:, column] = (equations(values+direction)-equations(values-direction))/(2*step)
    assert np.allclose(jac, reference, rtol=2e-5, atol=2e-6)
