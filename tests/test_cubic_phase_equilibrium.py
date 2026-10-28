"""Shared phi–phi regressions for cubic and excess-Gibbs EOS methods."""

import math

import pytest

from chemical_properties import ChemicalDatabase
from thermodynamics import create_thermodynamics


@pytest.mark.parametrize('method', ['PR', 'SRK', 'PSRK', 'RKSMHV2'])
def test_single_root_gas_liquid_fugacity_iteration(method):
    model = create_thermodynamics(['CO2', 'water'], method,
                                  ChemicalDatabase(enable_online=False))
    T, P = 350., 50.
    x = {'CO2': .03, 'water': .97}
    eos = model.psrk if method == 'PSRK' else model.cubic
    native_x = model._cas_composition(x) if method == 'PSRK' else x
    assert len(eos.compressibility_roots(T, P, native_x)) == 1
    backend = eos._compiled_backend
    for use_compiled in (True, False):
        eos._compiled_backend = backend if use_compiled else None
        eos._phi_phi_k_cache.clear()
        K = model.K_values(T, P, x)
        total = sum(x[c]*K[c] for c in x)
        y = {c: x[c]*K[c]/total for c in x}
        pl = model.fugacity_coefficients(T, P, x, 'liquid')
        pv = model.fugacity_coefficients(T, P, y, 'vapor')
        assert max(abs(math.log(K[c]*pv[c]/pl[c])) for c in x) < 1e-7
    eos._compiled_backend = backend


@pytest.mark.parametrize('method', ['PR', 'PSRK', 'RKSMHV2'])
def test_identical_single_phase_roots_do_not_report_saturation(method):
    model = create_thermodynamics(['water'], method, ChemicalDatabase(enable_online=False))
    assert abs(model.K_values(800., 1., {'water': 1.})['water'] - 1.) > .1


def test_compiled_psrk_fugacity_underflow_falls_back_without_dividing_by_zero():
    model = create_thermodynamics(
        ['CO2', 'water'], 'PSRK', ChemicalDatabase(enable_online=False)
    )
    if model.psrk._compiled_backend is None:
        pytest.skip('Numba compiled PSRK backend is unavailable')
    values = model.K_values(9.166, 1.2, {'CO2': .1, 'water': .9})
    assert all(math.isfinite(value) and 1e-8 <= value <= 1e8
               for value in values.values())


@pytest.mark.parametrize('method', ['PR', 'PSRK', 'RKSMHV2'])
@pytest.mark.parametrize('gas_fraction', [.03, .5, .9])
def test_high_pressure_flash_closes_material_and_fugacity_balances(method, gas_fraction):
    model = create_thermodynamics(['CO2', 'water'], method,
                                  ChemicalDatabase(enable_online=False))
    z = {'CO2': gas_fraction, 'water': 1-gas_fraction}
    vf, x, y = model.flash_TP(z, 350., 50.)
    assert 0. < vf < 1.
    pl = model.fugacity_coefficients(350., 50., x, 'liquid')
    pv = model.fugacity_coefficients(350., 50., y, 'vapor')
    for c in z:
        assert (1-vf)*x[c] + vf*y[c] == pytest.approx(z[c], abs=1e-8)
        assert abs(math.log(x[c]*pl[c]/(y[c]*pv[c]))) < 1e-7


@pytest.mark.parametrize('method', ['RK', 'PR', 'SRK', 'PSRK', 'RKSMHV2', 'LKP'])
def test_wilson_initialized_flash_checks_actual_phase_fugacities(method):
    model = create_thermodynamics(['CH4', 'C2H6'], method,
                                  ChemicalDatabase(enable_online=False))
    T, P = 245.65, 50.72
    z = {'CH4': .664, 'C2H6': .336}
    vf, x, y = model.flash_TP(z, T, P)
    assert 0. < vf < 1.
    pl = model.fugacity_coefficients(T, P, x, 'liquid')
    pv = model.fugacity_coefficients(T, P, y, 'vapor')
    for c in z:
        assert (1 - vf) * x[c] + vf * y[c] == pytest.approx(z[c], abs=1e-9)
        assert abs(math.log(x[c] * pl[c] / (y[c] * pv[c]))) < 1e-8


def test_cubic_branch_identity_distinguishes_compressed_supercritical_fluid():
    thermo = create_thermodynamics(['CH4'], 'PR', ChemicalDatabase(enable_online=False))
    z = {'CH4': 1.0}
    # The 191 K state has low Z but is above the mechanical critical
    # temperature. A Z threshold would incorrectly reject its vapor branch.
    assert thermo.compressibility_factor(191., 50., z) < 0.3
    assert thermo._eos_root_kind(191., 50., z, 'vapor') == 'supercritical'
    assert thermo._eos_root_kind(190., 50., z, 'vapor') == 'liquid'
    assert thermo._eos_root_kind(150., 1., z, 'liquid') == 'liquid'
    assert thermo._eos_root_kind(150., 1., z, 'vapor') == 'vapor'


@pytest.mark.parametrize('method', ['PR', 'PSRK', 'LKP'])
def test_short_substitution_budget_still_establishes_equilibrium(method):
    thermo = create_thermodynamics(['CH4', 'C2H6'], method,
                                  ChemicalDatabase(enable_online=False))
    z = {'CH4': .664, 'C2H6': .336}
    T, P = 245.65, 50.72
    beta, x, y = thermo._iterative_K_flash_TP(z, T, P, max_iter=8)
    assert 0.0 < beta < 1.0
    liquid = thermo.fugacity_coefficients(T, P, x, 'liquid')
    vapor = thermo.fugacity_coefficients(T, P, y, 'vapor')
    for c in z:
        assert (1-beta)*x[c] + beta*y[c] == pytest.approx(z[c], abs=1e-9)
        assert abs(math.log(x[c]*liquid[c]/(y[c]*vapor[c]))) < 1e-8


@pytest.mark.parametrize('method', ['RK', 'PR', 'SRK', 'PSRK', 'RKSMHV2', 'LKP'])
def test_trace_component_does_not_break_constrained_single_liquid_flash(method):
    thermo = create_thermodynamics(['CH4', 'C2H6'], method,
                                  ChemicalDatabase(enable_online=False))
    z = {'CH4': 1e-300, 'C2H6': 1.0}
    beta, x, y = thermo.flash_TP(z, 150., 10.)
    assert beta == 0.0
    assert x['C2H6'] == 1.0
    assert x['CH4'] == z['CH4']
    assert y == x


@pytest.mark.parametrize('method', ['RK', 'PR', 'SRK', 'PSRK', 'RKSMHV2', 'LKP'])
def test_single_root_gas_uses_eos_volume_in_stream_density(method):
    model = create_thermodynamics(['CH4', 'ethanol'], method,
                                 ChemicalDatabase(enable_online=False))
    z = {'CH4': .5, 'ethanol': .5}
    T, P = 500., 10.
    assert model._eos_root_kind(T, P, z, 'vapor') == 'supercritical'
    state = model.calculate_state(T, P, 1., z, include=('rho',))
    assert state.phase_status == 'single_vapor'
    assert state.vapor_fraction == 1.
    assert state.rho == pytest.approx(1 / model.molar_volume(T, P, z, 'vapor'), rel=1e-12)


@pytest.mark.parametrize('method', ['PR', 'PSRK', 'LKP'])
@pytest.mark.parametrize('seed_scale', [1e-4, 1., 1e4])
def test_homogeneous_single_root_phase_is_independent_of_wilson_seed(method, seed_scale, monkeypatch):
    import phi_phi
    model = create_thermodynamics(['CH4', 'C2H6'], method,
                                 ChemicalDatabase(enable_online=False))
    original = phi_phi.wilson_K
    monkeypatch.setattr(phi_phi, 'wilson_K', lambda props, T, P, components: {
        c: value * seed_scale for c, value in original(props, T, P, components).items()
    })
    z = {'CH4': .5, 'C2H6': .5}
    beta, x, y = model.flash_TP(z, 300., 1.)
    assert beta == 1.
    assert x == y == z


def test_compressed_supercritical_single_root_keeps_eos_density():
    model = create_thermodynamics(['CH4'], 'PR', ChemicalDatabase(enable_online=False))
    z = {'CH4': 1.}
    assert model.compressibility_factor(191., 50., z) < .3
    state = model.calculate_state(191., 50., 1., z, include=('rho',))
    assert state.vapor_fraction == 1.
    assert state.rho == pytest.approx(1 / model.molar_volume(191., 50., z), rel=1e-12)


def test_single_root_identity_does_not_depend_on_gibbs_roundoff(monkeypatch):
    model = create_thermodynamics(['CH4', 'ethanol'], 'PR', ChemicalDatabase(enable_online=False))
    original = model.fugacity_coefficients
    def rounded(T, P, composition, phase='vapor'):
        values = original(T, P, composition, phase)
        return {c: value*math.exp(-1e-13 if phase == 'liquid' else 0.) for c, value in values.items()}
    monkeypatch.setattr(model, 'fugacity_coefficients', rounded)
    assert model.flash_TP({'CH4': .5, 'ethanol': .5}, 500., 10.)[0] == 1.


def test_invalid_feed_roots_cannot_be_replaced_by_a_guessed_stream(monkeypatch):
    from thermodynamics import ThermodynamicsError, ThermodynamicsValidityError
    model = create_thermodynamics(['CH4', 'C2H6'], 'PR', ChemicalDatabase(enable_online=False))
    def invalid(*args):
        raise ThermodynamicsError('No physically valid EOS root')
    monkeypatch.setattr(model, '_eos_root_kind', invalid)
    with pytest.raises(ThermodynamicsValidityError, match='no valid feed root'):
        model.calculate_state(200., 10., 1., {'CH4': .5, 'C2H6': .5}, include=())


@pytest.mark.parametrize('method', ['RK', 'PR', 'SRK', 'PSRK', 'RKSMHV2', 'LKP'])
@pytest.mark.parametrize('composition', [
    {}, {'CH4': 0.}, {'unknown': 1.}, {'CH4': .5, 'unknown': .5},
    {'CH4': -.1, 'C2H6': 1.1}, {'CH4': math.nan}, {'CH4': math.inf}, {'CH4': None},
])
def test_eos_flash_rejects_invalid_feed_without_fabricating_composition(method, composition):
    from thermodynamics import ThermodynamicsValidityError
    model = create_thermodynamics(['CH4', 'C2H6'], method, ChemicalDatabase(enable_online=False))
    with pytest.raises(ThermodynamicsValidityError, match='composition'):
        model.flash_TP(composition, 300., 1.)
    for phase in (None, 'vapor', 'liquid'):
        with pytest.raises(ThermodynamicsValidityError, match='composition'):
            model.calculate_state(300., 1., 1., composition, phase=phase, include=())


@pytest.mark.parametrize('method', ['RK', 'PR', 'SRK', 'PSRK', 'RKSMHV2', 'LKP'])
def test_eos_normalization_handles_large_finite_component_amounts(method):
    model = create_thermodynamics(['CH4', 'C2H6'], method, ChemicalDatabase(enable_online=False))
    composition = {'CH4': 1e308, 'C2H6': 1e308}
    beta, x, y = model.flash_TP(composition, 300., 1.)
    assert beta == 1.
    assert x == y == {'CH4': .5, 'C2H6': .5}
    state = model.calculate_state(300., 1., 1., composition, include=())
    assert state.composition == x
    assert state.phase_status == 'single_vapor'
    assert model.bubble_point_T(composition, 10., 200.) == pytest.approx(
        model.bubble_point_T(x, 10., 200.), abs=1e-8)


@pytest.mark.parametrize('T,P', [(0., 1.), (300., 0.), (math.nan, 1.), (300., math.inf)])
def test_eos_flash_rejects_invalid_temperature_pressure(T, P):
    from thermodynamics import ThermodynamicsValidityError
    model = create_thermodynamics(['CH4', 'C2H6'], 'PR', ChemicalDatabase(enable_online=False))
    with pytest.raises(ThermodynamicsValidityError, match='positive and finite'):
        model.flash_TP({'CH4': .5, 'C2H6': .5}, T, P)


@pytest.mark.parametrize('method', ['RK', 'PR', 'PSRK', 'RKSMHV2'])
def test_backend_root_errors_cannot_fall_back_to_a_guessed_stream(method, monkeypatch):
    from cubic_eos import CubicEOSError
    from rk_eos import RKError
    from thermodynamics import PSRKCalculationError, ThermodynamicsValidityError
    model = create_thermodynamics(['CH4', 'C2H6'], method, ChemicalDatabase(enable_online=False))
    error_type = {'RK': RKError, 'PSRK': PSRKCalculationError}.get(method, CubicEOSError)
    def invalid(*args):
        raise error_type('No physically valid backend root')
    monkeypatch.setattr(model, '_eos_root_kind', invalid)
    with pytest.raises(ThermodynamicsValidityError, match='no valid feed root'):
        model.calculate_state(200., 10., 1., {'CH4': .5, 'C2H6': .5}, include=())


@pytest.mark.parametrize('kind', ['bubble', 'dew'])
def test_psrk_boundary_recovers_from_invalid_low_temperature_trial(kind):
    model = create_thermodynamics(['CH4', 'C2H6'], 'PSRK', ChemicalDatabase(enable_online=False))
    z = {'CH4': .5, 'C2H6': .5}
    boundary = getattr(model, kind+'_point_T')
    assert boundary(z, 10., 1.) == pytest.approx(boundary(z, 10., 200.), abs=1e-7)


@pytest.mark.parametrize('method', ['PR', 'PSRK', 'LKP'])
def test_repeated_flash_results_are_independent_mutable_values(method):
    model = create_thermodynamics(['CH4', 'C2H6'], method, ChemicalDatabase(enable_online=False))
    z = {'CH4': .5, 'C2H6': .5}
    beta, x, y = model.flash_TP(z, 200., 10.)
    expected_x, expected_y = dict(x), dict(y)
    x['CH4'] = 0.
    y['C2H6'] = 0.
    repeated_beta, repeated_x, repeated_y = model.flash_TP(z, 200., 10.)
    assert repeated_beta == beta
    assert repeated_x == expected_x
    assert repeated_y == expected_y
    pl = model.fugacity_coefficients(200., 10., repeated_x, 'liquid')
    pv = model.fugacity_coefficients(200., 10., repeated_y, 'vapor')
    for c in z:
        assert (1-beta)*repeated_x[c]+beta*repeated_y[c] == pytest.approx(z[c], abs=1e-9)
        assert abs(math.log(repeated_x[c]*pl[c]/(repeated_y[c]*pv[c]))) < 1e-8


@pytest.mark.parametrize('method', ['PR', 'PSRK', 'LKP'])
@pytest.mark.parametrize('kind', ['bubble', 'dew'])
def test_pure_boundary_reuse_preserves_guess_validation(method, kind):
    from thermodynamics import ThermodynamicsValidityError
    model = create_thermodynamics(['CH4'], method, ChemicalDatabase(enable_online=False))
    boundary = getattr(model, kind+'_point_T')
    z = {'CH4': 1.}
    temperature = boundary(z, 1., 150.)
    assert boundary(z, 1., 300.) == temperature
    for invalid in (-1., 0., math.nan, math.inf):
        with pytest.raises(ThermodynamicsValidityError, match='guess must be positive and finite'):
            boundary(z, 1., invalid)


@pytest.mark.parametrize('method', ['RK', 'PR', 'SRK', 'PSRK', 'RKSMHV2', 'LKP'])
@pytest.mark.parametrize('P', [10., 50.])
def test_eos_temperature_and_pressure_boundaries_establish_actual_equilibrium(method, P):
    model = create_thermodynamics(['CH4', 'C2H6'], method,
                                 ChemicalDatabase(enable_online=False))
    x = {'CH4': .5, 'C2H6': .5}
    T = model.bubble_point_T(x, P, 200.)
    K = model.K_values(T, P, x)
    y = {c: x[c]*K[c] for c in x}
    assert sum(y.values()) == pytest.approx(1., abs=1e-7)
    pl = model.fugacity_coefficients(T, P, x, 'liquid')
    pv = model.fugacity_coefficients(T, P, y, 'vapor')
    assert max(abs(math.log(x[c]*pl[c]/(y[c]*pv[c]))) for c in x) < 1e-7
    assert model.molar_volume(T, P, y, 'vapor') > model.molar_volume(T, P, x, 'liquid')
    assert model.bubble_point_P(x, T) == pytest.approx(P, rel=1e-7)
    assert model.dew_point_P(y, T) == pytest.approx(P, rel=1e-7)
    assert model.dew_point_T(y, P, T*1.01) == pytest.approx(T, abs=1e-4)


@pytest.mark.parametrize('method', ['RK', 'PR', 'SRK', 'PSRK', 'RKSMHV2', 'LKP'])
@pytest.mark.parametrize('kind', ['bubble', 'dew'])
def test_eos_boundary_rejects_supercritical_wilson_roots(method, kind):
    from thermodynamics import ThermodynamicsValidityError
    model = create_thermodynamics(['CH4', 'C2H6'], method,
                                 ChemicalDatabase(enable_online=False))
    z = {'CH4': .5, 'C2H6': .5}
    with pytest.raises(ThermodynamicsValidityError, match='physical EOS'):
        getattr(model, kind+'_point_T')(z, 100., 250.)
    with pytest.raises(ThermodynamicsValidityError, match='physical EOS'):
        getattr(model, kind+'_point_P')(z, 300.)


@pytest.mark.parametrize('method', ['RK', 'PR', 'SRK', 'PSRK', 'RKSMHV2', 'LKP'])
def test_pure_eos_boundary_resolves_narrow_near_critical_root_interval(method):
    model = create_thermodynamics(['CH4'], method, ChemicalDatabase(enable_online=False))
    z = {'CH4': 1.}
    T = .9999 * model.props['CH4'].Tc
    P = model.bubble_point_P(z, T)
    for temperature in (T, model.bubble_point_T(z, P, .99*model.props['CH4'].Tc),
                        model.dew_point_T(z, P, .99*model.props['CH4'].Tc)):
        assert abs(temperature - T) < .01
        pl = model.fugacity_coefficients(temperature, P, z, 'liquid')['CH4']
        pv = model.fugacity_coefficients(temperature, P, z, 'vapor')['CH4']
        assert abs(math.log(pl/pv)) < 1e-8
        vl = model.molar_volume(temperature, P, z, 'liquid')
        vv = model.molar_volume(temperature, P, z, 'vapor')
        assert vv - vl > 1e-6 * vv
