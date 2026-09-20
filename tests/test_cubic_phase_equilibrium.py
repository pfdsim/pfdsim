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
