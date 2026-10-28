"""Forward-model consistency of homogeneous and wet IF97 inverse states."""

import math

import pytest

from chemical_properties import ChemicalDatabase
from thermodynamics import ThermodynamicsError, create_thermodynamics


@pytest.fixture
def steam():
    return create_thermodynamics(['water'], 'STEAM', ChemicalDatabase(enable_online=False))


@pytest.mark.parametrize('P,T', [(3., 300.), (3., 360.), (.1, 500.), (86., 773.15),
                                 (250., 650.), (300., 800.)])
@pytest.mark.parametrize('quantity', ['H', 'S'])
def test_inverse_state_agrees_with_its_forward_model(steam, P, T, quantity):
    z = {'water': 1.}
    reference = steam.calculate_state(T, P, 2., z, include=('H', 'S'))
    target = getattr(reference, quantity)
    inverse = getattr(steam, 'calculate_state_P'+quantity)(P, target, 2., z, include=('H', 'S', 'rho', 'Cp'))
    forward = steam.calculate_state(inverse.T, inverse.P, inverse.F, z, include=('H', 'S'))
    assert inverse.T == pytest.approx(T, abs=1e-7)
    assert getattr(forward, quantity) == pytest.approx(target, abs=1e-6)
    assert inverse.H == pytest.approx(forward.H, abs=1e-7)
    assert inverse.S == pytest.approx(forward.S, abs=1e-8)
    assert inverse.rho > 0 and inverse.Cp > 0


@pytest.mark.parametrize('quantity', ['H', 'S'])
@pytest.mark.parametrize('quality', [0., 1e-8, .25, .75, 1-1e-8, 1.])
def test_inverse_wet_states_keep_saturation_and_quality(steam, quantity, quality):
    z, P = {'water': 1.}, 3.
    reference = steam.calculate_state_PQ(P, quality, 1., z, include=('H', 'S', 'rho', 'Cp'))
    actual = getattr(steam, 'calculate_state_P'+quantity)(P, getattr(reference, quantity), 1., z,
                                                       include=('H', 'S', 'rho', 'Cp'))
    assert actual.T == pytest.approx(reference.T, abs=1e-9)
    assert actual.vapor_fraction == pytest.approx(quality, abs=1e-10)
    assert actual.H == pytest.approx(reference.H, abs=1e-6)
    assert actual.S == pytest.approx(reference.S, abs=1e-8)
    assert actual.rho == pytest.approx(reference.rho, rel=1e-8)
    if 0 < quality < 1:
        assert actual.Cp is None


@pytest.mark.parametrize('quantity', ['H', 'S'])
@pytest.mark.parametrize('side', [-1., 1.])
def test_homogeneous_inverse_is_consistent_near_saturation(steam, quantity, side):
    P, z = 3., {'water': 1.}
    # Stay outside IF97's PT ambiguity band, which rejects points within
    # 3.3e-3 percent of saturation pressure before any inverse refinement.
    T = steam.bubble_point_T(z, P)+side*.02
    reference = steam.calculate_state(T, P, 1., z, include=(quantity,))
    actual = getattr(steam, 'calculate_state_P'+quantity)(P, getattr(reference, quantity), 1., z,
                                                       include=('H', 'S'))
    forward = steam.calculate_state(actual.T, P, 1., z, include=(quantity,))
    assert getattr(forward, quantity) == pytest.approx(getattr(reference, quantity), abs=1e-6)
    assert actual.T == pytest.approx(T, abs=1e-7)


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_nonfinite_water_composition_cannot_be_silently_normalized(steam, bad):
    z = {'water': bad}
    with pytest.raises(ThermodynamicsError, match='finite'):
        steam.calculate_state(300., 3., 1., z)
    with pytest.raises(ThermodynamicsError, match='finite'):
        steam.calculate_state_PH(3., -280000., 1., z)
    with pytest.raises(ThermodynamicsError, match='finite'):
        steam.prepare_wall_prandtl(z, 3., 'liquid')


def test_unverifiable_homogeneous_inverse_explains_saturation_alternative(steam):
    z, P = {'water': 1.}, 3.
    liquid = steam.calculate_state_PQ(P, 0., 1., z, include=('H', 'Cp'))
    # This subcooling lies inside the backend's T/P ambiguity band. Returning
    # its unchecked approximate backward temperature would hide that failure.
    target = liquid.H-liquid.Cp*1e-4
    with pytest.raises(ThermodynamicsError, match='forward-consistent.*vapor quality'):
        steam.calculate_state_PH(P, target, 1., z, include=('H',))


@pytest.mark.parametrize('quantity', ['H', 'S'])
@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_nonfinite_inverse_target_cannot_become_a_saturated_state(steam, quantity, bad):
    with pytest.raises(ThermodynamicsError, match='finite'):
        getattr(steam, 'calculate_state_P'+quantity)(3., bad, 1., {'water': 1.})


@pytest.mark.parametrize('quantity', ['H', 'S'])
@pytest.mark.parametrize('pressure', [math.nan, math.inf, -math.inf, 0., -1.])
def test_inverse_pressure_must_be_positive_and_finite(steam, quantity, pressure):
    reference = steam.calculate_state(500., 3., 1., {'water': 1.}, include=(quantity,))
    with pytest.raises(ThermodynamicsError, match='finite.*positive pressure'):
        getattr(steam, 'calculate_state_P'+quantity)(
            pressure, getattr(reference, quantity), 1., {'water': 1.})


@pytest.mark.parametrize('quantity', ['H', 'S'])
@pytest.mark.parametrize('P,T', [(1., 1100.), (300., 1500.), (500., 2000.),
                                 (500., 2273.15), (500., 1073.14), (500., 1073.16),
                                 (501., 1073.15)])
def test_inverse_covers_the_pressure_dependent_if97_temperature_domain(steam, quantity, P, T):
    z = {'water': 1.}
    reference = steam.calculate_state(T, P, 1., z, include=('H', 'S'))
    actual = getattr(steam, 'calculate_state_P'+quantity)(
        P, getattr(reference, quantity), 1., z, include=('H', 'S'))
    assert actual.T == pytest.approx(T, abs=1e-7)
    assert actual.H == pytest.approx(reference.H, abs=1e-6)
    assert actual.S == pytest.approx(reference.S, abs=1e-8)
    if quantity == 'H':
        temperature, residual = steam.temperature_at_PH(P, reference.H, z, phase='vapor')
        assert temperature == pytest.approx(T, abs=1e-7)
        assert abs(residual) < 1e-6


@pytest.mark.parametrize('quantity', ['H', 'S'])
def test_inverse_does_not_extend_region_five_above_its_pressure_limit(steam, quantity):
    z = {'water': 1.}
    reference = steam.calculate_state(1500., 500., 1., z, include=(quantity,))
    with pytest.raises(ThermodynamicsError):
        getattr(steam, 'calculate_state_P'+quantity)(
            501., getattr(reference, quantity), 1., z, include=(quantity,))
