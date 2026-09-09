"""Liquid conductivity mixing and resolver integration."""

from types import SimpleNamespace

import pytest

from chemical_properties import ChemicalDatabase
from thermodynamics_models.base import IdealThermodynamics
from thermodynamics_models.common import ThermodynamicsError


@pytest.fixture
def thermo():
    return IdealThermodynamics(['water', 'ethanol'], ChemicalDatabase(enable_online=False))


def test_li_mixing_has_pure_limits_symmetry_and_bounds(thermo, monkeypatch):
    values = {'water': 0.6, 'ethanol': 0.2}
    monkeypatch.setattr(thermo, 'pure_thermal_conductivity', lambda c, T, phase: values[c])
    monkeypatch.setattr(thermo, 'mixture_liquid_molar_volume',
                        lambda composition, T: 0.02 if 'water' in composition else 0.06)
    assert thermo.mixture_liquid_thermal_conductivity({'water': 1}, 300) == 0.6
    value = thermo.mixture_liquid_thermal_conductivity({'water': 0.75, 'ethanol': 0.25}, 300)
    # Equal volume fractions, unlike the unequal mole fractions.
    assert value == pytest.approx(0.25 * 0.6 + 0.25 * 0.2 + 0.5 * 0.3)
    assert 0.2 < value < 0.6
    assert value == thermo.mixture_liquid_thermal_conductivity({'ethanol': 0.25, 'water': 0.75}, 300)


def test_pure_conductivity_preserves_phase_and_caches(thermo, monkeypatch):
    import property_resolver
    calls = []
    def resolve(identifier, T, *, phase, props):
        calls.append(phase)
        return SimpleNamespace(value=2.2 if phase == 'solid' else 0.6)
    monkeypatch.setattr(property_resolver, 'get_property_resolver',
                        lambda: SimpleNamespace(resolve_thermal_conductivity=resolve))
    monkeypatch.setattr(thermo, '_record_lazy_property_source', lambda *args: None)
    assert thermo.pure_thermal_conductivity('water', 250, 'solid') == 2.2
    assert thermo.pure_thermal_conductivity('water', 250, 'solid') == 2.2
    assert thermo.pure_thermal_conductivity('water', 250, 'liquid') == 0.6
    assert calls == ['solid', 'liquid']


def test_invalid_resolver_conductivity_fails(thermo, monkeypatch):
    import property_resolver
    monkeypatch.setattr(property_resolver, 'get_property_resolver', lambda: SimpleNamespace(
        resolve_thermal_conductivity=lambda *args, **kwargs: SimpleNamespace(value=float('nan')),
    ))
    with pytest.raises(ThermodynamicsError, match='positive and finite'):
        thermo.pure_thermal_conductivity('water', 250, 'solid')
