import math
from types import SimpleNamespace
from typing import ClassVar

import pytest

from equilibrium_washing import (
    contact_cell,
    equilibrate_enthalpy,
    inventory_at_temperature,
    resize_population,
    solve_warm_wash,
)
from particle_size_distributions import ParticleSizeDistribution
from thermodynamics_models.common import ThermodynamicsError


class FusionThermo:
    """Analytical pure fusion at 300 K; Hfus=10 kJ/mol, equal phase Cp."""

    components = ("A", "B")
    conventional_solid_components = ("A",)
    permanent_solid_components = ()
    props: ClassVar[dict] = {
        "A": SimpleNamespace(Tm=300.0, Hfus=10.0, MW=100.0),
        "B": SimpleNamespace(Tm=200.0, Hfus=5.0, MW=100.0),
    }

    def mark_property_source_context_once(self, *args, **kwargs):
        pass

    def _integrate_liquid_cp(self, c, t1, t2):
        return 0.1 * (t2 - t1)

    _integrate_solid_cp = _integrate_liquid_cp

    def _integrate_cp_over_T(self, c, t1, t2, phase):
        return 100 * math.log(t2 / t1)

    def _solid_molar_volume(self, c, t):
        return 0.1

    def _liquid_molar_volume_for_poynting(self, c, t):
        return 0.1, None

    def mixture_liquid_molar_volume(self, x, t):
        return 0.1

    def mixture_enthalpy(self, x, t, vapor_fraction, P=1):
        return 100 * (t - 300)

    def process_solid_enthalpy(self, c, t):
        return 0.1 * (t - 300) - 10

    def mixture_viscosity(self, x, t, p, vapor_fraction):
        return 0.001 * math.exp(-(t - 300) / 100)


def test_enthalpy_flash_resolves_pure_component_latent_plateau():
    thermo = FusionThermo()
    for solid_amount in (0.2, 0.5, 0.8):
        state = equilibrate_enthalpy(
            thermo,
            {"A": 1},
            -10000 * solid_amount,
            1,
            initial_temperature=298,
            initial_solids={"A": 0.6},
            temperature_bounds=(250, 350),
        )
        assert state.temperature == pytest.approx(300, abs=1e-5)
        assert state.solid["A"] == pytest.approx(solid_amount, abs=1e-7)
        assert state.enthalpy == pytest.approx(-10000 * solid_amount, abs=1e-4)


def test_hot_melt_contact_melts_crystals_and_conserves_energy():
    thermo = FusionThermo()
    old = inventory_at_temperature(thermo, 300, 1, {"A": 0.5}, {"A": 0.5})
    new, out, energy = contact_cell(
        thermo,
        old,
        {"A": 0.1},
        0.1 * 100 * (320 - 300),
        0.1,
        1,
        temperature_bounds=(250, 350),
        tolerance=1e-8,
    )
    assert new.temperature == pytest.approx(300, abs=1e-5)
    assert new.solid["A"] == pytest.approx(0.48, abs=1e-7)
    assert sum(out.values()) == pytest.approx(0.1)
    assert new.enthalpy + energy == pytest.approx(old.enthalpy + 200, abs=1e-4)
    assert new.liquid_volume + new.solid_volume == pytest.approx(0.1)


def test_cold_contact_recrystallizes_and_reduces_pore_space():
    thermo = FusionThermo()
    old = inventory_at_temperature(thermo, 300, 1, {"A": 0.5}, {"A": 0.5})
    new, _, _ = contact_cell(
        thermo,
        old,
        {"A": 0.1},
        -200,
        0.1,
        1,
        temperature_bounds=(250, 350),
        tolerance=1e-8,
    )
    assert new.solid["A"] == pytest.approx(0.52, abs=1e-7)
    assert new.temperature == pytest.approx(300, abs=1e-5)


def test_psd_resize_conserves_particle_number():
    thermo = FusionThermo()
    old = inventory_at_temperature(thermo, 300, 1, {"A": 0.5}, {"A": 0.5})
    new = inventory_at_temperature(thermo, 300, 1, {"A": 0.75}, {"A": 0.25})
    original = ParticleSizeDistribution((1e-5, 2e-5), (0.2, 0.3))
    psd = resize_population(thermo, old, new, {"A": original}, None)["A"]
    assert psd.total_molar_flow == pytest.approx(0.25)
    for d0, n0, d1, n1 in zip(
        original.diameters_m,
        original.molar_flows_kmol_per_h,
        psd.diameters_m,
        psd.molar_flows_kmol_per_h,
    ):
        assert n1 / d1**3 == pytest.approx(n0 / d0**3)


def run_wash(steps, temperature=320, cells=2):
    thermo = FusionThermo()
    initial = inventory_at_temperature(thermo, 300, 1, {"A": 0.5}, {"A": 0.5})
    psd = {"A": ParticleSizeDistribution((1e-4,), (0.5,))}
    return solve_warm_wash(
        thermo,
        initial,
        psd,
        {"A": 0.2},
        0.2 * 100 * (temperature - 300),
        temperature,
        1,
        cells=cells,
        steps=steps,
        porosity=0.5,
        sphericities={"A": 0.9},
        kozeny=5,
        medium_resistance=1e9,
        pressure_drop=1e5,
        temperature_bounds=(250, 350),
        tolerance=1e-8,
    )


def test_warm_hydraulics_melting_cooling_and_balances():
    warm, cold = run_wash(8), run_wash(8, 280)
    assert sum(s.solid["A"] for s in warm.cells) == pytest.approx(0.46, abs=1e-6)
    assert sum(s.solid["A"] for s in cold.cells) == pytest.approx(0.54, abs=1e-6)
    assert sum(warm.final_resistance_area) < sum(cold.final_resistance_area)
    assert warm.time_coefficients[0] < cold.time_coefficients[0]
    assert abs(warm.energy_residual) < 1e-3
    assert warm.component_residual < 1e-12
    assert all(e > 0.5 for e in warm.final_porosity[:1])


def test_hydraulic_step_refinement():
    coarse, medium, fine = (run_wash(n) for n in (4, 8, 16))
    assert abs(medium.time_coefficients[0] - fine.time_coefficients[0]) < abs(
        coarse.time_coefficients[0] - fine.time_coefficients[0]
    )


def test_new_crystals_require_population_closure():
    thermo = FusionThermo()
    old = inventory_at_temperature(thermo, 300, 1, {"A": 1}, {})
    new = inventory_at_temperature(thermo, 300, 1, {"A": 0.5}, {"A": 0.5})
    with pytest.raises(ThermodynamicsError, match="nucleus"):
        resize_population(thermo, old, new, {}, None)
    assert resize_population(thermo, old, new, {}, 1e-5)["A"].total_molar_flow == 0.5


def test_all_liquid_above_melting_point():
    state = equilibrate_enthalpy(
        FusionThermo(),
        {"A": 1},
        2000,
        1,
        initial_temperature=300,
        initial_solids={"A": 0.5},
        temperature_bounds=(250, 350),
    )
    assert state.temperature == pytest.approx(320, abs=1e-5)
    assert not state.solid


def test_sensible_heat_front_and_mixed_cell_analytic_limit():
    class InertThermo(FusionThermo):
        components = ("B",)
        permanent_solid_components = ("A",)
        conventional_solid_components = ()

    thermo = InertThermo()
    initial = inventory_at_temperature(thermo, 280, 1, {"B": 0.5}, {"A": 0.5})
    psd = {"A": ParticleSizeDistribution((1e-4,), (0.5,))}

    def wash(cells):
        return solve_warm_wash(
            thermo,
            initial,
            psd,
            {"B": 0.2},
            0.2 * 100 * 20,
            320,
            1,
            cells=cells,
            steps=20,
            porosity=0.5,
            sphericities={"A": 0.9},
            kozeny=5,
            medium_resistance=1e9,
            pressure_drop=1e5,
            temperature_bounds=(250, 350),
        )

    single = wash(1)
    assert single.cells[0].temperature == pytest.approx(
        320 - 40 / (1 + 0.2 / 20) ** 20, abs=1e-6
    )
    front = wash(3)
    t = [c.temperature for c in front.cells]
    assert 320 > t[0] > t[1] > t[2] > 280
    assert abs(front.energy_residual) < 1e-5


def test_seeded_continuation_avoids_speculative_liquid_temperature_solve():
    class CountingThermo(FusionThermo):
        def __init__(self):
            self.temperatures = []

        def mixture_enthalpy(self, x, t, vapor_fraction, P=1):
            self.temperatures.append(t)
            return super().mixture_enthalpy(x, t, vapor_fraction, P=P)

    thermo = CountingThermo()
    result = equilibrate_enthalpy(
        thermo,
        {"A": 1.0},
        -4990,
        1,
        initial_temperature=300,
        initial_solids={"A": 0.5},
        temperature_bounds=(250, 350),
    )
    assert result.solid["A"] == pytest.approx(0.499, abs=1e-8)
    assert max(abs(t - 300) for t in thermo.temperatures) < 1
    assert len(thermo.temperatures) < 20


def test_hydraulics_evaluated_once_per_new_cell_state(monkeypatch):
    import equilibrium_washing

    original = equilibrium_washing.cell_hydraulics
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(equilibrium_washing, "cell_hydraulics", counted)
    run_wash(8, cells=2)
    assert len(calls) == 2 * (8 + 1)
