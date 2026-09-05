"""Conservative equilibrium contacts and hydraulic integration for warm washing.

Inventories are kmol, energy is kJ, volume is m3, and pressure is bar.
The model uses local thermal equilibrium and fixed spatial cell volumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares, root

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from .particle_size_distributions import ParticleSizeDistribution
    from .thermodynamics_models.common import ThermodynamicsError
    from .thermodynamics_models.sle import (
        liquid_solution_activities,
        pure_solid_log_saturation_activity,
    )
else:
    from particle_size_distributions import ParticleSizeDistribution
    from thermodynamics_models.common import ThermodynamicsError
    from thermodynamics_models.sle import (
        liquid_solution_activities,
        pure_solid_log_saturation_activity,
    )


@dataclass
class EquilibriumInventory:
    temperature: float
    liquid: dict
    solid: dict
    enthalpy: float
    liquid_enthalpy: float
    liquid_molar_volume: float
    solid_volume: float
    residual: float = 0.0

    @property
    def liquid_volume(self):
        return sum(self.liquid.values()) * self.liquid_molar_volume


def _log_saturation(thermo, component, temperature, pressure):
    melting_temperature = thermo.props[component].Tm
    if melting_temperature is None:
        raise ThermodynamicsError(f"Warm washing requires Tm for {component}")
    return pure_solid_log_saturation_activity(
        thermo,
        component,
        temperature,
        pressure,
        solid_volume_temperature=min(temperature, float(melting_temperature)),
    )


def inventory_enthalpy(thermo, temperature, pressure, liquid, solid):
    liquid_total = sum(liquid.values())
    if liquid_total <= 0:
        raise ThermodynamicsError("Equilibrium washing requires a liquid phase")
    composition = {c: n / liquid_total for c, n in liquid.items() if n > 0}
    hl = thermo.mixture_enthalpy(composition, temperature, 0.0, P=pressure)
    energy = liquid_total * hl + sum(
        n * 1000 * thermo.process_solid_enthalpy(c, temperature)
        for c, n in solid.items()
        if n > 0
    )
    if not all(math.isfinite(v) for v in (hl, energy)):
        raise ThermodynamicsError("Invalid equilibrium washing caloric properties")
    return energy, hl


def inventory_at_temperature(thermo, temperature, pressure, liquid, solid):
    energy, hl = inventory_enthalpy(thermo, temperature, pressure, liquid, solid)
    composition = {c: n / sum(liquid.values()) for c, n in liquid.items() if n > 0}
    vm = thermo.mixture_liquid_molar_volume(composition, temperature)
    vs = sum(
        n * thermo._solid_molar_volume(c, temperature)
        for c, n in solid.items()
        if n > 0
    )
    if not all(math.isfinite(v) for v in (vm, vs)) or vm <= 0 or vs < 0:
        raise ThermodynamicsError(
            "Invalid equilibrium washing caloric or volume properties"
        )
    return EquilibriumInventory(
        temperature, dict(liquid), dict(solid), energy, hl, vm, vs
    )


def equilibrate_enthalpy(
    thermo,
    totals,
    energy,
    pressure,
    *,
    initial_temperature,
    initial_solids,
    temperature_bounds,
    tolerance=1e-8,
):
    """Simultaneously solve energy and pure-solid complementarity.

    Unlike a scalar root around TP-SLE, this retains the latent-heat degree of
    freedom for pure-component coexistence at the melting point. Permanent
    solids are fixed. No melting-point cutoff bypasses the pressure correction.
    """
    permanent = set(getattr(thermo, "permanent_solid_components", ()))
    fixed = {c: n for c, n in totals.items() if c in permanent and n > 0}
    fluid_totals = {c: n for c, n in totals.items() if c not in permanent and n > 0}
    candidates = [
        c for c in thermo.conventional_solid_components if fluid_totals.get(c, 0) > 0
    ]
    total = sum(totals.values())
    if (
        total <= 0
        or not fluid_totals
        or any(not math.isfinite(n) or n < 0 for n in totals.values())
    ):
        raise ThermodynamicsError("Invalid equilibrium washing inventory")
    heat_scale = max(total * 1e4, 1e-12)
    lower, upper = temperature_bounds

    # At a disappearing phase, a bounded least-squares gradient can become
    # small before the solid amount is sufficiently small. Test the exact
    # all-liquid PH branch and its chemical stability before complementarity.
    def liquid_energy(t):
        return inventory_enthalpy(thermo, t, pressure, fluid_totals, fixed)[0] - energy

    def stable_liquid():
        # Expand around the preceding contact temperature instead of probing
        # the property models at both distant global bounds on every contact.
        lo = hi = min(upper, max(lower, initial_temperature))
        flo = fhi = liquid_energy(lo)
        width = 1.0
        while flo > 0 and lo > lower:
            lo = max(lower, initial_temperature - width)
            flo = liquid_energy(lo)
            width *= 2
        width = 1.0
        while fhi < 0 and hi < upper:
            hi = min(upper, initial_temperature + width)
            fhi = liquid_energy(hi)
            width *= 2
        if flo > 0 or fhi < 0:
            return None
        liquid_temperature = (
            lo
            if flo == 0
            else (hi if fhi == 0 else brentq(liquid_energy, lo, hi, xtol=1e-9))
        )
        x = {c: n / sum(fluid_totals.values()) for c, n in fluid_totals.items()}
        activities = (
            liquid_solution_activities(thermo, liquid_temperature, pressure, x)
            if candidates
            else {}
        )
        if all(
            _log_saturation(thermo, c, liquid_temperature, pressure)
            - math.log(max(activities.get(c, 0), 1e-300))
            >= -tolerance
            for c in candidates
        ):
            return inventory_at_temperature(
                thermo, liquid_temperature, pressure, fluid_totals, fixed
            )
        return None

    has_seed = any(initial_solids.get(c, 0) > total * tolerance for c in candidates)
    if not has_seed:
        liquid_state = stable_liquid()
        if liquid_state is not None:
            return liquid_state
    guess = [min(upper, max(lower, initial_temperature)) / 300]
    guess.extend(
        min(1 - 1e-10, max(1e-10, initial_solids.get(c, 0) / fluid_totals[c]))
        for c in candidates
    )
    latest = {}
    saturation_cache = {}

    def saturation_at(temperature):
        # Numerical Jacobian columns for solid amounts share a temperature.
        # Fusion Cp integrals and pure-reference fugacities must not be redone
        # for each column. Exact keys preserve the unapproximated equations.
        if temperature not in saturation_cache:
            saturation_cache[temperature] = np.array(
                [_log_saturation(thermo, c, temperature, pressure) for c in candidates]
            )
        return saturation_cache[temperature]

    def evaluate(values):
        temperature = values[0] * 300
        solids = dict(fixed)
        liquid = dict(fluid_totals)
        for i, c in enumerate(candidates):
            solids[c] = float(values[i + 1]) * fluid_totals[c]
            liquid[c] -= solids[c]
        trial_energy = inventory_enthalpy(
            thermo, temperature, pressure, liquid, solids
        )[0]
        composition = {c: n / sum(liquid.values()) for c, n in liquid.items() if n > 0}
        activities = (
            liquid_solution_activities(thermo, temperature, pressure, composition)
            if candidates
            else {}
        )
        gaps = saturation_at(temperature) - np.array(
            [math.log(max(activities.get(c, 0), 1e-300)) for c in candidates]
        )
        amounts = np.array([solids[c] / total for c in candidates])
        complementarity = np.hypot(amounts, gaps) - amounts - gaps
        latest.update(
            temperature=temperature,
            liquid=liquid,
            solids=solids,
            gaps=gaps,
            amounts=amounts,
        )
        return np.r_[(trial_energy - energy) / heat_scale, complementarity]

    # Smooth continuation normally needs only a few Newton updates. The
    # bounded complementarity solve remains the fallback at phase boundaries.
    lower_values = np.array([lower / 300] + [0] * len(candidates))
    upper_values = np.array([upper / 300] + [1 - 1e-14] * len(candidates))

    class OutsideInventory(Exception):
        pass

    def continuation_residual(values):
        if np.any(values < lower_values) or np.any(values > upper_values):
            raise OutsideInventory
        return evaluate(values)

    def accepted(values):
        residual = evaluate(values)
        return float(np.max(np.abs(residual))) <= tolerance and all(
            gap >= -10 * tolerance
            and (amount <= tolerance or abs(gap) <= 10 * tolerance)
            for gap, amount in zip(latest["gaps"], latest["amounts"])
        )

    continued = None
    if has_seed:
        try:
            trial = root(
                continuation_residual,
                guess,
                method="hybr",
                options={"xtol": 1e-9, "maxfev": 30 + 4 * len(candidates)},
            )
            if (
                np.all(trial.x >= lower_values)
                and np.all(trial.x <= upper_values)
                and accepted(trial.x)
            ):
                continued = trial
        except OutsideInventory:
            pass
    solved = (
        continued
        if continued is not None
        else least_squares(
            evaluate,
            guess,
            bounds=(
                [lower / 300] + [0] * len(candidates),
                [upper / 300] + [1 - 1e-14] * len(candidates),
            ),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=200,
            x_scale="jac",
        )
    )
    residual = evaluate(solved.x)
    error = float(np.max(np.abs(residual)))
    gaps = latest["gaps"]
    feasible = all(
        gap >= -10 * tolerance and (amount <= tolerance or abs(gap) <= 10 * tolerance)
        for gap, amount in zip(gaps, latest["amounts"])
    )
    if (continued is None and not solved.success) or error > tolerance or not feasible:
        if has_seed:
            liquid_state = stable_liquid()
            if liquid_state is not None:
                return liquid_state
        raise ThermodynamicsError(
            f"Warm washing enthalpy/SLE solve failed (residual={error:.3g}); "
            "check phase data, temperature bounds, and liquid-phase feasibility"
        )
    # Remove numerical traces without losing their component inventory.
    for c in candidates:
        if latest["solids"][c] <= total * tolerance * 0.01:
            latest["liquid"][c] += latest["solids"].pop(c)
    state = inventory_at_temperature(
        thermo, latest["temperature"], pressure, latest["liquid"], latest["solids"]
    )
    state.residual = max(error, abs(state.enthalpy - energy) / heat_scale)
    return state


def contact_cell(
    thermo,
    old,
    incoming,
    incoming_energy,
    cell_volume,
    pressure,
    *,
    temperature_bounds,
    tolerance,
):
    totals = {
        c: old.liquid.get(c, 0) + old.solid.get(c, 0) + incoming.get(c, 0)
        for c in set(old.liquid) | set(old.solid) | set(incoming)
    }
    mixed = equilibrate_enthalpy(
        thermo,
        totals,
        old.enthalpy + incoming_energy,
        pressure,
        initial_temperature=old.temperature,
        initial_solids=old.solid,
        temperature_bounds=temperature_bounds,
        tolerance=tolerance,
    )
    displaced_volume = mixed.liquid_volume + mixed.solid_volume - cell_volume
    if displaced_volume < -1e-10 * cell_volume:
        raise ThermodynamicsError(
            "Warm washing contact becomes unsaturated through volume contraction; "
            "the saturated fixed-volume hydraulic model cannot represent this state"
        )
    if mixed.solid_volume >= cell_volume * (1 - 1e-8):
        raise ThermodynamicsError(
            "Recrystallization blocks the warm-washing cake pores"
        )
    fraction = max(0.0, displaced_volume) / mixed.liquid_volume
    if fraction >= 1:
        raise ThermodynamicsError("Warm washing leaves no retained pore liquid")
    outgoing = {c: n * fraction for c, n in mixed.liquid.items()}
    remaining = {c: n - outgoing[c] for c, n in mixed.liquid.items()}
    # Removing homogeneous liquid leaves all intensive properties unchanged.
    state = EquilibriumInventory(
        mixed.temperature,
        remaining,
        mixed.solid,
        mixed.enthalpy - sum(outgoing.values()) * mixed.liquid_enthalpy,
        mixed.liquid_enthalpy,
        mixed.liquid_molar_volume,
        mixed.solid_volume,
    )
    state.residual = mixed.residual
    return state, outgoing, sum(outgoing.values()) * mixed.liquid_enthalpy


def resize_population(thermo, previous, state, distributions, nucleus_diameter):
    """Homothetic growth/shrinkage at fixed class particle counts.

    Sphericity is unchanged. New solids without surviving particles require an
    explicit effective nucleus diameter; this is a closure, not nucleation kinetics.
    """
    result = {}
    for c, amount in state.solid.items():
        if amount <= 0:
            continue
        old = distributions.get(c)
        if old is None or previous.solid.get(c, 0) <= 0:
            if nucleus_diameter is None:
                raise ThermodynamicsError(
                    f"Warm washing forms new solid {c!r}; specify equilibrium_nucleus_diameter"
                )
            result[c] = ParticleSizeDistribution((nucleus_diameter,), (amount,))
            continue
        amount_ratio = amount / previous.solid[c]
        volume_ratio = (
            amount_ratio
            * thermo._solid_molar_volume(c, state.temperature)
            / thermo._solid_molar_volume(c, previous.temperature)
        )
        result[c] = ParticleSizeDistribution(
            tuple(d * volume_ratio ** (1 / 3) for d in old.diameters_m),
            tuple(n * amount_ratio for n in old.molar_flows_kmol_per_h),
        )
    return result


def cell_hydraulics(
    thermo, state, distributions, sphericities, cell_volume, kozeny, pressure
):
    porosity = 1 - state.solid_volume / cell_volume
    if not 1e-8 < porosity <= 1:
        raise ThermodynamicsError(
            "Warm washing cake porosity is outside the permeable range"
        )
    surface = sum(
        6 * n * thermo._solid_molar_volume(c, state.temperature) / (sphericities[c] * d)
        for c, psd in distributions.items()
        for d, n in zip(psd.diameters_m, psd.molar_flows_kmol_per_h)
    )
    # Rcake*A = L*A/k = K * (surface/Vbed)^2 * Vbed / epsilon^3.
    resistance_area = kozeny * surface**2 / (cell_volume * porosity**3)
    composition = {
        c: n / sum(state.liquid.values()) for c, n in state.liquid.items() if n > 0
    }
    viscosity = thermo.mixture_viscosity(composition, state.temperature, pressure, 0.0)
    if not math.isfinite(viscosity) or viscosity <= 0:
        raise ThermodynamicsError("Invalid local liquid viscosity in warm washing")
    return resistance_area, viscosity, porosity


def collected_population(thermo, state, entries, nucleus_diameter):
    """Equilibrated collection of populations at different temperatures.

    A common volume scale preserves the particle count of every surviving
    incoming class while satisfying the collected equilibrium solid amount.
    """
    result = {}
    for c, amount in state.solid.items():
        populations = entries.get(c, ())
        original_volume = sum(
            psd.total_molar_flow * thermo._solid_molar_volume(c, t)
            for psd, t in populations
        )
        if original_volume <= 0:
            if nucleus_diameter is None:
                raise ThermodynamicsError(
                    f"Collection forms new solid {c!r}; specify equilibrium_nucleus_diameter"
                )
            result[c] = ParticleSizeDistribution((nucleus_diameter,), (amount,))
            continue
        final_vm = thermo._solid_molar_volume(c, state.temperature)
        factor = amount * final_vm / original_volume
        resized = [
            ParticleSizeDistribution(
                tuple(d * factor ** (1 / 3) for d in psd.diameters_m),
                tuple(
                    n * thermo._solid_molar_volume(c, t) * factor / final_vm
                    for n in psd.molar_flows_kmol_per_h
                ),
            )
            for psd, t in populations
        ]
        result[c] = ParticleSizeDistribution.combine(resized)
    return result


@dataclass
class WarmWashResult:
    cells: list
    distributions: list
    effluent: dict
    effluent_enthalpy: float
    time_coefficients: tuple
    final_porosity: list
    final_resistance_area: list
    final_viscosity: list
    maximum_equilibrium_residual: float
    energy_residual: float
    component_residual: float


def solve_warm_wash(
    thermo,
    initial,
    initial_psd,
    wash,
    wash_energy,
    wash_temperature,
    pressure,
    *,
    cells,
    steps,
    porosity,
    sphericities,
    kozeny,
    medium_resistance,
    pressure_drop,
    temperature_bounds,
    tolerance=1e-8,
    nucleus_diameter=None,
    validate_liquid=None,
):
    """Backward-Euler mixed-cell displacement in delivered-wash coordinates.

    Each parcel conserves components and enthalpy exactly up to SLE tolerance.
    Its duration follows Darcy's law, accounting for different inlet/outlet
    volumes caused by thermal expansion and phase changes. Refining ``steps``
    controls time/displacement error; ``cells`` controls spatial dispersion.
    """
    cell_volume = initial.solid_volume / ((1 - porosity) * cells)
    states = [
        inventory_at_temperature(
            thermo,
            initial.temperature,
            pressure,
            {c: n / cells for c, n in initial.liquid.items()},
            {c: n / cells for c, n in initial.solid.items()},
        )
        for _ in range(cells)
    ]
    populations = [
        {c: psd.scaled(1 / cells) for c, psd in initial_psd.items()}
        for _ in range(cells)
    ]
    a = b = effluent_energy = maximum_error = 0.0
    effluent = {}
    wash_composition = {c: n / sum(wash.values()) for c, n in wash.items() if n > 0}
    inlet_vm = thermo.mixture_liquid_molar_volume(wash_composition, wash_temperature)
    hydraulics = [
        cell_hydraulics(thermo, state, psd, sphericities, cell_volume, kozeny, pressure)
        for state, psd in zip(states, populations)
    ]
    for _ in range(steps):
        previous_a, previous_b = a, b
        incoming = {c: n / steps for c, n in wash.items()}
        incoming_energy = wash_energy / steps
        incoming_volume = sum(incoming.values()) * inlet_vm
        for j in range(cells):
            old = states[j]
            old_r, old_mu, _ = hydraulics[j]
            state, outgoing, outgoing_energy = contact_cell(
                thermo,
                old,
                incoming,
                incoming_energy,
                cell_volume,
                pressure,
                temperature_bounds=temperature_bounds,
                tolerance=tolerance,
            )
            if validate_liquid is not None:
                validate_liquid(state)
            populations[j] = resize_population(
                thermo, old, state, populations[j], nucleus_diameter
            )
            hydraulics[j] = cell_hydraulics(
                thermo,
                state,
                populations[j],
                sphericities,
                cell_volume,
                kozeny,
                pressure,
            )
            new_r, new_mu, _ = hydraulics[j]
            outgoing_volume = sum(outgoing.values()) * state.liquid_molar_volume
            a += (
                (old_r * old_mu + new_r * new_mu)
                * (incoming_volume + outgoing_volume)
                / (4 * pressure_drop)
            )
            states[j] = state
            incoming, incoming_energy, incoming_volume = (
                outgoing,
                outgoing_energy,
                outgoing_volume,
            )
            maximum_error = max(maximum_error, state.residual)
        b += new_mu * medium_resistance * incoming_volume / pressure_drop
        if a == previous_a and b == previous_b:
            raise ThermodynamicsError(
                "Warm washing has no hydraulic resistance after melting; specify positive medium_resistance"
            )
        for c, amount in incoming.items():
            effluent[c] = effluent.get(c, 0) + amount
        effluent_energy += incoming_energy
    components = set(initial.liquid) | set(initial.solid) | set(wash)
    mass_error = max(
        abs(
            initial.liquid.get(c, 0)
            + initial.solid.get(c, 0)
            + wash.get(c, 0)
            - effluent.get(c, 0)
            - sum(s.liquid.get(c, 0) + s.solid.get(c, 0) for s in states)
        )
        for c in components
    )
    energy_error = (
        initial.enthalpy
        + wash_energy
        - effluent_energy
        - sum(s.enthalpy for s in states)
    )
    return WarmWashResult(
        states,
        populations,
        effluent,
        effluent_energy,
        (a, b),
        [h[2] for h in hydraulics],
        [h[0] for h in hydraulics],
        [h[1] for h in hydraulics],
        maximum_error,
        energy_error,
        mass_error,
    )
