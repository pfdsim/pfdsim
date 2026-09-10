"""Transient, finite-rate sweating of a porous solid-solution layer.

An ideal substitutional solid solution exchanges material with a well-mixed
pore liquid by reversible, dissipative kinetics. Finite heat input determines temperature. Liquid drains concurrently
through a capillary-bundle Darcy closure. This is a lumped transport model,
not a spatial heat-transfer or moving-boundary simulation.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .crystallizer_specs import CrystallizerSpecificationError, finite_crystallizer_number
    from .thermodynamics_models.common import R
    from .thermodynamics_models.sle import (
        liquid_solution_activities, pure_solid_log_saturation_activity,
    )
else:
    from crystallizer_specs import CrystallizerSpecificationError, finite_crystallizer_number
    from thermodynamics_models.common import R
    from thermodynamics_models.sle import (
        liquid_solution_activities, pure_solid_log_saturation_activity,
    )


class LayerSweatingError(ValueError):
    """Invalid input or an unsupported sweating state."""


def _number(value, name, minimum=0.0, maximum=None, inclusive=False):
    try:
        return finite_crystallizer_number(value, name, minimum=minimum,
                                           maximum=maximum, inclusive=inclusive)
    except CrystallizerSpecificationError as error:
        raise LayerSweatingError(str(error)) from error


@dataclass(frozen=True)
class SolidSolutionSolute:
    """Ideal-solid standard state relative to the pure liquid at operating P.

    partition = y_s / a_l at reference_temperature; transfer_enthalpy is
    h_s^0 - h_l^0 [kJ/mol], constant in temperature. This van't Hoff closure
    keeps the partition law and caloric model thermodynamically consistent.
    diffusivity [m2/s] enters a first-mode linear-driving-force approximation.
    """
    partition: float
    transfer_enthalpy: float
    diffusivity: float
    reference_temperature: float = 298.15

    def __post_init__(self):
        _number(self.partition, 'solid partition coefficient')
        _number(self.transfer_enthalpy, 'solid transfer enthalpy', -math.inf)
        _number(self.diffusivity, 'solid diffusivity', inclusive=True)
        _number(self.reference_temperature, 'partition reference temperature')

    def log_partition(self, temperature):
        return (math.log(self.partition) - 1000 * self.transfer_enthalpy / R
                * (1 / temperature - 1 / self.reference_temperature))


def layer_enthalpy(thermo, temperature, pressure, solid, liquid, host, solutes):
    """Total kJ on the supplied kmol basis; ideal solid has zero mixing enthalpy."""
    liquid_total = sum(liquid.values())
    energy = 0.0
    if liquid_total > 0:
        x = {c: n / liquid_total for c, n in liquid.items() if n > 0}
        energy = liquid_total * thermo.mixture_enthalpy(x, temperature, 0, P=pressure)
    for name, amount in solid.items():
        if amount == 0:
            continue
        if name == host:
            enthalpy = thermo.process_solid_enthalpy(name, temperature)
        else:
            enthalpy = (thermo.enthalpy_liquid(name, temperature)
                        + solutes[name].transfer_enthalpy)
        energy += amount * 1000 * enthalpy
    return energy


def allocate_layer_inventory(*, host, totals, host_solid, captured,
                             occluded_fractions, occluded_host_fraction=None,
                             empirical=False):
    """Allocate captured impurities, keeping separately retained liquor liquid.

    Empirical k_eff reports total deposited component amounts, not pore-liquid
    composition. Its solvent content therefore needs an independent closure.
    Mechanistic growth already supplies a complete mechanical-liquid inventory.
    """
    solids = {host: host_solid}
    if empirical:
        impurity_liquid = 0.0
        for name, amount in captured.items():
            if name == host or amount <= 0:
                continue
            if name not in occluded_fractions:
                raise LayerSweatingError(f'Missing occluded_fraction_{name}')
            fraction = _number(occluded_fractions[name], f'occluded_fraction_{name}',
                               maximum=1, inclusive=True)
            solids[name] = (1 - fraction) * amount
            impurity_liquid += fraction * amount
        if impurity_liquid > 0:
            x_host = _number(occluded_host_fraction, 'occluded_liquid_host_fraction',
                             maximum=1, inclusive=True)
            if x_host >= 1:
                raise LayerSweatingError('Occluded liquid containing impurities needs host fraction < 1')
            # Borrow the solvent from empirical deposited host, never create it.
            solids[host] -= impurity_liquid * x_host / (1 - x_host)
    elif occluded_fractions or occluded_host_fraction is not None:
        raise LayerSweatingError(
            'Occluded allocation specifications require empirical_layer_growth; '
            'mechanistic growth explicitly predicts mechanical liquid inclusions'
        )
    if solids[host] <= 0:
        raise LayerSweatingError('Occluded solvent exhausts the deposited host crystal')
    liquid = {c: n - solids.get(c, 0.0) for c, n in totals.items()}
    if any(n < 0 or not math.isfinite(n) for n in (*solids.values(), *liquid.values())):
        raise LayerSweatingError('Invalid initial solid/liquid layer allocation')
    return {c: n for c, n in solids.items() if n > 0}, liquid


@dataclass
class LayerSweatingResult:
    solid_amounts: dict
    liquid_amounts: dict
    sweat_amounts: dict
    connected_liquid_amounts: dict
    sealed_liquid_amounts: dict
    temperature: float
    supplied_heat_kJ: float
    drained_enthalpy_kJ: float
    remaining_enthalpy_kJ: float
    energy_balance_residual: float
    profile: list
    evaluations: int
    component_balance_residual: float


def solve_layer_sweating(
    thermo, *, host, initial_solid, initial_liquid, solutes, temperature, pressure,
    heater_temperature, thermal_conductance_W_K, opening_coefficient,
    duration_s, host_rate_constant, diffusion_length, drainage_length,
    pore_radius, tortuosity, connected_fraction, residual_saturation,
    capillary_pressure, solid_molar_volume, relative_tolerance=1e-7,
):
    """Coupled batch energy, phase exchange, pore opening, and Darcy drainage.

    Units: kmol, seconds, metres, Pa, m3/kmol, W/K. Connected and sealed
    regions have independent solid/liquid inventories and a common temperature.
    Each region uses the same dissipative congruent melting and substitutional
    exchange law. Only connected liquid drains. Melting in the sealed region
    opens it at hazard alpha * max(net melting rate, 0) / sealed solid amount;
    opening transfers both solid and liquid into the connected region.

    Remaining enthalpy is a state variable: dH/dt = UA*(Th-T)/1000 - hL*drain.
    Inverting the phase enthalpies accounts for latent heat, sensible heat and
    liquid mixing on opening. No imposed-temperature or infinite-heat path.
    Darcy geometry is still a lumped, fixed-envelope capillary-bundle closure.
    """
    for name, value in dict(temperature=temperature, pressure=pressure,
                            heater_temperature=heater_temperature,
                            duration_s=duration_s, host_rate_constant=host_rate_constant,
                            diffusion_length=diffusion_length, drainage_length=drainage_length,
                            pore_radius=pore_radius, solid_molar_volume=solid_molar_volume,
                            relative_tolerance=relative_tolerance).items():
        _number(value, name)
    _number(thermal_conductance_W_K, 'thermal conductance', inclusive=True)
    _number(opening_coefficient, 'opening coefficient', inclusive=True)
    _number(tortuosity, 'tortuosity', minimum=1, inclusive=True)
    _number(connected_fraction, 'initial connected fraction', maximum=1, inclusive=True)
    _number(residual_saturation, 'residual saturation', maximum=1, inclusive=True)
    if residual_saturation >= 1:
        raise LayerSweatingError('Residual saturation must be < 1')
    _number(capillary_pressure, 'capillary pressure', inclusive=True)
    melting_temperature = float(thermo.props[host].Tm)
    if max(temperature, heater_temperature) >= melting_temperature:
        raise LayerSweatingError('Initial and heating-medium temperatures must be below host Tm')
    names = tuple(c for c in thermo.components
                  if initial_solid.get(c, 0) + initial_liquid.get(c, 0) > 0)
    unknown = (set(initial_solid) | set(initial_liquid) | set(solutes)) - set(thermo.components)
    if unknown:
        raise LayerSweatingError('Unknown layer components: ' + ', '.join(sorted(unknown)))
    if not names:
        raise LayerSweatingError('Sweating requires a nonempty layer')
    if host in solutes:
        raise LayerSweatingError('Host partitioning uses its fusion thermodynamics')
    for name, amount in initial_solid.items():
        if amount > 0 and name != host and name not in solutes:
            raise LayerSweatingError(f'Missing solid-solution thermodynamics/transport for {name}')
    s0 = np.array([initial_solid.get(c, 0.0) for c in names], dtype=float)
    l0 = np.array([initial_liquid.get(c, 0.0) for c in names], dtype=float)
    if np.any(~np.isfinite(s0)) or np.any(~np.isfinite(l0)) or min(*s0, *l0) < 0:
        raise LayerSweatingError('Initial inventories must be finite and nonnegative')
    if sum(s0) <= 0 or sum(l0) <= 0:
        raise LayerSweatingError('Finite-rate sweating requires initial solid and pore liquid')
    if host not in names or s0[names.index(host)] <= 0:
        raise LayerSweatingError('Sweating requires a host crystal')
    count = len(names)
    totals = s0 + l0
    scale = sum(totals)
    energy_scale = scale * 1e4
    allowed = np.array([c == host or c in solutes for c in names])
    host_index = names.index(host)
    rates = np.array([math.pi**2 * solutes[c].diffusivity / diffusion_length**2
                      if c in solutes else 0 for c in names])

    def liquid_energy(T, liquid):
        total = sum(liquid)
        if total <= 0:
            return 0.0
        return total * thermo.mixture_enthalpy(
            dict(zip(names, liquid / total)), T, 0, P=pressure)

    def energy(T, sc, si, lc, li):
        return (layer_enthalpy(thermo, T, pressure, dict(zip(names, sc + si)),
                              dict(zip(names, lc)), host, solutes)
                + liquid_energy(T, li))

    initial_energy = energy(temperature, s0, np.zeros(count), l0, np.zeros(count))
    x0 = dict(zip(names, l0 / sum(l0)))
    initial_liquid_volume = sum(l0) / thermo.mixture_liquid_density(x0, temperature)
    envelope = sum(s0) * solid_molar_volume + initial_liquid_volume

    def phase_exchange(solid, liquid, partition, log_k, T):
        S, L = sum(solid), sum(liquid)
        if S <= 0 or L <= 0:
            return np.zeros(count), 0.0, 0.0
        y = solid / S
        activities = liquid_solution_activities(thermo, T, pressure,
                                                dict(zip(names, liquid / L)))
        a = np.array([activities.get(c, 0.0) for c in names])
        force = np.zeros(count)
        force[allowed] = (np.log(np.maximum(y[allowed], 1e-300))
                          - log_k[allowed] - np.log(np.maximum(a[allowed], 1e-300)))
        affinity = float(np.dot(y, force))
        if affinity < -600:
            raise LayerSweatingError('Phase conversion driving force exceeds the supported numerical range')
        melting_rate = -host_rate_constant * S * math.expm1(-affinity)
        exchange = melting_rate * y
        for i, name in enumerate(names):
            if name in solutes:
                diffusion = rates[i] * S * (y[i] * partition[host_index] * a[host_index]
                                           - y[host_index] * partition[i] * a[i])
                exchange[i] += diffusion
                exchange[host_index] -= diffusion
        return exchange, melting_rate, float(np.dot(exchange, force)) * R * T

    def evaluate(state):
        inventories = np.maximum(state[:5*count], 0).reshape(5, count) * totals
        # Insoluble species have no solid-state degree of freedom. Numerical
        # Jacobian probes must not manufacture a hypothetical solid standard state.
        inventories[:2, ~allowed] = 0.
        sc, si, lc, li, _sweat = inventories
        if sum(sc + si) <= 1e-14 * scale or sum(lc + li) <= 1e-14 * scale:
            raise LayerSweatingError('Sweating exhausted a phase; layer disappearance/dry-out is unsupported')
        target = initial_energy + state[5*count] * energy_scale
        def residual(T):
            return (energy(T, sc, si, lc, li) - target) / energy_scale
        # A distinct solid-solution + two-liquid energy state cannot be passed
        # to the existing single-stream enthalpy inversion without mixing pores.
        floor = max(1., min(temperature, heater_temperature) - 100.)
        lower, upper = max(floor, temperature - 5.), min(melting_temperature, temperature + 5.)
        span = 5.
        while residual(lower) > 0 and lower > floor:
            span *= 2
            lower = max(floor, temperature - span)
        while residual(upper) < 0 and upper < melting_temperature:
            span *= 2
            upper = min(melting_temperature, temperature + span)
        if residual(lower) > 0 or residual(upper) < 0:
            raise LayerSweatingError('Layer energy leaves the supported temperature interval')
        T = brentq(residual, lower, upper, xtol=1e-8)
        log_k = np.array([-pure_solid_log_saturation_activity(thermo, host, T, pressure)
                          if c == host else solutes[c].log_partition(T) if c in solutes else 0
                          for c in names])
        if np.any(np.abs(log_k) > 600):
            raise LayerSweatingError('Solid partition coefficients exceed the supported numerical range')
        partition = np.exp(log_k)
        jc, mc, dc = phase_exchange(sc, lc, partition, log_k, T)
        ji, mi, di = phase_exchange(si, li, partition, log_k, T)
        opening_rate = opening_coefficient * max(mi, 0) / sum(si) if sum(si) > 0 else 0.
        opened_solid, opened_liquid = opening_rate * si, opening_rate * li
        drain = np.zeros(count)
        drain_volume_rate = permeability = saturation = 0.
        viscosity = mass_density = 0.
        def volume_of(liquid):
            total = sum(liquid)
            if total <= 0:
                return 0.
            x = dict(zip(names, liquid / total))
            return total / _number(thermo.mixture_liquid_density(x, T), 'liquid density')
        vc, vi = volume_of(lc), volume_of(li)
        solid_volume = sum(sc + si) * solid_molar_volume
        volume = max(envelope, solid_volume + vc + vi)
        pore_volume = volume - solid_volume
        porosity = pore_volume / volume
        # Sealed pore volume is unavailable to the connected drainage network.
        connected_pore_volume = max(0., pore_volume - vi)
        if sum(lc) > 0 and connected_pore_volume > 0:
            x = dict(zip(names, lc / sum(lc)))
            density = sum(lc) / vc
            viscosity = _number(thermo.mixture_viscosity(x, T, pressure, 0), 'liquid viscosity')
            saturation = min(1., vc / connected_pore_volume)
            effective = max(0., (saturation - residual_saturation) / (1 - residual_saturation))
            mass_density = density * sum(x[c] * thermo.props[c].MW for c in names)
            driving = max(0., mass_density * 9.80665 * drainage_length - capillary_pressure)
            permeability = connected_pore_volume / volume * pore_radius**2 / (8 * tortuosity)
            drain_volume_rate = permeability * effective**3 * volume * driving / (viscosity * drainage_length**2)
            drain = drain_volume_rate * density * lc / sum(lc)
        heat_rate = thermal_conductance_W_K * (heater_temperature - T) / 1000
        drain_enthalpy_rate = liquid_energy(T, drain)
        changes = np.concatenate((-jc + opened_solid, -ji - opened_solid,
                                  jc + opened_liquid - drain, ji - opened_liquid, drain)) / np.tile(totals, 5)
        derivative = np.concatenate((changes, [(heat_rate - drain_enthalpy_rate) / energy_scale,
                                               heat_rate / energy_scale, drain_enthalpy_rate / energy_scale]))
        return derivative, {
            'temperature_K': T,
            'solid_amounts_kmol': dict(zip(names, sc + si)),
            'liquid_amounts_kmol': dict(zip(names, lc + li)),
            'connected_solid_amounts_kmol': dict(zip(names, sc)),
            'sealed_solid_amounts_kmol': dict(zip(names, si)),
            'connected_liquid_amounts_kmol': dict(zip(names, lc)),
            'sealed_liquid_amounts_kmol': dict(zip(names, li)),
            'porosity': porosity, 'liquid_saturation': saturation,
            'permeability_m2': permeability, 'drainage_m3_s': drain_volume_rate,
            'opening_rate_per_s': opening_rate,
            'congruent_melting_rate_kmol_s': mc + mi,
            'liquid_viscosity_Pa_s': viscosity, 'liquid_density_kg_m3': mass_density,
            'phase_transfer_dissipation_kJ_s': dc + di,
            'heater_duty_kW': heat_rate, 'drained_enthalpy_rate_kJ_s': drain_enthalpy_rate,
        }

    f = connected_fraction
    initial = np.concatenate((f*s0/totals, (1-f)*s0/totals, f*l0/totals,
                              (1-f)*l0/totals, np.zeros(count), np.zeros(3)))
    solved = solve_ivp(lambda _t, state: evaluate(state)[0], (0, duration_s), initial,
                       method='Radau', rtol=relative_tolerance, atol=relative_tolerance * 1e-4,
                       max_step=duration_s / 30, dense_output=True)
    if not solved.success:
        raise LayerSweatingError('Sweating integration failed: ' + solved.message)
    if np.min(solved.y[:5*count]) < -relative_tolerance * 1e-2:
        raise LayerSweatingError('Sweating integration violated nonnegative inventories')
    final = solved.y[:, -1]
    sc, si, lc, li, sweat = np.maximum(final[:5*count], 0).reshape(5, count) * totals
    balance = float(np.max(np.abs(sc + si + lc + li + sweat - totals) / totals))
    if balance > relative_tolerance:
        raise LayerSweatingError('Sweating integration violated component conservation')
    profile = [dict(time_s=float(t), **evaluate(solved.sol(t))[1])
               for t in np.linspace(0, duration_s, 21)]
    final_T = profile[-1]['temperature_K']
    remaining_energy = energy(final_T, sc, si, lc, li)
    supplied, drained = final[5*count+1:] * energy_scale
    energy_error = abs(remaining_energy + drained - initial_energy - supplied) / energy_scale
    if energy_error > relative_tolerance:
        raise LayerSweatingError('Sweating integration violated energy conservation')
    return LayerSweatingResult(
        dict(zip(names, sc+si)), dict(zip(names, lc+li)), dict(zip(names, sweat)),
        dict(zip(names, lc)), dict(zip(names, li)), final_T, float(supplied), float(drained),
        remaining_energy, energy_error, profile, solved.nfev, balance)
