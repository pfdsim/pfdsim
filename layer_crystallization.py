"""Finite-rate planar layer crystallization with one crystallizing component.

The solid and liquid-film temperature profiles are quasi-steady. The bulk
inventory evolves in time; the interface obeys pure-solid SLE and a Stefan
balance. No bulk nucleation, solid solution, sweating, or Soret effect is used.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics_models.common import ThermodynamicsError
    from .thermodynamics_models.sle import (
        liquid_solution_activities, pure_solid_log_saturation_activity,
    )
    from .unit_operations_basic import _ThermoStateSolver
    from .transport_correlations import laminar_flat_plate_transfer
else:
    from thermodynamics_models.common import ThermodynamicsError
    from thermodynamics_models.sle import (
        liquid_solution_activities, pure_solid_log_saturation_activity,
    )
    from unit_operations_basic import _ThermoStateSolver
    from transport_correlations import laminar_flat_plate_transfer


@dataclass(frozen=True)
class LayerGrowthResult:
    solid_amount_kmol: float
    thickness_m: float
    wall_energy_kJ: float
    profile: list[dict]
    evaluations: int
    bulk_temperature_K: float
    trapped_component_amounts_kmol: dict
    liquid_component_amounts_kmol: dict
    energy_residual_kJ: float


def _positive(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ThermodynamicsError(f'Layer growth {name} must be numeric') from error
    if not math.isfinite(value) or value <= 0:
        raise ThermodynamicsError(f'Layer growth {name} must be positive and finite')
    return value


def _wilke_chang(thermo, solute, solvent, T, P):
    """Infinite-dilution D [m2/s], DOI 10.1002/aic.690010222.

    Water-solute effective volume is 4x; association factors are 2.6/1.9/1.5
    for water/methanol/ethanol and 1 otherwise. Molecular-liquid estimate only.
    """
    props = thermo.props[solute]
    boiling = _positive(props.Tb, f'{solute} normal boiling temperature')
    volume = thermo.mixture_liquid_molar_volume({solute: 1.0}, boiling) * 1000
    if props.CAS == '7732-18-5' or solute.lower() == 'water':
        volume *= 4
    solvent_props = thermo.props[solvent]
    associations = {'7732-18-5': 2.6, '67-56-1': 1.9, '64-17-5': 1.5,
                    'water': 2.6, 'methanol': 1.9, 'ethanol': 1.5}
    association = associations.get(solvent_props.CAS, associations.get(solvent.lower(), 1.0))
    viscosity_cp = thermo._pure_viscosity(solvent, T, P, 'liquid') * 1000
    return _positive(7.4e-12 * math.sqrt(association * solvent_props.MW) * T
                     / (viscosity_cp * volume**0.6), 'Wilke-Chang diffusivity')


def solve_layer_growth(
    thermo, *, component, amounts_kmol, bulk_temperature_K, pressure_bar,
    wall_temperature_K, area_m2, growth_time_h, binary_diffusivity_m2_s=None,
    film_thickness_m=None, relative_tolerance=1e-6, profile_points=21,
    thermal_film_thickness_m=None,
    thermal_mode='isothermal', film_model='specified', plate_length_m=None,
    liquid_velocity_m_s=None, inclusion_max_fraction=0.0,
):
    """Grow an initially absent layer; amounts are per complete harvest cycle.

    Pseudo-binary film transport holds noncrystallizing-component ratios fixed,
    using supplied or estimated effective diffusivity and the liquid activity
    derivative along that composition path. Mechanical liquid capture bypasses this
    selective film; it is a bounded heuristic, not lattice incorporation.
    Cooling integrates bulk enthalpy and a deposition-enthalpy ledger; the
    existing solid layer has no transient sensible-heat redistribution.
    Solid density is fixed at the wall temperature; conductivity is integrated
    over the solid temperature range. The wall temperature is constant.
    """
    T = _positive(bulk_temperature_K, 'bulk temperature')
    initial_temperature = T
    P = _positive(pressure_bar, 'pressure')
    wall = _positive(wall_temperature_K, 'wall temperature')
    area = _positive(area_m2, 'area')
    if thermal_mode not in {'isothermal', 'cooling'}:
        raise ThermodynamicsError('Layer thermal_mode must be isothermal or cooling')
    if film_model not in {'specified', 'flat_plate'}:
        raise ThermodynamicsError('Layer film_model must be specified or flat_plate')
    if film_model == 'specified':
        film = _positive(film_thickness_m, 'film thickness')
        thermal_film = film if thermal_film_thickness_m is None else _positive(
            thermal_film_thickness_m, 'thermal film thickness'
        )
        if plate_length_m is not None or liquid_velocity_m_s is not None:
            raise ThermodynamicsError('Plate geometry/velocity requires film_model=flat_plate')
    else:
        if film_thickness_m is not None or thermal_film_thickness_m is not None:
            raise ThermodynamicsError('Flat-plate correlation cannot also specify film thickness')
        length = _positive(plate_length_m, 'plate length')
        velocity = _positive(liquid_velocity_m_s, 'liquid velocity')
        film = thermal_film = None
    specified_diffusivity = (None if binary_diffusivity_m2_s is None
                             else _positive(binary_diffusivity_m2_s, 'binary diffusivity'))
    inclusion_max_fraction = float(inclusion_max_fraction)
    if not math.isfinite(inclusion_max_fraction) or not 0 <= inclusion_max_fraction < 1:
        raise ThermodynamicsError('Layer inclusion_max_fraction must be in [0, 1)')
    duration = _positive(growth_time_h, 'growth time') * 3600
    tolerance = _positive(relative_tolerance, 'relative tolerance')
    if int(profile_points) != profile_points or profile_points < 2:
        raise ThermodynamicsError('Layer growth profile_points must be an integer >= 2')
    amounts = {c: float(n) for c, n in amounts_kmol.items()}
    if any(not math.isfinite(n) or n < 0 for n in amounts.values()):
        raise ThermodynamicsError('Layer growth amounts must be finite and nonnegative')
    present = [c for c, n in amounts.items() if n > 0]
    if len(present) < 2 or component not in present:
        raise ThermodynamicsError('Layer growth requires a crystallizing component and at least one other liquid component')
    unknown = set(present) - set(thermo.components)
    if unknown:
        raise ThermodynamicsError('Unknown layer liquid components: ' + ', '.join(sorted(unknown)))
    other_crystals = (set(present) & set(thermo.conventional_solid_components)) - {component}
    if other_crystals:
        raise ThermodynamicsError('Layer growth permits only one crystallizing component')
    melting = _positive(thermo.props[component].Tm, 'melting temperature')
    if thermal_mode == 'isothermal' and T >= melting:
        raise ThermodynamicsError('Layer bulk/product temperature must be below Tm')
    if wall >= min(T, melting):
        raise ThermodynamicsError('Layer wall temperature must be below bulk temperature and Tm')
    # Deliberately use the shared property path. Unsupported solid conductivity
    # is a real missing-property error, never replaced by liquid conductivity.
    thermo.pure_thermal_conductivity(component, wall, 'solid')
    solid_volume = _positive(thermo._solid_molar_volume(component, wall), 'solid volume')
    initial_a = amounts[component]
    solvent_amount = sum(amounts[c] for c in present if c != component)
    solvent_ratios = {c: amounts[c] / solvent_amount for c in present if c != component}
    initial_total = sum(amounts[c] for c in present)
    bulk_energy_index = 2 + len(present)
    layer_energy_index = bulk_energy_index + 1
    pore_volume_index = bulk_energy_index + 2
    state_solver = _ThermoStateSolver(thermo, 'Layer bulk cooling')

    def composition(x):
        return {component: x, **{c: ratio * (1 - x) for c, ratio in solvent_ratios.items()}}

    def log_activity(temperature, x):
        return math.log(max(liquid_solution_activities(
            thermo, temperature, P, composition(x)
        )[component], 1e-300))

    def saturation_x(temperature):
        target = pure_solid_log_saturation_activity(thermo, component, temperature, P)
        low, high = 1e-12, 1 - 1e-12
        f_low = log_activity(temperature, low) - target
        f_high = log_activity(temperature, high) - target
        if abs(f_high) < 1e-10:
            return high
        if f_low * f_high >= 0:
            raise ThermodynamicsError('Layer interface SLE has no bracketed liquid root along the solvent-blend path')
        return brentq(lambda x: log_activity(temperature, x) - target,
                      low, high, xtol=1e-13)

    wall_x = saturation_x(wall)

    def thermodynamic_factor(temperature, x):
        step = min(1e-5, x / 4, (1 - x) / 4)
        factor = x * (log_activity(temperature, x + step) - log_activity(temperature, x - step)) / (2 * step)
        if factor <= 0 or not math.isfinite(factor):
            raise ThermodynamicsError('Layer liquid film is thermodynamically unstable')
        return factor

    def effective_diffusivity(x, pairs):
        # Pairwise Vignes interpolation, then a solvent-fraction-weighted
        # harmonic blend. This is an effective-mixture closure, not a full
        # multicomponent Maxwell-Stefan matrix. It reduces exactly to binary.
        return 1 / sum(
            ratio / math.exp(
                (1 - x / (x + ratio * (1 - x))) * math.log(pairs[c][0])
                + x / (x + ratio * (1 - x)) * math.log(pairs[c][1])
            ) for c, ratio in solvent_ratios.items()
        )

    def mass_flux(x_interface, x_bulk, temperature, mass_film, pairs):
        # N_A = integral[c D_MS Gamma/(1-x_A) dx_A] / film thickness.
        span = x_bulk - x_interface
        if span == 0:
            return 0.0
        # Integrate on a fixed interval: near zero flux, interface and bulk
        # compositions can differ by only a few floating-point ulps.
        def integrand(position):
            x = x_interface + span * position
            gamma_factor = thermodynamic_factor(temperature, x)
            density = thermo.mixture_liquid_density(composition(x), temperature)
            diffusivity = effective_diffusivity(x, pairs)
            return density * diffusivity * gamma_factor / (1 - x)
        integral, error = quad(integrand, 0, 1,
                               epsabs=1e-18, epsrel=min(tolerance, 1e-7), limit=100)
        if error > max(1e-18, abs(integral) * min(tolerance, 1e-7)):
            raise ThermodynamicsError('Layer film mass-flux quadrature did not converge')
        return span * integral / mass_film

    def latent_heat(temperature, x):
        # Partial liquid molar enthalpy includes nonideal heat of mixing.
        step = min(1e-5, x / 4, (1 - x) / 4)
        def enthalpy(z):
            return thermo.mixture_enthalpy(composition(z), temperature, 0, P=P)
        partial = enthalpy(x) + (1 - x) * (enthalpy(x + step) - enthalpy(x - step)) / (2 * step)
        return _positive(partial - 1000 * thermo.process_solid_enthalpy(component, temperature), 'interfacial latent heat')

    initial_enthalpy = initial_total * thermo.mixture_enthalpy(
        composition(initial_a / initial_total), T, 0, P=P
    )

    def interface(state):
        solid_amount = float(state[0]) * initial_a
        remaining = {c: amounts[c] - state[2+i] * initial_total
                     - (solid_amount if c == component else 0)
                     for i, c in enumerate(present)}
        remaining_a = remaining[component]
        if min(remaining.values()) <= 0 or solid_amount < -1e-9 * initial_a:
            raise ThermodynamicsError('Layer integration left the feasible material inventory')
        solid_amount = max(solid_amount, 0.0)
        remaining_total = sum(remaining.values())
        x_bulk = remaining_a / remaining_total
        bulk_x = composition(x_bulk)
        T = initial_temperature
        if thermal_mode == 'cooling':
            target = state[bulk_energy_index] / remaining_total
            bulk, error = state_solver.state_at_enthalpy(
                P, remaining_total, bulk_x, target, initial_temperature,
                force_phase='liquid', include=('H',),
            )
            if abs(error) > max(1e-5, abs(target) * 1e-9):
                raise ThermodynamicsError('Layer bulk enthalpy inversion did not converge')
            T = bulk.T
        if not math.isfinite(T) or T < wall - 1e-6:
            raise ThermodynamicsError('Layer bulk temperature fell below the cold wall')
        liquid_k = thermo.mixture_liquid_thermal_conductivity(bulk_x, T)
        density = thermo.mixture_liquid_density(bulk_x, T)
        if specified_diffusivity is None:
            pairs = {c: (_wilke_chang(thermo, component, c, T, P),
                         _wilke_chang(thermo, c, component, T, P)) for c in solvent_ratios}
        else:
            pairs = {c: (specified_diffusivity, specified_diffusivity) for c in solvent_ratios}
        # Schmidt number uses Fick diffusivity; the flux integral applies the
        # thermodynamic factor locally to the Maxwell-Stefan coefficient.
        bulk_diffusivity = effective_diffusivity(x_bulk, pairs)
        fick_diffusivity = bulk_diffusivity * thermodynamic_factor(T, x_bulk)
        transfer = {}
        mass_film, heat_film = film, thermal_film
        if film_model == 'flat_plate':
            mw = sum(bulk_x[c] * thermo.props[c].MW for c in bulk_x)
            rho = density * mw
            mu = thermo.mixture_viscosity(bulk_x, T, P, 0)
            step = 1e-3
            cp = (thermo.mixture_enthalpy(bulk_x, T + step, 0, P=P)
                  - thermo.mixture_enthalpy(bulk_x, T - step, 0, P=P)) / (2 * step) * 1000 / mw
            transfer = laminar_flat_plate_transfer(
                rho * velocity * length / mu, cp * mu / liquid_k, mu / (rho * fick_diffusivity)
            )
            mass_film = length / transfer['sherwood']
            heat_film = length / transfer['nusselt']
        heat_coefficient = liquid_k / heat_film
        pore_volume = max(0.0, state[pore_volume_index])
        volume = solid_amount * solid_volume + pore_volume
        thickness = volume / area
        porosity = pore_volume / volume if volume > 0 else 0.0
        common = dict(bulk_temperature_K=T, bulk_mole_fraction=x_bulk,
                      film_thickness_m=mass_film, thermal_film_thickness_m=heat_film,
                      remaining_component_amounts_kmol=remaining,
                      porosity=porosity, bulk_maxwell_stefan_diffusivity_m2_s=bulk_diffusivity,
                      bulk_fick_diffusivity_m2_s=fick_diffusivity,
                      bulk_enthalpy_kJ=remaining_total * thermo.mixture_enthalpy(bulk_x, T, 0, P=P),
                      **transfer)
        if solid_amount == 0 and x_bulk <= wall_x:
            return dict(flux=0.0, interface_temperature_K=wall,
                        interface_mole_fraction=x_bulk,
                        wall_heat_flux_W_m2=heat_coefficient * (T - wall),
                        liquid_heat_flux_W_m2=heat_coefficient * (T - wall),
                        trapped_liquid_rate_kmol_s=0.0, trapped_volume_rate_m3_s=0.0,
                        inclusion_fraction=0.0, partial_liquid_enthalpy_kJ_kmol=0.0,
                        solid_enthalpy_kJ_kmol=0.0,
                        thickness_m=0.0, sle_residual=None, stefan_residual_W_m2=0.0, **common)

        def evaluate(temperature):
            x = saturation_x(temperature)
            flux = mass_flux(x, x_bulk, T, mass_film, pairs)
            latent = latent_heat(temperature, x)
            heat = heat_coefficient * (T - temperature) + 1000 * flux * latent
            # Integrating Fourier's law gives q*thickness = integral(k(T), dT).
            # A fixed integration interval remains well conditioned as Ti -> Tw.
            solid_k, error = quad(
                lambda position: 1 / (
                    (1 - porosity) / thermo.pure_thermal_conductivity(
                        component, wall + (temperature - wall) * position, 'solid'
                    ) + porosity / liquid_k
                ), 0, 1, epsabs=1e-12, epsrel=min(tolerance, 1e-7),
            )
            _positive(solid_k, 'integrated solid conductivity')
            if error > max(1e-12, solid_k * min(tolerance, 1e-7)):
                raise ThermodynamicsError('Layer solid-conductivity quadrature did not converge')
            return temperature - wall - thickness * heat / solid_k, x, flux, heat, solid_k

        if thickness == 0:
            temperature = wall
        else:
            liquidus = brentq(
                lambda temperature: log_activity(temperature, x_bulk)
                - pure_solid_log_saturation_activity(thermo, component, temperature, P),
                wall, melting, xtol=1e-10,
            )
            temperature = brentq(lambda temperature: evaluate(temperature)[0],
                                 wall, min(max(T, liquidus), melting - 1e-7), xtol=1e-9)
        residual, x, flux, heat, solid_k = evaluate(temperature)
        if flux < -1e-12:
            raise ThermodynamicsError('Layer remelting is outside the growth/inclusion model')
        flux = max(0.0, flux)
        growth_peclet = flux * solid_volume * mass_film / fick_diffusivity
        fraction = inclusion_max_fraction * growth_peclet / (1 + growth_peclet)
        pore_rate = area * flux * solid_volume * fraction / (1 - fraction)
        h_solid = 1000 * thermo.process_solid_enthalpy(component, temperature)
        return dict(flux=flux, interface_temperature_K=temperature,
                    interface_mole_fraction=x,
                    wall_heat_flux_W_m2=heat, thickness_m=thickness,
                    liquid_heat_flux_W_m2=heat_coefficient * (T - temperature),
                    trapped_liquid_rate_kmol_s=pore_rate * density,
                    trapped_volume_rate_m3_s=pore_rate, inclusion_fraction=fraction,
                    partial_liquid_enthalpy_kJ_kmol=latent_heat(temperature, x) + h_solid,
                    solid_enthalpy_kJ_kmol=h_solid,
                    sle_residual=log_activity(temperature, x)
                    - pure_solid_log_saturation_activity(thermo, component, temperature, P),
                    stefan_residual_W_m2=(residual * solid_k / thickness if thickness else 0.0), **common)

    def rhs(_time, state):
        values = interface(state)
        solid_rate = area * values['flux']
        trap_rate = values['trapped_liquid_rate_kmol_s']
        x = values['bulk_mole_fraction']
        trap_energy = trap_rate * thermo.mixture_enthalpy(
            composition(x), values['bulk_temperature_K'], 0, P=P
        )
        bulk_rate = (-area * values['liquid_heat_flux_W_m2'] / 1000
                     - solid_rate * values['partial_liquid_enthalpy_kJ_kmol'] - trap_energy)
        layer_rate = solid_rate * values['solid_enthalpy_kJ_kmol'] + trap_energy
        return [solid_rate / initial_a, -area * values['wall_heat_flux_W_m2'] / 1000,
                *(trap_rate * composition(x)[c] / initial_total for c in present),
                bulk_rate, layer_rate, values['trapped_volume_rate_m3_s']]

    solved = solve_ivp(
        rhs, (0, duration), [0.0, 0.0, *([0.0] * len(present)), initial_enthalpy, 0.0, 0.0], rtol=tolerance,
        atol=[tolerance * 1e-3, tolerance, *([tolerance * 1e-3] * len(present)),
              tolerance, tolerance, tolerance * 1e-6],
        t_eval=np.linspace(0, duration, int(profile_points)),
        max_step=duration / 20,
    )
    if not solved.success:
        raise ThermodynamicsError(f'Layer integration failed: {solved.message}')
    profile = []
    for time, state in zip(solved.t, solved.y.T, strict=True):
        values = interface(state)
        profile.append({'time_h': float(time) / 3600,
                        'solid_amount_kmol': float(state[0]) * initial_a,
                        'wall_energy_kJ': float(state[1]),
                        'bulk_energy_ledger_kJ': float(state[bulk_energy_index]),
                        'deposition_enthalpy_kJ': float(state[layer_energy_index]),
                        'temperature_control_energy_kJ': values['bulk_enthalpy_kJ'] - float(state[bulk_energy_index]),
                        'energy_residual_kJ': float(state[bulk_energy_index] + state[layer_energy_index] - initial_enthalpy - state[1]),
                        'trapped_component_amounts_kmol': {c: float(state[2+i]) * initial_total
                                                           for i, c in enumerate(present)},
                        **values})
    last = profile[-1]
    return LayerGrowthResult(last['solid_amount_kmol'], last['thickness_m'],
                             last['wall_energy_kJ'], profile, solved.nfev,
                             last['bulk_temperature_K'], last['trapped_component_amounts_kmol'],
                             last['remaining_component_amounts_kmol'], last['energy_residual_kJ'])
