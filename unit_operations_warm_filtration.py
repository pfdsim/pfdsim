"""PFD adapter for equilibrium warm washing of a stationary filter cake."""

from __future__ import annotations

import math

from scipy.optimize import brentq

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from .equilibrium_washing import (
        collected_population,
        equilibrate_enthalpy,
        inventory_at_temperature,
        solve_warm_wash,
    )
    from .filtration_models import deliquor, required_area
    from .unit_operations_base import UnitOperationError, UnitResult
else:
    from equilibrium_washing import (
        collected_population,
        equilibrate_enthalpy,
        inventory_at_temperature,
        solve_warm_wash,
    )
    from filtration_models import deliquor, required_area
    from unit_operations_base import UnitOperationError, UnitResult


def solve_equilibrium_filter(unit, inlets):
    thermo = unit.thermo
    feed = inlets["in"]
    feed_liquid = unit._liquid_flows(feed, "feed")
    wash = inlets.get("wash")
    if wash is None:
        raise UnitOperationError("Equilibrium washing requires a wash inlet")
    wash_flows = unit._liquid_flows(wash, "wash")
    if wash.solid_component_flows:
        raise UnitOperationError("Equilibrium wash inlet must be solids-free")
    if wash.P < feed.P - 1e-8:
        raise UnitOperationError("Wash pressure must be at least slurry feed pressure")
    if unit.get_param("wash_viscosity") is not None:
        raise UnitOperationError(
            "Equilibrium washing resolves local viscosity; wash_viscosity is not a constant override in this mode"
        )
    cycle = unit._number("cycle_time", kind="time")
    batch = cycle / 3600
    downtime = unit._number("downtime", 0, kind="time", zero=True)
    drain_time = unit._number("deliquoring_time", 0, kind="time", zero=True)
    available = cycle - downtime - drain_time
    if available <= 0:
        raise UnitOperationError("cycle_time must exceed downtime + deliquoring_time")
    cells = unit._number("wash_cells", 5)
    steps = unit._number("wash_steps", 100)
    if (
        cells != int(cells)
        or not 1 <= cells <= 100
        or steps != int(steps)
        or not 1 <= steps <= 10000
    ):
        raise UnitOperationError(
            "Equilibrium washing requires integer wash_cells (1–100) and wash_steps (1–10000)"
        )
    cells, steps = int(cells), int(steps)
    tolerance = unit._number("equilibrium_tolerance", 1e-8)
    if not 1e-12 <= tolerance <= 1e-5:
        raise UnitOperationError("equilibrium_tolerance must be between 1e-12 and 1e-5")
    tmin = (
        unit.get_temperature_param("T_equilibrium_min")
        if unit.get_param("T_equilibrium_min") is not None
        else max(1, min(feed.T, wash.T) - 25)
    )
    tmax = (
        unit.get_temperature_param("T_equilibrium_max")
        if unit.get_param("T_equilibrium_max") is not None
        else max(feed.T, wash.T) + 25
    )
    if (
        not all(math.isfinite(t) and t > 0 for t in (tmin, tmax))
        or tmin >= min(feed.T, wash.T)
        or tmax <= max(feed.T, wash.T)
    ):
        raise UnitOperationError(
            "Equilibrium temperature bounds must bracket feed and wash temperatures"
        )
    bounds = (tmin, tmax)
    nucleus = (
        unit._number("equilibrium_nucleus_diameter", kind="length")
        if unit.get_param("equilibrium_nucleus_diameter") is not None
        else None
    )
    porosity = unit._fraction("porosity", upper_open=True)
    medium = unit._number("medium_resistance", 0, kind="resistance", zero=True)
    wash_batch = {c: n * batch for c, n in wash_flows.items()}
    if wash.H is None or not math.isfinite(wash.H):
        raise UnitOperationError("Equilibrium washing requires wash enthalpy")
    wash_energy = wash.F * wash.H * batch
    area_spec = (
        unit._number("area", kind="area")
        if unit.get_param("area") is not None
        else None
    )
    dp_spec = (
        unit._number("P_drop", kind="pressure")
        if unit.get_param("P_drop") is not None
        else None
    )
    if area_spec is None and dp_spec is None:
        raise UnitOperationError("Equilibrium Filter requires area or P_drop")

    # Reuse the existing capture/formation calculation, including its property,
    # PSD and port validation. Unit area exposes the formation time coefficients.
    removed = {
        "washing_model",
        "wash_cells",
        "wash_steps",
        "wash_viscosity",
        "deliquoring_time",
        "deliquoring_pressure",
        "entry_pressure",
        "residual_saturation",
        "pore_index",
        "relative_permeability_exponent",
        "equilibrium_tolerance",
        "t_equilibrium_min",
        "t_equilibrium_max",
        "equilibrium_nucleus_diameter",
        "area",
        "p_drop",
    }
    formation_params = {
        k: v
        for k, v in unit.params.items()
        if str(k).lower().removeprefix("__unit__") not in removed
    }
    formation_params["area"] = 1.0
    cache = {}

    class LiquidRegionError(UnitOperationError):
        pass

    def validate_temperature_liquid(temperature, liquid, pressure):
        liquid_state = thermo.calculate_state(
            temperature,
            pressure,
            1.0,
            unit._composition(liquid),
            include=("H",),
        )
        try:
            unit._liquid_flows(liquid_state, "local equilibrium wash liquid")
        except UnitOperationError as error:
            raise LiquidRegionError(str(error)) from error

    def trial(drop):
        if drop in cache:
            return cache[drop]
        if drop <= 0 or drop >= feed.P:
            raise UnitOperationError(
                "Filtration pressure drop must leave positive outlet pressure"
            )
        pressure = feed.P - drop
        validate_temperature_liquid(feed.T, feed_liquid, pressure)
        formed = type(unit)(
            unit.unit_id, thermo, formation_params | {"P_drop": drop}
        ).solve({"in": feed})
        cake = formed.outlet_streams["cake"]
        phases = cake.phase_component_flows()
        initial = inventory_at_temperature(
            thermo,
            feed.T,
            pressure,
            {c: n * batch for c, n in phases["liquid1"].items()},
            {c: n * batch for c, n in phases["solid"].items()},
        )
        psds = {
            c: p.scaled(batch)
            for c, p in cake.solid_particle_size_distributions.items()
        }
        shapes = {}
        for c in initial.solid:
            if (
                c not in psds
                or cake.solid_particle_properties.get(c, {}).get("sphericity") is None
            ):
                raise UnitOperationError(
                    f"Equilibrium washing requires PSD and sphericity for {c}"
                )
            shapes[c] = cake.solid_particle_properties[c]["sphericity"]
        for c in thermo.conventional_solid_components:
            if c not in shapes:
                shapes[c] = thermo.solid_particle_defaults.get(c, {}).get(
                    "sphericity", 1.0
                )
        surface = sum(
            6 * n * thermo._solid_molar_volume(c, feed.T) / (shapes[c] * d)
            for c, psd in psds.items()
            for d, n in zip(psd.diameters_m, psd.molar_flows_kmol_per_h)
        )
        bed_volume = initial.solid_volume / (1 - porosity)
        geometric = surface**2 / (bed_volume * porosity**3)
        kozeny = formed.performance["cake_resistance_per_m"] / geometric

        def validate_liquid(state):
            validate_temperature_liquid(state.temperature, state.liquid, pressure)

        washed = solve_warm_wash(
            thermo,
            initial,
            psds,
            wash_batch,
            wash_energy,
            wash.T,
            pressure,
            cells=cells,
            steps=steps,
            porosity=porosity,
            sphericities=shapes,
            kozeny=kozeny,
            medium_resistance=medium,
            pressure_drop=drop * 1e5,
            temperature_bounds=bounds,
            tolerance=tolerance,
            nucleus_diameter=nucleus,
            validate_liquid=validate_liquid,
        )
        # The primary formation is unchanged: separate cake and medium terms.
        vf = formed.performance["primary_filtrate_m3_per_cycle"]
        mu = formed.performance["liquid_viscosity_Pa_s"]
        form_a = (
            mu * formed.performance["cake_resistance_per_m"] * vf / (2 * drop * 1e5)
        )
        form_b = mu * medium * vf / (drop * 1e5)
        a, b = (
            form_a + washed.time_coefficients[0],
            form_b + washed.time_coefficients[1],
        )
        cache[drop] = (formed, initial, washed, a, b, form_a, form_b)
        return cache[drop]

    if dp_spec is None:

        def time_residual(drop):
            result = trial(drop)
            return result[3] / area_spec**2 + result[4] / area_spec - available

        low = min(feed.P / 10, 0.1)
        while True:
            try:
                if time_residual(low) >= 0:
                    break
            except LiquidRegionError:
                # A nearly saturated feed may only permit a smaller drop
                # than the initial trial, even with ample installed area.
                pass
            low /= 10
            if low < 1e-10:
                raise UnitOperationError(
                    "Required washing pressure is below numerical resolution"
                )
        high = min(low * 2, (low + feed.P) / 2)
        boundary = feed.P
        for _ in range(40):
            try:
                high_residual = time_residual(high)
            except LiquidRegionError:
                boundary = high
                high = (low + boundary) / 2
                continue
            if high_residual <= 0:
                break
            low = high
            high = min(2 * low, (low + boundary) / 2)
            if boundary - low < max(1e-8, feed.P * 1e-8):
                raise UnitOperationError(
                    "Required warm-washing pressure exceeds the liquid-only operating range; increase area or feed pressure"
                )
        else:
            raise UnitOperationError(
                "Could not bracket washing pressure within the liquid-only operating range"
            )
        dp_spec = brentq(time_residual, low, high, xtol=1e-8, rtol=1e-8)
    formed, initial, washed, a, b, form_a, form_b = trial(dp_spec)
    pressure = feed.P - dp_spec
    size = required_area(a, b, available)
    area = area_spec or size
    saturation = equilibrium = 1.0
    if drain_time > 0:
        entry = unit._number("entry_pressure", kind="pressure")
        residual = unit._fraction("residual_saturation", upper_open=True)
        pore_index = unit._number("pore_index")
        exponent = unit._number("relative_permeability_exponent", 3 + 2 / pore_index)
        saturation, equilibrium = deliquor(
            duration=drain_time,
            pore_volume=sum(c.liquid_volume for c in washed.cells),
            area=area,
            cake_resistance=sum(
                r * mu
                for r, mu in zip(washed.final_resistance_area, washed.final_viscosity)
            )
            / area,
            medium_resistance=medium * washed.final_viscosity[-1],
            viscosity=1.0,
            pressure_drop=unit._number("deliquoring_pressure", dp_spec, kind="pressure")
            * 1e5,
            entry_pressure=entry * 1e5,
            residual_saturation=residual,
            pore_index=pore_index,
            relative_permeability_exponent=exponent,
        )
    elif any(
        unit.get_param(n) is not None
        for n in (
            "entry_pressure",
            "residual_saturation",
            "pore_index",
            "deliquoring_pressure",
            "relative_permeability_exponent",
        )
    ):
        raise UnitOperationError(
            "Deliquoring parameters require positive deliquoring_time"
        )
    cake_solids, cake_liquid, distributions = {}, {}, {}
    cake_energy = 0.0
    filtrate = formed.outlet_streams["filtrate"]
    filtrate_flows = {c: n * batch for c, n in filtrate.component_flows().items()}
    filtrate_energy = filtrate.F * filtrate.H * batch + washed.effluent_enthalpy
    for c, n in washed.effluent.items():
        filtrate_flows[c] = filtrate_flows.get(c, 0) + n
    for state, psds in zip(washed.cells, washed.distributions):
        drained_energy = (
            (1 - saturation) * sum(state.liquid.values()) * state.liquid_enthalpy
        )
        cake_energy += state.enthalpy - drained_energy
        filtrate_energy += drained_energy
        for c, n in state.liquid.items():
            cake_liquid[c] = cake_liquid.get(c, 0) + n * saturation
            filtrate_flows[c] = filtrate_flows.get(c, 0) + n * (1 - saturation)
        for c, n in state.solid.items():
            cake_solids[c] = cake_solids.get(c, 0) + n
        for c, psd in psds.items():
            distributions.setdefault(c, []).append((psd, state.temperature))

    def outlet(flows, solids, energy, populations):
        equilibrated = equilibrate_enthalpy(
            thermo,
            flows,
            energy,
            pressure,
            initial_temperature=feed.T,
            initial_solids=solids,
            temperature_bounds=bounds,
            tolerance=tolerance,
        )
        temperature, solids = equilibrated.temperature, equilibrated.solid
        populations = collected_population(thermo, equilibrated, populations, nucleus)
        total = sum(flows.values()) / batch
        state = thermo.calculate_state_with_solid_flows(
            temperature,
            pressure,
            total,
            {c: n / (total * batch) for c, n in flows.items() if n > 0},
            {
                c: n / batch
                for c, n in solids.items()
                if c in thermo.conventional_solid_components and n > 0
            },
        )
        unit._liquid_flows(state, "warm-washing outlet")
        state.solid_particle_size_distributions = {
            c: psd.scaled(1 / batch) for c, psd in populations.items()
        }
        state.validate_particle_size_distributions()
        state.phase_details["equilibrium_washing"] = {
            "collection_model": "adiabatic_pure_solid_equilibrium",
            "equilibrium_residual": equilibrated.residual,
        }
        return state

    cake_flows = {
        c: cake_solids.get(c, 0) + cake_liquid.get(c, 0)
        for c in set(cake_solids) | set(cake_liquid)
    }
    cake = outlet(cake_flows, cake_solids, cake_energy, distributions)
    escaped = {c: n * batch for c, n in filtrate.solid_component_flows.items()}
    filtrate = outlet(
        filtrate_flows,
        escaped,
        filtrate_energy,
        {
            c: [(psd.scaled(batch), feed.T)]
            for c, psd in filtrate.solid_particle_size_distributions.items()
        },
    )
    for stream in (cake, filtrate):
        for c in stream.solid_component_flows:
            source = feed.solid_particle_properties.get(
                c, thermo.solid_particle_defaults.get(c, {})
            )
            stream.solid_particle_properties[c] = {
                k: v for k, v in source.items() if k in {"diameter_m", "sphericity"}
            }
            stream.solid_particle_properties[c].setdefault("sphericity", 1.0)
            psd = stream.solid_particle_size_distributions.get(c)
            if psd is not None:
                stream.solid_particle_properties[c]["diameter_m"] = (
                    psd.sauter_mean_diameter_m
                )
    energy_residual = (
        sum(s.F * s.H for s in (cake, filtrate))
        - sum(s.F * s.H for s in inlets.values())
        - formed.heat_duty
    )
    # Use the flash energy normalization, independent of the arbitrary
    # formation-enthalpy reference, and account for accumulated contact error.
    scale = max(sum(s.F for s in inlets.values()) * 1e4, 1)
    if abs(energy_residual) > scale * max(1e-7, tolerance * (steps + 3)):
        raise UnitOperationError(
            f"Warm washing failed its energy balance: {energy_residual:g} kJ/h"
        )
    collected_solid_change = {
        c: cake.solid_component_flows.get(c, 0)
        + filtrate.solid_component_flows.get(c, 0)
        - (cake_solids.get(c, 0) + escaped.get(c, 0)) / batch
        for c in set(cake.solid_component_flows)
        | set(filtrate.solid_component_flows)
        | set(cake_solids)
        | set(escaped)
    }
    dry_mass = sum(
        n * thermo.props[c].MW for c, n in cake.solid_component_flows.items()
    )
    wet_mass = sum(
        n * thermo.props[c].MW
        for c, n in cake.phase_component_flows()["liquid1"].items()
    )
    form_time = form_a / area**2 + form_b / area
    wash_time = (
        washed.time_coefficients[0] / area**2 + washed.time_coefficients[1] / area
    )
    feasible = form_time + wash_time <= available * (1 + 1e-7)
    performance = {
        "model": "equilibrium_warm_cake_filter",
        "washing_model": "equilibrium",
        "mode": "pressure"
        if unit.get_param("P_drop") is None
        else ("sizing" if area_spec is None else "rating"),
        "area_m2": area,
        "required_area_m2": size,
        "capacity_ratio": area / size,
        "pressure_drop_bar": dp_spec,
        "P_out_bar": pressure,
        "cycle_feasible": feasible,
        "cycle_time_s": cycle,
        "formation_time_s": form_time,
        "washing_time_s": wash_time,
        "deliquoring_time_s": drain_time,
        "downtime_s": downtime,
        "required_cycle_time_s": form_time + wash_time + drain_time + downtime,
        "wash_cells": cells,
        "wash_steps": steps,
        "cell_temperatures_K": [c.temperature for c in washed.cells],
        "cell_porosity": washed.final_porosity,
        "cell_resistances_per_m": [r / area for r in washed.final_resistance_area],
        "cell_viscosities_Pa_s": washed.final_viscosity,
        "cell_solid_inventories_kmol": [c.solid for c in washed.cells],
        "cell_liquid_compositions": [unit._composition(c.liquid) for c in washed.cells],
        "initial_cake_resistance_per_m": formed.performance["cake_resistance_per_m"]
        / area,
        "final_cake_resistance_per_m": sum(washed.final_resistance_area) / area,
        "net_solid_change_kmol_h": {
            c: cake.solid_component_flows.get(c, 0) - initial.solid.get(c, 0) / batch
            for c in set(cake.solid_component_flows) | set(initial.solid)
        },
        "collection_solid_change_kmol_h": collected_solid_change,
        "solid_capture_mass_fraction": formed.performance[
            "solid_capture_mass_fraction"
        ],
        "cake_saturation": saturation,
        "equilibrium_saturation": equilibrium,
        "dry_solids_kg_per_h": dry_mass,
        "retained_liquid_kg_per_h": wet_mass,
        "cake_moisture_mass_fraction": wet_mass / (dry_mass + wet_mass),
        "maximum_equilibrium_residual": washed.maximum_equilibrium_residual,
        "washing_component_residual_kmol": washed.component_residual,
        "washing_energy_residual_kJ": washed.energy_residual,
        "energy_balance_residual_kJ_h": energy_residual,
        "duty_kW": formed.heat_duty / 3600,
    }
    warnings = (
        []
        if feasible
        else [
            "Specified area cannot complete formation and equilibrium washing within cycle_time."
        ]
    )
    return UnitResult(
        {"cake": cake, "filtrate": filtrate},
        heat_duty=formed.heat_duty,
        performance=performance,
        warnings=warnings,
    )
