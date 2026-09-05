"""Mechanistic pressure cake filtration with optional washing and deliquoring."""

from __future__ import annotations

import math
from typing import ClassVar

from scipy.optimize import brentq

if __package__ and __package__.split(".", 1)[0] == "pfdsim":
    from .filtration_models import (
        cycle_coefficients,
        deliquor,
        required_area,
        wash_profile,
    )
    from .particle_size_distributions import ParticleSizeDistribution
    from .unit_operations_base import UnitOperation, UnitOperationError, UnitResult
else:
    from filtration_models import (
        cycle_coefficients,
        deliquor,
        required_area,
        wash_profile,
    )
    from particle_size_distributions import ParticleSizeDistribution
    from unit_operations_base import UnitOperation, UnitOperationError, UnitResult


class Filter(UnitOperation):
    """Cycle-averaged, size-selective cake filter with optional warm washing.

    See docs/filtration.md for specifications and constitutive limitations.
    Pressure parameters follow pfdsim's internal bar convention; all other
    dimensional parameters are converted here using their original unit tags.
    """

    supports_permanent_solids = True
    particle_size_behavior = "custom"

    _UNITS: ClassVar[dict] = {
        "time": {"s": 1, "sec": 1, "min": 60, "h": 3600, "hr": 3600},
        "area": {"m2": 1, "m^2": 1, "cm2": 1e-4, "ft2": 0.09290304},
        "alpha": {"m/kg": 1},
        "resistance": {"1/m": 1, "m^-1": 1, "m-1": 1},
        "viscosity": {"pa*s": 1, "pa.s": 1, "cp": 1e-3, "mpa*s": 1e-3},
        "fraction": {"1": 1, "": 1},
        "length": {"m": 1, "mm": 1e-3, "um": 1e-6, "µm": 1e-6},
    }

    def _solid_split(self, feed):
        """Split each extensive particle-size class and compute retained area.

        Capture uses projected-area equivalent diameter d/sqrt(sphericity).
        Hydraulic surface uses actual surface per particle volume, 6/(phi*d).
        """
        cut = self._number("capture_cut_size", kind="length", zero=True)
        sharpness = self._number("capture_sharpness", 4)
        derived = self.get_param("specific_cake_resistance") is None
        captured, escaped, cake_psd, filtrate_psd = {}, {}, {}, {}
        surface_rate = 0.0
        for component, flow in feed.solid_component_flows.items():
            if flow <= 0:
                continue
            psd = feed.solid_particle_size_distributions.get(component)
            props = feed.solid_particle_properties.get(component, {})
            if psd is None and props.get("diameter_m") is not None:
                psd = ParticleSizeDistribution((props["diameter_m"],), (flow,))
            if (cut > 0 or derived) and (
                psd is None or props.get("sphericity") is None
            ):
                raise UnitOperationError(
                    f"Filter '{self.unit_id}' requires PSD (or particle diameter) and sphericity for {component}"
                )
            if psd is None:
                captured[component] = flow
                continue
            if not math.isclose(
                psd.total_molar_flow, flow, rel_tol=1e-8, abs_tol=1e-12
            ):
                raise UnitOperationError(
                    f"Filter '{self.unit_id}' PSD inventory differs from solid flow for {component}"
                )
            phi = float(props.get("sphericity", 1))
            if not math.isfinite(phi) or not 0 < phi <= 1:
                raise UnitOperationError(
                    f"Filter '{self.unit_id}' invalid sphericity for {component}"
                )
            cake_classes, filtrate_classes = [], []
            for diameter, class_flow in zip(
                psd.diameters_m, psd.molar_flows_kmol_per_h
            ):
                if cut == 0:
                    efficiency = 1.0
                else:
                    log_odds = sharpness * (
                        math.log(diameter / math.sqrt(phi)) - math.log(cut)
                    )
                    if log_odds >= 0:
                        efficiency = 1 / (1 + math.exp(-log_odds))
                    else:
                        odds = math.exp(log_odds)
                        efficiency = odds / (1 + odds)
                retained = class_flow * efficiency
                cake_classes.append(retained)
                filtrate_classes.append(class_flow - retained)
                if derived:
                    surface_rate += (
                        retained
                        * self.thermo._solid_molar_volume(component, feed.T)
                        * 6
                        / (phi * diameter)
                    )
            captured[component] = sum(cake_classes)
            escaped[component] = sum(filtrate_classes)
            if captured[component] > 0:
                cake_psd[component] = ParticleSizeDistribution(
                    psd.diameters_m, tuple(cake_classes)
                )
            if escaped[component] > 0:
                filtrate_psd[component] = ParticleSizeDistribution(
                    psd.diameters_m, tuple(filtrate_classes)
                )
        return captured, escaped, cake_psd, filtrate_psd, surface_rate

    def _number(self, name, default=None, *, kind="fraction", zero=False):
        raw = self.get_param(name, default)
        if raw is None:
            raise UnitOperationError(f"Filter '{self.unit_id}' requires {name}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' {name} must be numeric"
            ) from error
        # Pressure unit tags have already been converted by FlowsheetSolver.
        if kind != "pressure":
            units = self._UNITS[kind]
            tag = (
                str(self.get_param_unit(name) or next(iter(units)))
                .lower()
                .replace(" ", "")
            )
            if tag not in units:
                raise UnitOperationError(
                    f"Filter '{self.unit_id}' invalid unit for {name}: {tag}"
                )
            value *= units[tag]
        if not math.isfinite(value) or (value < 0 if zero else value <= 0):
            raise UnitOperationError(
                f"Filter '{self.unit_id}' {name} must be finite and "
                f"{'nonnegative' if zero else 'positive'}"
            )
        return value

    def _fraction(self, name, default=None, *, upper_open=False):
        value = self._number(name, default, zero=True)
        if value > 1 or (upper_open and value == 1):
            raise UnitOperationError(
                f"Filter '{self.unit_id}' {name} must be below {'1' if upper_open else 'or equal to 1'}"
            )
        return value

    def _liquid_flows(self, stream, label):
        if not math.isfinite(stream.F) or stream.F <= 0:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' {label} requires positive finite flow"
            )
        phases = stream.phase_component_flows()
        tolerance = 1e-10 * stream.F
        if (
            sum(phases["vapor"].values()) > tolerance
            or sum(phases["liquid2"].values()) > tolerance
        ):
            raise UnitOperationError(
                f"Filter '{self.unit_id}' {label} requires one liquid phase and no vapor"
            )
        liquid = {c: f for c, f in phases["liquid1"].items() if f > 0}
        if not liquid:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' {label} contains no liquid"
            )
        return liquid

    @staticmethod
    def _composition(flows):
        total = sum(flows.values())
        return {c: f / total for c, f in flows.items() if f > 0}

    def solve(self, inlets):
        if "in" not in inlets or set(inlets) - {"in", "wash"}:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' requires in and optional wash inlet ports"
            )
        connected = set(self.get_param("__connected_outlet_ports__", ()) or ())
        if connected and connected != {"cake", "filtrate"}:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' requires connected cake and filtrate outlets"
            )
        allowed = {
            "cycle_time",
            "downtime",
            "area",
            "porosity",
            "specific_cake_resistance",
            "medium_resistance",
            "compressibility",
            "reference_pressure",
            "p_drop",
            "liquid_viscosity",
            "wash_viscosity",
            "wash_cells",
            "deliquoring_time",
            "deliquoring_pressure",
            "entry_pressure",
            "residual_saturation",
            "pore_index",
            "relative_permeability_exponent",
            "capture_cut_size",
            "capture_sharpness",
            "kozeny_constant",
            "washing_model", "wash_steps", "equilibrium_tolerance",
            "t_equilibrium_min", "t_equilibrium_max",
            "equilibrium_nucleus_diameter",
        }
        unknown = [
            str(k)
            for k in self.params
            if not str(k).startswith("__") and str(k).lower() not in allowed
        ]
        if unknown:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' unknown parameter(s): {', '.join(unknown)}"
            )
        washing_model = str(self.get_param('washing_model', 'isothermal')).lower()
        if washing_model not in {'isothermal', 'equilibrium'}:
            raise UnitOperationError('washing_model must be isothermal or equilibrium')
        if washing_model == 'equilibrium':
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .thermodynamics_models.common import ThermodynamicsError
                from .unit_operations_warm_filtration import solve_equilibrium_filter
            else:
                from thermodynamics_models.common import ThermodynamicsError
                from unit_operations_warm_filtration import solve_equilibrium_filter
            try:
                return solve_equilibrium_filter(self, inlets)
            except ThermodynamicsError as error:
                raise UnitOperationError(f"Filter '{self.unit_id}' equilibrium washing failed: {error}") from error
        if any(str(k).lower().startswith(('equilibrium_', 't_equilibrium_')) or str(k).lower() == 'wash_steps' for k in self.params):
            raise UnitOperationError('Equilibrium washing parameters require washing_model=equilibrium')
        feed = inlets["in"]
        liquid = self._liquid_flows(feed, "feed")
        if not any(f > 0 for f in feed.solid_component_flows.values()):
            raise UnitOperationError(
                f"Filter '{self.unit_id}' requires suspended solids"
            )
        solids, escaped, cake_psd, filtrate_psd, surface_rate = self._solid_split(feed)
        cycle = self._number("cycle_time", kind="time")
        downtime = self._number("downtime", 0, kind="time", zero=True)
        drain_time = self._number("deliquoring_time", 0, kind="time", zero=True)
        available = cycle - downtime - drain_time
        if available <= 0:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' cycle_time must exceed downtime + deliquoring_time"
            )
        porosity = self._fraction("porosity", upper_open=True)
        if porosity == 0:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' porosity must be positive"
            )
        specified_dp = self.get_param("P_drop") is not None
        if not specified_dp and self.get_param("area") is None:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' requires P_drop for area sizing or area for pressure calculation"
            )
        dp_bar = self._number("P_drop", kind="pressure") if specified_dp else None
        exponent = self._fraction("compressibility", 0, upper_open=True)
        reference = self._number("reference_pressure", 1, kind="pressure")
        medium = self._number("medium_resistance", 0, kind="resistance", zero=True)
        batch_factor = cycle / 3600
        liquid_batch = {c: f * batch_factor for c, f in liquid.items()}
        solid_volume = (
            sum(
                f * self.thermo._solid_molar_volume(c, feed.T)
                for c, f in solids.items()
            )
            * batch_factor
        )
        dry_mass = (
            sum(f * self.thermo.props[c].MW for c, f in solids.items()) * batch_factor
        )
        if solid_volume <= 0 or dry_mass <= 0:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' capture curve retains no cake-forming solids"
            )
        if self.get_param("specific_cake_resistance") is not None:
            if self.get_param("kozeny_constant") is not None:
                raise UnitOperationError(
                    f"Filter '{self.unit_id}' choose specific_cake_resistance or kozeny_constant"
                )
            alpha_reference = self._number("specific_cake_resistance", kind="alpha")
            resistance_model = "specified_reference_resistance"
        else:
            surface_per_volume = surface_rate * batch_factor / solid_volume
            density = dry_mass / solid_volume
            alpha_reference = (
                self._number("kozeny_constant", 5)
                * surface_per_volume**2
                * (1 - porosity)
                / (density * porosity**3)
            )
            resistance_model = "kozeny_carman_retained_psd"
        pore_volume = solid_volume * porosity / (1 - porosity)
        composition = self._composition(liquid)
        volume = sum(liquid_batch.values()) * self.thermo.mixture_liquid_molar_volume(
            composition, feed.T
        )
        if not all(
            math.isfinite(v) and v > 0
            for v in (volume, pore_volume, dry_mass, alpha_reference)
        ):
            raise UnitOperationError(
                f"Filter '{self.unit_id}' invalid cake or liquid properties"
            )
        if volume <= pore_volume:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' insufficient liquid to form a saturated cake and positive filtrate"
            )
        retention = pore_volume / volume
        mu = (
            self._number("liquid_viscosity", kind="viscosity")
            if self.get_param("liquid_viscosity") is not None
            else self.thermo.mixture_viscosity(composition, feed.T, feed.P, 0.0)
        )
        wash = inlets.get("wash")
        wash_batch = {}
        wash_volume = 0.0
        wash_mu = mu
        cells_value = self._number("wash_cells", 10)
        cells = int(cells_value)
        if cells != cells_value or cells > 200:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' wash_cells must be an integer from 1 to 200"
            )
        if wash is not None:
            wash_flows = self._liquid_flows(wash, "wash")
            if sum(wash.solid_component_flows.values()) > 1e-10 * wash.F:
                raise UnitOperationError(
                    f"Filter '{self.unit_id}' wash must be solids-free"
                )
            if wash.P < feed.P - 1e-8:
                raise UnitOperationError(
                    f"Filter '{self.unit_id}' wash pressure must be at least feed pressure"
                )
            if abs(wash.T - feed.T) > 1e-6:
                raise UnitOperationError(
                    f"Filter '{self.unit_id}' isothermal washing requires wash and feed at the same temperature"
                )
            wash_batch = {c: f * batch_factor for c, f in wash_flows.items()}
            wash_composition = self._composition(wash_flows)
            wash_volume = sum(
                wash_batch.values()
            ) * self.thermo.mixture_liquid_molar_volume(wash_composition, feed.T)
            wash_mu = (
                self._number("wash_viscosity", kind="viscosity")
                if self.get_param("wash_viscosity") is not None
                else self.thermo.mixture_viscosity(
                    wash_composition, feed.T, feed.P, 0.0
                )
            )
        elif any(
            self.get_param(n) is not None for n in ("wash_cells", "wash_viscosity")
        ):
            raise UnitOperationError(
                f"Filter '{self.unit_id}' washing parameters require a wash inlet"
            )
        if not all(
            math.isfinite(v) and v > 0 for v in (mu, wash_mu)
        ) or not math.isfinite(wash_volume):
            raise UnitOperationError(
                f"Filter '{self.unit_id}' invalid liquid transport properties"
            )
        reference_coefficients = cycle_coefficients(
            dry_mass=dry_mass,
            filtrate_volume=volume - pore_volume,
            pore_volume=pore_volume,
            wash_volume=wash_volume,
            alpha=alpha_reference,
            medium_resistance=medium,
            pressure_drop=reference * 1e5,
            feed_viscosity=mu,
            wash_viscosity=wash_mu,
            wash_cells=cells,
        )

        def coefficients(drop):
            relative = drop / reference
            return tuple(
                (a * relative ** (exponent - 1), b / relative)
                for a, b in reference_coefficients
            )

        if not specified_dp:
            fixed_area = self._number("area", kind="area")

            def residual(drop):
                form, washing_coeff = coefficients(drop)
                return (
                    (form[0] + washing_coeff[0]) / fixed_area**2
                    + (form[1] + washing_coeff[1]) / fixed_area
                    - available
                )

            maximum = feed.P * (1 - 1e-10)
            if not math.isfinite(maximum) or maximum <= 0 or residual(maximum) > 0:
                raise UnitOperationError(
                    f"Filter '{self.unit_id}' required pressure drop exceeds available feed pressure; increase area or feed pressure"
                )
            dp_bar = brentq(residual, maximum * 1e-12, maximum, xtol=1e-14, rtol=1e-12)
        pressure = feed.P - dp_bar
        if not math.isfinite(pressure) or pressure <= 0:
            raise UnitOperationError(
                f"Filter '{self.unit_id}' P_drop must leave positive filtrate pressure"
            )
        alpha = alpha_reference * (dp_bar / reference) ** exponent
        formation, washing = coefficients(dp_bar)
        a, b = (formation[i] + washing[i] for i in range(2))
        sizing_area = required_area(a, b, available)
        area = (
            self._number("area", kind="area")
            if self.get_param("area") is not None
            else sizing_area
        )
        formation_time = formation[0] / area**2 + formation[1] / area
        washing_time = washing[0] / area**2 + washing[1] / area
        ratio = wash_volume / pore_volume
        remaining_profile = wash_profile(ratio, cells)
        remaining = float(remaining_profile.mean())
        components = set(liquid_batch) | set(wash_batch)
        retained = {
            c: retention * liquid_batch.get(c, 0) * remaining
            + (
                pore_volume * (1 - remaining) / wash_volume * wash_batch.get(c, 0)
                if wash_volume > 0
                else 0
            )
            for c in components
        }
        saturation = equilibrium = 1.0
        drain_parameters = {
            "deliquoring_pressure",
            "entry_pressure",
            "residual_saturation",
            "pore_index",
            "relative_permeability_exponent",
        }
        if drain_time > 0:
            drain_dp = self._number("deliquoring_pressure", dp_bar, kind="pressure")
            entry = self._number("entry_pressure", kind="pressure")
            residual = self._fraction("residual_saturation", upper_open=True)
            pore_index = self._number("pore_index")
            kr_exponent = self._number(
                "relative_permeability_exponent", 3 + 2 / pore_index
            )
            saturation, equilibrium = deliquor(
                duration=drain_time,
                pore_volume=pore_volume,
                area=area,
                cake_resistance=alpha * dry_mass / area,
                medium_resistance=medium,
                viscosity=wash_mu + (mu - wash_mu) * remaining,
                pressure_drop=drain_dp * 1e5,
                entry_pressure=entry * 1e5,
                residual_saturation=residual,
                pore_index=pore_index,
                relative_permeability_exponent=kr_exponent,
            )
        elif any(self.get_param(n) is not None for n in drain_parameters):
            raise UnitOperationError(
                f"Filter '{self.unit_id}' deliquoring parameters require positive deliquoring_time"
            )
        retained = {c: f * saturation / batch_factor for c, f in retained.items()}
        cake_flows = {
            c: solids.get(c, 0) + retained.get(c, 0) for c in set(solids) | components
        }
        filtrate_flows = {
            c: (liquid_batch.get(c, 0) + wash_batch.get(c, 0)) / batch_factor
            - retained.get(c, 0)
            + escaped.get(c, 0)
            for c in components | set(escaped)
        }
        # Differences are analytic component balances; tolerate only roundoff.
        if any(f < -1e-10 for f in filtrate_flows.values()):
            raise UnitOperationError(
                f"Filter '{self.unit_id}' washing violated component conservation"
            )
        filtrate_flows = {c: max(0.0, f) for c, f in filtrate_flows.items()}
        conventional = {
            c: f
            for c, f in solids.items()
            if c in self.thermo.conventional_solid_components
        }
        cake = self.thermo.calculate_state_with_solid_flows(
            feed.T,
            pressure,
            sum(cake_flows.values()),
            self._composition(cake_flows),
            conventional,
        )
        filtrate_conventional = {
            c: f
            for c, f in escaped.items()
            if c in self.thermo.conventional_solid_components and f > 0
        }
        filtrate = self.thermo.calculate_state_with_solid_flows(
            feed.T,
            pressure,
            sum(filtrate_flows.values()),
            self._composition(filtrate_flows),
            filtrate_conventional,
        )
        self._liquid_flows(cake, "cake outlet")
        self._liquid_flows(filtrate, "filtrate outlet")
        for stream, distributions in ((cake, cake_psd), (filtrate, filtrate_psd)):
            stream.solid_particle_properties = {
                c: dict(p)
                for c, p in feed.solid_particle_properties.items()
                if stream.solid_component_flows.get(c, 0) > 0
            }
            stream.solid_particle_size_distributions = distributions
            stream.validate_particle_size_distributions()
        if any(
            s.H is None or not math.isfinite(s.H)
            for s in (*inlets.values(), cake, filtrate)
        ):
            raise UnitOperationError(
                f"Filter '{self.unit_id}' requires finite inlet and outlet enthalpies"
            )
        duty = (
            cake.F * cake.H
            + filtrate.F * filtrate.H
            - sum(s.F * s.H for s in inlets.values())
        )
        wet_liquid_mass = sum(f * self.thermo.props[c].MW for c, f in retained.items())
        dry_mass_rate = dry_mass / batch_factor
        performance = {
            "model": "cycle_pressure_cake_filter",
            "mode": "pressure"
            if not specified_dp
            else ("rating" if self.get_param("area") is not None else "sizing"),
            "resistance_model": resistance_model,
            "pressure_drop_bar": dp_bar,
            "captured_solid_component_flows_kmol_h": solids,
            "escaped_solid_component_flows_kmol_h": escaped,
            "solid_capture_mass_fraction": dry_mass_rate
            / sum(
                f * self.thermo.props[c].MW
                for c, f in feed.solid_component_flows.items()
            ),
            "area_m2": area,
            "required_area_m2": sizing_area,
            "capacity_ratio": area / sizing_area,
            "cycle_feasible": formation_time + washing_time <= available * (1 + 1e-9),
            "cycle_time_s": cycle,
            "formation_time_s": formation_time,
            "washing_time_s": washing_time,
            "deliquoring_time_s": drain_time,
            "downtime_s": downtime,
            "required_cycle_time_s": formation_time
            + washing_time
            + drain_time
            + downtime,
            "cake_thickness_m": solid_volume / ((1 - porosity) * area),
            "porosity": porosity,
            "specific_cake_resistance_m_per_kg": alpha,
            "cake_resistance_per_m": alpha * dry_mass / area,
            "pore_volume_m3_per_cycle": pore_volume,
            "primary_filtrate_m3_per_cycle": volume - pore_volume,
            "wash_ratio": ratio,
            "wash_cells": cells,
            "mother_liquor_tracer_remaining_fraction": remaining,
            "wash_tracer_profile": remaining_profile.tolist(),
            "cake_saturation": saturation,
            "equilibrium_saturation": equilibrium,
            "dry_solids_kg_per_h": dry_mass_rate,
            "retained_liquid_kg_per_h": wet_liquid_mass,
            "cake_moisture_mass_fraction": wet_liquid_mass
            / (wet_liquid_mass + dry_mass_rate),
            "liquid_viscosity_Pa_s": mu,
            "wash_viscosity_Pa_s": wash_mu,
            "P_out_bar": pressure,
            "duty_kW": duty / 3600,
        }
        warnings = []
        if not performance["cycle_feasible"]:
            warnings.append(
                "Specified area cannot process the feed and wash within cycle_time; outlet balances describe the requested load, not achievable operation."
            )
        return UnitResult(
            outlet_streams={"cake": cake, "filtrate": filtrate},
            heat_duty=duty,
            performance=performance,
            warnings=warnings,
        )
