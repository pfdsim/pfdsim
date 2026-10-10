"""Geometry-based wall and film transport, shared by exchangers and utilities.

Overall coefficients and areas refer to the outer tube surface. Thermodynamic
curves and stream energy balances remain owned by HeatExchanger.
"""
from dataclasses import dataclass, field, replace
import math

from ht.boiling_nucleic import Rohsenow, Zuber
from ht.condensation import Shah
from ht.conv_tube_bank import (
    Nu_Zukauskas_Bejan, baffle_correction_Bell, baffle_leakage_Bell,
    bundle_bypassing_Bell, laminar_correction_Bell, unequal_baffle_spacing_Bell,
)
from ht.hx import F_LMTD_Fakheri, Ntubes_Phadkeb
from ht.insulation import building_materials
from scipy.optimize import brentq

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .transport_correlations import circular_area, flow_velocity, reynolds_number, internal_flow_transfer, shah_boiling_coefficient
    from .unit_operations_base import UnitOperationError
    from .thermodynamics import ThermodynamicsError
else:
    from transport_correlations import circular_area, flow_velocity, reynolds_number, internal_flow_transfer, shah_boiling_coefficient
    from unit_operations_base import UnitOperationError
    from thermodynamics import ThermodynamicsError


# Constant reference values: Perry 9th ed., Tables 2-149 (300 K, pure metals)
# and 2-150 (373.15 K, stainless; Btu/(h ft F) converted by 1.730735).
WALL_MATERIALS = {
    'carbon_steel': (building_materials['Metals, steel'][1], None,
                     'VDI Heat Atlas building-material table via ht; nominal generic steel'),
    'brass': (building_materials['Metals, brass'][1], None,
              'VDI Heat Atlas building-material table via ht; nominal brass'),
    'copper': (398., 300., 'Perry Table 2-149, high-purity copper'),
    'aluminum': (273., 300., 'Perry Table 2-149, high-purity aluminum'),
    'iron': (80., 300., 'Perry Table 2-149, high-purity iron'),
    'nickel': (91., 300., 'Perry Table 2-149, high-purity nickel'),
    'titanium': (21., 300., 'Perry Table 2-149, high-purity titanium'),
    'stainless_steel_304': (9.4*1.730735, 373.15, 'Perry Table 2-150, AISI 304'),
    'stainless_steel_316': (9.4*1.730735, 373.15, 'Perry Table 2-150, AISI 316'),
    'stainless_steel_430': (15.1*1.730735, 373.15, 'Perry Table 2-150, AISI 430'),
}


def wall_material_record(material):
    name = str(material).strip().lower().replace('-', '_').replace(' ', '_')
    name = {'aluminium': 'aluminum', 'stainless_steel': 'stainless_steel_304',
            'steel': 'carbon_steel', 'mild_steel': 'carbon_steel',
            '304': 'stainless_steel_304', '316': 'stainless_steel_316',
            'ss304': 'stainless_steel_304', 'ss316': 'stainless_steel_316'}.get(name, name)
    if name not in WALL_MATERIALS:
        raise UnitOperationError(f'Unknown wall material {material!r}; choose {", ".join(WALL_MATERIALS)} or supply wall_conductivity')
    k, temperature, source = WALL_MATERIALS[name]
    return {'material': name, 'conductivity_W_m_K': k, 'reference_T_K': temperature,
            'source': source, 'model': 'constant_reference_value'}


def transport_parameter(value, name, unit, factors, *, nonnegative=False):
    normalized = str(unit or '').strip().lower().replace(' ', '')
    if normalized not in factors:
        raise UnitOperationError(f'{name} has unsupported unit {unit!r}')
    try:
        number = float(value)*factors[normalized]
    except (TypeError, ValueError) as exc:
        raise UnitOperationError(f'{name} must be numeric') from exc
    if not math.isfinite(number) or number < 0 or (number == 0 and not nonnegative):
        raise UnitOperationError(f'{name} must be finite and {"nonnegative" if nonnegative else "positive"}')
    return number


def heat_transfer_area(value, unit, *, name='area'):
    return transport_parameter(value, name, unit,
        {'': 1, 'm2': 1, 'm^2': 1, 'ft2': .09290304, 'ft^2': .09290304})


def solve_thermal_domain_root(objective, upper_guess, *, label, maximum=None,
                              xtol=1e-6, rtol=1e-9):
    """Bracket a positive root using only finite, valid residual endpoints.

    Invalid property/phase trials bound the valid domain; they are never fed
    to Brent as artificial infinite residuals. Shared by wall and duty solves.
    """
    cache = {}
    def evaluate(value):
        if value not in cache:
            try:
                result = float(objective(value))
                if not math.isfinite(result):
                    raise UnitOperationError('Trial residual is not finite')
                cache[value] = result
            except (ValueError, UnitOperationError, ThermodynamicsError) as exc:
                cache[value] = exc
        result = cache[value]
        if isinstance(result, Exception):
            raise result
        return result
    initial = evaluate(0.0)
    if initial == 0:
        return 0.0
    lower, invalid_upper = 0.0, None
    candidate = float(upper_guess)
    last_error = None
    for _ in range(100):
        if candidate <= lower or not math.isfinite(candidate):
            break
        try:
            value = evaluate(candidate)
        except (ValueError, UnitOperationError, ThermodynamicsError) as exc:
            invalid_upper, last_error = candidate, exc
        else:
            if value*initial <= 0:
                return brentq(evaluate, lower, candidate, xtol=xtol, rtol=rtol, maxiter=100)
            lower = candidate
        if invalid_upper is not None:
            candidate = .5*(lower+invalid_upper)
            if invalid_upper-lower <= max(xtol*.1, abs(candidate)*1e-12):
                break
        else:
            candidate *= 2
            if maximum is not None:
                candidate = min(candidate, maximum)
    raise UnitOperationError(f'{label}: cannot bracket a finite valid root; {last_error or "specified capacity exceeds the supported domain"}')


def horizontal_film_condensation(liquid, vapor_density, latent_heat, diameter, delta_temperature, *, rows=1):
    """Perry Eqs.5-80,5-87,5-88: horizontal film and condensate inundation."""
    corrected_latent = latent_heat+.68*liquid['Cp_J_kg_K']*delta_temperature
    return .729*(liquid['density_kg_m3']*9.80665*(liquid['density_kg_m3']-vapor_density)
        *corrected_latent*liquid['conductivity_W_m_K']**3
        /(liquid['viscosity_Pa_s']*diameter*delta_temperature))**.25/rows**.25


@dataclass(frozen=True)
class ShellTubeGeometry:
    shell_diameter: float
    bundle_diameter: float
    tube_outer_diameter: float
    tube_pitch: float
    tube_count: int
    tube_passes: int
    baffle_spacing: float
    baffle_cut: float
    baffle_count: int
    tube_baffle_clearance: float
    shell_baffle_clearance: float
    sealing_strip_pairs: int = 0
    layout_angle: int = 30
    pool_boiling_Csf: float | None = None
    pool_boiling_n: float = 1.0
    pool_boiling_wetted: bool = False
    tube_length: float | None = None

    def __post_init__(self):
        if not self.tube_outer_diameter < self.bundle_diameter < self.shell_diameter:
            raise UnitOperationError('Shell geometry requires tube_outer_diameter < bundle_diameter < shell_inner_diameter')
        if self.tube_pitch < 1.25*self.tube_outer_diameter:
            raise UnitOperationError('tube_pitch must be at least 1.25 times tube_outer_diameter')
        if self.tube_passes not in (1, 2, 4, 6, 8) or self.tube_count % self.tube_passes:
            raise UnitOperationError('Tube passes must be 1, 2, 4, 6 or 8 with equal integer tube counts per pass')
        if self.layout_angle not in (30, 45, 90) or not .15 <= self.baffle_cut <= .45:
            raise UnitOperationError('Tube layout must be 30/45/90 degrees and baffle_cut between 0.15 and 0.45')
        if self.baffle_spacing <= 0 or self.baffle_count < 1 or self.sealing_strip_pairs < 0:
            raise UnitOperationError('Baffle spacing/count must be positive and sealing_strip_pairs nonnegative')
        capacity = Ntubes_Phadkeb(self.bundle_diameter, self.tube_outer_diameter,
                                  self.tube_pitch, self.tube_passes, self.layout_angle)
        if self.tube_count > capacity:
            raise UnitOperationError(f'Tube count {self.tube_count} exceeds bundle packing capacity {capacity}')
        if not 0 <= self.tube_baffle_clearance <= .1*self.tube_outer_diameter or not 0 <= self.shell_baffle_clearance <= .05*self.shell_diameter:
            raise UnitOperationError('Baffle diametral clearances exceed supported small-clearance geometry')
        if self.tube_length is not None and self.tube_length <= (self.baffle_count-1)*self.baffle_spacing:
            raise UnitOperationError('Tube length is too short for the specified baffle_count and baffle_spacing')

    @property
    def transverse_pitch(self):
        return self.tube_pitch/math.sqrt(2) if self.layout_angle == 45 else self.tube_pitch

    @property
    def longitudinal_pitch(self):
        return self.tube_pitch*math.sqrt(3)/2 if self.layout_angle == 30 else self.transverse_pitch

    @property
    def flow_area(self):
        return self.baffle_spacing*((self.shell_diameter-self.bundle_diameter)
            + (self.bundle_diameter-self.tube_outer_diameter)
            * (self.tube_pitch-self.tube_outer_diameter)/self.transverse_pitch)

    def shell_coefficient(self, props, wall_prandtl):
        ds, db, do = self.shell_diameter, self.bundle_diameter, self.tube_outer_diameter
        theta = math.acos(min(1., ds*(.5-self.baffle_cut)/((db-do)/2)))
        window_fraction = (theta-math.sin(theta)*math.cos(theta))/math.pi
        cross_fraction = 1-2*window_fraction
        rows = max(1, round(ds*(1-2*self.baffle_cut)/self.longitudinal_pitch))
        window_rows = max(0, round(.8*(ds*self.baffle_cut-(ds-db)/2)/self.longitudinal_pitch))
        re, pr = props['reynolds'], props['prandtl']
        if not 10 <= re <= 2e6 or not .7 <= pr <= 500:
            raise UnitOperationError(f'Shell crossflow requires 10<=Re<=2000000 and 0.7<=Pr<=500; got Re={re:g}, Pr={pr:g}')
        nu = Nu_Zukauskas_Bejan(re, pr, rows, self.longitudinal_pitch, self.transverse_pitch, Pr_wall=wall_prandtl)
        shell_arc = 2*math.pi-2*math.acos(1-2*self.baffle_cut)
        ssb = .5*ds*self.shell_baffle_clearance*shell_arc
        stb = self.tube_count*(1-window_fraction)*math.pi/4*((do+self.tube_baffle_clearance)**2-do**2)
        bypass = self.baffle_spacing*(ds-db)/self.flow_area
        if (ssb+stb)/self.flow_area > .8 or bypass > .8:
            raise UnitOperationError('Shell leakage/bypass area exceeds supported Bell correction range')
        end_spacing = ((self.tube_length-(self.baffle_count-1)*self.baffle_spacing)/2
                       if self.tube_length is not None else self.baffle_spacing)
        factors = {
            'Jc': baffle_correction_Bell(cross_fraction, method='HEDH'),
            'Jl': baffle_leakage_Bell(ssb, stb, self.flow_area, method='HEDH') if ssb+stb else 1.,
            'Jb': bundle_bypassing_Bell(bypass, self.sealing_strip_pairs, rows, laminar=re < 100,
                                      method='HEDH') if self.sealing_strip_pairs/rows < .5 else 1.,
            'Jr': laminar_correction_Bell(re, (rows+window_rows)*(self.baffle_count+1)),
            'Js': unequal_baffle_spacing_Bell(self.baffle_count, self.baffle_spacing,
                                             end_spacing, end_spacing, laminar=re < 100),
        }
        factor = math.prod(factors.values())
        if not math.isfinite(factor) or factor <= 0:
            raise UnitOperationError('Invalid shell correction factors')
        return {'h_W_m2_K': nu*props['conductivity_W_m_K']/do*factor, 'nusselt': nu*factor,
                'flow_regime': 'shell_crossflow', 'correlation': 'zukauskas_bell_corrections',
                'shell_corrections': factors, 'crossflow_rows': rows, 'crossflow_tube_fraction': cross_fraction}


@dataclass(frozen=True)
class TubularHeatTransfer:
    tube_inner_diameter_m: float
    tube_outer_diameter_m: float
    shell_inner_diameter_m: float
    wall_conductivity_W_m_K: float
    fouling_tube_m2_K_W: float
    fouling_shell_m2_K_W: float
    tube_is_hot: bool
    wall_material: dict | None = None
    shell_geometry: ShellTubeGeometry | None = None
    orientation: str = 'horizontal'
    saturation_cache: dict = field(default_factory=dict, compare=False, repr=False)
    property_cache: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        di, do, ds = self.tube_inner_diameter_m, self.tube_outer_diameter_m, self.shell_inner_diameter_m
        if not all(math.isfinite(v) and v > 0 for v in (di, do, ds, self.wall_conductivity_W_m_K)):
            raise UnitOperationError('Tube diameters and wall conductivity must be finite and positive')
        if not di < do < ds:
            raise UnitOperationError('Tubular geometry requires tube_inner_diameter < tube_outer_diameter < shell_inner_diameter')
        if self.shell_geometry is None and not .1 <= do/ds <= .9:
            raise UnitOperationError('Double-pipe annulus diameter ratio must be between 0.1 and 0.9')
        if self.orientation not in ('horizontal', 'vertical'):
            raise UnitOperationError('Exchanger orientation must be horizontal or vertical')
        if not all(math.isfinite(v) and v >= 0 for v in (self.fouling_tube_m2_K_W, self.fouling_shell_m2_K_W)):
            raise UnitOperationError('Fouling resistances must be finite and nonnegative')

    @property
    def tube_count(self):
        return self.shell_geometry.tube_count if self.shell_geometry else 1

    @property
    def perimeter_m(self):
        return math.pi*self.tube_outer_diameter_m*self.tube_count

    def with_available_area(self, area):
        if self.shell_geometry is None:
            return self
        return replace(self, shell_geometry=replace(self.shell_geometry, tube_length=area/self.perimeter_m))

    @staticmethod
    def _require_fluid(state):
        if state.solid_component_flows or state.solid_fraction > 1e-10 or state.liquid2_fraction > 1e-10:
            raise UnitOperationError('Calculated films require fluid-only streams without solids or liquid-liquid splitting')
        if state.fluid_vapor_fraction is None or not math.isfinite(state.F) or state.F <= 0:
            raise UnitOperationError('Calculated films require positive finite fluid flow')

    def _saturation(self, thermo, state):
        active = [c for c, z in state.composition.items() if z > 1e-12]
        if len(active) != 1 or not hasattr(thermo, 'bubble_point_T'):
            return None
        comp = active[0]
        pc = getattr(thermo.props[comp], 'Pc', None)
        if pc is None or state.P >= pc:
            return None
        key = id(thermo), comp, state.P
        if key not in self.saturation_cache:
            direct = getattr(thermo, 'calculate_state_PQ', None)
            if direct:
                liquid = direct(state.P, 0., state.F, state.composition, include=('H', 'rho', 'Cp'))
                vapor = direct(state.P, 1., state.F, state.composition, include=('H', 'rho', 'Cp'))
            else:
                temperature = thermo.bubble_point_T(state.composition, state.P, state.T)
                liquid = thermo.calculate_state(temperature, state.P, state.F, state.composition,
                                                phase='liquid', flash=False, include=('H', 'rho', 'Cp'))
                vapor = thermo.calculate_state(temperature, state.P, state.F, state.composition,
                                               phase='vapor', flash=False, include=('H', 'rho', 'Cp'))
            self.saturation_cache[key] = {'liquid_state': liquid, 'vapor_state': vapor, 'Pc_bar': pc}
        return self.saturation_cache[key]

    def _phase_properties(self, thermo, state, phase):
        cache_key = id(thermo), tuple(sorted(state.composition.items())), state.T, state.P, phase
        if cache_key in self.property_cache:
            return dict(self.property_cache[cache_key])
        calculated = thermo.calculate_state(state.T, state.P, state.F, state.composition,
                                             phase=phase, flash=False, include=('rho', 'Cp'))
        mw = thermo.mixture_MW(state.composition)
        vf = 0. if phase == 'liquid' else 1.
        mu = thermo.mixture_viscosity(state.composition, state.T, state.P, vf)
        conductivities = thermo.transport_mixture_thermal_conductivity(state.composition, state.T, state.P, vf)
        k = conductivities.liquid if phase == 'liquid' else conductivities.vapor
        if not all(v is not None and math.isfinite(v) and v > 0 for v in (mw, mu, k, calculated.rho, calculated.Cp)):
            raise UnitOperationError(f'{phase} transport properties must be finite and positive')
        cp = calculated.Cp*1000/mw
        result = {'density_kg_m3': calculated.rho*mw, 'viscosity_Pa_s': mu,
                  'conductivity_W_m_K': k, 'Cp_J_kg_K': cp, 'prandtl': cp*mu/k}
        if len(self.property_cache) >= 4096:
            self.property_cache.clear()
        self.property_cache[cache_key] = result
        return dict(result)

    def _saturation_properties(self, prepared, *, surface_tension=False):
        sat = prepared['sat']
        if 'liquid' not in sat:
            thermo, state = prepared['thermo'], prepared['state']
            lp = self._phase_properties(thermo, sat['liquid_state'], 'liquid')
            vp = self._phase_properties(thermo, sat['vapor_state'], 'vapor')
            hfg = (sat['vapor_state'].H-sat['liquid_state'].H)*1000/thermo.mixture_MW(state.composition)
            if not math.isfinite(hfg) or hfg <= 0 or lp['density_kg_m3'] <= vp['density_kg_m3']:
                raise UnitOperationError('Invalid pure-fluid saturation transport properties')
            sat.update(liquid=lp, vapor=vp, latent_heat_J_kg=hfg)
        if surface_tension and 'surface_tension_N_m' not in sat:
            state = prepared['state']
            sigma = prepared['thermo'].transport_mixture_surface_tension(
                state.composition, sat['liquid_state'].T, state.P, .5,
                sat['liquid_state'].composition, sat['vapor_state'].composition)
            if not math.isfinite(sigma) or sigma <= 0:
                raise UnitOperationError('Invalid pure-fluid saturation surface tension')
            sat['surface_tension_N_m'] = sigma
        return sat

    def _prepare(self, thermo, state, side):
        self._require_fluid(state)
        sat = self._saturation(thermo, state)
        vf = state.fluid_vapor_fraction
        if 1e-10 < vf < 1-1e-10 and len([z for z in state.composition.values() if z > 1e-12]) != 1:
            raise UnitOperationError('Calculated boiling/condensation films require pure fluids; two-phase mixtures and noncondensables are unsupported')
        if not (sat and abs(state.T-sat['liquid_state'].T) < 1e-5):
            equilibrium = thermo.calculate_state(state.T, state.P, state.F, state.composition, include=())
            self._require_fluid(equilibrium)
            if abs(equilibrium.fluid_vapor_fraction-vf) > 1e-7:
                raise UnitOperationError('Forced inlet phase conflicts with equilibrium')
        di, do, ds = self.tube_inner_diameter_m, self.tube_outer_diameter_m, self.shell_inner_diameter_m
        if side == 'tube':
            passes = self.shell_geometry.tube_passes if self.shell_geometry else 1
            area, diameter, annulus_ratio = circular_area(di)*self.tube_count/passes, di, None
        elif self.shell_geometry:
            area, diameter, annulus_ratio = self.shell_geometry.flow_area, do, None
        else:
            area, diameter, annulus_ratio = circular_area(ds)-circular_area(do), ds-do, do/ds
        mass_flow = state.F*thermo.mixture_MW(state.composition)/3600
        phase = 'vapor' if vf >= 1-1e-10 else 'liquid'
        reference = sat['liquid_state'] if sat and 0 < vf < 1 else state
        props = self._phase_properties(thermo, reference, phase)
        velocity = flow_velocity(mass_flow, props['density_kg_m3'], area)
        props.update(velocity_m_s=velocity, mass_flux_kg_m2_s=mass_flow/area,
            reynolds=reynolds_number(props['density_kg_m3'], velocity, diameter, props['viscosity_Pa_s']),
            phase=phase if vf in (0, 1) else 'two_phase', mass_quality=vf, hydraulic_diameter_m=diameter)
        prepared = {'thermo': thermo, 'state': state, 'side': side, 'sat': sat, 'properties': props,
                    'diameter': diameter, 'area': area, 'annulus_ratio': annulus_ratio, 'quality': vf}
        if sat and 0 < vf < 1:
            saturation = self._saturation_properties(prepared)
            rl, rv = saturation['liquid']['density_kg_m3'], saturation['vapor']['density_kg_m3']
            g = mass_flow/area
            props.update(liquid_only_reynolds=props['reynolds'], reynolds_basis='liquid_only',
                transport_property_basis='saturated_liquid_reference', density_basis='homogeneous_no_slip',
                density_kg_m3=1/((1-vf)/rl+vf/rv), velocity_m_s=g*((1-vf)/rl+vf/rv),
                superficial_liquid_velocity_m_s=g*(1-vf)/rl, superficial_vapor_velocity_m_s=g*vf/rv,
                liquid_density_kg_m3=rl, vapor_density_kg_m3=rv)
        return prepared

    def inlet_capacity_W_K(self, hot_thermo, cold_thermo, hot, cold):
        self._require_fluid(hot)
        self._require_fluid(cold)
        cold_limit = cold_thermo.calculate_state(hot.T, cold.P, cold.F, cold.composition, include=('H',))
        hot_limit = hot_thermo.calculate_state(cold.T, hot.P, hot.F, hot.composition, include=('H',))
        available = min(hot.F*(hot.H-hot_limit.H), cold.F*(cold_limit.H-cold.H))
        if not math.isfinite(available) or available <= 0:
            raise UnitOperationError('No positive duty bracket from inlet enthalpy limits')
        return available/3.6/(hot.T-cold.T)

    def validate_inlets(self, hot_thermo, cold_thermo, hot, cold):
        """Validate supplied phase inventories before reconstructing any curve."""
        tube, shell = (hot, cold) if self.tube_is_hot else (cold, hot)
        tt, st = (hot_thermo, cold_thermo) if self.tube_is_hot else (cold_thermo, hot_thermo)
        self._prepare(tt, tube, 'tube')
        self._prepare(st, shell, 'shell')

    def _single_phase(self, prepared, temperature):
        state, props = prepared['state'], prepared['properties']
        phase = 'vapor' if prepared['quality'] >= 1-1e-10 else 'liquid'
        wall = state.copy()
        wall.T = temperature
        wall_props = self._phase_properties(prepared['thermo'], wall, phase)
        if prepared['side'] == 'shell' and self.shell_geometry:
            return self.shell_geometry.shell_coefficient(props, wall_props['prandtl'])
        transfer = internal_flow_transfer(props['reynolds'], props['prandtl'], wall_props['prandtl'],
            annulus_diameter_ratio=prepared['annulus_ratio'],
            gas_temperature_ratio=state.T/temperature if phase == 'vapor' else None)
        return {**transfer, 'h_W_m2_K': transfer['nusselt']*props['conductivity_W_m_K']/prepared['diameter']}

    def _boiling(self, prepared, flux, check_limits):
        sat = self._saturation_properties(prepared, surface_tension=True)
        lp, vp = sat['liquid'], sat['vapor']
        g, d = prepared['properties']['mass_flux_kg_m2_s'], prepared['diameter']
        if prepared['annulus_ratio'] is not None and (self.shell_inner_diameter_m-self.tube_outer_diameter_m)/2 <= .004:
            d = 4*prepared['area']/(math.pi*self.tube_outer_diameter_m)
        if prepared['side'] == 'shell' and self.shell_geometry:
            if self.shell_geometry.pool_boiling_Csf is None or not self.shell_geometry.pool_boiling_wetted:
                raise UnitOperationError('Shell pool boiling requires shell_pool_boiling=true (fully submerged bundle) and boiling_Csf for the actual fluid/surface pair')
            h = Rohsenow(lp['density_kg_m3'], vp['density_kg_m3'], lp['viscosity_Pa_s'],
                lp['conductivity_W_m_K'], lp['Cp_J_kg_K'], sat['latent_heat_J_kg'],
                sat['surface_tension_N_m'], q=flux, Csf=self.shell_geometry.pool_boiling_Csf,
                n=self.shell_geometry.pool_boiling_n)
            result = {'h_W_m2_K': h, 'flow_regime': 'pool_boiling', 'correlation': 'rohsenow'}
        else:
            result = shah_boiling_coefficient(g, prepared['quality'], d,
                lp['density_kg_m3'], vp['density_kg_m3'], lp['viscosity_Pa_s'],
                lp['conductivity_W_m_K'], lp['Cp_J_kg_K'], sat['latent_heat_J_kg'], flux,
                horizontal=self.orientation == 'horizontal', check_limits=check_limits)
        if check_limits:
            if not (prepared['side'] == 'shell' and self.shell_geometry) and (
                not .0053 <= prepared['state'].P/sat['Pc_bar'] <= .78 or prepared['quality'] > .7
            ):
                raise UnitOperationError('Boiling supports reduced pressure 0.0053..0.78 and wetted-wall quality <=0.7; dryout/film boiling is unsupported')
            result['critical_flux_screen_W_m2'] = self._boiling_flux_screen(prepared, flux)
        return result

    def _boiling_flux_screen(self, prepared, flux):
        sat = self._saturation_properties(prepared, surface_tension=True)
        active = next(c for c, z in prepared['state'].composition.items() if z > 1e-12)
        component = prepared['thermo'].props[active]
        if not (prepared['side'] == 'shell' and self.shell_geometry) and (
            str(getattr(component, 'formula', '')).upper() == 'CO2' or getattr(component, 'CAS', '') == '124-38-9'
        ):
            raise UnitOperationError('CO2 boiling requires its dedicated correlation; this Shah model is unsupported')
        screen = .5*Zuber(sat['surface_tension_N_m'], sat['latent_heat_J_kg'],
                         sat['liquid']['density_kg_m3'], sat['vapor']['density_kg_m3'], K=.131)
        if flux > screen:
            raise UnitOperationError('Heat flux exceeds the conservative boiling critical-flux screen')
        return screen

    def _condensation(self, prepared, temperature):
        sat = self._saturation_properties(prepared)
        lp, vp = sat['liquid'], sat['vapor']
        dt = sat['liquid_state'].T-temperature
        if dt <= 0:
            raise UnitOperationError('Condensing film requires a wall below saturation')
        state = prepared['state']
        film = state.copy()
        film.T = .5*(sat['liquid_state'].T+temperature)
        fp = self._phase_properties(prepared['thermo'], film, 'liquid')
        d, g, x = prepared['diameter'], prepared['properties']['mass_flux_kg_m2_s'], prepared['quality']
        if prepared['side'] == 'shell' and self.shell_geometry:
            if self.orientation != 'horizontal':
                raise UnitOperationError('Shell gravity condensation requires horizontal tubes')
            rows = max(1, round(self.shell_geometry.bundle_diameter/self.shell_geometry.longitudinal_pitch))
            d = self.tube_outer_diameter_m
            h = horizontal_film_condensation(fp, vp['density_kg_m3'], sat['latent_heat_J_kg'], d, dt, rows=rows)
            if g*x/vp['density_kg_m3'] > 3:
                raise UnitOperationError('Shell gravity condensation requires vapor velocity<=3 m/s')
            return {'h_W_m2_K': h, 'flow_regime': 'film_condensation',
                    'correlation': 'nusselt_horizontal_bundle', 'vertical_tube_rows': rows}
        high_shear = (350 < g*d/lp['viscosity_Pa_s'] <= 100000 and g*d/vp['viscosity_Pa_s'] > 35000
                      and 3 < g/vp['density_kg_m3'] <= 300 and .0019 <= state.P/sat['Pc_bar'] <= .82
                      and .0028 <= d <= .04 and 11 <= g <= 4000)
        if prepared['annulus_ratio'] is not None and not high_shear:
            raise UnitOperationError('Annulus condensation requires high-shear Shah conditions: Re_LT>350, Re_GT>35000, vapor-only velocity>3 m/s')
        h = 0.
        if high_shear and x < 1:
            h = Shah(g*circular_area(d), x, d, lp['density_kg_m3'], lp['viscosity_Pa_s'],
                     lp['conductivity_W_m_K'], lp['Cp_J_kg_K'], state.P*1e5, sat['Pc_bar']*1e5)
        if prepared['annulus_ratio'] is not None:
            if h <= 0:
                raise UnitOperationError('Condensation at quality=1 requires interior phase-curve quadrature')
        elif self.orientation == 'horizontal':
            # Perry Sec.11: larger of gravity-film and shear correlations.
            gravity = horizontal_film_condensation(fp, vp['density_kg_m3'], sat['latent_heat_J_kg'], d, dt)
            h = max(h, gravity)
        elif not high_shear or x == 1:
            raise UnitOperationError('Vertical in-tube condensation requires high-shear flow and interior quality')
        return {'h_W_m2_K': h, 'flow_regime': 'film_condensation',
                'correlation': 'shah_1979' if prepared['annulus_ratio'] else 'perry_gravity_shear_condensation'}

    def _film_for_flux(self, prepared, signed_flux, check_limits=False):
        state, sat, x = prepared['state'], prepared['sat'], prepared['quality']
        flux = abs(signed_flux)
        if flux == 0:
            return state.T, {'h_W_m2_K': float('inf')}
        heating = signed_flux > 0
        at_saturation = sat and abs(state.T-sat['liquid_state'].T) < 1e-5
        phase_change = sat and (0 < x < 1 or (at_saturation and ((heating and x == 0) or (not heating and x == 1))))
        if phase_change:
            if heating:
                result = self._boiling(prepared, flux, check_limits)
                return state.T+flux/result['h_W_m2_K'], result
            def residual(drop):
                if drop == 0:
                    return -flux
                return self._condensation(prepared, state.T-drop)['h_W_m2_K']*drop-flux
            upper = state.T-prepared['wall_lower_K']
            if upper <= 0:
                raise UnitOperationError('No physical wall temperature for condensation')
            drop = brentq(residual, 0., upper, xtol=1e-8)
            return state.T-drop, self._condensation(prepared, state.T-drop)
        wall = state.T
        previous_error = float('inf')
        for _ in range(60):
            if sat and ((heating and x == 0 and wall > sat['liquid_state'].T)
                        or (not heating and x == 1 and wall < sat['liquid_state'].T)):
                updated = wall
                break
            result = self._single_phase(prepared, wall)
            updated = state.T+signed_flux/result['h_W_m2_K']
            if sat and ((heating and x == 0 and updated > sat['liquid_state'].T)
                        or (not heating and x == 1 and updated < sat['liquid_state'].T)):
                break
            if updated <= 1:
                raise UnitOperationError('Trial fluid-facing wall temperature is nonphysical')
            error = abs(updated-wall)
            if error < 1e-7:
                break
            relaxation = .5 if error >= previous_error else 1.
            wall += relaxation*(updated-wall)
            previous_error = error
        else:
            raise UnitOperationError('Single-phase wall-temperature solve failed')
        if sat and heating and x == 0 and updated > sat['liquid_state'].T:
            saturated = self._saturation_properties(prepared)
            if prepared['side'] == 'shell' and self.shell_geometry:
                raise UnitOperationError('Subcooled boiling across shell bundles is unsupported; use a saturated wetted pool-boiling specification')
            props = prepared['properties']
            g, d = props['mass_flux_kg_m2_s'], prepared['diameter']
            if prepared['annulus_ratio'] is not None and (self.shell_inner_diameter_m-self.tube_outer_diameter_m)/2 <= .004:
                d = 4*prepared['area']/(math.pi*self.tube_outer_diameter_m)
            bo = flux/(g*saturated['latent_heat_J_kg'])
            psi = max(230*math.sqrt(bo), 1+46*math.sqrt(bo))
            hf = .023*(g*d/props['viscosity_Pa_s'])**.8*props['prandtl']**.4*props['conductivity_W_m_K']/d
            subcool = sat['liquid_state'].T-state.T
            low_drop = flux/(hf*psi)
            low = subcool/low_drop <= max(2., 63000*bo**1.25)
            excess = low_drop if low else (flux/hf-subcool)/psi
            if excess > 0:
                updated = sat['liquid_state'].T+excess
                result = {'h_W_m2_K': flux/(updated-state.T), 'flow_regime': 'subcooled_boiling',
                          'correlation': 'shah_1977', 'boiling_number': bo}
                if check_limits and (
                    not 0 <= subcool <= 153 or not 1e-5 <= bo <= .0054
                    or not 1400 <= g*d/props['viscosity_Pa_s'] <= 360000
                    or not .005 <= state.P/sat['Pc_bar'] <= .89
                    or not 200 <= g <= 87000
                    or (prepared['annulus_ratio'] is None and not .0024 <= d <= .0271)
                    or (prepared['annulus_ratio'] is not None and not .001 <= (self.shell_inner_diameter_m-self.tube_outer_diameter_m)/2 <= .0064)
                ):
                    raise UnitOperationError('Subcooled boiling exceeds the supported Shah parameter range')
                if check_limits:
                    result['critical_flux_screen_W_m2'] = self._boiling_flux_screen(prepared, flux)
        if sat and not heating and x == 1 and updated < sat['liquid_state'].T:
            if prepared['annulus_ratio'] is not None:
                raise UnitOperationError('Superheated annulus condensation is unsupported; start with a saturated two-phase inlet')
            saturated = dict(prepared, state=sat['vapor_state'])
            def residual(drop):
                if drop == 0:
                    return -flux
                return self._condensation(saturated, sat['liquid_state'].T-drop)['h_W_m2_K']*drop-flux
            upper = sat['liquid_state'].T-prepared['wall_lower_K']
            if upper <= 0:
                raise UnitOperationError('No physical wall temperature for condensation')
            drop = brentq(residual, 0., upper, xtol=1e-8)
            updated = sat['liquid_state'].T-drop
            result = self._condensation(saturated, updated)
            result['h_W_m2_K'] = flux/(state.T-updated)
            result['superheat_treatment'] = 'perry_conservative_saturation_driving_temperature'
        return updated, result

    def segment(self, hot_thermo, cold_thermo, hot, cold):
        tube, shell = (hot, cold) if self.tube_is_hot else (cold, hot)
        tt, st = (hot_thermo, cold_thermo) if self.tube_is_hot else (cold_thermo, hot_thermo)
        tp, sp = self._prepare(tt, tube, 'tube'), self._prepare(st, shell, 'shell')
        for prepared in (tp, sp):
            prepared['wall_lower_K'] = min(tube.T, shell.T)
        ratio = self.tube_outer_diameter_m/self.tube_inner_diameter_m
        rw = self.tube_outer_diameter_m*math.log(ratio)/(2*self.wall_conductivity_W_m_K)
        rf = ratio*self.fouling_tube_m2_K_W+self.fouling_shell_m2_K_W
        sign = 1 if tube.T > shell.T else -1
        difference = abs(tube.T-shell.T)
        if difference <= 0:
            raise UnitOperationError('Local exchanger driving temperature must be positive')

        def residual(magnitude):
            if magnitude == 0:
                return difference
            signed = sign*magnitude
            tw, _ = self._film_for_flux(tp, -signed*ratio)
            sw, _ = self._film_for_flux(sp, signed)
            return sign*(tw-sw)-magnitude*(rw+rf)

        bound = difference/(rw+rf)
        magnitude = solve_thermal_domain_root(residual, bound, maximum=bound, label='Coupled wall/film solve')
        flux = sign*magnitude
        tw, tf = self._film_for_flux(tp, -flux*ratio, check_limits=True)
        sw, sf = self._film_for_flux(sp, flux, check_limits=True)
        for prepared, wall_temperature, film, side in ((tp, tw, tf, 'tube'), (sp, sw, sf, 'shell' if self.shell_geometry else 'annulus')):
            if film.get('flow_regime') in ('laminar', 'transition', 'turbulent', 'shell_crossflow'):
                state = prepared['state']
                equilibrium = prepared['thermo'].calculate_state(wall_temperature, state.P, state.F, state.composition, include=())
                self._require_fluid(equilibrium)
                expected = 1. if prepared['quality'] >= 1-1e-10 else 0.
                if abs(equilibrium.fluid_vapor_fraction-expected) > 1e-8:
                    raise UnitOperationError(f'Calculated {side} wall changes phase outside the supported saturation-film model')
        error = abs(tw-sw-flux*(rw+rf))
        if error > max(1e-6, difference*1e-7):
            raise UnitOperationError(
                f'Coupled wall/film residual {error:g} K exceeds tolerance: '
                f'flux={flux:g} W/m2, tube wall={tw:g} K ({tf.get("flow_regime")}), '
                f'outer wall={sw:g} K ({sf.get("flow_regime")}), '
                f'bulk temperatures={tube.T:g}/{shell.T:g} K'
            )
        if not min(tube.T, shell.T)-1e-6 <= min(tw, sw) <= max(tw, sw) <= max(tube.T, shell.T)+1e-6:
            raise UnitOperationError('Fluid-facing wall temperatures exceed the physical bulk-temperature bounds')
        shell_label = 'shell' if self.shell_geometry else 'annulus'
        return {'U_W_m2_K': magnitude/difference,
                'tube': {**tp['properties'], **tf, 'wall_T_K': tw},
                shell_label: {**sp['properties'], **sf, 'wall_T_K': sw},
                'tube_film_resistance_m2_K_W': ratio/tf['h_W_m2_K'],
                f'{shell_label}_film_resistance_m2_K_W': 1/sf['h_W_m2_K'],
                'wall_resistance_m2_K_W': rw, 'fouling_resistance_m2_K_W': rf,
                'heat_flux_W_m2': magnitude, 'wall_temperature_residual_K': error,
                'tube_metal_wall_T_K': tw-flux*ratio*self.fouling_tube_m2_K_W,
                'outer_metal_wall_T_K': sw+flux*self.fouling_shell_m2_K_W}

    def arrangement_factor(self, hot_in, hot_out, cold_in, cold_out):
        if self.shell_geometry is None or self.shell_geometry.tube_passes == 1:
            return 1.
        if abs(hot_in.T-hot_out.T) < 1e-7 or abs(cold_out.T-cold_in.T) < 1e-7:
            return 1.
        try:
            factor = F_LMTD_Fakheri(hot_in.T, hot_out.T, cold_in.T, cold_out.T, shells=1)
        except (ValueError, ZeroDivisionError) as exc:
            raise UnitOperationError('Duty exceeds the shell-and-tube pass arrangement') from exc
        if not math.isfinite(factor) or not 0 < factor <= 1+1e-10:
            raise UnitOperationError('Invalid shell-and-tube LMTD correction')
        return min(1., factor)
