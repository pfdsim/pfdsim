import math
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Iterable, Optional, Union

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..chemical_properties import ChemicalDatabase, ChemicalProperties, get_database
else:
    from chemical_properties import ChemicalDatabase, ChemicalProperties, get_database
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from ..fluid_phase_models import normalize_fluid_phase_model
else:
    from fluid_phase_models import normalize_fluid_phase_model

from .common import (
    DEFAULT_STATE_INCLUDE,
    LIQUID_VOLUME_EXTRAPOLATION_LIMIT_K,
    P_REF,
    R,
    R_BAR,
    STEAM_WATER_MW,
    T_REF,
    ThermodynamicsError,
    _property_lookup_identifier,
)

from .henry import (
    AqueousEquilibriumContext,
    HENRY_DEFAULT_TMAX_K,
    HENRY_DEFAULT_TMIN_K,
    HenryComponentData,
    get_henry_constant_database,
)


@dataclass(frozen=True)
class TransportPhaseValues:
    """Liquid and vapor values for a phase-separated transport property."""

    liquid: Optional[float] = None
    vapor: Optional[float] = None


@dataclass(frozen=True)
class FluidPhaseEquilibrium:
    """One unconstrained fluid-equilibrium result on the total-fluid basis."""

    vapor_fraction: float
    liquid1_fraction: float
    liquid2_fraction: float
    y: dict[str, float]
    x1: dict[str, float]
    x2: dict[str, float]
    status: str
    stability: str
    extra: dict = field(default_factory=dict)


@dataclass
class StreamState:
    """Thermodynamic state of a process stream"""
    T: float  # Temperature [K]
    P: float  # Pressure [bar]
    F: float  # Total molar flow [kmol/h]
    composition: dict[str, float]  # Mole fractions (must sum to 1)
    vapor_fraction: float = 1.0  # Vapor fraction (0=all liquid, 1=all vapor)
    
    # Calculated properties (filled by calculate_properties)
    H: Optional[float] = None  # Molar enthalpy [kJ/kmol]
    S: Optional[float] = None  # Molar entropy [kJ/kmol-K]
    MW: Optional[float] = None  # Mixture molecular weight [kg/kmol]
    rho: Optional[float] = None  # Molar density [kmol/m3]
    Cp: Optional[float] = None  # Heat capacity [kJ/kmol-K]
    mu: Optional[float] = None  # Dynamic viscosity [Pa*s]
    
    # Phase compositions (for two-phase)
    x: Optional[dict[str, float]] = None  # Liquid mole fractions
    y: Optional[dict[str, float]] = None  # Vapor mole fractions
    # Universal fluid/solid phase inventory. ``x`` remains the pooled liquid
    # composition compatibility view; x1/x2 retain distinct liquid phases.
    liquid1_fraction: Optional[float] = None
    liquid2_fraction: float = 0.0
    x1: Optional[dict[str, float]] = None
    x2: Optional[dict[str, float]] = None
    solid_fraction: float = 0.0
    solid_composition: Optional[dict[str, float]] = None
    solid_component_flows: dict[str, float] = field(default_factory=dict)
    solid_particle_properties: dict[str, dict[str, float]] = field(default_factory=dict)
    fluid_phase_model: str = 'VLE'
    phase_status: str = 'unspecified'
    phase_stability: str = 'not_checked'
    phase_details: dict = field(default_factory=dict)
    thermo_scope: str = 'global'

    @property
    def effective_liquid1_fraction(self) -> float:
        """Liquid-1 fraction, inferred for legacy VLE states when omitted."""
        if self.liquid1_fraction is not None:
            return max(0.0, float(self.liquid1_fraction))
        return max(
            0.0,
            1.0
            - float(self.vapor_fraction)
            - float(self.liquid2_fraction)
            - float(self.solid_fraction),
        )

    def phase_fractions(self) -> dict[str, float]:
        """Return explicit total-stream phase fractions."""
        return {
            'vapor': max(0.0, float(self.vapor_fraction)),
            'liquid1': self.effective_liquid1_fraction,
            'liquid2': max(0.0, float(self.liquid2_fraction)),
            'solid': max(0.0, float(self.solid_fraction)),
        }

    @property
    def fluid_vapor_fraction(self) -> Optional[float]:
        """Vapor fraction on the active-fluid subtotal basis, if fluid exists."""
        fractions = self.phase_fractions()
        fluid = fractions['vapor'] + fractions['liquid1'] + fractions['liquid2']
        if fluid <= 1.0e-15:
            return None
        return fractions['vapor'] / fluid

    def phase_component_flows(self) -> dict[str, dict[str, float]]:
        """Return phase component flows [kmol/h] from the retained inventory."""
        fractions = self.phase_fractions()

        def flows(fraction: float, phase_composition: Optional[dict[str, float]]):
            if fraction <= 0.0 or not phase_composition:
                return {}
            return {
                component: self.F * fraction * float(value)
                for component, value in phase_composition.items()
                if float(value) > 0.0
            }

        liquid1 = self.x1 or self.x
        return {
            'vapor': flows(fractions['vapor'], self.y),
            'liquid1': flows(fractions['liquid1'], liquid1),
            'liquid2': flows(fractions['liquid2'], self.x2),
            'solid': dict(self.solid_component_flows),
        }
    
    def copy(self) -> 'StreamState':
        """Create a copy of this state"""
        return StreamState(
            T=self.T,
            P=self.P,
            F=self.F,
            composition=dict(self.composition),
            vapor_fraction=self.vapor_fraction,
            H=self.H,
            S=self.S,
            MW=self.MW,
            rho=self.rho,
            Cp=self.Cp,
            mu=self.mu,
            x=dict(self.x) if self.x else None,
            y=dict(self.y) if self.y else None,
            liquid1_fraction=self.liquid1_fraction,
            liquid2_fraction=self.liquid2_fraction,
            x1=dict(self.x1) if self.x1 else None,
            x2=dict(self.x2) if self.x2 else None,
            solid_fraction=self.solid_fraction,
            solid_composition=(
                dict(self.solid_composition) if self.solid_composition else None
            ),
            solid_component_flows=dict(self.solid_component_flows),
            solid_particle_properties={
                component: dict(values)
                for component, values in self.solid_particle_properties.items()
            },
            fluid_phase_model=self.fluid_phase_model,
            phase_status=self.phase_status,
            phase_stability=self.phase_stability,
            phase_details=dict(self.phase_details),
            thermo_scope=self.thermo_scope,
        )
    
    def mass_flow(self) -> float:
        """Mass flow rate [kg/h]"""
        if self.MW is None:
            return 0.0
        return self.F * self.MW
    
    def component_flows(self) -> dict[str, float]:
        """Molar flow of each component [kmol/h]"""
        return {comp: self.F * z for comp, z in self.composition.items()}
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        phase_fractions = {
            name: value
            for name, value in self.phase_fractions().items()
            if value > 1.0e-15
        }
        phase_component_flows = {
            name: values
            for name, values in self.phase_component_flows().items()
            if values
        }
        payload = {
            'T': self.T,
            'T_C': self.T - 273.15,
            'P': self.P,
            'F': self.F,
            'composition': self.composition,
            'vapor_fraction': self.vapor_fraction,
            'H': self.H,
            'S': self.S,
            'MW': self.MW,
            'rho': self.rho,
            'Cp': self.Cp,
            'mu': self.mu,
            'x': self.x,
            'y': self.y,
            'liquid1_fraction': self.effective_liquid1_fraction,
            'fluid_phase_model': self.fluid_phase_model,
            'phase_status': self.phase_status,
            'phase_stability': self.phase_stability,
            'thermo_scope': self.thermo_scope,
            'phase_fractions': phase_fractions,
            'phase_component_flows': phase_component_flows,
            'mass_flow': self.mass_flow(),
        }
        if self.fluid_vapor_fraction is not None:
            payload['fluid_vapor_fraction'] = self.fluid_vapor_fraction
        if self.x1 is not None:
            payload['x1'] = self.x1
        if self.liquid2_fraction > 1.0e-15 or self.x2:
            payload['liquid2_fraction'] = self.liquid2_fraction
            payload['x2'] = self.x2
        if (
            self.solid_fraction > 1.0e-15
            or self.solid_composition
            or self.solid_component_flows
        ):
            payload['solid_fraction'] = self.solid_fraction
            payload['solid_composition'] = self.solid_composition
            payload['solid_component_flows'] = dict(self.solid_component_flows)
            if self.solid_particle_properties:
                payload['solid_particle_properties'] = {
                    component: dict(values)
                    for component, values in self.solid_particle_properties.items()
                }
        if self.phase_details:
            payload['phase_details'] = dict(self.phase_details)
        return payload


class IdealThermodynamics:
    """
    Ideal thermodynamic property calculator.
    
    Uses:
    - Ideal gas law for vapor phase
    - Raoult's law for VLE
    - Ideal mixing rules
    """

    LIQUID_TRANSPORT_TRACE_CUTOFF = 1.0e-6
    
    def __init__(
        self,
        components: list[str],
        db: Optional[ChemicalDatabase] = None,
        interaction_overrides: Optional[list[dict]] = None,
    ):
        """
        Initialize with list of component symbols.
        
        Args:
            components: List of chemical symbols (e.g., ['H2O', 'C2H5OH'])
            db: Chemical database (uses default if None)
        """
        self.db = db or get_database()
        self.components = list(components)
        self.process_components = list(components)
        self.permanent_solid_components: tuple[str, ...] = ()
        self.solid_particle_defaults: dict[str, dict[str, float]] = {}
        self.fluid_phase_model = 'VLE'
        self.interaction_overrides = list(interaction_overrides or [])
        self.estimated_interaction_metadata: dict[tuple[str, str], dict] = {}
        self._estimated_interaction_extrapolation_warnings: set[tuple] = set()
        self.activity_interaction_max_psat_bar: Optional[float] = 10.0
        self.activity_interaction_max_temperature_K: Optional[float] = None
        self._activity_interaction_psat_cap_cache: dict[
            tuple[str, float], float
        ] = {}
        self._activity_interaction_component_caps: dict[str, float] = {}
        self.props: dict[str, ChemicalProperties] = {}
        self._psat_cache: dict[tuple[str, float], float] = {}
        self._psat_coefficients_cache: dict[str, tuple[float, ...]] = {}
        self._cp_ideal_cache: dict[tuple[str, float], float] = {}
        self._cp_liquid_cache: dict[tuple[str, float], float] = {}
        self._cp_solid_cache: dict[tuple[str, float], float] = {}
        self._cp_integral_cache: dict[tuple[str, str, float, float], float] = {}
        self._enthalpy_ideal_cache: dict[tuple[str, float], float] = {}
        self._enthalpy_liquid_cache: dict[tuple[str, float], float] = {}
        self._enthalpy_solid_cache: dict[tuple[str, float], float] = {}
        self._entropy_ideal_cache: dict[tuple[str, float], float] = {}
        self._entropy_liquid_cache: dict[tuple[str, float], float] = {}
        self._entropy_solid_cache: dict[tuple[str, float], float] = {}
        self._hvap_cache: dict[str, float] = {}
        self._hvap_T_cache: dict[tuple[str, float], float] = {}
        self._liquid_molar_volume_cache: dict[tuple[str, float], float] = {}
        self._liquid_molar_volume_info_cache: dict[tuple[str, float], tuple[float, Optional[str]]] = {}
        self._solid_molar_volume_cache: dict[tuple[str, float], float] = {}
        self._viscosity_cache: dict[tuple[str, str, float, float], float] = {}
        self._surface_tension_cache: dict[tuple[str, float], float] = {}
        self._surface_tension_calculators: dict[tuple[str, ...], object] = {}
        self._liquid_molar_volume_sources: dict[str, list[dict]] = {}
        self._lazy_property_sources: dict[tuple[str, str, str, str], dict] = {}
        self._pure_saturation_temperature_cache: dict[tuple[str, float], float] = {}
        self._phi_sat_cache: dict[tuple[str, float], float] = {}
        self._poynting_cache: dict[tuple[str, float, float], float] = {}
        self._poynting_endpoint_warning_keys: set[tuple[str, str]] = set()
        self._k_values_cache: dict[tuple, dict[str, float]] = {}
        self._resolver_known_props: dict[str, dict] = {}
        self._perry_library = None
        self._perry_entries: dict[str, dict] = {}
        self._provided_cp_coeffs: dict[str, list[float]] = {}
        self._ideal_gas_cp_kernels: dict[str, object] = {}
        self._liquid_cp_kernels: dict[str, object] = {}
        self._solid_cp_kernels: dict[str, object] = {}
        self._henry_component_data_cache: dict[str, Optional[HenryComponentData]] = {}
        self._henry_temperature_anchor_cache: dict[tuple, tuple[float, float]] = {}
        self._aqueous_solvent_concentration_cache: dict[tuple, float] = {}
        self._aqueous_solvent_density_derivative_cache: dict[tuple, float] = {}
        self._quality_context_stack: list[dict] = []
        self._henry_quality_marked_contexts: set[tuple] = set()
        self._static_quality_marked_contexts: set[tuple] = set()
        self._runtime_initialized = False
        self.warnings: list[str] = []
        self._warning_keys: set[str] = set()
        
        # Load properties for each component
        for comp in components:
            if hasattr(self.db, 'get_user_component'):
                props = self.db.get_user_component(comp)
            else:
                props = self.db.get(comp)
            if props is None:
                raise ThermodynamicsError(f"Component '{comp}' not found in database")
            # Database records are shared hydrated templates. Thermodynamic
            # calculations attach per-run quality contexts and may fill local
            # fallback properties, so every model needs an isolated deep copy
            # rather than mutating the process-global ChemicalDatabase object.
            props = deepcopy(props)
            self.props[comp] = props
            self._hydrate_henry_properties(comp, props)
            self._resolver_known_props[comp] = props.to_dict()
            self._resolver_known_props[comp].setdefault('cas', props.CAS)
            self._resolver_known_props[comp].setdefault('antoine_source', 'chemicals.json')
            self._prebind_component_properties(comp, props)

    def configure_permanent_solids(
        self,
        process_components: Iterable[str],
        permanent_solid_components: Iterable[str],
        particle_defaults: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        """Attach process-only permanent solids without adding them to fluid backends."""
        process = list(dict.fromkeys(str(component) for component in process_components))
        solids = tuple(dict.fromkeys(str(component) for component in permanent_solid_components))
        unknown = sorted(set(solids) - set(process))
        if unknown:
            raise ThermodynamicsError(
                "Permanent-solid components are not process components: "
                + ', '.join(unknown)
            )
        fluid_expected = [component for component in process if component not in solids]
        if fluid_expected != list(self.components):
            raise ThermodynamicsError(
                "Thermodynamic fluid backend components do not match the "
                "conventional process-component subset"
            )

        self.process_components = process
        self.permanent_solid_components = solids
        self.solid_particle_defaults = {
            component: {
                key: float(value)
                for key, value in values.items()
                if value is not None
            }
            for component, values in (particle_defaults or {}).items()
            if component in solids
        }

        for comp in solids:
            if comp in self.props:
                continue
            if hasattr(self.db, 'get_user_component'):
                props = self.db.get_user_component(comp)
            else:
                props = self.db.get(comp)
            if props is None:
                raise ThermodynamicsError(
                    f"Permanent-solid component '{comp}' not found in database"
                )
            self.props[comp] = props
            known = props.to_dict()
            known.setdefault('cas', props.CAS)
            known.setdefault('antoine_source', 'chemicals.json')
            self._resolver_known_props[comp] = known

    @staticmethod
    def _normalize_fluid_phase_model(value: str) -> str:
        try:
            return normalize_fluid_phase_model(value)
        except ValueError as error:
            raise ThermodynamicsError(
                str(error)
            ) from error

    def set_fluid_phase_model(self, value: str) -> None:
        """Configure unconstrained stream flashes without affecting phase probes."""
        normalized = self._normalize_fluid_phase_model(value)
        if normalized != 'VLE' and self.components and not hasattr(self, 'flash3_TP'):
            raise ThermodynamicsError(
                f"Fluid phase model {normalized} requires an LLE-capable "
                "activity-coefficient thermodynamic model"
            )
        self.fluid_phase_model = normalized

    def configure_activity_interaction_limits(
        self,
        *,
        max_psat_bar: Optional[float] = 10.0,
        max_temperature_K: Optional[float] = None,
    ) -> None:
        if max_psat_bar is not None:
            max_psat_bar = float(max_psat_bar)
            if not math.isfinite(max_psat_bar) or max_psat_bar <= 0.0:
                raise ThermodynamicsError(
                    "Activity-interaction maximum Psat must be positive"
                )
        if max_temperature_K is not None:
            max_temperature_K = float(max_temperature_K)
            if not math.isfinite(max_temperature_K) or max_temperature_K <= 0.0:
                raise ThermodynamicsError(
                    "Activity-interaction maximum temperature must be positive"
                )
        self.activity_interaction_max_psat_bar = max_psat_bar
        self.activity_interaction_max_temperature_K = max_temperature_K
        self._activity_interaction_psat_cap_cache.clear()
        self._activity_interaction_component_caps = {
            component: self._compute_activity_interaction_component_temperature_limit(
                component
            )
            for component in self.components
        }
        for name in (
            '_activity_cache',
            '_nrtl_matrix_cache',
            '_uniquac_tau_cache',
            '_compiled_activity_cache',
            '_compiled_lle_cache',
        ):
            cache = getattr(self, name, None)
            if hasattr(cache, 'clear'):
                cache.clear()
        if hasattr(self, '_compiled_vlle_initialized'):
            self._compiled_vlle_initialized = False
            self._compiled_vlle = None

    def _activity_psat_cap_temperature(
        self,
        component: str,
        T: float,
    ) -> float:
        limit = self.activity_interaction_max_psat_bar
        if limit is None:
            return float(T)
        key = (component, float(limit))
        cached = self._activity_interaction_psat_cap_cache.get(key)
        if cached is not None:
            return min(float(T), cached)
        try:
            if self.Psat(component, float(T)) <= limit:
                return float(T)
        except Exception:
            return float(T)

        props = self.props.get(component)
        low = min(float(T), float(getattr(props, 'Tb', 0.0) or 0.0))
        if low <= 1.0:
            low = max(1.0, 0.5 * float(T))
        try:
            while low > 1.0 and self.Psat(component, low) > limit:
                low = max(1.0, 0.75 * low)
            if self.Psat(component, low) > limit:
                return float(T)
            high = float(T)
            for _ in range(70):
                middle = 0.5 * (low + high)
                if self.Psat(component, middle) > limit:
                    high = middle
                else:
                    low = middle
            cap = 0.5 * (low + high)
        except Exception:
            return float(T)
        self._activity_interaction_psat_cap_cache[key] = cap
        return min(float(T), cap)

    def _compute_activity_interaction_component_temperature_limit(
        self,
        component: str,
    ) -> float:
        limit = math.inf
        if self.activity_interaction_max_temperature_K is not None:
            limit = float(self.activity_interaction_max_temperature_K)
        if self.activity_interaction_max_psat_bar is None:
            return limit
        props = self.props.get(component)
        candidates = [
            float(value)
            for value in (
                getattr(props, 'Tc', None),
                (
                    2.0 * float(getattr(props, 'Tb', 0.0))
                    if getattr(props, 'Tb', None) else None
                ),
                limit if math.isfinite(limit) else None,
                1200.0,
            )
            if value is not None and math.isfinite(float(value))
            and float(value) > 1.0
        ]
        high = max(candidates, default=1200.0)
        pressure_limit = self._activity_psat_cap_temperature(component, high)
        try:
            if self.Psat(component, high) <= self.activity_interaction_max_psat_bar:
                pressure_limit = math.inf
        except Exception:
            pressure_limit = math.inf
        return min(limit, pressure_limit)

    def activity_interaction_component_temperature_limit(
        self,
        component: str,
    ) -> float:
        return self._activity_interaction_component_caps.get(component, math.inf)

    def activity_interaction_component_temperature_limits(self) -> list[float]:
        return [
            self.activity_interaction_component_temperature_limit(component)
            for component in self.components
        ]

    def activity_interaction_temperature(
        self,
        component1: str,
        component2: str,
        T: float,
    ) -> float:
        effective = float(T)
        if self.activity_interaction_max_temperature_K is not None:
            effective = min(
                effective,
                float(self.activity_interaction_max_temperature_K),
            )
        effective = min(
            effective,
            self.activity_interaction_component_temperature_limit(component1),
            self.activity_interaction_component_temperature_limit(component2),
        )
        return effective

    def activity_interaction_temperature_for_components(
        self,
        components,
        T: float,
    ) -> float:
        effective = float(T)
        if self.activity_interaction_max_temperature_K is not None:
            effective = min(
                effective,
                float(self.activity_interaction_max_temperature_K),
            )
        for component in components:
            effective = min(
                effective,
                self.activity_interaction_component_temperature_limit(component),
            )
        return effective

    def activity_interaction_clipping_active(
        self,
        T: float,
        components=None,
    ) -> bool:
        selected = self.components if components is None else components
        effective = self.activity_interaction_temperature_for_components(
            selected,
            T,
        )
        return effective < float(T) - 1.0e-10

    @staticmethod
    def _pooled_liquid_composition(
        liquid1_fraction: float,
        x1: dict[str, float],
        liquid2_fraction: float,
        x2: dict[str, float],
    ) -> dict[str, float]:
        total = max(0.0, liquid1_fraction) + max(0.0, liquid2_fraction)
        if total <= 1.0e-300:
            return dict(x1 or x2)
        components = set(x1) | set(x2)
        pooled = {
            component: (
                max(0.0, liquid1_fraction) * float(x1.get(component, 0.0))
                + max(0.0, liquid2_fraction) * float(x2.get(component, 0.0))
            ) / total
            for component in components
        }
        norm = sum(max(value, 0.0) for value in pooled.values())
        if norm > 0.0:
            pooled = {
                component: max(value, 0.0) / norm
                for component, value in pooled.items()
            }
        return pooled

    def _fluid_phase_equilibrium_TP(
        self,
        composition: dict[str, float],
        T: float,
        P: float,
    ) -> FluidPhaseEquilibrium:
        """Return the constrained VLE result used by the base fluid model."""
        V, x, y = self.flash_TP(composition, T, P)
        V = max(0.0, min(1.0, float(V)))
        return FluidPhaseEquilibrium(
            vapor_fraction=V,
            liquid1_fraction=1.0 - V,
            liquid2_fraction=0.0,
            y=dict(y or composition),
            x1=dict(x or composition),
            x2={},
            status=(
                'single_vapor' if V >= 1.0 - 1.0e-10
                else 'single_liquid' if V <= 1.0e-10
                else 'ordinary_vle'
            ),
            stability='vle_constrained',
            extra={},
        )

    def initialize(self) -> 'IdealThermodynamics':
        """Prepare persistent-process property backends without solving a state."""
        if self._runtime_initialized:
            return self

        # Default StreamState calculations consume resolved H, Cp, S, and rho,
        # so the resolver itself is an unconditional runtime dependency.
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..property_resolver import get_property_resolver
        else:
            from property_resolver import get_property_resolver
        get_property_resolver()

        # Import and identify CoolProp only for components with an exact bundled
        # CAS mapping. This prepares the provider the resolver will select while
        # avoiding CoolProp entirely for unsupported component sets.
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..property_resolution.coolprop import coolprop_reference_for
        else:
            from property_resolution.coolprop import coolprop_reference_for
        for comp in self.components:
            coolprop_reference_for(
                comp,
                self._resolver_known_props.get(comp, {}),
            )

        self._runtime_initialized = True
        return self

    @contextmanager
    def quality_context(self, **context):
        """Attach result-provenance metadata to lazy property lookups."""
        normalized = {
            str(key): value
            for key, value in context.items()
            if value is not None
        }
        self._quality_context_stack.append(normalized)
        try:
            yield
        finally:
            self._quality_context_stack.pop()

    def _current_quality_context(self) -> dict:
        context: dict = {}
        for item in self._quality_context_stack:
            context.update(item)
        context.setdefault('kind', 'unscoped')
        context.setdefault('phase', 'calculation')
        context.setdefault('affects_result', True)
        return context

    @staticmethod
    def _quality_context_key(context: dict) -> tuple:
        return (
            str(context.get('kind') or ''),
            str(context.get('unit_id') or ''),
            str(context.get('unit_type') or ''),
            str(context.get('stream_id') or ''),
            str(context.get('phase') or ''),
            bool(context.get('affects_result', True)),
        )

    @staticmethod
    def _quality_context_payload(context: dict) -> dict:
        payload = {
            'kind': str(context.get('kind') or 'unscoped'),
            'phase': str(context.get('phase') or 'calculation'),
            'affects_result': bool(context.get('affects_result', True)),
        }
        for key in ('unit_id', 'unit_type', 'stream_id', 'port_id', 'description'):
            if context.get(key) is not None:
                payload[key] = str(context[key])
        return payload

    def mark_property_source_context(self, comp: str, prop_names, **context) -> None:
        """Mark static property-source metadata as used by a result model."""
        props = self.props.get(comp)
        if props is None:
            return
        if isinstance(prop_names, str):
            prop_names = (prop_names,)
        payload = self._quality_context_payload({
            'kind': 'thermo_model',
            'phase': 'model_parameter',
            'affects_result': True,
            **context,
        })
        payload.setdefault('count', 1)
        for prop_name in prop_names:
            source = props.property_sources.get(prop_name)
            if not isinstance(source, dict):
                continue
            contexts = source.setdefault('contexts', [])
            key = self._quality_context_key(payload)
            for existing in contexts:
                if self._quality_context_key(existing) == key:
                    existing['count'] = int(existing.get('count') or 0) + int(payload['count'])
                    break
            else:
                contexts.append(dict(payload))

    def mark_property_source_context_once(
        self,
        comp: str,
        prop_names,
        phase: str,
        **context,
    ) -> None:
        """Mark static property-source usage once per component/context/phase."""
        current_context = self._current_quality_context()
        merged = {
            'kind': current_context.get('kind', 'thermo_model'),
            'unit_id': current_context.get('unit_id'),
            'unit_type': current_context.get('unit_type'),
            'stream_id': current_context.get('stream_id'),
            'affects_result': current_context.get('affects_result', True),
            **context,
            'phase': phase,
        }
        if isinstance(prop_names, str):
            prop_tuple = (prop_names,)
        else:
            prop_tuple = tuple(prop_names)
        key = (comp, prop_tuple, self._quality_context_key(merged))
        if key in self._static_quality_marked_contexts:
            return
        props = self.props.get(comp)
        if props is None:
            return
        if not any(isinstance(props.property_sources.get(prop), dict) for prop in prop_tuple):
            return
        self._static_quality_marked_contexts.add(key)
        self.mark_property_source_context(comp, prop_tuple, **merged)

    def _record_lazy_property_source(self, comp: str, prop_name: str, T: float, result) -> None:
        """Aggregate provenance for temperature-dependent resolver calls.

        The property resolver returns quality/source metadata for lazy
        temperature-dependent calls such as Psat(T), Cp(T), and liquid volume.
        Numeric thermodynamic caches only need the value, but the PFR quality
        report benefits from retaining compact provenance.  Aggregate by
        component/property/source/method so different methods over different
        temperature regions remain visible without emitting one row per call.
        """
        if result is None:
            return
        source = str(getattr(result, 'source', None) or 'unknown')
        method = str(getattr(result, 'method', None) or 'unknown')
        key = (str(comp), str(prop_name), source, method)
        try:
            T_value = float(T)
        except (TypeError, ValueError):
            T_value = None
        raw_quality = getattr(result, 'quality', None)
        try:
            quality = None if raw_quality is None else max(0.0, min(1.0, float(raw_quality)))
        except (TypeError, ValueError):
            quality = None

        entry = self._lazy_property_sources.get(key)
        if entry is None:
            entry = {
                'component': str(comp),
                'property': f"{prop_name}(T)",
                'source': source,
                'method': method,
                'quality': quality,
                'count': 0,
                'T_min': T_value,
                'T_max': T_value,
                'result_quality': None,
                'result_count': 0,
                'result_T_min': None,
                'result_T_max': None,
                'notes': [],
                'contexts': [],
                '_context_index': {},
            }
            self._lazy_property_sources[key] = entry

        entry['count'] += 1
        if quality is not None:
            previous_quality = entry.get('quality')
            entry['quality'] = quality if previous_quality is None else min(previous_quality, quality)
        if T_value is not None:
            entry['T_min'] = T_value if entry.get('T_min') is None else min(entry['T_min'], T_value)
            entry['T_max'] = T_value if entry.get('T_max') is None else max(entry['T_max'], T_value)

        context = self._current_quality_context()
        context_key = self._quality_context_key(context)
        context_index = entry.setdefault('_context_index', {})
        context_position = context_index.get(context_key)
        if context_position is None:
            context_record = self._quality_context_payload(context)
            context_record['count'] = 0
            context_record['T_min'] = T_value
            context_record['T_max'] = T_value
            entry.setdefault('contexts', []).append(context_record)
            context_position = len(entry['contexts']) - 1
            context_index[context_key] = context_position
        context_record = entry['contexts'][context_position]
        context_record['count'] += 1
        if T_value is not None:
            context_record['T_min'] = (
                T_value if context_record.get('T_min') is None
                else min(context_record['T_min'], T_value)
            )
            context_record['T_max'] = (
                T_value if context_record.get('T_max') is None
                else max(context_record['T_max'], T_value)
            )
        if context_record.get('affects_result', True):
            entry['result_count'] += 1
            if quality is not None:
                previous_result_quality = entry.get('result_quality')
                entry['result_quality'] = (
                    quality if previous_result_quality is None
                    else min(previous_result_quality, quality)
                )
            if T_value is not None:
                entry['result_T_min'] = (
                    T_value if entry.get('result_T_min') is None
                    else min(entry['result_T_min'], T_value)
                )
                entry['result_T_max'] = (
                    T_value if entry.get('result_T_max') is None
                    else max(entry['result_T_max'], T_value)
                )

        notes = str(getattr(result, 'notes', None) or '').strip()
        if notes:
            stored_notes = entry.setdefault('notes', [])
            if notes not in stored_notes:
                if len(stored_notes) < 2:
                    stored_notes.append(notes)
                else:
                    entry['additional_note_count'] = entry.get('additional_note_count', 0) + 1

        if len(self._lazy_property_sources) > 20000:
            self._lazy_property_sources.clear()

    def lazy_property_quality_sources(self) -> list[dict]:
        """Return aggregate provenance rows for lazy T-dependent properties."""
        rows = []
        for entry in self._lazy_property_sources.values():
            row = {
                key: value for key, value in entry.items()
                if key != '_context_index'
            }
            row['contexts'] = [dict(context) for context in entry.get('contexts', [])]
            rows.append(row)
        return rows

    @staticmethod
    def _henry_grade_quality(grade: Optional[str]) -> Optional[float]:
        return {
            'A+': 0.99, 'A': 0.96, 'A-': 0.92,
            'B+': 0.89, 'B': 0.86, 'B-': 0.82,
            'C+': 0.78, 'C': 0.74, 'C-': 0.69,
            'D+': 0.64, 'D': 0.58, 'D-': 0.50,
            'F': 0.30,
        }.get(str(grade or '').upper())

    def _hydrate_henry_properties(self, comp: str, props: ChemicalProperties) -> None:
        record = get_henry_constant_database().get(getattr(props, 'CAS', ''))
        if record is None:
            return
        if getattr(props, 'henry_Hcp', None) is None:
            props.henry_Hcp = record.hcp_298
            props.property_sources['henry_Hcp'] = {
                'source': 'Sander Henry database',
                'method': 'henry_database_hcp_298',
                'quality': (
                    record.quality_score_h
                    if record.quality_score_h is not None
                    else self._henry_grade_quality(record.quality_h)
                ),
                'grade': record.quality_h,
                'notes': 'Hcp at 298.15 K [mol/(m3*Pa)] from bundled CAS-keyed database',
            }
        if getattr(props, 'henry_B', None) is None and record.B is not None:
            props.henry_B = record.B
            props.property_sources['henry_B'] = {
                'source': 'Sander Henry database',
                'method': 'henry_database_temperature_coefficient',
                'quality': (
                    record.quality_score_b
                    if record.quality_score_b is not None
                    else self._henry_grade_quality(record.quality_b)
                ),
                'grade': record.quality_b,
                'notes': 'd(ln Hcp)/d(1/T) [K] from bundled CAS-keyed database',
            }
        if getattr(props, 'henry_Vinf', None) is None and record.vinf_298_cm3_per_mol is not None:
            props.henry_Vinf = record.vinf_298_cm3_per_mol
            props.henry_Vinf_uncertainty = record.vinf_uncertainty_cm3_per_mol
            props.property_sources['henry_Vinf'] = {
                'source': record.vinf_source or 'Zhou and Battino (2001)',
                'method': 'henry_vinf_measured_298K',
                'quality': 0.98,
                'doi': record.vinf_doi,
                'notes': (
                    'Infinite-dilution partial molar volume in water at 298.15 K '
                    '[cm3/mol]'
                ),
            }

    def add_warning(self, message: str) -> None:
        """Record a simulation warning once so reports can surface it."""
        text = str(message).strip()
        if not text or text in self._warning_keys:
            return
        self._warning_keys.add(text)
        self.warnings.append(text)

    def extend_warnings(self, messages) -> None:
        for message in messages or []:
            self.add_warning(message)

    def _component_label(self, comp: str) -> str:
        props = self.props.get(comp)
        name = getattr(props, 'name', None)
        if name and name != comp:
            return f"{comp} ({name})"
        return comp

    def _allow_online_lookup_for_component(self, comp: str) -> bool:
        return (self._resolver_known_props.get(comp) or {}).get('_allow_online_lookup') is not False

    def _warn_rk_binary_interactions_unavailable(self, context: str = "RK EOS") -> None:
        if len(self.components) < 2:
            return
        self.add_warning(
            f"{context} binary interaction parameters are not implemented; "
            "using classical van der Waals mixing with k_ij=0 for all component pairs."
        )

    def _prebind_component_properties(self, comp: str, props: ChemicalProperties) -> None:
        """Bind hot pure-property evaluators once per thermo object."""
        known = self._resolver_known_props.get(comp, {})
        cp_coeffs = known.get('Cp_coeffs')
        if cp_coeffs and len(cp_coeffs) >= 4:
            self._provided_cp_coeffs[comp] = list(cp_coeffs)
        self._prebind_provided_liquid_molar_volume_source(comp, known)

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..perry_properties import get_perry_property_library
            else:
                from perry_properties import get_perry_property_library
            library = get_perry_property_library()
        except Exception:
            return

        identifiers = [props.CAS, props.symbol, props.name, comp]
        formula = getattr(props, 'formula', None)
        if self._formula_is_safe_perry_prebind(comp, formula, props):
            identifiers.append(formula)

        for identifier in identifiers:
            if not identifier:
                continue
            entry = library.get(identifier)
            if entry:
                self._perry_library = library
                self._perry_entries[comp] = entry
                self._prebind_perry_liquid_molar_volume_sources(comp, entry)
                return

    @staticmethod
    def _formula_is_safe_perry_prebind(comp: str, formula: Optional[str], props: ChemicalProperties) -> bool:
        """Only use a bare formula for Perry binding when it names this component."""
        if not formula:
            return False
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..compound_identity import get_compound_identity_resolver, looks_like_formula
            else:
                from compound_identity import get_compound_identity_resolver, looks_like_formula
            if not looks_like_formula(comp):
                return False
            resolver = get_compound_identity_resolver()
            formula_symbol = resolver.resolve_symbol(formula, allow_formula=True)
            comp_symbol = resolver.resolve_symbol(comp, allow_formula=True)
            prop_symbol = resolver.resolve_symbol(props.symbol, allow_formula=False)
            return bool(formula_symbol and formula_symbol in {comp_symbol, prop_symbol, props.symbol})
        except Exception:
            return str(comp).strip().upper() == str(formula).strip().upper()

    def _prebind_provided_liquid_molar_volume_source(self, comp: str, known: dict) -> None:
        correlations = known.get('property_correlations') or {}
        if not isinstance(correlations, dict):
            return
        correlation = correlations.get('rhol')
        if not correlation or not known.get('MW'):
            return
        self._liquid_molar_volume_sources.setdefault(comp, []).append({
            'kind': 'provided_rhol',
            'correlation': correlation,
            'MW': float(known['MW']),
            'Tmin': correlation.get('Tmin_K'),
            'Tmax': correlation.get('Tmax_K'),
        })

    def _prebind_perry_liquid_molar_volume_sources(self, comp: str, entry: dict) -> None:
        sources = self._liquid_molar_volume_sources.setdefault(comp, [])
        for row in entry.get('liquid_density', []) or []:
            if row.get('equation_id') not in {100, 105}:
                continue
            sources.append({
                'kind': 'perry_density',
                'row': row,
                'Tmin': row.get('T_min_K'),
                'Tmax': row.get('T_max_K'),
            })

    @staticmethod
    def _source_covers_temperature(source: dict, T: float) -> bool:
        Tmin = source.get('Tmin')
        Tmax = source.get('Tmax')
        if Tmin is not None and T < float(Tmin) - 1e-9:
            return False
        if Tmax is not None and T > float(Tmax) + 1e-9:
            return False
        return True

    @staticmethod
    def _source_boundary_temperature(source: dict, T: float) -> Optional[float]:
        Tmin = source.get('Tmin')
        Tmax = source.get('Tmax')
        if Tmin is None or Tmax is None:
            return None
        Tmin = float(Tmin)
        Tmax = float(Tmax)
        if T < Tmin - 1e-9:
            return Tmin
        if T > Tmax + 1e-9:
            return Tmax
        return None

    @staticmethod
    def _source_range_width(source: dict) -> float:
        Tmin = source.get('Tmin')
        Tmax = source.get('Tmax')
        if Tmin is None or Tmax is None:
            return float('inf')
        return max(float(Tmax) - float(Tmin), 1e-12)

    @staticmethod
    def _source_range_center_distance(source: dict, T: float) -> float:
        Tmin = source.get('Tmin')
        Tmax = source.get('Tmax')
        if Tmin is None or Tmax is None:
            return 0.0
        return abs(0.5 * (float(Tmin) + float(Tmax)) - T)

    @staticmethod
    def _correlation_coefficients(correlation: dict) -> dict[str, float]:
        coefficients = correlation.get('coefficients') or {}
        if not isinstance(coefficients, dict):
            return {}
        return {
            str(key): float(value)
            for key, value in coefficients.items()
            if value is not None
        }

    def _evaluate_prebound_liquid_molar_volume_source(self, source: dict, T: float) -> Optional[float]:
        try:
            if source.get('kind') == 'provided_rhol':
                correlation = source['correlation']
                equation = str(correlation.get('equation', '')).lower()
                coeffs = self._correlation_coefficients(correlation)
                x = (T - 298.15) / 100.0
                if equation == 'poly_x':
                    rho_kg_m3 = sum(
                        coeffs.get(name, 0.0) * x**power
                        for power, name in enumerate(('A', 'B', 'C', 'D', 'E', 'F'))
                    )
                elif equation == 'exp_poly_x':
                    exponent = sum(
                        coeffs.get(name, 0.0) * x**power
                        for power, name in enumerate(('A', 'B', 'C', 'D', 'E', 'F'))
                    )
                    rho_kg_m3 = math.exp(exponent)
                else:
                    return None
                if rho_kg_m3 <= 0.0:
                    return None
                return float(source['MW']) / rho_kg_m3

            if source.get('kind') == 'perry_density':
                row = source['row']
                coeffs = row.get('coefficients', [])
                equation_id = row.get('equation_id')
                if equation_id == 100:
                    rho_mol_dm3 = sum(coef * T**power for power, coef in enumerate(coeffs))
                elif equation_id == 105 and len(coeffs) >= 4:
                    C1, C2, C3, C4 = coeffs[:4]
                    tau = 1.0 - T / C3
                    if tau < 0.0 and abs(C4 - round(C4)) > 1e-12:
                        return None
                    rho_mol_dm3 = C1 / (C2 ** (1.0 + tau**C4))
                else:
                    return None
                if rho_mol_dm3 <= 0.0:
                    return None
                return 1.0 / rho_mol_dm3
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
        return None

    def _prebound_liquid_molar_volume(self, comp: str, T: float) -> Optional[tuple[float, Optional[str]]]:
        sources = self._liquid_molar_volume_sources.get(comp) or []
        if not sources:
            return None

        provided_sources = [source for source in sources if source.get('kind') == 'provided_rhol']
        perry_sources = [source for source in sources if source.get('kind') == 'perry_density']
        ordered_groups = [
            provided_sources,
            sorted(
                perry_sources,
                key=lambda source: (
                    self._source_range_width(source),
                    self._source_range_center_distance(source, T),
                ),
            ),
        ]
        for group in ordered_groups:
            for source in group:
                if not self._source_covers_temperature(source, T):
                    continue
                value = self._evaluate_prebound_liquid_molar_volume_source(source, T)
                if value is not None:
                    return value, None

        endpoint_candidates = []
        for source in sources:
            boundary_T = self._source_boundary_temperature(source, T)
            if boundary_T is None:
                continue
            distance = abs(boundary_T - T)
            if distance <= LIQUID_VOLUME_EXTRAPOLATION_LIMIT_K + 1e-9:
                endpoint_candidates.append((distance, boundary_T, source))
        endpoint_candidates.sort(key=lambda item: item[:2])

        for _, boundary_T, source in endpoint_candidates:
            value = self._evaluate_prebound_liquid_molar_volume_source(source, T)
            if value is not None:
                endpoint_note = (
                    f"extrapolated from nearest correlation boundary "
                    f"T={boundary_T:.1f} K to T={T:.1f} K"
                )
                endpoint_key = (comp, float(T))
                self._set_limited_cache(self._liquid_molar_volume_cache, endpoint_key, value)
                return value, endpoint_note
        return None

    def _liquid_molar_volume_for_poynting(self, comp: str, T: float) -> tuple[Optional[float], Optional[str]]:
        cache_key = (comp, float(T))
        cached_info = self._liquid_molar_volume_info_cache.get(cache_key)
        if cached_info is not None:
            return cached_info
        cached_volume = self._liquid_molar_volume_cache.get(cache_key)
        if cached_volume is not None:
            info = (cached_volume, None)
            self._liquid_molar_volume_info_cache[cache_key] = info
            return info

        info = self._prebound_liquid_molar_volume(comp, T)
        if info is not None:
            volume, _ = info
            self._set_limited_cache(self._liquid_molar_volume_cache, cache_key, volume)
            if len(self._liquid_molar_volume_info_cache) > 20000:
                self._liquid_molar_volume_info_cache.clear()
            self._liquid_molar_volume_info_cache[cache_key] = info
            return info

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            result = get_property_resolver().resolve_liquid_molar_volume_nearest(
                comp,
                T,
                self._resolver_known_props.get(comp),
            )
        except Exception:
            return None, None

        self._record_lazy_property_source(comp, 'liquid_molar_volume', T, result)
        endpoint_note = None
        if result.method.endswith('_nearest_temperature'):
            endpoint_note = result.notes.rsplit('used nearest endpoint ', 1)[-1]
        info = (result.value, endpoint_note)
        self._set_limited_cache(self._liquid_molar_volume_cache, cache_key, result.value)
        if len(self._liquid_molar_volume_info_cache) > 20000:
            self._liquid_molar_volume_info_cache.clear()
        self._liquid_molar_volume_info_cache[cache_key] = info
        return info

    def _fast_perry_cp(self, comp: str, T: float, phase: str) -> Optional[float]:
        if self._perry_library is None:
            return None
        entry = self._perry_entries.get(comp)
        if entry is None:
            return None
        evaluated = self._perry_library.heat_capacity_value_from_entry(entry, T, phase)
        if evaluated is None:
            return None
        value, _, _ = evaluated
        return value

    def _ideal_gas_cp_kernel(self, comp: str):
        """Resolve and retain one executable ideal-gas Cp curve per component."""
        if comp in self._ideal_gas_cp_kernels:
            return self._ideal_gas_cp_kernels[comp]
        props = self.props.get(comp)
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            kernel = get_property_resolver().resolve_ideal_gas_cp_kernel(
                props.symbol if props else comp,
                self._resolver_known_props.get(comp),
                allow_online=self._allow_online_lookup_for_component(comp),
            )
        except Exception:
            kernel = None
        self._ideal_gas_cp_kernels[comp] = kernel
        if kernel is not None:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..property_resolver import PropertyResolutionResult
                else:
                    from property_resolver import PropertyResolutionResult
                self._record_lazy_property_source(
                    comp,
                    'Cp_ideal_gas',
                    T_REF,
                    PropertyResolutionResult(
                        value=kernel.cp(T_REF),
                        source=kernel.source,
                        method=kernel.method,
                        quality=kernel.quality_at(T_REF),
                        notes=kernel.notes,
                    ),
                )
            except Exception:
                pass
        return kernel

    def _record_ideal_gas_cp_kernel_range_use(self, comp: str, T: float, kernel) -> None:
        """Record degraded provenance when a thermal path leaves the fit range."""
        if kernel is None or kernel.covers(T):
            return
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import PropertyResolutionResult
            else:
                from property_resolver import PropertyResolutionResult
            evaluation = kernel.evaluate(T)
            self._record_lazy_property_source(
                comp,
                'Cp_ideal_gas',
                T,
                PropertyResolutionResult(
                    value=evaluation.value,
                    source=kernel.source,
                    method=kernel.method,
                    quality=evaluation.quality,
                    notes=evaluation.range_note,
                ),
            )
        except Exception:
            pass

    def _liquid_cp_kernel(self, comp: str):
        """Resolve and retain one executable ordinary-liquid Cp curve."""
        if comp in self._liquid_cp_kernels:
            return self._liquid_cp_kernels[comp]
        props = self.props.get(comp)
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            kernel = get_property_resolver().resolve_liquid_cp_kernel(
                props.symbol if props else comp,
                self._resolver_known_props.get(comp),
                allow_online=self._allow_online_lookup_for_component(comp),
            )
        except Exception:
            kernel = None
        self._liquid_cp_kernels[comp] = kernel
        if kernel is not None:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..property_resolver import PropertyResolutionResult
                else:
                    from property_resolver import PropertyResolutionResult
                self._record_lazy_property_source(
                    comp,
                    'Cp_liquid',
                    T_REF,
                    PropertyResolutionResult(
                        value=kernel.cp(T_REF),
                        source=kernel.source,
                        method=kernel.method,
                        quality=kernel.quality_at(T_REF),
                        notes=kernel.notes,
                    ),
                )
            except Exception:
                pass
        return kernel

    def _record_liquid_cp_kernel_range_use(self, comp: str, T: float, kernel) -> None:
        """Record degraded provenance when liquid Cp leaves its fitted range."""
        if kernel is None or kernel.covers(T):
            return
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import PropertyResolutionResult
            else:
                from property_resolver import PropertyResolutionResult
            evaluation = kernel.evaluate(T)
            self._record_lazy_property_source(
                comp,
                'Cp_liquid',
                T,
                PropertyResolutionResult(
                    value=evaluation.value,
                    source=kernel.source,
                    method=kernel.method,
                    quality=evaluation.quality,
                    notes=evaluation.range_note,
                ),
            )
        except Exception:
            pass

    def _solid_cp_kernel(self, comp: str):
        """Resolve and retain one executable material-form-aware solid Cp curve."""
        if comp in self._solid_cp_kernels:
            return self._solid_cp_kernels[comp]
        props = self.props.get(comp)
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            kernel = get_property_resolver().resolve_solid_cp_kernel(
                props.symbol if props else comp,
                self._resolver_known_props.get(comp),
                allow_online=self._allow_online_lookup_for_component(comp),
            )
        except Exception:
            kernel = None
        self._solid_cp_kernels[comp] = kernel
        if kernel is not None:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..property_resolver import PropertyResolutionResult
                else:
                    from property_resolver import PropertyResolutionResult
                active = kernel.active_kernel(T_REF)
                self._record_lazy_property_source(
                    comp,
                    'Cp_solid',
                    T_REF,
                    PropertyResolutionResult(
                        value=kernel.cp(T_REF), source=active.source,
                        method=active.method, quality=kernel.quality_at(T_REF),
                        notes=active.notes,
                    ),
                )
            except Exception:
                pass
        return kernel

    def _record_solid_cp_kernel_range_use(self, comp: str, T: float, kernel) -> None:
        if kernel is None or kernel.covers(T):
            return
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import PropertyResolutionResult
            else:
                from property_resolver import PropertyResolutionResult
            evaluation = kernel.evaluate(T)
            active = kernel.active_kernel(T)
            self._record_lazy_property_source(
                comp,
                'Cp_solid',
                T,
                PropertyResolutionResult(
                    value=evaluation.value, source=active.source,
                    method=active.method, quality=evaluation.quality,
                    notes=evaluation.range_note,
                ),
            )
        except Exception:
            pass

    @staticmethod
    def _temperature_range_covers(row: dict, T1: float, T2: float) -> bool:
        lo = min(T1, T2)
        hi = max(T1, T2)
        return (
            row.get('T_min_K', -math.inf) <= lo + 1e-9
            and hi <= row.get('T_max_K', math.inf) + 1e-9
        )

    @staticmethod
    def _polynomial_cp_integral(
        coeffs,
        T1: float,
        T2: float,
        *,
        coeff_units: str,
    ) -> Optional[float]:
        """Analytic integral of sum(C_i*T^i), returned as kJ/mol."""
        try:
            total = 0.0
            for power, coef in enumerate(coeffs):
                total += float(coef) * (
                    T2 ** (power + 1) - T1 ** (power + 1)
                ) / (power + 1)
        except (TypeError, ValueError, OverflowError):
            return None
        if coeff_units == 'J_per_kmol_K':
            return total / 1.0e6
        return total / 1000.0

    @staticmethod
    def _coth(value: float) -> float:
        if value > 350.0:
            return 1.0
        if value < -350.0:
            return -1.0
        return math.cosh(value) / math.sinh(value)

    @classmethod
    def _perry_hyperbolic_cp_integral(
        cls,
        coeffs,
        T1: float,
        T2: float,
    ) -> Optional[float]:
        """Analytic Perry ideal-gas hyperbolic Cp integral, returned as kJ/mol."""
        if len(coeffs) < 5:
            return None
        try:
            C1, C2, C3, C4, C5 = [float(value) for value in coeffs[:5]]
            total = C1 * (T2 - T1)
            total += C2 * C3 * (cls._coth(C3 / T2) - cls._coth(C3 / T1))
            total += -C4 * C5 * (math.tanh(C5 / T2) - math.tanh(C5 / T1))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
        return total / 1.0e6

    @staticmethod
    def _provided_poly_x_cp_integral(correlation: dict, T1: float, T2: float) -> Optional[float]:
        """Analytic portable poly_x Cp integral, returned as kJ/mol."""
        coefficients = correlation.get('coefficients') or {}
        if not isinstance(coefficients, dict):
            return None
        x1 = (T1 - 298.15) / 100.0
        x2 = (T2 - 298.15) / 100.0
        try:
            total = 0.0
            for power, name in enumerate(('A', 'B', 'C', 'D', 'E', 'F')):
                coef = float(coefficients.get(name, 0.0))
                total += coef * (x2 ** (power + 1) - x1 ** (power + 1)) / (power + 1)
        except (TypeError, ValueError, OverflowError):
            return None
        return 100.0 * total / 1000.0

    def _provided_correlation_cp_integral(
        self,
        comp: str,
        T1: float,
        T2: float,
        correlation_key: str,
    ) -> Optional[float]:
        """Analytic integral for portable Cp correlations when available."""
        props = self._resolver_known_props.get(comp) or {}
        correlations = props.get('property_correlations') or {}
        if not isinstance(correlations, dict):
            return None
        correlation = correlations.get(correlation_key)
        if not correlation or not self._temperature_range_covers(correlation, T1, T2):
            return None
        equation = str(correlation.get('equation', '')).lower()
        if equation == 'poly_x':
            return self._provided_poly_x_cp_integral(correlation, T1, T2)
        return None

    def _perry_cp_integral(
        self,
        comp: str,
        T1: float,
        T2: float,
        phase: str,
    ) -> Optional[float]:
        """Analytic Perry Cp integral for supported active forms, returned as kJ/mol."""
        entry = self._perry_entries.get(comp)
        if entry is None:
            return None

        if phase == 'liquid':
            rows = entry.get('liquid_heat_capacity', []) or []
            for row in rows:
                if row.get('equation_id') != 100 or not self._temperature_range_covers(row, T1, T2):
                    continue
                return self._polynomial_cp_integral(
                    row.get('coefficients', []),
                    T1,
                    T2,
                    coeff_units='J_per_kmol_K',
                )
            return None

        if phase == 'ideal_gas':
            rows = entry.get('ideal_gas_heat_capacity_polynomial', []) or []
            for row in rows:
                if not self._temperature_range_covers(row, T1, T2):
                    continue
                return self._polynomial_cp_integral(
                    row.get('coefficients', []),
                    T1,
                    T2,
                    coeff_units='J_per_kmol_K',
                )

            rows = entry.get('ideal_gas_heat_capacity_hyperbolic', []) or []
            for row in rows:
                if not self._temperature_range_covers(row, T1, T2):
                    continue
                return self._perry_hyperbolic_cp_integral(row.get('coefficients', []), T1, T2)
        return None

    def _integrate_cp_analytic(self, comp: str, T1: float, T2: float, phase: str) -> Optional[float]:
        """Return analytic Cp integral [kJ/mol] for supported sources."""
        if abs(T2 - T1) < 1e-12:
            return 0.0

        cache_key = (phase, comp, float(T1), float(T2))
        cached = self._cp_integral_cache.get(cache_key)
        if cached is not None:
            return cached

        value = None
        if phase == 'ideal_gas':
            kernel = self._ideal_gas_cp_kernel(comp)
            if kernel is not None:
                self._record_ideal_gas_cp_kernel_range_use(comp, T1, kernel)
                self._record_ideal_gas_cp_kernel_range_use(comp, T2, kernel)
                value = kernel.delta_h(T1, T2) / 1000.0
            if value is None:
                coeffs = self._provided_cp_coeffs.get(comp)
                if coeffs and len(coeffs) >= 4:
                    value = self._polynomial_cp_integral(
                        coeffs[:4],
                        T1,
                        T2,
                        coeff_units='J_per_mol_K',
                    )
            if value is None:
                value = self._perry_cp_integral(comp, T1, T2, 'ideal_gas')
            if value is None:
                value = self._provided_correlation_cp_integral(comp, T1, T2, 'Cpg')
        elif phase == 'liquid':
            kernel = self._liquid_cp_kernel(comp)
            if kernel is not None:
                self._record_liquid_cp_kernel_range_use(comp, T1, kernel)
                self._record_liquid_cp_kernel_range_use(comp, T2, kernel)
                value = kernel.delta_h(T1, T2) / 1000.0
        elif phase == 'solid':
            kernel = self._solid_cp_kernel(comp)
            if kernel is not None:
                self._record_solid_cp_kernel_range_use(comp, T1, kernel)
                self._record_solid_cp_kernel_range_use(comp, T2, kernel)
                value = kernel.delta_h(T1, T2) / 1000.0

        if value is None and phase != 'liquid':
            props = self.props.get(comp)
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..property_resolver import get_property_resolver
                else:
                    from property_resolver import get_property_resolver
                integral = get_property_resolver().integrate_cp_online(
                    props.symbol if props else comp,
                    T1,
                    T2,
                    phase,
                    props=self._resolver_known_props.get(comp),
                )
                if integral is not None:
                    value = integral.value
            except Exception:
                value = None

        if value is not None:
            return self._set_limited_cache(self._cp_integral_cache, cache_key, value)
        return None

    def _fast_perry_hvap(self, comp: str, T: float) -> Optional[float]:
        if self._perry_library is None:
            return None
        entry = self._perry_entries.get(comp)
        if entry is None:
            return None
        evaluated = self._perry_library.heat_of_vaporization_value_from_entry(entry, T)
        if evaluated is None:
            return None
        value, _, _ = evaluated
        return value
    
    def mixture_MW(self, composition: dict[str, float]) -> float:
        """Calculate mixture molecular weight [kg/kmol]"""
        mw = 0.0
        for comp, z in composition.items():
            if comp in self.props:
                self.mark_property_source_context_once(
                    comp,
                    'MW',
                    phase='molecular_weight',
                )
                mw += z * self.props[comp].MW
        return mw

    def _composition_cache_key(self, composition: dict[str, float]) -> tuple[tuple[str, float], ...]:
        """Exact component-order composition key for repeated solver evaluations."""
        return tuple(
            (comp, float(composition.get(comp, 0.0)))
            for comp in self.components
        )

    def _k_values_cache_key(
        self,
        kind: str,
        T: float,
        P: float,
        composition: dict[str, float],
    ) -> tuple:
        return (kind, float(T), float(P), self._composition_cache_key(composition))

    def _get_cached_k_values(self, key: tuple) -> Optional[dict[str, float]]:
        cached = self._k_values_cache.get(key)
        return dict(cached) if cached is not None else None

    def _set_cached_k_values(self, key: tuple, values: dict[str, float]) -> dict[str, float]:
        if len(self._k_values_cache) > 20000:
            self._k_values_cache.clear()
        self._k_values_cache[key] = dict(values)
        return values

    @staticmethod
    def _set_limited_cache(cache: dict, key, value, limit: int = 20000):
        if len(cache) > limit:
            cache.clear()
        cache[key] = value
        return value
    
    def Cp_ideal_gas(self, comp: str, T: float) -> float:
        """
        Ideal gas heat capacity [J/mol-K] at temperature T [K]
        """
        cache_key = (comp, float(T))
        cached = self._cp_ideal_cache.get(cache_key)
        if cached is not None:
            return cached
        kernel = self._ideal_gas_cp_kernel(comp)
        if kernel is not None:
            try:
                value = kernel.cp(T)
                self._record_ideal_gas_cp_kernel_range_use(comp, T, kernel)
                return self._set_limited_cache(self._cp_ideal_cache, cache_key, value)
            except Exception:
                pass
        known = self._resolver_known_props.get(comp) or {}
        if 'Cpg' in (known.get('property_correlations') or {}):
            props = self.props.get(comp)
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..property_resolver import get_property_resolver
                else:
                    from property_resolver import get_property_resolver
                result = get_property_resolver().resolve_heat_capacity(
                    props.symbol if props else comp,
                    T,
                    phase='ideal_gas',
                    props=known,
                    allow_online=self._allow_online_lookup_for_component(comp),
                )
                self._record_lazy_property_source(comp, 'Cp_ideal_gas', T, result)
                value = result.value
                return self._set_limited_cache(self._cp_ideal_cache, cache_key, value)
            except Exception:
                pass
        coeffs = self._provided_cp_coeffs.get(comp)
        if coeffs:
            self.mark_property_source_context_once(
                comp,
                'Cp_coeffs',
                phase='ideal_gas_heat_capacity',
            )
            value = coeffs[0] + coeffs[1]*T + coeffs[2]*T**2 + coeffs[3]*T**3
            return self._set_limited_cache(self._cp_ideal_cache, cache_key, value)
        value = self._fast_perry_cp(comp, T, 'ideal_gas')
        if value is not None:
            return self._set_limited_cache(self._cp_ideal_cache, cache_key, value)
        props = self.props.get(comp)
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            result = get_property_resolver().resolve_heat_capacity(
                props.symbol if props else comp,
                T,
                phase='ideal_gas',
                props=self._resolver_known_props.get(comp),
                allow_online=self._allow_online_lookup_for_component(comp),
            )
            self._record_lazy_property_source(comp, 'Cp_ideal_gas', T, result)
            value = result.value
            return self._set_limited_cache(self._cp_ideal_cache, cache_key, value)
        except Exception:
            return self._set_limited_cache(self._cp_ideal_cache, cache_key, 33.0)
    
    def Cp_liquid(self, comp: str, T: float) -> float:
        """
        Liquid heat capacity [J/mol-K]
        """
        cache_key = (comp, float(T))
        cached = self._cp_liquid_cache.get(cache_key)
        if cached is not None:
            return cached
        kernel = self._liquid_cp_kernel(comp)
        if kernel is None:
            raise ThermodynamicsError(
                f"Cannot resolve ordinary-liquid heat capacity for '{comp}'."
            )
        try:
            value = kernel.cp(T)
            self._record_liquid_cp_kernel_range_use(comp, T, kernel)
        except Exception as exc:
            raise ThermodynamicsError(
                f"Cannot evaluate ordinary-liquid heat capacity for '{comp}' at T={T:.1f} K."
            ) from exc
        return self._set_limited_cache(self._cp_liquid_cache, cache_key, value)

    def Cp_solid(self, comp: str, T: float) -> float:
        """Solid constant-pressure heat capacity [J/mol-K]."""
        cache_key = (comp, float(T))
        cached = self._cp_solid_cache.get(cache_key)
        if cached is not None:
            return cached
        kernel = self._solid_cp_kernel(comp)
        if kernel is None:
            raise ThermodynamicsError(f"Cannot resolve solid heat capacity for '{comp}'.")
        try:
            value = kernel.cp(T)
            self._record_solid_cp_kernel_range_use(comp, T, kernel)
        except Exception as exc:
            raise ThermodynamicsError(
                f"Cannot evaluate solid heat capacity for '{comp}' at T={T:.1f} K."
            ) from exc
        return self._set_limited_cache(self._cp_solid_cache, cache_key, value)

    def _normal_hvap(self, comp: str) -> float:
        """Heat of vaporization near normal boiling point [kJ/mol]."""
        cached = self._hvap_cache.get(comp)
        if cached is not None:
            return cached
        props = self.props.get(comp)
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            result = get_property_resolver().resolve_hvap(
                props.symbol if props else comp,
                self._resolver_known_props.get(comp),
                allow_online=self._allow_online_lookup_for_component(comp),
            )
            self._record_lazy_property_source(comp, 'Hvap', T_REF, result)
            value = result.value
        except Exception:
            value = None
        if value is None:
            self.mark_property_source_context_once(
                comp,
                'Hvap',
                phase='heat_of_vaporization',
            )
            value = props.Hvap if (props and props.Hvap) else 30.0
        return self._set_limited_cache(self._hvap_cache, comp, float(value))

    def Hvap_at_T(self, comp: str, T: float) -> float:
        """Temperature-dependent heat of vaporization [kJ/mol]."""
        cache_key = (comp, float(T))
        cached = self._hvap_T_cache.get(cache_key)
        if cached is not None:
            return cached

        props = self.props.get(comp)
        if props is not None and props.Hvap is not None:
            self.mark_property_source_context_once(
                comp,
                'Hvap',
                phase='heat_of_vaporization',
            )
        known = self._resolver_known_props.get(comp)
        sources = (known or {}).get('property_sources') or {}
        pfd_hvap = (sources.get('Hvap') or {}).get('method') == 'pfd_component_override'
        pfd_hvap_correlation = (
            (sources.get('property_correlations') or {}).get('method') == 'pfd_property_correlations'
            and 'Hvap' in ((known or {}).get('property_correlations') or {})
        )
        if pfd_hvap or pfd_hvap_correlation:
            try:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..property_resolver import get_property_resolver
                else:
                    from property_resolver import get_property_resolver
                result = get_property_resolver().resolve_hvap(
                    props.symbol if props else comp,
                    known,
                    T=T,
                    allow_online=False,
                    allow_estimation=False,
                )
                if result.value is not None:
                    self._record_lazy_property_source(comp, 'Hvap', T, result)
                    return self._set_limited_cache(self._hvap_T_cache, cache_key, float(result.value))
            except Exception:
                pass

        value = self._fast_perry_hvap(comp, T)
        if value is not None:
            return self._set_limited_cache(self._hvap_T_cache, cache_key, value)

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            result = get_property_resolver().resolve_hvap(
                props.symbol if props else comp,
                self._resolver_known_props.get(comp),
                T=T,
                allow_online=False,
                allow_estimation=False,
            )
            if result.value is not None:
                return self._set_limited_cache(self._hvap_T_cache, cache_key, float(result.value))
        except Exception:
            pass

        hvap_ref = self._normal_hvap(comp)
        return self._set_limited_cache(self._hvap_T_cache, cache_key, hvap_ref)

    def _integrate_liquid_cp(self, comp: str, T1: float, T2: float) -> float:
        """Integral of liquid Cp from T1 to T2 [kJ/mol]."""
        if abs(T2 - T1) < 1e-12:
            return 0.0
        analytic = self._integrate_cp_analytic(comp, T1, T2, 'liquid')
        if analytic is not None:
            return analytic
        steps = 16
        h = (T2 - T1) / steps
        total = self.Cp_liquid(comp, T1) + self.Cp_liquid(comp, T2)
        for index in range(1, steps):
            weight = 4 if index % 2 else 2
            total += weight * self.Cp_liquid(comp, T1 + index * h)
        return total * h / 3.0 / 1000.0

    def _integrate_solid_cp(self, comp: str, T1: float, T2: float) -> float:
        """Integral of solid Cp from T1 to T2 [kJ/mol]."""
        if abs(T2 - T1) < 1e-12:
            return 0.0
        analytic = self._integrate_cp_analytic(comp, T1, T2, 'solid')
        if analytic is None:
            raise ThermodynamicsError(
                f"Cannot integrate solid heat capacity for '{comp}' over {T1:g}-{T2:g} K."
            )
        return analytic

    def mixture_Cp(self, composition: dict[str, float], T: float,
                   vapor_fraction: float = 1.0,
                   P: Optional[float] = None) -> float:
        """
        Mixture heat capacity [kJ/kmol-K]

        Args:
            composition: Mole fractions
            T: Temperature [K]
            vapor_fraction: Vapor fraction (0-1)
            P: Pressure [bar]; ignored by the ideal model (ideal-gas and
                liquid-correlation Cp are pressure-independent), used by
                real-gas subclasses for the vapor departure.
        """
        cp = 0.0
        for comp, z in composition.items():
            if vapor_fraction > 0.999:
                cp_comp = self.Cp_ideal_gas(comp, T)
            elif vapor_fraction < 0.001:
                cp_comp = self.Cp_liquid(comp, T)
            else:
                cp_comp = vapor_fraction * self.Cp_ideal_gas(comp, T) + \
                         (1 - vapor_fraction) * self.Cp_liquid(comp, T)
            cp += z * cp_comp
        return cp  # J/mol-K is numerically kJ/kmol-K

    def phase_weighted_mixture_Cp(self, composition: dict[str, float], T: float,
                                  vapor_fraction: float,
                                  x: Optional[dict] = None,
                                  y: Optional[dict] = None,
                                  P: Optional[float] = None) -> float:
        """
        Mixture heat capacity [kJ/kmol-K] using phase compositions when available.
        """
        if vapor_fraction > 0.999:
            return self.mixture_Cp(y or composition, T, 1.0, P)
        if vapor_fraction < 0.001:
            return self.mixture_Cp(x or composition, T, 0.0, P)
        cp_liq = self.mixture_Cp(x or composition, T, 0.0, P)
        cp_vap = self.mixture_Cp(y or composition, T, 1.0, P)
        return (1.0 - vapor_fraction) * cp_liq + vapor_fraction * cp_vap
    
    def get_Psat_coefficients(self, comp: str) -> tuple[float, ...]:
        """Return and cache one component's canonical Psat coefficient payload."""
        props = self.props.get(comp)
        if props is None:
            raise ThermodynamicsError(f"No properties for component '{comp}'")

        cached = self._psat_coefficients_cache.get(comp)
        if cached is not None:
            return cached

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver

            resolver = get_property_resolver()
            known_props = self._resolver_known_props.get(comp)
            lookup_name = _property_lookup_identifier(comp, props)
            coefficients = tuple(float(value) for value in (
                resolver.resolve_vapor_pressure_coefficients(
                    lookup_name,
                    known_props,
                    allow_online=self._allow_online_lookup_for_component(comp),
                )
            ))
            if len(coefficients) != 13:
                raise ValueError(
                    "Canonical Psat coefficient payload must contain 13 values"
                )
            self._psat_coefficients_cache[comp] = coefficients
            return coefficients
        except Exception as exc:
            raise ThermodynamicsError(
                f"Cannot resolve vapor pressure coefficients for '{comp}'."
            ) from exc

    def Psat(self, comp: str, T: float) -> float:
        """
        Vapor pressure [bar] from cached canonical coefficients.

        This hot-path evaluator intentionally omits canonical temperature-range
        checks. ``get_Psat_coefficients`` performs the resolver work once per
        component, after which calls evaluate the canonical equation directly.
        
        Args:
            comp: Component symbol
            T: Temperature [K]
            
        Returns:
            Saturation pressure [bar]
        """
        temperature = float(T)
        cache_key = (comp, temperature)
        cached = self._psat_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            (
                A, B, C, D, E, F, G, H, Tc, inverse_power,
                supercritical_slope, T_min, lower_continuation_slope,
            ) = self.get_Psat_coefficients(comp)
            evaluation_temperature = min(max(temperature, T_min), Tc)
            ln_pressure = (
                A
                + B / evaluation_temperature
                + C * math.log(evaluation_temperature)
                + D * evaluation_temperature
                + E * evaluation_temperature**2
                + F * evaluation_temperature**5
                + G * evaluation_temperature**3
            )
            if H != 0.0:
                ln_pressure += H * (
                    (evaluation_temperature / Tc) ** int(inverse_power) - 1.0
                )
            if temperature < T_min:
                ln_pressure -= T_min**2 * lower_continuation_slope * (
                    1.0 / temperature - 1.0 / T_min
                )
            try:
                pressure = math.exp(ln_pressure)
            except OverflowError:
                pressure = math.inf
            if temperature > Tc:
                pressure += supercritical_slope * (temperature - Tc)
            self._psat_cache[cache_key] = pressure
            return pressure
        except Exception as exc:
            raise ThermodynamicsError(
                f"Cannot calculate vapor pressure for '{comp}' at T={T:.1f}K."
            ) from exc
    
    def henry_component_data(self, comp: str) -> Optional[HenryComponentData]:
        """Return hydrated Henry data for one component, including override provenance."""
        if comp in self._henry_component_data_cache:
            return self._henry_component_data_cache[comp]
        props = self.props.get(comp)
        if props is None or getattr(props, 'henry_Hcp', None) is None:
            self._henry_component_data_cache[comp] = None
            return None

        hcp = float(props.henry_Hcp)
        B = None if getattr(props, 'henry_B', None) is None else float(props.henry_B)
        if not math.isfinite(hcp) or hcp <= 0.0:
            raise ThermodynamicsError(
                f"Henry Hcp for {self._component_label(comp)} must be positive and finite"
            )
        if B is not None and not math.isfinite(B):
            raise ThermodynamicsError(
                f"Henry B for {self._component_label(comp)} must be finite"
            )

        record = get_henry_constant_database().get(getattr(props, 'CAS', ''))
        h_source_info = props.property_sources.get('henry_Hcp') or {}
        b_source_info = props.property_sources.get('henry_B') or {}
        vinf_source_info = props.property_sources.get('henry_Vinf') or {}
        h_method = h_source_info.get('method')
        b_method = b_source_info.get('method')
        h_is_provided = h_method == 'pfd_component_override'
        b_is_provided = b_method == 'pfd_component_override'
        h_is_database = h_method == 'henry_database_hcp_298'
        b_is_database = b_method == 'henry_database_temperature_coefficient'
        temperature_correlation = (
            record.temperature_correlation
            if record is not None and h_is_database and b_is_database
            else None
        )

        def source_quality(source: dict, grade: Optional[str]) -> Optional[float]:
            raw = source.get('quality')
            if raw is None:
                return self._henry_grade_quality(grade)
            try:
                return max(0.0, min(1.0, float(raw)))
            except (TypeError, ValueError):
                return self._henry_grade_quality(grade)

        quality_h_grade = (
            'provided' if h_is_provided
            else (record.quality_h if h_is_database and record else h_source_info.get('grade'))
        )
        quality_b_grade = (
            'provided' if b_is_provided
            else (
                record.quality_b
                if b_is_database and record and B is not None
                else b_source_info.get('grade')
            )
        )

        explicit_tmin = getattr(props, 'henry_Tmin', None)
        explicit_tmax = getattr(props, 'henry_Tmax', None)
        temperature_min = (
            float(explicit_tmin)
            if explicit_tmin is not None
            else (
                temperature_correlation.Tmin_K
                if temperature_correlation is not None
                else (HENRY_DEFAULT_TMIN_K if B is not None else T_REF)
            )
        )
        temperature_max = (
            float(explicit_tmax)
            if explicit_tmax is not None
            else (
                temperature_correlation.Tmax_K
                if temperature_correlation is not None
                else (HENRY_DEFAULT_TMAX_K if B is not None else T_REF)
            )
        )
        if (
            not math.isfinite(temperature_min)
            or not math.isfinite(temperature_max)
            or temperature_min <= 0.0
            or temperature_max <= 0.0
            or temperature_min > temperature_max
        ):
            raise ThermodynamicsError(
                f"Henry temperature range for {self._component_label(comp)} must be "
                "positive, finite, and ordered"
            )

        vinf = getattr(props, 'henry_Vinf', None)
        vinf_uncertainty = getattr(props, 'henry_Vinf_uncertainty', None)
        vinf_method = str(vinf_source_info.get('method') or '') or None
        vinf_source = str(vinf_source_info.get('source') or '') or None
        vinf_quality = source_quality(vinf_source_info, None)
        vinf_unavailable_reason = None
        vinf_estimated_relative_mae = None
        if vinf is None:
            critical_volume = getattr(props, 'Vc', None)
            vc_source_info = props.property_sources.get('Vc') or {}
            vc_quality = source_quality(vc_source_info, vc_source_info.get('grade'))
            if critical_volume is not None:
                critical_volume = float(critical_volume)
                if (
                    math.isfinite(critical_volume)
                    and critical_volume > 0.0
                    and vc_quality is not None
                    and vc_quality >= 0.8
                ):
                    vinf = 10.74 + 0.2683 * critical_volume
                    vinf_estimated_relative_mae = 0.09
                    vinf_method = 'henry_vinf_from_critical_volume'
                    vinf_source = 'Zhou and Battino (2001) correlation'
                    vinf_quality = 0.90
                    props.henry_Vinf = vinf
                    props.henry_Vinf_uncertainty = vinf_uncertainty
                    props.property_sources['henry_Vinf'] = {
                        'source': vinf_source,
                        'method': vinf_method,
                        'quality': vinf_quality,
                        'doi': '10.1021/je000215o',
                        'notes': (
                            'Estimated as 10.74 + 0.2683*Vc [cm3/mol]; '
                            'reported mean absolute error is approximately 8-10%'
                        ),
                    }
                elif math.isfinite(critical_volume) and critical_volume > 0.0:
                    vinf_unavailable_reason = (
                        'Vc_quality_below_0.8'
                        if vc_quality is not None else 'Vc_quality_unknown'
                    )
            if vinf is None and vinf_unavailable_reason is None:
                vinf_unavailable_reason = 'Vc_unavailable'
        if vinf is not None:
            vinf = float(vinf)
            if not math.isfinite(vinf) or vinf <= 0.0:
                raise ThermodynamicsError(
                    f"Henry Vinf for {self._component_label(comp)} must be positive and finite"
                )
        if vinf_uncertainty is not None:
            vinf_uncertainty = float(vinf_uncertainty)
            if not math.isfinite(vinf_uncertainty) or vinf_uncertainty < 0.0:
                raise ThermodynamicsError(
                    f"Henry Vinf uncertainty for {self._component_label(comp)} must be "
                    "nonnegative and finite"
                )

        data = HenryComponentData(
            component=comp,
            cas=str(getattr(props, 'CAS', '') or ''),
            hcp_298=hcp,
            B=B,
            quality_h=quality_h_grade,
            quality_b=quality_b_grade,
            quality_score_h=source_quality(h_source_info, quality_h_grade),
            quality_score_b=(
                source_quality(b_source_info, quality_b_grade) if B is not None else None
            ),
            temperature_min_K=temperature_min,
            temperature_max_K=temperature_max,
            temperature_range_source=(
                'provided' if explicit_tmin is not None or explicit_tmax is not None
                else (
                    temperature_correlation.range_source
                    if temperature_correlation is not None
                    else ('default_5_to_50C' if B is not None else 'reference_temperature_only')
                )
            ),
            vinf_cm3_per_mol=vinf,
            vinf_uncertainty_cm3_per_mol=vinf_uncertainty,
            vinf_estimated_relative_mae=vinf_estimated_relative_mae,
            vinf_source=vinf_source,
            vinf_method=vinf_method,
            vinf_quality=vinf_quality,
            vinf_unavailable_reason=vinf_unavailable_reason,
            more_temperature_dependence_available=bool(
                record and record.b_more_temperature_dependence and b_is_database
            ),
            temperature_correlation=temperature_correlation,
            h_source=(
                'pfd' if h_is_provided
                else ('database' if h_is_database else 'component_property')
            ),
            b_source=(
                'pfd' if b_is_provided
                else (
                    'database' if b_is_database
                    else ('component_property' if B is not None else None)
                )
            ),
        )
        self._henry_component_data_cache[comp] = data
        return data

    def create_aqueous_equilibrium_context(
        self,
        henry_components,
        water_component: str = 'water',
        pressure_warning_bar: float = 20.0,
    ) -> AqueousEquilibriumContext:
        """Build a frozen pure-water Henry context for a later equilibrium solve."""
        if self.fluid_phase_model != 'VLE':
            raise ThermodynamicsError(
                f"Aqueous Henry standard states currently support only "
                f"FLUID_PHASE_MODEL VLE, not {self.fluid_phase_model}; "
                "Henry-aware liquid-liquid stability is not implemented"
            )
        if water_component not in self.props:
            raise ThermodynamicsError(
                f"Aqueous Henry context water component '{water_component}' is not in the thermodynamic system"
            )
        if isinstance(henry_components, str):
            selected = [
                item.strip() for item in henry_components.replace(';', ',').split(',')
                if item.strip()
            ]
        else:
            selected = list(henry_components or [])

        component_data = {}
        low_quality_grades = {'D+', 'D', 'D-', 'F'}
        for comp in dict.fromkeys(selected):
            if comp == water_component:
                raise ThermodynamicsError(
                    f"Water component '{water_component}' cannot use its own aqueous Henry standard state"
                )
            if comp not in self.props:
                raise ThermodynamicsError(
                    f"Henry component '{comp}' is not in the thermodynamic system"
                )
            data = self.henry_component_data(comp)
            if data is None:
                raise ThermodynamicsError(
                    f"No Henry Hcp is available for {self._component_label(comp)}"
                )
            component_data[comp] = data
            if data.B is None:
                self.add_warning(
                    f"Henry temperature coefficient B is unavailable for "
                    f"{self._component_label(comp)}; holding Hcp at its 298.15 K value "
                    "and using the corresponding zero-B dissolution-enthalpy approximation."
                )
            if str(data.quality_h or '').upper() in low_quality_grades:
                self.add_warning(
                    f"Henry Hcp for {self._component_label(comp)} has low database "
                    f"quality grade {data.quality_h}; aqueous equilibrium results may be unreliable."
                )
            if data.B is not None and str(data.quality_b or '').upper() in low_quality_grades:
                self.add_warning(
                    f"Henry B for {self._component_label(comp)} has low database "
                    f"quality grade {data.quality_b}; temperature and heat effects may be unreliable."
                )

        reference_volume = None
        prebound = self._prebound_liquid_molar_volume(water_component, T_REF)
        if prebound is not None:
            reference_volume = prebound[0]
        if reference_volume is None:
            reference_volume, _ = self._liquid_molar_volume_for_poynting(
                water_component, T_REF
            )
        if reference_volume is None or reference_volume <= 0.0:
            props = self.props[water_component]
            reference_volume = float(props.MW or STEAM_WATER_MW) / 997.0474
            self.add_warning(
                f"Could not resolve liquid molar volume for {self._component_label(water_component)}; "
                "using the 25 C pure-water density fallback for Henry conversion."
            )

        pressure_warning_bar = float(pressure_warning_bar)
        if not math.isfinite(pressure_warning_bar) or pressure_warning_bar <= 0.0:
            raise ThermodynamicsError("Henry pressure warning threshold must be positive")
        return AqueousEquilibriumContext(
            water_component=water_component,
            component_data=MappingProxyType(dict(component_data)),
            solvent_molar_volume_ref_m3_per_kmol=float(reference_volume),
            pressure_warning_bar=pressure_warning_bar,
        )

    @staticmethod
    def _normalized_aqueous_composition(composition: dict[str, float], components) -> dict[str, float]:
        values = {
            comp: max(float(composition.get(comp, 0.0)), 0.0)
            for comp in components
        }
        total = sum(values.values())
        if total <= 0.0:
            raise ThermodynamicsError("Aqueous equilibrium composition must have positive total")
        return {comp: value / total for comp, value in values.items()}

    def aqueous_solvent_molar_concentration(
        self,
        T: float,
        context: AqueousEquilibriumContext,
    ) -> float:
        """Return water-rich solvent concentration [mol/m3] without hot-loop resolution."""
        T = float(T)
        if not math.isfinite(T) or T <= 0.0:
            raise ThermodynamicsError("Henry temperature must be positive")
        key = (context.water_component, T, context.solvent_molar_volume_ref_m3_per_kmol)
        cached = self._aqueous_solvent_concentration_cache.get(key)
        if cached is not None:
            return cached

        volume = None
        prebound = self._prebound_liquid_molar_volume(context.water_component, T)
        if prebound is not None:
            volume = prebound[0]
        if volume is None or volume <= 0.0:
            volume = context.solvent_molar_volume_ref_m3_per_kmol
        concentration = 1000.0 / float(volume)
        return self._set_limited_cache(
            self._aqueous_solvent_concentration_cache,
            key,
            concentration,
        )

    def _aqueous_solvent_dln_concentration_dinvT(
        self,
        T: float,
        context: AqueousEquilibriumContext,
    ) -> float:
        key = (context.water_component, float(T), context.solvent_molar_volume_ref_m3_per_kmol)
        cached = self._aqueous_solvent_density_derivative_cache.get(key)
        if cached is not None:
            return cached
        dT = max(0.05, 1e-3 * float(T))
        T_low = max(1.0, float(T) - dT)
        T_high = float(T) + dT
        c_low = self.aqueous_solvent_molar_concentration(T_low, context)
        c_high = self.aqueous_solvent_molar_concentration(T_high, context)
        denominator = 1.0 / T_high - 1.0 / T_low
        derivative = (
            (math.log(c_high) - math.log(c_low)) / denominator
            if abs(denominator) > 1e-30 else 0.0
        )
        return self._set_limited_cache(
            self._aqueous_solvent_density_derivative_cache,
            key,
            derivative,
        )

    @staticmethod
    def _iapws_water_psat_bar(T: float) -> float:
        reduced_temperature = float(T) / 647.096
        tau = max(0.0, 1.0 - reduced_temperature)
        coefficients = (
            (-7.85951783, 1.0),
            (1.84408259, 1.5),
            (-11.7866497, 3.0),
            (22.6807411, 3.5),
            (-15.9618719, 4.0),
            (1.80122502, 7.5),
        )
        exponent = sum(a * tau ** b for a, b in coefficients) / reduced_temperature
        return 220.64 * math.exp(exponent)

    def _raw_iapws_hcp(
        self,
        T: float,
        context: AqueousEquilibriumContext,
        data: HenryComponentData,
    ) -> float:
        correlation = data.temperature_correlation
        if correlation is None or correlation.model != 'iapws_g7_04':
            raise ThermodynamicsError("IAPWS Henry correlation is unavailable")
        reduced_temperature = float(T) / 647.096
        tau = max(0.0, 1.0 - reduced_temperature)
        ln_ratio = (
            correlation.A / reduced_temperature
            + correlation.B * tau ** 0.355 / reduced_temperature
            + correlation.C * reduced_temperature ** -0.41 * math.exp(tau)
        )
        water_psat_bar = self._iapws_water_psat_bar(T)
        volatility_bar = water_psat_bar * math.exp(max(min(ln_ratio, 100.0), -100.0))
        concentration = self.aqueous_solvent_molar_concentration(T, context)
        return concentration / (volatility_bar * 1.0e5)

    def _raw_brockbank_hcp(
        self,
        T: float,
        context: AqueousEquilibriumContext,
        data: HenryComponentData,
    ) -> float:
        correlation = data.temperature_correlation
        if correlation is None or correlation.model not in {
            'brockbank_dippr101', 'chapoy_dippr101'
        }:
            raise ThermodynamicsError("DIPPR-101 Henry correlation is unavailable")
        exponent = (
            correlation.A
            + correlation.B / T
            + correlation.C * math.log(T)
            + correlation.D * T ** correlation.E
        )
        volatility_kPa = math.exp(max(min(exponent, 100.0), -100.0))
        concentration = self.aqueous_solvent_molar_concentration(T, context)
        return concentration / (volatility_kPa * 1000.0)

    def _raw_volatility_hcp(
        self,
        T: float,
        context: AqueousEquilibriumContext,
        data: HenryComponentData,
    ) -> float:
        if data.temperature_correlation.model == 'iapws_g7_04':
            return self._raw_iapws_hcp(T, context, data)
        return self._raw_brockbank_hcp(T, context, data)

    def _volatility_henry_anchor(
        self,
        comp: str,
        context: AqueousEquilibriumContext,
        data: HenryComponentData,
    ) -> tuple[float, float]:
        correlation = data.temperature_correlation
        key = (
            comp,
            correlation,
            context.solvent_molar_volume_ref_m3_per_kmol,
            context.reference_temperature_K,
        )
        cached = self._henry_temperature_anchor_cache.get(key)
        if cached is not None:
            return cached
        reference_temperature = context.reference_temperature_K
        def at_reference_pressure(temperature: float) -> float:
            raw = self._raw_volatility_hcp(temperature, context, data)
            if data.vinf_cm3_per_mol is None:
                return raw
            pressure_reference = (
                self._iapws_water_psat_bar(temperature)
                if correlation.model == 'iapws_g7_04' else P_REF
            )
            exponent = (
                -data.vinf_cm3_per_mol * 1.0e-6
                * (P_REF - pressure_reference)
                / (R_BAR * temperature)
            )
            return raw * math.exp(max(min(exponent, 100.0), -100.0))

        raw_reference = at_reference_pressure(reference_temperature)
        dT = 0.05
        low = reference_temperature - dT
        high = reference_temperature + dT
        raw_low = at_reference_pressure(low)
        raw_high = at_reference_pressure(high)
        denominator = 1.0 / high - 1.0 / low
        raw_slope = (math.log(raw_high) - math.log(raw_low)) / denominator
        anchor = (
            math.log(data.hcp_298 / raw_reference),
            (data.B or raw_slope) - raw_slope,
        )
        return self._set_limited_cache(self._henry_temperature_anchor_cache, key, anchor)

    def henry_constant_hcp(
        self,
        comp: str,
        T: float,
        context: Optional[AqueousEquilibriumContext] = None,
        P: Optional[float] = None,
    ) -> float:
        data = (
            context.component_data.get(comp)
            if context is not None else self.henry_component_data(comp)
        )
        if data is None:
            raise ThermodynamicsError(f"No Henry Hcp is available for {self._component_label(comp)}")
        T = float(T)
        if not math.isfinite(T) or T <= 0.0:
            raise ThermodynamicsError("Henry temperature must be positive")
        correlation = data.temperature_correlation if context is not None else None
        if correlation is not None and correlation.model in {
            'iapws_g7_04', 'brockbank_dippr101', 'chapoy_dippr101'
        }:
            raw_hcp = self._raw_volatility_hcp(T, context, data)
            if correlation.normalize_to_reference:
                log_scale, slope_adjustment = self._volatility_henry_anchor(
                    comp, context, data
                )
                exponent = log_scale + slope_adjustment * (
                    1.0 / T - 1.0 / context.reference_temperature_K
                )
                hcp = raw_hcp * math.exp(max(min(exponent, 100.0), -100.0))
            else:
                hcp = raw_hcp
            pressure_reference_bar = (
                self._iapws_water_psat_bar(T)
                if correlation.model == 'iapws_g7_04' else P_REF
            )
        elif correlation is not None and correlation.model == 'exp_a_b_over_t_c_log_t':
            exponent = correlation.A + correlation.B / T + correlation.C * math.log(T)
            hcp = math.exp(max(min(exponent, 100.0), -100.0))
            pressure_reference_bar = P_REF
        else:
            exponent = 0.0
            if data.B is not None:
                T_ref = context.reference_temperature_K if context is not None else T_REF
                exponent = data.B * (1.0 / T - 1.0 / T_ref)
            hcp = data.hcp_298 * math.exp(max(min(exponent, 100.0), -100.0))
            pressure_reference_bar = P_REF
        evaluation_pressure = P
        if evaluation_pressure is None and correlation is not None and correlation.model in {
            'iapws_g7_04', 'brockbank_dippr101', 'chapoy_dippr101'
        }:
            evaluation_pressure = P_REF
        if evaluation_pressure is not None:
            evaluation_pressure = float(evaluation_pressure)
            if not math.isfinite(evaluation_pressure) or evaluation_pressure <= 0.0:
                raise ThermodynamicsError("Pressure must be positive for Henry equilibrium")
            if data.vinf_cm3_per_mol is not None:
                volume_m3_per_mol = data.vinf_cm3_per_mol * 1.0e-6
                pressure_exponent = (
                    -volume_m3_per_mol
                    * (evaluation_pressure - pressure_reference_bar)
                    / (R_BAR * T)
                )
                hcp *= math.exp(max(min(pressure_exponent, 100.0), -100.0))
        return hcp

    def henry_dln_hcp_dinvT(
        self,
        comp: str,
        T: float,
        P: float,
        context: AqueousEquilibriumContext,
    ) -> float:
        """Temperature derivative of the active pressure-corrected Hcp model [K]."""
        data = context.component_data[comp]
        correlation = data.temperature_correlation
        if correlation is not None and correlation.model in {
            'iapws_g7_04', 'brockbank_dippr101', 'chapoy_dippr101'
        }:
            dT = max(0.05, 1e-4 * float(T))
            low = max(1.0, float(T) - dT)
            high = float(T) + dT
            h_low = self.henry_constant_hcp(comp, low, context, P=P)
            h_high = self.henry_constant_hcp(comp, high, context, P=P)
            return (math.log(h_high) - math.log(h_low)) / (1.0 / high - 1.0 / low)
        if correlation is not None and correlation.model == 'exp_a_b_over_t_c_log_t':
            derivative = correlation.B - correlation.C * float(T)
        else:
            derivative = data.B or 0.0
        if data.vinf_cm3_per_mol is not None:
            derivative -= data.vinf_cm3_per_mol * 1.0e-6 * (P - P_REF) / R_BAR
        return derivative

    @staticmethod
    def _henry_score_grade(score: Optional[float]) -> Optional[str]:
        if score is None:
            return None
        for threshold, grade in (
            (0.95, 'A+'), (0.90, 'A'), (0.84, 'A-'),
            (0.78, 'B+'), (0.72, 'B'), (0.66, 'B-'),
            (0.58, 'C+'), (0.50, 'C'), (0.42, 'C-'),
            (0.34, 'D+'), (0.27, 'D'), (0.20, 'D-'),
        ):
            if score >= threshold:
                return grade
        return 'F'

    def henry_effective_quality(
        self,
        comp: str,
        T: float,
        P: float,
        context: AqueousEquilibriumContext,
    ) -> dict[str, object]:
        """Return T/P-adjusted Henry correlation quality and audit details."""
        data = context.component_data.get(comp)
        if data is None:
            raise ThermodynamicsError(f"No Henry data in context for {self._component_label(comp)}")
        T = float(T)
        P = float(P)
        if not math.isfinite(T) or T <= 0.0:
            raise ThermodynamicsError("Henry quality temperature must be positive and finite")
        if not math.isfinite(P) or P <= 0.0:
            raise ThermodynamicsError("Henry quality pressure must be positive and finite")
        h_quality = (
            data.quality_score_h
            if data.quality_score_h is not None
            else self._henry_grade_quality(data.quality_h)
        )
        b_quality = (
            data.quality_score_b
            if data.quality_score_b is not None
            else self._henry_grade_quality(data.quality_b)
        )
        base_quality = h_quality
        if data.B is not None and b_quality is not None:
            base_quality = b_quality if base_quality is None else min(base_quality, b_quality)
        if (
            data.temperature_correlation is not None
            and not data.temperature_correlation.normalize_to_reference
            and data.temperature_correlation.fit_quality is not None
        ):
            base_quality = data.temperature_correlation.fit_quality

        temperature_distance = max(
            data.temperature_min_K - T,
            T - data.temperature_max_K,
            0.0,
        )
        temperature_rate = 0.005 if data.B is not None else 0.01
        temperature_penalty = temperature_rate * temperature_distance

        if data.vinf_method == 'henry_vinf_from_critical_volume':
            pressure_method = 'Vc_correlation'
            pressure_rate = 0.001
        elif data.vinf_cm3_per_mol is not None:
            pressure_method = 'measured_or_provided_Vinf'
            pressure_rate = 0.0002
        else:
            pressure_method = 'uncorrected'
            pressure_rate = 0.003
        pressure_penalty = pressure_rate * max(P - 10.0, 0.0)
        effective_quality = (
            None
            if base_quality is None
            else max(0.0, min(1.0, base_quality - temperature_penalty - pressure_penalty))
        )
        return {
            'base_quality': base_quality,
            'base_grade': self._henry_score_grade(base_quality),
            'effective_quality': effective_quality,
            'effective_grade': self._henry_score_grade(effective_quality),
            'temperature_penalty': temperature_penalty,
            'temperature_penalty_per_K': temperature_rate,
            'temperature_distance_outside_range_K': temperature_distance,
            'temperature_min_K': data.temperature_min_K,
            'temperature_max_K': data.temperature_max_K,
            'temperature_range_source': data.temperature_range_source,
            'temperature_model': (
                data.temperature_correlation.model
                if data.temperature_correlation is not None
                else ('van_t_hoff_B' if data.B is not None else 'constant_Hcp')
            ),
            'temperature_fit_rms_lnH': (
                data.temperature_correlation.rms_lnH
                if data.temperature_correlation is not None else None
            ),
            'temperature_fit_quality': (
                data.temperature_correlation.fit_quality
                if data.temperature_correlation is not None else None
            ),
            'temperature_normalized_to_reference': (
                data.temperature_correlation.normalize_to_reference
                if data.temperature_correlation is not None else False
            ),
            'raw_reference_value_difference_percent': (
                data.temperature_correlation.raw_reference_value_difference_percent
                if data.temperature_correlation is not None else None
            ),
            'raw_reference_slope_difference_K': (
                data.temperature_correlation.raw_reference_slope_difference_K
                if data.temperature_correlation is not None else None
            ),
            'pressure_penalty': pressure_penalty,
            'pressure_penalty_per_bar_above_10': pressure_rate,
            'pressure_method': pressure_method,
            'pressure_correction_applied': data.vinf_cm3_per_mol is not None,
        }

    def record_henry_effective_quality(
        self,
        comp: str,
        T: float,
        P: float,
        context: AqueousEquilibriumContext,
    ) -> dict[str, object]:
        """Record one solved-state Henry quality observation outside hot residual loops."""
        quality = self.henry_effective_quality(comp, T, P, context)
        data = context.component_data[comp]
        observation = SimpleNamespace(
            source=(data.h_source or 'Henry correlation'),
            method='henry_effective_TP_quality',
            quality=quality['effective_quality'],
            notes=(
                f"Full-quality T range {data.temperature_min_K:.2f}-"
                f"{data.temperature_max_K:.2f} K ({data.temperature_range_source}); "
                f"pressure method {quality['pressure_method']}; evaluated pressure "
                f"{float(P):.6g} bar"
            ),
        )
        self._record_lazy_property_source(comp, 'henry_Hcp_effective', T, observation)
        return quality

    def _henry_ideal_vapor_K_value(
        self,
        comp: str,
        T: float,
        P: float,
        context: AqueousEquilibriumContext,
    ) -> float:
        if P <= 0.0:
            raise ThermodynamicsError("Pressure must be positive for Henry equilibrium")
        current_context = self._current_quality_context()
        context_key = (comp, self._quality_context_key(current_context))
        if context_key not in self._henry_quality_marked_contexts:
            self._henry_quality_marked_contexts.add(context_key)
            prop_names = ['henry_Hcp']
            data = context.component_data.get(comp)
            if data is not None and data.B is not None:
                prop_names.append('henry_B')
            if data is not None and data.vinf_cm3_per_mol is not None:
                prop_names.append('henry_Vinf')
            self.mark_property_source_context(
                comp,
                prop_names,
                kind=current_context.get('kind', 'thermo_model'),
                unit_id=current_context.get('unit_id'),
                unit_type=current_context.get('unit_type'),
                stream_id=current_context.get('stream_id'),
                phase='aqueous_henry_equilibrium',
                affects_result=current_context.get('affects_result', True),
            )
        hcp = self.henry_constant_hcp(comp, T, context, P=P)
        concentration = self.aqueous_solvent_molar_concentration(T, context)
        return concentration / (hcp * P * 1.0e5)

    def _warn_aqueous_henry_pressure(
        self,
        P: float,
        context: AqueousEquilibriumContext,
    ) -> None:
        if P > context.pressure_warning_bar:
            uncorrected = [
                comp for comp, data in context.component_data.items()
                if data.vinf_cm3_per_mol is None
            ]
            if uncorrected:
                self.add_warning(
                    f"Aqueous Henry equilibrium is being evaluated at {P:.4g} bar, above "
                    f"the {context.pressure_warning_bar:.4g} bar reference-pressure warning "
                    "threshold; no Krichevsky-Kasarnovsky pressure correction is available "
                    f"for {', '.join(uncorrected)}."
                )

    def aqueous_K_values(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        context: AqueousEquilibriumContext,
    ) -> dict[str, float]:
        """Ideal-vapor aqueous K-values using a frozen Henry component set."""
        self._warn_aqueous_henry_pressure(P, context)
        x = self._normalized_aqueous_composition(composition, self.components)
        cache_key = self._k_values_cache_key('aqueous_ideal', T, P, x) + (context.cache_key(),)
        cached = self._get_cached_k_values(cache_key)
        if cached is not None:
            return cached
        K = {}
        for comp in self.components:
            if comp in context.component_data:
                value = self._henry_ideal_vapor_K_value(comp, T, P, context)
            else:
                value = self.Psat(comp, T) / max(P, 1e-12)
            K[comp] = max(1e-12, min(1e12, float(value)))
        return self._set_cached_k_values(cache_key, K)

    def aqueous_flash_TP(
        self,
        composition: dict[str, float],
        T: float,
        P: float,
        context: AqueousEquilibriumContext,
        max_iter: int = 100,
        tol: float = 1e-9,
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Two-phase TP flash with a frozen aqueous Henry standard-state set."""
        z = self._normalized_aqueous_composition(composition, self.components)
        K = self.aqueous_K_values(T, P, z, context)
        V = 0.5
        for _ in range(max_iter):
            V = self._rachford_rice_bounded(z, K, V)
            x, _y = self._flash_phase_compositions(z, K, V)
            K_new = self.aqueous_K_values(T, P, x, context)
            max_change = max(
                abs(K_new.get(comp, 1.0) - K.get(comp, 1.0))
                / max(abs(K.get(comp, 1.0)), 1e-300)
                for comp in self.components
            )
            K = K_new
            if max_change < tol:
                break

        V = self._rachford_rice_bounded(z, K, V)
        x, y = self._flash_phase_compositions(z, K, V)
        return V, x, y

    def aqueous_liquid_enthalpy(
        self,
        composition: dict[str, float],
        T: float,
        P: float,
        context: AqueousEquilibriumContext,
    ) -> float:
        """Water-rich liquid enthalpy [kJ/kmol] with Henry partial molar solutes."""
        x = self._normalized_aqueous_composition(composition, self.components)
        henry_fraction = sum(x.get(comp, 0.0) for comp in context.component_data)
        bulk_fraction = 1.0 - henry_fraction
        if bulk_fraction <= 1e-12:
            raise ThermodynamicsError(
                "Aqueous Henry enthalpy requires a non-Henry bulk solvent fraction"
            )
        bulk = {
            comp: x.get(comp, 0.0) / bulk_fraction
            for comp in self.components
            if comp not in context.component_data
        }
        enthalpy = bulk_fraction * self.mixture_enthalpy(
            bulk, T, vapor_fraction=0.0, P=P
        )
        density_derivative = self._aqueous_solvent_dln_concentration_dinvT(T, context)
        for comp, data in context.component_data.items():
            xi = x.get(comp, 0.0)
            if xi <= 0.0:
                continue
            B_x = (
                self.henry_dln_hcp_dinvT(comp, T, P, context)
                - density_derivative
            )
            h_infinite_dilution = 1000.0 * self.enthalpy_ideal_gas(comp, T) - R * B_x
            enthalpy += xi * h_infinite_dilution
        return enthalpy

    def aqueous_mixture_enthalpy(
        self,
        composition: dict[str, float],
        T: float,
        vapor_fraction: float,
        context: AqueousEquilibriumContext,
        x: Optional[dict[str, float]] = None,
        y: Optional[dict[str, float]] = None,
        P: float = P_REF,
    ) -> float:
        """Two-phase enthalpy [kJ/kmol] using Henry treatment only in the liquid."""
        V = max(0.0, min(1.0, float(vapor_fraction)))
        if V >= 0.999:
            return self.mixture_enthalpy(composition, T, 1.0, y=y, P=P)
        if V <= 0.001:
            return self.aqueous_liquid_enthalpy(x or composition, T, P, context)
        h_liquid = self.aqueous_liquid_enthalpy(x or composition, T, P, context)
        h_vapor = self.mixture_enthalpy(y or composition, T, 1.0, P=P)
        return (1.0 - V) * h_liquid + V * h_vapor

    def K_value(self, comp: str, T: float, P: float) -> float:
        """
        Equilibrium K-value for Raoult's law: K_i = P_sat_i / P
        
        Args:
            comp: Component symbol
            T: Temperature [K]
            P: Pressure [bar]
            
        Returns:
            K = y/x equilibrium ratio
        """
        Psat = self.Psat(comp, T)
        return Psat / P

    def K_values(self, T: float, P: float,
                 composition: dict[str, float]) -> dict[str, float]:
        cache_key = self._k_values_cache_key('scalar', T, P, composition)
        cached = self._get_cached_k_values(cache_key)
        if cached is not None:
            return cached
        K = {comp: self.K_value(comp, T, P) for comp in self.components}
        return self._set_cached_k_values(cache_key, K)
    
    def bubble_point_T(self, composition: dict[str, float], P: float, 
                       T_guess: float = 300.0) -> float:
        """
        Calculate bubble point temperature at given pressure.
        
        At bubble point: sum(x_i * K_i) = 1
        
        Args:
            composition: Liquid mole fractions
            P: Pressure [bar]
            T_guess: Initial temperature guess [K]
            
        Returns:
            Bubble point temperature [K]
        """
        T = T_guess
        for _ in range(50):
            K = self.K_values(T, P, composition)
            sum_xK = sum(
                x * K.get(comp, 1.0)
                for comp, x in composition.items()
            )
            
            if abs(sum_xK - 1.0) < 1e-6:
                return T
            
            # When sum_xK > 1, temperature is too high (too much vapor)
            # When sum_xK < 1, temperature is too low
            # Use Newton-like update with proper direction
            if sum_xK > 1:
                dT = -5.0 * (sum_xK - 1.0)  # Decrease T
            else:
                dT = 5.0 * (1.0 - sum_xK)   # Increase T
            
            T += dT
            T = max(100, min(800, T))  # Keep in reasonable range
        
        return T
    
    def dew_point_T(self, composition: dict[str, float], P: float,
                    T_guess: float = 350.0) -> float:
        """
        Calculate dew point temperature at given pressure.
        
        At dew point: sum(y_i / K_i) = 1
        
        Args:
            composition: Vapor mole fractions
            P: Pressure [bar]
            T_guess: Initial temperature guess [K]
            
        Returns:
            Dew point temperature [K]
        """
        T = T_guess
        for _ in range(50):
            K = self.K_values(T, P, composition)
            sum_yK = sum(
                y / max(K.get(comp, 1.0), 1e-30)
                for comp, y in composition.items()
            )
            
            if abs(sum_yK - 1.0) < 1e-6:
                return T
            
            # When sum_yK > 1, temperature is too low (too much liquid)
            # When sum_yK < 1, temperature is too high
            if sum_yK > 1:
                dT = 5.0 * (sum_yK - 1.0)   # Increase T
            else:
                dT = -5.0 * (1.0 - sum_yK)  # Decrease T
            
            T += dT
            T = max(100, min(800, T))
        
        return T
    
    def flash_TP(self, composition: dict[str, float], T: float, P: float) -> tuple[float, dict, dict]:
        """
        TP Flash calculation using Rachford-Rice equation.
        
        Args:
            composition: Feed mole fractions
            T: Temperature [K]
            P: Pressure [bar]
            
        Returns:
            (vapor_fraction, x_liquid, y_vapor)
        """
        # Calculate K-values
        K = self.K_values(T, P, composition)
        
        # Check if all liquid or all vapor
        sum_zK = sum(z * K[comp] for comp, z in composition.items())
        sum_zKinv = sum(z / K[comp] for comp, z in composition.items())
        
        if sum_zK <= 1.0:
            # All liquid (below bubble point)
            return 0.0, dict(composition), dict(composition)
        
        if sum_zKinv <= 1.0:
            # All vapor (above dew point)
            return 1.0, dict(composition), dict(composition)
        
        # Two-phase: solve Rachford-Rice
        # f(V) = sum(z_i * (K_i - 1) / (1 + V*(K_i - 1))) = 0
        
        def rachford_rice(V):
            return sum(z * (K[comp] - 1) / (1 + V * (K[comp] - 1))
                      for comp, z in composition.items())
        
        def rachford_rice_deriv(V):
            return -sum(z * (K[comp] - 1)**2 / (1 + V * (K[comp] - 1))**2
                       for comp, z in composition.items())
        
        # Newton-Raphson to find V
        V = 0.5
        for _ in range(50):
            f = rachford_rice(V)
            if abs(f) < 1e-10:
                break
            df = rachford_rice_deriv(V)
            if abs(df) < 1e-15:
                break
            V = V - f / df
            V = max(0.0, min(1.0, V))
        
        # Calculate phase compositions
        x = {}
        y = {}
        for comp, z in composition.items():
            x[comp] = z / (1 + V * (K[comp] - 1))
            y[comp] = K[comp] * x[comp]
        
        # Normalize
        sum_x = sum(x.values())
        sum_y = sum(y.values())
        x = {comp: xi / sum_x for comp, xi in x.items()}
        y = {comp: yi / sum_y for comp, yi in y.items()}

        return V, x, y

    def _fallback_flash_TP(self, composition: dict[str, float], T: float,
                           P: float, exc: Exception) -> tuple[float, dict, dict]:
        """Single K-value evaluation phase fallback for a failed TP flash."""
        try:
            K = self.K_values(T, P, composition)
            sum_zK = sum(z * K.get(comp, 1.0) for comp, z in composition.items())
            sum_zK_inv = sum(
                z / max(K.get(comp, 1.0), 1e-300)
                for comp, z in composition.items()
            )
            if sum_zK <= 1.0:
                V, x, y = 0.0, dict(composition), dict(composition)
            elif sum_zK_inv <= 1.0:
                V, x, y = 1.0, dict(composition), dict(composition)
            else:
                V = self._rachford_rice_bounded(composition, K, 0.5)
                x, y = self._flash_phase_compositions(composition, K, V)
            # Bucketed T/P and the exception class keep the deduplicated
            # warning list bounded when a solver sweeps many nearby states.
            self.add_warning(
                f"TP flash failed near T={5.0 * round(T / 5.0):g} K, "
                f"P={P:.2g} bar ({type(exc).__name__}); using a single "
                "K-value evaluation as the phase fallback."
            )
            return V, x, y
        except Exception:
            V = 1.0 if T > 400.0 else 0.0
            self.add_warning(
                f"TP flash and its K-value fallback both failed near "
                f"T={5.0 * round(T / 5.0):g} K, P={P:.2g} bar "
                f"({type(exc).__name__}); assuming a single "
                f"{'vapor' if V > 0.5 else 'liquid'} phase from temperature."
            )
            return V, dict(composition), dict(composition)

    def _iterative_K_flash_TP(self, composition: dict[str, float], T: float, P: float,
                              max_iter: int = 50, tol: float = 1e-8) -> tuple[float, dict, dict]:
        """
        TP flash with outer direct substitution on composition-dependent K-values.

        This is used by EOS phi-phi models, where K-values depend on the
        equilibrium liquid composition. Ideal Raoult-law K-values do not need
        this outer loop.
        """
        z = {
            comp: max(float(composition.get(comp, 0.0)), 0.0)
            for comp in self.components
        }
        z_total = sum(z.values())
        if z_total <= 0.0:
            z = {comp: 1.0 / len(self.components) for comp in self.components}
        else:
            z = {comp: value / z_total for comp, value in z.items()}

        K = self.K_values(T, P, z)
        sum_zK = sum(z[comp] * K.get(comp, 1.0) for comp in self.components)
        if sum_zK <= 1.0:
            return 0.0, dict(z), dict(z)

        V = 0.5
        for _ in range(max_iter):
            V = self._rachford_rice_bounded(z, K, V)
            x, y = self._flash_phase_compositions(z, K, V)
            K_new = self.K_values(T, P, x)
            max_change = 0.0
            for comp in self.components:
                K_old = max(K.get(comp, 1.0), 1e-300)
                max_change = max(max_change, abs(K_new.get(comp, 1.0) - K_old) / K_old)
            K = K_new
            if max_change < tol:
                break

        V = self._rachford_rice_bounded(z, K, V)
        return (V, *self._flash_phase_compositions(z, K, V))

    def _rachford_rice_bounded(self, z: dict[str, float], K: dict[str, float],
                               V_guess: float = 0.5) -> float:
        def residual(V_value: float) -> float:
            total = 0.0
            for comp in self.components:
                K_i = K.get(comp, 1.0)
                term = K_i - 1.0
                denom = 1.0 + V_value * term
                if abs(denom) < 1e-14:
                    denom = 1e-14 if denom >= 0.0 else -1e-14
                total += z.get(comp, 0.0) * term / denom
            return total

        f_low = residual(0.0)
        f_high = residual(1.0)
        if f_low <= 0.0:
            return 0.0
        if f_high >= 0.0:
            return 1.0

        V = max(0.0, min(1.0, float(V_guess)))
        for _ in range(20):
            f = residual(V)
            if abs(f) < 1e-12:
                return V
            derivative = 0.0
            for comp in self.components:
                K_i = K.get(comp, 1.0)
                term = K_i - 1.0
                denom = 1.0 + V * term
                if abs(denom) < 1e-14:
                    denom = 1e-14 if denom >= 0.0 else -1e-14
                derivative -= z.get(comp, 0.0) * term * term / (denom * denom)
            if abs(derivative) < 1e-14:
                break
            V_new = V - f / derivative
            if not 0.0 < V_new < 1.0:
                break
            if abs(V_new - V) < 1e-12:
                return V_new
            V = V_new

        low = 0.0
        high = 1.0
        for _ in range(100):
            mid = 0.5 * (low + high)
            f_mid = residual(mid)
            if abs(f_mid) < 1e-13 or high - low < 1e-13:
                return mid
            if f_mid > 0.0:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    def _flash_phase_compositions(self, z: dict[str, float], K: dict[str, float],
                                  V: float) -> tuple[dict[str, float], dict[str, float]]:
        x = {}
        y = {}
        for comp in self.components:
            denom = 1.0 + V * (K.get(comp, 1.0) - 1.0)
            if abs(denom) < 1e-14:
                denom = 1e-14 if denom >= 0.0 else -1e-14
            x[comp] = z.get(comp, 0.0) / denom
            y[comp] = K.get(comp, 1.0) * x[comp]

        x_sum = sum(x.values())
        y_sum = sum(y.values())
        if x_sum > 0.0:
            x = {comp: value / x_sum for comp, value in x.items()}
        if y_sum > 0.0:
            y = {comp: value / y_sum for comp, value in y.items()}
        return x, y
    
    def flash_PV(self, composition: dict[str, float], P: float, 
                 vapor_frac: float) -> tuple[float, dict, dict]:
        """
        PV Flash: find temperature for specified vapor fraction.
        
        Args:
            composition: Feed mole fractions
            P: Pressure [bar]
            vapor_frac: Target vapor fraction
            
        Returns:
            (T, x_liquid, y_vapor)
        """
        # Bracket the temperature
        T_low = self.bubble_point_T(composition, P)
        T_high = self.dew_point_T(composition, P)
        
        if vapor_frac <= 0.001:
            V, x, y = self.flash_TP(composition, T_low, P)
            return T_low, x, y
        
        if vapor_frac >= 0.999:
            V, x, y = self.flash_TP(composition, T_high, P)
            return T_high, x, y
        
        # Bisection to find T
        for _ in range(50):
            T_mid = (T_low + T_high) / 2
            V, x, y = self.flash_TP(composition, T_mid, P)
            
            if abs(V - vapor_frac) < 1e-6:
                return T_mid, x, y
            
            if V < vapor_frac:
                T_low = T_mid
            else:
                T_high = T_mid
        
        return T_mid, x, y
    
    def enthalpy_ideal_gas(self, comp: str, T: float) -> float:
        """
        Ideal gas enthalpy [kJ/mol] relative to reference state.
        H = Hf + integral(Cp dT) from T_ref to T
        """
        cache_key = (comp, float(T))
        cached = self._enthalpy_ideal_cache.get(cache_key)
        if cached is not None:
            return cached
        props = self.props.get(comp)
        if props is None:
            return 0.0
        
        # Formation enthalpy
        Hf = props.Hf if props.Hf else 0.0
        if props.Hf is not None:
            self.mark_property_source_context_once(
                comp,
                'Hf',
                phase='ideal_gas_enthalpy',
            )
        
        if abs(T - T_REF) < 1e-12:
            delta_H = 0.0
        else:
            delta_H = self._integrate_cp_analytic(comp, T_REF, T, 'ideal_gas')
            if delta_H is None:
                steps = 16
                h = (T - T_REF) / steps
                total = self.Cp_ideal_gas(comp, T_REF) + self.Cp_ideal_gas(comp, T)
                for index in range(1, steps):
                    weight = 4 if index % 2 else 2
                    total += weight * self.Cp_ideal_gas(comp, T_REF + index * h)
                delta_H = total * h / 3.0 / 1000.0
        
        return self._set_limited_cache(self._enthalpy_ideal_cache, cache_key, Hf + delta_H)
    
    def enthalpy_liquid(self, comp: str, T: float) -> float:
        """
        Liquid enthalpy [kJ/mol] relative to reference state.

        Reference convention:
            H_liq(T_REF) = H_ig(T_REF) - Hvap(T_REF)

        Then integrate liquid heat capacity from T_REF to T. This keeps the
        liquid enthalpy slope tied to Cp_liquid while anchoring the liquid and
        vapor reference states through a temperature-dependent latent heat.
        """
        cache_key = (comp, float(T))
        cached = self._enthalpy_liquid_cache.get(cache_key)
        if cached is not None:
            return cached

        H_liq_ref = self.enthalpy_ideal_gas(comp, T_REF) - self.Hvap_at_T(comp, T_REF)
        delta_H = self._integrate_liquid_cp(comp, T_REF, T)
        return self._set_limited_cache(self._enthalpy_liquid_cache, cache_key, H_liq_ref + delta_H)

    def enthalpy_solid(self, comp: str, T: float) -> float:
        """Solid enthalpy [kJ/mol] anchored to Hf_solid at 298.15 K."""
        cache_key = (comp, float(T))
        cached = self._enthalpy_solid_cache.get(cache_key)
        if cached is not None:
            return cached
        props = self.props.get(comp)
        reference = float(props.Hf_solid) if props is not None and props.Hf_solid is not None else 0.0
        if props is not None and props.Hf_solid is not None:
            self.mark_property_source_context_once(comp, 'Hf_solid', phase='solid_enthalpy')
        value = reference + self._integrate_solid_cp(comp, T_REF, T)
        return self._set_limited_cache(self._enthalpy_solid_cache, cache_key, value)

    def _integrate_cp_over_T(self, comp: str, T1: float, T2: float,
                             phase: str = 'ideal_gas') -> float:
        """Integral of Cp/T from T1 to T2 [kJ/kmol-K]."""
        if abs(T2 - T1) < 1e-12:
            return 0.0
        if phase in {'ideal_gas', 'liquid', 'solid'}:
            kernel = (
                self._ideal_gas_cp_kernel(comp)
                if phase == 'ideal_gas'
                else self._liquid_cp_kernel(comp)
                if phase == 'liquid'
                else self._solid_cp_kernel(comp)
            )
            if kernel is not None:
                try:
                    if phase == 'ideal_gas':
                        self._record_ideal_gas_cp_kernel_range_use(comp, T1, kernel)
                        self._record_ideal_gas_cp_kernel_range_use(comp, T2, kernel)
                    else:
                        recorder = (
                            self._record_liquid_cp_kernel_range_use
                            if phase == 'liquid'
                            else self._record_solid_cp_kernel_range_use
                        )
                        recorder(comp, T1, kernel)
                        recorder(comp, T2, kernel)
                    return kernel.delta_s(T1, T2)
                except Exception as exc:
                    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                        from ..property_resolution.solid_cp import SolidCpTransitionError
                    else:
                        from property_resolution.solid_cp import SolidCpTransitionError
                    if isinstance(exc, SolidCpTransitionError):
                        raise
                    pass
        steps = 32
        h = (T2 - T1) / steps

        def cp_over_T(temperature: float) -> float:
            if phase == 'liquid':
                cp = self.Cp_liquid(comp, temperature)
            elif phase == 'solid':
                cp = self.Cp_solid(comp, temperature)
            else:
                cp = self.Cp_ideal_gas(comp, temperature)
            return cp / max(temperature, 1e-12)

        total = cp_over_T(T1) + cp_over_T(T2)
        for index in range(1, steps):
            weight = 4 if index % 2 else 2
            total += weight * cp_over_T(T1 + index * h)
        # Cp is J/mol-K, which is numerically kJ/kmol-K.
        return total * h / 3.0

    def entropy_ideal_gas(self, comp: str, T: float) -> float:
        """
        Ideal-gas molar entropy [kJ/kmol-K] at the standard pressure.

        Absolute standard entropies are used when available; otherwise the
        component gets a zero reference at T_REF. Isentropic same-composition
        calculations are unaffected by that arbitrary missing-data offset.
        """
        cache_key = (comp, float(T))
        cached = self._entropy_ideal_cache.get(cache_key)
        if cached is not None:
            return cached
        props = self.props.get(comp)
        S_ref = float(props.S) if props is not None and props.S is not None else 0.0
        if props is not None and props.S is not None:
            self.mark_property_source_context_once(
                comp,
                'S',
                phase='ideal_gas_entropy',
            )
        delta_S = self._integrate_cp_over_T(comp, T_REF, T, 'ideal_gas')
        return self._set_limited_cache(self._entropy_ideal_cache, cache_key, S_ref + delta_S)

    def entropy_liquid(self, comp: str, T: float) -> float:
        """
        Liquid molar entropy [kJ/kmol-K] at the standard pressure.

        Prefer a reversible vaporization path at the target temperature:
        S_liq(T) ~= S_ig(T, Psat) - Hvap(T) / T. Compressed-liquid pressure
        effects are neglected by the IDEAL model. If the saturation path is not
        available, fall back to a reference-temperature liquid Cp path.
        """
        cache_key = (comp, float(T))
        cached = self._entropy_liquid_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            Psat = max(float(self.Psat(comp, T)), 1e-300)
            S_vap_sat = self.entropy_ideal_gas(comp, T) + self._pressure_entropy_correction(Psat)
            S_liq = S_vap_sat - 1000.0 * self.Hvap_at_T(comp, T) / max(T, 1e-12)
            return self._set_limited_cache(self._entropy_liquid_cache, cache_key, S_liq)
        except Exception:
            pass

        S_liq_ref = self.entropy_ideal_gas(comp, T_REF) - 1000.0 * self.Hvap_at_T(comp, T_REF) / T_REF
        delta_S = self._integrate_cp_over_T(comp, T_REF, T, 'liquid')
        return self._set_limited_cache(self._entropy_liquid_cache, cache_key, S_liq_ref + delta_S)

    def entropy_solid(self, comp: str, T: float) -> float:
        """Solid molar entropy [kJ/kmol-K] anchored to S_solid at 298.15 K."""
        cache_key = (comp, float(T))
        cached = self._entropy_solid_cache.get(cache_key)
        if cached is not None:
            return cached
        props = self.props.get(comp)
        reference = float(props.S_solid) if props is not None and props.S_solid is not None else 0.0
        if props is not None and props.S_solid is not None:
            self.mark_property_source_context_once(comp, 'S_solid', phase='solid_entropy')
        value = reference + self._integrate_cp_over_T(comp, T_REF, T, 'solid')
        return self._set_limited_cache(self._entropy_solid_cache, cache_key, value)

    def standard_chemical_potential(self, comp: str, T: float) -> float:
        """Ideal-gas standard chemical potential at ``P_REF`` [kJ/kmol].

        The temperature path is constructed from the resolved ideal-gas
        formation enthalpy, absolute entropy, and heat capacity.  Chemical
        reaction models must not silently inherit the nonreacting-stream zero
        references used when formation data are absent.
        """
        props = self.props.get(comp)
        if props is None:
            raise ThermodynamicsError(f"Component '{comp}' is unavailable")
        missing = [
            name for name in ('Hf', 'S')
            if getattr(props, name, None) is None
        ]
        if missing:
            raise ThermodynamicsError(
                f"Component '{comp}' requires {', '.join(missing)} for a "
                "standard chemical potential"
            )
        self.mark_property_source_context_once(
            comp,
            ('Hf', 'S'),
            phase='reaction_standard_chemical_potential',
            description='Chemical-reaction standard-state Gibbs energy',
        )
        return (
            1000.0 * self.enthalpy_ideal_gas(comp, T)
            - float(T) * self.entropy_ideal_gas(comp, T)
        )

    def reaction_standard_gibbs(
        self,
        stoichiometry: dict[str, float],
        T: float,
    ) -> float:
        """Standard Gibbs energy for a stoichiometric reaction [kJ/kmol]."""
        return sum(
            float(coefficient) * self.standard_chemical_potential(component, T)
            for component, coefficient in stoichiometry.items()
        )

    def reaction_log_equilibrium_constant(
        self,
        stoichiometry: dict[str, float],
        T: float,
    ) -> float:
        """Return ``ln(K)`` on the dimensionless 1-bar standard-state basis."""
        temperature = float(T)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ThermodynamicsError(
                "Reaction equilibrium temperature must be positive and finite"
            )
        return -self.reaction_standard_gibbs(stoichiometry, temperature) / (
            R * temperature
        )

    def component_activities(
        self,
        T: float,
        P: float,
        composition: dict[str, float],
        phase: str,
    ) -> dict[str, float]:
        """Dimensionless component activities for homogeneous reactions.

        Vapor activities use ``y*phi*P/P_REF``.  Activity-coefficient liquids
        use the existing gamma-phi pure-liquid reference fugacity, while cubic
        EOS liquids use ``x*phi_L*P/P_REF``.  The IDEAL liquid fallback is the
        Raoult reference ``x*Psat/P_REF``.
        """
        temperature = float(T)
        pressure = float(P)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ThermodynamicsError(
                "Reaction activity temperature must be positive and finite"
            )
        if not math.isfinite(pressure) or pressure <= 0.0:
            raise ThermodynamicsError(
                "Reaction activity pressure must be positive and finite"
            )
        total = sum(max(float(value), 0.0) for value in composition.values())
        if total <= 0.0:
            raise ThermodynamicsError("Reaction activity composition is empty")
        values = {
            component: max(float(composition.get(component, 0.0)), 0.0) / total
            for component in self.components
        }
        phase_name = str(phase).strip().lower()
        if phase_name in {'vapor', 'gas'}:
            fugacity_method = getattr(self, 'fugacity_coefficients', None)
            phi = (
                fugacity_method(temperature, pressure, values, 'vapor')
                if callable(fugacity_method)
                else {component: 1.0 for component in self.components}
            )
            return {
                component: values[component]
                * max(float(phi.get(component, 1.0)), 1.0e-300)
                * pressure / P_REF
                for component in self.components
            }
        if phase_name != 'liquid':
            raise ThermodynamicsError(
                "Reaction activities require phase='vapor' or phase='liquid'"
            )

        activity_method = getattr(self, 'activity_coefficients', None)
        reference_method = getattr(self, '_gamma_phi_reference_factors', None)
        if callable(activity_method) and callable(reference_method):
            gamma = activity_method(temperature, values)
            reference = reference_method(temperature, pressure)
            return {
                component: values[component]
                * max(float(gamma.get(component, 1.0)), 1.0e-300)
                * max(float(reference[component]), 1.0e-300)
                / P_REF
                for component in self.components
            }

        fugacity_method = getattr(self, 'fugacity_coefficients', None)
        if callable(fugacity_method):
            phi = fugacity_method(temperature, pressure, values, 'liquid')
            return {
                component: values[component]
                * max(float(phi.get(component, 1.0)), 1.0e-300)
                * pressure / P_REF
                for component in self.components
            }

        return {
            component: values[component]
            * max(float(self.Psat(component, temperature)), 1.0e-300)
            / P_REF
            for component in self.components
        }

    def _ideal_mixing_entropy(self, composition: dict[str, float]) -> float:
        """Ideal entropy of mixing [kJ/kmol-K]."""
        return -R * sum(
            max(float(z), 0.0) * math.log(max(float(z), 1e-300))
            for z in composition.values()
            if z > 0.0
        )

    def _pressure_entropy_correction(self, P: float) -> float:
        """Ideal-gas pressure entropy correction from 1 bar standard state to P [kJ/kmol-K]."""
        return -R * math.log(max(float(P), 1e-300) / P_REF)

    def _ideal_gas_mixture_entropy(self, composition: dict[str, float], T: float,
                                   P: float = P_REF) -> float:
        return (
            sum(z * self.entropy_ideal_gas(comp, T) for comp, z in composition.items())
            + self._ideal_mixing_entropy(composition)
            + self._pressure_entropy_correction(P)
        )

    def _ideal_liquid_mixture_entropy(self, composition: dict[str, float], T: float) -> float:
        return (
            sum(z * self.entropy_liquid(comp, T) for comp, z in composition.items())
            + self._ideal_mixing_entropy(composition)
        )

    def mixture_entropy(self, composition: dict[str, float], T: float,
                        vapor_fraction: float = 1.0,
                        x: Optional[dict] = None,
                        y: Optional[dict] = None,
                        P: float = P_REF) -> float:
        """Mixture molar entropy [kJ/kmol-K]."""
        if vapor_fraction > 0.999:
            return self._ideal_gas_mixture_entropy(composition, T, P)
        if vapor_fraction < 0.001:
            return self._ideal_liquid_mixture_entropy(composition, T)

        x = x or composition
        y = y or composition
        S_liq = self._ideal_liquid_mixture_entropy(x, T)
        S_vap = self._ideal_gas_mixture_entropy(y, T, P)
        return vapor_fraction * S_vap + (1.0 - vapor_fraction) * S_liq
    
    def mixture_enthalpy(self, composition: dict[str, float], T: float,
                        vapor_fraction: float = 1.0,
                        x: Optional[dict] = None,
                        y: Optional[dict] = None,
                        P: float = P_REF) -> float:
        """
        Mixture molar enthalpy [kJ/kmol]
        
        Args:
            composition: Overall mole fractions
            T: Temperature [K]
            vapor_fraction: Vapor fraction
            x: Liquid composition (if two-phase)
            y: Vapor composition (if two-phase)
            P: Pressure [bar] (ignored by ideal/activity models)
        """
        if vapor_fraction > 0.999:
            # All vapor
            H = sum(z * self.enthalpy_ideal_gas(comp, T) 
                   for comp, z in composition.items())
        elif vapor_fraction < 0.001:
            # All liquid
            H = sum(z * self.enthalpy_liquid(comp, T)
                   for comp, z in composition.items())
        else:
            # Two-phase
            x = x or composition
            y = y or composition
            H_liq = sum(xi * self.enthalpy_liquid(comp, T) for comp, xi in x.items())
            H_vap = sum(yi * self.enthalpy_ideal_gas(comp, T) for comp, yi in y.items())
            H = vapor_fraction * H_vap + (1 - vapor_fraction) * H_liq
        
        return H * 1000  # mol to kmol

    def excess_enthalpy(self, composition: dict[str, float], T: float) -> float:
        """Excess enthalpy [kJ/kmol]. Ideal mixtures have zero excess enthalpy."""
        return 0.0

    def mixture_liquid_molar_volume(self, composition: dict[str, float], T: float) -> float:
        """Mixture liquid molar volume [m3/kmol] using resolver-backed pure volumes."""
        composition = self._normalized_liquid_transport_composition(composition)
        V_molar = 0.0
        for comp, x in composition.items():
            if comp not in self.props:
                raise ThermodynamicsError(f"Component '{comp}' not found for liquid volume calculation")
            props = self.props[comp]
            cache_key = (comp, float(T))
            volume = self._liquid_molar_volume_cache.get(cache_key)
            if volume is None:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from ..property_resolver import get_property_resolver
                else:
                    from property_resolver import get_property_resolver
                result = get_property_resolver().resolve_liquid_molar_volume(
                    props.symbol,
                    T,
                    self._resolver_known_props.get(comp),
                )
                self._record_lazy_property_source(comp, 'liquid_molar_volume', T, result)
                volume = result.value
                self._set_limited_cache(self._liquid_molar_volume_cache, cache_key, volume)
            if volume is None or volume <= 0.0:
                raise ThermodynamicsError(
                    f"Property resolver returned invalid liquid molar volume for '{comp}'"
                )
            V_molar += x * volume
        if V_molar > 0:
            return V_molar
        raise ThermodynamicsError("Cannot calculate liquid molar volume for empty composition")

    def mixture_liquid_density(self, composition: dict[str, float], T: float) -> float:
        """Mixture liquid molar density [kmol/m3]."""
        V_molar = self.mixture_liquid_molar_volume(composition, T)
        return 1.0 / V_molar

    def _pure_viscosity(self, comp: str, T: float, P: float, phase: str) -> float:
        """Resolve and cache pure-component dynamic viscosity [Pa*s]."""
        phase_key = 'vapor' if phase in {'gas', 'vapor'} else 'liquid'
        cache_key = (phase_key, comp, float(T), float(P))
        cached = self._viscosity_cache.get(cache_key)
        if cached is not None:
            return cached
        props = self.props.get(comp)
        if props is None:
            raise ThermodynamicsError(f"Component '{comp}' not found for viscosity calculation")
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            result = get_property_resolver().resolve_viscosity(
                _property_lookup_identifier(comp, props),
                T,
                phase=phase_key,
                props=self._resolver_known_props.get(comp),
                P=P,
            )
        except Exception as exc:
            raise ThermodynamicsError(
                f"Cannot resolve {phase_key} viscosity for {self._component_label(comp)} "
                f"at T={T:.1f} K and P={P:.4g} bar"
            ) from exc
        value = float(result.value)
        if value <= 0.0 or not math.isfinite(value):
            raise ThermodynamicsError(
                f"Property resolver returned invalid {phase_key} viscosity for "
                f"{self._component_label(comp)}"
            )
        self._record_lazy_property_source(comp, f'{phase_key}_viscosity', T, result)
        return self._set_limited_cache(self._viscosity_cache, cache_key, value)

    @staticmethod
    def _normalized_positive_composition(composition: dict[str, float]) -> dict[str, float]:
        values = {
            comp: max(float(value), 0.0)
            for comp, value in composition.items()
        }
        total = sum(values.values())
        if total <= 0.0:
            raise ThermodynamicsError("Cannot calculate mixture property for empty composition")
        return {comp: value / total for comp, value in values.items() if value > 0.0}

    def _normalized_liquid_transport_composition(
        self,
        composition: dict[str, float],
    ) -> dict[str, float]:
        """Drop solver-floor traces before resolving pure-liquid properties."""
        normalized = self._normalized_positive_composition(composition)
        retained = {
            comp: value
            for comp, value in normalized.items()
            if value > self.LIQUID_TRANSPORT_TRACE_CUTOFF
        }
        if not retained:
            retained = {max(normalized, key=normalized.get): 1.0}
        excluded = sorted(set(normalized) - set(retained))
        if excluded:
            self.add_warning(
                "Trace component(s) omitted from liquid density/viscosity mixing "
                f"at x <= {self.LIQUID_TRANSPORT_TRACE_CUTOFF:g}: "
                + ", ".join(self._component_label(comp) for comp in excluded)
            )
        total = sum(retained.values())
        return {comp: value / total for comp, value in retained.items()}

    def _mixture_vapor_viscosity(self, composition: dict[str, float], T: float, P: float) -> float:
        """Gas mixture viscosity [Pa*s] from Wilke's rule."""
        y = self._normalized_positive_composition(composition)
        if len(y) == 1:
            comp = next(iter(y))
            return self._pure_viscosity(comp, T, P, 'vapor')

        components = list(y)
        mus = {
            comp: self._pure_viscosity(comp, T, P, 'vapor')
            for comp in components
        }
        mws = {
            comp: float(self.props[comp].MW)
            for comp in components
        }
        total = 0.0
        for comp_i in components:
            denom = 0.0
            mu_i = mus[comp_i]
            mw_i = mws[comp_i]
            for comp_j in components:
                mu_j = mus[comp_j]
                mw_j = mws[comp_j]
                phi_ij = (
                    (1.0 + math.sqrt(mu_i / mu_j) * (mw_j / mw_i) ** 0.25) ** 2
                    / math.sqrt(8.0 * (1.0 + mw_i / mw_j))
                )
                denom += y[comp_j] * phi_ij
            total += y[comp_i] * mu_i / max(denom, 1e-300)
        if total <= 0.0 or not math.isfinite(total):
            raise ThermodynamicsError("Calculated nonpositive vapor mixture viscosity")
        return total

    def _mixture_liquid_viscosity(self, composition: dict[str, float], T: float, P: float) -> float:
        """Liquid mixture viscosity [Pa*s] from the configured mixture hierarchy."""
        x = self._normalized_liquid_transport_composition(composition)
        pure_viscosities = {
            comp: self._pure_viscosity(comp, T, P, 'liquid')
            for comp in x
        }
        if len(x) == 1:
            return next(iter(pure_viscosities.values()))

        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..liquid_mixture_viscosity import estimate_liquid_mixture_viscosity
        else:
            from liquid_mixture_viscosity import estimate_liquid_mixture_viscosity

        result = estimate_liquid_mixture_viscosity(
            x,
            pure_viscosities,
            T,
            component_props=self.props,
            component_groups=getattr(self, 'component_groups', None),
            component_group_variant=getattr(self, 'unifac_variant', None),
            interaction_overrides=self.interaction_overrides,
        )
        self.extend_warnings(result.warnings)
        value = result.value
        if value <= 0.0 or not math.isfinite(value):
            raise ThermodynamicsError("Calculated nonpositive liquid mixture viscosity")
        return value

    def mixture_viscosity(
        self,
        composition: dict[str, float],
        T: float,
        P: float,
        vapor_fraction: float = 1.0,
        x: Optional[dict] = None,
        y: Optional[dict] = None,
    ) -> float:
        """Mixture dynamic viscosity [Pa*s]."""
        V = max(0.0, min(1.0, float(vapor_fraction)))
        if V >= 0.999:
            return self._mixture_vapor_viscosity(y or composition, T, P)
        if V <= 0.001:
            return self._mixture_liquid_viscosity(x or composition, T, P)

        mu_liquid = self._mixture_liquid_viscosity(x or composition, T, P)
        mu_vapor = self._mixture_vapor_viscosity(y or composition, T, P)
        return (1.0 - V) * mu_liquid + V * mu_vapor

    def transport_mixture_viscosity(
        self,
        composition: dict[str, float],
        T: float,
        P: float,
        vapor_fraction: float = 1.0,
        x: Optional[dict] = None,
        y: Optional[dict] = None,
    ) -> TransportPhaseValues:
        """Active-phase mixture viscosities [Pa*s] for transport models."""
        V = max(0.0, min(1.0, float(vapor_fraction)))
        liquid = None
        vapor = None
        if V < 1.0:
            liquid_composition = self._normalized_positive_composition(x or composition)
            liquid = self.mixture_viscosity(
                liquid_composition, T, P, 0.0, x=liquid_composition
            )
        if V > 0.0:
            vapor_composition = self._normalized_positive_composition(y or composition)
            vapor = self.mixture_viscosity(
                vapor_composition, T, P, 1.0, y=vapor_composition
            )
        return TransportPhaseValues(liquid=liquid, vapor=vapor)

    def vapor_molar_volume_for_density(self, T: float, P: float,
                                       composition: dict[str, float]) -> float:
        """Vapor molar volume for stream density [m3/kmol]."""
        if P <= 0.0:
            raise ThermodynamicsError("Pressure must be positive for vapor density calculation")
        return 1000.0 * R_BAR * T / P

    def mixture_molar_density(self, composition: dict[str, float], T: float, P: float,
                              vapor_fraction: float = 1.0,
                              x: Optional[dict] = None,
                              y: Optional[dict] = None) -> float:
        """Bulk molar density [kmol/m3] from phase molar-volume averaging."""
        V = max(0.0, min(1.0, float(vapor_fraction)))
        if V <= 0.001:
            return self.mixture_liquid_density(x or composition, T)
        if V >= 0.999:
            vapor_volume = self.vapor_molar_volume_for_density(T, P, y or composition)
            return 1.0 / vapor_volume

        liquid_volume = self.mixture_liquid_molar_volume(x or composition, T)
        vapor_volume = self.vapor_molar_volume_for_density(T, P, y or composition)
        mixture_volume = (1.0 - V) * liquid_volume + V * vapor_volume
        if mixture_volume <= 0.0:
            raise ThermodynamicsError("Calculated nonpositive mixture molar volume")
        return 1.0 / mixture_volume

    def transport_mixture_density(
        self,
        composition: dict[str, float],
        T: float,
        P: float,
        vapor_fraction: float = 1.0,
        x: Optional[dict] = None,
        y: Optional[dict] = None,
    ) -> TransportPhaseValues:
        """Active-phase mixture mass densities [kg/m3] for transport models."""
        V = max(0.0, min(1.0, float(vapor_fraction)))
        liquid = None
        vapor = None
        if V < 1.0:
            liquid_composition = self._normalized_positive_composition(x or composition)
            liquid_molar_density = self.mixture_molar_density(
                liquid_composition, T, P, 0.0, x=liquid_composition
            )
            liquid = liquid_molar_density * self.mixture_MW(liquid_composition)
        if V > 0.0:
            vapor_composition = self._normalized_positive_composition(y or composition)
            vapor_molar_density = self.mixture_molar_density(
                vapor_composition, T, P, 1.0, y=vapor_composition
            )
            vapor = vapor_molar_density * self.mixture_MW(vapor_composition)
        return TransportPhaseValues(liquid=liquid, vapor=vapor)

    def _pure_surface_tension(self, comp: str, T: float) -> float:
        """Resolve and cache pure-component liquid-vapor surface tension [N/m]."""
        cache_key = (comp, float(T))
        cached = self._surface_tension_cache.get(cache_key)
        if cached is not None:
            return cached
        props = self.props.get(comp)
        if props is None:
            raise ThermodynamicsError(
                f"Component '{comp}' not found for surface-tension calculation"
            )
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            result = get_property_resolver().resolve_surface_tension(
                _property_lookup_identifier(comp, props),
                T,
                props=self._resolver_known_props.get(comp),
            )
        except Exception as exc:
            raise ThermodynamicsError(
                f"Cannot resolve surface tension for {self._component_label(comp)} "
                f"at T={T:.1f} K"
            ) from exc
        value = float(result.value)
        if value <= 0.0 or not math.isfinite(value):
            raise ThermodynamicsError(
                f"Property resolver returned invalid surface tension for "
                f"{self._component_label(comp)}"
            )
        self._record_lazy_property_source(comp, 'surface_tension', T, result)
        return self._set_limited_cache(self._surface_tension_cache, cache_key, value)

    def _surface_tension_calculator(self, components: tuple[str, ...], method: str):
        key = (str(method), *components)
        cached = self._surface_tension_calculators.get(key)
        if cached is not None:
            return cached
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..interfacial_properties import (
                                MixtureSurfaceTensionCalculator,
                                SurfaceComponent,
                            )
            else:
                from interfacial_properties import (
                                MixtureSurfaceTensionCalculator,
                                SurfaceComponent,
                            )
        except Exception as exc:
            raise ThermodynamicsError(
                "Mixture surface-tension calculator is unavailable"
            ) from exc

        surface_components = []
        for comp in components:
            props = self.props.get(comp)
            if props is None:
                raise ThermodynamicsError(
                    f"Component '{comp}' not found for surface-tension calculation"
                )
            critical_volume = (
                None if props.Vc is None else float(props.Vc) * 1.0e-6
            )
            surface_components.append(
                SurfaceComponent(
                    name=comp,
                    Vc=critical_volume,
                    rho_molar=(
                        lambda temperature, component=comp: (
                            self.mixture_liquid_density(
                                {component: 1.0},
                                float(temperature),
                            )
                            * 1000.0
                        )
                    ),
                    sigma=(
                        lambda temperature, component=comp: self._pure_surface_tension(
                            component,
                            float(temperature),
                        )
                    ),
                    CAS=getattr(props, 'CAS', None),
                )
            )
        activity_model = self if hasattr(self, 'activity_coefficients') else None
        calculator = MixtureSurfaceTensionCalculator(
            surface_components,
            activity_model=activity_model,
        )
        self._surface_tension_calculators[key] = calculator
        return calculator

    def transport_mixture_surface_tension(
        self,
        composition: dict[str, float],
        T: float,
        P: float,
        vapor_fraction: float = 1.0,
        x: Optional[dict] = None,
        y: Optional[dict] = None,
        *,
        method: str = 'auto',
    ) -> float:
        """Liquid-vapor/gas surface tension [N/m] for transport models."""
        V = max(0.0, min(1.0, float(vapor_fraction)))
        if V <= 0.0 or V >= 1.0:
            raise ThermodynamicsError(
                "Surface tension is only defined for two active phases"
            )
        liquid_composition = self._normalized_positive_composition(x or composition)
        active = tuple(liquid_composition)
        if len(active) == 1:
            return self._pure_surface_tension(active[0], T)
        calculator = self._surface_tension_calculator(active, method)
        try:
            result = calculator.calculate(
                T,
                liquid_composition,
                method=method,
            )
        except Exception as exc:
            raise ThermodynamicsError(
                f"Cannot calculate mixture surface tension at T={T:.1f} K "
                f"and P={P:.4g} bar"
            ) from exc
        value = float(result.sigma)
        if value <= 0.0 or not math.isfinite(value):
            raise ThermodynamicsError("Calculated invalid mixture surface tension")
        return value

    @staticmethod
    def _normalize_state_include(include: Optional[Union[str, Iterable[str]]]) -> frozenset[str]:
        """Normalize optional StreamState property names."""
        if include is None:
            return DEFAULT_STATE_INCLUDE
        if isinstance(include, str):
            raw_names = (include,)
        else:
            raw_names = include
        normalized = set()
        aliases = {
            'h': 'H',
            'enthalpy': 'H',
            'cp': 'Cp',
            'heat_capacity': 'Cp',
            's': 'S',
            'entropy': 'S',
            'rho': 'rho',
            'density': 'rho',
            'mu': 'mu',
            'viscosity': 'mu',
        }
        for name in raw_names:
            key = str(name).strip()
            if not key:
                continue
            canonical = aliases.get(key.lower())
            if canonical is None:
                raise ThermodynamicsError(f"Unknown calculate_state include property: {name!r}")
            normalized.add(canonical)
        return frozenset(normalized)

    def _split_permanent_solid_composition(
        self,
        composition: dict[str, float],
    ) -> tuple[dict[str, float], float, dict[str, float], float, dict[str, float]]:
        """Return normalized total, fluid, and permanent-solid compositions."""
        positive = {
            str(component): max(0.0, float(value))
            for component, value in composition.items()
            if float(value) > 0.0
        }
        total = sum(positive.values())
        if total <= 0.0:
            raise ThermodynamicsError("Stream composition must contain positive flow")
        normalized = {
            component: value / total
            for component, value in positive.items()
        }
        solid_set = set(self.permanent_solid_components)
        unknown = sorted(set(normalized) - set(self.process_components))
        if unknown:
            raise ThermodynamicsError(
                "Stream composition contains unknown process component(s): "
                + ', '.join(unknown)
            )
        solid_fraction = sum(
            value for component, value in normalized.items()
            if component in solid_set
        )
        fluid_fraction = max(0.0, 1.0 - solid_fraction)
        fluid_composition = (
            {
                component: value / fluid_fraction
                for component, value in normalized.items()
                if component not in solid_set and value > 0.0
            }
            if fluid_fraction > 1.0e-15 else {}
        )
        solid_composition = (
            {
                component: value / solid_fraction
                for component, value in normalized.items()
                if component in solid_set and value > 0.0
            }
            if solid_fraction > 1.0e-15 else {}
        )
        return (
            normalized,
            fluid_fraction,
            fluid_composition,
            solid_fraction,
            solid_composition,
        )

    def _solid_molar_volume(self, component: str, T: float) -> float:
        props = self.props.get(component)
        if props is None:
            raise ThermodynamicsError(
                f"No properties for permanent-solid component '{component}'"
            )
        cache_key = (component, float(T))
        cached = self._solid_molar_volume_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..property_resolver import get_property_resolver
            else:
                from property_resolver import get_property_resolver
            result = get_property_resolver().resolve_solid_molar_volume(
                props.symbol,
                T,
                self._resolver_known_props.get(component),
                allow_online=self._allow_online_lookup_for_component(component),
            )
            value = float(result.value)
        except Exception as exc:
            raise ThermodynamicsError(
                f"Cannot resolve solid molar volume for '{component}' at T={T:.1f} K"
            ) from exc
        if value <= 0.0 or not math.isfinite(value):
            raise ThermodynamicsError(
                f"Resolved invalid solid molar volume for '{component}'"
            )
        self._record_lazy_property_source(
            component,
            'solid_molar_volume',
            T,
            result,
        )
        return self._set_limited_cache(
            self._solid_molar_volume_cache,
            cache_key,
            value,
        )

    def _combine_permanent_solid_state(
        self,
        *,
        T: float,
        P: float,
        F: float,
        composition: dict[str, float],
        fluid_fraction: float,
        solid_fraction: float,
        solid_composition: dict[str, float],
        fluid_state: Optional[StreamState],
        include_set: frozenset[str],
    ) -> StreamState:
        """Combine a fluid-subtotal state with authoritative permanent solids."""
        state = StreamState(
            T=float(T),
            P=float(P),
            F=float(F),
            composition=dict(composition),
            vapor_fraction=(
                fluid_fraction * float(fluid_state.vapor_fraction)
                if fluid_state is not None else 0.0
            ),
            liquid1_fraction=(
                fluid_fraction * fluid_state.effective_liquid1_fraction
                if fluid_state is not None else 0.0
            ),
            liquid2_fraction=(
                fluid_fraction * float(fluid_state.liquid2_fraction)
                if fluid_state is not None else 0.0
            ),
            x=(dict(fluid_state.x) if fluid_state is not None and fluid_state.x else None),
            y=(dict(fluid_state.y) if fluid_state is not None and fluid_state.y else None),
            x1=(dict(fluid_state.x1) if fluid_state is not None and fluid_state.x1 else None),
            x2=(dict(fluid_state.x2) if fluid_state is not None and fluid_state.x2 else None),
            solid_fraction=solid_fraction,
            solid_composition=dict(solid_composition) or None,
            solid_component_flows={
                component: float(F) * float(composition.get(component, 0.0))
                for component in solid_composition
                if float(composition.get(component, 0.0)) > 0.0
            },
            solid_particle_properties={
                component: dict(self.solid_particle_defaults.get(component, {}))
                for component in solid_composition
                if self.solid_particle_defaults.get(component)
            },
            fluid_phase_model=self.fluid_phase_model,
            phase_status=(
                f"solid_bearing_{fluid_state.phase_status}"
                if fluid_state is not None else 'solid_only'
            ),
            phase_stability=(
                fluid_state.phase_stability
                if fluid_state is not None else 'permanent_solid_assignment'
            ),
            phase_details=(
                dict(fluid_state.phase_details)
                if fluid_state is not None else {}
            ),
        )
        state.phase_details['permanent_solids'] = {
            'components': list(solid_composition),
            'solid_fraction': solid_fraction,
            'fluid_fraction': fluid_fraction,
            'model': 'permanent_solid',
        }
        state.MW = self.mixture_MW(composition)

        solid_total_fractions = {
            component: float(composition.get(component, 0.0))
            for component in solid_composition
        }
        if 'Cp' in include_set:
            fluid_cp = fluid_state.Cp if fluid_state is not None else 0.0
            if fluid_state is not None and fluid_cp is None:
                state.Cp = None
            else:
                state.Cp = fluid_fraction * float(fluid_cp) + sum(
                    fraction * self.Cp_solid(component, T)
                    for component, fraction in solid_total_fractions.items()
                )
        if 'H' in include_set:
            fluid_H = fluid_state.H if fluid_state is not None else 0.0
            if fluid_state is not None and fluid_H is None:
                state.H = None
            else:
                state.H = fluid_fraction * float(fluid_H) + sum(
                    fraction * 1000.0 * self.enthalpy_solid(component, T)
                    for component, fraction in solid_total_fractions.items()
                )
        if 'S' in include_set:
            fluid_S = fluid_state.S if fluid_state is not None else 0.0
            if fluid_state is not None and fluid_S is None:
                state.S = None
            else:
                state.S = fluid_fraction * float(fluid_S) + sum(
                    fraction * self.entropy_solid(component, T)
                    for component, fraction in solid_total_fractions.items()
                )
        if 'rho' in include_set:
            if fluid_state is not None and fluid_state.rho is None:
                state.rho = None
            else:
                molar_volume = sum(
                    fraction * self._solid_molar_volume(component, T)
                    for component, fraction in solid_total_fractions.items()
                )
                if fluid_state is not None:
                    molar_volume += fluid_fraction / float(fluid_state.rho)
                if molar_volume <= 0.0:
                    raise ThermodynamicsError(
                        "Calculated nonpositive solid-bearing stream molar volume"
                    )
                state.rho = 1.0 / molar_volume
        if 'mu' in include_set:
            # Apparent slurry/powder viscosity requires particle and rheology
            # models that are intentionally outside the permanent-solid layer.
            state.mu = None
        return state

    def _permanent_solid_state_at_TP(
        self,
        T: float,
        P: float,
        F: float,
        composition: dict[str, float],
        phase: Optional[str],
        flash: bool,
        include: Optional[Union[str, Iterable[str]]],
    ) -> Optional[StreamState]:
        """Return a combined state when permanent solids are present."""
        if not self.permanent_solid_components:
            return None
        (
            normalized,
            fluid_fraction,
            fluid_composition,
            solid_fraction,
            solid_composition,
        ) = self._split_permanent_solid_composition(composition)
        if solid_fraction <= 1.0e-15:
            return None
        include_set = self._normalize_state_include(include)
        fluid_state = None
        if fluid_fraction > 1.0e-15:
            fluid_state = self.calculate_state(
                T,
                P,
                F * fluid_fraction,
                fluid_composition,
                phase=phase,
                flash=flash,
                include=include_set,
            )
        return self._combine_permanent_solid_state(
            T=T,
            P=P,
            F=F,
            composition=normalized,
            fluid_fraction=fluid_fraction,
            solid_fraction=solid_fraction,
            solid_composition=solid_composition,
            fluid_state=fluid_state,
            include_set=include_set,
        )
    
    def _populate_multifluid_state_properties(
        self,
        state: StreamState,
        include_set: frozenset[str],
    ) -> None:
        """Populate bulk properties from retained vapor, liquids, and solids."""
        V = max(0.0, float(state.vapor_fraction))
        L1 = max(0.0, state.effective_liquid1_fraction)
        L2 = max(0.0, float(state.liquid2_fraction))
        solid_fractions = (
            {
                component: max(0.0, float(flow)) / state.F
                for component, flow in state.solid_component_flows.items()
                if state.F > 0.0 and float(flow) > 0.0
            }
            if state.solid_component_flows else {
                component: max(0.0, float(state.solid_fraction)) * max(0.0, float(value))
                for component, value in (state.solid_composition or {}).items()
            }
        )
        S = sum(solid_fractions.values())
        total = V + L1 + L2 + S
        if total <= 0.0:
            raise ThermodynamicsError("Multiphase state has no active phase")
        if abs(total - 1.0) > 1.0e-9:
            V, L1, L2 = V / total, L1 / total, L2 / total
            solid_fractions = {
                component: value / total
                for component, value in solid_fractions.items()
            }
            S = sum(solid_fractions.values())
        y = state.y or state.composition
        x1 = state.x1 or state.x or state.composition
        x2 = state.x2 or state.x or state.composition

        if 'Cp' in include_set:
            state.Cp = (
                V * self.mixture_Cp(y, state.T, 1.0, state.P)
                + L1 * self.mixture_Cp(x1, state.T, 0.0, state.P)
                + L2 * self.mixture_Cp(x2, state.T, 0.0, state.P)
                + sum(
                    fraction * self.Cp_solid(component, state.T)
                    for component, fraction in solid_fractions.items()
                )
            )
        if 'H' in include_set:
            state.H = (
                V * self.mixture_enthalpy(y, state.T, 1.0, None, y, state.P)
                + L1 * self.mixture_enthalpy(x1, state.T, 0.0, x1, None, state.P)
                + L2 * self.mixture_enthalpy(x2, state.T, 0.0, x2, None, state.P)
                + sum(
                    fraction * 1000.0 * self.enthalpy_solid(component, state.T)
                    for component, fraction in solid_fractions.items()
                )
            )
        if 'S' in include_set:
            state.S = (
                V * self.mixture_entropy(y, state.T, 1.0, None, y, state.P)
                + L1 * self.mixture_entropy(x1, state.T, 0.0, x1, None, state.P)
                + L2 * self.mixture_entropy(x2, state.T, 0.0, x2, None, state.P)
                + sum(
                    fraction * self.entropy_solid(component, state.T)
                    for component, fraction in solid_fractions.items()
                )
            )
        if 'rho' in include_set:
            phase_volumes = []
            if V > 0.0:
                rho_v = self.mixture_molar_density(y, state.T, state.P, 1.0, y=y)
                phase_volumes.append(V / rho_v)
            if L1 > 0.0:
                rho_l1 = self.mixture_molar_density(x1, state.T, state.P, 0.0, x=x1)
                phase_volumes.append(L1 / rho_l1)
            if L2 > 0.0:
                rho_l2 = self.mixture_molar_density(x2, state.T, state.P, 0.0, x=x2)
                phase_volumes.append(L2 / rho_l2)
            phase_volumes.extend(
                fraction * self._solid_molar_volume(component, state.T)
                for component, fraction in solid_fractions.items()
            )
            volume = sum(phase_volumes)
            if volume <= 0.0:
                raise ThermodynamicsError("Calculated nonpositive multifluid molar volume")
            state.rho = 1.0 / volume
        if 'mu' in include_set:
            state.mu = None if S > 0.0 else (
                V * self.mixture_viscosity(y, state.T, state.P, 1.0, y=y)
                + L1 * self.mixture_viscosity(x1, state.T, state.P, 0.0, x=x1)
                + L2 * self.mixture_viscosity(x2, state.T, state.P, 0.0, x=x2)
            )

    def _apply_fluid_equilibrium_to_state(
        self,
        state: StreamState,
        equilibrium: FluidPhaseEquilibrium,
        include_set: frozenset[str],
    ) -> StreamState:
        """Attach one equilibrium result and calculate its bulk properties."""
        state.vapor_fraction = equilibrium.vapor_fraction
        state.liquid1_fraction = equilibrium.liquid1_fraction
        state.liquid2_fraction = equilibrium.liquid2_fraction
        state.x1 = dict(equilibrium.x1) if equilibrium.x1 else None
        state.x2 = dict(equilibrium.x2) if equilibrium.x2 else None
        state.x = self._pooled_liquid_composition(
            equilibrium.liquid1_fraction,
            equilibrium.x1,
            equilibrium.liquid2_fraction,
            equilibrium.x2,
        )
        if equilibrium.liquid1_fraction + equilibrium.liquid2_fraction <= 1.0e-12:
            state.x = None
        state.y = dict(equilibrium.y) if equilibrium.y else None
        state.phase_status = equilibrium.status
        state.phase_stability = equilibrium.stability
        state.phase_details = dict(equilibrium.extra or {})
        self._record_estimated_interaction_extrapolation(
            state.T,
            (
                (equilibrium.liquid1_fraction, equilibrium.x1),
                (equilibrium.liquid2_fraction, equilibrium.x2),
            ),
        )
        state.MW = self.mixture_MW(state.composition)
        if state.liquid2_fraction > 1.0e-12:
            self._populate_multifluid_state_properties(state, include_set)
        else:
            if 'Cp' in include_set:
                state.Cp = self.phase_weighted_mixture_Cp(
                    state.composition,
                    state.T,
                    state.vapor_fraction,
                    state.x,
                    state.y,
                    state.P,
                )
            if 'H' in include_set:
                state.H = self.mixture_enthalpy(
                    state.composition,
                    state.T,
                    state.vapor_fraction,
                    state.x,
                    state.y,
                    state.P,
                )
            if 'S' in include_set:
                state.S = self.mixture_entropy(
                    state.composition,
                    state.T,
                    state.vapor_fraction,
                    state.x,
                    state.y,
                    state.P,
                )
            if 'rho' in include_set:
                state.rho = self.mixture_molar_density(
                    state.composition,
                    state.T,
                    state.P,
                    state.vapor_fraction,
                    state.x,
                    state.y,
                )
            if 'mu' in include_set:
                state.mu = self.mixture_viscosity(
                    state.composition,
                    state.T,
                    state.P,
                    state.vapor_fraction,
                    state.x,
                    state.y,
                )
        return state

    def _record_estimated_interaction_extrapolation(
        self,
        T: float,
        liquid_phases,
    ) -> None:
        """Warn only after an accepted liquid equilibrium uses an extrapolated fit."""
        metadata = getattr(self, 'estimated_interaction_metadata', None) or {}
        if not metadata:
            return
        accepted_liquids = [
            composition
            for fraction, composition in liquid_phases
            if float(fraction or 0.0) > 1.0e-12 and composition
        ]
        if not accepted_liquids:
            return
        for pair_key, details in metadata.items():
            comp1 = details['component1']
            comp2 = details['component2']
            relevant = any(
                float(composition.get(comp1, 0.0)) > 1.0e-15
                and float(composition.get(comp2, 0.0)) > 1.0e-15
                for composition in accepted_liquids
            )
            if not relevant:
                continue
            low = float(details['fit_Tmin_K'])
            high = float(details['fit_Tmax_K'])
            direction = 'below' if T < low else 'above' if T > high else None
            if direction is None:
                continue
            warning_key = (tuple(pair_key), direction)
            if warning_key in self._estimated_interaction_extrapolation_warnings:
                continue
            self._estimated_interaction_extrapolation_warnings.add(warning_key)
            self.add_warning(
                f"Accepted liquid phase equilibrium at {T:g} K uses estimated "
                f"{details['model']} interaction {comp1}/{comp2} {direction} its "
                f"{low:g}-{high:g} K {details['source']} fitting range; the "
                "frozen interaction is being extrapolated."
            )

    def calculate_state(self, T: float, P: float, F: float,
                       composition: dict[str, float],
                       phase: Optional[str] = None,
                       flash: bool = True,
                       include: Optional[Union[str, Iterable[str]]] = None) -> StreamState:
        """
        Calculate complete stream state.
        
        Args:
            T: Temperature [K]
            P: Pressure [bar]
            F: Molar flow [kmol/h]
            composition: Mole fractions
            phase: Force phase ('vapor' or 'liquid') instead of flashing
            flash: Whether to do flash calculation
            include: Optional properties to calculate. Defaults to H, Cp, S,
                and rho. Temporary probes can request only the properties they
                consume, e.g. include=('H', 'Cp').
            
        Returns:
            StreamState with all properties calculated
        """
        solid_state = self._permanent_solid_state_at_TP(
            T,
            P,
            F,
            composition,
            phase,
            flash,
            include,
        )
        if solid_state is not None:
            return solid_state
        include_set = self._normalize_state_include(include)

        # Normalize composition
        total = sum(composition.values())
        if total > 0:
            composition = {k: v/total for k, v in composition.items()}
        
        state = StreamState(
            T=T,
            P=P,
            F=F,
            composition=composition,
            fluid_phase_model=self.fluid_phase_model,
        )
        
        # Do flash if requested, unless a phase is forced by the caller.
        if phase is not None:
            phase_lower = phase.lower()
            if phase_lower in ('vapor', 'gas'):
                state.vapor_fraction = 1.0
                state.liquid1_fraction = 0.0
                state.y = dict(composition)
                state.x = None
                state.phase_status = 'forced_vapor'
                state.phase_stability = 'explicit_phase_constraint'
            elif phase_lower == 'liquid':
                state.vapor_fraction = 0.0
                state.liquid1_fraction = 1.0
                state.x = dict(composition)
                state.y = None
                state.x1 = dict(composition)
                state.phase_status = 'forced_liquid'
                state.phase_stability = 'explicit_phase_constraint'
            else:
                raise ThermodynamicsError(f"Unknown forced phase: {phase}")
        elif flash:
            try:
                equilibrium = self._fluid_phase_equilibrium_TP(
                    composition,
                    T,
                    P,
                )
            except Exception as exc:
                if self.fluid_phase_model != 'VLE':
                    raise ThermodynamicsError(
                        f"{self.fluid_phase_model} fluid phase calculation failed"
                    ) from exc
                V, x, y = self._fallback_flash_TP(composition, T, P, exc)
                equilibrium = FluidPhaseEquilibrium(
                    vapor_fraction=V,
                    liquid1_fraction=1.0 - V,
                    liquid2_fraction=0.0,
                    y=dict(y or composition),
                    x1=dict(x or composition),
                    x2={},
                    status='fallback_vle',
                    stability='vle_fallback_after_error',
                    extra={'error': str(exc)},
                )
            return self._apply_fluid_equilibrium_to_state(
                state,
                equilibrium,
                include_set,
            )
        else:
            raise ThermodynamicsError(
                "calculate_state with flash=False requires an explicit "
                "phase ('vapor' or 'liquid')"
            )

        # Calculate properties
        state.MW = self.mixture_MW(composition)
        if 'Cp' in include_set:
            state.Cp = self.phase_weighted_mixture_Cp(
                composition, T, state.vapor_fraction, state.x, state.y, P
            )
        if 'H' in include_set:
            state.H = self.mixture_enthalpy(
                composition, T, state.vapor_fraction, state.x, state.y, P
            )
        if 'S' in include_set:
            state.S = self.mixture_entropy(
                composition, T, state.vapor_fraction, state.x, state.y, P
            )
        if 'rho' in include_set:
            state.rho = self.mixture_molar_density(
                composition, T, P, state.vapor_fraction, state.x, state.y
            )
        if 'mu' in include_set:
            state.mu = self.mixture_viscosity(
                composition, T, P, state.vapor_fraction, state.x, state.y
            )
        
        return state

    def _permanent_solid_state_at_PQ(
        self,
        P: float,
        vapor_fraction: float,
        F: float,
        composition: dict[str, float],
        include: Optional[Union[str, Iterable[str]]],
    ) -> Optional[StreamState]:
        """Return a combined P/quality state when permanent solids are present."""
        if not self.permanent_solid_components:
            return None
        (
            normalized,
            fluid_fraction,
            fluid_composition,
            solid_fraction,
            solid_composition,
        ) = self._split_permanent_solid_composition(composition)
        if solid_fraction <= 1.0e-15:
            return None
        target = max(0.0, min(1.0, float(vapor_fraction)))
        if fluid_fraction <= 1.0e-15:
            raise ThermodynamicsError(
                "A pressure/vapor-fraction specification cannot determine the "
                "temperature of a permanent-solid-only stream; specify T and P"
            )
        if target > fluid_fraction + 1.0e-12:
            raise ThermodynamicsError(
                f"Requested total-stream vapor fraction {target:g} exceeds the "
                f"available fluid fraction {fluid_fraction:g}"
            )
        include_set = self._normalize_state_include(include)
        fluid_target = max(0.0, min(1.0, target / fluid_fraction))
        fluid_state = self.calculate_state_PQ(
            P,
            fluid_target,
            F * fluid_fraction,
            fluid_composition,
            include=include_set,
        )
        return self._combine_permanent_solid_state(
            T=fluid_state.T,
            P=fluid_state.P,
            F=F,
            composition=normalized,
            fluid_fraction=fluid_fraction,
            solid_fraction=solid_fraction,
            solid_composition=solid_composition,
            fluid_state=fluid_state,
            include_set=include_set,
        )

    def calculate_state_PQ(
        self,
        P: float,
        vapor_fraction: float,
        F: float,
        composition: dict[str, float],
        include: Optional[Union[str, Iterable[str]]] = None,
    ) -> StreamState:
        """Calculate a pressure/vapor-fraction state under constrained VLE."""
        solid_state = self._permanent_solid_state_at_PQ(
            P,
            vapor_fraction,
            F,
            composition,
            include,
        )
        if solid_state is not None:
            return solid_state
        target = max(0.0, min(1.0, float(vapor_fraction)))
        total = sum(composition.values())
        normalized = (
            {component: value / total for component, value in composition.items()}
            if total > 0.0 else dict(composition)
        )
        T, x, y = self.flash_PV(normalized, P, target)
        equilibrium = FluidPhaseEquilibrium(
            vapor_fraction=target,
            liquid1_fraction=1.0 - target,
            liquid2_fraction=0.0,
            y=dict(y or normalized),
            x1=dict(x or normalized),
            x2={},
            status=(
                'single_vapor' if target >= 1.0 - 1.0e-10
                else 'single_liquid' if target <= 1.0e-10
                else 'ordinary_vle'
            ),
            stability='vle_constrained',
            extra={},
        )
        state = StreamState(
            T=float(T),
            P=float(P),
            F=float(F),
            composition=normalized,
            fluid_phase_model=self.fluid_phase_model,
        )
        return self._apply_fluid_equilibrium_to_state(
            state,
            equilibrium,
            self._normalize_state_include(include),
        )
