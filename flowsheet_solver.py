"""
Flowsheet Solver

Handles:
- Topological sorting of units for calculation order
- Detection of recycle loops
- Sequential modular simulation with recycle convergence
- Stream initialization from specifications
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
from collections import defaultdict
import time
import math

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics import StreamState, IdealThermodynamics, ThermodynamicsError
else:
    from thermodynamics import StreamState, IdealThermodynamics, ThermodynamicsError
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .recycle_controls import (
        normalize_recycle_method,
        resolved_recycle_options,
    )
else:
    from recycle_controls import (
        normalize_recycle_method,
        resolved_recycle_options,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_conversions import (
        mass_flow_to_kg_per_hour,
        molar_flow_to_kmol_per_hour,
        pressure_unit_factor,
        temperature_to_kelvin,
    )
else:
    from unit_conversions import (
        mass_flow_to_kg_per_hour,
        molar_flow_to_kmol_per_hour,
        pressure_unit_factor,
        temperature_to_kelvin,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations import (
        UnitOperation, UnitResult, UnitOperationError, 
        create_unit, UNIT_CLASSES
    )
else:
    from unit_operations import (
        UnitOperation, UnitResult, UnitOperationError, 
        create_unit, UNIT_CLASSES
    )


class FlowsheetError(Exception):
    """Error in flowsheet solving"""
    pass


@dataclass
class SimulationResult:
    """/mplete simulation results"""
    converged: bool
    iterations: int
    streams: dict[str, StreamState]
    units: dict[str, UnitResult]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mass_balance_error: float = 0.0
    energy_balance_error: float = 0.0
    thermo_scope_enthalpy_correction: float = 0.0
    thermo_scope_corrections: list[dict] = field(default_factory=list)
    recycle_info: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'converged': self.converged,
            'iterations': self.iterations,
            'streams': {k: v.to_dict() for k, v in self.streams.items()},
            'units': {k: v.to_dict() for k, v in self.units.items()},
            'errors': self.errors,
            'warnings': self.warnings,
            'mass_balance_error': self.mass_balance_error,
            'energy_balance_error': self.energy_balance_error,
            'thermo_scope_enthalpy_correction': (
                self.thermo_scope_enthalpy_correction
            ),
            'thermo_scope_corrections': list(self.thermo_scope_corrections),
            'recycle_info': self.recycle_info,
        }


@dataclass
class RecycleEvaluation:
    """Result of one tear-stream fixed-point evaluation."""
    success: bool
    x: object
    gx: object
    residual: object
    warnings: list[str] = field(default_factory=list)
    error: float = math.inf
    message: str = ""
    worst_variable: str = ""


class FlowsheetSolver:
    """
    Sequential modular flowsheet solver.
    
    Uses direct substitution with optional Wegstein acceleration
    for recycle convergence.
    """
    
    def __init__(self, pfd, thermo: Optional[IdealThermodynamics] = None,
                 thermo_packages: Optional[dict[str, IdealThermodynamics]] = None):
        """
        Initialize solver with a PFD object.
        
        Args:
            pfd: ProcessFlowDiagram from pfd_parser
            thermo: Thermodynamics instance (created if None)
        """
        self.pfd = pfd
        
        # Get component list
        self.components = [c.symbol for c in pfd.components]
        
        # Create thermodynamics if needed
        if thermo is None:
            self.thermo = IdealThermodynamics(self.components)
        else:
            self.thermo = thermo
        self.thermo_packages = dict(thermo_packages or {'global': self.thermo})
        self.thermo_packages.setdefault('global', self.thermo)
        
        # Build connectivity
        self._build_graph()
        self._build_thermo_scope_topology()
        
        # Create unit operation instances
        self._create_units()
        
        # Stream states
        self.streams: dict[str, StreamState] = {}
        
        # Unit results
        self.unit_results: dict[str, UnitResult] = {}
        self.progress_callback: Optional[Callable[[str], None]] = None
        self._recycle_failed_evaluations = 0
        self._recycle_last_failure = ""
        self._recycle_trace_tolerance = 1e-6
        self._recycle_component_scale_floor = 1e-3
        self._recycle_worst_variable = ""
        self._recycle_worst_error = 0.0
        self._auto_selected_tears: list[str] = []
        self._recycle_deferred_units: list[str] = []
        self._recycle_deferred_units_report: list[str] = []
        self._recycle_invariant_units: set[str] = set()
        self._recycle_invariant_units_ready: set[str] = set()
        self._recycle_invariant_units_report: list[str] = []
        self._recycle_blocks_report: list[dict] = []
        self._recycle_method_options: dict[str, float | int] = (
            resolved_recycle_options('WEGSTEIN', {})
        )
        self._thermo_scope_corrections: dict[str, dict] = {}

    def _emit_progress(self, message: str):
        """Send a progress message to the caller if progress reporting is enabled."""
        if self.progress_callback is not None:
            self.progress_callback(message)

    @staticmethod
    def _dedupe_warnings(warnings: list[str]) -> list[str]:
        deduped = []
        seen = set()
        for warning in warnings:
            text = str(warning).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    def _format_stream_state(self, state: StreamState) -> str:
        comps = ', '.join(
            f"{comp}:{state.composition.get(comp, 0.0):.5f}"
            for comp in self.components
        )
        return (
            f"T={state.T - 273.15:.2f} C, P={state.P:.4g} bar, "
            f"F={state.F:.6g} kmol/h, VF={state.vapor_fraction:.4f}, x=[{comps}]"
        )
    
    def _build_graph(self):
        """Build connectivity graph from PFD streams"""
        # Map: unit -> list of upstream units
        self.upstream = defaultdict(list)
        # Map: unit -> list of downstream units
        self.downstream = defaultdict(list)
        # Map: stream_id -> (source_unit, source_port, dest_unit, dest_port)
        self.stream_connections = {}
        # Map: unit -> list of inlet stream IDs
        self.unit_inlets = defaultdict(list)
        # Map: unit -> list of outlet stream IDs
        self.unit_outlets = defaultdict(list)
        
        for stream in self.pfd.streams:
            src = stream.source
            dst = stream.destination
            
            src_unit = src.unit_id if not src.is_feed else 'FEED'
            dst_unit = dst.unit_id if not dst.is_product else 'PRODUCT'
            
            self.stream_connections[stream.id] = (
                src_unit, src.port_id,
                dst_unit, dst.port_id
            )
            
            if src_unit != 'FEED':
                self.unit_outlets[src_unit].append(stream.id)
            if dst_unit != 'PRODUCT':
                self.unit_inlets[dst_unit].append(stream.id)
            
            if src_unit != 'FEED' and dst_unit != 'PRODUCT':
                self.upstream[dst_unit].append(src_unit)
                self.downstream[src_unit].append(dst_unit)

    @staticmethod
    def _declared_unit_thermo_scope(unit) -> str:
        values = [
            str(param.value).strip()
            for param in unit.params
            if str(param.name).strip().lower() == 'thermo_scope'
        ]
        return values[-1] if values else 'global'

    def _build_thermo_scope_topology(self) -> None:
        """Resolve unit package assignments and material-stream boundaries."""
        self.unit_thermo_scopes = {
            unit.id: self._declared_unit_thermo_scope(unit)
            for unit in self.pfd.units
        }
        for unit_id, scope in self.unit_thermo_scopes.items():
            if scope not in self.thermo_packages:
                raise FlowsheetError(
                    f"Unit '{unit_id}' references unavailable thermodynamic "
                    f"scope '{scope}'"
                )

        self.stream_thermo_scopes = {}
        self.thermo_scope_boundaries = {}
        for stream in self.pfd.streams:
            source_scope = (
                'global'
                if stream.source.is_feed
                else self.unit_thermo_scopes[stream.source.unit_id]
            )
            destination_scope = (
                'global'
                if stream.destination.is_product
                else self.unit_thermo_scopes[stream.destination.unit_id]
            )
            pair = (source_scope, destination_scope)
            self.stream_thermo_scopes[stream.id] = pair
            if source_scope != destination_scope:
                self.thermo_scope_boundaries[stream.id] = pair
    
    def _create_units(self):
        """Create unit operation instances"""
        self.units: dict[str, UnitOperation] = {}
        
        for unit in self.pfd.units:
            # Collect parameters
            params = {}
            for param in unit.params:
                if str(param.name).strip().lower() == 'thermo_scope':
                    continue
                value = param.value
                # Try to convert to number
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except (ValueError, TypeError):
                    pass
                
                # Handle temperature unit conversion
                param_lower = param.name.lower()
                if isinstance(value, (int, float)) and (
                    't_' in param_lower
                    or param_lower in {'t', 'temperature'}
                ):
                    unit_str = param.unit.upper() if param.unit else ''
                    if unit_str in ['C', '°C', 'CELSIUS']:
                        value = value + 273.15  # Convert to Kelvin
                    elif unit_str in ['F', '°F', 'FAHRENHEIT']:
                        value = (value - 32) * 5/9 + 273.15
                    # If already K or no unit, keep as is
                elif self._is_pressure_param(param.name):
                    value = self._convert_pressure_value(value, param.unit)
                
                params[param.name] = value
                if param.unit:
                    params[f"__unit__{param.name}"] = param.unit

            if unit.unit_type == 'Crystallizer':
                params['__connected_outlet_ports__'] = [
                    self.stream_connections[stream_id][1]
                    for stream_id in self.unit_outlets.get(unit.id, [])
                ]
            
            # Add reactions if present
            if unit.reactions:
                resolved_reactions = []
                for reaction in unit.reactions:
                    try:
                        resolved = self.pfd.resolve_reaction(reaction)
                    except ValueError as error:
                        raise FlowsheetError(
                            f"Cannot create unit '{unit.id}': {error}"
                        ) from error
                    resolved_reactions.append({
                        'equation': resolved.equation,
                        **resolved.parameters,
                    })
                params['reactions'] = resolved_reactions
            
            # Create unit
            try:
                self.units[unit.id] = create_unit(
                    unit.unit_type,
                    unit.id,
                    self.thermo_packages[self.unit_thermo_scopes[unit.id]],
                    params,
                )
            except UnitOperationError as e:
                raise FlowsheetError(f"Cannot create unit '{unit.id}': {e}")
    
    def _manual_tear_streams(self) -> list[str]:
        """Return PFD-specified tear streams, preserving order and validity."""
        requested = getattr(getattr(self.pfd, 'metadata', None), 'recycle_tear_streams', [])
        valid = []
        for stream_id in requested or []:
            if stream_id not in self.stream_connections:
                raise FlowsheetError(f"Manual tear stream '{stream_id}' does not exist")
            src, _, dst, _ = self.stream_connections[stream_id]
            if src in ('FEED', 'PRODUCT') or dst in ('FEED', 'PRODUCT'):
                raise FlowsheetError(
                    f"Manual tear stream '{stream_id}' must connect two unit operations"
                )
            if stream_id not in valid:
                valid.append(stream_id)
        return valid

    def _order_for_tears(self, tear_streams: list[str]) -> list[str]:
        """Topologically sort units while ignoring the selected tear streams."""
        in_degree = defaultdict(int)
        for unit in self.units:
            for upstream in self.upstream.get(unit, []):
                torn = False
                for stream_id in self.unit_inlets.get(unit, []):
                    if stream_id in tear_streams:
                        conn = self.stream_connections[stream_id]
                        if conn[0] == upstream:
                            torn = True
                            break
                if not torn and upstream in self.units:
                    in_degree[unit] += 1

        queue = [u for u in self.units if in_degree[u] == 0]
        ordered = []

        while queue:
            unit = queue.pop(0)
            ordered.append(unit)

            for downstream in self.downstream.get(unit, []):
                if downstream in self.units:
                    torn = False
                    for stream_id in self.unit_inlets.get(downstream, []):
                        if stream_id in tear_streams:
                            conn = self.stream_connections[stream_id]
                            if conn[0] == unit:
                                torn = True
                                break
                    if not torn:
                        in_degree[downstream] -= 1
                        if in_degree[downstream] == 0:
                            queue.append(downstream)

        return ordered

    def _strongly_connected_components(self, tear_streams: list[str]) -> list[list[str]]:
        """Tarjan SCCs for the unit graph with selected tear streams removed."""
        tear_set = set(tear_streams)
        index = 0
        stack = []
        on_stack = set()
        indices = {}
        lowlink = {}
        components = []

        def successors(unit: str) -> list[str]:
            result = []
            for stream_id in self.unit_outlets.get(unit, []):
                if stream_id in tear_set:
                    continue
                src, _, dst, _ = self.stream_connections[stream_id]
                if src == unit and dst in self.units:
                    result.append(dst)
            return result

        def strongconnect(unit: str):
            nonlocal index
            indices[unit] = index
            lowlink[unit] = index
            index += 1
            stack.append(unit)
            on_stack.add(unit)

            for downstream in successors(unit):
                if downstream not in indices:
                    strongconnect(downstream)
                    lowlink[unit] = min(lowlink[unit], lowlink[downstream])
                elif downstream in on_stack:
                    lowlink[unit] = min(lowlink[unit], indices[downstream])

            if lowlink[unit] == indices[unit]:
                scc = []
                while True:
                    member = stack.pop()
                    on_stack.remove(member)
                    scc.append(member)
                    if member == unit:
                        break
                components.append(scc)

        for unit in self.units:
            if unit not in indices:
                strongconnect(unit)

        cyclic = []
        for scc in components:
            if len(scc) > 1:
                cyclic.append(scc)
            else:
                unit = scc[0]
                if any(
                    self.stream_connections[stream_id][2] == unit
                    for stream_id in self.unit_outlets.get(unit, [])
                    if stream_id not in tear_set
                ):
                    cyclic.append(scc)
        return cyclic

    def _unit_category(self, unit_id: str) -> str:
        unit = self.units.get(unit_id)
        if unit is None:
            return ''
        return unit.__class__.__name__.lower()

    def _tear_candidate_score(self, stream_id: str) -> tuple[float, str]:
        """Lower score means a more numerically friendly tear location."""
        src, _, dst, _ = self.stream_connections[stream_id]
        src_type = self._unit_category(src)
        dst_type = self._unit_category(dst)
        score = 100.0

        # Stable state-setting blocks make good tear sources because T/P/H are
        # less likely to jump between iterations.
        if any(name in src_type for name in ('heater', 'cooler', 'pump', 'compressor')):
            score -= 45.0
        if any(name in src_type for name in ('valve', 'splitter', 'flash')):
            score -= 30.0
        if 'mixer' in src_type:
            score -= 20.0
        if 'splitter' in src_type and 'mixer' in dst_type:
            score -= 45.0
        if 'reactor' in src_type and 'heatexchanger' in dst_type:
            score -= 120.0

        # Tearing into strongly nonlinear units is often better than tearing
        # their products; it gives the unit a stable assumed feed each pass.
        if any(name in dst_type for name in ('distillation', 'reactor', 'extractor')):
            score -= 20.0
        if any(name in dst_type for name in ('pump', 'compressor')):
            score -= 15.0
        if 'heatexchanger' in dst_type:
            score -= 30.0
        if 'mixer' in dst_type:
            score -= 35.0

        # Avoid tearing streams directly leaving highly nonlinear units unless
        # no better stream is available.
        if any(name in src_type for name in ('distillation', 'reactor', 'extractor')):
            score += 25.0
        if stream_id.lower().startswith('recycle') or 'recycle' in stream_id.lower():
            score -= 90.0

        return score, stream_id

    def _choose_tear_for_scc(self, scc: list[str],
                             existing_tears: list[str]) -> Optional[str]:
        members = set(scc)
        candidates = []
        for src in scc:
            for stream_id in self.unit_outlets.get(src, []):
                if stream_id in existing_tears:
                    continue
                conn = self.stream_connections[stream_id]
                if conn[2] in members:
                    candidates.append(stream_id)
        if not candidates:
            return None
        return min(candidates, key=self._tear_candidate_score)

    def _topological_sort(self) -> tuple[list[str], list[str]]:
        """
        Sort units in calculation order and choose one tear stream per recycle SCC.

        Manual `TEAR_STREAMS:` metadata is honored first. Any remaining cycles
        are torn automatically using a simple Aspen-like preference for streams
        leaving state-setting blocks and entering nonlinear blocks.
        """
        tear_streams = self._manual_tear_streams()
        self._auto_selected_tears = []

        while True:
            order = self._order_for_tears(tear_streams)
            if len(order) == len(self.units):
                return order, tear_streams

            cyclic_sccs = self._strongly_connected_components(tear_streams)
            added = False
            for scc in cyclic_sccs:
                chosen = self._choose_tear_for_scc(scc, tear_streams)
                if chosen is not None:
                    tear_streams.append(chosen)
                    self._auto_selected_tears.append(chosen)
                    added = True
            if not added:
                unresolved = [unit for unit in self.units if unit not in order]
                raise FlowsheetError(
                    "Could not find a tear stream that breaks recycle loop(s): "
                    + ', '.join(unresolved)
                )
    
    def _state_from_stream_spec(self, stream, label: str,
                                allow_empty: bool = False) -> Optional[StreamState]:
        """Build a StreamState from PFD stream specs."""
        if allow_empty and not stream.properties and stream.composition is None:
            return None

        T = None
        P = None
        F = None
        mass_flow = None
        vapor_fraction = None
        composition = {}

        for prop in stream.properties:
            if prop.name.upper() == 'T':
                T = temperature_to_kelvin(
                    float(prop.value),
                    prop.unit,
                    infer_unitless_celsius_below=200.0,
                )
            elif prop.name.upper() == 'P':
                P = self._convert_pressure_value(float(prop.value), prop.unit)
            elif prop.name.upper() in ('F', 'FLOW', 'MOLAR_FLOW'):
                value = float(prop.value)
                unit = (prop.unit or '').lower()
                if any(token in unit for token in ('kg', 'lb', 'mass')):
                    mass_flow = mass_flow_to_kg_per_hour(value, unit)
                else:
                    F = molar_flow_to_kmol_per_hour(value, unit)
            elif prop.name.upper() in ('F_MASS', 'MASS_FLOW', 'FLOW_MASS'):
                value = float(prop.value)
                unit = (prop.unit or '').lower()
                mass_flow = mass_flow_to_kg_per_hour(value, unit)
            elif prop.name.upper() in ('VAP_FRAC', 'VAPOR_FRAC', 'VAPOR_FRACTION', 'VF'):
                vapor_fraction = float(prop.value)

        if stream.composition:
            composition = dict(stream.composition.fractions)

        if P is None or (F is None and mass_flow is None) or not composition:
            raise FlowsheetError(
                f"{label} must specify P, F, and composition"
            )
        if T is None and vapor_fraction is None:
            raise FlowsheetError(
                f"{label} must specify T or vapor_fraction"
            )
        if vapor_fraction is not None and not 0.0 <= vapor_fraction <= 1.0:
            raise FlowsheetError(
                f"{label} vapor_fraction must be between 0 and 1"
            )

        total = sum(composition.values())
        if total <= 0.0:
            raise FlowsheetError(f"{label} composition must be positive")
        composition = {k: v / total for k, v in composition.items()}
        if stream.composition and getattr(stream.composition, 'basis', 'mole') == 'mass':
            mole_amounts = {}
            for comp, fraction in composition.items():
                props = self.thermo.props.get(comp)
                MW = getattr(props, 'MW', None) if props else None
                if MW is None or MW <= 0.0:
                    raise FlowsheetError(
                        f"{label} cannot convert mass fraction for component "
                        f"'{comp}' without molecular weight"
                    )
                mole_amounts[comp] = fraction / MW
            mole_total = sum(mole_amounts.values())
            composition = {comp: amount / mole_total for comp, amount in mole_amounts.items()}
        if F is None:
            mixture_MW = self.thermo.mixture_MW(composition)
            if mixture_MW <= 0.0:
                raise FlowsheetError(
                    f"{label} cannot convert mass flow without mixture MW"
                )
            F = mass_flow / mixture_MW

        if vapor_fraction is not None:
            direct_pq = getattr(self.thermo, 'calculate_state_PQ', None)
            if direct_pq is not None:
                try:
                    return direct_pq(P, vapor_fraction, F, composition)
                except (NotImplementedError, AttributeError):
                    pass
            if T is None:
                raise FlowsheetError(
                    f"{label} specified vapor_fraction but the thermodynamic "
                    "method does not support direct P,VF states"
                )
            state = self.thermo.calculate_state(T, P, F, composition)
            state.vapor_fraction = vapor_fraction
            return state

        return self.thermo.calculate_state(T, P, F, composition)

    def _initialize_streams(self):
        """Initialize stream states from PFD specifications"""
        for stream in self.pfd.streams:
            if stream.source.is_feed:
                context = getattr(self.thermo, 'quality_context', None)
                if context is None:
                    self.streams[stream.id] = self._state_from_stream_spec(
                        stream,
                        f"Feed stream '{stream.id}'",
                    )
                else:
                    with context(
                        kind='stream',
                        stream_id=stream.id,
                        phase='feed_spec',
                        affects_result=True,
                    ):
                        self.streams[stream.id] = self._state_from_stream_spec(
                            stream,
                            f"Feed stream '{stream.id}'",
                        )

    @staticmethod
    def _is_pressure_param(name: str) -> bool:
        """Return True for unit-operation parameters whose internal unit is bar."""
        lower = name.lower()
        pressure_names = {
            'p', 'p_out', 'p_top', 'p_bottom', 'p_condenser', 'p_reboiler',
            'p_drop', 'p_drop_hot', 'p_drop_cold', 'p_drop_per_stage',
            'stage_pressures', 'pressure_profile',
        }
        return lower in pressure_names or 'pressure' in lower

    @classmethod
    def _convert_pressure_value(cls, value, unit: Optional[str]):
        """Convert supported pressure units to the simulator's internal bar basis."""
        factor = cls._pressure_unit_factor(unit)
        if factor == 1.0:
            return value
        if isinstance(value, str) and ',' in value:
            converted = []
            for part in value.split(','):
                converted.append(str(float(part.strip()) * factor))
            return ', '.join(converted)
        if isinstance(value, str):
            try:
                return float(value) * factor
            except ValueError:
                return value
        return value * factor

    @staticmethod
    def _pressure_unit_factor(unit: Optional[str]) -> float:
        return pressure_unit_factor(unit)
    
    def _initialize_tear_streams(self, tear_streams: list[str]):
        """Initialize tear streams with estimates"""
        for stream_id in tear_streams:
            stream_spec = self.pfd.get_stream(stream_id)
            if stream_spec is not None:
                context = getattr(self.thermo, 'quality_context', None)
                if context is None:
                    initial_state = self._state_from_stream_spec(
                        stream_spec,
                        f"Tear stream '{stream_id}' initial guess",
                        allow_empty=True,
                    )
                else:
                    with context(
                        kind='stream',
                        stream_id=stream_id,
                        phase='recycle_initial_guess',
                        affects_result=False,
                    ):
                        initial_state = self._state_from_stream_spec(
                            stream_spec,
                            f"Tear stream '{stream_id}' initial guess",
                            allow_empty=True,
                        )
                if initial_state is not None:
                    self.streams[stream_id] = initial_state
                    continue
            if stream_id not in self.streams:
                # Find a feed stream to use as initial guess
                for other_id, state in self.streams.items():
                    if state is not None:
                        # Use feed stream as initial guess
                        self.streams[stream_id] = state.copy()
                        self.streams[stream_id].F *= 0.5  # Assume 50% recycle
                        self.streams[stream_id].solid_component_flows = {
                            component: 0.5 * flow
                            for component, flow in state.solid_component_flows.items()
                        }
                        break
                else:
                    # No feed stream found - create default
                    composition = {c: 1.0/len(self.components) for c in self.components}
                    context = getattr(self.thermo, 'quality_context', None)
                    if context is None:
                        self.streams[stream_id] = self.thermo.calculate_state(
                            300, 1.0, 100, composition
                        )
                    else:
                        with context(
                            kind='stream',
                            stream_id=stream_id,
                            phase='recycle_default_initial_guess',
                            affects_result=False,
                        ):
                            self.streams[stream_id] = self.thermo.calculate_state(
                                300, 1.0, 100, composition
                            )
    
    @staticmethod
    def _scope_phase_constraint(state: StreamState) -> Optional[str]:
        if (
            state.phase_stability != 'explicit_phase_constraint'
            and not str(state.phase_status).startswith('forced_')
        ):
            return None
        fractions = state.phase_fractions()
        fluid_total = (
            fractions['vapor']
            + fractions['liquid1']
            + fractions['liquid2']
        )
        if fluid_total <= 1.0e-15:
            return None
        vapor = fractions['vapor'] / fluid_total
        liquid2 = fractions['liquid2'] / fluid_total
        if vapor >= 1.0 - 1.0e-12 and liquid2 <= 1.0e-12:
            return 'vapor'
        if vapor <= 1.0e-12 and liquid2 <= 1.0e-12:
            return 'liquid'
        return None

    def _state_in_thermo_scope(
        self,
        state: StreamState,
        scope: str,
    ) -> StreamState:
        """Re-evaluate thermodynamic state fields without transport properties."""
        thermo = self.thermo_packages[scope]
        phase = self._scope_phase_constraint(state)
        include = ['H', 'S', 'Cp', 'rho', 'mu']
        converted = thermo.calculate_state(
            state.T,
            state.P,
            state.F,
            dict(state.composition),
            phase=phase,
            flash=phase is None,
            include=include,
        )
        converted.thermo_scope = scope
        return converted

    def _transition_stream_state(
        self,
        stream_id: str,
        state: StreamState,
        destination_scope: str,
    ) -> StreamState:
        source_scope = getattr(state, 'thermo_scope', None) or 'global'
        if source_scope == destination_scope:
            # A material stream can already carry the destination package
            # because its source-unit outlet was converted at a declared
            # scope boundary.  Downstream consumption of that converted view
            # must not erase the boundary's energy-reconciliation record.
            # Only discard records for identifiers that are not real topology
            # boundaries (for example direct diagnostic probes).
            if stream_id not in self.thermo_scope_boundaries:
                self._thermo_scope_corrections.pop(stream_id, None)
            return state
        if source_scope not in self.thermo_packages:
            raise FlowsheetError(
                f"Stream '{stream_id}' carries unknown thermodynamic scope "
                f"'{source_scope}'"
            )

        source_view = state
        if (
            source_view.H is None
            or source_view.S is None
            or source_view.Cp is None
            or source_view.rho is None
            or source_view.mu is None
        ):
            source_view = self._state_in_thermo_scope(state, source_scope)
        destination_view = self._state_in_thermo_scope(
            state,
            destination_scope,
        )
        if source_view.H is None or destination_view.H is None:
            raise FlowsheetError(
                f"Thermodynamic scope transition for stream '{stream_id}' "
                "could not calculate enthalpy on both sides"
            )

        correction = state.F * (destination_view.H - source_view.H)
        self._thermo_scope_corrections[stream_id] = {
            'stream_id': stream_id,
            'source_scope': source_scope,
            'destination_scope': destination_scope,
            'flow_kmol_per_h': state.F,
            'source_H_kJ_per_kmol': source_view.H,
            'destination_H_kJ_per_kmol': destination_view.H,
            'enthalpy_flow_correction_kJ_per_h': correction,
            'source_S_kJ_per_kmol_K': source_view.S,
            'destination_S_kJ_per_kmol_K': destination_view.S,
            'source_Cp_kJ_per_kmol_K': source_view.Cp,
            'destination_Cp_kJ_per_kmol_K': destination_view.Cp,
            'source_density_kmol_per_m3': source_view.rho,
            'destination_density_kmol_per_m3': destination_view.rho,
            'source_viscosity_Pa_s': source_view.mu,
            'destination_viscosity_Pa_s': destination_view.mu,
        }
        return destination_view

    def _calculate_unit(self, unit_id: str, solve_context: Optional[dict] = None) -> UnitResult:
        """Calculate a single unit operation"""
        unit = self.units[unit_id]
        start = time.perf_counter()
        self._emit_progress(f"unit_start {unit_id} ({unit.__class__.__name__})")
        
        # Gather inlet streams
        inlets = {}
        for stream_id in self.unit_inlets[unit_id]:
            conn = self.stream_connections[stream_id]
            port_id = conn[3]  # Destination port
            
            if stream_id not in self.streams:
                raise FlowsheetError(
                    f"Stream '{stream_id}' not calculated before unit '{unit_id}'"
                )
            
            destination_scope = self.unit_thermo_scopes[unit_id]
            inlets[port_id] = self._transition_stream_state(
                stream_id,
                self.streams[stream_id],
                destination_scope,
            )
        
        # Solve unit
        context = dict(solve_context or {})
        context.setdefault('unit_id', unit_id)
        affects_result = True
        if context.get('recycle_evaluation') is not None:
            affects_result = context.get('recycle_final_pass') is True
        unit.solve_context = context
        try:
            quality_context = getattr(unit.thermo, 'quality_context', None)
            if quality_context is None:
                result = unit.solve(inlets)
            else:
                with quality_context(
                    kind='unit',
                    unit_id=unit_id,
                    unit_type=unit.__class__.__name__,
                    phase='solve',
                    affects_result=affects_result,
                ):
                    result = unit.solve(inlets)
        finally:
            unit.solve_context = {}
        
        # Store outlet streams
        for stream_id in self.unit_outlets[unit_id]:
            conn = self.stream_connections[stream_id]
            port_id = conn[1]  # Source port
            
            # Find matching outlet
            if port_id in result.outlet_streams:
                outlet_state = result.outlet_streams[port_id]
            elif len(result.outlet_streams) == 1:
                # Single outlet - use it regardless of port name
                outlet_state = list(result.outlet_streams.values())[0]
            else:
                # Try to match by common names
                for out_port, state in result.outlet_streams.items():
                    if out_port.lower() in port_id.lower() or port_id.lower() in out_port.lower():
                        outlet_state = state
                        break
                else:
                    # Use first available
                    outlet_state = list(result.outlet_streams.values())[0]
            source_scope = self.unit_thermo_scopes[unit_id]
            outlet_state.thermo_scope = source_scope
            destination_scope = self.stream_thermo_scopes[stream_id][1]
            self.streams[stream_id] = self._transition_stream_state(
                stream_id,
                outlet_state,
                destination_scope,
            ) if destination_scope == 'global' and source_scope != 'global' else outlet_state
        
        elapsed = time.perf_counter() - start
        outlet_summary = []
        for stream_id in self.unit_outlets[unit_id]:
            if stream_id in self.streams:
                outlet_summary.append(f"{stream_id}: {self._format_stream_state(self.streams[stream_id])}")
        suffix = '; '.join(outlet_summary)
        if suffix:
            suffix = f" | {suffix}"
        self._emit_progress(
            f"unit_done {unit_id} elapsed={elapsed:.3f}s "
            f"Q={result.heat_duty / 3600.0:.6g} kW W={result.work / 3600.0:.6g} kW{suffix}"
        )
        return result

    def _expensive_diagnostics_due(self, evaluation_number: int) -> bool:
        """Sparse cadence for diagnostics that do not affect recycle residuals."""
        if evaluation_number in (1, 2, 5):
            return True
        if evaluation_number <= 30:
            return evaluation_number % 10 == 0
        return evaluation_number % 25 == 0

    def _recycle_solve_context(self, evaluation_number: int, final_pass: bool = False) -> dict:
        diagnostics = True if final_pass else self._expensive_diagnostics_due(evaluation_number)
        return {
            'recycle_evaluation': evaluation_number,
            'recycle_final_pass': final_pass,
            'expensive_diagnostics': diagnostics,
        }

    def _units_required_for_recycle_residual(
        self,
        tear_streams: list[str],
        blocked_tear_streams: Optional[list[str]] = None,
    ) -> set[str]:
        """Units that can affect the current tear stream residuals."""
        tear_set = set(blocked_tear_streams if blocked_tear_streams is not None else tear_streams)
        required = set()
        stack = []
        for stream_id in tear_streams:
            src, _, _, _ = self.stream_connections[stream_id]
            if src in self.units and src not in required:
                required.add(src)
                stack.append(src)

        while stack:
            unit_id = stack.pop()
            for stream_id in self.unit_inlets.get(unit_id, []):
                if stream_id in tear_set:
                    continue
                src, _, _, _ = self.stream_connections[stream_id]
                if src in self.units and src not in required:
                    required.add(src)
                    stack.append(src)
        return required

    def _deferred_recycle_units(
        self,
        calc_order: list[str],
        tear_streams: list[str],
        blocked_tear_streams: Optional[list[str]] = None,
    ) -> list[str]:
        if not tear_streams:
            return []
        required = self._units_required_for_recycle_residual(
            tear_streams, blocked_tear_streams
        )
        return [unit_id for unit_id in calc_order if unit_id not in required]

    def _recycle_blocks(self, calc_order: list[str],
                        tear_streams: list[str]) -> list[dict]:
        if not tear_streams:
            return []

        original_sccs = [set(scc) for scc in self._strongly_connected_components([])]
        assigned = set()
        blocks = []

        for scc in original_sccs:
            block_tears = []
            for stream_id in tear_streams:
                src, _, dst, _ = self.stream_connections[stream_id]
                if src in scc and dst in scc:
                    block_tears.append(stream_id)
            if block_tears:
                assigned.update(block_tears)
                blocks.append(block_tears)

        for stream_id in tear_streams:
            if stream_id not in assigned:
                blocks.append([stream_id])

        order_index = {unit_id: index for index, unit_id in enumerate(calc_order)}
        result = []
        for block_tears in blocks:
            required = self._units_required_for_recycle_residual(
                block_tears, tear_streams
            )
            tear_dependent = set()
            stack = []
            for stream_id in block_tears:
                _, _, destination, _ = self.stream_connections[stream_id]
                if destination in required and destination not in tear_dependent:
                    tear_dependent.add(destination)
                    stack.append(destination)
            while stack:
                unit_id = stack.pop()
                for downstream_id in self.downstream.get(unit_id, []):
                    if (
                        downstream_id in required
                        and downstream_id not in tear_dependent
                    ):
                        tear_dependent.add(downstream_id)
                        stack.append(downstream_id)

            repeated = required & tear_dependent
            invariant = required - repeated
            deferred = [unit_id for unit_id in calc_order if unit_id not in required]
            first_required = min(
                (order_index[unit_id] for unit_id in required if unit_id in order_index),
                default=len(calc_order),
            )
            result.append({
                'tear_streams': block_tears,
                'required_units': required,
                'repeated_units': repeated,
                'invariant_units': invariant,
                'deferred_units': deferred,
                'order_index': first_required,
            })

        result.sort(key=lambda block: block['order_index'])
        return result
    
    def _check_convergence(self, tear_streams: list[str],
                           old_states: dict[str, StreamState],
                           tolerance: float = 1e-4) -> tuple[bool, float]:
        """Check convergence in Aspen-like tear variables."""
        import numpy as np

        if any(stream_id not in old_states or stream_id not in self.streams for stream_id in tear_streams):
            return False, 1.0
        old_vector = np.array(self._tear_states_to_vector(tear_streams, old_states), dtype=float)
        new_vector = np.array(
            self._tear_states_to_vector(
                tear_streams,
                {stream_id: self.streams[stream_id] for stream_id in tear_streams},
            ),
            dtype=float,
        )
        _, scale = self._make_recycle_scale(tear_streams, old_states)
        residual = (new_vector - old_vector) / scale
        residual = self._mask_trace_residuals(tear_streams, old_vector, new_vector, scale, residual)
        max_error, worst = self._max_recycle_error(tear_streams, residual)
        self._recycle_worst_error = max_error
        self._recycle_worst_variable = worst
        return max_error < tolerance, max_error
    
    def _state_to_vector(self, state: StreamState) -> list[float]:
        """Aspen-like tear variables: total flow, component flows, pressure, enthalpy."""
        H = state.H
        if H is None:
            thermo = self.thermo_packages.get(
                getattr(state, 'thermo_scope', 'global'),
                self.thermo,
            )
            H = thermo.mixture_enthalpy(
                state.composition,
                state.T,
                state.vapor_fraction,
                state.x,
                state.y,
                state.P,
            )
        return [
            state.F,
            *[state.F * state.composition.get(comp, 0.0) for comp in self.components],
            state.P,
            H,
        ]

    def _variable_names(self, tear_streams: list[str]) -> list[str]:
        names = []
        for stream_id in tear_streams:
            names.append(f"{stream_id}.F")
            names.extend(f"{stream_id}.n_{comp}" for comp in self.components)
            names.append(f"{stream_id}.P")
            names.append(f"{stream_id}.H")
        return names

    def _enthalpy_state_from_PH(self, P: float, F: float,
                                composition: dict[str, float],
                                H_target: float,
                                fallback: StreamState,
                                thermo,
                                thermo_scope: str) -> StreamState:
        """Build a stream state from P, component composition, and molar enthalpy."""
        from scipy.optimize import brentq

        def candidate_state(T: float, include=None) -> StreamState:
            state = thermo.calculate_state(
                T, P, F, composition,
                include=include,
            )
            state.thermo_scope = thermo_scope
            return state

        def residual(T: float) -> float:
            state = candidate_state(T, include=('H',))
            if state.H is None:
                raise ThermodynamicsError("State enthalpy was not calculated")
            return state.H - H_target

        candidate_temperatures = [
            fallback.T - 250.0,
            fallback.T - 100.0,
            fallback.T - 25.0,
            fallback.T,
            fallback.T + 25.0,
            fallback.T + 100.0,
            fallback.T + 250.0,
            80.0,
            120.0,
            180.0,
            250.0,
            298.15,
            350.0,
            500.0,
            800.0,
            1200.0,
            1800.0,
        ]
        for comp, z_i in composition.items():
            if z_i <= 1e-8:
                continue
            props = getattr(thermo, 'props', {}).get(comp)
            if props:
                for value in (props.Tb, props.Tc):
                    if value:
                        candidate_temperatures.extend([0.75 * value, value, 1.15 * value])

        grid = sorted(set(max(50.0, min(2500.0, float(T))) for T in candidate_temperatures))
        values = []
        for T in grid:
            try:
                values.append((T, residual(T)))
            except Exception:
                continue

        for (T1, f1), (T2, f2) in zip(values, values[1:]):
            if abs(f1) < 1e-7:
                return candidate_state(T1)
            if f1 * f2 < 0:
                T = brentq(residual, T1, T2, xtol=1e-7, rtol=1e-9, maxiter=100)
                return candidate_state(T)

        if values:
            T = min(values, key=lambda item: abs(item[1]))[0]
            return candidate_state(T)
        state = thermo.calculate_state(fallback.T, P, F, composition)
        state.thermo_scope = thermo_scope
        return state
    
    def _vector_to_state(self, vector: list[float],
                         fallback: StreamState,
                         thermo_scope: str = 'global') -> StreamState:
        F = max(0.0, float(vector[0]))
        component_flows = {
            comp: max(0.0, float(vector[1 + i]))
            for i, comp in enumerate(self.components)
        }
        flow_total = sum(component_flows.values())
        if flow_total > 1e-14:
            composition = {comp: value / flow_total for comp, value in component_flows.items()}
        else:
            composition = dict(fallback.composition)
        if F <= 1e-14 and flow_total > 0.0:
            F = flow_total

        P = max(1e-6, float(vector[1 + len(self.components)]))
        H = float(vector[2 + len(self.components)])
        thermo = self.thermo_packages[thermo_scope]
        try:
            return self._enthalpy_state_from_PH(
                P,
                F,
                composition,
                H,
                fallback,
                thermo,
                thermo_scope,
            )
        except Exception:
            state = thermo.calculate_state(fallback.T, P, F, composition)
            state.thermo_scope = thermo_scope
            return state

    def _tear_states_to_vector(self, tear_streams: list[str],
                               states: dict[str, StreamState]) -> list[float]:
        vector = []
        for stream_id in tear_streams:
            vector.extend(self._state_to_vector(states[stream_id]))
        return vector
    
    def _apply_tear_vector(self, tear_streams: list[str], vector: list[float],
                           fallback_states: dict[str, StreamState]):
        width = 3 + len(self.components)
        for i, stream_id in enumerate(tear_streams):
            start = i * width
            end = start + width
            fallback = fallback_states.get(stream_id, self.streams[stream_id])
            source_scope = self.stream_thermo_scopes[stream_id][0]
            thermo = self.thermo_packages[source_scope]
            context = getattr(thermo, 'quality_context', None)
            if context is None:
                state = self._vector_to_state(
                    vector[start:end],
                    fallback,
                    source_scope,
                )
            else:
                with context(
                    kind='stream',
                    stream_id=stream_id,
                    phase='recycle_tear_vector_decode',
                    affects_result=False,
                ):
                    state = self._vector_to_state(
                        vector[start:end],
                        fallback,
                        source_scope,
                    )
            self.streams[stream_id] = state
    
    def _make_recycle_scale(self, tear_streams: list[str],
                            states: dict[str, StreamState]):
        import numpy as np

        x0 = np.array(
            self._tear_states_to_vector(tear_streams, states),
            dtype=float,
        )
        scale = np.maximum(np.abs(x0), 1.0)
        width = 3 + len(self.components)
        for stream_index, stream_id in enumerate(tear_streams):
            base = stream_index * width
            state = states[stream_id]
            flow_scale = max(abs(state.F), 1.0)
            scale[base] = flow_scale
            for comp_index, _comp in enumerate(self.components):
                idx = base + 1 + comp_index
                scale[idx] = max(
                    abs(x0[idx]),
                    self._recycle_component_scale_floor * flow_scale,
                    self._recycle_trace_tolerance * flow_scale,
                    1e-12,
                )
            scale[base + 1 + len(self.components)] = max(abs(x0[base + 1 + len(self.components)]), 1.0)
            scale[base + 2 + len(self.components)] = max(abs(x0[base + 2 + len(self.components)]), 1000.0)
        return x0, scale

    def _mask_trace_residuals(self, tear_streams: list[str], old_vector,
                              new_vector, scale, residual):
        """Ignore components whose molar flows are trace on both sides."""
        import numpy as np

        residual = np.asarray(residual, dtype=float).copy()
        width = 3 + len(self.components)
        for stream_index, _stream_id in enumerate(tear_streams):
            base = stream_index * width
            flow_scale = max(scale[base], abs(old_vector[base]), abs(new_vector[base]), 1.0)
            trace_flow = self._recycle_trace_tolerance * flow_scale
            for comp_index, _comp in enumerate(self.components):
                idx = base + 1 + comp_index
                if abs(old_vector[idx]) < trace_flow and abs(new_vector[idx]) < trace_flow:
                    residual[idx] = 0.0
        return residual

    def _max_recycle_error(self, tear_streams: list[str], residual) -> tuple[float, str]:
        import numpy as np

        if len(residual) == 0:
            return 0.0, ""
        abs_residual = np.abs(residual)
        idx = int(np.nanargmax(abs_residual))
        names = self._variable_names(tear_streams)
        return float(abs_residual[idx]), names[idx] if idx < len(names) else f"var_{idx}"

    def _limit_recycle_step(self, x, candidate, scale, tear_count: int):
        """Clip an accelerated recycle step to physically plausible increments."""
        import numpy as np

        x = np.asarray(x, dtype=float)
        candidate = np.asarray(candidate, dtype=float)
        scale = np.asarray(scale, dtype=float)
        limited = candidate.copy()
        width = 3 + len(self.components)

        # These are step-size guards, not model simplifications. They prevent
        # accelerators from taking a tear stream into impossible pump/flash
        # states after one secant extrapolation.
        max_physical_step = {
            width - 1: 50000.0,  # H [kJ/kmol]
        }
        max_relative_step = {
            0: 0.6,                  # Total flow
            width - 2: 0.6,          # Pressure
        }
        max_component_relative_step = 0.8

        for stream_index in range(tear_count):
            base = stream_index * width
            for offset in range(width):
                idx = base + offset
                delta = limited[idx] - x[idx]

                if offset in max_physical_step and max_physical_step[offset] is not None:
                    max_delta = max_physical_step[offset] / scale[idx]
                elif offset in max_relative_step:
                    max_delta = max_relative_step[offset] * max(abs(x[idx]), 1e-6)
                elif 1 <= offset <= len(self.components):
                    max_delta = max(
                        1.0,
                        max_component_relative_step * max(abs(x[idx]), abs(limited[idx]), 1e-6),
                    )
                else:
                    max_delta = 1.0

                if abs(delta) > max_delta:
                    limited[idx] = x[idx] + math.copysign(max_delta, delta)

        # Hard physical bounds in scaled variables.
        for stream_index in range(tear_count):
            base = stream_index * width
            limited[base] = max(limited[base], 0.0)
            for idx in range(base + 1, base + 1 + len(self.components)):
                limited[idx] = max(limited[idx], 0.0)
            p_idx = base + 1 + len(self.components)
            limited[p_idx] = max(limited[p_idx], 1e-6 / scale[p_idx])
        return limited

    def _run_recycle_evaluation(self, calc_order: list[str],
                                tear_streams: list[str],
                                scaled_vector,
                                scale,
                                fallback_states: dict[str, StreamState],
                                evaluation_number: int) -> RecycleEvaluation:
        import numpy as np

        tear_vector = (np.asarray(scaled_vector, dtype=float) * scale).tolist()
        self._apply_tear_vector(tear_streams, tear_vector, fallback_states)
        applied_vector = np.array(
            self._tear_states_to_vector(
                tear_streams,
                {stream_id: self.streams[stream_id] for stream_id in tear_streams},
            ),
            dtype=float,
        )
        applied_scaled = applied_vector / scale

        tear_summary = '; '.join(
            f"{stream_id}: {self._format_stream_state(self.streams[stream_id])}"
            for stream_id in tear_streams
        )
        self._emit_progress(f"recycle_eval_start {evaluation_number} | {tear_summary}")

        iteration_warnings = []
        failed_unit_id = None
        try:
            solve_context = self._recycle_solve_context(evaluation_number)
            for unit_id in calc_order:
                if unit_id in self._recycle_deferred_units:
                    self._emit_progress(
                        f"unit_skipped {unit_id} deferred_until_recycle_final_pass"
                    )
                    continue
                if (
                    unit_id in self._recycle_invariant_units
                    and unit_id in self._recycle_invariant_units_ready
                ):
                    self._emit_progress(
                        f"unit_cached {unit_id} recycle_invariant"
                    )
                    continue
                failed_unit_id = unit_id
                result = self._calculate_unit(unit_id, solve_context=solve_context)
                self.unit_results[unit_id] = result
                if unit_id in self._recycle_invariant_units:
                    self._recycle_invariant_units_ready.add(unit_id)
                iteration_warnings.extend(result.warnings)
        except (UnitOperationError, ThermodynamicsError, FlowsheetError) as e:
            failed_unit = self.units.get(failed_unit_id) if failed_unit_id else None
            if failed_unit is not None and hasattr(failed_unit, '_last_recycle_profile'):
                failed_unit._recycle_warm_start_skip_count = 1
            self._emit_progress(f"recycle_eval_failed {evaluation_number}: {e}")
            self._recycle_failed_evaluations += 1
            self._recycle_last_failure = str(e)
            nan = np.full_like(applied_scaled, np.nan, dtype=float)
            return RecycleEvaluation(
                success=False,
                x=applied_scaled,
                gx=nan,
                residual=nan,
                warnings=iteration_warnings,
                message=str(e),
            )

        raw_states = {
            stream_id: self.streams[stream_id]
            for stream_id in tear_streams
        }
        raw_vector = np.array(
            self._tear_states_to_vector(tear_streams, raw_states),
            dtype=float,
        )
        gx = raw_vector / scale
        residual = gx - applied_scaled
        residual = self._mask_trace_residuals(
            tear_streams,
            applied_vector,
            raw_vector,
            scale,
            residual,
        )
        error, worst_variable = self._max_recycle_error(tear_streams, residual)
        self._recycle_worst_error = error
        self._recycle_worst_variable = worst_variable
        raw_summary = '; '.join(
            f"{stream_id}: {self._format_stream_state(self.streams[stream_id])}"
            for stream_id in tear_streams
        )
        self._emit_progress(
            f"recycle_eval_done {evaluation_number} max_residual={error:.6g} "
            f"worst={worst_variable} "
            f"| new {raw_summary}"
        )
        return RecycleEvaluation(
            success=True,
            x=applied_scaled,
            gx=gx,
            residual=residual,
            warnings=iteration_warnings,
            error=error,
            worst_variable=worst_variable,
        )

    def _finalize_recycle_solution(self, calc_order: list[str],
                                   tear_streams: list[str],
                                   scaled_vector,
                                   scale,
                                   fallback_states: dict[str, StreamState],
                                   tolerance: float,
                                   evaluations: int
                                   ) -> tuple[bool, list[str], float]:
        import numpy as np

        self._apply_tear_vector(
            tear_streams,
            (np.asarray(scaled_vector, dtype=float) * scale).tolist(),
            fallback_states,
        )
        old_tear_states = {
            stream_id: self.streams[stream_id].copy()
            for stream_id in tear_streams
        }

        final_warnings = []
        try:
            solve_context = self._recycle_solve_context(evaluations, final_pass=True)
            for unit_id in calc_order:
                if unit_id in self._recycle_deferred_units:
                    self._emit_progress(
                        f"unit_skipped {unit_id} deferred_until_recycle_final_pass"
                    )
                    continue
                if (
                    unit_id in self._recycle_invariant_units
                    and unit_id in self._recycle_invariant_units_ready
                ):
                    self._emit_progress(
                        f"unit_cached {unit_id} recycle_invariant"
                    )
                    continue
                result = self._calculate_unit(unit_id, solve_context=solve_context)
                self.unit_results[unit_id] = result
                if unit_id in self._recycle_invariant_units:
                    self._recycle_invariant_units_ready.add(unit_id)
                final_warnings.extend(result.warnings)
        except (UnitOperationError, ThermodynamicsError, FlowsheetError) as e:
            raise FlowsheetError(str(e)) from e

        converged, max_error = self._check_convergence(
            tear_streams,
            old_tear_states,
            tolerance,
        )
        self._emit_progress(
            f"recycle_solver_done converged={converged} evaluations={evaluations} "
            f"max_error={max_error:.6g}"
        )
        return converged, final_warnings, max_error

    def _solve_recycles_direct(self, calc_order: list[str],
                               tear_streams: list[str],
                               tolerance: float,
                               max_iterations: int,
                               x0,
                               scale,
                               fallback_states: dict[str, StreamState],
                               damping: float = 1.0
                               ) -> tuple[bool, int, list[str], float]:
        evaluations = 0
        x = x0 / scale
        best_x = x.copy()
        best_error = math.inf
        latest_warnings = []

        for iteration in range(1, max_iterations + 1):
            evaluations += 1
            evaluation = self._run_recycle_evaluation(
                calc_order, tear_streams, x, scale, fallback_states, evaluations
            )
            if not evaluation.success:
                x = 0.5 * (x + best_x)
                continue
            latest_warnings = evaluation.warnings
            if evaluation.error < best_error:
                best_error = evaluation.error
                best_x = evaluation.x.copy()
            if evaluation.error < tolerance:
                converged, final_warnings, final_error = self._finalize_recycle_solution(
                    calc_order, tear_streams, evaluation.x, scale, fallback_states, tolerance, evaluations
                )
                return converged, evaluations, final_warnings or latest_warnings, final_error
            x = evaluation.x + damping * evaluation.residual

        converged, final_warnings, final_error = self._finalize_recycle_solution(
            calc_order, tear_streams, best_x, scale, fallback_states, tolerance, evaluations
        )
        return converged, evaluations, final_warnings or latest_warnings, min(best_error, final_error)

    def _solve_recycles_wegstein(self, calc_order: list[str],
                                 tear_streams: list[str],
                                 tolerance: float,
                                 max_iterations: int,
                                 x0,
                                 scale,
                                 fallback_states: dict[str, StreamState]
                                 ) -> tuple[bool, int, list[str], float]:
        import numpy as np

        evaluations = 0
        x = x0 / scale
        previous_x = None
        previous_gx = None
        best_x = x.copy()
        best_error = math.inf
        latest_warnings = []
        tear_count = len(tear_streams)
        stagnant_iterations = 0
        max_acceleration = float(
            self._recycle_method_options['max_acceleration']
        )
        stagnation_limit = int(
            self._recycle_method_options['stagnation_iterations']
        )
        fallback_damping = float(
            self._recycle_method_options['fallback_damping']
        )

        for iteration in range(1, max_iterations + 1):
            evaluations += 1
            evaluation = self._run_recycle_evaluation(
                calc_order, tear_streams, x, scale, fallback_states, evaluations
            )
            if not evaluation.success:
                if previous_gx is not None:
                    x = 0.5 * (previous_gx + previous_x)
                else:
                    x = 0.5 * (x + best_x)
                continue

            latest_warnings = evaluation.warnings
            if evaluation.error < best_error:
                best_error = evaluation.error
                best_x = evaluation.x.copy()
                stagnant_iterations = 0
            else:
                stagnant_iterations += 1
            if evaluation.error < tolerance:
                converged, final_warnings, final_error = self._finalize_recycle_solution(
                    calc_order, tear_streams, evaluation.x, scale, fallback_states, tolerance, evaluations
                )
                return converged, evaluations, final_warnings or latest_warnings, final_error
            if (
                iteration >= max(5, max_iterations // 2)
                and tolerance * 2.0 < best_error < 1e-2
            ):
                remaining = max_iterations - evaluations
                if remaining > 0:
                    self._emit_progress(
                        "recycle_solver_wegstein_fallback "
                        f"reason=slow_progress best_error={best_error:.6g} "
                        f"remaining_iterations={remaining}"
                    )
                    converged, extra_evals, warnings, error = self._solve_recycles_direct(
                        calc_order,
                        tear_streams,
                        tolerance,
                        remaining,
                        best_x * scale,
                        scale,
                        fallback_states,
                        damping=fallback_damping,
                    )
                    return converged, evaluations + extra_evals, warnings or latest_warnings, error
            if stagnant_iterations >= stagnation_limit and best_error < 1e-2:
                remaining = max_iterations - evaluations
                if remaining > 0:
                    self._emit_progress(
                        "recycle_solver_wegstein_fallback "
                        f"reason=stagnation best_error={best_error:.6g} "
                        f"remaining_iterations={remaining}"
                    )
                    converged, extra_evals, warnings, error = self._solve_recycles_direct(
                        calc_order,
                        tear_streams,
                        tolerance,
                        remaining,
                        best_x * scale,
                        scale,
                        fallback_states,
                        damping=fallback_damping,
                    )
                    return converged, evaluations + extra_evals, warnings or latest_warnings, error

            if previous_x is None:
                candidate = evaluation.gx
            else:
                dx = evaluation.x - previous_x
                dg = evaluation.gx - previous_gx
                q = np.zeros_like(evaluation.x)
                mask = np.abs(dx) > 1e-12
                slope = np.zeros_like(evaluation.x)
                slope[mask] = dg[mask] / dx[mask]
                denom = slope - 1.0
                good = mask & (np.abs(denom) > 1e-12) & np.isfinite(slope)
                q[good] = slope[good] / denom[good]
                q = np.clip(q, max_acceleration, 0.0)
                candidate = q * evaluation.x + (1.0 - q) * evaluation.gx

            candidate = self._limit_recycle_step(
                evaluation.x, candidate, scale, tear_count
            )
            previous_x = evaluation.x.copy()
            previous_gx = evaluation.gx.copy()
            x = candidate

        if best_error > tolerance and best_error < 1e-2:
            self._emit_progress(
                "recycle_solver_wegstein_fallback "
                f"reason=final_polish best_error={best_error:.6g} "
                f"remaining_iterations={max_iterations}"
            )
            converged, extra_evals, warnings, error = self._solve_recycles_direct(
                calc_order,
                tear_streams,
                tolerance,
                max_iterations,
                best_x * scale,
                scale,
                fallback_states,
                damping=fallback_damping,
            )
            return converged, evaluations + extra_evals, warnings or latest_warnings, error

        converged, final_warnings, final_error = self._finalize_recycle_solution(
            calc_order, tear_streams, best_x, scale, fallback_states, tolerance, evaluations
        )
        if not converged and final_error < 1e-2:
            import numpy as np
            self._emit_progress(
                "recycle_solver_wegstein_fallback "
                f"reason=final_polish final_error={final_error:.6g} "
                f"remaining_iterations={max_iterations}"
            )
            current_states = {
                stream_id: self.streams[stream_id].copy()
                for stream_id in tear_streams
            }
            polish_x0 = np.array(self._tear_states_to_vector(tear_streams, current_states), dtype=float)
            converged, extra_evals, warnings, error = self._solve_recycles_direct(
                calc_order,
                tear_streams,
                tolerance,
                max_iterations,
                polish_x0,
                scale,
                fallback_states,
                damping=fallback_damping,
            )
            return converged, evaluations + extra_evals, warnings or final_warnings or latest_warnings, error
        return converged, evaluations, final_warnings or latest_warnings, min(best_error, final_error)

    def _solve_recycles_broyden(self, calc_order: list[str],
                                tear_streams: list[str],
                                tolerance: float,
                                max_iterations: int,
                                x0,
                                scale,
                                fallback_states: dict[str, StreamState]
                                ) -> tuple[bool, int, list[str], float]:
        import numpy as np

        evaluations = 0
        x = x0 / scale
        n = len(x)
        inverse_jacobian = -np.eye(n)
        best_x = x.copy()
        best_error = math.inf
        latest_warnings = []
        previous_eval = None
        tear_count = len(tear_streams)
        stagnant_iterations = 0
        stagnation_limit = int(
            self._recycle_method_options['stagnation_iterations']
        )
        divergence_factor = float(
            self._recycle_method_options['divergence_factor']
        )
        fallback_damping = float(
            self._recycle_method_options['fallback_damping']
        )

        for iteration in range(1, max_iterations + 1):
            evaluations += 1
            evaluation = self._run_recycle_evaluation(
                calc_order, tear_streams, x, scale, fallback_states, evaluations
            )
            if not evaluation.success:
                if previous_eval is not None:
                    x = 0.5 * (previous_eval.x + previous_eval.gx)
                else:
                    x = 0.5 * (x + best_x)
                continue

            latest_warnings = evaluation.warnings
            if best_error < math.inf and evaluation.error > max(
                divergence_factor * best_error,
                tolerance * 100.0,
            ):
                inverse_jacobian = -np.eye(n)
                x = 0.5 * (evaluation.x + best_x)
                continue
            if evaluation.error < best_error:
                best_error = evaluation.error
                best_x = evaluation.x.copy()
                stagnant_iterations = 0
            else:
                stagnant_iterations += 1
            if evaluation.error < tolerance:
                converged, final_warnings, final_error = self._finalize_recycle_solution(
                    calc_order, tear_streams, evaluation.x, scale, fallback_states, tolerance, evaluations
                )
                return converged, evaluations, final_warnings or latest_warnings, final_error
            if stagnant_iterations >= stagnation_limit:
                remaining = max_iterations - evaluations
                if remaining > 0:
                    self._emit_progress(
                        "recycle_solver_broyden_fallback "
                        f"reason=stagnation best_error={best_error:.6g} "
                        f"remaining_iterations={remaining}"
                    )
                    converged, extra_evals, warnings, error = self._solve_recycles_direct(
                        calc_order,
                        tear_streams,
                        tolerance,
                        remaining,
                        best_x * scale,
                        scale,
                        fallback_states,
                        damping=fallback_damping,
                    )
                    return converged, evaluations + extra_evals, warnings or latest_warnings, error

            if previous_eval is not None and iteration > 2:
                s = evaluation.x - previous_eval.x
                y = evaluation.residual - previous_eval.residual
                by = inverse_jacobian @ y
                denom = float(s @ by)
                if abs(denom) > 1e-14 and np.isfinite(denom):
                    inverse_jacobian = inverse_jacobian + np.outer(
                        (s - by),
                        s @ inverse_jacobian,
                    ) / denom
                else:
                    inverse_jacobian = -np.eye(n)

            if previous_eval is None or iteration <= 2:
                step = evaluation.residual
            else:
                step = -inverse_jacobian @ evaluation.residual
            if not np.all(np.isfinite(step)):
                step = evaluation.residual
                inverse_jacobian = -np.eye(n)
            candidate = evaluation.x + step
            candidate = self._limit_recycle_step(
                evaluation.x, candidate, scale, tear_count
            )

            previous_eval = evaluation
            x = candidate

        converged, final_warnings, final_error = self._finalize_recycle_solution(
            calc_order, tear_streams, best_x, scale, fallback_states, tolerance, evaluations
        )
        return converged, evaluations, final_warnings or latest_warnings, min(best_error, final_error)

    def _solve_recycle_block(self, calc_order: list[str],
                             tear_streams: list[str],
                             tolerance: float,
                             max_iterations: int,
                             method: str = 'WEGSTEIN'
                             ) -> tuple[bool, int, list[str], float]:
        fallback_states = {
            stream_id: self.streams[stream_id].copy()
            for stream_id in tear_streams
        }
        x0, scale = self._make_recycle_scale(tear_streams, fallback_states)

        try:
            method = normalize_recycle_method(method or 'WEGSTEIN')
        except ValueError as error:
            raise FlowsheetError(str(error)) from error
        
        self._emit_progress(
            "recycle_solver_start "
            f"tears={', '.join(tear_streams)} order={', '.join(calc_order)} "
            f"method={method} variable_basis=F,n_i,P,H "
            f"options={self._recycle_method_options} "
            f"tolerance={tolerance:g} trace_tolerance={self._recycle_trace_tolerance:g} "
            f"max_iterations={max_iterations}"
        )

        if method == 'DIRECT':
            return self._solve_recycles_direct(
                calc_order,
                tear_streams,
                tolerance,
                max_iterations,
                x0,
                scale,
                fallback_states,
                damping=float(self._recycle_method_options['damping']),
            )
        if method == 'BROYDEN':
            return self._solve_recycles_broyden(
                calc_order, tear_streams, tolerance, max_iterations, x0, scale, fallback_states
            )
        return self._solve_recycles_wegstein(
            calc_order, tear_streams, tolerance, max_iterations, x0, scale, fallback_states
        )

    def _solve_recycles(self, calc_order: list[str],
                        tear_streams: list[str],
                        tolerance: float,
                        max_iterations: int,
                        method: str = 'WEGSTEIN'
                        ) -> tuple[bool, int, list[str], float]:
        self._recycle_failed_evaluations = 0
        self._recycle_last_failure = ""
        self._recycle_worst_variable = ""
        self._recycle_worst_error = 0.0

        blocks = self._recycle_blocks(calc_order, tear_streams)
        self._recycle_blocks_report = [
            {
                'tear_streams': list(block['tear_streams']),
                'required_units': [
                    unit_id for unit_id in calc_order
                    if unit_id in block['required_units']
                ],
                'repeated_units': [
                    unit_id for unit_id in calc_order
                    if unit_id in block['repeated_units']
                ],
                'invariant_units': [
                    unit_id for unit_id in calc_order
                    if unit_id in block['invariant_units']
                ],
                'deferred_units': list(block['deferred_units']),
            }
            for block in blocks
        ]
        deferred_report = []
        seen_deferred = set()
        for block in blocks:
            for unit_id in block['deferred_units']:
                if unit_id not in seen_deferred:
                    seen_deferred.add(unit_id)
                    deferred_report.append(unit_id)
        self._recycle_deferred_units_report = deferred_report
        invariant_report = []
        seen_invariant = set()
        for block in blocks:
            for unit_id in calc_order:
                if (
                    unit_id in block['invariant_units']
                    and unit_id not in seen_invariant
                ):
                    seen_invariant.add(unit_id)
                    invariant_report.append(unit_id)
        self._recycle_invariant_units_report = invariant_report

        if len(blocks) > 1:
            self._emit_progress(
                "recycle_blocks "
                + '; '.join(
                    "tears="
                    + ','.join(block['tear_streams'])
                    + " required="
                    + ','.join(
                        unit_id for unit_id in calc_order
                        if unit_id in block['required_units']
                    )
                    for block in blocks
                )
            )

        total_evaluations = 0
        all_warnings = []
        converged = True
        max_error = 0.0

        for block in blocks:
            self._recycle_deferred_units = list(block['deferred_units'])
            self._recycle_invariant_units = set(block['invariant_units'])
            self._recycle_invariant_units_ready = set()
            if self._recycle_deferred_units:
                self._emit_progress(
                    "recycle_deferred_units "
                    + ', '.join(self._recycle_deferred_units)
                )
            if self._recycle_invariant_units:
                self._emit_progress(
                    "recycle_invariant_units "
                    + ', '.join(
                        unit_id for unit_id in calc_order
                        if unit_id in self._recycle_invariant_units
                    )
                )
            block_converged, evaluations, warnings, error = self._solve_recycle_block(
                calc_order,
                block['tear_streams'],
                tolerance,
                max_iterations,
                method,
            )
            total_evaluations += evaluations
            all_warnings.extend(warnings)
            max_error = max(max_error, error)
            if not block_converged:
                converged = False
                break

        self._recycle_deferred_units = []
        self._recycle_invariant_units = set()
        self._recycle_invariant_units_ready = set()
        if converged:
            fallback_states = {
                stream_id: self.streams[stream_id].copy()
                for stream_id in tear_streams
            }
            x0, scale = self._make_recycle_scale(tear_streams, fallback_states)
            converged, final_warnings, final_error = self._finalize_recycle_solution(
                calc_order,
                tear_streams,
                x0 / scale,
                scale,
                fallback_states,
                tolerance,
                total_evaluations,
            )
            all_warnings = final_warnings or all_warnings
            max_error = max(max_error, final_error)

        return converged, total_evaluations, all_warnings, max_error
    
    def _calculate_balances(self) -> tuple[float, float]:
        """Calculate overall mass and energy balance errors"""
        # Mass balance
        mass_in = 0.0
        mass_out = 0.0
        
        for stream in self.pfd.streams:
            state = self.streams.get(stream.id)
            if state is None:
                continue
            
            mass_flow = state.F * (state.MW or 0)  # kg/h
            
            if stream.source.is_feed:
                mass_in += mass_flow
            if stream.destination.is_product:
                mass_out += mass_flow
        
        if mass_in > 0:
            mass_error = abs(mass_out - mass_in) / mass_in
        else:
            mass_error = 0.0
        
        # Energy balance
        H_in = 0.0
        H_out = 0.0
        Q_total = 0.0
        W_total = 0.0
        
        for stream in self.pfd.streams:
            state = self.streams.get(stream.id)
            if state is None or state.H is None:
                continue
            
            H_flow = state.F * state.H  # kJ/h
            
            if stream.source.is_feed:
                H_in += H_flow
            if stream.destination.is_product:
                H_out += H_flow
        
        for unit_id, result in self.unit_results.items():
            Q_total += result.heat_duty
            W_total += result.work

        scope_correction = sum(
            float(record['enthalpy_flow_correction_kJ_per_h'])
            for record in self._thermo_scope_corrections.values()
        )
        
        # H_out = H_in + Q + W + thermodynamic-package reconciliation.
        expected_H_out = H_in + Q_total + W_total + scope_correction
        if abs(expected_H_out) > 0:
            energy_error = abs(H_out - expected_H_out) / abs(expected_H_out)
        else:
            energy_error = 0.0
        
        return mass_error, energy_error
    
    def solve(self, max_iterations: int = 100, tolerance: float = 1e-4,
              progress_callback: Optional[Callable[[str], None]] = None,
              recycle_method: Optional[str] = None) -> SimulationResult:
        """
        Solve the complete flowsheet.
        
        Args:
            max_iterations: Maximum iterations for recycle convergence
            tolerance: Convergence tolerance for tear streams
            progress_callback: Optional callable receiving progress messages
            recycle_method: DIRECT, WEGSTEIN, or BROYDEN. Defaults to PFD
                RECYCLE_METHOD metadata, then WEGSTEIN.
            
        Returns:
            SimulationResult with all stream and unit results
        """
        errors = []
        warnings = []
        # A solver is persistent after Simulator.initialize(), but stream and
        # unit results belong to one solve request. Starting clean preserves
        # deterministic repeated-run behavior while retaining immutable graph,
        # unit, thermodynamic, and property-backend initialization.
        self.streams.clear()
        self.unit_results.clear()
        self.progress_callback = progress_callback
        self._thermo_scope_corrections.clear()
        self._recycle_deferred_units = []
        self._recycle_invariant_units = set()
        self._recycle_invariant_units_ready = set()
        self._recycle_invariant_units_report = []
        metadata = getattr(self.pfd, 'metadata', None)
        metadata_method = getattr(metadata, 'recycle_method', None) or 'WEGSTEIN'
        try:
            metadata_method = normalize_recycle_method(metadata_method)
            selected_method = normalize_recycle_method(
                recycle_method or metadata_method
            )
        except ValueError as error:
            return SimulationResult(
                converged=False,
                iterations=0,
                streams={},
                units={},
                errors=[str(error)],
            )
        configured_options = (
            getattr(metadata, 'recycle_options', {})
            if recycle_method is None or selected_method == metadata_method
            else {}
        )
        try:
            self._recycle_method_options = resolved_recycle_options(
                selected_method,
                configured_options,
            )
        except ValueError as error:
            return SimulationResult(
                converged=False,
                iterations=0,
                streams={},
                units={},
                errors=[str(error)],
            )
        recycle_method = selected_method
        trace_override = getattr(
            getattr(self.pfd, 'metadata', None),
            'recycle_trace_tolerance',
            None,
        )
        if trace_override is None:
            self._recycle_trace_tolerance = max(tolerance / 100.0, 1e-12)
        else:
            self._recycle_trace_tolerance = max(float(trace_override), 0.0)
        
        # Get calculation order and tear streams
        try:
            calc_order, tear_streams = self._topological_sort()
            self._emit_progress(
                f"flowsheet_order order={', '.join(calc_order)} "
                f"tears={', '.join(tear_streams) if tear_streams else 'none'} "
                f"auto_tears={', '.join(self._auto_selected_tears) if self._auto_selected_tears else 'none'}"
            )
        except Exception as e:
            return SimulationResult(
                converged=False,
                iterations=0,
                streams={},
                units={},
                errors=[f"Failed to determine calculation order: {e}"]
            )
        
        # Initialize feed streams
        try:
            self._initialize_streams()
            self._emit_progress(
                "feed_streams_initialized "
                + '; '.join(
                    f"{stream_id}: {self._format_stream_state(state)}"
                    for stream_id, state in self.streams.items()
                )
            )
        except FlowsheetError as e:
            return SimulationResult(
                converged=False,
                iterations=0,
                streams={},
                units={},
                errors=[str(e)]
            )
        
        # Initialize tear streams
        if tear_streams:
            try:
                self._initialize_tear_streams(tear_streams)
            except FlowsheetError as e:
                return SimulationResult(
                    converged=False,
                    iterations=0,
                    streams=dict(self.streams),
                    units={},
                    errors=[str(e)],
                    warnings=warnings,
                )
            warnings.append(f"Recycle detected: tearing stream(s) {', '.join(tear_streams)}")
            self._emit_progress(
                "tear_streams_initialized "
                + '; '.join(
                    f"{stream_id}: {self._format_stream_state(self.streams[stream_id])}"
                    for stream_id in tear_streams
                )
            )
        
        # Iteration loop
        converged = False
        iteration = 0
        latest_unit_warnings = []
        max_error = 0.0
        
        if tear_streams:
            converged, iteration, latest_unit_warnings, max_error = self._solve_recycles(
                calc_order,
                tear_streams,
                tolerance,
                max_iterations,
                recycle_method,
            )
        else:
            for iteration in range(1, max_iterations + 1):
                iteration_warnings = []
                
                # Calculate all units in order
                for unit_id in calc_order:
                    try:
                        result = self._calculate_unit(unit_id)
                        self.unit_results[unit_id] = result
                        iteration_warnings.extend(result.warnings)
                    except (UnitOperationError, ThermodynamicsError) as e:
                        errors.append(f"Unit '{unit_id}': {e}")
                        warnings.extend(iteration_warnings)
                        return SimulationResult(
                            converged=False,
                            iterations=iteration,
                            streams=dict(self.streams),
                            units=dict(self.unit_results),
                            errors=errors,
                            warnings=warnings
                        )
                
                latest_unit_warnings = iteration_warnings
                converged = True
                break
        
        if not converged and tear_streams:
            warnings.append(
                f"Recycle did not converge after {iteration} iterations "
                f"(error: {max_error:.2e})"
            )
        
        warnings.extend(latest_unit_warnings)
        
        # Calculate overall balances
        mass_error, energy_error = self._calculate_balances()
        
        if mass_error > 0.01:
            warnings.append(f"Mass balance error: {mass_error*100:.2f}%")
        if energy_error > 0.05:
            warnings.append(f"Energy balance error: {energy_error*100:.2f}%")

        for thermo in self.thermo_packages.values():
            warnings.extend(getattr(thermo, 'warnings', []))
        warnings = self._dedupe_warnings(warnings)

        scope_corrections = [
            dict(self._thermo_scope_corrections[stream_id])
            for stream_id in sorted(self._thermo_scope_corrections)
        ]
        scope_enthalpy_correction = sum(
            float(record['enthalpy_flow_correction_kJ_per_h'])
            for record in scope_corrections
        )
        
        return SimulationResult(
            converged=converged,
            iterations=iteration,
            streams=dict(self.streams),
            units=dict(self.unit_results),
            errors=errors,
            warnings=warnings,
            mass_balance_error=mass_error,
            energy_balance_error=energy_error,
            thermo_scope_enthalpy_correction=scope_enthalpy_correction,
            thermo_scope_corrections=scope_corrections,
            recycle_info={
                'tear_streams': tear_streams,
                'calculation_order': calc_order,
                'method': recycle_method,
                'method_options': dict(self._recycle_method_options),
                'failed_evaluations': self._recycle_failed_evaluations,
                'last_failure': self._recycle_last_failure,
                'auto_selected_tears': list(self._auto_selected_tears),
                'manual_tears': list(getattr(getattr(self.pfd, 'metadata', None), 'recycle_tear_streams', []) or []),
                'deferred_units': list(self._recycle_deferred_units_report),
                'invariant_units': list(self._recycle_invariant_units_report),
                'recycle_blocks': list(self._recycle_blocks_report),
                'trace_tolerance': self._recycle_trace_tolerance,
                'variable_basis': 'total_flow_component_flows_pressure_enthalpy',
                'worst_variable': self._recycle_worst_variable,
                'worst_error': self._recycle_worst_error,
            }
        )
