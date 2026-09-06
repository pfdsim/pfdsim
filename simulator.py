"""
Process Simulator

Main interface for running process simulations and generating results.

Supported thermodynamic families include ideal/Raoult, cubic EOS, PSRK,
activity-coefficient and gamma-phi methods, and IF97 steam. Activity methods
may select global VLE, local-spinodal VL(L)E, or robust VLLE stream flashes.

Usage:
    from simulator import Simulator
    
    sim = Simulator.from_file('process.pfd')
    result = sim.run(thermo_method='IDEAL')  # or 'RK'
    sim.write_results('process.pfr')
"""

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .pfd_parser import (
        parse_pfd,
        validate_pfd,
        normalize_interaction_model,
        normalize_vdm_component_parameters,
        normalize_vdm_cross_parameters,
        ProcessFlowDiagram,
        ParseError,
    )
else:
    from pfd_parser import (
        parse_pfd,
        validate_pfd,
        normalize_interaction_model,
        normalize_vdm_component_parameters,
        normalize_vdm_cross_parameters,
        ProcessFlowDiagram,
        ParseError,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics import IdealThermodynamics, ThermodynamicsError, StreamState
else:
    from thermodynamics import IdealThermodynamics, ThermodynamicsError, StreamState
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .fluid_phase_models import LLE_CAPABLE_THERMO_METHODS
else:
    from fluid_phase_models import LLE_CAPABLE_THERMO_METHODS
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .phase_behaviors import (
        CONVENTIONAL_WITH_SOLID_PHASE_BEHAVIOR,
        PERMANENT_SOLID_PHASE_BEHAVIOR,
        normalize_phase_behavior,
    )
else:
    from phase_behaviors import (
        CONVENTIONAL_WITH_SOLID_PHASE_BEHAVIOR,
        PERMANENT_SOLID_PHASE_BEHAVIOR,
        normalize_phase_behavior,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .flowsheet_solver import FlowsheetSolver, SimulationResult, FlowsheetError
else:
    from flowsheet_solver import FlowsheetSolver, SimulationResult, FlowsheetError
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .dof_analyzer import analyze_dof, SpecificationStatus
else:
    from dof_analyzer import analyze_dof, SpecificationStatus


class SimulationError(Exception):
    """Error during simulation"""
    pass


class Simulator:
    """
    Process flow diagram simulator.
    
    Supports constrained VLE for every thermodynamic method and optional
    universal LLE/VLLE-aware flashes for activity-coefficient methods and
    explicit permanent-solid process components that remain outside fluid
    equilibrium.
    """
    
    SUPPORTED_PHASES = {'vapor', 'liquid', 'VLE', 'LLE', 'VLLE'}
    UNSUPPORTED_PHASES = {'solid', 'SLE'}
    
    def __init__(self, pfd: ProcessFlowDiagram):
        """
        Initialize simulator with a parsed PFD.
        
        Args:
            pfd: ProcessFlowDiagram object from pfd_parser
        """
        self.pfd = pfd
        self.thermo: Optional[IdealThermodynamics] = None
        self.thermo_packages: dict[str, IdealThermodynamics] = {}
        self.thermo_scope_methods: dict[str, str] = {}
        self.solver: Optional[FlowsheetSolver] = None
        self.result: Optional[SimulationResult] = None
        self.thermo_method: Optional[str] = None
        self._initialized = False
        self._initialized_thermo_method: Optional[str] = None
        self._initialized_fluid_phase_model: Optional[str] = None
        
        # Validate PFD
        self._validate()
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Simulator':
        """
        Create simulator from a .pfd file.
        
        Args:
            filepath: Path to .pfd file
            
        Returns:
            Simulator instance
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        pfd = parse_pfd(content)
        return cls(pfd)
    
    @classmethod
    def from_string(cls, content: str) -> 'Simulator':
        """
        Create simulator from PFD string content.
        
        Args:
            content: PFD file content as string
            
        Returns:
            Simulator instance
        """
        pfd = parse_pfd(content)
        return cls(pfd)
    
    def _validate(self):
        """Validate PFD before simulation"""
        # Basic structural validation
        errors, warnings = validate_pfd(self.pfd)
        
        if errors:
            raise SimulationError(
                f"PFD validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        
        # DOF analysis
        dof_result = analyze_dof(self.pfd)
        
        if dof_result.overall_status == SpecificationStatus.UNDER_SPECIFIED:
            raise SimulationError(
                f"Process is under-specified:\n" + 
                "\n".join(f"  - {e}" for e in dof_result.errors)
            )
        
        if dof_result.overall_status == SpecificationStatus.OVER_SPECIFIED:
            raise SimulationError(
                f"Process is over-specified:\n" +
                "\n".join(f"  - {e}" for e in dof_result.errors)
            )
        
        # Get thermo method from metadata
        thermo_method = getattr(self.pfd.metadata, 'thermo_method', 'IDEAL')
        if thermo_method is None:
            thermo_method = 'IDEAL'
        thermo_method = thermo_method.upper()
        scope_methods = {
            'global': thermo_method,
            **{
                scope.name: str(scope.method).upper()
                for scope in getattr(self.pfd, 'thermo_scopes', [])
            },
        }
        
        lle_methods = LLE_CAPABLE_THERMO_METHODS
        fluid_phase_model = str(
            getattr(self.pfd.metadata, 'fluid_phase_model', 'VLE') or 'VLE'
        ).upper()
        has_conventional_component = any(
            normalize_phase_behavior(component.phase_behavior)
            != PERMANENT_SOLID_PHASE_BEHAVIOR
            for component in self.pfd.components
        )
        if (
            has_conventional_component
            and fluid_phase_model != 'VLE'
            and thermo_method not in lle_methods
        ):
            raise SimulationError(
                f"FLUID_PHASE_MODEL {fluid_phase_model} requires an "
                "LLE-capable activity thermodynamic method such as "
                "UNIFAC, NRTL, or UNIQUAC"
            )

        # Check for unsupported unit types
        for unit in self.pfd.units:
            unit_type = unit.unit_type
            unit_scope = next((
                str(param.value).strip()
                for param in unit.params
                if param.name.lower() == 'thermo_scope'
            ), 'global')
            unit_thermo_method = scope_methods[unit_scope]
            
            # Check for LLE/solid units - only restrict if no liquid activity model is selected.
            if unit_thermo_method not in lle_methods:
                if any(phase in unit_type.lower() for phase in ['lle', 'decant', 'extract']):
                    raise SimulationError(
                        f"Unit '{unit.id}' ({unit_type}) requires LLE capability. "
                        f"Use a liquid activity model such as UNIFAC, NRTL, or UNIQUAC."
                    )
            
            # These solid operations implement explicit solid inventory routing.
            if (
                unit_type not in {'MolecularSieveDryer', 'Crystallizer', 'Filter'}
                and any(phase in unit_type.lower() for phase in ['solid', 'crystal', 'filter', 'dryer'])
            ):
                raise SimulationError(
                    f"Unit '{unit.id}' ({unit_type}) requires solid handling, "
                    "which is not yet supported"
                )
            
            # Check valid_phases parameter
            for param in unit.params:
                if param.name.lower() == 'valid_phases':
                    if (
                        str(param.value).upper().replace(' ', '')
                        in {'VLL', 'VLLE', 'LLE', 'VL(L)E', 'VLL(E)', 'ADAPTIVE_VLLE'}
                        and unit_thermo_method not in lle_methods
                    ):
                        raise SimulationError(
                            f"Unit '{unit.id}' specifies {param.value} phases. "
                            f"Use a liquid activity model such as UNIFAC, NRTL, or UNIQUAC."
                        )
    
    def _resolve_thermo_method(self, thermo_method: Optional[str]) -> str:
        """Return the explicit or PFD-selected thermodynamic method."""
        if thermo_method is not None:
            return str(thermo_method).upper()
        selected = 'IDEAL'
        if hasattr(self.pfd, 'metadata') and self.pfd.metadata:
            if (
                hasattr(self.pfd.metadata, 'thermo_method')
                and self.pfd.metadata.thermo_method
            ):
                selected = self.pfd.metadata.thermo_method
            elif hasattr(self.pfd.metadata, 'properties'):
                for prop in getattr(self.pfd.metadata, 'properties', []):
                    if (
                        hasattr(prop, 'name')
                        and prop.name.lower()
                        in ('thermo', 'thermo_method', 'property_method')
                    ):
                        selected = prop.value
                        break
        return str(selected).upper()

    def initialize(self, thermo_method: Optional[str] = None) -> 'Simulator':
        """
        Initialize deterministic resources required by this simulation.
        
        Args:
            thermo_method: Thermodynamic/property method (for example 'IDEAL',
                'PR', 'PSRK', 'UNIFAC', 'UNIQUAC-RK', or 'UNIFAC-PR')
            
        Returns:
            This simulator, fully initialized but not solved.
        """
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .thermodynamics import create_thermodynamics
        else:
            from thermodynamics import create_thermodynamics
        
        selected_phase_model = str(
            getattr(self.pfd.metadata, 'fluid_phase_model', 'VLE') or 'VLE'
        ).upper()
        if (
            thermo_method is None
            and self._initialized
            and self._initialized_fluid_phase_model == selected_phase_model
        ):
            return self
        selected_method = self._resolve_thermo_method(thermo_method)
        if (
            self._initialized
            and self._initialized_thermo_method == selected_method
            and self._initialized_fluid_phase_model == selected_phase_model
        ):
            return self

        # A different explicit method is a deliberate reconfiguration. Clear
        # only derived runtime state; the parsed PFD remains immutable input.
        self._initialized = False
        self._initialized_thermo_method = None
        self._initialized_fluid_phase_model = None
        self.thermo = None
        self.thermo_packages = {}
        self.thermo_scope_methods = {}
        self.solver = None
        self.result = None
        self.thermo_method = selected_method
        self.thermo_scope_methods = {
            'global': self.thermo_method,
            **{
                scope.name: str(scope.method).upper()
                for scope in getattr(self.pfd, 'thermo_scopes', [])
            },
        }
        scope_parents = {
            scope.name: (
                str(scope.inherit)
                if scope.inherit
                else None
            )
            for scope in getattr(self.pfd, 'thermo_scopes', [])
        }

        def thermo_scope_lineage(scope_name: str) -> list[str]:
            if scope_name == 'global':
                return ['global']
            lineage = []
            current = scope_name
            while current and current != 'global':
                lineage.append(current)
                current = scope_parents.get(current)
            if current == 'global':
                lineage.append('global')
            return list(reversed(lineage))
        normalized_thermo_methods = {
            method.replace('_', '-')
            for method in self.thermo_scope_methods.values()
        }
        uses_mathias_copeman = any(
            method.endswith('-MC')
            or method in {'PSRK', 'PREDICTIVE-SRK'}
            for method in normalized_thermo_methods
        )
        uses_prsv1 = bool(normalized_thermo_methods & {
            'PRSV', 'PRSV1', 'PR-SV', 'PR-SV1',
            'PENG-ROBINSON-SV', 'PENG-ROBINSON-SV1',
            'PENG-ROBINSON-STRYJEK-VERA',
        })
        uses_prsv2 = bool(normalized_thermo_methods & {
            'PRSV2', 'PR-SV2', 'PENG-ROBINSON-SV2',
        })
        uses_twu = bool(normalized_thermo_methods & {
            'SRK-TWU', 'RKS-TWU', 'RK-SOAVE-TWU',
            'PR-TWU', 'PENG-ROBINSON-TWU',
        })
        uses_uniquac = any(
            method.startswith('UNIQUAC')
            for method in normalized_thermo_methods
        )
        uses_vdm = any(
            method.endswith('-VDM')
            for method in normalized_thermo_methods
        )
        ignored_model_parameter_warnings: list[str] = []
        
        # Get database
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .chemical_properties import ChemicalDatabase, ChemicalProperties
        else:
            from chemical_properties import ChemicalDatabase, ChemicalProperties
        allow_online_lookup = getattr(self.pfd.metadata, 'online_lookup', True)
        psat_minimum_pressure_bar = getattr(
            self.pfd.metadata,
            'psat_minimum_pressure_bar',
            None,
        )
        db = ChemicalDatabase(enable_online=allow_online_lookup)

        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .compound_identity import (
                    get_compound_identity_resolver,
                    looks_like_formula,
                    parse_formula_counts,
                )
            else:
                from compound_identity import (
                    get_compound_identity_resolver,
                    looks_like_formula,
                    parse_formula_counts,
                )
        except ImportError:
            get_compound_identity_resolver = None
            looks_like_formula = lambda _value: False
            parse_formula_counts = lambda _value: None

        pfd_property_attrs = (
            ('formula', 'formula'),
            ('CAS', 'CAS'),
            ('smiles', 'smiles'),
            ('Tc', 'Tc'),
            ('Pc', 'Pc'),
            ('Vc', 'Vc'),
            ('Zc', 'Zc'),
            ('omega', 'omega'),
            ('mc_c1', 'mc_c1'),
            ('mc_c2', 'mc_c2'),
            ('mc_c3', 'mc_c3'),
            ('kappa1', 'kappa1'),
            ('kappa2', 'kappa2'),
            ('kappa3', 'kappa3'),
            ('twu_l', 'twu_l'),
            ('twu_m', 'twu_m'),
            ('twu_n', 'twu_n'),
            ('twu_c', 'twu_c'),
            ('henry_Hcp', 'henry_Hcp'),
            ('henry_B', 'henry_B'),
            ('henry_Tmin', 'henry_Tmin'),
            ('henry_Tmax', 'henry_Tmax'),
            ('henry_Vinf', 'henry_Vinf'),
            ('henry_Vinf_uncertainty', 'henry_Vinf_uncertainty'),
            ('Tb', 'Tb'),
            ('Tt', 'Tt'),
            ('Pt', 'Pt'),
            ('Tm', 'Tm'),
            ('Hf', 'Hf'),
            ('Gf', 'Gf'),
            ('S', 'S'),
            ('Hf_liquid', 'Hf_liquid'),
            ('Gf_liquid', 'Gf_liquid'),
            ('S_liquid', 'S_liquid'),
            ('Hf_solid', 'Hf_solid'),
            ('Gf_solid', 'Gf_solid'),
            ('S_solid', 'S_solid'),
            ('Hcomb', 'Hcomb'),
            ('Hcomb_gross', 'Hcomb_gross'),
            ('Hvap', 'Hvap'),
            ('Hfus', 'Hfus'),
            ('Cp_coeffs', 'Cp_coeffs'),
            ('Cp_liquid', 'Cp_liquid'),
            ('Cp_solid', 'Cp_solid'),
            ('rho_solid', 'rho_solid'),
            ('Vm_solid', 'Vm_solid'),
            ('solid_material_form', 'solid_material_form'),
            ('solid_polymorph', 'solid_polymorph'),
            ('antoine_A', 'antoine_A'),
            ('antoine_B', 'antoine_B'),
            ('antoine_C', 'antoine_C'),
            ('antoine_Tmin', 'antoine_Tmin'),
            ('antoine_Tmax', 'antoine_Tmax'),
            ('antoine_source', 'antoine_source'),
            ('vapor_dimerization', 'vapor_dimerization'),
            ('uniquac_r', 'uniquac_r'),
            ('uniquac_q', 'uniquac_q'),
            ('phase_at_STP', 'phase_at_STP'),
            ('critical_properties_unavailable', 'critical_properties_unavailable'),
        )

        def remember_pfd_override(props, attr: str) -> None:
            props.property_sources[attr] = {
                'source': 'provided',
                'method': 'pfd_component_override',
                'quality': 1.0,
                'notes': 'User-specified in .pfd component definition',
            }

        def apply_pfd_component_overrides(props, pfd_comp) -> None:
            ignored_mc_fields = []
            ignored_prsv_fields = []
            ignored_prsv2_fields = []
            ignored_twu_fields = []
            ignored_twu_vt_fields = []
            ignored_uniquac_fields = []
            ignored_vdm_fields = []
            for component_attr, property_attr in pfd_property_attrs:
                value = getattr(pfd_comp, component_attr, None)
                if value is not None:
                    if property_attr in {'mc_c1', 'mc_c2', 'mc_c3'} and not uses_mathias_copeman:
                        ignored_mc_fields.append(property_attr)
                        continue
                    if property_attr in {'kappa1', 'kappa2', 'kappa3'} and not (uses_prsv1 or uses_prsv2):
                        ignored_prsv_fields.append(property_attr)
                        continue
                    if (
                        property_attr in {'kappa2', 'kappa3'}
                        and uses_prsv1
                        and not uses_prsv2
                    ):
                        ignored_prsv2_fields.append(property_attr)
                        continue
                    if property_attr in {'twu_l', 'twu_m', 'twu_n'} and not uses_twu:
                        ignored_twu_fields.append(property_attr)
                        continue
                    if property_attr == 'twu_c':
                        ignored_twu_vt_fields.append(property_attr)
                        continue
                    if property_attr in {'uniquac_r', 'uniquac_q'} and not uses_uniquac:
                        ignored_uniquac_fields.append(property_attr)
                        continue
                    if property_attr == 'vapor_dimerization' and not uses_vdm:
                        ignored_vdm_fields.append(property_attr)
                        continue
                    if property_attr == 'vapor_dimerization':
                        value = normalize_vdm_component_parameters(value)
                    setattr(props, property_attr, value)
                    remember_pfd_override(props, property_attr)
            if ignored_mc_fields:
                ignored_model_parameter_warnings.append(
                    f"Ignoring Mathias-Copeman alpha parameters for {pfd_comp.symbol} "
                    f"({', '.join(ignored_mc_fields)}) because THERMO_METHOD "
                    f"{self.thermo_method} is not MC-based."
                )
            if ignored_prsv_fields:
                ignored_model_parameter_warnings.append(
                    f"Ignoring PRSV alpha parameters for {pfd_comp.symbol} "
                    f"({', '.join(ignored_prsv_fields)}) because THERMO_METHOD "
                    f"{self.thermo_method} is not PRSV-based."
                )
            if ignored_prsv2_fields:
                ignored_model_parameter_warnings.append(
                    f"Ignoring PRSV2 alpha parameters for {pfd_comp.symbol} "
                    f"({', '.join(ignored_prsv2_fields)}) because THERMO_METHOD "
                    f"{self.thermo_method} uses PRSV1."
                )
            if ignored_twu_fields:
                ignored_model_parameter_warnings.append(
                    f"Ignoring Twu alpha parameters for {pfd_comp.symbol} "
                    f"({', '.join(ignored_twu_fields)}) because THERMO_METHOD "
                    f"{self.thermo_method} is not Twu-based."
                )
            if ignored_twu_vt_fields:
                ignored_model_parameter_warnings.append(
                    f"Ignoring Twu volume-translation parameters for {pfd_comp.symbol} "
                    f"({', '.join(ignored_twu_vt_fields)}) because volume translation "
                    "is not currently applied."
                )
            if ignored_uniquac_fields:
                ignored_model_parameter_warnings.append(
                    f"Ignoring UNIQUAC pure-component parameters for {pfd_comp.symbol} "
                    f"({', '.join(ignored_uniquac_fields)}) because THERMO_METHOD "
                    f"{self.thermo_method} does not use UNIQUAC."
                )
            if ignored_vdm_fields:
                ignored_model_parameter_warnings.append(
                    f"Ignoring vapor-dimerization parameters for {pfd_comp.symbol} "
                    f"because THERMO_METHOD {self.thermo_method} is not VDM-based."
                )
            if pfd_comp.molecular_weight is not None:
                props.MW = pfd_comp.molecular_weight
                remember_pfd_override(props, 'MW')
            correlations = pfd_comp.resolver_property_correlations()
            if correlations:
                correlations = {
                    key: {**correlation, 'quality': 1.0, '_pfd_override': True}
                    for key, correlation in correlations.items()
                }
                props.property_correlations.update(correlations)
                props.property_sources['property_correlations'] = {
                    'source': 'provided',
                    'method': 'pfd_property_correlations',
                    'quality': 1.0,
                    'notes': 'User-specified in .pfd PROPERTY_CORRELATIONS or component density reference',
                }

        def lookup_pfd_component(pfd_comp, fetch_online: bool):
            return db.get(
                pfd_comp.identifier,
                fetch_online=fetch_online,
            )

        def formula_molecular_weight(formula: str) -> float | None:
            counts = parse_formula_counts(formula)
            if not counts:
                return None
            try:
                from chemicals.elements import periodic_table
                return sum(
                    float(periodic_table[element].MW) * count
                    for element, count in counts.items()
                )
            except (ImportError, KeyError, TypeError, ValueError):
                return None
        
        # Resolve components first so local/online lookup can fill gaps, then
        # apply PFD-provided values before thermodynamics constructors bind
        # critical properties, CAS identities, or UNIQUAC r/q parameters.
        for pfd_comp in self.pfd.components:
            structure_identifier = (
                get_compound_identity_resolver is not None
                and get_compound_identity_resolver().is_structure_identifier(
                    pfd_comp.identifier
                )
            )
            formula_identifier = (
                looks_like_formula(pfd_comp.identifier)
                and not structure_identifier
            )
            if formula_identifier and get_compound_identity_resolver is not None:
                identity_resolver = get_compound_identity_resolver()
                if identity_resolver.is_ambiguous_formula(pfd_comp.identifier):
                    raise SimulationError(
                        f"Component '{pfd_comp.symbol}' lookup identifier "
                        f"'{pfd_comp.identifier}' is an ambiguous molecular formula. "
                        "Use a unique name, CAS, SMILES, InChI, or InChIKey."
                    )
            existing = lookup_pfd_component(pfd_comp, fetch_online=False)
            if existing is None and allow_online_lookup:
                existing = lookup_pfd_component(pfd_comp, fetch_online=True)

            if (
                existing is not None
                and (
                    pfd_comp.symbol not in db.chemicals
                    or existing.symbol != pfd_comp.symbol
                )
            ):
                alias_data = existing.to_dict()
                alias_data['symbol'] = pfd_comp.symbol
                alias_data['name'] = existing.name
                alias_data['source'] = f"{existing.source}+pfd_alias"
                existing = ChemicalProperties(**alias_data)
                db.chemicals[pfd_comp.symbol] = existing

            if existing is None:
                inferred_formula = (
                    pfd_comp.identifier if formula_identifier else None
                )
                molecular_weight = pfd_comp.molecular_weight
                if molecular_weight is None and inferred_formula:
                    molecular_weight = formula_molecular_weight(inferred_formula)
                if molecular_weight is None:
                    raise SimulationError(
                        f"Component '{pfd_comp.symbol}' lookup identifier "
                        f"'{pfd_comp.identifier}' is not in the property "
                        "database and does not specify MW. Add MW=... for a custom component."
                    )
                # Create a new entry from PFD-defined properties
                new_props = ChemicalProperties(
                    symbol=pfd_comp.symbol,
                    name=pfd_comp.identifier,
                    formula=pfd_comp.formula or inferred_formula or pfd_comp.symbol,
                    CAS=pfd_comp.CAS or "",
                    smiles=pfd_comp.smiles,
                    MW=molecular_weight,
                    Cp_coeffs=pfd_comp.Cp_coeffs if pfd_comp.Cp_coeffs else [33, 0, 0, 0],
                    phase_at_STP=pfd_comp.phase_at_STP or 'unknown',
                    critical_properties_unavailable=bool(pfd_comp.critical_properties_unavailable),
                    source='pfd_defined'
                )
                existing = new_props
                db.chemicals[pfd_comp.symbol] = existing
                if formula_identifier:
                    ignored_model_parameter_warnings.append(
                        f"Component '{pfd_comp.symbol}' uses unambiguous formula "
                        f"identifier '{pfd_comp.identifier}'; treating it as a "
                        "custom component rather than selecting a database identity. "
                        "Use a unique name, CAS, SMILES, InChI, or InChIKey to "
                        "inherit known compound properties."
                    )

            apply_pfd_component_overrides(existing, pfd_comp)
            if (
                normalize_phase_behavior(pfd_comp.phase_behavior)
                != PERMANENT_SOLID_PHASE_BEHAVIOR
            ):
                db._hydrate_properties(existing, allow_online=False)
            for warning in getattr(existing, 'lookup_warnings', ()) or ():
                if warning not in ignored_model_parameter_warnings:
                    ignored_model_parameter_warnings.append(warning)

        def component_lookup_key(value) -> str:
            return str(value).strip().lower()

        interaction_component_lookup = {}
        for pfd_comp in self.pfd.components:
            props = db.get_user_component(pfd_comp.symbol)
            candidates = [
                pfd_comp.symbol,
                pfd_comp.name,
                getattr(props, 'symbol', None),
                getattr(props, 'name', None),
                getattr(props, 'CAS', None),
            ]
            for candidate in candidates:
                if candidate:
                    interaction_component_lookup.setdefault(
                        component_lookup_key(candidate),
                        pfd_comp.symbol,
                    )

        def resolve_interaction_component(identifier: str) -> str:
            key = component_lookup_key(identifier)
            if key in interaction_component_lookup:
                return interaction_component_lookup[key]
            raise SimulationError(
                f"INTERACTION_PARAMETERS references unknown component '{identifier}'. "
                "Use a component symbol, component name, or CAS from the PFD."
            )

        def numeric(value, field: str) -> float:
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise SimulationError(
                    f"INTERACTION_PARAMETERS field '{field}' must be numeric; got {value!r}."
                ) from exc

        def boolean(value) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in {'true', 'yes', '1', 'on'}
            return bool(value)

        def strict_boolean(value, field: str) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {'true', 'yes', '1', 'on'}:
                    return True
                if normalized in {'false', 'no', '0', 'off'}:
                    return False
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            raise SimulationError(
                f"INTERACTION_PARAMETERS field '{field}' must be boolean; got {value!r}."
            )

        def normalize_interaction_parameters() -> list[dict]:
            overrides = []
            activity_override_keys = set()
            viscosity_static_records: dict[tuple, dict] = {}
            viscosity_ranged_records: dict[tuple, list[dict]] = {}
            eos_static_records: dict[tuple, dict] = {}
            eos_ranged_records: dict[tuple, list[dict]] = {}

            def unordered_pair_key(
                scope: str,
                model: str,
                comp1: str,
                comp2: str,
            ) -> tuple:
                return (scope, model, tuple(sorted((comp1, comp2))))

            def eos_effective_range(record: dict) -> tuple[float, float] | None:
                if 'Tmin_K' not in record or 'Tmax_K' not in record:
                    return None
                low = float(record['Tmin_K'])
                high = float(record['Tmax_K'])
                if low > high:
                    low, high = high, low
                if low == high:
                    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                        from .interaction_parameters import EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K
                    else:
                        from interaction_parameters import EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K
                    return (
                        low - EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K,
                        high + EOS_SINGLE_TEMPERATURE_HALF_WIDTH_K,
                    )
                return low, high

            def eos_records_equivalent(left: dict, right: dict) -> bool:
                keys = ('kij', 'kij_a', 'kij_b', 'kij_c', 'T_ref_K', 'Tmin_K', 'Tmax_K')
                for key in keys:
                    left_has = key in left
                    right_has = key in right
                    if left_has != right_has:
                        return False
                    if left_has and abs(float(left[key]) - float(right[key])) > 1e-12:
                        return False
                return True

            def eos_ranges_overlap(left: dict, right: dict) -> bool:
                left_range = eos_effective_range(left)
                right_range = eos_effective_range(right)
                if left_range is None or right_range is None:
                    return False
                return max(left_range[0], right_range[0]) <= min(left_range[1], right_range[1]) + 1e-12

            def remember_activity_override(
                scope: str,
                model: str,
                comp1: str,
                comp2: str,
            ) -> None:
                key = unordered_pair_key(scope, model, comp1, comp2)
                if key in activity_override_keys:
                    raise SimulationError(
                        f"Duplicate {model} INTERACTION_PARAMETERS override for {comp1}/{comp2}."
                    )
                activity_override_keys.add(key)

            def remember_eos_override(record: dict) -> bool:
                key = unordered_pair_key(
                    record['scope'],
                    record['model'],
                    record['component1'],
                    record['component2'],
                )
                if eos_effective_range(record) is None:
                    existing = eos_static_records.get(key)
                    if existing is not None:
                        if eos_records_equivalent(existing, record):
                            return False
                        raise SimulationError(
                            f"Conflicting static {record['model']} INTERACTION_PARAMETERS overrides "
                            f"for {record['component1']}/{record['component2']}."
                        )
                    eos_static_records[key] = record
                    return True

                records = eos_ranged_records.setdefault(key, [])
                for existing in records:
                    if not eos_ranges_overlap(existing, record):
                        continue
                    if eos_records_equivalent(existing, record):
                        return False
                    raise SimulationError(
                        f"Overlapping {record['model']} INTERACTION_PARAMETERS temperature ranges "
                        f"for {record['component1']}/{record['component2']}."
                    )
                records.append(record)
                return True

            def viscosity_effective_range(record: dict) -> tuple[float, float] | None:
                if 'Tmin_K' not in record and 'Tmax_K' not in record:
                    return None
                low = float(record.get('Tmin_K', float('-inf')))
                high = float(record.get('Tmax_K', float('inf')))
                if low > high:
                    low, high = high, low
                return low, high

            def viscosity_records_equivalent(left: dict, right: dict) -> bool:
                keys = (
                    'viscosity_form', 'G', 'excess_g_over_rt',
                    'A0_K', 'A1_K', 'A2_K',
                    'A', 'B', 'C', 'D', 'E', 'F',
                    'T_ref_K', 'Tmin_K', 'Tmax_K',
                )
                for key in keys:
                    left_has = key in left
                    right_has = key in right
                    if left_has != right_has:
                        return False
                    if not left_has:
                        continue
                    if key == 'viscosity_form':
                        if str(left[key]) != str(right[key]):
                            return False
                    elif abs(float(left[key]) - float(right[key])) > 1e-12:
                        return False
                return True

            def viscosity_ranges_overlap(left: dict, right: dict) -> bool:
                left_range = viscosity_effective_range(left)
                right_range = viscosity_effective_range(right)
                if left_range is None or right_range is None:
                    return False
                return max(left_range[0], right_range[0]) <= min(left_range[1], right_range[1]) + 1e-12

            def remember_viscosity_override(record: dict) -> bool:
                key = unordered_pair_key(
                    record['scope'],
                    record['model'],
                    record['component1'],
                    record['component2'],
                )
                if viscosity_effective_range(record) is None:
                    existing = viscosity_static_records.get(key)
                    if existing is not None:
                        if viscosity_records_equivalent(existing, record):
                            return False
                        raise SimulationError(
                            f"Conflicting static {record['model']} INTERACTION_PARAMETERS overrides "
                            f"for {record['component1']}/{record['component2']}."
                        )
                    if key in viscosity_ranged_records:
                        raise SimulationError(
                            f"Static {record['model']} INTERACTION_PARAMETERS override conflicts "
                            f"with ranged overrides for {record['component1']}/{record['component2']}."
                        )
                    viscosity_static_records[key] = record
                    return True

                if key in viscosity_static_records:
                    existing = viscosity_static_records[key]
                    if viscosity_records_equivalent(existing, record):
                        return False
                    raise SimulationError(
                        f"Ranged {record['model']} INTERACTION_PARAMETERS override conflicts "
                        f"with static override for {record['component1']}/{record['component2']}."
                    )
                records = viscosity_ranged_records.setdefault(key, [])
                for existing in records:
                    if not viscosity_ranges_overlap(existing, record):
                        continue
                    if viscosity_records_equivalent(existing, record):
                        return False
                    raise SimulationError(
                        f"Overlapping {record['model']} INTERACTION_PARAMETERS temperature ranges "
                        f"for {record['component1']}/{record['component2']}."
                    )
                records.append(record)
                return True

            for item in getattr(self.pfd, 'interaction_parameters', []):
                model = normalize_interaction_model(item.model)
                scope = str(item.scope or 'global')
                comp1 = resolve_interaction_component(item.component1)
                comp2 = resolve_interaction_component(item.component2)
                if comp1 == comp2:
                    raise SimulationError(
                        f"INTERACTION_PARAMETERS cannot override self-interaction for '{comp1}'."
                    )
                raw = dict(item.parameters or {})
                params = {
                    str(key).strip().lower().replace('-', '_'): value
                    for key, value in raw.items()
                }
                record = {
                    'component1': comp1,
                    'component2': comp2,
                    'model': model,
                    'scope': scope,
                    'comment': str(params.get('comment') or 'PFD interaction override'),
                }

                if model in {'PR', 'SRK'}:
                    if 'k_ij' in params and 'kij' not in params:
                        params['kij'] = params['k_ij']
                    for source, target in (
                        ('tmin', 'Tmin_K'),
                        ('tmax', 'Tmax_K'),
                        ('tmin_k', 'Tmin_K'),
                        ('tmax_k', 'Tmax_K'),
                        ('t_ref', 'T_ref_K'),
                        ('t_ref_k', 'T_ref_K'),
                        ('tref', 'T_ref_K'),
                        ('tref_k', 'T_ref_K'),
                    ):
                        if source in params:
                            record[target] = numeric(params[source], source)
                    for key in ('kij', 'kij_a', 'kij_b', 'kij_c'):
                        if key in params:
                            record[key] = numeric(params[key], key)
                    if 'kij' not in record and not any(key in record for key in ('kij_a', 'kij_b', 'kij_c')):
                        raise SimulationError(
                            "EOS INTERACTION_PARAMETERS require kij=... or kij_a/kij_b/kij_c."
                        )
                    if ('Tmin_K' in record) != ('Tmax_K' in record):
                        raise SimulationError(
                            "EOS INTERACTION_PARAMETERS temperature ranges require both Tmin_K and Tmax_K."
                        )
                    if remember_eos_override(record):
                        overrides.append(record)
                    continue

                if model == 'NRTL':
                    remember_activity_override(scope, model, comp1, comp2)
                    record['do_not_extrapolate'] = strict_boolean(
                        params.get('do_not_extrapolate', False),
                        'do_not_extrapolate',
                    )
                    for aliases, target in (
                        (('tmin', 'tmin_k'), 'Tmin_K'),
                        (('tmax', 'tmax_k'), 'Tmax_K'),
                    ):
                        present = [name for name in aliases if name in params]
                        if len(present) > 1:
                            raise SimulationError(
                                f"NRTL INTERACTION_PARAMETERS specifies duplicate aliases for {target}."
                            )
                        if present:
                            record[target] = numeric(params[present[0]], present[0])
                    if record['do_not_extrapolate'] and (
                        'Tmin_K' not in record or 'Tmax_K' not in record
                    ):
                        raise SimulationError(
                            "NRTL do_not_extrapolate=true requires Tmin_K and Tmax_K."
                        )
                    if 'Tmin_K' in record and 'Tmax_K' in record and not (
                        math.isfinite(record['Tmin_K'])
                        and math.isfinite(record['Tmax_K'])
                        and 0.0 < record['Tmin_K'] < record['Tmax_K']
                    ):
                        raise SimulationError(
                            "NRTL INTERACTION_PARAMETERS requires 0 < Tmin_K < Tmax_K."
                        )
                    alpha = params.get('alpha12', params.get('alpha'))
                    record['alpha12'] = numeric(alpha if alpha is not None else 0.3, 'alpha')
                    has_tau = any(key in params for key in (
                        'tau12_c', 'tau12_d', 'tau12_e', 'tau12_f', 'tau12_g',
                        'tau21_c', 'tau21_d', 'tau21_e', 'tau21_f', 'tau21_g',
                    ))
                    if has_tau:
                        if 'tau12_c' not in params or 'tau21_c' not in params:
                            raise SimulationError(
                                "NRTL tau-form overrides require at least tau12_c and tau21_c."
                            )
                        for key in (
                            'tau12_c', 'tau12_d', 'tau12_e', 'tau12_f', 'tau12_g',
                            'tau21_c', 'tau21_d', 'tau21_e', 'tau21_f', 'tau21_g',
                        ):
                            record[key] = numeric(params.get(key, 0.0), key)
                        record['tau_tref'] = numeric(params.get('tau_tref', params.get('tref', 298.15)), 'tau_tref')
                    else:
                        a12 = params.get('a12_cal_per_mol', params.get('a12'))
                        a21 = params.get('a21_cal_per_mol', params.get('a21'))
                        if a12 is None or a21 is None:
                            raise SimulationError(
                                "NRTL scalar overrides require a12/a21 or tau12_c/tau21_c."
                            )
                        record['a12_cal_per_mol'] = numeric(a12, 'a12')
                        record['a21_cal_per_mol'] = numeric(a21, 'a21')
                    overrides.append(record)
                    continue

                if model == 'UNIQUAC':
                    remember_activity_override(scope, model, comp1, comp2)
                    record['do_not_extrapolate'] = strict_boolean(
                        params.get('do_not_extrapolate', False),
                        'do_not_extrapolate',
                    )
                    for aliases, target in (
                        (('tmin', 'tmin_k'), 'Tmin_K'),
                        (('tmax', 'tmax_k'), 'Tmax_K'),
                    ):
                        present = [name for name in aliases if name in params]
                        if len(present) > 1:
                            raise SimulationError(
                                f"UNIQUAC INTERACTION_PARAMETERS specifies duplicate aliases for {target}."
                            )
                        if present:
                            record[target] = numeric(params[present[0]], present[0])
                    if record['do_not_extrapolate'] and (
                        'Tmin_K' not in record or 'Tmax_K' not in record
                    ):
                        raise SimulationError(
                            "UNIQUAC do_not_extrapolate=true requires Tmin_K and Tmax_K."
                        )
                    if 'Tmin_K' in record and 'Tmax_K' in record and not (
                        math.isfinite(record['Tmin_K'])
                        and math.isfinite(record['Tmax_K'])
                        and 0.0 < record['Tmin_K'] < record['Tmax_K']
                    ):
                        raise SimulationError(
                            "UNIQUAC INTERACTION_PARAMETERS requires 0 < Tmin_K < Tmax_K."
                        )
                    record['model_variant'] = str(params.get('model_variant') or 'standard_uniquac')
                    record['use_q_prime'] = boolean(params.get('use_q_prime', False))
                    uniquac_tau_fields = (
                        'tau12_a', 'tau12_b', 'tau12_c', 'tau12_d', 'tau12_e',
                        'tau21_a', 'tau21_b', 'tau21_c', 'tau21_d', 'tau21_e',
                    )
                    has_tau = any(key in params for key in uniquac_tau_fields)
                    if has_tau:
                        if 'tau12_a' not in params or 'tau21_a' not in params:
                            raise SimulationError(
                                "UNIQUAC tau-form overrides require at least tau12_a and tau21_a."
                            )
                        for key in uniquac_tau_fields:
                            record[key] = numeric(params.get(key, 0.0), key)
                        record['tau_tref'] = numeric(
                            params.get('tau_tref', params.get('tref', 298.15)),
                            'tau_tref',
                        )
                    else:
                        a12 = params.get('a12_cal_per_mol', params.get('a12'))
                        a21 = params.get('a21_cal_per_mol', params.get('a21'))
                        if a12 is None or a21 is None:
                            raise SimulationError(
                                "UNIQUAC scalar overrides require a12/a21 or tau12_a/tau21_a."
                            )
                        record['a12_cal_per_mol'] = numeric(a12, 'a12')
                        record['a21_cal_per_mol'] = numeric(a21, 'a21')
                    overrides.append(record)
                    continue

                if model == 'LIQUID_VISCOSITY':
                    for source, target in (
                        ('tmin', 'Tmin_K'),
                        ('tmax', 'Tmax_K'),
                        ('tmin_k', 'Tmin_K'),
                        ('tmax_k', 'Tmax_K'),
                        ('t_ref', 'T_ref_K'),
                        ('t_ref_k', 'T_ref_K'),
                        ('tref', 'T_ref_K'),
                        ('tref_k', 'T_ref_K'),
                    ):
                        if source in params:
                            record[target] = numeric(params[source], source)
                    form = str(params.get('form', params.get('viscosity_form', 'grunberg_nissan')))
                    form_key = form.strip().lower().replace('-', '_')
                    form_aliases = {
                        'gn': 'grunberg_nissan',
                        'grunberg': 'grunberg_nissan',
                        'grunberg_nissan': 'grunberg_nissan',
                        'constant': 'grunberg_nissan',
                        'constant_excess': 'grunberg_nissan',
                        'jouyban': 'jouyban_acree',
                        'ja': 'jouyban_acree',
                        'jouyban_acree': 'jouyban_acree',
                        'poly': 'excess_poly',
                        'polynomial': 'excess_poly',
                        'excess_poly': 'excess_poly',
                    }
                    if form_key not in form_aliases:
                        raise SimulationError(
                            f"Unsupported LIQUID_VISCOSITY form '{form}'. "
                            "Supported forms are grunberg_nissan, jouyban_acree, and excess_poly."
                        )
                    record['viscosity_form'] = form_aliases[form_key]
                    if record['viscosity_form'] == 'grunberg_nissan':
                        G = params.get('g', params.get('g12', params.get('excess_g_over_rt')))
                        if G is None:
                            raise SimulationError(
                                "LIQUID_VISCOSITY grunberg_nissan overrides require G=..."
                            )
                        record['G'] = numeric(G, 'G')
                    elif record['viscosity_form'] == 'jouyban_acree':
                        A0 = params.get('a0_k', params.get('a0'))
                        if A0 is None:
                            raise SimulationError(
                                "LIQUID_VISCOSITY jouyban_acree overrides require A0_K=..."
                            )
                        record['A0_K'] = numeric(A0, 'A0_K')
                        record['A1_K'] = numeric(params.get('a1_k', params.get('a1', 0.0)), 'A1_K')
                        record['A2_K'] = numeric(params.get('a2_k', params.get('a2', 0.0)), 'A2_K')
                    else:
                        if not any(key in params for key in ('a', 'b', 'c', 'd', 'e', 'f')):
                            raise SimulationError(
                                "LIQUID_VISCOSITY excess_poly overrides require at least one of A-F."
                            )
                        for key in ('a', 'b', 'c', 'd', 'e', 'f'):
                            record[key.upper()] = numeric(params.get(key, 0.0), key)
                        if 'T_ref_K' not in record:
                            record['T_ref_K'] = numeric(
                                params.get(
                                    't_ref_k',
                                    params.get(
                                        'tref_k',
                                        params.get('t_ref', params.get('tref', 298.15)),
                                    ),
                                ),
                                'T_ref_K',
                            )
                    if remember_viscosity_override(record):
                        overrides.append(record)
                    continue

                if model == 'VDM':
                    remember_activity_override(scope, model, comp1, comp2)
                    try:
                        record.update(normalize_vdm_cross_parameters(raw))
                    except ValueError as error:
                        raise SimulationError(str(error)) from error
                    overrides.append(record)
                    continue

                raise SimulationError(
                    f"Unsupported INTERACTION_PARAMETERS model '{item.model}'. "
                    "Supported models are NRTL, UNIQUAC, PR, SRK, "
                    "LIQUID_VISCOSITY, and VDM."
                )
            return overrides

        interaction_overrides = normalize_interaction_parameters()

        def interaction_estimation_temperature(value, field: str) -> float:
            if isinstance(value, (int, float)):
                result = float(value)
            else:
                text = str(value).strip()
                match = re.fullmatch(
                    r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
                    r'\s*(?:\[\s*([^\]]+)\s*\])?',
                    text,
                )
                if match is None:
                    raise SimulationError(
                        f"INTERACTION_ESTIMATION field '{field}' must be a "
                        f"temperature; got {value!r}."
                    )
                result = float(match.group(1))
                unit = str(match.group(2) or 'K').strip().lower()
                if unit in {'k', 'kelvin'}:
                    pass
                elif unit in {'c', 'degc', 'celsius'}:
                    result += 273.15
                elif unit in {'f', 'degf', 'fahrenheit'}:
                    result = (result - 32.0) * 5.0 / 9.0 + 273.15
                else:
                    raise SimulationError(
                        f"INTERACTION_ESTIMATION field '{field}' has unsupported "
                        f"temperature unit '{match.group(2)}'."
                    )
            if not math.isfinite(result) or result <= 0.0:
                raise SimulationError(
                    f"INTERACTION_ESTIMATION field '{field}' must be a positive "
                    "finite absolute temperature."
                )
            return result

        def normalize_interaction_estimation() -> list[dict]:
            source_aliases = {
                'UNIFAC': 'UNIFAC',
                'ORIGINAL-UNIFAC': 'UNIFAC',
                'ORIGINAL_UNIFAC': 'UNIFAC',
                'UNIFAC2': 'UNIFAC2',
                'UNIFAC-2': 'UNIFAC2',
                'UNIFDMD': 'UNIFDMD',
                'UNIFAC-DMD': 'UNIFDMD',
                'UNIFAC_DMD': 'UNIFDMD',
                'DORTMUND-UNIFAC': 'UNIFDMD',
                'DORTMUND_UNIFAC': 'UNIFDMD',
                'MODIFIED-UNIFAC': 'UNIFDMD',
                'MODIFIED_UNIFAC': 'UNIFDMD',
                'UNIFM2': 'UNIFM2',
                'UNIFAC-M2': 'UNIFM2',
                'UNIFAC_M2': 'UNIFM2',
                'UNIFNIST': 'UNIFNIST',
                'UNIFAC-NIST': 'UNIFNIST',
                'UNIFAC_NIST': 'UNIFNIST',
                'NIST-UNIFAC': 'UNIFNIST',
                'NIST_UNIFAC': 'UNIFNIST',
            }
            normalized = []
            global_models = set()
            pair_keys = set()
            for item in getattr(self.pfd, 'interaction_estimation', []):
                model = normalize_interaction_model(item.model)
                scope = str(item.scope or 'global')
                if model not in {'NRTL', 'UNIQUAC'}:
                    raise SimulationError(
                        "INTERACTION_ESTIMATION destination model must be NRTL "
                        "or UNIQUAC."
                    )
                raw = {
                    str(key).strip().lower().replace('-', '_'): value
                    for key, value in dict(item.parameters or {}).items()
                }
                record = {'model': model, 'scope': scope}
                pair_specific = item.component1 is not None or item.component2 is not None
                if pair_specific:
                    if item.component1 is None or item.component2 is None:
                        raise SimulationError(
                            "Pair-specific INTERACTION_ESTIMATION rules require two components."
                        )
                    comp1 = resolve_interaction_component(item.component1)
                    comp2 = resolve_interaction_component(item.component2)
                    if comp1 == comp2:
                        raise SimulationError(
                            "INTERACTION_ESTIMATION cannot target self-interaction "
                            f"for '{comp1}'."
                        )
                    key = (scope, model, tuple(sorted((comp1, comp2))))
                    if key in pair_keys:
                        raise SimulationError(
                            f"Duplicate {model} INTERACTION_ESTIMATION override "
                            f"for {comp1}/{comp2}."
                        )
                    pair_keys.add(key)
                    record.update(component1=comp1, component2=comp2)
                else:
                    global_key = (scope, model)
                    if global_key in global_models:
                        raise SimulationError(
                            f"Duplicate global {model} INTERACTION_ESTIMATION rule."
                        )
                    global_models.add(global_key)

                if 'source' in raw:
                    source_key = str(raw['source']).strip().upper().replace(' ', '-')
                    if source_key not in source_aliases:
                        raise SimulationError(
                            f"Unsupported INTERACTION_ESTIMATION source '{raw['source']}'. "
                            "Supported sources are UNIFAC, UNIFAC2, UNIFDMD, "
                            "UNIFM2, and UNIFNIST."
                        )
                    record['source'] = source_aliases[source_key]
                if 'policy' in raw:
                    policy = str(raw['policy']).strip().lower().replace('-', '_')
                    if policy not in {'missing_only', 'always'}:
                        raise SimulationError(
                            "INTERACTION_ESTIMATION policy must be missing_only "
                            "or always."
                        )
                    record['policy'] = policy
                if 'parameter_order' in raw:
                    order = str(raw['parameter_order']).strip().lower().replace('-', '_')
                    if order != 'source':
                        raise SimulationError(
                            "INTERACTION_ESTIMATION currently supports only "
                            "parameter_order=source."
                        )
                    record['parameter_order'] = order
                if 'do_not_extrapolate' in raw:
                    record['do_not_extrapolate'] = strict_boolean(
                        raw['do_not_extrapolate'],
                        'do_not_extrapolate',
                    )
                if 'alpha' in raw or 'alpha12' in raw:
                    if model != 'NRTL':
                        raise SimulationError(
                            "INTERACTION_ESTIMATION alpha is valid only for NRTL."
                        )
                    alpha = numeric(raw.get('alpha12', raw.get('alpha')), 'alpha')
                    if not 0.0 < alpha <= 1.0:
                        raise SimulationError(
                            "NRTL INTERACTION_ESTIMATION alpha must satisfy 0 < alpha <= 1."
                        )
                    record['alpha12'] = alpha
                for aliases, target in (
                    (('tmin', 'tmin_k'), 'Tmin_K'),
                    (('tmax', 'tmax_k'), 'Tmax_K'),
                    (('t_ref', 't_ref_k', 'tref', 'tref_k'), 'T_ref_K'),
                ):
                    present = [name for name in aliases if name in raw]
                    if len(present) > 1:
                        raise SimulationError(
                            f"INTERACTION_ESTIMATION specifies duplicate aliases "
                            f"for {target}: {', '.join(present)}."
                        )
                    if present:
                        record[target] = interaction_estimation_temperature(
                            raw[present[0]], present[0]
                        )
                if 'comment' in raw:
                    record['comment'] = str(raw['comment'])
                normalized.append(record)

            for record in normalized:
                if (
                    'component1' in record
                    and not any(
                        (ancestor, record['model']) in global_models
                        for ancestor in thermo_scope_lineage(record['scope'])
                    )
                ):
                    raise SimulationError(
                        f"Pair-specific {record['model']} INTERACTION_ESTIMATION "
                        "requires a global rule for that destination model."
                    )
                if 'component1' not in record and 'source' not in record:
                    raise SimulationError(
                        f"Global {record['model']} INTERACTION_ESTIMATION rule "
                        "requires source=... ."
                    )
                if 'component1' not in record:
                    if 'Tmin_K' not in record or 'Tmax_K' not in record:
                        raise SimulationError(
                            f"Global {record['model']} INTERACTION_ESTIMATION rule "
                            "requires Tmin and Tmax."
                        )
                    if record['Tmax_K'] <= record['Tmin_K']:
                        raise SimulationError(
                            f"Global {record['model']} INTERACTION_ESTIMATION rule "
                            "requires Tmin < Tmax."
                        )
                    record.setdefault('policy', 'missing_only')
                    record.setdefault('parameter_order', 'source')
                    if record['model'] == 'NRTL':
                        record.setdefault('alpha12', 0.3)
            global_by_model = {
                (record['scope'], record['model']): record
                for record in normalized
                if 'component1' not in record
            }
            for record in normalized:
                if 'component1' not in record:
                    continue
                inherited_global = next(
                    global_by_model[(ancestor, record['model'])]
                    for ancestor in reversed(
                        thermo_scope_lineage(record['scope'])
                    )
                    if (ancestor, record['model']) in global_by_model
                )
                merged = dict(inherited_global)
                merged.update(record)
                if merged['Tmax_K'] <= merged['Tmin_K']:
                    raise SimulationError(
                        f"{record['model']} INTERACTION_ESTIMATION override for "
                        f"{record['component1']}/{record['component2']} requires "
                        "inherited/overridden Tmin < Tmax."
                    )
            return normalized

        interaction_estimation = normalize_interaction_estimation()
        # Compile fluid backends only from conventional components. Permanent
        # solids remain process components and are attached after construction.
        components = [c.symbol for c in self.pfd.components]
        permanent_solid_components = [
            component.symbol
            for component in self.pfd.components
            if normalize_phase_behavior(component.phase_behavior)
            == PERMANENT_SOLID_PHASE_BEHAVIOR
        ]
        permanent_solid_set = set(permanent_solid_components)
        conventional_solid_components = [
            component.symbol
            for component in self.pfd.components
            if normalize_phase_behavior(component.phase_behavior)
            == CONVENTIONAL_WITH_SOLID_PHASE_BEHAVIOR
        ]
        fluid_components = [
            component for component in components
            if component not in permanent_solid_set
        ]
        particle_defaults = {}
        solid_enabled_set = permanent_solid_set | set(
            conventional_solid_components
        )
        for component in self.pfd.components:
            if component.symbol not in solid_enabled_set:
                continue
            values = {'sphericity': float(component.particle_sphericity or 1.0)}
            if component.particle_diameter is not None:
                values['diameter_m'] = float(component.particle_diameter)
            if component.particle_size_distribution is not None:
                values['particle_size_distribution'] = dict(
                    component.particle_size_distribution
                )
            particle_defaults[component.symbol] = values
        
        # Collect UNIFAC groups if specified in PFD
        unifac_groups = None
        if any(
            method.startswith(('UNIFAC', 'UNIFDMD', 'UNIFM2', 'UNIFNIST'))
            for method in self.thermo_scope_methods.values()
        ) or interaction_estimation:
            unifac_groups = {}
            for pfd_comp in self.pfd.components:
                if pfd_comp.symbol in permanent_solid_set:
                    continue
                if pfd_comp.unifac_groups:
                    unifac_groups[pfd_comp.symbol] = pfd_comp.unifac_groups
                elif pfd_comp.smiles and self.thermo_method.upper().startswith('UNIF'):
                    # Try to parse SMILES to get UNIFAC groups
                    try:
                        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                            from .unifac import parse_smiles_to_unifac
                        else:
                            from unifac import parse_smiles_to_unifac
                        variant = self.thermo_method.upper().split('-', 1)[0]
                        groups = parse_smiles_to_unifac(pfd_comp.smiles, variant)
                        if groups:
                            unifac_groups[pfd_comp.symbol] = groups
                    except Exception:
                        pass  # Will try to look up from known molecules
        
        def effective_scoped_records(records: list[dict], scope: str) -> list[dict]:
            """Return isolated/inherited records with nearest-scope precedence."""
            effective = {}
            for level in thermo_scope_lineage(scope):
                local = {}
                for record in records:
                    if record.get('scope', 'global') != level:
                        continue
                    pair = tuple(sorted((
                        str(record.get('component1') or ''),
                        str(record.get('component2') or ''),
                    )))
                    key = (str(record.get('model') or ''), pair)
                    local.setdefault(key, []).append(record)
                for key, values in local.items():
                    effective[key] = values
            return [
                dict(record)
                for values in effective.values()
                for record in values
            ]

        def unit_scope_name(unit) -> str:
            return next((
                str(param.value).strip()
                for param in unit.params
                if str(param.name).strip().lower() == 'thermo_scope'
            ), 'global')

        units_by_scope = {
            scope: [
                unit for unit in self.pfd.units
                if unit_scope_name(unit) == scope
            ]
            for scope in self.thermo_scope_methods
        }

        def backend_needs(scope: str) -> tuple[bool, bool]:
            units = units_by_scope[scope]
            unit_types = {unit.unit_type for unit in units}
            need_vlle = (
                selected_phase_model != 'VLE'
                or 'Flash3' in unit_types
            )
            need_lle = need_vlle or bool(unit_types & {
                'Decanter',
                'ShortcutExtractor',
                'RigorousExtractor',
            })
            for unit in units:
                if unit.unit_type != 'RigorousDistillation':
                    continue
                params = {
                    str(param.name).strip().lower(): str(param.value).strip().lower()
                    for param in unit.params
                }
                stage_model = params.get('stage_phase_model', 'vle')
                condenser = params.get('condenser_type', 'total')
                if stage_model not in {'vle', ''}:
                    need_vlle = True
                    need_lle = True
                if condenser in {
                    'decanter', 'heterogeneous', 'heterogeneous_decanter',
                    'top_decanter',
                }:
                    need_lle = True
            return need_lle, need_vlle

        def refresh_thermo_property_snapshot(
            thermo,
            comp_symbol: str,
            props,
        ) -> None:
            known = props.to_dict()
            known.setdefault('antoine_source', 'PFD component definition')
            known['_allow_online_lookup'] = allow_online_lookup
            if psat_minimum_pressure_bar is not None:
                known['_psat_minimum_pressure_bar'] = (
                    psat_minimum_pressure_bar
                )
            if hasattr(thermo, '_resolver_known_props'):
                thermo._resolver_known_props[comp_symbol] = known
            for cache_name in (
                '_psat_cache', '_psat_coefficients_cache',
                '_cp_ideal_cache', '_cp_liquid_cache',
                '_cp_solid_cache', '_cp_integral_cache', '_enthalpy_ideal_cache',
                '_enthalpy_liquid_cache', '_enthalpy_solid_cache',
                '_enthalpy_process_solid_cache',
                '_entropy_ideal_cache', '_entropy_liquid_cache',
                '_entropy_solid_cache', '_hvap_cache', '_hvap_T_cache',
                '_entropy_process_solid_cache',
                '_liquid_molar_volume_cache', '_liquid_molar_volume_info_cache',
                '_solid_molar_volume_cache',
                '_pure_saturation_temperature_cache', '_phi_sat_cache',
                '_poynting_cache', '_k_values_cache',
                '_henry_component_data_cache',
                '_aqueous_solvent_concentration_cache',
                '_aqueous_solvent_density_derivative_cache',
                '_ideal_gas_cp_kernels',
                '_liquid_cp_kernels',
                '_solid_cp_kernels',
            ):
                cache = getattr(thermo, cache_name, None)
                if isinstance(cache, dict):
                    cache.clear()
            if hasattr(thermo, '_provided_cp_coeffs'):
                if known.get('Cp_coeffs'):
                    thermo._provided_cp_coeffs[comp_symbol] = list(known['Cp_coeffs'])
                else:
                    thermo._provided_cp_coeffs.pop(comp_symbol, None)
            if hasattr(props, '_ideal_gas_cp_kernel'):
                props._ideal_gas_cp_kernel = None
            sources = getattr(thermo, '_liquid_molar_volume_sources', None)
            if isinstance(sources, dict):
                sources[comp_symbol] = [
                    source
                    for source in sources.get(comp_symbol, [])
                    if source.get('kind') != 'provided_rhol'
                ]
            if (
                comp_symbol not in permanent_solid_set
                and hasattr(thermo, '_prebind_provided_liquid_molar_volume_source')
            ):
                thermo._prebind_provided_liquid_molar_volume_source(
                    comp_symbol,
                    known,
                )

        self.thermo_packages = {}
        try:
            for scope, method in self.thermo_scope_methods.items():
                package_warnings = list(ignored_model_parameter_warnings)
                scoped_overrides = effective_scoped_records(
                    interaction_overrides,
                    scope,
                )
                retained_overrides = []
                for record in scoped_overrides:
                    if (
                        record.get('model') == 'VDM'
                        and not method.endswith('-VDM')
                    ):
                        package_warnings.append(
                            "Ignoring VDM cross-interaction parameters for "
                            f"{record['component1']}/{record['component2']} "
                            f"because THERMO_METHOD {method} is not VDM-based."
                        )
                        continue
                    if (
                        record.get('component1') in permanent_solid_set
                        or record.get('component2') in permanent_solid_set
                    ):
                        package_warnings.append(
                            "Ignoring fluid interaction parameters for "
                            "permanent-solid pair "
                            f"{record.get('component1')}/"
                            f"{record.get('component2')}."
                        )
                        continue
                    retained_overrides.append(record)
                scoped_overrides = retained_overrides

                candidate_estimation = effective_scoped_records(
                    interaction_estimation,
                    scope,
                )
                active_estimation_model = None
                if method.startswith('NRTL'):
                    active_estimation_model = 'NRTL'
                elif method.startswith('UNIQUAC'):
                    active_estimation_model = 'UNIQUAC'
                scoped_estimation = []
                for record in candidate_estimation:
                    if record['model'] != active_estimation_model:
                        package_warnings.append(
                            f"Ignoring {record['model']} interaction-estimation "
                            f"rule because THERMO_METHOD {method} does not use "
                            f"{record['model']} liquid activity coefficients."
                        )
                        continue
                    if (
                        record.get('component1') in permanent_solid_set
                        or record.get('component2') in permanent_solid_set
                    ):
                        package_warnings.append(
                            "Ignoring interaction-estimation override for "
                            "permanent-solid pair "
                            f"{record.get('component1')}/"
                            f"{record.get('component2')}."
                        )
                        continue
                    scoped_estimation.append(record)
                constructor_overrides = []
                for record in scoped_overrides:
                    clean = dict(record)
                    clean.pop('scope', None)
                    constructor_overrides.append(clean)
                constructor_estimation = []
                for record in scoped_estimation:
                    clean = dict(record)
                    clean.pop('scope', None)
                    constructor_estimation.append(clean)

                if fluid_components:
                    thermo = create_thermodynamics(
                        fluid_components,
                        method,
                        db,
                        unifac_groups,
                        constructor_overrides,
                        constructor_estimation,
                    )
                else:
                    thermo = IdealThermodynamics([], db, [])
                    if method != 'IDEAL':
                        package_warnings.append(
                            f"Thermodynamic scope '{scope}' method {method} has "
                            "no fluid components; using the permanent-solid "
                            "property layer only."
                        )
                thermo.configure_permanent_solids(
                    components,
                    permanent_solid_components,
                    particle_defaults,
                    conventional_solid_components=conventional_solid_components,
                )
                thermo.set_fluid_phase_model(
                    getattr(self.pfd.metadata, 'fluid_phase_model', 'VLE')
                )
                for pfd_comp in self.pfd.components:
                    if pfd_comp.symbol in thermo.props:
                        refresh_thermo_property_snapshot(
                            thermo,
                            pfd_comp.symbol,
                            thermo.props[pfd_comp.symbol],
                        )
                thermo_warnings = getattr(thermo, 'warnings', None)
                if isinstance(thermo_warnings, list):
                    for warning in package_warnings:
                        scoped_warning = (
                            warning
                            if scope == 'global'
                            else f"[{scope}] {warning}"
                        )
                        if scoped_warning not in thermo_warnings:
                            thermo_warnings.append(scoped_warning)

                initialize_thermo = getattr(thermo, 'initialize', None)
                if callable(initialize_thermo):
                    initialize_thermo()
                prepare_compiled = getattr(
                    thermo,
                    'prepare_compiled_backends',
                    None,
                )
                if callable(prepare_compiled):
                    need_lle, need_vlle = backend_needs(scope)
                    prepare_compiled(
                        need_lle=need_lle,
                        need_vlle=need_vlle,
                    )
                self.thermo_packages[scope] = thermo

            self.thermo = self.thermo_packages['global']
            self.solver = FlowsheetSolver(
                self.pfd,
                self.thermo,
                self.thermo_packages,
            )
        except (FlowsheetError, ThermodynamicsError) as e:
            self.thermo = None
            self.thermo_packages = {}
            self.solver = None
            raise SimulationError(f"Failed to initialize simulation: {e}")

        self._initialized = True
        self._initialized_thermo_method = self.thermo_method
        self._initialized_fluid_phase_model = selected_phase_model
        return self

    def run(self, max_iterations: int = 100, tolerance: float = 1e-4,
            thermo_method: Optional[str] = None,
            progress_callback: Optional[Callable[[str], None]] = None,
            verbose: bool = False,
            recycle_method: Optional[str] = None) -> SimulationResult:
        """
        Run the simulation, initializing it first when necessary.

        Args:
            max_iterations: Maximum iterations for recycle convergence
            tolerance: Convergence tolerance
            thermo_method: Optional thermodynamic/property method override
            progress_callback: Optional callable receiving progress messages
            verbose: Print progress messages when no callback is supplied
            recycle_method: DIRECT, WEGSTEIN, or BROYDEN recycle method

        Returns:
            SimulationResult with all calculated values
        """
        self.initialize(thermo_method=thermo_method)
        if self.solver is None:
            raise SimulationError("Simulation initialization did not create a solver")

        try:
            callback = progress_callback
            if verbose and callback is None:
                callback = lambda message: print(message, flush=True)
            self.result = self.solver.solve(max_iterations, tolerance, callback, recycle_method)
        except (FlowsheetError, ThermodynamicsError) as e:
            raise SimulationError(f"Simulation failed: {e}")
        
        return self.result
    
    def write_results(self, filepath: str):
        """
        Write simulation results to a .pfr file.
        
        Args:
            filepath: Output file path
        """
        if self.result is None:
            raise SimulationError("No simulation results - run simulation first")
        
        content = self._generate_pfr()
        
        with open(filepath, 'w') as f:
            f.write(content)

    @staticmethod
    def _unit_heat_summary(unit_result):
        """
        Return heat utility totals as (heating, cooling, net) in kJ/h.

        Distillation columns expose condenser and reboiler duties separately;
        those are the utility loads users expect in reports. Other units only
        have the net heat_duty convention, where positive means heat added.
        """
        performance = unit_result.performance or {}
        heating = 0.0
        cooling = 0.0
        has_explicit_utility = False

        for key in ('reboiler_duty_kW', 'condenser_duty_kW'):
            if key not in performance:
                continue
            has_explicit_utility = True
            duty = float(performance[key]) * 3600.0
            if duty >= 0.0:
                heating += duty
            else:
                cooling += abs(duty)

        if has_explicit_utility:
            return heating, cooling, unit_result.heat_duty

        if unit_result.heat_duty >= 0.0:
            return unit_result.heat_duty, 0.0, unit_result.heat_duty
        return 0.0, abs(unit_result.heat_duty), unit_result.heat_duty

    @staticmethod
    def _unit_utility_work(unit_result) -> float:
        """
        Return shaft/electrical work in kJ/h for reporting.

        UnitResult.work is process work used by the energy balance. Some units
        also report utility work after mechanical losses or recoveries.
        """
        performance = unit_result.performance or {}
        if 'shaft_power_kW' in performance:
            return float(performance['shaft_power_kW']) * 3600.0
        if 'power_recovered_kW' in performance:
            return -float(performance['power_recovered_kW']) * 3600.0
        if 'power_kW' in performance:
            return float(performance['power_kW']) * 3600.0
        return unit_result.work

    @staticmethod
    def _source_quality(source: dict) -> Optional[float]:
        if not isinstance(source, dict):
            return None
        value = source.get('quality')
        try:
            return None if value is None else max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _quality_severity(quality: Optional[float], source: str, method: str) -> Optional[str]:
        source_l = (source or '').lower()
        method_l = (method or '').lower()
        if source_l == 'missing':
            return 'missing'
        if source_l == 'estimated' or method_l in {
            'mw_boiling_point',
            'mw_correlation',
            'guldberg_rule',
            'lydersen_style',
            'trouton',
        }:
            return 'estimated'
        if quality is None:
            return 'unknown_quality'
        if quality < 0.60:
            return 'very_low'
        if quality < 0.80:
            return 'low'
        if quality < 0.90:
            return 'medium'
        if quality < 0.95:
            return 'watch'
        return None

    @staticmethod
    def _result_quality_contexts(contexts) -> list[dict]:
        result = []
        for context in contexts or []:
            if not isinstance(context, dict):
                continue
            if int(context.get('count') or 0) <= 0:
                continue
            if context.get('affects_result', True):
                result.append(dict(context))
        return result

    @staticmethod
    def _suppression_reason(severity: Optional[str], affects_result: bool) -> Optional[str]:
        if not affects_result:
            return 'no_result_context'
        if severity is None:
            return 'high_quality'
        return None

    def _property_quality_report(self, include_suppressed: bool = False) -> list[dict]:
        """Collect context-aware component property-source quality entries.

        By default, returns only low/unknown quality entries used in
        result-affecting contexts.  Set include_suppressed=True for diagnostics
        and tests that need the retained high-quality or auxiliary entries.
        """
        if self.thermo is None:
            return []
        report = []
        for comp in sorted(getattr(self.thermo, 'props', {})):
            props = self.thermo.props[comp]
            sources = getattr(props, 'property_sources', {}) or {}
            for prop_name, source in sorted(sources.items()):
                if not isinstance(source, dict):
                    continue
                quality = self._source_quality(source)
                source_name = str(source.get('source') or '')
                method = str(source.get('method') or '')
                result_contexts = self._result_quality_contexts(source.get('contexts'))
                affects_result = bool(result_contexts)
                severity = self._quality_severity(quality, source_name, method)
                suppressed_reason = self._suppression_reason(
                    severity,
                    affects_result=affects_result,
                )
                if suppressed_reason and not include_suppressed:
                    continue
                contexts = (
                    [
                        dict(context) for context in source.get('contexts') or []
                        if isinstance(context, dict)
                    ]
                    if include_suppressed
                    else result_contexts
                )
                report.append({
                    'component': comp,
                    'property': prop_name,
                    'quality': quality,
                    'severity': severity or 'suppressed',
                    'source': source_name or 'unknown',
                    'method': method or 'unknown',
                    'notes': str(source.get('notes') or '').strip(),
                    'count': sum(int(context.get('count') or 0) for context in result_contexts)
                    if affects_result else None,
                    'count_label': 'uses',
                    'affects_result': affects_result,
                    'suppressed_reason': suppressed_reason,
                    'contexts': contexts,
                })
        lazy_sources = getattr(self.thermo, 'lazy_property_quality_sources', lambda: [])()
        for source in lazy_sources:
            if not isinstance(source, dict):
                continue
            quality = self._source_quality(source)
            source_name = str(source.get('source') or '')
            method = str(source.get('method') or '')
            result_contexts = self._result_quality_contexts(source.get('contexts'))
            affects_result = bool(result_contexts) or int(source.get('result_count') or 0) > 0
            if affects_result and source.get('result_quality') is not None:
                quality = self._source_quality({'quality': source.get('result_quality')})
            severity = self._quality_severity(quality, source_name, method)
            suppressed_reason = self._suppression_reason(severity, affects_result)
            if suppressed_reason and not include_suppressed:
                continue
            notes = source.get('notes') or []
            if isinstance(notes, (list, tuple)):
                notes_text = ' | '.join(str(note).strip() for note in notes if str(note).strip())
            else:
                notes_text = str(notes).strip()
            additional_notes = source.get('additional_note_count')
            if additional_notes:
                notes_text = (
                    f"{notes_text} | " if notes_text else ""
                ) + f"{additional_notes} additional note variant(s) omitted"
            if affects_result:
                count = int(source.get('result_count') or sum(
                    int(context.get('count') or 0) for context in result_contexts
                ))
                T_min = source.get('result_T_min')
                T_max = source.get('result_T_max')
                if include_suppressed:
                    contexts = [
                        dict(context) for context in source.get('contexts') or []
                        if isinstance(context, dict)
                    ]
                else:
                    contexts = result_contexts
            else:
                count = int(source.get('count') or 0)
                T_min = source.get('T_min')
                T_max = source.get('T_max')
                contexts = [
                    dict(context) for context in source.get('contexts') or []
                    if isinstance(context, dict)
                ]
            report.append({
                'component': str(source.get('component') or ''),
                'property': str(source.get('property') or ''),
                'quality': quality,
                'severity': severity or 'suppressed',
                'source': source_name or 'unknown',
                'method': method or 'unknown',
                'notes': notes_text,
                'T_min': T_min,
                'T_max': T_max,
                'count': count,
                'count_label': 'resolver_calls',
                'total_count': int(source.get('count') or 0),
                'affects_result': affects_result,
                'suppressed_reason': suppressed_reason,
                'contexts': contexts,
            })
        report.sort(key=lambda item: (
            item['component'],
            item['property'],
            item['severity'],
            item['source'],
            item['method'],
        ))
        return report

    @staticmethod
    def _format_quality_context(context: dict) -> str:
        kind = str(context.get('kind') or 'unscoped')
        phase = str(context.get('phase') or 'calculation')
        count = int(context.get('count') or 0)
        if kind == 'unit' and context.get('unit_id'):
            label = f"unit {context['unit_id']}"
            if context.get('unit_type'):
                label += f" ({context['unit_type']})"
        elif kind == 'stream' and context.get('stream_id'):
            label = f"stream {context['stream_id']}"
        else:
            label = kind
        return f"{label} {phase}: {count} call(s)"

    def _format_quality_contexts(self, contexts: list[dict]) -> str:
        parts = [self._format_quality_context(context) for context in contexts[:4]]
        omitted = len(contexts) - len(parts)
        if omitted > 0:
            parts.append(f"{omitted} additional context(s)")
        return '; '.join(parts)
    
    def _generate_pfr(self) -> str:
        """Generate .pfr file content"""
        lines = []
        has_recycle = bool(self.result.recycle_info.get('tear_streams'))
        
        # Header
        lines.append("#" + "=" * 78)
        lines.append("# PROCESS FLOW RESULTS")
        lines.append("#" + "=" * 78)
        lines.append("")
        
        # Metadata
        lines.append("RESULTS_FOR: " + (self.pfd.metadata.process_name or "Unnamed Process"))
        lines.append("VERSION: " + (self.pfd.metadata.version or "1.0"))
        lines.append(f"GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("SIMULATOR: PFD-Editor v1.0")
        lines.append(f"THERMO_METHOD: {getattr(self, 'thermo_method', 'IDEAL')}")
        for scope, method in getattr(
            self,
            'thermo_scope_methods',
            {'global': getattr(self, 'thermo_method', 'IDEAL')},
        ).items():
            if scope != 'global':
                lines.append(f"THERMO_SCOPE {scope}: {method}")
        lines.append(
            "FLUID_PHASE_MODEL: "
            f"{getattr(self.pfd.metadata, 'fluid_phase_model', 'VLE')}"
        )
        lines.append(f"CONVERGED: {'yes' if self.result.converged else 'no'}")
        if has_recycle:
            lines.append(f"ITERATIONS: {self.result.iterations}")
        lines.append("")
        
        # Summary
        lines.append("#" + "-" * 78)
        lines.append("# SIMULATION SUMMARY")
        lines.append("#" + "-" * 78)
        lines.append("")
        lines.append("SUMMARY:")
        lines.append(f"    convergence_status = {'converged' if self.result.converged else 'not_converged'}")
        if has_recycle:
            lines.append(f"    iterations = {self.result.iterations}")
        lines.append(f"    overall_mass_balance_error = {self.result.mass_balance_error*100:.4f} [%]")
        lines.append(f"    overall_energy_balance_error = {self.result.energy_balance_error*100:.4f} [%]")
        lines.append(
            "    thermo_scope_enthalpy_correction = "
            f"{getattr(self.result, 'thermo_scope_enthalpy_correction', 0.0) / 3600:.6g} [kW]"
        )
        
        # Overall duties
        total_heating = 0.0
        total_cooling = 0.0
        total_net_heat = 0.0
        total_work = 0.0
        total_process_work = 0.0
        
        for unit_id, unit_result in self.result.units.items():
            heating, cooling, net_heat = self._unit_heat_summary(unit_result)
            total_heating += heating
            total_cooling += cooling
            total_net_heat += net_heat
            total_work += self._unit_utility_work(unit_result)
            total_process_work += unit_result.work
        
        lines.append(f"    total_heating_duty = {total_heating/3600:.2f} [kW]")
        lines.append(f"    total_cooling_duty = {total_cooling/3600:.2f} [kW]")
        lines.append(f"    total_net_heat_duty = {total_net_heat/3600:.2f} [kW]")
        lines.append(f"    total_work = {total_work/3600:.2f} [kW]")
        lines.append(f"    total_process_work = {total_process_work/3600:.2f} [kW]")
        lines.append("")

        scope_corrections = getattr(
            self.result,
            'thermo_scope_corrections',
            [],
        )
        if scope_corrections:
            lines.append("THERMO_SCOPE_CORRECTIONS:")
            for correction in scope_corrections:
                lines.append(
                    f"    {correction['stream_id']} | "
                    f"source={correction['source_scope']}, "
                    f"destination={correction['destination_scope']}, "
                    "enthalpy_correction="
                    f"{correction['enthalpy_flow_correction_kJ_per_h'] / 3600:.6g} [kW]"
                )
            lines.append("")
        
        # Stream results
        lines.append("#" + "-" * 78)
        lines.append("# STREAM RESULTS")
        lines.append("#" + "-" * 78)
        lines.append("")
        
        for stream in self.pfd.streams:
            state = self.result.streams.get(stream.id)
            if state is None:
                continue
            
            lines.append(f"STREAM_RESULT {stream.id}:")
            lines.append(f"    source = {stream.source.to_string()}")
            lines.append(f"    destination = {stream.destination.to_string()}")
            lines.append(f"    T = {state.T - 273.15:.2f} [C]")
            lines.append(f"    T_K = {state.T:.2f} [K]")
            lines.append(f"    P = {state.P:.4f} [bar]")
            lines.append(f"    F = {state.F:.4f} [kmol/h]")
            lines.append(f"    F_mass = {state.mass_flow():.2f} [kg/h]")
            if state.rho is not None and state.rho > 0.0:
                lines.append(f"    F_vol = {state.F / state.rho:.4f} [m3/h]")
            lines.append(f"    vapor_fraction = {state.vapor_fraction:.4f}")
            lines.append(
                f"    liquid1_fraction = {state.effective_liquid1_fraction:.4f}"
            )
            if state.liquid2_fraction > 1.0e-15 or state.x2:
                lines.append(f"    liquid2_fraction = {state.liquid2_fraction:.4f}")
            if (
                state.solid_fraction > 1.0e-15
                or state.solid_composition
                or state.solid_component_flows
            ):
                lines.append(f"    solid_fraction = {state.solid_fraction:.4f}")
                if state.fluid_vapor_fraction is not None:
                    lines.append(
                        "    fluid_vapor_fraction = "
                        f"{state.fluid_vapor_fraction:.4f}"
                    )
            lines.append(f"    phase_status = {state.phase_status}")
            lines.append(f"    phase_stability = {state.phase_stability}")
            spinodal = state.phase_details.get('spinodal')
            if isinstance(spinodal, dict):
                minimum = spinodal.get('minimum_eigenvalue')
                if minimum is not None and math.isfinite(float(minimum)):
                    lines.append(
                        "    spinodal_minimum_eigenvalue = "
                        f"{float(minimum):.8g}"
                    )
                if spinodal.get('activity_backend'):
                    lines.append(
                        "    spinodal_activity_backend = "
                        f"{spinodal['activity_backend']}"
                    )
            lines.append(f"    MW = {state.MW:.2f} [kg/kmol]")
            
            if state.H is not None:
                lines.append(f"    H = {state.H:.2f} [kJ/kmol]")
            if state.S is not None:
                lines.append(f"    S = {state.S:.4f} [kJ/kmol-K]")
            if state.Cp is not None:
                lines.append(f"    Cp = {state.Cp:.4f} [kJ/kmol-K]")
            if state.rho is not None:
                lines.append(f"    rho = {state.rho:.4f} [kmol/m3]")
            
            # Composition
            lines.append("    COMPOSITION:")
            for comp, z in sorted(state.composition.items(), key=lambda x: -x[1]):
                mole_flow = state.F * z
                lines.append(f"        {comp}: x={z:.6f}, F={mole_flow:.4f} [kmol/h]")
            
            # Explicit phase compositions. LIQUID_COMPOSITION remains the
            # pooled compatibility view when two liquids coexist.
            if state.x and (
                state.vapor_fraction > 0.001 or state.liquid2_fraction > 0.001
            ):
                lines.append("    LIQUID_COMPOSITION:")
                for comp, x in sorted(state.x.items(), key=lambda c: -c[1]):
                    lines.append(f"        {comp}: {x:.6f}")
            if state.liquid2_fraction > 0.001:
                if state.x1:
                    lines.append("    LIQUID1_COMPOSITION:")
                    for comp, x in sorted(state.x1.items(), key=lambda c: -c[1]):
                        lines.append(f"        {comp}: {x:.6f}")
                if state.x2:
                    lines.append("    LIQUID2_COMPOSITION:")
                    for comp, x in sorted(state.x2.items(), key=lambda c: -c[1]):
                        lines.append(f"        {comp}: {x:.6f}")
            if state.y and state.vapor_fraction > 0.001:
                lines.append("    VAPOR_COMPOSITION:")
                for comp, y in sorted(state.y.items(), key=lambda c: -c[1]):
                    lines.append(f"        {comp}: {y:.6f}")
            if state.solid_composition and state.solid_fraction > 0.001:
                lines.append("    SOLID_COMPOSITION:")
                for comp, value in sorted(
                    state.solid_composition.items(), key=lambda c: -c[1]
                ):
                    lines.append(f"        {comp}: {value:.6f}")
            if state.solid_component_flows:
                lines.append("    SOLID_COMPONENT_FLOWS:")
                for comp, flow in sorted(state.solid_component_flows.items()):
                    lines.append(
                        f"        {comp}: F={flow:.4f} [kmol/h]"
                    )
            if state.solid_particle_properties:
                lines.append("    SOLID_PARTICLE_PROPERTIES:")
                for comp, values in sorted(
                    state.solid_particle_properties.items()
                ):
                    fields = []
                    if values.get('diameter_m') is not None:
                        fields.append(
                            f"diameter={values['diameter_m']:.8g} [m]"
                        )
                    if values.get('sphericity') is not None:
                        fields.append(
                            f"sphericity={values['sphericity']:.8g}"
                        )
                    if fields:
                        lines.append(f"        {comp}: {', '.join(fields)}")
            if state.solid_particle_size_distributions:
                state.validate_particle_size_distributions()
                lines.append("    SOLID_PARTICLE_SIZE_DISTRIBUTIONS:")
                for comp, distribution in sorted(
                    state.solid_particle_size_distributions.items()
                ):
                    d32 = distribution.sauter_mean_diameter_m
                    summary = "basis=component_molar_flow"
                    if d32 is not None:
                        summary += f", D32={d32:.8g} [m]"
                    lines.append(f"        {comp}: {summary}")
                    for diameter, flow, fraction in zip(
                        distribution.diameters_m,
                        distribution.molar_flows_kmol_per_h,
                        distribution.molar_fractions,
                    ):
                        lines.append(
                            f"            diameter={diameter:.8g} [m], "
                            f"F={flow:.8g} [kmol/h], fraction={fraction:.8g}"
                        )
            
            lines.append("")
        
        # Unit results
        lines.append("#" + "-" * 78)
        lines.append("# UNIT OPERATION RESULTS")
        lines.append("#" + "-" * 78)
        lines.append("")
        
        for unit in self.pfd.units:
            unit_result = self.result.units.get(unit.id)
            if unit_result is None:
                continue
            
            lines.append(f"UNIT_RESULT {unit.id}:")
            lines.append(f"    type = {unit.unit_type}")
            heating, cooling, net_heat = self._unit_heat_summary(unit_result)
            if heating > 0.0 or cooling > 0.0:
                lines.append(f"    heating_duty = {heating/3600:.4f} [kW]")
                lines.append(f"    cooling_duty = {cooling/3600:.4f} [kW]")
                lines.append(f"    net_heat_duty = {net_heat/3600:.4f} [kW]")
            else:
                lines.append(f"    heat_duty = {unit_result.heat_duty/3600:.4f} [kW]")
            utility_work = self._unit_utility_work(unit_result)
            lines.append(f"    work = {utility_work/3600:.4f} [kW]")
            lines.append(f"    process_work = {unit_result.work/3600:.4f} [kW]")
            
            # Performance metrics
            lines.append("    PERFORMANCE:")
            grouped_vlle = str(
                unit_result.performance.get('stage_phase_model', '')
            ).strip().upper() == 'VLLE'
            grouped_vlle_keys = {
                'stage_phase_counts',
                'stage_liquid1_compositions',
                'stage_liquid2_compositions',
                'stage_liquid2_fractions',
                'stage_liquid1_flows',
                'stage_liquid2_flows',
                'vlle_active_stages',
            }
            for key, value in unit_result.performance.items():
                if grouped_vlle and key in grouped_vlle_keys:
                    continue
                if isinstance(value, float):
                    if abs(value) < 1e-3 and value != 0.0:
                        formatted_value = f"{value:.6e}"
                    else:
                        formatted_value = f"{value:.4f}"
                    lines.append(f"        {key} = {formatted_value}")
                elif isinstance(value, dict):
                    lines.append(f"        {key}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            lines.append(f"            {k}: {v:.6f}")
                        else:
                            lines.append(f"            {k}: {v}")
                elif isinstance(value, list):
                    lines.append(f"        {key}:")
                    for item in value:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                lines.append(f"            {k}: {v}")
                        else:
                            lines.append(f"            - {item}")
                else:
                    lines.append(f"        {key} = {value}")

            if grouped_vlle:
                performance = unit_result.performance
                active_stages = [
                    int(value) for value in performance.get('vlle_active_stages', [])
                ]
                liquid1 = performance.get('stage_liquid1_compositions', [])
                liquid2 = performance.get('stage_liquid2_compositions', [])
                liquid2_fractions = performance.get('stage_liquid2_fractions', [])
                liquid1_flows = performance.get('stage_liquid1_flows', [])
                liquid2_flows = performance.get('stage_liquid2_flows', [])
                phase_counts = performance.get('stage_phase_counts', [])
                lines.append("        VLLE_STAGES:")
                lines.append(f"            active_stage_count = {len(active_stages)}")
                if active_stages:
                    lines.append(
                        "            active_stages = "
                        + ", ".join(str(stage) for stage in active_stages)
                    )
                for stage_number in active_stages:
                    index = stage_number - 1
                    if not 0 <= index < len(liquid1) or index >= len(liquid2):
                        continue
                    phase_count = (
                        int(phase_counts[index]) if index < len(phase_counts) else 3
                    )
                    beta = (
                        float(liquid2_fractions[index])
                        if index < len(liquid2_fractions) else 0.0
                    )
                    flow1 = (
                        float(liquid1_flows[index])
                        if index < len(liquid1_flows) else 0.0
                    )
                    flow2 = (
                        float(liquid2_flows[index])
                        if index < len(liquid2_flows) else 0.0
                    )
                    lines.append(f"            STAGE {stage_number}:")
                    lines.append(f"                phase_count = {phase_count}")
                    lines.append(f"                liquid2_fraction = {beta:.6f}")
                    lines.append(f"                liquid1_flow = {flow1:.4f} [kmol/h]")
                    lines.append(f"                liquid2_flow = {flow2:.4f} [kmol/h]")
                    lines.append("                LIQUID1_COMPOSITION:")
                    for comp, fraction in sorted(
                        liquid1[index].items(), key=lambda item: -item[1]
                    ):
                        lines.append(f"                    {comp}: {fraction:.6f}")
                    lines.append("                LIQUID2_COMPOSITION:")
                    for comp, fraction in sorted(
                        liquid2[index].items(), key=lambda item: -item[1]
                    ):
                        lines.append(f"                    {comp}: {fraction:.6f}")
            
            # Warnings
            if unit_result.warnings:
                lines.append("    WARNINGS:")
                for warning in unit_result.warnings:
                    lines.append(f"        - {warning}")
            
            lines.append("")
        
        # Recycle information
        if self.result.recycle_info.get('tear_streams'):
            lines.append("#" + "-" * 78)
            lines.append("# RECYCLE INFORMATION")
            lines.append("#" + "-" * 78)
            lines.append("")
            lines.append("RECYCLE:")
            lines.append(f"    tear_streams = {', '.join(self.result.recycle_info['tear_streams'])}")
            lines.append(f"    calculation_order = {', '.join(self.result.recycle_info['calculation_order'])}")
            lines.append(f"    method = {self.result.recycle_info.get('method', 'WEGSTEIN')}")
            lines.append(f"    variable_basis = {self.result.recycle_info.get('variable_basis', 'unknown')}")
            lines.append(f"    trace_tolerance = {self.result.recycle_info.get('trace_tolerance', '')}")
            if self.result.recycle_info.get('auto_selected_tears'):
                lines.append(f"    auto_selected_tears = {', '.join(self.result.recycle_info['auto_selected_tears'])}")
            if self.result.recycle_info.get('manual_tears'):
                lines.append(f"    manual_tears = {', '.join(self.result.recycle_info['manual_tears'])}")
            if self.result.recycle_info.get('worst_variable'):
                lines.append(f"    worst_variable = {self.result.recycle_info['worst_variable']}")
                lines.append(f"    worst_error = {self.result.recycle_info.get('worst_error', 0.0):.6g}")
            lines.append(f"    failed_evaluations = {self.result.recycle_info.get('failed_evaluations', 0)}")
            if self.result.recycle_info.get('last_failure'):
                lines.append(f"    last_failure = {self.result.recycle_info['last_failure']}")
            lines.append("")
        
        # Errors and warnings
        if self.result.errors:
            lines.append("#" + "-" * 78)
            lines.append("# ERRORS")
            lines.append("#" + "-" * 78)
            lines.append("")
            lines.append("ERRORS:")
            for error in self.result.errors:
                lines.append(f"    - {error}")
            lines.append("")
        
        if self.result.warnings:
            lines.append("#" + "-" * 78)
            lines.append("# WARNINGS")
            lines.append("#" + "-" * 78)
            lines.append("")
            lines.append("WARNINGS:")
            for warning in self.result.warnings:
                lines.append(f"    - {warning}")
            lines.append("")

        quality_report = self._property_quality_report()
        lines.append("#" + "-" * 78)
        lines.append("# QUALITY REPORT")
        lines.append("#" + "-" * 78)
        lines.append("")
        lines.append("QUALITY_REPORT:")
        if not quality_report:
            lines.append("    status = no_flagged_property_resolution_quality_issues")
        else:
            lines.append(f"    flagged_properties = {len(quality_report)}")
            for item in quality_report:
                quality = item['quality']
                quality_text = "unknown" if quality is None else f"{quality:.3f}"
                lines.append(
                    "    - "
                    f"{item['severity']}: {item['component']}.{item['property']} "
                    f"quality={quality_text} "
                    f"source={item['source']} method={item['method']}"
                )
                if item.get('count') is not None:
                    try:
                        T_min = float(item.get('T_min'))
                        T_max = float(item.get('T_max'))
                    except (TypeError, ValueError):
                        T_min = T_max = None
                    temperature_text = ""
                    if T_min is not None and T_max is not None:
                        if abs(T_max - T_min) < 5e-7:
                            temperature_text = f", T = {T_min:.2f} K"
                        else:
                            temperature_text = f", T_range = {T_min:.2f}-{T_max:.2f} K"
                    count_label = item.get('count_label') or 'resolver_calls'
                    lines.append(
                        f"      {count_label} = {int(item['count'])}{temperature_text}"
                    )
                if item.get('contexts'):
                    lines.append(
                        f"      contexts = {self._format_quality_contexts(item['contexts'])}"
                    )
                if item['notes']:
                    lines.append(f"      notes = {item['notes']}")
        lines.append("")
        
        # Footer
        lines.append("#" + "=" * 78)
        lines.append("# END OF RESULTS")
        lines.append("#" + "=" * 78)
        
        return "\n".join(lines)
    
    def get_results_dict(self) -> dict:
        """Get results as a dictionary for JSON serialization"""
        if self.result is None:
            return {'error': 'No simulation results'}
        
        return {
            'metadata': {
                'process_name': self.pfd.metadata.process_name,
                'version': self.pfd.metadata.version,
                'generated': datetime.now().isoformat(),
                'thermo_method': getattr(self, 'thermo_method', 'IDEAL'),
                'thermo_scopes': dict(getattr(
                    self,
                    'thermo_scope_methods',
                    {'global': getattr(self, 'thermo_method', 'IDEAL')},
                )),
                'fluid_phase_model': getattr(
                    self.pfd.metadata, 'fluid_phase_model', 'VLE'
                ),
            },
            'summary': {
                'converged': self.result.converged,
                'iterations': self.result.iterations,
                'mass_balance_error': self.result.mass_balance_error,
                'energy_balance_error': self.result.energy_balance_error,
                'thermo_scope_enthalpy_correction': (
                    getattr(
                        self.result,
                        'thermo_scope_enthalpy_correction',
                        0.0,
                    )
                ),
            },
            'streams': {
                stream_id: state.to_dict() 
                for stream_id, state in self.result.streams.items()
            },
            'units': {
                unit_id: result.to_dict()
                for unit_id, result in self.result.units.items()
            },
            'recycle_info': self.result.recycle_info,
            'thermo_scope_corrections': (
                getattr(self.result, 'thermo_scope_corrections', [])
            ),
            'errors': self.result.errors,
            'warnings': self.result.warnings,
        }


def simulate_pfd(pfd_content: str, max_iterations: int = 100) -> dict:
    """
    Convenience function to simulate a PFD from string content.
    
    Args:
        pfd_content: PFD file content
        max_iterations: Maximum recycle iterations
        
    Returns:
        Dictionary with simulation results
    """
    try:
        sim = Simulator.from_string(pfd_content)
        sim.run(max_iterations)
        return sim.get_results_dict()
    except (ParseError, SimulationError) as e:
        return {
            'error': str(e),
            'converged': False,
        }


def simulate_file(input_path: str, output_path: Optional[str] = None) -> SimulationResult:
    """
    Simulate a PFD file and optionally write results.
    
    Args:
        input_path: Path to .pfd file
        output_path: Path for .pfr output (auto-generated if None)
        
    Returns:
        SimulationResult
    """
    sim = Simulator.from_file(input_path)
    result = sim.run()
    
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('.pfr'))
    
    sim.write_results(output_path)
    
    return result
