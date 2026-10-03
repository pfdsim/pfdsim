"""
Degrees of Freedom (DOF) Analyzer for Process Flow Diagrams

This module analyzes whether a process specification is:
- Properly specified (DOF = 0 for each unit)
- Under-specified (DOF > 0, simulation cannot solve)
- Over-specified (DOF < 0, conflicting specifications)

Supports:
- VLE (vapor-liquid equilibrium)
- LLE (liquid-liquid equilibrium)
- VLLE (vapor-liquid-liquid equilibrium)
- Reactive systems
"""

from dataclasses import dataclass, field
from enum import Enum

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_syntax import UNIT_TYPE_ALIASES, canonical_unit_type
else:
    from unit_syntax import UNIT_TYPE_ALIASES, canonical_unit_type

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .crystallizer_specs import (
        CrystallizerSpecificationError,
        validate_crystallizer_specification,
        validate_layer_crystallizer_specification,
    )
else:
    from crystallizer_specs import (
        CrystallizerSpecificationError,
        validate_crystallizer_specification,
        validate_layer_crystallizer_specification,
    )


class SpecificationStatus(Enum):
    OK = "ok"
    UNDER_SPECIFIED = "under_specified"
    OVER_SPECIFIED = "over_specified"
    WARNING = "warning"


@dataclass
class DOFResult:
    """Result of DOF analysis for a single unit or stream"""
    entity_id: str
    entity_type: str
    total_variables: int
    equations: int
    specifications: int
    dof: int
    status: SpecificationStatus
    message: str
    details: list[str] = field(default_factory=list)


@dataclass 
class ProcessDOFResult:
    """Complete DOF analysis for entire process"""
    overall_status: SpecificationStatus
    total_dof: int
    unit_results: list[DOFResult] = field(default_factory=list)
    stream_results: list[DOFResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


_CRYSTALLIZER_COMMON_OPTIONAL_SPECS = {
    'P_out': {'unit': 'bar'},
    'P_drop': {'unit': 'bar', 'default': 0.0},
    'equilibrium_tolerance': {'default': 1e-8},
    'max_iterations': {'default': 500},
    'mother_liquor_retention': {
        'description': (
            'Fraction of equilibrium mother liquor retained with the cake; '
            'specifying it enables cake/mother-liquor outlets'
        ),
    },
    'mother_liquor_retention_rate': {
        'description': (
            'Mass of equilibrium mother liquor retained per mass of conventional '
            'crystals; specifying it enables cake/mother-liquor outlets'
        ),
    },
}


# Comprehensive DOF rules for each unit type
UNIT_DOF_RULES = {
    # =========================================================================
    # MIXING AND SPLITTING
    # =========================================================================
    'Mixer': {
        'description': 'Combine multiple streams into one',
        'category': 'mixing',
        'phase_support': ['VLE', 'LLE', 'VLLE'],
        'required_specs': [],
        'optional_specs': {
            'mode': {'values': ['adiabatic', 'isothermal'], 'default': 'adiabatic'},
            'P_drop': {'unit': 'bar', 'default': 0},
        },
        'dof_notes': 'Fully determined by inlet streams. DOF=0 for adiabatic mixing.',
    },
    'Splitter': {
        'description': 'Split stream into multiple outlets with same composition',
        'category': 'mixing',
        'phase_support': ['VLE', 'LLE'],
        'required_specs': ['split_frac'],
        'optional_specs': {},
        'dof_notes': 'Requires (N_outlets - 1) split fraction specifications.',
    },
    'ComponentSplitter': {
        'description': 'Split stream by component (membrane separation)',
        'category': 'mixing',
        'phase_support': ['VLE'],
        'required_specs': ['split_fracs'],
        'optional_specs': {'key_component': {}},
        'dof_notes': 'Specify split fraction for each component.',
    },

    # =========================================================================
    # PRESSURE CHANGE
    # =========================================================================
    'Pump': {
        'description': 'Increase liquid pressure',
        'category': 'pressure',
        'phase_support': ['liquid'],
        'required_specs': ['P_out|delta_P|pressure_ratio'],
        'optional_specs': {
            'eta': {'description': 'Overall efficiency', 'default': 0.75, 'range': [0.5, 0.95]},
            'eta_mech': {'description': 'Mechanical efficiency', 'default': 0.95, 'range': [0.85, 0.99]},
            'eta_hyd': {'description': 'Hydraulic efficiency', 'default': 0.80, 'range': [0.6, 0.92]},
            'NPSH_avail': {'unit': 'm', 'description': 'Available NPSH'},
            'curve_file': {'description': 'Pump curve data file'},
        },
        'calculated': ['work', 'T_out', 'NPSH_req'],
        'dof_notes': 'Specify exactly one of outlet pressure, pressure rise, or pressure ratio.',
    },
    'Compressor': {
        'description': 'Increase gas pressure',
        'category': 'pressure',
        'phase_support': ['vapor'],
        'required_specs': ['P_out|delta_P|pressure_ratio'],
        'optional_specs': {
            'eta_isen': {'description': 'Isentropic efficiency', 'default': 0.75, 'range': [0.6, 0.9]},
            'eta_poly': {'description': 'Polytropic efficiency', 'range': [0.65, 0.88]},
            'eta_mech': {'description': 'Mechanical efficiency', 'default': 0.98, 'range': [0.9, 0.99]},
            'n_stages': {'description': 'Number of stages', 'default': 1},
            'intercool': {'values': ['none', 'ideal', 'specified'], 'default': 'none'},
            'intercool_T': {'unit': 'C', 'description': 'Intercooler outlet temperature'},
            'model': {'values': ['isentropic', 'polytropic', 'curve'], 'default': 'isentropic'},
            'curve_file': {'description': 'Compressor curve data'},
            'surge_margin': {'default': 0.1},
        },
        'calculated': ['work', 'T_out', 'surge_flow', 'choke_flow'],
        'dof_notes': 'Specify exactly one of outlet pressure, pressure rise, or pressure ratio. Use eta_isen; eta_poly is not yet implemented.',
    },
    'Expander': {
        'description': 'Reduce pressure and recover work (turbine)',
        'category': 'pressure',
        'phase_support': ['vapor', 'two-phase'],
        'required_specs': ['P_out|delta_P|pressure_ratio'],
        'optional_specs': {
            'eta_isen': {'default': 0.80},
            'eta_mech': {'default': 0.98},
        },
        'calculated': ['work', 'T_out'],
        'dof_notes': 'Specify exactly one of outlet pressure, pressure drop, or pressure ratio.',
    },
    'Valve': {
        'description': 'Reduce pressure (isenthalpic)',
        'category': 'pressure',
        'phase_support': ['VLE', 'LLE'],
        'required_specs': ['P_out'],
        'optional_specs': {
            'Cv': {'description': 'Valve coefficient'},
            'type': {'values': ['globe', 'ball', 'butterfly', 'gate'], 'default': 'globe'},
        },
        'calculated': ['T_out', 'vapor_frac'],
        'dof_notes': 'Isenthalpic flash. May produce two-phase outlet.',
    },
    'Pipe': {
        'description': 'Integrated pressure drop through a circular pipe',
        'category': 'pressure',
        'phase_support': ['liquid', 'vapor', 'two-phase'],
        'required_specs': ['length', 'diameter|velocity'],
        'optional_specs': {
            'diameter_out': {'unit': 'm', 'description': 'Outlet diameter for a taper'},
            'diameter_profile': {
                'values': ['constant', 'linear', 'smooth', 'smoothstep'],
            },
            'material': {'default': 'commercial_steel'},
            'roughness': {'unit': 'm'},
            'orientation': {'values': ['horizontal', 'vertical_up', 'vertical_down']},
            'angle': {'unit': 'degree'},
            'elevation_change': {'unit': 'm'},
            'friction_model': {
                'values': ['churchill', 'haaland', 'swamee-jain', 'colebrook'],
                'default': 'churchill',
            },
            'phase_model': {
                'values': ['auto', 'single_phase', 'two_phase'],
                'default': 'auto',
            },
            'two_phase_model': {
                'values': ['beggs_brill'],
                'default': 'beggs_brill',
            },
            'surface_tension_method': {
                'values': ['auto', 'butler', 'butler-unifac', 'wsd'],
                'default': 'auto',
            },
            'energy_tolerance': {
                'unit': 'kJ/kmol',
                'default': 1e-6,
                'description': 'Local static-enthalpy/velocity coupling tolerance',
            },
            'energy_max_iterations': {
                'default': '12 single-phase, 1 two-phase',
                'description': 'Maximum local energy-coupling iterations',
            },
        },
        'calculated': ['P_out', 'T_out', 'pressure_drop', 'Re', 'friction_factor', 'liquid_holdup'],
        'dof_notes': 'Specify length and exactly one inlet sizing method: diameter or velocity.',
    },

    # =========================================================================
    # HEAT TRANSFER
    # =========================================================================
    'Heater': {
        'description': 'Add heat to stream (utility heater)',
        'category': 'heat_transfer',
        'phase_support': ['VLE'],
        'required_specs': ['T_out|Q|vap_frac'],
        'optional_specs': {
            'P_drop': {'unit': 'bar', 'default': 0},
            'utility': {'values': ['steam', 'hot_oil', 'electric', 'fired'], 'default': 'steam'},
            'utility_T': {'unit': 'C'},
            'U': {'unit': 'W/m2-K'},
        },
        'calculated': ['Q', 'T_out', 'utility_flow', 'area'],
        'dof_notes': 'Specify EXACTLY ONE of: T_out, Q, or outlet vapor_frac.',
    },
    'Cooler': {
        'description': 'Remove heat from stream',
        'category': 'heat_transfer',
        'phase_support': ['VLE'],
        'required_specs': ['T_out|Q|vap_frac'],
        'optional_specs': {
            'P_drop': {'unit': 'bar', 'default': 0},
            'utility': {'values': ['cooling_water', 'chilled_water', 'refrigerant', 'air'], 'default': 'cooling_water'},
            'utility_T_in': {'unit': 'C', 'default': 30},
            'utility_T_out': {'unit': 'C', 'default': 45},
            'U': {'unit': 'W/m2-K'},
        },
        'calculated': ['Q', 'T_out', 'utility_flow', 'area'],
        'dof_notes': 'Specify EXACTLY ONE of: T_out, Q, or outlet vapor_frac.',
    },
    'HeatExchanger': {
        'description': 'Exchange heat between two process streams',
        'category': 'heat_transfer',
        'phase_support': ['VLE'],
        'required_specs': ['thermal target OR rating capacity'],
        'optional_specs': {
            'type': {'values': ['countercurrent', 'cocurrent', 'shell_tube', 'plate', 'double_pipe'], 'default': 'countercurrent'},
            'flow_pattern': {'values': ['countercurrent', 'cocurrent'], 'default': 'countercurrent'},
            'shell_passes': {'default': 1},
            'tube_passes': {'default': 2},
            'P_drop_tube': {'unit': 'bar', 'default': 0},
            'P_drop_shell': {'unit': 'bar', 'default': 0},
            'P_drop_hot': {'unit': 'bar', 'default': 0},
            'P_drop_cold': {'unit': 'bar', 'default': 0},
            'curve_segments': {'default': 40, 'range': [4, 400]},
            'UA': {'unit': 'W/K'},
            'UA_available': {'unit': 'W/K'},
            'LMTD_correction': {'default': 1.0, 'range': [0.7, 1.0]},
            'min_approach': {'unit': 'C'},
            'fouling_tube': {'unit': 'm2-K/W', 'default': 0.0001},
            'fouling_shell': {'unit': 'm2-K/W', 'default': 0.0002},
            'Q': {'unit': 'kW'},
            'U': {'unit': 'W/m2-K'},
            'A': {'unit': 'm2'},
            'estimate_U': {'default': False},
            'allow_temperature_cross': {'default': False},
            'hot_vapor_fraction': {'range': [0, 1]},
            'cold_vapor_fraction': {'range': [0, 1]},
            'tube_vapor_fraction': {'range': [0, 1]},
            'shell_vapor_fraction': {'range': [0, 1]},
        },
        'calculated': ['Q', 'UA_required', 'area', 'effectiveness', 'min_approach'],
        'dof_notes': 'Design mode: exactly one outlet T, Q, or vapor_frac; optional U/A/UA report sizing. Rating mode: no thermal target, but requires UA, U+A, or A+auto-U.',
    },
    'FiredHeater': {
        'description': 'Process furnace with radiant and convective sections',
        'category': 'heat_transfer',
        'phase_support': ['VLE'],
        'required_specs': ['T_out|Q'],
        'optional_specs': {
            'fuel': {'values': ['natural_gas', 'fuel_oil', 'H2'], 'default': 'natural_gas'},
            'efficiency': {'default': 0.85, 'range': [0.7, 0.95]},
            'excess_air': {'default': 0.15},
            'radiant_frac': {'default': 0.6},
            'stack_T': {'unit': 'C', 'default': 200},
            'P_drop': {'unit': 'bar', 'default': 1.0},
        },
        'calculated': ['fuel_flow', 'flue_gas_flow', 'radiant_duty', 'convective_duty'],
        'dof_notes': 'Specify outlet T or total duty.',
    },

    # =========================================================================
    # VLE SEPARATION
    # =========================================================================
    'Flash': {
        'description': 'Single-stage vapor-liquid equilibrium separation',
        'category': 'separation',
        'phase_support': ['VLE'],
        'required_specs': ['T,P|T,VF|P,VF|P,Q'],
        'optional_specs': {
            'thermo_model': {'values': ['ideal', 'SRK', 'PR', 'NRTL', 'UNIQUAC'], 'default': 'PR'},
            'valid_phases': {'values': ['VL', 'VLL', 'VLS'], 'default': 'VL'},
        },
        'calculated': ['T', 'P', 'vapor_frac', 'Q', 'K_values'],
        'dof_notes': 'Specify exactly 2 of: T, P, vapor_fraction/VF, or Q/duty/heat_duty.',
    },
    'Flash3': {
        'description': 'Three-phase flash (vapor-liquid-liquid)',
        'category': 'separation',
        'phase_support': ['VLLE'],
        'required_specs': ['T', 'P'],
        'optional_specs': {
            'thermo_model': {'values': ['NRTL', 'UNIQUAC', 'UNIFAC'], 'default': 'NRTL'},
            'heavy_key': {'description': 'Component identifying heavy liquid phase'},
        },
        'calculated': ['vapor_frac', 'liquid1_frac', 'liquid2_frac', 'Q'],
        'dof_notes': 'Three-phase flash requires LLE-capable thermodynamic model.',
    },
    'ShortcutDistillation': {
        'description': 'Multi-stage vapor-liquid separation column',
        'category': 'separation',
        'phase_support': ['VLE'],
        'required_specs': ['N_stages', 'reflux_ratio|D_rate|recovery|purity'],
        'optional_specs': {
            'condenser_type': {'values': ['total', 'partial', 'subcooled', 'none'], 'default': 'total'},
            'reboiler_type': {'values': ['kettle', 'thermosiphon', 'forced_circ', 'fired', 'none'], 'default': 'kettle'},
            'P_condenser': {'unit': 'bar'},
            'P_drop_per_stage': {'unit': 'bar', 'default': 0.01},
            'condenser_subcool': {'unit': 'C', 'default': 0},
            'stage_efficiency': {'default': 1.0, 'range': [0.5, 1.0]},
            'murphree_eff': {'description': 'Murphree efficiency per stage'},
            'boilup_ratio': {'description': 'Alternative to reflux ratio'},
            'side_draws': {'description': 'List of side product draws'},
            'pumparounds': {'description': 'Internal liquid recirculation'},
            'valid_phases': {'values': ['VL', 'VLL'], 'default': 'VL'},
        },
        'calculated': ['condenser_duty', 'reboiler_duty', 'stage_T', 'stage_compositions'],
        'dof_notes': 'Specify stages and ONE of: reflux_ratio, D_rate, recovery, purity.',
    },
    'ReactiveDistillation': {
        'description': 'ShortcutDistillation with chemical reactions on stages',
        'category': 'separation',
        'phase_support': ['VLE'],
        'required_specs': ['N_stages', 'feed_stage', 'reflux_ratio', 'reactions'],
        'optional_specs': {
            'reactive_stages': {'description': 'Range of reactive stages [start, end]'},
            'catalyst_loading': {'unit': 'kg'},
            'holdup': {'unit': 'm3'},
            'condenser_type': {'values': ['total', 'partial'], 'default': 'total'},
            'reboiler_type': {'values': ['kettle', 'thermosiphon'], 'default': 'kettle'},
            'reaction_model': {'values': ['equilibrium', 'kinetic'], 'default': 'equilibrium'},
        },
        'dof_notes': 'Reactions occur on specified stages.',
    },
    'Absorber': {
        'description': 'Gas absorption column (no reboiler/condenser)',
        'category': 'separation',
        'phase_support': ['VLE'],
        'required_specs': ['N_stages'],
        'optional_specs': {
            'P': {'unit': 'bar'},
            'stage_efficiency': {'default': 0.7},
            'column_type': {'values': ['tray', 'packed'], 'default': 'packed'},
        },
        'dof_notes': 'Two feeds: gas at bottom, liquid solvent at top.',
    },
    'Stripper': {
        'description': 'Stripping column (reboiler, no condenser)',
        'category': 'separation',
        'phase_support': ['VLE'],
        'required_specs': ['N_stages', 'reboiler_duty|boilup_ratio'],
        'optional_specs': {
            'reboiler_type': {'values': ['kettle', 'thermosiphon'], 'default': 'kettle'},
            'P': {'unit': 'bar'},
        },
        'dof_notes': 'Liquid feed at top, vapor product at top.',
    },

    # =========================================================================
    # LLE SEPARATION
    # =========================================================================
    'Decanter': {
        'description': 'Liquid-liquid phase separator (settler)',
        'category': 'separation',
        'phase_support': ['LLE'],
        'required_specs': [],
        'optional_specs': {
            'T': {'unit': 'K or C', 'description': 'Optional isothermal operating temperature'},
            'P': {'unit': 'bar', 'description': 'Optional operating pressure'},
            'P_drop': {'unit': 'bar', 'default': 0.0},
            'Q': {'unit': 'kW', 'description': 'Optional heat duty target'},
            'mode': {'values': ['adiabatic', 'isothermal'], 'default': 'adiabatic'},
            'thermo_model': {'values': ['NRTL', 'UNIQUAC', 'UNIFAC'], 'default': 'NRTL'},
            'heavy_component': {'description': 'Fallback component for heavy phase selection if densities are unavailable'},
            'lle_tolerance': {'default': 1e-6},
        },
        'calculated': [
            'light_phase_frac', 'heavy_phase_frac', 'phase_compositions',
            'phase_densities', 'heat_duty', 'enthalpy_residual'
        ],
        'dof_notes': (
            'Adiabatic by default; optional T/temperature makes the decanter '
            'isothermal. Vapor feeds require Flash3/ThreePhaseFlash.'
        ),
    },
    'FlashLLE': {
        'description': 'Alias for Decanter liquid-liquid equilibrium separation',
        'category': 'separation',
        'phase_support': ['LLE'],
        'required_specs': [],
        'optional_specs': {
            'T': {'unit': 'K or C'},
            'P': {'unit': 'bar'},
            'P_drop': {'unit': 'bar', 'default': 0.0},
            'thermo_model': {'values': ['NRTL', 'UNIQUAC', 'UNIFAC'], 'default': 'NRTL'},
        },
        'calculated': ['light_phase_frac', 'heavy_phase_frac', 'phase_compositions'],
        'dof_notes': 'Uses Decanter ports light/heavy; vapor feeds require Flash3/ThreePhaseFlash.',
    },
    'LLExtractor': {
        'description': 'Multi-stage liquid-liquid extraction column',
        'category': 'separation',
        'phase_support': ['LLE'],
        'required_specs': ['N_stages', 'solvent_ratio'],
        'optional_specs': {
            'column_type': {'values': ['packed', 'tray', 'RDC', 'Karr', 'Scheibel', 'mixer_settler'], 'default': 'packed'},
            'T': {'unit': 'C'},
            'stage_efficiency': {'default': 0.8},
            'extract_phase': {'values': ['light', 'heavy'], 'default': 'light'},
            'HETS': {'unit': 'm', 'description': 'Height equivalent to theoretical stage'},
        },
        'calculated': ['extract_composition', 'raffinate_composition', 'stage_profiles'],
        'dof_notes': 'Feed and solvent enter at opposite ends.',
    },
    'CentrifugalExtractor': {
        'description': 'High-speed centrifugal liquid-liquid contactor',
        'category': 'separation',
        'phase_support': ['LLE'],
        'required_specs': ['N_stages', 'solvent_ratio'],
        'optional_specs': {
            'rpm': {'description': 'Rotational speed'},
            'stage_efficiency': {'default': 0.95},
        },
        'dof_notes': 'High efficiency for difficult separations.',
    },

    # =========================================================================
    # REACTORS
    # =========================================================================
    'Reactor': {
        'description': 'Conversion reactor (specified conversion)',
        'category': 'reactor',
        'phase_support': ['VLE', 'LLE'],
        'required_specs': ['reactions'],
        'optional_specs': {
            'T_out': {'unit': 'C'},
            'P_drop': {'unit': 'bar', 'default': 0},
            'mode': {'values': ['isothermal', 'adiabatic', 'duty'], 'default': 'isothermal'},
            'Q': {'unit': 'kW'},
            'desired_product': {'description': 'Optional yield/selectivity product'},
            'report_basis': {'description': 'Basis for desired-product reporting'},
        },
        'calculated': [
            'outlet_composition', 'heat_duty', 'T_out', 'reaction_extents',
            'component_conversion', 'yield', 'selectivity',
        ],
        'dof_notes': 'Specify inlet-basis conversion for each reaction; isothermal mode defaults to inlet temperature.',
    },
    'EquilibriumReactor': {
        'description': 'Homogeneous stoichiometric chemical-equilibrium reactor',
        'category': 'reactor',
        'phase_support': ['vapor', 'liquid'],
        'required_specs': ['reactions', 'phase'],
        'optional_specs': {
            'T_out': {'unit': 'C'},
            'mode': {'values': ['isothermal', 'adiabatic', 'duty'], 'default': 'isothermal'},
            'Q': {'unit': 'kW'},
            'P_drop': {'unit': 'bar', 'default': 0},
            'phase': {'values': ['vapor', 'liquid']},
        },
        'calculated': [
            'equilibrium_composition', 'equilibrium_constant', 'reaction_quotient',
            'extent', 'heat_duty', 'T_out', 'phase_stability',
        ],
        'dof_notes': 'Requires independent reversible reactions and one explicit homogeneous phase.',
    },
    'GibbsReactor': {
        'description': 'Gibbs free energy minimization reactor',
        'category': 'reactor',
        'phase_support': ['VLE'],
        'required_specs': ['T|adiabatic', 'P'],
        'optional_specs': {
            'inerts': {'description': 'Non-participating components'},
            'possible_products': {'description': 'Limit product species'},
        },
        'calculated': ['equilibrium_composition'],
        'dof_notes': 'No reactions needed - finds minimum Gibbs energy.',
    },
    'PFR': {
        'description': 'Plug flow reactor',
        'category': 'reactor',
        'phase_support': ['vapor', 'liquid'],
        'required_specs': ['volume|length+diameter', 'phase', 'kinetics'],
        'optional_specs': {
            'mode': {
                'values': ['isothermal', 'adiabatic', 'duty', 'jacketed'],
                'default': 'isothermal',
            },
            'T': {'unit': 'C'},
            'P_drop_model': {
                'values': ['none', 'specified', 'darcy', 'Ergun'],
                'default': 'none',
            },
            'void_fraction': {'default': 0.4},
            'particle_diameter': {'unit': 'm'},
            'roughness': {'unit': 'm'},
            'n_tubes': {'default': 1},
            'solver': {'values': ['auto', 'RK45', 'BDF', 'Radau', 'LSODA']},
        },
        'calculated': [
            'conversion_profile', 'T_profile', 'P_profile',
            'reaction_rate_profile', 'residence_time', 'heat_duty',
            'phase_stability', 'solver_diagnostics',
        ],
        'dof_notes': (
            'Requires explicitly unit-declared kinetics and a homogeneous phase. '
            'Jacketed and pressure-drop models require physical tube geometry.'
        ),
    },
    'PackedBedReactor': {
        'description': 'Fixed-catalyst packed-bed reactor',
        'category': 'reactor',
        'phase_support': ['vapor', 'liquid'],
        'required_specs': [
            'diameter', 'bulk_catalyst_density', 'bed_void_fraction',
            'length|bed_volume|catalyst_mass', 'catalyst-mass kinetics',
        ],
        'optional_specs': {
            'mode': {
                'values': ['isothermal', 'adiabatic', 'duty', 'jacketed'],
                'default': 'isothermal',
            },
            'T': {'unit': 'C'},
            'pressure_drop_model': {
                'values': ['none', 'specified', 'Ergun'],
                'default': 'none',
            },
            'particle_diameter': {'unit': 'm'},
            'kinetic_basis': {
                'values': ['apparent', 'intrinsic'],
                'default': 'apparent',
            },
            'diffusion_model': {
                'values': [
                    'none', 'specified', 'first_order_sphere',
                    'generalized_power_law_sphere', 'rigorous_power_law_sphere',
                ],
                'default': 'none',
            },
            'effective_diffusivity': {'unit': 'm2/s'},
            'effectiveness_factor': {'description': 'Specified factor in (0,1]'},
            'diffusion_limiting_component': {
                'description': 'Shared pellet diffusion-limiting reactant'
            },
            'effectiveness_factor_policy': {
                'values': ['constant_inlet', 'local'],
                'default': 'constant_inlet',
            },
            'effectiveness_factor_resolves': {
                'description': 'Rigorous pellet support solves, 1-101',
                'default': 1,
            },
            'pellet_relative_tolerance': {'default': 1.0e-6},
            'pellet_maximum_nodes': {'default': 2000},
        },
        'calculated': [
            'conversion_profile', 'catalyst_mass_profile', 'T_profile',
            'P_profile', 'reaction_rate_profile', 'effectiveness_factor_profile',
            'residence_time', 'heat_duty', 'phase_stability',
        ],
        'dof_notes': (
            'Rates must use catalyst-mass units. Bulk catalyst density means '
            'kg catalyst per total packed-bed volume. A calculated common '
            'factor requires one positive-order limiting species shared by '
            'compatible power-law reactions.'
        ),
    },
    'CSTR': {
        'description': 'Continuous stirred tank reactor',
        'category': 'reactor',
        'phase_support': ['vapor', 'liquid'],
        'required_specs': ['volume', 'phase', 'kinetics'],
        'optional_specs': {
            'mode': {
                'values': ['isothermal', 'adiabatic', 'duty', 'jacketed'],
                'default': 'isothermal',
            },
            'T': {'unit': 'C'},
            'P_drop': {'unit': 'bar', 'default': 0.0},
            'UA': {'unit': 'W/K'},
            'U': {'unit': 'W/m2-K'},
            'A_heat': {'unit': 'm2'},
            'T_jacket': {'unit': 'C'},
            'thermal_branch': {
                'values': ['lowest', 'nearest_inlet', 'highest'],
            },
        },
        'calculated': [
            'outlet_composition', 'reaction_rates', 'reaction_extents',
            'heat_duty', 'residence_time', 'phase_stability',
        ],
        'dof_notes': (
            'Specify volume, homogeneous phase, and explicitly unit-declared '
            'kinetics. Thermal modes solve material/rate and full enthalpy '
            'balances together; multiple roots require thermal_branch.'
        ),
    },
    'BatchReactor': {
        'description': 'Transient stirred batch/semi-batch reactor on a continuous-equivalent basis',
        'category': 'reactor',
        'phase_support': ['vapor', 'liquid'],
        'required_specs': ['two of V_batch|N|t_rxn', 'phase', 'kinetics'],
        'optional_specs': {
            'mode': {
                'values': ['isothermal', 'adiabatic', 'duty', 'jacketed'],
                'default': 'isothermal',
            },
            'semi_batch_feeds': {'description': 'Scheduled inlet port names'},
            'feed_start_times': {'description': 'port:hours mappings'},
            'feed_stop_times': {'description': 'port:hours mappings'},
            't_fill': {'unit': 'h'},
            't_drain': {'unit': 'h'},
            't_turnaround': {'unit': 'h', 'default': 0.0},
            'Q_batch': {'unit': 'kJ'},
            'UA': {'unit': 'W/K'},
            'T_jacket': {'unit': 'C'},
        },
        'calculated': [
            'time_averaged_outlet', 'batch_frequency', 'cycle_time',
            'fleet_utilization', 'transient_profile', 'heat_per_batch',
            'average_heat_duty', 'phase_stability',
        ],
        'dof_notes': (
            'Exactly two of V_batch, integer N, and t_rxn determine the '
            'continuous-equivalent fleet schedule. Unspecified fill and drain '
            'times equal the batch start interval.'
        ),
    },

    # =========================================================================
    # SOLID HANDLING
    # =========================================================================
    'Crystallizer': {
        'description': 'Equilibrium or steady MSMPR suspension crystallizer',
        'category': 'solid',
        'phase_support': ['SLE'],
        'required_specs': ['T_out|T'],
        'optional_specs': {
            **_CRYSTALLIZER_COMMON_OPTIONAL_SPECS,
            'model': {'values': ['equilibrium', 'MSMPR'], 'default': 'equilibrium'},
            'residence_time': {'unit': 'h'},
            'volume': {'unit': 'm3'},
            'msmpr_tolerance': {'unit': 'kmol/h', 'default': 1e-8},
            'msmpr_relative_tolerance': {'default': 0.0},
            'outlet_sphericity': {
                'description': 'Assumed outlet-crystal sphericity in (0, 1]',
            },
            'growth_*': {
                'description': 'MSMPR crystal-growth kinetic definition',
            },
            'nucleation_*': {
                'description': 'MSMPR nucleation kinetic definition',
            },
        },
        'calculated': [
            'solid_component_flows', 'crystal_yields',
            'mother_liquor_composition', 'supersaturation',
            'particle_size_distribution', 'particle_sphericity', 'heat_duty',
        ],
        'dof_notes': (
            'Specify outlet temperature. Pressure defaults to inlet pressure; '
            'without a retention specification, solid and mother liquor remain '
            'in one slurry outlet.'
        ),
    },
    'LayerCrystallizer': {
        'description': (
            'Equilibrium, mechanistic, or empirical layer crystallizer with '
            'optional bulk cooling and liquid inclusions'
        ),
        'category': 'solid',
        'phase_support': ['SLE'],
        'required_specs': ['T_out|T'],
        'optional_specs': {
            **_CRYSTALLIZER_COMMON_OPTIONAL_SPECS,
            'model': {
                'values': [
                    'equilibrium', 'layer_growth', 'empirical_layer_growth',
                ],
                'default': 'equilibrium',
            },
            'cooled_area': {'unit': 'm2'},
            'film_thickness': {'unit': 'm'},
            'thermal_film_thickness': {'unit': 'm'},
            'thermal_mode': {'values': ['isothermal', 'cooling']},
            'film_model': {'values': ['specified', 'flat_plate']},
            'plate_length': {'unit': 'm'},
            'liquid_velocity': {'unit': 'm/s'},
            'inclusion_max_fraction': {'default': 0.0},
            'growth_time': {'unit': 'h'},
            'cycle_time': {'unit': 'h'},
            'T_wall': {'unit': 'K'},
            'binary_diffusivity': {'unit': 'm2/s'},
            'layer_relative_tolerance': {'default': 1e-6},
            'empirical_relative_tolerance': {'default': 1e-7},
            'layer_profile_points': {'default': 21},
            'layer_solid_density': {'unit': 'kg/m3'},
            'growth_*': {
                'description': 'Empirical layer-growth kinetic definition',
            },
            'effective_distribution_*': {
                'description': 'Empirical impurity distribution definitions',
            },
            'sweat_heater_temperature': {'unit': 'K'},
            'sweat_thermal_conductance': {'unit': 'W/K'},
            'sweat_opening_coefficient': {},
            'sweat_collection_temperature': {'unit': 'K'},
            'harvest_temperature': {'unit': 'K'},
            'sweat_time': {'unit': 'h'},
            'sweat_host_rate_constant': {'unit': '1/s'},
            'sweat_drainage_length': {'unit': 'm'},
            'sweat_pore_radius': {'unit': 'm'},
            'sweat_tortuosity': {},
            'sweat_connected_fraction': {},
            'sweat_residual_saturation': {},
            'sweat_capillary_pressure': {'unit': 'Pa'},
            'solid_diffusion_length': {'unit': 'm'},
            'occluded_liquid_host_fraction': {},
            'occluded_fraction_*': {},
            'solid_partition_*': {},
            'solid_transfer_enthalpy_*': {'unit': 'kJ/mol'},
            'solid_diffusivity_*': {'unit': 'm2/s'},
        },
        'calculated': [
            'solid_component_flows', 'crystal_yields',
            'mother_liquor_composition', 'layer_thickness', 'sweat_composition',
            'harvest_product_composition', 'impurity_rejection_to_sweat',
            'heat_duty',
        ],
        'dof_notes': (
            'Layer material exits through cake and mother_liquor outlets. '
            'Equilibrium mode defaults to complete mother-liquor drainage. '
            'Enabling sweating replaces cake with final melted product and '
            'requires product, mother_liquor, and sweat outlets.'
        ),
    },
    'Filter': {
        'description': 'Cycle pressure cake filtration, washing and capillary deliquoring',
        'category': 'solid',
        'phase_support': ['SL'],
        'required_specs': ['cycle_time', 'P_drop|area', 'porosity', 'capture_cut_size'],
        'optional_specs': {
            'area': {'unit': 'm2', 'description': 'Omit for sizing; specify for rating'},
            'specific_cake_resistance': {'unit': 'm/kg', 'description': 'Overrides PSD-derived Kozeny-Carman resistance'},
            'kozeny_constant': {'default': 5},
            'capture_sharpness': {'default': 4},
            'medium_resistance': {'unit': '1/m', 'default': 0},
            'compressibility': {'default': 0},
            'reference_pressure': {'unit': 'bar', 'default': 1},
            'downtime': {'unit': 's', 'default': 0},
            'wash_cells': {'default': 10},
            'washing_model': {'values': ['isothermal', 'equilibrium'], 'default': 'isothermal'},
            'wash_steps': {'default': 100},
            'equilibrium_tolerance': {'default': 1e-8},
            'T_equilibrium_min': {'unit': 'K'},
            'T_equilibrium_max': {'unit': 'K'},
            'equilibrium_nucleus_diameter': {'unit': 'm'},
            'liquid_viscosity': {'unit': 'Pa*s'},
            'wash_viscosity': {'unit': 'Pa*s'},
            'deliquoring_time': {'unit': 's', 'default': 0},
            'deliquoring_pressure': {'unit': 'bar'},
            'entry_pressure': {'unit': 'bar'},
            'residual_saturation': {},
            'pore_index': {},
            'relative_permeability_exponent': {},
        },
        'calculated': ['filtrate_flow', 'cake_flow', 'required_area', 'cake_saturation', 'heat_duty'],
        'dof_notes': 'Optional wash inlet sets wash amount. Positive deliquoring_time additionally requires entry_pressure, residual_saturation and pore_index.',
    },
    'Dryer': {
        'description': 'Remove moisture from solids',
        'category': 'solid',
        'phase_support': ['SG'],
        'required_specs': ['outlet_moisture|T_out'],
        'optional_specs': {
            'type': {'values': ['rotary', 'spray', 'fluidized_bed', 'tray'], 'default': 'rotary'},
            'gas_flow': {'unit': 'kg/h'},
        },
        'calculated': ['heat_duty', 'evaporation_rate'],
        'dof_notes': 'Specify target moisture or outlet temperature.',
    },

    # =========================================================================
    # MEMBRANE AND ADSORPTION
    # =========================================================================
    'MolecularSieveDryer': {
        'description': 'Competitive equilibrium adsorption on molecular sieves',
        'category': 'separation',
        'phase_support': ['vapor', 'liquid', 'VLE'],
        'required_specs': ['adsorbent_mass_flow|target_mole_fraction|target_water_mole_fraction|removal_fraction'],
        'optional_specs': {
            'sieve_type': {'default': '3A'},
            'target_component': {},
            'isotherms': {'description': 'Per-component pure isotherm settings'},
            'initial_loadings': {'unit': 'kg/kg'},
            'kinetic_diameters': {'unit': 'angstrom'},
            'pore_diameter': {'unit': 'angstrom'},
        },
        'calculated': ['adsorbent_mass_flow_kg_h', 'removed_kmol_h', 'equilibrium_loadings_mol_per_kg'],
        'dof_notes': 'Specify exactly one sieve flow or target. All components with valid isotherms compete through IAST.',
    },
    'Membrane': {
        'description': 'Membrane separation unit',
        'category': 'separation',
        'phase_support': ['vapor', 'liquid'],
        'required_specs': ['area|recovery'],
        'optional_specs': {
            'type': {'values': ['gas', 'RO', 'NF', 'UF', 'pervaporation'], 'default': 'gas'},
            'permeances': {'description': 'Per component'},
            'selectivity': {'description': 'Relative to reference'},
            'P_permeate': {'unit': 'bar'},
            'module_type': {'values': ['hollow_fiber', 'spiral_wound', 'plate'], 'default': 'hollow_fiber'},
        },
        'calculated': ['permeate_composition', 'retentate_composition', 'stage_cut'],
        'dof_notes': 'Specify area or key component recovery.',
    },
    'PSA': {
        'description': 'Pressure swing adsorption',
        'category': 'separation',
        'phase_support': ['vapor'],
        'required_specs': ['P_high', 'P_low', 'product_purity|recovery'],
        'optional_specs': {
            'adsorbent': {'values': ['zeolite', 'carbon', 'alumina', 'silica']},
            'cycle_time': {'unit': 's'},
            'n_beds': {'default': 2},
        },
        'calculated': ['product_flow', 'tail_gas_flow', 'power'],
        'dof_notes': 'Specify pressures and target purity or recovery.',
    },
    'TSA': {
        'description': 'Temperature swing adsorption',
        'category': 'separation',
        'phase_support': ['vapor'],
        'required_specs': ['T_ads', 'T_regen', 'product_purity|recovery'],
        'optional_specs': {
            'adsorbent': {'values': ['zeolite', 'carbon', 'silica_gel']},
            'cycle_time': {'unit': 'min'},
        },
        'calculated': ['product_flow', 'regeneration_duty'],
        'dof_notes': 'Specify adsorption and regeneration temperatures.',
    },
}


# These registered columns are closed by inlet states and runtime defaults;
# optional operating settings are not missing degrees of freedom.
_COLUMN_SPECS = {
    'N_stages': {'description': 'Optional stage count; the unit supplies its default'},
    'T': {'unit': 'C', 'description': 'Optional operating temperature'},
    'mode': {'description': 'Thermal mode, where supported'},
    'P': {'unit': 'bar'}, 'P_drop_per_stage': {'unit': 'bar'},
}
UNIT_DOF_RULES['ShortcutDistillation']['required_specs'] = []
UNIT_DOF_RULES['ShortcutDistillation']['dof_notes'] = 'Fenske–Underwood–Gilliland design supplies missing stages/reflux from key recoveries and runtime defaults. Explicit operating targets override those defaults.'
for _name, _description, _phases in (
    ('RigorousDistillation', 'Equation-oriented MESH distillation column', ['VLE', 'VLLE']),
    ('CMODistillation', 'Multicomponent constant-molar-overflow column', ['VLE']),
    ('McCabeThieleDistillation', 'Binary McCabe-Thiele column', ['VLE']),
    ('ShortcutExtractor', 'Shortcut counter-current liquid-liquid extraction', ['LLE']),
    ('RigorousExtractor', 'Rigorous equilibrium-stage liquid-liquid extraction', ['LLE']),
    ('RigorousAbsorber', 'Rigorous equilibrium-stage absorption', ['VLE']),
    ('RigorousStripper', 'Rigorous equilibrium-stage stripping', ['VLE']),
):
    UNIT_DOF_RULES[_name] = {
        'description': _description, 'category': 'separation',
        'phase_support': _phases, 'required_specs': [],
        'optional_specs': dict(_COLUMN_SPECS),
        'dof_notes': 'Inlet states and runtime defaults close the column. Stage, pressure and operating settings may override those defaults.',
    }
for _name in ('RigorousDistillation', 'CMODistillation', 'McCabeThieleDistillation'):
    UNIT_DOF_RULES[_name]['optional_specs'].update({
        'reflux_ratio': {}, 'D_to_F': {}, 'D_rate': {'unit': 'kmol/h'},
        'feed_stage': {}, 'P_condenser': {'unit': 'bar'},
    })

# UI importance is independent of whether a runtime default closes a DOF.
_PRIMARY_SPECS = {
    'distillation': ('N_stages', 'stages', 'feed_stage', 'feed_stages', 'reflux_ratio', 'RR',
        'D_to_F', 'D_rate', 'D_mass', 'distillate_flow', 'light_key', 'heavy_key',
        'light_key_recovery', 'heavy_key_recovery', 'condenser_type', 'P_condenser', 'P_top'),
    'extractor': ('N_stages', 'stages', 'T', 'mode', 'extract_phase', 'heavy_component'),
    'absorber': ('N_stages', 'stages', 'mode', 'T', 'stage_temperature', 'P', 'P_top', 'gas_stage', 'liquid_stage'),
    'stripper': ('N_stages', 'stages', 'mode', 'T', 'stage_temperature', 'P', 'P_top', 'stripping_gas_stage', 'liquid_stage'),
    'pump': ('P_out', 'delta_P', 'pressure_ratio', 'eta', 'eta_mech'),
    'compressor': ('P_out', 'delta_P', 'pressure_ratio', 'eta_isen', 'eta_mech'),
    'expander': ('P_out', 'delta_P', 'pressure_ratio', 'eta_isen', 'eta_mech'),
    'valve': ('P_out',),
    'pipe': ('length', 'diameter', 'velocity', 'diameter_out', 'roughness', 'elevation_change'),
    'heat_exchanger': ('T_hot_out', 'T_cold_out', 'T_tube_out', 'T_shell_out', 'Q', 'UA', 'U', 'A', 'area', 'estimate_U'),
    'reactor': ('volume', 'V', 'length', 'diameter', 'phase', 'mode', 'T', 'P', 'catalyst_mass'),
    'batch_reactor': ('V_batch', 'N', 't_rxn', 'phase', 'mode', 'T', 'P'),
    'filter': ('cycle_time', 'P_drop', 'area', 'porosity', 'capture_cut_size', 'washing_model', 'deliquoring_time'),
    'dryer': ('sieve_type', 'adsorbent_mass_flow', 'target_component', 'target_mole_fraction', 'target_water_mole_fraction', 'removal_fraction'),
    'splitter': ('split_frac', 'split_fractions', 'flows', 'outlet_flows', 'mass_flows'),
}


class DOFAnalyzer:
    """Analyzes degrees of freedom for process specifications"""
    
    def __init__(self, pfd):
        self.pfd = pfd
        self.n_components = len(pfd.components)
        
    def analyze(self) -> ProcessDOFResult:
        """Run complete DOF analysis on the process"""
        result = ProcessDOFResult(
            overall_status=SpecificationStatus.OK,
            total_dof=0
        )
        
        for stream in self.pfd.streams:
            stream_result = self._analyze_stream(stream)
            result.stream_results.append(stream_result)
            if stream_result.status != SpecificationStatus.OK:
                if stream_result.status == SpecificationStatus.UNDER_SPECIFIED:
                    result.errors.append(stream_result.message)
                else:
                    result.warnings.append(stream_result.message)
        
        for unit in self.pfd.units:
            unit_result = self._analyze_unit(unit)
            result.unit_results.append(unit_result)
            result.total_dof += unit_result.dof
            
            if unit_result.status == SpecificationStatus.UNDER_SPECIFIED:
                result.errors.append(unit_result.message)
            elif unit_result.status == SpecificationStatus.OVER_SPECIFIED:
                result.errors.append(unit_result.message)
            elif unit_result.status == SpecificationStatus.WARNING:
                result.warnings.append(unit_result.message)
        
        if result.errors:
            if any(item.status is SpecificationStatus.UNDER_SPECIFIED for item in result.unit_results+result.stream_results):
                result.overall_status = SpecificationStatus.UNDER_SPECIFIED
            else:
                result.overall_status = SpecificationStatus.OVER_SPECIFIED
        elif result.warnings:
            result.overall_status = SpecificationStatus.WARNING
            
        result.suggestions = self._generate_suggestions(result)
        return result
    
    def _analyze_stream(self, stream) -> DOFResult:
        """Analyze DOF for a single stream"""
        is_feed = stream.source.is_feed
        
        has_T = any(p.name.upper() == 'T' for p in stream.properties)
        has_vapor_fraction = any(
            p.name.upper() in ('VAP_FRAC', 'VAPOR_FRAC', 'VAPOR_FRACTION', 'VF')
            for p in stream.properties
        )
        has_thermal_spec = has_T or has_vapor_fraction
        has_P = any(p.name.upper() == 'P' for p in stream.properties)
        has_F = any(
            p.name.upper() in ('F', 'FLOW', 'MOLAR_FLOW', 'F_MASS', 'MASS_FLOW', 'FLOW_MASS')
            for p in stream.properties
        )
        has_composition = stream.composition is not None
        
        specs_provided = sum([has_thermal_spec, has_P, has_F, has_composition])
        
        if is_feed:
            required_specs = 4
            dof = required_specs - specs_provided
            
            missing = []
            if not has_thermal_spec:
                missing.append('T or vapor_fraction')
            if not has_P:
                missing.append('P')
            if not has_F:
                missing.append('F or F_mass')
            if not has_composition:
                missing.append('composition (x or w)')
            
            if dof > 0:
                return DOFResult(
                    entity_id=stream.id,
                    entity_type='stream',
                    total_variables=required_specs,
                    equations=0,
                    specifications=specs_provided,
                    dof=dof,
                    status=SpecificationStatus.UNDER_SPECIFIED,
                    message=f"FEED stream '{stream.id}' missing: {', '.join(missing)}",
                    details=["Feed streams require T or vapor_fraction, P, flow rate, and composition"]
                )
            
            if has_composition:
                total = sum(stream.composition.fractions.values())
                if abs(total - 1.0) > 0.001:
                    return DOFResult(
                        entity_id=stream.id,
                        entity_type='stream',
                        total_variables=required_specs,
                        equations=0,
                        specifications=specs_provided,
                        dof=0,
                        status=SpecificationStatus.WARNING,
                        message=f"Stream '{stream.id}' composition sums to {total:.4f}, not 1.0",
                        details=[]
                    )
        # Non-feed values are initialization data, not additional equations.
        # Tear-stream values seed recycles; upstream units determine the result.
        
        return DOFResult(
            entity_id=stream.id,
            entity_type='stream',
            total_variables=4,
            equations=0,
            specifications=specs_provided,
            dof=0,
            status=SpecificationStatus.OK,
            message=f"Stream '{stream.id}' OK",
            details=[]
        )
    
    def _analyze_unit(self, unit) -> DOFResult:
        """Analyze DOF for a single unit operation"""
        rules = get_unit_info(unit.unit_type)
        
        if not rules:
            return DOFResult(
                entity_id=unit.id,
                entity_type='unit',
                total_variables=0,
                equations=0,
                specifications=0,
                dof=0,
                status=SpecificationStatus.WARNING,
                message=f"Unknown unit type '{unit.unit_type}' - cannot analyze DOF",
                details=[f"Supported: {', '.join(sorted(set(UNIT_TYPE_ALIASES.values())))}"]
            )
        
        param_names = {p.name.lower() for p in unit.params}
        return self._check_unit_specs(unit, rules, param_names)
    
    def _check_unit_specs(self, unit, rules, param_names) -> DOFResult:
        """Check if unit has correct specifications"""
        unit_type = canonical_unit_type(unit.unit_type) or unit.unit_type
        details = []
        dof = 0
        status = SpecificationStatus.OK
        message = f"Unit '{unit.id}' ({unit_type}) OK"
        
        if unit_type == 'Mixer':
            has_t_out = any(p in param_names for p in ['t_out', 'tout', 'temperature', 't'])
            has_q = any(p in param_names for p in ['q', 'duty', 'heat_duty'])
            if has_t_out and has_q:
                dof = -1
                status = SpecificationStatus.OVER_SPECIFIED
                message = f"Unit '{unit.id}' has both outlet temperature and heat duty"
                details.append("Specify at most one of T_out/T/temperature or Q/duty/heat_duty")
            
        elif unit_type == 'MolecularSieveDryer':
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from .molecular_sieve import MolecularSieveDryer
            else:
                from molecular_sieve import MolecularSieveDryer
            names = set(MolecularSieveDryer.ADSORBENT_MASS_NAMES) | {
                'target_mole_fraction', 'target_water_mole_fraction', 'removal_fraction',
            }
            count = len(names & param_names)
            dof = 1-count
            if dof:
                status = SpecificationStatus.UNDER_SPECIFIED if dof>0 else SpecificationStatus.OVER_SPECIFIED
                message = f"Unit '{unit.id}' requires exactly one sieve flow or sizing target"
                details.append('Specify a target component for non-water sizing; isotherms determine every coadsorbate.')

        elif unit_type == 'Splitter':
            has_molar_flow_specs = any(
                p.startswith(('f_', 'flow_', 'molar_flow_'))
                or p.endswith(('_f', '_flow', '_molar_flow'))
                or p in ('flows', 'outlet_flows')
                for p in param_names
            )
            has_mass_flow_specs = any(
                p.startswith(('mass_flow_', 'f_mass_'))
                or p.endswith(('_mass_flow', '_f_mass'))
                or p in ('mass_flows', 'outlet_mass_flows')
                for p in param_names
            )
            if has_molar_flow_specs and has_mass_flow_specs:
                dof = -1
                status = SpecificationStatus.OVER_SPECIFIED
                message = f"Unit '{unit.id}' mixes molar-flow and mass-flow split specs"
                details.append("Use either molar outlet flows or mass outlet flows, not both")
                
        elif unit_type in ['Pump', 'Compressor', 'Expander', 'Valve']:
            absolute_pressure_specs = ['p_out', 'pout', 'outlet_p', 'pressure', 'p']
            if unit_type in ['Pump', 'Compressor', 'Expander']:
                has_absolute = any(p in param_names for p in absolute_pressure_specs)
                delta_names = (
                    ['delta_p', 'dp', 'p_rise', 'pressure_rise']
                    if unit_type in ['Pump', 'Compressor']
                    else ['delta_p', 'dp', 'p_drop', 'pressure_drop']
                )
                has_delta = any(
                    p in param_names
                    for p in delta_names
                )
                has_ratio = any(
                    p in param_names
                    for p in ['pressure_ratio', 'p_ratio', 'ratio']
                )
                n_pressure_specs = sum([has_absolute, has_delta, has_ratio])
                if n_pressure_specs < 1:
                    dof = 1
                    status = SpecificationStatus.UNDER_SPECIFIED
                    message = (
                        f"Unit '{unit.id}' missing {unit_type.lower()} pressure target "
                        "(P_out, delta_P, or pressure_ratio)"
                    )
                elif n_pressure_specs > 1:
                    dof = 1 - n_pressure_specs
                    status = SpecificationStatus.OVER_SPECIFIED
                    message = (
                        f"Unit '{unit.id}' has {n_pressure_specs} {unit_type.lower()} pressure "
                        "target groups - specify only one of P_out, delta_P, "
                        "or pressure_ratio"
                    )
            else:
                has_p_out = any(p in param_names for p in absolute_pressure_specs)
                if not has_p_out:
                    dof = 1
                    status = SpecificationStatus.UNDER_SPECIFIED
                    message = f"Unit '{unit.id}' missing P_out specification"
            if unit_type == 'Compressor':
                has_eta_isen = any('isen' in p for p in param_names)
                has_eta_poly = any('poly' in p for p in param_names)
                if status == SpecificationStatus.OK and has_eta_isen and has_eta_poly:
                    status = SpecificationStatus.WARNING
                    message = f"Unit '{unit.id}' has both eta_isen and eta_poly; eta_poly is not yet implemented"
                elif status == SpecificationStatus.OK and has_eta_poly:
                    status = SpecificationStatus.WARNING
                    message = f"Unit '{unit.id}' specifies eta_poly, which is not yet implemented"

        elif unit_type == 'Pipe':
            has_length = any(name in param_names for name in ('length', 'pipe_length', 'l'))
            has_diameter = any(name in param_names for name in (
                'diameter', 'd', 'pipe_diameter', 'diameter_in',
                'inlet_diameter', 'd_in',
            ))
            has_velocity = any(name in param_names for name in (
                'velocity', 'target_velocity', 'inlet_velocity',
            ))
            missing = []
            if not has_length:
                missing.append('length')
            if not has_diameter and not has_velocity:
                missing.append('diameter or velocity')
            if missing:
                dof = len(missing)
                status = SpecificationStatus.UNDER_SPECIFIED
                message = f"Unit '{unit.id}' missing: {', '.join(missing)}"
            elif has_diameter and has_velocity:
                dof = -1
                status = SpecificationStatus.OVER_SPECIFIED
                message = (
                    f"Unit '{unit.id}' specifies both diameter and velocity; "
                    "choose one inlet sizing method"
                )
                    
        elif unit_type in ['Heater', 'Cooler']:
            has_t_out = any(p in param_names for p in ['t_out', 'tout', 'temperature', 't'])
            has_q = any(p in param_names for p in ['q', 'duty', 'heat_duty'])
            has_vf = any(p in param_names for p in ['vap_frac', 'vapor_frac', 'vapor_fraction', 'vf'])
            
            n_specs = sum([has_t_out, has_q, has_vf])
            if n_specs > 1:
                dof = -1
                status = SpecificationStatus.OVER_SPECIFIED
                message = f"Unit '{unit.id}' has {n_specs} thermal specs - over-specified"
                details.append("Specify exactly ONE of: T_out, Q, or vapor_frac")
            elif n_specs < 1:
                dof = 1
                status = SpecificationStatus.UNDER_SPECIFIED
                message = f"Unit '{unit.id}' needs T_out, Q, or vapor_frac"

        elif unit_type == 'Filter':
            values = {param.name.lower(): param.value for param in unit.params}
            missing = [name for name in ('cycle_time', 'porosity', 'capture_cut_size') if name not in param_names]
            if not ({'p_drop', 'area'} & param_names):
                missing.append('P_drop or area')
            try:
                drains = float(values.get('deliquoring_time', 0)) > 0
            except (TypeError, ValueError):
                drains = False
            if drains:
                missing.extend(name for name in ('entry_pressure', 'residual_saturation', 'pore_index') if name not in param_names)
            if missing:
                dof = len(missing)
                status = SpecificationStatus.UNDER_SPECIFIED
                message = f"Unit '{unit.id}' requires: {', '.join(missing)}"

        elif unit_type in {'Crystallizer', 'LayerCrystallizer'}:
            try:
                validator = (
                    validate_layer_crystallizer_specification
                    if unit_type == 'LayerCrystallizer'
                    else validate_crystallizer_specification
                )
                validator(
                    {param.name: param.value for param in unit.params},
                    {port.id for port in unit.ports
                     if port.port_type.value.endswith('outlet')},
                )
            except CrystallizerSpecificationError as error:
                dof = error.dof
                status = (SpecificationStatus.UNDER_SPECIFIED if dof > 0
                          else SpecificationStatus.OVER_SPECIFIED)
                message = f"Unit '{unit.id}' {error}"

        elif unit_type == 'HeatExchanger':
            has_u = 'u' in param_names
            has_a = 'a' in param_names or 'area' in param_names
            has_auto_u = (
                'estimate_u' in param_names
                or 'u' in param_names
            )
            has_ua = any(p in param_names for p in ['ua', 'ua_available'])
            has_hot_out = any(p in param_names for p in ['t_hot_out', 'hot_t_out', 'hot_out_t'])
            has_cold_out = any(p in param_names for p in ['t_cold_out', 'cold_t_out', 'cold_out_t'])
            has_tube_out = any(p in param_names for p in ['t_tube_out', 'tube_t_out', 'tube_out_t'])
            has_shell_out = any(p in param_names for p in ['t_shell_out', 'shell_t_out', 'shell_out_t'])
            has_q = any(p in param_names for p in ['q', 'duty', 'heat_duty'])
            has_hot_vf = any(p in param_names for p in [
                'hot_vap_frac', 'hot_vapor_frac', 'hot_vapor_fraction',
                'vap_frac_hot', 'vapor_frac_hot', 'vapor_fraction_hot',
                'vap_frac_hot_out', 'vapor_frac_hot_out', 'vapor_fraction_hot_out',
                'hot_out_vap_frac', 'hot_out_vapor_frac', 'hot_out_vapor_fraction',
                'vf_hot', 'vf_hot_out', 'hot_vf', 'hot_out_vf',
            ])
            has_cold_vf = any(p in param_names for p in [
                'cold_vap_frac', 'cold_vapor_frac', 'cold_vapor_fraction',
                'vap_frac_cold', 'vapor_frac_cold', 'vapor_fraction_cold',
                'vap_frac_cold_out', 'vapor_frac_cold_out', 'vapor_fraction_cold_out',
                'cold_out_vap_frac', 'cold_out_vapor_frac', 'cold_out_vapor_fraction',
                'vf_cold', 'vf_cold_out', 'cold_vf', 'cold_out_vf',
            ])
            has_tube_vf = any(p in param_names for p in [
                'tube_vap_frac', 'tube_vapor_frac', 'tube_vapor_fraction',
                'vap_frac_tube', 'vapor_frac_tube', 'vapor_fraction_tube',
                'vap_frac_tube_out', 'vapor_frac_tube_out', 'vapor_fraction_tube_out',
                'tube_out_vap_frac', 'tube_out_vapor_frac', 'tube_out_vapor_fraction',
                'vf_tube', 'vf_tube_out', 'tube_vf', 'tube_out_vf',
            ])
            has_shell_vf = any(p in param_names for p in [
                'shell_vap_frac', 'shell_vapor_frac', 'shell_vapor_fraction',
                'vap_frac_shell', 'vapor_frac_shell', 'vapor_fraction_shell',
                'vap_frac_shell_out', 'vapor_frac_shell_out', 'vapor_fraction_shell_out',
                'shell_out_vap_frac', 'shell_out_vapor_frac', 'shell_out_vapor_fraction',
                'vf_shell', 'vf_shell_out', 'shell_vf', 'shell_out_vf',
            ])

            n_thermal_specs = sum([
                has_hot_out,
                has_cold_out,
                has_tube_out,
                has_shell_out,
                has_q,
                has_hot_vf,
                has_cold_vf,
                has_tube_vf,
                has_shell_vf,
            ])
            has_rating_capacity = has_ua or (has_u and has_a) or (has_auto_u and has_a)
            if n_thermal_specs == 1:
                dof = 0
            elif n_thermal_specs > 1:
                dof = 1 - n_thermal_specs
                status = SpecificationStatus.OVER_SPECIFIED
                message = f"Unit '{unit.id}' has multiple heat exchanger specifications"
                details.append("Specify exactly one of: T_hot_out, T_cold_out, T_tube_out, T_shell_out, Q, or one outlet vapor_frac")
            elif has_rating_capacity:
                dof = 0
            else:
                dof = 1
                status = SpecificationStatus.UNDER_SPECIFIED
                message = f"Unit '{unit.id}' needs an outlet target, Q, vapor_frac, UA, U+A, or A+auto-U"
                
        elif unit_type in ['Flash', 'Flash3']:
            has_t = 't' in param_names or 'temperature' in param_names
            has_p = 'p' in param_names or 'pressure' in param_names
            has_vf = any(p in param_names for p in ['vf', 'vapor_frac', 'vapor_fraction'])
            has_q = any(p in param_names for p in ['q', 'duty', 'heat_duty'])
            
            n_specs = sum([has_t, has_p, has_vf, has_q])
            if (
                unit_type == 'Flash'
                and not has_p
                and sum([has_t, has_vf, has_q]) == 1
            ):
                # An ordinary Flash inherits its connected inlet pressure when
                # exactly one thermal/quality specification is supplied.
                n_specs += 1
            if n_specs < 2:
                dof = 2 - n_specs
                status = SpecificationStatus.UNDER_SPECIFIED
                message = f"Unit '{unit.id}' needs 2 flash specs, has {n_specs}"
            elif n_specs > 2:
                dof = 2 - n_specs
                status = SpecificationStatus.OVER_SPECIFIED
                message = f"Unit '{unit.id}' over-specified with {n_specs} flash specs"
                
        elif unit_type in ['Decanter', 'FlashLLE']:
            has_t = 't' in param_names or 'temperature' in param_names
            has_q = any(p in param_names for p in ['q', 'duty', 'heat_duty'])
            if has_t and has_q:
                dof = -1
                status = SpecificationStatus.OVER_SPECIFIED
                message = f"Unit '{unit.id}' cannot specify both decanter temperature and heat duty"
                
        elif unit_type == 'ReactiveDistillation':
            has_stages = any(p in param_names for p in ['n_stages', 'nstages', 'stages'])
            has_feed_stage = any(p in param_names for p in ['feed_stage', 'feedstage'])
            has_reflux = any(p in param_names for p in ['reflux', 'reflux_ratio', 'rr'])
            has_product_spec = any(p in param_names for p in ['recovery', 'd_rate', 'distillate', 'purity', 'boilup'])
            
            missing = []
            if not has_stages:
                missing.append('N_stages')
            if unit_type == 'ReactiveDistillation' and not has_feed_stage:
                missing.append('feed_stage')
            if not has_reflux and not has_product_spec:
                missing.append('reflux_ratio or product spec')
            
            if unit_type == 'ReactiveDistillation' and len(unit.reactions) == 0:
                missing.append('reactions')
            
            if missing:
                dof = len(missing)
                status = SpecificationStatus.UNDER_SPECIFIED
                message = f"Unit '{unit.id}' missing: {', '.join(missing)}"
                
        elif unit_type in ['Reactor', 'EquilibriumReactor']:
            has_reactions = len(unit.reactions) > 0
            if not has_reactions:
                dof = 1
                status = SpecificationStatus.UNDER_SPECIFIED
                message = f"Unit '{unit.id}' has no reactions defined"
            elif unit_type == 'Reactor':
                missing_conv = []
                for i, rxn in enumerate(unit.reactions):
                    try:
                        resolved = self.pfd.resolve_reaction(rxn)
                    except ValueError:
                        resolved = rxn
                    if 'conversion' not in resolved.parameters:
                        missing_conv.append(f"reaction {i+1}")
                if missing_conv:
                    dof = len(missing_conv)
                    status = SpecificationStatus.UNDER_SPECIFIED
                    message = f"Unit '{unit.id}' needs conversion for: {', '.join(missing_conv)}"
            elif unit_type == 'EquilibriumReactor':
                if 'phase' not in param_names:
                    dof = 1
                    status = SpecificationStatus.UNDER_SPECIFIED
                    message = f"Unit '{unit.id}' needs phase=vapor or phase=liquid"
                    
        elif unit_type in ['PFR', 'CSTR']:
            has_volume = any(p in param_names for p in ['volume', 'v'])
            has_length_diameter = (
                any(p in param_names for p in ['length', 'l'])
                and any(p in param_names for p in ['diameter', 'd'])
            )
            has_kinetics = len(unit.reactions) > 0
            
            missing = []
            if not has_volume and not has_length_diameter:
                missing.append('volume or length+diameter')
            if not has_kinetics:
                missing.append('reactions with kinetics')
            if 'phase' not in param_names:
                missing.append('phase=vapor or phase=liquid')
            
            if missing:
                dof = len(missing)
                status = SpecificationStatus.UNDER_SPECIFIED
                message = f"Unit '{unit.id}' missing: {', '.join(missing)}"

        elif unit_type == 'PackedBedReactor':
            has_size = any(
                p in param_names for p in (
                    'length', 'l', 'bed_length', 'bed_volume', 'volume', 'v',
                    'catalyst_mass', 'w_cat', 'wcat',
                )
            )
            missing = []
            if not any(p in param_names for p in ('diameter', 'd', 'bed_diameter')):
                missing.append('diameter')
            if not has_size:
                missing.append('length, bed_volume, or catalyst_mass')
            if not any(p in param_names for p in (
                'bulk_catalyst_density', 'catalyst_bulk_density', 'rho_bulk',
            )):
                missing.append('bulk_catalyst_density')
            if not any(p in param_names for p in (
                'bed_void_fraction', 'void_fraction',
            )):
                missing.append('bed_void_fraction')
            if not unit.reactions:
                missing.append('reactions with catalyst-mass kinetics')
            if 'phase' not in param_names:
                missing.append('phase=vapor or phase=liquid')
            if missing:
                dof = len(missing)
                status = SpecificationStatus.UNDER_SPECIFIED
                message = f"Unit '{unit.id}' missing: {', '.join(missing)}"

        elif unit_type == 'BatchReactor':
            scheduling = sum(
                any(name in param_names for name in aliases)
                for aliases in (
                    ('v_batch', 'batch_volume'),
                    ('n', 'n_vessels', 'n_tanks'),
                    ('t_rxn', 't_reaction', 'reaction_time'),
                )
            )
            missing = []
            if scheduling < 2:
                missing.append('two of V_batch, N, and t_rxn')
            if len(unit.reactions) == 0:
                missing.append('reactions with kinetics')
            if 'phase' not in param_names:
                missing.append('phase=vapor or phase=liquid')
            if scheduling > 2:
                dof = -1
                status = SpecificationStatus.OVER_SPECIFIED
                message = (
                    f"Unit '{unit.id}' must specify exactly two of V_batch, "
                    "N, and t_rxn"
                )
            elif missing:
                dof = len(missing)
                status = SpecificationStatus.UNDER_SPECIFIED
                message = f"Unit '{unit.id}' missing: {', '.join(missing)}"
                
        elif unit_type == 'LLExtractor':
            has_stages = any(p in param_names for p in ['n_stages', 'stages'])
            has_sf = any(p in param_names for p in ['solvent_ratio', 's_f', 'sf'])
            
            missing = []
            if not has_stages:
                missing.append('N_stages')
            if not has_sf:
                missing.append('solvent_ratio')
            
            if missing:
                dof = len(missing)
                status = SpecificationStatus.UNDER_SPECIFIED
                message = f"Unit '{unit.id}' missing: {', '.join(missing)}"
        
        return DOFResult(
            entity_id=unit.id,
            entity_type='unit',
            total_variables=0,
            equations=0,
            specifications=len(unit.params),
            dof=dof,
            status=status,
            message=message,
            details=details
        )
    
    def _generate_suggestions(self, result: ProcessDOFResult) -> list[str]:
        """Generate suggestions based on analysis"""
        suggestions = []
        
        under_spec = [r for r in result.unit_results if r.status == SpecificationStatus.UNDER_SPECIFIED]
        over_spec = [r for r in result.unit_results if r.status == SpecificationStatus.OVER_SPECIFIED]
        
        if under_spec:
            suggestions.append(f"Add specifications to: {', '.join(r.entity_id for r in under_spec)}")
        if over_spec:
            suggestions.append(f"Remove conflicting specs from: {', '.join(r.entity_id for r in over_spec)}")
        
        under_feeds = [r for r in result.stream_results if r.status == SpecificationStatus.UNDER_SPECIFIED]
        if under_feeds:
            suggestions.append("Ensure all FEED streams have T, P, F, and composition")
        
        return suggestions


def analyze_dof(pfd) -> ProcessDOFResult:
    """Analyze DOF for a PFD"""
    return DOFAnalyzer(pfd).analyze()


def get_unit_info(unit_type: str) -> dict:
    """Get DOF rules for a unit type"""
    canonical = canonical_unit_type(unit_type)
    rules = UNIT_DOF_RULES.get(canonical, {})
    if not rules:
        return {}
    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
        from .unit_syntax import port_family_for_unit_type
    else:
        from unit_syntax import port_family_for_unit_type
    return {**rules, 'primary_specs': list(_PRIMARY_SPECS.get(port_family_for_unit_type(canonical), ()))}


def list_unit_types() -> list[str]:
    """List all supported unit types"""
    return list(dict.fromkeys(UNIT_TYPE_ALIASES.values()))


def get_units_by_category(category: str) -> list[str]:
    """Get unit types in a category"""
    return [name for name in list_unit_types() if get_unit_info(name).get('category') == category]
