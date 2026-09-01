"""
PFD Parser - A robust parser for Process Flow Diagram (.pfd) files

This module provides:
- Parsing of .pfd files into structured Python objects
- Validation of PFD structure and connections
- Serialization back to .pfd format
"""

import math
import re
from difflib import SequenceMatcher, get_close_matches
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_conversions import pressure_to_bar, temperature_to_kelvin
else:
    from unit_conversions import pressure_to_bar, temperature_to_kelvin
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .fluid_phase_models import normalize_fluid_phase_model
else:
    from fluid_phase_models import normalize_fluid_phase_model
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .phase_behaviors import normalize_phase_behavior
else:
    from phase_behaviors import normalize_phase_behavior
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .recycle_controls import (
        RECYCLE_METHOD_ALIASES,
        normalize_recycle_method,
        normalize_recycle_options,
    )
else:
    from recycle_controls import (
        RECYCLE_METHOD_ALIASES,
        normalize_recycle_method,
        normalize_recycle_options,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .solid_material_forms import normalize_solid_material_form
else:
    from solid_material_forms import normalize_solid_material_form
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_syntax import (
        NUMERIC_PORT_LAYOUTS,
        UNIT_TYPE_ALIASES,
        canonical_unit_type,
        port_family_for_unit_type,
        port_schema_for_unit_type,
        unit_allows_variable_port,
    )
else:
    from unit_syntax import (
        NUMERIC_PORT_LAYOUTS,
        UNIT_TYPE_ALIASES,
        canonical_unit_type,
        port_family_for_unit_type,
        port_schema_for_unit_type,
        unit_allows_variable_port,
    )

_ANTOINE_OVERRIDE_FIELDS = (
    'antoine_A',
    'antoine_B',
    'antoine_C',
    'antoine_Tmin',
    'antoine_Tmax',
)

_PSAT_EQUATION_COEFFICIENTS = {
    'poly_x': (('A',), frozenset('ABCDEF')),
    'exp_poly_x': (('A',), frozenset('ABCDEF')),
    'reduced_vapor_pressure': (('A', 'B', 'C', 'D'), frozenset('ABCD')),
    'psat_mercury': (tuple('ABCDEF'), frozenset('ABCDEF')),
    'canonical_psat': (tuple('ABCDEF'), frozenset('ABCDEF')),
    'canonical_psat_af': (tuple('ABCDEF'), frozenset('ABCDEF')),
    'canonical_psat_ag': (tuple('ABCDEFG'), frozenset('ABCDEFG')),
    'canonical_psat_ah': (tuple('ABCDEFGH'), frozenset('ABCDEFGH')),
}

_COMPONENT_PROPERTY_ALIASES = {
    'mw': 'molecular_weight',
    'molecular_weight': 'molecular_weight',
    'cas': 'CAS',
    'unifac': 'unifac_groups',
    'smiles': 'smiles',
    'rho_t': 'rho_T',
    't_rho': 'rho_T',
    'rho_t_k': 'rho_T',
    'cps': 'Cp_solid',
    'cp_solid': 'Cp_solid',
    'solid_cp': 'Cp_solid',
    'rhos': 'rho_solid',
    'rho_solid': 'rho_solid',
    'solid_density': 'rho_solid',
    'vms': 'Vm_solid',
    'vm_solid': 'Vm_solid',
    'solid_molar_volume': 'Vm_solid',
    'solid_form': 'solid_material_form',
    'phase_model': 'phase_behavior',
    'type': 'phase_behavior',
    'particle_diameter_m': 'particle_diameter',
    'diameter_particle': 'particle_diameter',
    'sphericity': 'particle_sphericity',
    'vdm': 'vapor_dimerization',
}

_COMPONENT_PROPERTY_KEYS = frozenset({
    'formula', 'Tc', 'Pc', 'Vc', 'Zc', 'omega', 'mc_c1', 'mc_c2',
    'mc_c3', 'kappa1', 'kappa2', 'kappa3',
    'twu_l', 'twu_m', 'twu_n', 'twu_c', 'Hf', 'Gf', 'S',
    'henry_Hcp', 'henry_B', 'henry_Tmin', 'henry_Tmax',
    'henry_Vinf', 'henry_Vinf_uncertainty',
    'Hf_liquid', 'Gf_liquid', 'S_liquid', 'Hf_solid', 'Gf_solid',
    'S_solid', 'Hcomb', 'Hcomb_gross', 'Tb', 'Tt', 'Pt', 'Tm',
    'Hvap', 'Hfus',
    'Cp_coeffs', 'Cp_liquid', 'Cp_solid', 'rho_solid', 'Vm_solid',
    'solid_material_form', 'solid_polymorph',
    'phase_behavior', 'particle_diameter', 'particle_sphericity',
    'antoine_A', 'antoine_B', 'antoine_C',
    'antoine_Tmin', 'antoine_Tmax', 'antoine_source', 'rho', 'rho_T',
    'uniquac_r', 'uniquac_q', 'phase_at_STP',
    'vapor_dimerization',
    'critical_properties_unavailable',
})

_COMPONENT_PROPERTY_NAMES = tuple(sorted(
    _COMPONENT_PROPERTY_KEYS | {
        'MW', 'molecular_weight', 'CAS', 'UNIFAC', 'SMILES',
        'T_rho', 'rho_T_K', 'VDM',
    },
    key=str.casefold,
))

_PROPERTY_CORRELATION_EQUATIONS = {
    'Psat': frozenset(_PSAT_EQUATION_COEFFICIENTS),
    'Hvap': frozenset({
        'poly_x', 'exp_poly_x', 'reduced_hvap_log', 'dippr_eq106', 'eq106',
    }),
    'Cpl': frozenset({'poly_x', 'exp_poly_x', 'shomate'}),
    'Cpg': frozenset({'poly_x', 'exp_poly_x', 'shomate'}),
    'Cps': frozenset({'poly_x', 'shomate', 'perry_151'}),
    'rhol': frozenset({'poly_x', 'exp_poly_x', 'density_reference'}),
    'rhos': frozenset({'poly_x', 'exp_poly_x', 'density_reference'}),
    'mug': frozenset({
        'poly_x', 'exp_poly_x', 'poly_tp', 'exp_poly_tp', 'dippr_eq101',
        'viscosity_exp_rhor',
    }),
    'mul': frozenset({
        'poly_x', 'exp_poly_x', 'poly_tp', 'exp_poly_tp', 'dippr_eq101',
        'viscosity_exp_rhor',
    }),
    'sigma': frozenset({
        'poly_x', 'exp_poly_x', 'dippr_eq106', 'eq106', 'constant',
        'constant_surface_tension', 'surface_tension_reference', 'jasper',
        'jasper_lange', 'somayajulu', 'somayajulu_revised', 'refprop_sigma',
        'refprop', 'refprop_surface_tension', 'vdi_ppds_11',
    }),
    'surface_tension': frozenset({
        'poly_x', 'exp_poly_x', 'dippr_eq106', 'eq106', 'constant',
        'constant_surface_tension', 'surface_tension_reference', 'jasper',
        'jasper_lange', 'somayajulu', 'somayajulu_revised', 'refprop_sigma',
        'refprop', 'refprop_surface_tension', 'vdi_ppds_11',
    }),
}

_PROPERTY_CORRELATION_FIELDS = frozenset({
    'equation', 'coefficients',
    'Tmin', 'Tmax', 'Tmin_K', 'Tmax_K',
    'Tc', 'Tc_K', 'Tb', 'Tb_K', 'Pc', 'Pc_bar', 'Pc_Pa',
    'Pmin_bar', 'Pmax_bar', 'P_ref_bar',
    'T_ref', 'T_ref_K', 'rho', 'rho_kg_m3', 'inverse_power',
    'quality', 'quality_note', 'source', 'selected_model', 'units_note',
    'sigma_N_per_m', 'value_N_per_m',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'x', 'y', 'X', 'Y',
})

_TOP_LEVEL_CORRELATION_COEFFICIENTS = frozenset({
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'x', 'y', 'X', 'Y',
})

_PROPERTY_CORRELATION_FIELD_ALIASES = {
    'tmin': 'Tmin_K',
    'tmax': 'Tmax_K',
    'tmin_k': 'Tmin_K',
    'tmax_k': 'Tmax_K',
    'tc': 'Tc_K',
    'tc_k': 'Tc_K',
    'tb': 'Tb_K',
    'tb_k': 'Tb_K',
    'pc': 'Pc_bar',
    'pc_bar': 'Pc_bar',
    'pc_pa': 'Pc_Pa',
    'pmin_bar': 'Pmin_bar',
    'pmax_bar': 'Pmax_bar',
    't_ref': 'T_ref_K',
    't_ref_k': 'T_ref_K',
    'rho': 'rho_kg_m3',
    'rho_kg_m3': 'rho_kg_m3',
    'inverse_power': 'inverse_power',
}

_PROPERTY_CORRELATION_FIELD_CASE_LOOKUP = {
    name.casefold(): name
    for name in _PROPERTY_CORRELATION_FIELDS - _TOP_LEVEL_CORRELATION_COEFFICIENTS
}

_PROPERTY_CORRELATION_COEFFICIENTS = {
    'poly_x': frozenset('ABCDEF'),
    'exp_poly_x': frozenset('ABCDEF'),
    'poly_tp': frozenset('ABCDEF'),
    'exp_poly_tp': frozenset('ABCDEF'),
    'shomate': frozenset('ABCDE'),
    'perry_151': frozenset('ABCD'),
    'reduced_hvap_log': frozenset('ABCD'),
    'reduced_vapor_pressure': frozenset('ABCD'),
    'psat_mercury': frozenset('ABCDEF'),
    'canonical_psat': frozenset('ABCDEF'),
    'canonical_psat_af': frozenset('ABCDEF'),
    'canonical_psat_ag': frozenset('ABCDEFG'),
    'canonical_psat_ah': frozenset('ABCDEFGH'),
    'dippr_eq101': frozenset('ABCDE'),
    'dippr_eq106': frozenset('ABCDE'),
    'eq106': frozenset('ABCDE'),
    'vdi_ppds_11': frozenset('ABCDE'),
    'viscosity_exp_rhor': frozenset({'A', 'B', 'C', 'D', 'E', 'F', 'x', 'y', 'X', 'Y'}),
    'density_reference': frozenset(),
    'constant': frozenset({'sigma', 'value'}),
    'constant_surface_tension': frozenset({'sigma', 'value'}),
    'surface_tension_reference': frozenset({'sigma', 'value'}),
    'jasper': frozenset({'a', 'b'}),
    'jasper_lange': frozenset({'a', 'b'}),
    'somayajulu': frozenset({'A', 'B', 'C', 'Tc'}),
    'somayajulu_revised': frozenset({'A', 'B', 'C', 'Tc'}),
    'refprop_sigma': frozenset({'Tc', 'sigma0', 'sigma1', 'sigma2', 'n0', 'n1', 'n2'}),
    'refprop': frozenset({'Tc', 'sigma0', 'sigma1', 'sigma2', 'n0', 'n1', 'n2'}),
    'refprop_surface_tension': frozenset({'Tc', 'sigma0', 'sigma1', 'sigma2', 'n0', 'n1', 'n2'}),
}

_INTERACTION_MODEL_ALIASES = {
    'PENG-ROBINSON': 'PR',
    'PENG ROBINSON': 'PR',
    'PR-BM': 'PR',
    'PENG-ROBINSON-BM': 'PR',
    'SRK': 'SRK',
    'RKS': 'SRK',
    'RKS-BM': 'SRK',
    'SRK-BM': 'SRK',
    'SOAVE-REDLICH-KWONG': 'SRK',
    'RK-SOAVE': 'SRK',
    'NRTL-RK': 'NRTL',
    'NRTL-PR': 'NRTL',
    'NRTL-VDM': 'NRTL',
    'UNIQUAC-RK': 'UNIQUAC',
    'UNIQUAC-PR': 'UNIQUAC',
    'UNIQUAC-VDM': 'UNIQUAC',
    'VISCOSITY': 'LIQUID_VISCOSITY',
    'LIQUID-VISCOSITY': 'LIQUID_VISCOSITY',
    'LIQUID-VISCOSITY-MIXING': 'LIQUID_VISCOSITY',
    'LIQUID-MIXTURE-VISCOSITY': 'LIQUID_VISCOSITY',
    'MIXTURE-VISCOSITY': 'LIQUID_VISCOSITY',
    'VAPOR-DIMERIZATION': 'VDM',
    'VAPOR-DIMERISATION': 'VDM',
}

_INTERACTION_PARAMETER_FIELDS = {
    'PR': frozenset({
        'comment', 'kij', 'k_ij', 'kij_a', 'kij_b', 'kij_c',
        'tmin', 'tmax', 'tmin_k', 'tmax_k',
        't_ref', 't_ref_k', 'tref', 'tref_k',
    }),
    'SRK': frozenset({
        'comment', 'kij', 'k_ij', 'kij_a', 'kij_b', 'kij_c',
        'tmin', 'tmax', 'tmin_k', 'tmax_k',
        't_ref', 't_ref_k', 'tref', 'tref_k',
    }),
    'NRTL': frozenset({
        'comment', 'alpha', 'alpha12', 'a12', 'a21',
        'a12_cal_per_mol', 'a21_cal_per_mol',
        'tau12_c', 'tau12_d', 'tau12_e', 'tau12_f', 'tau12_g',
        'tau21_c', 'tau21_d', 'tau21_e', 'tau21_f', 'tau21_g',
        'tau_tref', 'tref',
    }),
    'UNIQUAC': frozenset({
        'comment', 'a12', 'a21', 'a12_cal_per_mol', 'a21_cal_per_mol',
        'tau12_a', 'tau12_b', 'tau12_c', 'tau12_d', 'tau12_e',
        'tau21_a', 'tau21_b', 'tau21_c', 'tau21_d', 'tau21_e',
        'tau_tref', 'tref',
        'use_q_prime', 'model_variant',
    }),
    'LIQUID_VISCOSITY': frozenset({
        'comment', 'form', 'viscosity_form',
        'g', 'g12', 'excess_g_over_rt',
        'a0', 'a1', 'a2', 'a0_k', 'a1_k', 'a2_k',
        'a', 'b', 'c', 'd', 'e', 'f',
        'tmin', 'tmax', 'tmin_k', 'tmax_k',
        't_ref', 't_ref_k', 'tref', 'tref_k',
    }),
    'VDM': frozenset({
        'comment',
        'delta_h_residual', 'delta_h_residual_j_per_mol',
        'delta_s_residual', 'delta_s_residual_j_per_mol_k',
    }),
}

_INTERACTION_ESTIMATION_FIELDS = frozenset({
    'source', 'policy', 'parameter_order', 'alpha', 'alpha12',
    'tmin', 'tmax', 'tmin_k', 'tmax_k',
    't_ref', 't_ref_k', 'tref', 'tref_k',
    'comment',
})

_TOP_LEVEL_DIRECTIVES = (
    'PROCESS', 'VERSION', 'PFD_VERSION', 'AUTHOR', 'DATE', 'DESCRIPTION',
    'THERMO_METHOD', 'THERMO', 'PROPERTY_METHOD', 'FLUID_PHASE_MODEL',
    'FLUID_PHASES', 'ONLINE_LOOKUP', 'ALLOW_ONLINE_LOOKUP', 'FETCH_ONLINE',
    'PSAT_MINIMUM_PRESSURE', 'MINIMUM_PRESSURE', 'MINIMUM_PSAT_PRESSURE',
    'MINIMUM_VAPOR_PRESSURE', 'RECYCLE_METHOD', 'RECYCLE_SOLVER',
    'ACTIVITY_INTERACTION_MAX_PSAT', 'ACTIVITY_INTERACTION_MAX_TEMPERATURE',
    'TEAR_STREAMS', 'RECYCLE_TEAR_STREAMS', 'TEAR_STREAM',
    'RECYCLE_TRACE_TOLERANCE', 'TRACE_TOLERANCE', 'COMPONENTS',
    'THERMO_SCOPES', 'PROPERTY_CORRELATIONS', 'INTERACTION_ESTIMATION',
    'INTERACTION_PARAMETERS', 'REACTIONS',
    'STREAM', 'UNIT',
)

_STREAM_PROPERTY_NAMES = frozenset({
    'T', 'P', 'F', 'FLOW', 'MOLAR_FLOW',
    'F_MASS', 'MASS_FLOW', 'FLOW_MASS',
    'VF', 'VAP_FRAC', 'VAPOR_FRAC', 'VAPOR_FRACTION',
})

_SOLID_EQUILIBRIUM_MODEL_NAMES = frozenset({
    'SLE', 'SLLE', 'SVLE', 'SVLLE', 'VLS', 'VLSE',
    'SOLID-LIQUID', 'SOLID-LIQUID-EQUILIBRIUM',
    'SOLID-LIQUID-VAPOR', 'SOLID-VAPOR-LIQUID',
})

_SOLID_EQUILIBRIUM_HINTS = (
    'SOLID', 'CRYSTAL', 'PRECIP', 'DISSOL', 'SOLUB', 'FREEZ', 'MELT',
    'SUBLIM', 'FUSION',
)

# Recognized names from other simulators and common thermodynamic literature.
# These are diagnostics only: they must never silently change the requested
# physical model.
_KNOWN_UNSUPPORTED_THERMO_METHODS = {
    'PRWS': (
        "Try PSRK for a predictive GE-EOS model, or NRTL-PR/UNIQUAC-PR "
        "when a gamma-phi treatment is suitable; neither is an exact "
        "Wong-Sandler replacement.",
        "Peng-Robinson with Wong-Sandler mixing rules will be implemented "
        "in the future.",
    ),
    'PR-WS': (
        "Try PSRK for a predictive GE-EOS model, or NRTL-PR/UNIQUAC-PR "
        "when a gamma-phi treatment is suitable; neither is an exact "
        "Wong-Sandler replacement.",
        "Peng-Robinson with Wong-Sandler mixing rules will be implemented "
        "in the future.",
    ),
    'PR-WONG-SANDLER': (
        "Try PSRK for a predictive GE-EOS model, or NRTL-PR/UNIQUAC-PR "
        "when a gamma-phi treatment is suitable; neither is an exact "
        "Wong-Sandler replacement.",
        "Peng-Robinson with Wong-Sandler mixing rules will be implemented "
        "in the future.",
    ),
    'RKSWS': (
        "Try PSRK for the currently available SRK-based predictive GE-EOS "
        "model; it does not use Wong-Sandler mixing rules.",
        "SRK with Wong-Sandler mixing rules will be implemented in the future.",
    ),
    'RKS-WS': (
        "Try PSRK for the currently available SRK-based predictive GE-EOS "
        "model; it does not use Wong-Sandler mixing rules.",
        "SRK with Wong-Sandler mixing rules will be implemented in the future.",
    ),
    'RKS-WONG-SANDLER': (
        "Try PSRK for the currently available SRK-based predictive GE-EOS "
        "model; it does not use Wong-Sandler mixing rules.",
        "SRK with Wong-Sandler mixing rules will be implemented in the future.",
    ),
    'SRKWS': (
        "Try PSRK for the currently available SRK-based predictive GE-EOS "
        "model; it does not use Wong-Sandler mixing rules.",
        "SRK with Wong-Sandler mixing rules will be implemented in the future.",
    ),
    'SRK-WS': (
        "Try PSRK for the currently available SRK-based predictive GE-EOS "
        "model; it does not use Wong-Sandler mixing rules.",
        "SRK with Wong-Sandler mixing rules will be implemented in the future.",
    ),
    'WILSON': (
        "Try NRTL or UNIQUAC for a fitted activity-coefficient model.",
        "Wilson is a future implementation target.",
    ),
    'ELECNRTL': (
        "There is no electrolyte-capable substitute yet; plain NRTL only "
        "models nonelectrolyte liquid nonideality.",
        "Electrolyte NRTL is a future implementation target.",
    ),
    'ENRTL-RK': (
        "There is no electrolyte-capable substitute yet; NRTL-RK only "
        "models nonelectrolyte liquid nonideality.",
        "Electrolyte NRTL is a future implementation target.",
    ),
    'PITZER': (
        "There is no electrolyte-capable substitute yet.",
        "Electrolyte thermodynamics is a future implementation target.",
    ),
    'NRTL-HOC': (
        "Try NRTL-VDM when carboxylic-acid vapor dimerization is the important "
        "association correction.",
        "Hayden-O'Connell vapor association is a future implementation target.",
    ),
    'UNIQUAC-HOC': (
        "Try UNIQUAC-VDM when vapor dimerization is the important correction.",
        "General Hayden-O'Connell vapor association is a future implementation target.",
    ),
    'UNIF-HOC': (
        "Try UNIFAC-VDM when vapor dimerization is the important correction.",
        "General Hayden-O'Connell vapor association is a future implementation target.",
    ),
    'UNIF-LBY': (
        "Try UNIFAC, UNIFDMD, or UNIFNIST depending on the parameter set needed.",
        "",
    ),
    'CHAO-SEA': (
        "Try SRK or PR for hydrocarbon service.",
        "",
    ),
    'CHAO-SEADER': (
        "Try SRK or PR for hydrocarbon service.",
        "",
    ),
    'GRAYSON': (
        "Try SRK or PR for hydrocarbon service.",
        "",
    ),
    'GRAYSON-STREED': (
        "Try SRK or PR for hydrocarbon service.",
        "",
    ),
    'PC-SAFT': (
        "Try PR or SRK when a cubic equation of state is adequate.",
        "PC-SAFT will be implemented in the future.",
    ),
    'CPA': (
        "Try PR or SRK when association is not essential; *-VDM methods "
        "cover vapor dimerization only and are not a CPA replacement.",
        "Cubic-Plus-Association thermodynamics will be implemented in the future.",
    ),
    'LKP': (
        "Try PR or SRK for a cubic-EOS corresponding-states approximation.",
        "Lee-Kesler-Plocker thermodynamics will be implemented in the future.",
    ),
    'LEE-KESLER-PLOCKER': (
        "Try PR or SRK for a cubic-EOS corresponding-states approximation.",
        "Lee-Kesler-Plocker thermodynamics will be implemented in the future.",
    ),
    'IAPWS-95': (
        "Try STEAM, which currently uses CoolProp's IF97 water/steam properties.",
        "IAPWS-95 water/steam properties will be implemented in the future.",
    ),
    'IAWPS-95': (
        "Try STEAM, which currently uses CoolProp's IF97 water/steam properties. "
        "The standard formulation name is IAPWS-95.",
        "IAPWS-95 water/steam properties will be implemented in the future.",
    ),
    'AMINES': (
        "There is no amine/acid-gas-capable substitute yet; ordinary NRTL "
        "variants do not model the required reactive electrolyte chemistry.",
        "Amine and acidic-gas thermodynamics will be implemented in the future.",
    ),
    'COSMO-SAC': (
        "Try UNIFAC, NRTL, or UNIQUAC for liquid nonideality.",
        "COSMO-family thermodynamics is a future implementation target.",
    ),
    'GERG-2008': (
        "Try PR or SRK for a cubic-EOS approximation.",
        "",
    ),
    'STEAMNBS': (
        "Try STEAM, which uses CoolProp's IF97 water/steam properties.",
        "",
    ),
}

_SUPPORTED_THERMO_SCOPE_METHODS = frozenset({
    'IDEAL', 'STEAM',
    'RK', 'SRK', 'PR', 'PSRK', 'RKS-BM', 'PR-BM', 'SRK-MC', 'PR-MC',
    'SRK-TWU', 'PR-TWU', 'PRSV1', 'PRSV2',
    'UNIFAC', 'UNIFAC2', 'UNIFDMD', 'UNIFM2', 'UNIFNIST',
    'UNIFAC-VDM', 'UNIFDMD-VDM', 'UNIFNIST-VDM',
    'UNIFAC-RK', 'UNIFAC-PR', 'UNIFDMD-RK', 'UNIFDMD-PR',
    'UNIFNIST-RK', 'UNIFNIST-PR',
    'NRTL', 'NRTL-VDM', 'NRTL-RK', 'NRTL-PR',
    'UNIQUAC', 'UNIQUAC-VDM', 'UNIQUAC-RK', 'UNIQUAC-PR',
})

_VDM_COMPONENT_PARAMETER_ALIASES = {
    'delta_h': 'delta_H_J_per_mol',
    'deltah': 'delta_H_J_per_mol',
    'delta_h_j_per_mol': 'delta_H_J_per_mol',
    'delta_s': 'delta_S_J_per_mol_K',
    'deltas': 'delta_S_J_per_mol_K',
    'delta_s_j_per_mol_k': 'delta_S_J_per_mol_K',
}

_VDM_CROSS_PARAMETER_ALIASES = {
    'delta_h_residual': 'delta_H_residual_J_per_mol',
    'delta_h_residual_j_per_mol': 'delta_H_residual_J_per_mol',
    'delta_s_residual': 'delta_S_residual_J_per_mol_K',
    'delta_s_residual_j_per_mol_k': 'delta_S_residual_J_per_mol_K',
}


def normalize_interaction_model(model: str) -> str:
    """Return the runtime model family used to validate an interaction row."""
    text = str(model).strip().upper().replace('_', '-')
    return _INTERACTION_MODEL_ALIASES.get(text, text)


def canonical_correlation_field_name(name: object) -> str:
    """Normalize a non-coefficient correlation field exactly as the parser does."""
    text = str(name)
    folded = text.casefold()
    return (
        _PROPERTY_CORRELATION_FIELD_ALIASES.get(folded)
        or _PROPERTY_CORRELATION_FIELD_CASE_LOOKUP.get(folded)
        or text
    )


def closest_name(name: object, candidates) -> Optional[str]:
    """Return one conservative, case-insensitive close match when available."""
    text = str(name)
    by_folded_name = {
        str(candidate).casefold(): str(candidate)
        for candidate in candidates
    }
    matches = get_close_matches(
        text.casefold(),
        tuple(by_folded_name),
        n=1,
        cutoff=0.65,
    )
    return by_folded_name[matches[0]] if matches else None


def unknown_name_message(kind: str, name: object, candidates) -> str:
    """Format a consistent unknown-name error with a conservative suggestion."""
    text = str(name)
    message = f"Unknown {kind} '{text}'."
    suggestion = closest_name(text, candidates)
    if suggestion and suggestion.casefold() != text.casefold():
        message += f" Did you mean '{suggestion}'?"
    return message


def unsupported_thermo_method_message(
    method: str,
    supported,
    aliases: dict[str, str],
) -> str:
    """Explain a bad method name without silently selecting another model."""
    available_names = tuple(sorted(set(supported) | set(aliases)))
    unavailable_names = tuple(_KNOWN_UNSUPPORTED_THERMO_METHODS)
    suggestion = closest_name(
        method,
        available_names + unavailable_names,
    )
    message = f"Unsupported thermodynamics method '{method}'."
    if suggestion in available_names:
        canonical = aliases.get(suggestion, suggestion)
        if canonical == suggestion:
            return f"{message} Did you mean '{suggestion}'?"
        return (
            f"{message} Did you mean '{suggestion}' "
            f"(available as {canonical})?"
        )
    if suggestion in _KNOWN_UNSUPPORTED_THERMO_METHODS:
        replacement, future = _KNOWN_UNSUPPORTED_THERMO_METHODS[suggestion]
        if suggestion != method:
            message += f" This looks like '{suggestion}'."
        message += f" {replacement}"
        if future:
            message += f" {future}"
        return message
    return message + " See the format documentation for available methods."


def unsupported_fluid_phase_model_message(value: object) -> str:
    """Explain fluid-phase typos separately from unsupported solid equilibria."""
    text = str(value).strip()
    normalized = re.sub(r'[\s_]+', '-', text.upper())
    phase_letters = re.sub(r'[^A-Z]', '', normalized)
    resembles_phase_acronym = (
        bool(phase_letters)
        and set(phase_letters) <= set('SVLE')
        and 'S' in phase_letters
    )
    resembles_solid_equilibrium = (
        normalized in _SOLID_EQUILIBRIUM_MODEL_NAMES
        or resembles_phase_acronym
        or any(hint in normalized for hint in _SOLID_EQUILIBRIUM_HINTS)
    )
    if resembles_solid_equilibrium:
        return (
            f"Solid-equilibrium phase model '{text}' is not supported. "
            "For a nonparticipating process solid, declare "
            "phase_behavior=permanent_solid on that component and optionally "
            "choose solid_material_form=crystalline, glass, hydrate, or solvate. "
            "Dissolving or precipitating components will use "
            "phase_behavior=soluble_solid when soluble-solid/SLE support is "
            "implemented in the future."
        )
    return unknown_name_message(
        'fluid phase model',
        text,
        ('VLE', 'VL(L)E', 'VLLE'),
    )


def _finite_vdm_number(value: object, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"VDM parameter '{field_name}' must be numeric.") from error
    if not math.isfinite(number):
        raise ValueError(f"VDM parameter '{field_name}' must be finite.")
    return number


def normalize_vdm_component_parameters(value: object) -> dict:
    """Validate and canonicalize one component's atomic VDM H/S bundle."""
    if not isinstance(value, dict):
        raise ValueError("VDM must be a {key:value} parameter bundle.")
    normalized = {}
    for raw_name, raw_value in value.items():
        lookup_name = str(raw_name).strip().lower().replace('-', '_')
        canonical_name = _VDM_COMPONENT_PARAMETER_ALIASES.get(lookup_name)
        if canonical_name is None:
            raise ValueError(
                unknown_name_message(
                    'VDM parameter',
                    raw_name,
                    ('delta_H', 'delta_S', 'delta_H_J_per_mol', 'delta_S_J_per_mol_K'),
                )
            )
        if canonical_name in normalized:
            raise ValueError(
                f"VDM parameter '{raw_name}' duplicates another alias for "
                f"'{canonical_name}'."
            )
        normalized[canonical_name] = _finite_vdm_number(raw_value, str(raw_name))
    missing = [
        name
        for name in ('delta_H_J_per_mol', 'delta_S_J_per_mol_K')
        if name not in normalized
    ]
    if missing:
        raise ValueError(
            "VDM requires the atomic delta_H and delta_S bundle; missing "
            + ', '.join(missing)
            + "."
        )
    return normalized


def normalize_vdm_cross_parameters(parameters: dict) -> dict:
    """Validate and canonicalize one cross-dimer residual H/S bundle."""
    normalized = {}
    for raw_name, raw_value in parameters.items():
        lookup_name = str(raw_name).strip().lower().replace('-', '_')
        if lookup_name == 'comment':
            continue
        canonical_name = _VDM_CROSS_PARAMETER_ALIASES.get(lookup_name)
        if canonical_name is None:
            raise ValueError(
                unknown_name_message(
                    'VDM cross parameter',
                    raw_name,
                    (
                        'delta_H_residual',
                        'delta_S_residual',
                        'delta_H_residual_J_per_mol',
                        'delta_S_residual_J_per_mol_K',
                    ),
                )
            )
        if canonical_name in normalized:
            raise ValueError(
                f"VDM cross parameter '{raw_name}' duplicates another alias "
                f"for '{canonical_name}'."
            )
        normalized[canonical_name] = _finite_vdm_number(raw_value, str(raw_name))
    missing = [
        name
        for name in (
            'delta_H_residual_J_per_mol',
            'delta_S_residual_J_per_mol_K',
        )
        if name not in normalized
    ]
    if missing:
        raise ValueError(
            "VDM cross overrides require the atomic delta_H_residual and "
            "delta_S_residual bundle; missing "
            + ', '.join(missing)
            + "."
        )
    return normalized


class PortType(Enum):
    INLET = "inlet"
    OUTLET = "outlet"
    VAPOR_OUTLET = "vapor_outlet"
    LIQUID_OUTLET = "liquid_outlet"
    LIGHT_LIQUID_OUTLET = "light_liquid_outlet"
    HEAVY_LIQUID_OUTLET = "heavy_liquid_outlet"
    LIQUID1_OUTLET = "liquid1_outlet"
    LIQUID2_OUTLET = "liquid2_outlet"
    SOLID_OUTLET = "solid_outlet"
    GAS_INLET = "gas_inlet"
    SOLVENT_INLET = "solvent_inlet"
    EXTRACT_OUTLET = "extract_outlet"
    RAFFINATE_OUTLET = "raffinate_outlet"


_INLET_PORT_TYPES = frozenset({
    PortType.INLET,
    PortType.GAS_INLET,
    PortType.SOLVENT_INLET,
})


class UnitType(Enum):
    # Mixing/Splitting
    MIXER = "Mixer"
    SPLITTER = "Splitter"
    
    # Pressure Change
    PUMP = "Pump"
    COMPRESSOR = "Compressor"
    EXPANDER = "Expander"
    VALVE = "Valve"
    PIPE = "Pipe"
    
    # Heat Transfer
    HEATER = "Heater"
    COOLER = "Cooler"
    HEAT_EXCHANGER = "HeatExchanger"
    
    # Vapor-Liquid Separation
    FLASH = "Flash"
    FLASH3 = "Flash3"
    FLASH_LLE = "FlashLLE"
    THREE_PHASE_FLASH = "ThreePhaseFlash"
    DECANTER = "Decanter"
    
    # Multi-stage Separation
    DISTILLATION = "ShortcutDistillation"
    DISTILLATION_LLE = "DistillationLLE"
    REACTIVE_DISTILLATION = "ReactiveDistillation"
    ABSORBER = "Absorber"
    RIGOROUS_ABSORBER = "RigorousAbsorber"
    STRIPPER = "Stripper"
    RIGOROUS_STRIPPER = "RigorousStripper"
    SHORTCUT_EXTRACTOR = "ShortcutExtractor"
    EXTRACTOR = "Extractor"
    
    # Reactors
    REACTOR = "Reactor"
    EQUILIBRIUM_REACTOR = "EquilibriumReactor"
    GIBBS_REACTOR = "GibbsReactor"
    PFR = "PFR"
    PACKED_BED_REACTOR = "PackedBedReactor"
    CSTR = "CSTR"
    BATCH_REACTOR = "BatchReactor"
    
    # Solids Handling
    CRYSTALLIZER = "Crystallizer"
    FILTER = "Filter"
    DRYER = "Dryer"


@dataclass
class Component:
    """Chemical component definition"""
    symbol: str
    # Kept as ``name`` in the Python/JSON model for backward compatibility;
    # in PFD text this field is the sole external lookup identifier.
    name: str
    formula: Optional[str] = None
    CAS: Optional[str] = None
    molecular_weight: Optional[float] = None
    # Optional thermodynamic properties (can be specified in PFD or looked up)
    Tc: Optional[float] = None  # Critical temperature [K]
    Pc: Optional[float] = None  # Critical pressure [bar]
    Vc: Optional[float] = None  # Critical volume [cm3/mol]
    Zc: Optional[float] = None  # Critical compressibility factor
    omega: Optional[float] = None  # Acentric factor
    mc_c1: Optional[float] = None  # Mathias-Copeman alpha parameter c1
    mc_c2: Optional[float] = None  # Mathias-Copeman alpha parameter c2
    mc_c3: Optional[float] = None  # Mathias-Copeman alpha parameter c3
    kappa1: Optional[float] = None  # PRSV alpha parameter kappa1
    kappa2: Optional[float] = None  # PRSV2 alpha parameter kappa2
    kappa3: Optional[float] = None  # PRSV2 alpha parameter kappa3
    twu_l: Optional[float] = None  # Twu alpha parameter L
    twu_m: Optional[float] = None  # Twu alpha parameter M
    twu_n: Optional[float] = None  # Twu alpha parameter N
    twu_c: Optional[float] = None  # Twu volume-translation parameter c
    henry_Hcp: Optional[float] = None  # Hcp at 298.15 K [mol/(m3*Pa)]
    henry_B: Optional[float] = None  # d(ln Hcp)/d(1/T) [K]
    henry_Tmin: Optional[float] = None  # Full-quality correlation lower bound [K]
    henry_Tmax: Optional[float] = None  # Full-quality correlation upper bound [K]
    henry_Vinf: Optional[float] = None  # Aqueous partial molar volume at infinite dilution [cm3/mol]
    henry_Vinf_uncertainty: Optional[float] = None  # Standard uncertainty [cm3/mol]
    Hf: Optional[float] = None  # Heat of formation [kJ/mol]
    Gf: Optional[float] = None  # Gibbs energy of formation [kJ/mol]
    S: Optional[float] = None  # Standard molar entropy [J/mol-K]
    Hf_liquid: Optional[float] = None  # Liquid heat of formation [kJ/mol]
    Gf_liquid: Optional[float] = None  # Liquid Gibbs energy of formation [kJ/mol]
    S_liquid: Optional[float] = None  # Liquid standard molar entropy [J/mol-K]
    Hf_solid: Optional[float] = None  # Solid heat of formation [kJ/mol]
    Gf_solid: Optional[float] = None  # Solid Gibbs energy of formation [kJ/mol]
    S_solid: Optional[float] = None  # Solid standard molar entropy [J/mol-K]
    Hcomb: Optional[float] = None  # Net heat of combustion [kJ/mol]
    Hcomb_gross: Optional[float] = None  # Gross heat of combustion [kJ/mol]
    Tb: Optional[float] = None  # Boiling point [K]
    Tm: Optional[float] = None  # Melting point [K]
    Hvap: Optional[float] = None  # Heat of vaporization [kJ/mol]
    Hfus: Optional[float] = None  # Heat of fusion [kJ/mol]
    Cp_coeffs: Optional[list] = None  # Heat capacity polynomial coefficients
    Cp_liquid: Optional[float] = None  # Liquid heat capacity [J/mol-K]
    Cp_solid: Optional[float] = None  # Solid heat capacity [J/mol-K]
    rho_solid: Optional[float] = None  # Solid mass density [kg/m3]
    Vm_solid: Optional[float] = None  # Solid molar volume [m3/kmol]
    solid_material_form: Optional[str] = None
    solid_polymorph: Optional[str] = None
    phase_behavior: Optional[str] = None
    particle_diameter: Optional[float] = None  # Representative diameter [m]
    particle_sphericity: Optional[float] = None  # Dimensionless, (0, 1]
    # Antoine equation: log10(P_bar) = A - B/(C + T_C)
    antoine_A: Optional[float] = None
    antoine_B: Optional[float] = None
    antoine_C: Optional[float] = None
    antoine_Tmin: Optional[float] = None  # K
    antoine_Tmax: Optional[float] = None  # K
    antoine_source: Optional[str] = None
    # Single liquid-density reference point, rho in kg/m3 at rho_T.
    rho: Optional[float] = None
    rho_T: Optional[float] = None
    # UNIQUAC pure-component combinatorial parameters.
    uniquac_r: Optional[float] = None
    uniquac_q: Optional[float] = None
    # UNIFAC groups - dict of group name to count, e.g., {'CH3': 2, 'OH': 1}
    unifac_groups: Optional[dict] = None
    # SMILES string for structure (can be used to auto-detect UNIFAC groups)
    smiles: Optional[str] = None
    phase_at_STP: Optional[str] = None
    critical_properties_unavailable: Optional[bool] = None
    vapor_dimerization: Optional[dict] = None
    property_correlations: dict = field(default_factory=dict)
    Tt: Optional[float] = None  # Triple-point temperature [K]
    Pt: Optional[float] = None  # Triple-point pressure [bar]

    @property
    def identifier(self) -> str:
        """External property lookup identifier from the second PFD field."""
        return self.name
    
    def to_pfd(self) -> str:
        parts = []
        scalar_fields = [
            ('formula', 'formula'),
            ('CAS', 'CAS'),
            ('molecular_weight', 'MW'),
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
            ('Cp_liquid', 'Cp_liquid'),
            ('Cp_solid', 'Cp_solid'),
            ('rho_solid', 'rho_solid'),
            ('Vm_solid', 'Vm_solid'),
            ('solid_material_form', 'solid_material_form'),
            ('solid_polymorph', 'solid_polymorph'),
            ('phase_behavior', 'phase_behavior'),
            ('particle_diameter', 'particle_diameter'),
            ('particle_sphericity', 'particle_sphericity'),
            ('antoine_A', 'antoine_A'),
            ('antoine_B', 'antoine_B'),
            ('antoine_C', 'antoine_C'),
            ('antoine_Tmin', 'antoine_Tmin'),
            ('antoine_Tmax', 'antoine_Tmax'),
            ('antoine_source', 'antoine_source'),
            ('rho', 'rho'),
            ('rho_T', 'rho_T'),
            ('uniquac_r', 'uniquac_r'),
            ('uniquac_q', 'uniquac_q'),
            ('phase_at_STP', 'phase_at_STP'),
            ('critical_properties_unavailable', 'critical_properties_unavailable'),
        ]
        for attr, key in scalar_fields:
            value = getattr(self, attr)
            if value is not None:
                parts.append(f"{key}={self._format_pfd_value(value)}")
        if self.Cp_coeffs is not None:
            parts.append(f"Cp_coeffs={self._format_pfd_value(self.Cp_coeffs)}")
        if self.unifac_groups is not None:
            groups_str = '+'.join(f"{v}{k}" for k, v in self.unifac_groups.items())
            parts.append(f"UNIFAC={groups_str}")
        if self.vapor_dimerization is not None:
            vdm = {
                'delta_H': self.vapor_dimerization['delta_H_J_per_mol'],
                'delta_S': self.vapor_dimerization['delta_S_J_per_mol_K'],
            }
            parts.append(f"VDM={self._format_pfd_value(vdm)}")
        if self.smiles is not None:
            parts.append(f"SMILES={self._format_pfd_value(self.smiles)}")
        if parts:
            return f"    {self.symbol} | {self.identifier} | {', '.join(parts)}"
        return f"    {self.symbol} | {self.identifier}"

    def to_property_correlations_pfd(self) -> list[str]:
        lines = []
        for key, correlation in self.property_correlations.items():
            entries = []
            for field_name, value in correlation.items():
                if field_name == 'coefficients' and isinstance(value, dict):
                    for coeff_name, coeff_value in value.items():
                        entries.append(f"{coeff_name}={self._format_pfd_value(coeff_value)}")
                else:
                    entries.append(f"{field_name}={self._format_pfd_value(value)}")
            lines.append(f"    {self.symbol}.{key} | {', '.join(entries)}")
        return lines

    def resolver_property_correlations(self) -> dict:
        correlations = {
            key: dict(value)
            for key, value in (self.property_correlations or {}).items()
        }
        if self.rho is not None and 'rhol' not in correlations:
            correlations['rhol'] = {
                'equation': 'density_reference',
                'rho_kg_m3': self.rho,
                'T_ref_K': self.rho_T if self.rho_T is not None else 298.15,
                'source': 'PFD component rho',
                'quality': 1.0,
                'units_note': 'rho in kg/m^3 at T_ref_K',
            }
        return correlations

    @staticmethod
    def _format_pfd_value(value) -> str:
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if value.is_integer() and abs(value) < 1.0e16:
                return f"{value:.0f}"
            return repr(value)
        if isinstance(value, (list, tuple)):
            return '[' + ', '.join(Component._format_pfd_value(item) for item in value) + ']'
        if isinstance(value, dict):
            return '{' + ', '.join(
                f"{key}:{Component._format_pfd_value(item)}"
                for key, item in value.items()
            ) + '}'
        text = str(value)
        if any(char in text for char in ',|[]{}') or text != text.strip() or ' ' in text:
            return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'
        return text


@dataclass
class PortReference:
    """Reference to a port: either FEED, PRODUCT, or UNIT.port"""
    unit_id: Optional[str]  # None for FEED/PRODUCT
    port_id: Optional[str]  # None for FEED/PRODUCT
    is_feed: bool = False
    is_product: bool = False
    
    @classmethod
    def from_string(cls, s: str) -> 'PortReference':
        s = s.strip()
        if s == "FEED":
            return cls(unit_id=None, port_id=None, is_feed=True)
        elif s == "PRODUCT":
            return cls(unit_id=None, port_id=None, is_product=True)
        else:
            if '.' not in s:
                raise ParseError(f"Invalid port reference: '{s}'. Expected 'UNIT.port' format.")
            unit_id, port_id = s.split('.', 1)
            return cls(unit_id=unit_id.strip(), port_id=port_id.strip())
    
    def to_string(self) -> str:
        if self.is_feed:
            return "FEED"
        elif self.is_product:
            return "PRODUCT"
        else:
            return f"{self.unit_id}.{self.port_id}"


@dataclass
class StreamProperty:
    """A property specification for a stream"""
    name: str
    value: str
    unit: Optional[str] = None
    
    def to_pfd(self) -> str:
        if self.unit:
            return f"    {self.name} = {self.value} [{self.unit}]"
        else:
            return f"    {self.name} = {self.value}"


@dataclass
class Composition:
    """Stream composition as component fractions"""
    fractions: dict[str, float] = field(default_factory=dict)
    basis: str = "mole"
    
    def to_pfd(self) -> str:
        parts = [f"{comp}:{frac}" for comp, frac in self.fractions.items()]
        key = "w" if self.basis == "mass" else "x"
        return f"    {key} = {', '.join(parts)}"


@dataclass
class Stream:
    """A process stream connecting two ports"""
    id: str
    source: PortReference
    destination: PortReference
    properties: list[StreamProperty] = field(default_factory=list)
    composition: Optional[Composition] = None
    
    def to_pfd(self) -> str:
        lines = [f"STREAM {self.id} : {self.source.to_string()} -> {self.destination.to_string()}"]
        for prop in self.properties:
            lines.append(prop.to_pfd())
        if self.composition:
            lines.append(self.composition.to_pfd())
        return '\n'.join(lines)


@dataclass
class Port:
    """A port on a unit operation"""
    id: str
    port_type: PortType
    
    def to_pfd(self) -> str:
        return f"        {self.id} : {self.port_type.value}"


@dataclass
class Parameter:
    """A unit operation parameter"""
    name: str
    value: str
    unit: Optional[str] = None
    
    def to_pfd(self) -> str:
        if self.unit:
            return f"        {self.name} = {self.value} [{self.unit}]"
        else:
            return f"        {self.name} = {self.value}"


@dataclass
class Reaction:
    """A chemical reaction specification"""
    equation: str
    parameters: dict[str, str] = field(default_factory=dict)
    reference: Optional[str] = None
    
    def to_pfd(self) -> str:
        if self.reference:
            return f"        @{self.reference}"
        param_str = ', '.join(f"{k}={v}" for k, v in self.parameters.items())
        if param_str:
            return f"        {self.equation} | {param_str}"
        return f"        {self.equation}"


@dataclass
class NamedReaction:
    """A reusable top-level reaction catalog entry."""

    name: str
    reaction: Reaction

    def to_pfd(self) -> str:
        rendered = self.reaction.to_pfd().strip()
        return f"    {self.name} : {rendered}"


@dataclass
class ThermoScope:
    """Named thermodynamic-method context assigned to unit operations."""

    name: str
    method: str
    inherit: Optional[str] = None

    def to_pfd(self) -> str:
        entries = [f"method={Component._format_pfd_value(self.method)}"]
        if self.inherit:
            entries.append(
                f"inherit={Component._format_pfd_value(self.inherit)}"
            )
        return f"    {self.name} | {', '.join(entries)}"


@dataclass
class InteractionParameter:
    """A user-supplied binary interaction parameter override."""
    component1: str
    component2: str
    model: str
    scope: Optional[str] = None
    parameters: dict = field(default_factory=dict)

    def to_pfd(self) -> str:
        entries = [f"model={Component._format_pfd_value(self.model)}"]
        if self.scope:
            entries.append(f"scope={Component._format_pfd_value(self.scope)}")
        for key, value in self.parameters.items():
            if key == 'model':
                continue
            entries.append(f"{key}={Component._format_pfd_value(value)}")
        return f"    {self.component1}/{self.component2} | {', '.join(entries)}"


@dataclass
class InteractionEstimation:
    """A global or pair-specific activity-interaction estimation rule."""

    model: str
    component1: Optional[str] = None
    component2: Optional[str] = None
    scope: Optional[str] = None
    parameters: dict = field(default_factory=dict)

    def to_pfd(self) -> str:
        pair_specific = self.component1 is not None and self.component2 is not None
        label = (
            f"{self.component1}/{self.component2}"
            if pair_specific else self.model
        )
        entries = []
        if pair_specific:
            entries.append(f"model={Component._format_pfd_value(self.model)}")
        if self.scope:
            entries.append(f"scope={Component._format_pfd_value(self.scope)}")
        for key, value in self.parameters.items():
            if key == 'model':
                continue
            entries.append(f"{key}={Component._format_pfd_value(value)}")
        return f"    {label} | {', '.join(entries)}"


@dataclass
class Unit:
    """A unit operation in the process"""
    id: str
    unit_type: str = ""
    ports: list[Port] = field(default_factory=list)
    params: list[Parameter] = field(default_factory=list)
    reactions: list[Reaction] = field(default_factory=list)
    # Visual positioning for the editor
    x: float = 0.0
    y: float = 0.0

    def __post_init__(self) -> None:
        canonical = canonical_unit_type(self.unit_type)
        if canonical is not None:
            self.unit_type = canonical
    
    def to_pfd(self) -> str:
        lines = [
            f"UNIT {self.id}",
            f"    TYPE: {self.unit_type}",
            f"    PORTS:"
        ]
        for port in self.ports:
            lines.append(port.to_pfd())
        lines.append("    PARAMS:")
        for param in self.params:
            lines.append(param.to_pfd())
        if self.reactions:
            lines.append("    REACTIONS:")
            for rxn in self.reactions:
                lines.append(rxn.to_pfd())
        return '\n'.join(lines)
    
    def get_port(self, port_id: str) -> Optional[Port]:
        for port in self.ports:
            if port.id == port_id:
                return port
        return None


@dataclass
class Metadata:
    """Process metadata"""
    process_name: str = ""
    version: str = "1.0"
    description: str = ""
    author: str = ""
    date: str = ""
    thermo_method: str = "IDEAL"
    online_lookup: bool = True
    psat_minimum_pressure_bar: Optional[float] = None
    activity_interaction_max_psat_bar: Optional[float] = None
    activity_interaction_max_temperature_K: Optional[float] = None
    recycle_method: str = "WEGSTEIN"
    recycle_options: dict[str, float | int] = field(default_factory=dict)
    recycle_tear_streams: list[str] = field(default_factory=list)
    recycle_trace_tolerance: Optional[float] = None
    fluid_phase_model: str = "VLE"

    def __post_init__(self) -> None:
        self.fluid_phase_model = normalize_fluid_phase_model(
            self.fluid_phase_model
        )
        self.recycle_method = normalize_recycle_method(self.recycle_method)
        self.recycle_options = normalize_recycle_options(
            self.recycle_method,
            self.recycle_options,
        )
    
    def to_pfd(self) -> str:
        lines = []
        if self.process_name:
            lines.append(f"PROCESS: {self.process_name}")
        if self.version:
            lines.append(f"VERSION: {self.version}")
        if self.author:
            lines.append(f"AUTHOR: {self.author}")
        if self.date:
            lines.append(f"DATE: {self.date}")
        if self.thermo_method and self.thermo_method != "IDEAL":
            lines.append(f"THERMO_METHOD: {self.thermo_method}")
        if self.fluid_phase_model and self.fluid_phase_model != "VLE":
            lines.append(f"FLUID_PHASE_MODEL: {self.fluid_phase_model}")
        if self.online_lookup is False:
            lines.append("ONLINE_LOOKUP: false")
        if self.psat_minimum_pressure_bar is not None:
            lines.append(
                "PSAT_MINIMUM_PRESSURE: "
                f"{self.psat_minimum_pressure_bar:g} [bar]"
            )
        if self.activity_interaction_max_psat_bar is not None:
            lines.append(
                "ACTIVITY_INTERACTION_MAX_PSAT: "
                f"{self.activity_interaction_max_psat_bar:g} [bar]"
            )
        if self.activity_interaction_max_temperature_K is not None:
            lines.append(
                "ACTIVITY_INTERACTION_MAX_TEMPERATURE: "
                f"{self.activity_interaction_max_temperature_K:g} [K]"
            )
        if self.recycle_method and (
            self.recycle_method != "WEGSTEIN" or self.recycle_options
        ):
            directive = f"RECYCLE_METHOD: {self.recycle_method}"
            if self.recycle_options:
                rendered_options = ', '.join(
                    f"{name}={value:g}"
                    for name, value in self.recycle_options.items()
                )
                directive += f" | {rendered_options}"
            lines.append(directive)
        if self.recycle_tear_streams:
            lines.append(f"TEAR_STREAMS: {', '.join(self.recycle_tear_streams)}")
        if self.recycle_trace_tolerance is not None:
            lines.append(f"RECYCLE_TRACE_TOLERANCE: {self.recycle_trace_tolerance:g}")
        if self.description:
            lines.append(f"DESCRIPTION: {self.description}")
        return '\n'.join(lines)


@dataclass
class ProcessFlowDiagram:
    """Complete process flow diagram"""
    metadata: Metadata = field(default_factory=Metadata)
    components: list[Component] = field(default_factory=list)
    thermo_scopes: list[ThermoScope] = field(default_factory=list)
    interaction_estimation: list[InteractionEstimation] = field(default_factory=list)
    interaction_parameters: list[InteractionParameter] = field(default_factory=list)
    reaction_definitions: list[NamedReaction] = field(default_factory=list)
    streams: list[Stream] = field(default_factory=list)
    units: list[Unit] = field(default_factory=list)
    
    def get_unit(self, unit_id: str) -> Optional[Unit]:
        for unit in self.units:
            if unit.id == unit_id:
                return unit
        return None
    
    def get_stream(self, stream_id: str) -> Optional[Stream]:
        for stream in self.streams:
            if stream.id == stream_id:
                return stream
        return None
    
    def get_component(self, symbol: str) -> Optional[Component]:
        for comp in self.components:
            if comp.symbol == symbol:
                return comp
        return None

    def get_thermo_scope(self, name: str) -> Optional[ThermoScope]:
        for scope in self.thermo_scopes:
            if scope.name == name:
                return scope
        return None

    def get_reaction_definition(self, name: str) -> Optional[NamedReaction]:
        for definition in self.reaction_definitions:
            if definition.name == name:
                return definition
        return None

    def resolve_reaction(self, reaction: Reaction) -> Reaction:
        """Resolve one unit reaction reference without mutating either object."""
        if not reaction.reference:
            return reaction
        definition = self.get_reaction_definition(reaction.reference)
        if definition is None:
            raise ValueError(f"Undefined reaction reference '@{reaction.reference}'")
        return Reaction(
            equation=definition.reaction.equation,
            parameters=dict(definition.reaction.parameters),
        )
    
    def to_pfd(self) -> str:
        """Serialize the PFD back to .pfd format"""
        sections = []
        
        # Header
        sections.append("#" + "=" * 78)
        sections.append("# PROCESS FLOW DIAGRAM")
        sections.append("#" + "=" * 78)
        sections.append("")
        
        # Metadata
        sections.append("#" + "-" * 78)
        sections.append("# METADATA")
        sections.append("#" + "-" * 78)
        sections.append(self.metadata.to_pfd())
        sections.append("")

        if self.thermo_scopes:
            sections.append("#" + "-" * 78)
            sections.append("# THERMODYNAMIC SCOPES")
            sections.append("#" + "-" * 78)
            sections.append("THERMO_SCOPES:")
            for scope in self.thermo_scopes:
                sections.append(scope.to_pfd())
            sections.append("")
        
        # Components
        sections.append("#" + "-" * 78)
        sections.append("# COMPONENTS")
        sections.append("#" + "-" * 78)
        sections.append("COMPONENTS:")
        for comp in self.components:
            sections.append(comp.to_pfd())
        sections.append("")

        correlation_lines = []
        for comp in self.components:
            correlation_lines.extend(comp.to_property_correlations_pfd())
        if correlation_lines:
            sections.append("#" + "-" * 78)
            sections.append("# PROPERTY CORRELATIONS")
            sections.append("#" + "-" * 78)
            sections.append("PROPERTY_CORRELATIONS:")
            sections.extend(correlation_lines)
            sections.append("")

        if self.interaction_estimation:
            sections.append("#" + "-" * 78)
            sections.append("# INTERACTION ESTIMATION")
            sections.append("#" + "-" * 78)
            sections.append("INTERACTION_ESTIMATION:")
            for rule in self.interaction_estimation:
                sections.append(rule.to_pfd())
            sections.append("")

        if self.interaction_parameters:
            sections.append("#" + "-" * 78)
            sections.append("# INTERACTION PARAMETERS")
            sections.append("#" + "-" * 78)
            sections.append("INTERACTION_PARAMETERS:")
            for interaction in self.interaction_parameters:
                sections.append(interaction.to_pfd())
            sections.append("")

        if self.reaction_definitions:
            sections.append("#" + "-" * 78)
            sections.append("# REACTION CATALOG")
            sections.append("#" + "-" * 78)
            sections.append("REACTIONS:")
            for definition in self.reaction_definitions:
                sections.append(definition.to_pfd())
            sections.append("")
        
        # Streams
        sections.append("#" + "-" * 78)
        sections.append("# STREAMS")
        sections.append("#" + "-" * 78)
        for stream in self.streams:
            sections.append(stream.to_pfd())
            sections.append("")
        
        # Units
        sections.append("#" + "-" * 78)
        sections.append("# UNITS")
        sections.append("#" + "-" * 78)
        for unit in self.units:
            sections.append(unit.to_pfd())
            sections.append("")
        
        sections.append("#" + "-" * 78)
        sections.append("# END OF FILE")
        sections.append("#" + "-" * 78)
        
        return '\n'.join(sections)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'metadata': {
                'process_name': self.metadata.process_name,
                'version': self.metadata.version,
                'description': self.metadata.description,
                'author': self.metadata.author,
                'date': self.metadata.date,
                'thermo_method': self.metadata.thermo_method,
                'fluid_phase_model': self.metadata.fluid_phase_model,
                'online_lookup': self.metadata.online_lookup,
                'psat_minimum_pressure_bar': (
                    self.metadata.psat_minimum_pressure_bar
                ),
                'activity_interaction_max_psat_bar': (
                    self.metadata.activity_interaction_max_psat_bar
                ),
                'activity_interaction_max_temperature_K': (
                    self.metadata.activity_interaction_max_temperature_K
                ),
                'recycle_method': self.metadata.recycle_method,
                'recycle_options': dict(self.metadata.recycle_options),
                'recycle_tear_streams': self.metadata.recycle_tear_streams,
                'recycle_trace_tolerance': self.metadata.recycle_trace_tolerance,
            },
            'components': [
                {
                    key: value
                    for key, value in c.__dict__.items()
                    if value is not None and value != {}
                }
                for c in self.components
            ],
            'thermo_scopes': [
                {
                    'name': scope.name,
                    'method': scope.method,
                    'inherit': scope.inherit,
                }
                for scope in self.thermo_scopes
            ],
            'interaction_parameters': [
                {
                    'component1': item.component1,
                    'component2': item.component2,
                    'model': item.model,
                    'scope': item.scope,
                    'parameters': item.parameters,
                }
                for item in self.interaction_parameters
            ],
            'interaction_estimation': [
                {
                    'model': item.model,
                    'component1': item.component1,
                    'component2': item.component2,
                    'scope': item.scope,
                    'parameters': item.parameters,
                }
                for item in self.interaction_estimation
            ],
            'reaction_definitions': [
                {
                    'name': item.name,
                    'equation': item.reaction.equation,
                    'parameters': item.reaction.parameters,
                }
                for item in self.reaction_definitions
            ],
            'streams': [
                {
                    'id': s.id,
                    'source': {
                        'unit_id': s.source.unit_id,
                        'port_id': s.source.port_id,
                        'is_feed': s.source.is_feed,
                        'is_product': s.source.is_product,
                    },
                    'destination': {
                        'unit_id': s.destination.unit_id,
                        'port_id': s.destination.port_id,
                        'is_feed': s.destination.is_feed,
                        'is_product': s.destination.is_product,
                    },
                    'properties': [
                        {'name': p.name, 'value': p.value, 'unit': p.unit}
                        for p in s.properties
                    ],
                    'composition': s.composition.fractions if s.composition else None,
                    'composition_basis': s.composition.basis if s.composition else None,
                }
                for s in self.streams
            ],
            'units': [
                {
                    'id': u.id,
                    'unit_type': u.unit_type,
                    'x': u.x,
                    'y': u.y,
                    'ports': [
                        {'id': p.id, 'port_type': p.port_type.value}
                        for p in u.ports
                    ],
                    'params': [
                        {'name': p.name, 'value': p.value, 'unit': p.unit}
                        for p in u.params
                    ],
                    'reactions': [
                        {
                            'equation': r.equation,
                            'parameters': r.parameters,
                            'reference': r.reference,
                        }
                        for r in u.reactions
                    ]
                }
                for u in self.units
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProcessFlowDiagram':
        """Create PFD from dictionary (JSON deserialization)"""
        pfd = cls()
        
        # Metadata
        meta = data.get('metadata', {})
        pfd.metadata = Metadata(
            process_name=meta.get('process_name', ''),
            version=meta.get('version', '1.0'),
            description=meta.get('description', ''),
            author=meta.get('author', ''),
            date=meta.get('date', ''),
            thermo_method=meta.get('thermo_method', 'IDEAL'),
            fluid_phase_model=normalize_fluid_phase_model(
                meta.get('fluid_phase_model', 'VLE')
            ),
            online_lookup=meta.get('online_lookup', True),
            psat_minimum_pressure_bar=meta.get(
                'psat_minimum_pressure_bar'
            ),
            activity_interaction_max_psat_bar=meta.get(
                'activity_interaction_max_psat_bar'
            ),
            activity_interaction_max_temperature_K=meta.get(
                'activity_interaction_max_temperature_K'
            ),
            recycle_method=meta.get('recycle_method', 'WEGSTEIN'),
            recycle_options=meta.get('recycle_options', {}),
            recycle_tear_streams=meta.get('recycle_tear_streams', []),
            recycle_trace_tolerance=meta.get('recycle_trace_tolerance'),
        )
        
        # Components
        for c in data.get('components', []):
            component_data = dict(c)
            if component_data.get('phase_behavior') is not None:
                component_data['phase_behavior'] = normalize_phase_behavior(
                    component_data['phase_behavior']
                )
            if component_data.get('solid_material_form') is not None:
                component_data['solid_material_form'] = (
                    normalize_solid_material_form(
                        component_data['solid_material_form']
                    )
                )
            if component_data.get('vapor_dimerization') is not None:
                component_data['vapor_dimerization'] = (
                    normalize_vdm_component_parameters(
                        component_data['vapor_dimerization']
                    )
                )
            pfd.components.append(Component(
                **component_data
            ))

        for item in data.get('thermo_scopes', []):
            inherit = item.get('inherit')
            if inherit and str(inherit).lower() == 'global':
                inherit = 'global'
            pfd.thermo_scopes.append(ThermoScope(
                name=item['name'],
                method=str(item['method']).upper(),
                inherit=inherit,
            ))

        for item in data.get('interaction_parameters', []):
            pfd.interaction_parameters.append(InteractionParameter(
                component1=item['component1'],
                component2=item['component2'],
                model=item['model'],
                scope=item.get('scope'),
                parameters=item.get('parameters', {}),
            ))

        for item in data.get('interaction_estimation', []):
            pfd.interaction_estimation.append(InteractionEstimation(
                model=item['model'],
                component1=item.get('component1'),
                component2=item.get('component2'),
                scope=item.get('scope'),
                parameters=item.get('parameters', {}),
            ))

        for item in data.get('reaction_definitions', []):
            pfd.reaction_definitions.append(NamedReaction(
                name=item['name'],
                reaction=Reaction(
                    equation=item['equation'],
                    parameters=item.get('parameters', {}),
                ),
            ))
        
        # Units
        for u in data.get('units', []):
            unit = Unit(
                id=u['id'],
                unit_type=u['unit_type'],
                x=u.get('x', 0),
                y=u.get('y', 0),
            )
            for p in u.get('ports', []):
                unit.ports.append(Port(
                    id=p['id'],
                    port_type=PortType(p['port_type'])
                ))
            for p in u.get('params', []):
                unit.params.append(Parameter(
                    name=p['name'],
                    value=p['value'],
                    unit=p.get('unit')
                ))
            for r in u.get('reactions', []):
                unit.reactions.append(Reaction(
                    equation=r.get('equation', ''),
                    parameters=r.get('parameters', {}),
                    reference=r.get('reference'),
                ))
            pfd.units.append(unit)
        
        # Streams
        for s in data.get('streams', []):
            src = s['source']
            dst = s['destination']
            stream = Stream(
                id=s['id'],
                source=PortReference(
                    unit_id=src.get('unit_id'),
                    port_id=src.get('port_id'),
                    is_feed=src.get('is_feed', False),
                    is_product=src.get('is_product', False),
                ),
                destination=PortReference(
                    unit_id=dst.get('unit_id'),
                    port_id=dst.get('port_id'),
                    is_feed=dst.get('is_feed', False),
                    is_product=dst.get('is_product', False),
                ),
            )
            for p in s.get('properties', []):
                stream.properties.append(StreamProperty(
                    name=p['name'],
                    value=p['value'],
                    unit=p.get('unit')
                ))
            if s.get('composition'):
                stream.composition = Composition(
                    fractions=s['composition'],
                    basis=s.get('composition_basis') or 'mole',
                )
            pfd.streams.append(stream)
        
        return pfd


class ParseError(Exception):
    """Error during PFD parsing"""
    def __init__(self, message: str, line_number: int = None):
        self.message = message
        self.line_number = line_number
        if line_number:
            super().__init__(f"Line {line_number}: {message}")
        else:
            super().__init__(message)


class ValidationError(Exception):
    """Error during PFD validation"""
    pass


class PFDParser:
    """Parser for .pfd files"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.pfd = ProcessFlowDiagram()
        self.current_line = 0
        self.lines = []
        self.parse_errors: list[tuple[int, str]] = []
        self._compact_unit_objects: set[int] = set()
        self._explicit_port_unit_objects: set[int] = set()
        self._stream_line_numbers: dict[int, int] = {}

    def _record_error(self, error, line_number: Optional[int] = None) -> None:
        """Retain one parse diagnostic while allowing independent rows to parse."""
        if isinstance(error, ParseError):
            message = error.message
            line_number = error.line_number or line_number
        else:
            message = str(error)
        self.parse_errors.append((line_number or self.current_line, message))

    def _raise_recorded_errors(self) -> None:
        if not self.parse_errors:
            return
        count = len(self.parse_errors)
        label = "error" if count == 1 else "errors"
        details = '\n'.join(
            f"  - Line {line_number}: {message}"
            for line_number, message in self.parse_errors
        )
        raise ParseError(f"PFD parsing failed with {count} {label}:\n{details}")

    @staticmethod
    def _strip_inline_comment(line: str) -> str:
        """Strip a token-delimited # comment without damaging quoted text."""
        quote = None
        escape = False
        for index, char in enumerate(line):
            if escape:
                escape = False
                continue
            if quote is not None:
                if char == '\\':
                    escape = True
                elif char == quote:
                    quote = None
                continue
            if char in {'"', "'"}:
                quote = char
                continue
            if char == '#' and (index == 0 or line[index - 1].isspace()):
                return line[:index].rstrip()
        return line
    
    def parse(self, text: str) -> ProcessFlowDiagram:
        """Parse a .pfd file from text content"""
        self.reset()
        self.lines = text.split('\n')
        
        i = 0
        while i < len(self.lines):
            self.current_line = i + 1
            line = self.lines[i]
            stripped = self._strip_inline_comment(line).strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                i += 1
                continue
            if line.startswith((' ', '\t')):
                self._record_error(
                    "Unexpected indented line outside a section, STREAM, or UNIT block."
                )
                i += 1
                continue
            
            # Metadata fields
            if stripped.startswith('PROCESS:'):
                self.pfd.metadata.process_name = stripped[8:].strip()
            elif stripped.startswith('VERSION:') or stripped.startswith('PFD_VERSION:'):
                self.pfd.metadata.version = stripped.split(':', 1)[1].strip()
            elif stripped.startswith('AUTHOR:'):
                self.pfd.metadata.author = stripped[7:].strip()
            elif stripped.startswith('DATE:'):
                self.pfd.metadata.date = stripped[5:].strip()
            elif stripped.startswith('DESCRIPTION:'):
                self.pfd.metadata.description = stripped[12:].strip()
            elif stripped.startswith('THERMO_METHOD:') or stripped.startswith('THERMO:') or stripped.startswith('PROPERTY_METHOD:'):
                # Parse thermodynamic method. Keep the explicit method string so
                # simulator.py can pass it through to create_thermodynamics().
                if stripped.startswith('THERMO_METHOD:'):
                    method = stripped[14:].strip().upper()
                elif stripped.startswith('THERMO:'):
                    method = stripped[7:].strip().upper()
                else:  # PROPERTY_METHOD:
                    method = stripped[16:].strip().upper()
                aliases = {
                    'REDLICH-KWONG': 'RK',
                    'REDLICHKWONG': 'RK',
                    'RKS': 'SRK',
                    'SOAVE-REDLICH-KWONG': 'SRK',
                    'RK-SOAVE': 'SRK',
                    'PENG-ROBINSON': 'PR',
                    'PENG_ROBINSON': 'PR',
                    'PREDICTIVE-SRK': 'PSRK',
                    'PREDICTIVE_SRK': 'PSRK',
                    'RKS_BM': 'RKS-BM',
                    'SRK-BM': 'RKS-BM',
                    'SRK_BM': 'RKS-BM',
                    'RKS_MC': 'SRK-MC',
                    'SRK_MC': 'SRK-MC',
                    'RKS-MC': 'SRK-MC',
                    'RK-SOAVE-MC': 'SRK-MC',
                    'PR_BM': 'PR-BM',
                    'PENG-ROBINSON-BM': 'PR-BM',
                    'PENG_ROBINSON_BM': 'PR-BM',
                    'PR_MC': 'PR-MC',
                    'PENG-ROBINSON-MC': 'PR-MC',
                    'PENG_ROBINSON_MC': 'PR-MC',
                    'PRSV': 'PRSV1',
                    'PR-SV': 'PRSV1',
                    'PR-SV1': 'PRSV1',
                    'PENG-ROBINSON-SV': 'PRSV1',
                    'PENG_ROBINSON_SV': 'PRSV1',
                    'PENG-ROBINSON-SV1': 'PRSV1',
                    'PENG_ROBINSON_SV1': 'PRSV1',
                    'PENG-ROBINSON-STRYJEK-VERA': 'PRSV1',
                    'PENG_ROBINSON_STRYJEK_VERA': 'PRSV1',
                    'PR-SV2': 'PRSV2',
                    'PENG-ROBINSON-SV2': 'PRSV2',
                    'PENG_ROBINSON_SV2': 'PRSV2',
                    'SRK_TWU': 'SRK-TWU',
                    'RKS-TWU': 'SRK-TWU',
                    'RKS_TWU': 'SRK-TWU',
                    'RK-SOAVE-TWU': 'SRK-TWU',
                    'PR_TWU': 'PR-TWU',
                    'PENG-ROBINSON-TWU': 'PR-TWU',
                    'PENG_ROBINSON_TWU': 'PR-TWU',
                    'UNIFAC_RK': 'UNIFAC-RK',
                    'GAMMA-PHI-RK': 'UNIFAC-RK',
                    'GAMMA_PHI_RK': 'UNIFAC-RK',
                    'UNIFAC_PR': 'UNIFAC-PR',
                    'UNIFAC-PENG-ROBINSON': 'UNIFAC-PR',
                    'UNIFAC_PENG_ROBINSON': 'UNIFAC-PR',
                    'GAMMA-PHI-PR': 'UNIFAC-PR',
                    'GAMMA_PHI_PR': 'UNIFAC-PR',
                    'UNIFAC_VDM': 'UNIFAC-VDM',
                    'UNIFAC-DMD': 'UNIFDMD',
                    'UNIFAC_DMD': 'UNIFDMD',
                    'DORTMUND-UNIFAC': 'UNIFDMD',
                    'DORTMUND_UNIFAC': 'UNIFDMD',
                    'MODIFIED-UNIFAC': 'UNIFDMD',
                    'MODIFIED_UNIFAC': 'UNIFDMD',
                    'UNIFAC-NIST': 'UNIFNIST',
                    'UNIFAC_NIST': 'UNIFNIST',
                    'NIST-UNIFAC': 'UNIFNIST',
                    'NIST_UNIFAC': 'UNIFNIST',
                    'NIST-MODIFIED-UNIFAC': 'UNIFNIST',
                    'NIST_MODIFIED_UNIFAC': 'UNIFNIST',
                    'UNIFDMD_VDM': 'UNIFDMD-VDM',
                    'UNIFAC-DMD-VDM': 'UNIFDMD-VDM',
                    'UNIFAC_DMD_VDM': 'UNIFDMD-VDM',
                    'DORTMUND-UNIFAC-VDM': 'UNIFDMD-VDM',
                    'DORTMUND_UNIFAC_VDM': 'UNIFDMD-VDM',
                    'UNIFNIST_VDM': 'UNIFNIST-VDM',
                    'UNIFAC-NIST-VDM': 'UNIFNIST-VDM',
                    'UNIFAC_NIST_VDM': 'UNIFNIST-VDM',
                    'NIST-UNIFAC-VDM': 'UNIFNIST-VDM',
                    'NIST_UNIFAC_VDM': 'UNIFNIST-VDM',
                    'UNIFDMD_RK': 'UNIFDMD-RK',
                    'UNIFAC-DMD-RK': 'UNIFDMD-RK',
                    'UNIFAC_DMD_RK': 'UNIFDMD-RK',
                    'DORTMUND-UNIFAC-RK': 'UNIFDMD-RK',
                    'DORTMUND_UNIFAC_RK': 'UNIFDMD-RK',
                    'UNIFDMD_PR': 'UNIFDMD-PR',
                    'UNIFAC-DMD-PR': 'UNIFDMD-PR',
                    'UNIFAC_DMD_PR': 'UNIFDMD-PR',
                    'DORTMUND-UNIFAC-PR': 'UNIFDMD-PR',
                    'DORTMUND_UNIFAC_PR': 'UNIFDMD-PR',
                    'MODIFIED-UNIFAC-PR': 'UNIFDMD-PR',
                    'MODIFIED_UNIFAC_PR': 'UNIFDMD-PR',
                    'UNIFNIST_RK': 'UNIFNIST-RK',
                    'UNIFAC-NIST-RK': 'UNIFNIST-RK',
                    'UNIFAC_NIST_RK': 'UNIFNIST-RK',
                    'NIST-UNIFAC-RK': 'UNIFNIST-RK',
                    'NIST_UNIFAC_RK': 'UNIFNIST-RK',
                    'UNIFNIST_PR': 'UNIFNIST-PR',
                    'UNIFAC-NIST-PR': 'UNIFNIST-PR',
                    'UNIFAC_NIST_PR': 'UNIFNIST-PR',
                    'NIST-UNIFAC-PR': 'UNIFNIST-PR',
                    'NIST_UNIFAC_PR': 'UNIFNIST-PR',
                    'NIST-MODIFIED-UNIFAC-PR': 'UNIFNIST-PR',
                    'NIST_MODIFIED_UNIFAC_PR': 'UNIFNIST-PR',
                    'NRTL_RK': 'NRTL-RK',
                    'NRTL_PR': 'NRTL-PR',
                    'NRTL_VDM': 'NRTL-VDM',
                    'NRTL-PENG-ROBINSON': 'NRTL-PR',
                    'NRTL_PENG_ROBINSON': 'NRTL-PR',
                    'UNIQUAC_RK': 'UNIQUAC-RK',
                    'UNIQUAC_PR': 'UNIQUAC-PR',
                    'UNIQUAC-PENG-ROBINSON': 'UNIQUAC-PR',
                    'UNIQUAC_PENG_ROBINSON': 'UNIQUAC-PR',
                    'UNIQUAC_VDM': 'UNIQUAC-VDM',
                    'IF97': 'STEAM',
                    'IAPWS-IF97': 'STEAM',
                    'IAPWS_IF97': 'STEAM',
                }
                supported = {
                    'IDEAL', 'STEAM', 'RK', 'SRK', 'PR', 'PSRK', 'RKS-BM', 'PR-BM',
                    'SRK-MC', 'PR-MC', 'PRSV1', 'PRSV2', 'SRK-TWU', 'PR-TWU',
                    'UNIFAC', 'UNIFAC2', 'UNIFDMD', 'UNIFM2', 'UNIFNIST',
                    'UNIFAC-VDM', 'UNIFDMD-VDM', 'UNIFNIST-VDM',
                    'UNIFAC-RK', 'UNIFDMD-RK', 'UNIFNIST-RK',
                    'UNIFAC-PR', 'UNIFDMD-PR', 'UNIFNIST-PR',
                    'NRTL', 'NRTL-VDM', 'NRTL-RK', 'NRTL-PR',
                    'UNIQUAC', 'UNIQUAC-VDM', 'UNIQUAC-RK', 'UNIQUAC-PR',
                }
                method = aliases.get(method, method)
                if method in supported:
                    self.pfd.metadata.thermo_method = method
                else:
                    self._record_error(
                        unsupported_thermo_method_message(
                            method,
                            supported,
                            aliases,
                        )
                    )
            elif (
                stripped.startswith('FLUID_PHASE_MODEL:')
                or stripped.startswith('FLUID_PHASES:')
            ):
                value = stripped.split(':', 1)[1].strip()
                try:
                    self.pfd.metadata.fluid_phase_model = (
                        normalize_fluid_phase_model(value)
                    )
                except ValueError:
                    self._record_error(
                        unsupported_fluid_phase_model_message(value)
                    )
            elif (
                stripped.startswith('ONLINE_LOOKUP:')
                or stripped.startswith('ALLOW_ONLINE_LOOKUP:')
                or stripped.startswith('FETCH_ONLINE:')
            ):
                value = stripped.split(':', 1)[1].strip().lower()
                if value in {'true', 'yes', '1', 'on', 'enabled', 'enable'}:
                    self.pfd.metadata.online_lookup = True
                elif value in {'false', 'no', '0', 'off', 'disabled', 'disable'}:
                    self.pfd.metadata.online_lookup = False
                else:
                    self._record_error(
                        unknown_name_message(
                            'ONLINE_LOOKUP value',
                            value,
                            ('true', 'false'),
                        )
                    )
            elif any(
                stripped.startswith(f"{name}:")
                for name in (
                    "PSAT_MINIMUM_PRESSURE",
                    "MINIMUM_PRESSURE",
                    "MINIMUM_PSAT_PRESSURE",
                    "MINIMUM_VAPOR_PRESSURE",
                )
            ):
                value = stripped.split(':', 1)[1].strip()
                try:
                    self.pfd.metadata.psat_minimum_pressure_bar = (
                        self._parse_pressure_bar(value)
                    )
                except ParseError as error:
                    self._record_error(error)
            elif stripped.startswith('ACTIVITY_INTERACTION_MAX_PSAT:'):
                value = stripped.split(':', 1)[1].strip()
                try:
                    self.pfd.metadata.activity_interaction_max_psat_bar = (
                        self._parse_pressure_bar(
                            value,
                            'ACTIVITY_INTERACTION_MAX_PSAT',
                        )
                    )
                except ParseError as error:
                    self._record_error(error)
            elif stripped.startswith('ACTIVITY_INTERACTION_MAX_TEMPERATURE:'):
                value = stripped.split(':', 1)[1].strip()
                try:
                    self.pfd.metadata.activity_interaction_max_temperature_K = (
                        self._parse_temperature_K(
                            value,
                            'ACTIVITY_INTERACTION_MAX_TEMPERATURE',
                        )
                    )
                except ParseError as error:
                    self._record_error(error)
            elif stripped.startswith('RECYCLE_METHOD:') or stripped.startswith('RECYCLE_SOLVER:'):
                value = stripped.split(':', 1)[1].strip()
                pieces = self._split_top_level(value, '|')
                if len(pieces) > 2:
                    self._record_error(
                        "RECYCLE_METHOD accepts one optional '| key=value, ...' block"
                    )
                    continue
                method = pieces[0].strip().upper().replace('-', '_')
                normalized_method = RECYCLE_METHOD_ALIASES.get(method)
                if normalized_method is None:
                    self._record_error(
                        unknown_name_message(
                            'recycle method', method, RECYCLE_METHOD_ALIASES
                        )
                    )
                else:
                    try:
                        raw_options = (
                            self._parse_key_value_properties(pieces[1])
                            if len(pieces) == 2 else {}
                        )
                        options = {}
                        for raw_name, option_value in raw_options.items():
                            name = str(raw_name).strip().lower()
                            if name in options:
                                raise ValueError(
                                    f"Duplicate recycle option '{name}'."
                                )
                            options[name] = option_value
                        normalized_options = normalize_recycle_options(
                            normalized_method,
                            options,
                        )
                    except ValueError as error:
                        self._record_error(str(error))
                    else:
                        self.pfd.metadata.recycle_method = normalized_method
                        self.pfd.metadata.recycle_options = normalized_options
            elif (
                stripped.startswith('TEAR_STREAMS:')
                or stripped.startswith('RECYCLE_TEAR_STREAMS:')
                or stripped.startswith('TEAR_STREAM:')
            ):
                value = stripped.split(':', 1)[1].strip()
                streams = [
                    item.strip()
                    for item in value.replace(';', ',').split(',')
                    if item.strip()
                ]
                self.pfd.metadata.recycle_tear_streams = streams
            elif stripped.startswith('RECYCLE_TRACE_TOLERANCE:') or stripped.startswith('TRACE_TOLERANCE:'):
                value = stripped.split(':', 1)[1].strip()
                try:
                    self.pfd.metadata.recycle_trace_tolerance = float(value)
                except ValueError:
                    self._record_error(
                        f"Invalid recycle trace tolerance '{value}'. Expected a number."
                    )
            
            # Components section
            elif stripped == 'COMPONENTS:':
                i = self._parse_components(i + 1)
                continue

            elif stripped == 'THERMO_SCOPES:':
                i = self._parse_thermo_scopes(i + 1)
                continue

            elif stripped == 'PROPERTY_CORRELATIONS:':
                i = self._parse_property_correlations(i + 1)
                continue

            elif stripped == 'INTERACTION_PARAMETERS:':
                i = self._parse_interaction_parameters(i + 1)
                continue

            elif stripped == 'INTERACTION_ESTIMATION:':
                i = self._parse_interaction_estimation(i + 1)
                continue

            elif stripped == 'REACTIONS:':
                i = self._parse_reaction_definitions(i + 1)
                continue
            
            # Stream definition
            elif stripped.startswith('STREAM '):
                i = self._parse_stream(i)
                continue
            
            # Unit definition
            elif stripped.startswith('UNIT '):
                i = self._parse_unit(i)
                continue

            else:
                token = stripped.split(':', 1)[0].split(None, 1)[0]
                if token in _TOP_LEVEL_DIRECTIVES:
                    message = f"Malformed {token} directive: {stripped!r}."
                else:
                    message = unknown_name_message(
                        'top-level directive', token, _TOP_LEVEL_DIRECTIVES
                    )
                self._record_error(message)

            i += 1

        self._finalize_compact_unit_ports()
        self._raise_recorded_errors()
        return self.pfd
    
    def _parse_components(self, start: int) -> int:
        """Parse the COMPONENTS section"""
        i = start
        while i < len(self.lines):
            line = self.lines[i]
            stripped = self._strip_inline_comment(line).strip()
            
            # Check if we've left the components section
            if stripped and not stripped.startswith('#') and not line.startswith(' ') and not line.startswith('\t'):
                return i
            
            if stripped and not stripped.startswith('#'):
                # Parse component line: PFD symbol | lookup identifier | overrides
                parts = [p.strip() for p in stripped.split('|')]
                if len(parts) < 2 or not parts[0] or not parts[1]:
                    self._record_error(
                        "Invalid COMPONENTS row. Expected "
                        "'PFD symbol | identifier | key=value, ...'.",
                        i + 1,
                    )
                    i += 1
                    continue
                symbol = parts[0]
                identifier = parts[1]
                props_str = '|'.join(parts[2:]) if len(parts) >= 3 else ''
                try:
                    properties = self._parse_key_value_properties(props_str)
                    component_properties = self._component_properties_from_mapping(
                        properties
                    )
                    self._validate_component_antoine(
                        symbol,
                        component_properties,
                        i + 1,
                    )
                    self.pfd.components.append(Component(
                        symbol=symbol,
                        name=identifier,
                        **component_properties,
                    ))
                except (ParseError, TypeError, ValueError) as error:
                    if isinstance(error, ParseError):
                        self._record_error(error, i + 1)
                    else:
                        self._record_error(
                            f"Invalid component property for '{symbol}': {error}",
                            i + 1,
                        )
            
            i += 1
        
        return i

    def _parse_reaction_definitions(self, start: int) -> int:
        """Parse reusable ``name : equation | parameters`` reactions."""
        i = start
        names = {item.name for item in self.pfd.reaction_definitions}
        while i < len(self.lines):
            line = self.lines[i]
            stripped = self._strip_inline_comment(line).strip()
            if (
                stripped
                and not stripped.startswith('#')
                and not line.startswith((' ', '\t'))
            ):
                return i
            if stripped and not stripped.startswith('#'):
                match = re.fullmatch(
                    r'([A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*(.+)',
                    stripped,
                )
                if not match:
                    self._record_error(
                        "Invalid top-level REACTIONS row. Expected "
                        "'name : equation | key=value, ...'.",
                        i + 1,
                    )
                    i += 1
                    continue
                name = match.group(1)
                if name in names:
                    self._record_error(
                        f"Duplicate reaction definition: {name}",
                        i + 1,
                    )
                    i += 1
                    continue
                try:
                    reaction = self._parse_reaction(match.group(2))
                    if reaction.reference:
                        raise ValueError(
                            "top-level reaction definitions cannot reference another reaction"
                        )
                    self.pfd.reaction_definitions.append(NamedReaction(
                        name=name,
                        reaction=reaction,
                    ))
                    names.add(name)
                except ValueError as error:
                    self._record_error(
                        f"Invalid top-level reaction '{name}': {error}",
                        i + 1,
                    )
            i += 1
        return i

    def _parse_property_correlations(self, start: int) -> int:
        """Parse the PROPERTY_CORRELATIONS section."""
        i = start
        while i < len(self.lines):
            line = self.lines[i]
            stripped = self._strip_inline_comment(line).strip()

            if stripped and not stripped.startswith('#') and not line.startswith(' ') and not line.startswith('\t'):
                return i

            if stripped and not stripped.startswith('#'):
                component_symbol = ''
                correlation_key = ''
                try:
                    parts = [p.strip() for p in stripped.split('|', 1)]
                    if len(parts) != 2 or '.' not in parts[0]:
                        raise ParseError(
                            "Invalid PROPERTY_CORRELATIONS row. Expected "
                            "'component.property | key=value, ...'.",
                            i + 1,
                        )
                    component_symbol, correlation_key = [p.strip() for p in parts[0].split('.', 1)]
                    if not component_symbol or not correlation_key:
                        raise ParseError(
                            "Invalid PROPERTY_CORRELATIONS component/property reference.",
                            i + 1,
                        )
                    component = self.pfd.get_component(component_symbol)
                    if component is None:
                        raise ParseError(
                            unknown_name_message(
                                'PROPERTY_CORRELATIONS component',
                                component_symbol,
                                (item.symbol for item in self.pfd.components),
                            ),
                            i + 1,
                        )
                    if correlation_key not in _PROPERTY_CORRELATION_EQUATIONS:
                        raise ParseError(
                            unknown_name_message(
                                'PROPERTY_CORRELATIONS property',
                                correlation_key,
                                _PROPERTY_CORRELATION_EQUATIONS,
                            ),
                            i + 1,
                        )
                    raw_correlation = self._parse_key_value_properties(parts[1])
                    self._validate_property_correlation_fields(
                        correlation_key,
                        raw_correlation,
                    )
                    correlation = self._correlation_from_mapping(
                        raw_correlation
                    )
                    if correlation_key.strip().lower() == 'psat':
                        self._validate_psat_correlation(
                            component_symbol,
                            correlation,
                            i + 1,
                        )
                    component.property_correlations[correlation_key] = correlation
                except ParseError as error:
                    self._record_error(error, i + 1)
                except (TypeError, ValueError) as error:
                    label = (
                        f"{component_symbol}.{correlation_key}"
                        if component_symbol and correlation_key
                        else 'property'
                    )
                    self._record_error(
                        f"Invalid {label} correlation: {error}",
                        i + 1,
                    )

            i += 1

        return i

    def _parse_thermo_scopes(self, start: int) -> int:
        """Parse named thermodynamic-method contexts."""
        i = start
        while i < len(self.lines):
            line = self.lines[i]
            stripped = self._strip_inline_comment(line).strip()

            if (
                stripped
                and not stripped.startswith('#')
                and not line.startswith((' ', '\t'))
            ):
                return i

            if stripped and not stripped.startswith('#'):
                try:
                    parts = [part.strip() for part in stripped.split('|', 1)]
                    if len(parts) != 2 or not parts[0]:
                        raise ParseError(
                            "Invalid THERMO_SCOPES row. Expected "
                            "'name | method=..., inherit=...'.",
                            i + 1,
                        )
                    name = parts[0]
                    parameters = self._parse_key_value_properties(parts[1])
                    unknown = sorted(set(parameters) - {'method', 'inherit'})
                    if unknown:
                        raise ParseError(
                            unknown_name_message(
                                'THERMO_SCOPES field',
                                unknown[0],
                                ('method', 'inherit'),
                            ),
                            i + 1,
                        )
                    method = parameters.get('method')
                    if method is None or not str(method).strip():
                        raise ParseError(
                            f"THERMO_SCOPES scope '{name}' requires method=... .",
                            i + 1,
                        )
                    inherit = parameters.get('inherit')
                    inherit_name = (
                        str(inherit).strip()
                        if inherit is not None and str(inherit).strip()
                        else None
                    )
                    if inherit_name and inherit_name.lower() == 'global':
                        inherit_name = 'global'
                    self.pfd.thermo_scopes.append(ThermoScope(
                        name=name,
                        method=str(method).strip().upper(),
                        inherit=inherit_name,
                    ))
                except ParseError as error:
                    self._record_error(error, i + 1)
                except (TypeError, ValueError) as error:
                    self._record_error(
                        f"Invalid THERMO_SCOPES row: {error}",
                        i + 1,
                    )
            i += 1
        return i

    def _parse_interaction_parameters(self, start: int) -> int:
        """Parse the INTERACTION_PARAMETERS section."""
        i = start
        while i < len(self.lines):
            line = self.lines[i]
            stripped = self._strip_inline_comment(line).strip()

            if stripped and not stripped.startswith('#') and not line.startswith(' ') and not line.startswith('\t'):
                return i

            if stripped and not stripped.startswith('#'):
                try:
                    parts = [p.strip() for p in stripped.split('|', 1)]
                    if len(parts) != 2 or '/' not in parts[0]:
                        raise ParseError(
                            "Invalid INTERACTION_PARAMETERS row. Expected "
                            "'component1/component2 | model=..., key=value, ...'.",
                            i + 1,
                        )
                    component1, component2 = [p.strip() for p in parts[0].split('/', 1)]
                    if not component1 or not component2:
                        raise ParseError(
                            "Invalid INTERACTION_PARAMETERS component pair.",
                            i + 1,
                        )
                    parameters = self._parse_key_value_properties(parts[1])
                    model = parameters.pop('model', None)
                    scope = parameters.pop('scope', None)
                    if scope is not None:
                        scope = str(scope).strip()
                        if not scope:
                            raise ParseError(
                                "INTERACTION_PARAMETERS scope cannot be empty.",
                                i + 1,
                            )
                        if (
                            scope != 'global'
                            and self.pfd.get_thermo_scope(scope) is None
                        ):
                            raise ParseError(
                                f"INTERACTION_PARAMETERS scope '{scope}' must "
                                "be declared earlier in THERMO_SCOPES.",
                                i + 1,
                            )
                    if not model:
                        raise ParseError(
                            "INTERACTION_PARAMETERS row must include model=...",
                            i + 1,
                        )
                    normalized_model = normalize_interaction_model(model)
                    allowed_fields = _INTERACTION_PARAMETER_FIELDS.get(normalized_model)
                    if allowed_fields is None:
                        raise ParseError(
                            unknown_name_message(
                                'interaction model',
                                model,
                                sorted(
                                    set(_INTERACTION_PARAMETER_FIELDS)
                                    | set(_INTERACTION_MODEL_ALIASES)
                                ),
                            ),
                            i + 1,
                        )
                    component_symbols = tuple(
                        component.symbol for component in self.pfd.components
                    )
                    for component_symbol in (component1, component2):
                        if component_symbol not in component_symbols:
                            raise ParseError(
                                unknown_name_message(
                                    'INTERACTION_PARAMETERS component',
                                    component_symbol,
                                    component_symbols,
                                ),
                                i + 1,
                            )
                    for field_name in parameters:
                        normalized_field = str(field_name).strip().lower().replace('-', '_')
                        if normalized_field not in allowed_fields:
                            raise ParseError(
                                unknown_name_message(
                                    f'{normalized_model} INTERACTION_PARAMETERS field',
                                    field_name,
                                    allowed_fields,
                                ),
                                i + 1,
                            )
                    if normalized_model == 'VDM':
                        normalize_vdm_cross_parameters(parameters)
                    self.pfd.interaction_parameters.append(InteractionParameter(
                        component1=component1,
                        component2=component2,
                        model=str(model),
                        scope=(str(scope).strip() if scope is not None else None),
                        parameters=parameters,
                    ))
                except ParseError as error:
                    self._record_error(error, i + 1)
                except (TypeError, ValueError) as error:
                    self._record_error(
                        f"Invalid INTERACTION_PARAMETERS row: {error}",
                        i + 1,
                    )

            i += 1

        return i

    def _parse_interaction_estimation(self, start: int) -> int:
        """Parse global and pair-specific interaction-estimation rules."""
        i = start
        component_symbols = tuple(
            component.symbol for component in self.pfd.components
        )
        while i < len(self.lines):
            line = self.lines[i]
            stripped = self._strip_inline_comment(line).strip()

            if (
                stripped
                and not stripped.startswith('#')
                and not line.startswith(' ')
                and not line.startswith('\t')
            ):
                return i

            if stripped and not stripped.startswith('#'):
                try:
                    parts = [part.strip() for part in stripped.split('|', 1)]
                    if len(parts) != 2 or not parts[0]:
                        raise ParseError(
                            "Invalid INTERACTION_ESTIMATION row. Expected "
                            "'UNIQUAC | source=...' or "
                            "'component1/component2 | model=UNIQUAC, ...'.",
                            i + 1,
                        )
                    parameters = self._parse_key_value_properties(parts[1])
                    scope = parameters.pop('scope', None)
                    if scope is not None:
                        scope = str(scope).strip()
                        if not scope:
                            raise ParseError(
                                "INTERACTION_ESTIMATION scope cannot be empty.",
                                i + 1,
                            )
                        if (
                            scope != 'global'
                            and self.pfd.get_thermo_scope(scope) is None
                        ):
                            raise ParseError(
                                f"INTERACTION_ESTIMATION scope '{scope}' must "
                                "be declared earlier in THERMO_SCOPES.",
                                i + 1,
                            )
                    component1 = component2 = None
                    if '/' in parts[0]:
                        component1, component2 = [
                            value.strip() for value in parts[0].split('/', 1)
                        ]
                        if not component1 or not component2:
                            raise ParseError(
                                "Invalid INTERACTION_ESTIMATION component pair.",
                                i + 1,
                            )
                        model = parameters.pop('model', None)
                        if not model:
                            raise ParseError(
                                "Pair-specific INTERACTION_ESTIMATION rows require "
                                "model=NRTL or model=UNIQUAC.",
                                i + 1,
                            )
                        for component_symbol in (component1, component2):
                            if component_symbol not in component_symbols:
                                raise ParseError(
                                    unknown_name_message(
                                        'INTERACTION_ESTIMATION component',
                                        component_symbol,
                                        component_symbols,
                                    ),
                                    i + 1,
                                )
                    else:
                        model = parts[0]
                        if 'model' in parameters:
                            raise ParseError(
                                "Global INTERACTION_ESTIMATION rows declare the "
                                "destination model before '|'; remove model=... .",
                                i + 1,
                            )

                    normalized_model = normalize_interaction_model(model)
                    if normalized_model not in {'NRTL', 'UNIQUAC'}:
                        raise ParseError(
                            "INTERACTION_ESTIMATION destination model must be "
                            "NRTL or UNIQUAC.",
                            i + 1,
                        )
                    for field_name in parameters:
                        normalized_field = (
                            str(field_name).strip().lower().replace('-', '_')
                        )
                        if normalized_field not in _INTERACTION_ESTIMATION_FIELDS:
                            raise ParseError(
                                unknown_name_message(
                                    f'{normalized_model} INTERACTION_ESTIMATION field',
                                    field_name,
                                    _INTERACTION_ESTIMATION_FIELDS,
                                ),
                                i + 1,
                            )
                    self.pfd.interaction_estimation.append(
                        InteractionEstimation(
                            model=normalized_model,
                            component1=component1,
                            component2=component2,
                            scope=(
                                str(scope).strip() if scope is not None else None
                            ),
                            parameters=parameters,
                        )
                    )
                except ParseError as error:
                    self._record_error(error, i + 1)
                except (TypeError, ValueError) as error:
                    self._record_error(
                        f"Invalid INTERACTION_ESTIMATION row: {error}",
                        i + 1,
                    )
            i += 1
        return i

    def _component_properties_from_mapping(self, properties: dict) -> dict:
        case_lookup = {key.lower(): key for key in _COMPONENT_PROPERTY_KEYS}
        parsed = {}
        for key, value in properties.items():
            attr = _COMPONENT_PROPERTY_ALIASES.get(key.lower()) or case_lookup.get(key.lower())
            if not attr:
                raise ValueError(
                    unknown_name_message(
                        'component property',
                        key,
                        _COMPONENT_PROPERTY_NAMES,
                    )
                )
            if attr == 'unifac_groups':
                parsed[attr] = self._parse_unifac_groups(value)
            elif attr == 'vapor_dimerization':
                parsed[attr] = normalize_vdm_component_parameters(value)
            elif attr == 'Cp_coeffs':
                parsed[attr] = self._as_float_list(value)
            elif attr == 'critical_properties_unavailable':
                parsed[attr] = self._as_bool(value)
            elif attr == 'solid_material_form':
                parsed[attr] = normalize_solid_material_form(value)
            elif attr == 'phase_behavior':
                parsed[attr] = normalize_phase_behavior(value)
            elif attr in {
                'formula', 'CAS', 'smiles', 'antoine_source', 'phase_at_STP',
                'solid_material_form', 'solid_polymorph', 'phase_behavior',
            }:
                parsed[attr] = str(value)
            else:
                parsed[attr] = float(value)
        return parsed

    @staticmethod
    def _validate_property_correlation_fields(
        correlation_key: str,
        properties: dict,
    ) -> None:
        canonical_properties = {
            (
                name
                if name in _TOP_LEVEL_CORRELATION_COEFFICIENTS
                else canonical_correlation_field_name(name)
            ): value
            for name, value in properties.items()
        }
        for field_name in properties:
            if (
                field_name in _TOP_LEVEL_CORRELATION_COEFFICIENTS
                or canonical_correlation_field_name(field_name)
                in _PROPERTY_CORRELATION_FIELDS
            ):
                continue
            raise ValueError(
                unknown_name_message(
                    f'{correlation_key} correlation field',
                    field_name,
                    _PROPERTY_CORRELATION_FIELDS,
                )
            )

        # Psat already has a stricter, domain-specific validator. Leave its
        # established missing/extra coefficient diagnostics authoritative.
        if correlation_key == 'Psat':
            return

        equation = str(canonical_properties.get('equation') or '').strip().lower()
        if not equation:
            raise ValueError(
                f"{correlation_key} correlation requires equation=..."
            )
        supported_equations = _PROPERTY_CORRELATION_EQUATIONS[correlation_key]
        if equation not in supported_equations:
            raise ValueError(
                unknown_name_message(
                    f'{correlation_key} correlation equation',
                    canonical_properties.get('equation'),
                    supported_equations,
                )
            )

        allowed_coefficients = _PROPERTY_CORRELATION_COEFFICIENTS[equation]
        supplied_coefficients = {
            name
            for name in properties
            if name in _TOP_LEVEL_CORRELATION_COEFFICIENTS
        }
        nested_coefficients = canonical_properties.get('coefficients')
        if isinstance(nested_coefficients, dict):
            supplied_coefficients.update(str(name) for name in nested_coefficients)
        for coefficient_name in supplied_coefficients:
            if coefficient_name not in allowed_coefficients:
                raise ValueError(
                    unknown_name_message(
                        f"{correlation_key} {equation} coefficient",
                        coefficient_name,
                        allowed_coefficients,
                    )
                )

    @staticmethod
    def _validate_component_antoine(
        component_symbol: str,
        properties: dict,
        line_number: int,
    ) -> None:
        supplied = tuple(
            field_name
            for field_name in _ANTOINE_OVERRIDE_FIELDS
            if properties.get(field_name) is not None
        )
        if not supplied:
            return
        missing = tuple(
            field_name
            for field_name in _ANTOINE_OVERRIDE_FIELDS
            if properties.get(field_name) is None
        )
        if missing:
            raise ParseError(
                f"Component '{component_symbol}' Antoine override must provide the "
                "atomic A, B, C, Tmin, Tmax bundle; missing " + ', '.join(missing),
                line_number,
            )

        A, B, C, T_min, T_max = (
            float(properties[field_name])
            for field_name in _ANTOINE_OVERRIDE_FIELDS
        )
        if any(not math.isfinite(value) for value in (A, B, C, T_min, T_max)):
            raise ParseError(
                f"Component '{component_symbol}' Antoine values must be finite",
                line_number,
            )
        if B <= 0.0:
            raise ParseError(
                f"Component '{component_symbol}' Antoine B must be positive",
                line_number,
            )
        if T_min <= 0.0 or T_max <= T_min:
            raise ParseError(
                f"Component '{component_symbol}' Antoine requires "
                "0 < antoine_Tmin < antoine_Tmax",
                line_number,
            )
        denominator_min = C + T_min - 273.15
        denominator_max = C + T_max - 273.15
        if (
            denominator_min == 0.0
            or denominator_max == 0.0
            or min(denominator_min, denominator_max) < 0.0
            < max(denominator_min, denominator_max)
        ):
            raise ParseError(
                f"Component '{component_symbol}' Antoine denominator is singular "
                "inside its declared range",
                line_number,
            )
        for temperature, denominator in (
            (T_min, denominator_min),
            (T_max, denominator_max),
        ):
            ln_pressure = math.log(10.0) * (A - B / denominator)
            if not math.isfinite(ln_pressure):
                raise ParseError(
                    f"Component '{component_symbol}' Antoine pressure is invalid "
                    f"at {temperature:g} K",
                    line_number,
                )

    @staticmethod
    def _validate_psat_correlation(
        component_symbol: str,
        correlation: dict,
        line_number: int,
    ) -> None:
        equation = str(correlation.get('equation') or '').strip().lower()
        if not equation:
            raise ParseError(
                f"Component '{component_symbol}' Psat correlation requires equation=...",
                line_number,
            )
        contract = _PSAT_EQUATION_COEFFICIENTS.get(equation)
        if contract is None:
            suggestion = closest_name(equation, _PSAT_EQUATION_COEFFICIENTS)
            suggestion_text = (
                f" Did you mean '{suggestion}'?"
                if suggestion else ''
            )
            raise ParseError(
                f"Component '{component_symbol}' Psat equation {equation!r} "
                f"is unsupported.{suggestion_text}",
                line_number,
            )

        required, allowed = contract
        coefficients = correlation.get('coefficients')
        if not isinstance(coefficients, dict):
            coefficients = {}
        missing = tuple(name for name in required if name not in coefficients)
        if missing:
            raise ParseError(
                f"Component '{component_symbol}' Psat equation {equation!r} "
                "requires coefficient(s) " + ', '.join(missing),
                line_number,
            )
        unexpected = tuple(sorted(set(coefficients) - allowed))
        if unexpected:
            raise ParseError(
                f"Component '{component_symbol}' Psat equation {equation!r} "
                "does not accept coefficient(s) " + ', '.join(unexpected),
                line_number,
            )
        for name, value in coefficients.items():
            if not math.isfinite(float(value)):
                raise ParseError(
                    f"Component '{component_symbol}' Psat coefficient {name} "
                    "must be finite",
                    line_number,
                )

        for name in (
            'Tmin_K', 'Tmax_K', 'Tc_K', 'Tb_K', 'Pc_bar', 'Pc_Pa',
            'Pmin_bar', 'Pmax_bar',
        ):
            value = correlation.get(name)
            if value is None:
                continue
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ParseError(
                    f"Component '{component_symbol}' Psat {name} must be positive "
                    "and finite",
                    line_number,
                )
        T_min = correlation.get('Tmin_K')
        T_max = correlation.get('Tmax_K')
        if T_min is not None and T_max is not None and float(T_max) <= float(T_min):
            raise ParseError(
                f"Component '{component_symbol}' Psat requires Tmin_K < Tmax_K",
                line_number,
            )
        P_min = correlation.get('Pmin_bar')
        P_max = correlation.get('Pmax_bar')
        if P_min is not None and P_max is not None and float(P_max) < float(P_min):
            raise ParseError(
                f"Component '{component_symbol}' Psat requires Pmin_bar <= Pmax_bar",
                line_number,
            )

        if equation == 'poly_x' and T_min is not None and T_max is not None:
            from numpy.polynomial import Polynomial

            polynomial = Polynomial([
                float(coefficients.get(name, 0.0))
                for name in 'ABCDEF'
            ])
            x_min = (float(T_min) - 298.15) / 100.0
            x_max = (float(T_max) - 298.15) / 100.0
            candidates = [x_min, x_max]
            for root in polynomial.deriv().roots():
                if abs(float(root.imag)) > 1.0e-10 * (
                    1.0 + abs(float(root.real))
                ):
                    continue
                value = float(root.real)
                if x_min <= value <= x_max:
                    candidates.append(value)
            minimum_pressure = min(
                float(polynomial(value))
                for value in candidates
            )
            if (
                not math.isfinite(minimum_pressure)
                or minimum_pressure <= 0.0
            ):
                raise ParseError(
                    f"Component '{component_symbol}' Psat poly_x must remain "
                    "positive and finite throughout its declared temperature "
                    "range",
                    line_number,
                )

        quality = correlation.get('quality')
        if quality is not None and (
            not math.isfinite(float(quality))
            or not 0.0 <= float(quality) <= 1.0
        ):
            raise ParseError(
                f"Component '{component_symbol}' Psat quality must be between 0 and 1",
                line_number,
            )

        inverse_power = correlation.get('inverse_power')
        if inverse_power is not None and not math.isfinite(float(inverse_power)):
            raise ParseError(
                f"Component '{component_symbol}' Psat inverse_power must be finite",
                line_number,
            )
        if equation == 'canonical_psat_ah':
            if inverse_power is None:
                raise ParseError(
                    f"Component '{component_symbol}' canonical A-H Psat requires "
                    "inverse_power",
                    line_number,
                )
            power = int(inverse_power)
            if float(inverse_power) != power or power not in {-3, -5, -7}:
                raise ParseError(
                    f"Component '{component_symbol}' canonical A-H Psat "
                    "inverse_power must be -3, -5, or -7",
                    line_number,
                )
        elif inverse_power is not None:
            raise ParseError(
                f"Component '{component_symbol}' Psat equation {equation!r} "
                "cannot declare inverse_power",
                line_number,
            )

        Pc_bar = correlation.get('Pc_bar')
        Pc_pa = correlation.get('Pc_Pa')
        if Pc_bar is not None and Pc_pa is not None:
            pressure_from_pa = float(Pc_pa) / 100000.0
            scale = max(abs(float(Pc_bar)), abs(pressure_from_pa), 1.0)
            if abs(float(Pc_bar) - pressure_from_pa) > 1.0e-9 * scale:
                raise ParseError(
                    f"Component '{component_symbol}' Psat Pc_bar and Pc_Pa conflict",
                    line_number,
                )

    def _correlation_from_mapping(self, properties: dict) -> dict:
        correlation = {}
        coefficients = {}
        for key, value in properties.items():
            canonical_key = canonical_correlation_field_name(key)
            if canonical_key == 'coefficients' and isinstance(value, dict):
                for coeff_key, coeff_value in value.items():
                    coefficients[str(coeff_key)] = float(coeff_value)
            elif canonical_key in _TOP_LEVEL_CORRELATION_COEFFICIENTS:
                coefficients[canonical_key] = float(value)
            elif canonical_key in {
                'Tmin_K', 'Tmax_K', 'Tc_K', 'Tb_K', 'Pc_bar', 'Pc_Pa',
                'Pmin_bar', 'Pmax_bar', 'T_ref_K', 'rho_kg_m3', 'quality',
                'inverse_power',
            }:
                correlation[canonical_key] = float(value)
            else:
                correlation[canonical_key] = value
        if coefficients:
            correlation['coefficients'] = coefficients
        return correlation

    def _parse_key_value_properties(self, text: str) -> dict:
        properties = {}
        for item in self._split_top_level(text, ','):
            if not item.strip():
                continue
            if '=' not in item:
                raise ValueError(
                    f"Invalid override entry {item.strip()!r}; expected key=value."
                )
            key, value = item.split('=', 1)
            key = key.strip()
            if not key:
                raise ValueError("Override property names cannot be empty.")
            if key in properties:
                raise ValueError(f"Duplicate override property '{key}'.")
            properties[key] = self._parse_property_value(value.strip())
        return properties

    def _parse_property_value(self, value: str):
        value = value.strip()
        if not value:
            return ''
        if value[0] in {'"', "'"} and (
            len(value) < 2 or value[-1] != value[0]
        ):
            raise ValueError("unterminated quoted value")
        if value[0] in '[{' and (
            len(value) < 2 or value[-1] != {'[': ']', '{': '}'}[value[0]]
        ):
            raise ValueError("unterminated collection value")
        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            return self._unescape_quoted(value[1:-1])
        if value.startswith('[') and value.endswith(']'):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [
                self._parse_property_value(part)
                for part in self._split_top_level(inner, ',')
            ]
        if value.startswith('{') and value.endswith('}'):
            inner = value[1:-1].strip()
            result = {}
            if not inner:
                return result
            for part in self._split_top_level(inner, ','):
                if ':' in part:
                    key, item_value = part.split(':', 1)
                elif '=' in part:
                    key, item_value = part.split('=', 1)
                else:
                    raise ValueError(
                        f"Invalid map entry {part.strip()!r}; expected key:value."
                    )
                parsed_key = self._strip_value_token(key.strip())
                if not parsed_key:
                    raise ValueError("Map keys cannot be empty.")
                if parsed_key in result:
                    raise ValueError(f"Duplicate map key '{parsed_key}'.")
                result[parsed_key] = self._parse_property_value(item_value)
            return result
        lowered = value.lower()
        if lowered in {'true', 'yes'}:
            return True
        if lowered in {'false', 'no'}:
            return False
        try:
            return float(value)
        except ValueError:
            return self._strip_value_token(value)

    @staticmethod
    def _unescape_quoted(value: str) -> str:
        result = []
        escape = False
        for char in value:
            if escape:
                result.append(char)
                escape = False
            elif char == '\\':
                escape = True
            else:
                result.append(char)
        if escape:
            result.append('\\')
        return ''.join(result)

    @staticmethod
    def _strip_value_token(value: str) -> str:
        value = value.strip()
        if (
            len(value) >= 2
            and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"))
        ):
            return PFDParser._unescape_quoted(value[1:-1])
        return value

    def _parse_pressure_bar(
        self,
        text: str,
        field_name: str = 'PSAT_MINIMUM_PRESSURE',
    ) -> float:
        match = re.fullmatch(
            r"\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            r"(?:\s*\[\s*([^\]]+?)\s*\]|\s+([A-Za-z]+))?"
            r"\s*",
            text,
        )
        if match is None:
            raise ParseError(
                f"Invalid {field_name} value on line "
                f"{self.current_line}: {text}"
            )
        value = float(match.group(1))
        unit = (match.group(2) or match.group(3) or "bar").strip().lower()
        try:
            pressure_bar = pressure_to_bar(value, unit, strict=True)
        except ValueError:
            raise ParseError(
                f"Unsupported {field_name} unit on line "
                f"{self.current_line}: {unit}"
            )
        if not math.isfinite(pressure_bar) or pressure_bar <= 0.0:
            raise ParseError(
                f"{field_name} must be positive on line "
                f"{self.current_line}: {text}"
            )
        return pressure_bar

    def _parse_temperature_K(self, text: str, field_name: str) -> float:
        match = re.fullmatch(
            r"\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            r"(?:\s*\[\s*([^\]]+?)\s*\]|\s+([A-Za-z°]+))?"
            r"\s*",
            text,
        )
        if match is None:
            raise ParseError(
                f"Invalid {field_name} value on line {self.current_line}: {text}"
            )
        value = float(match.group(1))
        unit = (match.group(2) or match.group(3) or 'K').strip()
        if unit.lower() not in {
            'k', 'kelvin', 'c', '°c', 'celsius', 'f', '°f', 'fahrenheit',
        }:
            raise ParseError(
                f"Unsupported {field_name} unit on line "
                f"{self.current_line}: {unit}"
            )
        temperature_K = temperature_to_kelvin(value, unit)
        if not math.isfinite(temperature_K) or temperature_K <= 0.0:
            raise ParseError(
                f"{field_name} must be positive on line "
                f"{self.current_line}: {text}"
            )
        return temperature_K

    @staticmethod
    def _split_top_level(text: str, delimiter: str) -> list[str]:
        parts = []
        start = 0
        stack = []
        quote = None
        escape = False
        pairs = {'[': ']', '{': '}', '(': ')'}
        closers = set(pairs.values())
        for index, char in enumerate(text):
            if escape:
                escape = False
                continue
            if quote:
                if char == '\\':
                    escape = True
                elif char == quote:
                    quote = None
                continue
            if char in {'"', "'"}:
                quote = char
                continue
            if char in pairs:
                stack.append(pairs[char])
                continue
            if char in closers:
                if not stack or stack[-1] != char:
                    raise ValueError(f"Unexpected closing delimiter '{char}'.")
                stack.pop()
                continue
            if char == delimiter and not stack:
                parts.append(text[start:index])
                start = index + 1
        if quote is not None:
            raise ValueError("Unterminated quoted value.")
        if stack:
            raise ValueError(
                f"Unterminated collection; expected closing delimiter '{stack[-1]}'."
            )
        parts.append(text[start:])
        return parts

    @staticmethod
    def _as_float_list(value) -> list[float]:
        if isinstance(value, list):
            return [float(item) for item in value]
        if isinstance(value, str):
            text = value.strip()
            if text.startswith('[') and text.endswith(']'):
                text = text[1:-1]
            return [float(item.strip()) for item in text.split(',') if item.strip()]
        return [float(value)]

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}
    
    def _parse_unifac_groups(self, unifac_str) -> dict:
        """
        Parse UNIFAC group specification string.
        
        Supports formats:
        - '1CH3+2CH2+1OH' or 'CH3+2CH2+OH'
        - '{CH3:1,CH2:2,OH:1}'
        """
        groups = {}

        if isinstance(unifac_str, dict):
            groups = {str(name): int(count) for name, count in unifac_str.items()}
            if not groups or any(not name for name in groups):
                raise ValueError("UNIFAC groups cannot be empty")
            return groups
        unifac_str = str(unifac_str).strip()
        if not unifac_str:
            raise ValueError("UNIFAC groups cannot be empty")
        
        # Check for dict-style format: {CH3:1,CH2:2,OH:1}
        if unifac_str.startswith('{') and unifac_str.endswith('}'):
            inner = unifac_str[1:-1]
            for part in self._split_top_level(inner, ','):
                if ':' not in part:
                    raise ValueError(
                        f"Invalid UNIFAC group entry {part.strip()!r}"
                    )
                name, count = part.split(':', 1)
                name = name.strip()
                if not name:
                    raise ValueError("UNIFAC group names cannot be empty")
                groups[name] = int(count.strip())
            return groups
        
        # Parse plus-separated format: 1CH3+2CH2+1OH
        for part in unifac_str.split('+'):
            part = part.strip()
            if not part:
                continue
            
            # Match optional count followed by group name
            match = re.fullmatch(r'(\d*)([A-Za-z0-9=_]+)', part)
            if not match:
                raise ValueError(f"Invalid UNIFAC group entry {part!r}")
            count_str, name = match.groups()
            count = int(count_str) if count_str else 1
            groups[name] = count
        
        return groups
    
    def _extract_float(self, text: str, pattern: str) -> Optional[float]:
        """Extract a float value using regex pattern"""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        return None

    def _next_top_level_line(self, start: int) -> int:
        """Return the next non-comment, nonblank unindented line."""
        i = start
        while i < len(self.lines):
            line = self.lines[i]
            stripped = self._strip_inline_comment(line).strip()
            if (
                stripped
                and not stripped.startswith('#')
                and not line.startswith((' ', '\t'))
            ):
                break
            i += 1
        return i

    @staticmethod
    def _normalized_port_alias(value: str) -> str:
        return re.sub(
            r'[^a-z0-9]+',
            '_',
            str(value).strip().lower(),
        ).strip('_')

    @classmethod
    def _closest_port_target(
        cls,
        alias: str,
        aliases: dict[str, str],
    ) -> tuple[Optional[str], float, tuple[str, ...]]:
        """Return one confident semantic target, or tied plausible targets."""
        probe = cls._normalized_port_alias(alias).replace('_', '')
        scores = {}
        for spelling, target in aliases.items():
            candidate = cls._normalized_port_alias(spelling).replace('_', '')
            score = SequenceMatcher(None, probe, candidate).ratio()
            scores[target] = max(scores.get(target, 0.0), score)
        if not scores:
            return None, 0.0, ()
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_target, best_score = ranked[0]
        if best_score < 0.68:
            return None, best_score, ()
        tied = tuple(
            target for target, score in ranked
            if best_score - score < 0.06
        )
        if len(tied) > 1:
            return None, best_score, tied
        return best_target, best_score, ()

    def _finalize_compact_unit_ports(self) -> None:
        """Resolve short endpoint aliases and materialize used compact ports."""
        numeric_by_unit = {}
        for stream in self.pfd.streams:
            line_number = self._stream_line_numbers.get(id(stream), 1)
            for reference, direction in (
                (stream.source, 'outlet'),
                (stream.destination, 'inlet'),
            ):
                if reference.is_feed or reference.is_product:
                    continue
                if reference.port_id is None or not reference.port_id.isdigit():
                    continue
                unit = self.pfd.get_unit(reference.unit_id)
                if unit is None:
                    continue
                numeric_by_unit.setdefault(id(unit), (unit, []) )[1].append((
                    int(reference.port_id), direction, reference, line_number,
                ))
        for unit, entries in numeric_by_unit.values():
            self._resolve_numeric_unit_ports(unit, entries)

        for stream in self.pfd.streams:
            line_number = self._stream_line_numbers.get(id(stream), 1)
            if not stream.source.is_feed:
                self._resolve_stream_port(
                    stream.source,
                    direction='outlet',
                    line_number=line_number,
                )
            if not stream.destination.is_product:
                self._resolve_stream_port(
                    stream.destination,
                    direction='inlet',
                    line_number=line_number,
                )

    @staticmethod
    def _unit_parameter(unit: Unit, *names: str) -> Optional[Parameter]:
        wanted = {name.casefold() for name in names}
        return next(
            (param for param in unit.params if param.name.casefold() in wanted),
            None,
        )

    def _numeric_clockwise_order_is_valid(
        self,
        unit: Unit,
        entries,
    ) -> bool:
        inlet_numbers = [number for number, direction, _, _ in entries if direction == 'inlet']
        outlet_numbers = [number for number, direction, _, _ in entries if direction == 'outlet']
        if inlet_numbers and outlet_numbers and max(inlet_numbers) >= min(outlet_numbers):
            self._record_error(
                f"Numeric ports on UNIT '{unit.id}' must increase clockwise from "
                "left-side inlets to right-side outlets; every numbered inlet "
                "must precede every numbered outlet.",
                min(entry[3] for entry in entries),
            )
            return False
        return True

    def _resolve_numeric_unit_ports(self, unit: Unit, entries) -> None:
        """Resolve a unit's numeric ports together using clockwise topology."""
        if not entries:
            return
        if (
            id(unit) not in self._compact_unit_objects
            or id(unit) in self._explicit_port_unit_objects
        ):
            # Explicit numeric IDs are declarations, not compact topology.
            return
        if any(number <= 0 for number, _, _, _ in entries):
            self._record_error(
                f"Numeric ports on UNIT '{unit.id}' start at 1.",
                min(entry[3] for entry in entries),
            )
            return
        directions_by_number = {}
        uses_by_number = {}
        for number, direction, _, line_number in entries:
            previous = directions_by_number.setdefault(number, direction)
            if previous != direction:
                self._record_error(
                    f"Numeric port {number} on UNIT '{unit.id}' is used as both "
                    "an inlet and an outlet.",
                    line_number,
                )
                return
            uses_by_number[number] = uses_by_number.get(number, 0) + 1
            if uses_by_number[number] > 1:
                self._record_error(
                    f"Numeric port {number} on UNIT '{unit.id}' is connected more "
                    "than once.",
                    line_number,
                )
                return
        if not self._numeric_clockwise_order_is_valid(unit, entries):
            return

        family = port_family_for_unit_type(unit.unit_type)
        if family in {'mixer', 'splitter', 'distillation'}:
            numbers = sorted(directions_by_number)
            if numbers != list(range(1, len(numbers) + 1)):
                self._record_error(
                    f"Numeric ports on {unit.unit_type} UNIT '{unit.id}' must be "
                    f"contiguous from 1; found {', '.join(map(str, numbers))}.",
                    min(entry[3] for entry in entries),
                )
                return
        layout = NUMERIC_PORT_LAYOUTS.get(family)
        if layout is not None:
            for number, direction, reference, line_number in entries:
                if number > len(layout):
                    self._record_error(
                        f"{unit.unit_type} UNIT '{unit.id}' has no clockwise "
                        f"numeric port {number}; its layout has {len(layout)} ports.",
                        line_number,
                    )
                    continue
                expected_direction, canonical = layout[number - 1]
                if direction != expected_direction:
                    self._record_error(
                        f"Numeric port {number} on {unit.unit_type} UNIT "
                        f"'{unit.id}' is a clockwise {expected_direction}, not an "
                        f"{direction}.",
                        line_number,
                    )
                    continue
                reference.port_id = canonical
            return

        if family == 'mixer':
            inlets = sorted(
                (entry for entry in entries if entry[1] == 'inlet'),
                key=lambda entry: entry[0],
            )
            outlets = [entry for entry in entries if entry[1] == 'outlet']
            if len({entry[0] for entry in outlets}) > 1:
                self._record_error(
                    f"Mixer UNIT '{unit.id}' has one outlet, but multiple numbered "
                    "outlet ports were used.",
                    min(entry[3] for entry in outlets),
                )
            for rank, (_, _, reference, _) in enumerate(inlets, start=1):
                reference.port_id = f'in{rank}'
            for _, _, reference, _ in outlets:
                reference.port_id = 'out'
            return

        if family == 'splitter':
            inlets = [entry for entry in entries if entry[1] == 'inlet']
            if len({entry[0] for entry in inlets}) > 1:
                self._record_error(
                    f"Splitter UNIT '{unit.id}' has one inlet, but multiple numbered "
                    "inlet ports were used.",
                    min(entry[3] for entry in inlets),
                )
            for _, _, reference, _ in inlets:
                reference.port_id = 'in'
            outlets = sorted(
                (entry for entry in entries if entry[1] == 'outlet'),
                key=lambda entry: entry[0],
            )
            outlet_names = []
            for rank, (_, _, reference, _) in enumerate(outlets, start=1):
                canonical = 'out' if rank == 1 else f'out{rank}'
                reference.port_id = canonical
                outlet_names.append(canonical)
            if (
                len(outlet_names) >= 2
                and self._unit_parameter(unit, 'outlets', 'outlet_names') is None
            ):
                unit.params.append(Parameter('outlets', ','.join(outlet_names)))
            return

        if family == 'distillation':
            self._resolve_numeric_distillation_ports(unit, entries)

    def _resolve_numeric_distillation_ports(self, unit: Unit, entries) -> None:
        inlets = sorted(
            (entry for entry in entries if entry[1] == 'inlet'),
            key=lambda entry: entry[0],
        )
        outlets = sorted(
            (entry for entry in entries if entry[1] == 'outlet'),
            key=lambda entry: entry[0],
        )
        canonical_type = canonical_unit_type(unit.unit_type)
        if len(inlets) > 1 and canonical_type != 'RigorousDistillation':
            self._record_error(
                f"{unit.unit_type} UNIT '{unit.id}' does not support multiple "
                "numbered feeds; use RigorousDistillation.",
                inlets[1][3],
            )
        elif len(inlets) == 1:
            inlets[0][2].port_id = 'feed'

        if len(inlets) > 1 and canonical_type == 'RigorousDistillation':
            for rank, (_, _, reference, _) in enumerate(inlets, start=1):
                reference.port_id = f'feed{rank}'
            feed_stages_parameter = self._unit_parameter(unit, 'feed_stages')
            if feed_stages_parameter is not None:
                self._canonicalize_numeric_feed_stages(
                    feed_stages_parameter,
                    len(inlets),
                )
            else:
                stages_param = self._unit_parameter(unit, 'N_stages', 'stages')
                try:
                    stage_count = int(float(stages_param.value)) if stages_param else 10
                except (TypeError, ValueError):
                    stage_count = 10
                feed_stages = []
                count = len(inlets)
                for rank, (_, _, reference, _) in enumerate(inlets, start=1):
                    # Stage 1 is the top: the first clockwise/lowest feed gets
                    # the largest stage number.
                    stage = round(stage_count - rank * stage_count / (count + 1))
                    stage = max(1, min(stage_count, stage))
                    feed_stages.append(f'{reference.port_id}:{stage}')
                unit.params.append(Parameter('feed_stages', ','.join(feed_stages)))

        if not outlets:
            return
        if len(outlets) == 1:
            outlets[0][2].port_id = 'distillate'
            return
        if len(outlets) == 2:
            outlets[0][2].port_id = 'distillate'
            outlets[1][2].port_id = 'bottoms'
            return

        condenser_param = self._unit_parameter(
            unit, 'condenser', 'condenser_type'
        )
        condenser = str(condenser_param.value).strip().lower() if condenser_param else ''
        mixed = condenser in {
            'mixed', 'mixed_distillate', 'two_phase', 'two-phase',
            'partial_liquid',
        }
        if mixed and canonical_type == 'RigorousDistillation':
            outlets[0][2].port_id = 'distillate_vapor'
            outlets[1][2].port_id = 'distillate_liquid'
            side_draw_entries = outlets[2:-1]
        else:
            outlets[0][2].port_id = 'distillate'
            side_draw_entries = outlets[1:-1]
        if side_draw_entries:
            self._bind_numeric_side_draws(unit, side_draw_entries)
        outlets[-1][2].port_id = 'bottoms'

    def _bind_numeric_side_draws(self, unit: Unit, entries) -> None:
        """Bind numeric intermediate column outlets to explicit side-draw specs."""
        parameter = self._unit_parameter(unit, 'side_draws')
        if parameter is None:
            self._record_error(
                f"Numeric intermediate outlet(s) on RigorousDistillation UNIT "
                f"'{unit.id}' require side_draws specifications with stage and "
                "flow or fraction; numeric position alone cannot infer those values.",
                min(entry[3] for entry in entries),
            )
            return
        raw_entries = [
            item.strip() for item in str(parameter.value).split(';')
            if item.strip()
        ]
        if len(raw_entries) != len(entries):
            self._record_error(
                f"RigorousDistillation UNIT '{unit.id}' has {len(entries)} numeric "
                f"side-draw outlet(s) but {len(raw_entries)} side_draws specification(s).",
                min(entry[3] for entry in entries),
            )
            return

        parsed = []
        for item in raw_entries:
            fields = []
            stage = None
            for token in item.split(','):
                token = token.strip()
                if not token:
                    continue
                delimiter = ':' if ':' in token else ('=' if '=' in token else None)
                if delimiter is None:
                    fields.append((token, None))
                    continue
                key, value = (part.strip() for part in token.split(delimiter, 1))
                if key.casefold() in {'stage', 'tray', 'stage_number'}:
                    try:
                        stage = int(value)
                    except ValueError:
                        stage = None
                fields.append((key, value))
            parsed.append((stage, fields))
        if all(stage is not None for stage, _ in parsed):
            # Stage 1 is the top, matching clockwise top-to-bottom outlet order.
            parsed.sort(key=lambda item: item[0])

        canonical_rows = []
        for rank, ((_, fields), (_, _, reference, _)) in enumerate(
            zip(parsed, entries), start=1
        ):
            canonical_port = f'side_draw{rank}'
            reference.port_id = canonical_port
            filtered = [
                (key, value) for key, value in fields
                if str(key).casefold() != 'port'
            ]
            filtered.append(('port', canonical_port))
            canonical_rows.append(','.join(
                str(key) if value is None else f'{key}:{value}'
                for key, value in filtered
            ))
        parameter.value = ';'.join(canonical_rows)

    @staticmethod
    def _canonicalize_numeric_feed_stages(
        parameter: Parameter,
        feed_count: int,
    ) -> None:
        rows = []
        for item in str(parameter.value).replace(';', ',').split(','):
            item = item.strip()
            if not item:
                continue
            delimiter = ':' if ':' in item else ('=' if '=' in item else None)
            if delimiter is None:
                rows.append(item)
                continue
            key, value = (part.strip() for part in item.split(delimiter, 1))
            if key.isdigit() and 1 <= int(key) <= feed_count:
                key = f'feed{int(key)}'
            rows.append(f'{key}:{value}')
        parameter.value = ','.join(rows)

    def _resolve_stream_port(
        self,
        reference: PortReference,
        *,
        direction: str,
        line_number: int,
    ) -> None:
        unit = self.pfd.get_unit(reference.unit_id)
        if unit is None or reference.port_id is None:
            return
        schema = port_schema_for_unit_type(unit.unit_type)
        if schema is None:
            return

        alias = self._normalized_port_alias(reference.port_id)
        aliases = schema['outlets' if direction == 'outlet' else 'inlets']
        opposite = schema['inlets' if direction == 'outlet' else 'outlets']
        direction_word = 'source' if direction == 'outlet' else 'destination'
        compact = (
            id(unit) in self._compact_unit_objects
            and id(unit) not in self._explicit_port_unit_objects
        )
        if not compact and unit.get_port(reference.port_id) is not None:
            return
        if alias in opposite and alias not in aliases:
            opposite_word = 'inlet-only' if direction == 'outlet' else 'outlet-only'
            self._record_error(
                f"Port alias '{reference.port_id}' on {unit.unit_type} UNIT "
                f"'{unit.id}' is {opposite_word} and cannot be used as a stream "
                f"{direction_word}.",
                line_number,
            )
            return

        canonical = aliases.get(alias)
        variable = unit_allows_variable_port(unit.unit_type, direction)
        if canonical is None:
            inferred, inferred_score, tied = self._closest_port_target(
                alias, aliases
            )
            opposite_inferred, opposite_score, _ = self._closest_port_target(
                alias, opposite
            )
            if (
                opposite_inferred is not None
                and opposite_score > inferred_score + 0.03
            ):
                opposite_word = (
                    'inlet-only' if direction == 'outlet' else 'outlet-only'
                )
                self._record_error(
                    f"Port name '{reference.port_id}' resembles the {opposite_word} "
                    f"port '{opposite_inferred}' on {unit.unit_type} UNIT "
                    f"'{unit.id}' and cannot be used as a stream {direction_word}.",
                    line_number,
                )
                return
            if (
                compact
                and variable
                and (
                    inferred is None
                    or inferred_score < 0.86
                    or any(char.isdigit() for char in alias)
                )
            ):
                canonical = reference.port_id
            elif inferred is not None:
                canonical = inferred
            elif tied:
                self._record_error(
                    f"Ambiguous {unit.unit_type} {direction} port name "
                    f"'{reference.port_id}'; it could refer to "
                    f"{', '.join(tied)}.",
                    line_number,
                )
                return
            elif compact and len(set(aliases.values())) == 1:
                canonical = next(iter(aliases.values()))
            elif compact:
                self._record_error(
                    unknown_name_message(
                        f"{unit.unit_type} {direction} port alias",
                        reference.port_id,
                        aliases,
                    ),
                    line_number,
                )
                return
            else:
                # Explicit PORTS remain authoritative and may intentionally use
                # names outside the compact vocabulary.
                return

        if compact:
            reference.port_id = canonical
            existing = unit.get_port(canonical)
            port_type_value = schema['types'].get(
                canonical,
                'outlet' if direction == 'outlet' else 'inlet',
            )
            port_type = PortType(port_type_value)
            if existing is None:
                unit.ports.append(Port(id=canonical, port_type=port_type))
            elif existing.port_type != port_type:
                self._record_error(
                    f"Compact UNIT '{unit.id}' port '{canonical}' is used as both "
                    "an inlet and an outlet.",
                    line_number,
                )
            return

        # Resolve a compact alias against an explicit port declaration. Prefer
        # declared spellings in the same alias family, then a unique semantic
        # port type (e.g. any one vapor_outlet for Flash.v).
        alias_family = {
            name for name, target in aliases.items() if target == canonical
        }
        named_matches = [
            port for port in unit.ports
            if self._normalized_port_alias(port.id) in alias_family
        ]
        if len(named_matches) == 1:
            reference.port_id = named_matches[0].id
            return
        expected_type_value = schema['types'].get(canonical)
        if expected_type_value is None:
            return
        expected_type = PortType(expected_type_value)
        typed_matches = [
            port for port in unit.ports if port.port_type == expected_type
        ]
        if len(typed_matches) == 1:
            reference.port_id = typed_matches[0].id
        elif len(named_matches) > 1 or len(typed_matches) > 1:
            self._record_error(
                f"Port alias '{reference.port_id}' is ambiguous for UNIT "
                f"'{unit.id}'; use an explicit declared port ID.",
                line_number,
            )
    
    def _parse_stream(self, start: int) -> int:
        """Parse a STREAM definition"""
        line = self._strip_inline_comment(self.lines[start]).strip()

        # Full form: STREAM name : source -> destination
        # Feed shorthand: STREAM name : -> destination
        # Product shorthand: STREAM name : source
        match = re.fullmatch(r'STREAM\s+(\S+)\s*:\s*(.*)', line)
        if not match:
            self._record_error(
                "Invalid STREAM header. Expected "
                "'STREAM id : source -> destination', 'STREAM id : -> destination', "
                "or 'STREAM id : source'.",
                start + 1,
            )
            return self._next_top_level_line(start + 1)

        stream_id = match.group(1)
        route = match.group(2).strip()
        if not route:
            self._record_error("STREAM route cannot be empty.", start + 1)
            return self._next_top_level_line(start + 1)
        if route.count('->') > 1:
            self._record_error(
                "Invalid STREAM route; expected at most one '->'.",
                start + 1,
            )
            return self._next_top_level_line(start + 1)
        if '->' in route:
            source_text, destination_text = (
                part.strip() for part in route.split('->', 1)
            )
            if not source_text and not destination_text:
                self._record_error(
                    "STREAM route must include a source or destination around '->'.",
                    start + 1,
                )
                return self._next_top_level_line(start + 1)
            source_text = source_text or 'FEED'
            destination_text = destination_text or 'PRODUCT'
        else:
            source_text = route
            destination_text = 'PRODUCT'
        try:
            source = PortReference.from_string(source_text)
            destination = PortReference.from_string(destination_text)
            if source.is_product:
                raise ParseError("PRODUCT cannot be a stream source.")
            if destination.is_feed:
                raise ParseError("FEED cannot be a stream destination.")
        except ParseError as error:
            self._record_error(error, start + 1)
            return self._next_top_level_line(start + 1)
        
        stream = Stream(id=stream_id, source=source, destination=destination)
        
        # Parse stream properties
        i = start + 1
        while i < len(self.lines):
            line = self.lines[i]
            stripped = self._strip_inline_comment(line).strip()
            
            # Check if we've left this stream's properties
            if stripped and not stripped.startswith('#') and not line.startswith(' ') and not line.startswith('\t'):
                break
            
            if stripped and not stripped.startswith('#'):
                # Check for composition
                comp_key = None
                if re.match(r'(x|mole_fractions?|molar_fractions?)\s*=', stripped, re.IGNORECASE):
                    comp_key = 'mole'
                    comp_match = re.match(r'(?:x|mole_fractions?|molar_fractions?)\s*=\s*(.+)', stripped, re.IGNORECASE)
                elif re.match(r'(w|mass_fractions?|weight_fractions?|wt_fractions?)\s*=', stripped, re.IGNORECASE):
                    comp_key = 'mass'
                    comp_match = re.match(r'(?:w|mass_fractions?|weight_fractions?|wt_fractions?)\s*=\s*(.+)', stripped, re.IGNORECASE)
                else:
                    comp_match = None
                if comp_match:
                    try:
                        stream.composition = self._parse_composition(
                            comp_match.group(1), basis=comp_key
                        )
                    except ValueError as error:
                        self._record_error(
                            f"Invalid composition for stream '{stream_id}': {error}",
                            i + 1,
                        )
                else:
                    # Regular property
                    prop = self._parse_property(stripped)
                    if prop and prop.name.upper() in _STREAM_PROPERTY_NAMES:
                        stream.properties.append(prop)
                    elif prop:
                        message = unknown_name_message(
                            'stream property',
                            prop.name,
                            sorted(_STREAM_PROPERTY_NAMES),
                        )
                        self._record_error(
                            f"{message} Stream '{stream_id}'.",
                            i + 1,
                        )
                    else:
                        self._record_error(
                            f"Invalid stream property for '{stream_id}'. Expected "
                            "'name = value [unit]'.",
                            i + 1,
                        )
            
            i += 1
        
        self.pfd.streams.append(stream)
        self._stream_line_numbers[id(stream)] = start + 1
        return i
    
    def _parse_composition(self, text: str, basis: str = 'mole') -> Composition:
        """Parse composition string like 'N2:0.25, H2:0.75'"""
        comp = Composition(basis=basis)
        parts = [p.strip() for p in text.split(',')]
        for part in parts:
            if not part or ':' not in part:
                raise ValueError(
                    f"malformed entry {part!r}; expected component:fraction"
                )
            symbol, frac = part.split(':', 1)
            symbol = symbol.strip()
            if not symbol:
                raise ValueError("component symbol cannot be empty")
            if symbol in comp.fractions:
                raise ValueError(f"duplicate component '{symbol}'")
            try:
                fraction = float(frac.strip())
            except ValueError as error:
                raise ValueError(
                    f"fraction for component '{symbol}' must be a number"
                ) from error
            if not math.isfinite(fraction) or fraction < 0.0:
                raise ValueError(
                    f"fraction for component '{symbol}' must be finite and nonnegative"
                )
            comp.fractions[symbol] = fraction
        return comp
    
    def _parse_property(self, text: str) -> Optional[StreamProperty]:
        """Parse a property line like 'T = 25 [C]'"""
        # Match: name = value [unit] or name = value
        match = re.fullmatch(r'(\w+)\s*=\s*(.+?)(?:\s*\[([^\]]+)\])?\s*', text)
        if match:
            if any(char in match.group(2) for char in '[]') or (
                match.group(3) is not None and '[' in match.group(3)
            ):
                return None
            return StreamProperty(
                name=match.group(1),
                value=match.group(2).strip(),
                unit=match.group(3)
            )
        return None
    
    def _parse_unit(self, start: int) -> int:
        """Parse a UNIT definition"""
        line = self._strip_inline_comment(self.lines[start]).strip()

        # Full form: UNIT id, followed by TYPE/PORTS/PARAMS sections.
        # Compact form: UNIT id : type, followed directly by parameters.
        match = re.fullmatch(r'UNIT\s+([^\s:]+)(?:\s*:\s*(\S+))?\s*', line)
        if not match:
            self._record_error(
                "Invalid UNIT header. Expected 'UNIT id' or 'UNIT id : type'.",
                start + 1,
            )
            return self._next_top_level_line(start + 1)

        compact_type = match.group(2)
        compact_canonical_type = (
            canonical_unit_type(compact_type) if compact_type is not None else None
        )
        unit = Unit(
            id=match.group(1),
            unit_type=compact_canonical_type or compact_type or '',
        )
        if compact_type is not None:
            self._compact_unit_objects.add(id(unit))
            if canonical_unit_type(compact_type) is None:
                self._record_error(
                    unknown_name_message(
                        'unit type', compact_type, UNIT_TYPE_ALIASES
                    ),
                    start + 1,
                )
        
        i = start + 1
        current_section = None

        def validate_thermo_scope_parameter(param: Optional[Parameter], line_number: int) -> None:
            if param is None or param.name.lower() != 'thermo_scope':
                return
            scope = str(param.value).strip()
            if scope == 'global':
                return
            if self.pfd.get_thermo_scope(scope) is None:
                self._record_error(
                    f"Unit '{unit.id}' thermo_scope '{scope}' must be declared "
                    "earlier in THERMO_SCOPES.",
                    line_number,
                )
        
        while i < len(self.lines):
            line = self.lines[i]
            stripped = self._strip_inline_comment(line).strip()
            
            # Check if we've left this unit
            if stripped and not stripped.startswith('#'):
                if not line.startswith(' ') and not line.startswith('\t'):
                    break
            
            if stripped and not stripped.startswith('#'):
                # Section headers
                if stripped.startswith('TYPE:'):
                    unit_type = stripped[5:].strip()
                    if compact_type is not None:
                        self._record_error(
                            f"UNIT '{unit.id}' already declares type "
                            f"'{compact_type}' in its compact header.",
                            i + 1,
                        )
                    elif unit_type:
                        canonical_type = canonical_unit_type(unit_type)
                        unit.unit_type = canonical_type or unit_type
                        if canonical_type is None:
                            self._record_error(
                                unknown_name_message(
                                    'unit type', unit_type, UNIT_TYPE_ALIASES
                                ),
                                i + 1,
                            )
                    else:
                        self._record_error(
                            f"UNIT '{unit.id}' TYPE cannot be empty.",
                            i + 1,
                        )
                elif stripped == 'PORTS:':
                    current_section = 'ports'
                    self._explicit_port_unit_objects.add(id(unit))
                elif stripped == 'PARAMS:':
                    current_section = 'params'
                elif stripped == 'REACTIONS:':
                    current_section = 'reactions'
                elif current_section == 'ports':
                    try:
                        port = self._parse_port(stripped)
                        unit.ports.append(port)
                    except ValueError as error:
                        self._record_error(
                            f"Invalid port in UNIT '{unit.id}': {error}",
                            i + 1,
                        )
                elif current_section == 'params':
                    param = self._parse_parameter(stripped)
                    if param:
                        validate_thermo_scope_parameter(param, i + 1)
                        unit.params.append(param)
                    else:
                        self._record_error(
                            f"Invalid parameter in UNIT '{unit.id}'. Expected "
                            "'name = value [unit]'.",
                            i + 1,
                        )
                elif current_section == 'reactions':
                    try:
                        rxn = self._parse_reaction(stripped)
                        unit.reactions.append(rxn)
                    except ValueError as error:
                        self._record_error(
                            f"Invalid reaction in UNIT '{unit.id}': {error}",
                            i + 1,
                        )
                elif compact_type is not None:
                    param = self._parse_parameter(stripped)
                    if param:
                        validate_thermo_scope_parameter(param, i + 1)
                        unit.params.append(param)
                    else:
                        self._record_error(
                            f"Invalid compact parameter in UNIT '{unit.id}'. "
                            "Expected 'name = value [unit]'.",
                            i + 1,
                        )
                else:
                    token = stripped.split(':', 1)[0]
                    self._record_error(
                        unknown_name_message(
                            f"UNIT '{unit.id}' field",
                            token,
                            ('TYPE', 'PORTS', 'PARAMS', 'REACTIONS'),
                        ),
                        i + 1,
                    )
            
            i += 1
        
        self.pfd.units.append(unit)
        return i
    
    def _parse_port(self, text: str) -> Port:
        """Parse a port line like 'in : inlet'"""
        match = re.fullmatch(r'(\w+)\s*:\s*(\w+)\s*', text)
        if not match:
            raise ValueError("expected 'port_id : port_type'")
        port_id = match.group(1)
        port_type_str = match.group(2)
        try:
            port_type = PortType(port_type_str)
        except ValueError as error:
            raise ValueError(
                unknown_name_message(
                    'port type', port_type_str, (item.value for item in PortType)
                )
            ) from error
        return Port(id=port_id, port_type=port_type)
    
    def _parse_parameter(self, text: str) -> Optional[Parameter]:
        """Parse a parameter line like 'T = 450 [C]'"""
        match = re.fullmatch(r'(\w+)\s*=\s*(.+?)(?:\s*\[([^\]]+)\])?\s*', text)
        if match:
            if any(char in match.group(2) for char in '[]') or (
                match.group(3) is not None and '[' in match.group(3)
            ):
                return None
            return Parameter(
                name=match.group(1),
                value=match.group(2).strip(),
                unit=match.group(3)
            )
        return None
    
    def _parse_reaction(self, text: str) -> Optional[Reaction]:
        """Parse a reaction line like 'A + B -> C | conversion=0.95'"""
        if text.startswith('@'):
            reference = text[1:].strip()
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*', reference):
                raise ValueError(
                    "reaction references must use @ followed by a catalog name"
                )
            return Reaction(equation='', reference=reference)
        if '|' in text:
            equation, params_str = text.split('|', 1)
            if not equation.strip():
                raise ValueError("reaction equation cannot be empty")
            params = {}
            for part in params_str.split(','):
                if '=' not in part:
                    raise ValueError(
                        f"malformed parameter {part.strip()!r}; expected key=value"
                    )
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                if not key or not value:
                    raise ValueError("reaction parameter keys and values cannot be empty")
                params[key] = value
            return Reaction(equation=equation.strip(), parameters=params)
        if not text.strip():
            raise ValueError("reaction equation cannot be empty")
        return Reaction(equation=text.strip())


class PFDValidator:
    """Validator for ProcessFlowDiagram objects"""
    
    def __init__(self, pfd: ProcessFlowDiagram):
        self.pfd = pfd
        self.errors: list[str] = []
        self.warnings: list[str] = []
    
    def validate(self) -> tuple[list[str], list[str]]:
        """Run all validations, return (errors, warnings)"""
        self.errors = []
        self.warnings = []
        
        self._validate_metadata()
        self._validate_components()
        self._validate_reaction_definitions()
        self._validate_units()
        self._validate_streams()
        self._validate_connections()
        
        return self.errors, self.warnings

    def _validate_metadata(self):
        """Validate method-specific process-level numerical controls."""
        try:
            method = normalize_recycle_method(self.pfd.metadata.recycle_method)
            normalize_recycle_options(
                method,
                self.pfd.metadata.recycle_options,
            )
        except ValueError as error:
            self.errors.append(str(error))

        scope_names = set()
        scope_by_name = {}
        for scope in self.pfd.thermo_scopes:
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]*', scope.name):
                self.errors.append(
                    f"Invalid thermodynamic scope name '{scope.name}'."
                )
                continue
            if scope.name.lower() == 'global':
                self.errors.append(
                    "THERMO_SCOPES cannot redefine reserved scope 'global'."
                )
            if scope.name in scope_names:
                self.errors.append(
                    f"Duplicate thermodynamic scope: {scope.name}"
                )
            scope_names.add(scope.name)
            scope_by_name[scope.name] = scope
            if scope.method not in _SUPPORTED_THERMO_SCOPE_METHODS:
                self.errors.append(
                    unsupported_thermo_method_message(
                        scope.method,
                        _SUPPORTED_THERMO_SCOPE_METHODS,
                        {},
                    )
                )

        for scope in self.pfd.thermo_scopes:
            if not scope.inherit or scope.inherit.lower() == 'global':
                continue
            if scope.inherit not in scope_by_name:
                self.errors.append(
                    f"Thermodynamic scope '{scope.name}' inherits unknown scope "
                    f"'{scope.inherit}'."
                )

        visiting = set()
        visited = set()

        def visit_scope(name: str, path: list[str]) -> None:
            if name in visited or name not in scope_by_name:
                return
            if name in visiting:
                cycle_start = path.index(name)
                cycle = path[cycle_start:] + [name]
                self.errors.append(
                    "Thermodynamic scope inheritance cycle: "
                    + " -> ".join(cycle)
                )
                return
            visiting.add(name)
            scope = scope_by_name[name]
            parent = scope.inherit
            if parent and parent.lower() != 'global':
                visit_scope(parent, path + [name])
            visiting.remove(name)
            visited.add(name)

        for name in scope_by_name:
            visit_scope(name, [])

        valid_scope_names = scope_names | {'global'}
        for collection_name, records in (
            ('INTERACTION_PARAMETERS', self.pfd.interaction_parameters),
            ('INTERACTION_ESTIMATION', self.pfd.interaction_estimation),
        ):
            for record in records:
                if record.scope and record.scope not in valid_scope_names:
                    self.errors.append(
                        f"{collection_name} references unknown thermodynamic "
                        f"scope '{record.scope}'."
                    )
    
    def _validate_components(self):
        """Validate component definitions"""
        symbols = set()
        for comp in self.pfd.components:
            if comp.symbol in symbols:
                self.errors.append(f"Duplicate component symbol: {comp.symbol}")
            symbols.add(comp.symbol)
            
            if comp.molecular_weight is not None and comp.molecular_weight <= 0:
                self.errors.append(f"Invalid molecular weight for {comp.symbol}: {comp.molecular_weight}")
            if comp.phase_behavior is not None:
                try:
                    normalize_phase_behavior(comp.phase_behavior)
                except ValueError as error:
                    self.errors.append(str(error))
            if comp.particle_diameter is not None and comp.particle_diameter <= 0:
                self.errors.append(
                    f"Invalid particle_diameter for {comp.symbol}: {comp.particle_diameter}"
                )

            if (
                comp.particle_sphericity is not None
                and not 0.0 < comp.particle_sphericity <= 1.0
            ):
                self.errors.append(
                    f"Invalid particle_sphericity for {comp.symbol}: {comp.particle_sphericity}"
                )
            if (
                (comp.particle_diameter is not None or comp.particle_sphericity is not None)
                and normalize_phase_behavior(comp.phase_behavior) != 'permanent_solid'
            ):
                self.errors.append(
                    f"Particle defaults for {comp.symbol} require "
                    "phase_behavior=permanent_solid"
                )
            for label, value in (
                ('Cp_solid', comp.Cp_solid),
                ('rho_solid', comp.rho_solid),
                ('Vm_solid', comp.Vm_solid),
            ):
                if value is not None and value <= 0:
                    self.errors.append(f"Invalid {label} for {comp.symbol}: {value}")
            if (
                comp.molecular_weight is not None
                and comp.rho_solid is not None
                and comp.Vm_solid is not None
            ):
                implied = comp.molecular_weight / comp.Vm_solid
                relative = abs(implied / comp.rho_solid - 1.0)
                if relative > 0.02:
                    self.errors.append(
                        f"Inconsistent solid density and molar volume for {comp.symbol}: "
                        f"MW/Vm_solid={implied:g} kg/m3 versus rho_solid={comp.rho_solid:g} kg/m3"
                    )

            cpg = (comp.property_correlations or {}).get('Cpg')
            if isinstance(cpg, dict) and (
                cpg.get('Tmin_K') is None or cpg.get('Tmax_K') is None
            ):
                self.warnings.append(
                    f"{comp.symbol}.Cpg has no complete temperature range; "
                    "defaulting to 273.15-1500 K"
                )
            cpl = (comp.property_correlations or {}).get('Cpl')
            if isinstance(cpl, dict) and (
                cpl.get('Tmin_K') is None or cpl.get('Tmax_K') is None
            ):
                self.warnings.append(
                    f"{comp.symbol}.Cpl has no complete temperature range; "
                    "defaulting to 273.15-1500 K"
                )
            cps = (comp.property_correlations or {}).get('Cps')
            if isinstance(cps, dict) and (
                cps.get('Tmin_K') is None or cps.get('Tmax_K') is None
            ):
                self.warnings.append(
                    f"{comp.symbol}.Cps has no complete temperature range; "
                    "defaulting to 273.15-1500 K"
                )
            if comp.Cp_coeffs is not None:
                self.warnings.append(
                    f"{comp.symbol}.Cp_coeffs has no temperature-range fields; "
                    "defaulting to 273.15-1500 K"
                )

    def _validate_reaction_definitions(self):
        """Validate catalog identity and equation structure independent of unit use."""
        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from .reaction_models import (
                        ReactionDefinitionError,
                        parse_reaction_equation,
                        validate_reaction_balance,
                    )
        else:
            from reaction_models import (
                        ReactionDefinitionError,
                        parse_reaction_equation,
                        validate_reaction_balance,
                    )

        names = set()
        component_symbols = [component.symbol for component in self.pfd.components]
        component_metadata = {
            component.symbol: component for component in self.pfd.components
        }
        for definition in self.pfd.reaction_definitions:
            if definition.name in names:
                self.errors.append(
                    f"Duplicate reaction definition: {definition.name}"
                )
            names.add(definition.name)
            try:
                parsed = parse_reaction_equation(
                    definition.reaction.equation,
                    component_symbols,
                )
                if all(
                    component_metadata[component].formula
                    for component in parsed.components
                ):
                    self.warnings.extend(validate_reaction_balance(
                        parsed,
                        component_metadata,
                    ))
            except ReactionDefinitionError as error:
                self.errors.append(
                    f"Reaction definition {definition.name}: {error}"
                )
    
    def _validate_units(self):
        """Validate unit definitions"""
        unit_ids = set()
        component_symbols = [component.symbol for component in self.pfd.components]
        component_metadata = {
            component.symbol: component for component in self.pfd.components
        }
        valid_thermo_scopes = {
            scope.name for scope in self.pfd.thermo_scopes
        } | {'global'}
        for unit in self.pfd.units:
            if unit.id in unit_ids:
                self.errors.append(f"Duplicate unit ID: {unit.id}")
            unit_ids.add(unit.id)

            scope_params = [
                param for param in unit.params
                if param.name.lower() == 'thermo_scope'
            ]
            if len(scope_params) > 1:
                self.errors.append(
                    f"Unit {unit.id} declares thermo_scope more than once"
                )
            elif scope_params:
                scope_name = str(scope_params[0].value).strip()
                if scope_name not in valid_thermo_scopes:
                    self.errors.append(
                        f"Unit {unit.id} references unknown thermodynamic scope "
                        f"'{scope_name}'"
                    )
            
            if not unit.unit_type:
                self.errors.append(f"Unit {unit.id} has no TYPE specified")
            
            if not unit.ports:
                self.warnings.append(f"Unit {unit.id} has no ports defined")
            
            # Check for duplicate port IDs within unit
            port_ids = set()
            for port in unit.ports:
                if port.id in port_ids:
                    self.errors.append(f"Duplicate port ID '{port.id}' in unit {unit.id}")
                port_ids.add(port.id)

            if unit.reactions:
                if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                    from .reaction_models import (
                                        ReactionDefinitionError,
                                        conversion_reaction_from_mapping,
                                        equilibrium_reaction_from_mapping,
                                        parse_reaction_equation,
                                        validate_reaction_balance,
                                    )
                else:
                    from reaction_models import (
                                        ReactionDefinitionError,
                                        conversion_reaction_from_mapping,
                                        equilibrium_reaction_from_mapping,
                                        parse_reaction_equation,
                                        validate_reaction_balance,
                                    )

                for index, reaction in enumerate(unit.reactions, start=1):
                    if reaction.reference and unit.unit_type not in {
                        UnitType.REACTOR.value,
                        UnitType.EQUILIBRIUM_REACTOR.value,
                        UnitType.CSTR.value,
                        UnitType.BATCH_REACTOR.value,
                        UnitType.PFR.value,
                        UnitType.PACKED_BED_REACTOR.value,
                    }:
                        self.errors.append(
                            f"Unit {unit.id} cannot load reaction reference "
                            f"'@{reaction.reference}' for type {unit.unit_type}"
                        )
                        continue
                    try:
                        resolved_reaction = self.pfd.resolve_reaction(reaction)
                    except ValueError as error:
                        self.errors.append(f"Unit {unit.id} reaction {index}: {error}")
                        continue
                    definition = {
                        'equation': resolved_reaction.equation,
                        **resolved_reaction.parameters,
                    }
                    try:
                        parsed = parse_reaction_equation(
                            resolved_reaction.equation,
                            component_symbols,
                        )
                        explicit_metadata = (
                            component_metadata
                            if all(
                                component_metadata[component].formula
                                for component in parsed.components
                            )
                            else None
                        )
                        if unit.unit_type == UnitType.REACTOR.value:
                            specification = conversion_reaction_from_mapping(
                                definition,
                                component_symbols,
                                explicit_metadata,
                            )
                            self.warnings.extend(specification.validation_warnings)
                        elif unit.unit_type == UnitType.EQUILIBRIUM_REACTOR.value:
                            specification = equilibrium_reaction_from_mapping(
                                definition,
                                component_symbols,
                                explicit_metadata,
                            )
                            self.warnings.extend(specification.validation_warnings)
                        elif unit.unit_type in {
                            UnitType.CSTR.value,
                            UnitType.BATCH_REACTOR.value,
                            UnitType.PFR.value,
                            UnitType.PACKED_BED_REACTOR.value,
                        }:
                            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                                from .kinetic_models import (
                                                                kinetic_reaction_from_mapping,
                                                            )
                            else:
                                from kinetic_models import (
                                                                kinetic_reaction_from_mapping,
                                                            )

                            specification = kinetic_reaction_from_mapping(
                                definition,
                                component_symbols,
                                explicit_metadata,
                            )
                            expected_rate_basis = (
                                'catalyst_mass'
                                if unit.unit_type
                                == UnitType.PACKED_BED_REACTOR.value
                                else 'fluid_volume'
                            )
                            if specification.rate_output_basis != expected_rate_basis:
                                expected_label = (
                                    'catalyst-mass'
                                    if expected_rate_basis == 'catalyst_mass'
                                    else 'fluid-volume'
                                )
                                raise ReactionDefinitionError(
                                    f"{unit.unit_type} requires {expected_label} "
                                    f"rate units, not {specification.rate_unit}"
                                )
                            self.warnings.extend(specification.validation_warnings)
                        else:
                            if explicit_metadata is not None:
                                self.warnings.extend(validate_reaction_balance(
                                    parsed,
                                    explicit_metadata,
                                ))
                    except ReactionDefinitionError as error:
                        self.errors.append(
                            f"Unit {unit.id} reaction {index}: {error}"
                        )
    
    def _validate_streams(self):
        """Validate stream definitions"""
        stream_ids = set()
        for stream in self.pfd.streams:
            if stream.id in stream_ids:
                self.errors.append(f"Duplicate stream ID: {stream.id}")
            stream_ids.add(stream.id)
            
            # Validate composition references
            if stream.composition:
                for symbol in stream.composition.fractions:
                    if not self.pfd.get_component(symbol):
                        self.errors.append(
                            f"Stream {stream.id} references undefined component: {symbol}"
                        )
                
                total = sum(stream.composition.fractions.values())
                if abs(total - 1.0) > 0.001:
                    self.warnings.append(
                        f"Stream {stream.id} composition sums to {total:.4f}, not 1.0"
                    )
    
    def _validate_connections(self):
        """Validate that all stream connections reference valid ports"""
        source_connections: dict[tuple[str, str], str] = {}
        destination_connections: dict[tuple[str, str], str] = {}
        for stream in self.pfd.streams:
            # Validate source
            if not stream.source.is_feed:
                unit = self.pfd.get_unit(stream.source.unit_id)
                if not unit:
                    self.errors.append(
                        f"Stream {stream.id} source references undefined unit: {stream.source.unit_id}"
                    )
                else:
                    port = unit.get_port(stream.source.port_id)
                    if not port:
                        self.errors.append(
                            f"Stream {stream.id} source references undefined port: "
                            f"{stream.source.unit_id}.{stream.source.port_id}"
                        )
                    elif port.port_type in _INLET_PORT_TYPES:
                        self.errors.append(
                            f"Stream {stream.id} source port is not an outlet: "
                            f"{stream.source.unit_id}.{stream.source.port_id}"
                        )
                    else:
                        key = (stream.source.unit_id, stream.source.port_id)
                        previous = source_connections.get(key)
                        if previous is not None:
                            self.errors.append(
                                f"Streams {previous} and {stream.id} both leave "
                                f"material outlet {key[0]}.{key[1]}; use a Splitter"
                            )
                        else:
                            source_connections[key] = stream.id
            
            # Validate destination
            if not stream.destination.is_product:
                unit = self.pfd.get_unit(stream.destination.unit_id)
                if not unit:
                    self.errors.append(
                        f"Stream {stream.id} destination references undefined unit: {stream.destination.unit_id}"
                    )
                else:
                    port = unit.get_port(stream.destination.port_id)
                    if not port:
                        self.errors.append(
                            f"Stream {stream.id} destination references undefined port: "
                            f"{stream.destination.unit_id}.{stream.destination.port_id}"
                        )
                    elif port.port_type not in _INLET_PORT_TYPES:
                        self.errors.append(
                            f"Stream {stream.id} destination port is not an inlet: "
                            f"{stream.destination.unit_id}.{stream.destination.port_id}"
                        )
                    else:
                        key = (
                            stream.destination.unit_id,
                            stream.destination.port_id,
                        )
                        previous = destination_connections.get(key)
                        if previous is not None:
                            self.errors.append(
                                f"Streams {previous} and {stream.id} both enter "
                                f"material inlet {key[0]}.{key[1]}; use distinct "
                                "inlet ports or a Mixer"
                            )
                        else:
                            destination_connections[key] = stream.id


def parse_pfd(text: str) -> ProcessFlowDiagram:
    """Convenience function to parse PFD text"""
    parser = PFDParser()
    return parser.parse(text)


def validate_pfd(pfd: ProcessFlowDiagram) -> tuple[list[str], list[str]]:
    """Convenience function to validate a PFD"""
    validator = PFDValidator(pfd)
    return validator.validate()


def parse_and_validate(text: str) -> tuple[ProcessFlowDiagram, list[str], list[str]]:
    """Parse and validate PFD text, returning (pfd, errors, warnings)"""
    pfd = parse_pfd(text)
    errors, warnings = validate_pfd(pfd)
    return pfd, errors, warnings
