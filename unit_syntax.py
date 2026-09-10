"""Shared unit-type registration and compact PFD port syntax."""

from __future__ import annotations


# Public unit spellings mapped to the canonical implementation name. Keep this
# lightweight so pfd_parser can use it without importing the unit-operation
# implementations and their thermodynamic dependencies.
UNIT_TYPE_ALIASES = {
    'Mixer': 'Mixer',
    'Splitter': 'Splitter',
    'Pump': 'Pump',
    'Compressor': 'Compressor',
    'Expander': 'Expander',
    'Valve': 'Valve',
    'Pipe': 'Pipe',
    'Heater': 'Heater',
    'Cooler': 'Cooler',
    'HeatExchanger': 'HeatExchanger',
    'Flash': 'Flash',
    'ShortcutDistillation': 'ShortcutDistillation',
    'McCabeThieleDistillation': 'McCabeThieleDistillation',
    'CMODistillation': 'CMODistillation',
    'RigorousDistillation': 'RigorousDistillation',
    'Flash3': 'Flash3',
    'ThreePhaseFlash': 'Flash3',
    'VLLEFlash': 'Flash3',
    'Decanter': 'Decanter',
    'FlashLLE': 'Decanter',
    'LLSeparator': 'Decanter',
    'Settler': 'Decanter',
    'MolecularSieveDryer': 'MolecularSieveDryer',
    'Dryer': 'MolecularSieveDryer',
    'ShortcutExtractor': 'ShortcutExtractor',
    'Extractor': 'ShortcutExtractor',
    'LiquidLiquidExtractor': 'ShortcutExtractor',
    'LLE': 'ShortcutExtractor',
    'RigorousExtractor': 'RigorousExtractor',
    'RigorousLiquidLiquidExtractor': 'RigorousExtractor',
    'Absorber': 'Absorber',
    'AbsorptionColumn': 'Absorber',
    'RigorousAbsorber': 'RigorousAbsorber',
    'RigorousAbsorptionColumn': 'RigorousAbsorber',
    'Stripper': 'Stripper',
    'StrippingColumn': 'Stripper',
    'RigorousStripper': 'RigorousStripper',
    'RigorousStrippingColumn': 'RigorousStripper',
    'Reactor': 'Reactor',
    'EquilibriumReactor': 'EquilibriumReactor',
    'CSTR': 'CSTR',
    'KineticsCSTR': 'CSTR',
    'BatchReactor': 'BatchReactor',
    'KineticsBatch': 'BatchReactor',
    'PFR': 'PFR',
    'KineticsPFR': 'PFR',
    'PlugFlowReactor': 'PFR',
    'PackedBedReactor': 'PackedBedReactor',
    'PBR': 'PackedBedReactor',
    'KineticsPackedBed': 'PackedBedReactor',
    'Crystallizer': 'Crystallizer',
    'LayerCrystallizer': 'LayerCrystallizer',
    'Filter': 'Filter',
}


UNIT_PORT_FAMILIES = {
    'Mixer': 'mixer',
    'Splitter': 'splitter',
    'Pump': 'pump',
    'Compressor': 'compressor',
    'Expander': 'expander',
    'Valve': 'valve',
    'Pipe': 'pipe',
    'Heater': 'heater',
    'Cooler': 'cooler',
    'Reactor': 'reactor',
    'EquilibriumReactor': 'reactor',
    'CSTR': 'reactor',
    'BatchReactor': 'batch_reactor',
    'PFR': 'reactor',
    'PackedBedReactor': 'reactor',
    'Crystallizer': 'crystallizer',
    'LayerCrystallizer': 'layer_crystallizer',
    'Filter': 'filter',
    'HeatExchanger': 'heat_exchanger',
    'Flash': 'flash',
    'Flash3': 'flash3',
    'Decanter': 'decanter',
    'ShortcutDistillation': 'distillation',
    'McCabeThieleDistillation': 'distillation',
    'CMODistillation': 'distillation',
    'RigorousDistillation': 'distillation',
    'MolecularSieveDryer': 'dryer',
    'ShortcutExtractor': 'extractor',
    'RigorousExtractor': 'extractor',
    'Absorber': 'absorber',
    'RigorousAbsorber': 'absorber',
    'Stripper': 'stripper',
    'RigorousStripper': 'stripper',
}


VARIABLE_UNIT_PORT_DIRECTIONS = frozenset({
    ('Mixer', 'inlet'),
    ('Splitter', 'outlet'),
    ('RigorousDistillation', 'inlet'),
    ('RigorousDistillation', 'outlet'),
    ('RigorousAbsorber', 'inlet'),
    ('RigorousStripper', 'inlet'),
    ('BatchReactor', 'inlet'),
})


# Clockwise numeric layouts assume a conventional left-to-right flowsheet.
# Each entry describes fixed numbered positions beginning at 1. Mixer,
# splitter, and rigorous-distillation layouts depend on connection counts and
# are completed by pfd_parser's collective numeric-port pass.
NUMERIC_PORT_LAYOUTS = {
    'filter': (
        ('inlet', 'in'), ('inlet', 'wash'),
        ('outlet', 'filtrate'), ('outlet', 'cake'),
    ),
    'pump': (
        ('inlet', 'in'), ('outlet', 'out'),
    ),
    'compressor': (
        ('inlet', 'in'), ('outlet', 'out'),
    ),
    'expander': (
        ('inlet', 'in'), ('outlet', 'out'),
    ),
    'valve': (
        ('inlet', 'in'), ('outlet', 'out'),
    ),
    'pipe': (
        ('inlet', 'in'), ('outlet', 'out'),
    ),
    'heater': (
        ('inlet', 'in'), ('outlet', 'out'),
    ),
    'cooler': (
        ('inlet', 'in'), ('outlet', 'out'),
    ),
    'reactor': (
        ('inlet', 'in'), ('outlet', 'out'),
    ),
    # Lower-left cold inlet -> upper-left hot inlet -> upper-right hot outlet
    # -> lower-right cold outlet.
    'heat_exchanger': (
        ('inlet', 'cold_in'), ('inlet', 'hot_in'),
        ('outlet', 'hot_out'), ('outlet', 'cold_out'),
    ),
    'flash': (
        ('inlet', 'in'), ('outlet', 'vapor_out'),
        ('outlet', 'liquid_out'),
    ),
    'flash3': (
        ('inlet', 'in'), ('outlet', 'vapor_out'),
        ('outlet', 'liquid1_out'), ('outlet', 'liquid2_out'),
    ),
    'decanter': (
        ('inlet', 'in'), ('outlet', 'light'), ('outlet', 'heavy'),
    ),
    'dryer': (
        ('inlet', 'feed'), ('outlet', 'product'),
        ('outlet', 'adsorbate'),
    ),
    'extractor': (
        ('inlet', 'feed'), ('inlet', 'solvent'),
        ('outlet', 'extract'), ('outlet', 'raffinate'),
    ),
    'absorber': (
        ('inlet', 'gas'), ('inlet', 'liquid'),
        ('outlet', 'gas_out'), ('outlet', 'liquid_out'),
    ),
    'stripper': (
        ('inlet', 'gas'), ('inlet', 'liquid'),
        ('outlet', 'gas_out'), ('outlet', 'liquid_out'),
    ),
}


def _names(canonical: str, *aliases: str) -> dict[str, str]:
    result = {}
    for name in (canonical, *aliases):
        normalized = str(name).strip().lower().replace('-', '_').replace(' ', '_')
        result[normalized] = canonical
        collapsed = normalized.replace('_', '')
        if collapsed:
            result[collapsed] = canonical
    return result


_COMMON_IN = _names(
    'in', 'i', 'inlet', 'input', 'feed', 'f', 'feed_in', 'process_in',
    'material_in', 'stream_in', 'upstream',
)
_COMMON_OUT = _names(
    'out', 'o', 'outlet', 'output', 'product', 'p', 'product_out',
    'process_out', 'material_out', 'stream_out', 'downstream',
)


def _single_schema(inlet_aliases=(), outlet_aliases=()):
    return {
        'inlets': {**_COMMON_IN, **_names('in', *inlet_aliases)},
        'outlets': {**_COMMON_OUT, **_names('out', *outlet_aliases)},
        'types': {'in': 'inlet', 'out': 'outlet'},
    }


_SPLITTER_OUTLETS = dict(_COMMON_OUT)
for _index in range(1, 11):
    _canonical = 'out' if _index == 1 else f'out{_index}'
    _SPLITTER_OUTLETS.update(_names(
        _canonical,
        f'o{_index}', f'outlet{_index}', f'output{_index}',
        f'product{_index}', f'p{_index}', f'branch{_index}', f'split{_index}',
    ))


# Port types are stored as their PFD string values to keep this module
# independent of pfd_parser.PortType.
PORT_FAMILY_SCHEMAS = {
    'filter': {
        'inlets': {**_COMMON_IN, **_names('wash', 'wash_liquid', 'wash_in')},
        'outlets': {
            **_names('cake', 'wet_cake', 'solids'),
            **_names('filtrate', 'liquor', 'liquid_out'),
        },
        'types': {
            'in': 'inlet', 'wash': 'inlet',
            'cake': 'solid_outlet', 'filtrate': 'liquid_outlet',
        },
    },
    'pump': _single_schema(
        ('suction', 'suct', 'suction_in', 'pump_in', 'low_pressure_in', 'lp_in'),
        ('discharge', 'disch', 'discharge_out', 'pump_out',
         'high_pressure_out', 'hp_out'),
    ),
    'compressor': _single_schema(
        ('suction', 'suct', 'suction_in', 'compressor_in', 'gas_in',
         'low_pressure_in', 'lp_in'),
        ('discharge', 'disch', 'discharge_out', 'compressor_out', 'gas_out',
         'high_pressure_out', 'hp_out'),
    ),
    'expander': _single_schema(
        ('high_pressure_in', 'hp_in', 'expander_in', 'turbine_in'),
        ('exhaust', 'exh', 'low_pressure_out', 'lp_out', 'expander_out',
         'turbine_out'),
    ),
    'valve': _single_schema(
        ('valve_in', 'upstream_in', 'high_pressure_in', 'hp_in'),
        ('valve_out', 'downstream_out', 'low_pressure_out', 'lp_out'),
    ),
    'pipe': _single_schema(
        ('pipe_in', 'upstream_in', 'entrance', 'entry'),
        ('pipe_out', 'downstream_out', 'exit'),
    ),
    'heater': _single_schema(
        ('heater_in', 'cold_in', 'unheated', 'unheated_feed'),
        ('heater_out', 'hot_out', 'heated', 'heated_out', 'heated_product'),
    ),
    'cooler': _single_schema(
        ('cooler_in', 'hot_in', 'uncooled', 'uncooled_feed'),
        ('cooler_out', 'cold_out', 'cooled', 'cooled_out', 'cooled_product'),
    ),
    'reactor': _single_schema(
        ('reactant', 'reactants', 'reactant_feed', 'reactor_feed', 'charge'),
        ('effluent', 'reactor_effluent', 'products', 'reaction_products',
         'reactor_product'),
    ),
    'crystallizer': {
        'inlets': {
            **_COMMON_IN,
            **_names('in', 'crystallizer_feed', 'solution',
                     'mother_liquor_feed'),
        },
        'outlets': {
            **_COMMON_OUT,
            **_names('out', 'slurry', 'slurry_out', 'crystallizer_product'),
            **_names('cake', 'crystals', 'crystal_cake', 'wet_cake'),
            **_names('mother_liquor', 'liquor', 'filtrate', 'mother'),
        },
        'types': {
            'in': 'inlet',
            'out': 'outlet',
            'cake': 'solid_outlet',
            'mother_liquor': 'liquid_outlet',
        },
    },
    'layer_crystallizer': {
        'inlets': {
            **_COMMON_IN,
            **_names('in', 'crystallizer_feed', 'solution',
                     'mother_liquor_feed'),
        },
        'outlets': {
            **_names('cake', 'layer', 'crystals', 'crystal_cake', 'wet_cake'),
            **_names('mother_liquor', 'liquor', 'filtrate', 'mother'),
        },
        'types': {
            'in': 'inlet',
            'cake': 'solid_outlet',
            'mother_liquor': 'liquid_outlet',
        },
    },
    'batch_reactor': {
        'inlets': {
            **_COMMON_IN,
            **_names('charge', 'reactant', 'reactants', 'reactor_feed',
                     'initial_charge'),
            **_names('addition', 'semi_batch_feed', 'semi_batch_addition'),
        },
        'outlets': {
            **_COMMON_OUT,
            **_names('out', 'effluent', 'products', 'reaction_products',
                     'reactor_product'),
        },
        'types': {
            'in': 'inlet', 'charge': 'inlet', 'addition': 'inlet',
            'out': 'outlet',
        },
    },
    'mixer': {
        'inlets': _COMMON_IN,
        'outlets': {
            **_COMMON_OUT,
            **_names('out', 'mixed', 'mix', 'mixed_out', 'mixed_product', 'blend'),
        },
        'types': {'in': 'inlet', 'out': 'outlet'},
        'variable_inlets': True,
    },
    'splitter': {
        'inlets': {
            **_COMMON_IN,
            **_names('in', 'split_feed', 'parent', 'unsplit'),
        },
        'outlets': _SPLITTER_OUTLETS,
        'types': {
            'in': 'inlet', 'out': 'outlet', 'out2': 'outlet',
            'out3': 'outlet', 'out4': 'outlet', 'out5': 'outlet',
            'out6': 'outlet', 'out7': 'outlet', 'out8': 'outlet',
            'out9': 'outlet', 'out10': 'outlet',
        },
        'variable_outlets': True,
    },
    'heat_exchanger': {
        'inlets': {
            **_names('hot_in', 'hi', 'hot', 'h', 'hot_feed', 'hot_inlet',
                     'hot_side_in', 'hot_stream_in'),
            **_names('cold_in', 'ci', 'cold', 'c', 'cold_feed', 'cold_inlet',
                     'cold_side_in', 'cold_stream_in'),
            **_names('shell_in', 'si', 'shell', 'sh', 'shell_feed',
                     'shell_inlet', 'shell_side_in'),
            **_names('tube_in', 'ti', 'tube', 't', 'tube_feed', 'tube_inlet',
                     'tube_side_in'),
        },
        'outlets': {
            **_names('hot_out', 'ho', 'hot', 'h', 'hot_product', 'hot_outlet',
                     'hot_side_out', 'hot_stream_out'),
            **_names('cold_out', 'co', 'cold', 'c', 'cold_product',
                     'cold_outlet', 'cold_side_out', 'cold_stream_out'),
            **_names('shell_out', 'so', 'shell', 'sh', 'shell_product',
                     'shell_outlet', 'shell_side_out'),
            **_names('tube_out', 'to', 'tube', 't', 'tube_product',
                     'tube_outlet', 'tube_side_out'),
        },
        'types': {
            'hot_in': 'inlet', 'cold_in': 'inlet',
            'hot_out': 'outlet', 'cold_out': 'outlet',
            'shell_in': 'inlet', 'tube_in': 'inlet',
            'shell_out': 'outlet', 'tube_out': 'outlet',
        },
    },
    'flash': {
        'inlets': {
            **_COMMON_IN,
            **_names('in', 'flash_feed', 'vessel_feed', 'drum_feed'),
        },
        'outlets': {
            **_names('vapor_out', 'v', 'vap', 'vapor', 'vapour', 'vapour_out',
                     'gas', 'g', 'gas_out', 'overhead', 'top', 'vent',
                     'vapor_product', 'vap_product', 'gas_product'),
            **_names('liquid_out', 'l', 'liq', 'liquid', 'bottom', 'bottoms',
                     'b', 'liquid_product', 'liq_product', 'residue',
                     'drum_liquid'),
        },
        'types': {
            'in': 'inlet',
            'vapor_out': 'vapor_outlet',
            'liquid_out': 'liquid_outlet',
        },
    },
    'flash3': {
        'inlets': {
            **_COMMON_IN,
            **_names('in', 'flash_feed', 'vessel_feed', 'drum_feed'),
        },
        'outlets': {
            **_names('vapor_out', 'v', 'vap', 'vapor', 'vapour', 'vapour_out',
                     'gas', 'g', 'gas_out', 'overhead', 'top', 'vent',
                     'vapor_product'),
            **_names('liquid1_out', 'l1', 'liq1', 'liquid1', 'liquid_1',
                     'first_liquid', 'liquid1_product', 'phase1'),
            **_names('liquid2_out', 'l2', 'liq2', 'liquid2', 'liquid_2',
                     'second_liquid', 'liquid2_product', 'phase2'),
        },
        'types': {
            'in': 'inlet',
            'vapor_out': 'vapor_outlet',
            'liquid1_out': 'liquid1_outlet',
            'liquid2_out': 'liquid2_outlet',
        },
    },
    'decanter': {
        'inlets': {
            **_COMMON_IN,
            **_names('in', 'decanter_feed', 'settler_feed', 'mixed_liquid'),
        },
        'outlets': {
            **_names('light', 'l', 'll', 'light_liquid', 'organic',
                     'organic_phase', 'oil', 'hydrocarbon', 'top',
                     'light_product'),
            **_names('heavy', 'h', 'hl', 'heavy_liquid', 'aqueous',
                     'aqueous_phase', 'water', 'water_phase', 'bottom',
                     'heavy_product'),
        },
        'types': {
            'in': 'inlet',
            'light': 'light_liquid_outlet',
            'heavy': 'heavy_liquid_outlet',
        },
    },
    'distillation': {
        'inlets': _names(
            'feed', 'f', 'in', 'i', 'inlet', 'input', 'column_feed',
            'tower_feed', 'charge', 'z', 'middle_feed',
        ),
        'outlets': {
            **_names('distillate', 'd', 'dist', 'top', 'overhead', 'ovhd',
                     'ovh', 'overhead_product', 'top_product', 'light_product',
                     'light_ends'),
            **_names('distillate_vapor', 'dv', 'vapor_distillate',
                     'vap_distillate', 'overhead_vapor'),
            **_names('distillate_liquid', 'dl', 'liquid_distillate',
                     'liq_distillate', 'overhead_liquid'),
            **_names('bottoms', 'b', 'bot', 'bottom', 'btm', 'btms',
                     'bottom_product', 'heavy_product', 'residue',
                     'bottom_stream'),
        },
        'types': {
            'feed': 'inlet',
            'distillate': 'outlet',
            'distillate_vapor': 'vapor_outlet',
            'distillate_liquid': 'liquid_outlet',
            'bottoms': 'outlet',
        },
        'variable_inlets': True,
        'variable_outlets': True,
    },
    'dryer': {
        'inlets': _names(
            'feed', 'f', 'in', 'i', 'inlet', 'input', 'wet', 'wet_feed',
            'wet_gas', 'wet_stream',
        ),
        'outlets': {
            **_names('product', 'p', 'out', 'o', 'dry', 'dry_product',
                     'dried', 'dried_product', 'dry_gas', 'dry_stream'),
            **_names('adsorbate', 'a', 'ads', 'water', 'water_out',
                     'removed_water', 'moisture', 'waste'),
        },
        'types': {
            'feed': 'inlet',
            'product': 'outlet',
            'adsorbate': 'solid_outlet',
        },
    },
    'extractor': {
        'inlets': {
            **_names('feed', 'f', 'in', 'i', 'inlet', 'solute_feed',
                     'process_feed', 'carrier_feed', 'raffinate_feed'),
            **_names('solvent', 's', 'solv', 'solvent_in', 'solvent_feed',
                     'extractant', 'extractant_feed'),
        },
        'outlets': {
            **_names('raffinate', 'r', 'raf', 'raff', 'raffinate_out',
                     'raffinate_product', 'carrier_product'),
            **_names('extract', 'e', 'ext', 'extract_out', 'extract_product',
                     'solvent_rich'),
        },
        'types': {
            'feed': 'inlet', 'solvent': 'solvent_inlet',
            'raffinate': 'raffinate_outlet', 'extract': 'extract_outlet',
        },
    },
    'absorber': {
        'inlets': {
            **_names('gas', 'g', 'v', 'vapor', 'gas_in', 'vapor_in', 'vap_in',
                     'feed_gas', 'sour_gas', 'raw_gas'),
            **_names('liquid', 'l', 'solvent', 's', 'liquid_in', 'solvent_in',
                     'solvent_feed', 'lean_solvent', 'lean_amine', 'wash'),
        },
        'outlets': {
            **_names('gas_out', 'go', 'vapor_out', 'v', 'treated_gas',
                     'clean_gas', 'sweet_gas', 'overhead', 'top', 'offgas'),
            **_names('liquid_out', 'lo', 'solvent_out', 'l', 'rich_solvent',
                     'rich_amine', 'loaded_solvent', 'bottoms', 'bottom'),
        },
        'types': {
            'gas': 'gas_inlet', 'liquid': 'inlet',
            'gas_out': 'vapor_outlet', 'liquid_out': 'liquid_outlet',
        },
        'variable_inlets': True,
    },
    'stripper': {
        'inlets': {
            **_names('liquid', 'l', 'feed', 'f', 'liquid_in', 'rich_solvent',
                     'rich_amine', 'loaded_solvent', 'stripper_feed'),
            **_names('gas', 'g', 'strip_gas', 'steam', 's', 'gas_in',
                     'steam_in', 'stripping_gas', 'stripping_steam'),
        },
        'outlets': {
            **_names('gas_out', 'go', 'vapor_out', 'v', 'overhead', 'top',
                     'offgas', 'acid_gas', 'stripped_gas'),
            **_names('liquid_out', 'lo', 'bottoms', 'b', 'l', 'bottom',
                     'lean_solvent', 'lean_amine', 'stripped_liquid'),
        },
        'types': {
            'liquid': 'inlet', 'gas': 'gas_inlet',
            'gas_out': 'vapor_outlet', 'liquid_out': 'liquid_outlet',
        },
        'variable_inlets': True,
    },
}


def canonical_unit_type(unit_type: str) -> str | None:
    """Return the canonical implemented unit type, if registered."""
    return UNIT_TYPE_ALIASES.get(str(unit_type).strip())


def port_schema_for_unit_type(unit_type: str) -> dict | None:
    """Return the compact-port schema for one public unit spelling."""
    canonical = canonical_unit_type(unit_type)
    family = UNIT_PORT_FAMILIES.get(canonical) if canonical else None
    schema = PORT_FAMILY_SCHEMAS.get(family) if family else None
    if (
        schema is not None
        and family == 'distillation'
        and canonical != 'RigorousDistillation'
    ):
        rigorous_only = {'distillate_vapor', 'distillate_liquid'}
        return {
            **schema,
            'outlets': {
                alias: target for alias, target in schema['outlets'].items()
                if target not in rigorous_only
            },
            'types': {
                name: port_type for name, port_type in schema['types'].items()
                if name not in rigorous_only
            },
        }
    return schema


def port_family_for_unit_type(unit_type: str) -> str | None:
    """Return the compact-port family for one public unit spelling."""
    canonical = canonical_unit_type(unit_type)
    return UNIT_PORT_FAMILIES.get(canonical) if canonical else None


def unit_allows_variable_port(unit_type: str, direction: str) -> bool:
    """Whether a compact unit permits arbitrary named ports in one direction."""
    canonical = canonical_unit_type(unit_type)
    return (canonical, direction) in VARIABLE_UNIT_PORT_DIRECTIONS
