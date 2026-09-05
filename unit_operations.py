"""
Unit Operations Module

Public facade for all supported unit operation classes.
"""

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics import StreamState, IdealThermodynamics, ThermodynamicsError
else:
    from thermodynamics import StreamState, IdealThermodynamics, ThermodynamicsError
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_base import UnitOperation, UnitOperationError, UnitResult
else:
    from unit_operations_base import UnitOperation, UnitOperationError, UnitResult
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_basic import (
        Mixer,
        Splitter,
        Pump,
        Compressor,
        Expander,
        Valve,
        Heater,
        Cooler,
        HeatExchanger,
        Flash,
    )
else:
    from unit_operations_basic import (
        Mixer,
        Splitter,
        Pump,
        Compressor,
        Expander,
        Valve,
        Heater,
        Cooler,
        HeatExchanger,
        Flash,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_reactors import (
        Reactor,
        EquilibriumReactor,
        KineticsCSTR,
        KineticsBatch,
        KineticsPFR,
        KineticsPackedBed,
    )
else:
    from unit_operations_reactors import (
        Reactor,
        EquilibriumReactor,
        KineticsCSTR,
        KineticsBatch,
        KineticsPFR,
        KineticsPackedBed,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_transport import Pipe
else:
    from unit_operations_transport import Pipe
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_solids import Crystallizer
else:
    from unit_operations_solids import Crystallizer
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_filtration import Filter
else:
    from unit_operations_filtration import Filter
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_distillation import (
        ShortcutDistillation,
        McCabeThieleDistillation,
        CMODistillation,
        RigorousDistillation,
    )
else:
    from unit_operations_distillation import (
        ShortcutDistillation,
        McCabeThieleDistillation,
        CMODistillation,
        RigorousDistillation,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_operations_separation import (
        Decanter,
        Flash3,
        MolecularSieveDryer,
        ShortcutExtractor,
        LiquidLiquidExtractor,
        RigorousLiquidLiquidExtractor,
        Absorber,
        RigorousAbsorber,
        RigorousStripper,
        Stripper,
    )
else:
    from unit_operations_separation import (
        Decanter,
        Flash3,
        MolecularSieveDryer,
        ShortcutExtractor,
        LiquidLiquidExtractor,
        RigorousLiquidLiquidExtractor,
        Absorber,
        RigorousAbsorber,
        RigorousStripper,
        Stripper,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_syntax import UNIT_TYPE_ALIASES
else:
    from unit_syntax import UNIT_TYPE_ALIASES


_CANONICAL_UNIT_CLASSES = {
    'Mixer': Mixer,
    'Splitter': Splitter,
    'Pump': Pump,
    'Compressor': Compressor,
    'Expander': Expander,
    'Valve': Valve,
    'Pipe': Pipe,
    'Heater': Heater,
    'Cooler': Cooler,
    'HeatExchanger': HeatExchanger,
    'Flash': Flash,
    'ShortcutDistillation': ShortcutDistillation,
    'McCabeThieleDistillation': McCabeThieleDistillation,
    'CMODistillation': CMODistillation,
    'RigorousDistillation': RigorousDistillation,
    'Flash3': Flash3,
    'Decanter': Decanter,
    'MolecularSieveDryer': MolecularSieveDryer,
    'ShortcutExtractor': ShortcutExtractor,
    'RigorousExtractor': RigorousLiquidLiquidExtractor,
    'Absorber': Absorber,
    'RigorousAbsorber': RigorousAbsorber,
    'Stripper': Stripper,
    'RigorousStripper': RigorousStripper,
    'Reactor': Reactor,
    'EquilibriumReactor': EquilibriumReactor,
    'CSTR': KineticsCSTR,
    'BatchReactor': KineticsBatch,
    'PFR': KineticsPFR,
    'PackedBedReactor': KineticsPackedBed,
    'Crystallizer': Crystallizer,
    'Filter': Filter,
}

UNIT_CLASSES = {
    public_name: _CANONICAL_UNIT_CLASSES[canonical_name]
    for public_name, canonical_name in UNIT_TYPE_ALIASES.items()
}


def create_unit(unit_type: str, unit_id: str, thermo: IdealThermodynamics,
                params: dict) -> UnitOperation:
    """Factory function to create unit operation instances"""

    cls = UNIT_CLASSES.get(unit_type)
    if cls is None:
        raise UnitOperationError(f"Unsupported unit type: {unit_type}")

    return cls(unit_id, thermo, params)


__all__ = [
    'UnitOperationError',
    'UnitResult',
    'UnitOperation',
    'StreamState',
    'IdealThermodynamics',
    'Mixer',
    'Splitter',
    'Pump',
    'Compressor',
    'Expander',
    'Valve',
    'Pipe',
    'Heater',
    'Cooler',
    'HeatExchanger',
    'Flash',
    'Reactor',
    'EquilibriumReactor',
    'ShortcutDistillation',
    'McCabeThieleDistillation',
    'KineticsCSTR',
    'KineticsBatch',
    'KineticsPFR',
    'KineticsPackedBed',
    'Crystallizer',
    'Filter',
    'CMODistillation',
    'RigorousDistillation',
    'Flash3',
    'Decanter',
    'MolecularSieveDryer',
    'ShortcutExtractor',
    'LiquidLiquidExtractor',
    'RigorousLiquidLiquidExtractor',
    'Absorber',
    'RigorousAbsorber',
    'RigorousStripper',
    'Stripper',
    'UNIT_CLASSES',
    'create_unit',
    'ThermodynamicsError',
]
