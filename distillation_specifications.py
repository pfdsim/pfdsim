"""Shared input contract for coupled distillation overhead liquid routing."""

import math
import ast
import re

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .unit_conversions import temperature_to_kelvin
else:
    from unit_conversions import temperature_to_kelvin


CONDENSER_THERMAL_PARAMETERS = frozenset({'condenser_temperature', 'condenser_subcooling'})
STAGE_EFFICIENCY_PARAMETERS = frozenset({'stage_efficiency', 'stage_efficiencies'})


def stage_efficiency_specification(params, stages=None, *, supports_efficiency=True):
    """Return vapor Murphree efficiencies, with equilibrium end boundaries.

    Uniform efficiency is the base. A profile replaces all tray values; a map
    overrides individual stages or nonoverlapping inclusive ranges. Profiles
    can contain N entries (including end boundaries) or N-2 tray entries.
    """
    pairs = params.items() if hasattr(params, 'items') else params
    values = {}
    for key,value in pairs:
        key = str(key).strip().lower()
        if key in values and key in STAGE_EFFICIENCY_PARAMETERS:
            raise ValueError(f'Duplicate distillation specification: {key}')
        values[key] = value
    present = STAGE_EFFICIENCY_PARAMETERS.intersection(values)
    if not present and not supports_efficiency:
        return None
    if present and not supports_efficiency:
        raise ValueError('Stage efficiencies require RigorousDistillation')
    if stages is None:
        stages = values.get('n_stages',values.get('stages',10))
    try:
        number = float(stages)
        count = int(number)
        if count != number or count < 2:
            raise ValueError()
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError('Stage efficiencies require an integer N_stages >= 2') from error
    def fraction(value):
        try:
            number = float(value)
        except (TypeError,ValueError) as error:
            raise ValueError('Stage efficiency must be a finite fraction between 0 and 1') from error
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError('Stage efficiency must be a finite fraction between 0 and 1')
        return number
    for name in present:
        if values.get(f'__unit__{name}'):
            raise ValueError(f'{name} is dimensionless and must not specify units')
    result = [1.]*count
    base = fraction(values.get('stage_efficiency',1.))
    result[1:-1] = [base]*(count-2)
    profile = values.get('stage_efficiencies')
    if profile is None:
        return tuple(result)
    if isinstance(profile,str):
        try:
            profile = ast.literal_eval(profile)
        except (ValueError,SyntaxError) as error:
            raise ValueError('stage_efficiencies must be an ordered list or stage/range map') from error
    if isinstance(profile,(list,tuple)):
        if len(profile) == count-2:
            result[1:-1] = [fraction(v) for v in profile]
        elif len(profile) == count:
            result = [fraction(v) for v in profile]
        else:
            raise ValueError(f'stage_efficiencies profile must contain {count} stage values or {count-2} tray values')
    elif isinstance(profile,dict):
        used = set()
        for key,value in profile.items():
            match = re.fullmatch(r'(\d+)(?:\s*-\s*(\d+))?',str(key).strip())
            if match is None:
                raise ValueError(f'Invalid stage selector {key!r}; use a stage number or inclusive range such as 2-8')
            first,last = int(match[1]),int(match[2] or match[1])
            if not 1 <= first <= last <= count:
                raise ValueError(f'Stage selector {key!r} lies outside stages 1-{count}')
            number = fraction(value)
            for stage in range(first,last+1):
                if stage in used:
                    raise ValueError(f'Overlapping stage-efficiency selectors on stage {stage}')
                used.add(stage)
                result[stage-1] = number
    else:
        raise ValueError('stage_efficiencies must be an ordered list or stage/range map')
    if result[0] != 1. or result[-1] != 1.:
        raise ValueError('Condenser stage 1 and reboiler stage N must have efficiency 1')
    return tuple(result)


def total_condenser_specification(params, *, supports_subcooling=True):
    """Normalize absolute temperature or temperature-difference input once.

    These fields retain their raw PFD values/units until this shared validator;
    subcooling must never receive the offset used for absolute temperatures.
    """
    pairs = params.items() if hasattr(params, 'items') else params
    values = {}
    for key, value in pairs:
        key = str(key).strip().lower()
        if key in values and key in CONDENSER_THERMAL_PARAMETERS:
            raise ValueError(f'Duplicate distillation specification: {key}')
        values[key] = value
    present = CONDENSER_THERMAL_PARAMETERS.intersection(values)
    if not present:
        return None
    if not supports_subcooling:
        raise ValueError('Condenser temperature/subcooling requires RigorousDistillation')
    if len(present) != 1:
        raise ValueError('Specify only one of condenser_temperature and condenser_subcooling')
    condenser = str(values.get('condenser_type', 'total')).strip().lower()
    if condenser not in ('total', 'complete', 'liquid', 'total_condenser'):
        raise ValueError('Condenser temperature/subcooling requires a total condenser')
    name = next(iter(present))
    try:
        value = float(values[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be a finite temperature') from error
    unit = str(values.get(f'__unit__{name}') or '').strip().lower()
    if unit not in ('', 'k', 'kelvin', 'c', '°c', 'celsius', 'f', '°f', 'fahrenheit'):
        raise ValueError(f'Unsupported {name} unit: {unit}')
    if name == 'condenser_temperature':
        value = temperature_to_kelvin(value, unit, infer_unitless_celsius_below=200.)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('condenser_temperature must be a positive finite absolute temperature')
        return {'temperature_K':value, 'subcooling_K':None}
    if unit in ('f', '°f', 'fahrenheit'):
        value *= 5./9.
    if not math.isfinite(value) or value < 0:
        raise ValueError('condenser_subcooling must be a finite nonnegative temperature difference')
    return {'temperature_K':None, 'subcooling_K':value}


REMOVED_DECANTER_CONDENSERS = frozenset({
    'decanter', 'heterogeneous', 'heterogeneous_decanter', 'top_decanter',
})
REMOVED_DECANTER_PARAMETERS = frozenset({
    'decanter_reflux_phase', 'decanter_distillate_phase',
    'decanter_reflux_component', 'decanter_distillate_component',
    'decanter_reflux_purge_fraction', 'reflux_phase_component',
    'distillate_phase_component', 'reflux_purge_fraction',
    'decanter_T', 'condenser_T', 'T_condenser', 'decanter_distillate_guess',
    'distillate_flow_guess', 'D_guess',
})
LIQUID_ROUTING_PARAMETERS = frozenset({
    'distillate_liquid1_fraction', 'distillate_liquid2_fraction',
    'distillate_liquid1_component',
})
DECANTER_REMOVAL_MESSAGE = (
    'The VLE decanter condenser was removed. Enable stage_phase_model=VLLE '
    'for a proper coupled three-phase solve, use condenser_type=total or mixed, '
    'and specify distillate_liquid1_fraction, distillate_liquid2_fraction, '
    'and distillate_liquid1_component for phase-selective withdrawal.'
)


def stage_phase_model(params, default='VLE'):
    params = {str(key).strip().lower():value for key,value in params.items()}
    value = next((params[key] for key in ('stage_phase_model', 'valid_phases', 'stage_phases')
                  if params.get(key) is not None), default)
    normalized = str(value).strip().upper().replace('-', '').replace('_', '')
    if normalized in ('VLE', 'VL'):
        return 'VLE'
    if normalized in ('VL(L)E', 'VLL(E)', 'ADAPTIVEVLLE', 'SPINODALVLLE'):
        return 'VL(L)E'
    if normalized in ('VLLE', 'VLL'):
        return 'VLLE'
    raise ValueError('stage_phase_model must be VLE, VL(L)E, or VLLE')


def liquid_distillate_routing(params, *, default_phase_model='VLE', components=None,
                              supports_phase_routing=True):
    """Validate routing without running thermodynamics; return canonical options."""
    pairs = params.items() if hasattr(params, 'items') else params
    values = {}
    removed = {name.lower() for name in REMOVED_DECANTER_PARAMETERS}
    for key,value in pairs:
        key = str(key).strip().lower()
        if key in removed or (key == 'condenser_type' and
                str(value).strip().lower().replace('-', '_') in REMOVED_DECANTER_CONDENSERS):
            raise ValueError(DECANTER_REMOVAL_MESSAGE)
        if key in values and key in LIQUID_ROUTING_PARAMETERS | {'condenser_type'}:
            raise ValueError(f'Duplicate distillation specification: {key}')
        values[key] = value
    condenser = str(values.get('condenser_type', 'total')).strip().lower().replace('-', '_')
    present = LIQUID_ROUTING_PARAMETERS.intersection(values)
    if not present:
        return None
    if not supports_phase_routing:
        raise ValueError('Liquid phase withdrawal requires RigorousDistillation with stage_phase_model=VLLE')
    for name in present:
        if values.get(f'__unit__{name}'):
            raise ValueError(f'{name} is dimensionless and must not specify units')
    required = {'distillate_liquid1_fraction', 'distillate_liquid2_fraction'}
    if not required.issubset(present):
        raise ValueError('Specify both distillate_liquid1_fraction and distillate_liquid2_fraction')
    fractions = []
    for name in ('distillate_liquid1_fraction', 'distillate_liquid2_fraction'):
        try:
            value = float(values[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f'{name} must be a finite fraction between 0 and 1') from error
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be a finite fraction between 0 and 1')
        fractions.append(value)
    if max(fractions) == 0:
        raise ValueError('At least one top liquid must have a positive distillate withdrawal fraction')
    if stage_phase_model(values, default_phase_model) != 'VLLE':
        raise ValueError('Phase-selective liquid withdrawal requires stage_phase_model=VLLE for a coupled three-phase solve')
    if condenser not in ('total', 'complete', 'liquid', 'total_condenser',
                         'mixed', 'two_phase', 'mixed_distillate', 'partial_liquid'):
        raise ValueError('Liquid withdrawal fractions require a total or mixed condenser')
    if any(name in values for name in ('reflux_ratio', 'rr')):
        raise ValueError('Liquid withdrawal fractions determine reflux flow; omit reflux_ratio/RR')
    selector = str(values.get('distillate_liquid1_component', '') or '').strip()
    if fractions[0] != fractions[1] and not selector:
        raise ValueError('Unequal liquid withdrawal fractions require distillate_liquid1_component to identify the richer phase')
    if selector and components is not None:
        matches = [comp for comp in components if comp.casefold() == selector.casefold()]
        if len(matches) != 1:
            raise ValueError(f"distillate_liquid1_component '{selector}' must identify a declared column component")
        selector = matches[0]
    return {'fractions':tuple(fractions), 'component':selector or None}
