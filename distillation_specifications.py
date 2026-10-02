"""Shared input contract for coupled distillation overhead liquid routing."""

import math


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
