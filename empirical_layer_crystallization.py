"""Empirical finite-rate layer growth and impurity incorporation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from scipy.integrate import solve_ivp

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .kinetic_models import KineticsError, SafeRateExpression
    from .msmpr_models import (
        CRYSTALLIZATION_EXPRESSION_FUNCTIONS,
        GROWTH_RATE_FACTORS_M_PER_H,
        crystallization_saturation_temperature,
    )
    from .reaction_models import ReactionDefinitionError
    from .thermodynamics_models.common import R, ThermodynamicsError
    from .thermodynamics_models.sle import (
        liquid_solution_activities,
        pure_solid_log_saturation_activity,
    )
else:
    from kinetic_models import KineticsError, SafeRateExpression
    from msmpr_models import (
        CRYSTALLIZATION_EXPRESSION_FUNCTIONS,
        GROWTH_RATE_FACTORS_M_PER_H,
        crystallization_saturation_temperature,
    )
    from reaction_models import ReactionDefinitionError
    from thermodynamics_models.common import R, ThermodynamicsError
    from thermodynamics_models.sle import (
        liquid_solution_activities,
        pure_solid_log_saturation_activity,
    )


class EmpiricalLayerDefinitionError(ValueError):
    """Raised when an empirical layer law is invalid."""


@dataclass(frozen=True)
class EmpiricalLayerLaw:
    kind: str
    model: str
    expression: SafeRateExpression
    parameters: dict[str, float]
    declared_unit: str
    canonical_factor: float

    def evaluate(self, context: Mapping[str, object]) -> float:
        values = dict(context)
        values.update(self.parameters)
        missing = sorted(self.expression.names - set(values))
        if missing:
            raise ThermodynamicsError(
                f'Empirical layer {self.kind} expression requires unavailable '
                'variable(s): ' + ', '.join(missing)
            )
        try:
            value = self.expression.evaluate(values) * self.canonical_factor
        except (KineticsError, KeyError) as error:
            raise ThermodynamicsError(
                f'Empirical layer {self.kind} expression failed: {error}'
            ) from error
        if not math.isfinite(value) or value < 0:
            raise ThermodynamicsError(
                f'Empirical layer {self.kind} expression must return a finite '
                'nonnegative value'
            )
        return value


@dataclass(frozen=True)
class EmpiricalLayerGrowthResult:
    solid_amount_kmol: float
    thickness_m: float
    trapped_component_amounts_kmol: dict[str, float]
    liquid_component_amounts_kmol: dict[str, float]
    profile: list[dict]
    evaluations: int
    elapsed_growth_time_h: float
    stopped_by_inventory: bool
    solid_density_kg_m3: float


_LAYER_SCALARS = frozenset({
    'T', 'T_bulk', 'P', 'Twall', 'T_wall', 'Tcool', 'T_coolant',
    'Tsat', 'Teq', 'T_eq', 'deltaT', 'dT', 'deltaT_bulk',
    'deltaT_wall', 'wall_undercooling', 'S', 'sigma',
    'relative_supersaturation', 'lnS', 'activity', 'a', 'a_sat',
    'asat', 'time', 't', 'time_h', 'time_s', 'duration', 'duration_h',
    'thickness', 'L', 'area', 'recovery', 'x_crystal', 'w_crystal',
    'x_impurity', 'w_impurity', 'G', 'G_m_h', 'G_m_s', 'Tm',
    'Hfus', 'rho_s', 'Vm_solid', 'pi', 'R',
})
_LAYER_MAPPINGS = frozenset({'x', 'w', 'x0', 'w0'})
_GROWTH_SCALARS = _LAYER_SCALARS - frozenset({
    'x_impurity', 'w_impurity', 'G', 'G_m_h', 'G_m_s',
})
_FLAT_GROWTH_FIELDS = frozenset({
    'growth_model', 'growth_expression', 'growth_rate_unit', 'growth_unit',
    'growth_value', 'growth_rate', 'growth_coefficient', 'growth_k',
    'growth_kg', 'growth_exponent', 'growth_g',
})


def _is_flat_growth_field(name):
    lower = str(name).lower()
    return lower in _FLAT_GROWTH_FIELDS or lower.startswith('growth_param_')


def _finite(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EmpiricalLayerDefinitionError(f'{label} must be numeric') from error
    if not math.isfinite(result):
        raise EmpiricalLayerDefinitionError(f'{label} must be finite')
    return result


def _mapping_fields(definition, label):
    if not isinstance(definition, Mapping):
        raise EmpiricalLayerDefinitionError(f'{label} must be a mapping')
    fields = {str(key): value for key, value in definition.items()}
    lookup = {key.lower(): key for key in fields}
    if len(lookup) != len(fields):
        raise EmpiricalLayerDefinitionError(f'{label} has duplicate case-insensitive fields')
    return fields, lookup


def _parameters(fields, lookup, label, *, custom):
    result = {}
    nested_key = lookup.get('parameters')
    nested = fields[nested_key] if nested_key is not None else {}
    if nested in (None, ''):
        nested = {}
    if not isinstance(nested, Mapping):
        raise EmpiricalLayerDefinitionError(f'{label} parameters must be a mapping')
    if nested and not custom:
        raise EmpiricalLayerDefinitionError(f'{label} parameters require model=custom')
    for name, value in nested.items():
        result[str(name)] = _finite(value, f'{label} parameter {name}')
    for raw_name, value in fields.items():
        if not raw_name.lower().startswith('param_'):
            continue
        if not custom:
            raise EmpiricalLayerDefinitionError(f'{label} param_* values require model=custom')
        name = raw_name[6:]
        if name in result:
            raise EmpiricalLayerDefinitionError(f'duplicate {label} parameter {name!r}')
        result[name] = _finite(value, f'{label} parameter {name}')
    for name in result:
        if (
            not name.isidentifier()
            or name in _LAYER_SCALARS
            or name in _LAYER_MAPPINGS
            or name in CRYSTALLIZATION_EXPRESSION_FUNCTIONS
        ):
            raise EmpiricalLayerDefinitionError(f'invalid {label} parameter name {name!r}')
    return result


def _safe_expression(text, parameters, label, *, growth=False):
    try:
        return SafeRateExpression(
            str(text).strip().strip('"\''),
            parameters,
            scalar_names=_GROWTH_SCALARS if growth else _LAYER_SCALARS,
            mapping_names=_LAYER_MAPPINGS,
            function_names=CRYSTALLIZATION_EXPRESSION_FUNCTIONS,
        )
    except ReactionDefinitionError as error:
        raise EmpiricalLayerDefinitionError(f'invalid {label} expression: {error}') from error


def empirical_layer_growth_law_from_mapping(definition):
    """Build a constant, undercooling-power-law, or custom growth law."""
    fields, lookup = _mapping_fields(definition, 'empirical layer growth definition')

    def get(name, default=None):
        key = lookup.get(name)
        return fields[key] if key is not None else default

    def one(names, label, default=None):
        found = [name for name in names if get(name) is not None]
        if len(found) > 1:
            raise EmpiricalLayerDefinitionError(
                f'empirical layer growth received duplicate {label} aliases: '
                + ', '.join(found)
            )
        return get(found[0]) if found else default

    expression = get('expression')
    value = one(('value', 'rate'), 'rate')
    raw_model = one(('model', 'type'), 'model')
    if raw_model is None:
        model = 'custom' if expression is not None else (
            'constant' if value is not None else 'undercooling_power_law'
        )
    else:
        model = str(raw_model).strip().lower().replace('-', '_')
    model = {
        'power_law': 'undercooling_power_law',
        'delta_t_power_law': 'undercooling_power_law',
        'deltat_power_law': 'undercooling_power_law',
    }.get(model, model)
    if model not in {'constant', 'undercooling_power_law', 'custom'}:
        raise EmpiricalLayerDefinitionError(
            'empirical layer growth model must be constant, '
            'undercooling_power_law, or custom'
        )
    custom = model == 'custom'
    parameters = _parameters(fields, lookup, 'empirical layer growth', custom=custom)
    if model == 'constant':
        value = one(
            ('value', 'rate', 'coefficient', 'k', 'kg'), 'constant-rate'
        )
        value = _finite(value, 'empirical layer growth rate')
        if value < 0:
            raise EmpiricalLayerDefinitionError('empirical layer growth rate must be nonnegative')
        parameters['constant_rate'] = value
        expression_text = 'constant_rate'
    elif model == 'undercooling_power_law':
        coefficient = one(('coefficient', 'k', 'kg'), 'coefficient')
        exponent = one(('exponent', 'g'), 'exponent')
        if coefficient is None or exponent is None:
            raise EmpiricalLayerDefinitionError(
                'undercooling_power_law requires coefficient and exponent'
            )
        coefficient = _finite(coefficient, 'empirical layer growth coefficient')
        exponent = _finite(exponent, 'empirical layer growth exponent')
        if coefficient < 0 or exponent < 0:
            raise EmpiricalLayerDefinitionError(
                'empirical layer growth coefficient and exponent must be nonnegative'
            )
        parameters.update({'coefficient': coefficient, 'exponent': exponent})
        expression_text = 'coefficient * max(deltaT, 0)**exponent'
    else:
        if expression is None:
            raise EmpiricalLayerDefinitionError('custom empirical layer growth requires expression')
        expression_text = expression
    unit_value = one(('rate_unit', 'unit'), 'rate-unit')
    if unit_value is None or not str(unit_value).strip():
        raise EmpiricalLayerDefinitionError('empirical layer growth requires explicit rate_unit')
    unit = str(unit_value).strip().lower().replace(' ', '')
    factor = GROWTH_RATE_FACTORS_M_PER_H.get(unit)
    if factor is None:
        raise EmpiricalLayerDefinitionError(
            f'unsupported empirical layer growth rate_unit={unit_value!r}; supported values are '
            + ', '.join(sorted(GROWTH_RATE_FACTORS_M_PER_H))
        )
    common = {'model', 'type', 'rate_unit', 'unit', 'parameters'}
    allowed = common | (
        {'expression'} if model == 'custom'
        else {'value', 'rate', 'coefficient', 'k', 'kg'}
        if model == 'constant'
        else {'coefficient', 'k', 'kg', 'exponent', 'g'}
    )
    unexpected = sorted(
        key for key in fields
        if key.lower() not in allowed and not key.lower().startswith('param_')
    )
    if unexpected:
        raise EmpiricalLayerDefinitionError(
            'empirical layer growth does not accept field(s): ' + ', '.join(unexpected)
        )
    return EmpiricalLayerLaw(
        'growth', model,
        _safe_expression(expression_text, parameters, 'growth', growth=True),
        parameters, unit, factor,
    )


def empirical_layer_distribution_law_from_mapping(definition, component):
    """Build a constant or custom effective-distribution-coefficient law."""
    if not isinstance(definition, Mapping):
        definition = {'value': definition}
    fields, lookup = _mapping_fields(definition, f'{component} k_eff definition')

    def get(name, default=None):
        key = lookup.get(name)
        return fields[key] if key is not None else default

    def one(names, label, default=None):
        found = [name for name in names if get(name) is not None]
        if len(found) > 1:
            raise EmpiricalLayerDefinitionError(
                f'{component} k_eff received duplicate {label} aliases: '
                + ', '.join(found)
            )
        return get(found[0]) if found else default

    expression = get('expression')
    value = one(('value', 'coefficient', 'k'), 'value')
    raw_model = one(('model', 'type'), 'model')
    model = (
        str(raw_model).strip().lower().replace('-', '_')
        if raw_model is not None else 'custom' if expression is not None else 'constant'
    )
    if model not in {'constant', 'custom'}:
        raise EmpiricalLayerDefinitionError(
            f'{component} k_eff model must be constant or custom'
        )
    parameters = _parameters(
        fields, lookup, f'{component} k_eff', custom=model == 'custom'
    )
    if model == 'constant':
        if value is None:
            raise EmpiricalLayerDefinitionError(f'{component} constant k_eff requires value')
        value = _finite(value, f'{component} k_eff')
        if value < 0:
            raise EmpiricalLayerDefinitionError(f'{component} k_eff must be nonnegative')
        parameters['constant_keff'] = value
        expression_text = 'constant_keff'
    else:
        if expression is None:
            raise EmpiricalLayerDefinitionError(f'{component} custom k_eff requires expression')
        expression_text = expression
    allowed = {'model', 'type', 'parameters'} | (
        {'expression'} if model == 'custom' else {'value', 'coefficient', 'k'}
    )
    unexpected = sorted(
        key for key in fields
        if key.lower() not in allowed and not key.lower().startswith('param_')
    )
    if unexpected:
        raise EmpiricalLayerDefinitionError(
            f'{component} k_eff does not accept field(s): ' + ', '.join(unexpected)
        )
    return EmpiricalLayerLaw(
        f'k_eff[{component}]', model,
        _safe_expression(expression_text, parameters, f'{component} k_eff'),
        parameters, 'dimensionless', 1.0,
    )


def empirical_layer_growth_definition_from_parameters(parameters):
    direct_definitions = [
        parameters[name]
        for name in ('growth', 'g')
        if isinstance(parameters.get(name), Mapping)
    ]
    if len(direct_definitions) > 1:
        raise EmpiricalLayerDefinitionError(
            'empirical layer growth received duplicate growth mappings'
        )
    if direct_definitions:
        if any(_is_flat_growth_field(key) for key in parameters) or any(
            parameters.get(name) is not None
            and not isinstance(parameters.get(name), Mapping)
            for name in ('growth', 'g')
        ):
            raise EmpiricalLayerDefinitionError(
                'empirical layer growth received both mapping and flattened fields'
            )
        return direct_definitions[0]
    direct_rate_names = [
        name for name in ('growth_rate', 'growth', 'g')
        if parameters.get(name) is not None
        and not isinstance(parameters.get(name), Mapping)
    ]
    if len(direct_rate_names) > 1:
        raise EmpiricalLayerDefinitionError(
            'empirical layer growth received duplicate direct growth rates'
        )
    direct_rate_name = direct_rate_names[0] if direct_rate_names else None
    if direct_rate_name is not None:
        conflicting = [
            key for key in parameters
            if _is_flat_growth_field(key)
            and str(key).lower() not in {
                'growth_rate', 'growth_rate_unit', 'growth_unit',
            }
        ]
        if conflicting:
            raise EmpiricalLayerDefinitionError(
                'growth_rate conflicts with ' + ', '.join(sorted(conflicting))
            )
        attached_unit = parameters.get(f'__unit__{direct_rate_name}')
        declared_units = [
            parameters[name]
            for name in ('growth_rate_unit', 'growth_unit')
            if parameters.get(name) is not None
        ]
        if attached_unit is not None and declared_units:
            raise EmpiricalLayerDefinitionError(
                'direct growth rate received both an attached unit and growth_rate_unit'
            )
        if len(declared_units) > 1:
            raise EmpiricalLayerDefinitionError(
                'direct growth rate received duplicate unit aliases'
            )
        unit = attached_unit if attached_unit is not None else (
            declared_units[0] if declared_units else None
        )
        return {'value': parameters[direct_rate_name], 'rate_unit': unit}
    definition = {}
    for raw_key, value in parameters.items():
        key = str(raw_key)
        lower = key.lower()
        if lower.startswith('__unit__'):
            continue
        if lower.startswith('growth_param_'):
            definition['param_' + key[len('growth_param_'):]] = value
        elif lower in _FLAT_GROWTH_FIELDS:
            field = lower[len('growth_'):]
            if field in definition:
                raise EmpiricalLayerDefinitionError(f'duplicate growth field {field!r}')
            definition[field] = value
    if not definition:
        raise EmpiricalLayerDefinitionError(
            'empirical layer growth requires growth_rate or growth kinetics'
        )
    return definition


def empirical_layer_distribution_definitions_from_parameters(parameters, components):
    """Collect nested API or flattened ``keff_COMPONENT_*`` definitions."""
    component_lookup = {str(component).casefold(): component for component in components}
    definitions = {}
    nested_names = (
        'effective_distributions', 'distribution_coefficients', 'keff', 'k_eff',
    )
    for nested_name in nested_names:
        nested = parameters.get(nested_name)
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            raise EmpiricalLayerDefinitionError(f'{nested_name} must be a mapping')
        for raw_component, definition in nested.items():
            component = component_lookup.get(str(raw_component).casefold())
            if component is None:
                raise EmpiricalLayerDefinitionError(
                    f'unknown empirical layer impurity {raw_component!r}'
                )
            if component in definitions:
                raise EmpiricalLayerDefinitionError(
                    f'duplicate empirical layer k_eff for {component!r}'
                )
            definitions[component] = definition
    prefixes = ('effective_distribution_', 'distribution_', 'k_eff_', 'keff_')
    for raw_key, value in parameters.items():
        key = str(raw_key)
        lower = key.casefold()
        if lower in nested_names:
            continue
        prefix = next((item for item in prefixes if lower.startswith(item)), None)
        if prefix is None:
            continue
        remainder = key[len(prefix):]
        matches = []
        for folded, component in component_lookup.items():
            if remainder.casefold() == folded:
                matches.append((component, 'value'))
            elif remainder.casefold().startswith(folded + '_'):
                matches.append((component, remainder[len(str(component)) + 1:]))
        if not matches:
            raise EmpiricalLayerDefinitionError(
                f'cannot identify impurity component in parameter {key!r}'
            )
        component, field = max(matches, key=lambda item: len(str(item[0])))
        existing = definitions.get(component)
        if existing is None:
            existing = {}
            definitions[component] = existing
        elif not isinstance(existing, Mapping):
            if field == 'value':
                raise EmpiricalLayerDefinitionError(f'duplicate k_eff for {component!r}')
            existing = {'value': existing}
            definitions[component] = existing
        else:
            existing = dict(existing)
            definitions[component] = existing
        canonical = 'param_' + field[6:] if field.casefold().startswith('param_') else field.lower()
        if canonical in existing:
            raise EmpiricalLayerDefinitionError(
                f'duplicate {component} k_eff field {canonical!r}'
            )
        existing[canonical] = value
    return definitions


def _mass_fractions(thermo, mole_fractions):
    masses = {
        component: fraction * thermo.props[component].MW
        for component, fraction in mole_fractions.items()
    }
    total = sum(masses.values())
    return {component: mass / total for component, mass in masses.items()}


def solve_empirical_layer_growth(
    thermo, *, component, amounts_kmol, bulk_temperature_K, pressure_bar,
    wall_temperature_K, area_m2, growth_time_h, growth_law,
    distribution_laws, solid_density_kg_m3=None, relative_tolerance=1e-7,
    profile_points=21,
):
    """Integrate empirical layer thickness and component incorporation."""
    T = float(bulk_temperature_K)
    P = float(pressure_bar)
    wall = float(wall_temperature_K)
    area = float(area_m2)
    duration = float(growth_time_h)
    tolerance = float(relative_tolerance)
    if not all(math.isfinite(value) and value > 0 for value in (T, P, area, duration, tolerance)):
        raise ThermodynamicsError('Empirical layer temperatures, pressure, area, time, and tolerance must be positive')
    if not math.isfinite(wall) or wall <= 0 or wall >= T:
        raise ThermodynamicsError('Empirical layer wall temperature must be positive and below bulk temperature')
    if int(profile_points) != profile_points or profile_points < 2:
        raise ThermodynamicsError('Empirical layer profile_points must be an integer >= 2')
    amounts = {name: float(value) for name, value in amounts_kmol.items()}
    if component not in amounts or amounts[component] <= 0:
        raise ThermodynamicsError('Empirical layer feed must contain the crystallizing component')
    if any(not math.isfinite(value) or value < 0 for value in amounts.values()):
        raise ThermodynamicsError('Empirical layer amounts must be finite and nonnegative')
    unknown = set(amounts) - set(thermo.components)
    if unknown:
        raise ThermodynamicsError('Unknown empirical layer components: ' + ', '.join(sorted(unknown)))
    unknown_laws = set(distribution_laws) - set(amounts)
    if unknown_laws:
        raise ThermodynamicsError('Unknown empirical layer distribution components: ' + ', '.join(sorted(unknown_laws)))
    if component in distribution_laws:
        raise ThermodynamicsError('The crystallizing component cannot have an impurity k_eff law')
    melting = float(thermo.props[component].Tm)
    hfus = float(thermo.props[component].Hfus) * 1000.0
    if solid_density_kg_m3 is None:
        solid_volume = float(thermo._solid_molar_volume(component, wall))
        solid_density = thermo.props[component].MW / solid_volume
    else:
        solid_density = float(solid_density_kg_m3)
        if not math.isfinite(solid_density) or solid_density <= 0:
            raise ThermodynamicsError('Empirical layer solid density must be positive and finite')
        solid_volume = thermo.props[component].MW / solid_density
    initial_total = sum(amounts.values())
    initial_x = {name: value / initial_total for name, value in amounts.items()}
    initial_w = _mass_fractions(thermo, initial_x)
    impurities = tuple(
        name for name in distribution_laws if amounts.get(name, 0.0) > 0.0
    )
    active_distribution_laws = {
        name: distribution_laws[name] for name in impurities
    }
    evaluations = 0

    def evaluate(time_h, state):
        nonlocal evaluations
        evaluations += 1
        deposited = {
            component: min(amounts[component], max(0.0, float(state[0])))
        }
        deposited.update({
            name: min(amounts[name], max(0.0, float(state[index + 1])))
            for index, name in enumerate(impurities)
        })
        remaining = {
            name: max(0.0, amounts[name] - deposited.get(name, 0.0))
            for name in amounts
        }
        liquid_total = sum(remaining.values())
        if liquid_total <= 0:
            # Extend the RHS to trial steps beyond complete depletion so the
            # integrator can locate its terminal inventory event. Along this
            # ray the limiting liquid composition is the initial composition.
            x = dict(initial_x)
        else:
            x = {name: value / liquid_total for name, value in remaining.items()}
        w = _mass_fractions(thermo, x)
        tsat = crystallization_saturation_temperature(
            thermo, component, x, T, P, melting
        )
        activities = liquid_solution_activities(thermo, T, P, x)
        activity = max(float(activities[component]), 1e-300)
        asat = math.exp(pure_solid_log_saturation_activity(thermo, component, T, P))
        ratio = activity / asat
        recovery = deposited[component] / amounts[component]
        thickness = sum(deposited.values()) * solid_volume / area
        context = {
            'T': T, 'T_bulk': T, 'P': P,
            'Twall': wall, 'T_wall': wall, 'Tcool': wall, 'T_coolant': wall,
            'S': ratio, 'sigma': max(ratio - 1, 0),
            'relative_supersaturation': ratio - 1,
            'lnS': math.log(max(ratio, 1e-300)),
            'activity': activity, 'a': activity, 'a_sat': asat, 'asat': asat,
            'time': time_h, 't': time_h, 'time_h': time_h,
            'time_s': time_h * 3600, 'duration': duration,
            'duration_h': duration, 'thickness': thickness, 'L': thickness,
            'area': area, 'recovery': recovery,
            'x_crystal': x[component], 'w_crystal': w[component],
            'Tm': melting, 'Hfus': hfus, 'rho_s': solid_density,
            'Vm_solid': solid_volume, 'pi': math.pi, 'R': R,
            'x': x, 'w': w, 'x0': initial_x, 'w0': initial_w,
        }
        if tsat is not None:
            context.update({
                'Tsat': tsat, 'Teq': tsat, 'T_eq': tsat,
                'deltaT': tsat - wall, 'dT': tsat - wall,
                'deltaT_bulk': tsat - T,
            })
        context.update({
            'deltaT_wall': T - wall,
            'wall_undercooling': T - wall,
        })
        wall_activities = liquid_solution_activities(thermo, wall, P, x)
        wall_driving_force = (
            math.log(max(float(wall_activities.get(component, 0.0)), 1e-300))
            - pure_solid_log_saturation_activity(thermo, component, wall, P)
        )
        # A missing saturation-temperature root does not establish a driving
        # force. Check equilibrium at the wall before evaluating a growth law
        # that may only be defined for positive undercooling.
        growth = growth_law.evaluate(context) if wall_driving_force > 0 else 0.0
        context.update({'G': growth, 'G_m_h': growth, 'G_m_s': growth / 3600})
        layer_impurities = {}
        coefficients = {}
        for impurity, law in active_distribution_laws.items():
            impurity_context = {
                **context,
                'x_impurity': x.get(impurity, 0.0),
                'w_impurity': w.get(impurity, 0.0),
            }
            coefficient = law.evaluate(impurity_context)
            coefficients[impurity] = coefficient
            layer_impurities[impurity] = coefficient * x.get(impurity, 0.0)
        impurity_sum = sum(layer_impurities.values())
        if impurity_sum >= 1:
            raise ThermodynamicsError(
                'Empirical layer k_eff laws imply total impurity mole fraction >= 1'
            )
        layer_composition = {
            component: 1 - impurity_sum,
            **layer_impurities,
        }
        total_rate = area * growth / solid_volume
        rates = {name: total_rate * fraction for name, fraction in layer_composition.items()}
        return {
            'rates': rates, 'remaining': remaining, 'x': x, 'w': w,
            'context': context, 'growth': growth, 'coefficients': coefficients,
            'layer_composition': layer_composition, 'thickness': thickness,
            'recovery': recovery,
        }

    def rhs(time_h, state):
        values = evaluate(time_h, state)
        return [
            values['rates'].get(component, 0.0),
            *(values['rates'].get(name, 0.0) for name in impurities),
        ]

    events = []
    tracked = (component, *impurities)
    for index, name in enumerate(tracked):
        def inventory_event(_time, state, index=index, name=name):
            return amounts[name] - state[index]
        inventory_event.terminal = True
        inventory_event.direction = -1
        events.append(inventory_event)

    solved = solve_ivp(
        rhs, (0, duration), np.zeros(1 + len(impurities)),
        rtol=tolerance,
        atol=[max(1e-13, tolerance * amounts[name] * 1e-3) for name in tracked],
        events=events, dense_output=True, max_step=duration / 20,
    )
    if not solved.success:
        raise ThermodynamicsError(f'Empirical layer integration failed: {solved.message}')
    elapsed = float(solved.t[-1])
    final_state = solved.y[:, -1]
    times = np.linspace(0, elapsed, int(profile_points))
    states = solved.sol(times).T if solved.sol is not None else solved.y.T
    profile = []
    for time_h, state in zip(times, states, strict=True):
        values = evaluate(float(time_h), state)
        profile.append({
            'time_h': float(time_h),
            'time_s': float(time_h) * 3600,
            'layer_thickness_m': values['thickness'],
            'growth_rate_m_h': values['growth'],
            'growth_rate_m_s': values['growth'] / 3600,
            'crystal_recovery': values['recovery'],
            'bulk_temperature_K': T,
            'wall_temperature_K': wall,
            'saturation_temperature_K': values['context'].get('Tsat'),
            'undercooling_K': values['context'].get('deltaT'),
            'liquid_composition': dict(values['x']),
            'liquid_mass_composition': dict(values['w']),
            'effective_distribution_coefficients': dict(values['coefficients']),
            'instantaneous_layer_composition': dict(values['layer_composition']),
            'remaining_component_amounts_kmol': dict(values['remaining']),
        })
    deposited = {
        component: min(amounts[component], max(0.0, float(final_state[0])))
    }
    deposited.update({
        name: min(amounts[name], max(0.0, float(final_state[index + 1])))
        for index, name in enumerate(impurities)
    })
    remaining = {
        name: max(0.0, amounts[name] - deposited.get(name, 0.0))
        for name in amounts
    }
    return EmpiricalLayerGrowthResult(
        solid_amount_kmol=deposited[component],
        thickness_m=profile[-1]['layer_thickness_m'],
        trapped_component_amounts_kmol={
            name: deposited[name] for name in impurities if deposited[name] > 0
        },
        liquid_component_amounts_kmol=remaining,
        profile=profile,
        evaluations=evaluations,
        elapsed_growth_time_h=elapsed,
        stopped_by_inventory=elapsed < duration * (1 - 1e-12),
        solid_density_kg_m3=solid_density,
    )


__all__ = [
    'EmpiricalLayerDefinitionError',
    'EmpiricalLayerGrowthResult',
    'EmpiricalLayerLaw',
    'empirical_layer_distribution_definitions_from_parameters',
    'empirical_layer_distribution_law_from_mapping',
    'empirical_layer_growth_definition_from_parameters',
    'empirical_layer_growth_law_from_mapping',
    'solve_empirical_layer_growth',
]
