"""Steady mixed-suspension, mixed-product-removal crystallization models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .kinetic_models import KineticsError, SafeRateExpression
    from .particle_size_distributions import ParticleSizeDistribution
    from .reaction_models import ReactionDefinitionError
    from .thermodynamics_models.common import R
    from .thermodynamics_models.sle import (
        liquid_solution_activities,
        pure_solid_log_saturation_activity,
    )
else:
    from kinetic_models import KineticsError, SafeRateExpression
    from particle_size_distributions import ParticleSizeDistribution
    from reaction_models import ReactionDefinitionError
    from thermodynamics_models.common import R
    from thermodynamics_models.sle import (
        liquid_solution_activities,
        pure_solid_log_saturation_activity,
    )


class MSMPRDefinitionError(ValueError):
    """Raised when an MSMPR kinetic definition is invalid."""


class MSMPRConvergenceError(ValueError):
    """Raised when the coupled steady MSMPR balance has no solution."""


_GROWTH_RATE_FACTORS_M_PER_H = {
    'm/h': 1.0,
    'm/hr': 1.0,
    'm/s': 3600.0,
    'm/min': 60.0,
    'mm/h': 1.0e-3,
    'mm/hr': 1.0e-3,
    'mm/min': 0.06,
    'mm/s': 3.6,
    'um/h': 1.0e-6,
    'um/hr': 1.0e-6,
    'um/min': 6.0e-5,
    'um/s': 3.6e-3,
    'µm/h': 1.0e-6,
    'µm/min': 6.0e-5,
    'µm/s': 3.6e-3,
}

_NUCLEATION_RATE_FACTORS_PER_M3_H = {
    '1/m3/h': 1.0,
    '1/m^3/h': 1.0,
    '#/m3/h': 1.0,
    'particles/m3/h': 1.0,
    '1/m3/s': 3600.0,
    '1/m^3/s': 3600.0,
    '#/m3/s': 3600.0,
    'particles/m3/s': 3600.0,
    '1/l/h': 1000.0,
    '#/l/h': 1000.0,
    '1/l/s': 3.6e6,
    '#/l/s': 3.6e6,
}

_EXPRESSION_SCALARS = frozenset({
    'T', 'P', 'S', 'sigma', 'relative_supersaturation', 'lnS',
    'MT', 'L', 'age', 'tau', 'V', 'Q', 'x', 'C', 'activity',
    'a', 'a_sat', 'asat', 'G0', 'G0_m_h',
    'Tsat', 'deltaT', 'dT', 'deltaT_reduced', 'deltaT_fusion',
    'Tm', 'Hfus', 'pi', 'R',
})

_EXPRESSION_FUNCTIONS = frozenset({
    'abs', 'exp', 'log', 'log10', 'max', 'min', 'sqrt',
})


def _unit_token(value: object) -> str:
    return str(value).strip().lower().replace(' ', '')


def _finite_number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise MSMPRDefinitionError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise MSMPRDefinitionError(f"{label} must be finite")
    return number


def _one_field(
    definition: Mapping[str, object],
    names: tuple[str, ...],
    label: str,
    *,
    required: bool = True,
    default: object = None,
) -> object:
    found = [name for name in names if name in definition]
    if len(found) > 1:
        raise MSMPRDefinitionError(
            f"MSMPR {label} received duplicate aliases: " + ', '.join(found)
        )
    if not found:
        if required:
            raise MSMPRDefinitionError(f"MSMPR {label} is required")
        return default
    return definition[found[0]]


@dataclass(frozen=True)
class MSMPRRateLaw:
    """Validated nucleation or linear-growth rate expression."""

    kind: str
    model: str
    expression: SafeRateExpression
    parameters: dict[str, float]
    declared_rate_unit: str
    canonical_factor: float

    @property
    def canonical_rate_unit(self) -> str:
        return 'm/h' if self.kind == 'growth' else '1/m3/h'

    @property
    def depends_on_size(self) -> bool:
        return bool({'L', 'age'} & self.expression.names)

    def evaluate(self, context: Mapping[str, object]) -> float:
        if 'S' in context and float(context['S']) <= 1.0:
            return 0.0
        values = dict(context)
        values.update(self.parameters)
        missing = sorted(self.expression.names - set(values))
        if missing:
            raise MSMPRConvergenceError(
                f"MSMPR {self.kind} expression requires unavailable variable(s): "
                + ', '.join(missing)
            )
        try:
            raw_value = self.expression.evaluate(values)
        except (KineticsError, KeyError) as error:
            raise MSMPRConvergenceError(
                f"MSMPR {self.kind} expression failed: {error}"
            ) from error
        value = raw_value * self.canonical_factor
        if not math.isfinite(value) or value < 0.0:
            raise MSMPRConvergenceError(
                f"MSMPR {self.kind} expression must return a finite "
                "nonnegative rate"
            )
        return value


def msmpr_rate_law_from_mapping(
    definition: Mapping[str, object],
    kind: str,
) -> MSMPRRateLaw:
    """Create an MSMPR nucleation or growth law from a PFD-style mapping."""
    if not isinstance(definition, Mapping):
        raise MSMPRDefinitionError(f"MSMPR {kind} definition must be a mapping")
    rate_kind = str(kind).strip().lower()
    if rate_kind not in {'growth', 'nucleation'}:
        raise MSMPRDefinitionError("MSMPR rate kind must be growth or nucleation")
    fields = {str(key): value for key, value in definition.items()}
    lower_lookup = {key.lower(): key for key in fields}

    def get(name, default=None):
        key = lower_lookup.get(name.lower())
        return fields[key] if key is not None else default

    model_fields = [name for name in ('model', 'type') if get(name) is not None]
    if len(model_fields) > 1:
        raise MSMPRDefinitionError(
            f"MSMPR {rate_kind} accepts only one model/type field"
        )
    explicit_model = get(model_fields[0]) if model_fields else None
    expression_value = get('expression')
    if explicit_model is None:
        model = 'custom' if expression_value is not None else 'power_law'
    else:
        model = str(explicit_model).strip().lower().replace('-', '_')
    aliases = {
        'primary': 'primary_power_law',
        'primary_powerlaw': 'primary_power_law',
        'secondary': 'secondary_power_law',
        'secondary_powerlaw': 'secondary_power_law',
        'powerlaw': 'power_law',
        'size_dependent': 'custom',
    }
    model = aliases.get(model, model)
    if rate_kind == 'nucleation' and model == 'power_law':
        model = 'primary_power_law'
    allowed_models = (
        {'power_law', 'custom'}
        if rate_kind == 'growth'
        else {'power_law', 'primary_power_law', 'secondary_power_law', 'custom'}
    )
    if model not in allowed_models:
        raise MSMPRDefinitionError(
            f"Unsupported MSMPR {rate_kind} model {model!r}; expected "
            + ', '.join(sorted(allowed_models))
        )
    if model != 'custom' and expression_value is not None:
        raise MSMPRDefinitionError(
            f"MSMPR {rate_kind} expression requires model=custom"
        )

    common_fields = {'model', 'type', 'expression', 'rate_unit', 'unit', 'parameters'}
    preset_fields = {
        'coefficient', 'k',
        'supersaturation_exponent', 'exponent',
    }
    if rate_kind == 'growth':
        preset_fields.update({'kg', 'g'})
    else:
        preset_fields.update({'kb', 'b'})
        if model == 'secondary_power_law':
            preset_fields.update({
                'suspension_exponent', 'mt_exponent', 'j',
                'growth_exponent', 'i',
            })
    allowed_fields = common_fields | (
        set() if model == 'custom' else preset_fields
    )
    unexpected = sorted(
        raw_key for raw_key in fields
        if raw_key.lower() not in allowed_fields
        and not raw_key.lower().startswith('param_')
    )
    if unexpected:
        raise MSMPRDefinitionError(
            f"MSMPR {rate_kind} does not accept field(s): "
            + ', '.join(unexpected)
        )

    unit_value = get('rate_unit', get('unit'))
    if unit_value is None or not str(unit_value).strip():
        raise MSMPRDefinitionError(
            f"MSMPR {rate_kind} requires explicit rate_unit"
        )
    unit = _unit_token(unit_value)
    factors = (
        _GROWTH_RATE_FACTORS_M_PER_H
        if rate_kind == 'growth'
        else _NUCLEATION_RATE_FACTORS_PER_M3_H
    )
    if unit not in factors:
        raise MSMPRDefinitionError(
            f"Unsupported MSMPR {rate_kind} rate_unit={unit_value!r}; "
            "supported values are " + ', '.join(sorted(factors))
        )

    parameters = {}
    nested_parameters = get('parameters', {}) or {}
    if not isinstance(nested_parameters, Mapping):
        raise MSMPRDefinitionError(
            f"MSMPR {rate_kind} parameters must be a mapping"
        )
    if model != 'custom' and nested_parameters:
        raise MSMPRDefinitionError(
            f"MSMPR {rate_kind} parameters mapping requires model=custom"
        )
    for raw_name, raw_value in nested_parameters.items():
        parameters[str(raw_name)] = _finite_number(
            raw_value, f"MSMPR {rate_kind} parameter {raw_name}"
        )
    for raw_key, raw_value in fields.items():
        if raw_key.lower().startswith('param_'):
            if model != 'custom':
                raise MSMPRDefinitionError(
                    f"MSMPR {rate_kind} param_* values require model=custom"
                )
            name = raw_key[6:]
            if name in parameters:
                raise MSMPRDefinitionError(
                    f"Duplicate MSMPR {rate_kind} parameter {name!r}"
                )
            parameters[name] = _finite_number(
                raw_value, f"MSMPR {rate_kind} parameter {name}"
            )

    if model == 'custom':
        if expression_value is None:
            raise MSMPRDefinitionError(
                f"Custom MSMPR {rate_kind} kinetics requires expression"
            )
        expression_text = str(expression_value).strip().strip('"\'')
    else:
        coefficient_names = (
            ('coefficient', 'k', 'kg')
            if rate_kind == 'growth'
            else ('coefficient', 'k', 'kb')
        )
        normalized = {
            lower: fields[key] for lower, key in lower_lookup.items()
        }
        coefficient = _finite_number(
            _one_field(normalized, coefficient_names, f"{rate_kind} coefficient"),
            f"MSMPR {rate_kind} coefficient",
        )
        exponent = _finite_number(
            _one_field(
                normalized,
                ('supersaturation_exponent', 'exponent', 'g' if rate_kind == 'growth' else 'b'),
                f"{rate_kind} supersaturation exponent",
            ),
            f"MSMPR {rate_kind} supersaturation exponent",
        )
        if coefficient < 0.0:
            raise MSMPRDefinitionError(
                f"MSMPR {rate_kind} coefficient must be nonnegative"
            )
        if exponent < 0.0:
            raise MSMPRDefinitionError(
                f"MSMPR {rate_kind} supersaturation exponent must be nonnegative"
            )
        parameters.update({'coefficient': coefficient, 'exponent': exponent})
        expression_text = 'coefficient * sigma**exponent'
        if model == 'secondary_power_law':
            suspension_exponent = _finite_number(
                _one_field(
                    normalized,
                    ('suspension_exponent', 'mt_exponent', 'j'),
                    'nucleation suspension-density exponent',
                ),
                'MSMPR nucleation suspension-density exponent',
            )
            parameters['suspension_exponent'] = suspension_exponent
            if suspension_exponent < 0.0:
                raise MSMPRDefinitionError(
                    "MSMPR nucleation suspension-density exponent must be "
                    "nonnegative"
                )
            growth_exponent = _finite_number(
                _one_field(
                    normalized,
                    ('growth_exponent', 'i'),
                    'nucleation growth-rate exponent',
                    required=False,
                    default=0.0,
                ),
                'MSMPR nucleation growth-rate exponent',
            )
            parameters['growth_exponent'] = growth_exponent
            if growth_exponent != 0.0:
                expression_text += ' * G0**growth_exponent'
            expression_text += ' * MT**suspension_exponent'

    for name in parameters:
        if (
            not name.isidentifier()
            or name in _EXPRESSION_SCALARS
            or name in _EXPRESSION_FUNCTIONS
        ):
            raise MSMPRDefinitionError(
                f"Invalid MSMPR {rate_kind} parameter name {name!r}"
            )
    scalar_names = set(_EXPRESSION_SCALARS)
    if rate_kind == 'nucleation':
        scalar_names -= {'L', 'age'}
    else:
        scalar_names -= {'G0', 'G0_m_h'}
    try:
        expression = SafeRateExpression(
            expression_text,
            parameters,
            scalar_names=scalar_names,
            mapping_names=(),
            function_names=_EXPRESSION_FUNCTIONS,
        )
    except ReactionDefinitionError as error:
        raise MSMPRDefinitionError(
            f"Invalid MSMPR {rate_kind} expression: {error}"
        ) from error
    return MSMPRRateLaw(
        kind=rate_kind,
        model=model,
        expression=expression,
        parameters=parameters,
        declared_rate_unit=unit,
        canonical_factor=factors[unit],
    )


def msmpr_rate_definition_from_parameters(
    parameters: Mapping[str, object],
    kind: str,
) -> dict[str, object]:
    """Collect a nested or flattened MSMPR rate definition from unit params."""
    aliases = ('growth', 'g') if kind == 'growth' else ('nucleation', 'b0')
    lower_lookup = {
        str(key).lower(): key
        for key in parameters
        if not str(key).startswith('__unit__')
    }
    direct = [
        (name, parameters[lower_lookup[name]])
        for name in aliases if name in lower_lookup
    ]
    if len(direct) > 1:
        raise MSMPRDefinitionError(
            f"MSMPR received duplicate {kind} definitions"
        )
    if direct:
        if not isinstance(direct[0][1], Mapping):
            raise MSMPRDefinitionError(
                f"MSMPR {direct[0][0]} must be a mapping"
            )
        definition = dict(direct[0][1])
    else:
        definition = {}
    definition_names = {str(key).lower(): key for key in definition}
    for lower_key, raw_key in lower_lookup.items():
        matches = [
            alias for alias in aliases
            if lower_key.startswith(alias + '_')
        ]
        if not matches:
            continue
        alias = max(matches, key=len)
        field_name = str(raw_key)[len(alias) + 1:]
        canonical = 'rate_unit' if field_name.lower() == 'unit' else field_name
        if canonical.lower() in definition_names:
            raise MSMPRDefinitionError(
                f"MSMPR received duplicate {kind} field {canonical!r}"
            )
        definition[canonical] = parameters[raw_key]
        definition_names[canonical.lower()] = canonical
    if not definition:
        raise MSMPRDefinitionError(f"MSMPR mode requires {kind} kinetics")
    return definition


@dataclass(frozen=True)
class SteadyMSMPRResult:
    component: str
    solid_flow_kmol_h: float
    liquid_component_flows_kmol_h: dict[str, float]
    liquid_composition: dict[str, float]
    particle_size_distribution: Optional[ParticleSizeDistribution]
    saturation_ratio: float
    relative_supersaturation: float
    log_saturation_ratio: float
    liquid_activity: float
    saturation_activity: float
    saturation_temperature_K: Optional[float]
    undercooling_K: Optional[float]
    reduced_undercooling: Optional[float]
    fusion_scaled_undercooling: Optional[float]
    melting_temperature_K: float
    heat_of_fusion_J_mol: float
    solute_concentration_kmol_m3: float
    suspension_density_kg_m3: float
    nucleation_rate_per_m3_h: float
    growth_rate_range_m_h: tuple[float, float]
    birth_growth_rate_m_h: float
    residence_time_h: float
    volume_m3: float
    volumetric_flow_m3_h: float
    nucleated_particle_rate_per_h: float
    seed_particle_rate_per_h: float
    total_particle_rate_per_h: float
    number_mean_diameter_m: float
    material_residual_kmol_h: float
    absolute_residual_tolerance_kmol_h: float
    relative_residual_tolerance: float
    effective_residual_tolerance_kmol_h: float
    iterations: int


def _base_context(
    *,
    T: float,
    P: float,
    saturation_ratio: float,
    MT: float,
    tau: float,
    volume: float,
    volumetric_flow: float,
    mole_fraction: float,
    concentration: float,
    activity: float,
    saturation_activity: float,
    saturation_temperature: Optional[float],
    melting_temperature: float,
    heat_of_fusion_J_mol: float,
) -> dict[str, float]:
    signed_relative = saturation_ratio - 1.0
    context = {
        'T': T,
        'P': P,
        'S': saturation_ratio,
        'sigma': max(0.0, signed_relative),
        'relative_supersaturation': signed_relative,
        'lnS': math.log(max(saturation_ratio, 1.0e-300)),
        'MT': MT,
        'tau': tau,
        'V': volume,
        'Q': volumetric_flow,
        'x': mole_fraction,
        'C': concentration,
        'activity': activity,
        'a': activity,
        'a_sat': saturation_activity,
        'asat': saturation_activity,
        'Tm': melting_temperature,
        'Hfus': heat_of_fusion_J_mol,
        'pi': math.pi,
        'R': R,
    }
    if saturation_temperature is not None:
        undercooling = saturation_temperature - T
        context.update({
            'Tsat': saturation_temperature,
            'deltaT': undercooling,
            'dT': undercooling,
            'deltaT_reduced': undercooling / saturation_temperature,
            'deltaT_fusion': (
                heat_of_fusion_J_mol
                * undercooling
                / (R * T * melting_temperature)
            ),
        })
    return context


def _saturation_temperature(
    thermo,
    component: str,
    composition: Mapping[str, float],
    T: float,
    P: float,
    melting_temperature: float,
) -> Optional[float]:
    """Return the temperature at which the fixed liquid composition saturates."""
    from scipy.optimize import brentq

    def residual(temperature):
        activities = liquid_solution_activities(
            thermo, temperature, P, dict(composition)
        )
        activity = max(float(activities.get(component, 0.0)), 1.0e-300)
        return (
            math.log(activity)
            - pure_solid_log_saturation_activity(
                thermo, component, temperature, P
            )
        )

    current = residual(T)
    if abs(current) <= 1.0e-12:
        return float(T)
    if current > 0.0:
        candidates = [
            T + (melting_temperature - T) * index / 32.0
            for index in range(1, 33)
        ]
    else:
        lower = max(1.0, 0.25 * T)
        candidates = [
            T - (T - lower) * index / 64.0
            for index in range(1, 65)
        ]
    previous_temperature = T
    previous_residual = current
    for candidate in candidates:
        try:
            candidate_residual = residual(candidate)
        except Exception:
            continue
        if previous_residual * candidate_residual <= 0.0:
            low, high = sorted((previous_temperature, candidate))
            return float(brentq(
                residual,
                low,
                high,
                xtol=1.0e-10,
                rtol=1.0e-12,
                maxiter=100,
            ))
        previous_temperature = candidate
        previous_residual = candidate_residual
    return None


def _growth_diameters(
    growth_law: MSMPRRateLaw,
    context: Mapping[str, float],
    initial_diameter_m: float,
    ages_h,
) -> tuple[tuple[float, ...], tuple[float, float]]:
    initial = float(initial_diameter_m)
    if initial < 0.0 or not math.isfinite(initial):
        raise MSMPRDefinitionError(
            "MSMPR initial particle diameter must be finite and nonnegative"
        )
    ages = tuple(float(value) for value in ages_h)
    if not ages:
        return (), (0.0, 0.0)

    def rate(length: float, age: float) -> float:
        local = dict(context)
        local.update({'L': max(0.0, float(length)), 'age': float(age)})
        return growth_law.evaluate(local)

    if not growth_law.depends_on_size:
        growth_rate = rate(initial, 0.0)
        return (
            tuple(initial + growth_rate * age for age in ages),
            (growth_rate, growth_rate),
        )

    from scipy.integrate import solve_ivp

    observed_rates = []

    def derivative(age, values):
        value = rate(float(values[0]), float(age))
        observed_rates.append(value)
        return [value]

    solved = solve_ivp(
        derivative,
        (0.0, max(ages)),
        [initial],
        t_eval=ages,
        rtol=1.0e-8,
        atol=1.0e-15,
        max_step=max(max(ages) / 100.0, 1.0e-12),
    )
    if not solved.success or len(solved.y[0]) != len(ages):
        raise MSMPRConvergenceError(
            "MSMPR size-dependent growth integration failed: "
            + str(solved.message)
        )
    diameters = tuple(float(value) for value in solved.y[0])
    if any(not math.isfinite(value) or value < initial for value in diameters):
        raise MSMPRConvergenceError(
            "MSMPR growth integration produced an invalid particle diameter"
        )
    rates = observed_rates or [rate(initial, 0.0)]
    return diameters, (min(rates), max(rates))


def _seed_number_rates(
    distribution: Optional[ParticleSizeDistribution],
    solid_molar_volume_m3_kmol: float,
) -> tuple[tuple[float, float], ...]:
    if distribution is None:
        return ()
    cohorts = []
    for diameter, component_flow in zip(
        distribution.diameters_m,
        distribution.molar_flows_kmol_per_h,
    ):
        particle_volume = math.pi * diameter**3 / 6.0
        number_rate = component_flow * solid_molar_volume_m3_kmol / particle_volume
        cohorts.append((diameter, number_rate))
    return tuple(cohorts)


def _population_for_conditions(
    *,
    growth_law: MSMPRRateLaw,
    context: Mapping[str, float],
    residence_time_h: float,
    volume_m3: float,
    nucleation_rate_per_m3_h: float,
    nucleus_diameter_m: float,
    quadrature_classes: int,
    maximum_output_classes: int,
    solid_molar_volume_m3_kmol: float,
    seed_distribution: Optional[ParticleSizeDistribution],
) -> tuple[Optional[ParticleSizeDistribution], dict[str, float]]:
    from scipy.special import roots_laguerre

    nodes, weights = roots_laguerre(int(quadrature_classes))
    ages = tuple(float(node) * residence_time_h for node in nodes)
    seed_cohorts = _seed_number_rates(
        seed_distribution, solid_molar_volume_m3_kmol
    )
    cohorts = [
        (diameter, rate, 'seed') for diameter, rate in seed_cohorts
    ]
    seed_particle_rate = sum(rate for _diameter, rate in seed_cohorts)
    nucleated_particle_rate = nucleation_rate_per_m3_h * volume_m3
    if nucleated_particle_rate > 0.0:
        cohorts.append((nucleus_diameter_m, nucleated_particle_rate, 'nucleated'))

    by_diameter: dict[float, list[float]] = {}
    minimum_growth_rate = math.inf
    maximum_growth_rate = 0.0
    total_number_rate = 0.0
    number_diameter_sum = 0.0
    for initial_diameter, cohort_rate, source in cohorts:
        if cohort_rate <= 0.0:
            continue
        diameters, rate_range = _growth_diameters(
            growth_law, context, initial_diameter, ages
        )
        if source == 'nucleated' and not any(
            diameter > 0.0 for diameter in diameters
        ):
            raise MSMPRConvergenceError(
                "MSMPR has positive nucleation but newborn particles have zero "
                "diameter and zero growth; specify a positive nucleus_diameter "
                "or a growth law with G(L=0) > 0"
            )
        minimum_growth_rate = min(minimum_growth_rate, rate_range[0])
        maximum_growth_rate = max(maximum_growth_rate, rate_range[1])
        for diameter, raw_weight in zip(diameters, weights):
            weight = float(raw_weight)
            number_rate = cohort_rate * weight
            if diameter <= 0.0 or number_rate <= 0.0:
                continue
            particle_volume = math.pi * diameter**3 / 6.0
            component_flow = (
                number_rate * particle_volume / solid_molar_volume_m3_kmol
            )
            canonical_diameter = float(f"{diameter:.15g}")
            accumulated = by_diameter.setdefault(
                canonical_diameter, [0.0, 0.0]
            )
            accumulated[0] += component_flow
            accumulated[1] += number_rate
            total_number_rate += number_rate
            number_diameter_sum += number_rate * diameter

    distribution = None
    if by_diameter:
        ordered = [
            (diameter, values[0], values[1])
            for diameter, values in sorted(by_diameter.items())
        ]
        if len(ordered) > maximum_output_classes:
            ordered = _rebin_population_by_number(
                ordered,
                maximum_output_classes,
                solid_molar_volume_m3_kmol,
            )
        distribution = ParticleSizeDistribution(
            tuple(diameter for diameter, _flow, _number in ordered),
            tuple(flow for _diameter, flow, _number in ordered),
        )
    if minimum_growth_rate == math.inf:
        probe = dict(context)
        probe.update({'L': nucleus_diameter_m, 'age': 0.0})
        rate = growth_law.evaluate(probe)
        minimum_growth_rate = maximum_growth_rate = rate
    return distribution, {
        'minimum_growth_rate_m_h': minimum_growth_rate,
        'maximum_growth_rate_m_h': maximum_growth_rate,
        'nucleated_particle_rate_per_h': nucleated_particle_rate,
        'seed_particle_rate_per_h': seed_particle_rate,
        'total_particle_rate_per_h': total_number_rate,
        'number_mean_diameter_m': (
            number_diameter_sum / total_number_rate
            if total_number_rate > 0.0 else 0.0
        ),
    }


def _rebin_population_by_number(
    entries: list[tuple[float, float, float]],
    maximum_classes: int,
    solid_molar_volume_m3_kmol: float,
) -> list[tuple[float, float, float]]:
    """Rebin ordered cohorts while conserving particle number and volume."""
    total_number = sum(number for _diameter, _flow, number in entries)
    if total_number <= 0.0:
        return entries
    target = total_number / maximum_classes
    groups = []
    group_number = 0.0
    group_flow = 0.0

    def finish_group():
        nonlocal group_number, group_flow
        if group_number <= 0.0 or group_flow <= 0.0:
            return
        diameter = (
            6.0
            * group_flow
            * solid_molar_volume_m3_kmol
            / (math.pi * group_number)
        ) ** (1.0 / 3.0)
        groups.append((float(f"{diameter:.15g}"), group_flow, group_number))
        group_number = 0.0
        group_flow = 0.0

    for _diameter, raw_flow, raw_number in entries:
        remaining_number = raw_number
        remaining_flow = raw_flow
        while remaining_number > 0.0:
            capacity = target - group_number
            if capacity <= max(1.0e-15 * target, 1.0e-300):
                finish_group()
                capacity = target
            taken_number = min(remaining_number, capacity)
            fraction = taken_number / remaining_number
            taken_flow = remaining_flow * fraction
            group_number += taken_number
            group_flow += taken_flow
            remaining_number -= taken_number
            remaining_flow -= taken_flow
            if group_number >= target * (1.0 - 1.0e-14):
                finish_group()
    finish_group()

    while len(groups) > maximum_classes:
        _left_diameter, left_flow, left_number = groups[-2]
        _right_diameter, right_flow, right_number = groups[-1]
        combined_flow = left_flow + right_flow
        combined_number = left_number + right_number
        combined_diameter = (
            6.0
            * combined_flow
            * solid_molar_volume_m3_kmol
            / (math.pi * combined_number)
        ) ** (1.0 / 3.0)
        groups[-2:] = [(
            float(f"{combined_diameter:.15g}"),
            combined_flow,
            combined_number,
        )]

    merged: dict[float, list[float]] = {}
    for diameter, flow, number in groups:
        values = merged.setdefault(diameter, [0.0, 0.0])
        values[0] += flow
        values[1] += number
    return [
        (diameter, values[0], values[1])
        for diameter, values in sorted(merged.items())
    ]


def solve_steady_msmpr(
    thermo,
    *,
    component: str,
    total_component_flows_kmol_h: Mapping[str, float],
    T: float,
    P: float,
    growth_law: MSMPRRateLaw,
    nucleation_law: MSMPRRateLaw,
    residence_time_h: Optional[float] = None,
    volume_m3: Optional[float] = None,
    seed_solid_flow_kmol_h: float = 0.0,
    seed_distribution: Optional[ParticleSizeDistribution] = None,
    nucleus_diameter_m: float = 0.0,
    quadrature_classes: int = 20,
    maximum_output_classes: int = 200,
    residual_tolerance: float = 1.0e-8,
    relative_residual_tolerance: float = 0.0,
    max_iterations: int = 200,
) -> SteadyMSMPRResult:
    """Solve one steady ideal-MSMPR population and solute material balance."""
    from scipy.optimize import brentq

    component = str(component)
    if component not in set(thermo.conventional_solid_components):
        raise MSMPRDefinitionError(
            f"MSMPR component {component!r} must be conventional_with_solid"
        )
    total_flows = {
        name: float(total_component_flows_kmol_h.get(name, 0.0))
        for name in thermo.components
    }
    if any(not math.isfinite(flow) or flow < 0.0 for flow in total_flows.values()):
        raise MSMPRDefinitionError(
            "MSMPR total component flows must be finite and nonnegative"
        )
    available = total_flows.get(component, 0.0)
    if available <= 1.0e-15:
        raise MSMPRDefinitionError(
            f"MSMPR feed contains no crystallizing component {component!r}"
        )
    seed_flow = float(seed_solid_flow_kmol_h)
    if not math.isfinite(seed_flow) or seed_flow < 0.0 or seed_flow > available:
        raise MSMPRDefinitionError(
            "MSMPR seed solid flow must be finite, nonnegative, and no greater "
            "than total crystallizing-component flow"
        )
    if seed_flow > 1.0e-15:
        if seed_distribution is None:
            raise MSMPRDefinitionError(
                "A solid-bearing MSMPR feed requires a complete PSD for its seed solid"
            )
        tolerance = max(1.0e-12, seed_flow * 1.0e-10)
        if abs(seed_distribution.total_molar_flow - seed_flow) > tolerance:
            raise MSMPRDefinitionError(
                "MSMPR inlet seed PSD must account for the complete inlet solid flow"
            )
    elif seed_distribution is not None:
        raise MSMPRDefinitionError(
            "MSMPR inlet seed PSD has no corresponding inlet solid flow"
        )

    temperature = float(T)
    pressure = float(P)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise MSMPRDefinitionError("MSMPR temperature must be positive and finite")
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise MSMPRDefinitionError("MSMPR pressure must be positive and finite")
    if (residence_time_h is None) == (volume_m3 is None):
        raise MSMPRDefinitionError(
            "MSMPR requires exactly one of residence_time_h or volume_m3"
        )
    fixed_residence_time = (
        float(residence_time_h) if residence_time_h is not None else None
    )
    fixed_volume = float(volume_m3) if volume_m3 is not None else None
    if fixed_residence_time is not None and (
        not math.isfinite(fixed_residence_time) or fixed_residence_time <= 0.0
    ):
        raise MSMPRDefinitionError(
            "MSMPR residence time must be positive and finite"
        )
    if fixed_volume is not None and (
        not math.isfinite(fixed_volume) or fixed_volume <= 0.0
    ):
        raise MSMPRDefinitionError("MSMPR volume must be positive and finite")
    classes = int(quadrature_classes)
    if float(quadrature_classes) != classes or not 2 <= classes <= 200:
        raise MSMPRDefinitionError(
            "MSMPR quadrature_classes must be an integer from 2 to 200"
        )
    output_classes = int(maximum_output_classes)
    if (
        float(maximum_output_classes) != output_classes
        or not 2 <= output_classes <= 1000
    ):
        raise MSMPRDefinitionError(
            "MSMPR maximum_output_classes must be an integer from 2 to 1000"
        )
    tolerance = float(residual_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise MSMPRDefinitionError(
            "MSMPR residual_tolerance must be positive and finite"
        )
    relative_tolerance = float(relative_residual_tolerance)
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise MSMPRDefinitionError(
            "MSMPR relative_residual_tolerance must be nonnegative and finite"
        )
    iterations_limit = int(max_iterations)
    if iterations_limit <= 0:
        raise MSMPRDefinitionError("MSMPR max_iterations must be positive")

    props = thermo.props.get(component)
    molecular_weight = float(getattr(props, 'MW', 0.0) or 0.0)
    if not math.isfinite(molecular_weight) or molecular_weight <= 0.0:
        raise MSMPRDefinitionError(
            f"MSMPR component {component!r} requires molecular weight"
        )
    melting_temperature = getattr(props, 'Tm', None)
    try:
        melting_temperature = float(melting_temperature)
    except (TypeError, ValueError) as error:
        raise MSMPRDefinitionError(
            f"MSMPR component {component!r} requires a valid melting point"
        ) from error
    if not math.isfinite(melting_temperature) or melting_temperature <= 0.0:
        raise MSMPRDefinitionError(
            f"MSMPR component {component!r} requires a valid melting point"
        )
    if temperature >= melting_temperature:
        raise MSMPRDefinitionError(
            f"MSMPR temperature {temperature:g} K is not below the melting "
            f"point of {component!r} ({melting_temperature:g} K)"
        )
    try:
        heat_of_fusion = float(getattr(props, 'Hfus', 0.0) or 0.0)
    except (TypeError, ValueError) as error:
        raise MSMPRDefinitionError(
            f"MSMPR component {component!r} requires positive heat of fusion"
        ) from error
    if not math.isfinite(heat_of_fusion) or heat_of_fusion <= 0.0:
        raise MSMPRDefinitionError(
            f"MSMPR component {component!r} requires positive heat of fusion"
        )
    heat_of_fusion_J_mol = 1000.0 * heat_of_fusion
    solid_molar_volume = float(thermo._solid_molar_volume(component, temperature))
    if not math.isfinite(solid_molar_volume) or solid_molar_volume <= 0.0:
        raise MSMPRDefinitionError(
            f"MSMPR component {component!r} requires solid molar volume"
        )
    log_saturation_activity = pure_solid_log_saturation_activity(
        thermo, component, temperature, pressure
    )
    saturation_activity = math.exp(min(log_saturation_activity, 700.0))
    evaluation_count = 0
    last_result = None

    undercooling_names = {
        'Tsat', 'deltaT', 'dT', 'deltaT_reduced', 'deltaT_fusion',
    }
    kinetics_require_undercooling = bool(
        undercooling_names
        & (growth_law.expression.names | nucleation_law.expression.names)
    )

    def evaluate(solid_flow: float, *, include_thermal_diagnostics=False):
        nonlocal evaluation_count, last_result
        evaluation_count += 1
        liquid_flows = dict(total_flows)
        liquid_flows[component] = max(0.0, available - float(solid_flow))
        liquid_total = sum(liquid_flows.values())
        if liquid_total <= 1.0e-15:
            raise MSMPRConvergenceError(
                "MSMPR trial reached an all-solid state; liquid mother liquor is required"
            )
        composition = {
            name: flow / liquid_total
            for name, flow in liquid_flows.items()
            if flow > 0.0
        }
        activities = liquid_solution_activities(
            thermo, temperature, pressure, composition
        )
        activity = max(float(activities.get(component, 0.0)), 0.0)
        saturation_ratio = activity / max(saturation_activity, 1.0e-300)
        saturation_temperature = None
        if kinetics_require_undercooling or include_thermal_diagnostics:
            saturation_temperature = _saturation_temperature(
                thermo,
                component,
                composition,
                temperature,
                pressure,
                melting_temperature,
            )
        density = float(thermo.mixture_molar_density(
            composition,
            temperature,
            pressure,
            0.0,
            x=composition,
        ))
        if not math.isfinite(density) or density <= 0.0:
            raise MSMPRConvergenceError(
                "MSMPR thermodynamic model returned nonpositive liquid molar density"
            )
        concentration = composition.get(component, 0.0) * density
        volumetric_flow = (
            liquid_total / density + float(solid_flow) * solid_molar_volume
        )
        if not math.isfinite(volumetric_flow) or volumetric_flow <= 0.0:
            raise MSMPRConvergenceError(
                "MSMPR trial produced nonpositive product volumetric flow"
            )
        residence_time = (
            fixed_residence_time
            if fixed_residence_time is not None
            else fixed_volume / volumetric_flow
        )
        volume = (
            fixed_volume
            if fixed_volume is not None
            else residence_time * volumetric_flow
        )
        suspension_density = float(solid_flow) * molecular_weight / volumetric_flow
        context = _base_context(
            T=temperature,
            P=pressure,
            saturation_ratio=saturation_ratio,
            MT=suspension_density,
            tau=residence_time,
            volume=volume,
            volumetric_flow=volumetric_flow,
            mole_fraction=composition.get(component, 0.0),
            concentration=concentration,
            activity=activity,
            saturation_activity=saturation_activity,
            saturation_temperature=saturation_temperature,
            melting_temperature=melting_temperature,
            heat_of_fusion_J_mol=heat_of_fusion_J_mol,
        )
        growth_context = dict(context)
        growth_context.update({
            'L': float(nucleus_diameter_m),
            'age': 0.0,
        })
        birth_growth_rate = growth_law.evaluate(growth_context)
        context.update({
            'G0': birth_growth_rate,
            'G0_m_h': birth_growth_rate,
        })
        nucleation_rate = nucleation_law.evaluate(context)
        distribution, population = _population_for_conditions(
            growth_law=growth_law,
            context=context,
            residence_time_h=residence_time,
            volume_m3=volume,
            nucleation_rate_per_m3_h=nucleation_rate,
            nucleus_diameter_m=float(nucleus_diameter_m),
            quadrature_classes=classes,
            maximum_output_classes=output_classes,
            solid_molar_volume_m3_kmol=solid_molar_volume,
            seed_distribution=seed_distribution,
        )
        predicted_solid = (
            distribution.total_molar_flow if distribution is not None else 0.0
        )
        last_result = {
            'solid_flow': float(solid_flow),
            'predicted_solid': predicted_solid,
            'liquid_flows': liquid_flows,
            'composition': composition,
            'activity': activity,
            'saturation_ratio': saturation_ratio,
            'saturation_temperature': saturation_temperature,
            'concentration': concentration,
            'suspension_density': suspension_density,
            'nucleation_rate': nucleation_rate,
            'birth_growth_rate': birth_growth_rate,
            'distribution': distribution,
            'population': population,
            'residence_time': residence_time,
            'volume': volume,
            'volumetric_flow': volumetric_flow,
        }
        return predicted_solid - float(solid_flow)

    flow_tolerance = max(tolerance, relative_tolerance * available)
    lower = seed_flow
    upper = available * (1.0 - 1.0e-12)
    if upper < lower:
        upper = lower
    f_lower = evaluate(lower)
    endpoint_tolerance = max(1.0e-15, available * 1.0e-14)
    if abs(f_lower) <= endpoint_tolerance:
        solved_solid = lower
    else:
        bracket = None
        previous_x = lower
        previous_value = f_lower
        for index in range(1, 65):
            candidate = lower + (upper - lower) * index / 64.0
            candidate_value = evaluate(candidate)
            if previous_value * candidate_value <= 0.0:
                bracket = (previous_x, candidate)
                break
            previous_x = candidate
            previous_value = candidate_value
        if bracket is None:
            raise MSMPRConvergenceError(
                "MSMPR kinetic solid-production balance is not bracketed; "
                f"residuals are {f_lower:g} kmol/h at the seed-flow bound and "
                f"{previous_value:g} kmol/h near complete crystallization"
            )
        solved_solid, root_result = brentq(
            evaluate,
            bracket[0],
            bracket[1],
            xtol=max(5.0e-324, flow_tolerance * 0.01),
            rtol=8.881784197001252e-16,
            maxiter=iterations_limit,
            full_output=True,
            disp=False,
        )
        if not root_result.converged:
            raise MSMPRConvergenceError(
                "MSMPR kinetic solid-production balance did not converge"
            )
    residual = evaluate(solved_solid, include_thermal_diagnostics=True)
    final = last_result
    if final is None or abs(residual) > flow_tolerance:
        raise MSMPRConvergenceError(
            f"MSMPR material balance residual {residual:g} kmol/h exceeds "
            f"tolerance {flow_tolerance:g} kmol/h"
        )
    if seed_flow > 1.0e-15 and float(final['saturation_ratio']) < 1.0 - 1.0e-10:
        raise MSMPRConvergenceError(
            "MSMPR seed feed is undersaturated; seed dissolution is not modeled"
        )
    distribution = final['distribution']
    final_solid_flow = float(solved_solid)
    if distribution is not None:
        distribution = distribution.with_total_molar_flow(final_solid_flow)
    liquid_flows = dict(total_flows)
    liquid_flows[component] = max(0.0, available - final_solid_flow)
    population = final['population']
    saturation_ratio = float(final['saturation_ratio'])
    saturation_temperature = final['saturation_temperature']
    undercooling = (
        None
        if saturation_temperature is None
        else float(saturation_temperature) - temperature
    )
    return SteadyMSMPRResult(
        component=component,
        solid_flow_kmol_h=final_solid_flow,
        liquid_component_flows_kmol_h=liquid_flows,
        liquid_composition=dict(final['composition']),
        particle_size_distribution=distribution,
        saturation_ratio=saturation_ratio,
        relative_supersaturation=saturation_ratio - 1.0,
        log_saturation_ratio=math.log(max(saturation_ratio, 1.0e-300)),
        liquid_activity=float(final['activity']),
        saturation_activity=saturation_activity,
        saturation_temperature_K=(
            None
            if saturation_temperature is None
            else float(saturation_temperature)
        ),
        undercooling_K=undercooling,
        reduced_undercooling=(
            None
            if undercooling is None
            else undercooling / float(saturation_temperature)
        ),
        fusion_scaled_undercooling=(
            None
            if undercooling is None
            else heat_of_fusion_J_mol
            * undercooling
            / (R * temperature * melting_temperature)
        ),
        melting_temperature_K=melting_temperature,
        heat_of_fusion_J_mol=heat_of_fusion_J_mol,
        solute_concentration_kmol_m3=float(final['concentration']),
        suspension_density_kg_m3=float(final['suspension_density']),
        nucleation_rate_per_m3_h=float(final['nucleation_rate']),
        growth_rate_range_m_h=(
            float(population['minimum_growth_rate_m_h']),
            float(population['maximum_growth_rate_m_h']),
        ),
        birth_growth_rate_m_h=float(final['birth_growth_rate']),
        residence_time_h=float(final['residence_time']),
        volume_m3=float(final['volume']),
        volumetric_flow_m3_h=float(final['volumetric_flow']),
        nucleated_particle_rate_per_h=float(
            population['nucleated_particle_rate_per_h']
        ),
        seed_particle_rate_per_h=float(population['seed_particle_rate_per_h']),
        total_particle_rate_per_h=float(population['total_particle_rate_per_h']),
        number_mean_diameter_m=float(population['number_mean_diameter_m']),
        material_residual_kmol_h=residual,
        absolute_residual_tolerance_kmol_h=tolerance,
        relative_residual_tolerance=relative_tolerance,
        effective_residual_tolerance_kmol_h=flow_tolerance,
        iterations=evaluation_count,
    )


__all__ = [
    'MSMPRConvergenceError',
    'MSMPRDefinitionError',
    'MSMPRRateLaw',
    'SteadyMSMPRResult',
    'msmpr_rate_definition_from_parameters',
    'msmpr_rate_law_from_mapping',
    'solve_steady_msmpr',
]
