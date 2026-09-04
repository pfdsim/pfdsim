"""Conserved particle-size populations for solid process components."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from statistics import NormalDist

_BASIS_ALIASES = {
    'mole': 'mole',
    'molar': 'mole',
    'mass': 'mass',
    'volume': 'volume',
    'number': 'number',
    'count': 'number',
}

_DISTRIBUTION_ALIASES = {
    'discrete': 'discrete',
    'tabulated': 'discrete',
    'classes': 'discrete',
    'lognormal': 'lognormal',
    'log_normal': 'lognormal',
    'weibull': 'weibull',
    'rosin_rammler': 'weibull',
    'rosin_rammler_bennett': 'weibull',
    'rrb': 'weibull',
    'rrsb': 'weibull',
}

_DIAMETER_UNIT_FACTORS = {
    'm': 1.0,
    'meter': 1.0,
    'meters': 1.0,
    'mm': 1.0e-3,
    'millimeter': 1.0e-3,
    'millimeters': 1.0e-3,
    'um': 1.0e-6,
    'µm': 1.0e-6,
    'micrometer': 1.0e-6,
    'micrometers': 1.0e-6,
}


def _canonical_diameter(value: float) -> float:
    """Remove unit-conversion noise without merging physically distinct bins."""
    return float(f"{float(value):.15g}")


@dataclass(frozen=True)
class ParticleSizeDistribution:
    """Discrete solid population on an extensive component-molar-flow basis.

    Each diameter is the representative volume-equivalent diameter of one
    size class. The corresponding flow is the amount of the solid component
    carried by that class, in kmol/h. Keeping the stored values extensive
    makes mixing and splitting ordinary material-balance operations.
    """

    diameters_m: tuple[float, ...]
    molar_flows_kmol_per_h: tuple[float, ...]

    def __post_init__(self) -> None:
        diameters = tuple(
            _canonical_diameter(value) for value in self.diameters_m
        )
        flows = tuple(float(value) for value in self.molar_flows_kmol_per_h)
        if not diameters:
            raise ValueError("A particle-size distribution requires at least one class")
        if len(diameters) != len(flows):
            raise ValueError(
                "Particle diameters and class molar flows must have equal lengths"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in diameters):
            raise ValueError("Particle diameters must be positive and finite")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in flows
        ):
            raise ValueError(
                "Particle-size class molar flows must be finite and nonnegative"
            )
        if any(right <= left for left, right in pairwise(diameters)):
            raise ValueError(
                "Particle-size class diameters must be strictly increasing"
            )
        object.__setattr__(self, 'diameters_m', diameters)
        object.__setattr__(self, 'molar_flows_kmol_per_h', flows)

    @property
    def total_molar_flow(self) -> float:
        return sum(self.molar_flows_kmol_per_h)

    @property
    def molar_fractions(self) -> tuple[float, ...]:
        total = self.total_molar_flow
        if total <= 0.0:
            return tuple(0.0 for _ in self.molar_flows_kmol_per_h)
        return tuple(value / total for value in self.molar_flows_kmol_per_h)

    @property
    def number_fractions(self) -> tuple[float, ...]:
        # For one pure component, material volume is proportional to molar
        # flow. With volume-equivalent diameter, particle count is therefore
        # proportional to component flow divided by d**3.
        counts = tuple(
            flow / diameter**3
            for diameter, flow in zip(
                self.diameters_m, self.molar_flows_kmol_per_h
            )
        )
        total = sum(counts)
        if total <= 0.0:
            return tuple(0.0 for _ in counts)
        return tuple(value / total for value in counts)

    @property
    def sauter_mean_diameter_m(self) -> float | None:
        """Return D[3,2], using component flow as the volume-equivalent basis."""
        total = self.total_molar_flow
        denominator = sum(
            flow / diameter
            for diameter, flow in zip(
                self.diameters_m, self.molar_flows_kmol_per_h
            )
        )
        if total <= 0.0 or denominator <= 0.0:
            return None
        return total / denominator

    def scaled(self, factor: float) -> ParticleSizeDistribution:
        factor = float(factor)
        if not math.isfinite(factor) or factor < 0.0:
            raise ValueError("Particle-size distribution scale must be nonnegative")
        return ParticleSizeDistribution(
            self.diameters_m,
            tuple(value * factor for value in self.molar_flows_kmol_per_h),
        )

    def with_total_molar_flow(self, total: float) -> ParticleSizeDistribution:
        total = float(total)
        if not math.isfinite(total) or total < 0.0:
            raise ValueError("Particle-size distribution total flow must be nonnegative")
        current = self.total_molar_flow
        if current <= 0.0:
            if total <= 0.0:
                return self
            raise ValueError("A zero-flow particle distribution cannot be rescaled")
        return self.scaled(total / current)

    @classmethod
    def combine(
        cls,
        distributions: Iterable[ParticleSizeDistribution],
    ) -> ParticleSizeDistribution:
        """Combine populations without assuming a continuous bin shape."""
        by_diameter: dict[float, float] = {}
        for distribution in distributions:
            for diameter, flow in zip(
                distribution.diameters_m,
                distribution.molar_flows_kmol_per_h,
            ):
                by_diameter[diameter] = by_diameter.get(diameter, 0.0) + flow
        if not by_diameter:
            raise ValueError("At least one particle-size distribution is required")
        ordered = sorted(by_diameter.items())
        return cls(
            tuple(diameter for diameter, _flow in ordered),
            tuple(flow for _diameter, flow in ordered),
        )

    def to_dict(self) -> dict:
        payload = {
            'basis': 'component_molar_flow',
            'diameters_m': list(self.diameters_m),
            'molar_flows_kmol_per_h': list(self.molar_flows_kmol_per_h),
            'molar_fractions': list(self.molar_fractions),
        }
        if self.sauter_mean_diameter_m is not None:
            payload['sauter_mean_diameter_m'] = self.sauter_mean_diameter_m
        return payload


def _normalized_distribution_name(fields: dict[str, object]) -> str:
    selectors = [
        name for name in ('distribution', 'model')
        if name in fields
    ]
    if len(selectors) > 1:
        raise ValueError(
            "particle_size_distribution accepts only one distribution/model field"
        )
    if not selectors:
        return 'discrete'
    raw_name = str(fields[selectors[0]]).strip().lower().replace('-', '_')
    name = _DISTRIBUTION_ALIASES.get(raw_name)
    if name is None:
        raise ValueError(
            "particle_size_distribution distribution must be discrete, "
            "lognormal, or weibull/rosin_rammler"
        )
    return name


def _normalized_basis(
    fields: dict[str, object],
    *,
    required: bool,
) -> str:
    if required and 'basis' not in fields:
        raise ValueError(
            "Parametric particle_size_distribution requires an explicit basis"
        )
    raw_basis = str(fields.get('basis', 'mass')).strip().lower()
    basis = _BASIS_ALIASES.get(raw_basis)
    if basis is None:
        raise ValueError(
            "particle_size_distribution basis must be mole, mass, volume, or number"
        )
    return basis


def _diameter_factor(unit: object) -> float:
    normalized = str(unit).strip().lower()
    factor = _DIAMETER_UNIT_FACTORS.get(normalized)
    if factor is None:
        raise ValueError(
            f"Unsupported particle_size_distribution diameter_unit {normalized!r}"
        )
    return factor


def _single_diameter_m(
    fields: dict[str, object],
    names: tuple[str, ...],
    label: str,
) -> float:
    candidates = []
    for name in names:
        for suffix, factor in (('_m', 1.0), ('_mm', 1.0e-3), ('_um', 1.0e-6)):
            field_name = name + suffix
            if field_name in fields:
                candidates.append((field_name, factor))
        if name in fields:
            candidates.append((
                name,
                _diameter_factor(fields.get('diameter_unit', 'm')),
            ))
    if len(candidates) != 1:
        accepted = ', '.join(names)
        raise ValueError(
            f"particle_size_distribution requires exactly one {label} "
            f"diameter field ({accepted}, optionally suffixed _m/_mm/_um)"
        )
    field_name, factor = candidates[0]
    diameter = _canonical_diameter(float(fields[field_name]) * factor)
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError(
            f"particle_size_distribution {label} diameter must be positive and finite"
        )
    return diameter


def _class_count(fields: dict[str, object]) -> int:
    raw_value = fields.get('classes', 20)
    value = int(raw_value)
    if float(raw_value) != value or not 2 <= value <= 1000:
        raise ValueError(
            "particle_size_distribution classes must be an integer from 2 to 1000"
        )
    return value


def _normalize_discrete_distribution(fields: dict[str, object]) -> dict:
    diameter_fields = [
        name for name in ('diameters_m', 'diameters_mm', 'diameters_um', 'diameters')
        if name in fields
    ]
    if len(diameter_fields) != 1:
        raise ValueError(
            "particle_size_distribution requires exactly one of diameters_m, "
            "diameters_mm, diameters_um, or diameters"
        )
    allowed = {
        'diameters_m', 'diameters_mm', 'diameters_um', 'diameters',
        'diameter_unit', 'fractions', 'basis', 'distribution', 'model',
    }
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise ValueError(
            "Unknown particle_size_distribution field(s): " + ', '.join(unknown)
        )
    raw_diameters = fields[diameter_fields[0]]
    raw_fractions = fields.get('fractions')
    if not isinstance(raw_diameters, (list, tuple)):
        raise TypeError("particle_size_distribution diameters must be a list")
    if not isinstance(raw_fractions, (list, tuple)):
        raise TypeError("particle_size_distribution fractions must be a list")
    diameters = [float(item) for item in raw_diameters]
    fractions = [float(item) for item in raw_fractions]
    if not diameters or len(diameters) != len(fractions):
        raise ValueError(
            "particle_size_distribution diameters and fractions must be nonempty "
            "lists of equal length"
        )

    field_name = diameter_fields[0]
    if field_name == 'diameters_m':
        factor = 1.0
    elif field_name == 'diameters_mm':
        factor = 1.0e-3
    elif field_name == 'diameters_um':
        factor = 1.0e-6
    else:
        factor = _diameter_factor(fields.get('diameter_unit', 'm'))
    diameters = [_canonical_diameter(value * factor) for value in diameters]
    if any(not math.isfinite(value) or value <= 0.0 for value in diameters):
        raise ValueError("particle_size_distribution diameters must be positive and finite")
    if any(right <= left for left, right in pairwise(diameters)):
        raise ValueError(
            "particle_size_distribution diameters must be strictly increasing"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in fractions):
        raise ValueError(
            "particle_size_distribution fractions must be finite and nonnegative"
        )
    fraction_total = sum(fractions)
    if fraction_total <= 0.0:
        raise ValueError("particle_size_distribution fractions must have a positive sum")

    return {
        'diameters_m': diameters,
        'fractions': [value / fraction_total for value in fractions],
        'basis': _normalized_basis(fields, required=False),
    }


def _normalize_lognormal_distribution(fields: dict[str, object]) -> dict:
    allowed = {
        'distribution', 'model', 'basis', 'classes', 'diameter_unit',
        'd50', 'd50_m', 'd50_mm', 'd50_um',
        'median_diameter', 'median_diameter_m', 'median_diameter_mm',
        'median_diameter_um',
        'geometric_standard_deviation', 'gsd', 'sigma_g',
    }
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise ValueError(
            "Unknown lognormal particle_size_distribution field(s): "
            + ', '.join(unknown)
        )
    spread_fields = [
        name for name in ('geometric_standard_deviation', 'gsd', 'sigma_g')
        if name in fields
    ]
    if len(spread_fields) != 1:
        raise ValueError(
            "Lognormal particle_size_distribution requires exactly one "
            "geometric_standard_deviation/GSD/sigma_g field"
        )
    geometric_standard_deviation = float(fields[spread_fields[0]])
    if (
        not math.isfinite(geometric_standard_deviation)
        or geometric_standard_deviation < 1.0
    ):
        raise ValueError(
            "Lognormal geometric_standard_deviation must be finite and at least 1"
        )
    return {
        'distribution': 'lognormal',
        'd50_m': _single_diameter_m(
            fields, ('d50', 'median_diameter'), 'median (d50)'
        ),
        'geometric_standard_deviation': geometric_standard_deviation,
        'basis': _normalized_basis(fields, required=True),
        'classes': _class_count(fields),
    }


def _normalize_weibull_distribution(fields: dict[str, object]) -> dict:
    diameter_names = ('scale_diameter', 'characteristic_diameter', 'd63_2')
    allowed = {
        'distribution', 'model', 'basis', 'classes', 'diameter_unit',
        'shape', 'spread_parameter', 'exponent',
    }
    for name in diameter_names:
        allowed.add(name)
        allowed.update({name + '_m', name + '_mm', name + '_um'})
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise ValueError(
            "Unknown Weibull/Rosin-Rammler particle_size_distribution field(s): "
            + ', '.join(unknown)
        )
    shape_fields = [
        name for name in ('shape', 'spread_parameter', 'exponent')
        if name in fields
    ]
    if len(shape_fields) != 1:
        raise ValueError(
            "Weibull/Rosin-Rammler particle_size_distribution requires exactly "
            "one shape/spread_parameter/exponent field"
        )
    shape = float(fields[shape_fields[0]])
    if not math.isfinite(shape) or shape <= 0.0:
        raise ValueError(
            "Weibull/Rosin-Rammler shape must be positive and finite"
        )
    return {
        'distribution': 'weibull',
        'scale_diameter_m': _single_diameter_m(
            fields, diameter_names, 'scale (d63.2)'
        ),
        'shape': shape,
        'basis': _normalized_basis(fields, required=True),
        'classes': _class_count(fields),
    }


def normalize_particle_size_distribution(value: object) -> dict:
    """Validate and canonicalize a component-level normalized PSD default."""
    if not isinstance(value, Mapping):
        raise TypeError("particle_size_distribution must be a map")
    fields = {str(key).strip().lower(): item for key, item in value.items()}
    distribution = _normalized_distribution_name(fields)
    if distribution == 'lognormal':
        return _normalize_lognormal_distribution(fields)
    if distribution == 'weibull':
        return _normalize_weibull_distribution(fields)
    return _normalize_discrete_distribution(fields)


def _discretize_parametric_distribution(
    normalized: Mapping[str, object],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    distribution = normalized.get('distribution')
    if distribution == 'lognormal':
        median = float(normalized['d50_m'])
        geometric_standard_deviation = float(
            normalized['geometric_standard_deviation']
        )
        if geometric_standard_deviation == 1.0:
            return (median,), (1.0,)
        log_sigma = math.log(geometric_standard_deviation)

        def quantile(probability: float) -> float:
            return median * math.exp(
                log_sigma * NormalDist().inv_cdf(probability)
            )

    elif distribution == 'weibull':
        scale = float(normalized['scale_diameter_m'])
        shape = float(normalized['shape'])

        def quantile(probability: float) -> float:
            return scale * (-math.log1p(-probability)) ** (1.0 / shape)

    else:
        return (
            tuple(float(value) for value in normalized['diameters_m']),
            tuple(float(value) for value in normalized['fractions']),
        )

    classes = int(normalized['classes'])
    fractions = tuple(1.0 / classes for _ in range(classes))
    try:
        diameters = tuple(
            quantile((index + 0.5) / classes)
            for index in range(classes)
        )
    except OverflowError as error:
        raise ValueError(
            "Parametric particle_size_distribution generated a nonfinite diameter"
        ) from error
    if any(not math.isfinite(value) or value <= 0.0 for value in diameters):
        raise ValueError(
            "Parametric particle_size_distribution generated a nonpositive or "
            "nonfinite diameter"
        )
    return diameters, fractions


def instantiate_particle_size_distribution(
    specification: Mapping[str, object],
    total_molar_flow: float,
) -> ParticleSizeDistribution:
    """Instantiate a normalized default as an extensive stream population."""
    normalized = normalize_particle_size_distribution(specification)
    diameters, fractions = _discretize_parametric_distribution(normalized)
    total = float(total_molar_flow)
    if not math.isfinite(total) or total < 0.0:
        raise ValueError("Particle-size distribution total flow must be nonnegative")
    if normalized['basis'] == 'number':
        material_weights = tuple(
            fraction * diameter**3
            for diameter, fraction in zip(diameters, fractions)
        )
        weight_total = sum(material_weights)
        molar_fractions = tuple(value / weight_total for value in material_weights)
    else:
        # Within one pure component, mole, mass, and solid-volume fractions
        # are equivalent when density does not vary by size class.
        molar_fractions = fractions
    return ParticleSizeDistribution(
        diameters,
        tuple(total * fraction for fraction in molar_fractions),
    )


def propagate_particle_size_distributions(
    source_states: Iterable[object],
    target_state: object,
) -> None:
    """Propagate complete component populations through a non-size-selective step.

    Any PSD already attached to the target is treated as the configured default
    for newly created solid. If an input solid has no complete PSD and no target
    default exists, the output PSD is omitted rather than reporting a partial
    distribution as though it represented the whole component inventory.
    """
    sources = tuple(source_states)
    fallback = dict(
        getattr(target_state, 'solid_particle_size_distributions', {}) or {}
    )
    target_flows = dict(getattr(target_state, 'solid_component_flows', {}) or {})
    propagated = {}
    for component, raw_target_flow in target_flows.items():
        target_flow = float(raw_target_flow)
        if target_flow <= 1.0e-15:
            continue
        source_flow = 0.0
        missing_source_flow = 0.0
        source_distributions = []
        for source in sources:
            component_flow = float(
                (getattr(source, 'solid_component_flows', {}) or {}).get(
                    component, 0.0
                )
            )
            if component_flow <= 1.0e-15:
                continue
            source_flow += component_flow
            distribution = (
                getattr(source, 'solid_particle_size_distributions', {}) or {}
            ).get(component)
            tolerance = max(1.0e-12, component_flow * 1.0e-10)
            if (
                distribution is None
                or abs(distribution.total_molar_flow - component_flow) > tolerance
            ):
                missing_source_flow += component_flow
            else:
                source_distributions.append(distribution)

        default = fallback.get(component)
        if missing_source_flow > 1.0e-15:
            if default is None:
                continue
            source_distributions.append(
                default.with_total_molar_flow(missing_source_flow)
            )
        if not source_distributions:
            if default is not None:
                propagated[component] = default.with_total_molar_flow(
                    target_flow
                )
            continue

        combined = ParticleSizeDistribution.combine(source_distributions)
        if target_flow <= source_flow + max(1.0e-12, source_flow * 1.0e-10):
            propagated[component] = combined.with_total_molar_flow(target_flow)
            continue
        if default is None:
            continue
        created_flow = target_flow - source_flow
        created = default.with_total_molar_flow(created_flow)
        propagated[component] = ParticleSizeDistribution.combine((combined, created))

    target_state.solid_particle_size_distributions = propagated


def carry_particle_size_distributions(
    source_state: object,
    target_state: object,
) -> None:
    """Carry PSD shapes across a numerical state reconstruction.

    Unlike physical propagation, reconstruction does not create or destroy
    particles when its trial component flow changes. A complete source
    population is therefore rescaled to the reconstructed solid inventory
    instead of blending in the target's configured particle default.
    """
    source_flows = dict(
        getattr(source_state, 'solid_component_flows', {}) or {}
    )
    source_distributions = dict(
        getattr(source_state, 'solid_particle_size_distributions', {}) or {}
    )
    fallback = dict(
        getattr(target_state, 'solid_particle_size_distributions', {}) or {}
    )
    target_flows = dict(
        getattr(target_state, 'solid_component_flows', {}) or {}
    )
    carried = {}
    for component, raw_target_flow in target_flows.items():
        target_flow = float(raw_target_flow)
        if target_flow <= 1.0e-15:
            continue
        source_flow = float(source_flows.get(component, 0.0))
        distribution = source_distributions.get(component)
        source_tolerance = max(1.0e-12, abs(source_flow) * 1.0e-10)
        if (
            distribution is not None
            and source_flow > 1.0e-15
            and abs(distribution.total_molar_flow - source_flow)
            <= source_tolerance
        ):
            carried[component] = distribution.with_total_molar_flow(
                target_flow
            )
            continue
        default = fallback.get(component)
        if default is not None:
            carried[component] = default.with_total_molar_flow(target_flow)
    target_state.solid_particle_size_distributions = carried


__all__ = [
    'ParticleSizeDistribution',
    'carry_particle_size_distributions',
    'instantiate_particle_size_distribution',
    'normalize_particle_size_distribution',
    'propagate_particle_size_distributions',
]
