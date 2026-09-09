"""Canonical chemical-reaction parsing, validation, and extent accounting.

This module is deliberately independent of any particular reactor model.  A
conversion reactor, equilibrium reactor, and kinetic reactor should all agree
on stoichiometry and component accounting before they apply their own closure
equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Iterable, Mapping, Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .physical_constants import R_J_MOL_K
else:
    from physical_constants import R_J_MOL_K

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .compound_identity import parse_formula_counts
else:
    from compound_identity import parse_formula_counts


class ReactionDefinitionError(ValueError):
    """Raised when a reaction definition is malformed or inconsistent."""


_ARROW_PATTERN = re.compile(r"(<=>|<->|->|=>|=)")
_REVERSIBLE_ARROWS = frozenset({"<=>", "<->", "="})
_COEFFICIENT_PATTERN = re.compile(
    r"^(?:(?P<coefficient>(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?:\s+|\s*\*\s*))?(?P<component>\S(?:.*\S)?)$"
)


def _component_value(component, name: str):
    if isinstance(component, Mapping):
        return component.get(name)
    return getattr(component, name, None)


def _split_reaction_terms(side: str) -> list[str]:
    """Split one equation side while permitting charge-suffixed identifiers."""
    side = side.strip()
    if not side:
        return []

    # Whitespace-delimited plus signs are unambiguous even for identifiers such
    # as H+ and Fe3+.  Retain a compact-equation fallback for legacy A+B input,
    # except when the only plus sign is a terminal ionic charge.
    terms = re.split(r"\s+\+\s+", side)
    if len(terms) == 1 and "+" in side and not side.endswith("+"):
        terms = re.split(r"\s*\+\s*", side)
    return [term.strip() for term in terms if term.strip()]


def _parse_term(
    term: str,
    component_symbols: Optional[set[str]],
) -> tuple[str, float]:
    if component_symbols is not None and term in component_symbols:
        return term, 1.0

    if component_symbols is not None:
        compact = re.fullmatch(
            r"(?P<coefficient>(?:\d+(?:\.\d*)?|\.\d+))(?P<component>.+)",
            term,
        )
        if compact and compact.group('component') in component_symbols:
            coefficient = float(compact.group('coefficient'))
            if not math.isfinite(coefficient) or coefficient <= 0.0:
                raise ReactionDefinitionError(
                    f"Reaction coefficient for '{compact.group('component')}' "
                    "must be finite and positive"
                )
            return compact.group('component'), coefficient

    match = _COEFFICIENT_PATTERN.fullmatch(term)
    if not match:
        raise ReactionDefinitionError(f"Invalid reaction term '{term}'")

    component = match.group("component").strip()
    coefficient_text = match.group("coefficient")
    coefficient = float(coefficient_text) if coefficient_text else 1.0
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ReactionDefinitionError(
            f"Reaction coefficient for '{component}' must be finite and positive"
        )
    if any(token in component for token in ("->", "=>", "<->", "<=>", "|")):
        raise ReactionDefinitionError(f"Invalid reaction component identifier '{component}'")
    if component_symbols is not None and component not in component_symbols:
        raise ReactionDefinitionError(
            f"Reaction references undefined component '{component}'"
        )
    return component, coefficient


def _parse_side(
    side: str,
    component_symbols: Optional[set[str]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for term in _split_reaction_terms(side):
        component, coefficient = _parse_term(term, component_symbols)
        result[component] = result.get(component, 0.0) + coefficient
    if not result:
        raise ReactionDefinitionError("Each reaction side must contain a component")
    return result


@dataclass(frozen=True)
class StoichiometricReaction:
    """Normalized stoichiometry for one chemical reaction."""

    equation: str
    stoichiometry: dict[str, float]
    reactants: dict[str, float]
    products: dict[str, float]
    reversible: bool = False
    arrow: str = "->"

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.stoichiometry)

    @property
    def default_basis_component(self) -> str:
        for component in self.reactants:
            if self.stoichiometry.get(component, 0.0) < 0.0:
                return component
        raise ReactionDefinitionError(
            f"Reaction '{self.equation}' has no net-consumed reactant"
        )


def parse_reaction_equation(
    equation: object,
    component_symbols: Optional[Iterable[str]] = None,
) -> StoichiometricReaction:
    """Parse and normalize a reaction equation.

    Repeated species are summed on each side and species present on both sides
    are reduced to their net stoichiometric coefficient.
    """
    text = str(equation or "").strip()
    if not text:
        raise ReactionDefinitionError("Reaction equation is empty")

    matches = list(_ARROW_PATTERN.finditer(text))
    if len(matches) != 1:
        raise ReactionDefinitionError(
            f"Reaction equation '{text}' must contain exactly one reaction arrow"
        )
    arrow = matches[0].group(1)
    left = text[:matches[0].start()].strip()
    right = text[matches[0].end():].strip()
    known = set(str(item) for item in component_symbols) if component_symbols is not None else None
    reactants = _parse_side(left, known)
    products = _parse_side(right, known)

    ordered_components = list(reactants)
    ordered_components.extend(component for component in products if component not in reactants)
    stoichiometry = {}
    for component in ordered_components:
        coefficient = products.get(component, 0.0) - reactants.get(component, 0.0)
        if abs(coefficient) > 1.0e-14:
            stoichiometry[component] = coefficient

    if not any(value < 0.0 for value in stoichiometry.values()):
        raise ReactionDefinitionError(
            f"Reaction '{text}' has no net-consumed reactant"
        )
    if not any(value > 0.0 for value in stoichiometry.values()):
        raise ReactionDefinitionError(
            f"Reaction '{text}' has no net-formed product"
        )

    return StoichiometricReaction(
        equation=text,
        stoichiometry=stoichiometry,
        reactants=reactants,
        products=products,
        reversible=arrow in _REVERSIBLE_ARROWS,
        arrow=arrow,
    )


def validate_reaction_balance(
    reaction: StoichiometricReaction,
    component_metadata: Mapping[str, object],
) -> list[str]:
    """Validate elemental balance when metadata is complete.

    Missing formula data do not make the reaction unusable.  They produce an
    explicit warning stating which validation could not be performed.
    """
    warnings: list[str] = []
    missing_formulas = []
    formula_counts: dict[str, dict[str, int]] = {}
    for component in reaction.components:
        metadata = component_metadata.get(component)
        formula = _component_value(metadata, "formula") if metadata is not None else None
        counts = parse_formula_counts(str(formula)) if formula else None
        if not counts:
            missing_formulas.append(component)
        else:
            formula_counts[component] = counts

    if missing_formulas:
        warnings.append(
            f"Reaction '{reaction.equation}' elemental balance was not checked; "
            "missing or unparseable formula for: " + ", ".join(missing_formulas)
        )
    else:
        elements = sorted({
            element
            for counts in formula_counts.values()
            for element in counts
        })
        residuals = {
            element: sum(
                reaction.stoichiometry[component]
                * formula_counts[component].get(element, 0)
                for component in reaction.components
            )
            for element in elements
        }
        unbalanced = {
            element: residual
            for element, residual in residuals.items()
            if abs(residual) > 1.0e-10
        }
        if unbalanced:
            details = ", ".join(
                f"{element}={residual:+g}" for element, residual in unbalanced.items()
            )
            raise ReactionDefinitionError(
                f"Reaction '{reaction.equation}' is not elementally balanced ({details})"
            )

    return warnings


@dataclass(frozen=True)
class ConversionReaction:
    """A stoichiometric reaction with an inlet-basis conversion specification."""

    reaction: StoichiometricReaction
    basis_component: str
    specified_conversion: float
    name: Optional[str] = None
    validation_warnings: tuple[str, ...] = ()


def conversion_reaction_from_mapping(
    definition: Mapping[str, object],
    component_symbols: Iterable[str],
    component_metadata: Optional[Mapping[str, object]] = None,
) -> ConversionReaction:
    """Build a validated conversion-reaction specification."""
    if not isinstance(definition, Mapping):
        raise ReactionDefinitionError("Reaction definition must be a mapping")
    if "selectivity" in definition:
        raise ReactionDefinitionError(
            "selectivity is a calculated reactor result, not a reaction specification; "
            "represent side reactions explicitly"
        )
    if "yield" in definition:
        raise ReactionDefinitionError(
            "yield is a calculated reactor result, not a reaction specification"
        )
    if definition.get("conversion") is None:
        raise ReactionDefinitionError("Conversion reactor reactions require conversion")

    try:
        conversion = float(definition["conversion"])
    except (TypeError, ValueError) as exc:
        raise ReactionDefinitionError("Reaction conversion must be numeric") from exc
    if not math.isfinite(conversion) or not 0.0 <= conversion <= 1.0:
        raise ReactionDefinitionError("Reaction conversion must be between 0 and 1")

    symbols = tuple(str(item) for item in component_symbols)
    reaction = parse_reaction_equation(definition.get("equation"), symbols)
    basis = definition.get("basis", definition.get("basis_component"))
    basis_component = str(basis).strip() if basis is not None else reaction.default_basis_component
    if reaction.stoichiometry.get(basis_component, 0.0) >= 0.0:
        raise ReactionDefinitionError(
            f"Reaction basis '{basis_component}' must be a net-consumed reactant in "
            f"'{reaction.equation}'"
        )

    warnings = (
        validate_reaction_balance(reaction, component_metadata)
        if component_metadata is not None else []
    )
    name = definition.get("name", definition.get("label"))
    return ConversionReaction(
        reaction=reaction,
        basis_component=basis_component,
        specified_conversion=conversion,
        name=str(name) if name is not None else None,
        validation_warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class ReactionExtent:
    """Calculated extent and component changes for one reaction."""

    specification: ConversionReaction
    extent_kmol_h: float
    basis_inlet_kmol_h: float
    basis_consumed_kmol_h: float
    component_changes_kmol_h: dict[str, float]


@dataclass(frozen=True)
class ExtentSolution:
    """Simultaneous, order-independent application of reaction extents."""

    inlet_component_flows: dict[str, float]
    outlet_component_flows: dict[str, float]
    reaction_extents: tuple[ReactionExtent, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def component_conversions(self) -> dict[str, float]:
        return {
            component: (
                (flow - self.outlet_component_flows.get(component, 0.0)) / flow
                if flow > 0.0 else 0.0
            )
            for component, flow in self.inlet_component_flows.items()
            if flow > 0.0
        }

    @property
    def component_formation_flows(self) -> dict[str, float]:
        components = set(self.inlet_component_flows) | set(self.outlet_component_flows)
        return {
            component: formation
            for component in sorted(components)
            if (
                formation := self.outlet_component_flows.get(component, 0.0)
                - self.inlet_component_flows.get(component, 0.0)
            ) > 1.0e-14
        }

    @property
    def component_consumption_flows(self) -> dict[str, float]:
        components = set(self.inlet_component_flows) | set(self.outlet_component_flows)
        return {
            component: consumption
            for component in sorted(components)
            if (
                consumption := self.inlet_component_flows.get(component, 0.0)
                - self.outlet_component_flows.get(component, 0.0)
            ) > 1.0e-14
        }


def solve_conversion_extents(
    inlet_component_flows: Mapping[str, float],
    reactions: Iterable[ConversionReaction],
) -> ExtentSolution:
    """Apply fixed inlet-basis conversions simultaneously and exactly."""
    inlet = {str(component): float(flow) for component, flow in inlet_component_flows.items()}
    for component, flow in inlet.items():
        if not math.isfinite(flow) or flow < 0.0:
            raise ReactionDefinitionError(
                f"Inlet component flow for '{component}' must be finite and nonnegative"
            )

    extent_results = []
    total_changes: dict[str, float] = {}
    warnings = []
    for specification in reactions:
        basis = specification.basis_component
        basis_inlet = inlet.get(basis, 0.0)
        basis_coefficient = -specification.reaction.stoichiometry[basis]
        extent = specification.specified_conversion * basis_inlet / basis_coefficient
        changes = {
            component: coefficient * extent
            for component, coefficient in specification.reaction.stoichiometry.items()
        }
        for component, change in changes.items():
            total_changes[component] = total_changes.get(component, 0.0) + change
        extent_results.append(ReactionExtent(
            specification=specification,
            extent_kmol_h=extent,
            basis_inlet_kmol_h=basis_inlet,
            basis_consumed_kmol_h=basis_coefficient * extent,
            component_changes_kmol_h=changes,
        ))
        warnings.extend(specification.validation_warnings)

    components = set(inlet) | set(total_changes)
    raw_outlet = {
        component: inlet.get(component, 0.0) + total_changes.get(component, 0.0)
        for component in components
    }
    scale = max(1.0, sum(inlet.values()))
    tolerance = 1.0e-12 * scale
    infeasible = {
        component: flow
        for component, flow in raw_outlet.items()
        if flow < -tolerance
    }
    if infeasible:
        details = ", ".join(
            f"{component}={flow:.8g} kmol/h" for component, flow in sorted(infeasible.items())
        )
        raise ReactionDefinitionError(
            "Specified reaction conversions are jointly infeasible; calculated "
            f"negative outlet flow(s): {details}"
        )

    outlet = {
        component: max(0.0, flow)
        for component, flow in raw_outlet.items()
        if flow > tolerance
    }
    return ExtentSolution(
        inlet_component_flows=inlet,
        outlet_component_flows=outlet,
        reaction_extents=tuple(extent_results),
        warnings=tuple(dict.fromkeys(warnings)),
    )


@dataclass(frozen=True)
class EquilibriumReaction:
    """A reversible stoichiometric reaction used for chemical equilibrium."""

    reaction: StoichiometricReaction
    name: Optional[str] = None
    validation_warnings: tuple[str, ...] = ()


def equilibrium_reaction_from_mapping(
    definition: Mapping[str, object],
    component_symbols: Iterable[str],
    component_metadata: Optional[Mapping[str, object]] = None,
) -> EquilibriumReaction:
    """Build a validated homogeneous-equilibrium reaction definition."""
    if not isinstance(definition, Mapping):
        raise ReactionDefinitionError("Reaction definition must be a mapping")
    allowed = {'equation', 'name', 'label'}
    unexpected = sorted(str(key) for key in definition if key not in allowed)
    if unexpected:
        raise ReactionDefinitionError(
            "Equilibrium reaction does not accept parameter(s): "
            + ', '.join(unexpected)
        )
    symbols = tuple(str(item) for item in component_symbols)
    reaction = parse_reaction_equation(definition.get('equation'), symbols)
    if reaction.arrow not in {'<=>', '<->'}:
        raise ReactionDefinitionError(
            "EquilibriumReactor reactions require the reversible arrow <=> or <->"
        )
    warnings = (
        validate_reaction_balance(reaction, component_metadata)
        if component_metadata is not None else []
    )
    name = definition.get('name', definition.get('label'))
    return EquilibriumReaction(
        reaction=reaction,
        name=str(name) if name is not None else None,
        validation_warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class EquilibriumExtentSolution:
    """Result of a constrained homogeneous chemical-equilibrium solve."""

    inlet_component_flows: dict[str, float]
    outlet_component_flows: dict[str, float]
    extents_kmol_h: tuple[float, ...]
    log_equilibrium_constants: tuple[float, ...]
    log_reaction_quotients: tuple[float, ...]
    residuals: tuple[float, ...]
    reaction_statuses: tuple[str, ...]
    iterations: int
    objective: float
    warnings: tuple[str, ...] = ()

    @property
    def max_residual(self) -> float:
        return max((abs(value) for value in self.residuals), default=0.0)

    @property
    def max_interior_residual(self) -> float:
        return max((
            abs(value)
            for value, status in zip(self.residuals, self.reaction_statuses)
            if status == 'interior_equilibrium'
        ), default=0.0)


def _equilibrium_stoichiometric_matrix(
    components: tuple[str, ...],
    reactions: tuple[EquilibriumReaction, ...],
):
    import numpy as np

    matrix = np.asarray([
        [
            specification.reaction.stoichiometry.get(component, 0.0)
            for specification in reactions
        ]
        for component in components
    ], dtype=float)
    norms = np.linalg.norm(matrix, axis=0)
    if any(not math.isfinite(float(value)) or value <= 0.0 for value in norms):
        raise ReactionDefinitionError("Equilibrium reaction has empty stoichiometry")
    scaled = matrix / norms
    rank = int(np.linalg.matrix_rank(scaled, tol=1.0e-10))
    if rank != len(reactions):
        raise ReactionDefinitionError(
            "Equilibrium reactions are duplicate or linearly dependent; "
            f"stoichiometric rank {rank} for {len(reactions)} reactions"
        )
    return matrix


def solve_homogeneous_equilibrium(
    inlet_component_flows: Mapping[str, float],
    reactions: Iterable[EquilibriumReaction],
    thermo,
    T: float,
    P: float,
    phase: str,
    *,
    residual_tolerance: float = 1.0e-7,
    max_iterations: int = 500,
) -> EquilibriumExtentSolution:
    """Minimize homogeneous Gibbs energy in independent reaction extents.

    The linear constraints enforce ``n0 + N*xi >= 0`` exactly.  Equilibrium is
    independently verified from ``ln(Q)-ln(K)`` after minimization.
    """
    import numpy as np
    from scipy.optimize import LinearConstraint, brentq, minimize, root

    reaction_tuple = tuple(reactions)
    if not reaction_tuple:
        raise ReactionDefinitionError("At least one equilibrium reaction is required")
    phase_name = str(phase).strip().lower()
    if phase_name == 'gas':
        phase_name = 'vapor'
    if phase_name not in {'vapor', 'liquid'}:
        raise ReactionDefinitionError(
            "Homogeneous equilibrium requires phase='vapor' or phase='liquid'"
        )
    temperature = float(T)
    pressure = float(P)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ReactionDefinitionError("Equilibrium temperature must be positive and finite")
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ReactionDefinitionError("Equilibrium pressure must be positive and finite")

    components = tuple(str(component) for component in thermo.components)
    inlet = {
        component: float(inlet_component_flows.get(component, 0.0))
        for component in components
    }
    unknown = sorted(set(inlet_component_flows) - set(components))
    if unknown:
        raise ReactionDefinitionError(
            "Equilibrium inlet contains components outside the fluid backend: "
            + ', '.join(unknown)
        )
    for component, flow in inlet.items():
        if not math.isfinite(flow) or flow < 0.0:
            raise ReactionDefinitionError(
                f"Inlet component flow for '{component}' must be finite and nonnegative"
            )
    n0 = np.asarray([inlet[component] for component in components], dtype=float)
    flow_scale = max(float(np.sum(n0)), 1.0)
    matrix = _equilibrium_stoichiometric_matrix(components, reaction_tuple)
    participating = {
        component
        for specification in reaction_tuple
        for component in specification.reaction.components
    }
    mu0 = np.asarray([
        float(thermo.standard_chemical_potential(component, temperature))
        if component in participating else 0.0
        for component in components
    ], dtype=float)
    log_k = np.asarray([
        float(thermo.reaction_log_equilibrium_constant(
            specification.reaction.stoichiometry,
            temperature,
        ))
        for specification in reaction_tuple
    ], dtype=float)
    rt = R_J_MOL_K * temperature
    mole_tolerance = max(1.0e-14 * flow_scale, 1.0e-15)

    def flows_for(extents):
        return n0 + matrix @ np.asarray(extents, dtype=float)

    def composition_and_activities(extents):
        flows = flows_for(extents)
        if np.min(flows) < -mole_tolerance:
            raise ReactionDefinitionError(
                "Equilibrium trial extent produced a negative component flow"
            )
        flows = np.maximum(flows, 0.0)
        total = float(np.sum(flows))
        if total <= 0.0:
            raise ReactionDefinitionError("Equilibrium trial has zero total flow")
        composition = {
            component: float(flows[index] / total)
            for index, component in enumerate(components)
        }
        activities = thermo.component_activities(
            temperature,
            pressure,
            composition,
            phase_name,
        )
        return flows, activities

    def objective(extents):
        try:
            flows, activities = composition_and_activities(extents)
        except Exception:
            return 1.0e100
        total = 0.0
        for index, component in enumerate(components):
            amount = float(flows[index])
            if amount <= mole_tolerance:
                continue
            activity = max(float(activities.get(component, 0.0)), 1.0e-300)
            total += amount * (mu0[index] + rt * math.log(activity))
        return total / (rt * flow_scale)

    constraint = LinearConstraint(matrix, -n0, np.full(len(components), np.inf))
    starts = [np.zeros(len(reaction_tuple), dtype=float)]
    for index, specification in enumerate(reaction_tuple):
        forward = min(
            n0[components.index(component)] / (-coefficient)
            for component, coefficient in specification.reaction.stoichiometry.items()
            if coefficient < 0.0
        )
        reverse = min(
            n0[components.index(component)] / coefficient
            for component, coefficient in specification.reaction.stoichiometry.items()
            if coefficient > 0.0
        )
        for fraction in (0.1, 0.5, 0.9):
            if forward > 0.0:
                trial = np.zeros(len(reaction_tuple), dtype=float)
                trial[index] = fraction * forward
                starts.append(trial)
            if reverse > 0.0:
                trial = np.zeros(len(reaction_tuple), dtype=float)
                trial[index] = -fraction * reverse
                starts.append(trial)

    candidates = []
    messages = []
    for start in starts:
        solved = minimize(
            objective,
            start,
            method='SLSQP',
            constraints=(constraint,),
            options={
                'ftol': 1.0e-12,
                'maxiter': int(max_iterations),
                'disp': False,
            },
        )
        flows = flows_for(solved.x)
        feasible = bool(np.min(flows) >= -mole_tolerance)
        if feasible and math.isfinite(float(solved.fun)):
            candidates.append(solved)
        if not solved.success:
            messages.append(str(solved.message))
    if not candidates:
        detail = '; '.join(dict.fromkeys(messages)) or 'no feasible candidate'
        raise ReactionDefinitionError(
            "Homogeneous equilibrium minimization failed: " + detail
        )
    solved = min(candidates, key=lambda candidate: float(candidate.fun))
    final_extents = np.asarray(solved.x, dtype=float)

    def equilibrium_residual(extents):
        try:
            _flows, trial_activities = composition_and_activities(extents)
        except Exception:
            return np.full(len(reaction_tuple), 1.0e6, dtype=float)
        trial_log_q = np.asarray([
            sum(
                coefficient * math.log(max(
                    float(trial_activities[component]), 1.0e-300
                ))
                for component, coefficient
                in specification.reaction.stoichiometry.items()
            )
            for specification in reaction_tuple
        ], dtype=float)
        return trial_log_q - log_k

    forced_boundary = False
    if len(reaction_tuple) == 1:
        specification = reaction_tuple[0]
        lower = max(
            -n0[components.index(component)] / coefficient
            for component, coefficient in specification.reaction.stoichiometry.items()
            if coefficient > 0.0
        )
        upper = min(
            n0[components.index(component)] / (-coefficient)
            for component, coefficient in specification.reaction.stoichiometry.items()
            if coefficient < 0.0
        )
        reporting_amount_floor = 1.0e-14 * flow_scale
        lower_flows = flows_for([lower])
        upper_flows = flows_for([upper])
        lower_margin = max((
            (reporting_amount_floor - lower_flows[index]) / matrix[index, 0]
            for index in range(len(components))
            if matrix[index, 0] > 0.0
            and lower_flows[index] < reporting_amount_floor
        ), default=0.0)
        upper_margin = max((
            (reporting_amount_floor - upper_flows[index]) / (-matrix[index, 0])
            for index in range(len(components))
            if matrix[index, 0] < 0.0
            and upper_flows[index] < reporting_amount_floor
        ), default=0.0)
        lower_interior = min(max(lower + lower_margin, lower), upper)
        upper_interior = max(min(upper - upper_margin, upper), lower)
        lower_residual = float(equilibrium_residual([lower_interior])[0])
        upper_residual = float(equilibrium_residual([upper_interior])[0])
        if lower_residual == 0.0:
            final_extents[0] = lower_interior
        elif upper_residual == 0.0:
            final_extents[0] = upper_interior
        elif lower_residual * upper_residual < 0.0:
            final_extents[0] = brentq(
                lambda extent: float(equilibrium_residual([extent])[0]),
                lower_interior,
                upper_interior,
                xtol=5.0e-324,
                rtol=1.0e-15,
                maxiter=200,
            )
        elif lower_residual > 0.0:
            final_extents[0] = lower
            forced_boundary = True
        elif upper_residual < 0.0:
            final_extents[0] = upper
            forced_boundary = True
    else:
        polished = root(equilibrium_residual, final_extents, method='hybr')
        polished_flows = flows_for(polished.x)
        if (
            polished.success
            and np.min(polished_flows) >= -mole_tolerance
            and np.max(np.abs(equilibrium_residual(polished.x)))
            < np.max(np.abs(equilibrium_residual(final_extents)))
        ):
            final_extents = np.asarray(polished.x, dtype=float)

    flows, activities = composition_and_activities(final_extents)
    if np.min(flows) <= mole_tolerance:
        total_flow = float(np.sum(flows))
        reporting_floor = 1.0e-14 * total_flow
        reporting_flows = np.maximum(flows, reporting_floor)
        reporting_total = float(np.sum(reporting_flows))
        reporting_composition = {
            component: float(reporting_flows[index] / reporting_total)
            for index, component in enumerate(components)
        }
        activities = thermo.component_activities(
            temperature,
            pressure,
            reporting_composition,
            phase_name,
        )
    log_q = np.asarray([
        sum(
            coefficient * math.log(max(float(activities[component]), 1.0e-300))
            for component, coefficient in specification.reaction.stoichiometry.items()
        )
        for specification in reaction_tuple
    ], dtype=float)
    residual = log_q - log_k
    reaction_statuses = []
    for specification, value in zip(reaction_tuple, residual):
        boundary = any(
            flows[components.index(component)] <= mole_tolerance
            for component in specification.reaction.components
        )
        reaction_statuses.append(
            'boundary_limited'
            if (forced_boundary or boundary)
            and abs(float(value)) > float(residual_tolerance)
            else 'interior_equilibrium'
        )
    maximum_interior = max((
        abs(float(value))
        for value, status in zip(residual, reaction_statuses)
        if status == 'interior_equilibrium'
    ), default=0.0)
    if not solved.success or maximum_interior > float(residual_tolerance):
        raise ReactionDefinitionError(
            "Homogeneous equilibrium solve did not satisfy reaction equilibrium; "
            f"maximum interior |ln(Q)-ln(K)|={maximum_interior:.6g}; "
            f"minimum mole fraction={float(np.min(flows) / np.sum(flows)):.6g}; "
            f"extents={tuple(float(value) for value in final_extents)}; "
            f"solver={solved.message}"
        )

    outlet = {
        component: max(float(flows[index]), 0.0)
        for index, component in enumerate(components)
        if flows[index] > mole_tolerance
    }
    warnings = tuple(dict.fromkeys(
        warning
        for specification in reaction_tuple
        for warning in specification.validation_warnings
    ))
    return EquilibriumExtentSolution(
        inlet_component_flows=inlet,
        outlet_component_flows=outlet,
        extents_kmol_h=tuple(float(value) for value in final_extents),
        log_equilibrium_constants=tuple(float(value) for value in log_k),
        log_reaction_quotients=tuple(float(value) for value in log_q),
        residuals=tuple(float(value) for value in residual),
        reaction_statuses=tuple(reaction_statuses),
        iterations=int(getattr(solved, 'nit', 0) or 0),
        objective=float(solved.fun),
        warnings=warnings,
    )
