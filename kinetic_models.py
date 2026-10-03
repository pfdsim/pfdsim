"""Canonical homogeneous kinetic-reaction and CSTR calculations.

Rates are normalized internally to either kmol/m3/h for homogeneous
fluid-volume kinetics or kmol/kg_cat/h for fixed-catalyst mass kinetics.
Public reaction definitions must state the units in which their kinetic
numbers are written; there are no magnitude-based unit guesses in this module.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Optional

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .reaction_models import (
        ReactionDefinitionError,
        StoichiometricReaction,
        parse_reaction_equation,
        validate_reaction_balance,
    )
else:
    from reaction_models import (
        ReactionDefinitionError,
        StoichiometricReaction,
        parse_reaction_equation,
        validate_reaction_balance,
    )
if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .thermodynamics_models.common import P_REF, R
else:
    from thermodynamics_models.common import P_REF, R


class KineticsError(ValueError):
    """Raised when a kinetic definition or reactor solve is invalid."""


KINETIC_PARAMETER_FIELDS = frozenset({
    'equation', 'name', 'label', 'type', 'kinetics_type', 'A', 'Ea',
    'activation_energy', 'Ea_unit', 'rate_basis', 'basis',
    'concentration_unit', 'pressure_unit', 'rate_unit', 'expression',
})


def kinetic_input_catalog() -> dict:
    """Expose the runtime's supported kinetic choices and unit vocabulary."""
    return {'models':['power_law','custom','custom_net'],
            'basis':sorted(set(_RATE_BASIS_ALIASES.values())),
            'Ea_unit':list(_ACTIVATION_ENERGY_FACTORS),
            'concentration_unit':list(_CONCENTRATION_FACTORS),
            'pressure_unit':list(_PRESSURE_FACTORS), 'rate_unit':list(_RATE_FACTORS)}


_RATE_BASIS_ALIASES = {
    'concentration': 'concentration',
    'c': 'concentration',
    'partial_pressure': 'partial_pressure',
    'partial-pressure': 'partial_pressure',
    'pressure': 'partial_pressure',
    'p': 'partial_pressure',
    'fugacity': 'fugacity',
    'f': 'fugacity',
    'activity': 'activity',
    'a': 'activity',
}

_CONCENTRATION_FACTORS = {
    'kmol/m3': 1.0,
    'kmol/m^3': 1.0,
    'mol/m3': 1000.0,
    'mol/m^3': 1000.0,
    'mol/l': 1.0,
}

_PRESSURE_FACTORS = {
    'bar': 1.0,
    'kpa': 100.0,
    'pa': 100000.0,
    'atm': 1.0 / 1.01325,
}

# Multipliers from a declared rate to the canonical rate for its dimensional
# basis: kmol/m3/h for fluid-volume rates or kmol/kg_cat/h for catalyst-mass
# rates.
_RATE_FACTORS = {
    'kmol/m3/h': 1.0,
    'kmol/m^3/h': 1.0,
    'kmol/m3/s': 3600.0,
    'kmol/m^3/s': 3600.0,
    'mol/m3/h': 1.0e-3,
    'mol/m^3/h': 1.0e-3,
    'mol/m3/s': 3.6,
    'mol/m^3/s': 3.6,
    'mol/l/h': 1.0,
    'mol/l/min': 60.0,
    'mol/l/s': 3600.0,
    'kmol/kg_cat/h': 1.0,
    'kmol/kgcat/h': 1.0,
    'kmol/kg_cat/s': 3600.0,
    'kmol/kgcat/s': 3600.0,
    'mol/kg_cat/h': 1.0e-3,
    'mol/kgcat/h': 1.0e-3,
    'mol/kg_cat/s': 3.6,
    'mol/kgcat/s': 3.6,
}

_CATALYST_MASS_RATE_UNITS = frozenset({
    'kmol/kg_cat/h', 'kmol/kgcat/h',
    'kmol/kg_cat/s', 'kmol/kgcat/s',
    'mol/kg_cat/h', 'mol/kgcat/h',
    'mol/kg_cat/s', 'mol/kgcat/s',
})

_ACTIVATION_ENERGY_FACTORS = {
    'j/mol': 1.0,
    'kj/mol': 1000.0,
    'kj/kmol': 1.0,
    'j/kmol': 1.0e-3,
}

_RESERVED_EXPRESSION_NAMES = frozenset({
    'T', 'R', 'k', 'C', 'p', 'f', 'a', 'volpct', 'exp', 'log',
})


def _unit_token(value: object) -> str:
    return str(value).strip().lower().replace(' ', '')


def _finite_number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ReactionDefinitionError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ReactionDefinitionError(f"{label} must be finite")
    return number


def _required_unit(definition: Mapping[str, object], name: str, supported: Mapping[str, float]) -> str:
    value = definition.get(name)
    if value is None or not str(value).strip():
        raise ReactionDefinitionError(f"Kinetic reaction requires explicit {name}")
    token = _unit_token(value)
    if token not in supported:
        raise ReactionDefinitionError(
            f"Unsupported {name}={value!r}; supported values are "
            + ', '.join(sorted(supported))
        )
    return token


def _safe_power(base: float, exponent: float) -> float:
    if base < 0.0 and not float(exponent).is_integer():
        raise KineticsError("Fractional powers require a nonnegative base")
    if base == 0.0 and exponent < 0.0:
        raise KineticsError("A negative power of zero is undefined")
    value = base ** exponent
    if isinstance(value, complex) or not math.isfinite(float(value)):
        raise KineticsError("Kinetic expression produced a non-finite power")
    return float(value)


def _safe_expression_divide(left: float, right: float) -> float:
    if right == 0.0:
        raise KineticsError("Division by zero in kinetic expression")
    return left / right


def _safe_expression_log(value: float) -> float:
    if value <= 0.0:
        raise KineticsError("log(...) requires a positive argument")
    return math.log(value)


def _safe_expression_log10(value: float) -> float:
    if value <= 0.0:
        raise KineticsError("log10(...) requires a positive argument")
    return math.log10(value)


def _safe_expression_sqrt(value: float) -> float:
    if value < 0.0:
        raise KineticsError("sqrt(...) requires a nonnegative argument")
    return math.sqrt(value)


def _safe_expression_mapping_value(mapping: Mapping[str, object], key: str) -> float:
    return float(mapping.get(key, 0.0))


_SAFE_RATE_EXPRESSION_GLOBALS = {
    '__builtins__': {},
    '__rate_abs': abs,
    '__rate_divide': _safe_expression_divide,
    '__rate_exp': math.exp,
    '__rate_log': _safe_expression_log,
    '__rate_log10': _safe_expression_log10,
    '__rate_mapping_value': _safe_expression_mapping_value,
    '__rate_max': max,
    '__rate_min': min,
    '__rate_power': _safe_power,
    '__rate_sqrt': _safe_expression_sqrt,
}


class _SafeRateExpressionLowerer(ast.NodeTransformer):
    """Lower validated expressions to calls with the interpreter's semantics."""

    _function_names = {
        'abs': '__rate_abs',
        'exp': '__rate_exp',
        'log': '__rate_log',
        'log10': '__rate_log10',
        'max': '__rate_max',
        'min': '__rate_min',
        'sqrt': '__rate_sqrt',
    }

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        helper = None
        if isinstance(node.op, ast.Div):
            helper = '__rate_divide'
        elif isinstance(node.op, ast.Pow):
            helper = '__rate_power'
        if helper is None:
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=helper, ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            ),
            node,
        )

    def visit_Call(self, node):
        node = self.generic_visit(node)
        node.func.id = self._function_names[node.func.id]
        return node

    def visit_Subscript(self, node):
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id='__rate_mapping_value', ctx=ast.Load()),
                args=[self.visit(node.value), self.visit(node.slice)],
                keywords=[],
            ),
            node,
        )


class SafeRateExpression:
    """A tiny arithmetic expression language for custom kinetic rates."""

    _allowed_binary = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    _allowed_unary = (ast.UAdd, ast.USub)

    def __init__(
        self,
        expression: object,
        parameter_names: Iterable[str],
        *,
        scalar_names: Optional[Iterable[str]] = None,
        mapping_names: Optional[Iterable[str]] = None,
        function_names: Optional[Iterable[str]] = None,
    ):
        text = str(expression or '').strip()
        if not text:
            raise ReactionDefinitionError("Custom kinetics requires expression")
        if len(text) > 1000:
            raise ReactionDefinitionError("Custom kinetic expression is too long")
        try:
            tree = ast.parse(text, mode='eval')
        except SyntaxError as error:
            raise ReactionDefinitionError(
                f"Invalid custom kinetic expression: {error.msg}"
            ) from error
        self.text = text
        self.parameter_names = frozenset(parameter_names)
        self._scalar_names = frozenset(
            {'T', 'R', 'k'} if scalar_names is None else scalar_names
        )
        self._allowed_mapping_names = frozenset(
            {'C', 'p', 'f', 'a', 'volpct'}
            if mapping_names is None else mapping_names
        )
        self._function_names = frozenset(
            {'exp', 'log'} if function_names is None else function_names
        )
        supported_functions = {
            'abs', 'exp', 'log', 'log10', 'max', 'min', 'sqrt',
        }
        unsupported_functions = self._function_names - supported_functions
        if unsupported_functions:
            raise ReactionDefinitionError(
                "Unsupported kinetic expression function(s): "
                + ', '.join(sorted(unsupported_functions))
            )
        nodes = list(ast.walk(tree))
        if len(nodes) > 200:
            raise ReactionDefinitionError("Custom kinetic expression is too complex")
        self._validate(tree)
        self.mapping_names = frozenset(
            node.value.id
            for node in nodes
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
        )
        self.names = frozenset(
            node.id
            for node in nodes
            if isinstance(node, ast.Name)
            and node.id not in self._function_names
            and node.id not in self.mapping_names
        )
        lowered = _SafeRateExpressionLowerer().visit(tree)
        ast.fix_missing_locations(lowered)
        self._compiled_code = compile(
            lowered,
            '<kinetic expression>',
            'eval',
        )

    def _validate(self, node) -> None:
        if isinstance(node, ast.Expression):
            self._validate(node.body)
            return
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ReactionDefinitionError("Only numeric constants are allowed in kinetic expressions")
            return
        if isinstance(node, ast.Name):
            if (
                node.id not in self._scalar_names
                and node.id not in self.parameter_names
            ):
                raise ReactionDefinitionError(
                    f"Unknown kinetic expression name {node.id!r}"
                )
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, self._allowed_binary):
            self._validate(node.left)
            self._validate(node.right)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self._allowed_unary):
            self._validate(node.operand)
            return
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in self._function_names
                or not node.args
                or node.keywords
            ):
                raise ReactionDefinitionError(
                    "Kinetic expression call uses an unsupported function or signature"
                )
            if node.func.id not in {'min', 'max'} and len(node.args) != 1:
                raise ReactionDefinitionError(
                    f"Kinetic expression function {node.func.id}(...) requires one argument"
                )
            for argument in node.args:
                self._validate(argument)
            return
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id not in self._allowed_mapping_names:
                raise ReactionDefinitionError(
                    "Kinetic mapping is not available in this expression context"
                )
            key = node.slice
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise ReactionDefinitionError("Kinetic mapping keys must be quoted component names")
            return
        raise ReactionDefinitionError(
            f"Unsupported syntax in custom kinetic expression: {type(node).__name__}"
        )

    def evaluate(self, context: Mapping[str, object]) -> float:
        try:
            local_values = {
                name: float(context[name])
                for name in self.names
            }
            local_values.update({
                name: context[name]
                for name in self.mapping_names
            })
            result = float(eval(
                self._compiled_code,
                _SAFE_RATE_EXPRESSION_GLOBALS,
                local_values,
            ))
        except (ArithmeticError, OverflowError, ValueError) as error:
            if isinstance(error, KineticsError):
                raise
            raise KineticsError(
                f"Could not evaluate kinetic expression {self.text!r}: {error}"
            ) from error
        if not math.isfinite(result):
            raise KineticsError("Kinetic expression returned a non-finite rate")
        return result


@dataclass(frozen=True)
class KineticReaction:
    """Validated kinetic reaction with an explicit numerical unit basis."""

    reaction: StoichiometricReaction
    model: str
    pre_exponential: float
    activation_energy_J_mol: float
    reaction_orders: dict[str, float]
    rate_basis: Optional[str]
    concentration_unit: str
    pressure_unit: str
    rate_unit: str
    expression: Optional[SafeRateExpression]
    parameters: dict[str, float]
    name: Optional[str] = None
    validation_warnings: tuple[str, ...] = ()

    @property
    def reversible(self) -> bool:
        return self.reaction.arrow in {'<=>', '<->', '='}

    @property
    def signed_net_rate(self) -> bool:
        return self.model == 'custom_net'

    @property
    def rate_output_basis(self) -> str:
        return (
            'catalyst_mass'
            if self.rate_unit in _CATALYST_MASS_RATE_UNITS
            else 'fluid_volume'
        )

    def rate_constant(self, T: float) -> float:
        log_value = self.log_rate_constant(T)
        if log_value == -math.inf or log_value < -745.0:
            return 0.0
        if log_value > 709.0:
            raise KineticsError("Arrhenius rate constant overflowed")
        return math.exp(log_value)

    def log_rate_constant(self, T: float) -> float:
        if not math.isfinite(T) or T <= 0.0:
            raise KineticsError("Kinetic temperature must be positive and finite")
        if self.pre_exponential == 0.0:
            return -math.inf
        return (
            math.log(self.pre_exponential)
            - self.activation_energy_J_mol / (R * T)
        )


def kinetic_reaction_from_mapping(
    definition: Mapping[str, object],
    component_symbols: Iterable[str],
    component_metadata: Optional[Mapping[str, object]] = None,
) -> KineticReaction:
    """Build a canonical kinetic reaction from a normalized PFD mapping."""
    if not isinstance(definition, Mapping):
        raise ReactionDefinitionError("Kinetic reaction definition must be a mapping")
    symbols = tuple(str(component) for component in component_symbols)
    reaction = parse_reaction_equation(definition.get('equation'), symbols)
    if reaction.arrow not in {'->', '=>', '<=>', '<->'}:
        raise ReactionDefinitionError(
            "Kinetic reactions require -> or the reversible arrow <=> or <->"
        )

    allowed = KINETIC_PARAMETER_FIELDS
    dynamic = ('order_', 'param_')
    unexpected = sorted(
        str(key) for key in definition
        if key not in allowed and not any(str(key).startswith(prefix) for prefix in dynamic)
    )
    if unexpected:
        raise ReactionDefinitionError(
            "Kinetic reaction does not accept parameter(s): " + ', '.join(unexpected)
        )

    model = str(definition.get('kinetics_type', definition.get('type', 'power_law'))).strip().lower()
    if model in {'powerlaw', 'power-law'}:
        model = 'power_law'
    if model == 'custom-net':
        model = 'custom_net'
    if definition.get('expression') is not None:
        if model == 'power_law' and 'type' not in definition and 'kinetics_type' not in definition:
            model = 'custom'
    if model not in {'power_law', 'custom', 'custom_net'}:
        raise ReactionDefinitionError(
            "Kinetic type must be power_law, custom, or custom_net"
        )
    if model == 'custom_net' and reaction.arrow in {'<=>', '<->'}:
        raise ReactionDefinitionError(
            "custom_net kinetics requires an irreversible -> or => equation; "
            "reversible equations use a nonnegative forward prefactor with Q/K"
        )

    if 'A' not in definition and model == 'power_law':
        raise ReactionDefinitionError("Power-law kinetics requires pre-exponential factor A")
    pre_exponential = _finite_number(definition.get('A', 1.0), 'A')
    if pre_exponential < 0.0:
        raise ReactionDefinitionError("A must be nonnegative")
    activation = _finite_number(
        definition.get('Ea', definition.get('activation_energy', 0.0)),
        'Ea',
    )
    if activation < 0.0:
        raise ReactionDefinitionError("Ea must be nonnegative")
    ea_unit = _required_unit(definition, 'Ea_unit', _ACTIVATION_ENERGY_FACTORS)
    activation *= _ACTIVATION_ENERGY_FACTORS[ea_unit]
    concentration_unit = _required_unit(
        definition, 'concentration_unit', _CONCENTRATION_FACTORS
    )
    pressure_unit = _required_unit(definition, 'pressure_unit', _PRESSURE_FACTORS)
    rate_unit = _required_unit(definition, 'rate_unit', _RATE_FACTORS)

    basis = None
    if model == 'power_law':
        raw_basis = str(definition.get('rate_basis', definition.get('basis', ''))).strip().lower()
        basis = _RATE_BASIS_ALIASES.get(raw_basis)
        if basis is None:
            raise ReactionDefinitionError(
                "Power-law kinetics requires rate_basis=concentration, "
                "partial_pressure, fugacity, or activity"
            )

    orders: dict[str, float] = {}
    for key, value in definition.items():
        if not str(key).startswith('order_'):
            continue
        component = str(key)[6:]
        if component not in symbols:
            raise ReactionDefinitionError(
                f"Reaction order references unknown component '{component}'"
            )
        orders[component] = _finite_number(value, f"order_{component}")
    if model == 'power_law' and not orders:
        orders = {
            component: -coefficient
            for component, coefficient in reaction.stoichiometry.items()
            if coefficient < 0.0
        }

    parameters: dict[str, float] = {}
    for key, value in definition.items():
        if not str(key).startswith('param_'):
            continue
        name = str(key)[6:]
        if not name.isidentifier() or name in _RESERVED_EXPRESSION_NAMES:
            raise ReactionDefinitionError(f"Invalid custom kinetic parameter name {name!r}")
        parameters[name] = _finite_number(value, f"param_{name}")
    expression = (
        SafeRateExpression(definition.get('expression'), parameters)
        if model in {'custom', 'custom_net'} else None
    )

    warnings = (
        validate_reaction_balance(reaction, component_metadata)
        if component_metadata is not None else []
    )
    name = definition.get('name', definition.get('label'))
    return KineticReaction(
        reaction=reaction,
        model=model,
        pre_exponential=pre_exponential,
        activation_energy_J_mol=activation,
        reaction_orders=orders,
        rate_basis=basis,
        concentration_unit=concentration_unit,
        pressure_unit=pressure_unit,
        rate_unit=rate_unit,
        expression=expression,
        parameters=parameters,
        name=str(name) if name is not None else None,
        validation_warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class HomogeneousRateState:
    """Thermodynamic state and kinetic driving variables at one point."""

    T: float
    P: float
    phase: str
    component_flows: dict[str, float]
    composition: dict[str, float]
    molar_density_kmol_m3: float
    concentrations_kmol_m3: dict[str, float]
    activities: dict[str, float]
    partial_pressures_bar: dict[str, float]
    fugacities_bar: dict[str, float]

    @classmethod
    def from_flows(cls, thermo, component_flows, T: float, P: float, phase: str):
        phase_name = str(phase).strip().lower()
        if phase_name == 'gas':
            phase_name = 'vapor'
        if phase_name not in {'vapor', 'liquid'}:
            raise KineticsError("Homogeneous kinetics requires phase='vapor' or phase='liquid'")
        flows = {
            component: float(component_flows.get(component, 0.0))
            for component in thermo.components
        }
        if any(not math.isfinite(value) or value < 0.0 for value in flows.values()):
            raise KineticsError("Kinetic rate state requires finite nonnegative component flows")
        total = sum(flows.values())
        if total <= 0.0:
            raise KineticsError("Kinetic rate state has zero total flow")
        composition = {component: value / total for component, value in flows.items()}
        vapor_fraction = 1.0 if phase_name == 'vapor' else 0.0
        density = float(thermo.mixture_molar_density(
            composition, T, P, vapor_fraction,
            x=composition if phase_name == 'liquid' else None,
            y=composition if phase_name == 'vapor' else None,
        ))
        if not math.isfinite(density) or density <= 0.0:
            raise KineticsError("Thermodynamic model returned nonpositive molar density")
        activities = {
            component: float(value)
            for component, value in thermo.component_activities(
                T, P, composition, phase_name
            ).items()
        }
        return cls(
            T=float(T),
            P=float(P),
            phase=phase_name,
            component_flows=flows,
            composition=composition,
            molar_density_kmol_m3=density,
            concentrations_kmol_m3={
                component: fraction * density
                for component, fraction in composition.items()
            },
            activities=activities,
            partial_pressures_bar={
                component: fraction * P
                for component, fraction in composition.items()
            },
            fugacities_bar={
                component: activity * P_REF
                for component, activity in activities.items()
            },
        )

    def raw_liquid_volume_percent(self, thermo) -> dict[str, float]:
        """Apparent pure-component liquid volume over dynamic mixture volume."""
        if self.phase != 'liquid':
            raise KineticsError(
                "Kinetic volpct mapping requires a homogeneous liquid phase"
            )
        total = sum(self.component_flows.values())
        mixture_volume = total / self.molar_density_kmol_m3
        if not math.isfinite(mixture_volume) or mixture_volume <= 0.0:
            raise KineticsError(
                "Kinetic volpct mapping requires positive dynamic mixture volume"
            )
        values = {}
        for component, amount in self.component_flows.items():
            if amount <= 0.0:
                values[component] = 0.0
                continue
            pure_volume = float(thermo.mixture_liquid_molar_volume(
                {component: 1.0}, self.T
            ))
            if not math.isfinite(pure_volume) or pure_volume <= 0.0:
                raise KineticsError(
                    "Thermodynamic model returned nonpositive pure-liquid molar "
                    f"volume for {component!r}"
                )
            values[component] = 100.0 * amount * pure_volume / mixture_volume
        return values

    def expression_context(
        self,
        reaction: KineticReaction,
        thermo=None,
    ) -> dict[str, object]:
        c_factor = _CONCENTRATION_FACTORS[reaction.concentration_unit]
        p_factor = _PRESSURE_FACTORS[reaction.pressure_unit]
        context = {
            'T': self.T,
            'R': R,  # J/(mol K), independent of the declared rate units.
            'C': {key: value * c_factor for key, value in self.concentrations_kmol_m3.items()},
            'p': {key: value * p_factor for key, value in self.partial_pressures_bar.items()},
            'f': {key: value * p_factor for key, value in self.fugacities_bar.items()},
            'a': dict(self.activities),
            **reaction.parameters,
        }
        if reaction.expression is not None and 'volpct' in reaction.expression.mapping_names:
            if thermo is None:
                raise KineticsError(
                    "Kinetic volpct mapping requires the active thermodynamic model"
                )
            context['volpct'] = self.raw_liquid_volume_percent(thermo)
        return context


def _forward_rate_details(
    reaction: KineticReaction,
    state: HomogeneousRateState,
    thermo=None,
) -> tuple[float, float]:
    """Return the declared-unit forward prefactor and its natural logarithm."""
    context = state.expression_context(reaction, thermo)
    k = reaction.rate_constant(state.T)
    context['k'] = k
    if reaction.model in {'custom', 'custom_net'}:
        raw_rate = reaction.expression.evaluate(context)
        log_forward_rate = math.log(raw_rate) if raw_rate > 0.0 else -math.inf
    else:
        values = {
            'concentration': context['C'],
            'partial_pressure': context['p'],
            'fugacity': context['f'],
            'activity': context['a'],
        }[reaction.rate_basis]
        log_forward_rate = reaction.log_rate_constant(state.T)
        if reaction.pre_exponential == 0.0:
            raw_rate = 0.0
        for component, order in reaction.reaction_orders.items():
            value = float(values.get(component, 0.0))
            if value < 0.0:
                raise KineticsError("Power-law kinetic variables must be nonnegative")
            if value == 0.0 and order < 0.0:
                raise KineticsError("A negative kinetic order at zero driving variable is undefined")
            if value == 0.0 and order > 0.0:
                if reaction.reversible and math.isfinite(log_forward_rate):
                    log_forward_rate += order * math.log(1.0e-300)
                else:
                    log_forward_rate = -math.inf
            elif order != 0.0 and math.isfinite(log_forward_rate):
                log_forward_rate += order * math.log(value)
        raw_rate = (
            math.exp(log_forward_rate)
            if math.isfinite(log_forward_rate) and log_forward_rate <= 709.0
            else (0.0 if log_forward_rate == -math.inf else math.inf)
        )

    if raw_rate < 0.0 and not reaction.signed_net_rate:
        raise KineticsError("Forward kinetic expression returned a negative rate")
    return raw_rate, log_forward_rate


def reaction_log_driving_force(
    reaction: KineticReaction,
    state: HomogeneousRateState,
    thermo,
) -> float:
    """Return ln(Q/K) for a reversible kinetic reaction."""
    log_q = 0.0
    for component, coefficient in reaction.reaction.stoichiometry.items():
        activity = max(float(state.activities.get(component, 0.0)), 1.0e-300)
        log_q += coefficient * math.log(activity)
    log_k = float(thermo.reaction_log_equilibrium_constant(
        reaction.reaction.stoichiometry,
        state.T,
    ))
    return log_q - log_k


def evaluate_forward_reaction_rate(
    reaction: KineticReaction,
    state: HomogeneousRateState,
    thermo=None,
) -> float:
    """Evaluate the uncorrected rate in its canonical dimensional basis."""
    raw_rate, _log_rate = _forward_rate_details(reaction, state, thermo)
    rate = raw_rate * _RATE_FACTORS[reaction.rate_unit]
    if not math.isfinite(rate):
        raise KineticsError("Forward kinetic reaction rate is non-finite")
    return rate


def evaluate_reaction_rate(reaction: KineticReaction, state: HomogeneousRateState, thermo) -> float:
    """Evaluate one net rate in its canonical dimensional basis."""
    raw_rate, log_forward_rate = _forward_rate_details(reaction, state, thermo)
    if reaction.reversible:
        delta = reaction_log_driving_force(reaction, state, thermo)
        if raw_rate == 0.0 and reaction.model == 'custom' and delta > 0.0:
            raise KineticsError(
                "Reversible custom kinetic prefactor is zero on the product-side "
                "boundary; write a numerically nonzero symmetric prefactor"
            )
        if delta <= 0.0:
            raw_rate *= -math.expm1(max(delta, -700.0))
        else:
            reverse_log_rate = log_forward_rate + delta
            if reverse_log_rate > 709.0:
                raise KineticsError("Thermodynamic reverse rate overflowed")
            reverse_scale = (
                math.exp(reverse_log_rate)
                if math.isfinite(reverse_log_rate) else 0.0
            )
            raw_rate = -reverse_scale * (-math.expm1(-delta))

    rate = raw_rate * _RATE_FACTORS[reaction.rate_unit]
    if not math.isfinite(rate):
        raise KineticsError("Kinetic reaction rate is non-finite")
    return rate


@dataclass(frozen=True)
class CSTRKineticSolution:
    inlet_component_flows: dict[str, float]
    outlet_component_flows: dict[str, float]
    extents_kmol_h: tuple[float, ...]
    rates_kmol_m3_h: tuple[float, ...]
    residuals_kmol_h: tuple[float, ...]
    driving_residuals: tuple[Optional[float], ...]
    reaction_statuses: tuple[str, ...]
    iterations: int
    objective: float
    rate_state: HomogeneousRateState

    @property
    def maximum_residual_kmol_h(self) -> float:
        return max((abs(value) for value in self.residuals_kmol_h), default=0.0)


def solve_isothermal_cstr(
    inlet_component_flows: Mapping[str, float],
    reactions: Iterable[KineticReaction],
    thermo,
    T: float,
    P: float,
    phase: str,
    volume_m3: float,
    *,
    residual_tolerance: float = 1.0e-8,
    max_iterations: int = 500,
) -> CSTRKineticSolution:
    """Solve simultaneous steady-state CSTR balances in reaction extents."""
    import numpy as np
    from scipy.optimize import LinearConstraint, brentq, minimize

    reaction_tuple = tuple(reactions)
    if not reaction_tuple:
        raise KineticsError("At least one kinetic reaction is required")
    incompatible = [
        reaction.rate_unit for reaction in reaction_tuple
        if reaction.rate_output_basis != 'fluid_volume'
    ]
    if incompatible:
        raise KineticsError(
            "CSTR requires fluid-volume rate units; incompatible rate_unit(s): "
            + ', '.join(incompatible)
        )
    volume = float(volume_m3)
    if not math.isfinite(volume) or volume <= 0.0:
        raise KineticsError("CSTR volume must be positive and finite")
    components = tuple(str(component) for component in thermo.components)
    unknown = sorted(set(inlet_component_flows) - set(components))
    if unknown:
        raise KineticsError(
            "CSTR inlet contains components outside the fluid backend: " + ', '.join(unknown)
        )
    inlet = {component: float(inlet_component_flows.get(component, 0.0)) for component in components}
    if any(not math.isfinite(value) or value < 0.0 for value in inlet.values()):
        raise KineticsError("CSTR inlet component flows must be finite and nonnegative")
    n0 = np.asarray([inlet[component] for component in components], dtype=float)
    matrix = np.asarray([
        [specification.reaction.stoichiometry.get(component, 0.0) for specification in reaction_tuple]
        for component in components
    ], dtype=float)
    flow_scale = max(float(np.sum(n0)), 1.0)
    mole_tolerance = max(1.0e-13 * flow_scale, 1.0e-14)

    def flows_for(extents):
        return n0 + matrix @ np.asarray(extents, dtype=float)

    def state_and_rates(extents):
        flows = flows_for(extents)
        if float(np.min(flows)) < -mole_tolerance:
            raise KineticsError("CSTR trial produced a negative component flow")
        clean = np.maximum(flows, 0.0)
        state = HomogeneousRateState.from_flows(
            thermo,
            {component: float(clean[index]) for index, component in enumerate(components)},
            T, P, phase,
        )
        rates = np.asarray([
            evaluate_reaction_rate(reaction, state, thermo)
            for reaction in reaction_tuple
        ], dtype=float)
        return state, rates

    def residual_vector(extents):
        _state, rates = state_and_rates(extents)
        return np.asarray(extents, dtype=float) - volume * rates

    lower = []
    upper = []
    for reaction in reaction_tuple:
        forward = min(
            (inlet.get(component, 0.0) / (-coefficient)
             for component, coefficient in reaction.reaction.stoichiometry.items()
             if coefficient < 0.0),
            default=0.0,
        )
        reverse = min(
            (inlet.get(component, 0.0) / coefficient
             for component, coefficient in reaction.reaction.stoichiometry.items()
             if coefficient > 0.0),
            default=0.0,
        )
        lower.append(
            -reverse if reaction.reversible or reaction.signed_net_rate else 0.0
        )
        upper.append(forward)

    iterations = 0
    objective = 0.0
    scalar_driving_residual = None
    if len(reaction_tuple) == 1:
        low = float(lower[0])
        high = float(upper[0])
        if high < low:
            raise KineticsError("CSTR reaction has no feasible extent interval")
        if reaction_tuple[0].reversible:
            def scalar(value):
                state, _rates = state_and_rates([value])
                _raw_forward, log_forward = _forward_rate_details(
                    reaction_tuple[0], state, thermo
                )
                log_forward += math.log(_RATE_FACTORS[reaction_tuple[0].rate_unit])
                extent = float(value)
                if extent == 0.0:
                    required_delta = 0.0
                elif not math.isfinite(log_forward):
                    return math.copysign(1.0e300, extent)
                else:
                    log_ratio = (
                        math.log(abs(extent)) - math.log(volume) - log_forward
                    )
                    if extent > 0.0:
                        if log_ratio >= 0.0:
                            return 1.0e300
                        required_delta = math.log1p(-math.exp(log_ratio))
                    elif log_ratio > 50.0:
                        required_delta = log_ratio + math.log1p(math.exp(-log_ratio))
                    else:
                        required_delta = math.log1p(math.exp(log_ratio))
                return reaction_log_driving_force(
                    reaction_tuple[0], state, thermo
                ) - required_delta
        else:
            def scalar(value):
                return float(residual_vector([value])[0])
        f_low = scalar(low)
        f_high = scalar(high)
        if abs(f_low) <= residual_tolerance:
            extents = np.asarray([low])
        elif abs(f_high) <= residual_tolerance:
            extents = np.asarray([high])
        elif f_low * f_high > 0.0:
            raise KineticsError(
                "CSTR steady-state residual is not bracketed over the feasible extent interval"
            )
        else:
            extents, result = brentq(
                scalar, low, high, xtol=5.0e-324,
                rtol=8.881784197001252e-16,
                maxiter=int(max_iterations), full_output=True,
            )
            extents = np.asarray([extents])
            iterations = int(result.iterations)
        if reaction_tuple[0].reversible:
            scalar_driving_residual = float(scalar(float(extents[0])))
    else:
        constraint = LinearConstraint(matrix, -n0, np.full(len(components), np.inf))
        bounds = tuple(
            (None, None)
            if reaction.reversible or reaction.signed_net_rate
            else (0.0, None)
            for reaction in reaction_tuple
        )
        starts = [np.zeros(len(reaction_tuple), dtype=float)]
        for index, reaction in enumerate(reaction_tuple):
            if upper[index] > 0.0:
                trial = np.zeros(len(reaction_tuple), dtype=float)
                trial[index] = 0.1 * upper[index]
                starts.append(trial)
            if (reaction.reversible or reaction.signed_net_rate) and lower[index] < 0.0:
                trial = np.zeros(len(reaction_tuple), dtype=float)
                trial[index] = 0.1 * lower[index]
                starts.append(trial)

        candidates = []
        messages = []
        def objective_function(extents):
            try:
                scaled = residual_vector(extents) / flow_scale
                return 0.5 * float(np.dot(scaled, scaled))
            except Exception:
                return 1.0e100
        for start in starts:
            solved = minimize(
                objective_function,
                start,
                method='SLSQP',
                bounds=bounds,
                constraints=(constraint,),
                options={'ftol': 1.0e-14, 'maxiter': int(max_iterations), 'disp': False},
            )
            if float(np.min(flows_for(solved.x))) >= -mole_tolerance and math.isfinite(float(solved.fun)):
                candidates.append(solved)
            if not solved.success:
                messages.append(str(solved.message))
        if not candidates:
            raise KineticsError(
                "CSTR nonlinear solve found no feasible candidate: "
                + ('; '.join(dict.fromkeys(messages)) or 'unknown solver failure')
            )
        solved = min(candidates, key=lambda candidate: float(candidate.fun))
        extents = np.asarray(solved.x, dtype=float)
        iterations = int(solved.nit)
        objective = float(solved.fun)

    state, rates = state_and_rates(extents)
    residuals = np.asarray(extents, dtype=float) - volume * rates
    maximum = float(np.max(np.abs(residuals))) if len(residuals) else 0.0
    tolerance = max(float(residual_tolerance), flow_scale * 1.0e-8)
    driving_residuals: list[Optional[float]] = [None] * len(reaction_tuple)
    statuses = ['material_rate_solved'] * len(reaction_tuple)
    if scalar_driving_residual is not None:
        driving_residuals[0] = scalar_driving_residual
        statuses[0] = (
            'near_equilibrium_conditioned'
            if maximum > tolerance else 'material_rate_solved'
        )
    if scalar_driving_residual is not None and abs(scalar_driving_residual) > 1.0e-8:
        raise KineticsError(
            "Reversible CSTR driving-force solve did not converge; "
            f"residual {scalar_driving_residual:.6g}"
        )
    if scalar_driving_residual is None and maximum > tolerance:
        raise KineticsError(
            f"CSTR material/rate solve did not converge; maximum residual {maximum:.6g} kmol/h"
        )
    final_flows = flows_for(extents)
    if float(np.min(final_flows)) < -mole_tolerance:
        raise KineticsError("CSTR converged to a negative component flow")
    outlet = {
        component: float(max(final_flows[index], 0.0))
        for index, component in enumerate(components)
        if final_flows[index] > mole_tolerance
    }
    return CSTRKineticSolution(
        inlet_component_flows=inlet,
        outlet_component_flows=outlet,
        extents_kmol_h=tuple(float(value) for value in extents),
        rates_kmol_m3_h=tuple(float(value) for value in rates),
        residuals_kmol_h=tuple(float(value) for value in residuals),
        driving_residuals=tuple(driving_residuals),
        reaction_statuses=tuple(statuses),
        iterations=iterations,
        objective=objective,
        rate_state=state,
    )
