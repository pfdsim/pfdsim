"""Hayden-O'Connell second-virial coefficient provider.

Implements Hayden and O'Connell, Ind. Eng. Chem. Process Des. Dev. 14
(1975) 209-216, equations 5-38, together with generalized association and
solvation parameters from supplementary Tables IV and V.  The implementation
follows the supplied SECVIR program where the typeset paper is ambiguous.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .common import R, ThermodynamicsError


@dataclass(frozen=True)
class _Jet2:
    """Scalar value with exact first and second temperature derivatives."""

    value: float
    first: float = 0.0
    second: float = 0.0

    @staticmethod
    def coerce(value: float | '_Jet2') -> '_Jet2':
        return value if isinstance(value, _Jet2) else _Jet2(float(value))

    def __add__(self, other: float | '_Jet2') -> '_Jet2':
        other = self.coerce(other)
        return _Jet2(
            self.value + other.value,
            self.first + other.first,
            self.second + other.second,
        )

    __radd__ = __add__

    def __neg__(self) -> '_Jet2':
        return _Jet2(-self.value, -self.first, -self.second)

    def __sub__(self, other: float | '_Jet2') -> '_Jet2':
        return self + -self.coerce(other)

    def __rsub__(self, other: float | '_Jet2') -> '_Jet2':
        return self.coerce(other) - self

    def __mul__(self, other: float | '_Jet2') -> '_Jet2':
        other = self.coerce(other)
        return _Jet2(
            self.value * other.value,
            self.first * other.value + self.value * other.first,
            self.second * other.value
            + 2.0 * self.first * other.first
            + self.value * other.second,
        )

    __rmul__ = __mul__

    def reciprocal(self) -> '_Jet2':
        if self.value == 0.0:
            raise ZeroDivisionError('division by zero in HOC derivative')
        inverse = 1.0 / self.value
        return _Jet2(
            inverse,
            -self.first * inverse**2,
            2.0 * self.first**2 * inverse**3 - self.second * inverse**2,
        )

    def __truediv__(self, other: float | '_Jet2') -> '_Jet2':
        return self * self.coerce(other).reciprocal()

    def __rtruediv__(self, other: float | '_Jet2') -> '_Jet2':
        return self.coerce(other) * self.reciprocal()

    def __pow__(self, exponent: float) -> '_Jet2':
        power = float(exponent)
        value = self.value**power
        first_factor = power * self.value ** (power - 1.0)
        second_factor = power * (power - 1.0) * self.value ** (power - 2.0)
        return _Jet2(
            value,
            first_factor * self.first,
            second_factor * self.first**2 + first_factor * self.second,
        )

    def exp(self) -> '_Jet2':
        value = math.exp(self.value)
        return _Jet2(
            value,
            value * self.first,
            value * (self.second + self.first**2),
        )


@dataclass(frozen=True)
class HOCComponentParameters:
    """Temperature-independent HOC parameters for one component."""

    epsilon_K: float
    sigma3_A3: float
    omega_prime: float
    dipole_D: float
    eta: float
    association_group: str
    cas: str = ''


HOC_GROUP_ASSOCIATION_PARAMETERS = {
    'hydrogen cyanide': 0.12,
    'carbon dioxide': 0.16,
    'carbon disulfide': 0.34,
    'nitrous oxide': 0.09,
    'phosphine': 0.05,
    'water': 1.55,
    'hydroxyl': 1.55,
    'phenyl hydroxyl': 0.32,
    'furan': 0.13,
    'aldehyde': 0.58,
    'ketone': 0.90,
    'formate': 0.20,
    'ester': 0.70,
    'organic acid': 4.50,
    'ammonia': 0.20,
    'amine': 0.20,
    'nitrile': 1.65,
    'nitro': 1.66,
    'thiol': 0.15,
}


# A Table V component is generalized only when its Table IV entry supplies a
# parenthesized group label, or when Table V marks the row with superscript e
# as representing multiple pairs with the same solvating groups. Unlabelled
# components remain identity-specific through _HOC_IDENTITY_GROUPS.
HOC_GROUP_SOLVATION_PARAMETERS = {
    frozenset(('carbon dioxide', 'water')): 0.32,
    frozenset(('carbon dioxide', 'hydroxyl')): 0.32,
    frozenset(('carbon dioxide', 'nitro')): 0.16,
    frozenset(('carbon disulfide', 'ketone')): 0.13,
    frozenset(('nitrous oxide', 'water')): 0.17,
    frozenset(('nitrous oxide', 'hydroxyl')): 0.17,
    frozenset(('ethylene', 'hydroxyl')): 0.12,
    frozenset(('benzene', 'ketone')): 0.50,
    frozenset(('benzene', 'nitro')): 1.00,
    frozenset(('chloromethane', 'ketone')): 0.20,
    # The SI repeats "chloromethane-acetone" here; PFDSim's source note
    # identifies the second row as dichloromethane-acetone.
    frozenset(('dichloromethane', 'ketone')): 0.92,
    frozenset(('chloroform', 'ketone')): 1.26,
    frozenset(('chloroform', 'formate')): 1.20,
    frozenset(('chloroform', 'ester')): 1.65,
    frozenset(('chloroform', 'amine')): 1.57,
    frozenset(('ketone', 'nitro')): 1.63,
    frozenset(('aldehyde', 'nitrile')): 2.50,
}


HOC_EXACT_SOLVATION_PARAMETERS = {
    frozenset(('7647-01-0', '7732-18-5')): 1.38,
    frozenset(('7446-09-5', '628-28-4')): 0.58,
    frozenset(('7664-41-7', '74-85-1')): 0.20,
    frozenset(('71-43-2', '67-66-3')): 0.12,
    frozenset(('67-66-3', '60-29-7')): 0.95,
}


_HOC_IDENTITY_GROUPS = {
    '74-90-8': 'hydrogen cyanide',
    '124-38-9': 'carbon dioxide',
    '75-15-0': 'carbon disulfide',
    '10024-97-2': 'nitrous oxide',
    '7803-51-2': 'phosphine',
    # Table IV explicitly assigns water to the transferable Hydroxyl group.
    '7732-18-5': 'hydroxyl',
    '7664-41-7': 'ammonia',
    '7446-09-5': 'sulfur dioxide',
    '7647-01-0': 'hydrogen chloride',
    '67-66-3': 'chloroform',
    '74-87-3': 'chloromethane',
    '75-09-2': 'dichloromethane',
    '110-00-9': 'furan',
    '7783-06-4': 'thiol',
    '74-85-1': 'ethylene',
    '71-43-2': 'benzene',
    '628-28-4': '1-methoxybutane',
    '60-29-7': 'diethyl ether',
}


_HOC_PSEUDO_CRITICALS = {
    '124-38-9': (241.0, 53.1),
    '74-86-2': (241.0, 53.1),
    '10024-97-2': (277.0, 63.4),
}


def _props_dict(props: object) -> dict:
    converter = getattr(props, 'to_dict', None)
    if callable(converter):
        return dict(converter())
    return {
        name: getattr(props, name, None)
        for name in (
            'symbol', 'name', 'formula', 'CAS', 'smiles', 'MW', 'Tc', 'Pc',
            'dipole_moment', 'radius_of_gyration',
            'modified_radius_of_gyration', 'hoc_eta', 'property_sources',
        )
    }


def hoc_association_group(component: str, props: object) -> str:
    """Return the generalized HOC group represented in SI Tables IV/V."""
    cas = str(getattr(props, 'CAS', '') or '').strip()
    special = _HOC_IDENTITY_GROUPS.get(cas)
    if special is not None:
        return special

    smiles = str(getattr(props, 'smiles', '') or '').strip()
    if not smiles:
        return 'unclassified'
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)
    except Exception:
        molecule = None
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return 'unclassified'

    elements = {atom.GetSymbol() for atom in molecule.GetAtoms()}

    def count(smarts: str) -> int:
        pattern = Chem.MolFromSmarts(smarts)
        return (
            len(molecule.GetSubstructMatches(pattern))
            if pattern is not None else 0
        )

    flags = []
    acid_count = count('[CX3](=[OX1])[OX2H1]')
    formate_count = count('[CX3H1](=[OX1])[OX2][#6]')
    ester_count = count('[CX3](=[OX1])[OX2][#6]')
    aldehyde_count = (
        count('[CX3H1](=[OX1])[#6]')
        + count('[CX3H2]=[OX1]')
    )
    ketone_count = count('[#6][CX3](=[OX1])[#6]')
    nitro_count = count('[NX3+](=[OX1])[OX1-]')
    nitrile_count = count('[#6]#[NX1]')
    phenol_count = count('[OX2H1]-[c]')
    alcohol_count = count('[OX2H1;!$([O]-[C,S,P]=O)]-[#6;!a]')
    amine_count = (
        count('[NX3;!$(N-[CX3]=O)]') + count('[nH0]')
        if nitro_count == 0 else 0
    )
    thiol_count = count('[SX2H1]')
    furan_count = count('[o;r5]')
    ether_count = count('[OX2;!$([O]-[C,S,P]=O)]([#6])[#6]')

    for name, occurrences in (
        ('organic acid', acid_count),
        ('formate', formate_count),
        ('ester', ester_count if formate_count == 0 else 0),
        ('aldehyde', aldehyde_count),
        ('ketone', ketone_count),
        ('nitro', nitro_count),
        ('nitrile', nitrile_count),
        ('phenyl hydroxyl', phenol_count),
        ('hydroxyl', alcohol_count),
        ('amine', amine_count),
        ('thiol', thiol_count),
        ('furan', furan_count),
        ('ether', ether_count if ester_count == 0 and furan_count == 0 else 0),
    ):
        if occurrences:
            flags.append((name, occurrences))

    # HOC supplies one eta per interacting group, not an additive
    # multifunctional prescription. Multiple copies or distinct groups are
    # therefore deliberately left unclassified.
    if len(flags) == 1 and flags[0][1] == 1:
        return flags[0][0]
    if flags:
        return 'multifunctional'

    total_hydrogens = sum(atom.GetTotalNumHs() for atom in molecule.GetAtoms())
    if elements <= {'C', 'F'} and 'C' in elements and total_hydrogens == 0:
        return 'perfluorocarbon'
    if elements <= {'C', 'H'} and 'C' in elements:
        if any(atom.GetIsAromatic() for atom in molecule.GetAtoms()):
            return 'aromatic hydrocarbon'
        if any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            for bond in molecule.GetBonds()
        ):
            return 'olefin'
        return 'hydrocarbon'
    return 'unclassified'


class HaydenOConnellSecondVirialProvider:
    """HOC pure and cross second-virial provider with exact T derivatives."""

    name = 'HOC'

    def __init__(
        self,
        components: Sequence[str],
        component_properties: Mapping[str, object],
        *,
        binary_etas: Mapping[tuple[str, str], float] | None = None,
        resolver_properties: Mapping[str, dict] | None = None,
        allow_online: bool = True,
        chemical_database=None,
    ) -> None:
        self.components = tuple(components)
        self.parameters: list[HOCComponentParameters] = []
        self.dipole_results: dict[str, object] = {}
        self.modified_radius_results: dict[str, object] = {}
        self.pure_eta_overrides: dict[str, float] = {}
        self._binary_etas = {
            frozenset((str(first), str(second))): float(value)
            for (first, second), value in (binary_etas or {}).items()
        }
        self.binary_eta_overrides = dict(self._binary_etas)
        self.warnings: list[str] = []
        self._matrix_cache: dict[tuple[float, int], tuple[tuple[float, ...], ...]] = {}
        self._physical_matrix_cache: dict[
            tuple[float, int], tuple[tuple[float, ...], ...]
        ] = {}
        self._association_matrix_cache: dict[
            tuple[float, int], tuple[tuple[float, ...], ...]
        ] = {}
        self._term_cache: dict[tuple[float, int, int], tuple[_Jet2, _Jet2]] = {}

        if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
            from ..property_resolver import get_property_resolver
        else:
            from property_resolver import get_property_resolver
        resolver = get_property_resolver()

        for pair, value in self._binary_etas.items():
            if len(pair) != 2 or not math.isfinite(value):
                raise ThermodynamicsError(
                    "HOC binary eta values must be finite and reference two "
                    "distinct components"
                )
            unknown = pair - set(self.components)
            if unknown:
                raise ThermodynamicsError(
                    "HOC binary eta names unknown component(s): "
                    + ', '.join(sorted(unknown))
                )

        for component in self.components:
            props = component_properties.get(component)
            if props is None:
                raise ThermodynamicsError(
                    f"HOC second virial provider has no properties for '{component}'"
                )
            if not getattr(props, 'smiles', None) and chemical_database is not None:
                try:
                    structure = chemical_database.resolve_smiles_info(
                        component,
                        fetch_online=allow_online,
                        props=(resolver_properties or {}).get(component) or props,
                    )
                except Exception:
                    structure = None
                if structure is not None and structure.smiles:
                    props.smiles = str(structure.smiles)

            known = dict(
                (resolver_properties or {}).get(component) or _props_dict(props)
            )
            try:
                dipole_result = resolver.resolve_dipole_moment(
                    component,
                    known,
                    allow_online=allow_online,
                )
                radius_result = resolver.resolve_modified_radius_of_gyration(
                    component,
                    known,
                    allow_online=allow_online,
                )
            except Exception as error:
                raise ThermodynamicsError(
                    f"HOC molecular parameters unavailable for '{component}': {error}"
                ) from error
            dipole = float(dipole_result.value)
            radius = float(radius_result.value)
            if not math.isfinite(dipole) or dipole < 0.0:
                raise ThermodynamicsError(
                    f"HOC dipole for '{component}' must be finite and nonnegative"
                )
            if not math.isfinite(radius) or radius < 0.0:
                raise ThermodynamicsError(
                    f"HOC modified radius for '{component}' must be finite and nonnegative"
                )

            cas = str(getattr(props, 'CAS', '') or known.get('CAS') or '').strip()
            pseudo = _HOC_PSEUDO_CRITICALS.get(cas)
            if pseudo is None:
                tc_raw = getattr(props, 'Tc', None)
                pc_raw = getattr(props, 'Pc', None)
                if tc_raw is None or pc_raw is None:
                    raise ThermodynamicsError(
                        f"HOC requires Tc and Pc for '{component}'"
                    )
                tc = float(tc_raw)
                pc_atm = float(pc_raw) / 1.01325
            else:
                tc, pc_atm = pseudo
            if (
                not math.isfinite(tc) or tc <= 0.0
                or not math.isfinite(pc_atm) or pc_atm <= 0.0
            ):
                raise ThermodynamicsError(
                    f"HOC Tc and Pc for '{component}' must be finite and positive"
                )

            group = hoc_association_group(component, props)
            if cas == '74-86-2':
                group = 'carbon dioxide'
            provided_eta = getattr(props, 'hoc_eta', None)
            if provided_eta is None:
                provided_eta = known.get('hoc_eta')
            if provided_eta is None:
                eta = HOC_GROUP_ASSOCIATION_PARAMETERS.get(group, 0.0)
            else:
                eta = float(provided_eta)
                if not math.isfinite(eta) or eta < 0.0:
                    raise ThermodynamicsError(
                        f"HOC pure eta for '{component}' must be finite and "
                        "nonnegative"
                    )
                self.pure_eta_overrides[component] = eta
            if group == 'multifunctional' and provided_eta is None:
                self.warnings.append(
                    f"HOC found multiple association groups for '{component}'; "
                    "the 1975 correlation gives no additive multifunctional rule, "
                    "so eta=0 was used."
                )
            self.parameters.append(self._pure_parameters(
                tc,
                pc_atm,
                radius,
                dipole,
                eta,
                group,
                cas,
            ))
            self.dipole_results[component] = dipole_result
            self.modified_radius_results[component] = radius_result

        self.associating_components = tuple(
            component
            for component, parameter in zip(self.components, self.parameters)
            if parameter.association_group == 'organic acid'
        )
        self.has_chemical_theory = bool(self.associating_components)

    @staticmethod
    def _pure_parameters(
        Tc_K: float,
        Pc_atm: float,
        radius_prime_A: float,
        dipole_D: float,
        eta: float,
        group: str,
        cas: str = '',
    ) -> HOCComponentParameters:
        omega_prime = (
            0.006 * radius_prime_A
            + 0.02087 * radius_prime_A**2
            - 0.00136 * radius_prime_A**3
        )
        epsilon = Tc_K * (
            0.748 + 0.91 * omega_prime
            - 0.4 * eta / (2.0 + 20.0 * omega_prime)
        )
        sigma3 = (2.44 - omega_prime) ** 3 * Tc_K / Pc_atm
        if dipole_D >= 1.45:
            n_value = 16.0 + 400.0 * omega_prime
            exponent = n_value / (n_value - 6.0)
            C_value = 2.882 - 1.882 * omega_prime / (0.03 + omega_prime)
            xi = dipole_D**4 / (
                C_value * 5.723e-8 * epsilon * sigma3**2 * Tc_K
            )
            epsilon *= 1.0 - exponent * xi * (
                1.0 - (exponent + 1.0) * xi / 2.0
            )
            sigma3 *= 1.0 + 3.0 * xi / (n_value - 6.0)
        if (
            not math.isfinite(omega_prime)
            or not math.isfinite(epsilon) or epsilon <= 0.0
            or not math.isfinite(sigma3) or sigma3 <= 0.0
        ):
            raise ThermodynamicsError(
                "HOC molecular parameters produced nonphysical epsilon or sigma"
            )
        return HOCComponentParameters(
            epsilon_K=epsilon,
            sigma3_A3=sigma3,
            omega_prime=omega_prime,
            dipole_D=dipole_D,
            eta=eta,
            association_group=group,
            cas=cas,
        )

    @staticmethod
    def _cross_parameters(
        first: HOCComponentParameters,
        second: HOCComponentParameters,
    ) -> tuple[float, float, float, float]:
        epsilon = (
            0.7 * math.sqrt(first.epsilon_K * second.epsilon_K)
            + 0.6 / (1.0 / first.epsilon_K + 1.0 / second.epsilon_K)
        )
        sigma3 = math.sqrt(first.sigma3_A3 * second.sigma3_A3)
        omega_prime = 0.5 * (first.omega_prime + second.omega_prime)
        reduced_dipole = 0.0
        if first.dipole_D + second.dipole_D >= 2.0:
            if first.dipole_D != 0.0 and second.dipole_D != 0.0:
                reduced_dipole = (
                    first.dipole_D * second.dipole_D
                    / (1.3805e-4 * epsilon * sigma3)
                )
            else:
                xi = (
                    first.dipole_D**2
                    * (second.epsilon_K**2 * second.sigma3_A3) ** (1.0 / 3.0)
                    * second.sigma3_A3
                    + second.dipole_D**2
                    * (first.epsilon_K**2 * first.sigma3_A3) ** (1.0 / 3.0)
                    * first.sigma3_A3
                ) / (epsilon * sigma3**2)
                n_value = 16.0 + 400.0 * omega_prime
                epsilon *= 1.0 + xi * n_value / (n_value - 6.0)
                sigma3 *= 1.0 - 3.0 * xi / (n_value - 6.0)
        return epsilon, sigma3, omega_prime, reduced_dipole

    @staticmethod
    def _eta_for_pair(
        first: HOCComponentParameters,
        second: HOCComponentParameters,
        epsilon_K: float,
    ) -> float:
        groups = frozenset((first.association_group, second.association_group))
        exact = HOC_EXACT_SOLVATION_PARAMETERS.get(
            frozenset((first.cas, second.cas))
        )
        if exact is not None:
            return exact
        if (
            first.association_group == second.association_group
            and first.association_group in HOC_GROUP_ASSOCIATION_PARAMETERS
        ):
            return HOC_GROUP_ASSOCIATION_PARAMETERS[first.association_group]
        listed = HOC_GROUP_SOLVATION_PARAMETERS.get(groups)
        if listed is not None:
            return listed
        if groups == frozenset(('hydrocarbon', 'perfluorocarbon')):
            return (
                3.0e-5 * epsilon_K
                - 7.0e-7 * epsilon_K**2
                - 3.0e-23 * epsilon_K**8
            )
        return 0.0

    def _eta_for_indices(self, i: int, j: int, epsilon_K: float) -> float:
        pair = frozenset((self.components[i], self.components[j]))
        override = self._binary_etas.get(pair)
        if override is not None:
            return override
        return self._eta_for_pair(
            self.parameters[i],
            self.parameters[j],
            epsilon_K,
        )

    @staticmethod
    def _coefficient_terms_jet(
        T: float,
        epsilon_K: float,
        sigma3_A3: float,
        omega_prime: float,
        reduced_dipole: float,
        eta: float,
    ) -> tuple[_Jet2, _Jet2]:
        """Return the free and associating HOC contributions.

        The second result is ``B_bound + B_metastable + B_chem`` from
        equations 28-31.  It is kept separate because the HOC chemical
        theory converts it to an association equilibrium constant for
        carboxylic-acid vapors instead of putting it directly in the
        truncated virial equation.
        """
        temperature = _Jet2(float(T), 1.0, 0.0)
        inverse_reduced_temperature = epsilon_K / temperature - 1.6 * omega_prime
        if reduced_dipole >= 0.25:
            reduced_dipole_prime = reduced_dipole - 0.25
        elif reduced_dipole >= 0.04:
            reduced_dipole_prime = 0.0
        else:
            reduced_dipole_prime = reduced_dipole

        b0 = 1.2618 * sigma3_A3
        free = b0 * (
            0.94
            - 1.47 * inverse_reduced_temperature
            - 0.85 * inverse_reduced_temperature**2
            + 1.015 * inverse_reduced_temperature**3
            - reduced_dipole_prime * (
                0.75
                - 3.0 * inverse_reduced_temperature
                + 2.1 * inverse_reduced_temperature**2
                + 2.1 * inverse_reduced_temperature**3
            )
        )
        A_value = -0.3 - 0.05 * reduced_dipole
        delta_H = 1.99 + 0.2 * reduced_dipole**2
        physical_bound = b0 * A_value * (
            delta_H * epsilon_K / temperature
        ).exp()
        associating = physical_bound
        if eta != 0.0:
            delta_S = (
                42800.0 / (22400.0 + epsilon_K)
                if eta > 4.0
                else 650.0 / (epsilon_K + 300.0)
            )
            chemical = (
                b0
                * math.exp(eta * (delta_S - 4.27))
                * (1.0 - (eta * 1500.0 / temperature).exp())
            )
            associating += chemical
        return free * 1.0e-6, associating * 1.0e-6

    @classmethod
    def _coefficient_jet(
        cls,
        T: float,
        epsilon_K: float,
        sigma3_A3: float,
        omega_prime: float,
        reduced_dipole: float,
        eta: float,
    ) -> _Jet2:
        """Return total HOC ``B`` for compatibility and data comparison."""
        free, associating = cls._coefficient_terms_jet(
            T,
            epsilon_K,
            sigma3_A3,
            omega_prime,
            reduced_dipole,
            eta,
        )
        return free + associating

    def _pair_terms(self, T: float, i: int, j: int) -> tuple[_Jet2, _Jet2]:
        key = (float(T), i, j)
        cached = self._term_cache.get(key)
        if cached is not None:
            return cached
        first = self.parameters[i]
        second = self.parameters[j]
        if i == j:
            epsilon = first.epsilon_K
            sigma3 = first.sigma3_A3
            omega_prime = first.omega_prime
            reduced_dipole = (
                first.dipole_D**2 / (1.3805e-4 * epsilon * sigma3)
                if first.dipole_D >= 1.0 else 0.0
            )
            eta = first.eta
        else:
            epsilon, sigma3, omega_prime, reduced_dipole = (
                self._cross_parameters(first, second)
            )
            eta = self._eta_for_indices(i, j, epsilon)
        terms = self._coefficient_terms_jet(
            T,
            epsilon,
            sigma3,
            omega_prime,
            reduced_dipole,
            eta,
        )
        if len(self._term_cache) > 12288:
            self._term_cache.clear()
        self._term_cache[key] = terms
        return terms

    def _term_matrix(
        self,
        T: float,
        order: int,
        *,
        physical: bool,
    ) -> tuple[tuple[float, ...], ...]:
        temperature = float(T)
        derivative_order = int(order)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ThermodynamicsError('HOC temperature must be positive and finite')
        if derivative_order not in (0, 1, 2):
            raise ThermodynamicsError('HOC supports derivative orders 0, 1, and 2')
        cache = self._physical_matrix_cache if physical else self._matrix_cache
        cache_key = (temperature, derivative_order)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        count = len(self.components)
        rows = [[0.0] * count for _ in range(count)]
        acid_indices = {
            index
            for index, parameter in enumerate(self.parameters)
            if parameter.association_group == 'organic acid'
        }
        for i in range(count):
            for j in range(i, count):
                free, associating = self._pair_terms(temperature, i, j)
                jet = (
                    free
                    if physical and i in acid_indices and j in acid_indices
                    else free + associating
                )
                value = (jet.value, jet.first, jet.second)[derivative_order]
                if not math.isfinite(value):
                    raise ThermodynamicsError(
                        f"HOC produced a non-finite coefficient for "
                        f"{self.components[i]}/{self.components[j]}"
                    )
                rows[i][j] = value
                rows[j][i] = value
        matrix = tuple(tuple(row) for row in rows)
        if len(cache) > 4096:
            cache.clear()
        cache[cache_key] = matrix
        return matrix

    def second_virial_matrix(
        self,
        T: float,
        order: int = 0,
    ) -> tuple[tuple[float, ...], ...]:
        return self._term_matrix(T, order, physical=False)

    def physical_second_virial_matrix(
        self,
        T: float,
        order: int = 0,
    ) -> tuple[tuple[float, ...], ...]:
        """Return the matrix used alongside acid chemical association.

        Acid/acid entries contain only ``B_free``. Other entries retain the
        total HOC coefficient because the paper recommends chemical theory
        only for carboxylic-acid association.
        """
        return self._term_matrix(T, order, physical=True)

    def association_constant_matrix(
        self,
        T: float,
        order: int = 0,
    ) -> tuple[tuple[float, ...], ...]:
        """Return acid-pair ``K_p`` values [bar^-1] and T derivatives.

        Equation 31 gives ``K_p = -B_assoc/(R*T)`` for a homodimer. An
        unlike pair receives a factor of two because the quadratic mixture
        virial sum contains both ``y_i*y_j*B_ij`` and ``y_j*y_i*B_ji``.
        """
        temperature = float(T)
        derivative_order = int(order)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ThermodynamicsError('HOC temperature must be positive and finite')
        if derivative_order not in (0, 1, 2):
            raise ThermodynamicsError('HOC supports derivative orders 0, 1, and 2')
        cache_key = (temperature, derivative_order)
        cached = self._association_matrix_cache.get(cache_key)
        if cached is not None:
            return cached

        count = len(self.components)
        rows = [[0.0] * count for _ in range(count)]
        acid_indices = [
            index
            for index, parameter in enumerate(self.parameters)
            if parameter.association_group == 'organic acid'
        ]
        temperature_jet = _Jet2(temperature, 1.0, 0.0)
        # B is in m^3/mol and R is Pa*m^3/(mol*K); 1e5 converts Pa^-1
        # to bar^-1.
        for position, i in enumerate(acid_indices):
            for j in acid_indices[position:]:
                _free, associating = self._pair_terms(temperature, i, j)
                symmetry = 1.0 if i == j else 2.0
                equilibrium = (
                    -symmetry * 1.0e5 * associating
                    / (R * temperature_jet)
                )
                value = (
                    equilibrium.value,
                    equilibrium.first,
                    equilibrium.second,
                )[derivative_order]
                if not math.isfinite(value):
                    raise ThermodynamicsError(
                        f"HOC produced a non-finite association constant for "
                        f"{self.components[i]}/{self.components[j]}"
                    )
                if derivative_order == 0 and value < 0.0:
                    raise ThermodynamicsError(
                        f"HOC produced a negative association constant for "
                        f"{self.components[i]}/{self.components[j]}"
                    )
                rows[i][j] = value
                rows[j][i] = value
        matrix = tuple(tuple(row) for row in rows)
        if len(self._association_matrix_cache) > 4096:
            self._association_matrix_cache.clear()
        self._association_matrix_cache[cache_key] = matrix
        return matrix


__all__ = [
    'HOCComponentParameters',
    'HOC_GROUP_ASSOCIATION_PARAMETERS',
    'HOC_GROUP_SOLVATION_PARAMETERS',
    'HaydenOConnellSecondVirialProvider',
    'hoc_association_group',
]
