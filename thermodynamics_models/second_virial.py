"""Truncated-second-virial vapor backend and coefficient providers.

The backend in this module knows only the thermodynamics of a quadratic
second-virial mixture.  Correlations are supplied through
``SecondVirialCoefficientProvider`` so activity-model ``-BV`` variants do not
depend on a particular source of pure or cross coefficients.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from chemicals.virial import (
    BVirial_Abbott,
    BVirial_Pitzer_Curl,
    BVirial_Tsonopoulos_extended,
)

from .common import R, ThermodynamicsError
from .tsonopoulos_kij import (
    TsonopoulosKijRecord,
    TsonopoulosPolarRecord,
    published_tsonopoulos_polar_parameters,
    resolve_tsonopoulos_kij,
)


class SecondVirialCoefficientProvider(Protocol):
    """Provider contract for symmetric ``B_ij`` matrices in m^3/mol.

    ``order`` is the temperature-derivative order.  The ``-BV`` backend uses
    orders zero through two for fugacity, residual enthalpy/entropy, and heat
    capacity.
    """

    name: str

    def second_virial_matrix(
        self,
        T: float,
        order: int = 0,
    ) -> Sequence[Sequence[float]]:
        """Return ``B_ij`` or its temperature derivative at ``T``."""


class ChemicalAssociationSecondVirialProvider(
    SecondVirialCoefficientProvider,
    Protocol,
):
    """Optional provider capability for explicit pair association."""

    has_chemical_theory: bool
    associating_components: Sequence[str]

    def physical_second_virial_matrix(
        self,
        T: float,
        order: int = 0,
    ) -> Sequence[Sequence[float]]:
        """Return the non-associating matrix used for physical fugacity."""

    def association_constant_matrix(
        self,
        T: float,
        order: int = 0,
    ) -> Sequence[Sequence[float]]:
        """Return pair ``K_p`` values [bar^-1] or their T derivatives."""


def normalize_second_virial_correlation(value: object = None) -> str:
    """Return the canonical name of a supported second-virial provider."""
    text = str(value or 'TSONOPOULOS').strip().upper().replace('_', '-')
    aliases = {
        'TSONOPOULOS': 'TSONOPOULOS',
        'TSONOPOULOS-1974': 'TSONOPOULOS',
        'TSONOPOULOS-74': 'TSONOPOULOS',
        'PITZER-CURL': 'PITZER-CURL',
        'PITZERCURL': 'PITZER-CURL',
        'PITZER-CURL-1957': 'PITZER-CURL',
        'ABBOTT': 'ABBOTT',
        'ABBOTT-LEE-KESLER': 'ABBOTT',
        'HOC': 'HOC',
        'HAYDEN-OCONNELL': 'HOC',
        "HAYDEN-O'CONNELL": 'HOC',
        'HAYDEN-AND-OCONNELL': 'HOC',
    }
    normalized = aliases.get(text)
    if normalized is None:
        raise ThermodynamicsError(
            f"Unsupported second-virial correlation {value!r}; currently "
            "available: TSONOPOULOS, PITZER-CURL, ABBOTT, HOC"
        )
    return normalized


_TSONOPOULOS_DIPOLE_SPECIES = frozenset({
    'ketone',
    'aldehyde',
    'alkyl nitrile',
    'ether',
    'carboxylic acid',
    'ester',
    'alkyl halide',
    'mercaptan',
    'sulfide',
    'disulfide',
    'alkanol',
})


def _properties_as_dict(props: object) -> dict:
    converter = getattr(props, 'to_dict', None)
    if callable(converter):
        return dict(converter())
    return {
        name: getattr(props, name, None)
        for name in ('name', 'formula', 'CAS', 'smiles', 'dipole_moment')
    }


def _tsonopoulos_species_type(component: str, props: object) -> str:
    """Classify only the molecular families supported by Tsonopoulos.

    Multifunctional and unsupported polar structures deliberately remain
    ``normal``; applying a single-family polar term to them would claim more
    than the correlation establishes.
    """
    cas = str(getattr(props, 'CAS', '') or '').strip()
    formula = str(getattr(props, 'formula', '') or '').strip()
    if cas == '7732-18-5' or formula == 'H2O':
        return 'water'
    special_species = {
        '7647-01-0': 'hydrogen chloride',
        '7664-41-7': 'ammonia',
        '7446-09-5': 'sulfur dioxide',
    }.get(cas)
    if special_species is not None:
        return special_species

    smiles = str(getattr(props, 'smiles', '') or '').strip()
    if not smiles:
        return 'normal'
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)
    except Exception:
        molecule = None
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return 'normal'

    elements = {atom.GetSymbol() for atom in molecule.GetAtoms()}
    carbon_count = sum(atom.GetSymbol() == 'C' for atom in molecule.GetAtoms())

    def matches(smarts: str) -> tuple[tuple[int, ...], ...]:
        pattern = Chem.MolFromSmarts(smarts)
        return molecule.GetSubstructMatches(pattern) if pattern is not None else ()

    candidates = []
    acid = matches('[CX3](=[OX1])[OX2H1]')
    ester = matches('[CX3](=[OX1])[OX2][#6]')
    # A carbonyl bearing H is also present in formic acid and formates.
    # Require an aldehyde carbon's other substituent to be carbon (or H).
    aldehyde = (
        matches('[CX3H1](=[OX1])[#6]')
        + matches('[CX3H2]=[OX1]')
    )
    ketone = matches('[#6][CX3](=[OX1])[#6]')
    nitrile = matches('[#6;!a]#[NX1]')
    alcohol = matches('[OX2H1;!$([O]-[C,S,P]=O)]-[#6;!a]')
    ether = matches('[OX2;!$([O]-[C,S,P]=O)]([#6])[#6]')
    disulfide = matches('[SX2]-[SX2]')
    mercaptan = matches('[SX2H1]')
    sulfide = matches('[SX2]([#6])[#6]')
    alkyl_halide = matches('[#6;!a]-[F,Cl,Br,I]')

    if len(acid) == 1 and elements <= {'C', 'H', 'O'}:
        candidates.append('carboxylic acid')
    if len(ester) == 1 and elements <= {'C', 'H', 'O'}:
        candidates.append('ester')
    if len(aldehyde) == 1 and elements <= {'C', 'H', 'O'}:
        candidates.append('aldehyde')
    if len(ketone) == 1 and elements <= {'C', 'H', 'O'}:
        candidates.append('ketone')
    if len(nitrile) == 1 and elements <= {'C', 'H', 'N'}:
        candidates.append('alkyl nitrile')
    if len(alcohol) == 1 and elements <= {'C', 'H', 'O'}:
        candidates.append(
            'methyl alcohol' if carbon_count == 1 else 'alkanol'
        )
    if len(ether) == 1 and elements <= {'C', 'H', 'O'}:
        candidates.append('ether')
    if len(disulfide) == 1 and elements <= {'C', 'H', 'S'}:
        candidates.append('disulfide')
    elif len(mercaptan) == 1 and elements <= {'C', 'H', 'S'}:
        candidates.append('mercaptan')
    elif len(sulfide) == 1 and elements <= {'C', 'H', 'S'}:
        candidates.append('sulfide')
    if alkyl_halide and elements <= {'C', 'H', 'F', 'Cl', 'Br', 'I'}:
        candidates.append('alkyl halide')

    # Ester/acid/alcohol oxygen can also satisfy broader oxygen patterns.
    precedence = (
        'carboxylic acid',
        'ester',
        'aldehyde',
        'ketone',
        'alkyl nitrile',
        'methyl alcohol',
        'alkanol',
        'ether',
        'disulfide',
        'mercaptan',
        'sulfide',
        'alkyl halide',
    )
    distinct = [name for name in precedence if name in candidates]
    if not distinct:
        return 'normal'
    selected = distinct[0]
    compatible_overlap = {
        'carboxylic acid': {'carboxylic acid'},
        'ester': {'ester', 'ether'},
    }.get(selected, {selected})
    return selected if set(distinct) <= compatible_overlap else 'normal'


def _tsonopoulos_binary_family(props: object, species_type: str) -> str:
    """Return the molecular family used by published ``k_ij`` rules."""
    if species_type == 'water':
        return 'water'
    if species_type in {'methyl alcohol', 'alkanol'}:
        return 'alkanol'
    if species_type != 'normal':
        return species_type

    smiles = str(getattr(props, 'smiles', '') or '').strip()
    if not smiles:
        return 'normal'
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)
    except Exception:
        molecule = None
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return 'normal'
    if {atom.GetSymbol() for atom in molecule.GetAtoms()} != {'C'}:
        return 'normal'
    saturated = all(
        bond.GetBondType() == Chem.BondType.SINGLE and not bond.GetIsAromatic()
        for bond in molecule.GetBonds()
    )
    if saturated and molecule.GetRingInfo().NumRings() == 0:
        return 'alkane'
    return 'hydrocarbon'


def _tsonopoulos_polar_parameters(
    species_type: str,
    dipole_D: float,
    Tc_K: float,
    Pc_Pa: float,
) -> tuple[float, float]:
    """Return the extended-correlation ``a`` and ``b`` parameters."""
    if species_type in {'simple', 'normal'}:
        return 0.0, 0.0
    if species_type == 'methyl alcohol':
        return 0.0878, 0.0525
    if species_type == 'water':
        return -0.0109, 0.0

    reduced_dipole = 1.0e5 * dipole_D**2 * (Pc_Pa / 101325.0) / Tc_K**2
    if species_type in {
        'ketone', 'aldehyde', 'alkyl nitrile', 'ether',
        'carboxylic acid', 'ester',
    }:
        return (
            -2.14e-4 * reduced_dipole
            - 4.308e-21 * reduced_dipole**8,
            0.0,
        )
    if species_type in {'alkyl halide', 'mercaptan', 'sulfide', 'disulfide'}:
        return (
            # Tsonopoulos and Heidman (1990), equation 16: 10^-11.
            -2.188e-11 * reduced_dipole**4
            - 7.831e-21 * reduced_dipole**8,
            0.0,
        )
    if species_type == 'alkanol':
        return 0.0878, 0.00908 + 0.0006957 * reduced_dipole
    return 0.0, 0.0


class TsonopoulosSecondVirialProvider:
    """Original Tsonopoulos corresponding-states pure/cross correlation.

    Cross pseudo-critical properties follow Tsonopoulos (AIChE Journal 20,
    1974, 263-272):

    ``Tc_ij = sqrt(Tc_i*Tc_j)*(1-k_ij)``

    ``Pc_ij = 4*Tc_ij*(Pc_i*Vc_i/Tc_i + Pc_j*Vc_j/Tc_j)``
    ``/(Vc_i^(1/3) + Vc_j^(1/3))^3``

    ``omega_ij = (omega_i + omega_j)/2``

    Published fitted binary values take precedence over predictive family
    correlations and class defaults.  An explicit provider override has the
    highest precedence; unsupported pairs retain ``k_ij=0``.
    """

    name = 'TSONOPOULOS'
    uses_polar_corrections = True
    uses_builtin_binary_kijs = True

    def __init__(
        self,
        components: Sequence[str],
        component_properties: Mapping[str, object],
        binary_kijs: Mapping[tuple[str, str], float] | None = None,
        resolver_properties: Mapping[str, dict] | None = None,
        allow_online: bool = True,
        chemical_database=None,
    ) -> None:
        self.components = tuple(components)
        self._parameters: list[tuple[float, float, float, float, float, float]] = []
        self.species_types: dict[str, str] = {}
        self.binary_families: dict[str, str] = {}
        self._cas_numbers: list[str] = []
        self.dipole_results: dict[str, object] = {}
        self.polar_parameter_records: dict[str, TsonopoulosPolarRecord] = {}
        self._binary_kijs = {
            frozenset((str(first), str(second))): float(value)
            for (first, second), value in (binary_kijs or {}).items()
        }
        self.binary_kij_records: dict[
            frozenset[str], TsonopoulosKijRecord
        ] = {}
        self.unresolved_binary_pairs: list[tuple[str, str]] = []
        self._matrix_cache: dict[tuple[float, int], tuple[tuple[float, ...], ...]] = {}

        for component in self.components:
            props = component_properties.get(component)
            if props is None:
                raise ThermodynamicsError(
                    f"Tsonopoulos second virial provider has no properties for "
                    f"component '{component}'"
                )
            raw = {
                'Tc': getattr(props, 'Tc', None),
                'Pc': getattr(props, 'Pc', None),
                'Vc': getattr(props, 'Vc', None),
                'omega': getattr(props, 'omega', None),
            }
            missing = [name for name, value in raw.items() if value is None]
            if missing:
                raise ThermodynamicsError(
                    f"Tsonopoulos second virial provider requires Tc, Pc, Vc, "
                    f"and omega for '{component}'; missing {', '.join(missing)}"
                )
            try:
                tc = float(raw['Tc'])
                pc_pa = float(raw['Pc']) * 1.0e5
                vc_m3_per_mol = float(raw['Vc']) * 1.0e-6
                omega = float(raw['omega'])
            except (TypeError, ValueError) as error:
                raise ThermodynamicsError(
                    f"Tsonopoulos second virial inputs for '{component}' must be numeric"
                ) from error
            if (
                not all(math.isfinite(value) for value in (tc, pc_pa, vc_m3_per_mol, omega))
                or tc <= 0.0
                or pc_pa <= 0.0
                or vc_m3_per_mol <= 0.0
            ):
                raise ThermodynamicsError(
                    f"Tsonopoulos second virial inputs for '{component}' must "
                    "have finite positive Tc, Pc, and Vc and finite omega"
                )
            if (
                self.uses_polar_corrections
                and not getattr(props, 'smiles', None)
                and chemical_database is not None
            ):
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
            species_type = (
                _tsonopoulos_species_type(component, props)
                if self.uses_polar_corrections else 'normal'
            )
            dipole = 0.0
            if species_type in _TSONOPOULOS_DIPOLE_SPECIES:
                known = dict(
                    (resolver_properties or {}).get(component)
                    or _properties_as_dict(props)
                )
                try:
                    if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                        from ..property_resolver import get_property_resolver
                    else:
                        from property_resolver import get_property_resolver
                    result = get_property_resolver().resolve_dipole_moment(
                        component,
                        known,
                        allow_online=allow_online,
                    )
                    dipole = float(result.value)
                except Exception as error:
                    raise ThermodynamicsError(
                        f"Tsonopoulos polar correction requires a dipole moment "
                        f"for '{component}': {error}"
                    ) from error
                if not math.isfinite(dipole) or dipole < 0.0:
                    raise ThermodynamicsError(
                        f"Resolved dipole moment for '{component}' must be finite "
                        "and nonnegative"
                    )
                self.dipole_results[component] = result
            a_value, b_value = _tsonopoulos_polar_parameters(
                species_type,
                dipole,
                tc,
                pc_pa,
            )
            published_polar = published_tsonopoulos_polar_parameters(
                str(getattr(props, 'CAS', '') or '')
            )
            if published_polar is not None:
                a_value, b_value = published_polar.a, published_polar.b
                self.polar_parameter_records[component] = published_polar
            self.species_types[component] = species_type
            self.binary_families[component] = _tsonopoulos_binary_family(
                props,
                species_type,
            )
            self._cas_numbers.append(str(getattr(props, 'CAS', '') or '').strip())
            self._parameters.append(
                (tc, pc_pa, vc_m3_per_mol, omega, a_value, b_value)
            )

        for pair, value in self._binary_kijs.items():
            if len(pair) != 2 or not math.isfinite(value) or value >= 1.0:
                raise ThermodynamicsError(
                    "Tsonopoulos binary k_ij values must be finite and less than 1"
                )
            unknown = pair - set(self.components)
            if unknown:
                raise ThermodynamicsError(
                    "Tsonopoulos binary k_ij names unknown component(s): "
                    + ', '.join(sorted(unknown))
                )

        for i, first in enumerate(self.components):
            for j in range(i + 1, len(self.components)):
                second = self.components[j]
                pair = frozenset((first, second))
                explicit = self._binary_kijs.get(pair)
                if explicit is not None:
                    self.binary_kij_records[pair] = TsonopoulosKijRecord(
                        value=explicit,
                        source='explicit provider override',
                        method='explicit override',
                    )
                    continue
                record = None
                if self.uses_builtin_binary_kijs:
                    record = resolve_tsonopoulos_kij(
                        self._cas_numbers[i],
                        self._cas_numbers[j],
                        self.binary_families[first],
                        self.binary_families[second],
                        self._parameters[i][2] * 1.0e6,
                        self._parameters[j][2] * 1.0e6,
                    )
                if record is None:
                    self.unresolved_binary_pairs.append((first, second))
                else:
                    self.binary_kij_records[pair] = record
        self.has_binary_kijs = bool(self.binary_kij_records)

    def _cross_parameters(self, i: int, j: int) -> tuple[float, float, float]:
        tc_i, pc_i, vc_i, omega_i, _a_i, _b_i = self._parameters[i]
        tc_j, pc_j, vc_j, omega_j, _a_j, _b_j = self._parameters[j]
        pair = frozenset((self.components[i], self.components[j]))
        record = self.binary_kij_records.get(pair)
        kij = 0.0 if i == j or record is None else record.value
        tc_ij = math.sqrt(tc_i * tc_j) * (1.0 - kij)
        denominator = (vc_i ** (1.0 / 3.0) + vc_j ** (1.0 / 3.0)) ** 3
        pc_ij = (
            4.0
            * tc_ij
            * (pc_i * vc_i / tc_i + pc_j * vc_j / tc_j)
            / denominator
        )
        omega_ij = 0.5 * (omega_i + omega_j)
        return tc_ij, pc_ij, omega_ij

    def _cross_polar_parameters(self, i: int, j: int) -> tuple[float, float]:
        _tc_i, _pc_i, _vc_i, _omega_i, a_i, b_i = self._parameters[i]
        _tc_j, _pc_j, _vc_j, _omega_j, a_j, b_j = self._parameters[j]
        if i == j:
            return a_i, b_i
        polar_i = a_i != 0.0 or b_i != 0.0
        polar_j = a_j != 0.0 or b_j != 0.0
        if not (polar_i and polar_j):
            # Tsonopoulos recommends no polar term for polar/nonpolar pairs.
            return 0.0, 0.0
        return 0.5 * (a_i + a_j), 0.5 * (b_i + b_j)

    def second_virial_matrix(
        self,
        T: float,
        order: int = 0,
    ) -> tuple[tuple[float, ...], ...]:
        temperature = float(T)
        derivative_order = int(order)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ThermodynamicsError(
                "Second-virial temperature must be positive and finite"
            )
        if derivative_order not in (0, 1, 2):
            raise ThermodynamicsError(
                "The second-virial vapor backend supports derivative orders 0, 1, and 2"
            )
        cache_key = (temperature, derivative_order)
        cached = self._matrix_cache.get(cache_key)
        if cached is not None:
            return cached

        count = len(self.components)
        rows = [[0.0] * count for _ in range(count)]
        for i in range(count):
            for j in range(i, count):
                tc_ij, pc_ij, omega_ij = self._cross_parameters(i, j)
                a_ij, b_ij = self._cross_polar_parameters(i, j)
                value = float(self._coefficient(
                    temperature,
                    tc_ij,
                    pc_ij,
                    omega_ij,
                    a_ij,
                    b_ij,
                    derivative_order,
                ))
                if not math.isfinite(value):
                    raise ThermodynamicsError(
                        f"{self.name} produced a non-finite second virial coefficient "
                        f"for {self.components[i]}/{self.components[j]}"
                    )
                rows[i][j] = value
                rows[j][i] = value
        matrix = tuple(tuple(row) for row in rows)
        if len(self._matrix_cache) > 4096:
            self._matrix_cache.clear()
        self._matrix_cache[cache_key] = matrix
        return matrix

    @staticmethod
    def _coefficient(
        T: float,
        Tc: float,
        Pc: float,
        omega: float,
        a: float,
        b: float,
        order: int,
    ) -> float:
        return BVirial_Tsonopoulos_extended(
            T,
            Tc,
            Pc,
            omega,
            a=a,
            b=b,
            order=order,
        )


class PitzerCurlSecondVirialProvider(TsonopoulosSecondVirialProvider):
    """Pitzer-Curl pure/cross correlation with Tsonopoulos combining rules."""

    name = 'PITZER-CURL'
    uses_polar_corrections = False
    uses_builtin_binary_kijs = False

    @staticmethod
    def _coefficient(
        T: float,
        Tc: float,
        Pc: float,
        omega: float,
        _a: float,
        _b: float,
        order: int,
    ) -> float:
        return BVirial_Pitzer_Curl(T, Tc, Pc, omega, order=order)


class AbbottSecondVirialProvider(TsonopoulosSecondVirialProvider):
    """Abbott/Lee-Kesler pure/cross fit with Tsonopoulos combining rules."""

    name = 'ABBOTT'
    uses_polar_corrections = False
    uses_builtin_binary_kijs = False

    @staticmethod
    def _coefficient(
        T: float,
        Tc: float,
        Pc: float,
        omega: float,
        _a: float,
        _b: float,
        order: int,
    ) -> float:
        return BVirial_Abbott(T, Tc, Pc, omega, order=order)


def create_second_virial_provider(
    correlation: object,
    components: Sequence[str],
    component_properties: Mapping[str, object],
    *,
    resolver_properties: Mapping[str, dict] | None = None,
    allow_online: bool = True,
    chemical_database=None,
    interaction_overrides: Sequence[Mapping[str, object]] | None = None,
) -> SecondVirialCoefficientProvider:
    """Construct the selected coefficient provider."""
    normalized = normalize_second_virial_correlation(correlation)

    def pair_overrides(model: str, field: str) -> dict[tuple[str, str], float]:
        values = {}
        for record in interaction_overrides or ():
            record_model = str(record.get('model', '')).strip().upper().replace(
                '_', '-'
            )
            if record_model != model or field not in record:
                continue
            pair = (
                str(record.get('component1', '')).strip(),
                str(record.get('component2', '')).strip(),
            )
            if not all(pair):
                raise ThermodynamicsError(
                    f"{model} interaction override requires two component names"
                )
            key = frozenset(pair)
            if key in {frozenset(existing) for existing in values}:
                raise ThermodynamicsError(
                    f"Duplicate {model} interaction override for {pair[0]}/{pair[1]}"
                )
            values[pair] = float(record[field])
        return values

    if normalized == 'TSONOPOULOS':
        return TsonopoulosSecondVirialProvider(
            components,
            component_properties,
            binary_kijs=pair_overrides('TSONOPOULOS', 'kij'),
            resolver_properties=resolver_properties,
            allow_online=allow_online,
            chemical_database=chemical_database,
        )
    if normalized == 'PITZER-CURL':
        return PitzerCurlSecondVirialProvider(
            components,
            component_properties,
            resolver_properties=resolver_properties,
            allow_online=allow_online,
            chemical_database=chemical_database,
        )
    if normalized == 'ABBOTT':
        return AbbottSecondVirialProvider(
            components,
            component_properties,
            resolver_properties=resolver_properties,
            allow_online=allow_online,
            chemical_database=chemical_database,
        )
    if normalized == 'HOC':
        from .hayden_oconnell import HaydenOConnellSecondVirialProvider

        return HaydenOConnellSecondVirialProvider(
            components,
            component_properties,
            binary_etas=pair_overrides('HOC', 'eta'),
            resolver_properties=resolver_properties,
            allow_online=allow_online,
            chemical_database=chemical_database,
        )
    raise ThermodynamicsError(
        f"Unsupported second-virial correlation: {normalized}"
    )


class SecondVirialVaporBackend:
    """Pressure-truncated second-virial EOS for vapor gamma-phi corrections."""

    def __init__(
        self,
        components: Sequence[str],
        provider: SecondVirialCoefficientProvider,
    ) -> None:
        self.components = tuple(components)
        self.provider = provider
        provider_components = getattr(provider, 'components', None)
        if provider_components is not None and tuple(provider_components) != self.components:
            raise ThermodynamicsError(
                "Second-virial provider component order must match the vapor backend"
            )
        self.correlation = str(getattr(provider, 'name', type(provider).__name__))
        self.warnings: list[str] = list(getattr(provider, 'warnings', ()))
        unresolved = tuple(getattr(provider, 'unresolved_binary_pairs', ()))
        if unresolved and getattr(provider, 'uses_builtin_binary_kijs', False):
            pairs = ', '.join(f'{first}/{second}' for first, second in unresolved)
            self.warnings.append(
                "Second cross-virial coefficients use k_ij=0 where no published "
                f"binary or family rule was resolved: {pairs}."
            )

    def _normalized_composition(
        self,
        composition: Mapping[str, float],
    ) -> tuple[float, ...]:
        values = [max(float(composition.get(component, 0.0)), 0.0) for component in self.components]
        total = sum(values)
        if not math.isfinite(total) or total <= 0.0:
            raise ThermodynamicsError(
                "Second-virial vapor composition must have a positive finite total"
            )
        return tuple(value / total for value in values)

    def _validated_matrix(
        self,
        T: float,
        order: int,
    ) -> tuple[tuple[float, ...], ...]:
        raw = self._coefficient_matrix(float(T), order)
        count = len(self.components)
        if len(raw) != count or any(len(row) != count for row in raw):
            raise ThermodynamicsError(
                "Second-virial provider returned a matrix with the wrong dimensions"
            )
        matrix = tuple(tuple(float(value) for value in row) for row in raw)
        for i in range(count):
            for j in range(count):
                value = matrix[i][j]
                if not math.isfinite(value):
                    raise ThermodynamicsError(
                        "Second-virial provider returned a non-finite coefficient"
                    )
                tolerance = 1.0e-12 * max(1.0, abs(value), abs(matrix[j][i]))
                if abs(value - matrix[j][i]) > tolerance:
                    raise ThermodynamicsError(
                        "Second-virial provider returned a nonsymmetric B_ij matrix"
                    )
        return matrix

    def _coefficient_matrix(
        self,
        T: float,
        order: int,
    ) -> Sequence[Sequence[float]]:
        return self.provider.second_virial_matrix(T, order=order)

    @staticmethod
    def _mixture_value(
        fractions: Sequence[float],
        matrix: Sequence[Sequence[float]],
    ) -> float:
        return sum(
            fractions[i] * fractions[j] * matrix[i][j]
            for i in range(len(fractions))
            for j in range(len(fractions))
        )

    def mixture_second_virial(
        self,
        T: float,
        composition: Mapping[str, float],
        order: int = 0,
    ) -> float:
        fractions = self._normalized_composition(composition)
        matrix = self._validated_matrix(T, order)
        return self._mixture_value(fractions, matrix)

    @staticmethod
    def _validate_state(T: float, P: float, phase: str) -> tuple[float, float]:
        temperature = float(T)
        pressure = float(P)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ThermodynamicsError(
                "Second-virial vapor temperature must be positive and finite"
            )
        if not math.isfinite(pressure) or pressure < 0.0:
            raise ThermodynamicsError(
                "Second-virial vapor pressure must be nonnegative and finite"
            )
        if str(phase).strip().lower() not in {'vapor', 'gas'}:
            raise ThermodynamicsError(
                "The truncated second-virial backend supports only the vapor phase"
            )
        return temperature, pressure * 1.0e5

    def fugacity_coefficients(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> dict[str, float]:
        temperature, pressure_pa = self._validate_state(T, P, phase)
        fractions = self._normalized_composition(composition)
        matrix = self._validated_matrix(temperature, 0)
        mixture_B = self._mixture_value(fractions, matrix)
        factor = pressure_pa / (R * temperature)
        result = {}
        for i, component in enumerate(self.components):
            partial_B = 2.0 * sum(
                fractions[j] * matrix[i][j]
                for j in range(len(self.components))
            ) - mixture_B
            result[component] = math.exp(max(-700.0, min(700.0, factor * partial_B)))
        return result

    def compressibility_factor(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> float:
        """Return ``Z = 1 + B_mix*P/(R*T)`` for the truncated vapor EOS."""
        temperature, pressure_pa = self._validate_state(T, P, phase)
        mixture_B = self.mixture_second_virial(temperature, composition)
        value = 1.0 + mixture_B * pressure_pa / (R * temperature)
        if not math.isfinite(value) or value <= 0.0:
            raise ThermodynamicsError(
                "The truncated second-virial EOS produced a nonpositive "
                "compressibility factor; the vapor state is outside its useful range"
            )
        return value

    def molar_volume(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> float:
        """Return vapor molar volume [cm^3/mol], matching cubic backends."""
        temperature, pressure_pa = self._validate_state(T, P, phase)
        if pressure_pa <= 0.0:
            raise ThermodynamicsError(
                "Pressure must be positive for second-virial vapor volume"
            )
        compressibility = self.compressibility_factor(T, P, composition, phase)
        return compressibility * R * temperature / pressure_pa * 1.0e6

    def departure_gibbs(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> float:
        temperature, pressure_pa = self._validate_state(T, P, phase)
        return pressure_pa * self.mixture_second_virial(temperature, composition)

    def departure_enthalpy(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> float:
        temperature, pressure_pa = self._validate_state(T, P, phase)
        mixture_B = self.mixture_second_virial(temperature, composition, order=0)
        derivative = self.mixture_second_virial(temperature, composition, order=1)
        # J/mol is numerically equal to kJ/kmol.
        return pressure_pa * (mixture_B - temperature * derivative)

    def departure_entropy(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> float:
        temperature, pressure_pa = self._validate_state(T, P, phase)
        derivative = self.mixture_second_virial(temperature, composition, order=1)
        # J/mol/K is numerically equal to kJ/kmol/K.
        return -pressure_pa * derivative

    def departure_heat_capacity(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> float:
        temperature, pressure_pa = self._validate_state(T, P, phase)
        second_derivative = self.mixture_second_virial(
            temperature,
            composition,
            order=2,
        )
        return -pressure_pa * temperature * second_derivative

    def residual_enthalpy(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> float:
        """Alias for the vapor departure enthalpy [kJ/kmol]."""
        return self.departure_enthalpy(T, P, composition, phase)


class ChemicalAssociationSecondVirialVaporBackend(SecondVirialVaporBackend):
    """Second-virial vapor with explicit pair association.

    This backend is selected only when a provider exposes HOC-style
    ``physical_second_virial_matrix`` and ``association_constant_matrix``
    capabilities. Composition remains on the nominal monomer-equivalent
    basis used everywhere else in the simulator; internal dimers never
    become process-stream components.
    """

    def __init__(
        self,
        components: Sequence[str],
        provider: ChemicalAssociationSecondVirialProvider,
    ) -> None:
        super().__init__(components, provider)
        associating = tuple(getattr(provider, 'associating_components', ()))
        unknown = set(associating) - set(self.components)
        if not associating or unknown:
            raise ThermodynamicsError(
                "Chemical-association virial provider supplied an invalid "
                "associating-component list"
            )
        if not callable(getattr(provider, 'physical_second_virial_matrix', None)):
            raise ThermodynamicsError(
                "Chemical-association virial provider has no physical B matrix"
            )
        if not callable(getattr(provider, 'association_constant_matrix', None)):
            raise ThermodynamicsError(
                "Chemical-association virial provider has no K_p matrix"
            )
        self.associating_components = associating
        self._component_indices = {
            component: index for index, component in enumerate(self.components)
        }
        self._association_state_cache: dict[tuple, dict] = {}
        self.warnings.append(
            "HOC carboxylic-acid vapor association uses equation 31 chemical "
            "theory with B_free for physical nonideality."
        )

    def _coefficient_matrix(
        self,
        T: float,
        order: int,
    ) -> Sequence[Sequence[float]]:
        return self.provider.physical_second_virial_matrix(T, order=order)

    @staticmethod
    def _solve_association_fallback(
        nominal: Sequence[float],
        inert_total: float,
        pair_i: Sequence[int],
        pair_j: Sequence[int],
        pair_kappa: Sequence[float],
    ) -> tuple[list[float], list[float], float]:
        """Solve relative material balances in log monomer fractions.

        The no-association solution is log(monomer/nominal)=0, an interior
        point. Log balances retain relative accuracy for trace acids and
        strongly associated vapors without a concentration-dependent scale.
        """
        import numpy as np
        from scipy.optimize import least_squares

        nominal_values = np.asarray(nominal, dtype=float)
        pair_i_values = np.asarray(pair_i, dtype=int)
        pair_j_values = np.asarray(pair_j, dtype=int)
        kappa_values = np.maximum(np.asarray(pair_kappa, dtype=float), 0.0)
        def evaluate(log_fractions):
            monomers = nominal_values * np.exp(log_fractions)
            terms = (
                kappa_values
                * monomers[pair_i_values]
                * monomers[pair_j_values]
            )
            base_total = float(inert_total + np.sum(monomers))
            total = 0.5 * (
                base_total
                + math.sqrt(max(base_total**2 + 4.0 * float(np.sum(terms)), 0.0))
            )
            total = max(total, 1.0e-300)
            extents = terms / total
            binding = np.zeros(len(monomers))
            for pair, kappa in enumerate(kappa_values):
                i = pair_i_values[pair]
                j = pair_j_values[pair]
                if i == j:
                    binding[i] += 2.0 * kappa * monomers[i] / total
                else:
                    binding[i] += kappa * monomers[j] / total
                    binding[j] += kappa * monomers[i] / total
            residual = log_fractions + np.log1p(binding)
            return residual, monomers, extents, total

        initial = np.full(len(nominal_values), -math.log(2.0))
        solved = least_squares(
            lambda values: evaluate(values)[0],
            initial,
            # Permit trial fractions above one so the weak-association root
            # stays away from a bound, while preventing exponential overflow.
            bounds=(-np.inf, math.log(2.0)),
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=300,
        )
        residual, monomers, extents, total = evaluate(solved.x)
        if (
            not np.all(np.isfinite(residual))
            or not np.all(np.isfinite(monomers))
            or float(np.linalg.norm(residual, ord=np.inf)) > 1.0e-8
        ):
            raise ThermodynamicsError(
                "HOC chemical-association material-balance solve did not converge"
            )
        return monomers.tolist(), extents.tolist(), float(total)

    def _association_state(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
    ) -> dict:
        temperature, _pressure_pa = self._validate_state(T, P, 'vapor')
        fractions = self._normalized_composition(composition)
        cache_key = (
            temperature,
            float(P),
            tuple(fractions),
        )
        cached = self._association_state_cache.get(cache_key)
        if cached is not None:
            return cached

        active_components = [
            component
            for component in self.associating_components
            if fractions[self._component_indices[component]] > 0.0
        ]
        if not active_components:
            state = {
                'physical_moles_per_nominal': 1.0,
                'chemical_phi': {
                    component: 1.0 for component in self.components
                },
                'extents': {},
                'association_enthalpy': 0.0,
            }
            self._association_state_cache[cache_key] = state
            return state

        nominal = [
            fractions[self._component_indices[component]]
            for component in active_components
        ]
        inert_total = max(1.0 - sum(nominal), 0.0)
        constants = self.provider.association_constant_matrix(temperature)
        derivatives = self.provider.association_constant_matrix(
            temperature,
            order=1,
        )
        pair_i = []
        pair_j = []
        pair_kappa = []
        pair_keys = []
        pair_enthalpies = []
        for i, first in enumerate(active_components):
            first_index = self._component_indices[first]
            for j in range(i, len(active_components)):
                second = active_components[j]
                second_index = self._component_indices[second]
                equilibrium_constant = float(constants[first_index][second_index])
                if not math.isfinite(equilibrium_constant) or equilibrium_constant < 0.0:
                    raise ThermodynamicsError(
                        f"HOC returned an invalid association constant for "
                        f"{first}/{second}"
                    )
                derivative = float(derivatives[first_index][second_index])
                pair_i.append(i)
                pair_j.append(j)
                pair_kappa.append(equilibrium_constant * float(P))
                pair_keys.append((first, second))
                pair_enthalpies.append(
                    R * temperature**2 * derivative / equilibrium_constant
                    if equilibrium_constant > 0.0 else 0.0
                )

        solved = None
        try:
            if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
                from ..compiled_vdm import solve_n_acid_true_moles
            else:
                from compiled_vdm import solve_n_acid_true_moles
            solved = solve_n_acid_true_moles(
                nominal,
                inert_total,
                pair_i,
                pair_j,
                pair_kappa,
            )
        except Exception:
            solved = None
        if solved is not None:
            # The shared compiled solver has an absolute residual floor for
            # tiny inventories. Require relative material balance here too.
            monomers, extent_values, _total = solved
            recovered = list(monomers)
            for i, j, extent in zip(pair_i, pair_j, extent_values, strict=True):
                recovered[i] += extent
                recovered[j] += extent
            if any(
                not math.isfinite(value) or abs(value / amount - 1.0) > 1.0e-8
                for value, amount in zip(recovered, nominal, strict=True)
            ):
                solved = None
        if solved is None:
            solved = self._solve_association_fallback(
                nominal,
                inert_total,
                pair_i,
                pair_j,
                pair_kappa,
            )
        monomers, extent_values, physical_total = solved
        if (
            not math.isfinite(physical_total)
            or physical_total <= 0.0
            or physical_total > 1.0 + 1.0e-10
        ):
            raise ThermodynamicsError(
                "HOC chemical theory returned a nonphysical molecule count"
            )

        monomer_by_component = dict(zip(active_components, monomers))
        chemical_phi = {}
        associating_set = set(self.associating_components)
        for component in self.components:
            denominator = physical_total
            if component in associating_set:
                i = self._component_indices[component]
                for other, monomer in monomer_by_component.items():
                    j = self._component_indices[other]
                    constant = float(constants[i][j])
                    if not math.isfinite(constant) or constant < 0.0:
                        raise ThermodynamicsError(
                            f"HOC returned an invalid association constant for "
                            f"{component}/{other}"
                        )
                    denominator += (
                        (2.0 if component == other else 1.0)
                        * constant * float(P) * monomer
                    )
            # n_i = m_i * (1 + P*sum_j(s_ij*K_ij*m_j)/N).
            # Thus phi_i = m_i/(N*n_i) has this finite expression even
            # when n_i=0; absent acids still associate with present acids.
            value = 1.0 / denominator
            chemical_phi[component] = max(float(value), 1.0e-300)

        extents = dict(zip(pair_keys, extent_values))
        association_enthalpy = sum(
            float(extents[pair]) * pair_enthalpies[index]
            for index, pair in enumerate(pair_keys)
        )
        state = {
            'physical_moles_per_nominal': float(physical_total),
            'chemical_phi': chemical_phi,
            'extents': extents,
            'association_enthalpy': float(association_enthalpy),
        }
        if len(self._association_state_cache) > 20000:
            self._association_state_cache.clear()
        self._association_state_cache[cache_key] = state
        return state

    def association_state(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
    ) -> dict:
        """Return a copy of the internal HOC chemical-equilibrium state."""
        state = self._association_state(T, P, composition)
        return {
            **state,
            'chemical_phi': dict(state['chemical_phi']),
            'extents': dict(state['extents']),
        }

    def fugacity_coefficients(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> dict[str, float]:
        physical = super().fugacity_coefficients(T, P, composition, phase)
        chemical = self._association_state(T, P, composition)['chemical_phi']
        return {
            component: max(
                float(physical[component]) * float(chemical[component]),
                1.0e-300,
            )
            for component in self.components
        }

    def compressibility_factor(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> float:
        temperature, pressure_pa = self._validate_state(T, P, phase)
        physical_moles = self._association_state(
            temperature,
            P,
            composition,
        )['physical_moles_per_nominal']
        physical_B = self.mixture_second_virial(temperature, composition)
        value = physical_moles + physical_B * pressure_pa / (R * temperature)
        if not math.isfinite(value) or value <= 0.0:
            raise ThermodynamicsError(
                "HOC chemical theory produced a nonpositive compressibility factor"
            )
        return value

    def departure_gibbs(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> float:
        temperature, _pressure_pa = self._validate_state(T, P, phase)
        fractions = self._normalized_composition(composition)
        phi = self.fugacity_coefficients(T, P, composition, phase)
        return R * temperature * sum(
            fractions[index] * math.log(max(phi[component], 1.0e-300))
            for index, component in enumerate(self.components)
            if fractions[index] > 0.0
        )

    def departure_enthalpy(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> float:
        physical = super().departure_enthalpy(T, P, composition, phase)
        association = self._association_state(
            T,
            P,
            composition,
        )['association_enthalpy']
        return physical + association

    def departure_entropy(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> float:
        temperature, _pressure_pa = self._validate_state(T, P, phase)
        enthalpy = self.departure_enthalpy(T, P, composition, phase)
        gibbs = self.departure_gibbs(T, P, composition, phase)
        return (enthalpy - gibbs) / temperature

    def departure_heat_capacity(
        self,
        T: float,
        P: float,
        composition: Mapping[str, float],
        phase: str = 'vapor',
    ) -> float:
        temperature, _pressure_pa = self._validate_state(T, P, phase)
        step = max(0.05, 1.0e-4 * temperature)
        lower = max(1.0, temperature - step)
        upper = temperature + step
        return (
            self.departure_enthalpy(upper, P, composition, phase)
            - self.departure_enthalpy(lower, P, composition, phase)
        ) / (upper - lower)


def create_second_virial_vapor_backend(
    components: Sequence[str],
    provider: SecondVirialCoefficientProvider,
) -> SecondVirialVaporBackend:
    """Construct the ordinary or optional association-aware BV backend."""
    if bool(getattr(provider, 'has_chemical_theory', False)):
        return ChemicalAssociationSecondVirialVaporBackend(components, provider)
    return SecondVirialVaporBackend(components, provider)
